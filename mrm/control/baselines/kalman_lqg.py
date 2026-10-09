"""Kalman filter + LQG heater lock with delay compensation (ARCHITECTURE section 6 item 8; 04 C15; 07 P1-1).

The "model-based, honest nominal knowledge only" baseline. Its model is ``mrm.control.models.linearize_bank`` of
``spec.nominal`` (the nominal design, uncertainty nominal, variation off; the knowledge token ``model_nominal`` must be
in ``controller_interface.knowledge``). It never sees Truth or the sampled plant parameters.

Per ring (vectorized over B and N), every controller period Ts:

1. Measurement y[k]: the mean of the monitor samples released since the last step (``tx.monitor.samples``, valid
   count ``tx.monitor.count``), or the newest one (``measurement = latest``). No update when the count is 0,
   ``monitor_valid`` is False or ``adc_overrange`` is set (a clipped code carries no slope information).
   With a measured peak (acquisition, below) the code is normalized to the nominal scale:
   y_n = y_zero + (y - y_zero) * (y_pk,nominal - y_zero) / (y_pk,measured - y_zero).
2. Kalman update (steady-state gain L of the discrete Riccati equation) on the augmented model
       z = [x (thermal lag states); u[k-1..k-q-1] (codes in the delay line); d (pm); r (pm/ms)]
       z[k+1] = Phi z[k] + Gamma u[k],   y[k] = C_y z[k]
   x, u-line: ``LinearModel.delayed()`` (the whole loop delay tau = tau_u + tau_m on the input, exact fractional
   discretization, so the filter and the regulator see the delay: this is the delay compensation, a Kalman predictor
   with the commands in flight as known states); d: additive ring-temperature disturbance expressed as the
   open-loop resonance shift it causes (pm; random walk, process noise q_d); r: its rate (random walk, q_rate). The (d, r) pair makes the loop type 2 (zero error on a ramp, 04 4.0
   constraint 3 and C15).
3. Control by output regulation (Francis regulator equations) plus LQR:
       Pi W = Phi_c Pi + Gamma_c Gam + E,  0 = C_delta Pi + F       (W = [[1, Ts], [0, 1]], the (d, r) exosystem)
       u[k] = u0 + Gam w - K (z_c - Pi w),  K = dlqr(Phi_c, Gamma_c, C_delta' C_delta / q_pm^2, 1 / r_codes^2)
   so the nominal model holds delta at its target with zero steady error to steps and ramps of d.
4. Saturation: the command is rounded and clipped to the heater code range; the filter is propagated with the
   APPLIED (clipped) command, so the estimate never winds up.

Acquisition (optional, ``start``): ``warm`` assumes the ring already sits on the stable side near the target
(the first valid sample initializes the filter at the static state that explains it); with ``probe_peak`` the
controller first measures the drop peak by sweeping the heater down along the self-locked branch until the fold
(07 P0-1: the stable branch reaches past the peak; the fold drops the monitor), then heats back up through the jump
and hands over at f* of the measured peak. ``cold`` ramps the heater from the applied code to the nominal lock code
plus a margin first (07 P1-12 warns: in g26 the ramp crosses three foreign lines, which this baseline does not
arbitrate; the first fold below the ramp end is taken as the ring's own line). The target fraction f* is the
nominal model's drop power at its target over its peak, so the lock is the same relative point of the measured
resonance whatever the monitor gain and laser power are (07 4.8: gain 0.5-2x, chip loss 2-10 dB).

Limits (documented failure modes, tests/verification/test_ctrl_c_kalman_lqg.py and docs/handoff/controllers_c.md):
the lock is a monitor SET-POINT lock on the nominal code y0 (f* = 0.77 of the nominal drop peak in g26), so anything
that scales the drop code moves the lock point (laser power, monitor gain, PD dark current) and a drop of more than
-10 log10(f*) = 1.1 dB puts y0 above the reachable peak: the loop then cools the ring through the peak and it folds.
In LOCK there is no loss-of-lock detection and no re-acquisition (the acquisition FSM runs once); the ``locked``
telemetry (|estimated error| < lock_pm) drops to 0 when that happens, so the controller does not claim a lost lock.
Only ``controller_interface.schedule = simultaneous`` is supported (round_robin is refused at reset).

Tuning (all from spec.nominal and the defaults below, never from Truth): q_pm (detuning error scale), r_codes
(heater move scale), q_d_pm / q_rate_pm_per_ms (disturbance random walks per step), measurement noise from the
nominal AFE (shot + TIA + quantization, ``StaticPoint.y_sigma_codes``) averaged over the samples of a step.
Telemetry (07 P1-20): state (0 init, 1 ramp, 2 probe, 3 relock, 4 lock, 5 fail), locked, detuning_est_pm,
integrator (d estimate, pm), integrator_saturated (command at a code limit), plus innovation_codes.

A compiled numba version with the flat ABI (07 P0-9) is :func:`compiled_kalman_lqg` (registered as
``kalman_lqg_c``); it runs the same arithmetic in float64 and is checked against this class step by step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numba as nb
import numpy as np
import scipy.linalg as sla

from mrm.control import models as M
from mrm.control.api import (
    SPEC_I_HEATER_MAX, SPEC_I_HEATER_MIN, SPEC_I_M, SPEC_I_N_RINGS, BaseController, CompiledController, ControllerSpec,
    Observation, act_offset, obs_offset, user_f64, user_i64,
)
from mrm.control.registry import register
from mrm.core.channels import ActId, ObsId

__all__ = ["KalmanLQG", "LqgDesign", "design_lqg", "closed_loop_spectral_radius", "compiled_kalman_lqg",
           "ST_INIT", "ST_RAMP", "ST_PROBE", "ST_RELOCK", "ST_LOCK", "ST_FAIL", "Acquisition"]

ST_INIT, ST_RAMP, ST_PROBE, ST_RELOCK, ST_LOCK, ST_FAIL = 0, 1, 2, 3, 4, 5
_PM = 1e12          # m -> pm


# ------------------------------------------------------------------------------------------------ design
@dataclass(frozen=True)
class LqgDesign:
    """Per-ring matrices of the augmented model and its gains (arrays stacked over rings, axis 0 = ring).

    nz = nc + nw states: nc = thermal + delay line (codes), nw = 1 (d) or 2 (d, r). Phi (N, nz, nz), Gz (N, nz),
    Cy (N, nz) (codes per state), Cd (N, nz) (pm per state), Lk (N, nz) Kalman gain, Kf (N, nz) full feedback
    (u_dev = -Kf z), zu (N, nz) and zd (N, nz) steady states per unit u_dev / unit d (pm), u0, y0, y_zero, y_pk,
    det0_pm (N,) operating point, frac (N,) y0 over the nominal peak (net of y_zero), sigma_y (N,) codes.
    regulated (N,) bool: False where the regulator/LQR has no solution (no heater authority at the operating point);
    such a ring gets Kf = 0. models: the LinearModels (Python side only).
    """
    nc: int
    nw: int
    Phi: np.ndarray
    Gz: np.ndarray
    Cy: np.ndarray
    Cd: np.ndarray
    Lk: np.ndarray
    Kf: np.ndarray
    zu: np.ndarray
    zd: np.ndarray
    u0: np.ndarray
    y0: np.ndarray
    y_zero: np.ndarray
    y_pk: np.ndarray
    det0_pm: np.ndarray
    sigma_y: np.ndarray
    Ts: float
    regulated: np.ndarray | None = None
    models: list = field(default_factory=list, repr=False)

    @property
    def nz(self) -> int:
        return self.Phi.shape[1]

    @property
    def frac(self) -> np.ndarray:
        return (self.y0 - self.y_zero) / (self.y_pk - self.y_zero)


def _dlqr(A, B, Q, R):
    P = sla.solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def _augment(lm: M.LinearModel, ramp: bool, n_delay: int | None = None):
    """Delay-augmented model with the disturbance states (d in pm, r in pm/ms); the delay line has at least
    ``n_delay`` entries (a common state size for rings with different delays, e.g. an ADC mux)."""
    Pz, Gu, Gd, C, Dd, nd = lm.delayed(n_delay=n_delay)
    nc = Pz.shape[0]
    nw = 2 if ramp else 1
    nz = nc + nw
    Ts_ms = lm.Ts * 1e3
    s_pm = float(lm.Dc[1, 1]) * _PM            # pm of resonance per K of the ring-temperature disturbance
    Gd = Gd / s_pm                              # the d state is in pm of open-loop resonance shift: d_K = d_pm / s
    Dd = Dd / s_pm
    Phi = np.zeros((nz, nz))
    Phi[:nc, :nc] = Pz
    Phi[:nc, nc] = Gd
    Phi[nc, nc] = 1.0
    if ramp:
        Phi[nc, nc + 1] = Ts_ms                # d[k+1] = d[k] + Ts r[k], r in pm/ms
        Phi[nc + 1, nc + 1] = 1.0
    Gz = np.zeros(nz)
    Gz[:nc] = Gu
    Cy = np.zeros(nz)
    Cy[:nc] = C[0]
    Cy[nc] = Dd[0]
    Cd = np.zeros(nz)
    Cd[:nc] = C[1] * _PM
    Cd[nc] = Dd[1] * _PM                        # = 1 (pm per pm)
    return Phi, Gz, Cy, Cd, nc, nw


def _regulator(Phi, Gz, Cd, nc, nw):
    """Francis regulator equations for the exosystem W of (d, r): returns Pi (nc, nw), Gam (nw,)."""
    W = Phi[nc:, nc:]
    Pc = Phi[:nc, :nc]
    E = Phi[:nc, nc:]
    F = Cd[nc:]
    I = np.eye(nc)
    top = np.concatenate([np.kron(W.T, I) - np.kron(np.eye(nw), Pc), -np.kron(np.eye(nw), Gz[:nc, None])], axis=1)
    bot = np.concatenate([np.kron(np.eye(nw), Cd[None, :nc]), np.zeros((nw, nw))], axis=1)
    Mx = np.concatenate([top, bot], axis=0)
    rhs = np.concatenate([E.T.reshape(-1), -F])
    # raises LinAlgError when the equations have no solution: no heater authority at the operating point (a model
    # linearized beyond the code range or in the driver's compliance, dP/dcode = 0); design_lqg then marks the ring
    # unregulated
    sol = np.linalg.solve(Mx, rhs)
    if not (np.all(np.isfinite(sol)) and np.allclose(Mx @ sol, rhs, rtol=1e-6,
                                                     atol=1e-9 * max(1.0, float(np.max(np.abs(rhs)))))):
        raise np.linalg.LinAlgError("regulator equations are singular (no heater authority)")
    Pi = sol[:nc * nw].reshape(nw, nc).T
    Gam = sol[nc * nw:]
    return Pi, Gam


def design_lqg(models: list, *, q_pm: float = 1.0, r_codes: float = 40.0, q_d_pm: float = 0.3,
               q_rate_pm_per_ms: float = 2.0, q_x_pm: float = 0.01, sigma_extra_codes: float = 2.0,
               samples_per_step: float = 2.5, ramp: bool = True, n_delay: int | None = None,
               kalman: bool = True) -> LqgDesign:
    """Kalman + LQR gains of every ring's model (module docstring). Units: q_pm pm, r_codes codes, q_d_pm pm per
    step, q_rate_pm_per_ms pm/ms per step, q_x_pm pm per step (thermal-state model error, expressed as the detuning
    it causes; each lag state gets q_x_pm / |its detuning weight|), sigma_extra_codes codes (unmodelled measurement
    error added in quadrature to the nominal AFE noise). n_delay: least delay-line length (default: the longest
    the rings need; every ring gets the same state size so the arrays stack). kalman=False skips the filter (Lk = 0)
    for a full-state user such as the Truth-fed oracle: its Riccati equation needs the disturbance observable through
    the monitor, which fails where dP_drop/dT_R vanishes (a target at the drop peak or beyond the code range)."""
    cols = {k: [] for k in ("Phi", "Gz", "Cy", "Cd", "Lk", "Kf", "zu", "zd", "regulated")}
    nc = nw = None
    nd = max([lm.n_delay() for lm in models] + [0 if n_delay is None else int(n_delay)])
    for lm in models:
        Phi, Gz, Cy, Cd, nc, nw = _augment(lm, ramp, nd)
        nz = nc + nw
        nx = lm.Ac.shape[0]
        # --- Kalman (filter form): prior covariance from the DARE, gain L = P C'(C P C' + R)^-1
        q = np.zeros(nz)
        w = np.abs(Cd[:nx])
        w = np.where(w > 0, w, np.max(w) if np.any(w > 0) else 1.0)
        q[:nx] = (q_x_pm / w) ** 2
        q[nc] = q_d_pm ** 2
        if ramp:
            q[nc + 1] = q_rate_pm_per_ms ** 2
        sig = float(lm.op.y_sigma_codes[lm.ring])
        R = np.array([[sig ** 2 / max(samples_per_step, 1.0) + sigma_extra_codes ** 2]])
        if kalman:
            P = sla.solve_discrete_are(Phi.T, Cy[:, None], np.diag(q), R)
            S = Cy @ P @ Cy + R[0, 0]
            Lk = P @ Cy / S
        else:
            Lk = np.zeros(nz)
        # --- regulator equations + LQR on the controllable part
        Pc, Gc, Cdc = Phi[:nc, :nc], Gz[:nc, None], Cd[:nc]
        Kf = np.zeros(nz)
        zu = np.zeros(nz)
        zd = np.zeros(nz)
        zd[nc] = 1.0
        try:
            Pi, Gam = _regulator(Phi, Gz, Cd, nc, nw)
            Qc = np.outer(Cdc, Cdc) / q_pm ** 2 + 1e-12 * np.eye(nc)
            K = _dlqr(Pc, Gc, Qc, np.array([[1.0 / r_codes ** 2]]))[0]
            Kf[:nc] = K
            Kf[nc:] = -(Gam + K @ Pi)
            # --- steady states per unit u_dev and per unit d (pm): (I - Phi_c) z = Gamma u + E d
            A_ss = np.eye(nc) - Pc
            zu[:nc] = np.linalg.solve(A_ss, Gz[:nc])
            zd[:nc] = np.linalg.solve(A_ss, Phi[:nc, nc])
            ok = bool(np.all(np.isfinite(Kf)) and np.all(np.isfinite(zu)) and np.all(np.isfinite(zd)))
        except (np.linalg.LinAlgError, ValueError):
            # no heater authority at this operating point (beyond the code range, in the driver's compliance) or
            # an operating point that is not open-loop stable: no regulator for this ring (Kf = 0)
            ok = False
        if not ok:
            Kf[:] = 0.0
            zu[:] = 0.0
            zd[:nc] = 0.0
        cols["regulated"].append(ok)
        for k, v in (("Phi", Phi), ("Gz", Gz), ("Cy", Cy), ("Cd", Cd), ("Lk", Lk), ("Kf", Kf), ("zu", zu),
                     ("zd", zd)):
            cols[k].append(v)
    sp = models[0].op
    n = [lm.ring for lm in models]
    regulated = np.array(cols.pop("regulated"), bool)
    return LqgDesign(nc=nc, nw=nw, **{k: np.ascontiguousarray(np.stack(v)) for k, v in cols.items()},
                     regulated=regulated,
                     u0=np.array([lm.u0 for lm in models]), y0=np.array([lm.y0 for lm in models]),
                     y_zero=np.asarray(sp.y_zero_code, float)[n], y_pk=np.asarray(sp.y_peak_code, float)[n],
                     det0_pm=np.array([lm.detuning0_m * _PM for lm in models]),
                     sigma_y=np.asarray(sp.y_sigma_codes, float)[n], Ts=models[0].Ts, models=list(models))


def closed_loop_spectral_radius(design: LqgDesign, ring: int, plant: M.LinearModel, gain: float = 1.0) -> float:
    """Spectral radius of the LQG (designed on the nominal model) in closed loop with another linear plant (e.g. the
    linearization of a sampled world, or the nominal one scaled by ``gain`` at its input). < 1 = stable.

    Controller: prior zp, posterior z = zp + L (y - C_y zp), u = -K z, zp+ = Phi z + Gamma u. Plant: its own delayed
    model (thermal + delay line) driven by u, y = C_y,plant z_p."""
    Phi, Gz, Cy, Lk, Kf = (design.Phi[ring], design.Gz[ring], design.Cy[ring], design.Lk[ring], design.Kf[ring])
    Pp, Gp, _, Cp, _, _ = plant.delayed()
    Cpy = Cp[0]
    np_, nz = Pp.shape[0], Phi.shape[0]
    I = np.eye(nz)
    Mz = I - np.outer(Lk, Cy)                    # z = Mz zp + L Cpy zt
    Acl = np.zeros((np_ + nz, np_ + nz))
    # u = -Kf (Mz zp + L Cpy zt)
    ku_t = -gain * (Kf @ np.outer(Lk, Cpy))      # u contribution from zt (scaled at the plant input)
    ku_p = -gain * (Kf @ Mz)
    Acl[:np_, :np_] = Pp + np.outer(Gp, ku_t)
    Acl[:np_, np_:] = np.outer(Gp, ku_p)
    G2 = Phi - np.outer(Gz, Kf)
    Acl[np_:, :np_] = G2 @ np.outer(Lk, Cpy)
    Acl[np_:, np_:] = G2 @ Mz
    return float(np.max(np.abs(np.linalg.eigvals(Acl))))


# ------------------------------------------------------------------------------------------------ acquisition
@dataclass(frozen=True)
class Acquisition:
    """Acquisition settings (module docstring). Steps are heater codes per controller step.

    start: warm | cold. probe_peak: measure the drop peak before locking (warm start). ramp_step: cold ramp;
    ramp_margin_codes: the ramp ends at the nominal lock code + this margin (None: 1.5 x the nominal self-heating
    hysteresis + 1000 codes: a ring heated from the blue side is captured only near the end of its lower branch,
    about G P_abs,peak / (open-loop pm per code) above the lock code, some 4000 codes in g26, and the 3 ms package
    tail lags a fast ramp); probe_step / probe_fine_step: sweep-down steps (fine near the top); relock_step: heat-up
    step after the peak; relock_guard_codes: after a fold, resonances below the fold code + this many codes are
    ignored (None: half the nominal hysteresis); fold_frac: the fold is declared when the net monitor code falls below
    fold_frac x its running max; min_peak_frac: a max below this fraction of the nominal net peak is not a resonance;
    jump_frac: the relock sees the resonance when the net code exceeds jump_frac x the measured net peak (after the
    fold the ring sits far on the blue side, about 1 nm in g26, where the code is a few % of the peak; the jump
    lands on the stable side anywhere above about 0.3 of the peak)."""
    start: str = "warm"
    probe_peak: bool = False
    ramp_step: float = 40.0
    ramp_margin_codes: float | None = None
    probe_step: float = 10.0
    probe_fine_step: float = 2.0
    relock_step: float = 40.0
    relock_guard_codes: float | None = None
    fold_frac: float = 0.6
    min_peak_frac: float = 0.15
    jump_frac: float = 0.25


def _acq_step(state, u, y, ok, mem: dict, frac, y_zero, y_pk_nom, sigma_y, gyu, a: "Acquisition", code_min,
              code_max):
    """One vectorized acquisition step over (B, N) (module docstring). ``mem`` holds the per-ring memory arrays
    ymax, ypk, seen, y_prev, u_fold, u_end (updated in place). Returns (state, u_cmd, handoff (bool)).

    PROBE (cool along the self-locked branch): fast steps while the code still rises by more than 30 % of the nominal
    operating-point slope |dy/du| (``gyu``, codes per code) per code moved, fine steps near the top. The peak is passed when the code falls by more than
    max(5 sigma_y, 2e-4 peak) below its running max but stays above fold_frac x max: the ring is just past the peak
    and still on the stable branch (07 P0-1: the branch reaches past the peak, by h^2/(2 G P_pk), about 10 pm in g26),
    so it returns to the stable side by heating, without a fold. A larger drop is a fold (the ring fell to the blue
    branch, about 1.2 nm in g26): the relock then heats and ignores any resonance below the fold code plus
    ``relock_guard_codes`` (an adjacent order or a foreign line can capture the ring on the way back, 07 P1-12).
    RELOCK (heat): after the resonance is seen (a rise of jump_frac x peak above the running minimum), the first step
    whose code falls while heating means the stable side; hand over at f* of the peak."""
    st = state.copy()
    ucmd = u.copy()
    net = y - y_zero
    pk_nom = y_pk_nom - y_zero
    ymax, ypk, seen, y_prev, u_fold, u_end = (mem[k] for k in ("ymax", "ypk", "seen", "y_prev", "u_fold", "u_end"))
    # ramp up (cold): to the nominal lock code + margin, then probe downwards
    r = state == ST_RAMP
    ucmd = np.where(r, np.minimum(u + a.ramp_step, u_end), ucmd)
    done = r & (ucmd >= u_end)
    st = np.where(done, ST_PROBE, st)
    ymax[...] = np.where(done, 0.0, ymax)
    mem["probe_n"][...] = np.where(done, 0, mem["probe_n"])
    mem["slope_max"][...] = np.where(done, 0.0, mem["slope_max"])
    # probe
    p = (state == ST_PROBE) & ok
    last = np.maximum(mem["last_step"], 1.0)
    have = mem["probe_n"] > 0                       # a slope needs a previous probe sample
    slope = np.where(have, (net - (y_prev - y_zero)) / last, np.inf)
    smax = mem["slope_max"]
    smax[...] = np.where(p & have, np.maximum(smax, slope), smax)
    mem["probe_n"][...] = np.where(p, mem["probe_n"] + 1, mem["probe_n"])
    # fast steps while the code climbs steeply: above 30 % of the nominal slope AND above half the steepest slope
    # seen in this sweep (near the top the slope falls; the fold sits only ~h^2/(2 G P_pk) past the peak, so the
    # last part of the climb is taken in fine steps to let the thermal lag catch up)
    rising = p & (slope > 0.3 * np.abs(gyu)) & (slope > 0.5 * smax)
    ymax[...] = np.where(p, np.maximum(ymax, net), ymax)
    big = p & (ymax > a.min_peak_frac * pk_nom)
    drop = np.maximum(5.0 * sigma_y, 2e-4 * ymax)
    fold = big & (net < a.fold_frac * ymax)
    passed = big & ~fold & (net < ymax - drop)
    ypk[...] = np.where(fold | passed, ymax + y_zero, ypk)
    u_fold[...] = np.where(fold, u, np.where(passed, -np.inf, u_fold))
    st = np.where(fold | passed, ST_RELOCK, st)
    seen[...] = np.where(passed, True, np.where(fold, False, seen))
    step = np.where(rising | (ymax < a.min_peak_frac * pk_nom), a.probe_step, a.probe_fine_step)
    ucmd = np.where(p & ~fold & ~passed, u - step, ucmd)
    mem["last_step"][...] = np.where(p & ~fold & ~passed, step, mem["last_step"])
    ymin = ymax                       # in RELOCK, ymax holds the running minimum since the fold
    ymin[...] = np.where(fold, net, ymin)
    fail = (state == ST_PROBE) & (ucmd <= code_min)
    st = np.where(fail, ST_FAIL, st)
    # relock
    q = (state == ST_RELOCK) & ok
    pk_net = ypk - y_zero
    armed = q & (u >= u_fold + a.relock_guard_codes)
    ymin[...] = np.where(q & ~seen & ~armed, net, np.where(q & ~seen, np.minimum(ymin, net), ymin))
    seen[...] = np.where(armed & (net > ymin + a.jump_frac * pk_net), True, seen)
    # a fold during the return (the thermal lag carried the ring past the fold): the code collapses below
    # fold_frac x peak in one step; treat it as a fold, not as the stable side
    refold = q & seen & (net < a.fold_frac * pk_net) & ((y_prev - y) > 0.3 * pk_net)
    seen[...] = np.where(refold, False, seen)
    u_fold[...] = np.where(refold, u, u_fold)
    ymin[...] = np.where(refold, net, ymin)
    falling = q & seen & (y < y_prev)
    hand = falling & (net <= frac * pk_net)
    st = np.where(hand, ST_LOCK, st)
    ucmd = np.where(q & ~hand, u + a.relock_step, ucmd)
    fail2 = (state == ST_RELOCK) & (ucmd >= code_max)
    st = np.where(fail2 & ~hand, ST_FAIL, st)
    return st, np.clip(ucmd, code_min, code_max), hand


# ------------------------------------------------------------------------------------------------ controller
def _measure(obs: Observation, mode: str):
    """(y (B, N) float codes, ok (B, N) bool, n (B, N) samples) from a monitor observation."""
    cnt = np.asarray(obs.sensors["tx.monitor.count"]).astype(np.int64)
    smp = np.asarray(obs.sensors["tx.monitor.samples"]).astype(float)
    m = smp.shape[-1]
    sel = np.arange(m)[None, None, :] < cnt[..., None]
    if mode == "latest":
        idx = np.clip(cnt - 1, 0, m - 1)
        y = np.take_along_axis(smp, idx[..., None], axis=-1)[..., 0]
    else:
        y = np.where(sel, smp, 0.0).sum(axis=-1) / np.maximum(cnt, 1)
    ok = cnt > 0
    try:
        ok &= np.asarray(obs.sensors["monitor_valid"]).astype(bool)
    except KeyError:
        pass
    try:
        ok &= ~np.asarray(obs.sensors["adc_overrange"]).astype(bool)
    except KeyError:
        pass
    return y, ok, cnt


@register("kalman_lqg")
class KalmanLQG(BaseController):
    """Kalman filter + LQG lock on the drop monitor with delay compensation (module docstring)."""

    name = "kalman_lqg"

    @dataclass
    class Params:
        target: str = "oma_opt"              # oma_opt | detuning | frac_of_peak (nominal model)
        target_detuning_pm: float | None = None
        target_frac: float = 0.75
        latency_ticks: int = 1               # digital pipeline depth (07 P0-6)
        measurement: str = "mean"            # mean | latest
        q_pm: float = 1.0
        r_codes: float = 40.0
        q_d_pm: float = 0.3
        q_rate_pm_per_ms: float = 2.0
        q_x_pm: float = 0.01
        sigma_extra_codes: float = 2.0
        ramp: bool = True
        lock_pm: float = 5.0                 # telemetry: declared lock when |estimated error| < lock_pm
        start: str = "warm"                  # warm | cold
        probe_peak: bool = False
        ramp_step: float = 40.0
        ramp_margin_codes: float | None = None    # None: 1.5 x the nominal hysteresis + 1000 codes
        probe_step: float = 10.0
        probe_fine_step: float = 2.0
        relock_step: float = 40.0
        relock_guard_codes: float | None = None   # None: half the nominal hysteresis
        fold_frac: float = 0.6

    def __init__(self, params=None, **kw):
        super().__init__(params, **kw)
        self.latency_ticks = int(self.params.latency_ticks)

    # -- design (shared with the MPC and the compiled version)
    def _operating_point(self) -> M.OperatingPoint:
        p = self.params
        if p.target == "oma_opt":
            return M.OperatingPoint()
        if p.target == "detuning":
            if p.target_detuning_pm is None:
                raise ValueError("target = detuning needs target_detuning_pm")
            return M.OperatingPoint(detuning_m=p.target_detuning_pm * 1e-12)
        if p.target == "frac_of_peak":
            return M.OperatingPoint(detuning_frac_of_peak=p.target_frac)
        raise ValueError(f"target {p.target!r}: oma_opt | detuning | frac_of_peak")

    def build_design(self, spec: ControllerSpec) -> LqgDesign:
        """Linearize spec.nominal (07 P1-1) and design the Kalman/LQR gains (deterministic in spec)."""
        p = self.params
        M.check_knowledge(spec.knowledge, who=self.name)
        if spec.nominal is None:
            raise ValueError(f"{self.name}: ControllerSpec.nominal is None; a model-based controller needs it")
        if spec.schedule != "simultaneous":
            # round_robin applies one ring's command per edge; the model, the delay line and the gains assume every
            # ring's command acts each period, so the predictor would run on commands that were never applied
            raise ValueError(f"{self.name}: controller_interface.schedule = {spec.schedule!r} is not supported "
                             f"(the design assumes a command for every ring at every edge: simultaneous)")
        models = M.linearize_bank(spec.nominal, self._operating_point(), spec.period_s,
                                  latency_ticks=self.latency_ticks, measurement=p.measurement)
        if len(models) != spec.n_rings:
            raise ValueError(f"nominal model has {len(models)} rings, the spec {spec.n_rings}")
        spp = max(1.0, spec.period_ticks / spec.adc.period_ticks) if p.measurement == "mean" else 1.0
        d = design_lqg(models, q_pm=p.q_pm, r_codes=p.r_codes, q_d_pm=p.q_d_pm,
                       q_rate_pm_per_ms=p.q_rate_pm_per_ms, q_x_pm=p.q_x_pm, sigma_extra_codes=p.sigma_extra_codes,
                       samples_per_step=spp, ramp=p.ramp)
        if not np.all(d.regulated):
            raise ValueError(f"{self.name}: the nominal design has no regulator for rings "
                             f"{np.flatnonzero(~d.regulated).tolist()} (no heater authority at the nominal operating "
                             f"point: code range, driver compliance or an open-loop-unstable target)")
        return d

    def _acquisition(self) -> Acquisition:
        p = self.params
        return Acquisition(start=p.start, probe_peak=p.probe_peak, ramp_step=p.ramp_step,
                           ramp_margin_codes=p.ramp_margin_codes, probe_step=p.probe_step,
                           probe_fine_step=p.probe_fine_step, relock_step=p.relock_step,
                           relock_guard_codes=p.relock_guard_codes, fold_frac=p.fold_frac)

    def reset(self, spec: ControllerSpec) -> None:
        super().reset(spec)
        if self.params.measurement not in ("mean", "latest"):
            raise ValueError("measurement must be mean | latest")
        if self.params.start not in ("warm", "cold"):
            raise ValueError("start must be warm | cold")
        self.design = d = self.build_design(spec)
        self.acq = self._acquisition()
        hyst = self.hysteresis_codes(d)
        if self.acq.ramp_margin_codes is None or self.acq.relock_guard_codes is None:
            from dataclasses import replace as _rep
            self.acq = _rep(self.acq,
                            ramp_margin_codes=(1.5 * float(np.max(hyst)) + 1000.0 if self.acq.ramp_margin_codes is None
                                               else self.acq.ramp_margin_codes),
                            relock_guard_codes=(0.5 * float(np.min(hyst)) if self.acq.relock_guard_codes is None
                                                else self.acq.relock_guard_codes))
        B, N = spec.n_batch, spec.n_rings
        self.z = np.zeros((B, N, d.nz))
        self.u = np.zeros((B, N))                 # applied command (codes, float)
        self.state = np.full((B, N), ST_INIT, np.int64)
        self.mem = {"ymax": np.zeros((B, N)), "ypk": np.tile(d.y_pk, (B, 1)), "seen": np.zeros((B, N), bool),
                    "y_prev": np.zeros((B, N)), "u_fold": np.full((B, N), -np.inf), "last_step": np.ones((B, N)),
                    "slope_max": np.zeros((B, N)), "probe_n": np.zeros((B, N), np.int64),
                    "u_end": np.tile(d.u0 + self.acq.ramp_margin_codes, (B, 1))}
        self.innov = np.zeros((B, N))
        self.code_min = float(spec.heater_dac.code_min)
        self.code_max = float(spec.heater_dac.code_max)
        self._started = False

    @staticmethod
    def hysteresis_codes(d: LqgDesign) -> np.ndarray:
        """Nominal self-heating hysteresis of every ring in heater codes (07 P0-1): the resonance shift G P_abs at the
        peak, s Z_opt P_abs,peak, over the open-loop (self-heating-free) detuning per code."""
        out = []
        for lm in d.models:
            sp, n = lm.op, lm.ring
            p_pk = sp.p_abs_w[n] * sp.p_drop_peak_w[n] / sp.p_drop_w[n]
            shift = sp.s_m_per_k[n] * sp.z_opt_k_per_w[n] * p_pk
            per_code = lm.dc_gain()[1, 0] * lm.selfheat_factor
            out.append(shift / per_code)
        return np.array(out)

    # -- helpers
    def _norm(self, y):
        """Measurement on the nominal scale (identity until a peak has been measured)."""
        d = self.design
        ypk = self.mem["ypk"]
        scale = (d.y_pk - d.y_zero)[None] / np.maximum(ypk - d.y_zero[None], 1.0)
        return np.where(ypk == d.y_pk[None], y, d.y_zero[None] + (y - d.y_zero[None]) * scale)

    def _init_filter(self, mask, y_n):
        """Steady state that explains the measurement at the applied command (warm hand-over)."""
        d = self.design
        udev = self.u - d.u0[None]
        gy_u = np.einsum("nz,nz->n", d.Cy, d.zu)[None]
        gy_d = np.einsum("nz,nz->n", d.Cy, d.zd)[None]
        dd = (y_n - d.y0[None] - gy_u * udev) / gy_d
        z = udev[..., None] * d.zu[None] + dd[..., None] * d.zd[None]
        self.z = np.where(mask[..., None], z, self.z)

    def _control(self, lock: np.ndarray) -> np.ndarray:
        """Heater command (B, N) float codes from the posterior estimate (LQG: u0 - Kf z). ``lock`` marks the rings
        in ST_LOCK (the others are ignored by the caller). Subclasses (the MPC) replace this law."""
        d = self.design
        udev = -np.einsum("nz,bnz->bn", d.Kf, self.z)
        return d.u0[None] + udev

    def _predict(self, mask):
        d = self.design
        udev = self.u - d.u0[None]
        zp = np.einsum("nij,bnj->bni", d.Phi, self.z) + udev[..., None] * d.Gz[None]
        self.z = np.where(mask[..., None], zp, self.z)

    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        spec = self._require_spec()
        d = self.design
        y, ok, _ = _measure(obs, self.params.measurement)
        if not self._started:                    # start from the latched registers (07 P0-6 applied codes)
            try:
                self.u = np.asarray(obs.sensors["tx.heater.applied"], float).copy()
            except KeyError:                         # not exposed: the nominal lock code (as the compiled step)
                self.u = np.rint(np.broadcast_to(d.u0[None], self.u.shape)).astype(float)
            self._started = True
        init = self.state == ST_INIT
        if np.any(init & ok):
            go = init & ok
            if self.acq.start == "cold":
                nxt = ST_RAMP
            else:
                nxt = ST_PROBE if self.acq.probe_peak else ST_LOCK
            if nxt == ST_LOCK:
                self._init_filter(go, self._norm(y))
            self.state = np.where(go, nxt, self.state)
            self.mem["ymax"][...] = np.where(go, 0.0, self.mem["ymax"])
        lock = self.state == ST_LOCK
        # 1-2: Kalman update on the locked rings
        y_n = self._norm(y)
        yhat = np.einsum("nz,bnz->bn", d.Cy, self.z)
        self.innov = np.where(lock & ok, y_n - d.y0[None] - yhat, 0.0)
        upd = lock & ok
        self.z = np.where(upd[..., None], self.z + self.innov[..., None] * d.Lk[None], self.z)
        # 3: command
        u_new = self.u.copy()
        u_lqg = np.clip(np.rint(self._control(lock)), self.code_min, self.code_max)
        u_new = np.where(lock, u_lqg, u_new)
        acq = np.isin(self.state, (ST_RAMP, ST_PROBE, ST_RELOCK))
        if np.any(acq):
            gyu = np.einsum("nz,nz->n", d.Cy, d.zu)[None]
            st, ua, hand = _acq_step(self.state, self.u, y, ok, self.mem, d.frac[None], d.y_zero[None],
                                     d.y_pk[None], d.sigma_y[None], gyu, self.acq, self.code_min, self.code_max)
            u_new = np.where(acq, np.rint(ua), u_new)
            if np.any(hand):
                self._init_filter(hand, self._norm(y))
            self.state = np.where(acq, st, self.state)
        self.mem["y_prev"][...] = np.where(ok, y, self.mem["y_prev"])
        # 4: predict with the applied command
        self.u = u_new
        self._predict(self.state == ST_LOCK)
        return {"tx.heater": u_new.astype(np.int64)}

    def telemetry(self) -> dict[str, np.ndarray]:
        d = self.design
        lock = self.state == ST_LOCK
        det = np.einsum("nz,bnz->bn", d.Cd, self.z)          # pm deviation from the target
        return {"state": self.state.astype(float),
                "locked": (lock & (np.abs(det) < self.params.lock_pm)).astype(float),
                "detuning_est_pm": np.where(lock, d.det0_pm[None] + det, np.nan),
                "integrator": self.z[..., d.nc],
                "integrator_saturated": (lock & ((self.u <= self.code_min) | (self.u >= self.code_max))).astype(float),
                "innovation_codes": self.innov.copy(),
                "peak_codes": self.mem["ypk"].copy()}


# ------------------------------------------------------------------------------------------------ compiled version
# p_f64 layout (per ring blocks of length per = 7 nz + 6, after a header):
#   header: [n_rings, nz, nc, nw, mode_latest, code_min, code_max, lock_pm]
#   ring n: Phi (nz*nz) | Gz (nz) | Cy (nz) | Cd (nz) | Lk (nz) | Kf (nz) | zu (nz) | zd (nz) | u0 y0 y_zero y_pk det0
# s_f64: per ring [z (nz), u, innov], then nz of scratch; s_i64: [started, state per ring]
# p_i64: [n_rings, nz, off_samples, off_count, off_valid, off_over, off_applied, off_heater, m, mode]
_HDR = 8


def _ring_block(nz):
    return nz * nz + 7 * nz + 5


def _pack_design(d: LqgDesign, mode_latest: bool, code_min: float, code_max: float, lock_pm: float) -> np.ndarray:
    N, nz = d.Phi.shape[0], d.nz
    per = _ring_block(nz)
    out = np.zeros(_HDR + N * per)
    out[:_HDR] = [N, nz, d.nc, d.nw, 1.0 if mode_latest else 0.0, code_min, code_max, lock_pm]
    for n in range(N):
        o = _HDR + n * per
        blk = np.concatenate([d.Phi[n].reshape(-1), d.Gz[n], d.Cy[n], d.Cd[n], d.Lk[n], d.Kf[n], d.zu[n], d.zd[n],
                              [d.u0[n], d.y0[n], d.y_zero[n], d.y_pk[n], d.det0_pm[n]]])
        out[o:o + per] = blk
    return out


_SAMPLES = int(ObsId.TX_MONITOR_SAMPLES)
_COUNT = int(ObsId.TX_MONITOR_COUNT)
_VALID = int(ObsId.MONITOR_VALID)
_OVER = int(ObsId.ADC_OVERRANGE)
_APPLIED = int(ObsId.TX_HEATER_APPLIED)
_HEATER = int(ActId.TX_HEATER)


@nb.njit(cache=True)
def lqg_init(spec_i64, spec_f64):
    """Compiled init: unpack the packed design (user float parameters) and the channel offsets."""
    n_user_f = spec_f64.shape[0] - 8
    pf = np.empty(n_user_f)
    for i in range(n_user_f):
        pf[i] = user_f64(spec_f64, i)
    N = spec_i64[SPEC_I_N_RINGS]
    nz = int(pf[1])
    pi = np.zeros(10, np.int64)
    pi[0] = N
    pi[1] = nz
    pi[2] = obs_offset(spec_i64, _SAMPLES)
    pi[3] = obs_offset(spec_i64, _COUNT)
    pi[4] = obs_offset(spec_i64, _VALID)
    pi[5] = obs_offset(spec_i64, _OVER)
    pi[6] = obs_offset(spec_i64, _APPLIED)
    pi[7] = act_offset(spec_i64, _HEATER)
    pi[8] = spec_i64[SPEC_I_M]
    pi[9] = user_i64(spec_i64, 0)
    s_f = np.zeros(N * (nz + 2) + nz)           # per-ring state, then nz of scratch
    s_i = np.zeros(1 + N, np.int64)
    return pi, pf, s_i, s_f


@nb.njit(cache=True)
def lqg_step(p_i, p_f, s_i, s_f, obs, act):
    """Compiled step (warm start, lock only): identical arithmetic to KalmanLQG in ST_LOCK."""
    N = p_i[0]
    nz = p_i[1]
    m = p_i[8]
    nc = int(p_f[2])
    latest = p_f[4] > 0.5
    cmin = p_f[5]
    cmax = p_f[6]
    per = nz * nz + 7 * nz + 5
    first = s_i[0] == 0
    for n in range(N):
        o = _HDR + n * per
        oPhi = o
        oGz = oPhi + nz * nz
        oCy = oGz + nz
        oCd = oCy + nz
        oLk = oCd + nz
        oKf = oLk + nz
        ozu = oKf + nz
        ozd = ozu + nz
        osc = ozd + nz
        u0 = p_f[osc]
        y0 = p_f[osc + 1]
        so = n * (nz + 2)
        if first:                            # start from the latched register (or u0 when it is not exposed)
            s_f[so + nz] = float(obs[p_i[6] + n]) if p_i[6] >= 0 else np.rint(u0)
        cnt = obs[p_i[3] + n]
        ok = cnt > 0
        if p_i[4] >= 0 and obs[p_i[4] + n] == 0:
            ok = False
        if p_i[5] >= 0 and obs[p_i[5] + n] != 0:
            ok = False
        y = 0.0
        if cnt > 0:
            if latest:
                y = float(obs[p_i[2] + n * m + cnt - 1])
            else:
                acc = 0.0
                for j in range(cnt):
                    acc += float(obs[p_i[2] + n * m + j])
                y = acc / cnt
        st = s_i[1 + n]
        if st == ST_INIT and ok:
            # warm hand-over: steady state that explains y at the applied command
            udev = s_f[so + nz] - u0
            gyu = 0.0
            gyd = 0.0
            for i in range(nz):
                gyu += p_f[oCy + i] * p_f[ozu + i]
                gyd += p_f[oCy + i] * p_f[ozd + i]
            dd = (y - y0 - gyu * udev) / gyd
            for i in range(nz):
                s_f[so + i] = udev * p_f[ozu + i] + dd * p_f[ozd + i]
            st = ST_LOCK
            s_i[1 + n] = st
        if st != ST_LOCK:
            continue
        innov = 0.0
        if ok:
            yh = 0.0
            for i in range(nz):
                yh += p_f[oCy + i] * s_f[so + i]
            innov = y - y0 - yh
            for i in range(nz):
                s_f[so + i] += innov * p_f[oLk + i]
        s_f[so + nz + 1] = innov
        ud = 0.0
        for i in range(nz):
            ud -= p_f[oKf + i] * s_f[so + i]
        u = u0 + ud
        u = np.rint(u)
        if u < cmin:
            u = cmin
        if u > cmax:
            u = cmax
        s_f[so + nz] = u
        act[p_i[7] + n] = np.int64(u)
        # predict: z <- Phi z + Gz (u - u0)
        tmp = s_f[N * (nz + 2):N * (nz + 2) + nz]
        for i in range(nz):
            acc = 0.0
            for j in range(nz):
                acc += p_f[oPhi + i * nz + j] * s_f[so + j]
            tmp[i] = acc + p_f[oGz + i] * (u - u0)
        for i in range(nz):
            s_f[so + i] = tmp[i]
    s_i[0] = 1


def compiled_kalman_lqg(**params) -> "CompiledLqg":
    """The compiled LQG (lock only, warm start; 07 P0-9 flat ABI). Same Params as :class:`KalmanLQG`."""
    return CompiledLqg(params)


class CompiledLqg(CompiledController):
    """CompiledController whose parameters come from the nominal design at reset (spec-dependent packing)."""

    name = "kalman_lqg_c"

    def __init__(self, params=None, **kw):
        self._py = KalmanLQG(params, **kw)
        p = self._py.params
        if p.start != "warm" or p.probe_peak:
            raise ValueError("kalman_lqg_c implements the lock (warm start) only; use kalman_lqg for acquisition")
        super().__init__("kalman_lqg_c", lqg_init, lqg_step, latency_ticks=p.latency_ticks)
        self.params = p

    @classmethod
    def from_params(cls, params):
        return cls(params)

    def reset(self, spec: ControllerSpec) -> None:
        d = self._py.build_design(spec)
        self.design = d
        self.params_f64 = _pack_design(d, self.params.measurement == "latest", float(spec.heater_dac.code_min),
                                       float(spec.heater_dac.code_max), float(self.params.lock_pm))
        self.params_i64 = np.array([1 if self.params.measurement == "latest" else 0], np.int64)
        super().reset(spec)

    def telemetry(self) -> dict[str, np.ndarray]:
        d = self.design
        _, _, s_i, s_f = self.state
        B = s_f.shape[0]
        N, nz = d.Phi.shape[0], d.nz
        blk = s_f[:, :N * (nz + 2)].reshape(B, N, nz + 2)
        z = blk[..., :nz]
        det = np.einsum("nz,bnz->bn", d.Cd, z)
        st = s_i[:, 1:]
        lock = st == ST_LOCK
        return {"state": st.astype(float), "locked": (lock & (np.abs(det) < self.params.lock_pm)).astype(float),
                "detuning_est_pm": np.where(lock, d.det0_pm[None] + det, np.nan), "integrator": z[..., d.nc],
                "innovation_codes": blk[..., nz + 1].copy()}


register("kalman_lqg_c", CompiledLqg)
