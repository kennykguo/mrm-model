"""The oracle: a Truth-fed controller that holds each ring at its OMA-optimal, open-loop-stable detuning (07 P1-1).

It is NOT a controller a student may run against the plant: it reads Truth (the sampled PlantParams and the plant
state). ``Oracle.truth_fed = True``, so ``mrm.control.registry`` refuses to register or create it (07 P1-1); it is
never registered. It serves two users:

- the OMA-penalty metric (07 P0-8, 4.8): the OMA a controller loses against the oracle at the same instant,
  10 log10(OMA_oracle / OMA) (:func:`oma_penalty_db`), with OMA_oracle from :func:`oracle_target` (static: the best
  OMA among open-loop-stable points at the current conditions);
- the replicas (``mrm/validation/replicas.py``): stabilized relOMA maps across bias and temperature (VAL-4), the
  oracle flatness bound (BND-4), the quasi-static mode (07 P0-10), through :func:`oracle_target` and :func:`reloma_db`.

Target (07 P0-1 item 7). For each ring, scan the detuning delta = lambda_res - lambda_line with the exact bus optics
of the TRUE world (``mrm.control.models.oma_curve``: the heater supplies whatever the point needs) and take the
largest OMA = |P_out1 - P_out0| (its line at the bus output) among points where the self-heating loop is open-loop
stable, Z_opt(0) dP_abs/dT_R < 1 (the heater held, the loop gain magnitude is largest at DC for a passive Foster
network, so this DC test is sufficient); refined by golden-section search. The unrestricted optimum is reported next
to it (07 P0-1 item 7). In g26 nominal the optimum is about +107 pm (resonance red of the laser, self-locked side,
loop gain about -5) and the unrestricted optimum about -100 pm sits on the unstable side (loop gain about +5).

Holding (dynamic). The oracle knows the world exactly, so it needs no estimator. At reset it linearizes the TRUE
world at its target (``models.linearize_bank(..., pp=truth, measurement="none")``: the actuation delay only, no
measurement delay) and builds the LQR plus the regulator (Francis) feed-forward of
``mrm.control.baselines.kalman_lqg.design_lqg`` on that model (``kalman=False``: no filter). Designs, lag maps and
delay lines are per world. A target the heater cannot hold (:func:`reachable`: heater power <= 0, a code at or past
the DAC range or in the driver's compliance, or a bank above the array supply's power or current limit; 3 of 4
worlds of one LHS draw in g26) is not regulated: the oracle holds the code nearest to it and reports it in ``reach`` /
``OracleTarget.reachable``. At every edge it forms the exact model state from
Truth: the ring's own thermal lag states (``models.ring_block_lags``), the commands still in the DAC pipeline (it
issued them), and the disturbance d = (delta_true - delta*)/s - c_R z (everything that moves the ring other than its
own heater and absorption: ambient, aggressor, neighbours, laser wavelength), with its rate from the change since
the last edge. The command is u = u* - K_f [z; u-line; d + I; r], rounded and clipped to the code range, where I
integrates the TRUE detuning error (ki_per_step pm per pm per step): with the exact d the loop has no other DC
correction, and the linearization's heater gain is wrong far from u* (P ~ code^2: an aggressor that takes 8000
codes off the heater lowers dP/dcode by about 24 %). Its error comes only from the linearization transients and the
integer code; it is a benchmark, not a realizable controller.

Interface for the engine (07 P1-1 "used by the replicas and the OMA-penalty metric"): ``Oracle(pp, cfg)`` then
``reset(spec)`` (or ``reset(spec, truth_pp, truth_state, cfg)``) and ``step(obs, truth) -> {"tx.heater": (B, N)
int64}`` with ``truth`` a :class:`TruthState` filled by the plant at the edge (thermal lag states, ring temperatures,
per-world conditions). ``mrm.sim.runner`` calls ``ctrl.step(obs, truth=plant.truth())``; its ``Truth`` does not carry
the lag states or the Conditions yet (docs/handoff/controllers_c.md).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mrm.control import models as M
from mrm.control.baselines.kalman_lqg import design_lqg

__all__ = ["TruthState", "OracleTarget", "oracle_target", "oma_penalty_db", "reloma_db", "reachable", "Oracle"]

_PM = 1e12


@dataclass
class TruthState:
    """Plant Truth at a controller edge (for the oracle only).

    tick: plant tick of the edge; x: (B, M) thermal lag states in ThermalSystem order (the state at the edge);
    t_ring_k: (B, N) ring-mode temperatures (K); conditions: per world ``mrm.control.models.Conditions`` (line
    wavelengths and chip powers after the scenario, NRZ levels, data statistics, base temperature);
    heater_applied: (B, N) latched heater codes (used at the first step)."""
    tick: int
    x: np.ndarray
    t_ring_k: np.ndarray
    conditions: list
    heater_applied: np.ndarray | None = None


@dataclass(frozen=True)
class OracleTarget:
    """Per world and ring (B, N): detuning_m (the OMA-optimal stable detuning), oma_w, oma_unrestricted_w,
    detuning_unrestricted_m, p_in_w (bus power at the ring on its line), reloma_db = 10 log10(OMA / P_in) (07 D9
    oma_over_pin), loop_gain (Z_opt dP_abs/dT_R at the target, < 1), heater_code (static, float); reachable: the
    heater can hold the target (:func:`reachable`: positive heater power, a code inside the DAC range with heater
    authority dP/dcode > 0, and the bank's heater power and current within the array supply); an OMA penalty against an
    unreachable target measures the world, not the controller. static: the StaticPoint of each world at its
    target."""
    detuning_m: np.ndarray
    oma_w: np.ndarray
    oma_unrestricted_w: np.ndarray
    detuning_unrestricted_m: np.ndarray
    p_in_w: np.ndarray
    reloma_db: np.ndarray
    loop_gain: np.ndarray
    heater_code: np.ndarray
    reachable: np.ndarray | None = None
    static: list = field(default_factory=list, repr=False)


def reachable(pp, sp) -> np.ndarray:
    """(N,) True where world ``sp`` (a models.StaticPoint of PlantParams ``pp``) can be held by the heater: heater
    power > 0 (the ring is not already red of the target with the heater off), the code strictly inside the DAC range
    with dP/dcode > 0 (not clipped, not in the driver's compliance), and the array supply able to deliver the whole
    bank (``StaticPoint.supply_scale`` = 1: total power within array.heater_total_power_max_w AND total current
    within array.heater_supply_current_max_a; above either the supply droops every heater, 07 P1-12, and then no ring
    of the world is reachable)."""
    h = pp.heater
    ok = ((np.asarray(sp.heater_power_w) > 0) & (np.asarray(sp.heater_code) > float(h.code_min) + 0.5)
          & (np.asarray(sp.heater_code) < float(h.code_max) - 0.5))
    if np.all(np.isfinite(np.asarray(sp.dp_dcode, float))):
        ok &= np.asarray(sp.dp_dcode) > 0
    if float(getattr(sp, "supply_scale", 1.0)) < 1.0 - 1e-9:
        ok[:] = False
    return ok


def oracle_target(pp, conditions, *, stable_margin: float = 0.0, cfg=None) -> OracleTarget:
    """The oracle's target in every world (TRUE PlantParams ``pp``; ``conditions``: one Conditions per world, or
    None to take each world's nominal conditions from ``cfg`` via ``models.conditions_from_config``)."""
    B, N = int(pp.batch), int(pp.n_rings)
    out = {k: np.zeros((B, N)) for k in ("d", "o", "ou", "du", "pin", "lg", "code")}
    reach = np.zeros((B, N), bool)
    statics = []
    for b in range(B):
        cond = (M.conditions_from_config(pp, b, cfg=cfg) if conditions is None else conditions[b])
        d, o, ou, du = M.oma_optimal_detuning(pp, b, cond=cond, stable_margin=stable_margin)
        sp = M.static_point(pp, b, d, cond=cond)
        statics.append(sp)
        out["d"][b], out["o"][b], out["ou"][b], out["du"][b] = d, o, ou, du
        out["pin"][b] = sp.p_in_w
        out["lg"][b] = sp.loop_gain_opt
        out["code"][b] = sp.heater_code
        reach[b] = reachable(pp, sp)
    return OracleTarget(detuning_m=out["d"], oma_w=out["o"], oma_unrestricted_w=out["ou"],
                        detuning_unrestricted_m=out["du"], p_in_w=out["pin"],
                        reloma_db=reloma_db(out["o"], out["pin"]), loop_gain=out["lg"], heater_code=out["code"],
                        reachable=reach, static=statics)


def reloma_db(oma_w, p_in_w) -> np.ndarray:
    """relOMA = OMA / P_in in dB (07 D9: "1 mW of laser into the MRM", P_in = power at the ring on its line)."""
    return 10.0 * np.log10(np.maximum(np.asarray(oma_w, float), 1e-300) / np.asarray(p_in_w, float))


def oma_penalty_db(oma_w, oracle_oma_w) -> np.ndarray:
    """OMA penalty against the oracle at the same instant (07 P0-8): 10 log10(OMA_oracle / OMA), dB (>= 0 when the
    controller is worse). OMA 0 gives +inf."""
    o = np.asarray(oma_w, float)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(np.asarray(oracle_oma_w, float) / np.where(o > 0, o, 0.0))


class Oracle:
    """Truth-fed hold at the OMA-optimal stable detuning (module docstring). Never registered (truth_fed).

    Parameters: stable_margin (target loop gain < 1 - margin), q_pm / r_codes (LQR scales of the true-model design),
    ki_per_step (integral of the true detuning error added to d, per step), retarget (re-solve the target and the
    true linear model of a world when its conditions move: chip power by more than ``retarget_power_frac``, a line by
    more than ``retarget_lambda_m``, a junction level by more than ``retarget_v``), retarget_min_ticks (rate limit)."""

    name = "oracle"
    truth_fed = True
    latency_ticks = 0
    fast_kernel = None

    def __init__(self, pp=None, cfg=None, *, stable_margin: float = 0.0, q_pm: float = 0.5, r_codes: float = 400.0,
                 ki_per_step: float = 0.3, retarget: bool = True, retarget_power_frac: float = 2e-3,
                 retarget_lambda_m: float = 0.2e-12, retarget_v: float = 2e-3, retarget_min_ticks: int = 200):
        self._pp0, self._cfg0 = pp, cfg
        self.stable_margin = float(stable_margin)
        self.q_pm = float(q_pm)
        self.r_codes = float(r_codes)
        self.ki = float(ki_per_step)
        self.retarget = bool(retarget)
        self.tol = (float(retarget_power_frac), float(retarget_lambda_m), float(retarget_v))
        self.retarget_min_ticks = int(retarget_min_ticks)
        self.spec = None
        self.n_retargets = 0

    # -- per-world design
    def _design_world(self, b: int, cond) -> None:
        pp, spec = self.pp, self.spec
        N = int(pp.n_rings)
        d_opt, oma, oma_u, d_u = M.oma_optimal_detuning(pp, b, cond=cond, stable_margin=self.stable_margin)
        lms = M.linearize_bank(self.cfg, M.OperatingPoint(detuning_m=d_opt), spec.period_s,
                               latency_ticks=self.latency_ticks, measurement="none", pp=pp, b=b, cond=cond)
        d = design_lqg(lms, q_pm=self.q_pm, r_codes=self.r_codes, kalman=False)     # full state: no filter
        old = self.designs[b]
        self.designs[b] = d
        self.lagmaps[b] = [M.ring_block_lags(pp, b, n) for n in range(N)]
        self.cR[b] = [lm.Cc[1] / lm.Dc[1, 1] for lm in lms]                 # K per grouped state
        self.det_target[b] = d_opt
        self.oma_target[b] = oma
        self.reach[b] = reachable(pp, lms[0].op)
        self.conds[b] = cond
        if old is None:
            self.bumpless[b] = True               # first design: start from the code in the register
        else:
            # retarget: the commands in flight change reference; the integral (the linearization's gain error)
            # carries over, and the new model's feed-forward moves the ring to the new equilibrium at once
            self.u_line[b] = self.u_line[b] + (old.u0 - d.u0)[:, None]

    def reset(self, spec, truth_pp=None, truth: TruthState | None = None, cfg=None) -> None:
        """Targets from the TRUE world at the conditions of ``truth``; LQR + feed-forward on the true linear model.

        ``truth_pp`` / ``cfg`` default to the constructor's. With ``truth`` None (a runner that calls reset(spec)
        only) the design waits for the first step's TruthState."""
        self.spec = spec
        self.cfg = self._cfg0 if cfg is None else cfg
        self.pp = pp = self._pp0 if truth_pp is None else truth_pp
        if pp is None:
            raise ValueError("Oracle needs the TRUE PlantParams (reset(spec, truth_pp, ...) or Oracle(pp=...))")
        self._designed = False
        if truth is None:
            return
        self._design_all(truth)

    def _design_all(self, truth: TruthState) -> None:
        spec, pp = self.spec, self.pp
        B, N = int(pp.batch), int(pp.n_rings)
        if (spec.n_batch, spec.n_rings) != (B, N):
            raise ValueError(f"spec is (B, N) = {(spec.n_batch, spec.n_rings)}, Truth {(B, N)}")
        self.designs = [None] * B
        self.lagmaps = [None] * B
        self.cR = [None] * B
        self.conds = [None] * B
        self.det_target = np.zeros((B, N))
        self.oma_target = np.zeros((B, N))
        self.reach = np.ones((B, N), bool)
        self.bumpless = np.ones(B, bool)
        # commands in flight (deviations from each world's u*), newest first: one (N, nd_b) array per world (the
        # thermal block and the delay line may differ between worlds: sampled transport delay, ROM family)
        self.u_line = [None] * B
        for b in range(B):
            self._design_world(b, truth.conditions[b])
        applied = (np.asarray(truth.heater_applied, float) if truth.heater_applied is not None
                   else np.stack([d.u0 for d in self.designs]))
        for b in range(B):
            d = self.designs[b]
            nd = d.nc - d.models[0].Ac.shape[0]
            self.u_line[b] = np.repeat((applied[b] - d.u0)[:, None], nd, axis=1)
        self.d_prev = np.full((B, N), np.nan)
        self.integ = np.zeros((B, N))
        self.code_min = float(spec.heater_dac.code_min)
        self.code_max = float(spec.heater_dac.code_max)
        self._last = np.zeros((B, N))
        self._last_retarget = np.full(B, -(1 << 62), np.int64)
        self.n_retargets = 0
        self._designed = True

    @property
    def target(self) -> OracleTarget:
        """Current target (detuning and OMA per world and ring; the other fields are not tracked after reset)."""
        nan = np.full_like(self.det_target, np.nan)
        return OracleTarget(detuning_m=self.det_target.copy(), oma_w=self.oma_target.copy(), oma_unrestricted_w=nan,
                            detuning_unrestricted_m=nan, p_in_w=nan, reloma_db=nan, loop_gain=nan,
                            heater_code=np.stack([d.u0 for d in self.designs]), reachable=self.reach.copy())

    def _moved(self, a, b) -> bool:
        tp, tl, tv = self.tol
        p0 = np.asarray(a.p_chip_w, float)
        if np.any(np.abs(np.asarray(b.p_chip_w, float) - p0) > tp * np.maximum(p0, 1e-30)):
            return True
        if np.any(np.abs(np.asarray(b.lam_m) - np.asarray(a.lam_m)) > tl):
            return True
        if np.any(np.abs(np.asarray(b.v_lo) - np.asarray(a.v_lo)) > tv) or \
                np.any(np.abs(np.asarray(b.v_hi) - np.asarray(a.v_hi)) > tv):
            return True
        return bool(np.any(np.abs(np.asarray(b.rho) - np.asarray(a.rho)) > 1e-3))

    def _disturbance_pm(self, truth: TruthState):
        """(z_dev [(N, nx_b) per world], d (B, N) pm of open-loop resonance shift, detuning error (B, N) pm) from
        Truth."""
        B, N = self._last.shape
        zdev = []
        dpm = np.zeros((B, N))
        err = np.zeros((B, N))
        for b in range(B):
            d = self.designs[b]
            sp = d.models[0].op
            nx = d.models[0].Ac.shape[0]
            zb = np.zeros((N, nx))
            det = M.resonance_detuning_m(self.pp, b, truth.t_ring_k[b], truth.conditions[b])
            for n in range(N):
                idx, res, src = self.lagmaps[b][n]
                z = truth.x[b, idx] / res
                z_op = np.where(src == 0, sp.heater_power_w[n], sp.p_abs_w[n])
                zb[n, :idx.size] = z - z_op
                if nx > idx.size:                   # heater settle-filter state (models._model_for_ring): its
                    zb[n, idx.size:] = self.u_line[b][n, -1]     # input is the code leaving the delay line
                s = d.models[n].Dc[1, 1]
                d_k = (det[n] - self.det_target[b, n]) / s - self.cR[b][n] @ zb[n]
                dpm[b, n] = d_k * s * _PM
                err[b, n] = (det[n] - self.det_target[b, n]) * _PM
            zdev.append(zb)
        return zdev, dpm, err

    def step(self, obs, truth: TruthState) -> dict[str, np.ndarray]:
        if self.spec is None:
            raise RuntimeError("Oracle: call reset(spec, truth_pp, truth) first")
        if not (hasattr(truth, "x") and hasattr(truth, "conditions")):
            raise TypeError("Oracle.step needs a mrm.validation.oracle.TruthState (thermal lag states x, ring "
                            "temperatures, per-world Conditions); got " + type(truth).__name__)
        if not self._designed:
            self._design_all(truth)
        B, N = self._last.shape
        if self.retarget:
            for b in range(B):
                if (truth.tick - self._last_retarget[b] >= self.retarget_min_ticks
                        and self._moved(self.conds[b], truth.conditions[b])):
                    self._design_world(b, truth.conditions[b])
                    self._last_retarget[b] = truth.tick
                    self.n_retargets += 1
        zdev, dpm, err = self._disturbance_pm(truth)
        ts_ms = self.spec.period_s * 1e3
        rate = np.where(np.isfinite(self.d_prev), (dpm - self.d_prev) / ts_ms, 0.0)
        rate = np.where(self.bumpless[:, None], 0.0, rate)
        self.d_prev = dpm
        # integral of the TRUE detuning error, added to d: it removes what the linearization gets wrong (the heater
        # gain at a code far from the linearization point, the TCR); with the exact d the loop is otherwise open in DC
        self.integ = self.integ + self.ki * err
        codes = np.zeros((B, N))
        for b in range(B):
            d = self.designs[b]
            if self.bumpless[b]:
                # bumpless (re)start: the integrator takes the value that reproduces the code now in the register
                zc = np.concatenate([zdev[b], self.u_line[b]], axis=1)[:, :d.nc]
                kd = d.Kf[:, d.nc]
                rest = np.einsum("nz,nz->n", d.Kf[:, :d.nc], zc) + kd * dpm[b]
                applied = d.u0 + self.u_line[b][:, 0]
                kd_safe = np.where(np.abs(kd) > 0, kd, 1.0)
                self.integ[b] = np.where(np.abs(kd) > 0, (d.u0 - applied - rest) / kd_safe, 0.0)
                self.bumpless[b] = False
            z = np.concatenate([zdev[b], self.u_line[b], (dpm[b] + self.integ[b])[:, None], rate[b][:, None]],
                               axis=1)[:, :d.nz]
            udev = -np.einsum("nz,nz->n", d.Kf, z)
            # an unreachable target (reachable(): beyond the code range, no heater authority, bank over the supply)
            # is not regulated: the oracle holds the code nearest to it (the range end) instead of winding up
            u = np.where(self.reach[b], d.u0 + udev, d.u0)
            codes[b] = np.clip(np.rint(np.nan_to_num(u, nan=d.u0)), self.code_min, self.code_max)
            self.integ[b] = np.where(self.reach[b], self.integ[b], 0.0)
            self.u_line[b] = np.concatenate([(codes[b] - d.u0)[:, None], self.u_line[b][:, :-1]], axis=1)
        self._last = codes
        return {"tx.heater": codes.astype(np.int64)}

    def telemetry(self) -> dict[str, np.ndarray]:
        return {"detuning_est_pm": self.det_target * _PM, "integrator": self.integ.copy()}
