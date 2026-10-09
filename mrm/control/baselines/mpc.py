"""Linear MPC heater lock with code limits and slew limits (ARCHITECTURE section 6 item 8; 04 C15; 07 P1-1).

The estimator, the delay compensation, the acquisition and the knowledge rules are those of
:class:`mrm.control.baselines.kalman_lqg.KalmanLQG` (this class derives from it): a steady-state Kalman filter on the
delay-augmented nominal model z = [x; u-line; d; r] built by ``mrm.control.models.linearize_bank(spec.nominal)``.
Only the control law differs. Every controller period, per ring:

    minimize   sum_{j=1..Np} (delta_j / q_pm)^2 + sum_{i=0..Nu-1} ((U_i - U_{i-1}) / r_codes)^2
    subject to code_min - u0 <= U_i <= code_max - u0,   |U_i - U_{i-1}| <= slew   (U_{-1} = the last command)

with delta_j = F_j z + H_j U the predicted detuning error (pm) of the augmented model (the commands in flight are
states, so the first predictions are not affected by U; the (d, r) states extrapolate the disturbance as a ramp,
which acts as a ramp feed-forward), U the heater moves (deviations from the nominal lock code u0), held after Nu
moves (move blocking) until the end of the horizon. The first move is applied (receding horizon), rounded and
clipped once more to the code range and the slew window. Because the blocked moves cannot follow a ramp to the end
of the horizon, the loop is not exactly type 2 (the LQG is): on the nominal linear model a 20 pm/ms ramp leaves a
steady 0.005 pm error (zero for a step).

QP solver (no new dependency): ADMM in the OSQP form (Stellato et al., Math. Prog. Comp. 12, 637 (2020), section 3)
on min 1/2 U'PU + q'U s.t. l <= A U <= h with A = [I; D] (D the first-difference matrix): one precomputed
(P + sigma I + rho A'A)^-1 per ring, relaxation alpha, a fixed rho scaled to diag(P), warm start from the previous
solution shifted by one move, and a stop when the primal and dual residuals are below ``qp_tol`` (or after
``qp_max_iter`` iterations). The bounds are boxes on [U; D U], so the projection is a clip. A verification test
compares it with ``scipy.optimize.minimize`` (trust-constr) and the KKT conditions on random problems with active
constraints.

The slew limit: the smaller of ``slew_codes_per_step`` (a design limit; 500 codes per 10 us step by default, far
above what a lock needs: the textbook's 720-830 codes/ms at 2000 degC/s is 8 codes per step) and the hardware value
(``ring.heater.slew_codes_per_s`` of the nominal design, times Ts) when finite, floored to whole codes (at least 1:
the commands are integers). Telemetry adds qp_iterations, qp_residual, slew_active and bound_active (the first move
sits on a constraint).

A compiled numba version (``mpc_c``, flat ABI, lock only) runs the same ADMM in float64 and is checked against this
class step by step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numba as nb
import numpy as np

from mrm.control import models as M
from mrm.control.api import (
    SPEC_I_M, SPEC_I_N_RINGS, CompiledController, ControllerSpec, act_offset, obs_offset, user_f64, user_i64,
)
from mrm.control.baselines.kalman_lqg import (
    _APPLIED, _COUNT, _HEATER, _OVER, _SAMPLES, _VALID, ST_INIT, ST_LOCK, KalmanLQG, LqgDesign,
)
from mrm.control.registry import register

__all__ = ["LinearMPC", "MpcDesign", "design_mpc", "admm_box_qp", "mpc_equivalent_gain", "CompiledMpc"]


@dataclass(frozen=True)
class MpcDesign:
    """Per-ring MPC matrices (axis 0 = ring). F (N, Np, nz) pm per state, H (N, Np, Nu) pm per code, P (N, Nu, Nu)
    QP Hessian, Fq (N, Nu, nz) and gprev (N, Nu): q = Fq z + gprev u_prev, Minv (N, Nu, Nu) the ADMM matrix,
    A (2 Nu, Nu) = [I; D], rho (N,), sigma, alpha. lqg: the estimator design."""
    lqg: LqgDesign
    Np: int
    Nu: int
    F: np.ndarray
    H: np.ndarray
    P: np.ndarray
    Fq: np.ndarray
    gprev: np.ndarray
    Minv: np.ndarray
    A: np.ndarray
    rho: np.ndarray
    sigma: float
    alpha: float
    slew: float
    extra: dict = field(default_factory=dict)


def _diff_matrix(nu: int) -> np.ndarray:
    D = np.eye(nu)
    D[1:, :-1] -= np.eye(nu - 1)
    return D


def design_mpc(lqg: LqgDesign, *, Np: int = 20, Nu: int = 4, q_pm: float = 1.0, r_codes: float = 40.0,
               slew_codes: float = 500.0, rho_scale: float = 0.1, sigma: float = 1e-9, alpha: float = 1.6
               ) -> MpcDesign:
    """Prediction and QP matrices of every ring (module docstring). Np >= Nu >= 1."""
    if not (Np >= Nu >= 1):
        raise ValueError(f"need Np >= Nu >= 1, got Np={Np}, Nu={Nu}")
    N, nz = lqg.Phi.shape[0], lqg.nz
    D = _diff_matrix(Nu)
    A = np.concatenate([np.eye(Nu), D], axis=0)
    F = np.zeros((N, Np, nz))
    H = np.zeros((N, Np, Nu))
    P = np.zeros((N, Nu, Nu))
    Fq = np.zeros((N, Nu, nz))
    gprev = np.zeros((N, Nu))
    Minv = np.zeros((N, Nu, Nu))
    rho = np.zeros(N)
    e1 = np.zeros(Nu)
    e1[0] = 1.0
    wq = 1.0 / q_pm ** 2
    wr = 1.0 / r_codes ** 2
    for n in range(N):
        Phi, Gz, Cd = lqg.Phi[n], lqg.Gz[n], lqg.Cd[n]
        Pk = np.eye(nz)
        resp = np.zeros((Np + 1, nz))         # state response to a unit input applied from step i (cumulative)
        Gcols = np.zeros((Np, nz, Np))         # z_j sensitivity to u_i
        zfree = np.eye(nz)
        for j in range(1, Np + 1):
            zfree = Phi @ zfree
            F[n, j - 1] = Cd @ zfree
        # z_j = Phi^j z0 + sum_{i<j} Phi^{j-1-i} Gz u_i ; u_i = U[min(i, Nu-1)]
        powers = [np.eye(nz)]
        for _ in range(Np):
            powers.append(Phi @ powers[-1])
        for j in range(1, Np + 1):
            for i in range(j):
                col = min(i, Nu - 1)
                H[n, j - 1, col] += Cd @ powers[j - 1 - i] @ Gz
        Pn = 2.0 * (wq * H[n].T @ H[n] + wr * D.T @ D)
        P[n] = 0.5 * (Pn + Pn.T)
        Fq[n] = 2.0 * wq * H[n].T @ F[n]
        gprev[n] = -2.0 * wr * D.T @ e1
        rho[n] = rho_scale * float(np.mean(np.diag(P[n])))
        Minv[n] = np.linalg.inv(P[n] + sigma * np.eye(Nu) + rho[n] * A.T @ A)
        del Pk, resp, Gcols
    return MpcDesign(lqg=lqg, Np=int(Np), Nu=int(Nu), F=F, H=H, P=P, Fq=Fq, gprev=gprev, Minv=Minv, A=A, rho=rho,
                     sigma=float(sigma), alpha=float(alpha), slew=float(slew_codes))


def admm_box_qp(P, q, A, lo, hi, Minv, rho, sigma=1e-9, alpha=1.6, x0=None, z0=None, y0=None, max_iter=200,
                tol=1e-6):
    """ADMM (OSQP form) for min 1/2 x'Px + q'x s.t. lo <= A x <= hi, vectorized over leading axes.

    P (..., n, n) (or broadcastable), q (..., n), A (m, n), lo/hi (..., m), Minv (..., n, n) =
    (P + sigma I + rho A'A)^-1, rho (...,) or scalar. Each problem stops on its own when its primal residual
    max|A x - z| and dual residual max|rho A'(z_k - z_k-1)| are both below tol (absolute), so a batch gives the same
    result as solving each problem alone. Returns (x, z, y, iterations (...,), primal residual (...,), dual residual
    (...,))."""
    q = np.asarray(q, float)
    shp = q.shape[:-1]
    n = q.shape[-1]
    m = A.shape[0]
    x = np.zeros(shp + (n,)) if x0 is None else np.array(np.broadcast_to(x0, shp + (n,)), float)
    lo = np.broadcast_to(lo, shp + (m,))
    hi = np.broadcast_to(hi, shp + (m,))
    z = np.clip(np.einsum("mn,...n->...m", A, x), lo, hi) if z0 is None else np.array(z0, float)
    y = np.zeros(shp + (m,)) if y0 is None else np.array(np.broadcast_to(y0, shp + (m,)), float)
    r = np.broadcast_to(np.asarray(rho, float), shp)[..., None]
    Minv = np.broadcast_to(Minv, shp + (n, n))
    active = np.ones(shp, bool)
    iters = np.zeros(shp, np.int64)
    rp = np.full(shp, np.inf)
    rd = np.full(shp, np.inf)
    for it in range(1, max_iter + 1):
        if not active.any():
            break
        a = active[..., None]
        rhs = sigma * x - q + np.einsum("mn,...m->...n", A, r * z - y)
        xt = np.einsum("...ij,...j->...i", Minv, rhs)
        zt = np.einsum("mn,...n->...m", A, xt)
        x_new = alpha * xt + (1.0 - alpha) * x
        zr = alpha * zt + (1.0 - alpha) * z
        z_new = np.clip(zr + y / r, lo, hi)
        y_new = y + r * (zr - z_new)
        rd_new = np.max(np.abs(r * np.einsum("mn,...m->...n", A, z_new - z)), axis=-1)
        rp_new = np.max(np.abs(np.einsum("mn,...n->...m", A, x_new) - z_new), axis=-1)
        x = np.where(a, x_new, x)
        z = np.where(a, z_new, z)
        y = np.where(a, y_new, y)
        rd = np.where(active, rd_new, rd)
        rp = np.where(active, rp_new, rp)
        iters = np.where(active, it, iters)
        active = active & ~((rp_new < tol) & (rd_new < tol))
    return x, z, y, iters, rp, rd


def mpc_equivalent_gain(md: MpcDesign, ring: int):
    """The unconstrained MPC as a linear law U0 = -k_z z - k_p u_prev (first row of -P^-1 [Fq, gprev])."""
    Pinv = np.linalg.inv(md.P[ring])
    kz = (Pinv @ md.Fq[ring])[0]
    kp = (Pinv @ md.gprev[ring])[0]
    return kz, kp


def mpc_closed_loop_spectral_radius(md: MpcDesign, ring: int, plant: M.LinearModel, gain: float = 1.0) -> float:
    """Spectral radius of the unconstrained MPC + Kalman filter (nominal design) with another linear plant."""
    d = md.lqg
    Phi, Gz, Cy, Lk = d.Phi[ring], d.Gz[ring], d.Cy[ring], d.Lk[ring]
    kz, kp = mpc_equivalent_gain(md, ring)
    Pp, Gp, _, Cp, _, _ = plant.delayed()
    Cpy = Cp[0]
    npl, nz = Pp.shape[0], Phi.shape[0]
    Mz = np.eye(nz) - np.outer(Lk, Cy)
    # state [zt; zp; u_prev]; z = Mz zp + L Cpy zt; u = -kz z - kp u_prev
    n_all = npl + nz + 1
    Acl = np.zeros((n_all, n_all))
    uz_t = -kz @ np.outer(Lk, Cpy)
    uz_p = -kz @ Mz
    up = -kp
    Acl[:npl, :npl] = Pp + gain * np.outer(Gp, uz_t)
    Acl[:npl, npl:npl + nz] = gain * np.outer(Gp, uz_p)
    Acl[:npl, -1] = gain * Gp * up
    Acl[npl:npl + nz, :npl] = Phi @ np.outer(Lk, Cpy) + np.outer(Gz, uz_t)
    Acl[npl:npl + nz, npl:npl + nz] = Phi @ Mz + np.outer(Gz, uz_p)
    Acl[npl:npl + nz, -1] = Gz * up
    Acl[-1, :npl] = uz_t
    Acl[-1, npl:npl + nz] = uz_p
    Acl[-1, -1] = up
    return float(np.max(np.abs(np.linalg.eigvals(Acl))))


@register("mpc")
class LinearMPC(KalmanLQG):
    """Kalman filter + constrained linear MPC with code and slew limits (module docstring)."""

    name = "mpc"

    @dataclass
    class Params(KalmanLQG.Params):
        horizon_steps: int = 20               # Np
        control_moves: int = 4                # Nu (move blocking)
        r_du_codes: float = 40.0              # move penalty scale (codes per step)
        slew_codes_per_step: float = 500.0    # design slew limit (codes per controller step)
        code_margin: int = 0                  # keep the commands this many codes inside the DAC range
        qp_max_iter: int = 100
        qp_tol: float = 1e-3                  # codes (primal) and cost units (dual)
        qp_rho_scale: float = 0.1
        qp_alpha: float = 1.6

    def build_mpc(self, spec: ControllerSpec) -> MpcDesign:
        p = self.params
        lqg = self.build_design(spec)
        slew = float(p.slew_codes_per_step)
        try:
            hw = M._cfg_si(spec.nominal, "ring.heater.slew_codes_per_s", None)
        except Exception:      # noqa: BLE001
            hw = None
        if hw is not None and np.isfinite(hw) and hw > 0:
            slew = min(slew, float(hw) * spec.period_s)
        # commands are whole codes: a fractional limit would let the rounded command step past it (Python path) or
        # leave a fractional command in the compiled filter state; floor it (at least 1 code per step)
        slew = max(1.0, math.floor(slew + 1e-9))
        return design_mpc(lqg, Np=p.horizon_steps, Nu=p.control_moves, q_pm=p.q_pm, r_codes=p.r_du_codes,
                          slew_codes=slew, rho_scale=p.qp_rho_scale, alpha=p.qp_alpha)

    def reset(self, spec: ControllerSpec) -> None:
        self.mpc = md = self.build_mpc(spec)
        super().reset(spec)
        B, N = spec.n_batch, spec.n_rings
        Nu = md.Nu
        self._x = np.zeros((B, N, Nu))
        self._zq = np.zeros((B, N, 2 * Nu))
        self._yq = np.zeros((B, N, 2 * Nu))
        self._was_locked = np.zeros((B, N), bool)
        self.qp_iter = np.zeros((B, N), np.int64)
        self.qp_res = np.zeros((B, N))
        self.slew_active = np.zeros((B, N), bool)
        self.bound_active = np.zeros((B, N), bool)

    def build_design(self, spec):         # KalmanLQG.reset calls this; reuse the estimator part of the MPC design
        md = getattr(self, "mpc", None)
        if md is not None and getattr(self, "_mpc_spec", None) is spec:
            return md.lqg
        self._mpc_spec = spec
        return super().build_design(spec)

    def _bounds(self):
        md = self.mpc
        d = md.lqg
        p = self.params
        Nu = md.Nu
        lo_c = self.code_min + p.code_margin - d.u0[None]
        hi_c = self.code_max - p.code_margin - d.u0[None]
        uprev = self.u - d.u0[None]
        lo = np.empty(self.u.shape + (2 * Nu,))
        hi = np.empty(self.u.shape + (2 * Nu,))
        lo[..., :Nu] = lo_c[..., None]
        hi[..., :Nu] = hi_c[..., None]
        lo[..., Nu:] = -md.slew
        hi[..., Nu:] = md.slew
        lo[..., Nu] += uprev
        hi[..., Nu] += uprev
        return lo, hi, uprev

    def _control(self, lock: np.ndarray) -> np.ndarray:
        md = self.mpc
        d = md.lqg
        p = self.params
        lo, hi, uprev = self._bounds()
        q = np.einsum("nuz,bnz->bnu", md.Fq, self.z) + md.gprev[None] * uprev[..., None]
        # warm start: shift the previous solution by one move
        x0 = np.concatenate([self._x[..., 1:], self._x[..., -1:]], axis=-1)
        fresh = lock & ~self._was_locked          # a newly locked ring starts from "hold the command"
        x0 = np.where(fresh[..., None] | ~lock[..., None], uprev[..., None], x0)
        y0 = np.where(fresh[..., None], 0.0, self._yq)
        self._was_locked = lock.copy()
        x, zq, yq, it, rp, rd = admm_box_qp(md.P[None], q, md.A, lo, hi, md.Minv[None], md.rho[None], md.sigma,
                                            md.alpha, x0=x0, z0=None, y0=y0, max_iter=p.qp_max_iter,
                                            tol=p.qp_tol)
        self._x = np.where(lock[..., None], x, self._x)
        self._yq = np.where(lock[..., None], yq, self._yq)
        self.qp_iter = np.where(lock, it, 0)
        self.qp_res = np.where(lock, np.maximum(rp, rd), 0.0)
        u = d.u0[None] + x[..., 0]
        # exact limits after rounding (the caller rounds and clips to the code range)
        u = np.clip(np.rint(u), self.u - md.slew, self.u + md.slew)
        tol = 0.5
        self.slew_active = lock & (np.abs(u - self.u) >= md.slew - tol)
        self.bound_active = lock & ((u <= self.code_min + p.code_margin + tol) |
                                    (u >= self.code_max - p.code_margin - tol))
        return u

    def telemetry(self) -> dict[str, np.ndarray]:
        t = super().telemetry()
        B, N = self.u.shape
        t["qp_iterations"] = self.qp_iter.astype(float)
        t["qp_residual"] = self.qp_res.astype(float)
        t["slew_active"] = self.slew_active.astype(float)
        t["bound_active"] = self.bound_active.astype(float)
        return t


# ------------------------------------------------------------------------------------------------ compiled version
# p_f64 header: [N, nz, nc, nw, mode_latest, code_min, code_max, lock_pm, Np, Nu, slew, sigma, alpha, max_iter, tol,
#                code_margin]; per ring: LQG block (Phi, Gz, Cy, Cd, Lk, Kf, zu, zd, u0 y0 y_zero y_pk det0) then
#                Fq (Nu*nz), gprev (Nu), Minv (Nu*Nu), rho
# s_f64 per ring: z (nz), u, innov, x (Nu), yq (2 Nu), then scratch; s_i64: [started, state per ring, iterations
# per ring]
_MHDR = 16


def _mpc_ring_block(nz, nu):
    return nz * nz + 7 * nz + 5 + nu * nz + nu + nu * nu + 1


def _pack_mpc(md: MpcDesign, mode_latest: bool, code_min: float, code_max: float, lock_pm: float, max_iter: int,
              tol: float, code_margin: float) -> np.ndarray:
    d = md.lqg
    N, nz, nu = d.Phi.shape[0], d.nz, md.Nu
    per = _mpc_ring_block(nz, nu)
    out = np.zeros(_MHDR + N * per)
    out[:_MHDR] = [N, nz, d.nc, d.nw, 1.0 if mode_latest else 0.0, code_min, code_max, lock_pm, md.Np, nu,
                   md.slew, md.sigma, md.alpha, max_iter, tol, code_margin]
    for n in range(N):
        o = _MHDR + n * per
        blk = np.concatenate([d.Phi[n].reshape(-1), d.Gz[n], d.Cy[n], d.Cd[n], d.Lk[n], d.Kf[n], d.zu[n], d.zd[n],
                              [d.u0[n], d.y0[n], d.y_zero[n], d.y_pk[n], d.det0_pm[n]],
                              md.Fq[n].reshape(-1), md.gprev[n], md.Minv[n].reshape(-1), [md.rho[n]]])
        out[o:o + per] = blk
    return out


@nb.njit(cache=True)
def mpc_init(spec_i64, spec_f64):
    """Compiled init: unpack the packed design and the channel offsets."""
    n_user_f = spec_f64.shape[0] - 8
    pf = np.empty(n_user_f)
    for i in range(n_user_f):
        pf[i] = user_f64(spec_f64, i)
    N = spec_i64[SPEC_I_N_RINGS]
    nz = int(pf[1])
    nu = int(pf[9])
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
    s_f = np.zeros(N * (nz + 2 + 3 * nu) + nz + 11 * nu)        # per-ring state, then scratch
    s_i = np.zeros(1 + 2 * N, np.int64)                         # started, state per ring, iterations per ring
    return pi, pf, s_i, s_f


@nb.njit(cache=True)
def mpc_step(p_i, p_f, s_i, s_f, obs, act):
    """Compiled step (warm start, lock only): Kalman update as kalman_lqg, then the ADMM QP of LinearMPC (per-ring
    stopping rule, as admm_box_qp)."""
    N = p_i[0]
    nz = p_i[1]
    m = p_i[8]
    latest = p_f[4] > 0.5
    cmin = p_f[5]
    cmax = p_f[6]
    nu = int(p_f[9])
    slew = p_f[10]
    sigma = p_f[11]
    alpha = p_f[12]
    max_iter = int(p_f[13])
    tol = p_f[14]
    margin = p_f[15]
    per = nz * nz + 7 * nz + 5 + nu * nz + nu + nu * nu + 1
    sper = nz + 2 + 3 * nu
    scr = N * sper
    tmp = s_f[scr:scr + nz]
    q = s_f[scr + nz:scr + nz + nu]
    rhs = s_f[scr + nz + nu:scr + nz + 2 * nu]
    xt = s_f[scr + nz + 2 * nu:scr + nz + 3 * nu]
    zt = s_f[scr + nz + 3 * nu:scr + nz + 5 * nu]
    lo = s_f[scr + nz + 5 * nu:scr + nz + 7 * nu]
    hi = s_f[scr + nz + 7 * nu:scr + nz + 9 * nu]
    zq = s_f[scr + nz + 9 * nu:scr + nz + 11 * nu]
    first = s_i[0] == 0
    for n in range(N):
        o = _MHDR + n * per
        oPhi = o
        oGz = oPhi + nz * nz
        oCy = oGz + nz
        oCd = oCy + nz
        oLk = oCd + nz
        oKf = oLk + nz
        ozu = oKf + nz
        ozd = ozu + nz
        osc = ozd + nz
        oFq = osc + 5
        ogp = oFq + nu * nz
        oMi = ogp + nu
        orho = oMi + nu * nu
        u0 = p_f[osc]
        y0 = p_f[osc + 1]
        rho = p_f[orho]
        so = n * sper
        sx = so + nz + 2
        sy = sx + nu
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
            for i in range(nu):              # the previous solution of a newly locked ring: hold the command
                s_f[sx + i] = udev
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
        uprev = s_f[so + nz] - u0
        for i in range(nu):
            acc = 0.0
            for j in range(nz):
                acc += p_f[oFq + i * nz + j] * s_f[so + j]
            q[i] = acc + p_f[ogp + i] * uprev
        for i in range(nu):
            lo[i] = cmin + margin - u0
            hi[i] = cmax - margin - u0
            lo[nu + i] = -slew
            hi[nu + i] = slew
        lo[nu] += uprev
        hi[nu] += uprev
        # warm start: the previous solution shifted by one move
        for i in range(nu - 1):
            s_f[sx + i] = s_f[sx + i + 1]
        for i in range(nu):                  # z0 = clip(A x0)
            v = s_f[sx + i]
            zq[i] = min(max(v, lo[i]), hi[i])
            dv = v - (s_f[sx + i - 1] if i > 0 else 0.0)
            zq[nu + i] = min(max(dv, lo[nu + i]), hi[nu + i])
        it_done = 0
        for it in range(1, max_iter + 1):
            it_done = it
            for i in range(nu):              # rhs = sigma x - q + A'(rho z - y), A = [I; D]
                v = rho * zq[i] - s_f[sy + i]
                v += rho * zq[nu + i] - s_f[sy + nu + i]
                if i + 1 < nu:
                    v += -(rho * zq[nu + i + 1] - s_f[sy + nu + i + 1])
                rhs[i] = sigma * s_f[sx + i] - q[i] + v
            for i in range(nu):
                acc = 0.0
                for j in range(nu):
                    acc += p_f[oMi + i * nu + j] * rhs[j]
                xt[i] = acc
            for i in range(nu):              # x_new, z~ = A xt
                zt[i] = xt[i]
                zt[nu + i] = xt[i] - (xt[i - 1] if i > 0 else 0.0)
            for i in range(nu):
                s_f[sx + i] = alpha * xt[i] + (1.0 - alpha) * s_f[sx + i]
            for i in range(2 * nu):          # zr, z_new, y_new; zt <- z_new - z_old
                zr = alpha * zt[i] + (1.0 - alpha) * zq[i]
                zn = min(max(zr + s_f[sy + i] / rho, lo[i]), hi[i])
                s_f[sy + i] = s_f[sy + i] + rho * (zr - zn)
                zt[i] = zn - zq[i]
                zq[i] = zn
            rd = 0.0
            for i in range(nu):
                v = zt[i] + zt[nu + i]
                if i + 1 < nu:
                    v += -zt[nu + i + 1]
                v = abs(rho * v)
                if v > rd:
                    rd = v
            rp = 0.0
            for i in range(nu):
                v = abs(s_f[sx + i] - zq[i])
                if v > rp:
                    rp = v
                dx = s_f[sx + i] - (s_f[sx + i - 1] if i > 0 else 0.0)
                v = abs(dx - zq[nu + i])
                if v > rp:
                    rp = v
            if rp < tol and rd < tol:
                break
        s_i[1 + N + n] = it_done
        u = np.rint(u0 + s_f[sx])
        u_last = s_f[so + nz]
        if u < u_last - slew:
            u = u_last - slew
        if u > u_last + slew:
            u = u_last + slew
        if u < cmin:
            u = cmin
        if u > cmax:
            u = cmax
        s_f[so + nz] = u
        act[p_i[7] + n] = np.int64(u)
        for i in range(nz):
            acc = 0.0
            for j in range(nz):
                acc += p_f[oPhi + i * nz + j] * s_f[so + j]
            tmp[i] = acc + p_f[oGz + i] * (u - u0)
        for i in range(nz):
            s_f[so + i] = tmp[i]
    s_i[0] = 1


class CompiledMpc(CompiledController):
    """Compiled MPC (lock only, warm start; flat ABI of 07 P0-9). Same Params as :class:`LinearMPC`."""

    name = "mpc_c"

    def __init__(self, params=None, **kw):
        self._py = LinearMPC(params, **kw)
        p = self._py.params
        if p.start != "warm" or p.probe_peak:
            raise ValueError("mpc_c implements the lock (warm start) only; use mpc for acquisition")
        super().__init__("mpc_c", mpc_init, mpc_step, latency_ticks=p.latency_ticks)
        self.params = p

    @classmethod
    def from_params(cls, params):
        return cls(params)

    def reset(self, spec: ControllerSpec) -> None:
        self._py.spec = spec
        md = self._py.build_mpc(spec)
        self.mpc = md
        p = self.params
        self.params_f64 = _pack_mpc(md, p.measurement == "latest", float(spec.heater_dac.code_min),
                                    float(spec.heater_dac.code_max), float(p.lock_pm), int(p.qp_max_iter),
                                    float(p.qp_tol), float(p.code_margin))
        self.params_i64 = np.array([1 if p.measurement == "latest" else 0], np.int64)
        super().reset(spec)

    def telemetry(self) -> dict[str, np.ndarray]:
        d = self.mpc.lqg
        _, _, s_i, s_f = self.state
        B = s_f.shape[0]
        N, nz, nu = d.Phi.shape[0], d.nz, self.mpc.Nu
        blk = s_f[:, :N * (nz + 2 + 3 * nu)].reshape(B, N, nz + 2 + 3 * nu)
        z = blk[..., :nz]
        det = np.einsum("nz,bnz->bn", d.Cd, z)
        st = s_i[:, 1:1 + N]
        lock = st == ST_LOCK
        return {"state": st.astype(float), "locked": (lock & (np.abs(det) < self.params.lock_pm)).astype(float),
                "detuning_est_pm": np.where(lock, d.det0_pm[None] + det, np.nan), "integrator": z[..., d.nc],
                "qp_iterations": s_i[:, 1 + N:1 + 2 * N].astype(float)}


register("mpc_c", CompiledMpc)
