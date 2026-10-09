"""Model-order reduction and common-pole Foster fitting for the thermal FEM.

Contract: ARCHITECTURE section 5.3 (``thermal_rom/1``) and 07 P1-2, P1-3. Never imported by runtime code.

Pipeline (all real arithmetic except the frequency samples):

1. ``rational_krylov_basis``: block rational Krylov subspace of the FEM pencil (K, M) for the input matrix B, at a set
   of real shifts sigma_j >= 0 (sigma = 0 matches the DC moments exactly). With K and M symmetric positive definite, the
   one-sided Galerkin projection keeps the reduced model symmetric, stable and passive (an RC network), and it
   matches ``n_moments`` moments of every input column at every shift (Grimme 1997, PRIMA).
2. ``galerkin_modal``: the reduced generalized eigenproblem K_r x = lambda M_r x gives real poles lambda_k > 0, so the
   reduced transfer matrix is an exact Foster sum  H(s) = sum_k r_k / (1 + s tau_k),  tau_k = 1/lambda_k.  This modal
   ROM has many poles (one per basis vector).
3. ``vector_fit_real``: vector fitting with a COMMON real pole set for every source/observer pair (Gustavsen and
   Semlyen 1999; the fast multi-response QR form of Deschrijver et al. 2008), with real poles enforced because a
   linear heat-conduction pencil has only real, negative poles.
4. ``varpro_polish`` (optional): variable-projection refinement of log(tau) on the combined frequency and time
   residual (Golub and Pereyra 1973), with the residues eliminated by linear least squares.
5. ``fit_residues``: the final residues of every pair on the shared poles, by linear least squares on frequency and
   time samples with the DC gain imposed exactly (sum_k r_k = H(0)). Residues may have either sign (non-collocated
   pairs carry diffusion delay).

Units: frequencies in Hz, times in s, gains in K/W; ``s`` arguments are complex angular frequencies (rad/s).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = [
    "log_shifts", "rational_krylov_basis", "galerkin_modal", "Foster", "vector_fit_real", "varpro_polish",
    "fit_residues", "time_errors", "freq_errors", "log_pole_grid", "fit_common_foster", "splu_factory",
    "separate_poles",
]


# ------------------------------------------------------------------------------------------------- Foster model
@dataclass
class Foster:
    """Foster sum with shared real poles: H(s) = sum_k res[..., k] / (1 + s*taus[k]).

    ``res`` may have any leading shape (for example (O, S, K) or (P, K)); the last axis runs over the poles."""
    taus: np.ndarray
    res: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.taus = np.asarray(self.taus, float)
        self.res = np.asarray(self.res, float)
        if self.res.shape[-1] != self.taus.size:
            raise ValueError("last axis of res must match taus")

    @property
    def dc(self) -> np.ndarray:
        return self.res.sum(axis=-1)

    def freq(self, f_hz) -> np.ndarray:
        """Complex frequency response at f (Hz); returns shape res.shape[:-1] + (F,)."""
        s = 2j * np.pi * np.atleast_1d(np.asarray(f_hz, float))
        phi = 1.0 / (1.0 + s[None, :] * self.taus[:, None])          # (K, F)
        return self.res @ phi

    def step(self, t_s) -> np.ndarray:
        """Unit-step response sum_k r_k (1 - exp(-t/tau_k)); shape res.shape[:-1] + (T,)."""
        t = np.atleast_1d(np.asarray(t_s, float))
        phi = -np.expm1(-t[None, :] / self.taus[:, None])             # (K, T)
        return self.res @ phi

    def t_frac(self, frac: float = 1 - np.exp(-1.0), t_lo: float = 1e-11, t_hi: float = 1e2) -> np.ndarray:
        """Time at which the step response first reaches ``frac`` of its DC value (default the 63 % point), per pair.

        Bisection on a log time axis after a dense scan; NaN where the response never reaches it."""
        t = np.logspace(np.log10(t_lo), np.log10(t_hi), 4001)
        y = self.step(t)
        dc = self.dc[..., None]
        r = y / np.where(dc == 0, np.nan, dc)
        flat = r.reshape(-1, t.size)
        out = np.full(flat.shape[0], np.nan)
        for i, row in enumerate(flat):
            idx = np.nonzero(row >= frac)[0]
            if idx.size == 0 or idx[0] == 0:
                continue
            j = idx[0]
            # log-linear interpolation between samples j-1 and j
            x0, x1 = np.log(t[j - 1]), np.log(t[j])
            y0, y1 = row[j - 1], row[j]
            out[i] = np.exp(x0 + (frac - y0) * (x1 - x0) / (y1 - y0))
        return out.reshape(self.res.shape[:-1])


# ------------------------------------------------------------------------------------------------- Krylov ROM
def log_shifts(f_lo_hz: float, f_hi_hz: float, per_decade: float, include_zero: bool = True) -> np.ndarray:
    """Real Krylov shifts sigma = 2*pi*f (rad/s), log-spaced from f_lo to f_hi, plus sigma = 0 (the DC moments)."""
    n = max(2, int(round(np.log10(f_hi_hz / f_lo_hz) * per_decade)) + 1)
    sig = 2 * np.pi * np.logspace(np.log10(f_lo_hz), np.log10(f_hi_hz), n)
    return np.concatenate([[0.0], sig]) if include_zero else sig


def splu_factory(A, permc_spec: str = "MMD_AT_PLUS_A"):
    """Default factorization: SuperLU with a symmetric fill-reducing ordering (the FEM pencils are symmetric);
    returns a solve(X) callable for X of shape (n,) or (n, m)."""
    lu = spla.splu(sp.csc_matrix(A), permc_spec=permc_spec)
    return lu.solve


class _Basis:
    """Growing orthonormal basis stored in one preallocated array (avoids O(r^2 n) copies)."""

    def __init__(self, n: int, cap: int):
        self.V = np.empty((n, cap))
        self.k = 0

    def append(self, X: np.ndarray, tol: float) -> int:
        kept = 0
        for j in range(X.shape[1]):
            x = np.array(X[:, j], float)
            nx = np.linalg.norm(x)
            if not np.isfinite(nx) or nx == 0:
                continue
            x /= nx
            W = self.V[:, :self.k]
            for _ in range(2):          # twice-repeated classical Gram-Schmidt
                if self.k:
                    x -= W @ (W.T @ x)
            nr = np.linalg.norm(x)
            if nr < tol:
                continue
            if self.k == self.V.shape[1]:
                self.V = np.concatenate([self.V, np.empty_like(self.V)], axis=1)
                W = self.V[:, :self.k]
            self.V[:, self.k] = x / nr
            self.k += 1
            kept += 1
        return kept


def rational_krylov_basis(K, M, B, shifts, n_moments: int = 2, factor=splu_factory, tol: float = 1e-9,
                          log=None) -> np.ndarray:
    """Orthonormal basis of the block rational Krylov space  span{(sM+K)^-1 B, [(sM+K)^-1 M]^m (sM+K)^-1 B}.

    ``factor(A)`` must return a solve callable for (n, m) right-hand sides (SuperLU by default; an AMG-preconditioned
    CG for large 3-D systems). Returns V (n, r) with V^T V = I."""
    B = np.asarray(B, float)
    if B.ndim == 1:
        B = B[:, None]
    basis = _Basis(B.shape[0], len(shifts) * n_moments * B.shape[1])
    for sig in shifts:
        A = (K + sig * M) if sig else K
        solve = factor(A)
        X = np.asarray(solve(B), float).reshape(B.shape)
        kept = basis.append(X, tol)
        for _ in range(n_moments - 1):
            X = np.asarray(solve(M @ X), float).reshape(B.shape)
            kept += basis.append(X, tol)
        del solve, A
        if log:
            log(f"  krylov shift {sig:.3e} rad/s: kept {kept}, basis {basis.k}")
    return np.ascontiguousarray(basis.V[:, :basis.k])


def galerkin_modal(K, M, B, C, V) -> Foster:
    """Galerkin projection of (K, M, B, C) on V, diagonalized: returns the modal Foster model with res (O, S, r).

    The reduced pencil is symmetrized before ``scipy.linalg.eigh`` (it is symmetric up to round-off)."""
    B = np.asarray(B, float)
    C = np.asarray(C.toarray() if sp.issparse(C) else C, float)
    if B.ndim == 1:
        B = B[:, None]
    Kr = V.T @ (K @ V)
    Mr = V.T @ (M @ V)
    Kr = 0.5 * (Kr + Kr.T)
    Mr = 0.5 * (Mr + Mr.T)
    lam, X = sla.eigh(Kr, Mr)
    if np.any(lam <= 0):
        raise RuntimeError(f"non-positive reduced eigenvalue {lam.min():.3e}: K must be positive definite")
    Bm = X.T @ (V.T @ B)               # (r, S)
    Cm = (C @ V) @ X                    # (O, r)
    res = Cm[:, None, :] * Bm.T[None, :, :] / lam[None, None, :]
    order = np.argsort(-1.0 / lam)      # slowest first
    return Foster(taus=1.0 / lam[order], res=res[..., order], meta={"method": "galerkin_rational_krylov",
                                                                       "order": int(lam.size)})


# ------------------------------------------------------------------------------------------------- fitting
def log_pole_grid(tau_min: float, tau_max: float, n: int) -> np.ndarray:
    """n log-spaced time constants from tau_max down to tau_min (slowest first)."""
    return np.logspace(np.log10(tau_max), np.log10(tau_min), n)


def vector_fit_real(f_hz, H, weights, taus0, n_iter: int = 30, tau_bounds=(1e-10, 1e3), tol: float = 1e-4,
                    log=None):
    """Common-pole vector fitting with real poles.

    f_hz (F,), H (P, F) complex responses, weights (P, F) positive (or (F,)), taus0 (K,) initial poles.
    Fits H_p(s) ~ sum_k c_pk / (s + a_k) with a_k = 1/tau_k shared by all p (strictly proper: thermal transfer
    functions vanish at infinite frequency). Each iteration solves the sigma-function least squares in the fast
    multi-response QR form, then relocates the poles to the zeros of sigma; complex zeros are replaced by their real
    parts (a heat-conduction pencil has real poles only) and the poles are clipped to ``tau_bounds``.
    Returns (taus, history) with taus sorted slowest first."""
    s = 2j * np.pi * np.asarray(f_hz, float)
    H = np.atleast_2d(np.asarray(H, complex))
    W = np.broadcast_to(np.asarray(weights, float), H.shape)
    a = 1.0 / np.sort(np.asarray(taus0, float))[::-1]
    K = a.size
    hist = []
    a_lo, a_hi = 1.0 / tau_bounds[1], 1.0 / tau_bounds[0]
    for it in range(n_iter):
        Phi = 1.0 / (s[:, None] + a[None, :])                       # (F, K)
        rows, rhs = [], []
        for p in range(H.shape[0]):
            w = W[p]
            A = np.hstack([Phi * w[:, None], -(H[p] * w)[:, None] * Phi])
            b = H[p] * w
            Ar = np.vstack([A.real, A.imag])
            br = np.concatenate([b.real, b.imag])
            Q, R = np.linalg.qr(Ar)
            qb = Q.T @ br
            rows.append(R[K:, K:])
            rhs.append(qb[K:])
        Rs = np.vstack(rows)
        rs = np.concatenate(rhs)
        ct, *_ = np.linalg.lstsq(Rs, rs, rcond=None)
        z = np.linalg.eigvals(np.diag(-a) - np.outer(np.ones(K), ct))
        a_new = np.abs(z.real)
        a_new = np.clip(a_new, a_lo, a_hi)
        a_new = np.sort(a_new)
        # separate coincident poles (keeps the basis well conditioned)
        for k in range(1, K):
            if a_new[k] <= a_new[k - 1] * 1.02:
                a_new[k] = a_new[k - 1] * 1.02
        change = float(np.max(np.abs(np.log(a_new) - np.log(np.sort(a)))))
        hist.append(change)
        if log:
            log(f"  VF iter {it}: max |d ln tau| = {change:.2e}")
        a = a_new
        if change < tol:
            break
    return np.sort(1.0 / a)[::-1], hist


def _design(taus, f_hz, t_s):
    s = 2j * np.pi * np.asarray(f_hz, float)
    Pf = 1.0 / (1.0 + s[:, None] * taus[None, :])                  # (F, K)
    Pt = -np.expm1(-np.asarray(t_s, float)[:, None] / taus[None, :])  # (T, K)
    return Pf, Pt


def fit_residues(taus, f_hz, Hf, wf, t_s, Yt, wt, dc, nonneg=None):
    """Residues on fixed shared poles, per response, with sum_k r_k = dc exactly.

    Hf (P, F) complex with weights wf (P, F); Yt (P, T) step responses with weights wt (P, T); dc (P,).
    Minimizes sum |wf (Phi_f r - Hf)|^2 + sum |wt (Phi_t r - Yt)|^2 subject to 1^T r = dc (null-space method).
    ``nonneg`` (P,) bool: responses whose residues must be >= 0 (driving-point pairs, whose exact residues are
    positive); these are solved with bounds and a heavily weighted DC row, then rescaled to the exact DC.
    Returns res (P, K) and the weighted RMS residual per response."""
    from scipy.optimize import lsq_linear

    taus = np.asarray(taus, float)
    K = taus.size
    Pf, Pt = _design(taus, f_hz, t_s)
    Hf = np.atleast_2d(Hf)
    Yt = np.atleast_2d(Yt)
    P = Hf.shape[0]
    wf = np.broadcast_to(wf, Hf.shape)
    wt = np.broadcast_to(wt, Yt.shape)
    nn = np.zeros(P, bool) if nonneg is None else np.asarray(nonneg, bool)
    N = np.vstack([np.eye(K - 1), -np.ones((1, K - 1))])            # 1^T N = 0
    eK = np.zeros(K)
    eK[-1] = 1.0
    res = np.zeros((P, K))
    rms = np.zeros(P)
    for p in range(P):
        A = np.vstack([(Pf * wf[p][:, None]).real, (Pf * wf[p][:, None]).imag, Pt * wt[p][:, None]])
        y = np.concatenate([(Hf[p] * wf[p]).real, (Hf[p] * wf[p]).imag, Yt[p] * wt[p]])
        if nn[p]:
            wd = 1e3 * np.sqrt(A.shape[0]) * np.abs(A).max() / max(abs(dc[p]), 1e-300)
            A2 = np.vstack([A, wd * np.ones((1, K))])
            y2 = np.concatenate([y, [wd * dc[p]]])
            r = np.clip(lsq_linear(A2, y2, bounds=(0.0, np.inf), method="bvls").x, 0.0, None)
            r = r * (dc[p] / r.sum()) if r.sum() > 0 else np.full(K, dc[p] / K)
            res[p] = r
        else:
            r0 = dc[p] * eK
            z, *_ = np.linalg.lstsq(A @ N, y - A @ r0, rcond=None)
            res[p] = r0 + N @ z
        rms[p] = np.sqrt(np.mean((A @ res[p] - y) ** 2))
    return res, rms


def varpro_polish(taus0, f_hz, Hf, wf, t_s, Yt, wt, dc, tau_bounds=(1e-10, 1e3), max_nfev: int = 60,
                  min_ratio: float = 1.5, nonneg=None, log=None):
    """Variable-projection refinement of the shared poles on the combined weighted residual (residues eliminated).

    The poles are parameterized as log tau_1 plus log-gaps g_k >= ln(min_ratio): adjacent time constants can never
    cluster, which keeps the residues of the fitted Foster sum from cancelling (a near-coincident pair can emulate a
    t exp(-t/tau) term with residues many times the DC gain). Returns (taus, info)."""
    from scipy.optimize import least_squares

    lg_min = np.log(min_ratio)

    def unpack(x):
        return np.exp(x[0] + np.concatenate([[0.0], np.cumsum(x[1:])]))

    def resid(x):
        taus = unpack(x)
        Pf, Pt = _design(taus, f_hz, t_s)
        res, _ = fit_residues(taus, f_hz, Hf, wf, t_s, Yt, wt, dc, nonneg=nonneg)
        mf = (res @ Pf.T - Hf) * wf
        mt = (res @ Pt.T - Yt) * wt
        return np.concatenate([mf.real.ravel(), mf.imag.ravel(), mt.ravel()])

    t0 = np.sort(np.asarray(taus0, float))
    t0 = np.sort(separate_poles(t0, min_ratio * 1.0001))
    x0 = np.concatenate([[np.log(t0[0])], np.maximum(np.diff(np.log(t0)), lg_min * 1.0001)])
    lb = np.concatenate([[np.log(tau_bounds[0])], np.full(t0.size - 1, lg_min)])
    ub = np.concatenate([[np.log(tau_bounds[1])], np.full(t0.size - 1, np.log(tau_bounds[1] / tau_bounds[0]))])
    x0 = np.clip(x0, lb + 1e-9, ub - 1e-9)
    r0 = resid(x0)
    sol = least_squares(resid, x0, bounds=(lb, ub), max_nfev=max_nfev, x_scale=1.0, diff_step=1e-4)
    info = {"cost0": float(0.5 * r0 @ r0), "cost": float(sol.cost), "nfev": int(sol.nfev), "status": int(sol.status),
            "min_ratio": min_ratio}
    if log:
        log(f"  varpro: cost {info['cost0']:.3e} -> {info['cost']:.3e} in {info['nfev']} evaluations")
    taus = unpack(sol.x) if info["cost"] <= info["cost0"] else unpack(x0)
    return np.sort(taus)[::-1], info


def separate_poles(taus, min_ratio: float = 1.15) -> np.ndarray:
    """Merge-free spacing guard: adjacent time constants are pushed apart to at least ``min_ratio`` (geometric),
    keeping the geometric centre of each close pair. Avoids near-coincident poles with large opposite residues."""
    t = np.sort(np.asarray(taus, float))[::-1].copy()
    for _ in range(50):
        moved = False
        for k in range(1, t.size):
            if t[k - 1] / t[k] < min_ratio:
                c = np.sqrt(t[k - 1] * t[k])
                t[k - 1], t[k] = c * np.sqrt(min_ratio), c / np.sqrt(min_ratio)
                moved = True
        if not moved:
            break
        t = np.sort(t)[::-1]
    return t


def time_errors(model: Foster, t_s, Yref, dc_ref, t_lo: float = 1e-8, t_hi: float = 1e-2):
    """max_t |y_model - y_ref| / |dc_ref| over t_lo <= t <= t_hi, per pair (leading shape of model.res)."""
    t = np.asarray(t_s, float)
    m = (t >= t_lo * (1 - 1e-9)) & (t <= t_hi * (1 + 1e-9))
    y = model.step(t[m])
    e = np.max(np.abs(y - np.asarray(Yref)[..., m]), axis=-1) / np.abs(dc_ref)
    return e


def freq_errors(model: Foster, f_hz, Href, band_floor: float | None = None, dc_ref=None):
    """Magnitude (dB) and phase (deg) error of the model against Href at f, per pair.

    If ``band_floor`` is given, only frequencies where |Href| >= band_floor*|dc_ref| count (cross terms: their
    response vanishes into numerical noise long before 1 MHz). Returns (max |dB|, max |deg|, f_max_used)."""
    Hm = model.freq(f_hz)
    Href = np.asarray(Href)
    ratio = Hm / Href
    db = 20 * np.log10(np.abs(ratio))
    deg = np.degrees(np.angle(ratio))
    if band_floor is not None:
        mask = np.abs(Href) >= band_floor * np.abs(np.asarray(dc_ref))[..., None]
    else:
        mask = np.ones_like(db, bool)
    db = np.where(mask, np.abs(db), 0.0)
    deg = np.where(mask, np.abs(deg), 0.0)
    f = np.broadcast_to(np.asarray(f_hz, float), mask.shape)
    fmax = np.max(np.where(mask, f, 0.0), axis=-1)
    return db.max(axis=-1), deg.max(axis=-1), fmax


def fit_common_foster(f_hz, Hf, t_s, Yt, dc, n_poles: int, tau_min: float, tau_max: float,
                      floor_vf: float = 1e-2, floor_fit: float = 1e-3, time_weight: float = 1.0,
                      polish_idx=None, polish: bool = True, vf_iter: int = 30, nonneg=None, min_ratio: float = 1.5,
                      log=None):
    """Common real-pole Foster fit of P responses (the ARCHITECTURE 5.3 form), returns (taus, res (P, K), info).

    Hf (P, F) complex frequency data, Yt (P, T) step data, dc (P,) exact DC gains (imposed). Each response is
    weighted relative to itself: in frequency by 1/max(|H|, floor*|dc|) (relative accuracy down to the floor), in time
    by time_weight/|dc| (absolute error relative to the steady state). ``polish_idx`` selects the responses used by
    the variable-projection polish (default: all)."""
    dc = np.asarray(dc, float)
    Hf = np.asarray(Hf, complex)
    Yt = np.asarray(Yt, float)
    adc = np.abs(dc)[:, None]
    w_vf = 1.0 / np.maximum(np.abs(Hf), floor_vf * adc)
    w_fit = 1.0 / np.maximum(np.abs(Hf), floor_fit * adc)
    w_t = np.broadcast_to(time_weight / adc, Yt.shape) / np.sqrt(max(Yt.shape[1], 1) / max(Hf.shape[1], 1))
    taus0 = log_pole_grid(tau_min, tau_max, n_poles)
    taus_vf, hist = vector_fit_real(f_hz, Hf, w_vf, taus0, n_iter=vf_iter,
                                    tau_bounds=(tau_min / 10, tau_max * 10), log=log)
    info = {"vf_iterations": len(hist), "vf_last_change": hist[-1] if hist else None,
            "taus_vf_s": taus_vf.tolist(), "n_poles": int(n_poles)}
    taus = taus_vf
    if polish:
        idx = np.arange(Hf.shape[0]) if polish_idx is None else np.asarray(polish_idx)
        nn_idx = None if nonneg is None else np.asarray(nonneg, bool)[idx]
        taus, pinfo = varpro_polish(taus, f_hz, Hf[idx], w_fit[idx], t_s, Yt[idx], w_t[idx], dc[idx],
                                    tau_bounds=(tau_min / 10, tau_max * 10), min_ratio=min_ratio, nonneg=nn_idx,
                                    log=log)
        info["varpro"] = pinfo
    else:
        taus = separate_poles(taus, min_ratio=min_ratio)
    res, rms = fit_residues(taus, f_hz, Hf, w_fit, t_s, Yt, w_t, dc, nonneg=nonneg)
    info["fit_rms_max"] = float(rms.max())
    info["min_pole_ratio"] = float(np.min(np.sort(taus)[1:] / np.sort(taus)[:-1])) if taus.size > 1 else None
    info["max_abs_residue_over_dc"] = float(np.max(np.abs(res).max(axis=1) / np.abs(dc)))
    return taus, res, info
