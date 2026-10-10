"""Implicit per-ring optical self-heating solve with branch continuation (07 P0-1 step 2, D11).

Contract: docs/research/07_design_critique.md P0-1 item 2: process the rings in bus order (ring k's optical input is
fixed by rings 1..k-1); for ring k solve, with the thermal ROM advanced exactly over the substep h,

    T_k = T_k^other + sum_j [ a_j x_kj + (1 - a_j) r_j^opt P_abs,k(T_k) ],      a_j = exp(-h/tau_j)

where T_k^other holds the base temperature and every heater, junction, driver, aggressor and neighbour contribution,
already advanced. This is F(T) = T - T_other - g P_abs(T) = 0 with g = sum_j (1 - a_j) r_j^opt (K/W), solved by a
bracketed Newton/secant iteration of at most ``max_iter`` (10) evaluations that starts from the previous root (branch
continuation). The same-ring path never uses the previous substep's P_abs (that explicit coupling has the multiplier
a - (1 - a) g and oscillates with period 2 at dt = 1 us, 07 P0-1 "Numerics"). The cross-ring optical heating
(LAG_OPT_CROSS_RING lags) uses the previous substep's P_abs; mrm.plant.thermal.assert_cross_lag_gain checks that its
per-substep gain is below 0.05.

Branch continuation. Because P_abs >= 0 is bounded by p_max, every root lies in [T_other, T_other + g p_max]. The
search starts at the previous root T_prev and only moves in the direction -sign F(T_prev), which is the direction the
continuous dynamics move (tau dT/dt = -F for one pole). It therefore returns the first root met from T_prev, i.e. the
branch the ring is on; when that branch ends at a fold, the root on the other branch. With a curvature bound
d2p_max >= max|d2P_abs/dT2| (W/K^2), each Newton step is shortened to the distance within which no root can lie
(see solve_ring_temperature), so no root is ever skipped and convergence stays quadratic at a simple root. If
``max_iter`` is reached, the last point (still before the root) is returned with status 1: the ring then lags its
root, as the real ring does while it crosses to the other branch, and the next substep continues from there. If
P_abs exceeds the caller's p_max, the search ends at the bracket end without a sign change and returns status 2 (a
caller error, counted as a failure; it is never reported as converged).

The absorbed power is supplied by the caller in one of two ways:
- a numba first-class function ``pabs_fn(t_k, k, args_f, args_i) -> (P_abs_W, dP_abs/dT_W_per_K)`` evaluated at
  the mode-weighted ring temperature t_k (absolute, K), ring index k, with the caller's float64 and int64 parameter
  rows (the optics module's exact ring, summed over lines, directions and orders), plus ``commit_fn(t_k, k, args_f,
  args_i)`` called once ring k is solved (for example to propagate the bus field to ring k+1); or
- precomputed arguments for the built-in ``pabs_lorentzian``: per ring a resonance law
  lambda_r(T) = lambda_ref + s0 [(T - T_ref) + beta (T - T_ref)^2 / 2] (07 P0-3 R1, order 2) and per line a
  Lorentzian P_abs = p_peak / (1 + ((lambda_r - lambda_line)/HWHM)^2) (TCMT limit, 03 section 6.4), packed with
  ``pack_lorentzian_args``.

Reset (07 P0-1 item 6): the first substep's t_prev chooses the branch. ``branch_low``: t_prev = -inf (the root met
coming up from the cold side); ``branch_high``: t_prev = +inf (the hot, self-locked branch); ``laser_ramp``: start
with t_prev at the cold ring temperature and ramp the line powers (the continuation follows the physical history).

All kernels are numba @njit and run under NUMBA_DISABLE_JIT=1. Units: K, W, m, s.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

from mrm.plant.thermal import (LAG_EXOGENOUS, LAG_OPT_CROSS_RING, LAG_OPT_SELF, ThermalSystem,
                               _row)

__all__ = [
    "solve_ring_temperature", "implicit_step_b", "implicit_step", "implicit_run_b",
    "pabs_lorentzian", "commit_none", "pack_lorentzian_args", "lorentzian_bounds",
    "LORENTZ_MAX_SLOPE", "BISTABILITY_FACTOR", "lorentzian_folds", "steady_detuning", "bistability_ratio",
    "SelfHeatWork", "STATUS_CONVERGED", "STATUS_MAX_ITER", "STATUS_BOUND",
]

# max over u of 2|u|/(1 + u^2)^2 = 3 sqrt(3)/8 at |u| = 1/sqrt(3): the steepest Lorentzian slope per HWHM (exact).
LORENTZ_MAX_SLOPE = 3.0 * math.sqrt(3.0) / 8.0
# Bistability threshold G P_abs,peak >= (8/(3 sqrt 3)) HWHM = (4 sqrt(3)/9) FWHM ~ 0.770 FWHM (03 section 6.4; exact).
BISTABILITY_FACTOR = 8.0 / (3.0 * math.sqrt(3.0))

STATUS_CONVERGED = 0
STATUS_MAX_ITER = 1
STATUS_BOUND = 2         # the search reached the end of the bracket without a sign change: p_max does not bound P_abs


# =============================================================================================================
# Scalar solve
# =============================================================================================================
@njit
def solve_ring_temperature(pabs_fn, k, args_f, args_i, t_other, g, t_start, p_max, d2p_max, max_iter, tol_k):
    """Solve F(T) = T - t_other - g P_abs(T) = 0 for ring k from the previous root t_start (07 P0-1, D11).

    pabs_fn(t, k, args_f, args_i) -> (P W, dP/dT W/K). t_other K; g K/W (normally > 0; a negative g from a
    mixed-sign fit is handled with the mirrored bracket; g = 0 returns t_other); t_start K (the previous root; -inf
    selects the lowest root, the cold branch, and +inf the highest root, the hot branch); p_max W, an upper bound of
    P_abs (0 <= P_abs <= p_max), so every root lies between t_other and t_other + g p_max (the bracket); d2p_max
    W/K^2, an upper bound of |d2P_abs/dT2|; max_iter evaluations (<= 10 per 07); tol_k K on |F| and on the bracket
    width.
    Returns (T K, P_abs W, dP_abs/dT W/K, evaluations used, status): 0 converged; 1 max_iter reached (T lies before
    the root); 2 the bracket end was reached with F still of the start sign, i.e. p_max does not bound P_abs (the
    caller's bound is wrong; T is that bracket end and |F(T)| > tol_k).

    Step (curvature-certified Newton). From the current point t_n, moving toward the root (direction -sign F),
    |F| shrinks at most like |F(t_n)| - F'(t_n) d - K2 d^2/2 with K2 = g d2p_max, so no root lies closer than
        d_min = 2|F| / (F' + sqrt(F'^2 + 2 K2 |F|)),
    which tends to the Newton step |F|/F' as F -> 0 (quadratic convergence at a simple root). Every iterate therefore
    stays on the same side of the first root met from t_start: the iteration follows the branch the ring is on and,
    after a fold, moves to the next branch in the direction the dynamics move. With d2p_max <= 0 the step is the plain
    Newton step, safeguarded by the bracket (secant or bisection); it can then skip a root pair near a fold.
    """
    if g == 0.0 or g != g:
        p, d = pabs_fn(t_other, k, args_f, args_i)
        return t_other, p, d, 0, STATUS_CONVERGED
    # F(lo) = (lo - t_other) - g P(lo) <= 0 and F(hi) >= 0 for either sign of g when 0 <= P <= p_max
    lo = min(t_other, t_other + g * p_max)
    hi = max(t_other, t_other + g * p_max)
    t0 = t_start
    if t0 != t0:                              # NaN: start at the cold end; +-inf select the high/low end (below)
        t0 = t_other
    if t0 < lo:
        t0 = lo
    if t0 > hi:
        t0 = hi
    p0, d0 = pabs_fn(t0, k, args_f, args_i)
    f0 = t0 - t_other - g * p0
    if abs(f0) <= tol_k:
        return t0, p0, d0, 0, STATUS_CONVERGED
    sgn = 1.0 if f0 < 0.0 else -1.0          # the root lies in the direction sgn from t0
    near, f_near, p_near, d_near = t0, f0, p0, d0
    far = hi if sgn > 0.0 else lo
    f_far, p_far, d_far = 0.0, 0.0, 0.0
    have_far = False
    k2 = abs(g) * d2p_max if d2p_max > 0.0 else 0.0
    used = 0
    for it in range(1, max_iter + 1):
        if (far - near) * sgn <= tol_k:
            break
        used = it
        fp = 1.0 - g * d_near                # F'(near); |F| decreases at this rate along sgn
        af = abs(f_near)
        cand = math.nan
        if k2 > 0.0 and math.isfinite(fp):
            step = 2.0 * af / (fp + math.sqrt(fp * fp + 2.0 * k2 * af))
            c = near + sgn * step
            if (far - c) * sgn > 0.0:
                cand = c
            elif not have_far:
                cand = far                    # the bracket end: F(far) has the root's sign by the p_max bound
            # else: the step reached a sign change already found (inexact dP/dT or d2p_max): secant below
        elif math.isfinite(fp) and fp > 0.0:
            c = near + sgn * af / fp
            if (c - near) * sgn > 0.0 and (far - c) * sgn > 0.0:
                cand = c
        if math.isnan(cand):
            if have_far and f_far != f_near:
                c = near - f_near * (far - near) / (f_far - f_near)
                if (c - near) * sgn > 0.0 and (far - c) * sgn > 0.0:
                    cand = c
            if math.isnan(cand):
                cand = 0.5 * (near + far)
        pc, dc = pabs_fn(cand, k, args_f, args_i)
        fc = cand - t_other - g * pc
        if abs(fc) <= tol_k:
            return cand, pc, dc, it, STATUS_CONVERGED
        if fc * f_near > 0.0:
            near, f_near, p_near, d_near = cand, fc, pc, dc
        else:
            far, f_far, p_far, d_far = cand, fc, pc, dc
            have_far = True
    if (far - near) * sgn <= tol_k:
        if have_far:
            if abs(f_far) < abs(f_near):
                return far, p_far, d_far, used, STATUS_CONVERGED
            return near, p_near, d_near, used, STATUS_CONVERGED
        # no sign change met up to the bracket end: with a valid p_max, F(bracket end) has the root's sign or is 0
        # (then |F| <= tol_k returned above); here P_abs exceeded p_max
        return near, p_near, d_near, used, STATUS_BOUND
    return near, p_near, d_near, used, STATUS_MAX_ITER


@njit
def commit_none(t_k, k, args_f, args_i):
    """No-op commit function (rings that do not change each other's optical input)."""
    return None


# =============================================================================================================
# One implicit substep of the whole bank
# =============================================================================================================
@njit
def implicit_step_b(x, a, c0, c1, lag_src, lag_obs, lag_kind, opt_lags, obs_ptr, src_ring, ring_obs, bus_order,
                    u0, u1, src_scale, t_base, p_prev, t_prev,
                    pabs_fn, commit_fn, args_f, args_i, p_max, d2p_max, max_iter, tol_k, n_outer,
                    g_work, t_obs, p_new, t_ring, n_iter):
    """Advance one world's thermal state by one substep with implicit optical self-heating (07 P0-1). In place.

    Thermal arrays (ThermalSystem.kernel_arrays() and ThermalSystem.coefficients(h)): x, a, c0, c1 (M,); lag_src,
    lag_obs, lag_kind (M,); opt_lags (M_opt,) the LAG_OPT_SELF and LAG_OPT_OTHER lags; obs_ptr (O+1,); src_ring (S,);
    ring_obs (N,). bus_order (N,): ring indices in the order light reaches them.
    u0, u1 (S,) W: exogenous source powers at the start and the end of the substep (held inputs: u1 = u0); entries of
    waveguide-absorption sources are ignored. src_scale (S,): residue multipliers (1.0). t_base K: base (package)
    temperature at the end of the substep. p_prev (N,) W: the previous substep's absorbed power (drives the
    LAG_OPT_CROSS_RING lags). t_prev (N,) K: the previous roots (branch continuation starts there).
    pabs_fn, commit_fn, args_f, args_i: see the module docstring. p_max (N,) W and d2p_max (N,) W/K^2: bounds of P_abs
    and |d2P_abs/dT2| for each ring (solve_ring_temperature). n_outer >= 1: bank sweeps. 1 = the cross-ring optical
    heating lags one substep (07 P0-1; valid while thermal.cross_optical_lag_gain < 0.05, e.g. 1 us substeps); n > 1
    repeats the bus-order sweep with the cross-ring lags driven by the newest P_abs (Gauss-Seidel), which makes the
    coupling implicit for long steps (07 P0-10 multi-tick); the sweep stops early when P_abs no longer changes
    (1e-12 relative). Scratch g_work (O,).
    Outputs: t_obs (O,) K end-of-step observer temperatures (absolute); p_new (N,) W the absorbed power the thermal
    state received (= P_abs(t_ring) when converged); t_ring (N,) K the solved roots; n_iter (N,) evaluations.
    Returns the number of rings whose solve did not converge (status 1, max_iter, or 2, P_abs above p_max).

    Lag inputs: LAG_EXOGENOUS u0/u1 (exact FOH); LAG_OPT_SELF P_abs,k(T_k) held over the step (implicit);
    LAG_OPT_CROSS_RING p_prev (lagged one substep, 07 P0-1); LAG_OPT_OTHER (heater resistor, PD, sensors) the new
    P_abs. The ring observer of ring k ends equal to t_ring[k] (to rounding).
    Cost: one pass over all lags, the ring solves, then one pass over the optical lags only.
    """
    nr = ring_obs.shape[0]
    if bus_order.shape[0] != nr:
        raise ValueError("implicit_step: bus_order must list every ring once")
    for k in range(nr):
        n_iter[k] = 0
    for q in range(nr):                    # bus_order must be a permutation of 0..N-1 (n_iter as the marker)
        k = bus_order[q]
        if k < 0 or k >= nr or n_iter[k] != 0:
            raise ValueError("implicit_step: bus_order must be a permutation of the ring indices")
        if ring_obs[k] < 0:
            raise ValueError("implicit_step needs the ring_mode_weighted observer of every ring")
        n_iter[k] = 1
    n_obs = obs_ptr.shape[0] - 1
    # two passes (the same values and, per observer, the same summation order as one pass over the observers; the
    # lag update without the observer sums runs ~20 % faster): every lag with its known input, then the sums
    for m in range(lag_kind.shape[0]):
        s = lag_src[m]
        kd = lag_kind[m]
        xm = a[m] * x[m]
        if kd == LAG_EXOGENOUS:
            xm += src_scale[s] * (c0[m] * u0[s] + c1[m] * u1[s])
        elif kd == LAG_OPT_CROSS_RING:
            xm += src_scale[s] * (c0[m] + c1[m]) * p_prev[src_ring[s]]
        x[m] = xm
    for o in range(n_obs):
        e = t_base
        g = 0.0
        for m in range(obs_ptr[o], obs_ptr[o + 1]):
            if lag_kind[m] == LAG_OPT_SELF:
                g += src_scale[lag_src[m]] * (c0[m] + c1[m])
            e += x[m]
        t_obs[o] = e                       # everything except the new optical inputs
        g_work[o] = g
    n_fail = 0
    p_cross = np.empty(nr)                 # P_abs currently driving the LAG_OPT_CROSS_RING lags
    for k in range(nr):
        p_cross[k] = p_prev[k]
        t_ring[k] = t_prev[k]
        n_iter[k] = 0
    for sweep in range(max(n_outer, 1)):
        n_fail = 0
        for q in range(nr):
            k = bus_order[q]
            o = ring_obs[k]
            if o < 0:
                raise ValueError("implicit_step needs the ring_mode_weighted observer of every ring")
            t, p, d, it, st = solve_ring_temperature(pabs_fn, k, args_f, args_i, t_obs[o], g_work[o], t_ring[k],
                                                     p_max[k], d2p_max[k], max_iter, tol_k)
            t_ring[k] = t
            p_new[k] = p
            n_iter[k] += it
            if st != STATUS_CONVERGED:
                n_fail += 1
                # not converged (a jump between branches longer than max_iter steps, or a p_max that does not bound
                # P_abs): the thermal state receives the power that puts the ring exactly at the returned point,
                # which lies before the root, so the ring lags its root and never overshoots it; the next substep
                # continues the crossing.
                gk = g_work[o]
                if gk != 0.0:
                    p_new[k] = (t - t_obs[o]) / gk
            commit_fn(t, k, args_f, args_i)
        if sweep + 1 >= n_outer:
            break
        # Gauss-Seidel: drive the cross-ring lags with the newest P_abs, then sweep again
        change = 0.0
        scale = 0.0
        for k in range(nr):
            change = max(change, abs(p_new[k] - p_cross[k]))
            scale = max(scale, abs(p_new[k]))
        if change <= 1e-12 * scale:
            break
        for m in range(lag_kind.shape[0]):
            if lag_kind[m] == LAG_OPT_CROSS_RING:
                s = lag_src[m]
                j = src_ring[s]
                dx = src_scale[s] * (c0[m] + c1[m]) * (p_new[j] - p_cross[j])
                x[m] += dx
                t_obs[lag_obs[m]] += dx
        for k in range(nr):
            p_cross[k] = p_new[k]
    for i in range(opt_lags.shape[0]):
        m = opt_lags[i]
        s = lag_src[m]
        dx = src_scale[s] * (c0[m] + c1[m]) * p_new[src_ring[s]]
        x[m] += dx
        t_obs[lag_obs[m]] += dx
    return n_fail


@njit
def implicit_step(x, a, c0, c1, lag_src, lag_obs, lag_kind, opt_lags, obs_ptr, src_ring, ring_obs, bus_order,
                  u0, u1, src_scale, t_base, p_prev, t_prev,
                  pabs_fn, commit_fn, args_f, args_i, p_max, d2p_max, max_iter, tol_k, n_outer,
                  g_work, t_obs, p_new, t_ring, n_iter):
    """Batched implicit_step_b: x (B, M); a, c0, c1 (B|1, M); u0, u1, src_scale (B|1, S); t_base (B|1,);
    p_prev, t_prev (B, N); args_f (B|1, nf); args_i (B|1, ni); p_max, d2p_max (B|1, N); g_work, t_obs (B, O);
    p_new, t_ring (B, N) float64; n_iter (B, N) int64. Returns the total number of rings that did not converge."""
    B = x.shape[0]
    for n in (a.shape[0], c0.shape[0], c1.shape[0], u0.shape[0], u1.shape[0], src_scale.shape[0], t_base.shape[0],
              args_f.shape[0], args_i.shape[0], p_max.shape[0], d2p_max.shape[0]):
        if n != 1 and n != B:
            raise ValueError("implicit_step: every per-world input must have 1 or B rows")
    if c0.shape[0] != a.shape[0] or c1.shape[0] != a.shape[0]:
        raise ValueError("implicit_step: a, c0, c1 must have the same rows")
    for n in (p_prev.shape[0], t_prev.shape[0], g_work.shape[0], t_obs.shape[0], p_new.shape[0], t_ring.shape[0],
              n_iter.shape[0]):
        if n != B:
            raise ValueError("implicit_step: p_prev, t_prev and the work/output arrays must have B rows")
    n_fail = 0
    for b in range(x.shape[0]):
        ra = _row(b, a.shape[0])
        n_fail += implicit_step_b(
            x[b], a[ra], c0[ra], c1[ra], lag_src, lag_obs, lag_kind, opt_lags, obs_ptr, src_ring, ring_obs, bus_order,
            u0[_row(b, u0.shape[0])], u1[_row(b, u1.shape[0])], src_scale[_row(b, src_scale.shape[0])],
            t_base[_row(b, t_base.shape[0])], p_prev[b], t_prev[b], pabs_fn, commit_fn,
            args_f[_row(b, args_f.shape[0])], args_i[_row(b, args_i.shape[0])],
            p_max[_row(b, p_max.shape[0])], d2p_max[_row(b, d2p_max.shape[0])], max_iter, tol_k, n_outer,
            g_work[b], t_obs[b], p_new[b], t_ring[b], n_iter[b])
    return n_fail


@njit
def implicit_run_b(x, a, c0, c1, lag_src, lag_obs, lag_kind, opt_lags, obs_ptr, src_ring, ring_obs, bus_order,
                   u_steps, t_base_steps, src_scale, p_prev, t_prev,
                   pabs_fn, commit_fn, args_f, args_i, p_max, d2p_max, max_iter, tol_k, n_outer,
                   t_ring_hist, p_hist, t_obs_hist):
    """Run n held-input substeps of one world (open loop): u_steps (n, S) W, t_base_steps (n,) K.

    Updates x, p_prev, t_prev in place; fills t_ring_hist, p_hist (n, N) and t_obs_hist (n, O) with end-of-step
    values. Returns the number of (ring, step) solves that hit max_iter. For tests, replicas and quasi-static sweeps.
    """
    n = u_steps.shape[0]
    n_obs = obs_ptr.shape[0] - 1
    nr = ring_obs.shape[0]
    g_work = np.zeros(n_obs)
    p_new = np.zeros(nr)
    t_ring = np.zeros(nr)
    n_iter = np.zeros(nr, np.int64)
    fails = 0
    for i in range(n):
        fails += implicit_step_b(x, a, c0, c1, lag_src, lag_obs, lag_kind, opt_lags, obs_ptr, src_ring, ring_obs,
                                 bus_order, u_steps[i], u_steps[i], src_scale, t_base_steps[i], p_prev, t_prev,
                                 pabs_fn, commit_fn, args_f, args_i, p_max, d2p_max, max_iter, tol_k, n_outer,
                                 g_work, t_obs_hist[i], p_new, t_ring, n_iter)
        for k in range(nr):
            p_prev[k] = p_new[k]
            t_prev[k] = t_ring[k]
            t_ring_hist[i, k] = t_ring[k]
            p_hist[i, k] = p_new[k]
    return fails


# =============================================================================================================
# Built-in Lorentzian absorption (precomputed arguments)
# =============================================================================================================
@njit
def pabs_lorentzian(t, k, args_f, args_i):
    """Absorbed power of ring k as a sum of Lorentzian lines, and its temperature derivative (03 section 6.4).

    args_i = [N, off_0, ..., off_{N-1}]; args_f[off_k:] = [lambda_ref m, T_ref K, s0 m/K, beta 1/K, L,
    (lambda_line m, HWHM m, p_peak W) x L]. lambda_r(T) = lambda_ref + s0 [dT + beta dT^2/2], dT = T - T_ref
    (07 P0-3 R1); P = sum_l p_l / (1 + u_l^2), u_l = (lambda_r - lambda_l)/HWHM_l;
    dP/dT = -sum_l p_l 2 u_l / (1 + u_l^2)^2 / HWHM_l * s0 (1 + beta dT).
    """
    off = args_i[1 + k]
    lam_ref = args_f[off]
    t_ref = args_f[off + 1]
    s0 = args_f[off + 2]
    beta = args_f[off + 3]
    nl = int(args_f[off + 4])
    dt = t - t_ref
    lam_r = lam_ref + s0 * (dt + 0.5 * beta * dt * dt)
    dlam = s0 * (1.0 + beta * dt)
    p = 0.0
    dp = 0.0
    for l in range(nl):
        j = off + 5 + 3 * l
        hw = args_f[j + 1]
        u = (lam_r - args_f[j]) / hw
        den = 1.0 + u * u
        pk = args_f[j + 2]
        p += pk / den
        dp -= pk * 2.0 * u / (den * den) / hw * dlam
    return p, dp


def pack_lorentzian_args(lam_ref_m, t_ref_k, s0_m_per_k, beta_per_k, lam_line_m, hwhm_m, p_peak_w):
    """Pack pabs_lorentzian parameters: per ring (B|1, N) arrays and per (ring, line) (B|1, N, L) arrays.

    Returns (args_f (B, nf) float64, args_i (B, 1+N) int64). Every ring has the same L (use p_peak 0 to pad).
    """
    lp = np.asarray(p_peak_w, np.float64)
    if lp.ndim == 1:
        lp = lp[None, :, None]
    elif lp.ndim == 2:
        lp = lp[None]
    B0, N, L = lp.shape
    per_ring = [np.asarray(v, np.float64) for v in (lam_ref_m, t_ref_k, s0_m_per_k, beta_per_k)]
    per_line = [np.asarray(v, np.float64) for v in (lam_line_m, hwhm_m)]
    # batch rows: only the leading axis of a (B, N) per-ring or (B, N, L) per-line array counts (np.atleast_3d
    # would turn an (N, L) array into (N, L, 1) and take N for B)
    B = max([B0] + [v.shape[0] for v in per_ring if v.ndim == 2] + [v.shape[0] for v in per_line if v.ndim == 3])

    def r2(v):
        return np.broadcast_to(np.atleast_2d(v) if v.ndim < 2 else v, (B, N))

    def r3(v):
        v = v if v.ndim == 3 else (v[None] if v.ndim == 2 else np.reshape(v, (1, -1, 1) if v.ndim == 1 else (1, 1, 1)))
        return np.broadcast_to(v, (B, N, L))

    lam_ref, t_ref, s0, beta = (r2(v) for v in per_ring)
    lam_l, hw = (r3(v) for v in per_line)
    pk = np.broadcast_to(lp, (B, N, L))
    if np.any(hw <= 0):
        raise ValueError("HWHM must be > 0")
    block = 5 + 3 * L
    args_f = np.zeros((B, N * block))
    args_i = np.zeros((B, 1 + N), np.int64)
    args_i[:, 0] = N
    for k in range(N):
        off = k * block
        args_i[:, 1 + k] = off
        args_f[:, off] = lam_ref[:, k]
        args_f[:, off + 1] = t_ref[:, k]
        args_f[:, off + 2] = s0[:, k]
        args_f[:, off + 3] = beta[:, k]
        args_f[:, off + 4] = L
        for l in range(L):
            j = off + 5 + 3 * l
            args_f[:, j] = lam_l[:, k, l]
            args_f[:, j + 1] = hw[:, k, l]
            args_f[:, j + 2] = pk[:, k, l]
    return args_f, args_i


def lorentzian_bounds(p_peak_w, hwhm_m, s0_m_per_k, beta_per_k=0.0, dt_max_k=200.0):
    """Bounds for pabs_lorentzian: (p_max W, dpdt_max W/K, d2p_max W/K^2), each (..., N).

    With u = (lambda_r - lambda_l)/HWHM, P = p/(1 + u^2): max|dP/du| = (3 sqrt 3/8) p and max|d2P/du2| = 2 p (exact,
    at u = 1/sqrt 3 and u = 0). With |dlambda_r/dT| <= s_max = |s0| (1 + |beta| dt_max_k) and d2lambda_r/dT2 = s0 beta:
        p_max = sum_l p_l,  dpdt_max = sum_l (3 sqrt 3/8) p_l s_max / HWHM_l,
        d2p_max = sum_l [2 p_l s_max^2 / HWHM_l^2 + (3 sqrt 3/8) p_l |s0 beta| / HWHM_l].
    p_peak_w, hwhm_m (..., N, L); s0_m_per_k, beta_per_k broadcastable to (..., N); dt_max_k the largest |T - T_ref|
    the ring can reach (K; 200 K covers 0..105 C ambient plus heater and self-heating, assumed margin).
    dpdt_max feeds thermal.cross_optical_lag_gain; d2p_max feeds solve_ring_temperature.
    """
    p = np.asarray(p_peak_w, np.float64)
    hw = np.asarray(hwhm_m, np.float64)
    s0 = np.abs(np.asarray(s0_m_per_k, np.float64))
    bt = np.abs(np.asarray(beta_per_k, np.float64))
    smax = s0 * (1.0 + bt * float(dt_max_k))
    pmax = p.sum(axis=-1)
    dmax = (LORENTZ_MAX_SLOPE * p / hw).sum(axis=-1) * smax
    d2max = (2.0 * p / hw**2).sum(axis=-1) * smax**2 + (LORENTZ_MAX_SLOPE * p / hw).sum(axis=-1) * s0 * bt
    return pmax, dmax, d2max


# =============================================================================================================
# Analytic results of the single-line problem (03 section 6.4; verification targets of 07 P0-1)
# =============================================================================================================
def lorentzian_folds(A: float):
    """Fold points of y = y_o + A/(1 + y^2) (normalized: y = (lambda_r - lambda_L)/HWHM, y_o the detuning without
    self-heating, A = G P_abs,peak / HWHM with G in m/W). Folds solve (1 + y^2)^2 + 2 A y = 0 (dy_o/dy = 0).

    Returns (y_a, y_b, yo_a, yo_b) with y_a < y_b < 0 and yo_a > yo_b: an increasing y_o (heater up, resonance
    moving red) jumps at yo_a; a decreasing y_o jumps at yo_b. None if A <= 8/(3 sqrt 3) (no bistability).
    """
    if A <= BISTABILITY_FACTOR:
        return None
    r = np.roots([1.0, 0.0, 2.0, 2.0 * A, 1.0])
    y = np.sort(r[np.abs(r.imag) < 1e-9 * max(1.0, A)].real)
    if y.size < 2:
        return None
    ya, yb = float(y[0]), float(y[-1])
    yo = lambda v: v - A / (1.0 + v * v)
    return ya, yb, yo(ya), yo(yb)


def steady_detuning(y_o: float, A: float):
    """All steady normalized detunings y for a cold detuning y_o: real roots of (y - y_o)(1 + y^2) = A, sorted."""
    r = np.roots([1.0, -y_o, 1.0, -(y_o + A)])
    return np.sort(r[np.abs(r.imag) < 1e-7 * max(1.0, abs(A), abs(y_o))].real)


def bistability_ratio(g_sh_m_per_w, p_abs_peak_w, fwhm_m):
    """07 P0-1 item 5: G_sh P_abs,peak / (0.770 FWHM); bistable when > 1 (0.770 = 4 sqrt(3)/9, exact)."""
    return np.asarray(g_sh_m_per_w) * np.asarray(p_abs_peak_w) / (0.5 * BISTABILITY_FACTOR * np.asarray(fwhm_m))


class SelfHeatWork:
    """Scratch and state arrays for implicit_step over a ThermalSystem (Python convenience for the plant owner).

    Holds x (B, M) lag states, p_prev/t_prev (B, N), and the work arrays; ``step(...)`` calls implicit_step.
    """

    def __init__(self, system: ThermalSystem, batch: int | None = None, t_init_k=None):
        B = system.batch if batch is None else int(batch)
        self.system = system
        self.x = system.zero_state(B)
        n, O = system.n_rings, system.n_obs
        self.p_prev = np.zeros((B, n))
        self.t_prev = np.zeros((B, n)) if t_init_k is None else np.broadcast_to(
            np.asarray(t_init_k, np.float64), (B, n)).copy()
        self.g_work = np.zeros((B, O))
        self.t_obs = np.zeros((B, O))
        self.p_new = np.zeros((B, n))
        self.t_ring = np.zeros((B, n))
        self.n_iter = np.zeros((B, n), np.int64)
        self._h = None

    def step(self, h_s, u0, u1, t_base_k, pabs_fn, args_f, args_i, p_max, d2p_max, *, commit_fn=commit_none,
             bus_order=None, src_scale=None, max_iter: int = 10, tol_k: float = 1e-9, n_outer: int = 1) -> int:
        """One implicit substep (see implicit_step); returns the number of rings that hit max_iter."""
        s = self.system
        if self._h != h_s:
            self._coef = s.coefficients(h_s)
            self._h = h_s
        a, c0, c1 = self._coef
        bo = np.arange(s.n_rings, dtype=np.int64) if bus_order is None else np.asarray(bus_order, np.int64)
        ss = np.ones((1, s.n_src)) if src_scale is None else np.atleast_2d(np.asarray(src_scale, np.float64))
        nf = implicit_step(self.x, a, c0, c1, s.lag_src, s.lag_obs, s.lag_kind, s.opt_lags, s.obs_ptr, s.src_ring,
                           s.ring_obs, bo,
                           np.atleast_2d(u0), np.atleast_2d(u1), ss, np.atleast_1d(np.asarray(t_base_k, np.float64)),
                           self.p_prev, self.t_prev, pabs_fn, commit_fn, np.atleast_2d(args_f),
                           np.atleast_2d(args_i), np.atleast_2d(p_max), np.atleast_2d(d2p_max), max_iter, tol_k,
                           n_outer, self.g_work, self.t_obs, self.p_new, self.t_ring, self.n_iter)
        self.p_prev[:] = self.p_new
        self.t_prev[:] = self.t_ring
        return int(nf)
