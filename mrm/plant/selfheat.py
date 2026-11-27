# solve each ring temperature with optical absorption evaluated at that temperature.
# solve_ring_temperature follows one ring's previous thermal state.
# implicit_step_b advances one world in bus order, and implicit_step batches worlds.
# commit_fn updates downstream optical input after each ring solve.
# for a substep, solve F(T) = T - T_other - g*P_abs(T) = 0.
# T_other (K) includes non-optical heat and decayed heat from earlier absorption.
# g (K/W) is this substep's response to absorbed power, not the DC thermal resistance.
# for optical lag j, g = sum((1-exp(-h/tau_j))*r_j_opt), with r_j_opt in K/W.
# for one thermal state, C*dT/dt = P_heater + P_abs(T) - (T-T_ambient)/R.
# the graph of P_abs(T) and (T-T_ambient)/R-P_heater uses temperature as its horizontal axis.
# their vertical gap equals C*dT/dt - a positive gap warms the ring, and a negative gap cools it.
# intersections are equilibria - two attracting intersections around a repelling one give bistability.
# a stable branch ends at a fold, so the temperature moves to the remaining stable branch.
# the finite-substep residual F differs from that equilibrium heat budget until the steady limit.
# start from the previous temperature and search along -sign(F) to retain thermal history.
# valid power and curvature bounds prevent the search from skipping its first root.
# a failed numerical solve does not establish physical lock.
# branch_low starts at -infinity, branch_high at +infinity, and laser_ramp increases laser power.
# mrm.sim.kernel contains direct-call copies for numba inlining and caching.
# keep _solve_ring_eng and _implicit_step_eng aligned with this module.
# tests/verification/test_kernel_parity.py checks their agreement.
# pabs_lorentzian is a study model - engine optics sums all lines, orders, and directions.
# SelfHeatWork provides Python state and scratch arrays, and kernels also run without numba.
from __future__ import annotations

from collections.abc import Callable
import math
from numpy.typing import ArrayLike, NDArray

import numpy as np
from numba import njit

from mrm.plant.thermal import (LAG_EXOGENOUS, LAG_OPT_CROSS_RING, LAG_OPT_SELF, ThermalSystem,
                               helper_shared_or_world_row_index)

__all__ = [
    "solve_ring_temperature", "implicit_step_b", "implicit_step", "implicit_run_b",
    "pabs_lorentzian", "commit_none", "pack_lorentzian_args", "lorentzian_bounds",
    "LORENTZ_MAX_SLOPE", "BISTABILITY_FACTOR", "lorentzian_folds", "steady_detuning", "bistability_ratio",
    "SelfHeatWork", "STATUS_CONVERGED", "STATUS_MAX_ITER", "STATUS_BOUND", "STATUS_NAN_GAIN",
]

# max over u of 2|u|/(1 + u^2)^2 = 3 sqrt(3)/8 at |u| = 1/sqrt(3) - the steepest Lorentzian slope per HWHM (exact).
LORENTZ_MAX_SLOPE = 3.0 * math.sqrt(3.0) / 8.0
# Bistability threshold G_sh P_abs,peak >= (8/(3 sqrt 3)) HWHM = (4 sqrt(3)/9) FWHM ~ 0.770 FWHM (exact - compact-model notes, section 6.4).
# G_sh [m/W] is the self-heating resonance shift per absorbed watt.
BISTABILITY_FACTOR = 8.0 / (3.0 * math.sqrt(3.0))

# status of solve_ring_temperature
STATUS_CONVERGED = 0     # |F| <= tol_k (or the bracket closed to tol_k)
STATUS_MAX_ITER = 1      # max_iter reached - the point returned lies before the root
STATUS_BOUND = 2         # the search reached the end of the bracket without a sign change - p_max does not bound P_abs
STATUS_NAN_GAIN = 3      # g is NaN (broken thermal coefficients) - no solve, T = t_other is returned

# layout of the packed arrays of pabs_lorentzian (pack_lorentzian_args) - packed_i = [N, off_0, ..., off_{N-1}] - ring k's block starts at packed_f[off_k] with the LZ_* header, then LZ_PER_LINE values per line
LZ_LAM_REF = 0           # lambda_ref - resonance at T_ref [m]
LZ_T_REF = 1             # T_ref [K]
LZ_S0 = 2                # s0 = dlambda_r/dT at T_ref [m/K]
LZ_BETA = 3              # beta - relative curvature of the thermo-optic law [1/K]
LZ_N_LINES = 4           # L, the number of lines (stored as a float)
LZ_HEADER = 5            # header length - line l starts at off_k + LZ_HEADER + LZ_PER_LINE * l
LZ_LINE_LAM = 0          # line wavelength lambda_line [m]
LZ_LINE_HWHM = 1         # half width at half maximum of the line's Lorentzian [m]
LZ_LINE_PEAK = 2         # absorbed power at the peak p_peak [W]
LZ_PER_LINE = 3


# find one ring's end temperature. every trial refers to the same instant.
# start from the previous temperature and seek the first root of F(T) = T - (t_other + g*P_abs(T)).
# t_start=-infinity selects the cold endpoint, and +infinity selects the hot endpoint.
# at most max_iter+1 optical evaluations include the starting point.
# status 0 converges, 1 reaches max_iter, 2 violates the power bound, and 3 reports NaN gain.
# the power bound limits the whole search. the curvature bound limits each move inside it.
# near a fold, a Newton trial can pass two roots and see the original sign again.
# axes - F = packed floating fields, I = packed integer fields.
@njit
def solver_ring_temperature(
    pabs_fn: Callable,  # pabs_fn = P_abs (W, W/K) - callback for absorption and its temperature derivative at a trial temperature - type callable
    k: int,  # k = k (1) - physical ring index - type int scalar
    packed_f: NDArray[np.float64],  # packed_f = theta_opt (mixed units) - floating optical parameters and callback workspace - type float64 ndarray, shape (F,)
    packed_i: NDArray[np.int64],  # packed_i = i_opt (1) - integer optical dimensions and offsets - type int64 ndarray, shape (I,)
    t_other: float,  # t_other = T_other (K) - temperature before adding new same-ring optical heating - type float scalar
    g: float,  # g = g (K/W) - same-ring thermal gain during this substep - type float scalar
    t_start: float,  # t_start = T_prev (K) - previous branch temperature or an infinite endpoint selector - type float scalar
    p_max: float,  # p_max = P_max (W) - upper bound on absorbed power - type float scalar
    d2p_max: float,  # d2p_max = K_2P (W/K^2) - upper bound on absolute absorption curvature with temperature - type float scalar
    max_iter: int,  # max_iter = n_max (1) - maximum iterations per ring solve - type int scalar
    tol_k: float,  # tol_k = epsilon_T (K) - residual or bracket tolerance - type float scalar
) -> tuple[
    float,  # return[0] = T_next (K) - accepted ring temperature - type float scalar
    float,  # return[1] = P_abs (W) - absorption evaluated at the returned temperature - type float scalar
    float,  # return[2] = dP_abs/dT (W/K) - local slope at the returned temperature - type float scalar
    int,  # return[3] = n_iter (1) - number of iterations used - type int scalar
    int,  # return[4] = status (1) - convergence or failure code - type int scalar
]:
    # zero gain makes the optical temperature rise zero this step, so t_other already solves the equation.
    if g == 0.0:
        # the caller still needs the absorbed power and its slope at the solved temperature.
        absorbed_power_w, d_absorbed_power_dtemp_w_per_k = pabs_fn(t_other, k, packed_f, packed_i)
        # the callback supplies optical outputs even when the thermal solve takes no iterations.
        return (
            t_other,
            absorbed_power_w,
            d_absorbed_power_dtemp_w_per_k,
            0,
            STATUS_CONVERGED,
        )
    # a NaN gain gives no temperature bounds. report failure and evaluate the optical outputs at t_other.
    if g != g:
        absorbed_power_w, d_absorbed_power_dtemp_w_per_k = pabs_fn(t_other, k, packed_f, packed_i)
        # return the known temperature as a fallback and flag the undefined gain for the caller.
        return (
            t_other,
            absorbed_power_w,
            d_absorbed_power_dtemp_w_per_k,
            0,
            STATUS_NAN_GAIN,
        )
    # zero and peak absorption give the two extreme end temperatures. sort them for either sign of g.
    lower_temp_bound_k = min(t_other, t_other + g * p_max)
    upper_temp_bound_k = max(t_other, t_other + g * p_max)
    start_temp_k = t_start
    if start_temp_k != start_temp_k:  # NaN starts at t_other - infinities select bracket ends below
        start_temp_k = t_other
    # a cold start below the feasible interval begins at its cold endpoint.
    if start_temp_k < lower_temp_bound_k:
        start_temp_k = lower_temp_bound_k
    # a hot start above the feasible interval begins at its hot endpoint.
    if start_temp_k > upper_temp_bound_k:
        start_temp_k = upper_temp_bound_k
    start_power_w, d_start_power_dtemp_w_per_k = pabs_fn(start_temp_k, k, packed_f, packed_i)
    # the residual is the trial temperature minus the end temperature predicted by its absorption.
    start_residual_k = start_temp_k - t_other - g * start_power_w
    # an initial residual within tolerance already satisfies the temperature equation.
    if abs(start_residual_k) <= tol_k:
        return (
            start_temp_k,
            start_power_w,
            d_start_power_dtemp_w_per_k,
            0,
            STATUS_CONVERGED,
        )
    # a negative residual means the trial is too cold, so search warmer. a positive residual points cooler.
    search_direction = 1.0 if start_residual_k < 0.0 else -1.0
    # near - the last point on the start side of the root - far - the nearest point known beyond it (or the bracket end)
    near_temp_k, near_residual_k, near_absorbed_power_w, d_near_power_dtemp_w_per_k = (
        start_temp_k,
        start_residual_k,
        start_power_w,
        d_start_power_dtemp_w_per_k,
    )
    # the opposite endpoint bounds the search on the side toward which the residual closes.
    far_temp_k = upper_temp_bound_k if search_direction > 0.0 else lower_temp_bound_k
    far_residual_k, far_absorbed_power_w, d_far_power_dtemp_w_per_k = 0.0, 0.0, 0.0
    far_point_evaluated = False
    # only absorption bends the residual curve. F'' = -g*P_abs'', so K2 = abs(g)*d2p_max bounds abs(F'').
    residual_curvature_bound_per_k = abs(g) * d2p_max if d2p_max > 0.0 else 0.0
    iterations_used = 0
    # move from the previous temperature toward the first reachable root within the evaluation budget.
    for iteration in range(1, max_iter + 1):
        # a bracket narrower than the temperature tolerance needs no further trial.
        if (far_temp_k - near_temp_k) * search_direction <= tol_k:
            break
        iterations_used = iteration
        # a nearly flat tangent can put its zero far from this trial.
        d_residual_dtemp = (
            1.0 - g * d_near_power_dtemp_w_per_k
        )
        abs_residual_k = abs(near_residual_k)
        candidate_temp_k = math.nan  # the next temperature to evaluate
        if residual_curvature_bound_per_k > 0.0 and math.isfinite(d_residual_dtemp):
            # over a distance d in the search direction, the gap can close by at most F'*d + K2*d**2/2.
            # set this equal to abs(F). its positive root is the earliest possible crossing under the bound.
            temperature_step_k = (
                2.0
                * abs_residual_k
                / (
                    d_residual_dtemp
                    + math.sqrt(
                        d_residual_dtemp * d_residual_dtemp
                        + 2.0 * residual_curvature_bound_per_k * abs_residual_k
                    )
                )
            )
            trial_temp_k = near_temp_k + search_direction * temperature_step_k
            # the curvature-limited step is useful only while it stays inside the bracket.
            if (far_temp_k - trial_temp_k) * search_direction > 0.0:
                candidate_temp_k = trial_temp_k
            # an unsampled endpoint can establish the residual on the opposite side.
            elif not far_point_evaluated:
                candidate_temp_k = far_temp_k  # the power bound supplies the bracket end's residual sign
            # else - the step reached a sign change already found (inexact dP/dT or d2p_max) - secant below
        elif math.isfinite(d_residual_dtemp) and d_residual_dtemp > 0.0:
            # Newton follows the local tangent to zero at T - F/F'. a positive slope puts it in the search direction.
            trial_temp_k = near_temp_k + search_direction * abs_residual_k / d_residual_dtemp
            # reject a Newton trial that fails to advance or crosses the far endpoint.
            if (trial_temp_k - near_temp_k) * search_direction > 0.0 and (
                far_temp_k - trial_temp_k
            ) * search_direction > 0.0:
                candidate_temp_k = trial_temp_k
        # a rejected step still leaves a bracket that can be narrowed without using the derivative.
        if math.isnan(candidate_temp_k):
            # two distinct measured residuals determine a secant intercept.
            if far_point_evaluated and far_residual_k != near_residual_k:
                # interpolate the temperature at which the line between the measured residuals crosses zero.
                trial_temp_k = near_temp_k - near_residual_k * (far_temp_k - near_temp_k) / (
                    far_residual_k - near_residual_k
                )
                # the secant must make progress without leaving the bracket.
                if (trial_temp_k - near_temp_k) * search_direction > 0.0 and (
                    far_temp_k - trial_temp_k
                ) * search_direction > 0.0:
                    candidate_temp_k = trial_temp_k
            # bisection still halves the uncertainty when neither derivative nor secant gives a usable step.
            if math.isnan(candidate_temp_k):
                # the midpoint divides the remaining temperature interval into equal halves.
                candidate_temp_k = 0.5 * (near_temp_k + far_temp_k)
        # evaluate the optical model again to check what power this trial temperature actually absorbs.
        candidate_power_w, d_candidate_power_dtemp_w_per_k = pabs_fn(candidate_temp_k, k, packed_f, packed_i)
        # compare the actual optical rise at the candidate with the rise needed to reach it.
        candidate_residual_k = candidate_temp_k - t_other - g * candidate_power_w
        # agreement within the heat-balance tolerance accepts this temperature and its optical response.
        if abs(candidate_residual_k) <= tol_k:
            return (
                candidate_temp_k,
                candidate_power_w,
                d_candidate_power_dtemp_w_per_k,
                iteration,
                STATUS_CONVERGED,
            )
        # matching signs advance the near endpoint but can hide a crossed pair of roots without a curvature bound.
        if candidate_residual_k * near_residual_k > 0.0:
            near_temp_k, near_residual_k, near_absorbed_power_w, d_near_power_dtemp_w_per_k = (
                candidate_temp_k,
                candidate_residual_k,
                candidate_power_w,
                d_candidate_power_dtemp_w_per_k,
            )
        else:
            # a sign reversal puts the candidate across the crossing and tightens the far endpoint.
            far_temp_k, far_residual_k, far_absorbed_power_w, d_far_power_dtemp_w_per_k = (
                candidate_temp_k,
                candidate_residual_k,
                candidate_power_w,
                d_candidate_power_dtemp_w_per_k,
            )
            far_point_evaluated = True
    # a sufficiently narrow interval permits a temperature-based convergence test.
    if (far_temp_k - near_temp_k) * search_direction <= tol_k:
        # only a measured far endpoint can provide an optical response at that temperature.
        if far_point_evaluated:
            # choose the endpoint whose absorbed power agrees more closely with the temperature balance.
            if abs(far_residual_k) < abs(near_residual_k):
                return (
                    far_temp_k,
                    far_absorbed_power_w,
                    d_far_power_dtemp_w_per_k,
                    iterations_used,
                    STATUS_CONVERGED,
                )
            # the near endpoint has the smaller or equal measured imbalance.
            return (
                near_temp_k,
                near_absorbed_power_w,
                d_near_power_dtemp_w_per_k,
                iterations_used,
                STATUS_CONVERGED,
            )
        # if the end of the power-bounded interval still has the starting sign, the assumed power bound failed.
        return (
            near_temp_k,
            near_absorbed_power_w,
            d_near_power_dtemp_w_per_k,
            iterations_used,
            STATUS_BOUND,
        )
    return (
        near_temp_k,
        near_absorbed_power_w,
        d_near_power_dtemp_w_per_k,
        iterations_used,
        STATUS_MAX_ITER,
    )

# (todo) remove these alias names eventually
solve_ring_temperature = solver_ring_temperature


# leave downstream optical inputs unchanged after solving a ring.
# axes - F = packed floating fields, I = packed integer fields.
# x
@njit
def helper_keep_optical_inputs(
    t_k: float,  # t_k = T (K) - trial ring temperature - type float scalar
    k: int,  # k = k (1) - physical ring index - type int scalar
    packed_f: NDArray[np.float64],  # packed_f = theta_opt (mixed units) - floating optical parameters and callback workspace - type float64 ndarray, shape (F,)
    packed_i: NDArray[np.int64],  # packed_i = i_opt (1) - integer optical dimensions and offsets - type int64 ndarray, shape (I,)
) -> (
    None  # return = None - leave the optical buffers unchanged - type None
):
    return None

commit_none = helper_keep_optical_inputs


# advance one world by a thermal substep and solve its optical absorption in bus order.
# the ring temperature sets its absorption, and that same absorption heats the ring during this interval.
# one sweep uses the previous powers for cross-ring heating. additional sweeps replace those powers with the newest estimates.
# Lag inputs - LAG_EXOGENOUS u0/u1 (exact FOH) - LAG_OPT_SELF P_abs,k(T_k) held over the step (implicit) - LAG_OPT_CROSS_RING p_prev (lagged one substep) - LAG_OPT_OTHER (heater resistor, PD, sensors) the new P_abs.
# the ring observer of ring k ends equal to t_ring[k] (to rounding).
# Cost - one pass over all lags, the ring solves, then one pass over the optical lags only.
# on the negative-feedback flank of one thermal pole, let a = exp(-h/tau) and G = residue*abs(dP_abs/dT).
# explicit coupling has multiplier a-(1-a)*G.
# it can alternate and diverge for fast poles — about -1.1 to -5.9 at h=1 us, tau=0.3–0.5 us, and G=1.2–7.
# implicit coupling has multiplier a/(1+(1-a)*G), between zero and a for G>=0.
# it avoids this numerical oscillation.
# these are single-pole estimates when the model has several poles.
# cross-ring optical heat uses the previous substep's power in a single outer sweep.
# realization
# checks its per-step gain and rejects values >=0.05.
# additional outer sweeps make that coupling
# implicit for longer steps.
# this is distinct from the same-ring solve.
# if a ring solve does not converge, the thermal state receives the power consistent with its returned temperature when that power can be recovered.
# the next substep continues from this numerical result.
# do not interpret iteration error as an additional physical thermal time constant.
# axes - F = packed floating fields, I = packed integer fields, M = thermal lags, N = rings, O = temperature observers, Q = optical lag indices, S = heat sources.
# one world has 8 rings. solve the ring temperatures for all of them, one by one, left to right
# x
@njit
def solver_coupled_temperatures_for_world(
    x: NDArray[np.float64],  # x = x (K) - temperature rise stored in each thermal lag, updated in place - type float64 ndarray, shape (M,)
    a: NDArray[np.float64],  # a = a (1) - fraction of the previous lag state retained over the step - type float64 ndarray, shape (M,)
    c0: NDArray[np.float64],  # c0 = c_0 (K/W) - lag gain applied to the initial source power - type float64 ndarray, shape (M,)
    c1: NDArray[np.float64],  # c1 = c_1 (K/W) - lag gain applied to the final source power - type float64 ndarray, shape (M,)
    lag_src: NDArray[np.int64],  # lag_src = s_m (1) - source index for each lag - type int64 ndarray, shape (M,)
    lag_obs: NDArray[np.int64],  # lag_obs = o_m (1) - observer index for each lag - type int64 ndarray, shape (M,)
    lag_kind: np.ndarray,  # lag_kind = kind_m (1) - forcing type for each thermal lag - type integer ndarray, shape (M,)
    opt_lags: NDArray[np.int64],  # opt_lags = i_opt (1) - indices of lags driven by the newly accepted optical power - type int64 ndarray, shape (Q,)
    obs_ptr: NDArray[np.int64],  # obs_ptr = p_o (1) - offsets delimiting each observer’s contiguous lag rows - type int64 ndarray, shape (O+1,)
    src_ring: NDArray[np.int64],  # src_ring = k_s (1) - physical ring index associated with each optical source - type int64 ndarray, shape (S,)
    ring_obs: NDArray[np.int64],  # ring_obs = o_k (1) - waveguide-temperature observer index for each ring - type int64 ndarray, shape (N,)
    bus_order: NDArray[np.int64],  # bus_order = k_bus (1) - ring indices in forward optical propagation order - type int64 ndarray, shape (N,)
    u0: NDArray[np.float64],  # u0 = u_0 (W) - prescribed source power at the start of the step - type float64 ndarray, shape (S,)
    u1: NDArray[np.float64],  # u1 = u_1 (W) - prescribed source power at the end of the step - type float64 ndarray, shape (S,)
    src_scale: NDArray[np.float64],  # src_scale = s_u (1) - gain multiplier for each heat source - type float64 ndarray, shape (S,)
    t_base: float,  # t_base = T_base (K) - base temperature at the end of the substep - type float scalar
    p_prev: NDArray[np.float64],  # p_prev = P_prev (W) - previously accepted absorbed power used for cross-ring heating - type float64 ndarray, shape (N,)
    t_prev: NDArray[np.float64],  # t_prev = T_prev (K) - previous ring temperature used to continue its branch - type float64 ndarray, shape (N,)
    pabs_fn: Callable,  # pabs_fn = P_abs (W, W/K) - callback for absorption and its temperature derivative at a trial temperature - type callable
    commit_fn: Callable,  # commit_fn = commit (mixed units) - callback that propagates optical inputs after accepting a ring temperature - type callable
    packed_f: NDArray[np.float64],  # packed_f = theta_opt (mixed units) - floating optical parameters and callback workspace - type float64 ndarray, shape (F,)
    packed_i: NDArray[np.int64],  # packed_i = i_opt (1) - integer optical dimensions and offsets - type int64 ndarray, shape (I,)
    p_max: NDArray[np.float64],  # p_max = P_max (W) - upper bound on absorbed power - type float64 ndarray, shape (N,)
    d2p_max: NDArray[np.float64],  # d2p_max = K_2P (W/K^2) - upper bound on absolute absorption curvature with temperature - type float64 ndarray, shape (N,)
    max_iter: int,  # max_iter = n_max (1) - maximum iterations per ring solve - type int scalar
    tol_k: float,  # tol_k = epsilon_T (K) - residual or bracket tolerance - type float scalar
    n_outer: int,  # n_outer = n_sweep (1) - requested number of optical crosstalk sweeps within one step - type int scalar
    g_work: NDArray[np.float64],  # g_work = g_o (K/W) - scratch space for the local absorbed-power gain at each observer - type float64 ndarray, shape (O,)
    t_obs: NDArray[np.float64],  # t_obs = T_obs (K) - updated absolute temperature at each observer - type float64 ndarray, shape (O,)
    p_new: NDArray[np.float64],  # p_new = P_next (W) - absorbed power supplied to the thermal update - type float64 ndarray, shape (N,)
    t_ring: NDArray[np.float64],  # t_ring = T_next (K) - ring temperatures returned by the optical heating solve - type float64 ndarray, shape (N,)
    n_iter: NDArray[np.int64],  # n_iter = n_iter (1) - total solver iterations accumulated per ring - type int64 ndarray, shape (N,)
) -> (
    int  # return = n_fail (1) - number of unconverged ring solves - type int scalar
):
    ring_count = ring_obs.shape[0]
    # the optical bus must visit each physical ring exactly once.
    if bus_order.shape[0] != ring_count:
        raise ValueError("implicit_step: bus_order must list every ring once")
    # reuse the iteration counters as temporary visit flags before the first solve.
    for ring_index in range(ring_count):
        n_iter[ring_index] = 0
    for bus_position in range(ring_count):  # bus_order must be a permutation of 0..N-1 (n_iter as the marker)
        ring_index = bus_order[bus_position]
        # a repeated or out-of-range ring would propagate the wrong optical input downstream.
        if ring_index < 0 or ring_index >= ring_count or n_iter[ring_index] != 0:
            raise ValueError("implicit_step: bus_order must be a permutation of the ring indices")
        # each ring needs an observer temperature for its absorption callback.
        if ring_obs[ring_index] < 0:
            raise ValueError("implicit_step needs the ring_mode_weighted observer of every ring")
        n_iter[ring_index] = 1
    # the final pointer marks the end of the last observer’s lag interval.
    n_obs = obs_ptr.shape[0] - 1
    # advance every lag with its known input, then sum each observer.
    # Two passes give the same values and, per observer, the same summation order as one pass over the observers - the lag update without the sums is ~20 % faster.
    # LAG_OPT_SELF lags only decay here - their input is the unknown P_abs, and g_work collects their gain.
    for lag_index in range(lag_kind.shape[0]):
        source_index = lag_src[lag_index]
        lag_type = lag_kind[lag_index]
        # decay the previous contribution before adding heat delivered during this step.
        lag_temperature_rise_k = a[lag_index] * x[lag_index]
        # prescribed sources have known powers at both ends of the substep.
        if lag_type == LAG_EXOGENOUS:
            # the two endpoint weights integrate a linearly changing prescribed source power.
            lag_temperature_rise_k += src_scale[source_index] * (
                c0[lag_index] * u0[source_index] + c1[lag_index] * u1[source_index]
            )
        elif lag_type == LAG_OPT_CROSS_RING:
            # cross-ring optical power initially comes from the preceding accepted step.
            lag_temperature_rise_k += (
                src_scale[source_index] * (c0[lag_index] + c1[lag_index]) * p_prev[src_ring[source_index]]
            )
        x[lag_index] = lag_temperature_rise_k
    # separate each observer’s known temperature from its response to the unknown local absorption.
    for observer_index in range(n_obs):
        known_observer_temp_k = t_base
        self_heating_gain_k_per_w = 0.0
        # consecutive lag rows belong to the same observer through the pointer interval.
        for lag_index in range(obs_ptr[observer_index], obs_ptr[observer_index + 1]):
            # only same-ring absorption contributes to the unknown gain in the scalar equation.
            if lag_kind[lag_index] == LAG_OPT_SELF:
                # held absorption multiplies both endpoint weights so their gains add.
                self_heating_gain_k_per_w += src_scale[lag_src[lag_index]] * (c0[lag_index] + c1[lag_index])
            known_observer_temp_k += x[lag_index]
        t_obs[observer_index] = known_observer_temp_k  # everything except the new optical inputs
        g_work[observer_index] = self_heating_gain_k_per_w
    failed_ring_count = 0
    cross_ring_absorbed_power_w = np.empty(ring_count)  # P_abs currently driving the LAG_OPT_CROSS_RING lags
    # start each ring on its previous temperature branch with its previous absorption driving crosstalk.
    for ring_index in range(ring_count):
        cross_ring_absorbed_power_w[ring_index] = p_prev[ring_index]
        t_ring[ring_index] = t_prev[ring_index]
        n_iter[ring_index] = 0
    # another sweep updates crosstalk without advancing simulation time again.
    for sweep in range(max(n_outer, 1)):
        failed_ring_count = 0
        # upstream through power must reach a ring before its absorption is evaluated.
        for bus_position in range(ring_count):
            ring_index = bus_order[bus_position]
            observer_index = ring_obs[ring_index]
            # a missing observer leaves this ring without a temperature equation.
            if observer_index < 0:
                raise ValueError("implicit_step needs the ring_mode_weighted observer of every ring")
            (
                solved_temp_k,
                solved_absorbed_power_w,
                d_solved_power_dtemp_w_per_k,
                iterations_used,
                solve_status,
            ) = solver_ring_temperature(
                pabs_fn,
                ring_index,
                packed_f,
                packed_i,
                t_obs[observer_index],
                g_work[observer_index],
                t_ring[ring_index],
                p_max[ring_index],
                d2p_max[ring_index],
                max_iter,
                tol_k,
            )
            t_ring[ring_index] = solved_temp_k
            p_new[ring_index] = solved_absorbed_power_w
            # count iterations across sweeps to expose the total cost for this ring.
            n_iter[ring_index] += iterations_used
            # a failed root still needs a temperature-consistent heat input for the next step.
            if solve_status != STATUS_CONVERGED:
                failed_ring_count += 1
                # a branch jump can need more than max_iter steps, or p_max can fail to bound absorption.
                # feed the thermal state with power consistent with the returned temperature.
                # with valid bounds, that point stays before the root, and the next substep continues from it.
                # a NaN gain keeps P_abs at the returned point (no such power exists).
                ring_self_heating_gain_k_per_w = g_work[observer_index]
                # recover absorbed power from the temperature increment for a nonzero gain that passed the NaN guard.
                if ring_self_heating_gain_k_per_w != 0.0 and solve_status != STATUS_NAN_GAIN:
                    # invert the temperature balance to recover the heating that produced the returned temperature.
                    p_new[ring_index] = (
                        solved_temp_k - t_obs[observer_index]
                    ) / ring_self_heating_gain_k_per_w
            commit_fn(solved_temp_k, ring_index, packed_f, packed_i)
        # the requested sweep count limits how often neighboring rings can exchange updated power.
        if sweep + 1 >= n_outer:
            break
        # Gauss-Seidel - drive the cross-ring lags with the newest P_abs, then sweep again
        max_power_change_w = 0.0
        max_absorbed_power_w = 0.0
        # compare the largest power correction with the largest new absorbed power in the bank.
        for ring_index in range(ring_count):
            # keep the largest discrepancy so convergence covers even the most strongly coupled ring.
            max_power_change_w = max(
                max_power_change_w, abs(p_new[ring_index] - cross_ring_absorbed_power_w[ring_index])
            )
            max_absorbed_power_w = max(max_absorbed_power_w, abs(p_new[ring_index]))
        # another sweep would change cross-ring forcing by at most the relative tolerance.
        if max_power_change_w <= 1e-12 * max_absorbed_power_w:
            break
        # update only the paths between different rings before the next optical sweep.
        for lag_index in range(lag_kind.shape[0]):
            # the prior forcing is already in the lag state so apply only its correction.
            if lag_kind[lag_index] == LAG_OPT_CROSS_RING:
                source_index = lag_src[lag_index]
                source_ring_index = src_ring[source_index]
                # multiply the change in held absorbed power by this path’s gain over the step.
                lag_temperature_change_k = (
                    src_scale[source_index]
                    * (c0[lag_index] + c1[lag_index])
                    * (p_new[source_ring_index] - cross_ring_absorbed_power_w[source_ring_index])
                )
                x[lag_index] += lag_temperature_change_k
                t_obs[lag_obs[lag_index]] += lag_temperature_change_k
        # retain the forcing just applied so the next sweep does not count it twice.
        for ring_index in range(ring_count):
            cross_ring_absorbed_power_w[ring_index] = p_new[ring_index]
    # complete the same-ring and other optical paths once after accepting the final sweep.
    for i in range(opt_lags.shape[0]):
        lag_index = opt_lags[i]
        source_index = lag_src[lag_index]
        # both endpoint weights act on the same held optical power.
        lag_temperature_change_k = (
            src_scale[source_index] * (c0[lag_index] + c1[lag_index]) * p_new[src_ring[source_index]]
        )
        x[lag_index] += lag_temperature_change_k
        t_obs[lag_obs[lag_index]] += lag_temperature_change_k
    return failed_ring_count

implicit_step_b = solver_coupled_temperatures_for_world


# Batched form of implicit_step_b - one implicit self-heating step for every world.
# B counts worlds, and B|1 permits one shared row.
# x has shape (B,M), and a, c0, c1 have shape (B|1,M).
# u0, u1, and src_scale have shape (B|1,S), and t_base has shape (B|1,).
# p_prev and t_prev have shape (B,N).
# packed_f and packed_i have shapes (B|1,nf) and (B|1,ni).
# p_max and d2p_max have shape (B|1,N).
# g_work and t_obs have shape (B,O).
# p_new and t_ring are float64 (B,N), and n_iter is int64 (B,N).
# the other arguments and the units are those of `implicit_step_b`.
# returns the total number of rings that did not converge.
# axes - B = simulated worlds, F = packed floating fields, I = packed integer fields, M = thermal lags, N = rings, O = temperature observers, Q = optical lag indices, S = heat sources.
@njit
def solver_coupled_temperatures(
    x: NDArray[np.float64],  # x = x (K) - temperature rise stored in each thermal lag, updated in place - type float64 ndarray, shape (B,M)
    a: NDArray[np.float64],  # a = a (1) - fraction of the previous lag state retained over the step - type float64 ndarray, shape (B|1,M)
    c0: NDArray[np.float64],  # c0 = c_0 (K/W) - lag gain applied to the initial source power - type float64 ndarray, shape (B|1,M)
    c1: NDArray[np.float64],  # c1 = c_1 (K/W) - lag gain applied to the final source power - type float64 ndarray, shape (B|1,M)
    lag_src: NDArray[np.int64],  # lag_src = s_m (1) - source index for each lag - type int64 ndarray, shape (M,)
    lag_obs: NDArray[np.int64],  # lag_obs = o_m (1) - observer index for each lag - type int64 ndarray, shape (M,)
    lag_kind: np.ndarray,  # lag_kind = kind_m (1) - forcing type for each thermal lag - type integer ndarray, shape (M,)
    opt_lags: NDArray[np.int64],  # opt_lags = i_opt (1) - indices of lags driven by the newly accepted optical power - type int64 ndarray, shape (Q,)
    obs_ptr: NDArray[np.int64],  # obs_ptr = p_o (1) - offsets delimiting each observer’s contiguous lag rows - type int64 ndarray, shape (O+1,)
    src_ring: NDArray[np.int64],  # src_ring = k_s (1) - physical ring index associated with each optical source - type int64 ndarray, shape (S,)
    ring_obs: NDArray[np.int64],  # ring_obs = o_k (1) - waveguide-temperature observer index for each ring - type int64 ndarray, shape (N,)
    bus_order: NDArray[np.int64],  # bus_order = k_bus (1) - ring indices in forward optical propagation order - type int64 ndarray, shape (N,)
    u0: NDArray[np.float64],  # u0 = u_0 (W) - prescribed source power at the start of the step - type float64 ndarray, shape (B|1,S)
    u1: NDArray[np.float64],  # u1 = u_1 (W) - prescribed source power at the end of the step - type float64 ndarray, shape (B|1,S)
    src_scale: NDArray[np.float64],  # src_scale = s_u (1) - gain multiplier for each heat source - type float64 ndarray, shape (B|1,S)
    t_base: NDArray[np.float64],  # t_base = T_base (K) - base temperature at the end of the substep - type float64 ndarray, shape (B|1,)
    p_prev: NDArray[np.float64],  # p_prev = P_prev (W) - previously accepted absorbed power used for cross-ring heating - type float64 ndarray, shape (B,N)
    t_prev: NDArray[np.float64],  # t_prev = T_prev (K) - previous ring temperature used to continue its branch - type float64 ndarray, shape (B,N)
    pabs_fn: Callable,  # pabs_fn = P_abs (W, W/K) - callback for absorption and its temperature derivative at a trial temperature - type callable
    commit_fn: Callable,  # commit_fn = commit (mixed units) - callback that propagates optical inputs after accepting a ring temperature - type callable
    packed_f: NDArray[np.float64],  # packed_f = theta_opt (mixed units) - floating optical parameters and callback workspace - type float64 ndarray, shape (B|1,F)
    packed_i: NDArray[np.int64],  # packed_i = i_opt (1) - integer optical dimensions and offsets - type int64 ndarray, shape (B|1,I)
    p_max: NDArray[np.float64],  # p_max = P_max (W) - upper bound on absorbed power - type float64 ndarray, shape (B|1,N)
    d2p_max: NDArray[np.float64],  # d2p_max = K_2P (W/K^2) - upper bound on absolute absorption curvature with temperature - type float64 ndarray, shape (B|1,N)
    max_iter: int,  # max_iter = n_max (1) - maximum iterations per ring solve - type int scalar
    tol_k: float,  # tol_k = epsilon_T (K) - residual or bracket tolerance - type float scalar
    n_outer: int,  # n_outer = n_sweep (1) - requested number of optical crosstalk sweeps within one step - type int scalar
    g_work: NDArray[np.float64],  # g_work = g_o (K/W) - scratch space for the local absorbed-power gain at each observer - type float64 ndarray, shape (B,O)
    t_obs: NDArray[np.float64],  # t_obs = T_obs (K) - updated absolute temperature at each observer - type float64 ndarray, shape (B,O)
    p_new: NDArray[np.float64],  # p_new = P_next (W) - absorbed power supplied to the thermal update - type float64 ndarray, shape (B,N)
    t_ring: NDArray[np.float64],  # t_ring = T_next (K) - ring temperatures returned by the optical heating solve - type float64 ndarray, shape (B,N)
    n_iter: NDArray[np.int64],  # n_iter = n_iter (1) - total solver iterations accumulated per ring - type int64 ndarray, shape (B,N)
) -> (
    int  # return = n_fail (1) - number of unconverged ring solves - type int scalar
):
    B = x.shape[0]
    # read-only input rows may be common to all devices or specific to each device.
    for n in (a.shape[0], c0.shape[0], c1.shape[0], u0.shape[0], u1.shape[0], src_scale.shape[0], t_base.shape[0],
              packed_f.shape[0], packed_i.shape[0], p_max.shape[0], d2p_max.shape[0]):
        # other row counts cannot map a coefficient to every world.
        if n != 1 and n != B:
            raise ValueError("implicit_step: every per-world input must have 1 or B rows")
    # the decay and endpoint weights must refer to the same coefficient row.
    if c0.shape[0] != a.shape[0] or c1.shape[0] != a.shape[0]:
        raise ValueError("implicit_step: a, c0, c1 must have the same rows")
    # independent worlds require separate writable state and output rows.
    for n in (p_prev.shape[0], t_prev.shape[0], g_work.shape[0], t_obs.shape[0], p_new.shape[0], t_ring.shape[0],
              n_iter.shape[0]):
        # sharing writable rows would overwrite a previous world’s results.
        if n != B:
            raise ValueError("implicit_step: p_prev, t_prev and the work/output arrays must have B rows")
    n_fail = 0
    # advance each sampled bank using its own thermal state and the applicable coefficient rows.
    for b in range(x.shape[0]):
        ra = helper_shared_or_world_row_index(b, a.shape[0])
        n_fail += solver_coupled_temperatures_for_world(
            x[b], a[ra], c0[ra], c1[ra], lag_src, lag_obs, lag_kind, opt_lags, obs_ptr, src_ring, ring_obs, bus_order,
            u0[helper_shared_or_world_row_index(b, u0.shape[0])], u1[helper_shared_or_world_row_index(b, u1.shape[0])], src_scale[helper_shared_or_world_row_index(b, src_scale.shape[0])],
            t_base[helper_shared_or_world_row_index(b, t_base.shape[0])], p_prev[b], t_prev[b], pabs_fn, commit_fn,
            packed_f[helper_shared_or_world_row_index(b, packed_f.shape[0])], packed_i[helper_shared_or_world_row_index(b, packed_i.shape[0])],
            p_max[helper_shared_or_world_row_index(b, p_max.shape[0])], d2p_max[helper_shared_or_world_row_index(b, d2p_max.shape[0])], max_iter, tol_k, n_outer,
            g_work[b], t_obs[b], p_new[b], t_ring[b], n_iter[b])
    # aggregate failed ring solves across all simulated worlds.
    return n_fail

implicit_step = solver_coupled_temperatures


# run n held-input substeps of one world (open loop) - u_steps (n, S) W, t_base_steps (n,) K.
# Updates x, p_prev, t_prev in place - fills t_ring_hist, p_hist (n, N) and t_obs_hist (n, O) with end-of-step values.
# returns the number of (ring, step) solves that did not converge (status 1, max_iter - 2, P_abs above p_max - or 3, a NaN gain).
# for tests, replicas and quasi-static sweeps.
# axes - F = packed floating fields, H = thermal steps, I = packed integer fields, M = thermal lags, N = rings, O = temperature observers, Q = optical lag indices, S = heat sources.
@njit
def solver_temperature_history_for_world(
    x: NDArray[np.float64],  # x = x (K) - temperature rise stored in each thermal lag, updated in place - type float64 ndarray, shape (M,)
    a: NDArray[np.float64],  # a = a (1) - fraction of the previous lag state retained over the step - type float64 ndarray, shape (M,)
    c0: NDArray[np.float64],  # c0 = c_0 (K/W) - lag gain applied to the initial source power - type float64 ndarray, shape (M,)
    c1: NDArray[np.float64],  # c1 = c_1 (K/W) - lag gain applied to the final source power - type float64 ndarray, shape (M,)
    lag_src: NDArray[np.int64],  # lag_src = s_m (1) - source index for each lag - type int64 ndarray, shape (M,)
    lag_obs: NDArray[np.int64],  # lag_obs = o_m (1) - observer index for each lag - type int64 ndarray, shape (M,)
    lag_kind: np.ndarray,  # lag_kind = kind_m (1) - forcing type for each thermal lag - type integer ndarray, shape (M,)
    opt_lags: NDArray[np.int64],  # opt_lags = i_opt (1) - indices of lags driven by the newly accepted optical power - type int64 ndarray, shape (Q,)
    obs_ptr: NDArray[np.int64],  # obs_ptr = p_o (1) - offsets delimiting each observer’s contiguous lag rows - type int64 ndarray, shape (O+1,)
    src_ring: NDArray[np.int64],  # src_ring = k_s (1) - physical ring index associated with each optical source - type int64 ndarray, shape (S,)
    ring_obs: NDArray[np.int64],  # ring_obs = o_k (1) - waveguide-temperature observer index for each ring - type int64 ndarray, shape (N,)
    bus_order: NDArray[np.int64],  # bus_order = k_bus (1) - ring indices in forward optical propagation order - type int64 ndarray, shape (N,)
    u_steps: NDArray[np.float64],  # u_steps = u (W) - held source powers for each substep - type float64 ndarray, shape (H,S)
    t_base_steps: NDArray[np.float64],  # t_base_steps = T_base (K) - base temperature at the end of each substep - type float64 ndarray, shape (H,)
    src_scale: NDArray[np.float64],  # src_scale = s_u (1) - gain multiplier for each heat source - type float64 ndarray, shape (S,)
    p_prev: NDArray[np.float64],  # p_prev = P_prev (W) - previously accepted absorbed power used for cross-ring heating - type float64 ndarray, shape (N,)
    t_prev: NDArray[np.float64],  # t_prev = T_prev (K) - previous ring temperature used to continue its branch - type float64 ndarray, shape (N,)
    pabs_fn: Callable,  # pabs_fn = P_abs (W, W/K) - callback for absorption and its temperature derivative at a trial temperature - type callable
    commit_fn: Callable,  # commit_fn = commit (mixed units) - callback that propagates optical inputs after accepting a ring temperature - type callable
    packed_f: NDArray[np.float64],  # packed_f = theta_opt (mixed units) - floating optical parameters and callback workspace - type float64 ndarray, shape (F,)
    packed_i: NDArray[np.int64],  # packed_i = i_opt (1) - integer optical dimensions and offsets - type int64 ndarray, shape (I,)
    p_max: NDArray[np.float64],  # p_max = P_max (W) - upper bound on absorbed power - type float64 ndarray, shape (N,)
    d2p_max: NDArray[np.float64],  # d2p_max = K_2P (W/K^2) - upper bound on absolute absorption curvature with temperature - type float64 ndarray, shape (N,)
    max_iter: int,  # max_iter = n_max (1) - maximum iterations per ring solve - type int scalar
    tol_k: float,  # tol_k = epsilon_T (K) - residual or bracket tolerance - type float scalar
    n_outer: int,  # n_outer = n_sweep (1) - requested number of optical crosstalk sweeps within one step - type int scalar
    t_ring_hist: NDArray[np.float64],  # t_ring_hist = T_ring (K) - output ring temperatures at each step endpoint - type float64 ndarray, shape (H,N)
    p_hist: NDArray[np.float64],  # p_hist = P_abs (W) - output absorption history at each step endpoint - type float64 ndarray, shape (H,N)
    t_obs_hist: NDArray[np.float64],  # t_obs_hist = T_obs (K) - output temperature history for all observers - type float64 ndarray, shape (H,O)
) -> (
    int  # return = n_fail (1) - number of unconverged ring solves - type int scalar
):
    n_steps = u_steps.shape[0]
    # the last pointer is a sentinel so there is one fewer observer than pointers.
    n_obs = obs_ptr.shape[0] - 1
    n_rings = ring_obs.shape[0]
    g_work = np.zeros(n_obs)
    p_new = np.zeros(n_rings)
    t_ring = np.zeros(n_rings)
    # iteration counters and optical work arrays are reused over the full history.
    n_iter = np.zeros(n_rings, np.int64)
    failure_count = 0
    # using the same power at both endpoints makes this history a sequence of constant inputs.
    for i in range(n_steps):
        failure_count += solver_coupled_temperatures_for_world(x, a, c0, c1, lag_src, lag_obs, lag_kind, opt_lags, obs_ptr, src_ring,
            ring_obs,
                                 bus_order, u_steps[i], u_steps[i], src_scale, t_base_steps[i], p_prev, t_prev,
                                 pabs_fn, commit_fn, packed_f, packed_i, p_max, d2p_max, max_iter, tol_k, n_outer,
                                 g_work, t_obs_hist[i], p_new, t_ring, n_iter)
        # carry the accepted branch and absorbed power into the next substep.
        for k in range(n_rings):
            p_prev[k] = p_new[k]
            t_prev[k] = t_ring[k]
            t_ring_hist[i, k] = t_ring[k]
            p_hist[i, k] = p_new[k]
    return failure_count

implicit_run_b = solver_temperature_history_for_world


# built-in Lorentzian absorption (packed arguments)
# sum Lorentzian absorption and its temperature derivative for one ring.
# lambda_r = lambda_ref + s0*(dT + beta*dT^2/2), with dT = T-T_ref (K).
# for each line, u = (lambda_r-lambda_line)/HWHM (1).
# P_abs = sum(p_peak/(1+u^2)).
# include dlambda_r/dT = s0*(1+beta*dT) (m/K) in the derivative.
# this study model uses the TCMT Lorentzian limit.
# axes - F = packed floating fields, I = packed integer fields.
@njit
def physics_lorentzian_absorption_and_temperature_slope(
    t: float,  # t = T (K) - trial ring temperature - type float scalar
    k: int,  # k = k (1) - physical ring index - type int scalar
    packed_f: NDArray[np.float64],  # packed_f = theta_opt (mixed units) - floating optical parameters and callback workspace - type float64 ndarray, shape (F,)
    packed_i: NDArray[np.int64],  # packed_i = i_opt (1) - integer optical dimensions and offsets - type int64 ndarray, shape (I,)
) -> tuple[
    float,  # return[0] = P_abs (W) - total absorbed power - type float scalar
    float,  # return[1] = dP_abs/dT (W/K) - temperature slope of total absorption - type float scalar
]:
    # the ring index selects its parameter block through the packed offset table.
    ring_offset = packed_i[1 + k]
    reference_wavelength_m = packed_f[ring_offset + LZ_LAM_REF]
    reference_temp_k = packed_f[ring_offset + LZ_T_REF]
    reference_tuning_slope_m_per_k = packed_f[ring_offset + LZ_S0]
    relative_tuning_curvature_per_k = packed_f[ring_offset + LZ_BETA]
    n_lines = int(packed_f[ring_offset + LZ_N_LINES])
    # integrate the linear thermal tuning slope from the reference temperature to the trial temperature.
    temp_change_k = t - reference_temp_k
    resonance_m = reference_wavelength_m + reference_tuning_slope_m_per_k * (
        temp_change_k + 0.5 * relative_tuning_curvature_per_k * temp_change_k * temp_change_k
    )
    tuning_slope_m_per_k = reference_tuning_slope_m_per_k * (
        1.0 + relative_tuning_curvature_per_k * temp_change_k
    )
    absorbed_power_w = 0.0
    d_absorbed_power_dtemp_w_per_k = 0.0
    # every laser line contributes to the total heat at this same trial temperature.
    for line_index in range(n_lines):
        # each line stores its wavelength, half-width and on-resonance absorbed power together.
        line_offset = ring_offset + LZ_HEADER + LZ_PER_LINE * line_index
        half_linewidth_m = packed_f[line_offset + LZ_LINE_HWHM]
        detuning_in_half_widths = (resonance_m - packed_f[line_offset + LZ_LINE_LAM]) / half_linewidth_m
        denominator = 1.0 + detuning_in_half_widths * detuning_in_half_widths
        peak_absorbed_w = packed_f[line_offset + LZ_LINE_PEAK]
        absorbed_power_w += peak_absorbed_w / denominator
        # apply the chain rule from normalized detuning through resonance wavelength to ring temperature.
        d_absorbed_power_dtemp_w_per_k -= (
            peak_absorbed_w
            * 2.0
            * detuning_in_half_widths
            / (denominator * denominator)
            / half_linewidth_m
            * tuning_slope_m_per_k
        )
    # the thermal solver needs both the total optical heating and its local temperature slope.
    return (
        absorbed_power_w,
        d_absorbed_power_dtemp_w_per_k,
    )

pabs_lorentzian = physics_lorentzian_absorption_and_temperature_slope


# a per-ring array broadcast to (B, N).
# axes - B = simulated worlds, N = rings.
def helper_broadcast_ring_values(
    v: np.ndarray,  # v = v (input units) - parameter values with optional leading world axis - type numeric ndarray, shape follows supplied parameter
    B: int,  # B = B (1) - number of worlds - type int scalar
    N: int,  # N = N (1) - number of rings - type int scalar
) -> (
    NDArray[np.float64]  # return = v (input units) - parameter expanded over the requested axes - type float64 ndarray, shape (B,N)
):
    # trailing ring values may be shared across any number of devices.
    return np.broadcast_to(np.atleast_2d(v) if v.ndim < 2 else v, (B, N))

# a per-(ring, line) array broadcast to (B, N, L) - a 1-D array is per ring.
# axes - B = simulated worlds, L = laser lines, N = rings.
def helper_broadcast_ring_line_values(
    v: np.ndarray,  # v = v (input units) - parameter values with optional leading world axis - type numeric ndarray, shape follows supplied parameter
    B: int,  # B = B (1) - number of worlds - type int scalar
    N: int,  # N = N (1) - number of rings - type int scalar
    L: int,  # L = L (1) - number of laser lines - type int scalar
) -> (
    NDArray[np.float64]  # return = v (input units) - parameter expanded over the requested axes - type float64 ndarray, shape (B,N,L)
):
    # a vector indexes rings and therefore needs a trailing singleton laser axis.
    v = v if v.ndim == 3 else (v[None] if v.ndim == 2 else np.reshape(v, (1, -1, 1) if v.ndim == 1 else (1, 1, 1)))
    # broadcast the normalized axes without duplicating a shared parameter.
    return np.broadcast_to(v, (B, N, L))


# pack pabs_lorentzian parameters - per ring (B|1, N) arrays and per (ring, line) (B|1, N, L) arrays.
# returns (packed_f (B, nf) float64, packed_i (B, 1+N) int64).
# every ring has the same L (use p_peak 0 to pad).
# axes - B = simulated worlds, F = packed floating fields, I = packed integer fields, L = laser lines, N = rings.
def packing_lorentzian_absorption_parameters(
    lam_ref_m: ArrayLike,  # lam_ref_m = lambda_ref (m) - cold resonance at the reference temperature - type array-like, broadcast to (B,N)
    t_ref_k: ArrayLike,  # t_ref_k = T_ref (K) - reference temperature of the cold resonance - type array-like, broadcast to (B,N)
    s0_m_per_k: ArrayLike,  # s0_m_per_k = s_0 (m/K) - first-order tuning slope at the reference temperature - type array-like, broadcast to (B,N)
    beta_per_k: ArrayLike,  # beta_per_k = beta (1/K) - relative change of tuning slope per kelvin - type array-like, broadcast to (B,N)
    lam_line_m: ArrayLike,  # lam_line_m = lambda_L (m) - laser vacuum wavelength - type array-like, broadcast to (B,N,L)
    hwhm_m: ArrayLike,  # hwhm_m = gamma_lambda (m) - half-width at half maximum of the absorption peak - type array-like, broadcast to (B,N,L)
    p_peak_w: ArrayLike,  # p_peak_w = P_peak (W) - on-resonance absorption for each ring and laser line - type array-like, broadcast to (B,N,L)
) -> tuple[
    NDArray[np.float64],  # return[0] = theta_opt (mixed units) - floating optical parameters and callback workspace - type float64 ndarray, shape (B,F)
    NDArray[np.int64],  # return[1] = i_opt (1) - integer optical dimensions and offsets - type int64 ndarray, shape (B,I)
]:
    lp = np.asarray(p_peak_w, np.float64)
    # a vector of peak powers represents one line per ring in a single world.
    if lp.ndim == 1:
        lp = lp[None, :, None]
    elif lp.ndim == 2:
        # a ring-by-line matrix represents one device with multiple laser lines.
        lp = lp[None]
    B0, N, L = lp.shape
    per_ring = [np.asarray(v, np.float64) for v in (lam_ref_m, t_ref_k, s0_m_per_k, beta_per_k)]
    per_line = [np.asarray(v, np.float64) for v in (lam_line_m, hwhm_m)]
    # batch rows - only the leading axis of a (B, N) per-ring or (B, N, L) per-line array counts (np.atleast_3d would turn an (N, L) array into (N, L, 1) and take N for B)
    B = max([B0] + [v.shape[0] for v in per_ring if v.ndim == 2] + [v.shape[0] for v in per_line if v.ndim == 3])

    lam_ref, t_ref, s0, beta = (helper_broadcast_ring_values(v, B, N) for v in per_ring)
    lam_l, hw = (helper_broadcast_ring_line_values(v, B, N, L) for v in per_line)
    pk = np.broadcast_to(lp, (B, N, L))
    # a positive half-width is required to normalize wavelength detuning.
    if np.any(hw <= 0):
        raise ValueError("HWHM must be > 0")
    # every ring block holds a fixed header followed by three quantities per line.
    block = LZ_HEADER + LZ_PER_LINE * L
    packed_f = np.zeros((B, N * block))
    packed_i = np.zeros((B, 1 + N), np.int64)
    packed_i[:, 0] = N
    # the offset table lets the compiled callback jump directly to one ring.
    for k in range(N):
        # fixed-size ring records start one full block apart.
        off = k * block
        packed_i[:, 1 + k] = off
        packed_f[:, off + LZ_LAM_REF] = lam_ref[:, k]
        packed_f[:, off + LZ_T_REF] = t_ref[:, k]
        packed_f[:, off + LZ_S0] = s0[:, k]
        packed_f[:, off + LZ_BETA] = beta[:, k]
        # the header records how many line contributions to sum.
        packed_f[:, off + LZ_N_LINES] = L
        # pack each laser’s wavelength and Lorentzian scale beside its peak power.
        for l in range(L):
            # skip the fixed ring header before advancing by one line record.
            j = off + LZ_HEADER + LZ_PER_LINE * l
            packed_f[:, j + LZ_LINE_LAM] = lam_l[:, k, l]
            packed_f[:, j + LZ_LINE_HWHM] = hw[:, k, l]
            packed_f[:, j + LZ_LINE_PEAK] = pk[:, k, l]
    return (
        packed_f,
        packed_i,
    )

pack_lorentzian_args = packing_lorentzian_absorption_parameters


# bound absorption and its first two temperature derivatives over all laser lines.
# line arrays have shape (..., N, L), and tuning arrays broadcast to (..., N).
# with u=(lambda_r-lambda_L)/gamma_lambda, max|dP/du|=(3*sqrt(3)/8)*P_peak.
# max|d^2P/du^2|=2*P_peak, and |dlambda_r/dT| <= |s_0|*(1+|beta|*dt_max_k).
# sum line bounds to bound the full absorption curve.
# the default 200 K excursion is an assumed margin, not a solved temperature.
# dmax checks cross-ring lag gain, and d2max safeguards the temperature solve.
# axes - L = laser lines, N = rings.
def physics_lorentzian_absorption_bounds(
    p_peak_w: ArrayLike,  # p_peak_w = P_peak (W) - on-resonance absorption for each ring and laser line - type array-like, broadcast to (...,N,L)
    hwhm_m: ArrayLike,  # hwhm_m = gamma_lambda (m) - half-width at half maximum of the absorption peak - type array-like, broadcast to (...,N,L)
    s0_m_per_k: ArrayLike,  # s0_m_per_k = s_0 (m/K) - first-order tuning slope at the reference temperature - type array-like, broadcast to (...,N)
    beta_per_k: ArrayLike = 0.0,  # beta_per_k = beta (1/K) - relative change of tuning slope per kelvin - type array-like, broadcast to (...,N)
    dt_max_k: float = 200.0,  # dt_max_k = Delta_T_max (K) - assumed maximum absolute temperature excursion - type float scalar
) -> tuple[
    np.float64 | NDArray[np.float64],  # return[0] = P_max (W) - peak powers summed over the final line axis - type float64 scalar or float64 ndarray, shape equal to the leading power axes
    np.float64 | NDArray[np.float64],  # return[1] = D_T (W/K) - upper bound on the absolute temperature slope - type float64 scalar or float64 ndarray, shape broadcast from leading line-array axes and tuning parameters
    np.float64 | NDArray[np.float64],  # return[2] = D_TT (W/K^2) - upper bound on absolute temperature curvature - type float64 scalar or float64 ndarray, shape broadcast from leading line-array axes and tuning parameters
]:
    peak_absorbed_power_w = np.asarray(p_peak_w, np.float64)
    half_linewidth_m = np.asarray(hwhm_m, np.float64)
    # absolute slopes bound either sign of the thermo-optic response.
    abs_tuning_slope_m_per_k = np.abs(np.asarray(s0_m_per_k, np.float64))
    abs_relative_tuning_curvature_per_k = np.abs(np.asarray(beta_per_k, np.float64))
    tuning_slope_bound_m_per_k = abs_tuning_slope_m_per_k * (
        1.0 + abs_relative_tuning_curvature_per_k * float(dt_max_k)
    )
    absorbed_power_bound_w = peak_absorbed_power_w.sum(axis=-1)
    # the chain rule scales the worst Lorentzian slope by the largest local wavelength sensitivity.
    power_temp_derivative_bound_w_per_k = (LORENTZ_MAX_SLOPE * peak_absorbed_power_w / half_linewidth_m).sum(
        axis=-1
    ) * tuning_slope_bound_m_per_k
    power_temp_curvature_bound_w_per_k2 = (2.0 * peak_absorbed_power_w / half_linewidth_m**2).sum(
        axis=-1
    ) * tuning_slope_bound_m_per_k**2 + (LORENTZ_MAX_SLOPE * peak_absorbed_power_w / half_linewidth_m).sum(
        axis=-1
    ) * abs_tuning_slope_m_per_k * abs_relative_tuning_curvature_per_k
    return (
        absorbed_power_bound_w,
        power_temp_derivative_bound_w_per_k,
        power_temp_curvature_bound_w_per_k2,
    )

lorentzian_bounds = physics_lorentzian_absorption_bounds


# analytic results of the single-line problem (test targets)
# Compact-model notes, section 6.4.
# the self-heating tests check the solver against these closed forms.
# find folds of y = y_o + A/(1+y^2) for one laser line.
# y = (lambda_r-lambda_L)/HWHM (1), and y_o excludes optical heating.
# folds satisfy (1+y^2)^2 + 2*A*y = 0, with y_a < y_b < 0 and y_o,a > y_o,b.
# increasing heater input jumps at y_o,a, and decreasing input jumps at y_o,b.
# return None when A <= 8/(3*sqrt(3)) or no fold pair is found.
def solver_lorentzian_fold_detunings(
    A: float,  # A = A (1) - peak self-heating wavelength shift in units of the half-width - type float scalar
) -> (
    tuple[
        float,  # return[0] = y_a (1) - lower normalized fold detuning - type float scalar
        float,  # return[1] = y_b (1) - upper normalized fold detuning - type float scalar
        float,  # return[2] = y_o_a (1) - heater detuning for the upward sweep jump - type float scalar
        float,  # return[3] = y_o_b (1) - heater detuning for the downward sweep jump - type float scalar
    ]
    | None  # return = None - no fold pair exists - type None
):
    # below or at the cusp there are no two distinct switching temperatures.
    if A <= BISTABILITY_FACTOR:
        return None
    # extrema of the heater-only detuning give the quartic for the two folds.
    polynomial_roots = np.roots([1.0, 0.0, 2.0, 2.0 * A, 1.0])
    real_fold_detunings = np.sort(polynomial_roots[np.abs(polynomial_roots.imag) < 1e-9 * max(1.0, A)].real)
    if real_fold_detunings.size < 2:
        # fewer than two real roots cannot delimit a bistable interval.
        return None
    lower_fold_detuning, upper_fold_detuning = float(real_fold_detunings[0]), float(real_fold_detunings[-1])
    # subtract the Lorentzian heating shift to recover the externally controlled detuning.
    return (
        lower_fold_detuning,
        upper_fold_detuning,
        lower_fold_detuning - A / (1.0 + lower_fold_detuning * lower_fold_detuning),
        upper_fold_detuning - A / (1.0 + upper_fold_detuning * upper_fold_detuning),
    )

lorentzian_folds = solver_lorentzian_fold_detunings


# find every equilibrium detuning of the single-line Lorentzian model.
# lambda_0 includes heater and external heating, but excludes optical heating.
# these roots include unstable equilibria - root finding alone does not select a branch.
# axes - R = real roots.
def solver_lorentzian_equilibrium_detunings(
    y_o: float,  # y_o = y_o (1) - detuning in half-widths before optical heating - type float scalar
    A: float,  # A = A (1) - peak self-heating wavelength shift in units of the half-width - type float scalar
) -> (
    NDArray[np.float64]  # return = y (1) - sorted real equilibrium detunings - type float64 ndarray, shape (R,)
):
    # clearing the Lorentzian denominator gives the cubic for all steady temperatures.
    equilibrium_roots = np.roots([1.0, -y_o, 1.0, -(y_o + A)])
    return np.sort(equilibrium_roots[np.abs(equilibrium_roots.imag) < 1e-7 * max(1.0, abs(A), abs(y_o))].real)

steady_detuning = solver_lorentzian_equilibrium_detunings


# compare optical heating strength with the Lorentzian fold threshold.
# a ratio above one permits bistability over a detuning interval.
# the ratio alone does not establish bistability at the current operating point.
def physics_lorentzian_bistability_ratio(
    g_sh_m_per_w: ArrayLike,  # g_sh_m_per_w = G_sh (m/W) - resonance wavelength shift per absorbed watt - type array-like, broadcast to (...)
    p_abs_peak_w: ArrayLike,  # p_abs_peak_w = P_peak (W) - on-resonance absorbed power - type array-like, broadcast to (...)
    fwhm_m: ArrayLike,  # fwhm_m = FWHM (m) - full width at half maximum - type array-like, broadcast to (...)
) -> (
    float | NDArray[np.float64]  # return = r (1) - strength relative to the onset of folds - type real scalar or float64 ndarray, broadcast input shape
):
    # threshold heating shifts the resonance by the cusp fraction of one linewidth.
    return np.asarray(g_sh_m_per_w) * np.asarray(p_abs_peak_w) / (0.5 * BISTABILITY_FACTOR * np.asarray(fwhm_m))

bistability_ratio = physics_lorentzian_bistability_ratio


# hold thermal state and reusable scratch arrays for Python self-heating studies.
# x stores (B, M) thermal lag states.
# p_prev and t_prev store (B, N) absorbed powers and temperatures.
# step calls implicit_step, then retains the returned state as the next continuation starting point.
class SelfHeatWork:

    # retain accepted temperatures and powers for branch continuation.
    # axes - B = simulated worlds, N = rings.
    def assembly_self_heating_workspace(
        self: SelfHeatWork,  # self = work (mixed units) - persistent lag states and last accepted ring temperatures - type SelfHeatWork
        system: ThermalSystem,  # system = thermal (mixed units) - thermal lag system and source-to-observer indexing - type ThermalSystem
        batch: int | None = None,  # batch = B (1) - optional number of simulated devices - type int | None
        t_init_k: ArrayLike = None,  # t_init_k = T_init (K) - initial temperatures for selecting the starting branch - type array-like, broadcast to (B,N)
    ) -> (
        None  # return = None - allocate lag and optical history buffers - type None
    ):
        # use the thermal model’s device count unless a study requests an explicit batch.
        B = system.helper_world_count if batch is None else int(batch)
        self.system = system
        self.x = system.helper_zero_state(B)
        n, O = system.n_rings, system.helper_observer_count
        self.p_prev = np.zeros((B, n))
        # supplied temperatures select the initial branch independently for each ring.
        self.t_prev = np.zeros((B, n)) if t_init_k is None else np.broadcast_to(
            np.asarray(t_init_k, np.float64), (B, n)).copy()
        # separate observer work buffers from the per-ring optical results.
        self.g_work = np.zeros((B, O))
        self.t_obs = np.zeros((B, O))
        self.p_new = np.zeros((B, n))
        self.t_ring = np.zeros((B, n))
        self.n_iter = np.zeros((B, n), np.int64)
        self._h = None

    __init__ = assembly_self_heating_workspace

    # advance a substep of h_s seconds and return the number of unconverged rings.
    # cache thermal coefficients for the current step length.
    # u0 and u1 are source powers in W at the substep endpoints.
    # t_base_k is the end-of-step base temperature in K.
    # use the previous returned temperatures to continue the branch.
    # commit_fn updates optical input for downstream rings.
    # axes - B = simulated worlds, F = packed floating fields, I = packed integer fields, N = rings, S = heat sources.
    def solver_coupled_temperature_step(
        self: SelfHeatWork,  # self = work (mixed units) - persistent lag states and last accepted ring temperatures - type SelfHeatWork
        h_s: float,  # h_s = h (s) - thermal substep duration - type float scalar
        u0: ArrayLike,  # u0 = u_0 (W) - prescribed source power at the start of the step - type array-like, broadcast to (B|1,S)
        u1: ArrayLike,  # u1 = u_1 (W) - prescribed source power at the end of the step - type array-like, broadcast to (B|1,S)
        t_base_k: ArrayLike,  # t_base_k = T_base (K) - base temperature at the end of the substep - type array-like, broadcast to (B|1,)
        pabs_fn: Callable,  # pabs_fn = P_abs (W, W/K) - callback for absorption and its temperature derivative at a trial temperature - type callable
        packed_f: ArrayLike,  # packed_f = theta_opt (mixed units) - floating optical parameters and callback workspace - type array-like, broadcast to (B|1,F)
        packed_i: ArrayLike,  # packed_i = i_opt (1) - integer optical dimensions and offsets - type array-like, broadcast to (B|1,I)
        p_max: ArrayLike,  # p_max = P_max (W) - upper bound on absorbed power - type array-like, broadcast to (B|1,N)
        d2p_max: ArrayLike,  # d2p_max = K_2P (W/K^2) - upper bound on absolute absorption curvature with temperature - type array-like, broadcast to (B|1,N)
        *,
        commit_fn: Callable = commit_none,  # commit_fn = commit (mixed units) - callback that propagates optical inputs after accepting a ring temperature - type callable
        bus_order: ArrayLike = None,  # bus_order = k_bus (1) - ring indices in forward optical propagation order or physical index order by default - type array-like, broadcast to (N,)
        src_scale: ArrayLike = None,  # src_scale = s_u (1) - gain multiplier for each heat source or unit scales by default - type array-like, broadcast to (B|1,S)
        max_iter: int = 10,  # max_iter = n_max (1) - maximum iterations per ring solve - type int scalar
        tol_k: float = 1e-9,  # tol_k = epsilon_T (K) - residual or bracket tolerance - type float scalar
        n_outer: int = 1,  # n_outer = n_sweep (1) - requested number of optical crosstalk sweeps within one step - type int scalar
    ) -> (
        int  # return = n_fail (1) - number of unconverged ring solves - type int scalar
    ):
        thermal_system = self.system
        # a changed step length changes the lag decay and both endpoint gains.
        if self._h != h_s:
            self._coef = thermal_system.physics_linear_hold_coefficients(h_s)
            self._h = h_s
        a, c0, c1 = self._coef
        # the default optical path visits the rings in physical index order.
        ring_bus_order = np.arange(thermal_system.n_rings,
            dtype=np.int64) if bus_order is None else np.asarray(bus_order, np.int64)
        # unit source scales preserve the fitted thermal gains.
        source_scales = (
            np.ones((1, thermal_system.helper_source_count)) if src_scale is None
            else np.atleast_2d(np.asarray(src_scale, np.float64))
        )
        failure_count = solver_coupled_temperatures(self.x, a, c0, c1, thermal_system.lag_src, thermal_system.lag_obs,
            thermal_system.lag_kind, thermal_system.opt_lags, thermal_system.obs_ptr, thermal_system.src_ring,
                           thermal_system.ring_obs, ring_bus_order,
                           np.atleast_2d(u0), np.atleast_2d(u1), source_scales, np.atleast_1d(np.asarray(t_base_k,
                               np.float64)),
                           self.p_prev, self.t_prev, pabs_fn, commit_fn, np.atleast_2d(packed_f),
                           np.atleast_2d(packed_i), np.atleast_2d(p_max), np.atleast_2d(d2p_max), max_iter, tol_k,
                           n_outer, self.g_work, self.t_obs, self.p_new, self.t_ring, self.n_iter)
        # retain the accepted heat and temperature to continue the same history.
        self.p_prev[:] = self.p_new
        self.t_prev[:] = self.t_ring
        return int(failure_count)

    step = solver_coupled_temperature_step

__all__ += [
    "solver_ring_temperature",
    "helper_keep_optical_inputs",
    "solver_coupled_temperatures_for_world",
    "solver_coupled_temperatures",
    "solver_temperature_history_for_world",
    "physics_lorentzian_absorption_and_temperature_slope",
    "packing_lorentzian_absorption_parameters",
    "physics_lorentzian_absorption_bounds",
    "solver_lorentzian_fold_detunings",
    "solver_lorentzian_equilibrium_detunings",
    "physics_lorentzian_bistability_ratio",
]
