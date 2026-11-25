# model how heat sources change ring, heater, photodiode, and sensor temperatures.
#
# the whole model is one equation used many times.
# a lag is one temperature contribution x (K) from one heat source to one temperature readout.
# given source power P (W), it follows tau*dx/dt = r*P - x.
# gain r (K/W) sets the settled rise and tau (s) sets the warmup time.
# a source (S) is a place where power enters. an observer (O) is a place where a temperature is read.
# an observer temperature is T_base plus the sum of the lags that end at it.
# over one step of duration h, x_next = a*x + c_start*P_start + c_end*P_end, with a = exp(-h/tau).
#
# follow one heat source to one temperature readout:
# - assemble assigns the path gains and time constants from InterimRom or a saved ThermalRom.
# - system.coefficients prepares the update weights for the chosen step duration.
# - physics_advance_thermal_state_for_world keeps the warmth that survives and adds new heating.
# - physics_write_observer_temperatures_for_world adds the contributions at each readout to its base.
#
# the file has ten sections, in this order.
# each section starts with a one-line section comment.
# 1. one lag over one step - the weights a, c_start, and c_end. core math.
# 2. advance and read the lags - the update equation in each form the engines call. core math and batch wrappers.
# 3. saved thermal model - ThermalRom, its files, and its quality checks. storage.
# 4. layout and model records - LayoutSites, ThermalScales, and InterimRom. inputs.
# 5. distance laws - warmup time and settled gain against separation. modeling choices.
# 6. assembled system - ThermalSystem, the packed list of lags that the engines read. storage.
# 7. assembly - choose r and tau for every lag, apply chip uncertainty, then pack. modeling choices.
# 8. known power histories - simulate advances the lags for a supplied power history.
# 9. response queries and checks - step, ramp, frequency, settled gain, reciprocity, and the cross-ring bound.
# 10. previous names - aliases only.
# in section 7, assemble is the entry point. read it first, then the stage functions above it.
#
# simulate takes a known power history. in a coupled tick, the reference engine
# uses selfheat.implicit_step and the compiled engine uses kernel._implicit_step_eng
# to solve absorbed power and ring temperature together.
#
# B is worlds, M is lag states, S is sources, and O is observers.
# residues and time constants have shape (B, M).
# lag_src and lag_obs map each lag to its input and readout across all worlds.
# obs_ptr gives the start and end of each readout's contiguous range of lags.
# the analytic update is exact for the assumed power history within the step.
# source notes are in docs/research/04_thermal_and_control.md. [CO22-TCPMT] means Coenen et al., IEEE TCPMT 2022.
# its measured and inferred numbers form the interim model's main reference.
# detailed implementation assumptions are in docs/THERMAL_IMPLEMENTATION_NOTES.md.



from __future__ import annotations

import hashlib
import json
import math
import warnings
from functools import partial
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeAlias
from numpy.typing import ArrayLike, NDArray

import numpy as np
from numba import njit

__all__ = [
    # constants
    "SCHEMA_ROM", "SOURCE_KINDS", "OBSERVER_KINDS", "SCALING_HYPOTHESES",
    "SRC_HEATER", "SRC_WG", "SRC_JUNCTION", "SRC_DRIVER", "SRC_AGGRESSOR",
    "OBS_RING", "OBS_HEATER", "OBS_PD", "OBS_TEMP",
    "LAG_EXOGENOUS", "LAG_OPT_SELF", "LAG_OPT_CROSS_RING", "LAG_OPT_OTHER",
    "PAIR_SELF", "PAIR_CROSS", "PAIR_SITE", "LAYER_FAST", "LAYER_SLOW",
    "XT_TABLE_D_UM", "XT_TABLE_TAU_S", "DEFAULT_TAU_SPLIT_S",
    # data classes
    "RomIntegrityError", "ThermalRom", "InterimRom", "LayoutSites", "ThermalScales", "ThermalSystem",
    # artefact I/O
    "load_rom", "save_rom", "rom_path", "sha256_file", "rom_validation_numbers",
    # assembly
    "assemble", "interim_three_pole", "crosstalk_tau_s", "site_fraction",
    # coefficients and kernels
    "foh_scalar", "foh_coefficients", "zoh_coefficients",
    "thermal_step_b", "thermal_step", "thermal_step_h_b", "thermal_outputs_b", "thermal_outputs",
    "thermal_advance_b", "thermal_advance",
    "observer_split_b", "simulate",
    # analysis helpers
    "frequency_response", "step_response", "ramp_response", "dc_gain", "state_space",
    "reciprocity_error", "cross_optical_lag_gain", "assert_cross_lag_gain", "selfheat_dc_k_per_w",
    "k_temp_residue_scale",
]

SCHEMA_ROM = "thermal_rom/1"
# kind names of the sources (heat inputs) and observers (temperatures read).
# the codes below index these tuples
SOURCE_KINDS = ("heater", "waveguide_absorption", "junction", "driver", "aggressor")
OBSERVER_KINDS = ("ring_mode_weighted", "heater_resistor", "monitor_pd", "temp_sensor")
SRC_HEATER, SRC_WG, SRC_JUNCTION, SRC_DRIVER, SRC_AGGRESSOR = 0, 1, 2, 3, 4
OBS_RING, OBS_HEATER, OBS_PD, OBS_TEMP = 0, 1, 2, 3
# lag kinds - which input drives the lag in mrm.plant.selfheat.implicit_step (thermal_step ignores the kind).
LAG_EXOGENOUS = 0        # heater, junction, driver, aggressor - u0/u1 given by the caller
LAG_OPT_SELF = 1         # waveguide absorption of ring k -> ring k mode temperature - solved implicitly
LAG_OPT_CROSS_RING = 2   # waveguide absorption of ring j -> ring k != j mode temperature - lags one substep
LAG_OPT_OTHER = 3        # waveguide absorption -> any non-ring observer - uses the newly solved P_abs
# pair types - source and observer on the same ring, on two different rings, or one of them a free site
PAIR_SELF, PAIR_CROSS, PAIR_SITE = 0, 1, 2
# layers of the time-constant scaling - ring and oxide (fast), die and package (slow)
LAYER_FAST, LAYER_SLOW = 0, 1
# scaling hypotheses of ThermalScales.scaling_hypothesis.
# an int-coded per-world array uses these indices.
SCALING_HYPOTHESES = ("resistance", "capacitance", "independent")

# assign lags below this time constant to the fast ring and oxide layer.
# assign the remaining lags to the slow die and package layer.
# the source ranges are 1-124 us and 0.1-9 ms in thermal notes section 2.3.
# assumed - the boundary between L1 and L2.
DEFAULT_TAU_SPLIT_S = 3.0e-4

# sidecar acceptance thresholds of a FEM ROM (the extraction targets, docs/ARCHITECTURE.md section 5.3).
ROM_STEP_ERR_MAX = 0.02          # max |dT - dT_FEM| / dT_ss, 10 ns .. 10 ms
ROM_FREQ_DB_MAX = 0.3            # |H| error, DC .. 1 MHz
ROM_FREQ_DEG_MAX = 3.0           # phase error, DC .. 1 MHz
ROM_SUPERPOSITION_MAX = 0.10     # 8-ring bank superposition error on the crosstalk residues
ROM_TAU_FAST_MAX_S = 1.0e-6      # the ROM must carry a pole below 1 us (waveguide-core self-heating)
ROM_TAU_SLOW_MIN_S = 1.0e-3      # the shared tau set must include the millisecond crosstalk poles

# crosstalk times follow Coenen et al. 2022, Table II, for neighbours 1 through 7.
# the ring pitch is 100 um and the first source-to-victim distance is 60 um.
# the tabulated values use undercut - values without undercut differ by at most 1.3 percent.
# the distance convention is heat source to victim, as in thermal notes section 2.5.
# subtracting two inferred 20 um ring radii gives 60 um at a 100 um pitch.
# crosstalk_tau_s takes them in that convention, and the interim assembly converts its centre distances (_interim_lags).
# the distances 160..660 um assume that the 100 um pitch continues (inferred).
XT_TABLE_D_UM = np.array([60.0, 160.0, 260.0, 360.0, 460.0, 560.0, 660.0])
XT_TABLE_TAU_S = np.array([4.14, 5.67, 6.57, 7.21, 7.82, 8.48, 8.99]) * 1e-3

_FOH_SERIES_Y = 1.0e-3   # below this y = h/tau the FOH coefficients use their Taylor series (numerics only)


# array types specify storage precision - each parameter states its physical shape.
Float64Array: TypeAlias = NDArray[np.float64]
Complex128Array: TypeAlias = NDArray[np.complex128]
Int64Array: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
NumericArray: TypeAlias = NDArray[np.generic]
RealValue: TypeAlias = float | np.floating | NDArray[np.floating]


# report a thermal ROM schema, checksum, or array-integrity failure.
class RomIntegrityError(ValueError):
    pass


# section 1 - one lag over one step

# in the full picture - the simulator moves time forward in steps of duration h (s).
# this function answers one question for one lag - after one step, how much old warmth is left, and how much new warmth arrived?
# a lag is one temperature contribution x (K) that one heat source adds at one place where temperature is read.
# with a constant source power P (W), x settles at r*P. r (K/W) is the gain of the lag and tau (s) is how long the settling takes.
# this function does not use r. it returns three pure numbers that depend only on y = h/tau.
# a (1) is the fraction of the old x that is still there after the step.
# b_start (1) and b_end (1) say how much of the settled rise r*P develops during the step.
# the power can change during the step, so b_start weights the power at the start and b_end weights the power at the end.
# example - tau = h = 10 us gives a = 0.368, b_start = 0.264, and b_end = 0.368.
# b_end is larger because heat that arrives late in the step has had less time to leak away.
# a caller multiplies b_start and b_end by r to get the temperature rise per watt in K/W.
# integrate one thermal lag with linearly varying source power.
# for tau*dx/dt=-x+r*P, the update is x_next=a*x+r*(b_start*P_start+b_end*P_end).
# at time t within the step, P(t) = (1-t/h)*P_start + (t/h)*P_end.
# integrate each power weight against exp(-(h-t)/tau)/tau from t=0 to t=h.
# with y=h/tau, a=exp(-y), b_end=1-(1-a)/y, and b_start=(1-a)/y-a.
# use the Taylor forms at small y to avoid cancellation.
# a nonpositive step leaves the state unchanged.
@njit(cache=True)
def physics_linear_hold_lag_weights(
    tau: float,  # tau = tau (s) - thermal lag time constant - type float scalar
    h: float,  # h = h (s) - thermal step duration - type float scalar
) -> tuple[
    float,  # return[0] = a (1) - retained state fraction - type float scalar
    float,  # return[1] = b_start (1) - weight on starting source power - type float scalar
    float,  # return[2] = b_end (1) - weight on ending source power - type float scalar
]:
    # treat a nonpositive duration as a zero-length step so the stored rise survives and neither endpoint adds heat.
    if h <= 0.0:
        return (
            1.0,
            0.0,
            0.0,
        )
    # y = h/tau measures the step in units of one thermal time constant.
    step_over_time_constant = h / tau
    # a = exp(-y) is the fraction of the initial temperature contribution that remains.
    state_retention = math.exp(-step_over_time_constant)
    # for tiny y, both weights start near y/2 because the power ramps have equal area.
    # use their series directly because subtracting nearly equal terms loses precision.
    if step_over_time_constant < _FOH_SERIES_Y:
        # reuse powers of the dimensionless step h/tau in both Taylor expansions.
        step_ratio_squared = step_over_time_constant * step_over_time_constant
        step_ratio_cubed = step_ratio_squared * step_over_time_constant
        step_ratio_fourth = step_ratio_cubed * step_over_time_constant
        step_ratio_fifth = step_ratio_fourth * step_over_time_constant
        # expand b_end = 1 - mean_retention through y^5.
        end_power_weight = (
            step_over_time_constant / 2.0
            - step_ratio_squared / 6.0
            + step_ratio_cubed / 24.0
            - step_ratio_fourth / 120.0
            + step_ratio_fifth / 720.0
        )
        # expand b_start = mean_retention - a through y^5.
        start_power_weight = (
            step_over_time_constant / 2.0
            - step_ratio_squared / 3.0
            + step_ratio_cubed / 8.0
            - step_ratio_fourth / 30.0
            + step_ratio_fifth / 144.0
        )
        # the series already gives both weights without subtracting nearly equal numbers.
        return (
            state_retention,
            start_power_weight,
            end_power_weight,
        )
    # mean_retention = (1 - exp(-y))/y is the average of the exponential over the step.
    # expm1 computes the numerator accurately when y is small.
    mean_state_retention = -math.expm1(-step_over_time_constant) / step_over_time_constant
    # the two power weights sum to 1-a, the fraction developed under constant power.
    return (
        state_retention,
        mean_state_retention - state_retention,
        1.0 - mean_state_retention,
    )


# in the full picture - this function runs once when the step duration h (s) is chosen, not on every step.
# it gives every lag the three numbers that each later step reuses.
# a lag is one temperature contribution x (K) from one heat source. tau (s) is its warmup time and r (K/W) is its settled rise per watt.
# a (1) is the fraction of x that survives one step.
# c_start (K/W) and c_end (K/W) are the rise that one watt adds during one step, for the power at the start and at the end of the step.
# one step is then x_next = a*x + c_start*P_start + c_end*P_end, with x in K and P in W.
# this is the array form of physics_linear_hold_lag_weights, with r multiplied in.
# ThermalSystem.physics_linear_hold_coefficients calls it with the tau and r of every lag.
# calculate exact thermal update coefficients for linearly varying source power.
# x_next=a*x+c_start*P_start+c_end*P_end.
# multiply each scalar hold weight by its thermal residue in K/W.
# a zero-duration step retains every state.
# calculate how much temperature remains and how much the start and end powers add over one thermal time step.
def physics_linear_hold_coefficients(
    taus_s: ArrayLike,  # taus_s = tau (s) - thermal time constants - type array-like, broadcast input shape
    res_k_per_w: ArrayLike,  # res_k_per_w = r (K/W) - thermal residues - type array-like, broadcast input shape
    h_s: float,  # h_s = h (s) - thermal step duration - type float scalar
) -> tuple[
    Float64Array,  # return[0] = a (1) - retained state fractions - type float64 ndarray, shape broadcast shape of time constants and residues
    Float64Array,  # return[1] = c_start (K/W) - starting-power coefficients - type float64 ndarray, shape broadcast shape of time constants and residues
    Float64Array,  # return[2] = c_end (K/W) - ending-power coefficients - type float64 ndarray, shape broadcast shape of time constants and residues
]:
    time_constants_s = np.asarray(taus_s, dtype=np.float64)
    thermal_residues_k_per_w = np.asarray(res_k_per_w, dtype=np.float64)
    step_duration_s = float(h_s)
    # a negative duration would integrate cooling backward in time.
    if step_duration_s < 0.0:
        raise ValueError(f"step h_s must be >= 0, got {h_s}")
    # each decaying thermal mode needs a positive time constant.
    if np.any(~(time_constants_s > 0.0)):
        raise ValueError("every tau must be > 0")
    if step_duration_s == 0.0:
        # a = 1 and both power coefficients are zero because no time elapses.
        coefficient_shape = np.broadcast(time_constants_s, thermal_residues_k_per_w).shape
        state_retention = np.ones(coefficient_shape)
        # zero duration leaves each mode at its starting temperature.
        return (
            state_retention,
            np.zeros_like(state_retention),
            np.zeros_like(state_retention),
        )
    # compute the scalar lag formulas for each time constant tau.
    step_over_time_constant = step_duration_s / time_constants_s
    state_retention = np.exp(-step_over_time_constant)
    # mean_retention = (1 - exp(-y))/y, with y = h/tau.
    mean_state_retention = -np.expm1(-step_over_time_constant) / step_over_time_constant
    # the late ramp gets b_end = 1 - mean_retention.
    # subtract b_end from 1-a to get b_start = mean_retention - a.
    end_power_weight = 1.0 - mean_state_retention
    start_power_weight = mean_state_retention - state_retention
    # replace the cancellation-prone weights with their fifth-order Taylor series for small y.
    use_small_step_series = step_over_time_constant < _FOH_SERIES_Y
    # replace cancellation-prone differences only for the modes with a tiny step relative to tau.
    if np.any(use_small_step_series):
        small_step_ratios = step_over_time_constant[use_small_step_series]
        # the small-step series recovers the weight on heat applied at the end of the interval.
        end_power_weight[use_small_step_series] = (
            small_step_ratios / 2.0
            - small_step_ratios**2 / 6.0
            + small_step_ratios**3 / 24.0
            - small_step_ratios**4 / 120.0
            + small_step_ratios**5 / 720.0
        )
        start_power_weight[use_small_step_series] = (
            small_step_ratios / 2.0
            - small_step_ratios**2 / 3.0
            + small_step_ratios**3 / 8.0
            - small_step_ratios**4 / 30.0
            + small_step_ratios**5 / 144.0
        )
    # align the weight arrays before scaling each endpoint weight by its residue to get c = r*b (K/W).
    state_retention, start_power_weight, end_power_weight = np.broadcast_arrays(
        state_retention, start_power_weight, end_power_weight
    )
    return (
        state_retention.copy(),
        start_power_weight * thermal_residues_k_per_w,
        end_power_weight * thermal_residues_k_per_w,
    )


# in the full picture - this is the same preparation as physics_linear_hold_coefficients for a power that stays constant across the step.
# a (1) is the fraction of a lag's temperature contribution x (K) that survives one step of duration h (s).
# c (K/W) is the rise that one watt, held for the whole step, adds by the end of the step.
# calculate exact thermal update coefficients for constant source power.
# x_next=a*x+c*P, with c=r*(1-a).
# x_next is x at h
# (todo) why are there so many wrapper functions? get to the core of things. remove this in the future
# x
def physics_constant_hold_coefficients(
    taus_s: ArrayLike,  # taus_s = tau (s) - thermal time constants - type array-like, broadcast input shape
    res_k_per_w: ArrayLike,  # res_k_per_w = r (K/W) - thermal residues - type array-like, broadcast input shape
    h_s: float,  # h_s = h (s) - thermal step duration - type float scalar
) -> tuple[
    Float64Array,  # return[0] = a (1) - retained state fractions - type float64 ndarray, shape broadcast input shape
    Float64Array,  # return[1] = c (K/W) - constant-power coefficients - type float64 ndarray, shape broadcast input shape
]:
    state_retention, start_power_gain_k_per_w, end_power_gain_k_per_w = physics_linear_hold_coefficients(
        taus_s, res_k_per_w, h_s
    )
    # P_start = P_end = P, so the two power terms combine to (c_start+c_end)*P.
    return (
        state_retention,
        start_power_gain_k_per_w + end_power_gain_k_per_w,
    )


# section 2 - advance and read the lags
# every function in this section applies one of two equations.
# update - x_next = a*x + s_source*(c_start*P_start + c_end*P_end).
# readout - T_observer = T_base + sum of x over that observer's lags.
# the *_for_world functions hold the equations for one world.
# the plural functions check shapes and loop the same equations over B worlds.
# function                                             | equations          | weights
# physics_advance_thermal_state_for_world                      | update             | prepared a, c_start, c_end
# physics_advance_thermal_state_for_world_with_step_size       | update             | computed from tau and h per lag
# physics_write_observer_temperatures_for_world                | readout            | none
# physics_advance_thermal_state_and_temperatures_for_world     | update and readout | prepared a, c_start, c_end
# physics_separate_known_heat_and_source_gain_for_observer     | update and readout | prepared, with one source power left unknown

# in the full picture - this is the thermal update of one simulated chip (one world) for one time step.
# the thermal state is a list of M lags. lag m is one temperature contribution x[m] (K) from one heat source.
# in one step, every lag keeps part of its old value and gains new warmth from the power of its own source.
# x_next[m] = a[m]*x[m] + s*(c_start[m]*P_start + c_end[m]*P_end).
# P_start and P_end (W) are the power of the source that drives lag m, at the start and at the end of the step.
# a (1), c_start (K/W), and c_end (K/W) come from physics_linear_hold_coefficients for this step duration.
# lag_source_indices[m] names the source that drives lag m. several lags can share one source.
# s (1) multiplies the new warmth from one source. it is 1 unless a caller rescales that source's gain.
# example - a = 0.9, x = 2 K, c_start = c_end = 50 K/W, and 1 mW at both ends give 0.9*2 + 100*0.001 = 1.9 K.
# this function does not produce temperatures. physics_write_observer_temperatures_for_world sums the lags into temperatures.
# advance one world's lag states with weights prepared for this step duration.
# the variants below reuse this update to handle other execution paths.
# x
# axes - S = heat sources, M = thermal lags.
@njit(cache=True)
def physics_advance_thermal_state_for_world(
    state_rise_k: Float64Array,  # state_rise_k = x (K) - lag-state temperature rises updated in place - type float64 ndarray, shape (M,)
    state_retention: Float64Array,  # state_retention = a (1) - retained thermal state fractions - type float64 ndarray, shape (M,)
    start_power_gain_k_per_w: Float64Array,  # start_power_gain_k_per_w = c0 (K/W) - starting-power gain - type float64 ndarray, shape (M,)
    end_power_gain_k_per_w: Float64Array,  # end_power_gain_k_per_w = c1 (K/W) - ending-power gain - type float64 ndarray, shape (M,)
    lag_source_indices: Int64Array,  # lag_source_indices = s_m (1) - source index driving each lag - type int64 ndarray, shape (M,)
    start_source_power_w: Float64Array,  # start_source_power_w = P_start (W) - source powers at the step boundary - type float64 ndarray, shape (S,)
    end_source_power_w: Float64Array,  # end_source_power_w = P_end (W) - source powers at the step boundary - type float64 ndarray, shape (S,)
    source_residue_scale: Float64Array,  # source_residue_scale = s_source (1) - per-source residue multipliers - type float64 ndarray, shape (S,)
) -> (
    None  # return = None - update state_rise_k in kelvin - type None
):
    for lag_index in range(state_rise_k.shape[0]):
        # find the heat source that drives this lag.
        source_index = lag_source_indices[lag_index]
        # old warmth that survives the step - a*x (K).
        retained_rise_k = state_retention[lag_index] * state_rise_k[lag_index]
        # new warmth from this step's power - K/W times W gives K.
        added_rise_k = source_residue_scale[source_index] * (
            start_power_gain_k_per_w[lag_index] * start_source_power_w[source_index]
            + end_power_gain_k_per_w[lag_index] * end_source_power_w[source_index]
        )
        state_rise_k[lag_index] = retained_rise_k + added_rise_k


# in the full picture - this is the same one-world, one-step update as physics_advance_thermal_state_for_world.
# use it when no weights were prepared for the step duration h (s). verification code calls it.
# each lag is one temperature contribution x (K) with warmup time tau (s) and settled rise per watt r (K/W).
# the weights a, b_start, and b_end (1) come from physics_linear_hold_lag_weights inside the loop.
# use the same lag update when the caller supplies a step duration.
# calculate the weights inside the loop, once for each lag's time constant.
# x
# axes - S = heat sources, M = thermal lags.
@njit(cache=True)
def physics_advance_thermal_state_for_world_with_step_size(
    state_rise_k: Float64Array,  # state_rise_k = x (K) - lag-state temperature rises updated in place - type float64 ndarray, shape (M,)
    time_constants_s: Float64Array,  # time_constants_s = tau (s) - thermal time constants - type float64 ndarray, shape (M,)
    thermal_residues_k_per_w: Float64Array,  # thermal_residues_k_per_w = r (K/W) - thermal residues - type float64 ndarray, shape (M,)
    lag_source_indices: Int64Array,  # lag_source_indices = s_m (1) - source index driving each lag - type int64 ndarray, shape (M,)
    start_source_power_w: Float64Array,  # start_source_power_w = P_start (W) - source powers at the step boundary - type float64 ndarray, shape (S,)
    end_source_power_w: Float64Array,  # end_source_power_w = P_end (W) - source powers at the step boundary - type float64 ndarray, shape (S,)
    source_residue_scale: Float64Array,  # source_residue_scale = s_source (1) - per-source residue multipliers - type float64 ndarray, shape (S,)
    step_duration_s: float,  # step_duration_s = h (s) - thermal step duration - type float scalar
) -> (
    None  # return = None - update state_rise_k in kelvin - type None
):
    # update each lag with the power of its own driving source.
    for lag_index in range(state_rise_k.shape[0]):
        source_index = lag_source_indices[lag_index]
        state_retention, start_power_weight, end_power_weight = physics_linear_hold_lag_weights(
            time_constants_s[lag_index], step_duration_s
        )
        # old warmth that survives the step - a*x (K).
        retained_rise_k = state_retention * state_rise_k[lag_index]
        # settled rise per watt of this lag, with the caller's source multiplier - s*r (K/W).
        scaled_gain_k_per_w = source_residue_scale[source_index] * thermal_residues_k_per_w[lag_index]
        # new warmth - K/W times the weighted power in W gives K.
        added_rise_k = scaled_gain_k_per_w * (
            start_power_weight * start_source_power_w[source_index]
            + end_power_weight * end_source_power_w[source_index]
        )
        state_rise_k[lag_index] = retained_rise_k + added_rise_k


# in the full picture - this turns the lag states into the temperatures that the rest of the simulator uses.
# an observer is a place where temperature is read - a ring waveguide, a heater, a photodiode, or a sensor.
# a lag is one temperature contribution x (K) that one heat source adds at one observer.
# T_observer (K) = T_base (K) + the sum of every lag x[m] (K) that ends at that observer.
# T_base is the temperature of the chip with every heat source off.
# the lags are stored sorted by observer, so each observer owns one unbroken run of the lag array.
# observer o owns lags boundaries[o] up to boundaries[o+1]-1.
# example - boundaries = [0, 2, 3] means that observer 0 sums lags 0 and 1, and observer 1 reads lag 2.
# sum one world's thermal states into absolute observer temperatures.
# each observer reads the contiguous lag range in observer_lag_boundaries.
# x
# axes - O = temperature observers, M = thermal lags.
@njit(cache=True)
def physics_write_observer_temperatures_for_world(
    state_rise_k: Float64Array,  # state_rise_k = x (K) - lag-state temperature rises updated in place - type float64 ndarray, shape (M,)
    observer_lag_boundaries: Int64Array,  # observer_lag_boundaries = p_observer (1) - lag-array boundaries of each observer - type int64 ndarray, shape (O+1,)
    base_temperature_k: float,  # base_temperature_k = T_base (K) - base temperature - type float scalar
    out_observer_temperature_k: Float64Array,  # out_observer_temperature_k = T_observer (K) - absolute observer temperatures written in place - type float64 ndarray, shape (O,)
) -> (
    None  # return = None - update out_observer_temperature_k in kelvin - type None
):
    for observer_index in range(out_observer_temperature_k.shape[0]):
        # T_observer = T_base + sum(x_m) over the lags that reach this observer.
        temperature_k = base_temperature_k
        # an observer reads the sum of the temperature rises assigned to its contiguous lag range.
        for lag_index in range(observer_lag_boundaries[observer_index], observer_lag_boundaries[observer_index + 1]):
            temperature_k += state_rise_k[lag_index]
        out_observer_temperature_k[observer_index] = temperature_k


# row of world b in an array with n rows - b, or 0 when the row is shared by every world (n = 1).
# x
# (todo) many functions are just replicating or updating state, might as well move core functions to diff folder
# x
@njit(cache=True)
def helper_shared_or_world_row_index(
    world_index: int,  # world_index = b (1) - world index - type int scalar
    row_count: int,  # row_count = row_count (1) - number of shared or per-world rows - type int scalar
) -> (
    int  # return = row (1) - world row or shared row zero - type int scalar
):
    # use row zero for a parameter shared across the batch and the chip's own row for a per-world parameter.
    return world_index if row_count > 1 else 0


# validate state, coefficient, source, and batch dimensions before array access.
# a shared coefficient or power row has one row instead of B.
# the compiled kernels require these checks because bounds checking is disabled.
# (todo) just a validation func, could probably move to diff folder
# x
# axes - B = simulated worlds, S = heat sources, M = thermal lags.
@njit(cache=True)
def validation_thermal_step_array_shapes(
    state_rise_k: Float64Array,  # state_rise_k = x (K) - lag-state temperature rises whose dimensions set the expected batch and lag counts - type float64 ndarray, shape (B, M)
    state_retention: Float64Array,  # state_retention = a (1) - retained thermal state fractions - type float64 ndarray, shape (1 or B, M)
    start_power_gain_k_per_w: Float64Array,  # start_power_gain_k_per_w = c0 (K/W) - starting-power gain - type float64 ndarray, shape (1 or B, M)
    end_power_gain_k_per_w: Float64Array,  # end_power_gain_k_per_w = c1 (K/W) - ending-power gain - type float64 ndarray, shape (1 or B, M)
    lag_source_indices: Int64Array,  # lag_source_indices = s_m (1) - source index driving each lag - type int64 ndarray, shape (M,)
    start_source_power_w: Float64Array,  # start_source_power_w = P_start (W) - source powers at the step boundary - type float64 ndarray, shape (1 or B, S)
    end_source_power_w: Float64Array,  # end_source_power_w = P_end (W) - source powers at the step boundary - type float64 ndarray, shape (1 or B, S)
    source_residue_scale: Float64Array,  # source_residue_scale = s_source (1) - per-source residue multipliers - type float64 ndarray, shape (1 or B, S)
) -> (
    None  # return = None - raise ValueError if world, source or lag axes disagree - type None
):
    world_count = state_rise_k.shape[0]
    lag_count = state_rise_k.shape[1]

    # the three weight arrays share one row count - 1 for shared weights, or B.
    coefficient_row_count = state_retention.shape[0]
    weight_rows_differ = (
        start_power_gain_k_per_w.shape[0] != coefficient_row_count
        or end_power_gain_k_per_w.shape[0] != coefficient_row_count
    )
    weight_rows_fit_no_batch = coefficient_row_count != 1 and coefficient_row_count != world_count
    # all three weights must refer to the same world or share one row across worlds.
    if weight_rows_differ or weight_rows_fit_no_batch:
        raise ValueError("thermal kernel: a, c0, c1 must all have 1 or B rows")

    # every per-lag array has one column per lag.
    lag_columns_differ = (
        state_retention.shape[1] != lag_count
        or start_power_gain_k_per_w.shape[1] != lag_count
        or end_power_gain_k_per_w.shape[1] != lag_count
        or lag_source_indices.shape[0] != lag_count
    )
    # each temperature state needs one source index and one set of integration weights.
    if lag_columns_differ:
        raise ValueError("thermal kernel: a, c0, c1, lag_src must have M (= x.shape[1]) lags")

    # each per-source array has 1 shared row, or B rows.
    source_row_counts = (
        start_source_power_w.shape[0],
        end_source_power_w.shape[0],
        source_residue_scale.shape[0],
    )
    # check the start power, end power and source scale against the batch.
    for row_count in source_row_counts:
        # one row is shared. any larger array needs a row for every world.
        if row_count != 1 and row_count != world_count:
            raise ValueError("thermal kernel: u0, u1, src_scale must have 1 or B rows")


# advance every world's lag states using shared or per-world coefficient rows.
# wrapper function, prob move out/reorg
# x
# axes - B = simulated worlds, S = heat sources, M = thermal lags.
@njit(cache=True)
def physics_advance_thermal_states(
    state_rise_k: Float64Array,  # state_rise_k = x (K) - lag-state temperature rises updated in place - type float64 ndarray, shape (B, M)
    state_retention: Float64Array,  # state_retention = a (1) - retained thermal state fractions - type float64 ndarray, shape (1 or B, M)
    start_power_gain_k_per_w: Float64Array,  # start_power_gain_k_per_w = c0 (K/W) - starting-power gain - type float64 ndarray, shape (1 or B, M)
    end_power_gain_k_per_w: Float64Array,  # end_power_gain_k_per_w = c1 (K/W) - ending-power gain - type float64 ndarray, shape (1 or B, M)
    lag_source_indices: Int64Array,  # lag_source_indices = s_m (1) - source index driving each lag - type int64 ndarray, shape (M,)
    start_source_power_w: Float64Array,  # start_source_power_w = P_start (W) - source powers at the step boundary - type float64 ndarray, shape (1 or B, S)
    end_source_power_w: Float64Array,  # end_source_power_w = P_end (W) - source powers at the step boundary - type float64 ndarray, shape (1 or B, S)
    source_residue_scale: Float64Array,  # source_residue_scale = s_source (1) - per-source residue multipliers - type float64 ndarray, shape (1 or B, S)
) -> (
    None  # return = None - update state_rise_k in kelvin - type None
):
    validation_thermal_step_array_shapes(
        state_rise_k,
        state_retention,
        start_power_gain_k_per_w,
        end_power_gain_k_per_w,
        lag_source_indices,
        start_source_power_w,
        end_source_power_w,
        source_residue_scale,
    )
    for world_index in range(state_rise_k.shape[0]):
        # pick this world's row from each input. an input with one row is shared by every world.
        coefficient_row = helper_shared_or_world_row_index(world_index, state_retention.shape[0])
        start_power_row = helper_shared_or_world_row_index(world_index, start_source_power_w.shape[0])
        end_power_row = helper_shared_or_world_row_index(world_index, end_source_power_w.shape[0])
        source_scale_row = helper_shared_or_world_row_index(world_index, source_residue_scale.shape[0])
        physics_advance_thermal_state_for_world(
            state_rise_k[world_index],
            state_retention[coefficient_row],
            start_power_gain_k_per_w[coefficient_row],
            end_power_gain_k_per_w[coefficient_row],
            lag_source_indices,
            start_source_power_w[start_power_row],
            end_source_power_w[end_power_row],
            source_residue_scale[source_scale_row],
        )


# write all absolute observer temperatures from the current lag states and base temperatures.
# x
# axes - B = simulated worlds, O = temperature observers, M = thermal lags.
@njit(cache=True)
def physics_write_observer_temperatures(
    state_rise_k: Float64Array,  # state_rise_k = x (K) - lag-state temperature rises updated in place - type float64 ndarray, shape (B, M)
    observer_lag_boundaries: Int64Array,  # observer_lag_boundaries = p_observer (1) - lag-array boundaries of each observer - type int64 ndarray, shape (O+1,)
    base_temperature_k: ArrayLike,  # base_temperature_k = T_base (K) - base temperature of each world - type array-like, scalar or shape (1 or B,)
    out_observer_temperature_k: Float64Array,  # out_observer_temperature_k = T_observer (K) - absolute observer temperatures written in place - type float64 ndarray, shape (B, O)
) -> (
    None  # return = None - update out_observer_temperature_k in kelvin - type None
):
    world_count = state_rise_k.shape[0]
    base_temperature_count = base_temperature_k.shape[0]
    output_rows_differ = out_observer_temperature_k.shape[0] != world_count
    base_fits_no_batch = base_temperature_count != 1 and base_temperature_count != world_count
    # each world needs output space and either a shared or a private base temperature.
    if output_rows_differ or base_fits_no_batch:
        raise ValueError("thermal_outputs: out must have B rows and t_base 1 or B entries")
    # every adjacent pair of boundaries describes exactly one observer.
    if out_observer_temperature_k.shape[1] + 1 != observer_lag_boundaries.shape[0]:
        raise ValueError("thermal_outputs: out must have O = len(obs_ptr) - 1 columns")
    # sum each world's lag contributions into its own output row.
    for world_index in range(world_count):
        base_row = helper_shared_or_world_row_index(world_index, base_temperature_count)
        physics_write_observer_temperatures_for_world(
            state_rise_k[world_index],
            observer_lag_boundaries,
            base_temperature_k[base_row],
            out_observer_temperature_k[world_index],
        )


# in the full picture - the compiled engine calls this once per step for each simulated chip when self-heating is off.
# it does two jobs in one loop and gives the same numbers as doing them one after the other.
# job 1 - advance each lag. a lag is one temperature contribution x (K) from one heat source.
# x_next = a*x + s*(c_start*P_start + c_end*P_end), with a (1), c (K/W), P (W), and source multiplier s (1).
# job 2 - read each observer. T_observer (K) = T_base (K) + the sum of the lags that end at that observer.
# observer o owns lags boundaries[o] up to boundaries[o+1]-1.
# update each lag and add it to its observer in one pass.
# the state-only update above uses the same equation.
# x
# axes - S = heat sources, O = temperature observers, M = thermal lags.
@njit(cache=True, _nrt=False)
def physics_advance_thermal_state_and_temperatures_for_world(
    state_rise_k: Float64Array,  # state_rise_k = x (K) - lag-state temperature rises updated in place - type float64 ndarray, shape (M,)
    state_retention: Float64Array,  # state_retention = a (1) - retained thermal state fractions - type float64 ndarray, shape (M,)
    start_power_gain_k_per_w: Float64Array,  # start_power_gain_k_per_w = c0 (K/W) - starting-power gain - type float64 ndarray, shape (M,)
    end_power_gain_k_per_w: Float64Array,  # end_power_gain_k_per_w = c1 (K/W) - ending-power gain - type float64 ndarray, shape (M,)
    lag_source_indices: Int64Array,  # lag_source_indices = s_m (1) - source index driving each lag - type int64 ndarray, shape (M,)
    observer_lag_boundaries: Int64Array,  # observer_lag_boundaries = p_observer (1) - lag-array boundaries of each observer - type int64 ndarray, shape (O+1,)
    start_source_power_w: Float64Array,  # start_source_power_w = P_start (W) - source powers at the step boundary - type float64 ndarray, shape (S,)
    end_source_power_w: Float64Array,  # end_source_power_w = P_end (W) - source powers at the step boundary - type float64 ndarray, shape (S,)
    source_residue_scale: Float64Array,  # source_residue_scale = s_source (1) - per-source residue multipliers - type float64 ndarray, shape (S,)
    base_temperature_k: float,  # base_temperature_k = T_base (K) - base temperature - type float scalar
    out_observer_temperature_k: Float64Array,  # out_observer_temperature_k = T_observer (K) - absolute observer temperatures written in place - type float64 ndarray, shape (O,)
) -> (
    None  # return = None - update state_rise_k and out_observer_temperature_k in kelvin - type None
):
    # apply the lag update, then add each updated rise to its observer's base temperature.
    for observer_index in range(out_observer_temperature_k.shape[0]):
        temperature_k = base_temperature_k
        # advance the lags that contribute to this observer before summing its temperature.
        for lag_index in range(observer_lag_boundaries[observer_index], observer_lag_boundaries[observer_index + 1]):
            source_index = lag_source_indices[lag_index]
            # only this fraction of the lag's previous temperature rise survives the step.
            retained_rise_k = state_retention[lag_index] * state_rise_k[lag_index]
            added_rise_k = source_residue_scale[source_index] * (
                start_power_gain_k_per_w[lag_index] * start_source_power_w[source_index]
                + end_power_gain_k_per_w[lag_index] * end_source_power_w[source_index]
            )
            next_rise_k = retained_rise_k + added_rise_k
            state_rise_k[lag_index] = next_rise_k
            temperature_k += next_rise_k
        out_observer_temperature_k[observer_index] = temperature_k


# advance all worlds and write their end-of-step observer temperatures.
# validate the batch dimensions before selecting shared rows.
# x
# axes - B = simulated worlds, S = heat sources, O = temperature observers, M = thermal lags.
@njit(cache=True)
def physics_advance_thermal_states_and_temperatures(
    state_rise_k: Float64Array,  # state_rise_k = x (K) - lag-state temperature rises updated in place - type float64 ndarray, shape (B, M)
    state_retention: Float64Array,  # state_retention = a (1) - retained thermal state fractions - type float64 ndarray, shape (1 or B, M)
    start_power_gain_k_per_w: Float64Array,  # start_power_gain_k_per_w = c0 (K/W) - starting-power gain - type float64 ndarray, shape (1 or B, M)
    end_power_gain_k_per_w: Float64Array,  # end_power_gain_k_per_w = c1 (K/W) - ending-power gain - type float64 ndarray, shape (1 or B, M)
    lag_source_indices: Int64Array,  # lag_source_indices = s_m (1) - source index driving each lag - type int64 ndarray, shape (M,)
    observer_lag_boundaries: Int64Array,  # observer_lag_boundaries = p_observer (1) - lag-array boundaries of each observer - type int64 ndarray, shape (O+1,)
    start_source_power_w: Float64Array,  # start_source_power_w = P_start (W) - source powers at the step boundary - type float64 ndarray, shape (1 or B, S)
    end_source_power_w: Float64Array,  # end_source_power_w = P_end (W) - source powers at the step boundary - type float64 ndarray, shape (1 or B, S)
    source_residue_scale: Float64Array,  # source_residue_scale = s_source (1) - per-source residue multipliers - type float64 ndarray, shape (1 or B, S)
    base_temperature_k: ArrayLike,  # base_temperature_k = T_base (K) - base temperature of each world - type array-like, scalar or shape (1 or B,)
    out_observer_temperature_k: Float64Array,  # out_observer_temperature_k = T_observer (K) - absolute observer temperatures written in place - type float64 ndarray, shape (B, O)
) -> (
    None  # return = None - update state_rise_k and out_observer_temperature_k in kelvin - type None
):
    validation_thermal_step_array_shapes(
        state_rise_k,
        state_retention,
        start_power_gain_k_per_w,
        end_power_gain_k_per_w,
        lag_source_indices,
        start_source_power_w,
        end_source_power_w,
        source_residue_scale,
    )
    world_count = state_rise_k.shape[0]
    base_temperature_count = base_temperature_k.shape[0]
    output_rows_differ = out_observer_temperature_k.shape[0] != world_count
    base_fits_no_batch = base_temperature_count != 1 and base_temperature_count != world_count
    # each simulated world needs an output row and access to a base temperature.
    if output_rows_differ or base_fits_no_batch:
        raise ValueError("thermal_advance: out must have B rows and t_base 1 or B entries")
    # the last boundary marks the end of the final observer's lag range.
    if out_observer_temperature_k.shape[1] + 1 != observer_lag_boundaries.shape[0]:
        raise ValueError("thermal_advance: out must have O = len(obs_ptr) - 1 columns")
    for world_index in range(world_count):
        # pick this world's row from each input. an input with one row is shared by every world.
        coefficient_row = helper_shared_or_world_row_index(world_index, state_retention.shape[0])
        start_power_row = helper_shared_or_world_row_index(world_index, start_source_power_w.shape[0])
        end_power_row = helper_shared_or_world_row_index(world_index, end_source_power_w.shape[0])
        source_scale_row = helper_shared_or_world_row_index(world_index, source_residue_scale.shape[0])
        base_row = helper_shared_or_world_row_index(world_index, base_temperature_count)
        physics_advance_thermal_state_and_temperatures_for_world(
            state_rise_k[world_index],
            state_retention[coefficient_row],
            start_power_gain_k_per_w[coefficient_row],
            end_power_gain_k_per_w[coefficient_row],
            lag_source_indices,
            observer_lag_boundaries,
            start_source_power_w[start_power_row],
            end_source_power_w[end_power_row],
            source_residue_scale[source_scale_row],
            base_temperature_k[base_row],
            out_observer_temperature_k[world_index],
        )


# in the full picture - one source's power for this step is not known yet, because it depends on the temperature it produces.
# example - heater power is V^2/R, and the heater resistance R depends on the heater temperature at the end of the step.
# this function prepares that solve for one observer. it splits the observer's end-of-step rise into two parts.
# T_end - T_base = T_known + g_self*P_self.
# T_known (K) is everything that does not depend on the unknown power.
# it is the old warmth that survives the step, plus the new warmth from every other source.
# g_self (K/W) is how much one watt of the unknown source, held for the whole step, warms this observer by the end of the step.
# P_self (W) is the unknown power. the caller solves its own equation for it and then runs the normal update.
# a lag is one temperature contribution x (K). a (1), c_start (K/W), and c_end (K/W) are its prepared one-step weights.
# observer o owns lags boundaries[o] up to boundaries[o+1]-1.
# T_end-T_base=T_known+g_self*P_self, with P_self constant over the step.
# include stored heat and every other source in T_known.
# use the same other-source powers in the subsequent state update.
# the heater resistance solve uses this split for its temperature feedback.
# doing calculation for known temp and gain
# x
# axes - S = heat sources, O = temperature observers, M = thermal lags.
@njit(cache=True)
def physics_separate_known_heat_and_source_gain_for_observer(
    state_rise_k: Float64Array,  # state_rise_k = x (K) - lag-state temperature rises updated in place - type float64 ndarray, shape (M,)
    state_retention: Float64Array,  # state_retention = a (1) - retained thermal state fractions - type float64 ndarray, shape (M,)
    start_power_gain_k_per_w: Float64Array,  # start_power_gain_k_per_w = c0 (K/W) - starting-power gain - type float64 ndarray, shape (M,)
    end_power_gain_k_per_w: Float64Array,  # end_power_gain_k_per_w = c1 (K/W) - ending-power gain - type float64 ndarray, shape (M,)
    lag_source_indices: Int64Array,  # lag_source_indices = s_m (1) - source index driving each lag - type int64 ndarray, shape (M,)
    observer_lag_boundaries: Int64Array,  # observer_lag_boundaries = p_observer (1) - lag-array boundaries of each observer - type int64 ndarray, shape (O+1,)
    observer_index: int,  # observer_index = o (1) - observer index - type int scalar
    unknown_source_index: int,  # unknown_source_index = s_self (1) - unknown heat-source index - type int scalar
    start_source_power_w: Float64Array,  # start_source_power_w = P_start (W) - source powers at the step boundary - type float64 ndarray, shape (S,)
    end_source_power_w: Float64Array,  # end_source_power_w = P_end (W) - source powers at the step boundary - type float64 ndarray, shape (S,)
    source_residue_scale: Float64Array,  # source_residue_scale = s_source (1) - per-source residue multipliers - type float64 ndarray, shape (S,)
) -> tuple[
    float,  # return[0] = T_known (K) - temperature rise from stored heat and known sources - type float scalar
    float,  # return[1] = g_self (K/W) - gain from the unknown source held over the step - type float scalar
]:
    known_rise_k = 0.0
    self_gain_k_per_w = 0.0
    # split each lag contribution into known heat and the part proportional to the unknown source power.
    for lag_index in range(observer_lag_boundaries[observer_index], observer_lag_boundaries[observer_index + 1]):
        source_index = lag_source_indices[lag_index]
        # old warmth that survives the step is known whatever this step's power turns out to be.
        retained_rise_k = state_retention[lag_index] * state_rise_k[lag_index]
        known_rise_k += retained_rise_k

        start_gain = start_power_gain_k_per_w[lag_index]
        end_gain = end_power_gain_k_per_w[lag_index]
        source_scale = source_residue_scale[source_index]
        if source_index == unknown_source_index:
            # held power uses the same unknown at both ends of the step.
            self_gain_k_per_w += source_scale * (start_gain + end_gain)
        else:
            # every other source has a known power, so its new warmth (K) is known too.
            added_rise_k = source_scale * (
                start_gain * start_source_power_w[source_index] + end_gain * end_source_power_w[source_index]
            )
            known_rise_k += added_rise_k
    return (
        known_rise_k,
        self_gain_k_per_w,
    )


# section 3 - saved thermal model
# a ThermalRom holds lag gains and time constants fitted to a finite-element heat solve.
# the file format is thermal_rom/1 - see docs/ARCHITECTURE.md, 5.3 "Thermal ROM".
# apart from the ThermalRom record, this section reads, writes, and checks files. it holds no heat equation.

# hex sha256 of a file's bytes.
# x
def io_sha256_file(
    path: str | Path,  # path = path (1) - thermal artifact path - type str or Path
) -> (
    str  # return = digest (1) - hexadecimal SHA-256 checksum - type str
):
    digest = hashlib.sha256()
    # hash the bytes that will be loaded as the thermal model.
    with open(path, "rb") as stream:
        # read bounded chunks so the model can exceed the available working memory.
        for chunk in iter(partial(stream.read, 1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# `<stem>.sidecar.json` (the convention of mrm/extraction/common.py).
# `<name>.sidecar.json` and `<stem>.json` (the name mrm/extraction/thermal_fem.artefact_paths writes) are also read.
# x
# x all these helpers can be moved somewhere else?
def io_build_rom_sidecar_path(
    path: str | Path,  # path = path (1) - thermal artifact path - type str or Path
) -> (
    Path  # return = path (1) - canonical sidecar artifact path - type Path
):
    # give the metadata the same stem as the numeric archive so the loader can pair them.
    return path.with_name(path.stem + ".sidecar.json")


# find the first existing sidecar path.
# x
def io_find_rom_sidecar_path(
    path: str | Path,  # path = path (1) - thermal artifact path - type str or Path
) -> (
    Path | None  # return = path (1) - existing sidecar path or no match - type Path or None
):
    # same candidates as mrm.params.config.sidecar_candidates (minus the .meta.json names)
    for candidate in (io_build_rom_sidecar_path(path), path.with_name(path.name + ".sidecar.json"), path.with_suffix(".json")):
        # use the first existing metadata file in the supported search order.
        if candidate.exists():
            return candidate
    return None


# default artefact path `mrm/data/extracted/thermal_rom_<family>_<film>.npz`.
# x
def io_build_thermal_rom_path(
    family: str,  # family = family (1) - heater-family artifact name - type str
    film_nm: float,  # film_nm = d_film (nm) - silicon-film thickness - type float scalar
    root: str | Path | None = None,  # root = root (1) - optional artifact directory - type str, Path, or None
) -> (
    Path  # return = path (1) - thermal ROM artifact path - type Path
):
    # use the packaged extraction directory when the caller supplies no root.
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1] / "data" / "extracted"
    return base / f"thermal_rom_{family}_{int(round(float(film_nm)))}.npz"


# store a thermal_rom/1 model for heat paths on the same ring and between rings.
# S is the number of source kinds, O is the number of observer kinds, D is the number of distances, and K is the number of thermal lags.
# the cross-ring table assumes radial symmetry and applies at centre distances greater than two ring radii.
# x
@dataclass(frozen=True, eq=False)
class ThermalRom:
    taus_s: Float64Array  # tau_k (s) - lag time constants shared by all heat paths - shape (K,)
    sources: tuple[str, ...]  # heat source kinds in table order - length S
    observers: tuple[str, ...]  # temperature observer kinds in table order - length O
    self_res: Float64Array  # r_self (K/W) - lag gains for sites on the same ring - shape (S, O, K)
    cross_r_um: Float64Array  # d (um) - ring centre distances in the cross-ring table - shape (D,)
    cross_res: Float64Array  # r_cross (K/W) - lag gains between rings at each distance - shape (S, O, D, K)
    site_offsets_um: dict[str, Any] = field(default_factory=dict)  # (x, y) (um) - driver, PD, and sensor offsets used for the same-ring table
    meta: dict[str, Any] = field(default_factory=dict)  # sidecar provenance, schema, checksum, mesh details, and validation results

    # check that the residue tables match their axes and that the model values are valid.
    def validation_check_thermal_rom(
        self: ThermalRom,  # self = self (1) - extracted thermal model to validate - type ThermalRom
    ) -> (
        None  # return = None - raises RomIntegrityError if a validation check fails - type None
    ):
        source_kind_count = len(self.sources)
        observer_kind_count = len(self.observers)
        lag_count = np.asarray(self.taus_s).shape[0]

        same_ring_residues = np.asarray(self.self_res)
        expected_same_ring_shape = (source_kind_count, observer_kind_count, lag_count)
        # the same-ring table needs one fitted gain per source, observer and time constant.
        if same_ring_residues.shape != expected_same_ring_shape:
            raise RomIntegrityError(
                f"self_res shape {same_ring_residues.shape} must be (S, O, K) = {expected_same_ring_shape}"
            )

        # cross-ring gains also depend on the separation between ring centres.
        distance_count = np.asarray(self.cross_r_um).shape[0]
        cross_ring_residues = np.asarray(self.cross_res)
        expected_cross_ring_shape = (source_kind_count, observer_kind_count, distance_count, lag_count)
        # each sampled distance needs a complete source-by-observer set of fitted gains.
        if cross_ring_residues.shape != expected_cross_ring_shape:
            raise RomIntegrityError(
                f"cross_res shape {cross_ring_residues.shape} must be (S, O, D, K) = {expected_cross_ring_shape}"
            )

        unknown_sources = [source for source in self.sources if source not in SOURCE_KINDS]
        unknown_observers = [observer for observer in self.observers if observer not in OBSERVER_KINDS]
        # routing depends on recognized heat sources and temperature observers.
        if unknown_sources or unknown_observers:
            raise RomIntegrityError(
                f"unknown source/observer kinds - sources {unknown_sources}, observers {unknown_observers}"
            )

        time_constants_s = np.asarray(self.taus_s, dtype=float)
        # positive ordered time constants give decaying modes in fast-to-slow order.
        if not np.all(time_constants_s > 0) or np.any(np.diff(time_constants_s) <= 0):
            raise RomIntegrityError("taus_s must be positive and strictly increasing")

        # a model with a distance table must support unambiguous interpolation.
        if distance_count:
            centre_distances_um = np.asarray(self.cross_r_um, dtype=float)
            # each successive sample must be farther from the heated ring.
            if np.any(np.diff(centre_distances_um) <= 0):
                raise RomIntegrityError("cross_r_um must be strictly increasing")

        # an invalid local gain would corrupt every temperature calculated on that path.
        if not np.all(np.isfinite(same_ring_residues)):
            raise RomIntegrityError("self_res contains non-finite values")
        # a nonfinite cross-ring gain would propagate across the layout during interpolation.
        if not np.all(np.isfinite(cross_ring_residues)):
            raise RomIntegrityError("cross_res contains non-finite values")

    # established method names and Python hooks share the same descriptors.
    __post_init__ = validation_check_thermal_rom


# write the thermal ROM arrays and their checksum sidecar.
# keep the thermal_rom/1 field names and sidecar naming convention.
# add schema, SHA-256, and the data filename to the supplied provenance.
# x
def io_save_rom(
    path: str | Path,  # path = path (1) - thermal artifact path - type str or Path
    rom: ThermalRom,  # rom = model (mixed units) - extracted poles, thermal residues, and site metadata - type ThermalRom record
    sidecar: Mapping[str, Any] | None = None,  # sidecar = metadata (mixed units) - optional artifact provenance and validation fields - type mapping or None
) -> (
    Path  # return = path (1) - written ROM artifact path - type Path
):
    path = Path(path)
    names = sorted(rom.site_offsets_um)
    offs = np.array([rom.site_offsets_um[n] for n in names], dtype=np.float64).reshape(len(names), 2)
    # write the numerical arrays before hashing their exact file representation.
    with open(path, "wb") as f:
        np.savez(f, taus_s=np.asarray(rom.taus_s, np.float64), sources=np.array(rom.sources, dtype="U32"),
                 observers=np.array(rom.observers, dtype="U32"), self_res=np.asarray(rom.self_res, np.float64),
                 cross_r_um=np.asarray(rom.cross_r_um, np.float64), cross_res=np.asarray(rom.cross_res, np.float64),
                 site_offset_names=np.array(names, dtype="U32"), site_offsets_um=offs)
    side = dict(sidecar or {})
    side["schema"] = SCHEMA_ROM
    # record the digest so the loader can check which numerical data the metadata describes.
    side["sha256"] = io_sha256_file(path)
    side["data_file"] = path.name
    # store validation and provenance alongside the numerical data.
    with open(io_build_rom_sidecar_path(path), "w") as f:
        json.dump(side, f, indent=2, sort_keys=True, default=str)
    return path


# load thermal ROM arrays and their sidecar metadata.
# verify=True requires the expected schema and the SHA-256 of the data file.
# warn=True reports validation limits or required time scales that are missing.
# accept the flat writer format and the nested FEM extraction format.
# x
def io_load_rom(
    path: str | Path,  # path = path (1) - thermal artifact path - type str or Path
    *,
    verify: bool = True,  # verify = verify (1) - require schema and checksum verification - type bool scalar
    warn: bool = True,  # warn = warn (1) - emit warnings for missing or failed validation metrics - type bool scalar
) -> (
    ThermalRom  # return = model (mixed units) - validated poles, residues, and sidecar metadata - type ThermalRom record
):
    path = Path(path)
    side_path = io_find_rom_sidecar_path(path)
    meta: dict = {}
    # read the sidecar when present so its coordinate and validation data survive loading.
    if side_path is not None:
        meta = json.loads(side_path.read_text())
    # verification requires the metadata that identifies the expected file.
    elif verify:
        raise RomIntegrityError(f"missing sidecar {io_build_rom_sidecar_path(path)}")
    # verify metadata compatibility and file integrity before trusting the extracted gains.
    if verify:
        # the schema identifies how to interpret the recorded fields.
        if meta.get("schema") != SCHEMA_ROM:
            raise RomIntegrityError(f"{side_path}: schema {meta.get('schema')!r} != {SCHEMA_ROM!r}")
        digest = io_sha256_file(path)
        # a digest mismatch means the metadata describes different bytes.
        if meta.get("sha256") != digest:
            raise RomIntegrityError(f"{path}: sha256 {digest} does not match the sidecar ({meta.get('sha256')})")
    # load only numerical data from the model archive.
    with np.load(path, allow_pickle=False) as z:
        taus = np.asarray(z["taus_s"], np.float64)
        sources = tuple(str(s) for s in z["sources"])
        observers = tuple(str(s) for s in z["observers"])
        self_res = np.asarray(z["self_res"], np.float64)
        cross_r = np.asarray(z["cross_r_um"], np.float64)
        cross_res = np.asarray(z["cross_res"], np.float64)
        # collect optional site positions separately from the required gain tables.
        offsets = {}
        # names key - 'site_offset_names' (what both writers write).
        # 'site_names' is accepted as a fallback
        name_key = next((k for k in ("site_offset_names", "site_names") if k in z.files), None)
        # pair each stored offset with its recorded site name.
        if name_key is not None and "site_offsets_um" in z.files:
            # each row supplies one site's displacement from its ring centre.
            for n, xy in zip(z[name_key], np.asarray(z["site_offsets_um"], np.float64).reshape(-1, 2)):
                offsets[str(n)] = (float(xy[0]), float(xy[1]))
    # sidecar fallbacks - a top-level site_offsets_um dict, or the extraction config (inputs.config.site_offsets_um)
    cfg_offs = ((meta.get("inputs") or {}).get("config") or {}).get("site_offsets_um") or {}
    # merge the offset sources in precedence order.
    for src in (meta.get("site_offsets_um") or {}, cfg_offs):
        offsets.update({k: tuple(map(float, v)) for k, v in src.items() if k not in offsets})
    meta = dict(meta)
    meta["path"] = str(path)
    rom = ThermalRom(taus, sources, observers, self_res, cross_r, cross_res, offsets, meta)
    # quality warnings are optional because a caller may inspect an incomplete extraction.
    if warn:
        validation_warn_about_rom_quality(rom)
    # the record now contains both the numerical model and its metadata.
    return rom


# read step, magnitude, phase, and superposition validation errors from sidecar metadata.
# use the larger self-path or nearest-neighbour error in the nested FEM format.
# return None for a metric that is absent.
# x
# extract the largest reported errors across the measured heat paths.
def helper_read_rom_validation_metrics(
    meta: dict[str, Any],  # meta = metadata (mixed units) - provenance and validation fields - type dict with string keys and metadata values
) -> (
    dict[str, float | None]  # return = metrics (mixed units) - relative step error, magnitude error in dB, phase error in deg, and relative superposition error - type dict of named floats or None
):
    v = dict(meta.get("validation") or {})
    out = {k: v.get(k) for k in ("step_max_rel_err", "freq_max_db_err", "freq_max_deg_err", "superposition_rel_err")}
    nested = {"step_max_rel_err": "max_time_err_frac_of_dc", "freq_max_db_err": "max_db",
              "freq_max_deg_err": "max_deg"}
    # recover reported errors from older nested validation records.
    for key, sub in nested.items():
        # use the worst group error when a top-level error is absent.
        if out[key] is None:
            vals = [float(v[g][sub]) for g in ("self", "nearest_neighbour")
                    if isinstance(v.get(g), Mapping) and v[g].get(sub) is not None]
            # report the worst measured group error and keep a missing measurement distinguishable from zero error.
            out[key] = max(vals) if vals else None
    # older extractions can store the superposition test inside a stage.
    if out["superposition_rel_err"] is None:
        # inspect the extraction stages for a recorded superposition test.
        for st in (meta.get("stages") or {}).values():
            # a stage contributes only when it actually reports the metric.
            if isinstance(st, Mapping) and st.get("superposition_rel_err") is not None:
                out["superposition_rel_err"] = float(st["superposition_rel_err"])
    return out


# warn (RuntimeWarning) when a ROM misses an acceptance target or a required pole.
# x
def validation_warn_about_rom_quality(
    rom: ThermalRom,  # rom = model (mixed units) - extracted poles, thermal residues, and site metadata - type ThermalRom record
) -> (
    None  # return = None - issue warnings for missing metrics or pole coverage - type None
):
    v = helper_read_rom_validation_metrics(rom.meta)
    checks = (("step_max_rel_err", ROM_STEP_ERR_MAX), ("freq_max_db_err", ROM_FREQ_DB_MAX),
              ("freq_max_deg_err", ROM_FREQ_DEG_MAX), ("superposition_rel_err", ROM_SUPERPOSITION_MAX))
    # compare each reported error with its own acceptance limit.
    for key, lim in checks:
        val = v.get(key)
        # a missing measurement leaves this quality check unevaluated.
        if val is None:
            warnings.warn(f"thermal ROM {rom.meta.get('path')}: sidecar lacks validation.{key}",
                          RuntimeWarning, stacklevel=3)
        # report a measurement outside its allowed magnitude.
        elif not (abs(float(val)) <= lim):
            warnings.warn(f"thermal ROM {rom.meta.get('path')}: validation.{key} = {val} exceeds {lim}",
                          RuntimeWarning, stacklevel=3)
    # a model without a fast pole can miss the initial waveguide temperature rise.
    if rom.taus_s[0] > ROM_TAU_FAST_MAX_S:
        warnings.warn(f"thermal ROM has no pole below {ROM_TAU_FAST_MAX_S} s; the optical self-heating solve needs "
                      "the fast poles of the waveguide core", RuntimeWarning, stacklevel=3)
    # a model without a slow pole can miss heat transport between rings.
    if rom.taus_s[-1] < ROM_TAU_SLOW_MIN_S:
        warnings.warn(f"thermal ROM has no pole above {ROM_TAU_SLOW_MIN_S} s; the crosstalk needs the ms poles",
                      RuntimeWarning, stacklevel=3)


# section 4 - layout and model records
# LayoutSites says where each source and observer is.
# ThermalScales says how one simulated chip differs from the nominal model.
# InterimRom gives the heat-path gains and times used when no ThermalRom is fitted.

# site coordinates as an (n, 2) array in um (a single (2,) point becomes one row).
# x
# axes - N = sites.
def validation_site_coordinates_um(
    coordinates_um: ArrayLike,  # coordinates_um = r (um) - site coordinates - type array-like, shape (2,) or (N, 2)
    expected_site_count: int | None = None,  # expected_site_count = N (1) - optional required coordinate count - type int or None
    name: str = "xy",  # name = name (1) - element name or diagnostic parameter label - type str
) -> (
    Float64Array  # return = r_sites (um) - validated site coordinates - type float64 ndarray, shape (N, 2)
):
    coordinates = np.asarray(coordinates_um, dtype=np.float64)
    # a single coordinate pair represents one site.
    if coordinates.ndim == 1 and coordinates.shape[0] == 2:
        coordinates = coordinates[None, :]
    # each site needs two planar coordinates.
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"{name} must be (n, 2) in um, got shape {coordinates.shape}")
    # downstream layout arrays require one coordinate pair per declared site.
    if expected_site_count is not None and coordinates.shape[0] != expected_site_count:
        raise ValueError(f"{name} must have {expected_site_count} rows, got {coordinates.shape[0]}")
    return coordinates


# read a ring index from the site field or trailing digits.
def _helper_resolve_site_ring_index(
    name: str,  # name = name (1) - element name or diagnostic parameter label - type str
    spec: Mapping[str, Any],  # spec = site (mixed units) - site record with an optional ring field - type mapping
) -> (
    int | None  # return = k (1) - ring index from the field or trailing digits - type int or None
):
    # an explicit ring assignment takes precedence over the name suffix.
    if "ring" in spec and spec["ring"] is not None:
        return int(spec["ring"])
    digits = ""
    # read trailing digits from the end of the site name as its ring index.
    for character in reversed(str(name)):
        # retain digits until the numeric suffix ends.
        if character.isdigit():
            # prepend the digit because the suffix is read from right to left.
            digits = character + digits
        else:
            # the first nondigit terminates the ring index.
            break
    # return the extracted ring index or leave the site unattached if its name has no numeric suffix.
    return int(digits) if digits else None



# hold the bank's thermal site coordinates in um.
# ring_xy_um has shape (N, 2) and gives ring centres.
# pd_xy_um has shape (N, 2).
# None or a NaN row uses the model's default PD offset.
# driver_xy_um has shape (N, 2).
# None means no driver sources.
# a NaN row omits that ring's driver.
# temp_sensor_xy_um has shape (T, 2).
# temp_sensor_ring attaches each sensor to a ring, or -1 for a free site.
# aggressor_xy_um has shape (A, 2).
# names identify sensor and aggressor sites.
# x
@dataclass(frozen=True, eq=False)
class LayoutSites:
    ring_xy_um: Float64Array
    pd_xy_um: Float64Array | None = None
    driver_xy_um: Float64Array | None = None
    temp_sensor_xy_um: Float64Array | None = None
    temp_sensor_ring: Int64Array | None = None
    temp_sensor_names: tuple[str, ...] = ()
    aggressor_xy_um: Float64Array | None = None
    aggressor_names: tuple[str, ...] = ()

    # validate the record shape and physical indexing.
    def validation_check_layout_sites(
        self: LayoutSites,  # self = self (mixed units) - physical positions and ring assignments of heat sources and readouts - type LayoutSites record
    ) -> (
        None  # return = None - validate and normalize the record - type None
    ):
        # the record is frozen, so object.__setattr__ stores each validated field.
        ring_count = validation_site_coordinates_um(self.ring_xy_um, name="ring_xy_um").shape[0]
        object.__setattr__(self, "ring_xy_um", validation_site_coordinates_um(self.ring_xy_um, name="ring_xy_um"))

        # monitors and drivers have one row per ring when present.
        for field_name in ("pd_xy_um", "driver_xy_um"):
            coordinates_um = getattr(self, field_name)
            # validate coordinates only for site families supplied by this layout.
            if coordinates_um is not None:
                object.__setattr__(
                    self, field_name, validation_site_coordinates_um(coordinates_um, ring_count, field_name)
                )

        # each sensor gets a ring attachment (-1 for a free site) and a name.
        if self.temp_sensor_xy_um is not None:
            sensor_xy_um = validation_site_coordinates_um(self.temp_sensor_xy_um, name="temp_sensor_xy_um")
            object.__setattr__(self, "temp_sensor_xy_um", sensor_xy_um)
            sensor_count = sensor_xy_um.shape[0]
            # sensors without a ring assignment use the free-site convention.
            if self.temp_sensor_ring is None:
                sensor_rings = np.full(sensor_count, -1, np.int64)
            else:
                # normalize explicit sensor assignments to one ring index per sensor.
                sensor_rings = np.asarray(self.temp_sensor_ring, np.int64).reshape(-1)
            # each sensor must map to a physical ring or the free-site sentinel.
            if sensor_rings.shape[0] != sensor_count or np.any(sensor_rings >= ring_count) or np.any(sensor_rings < -1):
                raise ValueError("temp_sensor_ring must have one entry per sensor, each -1 or a ring index")
            object.__setattr__(self, "temp_sensor_ring", sensor_rings)
            # supply stable names for temperature sensors that lack explicit names.
            names = tuple(self.temp_sensor_names) or tuple(f"ts{i}" for i in range(sensor_count))
            object.__setattr__(self, "temp_sensor_names", names)

        # each aggressor gets a name.
        if self.aggressor_xy_um is not None:
            aggressor_xy_um = validation_site_coordinates_um(self.aggressor_xy_um, name="aggressor_xy_um")
            object.__setattr__(self, "aggressor_xy_um", aggressor_xy_um)
            names = tuple(self.aggressor_names) or tuple(f"aggressor{i}" for i in range(aggressor_xy_um.shape[0]))
            object.__setattr__(self, "aggressor_names", names)

    # number of rings N.
    @property
    def helper_ring_count(
        self: LayoutSites,  # self = self (mixed units) - physical positions and ring assignments of heat sources and readouts - type LayoutSites record
    ) -> (
        int  # return = N (1) - number of rings - type int scalar
    ):
        return self.ring_xy_um.shape[0]

    # a straight bank - ring i at i * pitch_um along x (axis 0) or y (axis 1).
    # axes - N = physical rings, T = temperature sensors, A = external heat sources.
    @staticmethod
    def assembly_straight_ring_layout(
        n_rings: int,  # n_rings = N (1) - number of rings - type int scalar
        pitch_um: float,  # pitch_um = d_pitch (um) - ring centre spacing - type float scalar
        *,
        axis: int = 0,  # axis = axis (1) - coordinate axis of the straight ring bank - type int scalar
        aggressor_xy_um: ArrayLike | None = None,  # aggressor_xy_um = r_aggressor (um) - aggressor coordinates - type array-like, shape (A, 2), or None
        temp_sensor_xy_um: ArrayLike | None = None,  # temp_sensor_xy_um = r_sensor (um) - temperature-sensor coordinates - type array-like, shape (T, 2), or None
        temp_sensor_ring: ArrayLike | None = None,  # temp_sensor_ring = k_sensor (1) - attached ring indices, with -1 for free sites - type integer array-like, shape (T,), or None
        temp_sensor_names: Sequence[str] = (),  # temp_sensor_names = sensor_names (1) - temperature-sensor names - type sequence of strings
        aggressor_names: Sequence[str] = (),  # aggressor_names = aggressor_names (1) - aggressor names - type sequence of strings
        pd_xy_um: ArrayLike | None = None,  # pd_xy_um = r_PD (um) - monitor-photodiode coordinates - type array-like, shape (N, 2), or None
        driver_xy_um: ArrayLike | None = None,  # driver_xy_um = r_driver (um) - driver coordinates - type array-like, shape (N, 2), or None
    ) -> (
        LayoutSites  # return = sites (um) - straight ring bank and supplied site coordinates - type LayoutSites record
    ):
        ring_xy_um = np.zeros((int(n_rings), 2))
        # successive centres lie one pitch apart along the chosen axis.
        ring_xy_um[:, axis] = np.arange(int(n_rings)) * float(pitch_um)
        return LayoutSites(
            ring_xy_um=ring_xy_um,
            pd_xy_um=pd_xy_um,
            driver_xy_um=driver_xy_um,
            temp_sensor_xy_um=temp_sensor_xy_um,
            temp_sensor_ring=temp_sensor_ring,
            temp_sensor_names=tuple(temp_sensor_names),
            aggressor_xy_um=aggressor_xy_um,
            aggressor_names=tuple(aggressor_names),
        )

    # construct ring and auxiliary-site coordinates from named layout records.
    # each site supplies coordinates in um, a kind, and an optional ring attachment.
    # infer ring indices from explicit fields or trailing digits where the site kind permits it.
    # place missing rings on a line using ring_pitch_um.
    # a driver without a ring index is shared by the rings.
    # a heater site without an explicit ring field is an aggressor.
    # referenced temperature-sensor sites are included only once.
    # sensor attachments select a ring, the bank centroid, or the assumed die location.
    # locate each source and temperature sensor in the same coordinate system as the ring centres.
    @staticmethod
    def assembly_layout_from_sites(
        sites: Mapping[str, Mapping[str, Any]] | None,  # sites = sites (mixed units) - named site records with x_um, y_um, kind, and optional ring - type mapping or None
        n_rings: int,  # n_rings = N (1) - number of rings - type int scalar
        ring_pitch_um: float | None = None,  # ring_pitch_um = d_pitch (um) - optional spacing for missing ring sites - type float or None
        temp_sensors: Sequence[Mapping[str, Any]] | None = None,  # temp_sensors = sensors (mixed units) - sensor site or attachment records - type sequence of mappings or None
    ) -> (
        LayoutSites  # return = sites (um) - parsed ring, driver, sensor, photodiode, and aggressor positions - type LayoutSites record
    ):
        sites = dict(sites or {})
        # a sensor record can point at a named site. such a site is added once, in step 3.
        referenced_site_names = {str(spec["site"]) for spec in (temp_sensors or ()) if spec.get("site")}
        # a NaN row means that no site was given for that ring.
        ring_xy_um = np.full((n_rings, 2), np.nan)
        monitor_xy_um = np.full((n_rings, 2), np.nan)
        driver_xy_um = np.full((n_rings, 2), np.nan)
        shared_driver_xy_um = None
        aggressor_xy_um, aggressor_names = [], []
        sensor_xy_um, sensor_rings, sensor_names = [], [], []


        # step 1 - route each named site to its list by kind.
        for name, spec in sites.items():
            kind = spec.get("kind", "other")
            position_um = (float(spec["x_um"]), float(spec["y_um"]))
            ring_index = _helper_resolve_site_ring_index(name, spec)
            # ring sites set the reference centres for all ring-relative distances.
            if kind == "ring":
                # a centre needs an index in the model's physical ring range.
                if ring_index is None or not 0 <= ring_index < n_rings:
                    raise ValueError(f"ring site {name!r} needs a ring index 0..{n_rings - 1}")
                ring_xy_um[ring_index] = position_um
            # each monitor belongs to the ring from which it receives light.
            elif kind == "monitor_pd":
                # the ring index selects the monitor coordinate row.
                if ring_index is None or not 0 <= ring_index < n_rings:
                    raise ValueError(f"monitor_pd site {name!r} needs a ring index")
                monitor_xy_um[ring_index] = position_um
            elif kind == "driver":
                # a driver without a ring index is shared by the rings.
                if ring_index is None:
                    shared_driver_xy_um = position_um
                # store the driver position with its assigned ring.
                elif 0 <= ring_index < n_rings:
                    driver_xy_um[ring_index] = position_um
            # a free heater or aggressor introduces an independent heat source.
            elif kind == "aggressor" or (kind == "heater" and (ring_index is None or "ring" not in spec)):
                aggressor_xy_um.append(position_um)
                aggressor_names.append(str(name))
            # keep unreferenced temperature sites as standalone observers.
            elif kind == "temp_sensor" and str(name) not in referenced_site_names:
                sensor_xy_um.append(position_um)
                sensor_rings.append(-1)
                sensor_names.append(str(name))

        # step 2 - fill the positions that no site gave.
        # missing rings go on a line along x at their index times the pitch.
        ring_is_missing = np.isnan(ring_xy_um[:, 0])
        # reconstruct missing ring centres only when the layout supplies enough geometry.
        if np.any(ring_is_missing):
            # multiple centres need a pitch to determine their relative displacement.
            if ring_pitch_um is None and n_rings > 1:
                raise ValueError(f"layout has no site for rings {np.flatnonzero(ring_is_missing).tolist()} and no "
                                 "layout.ring_pitch_um")
            # one unplaced ring can sit at the origin. multiple rings use the supplied pitch.
            pitch_um = 0.0 if ring_pitch_um is None else float(ring_pitch_um)
            # place each missing centre at its physical ring index times the nominal pitch.
            ring_xy_um[ring_is_missing, 0] = np.flatnonzero(ring_is_missing) * pitch_um
            ring_xy_um[ring_is_missing, 1] = 0.0
        # a shared driver location can supply otherwise missing per-ring driver coordinates.
        if shared_driver_xy_um is not None:
            driver_xy_um[np.isnan(driver_xy_um[:, 0])] = shared_driver_xy_um

        # step 3 - place each requested temperature sensor.
        bank_centroid_um = ring_xy_um.mean(axis=0)
        # resolve each declared temperature sensor to a physical measurement position.
        for sensor_index, spec in enumerate(temp_sensors or ()):
            name = str(spec.get("name", f"ts{sensor_index}"))
            # a named site supplies the sensor's coordinates directly.
            if spec.get("site"):
                site = sites[spec["site"]]
                sensor_xy_um.append((float(site["x_um"]), float(site["y_um"])))
                sensor_rings.append(-1)
            else:
                # without a named site, resolve the sensor's attachment relative to the die or a ring.
                attachment = str(spec.get("attach", "die"))
                # a ring attachment ties the sensor to a specific local ring.
                if attachment.startswith("ring_"):
                    ring_index = int(attachment.split("_", 1)[1])
                    sensor_xy_um.append(tuple(ring_xy_um[ring_index]))
                    sensor_rings.append(ring_index)
                # a bank attachment measures a site shared across the layout.
                elif attachment == "bank":
                    sensor_xy_um.append(tuple(bank_centroid_um))
                    sensor_rings.append(-1)
                else:   # die - assumed 1 mm from the bank centroid
                    sensor_xy_um.append((bank_centroid_um[0], bank_centroid_um[1] + 1000.0))
                    sensor_rings.append(-1)
            sensor_names.append(name)

        # step 4 - build the record. an empty table becomes None.
        # NaN rows - PD at the model's default offset, no driver for that ring
        has_monitor_site = not np.all(np.isnan(monitor_xy_um[:, 0]))
        has_driver_site = not np.all(np.isnan(driver_xy_um[:, 0]))
        # collect the resolved positions and use absent values for site families with no installed elements.
        return LayoutSites(
            ring_xy_um=ring_xy_um,
            pd_xy_um=monitor_xy_um if has_monitor_site else None,
            driver_xy_um=driver_xy_um if has_driver_site else None,
            temp_sensor_xy_um=np.array(sensor_xy_um, float).reshape(-1, 2) if sensor_xy_um else None,
            temp_sensor_ring=np.array(sensor_rings, np.int64) if sensor_xy_um else None,
            temp_sensor_names=tuple(sensor_names),
            aggressor_xy_um=np.array(aggressor_xy_um, float).reshape(-1, 2) if aggressor_xy_um else None,
            aggressor_names=tuple(aggressor_names),
        )

    # established method names and Python hooks share the same descriptors.
    __post_init__ = validation_check_layout_sites
    n_rings = helper_ring_count
    line = assembly_straight_ring_layout
    from_sites = assembly_layout_from_sites

# describe thermal uncertainty per world with scalar or (B,) factors.
# dc_scale multiplies residues.
# it is a scalar factor or a mapping with '*', '<source>-><observer>', '<source>->*', and '*-><observer>' keys.
# all matching factors multiply.
# ROM calibration uses rom_dc_scale=efficiency/(Z_FEM*dlambda_dT).
# time_scale_fast and time_scale_slow multiply time constants once per die, not once per ring.
# fast means tau<tau_split_s for ROM poles, or an interim self pole.
# slow means die/package/crosstalk.
# mixed-sign ROM pairs spanning both layers use one blended factor to preserve cancellation.
# see _whole_pair_time_scales.
# positive interim residues keep separate layer factors.
# scaling_hypothesis accepts a name, (B,) names, or codes in SCALING_HYPOTHESES order.
# 'resistance' scales tau with dc_scale because tau=RC.
# 'capacitance' and 'independent' leave the residue and time factors independent.
# let sigma be the matching DC scale and x the cross-ring coupling scale.
# cross-ring residues obey r_new=sigma*x*r.
# time constants obey tau_new=layer_factor*tau*(sigma if resistance else 1).
# crosstalk_scale never changes tau.
# it describes uncertain coupling magnitude rather than one RC resistance.
# g26 spans 1–5.2% nearest-neighbour coupling with a 10% corner.
# cross-path time has its own distance law and slow time factor.
# see the thermal notes, section 2.4.
# package_tail_share_frac assigns part of an eligible path's steady temperature rise to the slow package lag.
# package_tail_tau_s sets how fast that contribution develops.
# the uncertainty intervals are 1–10 ms and 2–10%.
# None retains the model's own tail or no ROM tail.
# a supplied tail time is absolute.
# do not multiply it by the slow factor or resistance factor.
# InterimRom.tail_tau_s is a family value and does receive time_scale_slow.
# transport_delay_s stores heat-transport delay, with an uncertainty interval of 0–2 us.
# the heater FIFO applies it in whole ticks.
# thermal kernels do not apply the delay again.
@dataclass(frozen=True, eq=False)
class ThermalScales:
    dc_scale: Any = 1.0
    time_scale_fast: Any = 1.0
    time_scale_slow: Any = 1.0
    scaling_hypothesis: Any = "resistance"
    crosstalk_scale: Any = 1.0
    package_tail_tau_s: Any = None
    package_tail_share_frac: Any = None
    tau_split_s: float = DEFAULT_TAU_SPLIT_S
    transport_delay_s: Any = 0.0


# describe interim heat paths until the heater family has a FEM ROM.
# field names follow ring.thermal.* in mrm.params.schema and use SI units.
# fields accept scalars or (B,) arrays.
# z_heater_ring_k_per_w also accepts (B, N).
# let Z0=z_heater_ring_k_per_w and S_H be the normalized heater-to-ring shape.
# S_H(s) = f_F/(1+s*tau_F) + (1-f_F-f_T)/(1+s*tau_S) + f_T/(1+s*tau_T)
# Z_RH(s) = Z0*S_H(s)
# Z_RR(s) = z*Z0*[f_sh/(1+s*tau_sh) + (1-f_sh)*S_H(s)]
# Z_HH(s) = Z0*[S_H(s) + (h-1)/(1+s*tau_hn)]
# Z_RH = Z_HR
# f_F is share_fast_frac.
# f_T is tail_share_frac.
# tau_F, tau_S, and tau_T are tau_fast_s, tau_slow_s, and tail_tau_s.
# z is z_ratio_opt_heater.
# f_sh and tau_sh are selfheat_share_frac and selfheat_tau_s.
# h and tau_hn are heater_self_ratio and heater_node_tau_s.
# reciprocity makes heater-to-ring equal to waveguide-absorption-to-heater.
# junction heat uses the waveguide residues.
# f_sh defaults to (z-1)/z.
# this is the largest share that keeps each pole's heater/ring 2x2 residue matrix positive semidefinite, as required for a passive RC network.
# ring-to-ring coupling is dlambda_victim/dlambda_heated.
# xt_nearest_frac sets nearest neighbours.
# xt_next_frac optionally sets second neighbours.
# xt_floor_frac sets the rest.
# each cross path uses crosstalk_tau_s, or S_H when crosstalk_dynamics='self'.
# PDs, drivers, sensors, and aggressors use site_fraction and the same crosstalk time law.
# site_fraction uses centre distance.
# crosstalk_tau_s uses source-to-victim distance - centre distance minus both ring-element radii.
# point sites have zero radius.
# site_floor_frac sets the site-law floor.
# thermal notes means docs/research/04_thermal_and_control.md.
# uncertainty means preset intervals.
# z_heater_ring_k_per_w = 6.1e3 K/W - derived under H_solver using the mode-solver dlambda/dT.
# realize passes heater efficiency divided by dlambda/dT.
# z_ratio_opt_heater = 1.5 - assumed midpoint of 1–2.
# reciprocity requires >=1 for an external heater.
# tau_fast_s = 3.2e-6 and tau_slow_s = 13.0e-6 - inferred from [CO22-TCPMT] Table I's 3.18/13.01 us fit.
# see thermal notes, row T2.
# share_fast_frac = 0.4085 = 0.43*(1-0.05) includes the tail share.
# tail_tau_s = 4e-3 and tail_share_frac = 0.05 - assumed inside 1–10 ms and 2–10% intervals. [CO22-TCPMT] reports a circuit-level 4 ms response.
# selfheat_tau_s = 0.5e-6 - de Cea 2019, about 0.5 us.
# compact-model notes, section 6.3.
# the literature band is 0.27–1.6 us, section 9.1.
# heater_self_ratio = 1.5 and heater_node_tau_s = 1e-6 - assumed.
# only R_th,H>=R_th is known.
# xt_nearest_frac = 0.03 and xt_floor_frac = 0.015 - inferred from thermal notes, rows X1–X5, section 2.5.
# xt_tau_ref_s = 4.1e-3 at xt_dist_ref_um = 60, with xt_tau_min_s = 1e-4 - near-field d^2 law.
# see [CO22-TCPMT] equation 4 and thermal notes, row X2, section 2.4.
# xt_tau_far_s = None - interpolate the printed X2 table.
# xt_far_um = 100 ends the near field.
# site_ref_um = 60 - assumed site law anchored at the paper's first-neighbour distance.
# site_floor_frac = None - use xt_floor_frac.
# imec measured 1.3–2.6% at 100–300 um, nearly constant beyond 200 um.
# this is a die/package term.
# the FEM's TEC-held package gives only 0.09% at 1 mm.
# ring_radius_um = 7.5 - about 15 um diameter in Kumar, OFC 2026 [LM26-S] slide 7, and Bunandar, Hot Chips 2025 [HC] slides 15–16.
# the FEM uses it too.
# accept scalar, (B,), or (B, N).
# realize passes each ring's r_ref from relation R3.
# pd_offset_um = (0, 25) and temp_sensor_offset_um = (40, 0) - assumed, matching FEM ThermalConfig.
# driver_offset_um = None - no default driver source.
# G26 uses a separate EIC beside the PIC.
# the FEM monolithic-driver offset is (0, -30).
# x
@dataclass(frozen=True, eq=False)
class InterimRom:
    z_heater_ring_k_per_w: Any = 6.1e3
    z_ratio_opt_heater: Any = 1.5
    tau_fast_s: Any = 3.2e-6
    share_fast_frac: Any = 0.4085
    tau_slow_s: Any = 13.0e-6
    tail_tau_s: Any = 4.0e-3
    tail_share_frac: Any = 0.05
    selfheat_tau_s: Any = 0.5e-6
    selfheat_share_frac: Any = None
    heater_self_ratio: Any = 1.5
    heater_node_tau_s: Any = 1.0e-6
    xt_nearest_frac: Any = 0.03
    xt_floor_frac: Any = 0.015
    xt_next_frac: Any = None
    xt_tau_ref_s: Any = 4.1e-3
    xt_dist_ref_um: float = 60.0
    xt_tau_min_s: Any = 1.0e-4
    xt_tau_far_s: Any = None
    xt_far_um: float = 100.0
    xt_tau_law: str = "capped"
    crosstalk_dynamics: str = "imec"
    site_ref_um: float = 60.0
    site_max_frac: float = 1.0
    site_floor_frac: Any = None
    ring_radius_um: Any = 7.5
    pd_offset_um: tuple[float, float] = (0.0, 25.0)
    temp_sensor_offset_um: tuple[float, float] = (40.0, 0.0)
    driver_offset_um: tuple | None = None
    name: str = "interim_three_pole"

    # construct the unverified textbook thermal corner.
    # use 10 us and 300 us self-path terms with no package tail.
    # set nearest and next-nearest coupling to 10 percent and 3 percent.
    # the default interim model uses the separate literature-based parameters.
    @staticmethod
    def assembly_textbook_thermal_corner(
        **kw: Any,  # kw = overrides (field units) - InterimRom field overrides - type keyword values with InterimRom field types
    ) -> (
        InterimRom  # return = model (mixed units) - interim parameters for the textbook corner - type InterimRom record
    ):
        base = dict(tau_fast_s=10e-6, share_fast_frac=0.75, tau_slow_s=300e-6, tail_share_frac=0.0,
                    xt_nearest_frac=0.10, xt_next_frac=0.03, xt_floor_frac=0.0, crosstalk_dynamics="self",
                    name="interim_textbook_corner")
        base.update(kw)
        return InterimRom(**base)

    # established method names and Python hooks share the same descriptors.
    textbook_corner = assembly_textbook_thermal_corner


# section 5 - distance laws
# two separate rules estimate a heat path between separated sites.
# the first gives its warmup time. the second gives its settled gain as a fraction of the heater-to-ring gain.

# in the full picture - assembly needs a warmup time tau (s) for a heat path between two separated places.
# an example is a heater on one ring that warms a neighbouring ring.
# heat takes longer to arrive over a longer distance. this function estimates that time from the gap d (um).
# it sets only how fast the far temperature rises. other rules set how far it rises.
# near the source, diffusion time grows with distance squared - tau_near = tau_ref*(d/d_ref)^2.
# by default that is 4.1 ms at 60 um, so 30 um gives about 1.0 ms.
# farther away, measured warmup times grow much more slowly, so tau_far comes from a measured table or a supplied value.
# tau_min keeps the estimate above zero when the gap is very small.
# calculate a crosstalk time constant from source-to-victim distance.
# the interim model subtracts the source and observer radii from centre spacing.
# the near-field law is max(tau_min,tau_ref*(d/d_ref)^2).
# the capped law takes the smaller near-field and far-field times, then applies tau_min.
# the literal law switches at far_um and can be nonmonotone.
# tau_far_s=None uses the measured distance table.
# x
def physics_crosstalk_time_constant_s(
    d_um: ArrayLike,  # d_um = d (um) - source-to-victim distance or site centre distance - type array-like, broadcast input shape
    tau_ref_s: ArrayLike = 4.1e-3,  # tau_ref_s = tau_ref (s) - reference or minimum crosstalk time constant - type array-like, broadcast input shape
    dist_ref_um: float = 60.0,  # dist_ref_um = d_ref (um) - reference distance for the diffusion law - type float scalar
    tau_min_s: ArrayLike = 1.0e-4,  # tau_min_s = tau_min (s) - reference or minimum crosstalk time constant - type array-like, broadcast input shape
    tau_far_s: ArrayLike | None = None,  # tau_far_s = tau_far (s) - optional far-field time constant, or the extracted distance table - type array-like or None, broadcast with gap and near-field times
    far_um: float = 100.0,  # far_um = d_far (um) - near-field cutoff for the literal distance law - type float scalar
    law: str = "capped",  # law = law (1) - capped or literal distance law - type str
) -> (
    RealValue  # return = tau_cross (s) - crosstalk time constant at each distance - type real scalar or float64 ndarray, shape broadcast input shape
):
    d = np.asarray(d_um, dtype=np.float64)
    # the near-field diffusion law makes twice the reference gap take four times as long, before the limits apply.
    near = np.maximum(np.asarray(tau_min_s, float), np.asarray(tau_ref_s, float) * (d / float(dist_ref_um)) ** 2)
    # use the measured distance table when a far-field time is unspecified.
    if tau_far_s is None:
        # interpolate the measured rise time between the two nearest tabulated separations.
        far = np.interp(d, XT_TABLE_D_UM, XT_TABLE_TAU_S)
    else:
        # a supplied far-field time replaces the distance-table estimate.
        far = np.broadcast_to(np.asarray(tau_far_s, float), np.broadcast(d, np.asarray(tau_far_s, float)).shape)
    # the literal law switches at d_far. the capped law limits the near-field time to the far-field value.
    if law == "literal":
        return np.where(d <= float(far_um), near, far)
    # the capped and literal models are the supported distance laws.
    if law != "capped":
        raise ValueError(f"crosstalk tau law must be 'capped' or 'literal', got {law!r}")
    # cap the near-field diffusion time at the far-field value and enforce the minimum rise time.
    return np.maximum(np.minimum(near, far), np.asarray(tau_min_s, float))


# in the full picture - assembly needs the settled gain of a heat path that involves a monitor photodiode, driver, sensor, or aggressor.
# settled gain means the final temperature rise per watt (K/W) after the warmup has finished.
# this function gives that gain as a fraction c (1) of Z0, the gain from a heater to its own ring (K/W).
# the caller multiplies - gain of the path = c*Z0.
# c falls as 1/d with centre distance d (um). a floor and a cap bound it.
# example with the defaults - at 180 um, 0.03*60/180 = 0.01, and the floor raises it to 0.015.
# with Z0 = 6100 K/W and 1 mW of heat, that path settles at 0.015*6100*0.001 = 0.09 K.
# estimate a site-path DC gain relative to the heater-to-ring gain.
# c(d)=clip(nearest_frac*ref_um/d,floor_frac,max_frac).
# use centre distance for this assumed law.
# the nonzero floor represents the long-range die or package term.
# x
def physics_site_coupling_fraction(
    d_um: ArrayLike,  # d_um = d (um) - source-to-victim distance or site centre distance - type array-like, broadcast input shape
    nearest_frac: ArrayLike = 0.03,  # nearest_frac = c_nearest (1) - coupling fraction at the reference distance - type array-like, broadcast input shape
    floor_frac: ArrayLike = 0.015,  # floor_frac = c_floor (1) - minimum site coupling fraction - type array-like, broadcast input shape
    ref_um: float = 60.0,  # ref_um = d_ref (um) - reference distance for site coupling - type float scalar
    max_frac: float = 1.0,  # max_frac = c_max (1) - maximum site coupling fraction - type float scalar
) -> (
    RealValue  # return = c_site (1) - site coupling relative to heater-to-ring DC gain - type real scalar or float64 ndarray, shape broadcast input shape
):
    # use a small positive distance at contact so the inverse-distance gain stays defined.
    d = np.maximum(np.asarray(d_um, dtype=np.float64), 1e-6)
    # double the centre distance to halve the raw coupling. the floor and cap limit that trend.
    c = np.asarray(nearest_frac, float) * float(ref_um) / d
    return np.clip(c, np.asarray(floor_frac, float), float(max_frac))


# section 6 - assembled system

# the assembled MIMO thermal plant of one bank for B uncertainty worlds.
# arrays (numba-ready) - taus_s, res_k_per_w (B, M) s and K/W.
# lag_src, lag_obs, lag_kind, lag_pair, lag_layer, lag_tail (M,) int64, lags sorted by observer with obs_ptr (O+1,).
# src_kind, src_ring (S,) and obs_kind, obs_ring (O,) int64 (ring -1 for free sites).
# per-ring index maps ring_obs, heater_obs, pd_obs, heater_src, wg_src, junction_src, driver_src (N,) (-1 when absent).
# temp_obs (T,), aggressor_src (A,).
# opt_lags (M_opt,) the lags of kind LAG_OPT_SELF or LAG_OPT_OTHER (the second pass of selfheat.implicit_step).
# names - src_names, obs_names.
# transport_delay_s (B,) s - ThermalScales.transport_delay_s, for the heater actuator FIFO (not applied here).
# x
@dataclass(frozen=True, eq=False)
class ThermalSystem:
    taus_s: Float64Array
    res_k_per_w: Float64Array
    lag_src: Int64Array
    lag_obs: Int64Array
    lag_kind: Int64Array
    lag_pair: Int64Array
    lag_layer: Int64Array
    lag_tail: Int64Array
    obs_ptr: Int64Array
    src_kind: Int64Array
    src_ring: Int64Array
    src_names: tuple[str, ...]
    obs_kind: Int64Array
    obs_ring: Int64Array
    obs_names: tuple[str, ...]
    ring_obs: Int64Array
    heater_obs: Int64Array
    pd_obs: Int64Array
    temp_obs: Int64Array
    heater_src: Int64Array
    wg_src: Int64Array
    junction_src: Int64Array
    driver_src: Int64Array
    aggressor_src: Int64Array
    opt_lags: Int64Array
    n_rings: int
    meta: dict[str, Any] = field(default_factory=dict)
    transport_delay_s: Float64Array | None = None

    # number of worlds B.
    @property
    def helper_world_count(
        self: ThermalSystem,  # self = self (mixed units) - thermal lag gains, time constants and their source and observer indices - type ThermalSystem record
    ) -> (
        int  # return = B (1) - number of worlds - type int scalar
    ):
        return self.taus_s.shape[0]

    # number of lags M.
    @property
    def helper_lag_count(
        self: ThermalSystem,  # self = self (mixed units) - thermal lag gains, time constants and their source and observer indices - type ThermalSystem record
    ) -> (
        int  # return = M (1) - number of thermal lag states - type int scalar
    ):
        return self.taus_s.shape[1]

    # number of sources S.
    @property
    def helper_source_count(
        self: ThermalSystem,  # self = self (mixed units) - thermal lag gains, time constants and their source and observer indices - type ThermalSystem record
    ) -> (
        int  # return = S (1) - number of heat sources - type int scalar
    ):
        return self.src_kind.shape[0]

    # number of observers O.
    @property
    def helper_observer_count(
        self: ThermalSystem,  # self = self (mixed units) - thermal lag gains, time constants and their source and observer indices - type ThermalSystem record
    ) -> (
        int  # return = O (1) - number of temperature observers - type int scalar
    ):
        return self.obs_kind.shape[0]

    # exact FOH coefficients (a, c0, c1), each (B, M), for a step h_s (s).
    # see foh_coefficients.
    # axes - B = simulated worlds, M = thermal lags.
    def physics_linear_hold_coefficients(
        self: ThermalSystem,  # self = self (mixed units) - thermal lag gains, time constants and their source and observer indices - type ThermalSystem record
        h_s: float,  # h_s = h (s) - thermal step duration - type float scalar
    ) -> tuple[
        Float64Array,  # return[0] = a (1) - retained state fractions - type float64 ndarray, shape (B, M)
        Float64Array,  # return[1] = c_start (K/W) - starting-power gains - type float64 ndarray, shape (B, M)
        Float64Array,  # return[2] = c_end (K/W) - ending-power gains - type float64 ndarray, shape (B, M)
    ]:
        # convert each lag's time constant and settled rise per watt into coefficients for the requested linear power step.
        return physics_linear_hold_coefficients(self.taus_s, self.res_k_per_w, h_s)

    # all-zero lag states (B, M) (every observer at T_base).
    # axes - B = simulated worlds, M = thermal lags.
    def helper_zero_state(
        self: ThermalSystem,  # self = self (mixed units) - thermal lag gains, time constants and their source and observer indices - type ThermalSystem record
        batch: int | None = None,  # batch = B (1) - optional number of worlds - type int or None
    ) -> (
        Float64Array  # return = x (K) - zero initial temperature rises - type float64 ndarray, shape (B, M)
    ):
        # allocate one zero rise per lag for either the model batch or the caller's requested number of worlds.
        return np.zeros((self.helper_world_count if batch is None else int(batch), self.helper_lag_count))

    # lag states (B, M) in steady state for constant source powers u_w (B|1, S) W - x_m = r_m u_src(m).
    # axes - B = simulated worlds, S = heat sources, M = thermal lags.
    def physics_steady_state(
        self: ThermalSystem,  # self = self (mixed units) - thermal lag gains, time constants and their source and observer indices - type ThermalSystem record
        u_w: ArrayLike,  # u_w = P_source (W) - constant source powers - type array-like, shape (S,) or (B, S)
    ) -> (
        Float64Array  # return = x_ss (K) - steady lag-state temperature rises - type float64 ndarray, shape (B, M)
    ):
        u = np.atleast_2d(np.asarray(u_w, dtype=np.float64))
        # dx_m/dt = 0 gives x_m = r_m*P_source. lag_src selects that power for each lag.
        return self.res_k_per_w * u[:, self.lag_src]

    # observer temperatures (B, O) = t_base_k + sum of lag states (Python form of thermal_outputs).
    # axes - B = simulated worlds, O = temperature observers, M = thermal lags.
    def physics_observer_temperatures(
        self: ThermalSystem,  # self = self (mixed units) - thermal lag gains, time constants and their source and observer indices - type ThermalSystem record
        x: ArrayLike,  # x = x (K) - lag-state temperature rises - type array-like, shape (B, M)
        t_base_k: ArrayLike = 0.0,  # t_base_k = T_base (K) - base temperature of each world - type array-like, scalar or shape (1 or B,)
    ) -> (
        Float64Array  # return = T_observer (K) - absolute observer temperatures - type float64 ndarray, shape (B, O)
    ):
        x = np.atleast_2d(x)
        out = np.zeros((x.shape[0], self.helper_observer_count))
        physics_write_observer_temperatures(x, self.obs_ptr, np.atleast_1d(np.asarray(t_base_k, np.float64)), out)
        return out

    # index of a source by kind and ring (or by name for aggressors).
    def helper_source_index(
        self: ThermalSystem,  # self = self (mixed units) - thermal lag gains, time constants and their source and observer indices - type ThermalSystem record
        kind: str,  # kind = kind (1) - source or observer kind name - type str
        ring: int | None = None,  # ring = k (1) - physical ring index or no ring filter - type int or None
        name: str | None = None,  # name = name (1) - optional exact element name - type str or None
    ) -> (
        int  # return = s (1) - matching heat-source index - type int scalar
    ):
        # map a named heat source or a unique kind-and-ring match to its power-input column.
        return helper_find_thermal_element_index(self.src_names, self.src_kind, self.src_ring, SOURCE_KINDS, kind, ring, name)

    # index of an observer by kind and ring (or by name for temperature sensors).
    def helper_observer_index(
        self: ThermalSystem,  # self = self (mixed units) - thermal lag gains, time constants and their source and observer indices - type ThermalSystem record
        kind: str,  # kind = kind (1) - source or observer kind name - type str
        ring: int | None = None,  # ring = k (1) - physical ring index or no ring filter - type int or None
        name: str | None = None,  # name = name (1) - optional exact element name - type str or None
    ) -> (
        int  # return = o (1) - matching temperature-observer index - type int scalar
    ):
        # map a named readout or a unique kind-and-ring match to its temperature-output column.
        return helper_find_thermal_element_index(self.obs_names, self.obs_kind, self.obs_ring, OBSERVER_KINDS, kind, ring, name)

    # (lag_src, lag_obs, lag_kind, opt_lags, obs_ptr, src_ring, ring_obs) - the index arrays of selfheat.implicit_step, in its argument order.
    # axes - N = physical rings, S = heat sources, O = temperature observers, M = thermal lags, M_opt = optical heating lags.
    def helper_kernel_array_views(
        self: ThermalSystem,  # self = self (mixed units) - thermal lag gains, time constants and their source and observer indices - type ThermalSystem record
    ) -> tuple[
        Int64Array,  # return[0] = s_m (1) - source index of each lag - type int64 ndarray, shape (M,)
        Int64Array,  # return[1] = o_m (1) - observer index of each lag - type int64 ndarray, shape (M,)
        Int64Array,  # return[2] = kind_m (1) - heat-path kind of each lag - type int64 ndarray, shape (M,)
        Int64Array,  # return[3] = m_opt (1) - optical lag indices - type int64 ndarray, shape (M_opt,)
        Int64Array,  # return[4] = p_observer (1) - observer boundaries in the lag array - type int64 ndarray, shape (O+1,)
        Int64Array,  # return[5] = k_source (1) - ring attachment of each source - type int64 ndarray, shape (S,)
        Int64Array,  # return[6] = o_ring (1) - mode-temperature observer of each ring - type int64 ndarray, shape (N,)
    ]:
        return (
            self.lag_src,
            self.lag_obs,
            self.lag_kind,
            self.opt_lags,
            self.obs_ptr,
            self.src_ring,
            self.ring_obs,
        )

    # established method names and Python hooks share the same descriptors.
    batch = helper_world_count
    n_lags = helper_lag_count
    n_src = helper_source_count
    n_obs = helper_observer_count
    coefficients = physics_linear_hold_coefficients
    zero_state = helper_zero_state
    steady_state = physics_steady_state
    outputs = physics_observer_temperatures
    source_index = helper_source_index
    observer_index = helper_observer_index
    kernel_arrays = helper_kernel_array_views



# index of the one element with this name, or with this kind and ring.
# raises KeyError otherwise.
# x
def helper_find_thermal_element_index(
    names: tuple[str, ...],  # names = names (1) - ordered element names - type tuple of strings
    kinds: Int64Array,  # kinds = kind_indices (1) - kind code of each element - type int64 ndarray, shape (element_count,)
    rings: Int64Array,  # rings = ring_indices (1) - ring attachment of each element - type int64 ndarray, shape (element_count,)
    kind_list: tuple[str, ...],  # kind_list = kind_names (1) - ordered source or observer kind names - type tuple of strings
    kind: str,  # kind = kind (1) - source or observer kind name - type str
    ring: int | None,  # ring = k (1) - physical ring index or no ring filter - type int or None
    name: str | None,  # name = name (1) - optional exact element name - type str or None
) -> (
    int  # return = index (1) - unique matching element index - type int scalar
):
    # a named lookup identifies one element without a ring-and-kind search.
    if name is not None:
        # return the stored position of the matching name.
        if name in names:
            return names.index(name)
        # an unknown explicit name cannot select a thermal element.
        raise KeyError(f"no element named {name!r}")
    k = kind_list.index(kind)
    # restrict the lookup to the requested kind and optional physical ring.
    hit = np.flatnonzero((kinds == k) & ((rings == ring) if ring is not None else True))
    # a kind-and-ring lookup must identify exactly one element.
    if hit.size != 1:
        raise KeyError(f"{kind} ring={ring}: {hit.size} matches")
    return int(hit[0])


# section 7 - assembly
# assemble is the entry point and sits near the end of this section. read it first.
# the functions before it appear in the order assemble uses them.
# - small input helpers and the two builders _Lags and _Elements.
# - place sources and observers - assembly_thermal_sources_and_observers.
# - choose r and tau per heat path - assembly_append_interim_heat_path_lags or assembly_append_extracted_heat_path_lags.
# - package share - physics_replace_package_tail_preserving_dc_gain.
# - chip uncertainty - physics_pair_dc_scale and physics_shared_pair_time_scales.
# - the stage functions that assemble calls, then assemble.

# a scalar, (B,) or (B, 1) parameter as a 1-D array of length 1 or B.
# x
# axes - B = simulated worlds.
def helper_normalize_world_parameter_vector(
    values: ArrayLike,  # values = values (input units) - parameter values to normalize - type array-like, scalar, (B,), or (B, 1)
    name: str,  # name = name (1) - element name or diagnostic parameter label - type str
) -> (
    Float64Array  # return = parameter (input units) - normalized world-parameter vector - type float64 ndarray, shape (1 or B,)
):
    parameter_array = np.asarray(values, dtype=np.float64)
    # a scalar specifies one shared world parameter.
    if parameter_array.ndim == 0:
        return parameter_array.reshape(1)
    # a single-column batch stores one value per simulated world.
    if parameter_array.ndim == 2 and parameter_array.shape[1] == 1:
        return parameter_array[:, 0]
    # each world needs one scalar parameter after normalization.
    if parameter_array.ndim != 1:
        raise ValueError(f"{name}: expected a scalar or (B,), got shape {parameter_array.shape}")
    return parameter_array


# the common batch length B of 1-row or B-row arrays.
# raises ValueError for two different lengths.
# x
# axes - B = simulated worlds.
def helper_infer_world_count(
    values: dict[str, Float64Array],  # values = parameters (parameter units) - world-parameter arrays keyed by parameter name - type dict of float64 ndarrays with 1 or B rows, each value shared or shape (B,)
) -> (
    int  # return = B (1) - common world count inferred from parameter arrays - type int scalar
):
    world_count = 1
    # infer the number of worlds from parameters that carry separate rows.
    for name, parameter_array in values.items():
        row_count = parameter_array.shape[0]
        # a singleton row broadcasts and does not constrain the batch size.
        if row_count != 1:
            # all nonshared parameter arrays must agree on the number of worlds.
            if world_count not in (1, row_count):
                raise ValueError(f"inconsistent batch sizes: {name} has {row_count}, another has {world_count}")
            world_count = row_count
    return world_count


# scaling_hypothesis (a name, names or int codes) as B int codes into SCALING_HYPOTHESES.
# x
# (todo) what is this function??
# axes - B = simulated worlds.
def helper_encode_thermal_scaling_hypotheses(
    hypotheses: str | Sequence[str] | ArrayLike,  # hypotheses = hypotheses (1) - scaling hypothesis names or integer codes - type str, sequence of strings, or integer array-like, scalar or shape (B,)
    world_count: int,  # world_count = B (1) - number of worlds - type int scalar
) -> (
    Int64Array  # return = hypothesis (1) - integer scaling-hypothesis codes - type int64 ndarray, shape (B,)
):
    # a single hypothesis name describes a one-entry selection.
    if isinstance(hypotheses, str):
        hypotheses = [hypotheses]
    hypothesis_values = np.asarray(hypotheses)
    # integer inputs already identify the scaling laws by index.
    if hypothesis_values.dtype.kind in "iu":
        codes = hypothesis_values.astype(np.int64).reshape(-1)
    else:
        # translate the requested names to the stored hypothesis codes.
        try:
            codes = np.array([SCALING_HYPOTHESES.index(str(x)) for x in hypothesis_values.reshape(-1)], np.int64)
        except ValueError as e:
            # report the unrecognized scaling law using its public parameter name.
            raise ValueError(f"scaling_hypothesis must be one of {SCALING_HYPOTHESES}, got {hypotheses!r}") from e
    # every code must select an entry in the hypothesis table.
    if np.any((codes < 0) | (codes >= len(SCALING_HYPOTHESES))):
        raise ValueError(f"scaling_hypothesis codes out of range: {codes}")
    # a hypothesis applies to either the whole batch or one world per entry.
    if codes.shape[0] not in (1, world_count):
        raise ValueError(f"scaling_hypothesis has {codes.shape[0]} entries, batch is {world_count}")
    return np.broadcast_to(codes, (world_count,)).copy()


# collect source-to-observer lag terms before sorting and scaling.
# time_constants_s and residues_k_per_w store per-world values in s and K/W.
# source_indices, observer_indices, layer_codes, and pair_codes are dimensionless indices.
# x
class _Lags:

    # initialize the builder without changing the physical model.
    def assembly_initialize_heat_path_storage(
        self: _Lags,  # self = self (mixed units) - lists of fitted gains, time constants and path indices - type _Lags record
        B: int,  # B = B (1) - number of worlds - type int scalar
    ) -> (
        None  # return = None - initialize the mutable storage - type None
    ):
        self.world_count = B
        # each list has one entry per lag. entry m of every list describes lag m.
        self.observer_indices = []  # observer that reads the lag
        self.source_indices = []  # source that drives the lag
        self.time_constants_s = []  # tau (s) - one value per world - shape (B,)
        self.residues_k_per_w = []  # r (K/W) - one value per world - shape (B,)
        self.layer_codes = []  # LAYER_FAST or LAYER_SLOW - selects the time-scale factor
        self.pair_codes = []  # PAIR_SELF, PAIR_CROSS, or PAIR_SITE
        self.tail_replaceable = []  # the lag belongs to a path whose package share can be reassigned
        self.tail_flags = []  # the lag is the package lag of its path
        self.fixed_time_constants = []  # tau is absolute and takes no time-scale factor

    # append a lag with its source, observer, time constant, and thermal residue.
    # axes - B = simulated worlds.
    def assembly_append_heat_path_lag(
        self: _Lags,  # self = self (mixed units) - lists of fitted gains, time constants and path indices - type _Lags record
        obs: int,  # obs = obs (1) - observer index - type int scalar
        src: int,  # src = src (1) - source index - type int scalar
        tau: ArrayLike,  # tau = tau (s) - lag time constant for each world - type array-like, scalar or shape (B,)
        res: ArrayLike,  # res = r (K/W) - thermal residue for each world - type array-like, scalar or shape (B,)
        layer: int,  # layer = layer (1) - fast or slow layer code - type int scalar
        pair: int,  # pair = pair (1) - self, cross-ring, or site pair code - type int scalar
        tailable: bool = False,  # tailable = tailable (1) - pair permits replacing its package tail - type bool scalar
        tail: bool = False,  # tail = tail (1) - this lag is a package tail - type bool scalar
        fixed_tau: bool = False,  # fixed_tau = fixed_tau (1) - keep the lag time constant absolute - type bool scalar
    ) -> (
        None  # return = None - append one thermal lag to the builder - type None
    ):
        self.observer_indices.append(obs)
        self.source_indices.append(src)
        self.time_constants_s.append(np.broadcast_to(np.asarray(tau, float), (self.world_count,)).copy())
        self.residues_k_per_w.append(np.broadcast_to(np.asarray(res, float), (self.world_count,)).copy())
        self.layer_codes.append(layer)
        self.pair_codes.append(pair)
        # record whether the optional package tail can redistribute this lag's gain.
        self.tail_replaceable.append(bool(tailable))
        self.tail_flags.append(bool(tail))
        self.fixed_time_constants.append(bool(fixed_tau))    # absolute tau - no time_scale, no 'resistance' factor

    # established method names and Python hooks share the same descriptors.
    __init__ = assembly_initialize_heat_path_storage


# source and observer instances of the bank with their positions (um) and ring attachment.
# x
class _Elements:

    # initialize the builder without changing the physical model.
    def assembly_initialize_site_storage(
        self: _Elements,  # self = self (mixed units) - names, positions and ring assignments of sources and observers - type _Elements record
    ) -> (
        None  # return = None - initialize the mutable storage - type None
    ):
        self.src_kind, self.src_ring, self.src_names, self.src_xy = [], [], [], []
        self.obs_kind, self.obs_ring, self.obs_names, self.obs_xy = [], [], [], []

    # append a heat source and return its source index.
    def assembly_append_heat_source(
        self: _Elements,  # self = self (mixed units) - names, positions and ring assignments of sources and observers - type _Elements record
        kind: int,  # kind = kind (1) - element kind code - type int scalar
        ring: int,  # ring = ring (1) - ring attachment, with -1 for a free site - type int scalar
        name: str,  # name = name (1) - element name or diagnostic parameter label - type str
        xy: ArrayLike,  # xy = r_site (um) - site coordinates - type array-like, shape (2,)
    ) -> (
        int  # return = s (1) - index of the appended source - type int scalar
    ):
        self.src_kind.append(kind)
        self.src_ring.append(ring)
        self.src_names.append(name)
        self.src_xy.append(np.asarray(xy, float))
        # the appended source occupies the final zero-based position.
        return len(self.src_kind) - 1

    # append a temperature observer and return its observer index.
    def assembly_append_temperature_observer(
        self: _Elements,  # self = self (mixed units) - names, positions and ring assignments of sources and observers - type _Elements record
        kind: int,  # kind = kind (1) - element kind code - type int scalar
        ring: int,  # ring = ring (1) - ring attachment, with -1 for a free site - type int scalar
        name: str,  # name = name (1) - element name or diagnostic parameter label - type str
        xy: ArrayLike,  # xy = r_site (um) - site coordinates - type array-like, shape (2,)
    ) -> (
        int  # return = o (1) - index of the appended observer - type int scalar
    ):
        self.obs_kind.append(kind)
        self.obs_ring.append(ring)
        self.obs_names.append(name)
        self.obs_xy.append(np.asarray(xy, float))
        # the appended observer occupies the final zero-based position.
        return len(self.obs_kind) - 1

    # established method names and Python hooks share the same descriptors.
    __init__ = assembly_initialize_site_storage


# build the bank's heat sources, temperature observers, and index maps.
# ring sources and ring or heater observers use ring-centre coordinates.
# photodiodes and attached sensors use their specified or default offsets.
# include a driver only where a site or default driver offset exists.
# turn the physical layout into two indexed lists - where power enters and where temperature is read.
def assembly_thermal_sources_and_observers(
    layout: LayoutSites,  # layout = sites (um) - source and observer site coordinates - type LayoutSites record
    n: int,  # n = N (1) - number of rings - type int scalar
    src_kinds: Sequence[str],  # src_kinds = source_kinds (1) - source kind names to include - type sequence of strings
    obs_kinds: Sequence[str],  # obs_kinds = observer_kinds (1) - observer kind names to include - type sequence of strings
    pd_offset: Sequence[float],  # pd_offset = Delta_r_PD (um) - default monitor offset from its ring centre - type two coordinates
    ts_offset: Sequence[float],  # ts_offset = Delta_r_sensor (um) - default attached-sensor offset - type two coordinates
    driver_offset: Sequence[float] | None = None,  # driver_offset = Delta_r_driver (um) - optional driver offset from its ring centre - type two coordinates or None
) -> tuple[
    _Elements,  # return[0] = elements (mixed units) - source and observer kinds, names, coordinates, and ring attachments - type thermal element builder
    dict[str, Int64Array],  # return[1] = indices (1) - per-ring, temperature-sensor, and aggressor index maps - type dict of named int64 ndarrays, one index vector per kind with length equal to its site count
]:

    elements = _Elements()

    # kind name              | kind code    | per-ring index map
    centre_sources = (
        ("heater",               SRC_HEATER,   "heater_src"),
        ("waveguide_absorption", SRC_WG,       "wg_src"),
        ("junction",             SRC_JUNCTION, "junction_src"),
    )
    centre_observers = (
        ("ring_mode_weighted",   OBS_RING,     "ring_obs"),
        ("heater_resistor",      OBS_HEATER,   "heater_obs"),
    )

    # each entry maps a ring to a source or observer index. -1 means the element is absent.
    per_ring_index_names = (
        "heater_src", "wg_src", "junction_src", "driver_src", "ring_obs", "heater_obs", "pd_obs",
    )
    indices = {name: np.full(n, -1, np.int64) for name in per_ring_index_names}
    ring_centres_um = layout.ring_xy_um

    # place the ring-local sources relative to each physical ring centre.
    for ring_index in range(n):
        ring_centre_um = ring_centres_um[ring_index]
        # the table maps source kinds to their codes and index arrays.
        for kind_name, kind_code, index_name in centre_sources:
            # install only the heat source types requested for this model.
            if kind_name in src_kinds:
                indices[index_name][ring_index] = elements.assembly_append_heat_source(
                    kind_code, ring_index, f"{kind_name}{ring_index}", ring_centre_um
                )

        # rings without driver heating need no driver position.
        if "driver" not in src_kinds:
            continue
        # use the layout position when a driver table exists. a nonfinite row marks an absent driver.
        if layout.driver_xy_um is not None:
            driver_position_um = layout.driver_xy_um[ring_index]
        # reconstruct the driver position from its fitted ring-relative offset.
        elif driver_offset is not None:
            # translate the measured ring-relative driver offset to absolute coordinates.
            driver_position_um = ring_centre_um + np.asarray(driver_offset, float)
        else:
            # skip the driver when neither a position nor an offset is available.
            continue
        # omit missing positions from the finite thermal layout.
        if np.all(np.isfinite(driver_position_um)):
            indices["driver_src"][ring_index] = elements.assembly_append_heat_source(
                SRC_DRIVER, ring_index, f"driver{ring_index}", driver_position_um
            )

    # aggressors use their layout positions with ring attachment -1 for a free site.
    aggressor_indices = []
    # free aggressors require both enabled heating and supplied positions.
    if "aggressor" in src_kinds and layout.aggressor_xy_um is not None:
        # each named aggressor adds an independent source of heat.
        for name, position_um in zip(layout.aggressor_names, layout.aggressor_xy_um):
            aggressor_indices.append(elements.assembly_append_heat_source(SRC_AGGRESSOR, -1, name, position_um))

    # place ring-local temperature observers at their physical sites.
    for ring_index in range(n):
        ring_centre_um = ring_centres_um[ring_index]
        # the table maps observer kinds to their lookup arrays.
        for kind_name, kind_code, index_name in centre_observers:
            # install only the requested temperature measurement types.
            if kind_name in obs_kinds:
                indices[index_name][ring_index] = elements.assembly_append_temperature_observer(
                    kind_code, ring_index, f"{kind_name}{ring_index}", ring_centre_um
                )

        if "monitor_pd" in obs_kinds:
            # use the model's offset when the layout has no finite monitor position.
            monitor_position_um = None
            # prefer the monitor location measured in the layout.
            if layout.pd_xy_um is not None:
                monitor_position_um = layout.pd_xy_um[ring_index]
            # use the fallback geometry when the supplied monitor location is missing.
            if monitor_position_um is None or not np.all(np.isfinite(monitor_position_um)):
                # translate the fitted monitor offset to the installed ring centre.
                monitor_position_um = ring_centre_um + np.asarray(pd_offset, float)
            indices["pd_obs"][ring_index] = elements.assembly_append_temperature_observer(
                OBS_PD, ring_index, f"monitor_pd{ring_index}", monitor_position_um
            )

    sensor_indices = []
    # temperature sensors need enabled readout and known positions.
    if "temp_sensor" in obs_kinds and layout.temp_sensor_xy_um is not None:
        # record one observer per named sensor at its own location.
        for sensor_index, name in enumerate(layout.temp_sensor_names):
            attached_ring = int(layout.temp_sensor_ring[sensor_index])
            # attached sensors use the offset of the ROM self table. free sensors use layout coordinates.
            if attached_ring >= 0:
                # place the attached sensor relative to its host ring.
                sensor_position_um = ring_centres_um[attached_ring] + np.asarray(ts_offset, float)
            else:
                # use the layout's absolute sensor position when the offset rule does not apply.
                sensor_position_um = layout.temp_sensor_xy_um[sensor_index]
            sensor_indices.append(elements.assembly_append_temperature_observer(OBS_TEMP, attached_ring, name, sensor_position_um))

    indices["temp_obs"] = np.array(sensor_indices, np.int64)
    # retain the lookup from each external aggressor to its thermal input.
    indices["aggressor_src"] = np.array(aggressor_indices, np.int64)
    return (
        elements,
        indices,
    )


# in the full picture - the interim model does not calculate ring-to-ring heating from the exact distance.
# it uses a few fixed coupling fractions - one for nearest neighbours, one for second neighbours, and one for all others.
# this function decides which of those bands each pair of rings is in. the caller picks the fraction from the rank.
# rank 1 is the nearest distinct distance from a ring. rank 2 is the next distinct distance. rank 99 is anything farther. rank 0 is a ring with itself.
# example - rings at 0, 100, and 200 um. ring 0 sees ring 1 at rank 1 and ring 2 at rank 2.
# the function also returns every centre-to-centre distance in um, which the caller uses for warmup times.
# rank ring pairs by their first and second distinct neighbour distances.
# (todo) helper func, move it out
# x
# axes - N = physical rings.
def physics_neighbor_ranks_and_distances(
    ring_xy: Float64Array,  # ring_xy = r_ring (um) - ring centre coordinates - type float64 ndarray, shape (N, 2)
) -> tuple[
    Int64Array,  # return[0] = rank (1) - symmetric neighbour ranks - type int64 ndarray, shape (N, N)
    Float64Array,  # return[1] = d (um) - ring centre distances - type float64 ndarray, shape (N, N)
]:

    ring_count = ring_xy.shape[0]
    # offset[i, j] is the vector from ring j to ring i. take its length across x and y.
    centre_offsets_um = ring_xy[:, None, :] - ring_xy[None, :, :]
    centre_distances_um = np.linalg.norm(centre_offsets_um, axis=-1)

    # rank 99 marks a pair outside the first two distinct neighbour distances.
    ranks_from_ring = np.full((ring_count, ring_count), 99, np.int64)
    ring_indices = np.arange(ring_count)
    relative_distance_tolerance = 1e-6
    absolute_distance_tolerance_um = 1e-9
    # measure each other ring's distance relative to this ring.
    for ring_index in range(ring_count):
        other_rings = ring_indices[ring_indices != ring_index]
        distances_to_other_rings_um = centre_distances_um[ring_index, other_rings]

        # sort the distances and group nearly equal spacings under the first distance in each group.
        distinct_distances_um = []
        # walk outward so equal-distance neighbours can share a rank.
        for distance_um in np.sort(distances_to_other_rings_um):
            # the nearest ring starts the first distance group.
            if not distinct_distances_um:
                distinct_distances_um.append(distance_um)
                # the first group has no preceding distance to compare.
                continue
            last_distance_um = distinct_distances_um[-1]
            # combine a relative tolerance with a fixed tolerance for very small spacings.
            group_limit_um = last_distance_um * (1 + relative_distance_tolerance) + absolute_distance_tolerance_um
            # start a new group only beyond the current group's tolerance.
            if distance_um > group_limit_um:
                distinct_distances_um.append(distance_um)

        # assign ranks one and two to the first two distinct distance groups.
        for neighbor_rank, group_distance_um in enumerate(distinct_distances_um[:2], start=1):
            # use the same spacing tolerance when assigning rings to a group.
            tolerance_um = group_distance_um * relative_distance_tolerance + absolute_distance_tolerance_um
            in_distance_group = np.abs(distances_to_other_rings_um - group_distance_um) <= tolerance_um
            ranks_from_ring[ring_index, other_rings[in_distance_group]] = neighbor_rank

    # use the closer rank assigned by either ring. rank 0 is reserved for a ring paired with itself.
    symmetric_ranks = np.minimum(ranks_from_ring, ranks_from_ring.T)
    np.fill_diagonal(symmetric_ranks, 0)
    return (
        symmetric_ranks,
        centre_distances_um,
    )


# broadcast one interim parameter over worlds and rings.
# axes - B = simulated worlds, N = physical rings.
def _helper_broadcast_interim_ring_parameter(
    n: int,  # n = N (1) - physical ring count - type int scalar
    v: ArrayLike,  # v = parameter (field units) - interim parameter to broadcast - type array-like, scalar, (B,), or (B, N)
    name: str,  # name = name (1) - element name or diagnostic parameter label - type str
) -> (
    Float64Array  # return = parameter (input units) - parameter broadcast over worlds and rings - type float64 ndarray, shape (1 or B, 1 or N)
):
    values = np.asarray(v, dtype=np.float64)
    # a scalar supplies the same parameter to all worlds and rings.
    if values.ndim == 0:
        return values.reshape(1, 1)
    # a vector supplies one value per world shared across its rings.
    if values.ndim == 1:
        return values[:, None]
    # a two-dimensional table already has the required world and ring axes.
    if values.ndim == 2 and values.shape[1] in (1, n):
        return values
    raise ValueError(f"InterimRom.{name}: expected scalar, (B,) or (B, N), got shape {values.shape}")



# InterimRom fields as (B|1, N|1) arrays, with defaults resolved and bounds checked.
# the returned dict uses short keys. the table in the body maps each key to its InterimRom field.
# express every interim thermal parameter on compatible world and ring axes before using it to construct a heat path.
# axes - B = simulated worlds, N = physical rings.
def validation_check_and_broadcast_interim_parameters(
    spec: InterimRom,  # spec = model (mixed units) - interim thermal resistance and time-constant parameters - type InterimRom record
    n: int,  # n = N (1) - number of rings - type int scalar
) -> (
    dict[str, Float64Array]  # return = parameters (field units) - validated interim parameters with defaults resolved - type dict of float64 ndarrays, shape (1 or B, 1 or N)
):

    # short key          | InterimRom field         | value used when the field is None
    parameter_fields = (
        ("z0",                "z_heater_ring_k_per_w",  None),
        ("z_ratio",           "z_ratio_opt_heater",     None),
        ("tau_fast",          "tau_fast_s",             None),
        ("share_fast",        "share_fast_frac",        None),
        ("tau_slow",          "tau_slow_s",             None),
        ("tail_tau",          "tail_tau_s",             None),
        ("tail_share",        "tail_share_frac",        None),
        ("selfheat_tau",      "selfheat_tau_s",         None),
        ("heater_self_ratio", "heater_self_ratio",      None),
        ("heater_node_tau",   "heater_node_tau_s",      None),
        ("xt_nearest",        "xt_nearest_frac",        None),
        ("xt_floor",          "xt_floor_frac",          None),
        ("xt_next",           "xt_next_frac",           0.0),
        ("xt_tau_ref",        "xt_tau_ref_s",           None),
        ("xt_tau_min",        "xt_tau_min_s",           None),
        ("xt_tau_far",        "xt_tau_far_s",           0.0),
        ("ring_radius",       "ring_radius_um",         None),
    )
    parameters = {}
    # convert each physical parameter into a world-by-ring array.
    for key, field_name, value_when_none in parameter_fields:
        field_value = getattr(spec, field_name)
        # use the model default when an optional parameter is absent.
        if field_value is None:
            field_value = value_when_none
        parameters[key] = _helper_broadcast_interim_ring_parameter(n, field_value, field_name)

    # the site floor defaults to the ring-to-ring floor.
    if spec.site_floor_frac is None:
        parameters["site_floor"] = parameters["xt_floor"]
    else:
        # the site floor controls the minimum coupling for a remote measurement site.
        parameters["site_floor"] = _helper_broadcast_interim_ring_parameter(n, spec.site_floor_frac, "site_floor_frac")

    # by default z*(1-f_sh) = 1. the optical gain left on the heater lags then equals the heater gain.
    optical_to_heater_gain_ratio = parameters["z_ratio"]
    # an explicit waveguide self-heating share overrides the inferred share.
    if spec.selfheat_share_frac is not None:
        parameters["f_sh"] = _helper_broadcast_interim_ring_parameter(n, spec.selfheat_share_frac, "selfheat_share_frac")
    else:
        # only the excess over the heater gain contributes to the inferred core share.
        parameters["f_sh"] = (optical_to_heater_gain_ratio - 1.0) / optical_to_heater_gain_ratio

    # positive heater resistance gives a positive temperature rise per applied watt.
    if np.any(parameters["z0"] <= 0):
        raise ValueError("z_heater_ring_k_per_w must be > 0")
    # reciprocity sets a lower bound on the optical-to-heater gain ratio.
    if np.any(optical_to_heater_gain_ratio < 1.0):
        raise ValueError("z_ratio_opt_heater must be >= 1 (Z_opt/Z_heater >= 1 by reciprocity)")
    # the heater self-response must cover the rise observed at the ring.
    if np.any(parameters["heater_self_ratio"] < 1.0):
        raise ValueError("heater_self_ratio must be >= 1 (R_th,H >= Z_heater->ring)")
    # the local self-heating share is a fraction of the total response.
    if np.any((parameters["f_sh"] < 0) | (parameters["f_sh"] > 1)):
        raise ValueError("selfheat_share_frac must be in [0, 1]")
    # a negative radius gives an invalid edge-to-edge gap.
    if np.any(parameters["ring_radius"] < 0):
        raise ValueError("ring_radius_um must be >= 0")
    # the site coupling floor represents a nonnegative share of heat.
    if np.any(parameters["site_floor"] < 0):
        raise ValueError("site_floor_frac must be >= 0")
    fast_share = parameters["share_fast"]
    tail_share = parameters["tail_share"]
    # the fast and tail shares must leave a nonnegative middle share.
    if np.any(fast_share < 0) or np.any(tail_share < 0) or np.any(fast_share + tail_share > 1):
        raise ValueError("share_fast_frac and tail_share_frac must be >= 0 with a sum <= 1")
    positive_time_keys = (
        "tau_fast", "tau_slow", "tail_tau", "selfheat_tau", "heater_node_tau", "xt_tau_ref", "xt_tau_min",
    )
    # check every time scale used by the interim heat paths.
    for key in positive_time_keys:
        # exponential lags decay only for positive time constants.
        if np.any(parameters[key] <= 0):
            raise ValueError(f"interim time constant {key} must be > 0")
    # the crosstalk response needs a supported choice of dynamics.
    if spec.crosstalk_dynamics not in ("imec", "self"):
        raise ValueError("crosstalk_dynamics must be 'imec' or 'self'")
    return parameters


# shared physical parameters for the three interim heat-path cases.
@dataclass
class _InterimHeatPaths:
    spec: InterimRom  # distance laws and timing assumptions for the interim thermal model
    elements: _Elements  # positions, kinds and host rings of heat sources and readouts
    parameter_arrays: dict[str, Float64Array]  # per-world and per-ring fitted parameters
    lag_builder: _Lags  # destination for the gain and time constant of each constructed lag
    heater_ring_dc_gain_k_per_w: Float64Array  # settled rise at each ring per heater watt, indexed by world and ring
    mean_heater_ring_dc_gain_k_per_w: Float64Array  # mean settled heater gain across rings in each world (K/W)
    optical_to_heater_gain_ratio: Float64Array  # ratio of optical-to-ring gain to heater-to-ring gain (1)
    heater_self_gain_ratio: Float64Array  # ratio of heater self-heating gain to heater-to-ring gain (1)
    optical_core_share: Float64Array  # fraction of the optical response carried by the fast core lag (1)
    ring_radii_um: Float64Array  # radii used to convert centre distances to edge gaps (um)
    neighbor_ranks: Int64Array  # distance group of each ring pair, zero for the same ring (1)
    ring_distances_um: Float64Array  # centre-to-centre distance for each ring pair (um)
    heater_response_lags: tuple[tuple[Float64Array, Float64Array, int, bool], ...]  # per component - time (s), share (1), layer code and package-tail flag

    # estimate how long heat takes to reach the other element using the gap between their edges.
    # axes - B = simulated worlds.
    def physics_path_time_constant_s(
        self: _InterimHeatPaths,  # self = paths (mixed units) - interim path parameters - type _InterimHeatPaths
        centre_distance_um: ArrayLike,  # centre_distance_um = d_centres (um) - separation between element centres - type array-like, scalar or broadcastable over worlds
        source_radius_um: ArrayLike,  # source_radius_um = R_source (um) - source radius or zero for a point site - type array-like, scalar or broadcastable over worlds
        observer_radius_um: ArrayLike,  # observer_radius_um = R_observer (um) - observer radius or zero for a point site - type array-like, scalar or broadcastable over worlds
    ) -> (
        Float64Array  # return = tau_cross (s) - one crosstalk time constant per world - shape (B,) - type float64 ndarray
    ):
        # the IMEC crosstalk law uses the gap between elements. clamp overlapping elements to zero gap.
        gap_um = np.maximum(centre_distance_um - source_radius_um - observer_radius_um, 0.0)
        # diffusion time grows with gap squared. the selected distance law sets the short- and long-range limits.
        return physics_crosstalk_time_constant_s(
            gap_um,
            self.parameter_arrays["xt_tau_ref"][:, 0],
            self.spec.xt_dist_ref_um,
            self.parameter_arrays["xt_tau_min"][:, 0],
            None if self.spec.xt_tau_far_s is None else self.parameter_arrays["xt_tau_far"][:, 0],
            self.spec.xt_far_um,
            self.spec.xt_tau_law,
        )

    # distribute a path gain Z_path across the reference ring's three heater lags.
    # axes - B = simulated worlds.
    def assembly_append_heater_response_shape(
        self: _InterimHeatPaths,  # self = paths (mixed units) - interim path parameters - type _InterimHeatPaths
        observer_index: int,  # observer_index = o (1) - observer index - type int scalar
        source_index: int,  # source_index = s (1) - source index - type int scalar
        thermal_gain_k_per_w: ArrayLike,  # thermal_gain_k_per_w = Z_0 (K/W) - dc gain per world - shape (B,) - type array-like
        ring_index: int,  # ring_index = i (1) - ring whose time constants and shares set the response - type int scalar
        pair_type: int,  # pair_type = pair (1) - heat path type code - type int scalar
    ) -> (
        None  # return = None - append the lags to lag_builder - type None
    ):
        for time_constants_s, gain_share, layer, is_package_tail in self.heater_response_lags:
            # r_m = Z_path*f_m so sum(r_m) gives the requested path gain.
            self.lag_builder.assembly_append_heat_path_lag(
                observer_index,
                source_index,
                time_constants_s[:, ring_index],
                thermal_gain_k_per_w * gain_share[:, ring_index],
                layer,
                pair_type,
                tailable=True,
                tail=is_package_tail,
            )

    # build the local temperature rise from direct core heating and the slower heater response.
    def assembly_append_same_ring_lags(
        self: _InterimHeatPaths,  # self = paths (mixed units) - interim path parameters - type _InterimHeatPaths
        observer_index: int,  # observer_index = o (1) - observer index - type int scalar
        source_index: int,  # source_index = s (1) - source index - type int scalar
        source_kind: int,  # source_kind = source_kind (1) - heat source code - type int scalar
        observer_kind: int,  # observer_kind = observer_kind (1) - temperature observer code - type int scalar
        ring_index: int,  # ring_index = i (1) - the ring that carries both elements - type int scalar
    ) -> (
        None  # return = None - append the lags to lag_builder - type None
    ):
        # Z_HR (K/W) is the settled heater-to-ring rise per watt on this ring.
        heater_gain_k_per_w = self.heater_ring_dc_gain_k_per_w[:, ring_index]

        if source_kind in (SRC_WG, SRC_JUNCTION) and observer_kind == OBS_RING:
            # waveguide-core heating gets an extra fast lag. the rest follows the heater response.
            core_share = self.optical_core_share[:, ring_index]
            gain_ratio = self.optical_to_heater_gain_ratio[:, ring_index]
            # optical or junction power has total gain z*Z_HR. assign its fraction f_core to the core lag.
            core_gain_k_per_w = gain_ratio * core_share * heater_gain_k_per_w
            self.lag_builder.assembly_append_heat_path_lag(
                observer_index,
                source_index,
                self.parameter_arrays["selfheat_tau"][:, ring_index],
                core_gain_k_per_w,
                LAYER_FAST,
                PAIR_SELF,
            )
            # distribute the remaining gain z*Z_HR*(1-f_core) over the three heater time scales.
            remaining_gain_k_per_w = gain_ratio * (1.0 - core_share) * heater_gain_k_per_w
            self.assembly_append_heater_response_shape(
                observer_index, source_index, remaining_gain_k_per_w, ring_index, PAIR_SELF
            )
            # the selected path already has its complete same-ring response.
            return

        # use Z_HR for heater-to-ring and for the reciprocal ring-to-heater path.
        self.assembly_append_heater_response_shape(
            observer_index, source_index, heater_gain_k_per_w, ring_index, PAIR_SELF
        )
        if source_kind == SRC_HEATER and observer_kind == OBS_HEATER:
            # add (h-1)*Z_HR to the heater's own path so the total is h*Z_HR.
            extra_gain_k_per_w = (self.heater_self_gain_ratio[:, ring_index] - 1.0) * heater_gain_k_per_w
            self.lag_builder.assembly_append_heat_path_lag(
                observer_index,
                source_index,
                self.parameter_arrays["heater_node_tau"][:, ring_index],
                extra_gain_k_per_w,
                LAYER_FAST,
                PAIR_SELF,
            )

    # turn neighbour rank and separation into the gain and rise time of a ring-to-ring heat path.
    def assembly_append_cross_ring_lags(
        self: _InterimHeatPaths,  # self = paths (mixed units) - interim path parameters - type _InterimHeatPaths
        observer_index: int,  # observer_index = o (1) - observer index - type int scalar
        source_index: int,  # source_index = s (1) - source index - type int scalar
        source_kind: int,  # source_kind = source_kind (1) - heat source code - type int scalar
        observer_kind: int,  # observer_kind = observer_kind (1) - temperature observer code - type int scalar
        source_ring: int,  # source_ring = j (1) - ring that carries the source - type int scalar
        observer_ring: int,  # observer_ring = i (1) - ring that carries the observer - type int scalar
    ) -> (
        None  # return = None - append the lags to lag_builder - type None
    ):
        # choose the coupling fraction from the first, second, or farther neighbour band.
        neighbor_rank = self.neighbor_ranks[observer_ring, source_ring]
        # use the fitted nearest-neighbour gain for the first distance group.
        if neighbor_rank == 1:
            coupling_fraction = self.parameter_arrays["xt_nearest"][:, 0]
        # use the second-neighbour gain when the model supplies one.
        elif neighbor_rank == 2 and self.spec.xt_next_frac is not None:
            coupling_fraction = self.parameter_arrays["xt_next"][:, 0]
        else:
            # more distant rings receive the model's background crosstalk fraction.
            coupling_fraction = self.parameter_arrays["xt_floor"][:, 0]

        source_heater_gain = self.heater_ring_dc_gain_k_per_w[:, source_ring]
        observer_heater_gain = self.heater_ring_dc_gain_k_per_w[:, observer_ring]
        # heater-to-ring and its reciprocal use the heater's ring for the reference gain.
        if source_kind == SRC_HEATER and observer_kind == OBS_RING:
            reference_gain_k_per_w = source_heater_gain
        # reverse the optical-to-heater path using its reciprocal heat coupling.
        elif source_kind in (SRC_WG, SRC_JUNCTION) and observer_kind == OBS_HEATER:
            reference_gain_k_per_w = observer_heater_gain
        else:
            # use sqrt(Z_source*Z_observer) so reversing the rings gives the same reference.
            reference_gain_k_per_w = np.sqrt(observer_heater_gain * source_heater_gain)

        # Z_cross = c_neighbor*Z_ref. a coupling fraction of 0.03 gives 3 percent of the reference gain.
        cross_gain_k_per_w = coupling_fraction * reference_gain_k_per_w
        # split Z_cross over the source ring's heater lags or assign it to one gap-based lag.
        if self.spec.crosstalk_dynamics == "self":
            self.assembly_append_heater_response_shape(
                observer_index, source_index, cross_gain_k_per_w, source_ring, PAIR_CROSS
            )
            # the heater-shaped lags already carry the full cross-ring gain. a gap-based lag would count it twice.
            return
        cross_time_constant_s = self.physics_path_time_constant_s(
            self.ring_distances_um[observer_ring, source_ring],
            self.ring_radii_um[:, source_ring],
            self.ring_radii_um[:, observer_ring],
        )
        self.lag_builder.assembly_append_heat_path_lag(
            observer_index, source_index, cross_time_constant_s, cross_gain_k_per_w,
            LAYER_SLOW, PAIR_CROSS,
        )

    # estimate the heat reaching a monitor, driver, sensor or aggressor from its distance to the other endpoint.
    def assembly_append_site_lags(
        self: _InterimHeatPaths,  # self = paths (mixed units) - interim path parameters - type _InterimHeatPaths
        observer_index: int,  # observer_index = o (1) - observer index - type int scalar
        source_index: int,  # source_index = s (1) - source index - type int scalar
        source_ring: int,  # source_ring = j (1) - ring that carries the source, or -1 for a free site - type int scalar
        observer_ring: int,  # observer_ring = i (1) - ring that carries the observer, or -1 for a free site - type int scalar
        source_is_ring_element: bool,  # source_is_ring_element = extended_source (1) - use the source ring radius when computing its edge - type bool scalar
        observer_is_ring_element: bool,  # observer_is_ring_element = extended_observer (1) - use the observer ring radius when computing its edge - type bool scalar
        pair_type: int,  # pair_type = pair (1) - heat path type code - type int scalar
    ) -> (
        None  # return = None - append one lag to lag_builder - type None
    ):
        # choose the source ring as the gain reference when present, then the observer ring, then the bank mean.
        if source_ring >= 0:
            reference_gain_k_per_w = self.heater_ring_dc_gain_k_per_w[:, source_ring]
        # use the observer's ring as a reference when the source has no ring.
        elif observer_ring >= 0:
            reference_gain_k_per_w = self.heater_ring_dc_gain_k_per_w[:, observer_ring]
        else:
            # use the bank-average heater gain when neither endpoint belongs to a ring.
            reference_gain_k_per_w = self.mean_heater_ring_dc_gain_k_per_w

        # use centre distance for site gain and edge separation for the lag time constant.
        source_xy_um = self.elements.src_xy[source_index]
        observer_xy_um = self.elements.obs_xy[observer_index]
        site_distance_um = float(np.linalg.norm(source_xy_um - observer_xy_um))
        # a ring element has the ring's radius. a point site has zero radius.
        source_radius_um = 0.0
        # measure the gap from the source ring edge for extended ring elements.
        if source_ring >= 0 and source_is_ring_element:
            source_radius_um = self.ring_radii_um[:, source_ring]
        observer_radius_um = 0.0
        # account for the observer radius at the other end of the gap.
        if observer_ring >= 0 and observer_is_ring_element:
            observer_radius_um = self.ring_radii_um[:, observer_ring]
        site_time_constant_s = self.physics_path_time_constant_s(
            site_distance_um, source_radius_um, observer_radius_um
        )
        # c_site falls as 1/d_centres between its bounds. the site lag gain is c_site*Z_ref.
        site_coupling_fraction = physics_site_coupling_fraction(
            site_distance_um,
            self.parameter_arrays["xt_nearest"][:, 0],
            self.parameter_arrays["site_floor"][:, 0],
            self.spec.site_ref_um,
            self.spec.site_max_frac,
        )
        self.lag_builder.assembly_append_heat_path_lag(
            observer_index,
            source_index,
            site_time_constant_s,
            site_coupling_fraction * reference_gain_k_per_w,
            LAYER_SLOW,
            pair_type,
        )



# in the full picture - assemble calls this once, before any simulation step, when no finite-element model is fitted.
# it decides the whole interim thermal model - for every source and observer pair, how far the temperature rises and how fast.
# a source is a place where power (W) enters. an observer is a place where temperature (K) is read.
# the result goes into lag_builder, a list of lags. each lag_builder.add(observer, source, tau, r, layer, pair) appends one lag.
# the layer and pair arguments are labels for the later uncertainty scaling. they do not change r or tau here.
# parameter_arrays holds the InterimRom values under short keys. each array has shape (B, N).
# "z0" = Z0 (K/W) - settled heater-to-ring gain. "z_ratio" = z (1) - optical-to-ring gain over Z0.
# "heater_self_ratio" = h (1) - heater-to-heater gain over Z0. "f_sh" = f_sh (1) - share of the optical gain on the waveguide-core lag.
# "share_fast" = f_F (1) and "tail_share" = f_T (1) - shares of a path's settled gain on the fast lag and on the package lag.
# "tau_fast", "tau_slow", "tail_tau", "selfheat_tau", "heater_node_tau" (s) - time constants tau_F, tau_S, tau_T, tau_sh, tau_hn.
# "xt_nearest", "xt_next", "xt_floor", "site_floor" (1) - coupling fractions. "ring_radius" (um) - ring radius.
# how much does one source warm an observer on each time scale?
# each path is a sum of lags. a power step P (W) gives each lag a rise P*r*(1-exp(-t/tau)) in K.
# r (K/W) sets the eventual rise per watt. tau (s) sets how fast that rise develops.
# choose the gains and times for a same-ring path, a cross-ring path, or a path involving a separate site.
# each parameter array has one row per world and one column per ring.
# Z0 is the heater-to-ring settled gain. the three heater lags are tau_F, tau_S, and tau_T with shares f_F, 1-f_F-f_T, and f_T.
# pair of elements                          | settled gain    | lags
# same ring, heater -> ring                 | Z0              | the three heater lags
# same ring, light or junction -> heater    | Z0              | the three heater lags
# same ring, heater -> heater               | h*Z0            | the three heater lags, plus (h-1)*Z0 at tau_hn
# same ring, light or junction -> ring      | z*Z0            | f_sh*z*Z0 at tau_sh, the rest on the three heater lags
# two rings                                 | c_rank*Z_ref    | one lag from the gap law, or the three heater lags when crosstalk_dynamics='self'
# any pair with a monitor, driver, sensor, or aggressor | c_site(d)*Z_ref | one lag from the gap law
# give each source-observer pair a settled rise per watt and the time scales on which that rise develops.
# axes - B = simulated worlds, N = physical rings.
def assembly_append_interim_heat_path_lags(
    spec: InterimRom,  # spec = model (mixed units) - interim thermal resistance and time-constant parameters - type InterimRom record
    elements: _Elements,  # elements = elements (mixed units) - source and observer instances with ring attachments and coordinates - type thermal element builder
    layout: LayoutSites,  # layout = sites (um) - source and observer site coordinates - type LayoutSites record
    ring_count: int,  # ring_count = N (1) - number of rings - type int scalar
    world_count: int,  # world_count = B (1) - number of worlds - type int scalar
    parameter_arrays: dict[str, Float64Array],  # parameter_arrays = parameters (field units) - broadcast interim parameters in their named physical units - type dict of float64 ndarrays, shape (B, N)
    lag_builder: _Lags,  # lag_builder = lags (mixed units) - lag indices, time constants, and thermal residues - type thermal lag builder
) -> (
    None  # return = None - append the interim heat paths to the lag builder - type None
):
    # the heater-to-ring dc gain sets the scale for all interim paths.
    heater_ring_dc_gain_k_per_w = parameter_arrays["z0"]
    mean_heater_ring_dc_gain_k_per_w = heater_ring_dc_gain_k_per_w.mean(axis=1)
    # z is the ratio of optical-to-ring gain to heater-to-ring gain.
    optical_to_heater_gain_ratio = parameter_arrays["z_ratio"]
    # h is the ratio of heater-to-heater gain to heater-to-ring gain.
    heater_self_gain_ratio = parameter_arrays["heater_self_ratio"]
    # f_core is the share of optical-to-ring gain assigned to the waveguide core lag.
    optical_core_share = parameter_arrays["f_sh"]
    ring_radii_um = parameter_arrays["ring_radius"]
    neighbor_ranks, ring_distances_um = physics_neighbor_ranks_and_distances(layout.ring_xy_um)

    # (time constants in s, share of dc gain, layer, package-tail flag) in append order.
    # both ring and oxide lags belong to the fast layer. the die and package lag belongs to the slow layer.
    fast_share = parameter_arrays["share_fast"]
    tail_share = parameter_arrays["tail_share"]
    # f_slow = 1 - f_fast - f_tail so these three fractions add to one.
    # with f_fast = 0.4 and f_tail = 0.05, the middle lag receives 0.55 of the path gain.
    heater_response_lags = (
        (parameter_arrays["tau_fast"], fast_share, LAYER_FAST, False),
        (parameter_arrays["tau_slow"], 1.0 - fast_share - tail_share, LAYER_FAST, False),
        (parameter_arrays["tail_tau"], tail_share, LAYER_SLOW, True),
    )

    ring_source_kinds = (SRC_HEATER, SRC_WG, SRC_JUNCTION)
    ring_observer_kinds = (OBS_RING, OBS_HEATER)

    paths = _InterimHeatPaths(
        spec=spec,
        elements=elements,
        parameter_arrays=parameter_arrays,
        lag_builder=lag_builder,
        heater_ring_dc_gain_k_per_w=heater_ring_dc_gain_k_per_w,
        mean_heater_ring_dc_gain_k_per_w=mean_heater_ring_dc_gain_k_per_w,
        optical_to_heater_gain_ratio=optical_to_heater_gain_ratio,
        heater_self_gain_ratio=heater_self_gain_ratio,
        optical_core_share=optical_core_share,
        ring_radii_um=ring_radii_um,
        neighbor_ranks=neighbor_ranks,
        ring_distances_um=ring_distances_um,
        heater_response_lags=heater_response_lags,
    )

    # visit every source and observer pair once and send it to its case.
    for source_index, source_kind in enumerate(elements.src_kind):
        source_ring = elements.src_ring[source_index]
        source_is_ring_element = source_kind in ring_source_kinds

        # pair this heat source with each enabled temperature observer.
        for observer_index, observer_kind in enumerate(elements.obs_kind):
            observer_ring = elements.obs_ring[observer_index]
            observer_is_ring_element = observer_kind in ring_observer_kinds

            # ring attachment -1 marks a free site.
            if source_ring < 0 or observer_ring < 0:
                pair_type = PAIR_SITE
            # elements attached to the same ring can share a local heat path.
            elif source_ring == observer_ring:
                pair_type = PAIR_SELF
            else:
                # distinct host rings require a cross-ring path.
                pair_type = PAIR_CROSS

            both_are_ring_elements = source_is_ring_element and observer_is_ring_element
            # build local ring lags from the fitted same-ring gain and time scales.
            if pair_type == PAIR_SELF and both_are_ring_elements:
                paths.assembly_append_same_ring_lags(observer_index, source_index, source_kind, observer_kind, source_ring)
            # build cross-ring lags from separation and neighbour rank.
            elif pair_type == PAIR_CROSS and both_are_ring_elements:
                paths.assembly_append_cross_ring_lags(
                    observer_index, source_index, source_kind, observer_kind, source_ring, observer_ring
                )
            else:
                # a site outside the ring-element model receives a distance-dependent site response.
                paths.assembly_append_site_lags(
                    observer_index,
                    source_index,
                    source_ring,
                    observer_ring,
                    source_is_ring_element,
                    observer_is_ring_element,
                    pair_type,
                )
# decide whether the fitted local heat path applies at the installed site and record any displaced site.
def _validation_matches_extracted_site_offset(
    rom: ThermalRom,  # rom = model (mixed units) - extracted thermal gains, positions and time constants - type ThermalRom
    layout: LayoutSites,  # layout = layout (um) - physical ring and site positions - type LayoutSites
    tol_um: float,  # tol_um = epsilon (um) - largest allowed displacement from the fitted site - type float scalar
    moved_site_names: set[str],  # moved_site_names = displaced (1) - site names collected for the assembly report - type set[str]
    site_name: str | None,  # site_name = site (1) - optional extracted site name - type str | None
    site_xy_um: ArrayLike,  # site_xy_um = r_site (um) - site coordinates - shape (2,) - type array-like
    ring_index: int,  # ring_index = i (1) - ring that carries the site - type int scalar
) -> (
    bool  # return = matches (1) - whether the same-ring table can use this site - type bool scalar
):
    if site_name is None or site_name not in rom.site_offsets_um:
        # keep the same-ring table when no extracted offset is available.
        return True
    ring_centre_um = layout.ring_xy_um[ring_index]
    extracted_offset_um = np.asarray(rom.site_offsets_um[site_name], float)
    # compare the site's position relative to its ring with the position used in the extraction.
    actual_offset_um = site_xy_um - ring_centre_um
    offset_error_um = np.linalg.norm(actual_offset_um - extracted_offset_um)
    # the extracted same-ring response applies only near the fitted site offset.
    if offset_error_um > tol_um:
        moved_site_names.add(site_name)
        # record the displaced site so assembly can choose another path and report it.
        return False
    return True


# estimate each thermal lag gain at the installed separation by interpolating between measured ring spacings.
# axes - K = fitted time constants, D = distance samples.
def _physics_interpolate_cross_ring_thermal_gains(
    rom: ThermalRom,  # rom = model (mixed units) - extracted thermal gains, positions and time constants - type ThermalRom
    distance_grid_um: Float64Array,  # distance_grid_um = d (um) - sampled distances - shape (D,) - type float64 ndarray
    time_constants_s: Float64Array,  # time_constants_s = tau (s) - shared lag times - shape (K,) - type float64 ndarray
    distances_outside_grid_um: list[float],  # distances_outside_grid_um = d_out (um) - requested separations outside the fitted distance grid - type list[float]
    rom_source_index: int,  # rom_source_index = s (1) - ROM source index - type int scalar
    rom_observer_index: int,  # rom_observer_index = o (1) - ROM observer index - type int scalar
    distance_um: float,  # distance_um = d (um) - distance between the actual element centres - type float scalar
) -> (
    Float64Array  # return = r (K/W) - interpolated thermal residues - shape (K,) - type float64 ndarray
):
    # an empty distance table supplies no measured cross-ring gain.
    if distance_grid_um.size == 0:
        return np.zeros_like(time_constants_s)
    # record extrapolation requests because interpolation clamps to the table endpoints.
    if distance_um < distance_grid_um[0] or distance_um > distance_grid_um[-1]:
        distances_outside_grid_um.append(distance_um)

    # each row is a distance and each column is a gain at one fixed time constant.
    # for d between grid points d0 and d1, alpha = (d-d0)/(d1-d0) and r = (1-alpha)*r0 + alpha*r1.
    # at 50 um, gains of 400 and 200 K/W at 40 and 60 um give 300 K/W for that lag.
    # np.interp holds the nearest endpoint value beyond the measured grid.
    path_residues_k_per_w = rom.cross_res[rom_source_index, rom_observer_index]
    return np.array([
        np.interp(distance_um, distance_grid_um, path_residues_k_per_w[:, lag_index])
        for lag_index in range(time_constants_s.size)
    ])



# in the full picture - assemble calls this once, before any simulation step, when a fitted ThermalRom is available.
# it does the same job as assembly_append_interim_heat_path_lags, but it reads the gains from tables instead of from rules.
# a ThermalRom is a compact fit to a finite-element heat solve. it has K shared time constants rom.taus_s (s).
# for a source and observer on the same ring, rom.self_res[source kind, observer kind] gives K gains (K/W), one per time constant.
# for any other pair, rom.cross_res[source kind, observer kind] gives the same K gains at each distance of rom.cross_r_um (um).
# a gain can be negative. the temperature is the sum of all K terms, so only the sum must make physical sense.
# the result goes into lag_builder, a list of lags. each nonzero gain becomes one lag with its time constant.
# append extracted lags from the same-ring table or interpolate from the distance table.
# the same-ring table applies when both sites match their extracted offsets.
# turn measured temperature responses into simulation lags for the actual separation of each source and observer.
def assembly_append_extracted_heat_path_lags(
    rom: ThermalRom,  # rom = model (mixed units) - extracted poles, thermal residues, and site metadata - type ThermalRom record
    elements: _Elements,  # elements = elements (mixed units) - source and observer instances with ring attachments and coordinates - type thermal element builder
    layout: LayoutSites,  # layout = sites (um) - source and observer site coordinates - type LayoutSites record
    world_count: int,  # world_count = B (1) - number of worlds - type int scalar
    tau_split: float,  # tau_split = tau_split (s) - boundary between fast and slow time scales - type float scalar
    lag_builder: _Lags,  # lag_builder = lags (mixed units) - lag indices, time constants, and thermal residues - type thermal lag builder
    meta: dict[str, Any],  # meta = metadata (mixed units) - provenance and validation fields - type dict with string keys and metadata values
    tol_um: float = 1.0,  # tol_um = d_tol (um) - allowed site-offset deviation from the ROM self table - type float scalar
) -> (
    None  # return = None - append fitted heat paths to the lag builder - type None
):   # assumed - 1 um, well below the site sizes and offsets (tens of um)
    time_constants_s = np.asarray(rom.taus_s, float)
    layer_codes = np.where(time_constants_s < tau_split, LAYER_FAST, LAYER_SLOW)
    distance_grid_um = np.asarray(rom.cross_r_um, float)
    source_offset_names = {SRC_DRIVER: "driver"}
    observer_offset_names = {OBS_PD: "monitor_pd", OBS_TEMP: "temp_sensor"}

    # translate the bank's kind codes to the ROM's source and observer axes.
    rom_source_indices = {
        SOURCE_KINDS.index(kind_name): index for index, kind_name in enumerate(rom.sources)
    }
    rom_observer_indices = {
        OBSERVER_KINDS.index(kind_name): index for index, kind_name in enumerate(rom.observers)
    }
    skipped_kind_names = set()
    moved_site_names = set()
    distances_outside_grid_um = []



    # select the extracted heat source row for each assembled source.
    for source_index, source_kind in enumerate(elements.src_kind):
        # a source missing from the extraction cannot supply a fitted path.
        if source_kind not in rom_source_indices:
            skipped_kind_names.add(SOURCE_KINDS[source_kind])
            # retain the missing kind for the assembly report and skip its paths.
            continue
        rom_source_index = rom_source_indices[source_kind]
        source_ring = elements.src_ring[source_index]
        source_xy_um = elements.src_xy[source_index]

        # match each temperature observer with its extracted gain column.
        for observer_index, observer_kind in enumerate(elements.obs_kind):
            # a missing observer kind has no fitted response to this source.
            if observer_kind not in rom_observer_indices:
                skipped_kind_names.add(OBSERVER_KINDS[observer_kind])
                # retain the missing kind and continue with other observers.
                continue
            rom_observer_index = rom_observer_indices[observer_kind]
            observer_ring = elements.obs_ring[observer_index]
            observer_xy_um = elements.obs_xy[observer_index]

            both_on_rings = source_ring >= 0 and observer_ring >= 0
            same_ring = both_on_rings and source_ring == observer_ring
            # choose the local table only when both sites match its fitted geometry.
            use_same_ring_table = (
                same_ring
                and _validation_matches_extracted_site_offset(
                    rom, layout, tol_um, moved_site_names, source_offset_names.get(source_kind), source_xy_um, source_ring
                )
                and _validation_matches_extracted_site_offset(
                    rom, layout, tol_um, moved_site_names, observer_offset_names.get(observer_kind), observer_xy_um, observer_ring
                )
            )
            # use the fitted local residues directly for a matching same-ring path.
            if use_same_ring_table:
                residues_k_per_w = np.asarray(rom.self_res[rom_source_index, rom_observer_index], float)
                pair_type = PAIR_SELF
            else:
                # look up the actual source-to-observer separation when the same-ring table does not apply.
                # a monitor at 50 um with a saved offset of 20 um has a 30 um mismatch.
                # for a heater at the ring centre, the lookup distance in that case is 50 um.
                site_distance_um = float(np.linalg.norm(source_xy_um - observer_xy_um))
                residues_k_per_w = _physics_interpolate_cross_ring_thermal_gains(
                    rom, distance_grid_um, time_constants_s, distances_outside_grid_um,
                    rom_source_index, rom_observer_index, site_distance_um
                )
                # use a cross-ring lag for two distinct host rings and a site lag for the remaining geometries.
                pair_type = PAIR_CROSS if both_on_rings and not same_ring else PAIR_SITE

            # each nonzero residue adds one lag. only same-ring table paths can replace their package tail.
            for lag_index, residue_k_per_w in enumerate(residues_k_per_w):
                # a zero residue adds no temperature rise at any time.
                if residue_k_per_w != 0.0:
                    lag_builder.assembly_append_heat_path_lag(
                        observer_index,
                        source_index,
                        time_constants_s[lag_index],
                        residue_k_per_w,
                        int(layer_codes[lag_index]),
                        pair_type,
                        tailable=(pair_type == PAIR_SELF),
                    )

    # report thermal element kinds absent from the extracted model.
    if skipped_kind_names:
        meta["rom_skipped_kinds"] = sorted(skipped_kind_names)
    # report offsets that prevent use of the fitted local response.
    if moved_site_names:
        moved_sites = sorted(moved_site_names)
        meta["rom_sites_off_default"] = moved_sites
        extracted_offsets_um = {name: rom.site_offsets_um[name] for name in moved_sites}
        warnings.warn(
            f"layout {moved_sites} sites are more than {tol_um} um from the ROM self-table offsets "
            f"{extracted_offsets_um} um: their same-ring pairs use the distance table cross_res "
            "at the actual site distance",
            RuntimeWarning,
            stacklevel=4,
        )
    # report distances for which the nearest tabulated response was used.
    if distances_outside_grid_um:
        example_distances_um = sorted({round(distance, 2) for distance in distances_outside_grid_um})[:6]
        warnings.warn(
            f"thermal ROM cross_r_um grid {distance_grid_um[0]}..{distance_grid_um[-1]} um does not cover distances "
            f"{example_distances_um} um; clamped to the grid ends (radially symmetric table, valid for d > 2R)",
            RuntimeWarning,
            stacklevel=4,
        )


# in the full picture - assemble calls this once when a simulated chip supplies its own package share.
# the package lag is the slowest part of a warmup. it represents the whole die and package warming over milliseconds.
# this function changes when the warmth arrives. it does not change how much arrives in the end.
# lag_builder is the list of lags built so far. each lag has a gain r (K/W) and a time constant tau (s).
# a path is every lag with the same source and the same observer. its settled gain Z (K/W) is the sum of its gains.
# only lags marked tail_replaceable take part. the lag marked as the tail is the package lag.
# how much of the settled temperature rise should develop on the package time scale?
# a constant power P (W) gives one lag an eventual rise P*r (K), where r is its gain (K/W).
# for each path, split the eligible gain Z between the package lag and the other eligible lags.
# axes - B = simulated worlds.
def physics_replace_package_tail_preserving_dc_gain(
    lag_builder: _Lags,  # lag_builder = lags (mixed units) - lag indices, time constants, and thermal residues - type thermal lag builder
    tail_tau: ArrayLike,  # tail_tau = tau_tail (s) - absolute package-tail time constants - type array-like, shape (B,)
    tail_share: ArrayLike,  # tail_share = f (1) - fraction of the eligible steady rise assigned to the package lag - shape (B,) - type array-like
) -> (
    None  # return = None - update the eligible lags in lag_builder - type None
):
    eligible_lags_by_path: dict[tuple[int, int], list[int]] = {}
    # group eligible lags by their source and observer before redistributing settled gain.
    for lag_index, observer_index in enumerate(lag_builder.observer_indices):
        # only adjustable lags can fund the replacement package tail.
        if lag_builder.tail_replaceable[lag_index]:
            heat_path = (observer_index, lag_builder.source_indices[lag_index])
            eligible_lags_by_path.setdefault(heat_path, []).append(lag_index)

    # preserve total settled gain separately for each heat path.
    for (observer_index, source_index), eligible_lags in eligible_lags_by_path.items():
        pair_type = lag_builder.pair_codes[eligible_lags[0]]
        other_lags = [index for index in eligible_lags if not lag_builder.tail_flags[index]]
        package_lags = [index for index in eligible_lags if lag_builder.tail_flags[index]]

        # save the original eligible gain Z before editing any lag.
        total_gain_k_per_w = np.sum([lag_builder.residues_k_per_w[index] for index in eligible_lags], axis=0)
        # sum the nonpackage gains that can fund the new tail. an empty group contributes zero.
        old_other_gain_k_per_w = (
            np.sum([lag_builder.residues_k_per_w[index] for index in other_lags], axis=0)
            if other_lags else np.zeros(lag_builder.world_count)
        )

        # assign f*Z to the package and leave (1-f)*Z for the other lags.
        # for Z = 1000 K/W and f = 0.2, that gives 200 K/W for the package and 800 K/W for the rest.
        package_gain_k_per_w = tail_share * total_gain_k_per_w
        target_other_gain_k_per_w = total_gain_k_per_w * (1.0 - tail_share)

        # target / old sum gives one multiplier for all other lags, preserving their relative gains.
        # if their old sum is 900 K/W, multiply each by 800/900.
        # a zero old sum cannot be rescaled to a nonzero target. the model uses zero for those gains.
        with np.errstate(divide="ignore", invalid="ignore"):
            # rescale the remaining terms so their settled sum plus the package gain is conserved.
            other_gain_scale = np.where(
                old_other_gain_k_per_w != 0.0, target_other_gain_k_per_w / old_other_gain_k_per_w, 0.0
            )
        # scale the other eligible gains to leave the requested fraction in the tail.
        for lag_index in other_lags:
            # apply the same factor to each eligible nonpackage residue within that path.
            lag_builder.residues_k_per_w[lag_index] = lag_builder.residues_k_per_w[lag_index] * other_gain_scale

        # store the requested package gain in one lag and keep its time constant fixed during later scaling.
        if package_lags:
            package_lag = package_lags[0]
            lag_builder.residues_k_per_w[package_lag] = package_gain_k_per_w
            lag_builder.time_constants_s[package_lag] = np.broadcast_to(tail_tau, (lag_builder.world_count,)).copy()
            lag_builder.fixed_time_constants[package_lag] = True
            # one package lag carries the new tail. remove the redundant contributions.
            for lag_index in package_lags[1:]:
                lag_builder.residues_k_per_w[lag_index] = np.zeros(lag_builder.world_count)
        else:
            # create a package term when there is no eligible tail to reuse.
            lag_builder.assembly_append_heat_path_lag(
                observer_index, source_index, tail_tau, package_gain_k_per_w, LAYER_SLOW, pair_type,
                tailable=True, tail=True, fixed_tau=True,
            )


# in the full picture - each simulated chip (world) can have stronger or weaker heating than the nominal model.
# s_DC (1) is the multiplier on the settled gain r (K/W) of a heat path. a value of 1.2 means 20 percent more temperature rise per watt.
# the user gives one number for every path, or rules keyed by source kind and observer kind such as "heater->ring_mode_weighted".
# this function returns the s_DC of one kind of path for every world. assemble multiplies each lag's gain by it.
# multiply all DC-scale rules matching this source/observer kind pair.
# axes - B = simulated worlds.
def physics_pair_dc_scale(
    dc_scale: ArrayLike | Mapping[str, ArrayLike],  # dc_scale = s_DC (1) - DC residue scales or source-to-observer matching rules - type array-like or mapping to array-like values, scalar or shape (B,) per rule
    source_kind_name: str,  # source_kind_name = source_kind (1) - source kind of the heat path - type str
    observer_kind_name: str,  # observer_kind_name = observer_kind (1) - observer kind of the heat path - type str
    world_count: int,  # world_count = B (1) - number of worlds - type int scalar
) -> (
    Float64Array  # return = s_DC (1) - product of all matching DC-scale rules - type float64 ndarray, shape (B,)
):
    # a scalar or vector scale applies to every source-observer pair.
    if not isinstance(dc_scale, Mapping):
        return np.broadcast_to(helper_normalize_world_parameter_vector(dc_scale, "dc_scale"), (world_count,))
    # start at 1 per world and multiply every applicable rule into that world's gain factor.
    # for example, a global factor of 1.2 and a matching path factor of 0.5 give 0.6.
    factor = np.ones(world_count)
    # match scaling rules against the heat source and temperature observer kinds.
    for key, value in dc_scale.items():
        # the wildcard supplies the fallback scale for every pair.
        if key == "*":
            # multiply the shared wildcard factor into the scale of every matching path.
            factor = factor * helper_normalize_world_parameter_vector(value, "dc_scale[*]")
            # the fallback needs no source or observer pattern matching.
            continue
        # a pair-specific rule must specify both ends of the heat path.
        if "->" not in key:
            raise ValueError(f"dc_scale key {key!r} must be '*' or '<source>-><observer>'")
        source_pattern, observer_pattern = (pattern.strip() for pattern in key.split("->", 1))
        # validate the source and observer patterns against their respective kind tables.
        for pattern, valid_kinds in ((source_pattern, SOURCE_KINDS), (observer_pattern, OBSERVER_KINDS)):
            # a named pattern must identify a supported thermal element.
            if pattern != "*" and pattern not in valid_kinds:
                raise ValueError(f"dc_scale key {key!r}: unknown kind {pattern!r}")
        # apply a rule only to a path matching both of its ends.
        if source_pattern in ("*", source_kind_name) and observer_pattern in ("*", observer_kind_name):
            # combine the pair-specific and wildcard scales multiplicatively.
            factor = factor * helper_normalize_world_parameter_vector(value, f"dc_scale[{key}]")
    return np.broadcast_to(factor, (world_count,))


# in the full picture - each simulated chip (world) draws two time factors, s_fast (1) and s_slow (1).
# they multiply the time constants tau (s) of the lags to model a chip that warms faster or slower than nominal.
# normally a lag takes the factor of its own layer - s_fast for the ring and oxide, s_slow for the die and package.
# that rule breaks for some fitted paths. a path is every lag with the same source and observer.
# a fitted path can hold lags with positive and negative gains r (K/W) whose sum gives a smooth warmup curve.
# if its fast and slow lags were stretched by different factors, they would stop cancelling and the curve would bend the wrong way.
# this function finds those paths and gives all their lags one shared factor k (1), between s_fast and s_slow.
# it returns a mask of the lags that need the shared factor and the factor itself. assemble applies it.
# how does thermal uncertainty change the speed of a fitted warmup curve?
# positive and negative lag contributions can cancel during warmup.
# scaling all adjustable time constants by k changes each exp(-t/tau) to exp(-(t/k)/tau).
# the adjustable part of the curve keeps its shape and runs on the stretched time axis.
# for example, k = 2 moves the adjustable response at 1 ms to 2 ms.
# x
# choose one time multiplier per qualifying heat path and chip so positive and negative fitted terms keep their cancellation.
# axes - B = simulated worlds, M = thermal lags.
def physics_shared_pair_time_scales(
    tau: Float64Array,  # tau = tau (s) - thermal time constants - type float64 ndarray, shape (B, M)
    res: Float64Array,  # res = r (K/W) - thermal residues - type float64 ndarray, shape (B, M)
    obs: Int64Array,  # obs = o_m (1) - observer indices - type int64 ndarray, shape (M,)
    src: Int64Array,  # src = s_m (1) - source indices - type int64 ndarray, shape (M,)
    layer: Int64Array,  # layer = layer (1) - fast or slow layer code of each lag - type int64 ndarray, shape (M,)
    fixed: BoolArray,  # fixed = fixed_tau (1) - lags whose absolute time constants must not be rescaled - type bool ndarray, shape (M,)
    s_fast: ArrayLike,  # s_fast = s_fast (1) - fast-layer time-scale multipliers - type array-like, shape (B,)
    s_slow: ArrayLike,  # s_slow = s_slow (1) - slow-layer time-scale multipliers - type array-like, shape (B,)
    tau_split: float,  # tau_split = tau_split (s) - boundary between fast and slow time scales - type float scalar
) -> tuple[
    BoolArray,  # return[0] = whole_pair (1) - lags that require one shared pair time scale - type bool ndarray, shape (M,)
    Float64Array | None,  # return[1] = s_pair (1) - shared time scales, or None when no pair needs coupling - type float64 ndarray, shape (B, M), or None
]:
    lag_count = obs.shape[0]
    shared_time_scale_mask = np.zeros(lag_count, bool)
    adjustable = ~fixed
    # without adjustable time constants there is no curve to rescale.
    if lag_count == 0 or not adjustable.any():
        return (
            shared_time_scale_mask,
            None,
        )

    # give lags with the same source and observer a common path index.
    source_index_span = int(src.max()) + 1
    path_keys = obs.astype(np.int64) * source_index_span + src.astype(np.int64)
    _, path_index = np.unique(path_keys, return_inverse=True)
    path_index = path_index.reshape(-1)
    path_count = int(path_index.max()) + 1
    adjustable_path_indices = path_index[adjustable]

    # first check whether the path has adjustable lags in both the fast and slow groups.
    has_fast_lag = np.zeros(path_count, bool)
    has_slow_lag = np.zeros(path_count, bool)
    has_fast_lag[adjustable_path_indices[layer[adjustable] == LAYER_FAST]] = True
    has_slow_lag[adjustable_path_indices[layer[adjustable] != LAYER_FAST]] = True
    # separately find the gain range across all adjustable lags on each path and chip.
    # both signs can occur in the same group. fixed lags do not enter either check.
    world_count = res.shape[0]
    min_residue = np.full((path_count, world_count), np.inf)
    max_residue = np.full((path_count, world_count), -np.inf)
    np.minimum.at(min_residue, adjustable_path_indices, res[:, adjustable].T)
    np.maximum.at(max_residue, adjustable_path_indices, res[:, adjustable].T)
    # one chip must have both signs on this path. np.any then checks across chips.
    has_mixed_signs = np.any((min_residue < 0.0) & (max_residue > 0.0), axis=1)
    # if both checks pass, share a multiplier across the path's adjustable lags on every chip.
    paths_needing_shared_scale = has_fast_lag & has_slow_lag & has_mixed_signs
    shared_time_scale_mask = adjustable & paths_needing_shared_scale[path_index]
    # if no path qualifies, keep the separate fast and slow factors.
    if not shared_time_scale_mask.any():
        return (
            shared_time_scale_mask,
            None,
        )

    selected_path_indices = path_index[shared_time_scale_mask]
    selected_gains_k_per_w = res[:, shared_time_scale_mask]
    selected_time_constants_s = tau[:, shared_time_scale_mask]

    # for a unit power step, each lag contributes r*(1-exp(-t/tau)) and settles to r.
    # sum both the cutoff response and the settled gains for each path and world.
    settled_gain_k_per_w = np.zeros((path_count, world_count))
    cutoff_gain_k_per_w = np.zeros((path_count, world_count))
    lag_fraction_at_cutoff = -np.expm1(-float(tau_split) / selected_time_constants_s)
    np.add.at(settled_gain_k_per_w, selected_path_indices, selected_gains_k_per_w.T)
    np.add.at(cutoff_gain_k_per_w, selected_path_indices, (selected_gains_k_per_w * lag_fraction_at_cutoff).T)
    with np.errstate(divide="ignore", invalid="ignore"):
        # w measures how much of the adjustable response has developed by the cutoff.
        # clip it to [0, 1] and use w = 0 for a zero settled gain.
        fast_weight = np.clip(
            np.where(settled_gain_k_per_w != 0.0, cutoff_gain_k_per_w / settled_gain_k_per_w, 0.0), 0.0, 1.0
        )
        fast_factor = np.asarray(s_fast, float)[None, :]
        slow_factor = np.asarray(s_slow, float)[None, :]
        # calculate k for each path and chip from the fitted step fraction w and the requested factors.
        # log(k) = w*log(s_fast) + (1-w)*log(s_slow), so k = s_fast^w*s_slow^(1-w).
        # for w = 0.5 with factors 2 and 0.5, the geometric mean is 1.
        path_time_factor = np.where(
            fast_factor == slow_factor,
            fast_factor,
            np.exp(fast_weight * np.log(fast_factor) + (1.0 - fast_weight) * np.log(slow_factor)),
        )
    # repeat the same path factor for its lags on each chip. the mask says where to use it.
    return (
        shared_time_scale_mask,
        np.ascontiguousarray(path_time_factor[path_index].T),
    )


# the stage functions from here to assemble each do one step of assembly.
# assemble calls them in order.

# turn the layout argument into a LayoutSites record with one centre per ring.
def assembly_resolve_layout_sites(
    layout: LayoutSites | Mapping[str, Mapping[str, Any]] | None,  # layout = layout (um) - site coordinates or layout records - type LayoutSites, mapping, or None
    ring_count: int,  # ring_count = N (1) - number of rings - type int scalar
    ring_pitch_um: float | None,  # ring_pitch_um = d_pitch (um) - optional spacing for missing ring sites - type float or None
    temp_sensors: Sequence[Mapping[str, Any]] | None,  # temp_sensors = sensors (mixed units) - sensor site or attachment records - type sequence of mappings or None
) -> (
    LayoutSites  # return = sites (um) - ring, monitor, driver, sensor, and aggressor positions - type LayoutSites record
):
    if layout is None:
        # one ring needs no positions. place it at the origin.
        if ring_count != 1:
            raise ValueError("layout is required for more than one ring")
        layout = LayoutSites(np.zeros((1, 2)))
    # construct a layout record from the supplied site description.
    elif not isinstance(layout, LayoutSites):
        layout = LayoutSites.assembly_layout_from_sites(layout, ring_count, ring_pitch_um, temp_sensors)
    # the site layout must account for every ring in the model.
    if layout.helper_ring_count != ring_count:
        raise ValueError(f"layout has {layout.helper_ring_count} rings, n_rings is {ring_count}")
    return layout


# turn each ThermalScales field into an array with 1 shared entry or B entries.
# the dict also lets helper_infer_world_count find B from the fields that have B entries.
# axes - B = simulated worlds.
def helper_normalize_thermal_scale_arrays(
    scales: ThermalScales,  # scales = scales (mixed units) - thermal uncertainty factors and optional package-tail parameters - type ThermalScales record
) -> (
    dict[str, Float64Array]  # return = scale_arrays (field units) - uncertainty factors keyed by short name - type dict of float64 ndarrays, shape (1 or B,)
):
    scale_arrays = {
        "time_scale_fast": helper_normalize_world_parameter_vector(scales.time_scale_fast, "time_scale_fast"),
        "time_scale_slow": helper_normalize_world_parameter_vector(scales.time_scale_slow, "time_scale_slow"),
        "crosstalk_scale": helper_normalize_world_parameter_vector(scales.crosstalk_scale, "crosstalk_scale"),
    }

    # include the package tail when a share of the dc gain was requested.
    if scales.package_tail_share_frac is not None:
        scale_arrays["pkg_tail_share"] = helper_normalize_world_parameter_vector(
            scales.package_tail_share_frac, "package_tail_share_frac"
        )
        # default tau when only the share is drawn - assumed 4 ms, [CO22-TCPMT] circuit-level start-up (row T2)
        package_tail_tau_s = 4.0e-3 if scales.package_tail_tau_s is None else scales.package_tail_tau_s
        scale_arrays["pkg_tail_tau"] = helper_normalize_world_parameter_vector(package_tail_tau_s, "package_tail_tau_s")

    # dc_scale is one factor, or a set of rules keyed by source and observer kind.
    if isinstance(scales.dc_scale, Mapping):
        # normalize each pair-specific scale over simulated worlds.
        for rule_key, rule_value in scales.dc_scale.items():
            scale_arrays[f"dc[{rule_key}]"] = helper_normalize_world_parameter_vector(rule_value, f"dc_scale[{rule_key}]")
    else:
        # a single dc scale applies uniformly across all heat paths.
        scale_arrays["dc"] = helper_normalize_world_parameter_vector(scales.dc_scale, "dc_scale")

    scale_arrays["transport_delay_s"] = helper_normalize_world_parameter_vector(scales.transport_delay_s, "transport_delay_s")
    # transport delay can postpone heat arrival only by a nonnegative interval.
    if np.any(~(scale_arrays["transport_delay_s"] >= 0.0)):
        raise ValueError("transport_delay_s must be >= 0")

    # the hypothesis entry is a placeholder. only its length matters, so that it counts toward B.
    hypothesis = scales.scaling_hypothesis
    # a single law is shared across chips. a sequence can select a different law for each chip.
    hypothesis_count = 1 if isinstance(hypothesis, str) else np.asarray(hypothesis).reshape(-1).shape[0]
    scale_arrays["scaling_hypothesis"] = np.zeros(hypothesis_count)
    return scale_arrays


# place the sources and observers, then choose r and tau for every heat path from the interim model.
# axes - B = simulated worlds.
def assembly_interim_heat_paths(
    spec: InterimRom,  # spec = model (mixed units) - interim thermal resistance and time-constant parameters - type InterimRom record
    layout: LayoutSites,  # layout = sites (um) - source and observer site coordinates - type LayoutSites record
    ring_count: int,  # ring_count = N (1) - number of rings - type int scalar
    source_kind_names: Sequence[str],  # source_kind_names = source_kinds (1) - source kind names to include - type sequence of strings
    observer_kind_names: Sequence[str],  # observer_kind_names = observer_kinds (1) - observer kind names to include - type sequence of strings
    scale_arrays: dict[str, Float64Array],  # scale_arrays = scales (field units) - uncertainty factors, used here only to find B - type dict of float64 ndarrays, shape (1 or B,)
    meta: dict[str, Any],  # meta = metadata (mixed units) - provenance record updated in place - type dict with string keys and metadata values
) -> tuple[
    int,  # return[0] = B (1) - number of worlds - type int scalar
    _Elements,  # return[1] = elements (mixed units) - source and observer kinds, names, coordinates, and ring attachments - type thermal element builder
    dict[str, Int64Array],  # return[2] = indices (1) - per-ring, temperature-sensor, and aggressor index maps - type dict of named int64 ndarrays, one index vector per kind with length equal to its site count
    _Lags,  # return[3] = lags (mixed units) - one entry per heat-path lag before scaling - type thermal lag builder
]:
    interim_params = validation_check_and_broadcast_interim_parameters(spec, ring_count)
    # a parameter with one column is the same for every ring. only its row count matters for B.
    params_for_world_count = {
        key: (values[:, 0] if values.shape[1] == 1 else values) for key, values in interim_params.items()
    }
    world_count = helper_infer_world_count({**scale_arrays, **params_for_world_count})
    interim_params = {
        key: np.broadcast_to(values, (world_count, ring_count)) for key, values in interim_params.items()
    }

    elements, element_indices = assembly_thermal_sources_and_observers(
        layout,
        ring_count,
        source_kind_names,
        observer_kind_names,
        spec.pd_offset_um,
        spec.temp_sensor_offset_um,
        spec.driver_offset_um,
    )
    lags = _Lags(world_count)
    assembly_append_interim_heat_path_lags(spec, elements, layout, ring_count, world_count, interim_params, lags)
    # record which assumptions were used to construct the interim thermal response.
    meta.update(
        model="interim", name=spec.name, crosstalk_dynamics=spec.crosstalk_dynamics, xt_tau_law=spec.xt_tau_law
    )

    # z*(1-f_sh) is the optical gain on the shared heater time scales, relative to the heater gain.
    shared_optical_gain_ratio = interim_params["z_ratio"] * (1.0 - interim_params["f_sh"])
    if np.any(shared_optical_gain_ratio < 1.0 - 1e-12):
        # stacklevel 3 reports the caller of assemble.
        warnings.warn("interim thermal model: z_ratio*(1 - selfheat_share) < 1, so the heater/ring residue matrix "
                      "of the heater poles is not positive semidefinite (not a passive RC network); "
                      "selfheat_share_frac = (z - 1)/z restores it", RuntimeWarning, stacklevel=3)
    return (
        world_count,
        elements,
        element_indices,
        lags,
    )


# place the sources and observers, then read r and tau for every heat path from a saved ThermalRom.
# axes - B = simulated worlds.
def assembly_extracted_heat_paths(
    rom: ThermalRom,  # rom = model (mixed units) - extracted poles, thermal residues, and site metadata - type ThermalRom record
    layout: LayoutSites,  # layout = sites (um) - source and observer site coordinates - type LayoutSites record
    ring_count: int,  # ring_count = N (1) - number of rings - type int scalar
    source_kind_names: Sequence[str],  # source_kind_names = source_kinds (1) - source kind names to include - type sequence of strings
    observer_kind_names: Sequence[str],  # observer_kind_names = observer_kinds (1) - observer kind names to include - type sequence of strings
    driver_requested: bool,  # driver_requested = driver_requested (1) - the caller named the driver source kind explicitly - type bool scalar
    scale_arrays: dict[str, Float64Array],  # scale_arrays = scales (field units) - uncertainty factors, used here only to find B - type dict of float64 ndarrays, shape (1 or B,)
    tau_split_s: float,  # tau_split_s = tau_split (s) - boundary between fast and slow time scales - type float scalar
    meta: dict[str, Any],  # meta = metadata (mixed units) - provenance record updated in place - type dict with string keys and metadata values
) -> tuple[
    int,  # return[0] = B (1) - number of worlds - type int scalar
    _Elements,  # return[1] = elements (mixed units) - source and observer kinds, names, coordinates, and ring attachments - type thermal element builder
    dict[str, Int64Array],  # return[2] = indices (1) - per-ring, temperature-sensor, and aggressor index maps - type dict of named int64 ndarrays, one index vector per kind with length equal to its site count
    _Lags,  # return[3] = lags (mixed units) - one entry per heat-path lag before scaling - type thermal lag builder
]:
    world_count = helper_infer_world_count(scale_arrays)
    # resolve the physical locations of off-centre temperature readouts.
    for site_kind in ("monitor_pd", "temp_sensor"):
        if site_kind in observer_kind_names and site_kind in rom.observers and site_kind not in rom.site_offsets_um:
            # stacklevel 3 reports the caller of assemble.
            warnings.warn(f"thermal ROM has no site offset for {site_kind}; its sites are placed at the ring centre",
                          RuntimeWarning, stacklevel=3)

    # include drivers at explicit layout sites.
    # an explicitly requested driver may use the ROM default offset.
    # G26 places its drivers on a separate EIC beside the PIC.
    driver_offset_um = None
    # use extracted driver heating when both the source and its request are present.
    if "driver" in rom.sources and driver_requested:
        driver_offset_um = rom.site_offsets_um.get("driver")

    elements, element_indices = assembly_thermal_sources_and_observers(
        layout,
        ring_count,
        source_kind_names,
        observer_kind_names,
        rom.site_offsets_um.get("monitor_pd", (0.0, 0.0)),
        rom.site_offsets_um.get("temp_sensor", (0.0, 0.0)),
        driver_offset_um,
    )
    lags = _Lags(world_count)
    assembly_append_extracted_heat_path_lags(rom, elements, layout, world_count, tau_split_s, lags, meta)
    meta.update(
        model="rom",
        rom_path=rom.meta.get("path"),
        rom_sha256=rom.meta.get("sha256"),
        family=rom.meta.get("family"),
        film_nm=rom.meta.get("film_nm"),
    )
    # return the fitted lag list with its element maps and assembly metadata.
    return (
        world_count,
        elements,
        element_indices,
        lags,
    )


# look up the settled-gain factor s_DC of every lag from its source kind and observer kind.
# axes - B = simulated worlds, S = heat sources, O = temperature observers, M = thermal lags.
def physics_lag_dc_factors(
    dc_scale: ArrayLike | Mapping[str, ArrayLike],  # dc_scale = s_DC (1) - DC residue scales or source-to-observer matching rules - type array-like or mapping to array-like values, scalar or shape (B,) per rule
    lag_source_indices: Int64Array,  # lag_source_indices = s_m (1) - source index of each lag - type int64 ndarray, shape (M,)
    lag_observer_indices: Int64Array,  # lag_observer_indices = o_m (1) - observer index of each lag - type int64 ndarray, shape (M,)
    src_kind: Int64Array,  # src_kind = source_kind (1) - kind code of each source - type int64 ndarray, shape (S,)
    obs_kind: Int64Array,  # obs_kind = observer_kind (1) - kind code of each observer - type int64 ndarray, shape (O,)
    world_count: int,  # world_count = B (1) - number of worlds - type int scalar
) -> (
    Float64Array  # return = s_DC (1) - settled-gain factor of each lag - type float64 ndarray, shape (B, M)
):
    lag_count = lag_source_indices.shape[0]
    dc_factors = np.ones((world_count, lag_count))
    # match each lag's scale to its source and observer kinds.
    for lag_index in range(lag_count):
        source_kind_name = SOURCE_KINDS[src_kind[lag_source_indices[lag_index]]]
        observer_kind_name = OBSERVER_KINDS[obs_kind[lag_observer_indices[lag_index]]]
        dc_factors[:, lag_index] = physics_pair_dc_scale(
            dc_scale, source_kind_name, observer_kind_name, world_count
        )
    return dc_factors


# choose the time factor of every lag - s_fast or s_slow by layer, or one shared factor per path.
# physics_shared_pair_time_scales decides which paths need the shared factor.
# axes - B = simulated worlds, M = thermal lags.
def physics_lag_time_factors(
    time_constants_s: Float64Array,  # time_constants_s = tau (s) - time constants before scaling - type float64 ndarray, shape (B, M)
    thermal_residues_k_per_w: Float64Array,  # thermal_residues_k_per_w = r (K/W) - residues before scaling - type float64 ndarray, shape (B, M)
    lag_observer_indices: Int64Array,  # lag_observer_indices = o_m (1) - observer index of each lag - type int64 ndarray, shape (M,)
    lag_source_indices: Int64Array,  # lag_source_indices = s_m (1) - source index of each lag - type int64 ndarray, shape (M,)
    lag_layers: Int64Array,  # lag_layers = layer (1) - fast or slow layer code of each lag - type int64 ndarray, shape (M,)
    fixed_time_constants: BoolArray,  # fixed_time_constants = fixed_tau (1) - lags whose time constants must not be rescaled - type bool ndarray, shape (M,)
    fast_time_scale: Float64Array,  # fast_time_scale = s_fast (1) - fast-layer time factor - type float64 ndarray, shape (B,)
    slow_time_scale: Float64Array,  # slow_time_scale = s_slow (1) - slow-layer time factor - type float64 ndarray, shape (B,)
    tau_split_s: float,  # tau_split_s = tau_split (s) - boundary between fast and slow time scales - type float scalar
    source_count: int,  # source_count = S (1) - number of sources - type int scalar
) -> tuple[
    Float64Array,  # return[0] = s_time (1) - time factor of each lag - type float64 ndarray, shape (B, M)
    int,  # return[1] = path_count (1) - number of paths that use one shared factor - type int scalar
]:
    # start from the layer rule - fast lags take s_fast and slow lags take s_slow.
    time_factors = np.where(lag_layers[None, :] == LAYER_FAST, fast_time_scale[:, None], slow_time_scale[:, None])
    # replace the layer factors where fitted lags need to keep their timing relative to each other.
    whole_pair_mask, whole_pair_factors = physics_shared_pair_time_scales(
        time_constants_s,
        thermal_residues_k_per_w,
        lag_observer_indices,
        lag_source_indices,
        lag_layers,
        fixed_time_constants,
        fast_time_scale,
        slow_time_scale,
        tau_split_s,
    )
    # unselected lags keep their requested fast or slow factor here.
    if whole_pair_factors is not None:
        time_factors = np.where(whole_pair_mask[None, :], whole_pair_factors, time_factors)
    # count the distinct source-to-observer paths among the selected lags.
    shared_path_keys = lag_observer_indices[whole_pair_mask] * source_count + lag_source_indices[whole_pair_mask]
    shared_path_count = int(np.unique(shared_path_keys).size)
    return (
        time_factors,
        shared_path_count,
    )


# choose which lags to keep and the order that puts each observer's lags together.
# axes - B = simulated worlds, M = thermal lags, M_kept = retained lags.
def helper_order_kept_lags_by_observer(
    time_constants_s: Float64Array,  # time_constants_s = tau (s) - scaled time constants - type float64 ndarray, shape (B, M)
    thermal_residues_k_per_w: Float64Array,  # thermal_residues_k_per_w = r (K/W) - scaled residues - type float64 ndarray, shape (B, M)
    lag_observer_indices: Int64Array,  # lag_observer_indices = o_m (1) - observer index of each lag - type int64 ndarray, shape (M,)
    lag_source_indices: Int64Array,  # lag_source_indices = s_m (1) - source index of each lag - type int64 ndarray, shape (M,)
    prune_abs_k_per_w: float,  # prune_abs_k_per_w = r_min (K/W) - prune lags whose residue magnitude never exceeds this threshold - type float scalar
) -> (
    Int64Array  # return = order (1) - indices of the kept lags in their final order - type int64 ndarray, shape (M_kept,)
):
    # keep a lag when its gain exceeds the threshold in at least one world.
    keep = np.any(np.abs(thermal_residues_k_per_w) > float(prune_abs_k_per_w), axis=0)
    # with zero threshold only exactly zero contributions can be removed.
    if prune_abs_k_per_w == 0.0:
        keep = np.any(thermal_residues_k_per_w != 0.0, axis=0)
    # sort by observer, then source, then world 0's time constant. lexsort takes its last key first.
    order = np.lexsort((time_constants_s[0], lag_source_indices, lag_observer_indices))
    return order[keep[order]]


# find where each observer's lags start and end in the sorted lag arrays.
# observer o reads lags obs_ptr[o] up to obs_ptr[o+1]-1.
# for example, lag observers [0, 0, 1, 2, 2, 2] give obs_ptr = [0, 2, 3, 6].
# axes - O = temperature observers, M = thermal lags.
def helper_count_observer_lag_boundaries(
    lag_observer_indices: Int64Array,  # lag_observer_indices = o_m (1) - observer index of each lag, sorted - type int64 ndarray, shape (M,)
    observer_count: int,  # observer_count = O (1) - number of observers - type int scalar
) -> (
    Int64Array  # return = p_observer (1) - lag-array boundaries of each observer - type int64 ndarray, shape (O+1,)
):
    # count the lags of observer o into slot o+1, then accumulate the counts.
    obs_ptr = np.zeros(observer_count + 1, np.int64)
    np.add.at(obs_ptr, lag_observer_indices + 1, 1)
    return np.cumsum(obs_ptr)


# label each lag by how the self-heating solver must treat its source power.
# only lags driven by waveguide absorption get an optical label. the rest stay LAG_EXOGENOUS.
# axes - S = heat sources, O = temperature observers, M = thermal lags.
def helper_classify_optical_lags(
    lag_source_indices: Int64Array,  # lag_source_indices = s_m (1) - source index of each lag - type int64 ndarray, shape (M,)
    lag_observer_indices: Int64Array,  # lag_observer_indices = o_m (1) - observer index of each lag - type int64 ndarray, shape (M,)
    src_kind: Int64Array,  # src_kind = source_kind (1) - kind code of each source - type int64 ndarray, shape (S,)
    src_ring: Int64Array,  # src_ring = k_source (1) - ring attachment of each source - type int64 ndarray, shape (S,)
    obs_kind: Int64Array,  # obs_kind = observer_kind (1) - kind code of each observer - type int64 ndarray, shape (O,)
    obs_ring: Int64Array,  # obs_ring = k_observer (1) - ring attachment of each observer - type int64 ndarray, shape (O,)
) -> (
    Int64Array  # return = kind_m (1) - LAG_* code of each lag - type int64 ndarray, shape (M,)
):
    lag_kinds = np.zeros(lag_observer_indices.shape[0], np.int64)
    optical_source_mask = src_kind[lag_source_indices] == SRC_WG
    ring_observer_mask = obs_kind[lag_observer_indices] == OBS_RING
    same_ring_mask = src_ring[lag_source_indices] == obs_ring[lag_observer_indices]
    # absorbed light warming its own ring - the solver finds this power and temperature together.
    lag_kinds[optical_source_mask & ring_observer_mask & same_ring_mask] = LAG_OPT_SELF
    # absorbed light warming another ring - the solver uses the absorbed power from the previous substep.
    lag_kinds[optical_source_mask & ring_observer_mask & ~same_ring_mask] = LAG_OPT_CROSS_RING
    # absorbed light warming a heater, monitor, or sensor - updated with the newly solved power.
    lag_kinds[optical_source_mask & ~ring_observer_mask] = LAG_OPT_OTHER
    return lag_kinds


# in the full picture - this runs once before a simulation. it builds the thermal model that the engines then read on every step.
# the model is a list of lags. a lag is one temperature contribution x (K) from one heat source to one place where temperature is read.
# each lag has a gain r (K/W), the settled rise per watt, and a time constant tau (s), the warmup time.
# the inputs say which model to use (rom), where things are (layout), and how this chip differs from nominal (scales).
# the output is a ThermalSystem - r and tau as arrays (B, M) for B simulated chips and M lags, plus index arrays that say which source and observer each lag joins.
# assemble the source-to-observer thermal system for the requested ring layout.
# choose the heat response for the layout, then assign the requested share of warming to the package lag.
# apply chip uncertainty last to set how much warming occurs and how quickly it develops.
# prune negligible residues and sort lags by observer, source, and time constant.
# classify optical heat paths for the implicit self-heating solver.
# stages 2, 3, and 5 change the predicted temperatures. stages 1, 4, 6, and 7 only prepare or pack.
# x
# build a thermal system that converts source powers in watts into the temperature rise at each measurement site.
def assembly_thermal_system(
    rom: ThermalRom | InterimRom,  # rom = model (mixed units) - extracted or interim thermal heat-path model - type ThermalRom or InterimRom record
    layout: LayoutSites | Mapping[str, Mapping[str, Any]] | None,  # layout = layout (um) - site coordinates or layout records - type LayoutSites, mapping, or None
    n_rings: int,  # n_rings = N (1) - number of rings - type int scalar
    scales: ThermalScales | Mapping[str, Any] | None = None,  # scales = scales (mixed units) - thermal uncertainty factors and optional package-tail parameters - type ThermalScales record, mapping, or None
    *,
    sources: Sequence[str] | None = None,  # sources = source_kinds (1) - source kind names to include, or all kinds - type sequence of strings or None
    observers: Sequence[str] | None = None,  # observers = observer_kinds (1) - observer kind names to include, or all kinds - type sequence of strings or None
    temp_sensors: Sequence[Mapping[str, Any]] | None = None,  # temp_sensors = sensors (mixed units) - sensor site or attachment records - type sequence of mappings or None
    ring_pitch_um: float | None = None,  # ring_pitch_um = d_pitch (um) - optional spacing for missing ring sites - type float or None
    prune_abs_k_per_w: float = 0.0,  # prune_abs_k_per_w = r_min (K/W) - prune lags whose residue magnitude never exceeds this threshold - type float scalar
) -> (
    ThermalSystem  # return = system (mixed units) - assembled and scaled source-to-observer heat paths - type ThermalSystem record
):
    # stage 1 - turn the arguments into records and per-world arrays.
    ring_count = int(n_rings)
    layout = assembly_resolve_layout_sites(layout, ring_count, ring_pitch_um, temp_sensors)
    # default scales reproduce the nominal thermal model.
    if scales is None:
        scales = ThermalScales()
    # translate a parameter mapping into the same scale record as explicit inputs.
    elif isinstance(scales, Mapping):
        scales = ThermalScales(**scales)
    # enable all known heat sources when the caller supplies no source selection.
    source_kind_names = tuple(SOURCE_KINDS if sources is None else sources)
    # enable all known temperature readouts when the caller supplies no observer selection.
    observer_kind_names = tuple(OBSERVER_KINDS if observers is None else observers)
    # confirm that all requested heat inputs have recognized source codes.
    for kind_name in source_kind_names:
        # a source without a code cannot be routed into the thermal state.
        if kind_name not in SOURCE_KINDS:
            raise ValueError(f"unknown source kind {kind_name!r}")
    # confirm that all requested temperature outputs have recognized observer codes.
    for kind_name in observer_kind_names:
        # an observer without a code cannot receive a thermal lag.
        if kind_name not in OBSERVER_KINDS:
            raise ValueError(f"unknown observer kind {kind_name!r}")
    scale_arrays = helper_normalize_thermal_scale_arrays(scales)
    tau_split_s = float(scales.tau_split_s)
    meta: dict = {"n_rings": ring_count}

    # stage 2 - place the sources and observers, then choose r and tau for every heat path.
    # lags is a list with one entry per lag. nothing is scaled yet.
    if isinstance(rom, InterimRom):
        world_count, elements, element_indices, lags = assembly_interim_heat_paths(
            rom, layout, ring_count, source_kind_names, observer_kind_names, scale_arrays, meta
        )
    # use measured residue tables when assembling an extracted thermal model.
    elif isinstance(rom, ThermalRom):
        driver_requested = sources is not None and "driver" in sources
        world_count, elements, element_indices, lags = assembly_extracted_heat_paths(
            rom, layout, ring_count, source_kind_names, observer_kind_names, driver_requested,
            scale_arrays, tau_split_s, meta,
        )
    else:
        # assembly needs either fitted thermal data or an interim parameter model.
        raise TypeError(f"rom must be a ThermalRom or an InterimRom, got {type(rom).__name__}")

    # stage 3 - move the requested share of each path's settled rise onto the package time scale.
    if "pkg_tail_share" in scale_arrays:
        # split each eligible path's original gain before applying the chip uncertainty factors.
        package_tail_share = np.broadcast_to(scale_arrays["pkg_tail_share"], (world_count,))
        # a fractional package tail must leave gain available for the remaining response.
        if np.any((package_tail_share < 0) | (package_tail_share >= 1)):
            raise ValueError("package_tail_share_frac must be in [0, 1)")
        package_time_constants_s = np.broadcast_to(scale_arrays["pkg_tail_tau"], (world_count,))
        physics_replace_package_tail_preserving_dc_gain(lags, package_time_constants_s, package_tail_share)

    # stage 4 - turn the list of lags into arrays with one column per lag.
    # tau and r have one row per world. the labels are the same in every world.
    lag_count = len(lags.observer_indices)
    # a thermal system needs at least one path connecting heat to temperature.
    if lag_count == 0:
        raise ValueError("the assembled thermal system has no lags (no sources or observers)")
    time_constants_s = np.stack(lags.time_constants_s, axis=1)
    thermal_residues_k_per_w = np.stack(lags.residues_k_per_w, axis=1)
    lag_observer_indices = np.array(lags.observer_indices, np.int64)
    lag_source_indices = np.array(lags.source_indices, np.int64)
    # keep the layer, source and observer labels aligned with each lag.
    lag_layers = np.array(lags.layer_codes, np.int64)
    lag_pair_types = np.array(lags.pair_codes, np.int64)
    lag_tail_flags = np.array(lags.tail_flags, np.int64)
    fixed_time_constants = np.array(lags.fixed_time_constants, bool)
    src_kind = np.array(elements.src_kind, np.int64)
    src_ring = np.array(elements.src_ring, np.int64)
    # record which temperature output receives each lag contribution.
    obs_kind = np.array(elements.obs_kind, np.int64)
    obs_ring = np.array(elements.obs_ring, np.int64)

    # stage 5 - apply this chip's uncertainty factors.
    # gain factors change the eventual temperature rise. time factors change how quickly it develops.
    hypothesis_codes = helper_encode_thermal_scaling_hypotheses(scales.scaling_hypothesis, world_count)
    dc_factors = physics_lag_dc_factors(
        scales.dc_scale, lag_source_indices, lag_observer_indices, src_kind, obs_kind, world_count
    )
    # s_cross applies to cross-ring lags only. it is one for every other lag.
    cross_factors = np.ones((world_count, lag_count))
    crosstalk_scale = np.broadcast_to(scale_arrays["crosstalk_scale"], (world_count,))
    cross_factors[:, lag_pair_types == PAIR_CROSS] = crosstalk_scale[:, None]
    fast_time_scale = np.broadcast_to(scale_arrays["time_scale_fast"], (world_count,))
    slow_time_scale = np.broadcast_to(scale_arrays["time_scale_slow"], (world_count,))
    time_factors, shared_path_count = physics_lag_time_factors(
        time_constants_s,
        thermal_residues_k_per_w,
        lag_observer_indices,
        lag_source_indices,
        lag_layers,
        fixed_time_constants,
        fast_time_scale,
        slow_time_scale,
        tau_split_s,
        len(elements.src_kind),
    )
    # record how many paths share one time multiplier across their fitted terms.
    meta["time_scale_whole_pairs"] = shared_path_count
    # under the resistance hypothesis, a gain multiplier of 1.5 also multiplies tau by 1.5 because tau = R*C.
    # the crosstalk gain factor affects only residues. an explicitly set package time stays fixed.
    uses_resistance_hypothesis = (hypothesis_codes == SCALING_HYPOTHESES.index("resistance"))[:, None]
    resistance_time_factor = np.where(uses_resistance_hypothesis, dc_factors, 1.0)
    # tau_new = tau*s_time*s_resistance, except where tau is fixed.
    time_constants_s = np.where(
        fixed_time_constants[None, :], time_constants_s, time_constants_s * time_factors * resistance_time_factor
    )
    # r_new = r*s_DC*s_cross. s_cross is one for lags outside the cross-ring class.
    thermal_residues_k_per_w = thermal_residues_k_per_w * dc_factors * cross_factors
    # every scaled pole must still give a finite decaying thermal response.
    if np.any(~(time_constants_s > 0)) or not np.all(np.isfinite(time_constants_s)):
        raise ValueError("assembled time constants must be finite and > 0 (check time scales and dc_scale)")

    # stage 6 - drop lags with no gain, then sort so each observer's lags sit together.
    order = helper_order_kept_lags_by_observer(
        time_constants_s, thermal_residues_k_per_w, lag_observer_indices, lag_source_indices, prune_abs_k_per_w
    )
    time_constants_s = time_constants_s[:, order]
    thermal_residues_k_per_w = thermal_residues_k_per_w[:, order]
    lag_observer_indices = lag_observer_indices[order]
    lag_source_indices = lag_source_indices[order]
    lag_layers = lag_layers[order]
    # permute the path labels together with their sorted temperature states.
    lag_pair_types = lag_pair_types[order]
    lag_tail_flags = lag_tail_flags[order]
    obs_ptr = helper_count_observer_lag_boundaries(lag_observer_indices, obs_kind.shape[0])

    # stage 7 - label the lags that the self-heating solver treats specially, then pack the record.
    lag_kinds = helper_classify_optical_lags(
        lag_source_indices, lag_observer_indices, src_kind, src_ring, obs_kind, obs_ring
    )
    # opt_lags lists the lags that are updated with the newly solved absorbed power.
    opt_lags = np.flatnonzero((lag_kinds == LAG_OPT_SELF) | (lag_kinds == LAG_OPT_OTHER)).astype(np.int64)
    meta.update(
        scaling_hypothesis=[SCALING_HYPOTHESES[code] for code in np.unique(hypothesis_codes)],
        tail_from_scales="pkg_tail_share" in scale_arrays,
        layout=layout,
    )
    return ThermalSystem(
        taus_s=np.ascontiguousarray(time_constants_s),
        res_k_per_w=np.ascontiguousarray(thermal_residues_k_per_w),
        lag_src=lag_source_indices,
        lag_obs=lag_observer_indices,
        lag_kind=lag_kinds,
        lag_pair=lag_pair_types,
        lag_layer=lag_layers,
        lag_tail=lag_tail_flags,
        obs_ptr=obs_ptr,
        src_kind=src_kind,
        src_ring=src_ring,
        src_names=tuple(elements.src_names),
        obs_kind=obs_kind,
        obs_ring=obs_ring,
        obs_names=tuple(elements.obs_names),
        ring_obs=element_indices["ring_obs"],
        heater_obs=element_indices["heater_obs"],
        pd_obs=element_indices["pd_obs"],
        temp_obs=element_indices["temp_obs"],
        heater_src=element_indices["heater_src"],
        wg_src=element_indices["wg_src"],
        junction_src=element_indices["junction_src"],
        driver_src=element_indices["driver_src"],
        aggressor_src=element_indices["aggressor_src"],
        opt_lags=opt_lags,
        n_rings=ring_count,
        meta=meta,
        transport_delay_s=np.broadcast_to(scale_arrays["transport_delay_s"], (world_count,)).copy(),
    )


# assemble an interim thermal system from the named physical parameter overrides.
# InterimRom defines the heat-path equations, default values, and provenance.
def assembly_interim_thermal_system(
    layout: LayoutSites | Mapping[str, Mapping[str, Any]] | None = None,  # layout = layout (um) - site coordinates or layout records - type LayoutSites, mapping, or None
    n_rings: int = 1,  # n_rings = N (1) - number of rings - type int scalar
    scales: ThermalScales | Mapping[str, Any] | None = None,  # scales = scales (mixed units) - thermal uncertainty factors and optional package-tail parameters - type ThermalScales record, mapping, or None
    *,
    sources: Sequence[str] | None = None,  # sources = source_kinds (1) - source kind names to include, or all kinds - type sequence of strings or None
    observers: Sequence[str] | None = None,  # observers = observer_kinds (1) - observer kind names to include, or all kinds - type sequence of strings or None
    temp_sensors: Sequence[Mapping[str, Any]] | None = None,  # temp_sensors = sensors (mixed units) - sensor site or attachment records - type sequence of mappings or None
    ring_pitch_um: float | None = None,  # ring_pitch_um = d_pitch (um) - optional spacing for missing ring sites - type float or None
    **preset: Any,  # preset = parameters (field units) - InterimRom field values - type keyword values with InterimRom field types
) -> (
    ThermalSystem  # return = system (mixed units) - assembled interim heat-path model - type ThermalSystem record
):
    # construct the interim heat-path model from this preset and return its assembled thermal states and site maps.
    return assembly_thermal_system(InterimRom(**preset), layout, n_rings, scales, sources=sources, observers=observers,
                    temp_sensors=temp_sensors, ring_pitch_um=ring_pitch_um)


# section 8 - known power histories

# simulate observer temperatures for supplied source-power samples.
# zoh holds the starting power constant within each interval.
# foh interpolates linearly between the two endpoint powers.
# the lag update is exact for that power hold.
# nonlinear optical feedback is solved separately by the self-heating module.
# simulation
# (todo) foh and zoh are ambigious names
# (todo) why not one fixed option instead of foh and zoh?
# x
# integrate the thermal lag equations over the supplied power history and sample the temperature at each observer.
# axes - B = simulated worlds, S = heat sources, O = temperature observers, M = thermal lags, T = time samples.
def physics_simulate_temperature_history(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    u_samples_w: ArrayLike,  # u_samples_w = P_samples (W) - source powers at sample boundaries - type array-like, shape (T, S) or (T, B, S)
    h_s: float,  # h_s = h (s) - thermal step duration - type float scalar
    *,
    hold: str = "zoh",  # hold = hold (1) - constant zoh or linear foh power interpolation - type str
    x0: ArrayLike | None = None,  # x0 = x_0 (K) - initial thermal lag-state temperature rises - type array-like, shape (B, M), or None
    t_base_k: ArrayLike = 0.0,  # t_base_k = T_base (K) - base temperature of each world - type array-like, scalar or shape (1 or B,)
    src_scale: ArrayLike | None = None,  # src_scale = s_source (1) - optional source residue multipliers - type array-like, shape (S,) or (1 or B, S), or None
) -> (
    Float64Array  # return = T_observer (K) - observer temperatures at every sample boundary - type float64 ndarray, shape (T, B, O)
):
    # source_power_w[k] holds every source's power at sample boundary k - shape (T, B or 1, S).
    source_power_w = np.asarray(u_samples_w, dtype=np.float64)
    # a two-dimensional history supplies a single shared series of source powers.
    if source_power_w.ndim == 2:
        source_power_w = source_power_w[:, None, :]
    # the input power is either constant or linear between samples.
    if hold not in ("zoh", "foh"):
        raise ValueError("hold must be 'zoh' or 'foh'")
    # each sample needs one input column per thermal source.
    if source_power_w.ndim != 3 or source_power_w.shape[2] != system.helper_source_count:
        raise ValueError(f"u_samples_w must be (n+1, B|1, S={system.helper_source_count}), got {source_power_w.shape}")
    sample_count = source_power_w.shape[0]
    world_count = max(system.helper_world_count, source_power_w.shape[1])
    # each world needs its own input history or access to one shared history.
    if system.helper_world_count not in (1, world_count) or source_power_w.shape[1] not in (1, world_count):
        raise ValueError(f"batch mismatch: system has {system.helper_world_count} worlds, u has {source_power_w.shape[1]}")

    # the step duration is fixed, so the weights a, c_start, and c_end are computed once.
    state_retention, start_power_gain_k_per_w, end_power_gain_k_per_w = system.physics_linear_hold_coefficients(h_s)
    # absent an initial state, start each lag at zero temperature rise.
    if x0 is None:
        state_rise_k = np.zeros((world_count, system.helper_lag_count))
    else:
        # copy a supplied initial rise so time integration leaves the caller's initial state intact.
        state_rise_k = np.array(x0, dtype=np.float64, copy=True)
    base_temperature_k = np.atleast_1d(np.asarray(t_base_k, np.float64))
    # absent a source calibration, use the specified power directly.
    if src_scale is None:
        source_scales = np.ones((1, system.helper_source_count))
    else:
        # normalize the supplied source gains to a world-by-source table.
        source_scales = np.atleast_2d(np.asarray(src_scale, np.float64))

    observer_temperature_k = np.zeros((sample_count, world_count, system.helper_observer_count))
    physics_write_observer_temperatures(state_rise_k, system.obs_ptr, base_temperature_k, observer_temperature_k[0])
    # integrate each interval using the powers at its sample boundaries.
    for sample_index in range(sample_count - 1):
        start_power_w = source_power_w[sample_index]
        # zoh holds the starting power to the end of the step. foh ramps to the next sample.
        end_power_w = start_power_w if hold == "zoh" else source_power_w[sample_index + 1]
        physics_advance_thermal_states(
            state_rise_k,
            state_retention,
            start_power_gain_k_per_w,
            end_power_gain_k_per_w,
            system.lag_src,
            np.ascontiguousarray(start_power_w),
            np.ascontiguousarray(end_power_w),
            source_scales,
        )
        physics_write_observer_temperatures(
            state_rise_k, system.obs_ptr, base_temperature_k, observer_temperature_k[sample_index + 1]
        )
    return observer_temperature_k


# section 9 - response queries and checks
# each query reads r and tau straight from the assembled lags. none advances a state.
# helper_select_thermal_response_lags and physics_sum_selected_thermal_responses pick the lags and sum them per path.
# the queries after them supply the formula for one lag.

# return selected observer/source indices, their lag indices, and local pair indices.
# axes - O_selected = selected observers, S_selected = selected sources, M_selected = selected lags.
def helper_select_thermal_response_lags(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    obs: ArrayLike | None,  # obs = observer_indices (1) - selected observers, or all observers when None - type integer scalar, index vector of shape (selection_count,), or None
    src: ArrayLike | None,  # src = source_indices (1) - selected sources, or all sources when None - type integer scalar, index vector of shape (selection_count,), or None
) -> tuple[
    Int64Array,  # return[0] = o_selected (1) - selected observer indices - type int64 ndarray, shape (O_selected,)
    Int64Array,  # return[1] = s_selected (1) - selected source indices - type int64 ndarray, shape (S_selected,)
    Int64Array,  # return[2] = m_selected (1) - selected lag indices - type int64 ndarray, shape (M_selected,)
    Int64Array,  # return[3] = o_local (1) - local observer index of each selected lag - type int64 ndarray, shape (M_selected,)
    Int64Array,  # return[4] = s_local (1) - local source index of each selected lag - type int64 ndarray, shape (M_selected,)
]:
    # use every temperature readout by default and preserve a requested selection's order.
    selected_observers = np.arange(system.helper_observer_count) if obs is None else np.atleast_1d(np.asarray(obs, np.int64))
    # select heat inputs independently so a query can request a subset of source-observer pairs.
    selected_sources = np.arange(system.helper_source_count) if src is None else np.atleast_1d(np.asarray(src, np.int64))
    observer_map = np.full(system.helper_observer_count, -1, np.int64)
    observer_map[selected_observers] = np.arange(selected_observers.size)
    source_map = np.full(system.helper_source_count, -1, np.int64)
    source_map[selected_sources] = np.arange(selected_sources.size)
    # map physical source and observer indices into the requested output axes.
    local_observer, local_source = observer_map[system.lag_obs], source_map[system.lag_src]
    selected_lags = np.flatnonzero((local_observer >= 0) & (local_source >= 0))
    return (
        selected_observers,
        selected_sources,
        selected_lags,
        local_observer[selected_lags],
        local_source[selected_lags],
    )


# sum per-lag kernel contributions into (B, selected observers, selected sources, samples).
# axes - B = simulated worlds, O_selected = selected observers, S_selected = selected sources, M_selected = selected lags.
def physics_sum_selected_thermal_responses(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    obs: ArrayLike | None,  # obs = observer_indices (1) - selected observers, or all observers when None - type integer scalar, index vector of shape (selection_count,), or None
    src: ArrayLike | None,  # src = source_indices (1) - selected sources, or all sources when None - type integer scalar, index vector of shape (selection_count,), or None
    kernel: Callable[[Float64Array, Float64Array], NumericArray],  # kernel = kernel (response units) - per-lag response function of time constants and thermal residues - type callable returning a numeric ndarray, maps shape (B, M_selected) to (B, M_selected, samples)
) -> (
    NumericArray  # return = response (kernel response units) - selected per-lag contributions summed by source and observer - type numeric ndarray, shape (B, O_selected, S_selected, samples)
):
    selected_observers, selected_sources, selected_lags, local_observer, local_source = helper_select_thermal_response_lags(system, obs, src)
    # evaluate only the lags whose heat source and temperature observer are both selected.
    contributions = kernel(system.taus_s[:, selected_lags], system.res_k_per_w[:, selected_lags])
    out = np.zeros((system.helper_world_count, selected_observers.size, selected_sources.size, contributions.shape[-1]),
        dtype=contributions.dtype)
    # add every lag to its observer and source cell, even when several lags share that cell.
    np.add.at(out, (slice(None), local_observer, local_source, slice(None)), contributions)
    return out


# calculate the amplitude and phase of each lag's temperature response to sinusoidal power.
# axes - B = simulated worlds, M = selected lags, F = frequencies.
def _physics_lag_frequency_gain(
    angular_frequency_rad_per_s: Float64Array,  # angular_frequency_rad_per_s = omega (rad/s) - source modulation frequencies - type float64 ndarray, shape (F,)
    time_constants_s: Float64Array,  # time_constants_s = tau (s) - time for each lag to develop - type float64 ndarray, shape (B, M)
    gains_k_per_w: Float64Array,  # gains_k_per_w = r (K/W) - settled temperature rise per watt for each lag - type float64 ndarray, shape (B, M)
) -> (
    Complex128Array  # return = Z (K/W) - complex temperature rise per watt at each frequency - type complex128 ndarray, shape (B, M, F)
):
    # for tau*dx/dt + x = r*P, a sinusoidal drive gives x/P = r/(1 + j*omega*tau).
    return gains_k_per_w[..., None] / (1.0 + 1j * angular_frequency_rad_per_s[None, None, :] * time_constants_s[..., None])


# calculate how each thermal lag responds when the source power increases at a constant rate.
# axes - B = simulated worlds, M = selected lags, T = time samples.
def _physics_lag_ramp_gain(
    times_s: Float64Array,  # times_s = t (s) - elapsed time since the ramp began - type float64 ndarray, shape (T,)
    time_constants_s: Float64Array,  # time_constants_s = tau (s) - time for each lag to develop - type float64 ndarray, shape (B, M)
    gains_k_per_w: Float64Array,  # gains_k_per_w = r (K/W) - settled temperature rise per watt for each lag - type float64 ndarray, shape (B, M)
) -> (
    Float64Array  # return = Z_ramp (K*s/W) - rise per unit power ramp rate - type float64 ndarray, shape (B, M, T)
):
    # integrating the step response r*(1-exp(-t/tau)) gives the rise per watt per second of ramp rate.
    time = times_s[None, None, :]
    tau = time_constants_s[..., None]
    return gains_k_per_w[..., None] * (time + tau * np.expm1(-time / tau))


# report each lag's steady temperature rise per watt after a constant power step settles.
# axes - B = simulated worlds, M = selected lags.
def _physics_lag_settled_gain(
    time_constants_s: Float64Array,  # time_constants_s = tau (s) - time for each lag to develop - type float64 ndarray, shape (B, M)
    gains_k_per_w: Float64Array,  # gains_k_per_w = r (K/W) - settled temperature rise per watt for each lag - type float64 ndarray, shape (B, M)
) -> (
    Float64Array  # return = Z_dc (K/W) - settled rise per watt with one response sample - type float64 ndarray, shape (B, M, 1)
):
    # at steady state dx/dt is zero, so tau drops out of tau*dx/dt + x = r*P and x/P equals r.
    return gains_k_per_w[..., None]


# complex transfer impedances Z(j 2 pi f) = sum_k r_k/(1 + j 2 pi f tau_k), shape (B, O', S', F), K/W.
# obs, src - index arrays selecting observers/sources (default - all).
# f_hz in Hz (cycles per second).
# transfer function from p phasor to x phasor (x phasor / p phasor)
# x
# axes - B = simulated worlds, F = frequencies, O_selected = selected observers, S_selected = selected sources.
def physics_thermal_frequency_response(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    f_hz: ArrayLike,  # f_hz = f (Hz) - frequencies at which to evaluate thermal response - type array-like, scalar or shape (F,)
    obs: ArrayLike | None = None,  # obs = observer_indices (1) - selected observers, or all observers when None - type integer scalar, index vector of shape (selection_count,), or None
    src: ArrayLike | None = None,  # src = source_indices (1) - selected sources, or all sources when None - type integer scalar, index vector of shape (selection_count,), or None
) -> (
    Complex128Array  # return = Z (K/W) - complex source-to-observer thermal impedances - type complex128 ndarray, shape (B, O_selected, S_selected, F)
):
    # convert cycles per second to angular frequency for the pole response 1/(1+j*omega*tau).
    w = 2.0 * np.pi * np.atleast_1d(np.asarray(f_hz, np.float64))
    # for a sinusoid, replace d/dt by j*omega in the lag ODE to get x/P = r/(1+j*omega*tau).
    # when omega*tau = 1, a positive-gain lag has gain r/sqrt(2) and a 45 degree phase lag.
    return physics_sum_selected_thermal_responses(system, obs, src, partial(_physics_lag_frequency_gain, w))


# calculate the temperature rise per watt after a constant power step for each lag and time.
# axes - B = simulated worlds, T = time samples, M_selected = selected lags.
def _physics_lag_step_gain(
    times_s: Float64Array,  # times_s = t (s) - evaluation times - shape (T,) - type float64 ndarray
    time_constants_s: Float64Array,  # time_constants_s = tau (s) - one time constant per world and selected lag - type float64 ndarray, shape (B, M_selected)
    gains_k_per_w: Float64Array,  # gains_k_per_w = r (K/W) - settled gain per world and selected lag - type float64 ndarray, shape (B, M_selected)
) -> (
    Float64Array  # return = Z_step (K/W) - gain per world, selected lag, and evaluation time - type float64 ndarray, shape (B, M_selected, T)
):
    # append a time axis and evaluate 1-exp(-t/tau) accurately near t = 0.
    fraction_developed = -np.expm1(-times_s[None, None, :] / time_constants_s[..., None])
    return gains_k_per_w[..., None] * fraction_developed



# how much does the observer warm after a step in source power?
# each lag contributes r_k*(1-exp(-t/tau_k)) in K/W, where r_k sets the final gain and tau_k sets the rise time.
# sum the lag gains at each time and multiply by the source power step (W) to get the temperature rise (K).
# x
# axes - B = simulated worlds, T = time samples, O_selected = selected observers, S_selected = selected sources.
def physics_thermal_step_response(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    t_s: ArrayLike,  # t_s = t (s) - times at which to evaluate thermal response - type array-like, scalar or shape (T,)
    obs: ArrayLike | None = None,  # obs = observer_indices (1) - selected observers, or all observers when None - type integer scalar, index vector of shape (selection_count,), or None
    src: ArrayLike | None = None,  # src = source_indices (1) - selected sources, or all sources when None - type integer scalar, index vector of shape (selection_count,), or None
) -> (
    Float64Array  # return = Z_step (K/W) - temperature rise per unit source-power step - type float64 ndarray, shape (B, O_selected, S_selected, T)
):
    times_s = np.atleast_1d(np.asarray(t_s, np.float64))


    return physics_sum_selected_thermal_responses(system, obs, src, partial(_physics_lag_step_gain, times_s))


# return the response kernel to a power ramp, with shape (B, O', S', T) in K*s/W.
# sum r_k*[t-tau_k*(1-exp(-t/tau_k))].
# multiply by ramp slope in W/s to obtain K.
# t_s uses s.
# evalulates the integral, from dp = ads change
# x
# axes - B = simulated worlds, T = time samples, O_selected = selected observers, S_selected = selected sources.
def physics_thermal_ramp_response(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    t_s: ArrayLike,  # t_s = t (s) - times at which to evaluate thermal response - type array-like, scalar or shape (T,)
    obs: ArrayLike | None = None,  # obs = observer_indices (1) - selected observers, or all observers when None - type integer scalar, index vector of shape (selection_count,), or None
    src: ArrayLike | None = None,  # src = source_indices (1) - selected sources, or all sources when None - type integer scalar, index vector of shape (selection_count,), or None
) -> (
    Float64Array  # return = Z_ramp (K*s/W) - temperature response per unit source-power ramp rate - type float64 ndarray, shape (B, O_selected, S_selected, T)
):
    t = np.atleast_1d(np.asarray(t_s, np.float64))
    # a power ramp is a stream of small power steps, so integrate the step gain from 0 to t.
    # integrating r*(1-exp(-t/tau)) gives r*[t-tau*(1-exp(-t/tau))].
    return physics_sum_selected_thermal_responses(system, obs, src, partial(_physics_lag_ramp_gain, t))


# static transfer resistances Z(0) = sum_k r_k, shape (B, O', S') in K/W.
# final temperature gain
# x
# axes - B = simulated worlds, O_selected = selected observers, S_selected = selected sources.
def physics_thermal_dc_gain(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    obs: ArrayLike | None = None,  # obs = observer_indices (1) - selected observers, or all observers when None - type integer scalar, index vector of shape (selection_count,), or None
    src: ArrayLike | None = None,  # src = source_indices (1) - selected sources, or all sources when None - type integer scalar, index vector of shape (selection_count,), or None
) -> (
    Float64Array  # return = Z_DC (K/W) - steady thermal resistance of each source-to-observer path - type float64 ndarray, shape (B, O_selected, S_selected)
):
    # as t tends to infinity, each lag's step fraction tends to 1 and only r remains.
    return physics_sum_selected_thermal_responses(system, obs, src, _physics_lag_settled_gain)[..., 0]


# construct continuous state-space matrices for selected thermal sources and observers.
# each selected lag obeys tau_m*dx_m/dt=-x_m+r_m*P_source.
# A=diag(-1/tau), B[m,s]=r_m/tau_m, C[o,m]=1, and D=0.
# the outputs are temperature rises above the base temperature.
# x matrix control system of slow and fast
# axes - O_selected = selected observers, S_selected = selected sources, M_selected = selected lags.
def physics_thermal_state_space(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    b: int = 0,  # b = b (1) - world index - type int scalar
    obs: ArrayLike | None = None,  # obs = observer_indices (1) - selected observers, or all observers when None - type integer scalar, index vector of shape (selection_count,), or None
    src: ArrayLike | None = None,  # src = source_indices (1) - selected sources, or all sources when None - type integer scalar, index vector of shape (selection_count,), or None
) -> tuple[
    Float64Array,  # return[0] = A (1/s) - continuous thermal state matrix - type float64 ndarray, shape (M_selected, M_selected)
    Float64Array,  # return[1] = B (K/(W*s)) - source-to-state drive matrix - type float64 ndarray, shape (M_selected, S_selected)
    Float64Array,  # return[2] = C (1) - state-to-observer summation matrix - type float64 ndarray, shape (O_selected, M_selected)
    Float64Array,  # return[3] = D (K/W) - zero direct feedthrough matrix - type float64 ndarray, shape (O_selected, S_selected)
]:
    selected_observers, selected_sources, selected_lags, lag_observer_rows, lag_source_columns = (
        helper_select_thermal_response_lags(system, obs, src)
    )
    lag_count = selected_lags.size
    lag_rows = np.arange(lag_count)
    tau = system.taus_s[b, selected_lags]
    r = system.res_k_per_w[b, selected_lags]
    # divide the lag ODE by tau to get the decay -1/tau and source drive r/tau.
    state_matrix = np.diag(-1.0 / tau)
    # each lag has one source, so each row of B has one nonzero entry.
    input_matrix = np.zeros((lag_count, selected_sources.size))
    input_matrix[lag_rows, lag_source_columns] = r / tau
    # C[o,m] = 1 routes lag m to observer o. D = 0 because power must act through the lag states.
    output_matrix = np.zeros((selected_observers.size, lag_count))
    output_matrix[lag_observer_rows, lag_rows] = 1.0
    feedthrough_matrix = np.zeros((selected_observers.size, selected_sources.size))
    return (
        state_matrix,
        input_matrix,
        output_matrix,
        feedthrough_matrix,
    )


# measure the largest relative difference between reciprocal heat paths.
# compare heater-to-heater, optical-to-ring, and heater-to-ring reciprocal pairs.
# ignore source or observer kinds that are absent from the assembled system.
# axes - F = frequencies.
def physics_reciprocity_error(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    f_hz: ArrayLike = (0.0, 1e3, 1e4, 1e5, 1e6),  # f_hz = f (Hz) - frequencies at which to evaluate thermal response - type array-like, scalar or shape (F,)
    b: int | None = None,  # b = b (1) - optional world index, or all worlds - type int or None
) -> (
    float  # return = error (1) - largest relative thermal reciprocity error - type float scalar
):
    impedance_k_per_w = physics_thermal_frequency_response(system, f_hz)
    # restrict the comparison to one chip when a world index is supplied.
    if b is not None:
        # retain the batch axis when selecting a single chip's response.
        impedance_k_per_w = impedance_k_per_w[b:b + 1]
    ring_count = system.n_rings
    largest_mismatch = 0.0
    largest_magnitude = 0.0
    # each row pairs a path with its reciprocal - swap where heat enters and where temperature is read.
    # forward source     | forward observer  | reverse source     | reverse observer
    reciprocal_pairs = (
        (system.heater_src, system.heater_obs, system.heater_src, system.heater_obs),
        (system.wg_src,     system.ring_obs,   system.wg_src,     system.ring_obs),
        (system.heater_src, system.ring_obs,   system.wg_src,     system.heater_obs),
    )
    for forward_src, forward_obs, reverse_src, reverse_obs in reciprocal_pairs:
        # the forward path runs from ring j to ring i. the reverse path runs from ring i to ring j.
        for i in range(ring_count):
            for j in range(ring_count):
                # index -1 marks a source or observer that is absent from the system.
                if min(forward_src[j], forward_obs[i], reverse_src[i], reverse_obs[j]) < 0:
                    continue
                forward_impedance = impedance_k_per_w[:, forward_obs[i], forward_src[j]]
                reverse_impedance = impedance_k_per_w[:, reverse_obs[j], reverse_src[i]]
                # track the largest path mismatch for normalization against the peak response.
                largest_mismatch = max(
                    largest_mismatch, float(np.max(np.abs(forward_impedance - reverse_impedance)))
                )
                largest_magnitude = max(
                    largest_magnitude,
                    float(np.max(np.abs(forward_impedance))),
                    float(np.max(np.abs(reverse_impedance))),
                )
    # divide the largest absolute mismatch by the largest magnitude among the compared responses.
    return largest_mismatch / largest_magnitude if largest_magnitude > 0 else 0.0


# DC optical self-heating resistance Z_opt->ring,i(0) of each ring, (B, N) K/W (sum of LAG_OPT_SELF residues).
# axes - B = simulated worlds, N = physical rings.
def physics_self_heating_dc_gain_k_per_w(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
) -> (
    Float64Array  # return = Z_optical_self (K/W) - optical self-heating resistance of each ring - type float64 ndarray, shape (B, N)
):
    out = np.zeros((system.helper_world_count, system.n_rings))
    m = np.flatnonzero(system.lag_kind == LAG_OPT_SELF)
    # sum the same-ring absorption gains by ring to get eventual warming per absorbed watt.
    np.add.at(out, (slice(None), system.src_ring[system.lag_src[m]]), system.res_k_per_w[:, m])
    return out


# bound the optical feedback gain introduced by using previous-step cross-ring absorbed power.
# g_i sums |c_start+c_end|*max|dP_abs,j/dT_j| over cross-ring lags into ring i.
# a small gain limits the effect of that one-step delay.
# bounds for calculating multiple effects together
# x
# axes - B = simulated worlds, N = physical rings.
def physics_cross_optical_lag_gain(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    h_s: float,  # h_s = h (s) - thermal step duration - type float scalar
    dpabs_dt_max_w_per_k: ArrayLike,  # dpabs_dt_max_w_per_k = B_dP_dT (W/K) - bounds on absorbed-power temperature derivatives - type array-like, shape (N,) or (B, N)
) -> (
    Float64Array  # return = g_cross (1) - lagged cross-ring optical feedback gain - type float64 ndarray, shape (B, N)
):
    power_slope_bounds = np.atleast_2d(np.asarray(dpabs_dt_max_w_per_k, np.float64))
    a, c0, c1 = system.physics_linear_hold_coefficients(h_s)
    cross_lags = np.flatnonzero(system.lag_kind == LAG_OPT_CROSS_RING)
    out = np.zeros((max(system.helper_world_count, power_slope_bounds.shape[0]), system.n_rings))
    # without cross-ring optical lags the feedback gain is zero.
    if cross_lags.size == 0:
        return out
    source_rings = system.src_ring[system.lag_src[cross_lags]]
    # identify the receiving ring for each cross-ring feedback path.
    observer_rings = system.obs_ring[system.lag_obs[cross_lags]]
    # K/W from one thermal step times W/K from the source ring gives K/K at the observer ring.
    # take absolute contributions before summing so opposite signs cannot hide a large gain.
    gain_contributions = np.abs(c0[:, cross_lags] + c1[:, cross_lags]) * np.broadcast_to(power_slope_bounds,
        (out.shape[0], power_slope_bounds.shape[1]))[:, source_rings]
    np.add.at(out, (slice(None), observer_rings), gain_contributions)
    return out


# raise ValueError if the lagged cross-ring optical gain reaches `limit` (default 0.05).
# returns its max.
# axes - N = physical rings.
def validation_assert_cross_lag_gain(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    h_s: float,  # h_s = h (s) - thermal step duration - type float scalar
    dpabs_dt_max_w_per_k: ArrayLike,  # dpabs_dt_max_w_per_k = B_dP_dT (W/K) - bounds on absorbed-power temperature derivatives - type array-like, shape (N,) or (B, N)
    limit: float = 0.05,  # limit = g_max (1) - maximum accepted lagged cross-ring optical gain - type float scalar
) -> (
    float  # return = g_cross_max (1) - largest cross-ring optical feedback gain below the limit - type float scalar
):
    g = physics_cross_optical_lag_gain(system, h_s, dpabs_dt_max_w_per_k)
    # an empty gain table contributes zero to the maximum feedback gain.
    gmax = float(np.max(g)) if g.size else 0.0
    # the largest loop gain determines whether the specified limit is exceeded.
    if gmax >= limit:
        b, i = np.unravel_index(int(np.argmax(g)), g.shape)
        # report the world and ring responsible for the excessive feedback.
        raise ValueError(f"lagged cross-ring optical heating gain {gmax:.3g} >= {limit} at world {b}, ring {i} for a "
                         f"{h_s:g} s substep; reduce the substep or solve the bank jointly")
    return gmax


# calculate the thermal-resistance multiplier from temperature-dependent conductivity.
# sum each material's reference-to-current conductivity ratio using its DC resistance share.
# the three shares describe silicon film, oxide, and silicon substrate.
# apply the result as a source-residue multiplier at a slowly varying temperature.
def physics_conductivity_residue_scale(
    t_k: ArrayLike,  # t_k = T (K) - temperatures for conductivity scaling - type array-like, broadcast input shape
    *,
    film_nm: float = 160.0,  # film_nm = d_film (nm) - silicon-film thickness - type float scalar
    weights: Sequence[float] = (0.15, 0.80, 0.05),  # weights = w (1) - film, oxide, and substrate resistance shares - type sequence of three floats
    t_ref_k: float = 300.0,  # t_ref_k = T_ref (K) - reference temperature for conductivity scaling - type float scalar
) -> (
    RealValue  # return = R(T)/R(T_ref) (1) - conductivity-based thermal-residue multiplier - type real scalar or float64 ndarray, shape input temperature shape
):
    from mrm import materials   # Python-level only (never inside a kernel)
    t = np.asarray(t_k, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    # the three material weights partition the effective thermal resistance.
    if w.shape != (3,) or np.any(w < 0) or abs(w.sum() - 1.0) > 1e-9:
        raise ValueError("weights must be three non-negative shares (film, oxide, substrate) summing to 1")
    # compare each material's conductivity at the operating temperature with its reference value.
    film_conductivity_ratio = np.asarray(materials.k_si_film_w_per_m_k(t,
        film_nm)) / float(materials.k_si_film_w_per_m_k(t_ref_k, film_nm))
    oxide_conductivity_ratio = np.asarray(materials.k_sio2_w_per_m_k(t)) / float(materials.k_sio2_w_per_m_k(t_ref_k))
    # scale the substrate contribution using its temperature-dependent conductivity.
    substrate_conductivity_ratio = (np.asarray(materials.k_si_bulk_w_per_m_k(t))
                                    / float(materials.k_si_bulk_w_per_m_k(t_ref_k)))
    # a material's resistance is proportional to 1/k. sum w_i*k_i(T_ref)/k_i(T) over the three materials.
    return w[0] / film_conductivity_ratio + w[1] / oxide_conductivity_ratio + w[2] / substrate_conductivity_ratio


# compatibility bindings for callers using the earlier function names.
calculate_linear_hold_lag_weights = physics_linear_hold_lag_weights
calculate_linear_hold_coefficients = physics_linear_hold_coefficients
calculate_constant_hold_coefficients = physics_constant_hold_coefficients
advance_thermal_state_for_world = physics_advance_thermal_state_for_world
advance_thermal_state_for_world_with_step_size = physics_advance_thermal_state_for_world_with_step_size
write_observer_temperatures_for_world = physics_write_observer_temperatures_for_world
shared_or_world_row_index = helper_shared_or_world_row_index
validate_thermal_step_array_shapes = validation_thermal_step_array_shapes
advance_thermal_states = physics_advance_thermal_states
write_observer_temperatures = physics_write_observer_temperatures
advance_thermal_state_and_temperatures_for_world = physics_advance_thermal_state_and_temperatures_for_world
advance_thermal_states_and_temperatures = physics_advance_thermal_states_and_temperatures
separate_known_heat_and_source_gain_for_observer = physics_separate_known_heat_and_source_gain_for_observer
sha256_file = io_sha256_file
build_rom_sidecar_path = io_build_rom_sidecar_path
find_rom_sidecar_path = io_find_rom_sidecar_path
build_thermal_rom_path = io_build_thermal_rom_path
save_rom = io_save_rom
load_rom = io_load_rom
read_rom_validation_metrics = helper_read_rom_validation_metrics
warn_about_rom_quality = validation_warn_about_rom_quality
validate_site_coordinates_um = validation_site_coordinates_um
calculate_crosstalk_time_constant_s = physics_crosstalk_time_constant_s
calculate_site_coupling_fraction = physics_site_coupling_fraction
find_thermal_element_index = helper_find_thermal_element_index
normalize_world_parameter_vector = helper_normalize_world_parameter_vector
infer_world_count = helper_infer_world_count
encode_thermal_scaling_hypotheses = helper_encode_thermal_scaling_hypotheses
build_thermal_sources_and_observers = assembly_thermal_sources_and_observers
calculate_neighbor_ranks_and_distances = physics_neighbor_ranks_and_distances
validate_and_broadcast_interim_parameters = validation_check_and_broadcast_interim_parameters
append_interim_heat_path_lags = assembly_append_interim_heat_path_lags
append_extracted_heat_path_lags = assembly_append_extracted_heat_path_lags
replace_package_tail_preserving_dc_gain = physics_replace_package_tail_preserving_dc_gain
calculate_pair_dc_scale = physics_pair_dc_scale
calculate_shared_pair_time_scales = physics_shared_pair_time_scales
resolve_layout_sites = assembly_resolve_layout_sites
normalize_thermal_scale_arrays = helper_normalize_thermal_scale_arrays
build_interim_heat_paths = assembly_interim_heat_paths
build_extracted_heat_paths = assembly_extracted_heat_paths
calculate_lag_dc_factors = physics_lag_dc_factors
calculate_lag_time_factors = physics_lag_time_factors
order_kept_lags_by_observer = helper_order_kept_lags_by_observer
count_observer_lag_boundaries = helper_count_observer_lag_boundaries
classify_optical_lags = helper_classify_optical_lags
assemble = assembly_thermal_system
assemble_interim_thermal_system = assembly_interim_thermal_system
simulate = physics_simulate_temperature_history
select_thermal_response_lags = helper_select_thermal_response_lags
sum_selected_thermal_responses = physics_sum_selected_thermal_responses
frequency_response = physics_thermal_frequency_response
step_response = physics_thermal_step_response
ramp_response = physics_thermal_ramp_response
dc_gain = physics_thermal_dc_gain
state_space = physics_thermal_state_space
reciprocity_error = physics_reciprocity_error
selfheat_dc_k_per_w = physics_self_heating_dc_gain_k_per_w
cross_optical_lag_gain = physics_cross_optical_lag_gain
assert_cross_lag_gain = validation_assert_cross_lag_gain
calculate_conductivity_residue_scale = physics_conductivity_residue_scale


# section 10 - previous names
# (todo) why not have separate observer/source arrays per 22 fitted ROM lags? would make naming more easier

# previous names remain aliases for callers that use the established API.
foh_scalar = physics_linear_hold_lag_weights
foh_coefficients = physics_linear_hold_coefficients
zoh_coefficients = physics_constant_hold_coefficients
thermal_step_b = physics_advance_thermal_state_for_world
thermal_step_h_b = physics_advance_thermal_state_for_world_with_step_size
thermal_outputs_b = physics_write_observer_temperatures_for_world
_row = helper_shared_or_world_row_index
_check_batch = validation_thermal_step_array_shapes
thermal_step = physics_advance_thermal_states
thermal_outputs = physics_write_observer_temperatures
thermal_advance_b = physics_advance_thermal_state_and_temperatures_for_world
thermal_advance = physics_advance_thermal_states_and_temperatures
observer_split_b = physics_separate_known_heat_and_source_gain_for_observer
_sidecar_path = io_build_rom_sidecar_path
_find_sidecar = io_find_rom_sidecar_path
rom_path = io_build_thermal_rom_path
rom_validation_numbers = helper_read_rom_validation_metrics
_check_rom_quality = validation_warn_about_rom_quality
_xy = validation_site_coordinates_um
crosstalk_tau_s = physics_crosstalk_time_constant_s
site_fraction = physics_site_coupling_fraction
_find = helper_find_thermal_element_index
_as_b = helper_normalize_world_parameter_vector
_batch_of = helper_infer_world_count
_hyp_codes = helper_encode_thermal_scaling_hypotheses
_build_elements = assembly_thermal_sources_and_observers
_ranks = physics_neighbor_ranks_and_distances
_interim_lags = assembly_append_interim_heat_path_lags
_rom_lags = assembly_append_extracted_heat_path_lags
_apply_tail = physics_replace_package_tail_preserving_dc_gain
_dc_factor = physics_pair_dc_scale
_whole_pair_time_scales = physics_shared_pair_time_scales
_interim_params = validation_check_and_broadcast_interim_parameters
interim_three_pole = assembly_interim_thermal_system
_select = helper_select_thermal_response_lags
_accumulate = physics_sum_selected_thermal_responses
k_temp_residue_scale = physics_conductivity_residue_scale

# the descriptive names are public as well.
__all__ += [
    "calculate_linear_hold_lag_weights",
    "calculate_linear_hold_coefficients",
    "calculate_constant_hold_coefficients",
    "advance_thermal_state_for_world",
    "advance_thermal_state_for_world_with_step_size",
    "write_observer_temperatures_for_world",
    "shared_or_world_row_index",
    "validate_thermal_step_array_shapes",
    "advance_thermal_states",
    "write_observer_temperatures",
    "advance_thermal_state_and_temperatures_for_world",
    "advance_thermal_states_and_temperatures",
    "separate_known_heat_and_source_gain_for_observer",
    "build_rom_sidecar_path",
    "find_rom_sidecar_path",
    "build_thermal_rom_path",
    "read_rom_validation_metrics",
    "warn_about_rom_quality",
    "validate_site_coordinates_um",
    "calculate_crosstalk_time_constant_s",
    "calculate_site_coupling_fraction",
    "find_thermal_element_index",
    "normalize_world_parameter_vector",
    "infer_world_count",
    "encode_thermal_scaling_hypotheses",
    "build_thermal_sources_and_observers",
    "calculate_neighbor_ranks_and_distances",
    "append_interim_heat_path_lags",
    "append_extracted_heat_path_lags",
    "replace_package_tail_preserving_dc_gain",
    "calculate_pair_dc_scale",
    "calculate_shared_pair_time_scales",
    "validate_and_broadcast_interim_parameters",
    "assemble_interim_thermal_system",
    "select_thermal_response_lags",
    "sum_selected_thermal_responses",
    "calculate_conductivity_residue_scale",
]

# category-prefixed names for the public functions exported by this module.
__all__ += [
    "physics_linear_hold_lag_weights",
    "physics_linear_hold_coefficients",
    "physics_constant_hold_coefficients",
    "physics_advance_thermal_state_for_world",
    "physics_advance_thermal_state_for_world_with_step_size",
    "physics_write_observer_temperatures_for_world",
    "helper_shared_or_world_row_index",
    "validation_thermal_step_array_shapes",
    "physics_advance_thermal_states",
    "physics_write_observer_temperatures",
    "physics_advance_thermal_state_and_temperatures_for_world",
    "physics_advance_thermal_states_and_temperatures",
    "physics_separate_known_heat_and_source_gain_for_observer",
    "io_sha256_file",
    "io_build_rom_sidecar_path",
    "io_find_rom_sidecar_path",
    "io_build_thermal_rom_path",
    "io_save_rom",
    "io_load_rom",
    "helper_read_rom_validation_metrics",
    "validation_warn_about_rom_quality",
    "validation_site_coordinates_um",
    "physics_crosstalk_time_constant_s",
    "physics_site_coupling_fraction",
    "helper_find_thermal_element_index",
    "helper_normalize_world_parameter_vector",
    "helper_infer_world_count",
    "helper_encode_thermal_scaling_hypotheses",
    "assembly_thermal_sources_and_observers",
    "physics_neighbor_ranks_and_distances",
    "validation_check_and_broadcast_interim_parameters",
    "assembly_append_interim_heat_path_lags",
    "assembly_append_extracted_heat_path_lags",
    "physics_replace_package_tail_preserving_dc_gain",
    "physics_pair_dc_scale",
    "physics_shared_pair_time_scales",
    "assembly_thermal_system",
    "assembly_interim_thermal_system",
    "physics_simulate_temperature_history",
    "helper_select_thermal_response_lags",
    "physics_sum_selected_thermal_responses",
    "physics_thermal_frequency_response",
    "physics_thermal_step_response",
    "physics_thermal_ramp_response",
    "physics_thermal_dc_gain",
    "physics_thermal_state_space",
    "physics_reciprocity_error",
    "physics_self_heating_dc_gain_k_per_w",
    "physics_cross_optical_lag_gain",
    "validation_assert_cross_lag_gain",
    "physics_conductivity_residue_scale",
]
