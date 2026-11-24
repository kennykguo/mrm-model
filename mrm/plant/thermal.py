# model how heat sources change ring, heater, photodiode, and sensor temperatures.
# each source-to-observer heat path is a sum of first-order lags, called a Foster model.
# tau_m*dx_m/dt = -x_m + r_m*u_source(t)
# T_observer = T_base + sum(x_m for that observer)
# x_m is a temperature contribution in K, not a separate ring.
# r_m uses K/W, tau_m uses s, and u uses W.
# several states represent heat stored and released on different time scales.
# absorbed optical power depends on ring temperature.
# mrm.plant.selfheat closes that nonlinear loop using these same states.
# foh_scalar gives one lag's exact update for a held or linearly changing input.
# InterimRom describes the default heat paths and records the source of each number.
# load_rom reads an extracted FEM reduced-order model and checks its sidecar metadata.
# assemble expands either model over all named sources and observers to create a ThermalSystem.
# ThermalScales applies uncertainty per world, with separate fast and slow time scales per die.
# thermal_advance updates temperatures when source power is known.
# use selfheat.implicit_step when optical power must be solved together with temperature.
# B is worlds, M is lag states, S is sources, and O is observers.
# residues and time constants have shape (B, M).
# source/observer index maps are shared across worlds. *_b kernels process one world's one-dimensional rows.
# batched kernels loop over worlds and accept shared rows where documented.
# coefficients can be cached for each step length.
# their exact update assumes the specified input hold within the step.
# it does not make a nonlinear optical feedback step exact.
# source notes are in docs/research/04_thermal_and_control.md. [CO22-TCPMT] means Coenen et al., IEEE TCPMT 2022.
# its measured and inferred numbers form the interim model's main reference.
# detailed implementation assumptions are in docs/THERMAL_IMPLEMENTATION_NOTES.md.



from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Callable, Mapping, Sequence, TypeAlias
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


# integrate one thermal lag with linearly varying source power.
# for tau*dx/dt=-x+r*P, the update is x_next=a*x+r*(b_start*P_start+b_end*P_end).
# at time t within the step, P(t) = (1-t/h)*P_start + (t/h)*P_end.
# integrate each power weight against exp(-(h-t)/tau)/tau from t=0 to t=h.
# with y=h/tau, a=exp(-y), b_end=1-(1-a)/y, and b_start=(1-a)/y-a.
# use the Taylor forms at small y to avoid cancellation.
# a nonpositive step leaves the state unchanged.
@njit(cache=True)
def calculate_linear_hold_lag_weights(
    tau: float,  # tau = tau (s) - thermal lag time constant - type float scalar
    h: float,  # h = h (s) - thermal step duration - type float scalar
) -> tuple[
    float,  # return[0] = a (1) - retained state fraction - type float scalar
    float,  # return[1] = b_start (1) - weight on starting source power - type float scalar
    float,  # return[2] = b_end (1) - weight on ending source power - type float scalar
]:
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


# calculate exact thermal update coefficients for linearly varying source power.
# x_next=a*x+c_start*P_start+c_end*P_end.
# multiply each scalar hold weight by its thermal residue in K/W.
# a zero-duration step retains every state.
def calculate_linear_hold_coefficients(
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
    if step_duration_s < 0.0:
        raise ValueError(f"step h_s must be >= 0, got {h_s}")
    if np.any(~(time_constants_s > 0.0)):
        raise ValueError("every tau must be > 0")
    if step_duration_s == 0.0:
        # a = 1 and both power coefficients are zero because no time elapses.
        coefficient_shape = np.broadcast(time_constants_s, thermal_residues_k_per_w).shape
        state_retention = np.ones(coefficient_shape)
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
    if np.any(use_small_step_series):
        small_step_ratios = step_over_time_constant[use_small_step_series]
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


# calculate exact thermal update coefficients for constant source power.
# x_next=a*x+c*P, with c=r*(1-a).
# x_next is x at h
# (todo) why are there so many wrapper functions? get to the core of things. remove this in the future
# x
def calculate_constant_hold_coefficients(
    taus_s: ArrayLike,  # taus_s = tau (s) - thermal time constants - type array-like, broadcast input shape
    res_k_per_w: ArrayLike,  # res_k_per_w = r (K/W) - thermal residues - type array-like, broadcast input shape
    h_s: float,  # h_s = h (s) - thermal step duration - type float scalar
) -> tuple[
    Float64Array,  # return[0] = a (1) - retained state fractions - type float64 ndarray, shape broadcast input shape
    Float64Array,  # return[1] = c (K/W) - constant-power coefficients - type float64 ndarray, shape broadcast input shape
]:
    state_retention, start_power_gain_k_per_w, end_power_gain_k_per_w = calculate_linear_hold_coefficients(
        taus_s, res_k_per_w, h_s
    )
    # P_start = P_end = P, so the two power terms combine to (c_start+c_end)*P.
    return (
        state_retention,
        start_power_gain_k_per_w + end_power_gain_k_per_w,
    )


# advance one world's thermal lag states in place.
# each lag receives the start and end powers of its mapped source.
# source_residue_scale multiplies the source-to-temperature gain.
# x
@njit(cache=True)
def advance_thermal_state_for_world(
    state_rise_k: Float64Array,  # state_rise_k = x (K) - lag-state temperature rises updated in place - type float64 ndarray, shape (M,)
    state_retention: Float64Array,  # state_retention = a (1) - retained thermal state fractions - type float64 ndarray, shape (M,)
    start_power_gain_k_per_w: Float64Array,  # start_power_gain_k_per_w = c0 (K/W) - starting-power gain - type float64 ndarray, shape (M,)
    end_power_gain_k_per_w: Float64Array,  # end_power_gain_k_per_w = c1 (K/W) - ending-power gain - type float64 ndarray, shape (M,)
    lag_source_indices: Int64Array,  # lag_source_indices = s_m (1) - source index driving each lag - type int64 ndarray, shape (M,)
    start_source_power_w: Float64Array,  # start_source_power_w = P_start (W) - source powers at the step boundary - type float64 ndarray, shape (S,)
    end_source_power_w: Float64Array,  # end_source_power_w = P_end (W) - source powers at the step boundary - type float64 ndarray, shape (S,)
    source_residue_scale: Float64Array,  # source_residue_scale = s_source (1) - per-source residue multipliers - type float64 ndarray, shape (S,)
) -> (
    None  # return = None - results are written into the supplied state or record - type None
):
    for lag_index in range(state_rise_k.shape[0]):
        source_index = lag_source_indices[lag_index]
        # a*x is the stored heat that remains. the power terms add new warming in K.
        # x_next = a*x + s_source*(c_start*P_start + c_end*P_end), in K.
        state_rise_k[lag_index] = state_retention[lag_index] * state_rise_k[lag_index] + source_residue_scale[source_index] * (start_power_gain_k_per_w[lag_index] * start_source_power_w[source_index] + end_power_gain_k_per_w[lag_index] * end_source_power_w[source_index])


# advance one world's lag states with a caller-selected step duration.
# calculate the exact linear-hold weights separately for each time constant.
# per lag
# x
@njit(cache=True)
def advance_thermal_state_for_world_with_step_size(
    state_rise_k: Float64Array,  # state_rise_k = x (K) - lag-state temperature rises updated in place - type float64 ndarray, shape (M,)
    time_constants_s: Float64Array,  # time_constants_s = tau (s) - thermal time constants - type float64 ndarray, shape (M,)
    thermal_residues_k_per_w: Float64Array,  # thermal_residues_k_per_w = r (K/W) - thermal residues - type float64 ndarray, shape (M,)
    lag_source_indices: Int64Array,  # lag_source_indices = s_m (1) - source index driving each lag - type int64 ndarray, shape (M,)
    start_source_power_w: Float64Array,  # start_source_power_w = P_start (W) - source powers at the step boundary - type float64 ndarray, shape (S,)
    end_source_power_w: Float64Array,  # end_source_power_w = P_end (W) - source powers at the step boundary - type float64 ndarray, shape (S,)
    source_residue_scale: Float64Array,  # source_residue_scale = s_source (1) - per-source residue multipliers - type float64 ndarray, shape (S,)
    step_duration_s: float,  # step_duration_s = h (s) - thermal step duration - type float scalar
) -> (
    None  # return = None - results are written into the supplied state or record - type None
):
    for lag_index in range(state_rise_k.shape[0]):
        source_index = lag_source_indices[lag_index]
        state_retention, start_power_weight, end_power_weight = calculate_linear_hold_lag_weights(time_constants_s[lag_index], step_duration_s)
        # x_next = a*x + s_source*r*(b_start*P_start + b_end*P_end), in K.
        state_rise_k[lag_index] = state_retention * state_rise_k[lag_index] + source_residue_scale[source_index] * thermal_residues_k_per_w[lag_index] * (
            start_power_weight * start_source_power_w[source_index] + end_power_weight * end_source_power_w[source_index]
        )


# sum one world's thermal states into absolute observer temperatures.
# each observer reads the contiguous lag range in observer_lag_boundaries.
# x
@njit(cache=True)
def write_observer_temperatures_for_world(
    state_rise_k: Float64Array,  # state_rise_k = x (K) - lag-state temperature rises updated in place - type float64 ndarray, shape (M,)
    observer_lag_boundaries: Int64Array,  # observer_lag_boundaries = p_observer (1) - lag-array boundaries of each observer - type int64 ndarray, shape (O+1,)
    base_temperature_k: float,  # base_temperature_k = T_base (K) - base temperature - type float scalar
    out_observer_temperature_k: Float64Array,  # out_observer_temperature_k = T_observer (K) - absolute observer temperatures written in place - type float64 ndarray, shape (O,)
) -> (
    None  # return = None - results are written into the supplied state or record - type None
):
    for observer_index in range(out_observer_temperature_k.shape[0]):
        # T_observer = T_base + sum(x_m) over the lags that reach this observer.
        temperature_k = base_temperature_k
        for lag_index in range(observer_lag_boundaries[observer_index], observer_lag_boundaries[observer_index + 1]):
            temperature_k += state_rise_k[lag_index]
        out_observer_temperature_k[observer_index] = temperature_k


# row of world b in an array with n rows - b, or 0 when the row is shared by every world (n = 1).
# x
# (todo) many functions are just replicating or updating state, might as well move core functions to diff folder
# x
# helper function
@njit(cache=True)
def shared_or_world_row_index(
    world_index: int,  # world_index = b (1) - world index - type int scalar
    row_count: int,  # row_count = row_count (1) - number of shared or per-world rows - type int scalar
) -> (
    int  # return = row (1) - world row or shared row zero - type int scalar
):
    return world_index if row_count > 1 else 0


# validate state, coefficient, source, and batch dimensions before array access.
# a shared coefficient or power row has one row instead of B.
# the compiled kernels require these checks because bounds checking is disabled.
# (todo) just a validation func, could probably move to diff folder
# x
# helper function
@njit(cache=True)
def validate_thermal_step_array_shapes(
    state_rise_k: Float64Array,  # state_rise_k = x (K) - lag-state temperature rises updated in place - type float64 ndarray, shape (B, M)
    state_retention: Float64Array,  # state_retention = a (1) - retained thermal state fractions - type float64 ndarray, shape (1 or B, M)
    start_power_gain_k_per_w: Float64Array,  # start_power_gain_k_per_w = c0 (K/W) - starting-power gain - type float64 ndarray, shape (1 or B, M)
    end_power_gain_k_per_w: Float64Array,  # end_power_gain_k_per_w = c1 (K/W) - ending-power gain - type float64 ndarray, shape (1 or B, M)
    lag_source_indices: Int64Array,  # lag_source_indices = s_m (1) - source index driving each lag - type int64 ndarray, shape (M,)
    start_source_power_w: Float64Array,  # start_source_power_w = P_start (W) - source powers at the step boundary - type float64 ndarray, shape (1 or B, S)
    end_source_power_w: Float64Array,  # end_source_power_w = P_end (W) - source powers at the step boundary - type float64 ndarray, shape (1 or B, S)
    source_residue_scale: Float64Array,  # source_residue_scale = s_source (1) - per-source residue multipliers - type float64 ndarray, shape (1 or B, S)
) -> (
    None  # return = None - results are written into the supplied state or record - type None
):
    world_count = state_rise_k.shape[0]
    lag_count = state_rise_k.shape[1]
    coefficient_row_count = state_retention.shape[0]
    if start_power_gain_k_per_w.shape[0] != coefficient_row_count or end_power_gain_k_per_w.shape[0] != coefficient_row_count or (coefficient_row_count != 1 and coefficient_row_count != world_count):
        raise ValueError("thermal kernel: a, c0, c1 must all have 1 or B rows")
    if state_retention.shape[1] != lag_count or start_power_gain_k_per_w.shape[1] != lag_count or end_power_gain_k_per_w.shape[1] != lag_count or lag_source_indices.shape[0] != lag_count:
        raise ValueError("thermal kernel: a, c0, c1, lag_src must have M (= x.shape[1]) lags")
    for n in (start_source_power_w.shape[0], end_source_power_w.shape[0], source_residue_scale.shape[0]):
        if n != 1 and n != world_count:
            raise ValueError("thermal kernel: u0, u1, src_scale must have 1 or B rows")


# advance every world's lag states using shared or per-world coefficient rows.
# wrapper function, prob move out/reorg
# x
# helper function
@njit(cache=True)
def advance_thermal_states(
    state_rise_k: Float64Array,  # state_rise_k = x (K) - lag-state temperature rises updated in place - type float64 ndarray, shape (B, M)
    state_retention: Float64Array,  # state_retention = a (1) - retained thermal state fractions - type float64 ndarray, shape (1 or B, M)
    start_power_gain_k_per_w: Float64Array,  # start_power_gain_k_per_w = c0 (K/W) - starting-power gain - type float64 ndarray, shape (1 or B, M)
    end_power_gain_k_per_w: Float64Array,  # end_power_gain_k_per_w = c1 (K/W) - ending-power gain - type float64 ndarray, shape (1 or B, M)
    lag_source_indices: Int64Array,  # lag_source_indices = s_m (1) - source index driving each lag - type int64 ndarray, shape (M,)
    start_source_power_w: Float64Array,  # start_source_power_w = P_start (W) - source powers at the step boundary - type float64 ndarray, shape (1 or B, S)
    end_source_power_w: Float64Array,  # end_source_power_w = P_end (W) - source powers at the step boundary - type float64 ndarray, shape (1 or B, S)
    source_residue_scale: Float64Array,  # source_residue_scale = s_source (1) - per-source residue multipliers - type float64 ndarray, shape (1 or B, S)
) -> (
    None  # return = None - results are written into the supplied state or record - type None
):
    validate_thermal_step_array_shapes(state_rise_k, state_retention, start_power_gain_k_per_w, end_power_gain_k_per_w, lag_source_indices, start_source_power_w, end_source_power_w, source_residue_scale)
    for world_index in range(state_rise_k.shape[0]):
        coefficient_row = shared_or_world_row_index(world_index, state_retention.shape[0])
        power_row = shared_or_world_row_index(world_index, start_source_power_w.shape[0])
        advance_thermal_state_for_world(state_rise_k[world_index], state_retention[coefficient_row], start_power_gain_k_per_w[coefficient_row], end_power_gain_k_per_w[coefficient_row], lag_source_indices, start_source_power_w[power_row], end_source_power_w[shared_or_world_row_index(world_index, end_source_power_w.shape[0])],
                       source_residue_scale[shared_or_world_row_index(world_index, source_residue_scale.shape[0])])


# write all absolute observer temperatures from the current lag states and base temperatures.
# x
# helper function
@njit(cache=True)
def write_observer_temperatures(
    state_rise_k: Float64Array,  # state_rise_k = x (K) - lag-state temperature rises updated in place - type float64 ndarray, shape (B, M)
    observer_lag_boundaries: Int64Array,  # observer_lag_boundaries = p_observer (1) - lag-array boundaries of each observer - type int64 ndarray, shape (O+1,)
    base_temperature_k: ArrayLike,  # base_temperature_k = T_base (K) - base temperature of each world - type array-like, scalar or shape (1 or B,)
    out_observer_temperature_k: Float64Array,  # out_observer_temperature_k = T_observer (K) - absolute observer temperatures written in place - type float64 ndarray, shape (B, O)
) -> (
    None  # return = None - results are written into the supplied state or record - type None
):
    if out_observer_temperature_k.shape[0] != state_rise_k.shape[0] or (base_temperature_k.shape[0] != 1 and base_temperature_k.shape[0] != state_rise_k.shape[0]):
        raise ValueError("thermal_outputs: out must have B rows and t_base 1 or B entries")
    if out_observer_temperature_k.shape[1] + 1 != observer_lag_boundaries.shape[0]:
        raise ValueError("thermal_outputs: out must have O = len(obs_ptr) - 1 columns")
    for world_index in range(state_rise_k.shape[0]):
        write_observer_temperatures_for_world(state_rise_k[world_index], observer_lag_boundaries, base_temperature_k[shared_or_world_row_index(world_index, base_temperature_k.shape[0])], out_observer_temperature_k[world_index])


# advance one world's states and write its observer temperatures in one pass.
# use the same linear-hold update as the state-only kernel.
# x
@njit(cache=True, _nrt=False)
def advance_thermal_state_and_temperatures_for_world(
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
    None  # return = None - results are written into the supplied state or record - type None
):
    # apply the lag update, then add each updated rise to its observer's base temperature.
    for observer_index in range(out_observer_temperature_k.shape[0]):
        temperature_k = base_temperature_k
        for lag_index in range(observer_lag_boundaries[observer_index], observer_lag_boundaries[observer_index + 1]):
            source_index = lag_source_indices[lag_index]
            # x_next = a*x + s_source*(c_start*P_start+c_end*P_end).
            next_lag_k = state_retention[lag_index] * state_rise_k[lag_index] + source_residue_scale[source_index] * (start_power_gain_k_per_w[lag_index] * start_source_power_w[source_index] + end_power_gain_k_per_w[lag_index] * end_source_power_w[source_index])
            state_rise_k[lag_index] = next_lag_k
            # sum the new lag rises above T_base to get the end-of-step temperature.
            temperature_k += next_lag_k
        out_observer_temperature_k[observer_index] = temperature_k


# advance all worlds and write their end-of-step observer temperatures.
# validate the batch dimensions before selecting shared rows.
# x
# helper function
@njit(cache=True)
def advance_thermal_states_and_temperatures(
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
    None  # return = None - results are written into the supplied state or record - type None
):
    validate_thermal_step_array_shapes(state_rise_k, state_retention, start_power_gain_k_per_w, end_power_gain_k_per_w, lag_source_indices, start_source_power_w, end_source_power_w, source_residue_scale)
    if out_observer_temperature_k.shape[0] != state_rise_k.shape[0] or (base_temperature_k.shape[0] != 1 and base_temperature_k.shape[0] != state_rise_k.shape[0]):
        raise ValueError("thermal_advance: out must have B rows and t_base 1 or B entries")
    if out_observer_temperature_k.shape[1] + 1 != observer_lag_boundaries.shape[0]:
        raise ValueError("thermal_advance: out must have O = len(obs_ptr) - 1 columns")
    for world_index in range(state_rise_k.shape[0]):
        coefficient_row = shared_or_world_row_index(world_index, state_retention.shape[0])
        advance_thermal_state_and_temperatures_for_world(state_rise_k[world_index], state_retention[coefficient_row], start_power_gain_k_per_w[coefficient_row], end_power_gain_k_per_w[coefficient_row], lag_source_indices, observer_lag_boundaries, start_source_power_w[shared_or_world_row_index(world_index, start_source_power_w.shape[0])],
                          end_source_power_w[shared_or_world_row_index(world_index, end_source_power_w.shape[0])], source_residue_scale[shared_or_world_row_index(world_index, source_residue_scale.shape[0])],
                          base_temperature_k[shared_or_world_row_index(world_index, base_temperature_k.shape[0])], out_observer_temperature_k[world_index])


# T_end-T_base=T_known+g_self*P_self, with P_self constant over the step.
# include stored heat and every other source in T_known.
# use the same other-source powers in the subsequent state update.
# the heater resistance solve uses this split for its temperature feedback.
# doing calculation for known temp and gain
# x
@njit(cache=True)
def separate_known_heat_and_source_gain_for_observer(
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
    for lag_index in range(observer_lag_boundaries[observer_index], observer_lag_boundaries[observer_index + 1]):
        source_index = lag_source_indices[lag_index]
        # a*x is known for every lag, including those driven by the unknown source.
        known_rise_k += state_retention[lag_index] * state_rise_k[lag_index]
        # for held unknown power, P_start = P_end = P_unknown, so its gain is s_source*(c_start+c_end).
        if source_index == unknown_source_index:
            self_gain_k_per_w += source_residue_scale[source_index] * (start_power_gain_k_per_w[lag_index] + end_power_gain_k_per_w[lag_index])
        else:
            # add the known input's s_source*(c_start*P_start+c_end*P_end) directly in K.
            known_rise_k += source_residue_scale[source_index] * (start_power_gain_k_per_w[lag_index] * start_source_power_w[source_index] + end_power_gain_k_per_w[lag_index] * end_source_power_w[source_index])
    return (
        known_rise_k,
        self_gain_k_per_w,
    )


# artefact I/O (format thermal_rom/1 - docs/ARCHITECTURE.md, 5.3 "Thermal ROM")
# hex sha256 of a file's bytes.
# x
# helper function
def sha256_file(
    path: str | Path,  # path = path (1) - thermal artifact path - type str or Path
) -> (
    str  # return = digest (1) - hexadecimal SHA-256 checksum - type str
):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# `<stem>.sidecar.json` (the convention of mrm/extraction/common.py).
# `<name>.sidecar.json` and `<stem>.json` (the name mrm/extraction/thermal_fem.artefact_paths writes) are also read.
# x
# x all these helpers can be moved somewhere else?
# helper function
def build_rom_sidecar_path(
    path: str | Path,  # path = path (1) - thermal artifact path - type str or Path
) -> (
    Path  # return = path (1) - canonical sidecar artifact path - type Path
):
    return path.with_name(path.stem + ".sidecar.json")


# find the first existing sidecar path.
# x
# helper function
def find_rom_sidecar_path(
    path: str | Path,  # path = path (1) - thermal artifact path - type str or Path
) -> (
    Path | None  # return = path (1) - existing sidecar path or no match - type Path or None
):
    # same candidates as mrm.params.config.sidecar_candidates (minus the .meta.json names)
    for candidate in (build_rom_sidecar_path(path), path.with_name(path.name + ".sidecar.json"), path.with_suffix(".json")):
        if candidate.exists():
            return candidate
    return None


# default artefact path `mrm/data/extracted/thermal_rom_<family>_<film>.npz`.
# x
# helper function
def build_thermal_rom_path(
    family: str,  # family = family (1) - heater-family artifact name - type str
    film_nm: float,  # film_nm = d_film (nm) - silicon-film thickness - type float scalar
    root: str | Path | None = None,  # root = root (1) - optional artifact directory - type str, Path, or None
) -> (
    Path  # return = path (1) - thermal ROM artifact path - type Path
):
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
    # helper function
    def __post_init__(
        self: ThermalRom,  # extracted thermal model to validate
    ) -> (
        None  # raises RomIntegrityError if a validation check fails
    ):
        source_kind_count = len(self.sources)
        observer_kind_count = len(self.observers)
        lag_count = np.asarray(self.taus_s).shape[0]

        same_ring_residues = np.asarray(self.self_res)
        expected_same_ring_shape = (source_kind_count, observer_kind_count, lag_count)
        if same_ring_residues.shape != expected_same_ring_shape:
            raise RomIntegrityError(
                f"self_res shape {same_ring_residues.shape} must be (S, O, K) = {expected_same_ring_shape}"
            )

        distance_count = np.asarray(self.cross_r_um).shape[0]
        cross_ring_residues = np.asarray(self.cross_res)
        expected_cross_ring_shape = (source_kind_count, observer_kind_count, distance_count, lag_count)
        if cross_ring_residues.shape != expected_cross_ring_shape:
            raise RomIntegrityError(
                f"cross_res shape {cross_ring_residues.shape} must be (S, O, D, K) = {expected_cross_ring_shape}"
            )

        unknown_sources = [source for source in self.sources if source not in SOURCE_KINDS]
        unknown_observers = [observer for observer in self.observers if observer not in OBSERVER_KINDS]
        if unknown_sources or unknown_observers:
            raise RomIntegrityError(
                f"unknown source/observer kinds - sources {unknown_sources}, observers {unknown_observers}"
            )

        time_constants_s = np.asarray(self.taus_s, dtype=float)
        if not np.all(time_constants_s > 0) or np.any(np.diff(time_constants_s) <= 0):
            raise RomIntegrityError("taus_s must be positive and strictly increasing")

        if distance_count:
            centre_distances_um = np.asarray(self.cross_r_um, dtype=float)
            if np.any(np.diff(centre_distances_um) <= 0):
                raise RomIntegrityError("cross_r_um must be strictly increasing")

        if not np.all(np.isfinite(same_ring_residues)):
            raise RomIntegrityError("self_res contains non-finite values")
        if not np.all(np.isfinite(cross_ring_residues)):
            raise RomIntegrityError("cross_res contains non-finite values")


# write the thermal ROM arrays and their checksum sidecar.
# keep the thermal_rom/1 field names and sidecar naming convention.
# add schema, SHA-256, and the data filename to the supplied provenance.
# x
# helper function
def save_rom(
    path: str | Path,  # path = path (1) - thermal artifact path - type str or Path
    rom: ThermalRom,  # rom = model (mixed units) - extracted poles, thermal residues, and site metadata - type ThermalRom record
    sidecar: Mapping[str, Any] | None = None,  # sidecar = metadata (mixed units) - optional artifact provenance and validation fields - type mapping or None
) -> (
    Path  # return = path (1) - written ROM artifact path - type Path
):
    path = Path(path)
    names = sorted(rom.site_offsets_um)
    offs = np.array([rom.site_offsets_um[n] for n in names], dtype=np.float64).reshape(len(names), 2)
    with open(path, "wb") as f:
        np.savez(f, taus_s=np.asarray(rom.taus_s, np.float64), sources=np.array(rom.sources, dtype="U32"),
                 observers=np.array(rom.observers, dtype="U32"), self_res=np.asarray(rom.self_res, np.float64),
                 cross_r_um=np.asarray(rom.cross_r_um, np.float64), cross_res=np.asarray(rom.cross_res, np.float64),
                 site_offset_names=np.array(names, dtype="U32"), site_offsets_um=offs)
    side = dict(sidecar or {})
    side["schema"] = SCHEMA_ROM
    side["sha256"] = sha256_file(path)
    side["data_file"] = path.name
    with open(build_rom_sidecar_path(path), "w") as f:
        json.dump(side, f, indent=2, sort_keys=True, default=str)
    return path


# load thermal ROM arrays and their sidecar metadata.
# verify=True requires the expected schema and the SHA-256 of the data file.
# warn=True reports validation limits or required time scales that are missing.
# accept the flat writer format and the nested FEM extraction format.
# x
# helper function
def load_rom(
    path: str | Path,  # path = path (1) - thermal artifact path - type str or Path
    *,
    verify: bool = True,  # verify = verify (1) - require schema and checksum verification - type bool scalar
    warn: bool = True,  # warn = warn (1) - emit warnings for missing or failed validation metrics - type bool scalar
) -> (
    ThermalRom  # return = model (mixed units) - validated poles, residues, and sidecar metadata - type ThermalRom record
):
    path = Path(path)
    side_path = find_rom_sidecar_path(path)
    meta: dict = {}
    if side_path is not None:
        meta = json.loads(side_path.read_text())
    elif verify:
        raise RomIntegrityError(f"missing sidecar {build_rom_sidecar_path(path)}")
    if verify:
        if meta.get("schema") != SCHEMA_ROM:
            raise RomIntegrityError(f"{side_path}: schema {meta.get('schema')!r} != {SCHEMA_ROM!r}")
        digest = sha256_file(path)
        if meta.get("sha256") != digest:
            raise RomIntegrityError(f"{path}: sha256 {digest} does not match the sidecar ({meta.get('sha256')})")
    with np.load(path, allow_pickle=False) as z:
        taus = np.asarray(z["taus_s"], np.float64)
        sources = tuple(str(s) for s in z["sources"])
        observers = tuple(str(s) for s in z["observers"])
        self_res = np.asarray(z["self_res"], np.float64)
        cross_r = np.asarray(z["cross_r_um"], np.float64)
        cross_res = np.asarray(z["cross_res"], np.float64)
        offsets = {}
        # names key - 'site_offset_names' (what both writers write).
        # 'site_names' is accepted as a fallback
        name_key = next((k for k in ("site_offset_names", "site_names") if k in z.files), None)
        if name_key is not None and "site_offsets_um" in z.files:
            for n, xy in zip(z[name_key], np.asarray(z["site_offsets_um"], np.float64).reshape(-1, 2)):
                offsets[str(n)] = (float(xy[0]), float(xy[1]))
    # sidecar fallbacks - a top-level site_offsets_um dict, or the extraction config (inputs.config.site_offsets_um)
    cfg_offs = ((meta.get("inputs") or {}).get("config") or {}).get("site_offsets_um") or {}
    for src in (meta.get("site_offsets_um") or {}, cfg_offs):
        offsets.update({k: tuple(map(float, v)) for k, v in src.items() if k not in offsets})
    meta = dict(meta)
    meta["path"] = str(path)
    rom = ThermalRom(taus, sources, observers, self_res, cross_r, cross_res, offsets, meta)
    if warn:
        warn_about_rom_quality(rom)
    return rom


# read step, magnitude, phase, and superposition validation errors from sidecar metadata.
# use the larger self-path or nearest-neighbour error in the nested FEM format.
# return None for a metric that is absent.
# x
# helper function
def read_rom_validation_metrics(
    meta: dict[str, Any],  # meta = metadata (mixed units) - provenance and validation fields - type dict with string keys and metadata values
) -> (
    dict[str, float | None]  # return = metrics (mixed units) - relative step error, magnitude error in dB, phase error in deg, and relative superposition error - type dict of named floats or None
):
    v = dict(meta.get("validation") or {})
    out = {k: v.get(k) for k in ("step_max_rel_err", "freq_max_db_err", "freq_max_deg_err", "superposition_rel_err")}
    nested = {"step_max_rel_err": "max_time_err_frac_of_dc", "freq_max_db_err": "max_db",
              "freq_max_deg_err": "max_deg"}
    for key, sub in nested.items():
        if out[key] is None:
            vals = [float(v[g][sub]) for g in ("self", "nearest_neighbour")
                    if isinstance(v.get(g), Mapping) and v[g].get(sub) is not None]
            out[key] = max(vals) if vals else None
    if out["superposition_rel_err"] is None:
        for st in (meta.get("stages") or {}).values():
            if isinstance(st, Mapping) and st.get("superposition_rel_err") is not None:
                out["superposition_rel_err"] = float(st["superposition_rel_err"])
    return out


# warn (RuntimeWarning) when a ROM misses an acceptance target or a required pole.
# x
# helper function
def warn_about_rom_quality(
    rom: ThermalRom,  # rom = model (mixed units) - extracted poles, thermal residues, and site metadata - type ThermalRom record
) -> (
    None  # return = None - results are written into the supplied state or record - type None
):
    v = read_rom_validation_metrics(rom.meta)
    checks = (("step_max_rel_err", ROM_STEP_ERR_MAX), ("freq_max_db_err", ROM_FREQ_DB_MAX),
              ("freq_max_deg_err", ROM_FREQ_DEG_MAX), ("superposition_rel_err", ROM_SUPERPOSITION_MAX))
    for key, lim in checks:
        val = v.get(key)
        if val is None:
            warnings.warn(f"thermal ROM {rom.meta.get('path')}: sidecar lacks validation.{key}",
                          RuntimeWarning, stacklevel=3)
        elif not (abs(float(val)) <= lim):
            warnings.warn(f"thermal ROM {rom.meta.get('path')}: validation.{key} = {val} exceeds {lim}",
                          RuntimeWarning, stacklevel=3)
    if rom.taus_s[0] > ROM_TAU_FAST_MAX_S:
        warnings.warn(f"thermal ROM has no pole below {ROM_TAU_FAST_MAX_S} s; the optical self-heating solve needs "
                      "the fast poles of the waveguide core", RuntimeWarning, stacklevel=3)
    if rom.taus_s[-1] < ROM_TAU_SLOW_MIN_S:
        warnings.warn(f"thermal ROM has no pole above {ROM_TAU_SLOW_MIN_S} s; the crosstalk needs the ms poles",
                      RuntimeWarning, stacklevel=3)


# layout, scales and the interim model specification
# site coordinates as an (n, 2) array in um (a single (2,) point becomes one row).
# x
# helper function
def validate_site_coordinates_um(
    coordinates_um: ArrayLike,  # coordinates_um = r (um) - site coordinates - type array-like, shape (2,) or (N, 2)
    expected_site_count: int | None = None,  # expected_site_count = N (1) - optional required coordinate count - type int or None
    name: str = "xy",  # name = name (1) - element name or diagnostic parameter label - type str
) -> (
    Float64Array  # return = r_sites (um) - validated site coordinates - type float64 ndarray, shape (N, 2)
):
    coordinates = np.asarray(coordinates_um, dtype=np.float64)
    if coordinates.ndim == 1 and coordinates.shape[0] == 2:
        coordinates = coordinates[None, :]
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"{name} must be (n, 2) in um, got shape {coordinates.shape}")
    if expected_site_count is not None and coordinates.shape[0] != expected_site_count:
        raise ValueError(f"{name} must have {expected_site_count} rows, got {coordinates.shape[0]}")
    return coordinates


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
    # helper function
    def __post_init__(
        self: LayoutSites,  # self = self (mixed units) - record fields and units described above the class - type LayoutSites record
    ) -> (
        None  # return = None - results are written into the supplied state or record - type None
    ):
        n = validate_site_coordinates_um(self.ring_xy_um, name="ring_xy_um").shape[0]
        object.__setattr__(self, "ring_xy_um", validate_site_coordinates_um(self.ring_xy_um, name="ring_xy_um"))
        for nm in ("pd_xy_um", "driver_xy_um"):
            v = getattr(self, nm)
            if v is not None:
                object.__setattr__(self, nm, validate_site_coordinates_um(v, n, nm))
        if self.temp_sensor_xy_um is not None:
            ts = validate_site_coordinates_um(self.temp_sensor_xy_um, name="temp_sensor_xy_um")
            object.__setattr__(self, "temp_sensor_xy_um", ts)
            ring = (np.full(ts.shape[0], -1, np.int64) if self.temp_sensor_ring is None
                    else np.asarray(self.temp_sensor_ring, np.int64).reshape(-1))
            if ring.shape[0] != ts.shape[0] or np.any(ring >= n) or np.any(ring < -1):
                raise ValueError("temp_sensor_ring must have one entry per sensor, each -1 or a ring index")
            object.__setattr__(self, "temp_sensor_ring", ring)
            names = tuple(self.temp_sensor_names) or tuple(f"ts{i}" for i in range(ts.shape[0]))
            object.__setattr__(self, "temp_sensor_names", names)
        if self.aggressor_xy_um is not None:
            ag = validate_site_coordinates_um(self.aggressor_xy_um, name="aggressor_xy_um")
            object.__setattr__(self, "aggressor_xy_um", ag)
            names = tuple(self.aggressor_names) or tuple(f"aggressor{i}" for i in range(ag.shape[0]))
            object.__setattr__(self, "aggressor_names", names)

    # number of rings N.
    # helper function
    @property
    def n_rings(
        self: LayoutSites,  # self = self (mixed units) - record fields and units described above the class - type LayoutSites record
    ) -> (
        int  # return = N (1) - number of rings - type int scalar
    ):
        return self.ring_xy_um.shape[0]

    # a straight bank - ring i at i * pitch_um along x (axis 0) or y (axis 1).
    # helper function
    @staticmethod
    def line(
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
        xy = np.zeros((int(n_rings), 2))
        xy[:, axis] = np.arange(int(n_rings)) * float(pitch_um)
        return LayoutSites(xy, pd_xy_um, driver_xy_um, temp_sensor_xy_um, temp_sensor_ring, tuple(temp_sensor_names),
                           aggressor_xy_um, tuple(aggressor_names))

    # construct ring and auxiliary-site coordinates from named layout records.
    # each site supplies coordinates in um, a kind, and an optional ring attachment.
    # infer ring indices from explicit fields or trailing digits where the site kind permits it.
    # place missing rings on a line using ring_pitch_um.
    # a driver without a ring index is shared by the rings.
    # a heater site without an explicit ring field is an aggressor.
    # referenced temperature-sensor sites are included only once.
    # sensor attachments select a ring, the bank centroid, or the assumed die location.
    # helper function
    @staticmethod
    def from_sites(
        sites: Mapping[str, Mapping[str, Any]] | None,  # sites = sites (mixed units) - named site records with x_um, y_um, kind, and optional ring - type mapping or None
        n_rings: int,  # n_rings = N (1) - number of rings - type int scalar
        ring_pitch_um: float | None = None,  # ring_pitch_um = d_pitch (um) - optional spacing for missing ring sites - type float or None
        temp_sensors: Sequence[Mapping[str, Any]] | None = None,  # temp_sensors = sensors (mixed units) - sensor site or attachment records - type sequence of mappings or None
    ) -> (
        LayoutSites  # return = sites (um) - parsed ring, driver, sensor, photodiode, and aggressor positions - type LayoutSites record
    ):
        sites = dict(sites or {})
        referenced = {str(spec["site"]) for spec in (temp_sensors or ()) if spec.get("site")}
        ring_xy = np.full((n_rings, 2), np.nan)
        pd = np.full((n_rings, 2), np.nan)
        drv = np.full((n_rings, 2), np.nan)
        shared_drv = None
        ag_xy, ag_names, ts_xy, ts_ring, ts_names = [], [], [], [], []

        # read a ring index from the site field or trailing digits.
        # helper function
        def site_ring_index(
            name: str,  # name = name (1) - element name or diagnostic parameter label - type str
            spec: Mapping[str, Any],  # spec = site (mixed units) - site record with an optional ring field - type mapping
        ) -> (
            int | None  # return = k (1) - ring index from the field or trailing digits - type int or None
        ):
            if "ring" in spec and spec["ring"] is not None:
                return int(spec["ring"])
            digits = ""
            for ch in reversed(str(name)):
                if ch.isdigit():
                    digits = ch + digits
                else:
                    break
            return int(digits) if digits else None

        for name, spec in sites.items():
            kind = spec.get("kind", "other")
            p = (float(spec["x_um"]), float(spec["y_um"]))
            i = site_ring_index(name, spec)
            if kind == "ring":
                if i is None or not 0 <= i < n_rings:
                    raise ValueError(f"ring site {name!r} needs a ring index 0..{n_rings - 1}")
                ring_xy[i] = p
            elif kind == "monitor_pd":
                if i is None or not 0 <= i < n_rings:
                    raise ValueError(f"monitor_pd site {name!r} needs a ring index")
                pd[i] = p
            elif kind == "driver":
                if i is None:
                    shared_drv = p
                elif 0 <= i < n_rings:
                    drv[i] = p
            elif kind == "aggressor" or (kind == "heater" and (i is None or "ring" not in spec)):
                ag_xy.append(p)
                ag_names.append(str(name))
            elif kind == "temp_sensor" and str(name) not in referenced:
                ts_xy.append(p)
                ts_ring.append(-1)
                ts_names.append(str(name))
        missing = np.isnan(ring_xy[:, 0])
        if np.any(missing):
            if ring_pitch_um is None and n_rings > 1:
                raise ValueError(f"layout has no site for rings {np.flatnonzero(missing).tolist()} and no "
                                 "layout.ring_pitch_um")
            pitch = 0.0 if ring_pitch_um is None else float(ring_pitch_um)
            ring_xy[missing, 0] = np.flatnonzero(missing) * pitch
            ring_xy[missing, 1] = 0.0
        if shared_drv is not None:
            drv[np.isnan(drv[:, 0])] = shared_drv
        centroid = ring_xy.mean(axis=0)
        for k, spec in enumerate(temp_sensors or ()):
            name = str(spec.get("name", f"ts{k}"))
            if spec.get("site"):
                s = sites[spec["site"]]
                ts_xy.append((float(s["x_um"]), float(s["y_um"])))
                ts_ring.append(-1)
            else:
                att = str(spec.get("attach", "die"))
                if att.startswith("ring_"):
                    i = int(att.split("_", 1)[1])
                    ts_xy.append(tuple(ring_xy[i]))
                    ts_ring.append(i)
                elif att == "bank":
                    ts_xy.append(tuple(centroid))
                    ts_ring.append(-1)
                else:   # die - assumed 1 mm from the bank centroid
                    ts_xy.append((centroid[0], centroid[1] + 1000.0))
                    ts_ring.append(-1)
            ts_names.append(name)
        return LayoutSites(          # NaN rows - PD at the model's default offset, no driver for that ring
            ring_xy,
            None if np.all(np.isnan(pd[:, 0])) else pd,
            None if np.all(np.isnan(drv[:, 0])) else drv,
            np.array(ts_xy, float).reshape(-1, 2) if ts_xy else None,
            np.array(ts_ring, np.int64) if ts_xy else None, tuple(ts_names),
            np.array(ag_xy, float).reshape(-1, 2) if ag_xy else None, tuple(ag_names))


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
    # helper function
    @staticmethod
    def textbook_corner(
        **kw: Any,  # kw = overrides (field units) - InterimRom field overrides - type keyword values with InterimRom field types
    ) -> (
        InterimRom  # return = model (mixed units) - interim parameters for the textbook corner - type InterimRom record
    ):
        base = dict(tau_fast_s=10e-6, share_fast_frac=0.75, tau_slow_s=300e-6, tail_share_frac=0.0,
                    xt_nearest_frac=0.10, xt_next_frac=0.03, xt_floor_frac=0.0, crosstalk_dynamics="self",
                    name="interim_textbook_corner")
        base.update(kw)
        return InterimRom(**base)


# calculate a crosstalk time constant from source-to-victim distance.
# the interim model subtracts the source and observer radii from centre spacing.
# the near-field law is max(tau_min,tau_ref*(d/d_ref)^2).
# the capped law takes the smaller near-field and far-field times, then applies tau_min.
# the literal law switches at far_um and can be nonmonotone.
# tau_far_s=None uses the measured distance table.
# x
def calculate_crosstalk_time_constant_s(
    d_um: ArrayLike,  # d_um = d (um) - source-to-victim distance or site centre distance - type array-like, broadcast input shape
    tau_ref_s: ArrayLike = 4.1e-3,  # tau_ref_s = tau_ref (s) - reference or minimum crosstalk time constant - type array-like, broadcast input shape
    dist_ref_um: float = 60.0,  # dist_ref_um = d_ref (um) - reference distance for the diffusion law - type float scalar
    tau_min_s: ArrayLike = 1.0e-4,  # tau_min_s = tau_min (s) - reference or minimum crosstalk time constant - type array-like, broadcast input shape
    tau_far_s: ArrayLike | None = None,  # tau_far_s = tau_far (s) - optional far-field time constant, or the extracted distance table - type array-like or None
    far_um: float = 100.0,  # far_um = d_far (um) - near-field cutoff for the literal distance law - type float scalar
    law: str = "capped",  # law = law (1) - capped or literal distance law - type str
) -> (
    RealValue  # return = tau_cross (s) - crosstalk time constant at each distance - type real scalar or float64 ndarray, shape broadcast input shape
):
    d = np.asarray(d_um, dtype=np.float64)
    # the near-field diffusion law makes twice the reference gap take four times as long, before the limits apply.
    near = np.maximum(np.asarray(tau_min_s, float), np.asarray(tau_ref_s, float) * (d / float(dist_ref_um)) ** 2)
    if tau_far_s is None:
        far = np.interp(d, XT_TABLE_D_UM, XT_TABLE_TAU_S)
    else:
        far = np.broadcast_to(np.asarray(tau_far_s, float), np.broadcast(d, np.asarray(tau_far_s, float)).shape)
    # the literal law switches at d_far. the capped law limits the near-field time to the far-field value.
    if law == "literal":
        return np.where(d <= float(far_um), near, far)
    if law != "capped":
        raise ValueError(f"crosstalk tau law must be 'capped' or 'literal', got {law!r}")
    return np.maximum(np.minimum(near, far), np.asarray(tau_min_s, float))


# estimate a site-path DC gain relative to the heater-to-ring gain.
# c(d)=clip(nearest_frac*ref_um/d,floor_frac,max_frac).
# use centre distance for this assumed law.
# the nonzero floor represents the long-range die or package term.
# x
def calculate_site_coupling_fraction(
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


# the assembled system
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
    # helper function
    @property
    def batch(
        self: ThermalSystem,  # self = self (mixed units) - record fields and units described above the class - type ThermalSystem record
    ) -> (
        int  # return = B (1) - number of worlds - type int scalar
    ):
        return self.taus_s.shape[0]

    # number of lags M.
    # helper function
    @property
    def n_lags(
        self: ThermalSystem,  # self = self (mixed units) - record fields and units described above the class - type ThermalSystem record
    ) -> (
        int  # return = M (1) - number of thermal lag states - type int scalar
    ):
        return self.taus_s.shape[1]

    # number of sources S.
    # helper function
    @property
    def n_src(
        self: ThermalSystem,  # self = self (mixed units) - record fields and units described above the class - type ThermalSystem record
    ) -> (
        int  # return = S (1) - number of heat sources - type int scalar
    ):
        return self.src_kind.shape[0]

    # number of observers O.
    # helper function
    @property
    def n_obs(
        self: ThermalSystem,  # self = self (mixed units) - record fields and units described above the class - type ThermalSystem record
    ) -> (
        int  # return = O (1) - number of temperature observers - type int scalar
    ):
        return self.obs_kind.shape[0]

    # exact FOH coefficients (a, c0, c1), each (B, M), for a step h_s (s).
    # see foh_coefficients.
    # helper function
    def coefficients(
        self: ThermalSystem,  # self = self (mixed units) - record fields and units described above the class - type ThermalSystem record
        h_s: float,  # h_s = h (s) - thermal step duration - type float scalar
    ) -> tuple[
        Float64Array,  # return[0] = a (1) - retained state fractions - type float64 ndarray, shape (B, M)
        Float64Array,  # return[1] = c_start (K/W) - starting-power gains - type float64 ndarray, shape (B, M)
        Float64Array,  # return[2] = c_end (K/W) - ending-power gains - type float64 ndarray, shape (B, M)
    ]:
        return calculate_linear_hold_coefficients(self.taus_s, self.res_k_per_w, h_s)

    # all-zero lag states (B, M) (every observer at T_base).
    # helper function
    def zero_state(
        self: ThermalSystem,  # self = self (mixed units) - record fields and units described above the class - type ThermalSystem record
        batch: int | None = None,  # batch = B (1) - optional number of worlds - type int or None
    ) -> (
        Float64Array  # return = x (K) - zero initial temperature rises - type float64 ndarray, shape (B, M)
    ):
        return np.zeros((self.batch if batch is None else int(batch), self.n_lags))

    # lag states (B, M) in steady state for constant source powers u_w (B|1, S) W - x_m = r_m u_src(m).
    def steady_state(
        self: ThermalSystem,  # self = self (mixed units) - record fields and units described above the class - type ThermalSystem record
        u_w: ArrayLike,  # u_w = P_source (W) - constant source powers - type array-like, shape (S,) or (B, S)
    ) -> (
        Float64Array  # return = x_ss (K) - steady lag-state temperature rises - type float64 ndarray, shape (B, M)
    ):
        u = np.atleast_2d(np.asarray(u_w, dtype=np.float64))
        # dx_m/dt = 0 gives x_m = r_m*P_source. lag_src selects that power for each lag.
        return self.res_k_per_w * u[:, self.lag_src]

    # observer temperatures (B, O) = t_base_k + sum of lag states (Python form of thermal_outputs).
    # helper function
    def outputs(
        self: ThermalSystem,  # self = self (mixed units) - record fields and units described above the class - type ThermalSystem record
        x: ArrayLike,  # x = x (K) - lag-state temperature rises - type array-like, shape (B, M)
        t_base_k: ArrayLike = 0.0,  # t_base_k = T_base (K) - base temperature of each world - type array-like, scalar or shape (1 or B,)
    ) -> (
        Float64Array  # return = T_observer (K) - absolute observer temperatures - type float64 ndarray, shape (B, O)
    ):
        x = np.atleast_2d(x)
        out = np.zeros((x.shape[0], self.n_obs))
        write_observer_temperatures(x, self.obs_ptr, np.atleast_1d(np.asarray(t_base_k, np.float64)), out)
        return out

    # index of a source by kind and ring (or by name for aggressors).
    # helper function
    def source_index(
        self: ThermalSystem,  # self = self (mixed units) - record fields and units described above the class - type ThermalSystem record
        kind: str,  # kind = kind (1) - source or observer kind name - type str
        ring: int | None = None,  # ring = k (1) - physical ring index or no ring filter - type int or None
        name: str | None = None,  # name = name (1) - optional exact element name - type str or None
    ) -> (
        int  # return = s (1) - matching heat-source index - type int scalar
    ):
        return find_thermal_element_index(self.src_names, self.src_kind, self.src_ring, SOURCE_KINDS, kind, ring, name)

    # index of an observer by kind and ring (or by name for temperature sensors).
    # helper function
    def observer_index(
        self: ThermalSystem,  # self = self (mixed units) - record fields and units described above the class - type ThermalSystem record
        kind: str,  # kind = kind (1) - source or observer kind name - type str
        ring: int | None = None,  # ring = k (1) - physical ring index or no ring filter - type int or None
        name: str | None = None,  # name = name (1) - optional exact element name - type str or None
    ) -> (
        int  # return = o (1) - matching temperature-observer index - type int scalar
    ):
        return find_thermal_element_index(self.obs_names, self.obs_kind, self.obs_ring, OBSERVER_KINDS, kind, ring, name)

    # (lag_src, lag_obs, lag_kind, opt_lags, obs_ptr, src_ring, ring_obs) - the index arrays of selfheat.implicit_step, in its argument order.
    # helper function
    def kernel_arrays(
        self: ThermalSystem,  # self = self (mixed units) - record fields and units described above the class - type ThermalSystem record
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



# index of the one element with this name, or with this kind and ring.
# raises KeyError otherwise.
# x
# helper function
def find_thermal_element_index(
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
    if name is not None:
        if name in names:
            return names.index(name)
        raise KeyError(f"no element named {name!r}")
    k = kind_list.index(kind)
    hit = np.flatnonzero((kinds == k) & ((rings == ring) if ring is not None else True))
    if hit.size != 1:
        raise KeyError(f"{kind} ring={ring}: {hit.size} matches")
    return int(hit[0])


# assembly
# a scalar, (B,) or (B, 1) parameter as a 1-D array of length 1 or B.
# x
# helper function
def normalize_world_parameter_vector(
    values: ArrayLike,  # values = values (input units) - parameter values to normalize - type array-like, scalar, (B,), or (B, 1)
    name: str,  # name = name (1) - element name or diagnostic parameter label - type str
) -> (
    Float64Array  # return = parameter (input units) - normalized world-parameter vector - type float64 ndarray, shape (1 or B,)
):
    parameter_array = np.asarray(values, dtype=np.float64)
    if parameter_array.ndim == 0:
        return parameter_array.reshape(1)
    if parameter_array.ndim == 2 and parameter_array.shape[1] == 1:
        return parameter_array[:, 0]
    if parameter_array.ndim != 1:
        raise ValueError(f"{name}: expected a scalar or (B,), got shape {parameter_array.shape}")
    return parameter_array


# the common batch length B of 1-row or B-row arrays.
# raises ValueError for two different lengths.
# x
# helper function
def infer_world_count(
    values: dict[str, Float64Array],  # values = parameters (parameter units) - world-parameter arrays keyed by parameter name - type dict of float64 ndarrays with 1 or B rows
) -> (
    int  # return = B (1) - common world count inferred from parameter arrays - type int scalar
):
    world_count = 1
    for name, parameter_array in values.items():
        row_count = parameter_array.shape[0]
        if row_count != 1:
            if world_count not in (1, row_count):
                raise ValueError(f"inconsistent batch sizes: {name} has {row_count}, another has {world_count}")
            world_count = row_count
    return world_count


# scaling_hypothesis (a name, names or int codes) as B int codes into SCALING_HYPOTHESES.
# x
# (todo) what is this function??
# helper function
def encode_thermal_scaling_hypotheses(
    hypotheses: str | Sequence[str] | ArrayLike,  # hypotheses = hypotheses (1) - scaling hypothesis names or integer codes - type str, sequence of strings, or integer array-like
    world_count: int,  # world_count = B (1) - number of worlds - type int scalar
) -> (
    Int64Array  # return = hypothesis (1) - integer scaling-hypothesis codes - type int64 ndarray, shape (B,)
):
    if isinstance(hypotheses, str):
        hypotheses = [hypotheses]
    hypothesis_values = np.asarray(hypotheses)
    if hypothesis_values.dtype.kind in "iu":
        codes = hypothesis_values.astype(np.int64).reshape(-1)
    else:
        try:
            codes = np.array([SCALING_HYPOTHESES.index(str(x)) for x in hypothesis_values.reshape(-1)], np.int64)
        except ValueError as e:
            raise ValueError(f"scaling_hypothesis must be one of {SCALING_HYPOTHESES}, got {hypotheses!r}") from e
    if np.any((codes < 0) | (codes >= len(SCALING_HYPOTHESES))):
        raise ValueError(f"scaling_hypothesis codes out of range: {codes}")
    if codes.shape[0] not in (1, world_count):
        raise ValueError(f"scaling_hypothesis has {codes.shape[0]} entries, batch is {world_count}")
    return np.broadcast_to(codes, (world_count,)).copy()


# collect source-to-observer lag terms before sorting and scaling.
# time_constants_s and residues_k_per_w store per-world values in s and K/W.
# source_indices, observer_indices, layer_codes, and pair_codes are dimensionless indices.
# x
class _Lags:

    # initialize the builder without changing the physical model.
    # helper function
    def __init__(
        self: _Lags,  # self = self (mixed units) - record fields and units described above the class - type _Lags record
        B: int,  # B = B (1) - number of worlds - type int scalar
    ) -> (
        None  # return = None - results are written into the supplied state or record - type None
    ):
        self.world_count = B
        self.observer_indices, self.source_indices, self.time_constants_s, self.residues_k_per_w = [], [], [], []
        self.layer_codes, self.pair_codes, self.tail_replaceable, self.tail_flags, self.fixed_time_constants = [], [], [], [], []

    # append a lag with its source, observer, time constant, and thermal residue.
    # helper function
    def add(
        self: _Lags,  # self = self (mixed units) - record fields and units described above the class - type _Lags record
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
        None  # return = None - results are written into the supplied state or record - type None
    ):
        self.observer_indices.append(obs)
        self.source_indices.append(src)
        self.time_constants_s.append(np.broadcast_to(np.asarray(tau, float), (self.world_count,)).copy())
        self.residues_k_per_w.append(np.broadcast_to(np.asarray(res, float), (self.world_count,)).copy())
        self.layer_codes.append(layer)
        self.pair_codes.append(pair)
        self.tail_replaceable.append(bool(tailable))
        self.tail_flags.append(bool(tail))
        self.fixed_time_constants.append(bool(fixed_tau))    # absolute tau - no time_scale, no 'resistance' factor


# source and observer instances of the bank with their positions (um) and ring attachment.
# x
class _Elements:

    # initialize the builder without changing the physical model.
    # helper function
    def __init__(
        self: _Elements,  # self = self (mixed units) - record fields and units described above the class - type _Elements record
    ) -> (
        None  # return = None - results are written into the supplied state or record - type None
    ):
        self.src_kind, self.src_ring, self.src_names, self.src_xy = [], [], [], []
        self.obs_kind, self.obs_ring, self.obs_names, self.obs_xy = [], [], [], []

    # append a heat source and return its source index.
    # helper function
    def add_src(
        self: _Elements,  # self = self (mixed units) - record fields and units described above the class - type _Elements record
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
        return len(self.src_kind) - 1

    # append a temperature observer and return its observer index.
    # helper function
    def add_obs(
        self: _Elements,  # self = self (mixed units) - record fields and units described above the class - type _Elements record
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
        return len(self.obs_kind) - 1


# build the bank's heat sources, temperature observers, and index maps.
# ring sources and ring or heater observers use ring-centre coordinates.
# photodiodes and attached sensors use their specified or default offsets.
# include a driver only where a site or default driver offset exists.
# helper function
def build_thermal_sources_and_observers(
    layout: LayoutSites,  # layout = sites (um) - source and observer site coordinates - type LayoutSites record
    n: int,  # n = N (1) - number of rings - type int scalar
    src_kinds: Sequence[str],  # src_kinds = source_kinds (1) - source kind names to include - type sequence of strings
    obs_kinds: Sequence[str],  # obs_kinds = observer_kinds (1) - observer kind names to include - type sequence of strings
    pd_offset: Sequence[float],  # pd_offset = Delta_r_PD (um) - default monitor offset from its ring centre - type two coordinates
    ts_offset: Sequence[float],  # ts_offset = Delta_r_sensor (um) - default attached-sensor offset - type two coordinates
    driver_offset: Sequence[float] | None = None,  # driver_offset = Delta_r_driver (um) - optional driver offset from its ring centre - type two coordinates or None
) -> tuple[
    _Elements,  # return[0] = elements (mixed units) - source and observer kinds, names, coordinates, and ring attachments - type thermal element builder
    dict[str, Int64Array],  # return[1] = indices (1) - per-ring, temperature-sensor, and aggressor index maps - type dict of named int64 ndarrays
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

    for ring_index in range(n):
        ring_centre_um = ring_centres_um[ring_index]
        for kind_name, kind_code, index_name in centre_sources:
            if kind_name in src_kinds:
                indices[index_name][ring_index] = elements.add_src(
                    kind_code, ring_index, f"{kind_name}{ring_index}", ring_centre_um
                )

        if "driver" not in src_kinds:
            continue
        # use the layout position when a driver table exists. a nonfinite row marks an absent driver.
        if layout.driver_xy_um is not None:
            driver_position_um = layout.driver_xy_um[ring_index]
        elif driver_offset is not None:
            driver_position_um = ring_centre_um + np.asarray(driver_offset, float)
        else:
            continue
        if np.all(np.isfinite(driver_position_um)):
            indices["driver_src"][ring_index] = elements.add_src(
                SRC_DRIVER, ring_index, f"driver{ring_index}", driver_position_um
            )

    # aggressors use their layout positions with ring attachment -1 for a free site.
    aggressor_indices = []
    if "aggressor" in src_kinds and layout.aggressor_xy_um is not None:
        for name, position_um in zip(layout.aggressor_names, layout.aggressor_xy_um):
            aggressor_indices.append(elements.add_src(SRC_AGGRESSOR, -1, name, position_um))

    for ring_index in range(n):
        ring_centre_um = ring_centres_um[ring_index]
        for kind_name, kind_code, index_name in centre_observers:
            if kind_name in obs_kinds:
                indices[index_name][ring_index] = elements.add_obs(
                    kind_code, ring_index, f"{kind_name}{ring_index}", ring_centre_um
                )

        if "monitor_pd" in obs_kinds:
            # use the model's offset when the layout has no finite monitor position.
            monitor_position_um = None
            if layout.pd_xy_um is not None:
                monitor_position_um = layout.pd_xy_um[ring_index]
            if monitor_position_um is None or not np.all(np.isfinite(monitor_position_um)):
                monitor_position_um = ring_centre_um + np.asarray(pd_offset, float)
            indices["pd_obs"][ring_index] = elements.add_obs(
                OBS_PD, ring_index, f"monitor_pd{ring_index}", monitor_position_um
            )

    sensor_indices = []
    if "temp_sensor" in obs_kinds and layout.temp_sensor_xy_um is not None:
        for sensor_index, name in enumerate(layout.temp_sensor_names):
            attached_ring = int(layout.temp_sensor_ring[sensor_index])
            # attached sensors use the offset of the ROM self table. free sensors use layout coordinates.
            if attached_ring >= 0:
                sensor_position_um = ring_centres_um[attached_ring] + np.asarray(ts_offset, float)
            else:
                sensor_position_um = layout.temp_sensor_xy_um[sensor_index]
            sensor_indices.append(elements.add_obs(OBS_TEMP, attached_ring, name, sensor_position_um))

    indices["temp_obs"] = np.array(sensor_indices, np.int64)
    indices["aggressor_src"] = np.array(aggressor_indices, np.int64)
    return (
        elements,
        indices,
    )


# rank ring pairs by their first and second distinct neighbour distances.
# (todo) helper func, move it out
# x
def calculate_neighbor_ranks_and_distances(
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
    for ring_index in range(ring_count):
        other_rings = ring_indices[ring_indices != ring_index]
        distances_to_other_rings_um = centre_distances_um[ring_index, other_rings]

        # sort the distances and group nearly equal spacings under the first distance in each group.
        distinct_distances_um = []
        for distance_um in np.sort(distances_to_other_rings_um):
            if not distinct_distances_um:
                distinct_distances_um.append(distance_um)
                continue
            last_distance_um = distinct_distances_um[-1]
            group_limit_um = last_distance_um * (1 + relative_distance_tolerance) + absolute_distance_tolerance_um
            if distance_um > group_limit_um:
                distinct_distances_um.append(distance_um)

        for neighbor_rank, group_distance_um in enumerate(distinct_distances_um[:2], start=1):
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


# how much does one source warm an observer on each time scale?
# each path is a sum of lags. a power step P (W) gives each lag a rise P*r*(1-exp(-t/tau)) in K.
# r (K/W) sets the eventual rise per watt. tau (s) sets how fast that rise develops.
# choose the gains and times for a same-ring path, a cross-ring path, or a path involving a separate site.
# each parameter array has one row per world and one column per ring.
def append_interim_heat_path_lags(
    spec: InterimRom,  # spec = model (mixed units) - interim thermal resistance and time-constant parameters - type InterimRom record
    elements: _Elements,  # elements = elements (mixed units) - source and observer instances with ring attachments and coordinates - type thermal element builder
    layout: LayoutSites,  # layout = sites (um) - source and observer site coordinates - type LayoutSites record
    ring_count: int,  # ring_count = N (1) - number of rings - type int scalar
    world_count: int,  # world_count = B (1) - number of worlds - type int scalar
    parameter_arrays: dict[str, Float64Array],  # parameter_arrays = parameters (field units) - broadcast interim parameters in their named physical units - type dict of float64 ndarrays, shape (B, N)
    lag_builder: _Lags,  # lag_builder = lags (mixed units) - lag indices, time constants, and thermal residues - type thermal lag builder
) -> (
    None  # return = None - results are written into the supplied state or record - type None
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
    neighbor_ranks, ring_distances_um = calculate_neighbor_ranks_and_distances(layout.ring_xy_um)

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

    # compute the gap used by the diffusion law - d_gap = max(d_centres - R_source - R_observer, 0).
    def calculate_path_time_constant_s(
        centre_distance_um: ArrayLike,  # d_centres (um) - separation between element centres
        source_radius_um: ArrayLike,  # R_source (um) - source radius or zero for a point site
        observer_radius_um: ArrayLike,  # R_observer (um) - observer radius or zero for a point site
    ) -> (
        Float64Array  # tau_cross (s) - one crosstalk time constant per world - shape (B,)
    ):
        # the IMEC crosstalk law uses the gap between elements. clamp overlapping elements to zero gap.
        gap_um = np.maximum(centre_distance_um - source_radius_um - observer_radius_um, 0.0)
        return calculate_crosstalk_time_constant_s(
            gap_um,
            parameter_arrays["xt_tau_ref"][:, 0],
            spec.xt_dist_ref_um,
            parameter_arrays["xt_tau_min"][:, 0],
            None if spec.xt_tau_far_s is None else parameter_arrays["xt_tau_far"][:, 0],
            spec.xt_far_um,
            spec.xt_tau_law,
        )

    # distribute a path gain Z_path across the reference ring's three heater lags.
    def append_heater_response_shape(
        observer_index: int,  # o (1) - observer index
        source_index: int,  # s (1) - source index
        thermal_gain_k_per_w: ArrayLike,  # Z_0 (K/W) - dc gain per world - shape (B,)
        ring_index: int,  # i (1) - ring whose time constants and shares set the response
        pair_type: int,  # pair (1) - heat path type code
    ) -> (
        None  # append the lags to lag_builder
    ):
        for time_constants_s, gain_share, layer, is_package_tail in heater_response_lags:
            # r_m = Z_path*f_m so sum(r_m) gives the requested path gain.
            lag_builder.add(
                observer_index,
                source_index,
                time_constants_s[:, ring_index],
                thermal_gain_k_per_w * gain_share[:, ring_index],
                layer,
                pair_type,
                tailable=True,
                tail=is_package_tail,
            )

    ring_source_kinds = (SRC_HEATER, SRC_WG, SRC_JUNCTION)
    ring_observer_kinds = (OBS_RING, OBS_HEATER)
    for source_index, source_kind in enumerate(elements.src_kind):
        source_ring = elements.src_ring[source_index]
        source_xy_um = elements.src_xy[source_index]
        source_is_ring_element = source_kind in ring_source_kinds

        for observer_index, observer_kind in enumerate(elements.obs_kind):
            observer_ring = elements.obs_ring[observer_index]
            observer_xy_um = elements.obs_xy[observer_index]
            observer_is_ring_element = observer_kind in ring_observer_kinds

            if source_ring < 0 or observer_ring < 0:
                pair_type = PAIR_SITE
            elif source_ring == observer_ring:
                pair_type = PAIR_SELF
            else:
                pair_type = PAIR_CROSS

            if pair_type == PAIR_SELF and source_is_ring_element and observer_is_ring_element:
                # Z_HR (K/W) is the settled heater-to-ring rise per watt on this ring.
                heater_gain_k_per_w = heater_ring_dc_gain_k_per_w[:, source_ring]
                if source_kind in (SRC_WG, SRC_JUNCTION) and observer_kind == OBS_RING:
                    # waveguide-core heating gets an extra fast lag. the rest follows the heater response.
                    core_share = optical_core_share[:, source_ring]
                    gain_ratio = optical_to_heater_gain_ratio[:, source_ring]
                    # optical or junction power has total gain z*Z_HR. assign its fraction f_core to the core lag.
                    core_gain_k_per_w = gain_ratio * core_share * heater_gain_k_per_w
                    lag_builder.add(
                        observer_index,
                        source_index,
                        parameter_arrays["selfheat_tau"][:, source_ring],
                        core_gain_k_per_w,
                        LAYER_FAST,
                        pair_type,
                    )
                    # distribute the remaining gain z*Z_HR*(1-f_core) over the three heater time scales.
                    remaining_gain_k_per_w = gain_ratio * (1.0 - core_share) * heater_gain_k_per_w
                    append_heater_response_shape(
                        observer_index, source_index, remaining_gain_k_per_w, source_ring, pair_type
                    )
                else:
                    # use Z_HR for heater-to-ring and for the reciprocal ring-to-heater path.
                    append_heater_response_shape(
                        observer_index, source_index, heater_gain_k_per_w, source_ring, pair_type
                    )
                    if source_kind == SRC_HEATER and observer_kind == OBS_HEATER:
                        # add (h-1)*Z_HR to the heater's own path so the total is h*Z_HR.
                        extra_gain_k_per_w = (heater_self_gain_ratio[:, source_ring] - 1.0) * heater_gain_k_per_w
                        lag_builder.add(
                            observer_index,
                            source_index,
                            parameter_arrays["heater_node_tau"][:, source_ring],
                            extra_gain_k_per_w,
                            LAYER_FAST,
                            pair_type,
                        )
                continue

            if pair_type == PAIR_CROSS and source_is_ring_element and observer_is_ring_element:
                # choose the coupling fraction from the first, second, or farther neighbour band.
                neighbor_rank = neighbor_ranks[observer_ring, source_ring]
                if neighbor_rank == 1:
                    coupling_fraction = parameter_arrays["xt_nearest"][:, 0]
                elif neighbor_rank == 2 and spec.xt_next_frac is not None:
                    coupling_fraction = parameter_arrays["xt_next"][:, 0]
                else:
                    coupling_fraction = parameter_arrays["xt_floor"][:, 0]

                source_heater_gain = heater_ring_dc_gain_k_per_w[:, source_ring]
                observer_heater_gain = heater_ring_dc_gain_k_per_w[:, observer_ring]
                # heater-to-ring and its reciprocal use the heater's ring for the reference gain.
                if source_kind == SRC_HEATER and observer_kind == OBS_RING:
                    reference_gain_k_per_w = source_heater_gain
                elif source_kind in (SRC_WG, SRC_JUNCTION) and observer_kind == OBS_HEATER:
                    reference_gain_k_per_w = observer_heater_gain
                else:
                    # use sqrt(Z_source*Z_observer) so reversing the rings gives the same reference.
                    reference_gain_k_per_w = np.sqrt(observer_heater_gain * source_heater_gain)

                # Z_cross = c_neighbor*Z_ref. a coupling fraction of 0.03 gives 3 percent of the reference gain.
                cross_gain_k_per_w = coupling_fraction * reference_gain_k_per_w
                # split Z_cross over the source ring's heater lags or assign it to one gap-based lag.
                if spec.crosstalk_dynamics == "self":
                    append_heater_response_shape(
                        observer_index, source_index, cross_gain_k_per_w, source_ring, pair_type
                    )
                else:
                    cross_time_constant_s = calculate_path_time_constant_s(
                        ring_distances_um[observer_ring, source_ring],
                        ring_radii_um[:, source_ring],
                        ring_radii_um[:, observer_ring],
                    )
                    lag_builder.add(
                        observer_index, source_index, cross_time_constant_s, cross_gain_k_per_w,
                        LAYER_SLOW, pair_type,
                    )
                continue

            # paths involving monitors, drivers, sensors, or aggressors use the site coupling law.
            # choose the source ring as the gain reference when present, then the observer ring, then the bank mean.
            if source_ring >= 0:
                reference_gain_k_per_w = heater_ring_dc_gain_k_per_w[:, source_ring]
            elif observer_ring >= 0:
                reference_gain_k_per_w = heater_ring_dc_gain_k_per_w[:, observer_ring]
            else:
                reference_gain_k_per_w = mean_heater_ring_dc_gain_k_per_w

            # use centre distance for site gain and edge separation for the lag time constant.
            site_distance_um = float(np.linalg.norm(source_xy_um - observer_xy_um))
            source_radius_um = ring_radii_um[:, source_ring] if source_ring >= 0 and source_is_ring_element else 0.0
            observer_radius_um = ring_radii_um[:, observer_ring] if observer_ring >= 0 and observer_is_ring_element else 0.0
            site_time_constant_s = calculate_path_time_constant_s(
                site_distance_um, source_radius_um, observer_radius_um
            )
            # c_site falls as 1/d_centres between its bounds. the site lag gain is c_site*Z_ref.
            site_coupling_fraction = calculate_site_coupling_fraction(
                site_distance_um,
                parameter_arrays["xt_nearest"][:, 0],
                parameter_arrays["site_floor"][:, 0],
                spec.site_ref_um,
                spec.site_max_frac,
            )
            lag_builder.add(
                observer_index,
                source_index,
                site_time_constant_s,
                site_coupling_fraction * reference_gain_k_per_w,
                LAYER_SLOW,
                pair_type,
            )


# append extracted lags from the same-ring table or interpolate from the distance table.
# the same-ring table applies when both sites match their extracted offsets.
def append_extracted_heat_path_lags(
    rom: ThermalRom,  # rom = model (mixed units) - extracted poles, thermal residues, and site metadata - type ThermalRom record
    elements: _Elements,  # elements = elements (mixed units) - source and observer instances with ring attachments and coordinates - type thermal element builder
    layout: LayoutSites,  # layout = sites (um) - source and observer site coordinates - type LayoutSites record
    world_count: int,  # world_count = B (1) - number of worlds - type int scalar
    tau_split: float,  # tau_split = tau_split (s) - boundary between fast and slow time scales - type float scalar
    lag_builder: _Lags,  # lag_builder = lags (mixed units) - lag indices, time constants, and thermal residues - type thermal lag builder
    meta: dict[str, Any],  # meta = metadata (mixed units) - provenance and validation fields - type dict with string keys and metadata values
    tol_um: float = 1.0,  # tol_um = d_tol (um) - allowed site-offset deviation from the ROM self table - type float scalar
) -> (
    None  # return = None - results are written into the supplied state or record - type None
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

    # helper function
    def site_matches_rom_offset(
        site_name: str | None,  # site name used to look up its extracted offset
        site_xy_um: ArrayLike,  # r_site (um) - site coordinates - shape (2,)
        ring_index: int,  # i (1) - ring that carries the site
    ) -> (
        bool  # whether the same-ring table can use this site
    ):
        if site_name is None or site_name not in rom.site_offsets_um:
            # keep the same-ring table when no extracted offset is available.
            return True
        ring_centre_um = layout.ring_xy_um[ring_index]
        extracted_offset_um = np.asarray(rom.site_offsets_um[site_name], float)
        # compare the site's position relative to its ring with the position used in the extraction.
        actual_offset_um = site_xy_um - ring_centre_um
        offset_error_um = np.linalg.norm(actual_offset_um - extracted_offset_um)
        if offset_error_um > tol_um:
            moved_site_names.add(site_name)
            return False
        return True

    def interpolate_cross_residues(
        rom_source_index: int,  # s (1) - ROM source index
        rom_observer_index: int,  # o (1) - ROM observer index
        distance_um: float,  # d (um) - distance between the actual element centres
    ) -> (
        Float64Array  # r (K/W) - interpolated thermal residues - shape (K,)
    ):
        if distance_grid_um.size == 0:
            return np.zeros_like(time_constants_s)
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

    for source_index, source_kind in enumerate(elements.src_kind):
        if source_kind not in rom_source_indices:
            skipped_kind_names.add(SOURCE_KINDS[source_kind])
            continue
        rom_source_index = rom_source_indices[source_kind]
        source_ring = elements.src_ring[source_index]
        source_xy_um = elements.src_xy[source_index]

        for observer_index, observer_kind in enumerate(elements.obs_kind):
            if observer_kind not in rom_observer_indices:
                skipped_kind_names.add(OBSERVER_KINDS[observer_kind])
                continue
            rom_observer_index = rom_observer_indices[observer_kind]
            observer_ring = elements.obs_ring[observer_index]
            observer_xy_um = elements.obs_xy[observer_index]

            both_on_rings = source_ring >= 0 and observer_ring >= 0
            same_ring = both_on_rings and source_ring == observer_ring
            use_same_ring_table = (
                same_ring
                and site_matches_rom_offset(source_offset_names.get(source_kind), source_xy_um, source_ring)
                and site_matches_rom_offset(observer_offset_names.get(observer_kind), observer_xy_um, observer_ring)
            )
            if use_same_ring_table:
                residues_k_per_w = np.asarray(rom.self_res[rom_source_index, rom_observer_index], float)
                pair_type = PAIR_SELF
            else:
                # look up the actual source-to-observer separation when the same-ring table does not apply.
                # a monitor at 50 um with a saved offset of 20 um has a 30 um mismatch.
                # for a heater at the ring centre, the lookup distance in that case is 50 um.
                site_distance_um = float(np.linalg.norm(source_xy_um - observer_xy_um))
                residues_k_per_w = interpolate_cross_residues(
                    rom_source_index, rom_observer_index, site_distance_um
                )
                pair_type = PAIR_CROSS if both_on_rings and not same_ring else PAIR_SITE

            # each nonzero residue adds one lag. only same-ring table paths can replace their package tail.
            for lag_index, residue_k_per_w in enumerate(residues_k_per_w):
                if residue_k_per_w != 0.0:
                    lag_builder.add(
                        observer_index,
                        source_index,
                        time_constants_s[lag_index],
                        residue_k_per_w,
                        int(layer_codes[lag_index]),
                        pair_type,
                        tailable=(pair_type == PAIR_SELF),
                    )

    if skipped_kind_names:
        meta["rom_skipped_kinds"] = sorted(skipped_kind_names)
    if moved_site_names:
        moved_sites = sorted(moved_site_names)
        meta["rom_sites_off_default"] = moved_sites
        extracted_offsets_um = {name: rom.site_offsets_um[name] for name in moved_sites}
        warnings.warn(
            f"layout {moved_sites} sites are more than {tol_um} um from the ROM self-table offsets "
            f"{extracted_offsets_um} um: their same-ring pairs use the distance table cross_res "
            "at the actual site distance",
            RuntimeWarning,
            stacklevel=3,
        )
    if distances_outside_grid_um:
        example_distances_um = sorted({round(distance, 2) for distance in distances_outside_grid_um})[:6]
        warnings.warn(
            f"thermal ROM cross_r_um grid {distance_grid_um[0]}..{distance_grid_um[-1]} um does not cover distances "
            f"{example_distances_um} um; clamped to the grid ends (radially symmetric table, valid for d > 2R)",
            RuntimeWarning,
            stacklevel=3,
        )


# how much of the settled temperature rise should develop on the package time scale?
# a constant power P (W) gives one lag an eventual rise P*r (K), where r is its gain (K/W).
# for each path, split the eligible gain Z between the package lag and the other eligible lags.
def replace_package_tail_preserving_dc_gain(
    lg: _Lags,  # lg = lags (mixed units) - lag indices, time constants, and thermal residues - type thermal lag builder
    tail_tau: ArrayLike,  # tail_tau = tau_tail (s) - absolute package-tail time constants - type array-like, shape (B,)
    tail_share: ArrayLike,  # tail_share = f (1) - fraction of the eligible steady rise assigned to the package lag - shape (B,)
) -> (
    None  # update the eligible lags in lg
):
    eligible_lags_by_path: dict[tuple[int, int], list[int]] = {}
    for lag_index, observer_index in enumerate(lg.observer_indices):
        if lg.tail_replaceable[lag_index]:
            heat_path = (observer_index, lg.source_indices[lag_index])
            eligible_lags_by_path.setdefault(heat_path, []).append(lag_index)

    for (observer_index, source_index), eligible_lags in eligible_lags_by_path.items():
        pair_type = lg.pair_codes[eligible_lags[0]]
        other_lags = [index for index in eligible_lags if not lg.tail_flags[index]]
        package_lags = [index for index in eligible_lags if lg.tail_flags[index]]

        # save the original eligible gain Z before editing any lag.
        total_gain_k_per_w = np.sum([lg.residues_k_per_w[index] for index in eligible_lags], axis=0)
        old_other_gain_k_per_w = (
            np.sum([lg.residues_k_per_w[index] for index in other_lags], axis=0)
            if other_lags else np.zeros(lg.world_count)
        )

        # assign f*Z to the package and leave (1-f)*Z for the other lags.
        # for Z = 1000 K/W and f = 0.2, that gives 200 K/W for the package and 800 K/W for the rest.
        package_gain_k_per_w = tail_share * total_gain_k_per_w
        target_other_gain_k_per_w = total_gain_k_per_w * (1.0 - tail_share)

        # target / old sum gives one multiplier for all other lags, preserving their relative gains.
        # if their old sum is 900 K/W, multiply each by 800/900.
        # a zero old sum cannot be rescaled to a nonzero target. the model uses zero for those gains.
        with np.errstate(divide="ignore", invalid="ignore"):
            other_gain_scale = np.where(
                old_other_gain_k_per_w != 0.0, target_other_gain_k_per_w / old_other_gain_k_per_w, 0.0
            )
        for lag_index in other_lags:
            lg.residues_k_per_w[lag_index] = lg.residues_k_per_w[lag_index] * other_gain_scale

        # store the requested package gain in one lag and keep its time constant fixed during later scaling.
        if package_lags:
            package_lag = package_lags[0]
            lg.residues_k_per_w[package_lag] = package_gain_k_per_w
            lg.time_constants_s[package_lag] = np.broadcast_to(tail_tau, (lg.world_count,)).copy()
            lg.fixed_time_constants[package_lag] = True
            for lag_index in package_lags[1:]:
                lg.residues_k_per_w[lag_index] = np.zeros(lg.world_count)
        else:
            lg.add(
                observer_index, source_index, tail_tau, package_gain_k_per_w, LAYER_SLOW, pair_type,
                tailable=True, tail=True, fixed_tau=True,
            )


# multiply all DC-scale rules matching this source/observer kind pair.
def calculate_pair_dc_scale(
    dc_scale: ArrayLike | Mapping[str, ArrayLike],  # dc_scale = s_DC (1) - DC residue scales or source-to-observer matching rules - type array-like or mapping to array-like values
    sk_name: str,  # sk_name = source_kind (1) - source kind of the heat path - type str
    ok_name: str,  # ok_name = observer_kind (1) - observer kind of the heat path - type str
    B: int,  # B = B (1) - number of worlds - type int scalar
) -> (
    Float64Array  # return = s_DC (1) - product of all matching DC-scale rules - type float64 ndarray, shape (B,)
):
    if not isinstance(dc_scale, Mapping):
        return np.broadcast_to(normalize_world_parameter_vector(dc_scale, "dc_scale"), (B,))
    # start at 1 per world and multiply every applicable rule into that world's gain factor.
    # for example, a global factor of 1.2 and a matching path factor of 0.5 give 0.6.
    factor = np.ones(B)
    for key, value in dc_scale.items():
        if key == "*":
            factor = factor * normalize_world_parameter_vector(value, "dc_scale[*]")
            continue
        if "->" not in key:
            raise ValueError(f"dc_scale key {key!r} must be '*' or '<source>-><observer>'")
        source_pattern, observer_pattern = (pattern.strip() for pattern in key.split("->", 1))
        for pattern, valid_kinds in ((source_pattern, SOURCE_KINDS), (observer_pattern, OBSERVER_KINDS)):
            if pattern != "*" and pattern not in valid_kinds:
                raise ValueError(f"dc_scale key {key!r}: unknown kind {pattern!r}")
        if source_pattern in ("*", sk_name) and observer_pattern in ("*", ok_name):
            factor = factor * normalize_world_parameter_vector(value, f"dc_scale[{key}]")
    return np.broadcast_to(factor, (B,))


# how does thermal uncertainty change the speed of a fitted warmup curve?
# positive and negative lag contributions can cancel during warmup.
# scaling all adjustable time constants by k changes each exp(-t/tau) to exp(-(t/k)/tau).
# the adjustable part of the curve keeps its shape and runs on the stretched time axis.
# for example, k = 2 moves the adjustable response at 1 ms to 2 ms.
# x
def calculate_shared_pair_time_scales(
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


# assemble the source-to-observer thermal system for the requested ring layout.
# choose the heat response for the layout, then assign the requested share of warming to the package lag.
# apply chip uncertainty last to set how much warming occurs and how quickly it develops.
# prune negligible residues and sort lags by observer, source, and time constant.
# classify optical heat paths for the implicit self-heating solver.
# helper function
# x
def assemble(
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
    ring_count = int(n_rings)
    if layout is None:
        if ring_count != 1:
            raise ValueError("layout is required for more than one ring")
        layout = LayoutSites(np.zeros((1, 2)))
    elif not isinstance(layout, LayoutSites):
        layout = LayoutSites.from_sites(layout, ring_count, ring_pitch_um, temp_sensors)
    if layout.n_rings != ring_count:
        raise ValueError(f"layout has {layout.n_rings} rings, n_rings is {ring_count}")
    if scales is None:
        scales = ThermalScales()
    elif isinstance(scales, Mapping):
        scales = ThermalScales(**scales)
    src_kinds = tuple(SOURCE_KINDS if sources is None else sources)
    obs_kinds = tuple(OBSERVER_KINDS if observers is None else observers)
    for k in src_kinds:
        if k not in SOURCE_KINDS:
            raise ValueError(f"unknown source kind {k!r}")
    for k in obs_kinds:
        if k not in OBSERVER_KINDS:
            raise ValueError(f"unknown observer kind {k!r}")

    scale_arrays = {"time_scale_fast": normalize_world_parameter_vector(scales.time_scale_fast, "time_scale_fast"),
          "time_scale_slow": normalize_world_parameter_vector(scales.time_scale_slow, "time_scale_slow"),
          "crosstalk_scale": normalize_world_parameter_vector(scales.crosstalk_scale, "crosstalk_scale")}
    if scales.package_tail_share_frac is not None:
        scale_arrays["pkg_tail_share"] = normalize_world_parameter_vector(scales.package_tail_share_frac, "package_tail_share_frac")
        # default tau when only the share is drawn - assumed 4 ms, [CO22-TCPMT] circuit-level start-up (row T2)
        scale_arrays["pkg_tail_tau"] = normalize_world_parameter_vector(4.0e-3 if scales.package_tail_tau_s is None else scales.package_tail_tau_s,
                               "package_tail_tau_s")
    if isinstance(scales.dc_scale, Mapping):
        for k, v in scales.dc_scale.items():
            scale_arrays[f"dc[{k}]"] = normalize_world_parameter_vector(v, f"dc_scale[{k}]")
    else:
        scale_arrays["dc"] = normalize_world_parameter_vector(scales.dc_scale, "dc_scale")
    scale_arrays["transport_delay_s"] = normalize_world_parameter_vector(scales.transport_delay_s, "transport_delay_s")
    if np.any(~(scale_arrays["transport_delay_s"] >= 0.0)):
        raise ValueError("transport_delay_s must be >= 0")
    hyp_raw = scales.scaling_hypothesis
    scale_arrays["scaling_hypothesis"] = np.zeros(1 if isinstance(hyp_raw,
        str) else np.asarray(hyp_raw).reshape(-1).shape[0])
    meta: dict = {"n_rings": ring_count}

    if isinstance(rom, InterimRom):
        interim_params = validate_and_broadcast_interim_parameters(rom, ring_count)
        world_count = infer_world_count({**scale_arrays, **{k: v[:, 0] if v.shape[1] == 1 else v for k, v in interim_params.items()}})
        interim_params = {k: np.broadcast_to(v, (world_count, ring_count)) for k, v in interim_params.items()}
        elements, element_indices = build_thermal_sources_and_observers(layout, ring_count, src_kinds, obs_kinds, rom.pd_offset_um,
            rom.temp_sensor_offset_um,
                                  rom.driver_offset_um)
        lags = _Lags(world_count)
        append_interim_heat_path_lags(rom, elements, layout, ring_count, world_count, interim_params, lags)
        meta.update(model="interim", name=rom.name, crosstalk_dynamics=rom.crosstalk_dynamics,
                    xt_tau_law=rom.xt_tau_law)
        bad = (interim_params["z_ratio"] * (1.0 - interim_params["f_sh"]) < 1.0 - 1e-12)
        if np.any(bad):
            warnings.warn("interim thermal model: z_ratio*(1 - selfheat_share) < 1, so the heater/ring residue matrix "
                          "of the heater poles is not positive semidefinite (not a passive RC network); "
                          "selfheat_share_frac = (z - 1)/z restores it", RuntimeWarning, stacklevel=2)
    elif isinstance(rom, ThermalRom):
        world_count = infer_world_count(scale_arrays)
        for nm in ("monitor_pd", "temp_sensor"):
            if nm in obs_kinds and nm in rom.observers and nm not in rom.site_offsets_um:
                warnings.warn(f"thermal ROM has no site offset for {nm}; its sites are placed at the ring centre",
                              RuntimeWarning, stacklevel=2)
        # include drivers at explicit layout sites.
        # an explicitly requested driver may use the ROM default offset.
        # G26 places its drivers on a separate EIC beside the PIC.
        driver_offset = (rom.site_offsets_um.get("driver") if ("driver" in rom.sources and sources is not None
                                                         and "driver" in sources) else None)
        elements, element_indices = build_thermal_sources_and_observers(layout, ring_count, src_kinds, obs_kinds,
                                  rom.site_offsets_um.get("monitor_pd", (0.0, 0.0)),
                                  rom.site_offsets_um.get("temp_sensor", (0.0, 0.0)), driver_offset)
        lags = _Lags(world_count)
        append_extracted_heat_path_lags(rom, elements, layout, world_count, float(scales.tau_split_s), lags, meta)
        meta.update(model="rom", rom_path=rom.meta.get("path"), rom_sha256=rom.meta.get("sha256"),
                    family=rom.meta.get("family"), film_nm=rom.meta.get("film_nm"))
    else:
        raise TypeError(f"rom must be a ThermalRom or an InterimRom, got {type(rom).__name__}")

    if "pkg_tail_share" in scale_arrays:
        # split each eligible path's original gain before applying the chip uncertainty factors.
        package_tail_share = np.broadcast_to(scale_arrays["pkg_tail_share"], (world_count,))
        if np.any((package_tail_share < 0) | (package_tail_share >= 1)):
            raise ValueError("package_tail_share_frac must be in [0, 1)")
        package_time_constants_s = np.broadcast_to(scale_arrays["pkg_tail_tau"], (world_count,))
        replace_package_tail_preserving_dc_gain(lags, package_time_constants_s, package_tail_share)

    lag_count = len(lags.observer_indices)
    if lag_count == 0:
        raise ValueError("the assembled thermal system has no lags (no sources or observers)")
    time_constants_s = np.stack(lags.time_constants_s, axis=1)
    thermal_residues_k_per_w = np.stack(lags.residues_k_per_w, axis=1)
    lag_observer_indices = np.array(lags.observer_indices, np.int64)
    lag_source_indices = np.array(lags.source_indices, np.int64)
    lag_layers = np.array(lags.layer_codes, np.int64)
    lag_pair_types = np.array(lags.pair_codes, np.int64)
    lag_tail_flags = np.array(lags.tail_flags, np.int64)
    fixed_time_constants = np.array(lags.fixed_time_constants, bool)
    src_kind = np.array(elements.src_kind, np.int64)
    src_ring = np.array(elements.src_ring, np.int64)
    obs_kind = np.array(elements.obs_kind, np.int64)
    obs_ring = np.array(elements.obs_ring, np.int64)

    # gain factors change the eventual temperature rise. time factors change how quickly it develops.
    hypothesis_codes = encode_thermal_scaling_hypotheses(scales.scaling_hypothesis, world_count)
    dc_factors = np.ones((world_count, lag_count))
    for lag_index in range(lag_count):
        source_index = lag_source_indices[lag_index]
        observer_index = lag_observer_indices[lag_index]
        source_name = SOURCE_KINDS[src_kind[source_index]]
        observer_name = OBSERVER_KINDS[obs_kind[observer_index]]
        dc_factors[:, lag_index] = calculate_pair_dc_scale(scales.dc_scale, source_name, observer_name, world_count)
    cross_factors = np.ones((world_count, lag_count))
    cross_factors[:, lag_pair_types == PAIR_CROSS] = np.broadcast_to(scale_arrays["crosstalk_scale"], (world_count,))[:, None]
    fast_time_scale = np.broadcast_to(scale_arrays["time_scale_fast"], (world_count,))
    slow_time_scale = np.broadcast_to(scale_arrays["time_scale_slow"], (world_count,))
    time_factors = np.where(lag_layers[None, :] == LAYER_FAST, fast_time_scale[:, None], slow_time_scale[:, None])
    # replace the layer factors where fitted lags need to keep their timing relative to each other.
    whole_pair_mask, whole_pair_factors = calculate_shared_pair_time_scales(
        time_constants_s,
        thermal_residues_k_per_w,
        lag_observer_indices,
        lag_source_indices,
        lag_layers,
        fixed_time_constants,
        fast_time_scale,
        slow_time_scale,
        float(scales.tau_split_s),
    )
    # unselected lags keep their requested fast or slow factor here. fixed times are kept below.
    if whole_pair_factors is not None:
        time_factors = np.where(whole_pair_mask[None, :], whole_pair_factors, time_factors)
    meta["time_scale_whole_pairs"] = int(np.unique(
        lag_observer_indices[whole_pair_mask] * len(elements.src_kind) + lag_source_indices[whole_pair_mask]).size)
    resistance = (hypothesis_codes == SCALING_HYPOTHESES.index("resistance"))[:, None]
    # under the resistance hypothesis, a gain multiplier of 1.5 also multiplies tau by 1.5 because tau = R*C.
    # the crosstalk gain factor affects only residues. an explicitly set package time stays fixed.
    resistance_time_factor = np.where(resistance, dc_factors, 1.0)
    time_constants_s = np.where(
        fixed_time_constants[None, :], time_constants_s, time_constants_s * time_factors * resistance_time_factor
    )
    # r_new = r*s_DC*s_cross. s_cross is one for lags outside the cross-ring class.
    thermal_residues_k_per_w = thermal_residues_k_per_w * dc_factors * cross_factors
    if np.any(~(time_constants_s > 0)) or not np.all(np.isfinite(time_constants_s)):
        raise ValueError("assembled time constants must be finite and > 0 (check time scales and dc_scale)")

    # prune, sort by observer, then source, then tau
    keep = np.any(np.abs(thermal_residues_k_per_w) > float(prune_abs_k_per_w), axis=0)
    if prune_abs_k_per_w == 0.0:
        keep = np.any(thermal_residues_k_per_w != 0.0, axis=0)
    order = np.lexsort((time_constants_s[0], lag_source_indices, lag_observer_indices))
    order = order[keep[order]]
    time_constants_s, thermal_residues_k_per_w, lag_observer_indices, lag_source_indices, lag_layers, lag_pair_types, lag_tail_flags = (time_constants_s[:, order], thermal_residues_k_per_w[:, order], lag_observer_indices[order], lag_source_indices[order], lag_layers[order],
                                             lag_pair_types[order], lag_tail_flags[order])
    observer_count = obs_kind.shape[0]
    obs_ptr = np.zeros(observer_count + 1, np.int64)
    np.add.at(obs_ptr, lag_observer_indices + 1, 1)
    obs_ptr = np.cumsum(obs_ptr)

    kind = np.zeros(lag_observer_indices.shape[0], np.int64)
    optical_source_mask = src_kind[lag_source_indices] == SRC_WG
    ring_observer_mask = obs_kind[lag_observer_indices] == OBS_RING
    same_ring_mask = src_ring[lag_source_indices] == obs_ring[lag_observer_indices]
    kind[optical_source_mask & ring_observer_mask & same_ring_mask] = LAG_OPT_SELF
    kind[optical_source_mask & ring_observer_mask & ~same_ring_mask] = LAG_OPT_CROSS_RING
    kind[optical_source_mask & ~ring_observer_mask] = LAG_OPT_OTHER

    meta.update(scaling_hypothesis=[SCALING_HYPOTHESES[c] for c in np.unique(hypothesis_codes)],
                tail_from_scales="pkg_tail_share" in scale_arrays, layout=layout)
    return ThermalSystem(
        np.ascontiguousarray(time_constants_s), np.ascontiguousarray(thermal_residues_k_per_w), lag_source_indices, lag_observer_indices, kind, lag_pair_types, lag_layers, lag_tail_flags, obs_ptr,
        src_kind, src_ring, tuple(elements.src_names), obs_kind, obs_ring, tuple(elements.obs_names),
        element_indices["ring_obs"], element_indices["heater_obs"], element_indices["pd_obs"],
        element_indices["temp_obs"], element_indices["heater_src"], element_indices["wg_src"],
        element_indices["junction_src"], element_indices["driver_src"], element_indices["aggressor_src"],
        np.flatnonzero((kind == LAG_OPT_SELF) | (kind == LAG_OPT_OTHER)).astype(np.int64), ring_count, meta,
        np.broadcast_to(scale_arrays["transport_delay_s"], (world_count,)).copy())


# InterimRom fields as (B|1, N|1) arrays, with defaults resolved and bounds checked.
# helper function
def validate_and_broadcast_interim_parameters(
    spec: InterimRom,  # spec = model (mixed units) - interim thermal resistance and time-constant parameters - type InterimRom record
    n: int,  # n = N (1) - number of rings - type int scalar
) -> (
    dict[str, Float64Array]  # return = parameters (field units) - validated interim parameters with defaults resolved - type dict of float64 ndarrays, shape (1 or B, 1 or N)
):
    # broadcast one interim parameter over worlds and rings.
    # helper function
    def broadcast_interim_ring_parameter(
        v: ArrayLike,  # v = parameter (field units) - interim parameter to broadcast - type array-like, scalar, (B,), or (B, N)
        name: str,  # name = name (1) - element name or diagnostic parameter label - type str
    ) -> (
        Float64Array  # return = parameter (input units) - parameter broadcast over worlds and rings - type float64 ndarray, shape (1 or B, 1 or N)
    ):
        a = np.asarray(v, dtype=np.float64)
        if a.ndim == 0:
            return a.reshape(1, 1)
        if a.ndim == 1:
            return a[:, None]
        if a.ndim == 2 and a.shape[1] in (1, n):
            return a
        raise ValueError(f"InterimRom.{name}: expected scalar, (B,) or (B, N), got shape {a.shape}")

    p = {"z0": broadcast_interim_ring_parameter(spec.z_heater_ring_k_per_w, "z_heater_ring_k_per_w"),
         "z_ratio": broadcast_interim_ring_parameter(spec.z_ratio_opt_heater, "z_ratio_opt_heater"),
         "tau_fast": broadcast_interim_ring_parameter(spec.tau_fast_s, "tau_fast_s"), "share_fast": broadcast_interim_ring_parameter(spec.share_fast_frac, "share_fast_frac"),
         "tau_slow": broadcast_interim_ring_parameter(spec.tau_slow_s, "tau_slow_s"), "tail_tau": broadcast_interim_ring_parameter(spec.tail_tau_s, "tail_tau_s"),
         "tail_share": broadcast_interim_ring_parameter(spec.tail_share_frac, "tail_share_frac"),
         "selfheat_tau": broadcast_interim_ring_parameter(spec.selfheat_tau_s, "selfheat_tau_s"),
         "heater_self_ratio": broadcast_interim_ring_parameter(spec.heater_self_ratio, "heater_self_ratio"),
         "heater_node_tau": broadcast_interim_ring_parameter(spec.heater_node_tau_s, "heater_node_tau_s"),
         "xt_nearest": broadcast_interim_ring_parameter(spec.xt_nearest_frac, "xt_nearest_frac"), "xt_floor": broadcast_interim_ring_parameter(spec.xt_floor_frac, "xt_floor_frac"),
         "xt_next": broadcast_interim_ring_parameter(0.0 if spec.xt_next_frac is None else spec.xt_next_frac, "xt_next_frac"),
         "xt_tau_ref": broadcast_interim_ring_parameter(spec.xt_tau_ref_s, "xt_tau_ref_s"), "xt_tau_min": broadcast_interim_ring_parameter(spec.xt_tau_min_s, "xt_tau_min_s"),
         "xt_tau_far": broadcast_interim_ring_parameter(0.0 if spec.xt_tau_far_s is None else spec.xt_tau_far_s, "xt_tau_far_s"),
         "ring_radius": broadcast_interim_ring_parameter(spec.ring_radius_um, "ring_radius_um")}
    p["site_floor"] = (p["xt_floor"] if spec.site_floor_frac is None
                       else broadcast_interim_ring_parameter(spec.site_floor_frac, "site_floor_frac"))
    z = p["z_ratio"]
    # by default z*(1-f_sh) = 1. the optical gain left on the heater lags then equals the heater gain.
    p["f_sh"] = (broadcast_interim_ring_parameter(spec.selfheat_share_frac, "selfheat_share_frac") if spec.selfheat_share_frac is not None
                 else (z - 1.0) / z)
    if np.any(p["z0"] <= 0):
        raise ValueError("z_heater_ring_k_per_w must be > 0")
    if np.any(z < 1.0):
        raise ValueError("z_ratio_opt_heater must be >= 1 (Z_opt/Z_heater >= 1 by reciprocity)")
    if np.any(p["heater_self_ratio"] < 1.0):
        raise ValueError("heater_self_ratio must be >= 1 (R_th,H >= Z_heater->ring)")
    if np.any((p["f_sh"] < 0) | (p["f_sh"] > 1)):
        raise ValueError("selfheat_share_frac must be in [0, 1]")
    if np.any(p["ring_radius"] < 0):
        raise ValueError("ring_radius_um must be >= 0")
    if np.any(p["site_floor"] < 0):
        raise ValueError("site_floor_frac must be >= 0")
    if np.any(p["share_fast"] < 0) or np.any(p["tail_share"] < 0) or np.any(p["share_fast"] + p["tail_share"] > 1):
        raise ValueError("share_fast_frac and tail_share_frac must be >= 0 with a sum <= 1")
    for k in ("tau_fast", "tau_slow", "tail_tau", "selfheat_tau", "heater_node_tau", "xt_tau_ref", "xt_tau_min"):
        if np.any(p[k] <= 0):
            raise ValueError(f"interim time constant {k} must be > 0")
    if spec.crosstalk_dynamics not in ("imec", "self"):
        raise ValueError("crosstalk_dynamics must be 'imec' or 'self'")
    return p


# assemble an interim thermal system from the named physical parameter overrides.
# InterimRom defines the heat-path equations, default values, and provenance.
# helper function
def assemble_interim_thermal_system(
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
    return assemble(InterimRom(**preset), layout, n_rings, scales, sources=sources, observers=observers,
                    temp_sensors=temp_sensors, ring_pitch_um=ring_pitch_um)


# simulate observer temperatures for supplied source-power samples.
# zoh holds the starting power constant within each interval.
# foh interpolates linearly between the two endpoint powers.
# the lag update is exact for that power hold.
# nonlinear optical feedback is solved separately by the self-heating module.
# helper function
# simulation
# (todo) foh and zoh are ambigious names
# (todo) why not one fixed option instead of foh and zoh?
# x
def simulate(
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
    u = np.asarray(u_samples_w, dtype=np.float64)
    if u.ndim == 2:
        u = u[:, None, :]
    if hold not in ("zoh", "foh"):
        raise ValueError("hold must be 'zoh' or 'foh'")
    if u.ndim != 3 or u.shape[2] != system.n_src:
        raise ValueError(f"u_samples_w must be (n+1, B|1, S={system.n_src}), got {u.shape}")
    B = max(system.batch, u.shape[1])
    if system.batch not in (1, B) or u.shape[1] not in (1, B):
        raise ValueError(f"batch mismatch: system has {system.batch} worlds, u has {u.shape[1]}")
    a, c0, c1 = system.coefficients(h_s)
    x = np.zeros((B, system.n_lags)) if x0 is None else np.array(x0, dtype=np.float64, copy=True)
    base_temp_k = np.atleast_1d(np.asarray(t_base_k, np.float64))
    source_scales = np.ones((1, system.n_src)) if src_scale is None else np.atleast_2d(np.asarray(src_scale,
        np.float64))
    out = np.zeros((u.shape[0], B, system.n_obs))
    write_observer_temperatures(x, system.obs_ptr, base_temp_k, out[0])
    for k in range(u.shape[0] - 1):
        u1 = u[k] if hold == "zoh" else u[k + 1]
        advance_thermal_states(x, a, c0, c1, system.lag_src, np.ascontiguousarray(u[k]), np.ascontiguousarray(u1), source_scales)
        write_observer_temperatures(x, system.obs_ptr, base_temp_k, out[k + 1])
    return out


# return selected observer/source indices, their lag indices, and local pair indices.
# helper function
def select_thermal_response_lags(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    obs: ArrayLike | None,  # obs = observer_indices (1) - selected observers, or all observers when None - type integer scalar, array-like, or None
    src: ArrayLike | None,  # src = source_indices (1) - selected sources, or all sources when None - type integer scalar, array-like, or None
) -> tuple[
    Int64Array,  # return[0] = o_selected (1) - selected observer indices - type int64 ndarray, shape (O_selected,)
    Int64Array,  # return[1] = s_selected (1) - selected source indices - type int64 ndarray, shape (S_selected,)
    Int64Array,  # return[2] = m_selected (1) - selected lag indices - type int64 ndarray, shape (M_selected,)
    Int64Array,  # return[3] = o_local (1) - local observer index of each selected lag - type int64 ndarray, shape (M_selected,)
    Int64Array,  # return[4] = s_local (1) - local source index of each selected lag - type int64 ndarray, shape (M_selected,)
]:
    selected_observers = np.arange(system.n_obs) if obs is None else np.atleast_1d(np.asarray(obs, np.int64))
    selected_sources = np.arange(system.n_src) if src is None else np.atleast_1d(np.asarray(src, np.int64))
    observer_map = np.full(system.n_obs, -1, np.int64)
    observer_map[selected_observers] = np.arange(selected_observers.size)
    source_map = np.full(system.n_src, -1, np.int64)
    source_map[selected_sources] = np.arange(selected_sources.size)
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
# helper function
def sum_selected_thermal_responses(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    obs: ArrayLike | None,  # obs = observer_indices (1) - selected observers, or all observers when None - type integer scalar, array-like, or None
    src: ArrayLike | None,  # src = source_indices (1) - selected sources, or all sources when None - type integer scalar, array-like, or None
    kernel: Callable[[Float64Array, Float64Array], NumericArray],  # kernel = kernel (response units) - per-lag response function of time constants and thermal residues - type callable returning a numeric ndarray
) -> (
    NumericArray  # return = response (kernel response units) - selected per-lag contributions summed by source and observer - type numeric ndarray, shape (B, O_selected, S_selected, samples)
):
    selected_observers, selected_sources, selected_lags, local_observer, local_source = select_thermal_response_lags(system, obs, src)
    contributions = kernel(system.taus_s[:, selected_lags], system.res_k_per_w[:,
        selected_lags])           # (B, Msel, F)
    out = np.zeros((system.batch, selected_observers.size, selected_sources.size, contributions.shape[-1]),
        dtype=contributions.dtype)
    # add every lag to its observer and source cell, even when several lags share that cell.
    np.add.at(out, (slice(None), local_observer, local_source, slice(None)), contributions)
    return out


# complex transfer impedances Z(j 2 pi f) = sum_k r_k/(1 + j 2 pi f tau_k), shape (B, O', S', F), K/W.
# obs, src - index arrays selecting observers/sources (default - all).
# f_hz in Hz (cycles per second).
# transfer function from p phasor to x phasor (x phasor / p phasor)
# x
def frequency_response(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    f_hz: ArrayLike,  # f_hz = f (Hz) - frequencies at which to evaluate thermal response - type array-like, scalar or shape (F,)
    obs: ArrayLike | None = None,  # obs = observer_indices (1) - selected observers, or all observers when None - type integer scalar, array-like, or None
    src: ArrayLike | None = None,  # src = source_indices (1) - selected sources, or all sources when None - type integer scalar, array-like, or None
) -> (
    Complex128Array  # return = Z (K/W) - complex source-to-observer thermal impedances - type complex128 ndarray, shape (B, O_selected, S_selected, F)
):
    w = 2.0 * np.pi * np.atleast_1d(np.asarray(f_hz, np.float64))
    # for a sinusoid, replace d/dt by j*omega in the lag ODE to get x/P = r/(1+j*omega*tau).
    # when omega*tau = 1, a positive-gain lag has gain r/sqrt(2) and a 45 degree phase lag.
    return sum_selected_thermal_responses(system, obs, src,
                       lambda tau, r: r[..., None] / (1.0 + 1j * w[None, None, :] * tau[..., None]))


# how much does the observer warm after a step in source power?
# each lag contributes r_k*(1-exp(-t/tau_k)) in K/W, where r_k sets the final gain and tau_k sets the rise time.
# sum the lag gains at each time and multiply by the source power step (W) to get the temperature rise (K).
# x
def step_response(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    t_s: ArrayLike,  # t_s = t (s) - times at which to evaluate thermal response - type array-like, scalar or shape (T,)
    obs: ArrayLike | None = None,  # obs = observer_indices (1) - selected observers, or all observers when None - type integer scalar, array-like, or None
    src: ArrayLike | None = None,  # src = source_indices (1) - selected sources, or all sources when None - type integer scalar, array-like, or None
) -> (
    Float64Array  # return = Z_step (K/W) - temperature rise per unit source-power step - type float64 ndarray, shape (B, O_selected, S_selected, T)
):
    times_s = np.atleast_1d(np.asarray(t_s, np.float64))

    def lag_step_gain(
        time_constants_s: Float64Array,  # tau (s) - one time constant per world and selected lag
        gains_k_per_w: Float64Array,  # r (K/W) - settled gain per world and selected lag
    ) -> (
        Float64Array  # Z_step (K/W) - gain per world, selected lag, and evaluation time
    ):
        # append a time axis and evaluate 1-exp(-t/tau) accurately near t = 0.
        fraction_developed = -np.expm1(-times_s[None, None, :] / time_constants_s[..., None])
        return gains_k_per_w[..., None] * fraction_developed

    return sum_selected_thermal_responses(system, obs, src, lag_step_gain)


# return the response kernel to a power ramp, with shape (B, O', S', T) in K*s/W.
# sum r_k*[t-tau_k*(1-exp(-t/tau_k))].
# multiply by ramp slope in W/s to obtain K.
# t_s uses s.
# evalulates the integral, from dp = ads change
# x 
def ramp_response(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    t_s: ArrayLike,  # t_s = t (s) - times at which to evaluate thermal response - type array-like, scalar or shape (T,)
    obs: ArrayLike | None = None,  # obs = observer_indices (1) - selected observers, or all observers when None - type integer scalar, array-like, or None
    src: ArrayLike | None = None,  # src = source_indices (1) - selected sources, or all sources when None - type integer scalar, array-like, or None
) -> (
    Float64Array  # return = Z_ramp (K*s/W) - temperature response per unit source-power ramp rate - type float64 ndarray, shape (B, O_selected, S_selected, T)
):
    t = np.atleast_1d(np.asarray(t_s, np.float64))
    # a power ramp is a stream of small power steps, so integrate the step gain from 0 to t.
    # integrating r*(1-exp(-t/tau)) gives r*[t-tau*(1-exp(-t/tau))].
    return sum_selected_thermal_responses(system, obs, src, lambda tau, r: r[..., None] * (
        t[None, None, :] + tau[..., None] * np.expm1(-t[None, None, :] / tau[..., None])))


# static transfer resistances Z(0) = sum_k r_k, shape (B, O', S') in K/W.
# final temperature gain
# x
def dc_gain(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    obs: ArrayLike | None = None,  # obs = observer_indices (1) - selected observers, or all observers when None - type integer scalar, array-like, or None
    src: ArrayLike | None = None,  # src = source_indices (1) - selected sources, or all sources when None - type integer scalar, array-like, or None
) -> (
    Float64Array  # return = Z_DC (K/W) - steady thermal resistance of each source-to-observer path - type float64 ndarray, shape (B, O_selected, S_selected)
):
    # as t tends to infinity, each lag's step fraction tends to 1 and only r remains.
    return sum_selected_thermal_responses(system, obs, src, lambda tau, r: r[..., None])[..., 0]


# construct continuous state-space matrices for selected thermal sources and observers.
# each selected lag obeys tau_m*dx_m/dt=-x_m+r_m*P_source.
# A=diag(-1/tau), B[m,s]=r_m/tau_m, C[o,m]=1, and D=0.
# the outputs are temperature rises above the base temperature.
# x matrix control system of slow and fast
def state_space(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    b: int = 0,  # b = b (1) - world index - type int scalar
    obs: ArrayLike | None = None,  # obs = observer_indices (1) - selected observers, or all observers when None - type integer scalar, array-like, or None
    src: ArrayLike | None = None,  # src = source_indices (1) - selected sources, or all sources when None - type integer scalar, array-like, or None
) -> tuple[
    Float64Array,  # return[0] = A (1/s) - continuous thermal state matrix - type float64 ndarray, shape (M_selected, M_selected)
    Float64Array,  # return[1] = B (K/(W*s)) - source-to-state drive matrix - type float64 ndarray, shape (M_selected, S_selected)
    Float64Array,  # return[2] = C (1) - state-to-observer summation matrix - type float64 ndarray, shape (O_selected, M_selected)
    Float64Array,  # return[3] = D (K/W) - zero direct feedthrough matrix - type float64 ndarray, shape (O_selected, S_selected)
]:
    o_sel, s_sel, m, lio, lis = select_thermal_response_lags(system, obs, src)
    tau = system.taus_s[b, m]
    r = system.res_k_per_w[b, m]
    # divide the lag ODE by tau to get the decay -1/tau and source drive r/tau.
    A = np.diag(-1.0 / tau)
    Bm = np.zeros((m.size, s_sel.size))
    Bm[np.arange(m.size), lis] = r / tau
    # C[o,m] = 1 routes lag m to observer o. D = 0 because power must act through the lag states.
    C = np.zeros((o_sel.size, m.size))
    C[lio, np.arange(m.size)] = 1.0
    return (
        A,
        Bm,
        C,
        np.zeros((o_sel.size, s_sel.size)),
    )


# measure the largest relative difference between reciprocal heat paths.
# compare heater-to-heater, optical-to-ring, and heater-to-ring reciprocal pairs.
# ignore source or observer kinds that are absent from the assembled system.
def reciprocity_error(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    f_hz: ArrayLike = (0.0, 1e3, 1e4, 1e5, 1e6),  # f_hz = f (Hz) - frequencies at which to evaluate thermal response - type array-like, scalar or shape (F,)
    b: int | None = None,  # b = b (1) - optional world index, or all worlds - type int or None
) -> (
    float  # return = error (1) - largest relative thermal reciprocity error - type float scalar
):
    H = frequency_response(system, f_hz)
    if b is not None:
        H = H[b:b + 1]
    n = system.n_rings
    num, scale = 0.0, 0.0
    combos = ((system.heater_src, system.heater_obs, system.heater_src, system.heater_obs),
              (system.wg_src, system.ring_obs, system.wg_src, system.ring_obs),
              (system.heater_src, system.ring_obs, system.wg_src, system.heater_obs))
    for s1, o1, s2, o2 in combos:
        for i in range(n):
            for j in range(n):
                if min(s1[j], o1[i], s2[i], o2[j]) < 0:
                    continue
                z1 = H[:, o1[i], s1[j]]
                z2 = H[:, o2[j], s2[i]]
                num = max(num, float(np.max(np.abs(z1 - z2))))
                scale = max(scale, float(np.max(np.abs(z1))), float(np.max(np.abs(z2))))
    # divide the largest absolute mismatch by the largest magnitude among the compared responses.
    return num / scale if scale > 0 else 0.0


# DC optical self-heating resistance Z_opt->ring,i(0) of each ring, (B, N) K/W (sum of LAG_OPT_SELF residues).
def selfheat_dc_k_per_w(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
) -> (
    Float64Array  # return = Z_optical_self (K/W) - optical self-heating resistance of each ring - type float64 ndarray, shape (B, N)
):
    out = np.zeros((system.batch, system.n_rings))
    m = np.flatnonzero(system.lag_kind == LAG_OPT_SELF)
    # sum the same-ring absorption gains by ring to get eventual warming per absorbed watt.
    np.add.at(out, (slice(None), system.src_ring[system.lag_src[m]]), system.res_k_per_w[:, m])
    return out


# bound the optical feedback gain introduced by using previous-step cross-ring absorbed power.
# g_i sums |c_start+c_end|*max|dP_abs,j/dT_j| over cross-ring lags into ring i.
# a small gain limits the effect of that one-step delay.
# bounds for calculating multiple effects together
# x
def cross_optical_lag_gain(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    h_s: float,  # h_s = h (s) - thermal step duration - type float scalar
    dpabs_dt_max_w_per_k: ArrayLike,  # dpabs_dt_max_w_per_k = B_dP_dT (W/K) - bounds on absorbed-power temperature derivatives - type array-like, shape (N,) or (B, N)
) -> (
    Float64Array  # return = g_cross (1) - lagged cross-ring optical feedback gain - type float64 ndarray, shape (B, N)
):
    power_slope_bounds = np.atleast_2d(np.asarray(dpabs_dt_max_w_per_k, np.float64))
    a, c0, c1 = system.coefficients(h_s)
    cross_lags = np.flatnonzero(system.lag_kind == LAG_OPT_CROSS_RING)
    out = np.zeros((max(system.batch, power_slope_bounds.shape[0]), system.n_rings))
    if cross_lags.size == 0:
        return out
    source_rings = system.src_ring[system.lag_src[cross_lags]]
    observer_rings = system.obs_ring[system.lag_obs[cross_lags]]
    # K/W from one thermal step times W/K from the source ring gives K/K at the observer ring.
    # take absolute contributions before summing so opposite signs cannot hide a large gain.
    gain_contributions = np.abs(c0[:, cross_lags] + c1[:, cross_lags]) * np.broadcast_to(power_slope_bounds,
        (out.shape[0], power_slope_bounds.shape[1]))[:, source_rings]
    np.add.at(out, (slice(None), observer_rings), gain_contributions)
    return out


# raise ValueError if the lagged cross-ring optical gain reaches `limit` (default 0.05).
# returns its max.
# helper function
def assert_cross_lag_gain(
    system: ThermalSystem,  # system = system (mixed units) - assembled thermal states, source maps, and observer maps - type ThermalSystem record
    h_s: float,  # h_s = h (s) - thermal step duration - type float scalar
    dpabs_dt_max_w_per_k: ArrayLike,  # dpabs_dt_max_w_per_k = B_dP_dT (W/K) - bounds on absorbed-power temperature derivatives - type array-like, shape (N,) or (B, N)
    limit: float = 0.05,  # limit = g_max (1) - maximum accepted lagged cross-ring optical gain - type float scalar
) -> (
    float  # return = g_cross_max (1) - largest cross-ring optical feedback gain below the limit - type float scalar
):
    g = cross_optical_lag_gain(system, h_s, dpabs_dt_max_w_per_k)
    gmax = float(np.max(g)) if g.size else 0.0
    if gmax >= limit:
        b, i = np.unravel_index(int(np.argmax(g)), g.shape)
        raise ValueError(f"lagged cross-ring optical heating gain {gmax:.3g} >= {limit} at world {b}, ring {i} for a "
                         f"{h_s:g} s substep; reduce the substep or solve the bank jointly")
    return gmax


# calculate the thermal-resistance multiplier from temperature-dependent conductivity.
# sum each material's reference-to-current conductivity ratio using its DC resistance share.
# the three shares describe silicon film, oxide, and silicon substrate.
# apply the result as a source-residue multiplier at a slowly varying temperature.
def calculate_conductivity_residue_scale(
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
    if w.shape != (3,) or np.any(w < 0) or abs(w.sum() - 1.0) > 1e-9:
        raise ValueError("weights must be three non-negative shares (film, oxide, substrate) summing to 1")
    film_conductivity_ratio = np.asarray(materials.k_si_film_w_per_m_k(t,
        film_nm)) / float(materials.k_si_film_w_per_m_k(t_ref_k, film_nm))
    oxide_conductivity_ratio = np.asarray(materials.k_sio2_w_per_m_k(t)) / float(materials.k_sio2_w_per_m_k(t_ref_k))
    substrate_conductivity_ratio = (np.asarray(materials.k_si_bulk_w_per_m_k(t))
                                    / float(materials.k_si_bulk_w_per_m_k(t_ref_k)))
    # a material's resistance is proportional to 1/k. sum w_i*k_i(T_ref)/k_i(T) over the three materials.
    return w[0] / film_conductivity_ratio + w[1] / oxide_conductivity_ratio + w[2] / substrate_conductivity_ratio


# previous names remain aliases for callers that use the established API.
foh_scalar = calculate_linear_hold_lag_weights
foh_coefficients = calculate_linear_hold_coefficients
zoh_coefficients = calculate_constant_hold_coefficients
thermal_step_b = advance_thermal_state_for_world
thermal_step_h_b = advance_thermal_state_for_world_with_step_size
thermal_outputs_b = write_observer_temperatures_for_world
_row = shared_or_world_row_index
_check_batch = validate_thermal_step_array_shapes
thermal_step = advance_thermal_states
thermal_outputs = write_observer_temperatures
thermal_advance_b = advance_thermal_state_and_temperatures_for_world
thermal_advance = advance_thermal_states_and_temperatures
observer_split_b = separate_known_heat_and_source_gain_for_observer
_sidecar_path = build_rom_sidecar_path
_find_sidecar = find_rom_sidecar_path
rom_path = build_thermal_rom_path
rom_validation_numbers = read_rom_validation_metrics
_check_rom_quality = warn_about_rom_quality
_xy = validate_site_coordinates_um
crosstalk_tau_s = calculate_crosstalk_time_constant_s
site_fraction = calculate_site_coupling_fraction
_find = find_thermal_element_index
_as_b = normalize_world_parameter_vector
_batch_of = infer_world_count
_hyp_codes = encode_thermal_scaling_hypotheses
_build_elements = build_thermal_sources_and_observers
_ranks = calculate_neighbor_ranks_and_distances
_interim_lags = append_interim_heat_path_lags
_rom_lags = append_extracted_heat_path_lags
_apply_tail = replace_package_tail_preserving_dc_gain
_dc_factor = calculate_pair_dc_scale
_whole_pair_time_scales = calculate_shared_pair_time_scales
_interim_params = validate_and_broadcast_interim_parameters
interim_three_pole = assemble_interim_thermal_system
_select = select_thermal_response_lags
_accumulate = sum_selected_thermal_responses
k_temp_residue_scale = calculate_conductivity_residue_scale

__all__ += ['calculate_linear_hold_lag_weights', 'calculate_linear_hold_coefficients', 'calculate_constant_hold_coefficients', 'advance_thermal_state_for_world', 'advance_thermal_state_for_world_with_step_size', 'write_observer_temperatures_for_world', 'shared_or_world_row_index', 'validate_thermal_step_array_shapes', 'advance_thermal_states', 'write_observer_temperatures', 'advance_thermal_state_and_temperatures_for_world', 'advance_thermal_states_and_temperatures', 'separate_known_heat_and_source_gain_for_observer', 'build_rom_sidecar_path', 'find_rom_sidecar_path', 'build_thermal_rom_path', 'read_rom_validation_metrics', 'warn_about_rom_quality', 'validate_site_coordinates_um', 'calculate_crosstalk_time_constant_s', 'calculate_site_coupling_fraction', 'find_thermal_element_index', 'normalize_world_parameter_vector', 'infer_world_count', 'encode_thermal_scaling_hypotheses', 'build_thermal_sources_and_observers', 'calculate_neighbor_ranks_and_distances', 'append_interim_heat_path_lags', 'append_extracted_heat_path_lags', 'replace_package_tail_preserving_dc_gain', 'calculate_pair_dc_scale', 'calculate_shared_pair_time_scales', 'validate_and_broadcast_interim_parameters', 'assemble_interim_thermal_system', 'select_thermal_response_lags', 'sum_selected_thermal_responses', 'calculate_conductivity_residue_scale']
