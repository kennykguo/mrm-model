# calculate settled optical fields, powers, and stored energies in the ring bank.

# desc
# evaluate_exact_ring_response solves the two-direction fields of one ring.
# evaluate_ring_response_per_input_watt uses the general field solve for both zero and nonzero backscatter.
# bus_propagate follows rings in bus order - each ring's through power feeds the next.
# the drop port feeds the monitor, and the absorbed share of intrinsic loss heats the ring.
# reflections and counter-clockwise add-port output do not reach the monitor.
# reflections are reported without repeated backward passes - bound estimates the omitted effect.
# optics settles in picoseconds, so control ticks use steady optics instead of an optical ODE.
# fields follow exp(+i*omega*t), and a round trip multiplies them by a_rt*exp(-i*phi).
# resonance occurs when phi+psi is an integer multiple of 2*pi.
# travelling field squared gives power (W).
# k2_th and k2_dr are power fractions (1), and a_rt is a field-amplitude fraction (1).
# t_rt_s is the round-trip group delay (s).
# rb = 2*pi*beta_hz/FSR (1), with FSR=1/t_rt_s (Hz).
# port order is input/reflection, through, drop/monitor, and add/counter-clockwise output.
# average binary powers as p1*P_one+(1-p1)*P_zero, then sum powers over laser lines.
# independent data makes averaged forward transfers multiply exactly.
# the archived steady-state TCMT comparison is in archive/optics/tcmt.py.
# compiled kernels use preallocated arrays with fixed RING_LINE_OUTPUT_*, RING_OUTPUT_*, LASER_LINE_OUTPUT_*, SELF_HEATING_HEADER_*, and SELF_HEATING_VIEW_INDEX_* layouts.
# absorbed_power_at_trial_temperature evaluates heating, and commit_optics_at_solved_temperature saves accepted outputs.

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Annotated, NotRequired, TypeAlias, TypedDict

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
from numba import njit

from mrm.units import C0


# row and field constants identify array positions.
# their comments give the units of the stored quantities.
# out_rl uses (ring-line output row, ring, line), out_r uses (ring output row, ring), and out_l uses (line output row, line).

# phase constants and array-index cast
# 2*pi (rad) - one full phase turn.
TWO_PI = 2.0 * math.pi
# cast nonnegative array offsets to unsigned 64-bit indices to avoid negative-index checks.
_as_unsigned_array_index = np.uint64

# outputs per ring and laser line
# P_input (W) - power entering the ring after the bus segment.
RING_LINE_OUTPUT_ROW_INPUT_POWER = 0
# P_through (W) - through-port power averaged over the binary data.
RING_LINE_OUTPUT_ROW_THROUGH_POWER = 1
# P_drop (W) - clockwise drop-port power seen by the monitor, averaged over the binary data.
RING_LINE_OUTPUT_ROW_CLOCKWISE_DROP_POWER = 2
# P_add (W) - counterclockwise power leaving the add port, averaged over the binary data.
RING_LINE_OUTPUT_ROW_COUNTERCLOCKWISE_ADD_PORT_POWER = 3
# P_reflected (W) - power reflected toward the laser, averaged over the binary data.
RING_LINE_OUTPUT_ROW_REFLECTED_POWER = 4
# P_intrinsic_loss (W) - absorption, scattering, and radiation loss averaged over the binary data.
RING_LINE_OUTPUT_ROW_INTRINSIC_LOSS_POWER = 5
# P_absorbed (W) - absorbed share of intrinsic loss averaged over the binary data.
RING_LINE_OUTPUT_ROW_ABSORBED_POWER = 6
# U_clockwise (J) - clockwise stored energy averaged over the binary data.
RING_LINE_OUTPUT_ROW_CLOCKWISE_STORED_ENERGY = 7
# U_counterclockwise (J) - counterclockwise stored energy averaged over the binary data.
RING_LINE_OUTPUT_ROW_COUNTERCLOCKWISE_STORED_ENERGY = 8
# P_through,one (W) - level-one through power with the incoming data averaged.
RING_LINE_OUTPUT_ROW_LEVEL_ONE_THROUGH_POWER = 9
# P_through,zero (W) - level-zero through power with the incoming data averaged.
RING_LINE_OUTPUT_ROW_LEVEL_ZERO_THROUGH_POWER = 10
# P_drop,one (W) - clockwise drop power at data level one.
RING_LINE_OUTPUT_ROW_LEVEL_ONE_CLOCKWISE_DROP_POWER = 11
# P_drop,zero (W) - clockwise drop power at data level zero.
RING_LINE_OUTPUT_ROW_LEVEL_ZERO_CLOCKWISE_DROP_POWER = 12
# P_bus_output,one (W) - bus output with this ring at level one and other rings averaged.
RING_LINE_OUTPUT_ROW_LEVEL_ONE_BUS_OUTPUT_POWER = 13
# P_bus_output,zero (W) - bus output with this ring at level zero and other rings averaged.
RING_LINE_OUTPUT_ROW_LEVEL_ZERO_BUS_OUTPUT_POWER = 14
# dP_absorbed/dphi (W/rad) - hold input power fixed and average over the binary data.
RING_LINE_OUTPUT_ROW_ABSORBED_POWER_ROUND_TRIP_PHASE_DERIVATIVE = 15
# number of rows in the per-ring and per-line output array (1).
RING_LINE_OUTPUT_ROW_COUNT = 16

# outputs per ring, summed over laser lines
# sum(P_drop) (W) - monitor optical input summed over laser lines and averaged over the binary data.
RING_OUTPUT_ROW_TOTAL_CLOCKWISE_DROP_POWER = 0
# sum(P_absorbed) (W) - optical self-heating from all lines and directions, averaged over the binary data.
RING_OUTPUT_ROW_TOTAL_ABSORBED_POWER = 1
# sum(P_intrinsic_loss) (W) - intrinsic loss power summed over laser lines.
RING_OUTPUT_ROW_TOTAL_INTRINSIC_LOSS_POWER = 2
# sum(P_drop,one) (W) - clockwise drop power summed over laser lines at data level one.
RING_OUTPUT_ROW_LEVEL_ONE_TOTAL_CLOCKWISE_DROP_POWER = 3
# sum(P_drop,zero) (W) - clockwise drop power summed over laser lines at data level zero.
RING_OUTPUT_ROW_LEVEL_ZERO_TOTAL_CLOCKWISE_DROP_POWER = 4
# d(sum(P_absorbed))/dlambda_res (W/m) - rigid resonance shift with input power held fixed.
RING_OUTPUT_ROW_ABSORBED_POWER_RESONANCE_WAVELENGTH_DERIVATIVE = 5
# number of rows in the per-ring output array (1).
RING_OUTPUT_ROW_COUNT = 6

# outputs per laser line
# P_bus_output (W) - data-averaged power after the output bus segment.
LASER_LINE_OUTPUT_ROW_BUS_OUTPUT_POWER = 0
# 1/(1-epsilon)^2-1 (1) - relative forward-power error bound from omitted repeated reflections.
LASER_LINE_OUTPUT_ROW_FORWARD_POWER_RELATIVE_ERROR_BOUND = 1
# P_returned (W) - incoherent ring reflections arriving back at the bus input.
LASER_LINE_OUTPUT_ROW_BUS_INPUT_RETURNED_REFLECTION_POWER = 2
# number of rows in the per-line output array (1).
LASER_LINE_OUTPUT_ROW_COUNT = 3

# single-ring responses per watt of input power
# P_reflected/P_input (1) - reflected power fraction at the input port.
RING_RESPONSE_ROW_REFLECTED_POWER_FRACTION = 0
# P_through/P_input (1) - through-port power fraction.
RING_RESPONSE_ROW_THROUGH_POWER_FRACTION = 1
# P_drop/P_input (1) - clockwise drop-port power fraction seen by the monitor.
RING_RESPONSE_ROW_CLOCKWISE_DROP_POWER_FRACTION = 2
# P_add/P_input (1) - counterclockwise power fraction leaving the add port.
RING_RESPONSE_ROW_COUNTERCLOCKWISE_ADD_PORT_POWER_FRACTION = 3
# U_clockwise/P_input (J/W) - clockwise stored energy per input watt.
RING_RESPONSE_ROW_CLOCKWISE_STORED_ENERGY_PER_INPUT_POWER = 4
# U_counterclockwise/P_input (J/W) - counterclockwise stored energy per input watt.
RING_RESPONSE_ROW_COUNTERCLOCKWISE_STORED_ENERGY_PER_INPUT_POWER = 5
# P_intrinsic_loss/P_input (1) - intrinsic loss power fraction, including unabsorbed loss.
RING_RESPONSE_ROW_INTRINSIC_LOSS_POWER_FRACTION = 6
# d(P_intrinsic_loss/P_input)/dphi (1/rad) - zero unless the derivative is requested.
RING_RESPONSE_ROW_INTRINSIC_LOSS_POWER_FRACTION_ROUND_TRIP_PHASE_DERIVATIVE = 7
# number of rows in one normalized ring response (1).
RING_RESPONSE_ROW_COUNT = 8

# complex field workspace rows
# p = exp(-(xi+i*phi)/4) (1) - quarter-ring field attenuation and phase rotation.
COMPLEX_WORKSPACE_ROW_QUARTER_RING_FIELD_PROPAGATION_FACTOR = 0
# 1/p (1) - inverse quarter-ring field propagation factor.
COMPLEX_WORKSPACE_ROW_INVERSE_QUARTER_RING_FIELD_PROPAGATION_FACTOR = 1
# rho (1) - each scatterer's complex field reflection coefficient.
COMPLEX_WORKSPACE_ROW_SCATTERER_FIELD_REFLECTION_COEFFICIENT = 2
# drho/dphi (1/rad) - field reflection derivative with respect to round-trip phase.
COMPLEX_WORKSPACE_ROW_SCATTERER_FIELD_REFLECTION_ROUND_TRIP_PHASE_DERIVATIVE = 3
# u (sqrt(W)) - clockwise travelling field amplitude at the current stage.
COMPLEX_WORKSPACE_ROW_CLOCKWISE_FIELD_AMPLITUDE = 4
# v (sqrt(W)) - counterclockwise travelling field amplitude at the current stage.
COMPLEX_WORKSPACE_ROW_COUNTERCLOCKWISE_FIELD_AMPLITUDE = 5
# du/dphi (sqrt(W)/rad) - clockwise field derivative with respect to round-trip phase.
COMPLEX_WORKSPACE_ROW_CLOCKWISE_FIELD_AMPLITUDE_ROUND_TRIP_PHASE_DERIVATIVE = 6
# dv/dphi (sqrt(W)/rad) - counterclockwise field derivative with respect to round-trip phase.
COMPLEX_WORKSPACE_ROW_COUNTERCLOCKWISE_FIELD_AMPLITUDE_ROUND_TRIP_PHASE_DERIVATIVE = 7
# number of rows in the complex field workspace (1).
COMPLEX_WORKSPACE_ROW_COUNT = 8

# real workspace rows and response blocks
# xi = -ln(a_rt) (1) - round-trip field attenuation exponent.
REAL_WORKSPACE_ROW_ROUND_TRIP_FIELD_LOSS_EXPONENT = 0
# t_th = sqrt(1-K_th^2) (1) - through-coupler field self-coupling coefficient.
REAL_WORKSPACE_ROW_THROUGH_COUPLER_FIELD_SELF_COUPLING = 1
# k_th = sqrt(K_th^2) (1) - through-coupler field cross-coupling coefficient.
REAL_WORKSPACE_ROW_THROUGH_COUPLER_FIELD_CROSS_COUPLING = 2
# t_drop = sqrt(1-K_drop^2) (1) - drop-coupler field self-coupling coefficient.
REAL_WORKSPACE_ROW_DROP_COUPLER_FIELD_SELF_COUPLING = 3
# k_drop = sqrt(K_drop^2) (1) - drop-coupler field cross-coupling coefficient.
REAL_WORKSPACE_ROW_DROP_COUPLER_FIELD_CROSS_COUPLING = 4
# 1/tau = 1/sqrt(1-|rho|^2) (1) - inverse scatterer field transmission.
REAL_WORKSPACE_ROW_INVERSE_SCATTERER_FIELD_TRANSMISSION = 5
# d(1/tau)/dphi (1/rad) - inverse field transmission derivative with respect to round-trip phase.
REAL_WORKSPACE_ROW_INVERSE_SCATTERER_FIELD_TRANSMISSION_ROUND_TRIP_PHASE_DERIVATIVE = 6
# sum(|u|^2) (W) - clockwise power entering the quarter-ring segments.
REAL_WORKSPACE_ROW_CLOCKWISE_SEGMENT_INPUT_POWER_SUM = 7
# sum(|v|^2) (W) - counterclockwise power entering the quarter-ring segments.
REAL_WORKSPACE_ROW_COUNTERCLOCKWISE_SEGMENT_INPUT_POWER_SUM = 8
# d(sum(|u|^2))/dphi (W/rad) - round-trip phase derivative of the clockwise segment power sum.
REAL_WORKSPACE_ROW_CLOCKWISE_SEGMENT_INPUT_POWER_SUM_ROUND_TRIP_PHASE_DERIVATIVE = 9
# d(sum(|v|^2))/dphi (W/rad) - round-trip phase derivative of the counterclockwise segment power sum.
REAL_WORKSPACE_ROW_COUNTERCLOCKWISE_SEGMENT_INPUT_POWER_SUM_ROUND_TRIP_PHASE_DERIVATIVE = 10
# first row of the data-level-zero response block (1).
REAL_WORKSPACE_LEVEL_ZERO_RESPONSE_START_ROW = 11
# first row of the data-level-one response block (1).
REAL_WORKSPACE_LEVEL_ONE_RESPONSE_START_ROW = (
    REAL_WORKSPACE_LEVEL_ZERO_RESPONSE_START_ROW + RING_RESPONSE_ROW_COUNT
)
# number of rows in the real workspace, including both response blocks (1).
REAL_WORKSPACE_ROW_COUNT = REAL_WORKSPACE_LEVEL_ONE_RESPONSE_START_ROW + RING_RESPONSE_ROW_COUNT

# verified range for optical power bounds
# max|r_b| (1) - verified backscatter range for the absorption power bounds.
MAX_ROUND_TRIP_BACKSCATTER_FOR_VERIFIED_POWER_BOUNDS = 1.0

# self-heating integer header fields
# N (1) - number of rings.
SELF_HEATING_HEADER_INDEX_RING_COUNT = 0
# L (1) - number of laser lines.
SELF_HEATING_HEADER_INDEX_LASER_LINE_COUNT = 1
# M_backscatter (1) - number of stored backscatter orders.
SELF_HEATING_HEADER_INDEX_BACKSCATTER_ORDER_COUNT = 2
# enabled (1) - evaluate and average both binary data levels.
SELF_HEATING_HEADER_INDEX_TWO_DATA_LEVELS_FLAG = 3
# reserved (1) - must remain zero for exact optics.
SELF_HEATING_HEADER_INDEX_RESERVED_MODEL_FLAG = 4
# reserved (1) - unused legacy mode-order count.
SELF_HEATING_HEADER_INDEX_RESERVED_MODE_ORDER_COUNT = 5
# enabled (1) - write absorbed-power phase and resonance-wavelength derivatives on commit.
SELF_HEATING_HEADER_INDEX_COMMIT_OPTICAL_DERIVATIVES_FLAG = 6
# offset (1) - start of the bus input power block in the packed real array.
SELF_HEATING_HEADER_INDEX_BUS_INPUT_POWER_OFFSET = 7
# offset (1) - start of the current ring input power block in the packed real array.
SELF_HEATING_HEADER_INDEX_CURRENT_RING_INPUT_POWER_OFFSET = 8
# offset (1) - start of the output bus segment transmission in the packed real array.
SELF_HEATING_HEADER_INDEX_BUS_OUTPUT_SEGMENT_TRANSMISSION_OFFSET = 9
# offset (1) - start of the per-ring parameter block in the packed real array.
SELF_HEATING_HEADER_INDEX_RING_PARAMETER_BLOCK_OFFSET = 10
# offset (1) - start of the per-ring and per-line parameter block in the packed real array.
SELF_HEATING_HEADER_INDEX_RING_LINE_PARAMETER_BLOCK_OFFSET = 11
# offset (1) - start of the complex backscatter coefficient block in the packed real array.
SELF_HEATING_HEADER_INDEX_BACKSCATTER_COEFFICIENTS_OFFSET = 12
# offset (1) - start of the per-ring and per-line outputs in the packed real array.
SELF_HEATING_HEADER_INDEX_RING_LINE_OUTPUTS_OFFSET = 13
# offset (1) - start of the per-ring outputs in the packed real array.
SELF_HEATING_HEADER_INDEX_RING_OUTPUTS_OFFSET = 14
# offset (1) - start of the per-line outputs in the packed real array.
SELF_HEATING_HEADER_INDEX_LASER_LINE_OUTPUTS_OFFSET = 15
# offset (1) - start of the complex field workspace in the packed real array.
SELF_HEATING_HEADER_INDEX_COMPLEX_WORKSPACE_OFFSET = 16
# offset (1) - start of the real workspace in the packed real array.
SELF_HEATING_HEADER_INDEX_REAL_WORKSPACE_OFFSET = 17
# offset (1) - start of the coupler cache in the packed real array.
SELF_HEATING_HEADER_INDEX_COUPLER_CACHE_OFFSET = 18
# reserved offset (1) - retired Airy cache block retained for packed-layout compatibility.
SELF_HEATING_HEADER_INDEX_AIRY_CACHE_OFFSET = 19
# offset (1) - start of the no-backscatter flags followed by the round-trip loss cache in the packed real array.
SELF_HEATING_HEADER_INDEX_NO_BACKSCATTER_FLAGS_OFFSET = 20
# offset (1) - start of the last evaluation cache followed by evaluation metadata in the packed real array.
SELF_HEATING_HEADER_INDEX_LAST_EVALUATION_CACHE_OFFSET = 21
# offset (1) - start of the far-line hold cache in the packed real array.
SELF_HEATING_HEADER_INDEX_FAR_LINE_CACHE_OFFSET = 22
# enabled (1) - allow first-order holds of far-detuned laser lines.
SELF_HEATING_HEADER_INDEX_FAR_LINE_HOLD_ENABLED_FLAG = 23
# field count (1) - bus order and per-ring bus positions follow this fixed header.
SELF_HEATING_HEADER_FIXED_FIELD_COUNT = 24

# self-heating rows per ring
# T_bus_segment (1) - power transmission of the bus segment before the ring.
SELF_HEATING_RING_ROW_BUS_SEGMENT_POWER_TRANSMISSION = 0
# Pr(bit=1) (1) - fraction of binary ones.
SELF_HEATING_RING_ROW_PROBABILITY_OF_ONE = 1
# T_reference (K) - reference temperature of the phase rows.
SELF_HEATING_RING_ROW_REFERENCE_TEMPERATURE = 2
# dlambda_res/dT (m/K) - thermal tuning slope at the reference temperature.
SELF_HEATING_RING_ROW_REFERENCE_THERMAL_TUNING_SLOPE = 3
# beta (1/K) - relative curvature of the thermal tuning law.
SELF_HEATING_RING_ROW_RELATIVE_THERMAL_TUNING_CURVATURE = 4
# number of self-heating parameter rows per ring (1).
SELF_HEATING_RING_ROW_COUNT = 5

# self-heating rows per ring and laser line
# phi_zero,reference (rad) - level-zero reference round-trip phase before coupler phase.
SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_REFERENCE_ROUND_TRIP_PHASE = 0
# phi_one,reference (rad) - level-one reference round-trip phase before coupler phase.
SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_REFERENCE_ROUND_TRIP_PHASE = 1
# phi_zero (rad) - level-zero round-trip phase at the trial or committed temperature.
SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_CURRENT_ROUND_TRIP_PHASE = 2
# phi_one (rad) - level-one round-trip phase at the trial or committed temperature.
SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_CURRENT_ROUND_TRIP_PHASE = 3
# a_zero (1) - retained round-trip field amplitude at data level zero.
SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_ROUND_TRIP_FIELD_TRANSMISSION = 4
# a_one (1) - retained round-trip field amplitude at data level one.
SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_ROUND_TRIP_FIELD_TRANSMISSION = 5
# eta_zero (1) - absorbed share of intrinsic loss at data level zero.
SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_ABSORBED_SHARE_OF_INTRINSIC_LOSS = 6
# eta_one (1) - absorbed share of intrinsic loss at data level one.
SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_ABSORBED_SHARE_OF_INTRINSIC_LOSS = 7
# psi (rad) - ring-side coupler phase.
SELF_HEATING_RING_LINE_ROW_COUPLER_PHASE = 8
# K_through^2 (1) - through-coupler power coupling at this laser line.
SELF_HEATING_RING_LINE_ROW_THROUGH_COUPLER_POWER_COUPLING = 9
# K_drop^2 (1) - drop-coupler power coupling at this laser line.
SELF_HEATING_RING_LINE_ROW_DROP_COUPLER_POWER_COUPLING = 10
# T_round_trip (s) - round-trip group delay.
SELF_HEATING_RING_LINE_ROW_ROUND_TRIP_GROUP_DELAY = 11
# dphi/dlambda_res (rad/m) - round-trip phase derivative for a resonance wavelength shift.
SELF_HEATING_RING_LINE_ROW_PHASE_RESONANCE_WAVELENGTH_DERIVATIVE = 12
# number of self-heating parameter rows per ring and laser line (1).
SELF_HEATING_RING_LINE_ROW_COUNT = 13

# shared views returned by unpack_self_heating_optical_state
# p_in (L,) - power per line at the bus input (W)
SELF_HEATING_VIEW_INDEX_BUS_INPUT_POWER = 0
# bus_t2 (N,) - power transmission of the bus segment before each ring (1)
SELF_HEATING_VIEW_INDEX_BUS_SEGMENT_POWER_TRANSMISSION = 1
# bus_t2_out (1,) - transmission of the segment after the last ring (1)
SELF_HEATING_VIEW_INDEX_BUS_OUTPUT_SEGMENT_POWER_TRANSMISSION = 2
# p1 (N,) - ones density (1)
SELF_HEATING_VIEW_INDEX_PROBABILITY_OF_ONE = 3
# phi0_ref (N, L) - round-trip phase at T_ref, level 0 (rad)
SELF_HEATING_VIEW_INDEX_LEVEL_ZERO_REFERENCE_ROUND_TRIP_PHASE = 4
# phi1_ref (N, L) - same, level 1 (rad)
SELF_HEATING_VIEW_INDEX_LEVEL_ONE_REFERENCE_ROUND_TRIP_PHASE = 5
# phi0 (N, L) - phase at the trial / committed temperature, level 0 (rad)
SELF_HEATING_VIEW_INDEX_LEVEL_ZERO_CURRENT_ROUND_TRIP_PHASE = 6
# phi1 (N, L) - same, level 1 (rad)
SELF_HEATING_VIEW_INDEX_LEVEL_ONE_CURRENT_ROUND_TRIP_PHASE = 7
# a0 (N, L) - round-trip amplitude, level 0 (1)
SELF_HEATING_VIEW_INDEX_LEVEL_ZERO_ROUND_TRIP_FIELD_TRANSMISSION = 8
# a1 (N, L) - round-trip amplitude, level 1 (1)
SELF_HEATING_VIEW_INDEX_LEVEL_ONE_ROUND_TRIP_FIELD_TRANSMISSION = 9
# eta0 (N, L) - heat fraction, level 0 (1)
SELF_HEATING_VIEW_INDEX_LEVEL_ZERO_ABSORBED_SHARE_OF_INTRINSIC_LOSS = 10
# eta1 (N, L) - heat fraction, level 1 (1)
SELF_HEATING_VIEW_INDEX_LEVEL_ONE_ABSORBED_SHARE_OF_INTRINSIC_LOSS = 11
# psi (N, L) - ring-side coupler phase (rad)
SELF_HEATING_VIEW_INDEX_COUPLER_PHASE = 12
# k2_th (N, L) - power coupling K^2 of the through coupler at the line (1)
SELF_HEATING_VIEW_INDEX_THROUGH_COUPLER_POWER_COUPLING = 13
# k2_dr (N, L) - power coupling K^2 of the drop coupler at the line (1)
SELF_HEATING_VIEW_INDEX_DROP_COUPLER_POWER_COUPLING = 14
# t_rt (N, L) - round-trip group delay (s)
SELF_HEATING_VIEW_INDEX_ROUND_TRIP_GROUP_DELAY = 15
# dphi_dlamres (N, L) - d phi / d lambda_res (rad/m)
SELF_HEATING_VIEW_INDEX_PHASE_RESONANCE_WAVELENGTH_DERIVATIVE = 16
# t_ref (N,) - reference temperature of the phase rows (K)
SELF_HEATING_VIEW_INDEX_REFERENCE_TEMPERATURE = 17
# s0 (N,) - d lambda_res / dT at T_ref (m/K)
SELF_HEATING_VIEW_INDEX_REFERENCE_THERMAL_TUNING_SLOPE = 18
# beta (N,) - relative curvature of the thermo-optic law (1/K)
SELF_HEATING_VIEW_INDEX_RELATIVE_THERMAL_TUNING_CURVATURE = 19
# rb (N, M_b) complex - backscatter per order (1)
SELF_HEATING_VIEW_INDEX_ROUND_TRIP_BACKSCATTER_COEFFICIENTS = 20
# p_cur (L,) - walk state, input power of the ring being solved (W)
SELF_HEATING_VIEW_INDEX_CURRENT_RING_INPUT_POWER = 21
# ring and line output array (mixed units) - rows use the ring-line output indices.
SELF_HEATING_VIEW_INDEX_RING_LINE_OUTPUTS = 22
# ring output array (mixed units) - rows use the per-ring output indices.
SELF_HEATING_VIEW_INDEX_RING_OUTPUTS = 23
# laser-line output array (mixed units) - rows use the per-line output indices.
SELF_HEATING_VIEW_INDEX_LASER_LINE_OUTPUTS = 24
# complex workspace (mixed units) - fields use the complex-workspace row indices.
SELF_HEATING_VIEW_INDEX_COMPLEX_WORKSPACE = 25
# real workspace (mixed units) - fields use the real-workspace row indices.
SELF_HEATING_VIEW_INDEX_REAL_WORKSPACE = 26
# ring indices in bus order (1).
SELF_HEATING_VIEW_INDEX_RING_INDICES_IN_BUS_ORDER = 27
# bus position of each ring (1).
SELF_HEATING_VIEW_INDEX_BUS_POSITIONS_BY_RING = 28
# stable field names in shared-view tuple order for existing interfaces.
SELF_HEATING_VIEW_FIELD_NAMES = (
    "p_in",
    "bus_t2",
    "bus_t2_out",
    "p1",
    "phi0r",
    "phi1r",
    "phi0",
    "phi1",
    "a0",
    "a1",
    "e0",
    "e1",
    "psi",
    "k2t",
    "k2d",
    "trt",
    "dpl",
    "tref",
    "s0",
    "beta",
    "rb",
    "p_cur",
    "out_rl",
    "out_r",
    "out_l",
    "cw",
    "fw",
    "bus_order",
    "bus_pos",
)

# number of fields in one coupler cache entry (1).
COUPLER_CACHE_FIELD_COUNT = 8

# reserved size (1) - retained for packed-layout compatibility.
AIRY_CACHE_FIELD_COUNT = 13


# last optical evaluation cache fields
# cos of the evaluated phase (1)
LAST_EVALUATION_CACHE_FIELD_TOTAL_PHASE_COSINE = 0
# 1/den (1)
LAST_EVALUATION_CACHE_FIELD_INVERSE_RESONANCE_DENOMINATOR = 1
# |u(0+)|^2 = k1^2/den, circulating power per watt in (1)
LAST_EVALUATION_CACHE_FIELD_CIRCULATING_POWER_RATIO = 2
# loss per watt in (1)
LAST_EVALUATION_CACHE_FIELD_INTRINSIC_LOSS_POWER_FRACTION = 3
# d(loss per watt)/d phi (1/rad)
LAST_EVALUATION_CACHE_FIELD_INTRINSIC_LOSS_FRACTION_ROUND_TRIP_PHASE_DERIVATIVE = 4
# number of response fields per data level and laser line in the last evaluation cache (1).
LAST_EVALUATION_CACHE_FIELD_COUNT = 5

# last optical evaluation metadata
# last ring index (1) - minus one means no cached evaluation.
LAST_EVALUATION_METADATA_FIELD_RING_INDEX = 0
# T_last (K) - ring temperature at the last optical evaluation.
LAST_EVALUATION_METADATA_FIELD_TEMPERATURE = 1
# Delta_lambda_last (m) - resonance shift at the last optical evaluation.
LAST_EVALUATION_METADATA_FIELD_RESONANCE_WAVELENGTH_SHIFT = 2
# number of metadata fields after the last evaluation response blocks (1).
LAST_EVALUATION_METADATA_FIELD_COUNT = 3

# number of round-trip loss cache fields per data level (1).
ROUND_TRIP_LOSS_CACHE_FIELD_COUNT_PER_DATA_LEVEL = 4
# number of round-trip loss cache fields for both data levels (1).
ROUND_TRIP_LOSS_CACHE_TOTAL_FIELD_COUNT = 2 * ROUND_TRIP_LOSS_CACHE_FIELD_COUNT_PER_DATA_LEVEL


# number of fields in one far-line hold cache entry (1).
FAR_LINE_CACHE_FIELD_COUNT = 19



# array type aliases describe storage precision and dimensions.
# shape metadata describes array dimensions.
# runtime guards enforce workspace dtype and capacity.
Float64Array: TypeAlias = NDArray[np.float64]
Complex128Array: TypeAlias = NDArray[np.complex128]
Float64Vector: TypeAlias = Annotated[np.ndarray[tuple[int], np.dtype[np.float64]], "shape (length,)"]
Float64Matrix: TypeAlias = Annotated[np.ndarray[tuple[int, int], np.dtype[np.float64]], "shape (rows, columns)"]
Float64Tensor: TypeAlias = Annotated[np.ndarray[tuple[int, int, int], np.dtype[np.float64]], "shape (rows, rings, lines)"]
Complex128Vector: TypeAlias = Annotated[np.ndarray[tuple[int], np.dtype[np.complex128]], "shape (length,)"]
Complex128Matrix: TypeAlias = Annotated[np.ndarray[tuple[int, int], np.dtype[np.complex128]], "shape (rows, columns)"]
Int64Vector: TypeAlias = Annotated[np.ndarray[tuple[int], np.dtype[np.int64]], "shape (length,)"]
RealValue: TypeAlias = float | np.floating | NDArray[np.floating]
ComplexValue: TypeAlias = complex | np.complexfloating | NDArray[np.complexfloating]
NumericValue: TypeAlias = int | float | complex | np.number | NDArray[np.number]
NumericArray: TypeAlias = NDArray[np.generic]
ComplexWorkspace: TypeAlias = Annotated[Complex128Matrix, "capacity (COMPLEX_WORKSPACE_ROW_COUNT, line_count)"]
RealWorkspace: TypeAlias = Annotated[Float64Matrix, "capacity (REAL_WORKSPACE_ROW_COUNT, line_count)"]
RingResponseBlock: TypeAlias = Annotated[Float64Matrix, "capacity (RING_RESPONSE_ROW_COUNT, line_count)"]
RingLineOutputs: TypeAlias = Annotated[Float64Tensor, "shape (RING_LINE_OUTPUT_ROW_COUNT, ring_count, line_count)"]
RingOutputs: TypeAlias = Annotated[Float64Matrix, "shape (RING_OUTPUT_ROW_COUNT, ring_count)"]
LaserLineOutputs: TypeAlias = Annotated[Float64Matrix, "shape (LASER_LINE_OUTPUT_ROW_COUNT, line_count)"]
PackedOpticalValues: TypeAlias = Annotated[Float64Vector, "mixed-unit values, workspaces, caches, and outputs"]
PackedOpticalIndices: TypeAlias = Annotated[Int64Vector, "integer header fields, offsets, and bus ordering"]
EnergyDecayRates: TypeAlias = tuple[
    RealValue,
    RealValue,
    RealValue,
]
EnergyDecayRatesWithBackscatter: TypeAlias = tuple[
    RealValue,
    RealValue,
    RealValue,
    ComplexValue,
]
OpticalPowerAndDerivative: TypeAlias = tuple[
    float,
    float,
]
RingFieldSolution: TypeAlias = tuple[
    complex,
    complex,
    complex,
    complex,
    float,
    float,
    float,
]
RingFieldSolutionWithDerivatives: TypeAlias = tuple[
    complex,
    complex,
    complex,
    complex,
    float,
    float,
    float,
    float,
    float,
    float,
]
WorkspaceBuffers: TypeAlias = tuple[
    ComplexWorkspace,
    RealWorkspace,
    Float64Vector,
]
BusOutputBuffers: TypeAlias = tuple[
    RingLineOutputs,
    RingOutputs,
    LaserLineOutputs,
]
SelfHeatingInputViews: TypeAlias = tuple[
    Float64Vector,
    Float64Vector,
    Float64Matrix,
    Float64Tensor,
    Complex128Matrix,
    ComplexWorkspace,
    RealWorkspace,
]
SelfHeatingOutputViews: TypeAlias = BusOutputBuffers
# tuple positions match SELF_HEATING_VIEW_FIELD_NAMES and SELF_HEATING_VIEW_INDEX_*.
SelfHeatingViews: TypeAlias = tuple[
    Float64Vector,
    Float64Vector,
    Float64Vector,
    Float64Vector,
    Float64Matrix,
    Float64Matrix,
    Float64Matrix,
    Float64Matrix,
    Float64Matrix,
    Float64Matrix,
    Float64Matrix,
    Float64Matrix,
    Float64Matrix,
    Float64Matrix,
    Float64Matrix,
    Float64Matrix,
    Float64Matrix,
    Float64Vector,
    Float64Vector,
    Float64Vector,
    Complex128Matrix,
    Float64Vector,
    RingLineOutputs,
    RingOutputs,
    LaserLineOutputs,
    ComplexWorkspace,
    RealWorkspace,
    Int64Vector,
    Int64Vector,
]
# return order follows the output contract of _validate_and_broadcast_bus_inputs.
ValidatedBusInputs: TypeAlias = tuple[
    Float64Vector,
    Int64Vector,
    Float64Vector,
    float,
    Float64Vector,
    Float64Matrix,
    Float64Matrix,
    Float64Matrix,
    Float64Matrix,
    Float64Matrix,
    Float64Matrix,
    Float64Matrix,
    Float64Matrix,
    Float64Matrix,
    Float64Matrix,
    Complex128Matrix,
    bool,
]

# binary data uses one required probability array and optional level-one optical inputs.
class BinaryDataLevels(TypedDict):
    p1: ArrayLike
    phi1: NotRequired[ArrayLike]
    a1: NotRequired[ArrayLike]
    eta1: NotRequired[ArrayLike]


# mapping between exact ring and TCMT rates + helper functions - - - -


## tcmt - total coupled mode theory
# map round-trip losses to temporal coupled-mode theory energy decay rates.
# kappa_i = -2*ln(a)/T_rt.
# kappa_e = -ln(1-K_th^2)/T_rt.
# kappa_d = -ln(1-K_dr^2)/T_rt.
# x
@njit(cache=True)
def energy_decay_rates_from_ring(
    k2_th: float,  # k2_th = K_th^2 (1) - through-coupler power fraction - type float scalar
    k2_dr: float,  # k2_dr = K_dr^2 (1) - drop-coupler power fraction - type float scalar
    a_rt: float,  # a_rt = a (1) - field amplitude retained per round trip - type float scalar
    t_rt_s: float,  # t_rt_s = T_rt (s) - round-trip group delay - type float scalar
) -> tuple[
    float,  # intrinsic_decay_rate = kappa_i (1/s) - intrinsic energy loss - type float scalar
    float,  # through_coupling_decay_rate = kappa_e (1/s) - energy leakage to the bus - type float scalar
    float,  # drop_coupling_decay_rate = kappa_d (1/s) - energy leakage to the drop guide - type float scalar
]:
    ### given a loss a, how do i model this as an exponential decay? calculate the correct constants?
    intrinsic_decay_rate = -2.0 * math.log(a_rt) / t_rt_s
    through_coupling_decay_rate = -math.log1p(-k2_th) / t_rt_s
    drop_coupling_decay_rate = -math.log1p(-k2_dr) / t_rt_s
    return (
        intrinsic_decay_rate,
        through_coupling_decay_rate,
        drop_coupling_decay_rate,
    )
### light in a ring depends on how much energy is staying, how much energy is not going through, and how much energy is not dropping

### convert coupling between cc and ccw into rate - rad/s
# convert round-trip backscatter to the complex mode-coupling rate.
# intuition - divide by time to get units per time instead of per trip
# x
@njit(cache=True)
def tcmt_backscatter_rate(
    rb: NumericValue,  # rb = r_b (1) - complex round-trip coupling - type numeric scalar or NumPy array
    t_rt_s: RealValue,  # t_rt_s = T_rt (s) - round-trip group delay - type real scalar or real NumPy array
) -> (
    NumericValue  # return = g (rad/s), with g = r_b/T_rt - complex mode-coupling rate - type numeric scalar or NumPy array
):
    return rb / t_rt_s

# array version of energy decay from ring, for mult rings at once
# convert exact ring parameters to broadcast arrays of TCMT rates.
# align array axes for NumPy broadcasting.
# for order n, Delta = (phi + psi - 2*pi*n)/T_rt (rad/s).
# x
def exact_to_tcmt(
    k2_th: ArrayLike,  # k2_th = K_th^2 (1) - through-coupler power fraction - type scalar, sequence, or NumPy array
    k2_dr: ArrayLike,  # k2_dr = K_dr^2 (1) - drop-coupler power fraction - type scalar, sequence, or NumPy array
    a_rt: ArrayLike,  # a_rt = a (1) - retained round-trip field amplitude - type scalar, sequence, or NumPy array
    t_rt_s: ArrayLike,  # t_rt_s = T_rt (s) - round-trip group delay - type scalar, sequence, or NumPy array
    rb: ArrayLike | None = None,  # rb = r_b (1) - optional complex round-trip backscatter - type scalar, sequence, NumPy array, or None
) -> (
    tuple[
        RealValue,  # intrinsic_decay_rate = kappa_i (1/s) - intrinsic energy decay rate - type float64 scalar or float64 ndarray, broadcast input shape
        RealValue,  # through_coupling_decay_rate = kappa_e (1/s) - energy leakage rate to the through bus - type float64 scalar or float64 ndarray, broadcast input shape
        RealValue,  # drop_coupling_decay_rate = kappa_d (1/s) - energy leakage rate to the drop bus - type float64 scalar or float64 ndarray, broadcast input shape
    ]
    | tuple[
        RealValue,  # intrinsic_decay_rate = kappa_i (1/s) - intrinsic energy decay rate - type float64 scalar or float64 ndarray, broadcast input shape
        RealValue,  # through_coupling_decay_rate = kappa_e (1/s) - energy leakage rate to the through bus - type float64 scalar or float64 ndarray, broadcast input shape
        RealValue,  # drop_coupling_decay_rate = kappa_d (1/s) - energy leakage rate to the drop bus - type float64 scalar or float64 ndarray, broadcast input shape
        ComplexValue,  # beta_hz = g/(2*pi) (Hz) - returned only when rb is supplied - type complex scalar or complex NumPy array
    ]
):
    k2_th, k2_dr, a_rt, t_rt_s = np.broadcast_arrays(
        *(np.asarray(input_value, dtype=float) for input_value in (k2_th, k2_dr, a_rt, t_rt_s))
    )
    intrinsic_decay_rate = -2.0 * np.log(a_rt) / t_rt_s
    through_coupling_decay_rate = -np.log1p(-k2_th) / t_rt_s
    drop_coupling_decay_rate = -np.log1p(-k2_dr) / t_rt_s
    if rb is None:
        return (
            intrinsic_decay_rate,
            through_coupling_decay_rate,
            drop_coupling_decay_rate,
        )
    beta_hz = np.asarray(rb, dtype=complex) / (TWO_PI * t_rt_s)
    return (
        intrinsic_decay_rate,
        through_coupling_decay_rate,
        drop_coupling_decay_rate,
        beta_hz,
    )

# reverse func of above
# recover exact ring parameters from energy decay rates.
# x
def tcmt_to_exact(
    kappa_i: ArrayLike,  # kappa_i = kappa_i (1/s) - intrinsic energy loss - type scalar, sequence, or NumPy array
    kappa_e: ArrayLike,  # kappa_e = kappa_e (1/s) - leakage to the bus - type scalar, sequence, or NumPy array
    kappa_d: ArrayLike,  # kappa_d = kappa_d (1/s) - leakage to the drop guide - type scalar, sequence, or NumPy array
    t_rt_s: ArrayLike,  # t_rt_s = T_rt (s) - round-trip group delay - type scalar, sequence, or NumPy array
) -> tuple[
    RealValue,  # return[0] = K_th^2 (1), with K_th^2 = 1-exp(-kappa_e*T_rt) - through-coupler power fraction - type float64 scalar or float64 ndarray, broadcast input shape
    RealValue,  # return[1] = K_dr^2 (1), with K_dr^2 = 1-exp(-kappa_d*T_rt) - drop-coupler power fraction - type float64 scalar or float64 ndarray, broadcast input shape
    RealValue,  # return[2] = a (1), with a = exp(-kappa_i*T_rt/2) - retained round-trip field amplitude - type float64 scalar or float64 ndarray, broadcast input shape
]:
    kappa_i, kappa_e, kappa_d, t_rt_s = (
        np.asarray(x, dtype=float) for x in (kappa_i, kappa_e, kappa_d, t_rt_s)
    )
    return (
        -np.expm1(-kappa_e * t_rt_s),
        -np.expm1(-kappa_d * t_rt_s),
        np.exp(-0.5 * kappa_i * t_rt_s),
    )


### (todo) not sure rn how this works yet
# convert a complex backscatter rate to round-trip coupling.
# x
@njit(cache=True)
def backscatter_hz_to_round_trip_coupling(
    beta_hz: NumericValue,  # beta_hz = g/(2*pi) (Hz) - complex coupling rate - type numeric scalar or NumPy array
    fsr_hz: RealValue,  # fsr_hz = FSR (Hz) - free spectral range - type real scalar or real NumPy array
) -> (
    NumericValue  # return = r_b (1), with r_b = 2*pi*beta_hz/FSR - round-trip backscatter coupling - type numeric scalar or NumPy array
):
    return TWO_PI * beta_hz / fsr_hz

### how does change in resonance lead to change in phase?
# find the phase change caused by a resonance shift at a fixed laser wavelength.
# dphi/dlambda_r = 2*pi*c*T_rt/lambda_L^2 is positive.
# the derivative with respect to laser wavelength has the opposite sign.
# x
@njit(cache=True)
def phase_sensitivity_to_resonance_shift(
    t_rt_s: RealValue,  # t_rt_s = T_rt (s) - round-trip group delay - type real scalar or real NumPy array
    lam_m: RealValue,  # lam_m = lambda_L (m) - laser wavelength - type real scalar or real NumPy array
) -> (
    RealValue  # return = dphi/dlambda_r (rad/m) - phase change per resonance-wavelength shift - type real scalar or real NumPy array
):
    return TWO_PI * C0 * t_rt_s / (lam_m * lam_m)


### fraction of 1s is p1, so this returns the avg data value between 0 and 1
# average a quantity over binary data.
# mean(x) = p1*x1 + (1-p1)*x0.
# x
@njit(cache=True)
def average_binary_data_levels(
    p1: RealValue,  # p1 = Pr(bit=1) (1) - fraction of ones - type real scalar or real NumPy array
    x1: NumericValue,  # x1 = x_1 (same units as x0) - value at level one - type numeric scalar or NumPy array
    x0: NumericValue,  # x0 = x_0 (same units as x1) - value at level zero - type numeric scalar or NumPy array
) -> (
    NumericValue  # return = mean(x) (same units as x1 and x0) - binary-data average - type numeric scalar or NumPy array
):
    return p1 * x1 + (1.0 - p1) * x0


# exact ring - per-line kernels - - - - -
# wrap an absolute resonance order into a repeating table
# assume positive passed in idx
# x
@njit(cache=True)
def _wrap_order_index(
    order_index: int,  # order_index = n (1) - signed absolute resonance order - type int scalar
    order_count: int,  # order_count = M_b (1) - positive number of table entries - type int scalar
) -> (
    int  # return = n mod M_b (1) - index in the repeating order table - type int scalar
):
    wrapped_index = order_index % order_count
    if wrapped_index < 0:
        wrapped_index += order_count
    return wrapped_index


# interpolate each scatterer's complex amplitude reflection coefficient and phase derivative.
# APPROXIMATION: the two scatterers lie at one-quarter and three-quarters of a ring trip.
# n=floor(phi/(2*pi)) selects the lower order, and theta=phi-2*pi*n (rad) locates phase within that order.
# rho_n = -(i/2)*(-1)^n*conj(r_b[n]) maps each table entry to a scatterer reflection coefficient.
# rho(phi) = cos^2(theta/4)*rho_n + sin^2(theta/4)*rho_(n+1).
# wrap indices by the backscatter table length and retain the absolute order's alternating sign.
# both interpolation weights have zero phase slope at each resonance.
# x
@njit(cache=True)
def interpolate_scatterer_reflection_and_phase_derivative(
    round_trip_phase_rad: float,  # round_trip_phase_rad = phi (rad) - absolute round-trip phase including coupler phase - type float scalar
    round_trip_backscatter: Complex128Vector,  # round_trip_backscatter = r_b[n] (1) - complex round-trip backscatter coupling by resonance order - type complex128 ndarray, shape (backscatter_order_count,)
) -> tuple[
    complex,  # scatterer_reflection = rho(phi) (1) - complex field reflection coefficient - type complex scalar
    complex,  # d_scatterer_reflection_dphi = d rho/d phi (1/rad) - reflection coefficient's phase derivative - type complex scalar
]:
    backscatter_order_count = round_trip_backscatter.shape[0]
    lower_order = int(
        np.floor(round_trip_phase_rad / TWO_PI)
    )  # floor calculation to find lowest resonance order
    phase_within_order_rad = round_trip_phase_rad - TWO_PI * lower_order
    quarter_phase_cos = math.cos(0.25 * phase_within_order_rad)
    quarter_phase_sin = math.sin(0.25 * phase_within_order_rad)

    # weights w
    lower_order_weight = quarter_phase_cos * quarter_phase_cos
    upper_order_weight = quarter_phase_sin * quarter_phase_sin
    d_upper_weight_dphi = 0.25 * math.sin(0.5 * phase_within_order_rad)
    # dw_hi/dphi = sin(theta/2)/4, and dw_lo/dphi = -dw_hi/dphi
    lower_order_sign = 1.0 if _wrap_order_index(lower_order, 2) == 0 else -1.0  # (-1) ^n

    # convert r.b.n into p.n
    lower_order_reflection = (
        -0.5j
        * lower_order_sign
        * round_trip_backscatter[_wrap_order_index(lower_order, backscatter_order_count)].conjugate()
    )
    # convert r.b.n+1 into p.n+1
    upper_order_reflection = (
        0.5j
        * lower_order_sign
        * round_trip_backscatter[_wrap_order_index(lower_order + 1, backscatter_order_count)].conjugate()
    )
    # (-1)^(n+1) = -(-1)^n

    # p(theta) = (1-w)(...) + (w)(...)
    scatterer_reflection = (
        lower_order_weight * lower_order_reflection + upper_order_weight * upper_order_reflection
    )
    # approx. slope
    d_scatterer_reflection_dphi = d_upper_weight_dphi * (upper_order_reflection - lower_order_reflection)
    return (
        scatterer_reflection,
        d_scatterer_reflection_dphi,
    )


# return amplitude.real**2 + amplitude.imag**2 - for a travelling field, this is power.
# (todo) move to general helper function file
# A uses 1 for a coefficient, sqrt(W) for a port field, or sqrt(J) for a TCMT mode.
# x
@njit(cache=True)
def _squared_magnitude(
    amplitude: complex,  # amplitude = A (input amplitude units) - complex field, mode, or normalized coefficient - type complex scalar
) -> (
    float  # return = |A|^2 (squared input units) - squared amplitude magnitude - type float scalar
):
    return amplitude.real * amplitude.real + amplitude.imag * amplitude.imag


# differentiate squared magnitude - 2*Re(conj(amplitude)*d_amplitude_dphi).
# (todo) move to helper file
# x
@njit(cache=True)
def _squared_magnitude_derivative(
    amplitude: complex,  # amplitude = A (input amplitude units) - complex field, mode, or normalized coefficient - type complex scalar
    d_amplitude_dphi: complex,  # d_amplitude_dphi = dA/dphi (input amplitude units/rad) - phase derivative of amplitude - type complex scalar
) -> (
    float  # return = d|A|^2/dphi (squared input units/rad) - phase derivative of squared amplitude magnitude - type float scalar
):
    return 2.0 * (amplitude.real * d_amplitude_dphi.real + amplitude.imag * d_amplitude_dphi.imag)


# integrate stored energy and loss along one quarter of the ring.
# a lossless segment retains finite energy and loses no power.
# x
@njit(cache=True)
def calculate_quarter_ring_energy_per_power_and_loss_fraction(
    round_trip_field_loss_exponent: float,  # round_trip_field_loss_exponent = -ln(a) (1) - full-trip field attenuation - type float scalar
    round_trip_delay_s: float,  # round_trip_delay_s = T_rt (s) - full-trip group delay - type float scalar
) -> tuple[
    float,  # return[0] = U_segment/P_entering (J/W) - energy per entering watt - type float scalar
    float,  # return[1] = P_lost/P_entering (1) - lost power fraction - type float scalar
]:
    quarter_power_loss_exponent = 0.5 * round_trip_field_loss_exponent # sqiggle/2 = q
    quarter_ring_loss_fraction = -math.expm1(-quarter_power_loss_exponent) # loss fraction
    if quarter_power_loss_exponent > 1e-300:
        mean_quarter_power_ratio = quarter_ring_loss_fraction / quarter_power_loss_exponent
    else:
        mean_quarter_power_ratio = 1.0
    return (
        0.25 * round_trip_delay_s * mean_quarter_power_ratio,
        quarter_ring_loss_fraction,
    )


# exact steady state of one ring at one line - the plain form of the loop, kept as the reference.
# nothing in the package calls it (the structure is in the evaluate_exact_ring_response comments).
# the derivative solver carries the same loop and its phase derivatives.
# the staged solver evaluates all wavelengths for unit forward input.
# tests/verification/test_optics.py checks that it agrees with `solve_exact_ring_fields_and_phase_derivatives`.
# solve x=M*x+c for the clockwise and counterclockwise fields at z=0+.
# M=C1*H*C2*H maps the fields through one round trip.
# H=Q*S*Q maps the fields through half a ring.
# Q=diag(p,1/p) propagates a quarter trip, with p=exp(-(xi+i*phi)/4).
# P2=p^2 and H=[[P2,-conj(rho)],[-rho,1/P2]]/tau.
# C_j=diag(t_j,1/t_j) maps fields through coupler j.
# c contains the bus-input drive terms.
# one 2x2 solve, then one pass for fields and energies.
# (todo) convert to numpy matmul to reduce code size
# x
@njit(cache=True)
def solve_exact_ring_fields_and_energy(
    round_trip_phase_rad: float,  # round_trip_phase_rad = phi (rad) - round-trip phase including coupler phase - type float scalar
    round_trip_field_loss_exponent: float,  # round_trip_field_loss_exponent = -ln(a_rt) (1) - round-trip field attenuation exponent - type float scalar
    through_self_coupling: float,  # through_self_coupling = t_th (1) - through-coupler field self-coupling - type float scalar
    through_cross_coupling: float,  # through_cross_coupling = K_th (1) - through-coupler field cross-coupling - type float scalar
    drop_self_coupling: float,  # drop_self_coupling = t_dr (1) - drop-coupler field self-coupling - type float scalar
    drop_cross_coupling: float,  # drop_cross_coupling = K_dr (1) - drop-coupler field cross-coupling - type float scalar
    round_trip_delay_s: float,  # round_trip_delay_s = T_rt (s) - round-trip group delay - type float scalar
    round_trip_backscatter: Complex128Vector,  # round_trip_backscatter = r_b[n] (1) - complex round-trip backscatter coupling by order - type complex128 ndarray, shape (backscatter_order_count,)
    input_field: complex,  # input_field = s_in,0 (sqrt(W)) - incoming field at the input port - type complex scalar
    backward_through_field: complex,  # backward_through_field = s_in,1 (sqrt(W)) - incoming field at the through port - type complex scalar
    backward_drop_field: complex,  # backward_drop_field = s_in,2 (sqrt(W)) - incoming field at the drop port - type complex scalar
    add_input_field: complex,  # add_input_field = s_in,3 (sqrt(W)) - incoming field at the add port - type complex scalar
) -> tuple[
    complex,  # reflected_field = s_out,0 (sqrt(W)) - reflected field at the input port - type complex scalar
    complex,  # through_field = s_out,1 (sqrt(W)) - through-port field - type complex scalar
    complex,  # drop_field = s_out,2 (sqrt(W)) - clockwise drop-port field - type complex scalar
    complex,  # add_output_field = s_out,3 (sqrt(W)) - counter-clockwise add-port field - type complex scalar
    float,  # return[4] = U_cw (J) - clockwise stored energy - type float scalar
    float,  # return[5] = U_ccw (J) - counter-clockwise stored energy - type float scalar
    float,  # return[6] = P_loss (W) - total intrinsic-loss power - type float scalar
]:
    # s0/s1 enter the input/through ports and drive CW/CCW fields [sqrt(W)].
    # s2/s3 enter the drop/add ports and drive CCW/CW fields [sqrt(W)].
    # rho is each scatterer's reflection coefficient - tau = sqrt(1 - |rho|^2).
    scatterer_reflection, _ = interpolate_scatterer_reflection_and_phase_derivative(round_trip_phase_rad, round_trip_backscatter)
    scatterer_reflection_conj = scatterer_reflection.conjugate()
    inverse_scatterer_field_transmission = 1.0 / math.sqrt(1.0 - _squared_magnitude(scatterer_reflection))

    # quarter-trip attenuation and phase
    quarter_propagation = cmath.exp(-0.25 * (round_trip_field_loss_exponent + 1j * round_trip_phase_rad))
    # undo propagation when tracing CCW light in the clockwise coordinate
    inverse_quarter_propagation = 1.0 / quarter_propagation
    # half-trip attenuation and phase
    half_propagation = quarter_propagation * quarter_propagation
    # inverse half-trip attenuation and phase
    inverse_half_propagation = inverse_quarter_propagation * inverse_quarter_propagation

    # | step | field pair after that step |
    # |---|---|
    # | start | \(x_0\) |
    # | first half-ring | \(Hx_0\) |
    # | drop coupler | \(C_2Hx_0+e\) |
    # | second half-ring | \(HC_2Hx_0+He\) |
    # | input coupler | \(C_1HC_2Hx_0+C_1He+d\) |

    # | incoming field | enters through | couples into |
    # |---|---|---|
    # | `s0` | input port | clockwise ring wave |
    # | `s1` | through port, travelling backward | counterclockwise ring wave |

    # with only the usual forward laser input, s1 = s2 = s3 = 0, e= 0, d only has upper term != 0

    # inverse coupler transmissions for the CCW transfer equations
    inverse_through_self_coupling = 1.0 / through_self_coupling
    inverse_drop_self_coupling = 1.0 / drop_self_coupling

    # H maps [CW, CCW] across one half-ring - h_ij maps component j to i.
    half_transfer_11 = half_propagation * inverse_scatterer_field_transmission
    half_transfer_12 = -scatterer_reflection_conj * inverse_scatterer_field_transmission
    half_transfer_21 = -scatterer_reflection * inverse_scatterer_field_transmission
    half_transfer_22 = inverse_half_propagation * inverse_scatterer_field_transmission

    # K includes both half-rings and the drop coupler, without external drive.
    internal_transfer_11 = (
        half_transfer_11 * drop_self_coupling * half_transfer_11
        + half_transfer_12 * inverse_drop_self_coupling * half_transfer_21
    )  # K = H C2 H
    internal_transfer_12 = (
        half_transfer_11 * drop_self_coupling * half_transfer_12
        + half_transfer_12 * inverse_drop_self_coupling * half_transfer_22
    )
    internal_transfer_21 = (
        half_transfer_21 * drop_self_coupling * half_transfer_11
        + half_transfer_22 * inverse_drop_self_coupling * half_transfer_21
    )
    internal_transfer_22 = (
        half_transfer_21 * drop_self_coupling * half_transfer_12
        + half_transfer_22 * inverse_drop_self_coupling * half_transfer_22
    )
    # e = [e3, e2] adds external drive at the drop coupler.
    cw_drop_coupler_drive = -1j * drop_cross_coupling * add_input_field  # add port -> CW at L/2+
    ccw_drop_coupler_drive = (
        1j * drop_cross_coupling * backward_drop_field * inverse_drop_self_coupling
    )  # drop-port input -> CCW at L/2+
    # c = C1*H*e + d, with d = [-i*k1*s0, i*k1*s1/t1].
    cw_loop_drive = (
        through_self_coupling
        * (half_transfer_11 * cw_drop_coupler_drive + half_transfer_12 * ccw_drop_coupler_drive)
        - 1j * through_cross_coupling * input_field
    )
    ccw_loop_drive = (
        inverse_through_self_coupling
        * (half_transfer_21 * cw_drop_coupler_drive + half_transfer_22 * ccw_drop_coupler_drive)
        + 1j * through_cross_coupling * backward_through_field * inverse_through_self_coupling
    )
    # A = I - M, with M = C1*K - match the fields when the loop closes.
    loop_matrix_11 = 1.0 - through_self_coupling * internal_transfer_11
    loop_matrix_12 = -through_self_coupling * internal_transfer_12
    loop_matrix_21 = -inverse_through_self_coupling * internal_transfer_21
    loop_matrix_22 = 1.0 - inverse_through_self_coupling * internal_transfer_22
    loop_determinant = loop_matrix_11 * loop_matrix_22 - loop_matrix_12 * loop_matrix_21
    # solve A*[CW, CCW] = [cu, cv] just after the input coupler
    cw_field = (
        loop_matrix_22 * cw_loop_drive - loop_matrix_12 * ccw_loop_drive
    ) / loop_determinant  # CW at 0+
    ccw_field = (
        loop_matrix_11 * ccw_loop_drive - loop_matrix_21 * cw_loop_drive
    ) / loop_determinant  # CCW at 0+

    # trace the solved fields around the ring to read outputs and segment powers
    # CW enters a segment at its left end - CCW enters at its right end
    reflected_field = (
        through_self_coupling * backward_through_field - 1j * through_cross_coupling * ccw_field
    )  # reflection out of the input port
    cw_segment_power_sum = _squared_magnitude(cw_field)  # segment 1 (0 -> L/4)
    cw_field = quarter_propagation * cw_field
    ccw_field = ccw_field * inverse_quarter_propagation
    ccw_segment_power_sum = _squared_magnitude(ccw_field)
    cw_field, ccw_field = (
        (cw_field - scatterer_reflection_conj * ccw_field) * inverse_scatterer_field_transmission,
        (ccw_field - scatterer_reflection * cw_field) * inverse_scatterer_field_transmission,
    )  # scatterer A
    cw_segment_power_sum += _squared_magnitude(cw_field)  # segment 2 (L/4 -> L/2)
    cw_field = quarter_propagation * cw_field
    ccw_field = ccw_field * inverse_quarter_propagation
    ccw_segment_power_sum += _squared_magnitude(ccw_field)
    drop_field = (
        drop_self_coupling * add_input_field - 1j * drop_cross_coupling * cw_field
    )  # CW drop (monitor)
    cw_field = drop_self_coupling * cw_field - 1j * drop_cross_coupling * add_input_field
    ccw_field = (
        ccw_field + 1j * drop_cross_coupling * backward_drop_field
    ) * inverse_drop_self_coupling  # CCW at L/2+
    add_output_field = (
        drop_self_coupling * backward_drop_field - 1j * drop_cross_coupling * ccw_field
    )  # CCW drop out of the add port
    cw_segment_power_sum += _squared_magnitude(cw_field)  # segment 3 (L/2 -> 3L/4)
    cw_field = quarter_propagation * cw_field
    ccw_field = ccw_field * inverse_quarter_propagation
    ccw_segment_power_sum += _squared_magnitude(ccw_field)
    cw_field, ccw_field = (
        (cw_field - scatterer_reflection_conj * ccw_field) * inverse_scatterer_field_transmission,
        (ccw_field - scatterer_reflection * cw_field) * inverse_scatterer_field_transmission,
    )  # scatterer B
    cw_segment_power_sum += _squared_magnitude(cw_field)  # segment 4 (3L/4 -> L)
    cw_field = quarter_propagation * cw_field
    ccw_field = ccw_field * inverse_quarter_propagation
    ccw_segment_power_sum += _squared_magnitude(ccw_field)
    through_field = through_self_coupling * input_field - 1j * through_cross_coupling * cw_field  # through
    # convert segment-entry powers to stored energy and intrinsic lost power
    quarter_ring_energy_per_power, quarter_ring_loss_fraction = calculate_quarter_ring_energy_per_power_and_loss_fraction(
        round_trip_field_loss_exponent, round_trip_delay_s
    )
    return (
        reflected_field,
        through_field,
        drop_field,
        add_output_field,
        cw_segment_power_sum * quarter_ring_energy_per_power,
        ccw_segment_power_sum * quarter_ring_energy_per_power,
        (cw_segment_power_sum + ccw_segment_power_sum) * quarter_ring_loss_fraction,
    )


# `solve_exact_ring_fields_and_energy` plus analytic derivatives with respect to the round-trip phase.
# return port fields, directional energies, intrinsic loss, and the loss/through/drop phase derivatives.
# differentiate each field with respect to round-trip phase phi.
# dM=C1*(dH*C2*H+H*C2*dH).
# differentiating (I-M)*x=c gives dx=(I-M)^-1*(dc+dM*x).
# then the field pass carries (field, d field).
# coupling and loss are held fixed (they do not depend on phi).
# complex set of chain rules and derivatives. not worth rlly reading. worth just verifying end to end though
# x
@njit(cache=True)
def solve_exact_ring_fields_and_phase_derivatives(
    round_trip_phase_rad: float,  # round_trip_phase_rad = phi (rad) - absolute round-trip phase including coupler phase - type float scalar
    round_trip_field_loss_exponent: float,  # round_trip_field_loss_exponent = -ln(a_rt) (1) - round-trip field attenuation exponent - type float scalar
    through_self_coupling: float,  # through_self_coupling = t_th (1) - through-coupler field self-coupling - type float scalar
    through_cross_coupling: float,  # through_cross_coupling = K_th (1) - through-coupler field cross-coupling - type float scalar
    drop_self_coupling: float,  # drop_self_coupling = t_dr (1) - drop-coupler field self-coupling - type float scalar
    drop_cross_coupling: float,  # drop_cross_coupling = K_dr (1) - drop-coupler field cross-coupling - type float scalar
    round_trip_delay_s: float,  # round_trip_delay_s = T_rt (s) - round-trip group delay - type float scalar
    round_trip_backscatter: Complex128Vector,  # round_trip_backscatter = r_b (1) - complex round-trip backscatter coupling by resonance order - type complex128 ndarray, shape (backscatter_order_count,)
    input_field: complex,  # input_field = s_in,0 (sqrt(W)) - incoming field at port 0 - type complex scalar
    backward_through_field: complex,  # backward_through_field = s_in,1 (sqrt(W)) - incoming field at port 1 - type complex scalar
    backward_drop_field: complex,  # backward_drop_field = s_in,2 (sqrt(W)) - incoming field at port 2 - type complex scalar
    add_input_field: complex,  # add_input_field = s_in,3 (sqrt(W)) - incoming field at port 3 - type complex scalar
) -> tuple[
    complex,  # reflected_field = s_out,0 (sqrt(W)) - reflected field at the input port - type complex scalar
    complex,  # through_field = s_out,1 (sqrt(W)) - through-port field - type complex scalar
    complex,  # drop_field = s_out,2 (sqrt(W)) - clockwise drop-port field - type complex scalar
    complex,  # add_output_field = s_out,3 (sqrt(W)) - counter-clockwise add-port field - type complex scalar
    float,  # return[4] = U_cw (J) - clockwise stored energy - type float scalar
    float,  # return[5] = U_ccw (J) - counter-clockwise stored energy - type float scalar
    float,  # return[6] = P_loss (W) - total intrinsic-loss power - type float scalar
    float,  # return[7] = dP_loss/dphi (W/rad) - intrinsic-loss power derivative with respect to phase - type float scalar
    float,  # return[8] = dP_through/dphi (W/rad) - through-port power derivative with respect to phase - type float scalar
    float,  # return[9] = dP_drop/dphi (W/rad) - drop-port power derivative with respect to phase - type float scalar
]:
    scatterer_reflection, d_scatterer_reflection_dphi = interpolate_scatterer_reflection_and_phase_derivative(
        round_trip_phase_rad, round_trip_backscatter
    )
    scatterer_reflection_conj = scatterer_reflection.conjugate()
    d_scatterer_reflection_conj_dphi = d_scatterer_reflection_dphi.conjugate()
    inverse_scatterer_field_transmission = 1.0 / math.sqrt(1.0 - _squared_magnitude(scatterer_reflection))
    d_inverse_scatterer_transmission_dphi = (
        (scatterer_reflection_conj * d_scatterer_reflection_dphi).real
        * inverse_scatterer_field_transmission
        * inverse_scatterer_field_transmission
        * inverse_scatterer_field_transmission
    )  # d(1/tau) = Re(conj(rho) d rho)/tau^3
    quarter_propagation = cmath.exp(-0.25 * (round_trip_field_loss_exponent + 1j * round_trip_phase_rad))
    inverse_quarter_propagation = 1.0 / quarter_propagation
    d_quarter_propagation_dphi = -0.25j * quarter_propagation
    d_inverse_quarter_propagation_dphi = 0.25j * inverse_quarter_propagation
    half_propagation = quarter_propagation * quarter_propagation
    inverse_half_propagation = inverse_quarter_propagation * inverse_quarter_propagation
    inverse_through_self_coupling = 1.0 / through_self_coupling
    inverse_drop_self_coupling = 1.0 / drop_self_coupling
    half_transfer_11 = half_propagation * inverse_scatterer_field_transmission
    half_transfer_12 = -scatterer_reflection_conj * inverse_scatterer_field_transmission
    half_transfer_21 = -scatterer_reflection * inverse_scatterer_field_transmission
    half_transfer_22 = inverse_half_propagation * inverse_scatterer_field_transmission
    d_half_transfer_11_dphi = (
        -0.5j * half_propagation * inverse_scatterer_field_transmission
        + half_propagation * d_inverse_scatterer_transmission_dphi
    )
    d_half_transfer_12_dphi = -(
        d_scatterer_reflection_conj_dphi * inverse_scatterer_field_transmission
        + scatterer_reflection_conj * d_inverse_scatterer_transmission_dphi
    )
    d_half_transfer_21_dphi = -(
        d_scatterer_reflection_dphi * inverse_scatterer_field_transmission
        + scatterer_reflection * d_inverse_scatterer_transmission_dphi
    )
    d_half_transfer_22_dphi = (
        0.5j * inverse_half_propagation * inverse_scatterer_field_transmission
        + inverse_half_propagation * d_inverse_scatterer_transmission_dphi
    )
    internal_transfer_11 = (
        half_transfer_11 * drop_self_coupling * half_transfer_11
        + half_transfer_12 * inverse_drop_self_coupling * half_transfer_21
    )
    internal_transfer_12 = (
        half_transfer_11 * drop_self_coupling * half_transfer_12
        + half_transfer_12 * inverse_drop_self_coupling * half_transfer_22
    )
    internal_transfer_21 = (
        half_transfer_21 * drop_self_coupling * half_transfer_11
        + half_transfer_22 * inverse_drop_self_coupling * half_transfer_21
    )
    internal_transfer_22 = (
        half_transfer_21 * drop_self_coupling * half_transfer_12
        + half_transfer_22 * inverse_drop_self_coupling * half_transfer_22
    )
    d_internal_transfer_11_dphi = (
        d_half_transfer_11_dphi * drop_self_coupling * half_transfer_11
        + d_half_transfer_12_dphi * inverse_drop_self_coupling * half_transfer_21
        + half_transfer_11 * drop_self_coupling * d_half_transfer_11_dphi
        + half_transfer_12 * inverse_drop_self_coupling * d_half_transfer_21_dphi
    )
    d_internal_transfer_12_dphi = (
        d_half_transfer_11_dphi * drop_self_coupling * half_transfer_12
        + d_half_transfer_12_dphi * inverse_drop_self_coupling * half_transfer_22
        + half_transfer_11 * drop_self_coupling * d_half_transfer_12_dphi
        + half_transfer_12 * inverse_drop_self_coupling * d_half_transfer_22_dphi
    )
    d_internal_transfer_21_dphi = (
        d_half_transfer_21_dphi * drop_self_coupling * half_transfer_11
        + d_half_transfer_22_dphi * inverse_drop_self_coupling * half_transfer_21
        + half_transfer_21 * drop_self_coupling * d_half_transfer_11_dphi
        + half_transfer_22 * inverse_drop_self_coupling * d_half_transfer_21_dphi
    )
    d_internal_transfer_22_dphi = (
        d_half_transfer_21_dphi * drop_self_coupling * half_transfer_12
        + d_half_transfer_22_dphi * inverse_drop_self_coupling * half_transfer_22
        + half_transfer_21 * drop_self_coupling * d_half_transfer_12_dphi
        + half_transfer_22 * inverse_drop_self_coupling * d_half_transfer_22_dphi
    )
    cw_drop_coupler_drive = -1j * drop_cross_coupling * add_input_field
    ccw_drop_coupler_drive = 1j * drop_cross_coupling * backward_drop_field * inverse_drop_self_coupling
    cw_loop_drive = (
        through_self_coupling
        * (half_transfer_11 * cw_drop_coupler_drive + half_transfer_12 * ccw_drop_coupler_drive)
        - 1j * through_cross_coupling * input_field
    )
    ccw_loop_drive = (
        inverse_through_self_coupling
        * (half_transfer_21 * cw_drop_coupler_drive + half_transfer_22 * ccw_drop_coupler_drive)
        + 1j * through_cross_coupling * backward_through_field * inverse_through_self_coupling
    )
    d_cw_loop_drive_dphi = through_self_coupling * (
        d_half_transfer_11_dphi * cw_drop_coupler_drive + d_half_transfer_12_dphi * ccw_drop_coupler_drive
    )
    d_ccw_loop_drive_dphi = inverse_through_self_coupling * (
        d_half_transfer_21_dphi * cw_drop_coupler_drive + d_half_transfer_22_dphi * ccw_drop_coupler_drive
    )
    loop_matrix_11 = 1.0 - through_self_coupling * internal_transfer_11
    loop_matrix_12 = -through_self_coupling * internal_transfer_12
    loop_matrix_21 = -inverse_through_self_coupling * internal_transfer_21
    loop_matrix_22 = 1.0 - inverse_through_self_coupling * internal_transfer_22
    loop_determinant = loop_matrix_11 * loop_matrix_22 - loop_matrix_12 * loop_matrix_21
    cw_field = (loop_matrix_22 * cw_loop_drive - loop_matrix_12 * ccw_loop_drive) / loop_determinant
    ccw_field = (loop_matrix_11 * ccw_loop_drive - loop_matrix_21 * cw_loop_drive) / loop_determinant
    # solve (I-M)*dx/dphi = dc/dphi + (dM/dphi)*x for the field phase derivatives.
    # cw_derivative_rhs = b_cw (sqrt(W)/rad) - clockwise entry on the right-hand side.
    cw_derivative_rhs = d_cw_loop_drive_dphi + through_self_coupling * (
        d_internal_transfer_11_dphi * cw_field + d_internal_transfer_12_dphi * ccw_field
    )
    # ccw_derivative_rhs = b_ccw (sqrt(W)/rad) - counter-clockwise entry on the right-hand side.
    ccw_derivative_rhs = d_ccw_loop_drive_dphi + inverse_through_self_coupling * (
        d_internal_transfer_21_dphi * cw_field + d_internal_transfer_22_dphi * ccw_field
    )
    d_cw_field_dphi = (
        loop_matrix_22 * cw_derivative_rhs - loop_matrix_12 * ccw_derivative_rhs
    ) / loop_determinant
    d_ccw_field_dphi = (
        loop_matrix_11 * ccw_derivative_rhs - loop_matrix_21 * cw_derivative_rhs
    ) / loop_determinant
    reflected_field = through_self_coupling * backward_through_field - 1j * through_cross_coupling * ccw_field
    # segment 1
    cw_segment_power_sum = _squared_magnitude(cw_field)
    d_cw_segment_power_sum_dphi = _squared_magnitude_derivative(cw_field, d_cw_field_dphi)
    d_cw_field_dphi = d_quarter_propagation_dphi * cw_field + quarter_propagation * d_cw_field_dphi
    cw_field = quarter_propagation * cw_field
    d_ccw_field_dphi = (
        d_ccw_field_dphi * inverse_quarter_propagation + ccw_field * d_inverse_quarter_propagation_dphi
    )
    ccw_field = ccw_field * inverse_quarter_propagation
    ccw_segment_power_sum = _squared_magnitude(ccw_field)
    d_ccw_segment_power_sum_dphi = _squared_magnitude_derivative(ccw_field, d_ccw_field_dphi)
    cw_scatter_numerator = cw_field - scatterer_reflection_conj * ccw_field
    ccw_scatter_numerator = ccw_field - scatterer_reflection * cw_field
    d_cw_field_dphi, d_ccw_field_dphi = (
        (
            d_cw_field_dphi
            - d_scatterer_reflection_conj_dphi * ccw_field
            - scatterer_reflection_conj * d_ccw_field_dphi
        )
        * inverse_scatterer_field_transmission
        + cw_scatter_numerator * d_inverse_scatterer_transmission_dphi,
        (d_ccw_field_dphi - d_scatterer_reflection_dphi * cw_field - scatterer_reflection * d_cw_field_dphi)
        * inverse_scatterer_field_transmission
        + ccw_scatter_numerator * d_inverse_scatterer_transmission_dphi,
    )
    cw_field, ccw_field = (
        cw_scatter_numerator * inverse_scatterer_field_transmission,
        ccw_scatter_numerator * inverse_scatterer_field_transmission,
    )
    # segment 2
    cw_segment_power_sum += _squared_magnitude(cw_field)
    d_cw_segment_power_sum_dphi += _squared_magnitude_derivative(cw_field, d_cw_field_dphi)
    d_cw_field_dphi = d_quarter_propagation_dphi * cw_field + quarter_propagation * d_cw_field_dphi
    cw_field = quarter_propagation * cw_field
    d_ccw_field_dphi = (
        d_ccw_field_dphi * inverse_quarter_propagation + ccw_field * d_inverse_quarter_propagation_dphi
    )
    ccw_field = ccw_field * inverse_quarter_propagation
    ccw_segment_power_sum += _squared_magnitude(ccw_field)
    d_ccw_segment_power_sum_dphi += _squared_magnitude_derivative(ccw_field, d_ccw_field_dphi)
    # drop coupler
    drop_field = drop_self_coupling * add_input_field - 1j * drop_cross_coupling * cw_field
    d_drop_field_dphi = -1j * drop_cross_coupling * d_cw_field_dphi
    d_cw_field_dphi = drop_self_coupling * d_cw_field_dphi
    cw_field = drop_self_coupling * cw_field - 1j * drop_cross_coupling * add_input_field
    d_ccw_field_dphi = d_ccw_field_dphi * inverse_drop_self_coupling
    ccw_field = (ccw_field + 1j * drop_cross_coupling * backward_drop_field) * inverse_drop_self_coupling
    add_output_field = drop_self_coupling * backward_drop_field - 1j * drop_cross_coupling * ccw_field
    # segment 3
    cw_segment_power_sum += _squared_magnitude(cw_field)
    d_cw_segment_power_sum_dphi += _squared_magnitude_derivative(cw_field, d_cw_field_dphi)
    d_cw_field_dphi = d_quarter_propagation_dphi * cw_field + quarter_propagation * d_cw_field_dphi
    cw_field = quarter_propagation * cw_field
    d_ccw_field_dphi = (
        d_ccw_field_dphi * inverse_quarter_propagation + ccw_field * d_inverse_quarter_propagation_dphi
    )
    ccw_field = ccw_field * inverse_quarter_propagation
    ccw_segment_power_sum += _squared_magnitude(ccw_field)
    d_ccw_segment_power_sum_dphi += _squared_magnitude_derivative(ccw_field, d_ccw_field_dphi)
    cw_scatter_numerator = cw_field - scatterer_reflection_conj * ccw_field
    ccw_scatter_numerator = ccw_field - scatterer_reflection * cw_field
    d_cw_field_dphi, d_ccw_field_dphi = (
        (
            d_cw_field_dphi
            - d_scatterer_reflection_conj_dphi * ccw_field
            - scatterer_reflection_conj * d_ccw_field_dphi
        )
        * inverse_scatterer_field_transmission
        + cw_scatter_numerator * d_inverse_scatterer_transmission_dphi,
        (d_ccw_field_dphi - d_scatterer_reflection_dphi * cw_field - scatterer_reflection * d_cw_field_dphi)
        * inverse_scatterer_field_transmission
        + ccw_scatter_numerator * d_inverse_scatterer_transmission_dphi,
    )
    cw_field, ccw_field = (
        cw_scatter_numerator * inverse_scatterer_field_transmission,
        ccw_scatter_numerator * inverse_scatterer_field_transmission,
    )
    # segment 4
    cw_segment_power_sum += _squared_magnitude(cw_field)
    d_cw_segment_power_sum_dphi += _squared_magnitude_derivative(cw_field, d_cw_field_dphi)
    d_cw_field_dphi = d_quarter_propagation_dphi * cw_field + quarter_propagation * d_cw_field_dphi
    cw_field = quarter_propagation * cw_field
    d_ccw_field_dphi = (
        d_ccw_field_dphi * inverse_quarter_propagation + ccw_field * d_inverse_quarter_propagation_dphi
    )
    ccw_field = ccw_field * inverse_quarter_propagation
    ccw_segment_power_sum += _squared_magnitude(ccw_field)
    d_ccw_segment_power_sum_dphi += _squared_magnitude_derivative(ccw_field, d_ccw_field_dphi)
    through_field = through_self_coupling * input_field - 1j * through_cross_coupling * cw_field
    d_through_field_dphi = -1j * through_cross_coupling * d_cw_field_dphi
    quarter_ring_energy_per_power, quarter_ring_loss_fraction = calculate_quarter_ring_energy_per_power_and_loss_fraction(
        round_trip_field_loss_exponent, round_trip_delay_s
    )
    return (
        reflected_field,
        through_field,
        drop_field,
        add_output_field,
        cw_segment_power_sum * quarter_ring_energy_per_power,
        ccw_segment_power_sum * quarter_ring_energy_per_power,
        (cw_segment_power_sum + ccw_segment_power_sum) * quarter_ring_loss_fraction,
        (d_cw_segment_power_sum_dphi + d_ccw_segment_power_sum_dphi) * quarter_ring_loss_fraction,
        _squared_magnitude_derivative(through_field, d_through_field_dphi),
        _squared_magnitude_derivative(drop_field, d_drop_field_dphi),
    )


# return 1/value for a finite nonzero complex value, using its squared magnitude.
# exact-model determinants use 1, while TCMT determinants use 1/s^2.
# x
@njit(cache=True)
def _complex_reciprocal(
    value: complex,  # value = z (input units) - finite nonzero complex value - type complex scalar
) -> (
    complex  # return = 1/z (inverse input units) - complex reciprocal - type complex scalar
):
    # 1/z = conj(z)/|z|^2 - reverse the angle and invert the magnitude.
    if value == 0j:
        raise ZeroDivisionError("singular ring field calculation")
    inverse_squared_magnitude = 1.0 / (value.real * value.real + value.imag * value.imag)
    return complex(value.real * inverse_squared_magnitude, -value.imag * inverse_squared_magnitude)


# prepare quarter-ring propagation and scatterer terms for one laser line.
# return (p, 1/p, rho, d rho/d phi).
# p = exp(-(xi + i*phi)/4) is the field propagation factor.
# rho is each scatterer's complex amplitude reflection coefficient.
# xi = -ln(a_rt) is dimensionless.
# phi includes the coupler phase, in rad.
# reuse the sine and cosine within one resonance order for propagation and interpolation.
# the absolute order supplies the remaining quarter turns.
# x
@njit(cache=True)
def calculate_propagation_and_scatterer_coefficients(
    round_trip_phase_rad: float,  # round_trip_phase_rad = phi (rad) - absolute round-trip phase including coupler phase - type float scalar
    round_trip_backscatter: Complex128Vector,  # round_trip_backscatter = r_b (1) - complex round-trip backscatter coupling by resonance order - type complex128 ndarray, shape (backscatter_order_count,)
    round_trip_field_loss_exponent: float,  # round_trip_field_loss_exponent = -ln(a_rt) (1) - round-trip field attenuation exponent - type float scalar
) -> tuple[
    complex,  # return[0] = p (1) - quarter-trip field propagation factor - type complex scalar
    complex,  # return[1] = 1/p (1) - inverse quarter-trip propagation factor - type complex scalar
    complex,  # return[2] = rho (1) - complex scatterer reflection coefficient - type complex scalar
    complex,  # return[3] = drho/dphi (1/rad) - scatterer reflection derivative with respect to phase - type complex scalar
]:
    backscatter_order_count = round_trip_backscatter.shape[0]
    lower_order = int(np.floor(round_trip_phase_rad / TWO_PI)) # n
    phase_within_order_rad = round_trip_phase_rad - TWO_PI * lower_order # theta
    quarter_phase_cos = math.cos(0.25 * phase_within_order_rad)
    quarter_phase_sin = math.sin(0.25 * phase_within_order_rad)
    order_sign = 1.0 - 2.0 * (lower_order & 1)
    lower_order_reflection = (
        -0.5j
        * order_sign
        * round_trip_backscatter[_wrap_order_index(lower_order, backscatter_order_count)].conjugate()
    )
    upper_order_reflection = (
        0.5j
        * order_sign
        * round_trip_backscatter[_wrap_order_index(lower_order + 1, backscatter_order_count)].conjugate()
    )
    scatterer_reflection = (
        quarter_phase_cos * quarter_phase_cos * lower_order_reflection
        + quarter_phase_sin * quarter_phase_sin * upper_order_reflection
    )
    d_scatterer_reflection_dphi = (
        0.5 * quarter_phase_sin * quarter_phase_cos * (upper_order_reflection - lower_order_reflection)
    )
    # phi/4 = theta/4 + n*pi/2 - restore the resonance order's quarter turns.
    # mainly for efficiency - implementing this w/this logic
    quarter_turn_index = lower_order & 3
    if quarter_turn_index == 0:
        quarter_total_phase_cos = quarter_phase_cos
        quarter_total_phase_sin = quarter_phase_sin
    elif quarter_turn_index == 1:
        quarter_total_phase_cos = -quarter_phase_sin
        quarter_total_phase_sin = quarter_phase_cos
    elif quarter_turn_index == 2:
        quarter_total_phase_cos = -quarter_phase_cos
        quarter_total_phase_sin = -quarter_phase_sin
    else:
        quarter_total_phase_cos = quarter_phase_sin
        quarter_total_phase_sin = -quarter_phase_cos

    quarter_field_transmission = math.exp(-0.25 * round_trip_field_loss_exponent)

    # return a weighted sum
    # p applies negative phase and field attenuation.
    # 1/p applies positive phase and inverse attenuation.
    return (
        complex(
            quarter_field_transmission * quarter_total_phase_cos,
            -quarter_field_transmission * quarter_total_phase_sin,
        ),
        complex(
            quarter_total_phase_cos / quarter_field_transmission,
            quarter_total_phase_sin / quarter_field_transmission,
        ),
        scatterer_reflection,
        d_scatterer_reflection_dphi,
    )


# exact ring - per-ring kernels over all rings
# the exact ring is latency-bound per line (a serial chain of ~20 dependent complex operations).
# so the ring kernels below run each stage as its own loop over the L lines ("vectorized over lines"), and independent lines overlap in the CPU pipeline.
# measured on the 16 GB M2 - about 80 ns per (ring, line) against about 150 ns for the scalar form.


# allocate reusable complex, real, and per-line scratch arrays for the ring kernels.
# return shapes (COMPLEX_WORKSPACE_ROW_COUNT, L), (REAL_WORKSPACE_ROW_COUNT, L), and (L,).
# each parallel simulated world needs its own arrays.
# x
def make_workspace(
    n_lines: int,  # n_lines = L (1) - number of laser lines - type int scalar
) -> tuple[
    ComplexWorkspace,  # return[0] = C_work (mixed units) - complex scratch with COMPLEX_WORKSPACE_* field units - type complex128 ndarray, capacity at least (8, L)
    RealWorkspace,  # return[1] = F_work (mixed units) - real scratch with REAL_WORKSPACE_* and RING_RESPONSE_* field units - type float64 ndarray, capacity at least (27, L)
    Float64Vector,  # return[2] = P_forward (W) - scratch for propagated per-line power - type float64 ndarray, shape (L,)
]:
    return (
        np.zeros((COMPLEX_WORKSPACE_ROW_COUNT, n_lines), np.complex128),
        np.zeros((REAL_WORKSPACE_ROW_COUNT, n_lines), np.float64),
        np.zeros(n_lines, np.float64),
    )


# allocate outputs indexed by (field, ring, line), (field, ring), and (field, line).
def make_bus_outputs(
    n_rings: int,  # n_rings = N (1) - number of rings - type int scalar
    n_lines: int,  # n_lines = L (1) - number of laser lines - type int scalar
) -> tuple[
    RingLineOutputs,  # return[0] = Y_ring,line (mixed units) - RING_LINE_OUTPUT_* powers (W), energies (J), and slope (W/rad) - type float64 ndarray, shape (16, N, L)
    RingOutputs,  # return[1] = Y_ring (mixed units) - RING_OUTPUT_* powers (W) and slope (W/m) - type float64 ndarray, shape (6, N)
    LaserLineOutputs,  # return[2] = Y_line (mixed units) - LASER_LINE_OUTPUT_* powers (W) and relative error bound (1) - type float64 ndarray, shape (3, L)
]:
    return (
        np.zeros((RING_LINE_OUTPUT_ROW_COUNT, n_rings, n_lines)),
        np.zeros((RING_OUTPUT_ROW_COUNT, n_rings)),
        np.zeros((LASER_LINE_OUTPUT_ROW_COUNT, n_lines)),
    )


# check whether any resonance order couples light into the opposite travelling direction.
# input - ring_backscatter = r_b (1) - complex round-trip backscatter coupling by resonance order
# output - return = backscatter_present (1) - true if any coupling coefficient is nonzero
# no callers since we always using backscatter now
# x
# @njit(cache=True)
# def _ring_has_backscatter(ring_backscatter):
#     for order_index in range(ring_backscatter.shape[0]):
#         if ring_backscatter[order_index].real != 0.0 or ring_backscatter[order_index].imag != 0.0:
#             return True
#     return False


# reject incompatible workspace storage before a ring solve reads or writes it.
# larger buffers and strided views remain valid when they contain every requested row and line.
@njit(cache=True, inline="always", _nrt=False)
def _validate_ring_workspace_buffers(
    line_count: int,  # line_count = L (1) - number of laser lines to evaluate - type int scalar
    complex_workspace: ComplexWorkspace,  # complex_workspace = C_work (mixed units) - two-dimensional complex128 workspace - type complex128 ndarray, capacity at least (8, L)
    real_workspace: RealWorkspace,  # real_workspace = F_work (mixed units) - two-dimensional float64 workspace - type float64 ndarray, capacity at least (27, L)
    round_trip_delay_row_s: Float64Vector,  # round_trip_delay_row_s = T_round_trip (s) - one-dimensional float64 delay row - type float64 ndarray, length at least L
    power_energy_response: RingResponseBlock,  # power_energy_response = R_opt (mixed units) - two-dimensional float64 response buffer - type float64 ndarray, capacity at least (8, L)
) -> (
    None  # return = None - incorrect dtypes, dimensions, or capacities raise an exception - type None
):
    if line_count < 0:
        raise ValueError("line_count must be nonnegative")
    if complex_workspace.ndim != 2 or real_workspace.ndim != 2 or power_energy_response.ndim != 2:
        raise ValueError("ring workspaces and response must be two-dimensional arrays")
    if round_trip_delay_row_s.ndim != 1:
        raise ValueError("round-trip delays must be a one-dimensional array")
    if not (complex_workspace.dtype == np.dtype(np.complex128)):
        raise TypeError("complex workspace must use complex128 storage")
    if not (real_workspace.dtype == np.dtype(np.float64)):
        raise TypeError("real workspace must use float64 storage")
    if not (round_trip_delay_row_s.dtype == np.dtype(np.float64)):
        raise TypeError("round-trip delays must use float64 storage")
    if not (power_energy_response.dtype == np.dtype(np.float64)):
        raise TypeError("ring response must use float64 storage")
    if complex_workspace.shape[0] < COMPLEX_WORKSPACE_ROW_COUNT or complex_workspace.shape[1] < line_count:
        raise ValueError("complex workspace has insufficient rows or laser-line capacity")
    if real_workspace.shape[0] < REAL_WORKSPACE_ROW_COUNT or real_workspace.shape[1] < line_count:
        raise ValueError("real workspace has insufficient rows or laser-line capacity")
    if power_energy_response.shape[0] < RING_RESPONSE_ROW_COUNT or power_energy_response.shape[1] < line_count:
        raise ValueError("ring response has insufficient rows or laser-line capacity")
    if round_trip_delay_row_s.shape[0] < line_count:
        raise ValueError("round-trip delay row has insufficient laser-line capacity")


# exact ring, unit forward input, values only - solve, then the field pass in two half-ring stages.
# use the closed-form half-ring matrix K=H*C2*H.
# P2=p^2, S=t2*P2+1/(t2*P2), and tau^2=1-|rho|^2.
# k11=(t2*P2^2+|rho|^2/t2)/tau^2.
# k22=(t2*|rho|^2+1/(t2*P2^2))/tau^2.
# k12=-conj(rho)*S/tau^2 and k21=-rho*S/tau^2.
# det(I-C1*K)=a11*a22-|rho|^2*S^2/tau^4.
# this is just doing prev. functions with multiple lasers and unpacking matrices
# x
@njit(cache=True, error_model="numpy", _nrt=False)
def solve_ring_fields_at_all_laser_wavelengths(
    line_count: int,  # line_count = L (1) - number of laser lines - type int scalar
    complex_workspace: ComplexWorkspace,  # complex_workspace = C_work (mixed units) - COMPLEX_WORKSPACE_* factors, fields, and field phase slopes - type complex128 ndarray, capacity at least (8, L)
    real_workspace: RealWorkspace,  # real_workspace = F_work (mixed units) - REAL_WORKSPACE_* factors, powers, slopes, and RING_RESPONSE_* response blocks - type float64 ndarray, capacity at least (27, L)
    round_trip_delay_row_s: Float64Vector,  # round_trip_delay_row_s = T_rt (s) - round-trip group delay - type float64 ndarray, length at least L
    power_energy_response: RingResponseBlock,  # power_energy_response = R_opt (mixed units) - RING_RESPONSE_* power fractions (1), energies (J/W), and loss slope (1/rad) - type float64 ndarray, capacity at least (8, L)
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    _validate_ring_workspace_buffers(
        line_count,
        complex_workspace,
        real_workspace,
        round_trip_delay_row_s,
        power_energy_response,
    )
    quarter_propagation_row = complex_workspace[COMPLEX_WORKSPACE_ROW_QUARTER_RING_FIELD_PROPAGATION_FACTOR]
    inverse_quarter_propagation_row = complex_workspace[
        COMPLEX_WORKSPACE_ROW_INVERSE_QUARTER_RING_FIELD_PROPAGATION_FACTOR
    ]
    scatterer_reflection_row = complex_workspace[COMPLEX_WORKSPACE_ROW_SCATTERER_FIELD_REFLECTION_COEFFICIENT]
    cw_field_row = complex_workspace[COMPLEX_WORKSPACE_ROW_CLOCKWISE_FIELD_AMPLITUDE]
    ccw_field_row = complex_workspace[COMPLEX_WORKSPACE_ROW_COUNTERCLOCKWISE_FIELD_AMPLITUDE]
    for line_index in range(line_count):
        quarter_propagation = quarter_propagation_row[line_index]
        inverse_quarter_propagation = inverse_quarter_propagation_row[line_index]
        scatterer_reflection = scatterer_reflection_row[line_index]
        through_self_coupling = real_workspace[
            REAL_WORKSPACE_ROW_THROUGH_COUPLER_FIELD_SELF_COUPLING, line_index
        ]
        drop_self_coupling = real_workspace[REAL_WORKSPACE_ROW_DROP_COUPLER_FIELD_SELF_COUPLING, line_index]
        scatterer_reflectance = _squared_magnitude(scatterer_reflection)
        inverse_scatterer_power_transmission = 1.0 / (1.0 - scatterer_reflectance)
        inverse_through_self_coupling = 1.0 / through_self_coupling
        inverse_drop_self_coupling = 1.0 / drop_self_coupling
        real_workspace[REAL_WORKSPACE_ROW_INVERSE_SCATTERER_FIELD_TRANSMISSION, line_index] = math.sqrt(
            inverse_scatterer_power_transmission
        )
        half_propagation = quarter_propagation * quarter_propagation
        inverse_half_propagation = inverse_quarter_propagation * inverse_quarter_propagation
        half_path_sum = (
            drop_self_coupling * half_propagation + inverse_drop_self_coupling * inverse_half_propagation
        )
        loop_matrix_11 = 1.0 - through_self_coupling * inverse_scatterer_power_transmission * (
            drop_self_coupling * half_propagation * half_propagation
            + inverse_drop_self_coupling * scatterer_reflectance
        )
        loop_matrix_22 = 1.0 - inverse_through_self_coupling * inverse_scatterer_power_transmission * (
            drop_self_coupling * scatterer_reflectance
            + inverse_drop_self_coupling * inverse_half_propagation * inverse_half_propagation
        )
        cw_drive_over_determinant = complex(
            0.0, -real_workspace[REAL_WORKSPACE_ROW_THROUGH_COUPLER_FIELD_CROSS_COUPLING, line_index]
        ) * _complex_reciprocal(
            loop_matrix_11 * loop_matrix_22
            - scatterer_reflectance
            * inverse_scatterer_power_transmission
            * inverse_scatterer_power_transmission
            * half_path_sum
            * half_path_sum
        )
        cw_field_row[line_index] = loop_matrix_22 * cw_drive_over_determinant  # CW at 0+
        ccw_field_row[line_index] = (
            -inverse_through_self_coupling
            * inverse_scatterer_power_transmission
            * half_path_sum
            * scatterer_reflection
            * cw_drive_over_determinant
        )  # CCW at 0+
    for line_index in range(line_count):  # 0 -> L/2, drop coupler
        quarter_propagation = quarter_propagation_row[line_index]
        inverse_quarter_propagation = inverse_quarter_propagation_row[line_index]
        scatterer_reflection = scatterer_reflection_row[line_index]
        scatterer_reflection_conj = scatterer_reflection.conjugate()
        inverse_scatterer_field_transmission = real_workspace[
            REAL_WORKSPACE_ROW_INVERSE_SCATTERER_FIELD_TRANSMISSION, line_index
        ]
        cw_field = cw_field_row[line_index]
        ccw_field = ccw_field_row[line_index]
        through_cross_coupling = real_workspace[
            REAL_WORKSPACE_ROW_THROUGH_COUPLER_FIELD_CROSS_COUPLING, line_index
        ]
        drop_cross_coupling = real_workspace[REAL_WORKSPACE_ROW_DROP_COUPLER_FIELD_CROSS_COUPLING, line_index]
        drop_self_coupling = real_workspace[REAL_WORKSPACE_ROW_DROP_COUPLER_FIELD_SELF_COUPLING, line_index]
        power_energy_response[RING_RESPONSE_ROW_REFLECTED_POWER_FRACTION, line_index] = (
            through_cross_coupling * through_cross_coupling * _squared_magnitude(ccw_field)
        )
        cw_segment_power_sum = _squared_magnitude(cw_field)
        cw_field = quarter_propagation * cw_field
        ccw_field = ccw_field * inverse_quarter_propagation
        ccw_segment_power_sum = _squared_magnitude(ccw_field)
        cw_field, ccw_field = (
            (cw_field - scatterer_reflection_conj * ccw_field) * inverse_scatterer_field_transmission,
            (ccw_field - scatterer_reflection * cw_field) * inverse_scatterer_field_transmission,
        )
        cw_segment_power_sum += _squared_magnitude(cw_field)
        cw_field = quarter_propagation * cw_field
        ccw_field = ccw_field * inverse_quarter_propagation
        ccw_segment_power_sum += _squared_magnitude(ccw_field)
        power_energy_response[RING_RESPONSE_ROW_CLOCKWISE_DROP_POWER_FRACTION, line_index] = (
            drop_cross_coupling * drop_cross_coupling * _squared_magnitude(cw_field)
        )
        cw_field = drop_self_coupling * cw_field
        ccw_field = ccw_field / drop_self_coupling
        power_energy_response[RING_RESPONSE_ROW_COUNTERCLOCKWISE_ADD_PORT_POWER_FRACTION, line_index] = (
            drop_cross_coupling * drop_cross_coupling * _squared_magnitude(ccw_field)
        )
        cw_field_row[line_index] = cw_field
        ccw_field_row[line_index] = ccw_field
        real_workspace[REAL_WORKSPACE_ROW_CLOCKWISE_SEGMENT_INPUT_POWER_SUM, line_index] = (
            cw_segment_power_sum
        )
        real_workspace[REAL_WORKSPACE_ROW_COUNTERCLOCKWISE_SEGMENT_INPUT_POWER_SUM, line_index] = (
            ccw_segment_power_sum
        )
    for line_index in range(line_count):  # L/2 -> L, input coupler
        quarter_propagation = quarter_propagation_row[line_index]
        inverse_quarter_propagation = inverse_quarter_propagation_row[line_index]
        scatterer_reflection = scatterer_reflection_row[line_index]
        scatterer_reflection_conj = scatterer_reflection.conjugate()
        inverse_scatterer_field_transmission = real_workspace[
            REAL_WORKSPACE_ROW_INVERSE_SCATTERER_FIELD_TRANSMISSION, line_index
        ]
        cw_field = cw_field_row[line_index]
        ccw_field = ccw_field_row[line_index]
        cw_segment_power_sum = real_workspace[
            REAL_WORKSPACE_ROW_CLOCKWISE_SEGMENT_INPUT_POWER_SUM, line_index
        ] + _squared_magnitude(cw_field)
        cw_field = quarter_propagation * cw_field
        ccw_field = ccw_field * inverse_quarter_propagation
        ccw_segment_power_sum = real_workspace[
            REAL_WORKSPACE_ROW_COUNTERCLOCKWISE_SEGMENT_INPUT_POWER_SUM, line_index
        ] + _squared_magnitude(ccw_field)
        cw_field, ccw_field = (
            (cw_field - scatterer_reflection_conj * ccw_field) * inverse_scatterer_field_transmission,
            (ccw_field - scatterer_reflection * cw_field) * inverse_scatterer_field_transmission,
        )
        cw_segment_power_sum += _squared_magnitude(cw_field)
        cw_field = quarter_propagation * cw_field
        ccw_field = ccw_field * inverse_quarter_propagation
        ccw_segment_power_sum += _squared_magnitude(ccw_field)
        power_energy_response[RING_RESPONSE_ROW_THROUGH_POWER_FRACTION, line_index] = _squared_magnitude(
            real_workspace[REAL_WORKSPACE_ROW_THROUGH_COUPLER_FIELD_SELF_COUPLING, line_index]
            - 1j
            * real_workspace[REAL_WORKSPACE_ROW_THROUGH_COUPLER_FIELD_CROSS_COUPLING, line_index]
            * cw_field
        )
        quarter_ring_energy_per_power, quarter_ring_loss_fraction = calculate_quarter_ring_energy_per_power_and_loss_fraction(
            real_workspace[REAL_WORKSPACE_ROW_ROUND_TRIP_FIELD_LOSS_EXPONENT, line_index],
            round_trip_delay_row_s[line_index],
        )
        power_energy_response[RING_RESPONSE_ROW_CLOCKWISE_STORED_ENERGY_PER_INPUT_POWER, line_index] = (
            cw_segment_power_sum * quarter_ring_energy_per_power
        )
        power_energy_response[
            RING_RESPONSE_ROW_COUNTERCLOCKWISE_STORED_ENERGY_PER_INPUT_POWER, line_index
        ] = (ccw_segment_power_sum * quarter_ring_energy_per_power)
        power_energy_response[RING_RESPONSE_ROW_INTRINSIC_LOSS_POWER_FRACTION, line_index] = (
            cw_segment_power_sum + ccw_segment_power_sum
        ) * quarter_ring_loss_fraction
        power_energy_response[
            RING_RESPONSE_ROW_INTRINSIC_LOSS_POWER_FRACTION_ROUND_TRIP_PHASE_DERIVATIVE, line_index
        ] = 0.0


# solve the fields and differentiate intrinsic loss with respect to round-trip phase.
# apply the quotient rule to the field solution, then differentiate each propagation step.
# hold couplings and round-trip loss fixed during this derivative.
# also doing similar things to the stuff above.
# x
@njit(cache=True, error_model="numpy", _nrt=False)
def solve_ring_fields_and_loss_phase_derivative_at_all_laser_wavelengths(
    line_count: int,  # line_count = L (1) - number of laser lines - type int scalar
    complex_workspace: ComplexWorkspace,  # complex_workspace = C_work (mixed units) - COMPLEX_WORKSPACE_* factors, fields, and field phase slopes - type complex128 ndarray, capacity at least (8, L)
    real_workspace: RealWorkspace,  # real_workspace = F_work (mixed units) - REAL_WORKSPACE_* factors, powers, slopes, and RING_RESPONSE_* response blocks - type float64 ndarray, capacity at least (27, L)
    round_trip_delay_row_s: Float64Vector,  # round_trip_delay_row_s = T_rt (s) - round-trip group delay - type float64 ndarray, length at least L
    power_energy_response: RingResponseBlock,  # power_energy_response = R_opt (mixed units) - RING_RESPONSE_* power fractions (1), energies (J/W), and loss slope (1/rad) - type float64 ndarray, capacity at least (8, L)
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    _validate_ring_workspace_buffers(
        line_count,
        complex_workspace,
        real_workspace,
        round_trip_delay_row_s,
        power_energy_response,
    )
    quarter_propagation_row = complex_workspace[COMPLEX_WORKSPACE_ROW_QUARTER_RING_FIELD_PROPAGATION_FACTOR]
    inverse_quarter_propagation_row = complex_workspace[
        COMPLEX_WORKSPACE_ROW_INVERSE_QUARTER_RING_FIELD_PROPAGATION_FACTOR
    ]
    scatterer_reflection_row = complex_workspace[COMPLEX_WORKSPACE_ROW_SCATTERER_FIELD_REFLECTION_COEFFICIENT]
    d_scatterer_reflection_dphi_row = complex_workspace[
        COMPLEX_WORKSPACE_ROW_SCATTERER_FIELD_REFLECTION_ROUND_TRIP_PHASE_DERIVATIVE
    ]
    cw_field_row = complex_workspace[COMPLEX_WORKSPACE_ROW_CLOCKWISE_FIELD_AMPLITUDE]
    ccw_field_row = complex_workspace[COMPLEX_WORKSPACE_ROW_COUNTERCLOCKWISE_FIELD_AMPLITUDE]
    d_cw_field_dphi_row = complex_workspace[
        COMPLEX_WORKSPACE_ROW_CLOCKWISE_FIELD_AMPLITUDE_ROUND_TRIP_PHASE_DERIVATIVE
    ]
    d_ccw_field_dphi_row = complex_workspace[
        COMPLEX_WORKSPACE_ROW_COUNTERCLOCKWISE_FIELD_AMPLITUDE_ROUND_TRIP_PHASE_DERIVATIVE
    ]
    for line_index in range(line_count):
        quarter_propagation = quarter_propagation_row[line_index]
        inverse_quarter_propagation = inverse_quarter_propagation_row[line_index]
        scatterer_reflection = scatterer_reflection_row[line_index]
        d_scatterer_reflection_dphi = d_scatterer_reflection_dphi_row[line_index]
        scatterer_reflection_conj = scatterer_reflection.conjugate()
        through_self_coupling = real_workspace[
            REAL_WORKSPACE_ROW_THROUGH_COUPLER_FIELD_SELF_COUPLING, line_index
        ]
        drop_self_coupling = real_workspace[REAL_WORKSPACE_ROW_DROP_COUPLER_FIELD_SELF_COUPLING, line_index]
        scatterer_reflectance = _squared_magnitude(scatterer_reflection)
        d_scatterer_reflectance_dphi = 2.0 * (scatterer_reflection_conj * d_scatterer_reflection_dphi).real
        inverse_scatterer_power_transmission = 1.0 / (1.0 - scatterer_reflectance)
        d_inverse_scatterer_power_transmission_dphi = (
            inverse_scatterer_power_transmission
            * inverse_scatterer_power_transmission
            * d_scatterer_reflectance_dphi
        )
        inverse_scatterer_field_transmission = math.sqrt(inverse_scatterer_power_transmission)
        real_workspace[REAL_WORKSPACE_ROW_INVERSE_SCATTERER_FIELD_TRANSMISSION, line_index] = (
            inverse_scatterer_field_transmission
        )
        real_workspace[
            REAL_WORKSPACE_ROW_INVERSE_SCATTERER_FIELD_TRANSMISSION_ROUND_TRIP_PHASE_DERIVATIVE, line_index
        ] = (0.5 * d_inverse_scatterer_power_transmission_dphi / inverse_scatterer_field_transmission)
        inverse_through_self_coupling = 1.0 / through_self_coupling
        inverse_drop_self_coupling = 1.0 / drop_self_coupling
        half_propagation = quarter_propagation * quarter_propagation
        inverse_half_propagation = inverse_quarter_propagation * inverse_quarter_propagation
        half_path_sum = (
            drop_self_coupling * half_propagation + inverse_drop_self_coupling * inverse_half_propagation
        )
        d_half_path_sum_dphi = (
            -0.5j * drop_self_coupling * half_propagation
            + 0.5j * inverse_drop_self_coupling * inverse_half_propagation
        )
        cw_internal_transfer_numerator = (
            drop_self_coupling * half_propagation * half_propagation
            + inverse_drop_self_coupling * scatterer_reflectance
        )
        d_cw_transfer_numerator_dphi = (
            -1j * drop_self_coupling * half_propagation * half_propagation
            + inverse_drop_self_coupling * d_scatterer_reflectance_dphi
        )
        ccw_internal_transfer_numerator = (
            drop_self_coupling * scatterer_reflectance
            + inverse_drop_self_coupling * inverse_half_propagation * inverse_half_propagation
        )
        d_ccw_transfer_numerator_dphi = (
            drop_self_coupling * d_scatterer_reflectance_dphi
            + 1j * inverse_drop_self_coupling * inverse_half_propagation * inverse_half_propagation
        )
        loop_matrix_11 = (
            1.0
            - through_self_coupling * inverse_scatterer_power_transmission * cw_internal_transfer_numerator
        )
        d_loop_matrix_11_dphi = -through_self_coupling * (
            d_inverse_scatterer_power_transmission_dphi * cw_internal_transfer_numerator
            + inverse_scatterer_power_transmission * d_cw_transfer_numerator_dphi
        )
        loop_matrix_22 = (
            1.0
            - inverse_through_self_coupling
            * inverse_scatterer_power_transmission
            * ccw_internal_transfer_numerator
        )
        d_loop_matrix_22_dphi = -inverse_through_self_coupling * (
            d_inverse_scatterer_power_transmission_dphi * ccw_internal_transfer_numerator
            + inverse_scatterer_power_transmission * d_ccw_transfer_numerator_dphi
        )
        off_diagonal_product = (
            scatterer_reflectance
            * inverse_scatterer_power_transmission
            * inverse_scatterer_power_transmission
            * half_path_sum
            * half_path_sum
        )
        d_off_diagonal_product_dphi = (
            (
                d_scatterer_reflectance_dphi * inverse_scatterer_power_transmission
                + 2.0 * scatterer_reflectance * d_inverse_scatterer_power_transmission_dphi
            )
            * inverse_scatterer_power_transmission
            * half_path_sum
            * half_path_sum
            + 2.0
            * scatterer_reflectance
            * inverse_scatterer_power_transmission
            * inverse_scatterer_power_transmission
            * half_path_sum
            * d_half_path_sum_dphi
        )
        inverse_loop_determinant = _complex_reciprocal(loop_matrix_11 * loop_matrix_22 - off_diagonal_product)
        d_loop_determinant_dphi = (
            d_loop_matrix_11_dphi * loop_matrix_22
            + loop_matrix_11 * d_loop_matrix_22_dphi
            - d_off_diagonal_product_dphi
        )
        cw_drive_over_determinant = (
            complex(0.0, -real_workspace[REAL_WORKSPACE_ROW_THROUGH_COUPLER_FIELD_CROSS_COUPLING, line_index])
            * inverse_loop_determinant
        )
        cw_field = loop_matrix_22 * cw_drive_over_determinant
        ccw_loop_coupling = (
            -inverse_through_self_coupling
            * inverse_scatterer_power_transmission
            * half_path_sum
            * scatterer_reflection
        )
        ccw_field = ccw_loop_coupling * cw_drive_over_determinant
        cw_field_row[line_index] = cw_field
        ccw_field_row[line_index] = ccw_field
        d_cw_field_dphi_row[line_index] = (
            d_loop_matrix_22_dphi * cw_drive_over_determinant
            - cw_field * d_loop_determinant_dphi * inverse_loop_determinant
        )
        d_ccw_field_dphi_row[line_index] = (
            -inverse_through_self_coupling
            * (
                (
                    d_inverse_scatterer_power_transmission_dphi * half_path_sum
                    + inverse_scatterer_power_transmission * d_half_path_sum_dphi
                )
                * scatterer_reflection
                + inverse_scatterer_power_transmission * half_path_sum * d_scatterer_reflection_dphi
            )
            * cw_drive_over_determinant
            - ccw_field * d_loop_determinant_dphi * inverse_loop_determinant
        )
    for line_index in range(line_count):
        quarter_propagation = quarter_propagation_row[line_index]
        inverse_quarter_propagation = inverse_quarter_propagation_row[line_index]
        d_quarter_propagation_dphi = -0.25j * quarter_propagation
        d_inverse_quarter_propagation_dphi = 0.25j * inverse_quarter_propagation
        scatterer_reflection = scatterer_reflection_row[line_index]
        d_scatterer_reflection_dphi = d_scatterer_reflection_dphi_row[line_index]
        scatterer_reflection_conj = scatterer_reflection.conjugate()
        d_scatterer_reflection_conj_dphi = d_scatterer_reflection_dphi.conjugate()
        inverse_scatterer_field_transmission = real_workspace[
            REAL_WORKSPACE_ROW_INVERSE_SCATTERER_FIELD_TRANSMISSION, line_index
        ]
        d_inverse_scatterer_transmission_dphi = real_workspace[
            REAL_WORKSPACE_ROW_INVERSE_SCATTERER_FIELD_TRANSMISSION_ROUND_TRIP_PHASE_DERIVATIVE, line_index
        ]
        cw_field = cw_field_row[line_index]
        ccw_field = ccw_field_row[line_index]
        d_cw_field_dphi = d_cw_field_dphi_row[line_index]
        d_ccw_field_dphi = d_ccw_field_dphi_row[line_index]
        through_cross_coupling = real_workspace[
            REAL_WORKSPACE_ROW_THROUGH_COUPLER_FIELD_CROSS_COUPLING, line_index
        ]
        drop_cross_coupling = real_workspace[REAL_WORKSPACE_ROW_DROP_COUPLER_FIELD_CROSS_COUPLING, line_index]
        drop_self_coupling = real_workspace[REAL_WORKSPACE_ROW_DROP_COUPLER_FIELD_SELF_COUPLING, line_index]
        power_energy_response[RING_RESPONSE_ROW_REFLECTED_POWER_FRACTION, line_index] = (
            through_cross_coupling * through_cross_coupling * _squared_magnitude(ccw_field)
        )
        cw_segment_power_sum = _squared_magnitude(cw_field)
        d_cw_segment_power_sum_dphi = _squared_magnitude_derivative(cw_field, d_cw_field_dphi)
        d_cw_field_dphi = d_quarter_propagation_dphi * cw_field + quarter_propagation * d_cw_field_dphi
        cw_field = quarter_propagation * cw_field
        d_ccw_field_dphi = (
            d_ccw_field_dphi * inverse_quarter_propagation + ccw_field * d_inverse_quarter_propagation_dphi
        )
        ccw_field = ccw_field * inverse_quarter_propagation
        ccw_segment_power_sum = _squared_magnitude(ccw_field)
        d_ccw_segment_power_sum_dphi = _squared_magnitude_derivative(ccw_field, d_ccw_field_dphi)
        cw_scatter_numerator = cw_field - scatterer_reflection_conj * ccw_field
        ccw_scatter_numerator = ccw_field - scatterer_reflection * cw_field
        d_cw_field_dphi, d_ccw_field_dphi = (
            (
                d_cw_field_dphi
                - d_scatterer_reflection_conj_dphi * ccw_field
                - scatterer_reflection_conj * d_ccw_field_dphi
            )
            * inverse_scatterer_field_transmission
            + cw_scatter_numerator * d_inverse_scatterer_transmission_dphi,
            (
                d_ccw_field_dphi
                - d_scatterer_reflection_dphi * cw_field
                - scatterer_reflection * d_cw_field_dphi
            )
            * inverse_scatterer_field_transmission
            + ccw_scatter_numerator * d_inverse_scatterer_transmission_dphi,
        )
        cw_field = cw_scatter_numerator * inverse_scatterer_field_transmission
        ccw_field = ccw_scatter_numerator * inverse_scatterer_field_transmission
        cw_segment_power_sum += _squared_magnitude(cw_field)
        d_cw_segment_power_sum_dphi += _squared_magnitude_derivative(cw_field, d_cw_field_dphi)
        d_cw_field_dphi = d_quarter_propagation_dphi * cw_field + quarter_propagation * d_cw_field_dphi
        cw_field = quarter_propagation * cw_field
        d_ccw_field_dphi = (
            d_ccw_field_dphi * inverse_quarter_propagation + ccw_field * d_inverse_quarter_propagation_dphi
        )
        ccw_field = ccw_field * inverse_quarter_propagation
        ccw_segment_power_sum += _squared_magnitude(ccw_field)
        d_ccw_segment_power_sum_dphi += _squared_magnitude_derivative(ccw_field, d_ccw_field_dphi)
        power_energy_response[RING_RESPONSE_ROW_CLOCKWISE_DROP_POWER_FRACTION, line_index] = (
            drop_cross_coupling * drop_cross_coupling * _squared_magnitude(cw_field)
        )
        d_cw_field_dphi = drop_self_coupling * d_cw_field_dphi
        cw_field = drop_self_coupling * cw_field
        d_ccw_field_dphi = d_ccw_field_dphi / drop_self_coupling
        ccw_field = ccw_field / drop_self_coupling
        power_energy_response[RING_RESPONSE_ROW_COUNTERCLOCKWISE_ADD_PORT_POWER_FRACTION, line_index] = (
            drop_cross_coupling * drop_cross_coupling * _squared_magnitude(ccw_field)
        )
        cw_field_row[line_index] = cw_field
        ccw_field_row[line_index] = ccw_field
        d_cw_field_dphi_row[line_index] = d_cw_field_dphi
        d_ccw_field_dphi_row[line_index] = d_ccw_field_dphi
        real_workspace[REAL_WORKSPACE_ROW_CLOCKWISE_SEGMENT_INPUT_POWER_SUM, line_index] = (
            cw_segment_power_sum
        )
        real_workspace[REAL_WORKSPACE_ROW_COUNTERCLOCKWISE_SEGMENT_INPUT_POWER_SUM, line_index] = (
            ccw_segment_power_sum
        )
        real_workspace[
            REAL_WORKSPACE_ROW_CLOCKWISE_SEGMENT_INPUT_POWER_SUM_ROUND_TRIP_PHASE_DERIVATIVE, line_index
        ] = d_cw_segment_power_sum_dphi
        real_workspace[
            REAL_WORKSPACE_ROW_COUNTERCLOCKWISE_SEGMENT_INPUT_POWER_SUM_ROUND_TRIP_PHASE_DERIVATIVE,
            line_index,
        ] = d_ccw_segment_power_sum_dphi
    for line_index in range(line_count):
        quarter_propagation = quarter_propagation_row[line_index]
        inverse_quarter_propagation = inverse_quarter_propagation_row[line_index]
        d_quarter_propagation_dphi = -0.25j * quarter_propagation
        d_inverse_quarter_propagation_dphi = 0.25j * inverse_quarter_propagation
        scatterer_reflection = scatterer_reflection_row[line_index]
        d_scatterer_reflection_dphi = d_scatterer_reflection_dphi_row[line_index]
        scatterer_reflection_conj = scatterer_reflection.conjugate()
        d_scatterer_reflection_conj_dphi = d_scatterer_reflection_dphi.conjugate()
        inverse_scatterer_field_transmission = real_workspace[
            REAL_WORKSPACE_ROW_INVERSE_SCATTERER_FIELD_TRANSMISSION, line_index
        ]
        d_inverse_scatterer_transmission_dphi = real_workspace[
            REAL_WORKSPACE_ROW_INVERSE_SCATTERER_FIELD_TRANSMISSION_ROUND_TRIP_PHASE_DERIVATIVE, line_index
        ]
        cw_field = cw_field_row[line_index]
        ccw_field = ccw_field_row[line_index]
        d_cw_field_dphi = d_cw_field_dphi_row[line_index]
        d_ccw_field_dphi = d_ccw_field_dphi_row[line_index]
        cw_segment_power_sum = real_workspace[
            REAL_WORKSPACE_ROW_CLOCKWISE_SEGMENT_INPUT_POWER_SUM, line_index
        ] + _squared_magnitude(cw_field)
        d_cw_segment_power_sum_dphi = real_workspace[
            REAL_WORKSPACE_ROW_CLOCKWISE_SEGMENT_INPUT_POWER_SUM_ROUND_TRIP_PHASE_DERIVATIVE, line_index
        ] + _squared_magnitude_derivative(cw_field, d_cw_field_dphi)
        d_cw_field_dphi = d_quarter_propagation_dphi * cw_field + quarter_propagation * d_cw_field_dphi
        cw_field = quarter_propagation * cw_field
        d_ccw_field_dphi = (
            d_ccw_field_dphi * inverse_quarter_propagation + ccw_field * d_inverse_quarter_propagation_dphi
        )
        ccw_field = ccw_field * inverse_quarter_propagation
        ccw_segment_power_sum = real_workspace[
            REAL_WORKSPACE_ROW_COUNTERCLOCKWISE_SEGMENT_INPUT_POWER_SUM, line_index
        ] + _squared_magnitude(ccw_field)
        d_ccw_segment_power_sum_dphi = real_workspace[
            REAL_WORKSPACE_ROW_COUNTERCLOCKWISE_SEGMENT_INPUT_POWER_SUM_ROUND_TRIP_PHASE_DERIVATIVE,
            line_index,
        ] + _squared_magnitude_derivative(ccw_field, d_ccw_field_dphi)
        cw_scatter_numerator = cw_field - scatterer_reflection_conj * ccw_field
        ccw_scatter_numerator = ccw_field - scatterer_reflection * cw_field
        d_cw_field_dphi, d_ccw_field_dphi = (
            (
                d_cw_field_dphi
                - d_scatterer_reflection_conj_dphi * ccw_field
                - scatterer_reflection_conj * d_ccw_field_dphi
            )
            * inverse_scatterer_field_transmission
            + cw_scatter_numerator * d_inverse_scatterer_transmission_dphi,
            (
                d_ccw_field_dphi
                - d_scatterer_reflection_dphi * cw_field
                - scatterer_reflection * d_cw_field_dphi
            )
            * inverse_scatterer_field_transmission
            + ccw_scatter_numerator * d_inverse_scatterer_transmission_dphi,
        )
        cw_field = cw_scatter_numerator * inverse_scatterer_field_transmission
        ccw_field = ccw_scatter_numerator * inverse_scatterer_field_transmission
        cw_segment_power_sum += _squared_magnitude(cw_field)
        d_cw_segment_power_sum_dphi += _squared_magnitude_derivative(cw_field, d_cw_field_dphi)
        d_ccw_field_dphi = (
            d_ccw_field_dphi * inverse_quarter_propagation + ccw_field * d_inverse_quarter_propagation_dphi
        )
        cw_field = quarter_propagation * cw_field
        ccw_field = ccw_field * inverse_quarter_propagation
        ccw_segment_power_sum += _squared_magnitude(ccw_field)
        d_ccw_segment_power_sum_dphi += _squared_magnitude_derivative(ccw_field, d_ccw_field_dphi)
        power_energy_response[RING_RESPONSE_ROW_THROUGH_POWER_FRACTION, line_index] = _squared_magnitude(
            real_workspace[REAL_WORKSPACE_ROW_THROUGH_COUPLER_FIELD_SELF_COUPLING, line_index]
            - 1j
            * real_workspace[REAL_WORKSPACE_ROW_THROUGH_COUPLER_FIELD_CROSS_COUPLING, line_index]
            * cw_field
        )
        quarter_ring_energy_per_power, quarter_ring_loss_fraction = calculate_quarter_ring_energy_per_power_and_loss_fraction(
            real_workspace[REAL_WORKSPACE_ROW_ROUND_TRIP_FIELD_LOSS_EXPONENT, line_index],
            round_trip_delay_row_s[line_index],
        )
        power_energy_response[RING_RESPONSE_ROW_CLOCKWISE_STORED_ENERGY_PER_INPUT_POWER, line_index] = (
            cw_segment_power_sum * quarter_ring_energy_per_power
        )
        power_energy_response[
            RING_RESPONSE_ROW_COUNTERCLOCKWISE_STORED_ENERGY_PER_INPUT_POWER, line_index
        ] = (ccw_segment_power_sum * quarter_ring_energy_per_power)
        power_energy_response[RING_RESPONSE_ROW_INTRINSIC_LOSS_POWER_FRACTION, line_index] = (
            cw_segment_power_sum + ccw_segment_power_sum
        ) * quarter_ring_loss_fraction
        power_energy_response[
            RING_RESPONSE_ROW_INTRINSIC_LOSS_POWER_FRACTION_ROUND_TRIP_PHASE_DERIVATIVE, line_index
        ] = (d_cw_segment_power_sum_dphi + d_ccw_segment_power_sum_dphi) * quarter_ring_loss_fraction


# calculate one ring's response at every laser line, for one data level and one input watt.
# exact optics uses the same scattering solve for zero and nonzero backscatter.
# with_d requests the loss phase slope.
# write res[RING_RESPONSE_*, line].
# port entries are power fractions, energy entries are J/W, and the loss slope is per radian.
# inputs are per-line phase/coupling/loss/delay rows - rbk contains that ring's per-order backscatter.
# complex_workspace and real_workspace are reusable workspaces, not physical outputs.
# also doing similar things to the things above.
# x
@njit(cache=True, error_model="numpy", _nrt=False)
def evaluate_ring_response_per_input_watt(
    phi_row: Float64Vector,  # phi_row = phi (rad) - round-trip phase before adding coupler phase - type float64 ndarray, shape (L,)
    psi_row: Float64Vector,  # psi_row = psi (rad) - additional coupler phase - type float64 ndarray, shape (L,)
    k2t_row: Float64Vector,  # k2t_row = K_th^2 (1) - through-coupler power fraction - type float64 ndarray, shape (L,)
    k2d_row: Float64Vector,  # k2d_row = K_dr^2 (1) - drop-coupler power fraction - type float64 ndarray, shape (L,)
    a_row: Float64Vector,  # a_row = a (1) - retained round-trip field amplitude - type float64 ndarray, shape (L,)
    trt_row: Float64Vector,  # trt_row = T_rt (s) - round-trip group delay - type float64 ndarray, shape (L,)
    rbk: Complex128Vector,  # rbk = r_b (1) - complex round-trip backscatter coupling by resonance order - type complex128 ndarray, shape (backscatter_order_count,)
    use_tcmt: bool,  # use_tcmt = reserved flag (1) - must be false to preserve the compiled call layout - type bool scalar
    m_orders: int,  # m_orders = reserved count (1) - retained for existing call layouts - type int scalar
    with_d: bool,  # with_d = slope_enabled (1) - also calculate the intrinsic-loss phase slope - type bool scalar
    complex_workspace: ComplexWorkspace,  # complex_workspace = C_work (mixed units) - COMPLEX_WORKSPACE_* factors, fields, and field phase slopes - type complex128 ndarray, capacity at least (8, L)
    real_workspace: RealWorkspace,  # real_workspace = F_work (mixed units) - REAL_WORKSPACE_* factors, powers, slopes, and RING_RESPONSE_* response blocks - type float64 ndarray, capacity at least (27, L)
    res: RingResponseBlock,  # res = R_opt (mixed units) - RING_RESPONSE_* power fractions (1), energies (J/W), and loss slope (1/rad) - type float64 ndarray, capacity at least (8, L)
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    line_count = phi_row.shape[0]
    if use_tcmt:
        raise ValueError("TCMT is archived - use exact ring optics")
    _validate_ring_workspace_buffers(
        line_count,
        complex_workspace,
        real_workspace,
        trt_row,
        res,
    )
    for line_index in range(line_count):
        round_trip_field_loss_exponent = -math.log(a_row[line_index])
        real_workspace[REAL_WORKSPACE_ROW_ROUND_TRIP_FIELD_LOSS_EXPONENT, line_index] = round_trip_field_loss_exponent
        real_workspace[REAL_WORKSPACE_ROW_THROUGH_COUPLER_FIELD_SELF_COUPLING, line_index] = math.sqrt(
            1.0 - k2t_row[line_index]
        )
        real_workspace[REAL_WORKSPACE_ROW_THROUGH_COUPLER_FIELD_CROSS_COUPLING, line_index] = math.sqrt(
            k2t_row[line_index]
        )
        real_workspace[REAL_WORKSPACE_ROW_DROP_COUPLER_FIELD_SELF_COUPLING, line_index] = math.sqrt(
            1.0 - k2d_row[line_index]
        )
        real_workspace[REAL_WORKSPACE_ROW_DROP_COUPLER_FIELD_CROSS_COUPLING, line_index] = math.sqrt(k2d_row[line_index])
        (
            quarter_propagation,
            inverse_quarter_propagation,
            scatterer_reflection,
            d_scatterer_reflection_dphi,
        ) = calculate_propagation_and_scatterer_coefficients(phi_row[line_index] + psi_row[line_index], rbk, round_trip_field_loss_exponent)
        complex_workspace[COMPLEX_WORKSPACE_ROW_QUARTER_RING_FIELD_PROPAGATION_FACTOR, line_index] = quarter_propagation
        complex_workspace[COMPLEX_WORKSPACE_ROW_INVERSE_QUARTER_RING_FIELD_PROPAGATION_FACTOR, line_index] = (
            inverse_quarter_propagation
        )
        complex_workspace[COMPLEX_WORKSPACE_ROW_SCATTERER_FIELD_REFLECTION_COEFFICIENT, line_index] = scatterer_reflection
        complex_workspace[COMPLEX_WORKSPACE_ROW_SCATTERER_FIELD_REFLECTION_ROUND_TRIP_PHASE_DERIVATIVE, line_index] = (
            d_scatterer_reflection_dphi
        )
    if with_d:
        solve_ring_fields_and_loss_phase_derivative_at_all_laser_wavelengths(line_count, complex_workspace, real_workspace, trt_row, res)
    else:
        solve_ring_fields_at_all_laser_wavelengths(line_count, complex_workspace, real_workspace, trt_row, res)


# one ring at every line (used by the bus and the self-heating solve)


# evaluate one ring at every laser wavelength and write its optical outputs.
# scale the response per input watt by the power arriving at this ring.
# average binary data levels with p1 when two_level is enabled, or use level zero alone.
# write powers and stored energies by wavelength to out_rl, and summed powers to out_r.
# write through power to p_thru for the next ring - p_thru may share storage with pin.
# with_slope also writes absorbed-power derivatives with respect to phase and resonance wavelength.
# update only ring k and reuse the supplied workspaces.
# x
@njit(cache=True, error_model="numpy", _nrt=False)
def evaluate_ring_optical_outputs(
    k: int,  # k = k (1) - physical ring index - type int scalar
    pin: Float64Vector,  # pin = P_in,k (W) - power delivered to this ring per laser line - type float64 ndarray, shape (L,)
    p1: float,  # p1 = p_1 (1) - probability of data level one - type float scalar
    phi0: Float64Matrix,  # phi0 = phi_0 (rad) - round-trip phase at data level zero - type float64 ndarray, shape (N, L)
    phi1: Float64Matrix,  # phi1 = phi_1 (rad) - round-trip phase at data level one - type float64 ndarray, shape (N, L)
    a0: Float64Matrix,  # a0 = a_0 (1) - retained round-trip field amplitude at data level zero - type float64 ndarray, shape (N, L)
    a1: Float64Matrix,  # a1 = a_1 (1) - retained round-trip field amplitude at data level one - type float64 ndarray, shape (N, L)
    eta0: Float64Matrix,  # eta0 = eta_0 (1) - absorbed share of intrinsic loss at data level zero - type float64 ndarray, shape (N, L)
    eta1: Float64Matrix,  # eta1 = eta_1 (1) - absorbed share of intrinsic loss at data level one - type float64 ndarray, shape (N, L)
    psi: Float64Matrix,  # psi = psi (rad) - additional coupler phase - type float64 ndarray, shape (N, L)
    k2_th: Float64Matrix,  # k2_th = K_th^2 (1) - through-coupler power fraction - type float64 ndarray, shape (N, L)
    k2_dr: Float64Matrix,  # k2_dr = K_dr^2 (1) - drop-coupler power fraction - type float64 ndarray, shape (N, L)
    t_rt: Float64Matrix,  # t_rt = T_rt (s) - round-trip group delay - type float64 ndarray, shape (N, L)
    rb: Complex128Matrix,  # rb = r_b (1) - complex round-trip backscatter coupling by resonance order - type complex128 ndarray, shape (N, backscatter_order_count)
    dphi_dlamres: Float64Matrix,  # dphi_dlamres = dphi/dlambda_r (rad/m) - phase slope for a resonance shift - type float64 ndarray, shape (N, L)
    two_level: bool,  # two_level = nrz_enabled (1) - evaluate and average both binary data levels - type bool scalar
    use_tcmt: bool,  # use_tcmt = reserved flag (1) - must be false to preserve the compiled call layout - type bool scalar
    m_orders: int,  # m_orders = reserved count (1) - retained for existing call layouts - type int scalar
    with_slope: bool,  # with_slope = slope_enabled (1) - also calculate absorbed-power slopes - type bool scalar
    out_rl: RingLineOutputs,  # out_rl = Y_ring,line (mixed units) - RING_LINE_OUTPUT_* powers (W), energies (J), and power slope (W/rad) - type float64 ndarray, shape (16, N, L)
    out_r: RingOutputs,  # out_r = Y_ring (mixed units) - RING_OUTPUT_* powers (W) and resonance-wavelength slope (W/m) - type float64 ndarray, shape (6, N)
    p_thru: Float64Vector,  # p_thru = P_through (W) - per-line power sent toward the next ring - type float64 ndarray, shape (L,)
    complex_workspace: ComplexWorkspace,  # complex_workspace = C_work (mixed units) - COMPLEX_WORKSPACE_* factors, fields, and field phase slopes - type complex128 ndarray, capacity at least (8, L)
    real_workspace: RealWorkspace,  # real_workspace = F_work (mixed units) - REAL_WORKSPACE_* factors, powers, slopes, and RING_RESPONSE_* response blocks - type float64 ndarray, capacity at least (27, L)
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    line_count = pin.shape[0]
    ring_backscatter = rb[k]
    level0_response = real_workspace[
        REAL_WORKSPACE_LEVEL_ZERO_RESPONSE_START_ROW : REAL_WORKSPACE_LEVEL_ZERO_RESPONSE_START_ROW
        + RING_RESPONSE_ROW_COUNT
    ]
    level1_response = real_workspace[
        REAL_WORKSPACE_LEVEL_ONE_RESPONSE_START_ROW : REAL_WORKSPACE_LEVEL_ONE_RESPONSE_START_ROW
        + RING_RESPONSE_ROW_COUNT
    ]
    # solve the response per input watt at data level zero.
    evaluate_ring_response_per_input_watt(
        phi0[k],
        psi[k],
        k2_th[k],
        k2_dr[k],
        a0[k],
        t_rt[k],
        ring_backscatter,
        use_tcmt,
        m_orders,
        with_slope,
        complex_workspace,
        real_workspace,
        level0_response,
    )
    # solve data level one only when binary modulation is enabled.
    if two_level:
        evaluate_ring_response_per_input_watt(
            phi1[k],
            psi[k],
            k2_th[k],
            k2_dr[k],
            a1[k],
            t_rt[k],
            ring_backscatter,
            use_tcmt,
            m_orders,
            with_slope,
            complex_workspace,
            real_workspace,
            level1_response,
        )
        one_probability = p1
    else:
        one_probability = 0.0
    zero_probability = 1.0 - one_probability  # NRZ weights - q1 of the time at level 1, q0 at level 0
    total_drop_power_w = 0.0  # sums over the lines - drop, absorbed, loss, drop at level 1 and at level 0
    total_absorbed_power_w = 0.0
    total_lost_power_w = 0.0
    level1_drop_power_w = 0.0
    level0_drop_power_w = 0.0
    d_absorbed_power_d_resonance_wavelength = 0.0
    # scale each wavelength response to its actual input power and average the data levels.
    for line_index in range(line_count):
        input_power_w = pin[line_index]
        # use the level-zero response for both levels when binary modulation is disabled.
        if two_level:
            level1_through_fraction = level1_response[RING_RESPONSE_ROW_THROUGH_POWER_FRACTION, line_index]
            level1_drop_fraction = level1_response[
                RING_RESPONSE_ROW_CLOCKWISE_DROP_POWER_FRACTION, line_index
            ]
            level1_add_fraction = level1_response[
                RING_RESPONSE_ROW_COUNTERCLOCKWISE_ADD_PORT_POWER_FRACTION, line_index
            ]
            level1_reflection_fraction = level1_response[
                RING_RESPONSE_ROW_REFLECTED_POWER_FRACTION, line_index
            ]
            level1_loss_fraction = level1_response[
                RING_RESPONSE_ROW_INTRINSIC_LOSS_POWER_FRACTION, line_index
            ]
            level1_cw_energy_per_power = level1_response[
                RING_RESPONSE_ROW_CLOCKWISE_STORED_ENERGY_PER_INPUT_POWER, line_index
            ]
            level1_ccw_energy_per_power = level1_response[
                RING_RESPONSE_ROW_COUNTERCLOCKWISE_STORED_ENERGY_PER_INPUT_POWER, line_index
            ]
            d_level1_loss_fraction_dphi = level1_response[
                RING_RESPONSE_ROW_INTRINSIC_LOSS_POWER_FRACTION_ROUND_TRIP_PHASE_DERIVATIVE, line_index
            ]
            level1_absorbed_share_of_loss = eta1[k, line_index]
        else:
            level1_through_fraction = level0_response[RING_RESPONSE_ROW_THROUGH_POWER_FRACTION, line_index]
            level1_drop_fraction = level0_response[
                RING_RESPONSE_ROW_CLOCKWISE_DROP_POWER_FRACTION, line_index
            ]
            level1_add_fraction = level0_response[
                RING_RESPONSE_ROW_COUNTERCLOCKWISE_ADD_PORT_POWER_FRACTION, line_index
            ]
            level1_reflection_fraction = level0_response[
                RING_RESPONSE_ROW_REFLECTED_POWER_FRACTION, line_index
            ]
            level1_loss_fraction = level0_response[
                RING_RESPONSE_ROW_INTRINSIC_LOSS_POWER_FRACTION, line_index
            ]
            level1_cw_energy_per_power = level0_response[
                RING_RESPONSE_ROW_CLOCKWISE_STORED_ENERGY_PER_INPUT_POWER, line_index
            ]
            level1_ccw_energy_per_power = level0_response[
                RING_RESPONSE_ROW_COUNTERCLOCKWISE_STORED_ENERGY_PER_INPUT_POWER, line_index
            ]
            d_level1_loss_fraction_dphi = level0_response[
                RING_RESPONSE_ROW_INTRINSIC_LOSS_POWER_FRACTION_ROUND_TRIP_PHASE_DERIVATIVE, line_index
            ]
            level1_absorbed_share_of_loss = eta0[k, line_index]
        level0_through_fraction = level0_response[RING_RESPONSE_ROW_THROUGH_POWER_FRACTION, line_index]
        level0_drop_fraction = level0_response[RING_RESPONSE_ROW_CLOCKWISE_DROP_POWER_FRACTION, line_index]
        level0_absorbed_share_of_loss = eta0[k, line_index]
        out_rl[RING_LINE_OUTPUT_ROW_INPUT_POWER, k, line_index] = input_power_w
        out_rl[RING_LINE_OUTPUT_ROW_THROUGH_POWER, k, line_index] = input_power_w * (
            one_probability * level1_through_fraction + zero_probability * level0_through_fraction
        )
        out_rl[RING_LINE_OUTPUT_ROW_CLOCKWISE_DROP_POWER, k, line_index] = input_power_w * (
            one_probability * level1_drop_fraction + zero_probability * level0_drop_fraction
        )
        out_rl[RING_LINE_OUTPUT_ROW_COUNTERCLOCKWISE_ADD_PORT_POWER, k, line_index] = input_power_w * (
            one_probability * level1_add_fraction
            + zero_probability
            * level0_response[RING_RESPONSE_ROW_COUNTERCLOCKWISE_ADD_PORT_POWER_FRACTION, line_index]
        )
        out_rl[RING_LINE_OUTPUT_ROW_REFLECTED_POWER, k, line_index] = input_power_w * (
            one_probability * level1_reflection_fraction
            + zero_probability * level0_response[RING_RESPONSE_ROW_REFLECTED_POWER_FRACTION, line_index]
        )
        out_rl[RING_LINE_OUTPUT_ROW_INTRINSIC_LOSS_POWER, k, line_index] = input_power_w * (
            one_probability * level1_loss_fraction
            + zero_probability * level0_response[RING_RESPONSE_ROW_INTRINSIC_LOSS_POWER_FRACTION, line_index]
        )
        out_rl[RING_LINE_OUTPUT_ROW_ABSORBED_POWER, k, line_index] = input_power_w * (
            one_probability * level1_absorbed_share_of_loss * level1_loss_fraction
            + zero_probability
            * level0_absorbed_share_of_loss
            * level0_response[RING_RESPONSE_ROW_INTRINSIC_LOSS_POWER_FRACTION, line_index]
        )
        out_rl[RING_LINE_OUTPUT_ROW_CLOCKWISE_STORED_ENERGY, k, line_index] = input_power_w * (
            one_probability * level1_cw_energy_per_power
            + zero_probability
            * level0_response[RING_RESPONSE_ROW_CLOCKWISE_STORED_ENERGY_PER_INPUT_POWER, line_index]
        )
        out_rl[RING_LINE_OUTPUT_ROW_COUNTERCLOCKWISE_STORED_ENERGY, k, line_index] = input_power_w * (
            one_probability * level1_ccw_energy_per_power
            + zero_probability
            * level0_response[RING_RESPONSE_ROW_COUNTERCLOCKWISE_STORED_ENERGY_PER_INPUT_POWER, line_index]
        )
        out_rl[RING_LINE_OUTPUT_ROW_LEVEL_ONE_THROUGH_POWER, k, line_index] = (
            input_power_w * level1_through_fraction
        )
        out_rl[RING_LINE_OUTPUT_ROW_LEVEL_ZERO_THROUGH_POWER, k, line_index] = (
            input_power_w * level0_through_fraction
        )
        out_rl[RING_LINE_OUTPUT_ROW_LEVEL_ONE_CLOCKWISE_DROP_POWER, k, line_index] = (
            input_power_w * level1_drop_fraction
        )
        out_rl[RING_LINE_OUTPUT_ROW_LEVEL_ZERO_CLOCKWISE_DROP_POWER, k, line_index] = (
            input_power_w * level0_drop_fraction
        )
        # convert the absorbed-power phase derivative to a resonance-wavelength derivative.
        if with_slope:
            d_absorbed_power_dphi = input_power_w * (
                one_probability * level1_absorbed_share_of_loss * d_level1_loss_fraction_dphi
                + zero_probability
                * level0_absorbed_share_of_loss
                * level0_response[
                    RING_RESPONSE_ROW_INTRINSIC_LOSS_POWER_FRACTION_ROUND_TRIP_PHASE_DERIVATIVE, line_index
                ]
            )
            out_rl[RING_LINE_OUTPUT_ROW_ABSORBED_POWER_ROUND_TRIP_PHASE_DERIVATIVE, k, line_index] = (
                d_absorbed_power_dphi
            )
            d_absorbed_power_d_resonance_wavelength += d_absorbed_power_dphi * dphi_dlamres[k, line_index]
        else:
            out_rl[RING_LINE_OUTPUT_ROW_ABSORBED_POWER_ROUND_TRIP_PHASE_DERIVATIVE, k, line_index] = 0.0
        total_drop_power_w += out_rl[RING_LINE_OUTPUT_ROW_CLOCKWISE_DROP_POWER, k, line_index]
        total_absorbed_power_w += out_rl[RING_LINE_OUTPUT_ROW_ABSORBED_POWER, k, line_index]
        total_lost_power_w += out_rl[RING_LINE_OUTPUT_ROW_INTRINSIC_LOSS_POWER, k, line_index]
        level1_drop_power_w += input_power_w * level1_drop_fraction
        level0_drop_power_w += input_power_w * level0_drop_fraction
        p_thru[line_index] = out_rl[RING_LINE_OUTPUT_ROW_THROUGH_POWER, k, line_index]
    # save this ring's totals across all laser wavelengths.
    out_r[RING_OUTPUT_ROW_TOTAL_CLOCKWISE_DROP_POWER, k] = total_drop_power_w
    out_r[RING_OUTPUT_ROW_TOTAL_ABSORBED_POWER, k] = total_absorbed_power_w
    out_r[RING_OUTPUT_ROW_TOTAL_INTRINSIC_LOSS_POWER, k] = total_lost_power_w
    out_r[RING_OUTPUT_ROW_LEVEL_ONE_TOTAL_CLOCKWISE_DROP_POWER, k] = level1_drop_power_w
    out_r[RING_OUTPUT_ROW_LEVEL_ZERO_TOTAL_CLOCKWISE_DROP_POWER, k] = level0_drop_power_w
    out_r[RING_OUTPUT_ROW_ABSORBED_POWER_RESONANCE_WAVELENGTH_DERIVATIVE, k] = (
        d_absorbed_power_d_resonance_wavelength
    )


# calculate absorbed power and its resonance-wavelength slope for one ring.
# hold its input power fixed - upstream rings have already set that power.
# average binary levels, then sum every line and travelling direction.
# return power in W and slope in W/m.
# couplings, attenuation, and absorbed-loss fractions stay fixed during this derivative - phase changes with resonance wavelength.
# this is the slope used by the thermal solver.
# absorbed power sensitivity calculation
# (todo) maybe wrap things in enums to make it easier to understand what's going on clearly?
# x
@njit(cache=True, error_model="numpy", _nrt=False)
def evaluate_ring_absorbed_power_and_resonance_wavelength_derivative(
    k: int,  # k = k (1) - physical ring index - type int scalar
    pin: Float64Vector,  # pin = P_in,k (W) - power delivered to this ring per laser line - type float64 ndarray, shape (L,)
    p1: float,  # p1 = p_1 (1) - probability of data level one - type float scalar
    phi0: Float64Matrix,  # phi0 = phi_0 (rad) - round-trip phase at data level zero - type float64 ndarray, shape (N, L)
    phi1: Float64Matrix,  # phi1 = phi_1 (rad) - round-trip phase at data level one - type float64 ndarray, shape (N, L)
    a0: Float64Matrix,  # a0 = a_0 (1) - retained round-trip field amplitude at data level zero - type float64 ndarray, shape (N, L)
    a1: Float64Matrix,  # a1 = a_1 (1) - retained round-trip field amplitude at data level one - type float64 ndarray, shape (N, L)
    eta0: Float64Matrix,  # eta0 = eta_0 (1) - absorbed share of intrinsic loss at data level zero - type float64 ndarray, shape (N, L)
    eta1: Float64Matrix,  # eta1 = eta_1 (1) - absorbed share of intrinsic loss at data level one - type float64 ndarray, shape (N, L)
    psi: Float64Matrix,  # psi = psi (rad) - additional coupler phase - type float64 ndarray, shape (N, L)
    k2_th: Float64Matrix,  # k2_th = K_th^2 (1) - through-coupler power fraction - type float64 ndarray, shape (N, L)
    k2_dr: Float64Matrix,  # k2_dr = K_dr^2 (1) - drop-coupler power fraction - type float64 ndarray, shape (N, L)
    t_rt: Float64Matrix,  # t_rt = T_rt (s) - round-trip group delay - type float64 ndarray, shape (N, L)
    rb: Complex128Matrix,  # rb = r_b (1) - complex round-trip backscatter coupling by resonance order - type complex128 ndarray, shape (N, backscatter_order_count)
    dphi_dlamres: Float64Matrix,  # dphi_dlamres = dphi/dlambda_r (rad/m) - phase slope for a resonance shift - type float64 ndarray, shape (N, L)
    two_level: bool,  # two_level = nrz_enabled (1) - evaluate and average both binary data levels - type bool scalar
    use_tcmt: bool,  # use_tcmt = reserved flag (1) - must be false to preserve the compiled call layout - type bool scalar
    m_orders: int,  # m_orders = reserved count (1) - retained for existing call layouts - type int scalar
    complex_workspace: ComplexWorkspace,  # complex_workspace = C_work (mixed units) - COMPLEX_WORKSPACE_* factors, fields, and field phase slopes - type complex128 ndarray, capacity at least (8, L)
    real_workspace: RealWorkspace,  # real_workspace = F_work (mixed units) - REAL_WORKSPACE_* factors, powers, slopes, and RING_RESPONSE_* response blocks - type float64 ndarray, capacity at least (27, L)
) -> tuple[
    float,  # return[0] = P_abs (W) - total absorbed power of ring k - type float scalar
    float,  # return[1] = dP_abs/dlambda_r (W/m) - rigid resonance-shift derivative - type float scalar
]:
    line_count = pin.shape[0]
    ring_backscatter = rb[k]
    level0_response = real_workspace[
        REAL_WORKSPACE_LEVEL_ZERO_RESPONSE_START_ROW : REAL_WORKSPACE_LEVEL_ZERO_RESPONSE_START_ROW
        + RING_RESPONSE_ROW_COUNT
    ]
    level1_response = real_workspace[
        REAL_WORKSPACE_LEVEL_ONE_RESPONSE_START_ROW : REAL_WORKSPACE_LEVEL_ONE_RESPONSE_START_ROW
        + RING_RESPONSE_ROW_COUNT
    ]
    evaluate_ring_response_per_input_watt(
        phi0[k],
        psi[k],
        k2_th[k],
        k2_dr[k],
        a0[k],
        t_rt[k],
        ring_backscatter,
        use_tcmt,
        m_orders,
        True,
        complex_workspace,
        real_workspace,
        level0_response,
    )
    one_probability = 0.0
    if two_level:
        evaluate_ring_response_per_input_watt(
            phi1[k],
            psi[k],
            k2_th[k],
            k2_dr[k],
            a1[k],
            t_rt[k],
            ring_backscatter,
            use_tcmt,
            m_orders,
            True,
            complex_workspace,
            real_workspace,
            level1_response,
        )
        one_probability = p1
    zero_probability = 1.0 - one_probability
    absorbed_power_w = 0.0
    d_absorbed_power_d_resonance_wavelength = 0.0
    for line_index in range(line_count):
        absorbed_input_fraction = (
            zero_probability
            * eta0[k, line_index]
            * level0_response[RING_RESPONSE_ROW_INTRINSIC_LOSS_POWER_FRACTION, line_index]
        )
        d_absorbed_fraction_dphi = (
            zero_probability
            * eta0[k, line_index]
            * level0_response[
                RING_RESPONSE_ROW_INTRINSIC_LOSS_POWER_FRACTION_ROUND_TRIP_PHASE_DERIVATIVE, line_index
            ]
        )
        if two_level:
            absorbed_input_fraction += (
                one_probability
                * eta1[k, line_index]
                * level1_response[RING_RESPONSE_ROW_INTRINSIC_LOSS_POWER_FRACTION, line_index]
            )
            d_absorbed_fraction_dphi += (
                one_probability
                * eta1[k, line_index]
                * level1_response[
                    RING_RESPONSE_ROW_INTRINSIC_LOSS_POWER_FRACTION_ROUND_TRIP_PHASE_DERIVATIVE, line_index
                ]
            )
        absorbed_power_w += pin[line_index] * absorbed_input_fraction
        d_absorbed_power_d_resonance_wavelength += (
            pin[line_index] * d_absorbed_fraction_dphi * dphi_dlamres[k, line_index]
        )
    return (
        absorbed_power_w,
        d_absorbed_power_d_resonance_wavelength,
    )


# bound a ring's absorbed power and phase curvature at fixed input power.
# return (peak power in W, maximum absolute phase curvature in W/rad²).
# the thermal solver uses these bounds to keep Newton steps inside intervals that can contain a root.
# sum laser-line contributions and average binary data levels with p1.
# without backscatter, loss fraction = C/den(phi), where r = t1*t2*a, C = k2_th*(1-a)*(1+t2**2*a), and den(phi) = 1-2*r*cos(phi)+r**2.
# the peak is C/(1-r)**2.
# maximum absolute curvature is 2*r*C/(1-r)**4.
# backscatter splits a peak into two half-weight Lorentzians of the same width.
# this reduces peak and curvature in TCMT.
# the exact ring has O(1/finesse²) corrections.
# allow margins of 5 % for power and 25 % for curvature.
# cap each line's loss fraction at one to preserve the energy-conservation limit.
# test_optics.py checks random rings.
# see docs/research/03_mrm_compact_models.md, section 2.5, for the split-mode response.
# the backscatter bound is verified only for |rb| <= MAX_ROUND_TRIP_BACKSCATTER_FOR_VERIFIED_POWER_BOUNDS = 1 at every order.
# larger values reshape the response and can exceed the curvature bound.
# test_optics.py includes |rb| = 1.6 as a counterexample.
# multiply phase curvature by (dphi/dT)**2 for temperature curvature.
# the adapter also includes the tuning law's d²phi/dT² term.
# (todo) 5% should be a hyperparameter in a dedicated file same with 25%
# (todo) honestly seems like this function belong to another file like selfheat.py?
# (todo) why are we not using backscatter for this again?
# (todo) seems like backscatter being off could be a major update, since it could lead to better accuracy. would have to change the simulation a lot.
    # - enabling backscatter models an additional physical effect.
    # - setting its strength too high or too low can make predictions less accurate.
    # - leaving it off can also make predictions less accurate if the real device has appreciable backscatter.
# x
@njit(cache=True, error_model="numpy", _nrt=False)
def bound_ring_absorbed_power_and_phase_curvature(
    k: int,  # k = k (1) - physical ring index - type int scalar
    pin: Float64Vector,  # pin = P_in,k (W) - power delivered to this ring per laser line - type float64 ndarray, shape (L,)
    p1: float,  # p1 = p_1 (1) - probability of data level one - type float scalar
    a0: Float64Matrix,  # a0 = a_0 (1) - retained round-trip field amplitude at data level zero - type float64 ndarray, shape (N, L)
    a1: Float64Matrix,  # a1 = a_1 (1) - retained round-trip field amplitude at data level one - type float64 ndarray, shape (N, L)
    eta0: Float64Matrix,  # eta0 = eta_0 (1) - absorbed share of intrinsic loss at data level zero - type float64 ndarray, shape (N, L)
    eta1: Float64Matrix,  # eta1 = eta_1 (1) - absorbed share of intrinsic loss at data level one - type float64 ndarray, shape (N, L)
    k2_th: Float64Matrix,  # k2_th = K_th^2 (1) - through-coupler power fraction - type float64 ndarray, shape (N, L)
    k2_dr: Float64Matrix,  # k2_dr = K_dr^2 (1) - drop-coupler power fraction - type float64 ndarray, shape (N, L)
    two_level: bool,  # two_level = nrz_enabled (1) - evaluate and average both binary data levels - type bool scalar
) -> tuple[
    float,  # absorbed_power_bound_w = B_P (W) - upper bound on absorbed power - type float scalar
    float,  # return[1] = B_phiphi (W/rad^2) - upper bound on |d^2P_abs/dphi^2| - type float scalar
]:
    line_count = pin.shape[0]
    one_probability = p1 if two_level else 0.0
    absorbed_power_bound_w = 0.0
    absorbed_power_phase_curvature_bound = 0.0
    for data_level in range(2):
        level_weight = (1.0 - one_probability) if data_level == 0 else one_probability
        if level_weight == 0.0:
            continue
        for line_index in range(line_count):
            round_trip_field_transmission = a0[k, line_index] if data_level == 0 else a1[k, line_index]
            absorbed_share_of_loss = eta0[k, line_index] if data_level == 0 else eta1[k, line_index]
            through_self_coupling = math.sqrt(1.0 - k2_th[k, line_index])
            drop_self_coupling = math.sqrt(1.0 - k2_dr[k, line_index])
            loop_field_retention = through_self_coupling * drop_self_coupling * round_trip_field_transmission
            loop_retention_gap = 1.0 - loop_field_retention
            loss_numerator = (
                k2_th[k, line_index]
                * (1.0 - round_trip_field_transmission)
                * (1.0 + drop_self_coupling * drop_self_coupling * round_trip_field_transmission)
            )
            absorbed_power_bound_w += (
                level_weight
                * pin[line_index]
                * absorbed_share_of_loss
                * min(1.05 * loss_numerator / (loop_retention_gap * loop_retention_gap), 1.0)
            )
            absorbed_power_phase_curvature_bound += (
                level_weight
                * pin[line_index]
                * absorbed_share_of_loss
                * 2.0
                * loop_field_retention
                * loss_numerator
                / (loop_retention_gap * loop_retention_gap * loop_retention_gap * loop_retention_gap)
            )
    return (
        absorbed_power_bound_w,
        1.25 * absorbed_power_phase_curvature_bound,
    )


# bound the absolute absorbed-power phase slope at fixed input power, in W/rad.
# multiply by |dphi/dT| for a W/K bound.
# the thermal model uses this to check that optical heating through other rings remains weak.
# sum lines and average binary levels with p1.
# use the Airy loss C/den(phi) defined in bound_ring_absorbed_power_and_phase_curvature().
# with A = 1+r**2,
# |d(loss)/dphi| = 2*r*C*|sin(phi)|/den(phi)**2.
# its maximum occurs at cos(phi) = (sqrt(A**2+32*r**2)-A)/(4*r), the root of 2*r*cos(phi)**2+A*cos(phi)-4*r = 0.
# in the Lorentzian limit this is (3*sqrt(3)/8)*peak/HWHM, where HWHM is half-width at half-maximum in rad.
# splitting into two half-weight Lorentzians cannot increase the maximum summed slope.
# allow a 10 % margin for exact-ring corrections.
# test_optics.py verifies this for |rb| <= MAX_ROUND_TRIP_BACKSCATTER_FOR_VERIFIED_POWER_BOUNDS at every order.
# x
@njit(cache=True, error_model="numpy", _nrt=False)
def bound_ring_absorbed_power_phase_derivative(
    k: int,  # k = k (1) - physical ring index - type int scalar
    pin: Float64Vector,  # pin = P_in,k (W) - power delivered to this ring per laser line - type float64 ndarray, shape (L,)
    p1: float,  # p1 = p_1 (1) - probability of data level one - type float scalar
    a0: Float64Matrix,  # a0 = a_0 (1) - retained round-trip field amplitude at data level zero - type float64 ndarray, shape (N, L)
    a1: Float64Matrix,  # a1 = a_1 (1) - retained round-trip field amplitude at data level one - type float64 ndarray, shape (N, L)
    eta0: Float64Matrix,  # eta0 = eta_0 (1) - absorbed share of intrinsic loss at data level zero - type float64 ndarray, shape (N, L)
    eta1: Float64Matrix,  # eta1 = eta_1 (1) - absorbed share of intrinsic loss at data level one - type float64 ndarray, shape (N, L)
    k2_th: Float64Matrix,  # k2_th = K_th^2 (1) - through-coupler power fraction - type float64 ndarray, shape (N, L)
    k2_dr: Float64Matrix,  # k2_dr = K_dr^2 (1) - drop-coupler power fraction - type float64 ndarray, shape (N, L)
    two_level: bool,  # two_level = nrz_enabled (1) - evaluate and average both binary data levels - type bool scalar
) -> (
    float  # return = B_phi (W/rad) - upper bound on |dP_abs/dphi| - type float scalar
):
    line_count = pin.shape[0]
    one_probability = p1 if two_level else 0.0
    absorbed_power_phase_derivative_bound = 0.0
    for data_level in range(2):
        level_weight = (1.0 - one_probability) if data_level == 0 else one_probability
        if level_weight == 0.0:
            continue
        for line_index in range(line_count):
            round_trip_field_transmission = a0[k, line_index] if data_level == 0 else a1[k, line_index]
            absorbed_share_of_loss = eta0[k, line_index] if data_level == 0 else eta1[k, line_index]
            through_self_coupling = math.sqrt(1.0 - k2_th[k, line_index])
            drop_self_coupling = math.sqrt(1.0 - k2_dr[k, line_index])
            loop_field_retention = through_self_coupling * drop_self_coupling * round_trip_field_transmission
            if loop_field_retention <= 0.0:
                continue
            loss_numerator = (
                k2_th[k, line_index]
                * (1.0 - round_trip_field_transmission)
                * (1.0 + drop_self_coupling * drop_self_coupling * round_trip_field_transmission)
            )
            denominator_constant = 1.0 + loop_field_retention * loop_field_retention
            peak_slope_cos = (
                math.sqrt(
                    denominator_constant * denominator_constant
                    + 32.0 * loop_field_retention * loop_field_retention
                )
                - denominator_constant
            ) / (4.0 * loop_field_retention)
            peak_slope_sin = math.sqrt(max(1.0 - peak_slope_cos * peak_slope_cos, 0.0))
            peak_slope_denominator = denominator_constant - 2.0 * loop_field_retention * peak_slope_cos
            absorbed_power_phase_derivative_bound += (
                level_weight
                * pin[line_index]
                * absorbed_share_of_loss
                * 2.0
                * loop_field_retention
                * loss_numerator
                * peak_slope_sin
                / (peak_slope_denominator * peak_slope_denominator)
            )
    return 1.10 * absorbed_power_phase_derivative_bound


# after every ring has been solved - bus output, per-level outputs, reflection return and the single-pass bound.
# write each ring's conditional zero-level and one-level powers at the bus output.
# average the data levels of every other ring.
# write total bus power, returned reflection, and the forward-power error bound.
# LASER_LINE_OUTPUT_ROW_BUS_INPUT_RETURNED_REFLECTION_POWER is the incoherent sum of single reflections returned to the bus input.
# reciprocity gives equal forward and backward through transmission - S_01 = S_10.
# returned light crosses each upstream bus segment and ring twice.
# ring k returns P_in*E[R_k]*prod_{j<=k}(t2_j^2)*prod_{j<k}(E[T_j^2]).
# the light crosses ring j twice within one data state, so the mean of the SQUARE of T_j enters.
# average squared through transmission over the same binary state on both crossings.
# E[(P*T)^2]=p1*THRU1^2+(1-p1)*THRU0^2.
# this equals THRU0^2+(THRU-THRU0)*(THRU1+THRU0).
# the saved mean and level powers therefore remove the need to pass p1.
# exact for independent data on different rings.
# with CW optics it reduces to refl_k * pin_k / P_in.
# with NRZ the bound uses the RMS reflection amplitude sqrt(E[R]) of each ring.
# to leading order, the error grows with products of reflection amplitudes from ring pairs.
# independent data gives E[|r_i|*|r_j|]<=sqrt(E[|r_i|^2])*sqrt(E[|r_j|^2]).
# this bounds the data-averaged error to leading order in eps.
# it does not bound every possible data pattern.
# x
@njit(cache=True, _nrt=False)
def write_bus_output_and_returned_reflection_powers(
    p_in_w: Float64Vector,  # p_in_w = P_in (W) - bus input power per laser line - type float64 ndarray, shape (L,)
    bus_order: Int64Vector,  # bus_order = b_j (1) - physical ring index at bus position j - type int64 ndarray, shape (N,)
    bus_t2: Float64Vector,  # bus_t2 = B_k (1) - power transmission of the bus segment before each ring - type float64 ndarray, shape (N,)
    bus_t2_out: float,  # bus_t2_out = B_out (1) - power transmission of the final bus segment - type float scalar
    out_rl: RingLineOutputs,  # out_rl = Y_ring,line (mixed units) - RING_LINE_OUTPUT_* powers (W), energies (J), and power slope (W/rad) - type float64 ndarray, shape (16, N, L)
    out_l: LaserLineOutputs,  # out_l = Y_line (mixed units) - LASER_LINE_OUTPUT_* powers (W) and relative error bound (1) - type float64 ndarray, shape (3, L)
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    write_bus_output_powers(p_in_w, bus_order, bus_t2, bus_t2_out, out_rl, out_l)
    write_returned_reflection_power_and_forward_error_bound(p_in_w, bus_order, bus_t2, out_rl, out_l)


# write conditional ring-level powers and total power at the bus output.
# x
@njit(cache=True, _nrt=False)
def write_bus_output_powers(
    p_in_w: Float64Vector,  # p_in_w = P_in (W) - bus input power per laser line - type float64 ndarray, shape (L,)
    bus_order: Int64Vector,  # bus_order = b_j (1) - physical ring index at bus position j - type int64 ndarray, shape (N,)
    bus_t2: Float64Vector,  # bus_t2 = B_k (1) - power transmission of the bus segment before each ring - type float64 ndarray, shape (N,)
    bus_t2_out: float,  # bus_t2_out = B_out (1) - power transmission of the final bus segment - type float scalar
    out_rl: RingLineOutputs,  # out_rl = Y_ring,line (mixed units) - RING_LINE_OUTPUT_* powers (W), energies (J), and power slope (W/rad) - type float64 ndarray, shape (16, N, L)
    out_l: LaserLineOutputs,  # out_l = Y_line (mixed units) - LASER_LINE_OUTPUT_* powers (W) and relative error bound (1) - type float64 ndarray, shape (3, L)
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    ring_count = bus_order.shape[0]
    line_count = p_in_w.shape[0]
    for line_index in range(line_count):
        # suffix product - averaged power transfer from just after ring at position pos to the bus output
        downstream_power_transmission = bus_t2_out
        for bus_position in range(ring_count - 1, -1, -1): # traversing backwards, over all laser lines
            ring_index = bus_order[bus_position]
            ring_input_power_w = out_rl[RING_LINE_OUTPUT_ROW_INPUT_POWER, ring_index, line_index]
            if ring_input_power_w > 0.0:
                # 0
                out_rl[RING_LINE_OUTPUT_ROW_LEVEL_ONE_BUS_OUTPUT_POWER, ring_index, line_index] = (
                    out_rl[RING_LINE_OUTPUT_ROW_LEVEL_ONE_THROUGH_POWER, ring_index, line_index]
                    * downstream_power_transmission
                )
                # 1
                out_rl[RING_LINE_OUTPUT_ROW_LEVEL_ZERO_BUS_OUTPUT_POWER, ring_index, line_index] = (
                    out_rl[RING_LINE_OUTPUT_ROW_LEVEL_ZERO_THROUGH_POWER, ring_index, line_index]
                    * downstream_power_transmission
                )
                mean_through_fraction = (
                    out_rl[RING_LINE_OUTPUT_ROW_THROUGH_POWER, ring_index, line_index] / ring_input_power_w
                )
            else:
                out_rl[RING_LINE_OUTPUT_ROW_LEVEL_ONE_BUS_OUTPUT_POWER, ring_index, line_index] = 0.0
                out_rl[RING_LINE_OUTPUT_ROW_LEVEL_ZERO_BUS_OUTPUT_POWER, ring_index, line_index] = 0.0
                mean_through_fraction = 0.0
            downstream_power_transmission *= mean_through_fraction * bus_t2[ring_index] # through fraction is alr reversed
            # traverse fwd
            # eg.
            # the waveguide segment before C.
            # C itself.
            # everything already accounted for after C.
        last_ring_index = bus_order[ring_count - 1]
        out_l[LASER_LINE_OUTPUT_ROW_BUS_OUTPUT_POWER, line_index] = (
            out_rl[RING_LINE_OUTPUT_ROW_THROUGH_POWER, last_ring_index, line_index] * bus_t2_out
        )


# write returned reflection and the forward-power error bound independently of bus-output powers.
# x
@njit(cache=True, _nrt=False)
def write_returned_reflection_power_and_forward_error_bound(
    p_in_w: Float64Vector,  # p_in_w = P_in (W) - bus input power per laser line - type float64 ndarray, shape (L,)
    bus_order: Int64Vector,  # bus_order = b_j (1) - physical ring index at bus position j - type int64 ndarray, shape (N,)
    bus_t2: Float64Vector,  # bus_t2 = B_k (1) - power transmission of the bus segment before each ring - type float64 ndarray, shape (N,)
    out_rl: RingLineOutputs,  # out_rl = Y_ring,line (mixed units) - RING_LINE_OUTPUT_* powers (W), energies (J), and power slope (W/rad) - type float64 ndarray, shape (16, N, L)
    out_l: LaserLineOutputs,  # out_l = Y_line (mixed units) - LASER_LINE_OUTPUT_* powers (W) and relative error bound (1) - type float64 ndarray, shape (3, L)
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    ring_count = bus_order.shape[0]
    line_count = p_in_w.shape[0]
    for line_index in range(line_count):
        # reflection returned to the input (E[T^2] for the double crossing), and the double-reflection bound
        returned_power_w = 0.0
        reflection_amplitude_sum = 0.0  # sum of |r_k|
        reflection_power_sum = 0.0  # sum of |r_k|^2
        round_trip_bus_power_factor = (
            1.0  # mean round-trip power factor, bus input -> ring k's input -> bus input
        )
        for bus_position in range(ring_count):
            ring_index = bus_order[bus_position]
            ring_input_power_w = out_rl[RING_LINE_OUTPUT_ROW_INPUT_POWER, ring_index, line_index]
            round_trip_bus_power_factor *= bus_t2[ring_index] * bus_t2[ring_index]
            if ring_input_power_w > 0.0 and p_in_w[line_index] > 0.0:
                reflected_power_w = out_rl[RING_LINE_OUTPUT_ROW_REFLECTED_POWER, ring_index, line_index]
                if reflected_power_w == 0.0:
                    # rr = 0 / pk = 0 - no division and no square root (the same sums, bit for bit)
                    returned_power_w += 0.0 * p_in_w[line_index] * round_trip_bus_power_factor
                else:
                    reflection_fraction = reflected_power_w / ring_input_power_w
                    returned_power_w += reflection_fraction * p_in_w[line_index] * round_trip_bus_power_factor # original laser power × this ring’s reflection fraction × attenuation on the journey there and back
                    reflection_amplitude_sum += math.sqrt(reflection_fraction)
                    reflection_power_sum += reflection_fraction
                mean_through_power_w = out_rl[RING_LINE_OUTPUT_ROW_THROUGH_POWER, ring_index, line_index]
                level0_through_power_w = out_rl[
                    RING_LINE_OUTPUT_ROW_LEVEL_ZERO_THROUGH_POWER, ring_index, line_index
                ]
                level1_through_power_w = out_rl[
                    RING_LINE_OUTPUT_ROW_LEVEL_ONE_THROUGH_POWER, ring_index, line_index
                ]
                round_trip_bus_power_factor *= (
                    level0_through_power_w * level0_through_power_w
                    + (mean_through_power_w - level0_through_power_w)
                    * (level1_through_power_w + level0_through_power_w)
                ) / (ring_input_power_w * ring_input_power_w)
            else:
                round_trip_bus_power_factor = 0.0
        out_l[LASER_LINE_OUTPUT_ROW_BUS_INPUT_RETURNED_REFLECTION_POWER, line_index] = returned_power_w
        reflection_pair_sum = 0.5 * (
            reflection_amplitude_sum * reflection_amplitude_sum - reflection_power_sum
        )  # sum over all pairs i < j of |r_i| |r_j| (bus_propagate comments)
        out_l[LASER_LINE_OUTPUT_ROW_FORWARD_POWER_RELATIVE_ERROR_BOUND, line_index] = (
            1.0 / ((1.0 - reflection_pair_sum) * (1.0 - reflection_pair_sum)) - 1.0
            if reflection_pair_sum < 1.0
            else np.inf
        )


# pass every laser line through the rings in bus_order, writing results in place.
# apply each preceding bus segment's power loss, solve that ring, and pass its through power on.
# finalize the bus after the last ring.
# this is the compiled full-bus entry point.
# the implicit thermal solver instead advances ring by ring as it solves their temperatures.
# outputs come from make_bus_outputs() - work, complex_workspace, and real_workspace come from make_workspace().
# x
@njit(cache=True, error_model="numpy", _nrt=False)
def propagate_bus_optical_power_into_output_buffers(
    p_in_w: Float64Vector,  # p_in_w = P_in (W) - bus input power per laser line - type float64 ndarray, shape (L,)
    bus_order: Int64Vector,  # bus_order = b_j (1) - physical ring index at bus position j - type int64 ndarray, shape (N,)
    bus_t2: Float64Vector,  # bus_t2 = B_k (1) - power transmission of the bus segment before each ring - type float64 ndarray, shape (N,)
    bus_t2_out: float,  # bus_t2_out = B_out (1) - power transmission of the final bus segment - type float scalar
    p1: Float64Vector,  # p1 = p_1 (1) - probability of data level one - type float64 ndarray, shape (N,)
    phi0: Float64Matrix,  # phi0 = phi_0 (rad) - round-trip phase at data level zero - type float64 ndarray, shape (N, L)
    phi1: Float64Matrix,  # phi1 = phi_1 (rad) - round-trip phase at data level one - type float64 ndarray, shape (N, L)
    a0: Float64Matrix,  # a0 = a_0 (1) - retained round-trip field amplitude at data level zero - type float64 ndarray, shape (N, L)
    a1: Float64Matrix,  # a1 = a_1 (1) - retained round-trip field amplitude at data level one - type float64 ndarray, shape (N, L)
    eta0: Float64Matrix,  # eta0 = eta_0 (1) - absorbed share of intrinsic loss at data level zero - type float64 ndarray, shape (N, L)
    eta1: Float64Matrix,  # eta1 = eta_1 (1) - absorbed share of intrinsic loss at data level one - type float64 ndarray, shape (N, L)
    psi: Float64Matrix,  # psi = psi (rad) - additional coupler phase - type float64 ndarray, shape (N, L)
    k2_th: Float64Matrix,  # k2_th = K_th^2 (1) - through-coupler power fraction - type float64 ndarray, shape (N, L)
    k2_dr: Float64Matrix,  # k2_dr = K_dr^2 (1) - drop-coupler power fraction - type float64 ndarray, shape (N, L)
    t_rt: Float64Matrix,  # t_rt = T_rt (s) - round-trip group delay - type float64 ndarray, shape (N, L)
    rb: Complex128Matrix,  # rb = r_b (1) - complex round-trip backscatter coupling by resonance order - type complex128 ndarray, shape (N, backscatter_order_count)
    dphi_dlamres: Float64Matrix,  # dphi_dlamres = dphi/dlambda_r (rad/m) - phase slope for a resonance shift - type float64 ndarray, shape (N, L)
    two_level: bool,  # two_level = nrz_enabled (1) - evaluate and average both binary data levels - type bool scalar
    use_tcmt: bool,  # use_tcmt = reserved flag (1) - must be false to preserve the compiled call layout - type bool scalar
    m_orders: int,  # m_orders = reserved count (1) - retained for existing call layouts - type int scalar
    with_slope: bool,  # with_slope = slope_enabled (1) - also calculate absorbed-power slopes - type bool scalar
    out_rl: RingLineOutputs,  # out_rl = Y_ring,line (mixed units) - RING_LINE_OUTPUT_* powers (W), energies (J), and power slope (W/rad) - type float64 ndarray, shape (16, N, L)
    out_r: RingOutputs,  # out_r = Y_ring (mixed units) - RING_OUTPUT_* powers (W) and resonance-wavelength slope (W/m) - type float64 ndarray, shape (6, N)
    out_l: LaserLineOutputs,  # out_l = Y_line (mixed units) - LASER_LINE_OUTPUT_* powers (W) and relative error bound (1) - type float64 ndarray, shape (3, L)
    work: Float64Vector,  # work = P_forward (W) - reusable per-line forward-power buffer - type float64 ndarray, shape (L,)
    complex_workspace: ComplexWorkspace,  # complex_workspace = C_work (mixed units) - COMPLEX_WORKSPACE_* factors, fields, and field phase slopes - type complex128 ndarray, capacity at least (8, L)
    real_workspace: RealWorkspace,  # real_workspace = F_work (mixed units) - REAL_WORKSPACE_* factors, powers, slopes, and RING_RESPONSE_* response blocks - type float64 ndarray, capacity at least (27, L)
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    line_count = p_in_w.shape[0]
    for line_index in range(line_count):
        work[line_index] = p_in_w[line_index]
    for bus_position in range(bus_order.shape[0]):
        ring_index = bus_order[bus_position]
        for line_index in range(line_count):
            work[line_index] *= bus_t2[ring_index]
        evaluate_ring_optical_outputs(
            ring_index,
            work,
            p1[ring_index],
            phi0,
            phi1,
            a0,
            a1,
            eta0,
            eta1,
            psi,
            k2_th,
            k2_dr,
            t_rt,
            rb,
            dphi_dlamres,
            two_level,
            use_tcmt,
            m_orders,
            with_slope,
            out_rl,
            out_r,
            work,
            complex_workspace,
            real_workspace,
        )
    write_bus_output_and_returned_reflection_powers(p_in_w, bus_order, bus_t2, bus_t2_out, out_rl, out_l)


# self-heating adapter - packed layout
# the self-heating solve (mrm.plant.selfheat.implicit_step_b) walks the rings in bus order.
# for ring k it calls
# pabs_fn(t, k, packed_f, packed_i) -> (P_abs [W], dP_abs/dT [W/K])
# at every trial temperature t [K] of its Newton solve, and commit_fn(t, k, packed_f, packed_i) once ring k is solved.
# `absorbed_power_at_trial_temperature` and `commit_optics_at_solved_temperature` are these two functions for the exact ring.
# temperature enters only as a rigid shift of each ring's resonance comb.
# the law is the thermo-optic relation R1 of mrm/params/relations.py to second order (the same law as selfheat.pabs_lorentzian) -
# lambda_res,k(T) - lambda_res,k(T_ref) = s0_k [dT + beta_k dT^2/2],     dT = T - T_ref,k
# so the phase at line l is phi_kl(T) = phi_kl(T_ref) + (dphi/dlambda_res)_kl * that shift (`phase_sensitivity_to_resonance_shift`).
# K^2, a_rt and eta are held at the values that the plant writes for the substep.
# evaluate a_rt and eta at the previous ring temperature.
# their roughly 1e-3/K temperature dependence is delayed by one substep.
# differentiate absorbed power while holding those values fixed.
# the root solver therefore uses the exact derivative of this frozen-parameter step.
# layout.
# packed_i holds the sizes, flags and block offsets SELF_HEATING_HEADER_* below, then bus_order (N,) and the bus position of every ring (N,).
# packed_f holds the blocks.
# store ring-line fields in one contiguous (SELF_HEATING_RING_LINE_ROW_COUNT, N, L) block.
# store ring-only fields in one contiguous (SELF_HEATING_RING_ROW_COUNT, N) block.
# each evaluation needs only a few shared views.
# a Numba reshape costs about 27 ns and a slice about 1 ns in the measured configuration.
# `pack_self_heating_optical_state` builds the arrays of one world.
# for a batch, stack them with np.stack - every world of a bank has the same layout, and the length is even, so the complex views stay 16-byte aligned.


# read views of bus input, current ring input, thermal/optical rows, backscatter, and workspaces.
# the views share packed_f storage.
# N is the ring count, L the line count, and M_b the number of backscatter orders.
# do not allocate copies inside the compiled thermal solve.
# x
@njit(cache=True, inline="always")
def unpack_self_heating_optical_inputs(
    packed_f: PackedOpticalValues,  # packed_f = optics_state (mixed units) - SELF_HEATING_HEADER_* blocks with SELF_HEATING_RING_ROW_*, SELF_HEATING_RING_LINE_ROW_*, and cache field units - type float64 ndarray, shape (packed_value_count,)
    packed_i: PackedOpticalIndices,  # packed_i = offsets (1) - SELF_HEATING_HEADER_* counts, flags, and packed-array positions - type int64 ndarray, shape (packed_index_count,)
) -> tuple[
    Float64Vector,  # bus_input_power_w = P_in (W) - bus input power by laser line - type float64 ndarray, shape (L,)
    Float64Vector,  # current_ring_input_power_w = P_current (W) - forward-pass input to the current ring - type float64 ndarray, shape (L,)
    Float64Matrix,  # ring_rows = R_ring (mixed units) - SELF_HEATING_RING_ROW_* parameter fields - type float64 ndarray, shape (5, N)
    Float64Tensor,  # line_rows = R_line (mixed units) - SELF_HEATING_RING_LINE_ROW_* parameter and phase fields - type float64 ndarray, shape (13, N, L)
    Complex128Matrix,  # rb = r_b (1) - complex round-trip backscatter by ring and order - type complex128 ndarray, shape (N, backscatter_order_count)
    ComplexWorkspace,  # complex_workspace = C_work (mixed units) - COMPLEX_WORKSPACE_* field layout - type complex128 ndarray, capacity at least (8, L)
    RealWorkspace,  # real_workspace = F_work (mixed units) - REAL_WORKSPACE_* field layout - type float64 ndarray, capacity at least (27, L)
]:
    if packed_i[SELF_HEATING_HEADER_INDEX_FAR_LINE_HOLD_ENABLED_FLAG] != 0:
        raise ValueError("far-line hold is retired - use the general ring calculation")
    ring_count = packed_i[SELF_HEATING_HEADER_INDEX_RING_COUNT]
    line_count = packed_i[SELF_HEATING_HEADER_INDEX_LASER_LINE_COUNT]
    backscatter_order_count = packed_i[SELF_HEATING_HEADER_INDEX_BACKSCATTER_ORDER_COUNT]
    block_offset = packed_i[SELF_HEATING_HEADER_INDEX_BUS_INPUT_POWER_OFFSET]
    bus_input_power_w = packed_f[block_offset : block_offset + line_count]
    block_offset = packed_i[SELF_HEATING_HEADER_INDEX_CURRENT_RING_INPUT_POWER_OFFSET]
    current_ring_input_power_w = packed_f[block_offset : block_offset + line_count]
    block_offset = packed_i[SELF_HEATING_HEADER_INDEX_RING_PARAMETER_BLOCK_OFFSET]
    ring_rows = packed_f[block_offset : block_offset + SELF_HEATING_RING_ROW_COUNT * ring_count].reshape(
        (SELF_HEATING_RING_ROW_COUNT, ring_count)
    )
    block_offset = packed_i[SELF_HEATING_HEADER_INDEX_RING_LINE_PARAMETER_BLOCK_OFFSET]
    line_rows = packed_f[
        block_offset : block_offset + SELF_HEATING_RING_LINE_ROW_COUNT * ring_count * line_count
    ].reshape((SELF_HEATING_RING_LINE_ROW_COUNT, ring_count, line_count))
    block_offset = packed_i[SELF_HEATING_HEADER_INDEX_BACKSCATTER_COEFFICIENTS_OFFSET]
    rb = (
        packed_f[block_offset : block_offset + 2 * ring_count * backscatter_order_count]
        .view(np.complex128)
        .reshape((ring_count, backscatter_order_count))
    )
    block_offset = packed_i[SELF_HEATING_HEADER_INDEX_COMPLEX_WORKSPACE_OFFSET]
    complex_workspace = (
        packed_f[block_offset : block_offset + 2 * COMPLEX_WORKSPACE_ROW_COUNT * line_count]
        .view(np.complex128)
        .reshape((COMPLEX_WORKSPACE_ROW_COUNT, line_count))
    )
    block_offset = packed_i[SELF_HEATING_HEADER_INDEX_REAL_WORKSPACE_OFFSET]
    real_workspace = packed_f[block_offset : block_offset + REAL_WORKSPACE_ROW_COUNT * line_count].reshape(
        (REAL_WORKSPACE_ROW_COUNT, line_count)
    )
    return (
        bus_input_power_w,
        current_ring_input_power_w,
        ring_rows,
        line_rows,
        rb,
        complex_workspace,
        real_workspace,
    )


# read shared views of packed outputs - per ring/line, per ring, and per laser line.
# x
@njit(cache=True, inline="always")
def unpack_self_heating_optical_outputs(
    packed_f: PackedOpticalValues,  # packed_f = optics_state (mixed units) - SELF_HEATING_HEADER_* blocks with SELF_HEATING_RING_ROW_*, SELF_HEATING_RING_LINE_ROW_*, and cache field units - type float64 ndarray, shape (packed_value_count,)
    packed_i: PackedOpticalIndices,  # packed_i = offsets (1) - SELF_HEATING_HEADER_* counts, flags, and packed-array positions - type int64 ndarray, shape (packed_index_count,)
) -> tuple[
    RingLineOutputs,  # ring_line_outputs = Y_ring,line (mixed units) - RING_LINE_OUTPUT_* powers (W), energies (J), and slope (W/rad) - type float64 ndarray, shape (16, N, L)
    RingOutputs,  # ring_outputs = Y_ring (mixed units) - RING_OUTPUT_* powers (W) and slope (W/m) - type float64 ndarray, shape (6, N)
    LaserLineOutputs,  # line_outputs = Y_line (mixed units) - LASER_LINE_OUTPUT_* powers (W) and relative error bound (1) - type float64 ndarray, shape (3, L)
]:
    ring_count = packed_i[SELF_HEATING_HEADER_INDEX_RING_COUNT]
    line_count = packed_i[SELF_HEATING_HEADER_INDEX_LASER_LINE_COUNT]
    block_offset = packed_i[SELF_HEATING_HEADER_INDEX_RING_LINE_OUTPUTS_OFFSET]
    ring_line_outputs = packed_f[
        block_offset : block_offset + RING_LINE_OUTPUT_ROW_COUNT * ring_count * line_count
    ].reshape((RING_LINE_OUTPUT_ROW_COUNT, ring_count, line_count))
    block_offset = packed_i[SELF_HEATING_HEADER_INDEX_RING_OUTPUTS_OFFSET]
    ring_outputs = packed_f[block_offset : block_offset + RING_OUTPUT_ROW_COUNT * ring_count].reshape(
        (RING_OUTPUT_ROW_COUNT, ring_count)
    )
    block_offset = packed_i[SELF_HEATING_HEADER_INDEX_LASER_LINE_OUTPUTS_OFFSET]
    line_outputs = packed_f[block_offset : block_offset + LASER_LINE_OUTPUT_ROW_COUNT * line_count].reshape(
        (LASER_LINE_OUTPUT_ROW_COUNT, line_count)
    )
    return (
        ring_line_outputs,
        ring_outputs,
        line_outputs,
    )


assert len(SELF_HEATING_VIEW_FIELD_NAMES) == SELF_HEATING_VIEW_INDEX_BUS_POSITIONS_BY_RING + 1  # one name per view


# expose named positions in the packed optical arrays without copying.
# the plant updates inputs through these views before a thermal substep, then reads outputs after the solve.
# select entries with SELF_HEATING_VIEW_INDEX_* constants - SELF_HEATING_VIEW_FIELD_NAMES gives their names.
# shapes, units, and fixed tuple positions are documented by SELF_HEATING_VIEW_INDEX_*, SELF_HEATING_RING_ROW_*, and SELF_HEATING_RING_LINE_ROW_* above.
# x
@njit(cache=True, inline="always")
def unpack_self_heating_optical_state(
    packed_f: PackedOpticalValues,  # packed_f = optics_state (mixed units) - SELF_HEATING_HEADER_* blocks with SELF_HEATING_RING_ROW_*, SELF_HEATING_RING_LINE_ROW_*, and cache field units - type float64 ndarray, shape (packed_value_count,)
    packed_i: PackedOpticalIndices,  # packed_i = offsets (1) - SELF_HEATING_HEADER_* counts, flags, and packed-array positions - type int64 ndarray, shape (packed_index_count,)
) -> tuple[
    Float64Vector,  # return[0] = P_bus_input (W) - input power per laser line - type float64 ndarray, shape (L,)
    Float64Vector,  # return[1] = T_bus (1) - bus segment power transmission per ring - type float64 ndarray, shape (N,)
    Float64Vector,  # return[2] = T_bus_output (1) - final bus segment power transmission - type float64 ndarray, shape (1,)
    Float64Vector,  # return[3] = p_one (1) - ones probability per ring - type float64 ndarray, shape (N,)
    Float64Matrix,  # return[4] = phi_zero,reference (rad) - level-zero reference phases - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[5] = phi_one,reference (rad) - level-one reference phases - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[6] = phi_zero (rad) - level-zero current phases - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[7] = phi_one (rad) - level-one current phases - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[8] = a_zero (1) - level-zero round-trip field transmissions - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[9] = a_one (1) - level-one round-trip field transmissions - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[10] = eta_zero (1) - level-zero absorbed shares of intrinsic loss - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[11] = eta_one (1) - level-one absorbed shares of intrinsic loss - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[12] = psi (rad) - coupler phases - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[13] = K_through^2 (1) - through-coupler power coupling - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[14] = K_drop^2 (1) - drop-coupler power coupling - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[15] = T_round_trip (s) - round-trip group delays - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[16] = dphi/dlambda_res (rad/m) - phase derivatives for resonance shifts - type float64 ndarray, shape (N, L)
    Float64Vector,  # return[17] = T_reference (K) - reference ring temperatures - type float64 ndarray, shape (N,)
    Float64Vector,  # return[18] = s_reference (m/K) - reference thermal tuning slopes - type float64 ndarray, shape (N,)
    Float64Vector,  # return[19] = beta (1/K) - relative thermal tuning curvatures - type float64 ndarray, shape (N,)
    Complex128Matrix,  # return[20] = r_backscatter (1) - complex round-trip backscatter by ring and order - type complex128 ndarray, shape (N, backscatter_order_count)
    Float64Vector,  # return[21] = P_current (W) - current ring input power by line - type float64 ndarray, shape (L,)
    RingLineOutputs,  # return[22] = Y_ring,line (mixed units) - per-ring and per-line outputs - type float64 ndarray, shape (16, N, L)
    RingOutputs,  # return[23] = Y_ring (mixed units) - per-ring outputs - type float64 ndarray, shape (6, N)
    LaserLineOutputs,  # return[24] = Y_line (mixed units) - per-line outputs - type float64 ndarray, shape (3, L)
    ComplexWorkspace,  # return[25] = C_work (mixed units) - complex field workspace - type complex128 ndarray, capacity at least (8, L)
    RealWorkspace,  # return[26] = F_work (mixed units) - real workspace - type float64 ndarray, capacity at least (27, L)
    Int64Vector,  # return[27] = bus_order (1) - ring indices in bus order - type int64 ndarray, shape (N,)
    Int64Vector,  # return[28] = bus_positions (1) - bus positions indexed by ring - type int64 ndarray, shape (N,)
]:
    p_in, current_ring_input_power_w, ring_rows, line_rows, rb, complex_workspace, real_workspace = unpack_self_heating_optical_inputs(
        packed_f, packed_i
    )
    out_rl, out_r, out_l = unpack_self_heating_optical_outputs(packed_f, packed_i)
    ring_count = packed_i[SELF_HEATING_HEADER_INDEX_RING_COUNT]
    block_offset = packed_i[SELF_HEATING_HEADER_INDEX_BUS_OUTPUT_SEGMENT_TRANSMISSION_OFFSET]
    return (
        p_in,
        ring_rows[SELF_HEATING_RING_ROW_BUS_SEGMENT_POWER_TRANSMISSION],
        packed_f[block_offset : block_offset + 1],
        ring_rows[SELF_HEATING_RING_ROW_PROBABILITY_OF_ONE],
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_REFERENCE_ROUND_TRIP_PHASE],
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_REFERENCE_ROUND_TRIP_PHASE],
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_CURRENT_ROUND_TRIP_PHASE],
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_CURRENT_ROUND_TRIP_PHASE],
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_ROUND_TRIP_FIELD_TRANSMISSION],
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_ROUND_TRIP_FIELD_TRANSMISSION],
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_ABSORBED_SHARE_OF_INTRINSIC_LOSS],
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_ABSORBED_SHARE_OF_INTRINSIC_LOSS],
        line_rows[SELF_HEATING_RING_LINE_ROW_COUPLER_PHASE],
        line_rows[SELF_HEATING_RING_LINE_ROW_THROUGH_COUPLER_POWER_COUPLING],
        line_rows[SELF_HEATING_RING_LINE_ROW_DROP_COUPLER_POWER_COUPLING],
        line_rows[SELF_HEATING_RING_LINE_ROW_ROUND_TRIP_GROUP_DELAY],
        line_rows[SELF_HEATING_RING_LINE_ROW_PHASE_RESONANCE_WAVELENGTH_DERIVATIVE],
        ring_rows[SELF_HEATING_RING_ROW_REFERENCE_TEMPERATURE],
        ring_rows[SELF_HEATING_RING_ROW_REFERENCE_THERMAL_TUNING_SLOPE],
        ring_rows[SELF_HEATING_RING_ROW_RELATIVE_THERMAL_TUNING_CURVATURE],
        rb,
        current_ring_input_power_w,
        out_rl,
        out_r,
        out_l,
        complex_workspace,
        real_workspace,
        packed_i[SELF_HEATING_HEADER_FIXED_FIELD_COUNT : SELF_HEATING_HEADER_FIXED_FIELD_COUNT + ring_count],
        packed_i[
            SELF_HEATING_HEADER_FIXED_FIELD_COUNT
            + ring_count : SELF_HEATING_HEADER_FIXED_FIELD_COUNT
            + 2 * ring_count
        ],
    )


# set a ring's optical phase for a trial temperature and return its tuning slope.
# with temperature change dT from the reference, shift resonance by s0*(dT + beta*dT**2/2).
# convert that shift into phase at each fixed laser wavelength.
# if this is the first ring in bus order, also initialize its input power after the first bus segment.
# units - K, m, rad.
# x
@njit(cache=True, _nrt=False)
def _set_trial_temperature_phase(
    trial_temperature_k: float,  # trial_temperature_k = T (K) - trial or accepted ring temperature - type float scalar
    ring_index: int,  # ring_index = k (1) - physical ring index - type int scalar
    bus_position: int,  # bus_position = j (1) - position in the forward bus traversal - type int scalar
    p_in: Float64Vector,  # p_in = P_in (W) - bus input power per laser line - type float64 ndarray, shape (L,)
    pcur: Float64Vector,  # pcur = P_current (W) - input power of the ring currently being solved - type float64 ndarray, shape (L,)
    ring_rows: Float64Matrix,  # ring_rows = R_ring (mixed units) - SELF_HEATING_RING_ROW_* ring parameters - type float64 ndarray, shape (5, N)
    line_rows: Float64Tensor,  # line_rows = R_line (mixed units) - SELF_HEATING_RING_LINE_ROW_* optical parameters and phase work arrays - type float64 ndarray, shape (13, N, L)
) -> (
    float  # return = dlambda_r/dT (m/K) - tuning slope at the trial temperature - type float scalar
):
    temperature_change_k = (
        trial_temperature_k - ring_rows[SELF_HEATING_RING_ROW_REFERENCE_TEMPERATURE, ring_index]
    )
    reference_tuning_slope_m_per_k = ring_rows[
        SELF_HEATING_RING_ROW_REFERENCE_THERMAL_TUNING_SLOPE, ring_index
    ]
    tuning_curvature_per_k = ring_rows[SELF_HEATING_RING_ROW_RELATIVE_THERMAL_TUNING_CURVATURE, ring_index]
    resonance_shift_m = reference_tuning_slope_m_per_k * (
        temperature_change_k + 0.5 * tuning_curvature_per_k * temperature_change_k * temperature_change_k
    )
    for line_index in range(line_rows.shape[2]):
        d_phase_d_resonance_wavelength = line_rows[
            SELF_HEATING_RING_LINE_ROW_PHASE_RESONANCE_WAVELENGTH_DERIVATIVE, ring_index, line_index
        ]
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_CURRENT_ROUND_TRIP_PHASE, ring_index, line_index] = (
            line_rows[
                SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_REFERENCE_ROUND_TRIP_PHASE, ring_index, line_index
            ]
            + d_phase_d_resonance_wavelength * resonance_shift_m
        )
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_CURRENT_ROUND_TRIP_PHASE, ring_index, line_index] = (
            line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_REFERENCE_ROUND_TRIP_PHASE, ring_index, line_index]
            + d_phase_d_resonance_wavelength * resonance_shift_m
        )
    if bus_position == 0:  # the walk (re)starts at the first ring in bus order
        bus_segment_power_transmission = ring_rows[
            SELF_HEATING_RING_ROW_BUS_SEGMENT_POWER_TRANSMISSION, ring_index
        ]
        for line_index in range(pcur.shape[0]):
            pcur[line_index] = p_in[line_index] * bus_segment_power_transmission
    return reference_tuning_slope_m_per_k * (1.0 + tuning_curvature_per_k * temperature_change_k)


# evaluate optical heating at a trial temperature without advancing time.
# include every laser line and both data levels when enabled.
# use input power from the already-solved upstream rings.
# use the same two-direction field calculation for zero and nonzero backscatter.
# the thermal solver accepts or rejects this trial temperature.
# x
@njit(cache=True, _nrt=False)
def absorbed_power_at_trial_temperature(
    t: float,  # t = T (K) - trial ring temperature - type float scalar
    k: int,  # k = k (1) - ring index - type int scalar
    packed_f: PackedOpticalValues,  # packed_f = optics_state (mixed units) - SELF_HEATING_HEADER_* blocks with SELF_HEATING_RING_ROW_* and SELF_HEATING_RING_LINE_ROW_* field units - type float64 ndarray, shape (packed_value_count,)
    packed_i: PackedOpticalIndices,  # packed_i = offsets (1) - integer counts, indices, and block positions - type int64 ndarray, shape (packed_index_count,)
) -> tuple[
    float,  # return[0] = P_abs(T) (W) - total absorbed power - type float scalar
    float,  # return[1] = dP_abs/dT (W/K) - temperature derivative - type float scalar
]:
    p_in, current_ring_input_power_w, ring_rows, line_rows, rb, complex_workspace, real_workspace = unpack_self_heating_optical_inputs(
        packed_f, packed_i
    )
    ring_count = packed_i[SELF_HEATING_HEADER_INDEX_RING_COUNT]
    tuning_slope_m_per_k = _set_trial_temperature_phase(
        t,
        k,
        packed_i[SELF_HEATING_HEADER_FIXED_FIELD_COUNT + ring_count + k],
        p_in,
        current_ring_input_power_w,
        ring_rows,
        line_rows,
    )
    absorbed_power_w, d_absorbed_power_d_resonance_wavelength = evaluate_ring_absorbed_power_and_resonance_wavelength_derivative(
        k,
        current_ring_input_power_w,
        ring_rows[SELF_HEATING_RING_ROW_PROBABILITY_OF_ONE, k],
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_CURRENT_ROUND_TRIP_PHASE],
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_CURRENT_ROUND_TRIP_PHASE],
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_ROUND_TRIP_FIELD_TRANSMISSION],
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_ROUND_TRIP_FIELD_TRANSMISSION],
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_ABSORBED_SHARE_OF_INTRINSIC_LOSS],
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_ABSORBED_SHARE_OF_INTRINSIC_LOSS],
        line_rows[SELF_HEATING_RING_LINE_ROW_COUPLER_PHASE],
        line_rows[SELF_HEATING_RING_LINE_ROW_THROUGH_COUPLER_POWER_COUPLING],
        line_rows[SELF_HEATING_RING_LINE_ROW_DROP_COUPLER_POWER_COUPLING],
        line_rows[SELF_HEATING_RING_LINE_ROW_ROUND_TRIP_GROUP_DELAY],
        rb,
        line_rows[SELF_HEATING_RING_LINE_ROW_PHASE_RESONANCE_WAVELENGTH_DERIVATIVE],
        packed_i[SELF_HEATING_HEADER_INDEX_TWO_DATA_LEVELS_FLAG] != 0,
        packed_i[SELF_HEATING_HEADER_INDEX_RESERVED_MODEL_FLAG] != 0,
        packed_i[SELF_HEATING_HEADER_INDEX_RESERVED_MODE_ORDER_COUNT],
        complex_workspace,
        real_workspace,
    )
    return (
        absorbed_power_w,
        d_absorbed_power_d_resonance_wavelength * tuning_slope_m_per_k,
    )


# save optical outputs after the thermal solver accepts a ring temperature.
# write ring outputs, pass through power to the next ring, and finalize bus outputs after the last ring.
# starting at the first ring resets the forward pass, so repeated outer sweeps need no separate reset.
# x
@njit(cache=True, _nrt=False)
def commit_optics_at_solved_temperature(
    t: float,  # t = T (K) - trial or accepted ring temperature - type float scalar
    k: int,  # k = k (1) - physical ring index - type int scalar
    packed_f: PackedOpticalValues,  # packed_f = optics_state (mixed units) - SELF_HEATING_HEADER_* blocks with SELF_HEATING_RING_ROW_*, SELF_HEATING_RING_LINE_ROW_*, and cache field units - type float64 ndarray, shape (packed_value_count,)
    packed_i: PackedOpticalIndices,  # packed_i = offsets (1) - SELF_HEATING_HEADER_* counts, flags, and packed-array positions - type int64 ndarray, shape (packed_index_count,)
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    p_in, current_ring_input_power_w, ring_rows, line_rows, rb, complex_workspace, real_workspace = unpack_self_heating_optical_inputs(
        packed_f, packed_i
    )
    out_rl, out_r, out_l = unpack_self_heating_optical_outputs(packed_f, packed_i)
    ring_count = packed_i[SELF_HEATING_HEADER_INDEX_RING_COUNT]
    bus_position = packed_i[SELF_HEATING_HEADER_FIXED_FIELD_COUNT + ring_count + k]
    _set_trial_temperature_phase(t, k, bus_position, p_in, current_ring_input_power_w, ring_rows, line_rows)
    evaluate_ring_optical_outputs(
        k,
        current_ring_input_power_w,
        ring_rows[SELF_HEATING_RING_ROW_PROBABILITY_OF_ONE, k],
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_CURRENT_ROUND_TRIP_PHASE],
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_CURRENT_ROUND_TRIP_PHASE],
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_ROUND_TRIP_FIELD_TRANSMISSION],
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_ROUND_TRIP_FIELD_TRANSMISSION],
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_ABSORBED_SHARE_OF_INTRINSIC_LOSS],
        line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_ABSORBED_SHARE_OF_INTRINSIC_LOSS],
        line_rows[SELF_HEATING_RING_LINE_ROW_COUPLER_PHASE],
        line_rows[SELF_HEATING_RING_LINE_ROW_THROUGH_COUPLER_POWER_COUPLING],
        line_rows[SELF_HEATING_RING_LINE_ROW_DROP_COUPLER_POWER_COUPLING],
        line_rows[SELF_HEATING_RING_LINE_ROW_ROUND_TRIP_GROUP_DELAY],
        rb,
        line_rows[SELF_HEATING_RING_LINE_ROW_PHASE_RESONANCE_WAVELENGTH_DERIVATIVE],
        packed_i[SELF_HEATING_HEADER_INDEX_TWO_DATA_LEVELS_FLAG] != 0,
        packed_i[SELF_HEATING_HEADER_INDEX_RESERVED_MODEL_FLAG] != 0,
        packed_i[SELF_HEATING_HEADER_INDEX_RESERVED_MODE_ORDER_COUNT],
        packed_i[SELF_HEATING_HEADER_INDEX_COMMIT_OPTICAL_DERIVATIVES_FLAG] != 0,
        out_rl,
        out_r,
        current_ring_input_power_w,
        complex_workspace,
        real_workspace,
    )
    if bus_position + 1 < ring_count:
        next_segment_transmission = ring_rows[
            SELF_HEATING_RING_ROW_BUS_SEGMENT_POWER_TRANSMISSION,
            packed_i[SELF_HEATING_HEADER_FIXED_FIELD_COUNT + bus_position + 1],
        ]
        for line_index in range(current_ring_input_power_w.shape[0]):
            current_ring_input_power_w[line_index] *= next_segment_transmission
    else:
        output_transmission_offset = packed_i[
            SELF_HEATING_HEADER_INDEX_BUS_OUTPUT_SEGMENT_TRANSMISSION_OFFSET
        ]
        write_bus_output_and_returned_reflection_powers(
            p_in,
            packed_i[
                SELF_HEATING_HEADER_FIXED_FIELD_COUNT : SELF_HEATING_HEADER_FIXED_FIELD_COUNT + ring_count
            ],
            ring_rows[SELF_HEATING_RING_ROW_BUS_SEGMENT_POWER_TRANSMISSION],
            packed_f[output_transmission_offset],
            out_rl,
            out_l,
        )


# fill conservative limits for absorbed power and its temperature derivatives, for each ring.
# p_max is in W, dpdt_max in W/K, and d2p_max in W/K².
# upstream rings can only reduce input power.
# convert phase derivative bounds using the tuning law over |dT| <= dt_max_k.
# backscatter bounds require |rb| <= MAX_ROUND_TRIP_BACKSCATTER_FOR_VERIFIED_POWER_BOUNDS.
# use the forward-pass buffer as scratch between substeps.
# caching step
# x
@njit(cache=True, _nrt=False)
def bound_absorbed_power_temperature_response(
    packed_f: PackedOpticalValues,  # packed_f = optics_state (mixed units) - SELF_HEATING_HEADER_* blocks with SELF_HEATING_RING_ROW_*, SELF_HEATING_RING_LINE_ROW_*, and cache field units - type float64 ndarray, shape (packed_value_count,)
    packed_i: PackedOpticalIndices,  # packed_i = offsets (1) - SELF_HEATING_HEADER_* counts, flags, and packed-array positions - type int64 ndarray, shape (packed_index_count,)
    dt_max_k: float,  # dt_max_k = Delta_T_max (K) - bound on the temperature excursion from the reference - type float scalar
    p_max: Float64Vector,  # p_max = P_max (W) - per-ring absorbed-power upper bounds - type float64 ndarray, shape (N,)
    d2p_max: Float64Vector,  # d2p_max = B_TT (W/K^2) - upper bounds on |d^2P_abs/dT^2| - type float64 ndarray, shape (N,)
    dpdt_max: Float64Vector,  # dpdt_max = B_T (W/K) - upper bounds on |dP_abs/dT| - type float64 ndarray, shape (N,)
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    bound_absorbed_power_temperature_response_from_ring_parameters(packed_f, packed_i, dt_max_k, p_max, d2p_max, dpdt_max)


# calculate thermal-solver bounds through the separate power and slope kernels.
# use the general ring response to calculate each power and derivative bound.
# the public bounds wrapper and sh_bounds alias both call this implementation.
# outputs are p_max in W, d2p_max in W/K², and dpdt_max in W/K.
# use the forward-pass buffer as scratch between substeps.
# x
@njit(cache=True, _nrt=False)
def bound_absorbed_power_temperature_response_from_ring_parameters(
    packed_f: PackedOpticalValues,  # packed_f = optics_state (mixed units) - SELF_HEATING_HEADER_* blocks with SELF_HEATING_RING_ROW_*, SELF_HEATING_RING_LINE_ROW_*, and cache field units - type float64 ndarray, shape (packed_value_count,)
    packed_i: PackedOpticalIndices,  # packed_i = offsets (1) - SELF_HEATING_HEADER_* counts, flags, and packed-array positions - type int64 ndarray, shape (packed_index_count,)
    dt_max_k: float,  # dt_max_k = Delta_T_max (K) - bound on the temperature excursion from the reference - type float scalar
    p_max: Float64Vector,  # p_max = P_max (W) - per-ring absorbed-power upper bounds - type float64 ndarray, shape (N,)
    d2p_max: Float64Vector,  # d2p_max = B_TT (W/K^2) - upper bounds on |d^2P_abs/dT^2| - type float64 ndarray, shape (N,)
    dpdt_max: Float64Vector,  # dpdt_max = B_T (W/K) - upper bounds on |dP_abs/dT| - type float64 ndarray, shape (N,)
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    p_in, current_ring_input_power_w, ring_rows, line_rows, rb, complex_workspace, real_workspace = unpack_self_heating_optical_inputs(
        packed_f, packed_i
    )
    ring_count = packed_i[SELF_HEATING_HEADER_INDEX_RING_COUNT]
    use_two_data_levels = packed_i[SELF_HEATING_HEADER_INDEX_TWO_DATA_LEVELS_FLAG] != 0
    for line_index in range(current_ring_input_power_w.shape[0]):
        current_ring_input_power_w[line_index] = p_in[line_index]
    for bus_position in range(ring_count):
        ring_index = packed_i[SELF_HEATING_HEADER_FIXED_FIELD_COUNT + bus_position]
        bus_segment_transmission = ring_rows[SELF_HEATING_RING_ROW_BUS_SEGMENT_POWER_TRANSMISSION, ring_index]
        for line_index in range(current_ring_input_power_w.shape[0]):
            current_ring_input_power_w[line_index] *= bus_segment_transmission
        absorbed_power_bound_w, absorbed_power_phase_curvature_bound = bound_ring_absorbed_power_and_phase_curvature(
            ring_index,
            current_ring_input_power_w,
            ring_rows[SELF_HEATING_RING_ROW_PROBABILITY_OF_ONE, ring_index],
            line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_ROUND_TRIP_FIELD_TRANSMISSION],
            line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_ROUND_TRIP_FIELD_TRANSMISSION],
            line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_ABSORBED_SHARE_OF_INTRINSIC_LOSS],
            line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_ABSORBED_SHARE_OF_INTRINSIC_LOSS],
            line_rows[SELF_HEATING_RING_LINE_ROW_THROUGH_COUPLER_POWER_COUPLING],
            line_rows[SELF_HEATING_RING_LINE_ROW_DROP_COUPLER_POWER_COUPLING],
            use_two_data_levels,
        )
        absorbed_power_phase_derivative_bound = bound_ring_absorbed_power_phase_derivative(
            ring_index,
            current_ring_input_power_w,
            ring_rows[SELF_HEATING_RING_ROW_PROBABILITY_OF_ONE, ring_index],
            line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_ROUND_TRIP_FIELD_TRANSMISSION],
            line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_ROUND_TRIP_FIELD_TRANSMISSION],
            line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ZERO_ABSORBED_SHARE_OF_INTRINSIC_LOSS],
            line_rows[SELF_HEATING_RING_LINE_ROW_LEVEL_ONE_ABSORBED_SHARE_OF_INTRINSIC_LOSS],
            line_rows[SELF_HEATING_RING_LINE_ROW_THROUGH_COUPLER_POWER_COUPLING],
            line_rows[SELF_HEATING_RING_LINE_ROW_DROP_COUPLER_POWER_COUPLING],
            use_two_data_levels,
        )
        phase_resonance_wavelength_derivative_bound = 0.0
        for line_index in range(current_ring_input_power_w.shape[0]):
            phase_resonance_wavelength_derivative_bound = max(
                phase_resonance_wavelength_derivative_bound,
                abs(
                    line_rows[
                        SELF_HEATING_RING_LINE_ROW_PHASE_RESONANCE_WAVELENGTH_DERIVATIVE,
                        ring_index,
                        line_index,
                    ]
                ),
            )
        reference_tuning_slope_m_per_k = ring_rows[
            SELF_HEATING_RING_ROW_REFERENCE_THERMAL_TUNING_SLOPE, ring_index
        ]
        tuning_curvature_per_k = ring_rows[
            SELF_HEATING_RING_ROW_RELATIVE_THERMAL_TUNING_CURVATURE, ring_index
        ]
        max_tuning_slope_m_per_k = abs(reference_tuning_slope_m_per_k) * (
            1.0 + abs(tuning_curvature_per_k) * dt_max_k
        )
        p_max[ring_index] = absorbed_power_bound_w
        dpdt_max[ring_index] = (
            absorbed_power_phase_derivative_bound
            * phase_resonance_wavelength_derivative_bound
            * max_tuning_slope_m_per_k
        )
        d2p_max[ring_index] = absorbed_power_phase_curvature_bound * (
            phase_resonance_wavelength_derivative_bound * max_tuning_slope_m_per_k
        ) ** 2 + absorbed_power_phase_derivative_bound * phase_resonance_wavelength_derivative_bound * abs(
            reference_tuning_slope_m_per_k * tuning_curvature_per_k
        )




# send the solved ring's through power to the next ring, including bus-segment loss.
# after the last ring, finalize the bus.
# full=False writes the bus powers needed inside a simulation block.
# full=True also writes returned reflection and the forward-power error bound.
# once ring temp is accepted, apply that segment power transmission to the next ring
@njit(cache=True, _nrt=False)
def pass_through_power_to_next_ring(
    k: int,  # k = k (1) - physical ring index - type int scalar
    packed_f: PackedOpticalValues,  # packed_f = optics_state (mixed units) - SELF_HEATING_HEADER_* blocks with SELF_HEATING_RING_ROW_*, SELF_HEATING_RING_LINE_ROW_*, and cache field units - type float64 ndarray, shape (packed_value_count,)
    packed_i: PackedOpticalIndices,  # packed_i = offsets (1) - SELF_HEATING_HEADER_* counts, flags, and packed-array positions - type int64 ndarray, shape (packed_index_count,)
    full: bool = True,  # full = full_output (1) - write every output instead of the per-tick subset - type bool scalar
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    ring_count = packed_i[SELF_HEATING_HEADER_INDEX_RING_COUNT]
    line_count = packed_i[SELF_HEATING_HEADER_INDEX_LASER_LINE_COUNT]
    bus_position = packed_i[SELF_HEATING_HEADER_FIXED_FIELD_COUNT + ring_count + k]
    current_input_offset = packed_i[SELF_HEATING_HEADER_INDEX_CURRENT_RING_INPUT_POWER_OFFSET]
    if bus_position + 1 < ring_count:
        next_segment_transmission = packed_f[
            _as_unsigned_array_index(
                packed_i[SELF_HEATING_HEADER_INDEX_RING_PARAMETER_BLOCK_OFFSET]
                + SELF_HEATING_RING_ROW_BUS_SEGMENT_POWER_TRANSMISSION * ring_count
                + packed_i[SELF_HEATING_HEADER_FIXED_FIELD_COUNT + bus_position + 1]
            )
        ]
        for line_index in range(line_count):
            packed_f[_as_unsigned_array_index(current_input_offset + line_index)] *= next_segment_transmission
    else:
        # build only the views needed to finalize bus outputs
        block_offset = packed_i[SELF_HEATING_HEADER_INDEX_BUS_INPUT_POWER_OFFSET]
        p_in = packed_f[block_offset : block_offset + line_count]
        block_offset = (
            packed_i[SELF_HEATING_HEADER_INDEX_RING_PARAMETER_BLOCK_OFFSET]
            + SELF_HEATING_RING_ROW_BUS_SEGMENT_POWER_TRANSMISSION * ring_count
        )
        bus_segment_transmission = packed_f[block_offset : block_offset + ring_count]
        block_offset = packed_i[SELF_HEATING_HEADER_INDEX_RING_LINE_OUTPUTS_OFFSET]
        out_rl = packed_f[
            block_offset : block_offset + RING_LINE_OUTPUT_ROW_COUNT * ring_count * line_count
        ].reshape((RING_LINE_OUTPUT_ROW_COUNT, ring_count, line_count))
        block_offset = packed_i[SELF_HEATING_HEADER_INDEX_LASER_LINE_OUTPUTS_OFFSET]
        out_l = packed_f[block_offset : block_offset + LASER_LINE_OUTPUT_ROW_COUNT * line_count].reshape(
            (LASER_LINE_OUTPUT_ROW_COUNT, line_count)
        )
        if full:
            write_bus_output_and_returned_reflection_powers(
                p_in,
                packed_i[
                    SELF_HEATING_HEADER_FIXED_FIELD_COUNT : SELF_HEATING_HEADER_FIXED_FIELD_COUNT + ring_count
                ],
                bus_segment_transmission,
                packed_f[packed_i[SELF_HEATING_HEADER_INDEX_BUS_OUTPUT_SEGMENT_TRANSMISSION_OFFSET]],
                out_rl,
                out_l,
            )
        else:
            write_bus_output_powers(
                p_in,
                packed_i[
                    SELF_HEATING_HEADER_FIXED_FIELD_COUNT : SELF_HEATING_HEADER_FIXED_FIELD_COUNT + ring_count
                ],
                bus_segment_transmission,
                packed_f[packed_i[SELF_HEATING_HEADER_INDEX_BUS_OUTPUT_SEGMENT_TRANSMISSION_OFFSET]],
                out_rl,
                out_l,
            )


# check whether the last cached optical evaluation used ring k at temperature t.
# x
@njit(cache=True, _nrt=False)
def cache_matches_last_optical_evaluation(
    k: int,  # k = k (1) - physical ring index - type int scalar
    t: float,  # t = T (K) - trial or accepted ring temperature - type float scalar
    packed_f: PackedOpticalValues,  # packed_f = optics_state (mixed units) - SELF_HEATING_HEADER_* blocks with SELF_HEATING_RING_ROW_*, SELF_HEATING_RING_LINE_ROW_*, and cache field units - type float64 ndarray, shape (packed_value_count,)
    packed_i: PackedOpticalIndices,  # packed_i = offsets (1) - SELF_HEATING_HEADER_* counts, flags, and packed-array positions - type int64 ndarray, shape (packed_index_count,)
) -> (
    bool  # return = same_evaluation (1) - true when ring index and temperature match the cache - type bool scalar
):
    last_evaluation_header_offset = (
        packed_i[SELF_HEATING_HEADER_INDEX_LAST_EVALUATION_CACHE_OFFSET]
        + 2 * packed_i[SELF_HEATING_HEADER_INDEX_LASER_LINE_COUNT] * LAST_EVALUATION_CACHE_FIELD_COUNT
    )
    return (
        packed_f[last_evaluation_header_offset + LAST_EVALUATION_METADATA_FIELD_RING_INDEX] == k
        and packed_f[last_evaluation_header_offset + LAST_EVALUATION_METADATA_FIELD_TEMPERATURE] == t
    )


# invalidate the last optical evaluation so the next commit must evaluate again.
# x
@njit(cache=True, _nrt=False)
def cache_clear_last_optical_evaluation(
    packed_f: PackedOpticalValues,  # packed_f = optics_state (mixed units) - SELF_HEATING_HEADER_* blocks with SELF_HEATING_RING_ROW_*, SELF_HEATING_RING_LINE_ROW_*, and cache field units - type float64 ndarray, shape (packed_value_count,)
    packed_i: PackedOpticalIndices,  # packed_i = offsets (1) - SELF_HEATING_HEADER_* counts, flags, and packed-array positions - type int64 ndarray, shape (packed_index_count,)
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    packed_f[
        packed_i[SELF_HEATING_HEADER_INDEX_LAST_EVALUATION_CACHE_OFFSET]
        + 2 * packed_i[SELF_HEATING_HEADER_INDEX_LASER_LINE_COUNT] * LAST_EVALUATION_CACHE_FIELD_COUNT
        + LAST_EVALUATION_METADATA_FIELD_RING_INDEX
    ] = -1.0


# prepare one world's optical arrays for the implicit thermal solver.
# if this ring ended the step at this temperature, how much optical power would it absorb?
# use the bus_propagate inputs, with phi_ref and nrz["phi1"] evaluated at t_ref_k.
# lam_m gives laser wavelengths in m.
# s0_m_per_k and beta_per_k define the resonance shift -
# shift = s0*(dT + beta*dT**2/2).
# commit_slope requests derivative outputs.
# (packed_f, packed_i) - float values/caches/outputs and integer sizes/flags/positions.
# offsets are even so complex views stay aligned.
# use unpack_self_heating_optical_state() to update inputs before each substep and read results afterwards.
# for parallel worlds, allocate separate packed arrays.
# (todo) still has some confusing var names
# (todo) remove all retired/deprecated references
# x
def pack_self_heating_optical_state(
    p_in_w: ArrayLike,  # p_in_w = P_in (W) - bus input power per laser line - type scalar, sequence, or NumPy array
    phi_ref: ArrayLike,  # phi_ref = phi_ref (rad) - round-trip phase at the reference temperature - type array-like, shape (N, L), converted to float64
    k2_th: ArrayLike,  # k2_th = K_th^2 (1) - through-coupler power fraction - type scalar, sequence, or NumPy array
    k2_dr: ArrayLike,  # k2_dr = K_dr^2 (1) - drop-coupler power fraction - type scalar, sequence, or NumPy array
    a_rt: ArrayLike,  # a_rt = a (1) - retained round-trip field amplitude - type scalar, sequence, or NumPy array
    t_rt_s: ArrayLike,  # t_rt_s = T_rt (s) - round-trip group delay - type scalar, sequence, or NumPy array
    eta: ArrayLike,  # eta = eta (1) - absorbed share of intrinsic loss - type scalar, sequence, or NumPy array
    lam_m: ArrayLike,  # lam_m = lambda_L (m) - laser wavelength - type scalar, sequence, or NumPy array
    t_ref_k: ArrayLike,  # t_ref_k = T_ref (K) - reference temperature of the supplied optical phase - type scalar, sequence, or NumPy array
    s0_m_per_k: ArrayLike,  # s0_m_per_k = s_0 (m/K) - resonance tuning slope at the reference temperature - type scalar, sequence, or NumPy array
    beta_per_k: ArrayLike = 0.0,  # beta_per_k = beta (1/K) - relative curvature of the thermal tuning slope - type scalar, sequence, or NumPy array
    bus_order: ArrayLike | None = None,  # bus_order = b_j (1) - physical ring index at bus position j - type integer array-like, shape (N,), or None
    bus_t2: ArrayLike | None = None,  # bus_t2 = B_k (1) - power transmission of the bus segment before each ring - type scalar, sequence, NumPy array, or None
    bus_t2_out: float = 1.0,  # bus_t2_out = B_out (1) - power transmission of the final bus segment - type float scalar
    psi: ArrayLike | None = None,  # psi = psi (rad) - additional coupler phase - type scalar, sequence, NumPy array, or None
    rb: ArrayLike | None = None,  # rb = r_b (1) - complex round-trip backscatter coupling by resonance order - type scalar, sequence, NumPy array, or None
    nrz: BinaryDataLevels | None = None,  # nrz = data_levels (mixed units) - optional p1 (1), phi1 (rad), a1 (1), and eta1 (1) fields - type BinaryDataLevels dict or None
    method: str = "exact",  # method = model (1) - must be exact - type str
    m_orders: int = 3,  # m_orders = reserved count (1) - retained for existing call layouts - type int scalar
    commit_slope: bool = False,  # commit_slope = slope_enabled (1) - write phase and resonance-wavelength derivatives on commit - type bool scalar
) -> tuple[
    PackedOpticalValues,  # packed_f = optics_state (mixed units) - SELF_HEATING_HEADER_* blocks with typed parameters, caches, and outputs - type float64 ndarray, shape (packed_value_count,)
    PackedOpticalIndices,  # packed_i = offsets (1) - SELF_HEATING_HEADER_* sizes, selectors, bus indices, and block offsets - type int64 ndarray, shape (packed_index_count,)
]:
    prepared_inputs = _validate_and_broadcast_bus_inputs(
        p_in_w, phi_ref, k2_th, k2_dr, a_rt, t_rt_s, eta, bus_order, bus_t2, bus_t2_out, psi, rb, nrz
    )
    (
        bus_input_power_w,
        ordered_ring_indices,
        bus_segment_power_transmission,
        output_segment_power_transmission,
        one_probability,
        level0_phase,
        level1_phase,
        level0_round_trip_field_transmission,
        level1_round_trip_field_transmission,
        level0_absorbed_share_of_loss,
        level1_absorbed_share_of_loss,
        coupler_phase,
        through_power_coupling,
        drop_power_coupling,
        round_trip_delay_s,
        backscatter_by_ring,
        use_two_data_levels,
    ) = prepared_inputs
    if method != "exact":
        raise ValueError("only exact ring optics is supported - TCMT is archived")
    if int(m_orders) < 1:
        raise ValueError("m_orders must be >= 1")
    ring_count, line_count = level0_phase.shape
    backscatter_order_count = backscatter_by_ring.shape[1]
    line_wavelength_m = _broadcast_finite_array(lam_m, (ring_count, line_count), "lam_m")
    d_phase_d_resonance_wavelength = (
        TWO_PI * C0 * round_trip_delay_s / (line_wavelength_m * line_wavelength_m)
    )
    reference_temperature_k = _broadcast_finite_array(t_ref_k, (ring_count,), "t_ref_k")
    reference_tuning_slope_m_per_k = _broadcast_finite_array(s0_m_per_k, (ring_count,), "s0_m_per_k")
    tuning_curvature_per_k = _broadcast_finite_array(beta_per_k, (ring_count,), "beta_per_k")
    sizes = (
        (SELF_HEATING_HEADER_INDEX_BUS_INPUT_POWER_OFFSET, line_count),
        (SELF_HEATING_HEADER_INDEX_CURRENT_RING_INPUT_POWER_OFFSET, line_count),
        (SELF_HEATING_HEADER_INDEX_BUS_OUTPUT_SEGMENT_TRANSMISSION_OFFSET, 1),
        (SELF_HEATING_HEADER_INDEX_RING_PARAMETER_BLOCK_OFFSET, SELF_HEATING_RING_ROW_COUNT * ring_count),
        (
            SELF_HEATING_HEADER_INDEX_RING_LINE_PARAMETER_BLOCK_OFFSET,
            SELF_HEATING_RING_LINE_ROW_COUNT * ring_count * line_count,
        ),
        (SELF_HEATING_HEADER_INDEX_BACKSCATTER_COEFFICIENTS_OFFSET, 2 * ring_count * backscatter_order_count),
        (
            SELF_HEATING_HEADER_INDEX_RING_LINE_OUTPUTS_OFFSET,
            RING_LINE_OUTPUT_ROW_COUNT * ring_count * line_count,
        ),
        (SELF_HEATING_HEADER_INDEX_RING_OUTPUTS_OFFSET, RING_OUTPUT_ROW_COUNT * ring_count),
        (SELF_HEATING_HEADER_INDEX_LASER_LINE_OUTPUTS_OFFSET, LASER_LINE_OUTPUT_ROW_COUNT * line_count),
        (SELF_HEATING_HEADER_INDEX_COMPLEX_WORKSPACE_OFFSET, 2 * COMPLEX_WORKSPACE_ROW_COUNT * line_count),
        (SELF_HEATING_HEADER_INDEX_REAL_WORKSPACE_OFFSET, REAL_WORKSPACE_ROW_COUNT * line_count),
        (SELF_HEATING_HEADER_INDEX_COUPLER_CACHE_OFFSET, ring_count * line_count * COUPLER_CACHE_FIELD_COUNT),
        (SELF_HEATING_HEADER_INDEX_AIRY_CACHE_OFFSET, 2 * ring_count * line_count * AIRY_CACHE_FIELD_COUNT),
        (
            SELF_HEATING_HEADER_INDEX_NO_BACKSCATTER_FLAGS_OFFSET,
            ring_count + ROUND_TRIP_LOSS_CACHE_TOTAL_FIELD_COUNT,
        ),
        (
            SELF_HEATING_HEADER_INDEX_LAST_EVALUATION_CACHE_OFFSET,
            2 * line_count * LAST_EVALUATION_CACHE_FIELD_COUNT + LAST_EVALUATION_METADATA_FIELD_COUNT,
        ),
        (
            SELF_HEATING_HEADER_INDEX_FAR_LINE_CACHE_OFFSET,
            2 * ring_count * line_count * FAR_LINE_CACHE_FIELD_COUNT,
        ),
    )
    packed_i = np.zeros(SELF_HEATING_HEADER_FIXED_FIELD_COUNT + 2 * ring_count, np.int64)
    (
        packed_i[SELF_HEATING_HEADER_INDEX_RING_COUNT],
        packed_i[SELF_HEATING_HEADER_INDEX_LASER_LINE_COUNT],
        packed_i[SELF_HEATING_HEADER_INDEX_BACKSCATTER_ORDER_COUNT],
    ) = (ring_count, line_count, backscatter_order_count)
    (
        packed_i[SELF_HEATING_HEADER_INDEX_TWO_DATA_LEVELS_FLAG],
        packed_i[SELF_HEATING_HEADER_INDEX_RESERVED_MODEL_FLAG],
        packed_i[SELF_HEATING_HEADER_INDEX_RESERVED_MODE_ORDER_COUNT],
    ) = (
        int(use_two_data_levels),
        0,
        int(m_orders),
    )
    packed_i[SELF_HEATING_HEADER_INDEX_COMMIT_OPTICAL_DERIVATIVES_FLAG] = int(bool(commit_slope))
    packed_i[SELF_HEATING_HEADER_INDEX_FAR_LINE_HOLD_ENABLED_FLAG] = 0
    block_offset = 0
    for key, size in sizes:
        block_offset += block_offset % 2  # even offsets - the complex views stay 16-byte aligned
        packed_i[key] = block_offset
        block_offset += size
    packed_f = np.zeros(block_offset + block_offset % 2)
    packed_i[SELF_HEATING_HEADER_FIXED_FIELD_COUNT : SELF_HEATING_HEADER_FIXED_FIELD_COUNT + ring_count] = (
        ordered_ring_indices
    )
    packed_i[SELF_HEATING_HEADER_FIXED_FIELD_COUNT + ring_count + ordered_ring_indices] = np.arange(
        ring_count
    )
    input_output_views = unpack_self_heating_optical_state(packed_f, packed_i)
    for destination_view, source_values in (
        (0, bus_input_power_w),
        (1, bus_segment_power_transmission),
        (2, output_segment_power_transmission),
        (3, one_probability),
        (4, level0_phase),
        (5, level1_phase),
        (6, level0_phase),
        (7, level1_phase),
        (8, level0_round_trip_field_transmission),
        (9, level1_round_trip_field_transmission),
        (10, level0_absorbed_share_of_loss),
        (11, level1_absorbed_share_of_loss),
        (12, coupler_phase),
        (13, through_power_coupling),
        (14, drop_power_coupling),
        (15, round_trip_delay_s),
        (16, d_phase_d_resonance_wavelength),
        (17, reference_temperature_k),
        (18, reference_tuning_slope_m_per_k),
        (19, tuning_curvature_per_k),
        (20, backscatter_by_ring),
    ):
        input_output_views[destination_view][...] = source_values
    block_offset = (
        packed_i[SELF_HEATING_HEADER_INDEX_LAST_EVALUATION_CACHE_OFFSET]
        + 2 * line_count * LAST_EVALUATION_CACHE_FIELD_COUNT
    )
    packed_f[block_offset + LAST_EVALUATION_METADATA_FIELD_RING_INDEX] = (
        -1.0
    )
    return (
        packed_f,
        packed_i,
    )


# Python API - one ring with the general steady-state field solve
# the loop of `_evaluate_ring_response` over flattened inputs.
# out_c stores four outgoing port fields in sqrt(W).
# out_f stores two energies in J, intrinsic loss in W, and three phase derivatives in W/rad.
# x
@njit(cache=True)
def evaluate_ring_response_for_input_samples(
    phi: Float64Vector,  # phi = phi (rad) - total round-trip phase including coupler phase - type float64 ndarray, shape (evaluation_count,)
    k2_th: Float64Vector,  # k2_th = K_th^2 (1) - through-coupler power fraction - type float64 ndarray, shape (evaluation_count,)
    k2_dr: Float64Vector,  # k2_dr = K_dr^2 (1) - drop-coupler power fraction - type float64 ndarray, shape (evaluation_count,)
    a_rt: Float64Vector,  # a_rt = a (1) - retained round-trip field amplitude - type float64 ndarray, shape (evaluation_count,)
    t_rt: Float64Vector,  # t_rt = T_rt (s) - round-trip group delay - type float64 ndarray, shape (evaluation_count,)
    rb: Complex128Vector,  # rb = r_b (1) - complex round-trip backscatter coupling by resonance order - type complex128 ndarray, shape (backscatter_order_count,)
    s0: Complex128Vector,  # s0 = s_in,0 (sqrt(W)) - incoming field at port 0 - type complex128 ndarray, shape (evaluation_count,)
    s1: Complex128Vector,  # s1 = s_in,1 (sqrt(W)) - incoming field at port 1 - type complex128 ndarray, shape (evaluation_count,)
    s2: Complex128Vector,  # s2 = s_in,2 (sqrt(W)) - incoming field at port 2 - type complex128 ndarray, shape (evaluation_count,)
    s3: Complex128Vector,  # s3 = s_in,3 (sqrt(W)) - incoming field at port 3 - type complex128 ndarray, shape (evaluation_count,)
    use_tcmt: bool,  # use_tcmt = reserved flag (1) - must be false to preserve the compiled call layout - type bool scalar
    m_orders: int,  # m_orders = reserved count (1) - retained for existing call layouts - type int scalar
    out_c: Complex128Matrix,  # out_c = S_out (sqrt(W)) - complex outgoing fields at ports zero through three - type complex128 ndarray, shape (4, evaluation_count)
    out_f: Float64Matrix,  # out_f = R_out (mixed units) - rows U_cw/U_ccw (J), P_loss (W), then three power slopes (W/rad) - type float64 ndarray, shape (6, evaluation_count)
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    if use_tcmt:
        raise ValueError("TCMT is archived - use exact ring optics")
    evaluation_count = phi.shape[0]
    for evaluation_index in range(evaluation_count):
        line_response = solve_exact_ring_fields_and_phase_derivatives(
            phi[evaluation_index],
            -math.log(a_rt[evaluation_index]),
            math.sqrt(1.0 - k2_th[evaluation_index]),
            math.sqrt(k2_th[evaluation_index]),
            math.sqrt(1.0 - k2_dr[evaluation_index]),
            math.sqrt(k2_dr[evaluation_index]),
            t_rt[evaluation_index],
            rb,
            s0[evaluation_index],
            s1[evaluation_index],
            s2[evaluation_index],
            s3[evaluation_index],
        )
        out_c[0, evaluation_index] = line_response[0]
        out_c[1, evaluation_index] = line_response[1]
        out_c[2, evaluation_index] = line_response[2]
        out_c[3, evaluation_index] = line_response[3]
        out_f[0, evaluation_index] = line_response[4]
        out_f[1, evaluation_index] = line_response[5]
        out_f[2, evaluation_index] = line_response[6]
        out_f[3, evaluation_index] = line_response[7]
        out_f[4, evaluation_index] = line_response[8]
        out_f[5, evaluation_index] = line_response[9]


# hold one ring's settled optical response at the broadcast input shape.
# refl = s_reflected (sqrt(W)) - complex field at port 0
# thru = s_through (sqrt(W)) - complex field at port 1
# drop = s_drop (sqrt(W)) - complex field at the monitor port 2
# add_out = s_add_out (sqrt(W)) - complex field at port 3
# u_cw = U_cw (J) - clockwise stored energy
# u_ccw = U_ccw (J) - counter-clockwise stored energy
# p_loss = P_loss (W) - absorption, scattering, and radiation
# dloss_dphi = dP_loss/dphi (W/rad)
# dthru_dphi = dP_through/dphi (W/rad)
# ddrop_dphi = dP_drop/dphi (W/rad)
# p_refl, p_thru, p_drop, and p_add_out return |s|^2 (W).
# the monitor receives p_drop, and eta*p_loss heats the ring.
@dataclass
class RingResponse:

    refl: Complex128Array
    thru: Complex128Array
    drop: Complex128Array
    add_out: Complex128Array
    u_cw: Float64Array
    u_ccw: Float64Array
    p_loss: Float64Array
    dloss_dphi: Float64Array
    dthru_dphi: Float64Array
    ddrop_dphi: Float64Array

    # return reflected power in W from the squared field magnitude.
    @property
    def p_refl(
        self: RingResponse,  # self = response (mixed units) - RingResponse fields with units defined above the class - type RingResponse record
    ) -> (
        RealValue  # return = P_reflected (W) - squared magnitude of the corresponding port field - type real scalar or real NumPy array
    ):
        return np.abs(self.refl) ** 2

    # return through-port power in W from the squared field magnitude.
    @property
    def p_thru(
        self: RingResponse,  # self = response (mixed units) - RingResponse fields with units defined above the class - type RingResponse record
    ) -> (
        RealValue  # return = P_through (W) - squared magnitude of the corresponding port field - type real scalar or real NumPy array
    ):
        return np.abs(self.thru) ** 2

    # return monitor drop-port power in W from the squared field magnitude.
    @property
    def p_drop(
        self: RingResponse,  # self = response (mixed units) - RingResponse fields with units defined above the class - type RingResponse record
    ) -> (
        RealValue  # return = P_drop (W) - squared magnitude of the corresponding port field - type real scalar or real NumPy array
    ):
        return np.abs(self.drop) ** 2

    # return add-port power in W from the squared field magnitude.
    @property
    def p_add_out(
        self: RingResponse,  # self = response (mixed units) - RingResponse fields with units defined above the class - type RingResponse record
    ) -> (
        RealValue  # return = P_add (W) - squared magnitude of the corresponding port field - type real scalar or real NumPy array
    ):
        return np.abs(self.add_out) ** 2

    # total incoming power [W] for an energy-balance check.
    def p_in_total(
        self: RingResponse,  # self = response (mixed units) - RingResponse fields with units defined above the class - type RingResponse record
        s_in: ArrayLike,  # s_in = s_in,0 (sqrt(W)) - incoming field at port 0 - type scalar, sequence, or NumPy array
        s_thru_in: ArrayLike = 0.0,  # s_thru_in = s_in,1 (sqrt(W)) - incoming field at port 1 - type scalar, sequence, or NumPy array
        s_drop_in: ArrayLike = 0.0,  # s_drop_in = s_in,2 (sqrt(W)) - incoming field at port 2 - type scalar, sequence, or NumPy array
        s_add: ArrayLike = 0.0,  # s_add = s_in,3 (sqrt(W)) - incoming field at port 3 - type scalar, sequence, or NumPy array
    ) -> (
        RealValue  # return = sum_j|s_in,j|^2 (W) - total incoming power across all four ports - type real scalar or real NumPy array
    ):
        return sum(
            np.abs(np.asarray(incoming_amplitude)) ** 2
            for incoming_amplitude in (s_in, s_thru_in, s_drop_in, s_add)
        )


# check per-order backscatter and return a one-dimensional complex array.
# None means no backscatter and becomes one zero.
# require at least one entry and |rb| < 2, so each scatterer's amplitude reflection magnitude |rb|/2 remains below one.
# x
def _validate_backscatter_array(
    rb: ArrayLike | None,  # rb = r_b (1) - complex round-trip backscatter coupling by resonance order - type scalar, sequence, NumPy array, or None
) -> (
    Complex128Vector  # return = r_b[n] (1) - one-dimensional complex coupling table - type complex128 ndarray, shape (backscatter_order_count,)
):
    if rb is None:
        return np.zeros(1, dtype=np.complex128)
    rb = np.atleast_1d(np.asarray(rb, dtype=np.complex128))
    if rb.ndim != 1 or rb.size == 0:
        raise ValueError("rb must be a 1-D array of per-order round-trip backscatter values")
    if np.any(np.abs(rb) >= 2.0):
        raise ValueError("|r_b| must be < 2 (each scatterer of the pair has |rho| = |r_b|/2 < 1)")
    return rb


# broadcast inputs to a shared shape, validate them, solve, and return RingResponse.
# x
def _evaluate_ring_response(
    phi_rad: ArrayLike,  # phi_rad = phi (rad) - round-trip phase before adding coupler phase - type scalar, sequence, or NumPy array
    k2_th: ArrayLike,  # k2_th = K_th^2 (1) - through-coupler power fraction - type scalar, sequence, or NumPy array
    k2_dr: ArrayLike,  # k2_dr = K_dr^2 (1) - drop-coupler power fraction - type scalar, sequence, or NumPy array
    a_rt: ArrayLike,  # a_rt = a (1) - retained round-trip field amplitude - type scalar, sequence, or NumPy array
    t_rt_s: ArrayLike,  # t_rt_s = T_rt (s) - round-trip group delay - type scalar, sequence, or NumPy array
    rb: ArrayLike | None,  # rb = r_b (1) - complex round-trip backscatter coupling by resonance order - type scalar, sequence, NumPy array, or None
    psi_rad: ArrayLike,  # psi_rad = psi (rad) - additional coupler phase - type scalar, sequence, or NumPy array
    s_in: ArrayLike,  # s_in = s_in,0 (sqrt(W)) - incoming field at port 0 - type scalar, sequence, or NumPy array
    s_thru_in: ArrayLike,  # s_thru_in = s_in,1 (sqrt(W)) - incoming field at port 1 - type scalar, sequence, or NumPy array
    s_drop_in: ArrayLike,  # s_drop_in = s_in,2 (sqrt(W)) - incoming field at port 2 - type scalar, sequence, or NumPy array
    s_add: ArrayLike,  # s_add = s_in,3 (sqrt(W)) - incoming field at port 3 - type scalar, sequence, or NumPy array
    use_tcmt: bool,  # use_tcmt = reserved flag (1) - must be false to preserve the compiled call layout - type bool scalar
    m_orders: int,  # m_orders = reserved count (1) - retained for existing call layouts - type int scalar
) -> (
    RingResponse  # return = RingResponse (mixed units) - fields (sqrt(W)), energies (J), loss (W), and power slopes (W/rad) - type RingResponse record
):
    broadcast_inputs = np.broadcast_arrays(
        np.asarray(phi_rad, float),
        np.asarray(k2_th, float),
        np.asarray(k2_dr, float),
        np.asarray(a_rt, float),
        np.asarray(t_rt_s, float),
        np.asarray(psi_rad, float),
        np.asarray(s_in, complex),
        np.asarray(s_thru_in, complex),
        np.asarray(s_drop_in, complex),
        np.asarray(s_add, complex),
    )
    output_shape = broadcast_inputs[0].shape
    (
        round_trip_phase,
        through_power_coupling,
        drop_power_coupling,
        round_trip_field_transmission,
        round_trip_delay_s,
        coupler_phase,
        input_field,
        backward_through_field,
        backward_drop_field,
        add_field,
    ) = (np.ascontiguousarray(input_array).ravel() for input_array in broadcast_inputs)
    if not all(
        np.all(np.isfinite(input_array))
        for input_array in (
            round_trip_phase,
            through_power_coupling,
            drop_power_coupling,
            round_trip_field_transmission,
            round_trip_delay_s,
            coupler_phase,
            input_field,
            backward_through_field,
            backward_drop_field,
            add_field,
        )
    ):
        raise ValueError("inputs must be finite")
    if np.any((through_power_coupling < 0) | (through_power_coupling >= 1)) or np.any(
        (drop_power_coupling < 0) | (drop_power_coupling >= 1)
    ):
        raise ValueError("K^2 must lie in [0, 1)")
    if np.any((round_trip_field_transmission <= 0) | (round_trip_field_transmission > 1)):
        raise ValueError("a_rt must lie in (0, 1]")
    if np.any(round_trip_delay_s <= 0):
        raise ValueError("t_rt_s must be > 0")
    outgoing_fields = np.empty((4, round_trip_phase.size), np.complex128)
    power_energy_and_slopes = np.empty((6, round_trip_phase.size), np.float64)
    evaluate_ring_response_for_input_samples(
        round_trip_phase + coupler_phase,
        through_power_coupling,
        drop_power_coupling,
        round_trip_field_transmission,
        round_trip_delay_s,
        _validate_backscatter_array(rb),
        input_field,
        backward_through_field,
        backward_drop_field,
        add_field,
        bool(use_tcmt),
        int(m_orders),
        outgoing_fields,
        power_energy_and_slopes,
    )
    field_outputs = [input_array.reshape(output_shape) for input_array in outgoing_fields]
    scalar_outputs = [input_array.reshape(output_shape) for input_array in power_energy_and_slopes]
    return RingResponse(
        field_outputs[0],
        field_outputs[1],
        field_outputs[2],
        field_outputs[3],
        scalar_outputs[0],
        scalar_outputs[1],
        scalar_outputs[2],
        scalar_outputs[3],
        scalar_outputs[4],
        scalar_outputs[5],
    )


# solve the settled, two-direction response of one add-drop ring.
# inputs follow NumPy broadcasting, and k2_dr=0 gives an all-pass ring.
# solve a 2x2 complex system with no resonance-order truncation.
# the drop coupler is halfway around the ring.
# reciprocal scatterers at one-quarter and three-quarters of a trip preserve energy.
# outgoing power plus intrinsic loss equals total incoming power to rounding.
# the scattering matrix is reciprocal and is unitary when a_rt=1.
# the scatterer pair gives near-resonance TCMT differences of order 1/finesse^2.
# use absolute phase to keep the correct rb entry for each resonance order.
# cos^2/sin^2 interpolation is flat at each resonance and preserves energy.
# re-anchoring by 2*pi*len(rb) preserves the order assignment.
# see docs/research/03_mrm_compact_models.md for the equivalent-scatterer assumptions.
# (todo) - what's the point of this function if we can just wrap the other function in here?
def evaluate_exact_ring_response(
    phi_rad: ArrayLike,  # phi_rad = phi (rad) - absolute round-trip phase - type scalar, sequence, or NumPy array
    k2_th: ArrayLike,  # k2_th = K_th^2 (1) - through-coupler power fraction - type scalar, sequence, or NumPy array
    k2_dr: ArrayLike,  # k2_dr = K_dr^2 (1) - drop-coupler power fraction - type scalar, sequence, or NumPy array
    a_rt: ArrayLike,  # a_rt = a (1) - retained round-trip field amplitude - type scalar, sequence, or NumPy array
    t_rt_s: ArrayLike,  # t_rt_s = T_rt (s) - round-trip group delay - type scalar, sequence, or NumPy array
    rb: ArrayLike | None = None,  # rb = r_b (1) - optional complex backscatter by resonance order - type scalar, sequence, NumPy array, or None
    psi_rad: ArrayLike = 0.0,  # psi_rad = psi (rad) - additional coupler phase - type scalar, sequence, or NumPy array
    s_in: ArrayLike = 1.0,  # s_in = s_0 (sqrt(W)) - incoming field at port 0 - type scalar, sequence, or NumPy array
    s_thru_in: ArrayLike = 0.0,  # s_thru_in = s_1 (sqrt(W)) - incoming field at port 1 - type scalar, sequence, or NumPy array
    s_drop_in: ArrayLike = 0.0,  # s_drop_in = s_2 (sqrt(W)) - incoming field at port 2 - type scalar, sequence, or NumPy array
    s_add: ArrayLike = 0.0,  # s_add = s_3 (sqrt(W)) - incoming field at port 3 - type scalar, sequence, or NumPy array
) -> (
    RingResponse  # return = RingResponse (mixed units) - field, energy, power, and slope units are defined above - type RingResponse record
):
    return _evaluate_ring_response(
        phi_rad, k2_th, k2_dr, a_rt, t_rt_s, rb, psi_rad, s_in, s_thru_in, s_drop_in, s_add, False, 1
    )


# return S[output_port, input_port], the 4×4 complex field scattering matrix.
# excite one port at a time with unit amplitude.
# ports follow the module's 0–3 order.
# the exact matrix is reciprocal, S=S.T, and passive, S.H*S<=I - it is unitary when a_rt=1.
# x
def ring_scattering_matrix(
    phi_rad: float,  # phi_rad = phi (rad) - round-trip phase before adding coupler phase - type float scalar
    k2_th: float,  # k2_th = K_th^2 (1) - through-coupler power fraction - type float scalar
    k2_dr: float,  # k2_dr = K_dr^2 (1) - drop-coupler power fraction - type float scalar
    a_rt: float,  # a_rt = a (1) - retained round-trip field amplitude - type float scalar
    t_rt_s: float,  # t_rt_s = T_rt (s) - round-trip group delay - type float scalar
    rb: ArrayLike | None = None,  # rb = r_b (1) - complex round-trip backscatter coupling by resonance order - type scalar, sequence, NumPy array, or None
    psi_rad: float = 0.0,  # psi_rad = psi (rad) - additional coupler phase - type float scalar
    method: str = "exact",  # method = model (1) - must be exact - type str
    m_orders: int = 3,  # m_orders = reserved count (1) - retained for existing call layouts - type int scalar
) -> (
    Complex128Matrix  # scattering_matrix = S (1) - output field divided by input field at each port pair - type complex128 ndarray, shape (4, 4)
):
    if method != "exact":
        raise ValueError("only exact ring optics is supported - TCMT is archived")
    scattering_matrix = np.empty((4, 4), np.complex128)
    for input_port, input_argument in enumerate(("s_in", "s_thru_in", "s_drop_in", "s_add")):
        response = evaluate_exact_ring_response(
            float(phi_rad),
            float(k2_th),
            float(k2_dr),
            float(a_rt),
            float(t_rt_s),
            rb=rb,
            psi_rad=float(psi_rad),
            **{"s_in": 0.0, input_argument: 1.0},
        )
        scattering_matrix[:, input_port] = [
            complex(response.refl),
            complex(response.thru),
            complex(response.drop),
            complex(response.add_out),
        ]
    return scattering_matrix


# Python API - the bus


# optical powers and stored energies after following the shared bus.
# use [physical_ring_index, laser_line_index], not bus position.
# p_in, thru, drop, add_out, refl, loss, and absorbed are powers in W - u_cw/u_ccw are energies in J.
# suffixes 0/1 are individual data levels - unsuffixed powers are averaged.
# drop_sum is total monitor input power per ring.
# abs_sum is optical heating power per ring.
# p_out is power per laser line at the bus end.
# pout0/pout1 hold one ring at a fixed data level while other rings are averaged.
# refl_in reports returned reflection power - bound estimates the omitted multiple-reflection effect.
# dabs_dphi is in W/rad and dabs_dlam in W/m when requested - otherwise those slopes are zero.
# loss_sum and drop_sum0/drop_sum1 are per-ring totals.
# x
@dataclass
class BusResult:

    p_in: Float64Matrix
    thru: Float64Matrix
    drop: Float64Matrix
    add_out: Float64Matrix
    refl: Float64Matrix
    loss: Float64Matrix
    absorbed: Float64Matrix
    u_cw: Float64Matrix
    u_ccw: Float64Matrix
    thru1: Float64Matrix
    thru0: Float64Matrix
    drop1: Float64Matrix
    drop0: Float64Matrix
    pout1: Float64Matrix
    pout0: Float64Matrix
    drop_sum: Float64Vector
    abs_sum: Float64Vector
    loss_sum: Float64Vector
    drop_sum1: Float64Vector
    drop_sum0: Float64Vector
    p_out: Float64Vector
    bound: Float64Vector
    refl_in: Float64Vector
    dabs_dphi: Float64Matrix
    dabs_dlam: Float64Vector


# broadcast values to the requested shape, make them contiguous, and reject nonfinite entries.
# x
def _broadcast_finite_array(
    values: ArrayLike,  # values = A (input units) - numerical values to broadcast - type scalar, sequence, or NumPy array
    shape: tuple[int, ...],  # shape = dimensions (1) - target array dimensions - type tuple of integer dimensions
    name: str,  # name = label (1) - diagnostic name used in an error message - type str
    dtype: DTypeLike = np.float64,  # dtype = storage_type (1) - NumPy dtype selector - type NumPy dtype selector
) -> (
    NumericArray  # broadcast_values = A_broadcast (input units) - contiguous finite array with the requested shape - type NumPy array with the requested dtype and shape
):
    broadcast_values = np.ascontiguousarray(np.broadcast_to(np.asarray(values, dtype=dtype), shape))
    if not np.all(np.isfinite(broadcast_values)):
        raise ValueError(f"{name} has non-finite entries")
    return broadcast_values


# validate bus inputs and expand scalars/arrays into consistent ring/line shapes.
# return the arrays needed by the kernels, preserving the fixed tuple order.
# per-ring/per-line inputs have shape (N, L) - per-ring values have shape (N,).
# without binary data, level one reuses level zero.
# a bus-order entry is a physical ring index, not a resonance order.
# x
def _validate_and_broadcast_bus_inputs(
    p_in_w: ArrayLike,  # p_in_w = P_in (W) - bus input power per laser line - type scalar, sequence, or NumPy array
    phi: ArrayLike,  # phi = phi (rad) - round-trip phase before adding coupler phase - type array-like, shape (N, L), converted to float64
    k2_th: ArrayLike,  # k2_th = K_th^2 (1) - through-coupler power fraction - type scalar, sequence, or NumPy array
    k2_dr: ArrayLike,  # k2_dr = K_dr^2 (1) - drop-coupler power fraction - type scalar, sequence, or NumPy array
    a_rt: ArrayLike,  # a_rt = a (1) - retained round-trip field amplitude - type scalar, sequence, or NumPy array
    t_rt_s: ArrayLike,  # t_rt_s = T_rt (s) - round-trip group delay - type scalar, sequence, or NumPy array
    eta: ArrayLike,  # eta = eta (1) - absorbed share of intrinsic loss - type scalar, sequence, or NumPy array
    bus_order: ArrayLike | None,  # bus_order = b_j (1) - physical ring index at bus position j - type integer array-like, shape (N,), or None
    bus_t2: ArrayLike | None,  # bus_t2 = B_k (1) - power transmission of the bus segment before each ring - type scalar, sequence, NumPy array, or None
    bus_t2_out: float,  # bus_t2_out = B_out (1) - power transmission of the final bus segment - type float scalar
    psi: ArrayLike | None,  # psi = psi (rad) - additional coupler phase - type scalar, sequence, NumPy array, or None
    rb: ArrayLike | None,  # rb = r_b (1) - complex round-trip backscatter coupling by resonance order - type scalar, sequence, NumPy array, or None
    nrz: BinaryDataLevels | None,  # nrz = data_levels (mixed units) - optional p1 (1), phi1 (rad), a1 (1), and eta1 (1) fields - type BinaryDataLevels dict or None
) -> tuple[
    Float64Vector,  # return[0] = P_in (W) - input power per laser line - type float64 ndarray, shape (L,)
    Int64Vector,  # return[1] = bus_order (1) - physical ring indices in traversal order - type int64 ndarray, shape (N,)
    Float64Vector,  # return[2] = B_k (1) - preceding bus segment power transmissions - type float64 ndarray, shape (N,)
    float,  # return[3] = B_out (1) - final bus segment power transmission - type float scalar
    Float64Vector,  # return[4] = p_1 (1) - ones density per ring - type float64 ndarray, shape (N,)
    Float64Matrix,  # return[5] = phi_0 (rad) - level-zero round-trip phases - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[6] = phi_1 (rad) - level-one round-trip phases - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[7] = a_0 (1) - level-zero round-trip field transmissions - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[8] = a_1 (1) - level-one round-trip field transmissions - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[9] = eta_0 (1) - level-zero absorbed shares of intrinsic loss - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[10] = eta_1 (1) - level-one absorbed shares of intrinsic loss - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[11] = psi (rad) - coupler phase - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[12] = K_th^2 (1) - through-coupler power coupling - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[13] = K_dr^2 (1) - drop-coupler power coupling - type float64 ndarray, shape (N, L)
    Float64Matrix,  # return[14] = T_rt (s) - round-trip group delay - type float64 ndarray, shape (N, L)
    Complex128Matrix,  # return[15] = r_b (1) - backscatter by ring and resonance order - type complex128 ndarray, shape (N, backscatter_order_count)
    bool,  # return[16] = nrz_enabled (1) - true when two data levels are supplied - type bool scalar
]:
    p_in_w = np.ascontiguousarray(np.atleast_1d(np.asarray(p_in_w, dtype=np.float64)))
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 2 or phi.shape[1] != p_in_w.shape[0]:
        raise ValueError("phi must be (N, L) with L = len(p_in_w)")
    ring_count, line_count = phi.shape
    ring_line_shape = (ring_count, line_count)
    level0_phase = _broadcast_finite_array(phi, ring_line_shape, "phi")
    through_power_coupling = _broadcast_finite_array(k2_th, ring_line_shape, "k2_th")
    drop_power_coupling = _broadcast_finite_array(k2_dr, ring_line_shape, "k2_dr")
    level0_round_trip_field_transmission = _broadcast_finite_array(a_rt, ring_line_shape, "a_rt")
    round_trip_delay_s = _broadcast_finite_array(t_rt_s, ring_line_shape, "t_rt_s")
    level0_absorbed_share_of_loss = _broadcast_finite_array(eta, ring_line_shape, "eta")
    coupler_phase = _broadcast_finite_array(0.0 if psi is None else psi, ring_line_shape, "psi")
    if np.any((through_power_coupling < 0) | (through_power_coupling >= 1)) or np.any(
        (drop_power_coupling < 0) | (drop_power_coupling >= 1)
    ):
        raise ValueError("K^2 must lie in [0, 1)")
    if np.any(
        (level0_round_trip_field_transmission <= 0) | (level0_round_trip_field_transmission > 1)
    ) or np.any(round_trip_delay_s <= 0):
        raise ValueError("a_rt must lie in (0, 1] and t_rt_s > 0")
    if np.any((level0_absorbed_share_of_loss < 0) | (level0_absorbed_share_of_loss > 1)):
        raise ValueError("eta must lie in [0, 1]")
    if not np.all(np.isfinite(p_in_w)) or np.any(p_in_w < 0):
        raise ValueError("p_in_w must be finite and >= 0")
    ordered_ring_indices = (
        np.arange(ring_count, dtype=np.int64)
        if bus_order is None
        else np.ascontiguousarray(bus_order, dtype=np.int64)
    )
    if ordered_ring_indices.shape != (ring_count,) or not np.array_equal(
        np.sort(ordered_ring_indices), np.arange(ring_count)
    ):
        raise ValueError("bus_order must be a permutation of range(N)")
    bus_segment_power_transmission = _broadcast_finite_array(
        1.0 if bus_t2 is None else bus_t2, (ring_count,), "bus_t2"
    )
    if (
        np.any((bus_segment_power_transmission < 0) | (bus_segment_power_transmission > 1))
        or not 0.0 <= float(bus_t2_out) <= 1.0
    ):
        raise ValueError("bus_t2 and bus_t2_out must lie in [0, 1]")
    if rb is None:
        backscatter_by_ring = np.zeros((ring_count, 1), np.complex128)
    else:
        backscatter_by_ring = np.asarray(rb, dtype=np.complex128)
        if backscatter_by_ring.ndim == 1:
            backscatter_by_ring = np.broadcast_to(
                backscatter_by_ring, (ring_count, backscatter_by_ring.shape[0])
            )
        backscatter_by_ring = np.ascontiguousarray(backscatter_by_ring)
        if backscatter_by_ring.shape[0] != ring_count or backscatter_by_ring.shape[1] < 1:
            raise ValueError("rb must be (M_b,) or (N, M_b)")
        if np.any(np.abs(backscatter_by_ring) >= 2.0):
            raise ValueError("|r_b| must be < 2")
    if nrz is None:
        use_two_data_levels = False
        one_probability = np.zeros(ring_count)
        level1_phase, level1_round_trip_field_transmission, level1_absorbed_share_of_loss = (
            level0_phase,
            level0_round_trip_field_transmission,
            level0_absorbed_share_of_loss,
        )
    else:
        use_two_data_levels = True
        one_probability = _broadcast_finite_array(nrz["p1"], (ring_count,), "p1")
        level1_phase = _broadcast_finite_array(nrz.get("phi1", level0_phase), ring_line_shape, "phi1")
        level1_round_trip_field_transmission = _broadcast_finite_array(
            nrz.get("a1", level0_round_trip_field_transmission), ring_line_shape, "a1"
        )
        level1_absorbed_share_of_loss = _broadcast_finite_array(
            nrz.get("eta1", level0_absorbed_share_of_loss), ring_line_shape, "eta1"
        )
        if np.any((one_probability < 0) | (one_probability > 1)):
            raise ValueError("p1 must lie in [0, 1]")
        if np.any(
            (level1_round_trip_field_transmission <= 0) | (level1_round_trip_field_transmission > 1)
        ) or np.any((level1_absorbed_share_of_loss < 0) | (level1_absorbed_share_of_loss > 1)):
            raise ValueError("nrz a1 must lie in (0, 1] and eta1 in [0, 1]")
    return (
        p_in_w,
        ordered_ring_indices,
        bus_segment_power_transmission,
        float(bus_t2_out),
        one_probability,
        level0_phase,
        level1_phase,
        level0_round_trip_field_transmission,
        level1_round_trip_field_transmission,
        level0_absorbed_share_of_loss,
        level1_absorbed_share_of_loss,
        coupler_phase,
        through_power_coupling,
        drop_power_coupling,
        round_trip_delay_s,
        backscatter_by_ring,
        use_two_data_levels,
    )


# follow each laser line through N rings and return BusResult.
# p_in_w is input power per laser line (L,), in W.
# phi is phase at level zero (N,L), in rad.
# couplings k2_th/k2_dr, amplitude a_rt, delay t_rt_s, and absorbed-loss fraction eta broadcast to (N,L).
# bus_order gives physical ring indices in encounter order.
# bus_t2 is segment POWER transmission before each ring - bus_t2_out is transmission after the last ring.
# psi adds coupler phase.
# rb holds per-order backscatter.
# nrz supplies p1 and level-one phi1/a1/eta1 - None uses one fixed data level.
# method must be exact - m_orders is retained for call compatibility.
# supply lam_m, in m, to request absorbed-power derivatives as well.
# multiply upstream through power by the intervening bus transmission.
# average independent binary data before passing power on.
# different laser lines contribute powers independently.
# reflected light is reported but not re-injected for repeated passes between rings.
# for steady input, eps sums |reflection_i|*|reflection_j| over ring pairs.
# each reflection is a dimensionless field coefficient.
# the forward-power relative error bound is 1/(1-eps)**2-1 for eps<1 - it is infinite otherwise.
# with binary data it uses RMS reflection amplitudes to bound the averaged error to leading order, not the worst data pattern.
# it can grow when several rings approach the same laser line.
# see the compact-model notes and tests/verification/test_optics.py.
# x
def bus_propagate(
    p_in_w: ArrayLike,  # p_in_w = P_in (W) - bus input power per laser line - type scalar, sequence, or NumPy array
    phi: ArrayLike,  # phi = phi (rad) - round-trip phase before adding coupler phase - type array-like, shape (N, L), converted to float64
    k2_th: ArrayLike,  # k2_th = K_th^2 (1) - through-coupler power fraction - type scalar, sequence, or NumPy array
    k2_dr: ArrayLike,  # k2_dr = K_dr^2 (1) - drop-coupler power fraction - type scalar, sequence, or NumPy array
    a_rt: ArrayLike,  # a_rt = a (1) - retained round-trip field amplitude - type scalar, sequence, or NumPy array
    t_rt_s: ArrayLike,  # t_rt_s = T_rt (s) - round-trip group delay - type scalar, sequence, or NumPy array
    eta: ArrayLike,  # eta = eta (1) - absorbed share of intrinsic loss - type scalar, sequence, or NumPy array
    bus_order: ArrayLike | None = None,  # bus_order = b_j (1) - physical ring index at bus position j - type integer array-like, shape (N,), or None
    bus_t2: ArrayLike | None = None,  # bus_t2 = B_k (1) - power transmission of the bus segment before each ring - type scalar, sequence, NumPy array, or None
    bus_t2_out: float = 1.0,  # bus_t2_out = B_out (1) - power transmission of the final bus segment - type float scalar
    psi: ArrayLike | None = None,  # psi = psi (rad) - additional coupler phase - type scalar, sequence, NumPy array, or None
    rb: ArrayLike | None = None,  # rb = r_b (1) - complex round-trip backscatter coupling by resonance order - type scalar, sequence, NumPy array, or None
    nrz: BinaryDataLevels | None = None,  # nrz = data_levels (mixed units) - optional p1 (1), phi1 (rad), a1 (1), and eta1 (1) fields - type BinaryDataLevels dict or None
    method: str = "exact",  # method = model (1) - must be exact - type str
    m_orders: int = 3,  # m_orders = reserved count (1) - retained for existing call layouts - type int scalar
    lam_m: ArrayLike | None = None,  # lam_m = lambda_L (m) - laser wavelength - type scalar, sequence, NumPy array, or None
) -> (
    BusResult  # return = BusResult (mixed units) - powers (W), energies (J), slopes (W/rad and W/m), and bound (1) - type BusResult record
):
    prepared_inputs = _validate_and_broadcast_bus_inputs(
        p_in_w, phi, k2_th, k2_dr, a_rt, t_rt_s, eta, bus_order, bus_t2, bus_t2_out, psi, rb, nrz
    )
    (
        bus_input_power_w,
        ordered_ring_indices,
        bus_segment_power_transmission,
        output_segment_power_transmission,
        one_probability,
        level0_phase,
        level1_phase,
        level0_round_trip_field_transmission,
        level1_round_trip_field_transmission,
        level0_absorbed_share_of_loss,
        level1_absorbed_share_of_loss,
        coupler_phase,
        through_power_coupling,
        drop_power_coupling,
        round_trip_delay_s,
        backscatter_by_ring,
        use_two_data_levels,
    ) = prepared_inputs
    ring_count, line_count = level0_phase.shape
    if method != "exact":
        raise ValueError("only exact ring optics is supported - TCMT is archived")
    if int(m_orders) < 1:
        raise ValueError("m_orders must be >= 1")
    with_slope = lam_m is not None
    if with_slope:
        line_wavelength_m = _broadcast_finite_array(lam_m, (ring_count, line_count), "lam_m")
        d_phase_d_resonance_wavelength = np.ascontiguousarray(
            TWO_PI * C0 * round_trip_delay_s / (line_wavelength_m * line_wavelength_m)
        )
    else:
        d_phase_d_resonance_wavelength = np.zeros((ring_count, line_count))
    ring_line_outputs, ring_outputs, line_outputs = make_bus_outputs(ring_count, line_count)
    complex_workspace, real_workspace, current_ring_power_w = make_workspace(line_count)
    propagate_bus_optical_power_into_output_buffers(
        bus_input_power_w,
        ordered_ring_indices,
        bus_segment_power_transmission,
        output_segment_power_transmission,
        one_probability,
        level0_phase,
        level1_phase,
        level0_round_trip_field_transmission,
        level1_round_trip_field_transmission,
        level0_absorbed_share_of_loss,
        level1_absorbed_share_of_loss,
        coupler_phase,
        through_power_coupling,
        drop_power_coupling,
        round_trip_delay_s,
        backscatter_by_ring,
        d_phase_d_resonance_wavelength,
        use_two_data_levels,
        False,
        int(m_orders),
        with_slope,
        ring_line_outputs,
        ring_outputs,
        line_outputs,
        current_ring_power_w,
        complex_workspace,
        real_workspace,
    )
    return BusResult(
        p_in=ring_line_outputs[RING_LINE_OUTPUT_ROW_INPUT_POWER],
        thru=ring_line_outputs[RING_LINE_OUTPUT_ROW_THROUGH_POWER],
        drop=ring_line_outputs[RING_LINE_OUTPUT_ROW_CLOCKWISE_DROP_POWER],
        add_out=ring_line_outputs[RING_LINE_OUTPUT_ROW_COUNTERCLOCKWISE_ADD_PORT_POWER],
        refl=ring_line_outputs[RING_LINE_OUTPUT_ROW_REFLECTED_POWER],
        loss=ring_line_outputs[RING_LINE_OUTPUT_ROW_INTRINSIC_LOSS_POWER],
        absorbed=ring_line_outputs[RING_LINE_OUTPUT_ROW_ABSORBED_POWER],
        u_cw=ring_line_outputs[RING_LINE_OUTPUT_ROW_CLOCKWISE_STORED_ENERGY],
        u_ccw=ring_line_outputs[RING_LINE_OUTPUT_ROW_COUNTERCLOCKWISE_STORED_ENERGY],
        thru1=ring_line_outputs[RING_LINE_OUTPUT_ROW_LEVEL_ONE_THROUGH_POWER],
        thru0=ring_line_outputs[RING_LINE_OUTPUT_ROW_LEVEL_ZERO_THROUGH_POWER],
        drop1=ring_line_outputs[RING_LINE_OUTPUT_ROW_LEVEL_ONE_CLOCKWISE_DROP_POWER],
        drop0=ring_line_outputs[RING_LINE_OUTPUT_ROW_LEVEL_ZERO_CLOCKWISE_DROP_POWER],
        pout1=ring_line_outputs[RING_LINE_OUTPUT_ROW_LEVEL_ONE_BUS_OUTPUT_POWER],
        pout0=ring_line_outputs[RING_LINE_OUTPUT_ROW_LEVEL_ZERO_BUS_OUTPUT_POWER],
        drop_sum=ring_outputs[RING_OUTPUT_ROW_TOTAL_CLOCKWISE_DROP_POWER],
        abs_sum=ring_outputs[RING_OUTPUT_ROW_TOTAL_ABSORBED_POWER],
        loss_sum=ring_outputs[RING_OUTPUT_ROW_TOTAL_INTRINSIC_LOSS_POWER],
        drop_sum1=ring_outputs[RING_OUTPUT_ROW_LEVEL_ONE_TOTAL_CLOCKWISE_DROP_POWER],
        drop_sum0=ring_outputs[RING_OUTPUT_ROW_LEVEL_ZERO_TOTAL_CLOCKWISE_DROP_POWER],
        p_out=line_outputs[LASER_LINE_OUTPUT_ROW_BUS_OUTPUT_POWER],
        bound=line_outputs[LASER_LINE_OUTPUT_ROW_FORWARD_POWER_RELATIVE_ERROR_BOUND],
        refl_in=line_outputs[LASER_LINE_OUTPUT_ROW_BUS_INPUT_RETURNED_REFLECTION_POWER],
        dabs_dphi=ring_line_outputs[RING_LINE_OUTPUT_ROW_ABSORBED_POWER_ROUND_TRIP_PHASE_DERIVATIVE],
        dabs_dlam=ring_outputs[RING_OUTPUT_ROW_ABSORBED_POWER_RESONANCE_WAVELENGTH_DERIVATIVE],
    )


# return each ring's absorbed power (W) and resonance-wavelength derivative (W/m).
# p_in_ring_w has shape (N,L) and fixes the input power already delivered to each ring.
# lam_m is the laser wavelength in m.
# other inputs follow bus_propagate().
# shift each ring's resonance comb rigidly - hold input power, coupling, attenuation, and absorption fraction fixed.
# this analytic derivative is used by the implicit thermal solver.
# (todo) why are there so many wrappers??
# x
def absorbed_power_and_resonance_slope(
    p_in_ring_w: ArrayLike,  # p_in_ring_w = P_in,k,l (W) - input power at each ring and laser line - type array-like, shape (N, L), converted to float64
    phi: ArrayLike,  # phi = phi (rad) - round-trip phase before adding coupler phase - type array-like, shape (N, L), converted to float64
    k2_th: ArrayLike,  # k2_th = K_th^2 (1) - through-coupler power fraction - type scalar, sequence, or NumPy array
    k2_dr: ArrayLike,  # k2_dr = K_dr^2 (1) - drop-coupler power fraction - type scalar, sequence, or NumPy array
    a_rt: ArrayLike,  # a_rt = a (1) - retained round-trip field amplitude - type scalar, sequence, or NumPy array
    t_rt_s: ArrayLike,  # t_rt_s = T_rt (s) - round-trip group delay - type scalar, sequence, or NumPy array
    eta: ArrayLike,  # eta = eta (1) - absorbed share of intrinsic loss - type scalar, sequence, or NumPy array
    lam_m: ArrayLike,  # lam_m = lambda_L (m) - laser wavelength - type scalar, sequence, or NumPy array
    psi: ArrayLike | None = None,  # psi = psi (rad) - additional coupler phase - type scalar, sequence, NumPy array, or None
    rb: ArrayLike | None = None,  # rb = r_b (1) - complex round-trip backscatter coupling by resonance order - type scalar, sequence, NumPy array, or None
    nrz: BinaryDataLevels | None = None,  # nrz = data_levels (mixed units) - optional p1 (1), phi1 (rad), a1 (1), and eta1 (1) fields - type BinaryDataLevels dict or None
    method: str = "exact",  # method = model (1) - must be exact - type str
    m_orders: int = 3,  # m_orders = reserved count (1) - retained for existing call layouts - type int scalar
) -> tuple[
    Float64Vector,  # absorbed_power_w = P_abs (W) - absorbed power per ring - type float64 ndarray, shape (N,)
    Float64Vector,  # d_absorbed_power_d_resonance_wavelength = dP_abs/dlambda_r (W/m) - rigid resonance-shift derivative - type float64 ndarray, shape (N,)
]:
    if method != "exact":
        raise ValueError("only exact ring optics is supported - TCMT is archived")
    p_in_ring_w = np.ascontiguousarray(np.asarray(p_in_ring_w, dtype=np.float64))
    if p_in_ring_w.ndim != 2:
        raise ValueError("p_in_ring_w must be (N, L)")
    prepared_inputs = _validate_and_broadcast_bus_inputs(
        p_in_ring_w[0], phi, k2_th, k2_dr, a_rt, t_rt_s, eta, None, None, 1.0, psi, rb, nrz
    )
    (
        _,
        _,
        _,
        _,
        one_probability,
        level0_phase,
        level1_phase,
        level0_round_trip_field_transmission,
        level1_round_trip_field_transmission,
        level0_absorbed_share_of_loss,
        level1_absorbed_share_of_loss,
        coupler_phase,
        through_power_coupling,
        drop_power_coupling,
        round_trip_delay_s,
        backscatter_by_ring,
        use_two_data_levels,
    ) = prepared_inputs
    ring_count, line_count = level0_phase.shape
    line_wavelength_m = _broadcast_finite_array(lam_m, (ring_count, line_count), "lam_m")
    d_phase_d_resonance_wavelength = np.ascontiguousarray(
        TWO_PI * C0 * round_trip_delay_s / (line_wavelength_m * line_wavelength_m)
    )
    if p_in_ring_w.shape != (ring_count, line_count):
        raise ValueError("p_in_ring_w must be (N, L)")
    complex_workspace, real_workspace, _ = make_workspace(line_count)
    absorbed_power_w = np.empty(ring_count)
    d_absorbed_power_d_resonance_wavelength = np.empty(ring_count)
    for ring_index in range(ring_count):
        absorbed_power_w[ring_index], d_absorbed_power_d_resonance_wavelength[ring_index] = (
            evaluate_ring_absorbed_power_and_resonance_wavelength_derivative(
                ring_index,
                np.ascontiguousarray(p_in_ring_w[ring_index]),
                one_probability[ring_index],
                level0_phase,
                level1_phase,
                level0_round_trip_field_transmission,
                level1_round_trip_field_transmission,
                level0_absorbed_share_of_loss,
                level1_absorbed_share_of_loss,
                coupler_phase,
                through_power_coupling,
                drop_power_coupling,
                round_trip_delay_s,
                backscatter_by_ring,
                d_phase_d_resonance_wavelength,
                use_two_data_levels,
                False,
                int(m_orders),
                complex_workspace,
                real_workspace,
            )
        )
    return (
        absorbed_power_w,
        d_absorbed_power_d_resonance_wavelength,
    )


# previous names remain available to existing callers, tests, and compiled kernels.
_interpolated_scatterer_amplitude = interpolate_scatterer_reflection_and_phase_derivative
_quarter_ring_energy_and_loss = calculate_quarter_ring_energy_per_power_and_loss_fraction
_exact_line = solve_exact_ring_fields_and_energy
_exact_line_d = solve_exact_ring_fields_and_phase_derivatives
_phase_parts = calculate_propagation_and_scatterer_coefficients
_exact_stages = solve_ring_fields_at_all_laser_wavelengths
_exact_stages_slope = solve_ring_fields_and_loss_phase_derivative_at_all_laser_wavelengths
ring_rows_nb = evaluate_ring_response_per_input_watt
dpabs_dlambda_res_nb = evaluate_ring_absorbed_power_and_resonance_wavelength_derivative
pabs_bounds_nb = bound_ring_absorbed_power_and_phase_curvature
pabs_slope_bound_nb = bound_ring_absorbed_power_phase_derivative
bus_finalize_nb = write_bus_output_and_returned_reflection_powers
_bus_pout_nb = write_bus_output_powers
_bus_refl_nb = write_returned_reflection_power_and_forward_error_bound
bus_propagate_nb = propagate_bus_optical_power_into_output_buffers
_selfheating_input_views = unpack_self_heating_optical_inputs
_selfheating_output_views = unpack_self_heating_optical_outputs
selfheat_views = unpack_self_heating_optical_state
selfheat_bounds = bound_absorbed_power_temperature_response
selfheat_bounds_general = bound_absorbed_power_temperature_response_from_ring_parameters
pack_selfheat_args = pack_self_heating_optical_state
_evaluate_ring_input_array = evaluate_ring_response_for_input_samples
ring_exact = evaluate_exact_ring_response
last_optical_evaluation_matches = cache_matches_last_optical_evaluation
clear_last_optical_evaluation = cache_clear_last_optical_evaluation
# (todo) get rid of this
ring_lines_nb = evaluate_ring_optical_outputs
tcmt_rates = energy_decay_rates_from_ring
beta_to_rb = backscatter_hz_to_round_trip_coupling
dphi_dlambda_res = phase_sensitivity_to_resonance_shift
nrz_average = average_binary_data_levels
ring_smatrix = ring_scattering_matrix
dpabs_dlambda_res = absorbed_power_and_resonance_slope
_pos_mod = _wrap_order_index
_rho_pair = interpolate_scatterer_reflection_and_phase_derivative
_abs2 = _squared_magnitude
_dabs2 = _squared_magnitude_derivative
_seg_factors = calculate_quarter_ring_energy_per_power_and_loss_fraction
_inv = _complex_reciprocal
# _has_backscatter = _ring_has_backscatter
_sh_core = unpack_self_heating_optical_inputs
_sh_outputs = unpack_self_heating_optical_outputs
_sh_prepare = _set_trial_temperature_phase
_ring_vec = evaluate_ring_response_for_input_samples
_as_rb = _validate_backscatter_array
_ring_eval = _evaluate_ring_response
_finite_array = _broadcast_finite_array
_prepare_bus = _validate_and_broadcast_bus_inputs
selfheat_pabs = absorbed_power_at_trial_temperature
selfheat_commit = commit_optics_at_solved_temperature
sh_pass_on = pass_through_power_to_next_ring
sh_last_is = cache_matches_last_optical_evaluation
sh_last_clear = cache_clear_last_optical_evaluation

# compatibility names share the single general implementation
selfheat_pabs_general = absorbed_power_at_trial_temperature
selfheat_commit_general = commit_optics_at_solved_temperature
sh_bounds = bound_absorbed_power_temperature_response
