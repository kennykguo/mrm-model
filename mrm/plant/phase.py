# convert temperature, junction bias, and laser wavelength into ring-optics inputs.
# PhaseParams packs fixed ring parameters and the junction table.
# evaluate_ring_optical_inputs computes one world's optical inputs at both data levels.
# the engine calls evaluate_binary_ring_optical_inputs_with_workspace and evaluate_coupler_power_fractions separately so it can reuse quantities that have not changed.
# heat and bias shift the resonance.
# the cold phase curve moves by that shift, so phase changes at a fixed laser wavelength.
# evaluate_ring_resonance_shifts describes the shift.
# write_ring_optical_inputs_for_data_level describes phase, loss, absorbed fraction, and round-trip delay.
# evaluate_coupler_power_fractions evaluates couplers at the actual laser wavelength.
# phase changes circulating light and absorbed heat.
# temperature changes phase in return.
# this module evaluates optics inputs at a supplied temperature.
# mrm.plant.selfheat solves the coupled thermal update.
# it does not follow optical picosecond dynamics in the control simulation.
# reference state - ring temperature optics.t_ref_k, reverse bias optics.bias_ref_v, heater off, and vanishing optical input.
# optics.lambda_res_cold_m is the resonance of order m in that state.
# n_eff*L = m*lambda_res exactly.
# the phase-consistent n_eff comes from relation R3 in mrm.params.relations.
# outputs are phase in rad, per-pass power coupling, round-trip field transmission, absorbed loss fraction, and group delay in s.
# lengths and wavelengths use m, temperatures K, and junction reverse bias V.
# B denotes worlds, N rings, L laser lines, and G junction-grid points.
# kernels also run without numba.
# model limitation - all resonance orders receive the same wavelength shift.
# write_ring_optical_inputs_for_data_level explains its accuracy limits.
# see docs/KNOWN_ISSUES.md, 'the phase model shifts all resonance orders together'.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Any, TypeAlias
from numpy.typing import ArrayLike, NDArray

import numpy as np
from numba import njit

from mrm.plant.electrical import JunctionTable, junction_eval_row, junction_table_from_plant_params
from mrm.units import C0

__all__ = ["RING_PARAMETER_COLUMN_COLD_RESONANCE_WAVELENGTH", "RING_PARAMETER_COLUMN_EFFECTIVE_INDEX", "RING_PARAMETER_COLUMN_GROUP_INDEX", "RING_PARAMETER_COLUMN_GROUP_INDEX_WAVELENGTH_DERIVATIVE", "RING_PARAMETER_COLUMN_ROUND_TRIP_LENGTH", "RING_PARAMETER_COLUMN_REFERENCE_TEMPERATURE", "RING_PARAMETER_COLUMN_THERMAL_TUNING_SLOPE", "RING_PARAMETER_COLUMN_RELATIVE_THERMAL_TUNING_CURVATURE", "RING_PARAMETER_COLUMN_TUNING_EXPANSION_TEMPERATURE",
           "RING_PARAMETER_COLUMN_THROUGH_POWER_COUPLING", "RING_PARAMETER_COLUMN_DROP_POWER_COUPLING", "RING_PARAMETER_COLUMN_LOG_THROUGH_COUPLING_WAVELENGTH_DERIVATIVE", "RING_PARAMETER_COLUMN_LOG_DROP_COUPLING_WAVELENGTH_DERIVATIVE", "RING_PARAMETER_COLUMN_INTRINSIC_POWER_LOSS_PER_LENGTH", "RING_PARAMETER_COLUMN_NONABSORPTIVE_POWER_LOSS_PER_LENGTH", "RING_PARAMETER_COLUMN_LOG_LOSS_RING_TEMPERATURE_DERIVATIVE",
           "RING_PARAMETER_COLUMN_LOG_LOSS_HEATER_TEMPERATURE_DERIVATIVE", "RING_PARAMETER_COLUMN_JUNCTION_FILL_FRACTION", "RING_PARAMETER_COLUMN_REFERENCE_REVERSE_BIAS", "RING_PARAMETER_COLUMN_REFERENCE_JUNCTION_INDEX_CHANGE", "RING_PARAMETER_COLUMN_REFERENCE_JUNCTION_POWER_LOSS_PER_LENGTH", "RING_PARAMETER_COLUMN_COUNT", "PhaseParams",
           "evaluate_ring_optical_inputs", "evaluate_binary_ring_optical_inputs", "evaluate_coupler_power_fractions", "evaluate_ring_resonance_shifts", "rebase_thermal_tuning_coefficients",
           "calculate_thermal_resonance_shift_m", "calculate_bus_segment_power_transmission"]


# packed per-ring parameters (PhaseParams)
# column indices of PhaseParams.ring (B, N, RING_PARAMETER_COLUMN_COUNT), one row per ring - all SI
RING_PARAMETER_COLUMN_COLD_RESONANCE_WAVELENGTH = 0       # cold resonance wavelength lambda_res (m)
RING_PARAMETER_COLUMN_EFFECTIVE_INDEX = 1         # phase-consistent effective index n_eff at lambda_res, T_ref, V_ref (1)
RING_PARAMETER_COLUMN_GROUP_INDEX = 2           # group index n_g at lambda_res (1)
RING_PARAMETER_COLUMN_GROUP_INDEX_WAVELENGTH_DERIVATIVE = 3           # group-index dispersion dn_g/dlambda (1/m)
RING_PARAMETER_COLUMN_ROUND_TRIP_LENGTH = 4           # round-trip length L = 2 pi r_ref (m)
RING_PARAMETER_COLUMN_REFERENCE_TEMPERATURE = 5         # reference ring temperature T_ref of lambda_res (K)
RING_PARAMETER_COLUMN_THERMAL_TUNING_SLOPE = 6            # thermo-optic slope s0 = dlambda_res/dT at T0 (m/K)
RING_PARAMETER_COLUMN_RELATIVE_THERMAL_TUNING_CURVATURE = 7          # relative curvature beta of the thermo-optic slope (1/K)
RING_PARAMETER_COLUMN_TUNING_EXPANSION_TEMPERATURE = 8            # temperature T0 where the slope is s0 (K)
RING_PARAMETER_COLUMN_THROUGH_POWER_COUPLING = 9           # through-coupler per-pass power coupling K^2 at lambda_res (1)
RING_PARAMETER_COLUMN_DROP_POWER_COUPLING = 10          # drop-coupler per-pass power coupling K^2 at lambda_res (1)
RING_PARAMETER_COLUMN_LOG_THROUGH_COUPLING_WAVELENGTH_DERIVATIVE = 11       # through-coupling dispersion d ln K^2_t / d lambda (1/m)
RING_PARAMETER_COLUMN_LOG_DROP_COUPLING_WAVELENGTH_DERIVATIVE = 12       # drop-coupling dispersion d ln K^2_d / d lambda (1/m)
RING_PARAMETER_COLUMN_INTRINSIC_POWER_LOSS_PER_LENGTH = 13      # intrinsic power loss alpha_i at the reference state (1/m)
RING_PARAMETER_COLUMN_NONABSORPTIVE_POWER_LOSS_PER_LENGTH = 14  # alpha_i - alpha_heat at the reference state - scattering + radiation (1/m)
RING_PARAMETER_COLUMN_LOG_LOSS_RING_TEMPERATURE_DERIVATIVE = 15      # d ln alpha / dT_ring (1/K)
RING_PARAMETER_COLUMN_LOG_LOSS_HEATER_TEMPERATURE_DERIVATIVE = 16    # d ln alpha / dT_heater (1/K)
RING_PARAMETER_COLUMN_JUNCTION_FILL_FRACTION = 17         # junction fill fraction - share of the ring length that is junction (1)
RING_PARAMETER_COLUMN_REFERENCE_REVERSE_BIAS = 18        # reference reverse bias V_ref (V)
RING_PARAMETER_COLUMN_REFERENCE_JUNCTION_INDEX_CHANGE = 19      # ring-level index change f dneff(V_ref) of the junction table (1)
RING_PARAMETER_COLUMN_REFERENCE_JUNCTION_POWER_LOSS_PER_LENGTH = 20      # ring-level free-carrier loss f alpha_fca(V_ref) of the junction table (1/m)
RING_PARAMETER_COLUMN_COUNT = 21

# previous column names keep the same integer positions.
RP_LAM_RES = RING_PARAMETER_COLUMN_COLD_RESONANCE_WAVELENGTH
RP_N_EFF = RING_PARAMETER_COLUMN_EFFECTIVE_INDEX
RP_N_G = RING_PARAMETER_COLUMN_GROUP_INDEX
RP_DNG = RING_PARAMETER_COLUMN_GROUP_INDEX_WAVELENGTH_DERIVATIVE
RP_LEN = RING_PARAMETER_COLUMN_ROUND_TRIP_LENGTH
RP_T_REF = RING_PARAMETER_COLUMN_REFERENCE_TEMPERATURE
RP_S0 = RING_PARAMETER_COLUMN_THERMAL_TUNING_SLOPE
RP_BETA = RING_PARAMETER_COLUMN_RELATIVE_THERMAL_TUNING_CURVATURE
RP_T0 = RING_PARAMETER_COLUMN_TUNING_EXPANSION_TEMPERATURE
RP_K2T = RING_PARAMETER_COLUMN_THROUGH_POWER_COUPLING
RP_K2D = RING_PARAMETER_COLUMN_DROP_POWER_COUPLING
RP_DLNK2T = RING_PARAMETER_COLUMN_LOG_THROUGH_COUPLING_WAVELENGTH_DERIVATIVE
RP_DLNK2D = RING_PARAMETER_COLUMN_LOG_DROP_COUPLING_WAVELENGTH_DERIVATIVE
RP_ALPHA_I = RING_PARAMETER_COLUMN_INTRINSIC_POWER_LOSS_PER_LENGTH
RP_ALPHA_NONHEAT = RING_PARAMETER_COLUMN_NONABSORPTIVE_POWER_LOSS_PER_LENGTH
RP_CT_RING = RING_PARAMETER_COLUMN_LOG_LOSS_RING_TEMPERATURE_DERIVATIVE
RP_CT_HEATER = RING_PARAMETER_COLUMN_LOG_LOSS_HEATER_TEMPERATURE_DERIVATIVE
RP_FILL = RING_PARAMETER_COLUMN_JUNCTION_FILL_FRACTION
RP_V_REF = RING_PARAMETER_COLUMN_REFERENCE_REVERSE_BIAS
RP_FDN_REF = RING_PARAMETER_COLUMN_REFERENCE_JUNCTION_INDEX_CHANGE
RP_FAL_REF = RING_PARAMETER_COLUMN_REFERENCE_JUNCTION_POWER_LOSS_PER_LENGTH
N_RP = RING_PARAMETER_COLUMN_COUNT

_TWO_PI = 2.0 * math.pi


# array types identify precision and rank - each argument states its physical shape.
Float64Vector: TypeAlias = Annotated[NDArray[np.float64], "one-dimensional"]
Float64Matrix: TypeAlias = Annotated[NDArray[np.float64], "two-dimensional"]
Int8Vector: TypeAlias = Annotated[NDArray[np.int8], "one-dimensional"]
RealValue: TypeAlias = float | np.floating | NDArray[np.floating]


# hold packed ring parameters and junction-table columns for the optics-input kernels.
# ring has shape (B, N, RING_PARAMETER_COLUMN_COUNT), with RP_* columns.
# grid has G reverse-bias points in V.
# the table columns have shape (Bt, G).
# Bt is one for a shared table or B for one table per world.
# index change is dimensionless, loss is 1/m, capacitance density F/m, and resistance-length product ohm*m.
# x
@dataclass(frozen=True)
class PhaseParams:
    ring: Annotated[NDArray[np.float64], "shape (B, N, RING_PARAMETER_COLUMN_COUNT)"]
    grid: Float64Vector
    dneff: Float64Matrix
    alpha_fca: Float64Matrix
    capacitance: Float64Matrix
    r_series: Float64Matrix

    # return the number of simulated worlds.
    @property
    def batch(
        self: PhaseParams,  # self = parameters (mixed units) - packed ring parameters and junction tables - type PhaseParams record
    ) -> (
        int  # return = B (1) - number of worlds - type int scalar
    ):
        return int(self.ring.shape[0])

    # return world b's junction-table row, or zero for a shared table.
    def table_row(
        self: PhaseParams,  # self = parameters (mixed units) - packed ring parameters and junction tables - type PhaseParams record
        world_index: int,  # world_index = b (1) - world index - type int scalar
    ) -> (
        int  # return = b_table (1) - junction-table row for this world - type int scalar
    ):
        return world_index if self.dneff.shape[0] > 1 else 0

    # pack realized optics, thermal, and junction parameters into the fixed RP_* layout.
    # use the supplied junction table or build one from pp.
    # cache its ring-level values at the reference bias so this reference state reproduces the specified resonance and intrinsic loss.
    @classmethod
    def from_plant_params(
        cls: type[PhaseParams],  # cls = parameter_type (1) - class used to construct the parameter record - type PhaseParams class
        pp: Any,  # pp = parameters (mixed units) - realized optics, thermal, junction, layout, and laser parameters - type PlantParams record
        table: JunctionTable | None = None,  # table = table (mixed units) - optional junction table with voltage and per-length electrical fields - type JunctionTable record or None
    ) -> (
        PhaseParams  # return = parameters (mixed units) - packed ring parameters and junction tables - type PhaseParams record
    ):
        optics, thermal, junction = pp.optics, pp.thermal, pp.junction
        world_count, ring_count = int(pp.batch), int(pp.n_rings)
        junction_table = junction_table_from_plant_params(pp) if table is None else table
        ring = np.zeros((world_count, ring_count, RING_PARAMETER_COLUMN_COUNT))

        # broadcast x to (B, N) using NumPy's trailing-axis rules.
        # pass (B, 1) explicitly for per-world values.
        # a 1-D input aligns with the ring axis.
        def per_ring(
            x: ArrayLike,  # x = values (input units) - values to broadcast across worlds and rings - type array-like with broadcast shape (B, N)
        ) -> (
            Float64Matrix  # return = values (input units) - values broadcast to every world and ring - type float64 ndarray, shape (B, N)
        ):
            return np.broadcast_to(np.asarray(x, dtype=float), (world_count, ring_count))
        ring[:, :, RING_PARAMETER_COLUMN_COLD_RESONANCE_WAVELENGTH] = per_ring(optics.lambda_res_cold_m)
        ring[:, :, RING_PARAMETER_COLUMN_EFFECTIVE_INDEX] = per_ring(optics.n_eff)
        ring[:, :, RING_PARAMETER_COLUMN_GROUP_INDEX] = per_ring(optics.n_g)
        ring[:, :, RING_PARAMETER_COLUMN_GROUP_INDEX_WAVELENGTH_DERIVATIVE] = per_ring(optics.dn_g_dlambda_per_m)
        ring[:, :, RING_PARAMETER_COLUMN_ROUND_TRIP_LENGTH] = per_ring(optics.length_m)
        ring[:, :, RING_PARAMETER_COLUMN_REFERENCE_TEMPERATURE] = per_ring(optics.t_ref_k)
        ring[:, :, RING_PARAMETER_COLUMN_THERMAL_TUNING_SLOPE] = per_ring(thermal.dlambda_dt_s0_m_per_k)
        ring[:, :, RING_PARAMETER_COLUMN_RELATIVE_THERMAL_TUNING_CURVATURE] = per_ring(thermal.dlambda_dt_beta_per_k)
        ring[:, :, RING_PARAMETER_COLUMN_TUNING_EXPANSION_TEMPERATURE] = per_ring(thermal.dlambda_dt_t0_k)
        ring[:, :, RING_PARAMETER_COLUMN_THROUGH_POWER_COUPLING] = per_ring(optics.kappa2_through)
        ring[:, :, RING_PARAMETER_COLUMN_DROP_POWER_COUPLING] = per_ring(optics.kappa2_drop)
        ring[:, :, RING_PARAMETER_COLUMN_LOG_THROUGH_COUPLING_WAVELENGTH_DERIVATIVE] = per_ring(optics.dln_kappa2_through_dlambda_per_m)
        ring[:, :, RING_PARAMETER_COLUMN_LOG_DROP_COUPLING_WAVELENGTH_DERIVATIVE] = per_ring(optics.dln_kappa2_drop_dlambda_per_m)
        ring[:, :, RING_PARAMETER_COLUMN_INTRINSIC_POWER_LOSS_PER_LENGTH] = per_ring(optics.alpha_i_per_m)
        ring[:, :, RING_PARAMETER_COLUMN_NONABSORPTIVE_POWER_LOSS_PER_LENGTH] = per_ring(optics.alpha_i_per_m) - per_ring(optics.alpha_heat_per_m)
        ring[:, :, RING_PARAMETER_COLUMN_LOG_LOSS_RING_TEMPERATURE_DERIVATIVE] = per_ring(optics.dln_alpha_dt_ring_per_k)
        ring[:, :, RING_PARAMETER_COLUMN_LOG_LOSS_HEATER_TEMPERATURE_DERIVATIVE] = per_ring(optics.dln_alpha_dt_heater_per_k)
        ring[:, :, RING_PARAMETER_COLUMN_JUNCTION_FILL_FRACTION] = per_ring(junction.fill_fraction)
        ring[:, :, RING_PARAMETER_COLUMN_REFERENCE_REVERSE_BIAS] = per_ring(optics.bias_ref_v)
        # the table at the reference bias (ring level), so the reference state reproduces lambda_res and alpha_i
        v_grid = np.asarray(junction_table.v_reverse_v, float)
        for world_index in range(world_count):
            junction_table_index = world_index if junction_table.dneff.shape[0] > 1 else 0          # a shared LUT has one row
            reference_junction = np.zeros((4, ring_count))                             # f dneff, f alpha_fca, C_j, R_s at V_ref
            flags = np.zeros(ring_count, np.int8)
            junction_eval_row(np.ascontiguousarray(ring[world_index, :, RING_PARAMETER_COLUMN_REFERENCE_REVERSE_BIAS]), v_grid, junction_table.dneff[junction_table_index],
                              junction_table.alpha_fca_per_m[junction_table_index], junction_table.capacitance_f_per_m[junction_table_index],
                                  junction_table.r_series_ohm_m[junction_table_index],
                              np.ascontiguousarray(ring[world_index, :, RING_PARAMETER_COLUMN_JUNCTION_FILL_FRACTION]), np.ascontiguousarray(ring[world_index, :, RING_PARAMETER_COLUMN_ROUND_TRIP_LENGTH]),
                              reference_junction[0], reference_junction[1], reference_junction[2],
                                  reference_junction[3], flags)
            ring[world_index, :, RING_PARAMETER_COLUMN_REFERENCE_JUNCTION_INDEX_CHANGE] = reference_junction[0]
            ring[world_index, :, RING_PARAMETER_COLUMN_REFERENCE_JUNCTION_POWER_LOSS_PER_LENGTH] = reference_junction[1]
        if not np.all(np.isfinite(ring)):
            invalid_columns = sorted({i for i in range(RING_PARAMETER_COLUMN_COUNT) if not np.all(np.isfinite(ring[:, :, i]))})
            raise ValueError(f"PhaseParams: non-finite columns {invalid_columns} (check optics/thermal/junction in PlantParams)")
        return cls(ring=np.ascontiguousarray(ring), grid=v_grid, dneff=np.ascontiguousarray(junction_table.dneff),
                   alpha_fca=np.ascontiguousarray(junction_table.alpha_fca_per_m),
                   capacitance=np.ascontiguousarray(junction_table.capacitance_f_per_m),
                   r_series=np.ascontiguousarray(junction_table.r_series_ohm_m))


# bus-segment transmission
# return the bus segment's power transmission just before each ring, shaped (B, N).
# index by ring, not bus position.
# the first ring in bus_order gets one because laser.input_transmission already accounts for chip input loss.
# later rings get exp(-bus_loss_per_m*ring_spacing_on_bus_m).
# missing spacing or zero loss gives one.
# the segment after the last ring is unmodeled and uses one.
# this is the bus_t2 input to the optics bus walk and self-heating adapter.
# x
def calculate_bus_segment_power_transmission(
    pp: Any,  # pp = parameters (mixed units) - realized optics, thermal, junction, layout, and laser parameters - type PlantParams record
) -> (
    Float64Matrix  # return = B_segment (1) - bus power transmission before each physical ring - type float64 ndarray, shape (B, N)
):
    world_count, ring_count = int(pp.batch), int(pp.n_rings)
    order = np.asarray(pp.layout.bus_order, dtype=np.int64).reshape(-1)
    ring_spacing_m = float(getattr(pp.layout, "ring_spacing_on_bus_m", 0.0) or 0.0)
    bus_loss_per_m = np.broadcast_to(np.asarray(pp.laser.bus_loss_per_m, dtype=float).reshape(-1), (world_count,))
    power_transmission = np.ones((world_count, ring_count))
    if ring_count > 1 and ring_spacing_m > 0.0:
        segment_power_transmission = np.exp(-bus_loss_per_m * ring_spacing_m)
        if not np.all(np.isfinite(segment_power_transmission)) or np.any(segment_power_transmission > 1.0):
            raise ValueError(
                f"bus segment transmission must lie in [0, 1]: {segment_power_transmission} (bus_loss_per_m {bus_loss_per_m}, spacing "
                f"{ring_spacing_m} m)"
            )
        power_transmission[:, order[1:]] = segment_power_transmission[:, None]
    return power_transmission


# resonance shift and optical-input kernels

# integrate the thermal tuning slope to find the resonance shift.
# s(T) = s_0*(1 + beta*(T-T_0)).
# Delta_lambda_r = s_0*((T-T_ref) + beta*((T-T_0)^2-(T_ref-T_0)^2)/2).
# x
@njit(cache=True, _nrt=False)
def calculate_thermal_resonance_shift_m(
    t_k: RealValue,  # t_k = T (K) - current ring temperature - type real scalar or ndarray, broadcast input shape
    t_ref_k: RealValue,  # t_ref_k = T_ref (K) - reference temperature for the shift - type real scalar or ndarray, broadcast input shape
    s0: RealValue,  # s0 = s_0 (m/K) - thermal tuning slope at T_0 - type real scalar or ndarray, broadcast input shape
    beta: RealValue,  # beta = beta (1/K) - relative curvature of the tuning slope - type real scalar or ndarray, broadcast input shape
    t0: RealValue,  # t0 = T_0 (K) - temperature at which the tuning slope equals s_0 - type real scalar or ndarray, broadcast input shape
) -> (
    RealValue  # return = Delta_lambda_r (m) - shift from the reference resonance - type real scalar or ndarray, broadcast input shape
):
    return s0 * ((t_k - t_ref_k) + 0.5 * beta * ((t_k - t0) ** 2 - (t_ref_k - t0) ** 2))


# express the same thermal tuning law about the previous temperature.
# s_local = s_0*(1 + beta*(T_prev-T_0)).
# beta_local = beta/(1 + beta*(T_prev-T_0)).
# the shift for dT is exactly s_local*(dT + beta_local*dT^2/2).
# use t_ref_k=t_prev_k in the optical self-heating adapter.
# x
@njit(cache=True, _nrt=False)
def rebase_thermal_tuning_coefficients(
    t_prev_k: RealValue,  # t_prev_k = T_prev (K) - new expansion temperature - type real scalar or ndarray, broadcast input shape
    s0: RealValue,  # s0 = s_0 (m/K) - thermal tuning slope at T_0 - type real scalar or ndarray, broadcast input shape
    beta: RealValue,  # beta = beta (1/K) - relative curvature of the tuning slope - type real scalar or ndarray, broadcast input shape
    t0: RealValue,  # t0 = T_0 (K) - temperature at which the tuning slope equals s_0 - type real scalar or ndarray, broadcast input shape
) -> tuple[
    RealValue,  # return[0] = s_local (m/K) - tuning slope at T_prev - type real scalar or ndarray, broadcast input shape
    RealValue,  # return[1] = beta_local (1/K) - relative curvature at T_prev - type real scalar or ndarray, broadcast input shape
]:
    tuning_slope_ratio = 1.0 + beta * (t_prev_k - t0)
    return (
        s0 * tuning_slope_ratio,
        beta / tuning_slope_ratio,
    )


# write each ring's resonance shift and junction properties at the supplied temperature and bias.
# calculate_thermal_resonance_shift_m gives the thermal contribution in m.
# add lambda_res*(fdn(V) - fdn(V_ref))/n_g, where fdn is the ring-level effective-index change.
# this uses d(lambda)/lambda = d(n_eff)/n_g.
# source - docs/research/03_mrm_compact_models.md, section 5.2.
# rp has shape (N, RING_PARAMETER_COLUMN_COUNT).
# t_ring_k and v have N entries in K and reverse V.
# grid and *_tab are this world's junction table.
# write N shifts in m, free-carrier losses in 1/m, capacitances in F, and series resistances in ohm.
# OR FLAG_GRID_CLIP into out_flags for voltages outside the table.
# this wrapper allocates scratch space.
# evaluate_ring_resonance_shifts_with_workspace accepts reusable scratch instead.
# x
@njit(cache=True)
def evaluate_ring_resonance_shifts(
    ring_parameters: Float64Matrix,  # ring_parameters = R_ring (mixed units) - packed RP_* parameters, one row per ring - type float64 ndarray, shape (N, RING_PARAMETER_COLUMN_COUNT)
    t_ring_k: Float64Vector,  # t_ring_k = T_ring (K) - ring temperatures - type float64 ndarray, shape (N,)
    reverse_bias_v: Float64Vector,  # reverse_bias_v = V (V) - reverse bias of each ring - type float64 ndarray, shape (N,)
    bias_grid_v: Float64Vector,  # bias_grid_v = V_grid (V) - junction reverse-bias grid - type float64 ndarray, shape (G,)
    junction_index_change_table: Float64Vector,  # junction_index_change_table = Delta_n_eff (1) - junction effective-index changes on the bias grid - type float64 ndarray, shape (G,)
    junction_loss_table_per_m: Float64Vector,  # junction_loss_table_per_m = alpha_FCA (1/m) - junction power-loss coefficients on the bias grid - type float64 ndarray, shape (G,)
    junction_capacitance_table_f_per_m: Float64Vector,  # junction_capacitance_table_f_per_m = c_j (F/m) - junction capacitance per length on the bias grid - type float64 ndarray, shape (G,)
    junction_resistance_table_ohm_m: Float64Vector,  # junction_resistance_table_ohm_m = r_s (ohm*m) - junction resistance-length products on the bias grid - type float64 ndarray, shape (G,)
    out_resonance_shift_m: Float64Vector,  # out_resonance_shift_m = Delta_lambda_r (m) - resonance shifts written in place - type float64 ndarray, shape (N,)
    out_junction_loss_per_m: Float64Vector,  # out_junction_loss_per_m = f*alpha_FCA (1/m) - ring-level junction power loss written in place - type float64 ndarray, shape (N,)
    out_junction_capacitance_f: Float64Vector,  # out_junction_capacitance_f = C_j (F) - ring junction capacitance written in place - type float64 ndarray, shape (N,)
    out_series_resistance_ohm: Float64Vector,  # out_series_resistance_ohm = R_s (ohm) - ring series resistance written in place - type float64 ndarray, shape (N,)
    out_junction_flags: Int8Vector,  # out_junction_flags = flags (1) - junction clipping flags updated in place - type int8 ndarray, shape (N,)
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    evaluate_ring_resonance_shifts_with_workspace(ring_parameters, t_ring_k, reverse_bias_v, bias_grid_v, junction_index_change_table, junction_loss_table_per_m, junction_capacitance_table_f_per_m, junction_resistance_table_ohm_m, out_resonance_shift_m, out_junction_loss_per_m, out_junction_capacitance_f, out_series_resistance_ohm,
                       out_junction_flags, np.empty(ring_parameters.shape[0]))


# compute evaluate_ring_resonance_shifts with caller-owned scratch fdn, avoiding a per-call allocation.
# fdn has N entries and receives ring-level effective-index change, including the junction fill fraction.
# x
@njit(cache=True, _nrt=False)
def evaluate_ring_resonance_shifts_with_workspace(
    ring_parameters: Float64Matrix,  # ring_parameters = R_ring (mixed units) - packed RP_* parameters, one row per ring - type float64 ndarray, shape (N, RING_PARAMETER_COLUMN_COUNT)
    t_ring_k: Float64Vector,  # t_ring_k = T_ring (K) - ring temperatures - type float64 ndarray, shape (N,)
    reverse_bias_v: Float64Vector,  # reverse_bias_v = V (V) - reverse bias of each ring - type float64 ndarray, shape (N,)
    bias_grid_v: Float64Vector,  # bias_grid_v = V_grid (V) - junction reverse-bias grid - type float64 ndarray, shape (G,)
    junction_index_change_table: Float64Vector,  # junction_index_change_table = Delta_n_eff (1) - junction effective-index changes on the bias grid - type float64 ndarray, shape (G,)
    junction_loss_table_per_m: Float64Vector,  # junction_loss_table_per_m = alpha_FCA (1/m) - junction power-loss coefficients on the bias grid - type float64 ndarray, shape (G,)
    junction_capacitance_table_f_per_m: Float64Vector,  # junction_capacitance_table_f_per_m = c_j (F/m) - junction capacitance per length on the bias grid - type float64 ndarray, shape (G,)
    junction_resistance_table_ohm_m: Float64Vector,  # junction_resistance_table_ohm_m = r_s (ohm*m) - junction resistance-length products on the bias grid - type float64 ndarray, shape (G,)
    out_resonance_shift_m: Float64Vector,  # out_resonance_shift_m = Delta_lambda_r (m) - resonance shifts written in place - type float64 ndarray, shape (N,)
    out_junction_loss_per_m: Float64Vector,  # out_junction_loss_per_m = f*alpha_FCA (1/m) - ring-level junction power loss written in place - type float64 ndarray, shape (N,)
    out_junction_capacitance_f: Float64Vector,  # out_junction_capacitance_f = C_j (F) - ring junction capacitance written in place - type float64 ndarray, shape (N,)
    out_series_resistance_ohm: Float64Vector,  # out_series_resistance_ohm = R_s (ohm) - ring series resistance written in place - type float64 ndarray, shape (N,)
    out_junction_flags: Int8Vector,  # out_junction_flags = flags (1) - junction clipping flags updated in place - type int8 ndarray, shape (N,)
    junction_index_change: Float64Vector,  # junction_index_change = f*Delta_n_eff (1) - workspace for ring-level junction index change - type float64 ndarray, shape (N,)
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    n_rings = ring_parameters.shape[0]
    junction_eval_row(reverse_bias_v, bias_grid_v, junction_index_change_table, junction_loss_table_per_m, junction_capacitance_table_f_per_m, junction_resistance_table_ohm_m, ring_parameters[:, RING_PARAMETER_COLUMN_JUNCTION_FILL_FRACTION], ring_parameters[:, RING_PARAMETER_COLUMN_ROUND_TRIP_LENGTH], junction_index_change, out_junction_loss_per_m, out_junction_capacitance_f,
                      out_series_resistance_ohm, out_junction_flags)
    for ring_index in range(n_rings):
        thermal_shift = calculate_thermal_resonance_shift_m(t_ring_k[ring_index], ring_parameters[ring_index, RING_PARAMETER_COLUMN_REFERENCE_TEMPERATURE], ring_parameters[ring_index, RING_PARAMETER_COLUMN_THERMAL_TUNING_SLOPE], ring_parameters[ring_index, RING_PARAMETER_COLUMN_RELATIVE_THERMAL_TUNING_CURVATURE], ring_parameters[ring_index, RING_PARAMETER_COLUMN_TUNING_EXPANSION_TEMPERATURE])
        out_resonance_shift_m[ring_index] = thermal_shift + ring_parameters[ring_index, RING_PARAMETER_COLUMN_COLD_RESONANCE_WAVELENGTH] * (junction_index_change[ring_index] - ring_parameters[ring_index, RING_PARAMETER_COLUMN_REFERENCE_JUNCTION_INDEX_CHANGE]) / ring_parameters[ring_index, RING_PARAMETER_COLUMN_GROUP_INDEX]


# return the common temperature multiplier for both data levels' intrinsic loss.
# exp(c_ring*(T_ring - T_ref) + c_heater*(T_heater - T_ref)).
# temperatures use K, and coefficients 1/K.
# the heater temperature is the thermal model's heater-resistor observer.
# x
@njit(cache=True, inline="always")
def calculate_intrinsic_loss_temperature_factor(
    ring_parameters: Float64Matrix,  # ring_parameters = R_ring (mixed units) - packed RP_* parameters, one row per ring - type float64 ndarray, shape (N, RING_PARAMETER_COLUMN_COUNT)
    ring_index: int,  # ring_index = k (1) - physical ring index - type int scalar
    ring_temperature_k: float,  # ring_temperature_k = T_ring (K) - current ring temperature - type float scalar
    heater_temperature_k: float,  # heater_temperature_k = T_heater (K) - current heater-resistor temperature - type float scalar
) -> (
    float  # return = s_alpha (1) - temperature multiplier for intrinsic loss - type float scalar
):
    reference_temperature_k = ring_parameters[ring_index, RING_PARAMETER_COLUMN_REFERENCE_TEMPERATURE]
    return math.exp(ring_parameters[ring_index, RING_PARAMETER_COLUMN_LOG_LOSS_RING_TEMPERATURE_DERIVATIVE] * (ring_temperature_k - reference_temperature_k) + ring_parameters[ring_index, RING_PARAMETER_COLUMN_LOG_LOSS_HEATER_TEMPERATURE_DERIVATIVE] * (heater_temperature_k - reference_temperature_k))


# write phase, field transmission, absorbed loss fraction, and optional delay for one ring/data level.
# lam_l contains L laser wavelengths in m.
# shift is the resonance shift in m.
# fca_loss is the ring-level junction loss in 1/m.
# loss_tfac is the dimensionless temperature multiplier.
# write row k of phase in rad, amplitude transmission, absorbed fraction, and optional round-trip delay in s.
# x = lambda_line - shift and d = x - lambda_res locate the laser on the cold phase curve.
# n(x) = n_eff + (n_eff - n_g)*d/lambda_res - (dn_g/dlambda)*d^2/(2*lambda_res).
# phase = 2*pi*n(x)*L/x.
# this second-order expansion uses dn_eff/dlambda = (n_eff - n_g)/lambda.
# its phase equals 2*pi*m when the shifted resonance reaches the laser, for any shift.
# round-trip delay is (n_g + (dn_g/dlambda)*d)*L/C0.
# phase sensitivity to shift is 2*pi*n_g(x)*L/x^2.
# the measured cold resonance already contains the coupler phase.
# downstream optics uses psi=0 to avoid counting the coupler phase twice.
# alpha = (alpha_ref + fca_loss - fca_ref)*loss_tfac.
# keep non-absorbing loss fixed, so absorbing loss is alpha - alpha_nonheat.
# field transmission is exp(-alpha*L/2), and absorbed fraction is alpha_heat/alpha.
# clamp loss and absorbed fraction to their physical bounds.
# temperature scales the full intrinsic loss - the added loss is treated as absorptive.
# every resonance order shifts by the same amount, preserving cold wavelength FSR.
# the extracted rib160 mode table has thermal dispersion kappa_to=4.1–4.5e-4 per nm.
# this is not an optical decay rate.
# the missing correction is approximately kappa_to*shift*abs(order_wavelength - lambda_res).
# it is 0.43 percent of shift per FSR — about 33 pm at adjacent orders and 66 pm at second orders for the 7.6 nm g26 lock shift.
# at 13 nm shift, adjacent-order error is about 57 pm.
# the locked order is exact.
# this affects foreign-line heating and monitor background — about -0.15 to +0.14 percent of g26 absorbed power and up to about 1 percent in FSR stress corners.
# BND-1 checks the other-order spacing in linewidths.
# model versus solver spacings are 2.79 versus about 2.88 for g26, 1.68 versus about 1.82 for window A, and 1.33 versus about 1.26 for fsr_edge.
# their classes agree.
# test_rigid_comb_against_the_solver_table in tests/verification/test_phase.py checks this limitation.
# the full n_eff(lambda, T, V)*L(T) law needs a thermal-dispersion config key and coordinated changes to the engine, oracle, replicas, and calibration.
# x
@njit(cache=True, _nrt=False)
def write_ring_optical_inputs_for_data_level(
    ring_parameters: Float64Matrix,  # ring_parameters = R_ring (mixed units) - packed RP_* parameters, one row per ring - type float64 ndarray, shape (N, RING_PARAMETER_COLUMN_COUNT)
    ring_index: int,  # ring_index = k (1) - physical ring index - type int scalar
    laser_wavelengths_m: Float64Vector,  # laser_wavelengths_m = lambda_L (m) - vacuum laser wavelengths - type float64 ndarray, shape (L,)
    resonance_shift_m: float,  # resonance_shift_m = Delta_lambda_r (m) - thermal and junction resonance shift - type float scalar
    junction_power_loss_per_m: float,  # junction_power_loss_per_m = f*alpha_FCA (1/m) - ring-level junction power-loss coefficient - type float scalar
    loss_temperature_factor: float,  # loss_temperature_factor = s_alpha (1) - temperature multiplier for intrinsic loss - type float scalar
    out_round_trip_phase_rad: Float64Matrix,  # out_round_trip_phase_rad = phi (rad) - round-trip phases written in row k - type float64 ndarray, shape (N, L)
    out_round_trip_field_transmission: Float64Matrix,  # out_round_trip_field_transmission = a (1) - retained round-trip field amplitudes written in row k - type float64 ndarray, shape (N, L)
    out_absorbed_loss_fraction: Float64Matrix,  # out_absorbed_loss_fraction = eta (1) - absorbed shares of intrinsic loss written in row k - type float64 ndarray, shape (N, L)
    out_round_trip_delay_s: Float64Matrix,  # out_round_trip_delay_s = T_rt (s) - round-trip group delays written at data level zero - type float64 ndarray, shape (N, L)
    write_round_trip_delay: bool,  # write_round_trip_delay = delay_enabled (1) - also write round-trip group delay - type bool scalar
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    cold_resonance_m = ring_parameters[ring_index, RING_PARAMETER_COLUMN_COLD_RESONANCE_WAVELENGTH]
    reference_effective_index = ring_parameters[ring_index, RING_PARAMETER_COLUMN_EFFECTIVE_INDEX]
    reference_group_index = ring_parameters[ring_index, RING_PARAMETER_COLUMN_GROUP_INDEX]
    d_group_index_d_wavelength_per_m = ring_parameters[ring_index, RING_PARAMETER_COLUMN_GROUP_INDEX_WAVELENGTH_DERIVATIVE]
    ring_length_m = ring_parameters[ring_index, RING_PARAMETER_COLUMN_ROUND_TRIP_LENGTH]
    intrinsic_power_loss_per_m = (
        ring_parameters[ring_index, RING_PARAMETER_COLUMN_INTRINSIC_POWER_LOSS_PER_LENGTH] + junction_power_loss_per_m - ring_parameters[ring_index, RING_PARAMETER_COLUMN_REFERENCE_JUNCTION_POWER_LOSS_PER_LENGTH]
    )  # intrinsic loss before temperature scaling, 1/m
    intrinsic_power_loss_per_m *= loss_temperature_factor  # intrinsic loss after temperature scaling, 1/m
    if intrinsic_power_loss_per_m < 0.0:
        intrinsic_power_loss_per_m = 0.0
    round_trip_field_transmission = math.exp(-0.5 * intrinsic_power_loss_per_m * ring_length_m)
    absorptive_power_loss_per_m = intrinsic_power_loss_per_m - ring_parameters[ring_index, RING_PARAMETER_COLUMN_NONABSORPTIVE_POWER_LOSS_PER_LENGTH]  # alpha_heat [1/m]
    if absorptive_power_loss_per_m < 0.0:
        absorptive_power_loss_per_m = 0.0
    absorbed_share_of_intrinsic_loss = (
        absorptive_power_loss_per_m / intrinsic_power_loss_per_m if intrinsic_power_loss_per_m > 0.0 else 0.0
    )
    if absorbed_share_of_intrinsic_loss > 1.0:
        absorbed_share_of_intrinsic_loss = 1.0
    for line_index in range(laser_wavelengths_m.shape[0]):
        cold_wavelength_m = laser_wavelengths_m[line_index] - resonance_shift_m  # laser wavelength on the cold phase curve
        wavelength_offset_m = cold_wavelength_m - cold_resonance_m
        effective_index_at_cold_wavelength = (
            reference_effective_index
            + (reference_effective_index - reference_group_index) * wavelength_offset_m / cold_resonance_m
            - 0.5
            * d_group_index_d_wavelength_per_m
            * wavelength_offset_m
            * wavelength_offset_m
            / cold_resonance_m
        )
        out_round_trip_phase_rad[ring_index, line_index] = (
            _TWO_PI * effective_index_at_cold_wavelength * ring_length_m / cold_wavelength_m
        )
        out_round_trip_field_transmission[ring_index, line_index] = round_trip_field_transmission
        out_absorbed_loss_fraction[ring_index, line_index] = absorbed_share_of_intrinsic_loss
        if write_round_trip_delay:
            # group delay in s
            out_round_trip_delay_s[ring_index, line_index] = (
                (reference_group_index + d_group_index_d_wavelength_per_m * wavelength_offset_m)
                * ring_length_m
                / C0
            )


# the part of `evaluate_ring_optical_inputs` that depends on the ring state - phases, loss, heat fraction, group delay, level shifts and junction values.
# arguments follow evaluate_ring_optical_inputs.
@njit(cache=True)
def evaluate_binary_ring_optical_inputs(
    ring_parameters: Float64Matrix,  # ring_parameters = R_ring (mixed units) - packed RP_* parameters, one row per ring - type float64 ndarray, shape (N, RING_PARAMETER_COLUMN_COUNT)
    laser_wavelengths_m: Float64Vector,  # laser_wavelengths_m = lambda_L (m) - vacuum laser wavelengths - type float64 ndarray, shape (L,)
    t_ring_k: Float64Vector,  # t_ring_k = T_ring (K) - ring temperatures - type float64 ndarray, shape (N,)
    t_heater_k: Float64Vector,  # t_heater_k = T_heater (K) - heater-resistor temperatures - type float64 ndarray, shape (N,)
    level_zero_reverse_bias_v: Float64Vector,  # level_zero_reverse_bias_v = V_0 (V) - reverse bias at data level zero - type float64 ndarray, shape (N,)
    level_one_reverse_bias_v: Float64Vector,  # level_one_reverse_bias_v = V_1 (V) - reverse bias at data level one - type float64 ndarray, shape (N,)
    bias_grid_v: Float64Vector,  # bias_grid_v = V_grid (V) - junction reverse-bias grid - type float64 ndarray, shape (G,)
    junction_index_change_table: Float64Vector,  # junction_index_change_table = Delta_n_eff (1) - junction effective-index changes on the bias grid - type float64 ndarray, shape (G,)
    junction_loss_table_per_m: Float64Vector,  # junction_loss_table_per_m = alpha_FCA (1/m) - junction power-loss coefficients on the bias grid - type float64 ndarray, shape (G,)
    junction_capacitance_table_f_per_m: Float64Vector,  # junction_capacitance_table_f_per_m = c_j (F/m) - junction capacitance per length on the bias grid - type float64 ndarray, shape (G,)
    junction_resistance_table_ohm_m: Float64Vector,  # junction_resistance_table_ohm_m = r_s (ohm*m) - junction resistance-length products on the bias grid - type float64 ndarray, shape (G,)
    out_level_zero_phase_rad: Float64Matrix,  # out_level_zero_phase_rad = phi_0 (rad) - level-0 round-trip phases written in place - type float64 ndarray, shape (N, L)
    out_level_one_phase_rad: Float64Matrix,  # out_level_one_phase_rad = phi_1 (rad) - level-1 round-trip phases written in place - type float64 ndarray, shape (N, L)
    out_level_zero_field_transmission: Float64Matrix,  # out_level_zero_field_transmission = a_0 (1) - level-0 retained round-trip field amplitudes written in place - type float64 ndarray, shape (N, L)
    out_level_one_field_transmission: Float64Matrix,  # out_level_one_field_transmission = a_1 (1) - level-1 retained round-trip field amplitudes written in place - type float64 ndarray, shape (N, L)
    out_level_zero_absorbed_loss_fraction: Float64Matrix,  # out_level_zero_absorbed_loss_fraction = eta_0 (1) - level-0 absorbed shares of intrinsic loss written in place - type float64 ndarray, shape (N, L)
    out_level_one_absorbed_loss_fraction: Float64Matrix,  # out_level_one_absorbed_loss_fraction = eta_1 (1) - level-1 absorbed shares of intrinsic loss written in place - type float64 ndarray, shape (N, L)
    out_round_trip_delay_s: Float64Matrix,  # out_round_trip_delay_s = T_rt (s) - round-trip group delays written at data level zero - type float64 ndarray, shape (N, L)
    out_resonance_shift_m: Float64Matrix,  # out_resonance_shift_m = Delta_lambda_r (m) - resonance shifts written for both binary data levels - type float64 ndarray, shape (N, 2)
    out_junction_capacitance_f: Float64Matrix,  # out_junction_capacitance_f = C_j (F) - ring junction capacitance written for both binary data levels - type float64 ndarray, shape (N, 2)
    out_series_resistance_ohm: Float64Matrix,  # out_series_resistance_ohm = R_s (ohm) - ring series resistance written for both binary data levels - type float64 ndarray, shape (N, 2)
    out_junction_flags: Int8Vector,  # out_junction_flags = flags (1) - junction clipping flags updated in place - type int8 ndarray, shape (N,)
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    evaluate_binary_ring_optical_inputs_with_workspace(ring_parameters, laser_wavelengths_m, t_ring_k, t_heater_k, level_zero_reverse_bias_v, level_one_reverse_bias_v, bias_grid_v, junction_index_change_table, junction_loss_table_per_m, junction_capacitance_table_f_per_m, junction_resistance_table_ohm_m, out_level_zero_phase_rad,
                          out_level_one_phase_rad, out_level_zero_field_transmission, out_level_one_field_transmission, out_level_zero_absorbed_loss_fraction, out_level_one_absorbed_loss_fraction, out_round_trip_delay_s, out_resonance_shift_m, out_junction_capacitance_f, out_series_resistance_ohm, out_junction_flags,
                          np.empty((6, ring_parameters.shape[0])))


# `evaluate_binary_ring_optical_inputs` with the scratch ws (6, N) passed in, so the per-tick call of the kernel allocates nothing.
# workspace row 0 stores the resonance shift in m.
# row 1 stores ring-level free-carrier power loss in 1/m.
# row 2 stores junction capacitance in F.
# row 3 stores series resistance in ohm.
# row 4 stores the dimensionless junction index change.
# row 5 stores the dimensionless intrinsic-loss temperature multiplier.
@njit(cache=True, _nrt=False, inline="always")
def evaluate_binary_ring_optical_inputs_with_workspace(
    ring_parameters: Float64Matrix,  # ring_parameters = R_ring (mixed units) - packed RP_* parameters, one row per ring - type float64 ndarray, shape (N, RING_PARAMETER_COLUMN_COUNT)
    laser_wavelengths_m: Float64Vector,  # laser_wavelengths_m = lambda_L (m) - vacuum laser wavelengths - type float64 ndarray, shape (L,)
    t_ring_k: Float64Vector,  # t_ring_k = T_ring (K) - ring temperatures - type float64 ndarray, shape (N,)
    t_heater_k: Float64Vector,  # t_heater_k = T_heater (K) - heater-resistor temperatures - type float64 ndarray, shape (N,)
    level_zero_reverse_bias_v: Float64Vector,  # level_zero_reverse_bias_v = V_0 (V) - reverse bias at data level zero - type float64 ndarray, shape (N,)
    level_one_reverse_bias_v: Float64Vector,  # level_one_reverse_bias_v = V_1 (V) - reverse bias at data level one - type float64 ndarray, shape (N,)
    bias_grid_v: Float64Vector,  # bias_grid_v = V_grid (V) - junction reverse-bias grid - type float64 ndarray, shape (G,)
    junction_index_change_table: Float64Vector,  # junction_index_change_table = Delta_n_eff (1) - junction effective-index changes on the bias grid - type float64 ndarray, shape (G,)
    junction_loss_table_per_m: Float64Vector,  # junction_loss_table_per_m = alpha_FCA (1/m) - junction power-loss coefficients on the bias grid - type float64 ndarray, shape (G,)
    junction_capacitance_table_f_per_m: Float64Vector,  # junction_capacitance_table_f_per_m = c_j (F/m) - junction capacitance per length on the bias grid - type float64 ndarray, shape (G,)
    junction_resistance_table_ohm_m: Float64Vector,  # junction_resistance_table_ohm_m = r_s (ohm*m) - junction resistance-length products on the bias grid - type float64 ndarray, shape (G,)
    out_level_zero_phase_rad: Float64Matrix,  # out_level_zero_phase_rad = phi_0 (rad) - level-0 round-trip phases written in place - type float64 ndarray, shape (N, L)
    out_level_one_phase_rad: Float64Matrix,  # out_level_one_phase_rad = phi_1 (rad) - level-1 round-trip phases written in place - type float64 ndarray, shape (N, L)
    out_level_zero_field_transmission: Float64Matrix,  # out_level_zero_field_transmission = a_0 (1) - level-0 retained round-trip field amplitudes written in place - type float64 ndarray, shape (N, L)
    out_level_one_field_transmission: Float64Matrix,  # out_level_one_field_transmission = a_1 (1) - level-1 retained round-trip field amplitudes written in place - type float64 ndarray, shape (N, L)
    out_level_zero_absorbed_loss_fraction: Float64Matrix,  # out_level_zero_absorbed_loss_fraction = eta_0 (1) - level-0 absorbed shares of intrinsic loss written in place - type float64 ndarray, shape (N, L)
    out_level_one_absorbed_loss_fraction: Float64Matrix,  # out_level_one_absorbed_loss_fraction = eta_1 (1) - level-1 absorbed shares of intrinsic loss written in place - type float64 ndarray, shape (N, L)
    out_round_trip_delay_s: Float64Matrix,  # out_round_trip_delay_s = T_rt (s) - round-trip group delays written at data level zero - type float64 ndarray, shape (N, L)
    out_resonance_shift_m: Float64Matrix,  # out_resonance_shift_m = Delta_lambda_r (m) - resonance shifts written for both binary data levels - type float64 ndarray, shape (N, 2)
    out_junction_capacitance_f: Float64Matrix,  # out_junction_capacitance_f = C_j (F) - ring junction capacitance written for both binary data levels - type float64 ndarray, shape (N, 2)
    out_series_resistance_ohm: Float64Matrix,  # out_series_resistance_ohm = R_s (ohm) - ring series resistance written for both binary data levels - type float64 ndarray, shape (N, 2)
    out_junction_flags: Int8Vector,  # out_junction_flags = flags (1) - junction clipping flags updated in place - type int8 ndarray, shape (N,)
    workspace: Float64Matrix,  # workspace = workspace (mixed units) - shift, loss, capacitance, resistance, index change, and loss multiplier - type float64 ndarray, shape (6, N)
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    ring_count = ring_parameters.shape[0]
    resonance_shift_m = workspace[0]
    junction_power_loss_per_m = workspace[1]
    junction_capacitance_f = workspace[2]
    series_resistance_ohm = workspace[3]
    loss_temperature_factor = workspace[5]
    for data_level in range(2):
        reverse_bias_v = level_zero_reverse_bias_v if data_level == 0 else level_one_reverse_bias_v
        evaluate_ring_resonance_shifts_with_workspace(ring_parameters, t_ring_k, reverse_bias_v, bias_grid_v, junction_index_change_table, junction_loss_table_per_m, junction_capacitance_table_f_per_m, junction_resistance_table_ohm_m, resonance_shift_m, junction_power_loss_per_m, junction_capacitance_f, series_resistance_ohm, out_junction_flags,
                           workspace[4])
        for ring_index in range(ring_count):
            out_resonance_shift_m[ring_index, data_level] = resonance_shift_m[ring_index]
            out_junction_capacitance_f[ring_index, data_level] = junction_capacitance_f[ring_index]
            out_series_resistance_ohm[ring_index, data_level] = series_resistance_ohm[ring_index]
            if data_level == 0:
                loss_temperature_factor[ring_index] = calculate_intrinsic_loss_temperature_factor(ring_parameters, ring_index, t_ring_k[ring_index], t_heater_k[ring_index])   # one exp for both levels
                write_ring_optical_inputs_for_data_level(ring_parameters, ring_index, laser_wavelengths_m, resonance_shift_m[ring_index], junction_power_loss_per_m[ring_index], loss_temperature_factor[ring_index], out_level_zero_phase_rad, out_level_zero_field_transmission, out_level_zero_absorbed_loss_fraction, out_round_trip_delay_s,
                            True)
            else:
                write_ring_optical_inputs_for_data_level(ring_parameters, ring_index, laser_wavelengths_m, resonance_shift_m[ring_index], junction_power_loss_per_m[ring_index], loss_temperature_factor[ring_index], out_level_one_phase_rad, out_level_one_field_transmission, out_level_one_absorbed_loss_fraction, out_round_trip_delay_s,
                            False)


# write per-pass through/drop power coupling at each actual laser wavelength.
# K_j^2(lambda) = K_j,ref^2*exp(dln_kappa2_j_dlambda*(lambda - lambda_res_cold)).
# coupling dispersion uses 1/m.
# lam_l has L wavelengths in m.
# outputs have shape (N, L) and are dimensionless.
# these values depend on laser wavelength and fixed parameters, not heater code or shifted resonance.
# reuse them while lam_l is unchanged.
# clamp just below one so field self-coupling stays positive.
# x
@njit(cache=True, _nrt=False)
def evaluate_coupler_power_fractions(
    ring_parameters: Float64Matrix,  # ring_parameters = R_ring (mixed units) - packed RP_* parameters, one row per ring - type float64 ndarray, shape (N, RING_PARAMETER_COLUMN_COUNT)
    laser_wavelengths_m: Float64Vector,  # laser_wavelengths_m = lambda_L (m) - vacuum laser wavelengths - type float64 ndarray, shape (L,)
    out_through_power_coupling: Float64Matrix,  # out_through_power_coupling = K_th^2 (1) - per-pass coupler power fractions written in place - type float64 ndarray, shape (N, L)
    out_drop_power_coupling: Float64Matrix,  # out_drop_power_coupling = K_dr^2 (1) - per-pass coupler power fractions written in place - type float64 ndarray, shape (N, L)
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    n_rings = ring_parameters.shape[0]
    for ring_index in range(n_rings):
        cold_resonance_m = ring_parameters[ring_index, RING_PARAMETER_COLUMN_COLD_RESONANCE_WAVELENGTH]
        for line_index in range(laser_wavelengths_m.shape[0]):
            wavelength_offset_m = laser_wavelengths_m[line_index] - cold_resonance_m
            through_power_coupling = ring_parameters[ring_index, RING_PARAMETER_COLUMN_THROUGH_POWER_COUPLING] * math.exp(ring_parameters[ring_index, RING_PARAMETER_COLUMN_LOG_THROUGH_COUPLING_WAVELENGTH_DERIVATIVE] * wavelength_offset_m)
            drop_power_coupling = ring_parameters[ring_index, RING_PARAMETER_COLUMN_DROP_POWER_COUPLING] * math.exp(ring_parameters[ring_index, RING_PARAMETER_COLUMN_LOG_DROP_COUPLING_WAVELENGTH_DERIVATIVE] * wavelength_offset_m)
            out_through_power_coupling[ring_index, line_index] = through_power_coupling if through_power_coupling < 1.0 else 1.0 - 1e-15
            out_drop_power_coupling[ring_index, line_index] = drop_power_coupling if drop_power_coupling < 1.0 else 1.0 - 1e-15


# compute all optics inputs for one world at both data levels.
# rp is (N, RING_PARAMETER_COLUMN_COUNT).
# lam_l contains L actual vacuum laser wavelengths in m.
# t_ring_k and t_heater_k contain N temperatures in K.
# v0 and v1 contain N reverse-bias levels in V.
# use equal levels for CW or a squelched driver.
# grid and *_tab are this world's junction table.
# write (N, L) arrays for phases in rad, field transmissions, absorbed loss fractions, per-pass couplings, and round-trip delay in s.
# the delay uses level 0.
# write (N, 2) arrays for resonance shifts in m, capacitances in F, and series resistances in ohm.
# out_flags has N int8 junction-clipping flags.
# this is evaluate_binary_ring_optical_inputs followed by evaluate_coupler_power_fractions.
# x
@njit(cache=True)
def evaluate_ring_optical_inputs(
    ring_parameters: Float64Matrix,  # ring_parameters = R_ring (mixed units) - packed RP_* parameters, one row per ring - type float64 ndarray, shape (N, RING_PARAMETER_COLUMN_COUNT)
    laser_wavelengths_m: Float64Vector,  # laser_wavelengths_m = lambda_L (m) - vacuum laser wavelengths - type float64 ndarray, shape (L,)
    t_ring_k: Float64Vector,  # t_ring_k = T_ring (K) - ring temperatures - type float64 ndarray, shape (N,)
    t_heater_k: Float64Vector,  # t_heater_k = T_heater (K) - heater-resistor temperatures - type float64 ndarray, shape (N,)
    level_zero_reverse_bias_v: Float64Vector,  # level_zero_reverse_bias_v = V_0 (V) - reverse bias at data level zero - type float64 ndarray, shape (N,)
    level_one_reverse_bias_v: Float64Vector,  # level_one_reverse_bias_v = V_1 (V) - reverse bias at data level one - type float64 ndarray, shape (N,)
    bias_grid_v: Float64Vector,  # bias_grid_v = V_grid (V) - junction reverse-bias grid - type float64 ndarray, shape (G,)
    junction_index_change_table: Float64Vector,  # junction_index_change_table = Delta_n_eff (1) - junction effective-index changes on the bias grid - type float64 ndarray, shape (G,)
    junction_loss_table_per_m: Float64Vector,  # junction_loss_table_per_m = alpha_FCA (1/m) - junction power-loss coefficients on the bias grid - type float64 ndarray, shape (G,)
    junction_capacitance_table_f_per_m: Float64Vector,  # junction_capacitance_table_f_per_m = c_j (F/m) - junction capacitance per length on the bias grid - type float64 ndarray, shape (G,)
    junction_resistance_table_ohm_m: Float64Vector,  # junction_resistance_table_ohm_m = r_s (ohm*m) - junction resistance-length products on the bias grid - type float64 ndarray, shape (G,)
    out_level_zero_phase_rad: Float64Matrix,  # out_level_zero_phase_rad = phi_0 (rad) - level-0 round-trip phases written in place - type float64 ndarray, shape (N, L)
    out_level_one_phase_rad: Float64Matrix,  # out_level_one_phase_rad = phi_1 (rad) - level-1 round-trip phases written in place - type float64 ndarray, shape (N, L)
    out_level_zero_field_transmission: Float64Matrix,  # out_level_zero_field_transmission = a_0 (1) - level-0 retained round-trip field amplitudes written in place - type float64 ndarray, shape (N, L)
    out_level_one_field_transmission: Float64Matrix,  # out_level_one_field_transmission = a_1 (1) - level-1 retained round-trip field amplitudes written in place - type float64 ndarray, shape (N, L)
    out_level_zero_absorbed_loss_fraction: Float64Matrix,  # out_level_zero_absorbed_loss_fraction = eta_0 (1) - level-0 absorbed shares of intrinsic loss written in place - type float64 ndarray, shape (N, L)
    out_level_one_absorbed_loss_fraction: Float64Matrix,  # out_level_one_absorbed_loss_fraction = eta_1 (1) - level-1 absorbed shares of intrinsic loss written in place - type float64 ndarray, shape (N, L)
    out_through_power_coupling: Float64Matrix,  # out_through_power_coupling = K_th^2 (1) - per-pass coupler power fractions written in place - type float64 ndarray, shape (N, L)
    out_drop_power_coupling: Float64Matrix,  # out_drop_power_coupling = K_dr^2 (1) - per-pass coupler power fractions written in place - type float64 ndarray, shape (N, L)
    out_round_trip_delay_s: Float64Matrix,  # out_round_trip_delay_s = T_rt (s) - round-trip group delays written at data level zero - type float64 ndarray, shape (N, L)
    out_resonance_shift_m: Float64Matrix,  # out_resonance_shift_m = Delta_lambda_r (m) - resonance shifts written for both binary data levels - type float64 ndarray, shape (N, 2)
    out_junction_capacitance_f: Float64Matrix,  # out_junction_capacitance_f = C_j (F) - ring junction capacitance written for both binary data levels - type float64 ndarray, shape (N, 2)
    out_series_resistance_ohm: Float64Matrix,  # out_series_resistance_ohm = R_s (ohm) - ring series resistance written for both binary data levels - type float64 ndarray, shape (N, 2)
    out_junction_flags: Int8Vector,  # out_junction_flags = flags (1) - junction clipping flags updated in place - type int8 ndarray, shape (N,)
) -> (
    None  # return = None - results are written into the output buffers - type None
):
    evaluate_binary_ring_optical_inputs(ring_parameters, laser_wavelengths_m, t_ring_k, t_heater_k, level_zero_reverse_bias_v, level_one_reverse_bias_v, bias_grid_v, junction_index_change_table, junction_loss_table_per_m, junction_capacitance_table_f_per_m, junction_resistance_table_ohm_m, out_level_zero_phase_rad,
                           out_level_one_phase_rad, out_level_zero_field_transmission, out_level_one_field_transmission, out_level_zero_absorbed_loss_fraction, out_level_one_absorbed_loss_fraction, out_round_trip_delay_s, out_resonance_shift_m, out_junction_capacitance_f, out_series_resistance_ohm,
                           out_junction_flags)
    evaluate_coupler_power_fractions(ring_parameters, laser_wavelengths_m, out_through_power_coupling, out_drop_power_coupling)


# previous names remain aliases for callers that use the established API.
bus_t2_from_plant_params = calculate_bus_segment_power_transmission
thermal_shift_m = calculate_thermal_resonance_shift_m
local_thermo_optic = rebase_thermal_tuning_coefficients
resonance_shift_row = evaluate_ring_resonance_shifts
resonance_shift_ws = evaluate_ring_resonance_shifts_with_workspace
_loss_temperature_factor = calculate_intrinsic_loss_temperature_factor
_fill_level = write_ring_optical_inputs_for_data_level
ring_optics_levels_row = evaluate_binary_ring_optical_inputs
ring_optics_levels_ws = evaluate_binary_ring_optical_inputs_with_workspace
coupling_row = evaluate_coupler_power_fractions
ring_optics_row = evaluate_ring_optical_inputs

__all__ += ['bus_t2_from_plant_params', 'thermal_shift_m', 'local_thermo_optic', 'resonance_shift_row', 'resonance_shift_ws', '_loss_temperature_factor', '_fill_level', 'ring_optics_levels_row', 'ring_optics_levels_ws', 'coupling_row', 'ring_optics_row']

__all__ += ['RP_LAM_RES', 'RP_N_EFF', 'RP_N_G', 'RP_DNG', 'RP_LEN', 'RP_T_REF', 'RP_S0', 'RP_BETA', 'RP_T0', 'RP_K2T', 'RP_K2D', 'RP_DLNK2T', 'RP_DLNK2D', 'RP_ALPHA_I', 'RP_ALPHA_NONHEAT', 'RP_CT_RING', 'RP_CT_HEATER', 'RP_FILL', 'RP_V_REF', 'RP_FDN_REF', 'RP_FAL_REF', 'N_RP']
