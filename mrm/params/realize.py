# build the SI parameter arrays that the plant uses.
# `realize` turns a config into `PlantParams` for B simulated worlds.
# each world combines one set of uncertain design values with one device's fabrication errors.
# uncertainty is shared by the rings in a world.
# device variation can differ by ring or line.
# the order is base values, world uncertainty, optional sweep, device variation, SI conversion, physical relations, parameter groups, then consistency checks.
# solve relations after sampling so derived values agree with the sampled inputs.
# geometry variation also needs a nominal-device solve to hold mask radius and mode order while moving the couplers and material properties.
# arrays usually have shape `(B, N)` per ring, `(B, L)` per laser line, or `(B, 1)` per die.
# parameter-group arrays are read-only.
# field names state units.
# thermal assembly also receives layout distances in um for its API.
# counter-based draws use the world seed, stream, entry name, axis index, and slot.
# adding an entry leaves other draws unchanged.
# a batch element matches a run with its own seed.
# Latin-hypercube sampling is the exception — it uses the first uncertainty seed to distribute draws across the whole batch.
# config imports are deferred so plant code can import parameter types directly.

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import ndtr, ndtri

from mrm.core import rng
from mrm.params import relations
from mrm.params import schema
from mrm.params.units import to_si
from mrm.units import C0

if TYPE_CHECKING:  # pragma: no cover
    from mrm.params.config import Config

__all__ = ["PlantParams", "OpticsParams", "ThermalParams", "JunctionParams", "HeaterParams", "BiasDacParams",
           "MonitorParams", "TempSensorParams", "DatapathParams", "LaserParams", "LayoutParams", "TimingParams",
           "FidelityFlags", "ThermalInputs", "realize", "RealizeError", "JACOBIAN_TARGETS", "WORLD_VARIABLES",
           "line_frequencies_hz", "SELFHEAT_LAG_GAIN_MAX"]

# cross-ring optical heating uses the previous substep's absorption.
# bound the gain introduced by this numerical lag.
# this dimensionless limit is 0.05.
SELFHEAT_LAG_GAIN_MAX = 0.05
# max |dP_abs/d lambda| of a Lorentzian P_max/(1+(d/HWHM)^2) is P_max * 3*sqrt(3)/8 / HWHM (at d = HWHM/sqrt(3)).
_LORENTZ_MAX_SLOPE = 3.0 * math.sqrt(3.0) / 8.0
# untargeted uncertainty entries that realize() consumes by name (others are passed through in PlantParams.world)
WORLD_VARIABLES = {
    "timing.latency_scale": "multiplies every I/O latency before it is rounded to ticks (the loop-latency "
                            "uncertainty)",
}
# each output names a target, application rule, and sign.
# offset adds a linear change in config units.
# lnscale multiplies by an exponential.
# exp derives the log-slope from stored evaluations, or J/leaf, to keep exponential loss positive.
# kappa2 accumulates log-coupling changes for the sampled-geometry Q solve.
JACOBIAN_TARGETS = {
    "lambda_res_nm_per_nm": (relations.P_OFFSET, "offset", -1.0),       # offset = line - resonance
    "resonance_offset_nm_per_nm": (relations.P_OFFSET, "offset", 1.0),
    "n_eff_per_nm": (relations.P_NEFF, "offset", 1.0),
    "n_g_per_nm": (relations.P_NG, "offset", 1.0),
    "dlambda_dt_pm_per_k_per_nm": (relations.P_S0, "offset", 1.0),
    "ln_kappa2_through_per_nm": ("derived.geometry.dln_kappa2_through", "kappa2", 1.0),
    "ln_kappa2_drop_per_nm": ("derived.geometry.dln_kappa2_drop", "kappa2", 1.0),
    "alpha_rad_db_per_cm_per_nm": (relations.P_A_RAD, "exp", 1.0),
    "ln_alpha_rad_per_nm": (relations.P_A_RAD, "lnscale", 1.0),
}
# the stored evaluation (jacobian.evaluations.<input>.plus/minus) of an 'exp' output
_JACOBIAN_EVAL_NAME = {"alpha_rad_db_per_cm_per_nm": "alpha_rad"}
# coupler dispersion of each 'kappa2' output - the device's cold resonance moves by the lambda_res Jacobian, and the anchors hold at the cold resonance, so K^2 there also moves by d ln K^2/d lambda x d lambda_res
_KAPPA2_DISPERSION = {"derived.geometry.dln_kappa2_through": "ring.optics.dln_kappa2_through_dlambda_per_nm",
                      "derived.geometry.dln_kappa2_drop": "ring.optics.dln_kappa2_drop_dlambda_per_nm"}
# Jacobian input names -> variation.geometry dimension
_JACOBIAN_INPUT_DIMS = {"width_nm": "width", "film_nm": "thickness", "thickness_nm": "thickness", "slab_nm": "slab",
                        "gap_through_nm": "gap_through", "gap_drop_nm": "gap_drop"}
# the geometry dimensions, in the order they are drawn (only the keys are used - each dimension has its own RNG name, so the order does not change any draw)
_GEOMETRY_DIM_ORDER = {"width": 0, "thickness": 1, "slab": 2, "gap_through": 3, "gap_drop": 4}


# realize() could not build a consistent PlantParams (timing rules, lag bound, missing artefacts).
class RealizeError(ValueError):
    pass


# make an ndarray contiguous and read-only (other values pass through).
def _freeze(x):
    if isinstance(x, np.ndarray):
        x = np.ascontiguousarray(x)
        x.setflags(write=False)
    return x


# make each array field contiguous and read-only after construction.
class _Frozen:

    def __post_init__(self):
        for field_info in fields(self):
            value = getattr(self, field_info.name)
            if isinstance(value, np.ndarray):
                object.__setattr__(self, field_info.name, _freeze(value))


# PlantParams groups
# SI array shapes — die (B, 1), ring (B, N), line (B, L), sensor (B, S) B is the world count.
# N, L, and S count rings, laser lines, and sensors.

# cold-cavity optics at the stated reference temperature and bias.
# ring arrays have shape `(B, N)`.
# `n_eff` is phase-consistent after order rounding.
# `n_eff_configured` retains the original effective index.
@dataclass(frozen=True)
class OpticsParams(_Frozen):
    lambda_line_m: np.ndarray          # (B,N) nominal grid wavelength of each ring's assigned line
    lambda_res_cold_m: np.ndarray      # (B,N) cold resonance = lambda_line - resonance_offset
    resonance_offset_m: np.ndarray     # (B,N) blue offset (line - cold resonance)
    t_ref_k: np.ndarray                # (B,1) temperature of lambda_res_cold
    bias_ref_v: np.ndarray             # (B,1) reverse bias of lambda_res_cold
    radius_m: np.ndarray               # (B,N) r_ref of the phase model
    length_m: np.ndarray               # (B,N) 2 pi R
    n_eff: np.ndarray                  # (B,N) phase-consistent n_eff at lambda_res_cold (m lambda_res / L)
    n_eff_configured: np.ndarray       # (B,N) n_eff as configured (solver), before the order rounding
    n_g: np.ndarray                    # (B,N)
    dn_g_dlambda_per_m: np.ndarray     # (B,N) group-index dispersion (0 when absent)
    order: np.ndarray                  # (B,N) int64 longitudinal order m
    fsr_hz: np.ndarray                 # (B,N)
    round_trip_time_s: np.ndarray      # (B,N) n_g L / c
    q_loaded: np.ndarray               # (B,N)
    q_intrinsic: np.ndarray            # (B,N)
    q_through: np.ndarray              # (B,N)
    q_drop: np.ndarray                 # (B,N)
    kappa2_through: np.ndarray         # (B,N) per-pass power coupling at lambda_res_cold
    kappa2_drop: np.ndarray            # (B,N)
    round_trip_amplitude: np.ndarray   # (B,N) a = exp(-alpha_i L / 2)
    alpha_i_per_m: np.ndarray          # (B,N) intrinsic power attenuation
    alpha_heat_per_m: np.ndarray       # (B,N) heat-producing part of alpha_i
    heat_fraction: np.ndarray          # (B,N) eta
    dln_kappa2_through_dlambda_per_m: np.ndarray  # (B,N) (0 when absent)
    dln_kappa2_drop_dlambda_per_m: np.ndarray     # (B,N) (0 when absent)
    dln_alpha_dt_ring_per_k: np.ndarray           # (B,N) (0 when absent)
    dln_alpha_dt_heater_per_k: np.ndarray         # (B,N) (0 when absent)
    dn_eff_dt_per_k: np.ndarray        # (B,N) at T0 (R1)
    d2n_eff_dt2_per_k2: np.ndarray     # (B,N) (R1)
    alpha_l_per_k: np.ndarray          # (B,1) length expansion at T0
    alpha_l_slope_per_k2: np.ndarray   # (B,1) (0 when absent or order 1)
    backscatter_splitting_hz: np.ndarray  # (B,N) mode splitting (0 when backscatter is off or absent)
    fwhm_hz: np.ndarray                # (B,N) c/(lambda Q_L)
    alpha_fca_per_m: np.ndarray        # (B,N) junction FCA loss at the reference bias (nan without a loss budget)
    backscatter_rb: np.ndarray         # (B,N,M_b) complex128 rb_n = 2 pi beta_n/FSR per order (zeros when off)
    tcmt_orders: int                   # reserved legacy count - unused by exact optics


# temperature sensitivity, DC thermal gains, and the optional dynamic system.
# R1 and R2 provide the anchors.
# `system` contains the assembled dynamics.
# `inputs` retains the assembly inputs even when no system is assembled.
@dataclass(frozen=True)
class ThermalParams(_Frozen):
    dlambda_dt_s0_m_per_k: np.ndarray  # (B,N) s0 at T0
    dlambda_dt_beta_per_k: np.ndarray  # (B,1) beta (0 when thermo_optic_order = 1)
    dlambda_dt_t0_k: np.ndarray        # (B,1) T0
    t_ring_lock_k: np.ndarray          # (B,N) diagnostic - lock temperature at ambient T_ref
    s_lock_m_per_k: np.ndarray         # (B,N) s(T_lock)
    heater_efficiency_m_per_w: np.ndarray  # (B,N) FSR/P_FSR (R2)
    z_heater_ring_k_per_w: np.ndarray  # (B,N) Z_heater->ring(0) (R2)
    z_ratio_opt_heater: np.ndarray     # (B,1)
    z_opt_ring_k_per_w: np.ndarray     # (B,N) z_ratio * Z_heater->ring
    self_heating_gain_m_per_w: np.ndarray  # (B,N) s(T_lock) * Z_opt->ring  (per absorbed W)
    system: Any = None                 # result of mrm.plant.thermal.assemble, or None
    inputs: Any = None                 # ThermalInputs (always present)


# inputs for thermal assembly, retained even when no assembly hook runs.
# `interim` holds per-world `(B, 1)` values in SI.
# its distance names use m.
# heater and optical DC gains are `(B, N)` arrays in K/W.
# family and stack are enum codes with name tables.
# `scales` holds untargeted thermal uncertainty draws as `(B,)` factors.
# `dt_s` is the plant tick in seconds.
# `model` selects interim dynamics or a reduced-order model loaded from `rom_path`.
# layout sites and ring pitch keep um for the assembly API.
@dataclass(frozen=True)
class ThermalInputs:

    model: str
    rom_path: str | None
    interim: dict
    z_heater_ring_k_per_w: np.ndarray
    z_opt_ring_k_per_w: np.ndarray
    dlambda_dt_s0_m_per_k: np.ndarray
    dlambda_dt_beta_per_k: np.ndarray
    dlambda_dt_t0_k: np.ndarray
    family: np.ndarray
    family_names: tuple
    stack: np.ndarray
    stack_names: tuple
    scales: dict
    dt_s: float
    scaling_hypothesis: tuple = ()
    sites_um: dict = field(default_factory=dict)
    temp_sensors: list = field(default_factory=list)
    ring_pitch_um: float | None = None
    sha256: str | None = None


# reverse-bias junction model using an analytic depletion law or a lookup table.
# the analytic wavelength shift is `A*((V + Vbi)**m - Vbi**m)` in m.
# table values describe full-circumference index, absorption, and electrical properties.
# `fill_fraction` selects the active part of the ring.
@dataclass(frozen=True)
class JunctionParams(_Frozen):
    model: int                         # 0 analytic, 1 lut
    fill_fraction: np.ndarray          # (B,N)
    depletion_a_m: np.ndarray          # (B,N) A of the analytic law, metres (A = eff0 / (m Vbi^(m-1)))
    v_bi_v: np.ndarray                 # (B,1)
    depletion_exponent: np.ndarray     # (B,1) m
    eff_at_0v_m_per_v: np.ndarray      # (B,N) dlambda/dV at 0 V
    dqi_dv_frac_per_v: np.ndarray      # (B,N) (0 when absent)
    capacitance_at_0v_f: np.ndarray    # (B,N) (0 when absent)
    series_resistance_ohm: np.ndarray  # (B,N) (0 when absent)
    lut_v_reverse_v: np.ndarray        # (V,) LUT bias grid (empty for analytic)
    lut_dneff: np.ndarray              # (V,) full-circumference dn_eff relative to 0 V
    lut_alpha_fca_per_m: np.ndarray    # (V,) absolute FCA loss
    lut_capacitance_f_per_m: np.ndarray  # (V,)
    lut_r_series_ohm_m: np.ndarray     # (V,)
    loss_ratio_q: np.ndarray           # (B,1) q = lambda d_alpha/(4 pi d_n) (nan when absent)


# heater resistance, driver limits, DAC errors, and readback settings.
# driver codes are 0 for voltage, 1 for current, 2 for power, and 3 for DAC PWM.
# each world uses its selected law and that law's corresponding full scale.
@dataclass(frozen=True)
class HeaterParams(_Frozen):
    p_fsr_w: np.ndarray                # (B,N)
    r_cold_ohm: np.ndarray             # (B,N) at r_ref_temp_k
    r_ref_temp_k: np.ndarray           # (B,1)
    tcr_per_k: np.ndarray              # (B,1)
    driver_mode: np.ndarray            # (B,1) int64
    full_scale_v: np.ndarray           # (B,1) (nan when not given)
    full_scale_a: np.ndarray           # (B,1)
    full_scale_w: np.ndarray           # (B,1)
    dac_bits: int
    code_min: int
    code_max: int
    slew_codes_per_s: np.ndarray       # (B,1) inf = unlimited
    settle_tau_s: np.ndarray           # (B,1) 0 = no filter
    dac_inl_lsb: np.ndarray            # (B,1)
    dac_dnl_lsb: np.ndarray            # (B,1)
    compliance_v: np.ndarray           # (B,1) inf when absent
    pwm_period_s: np.ndarray           # (B,1) (nan unless dac_pwm)
    max_update_hz: float               # inf when absent
    readback_present: bool
    pwm_bits: int                      # dac_pwm PWM bits (0 when absent)
    supply_power_max_w: float          # array.heater_total_power_max_w (inf when absent)
    supply_current_max_a: float        # array.heater_supply_current_max_a (inf when absent)
    dac_bit_err_z: np.ndarray          # (B,N,dac_bits) standard normals for the per-bit DAC errors (stream variation)
    inl_bow_u: np.ndarray              # (B,N) uniforms in [-1, 1] for the INL bow (stream variation)
    # heater V/I readback (read by mrm.plant.sensing.ReadbackHw.from_plant_params)
    readback_adc_bits: int = 0                       # 0 when absent
    readback_v_full_scale_v: np.ndarray | None = None  # (B,1) nan when absent
    readback_i_full_scale_a: np.ndarray | None = None  # (B,1) nan when absent
    readback_noise_rms_codes: np.ndarray | None = None  # (B,1) 0 when absent
    readback_average: bool = False                   # boxcar over the sample period (True) or track-and-hold


# bias actuator tx.bias.
@dataclass(frozen=True)
class BiasDacParams(_Frozen):
    bits: int
    v_min_v: np.ndarray                # (B,1)
    v_max_v: np.ndarray                # (B,1)
    update_period_ticks: int           # 1 when no update rate is given
    slew_v_per_s: np.ndarray           # (B,1) inf when absent
    settle_tau_s: np.ndarray           # (B,1)
    noise_rms_codes: np.ndarray        # (B,1)


# photodiode, analog front end, and ADC parameters for the drop-port monitor.
# include dark-current temperature dependence, background, noise, gain errors, and optional reference tap.
# front-end code 0 is RC and code 1 is integrate-dump.
@dataclass(frozen=True)
class MonitorParams(_Frozen):
    responsivity_a_per_w: np.ndarray   # (B,N)
    gain_scale: np.ndarray             # (B,1) lumped nuisance factor (1 when absent)
    dark_i0_a: np.ndarray              # (B,1)
    dark_t0_k: np.ndarray              # (B,1)
    dark_activation_j: np.ndarray      # (B,1)
    background_fraction: np.ndarray    # (B,N)
    afe_type: int
    afe_tau_s: float
    integration_window_ticks: int
    adc_period_ticks: int
    adc_bits: int
    adc_full_scale_a: np.ndarray       # (B,1)
    adc_offset_codes: np.ndarray       # (B,N)
    adc_gain_error_frac: np.ndarray    # (B,N)
    adc_inl_lsb: np.ndarray            # (B,1)
    adc_dnl_lsb: np.ndarray            # (B,1)
    input_noise_a_per_rt_hz: np.ndarray  # (B,1)
    flicker_corner_hz: np.ndarray      # (B,1)
    adc_ref_tempco_per_k: np.ndarray   # (B,1)
    pga_gains: np.ndarray              # (G,) (empty when absent)
    channels_per_adc: int
    # read by mrm.plant.sensing
    mux_order: str = "round_robin"     # ring.monitor.adc_mux.order
    baseline_a: np.ndarray | None = None  # (B,1) analog baseline subtracted before the ADC (0 when absent)
    stuck_flagged: bool = True         # a stuck converter clears monitor_valid
    flicker_band_min_hz: float = 1e-3  # lowest 1/f pole of the flicker synthesis
    ref_tap_present: bool = False      # reference-tap PD on the bus before the bank
    ref_tap_coupling_frac: np.ndarray | None = None  # (B,1) bus power fraction to the tap (nan when absent)
    ref_tap_responsivity_a_per_w: np.ndarray | None = None  # (B,1) nan = the drop PD's
    ref_tap_dark_i0_a: np.ndarray | None = None  # (B,1) nan = the drop PD's
    adc_ref_site: str = "package_base"  # ring.monitor.adc_ref_site - package_base | monitor_pd


# temperature-sensor placement, sample timing, noise, and calibration errors.
# type codes are 0 for diode, 1 for RTD, and 2 for thermistor.
# site index -1 uses the attachment string instead of a named layout site.
@dataclass(frozen=True)
class TempSensorParams(_Frozen):
    names: tuple
    type: np.ndarray                   # (S,) int64
    site_index: np.ndarray             # (S,) int64 index into LayoutParams.site_names (-1 when attached)
    attach: tuple                      # (S,) attach strings ('' when a site is given)
    sample_period_ticks: np.ndarray    # (S,) int64
    latency_ticks: np.ndarray          # (B,S) int64 (scaled by the latency uncertainty)
    noise_rms_k: np.ndarray            # (B,S)
    flicker_corner_hz: np.ndarray      # (B,S)
    resolution_k: np.ndarray           # (B,S)
    offset_k: np.ndarray               # (B,S) drawn per device from offset_sigma_k
    gain_error_frac: np.ndarray        # (B,S) drawn per device from gain_error_sigma_frac
    self_heating_k: np.ndarray         # (B,S)
    adc_bits: np.ndarray | None = None  # (S,) int64 converter bits (0 when absent)
    code_zero_k: np.ndarray | None = None  # (S,) absolute temperature that code 0 reads (nan when absent)


# data path (bit statistics, error counter, BER) and the data-activity heat inputs — the junction CV^2 heat in the ring (swing, transition density, R_drv) and the driver's own heat (static power, energy per transition).
@dataclass(frozen=True)
class DatapathParams(_Frozen):
    bitrate_hz: np.ndarray             # (B,1) (nan when absent)
    bitstat_window_bits: int           # 0 when absent
    error_count_period_ticks: int      # 0 when absent
    transition_density: np.ndarray     # (B,1) (nan when absent)
    driver_output_resistance_ohm: np.ndarray  # (B,1) (nan when absent)
    driver_static_power_w: np.ndarray  # (B,1) (0 when absent)
    driver_energy_per_transition_j: np.ndarray  # (B,1) (0 when absent)
    swing_pp_v: np.ndarray             # (B,N) (nan when absent)
    level_convention: np.ndarray       # (B,1) int64 - 0 centered, 1 one_sided (driver.level_convention, per world)
    # BER and bit-statistics hardware (nan when absent - then mrm.plant.datapath uses its module constants)
    rx_sensitivity_oma_w: np.ndarray | None = None   # (B,N) receiver OMA sensitivity at ber_ref, W
    ber_ref: np.ndarray | None = None                # (B,1)
    link_transmission: np.ndarray | None = None      # (B,1) exp(-link_loss) TX bus output -> RX PD
    ber_floor: np.ndarray | None = None              # (B,1)
    bitstat_bw_hz: np.ndarray | None = None          # (B,1)
    bitstat_noise_a_per_rt_hz: np.ndarray | None = None  # (B,1)


# laser lines.
# Line k = 0..L-1 in increasing frequency unless a lines table is given.
@dataclass(frozen=True)
class LaserParams(_Frozen):
    line_freq_hz: np.ndarray           # (B,L) actual frequency = grid + static error
    line_freq_grid_hz: np.ndarray      # (L,) nominal grid
    power_w: np.ndarray                # (B,L) at the laser-module output
    input_transmission: np.ndarray     # (B,1) exp(-chip_input_loss) (laser output -> first ring)
    bus_loss_per_m: np.ndarray         # (B,1) attenuation coefficient between rings (0 when absent)
    rin_per_hz: np.ndarray             # (B,1) (0 when absent)
    linewidth_hz: np.ndarray           # (B,1) (0 when absent)
    direction: tuple                   # (L,) 'tx' | 'rx'
    source_laser: tuple = ()           # (L,) laser module of each line ('' when the lines table does not say)
    pol: tuple = ()                    # (L,) 'te' | 'tm'


# named sites on the die.
# Rings are sites ring0..ring{N-1}.
@dataclass(frozen=True)
class LayoutParams(_Frozen):
    site_names: tuple                  # (S_sites,)
    site_xy_m: np.ndarray              # (S_sites, 2)
    site_kind: tuple                   # (S_sites,)
    ring_site_index: np.ndarray        # (N,) int64
    ring_xy_m: np.ndarray              # (N, 2)
    ring_distance_m: np.ndarray        # (N, N)
    bus_order: np.ndarray              # (N,) int64 ring index at each bus position
    ring_to_line: np.ndarray           # (N,) int64 assigned line of each ring
    ring_spacing_on_bus_m: float       # 0 when absent


# timing in whole plant ticks of dt_s (s).
@dataclass(frozen=True)
class TimingParams(_Frozen):
    dt_s: float
    controller_period_ticks: int       # 0 when controller_interface.update_rate_hz is absent
    adc_period_ticks: int
    integration_window_ticks: int      # 0 for an rc front end
    monitor_latency_ticks: np.ndarray  # (B,1) int64
    heater_latency_ticks: np.ndarray   # (B,1) int64
    bias_latency_ticks: np.ndarray     # (B,1) int64
    readback_latency_ticks: np.ndarray  # (B,1) int64 (0 when absent)
    readback_period_ticks: int         # 0 when absent
    error_count_latency_ticks: np.ndarray  # (B,1) int64 (0 when absent)
    latency_scale: np.ndarray          # (B,1) the world's latency scale (1 when no entry)
    # sim.* settings the kernel needs (reset protocol, auto substeps)
    reset_protocol: int = 0            # 0 laser_ramp, 1 branch_low, 2 branch_high (sim.reset_protocol)
    reset_ramp_ticks: int = 0          # laser turn-on ramp of laser_ramp (sim.reset_ramp_s - 0 when absent)
    substep_policy: int = 0            # 0 fixed, 1 auto (sim.substep_policy)
    max_detuning_step_m: float = 0.2e-12   # sim.max_detuning_step_pm (schema default 0.2 pm)
    min_samples_per_dither_period: int = 8  # sim.min_samples_per_dither_period (schema default 8)


# model switches in the fixed integer order consumed by the compiled kernel.
@dataclass(frozen=True)
class FidelityFlags(_Frozen):
    self_heating: int        # 1 on, 0 off
    thermo_optic_order: int  # 1 or 2
    thermal_model: int       # 0 interim, 1 rom
    junction_model: int      # 0 analytic, 1 lut
    optics_model: int        # reserved model flag - zero for exact optics
    backscatter: int         # 0 off, 1 on
    shot_noise: int          # 1 on, 0 off
    pattern_noise: int       # index in the schema enum (0 off)
    data_modulation: int     # 0 mean, 1 lut
    heater_anchor: int       # 0 lock_slope, 1 fsr_integral
    heater_pwm_ripple: int   # 0 off, 1 on

    ORDER = ("self_heating", "thermo_optic_order", "thermal_model", "junction_model", "optics_model", "backscatter",
             "shot_noise", "pattern_noise", "data_modulation", "heater_anchor", "heater_pwm_ripple")

    # int32 array in the order of `FidelityFlags.ORDER`.
    def as_array(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in self.ORDER], dtype=np.int32)


# all parameter groups for a batch of simulated worlds and devices.
# `world` records each shared draw as a `(B,)` array.
# hypotheses use integer indices into `world_choices`.
# `values` maps paths to SI values for inspection.
# `derivations` records equations and inputs.
# `diagnostics` records residuals, clipping, defaults, and lag gain.
# metadata dictionaries remain ordinary mappings.
@dataclass(frozen=True)
class PlantParams:

    batch: int
    n_rings: int
    n_lines: int
    optics: OpticsParams
    thermal: ThermalParams
    junction: JunctionParams
    heater: HeaterParams
    bias_dac: BiasDacParams
    monitor: MonitorParams
    temp_sensors: TempSensorParams
    datapath: DatapathParams
    laser: LaserParams
    layout: LayoutParams
    timing: TimingParams
    flags: FidelityFlags
    world: dict
    world_choices: dict
    seeds: dict
    config_hash: str
    values: dict = field(repr=False)
    derivations: dict = field(repr=False)
    diagnostics: dict = field(repr=False)
    thermal_inputs: ThermalInputs | None = field(default=None, repr=False)

    # return one batch element with its leading batch dimension kept at one.
    # slice group arrays, nested thermal objects, world draws, seeds, values, and per-world diagnostics.
    # retain shared layout and index arrays.
    # support negative indices in the same range as Python indexing.
    def select(self, b: int) -> "PlantParams":
        import dataclasses
        B = self.batch
        if not -B <= b < B:
            raise IndexError(f"batch element {b} out of range for B = {B}")
        b = b % B

        def cut(x, min_ndim=2):
            # group arrays have a batch axis plus a value axis.
            # draws, seeds, and scales can be (B,).
            if isinstance(x, np.ndarray) and x.ndim >= min_ndim and x.shape[0] == B:
                return _freeze(x[b:b + 1].copy())
            return x

        def cut_group(g, nested=False):
            if g is None or isinstance(g, type) or not dataclasses.is_dataclass(g):
                return g
            selected_fields = {}
            for field_info in fields(g):
                value = getattr(g, field_info.name)
                if dataclasses.is_dataclass(value) and not isinstance(value, type):
                    selected_fields[field_info.name] = cut_group(value, nested=True)
                elif nested and isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype.kind == "f":
                    # nested float vectors can be per world.
                    # retain the shared integer index vectors.
                    selected_fields[field_info.name] = cut(value, 1)
                else:
                    selected_fields[field_info.name] = cut(value)
            if isinstance(g, ThermalInputs):
                selected_fields["scales"] = {k: cut(np.asarray(x), 1) for k, x in g.scales.items()}
                selected_fields["interim"] = {k: cut(np.asarray(x)) for k, x in g.interim.items()}
                if len(g.scaling_hypothesis) == B:
                    selected_fields["scaling_hypothesis"] = (g.scaling_hypothesis[b],)
            return dataclasses.replace(g, **selected_fields)

        selected_fields = {}
        for field_info in fields(self):
            value = getattr(self, field_info.name)
            if field_info.name == "layout":
                selected_fields[field_info.name] = value          # not batched
            elif dataclasses.is_dataclass(value) and not isinstance(value, type):
                selected_fields[field_info.name] = cut_group(value)
            elif field_info.name in ("world", "seeds"):
                selected_fields[field_info.name] = {k: cut(x, 1) for k, x in value.items()}
            elif field_info.name == "values":
                selected_fields[field_info.name] = {k: cut(x) for k, x in value.items()}
            elif field_info.name == "diagnostics":
                diagnostics = dict(value)
                if "rom_dc_scale" in diagnostics:
                    diagnostics["rom_dc_scale"] = cut(np.asarray(diagnostics["rom_dc_scale"]), 1)
                if "geometry_draws_nm" in diagnostics:
                    diagnostics["geometry_draws_nm"] = {k: cut(x) for k, x in diagnostics["geometry_draws_nm"].items()}
                if "geometry_dlambda_res_nm" in diagnostics:
                    diagnostics["geometry_dlambda_res_nm"] = cut(np.asarray(diagnostics["geometry_dlambda_res_nm"]))
                selected_fields[field_info.name] = diagnostics
            else:
                selected_fields[field_info.name] = value
        selected_fields["batch"] = 1
        return PlantParams(**selected_fields)


# RNG helpers

# stable 31-bit id of an entry name (the 'tick' word of the Philox counter).
def _entry_id(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF


# return `(B, k)` uniform draws keyed independently of call order.
# the seed and stream form the random key.
# the counter uses a stable entry-name ID, the index along the drawn axis, and `slot` for separate draw components.
def _uniform(seeds: np.ndarray, stream: int, name: str, k: int, slot: int = 0) -> np.ndarray:
    out = np.empty((seeds.shape[0], k), dtype=np.float64)
    rng.fill_uniform(out, seeds.astype(np.int64), np.int64(stream), np.int64(_entry_id(name)), np.int64(slot))
    return out


# return `(B, k)` standard normal draws using the inverse normal CDF.
def _normal(seeds: np.ndarray, stream: int, name: str, k: int, slot: int = 0) -> np.ndarray:
    return ndtri(_uniform(seeds, stream, name, k, slot))


# return separate device and uncertainty seed arrays, each shaped `(B,)`.
# an integer seed s expands to s + b for batch element b.
# explicit seed arrays must have B nonnegative values.
# a mapping can specify the two streams separately.
def _seed_arrays(seeds, batch: int, cfg: "Config") -> dict[str, np.ndarray]:
    def one(s, default):
        if s is None:
            s = default
        if isinstance(s, (int, np.integer)):
            return rng.batch_seeds(int(s), batch)
        seed_array = np.asarray(s, dtype=np.int64).reshape(-1)
        if seed_array.shape[0] != batch:
            raise RealizeError(f"seeds: expected {batch} per-element seeds, got {seed_array.shape[0]}")
        if np.any(seed_array < 0):
            raise RealizeError("seeds must be >= 0")
        return seed_array
    if isinstance(seeds, dict):
        device_seeds = seeds.get("device", cfg.get("experiment.seeds.device", 0))
        uncertainty_seeds = seeds.get("uncertainty", cfg.get("experiment.seeds.uncertainty", device_seeds))
        return {"device": one(device_seeds, 0), "uncertainty": one(uncertainty_seeds, 0)}
    s = one(seeds, 0)
    return {"device": s, "uncertainty": s.copy()}


# shapes

# expand a config value to its world and device axes as float64.
# a scalar fills the level.
# ring and line inputs may instead supply one value per ring or line.
# die-level inputs must be scalars.
# output shapes are `(B, 1)`, `(B, N)`, and `(B, L)` respectively.
def _broadcast(value, level: str, B: int, N: int, L: int, path: str) -> np.ndarray:
    v = np.asarray(value, dtype=np.float64)
    if level == "die":
        if v.ndim != 0:
            raise RealizeError(f"{path}: a die-level leaf must be a scalar")
        return np.full((B, 1), float(v))
    n = N if level == "ring" else L
    if v.ndim == 0:
        return np.full((B, n), float(v))
    if v.shape != (n,):
        raise RealizeError(f"{path}: expected {n} values for a {level}-level leaf, got shape {v.shape}")
    return np.tile(v[None, :], (B, 1))


# nominal grid frequencies (L,) in Hz and line directions.
# from the `channel_plan.lines` table when given, else `center + (k - (L - 1)/2) * spacing` for k = 0..L-1 (a grid "centered at" the centre frequency, increasing frequency with k).
def line_frequencies_hz(cfg: "Config") -> tuple[np.ndarray, tuple]:
    lines = cfg.get("channel_plan.lines")
    if lines:
        frequencies_hz = np.array([to_si("freq_thz", r["freq_thz"]) for r in lines], dtype=float)
        return frequencies_hz, tuple(r.get("direction", "tx") for r in lines)
    L = int(cfg.get("channel_plan.n_lines"))
    center_hz = to_si("channel_plan.center_thz", cfg.get("channel_plan.center_thz"))
    spacing_hz = to_si("channel_plan.spacing_ghz", cfg.get("channel_plan.spacing_ghz"))
    k = np.arange(L, dtype=float)
    return center_hz + (k - (L - 1) / 2.0) * spacing_hz, tuple(["tx"] * L)


# source_laser and pol of each line from the `channel_plan.lines` table ('' and 'te' when not given).
def _line_meta(cfg: "Config", L: int) -> dict:
    lines = cfg.get("channel_plan.lines") or []
    if lines:
        return {"source_laser": tuple(str(r.get("source_laser") or "") for r in lines),
                "pol": tuple(str(r.get("pol") or "te") for r in lines)}
    return {"source_laser": ("",) * L, "pol": ("te",) * L}


# default Ge PD dark-current activation energy in J — mrm.materials.ge_pd_dark_ea_ev() (0.37 eV, inferred).
def _dark_ea_default_j() -> float:
    from mrm.materials import ge_pd_dark_ea_ev
    return float(to_si("dark_activation_ev", ge_pd_dark_ea_ev()))


# named sites (ring<i> first, then layout.sites), ring distances, bus order and ring-to-line assignment.
def _layout(cfg: "Config", N: int, L: int) -> LayoutParams:
    sites = dict(cfg.get("layout.sites") or {})
    pitch = cfg.get("layout.ring_pitch_um")
    names, positions_um, kinds = [], [], []
    ring_indices = []
    for i in range(N):
        site_name = f"ring{i}"
        if site_name in sites:
            r = sites[site_name]
            p = (r["x_um"], r["y_um"])
        elif pitch is not None:
            p = (i * float(pitch), 0.0)
        else:
            if N > 1:
                raise RealizeError("layout: give layout.ring_pitch_um or ring<i> sites for every ring")
            p = (0.0, 0.0)
        ring_indices.append(len(names))
        names.append(site_name)
        positions_um.append(p)
        kinds.append("ring")
    for site_name, r in sites.items():
        if site_name in names:
            continue
        names.append(site_name)
        positions_um.append((r["x_um"], r["y_um"]))
        kinds.append(r.get("kind", "other"))
    xy_m = np.asarray(to_si("x_um", np.asarray(positions_um, dtype=float)), dtype=float).reshape(-1, 2)   # um -> m
    ring_positions_m = xy_m[ring_indices]
    ring_distances_m = np.sqrt(((ring_positions_m[:, None, :] - ring_positions_m[None, :, :]) ** 2).sum(-1))
    bus_order = np.asarray(cfg.get("array.bus_order") or list(range(N)), dtype=np.int64)
    ring_to_line = np.asarray(cfg.get("channel_plan.ring_to_line") or list(range(N)), dtype=np.int64)
    if sorted(bus_order.tolist()) != list(range(N)):
        raise RealizeError(f"array.bus_order must be a permutation of 0..{N - 1}")
    if ring_to_line.shape != (N,) or np.any(ring_to_line < 0) or np.any(ring_to_line >= L):
        raise RealizeError(f"channel_plan.ring_to_line must hold {N} line indices in 0..{L - 1}")
    bus_spacing_um = cfg.get("optical_path.ring_spacing_on_bus_um")
    return LayoutParams(site_names=tuple(names), site_xy_m=xy_m, site_kind=tuple(kinds),
                        ring_site_index=np.asarray(ring_indices, dtype=np.int64), ring_xy_m=ring_positions_m,
                            ring_distance_m=ring_distances_m,
                        bus_order=bus_order, ring_to_line=ring_to_line,
                        ring_spacing_on_bus_m=float(to_si("optical_path.ring_spacing_on_bus_um",
                            bus_spacing_um)) if bus_spacing_um else 0.0)


# uncertainty

# weakest provenance of `provs` (imported lazily — config stays out of this module's import graph).
def weakest(provs):
    from mrm.params.config import weakest as config_weakest
    return config_weakest(provs)


# {choice — {path — alternative name | value}} of a hypothesis entry (config's helper, imported lazily).
def _hypothesis_choices(e) -> dict:
    from mrm.params.config import _hypothesis_choice_map
    return _hypothesis_choice_map(e)


# (B,) uniforms of one uncertainty entry — its Latin-hypercube column, or counter-RNG draws.
def _draw_uniform(e_name: str, mode: str, seeds: np.ndarray, lhs_cols: dict) -> np.ndarray:
    if mode == "lhs":
        return lhs_cols[e_name]
    return _uniform(seeds, int(rng.Stream.UNCERTAINTY), "uncertainty." + e_name, 1)[:, 0]


# transform `(B,)` uniforms into the entry's requested distribution.
# intervals use uniform, log-uniform, or triangular draws.
# distributions use normal or lognormal draws.
# truncate by mapping uniforms into the allowed normal-CDF interval before inversion.
# for lognormal entries, `mean` is the multiplicative center in `mean * exp(sigma*z)`, which is the median.
def _interval_x(e, u: np.ndarray) -> np.ndarray:
    spec = e.fields
    if e.kind == "interval":
        lo, hi = float(spec["low"]), float(spec["high"])
        distribution = spec.get("dist", "uniform")
        if distribution == "uniform":
            return lo + (hi - lo) * u
        if distribution == "loguniform":
            return np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * u)
        # triangular with the mode at the nominal (or the midpoint)
        peak = float(spec.get("nominal", 0.5 * (lo + hi))) if isinstance(spec.get("nominal"), (int,
            float)) else 0.5 * (lo + hi)
        peak_cdf = (peak - lo) / (hi - lo) if hi > lo else 0.5
        return np.where(u < peak_cdf, lo + np.sqrt(u * (hi - lo) * (peak - lo)),
            hi - np.sqrt((1 - u) * (hi - lo) * (hi - peak)))
    mean, sigma = float(spec["mean"]), float(spec["sigma"])
    lognormal = spec.get("dist", "normal") == "lognormal"
    truncate = spec.get("truncate")
    if truncate:
        # map bounds into normal coordinates before the inverse CDF.
        # for lognormal draws, z = ln(x/mean)/sigma.
        # a nonpositive lower bound gives no lower restriction.
        bounds = np.asarray(truncate, float)
        if lognormal:
            with np.errstate(divide="ignore"):
                normal_bounds = np.log(np.maximum(bounds, 0.0) / mean) / max(sigma, 1e-300)
        else:
            normal_bounds = (bounds - mean) / max(sigma, 1e-300)
        a, b = normal_bounds
        u = ndtr(a) + u * (ndtr(b) - ndtr(a))
    z = ndtri(np.clip(u, 1e-300, 1 - 1e-16))
    if lognormal:
        return mean * np.exp(sigma * z)
    return mean + sigma * z


# return a numeric nominal draw, or the neutral scale or offset.
# without an explicit nominal, use 1 for a targeted scale and 0 for an offset.
# return None for an absolute or untargeted entry so the caller keeps its base value.
def _interval_nominal(e, base_mode: str) -> float | None:
    nominal = e.fields.get("nominal")
    if isinstance(nominal, (int, float)):
        return float(nominal)
    if e.target is None:
        return None
    return {"scale": 1.0, "offset": 0.0}.get(base_mode)


# realize

# build parameter groups and diagnostics for B worlds in SI units.
# `cfg` is a validated config.
# `batch` is B.
# `seeds` accepts an integer, a `(B,)` seed array, or separate device and uncertainty seeds in a mapping.
# `grid` supplies B values per path in config units after uncertainty and before device variation.
# `uncertainty` selects nominal values, independent sampling, or a Latin hypercube.
# `variation` is off or sampled.
# omitted modes use the experiment settings.
# `thermal_hook` can use the plant assembly automatically, supply a compatible module, or be None to retain only the thermal inputs.
# reject invalid timing, non-finite required inputs, relation closure errors, and excessive gain from lagged cross-ring optical heating.
def realize(cfg: "Config", batch: int = 1, seeds: int | np.ndarray | dict = 0, grid: dict | None = None, *,
            uncertainty: str | None = None, variation: str | None = None,
            thermal_hook: Any = "auto") -> PlantParams:
    B = int(batch)
    if B < 1:
        raise RealizeError("batch must be >= 1")
    N, L = cfg.n_rings, cfg.n_lines
    uncertainty_mode = uncertainty or cfg.get("experiment.uncertainty", "sample")
    variation_mode = variation or cfg.get("experiment.variation", "sample")
    if uncertainty_mode not in ("nominal", "sample", "lhs"):
        raise RealizeError(f"uncertainty mode {uncertainty_mode!r} must be nominal, sample or lhs")
    if variation_mode not in ("off", "sample"):
        raise RealizeError(f"variation mode {variation_mode!r} must be off or sample")
    element_seeds = _seed_arrays(seeds, B, cfg)          # {"device": (B,), "uncertainty": (B,)} int64
    layout = _layout(cfg, N, L)
    diagnostics: dict[str, Any] = {"uncertainty_mode": uncertainty_mode, "variation_mode": variation_mode,
                            "defaults_used": [], "clipped": {}}

    # 0. expand config-unit values to world, ring, or line arrays
    leaf_arrays: dict[str, np.ndarray] = {}
    # per-element values of table leaves (a hypothesis may set them per world) - only the geometry Jacobians are read (by _apply_geometry)
    tables: dict[str, list] = {}
    for path, leaf in cfg.leaves.items():
        key = schema.lookup(path)
        if key is None or leaf.is_free:
            continue
        if key.type == "float" and key.level in ("die", "ring", "line"):
            leaf_arrays[path] = _broadcast(leaf.value, key.level, B, N, L, path)
        elif key.type == "table" and path.startswith("ring.geometry.jacobian"):
            tables[path] = [leaf.value] * B
    switches: dict[str, np.ndarray] = {}                 # per-world enum switches as (B, 1) int codes
    for k in schema.SCHEMA.values():
        if k.per_world:
            switches[k.path] = np.full((B, 1), schema.enum_code(k.path, cfg.get(k.path)), dtype=np.int64)
    provenance = {p: lf.provenance for p, lf in cfg.leaves.items()}

    # 1. draw shared model uncertainty once per world
    world: dict[str, np.ndarray] = {}
    world_choices: dict[str, tuple] = {}
    names = sorted(cfg.uncertainty)
    lhs_columns: dict[str, np.ndarray] = {}
    if uncertainty_mode == "lhs" and names:
        from scipy.stats import qmc
        u = qmc.LatinHypercube(d=len(names), seed=int(element_seeds["uncertainty"][0])).random(B)
        lhs_columns = {n: np.clip(u[:, i], 1e-12, 1 - 1e-12) for i, n in enumerate(names)}
    for name in names:
        entry = cfg.uncertainty[name]
        if entry.kind == "hypothesis" and entry.follows is not None:
            continue                        # derive this label from its leader below
        if entry.kind == "hypothesis":
            # each world choice selects leaf alternatives or switch values.
            choice_map = _hypothesis_choices(entry)
            choices = list(choice_map)
            allowed = entry.restrict or choices
            nominal = str(entry.fields.get("nominal"))
            if entry.pinned is not None:
                choice_index = np.full(B, choices.index(str(entry.pinned)), dtype=np.int64)
            elif uncertainty_mode == "nominal":
                nominal_choice_or_value = nominal if nominal in allowed else allowed[0]
                choice_index = np.full(B, choices.index(nominal_choice_or_value), dtype=np.int64)
            else:
                # choice names are strings (config._hypothesis_choice_map) - YAML may give weight keys as numbers (160 - 0.7)
                weight_by_choice = {str(k): v for k, v in (entry.fields.get("weights") or {}).items()}
                weights = np.array([float(weight_by_choice.get(c, 1.0)) if c in allowed else 0.0 for c in choices])
                if weights.sum() <= 0:
                    raise RealizeError(f"uncertainty.{name}: weights sum to zero")
                cdf = np.cumsum(weights / weights.sum())
                u = _draw_uniform(name, uncertainty_mode, element_seeds["uncertainty"], lhs_columns)
                choice_index = np.minimum(np.searchsorted(cdf, u, side="right"),
                                          len(choices) - 1).astype(np.int64)
            world[name] = choice_index
            world_choices[name] = tuple(choices)
            override_paths = set(entry.fields.get("_override_paths", []))
            for choice_number, c in enumerate(choices):
                mask = choice_index == choice_number
                if not mask.any():
                    continue
                for p, a in choice_map[c].items():
                    if p in override_paths:
                        continue
                    key = schema.lookup(p)
                    if key is not None and not key.physical:
                        switches[p][mask, 0] = schema.enum_code(p, a)
                        continue
                    if isinstance(a, str):
                        alternative = cfg.alt_value(p, a)
                        selected_value, alt_provenance = alternative["value"], alternative["provenance"]
                    else:
                        selected_value, alt_provenance = a.get("value"), a.get("provenance",
                                                                    entry.fields.get("provenance", "assumed"))
                    if key.type == "table":
                        for b in np.nonzero(mask)[0]:
                            tables.setdefault(p, [cfg.get(p)] * B)[b] = selected_value
                        continue
                    if p not in leaf_arrays:
                        raise RealizeError(f"uncertainty.{name}: {p} is not a realizable numeric leaf")
                    leaf_arrays[p][mask] = _broadcast(selected_value, key.level, int(mask.sum()), N, L, p)
                    # provenance used to pick the member recomputed in an overdetermined relation
                    provenance[p] = (alt_provenance if mask.all()
                                     else weakest([provenance.get(p, "assumed"), alt_provenance]))
        else:
            # each numeric world draw replaces, scales, or offsets its target.
            mode = entry.fields.get("mode", "absolute")
            if entry.pinned == "override":
                continue
            if entry.pinned == "nominal" or (entry.pinned is None and uncertainty_mode == "nominal"):
                nominal_choice_or_value = _interval_nominal(entry, mode)
                if nominal_choice_or_value is None:   # keep the base value when no absolute nominal is supplied
                    world[name] = (leaf_arrays[entry.target][:, 0].copy() if entry.target in leaf_arrays
                                   else np.full(B, np.nan))
                    continue
                x = np.full(B, nominal_choice_or_value)
            elif entry.pinned is not None:
                x = np.full(B, float(entry.pinned))
            else:
                x = _interval_x(entry, _draw_uniform(name, uncertainty_mode, element_seeds["uncertainty"], lhs_columns))
            world[name] = x
            if entry.target is not None:
                if entry.target not in leaf_arrays:
                    raise RealizeError(f"uncertainty.{name}: target {entry.target} is not a realizable numeric leaf")
                if mode == "absolute":
                    leaf_arrays[entry.target][...] = x[:, None]
                elif mode == "scale":
                    leaf_arrays[entry.target] *= x[:, None]
                else:
                    leaf_arrays[entry.target] += x[:, None]
    # assign labels from the joint hypothesis choices already drawn.
    # for example, the calibration fit couples extinction mechanism and coupling regime.
    for name in names:
        entry = cfg.uncertainty[name]
        if entry.kind != "hypothesis" or entry.follows is None:
            continue
        choices = list(_hypothesis_choices(entry))
        leader_choices = world_choices[entry.follows]
        choice_index = np.array([choices.index(entry.label_of(leader_choices[i])) for i in world[entry.follows]],
                                dtype=np.int64)
        if entry.pinned is not None and np.any(choice_index != choices.index(str(entry.pinned))):
            raise RealizeError(f"uncertainty.{name} is pinned to {entry.pinned!r} but uncertainty.{entry.follows} "
                               f"drew a choice that implies another label")
        world[name] = choice_index
        world_choices[name] = tuple(choices)

    # 2. sweep grid
    for p, si_values in (grid or {}).items():
        if p not in leaf_arrays:
            raise RealizeError(f"grid: {p} is not a realizable numeric leaf")
        v = np.asarray(si_values, dtype=float).reshape(-1)
        if v.shape[0] != B:
            raise RealizeError(f"grid: {p} needs {B} values, got {v.shape[0]}")
        leaf_arrays[p][...] = v[:, None]
        provenance[p] = "override"

    # 3. draw geometry errors, then path-specific device variation
    nominal_arrays = None
    geometry_dln_kappa2: dict[str, np.ndarray] = {}
    if variation_mode == "sample":
        if cfg.geometry:
            # the nominal-geometry device of every world (after the uncertainty and the grid, before any device draw) - its relation solve gives the mask radius, the order and the per-pass couplings the anchors mean
            nominal_arrays = {p: a.copy() for p, a in leaf_arrays.items()}
        geometry_dln_kappa2 = _apply_geometry(cfg, leaf_arrays, tables, element_seeds["device"], layout, B, N,
            diagnostics)
        _apply_variation(cfg, leaf_arrays, element_seeds["device"], layout, B, N, L, diagnostics)

    # 4. convert sampled values to SI
    grid_frequencies_hz, directions = line_frequencies_hz(cfg)
    grid_wavelengths_m = C0 / grid_frequencies_hz
    line_wavelengths_m = np.tile(grid_wavelengths_m[layout.ring_to_line][None, :], (B,
        1))   # (B, N) m - wavelength of each ring's line
    si_values, given = _si_values(cfg, leaf_arrays, line_wavelengths_m)

    # 5. solve the physical relations using the sampled inputs
    latency_scale = world.get("timing.latency_scale")
    latency_scale = np.ones((B, 1)) if latency_scale is None else np.asarray(latency_scale, float).reshape(B, 1)
    relation_options = dict(order2=int(cfg.get("fidelity.thermo_optic_order")) == 2,
                       heater_anchor=str(cfg.get("fidelity.heater_anchor")),
                       tolerance={r: float(d["tolerance_rel"]) for r, d in cfg.relations_cfg.items()
                                  if "tolerance_rel" in d},
                       dt=float(cfg.si("sim.dt_plant_s")), latency_scale=latency_scale)
    ctx = relations.RelationContext(vals=si_values, given=given, prov=provenance, **relation_options)
    if nominal_arrays is not None:
        # the device's own R4 anchor members (after a path-keyed device draw of q_loaded, say) with the nominal geometry - a coupler spread drawn on an anchor stays a coupler spread
        for p in (relations.P_QL, relations.P_QT, relations.P_QD, relations.P_XD):
            if p in leaf_arrays:
                nominal_arrays[p] = leaf_arrays[p]
        nominal_si_values, nominal_given = _si_values(cfg, nominal_arrays, line_wavelengths_m)
        nominal_ctx = relations.solve(relations.RelationContext(vals=nominal_si_values, given=nominal_given,
                                                                prov=dict(provenance), **relation_options))
        nominal_solved = nominal_ctx.vals
        ctx.radius_hold = np.broadcast_to(np.asarray(nominal_solved[relations.P_RADIUS], float), (B, N)).copy()
        ctx.order_hold = np.broadcast_to(np.asarray(nominal_solved[relations.P_ORDER]), (B, N)).copy()
        ctx.coupling_anchor = {
            "kappa2_through": np.broadcast_to(nominal_solved["derived.optics.kappa2_through"], (B, N)).copy(),
            "kappa2_drop": np.broadcast_to(nominal_solved["derived.optics.kappa2_drop"], (B, N)).copy(),
            "alpha_i_per_m": np.broadcast_to(nominal_solved["derived.optics.alpha_i_per_m"], (B, N)).copy(),
            "dln_kappa2_through": geometry_dln_kappa2["derived.geometry.dln_kappa2_through"],
            "dln_kappa2_drop": geometry_dln_kappa2["derived.geometry.dln_kappa2_drop"]}
        for t, a in geometry_dln_kappa2.items():
            si_values[t] = a
            ctx.derivations[t] = relations.Derivation(
                t, "variation.geometry", ("ring.geometry.jacobian", "ring.geometry.jacobian_coupling",
                                          "ring.optics.dln_kappa2_through_dlambda_per_nm"
                                          if t.endswith("through") else "ring.optics.dln_kappa2_drop_dlambda_per_nm"),
                "sum_k J_k dg_k + D d lambda_res: d ln K^2 of the device's coupler at its cold resonance",
                formula_tex=(r"\Delta\ln\kappa_%s^2 = \sum_k J_k\,\Delta g_k + D\,\Delta\lambda_{r0}"
                             % ("t" if t.endswith("through") else "d")))
        for p in ("derived.optics.kappa2_through", "derived.optics.kappa2_drop"):
            si_values[p.replace("derived.optics.", "derived.geometry.nominal_")] = ctx.coupling_anchor[
                p.rsplit(".", 1)[1]]
    _line_derivations(cfg, ctx)
    relations.solve(ctx)
    si_values = ctx.vals
    residuals = relations.residuals(si_values)
    diagnostics["relation_residuals"] = residuals
    diagnostics["relation_notes"] = dict(ctx.notes)

    # 6a. build optics and thermal parameter groups
    # return the SI value or a shaped default and record why the default was used.
    # raise an error when the value is absent and no default is provided.
    def optional(p, shape_like=None, default=None, what="0"):
        if p in si_values:
            return np.asarray(si_values[p], dtype=float)
        if default is None:
            raise RealizeError(f"{p} is required")
        diagnostics["defaults_used"].append(f"{p}: absent -> {what}")
        return np.full(shape_like, default, dtype=float)

    BN, B1, BL = (B, N), (B, 1), (B, L)                   # per-ring, per-die and per-line shapes

    # `x` as a new (B, N) float array.
    def per_ring(x):
        return np.broadcast_to(np.asarray(x, float), BN).copy()

    # `x` as a new (B, 1) float array.
    def per_die(x):
        x = np.asarray(x, float)
        return np.broadcast_to(x.reshape(x.shape[0], 1) if x.ndim == 2 and x.shape[1] == 1 else x, B1).copy()

    order2 = ctx.order2
    s0 = per_ring(si_values[relations.P_S0])
    beta = per_die(si_values["derived.thermal.beta_eff_per_k"])
    t0 = per_die(si_values[relations.P_T0])
    q_l = per_ring(si_values[relations.P_QL])
    lam_res = per_ring(si_values["derived.optics.lambda_res_cold_m"])
    optics = OpticsParams(
        lambda_line_m=per_ring(line_wavelengths_m), lambda_res_cold_m=lam_res,
        resonance_offset_m=per_ring(si_values[relations.P_OFFSET]),
        t_ref_k=per_die(si_values[relations.P_OFFSET_T]),
        bias_ref_v=per_die(optional("ring.optics.resonance_offset_ref_bias_v", B1, 0.0, "0 V")),
        radius_m=per_ring(si_values[relations.P_RADIUS]), length_m=per_ring(si_values["derived.optics.length_m"]),
        n_eff=per_ring(si_values["derived.optics.n_eff_phase"]), n_eff_configured=per_ring(si_values[relations.P_NEFF]),
        n_g=per_ring(si_values[relations.P_NG]),
        dn_g_dlambda_per_m=per_ring(optional("ring.optics.dn_g_dlambda_per_nm", BN, 0.0)),
        order=np.broadcast_to(si_values[relations.P_ORDER], BN).astype(np.int64).copy(),
        fsr_hz=per_ring(si_values["derived.optics.fsr_hz"]),
        round_trip_time_s=per_ring(si_values["derived.optics.round_trip_time_s"]), q_loaded=q_l,
        q_intrinsic=per_ring(si_values[relations.P_QI]), q_through=per_ring(si_values[relations.P_QT]),
        q_drop=per_ring(si_values[relations.P_QD]),
        kappa2_through=per_ring(si_values["derived.optics.kappa2_through"]),
        kappa2_drop=per_ring(si_values["derived.optics.kappa2_drop"]),
        round_trip_amplitude=per_ring(si_values["derived.optics.round_trip_amplitude"]),
        alpha_i_per_m=per_ring(si_values["derived.optics.alpha_i_per_m"]),
        alpha_heat_per_m=per_ring(si_values["derived.optics.alpha_heat_per_m"]),
        heat_fraction=per_ring(si_values[relations.P_ETA]) if relations.P_ETA in si_values else per_ring(
            si_values["derived.optics.alpha_heat_per_m"] / si_values["derived.optics.alpha_i_per_m"]),
        dln_kappa2_through_dlambda_per_m=per_ring(optional("ring.optics.dln_kappa2_through_dlambda_per_nm", BN,
                                                           0.0)),
        dln_kappa2_drop_dlambda_per_m=per_ring(optional("ring.optics.dln_kappa2_drop_dlambda_per_nm", BN, 0.0)),
        dln_alpha_dt_ring_per_k=per_ring(optional("ring.optics.dln_alpha_dt_ring_per_k", BN, 0.0)),
        dln_alpha_dt_heater_per_k=per_ring(optional("ring.optics.dln_alpha_dt_heater_per_k", BN, 0.0)),
        dn_eff_dt_per_k=per_ring(si_values[relations.P_DNDT]),
        d2n_eff_dt2_per_k2=per_ring(si_values["derived.optics.d2n_eff_dt2_per_k2"]),
        alpha_l_per_k=per_die(si_values[relations.P_ALPHA_L]),
        alpha_l_slope_per_k2=per_die(optional(relations.P_ALPHA_L_SLOPE, B1, 0.0)) if order2 else np.zeros(B1),
        backscatter_splitting_hz=per_ring(optional("ring.optics.backscatter_splitting_ghz", BN, 0.0, "0 (off)"))
        if cfg.get("fidelity.backscatter") == "on" else np.zeros(BN),
        fwhm_hz=C0 / (lam_res * q_l),
        alpha_fca_per_m=per_ring(si_values[relations.P_A_FCA]) if relations.P_A_FCA in si_values else np.full(BN,
            np.nan),
        backscatter_rb=_backscatter_rb(cfg, si_values, element_seeds["device"], variation_mode, B, N),
        tcmt_orders=int(cfg.get("ring.optics.tcmt_orders")),
    )
    s_lock = per_ring(si_values["derived.thermal.s_lock_m_per_k"])     # m/K - dlambda/dT at the lock temperature
    z_h = per_ring(si_values[relations.P_Z])                           # K/W - Z_heater->ring(0)
    z_ratio = per_die(si_values["ring.thermal.z_ratio_opt_heater"])
    z_opt = z_h * z_ratio                                         # K/W - Z_opt->ring(0)
    thermal_core = dict(
        dlambda_dt_s0_m_per_k=s0, dlambda_dt_beta_per_k=beta, dlambda_dt_t0_k=t0,
        t_ring_lock_k=per_ring(si_values[relations.P_TLOCK]), s_lock_m_per_k=s_lock,
        heater_efficiency_m_per_w=per_ring(si_values[relations.P_EFF]), z_heater_ring_k_per_w=z_h,
        z_ratio_opt_heater=z_ratio, z_opt_ring_k_per_w=z_opt, self_heating_gain_m_per_w=s_lock * z_opt)

    # 6b. thermal inputs and the assembly hook (owned by mrm.plant.thermal)
    thermal_model = str(cfg.get("fidelity.thermal_model"))
    interim = {}
    for p in schema.SCHEMA:
        if p.startswith("ring.thermal.") and (p.endswith("_s") or p.endswith("_frac") or p.endswith("_um")):
            if p in si_values and schema.SCHEMA[p].level != "ring":
                nm = p.rsplit(".", 1)[1]
                interim[nm[:-3] + "_m" if nm.endswith("_um") else nm] = per_die(si_values[p])   # SI names (um -> m)
    # untargeted interval/dist draws under uncertainty.thermal.* are ThermalScales factors - hypotheses there (for example thermal.family and thermal.stack) act through their per-world switches and leaves, not as scales
    scales = {n[len("thermal."):]: np.asarray(v) for n, v in world.items() if n.startswith("thermal.")
              and cfg.uncertainty[n].target is None and cfg.uncertainty[n].kind != "hypothesis"}
    family_key = schema.SCHEMA["ring.thermal.family"]
    stack_key = schema.SCHEMA["ring.thermal.stack"]
    scaling_key = schema.SCHEMA["ring.thermal.scaling_hypothesis"]
    rom_path = None
    if thermal_model == "rom":
        rom_path = str((cfg.data_root / cfg.get("ring.thermal.rom_file")).resolve())
    sites = dict(cfg.get("layout.sites") or {})
    sensor_records = list(cfg.get("temp_sensors") or [])
    thermal_inputs = ThermalInputs(
        model=thermal_model, rom_path=rom_path, interim=interim, z_heater_ring_k_per_w=_freeze(z_h),
        z_opt_ring_k_per_w=_freeze(z_opt), dlambda_dt_s0_m_per_k=_freeze(s0),
        dlambda_dt_beta_per_k=_freeze(beta), dlambda_dt_t0_k=_freeze(t0),
        family=_freeze(switches["ring.thermal.family"]), family_names=family_key.enum,
        stack=_freeze(switches["ring.thermal.stack"]), stack_names=stack_key.enum, scales=scales,
        dt_s=float(cfg.si("sim.dt_plant_s")),
        scaling_hypothesis=tuple(scaling_key.enum[int(c)] for c in switches[scaling_key.path][:, 0]),
        sites_um=sites, temp_sensors=sensor_records, ring_pitch_um=cfg.get("layout.ring_pitch_um"))
    system = None
    thermal_module = None
    if thermal_hook == "auto":
        try:
            from mrm.plant import thermal as thermal_module  # optional at import
        except ImportError:
            thermal_module = None
        if thermal_module is not None and not hasattr(thermal_module, "assemble"):
            thermal_module = None
    elif thermal_hook is not None:
        thermal_module = thermal_hook            # a module-like object exposing the same API (tests)
    if thermal_module is not None:
        system, z_opt_rom = _assemble_thermal(thermal_module, cfg, thermal_inputs, si_values, N, diagnostics)
        if z_opt_rom is not None:               # ROM model - Z_opt comes from the FEM, not from z_ratio
            z_opt = z_opt_rom
            thermal_core["z_opt_ring_k_per_w"] = z_opt
            thermal_core["z_ratio_opt_heater"] = per_die(np.mean(z_opt / z_h, axis=1, keepdims=True))
            thermal_core["self_heating_gain_m_per_w"] = s_lock * z_opt
    diagnostics["thermal_hook"] = getattr(thermal_module, "__name__", None) if thermal_module is not None else None
    thermal = ThermalParams(**thermal_core, system=system, inputs=thermal_inputs)

    # 6c. remaining groups
    junction = _junction(cfg, si_values, optional, per_ring, per_die, BN, B1)
    heater = _heater(cfg, si_values, switches, optional, per_ring, per_die, BN, B1, element_seeds["device"])
    timing = _timing(cfg, si_values, latency_scale, diagnostics)
    bias = BiasDacParams(
        bits=int(cfg.get("ring.bias_dac.bits")), v_min_v=per_die(si_values["ring.bias_dac.v_min_v"]),
        v_max_v=per_die(si_values["ring.bias_dac.v_max_v"]),
        update_period_ticks=_period_ticks(cfg, "ring.bias_dac.update_rate_hz", default=1),
        slew_v_per_s=per_die(optional("ring.bias_dac.slew_v_per_s", B1, np.inf, "inf (unlimited)")),
        settle_tau_s=per_die(optional("ring.bias_dac.settle_tau_s", B1, 0.0, "0 (no filter)")),
        noise_rms_codes=per_die(optional("ring.bias_dac.noise_rms_codes", B1, 0.0)))
    if np.any(bias.v_max_v <= bias.v_min_v):
        raise RealizeError("ring.bias_dac: v_max_v must exceed v_min_v")
    monitor = _monitor(cfg, si_values, timing, optional, per_ring, per_die, BN, B1)
    temp_sensors = _temp_sensors(cfg, layout, element_seeds["device"], latency_scale, B, timing.dt_s)
    datapath_default = "nan (mrm.plant.datapath default)"
    datapath = DatapathParams(
        bitrate_hz=per_die(optional("datapath.bitrate_gbps", B1, np.nan, "nan (absent)")),
        bitstat_window_bits=int(cfg.get("datapath.bitstat_window_bits") or 0),
        error_count_period_ticks=_period_ticks(cfg, "datapath.error_count_period_s", default=0, rate=False),
        transition_density=per_die(optional("datapath.transition_density", B1, np.nan, "nan (absent)")),
        driver_output_resistance_ohm=per_die(optional("driver.output_resistance_ohm", B1, np.nan, "nan (absent)")),
        driver_static_power_w=per_die(optional("driver.static_power_mw", B1, 0.0)),
        driver_energy_per_transition_j=per_die(optional("driver.energy_per_transition_fj", B1, 0.0)),
        swing_pp_v=per_ring(optional("driver.swing_pp_v", BN, np.nan, "nan (absent)")),
        level_convention=switches["driver.level_convention"].copy(),
        rx_sensitivity_oma_w=per_ring(optional("datapath.rx_sensitivity_oma_dbm", BN, np.nan, datapath_default)),
        ber_ref=per_die(optional("datapath.ber_ref", B1, np.nan, datapath_default)),
        link_transmission=per_die(np.exp(-optional("datapath.link_loss_db", B1, np.nan, datapath_default))),
        ber_floor=per_die(optional("datapath.ber_floor", B1, np.nan, datapath_default)),
        bitstat_bw_hz=per_die(optional("datapath.bitstat_bw_ghz", B1, np.nan, datapath_default)),
        bitstat_noise_a_per_rt_hz=per_die(optional("datapath.bitstat_noise_pa_per_rt_hz", B1, np.nan,
                                                   "nan (the monitor TIA density)")))
    line_power_w = np.broadcast_to(np.asarray(si_values["laser.power_per_line_dbm"], float),
        BL).copy()   # W (SI of dBm)
    actual_frequencies_hz = grid_frequencies_hz[None,
        :] + np.broadcast_to(np.asarray(optional("laser.line_offset_ghz", BL, 0.0), float), BL)
    laser = LaserParams(
        line_freq_hz=actual_frequencies_hz, line_freq_grid_hz=grid_frequencies_hz, power_w=line_power_w,
        input_transmission=per_die(np.exp(-np.asarray(si_values["optical_path.chip_input_loss_db"], float))),
        bus_loss_per_m=per_die(optional("optical_path.bus_loss_db_per_mm", B1, 0.0)),
        rin_per_hz=per_die(optional("laser.rin_db_per_hz", B1, 0.0)),
        linewidth_hz=per_die(optional("laser.linewidth_mhz", B1, 0.0)),
        direction=directions, **_line_meta(cfg, L))
    flags = FidelityFlags(
        self_heating=int(cfg.get("fidelity.self_heating") == "on"),
        thermo_optic_order=int(cfg.get("fidelity.thermo_optic_order")),
        thermal_model=int(thermal_model == "rom"), junction_model=int(cfg.get("fidelity.junction_model") == "lut"),
        optics_model=0,
        backscatter=int(cfg.get("fidelity.backscatter") == "on"),
        shot_noise=int(cfg.get("fidelity.shot_noise") == "on"),
        pattern_noise=schema.enum_code("fidelity.pattern_noise", cfg.get("fidelity.pattern_noise")),
        data_modulation=int(cfg.get("fidelity.data_modulation") == "lut"),
        heater_anchor=int(cfg.get("fidelity.heater_anchor") == "fsr_integral"),
        heater_pwm_ripple=int(cfg.get("fidelity.heater_pwm_ripple") == "on"))

    # 7a. check required finite inputs and each driver law's full scale
    _assert_finite(optics, thermal, heater, laser, junction)

    # 7b. check the numerical cross-ring lag and physical relation closure
    lag = _selfheat_lag_gain(cfg, optics, thermal, laser, layout, timing.dt_s, thermal_model, system, thermal_module)
    diagnostics["selfheat_lag_gain"] = lag
    if flags.self_heating and lag is not None and not lag < SELFHEAT_LAG_GAIN_MAX:
        raise RealizeError(f"self-heating: the lagged cross-ring optical heating has a per-substep gain of {lag:.3g} "
                           f">= {SELFHEAT_LAG_GAIN_MAX} at dt = {timing.dt_s:g} s; reduce sim.dt_plant_s")
    bad = {k: v for k, v in residuals.items() if not v <= 1e-12}      # nan (a non-finite input) fails too
    if bad:
        raise RealizeError(f"relations do not close to 1e-12 after realize(): {bad}")

    frozen_world = {k: _freeze(np.asarray(v)) for k, v in world.items()}
    frozen_values = {k: _freeze(np.asarray(v)) if isinstance(v, np.ndarray) else v for k, v in si_values.items()}
    frozen_values.update({f"switch.{k}": _freeze(v) for k, v in switches.items()})
    return PlantParams(
        batch=B, n_rings=N, n_lines=L, optics=optics, thermal=thermal, junction=junction, heater=heater,
        bias_dac=bias, monitor=monitor, temp_sensors=temp_sensors, datapath=datapath, laser=laser, layout=layout,
        timing=timing, flags=flags, world=frozen_world, world_choices=world_choices,
        seeds={k: _freeze(v) for k, v in element_seeds.items()}, config_hash=cfg.hash(), values=frozen_values,
        derivations=dict(ctx.derivations), diagnostics=diagnostics, thermal_inputs=thermal_inputs)


# fields that may legitimately hold nan (absent optional inputs, or not used by the selected model)
_NAN_ALLOWED = {
    "optics": {"alpha_fca_per_m"},
    "junction": {"v_bi_v", "depletion_exponent", "depletion_a_m", "loss_ratio_q"},
    "heater": {"full_scale_v", "full_scale_a", "full_scale_w", "pwm_period_s"},
}
# required full-scale field by driver code — voltage, current, power, DAC PWM
_DRIVER_LAW_FULL_SCALE = {0: "full_scale_v", 1: "full_scale_a", 2: "full_scale_w", 3: "full_scale_v"}


# reject non-finite required values and missing heater-law full scales.
# allow documented placeholders for optional or unused fields.
# check only the full-scale field required by each world's selected heater driver law.
def _assert_finite(optics: OpticsParams, thermal: ThermalParams, heater: HeaterParams, laser: LaserParams,
                   junction: JunctionParams) -> None:
    bad = []
    for group_name, group in (("optics", optics), ("thermal", thermal), ("laser", laser), ("junction", junction)):
        for field_info in fields(group):
            value = getattr(group, field_info.name)
            if (isinstance(value, np.ndarray) and value.dtype.kind in "fc" and field_info.name not in _NAN_ALLOWED.get(group_name, ())
                    and not np.all(np.isfinite(value))):
                bad.append(f"{group_name}.{field_info.name}")
    for name in ("p_fsr_w", "r_cold_ohm", "r_ref_temp_k", "tcr_per_k"):
        if not np.all(np.isfinite(getattr(heater, name))):
            bad.append(f"heater.{name}")
    for code, full_scale_field in _DRIVER_LAW_FULL_SCALE.items():
        rows = heater.driver_mode[:, 0] == code
        if rows.any() and not np.all(np.isfinite(getattr(heater, full_scale_field)[rows])):
            bad.append(f"heater.{full_scale_field} (needed by driver_mode code {code} in {int(rows.sum())} worlds)")
    if bad:
        raise RealizeError(f"non-finite values after realize(): {', '.join(bad)}; check the inputs of these "
                           f"quantities (an uncertainty draw, a hypothesis choice or a missing key)")


# convert sampled config values to SI and identify known relation members.
# include scalar global values, assigned line wavelengths, and cold resonances.
# the cold resonance is line wavelength minus the sampled offset.
def _si_values(cfg: "Config", leaf_arrays: dict, lam_line: np.ndarray) -> tuple[dict, set]:
    si_values: dict[str, Any] = {}
    for p, a in leaf_arrays.items():
        si_values[p] = to_si(p, a)
    for path, leaf in cfg.leaves.items():
        if path in si_values or leaf.is_free:
            continue
        key = schema.lookup(path)
        if key is not None and key.type in ("float", "int") and key.level == "global" and not isinstance(
                leaf.value, list):
            si_values[path] = to_si(path, leaf.value)
    si_values["derived.optics.lambda_line_m"] = lam_line
    si_values["derived.optics.lambda_res_cold_m"] = lam_line - si_values[relations.P_OFFSET]
    given = {p for p in si_values if p in cfg.leaves and not cfg.leaves[p].is_free}
    given |= {"derived.optics.lambda_line_m", "derived.optics.lambda_res_cold_m"}
    return si_values, given


# the channel-plan leaves the line wavelengths come from (a derived value carries the weakest provenance of its inputs).
# channel_plan.ring_to_line is left out on purpose - it is the design convention every per-ring value is stated against (each ring is designed relative to its own line, ring.optics.resonance_offset_nm), and it only permutes the plan's wavelengths over the rings
LINE_PLAN_INPUTS = ("channel_plan.lines", "channel_plan.center_thz", "channel_plan.spacing_ghz", "channel_plan.n_lines")


# record how the line and cold-resonance wavelengths are made (`mrm params explain`, provenance).
def _line_derivations(cfg: "Config", ctx: relations.RelationContext) -> None:
    inputs = ("channel_plan.lines",) if cfg.get("channel_plan.lines") else LINE_PLAN_INPUTS[1:]
    ctx.derivations["derived.optics.lambda_line_m"] = relations.Derivation(
        "derived.optics.lambda_line_m", "channel_plan", inputs,
        "lambda_line = c/f of the ring's line (channel_plan.ring_to_line); f = center + (k - (L - 1)/2) spacing, or "
        "the channel_plan.lines table",
        formula_tex=r"\lambda_L = \frac{c}{f_L},\quad f_L = f_{\mathrm{center}} + "
                    r"\left(k - \frac{N_{\mathrm{lines}} - 1}{2}\right)\Delta f_{\mathrm{ch}}"
                    r"\ \text{(or the channel\_plan.lines table)}", eq="1.7")
    ctx.derivations["derived.optics.lambda_res_cold_m"] = relations.Derivation(
        "derived.optics.lambda_res_cold_m", "channel_plan", ("derived.optics.lambda_line_m", relations.P_OFFSET),
        "lambda_res = lambda_line - resonance_offset (the cold resonance)",
        formula_tex=r"\lambda_{r0} = \lambda_L - \delta_0", eq="0.8")


# variation

# return a factor that turns independent ring draws into correlated draws.
# use `correlation = exp(-distance / correlation_length)` with both distances in um.
# add small diagonal jitter for the factorization.
# return None when no correlation length is set.
# apply the factor with `z @ chol.T`.
def _correlation_cholesky(layout: LayoutParams, corr_um) -> np.ndarray | None:
    if not corr_um:
        return None
    distances_um = layout.ring_distance_m * 1e6
    correlation = np.exp(-distances_um / float(corr_um))
    return np.linalg.cholesky(correlation + 1e-12 * np.eye(correlation.shape[0]))


# return log-sensitivity per geometry input for an exponential output.
# prefer stored positive plus/minus evaluations and compute their central log-difference.
# this is exact for an exponential geometry law.
# a linear secant can overstate bend-radiation sensitivity by 4.8 times for rib160 slab errors.
# without suitable evaluations, use the first-order slope `J / leaf`.
# zero coefficients stay zero.
# output rows broadcast to the target leaf.
def _jacobian_log_derivative(jac: dict, out: str, coeffs, dims_in, leaf: np.ndarray) -> np.ndarray:
    evaluations = jac.get("evaluations") if isinstance(jac.get("evaluations"), dict) else {}
    name = _JACOBIAN_EVAL_NAME.get(out)
    leaf = np.asarray(leaf, float)
    rows = []
    for coefficient, input_name in zip(coeffs, dims_in):
        coefficient = float(coefficient)
        evaluation = evaluations.get(input_name) if name else None
        plus_value = minus_value = None
        if isinstance(evaluation, dict) and isinstance(evaluation.get("plus"),
            dict) and isinstance(evaluation.get("minus"), dict):
            plus_value, minus_value, step_nm = evaluation["plus"].get(name), evaluation["minus"].get(name), evaluation.get("step_nm")
        if coefficient == 0.0:
            rows.append(np.zeros_like(leaf))
        elif plus_value is not None and minus_value is not None and step_nm and float(plus_value) > 0.0 and float(minus_value) > 0.0:
            rows.append(np.full_like(leaf, (math.log(float(plus_value)) - math.log(float(minus_value))) / (2.0 * float(step_nm))))
        else:
            if np.any(leaf <= 0.0):
                raise RealizeError(f"Jacobian output {out}: the leaf must be > 0 for the exponential law")
            rows.append(coefficient / leaf)
    return np.stack(rows)


# draw geometry errors in nm and apply their optical Jacobians in place.
# combine one shared die error with local ring errors, optionally correlated by distance.
# widening all waveguides can also close both coupling gaps.
# group worlds that share the same Jacobian table before applying its coefficients.
# add index and wavelength shifts.
# apply loss changes multiplicatively to keep exponential laws positive.
# return `(B, N)` changes in log coupling fraction for the Q solver, including dispersion at the shifted cold resonance.
# return an empty mapping when geometry variation is absent.
def _apply_geometry(cfg, leaf_arrays, tables, seeds, layout, B, N, diag) -> dict:
    geometry = cfg.geometry
    if not geometry:
        return {}
    geometry_error_nm: dict[str, np.ndarray] = {}
    for dim in _GEOMETRY_DIM_ORDER:
        spec = geometry.get(dim)
        if not spec:
            continue
        name = f"variation.geometry.{dim}"
        die_normal = _normal(seeds, int(rng.Stream.VARIATION), name, 1, slot=0)
        ring_normal = _normal(seeds, int(rng.Stream.VARIATION), name, N, slot=1)
        chol = _correlation_cholesky(layout, spec.get("corr_length_um"))
        if chol is not None:
            ring_normal = ring_normal @ chol.T
        geometry_error_nm[dim] = float(spec.get("die_sigma_nm", 0.0)) * die_normal + float(spec.get("ring_sigma_nm",
            0.0)) * ring_normal
    diag["geometry_draws_nm"] = {k: v.copy() for k, v in geometry_error_nm.items()}
    if geometry.get("width_closes_gaps") and "width" in geometry_error_nm:
        # coupler Jacobians hold edge-to-edge gaps fixed.
        # a common waveguide width error closes each gap by the same amount.
        # apply the total derivative d/dw-d/dgap, then add independent gap errors.
        for gap_dimension in ("gap_through", "gap_drop"):
            geometry_error_nm[gap_dimension] = geometry_error_nm.get(gap_dimension, 0.0) - geometry_error_nm["width"]
    dln_kappa2 = {t: np.zeros((B, N)) for t in _KAPPA2_DISPERSION}
    dlambda_res_nm = np.zeros((B, N))                       # geometry move of the cold resonance (nm)
    jacobian_paths = geometry.get("jacobians", ["ring.geometry.jacobian"])
    if isinstance(jacobian_paths, str):
        jacobian_paths = [jacobian_paths]
    for jacobian_path in jacobian_paths:
        per_world = tables.get(jacobian_path) or [cfg.get(jacobian_path)] * B
        groups: dict[int, list[int]] = {}
        tables_by_id: dict[int, Any] = {}
        for b, t in enumerate(per_world):
            groups.setdefault(id(t), []).append(b)
            tables_by_id[id(t)] = t
        for table_id, world_indices in groups.items():
            jac = tables_by_id[table_id]
            if not isinstance(jac, dict) or "d" not in jac or "of" not in jac:
                raise RealizeError(f"{jacobian_path}: a Jacobian is {{d: [inputs], of: {{output: [one per input]}}}}")
            dims = []
            for input_name in jac["d"]:
                if input_name not in _JACOBIAN_INPUT_DIMS:
                    raise RealizeError(f"{jacobian_path}: unknown Jacobian input {input_name!r}; "
                                       f"known {sorted(_JACOBIAN_INPUT_DIMS)}")
                dims.append(_JACOBIAN_INPUT_DIMS[input_name])
            world_selection = np.asarray(world_indices)
            for out, coeffs in jac["of"].items():
                if out not in JACOBIAN_TARGETS:
                    raise RealizeError(f"{jacobian_path}: unknown Jacobian output {out!r}; "
                                       f"known {sorted(JACOBIAN_TARGETS)}")
                if len(coeffs) != len(dims):
                    raise RealizeError(f"{jacobian_path}.of.{out}: {len(coeffs)} coefficients for {len(dims)} inputs")
                target, application, sign = JACOBIAN_TARGETS[out]
                delta = np.zeros((len(world_indices), N))
                for c, dim in zip(coeffs, dims):
                    if dim in geometry_error_nm and c:
                        delta += float(c) * np.broadcast_to(geometry_error_nm[dim], (B, N))[world_selection]
                if not np.any(delta):
                    continue
                if application == "kappa2":
                    dln_kappa2[target][world_selection] += sign * delta
                    continue
                if out in ("lambda_res_nm_per_nm", "resonance_offset_nm_per_nm"):
                    dlambda_res_nm[world_selection] -= sign * delta    # offset = line - resonance
                if target not in leaf_arrays:
                    raise RealizeError(f"{jacobian_path}.of.{out}: target {target} is not a given input of this config "
                                       f"(it is free or absent); the geometry cannot move it")
                if leaf_arrays[target].shape[1] != N:
                    leaf_arrays[target] = np.broadcast_to(leaf_arrays[target], (B, N)).copy()
                if application == "offset":
                    leaf_arrays[target][world_selection] += sign * delta
                elif application == "lnscale":
                    leaf_arrays[target][world_selection] *= np.exp(sign * delta)
                else:                                 # 'exp' - the secant as a log-derivative (positive values)
                    log_derivatives = _jacobian_log_derivative(jac, out, coeffs, jac["d"],
                        leaf_arrays[target][world_selection])
                    log_change = np.zeros((len(world_indices), N))
                    for k, dim in enumerate(dims):
                        if dim in geometry_error_nm:
                            log_change += log_derivatives[k] * np.broadcast_to(geometry_error_nm[dim], (B,
                                N))[world_selection]
                    leaf_arrays[target][world_selection] *= np.exp(sign * log_change)
    coupling_changes = {}
    for t, disp in _KAPPA2_DISPERSION.items():
        # anchors hold at each device's cold resonance, after the geometry shift dlambda_res_nm.
        # coupler dispersion also changes K^2 at that shifted resonance.
        # phase.py evaluates K^2(lambda_line) = K^2(lambda_res)*exp(D*(lambda_line-lambda_res)).
        # this applies dispersion once at the laser wavelength.
        dispersion_per_nm = np.broadcast_to(np.asarray(leaf_arrays.get(disp, 0.0), float), (B, N))
        coupling_changes[t] = dln_kappa2[t] + dispersion_per_nm * dlambda_res_nm
    diag["geometry_dlambda_res_nm"] = dlambda_res_nm.copy()
    return coupling_changes


# apply path-specific device variation to config-unit arrays in place.
# draw a shared die component and a ring or line component from separate slots.
# offset adds the draw.
# scale multiplies by an exponential.
# `rel_sigma` uses a relative local spread.
# ring draws can be correlated by layout distance.
# when truncation is enabled, a supported single normal offset component uses exact inverse-CDF truncation.
# other cases clip to the leaf range and record the clipped fraction in diagnostics.
def _apply_variation(cfg, leaf_arrays, seeds, layout, B, N, L, diag) -> None:
    for path in sorted(cfg.variation):
        entry = cfg.variation[path].fields
        key = schema.lookup(path)
        level = entry.get("level", key.level)
        dist = entry.get("dist", "normal")
        mode = entry.get("mode", "scale" if (dist == "lognormal" or "rel_sigma" in entry) else "offset")
        name = f"variation.{path}"
        base = leaf_arrays[path]
        if level == "ring" and base.shape[1] != N:
            base = np.broadcast_to(base, (B, N)).copy()
        if level == "line" and base.shape[1] != L:
            base = np.broadcast_to(base, (B, L)).copy()
        k = {"die": 1, "ring": N, "line": L}[level]
        sigma_die = float(entry.get("die_sigma", 0.0))
        sigma_element = float(entry.get("ring_sigma", entry.get("line_sigma", 0.0)))
        rel_sigma = float(entry.get("rel_sigma", 0.0))
        stream = int(rng.Stream.VARIATION)
        range_lo, range_hi = (cfg.leaves[path].range or (None, None)) if entry.get("truncate", True) else (None, None)
        # exact truncation is available for one spread component without ring correlation.
        single_component = ((sigma_die == 0.0) + (sigma_element == 0.0 and rel_sigma == 0.0) == 1
                  and not entry.get("corr_length_um"))
        if dist == "uniform":
            die_draw = math.sqrt(3.0) * (2.0 * _uniform(seeds, stream, name, 1, 0) - 1.0)
            element_draw = math.sqrt(3.0) * (2.0 * _uniform(seeds, stream, name, k, 1) - 1.0)
        else:
            die_uniform = _uniform(seeds, stream, name, 1, 0)
            element_uniform = _uniform(seeds, stream, name, k, 1)
            if (single_component and mode == "offset" and range_lo is not None and dist == "normal"
                    and (sigma_die == 0.0 or level == "die")):
                # exact truncation of a single normal component to the leaf range (inverse CDF)
                sigma = sigma_element if sigma_element else sigma_die
                u = element_uniform
                a = (range_lo - base) / sigma
                b = (range_hi - base) / sigma
                u = ndtr(a) + u * (ndtr(b) - ndtr(a))
                leaf_arrays[path] = base + sigma * ndtri(np.clip(u, 1e-300, 1 - 1e-16))
                continue
            die_draw, element_draw = ndtri(die_uniform), ndtri(element_uniform)
        if level == "ring":
            chol = _correlation_cholesky(layout, entry.get("corr_length_um"))
            if chol is not None:
                element_draw = element_draw @ chol.T
        if rel_sigma:
            new = base * (1.0 + rel_sigma * element_draw) * (np.exp(sigma_die * die_draw) if sigma_die else 1.0)
        else:
            delta = sigma_die * die_draw + sigma_element * element_draw
            if mode == "offset":
                new = base + delta
            else:
                new = base * np.exp(delta)
        if range_lo is not None:
            clipped = np.clip(new, range_lo, range_hi)
            clipped_count = int(np.count_nonzero(clipped != new))
            if clipped_count:
                diag["clipped"][path] = clipped_count / new.size
            new = clipped
        leaf_arrays[path] = new


# group builders

# return complex backscatter coefficients shaped `(B, N, orders)`.
# sample independent real and imaginary normals for each device and order.
# normalize so `E[abs(beta)**2] = (splitting/2)**2` and set `rb = 2*pi*beta/FSR`.
# with variation off, beta is real and equals half the splitting.
# return zeros when backscatter is off.
def _backscatter_rb(cfg, vals, seeds, variation_mode, B: int, N: int) -> np.ndarray:
    n_orders = int(cfg.get("ring.optics.backscatter_orders"))
    out = np.zeros((B, N, n_orders), dtype=np.complex128)
    if cfg.get("fidelity.backscatter") != "on" or "ring.optics.backscatter_splitting_ghz" not in vals:
        return out
    half_splitting_hz = np.broadcast_to(np.asarray(vals["ring.optics.backscatter_splitting_ghz"], float), (B, N)) / 2.0
    fsr = np.broadcast_to(np.asarray(vals["derived.optics.fsr_hz"], float), (B, N))
    if variation_mode == "off":
        beta = np.repeat(half_splitting_hz[:, :, None], n_orders, axis=2).astype(np.complex128)
    else:
        stream = int(rng.Stream.VARIATION)
        real_normal = _normal(seeds, stream, "ring.optics.backscatter", N * n_orders, 0).reshape(B, N, n_orders)
        imag_normal = _normal(seeds, stream, "ring.optics.backscatter", N * n_orders, 1).reshape(B, N, n_orders)
        beta = half_splitting_hz[:, :, None] * (real_normal + 1j * imag_normal) / math.sqrt(2.0)
    out[...] = 2.0 * math.pi * beta / fsr[:, :, None]
    return _freeze(out)


# whole plant ticks of a period key, or of the period 1/rate of a rate key (`default` when absent or <= 0).
def _period_ticks(cfg, path: str, default: int, rate: bool = True) -> int:
    value = cfg.get(path)
    if value is None:
        return default
    dt = float(cfg.si("sim.dt_plant_s"))
    si_value = float(to_si(path, value))
    if si_value <= 0:
        return default
    ticks = (1.0 / si_value if rate else si_value) / dt
    return int(round(ticks))


# build the analytic junction parameters or load the junction table.
# for a table, derive the report-only slope at zero bias from its index gradient.
# mark unused analytic parameters with NaN.
# use the caller's shaping and default helpers so both model choices have the same parameter-group interface.
def _junction(cfg, vals, optional, per_ring, per_die, BN, B1) -> JunctionParams:
    use_lut = cfg.get("fidelity.junction_model") == "lut"
    empty = np.zeros(0)
    fill_fraction = per_ring(vals["ring.junction.fill_fraction"])
    if use_lut:
        from mrm.params.config import resolve_from
        data, _ = resolve_from(f"{cfg.get('ring.junction.lut_file')}:v_reverse_v", None, cfg.data_root)
        grid = np.asarray(data, float)

        def col(k):
            d, _ = resolve_from(f"{cfg.get('ring.junction.lut_file')}:{k}", None, cfg.data_root)
            return np.asarray(d, float)
        lut_dn, lut_a, lut_c, lut_r = col("dneff"), col("alpha_fca_per_m"), col("capacitance_f_per_m"), col(
            "r_series_ohm_m")
        # derive the report-only wavelength slope at zero bias from the lookup table.
        index_slope_at_zero = np.interp(0.0, grid, np.gradient(lut_dn, grid))
        wavelength_m = vals["derived.optics.lambda_res_cold_m"]
        eff_at_0v = per_ring(fill_fraction * wavelength_m * index_slope_at_zero / vals[relations.P_NG])
        v_bi = np.full(B1, np.nan)
        depletion_exponent = np.full(B1, np.nan)
        depletion_a_m = np.full(BN, np.nan)
    else:
        grid = lut_dn = lut_a = lut_c = lut_r = empty
        eff_at_0v = per_ring(vals["ring.junction.eff_at_0v_pm_per_v"])
        v_bi = per_die(vals["ring.junction.v_bi_v"])
        depletion_exponent = per_die(vals["ring.junction.depletion_exponent"])
        depletion_a_m = eff_at_0v / (depletion_exponent * v_bi ** (depletion_exponent - 1.0))
    return JunctionParams(
        model=int(use_lut), fill_fraction=fill_fraction, depletion_a_m=depletion_a_m, v_bi_v=v_bi,
        depletion_exponent=depletion_exponent, eff_at_0v_m_per_v=eff_at_0v,
        dqi_dv_frac_per_v=per_ring(optional("ring.junction.dqi_dv_frac_per_v", BN, 0.0)),
        capacitance_at_0v_f=per_ring(optional("ring.junction.capacitance_at_0v_ff", BN, 0.0)),
        series_resistance_ohm=per_ring(optional("ring.junction.series_resistance_ohm", BN, 0.0)),
        lut_v_reverse_v=grid, lut_dneff=lut_dn, lut_alpha_fca_per_m=lut_a, lut_capacitance_f_per_m=lut_c,
        lut_r_series_ohm_m=lut_r,
        loss_ratio_q=per_die(optional("ring.junction.loss_ratio_q", B1, np.nan, "nan (absent)")))


# build driver limits, DAC errors, and readback parameters.
# check the usable code range against DAC bits.
# draw per-bit normal errors and a uniform INL bow for each device from separate variation keys.
def _heater(cfg, vals, switches, optional, per_ring, per_die, BN, B1, seeds) -> HeaterParams:
    bits = int(cfg.get("ring.heater.dac_bits"))
    code_min = cfg.get("ring.heater.code_min")
    code_max = cfg.get("ring.heater.code_max")
    code_min = 0 if code_min is None else int(code_min)
    code_max = (1 << bits) - 1 if code_max is None else int(code_max)
    if not (0 <= code_min < code_max <= (1 << bits) - 1):
        raise RealizeError(f"ring.heater: need 0 <= code_min < code_max <= 2^dac_bits - 1 (got {code_min}, {code_max})")
    max_update_hz = cfg.get("ring.heater.dac_max_update_hz")
    B, N = BN
    stream = int(rng.Stream.VARIATION)
    bit_error_normal = _normal(seeds, stream, "ring.heater.dac_bit_errors", N * bits, 0).reshape(B, N, bits)
    inl_bow = 2.0 * _uniform(seeds, stream, "ring.heater.dac_inl_bow", N, 0) - 1.0
    power_max_w = cfg.get("array.heater_total_power_max_w")
    current_max_a = cfg.get("array.heater_supply_current_max_a")
    return HeaterParams(
        p_fsr_w=per_ring(vals[relations.P_PFSR]), r_cold_ohm=per_ring(vals["ring.heater.r_cold_ohm"]),
        r_ref_temp_k=per_die(vals["ring.heater.r_ref_temp_c"]), tcr_per_k=per_die(vals["ring.heater.tcr_per_k"]),
        driver_mode=switches["ring.heater.driver_mode"].copy(),
        full_scale_v=per_die(optional("ring.heater.full_scale_v", B1, np.nan, "nan (not this driver law)")),
        full_scale_a=per_die(optional("ring.heater.full_scale_a", B1, np.nan, "nan (not this driver law)")),
        full_scale_w=per_die(optional("ring.heater.full_scale_w", B1, np.nan, "nan (not this driver law)")),
        dac_bits=bits, code_min=code_min, code_max=code_max,
        slew_codes_per_s=per_die(optional("ring.heater.slew_codes_per_s", B1, np.inf, "inf (no hardware limit given)")),
        settle_tau_s=per_die(optional("ring.heater.settle_tau_s", B1, 0.0, "0 (no filter)")),
        dac_inl_lsb=per_die(optional("ring.heater.dac_inl_lsb", B1, 0.0)),
        dac_dnl_lsb=per_die(optional("ring.heater.dac_dnl_lsb", B1, 0.0)),
        compliance_v=per_die(optional("ring.heater.driver_compliance_v", B1, np.inf, "inf (no compliance limit)")),
        pwm_period_s=per_die(optional("ring.heater.pwm_period_s", B1, np.nan, "nan (not dac_pwm)")),
        max_update_hz=float(to_si("dac_max_update_hz", max_update_hz)) if max_update_hz is not None else math.inf,
        readback_present=bool(cfg.get("ring.heater.readback.present")),
        pwm_bits=int(cfg.get("ring.heater.pwm_bits") or 0),
        supply_power_max_w=float(to_si("max_w", power_max_w)) if power_max_w is not None else math.inf,
        supply_current_max_a=float(to_si("max_a", current_max_a)) if current_max_a is not None else math.inf,
        dac_bit_err_z=bit_error_normal, inl_bow_u=inl_bow,
        readback_adc_bits=int(cfg.get("ring.heater.readback.adc_bits") or 0),
        readback_v_full_scale_v=per_die(optional("ring.heater.readback.v_full_scale_v", B1, np.nan, "nan (absent)")),
        readback_i_full_scale_a=per_die(optional("ring.heater.readback.i_full_scale_a", B1, np.nan, "nan (absent)")),
        readback_noise_rms_codes=per_die(optional("ring.heater.readback.noise_rms_codes", B1, 0.0)),
        readback_average=bool(cfg.get("ring.heater.readback.average", False)))


# build photodiode, dark-current, analog, and ADC parameters in SI.
# a given dark current needs its reference temperature.
# use the inferred Ge activation energy when absent so dark current retains its temperature law.
# convert reference-tap options and record defaults through the caller's helper.
def _monitor(cfg, vals, timing, optional, per_ring, per_die, BN, B1) -> MonitorParams:
    front_end = str(cfg.get("ring.monitor.afe_type"))
    pga_gains = cfg.get("ring.monitor.pga_gains")
    return MonitorParams(
        responsivity_a_per_w=per_ring(vals["ring.monitor.responsivity_a_per_w"]),
        gain_scale=per_die(optional("ring.monitor.gain_scale", B1, 1.0, "1 (no nuisance scale)")),
        dark_i0_a=per_die(optional("ring.monitor.dark_i0_na", B1, 0.0)),
        dark_t0_k=per_die(vals["ring.monitor.dark_t0_c"]) if "ring.monitor.dark_t0_c" in vals else _need_absent(
            vals, "ring.monitor.dark_i0_na", "ring.monitor.dark_t0_c", B1),
        # retain the Arrhenius temperature law using the inferred 0.37 eV activation energy.
        dark_activation_j=per_die(optional("ring.monitor.dark_activation_ev", B1, _dark_ea_default_j(),
                                           "0.37 eV (inferred, mrm.materials.ge_pd_dark_ea_ev)")),
        background_fraction=per_ring(optional("ring.monitor.background_fraction", BN, 0.0)),
        afe_type=int(front_end == "integrate_dump"),
        afe_tau_s=float(cfg.si("ring.monitor.afe_tau_s", 0.0)),
        integration_window_ticks=timing.integration_window_ticks, adc_period_ticks=timing.adc_period_ticks,
        adc_bits=int(cfg.get("ring.monitor.adc_bits")),
        adc_full_scale_a=per_die(vals["ring.monitor.adc_full_scale_ua"]),
        adc_offset_codes=per_ring(optional("ring.monitor.adc_offset_codes", BN, 0.0)),
        adc_gain_error_frac=per_ring(optional("ring.monitor.adc_gain_error_frac", BN, 0.0)),
        adc_inl_lsb=per_die(optional("ring.monitor.adc_inl_lsb", B1, 0.0)),
        adc_dnl_lsb=per_die(optional("ring.monitor.adc_dnl_lsb", B1, 0.0)),
        input_noise_a_per_rt_hz=per_die(optional("ring.monitor.input_noise_pa_per_rt_hz", B1, 0.0)),
        flicker_corner_hz=per_die(optional("ring.monitor.flicker_corner_hz", B1, 0.0)),
        adc_ref_tempco_per_k=per_die(optional("ring.monitor.adc_ref_tempco_ppm_per_k", B1, 0.0)),
        pga_gains=np.asarray(pga_gains if pga_gains else [], dtype=float),
        channels_per_adc=int(cfg.get("ring.monitor.adc_mux.channels_per_adc") or 1),
        mux_order=str(cfg.get("ring.monitor.adc_mux.order") or "round_robin"),
        baseline_a=per_die(optional("ring.monitor.baseline_ua", B1, 0.0, "0 (no analog baseline subtraction)")),
        stuck_flagged=bool(cfg.get("ring.monitor.stuck_flagged", True)),
        flicker_band_min_hz=float(cfg.get("ring.monitor.flicker_band_min_hz") or 1e-3),
        ref_tap_present=bool(cfg.get("ring.monitor.reference_tap.present", False)),
        ref_tap_coupling_frac=per_die(optional("ring.monitor.reference_tap.coupling_frac", B1, np.nan,
                                               "nan (no reference tap)")),
        ref_tap_responsivity_a_per_w=per_die(optional("ring.monitor.reference_tap.responsivity_a_per_w", B1, np.nan,
                                                      "nan (the drop PD's)")),
        ref_tap_dark_i0_a=per_die(optional("ring.monitor.reference_tap.dark_i0_na", B1, np.nan, "nan (the drop PD's)")),
        adc_ref_site=str(cfg.get("ring.monitor.adc_ref_site") or "package_base"))


# return a NaN placeholder only when the input that requires it is also absent.
def _need_absent(vals, given: str, missing: str, shape) -> np.ndarray:
    if given in vals:
        raise RealizeError(f"{given} is given, so {missing} is required")
    return np.full(shape, np.nan)


# build hardware timing in whole plant ticks after checking compatibility.
# the integration window must fit the ADC period, including all multiplexed windows.
# controller updates must respect the heater DAC rate.
# reject all detected timing violations together before building latency arrays.
def _timing(cfg, vals, latency_scale, diag) -> TimingParams:
    dt_s = float(cfg.si("sim.dt_plant_s"))
    adc_ticks = _period_ticks(cfg, "ring.monitor.adc_rate_hz", default=0)
    controller_ticks = _period_ticks(cfg, "controller_interface.update_rate_hz", default=0)
    front_end = str(cfg.get("ring.monitor.afe_type"))
    window_ticks = (_period_ticks(cfg, "ring.monitor.integration_window_s", default=0, rate=False)
                    if front_end == "integrate_dump" else 0)
    errors = []
    if adc_ticks < 1:
        errors.append("ring.monitor.adc_rate_hz must give an ADC period of at least one tick")
    if front_end == "integrate_dump":
        if window_ticks < 1:
            errors.append("ring.monitor.integration_window_s must be at least one tick")
        if window_ticks > adc_ticks:
            errors.append(f"ring.monitor.integration_window_s ({window_ticks} ticks) must fit in one ADC period "
                        f"({adc_ticks} ticks)")
        channels_per_adc = int(cfg.get("ring.monitor.adc_mux.channels_per_adc") or 1)
        if (str(cfg.get("ring.monitor.adc_mux.order") or "round_robin") == "round_robin" and channels_per_adc > 1
                and channels_per_adc * window_ticks > adc_ticks >= 1):
            errors.append(f"ring.monitor.adc_mux: {channels_per_adc} rings take turns on one integrating converter, so "
                        f"their {window_ticks}-tick windows need {channels_per_adc * window_ticks} ticks per ADC "
                        f"period ({adc_ticks} ticks)")
    if (str(cfg.get("ring.monitor.adc_ref_site") or "package_base") == "monitor_pd"
            and int(cfg.get("ring.monitor.adc_mux.channels_per_adc") or 1) > 1):
        errors.append("ring.monitor.adc_ref_site monitor_pd puts each ring's converter at its PD, which needs "
                    "ring.monitor.adc_mux.channels_per_adc = 1")
    max_update_hz = cfg.get("ring.heater.dac_max_update_hz")
    if controller_ticks and max_update_hz:
        min_period = 1.0 / float(to_si("dac_max_update_hz", max_update_hz))
        if controller_ticks * dt_s < min_period * (1 - 1e-9):
            errors.append(f"controller period {controller_ticks * dt_s:g} s is shorter than the heater DAC's minimum "
                        f"update period {min_period:g} s")
    bias_rate = cfg.get("ring.bias_dac.update_rate_hz")
    if bias_rate is not None and _period_ticks(cfg, "ring.bias_dac.update_rate_hz", default=1) < 1:
        errors.append("ring.bias_dac.update_rate_hz must give a period of at least one tick")
    if errors:
        raise RealizeError("timing rules: " + "; ".join(errors))

    def latency_ticks(ch):
        path = f"derived.timing.{ch}.latency_ticks"
        if path in vals:
            return np.broadcast_to(np.asarray(vals[path], np.int64), (latency_scale.shape[0], 1)).copy()
        return np.zeros((latency_scale.shape[0], 1), dtype=np.int64)
    ramp_s = cfg.get("sim.reset_ramp_s")
    return TimingParams(
        dt_s=dt_s, controller_period_ticks=controller_ticks, adc_period_ticks=adc_ticks,
        integration_window_ticks=window_ticks,
        monitor_latency_ticks=latency_ticks("tx.monitor"), heater_latency_ticks=latency_ticks("tx.heater"),
        bias_latency_ticks=latency_ticks("tx.bias"), readback_latency_ticks=latency_ticks("heater.readback"),
        readback_period_ticks=_period_ticks(cfg, "ring.heater.readback.rate_hz", default=0),
        error_count_latency_ticks=latency_ticks("rx.error_count"), latency_scale=latency_scale.copy(),
        reset_protocol=schema.enum_code("sim.reset_protocol", cfg.get("sim.reset_protocol")),
        reset_ramp_ticks=int(round(float(ramp_s) / dt_s)) if ramp_s else 0,
        substep_policy=schema.enum_code("sim.substep_policy", cfg.get("sim.substep_policy")),
        max_detuning_step_m=float(cfg.get("sim.max_detuning_step_pm") or 0.2) * 1e-12,
        min_samples_per_dither_period=int(cfg.get("sim.min_samples_per_dither_period") or 8))


# sensor type codes are indices into this tuple
_SENSOR_TYPES = ("diode", "rtd", "thermistor")


# build sensor arrays from the table and draw device offset and gain errors.
# resolve named sites or retain attachment strings.
# convert values to SI and scale each sensor latency by the world's latency factor before rounding.
def _temp_sensors(cfg, layout: LayoutParams, seeds, latency_scale, B: int, dt: float) -> TempSensorParams:
    records = cfg.get("temp_sensors") or []
    S = len(records)

    def col(fname, unit_key, default=0.0):
        out = np.zeros((B, S))
        for i, r in enumerate(records):
            v = r.get(fname)
            out[:, i] = float(to_si(unit_key, v)) if v is not None else default
        return out
    names, types, site_indices, attach, period_ticks, latency_ticks = [], [], [], [], [], np.zeros((B, S),
        dtype=np.int64)
    for i, r in enumerate(records):
        names.append(str(r.get("name", f"t{i}")))
        types.append(_SENSOR_TYPES.index(r.get("type", "diode")))
        s = r.get("site")
        if s is not None:
            if s not in layout.site_names:
                raise RealizeError(f"temp_sensors[{i}]: site {s!r} is not in layout.sites")
            site_indices.append(layout.site_names.index(s))
            attach.append("")
        else:
            site_indices.append(-1)
            attach.append(str(r.get("attach", "die")))
        rate = r.get("sample_rate_hz")
        period_ticks.append(int(round(1.0 / (float(rate) * dt))) if rate else 0)
        if r.get("latency_s") is not None:
            latency_ticks[:, i] = np.rint(float(r["latency_s"]) * latency_scale[:, 0] / dt).astype(np.int64)
    stream = int(rng.Stream.VARIATION)
    z_offset = _normal(seeds, stream, "temp_sensors.offset", max(S, 1), 0)[:, :S]
    z_gain = _normal(seeds, stream, "temp_sensors.gain_error", max(S, 1), 0)[:, :S]
    return TempSensorParams(
        names=tuple(names), type=np.asarray(types, dtype=np.int64), site_index=np.asarray(site_indices, dtype=np.int64),
        attach=tuple(attach), sample_period_ticks=np.asarray(period_ticks, dtype=np.int64), latency_ticks=latency_ticks,
        noise_rms_k=col("noise_rms_k", "noise_rms_k"), flicker_corner_hz=col("flicker_corner_hz", "flicker_corner_hz"),
        resolution_k=col("resolution_k", "resolution_k"),
        offset_k=col("offset_sigma_k", "offset_sigma_k") * z_offset,
        gain_error_frac=col("gain_error_sigma_frac", "gain_error_sigma_frac") * z_gain,
        self_heating_k=col("self_heating_k", "self_heating_k"),
        adc_bits=np.asarray([int(r.get("adc_bits") or 0) for r in records], dtype=np.int64),
        code_zero_k=np.asarray([float(to_si("code_zero_c", r["code_zero_c"])) if r.get("code_zero_c") is not None
                                else np.nan for r in records], dtype=float))


# ring.thermal.* interim keys passed to InterimRom in SI, and those passed in um (the InterimRom unit)
_INTERIM_FIELDS_SI = ("tau_fast_s", "share_fast_frac", "tau_slow_s", "tail_tau_s", "tail_share_frac",
                      "selfheat_tau_s", "selfheat_share_frac", "heater_self_ratio", "heater_node_tau_s",
                      "xt_nearest_frac", "xt_floor_frac", "xt_next_frac", "xt_tau_ref_s", "xt_tau_min_s",
                      "xt_tau_far_s", "site_floor_frac")
_INTERIM_FIELDS_UM = ("xt_dist_ref_um", "xt_far_um", "site_ref_um")


# assemble dynamics and return the system plus optional optical DC gains.
# the interim model uses config time constants and the derived heater gain.
# convert distance fields back to um where its API requires them.
# a scalar distance option must agree across worlds.
# the reduced-order model uses a checksum-verified artefact.
# scale its DC gains per world to match the mean derived heater gain.
# warn when that factor is outside 0.5 to 2.
# return its optical-to-ring gains when the hook supports them.
def _assemble_thermal(thermal_module, cfg, thermal_inputs: ThermalInputs, vals: dict, N: int, diag: dict):
    B = thermal_inputs.z_heater_ring_k_per_w.shape[0]
    scale_fields = {f.name for f in fields(thermal_module.ThermalScales)}
    unknown = set(thermal_inputs.scales) - scale_fields
    if unknown:
        raise RealizeError(f"uncertainty.thermal.{sorted(unknown)} are not ThermalScales fields {sorted(scale_fields)}")
    scales = {k: np.asarray(v, float) for k, v in thermal_inputs.scales.items()}
    scales["scaling_hypothesis"] = list(thermal_inputs.scaling_hypothesis)
    z_opt_rom = None
    if thermal_inputs.model == "interim":
        model_options = {"z_heater_ring_k_per_w": np.asarray(thermal_inputs.z_heater_ring_k_per_w),
              "z_ratio_opt_heater": np.asarray(vals["ring.thermal.z_ratio_opt_heater"], float).reshape(-1)}
        for f in _INTERIM_FIELDS_SI:
            p = "ring.thermal." + f
            if p in vals:
                model_options[f] = np.broadcast_to(np.asarray(vals[p], float), (B, 1))[:, 0].copy()
        for f in _INTERIM_FIELDS_UM:
            p = "ring.thermal." + f
            if p in vals:
                # the per-world value (a hypothesis may set it) - InterimRom takes one float, so worlds must agree
                v = np.asarray(vals[p], float).reshape(-1) / 1e-6          # SI m -> um (the InterimRom unit)
                if np.max(v) - np.min(v) > 1e-9 * max(float(np.max(np.abs(v))), 1.0):
                    raise RealizeError(f"{p} differs between worlds ({np.min(v):g}..{np.max(v):g} um); the interim "
                                       f"thermal model takes one value per batch: pin the hypothesis that sets it")
                model_options[f] = float(v[0])
            elif cfg.get(p) is not None:
                model_options[f] = float(cfg.get(p))
        for f in ("pd_offset_um", "temp_sensor_offset_um", "driver_offset_um"):
            v = cfg.get("ring.thermal." + f)
            if v is not None:
                model_options[f] = tuple(float(x) for x in v)
        model_options["xt_tau_law"] = str(cfg.get("ring.thermal.xt_tau_law"))
        model_options["crosstalk_dynamics"] = str(cfg.get("ring.thermal.crosstalk_dynamics"))
        # each ring's r_ref (R3, (B, N) in um) - the crosstalk tau law takes heat-source-to-victim distances, the centre distance less the ring radii (mrm.plant.thermal.crosstalk_tau_s)
        model_options["ring_radius_um"] = np.broadcast_to(np.asarray(vals[relations.P_RADIUS], float), (B, N)) / 1e-6
        rom = thermal_module.InterimRom(**model_options)
    else:
        rom = thermal_module.load_rom(thermal_inputs.rom_path)
        source_index = list(rom.sources).index("heater")
        observer_index = list(rom.observers).index("ring_mode_weighted")
        fem_gain_k_per_w = float(np.sum(rom.self_res[source_index, observer_index]))
        dc_scale = np.mean(np.asarray(thermal_inputs.z_heater_ring_k_per_w), axis=1) / fem_gain_k_per_w
        diag["rom_dc_scale"] = dc_scale.copy()
        if np.any((dc_scale < 0.5) | (dc_scale > 2.0)):
            import warnings
            warnings.warn(f"rom_dc_scale {dc_scale.min():.3g}..{dc_scale.max():.3g} is outside [0.5, 2]: the FEM "
                          f"stack disagrees with the measured heater efficiency under this dlambda/dT", stacklevel=3)
        previous_scale = scales.get("dc_scale")
        scales["dc_scale"] = dc_scale if previous_scale is None else dc_scale * previous_scale
    system = thermal_module.assemble(rom, thermal_inputs.sites_um, N, thermal_module.ThermalScales(**scales),
                                     temp_sensors=thermal_inputs.temp_sensors,
                                     ring_pitch_um=thermal_inputs.ring_pitch_um)
    if thermal_inputs.model == "rom" and hasattr(thermal_module, "selfheat_dc_k_per_w"):
        z_opt_rom = np.broadcast_to(thermal_module.selfheat_dc_k_per_w(system), (B, N)).copy()
    return system, z_opt_rom


# bound the feedback gain from delaying cross-ring absorption by one step.
# the optical-heating calculation uses the previous substep's absorbed power for other rings.
# estimate its largest small-signal gain and require less than `SELFHEAT_LAG_GAIN_MAX` when self-heating is enabled.
# this checks a numerical lag approximation, separately from physical thermal bistability.
# combine the largest Lorentzian absorption slope with thermal gains and the fraction of each cross-ring response reached in one step.
# use the assembled system's cross-path calculation when available.
# otherwise use interim poles and distances.
# return None for a ROM without a usable assembled-system hook.
def _selfheat_lag_gain(
    cfg,
    optics: OpticsParams,
    thermal: ThermalParams,
    laser: LaserParams,
    layout: LayoutParams,
    dt: float,
    thermal_model: str,
    system,
    thermal_module=None,
) -> float | None:

    N = optics.n_g.shape[1]
    if N < 2:
        return 0.0
    omega_rad_per_s = 2 * math.pi * C0 / optics.lambda_res_cold_m
    intrinsic_decay_rate, through_coupling_decay_rate, drop_coupling_decay_rate = (
        omega_rad_per_s / optics.q_intrinsic,
        omega_rad_per_s / optics.q_through,
        omega_rad_per_s / optics.q_drop,
    )
    total_decay_rate = intrinsic_decay_rate + through_coupling_decay_rate + drop_coupling_decay_rate
    input_power_w = np.max(laser.power_w, axis=1, keepdims=True) * laser.input_transmission
    peak_absorbed_w = (
        optics.heat_fraction
        * 4.0
        * through_coupling_decay_rate
        * intrinsic_decay_rate
        / (total_decay_rate * total_decay_rate)
        * input_power_w
    )
    half_linewidth_m = optics.lambda_res_cold_m / (2.0 * optics.q_loaded)
    power_resonance_derivative_bound_w_per_m = (
        peak_absorbed_w * _LORENTZ_MAX_SLOPE / half_linewidth_m
    )  # (B,N) W/m
    selfheat_gain_m_per_w = thermal.self_heating_gain_m_per_w  # (B,N) m/W
    if (
        system is not None
        and thermal_module is not None
        and hasattr(thermal_module, "cross_optical_lag_gain")
    ):
        power_temp_derivative_bound_w_per_k = (
            power_resonance_derivative_bound_w_per_m * thermal.s_lock_m_per_k
        )  # (B,N) W/K
        return float(
            np.max(thermal_module.cross_optical_lag_gain(system, dt, power_temp_derivative_bound_w_per_k))
        )
    if thermal_model == "rom":
        return None
    interim = thermal.inputs.interim
    dist_m = layout.ring_distance_m  # (N,N) m - ring centre-to-centre distance
    positive_distances_m = dist_m[dist_m > 0]
    if positive_distances_m.size == 0:  # every ring on one site - no cross path
        return 0.0
    ranks = np.unique(np.round(positive_distances_m / positive_distances_m.min(), 9))
    near = np.isclose(dist_m, positive_distances_m.min()) & (dist_m > 0)
    second = (
        (np.isclose(dist_m, ranks[1] * positive_distances_m.min()) & (dist_m > 0))
        if ranks.size > 1
        else np.zeros_like(near)
    )

    # the per-world interim value `name` as a (B, 1, 1) array (`default` when absent).
    def interim_value(name, default=None):
        if name not in interim:
            return default
        return np.asarray(interim[name], float).reshape(-1, 1, 1)

    floor = interim_value("xt_floor_frac")
    cross_fraction = np.where(
        near[None],
        interim_value("xt_nearest_frac"),
        np.where(second[None], interim_value("xt_next_frac", floor), floor),
    )
    # use the plant's distance-to-time-constant law when available.
    # its distances are in um and run from ring edge to ring edge.
    # otherwise use the near-field distance-squared law.
    r_um = np.asarray(optics.radius_m, float) * 1e6  # (B,N)
    d_um = np.maximum(dist_m[None] * 1e6 - r_um[:, :, None] - r_um[:, None, :], 0.0)
    far_m = interim_value("xt_far_m")
    law = str(cfg.get("ring.thermal.xt_tau_law"))
    try:
        from mrm.plant.thermal import crosstalk_tau_s
    except ImportError:  # pragma: no cover - the thermal module is part of the package
        crosstalk_tau_s = None
    if crosstalk_tau_s is not None:
        far_s = interim_value("xt_tau_far_s")
        tau = crosstalk_tau_s(
            d_um,
            interim_value("xt_tau_ref_s"),
            float(np.asarray(interim["xt_dist_ref_m"]).ravel()[0]) * 1e6,
            interim_value("xt_tau_min_s"),
            far_s,
            float(far_m.ravel()[0]) * 1e6 if far_m is not None else 100.0,
            law,
        )  # 100 um - InterimRom.xt_far_um default
    else:  # pragma: no cover
        tau = np.maximum(
            interim_value("xt_tau_min_s"),
            interim_value("xt_tau_ref_s") * (d_um * 1e-6 / interim_value("xt_dist_ref_m")) ** 2,
        )
        if far_m is not None and "xt_tau_far_s" in interim:
            tau = np.where(d_um * 1e-6 > far_m, interim_value("xt_tau_far_s"), tau)
    scale = thermal.inputs.scales.get("time_scale_slow")
    if scale is not None:
        tau = tau * np.asarray(scale, float).reshape(-1, 1, 1)
    xt_scale = thermal.inputs.scales.get("crosstalk_scale")
    if xt_scale is not None:
        cross_fraction = cross_fraction * np.asarray(xt_scale, float).reshape(-1, 1, 1)
    self_paths = np.eye(N, dtype=bool)[None]
    cross_fraction = np.where(self_paths, 0.0, cross_fraction)
    step_fraction = -np.expm1(-dt / np.where(tau > 0, tau, np.inf))
    gain = (
        step_fraction
        * cross_fraction
        * (selfheat_gain_m_per_w * power_resonance_derivative_bound_w_per_m)[:, None, :]
    ).sum(axis=2)
    return float(np.max(gain))
