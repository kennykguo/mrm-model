"""realize(cfg, batch, seeds) -> PlantParams: derive after sampling, worlds apart from devices (07 P0-4, P1-16).

Order of work (07 P0-4):

1. Draw the epistemic ``uncertainty`` entries once per world, shape (B,), from the counter-RNG stream
   ``Stream.UNCERTAINTY`` (or a Latin hypercube), and apply them to their target leaves (hypotheses select leaf
   alternatives or per-world switches; intervals replace, scale or offset a leaf).
2. Apply an optional sweep ``grid`` ({path: (B,) values}).
3. Draw the per-device ``variation`` entries, shape (B, N), (B, L) or (B, 1), from ``Stream.VARIATION``: first the
   correlated geometry (width, thickness, slab, gaps) mapped through the Jacobians stored with the modes artefact
   (07 P1-16), then the path-keyed entries.
4. Convert every value to SI (the single conversion of :mod:`mrm.params.units`).
5. Run the relation solvers R1..R5 on the arrays (:mod:`mrm.params.relations`).
6. Assemble the thermal MIMO system with the hook published by the plant agent,
   ``mrm.plant.thermal.assemble(rom: ThermalRom | InterimRom, layout_sites, n_rings, scales: ThermalScales, *,
   temp_sensors, ring_pitch_um)``, when it is importable; the inputs are always kept in
   ``PlantParams.thermal_inputs`` (the assembly itself belongs to the plant).
7. Quantize every latency and period to ticks, assert the timing rules (07 P0-6), that no value the plant uses is
   nan (and that every world's heater driver law has its full scale), that every relation closes to 1e-12, and
   the self-heating lag bound (07 P0-1 item 2: cross-ring optical heating may lag one substep only if its
   per-substep gain is below 0.05).

Random keying: draw (seed_b, stream) -> Philox key; counter (tick = crc32(entry name), ring = element index,
slot = component). Element b of a batch therefore equals a B = 1 run with seed ``seeds[b]`` (07 P1-13), and adding
an entry never changes the draws of another.

This module never imports :mod:`mrm.params.config` at import time, so plant code may import :class:`PlantParams`.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from scipy.special import ndtr, ndtri

from mrm.core import rng as _rng
from mrm.params import relations as rel
from mrm.params import schema as sch
from mrm.params.units import to_si
from mrm.units import C0

if TYPE_CHECKING:  # pragma: no cover
    from mrm.params.config import Config

__all__ = ["PlantParams", "OpticsParams", "ThermalParams", "JunctionParams", "HeaterParams", "BiasDacParams",
           "MonitorParams", "TempSensorParams", "DatapathParams", "LaserParams", "LayoutParams", "TimingParams",
           "FidelityFlags", "ThermalInputs", "realize", "RealizeError", "JACOBIAN_TARGETS", "WORLD_VARIABLES",
           "line_frequencies_hz", "SELFHEAT_LAG_GAIN_MAX"]

#: 07 P0-1 item 2: the per-substep gain of the lagged cross-ring optical heating must stay below this value.
SELFHEAT_LAG_GAIN_MAX = 0.05
#: max |dP_abs/d lambda| of a Lorentzian P_max/(1+(d/HWHM)^2) is P_max * 3*sqrt(3)/8 / HWHM (at d = HWHM/sqrt(3)).
_LORENTZ_MAX_SLOPE = 3.0 * math.sqrt(3.0) / 8.0
#: untargeted uncertainty entries that realize() consumes by name (others are passed through in PlantParams.world)
WORLD_VARIABLES = {
    "timing.latency_scale": "multiplies every I/O latency before quantization (07 4.8 'loop latency')",
}
#: Jacobian output name -> (target, how it applies, sign). 'offset' adds sign*sum(J*dg) in the leaf's units;
#: 'lnscale' multiplies the leaf by exp(sign*sum(J*dg)); 'exp' takes the coefficients as the secant of a quantity that
#: is exponential in the geometry (the bend radiation, extract_modes handoff): the leaf is multiplied by
#: exp(sum(d_k*dg_k)) with d_k = d ln(value)/dp_k from the table's stored evaluations (J_k/leaf without them), so it
#: stays positive; 'kappa2' accumulates d ln K^2 at a fixed wavelength for the R4_q geometry branch (07 P1-16: the
#: anchors describe the nominal-geometry device, mrm.params.relations).
JACOBIAN_TARGETS = {
    "lambda_res_nm_per_nm": (rel.P_OFFSET, "offset", -1.0),       # offset = line - resonance
    "resonance_offset_nm_per_nm": (rel.P_OFFSET, "offset", 1.0),
    "n_eff_per_nm": (rel.P_NEFF, "offset", 1.0),
    "n_g_per_nm": (rel.P_NG, "offset", 1.0),
    "dlambda_dt_pm_per_k_per_nm": (rel.P_S0, "offset", 1.0),
    "ln_kappa2_through_per_nm": ("derived.geometry.dln_kappa2_through", "kappa2", 1.0),
    "ln_kappa2_drop_per_nm": ("derived.geometry.dln_kappa2_drop", "kappa2", 1.0),
    "alpha_rad_db_per_cm_per_nm": (rel.P_A_RAD, "exp", 1.0),
    "ln_alpha_rad_per_nm": (rel.P_A_RAD, "lnscale", 1.0),
}
#: the stored evaluation (jacobian.evaluations.<input>.plus/minus) of an 'exp' output
_JAC_EVAL_NAME = {"alpha_rad_db_per_cm_per_nm": "alpha_rad"}
#: coupler dispersion of each 'kappa2' output: the device's cold resonance moves by the lambda_res Jacobian, and the
#: anchors hold at the cold resonance, so K^2 there also moves by d ln K^2/d lambda x d lambda_res
_KAPPA2_DISPERSION = {"derived.geometry.dln_kappa2_through": "ring.optics.dln_kappa2_through_dlambda_per_nm",
                      "derived.geometry.dln_kappa2_drop": "ring.optics.dln_kappa2_drop_dlambda_per_nm"}
#: Jacobian input names -> variation.geometry dimension
_JAC_DIMS = {"width_nm": "width", "film_nm": "thickness", "thickness_nm": "thickness", "slab_nm": "slab",
             "gap_through_nm": "gap_through", "gap_drop_nm": "gap_drop"}
_GEOM_SLOT = {"width": 0, "thickness": 1, "slab": 2, "gap_through": 3, "gap_drop": 4}


class RealizeError(ValueError):
    """realize() could not build a consistent PlantParams (timing rules, lag bound, missing artefacts)."""


def _freeze(x):
    if isinstance(x, np.ndarray):
        x = np.ascontiguousarray(x)
        x.setflags(write=False)
    return x


class _Frozen:
    """Mixin: make every ndarray field read-only after construction."""

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray):
                object.__setattr__(self, f.name, _freeze(v))


# ------------------------------------------------------------------------------------------------ PlantParams groups
# Shapes: die (B, 1); ring (B, N); line (B, L); sensor (B, S); global scalars are python int/float. SI units.

@dataclass(frozen=True)
class OpticsParams(_Frozen):
    """Ring optics at reference conditions (cold cavity, T_ref, reference bias). See docs/PARAMETERS_FORMAT.md."""
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
    tcmt_orders: int                   # orders summed by the TCMT model


@dataclass(frozen=True)
class ThermalParams(_Frozen):
    """Thermo-optic anchor and DC thermal gains (R1, R2). The MIMO dynamics are in ``system`` / ``inputs``."""
    dlambda_dt_s0_m_per_k: np.ndarray  # (B,N) s0 at T0
    dlambda_dt_beta_per_k: np.ndarray  # (B,1) beta (0 when thermo_optic_order = 1)
    dlambda_dt_t0_k: np.ndarray        # (B,1) T0
    t_ring_lock_k: np.ndarray          # (B,N) diagnostic: lock temperature at ambient T_ref
    s_lock_m_per_k: np.ndarray         # (B,N) s(T_lock)
    heater_efficiency_m_per_w: np.ndarray  # (B,N) FSR/P_FSR (R2)
    z_heater_ring_k_per_w: np.ndarray  # (B,N) Z_heater->ring(0) (R2)
    z_ratio_opt_heater: np.ndarray     # (B,1)
    z_opt_ring_k_per_w: np.ndarray     # (B,N) z_ratio * Z_heater->ring
    self_heating_gain_m_per_w: np.ndarray  # (B,N) s(T_lock) * Z_opt->ring  (per absorbed W)
    system: Any = None                 # result of mrm.plant.thermal.assemble, or None
    inputs: Any = None                 # ThermalInputs (always present)


@dataclass(frozen=True)
class ThermalInputs:
    """What the thermal assembly hook receives (and what PlantParams keeps when the hook is absent).

    model: 'interim' or 'rom'; rom_path: absolute path of the ROM artefact (rom) or None; interim: dict of SI arrays
    (B, 1) with the ring.thermal interim keys (tau_fast_s, share_fast_frac, tau_slow_s, tail_tau_s, tail_share_frac,
    selfheat_tau_s, selfheat_share_frac, xt_nearest_frac, xt_floor_frac, xt_tau_ref_s, xt_dist_ref_um (in m),
    xt_tau_min_s, xt_tau_far_s, xt_far_um (in m), transport_delay_s); z_heater_ring_k_per_w and z_opt_ring_k_per_w
    (B, N) DC gains from R2; dlambda_dt_*: the R1 anchor; family/stack: (B, 1) int codes of the schema enums;
    scales: {name: (B,) array} of every untargeted ``uncertainty.thermal.*`` draw (e.g. time_scale_fast); dt_s.
    """
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


@dataclass(frozen=True)
class JunctionParams(_Frozen):
    """Depletion junction (07 P1-7). Analytic law dlambda(V) = A*((V+Vbi)^m - Vbi^m) [V in volts], or a LUT."""
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


@dataclass(frozen=True)
class HeaterParams(_Frozen):
    """Heater, DAC and driver law (07 P1-11). driver_mode codes: 0 voltage, 1 current, 2 power, 3 dac_pwm."""
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
    # heater V/I readback (07 P1-10; integrator addition for mrm.plant.sensing.ReadbackHw.from_plant_params)
    readback_adc_bits: int = 0                       # 0 when absent
    readback_v_full_scale_v: np.ndarray | None = None  # (B,1) nan when absent
    readback_i_full_scale_a: np.ndarray | None = None  # (B,1) nan when absent
    readback_noise_rms_codes: np.ndarray | None = None  # (B,1) 0 when absent
    readback_average: bool = False                   # boxcar over the sample period (True) or track-and-hold


@dataclass(frozen=True)
class BiasDacParams(_Frozen):
    """Bias actuator tx.bias (07 P0-7)."""
    bits: int
    v_min_v: np.ndarray                # (B,1)
    v_max_v: np.ndarray                # (B,1)
    update_period_ticks: int           # 1 when no update rate is given
    slew_v_per_s: np.ndarray           # (B,1) inf when absent
    settle_tau_s: np.ndarray           # (B,1)
    noise_rms_codes: np.ndarray        # (B,1)


@dataclass(frozen=True)
class MonitorParams(_Frozen):
    """Drop-port monitor PD, AFE and ADC (07 P0-7, P1-9, P1-10). afe_type: 0 rc, 1 integrate_dump."""
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
    # integrator additions (docs/handoff/plant_sensing.md section 4)
    mux_order: str = "round_robin"     # ring.monitor.adc_mux.order
    baseline_a: np.ndarray | None = None  # (B,1) analog baseline subtracted before the ADC (0 when absent)
    stuck_flagged: bool = True         # a stuck converter clears monitor_valid
    flicker_band_min_hz: float = 1e-3  # lowest 1/f pole of the flicker synthesis
    ref_tap_present: bool = False      # reference-tap PD (07 P1-9)
    ref_tap_coupling_frac: np.ndarray | None = None  # (B,1) bus power fraction to the tap (nan when absent)
    ref_tap_responsivity_a_per_w: np.ndarray | None = None  # (B,1) nan = the drop PD's
    ref_tap_dark_i0_a: np.ndarray | None = None  # (B,1) nan = the drop PD's
    adc_ref_site: str = "package_base"  # ring.monitor.adc_ref_site: package_base | monitor_pd


@dataclass(frozen=True)
class TempSensorParams(_Frozen):
    """Temperature sensors (07 P1-10). type codes: 0 diode, 1 rtd, 2 thermistor. site_index -1 = attach."""
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


@dataclass(frozen=True)
class DatapathParams(_Frozen):
    """Data path (07 P0-11) and data-activity heat inputs (07 P1-8)."""
    bitrate_hz: np.ndarray             # (B,1) (nan when absent)
    bitstat_window_bits: int           # 0 when absent
    error_count_period_ticks: int      # 0 when absent
    transition_density: np.ndarray     # (B,1) (nan when absent)
    driver_output_resistance_ohm: np.ndarray  # (B,1) (nan when absent)
    driver_static_power_w: np.ndarray  # (B,1) (0 when absent)
    driver_energy_per_transition_j: np.ndarray  # (B,1) (0 when absent)
    swing_pp_v: np.ndarray             # (B,N) (nan when absent)
    level_convention: np.ndarray       # (B,1) int64: 0 centered, 1 one_sided (driver.level_convention, per world)
    # BER and bit-statistics hardware (integrator additions, docs/handoff/plant_sensing.md section 4; nan when absent,
    # then mrm.plant.datapath uses its module constants)
    rx_sensitivity_oma_w: np.ndarray | None = None   # (B,N) receiver OMA sensitivity at ber_ref, W
    ber_ref: np.ndarray | None = None                # (B,1)
    link_transmission: np.ndarray | None = None      # (B,1) exp(-link_loss) TX bus output -> RX PD
    ber_floor: np.ndarray | None = None              # (B,1)
    bitstat_bw_hz: np.ndarray | None = None          # (B,1)
    bitstat_noise_a_per_rt_hz: np.ndarray | None = None  # (B,1)


@dataclass(frozen=True)
class LaserParams(_Frozen):
    """Laser lines (07 P1-12). Line k = 0..L-1 in increasing frequency unless a lines table is given."""
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


@dataclass(frozen=True)
class LayoutParams(_Frozen):
    """Named sites (07 P1-2). Rings are sites ring0..ring{N-1}."""
    site_names: tuple                  # (S_sites,)
    site_xy_m: np.ndarray              # (S_sites, 2)
    site_kind: tuple                   # (S_sites,)
    ring_site_index: np.ndarray        # (N,) int64
    ring_xy_m: np.ndarray              # (N, 2)
    ring_distance_m: np.ndarray        # (N, N)
    bus_order: np.ndarray              # (N,) int64 ring index at each bus position
    ring_to_line: np.ndarray           # (N,) int64 assigned line of each ring
    ring_spacing_on_bus_m: float       # 0 when absent


@dataclass(frozen=True)
class TimingParams(_Frozen):
    """Integer-tick timing (07 P0-6)."""
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
    # sim.* settings the kernel needs (integrator addition; 07 P0-1 item 6, P0-10)
    reset_protocol: int = 0            # 0 laser_ramp, 1 branch_low, 2 branch_high (sim.reset_protocol)
    reset_ramp_ticks: int = 0          # laser turn-on ramp of laser_ramp (sim.reset_ramp_s; 0 when absent)
    substep_policy: int = 0            # 0 fixed, 1 auto (sim.substep_policy)
    max_detuning_step_m: float = 0.2e-12   # sim.max_detuning_step_pm (schema default 0.2 pm)
    min_samples_per_dither_period: int = 8  # sim.min_samples_per_dither_period (schema default 8)


@dataclass(frozen=True)
class FidelityFlags(_Frozen):
    """Fidelity switches as ints (07 P0-9: kernels take an int32 flag array; see ``as_array``)."""
    self_heating: int        # 1 on, 0 off
    thermo_optic_order: int  # 1 or 2
    thermal_model: int       # 0 interim, 1 rom
    junction_model: int      # 0 analytic, 1 lut
    optics_model: int        # 0 exact, 1 tcmt
    backscatter: int         # 0 off, 1 on
    shot_noise: int          # 1 on, 0 off
    pattern_noise: int       # index in the schema enum (0 off)
    data_modulation: int     # 0 mean, 1 lut
    heater_anchor: int       # 0 lock_slope, 1 fsr_integral
    heater_pwm_ripple: int   # 0 off, 1 on

    ORDER = ("self_heating", "thermo_optic_order", "thermal_model", "junction_model", "optics_model", "backscatter",
             "shot_noise", "pattern_noise", "data_modulation", "heater_anchor", "heater_pwm_ripple")

    def as_array(self) -> np.ndarray:
        """int32 array in the order of ``FidelityFlags.ORDER``."""
        return np.array([getattr(self, n) for n in self.ORDER], dtype=np.int32)


@dataclass(frozen=True)
class PlantParams:
    """Everything the plant needs, realized for a batch of B worlds x devices (07 P0-4). See PARAMETERS_FORMAT.md.

    ``world``: {uncertainty entry name: (B,) draw} (hypotheses as int64 choice indices, names in
    ``world_choices``). ``values``: every SI array by config path (and ``derived.*``), for explain and tests.
    ``derivations``: how each derived value was computed. ``diagnostics``: residuals, clip counts, lag gain.
    """
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

    def select(self, b: int) -> "PlantParams":
        """The B = 1 view of batch element ``b`` (keeps every leading dimension as 1).

        Every batched array is cut, including those inside nested dataclasses: ``thermal.system`` (the
        ThermalSystem's (B, M) residues and poles), ``thermal.inputs`` / ``thermal_inputs`` (their (B, N) gains,
        (B, 1) interim values, (B,) scales and the B scaling-hypothesis names), ``world``, ``seeds``, ``values`` and
        the per-element diagnostics ``rom_dc_scale``, ``geometry_draws_nm`` and ``geometry_dlambda_res_nm``.
        """
        import dataclasses
        B = self.batch
        if not -B <= b < B:
            raise IndexError(f"batch element {b} out of range for B = {B}")
        b = b % B

        def cut(x, min_ndim=2):
            # batched arrays are (B, ...) with ndim >= 2; world draws, seeds and scales are (B,)
            if isinstance(x, np.ndarray) and x.ndim >= min_ndim and x.shape[0] == B:
                return _freeze(x[b:b + 1].copy())
            return x

        def cut_group(g, nested=False):
            if g is None or isinstance(g, type) or not dataclasses.is_dataclass(g):
                return g
            kw = {}
            for f in fields(g):
                v = getattr(g, f.name)
                if dataclasses.is_dataclass(v) and not isinstance(v, type):
                    kw[f.name] = cut_group(v, nested=True)
                elif nested and isinstance(v, np.ndarray) and v.ndim == 1 and v.dtype.kind == "f":
                    # inside a hook object (e.g. ThermalSystem.transport_delay_s) a float (B,) array is per world;
                    # its int64 (M,) index arrays are never cut
                    kw[f.name] = cut(v, 1)
                else:
                    kw[f.name] = cut(v)
            if isinstance(g, ThermalInputs):
                kw["scales"] = {k: cut(np.asarray(x), 1) for k, x in g.scales.items()}
                kw["interim"] = {k: cut(np.asarray(x)) for k, x in g.interim.items()}
                if len(g.scaling_hypothesis) == B:
                    kw["scaling_hypothesis"] = (g.scaling_hypothesis[b],)
            return dataclasses.replace(g, **kw)

        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "layout":
                kw[f.name] = v          # not batched
            elif dataclasses.is_dataclass(v) and not isinstance(v, type):
                kw[f.name] = cut_group(v)
            elif f.name in ("world", "seeds"):
                kw[f.name] = {k: cut(x, 1) for k, x in v.items()}
            elif f.name == "values":
                kw[f.name] = {k: cut(x) for k, x in v.items()}
            elif f.name == "diagnostics":
                d = dict(v)
                if "rom_dc_scale" in d:
                    d["rom_dc_scale"] = cut(np.asarray(d["rom_dc_scale"]), 1)
                if "geometry_draws_nm" in d:
                    d["geometry_draws_nm"] = {k: cut(x) for k, x in d["geometry_draws_nm"].items()}
                if "geometry_dlambda_res_nm" in d:
                    d["geometry_dlambda_res_nm"] = cut(np.asarray(d["geometry_dlambda_res_nm"]))
                kw[f.name] = d
            else:
                kw[f.name] = v
        kw["batch"] = 1
        return PlantParams(**kw)


# ------------------------------------------------------------------------------------------------ RNG helpers

def _entry_id(name: str) -> int:
    """Stable 31-bit id of an entry name (the 'tick' word of the Philox counter)."""
    return zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF


def _uniform(seeds: np.ndarray, stream: int, name: str, k: int, slot: int = 0) -> np.ndarray:
    """(B, k) uniforms in (0, 1) keyed by (seed_b, stream; tick = id(name), ring = element, slot)."""
    out = np.empty((seeds.shape[0], k), dtype=np.float64)
    _rng.fill_uniform(out, seeds.astype(np.int64), np.int64(stream), np.int64(_entry_id(name)), np.int64(slot))
    return out


def _normal(seeds: np.ndarray, stream: int, name: str, k: int, slot: int = 0) -> np.ndarray:
    """(B, k) standard normals by the inverse CDF of :func:`_uniform` (exact truncation possible)."""
    return ndtri(_uniform(seeds, stream, name, k, slot))


def _seed_arrays(seeds, batch: int, cfg: "Config") -> dict[str, np.ndarray]:
    """Per-element seeds {device, uncertainty} as (B,) int64. An int s gives s + b (mrm.core.rng.batch_seeds)."""
    def one(s, default):
        if s is None:
            s = default
        if isinstance(s, (int, np.integer)):
            return _rng.batch_seeds(int(s), batch)
        a = np.asarray(s, dtype=np.int64).reshape(-1)
        if a.shape[0] != batch:
            raise RealizeError(f"seeds: expected {batch} per-element seeds, got {a.shape[0]}")
        if np.any(a < 0):
            raise RealizeError("seeds must be >= 0")
        return a
    if isinstance(seeds, dict):
        dev = seeds.get("device", cfg.get("experiment.seeds.device", 0))
        unc = seeds.get("uncertainty", cfg.get("experiment.seeds.uncertainty", dev))
        return {"device": one(dev, 0), "uncertainty": one(unc, 0)}
    s = one(seeds, 0)
    return {"device": s, "uncertainty": s.copy()}


# ------------------------------------------------------------------------------------------------ shapes

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


def line_frequencies_hz(cfg: "Config") -> tuple[np.ndarray, tuple]:
    """Nominal grid frequencies (L,) in Hz and line directions.

    From the ``channel_plan.lines`` table when given, else ``center + (k - (L - 1)/2) * spacing`` for k = 0..L-1
    (a grid "centered at" the centre frequency, increasing frequency with k).
    """
    lines = cfg.get("channel_plan.lines")
    if lines:
        f = np.array([to_si("freq_thz", r["freq_thz"]) for r in lines], dtype=float)
        return f, tuple(r.get("direction", "tx") for r in lines)
    L = int(cfg.get("channel_plan.n_lines"))
    fc = to_si("channel_plan.center_thz", cfg.get("channel_plan.center_thz"))
    df = to_si("channel_plan.spacing_ghz", cfg.get("channel_plan.spacing_ghz"))
    k = np.arange(L, dtype=float)
    return fc + (k - (L - 1) / 2.0) * df, tuple(["tx"] * L)


def _line_meta(cfg: "Config", L: int) -> dict:
    """source_laser and pol of each line from the ``channel_plan.lines`` table ('' and 'te' when not given)."""
    lines = cfg.get("channel_plan.lines") or []
    if lines:
        return {"source_laser": tuple(str(r.get("source_laser") or "") for r in lines),
                "pol": tuple(str(r.get("pol") or "te") for r in lines)}
    return {"source_laser": ("",) * L, "pol": ("te",) * L}


def _dark_ea_default_j() -> float:
    """Default Ge PD dark-current activation energy in J: mrm.materials.ge_pd_dark_ea_ev() (0.37 eV, inferred)."""
    from mrm.materials import ge_pd_dark_ea_ev
    return float(to_si("dark_activation_ev", ge_pd_dark_ea_ev()))


def _layout(cfg: "Config", N: int, L: int) -> LayoutParams:
    sites = dict(cfg.get("layout.sites") or {})
    pitch = cfg.get("layout.ring_pitch_um")
    names, xy, kinds = [], [], []
    ring_idx = []
    for i in range(N):
        nm = f"ring{i}"
        if nm in sites:
            r = sites[nm]
            p = (r["x_um"], r["y_um"])
        elif pitch is not None:
            p = (i * float(pitch), 0.0)
        else:
            if N > 1:
                raise RealizeError("layout: give layout.ring_pitch_um or ring<i> sites for every ring (07 P1-2)")
            p = (0.0, 0.0)
        ring_idx.append(len(names))
        names.append(nm)
        xy.append(p)
        kinds.append("ring")
    for nm, r in sites.items():
        if nm in names:
            continue
        names.append(nm)
        xy.append((r["x_um"], r["y_um"]))
        kinds.append(r.get("kind", "other"))
    xy_m = np.asarray(to_si("x_um", np.asarray(xy, dtype=float)), dtype=float).reshape(-1, 2)   # um -> m
    ring_xy = xy_m[ring_idx]
    d = np.sqrt(((ring_xy[:, None, :] - ring_xy[None, :, :]) ** 2).sum(-1))
    bus = np.asarray(cfg.get("array.bus_order") or list(range(N)), dtype=np.int64)
    r2l = np.asarray(cfg.get("channel_plan.ring_to_line") or list(range(N)), dtype=np.int64)
    if sorted(bus.tolist()) != list(range(N)):
        raise RealizeError(f"array.bus_order must be a permutation of 0..{N - 1}")
    if r2l.shape != (N,) or np.any(r2l < 0) or np.any(r2l >= L):
        raise RealizeError(f"channel_plan.ring_to_line must hold {N} line indices in 0..{L - 1}")
    sp = cfg.get("optical_path.ring_spacing_on_bus_um")
    return LayoutParams(site_names=tuple(names), site_xy_m=xy_m, site_kind=tuple(kinds),
                        ring_site_index=np.asarray(ring_idx, dtype=np.int64), ring_xy_m=ring_xy, ring_distance_m=d,
                        bus_order=bus, ring_to_line=r2l,
                        ring_spacing_on_bus_m=float(to_si("optical_path.ring_spacing_on_bus_um", sp)) if sp else 0.0)


# ------------------------------------------------------------------------------------------------ uncertainty

def weakest(provs):
    """Weakest provenance (lazy import keeps config out of this module's import graph)."""
    from mrm.params.config import weakest as _w
    return _w(provs)


def _hyp_choices(e) -> dict:
    from mrm.params.config import _hyp_choice_map
    return _hyp_choice_map(e)


def _draw_u(e_name: str, mode: str, seeds: np.ndarray, lhs_cols: dict) -> np.ndarray:
    if mode == "lhs":
        return lhs_cols[e_name]
    return _uniform(seeds, int(_rng.Stream.UNCERTAINTY), "uncertainty." + e_name, 1)[:, 0]


def _interval_x(e, u: np.ndarray) -> np.ndarray:
    f = e.fields
    if e.kind == "interval":
        lo, hi = float(f["low"]), float(f["high"])
        d = f.get("dist", "uniform")
        if d == "uniform":
            return lo + (hi - lo) * u
        if d == "loguniform":
            return np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * u)
        # triangular with the mode at the nominal (or the midpoint)
        c = float(f.get("nominal", 0.5 * (lo + hi))) if isinstance(f.get("nominal"), (int, float)) else 0.5 * (lo + hi)
        fc = (c - lo) / (hi - lo) if hi > lo else 0.5
        return np.where(u < fc, lo + np.sqrt(u * (hi - lo) * (c - lo)), hi - np.sqrt((1 - u) * (hi - lo) * (hi - c)))
    mean, sigma = float(f["mean"]), float(f["sigma"])
    lognormal = f.get("dist", "normal") == "lognormal"
    tr = f.get("truncate")
    if tr:
        # truncate in z (exact, inverse CDF): normal x = mean + sigma z; lognormal x = mean exp(sigma z), so the
        # bounds map through z = ln(x/mean)/sigma (a lower bound <= 0 is no bound)
        lo_hi = np.asarray(tr, float)
        if lognormal:
            with np.errstate(divide="ignore"):
                zb = np.log(np.maximum(lo_hi, 0.0) / mean) / max(sigma, 1e-300)
        else:
            zb = (lo_hi - mean) / max(sigma, 1e-300)
        a, b = zb
        u = ndtr(a) + u * (ndtr(b) - ndtr(a))
    z = ndtri(np.clip(u, 1e-300, 1 - 1e-16))
    if lognormal:
        return mean * np.exp(sigma * z)
    return mean + sigma * z


def _interval_nominal(e, base_mode: str) -> float | None:
    nom = e.fields.get("nominal")
    if isinstance(nom, (int, float)):
        return float(nom)
    if e.target is None:
        return None
    return {"scale": 1.0, "offset": 0.0}.get(base_mode)


# ------------------------------------------------------------------------------------------------ realize

def realize(cfg: "Config", batch: int = 1, seeds: int | np.ndarray | dict = 0, grid: dict | None = None, *,
            uncertainty: str | None = None, variation: str | None = None,
            thermal_hook: Callable | str | None = "auto") -> PlantParams:
    """Realize a config for a batch of B elements (07 P0-4). Returns a frozen :class:`PlantParams` (SI units).

    Args:
        cfg: a :class:`mrm.params.config.Config` from ``mrm.params.load``.
        batch: B.
        seeds: an int s (element b uses s + b), a (B,) array of per-element seeds, or
            ``{"device": ..., "uncertainty": ...}`` with either form (separate streams for devices and worlds).
        grid: optional ``{config path: (B,) values}`` in config units (a sweep), applied after the uncertainty draw
            and before the per-device variation.
        uncertainty: ``nominal`` (every entry at its nominal), ``sample`` (counter RNG) or ``lhs`` (Latin hypercube
            over the batch, scipy.stats.qmc). Default: ``experiment.uncertainty`` (sample).
        variation: ``off`` or ``sample``. Default: ``experiment.variation`` (sample).
        thermal_hook: ``"auto"`` calls ``mrm.plant.thermal.assemble`` when it exists; a callable is used instead;
            None keeps only ``thermal_inputs``.
    """
    B = int(batch)
    if B < 1:
        raise RealizeError("batch must be >= 1")
    N, L = cfg.n_rings, cfg.n_lines
    unc_mode = uncertainty or cfg.get("experiment.uncertainty", "sample")
    var_mode = variation or cfg.get("experiment.variation", "sample")
    if unc_mode not in ("nominal", "sample", "lhs"):
        raise RealizeError(f"uncertainty mode {unc_mode!r} must be nominal, sample or lhs")
    if var_mode not in ("off", "sample"):
        raise RealizeError(f"variation mode {var_mode!r} must be off or sample")
    sd = _seed_arrays(seeds, B, cfg)
    layout = _layout(cfg, N, L)
    diag: dict[str, Any] = {"uncertainty_mode": unc_mode, "variation_mode": var_mode, "defaults_used": [],
                            "clipped": {}}

    # ---- 1. base arrays (config units)
    arr: dict[str, np.ndarray] = {}
    tables: dict[str, list] = {}
    for path, leaf in cfg.leaves.items():
        key = sch.lookup(path)
        if key is None or leaf.is_free:
            continue
        if key.type == "float" and key.level in ("die", "ring", "line"):
            arr[path] = _broadcast(leaf.value, key.level, B, N, L, path)
        elif key.type == "table" and path.startswith("ring.geometry.jacobian"):
            tables[path] = [leaf.value] * B
    switches: dict[str, np.ndarray] = {}
    for k in sch.SCHEMA.values():
        if k.per_world:
            switches[k.path] = np.full((B, 1), sch.enum_code(k.path, cfg.get(k.path)), dtype=np.int64)
    prov = {p: lf.provenance for p, lf in cfg.leaves.items()}

    # ---- 2. uncertainty (worlds)
    world: dict[str, np.ndarray] = {}
    world_choices: dict[str, tuple] = {}
    names = sorted(cfg.uncertainty)
    lhs_cols: dict[str, np.ndarray] = {}
    if unc_mode == "lhs" and names:
        from scipy.stats import qmc
        u = qmc.LatinHypercube(d=len(names), seed=int(sd["uncertainty"][0])).random(B)
        lhs_cols = {n: np.clip(u[:, i], 1e-12, 1 - 1e-12) for i, n in enumerate(names)}
    for name in names:
        e = cfg.uncertainty[name]
        if e.kind == "hypothesis" and e.follows is not None:
            continue                        # a label entry is not drawn: set from its leader below
        if e.kind == "hypothesis":
            cm = _hyp_choices(e)
            choices = list(cm)
            allowed = e.restrict or choices
            nominal = str(e.fields.get("nominal"))
            if e.pinned is not None:
                idx = np.full(B, choices.index(str(e.pinned)), dtype=np.int64)
            elif unc_mode == "nominal":
                nom = nominal if nominal in allowed else allowed[0]
                idx = np.full(B, choices.index(nom), dtype=np.int64)
            else:
                # choice names are strings (config._hyp_choice_map); YAML may give weight keys as numbers (160: 0.7)
                w = {str(k): v for k, v in (e.fields.get("weights") or {}).items()}
                wv = np.array([float(w.get(c, 1.0)) if c in allowed else 0.0 for c in choices])
                if wv.sum() <= 0:
                    raise RealizeError(f"uncertainty.{name}: weights sum to zero")
                cdf = np.cumsum(wv / wv.sum())
                u = _draw_u(name, unc_mode, sd["uncertainty"], lhs_cols)
                idx = np.minimum(np.searchsorted(cdf, u, side="right"), len(choices) - 1).astype(np.int64)
            world[name] = idx
            world_choices[name] = tuple(choices)
            skip = set(e.fields.get("_override_paths", []))
            for ci, c in enumerate(choices):
                mask = idx == ci
                if not mask.any():
                    continue
                for p, a in cm[c].items():
                    if p in skip:
                        continue
                    key = sch.lookup(p)
                    if key is not None and not key.physical:
                        switches[p][mask, 0] = sch.enum_code(p, a)
                        continue
                    if isinstance(a, str):
                        av = cfg.alt_value(p, a)
                        val, aprov = av["value"], av["provenance"]
                    else:
                        val, aprov = a.get("value"), a.get("provenance", e.fields.get("provenance", "assumed"))
                    if key.type == "table":
                        for b in np.nonzero(mask)[0]:
                            tables.setdefault(p, [cfg.get(p)] * B)[b] = val
                        continue
                    if p not in arr:
                        raise RealizeError(f"uncertainty.{name}: {p} is not a realizable numeric leaf")
                    arr[p][mask] = _broadcast(val, key.level, int(mask.sum()), N, L, p)
                    # provenance used to pick the member recomputed in an overdetermined relation
                    prov[p] = aprov if mask.all() else weakest([prov.get(p, "assumed"), aprov])
        else:
            mode = e.fields.get("mode", "absolute")
            if e.pinned == "override":
                continue
            if e.pinned == "nominal" or (e.pinned is None and unc_mode == "nominal"):
                nom = _interval_nominal(e, mode)
                if nom is None:   # targeted absolute entry without a nominal: the leaf keeps its own value
                    world[name] = arr[e.target][:, 0].copy() if e.target in arr else np.full(B, np.nan)
                    continue
                x = np.full(B, nom)
            elif e.pinned is not None:
                x = np.full(B, float(e.pinned))
            else:
                x = _interval_x(e, _draw_u(name, unc_mode, sd["uncertainty"], lhs_cols))
            world[name] = x
            if e.target is not None:
                if e.target not in arr:
                    raise RealizeError(f"uncertainty.{name}: target {e.target} is not a realizable numeric leaf")
                if mode == "absolute":
                    arr[e.target][...] = x[:, None]
                elif mode == "scale":
                    arr[e.target] *= x[:, None]
                else:
                    arr[e.target] += x[:, None]
    # label entries (follows: + map:): the second axis of a joint hypothesis, e.g. g26_calibrated's coupling
    # regime, which the CAL-2 fit ties to the extinction mechanism; the label is the one the leader's choice implies
    for name in names:
        e = cfg.uncertainty[name]
        if e.kind != "hypothesis" or e.follows is None:
            continue
        choices = list(_hyp_choices(e))
        lead = world_choices[e.follows]
        idx = np.array([choices.index(e.label_of(lead[i])) for i in world[e.follows]], dtype=np.int64)
        if e.pinned is not None and np.any(idx != choices.index(str(e.pinned))):
            raise RealizeError(f"uncertainty.{name} is pinned to {e.pinned!r} but uncertainty.{e.follows} drew a "
                               f"choice that implies another label")
        world[name] = idx
        world_choices[name] = tuple(choices)

    # ---- 3. sweep grid
    for p, vals in (grid or {}).items():
        if p not in arr:
            raise RealizeError(f"grid: {p} is not a realizable numeric leaf")
        v = np.asarray(vals, dtype=float).reshape(-1)
        if v.shape[0] != B:
            raise RealizeError(f"grid: {p} needs {B} values, got {v.shape[0]}")
        arr[p][...] = v[:, None]
        prov[p] = "override"

    # ---- 4. variation (devices)
    arr_nom = None
    geo_kappa: dict[str, np.ndarray] = {}
    if var_mode == "sample":
        if cfg.geometry:
            # the nominal-geometry device of every world (after the uncertainty and the grid, before any device
            # draw): its relation solve gives the mask radius, the order and the per-pass couplings the anchors mean
            arr_nom = {p: a.copy() for p, a in arr.items()}
        geo_kappa = _apply_geometry(cfg, arr, tables, sd["device"], layout, B, N, diag)
        _apply_variation(cfg, arr, sd["device"], layout, B, N, L, diag)

    # ---- 5. SI
    f_grid, directions = line_frequencies_hz(cfg)
    lam_grid = C0 / f_grid
    lam_line = np.tile(lam_grid[layout.ring_to_line][None, :], (B, 1))
    vals, given = _si_values(cfg, arr, lam_line)

    # ---- 6. relations
    lat_scale = world.get("timing.latency_scale")
    lat_scale = np.ones((B, 1)) if lat_scale is None else np.asarray(lat_scale, float).reshape(B, 1)
    rel_kw = dict(order2=int(cfg.get("fidelity.thermo_optic_order")) == 2,
                  heater_anchor=str(cfg.get("fidelity.heater_anchor")),
                  tolerance={r: float(d["tolerance_rel"]) for r, d in cfg.relations_cfg.items() if "tolerance_rel" in d},
                  dt=float(cfg.si("sim.dt_plant_s")), latency_scale=lat_scale)
    ctx = rel.RelationContext(vals=vals, given=given, prov=prov, **rel_kw)
    if arr_nom is not None:
        # the device's own R4 anchor members (after a path-keyed device draw of q_loaded, say) with the nominal
        # geometry: a coupler spread drawn on an anchor stays a coupler spread
        for p in (rel.P_QL, rel.P_QT, rel.P_QD, rel.P_XD):
            if p in arr:
                arr_nom[p] = arr[p]
        vals_n, given_n = _si_values(cfg, arr_nom, lam_line)
        ctx_n = rel.solve(rel.RelationContext(vals=vals_n, given=given_n, prov=dict(prov), **rel_kw))
        vn = ctx_n.vals
        ctx.radius_hold = np.broadcast_to(np.asarray(vn[rel.P_RADIUS], float), (B, N)).copy()
        ctx.order_hold = np.broadcast_to(np.asarray(vn[rel.P_ORDER]), (B, N)).copy()
        ctx.coupling_anchor = {
            "kappa2_through": np.broadcast_to(vn["derived.optics.kappa2_through"], (B, N)).copy(),
            "kappa2_drop": np.broadcast_to(vn["derived.optics.kappa2_drop"], (B, N)).copy(),
            "alpha_i_per_m": np.broadcast_to(vn["derived.optics.alpha_i_per_m"], (B, N)).copy(),
            "dln_kappa2_through": geo_kappa["derived.geometry.dln_kappa2_through"],
            "dln_kappa2_drop": geo_kappa["derived.geometry.dln_kappa2_drop"]}
        for t, a in geo_kappa.items():
            vals[t] = a
            ctx.derivations[t] = rel.Derivation(
                t, "variation.geometry", ("ring.geometry.jacobian", "ring.geometry.jacobian_coupling",
                                          "ring.optics.dln_kappa2_through_dlambda_per_nm"
                                          if t.endswith("through") else "ring.optics.dln_kappa2_drop_dlambda_per_nm"),
                "sum_k J_k dg_k + D d lambda_res: d ln K^2 of the device's coupler at its cold resonance (07 P1-16)")
        for p in ("derived.optics.kappa2_through", "derived.optics.kappa2_drop"):
            vals[p.replace("derived.optics.", "derived.geometry.nominal_")] = ctx.coupling_anchor[
                p.rsplit(".", 1)[1]]
    _line_derivations(cfg, ctx)
    rel.solve(ctx)
    vals = ctx.vals
    resid = rel.residuals(vals)
    diag["relation_residuals"] = resid
    diag["relation_notes"] = dict(ctx.notes)

    # ---- 7. groups
    def g(p, shape_like=None, default=None, what="0"):
        if p in vals:
            return np.asarray(vals[p], dtype=float)
        if default is None:
            raise RealizeError(f"{p} is required")
        diag["defaults_used"].append(f"{p}: absent -> {what}")
        return np.full(shape_like, default, dtype=float)

    BN, B1, BL = (B, N), (B, 1), (B, L)

    def bn(x):
        return np.broadcast_to(np.asarray(x, float), BN).copy()

    def b1(x):
        x = np.asarray(x, float)
        return np.broadcast_to(x.reshape(x.shape[0], 1) if x.ndim == 2 and x.shape[1] == 1 else x, B1).copy()

    order2 = ctx.order2
    s0 = bn(vals[rel.P_S0])
    beta = b1(vals["derived.thermal.beta_eff_per_k"])
    t0 = b1(vals[rel.P_T0])
    q_l = bn(vals[rel.P_QL])
    lam_res = bn(vals["derived.optics.lambda_res_cold_m"])
    optics = OpticsParams(
        lambda_line_m=bn(lam_line), lambda_res_cold_m=lam_res, resonance_offset_m=bn(vals[rel.P_OFFSET]),
        t_ref_k=b1(vals[rel.P_OFFSET_T]),
        bias_ref_v=b1(g("ring.optics.resonance_offset_ref_bias_v", B1, 0.0, "0 V")),
        radius_m=bn(vals[rel.P_RADIUS]), length_m=bn(vals["derived.optics.length_m"]),
        n_eff=bn(vals["derived.optics.n_eff_phase"]), n_eff_configured=bn(vals[rel.P_NEFF]), n_g=bn(vals[rel.P_NG]),
        dn_g_dlambda_per_m=bn(g("ring.optics.dn_g_dlambda_per_nm", BN, 0.0)),
        order=np.broadcast_to(vals[rel.P_ORDER], BN).astype(np.int64).copy(), fsr_hz=bn(vals["derived.optics.fsr_hz"]),
        round_trip_time_s=bn(vals["derived.optics.round_trip_time_s"]), q_loaded=q_l,
        q_intrinsic=bn(vals[rel.P_QI]), q_through=bn(vals[rel.P_QT]), q_drop=bn(vals[rel.P_QD]),
        kappa2_through=bn(vals["derived.optics.kappa2_through"]), kappa2_drop=bn(vals["derived.optics.kappa2_drop"]),
        round_trip_amplitude=bn(vals["derived.optics.round_trip_amplitude"]),
        alpha_i_per_m=bn(vals["derived.optics.alpha_i_per_m"]),
        alpha_heat_per_m=bn(vals["derived.optics.alpha_heat_per_m"]),
        heat_fraction=bn(vals[rel.P_ETA]) if rel.P_ETA in vals else bn(
            vals["derived.optics.alpha_heat_per_m"] / vals["derived.optics.alpha_i_per_m"]),
        dln_kappa2_through_dlambda_per_m=bn(g("ring.optics.dln_kappa2_through_dlambda_per_nm", BN, 0.0)),
        dln_kappa2_drop_dlambda_per_m=bn(g("ring.optics.dln_kappa2_drop_dlambda_per_nm", BN, 0.0)),
        dln_alpha_dt_ring_per_k=bn(g("ring.optics.dln_alpha_dt_ring_per_k", BN, 0.0)),
        dln_alpha_dt_heater_per_k=bn(g("ring.optics.dln_alpha_dt_heater_per_k", BN, 0.0)),
        dn_eff_dt_per_k=bn(vals[rel.P_DNDT]), d2n_eff_dt2_per_k2=bn(vals["derived.optics.d2n_eff_dt2_per_k2"]),
        alpha_l_per_k=b1(vals[rel.P_ALPHA_L]),
        alpha_l_slope_per_k2=b1(g(rel.P_ALPHA_L_SLOPE, B1, 0.0)) if order2 else np.zeros(B1),
        backscatter_splitting_hz=bn(g("ring.optics.backscatter_splitting_ghz", BN, 0.0, "0 (off)"))
        if cfg.get("fidelity.backscatter") == "on" else np.zeros(BN),
        fwhm_hz=C0 / (lam_res * q_l),
        alpha_fca_per_m=bn(vals[rel.P_A_FCA]) if rel.P_A_FCA in vals else np.full(BN, np.nan),
        backscatter_rb=_backscatter_rb(cfg, vals, sd["device"], var_mode, B, N),
        tcmt_orders=int(cfg.get("ring.optics.tcmt_orders")),
    )
    s_lock = bn(vals["derived.thermal.s_lock_m_per_k"])
    z_h = bn(vals[rel.P_Z])
    z_ratio = b1(vals["ring.thermal.z_ratio_opt_heater"])
    z_opt = z_h * z_ratio
    thermal_core = dict(
        dlambda_dt_s0_m_per_k=s0, dlambda_dt_beta_per_k=beta, dlambda_dt_t0_k=t0,
        t_ring_lock_k=bn(vals[rel.P_TLOCK]), s_lock_m_per_k=s_lock,
        heater_efficiency_m_per_w=bn(vals[rel.P_EFF]), z_heater_ring_k_per_w=z_h, z_ratio_opt_heater=z_ratio,
        z_opt_ring_k_per_w=z_opt, self_heating_gain_m_per_w=s_lock * z_opt)

    # ---- 8. thermal inputs and the assembly hook (owned by mrm.plant.thermal)
    tmodel = str(cfg.get("fidelity.thermal_model"))
    interim = {}
    for p in sch.SCHEMA:
        if p.startswith("ring.thermal.") and (p.endswith("_s") or p.endswith("_frac") or p.endswith("_um")):
            if p in vals and sch.SCHEMA[p].level != "ring":
                nm = p.rsplit(".", 1)[1]
                interim[nm[:-3] + "_m" if nm.endswith("_um") else nm] = b1(vals[p])   # SI names (um -> m)
    # untargeted interval/dist draws under uncertainty.thermal.* are ThermalScales factors; hypotheses there (07 4.8
    # names thermal.family and thermal.stack) act through their per-world switches and leaves, not as scales
    scales = {n[len("thermal."):]: np.asarray(v) for n, v in world.items() if n.startswith("thermal.")
              and cfg.uncertainty[n].target is None and cfg.uncertainty[n].kind != "hypothesis"}
    fam = sch.SCHEMA["ring.thermal.family"]
    stk = sch.SCHEMA["ring.thermal.stack"]
    hyp = sch.SCHEMA["ring.thermal.scaling_hypothesis"]
    rom_path = None
    if tmodel == "rom":
        rom_path = str((cfg.data_root / cfg.get("ring.thermal.rom_file")).resolve())
    sites = dict(cfg.get("layout.sites") or {})
    tsens = list(cfg.get("temp_sensors") or [])
    tin = ThermalInputs(model=tmodel, rom_path=rom_path, interim=interim, z_heater_ring_k_per_w=_freeze(z_h),
                        z_opt_ring_k_per_w=_freeze(z_opt), dlambda_dt_s0_m_per_k=_freeze(s0),
                        dlambda_dt_beta_per_k=_freeze(beta), dlambda_dt_t0_k=_freeze(t0),
                        family=_freeze(switches["ring.thermal.family"]), family_names=fam.enum,
                        stack=_freeze(switches["ring.thermal.stack"]), stack_names=stk.enum, scales=scales,
                        dt_s=float(cfg.si("sim.dt_plant_s")),
                        scaling_hypothesis=tuple(hyp.enum[int(c)] for c in switches[hyp.path][:, 0]),
                        sites_um=sites, temp_sensors=tsens, ring_pitch_um=cfg.get("layout.ring_pitch_um"))
    system = None
    tmod = None
    if thermal_hook == "auto":
        try:
            from mrm.plant import thermal as tmod  # noqa: WPS433 (owned by the plant agent; optional at import)
        except ImportError:
            tmod = None
        if tmod is not None and not hasattr(tmod, "assemble"):
            tmod = None
    elif thermal_hook is not None:
        tmod = thermal_hook                      # a module-like object exposing the same API (tests)
    if tmod is not None:
        system, z_opt_rom = _assemble_thermal(tmod, cfg, tin, vals, N, diag)
        if z_opt_rom is not None:               # ROM model: Z_opt comes from the FEM, not from z_ratio
            z_opt = z_opt_rom
            thermal_core["z_opt_ring_k_per_w"] = z_opt
            thermal_core["z_ratio_opt_heater"] = b1(np.mean(z_opt / z_h, axis=1, keepdims=True))
            thermal_core["self_heating_gain_m_per_w"] = s_lock * z_opt
    diag["thermal_hook"] = getattr(tmod, "__name__", None) if tmod is not None else None
    thermal = ThermalParams(**thermal_core, system=system, inputs=tin)

    # ---- 9. remaining groups
    junction = _junction(cfg, vals, g, bn, b1, BN, B1)
    heater = _heater(cfg, vals, switches, g, bn, b1, BN, B1, sd["device"])
    timing = _timing(cfg, vals, lat_scale, diag)
    bias = BiasDacParams(
        bits=int(cfg.get("ring.bias_dac.bits")), v_min_v=b1(vals["ring.bias_dac.v_min_v"]),
        v_max_v=b1(vals["ring.bias_dac.v_max_v"]),
        update_period_ticks=_period_ticks(cfg, "ring.bias_dac.update_rate_hz", default=1),
        slew_v_per_s=b1(g("ring.bias_dac.slew_v_per_s", B1, np.inf, "inf (unlimited)")),
        settle_tau_s=b1(g("ring.bias_dac.settle_tau_s", B1, 0.0, "0 (no filter)")),
        noise_rms_codes=b1(g("ring.bias_dac.noise_rms_codes", B1, 0.0)))
    if np.any(bias.v_max_v <= bias.v_min_v):
        raise RealizeError("ring.bias_dac: v_max_v must exceed v_min_v")
    monitor = _monitor(cfg, vals, timing, g, bn, b1, BN, B1)
    temps = _temp_sensors(cfg, layout, sd["device"], lat_scale, B, timing.dt_s)
    datapath = DatapathParams(
        bitrate_hz=b1(g("datapath.bitrate_gbps", B1, np.nan, "nan (absent)")),
        bitstat_window_bits=int(cfg.get("datapath.bitstat_window_bits") or 0),
        error_count_period_ticks=_period_ticks(cfg, "datapath.error_count_period_s", default=0, rate=False),
        transition_density=b1(g("datapath.transition_density", B1, np.nan, "nan (absent)")),
        driver_output_resistance_ohm=b1(g("driver.output_resistance_ohm", B1, np.nan, "nan (absent)")),
        driver_static_power_w=b1(g("driver.static_power_mw", B1, 0.0)),
        driver_energy_per_transition_j=b1(g("driver.energy_per_transition_fj", B1, 0.0)),
        swing_pp_v=bn(g("driver.swing_pp_v", BN, np.nan, "nan (absent)")),
        level_convention=switches["driver.level_convention"].copy(),
        rx_sensitivity_oma_w=bn(g("datapath.rx_sensitivity_oma_dbm", BN, np.nan, "nan (mrm.plant.datapath default)")),
        ber_ref=b1(g("datapath.ber_ref", B1, np.nan, "nan (mrm.plant.datapath default)")),
        link_transmission=b1(np.exp(-g("datapath.link_loss_db", B1, np.nan, "nan (mrm.plant.datapath default)"))),
        ber_floor=b1(g("datapath.ber_floor", B1, np.nan, "nan (mrm.plant.datapath default)")),
        bitstat_bw_hz=b1(g("datapath.bitstat_bw_ghz", B1, np.nan, "nan (mrm.plant.datapath default)")),
        bitstat_noise_a_per_rt_hz=b1(g("datapath.bitstat_noise_pa_per_rt_hz", B1, np.nan,
                                       "nan (the monitor TIA density)")))
    p_line = np.broadcast_to(np.asarray(vals["laser.power_per_line_dbm"], float), BL).copy()
    f_act = f_grid[None, :] + np.broadcast_to(np.asarray(g("laser.line_offset_ghz", BL, 0.0), float), BL)
    laser = LaserParams(
        line_freq_hz=f_act, line_freq_grid_hz=f_grid, power_w=p_line,
        input_transmission=b1(np.exp(-np.asarray(vals["optical_path.chip_input_loss_db"], float))),
        bus_loss_per_m=b1(g("optical_path.bus_loss_db_per_mm", B1, 0.0)),
        rin_per_hz=b1(g("laser.rin_db_per_hz", B1, 0.0)), linewidth_hz=b1(g("laser.linewidth_mhz", B1, 0.0)),
        direction=directions, **_line_meta(cfg, L))
    flags = FidelityFlags(
        self_heating=int(cfg.get("fidelity.self_heating") == "on"),
        thermo_optic_order=int(cfg.get("fidelity.thermo_optic_order")),
        thermal_model=int(tmodel == "rom"), junction_model=int(cfg.get("fidelity.junction_model") == "lut"),
        optics_model=int(cfg.get("fidelity.optics_model") == "tcmt"),
        backscatter=int(cfg.get("fidelity.backscatter") == "on"),
        shot_noise=int(cfg.get("fidelity.shot_noise") == "on"),
        pattern_noise=sch.enum_code("fidelity.pattern_noise", cfg.get("fidelity.pattern_noise")),
        data_modulation=int(cfg.get("fidelity.data_modulation") == "lut"),
        heater_anchor=int(cfg.get("fidelity.heater_anchor") == "fsr_integral"),
        heater_pwm_ripple=int(cfg.get("fidelity.heater_pwm_ripple") == "on"))

    # ---- 10. no silent nan/inf in the quantities every plant path uses, and a full scale for each world's law
    _assert_finite(optics, thermal, heater, laser, junction)

    # ---- 11. self-heating lag bound (07 P0-1 item 2)
    lag = _selfheat_lag_gain(cfg, optics, thermal, laser, layout, timing.dt_s, tmodel, system, tmod)
    diag["selfheat_lag_gain"] = lag
    if flags.self_heating and lag is not None and not lag < SELFHEAT_LAG_GAIN_MAX:
        raise RealizeError(f"self-heating: the lagged cross-ring optical heating has a per-substep gain of {lag:.3g} "
                           f">= {SELFHEAT_LAG_GAIN_MAX} at dt = {timing.dt_s:g} s (07 P0-1); reduce sim.dt_plant_s")
    bad = {k: v for k, v in resid.items() if not v <= 1e-12}      # nan (a non-finite input) fails too
    if bad:
        raise RealizeError(f"relations do not close to 1e-12 after realize(): {bad}")

    world_f = {k: _freeze(np.asarray(v)) for k, v in world.items()}
    vals_f = {k: _freeze(np.asarray(v)) if isinstance(v, np.ndarray) else v for k, v in vals.items()}
    vals_f.update({f"switch.{k}": _freeze(v) for k, v in switches.items()})
    return PlantParams(
        batch=B, n_rings=N, n_lines=L, optics=optics, thermal=thermal, junction=junction, heater=heater,
        bias_dac=bias, monitor=monitor, temp_sensors=temps, datapath=datapath, laser=laser, layout=layout,
        timing=timing, flags=flags, world=world_f, world_choices=world_choices,
        seeds={k: _freeze(v) for k, v in sd.items()}, config_hash=cfg.hash(), values=vals_f,
        derivations=dict(ctx.derivations), diagnostics=diag, thermal_inputs=tin)


#: fields that may legitimately hold nan (absent optional inputs, or not used by the selected model)
_NAN_ALLOWED = {
    "optics": {"alpha_fca_per_m"},
    "junction": {"v_bi_v", "depletion_exponent", "depletion_a_m", "loss_ratio_q"},
    "heater": {"full_scale_v", "full_scale_a", "full_scale_w", "pwm_period_s"},
}
#: heater driver law code -> the full-scale field it needs (codes in schema order: voltage, current, power, dac_pwm)
_LAW_FULL_SCALE = {0: "full_scale_v", 1: "full_scale_a", 2: "full_scale_w", 3: "full_scale_v"}


def _assert_finite(optics: OpticsParams, thermal: ThermalParams, heater: HeaterParams, laser: LaserParams,
                   junction: JunctionParams) -> None:
    """Raise RealizeError when a value the plant always uses is nan (or +-inf where a finite number is needed),
    or when a world's heater driver law has no full scale (a per-world switch reached by a hypothesis)."""
    bad = []
    for gname, grp in (("optics", optics), ("thermal", thermal), ("laser", laser), ("junction", junction)):
        for f in fields(grp):
            v = getattr(grp, f.name)
            if (isinstance(v, np.ndarray) and v.dtype.kind in "fc" and f.name not in _NAN_ALLOWED.get(gname, ())
                    and not np.all(np.isfinite(v))):
                bad.append(f"{gname}.{f.name}")
    for name in ("p_fsr_w", "r_cold_ohm", "r_ref_temp_k", "tcr_per_k"):
        if not np.all(np.isfinite(getattr(heater, name))):
            bad.append(f"heater.{name}")
    for code, fs in _LAW_FULL_SCALE.items():
        rows = heater.driver_mode[:, 0] == code
        if rows.any() and not np.all(np.isfinite(getattr(heater, fs)[rows])):
            bad.append(f"heater.{fs} (needed by driver_mode code {code} in {int(rows.sum())} worlds)")
    if bad:
        raise RealizeError(f"non-finite values after realize(): {', '.join(bad)}; check the inputs of these "
                           f"quantities (an uncertainty draw, a hypothesis choice or a missing key)")


def _si_values(cfg: "Config", arr: dict, lam_line: np.ndarray) -> tuple[dict, set]:
    """SI values of every realizable leaf (the arrays of ``arr`` and the global numbers) and the line and cold
    resonance wavelengths, with the set of given members for the relation solvers (step 5 of realize)."""
    vals: dict[str, Any] = {}
    for p, a in arr.items():
        vals[p] = to_si(p, a)
    for path, leaf in cfg.leaves.items():
        if path in vals or leaf.is_free:
            continue
        key = sch.lookup(path)
        if key is not None and key.type in ("float", "int") and key.level == "global" and not isinstance(
                leaf.value, list):
            vals[path] = to_si(path, leaf.value)
    vals["derived.optics.lambda_line_m"] = lam_line
    vals["derived.optics.lambda_res_cold_m"] = lam_line - vals[rel.P_OFFSET]
    given = {p for p in vals if p in cfg.leaves and not cfg.leaves[p].is_free}
    given |= {"derived.optics.lambda_line_m", "derived.optics.lambda_res_cold_m"}
    return vals, given


#: the channel-plan leaves the line wavelengths come from (07 P1-14: a derived value carries its inputs'
#: provenance). channel_plan.ring_to_line is left out on purpose: it is the design convention every per-ring value is
#: stated against (each ring is designed relative to its own line, ring.optics.resonance_offset_nm), and it only
#: permutes the plan's wavelengths over the rings
LINE_PLAN_INPUTS = ("channel_plan.lines", "channel_plan.center_thz", "channel_plan.spacing_ghz", "channel_plan.n_lines")


def _line_derivations(cfg: "Config", ctx: rel.RelationContext) -> None:
    """Record how the line and cold-resonance wavelengths are made (``mrm params explain``, provenance)."""
    ins = ("channel_plan.lines",) if cfg.get("channel_plan.lines") else LINE_PLAN_INPUTS[1:]
    ctx.derivations["derived.optics.lambda_line_m"] = rel.Derivation(
        "derived.optics.lambda_line_m", "channel_plan", ins,
        "lambda_line = c/f of the ring's line (channel_plan.ring_to_line); f = center + (k - (L - 1)/2) spacing, or "
        "the channel_plan.lines table")
    ctx.derivations["derived.optics.lambda_res_cold_m"] = rel.Derivation(
        "derived.optics.lambda_res_cold_m", "channel_plan", ("derived.optics.lambda_line_m", rel.P_OFFSET),
        "lambda_res = lambda_line - resonance_offset (the cold resonance, 07 P1-6)")


# ------------------------------------------------------------------------------------------------ variation

def _corr_chol(layout: LayoutParams, corr_um) -> np.ndarray | None:
    if not corr_um:
        return None
    d = layout.ring_distance_m * 1e6
    c = np.exp(-d / float(corr_um))
    return np.linalg.cholesky(c + 1e-12 * np.eye(c.shape[0]))


def _jac_log_derivative(jac: dict, out: str, coeffs, dims_in, leaf: np.ndarray) -> np.ndarray:
    """d ln(value)/dp per input of an 'exp' Jacobian output, shape (n_inputs, ...) broadcastable to the leaf.

    From the table's stored central-difference evaluations (``evaluations.<input>.plus/minus.<name>``, both > 0):
    (ln v+ - ln v-)/(2 step), the exact log-slope of an exponential (extract_modes handoff: the bend radiation's
    secant overstates the slab slope 4.8x). An input without positive evaluations uses J/leaf (the same first-order
    slope at the leaf's value); a zero coefficient stays zero."""
    ev = jac.get("evaluations") if isinstance(jac.get("evaluations"), dict) else {}
    name = _JAC_EVAL_NAME.get(out)
    leaf = np.asarray(leaf, float)
    rows = []
    for c, dn in zip(coeffs, dims_in):
        c = float(c)
        e = ev.get(dn) if name else None
        vp = vm = None
        if isinstance(e, dict) and isinstance(e.get("plus"), dict) and isinstance(e.get("minus"), dict):
            vp, vm, st = e["plus"].get(name), e["minus"].get(name), e.get("step_nm")
        if c == 0.0:
            rows.append(np.zeros_like(leaf))
        elif vp is not None and vm is not None and st and float(vp) > 0.0 and float(vm) > 0.0:
            rows.append(np.full_like(leaf, (math.log(float(vp)) - math.log(float(vm))) / (2.0 * float(st))))
        else:
            if np.any(leaf <= 0.0):
                raise RealizeError(f"Jacobian output {out}: the leaf must be > 0 for the exponential law")
            rows.append(c / leaf)
    return np.stack(rows)


def _apply_geometry(cfg, arr, tables, seeds, layout, B, N, diag) -> dict:
    """Draw the correlated geometry and map it through the Jacobians (07 P1-16). Moves the given leaves in ``arr``
    and returns the per-device coupling moves {'derived.geometry.dln_kappa2_<through|drop>': (B, N) d ln K^2 at the
    device's cold resonance} for the R4_q geometry branch ({} without a geometry)."""
    g = cfg.geometry
    if not g:
        return {}
    dg: dict[str, np.ndarray] = {}
    for dim in _GEOM_SLOT:
        spec = g.get(dim)
        if not spec:
            continue
        name = f"variation.geometry.{dim}"
        zd = _normal(seeds, int(_rng.Stream.VARIATION), name, 1, slot=0)
        zr = _normal(seeds, int(_rng.Stream.VARIATION), name, N, slot=1)
        ch = _corr_chol(layout, spec.get("corr_length_um"))
        if ch is not None:
            zr = zr @ ch.T
        dg[dim] = float(spec.get("die_sigma_nm", 0.0)) * zd + float(spec.get("ring_sigma_nm", 0.0)) * zr
    diag["geometry_draws_nm"] = {k: v.copy() for k, v in dg.items()}
    if g.get("width_closes_gaps") and "width" in dg:
        # the coupler Jacobians are partial derivatives at fixed edge-to-edge gaps: a lithographic width error of
        # every waveguide moves both edges of each gap, so it closes the gaps by the same amount (coupling artefact
        # note: total = d/dw - d/dgap); the gap draws are then the gaps' own error on top of it
        for gd in ("gap_through", "gap_drop"):
            dg[gd] = dg.get(gd, 0.0) - dg["width"]
    kap = {t: np.zeros((B, N)) for t in _KAPPA2_DISPERSION}
    dlam_res = np.zeros((B, N))                       # geometry move of the cold resonance (nm)
    jac_paths = g.get("jacobians", ["ring.geometry.jacobian"])
    if isinstance(jac_paths, str):
        jac_paths = [jac_paths]
    for jp in jac_paths:
        per_world = tables.get(jp) or [cfg.get(jp)] * B
        groups: dict[int, list[int]] = {}
        objs: dict[int, Any] = {}
        for b, t in enumerate(per_world):
            groups.setdefault(id(t), []).append(b)
            objs[id(t)] = t
        for oid, bs in groups.items():
            jac = objs[oid]
            if not isinstance(jac, dict) or "d" not in jac or "of" not in jac:
                raise RealizeError(f"{jp}: a Jacobian is {{d: [inputs], of: {{output: [one per input]}}}}")
            dims = []
            for dn in jac["d"]:
                if dn not in _JAC_DIMS:
                    raise RealizeError(f"{jp}: unknown Jacobian input {dn!r}; known {sorted(_JAC_DIMS)}")
                dims.append(_JAC_DIMS[dn])
            bsel = np.asarray(bs)
            for out, coeffs in jac["of"].items():
                if out not in JACOBIAN_TARGETS:
                    raise RealizeError(f"{jp}: unknown Jacobian output {out!r}; known {sorted(JACOBIAN_TARGETS)}")
                if len(coeffs) != len(dims):
                    raise RealizeError(f"{jp}.of.{out}: {len(coeffs)} coefficients for {len(dims)} inputs")
                tgt, how, sign = JACOBIAN_TARGETS[out]
                delta = np.zeros((len(bs), N))
                for c, dim in zip(coeffs, dims):
                    if dim in dg and c:
                        delta += float(c) * np.broadcast_to(dg[dim], (B, N))[bsel]
                if not np.any(delta):
                    continue
                if how == "kappa2":
                    kap[tgt][bsel] += sign * delta
                    continue
                if out in ("lambda_res_nm_per_nm", "resonance_offset_nm_per_nm"):
                    dlam_res[bsel] -= sign * delta    # offset = line - resonance
                if tgt not in arr:
                    raise RealizeError(f"{jp}.of.{out}: target {tgt} is not a given input of this config (it is "
                                       f"free or absent); the geometry cannot move it (07 P1-16)")
                if arr[tgt].shape[1] != N:
                    arr[tgt] = np.broadcast_to(arr[tgt], (B, N)).copy()
                if how == "offset":
                    arr[tgt][bsel] += sign * delta
                elif how == "lnscale":
                    arr[tgt][bsel] *= np.exp(sign * delta)
                else:                                 # 'exp': the secant as a log-derivative (positive values)
                    d = _jac_log_derivative(jac, out, coeffs, jac["d"], arr[tgt][bsel])
                    ln = np.zeros((len(bs), N))
                    for k, dim in enumerate(dims):
                        if dim in dg:
                            ln += d[k] * np.broadcast_to(dg[dim], (B, N))[bsel]
                    arr[tgt][bsel] *= np.exp(sign * ln)
    out_k = {}
    for t, disp in _KAPPA2_DISPERSION.items():
        # the anchors hold at each device's own cold resonance, which the geometry moved by dlam_res: the coupler's
        # K^2 there differs from the nominal device's by its dispersion as well (no double count at the line:
        # mrm.plant.phase evaluates K^2(lambda_line) = K^2(lambda_res) exp(D (lambda_line - lambda_res)))
        dd = np.broadcast_to(np.asarray(arr.get(disp, 0.0), float), (B, N))
        out_k[t] = kap[t] + dd * dlam_res
    diag["geometry_dlambda_res_nm"] = dlam_res.copy()
    return out_k


def _apply_variation(cfg, arr, seeds, layout, B, N, L, diag) -> None:
    for path in sorted(cfg.variation):
        e = cfg.variation[path].fields
        key = sch.lookup(path)
        lvl = e.get("level", key.level)
        dist = e.get("dist", "normal")
        mode = e.get("mode", "scale" if (dist == "lognormal" or "rel_sigma" in e) else "offset")
        name = f"variation.{path}"
        base = arr[path]
        if lvl == "ring" and base.shape[1] != N:
            base = np.broadcast_to(base, (B, N)).copy()
        if lvl == "line" and base.shape[1] != L:
            base = np.broadcast_to(base, (B, L)).copy()
        k = {"die": 1, "ring": N, "line": L}[lvl]
        sd_die = float(e.get("die_sigma", 0.0))
        sd_el = float(e.get("ring_sigma", e.get("line_sigma", 0.0)))
        rel_sd = float(e.get("rel_sigma", 0.0))
        stream = int(_rng.Stream.VARIATION)
        rng_lo, rng_hi = (cfg.leaves[path].range or (None, None)) if e.get("truncate", True) else (None, None)
        single = (sd_die == 0.0) + (sd_el == 0.0 and rel_sd == 0.0) == 1 and not e.get("corr_length_um")
        if dist == "uniform":
            zd = math.sqrt(3.0) * (2.0 * _uniform(seeds, stream, name, 1, 0) - 1.0)
            ze = math.sqrt(3.0) * (2.0 * _uniform(seeds, stream, name, k, 1) - 1.0)
        else:
            ud = _uniform(seeds, stream, name, 1, 0)
            ue = _uniform(seeds, stream, name, k, 1)
            if single and mode == "offset" and rng_lo is not None and dist == "normal" and (sd_die == 0.0 or lvl == "die"):
                # exact truncation of a single normal component to the leaf range (inverse CDF)
                sig = sd_el if sd_el else sd_die
                u = ue
                a = (rng_lo - base) / sig
                b = (rng_hi - base) / sig
                u = ndtr(a) + u * (ndtr(b) - ndtr(a))
                arr[path] = base + sig * ndtri(np.clip(u, 1e-300, 1 - 1e-16))
                continue
            zd, ze = ndtri(ud), ndtri(ue)
        if lvl == "ring":
            ch = _corr_chol(layout, e.get("corr_length_um"))
            if ch is not None:
                ze = ze @ ch.T
        if rel_sd:
            new = base * (1.0 + rel_sd * ze) * (np.exp(sd_die * zd) if sd_die else 1.0)
        else:
            delta = sd_die * zd + sd_el * ze
            if mode == "offset":
                new = base + delta
            else:
                new = base * np.exp(delta)
        if rng_lo is not None:
            clipped = np.clip(new, rng_lo, rng_hi)
            n_clip = int(np.count_nonzero(clipped != new))
            if n_clip:
                diag["clipped"][path] = n_clip / new.size
            new = clipped
        arr[path] = new


# ------------------------------------------------------------------------------------------------ group builders

def _backscatter_rb(cfg, vals, seeds, var_mode, B: int, N: int) -> np.ndarray:
    """Complex backscatter per order (07 P2-2): beta_n complex Gaussian with E|beta_n|^2 = (splitting/2)^2 and a
    uniform phase (per device, stream VARIATION); rb_n = 2 pi beta_n / FSR (mrm.plant.optics.beta_to_rb).
    With variation off: beta_n = splitting/2 (real) for every order. Zeros when fidelity.backscatter is off."""
    m = int(cfg.get("ring.optics.backscatter_orders"))
    out = np.zeros((B, N, m), dtype=np.complex128)
    if cfg.get("fidelity.backscatter") != "on" or "ring.optics.backscatter_splitting_ghz" not in vals:
        return out
    half = np.broadcast_to(np.asarray(vals["ring.optics.backscatter_splitting_ghz"], float), (B, N)) / 2.0
    fsr = np.broadcast_to(np.asarray(vals["derived.optics.fsr_hz"], float), (B, N))
    if var_mode == "off":
        beta = np.repeat(half[:, :, None], m, axis=2).astype(np.complex128)
    else:
        stream = int(_rng.Stream.VARIATION)
        zr = _normal(seeds, stream, "ring.optics.backscatter", N * m, 0).reshape(B, N, m)
        zi = _normal(seeds, stream, "ring.optics.backscatter", N * m, 1).reshape(B, N, m)
        beta = half[:, :, None] * (zr + 1j * zi) / math.sqrt(2.0)
    out[...] = 2.0 * math.pi * beta / fsr[:, :, None]
    return _freeze(out)


def _period_ticks(cfg, path: str, default: int, rate: bool = True) -> int:
    v = cfg.get(path)
    if v is None:
        return default
    dt = float(cfg.si("sim.dt_plant_s"))
    x = float(to_si(path, v))
    if x <= 0:
        return default
    t = (1.0 / x if rate else x) / dt
    return int(round(t))


def _junction(cfg, vals, g, bn, b1, BN, B1) -> JunctionParams:
    lut = cfg.get("fidelity.junction_model") == "lut"
    empty = np.zeros(0)
    fill = bn(vals["ring.junction.fill_fraction"])
    if lut:
        from mrm.params.config import resolve_from
        data, _ = resolve_from(f"{cfg.get('ring.junction.lut_file')}:v_reverse_v", None, cfg.data_root)
        grid = np.asarray(data, float)

        def col(k):
            d, _ = resolve_from(f"{cfg.get('ring.junction.lut_file')}:{k}", None, cfg.data_root)
            return np.asarray(d, float)
        lut_dn, lut_a, lut_c, lut_r = col("dneff"), col("alpha_fca_per_m"), col("capacitance_f_per_m"), col(
            "r_series_ohm_m")
        # derived report-only quantities (07 P1-7): slope of the shift at 0 V from the LUT
        dndv0 = np.interp(0.0, grid, np.gradient(lut_dn, grid))
        lam = vals["derived.optics.lambda_res_cold_m"]
        eff0 = bn(fill * lam * dndv0 / vals[rel.P_NG])
        vbi = np.full(B1, np.nan)
        mexp = np.full(B1, np.nan)
        a_m = np.full(BN, np.nan)
    else:
        grid = lut_dn = lut_a = lut_c = lut_r = empty
        eff0 = bn(vals["ring.junction.eff_at_0v_pm_per_v"])
        vbi = b1(vals["ring.junction.v_bi_v"])
        mexp = b1(vals["ring.junction.depletion_exponent"])
        a_m = eff0 / (mexp * vbi ** (mexp - 1.0))
    return JunctionParams(
        model=int(lut), fill_fraction=fill, depletion_a_m=a_m, v_bi_v=vbi, depletion_exponent=mexp,
        eff_at_0v_m_per_v=eff0, dqi_dv_frac_per_v=bn(g("ring.junction.dqi_dv_frac_per_v", BN, 0.0)),
        capacitance_at_0v_f=bn(g("ring.junction.capacitance_at_0v_ff", BN, 0.0)),
        series_resistance_ohm=bn(g("ring.junction.series_resistance_ohm", BN, 0.0)),
        lut_v_reverse_v=grid, lut_dneff=lut_dn, lut_alpha_fca_per_m=lut_a, lut_capacitance_f_per_m=lut_c,
        lut_r_series_ohm_m=lut_r, loss_ratio_q=b1(g("ring.junction.loss_ratio_q", B1, np.nan, "nan (absent)")))


def _heater(cfg, vals, switches, g, bn, b1, BN, B1, seeds) -> HeaterParams:
    bits = int(cfg.get("ring.heater.dac_bits"))
    cmin = cfg.get("ring.heater.code_min")
    cmax = cfg.get("ring.heater.code_max")
    cmin = 0 if cmin is None else int(cmin)
    cmax = (1 << bits) - 1 if cmax is None else int(cmax)
    if not (0 <= cmin < cmax <= (1 << bits) - 1):
        raise RealizeError(f"ring.heater: need 0 <= code_min < code_max <= 2^dac_bits - 1 (got {cmin}, {cmax})")
    mu = cfg.get("ring.heater.dac_max_update_hz")
    B, N = BN
    stream = int(_rng.Stream.VARIATION)
    bit_z = _normal(seeds, stream, "ring.heater.dac_bit_errors", N * bits, 0).reshape(B, N, bits)
    bow = 2.0 * _uniform(seeds, stream, "ring.heater.dac_inl_bow", N, 0) - 1.0
    pmax = cfg.get("array.heater_total_power_max_w")
    imax = cfg.get("array.heater_supply_current_max_a")
    return HeaterParams(
        p_fsr_w=bn(vals[rel.P_PFSR]), r_cold_ohm=bn(vals["ring.heater.r_cold_ohm"]),
        r_ref_temp_k=b1(vals["ring.heater.r_ref_temp_c"]), tcr_per_k=b1(vals["ring.heater.tcr_per_k"]),
        driver_mode=switches["ring.heater.driver_mode"].copy(),
        full_scale_v=b1(g("ring.heater.full_scale_v", B1, np.nan, "nan (not this driver law)")),
        full_scale_a=b1(g("ring.heater.full_scale_a", B1, np.nan, "nan (not this driver law)")),
        full_scale_w=b1(g("ring.heater.full_scale_w", B1, np.nan, "nan (not this driver law)")),
        dac_bits=bits, code_min=cmin, code_max=cmax,
        slew_codes_per_s=b1(g("ring.heater.slew_codes_per_s", B1, np.inf, "inf (07 P1-11: no demand-based limit)")),
        settle_tau_s=b1(g("ring.heater.settle_tau_s", B1, 0.0, "0 (no filter)")),
        dac_inl_lsb=b1(g("ring.heater.dac_inl_lsb", B1, 0.0)), dac_dnl_lsb=b1(g("ring.heater.dac_dnl_lsb", B1, 0.0)),
        compliance_v=b1(g("ring.heater.driver_compliance_v", B1, np.inf, "inf (no compliance limit)")),
        pwm_period_s=b1(g("ring.heater.pwm_period_s", B1, np.nan, "nan (not dac_pwm)")),
        max_update_hz=float(to_si("dac_max_update_hz", mu)) if mu is not None else math.inf,
        readback_present=bool(cfg.get("ring.heater.readback.present")),
        pwm_bits=int(cfg.get("ring.heater.pwm_bits") or 0),
        supply_power_max_w=float(to_si("max_w", pmax)) if pmax is not None else math.inf,
        supply_current_max_a=float(to_si("max_a", imax)) if imax is not None else math.inf,
        dac_bit_err_z=bit_z, inl_bow_u=bow,
        readback_adc_bits=int(cfg.get("ring.heater.readback.adc_bits") or 0),
        readback_v_full_scale_v=b1(g("ring.heater.readback.v_full_scale_v", B1, np.nan, "nan (absent)")),
        readback_i_full_scale_a=b1(g("ring.heater.readback.i_full_scale_a", B1, np.nan, "nan (absent)")),
        readback_noise_rms_codes=b1(g("ring.heater.readback.noise_rms_codes", B1, 0.0)),
        readback_average=bool(cfg.get("ring.heater.readback.average", False)))


def _monitor(cfg, vals, timing, g, bn, b1, BN, B1) -> MonitorParams:
    afe = str(cfg.get("ring.monitor.afe_type"))
    pga = cfg.get("ring.monitor.pga_gains")
    return MonitorParams(
        responsivity_a_per_w=bn(vals["ring.monitor.responsivity_a_per_w"]),
        gain_scale=b1(g("ring.monitor.gain_scale", B1, 1.0, "1 (no nuisance scale)")),
        dark_i0_a=b1(g("ring.monitor.dark_i0_na", B1, 0.0)),
        dark_t0_k=b1(vals["ring.monitor.dark_t0_c"]) if "ring.monitor.dark_t0_c" in vals else _need_absent(
            vals, "ring.monitor.dark_i0_na", "ring.monitor.dark_t0_c", B1),
        # absent E_a: 0.37 eV (inferred, mrm.materials.ge_pd_dark_ea_ev, 02 section 10), not 0, so the dark current
        # keeps the Arrhenius law of 07 P1-9 / D20 (plant_sensing handoff section 4)
        dark_activation_j=b1(g("ring.monitor.dark_activation_ev", B1, _dark_ea_default_j(),
                               "0.37 eV (inferred, mrm.materials.ge_pd_dark_ea_ev)")),
        background_fraction=bn(g("ring.monitor.background_fraction", BN, 0.0)),
        afe_type=int(afe == "integrate_dump"),
        afe_tau_s=float(cfg.si("ring.monitor.afe_tau_s", 0.0)),
        integration_window_ticks=timing.integration_window_ticks, adc_period_ticks=timing.adc_period_ticks,
        adc_bits=int(cfg.get("ring.monitor.adc_bits")), adc_full_scale_a=b1(vals["ring.monitor.adc_full_scale_ua"]),
        adc_offset_codes=bn(g("ring.monitor.adc_offset_codes", BN, 0.0)),
        adc_gain_error_frac=bn(g("ring.monitor.adc_gain_error_frac", BN, 0.0)),
        adc_inl_lsb=b1(g("ring.monitor.adc_inl_lsb", B1, 0.0)), adc_dnl_lsb=b1(g("ring.monitor.adc_dnl_lsb", B1, 0.0)),
        input_noise_a_per_rt_hz=b1(g("ring.monitor.input_noise_pa_per_rt_hz", B1, 0.0)),
        flicker_corner_hz=b1(g("ring.monitor.flicker_corner_hz", B1, 0.0)),
        adc_ref_tempco_per_k=b1(g("ring.monitor.adc_ref_tempco_ppm_per_k", B1, 0.0)),
        pga_gains=np.asarray(pga if pga else [], dtype=float),
        channels_per_adc=int(cfg.get("ring.monitor.adc_mux.channels_per_adc") or 1),
        mux_order=str(cfg.get("ring.monitor.adc_mux.order") or "round_robin"),
        baseline_a=b1(g("ring.monitor.baseline_ua", B1, 0.0, "0 (no analog baseline subtraction)")),
        stuck_flagged=bool(cfg.get("ring.monitor.stuck_flagged", True)),
        flicker_band_min_hz=float(cfg.get("ring.monitor.flicker_band_min_hz") or 1e-3),
        ref_tap_present=bool(cfg.get("ring.monitor.reference_tap.present", False)),
        ref_tap_coupling_frac=b1(g("ring.monitor.reference_tap.coupling_frac", B1, np.nan, "nan (no reference tap)")),
        ref_tap_responsivity_a_per_w=b1(g("ring.monitor.reference_tap.responsivity_a_per_w", B1, np.nan,
                                          "nan (the drop PD's)")),
        ref_tap_dark_i0_a=b1(g("ring.monitor.reference_tap.dark_i0_na", B1, np.nan, "nan (the drop PD's)")),
        adc_ref_site=str(cfg.get("ring.monitor.adc_ref_site") or "package_base"))


def _need_absent(vals, given: str, missing: str, shape) -> np.ndarray:
    """nan placeholder for ``missing``, allowed only while ``given`` (which needs it) is absent too."""
    if given in vals:
        raise RealizeError(f"{given} is given, so {missing} is required")
    return np.full(shape, np.nan)


def _timing(cfg, vals, lat_scale, diag) -> TimingParams:
    dt = float(cfg.si("sim.dt_plant_s"))
    adc = _period_ticks(cfg, "ring.monitor.adc_rate_hz", default=0)
    ctrl = _period_ticks(cfg, "controller_interface.update_rate_hz", default=0)
    afe = str(cfg.get("ring.monitor.afe_type"))
    win = _period_ticks(cfg, "ring.monitor.integration_window_s", default=0, rate=False) if afe == "integrate_dump" else 0
    errs = []
    if adc < 1:
        errs.append("ring.monitor.adc_rate_hz must give an ADC period of at least one tick")
    if afe == "integrate_dump":
        if win < 1:
            errs.append("ring.monitor.integration_window_s must be at least one tick")
        if win > adc:
            errs.append(f"ring.monitor.integration_window_s ({win} ticks) must fit in one ADC period ({adc} ticks)")
        cpa = int(cfg.get("ring.monitor.adc_mux.channels_per_adc") or 1)
        if (str(cfg.get("ring.monitor.adc_mux.order") or "round_robin") == "round_robin" and cpa > 1
                and cpa * win > adc >= 1):
            errs.append(f"ring.monitor.adc_mux: {cpa} rings take turns on one integrating converter, so their "
                        f"{win}-tick windows need {cpa * win} ticks per ADC period ({adc} ticks; 07 P0-7)")
    if (str(cfg.get("ring.monitor.adc_ref_site") or "package_base") == "monitor_pd"
            and int(cfg.get("ring.monitor.adc_mux.channels_per_adc") or 1) > 1):
        errs.append("ring.monitor.adc_ref_site monitor_pd puts each ring's converter at its PD, which needs "
                    "ring.monitor.adc_mux.channels_per_adc = 1")
    mu = cfg.get("ring.heater.dac_max_update_hz")
    if ctrl and mu:
        min_period = 1.0 / float(to_si("dac_max_update_hz", mu))
        if ctrl * dt < min_period * (1 - 1e-9):
            errs.append(f"controller period {ctrl * dt:g} s is shorter than the heater DAC's minimum update period "
                        f"{min_period:g} s")
    bias_rate = cfg.get("ring.bias_dac.update_rate_hz")
    if bias_rate is not None and _period_ticks(cfg, "ring.bias_dac.update_rate_hz", default=1) < 1:
        errs.append("ring.bias_dac.update_rate_hz must give a period of at least one tick")
    if errs:
        raise RealizeError("timing rules (07 P0-6): " + "; ".join(errs))

    def lat(ch):
        k = f"derived.timing.{ch}.latency_ticks"
        if k in vals:
            return np.broadcast_to(np.asarray(vals[k], np.int64), (lat_scale.shape[0], 1)).copy()
        return np.zeros((lat_scale.shape[0], 1), dtype=np.int64)
    ramp = cfg.get("sim.reset_ramp_s")
    return TimingParams(
        dt_s=dt, controller_period_ticks=ctrl, adc_period_ticks=adc, integration_window_ticks=win,
        monitor_latency_ticks=lat("tx.monitor"), heater_latency_ticks=lat("tx.heater"),
        bias_latency_ticks=lat("tx.bias"), readback_latency_ticks=lat("heater.readback"),
        readback_period_ticks=_period_ticks(cfg, "ring.heater.readback.rate_hz", default=0),
        error_count_latency_ticks=lat("rx.error_count"), latency_scale=lat_scale.copy(),
        reset_protocol=sch.enum_code("sim.reset_protocol", cfg.get("sim.reset_protocol")),
        reset_ramp_ticks=int(round(float(ramp) / dt)) if ramp else 0,
        substep_policy=sch.enum_code("sim.substep_policy", cfg.get("sim.substep_policy")),
        max_detuning_step_m=float(cfg.get("sim.max_detuning_step_pm") or 0.2) * 1e-12,
        min_samples_per_dither_period=int(cfg.get("sim.min_samples_per_dither_period") or 8))


_SENSOR_TYPES = ("diode", "rtd", "thermistor")


def _temp_sensors(cfg, layout: LayoutParams, seeds, lat_scale, B: int, dt: float) -> TempSensorParams:
    recs = cfg.get("temp_sensors") or []
    S = len(recs)

    def col(fname, unit_key, default=0.0):
        out = np.zeros((B, S))
        for i, r in enumerate(recs):
            v = r.get(fname)
            out[:, i] = float(to_si(unit_key, v)) if v is not None else default
        return out
    names, types, site_idx, attach, per, lat = [], [], [], [], [], np.zeros((B, S), dtype=np.int64)
    for i, r in enumerate(recs):
        names.append(str(r.get("name", f"t{i}")))
        types.append(_SENSOR_TYPES.index(r.get("type", "diode")))
        s = r.get("site")
        if s is not None:
            if s not in layout.site_names:
                raise RealizeError(f"temp_sensors[{i}]: site {s!r} is not in layout.sites")
            site_idx.append(layout.site_names.index(s))
            attach.append("")
        else:
            site_idx.append(-1)
            attach.append(str(r.get("attach", "die")))
        rate = r.get("sample_rate_hz")
        per.append(int(round(1.0 / (float(rate) * dt))) if rate else 0)
        if r.get("latency_s") is not None:
            lat[:, i] = np.rint(float(r["latency_s"]) * lat_scale[:, 0] / dt).astype(np.int64)
    stream = int(_rng.Stream.VARIATION)
    zo = _normal(seeds, stream, "temp_sensors.offset", max(S, 1), 0)[:, :S]
    zg = _normal(seeds, stream, "temp_sensors.gain_error", max(S, 1), 0)[:, :S]
    return TempSensorParams(
        names=tuple(names), type=np.asarray(types, dtype=np.int64), site_index=np.asarray(site_idx, dtype=np.int64),
        attach=tuple(attach), sample_period_ticks=np.asarray(per, dtype=np.int64), latency_ticks=lat,
        noise_rms_k=col("noise_rms_k", "noise_rms_k"), flicker_corner_hz=col("flicker_corner_hz", "flicker_corner_hz"),
        resolution_k=col("resolution_k", "resolution_k"),
        offset_k=col("offset_sigma_k", "offset_sigma_k") * zo,
        gain_error_frac=col("gain_error_sigma_frac", "gain_error_sigma_frac") * zg,
        self_heating_k=col("self_heating_k", "self_heating_k"),
        adc_bits=np.asarray([int(r.get("adc_bits") or 0) for r in recs], dtype=np.int64),
        code_zero_k=np.asarray([float(to_si("code_zero_c", r["code_zero_c"])) if r.get("code_zero_c") is not None
                                else np.nan for r in recs], dtype=float))


_INTERIM_FIELDS_SI = ("tau_fast_s", "share_fast_frac", "tau_slow_s", "tail_tau_s", "tail_share_frac",
                      "selfheat_tau_s", "selfheat_share_frac", "heater_self_ratio", "heater_node_tau_s",
                      "xt_nearest_frac", "xt_floor_frac", "xt_next_frac", "xt_tau_ref_s", "xt_tau_min_s",
                      "xt_tau_far_s", "site_floor_frac")
_INTERIM_FIELDS_UM = ("xt_dist_ref_um", "xt_far_um", "site_ref_um")


def _assemble_thermal(tmod, cfg, tin: ThermalInputs, vals: dict, N: int, diag: dict):
    """Call ``mrm.plant.thermal.assemble`` (its published signature, docs/handoff/plant_thermal.md):
    ``assemble(rom: ThermalRom | InterimRom, layout, n_rings, scales: ThermalScales, *, temp_sensors, ring_pitch_um)``.

    Interim: an InterimRom built from the ring.thermal.* keys (SI; *_um fields in um as that API expects) and the
    R2 gain Z_heater->ring (B, N). ROM: load_rom (sha256 verified) and a global dc_scale per world =
    mean_i(Z_heater->ring,i)/Z_FEM(heater->ring), the 07 P1-2 rom_dc_scale (warned outside [0.5, 2]).
    Returns (system, Z_opt->ring (B, N) for the ROM model or None).
    """
    B = tin.z_heater_ring_k_per_w.shape[0]
    sc_fields = {f.name for f in fields(tmod.ThermalScales)}
    unknown = set(tin.scales) - sc_fields
    if unknown:
        raise RealizeError(f"uncertainty.thermal.{sorted(unknown)} are not ThermalScales fields {sorted(sc_fields)}")
    sc = {k: np.asarray(v, float) for k, v in tin.scales.items()}
    sc["scaling_hypothesis"] = list(tin.scaling_hypothesis)
    z_opt_rom = None
    if tin.model == "interim":
        kw = {"z_heater_ring_k_per_w": np.asarray(tin.z_heater_ring_k_per_w),
              "z_ratio_opt_heater": np.asarray(vals["ring.thermal.z_ratio_opt_heater"], float).reshape(-1)}
        for f in _INTERIM_FIELDS_SI:
            p = "ring.thermal." + f
            if p in vals:
                kw[f] = np.broadcast_to(np.asarray(vals[p], float), (B, 1))[:, 0].copy()
        for f in _INTERIM_FIELDS_UM:
            p = "ring.thermal." + f
            if p in vals:
                # the per-world value (a hypothesis may set it); InterimRom takes one float, so worlds must agree
                v = np.asarray(vals[p], float).reshape(-1) / 1e-6          # SI m -> um (the InterimRom unit)
                if np.max(v) - np.min(v) > 1e-9 * max(float(np.max(np.abs(v))), 1.0):
                    raise RealizeError(f"{p} differs between worlds ({np.min(v):g}..{np.max(v):g} um); the interim "
                                       f"thermal model takes one value per batch: pin the hypothesis that sets it")
                kw[f] = float(v[0])
            elif cfg.get(p) is not None:
                kw[f] = float(cfg.get(p))
        for f in ("pd_offset_um", "temp_sensor_offset_um", "driver_offset_um"):
            v = cfg.get("ring.thermal." + f)
            if v is not None:
                kw[f] = tuple(float(x) for x in v)
        kw["xt_tau_law"] = str(cfg.get("ring.thermal.xt_tau_law"))
        kw["crosstalk_dynamics"] = str(cfg.get("ring.thermal.crosstalk_dynamics"))
        # each ring's r_ref (R3, (B, N) in um): the crosstalk tau law takes heat-source-to-victim distances, the
        # centre distance less the ring radii (mrm.plant.thermal.crosstalk_tau_s)
        kw["ring_radius_um"] = np.broadcast_to(np.asarray(vals[rel.P_RADIUS], float), (B, N)) / 1e-6
        rom = tmod.InterimRom(**kw)
    else:
        rom = tmod.load_rom(tin.rom_path)
        si = list(rom.sources).index("heater")
        oi = list(rom.observers).index("ring_mode_weighted")
        z_fem = float(np.sum(rom.self_res[si, oi]))
        dc = np.mean(np.asarray(tin.z_heater_ring_k_per_w), axis=1) / z_fem
        diag["rom_dc_scale"] = dc.copy()
        if np.any((dc < 0.5) | (dc > 2.0)):
            import warnings
            warnings.warn(f"rom_dc_scale {dc.min():.3g}..{dc.max():.3g} is outside [0.5, 2] (07 P1-2): the FEM "
                          f"stack disagrees with the measured heater efficiency under this dlambda/dT", stacklevel=3)
        prev = sc.get("dc_scale")
        sc["dc_scale"] = dc if prev is None else dc * prev
    system = tmod.assemble(rom, tin.sites_um, N, tmod.ThermalScales(**sc), temp_sensors=tin.temp_sensors,
                           ring_pitch_um=tin.ring_pitch_um)
    if tin.model == "rom" and hasattr(tmod, "selfheat_dc_k_per_w"):
        z_opt_rom = np.broadcast_to(tmod.selfheat_dc_k_per_w(system), (B, N)).copy()
    return system, z_opt_rom


def _selfheat_lag_gain(cfg, optics: OpticsParams, thermal: ThermalParams, laser: LaserParams,
                       layout: LayoutParams, dt: float, tmodel: str, system, tmod=None) -> float | None:
    """Upper bound of the per-substep gain of the lagged cross-ring optical heating (07 P0-1 item 2).

    g_i = sum_{j != i} (1 - exp(-dt/tau_ij)) * x_ij * G_sh,j * max|dP_abs,j/dlambda|, with x_ij the DC crosstalk
    fraction of Z_opt, tau_ij the crosstalk pole (interim model of 07 P1-2), G_sh = s(T_lock) Z_opt and
    max|dP_abs/dlambda| = P_abs,max * (3 sqrt 3/8)/HWHM for the Lorentzian line (HWHM = lambda/(2 Q_L)).
    P_abs,max = eta * 4 kappa_e kappa_i/kappa^2 * P_in (on resonance), P_in = the largest line power at the chip.
    When the assembled system exists, ``mrm.plant.thermal.cross_optical_lag_gain(system, dt, dP/dT max)`` gives
    the exact sum over its cross-ring optical lags; otherwise the interim numbers are used (None for a ROM without
    a system).
    """
    N = optics.n_g.shape[1]
    if N < 2:
        return 0.0
    w = 2 * math.pi * C0 / optics.lambda_res_cold_m
    k_i, k_t, k_d = w / optics.q_intrinsic, w / optics.q_through, w / optics.q_drop
    k = k_i + k_t + k_d
    p_in = np.max(laser.power_w, axis=1, keepdims=True) * laser.input_transmission
    p_abs_max = optics.heat_fraction * 4.0 * k_t * k_i / (k * k) * p_in
    hwhm = optics.lambda_res_cold_m / (2.0 * optics.q_loaded)
    slope = p_abs_max * _LORENTZ_MAX_SLOPE / hwhm                    # (B,N) W/m
    g_sh = thermal.self_heating_gain_m_per_w                          # (B,N) m/W
    if system is not None and tmod is not None and hasattr(tmod, "cross_optical_lag_gain"):
        dpdt = slope * thermal.s_lock_m_per_k                         # (B,N) W/K
        return float(np.max(tmod.cross_optical_lag_gain(system, dt, dpdt)))
    if tmodel == "rom":
        return None
    t = thermal.inputs.interim
    d = layout.ring_distance_m                                      # (N,N) m
    pos = d[d > 0]
    if pos.size == 0:                                               # every ring on one site: no cross path
        return 0.0
    ranks = np.unique(np.round(pos / pos.min(), 9))
    near = np.isclose(d, pos.min()) & (d > 0)
    second = (np.isclose(d, ranks[1] * pos.min()) & (d > 0)) if ranks.size > 1 else np.zeros_like(near)

    def iv(name, default=None):                                     # (B,1,1) per-world interim value
        if name not in t:
            return default
        return np.asarray(t[name], float).reshape(-1, 1, 1)
    floor = iv("xt_floor_frac")
    frac = np.where(near[None], iv("xt_nearest_frac"), np.where(second[None], iv("xt_next_frac", floor), floor))
    # crosstalk pole: the plant's own law when importable (07 P1-2, 04 section 2.4: capped d^2 law, printed X2 table
    # when xt_tau_far_s is absent), else the d^2 near-field law with the given far value
    # m -> um for the plant's law (um API); the law takes heat-source-to-victim distances: the centre distance less
    # both ring radii (mrm.plant.thermal.crosstalk_tau_s)
    r_um = np.asarray(optics.radius_m, float) * 1e6                 # (B,N)
    d_um = np.maximum(d[None] * 1e6 - r_um[:, :, None] - r_um[:, None, :], 0.0)
    far_m = iv("xt_far_m")
    law = str(cfg.get("ring.thermal.xt_tau_law"))
    try:
        from mrm.plant.thermal import crosstalk_tau_s
    except ImportError:  # pragma: no cover - the thermal module is part of the package
        crosstalk_tau_s = None
    if crosstalk_tau_s is not None:
        far_s = iv("xt_tau_far_s")
        tau = crosstalk_tau_s(d_um, iv("xt_tau_ref_s"), float(np.asarray(t["xt_dist_ref_m"]).ravel()[0]) * 1e6,
                              iv("xt_tau_min_s"), far_s, float(far_m.ravel()[0]) * 1e6 if far_m is not None
                              else 100.0, law)                      # 100 um: InterimRom.xt_far_um default (07 P1-2)
    else:  # pragma: no cover
        tau = np.maximum(iv("xt_tau_min_s"), iv("xt_tau_ref_s") * (d_um * 1e-6 / iv("xt_dist_ref_m")) ** 2)
        if far_m is not None and "xt_tau_far_s" in t:
            tau = np.where(d_um * 1e-6 > far_m, iv("xt_tau_far_s"), tau)
    scale = thermal.inputs.scales.get("time_scale_slow")
    if scale is not None:
        tau = tau * np.asarray(scale, float).reshape(-1, 1, 1)
    xt_scale = thermal.inputs.scales.get("crosstalk_scale")
    if xt_scale is not None:
        frac = frac * np.asarray(xt_scale, float).reshape(-1, 1, 1)
    eye = np.eye(N, dtype=bool)[None]
    frac = np.where(eye, 0.0, frac)
    one_minus_a = -np.expm1(-dt / np.where(tau > 0, tau, np.inf))
    gain = (one_minus_a * frac * (g_sh * slope)[:, None, :]).sum(axis=2)
    return float(np.max(gain))
