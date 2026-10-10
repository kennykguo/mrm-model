"""Strict config schema: every key, its type, enum, level, unit and the fidelity rules (07 P1-15).

Format
------
Each key is a :class:`Key` with a dotted ``path``. Units are never stated here: they come from the key suffix
(:mod:`mrm.params.units`) and :func:`check_units` asserts that every suffix parses. The fields are:

- ``type``: ``float | int | bool | str | enum | list_float | list_int | list_str | table | map``.
- ``physical``: True for a claim about the hardware or the world. In a preset it must be a provenance leaf mapping
  ``{value, provenance, generation, source, ...}`` (07 P1-14). False for a setting (simulation step, experiment,
  metric thresholds, controller choices), which may be a bare value.
- ``level``: the batched shape after ``realize()``: ``die`` (B, 1), ``ring`` (B, N), ``line`` (B, L), or
  ``global`` (not batched). A ring- or line-level leaf may be a scalar or a list of N (L) values.
- ``required`` / ``required_if`` / ``forbidden_if``: fidelity rules. A condition is a tuple of
  ``(path, allowed_values)`` pairs that must all hold. Example (07 P1-15): ``thermal_model=rom`` requires
  ``rom_file`` and forbids ``tau_fast_s``.
- ``derived_only``: a relation computes the key; a preset that sets it is a load error (07 P0-3).
- ``relation``: the relation group(s) of 07 P0-3 the key belongs to (documentation and ``explain``).
- ``per_world``: an enum switch that an uncertainty hypothesis may set per world (it becomes an int code array).
- ``tick``: ``period`` or ``rate`` when the value must be a whole number of ``sim.dt_plant_s`` ticks (07 P0-6).
- ``fields``: record fields of a ``table`` key (list or map of records with bare numbers; the provenance sits on
  the table leaf).
- ``default``: only for settings and switches. A physical number never has a schema default.
- ``bounds``: loose sanity limits in config units (assumed, chosen here to catch unit slips such as a value off by
  1e3; they are not physics and carry no source). The physical range of a leaf is its own ``range`` field.

The initial key list below covers what ``realize()`` consumes. Later agents add keys here (or with :func:`extend`)
for their sections; unknown keys are always an error with a did-you-mean suggestion.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from mrm.params.units import UnitError, parse_key

__all__ = ["Key", "SCHEMA", "OPEN_PREFIXES", "SPECIAL_TOP", "DELETED", "lookup", "is_open", "suggest", "extend",
           "check_units", "check_conditions", "enum_code", "TOP_SECTIONS", "relation_members", "table_field"]

Cond = tuple[tuple[str, tuple[Any, ...]], ...]


@dataclass(frozen=True)
class Key:
    """One config key. See the module docstring for the meaning of each field."""
    path: str
    type: str = "float"
    doc: str = ""
    level: str = "global"
    physical: bool = True
    required: bool = False
    required_if: Cond = ()
    forbidden_if: Cond = ()
    derived_only: bool = False
    relation: tuple[str, ...] = ()
    enum: tuple[Any, ...] = ()
    default: Any = None
    bounds: tuple[float, float] | None = None
    per_world: bool = False
    tick: str | None = None
    fields: tuple["Key", ...] = ()
    table_kind: str = ""

    @property
    def name(self) -> str:
        return self.path.rsplit(".", 1)[-1]


def when(*pairs) -> Cond:
    """Build a condition: when("fidelity.thermal_model", "rom") or when((p1, (v1, v2)), (p2, (v3,)))."""
    if len(pairs) >= 2 and isinstance(pairs[0], str):
        return ((pairs[0], tuple(pairs[1:])),)
    return tuple((p, tuple(v)) for p, v in pairs)


def P(path, type="float", doc="", level="die", **kw) -> Key:
    """A physical key (needs a provenance leaf in presets)."""
    return Key(path=path, type=type, doc=doc, level=level, physical=True, **kw)


def S(path, type="str", doc="", **kw) -> Key:
    """A setting or switch (bare values allowed)."""
    return Key(path=path, type=type, doc=doc, level="global", physical=False, **kw)


TOP_SECTIONS = ("meta", "fidelity", "sim", "channel_plan", "laser", "optical_path", "ring", "driver", "temp_sensors",
                "layout", "array", "uncertainty", "variation", "controller_interface", "controller", "scenario",
                "metrics", "experiment", "datapath")
#: top-level keys that are not sections of the physical tree
SPECIAL_TOP = ("extends", "alts", "relations")
#: subtrees validated by their own registries (controller and scenario params, sweep grids); meta.inventory is
#: validated by tests/closure/test_presets.py against docs/research/00_prior_notes_checklist.md
OPEN_PREFIXES = ("meta.sources", "meta.inventory", "controller.params", "scenario.params", "scenario.aggressor",
                 "scenario.target", "scenario.external_reflection", "experiment.grid", "experiment.controllers",
                 "relations")
#: keys deleted by 07; using them gives this message instead of a did-you-mean
DELETED = {
    "ring.optics.absorbed_fraction": "deleted by 07 P0-2: the heat fraction comes from ring.optics.loss_budget",
    "ring.optics.d2n_eff_dt2_per_k2": "deleted by 07 P0-3: use ring.thermal.dlambda_dt_beta_per_k",
    "controller_interface.sensor_latency_samples": "deleted by 07 P0-6: latencies are owned by the I/O path in s",
    "controller_interface.actuator_latency_samples": "deleted by 07 P0-6: latencies are owned by the I/O path in s",
    "array.pitch_um": "deleted by 07 P1-2: use layout.ring_pitch_um or layout.sites",
    "temp_sensor": "renamed by 07 P1-10: use temp_sensors (a table)",
    "ring.thermal.time_scale": "moved by 07 P0-4: uncertainty.thermal.time_scale_fast / time_scale_slow",
    "variation.time_scale_rel_sigma": "deleted by 07 P0-4 (die-level and epistemic, not per ring)",
    "variation.dlambda_dt_rel_sigma": "deleted by 07 P0-4 (die-level and epistemic, not per ring)",
    "ring.heater.heater_efficiency_nm_per_mw": "07 P0-3 R2: efficiency is derived as ring.heater.efficiency_nm_per_mw",
    "ring.optics.coupling_dispersion_per_nm": "replaced by 07 P1-5: dln_kappa2_through/drop_dlambda_per_nm",
    "ring.optics.dlnqi_dt_per_k": "split by 07 P1-5: dln_alpha_dt_ring_per_k and dln_alpha_dt_heater_per_k",
    "ring.monitor.dark_doubling_k": "renamed by 07 P1-9: ring.monitor.dark_activation_ev",
    "ring.heater.full_scale": "split by 07 P1-15: full_scale_v, full_scale_a, full_scale_w",
    "ring.optics.tpa_beta_m_per_w": "removed by 07 P2-1 (excluded with a computed bound)",
    "ring.optics.n2_m2_per_w": "removed by 07 P2-1 (excluded with a computed bound)",
    "ring.optics.mode_area_um2": "removed by 07 P2-1 (excluded with a computed bound)",
}

_THERMAL_ROM = when("fidelity.thermal_model", "rom")
_THERMAL_INTERIM = when("fidelity.thermal_model", "interim")
_ORDER2 = when("fidelity.thermo_optic_order", 2)
_LUT = when("fidelity.junction_model", "lut")
_ANALYTIC = when("fidelity.junction_model", "analytic")

_SITE_FIELDS = (
    P("x_um", doc="site x position", level="global"),
    P("y_um", doc="site y position", level="global"),
    S("kind", "enum", "site kind", enum=("ring", "heater", "aggressor", "monitor_pd", "temp_sensor", "driver", "other")),
    S("note", "str", "free text: why the site is there (not read by the model)"),
)
_LINE_FIELDS = (
    P("freq_thz", doc="line optical frequency", level="global"),
    S("direction", "enum", "tx or rx comb", enum=("tx", "rx"), default="tx"),
    S("source_laser", "str", "laser module that emits the line"),
    S("pol", "enum", "polarization", enum=("te", "tm"), default="te"),
)
_SENSOR_FIELDS = (
    S("name", "str", "sensor name (channel temp.<name>)"),
    S("site", "str", "layout site name (or use attach)"),
    S("attach", "str", "ring_<i> | bank | die (when no site is given)"),
    S("type", "enum", "sensor type", enum=("diode", "rtd", "thermistor")),
    P("sample_rate_hz", doc="conversion rate", level="global", tick="rate"),
    P("latency_s", doc="conversion latency", level="global", tick="period"),
    P("noise_rms_k", doc="white noise per sample", level="global"),
    P("flicker_corner_hz", doc="1/f corner", level="global"),
    P("resolution_k", doc="quantization step", level="global"),
    P("offset_sigma_k", doc="uncalibrated offset sigma (drawn per device)", level="global"),
    P("gain_error_sigma_frac", doc="gain error sigma (drawn per device)", level="global"),
    P("self_heating_k", doc="sensor self-heating", level="global"),
    P("adc_bits", "int", "sensor converter bits", level="global"),
    P("code_zero_c", doc="temperature that code 0 reads (absolute)", level="global"),
)

_KEYS: list[Key] = [
    # ---------------------------------------------------------------------------------------------- meta
    S("meta.name", "str", "preset name"),
    S("meta.kind", "enum", "preset: strict (no bare physical numbers); experiment: bare physical numbers become "
      "overrides; corner: an extends: file that only selects alternatives; private: NDA data (provenance private); "
      "test: unit-test preset", enum=("preset", "experiment", "corner", "private", "test"), default="preset"),
    S("meta.description", "str", "free text"),
    S("meta.sources", "map", "extra source keys declared by this file {key: citation} (private and test presets)"),
    S("meta.generation_mix", "str", "free text: which generation each block comes from (07 4.9 item 6)"),
    S("meta.inventory", "map", "completeness inventory of docs/research/00_prior_notes_checklist.md: {section: "
      "{item: {status: key | excluded | unknown, keys: [...], reason: ...}}} (validated by "
      "tests/closure/test_presets.py)"),

    # ---------------------------------------------------------------------------------------------- fidelity
    S("fidelity.self_heating", "enum", "optical self-heating (07 P0-1)", enum=("on", "off"), default="on"),
    S("fidelity.thermo_optic_order", "enum", "1: constant dlambda/dT; 2: s0*(1+beta*(T-T0)) (07 P0-3)",
      enum=(1, 2), default=2),
    S("fidelity.thermal_model", "enum", "interim three-pole model (07 P1-2) or a FEM ROM artefact",
      enum=("interim", "rom"), default="interim"),
    S("fidelity.junction_model", "enum", "analytic depletion law or junction LUT artefact (07 P1-7)",
      enum=("analytic", "lut"), default="analytic"),
    S("fidelity.optics_model", "enum", "exact add-drop ring or TCMT (07 P2-4)", enum=("exact", "tcmt"),
      default="exact"),
    S("fidelity.backscatter", "enum", "CW/CCW coupling (07 P2-2)", enum=("off", "on"), default="off"),
    S("fidelity.shot_noise", "enum", "photodiode shot noise", enum=("on", "off"), default="on"),
    S("fidelity.pattern_noise", "enum", "ones-density noise (07 P0-11)",
      enum=("off", "binomial", "prbs7", "prbs9", "prbs15", "prbs23", "prbs31"), default="off"),
    S("fidelity.data_modulation", "enum", "mark-density mean or LUT correction (07 P2-6)", enum=("mean", "lut"),
      default="mean"),
    S("fidelity.nonlinear_optics", "enum", "TPA/FCA/Kerr excluded with a bound (07 P2-1)", enum=("off",),
      default="off"),
    S("fidelity.heater_anchor", "enum", "R2 form: fsr_integral (default, integrator decision: Z = dT_FSR/P_FSR with "
      "the integral of s(T) over the P_FSR sweep equal to one FSR, the exact inversion of the anchor measurement, "
      "07 D16) or lock_slope (07 P0-3 as written: Z = efficiency/s(T_lock); 4-5 % low with beta = 1.4e-3 /K and it "
      "makes Z depend on each ring's sampled offset; docs/handoff/integrator.md)",
      enum=("lock_slope", "fsr_integral"), default="fsr_integral"),
    S("fidelity.heater_pwm_ripple", "enum", "dac_pwm ripple term (07 P1-11)", enum=("off", "on"), default="off"),
    S("fidelity.plasma_dispersion", "enum", "free-carrier model (07 P1-7)",
      enum=("nedeljkovic2011", "soref_bennett1987"), default="nedeljkovic2011"),

    # ---------------------------------------------------------------------------------------------- sim
    S("sim.dt_plant_s", "float", "plant tick (07 P0-6)", required=True, bounds=(1e-8, 1e-4)),
    S("sim.substep_policy", "enum", "07 P0-10", enum=("fixed", "auto"), default="fixed"),
    S("sim.min_samples_per_dither_period", "int", "07 P0-10", default=8),
    S("sim.max_detuning_step_pm", "float", "07 P0-10", default=0.2),
    S("sim.reset_protocol", "enum", "07 P0-1 item 6", enum=("laser_ramp", "branch_low", "branch_high"),
      default="laser_ramp"),
    S("sim.reset_ramp_s", "float", "laser turn-on ramp of laser_ramp (>= 10 tau_slow)"),

    # ---------------------------------------------------------------------------------------------- channel plan
    P("channel_plan.center_thz", doc="centre of the per-direction grid", level="global", bounds=(150.0, 300.0)),
    P("channel_plan.spacing_ghz", doc="line spacing seen by one ring bank", level="global", bounds=(10.0, 5000.0)),
    P("channel_plan.n_lines", "int", "lines per bank", level="global", bounds=(1, 64)),
    P("channel_plan.lines", "table", "explicit line table (overrides centre/spacing)", level="global",
      fields=_LINE_FIELDS, table_kind="list"),
    P("channel_plan.ring_to_line", "list_int", "assigned line of each ring (default: identity)", level="global"),

    # ---------------------------------------------------------------------------------------------- laser
    P("laser.power_per_line_dbm", doc="power per line at the laser-module output (07 P0-1 item 4)", level="line",
      required=True, bounds=(-30.0, 30.0)),
    P("laser.line_offset_ghz", doc="static frequency error of each line from the grid", level="line",
      bounds=(-500.0, 500.0)),
    P("laser.rin_db_per_hz", doc="relative intensity noise", level="die", bounds=(-200.0, -100.0)),
    P("laser.linewidth_mhz", doc="laser linewidth", level="die"),
    S("laser.tune.present", "bool", "optional laser.tune actuator (07 P0-7, P2-8)", default=False),
    P("laser.tune.bits", "int", "laser tuning DAC bits", required_if=when("laser.tune.present", True),
      bounds=(1, 32)),
    P("laser.tune.lsb_hz", doc="laser tuning step per code", required_if=when("laser.tune.present", True),
      bounds=(0.0, 1e12)),

    # ---------------------------------------------------------------------------------------------- optical path
    P("optical_path.chip_input_loss_db", doc="laser output to the first ring (fibre, coupler, splitters)",
      required=True, bounds=(0.0, 30.0)),
    P("optical_path.bus_loss_db_per_mm", doc="bus waveguide loss between rings", bounds=(0.0, 20.0)),
    P("optical_path.ring_spacing_on_bus_um", doc="bus length between adjacent rings", bounds=(0.0, 1e5)),

    # ---------------------------------------------------------------------------------------------- ring.geometry
    P("ring.geometry.radius_um", doc="ring radius = the phase reference radius r_ref of n_eff_bent (07 P1-6); R3",
      level="ring", relation=("R3_geometry",), bounds=(1.0, 100.0)),
    P("ring.geometry.width_nm", doc="rib/strip width (informational; sets the artefact)", bounds=(100.0, 2000.0)),
    P("ring.geometry.film_nm", doc="silicon film thickness (D21: 160 or 200)", bounds=(50.0, 500.0)),
    P("ring.geometry.slab_nm", doc="rib slab thickness (0 for a strip)", bounds=(0.0, 300.0)),
    S("ring.geometry.xsec", "str", "cross-section hypothesis name (rib160, rib200, strip400x200, ...)"),
    P("ring.geometry.jacobian", "table", "geometry Jacobian {d: [names], of: {output: [per d]}} (07 P1-16)",
      level="global", table_kind="map"),
    P("ring.geometry.jacobian_coupling", "table", "second Jacobian block (coupling artefact)", level="global",
      table_kind="map"),

    # ---------------------------------------------------------------------------------------------- ring.optics
    P("ring.optics.resonance_offset_nm", doc="cold-cavity blue offset: assigned line minus resonance, at the "
      "reference temperature and bias, heater off, P_in -> 0 (07 P1-6); R3", level="ring", required=True,
      relation=("R3_geometry",), bounds=(-50.0, 50.0)),
    P("ring.optics.resonance_offset_ref_temp_c", doc="temperature at which resonance_offset_nm is defined",
      required=True, bounds=(-50.0, 150.0)),
    P("ring.optics.resonance_offset_ref_bias_v", doc="reverse bias at which resonance_offset_nm is defined",
      bounds=(-1.0, 5.0)),
    P("ring.optics.n_eff", doc="effective index of the bent mode at the resonance, T_ref (07 P1-6); R3",
      level="ring", required=True, relation=("R1_thermo_optic", "R3_geometry"), bounds=(1.0, 4.0)),
    P("ring.optics.n_g", doc="group index of the bent mode; R1, R3", level="ring", relation=("R1_thermo_optic",
      "R3_geometry", "R2_heater_efficiency", "R4_q"), bounds=(1.5, 6.0)),
    P("ring.optics.dn_g_dlambda_per_nm", doc="group-index dispersion", level="ring", bounds=(-0.1, 0.1)),
    P("ring.optics.fsr_nm", doc="FSR in wavelength at the assigned line wavelength, i.e. at the locked resonance "
      "(optional R3 member; FSR_lambda = lambda_line^2 FSR/c)",
      level="ring", relation=("R3_geometry",), bounds=(1.0, 50.0)),
    P("ring.optics.fsr_thz", doc="FSR in frequency (optional R3 member; the 07 P1-6 windows are in THz)",
      level="ring", relation=("R3_geometry",), bounds=(0.1, 10.0)),
    P("ring.optics.order", "int", "longitudinal order m = round(n_eff*L/lambda_res) (R3)", level="ring",
      derived_only=True, relation=("R3_geometry",)),
    P("ring.optics.q_loaded", doc="loaded Q at reference conditions; R4", level="ring", relation=("R4_q",),
      bounds=(100.0, 1e6)),
    P("ring.optics.q_intrinsic", doc="intrinsic Q; R4 (derived from the loss budget when it is given)",
      level="ring", relation=("R4_q", "R4_loss_budget"), bounds=(100.0, 1e7)),
    P("ring.optics.q_through", doc="through-coupler Q; R4", level="ring", relation=("R4_q",), bounds=(100.0, 1e8)),
    P("ring.optics.q_drop", doc="drop-coupler Q; R4", level="ring", relation=("R4_q",), bounds=(100.0, 1e9)),
    P("ring.optics.kappa_d_over_kappa", doc="drop decay rate over total decay rate (sets q_drop = q_loaded/x)",
      level="ring", relation=("R4_q",), bounds=(0.0, 0.5)),
    P("ring.optics.dln_kappa2_through_dlambda_per_nm", doc="through-coupler dispersion d ln K^2/d lambda "
      "(07 P1-5); evaluated at each line", level="ring", bounds=(-1.0, 1.0)),
    P("ring.optics.dln_kappa2_drop_dlambda_per_nm", doc="drop-coupler dispersion", level="ring",
      bounds=(-1.0, 1.0)),
    P("ring.optics.dln_alpha_dt_ring_per_k", doc="loss change with the mode-weighted ring temperature (07 P1-5)",
      level="ring", bounds=(-0.1, 0.1)),
    P("ring.optics.dln_alpha_dt_heater_per_k", doc="loss change with the ROM heater temperature (07 P1-5)",
      level="ring", bounds=(-0.1, 0.1)),
    P("ring.optics.loss_budget.alpha_fca_db_per_cm", doc="free-carrier absorption of the junction at the reference "
      "bias (07 P0-2)", level="ring", relation=("R4_loss_budget",), bounds=(0.0, 1000.0)),
    P("ring.optics.loss_budget.alpha_other_abs_db_per_cm", doc="other heat-producing absorption (heater doping, "
      "silicide, metal, surface states)", level="ring", relation=("R4_loss_budget",), bounds=(0.0, 1000.0)),
    P("ring.optics.loss_budget.alpha_scatter_db_per_cm", doc="sidewall scattering (no heat)", level="ring",
      relation=("R4_loss_budget",), bounds=(0.0, 1000.0)),
    P("ring.optics.loss_budget.alpha_bend_radiation_db_per_cm", doc="bend and slab radiation (no heat)",
      level="ring", relation=("R4_loss_budget",), bounds=(0.0, 1000.0)),
    P("ring.optics.heat_fraction", doc="eta = heat-producing loss / intrinsic loss (derived from the budget, or "
      "given with q_intrinsic when no component is given)", level="ring", relation=("R4_loss_budget",),
      bounds=(0.0, 1.0)),
    P("ring.optics.index_weight_oxide", doc="dn_eff/dn_SiO2 of the bent mode (solver; used only by the "
      "implied_index_weight diagnostic)", level="ring", bounds=(0.0, 2.0)),
    P("ring.optics.dn_eff_dt_per_k", doc="thermo-optic coefficient of the bent mode at T0 (R1); never set it",
      level="ring", derived_only=True, relation=("R1_thermo_optic",)),
    P("ring.optics.alpha_l_per_k", doc="ring length expansion at T0, substrate-clamped (07 P0-3)",
      bounds=(0.0, 1e-4)),
    P("ring.optics.alpha_l_slope_per_k2", doc="d alpha_L / dT", bounds=(-1e-7, 1e-7), required_if=_ORDER2),
    P("ring.optics.backscatter_splitting_ghz", doc="CW/CCW mode splitting 2|beta|/2pi (not the coupling rate)",
      level="ring", required_if=when("fidelity.backscatter", "on"), bounds=(0.0, 100.0)),
    S("ring.optics.backscatter_orders", "int", "M_b: independent complex beta per longitudinal order, applied "
      "periodically in the order (07 P2-2; plant_optics handoff: >= 5)", default=5, bounds=(1, 64)),
    S("ring.optics.tcmt_orders", "int", "orders summed by the TCMT model (07 P2-4); use an odd number (symmetric "
      "about every resonance, plant_optics handoff)", default=3, bounds=(1, 63)),

    # ---------------------------------------------------------------------------------------------- ring.thermal
    P("ring.thermal.dlambda_dt_pm_per_k", doc="s0 = dlambda/dT at the ring temperature T0 (single thermo-optic "
      "anchor, 07 P0-3 R1)", level="ring", required=True, relation=("R1_thermo_optic", "R2_heater_z"),
      bounds=(10.0, 200.0)),
    P("ring.thermal.dlambda_dt_ref_ring_temp_c", doc="T0 of s0", required=True, relation=("R1_thermo_optic",),
      bounds=(-50.0, 250.0)),
    P("ring.thermal.dlambda_dt_beta_per_k", doc="beta of s(T) = s0*(1+beta*(T-T0))", required_if=_ORDER2,
      relation=("R1_thermo_optic",), bounds=(-0.01, 0.01)),
    P("ring.thermal.z_heater_ring_k_per_mw", doc="Z_heater->ring(0) (R2)", level="ring", derived_only=True,
      relation=("R2_heater_z",)),
    P("ring.thermal.t_ring_lock_c", doc="ring temperature at lock when the ambient is the offset reference "
      "temperature (diagnostic)", level="ring", derived_only=True),
    P("ring.thermal.z_ratio_opt_heater", doc="Z_opt->ring(0)/Z_heater->ring(0) (>= 1 for an external heater, "
      "07 P0-1 item 5)", required=True, bounds=(0.1, 10.0)),
    P("ring.thermal.reference_rise_frac", doc="fraction of the ring's mode-weighted temperature rise registered by "
      "the thermometer behind Lightmatter's printed degC values (07 D2 H86_sensor: 0.77-0.86; 1 = the ring "
      "itself). Not used by the plant; replicas and degC scenario targets read it from PlantParams.values",
      bounds=(0.1, 1.0)),
    S("ring.thermal.family", "enum", "heater family (07 P1-3)",
      enum=("inring_noundercut", "silicide_undercut", "metal_above"), default="inring_noundercut", per_world=True),
    S("ring.thermal.stack", "enum", "stack hypothesis (07 P1-3)", enum=("side_by_side", "stacked_eic"),
      default="side_by_side", per_world=True),
    S("ring.thermal.rom_file", "str", "extracted/thermal_rom_<family>_<film>.npz", required_if=_THERMAL_ROM,
      forbidden_if=_THERMAL_INTERIM),
    P("ring.thermal.tau_fast_s", doc="interim self path, fast pole (07 P1-2)", required_if=_THERMAL_INTERIM,
      forbidden_if=_THERMAL_ROM, bounds=(1e-8, 1e-2)),
    P("ring.thermal.share_fast_frac", doc="interim self path, DC share of the fast pole", required_if=_THERMAL_INTERIM,
      forbidden_if=_THERMAL_ROM, bounds=(0.0, 1.0)),
    P("ring.thermal.tau_slow_s", doc="interim self path, slow pole", required_if=_THERMAL_INTERIM,
      forbidden_if=_THERMAL_ROM, bounds=(1e-8, 1.0)),
    P("ring.thermal.tail_tau_s", doc="interim die/package tail pole", required_if=_THERMAL_INTERIM,
      forbidden_if=_THERMAL_ROM, bounds=(1e-6, 10.0)),
    P("ring.thermal.tail_share_frac", doc="DC share of the tail (slow share = 1 - fast - tail)",
      required_if=_THERMAL_INTERIM, forbidden_if=_THERMAL_ROM, bounds=(0.0, 1.0)),
    P("ring.thermal.selfheat_tau_s", doc="sub-us waveguide-core pole of the optical self-heating path (07 P0-1)",
      required_if=_THERMAL_INTERIM, forbidden_if=_THERMAL_ROM, bounds=(1e-9, 1e-4)),
    P("ring.thermal.selfheat_share_frac", doc="DC share of Z_opt in the sub-us pole (rest follows the heater "
      "poles); absent: (z - 1)/z, the largest passive share (mrm.plant.thermal.InterimRom)",
      forbidden_if=_THERMAL_ROM, bounds=(0.0, 1.0)),
    P("ring.thermal.heater_self_ratio", doc="R_th,H / Z_heater->ring(0): heater resistor observing its own heater",
      forbidden_if=_THERMAL_ROM, bounds=(1.0, 10.0)),
    P("ring.thermal.heater_node_tau_s", doc="extra pole of the heater-resistor observer", forbidden_if=_THERMAL_ROM,
      bounds=(1e-9, 1e-3)),
    P("ring.thermal.xt_next_frac", doc="interim crosstalk to the second rank of neighbours", forbidden_if=_THERMAL_ROM,
      bounds=(0.0, 0.5)),
    P("ring.thermal.site_ref_um", doc="interim site coupling: equals xt_nearest_frac at this distance",
      forbidden_if=_THERMAL_ROM, bounds=(1.0, 1e4)),
    P("ring.thermal.site_max_frac", doc="interim site coupling cap", forbidden_if=_THERMAL_ROM, bounds=(0.0, 1.0)),
    P("ring.thermal.pd_offset_um", "list_float", "monitor PD offset from the ring centre (x, y)", level="global",
      forbidden_if=_THERMAL_ROM),
    P("ring.thermal.temp_sensor_offset_um", "list_float", "ring-attached sensor offset (x, y)", level="global",
      forbidden_if=_THERMAL_ROM),
    P("ring.thermal.driver_offset_um", "list_float", "driver offset from the ring centre (x, y)", level="global",
      forbidden_if=_THERMAL_ROM),
    S("ring.thermal.xt_tau_law", "enum", "capped (monotone) or literal (07 P1-2 as written)",
      enum=("capped", "literal"), default="capped"),
    S("ring.thermal.crosstalk_dynamics", "enum", "imec first-order crosstalk, or the self shape (textbook corner)",
      enum=("imec", "self"), default="imec"),
    S("ring.thermal.scaling_hypothesis", "enum", "how tau scales with the residue (07 P1-2)",
      enum=("resistance", "capacitance", "independent"), default="resistance", per_world=True),
    P("ring.thermal.xt_nearest_frac", doc="interim crosstalk to the nearest neighbour, fraction of the self DC",
      required_if=_THERMAL_INTERIM, forbidden_if=_THERMAL_ROM, bounds=(0.0, 0.5)),
    P("ring.thermal.xt_floor_frac", doc="interim crosstalk floor to every other ring", required_if=_THERMAL_INTERIM,
      forbidden_if=_THERMAL_ROM, bounds=(0.0, 0.5)),
    P("ring.thermal.xt_tau_ref_s", doc="tau_xt = max(xt_tau_min_s, xt_tau_ref_s*(d/xt_dist_ref_um)^2) (07 P1-2)",
      required_if=_THERMAL_INTERIM, forbidden_if=_THERMAL_ROM, bounds=(1e-6, 1.0)),
    P("ring.thermal.xt_dist_ref_um", doc="reference distance of xt_tau_ref_s", required_if=_THERMAL_INTERIM,
      forbidden_if=_THERMAL_ROM, bounds=(1.0, 1e4)),
    P("ring.thermal.xt_tau_min_s", doc="floor of the crosstalk time constant", required_if=_THERMAL_INTERIM,
      forbidden_if=_THERMAL_ROM, bounds=(1e-7, 1.0)),
    P("ring.thermal.xt_tau_far_s", doc="crosstalk time constant beyond xt_far_um (absent: the printed X2 table of "
      "04 section 2.4)", forbidden_if=_THERMAL_ROM, bounds=(1e-6, 1.0)),
    P("ring.thermal.xt_far_um", doc="distance beyond which xt_tau_far_s applies", forbidden_if=_THERMAL_ROM,
      bounds=(1.0, 1e5)),
    P("ring.thermal.transport_delay_s", doc="optional per-pair transport delay", bounds=(0.0, 1e-3)),

    # ---------------------------------------------------------------------------------------------- ring.heater
    P("ring.heater.p_fsr_mw", doc="heater power per FSR (the heater anchor, 07 D16, R2)", level="ring",
      required=True, relation=("R2_heater_efficiency",), bounds=(1.0, 200.0)),
    P("ring.heater.p_fsr_ref_temp_c", doc="stage temperature of the P_FSR measurement (fsr_integral form)",
      bounds=(-50.0, 150.0), required_if=when("fidelity.heater_anchor", "fsr_integral")),
    P("ring.heater.efficiency_nm_per_mw", doc="FSR/P_FSR (R2; derived)", level="ring", derived_only=True,
      relation=("R2_heater_efficiency", "R2_heater_z")),
    P("ring.heater.r_cold_ohm", doc="heater resistance at r_ref_temp_c", level="ring", required=True,
      bounds=(1.0, 1e6)),
    P("ring.heater.r_ref_temp_c", doc="temperature of r_cold_ohm", required=True, bounds=(-50.0, 150.0)),
    P("ring.heater.tcr_per_k", doc="R = R0*(1 + tcr*(T_heater - T_ref)) (07 P1-2, P1-11)", required=True,
      bounds=(-0.01, 0.01)),
    S("ring.heater.driver_mode", "enum", "driver law hypothesis (07 P1-11)",
      enum=("voltage", "current", "power", "dac_pwm"), default="voltage", per_world=True),
    P("ring.heater.full_scale_v", doc="DAC full scale in voltage drive",
      required_if=when("ring.heater.driver_mode", "voltage", "dac_pwm"), bounds=(0.0, 50.0)),
    P("ring.heater.full_scale_a", doc="DAC full scale in current drive",
      required_if=when("ring.heater.driver_mode", "current"), bounds=(0.0, 1.0)),
    P("ring.heater.full_scale_w", doc="DAC full scale in power drive",
      required_if=when("ring.heater.driver_mode", "power"), bounds=(0.0, 1.0)),
    P("ring.heater.dac_bits", "int", "heater DAC resolution", required=True, bounds=(1, 32)),
    P("ring.heater.code_min", "int", "lowest code (07 P1-11: hardware, not measured demand)", bounds=(0, 2**32)),
    P("ring.heater.code_max", "int", "highest code", bounds=(0, 2**32)),
    P("ring.heater.slew_codes_per_s", doc="hardware slew limit (absent: unlimited)", bounds=(0.0, 1e15)),
    P("ring.heater.settle_tau_s", doc="analog settle filter", bounds=(0.0, 1e-2)),
    P("ring.heater.latency_s", doc="DAC latency (one owner, 07 P0-6)", required=True, tick="period",
      bounds=(0.0, 1e-2)),
    P("ring.heater.dac_inl_lsb", doc="DAC INL", bounds=(0.0, 1e4)),
    P("ring.heater.dac_dnl_lsb", doc="DAC DNL", bounds=(0.0, 1e4)),
    P("ring.heater.dac_max_update_hz", doc="max DAC update rate", bounds=(0.0, 1e10)),
    P("ring.heater.driver_compliance_v", doc="current-drive headroom", bounds=(0.0, 100.0)),
    P("ring.heater.pwm_period_s", doc="dac_pwm period (PAT-2: 2.56 us)",
      required_if=when("ring.heater.driver_mode", "dac_pwm"), bounds=(0.0, 1e-3)),
    P("ring.heater.pwm_bits", "int", "PWM bits of dac_pwm (PAT-2: 8)", bounds=(1, 16)),
    S("ring.heater.readback.present", "bool", "heater V/I readback (07 P1-10)", default=False),
    P("ring.heater.readback.adc_bits", "int", "readback ADC bits", bounds=(1, 32),
      required_if=when("ring.heater.readback.present", True)),
    P("ring.heater.readback.v_full_scale_v", doc="readback voltage full scale", bounds=(0.0, 100.0)),
    P("ring.heater.readback.i_full_scale_a", doc="readback current full scale", bounds=(0.0, 10.0)),
    P("ring.heater.readback.rate_hz", doc="readback rate", tick="rate", bounds=(0.0, 1e9)),
    P("ring.heater.readback.latency_s", doc="readback latency", tick="period", bounds=(0.0, 1.0)),
    P("ring.heater.readback.noise_rms_codes", doc="readback ADC noise per sample (absent -> 0)", bounds=(0.0, 1e6)),
    S("ring.heater.readback.average", "bool", "readback ADC averages V and I over its sample period (boxcar) instead "
      "of a track-and-hold of the last tick (mrm.plant.sensing.ReadbackHw)", default=False),

    # ---------------------------------------------------------------------------------------------- ring.junction
    P("ring.junction.fill_fraction", doc="fraction of the circumference with the junction (07 P1-7, CAL-5)",
      level="ring", required=True, bounds=(0.0, 1.0)),
    S("ring.junction.lut_file", "str", "extracted/junction_<xsec>.json", required_if=_LUT, forbidden_if=_ANALYTIC),
    P("ring.junction.eff_at_0v_pm_per_v", doc="dlambda/dV at 0 V for the analytic law (derived in LUT mode)",
      level="ring", required_if=_ANALYTIC, forbidden_if=_LUT, bounds=(0.0, 1000.0)),
    P("ring.junction.v_bi_v", doc="built-in voltage", required_if=_ANALYTIC, bounds=(0.0, 2.0)),
    P("ring.junction.depletion_exponent", doc="m of dlambda = A*((V+Vbi)^m - Vbi^m): 1/2 (abrupt) to 2/3 (linearly "
      "graded) by charge control, dlambda ~ removed charge (03 section 5.3, plant_electrical note 1); 07 P1-7 wrote "
      "1/3..1/2 (the depletion-width exponents)", required_if=_ANALYTIC, bounds=(0.2, 0.7)),
    P("ring.junction.dqi_dv_frac_per_v", doc="relative Q_i change per volt (derived in LUT mode)", level="ring",
      forbidden_if=_LUT, bounds=(-1.0, 1.0)),
    P("ring.junction.loss_ratio_q", doc="q = lambda*d_alpha/(4 pi d_n) of the plasma effect (03 section 5.4)",
      bounds=(0.0, 1.0)),
    P("ring.junction.capacitance_at_0v_ff", doc="junction capacitance", level="ring", bounds=(0.0, 1000.0)),
    P("ring.junction.series_resistance_ohm", doc="junction series resistance", level="ring", bounds=(0.0, 1e4)),

    # ---------------------------------------------------------------------------------------------- ring.monitor
    P("ring.monitor.responsivity_a_per_w", doc="drop PD responsivity", level="ring", required=True,
      bounds=(0.0, 2.0)),
    P("ring.monitor.gain_scale", doc="lumped monitor-gain nuisance factor (07 P1-9)", bounds=(0.01, 100.0)),
    P("ring.monitor.dark_i0_na", doc="dark current at dark_t0_c (07 D20)", bounds=(0.0, 1e4)),
    P("ring.monitor.dark_t0_c", doc="reference temperature of dark_i0_na", bounds=(-50.0, 150.0)),
    P("ring.monitor.dark_activation_ev", doc="Arrhenius activation energy (07 P1-9)", bounds=(0.0, 2.0)),
    P("ring.monitor.background_fraction", doc="stray light at the PD, fraction of the line power", level="ring",
      bounds=(0.0, 1.0)),
    S("ring.monitor.afe_type", "enum", "07 P0-7", enum=("rc", "integrate_dump"), default="integrate_dump"),
    P("ring.monitor.integration_window_s", doc="integrate-and-dump boxcar (07 P0-7)", tick="period",
      required_if=when("ring.monitor.afe_type", "integrate_dump"), bounds=(0.0, 1.0)),
    P("ring.monitor.afe_tau_s", doc="RC front-end time constant", required_if=when("ring.monitor.afe_type", "rc"),
      bounds=(0.0, 1.0)),
    P("ring.monitor.adc_rate_hz", doc="monitor ADC sample rate", required=True, tick="rate", bounds=(1.0, 1e9)),
    P("ring.monitor.adc_latency_s", doc="conversion latency (one owner, 07 P0-6)", required=True, tick="period",
      bounds=(0.0, 1.0)),
    P("ring.monitor.adc_bits", "int", "monitor ADC bits", required=True, bounds=(1, 32)),
    P("ring.monitor.adc_full_scale_ua", doc="PD current at ADC full scale", required=True, bounds=(0.0, 1e6)),
    P("ring.monitor.adc_offset_codes", doc="ADC offset", level="ring", bounds=(-1e6, 1e6)),
    P("ring.monitor.adc_gain_error_frac", doc="ADC gain error", level="ring", bounds=(-1.0, 1.0)),
    P("ring.monitor.adc_inl_lsb", doc="ADC INL", bounds=(0.0, 1e4)),
    P("ring.monitor.adc_dnl_lsb", doc="ADC DNL", bounds=(0.0, 1e4)),
    P("ring.monitor.input_noise_pa_per_rt_hz", doc="input-referred current noise density", bounds=(0.0, 1e6)),
    P("ring.monitor.flicker_corner_hz", doc="1/f corner", bounds=(0.0, 1e9)),
    P("ring.monitor.adc_ref_tempco_ppm_per_k", doc="reference tempco", bounds=(-1e4, 1e4)),
    P("ring.monitor.pga_gains", "list_float", "PGA gain steps (07 P1-10)", level="global"),
    P("ring.monitor.adc_mux.channels_per_adc", "int", "rings per converter (07 P0-7)", bounds=(1, 64)),
    S("ring.monitor.adc_mux.order", "enum", "mux order", enum=("round_robin", "bank"), default="round_robin"),
    # integrator additions requested by plant_sensing (docs/handoff/plant_sensing.md section 4)
    P("ring.monitor.baseline_ua", doc="analog baseline current subtracted before the PGA and ADC (PAT-2 i_os,det, "
      "07 P1-9); absent -> 0", bounds=(0.0, 1e6)),
    S("ring.monitor.stuck_flagged", "bool", "a stuck converter clears monitor_valid (07 P1-10); false = silent",
      default=True),
    S("ring.monitor.flicker_band_min_hz", "float", "lowest pole of the 1/f noise synthesis (mrm.plant.sensing)",
      default=1e-3),
    S("ring.monitor.reference_tap.present", "bool", "reference-tap PD on the bus before the bank (07 P1-9, 06 S8)",
      default=False),
    P("ring.monitor.reference_tap.coupling_frac", doc="bus power fraction tapped to the reference PD",
      required_if=when("ring.monitor.reference_tap.present", True), bounds=(0.0, 1.0)),
    P("ring.monitor.reference_tap.responsivity_a_per_w", doc="reference PD responsivity (absent: the drop PD's)",
      bounds=(0.0, 2.0)),
    P("ring.monitor.reference_tap.dark_i0_na", doc="reference PD dark current (absent: the drop PD's)",
      bounds=(0.0, 1e4)),

    # ---------------------------------------------------------------------------------------------- ring.bias_dac
    P("ring.bias_dac.bits", "int", "bias DAC bits (07 P0-7)", required=True, bounds=(1, 32)),
    P("ring.bias_dac.v_min_v", doc="bias range low end (printed G26 axis)", required=True, bounds=(-5.0, 10.0)),
    P("ring.bias_dac.v_max_v", doc="bias range high end", required=True, bounds=(-5.0, 10.0)),
    P("ring.bias_dac.update_rate_hz", doc="bias DAC update rate", tick="rate", bounds=(0.0, 1e10)),
    P("ring.bias_dac.latency_s", doc="bias DAC latency (one owner)", required=True, tick="period",
      bounds=(0.0, 1.0)),
    P("ring.bias_dac.slew_v_per_s", doc="bias slew limit", bounds=(0.0, 1e15)),
    P("ring.bias_dac.settle_tau_s", doc="bias settle filter", bounds=(0.0, 1.0)),
    P("ring.bias_dac.noise_rms_codes", doc="bias DAC output noise", bounds=(0.0, 1e6)),

    # ---------------------------------------------------------------------------------------------- driver
    P("driver.output_resistance_ohm", doc="driver output resistance (07 P1-8)", bounds=(0.0, 1e4)),
    P("driver.static_power_mw", doc="driver static power", bounds=(0.0, 1e3)),
    P("driver.energy_per_transition_fj", doc="driver energy per transition", bounds=(0.0, 1e5)),
    P("driver.swing_pp_v", doc="junction peak-to-peak swing", level="ring", bounds=(0.0, 10.0)),
    S("driver.site", "str", "layout site of the driver (07 P1-8)"),
    S("driver.level_convention", "enum", "NRZ levels: centered (bias +- Vpp/2, 07 P1-7) or one_sided (bias - Vpp "
      "and bias)", enum=("centered", "one_sided"), default="centered", per_world=True),

    # ---------------------------------------------------------------------------------------------- temp sensors
    P("temp_sensors", "table", "temperature sensors (07 P1-10)", level="global", fields=_SENSOR_FIELDS,
      table_kind="list"),

    # ---------------------------------------------------------------------------------------------- layout, array
    P("layout.ring_pitch_um", doc="ring pitch of the default straight bank", bounds=(1.0, 1e4), level="global"),
    P("layout.sites", "table", "named sites {name: {x_um, y_um, kind}} (07 P1-2); rings default to ring<i> on a "
      "line at ring_pitch_um", level="global", fields=_SITE_FIELDS, table_kind="map"),
    P("array.n_rings", "int", "rings per bank", required=True, level="global", bounds=(1, 256)),
    P("array.bus_order", "list_int", "order of the rings along the bus (default: identity)", level="global"),
    P("array.heater_total_power_max_w", doc="die heater power limit (07 P1-12)", bounds=(0.0, 100.0)),
    P("array.heater_supply_current_max_a", doc="die heater supply current limit", bounds=(0.0, 100.0)),

    # ---------------------------------------------------------------------------------------------- controller I/F
    S("controller_interface.update_rate_hz", "float", "controller rate (a whole number of ticks)", tick="rate"),
    S("controller_interface.bind.tx.heater", "enum", "07 P0-7", enum=("controller",), default="controller"),
    S("controller_interface.bind.tx.bias", "enum", "07 P0-7", enum=("scenario", "controller"), default="scenario"),
    S("controller_interface.bind.laser.tune", "enum", "07 P0-7", enum=("scenario", "controller"), default="scenario"),
    S("controller_interface.bind.tx.assignment", "enum", "07 P1-12", enum=("scenario", "controller"),
      default="scenario"),
    S("controller_interface.bind.datapath.bitstat_threshold", "enum", "07 P0-11", enum=("controller", "scenario"),
      default="controller"),
    S("controller_interface.bind.monitor.pga_select", "enum", "07 P1-10", enum=("controller", "scenario"),
      default="controller"),
    S("controller_interface.expose", "list_str", "sensor channels the controller may read (default: all bound)"),
    S("controller_interface.knowledge", "list_str", "what the controller may know (07 P1-1)"),
    S("controller_interface.schedule", "enum", "07 P2-7", enum=("simultaneous", "round_robin"),
      default="simultaneous"),
    S("controller_interface.on_out_of_range", "enum", "07 P1-19", enum=("clip", "error"), default="clip"),
    S("controller_interface.tick_jitter_s", "float", "07 P2-7"),

    # ---------------------------------------------------------------------------------------------- controller
    S("controller.name", "str", "registry name or module:Class"),
    S("controller.params", "map", "controller parameters (validated by the controller)"),

    # ---------------------------------------------------------------------------------------------- scenario
    S("scenario.name", "str", "registry name"),
    S("scenario.stack", "list_str", "scenario stack"),
    S("scenario.params", "map", "scenario parameters (validated by the scenario registry)"),
    S("scenario.time_compression", "float", "07 P0-10", default=1.0),
    S("scenario.ambient_c", "float", "constant ambient when no scenario sets it"),
    S("scenario.bias_v", "float", "absolute bias when the scenario owns tx.bias",
      forbidden_if=when("controller_interface.bind.tx.bias", "controller")),
    S("scenario.bias_offset_v", "float", "additive bias offset (always allowed)"),
    S("scenario.aggressor", "map", "07 P1-4"),
    S("scenario.target", "map", "07 P1-4"),
    S("scenario.external_reflection", "map", "07 P2-2"),

    # ---------------------------------------------------------------------------------------------- metrics
    S("metrics.reloma_definition", "enum", "07 D9", enum=("oma_over_pin",), default="oma_over_pin"),
    S("metrics.ber_sample_period_s", "float", "G26 instrument interval", default=2.0),
    S("metrics.lock_threshold_pm", "float", "lock excursion threshold"),
    S("metrics.hysteresis_pm", "float", "excursion hysteresis"),
    S("metrics.allowed_assignments", "enum", "07 P1-12", enum=("fixed", "cyclic", "any"), default="fixed"),
    S("metrics.lock_reference", "enum", "reference of the lock error: the line (07 literal), the oracle operating "
      "point, or metrics.lock_target_pm (mrm.sim.metrics; engine handoff 3 item 21)",
      enum=("line", "oracle", "target"), default="line"),
    S("metrics.lock_target_pm", "float", "lock-error reference detuning (pm) when metrics.lock_reference is target"),

    # ---------------------------------------------------------------------------------------------- experiment
    S("experiment.kind", "enum", "07 P1-17",
      enum=("run", "sweep", "mc", "compare", "robust", "acceptance", "quasi_static"), default="run"),
    S("experiment.duration_s", "float", "simulated duration"),
    S("experiment.batch", "int", "batch size B"),
    S("experiment.n_mc", "int", "Monte Carlo worlds"),
    S("experiment.seeds.device", "int", "device (variation) seed base"),
    S("experiment.seeds.uncertainty", "int", "world seed base"),
    S("experiment.seeds.noise", "int", "noise seed base"),
    S("experiment.seeds.scenario", "int", "scenario seed base"),
    S("experiment.uncertainty", "enum", "how realize() draws worlds", enum=("nominal", "sample", "lhs"),
      default="sample"),
    S("experiment.variation", "enum", "per-device variation on or off", enum=("off", "sample"), default="sample"),
    S("experiment.grid", "map", "sweep grid {path: [values]}"),
    S("experiment.controllers", "list_str", "controllers of a compare experiment"),

    # ---------------------------------------------------------------------------------------------- datapath
    P("datapath.bitrate_gbps", doc="line rate (G26: 50 Gb/s NRZ)", bounds=(0.0, 1000.0)),
    P("datapath.bitstat_window_bits", "int", "bit-statistics window (07 P0-11)", bounds=(1, 1 << 20)),
    P("datapath.error_count_period_s", doc="error-counter period (G26: 2.0 s)", tick="period", bounds=(0.0, 1e4)),
    P("datapath.error_count_latency_s", doc="error-counter latency (one owner)", tick="period", bounds=(0.0, 1e4)),
    P("datapath.transition_density", doc="mean transition density of the traffic", bounds=(0.0, 1.0)),
    # integrator additions requested by plant_sensing (docs/handoff/plant_sensing.md section 4); absent -> the module
    # constants of mrm.plant.datapath (G26 worst lane -12.8 dBm OMA at 1e-9, 3 dB link, 16 GHz sampler)
    P("datapath.rx_sensitivity_oma_dbm", doc="receiver OMA sensitivity at datapath.ber_ref, per lane", level="ring",
      bounds=(-40.0, 10.0)),
    P("datapath.ber_ref", doc="BER at which rx_sensitivity_oma_dbm is quoted", bounds=(1e-30, 0.5)),
    P("datapath.link_loss_db", doc="loss from the TX bus output to the RX PD (the OMA reference planes)",
      bounds=(0.0, 60.0)),
    P("datapath.ber_floor", doc="BER floor of the error counter", bounds=(0.0, 0.5)),
    P("datapath.bitstat_bw_ghz", doc="bandwidth of the bit-rate sampler of the bit statistics (07 P0-11)",
      bounds=(0.0, 1000.0)),
    P("datapath.bitstat_noise_pa_per_rt_hz", doc="input noise density of the bit-rate sampler (absent: the monitor "
      "TIA density)", bounds=(0.0, 1e6)),
]

SCHEMA: dict[str, Key] = {}


def extend(keys: Iterable[Key]) -> None:
    """Register additional keys (used by later agents for their sections). A duplicate path is an error."""
    for k in keys:
        if k.path in SCHEMA:
            raise ValueError(f"schema key {k.path!r} registered twice")
        SCHEMA[k.path] = k


extend(_KEYS)


def lookup(path: str) -> Key | None:
    """Exact schema entry of ``path`` (None if unknown)."""
    return SCHEMA.get(path)


def is_open(path: str) -> bool:
    """True when ``path`` lies inside an open subtree (validated elsewhere)."""
    return any(path == p or path.startswith(p + ".") for p in OPEN_PREFIXES)


def suggest(path: str, n: int = 3) -> list[str]:
    """Did-you-mean candidates for an unknown key: close full paths, then close siblings by last segment."""
    if path in DELETED:
        return []
    keys = list(SCHEMA)
    out = difflib.get_close_matches(path, keys, n=n, cutoff=0.75)
    parent, _, last = path.rpartition(".")
    sibs = [k for k in keys if k.rpartition(".")[0] == parent]
    for c in difflib.get_close_matches(last, [s.rpartition(".")[2] for s in sibs], n=n, cutoff=0.6):
        full = f"{parent}.{c}" if parent else c
        if full not in out:
            out.append(full)
    return out[:n]


def table_field(table_key: Key, name: str) -> Key | None:
    """Schema of one record field of a table key."""
    for f in table_key.fields:
        if f.name == name or f.path == name:
            return f
    return None


def check_units() -> list[str]:
    """Return problems with the unit suffix of every schema key (empty list when all parse)."""
    errs = []
    for k in SCHEMA.values():
        try:
            parse_key(k.path)
        except UnitError as e:
            errs.append(str(e))
        for f in k.fields:
            try:
                parse_key(f.path)
            except UnitError as e:
                errs.append(f"{k.path}[]: {e}")
    return errs


def _cond_holds(cond: Cond, get: Callable[[str], Any]) -> bool:
    for path, values in cond:
        v = get(path)
        if v is None:
            key = SCHEMA.get(path)
            v = key.default if key is not None else None
        if v not in values:
            return False
    return True


def check_conditions(get: Callable[[str], Any], present: set[str]) -> list[str]:
    """Check required / required_if / forbidden_if over the resolved config.

    ``get(path)`` returns the resolved value of a switch (or None); ``present`` holds every set key path.
    Returns a list of human-readable errors.
    """
    errs = []
    for k in SCHEMA.values():
        is_set = k.path in present
        if k.required and not is_set and not k.derived_only:
            errs.append(f"missing required key {k.path!r} ({k.doc})")
        if k.required_if and _cond_holds(k.required_if, get) and not is_set:
            cond = ", ".join(f"{p} in {list(v)}" for p, v in k.required_if)
            errs.append(f"missing key {k.path!r}: required when {cond}")
        if k.forbidden_if and _cond_holds(k.forbidden_if, get) and is_set:
            cond = ", ".join(f"{p} in {list(v)}" for p, v in k.forbidden_if)
            errs.append(f"key {k.path!r} is forbidden when {cond}")
    return errs


def enum_code(path: str, value: Any) -> int:
    """Integer code of an enum value (its index in the schema enum); used for per-world switch arrays."""
    k = SCHEMA[path]
    return k.enum.index(value)


def relation_members(name: str) -> list[str]:
    """All schema keys that declare membership in relation ``name``."""
    return [k.path for k in SCHEMA.values() if name in k.relation]
