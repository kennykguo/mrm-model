# declare allowed config keys and their validation rules.
# _KEYS lists every supported path, grouped by subsystem.
# P declares a physical claim, which needs a provenance leaf in a strict preset.
# S declares a setting or switch, which can be a bare value.
# SCHEMA indexes these entries by path.
# config.py validates loaded values against them.
# units come from key suffixes, not a separate schema field.
# see params.units.
# OPEN_PREFIXES delegates flexible subtrees to their own registries.
# other unknown paths are rejected with suggest candidates.
# DELETED supplies migration errors.
# ENUM_ALIASES supports older enum words.
# presets, experiments, and saved runs depend on stable key and enum names.
# extend permits new keys but rejects duplicate paths.
# runtime descriptions below also serve parameter documentation.
# Key explains field semantics.
# P/S and when build entries.
# the registry helpers follow the key table.
# check_conditions enforces required/forbidden rules.
# check_units reports suffix-parser errors.
# keep the declaration table, defaults, bounds, and descriptions unchanged during a readability pass.
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from mrm.params.units import UnitError, parse_key

__all__ = ["Key", "SCHEMA", "OPEN_PREFIXES", "SPECIAL_TOP", "DELETED", "ENUM_ALIASES", "lookup", "is_open", "suggest",
           "extend", "check_units", "check_conditions", "enum_code", "TOP_SECTIONS", "relation_members",
           "table_field"]

# a fidelity condition - ((path, allowed values), ...) - it holds when every path has one of its allowed values
Condition = tuple[tuple[str, tuple[Any, ...]], ...]


# describe a config path's type, scope, provenance requirements, and conditional rules.
# type selects float/int/bool/str/enum, typed lists, table, or map.
# physical marks hardware/world claims.
# level maps realized arrays to global, die (B, 1), ring (B, N), or line (B, L).
# ring/line leaves can use a scalar or a list.
# enum lists allowed words and defines their code order.
# required is unconditional.
# required_if/forbidden_if use conditions where every path must match.
# derived_only prevents presets from specifying a value computed by a relation.
# relation names its R1–R5 groups for documentation and explain.
# per_world allows uncertainty hypotheses to select enum values per world.
# tick marks a period or rate that must align with plant ticks.
# fields describes table record keys.
# table_kind selects list or map layout.
# table provenance belongs to its enclosing leaf.
# default is for settings only, never a physical number.
# bounds are loose sanity checks in config units, chosen to catch unit slips.
# they are not measured physical ranges.
# each leaf's range holds that information.
# doc describes the key for generated views.
@dataclass(frozen=True)
class Key:
    path: str
    type: str = "float"
    doc: str = ""
    level: str = "global"
    physical: bool = True
    required: bool = False
    required_if: Condition = ()
    forbidden_if: Condition = ()
    derived_only: bool = False
    relation: tuple[str, ...] = ()
    enum: tuple[Any, ...] = ()
    default: Any = None
    bounds: tuple[float, float] | None = None
    per_world: bool = False
    tick: str | None = None
    fields: tuple["Key", ...] = ()
    table_kind: str = ""

    # last segment of the dotted path.
    @property
    def name(self) -> str:
        return self.path.rsplit(".", 1)[-1]


# build an all-pairs condition, allowing multiple values at each path.
# when("fidelity.thermal_model", "rom") tests one path.
# pass (path, allowed_values) pairs to combine paths.
def when(*pairs) -> Condition:
    if len(pairs) >= 2 and isinstance(pairs[0], str):
        return ((pairs[0], tuple(pairs[1:])),)
    return tuple((p, tuple(v)) for p, v in pairs)


# declare a physical key.
# strict presets must supply its provenance leaf.
def P(path, type="float", doc="", level="die", **kw) -> Key:
    return Key(path=path, type=type, doc=doc, level=level, physical=True, **kw)


# declare a global setting or switch.
# a bare value is allowed.
def S(path, type="str", doc="", **kw) -> Key:
    return Key(path=path, type=type, doc=doc, level="global", physical=False, **kw)


# top level, aliases, deleted

# top-level sections of the physical and settings tree
TOP_SECTIONS = ("meta", "fidelity", "sim", "channel_plan", "laser", "optical_path", "ring", "driver", "temp_sensors",
                "layout", "array", "uncertainty", "variation", "controller_interface", "controller", "scenario",
                "metrics", "experiment", "datapath")
# top-level keys that are not sections of the physical tree
SPECIAL_TOP = ("extends", "alts", "relations")
# subtrees that other registries validate - controller and scenario parameters, sweep grids, and the experiment
# records, corners, self reference and acceptance criteria (mrm.sim.experiment). tests/closure/test_presets.py
# checks meta.inventory against docs/research/00_prior_notes_checklist.md.
OPEN_PREFIXES = ("meta.sources", "meta.inventory", "controller.params", "scenario.params", "scenario.aggressor",
                 "scenario.target", "scenario.external_reflection", "experiment.grid", "experiment.controllers",
                 "experiment.record", "experiment.corners", "experiment.self_reference", "experiment.acceptance",
                 "relations")
# ENUM_ALIASES maps each config path to its old and new enum words.
# the loader replaces old words before validation, so saved experiments and run directories still load.
ENUM_ALIASES = {
    "experiment.kind": {"quasi_static": "locked_steady_state"},   # renamed 2026-10-21 (docs/CHANGELOG.md)
}
# keys that were deleted or renamed - a file that still uses one gets this message instead of a did-you-mean
DELETED = {
    "ring.optics.absorbed_fraction": "deleted: the heat fraction now comes from ring.optics.loss_budget",
    "ring.optics.d2n_eff_dt2_per_k2": "deleted: relation R1 derives d2n_eff/dT2 from s(T) = s0*(1 + beta*(T - T0)); "
                                      "set ring.thermal.dlambda_dt_beta_per_k",
    "controller_interface.sensor_latency_samples": "deleted: each latency is stated once, in seconds, by its I/O path",
    "controller_interface.actuator_latency_samples": "deleted: each latency is stated once, in seconds, by its I/O "
                                                     "path",
    "array.pitch_um": "deleted: use layout.ring_pitch_um or layout.sites",
    "temp_sensor": "renamed: use temp_sensors (a table)",
    "ring.thermal.time_scale": "moved: uncertainty.thermal.time_scale_fast / time_scale_slow",
    "variation.time_scale_rel_sigma": "deleted: the thermal time scale is die-level and epistemic, not a per-ring "
                                      "variation",
    "variation.dlambda_dt_rel_sigma": "deleted: dlambda/dT is die-level and epistemic, not a per-ring variation",
    "ring.heater.heater_efficiency_nm_per_mw": "deleted: relation R2 derives the efficiency as "
                                               "ring.heater.efficiency_nm_per_mw",
    "ring.optics.coupling_dispersion_per_nm": "replaced by dln_kappa2_through_dlambda_per_nm and "
                                              "dln_kappa2_drop_dlambda_per_nm",
    "ring.optics.dlnqi_dt_per_k": "split into dln_alpha_dt_ring_per_k and dln_alpha_dt_heater_per_k",
    "ring.monitor.dark_doubling_k": "renamed: ring.monitor.dark_activation_ev (Arrhenius law)",
    "ring.heater.full_scale": "split into full_scale_v, full_scale_a and full_scale_w",
    "ring.optics.tpa_beta_m_per_w": "removed: nonlinear optics is excluded with a computed bound",
    "ring.optics.n2_m2_per_w": "removed: nonlinear optics is excluded with a computed bound",
    "ring.optics.mode_area_um2": "removed: nonlinear optics is excluded with a computed bound",
}

# shared conditions, records

_THERMAL_ROM = when("fidelity.thermal_model", "rom")
_THERMAL_INTERIM = when("fidelity.thermal_model", "interim")
_THERMO_OPTIC_ORDER2 = when("fidelity.thermo_optic_order", 2)
_JUNCTION_LUT = when("fidelity.junction_model", "lut")
_JUNCTION_ANALYTIC = when("fidelity.junction_model", "analytic")

_SITE_FIELDS = (
    P("x_um", doc="site x position", level="global"),
    P("y_um", doc="site y position", level="global"),
    S("kind", "enum", "site kind",
      enum=("ring", "heater", "aggressor", "monitor_pd", "temp_sensor", "driver", "other")),
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

# key list

_KEYS: list[Key] = [
    # meta
    S("meta.name", "str", "preset name"),
    S("meta.kind", "enum", "preset: strict (no bare physical numbers); experiment: bare physical numbers become "
      "overrides; corner: an extends: file that only selects alternatives; private: NDA data (provenance private); "
      "test: unit-test preset", enum=("preset", "experiment", "corner", "private", "test"), default="preset"),
    S("meta.description", "str", "free text"),
    S("meta.sources", "map", "extra source keys declared by this file {key: citation} (private and test presets)"),
    S("meta.generation_mix", "str", "free text: which hardware generation each block comes from"),
    S("meta.inventory", "map", "completeness inventory of docs/research/00_prior_notes_checklist.md: {section: "
      "{item: {status: key | excluded | unknown, keys: [...], reason: ...}}} (validated by "
      "tests/closure/test_presets.py)"),

    # fidelity
    S("fidelity.self_heating", "enum", "optical self-heating: the absorbed light heats its own ring",
      enum=("on", "off"), default="on"),
    S("fidelity.thermo_optic_order", "enum", "1: constant dlambda/dT; 2: s0*(1+beta*(T-T0))",
      enum=(1, 2), default=2),
    S("fidelity.thermal_model", "enum", "interim three-pole model with preset numbers (used until the FEM ROM of the "
      "heater family exists), or a FEM reduced-order model (ROM) artefact", enum=("interim", "rom"),
      default="interim"),
    S("fidelity.junction_model", "enum", "analytic depletion law or junction LUT artefact",
      enum=("analytic", "lut"), default="analytic"),
    S("fidelity.optics_model", "enum", "exact two-direction add-drop ring - the steady-state TCMT comparison is archived",
      enum=("exact",), default="exact"),
    S("fidelity.backscatter", "enum", "CW/CCW coupling by backscatter", enum=("off", "on"), default="off"),
    S("fidelity.shot_noise", "enum", "photodiode shot noise", enum=("on", "off"), default="on"),
    S("fidelity.pattern_noise", "enum", "ones-density noise of the data pattern",
      enum=("off", "binomial", "prbs7", "prbs9", "prbs15", "prbs23", "prbs31"), default="off"),
    S("fidelity.data_modulation", "enum", "mark-density mean or LUT correction", enum=("mean", "lut"),
      default="mean"),
    S("fidelity.nonlinear_optics", "enum", "TPA/FCA/Kerr: excluded with a computed bound", enum=("off",),
      default="off"),
    S("fidelity.heater_anchor", "enum", "how relation R2 turns the measured P_FSR into Z_heater->ring. fsr_integral "
      "(default): Z = dT_FSR/P_FSR, with the integral of s(T) over the P_FSR sweep equal to one FSR; this is the "
      "exact inversion of the anchor measurement. lock_slope (the first design form): Z = efficiency/s(T_lock); it "
      "is 4 % low at the g26 nominal, and it makes Z depend on each ring's sampled offset",
      enum=("lock_slope", "fsr_integral"), default="fsr_integral"),
    S("fidelity.heater_pwm_ripple", "enum", "ripple term of the dac_pwm driver law", enum=("off", "on"),
      default="off"),
    S("fidelity.plasma_dispersion", "enum", "free-carrier plasma-dispersion model of the junction",
      enum=("nedeljkovic2011", "soref_bennett1987"), default="nedeljkovic2011"),

    # sim
    S("sim.dt_plant_s", "float", "plant tick; every hardware period and latency is a whole number of ticks",
      required=True, bounds=(1e-8, 1e-4)),
    S("sim.substep_policy", "enum", "fixed: one substep per tick; auto: exact multi-tick thermal steps, with optics "
      "substeps only where needed", enum=("fixed", "auto"), default="fixed"),
    S("sim.min_samples_per_dither_period", "int", "auto substeps: least number of samples per dither period while "
      "a dither is active (a rule of the auto-substep design; realize() keeps it in TimingParams, but the current "
      "kernel does not read it)", default=8),
    S("sim.max_detuning_step_pm", "float", "auto substeps: the largest resonance move in one substep", default=0.2),
    S("sim.reset_protocol", "enum", "start-up history of Plant.reset(): laser_ramp (the lasers turn on over "
      "sim.reset_ramp_s with every heater at code_min), branch_low (lasers on at once, cold branch) or branch_high "
      "(lasers on at once, hot self-locked branch)", enum=("laser_ramp", "branch_low", "branch_high"),
      default="laser_ramp"),
    S("sim.reset_ramp_s", "float", "laser turn-on ramp of laser_ramp (>= 10 tau_slow)"),

    # channel plan
    P("channel_plan.center_thz", doc="centre of the per-direction grid", level="global", bounds=(150.0, 300.0)),
    P("channel_plan.spacing_ghz", doc="line spacing seen by one ring bank", level="global", bounds=(10.0, 5000.0)),
    P("channel_plan.n_lines", "int", "lines per bank", level="global", bounds=(1, 64)),
    P("channel_plan.lines", "table", "explicit line table (overrides centre/spacing)", level="global",
      fields=_LINE_FIELDS, table_kind="list"),
    P("channel_plan.ring_to_line", "list_int", "assigned line of each ring (default: identity)", level="global"),

    # laser
    P("laser.power_per_line_dbm", doc="power per line at the laser-module output", level="line",
      required=True, bounds=(-30.0, 30.0)),
    P("laser.line_offset_ghz", doc="static frequency error of each line from the grid", level="line",
      bounds=(-500.0, 500.0)),
    P("laser.rin_db_per_hz", doc="relative intensity noise", level="die", bounds=(-200.0, -100.0)),
    P("laser.linewidth_mhz", doc="laser linewidth", level="die"),
    S("laser.tune.present", "bool", "optional laser.tune actuator", default=False),
    P("laser.tune.bits", "int", "laser tuning DAC bits", required_if=when("laser.tune.present", True),
      bounds=(1, 32)),
    P("laser.tune.lsb_hz", doc="laser tuning step per code", required_if=when("laser.tune.present", True),
      bounds=(0.0, 1e12)),

    # optical path
    P("optical_path.chip_input_loss_db", doc="laser output to the first ring (fibre, coupler, splitters)",
      required=True, bounds=(0.0, 30.0)),
    P("optical_path.bus_loss_db_per_mm", doc="bus waveguide loss between rings", bounds=(0.0, 20.0)),
    P("optical_path.ring_spacing_on_bus_um", doc="bus length between adjacent rings", bounds=(0.0, 1e5)),

    # ring.geometry
    P("ring.geometry.radius_um", doc="ring radius = the phase reference radius r_ref of n_eff_bent; R3",
      level="ring", relation=("R3_geometry",), bounds=(1.0, 100.0)),
    P("ring.geometry.width_nm", doc="rib/strip width (informational; sets the artefact)", bounds=(100.0, 2000.0)),
    P("ring.geometry.film_nm", doc="silicon film thickness (the cross-section hypotheses use 160 or 200)",
      bounds=(50.0, 500.0)),
    P("ring.geometry.slab_nm", doc="rib slab thickness (0 for a strip)", bounds=(0.0, 300.0)),
    S("ring.geometry.xsec", "str", "cross-section hypothesis name (rib160, rib200, strip400x200, ...)"),
    P("ring.geometry.jacobian", "table", "geometry Jacobian {d: [input names], of: {output: [one coefficient per "
      "input]}}: how each output moves per nm of each geometry error", level="global", table_kind="map"),
    P("ring.geometry.jacobian_coupling", "table", "second Jacobian block (coupling artefact)", level="global",
      table_kind="map"),

    # ring.optics
    P("ring.optics.resonance_offset_nm", doc="cold-cavity blue offset: assigned line minus resonance, at the "
      "reference temperature and bias, heater off, P_in -> 0; R3", level="ring", required=True,
      relation=("R3_geometry",), bounds=(-50.0, 50.0)),
    P("ring.optics.resonance_offset_ref_temp_c", doc="temperature at which resonance_offset_nm is defined",
      required=True, bounds=(-50.0, 150.0)),
    P("ring.optics.resonance_offset_ref_bias_v", doc="reverse bias at which resonance_offset_nm is defined",
      bounds=(-1.0, 5.0)),
    P("ring.optics.n_eff", doc="effective index of the bent mode at the resonance, T_ref; R3",
      level="ring", required=True, relation=("R1_thermo_optic", "R3_geometry"), bounds=(1.0, 4.0)),
    P("ring.optics.n_g", doc="group index of the bent mode; R1, R3", level="ring", relation=("R1_thermo_optic",
      "R3_geometry", "R2_heater_efficiency", "R4_q"), bounds=(1.5, 6.0)),
    P("ring.optics.dn_g_dlambda_per_nm", doc="group-index dispersion", level="ring", bounds=(-0.1, 0.1)),
    P("ring.optics.fsr_nm", doc="FSR in wavelength at the assigned line wavelength, i.e. at the locked resonance "
      "(optional R3 member; FSR_lambda = lambda_line^2 FSR/c)",
      level="ring", relation=("R3_geometry",), bounds=(1.0, 50.0)),
    P("ring.optics.fsr_thz", doc="FSR in frequency (optional R3 member; the FSR hypothesis windows of "
      "uncertainty.fsr are in THz)",
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
    P("ring.optics.dln_kappa2_through_dlambda_per_nm", doc="through-coupler dispersion d ln K^2/d lambda, "
      "evaluated at each line", level="ring", bounds=(-1.0, 1.0)),
    P("ring.optics.dln_kappa2_drop_dlambda_per_nm", doc="drop-coupler dispersion", level="ring",
      bounds=(-1.0, 1.0)),
    P("ring.optics.dln_alpha_dt_ring_per_k", doc="loss change with the mode-weighted ring temperature",
      level="ring", bounds=(-0.1, 0.1)),
    P("ring.optics.dln_alpha_dt_heater_per_k", doc="loss change with the ROM heater temperature",
      level="ring", bounds=(-0.1, 0.1)),
    P("ring.optics.loss_budget.alpha_fca_db_per_cm", doc="free-carrier absorption of the junction at the reference "
      "bias", level="ring", relation=("R4_loss_budget",), bounds=(0.0, 1000.0)),
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
    P("ring.optics.alpha_l_per_k", doc="ring length expansion at T0, substrate-clamped",
      bounds=(0.0, 1e-4)),
    P("ring.optics.alpha_l_slope_per_k2", doc="d alpha_L / dT", bounds=(-1e-7, 1e-7), required_if=_THERMO_OPTIC_ORDER2),
    P("ring.optics.backscatter_splitting_ghz", doc="CW/CCW mode splitting 2|beta|/2pi (not the coupling rate)",
      level="ring", required_if=when("fidelity.backscatter", "on"), bounds=(0.0, 100.0)),
    S("ring.optics.backscatter_orders", "int", "M_b: number of independent complex beta values, one per longitudinal "
      "order, repeated periodically over the orders (use 5 or more)", default=5, bounds=(1, 64)),
    S("ring.optics.tcmt_orders", "int", "reserved legacy setting - unused by exact ring optics", default=3, bounds=(1, 63)),

    # ring.thermal
    P("ring.thermal.dlambda_dt_pm_per_k", doc="s0 = dlambda/dT at the ring temperature T0: the single thermo-optic "
      "anchor (R1)", level="ring", required=True, relation=("R1_thermo_optic", "R2_heater_z"),
      bounds=(10.0, 200.0)),
    P("ring.thermal.dlambda_dt_ref_ring_temp_c", doc="T0 of s0", required=True, relation=("R1_thermo_optic",),
      bounds=(-50.0, 250.0)),
    P("ring.thermal.dlambda_dt_beta_per_k", doc="beta of s(T) = s0*(1+beta*(T-T0))", required_if=_THERMO_OPTIC_ORDER2,
      relation=("R1_thermo_optic",), bounds=(-0.01, 0.01)),
    P("ring.thermal.z_heater_ring_k_per_mw", doc="Z_heater->ring(0) (R2)", level="ring", derived_only=True,
      relation=("R2_heater_z",)),
    P("ring.thermal.t_ring_lock_c", doc="ring temperature at lock when the ambient is the offset reference "
      "temperature (diagnostic)", level="ring", derived_only=True),
    P("ring.thermal.z_ratio_opt_heater", doc="Z_opt->ring(0)/Z_heater->ring(0) (>= 1 for an external heater, by "
      "reciprocity)", required=True, bounds=(0.1, 10.0)),
    P("ring.thermal.reference_rise_frac", doc="fraction of the ring's mode-weighted temperature rise registered by "
      "the thermometer behind Lightmatter's printed degC values (hypothesis H86_sensor: 0.77-0.86; 1 = the ring "
      "itself). Not used by the plant; replicas and degC scenario targets read it from PlantParams.values",
      bounds=(0.1, 1.0)),
    S("ring.thermal.family", "enum", "heater family hypothesis",
      enum=("inring_noundercut", "silicide_undercut", "metal_above"), default="inring_noundercut", per_world=True),
    S("ring.thermal.stack", "enum", "die-stack hypothesis", enum=("side_by_side", "stacked_eic"),
      default="side_by_side", per_world=True),
    S("ring.thermal.rom_file", "str", "extracted/thermal_rom_<family>_<film>.npz", required_if=_THERMAL_ROM,
      forbidden_if=_THERMAL_INTERIM),
    P("ring.thermal.tau_fast_s", doc="interim model, heater-to-own-ring path: fast pole", required_if=_THERMAL_INTERIM,
      forbidden_if=_THERMAL_ROM, bounds=(1e-8, 1e-2)),
    P("ring.thermal.share_fast_frac", doc="interim model, heater-to-own-ring path: DC share of the fast pole",
      required_if=_THERMAL_INTERIM, forbidden_if=_THERMAL_ROM, bounds=(0.0, 1.0)),
    P("ring.thermal.tau_slow_s", doc="interim model, heater-to-own-ring path: slow pole", required_if=_THERMAL_INTERIM,
      forbidden_if=_THERMAL_ROM, bounds=(1e-8, 1.0)),
    P("ring.thermal.tail_tau_s", doc="interim model, heater-to-own-ring path: die/package tail pole",
      required_if=_THERMAL_INTERIM, forbidden_if=_THERMAL_ROM, bounds=(1e-6, 10.0)),
    P("ring.thermal.tail_share_frac", doc="interim model, heater-to-own-ring path: DC share of the tail (slow share "
      "= 1 - fast - tail)", required_if=_THERMAL_INTERIM, forbidden_if=_THERMAL_ROM, bounds=(0.0, 1.0)),
    P("ring.thermal.selfheat_tau_s", doc="sub-us waveguide-core pole of the optical self-heating path",
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
    P("ring.thermal.site_floor_frac", doc="interim site coupling floor at long range (die sensors, far sites), "
      "fraction of the self DC: the package term; absent: xt_floor_frac", forbidden_if=_THERMAL_ROM,
      bounds=(0.0, 0.5)),
    P("ring.thermal.pd_offset_um", "list_float", "monitor PD offset from the ring centre (x, y)", level="global",
      forbidden_if=_THERMAL_ROM),
    P("ring.thermal.temp_sensor_offset_um", "list_float", "ring-attached sensor offset (x, y)", level="global",
      forbidden_if=_THERMAL_ROM),
    P("ring.thermal.driver_offset_um", "list_float", "driver offset from the ring centre (x, y)", level="global",
      forbidden_if=_THERMAL_ROM),
    S("ring.thermal.xt_tau_law", "enum", "crosstalk time-constant law: capped (monotone in distance) or literal (the "
      "interim law as first specified; not monotone)", enum=("capped", "literal"), default="capped"),
    S("ring.thermal.crosstalk_dynamics", "enum", "imec first-order crosstalk, or the self shape (textbook corner)",
      enum=("imec", "self"), default="imec"),
    S("ring.thermal.scaling_hypothesis", "enum", "how tau scales with the residue",
      enum=("resistance", "capacitance", "independent"), default="resistance", per_world=True),
    P("ring.thermal.xt_nearest_frac", doc="interim crosstalk to the nearest neighbour, fraction of the self DC",
      required_if=_THERMAL_INTERIM, forbidden_if=_THERMAL_ROM, bounds=(0.0, 0.5)),
    P("ring.thermal.xt_floor_frac", doc="interim crosstalk floor to every other ring", required_if=_THERMAL_INTERIM,
      forbidden_if=_THERMAL_ROM, bounds=(0.0, 0.5)),
    P("ring.thermal.xt_tau_ref_s", doc="tau_xt = max(xt_tau_min_s, xt_tau_ref_s*(d/xt_dist_ref_um)^2), d the "
      "heat-source-to-victim distance (centre distance less the ring radii)",
      required_if=_THERMAL_INTERIM, forbidden_if=_THERMAL_ROM, bounds=(1e-6, 1.0)),
    P("ring.thermal.xt_dist_ref_um", doc="reference heat-source-to-victim distance of xt_tau_ref_s (imec: 60 um at "
      "a 100 um pitch)", required_if=_THERMAL_INTERIM,
      forbidden_if=_THERMAL_ROM, bounds=(1.0, 1e4)),
    P("ring.thermal.xt_tau_min_s", doc="floor of the crosstalk time constant", required_if=_THERMAL_INTERIM,
      forbidden_if=_THERMAL_ROM, bounds=(1e-7, 1.0)),
    P("ring.thermal.xt_tau_far_s", doc="crosstalk time constant beyond xt_far_um (absent: the printed X2 table, see "
      "mrm.plant.thermal.crosstalk_tau_s)", forbidden_if=_THERMAL_ROM, bounds=(1e-6, 1.0)),
    P("ring.thermal.xt_far_um", doc="distance beyond which xt_tau_far_s applies", forbidden_if=_THERMAL_ROM,
      bounds=(1.0, 1e5)),
    P("ring.thermal.transport_delay_s", doc="optional per-pair transport delay", bounds=(0.0, 1e-3)),

    # ring.heater
    P("ring.heater.p_fsr_mw", doc="heater power per FSR: the heater anchor (R2)", level="ring",
      required=True, relation=("R2_heater_efficiency",), bounds=(1.0, 200.0)),
    P("ring.heater.p_fsr_ref_temp_c", doc="stage temperature of the P_FSR measurement (fsr_integral form)",
      bounds=(-50.0, 150.0), required_if=when("fidelity.heater_anchor", "fsr_integral")),
    P("ring.heater.p_fsr_ref_wavelength_nm", doc="wavelength of the line the P_FSR measurement was made on: R2 takes "
      "FSR_lambda there, so the efficiency and Z_heater->ring are the same on every line (absent: each ring's line)",
      relation=("R2_heater_efficiency", "R2_heater_z"), bounds=(500.0, 3000.0)),
    P("ring.heater.efficiency_nm_per_mw", doc="FSR/P_FSR (R2; derived)", level="ring", derived_only=True,
      relation=("R2_heater_efficiency", "R2_heater_z")),
    P("ring.heater.r_cold_ohm", doc="heater resistance at r_ref_temp_c", level="ring", required=True,
      bounds=(1.0, 1e6)),
    P("ring.heater.r_ref_temp_c", doc="temperature of r_cold_ohm", required=True, bounds=(-50.0, 150.0)),
    P("ring.heater.tcr_per_k", doc="R = R0*(1 + tcr*(T_heater - T_ref))", required=True,
      bounds=(-0.01, 0.01)),
    S("ring.heater.driver_mode", "enum", "heater driver law hypothesis: the quantity the DAC code sets",
      enum=("voltage", "current", "power", "dac_pwm"), default="voltage", per_world=True),
    P("ring.heater.full_scale_v", doc="DAC full scale in voltage drive",
      required_if=when("ring.heater.driver_mode", "voltage", "dac_pwm"), bounds=(0.0, 50.0)),
    P("ring.heater.full_scale_a", doc="DAC full scale in current drive",
      required_if=when("ring.heater.driver_mode", "current"), bounds=(0.0, 1.0)),
    P("ring.heater.full_scale_w", doc="DAC full scale in power drive",
      required_if=when("ring.heater.driver_mode", "power"), bounds=(0.0, 1.0)),
    P("ring.heater.dac_bits", "int", "heater DAC resolution", required=True, bounds=(1, 32)),
    P("ring.heater.code_min", "int", "lowest code (a hardware limit, not the measured demand)", bounds=(0, 2**32)),
    P("ring.heater.code_max", "int", "highest code", bounds=(0, 2**32)),
    P("ring.heater.slew_codes_per_s", doc="hardware slew limit (absent: unlimited)", bounds=(0.0, 1e15)),
    P("ring.heater.settle_tau_s", doc="analog settle filter", bounds=(0.0, 1e-2)),
    P("ring.heater.latency_s", doc="DAC latency (the only place this delay is stated)", required=True, tick="period",
      bounds=(0.0, 1e-2)),
    P("ring.heater.dac_inl_lsb", doc="DAC INL", bounds=(0.0, 1e4)),
    P("ring.heater.dac_dnl_lsb", doc="DAC DNL", bounds=(0.0, 1e4)),
    P("ring.heater.dac_max_update_hz", doc="max DAC update rate", bounds=(0.0, 1e10)),
    P("ring.heater.driver_compliance_v", doc="current-drive headroom", bounds=(0.0, 100.0)),
    P("ring.heater.pwm_period_s", doc="dac_pwm period (PAT-2: 2.56 us)",
      required_if=when("ring.heater.driver_mode", "dac_pwm"), bounds=(0.0, 1e-3)),
    P("ring.heater.pwm_bits", "int", "PWM bits of dac_pwm (PAT-2: 8)", bounds=(1, 16)),
    S("ring.heater.readback.present", "bool", "heater V/I readback", default=False),
    P("ring.heater.readback.adc_bits", "int", "readback ADC bits", bounds=(1, 32),
      required_if=when("ring.heater.readback.present", True)),
    P("ring.heater.readback.v_full_scale_v", doc="readback voltage full scale", bounds=(0.0, 100.0)),
    P("ring.heater.readback.i_full_scale_a", doc="readback current full scale", bounds=(0.0, 10.0)),
    P("ring.heater.readback.rate_hz", doc="readback rate", tick="rate", bounds=(0.0, 1e9)),
    P("ring.heater.readback.latency_s", doc="readback latency", tick="period", bounds=(0.0, 1.0)),
    P("ring.heater.readback.noise_rms_codes", doc="readback ADC noise per sample (absent -> 0)", bounds=(0.0, 1e6)),
    S("ring.heater.readback.average", "bool", "readback ADC averages V and I over its sample period (boxcar) instead "
      "of a track-and-hold of the last tick (mrm.plant.sensing.ReadbackHw)", default=False),

    # ring.junction
    P("ring.junction.fill_fraction", doc="fraction of the circumference with the junction (CAL-5, the resonance "
      "shift against reverse bias, constrains it only in LUT mode; in the analytic law it cancels)", level="ring",
      required=True, bounds=(0.0, 1.0)),
    S("ring.junction.lut_file", "str", "extracted/junction_<xsec>.json", required_if=_JUNCTION_LUT,
      forbidden_if=_JUNCTION_ANALYTIC),
    P("ring.junction.eff_at_0v_pm_per_v", doc="dlambda/dV at 0 V for the analytic law (derived in LUT mode)",
      level="ring", required_if=_JUNCTION_ANALYTIC, forbidden_if=_JUNCTION_LUT, bounds=(0.0, 1000.0)),
    P("ring.junction.v_bi_v", doc="built-in voltage", required_if=_JUNCTION_ANALYTIC, bounds=(0.0, 2.0)),
    P("ring.junction.depletion_exponent", doc="m of dlambda = A*((V+Vbi)^m - Vbi^m). Charge control (dlambda ~ "
      "removed charge) gives 1/2 for an abrupt junction and 2/3 for a linearly graded one (docs/research/"
      "03_mrm_compact_models.md, 'Efficiency against bias: the charge-control form'). An earlier design note used "
      "1/3..1/2, the depletion-width exponents. The bounds [1/3, 2/3] cover both: the span "
      "mrm.plant.electrical.analytic_junction_table accepts",
      required_if=_JUNCTION_ANALYTIC, bounds=(1.0 / 3.0, 2.0 / 3.0)),
    P("ring.junction.dqi_dv_frac_per_v", doc="relative Q_i change per volt (derived in LUT mode)", level="ring",
      forbidden_if=_JUNCTION_LUT, bounds=(-1.0, 1.0)),
    P("ring.junction.loss_ratio_q", doc="q = lambda*d_alpha/(4 pi d_n) of the plasma effect",
      bounds=(0.0, 1.0)),
    P("ring.junction.capacitance_at_0v_ff", doc="junction capacitance", level="ring", bounds=(0.0, 1000.0)),
    P("ring.junction.series_resistance_ohm", doc="junction series resistance", level="ring", bounds=(0.0, 1e4)),

    # ring.monitor
    P("ring.monitor.responsivity_a_per_w", doc="drop PD responsivity", level="ring", required=True,
      bounds=(0.0, 2.0)),
    P("ring.monitor.gain_scale", doc="lumped monitor-gain nuisance factor", bounds=(0.01, 100.0)),
    P("ring.monitor.dark_i0_na", doc="dark current at dark_t0_c", bounds=(0.0, 1e4)),
    P("ring.monitor.dark_t0_c", doc="reference temperature of dark_i0_na", bounds=(-50.0, 150.0)),
    P("ring.monitor.dark_activation_ev", doc="Arrhenius activation energy of the dark current", bounds=(0.0, 2.0)),
    P("ring.monitor.background_fraction", doc="stray light at the PD, as a fraction of the total bus power at the "
      "ring (all lines, not one line; i += gain_scale R f P_bus)", level="ring", bounds=(0.0, 1.0)),
    S("ring.monitor.afe_type", "enum", "monitor analog front end: rc (first-order filter) or integrate_dump (boxcar "
      "integrator, dumped at each ADC sample)", enum=("rc", "integrate_dump"), default="integrate_dump"),
    P("ring.monitor.integration_window_s", doc="integrate-and-dump boxcar length", tick="period",
      required_if=when("ring.monitor.afe_type", "integrate_dump"), bounds=(0.0, 1.0)),
    P("ring.monitor.afe_tau_s", doc="RC front-end time constant", required_if=when("ring.monitor.afe_type", "rc"),
      bounds=(0.0, 1.0)),
    P("ring.monitor.adc_rate_hz", doc="monitor ADC sample rate", required=True, tick="rate", bounds=(1.0, 1e9)),
    P("ring.monitor.adc_latency_s", doc="conversion latency (the only place this delay is stated)", required=True,
      tick="period", bounds=(0.0, 1.0)),
    P("ring.monitor.adc_bits", "int", "monitor ADC bits", required=True, bounds=(1, 32)),
    P("ring.monitor.adc_full_scale_ua", doc="PD current at ADC full scale", required=True, bounds=(0.0, 1e6)),
    P("ring.monitor.adc_offset_codes", doc="ADC offset", level="ring", bounds=(-1e6, 1e6)),
    P("ring.monitor.adc_gain_error_frac", doc="ADC gain error", level="ring", bounds=(-1.0, 1.0)),
    P("ring.monitor.adc_inl_lsb", doc="ADC INL", bounds=(0.0, 1e4)),
    P("ring.monitor.adc_dnl_lsb", doc="ADC DNL", bounds=(0.0, 1e4)),
    P("ring.monitor.input_noise_pa_per_rt_hz", doc="input-referred current noise density", bounds=(0.0, 1e6)),
    P("ring.monitor.flicker_corner_hz", doc="1/f corner", bounds=(0.0, 1e9)),
    P("ring.monitor.adc_ref_tempco_ppm_per_k", doc="tempco of the ADC reference about 25 degC, at the temperature "
      "of ring.monitor.adc_ref_site", bounds=(-1e4, 1e4)),
    S("ring.monitor.adc_ref_site", "enum", "where the monitor ADC reference sits, i.e. the temperature its tempco "
      "sees: package_base = thermal.t_base_k + thermal.xpu_rise_k (a converter on its own die beside the PIC; its "
      "own dissipation is not modelled); monitor_pd = each ring's PD observer (one converter per ring, next to its "
      "PD)", enum=("package_base", "monitor_pd"), default="package_base"),
    P("ring.monitor.pga_gains", "list_float", "PGA gain steps", level="global"),
    P("ring.monitor.adc_mux.channels_per_adc", "int", "rings per converter: ring n on converter n // C; they share "
      "its DNL and INL; round_robin needs C x integration window <= the ADC period", bounds=(1, 64)),
    S("ring.monitor.adc_mux.order", "enum", "mux order", enum=("round_robin", "bank"), default="round_robin"),
    # monitor details read by mrm.plant.sensing
    P("ring.monitor.baseline_ua", doc="analog baseline current subtracted before the PGA and ADC (PAT-2 i_os,det); "
      "absent -> 0", bounds=(0.0, 1e6)),
    S("ring.monitor.stuck_flagged", "bool", "a stuck converter clears monitor_valid; false = silent",
      default=True),
    S("ring.monitor.flicker_band_min_hz", "float", "lowest pole of the 1/f noise synthesis (mrm.plant.sensing)",
      default=1e-3),
    S("ring.monitor.reference_tap.present", "bool", "reference-tap PD on the bus before the bank; not built in the "
      "engine yet: true is refused when the plant is built (docs/ENGINE.md, 'Physics to expect, and open items for "
      "the next stage')", default=False),
    P("ring.monitor.reference_tap.coupling_frac", doc="bus power fraction tapped to the reference PD",
      required_if=when("ring.monitor.reference_tap.present", True), bounds=(0.0, 1.0)),
    P("ring.monitor.reference_tap.responsivity_a_per_w", doc="reference PD responsivity (absent: the drop PD's)",
      bounds=(0.0, 2.0)),
    P("ring.monitor.reference_tap.dark_i0_na", doc="reference PD dark current (absent: the drop PD's)",
      bounds=(0.0, 1e4)),

    # ring.bias_dac
    P("ring.bias_dac.bits", "int", "bias DAC bits", required=True, bounds=(1, 32)),
    P("ring.bias_dac.v_min_v", doc="bias range low end (printed G26 axis)", required=True, bounds=(-5.0, 10.0)),
    P("ring.bias_dac.v_max_v", doc="bias range high end", required=True, bounds=(-5.0, 10.0)),
    P("ring.bias_dac.update_rate_hz", doc="bias DAC update rate", tick="rate", bounds=(0.0, 1e10)),
    P("ring.bias_dac.latency_s", doc="bias DAC latency (the only place this delay is stated)", required=True,
      tick="period", bounds=(0.0, 1.0)),
    P("ring.bias_dac.slew_v_per_s", doc="bias slew limit", bounds=(0.0, 1e15)),
    P("ring.bias_dac.settle_tau_s", doc="bias settle filter", bounds=(0.0, 1.0)),
    P("ring.bias_dac.noise_rms_codes", doc="bias DAC output noise", bounds=(0.0, 1e6)),

    # driver
    P("driver.output_resistance_ohm", doc="driver output resistance", bounds=(0.0, 1e4)),
    P("driver.static_power_mw", doc="driver static power", bounds=(0.0, 1e3)),
    P("driver.energy_per_transition_fj", doc="driver energy per transition", bounds=(0.0, 1e5)),
    P("driver.swing_pp_v", doc="junction peak-to-peak swing", level="ring", bounds=(0.0, 10.0)),
    S("driver.site", "str", "layout site of the driver"),
    S("driver.level_convention", "enum", "NRZ levels: centered (bias +- Vpp/2) or one_sided (bias - Vpp and bias)",
      enum=("centered", "one_sided"), default="centered", per_world=True),

    # temp sensors
    P("temp_sensors", "table", "temperature sensors", level="global", fields=_SENSOR_FIELDS,
      table_kind="list"),

    # layout, array
    P("layout.ring_pitch_um", doc="ring pitch of the default straight bank", bounds=(1.0, 1e4), level="global"),
    P("layout.sites", "table", "named sites {name: {x_um, y_um, kind}}; rings default to ring<i> on a "
      "line at ring_pitch_um", level="global", fields=_SITE_FIELDS, table_kind="map"),
    P("array.n_rings", "int", "rings per bank", required=True, level="global", bounds=(1, 256)),
    P("array.bus_order", "list_int", "order of the rings along the bus (default: identity)", level="global"),
    P("array.heater_total_power_max_w", doc="die heater power limit", bounds=(0.0, 100.0)),
    P("array.heater_supply_current_max_a", doc="die heater supply current limit", bounds=(0.0, 100.0)),

    # controller I/F
    # bind.<channel> - who drives that actuator channel (the port binding) - the controller or the scenario
    S("controller_interface.update_rate_hz", "float", "controller rate (a whole number of ticks)", tick="rate"),
    S("controller_interface.bind.tx.heater", "enum", "who drives tx.heater", enum=("controller",),
      default="controller"),
    S("controller_interface.bind.tx.bias", "enum", "who drives tx.bias", enum=("scenario", "controller"),
      default="scenario"),
    S("controller_interface.bind.laser.tune", "enum", "who drives laser.tune", enum=("scenario", "controller"),
      default="scenario"),
    S("controller_interface.bind.tx.assignment", "enum", "who drives tx.assignment (the line of each ring)",
      enum=("scenario", "controller"), default="scenario"),
    S("controller_interface.bind.datapath.bitstat_threshold", "enum", "who drives datapath.bitstat_threshold",
      enum=("controller", "scenario"), default="controller"),
    S("controller_interface.bind.monitor.pga_select", "enum", "who drives monitor.pga_select",
      enum=("controller", "scenario"), default="controller"),
    S("controller_interface.expose", "list_str", "sensor channels the controller may read (default: all bound)"),
    S("controller_interface.knowledge", "list_str", "knowledge tokens: what the controller may know (for example "
      "model_nominal)"),
    S("controller_interface.schedule", "enum", "simultaneous: each controller edge updates every ring; round_robin: "
      "edge j updates ring j % N", enum=("simultaneous", "round_robin"), default="simultaneous"),
    S("controller_interface.on_out_of_range", "enum", "an out-of-range controller command is clipped, or raises an "
      "error", enum=("clip", "error"), default="clip"),
    S("controller_interface.tick_jitter_s", "float", "largest delay of a controller edge from its nominal tick"),

    # controller
    S("controller.name", "str", "registry name or module:Class"),
    S("controller.params", "map", "controller parameters (validated by the controller)"),

    # scenario
    S("scenario.name", "str", "registry name"),
    S("scenario.stack", "list_str", "scenario stack"),
    S("scenario.params", "map", "scenario parameters (validated by the scenario registry)"),
    S("scenario.time_compression", "float", "speed-up of the slow exogenous signals (ramps, drift) only; never of "
      "the plant time constants or the controller rate", default=1.0),
    S("scenario.ambient_c", "float", "constant ambient when no scenario sets it"),
    S("scenario.bias_v", "float", "absolute bias when the scenario owns tx.bias",
      forbidden_if=when("controller_interface.bind.tx.bias", "controller")),
    S("scenario.bias_offset_v", "float", "additive bias offset (always allowed)"),
    S("scenario.aggressor", "map", "aggressor heat source (power into the thermal model)"),
    S("scenario.target", "map", "aggressor target at the victim ring {units: resonance | temperature, trajectory, "
      "...}; the aggressor power waveform is calibrated to meet it"),
    S("scenario.external_reflection", "map", "external reflection at the bus output {r_db, phase_drift_rad_per_s}; "
      "not built: the engine does not read it"),

    # metrics
    S("metrics.reloma_definition", "enum", "relOMA definition: oma_over_pin (relOMA = OMA / P_in)",
      enum=("oma_over_pin",), default="oma_over_pin"),
    S("metrics.ber_sample_period_s", "float", "G26 instrument interval", default=2.0),
    S("metrics.lock_threshold_pm", "float", "lock excursion threshold"),
    S("metrics.hysteresis_pm", "float", "excursion hysteresis"),
    S("metrics.allowed_assignments", "enum", "which ring-to-line assignments count as correct",
      enum=("fixed", "cyclic", "any"), default="fixed"),
    S("metrics.lock_reference", "enum", "reference of the lock error: the line, the oracle operating point, or "
      "metrics.lock_target_pm (mrm.sim.metrics)", enum=("line", "oracle", "target"), default="line"),
    S("metrics.lock_target_pm", "float", "lock-error reference detuning (pm) when metrics.lock_reference is target"),

    # experiment
    S("experiment.kind", "enum", "experiment type",
      enum=("run", "sweep", "mc", "compare", "robust", "acceptance", "locked_steady_state"), default="run"),
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
    # experiment-layer keys read by mrm.sim.experiment.
    # the defaults equal mrm.sim.experiment.EXTENSION_KEYS (the experiment hash uses the resolved options).
    # mrm.sim.experiment validates every value below again, and the open subtrees experiment.record / corners / self_reference / acceptance (OPEN_PREFIXES).
    S("experiment.start", "enum", "start state: cold (code_min, reset protocol, cold lags), cold_settled (code_min, "
      "settled, branch_low), warm (steady-state lock codes, settled, branch_high); default: the scenario's ResetSpec",
      enum=("cold", "cold_settled", "warm")),
    S("experiment.n_threads", "int", "kernel worker threads (default os.cpu_count())"),
    S("experiment.score_from_s", "float", "the metrics accumulate from this time (runner score_from_s)"),
    S("experiment.convergence_certificate", "str", "run directory whose time-compression convergence certificate "
      "covers this compressed scenario and plant"),
    S("experiment.compression_factors", "list_float", "time-compression factors of the convergence experiment",
      default=(1.0, 10.0, 100.0)),
    S("experiment.histogram", "bool", "detuning histogram accumulators", default=False),
    S("experiment.oracle", "bool", "in-kernel oracle (OMA penalty, lock_reference oracle)", default=True),
    S("experiment.p_in_w", "float", "locked_steady_state: power per line at the chip (default the plant's own)"),
    S("experiment.max_worlds_per_plant", "int", "largest B in one Plant (Monte Carlo chunk)", default=512),
    S("experiment.rank_metrics", "list_str", "compare: world metrics to rank by (lower is better; "
      "mrm.sim.experiment.RANKABLE)"),

    # datapath
    P("datapath.bitrate_gbps", doc="line rate (G26: 50 Gb/s NRZ)", bounds=(0.0, 1000.0)),
    P("datapath.bitstat_window_bits", "int", "bit-statistics window", bounds=(1, 1 << 20)),
    P("datapath.error_count_period_s", doc="error-counter period (G26: 2.0 s)", tick="period", bounds=(0.0, 1e4)),
    P("datapath.error_count_latency_s", doc="error-counter latency (the only place this delay is stated)",
      tick="period", bounds=(0.0, 1e4)),
    P("datapath.transition_density", doc="mean transition density of the traffic", bounds=(0.0, 1.0)),
    # BER and bit-statistics hardware - absent -> the module constants of mrm.plant.datapath (G26 worst lane -12.8 dBm OMA at 1e-9, 3 dB link, 16 GHz sampler)
    P("datapath.rx_sensitivity_oma_dbm", doc="receiver OMA sensitivity at datapath.ber_ref, per lane", level="ring",
      bounds=(-40.0, 10.0)),
    P("datapath.ber_ref", doc="BER at which rx_sensitivity_oma_dbm is quoted", bounds=(1e-30, 0.5)),
    P("datapath.link_loss_db", doc="loss from the TX bus output to the RX PD (the OMA reference planes)",
      bounds=(0.0, 60.0)),
    P("datapath.ber_floor", doc="BER floor of the error counter", bounds=(0.0, 0.5)),
    P("datapath.bitstat_bw_ghz", doc="bandwidth of the bit-rate sampler of the bit statistics",
      bounds=(0.0, 1000.0)),
    P("datapath.bitstat_noise_pa_per_rt_hz", doc="input noise density of the bit-rate sampler (absent: the monitor "
      "TIA density)", bounds=(0.0, 1e6)),
]

# registry, API

# every key by dotted path - filled from _KEYS just below (and by extend)
SCHEMA: dict[str, Key] = {}


# Register additional keys.
# a path that is already registered is an error.
def extend(keys: Iterable[Key]) -> None:
    for key in keys:
        if key.path in SCHEMA:
            raise ValueError(f"schema key {key.path!r} registered twice")
        SCHEMA[key.path] = key


extend(_KEYS)


# exact schema entry of `path` (None if unknown).
def lookup(path: str) -> Key | None:
    return SCHEMA.get(path)


# True when `path` lies inside an open subtree (validated elsewhere).
def is_open(path: str) -> bool:
    return any(path == p or path.startswith(p + ".") for p in OPEN_PREFIXES)


# Did-you-mean candidates for an unknown key - close full paths, then close siblings by last segment.
def suggest(path: str, n: int = 3) -> list[str]:
    if path in DELETED:
        return []
    keys = list(SCHEMA)
    out = difflib.get_close_matches(path, keys, n=n, cutoff=0.75)
    parent, _, last = path.rpartition(".")
    siblings = [k for k in keys if k.rpartition(".")[0] == parent]
    for candidate in difflib.get_close_matches(last, [s.rpartition(".")[2] for s in siblings], n=n, cutoff=0.6):
        candidate_path = f"{parent}.{candidate}" if parent else candidate
        if candidate_path not in out:
            out.append(candidate_path)
    return out[:n]


# Schema of one record field of a table key.
def table_field(table_key: Key, name: str) -> Key | None:
    for field in table_key.fields:
        if field.name == name or field.path == name:
            return field
    return None


# return suffix-parser errors for all keys and table fields.
# unrecognized suffixes can be valid unitless keys.
# this checks parser rules, not dimensional consistency.
def check_units() -> list[str]:
    errors = []
    for key in SCHEMA.values():
        try:
            parse_key(key.path)
        except UnitError as e:
            errors.append(str(e))
        for field in key.fields:
            try:
                parse_key(field.path)
            except UnitError as e:
                errors.append(f"{key.path}[]: {e}")
    return errors


# return all required and forbidden-key errors for the resolved config.
# get(path) supplies a switch value.
# present contains explicitly set paths.
# missing condition values use their schema defaults.
# every pair in a condition must hold.
def check_conditions(get: Callable[[str], Any], present: set[str]) -> list[str]:
    errors = []
    for key in SCHEMA.values():
        is_set = key.path in present
        if key.required and not is_set and not key.derived_only:
            errors.append(f"missing required key {key.path!r} ({key.doc})")
        if key.required_if and _condition_holds(key.required_if, get) and not is_set:
            condition_text = ", ".join(f"{p} in {list(v)}" for p, v in key.required_if)
            errors.append(f"missing key {key.path!r}: required when {condition_text}")
        if key.forbidden_if and _condition_holds(key.forbidden_if, get) and is_set:
            condition_text = ", ".join(f"{p} in {list(v)}" for p, v in key.forbidden_if)
            errors.append(f"key {key.path!r} is forbidden when {condition_text}")
    return errors


# Integer code of an enum value (its index in the schema enum) - used for per-world switch arrays.
def enum_code(path: str, value: Any) -> int:
    key = SCHEMA[path]
    return key.enum.index(value)


# all schema keys that declare membership in relation `name`.
def relation_members(name: str) -> list[str]:
    return [k.path for k in SCHEMA.values() if name in k.relation]


# helpers

# True when every (path, allowed values) pair of `cond` holds - an unset path takes its schema default.
def _condition_holds(cond: Condition, get: Callable[[str], Any]) -> bool:
    for path, values in cond:
        value = get(path)
        if value is None:
            key = SCHEMA.get(path)
            value = key.default if key is not None else None
        if value not in values:
            return False
    return True
