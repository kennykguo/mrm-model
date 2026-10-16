# Parameter format and the `mrm.params` machinery

This document is the contract for everyone who writes a preset (the content agent, students, Lightmatter) and for
everyone who consumes a realized plant (`PlantParams`, the engine agent). It implements 07 P0-3, P0-4, P1-14, P1-15,
P1-16 and the params part of P2-12. The authoritative key list is `mrm/params/schema.py`; the appendix at the end is
generated from it (`mrm params keys`).

Code: `mrm/params/{units,schema,config,relations,realize,derive,table}.py`, CLI in `mrm/cli.py`.

```python
from mrm import params
cfg = params.load("g26", overrides=["ring.optics.q_loaded=4000"], alts={"uncertainty.thermo_optic": "H86"})
pp = params.realize(cfg, batch=64, seeds=0)    # PlantParams (SI arrays)
nom = params.derive(cfg)                        # B = 1 view + diagnostics
print(params.explain(cfg, "ring.heater.efficiency_nm_per_mw"))
```

Contents: 1 files and sections, 2 leaves and provenance, 3 units, 4 artefacts, 5 alternatives, alts and overrides,
6 uncertainty entries, 7 variation entries, 8 relations, 9 realize, 10 PlantParams reference, 11 derive,
12 CLI, 13 deviations from 07, appendix: key list.

---

## 1. Files and sections

A config file is YAML. Presets live in `mrm/params/presets/<name>.yaml`; NDA presets in the git-ignored
`mrm/params/presets/private/`. `load("g26")` looks there; `load("path/to/file.yaml")` loads a path; `load(dict)`
loads a parsed mapping (no `extends:`).

Top level:

| key | meaning |
|---|---|
| `extends` | a preset name or path, or a list of them. Parents merge left to right, then this file. |
| `meta` | `name`, `kind`, `description`, `sources` (extra source keys `{KEY: citation}`), `generation_mix`. |
| `alts` | `{path: name}`: select leaf alternatives or pin uncertainty entries (section 5). |
| `relations` | `{R..._name: {tolerance_rel: x, note: ...}}` (section 8). |
| `fidelity sim channel_plan laser optical_path ring driver temp_sensors layout array` | the physical tree and switches. |
| `uncertainty` | epistemic entries, one draw per world (section 6). |
| `variation` | per-device entries keyed by config path, plus `geometry` (section 7). |
| `controller_interface controller scenario metrics experiment datapath` | interfaces and experiment settings. |

`meta.kind` decides how strict the loader is:

| kind | rule |
|---|---|
| `preset` (default) | every physical number is a leaf mapping (section 2). Bare physical numbers are an error. |
| `corner` | may contain only `extends`, `meta` and `alts` (07 D3: corner presets are thin files that select alternatives). |
| `private` | like `preset`; the only kind (with files under `presets/private/`) that may use provenance `private`. |
| `test` | like `preset`; when `mrm/params/sources.yaml` is missing, source keys are checked against `meta.sources`. |
| `experiment` | bare physical numbers are allowed and are tagged provenance `override`. Settings are bare. |

**Physical keys and settings.** The schema marks each key as *physical* (a claim about hardware or the world) or as a
*setting* (a switch, a simulation step, a metric threshold, an experiment choice, controller parameters). A setting
is written bare (`sim: {dt_plant_s: 1.0e-6}`); it may also be written as a leaf mapping, which is kept for display.
Physical keys need a leaf mapping in every strict kind.

**Open subtrees** are validated by their own owners, not by the schema: `controller.params`, `scenario.params`,
`scenario.aggressor`, `scenario.target`, `scenario.external_reflection`, `experiment.grid`,
`experiment.controllers`, `meta.sources`, and the experiment layer's `experiment.record`, `experiment.corners`,
`experiment.self_reference`, `experiment.acceptance` (validated by `mrm.sim.experiment`, which also checks every
other `experiment.*` key again; docs/handoff/experiments.md section 3).

**Unknown keys are errors** everywhere else, with a did-you-mean suggestion; keys deleted by 07 give the reason
(for example `ring.optics.absorbed_fraction` -> "deleted by 07 P0-2"). An override typo such as
`ring.heater.r_cold_ohms=50` therefore fails instead of silently running the nominal model (07 P1-15).

**YAML note.** The loader reads `1e6` and `1.0e6` as numbers (plain YAML 1.1 reads them as strings).

## 2. Leaves and provenance (07 P1-14)

```yaml
ring:
  optics:
    q_loaded:
      value: 3800                 # number, list (per ring / per line), string, or {from: ...}
      provenance: read-off        # closed set, below
      generation: G25             # closed set, below
      source: LM-T                # a key of mrm/params/sources.yaml (or meta.sources)
      range: [3500, 4200]         # optional: plausible range; value must lie inside; variation is truncated to it
      alternatives:               # optional, section 5
        talk: {value: 3500, provenance: printed}
      note: "read from the measured -3 dB width"
      textbook: REF.Q             # optional: the textbook field this leaf corresponds to
      disagrees: "..."            # optional: why we differ from the textbook (generates the disagreement table)
      targets: [CAL-1]            # required when provenance is calibrated (07 P0-5)
      residual: 0.02              # optional, calibrated leaves
      identifiable: true          # optional, calibrated leaves
```

Allowed leaf fields: `value from at provenance generation source range alternatives note textbook disagrees targets
residual identifiable`. Any other field is an error. A unit is never a field; it comes from the key suffix.

**Provenance** (closed set), strongest first:

`printed > private > read-off > calibrated > inferred > solver > literature > assumed > override`

plus `derived` (only on a free relation member with `value: null`; it resolves to the weakest input). Placement of
`private` (NDA measurements, just below printed) and `override` (nothing backs it, weakest) is this module's choice;
the others are ARCHITECTURE section 1. A qualifier is allowed: `solver(assumed-geometry)` ranks as the weaker of
`solver` and any provenance word inside the parentheses, here `assumed`.

**Generation** (closed set): `G25 G26 EVK100 L20 M1000 Guide OCP OCI GF PAT none`.

**Source**: a key of `mrm/params/sources.yaml` (owned by the content agent; it mirrors the textbook's tags plus
literature, GF and patents). A file may declare extra keys in `meta.sources` (private and test presets). While
`sources.yaml` does not exist the loader warns and skips the check (except for `kind: test`).

**Enforcement**
- Presets may not contain bare physical numbers. Experiment files and overrides may; they are tagged `override`.
- A child preset that changes `value` (or `from`/`at`) must restate `provenance` and `generation` in the same
  mapping. Other fields (source, range, note, ...) are inherited field by field. Alternatives merge by name.
- `calibrated` needs `targets`. `private` is allowed only in private presets.
- A key marked `derived only` in the schema (for example `ring.optics.dn_eff_dt_per_k`,
  `ring.heater.efficiency_nm_per_mw`) may not be set; a relation computes it (07 P0-3).
- A relation member that a preset wants computed is written `{value: null, provenance: derived}` or left out.
  `derived` on a key that no relation computes is an error.
- Values must lie inside their own `range` and inside the schema's hard physical bounds.
- Every hardware period and latency (schema rule "whole ticks") must be a whole number of `sim.dt_plant_s` ticks;
  the error names the key (07 P0-6). For a rate key the period 1/rate must be whole ticks.

**Levels and lists.** Each key has a level: `die` (one value per die, realized as (B, 1)), `ring` ((B, N)),
`line` ((B, L)) or `global` (not batched). A ring-level leaf may be a scalar (same for every ring) or a list of N
values (a designed stagger); a line-level leaf a scalar or a list of L values.

**Tables.** A `table` key (`channel_plan.lines`, `layout.sites`, `temp_sensors`, `ring.geometry.jacobian`) holds
records with bare numbers in their fields; provenance sits on the table leaf. A `layout.sites` record may carry a
free-text `note` (why the site is there; the model does not read it):

```yaml
temp_sensors:
  value:
    - {name: die0, attach: die, type: diode, sample_rate_hz: 1000.0, latency_s: 1.0e-3, noise_rms_k: 0.1,
       resolution_k: 0.0625, offset_sigma_k: 2.0, gain_error_sigma_frac: 0.01, self_heating_k: 0.0}
  provenance: assumed
  generation: none
  source: R04
layout:
  ring_pitch_um: {value: 30.0, provenance: assumed, generation: none, source: TEST}
  sites:                 # optional; rings default to ring<i> at (i * ring_pitch_um, 0)
    value: {agg0: {x_um: 0, y_um: 60, kind: aggressor}, pd0: {x_um: 0, y_um: 15, kind: monitor_pd}}
    provenance: assumed
    generation: none
    source: TEST
```

## 3. Units: the suffix grammar (07 P1-15)

A key is `<name>[_<qualifier>]_<unit>`. The unit is the longest run of trailing tokens that forms a unit
expression: factors multiply, `per` starts a denominator, `rt` is a square root of the next token, a trailing digit
2 or 3 is an exponent (`per_k2`, `um2`). Keys whose last tokens are not units are unitless (`n_g`, `q_loaded`,
`kappa2_per_pass`, `fill_fraction`).

Tokens (ARCHITECTURE section 1, plus additions marked +):

| group | tokens |
|---|---|
| length | `pm nm um mm m`, `cm`+ |
| time | `ps ns us ms s` |
| frequency (cycles per second, never angular) | `hz khz mhz ghz thz`, `gbps`+ (data rate) |
| power | `nw uw mw w` |
| current | `na ua ma a`, `pa`+ (picoampere; there is no pressure in the model) |
| voltage, resistance, capacitance | `mv v`, `ohm`, `ff f`, `pf`+ |
| temperature | `k` (kelvin; never "kilo"), `c` (Celsius, affine) |
| energy | `ev`, `j fj pj`+ |
| dimensionless | `pct frac`, `ppm`+, `codes`+, `lsb`+ |
| logarithmic | `db dbm` |
| operators | `per rt` |

Conversion to SI happens exactly once (`mrm.params.units.to_si`, used by `Config.si` and by `realize`):

| unit | SI value |
|---|---|
| linear units | value x scale (pm_per_k -> m/K, k_per_mw -> K/W, ...) |
| `_c` | K = degC + 273.15. Only for absolute temperatures. A difference or spread uses `_k`; `t_offset_c` or `x_sigma_c` is an error. |
| `_dbm` | W = 1e-3 x 10^(x/10). Only alone; a spread of a power is `_db`. |
| `_db` | the natural-log power ratio ln(P2/P1) = x ln(10)/10. A loss L_db becomes a transmission exp(-L); a sigma in dB becomes a sigma of ln P. |
| `_db_per_<length>` | the power attenuation coefficient alpha in 1/m, P(z) = P(0) exp(-alpha z): x ln(10)/10 / length. |
| `_db_per_hz` | a spectral density re 1/Hz (RIN): 10^(x/10) in 1/Hz. |

Sampling (uncertainty and variation) happens in the units of the key as written, before the conversion: an interval
on a `_dbm` key is uniform in dBm; a sigma on a `_db` key is additive in dB. A multiplicative draw (`mode: scale`,
`lognormal`, `rel_sigma`) of a `_c` or `_dbm` key is a load error (it has no physical meaning); use `mode: offset`.

The parser refuses keys whose special token sits inside a compound unit, because the longest-match rule would
otherwise misread them silently: `shift_ghz_per_c` (write `shift_ghz_per_k`; a coefficient per degree is per kelvin)
and `x_db_per_s` (only dB per length and dB per Hz are units).

## 4. Solver artefacts as values (ARCHITECTURE section 5)

```yaml
ring:
  optics:
    n_g: {from: "extracted/modes_rib160.json:n_g", generation: none, source: X-MODES}
  thermal:
    dlambda_dt_pm_per_k:
      value: {from: "extracted/modes_rib160.json:dlambda_dt_pm_per_k", at: {t_c: 25.0}}
      provenance: solver
      generation: none
      source: X-MODES
```

- `from: "extracted/<file>:<dotted.key>"` resolves under `mrm/data/` (or the `data_root` argument). List indices
  are allowed in the dotted key. Formats: `.json`, `.yaml`, `.npz` (arrays become lists).
- `at: {x: value}` interpolates linearly in a table node `{x: [...], value: [...]}`; outside the table is an error.
- **Sidecar and sha256.** The artefact must have a JSON sidecar whose `sha256` equals the sha256 of the artefact's
  bytes; otherwise `ArtefactError`. Sidecar names tried, in order: `<file>.sidecar.json`, `<stem>.sidecar.json`,
  `<file>.meta.json`, `<stem>.meta.json`, and for a non-JSON artefact `<stem>.json` (the `mrm.plant.thermal`
  convention for ROM `.npz` files).
- Provenance must be `solver` (it is set to `solver` when absent). If the sidecar's `inputs` record a provenance
  (`inputs: {geometry: {provenance: assumed, ...}}`), the leaf becomes `solver(assumed-geometry)`, which ranks as
  `assumed` (07 P1-14: "solver outputs carry their inputs").
- An alternative may also be a `from:` value; it is resolved when it is selected or drawn.

## 5. Alternatives, `alts` and overrides

**Alternatives on a leaf** (07 D3, P1-14) hold disputed values without forking presets. Each alternative is
`{value | from, provenance, generation?, source?, note?}`; missing generation and source are inherited from the
leaf. In strict presets an alternative must be a mapping, not a bare number. An alternative's value is checked like
the leaf's (type, list length, the schema's hard bounds) but not against the leaf's `range`, because stress
alternatives may lie outside the plausible range. Every leaf with alternatives also gets an implicit alternative
`default` (its own value), so a world can switch back. Alternatives are part of the resolved config and of
`Config.hash()`.

**`alts: {path: name}`** (in a file, or the `alts=` argument, or `--alt path=name`):
- a leaf path selects one of its alternatives: value, provenance, generation and source are replaced, and
  `explain` shows the choice. Any hypothesis entry that sets this leaf is pinned to the choice(s) that select the
  same alternative (or restricted to them when several do); an alternative that no choice reaches is an error. An
  interval or dist entry with `mode: absolute` that targets the leaf is pinned too, so the selection holds in
  every world (scale and offset entries keep modulating around it).
- `uncertainty.<entry>` pins a hypothesis to one choice in every world (and applies its leaf alternatives), or
  pins an interval/dist entry to `low`, `high`, `nominal` or a named corner.

**Overrides** (`overrides=` as a dict `{path: value}` or a list of `"path=value"` strings, values parsed as YAML;
`--set path=value` on the CLI):
- a physical leaf gets the value and provenance `override` (the old provenance is kept for `explain`); every
  interval entry that targets it is pinned so the override holds in all worlds; a hypothesis that touches it is
  pinned to its nominal and the override wins (with a warning). A bare physical number in an experiment file is
  the same kind of override and pins the same way;
- a setting gets the value; a per-world switch (for example `ring.heater.driver_mode=current`) also pins every
  hypothesis that sets it to the choice giving that value (restricted when several do; pinned to its nominal with a
  warning, the switch keeping the override, when none does), so the override holds in every world;
- `uncertainty.<entry>.<field>=v` and `variation.<path>.<field>=v` edit entries; `variation.<path>.<field>=v` on a
  path without an entry creates one (provenance `override`);
- unknown paths are errors with a did-you-mean.

Order inside `load`: extends merge -> `alts` of the files -> `alts` argument -> overrides -> validation.

## 6. Uncertainty entries (07 P0-4, section 4.8)

`uncertainty:` holds epistemic entries: each is drawn **once per world** (shape (B,)) and is shared by every ring.
An entry is any mapping with a `kind`; entries may be nested for grouping (`uncertainty.thermal.time_scale_fast`).
Every entry carries `provenance`, `generation` and `source` (in strict presets), plus an optional `note`.

### 6.1 `kind: hypothesis` (a discrete set)

```yaml
uncertainty:
  thermo_optic:                      # 07 D2: H_solver (default), H86, H86_sensor
    kind: hypothesis
    choices:
      H_solver: {ring.thermal.dlambda_dt_pm_per_k: default, ring.thermal.dlambda_dt_beta_per_k: default}
      H86:      {ring.thermal.dlambda_dt_pm_per_k: H86,     ring.thermal.dlambda_dt_beta_per_k: const}
    nominal: H_solver               # required; the nominal world and derive() use it
    weights: {H_solver: 2, H86: 1}  # optional; default equal
    provenance: assumed
    generation: none
    source: D07
  fsr:                               # short form: one target, choices are its alternative names
    kind: hypothesis
    target: ring.optics.fsr_thz
    choices: [default, window_a, g25_cluster, pdk_size]
    nominal: default
    provenance: assumed
    generation: none
    source: D07
  heater_driver:                     # a per-world switch (schema rule "per-world switch")
    kind: hypothesis
    choices:
      voltage: {ring.heater.driver_mode: voltage}
      dac_pwm: {ring.heater.driver_mode: dac_pwm}
    nominal: voltage
    provenance: assumed
    generation: none
    source: D07
```

A choice maps config paths to: an alternative name of that leaf (string), an explicit `{value, provenance}`
mapping, or, for a per-world switch (`ring.heater.driver_mode`, `ring.thermal.family`, `ring.thermal.stack`,
`ring.thermal.scaling_hypothesis`, `driver.level_convention`), one of its enum values. One choice may set several
leaves (H86 sets dlambda/dT and beta = 0, "a constant at the ring"). A hypothesis that changes a table leaf (for
example `ring.geometry.jacobian` per film thickness) is applied per world. Choice names are strings; `weights` keys
may be written as numbers (`{160: 0.7, 200: 0.3}`). The schema's fidelity rules are checked in every world a choice
reaches: a choice `ring.heater.driver_mode: power` needs `ring.heater.full_scale_w` even when the nominal world
uses voltage drive. The `nominal` choice must select what the config already holds (each leaf it names is at that
alternative, each switch at that value), because derive(), the parameter table and explain show the config as
written; to make another choice nominal, change the leaves or pin the entry with `alts`.

### 6.2 `kind: interval` and `kind: dist`

```yaml
uncertainty:
  q_loaded:
    kind: interval
    target: ring.optics.q_loaded     # optional; without a target the entry is a world variable
    mode: absolute                   # absolute (default) | scale | offset
    low: 3500
    high: 4200
    dist: uniform                    # uniform (default) | loguniform | triangular (mode at nominal)
    nominal: 3800                    # optional for a targeted absolute entry (default: the leaf value);
                                     # default 1 for scale and 0 for offset; required without a target
    corners: {low_q: 2600}           # named stress values for alts / corner presets
    provenance: read-off
    generation: G25
    source: LM-T
  monitor_dark:
    kind: dist
    target: ring.monitor.dark_i0_na
    dist: lognormal                  # normal | lognormal (lognormal: mean x exp(sigma z), so mean is the median)
    mean: 5.0
    sigma: 0.8
    truncate: [1.0, 30.0]            # optional, exact (inverse CDF; for lognormal in ln(x/mean)/sigma)
    provenance: assumed
    generation: GF
    source: R01
```

Values are in the units of the target key (or of the entry name's suffix). `mode: scale` multiplies the leaf,
`offset` adds to it. Targets must be given physical leaves (not free relation members, not derived-only keys).
Load checks: low <= high; loguniform needs low > 0; a triangular mode (`nominal`) inside [low, high]; lognormal
needs mean > 0; `truncate` is [low, high]; for `mode: absolute` the low/high (or truncate) values and the corners lie
inside the target's hard bounds. Pinning an absolute entry without a `nominal` to `nominal` keeps the leaf's own
value.

### 6.3 World variables (entries without a target)

The draw is stored in `PlantParams.world[name]` and passed to its consumer:
- `uncertainty.thermal.<field>` (interval or dist) -> `ThermalInputs.scales[<field>]` and the thermal hook's
  `ThermalScales`; allowed fields are the ThermalScales fields of `mrm.plant.thermal`: `dc_scale`,
  `time_scale_fast`, `time_scale_slow`, `crosstalk_scale`, `package_tail_tau_s`, `package_tail_share_frac`,
  `transport_delay_s` (07 D15, P1-2, 4.8). The scaling hypothesis is the per-world switch
  `ring.thermal.scaling_hypothesis`. A hypothesis under `uncertainty.thermal` (07 4.8 names `thermal.family` and
  `thermal.stack`) acts through the switches and leaves it sets, never as a scale.
- `uncertainty.timing.latency_scale` multiplies every I/O latency before it is quantized to ticks (07 4.8 "loop
  latency 0.5 to 2 times the declared total").
- any other untargeted entry is passed through unchanged in `PlantParams.world`.

### 6.4 How worlds are drawn

`realize(..., uncertainty=mode)` with mode `nominal` (every entry at its nominal), `sample` (counter RNG, default)
or `lhs` (a Latin hypercube over the batch, `scipy.stats.qmc`, one column per entry in sorted name order). A pinned
entry is constant in every mode. Hypothesis indices are stored as int64 in `world[name]` with the names in
`world_choices[name]`.

## 7. Variation entries (07 P1-16)

`variation:` holds per-device entries keyed by **config path** (the YAML key is the dotted path), drawn from a
stream separate from the worlds.

```yaml
variation:
  ring.heater.r_cold_ohm:
    level: ring                # die | ring | line (default: the leaf's level); die-level on a ring leaf is allowed
    dist: normal               # normal (default) | lognormal | uniform (sigma is the standard deviation)
    mode: offset               # offset (default) | scale (multiplicative; implied by lognormal and rel_sigma)
    die_sigma: 2.0             # leaf units (log-sigma in scale mode): one draw per device, shared by its rings
    ring_sigma: 0.5            # leaf units: per ring (use line_sigma for level line)
    corr_length_um: 500        # optional: exponential correlation exp(-d/l) of the ring component over layout sites
    truncate: true             # default true: keep values inside the leaf's range
    provenance: inferred
    generation: G25
    source: R01
  ring.monitor.responsivity_a_per_w: {level: ring, rel_sigma: 0.03, provenance: printed, generation: G25, source: LM-T}
```

Value of element i of device b:
- offset: `v = base + die_sigma*z_die[b] + ring_sigma*(C^(1/2) z)[b, i]`;
- scale with sigmas: `v = base * exp(die_sigma*z_die + ring_sigma*z_i)` (lognormal);
- `rel_sigma`: `v = base * (1 + rel_sigma*z_i) * exp(die_sigma*z_die)`.
Truncation: with a single independent normal offset component the draw is truncated exactly (inverse CDF);
otherwise (two components, a correlation length, scale mode, uniform) the value is clipped to the range and the
clipped fraction is reported in `PlantParams.diagnostics["clipped"]`.

### 7.1 Correlated geometry through Jacobians

```yaml
variation:
  geometry:
    width:       {die_sigma_nm: 3.0, ring_sigma_nm: 1.0, corr_length_um: 500}
    thickness:   {die_sigma_nm: 1.0, ring_sigma_nm: 0.3}
    slab:        {die_sigma_nm: 2.0}
    gap_through: {ring_sigma_nm: 2.0}
    gap_drop:    {ring_sigma_nm: 2.0}
    width_closes_gaps: true    # default false: a width error also closes both edge-to-edge gaps by itself
    jacobians: [ring.geometry.jacobian, ring.geometry.jacobian_coupling]   # default [ring.geometry.jacobian]
    provenance: literature
    generation: none
    source: BERKELEY
ring:
  geometry:
    jacobian:
      value: {from: "extracted/modes_rib160.json:jacobian"}    # or inline {d: [...], of: {...}}
      provenance: solver
      generation: none
      source: X-MODES
```

A Jacobian is `{d: [inputs], of: {output: [one coefficient per input]}}` per nm of geometry error (the modes
artefact format of ARCHITECTURE 5.1). Inputs: `width_nm`, `film_nm` (= `thickness_nm`), `slab_nm`,
`gap_through_nm`, `gap_drop_nm`. Outputs and what they move:

| output | leaf | how |
|---|---|---|
| `lambda_res_nm_per_nm` | `ring.optics.resonance_offset_nm` | offset -= sum(J dg) (offset = line - resonance) |
| `resonance_offset_nm_per_nm` | `ring.optics.resonance_offset_nm` | += |
| `n_eff_per_nm`, `n_g_per_nm` | `ring.optics.n_eff`, `ring.optics.n_g` | += |
| `dlambda_dt_pm_per_k_per_nm` | `ring.thermal.dlambda_dt_pm_per_k` | += |
| `ln_kappa2_through_per_nm`, `ln_kappa2_drop_per_nm` | the per-pass couplings K_t^2, K_d^2 (R4_q) | d ln K^2 at a fixed wavelength; see below |
| `alpha_rad_db_per_cm_per_nm` | `ring.optics.loss_budget.alpha_bend_radiation_db_per_cm` | x exp(sum(d_k dg_k)), d_k = d ln alpha_rad/dp_k from the table's `evaluations` (else J_k/leaf) |
| `ln_alpha_rad_per_nm` | the same leaf | x exp(sum(J dg)) |

The geometry error of each dimension is `die_sigma_nm*z_die + ring_sigma_nm*(C^(1/2) z)_i`; with
`width_closes_gaps: true` the gap errors the Jacobians see are the gap draws minus the width error (the coupling
artefact's derivatives are at fixed edge-to-edge gaps: a lithographic width bias closes each gap by itself, total =
d/dw - d/dgap). The radiation loss is exponential in the geometry (extract_modes handoff), so its secant is applied
as a log-slope and stays positive. A leaf output whose leaf is a free relation member is an error: the preset must
give that leaf as an input for the geometry to move it. Unknown inputs or outputs are errors. The direct path-keyed
sigmas remain available as overrides (07 P1-16).

**Sampled geometry and the relation anchors** (07 P1-16: one geometry error moves the resonance, the FSR and the
coupling together). With `variation: sample` and a `variation.geometry`, realize solves the relations twice: once
for the nominal-geometry device of every world (the arrays after the uncertainty draw and the grid, before any device
draw, with the device's own draws of the R4 coupling members q_loaded, q_through, q_drop, kappa_d_over_kappa), then
for the devices. The nominal solve gives
- the radius: a free radius (R3 solved from the FSR hypothesis and n_g) is held for every device of the world, a mask
  dimension, so a device's FSR = c/(n_g L) follows its own n_g;
- the order m, which every device keeps (the Jacobians are at fixed r_ref and order; n_eff_phase = m lambda_res/L is
  then the device's n_eff at its own resonance, and its correction from the configured n_eff can exceed lambda/(2L));
- the per-pass couplings K_a^2 the R4 anchors mean. A device's couplers are K^2 = K_a^2 exp(sum_k J_k dg_k +
  D d lambda_res) at its own cold resonance (D the coupler dispersion: the anchors hold at each device's cold
  resonance, which the geometry moved by d lambda_res, so at the line, where mrm.plant.phase evaluates K^2, the move
  is exactly sum_k J_k dg_k), Q_j = omega T_rt/(-ln(1 - K_j^2)) with the device's omega and T_rt, Q_i from the
  device's own loss budget, and q_loaded and kappa_d_over_kappa follow from the rate sum. A coupler therefore does not
  change when a device's loss or n_g does, and its Q_L does. The coupling outputs move a coupler whichever R4 member
  is free. `PlantParams.values` holds `derived.geometry.dln_kappa2_{through,drop}` and
  `derived.geometry.nominal_kappa2_{through,drop}`, the diagnostics `geometry_dlambda_res_nm`.
With `variation: off` (derive, nominal runs) none of this applies and the anchors hold for every ring. 07 wrote the sigma names as `width_nm_sigma`; that breaks the
suffix grammar, so they are `die_sigma_nm` / `ring_sigma_nm` per dimension here.

## 8. Relations (07 P0-3)

Each relation group needs exactly one free member per equation. A member is free when it is absent, `null`, or
tagged `derived`. All members given: the residual is computed and `OverdeterminedError` is raised, unless
`relations.<name>.tolerance_rel` is stated and the residual is inside it; then the member with the **weakest
provenance** is recomputed and `PlantParams.diagnostics["relation_notes"]` says so. Too many free members raise
`UnderdeterminedError`. After `realize()` every relation holds to 1e-12 (`PlantParams.diagnostics
["relation_residuals"]`; realize raises otherwise). Order of solution: R3, R1, R2, R4, R5.

| relation | members | equations (SI) |
|---|---|---|
| `R3_geometry` | `ring.geometry.radius_um`, `ring.optics.n_g`, `ring.optics.fsr_thz` or `fsr_nm`; `n_eff`, `resonance_offset_nm` -> `order` | FSR = c/(n_g 2 pi R). One of R, n_g, FSR free. m = round(n_eff L/lambda_res), lambda_res = lambda_line - offset; then n_eff_phase = m lambda_res/L (the order rounding moves into n_eff, \|correction\| <= lambda/(2L)), so the cold resonance sits exactly at the offset and the FSR keeps its hypothesis. `fsr_nm` is the wavelength FSR at the assigned line. With sampled geometry the radius and the order are the nominal-geometry device's and the FSR follows each device's n_g (section 7.1). |
| `R1_thermo_optic` | `ring.thermal.dlambda_dt_pm_per_k` (s0), `dlambda_dt_ref_ring_temp_c` (T0), `dlambda_dt_beta_per_k` (beta), `ring.optics.n_g`; -> `ring.optics.dn_eff_dt_per_k` | s(T) = (lambda/n_g)(dn_eff/dT(T) + n_eff alpha_L(T)) = s0 (1 + beta (T - T0)). dn_eff/dT(T0) = s0 n_g/lambda - n_eff alpha_L(T0); d2n_eff/dT2 = s0 beta n_g/lambda - n_eff dalpha_L/dT. beta = 0 when `fidelity.thermo_optic_order` = 1. s0 is the single anchor; setting dn_eff_dt is a load error. |
| `R2_heater_efficiency` | `ring.heater.p_fsr_mw` (anchor, 07 D16), optional `ring.heater.p_fsr_ref_wavelength_nm` -> `ring.heater.efficiency_nm_per_mw` | efficiency = FSR_lambda/P_FSR, FSR_lambda = lambda_ref^2 FSR/c at the line the P_FSR measurement was made on (`derived.heater.p_fsr_span_m`), so the efficiency and Z are the same on every line; without the key FSR_lambda is at each ring's assigned line (Z then follows lambda_line^2). The efficiency in pm/mW inherits the FSR hypothesis (07 D16). |
| `R2_heater_z` | -> `ring.thermal.z_heater_ring_k_per_mw`, `ring.thermal.t_ring_lock_c` | T_lock solves int_{T_ref}^{T_lock} s dT = resonance_offset mod FSR_lambda (`derived.optics.lock_offset_m`: the heater only red-shifts, so a sampled ring with a negative offset, or one above an FSR, locks the nearest order blue of its line; equal to the offset for 0 <= offset < FSR). `fidelity.heater_anchor: fsr_integral` (default since the integration): Z = dT_FSR/P_FSR with int_{T_meas}^{T_meas+dT_FSR} s dT = FSR_lambda of R2_heater_efficiency (needs `ring.heater.p_fsr_ref_temp_c`). `lock_slope` (07 as written): Z = efficiency/s(T_lock). See section 13 item 1. |
| `R4_loss_budget` | `ring.optics.loss_budget.{alpha_fca, alpha_other_abs, alpha_scatter, alpha_bend_radiation}_db_per_cm`, `q_intrinsic`, `heat_fraction` | all four components: alpha_i = sum, Q_i = 2 pi n_g/(lambda_res alpha_i), eta = (fca + other)/alpha_i (07 P0-2). Or no component and `heat_fraction` given: alpha_heat = eta alpha_i with alpha_i from Q_i. A partial budget is an error; neither is an error. |
| `R4_q` | `q_loaded`, `q_intrinsic`, `q_through`, `q_drop`, optional `kappa_d_over_kappa` | 1/Q_L = 1/Q_i + 1/Q_t + 1/Q_d (with kappa_d_over_kappa = x: Q_d = Q_L/x). Derived per-pass couplings use the exact energy decay per round trip: K^2 = 1 - exp(-omega T_rt/Q), a = exp(-omega T_rt/(2 Q_i)), T_rt = n_g L/c, so one parameter set serves the exact ring and TCMT (07 P2-4). With sampled geometry the anchors hold for the nominal-geometry device and each device's couplers move with its geometry (section 7.1). |
| `R5_latency` | one owner per I/O channel: `ring.monitor.adc_latency_s` (tx.monitor), `ring.heater.latency_s` (tx.heater), `ring.bias_dac.latency_s` (tx.bias), `ring.heater.readback.latency_s`, `datapath.error_count_latency_s`; sensors in `temp_sensors[].latency_s`; the controller declares `Controller.latency_ticks` in code | ticks = round(latency_s x latency_scale/dt). The deleted `controller_interface.*_latency_samples` keys are load errors. |

Overdetermined example (radius printed, n_g solver, FSR printed): give `relations: {R3_geometry: {tolerance_rel:
0.02}}`; within 2 % the weakest member (n_g) is recomputed.

## 9. `realize(cfg, batch, seeds, grid=None, *, uncertainty=None, variation=None, thermal_hook="auto")`

Order of work (07 P0-4):
1. base arrays from the leaves (config units), per-world switch codes;
2. uncertainty: one draw per world (section 6);
3. `grid` `{path: (B,) values}` (a sweep), in config units;
4. variation: geometry Jacobians, then path-keyed entries, then per-device draws of the DAC bit errors, the INL bow
   and the temperature-sensor offsets and gain errors;
5. conversion to SI;
6. relations R3, R1, R2, R4, R5 on the arrays; with a sampled geometry first on the nominal-geometry device of every
   world (its radius, order and anchor couplings, section 7.1), then on the devices;
7. thermal: `ThermalInputs` is always built; with `thermal_hook="auto"` and `mrm.plant.thermal` importable,
   `assemble(InterimRom(...) | load_rom(path), layout.sites, N, ThermalScales(...), temp_sensors=...,
   ring_pitch_um=...)` builds `PlantParams.thermal.system`. For the ROM model a per-world `dc_scale` =
   mean_i Z_heater->ring,i / Z_FEM(heater->ring) is applied (07 P1-2 `rom_dc_scale`, warned outside [0.5, 2]) and
   Z_opt comes from the ROM;
8. timing: periods and latencies to ticks; assertions (ADC period >= 1 tick, integrate-and-dump window >= 1 tick and
   <= one ADC period, controller period >= the heater DAC's minimum update period, bias DAC period >= 1 tick);
9. no silent nan: every optics, thermal, laser and junction array the plant always uses must be finite, and every
   world's heater driver law must have its full scale (`RealizeError` names the fields); every relation residual
   must be <= 1e-12 (nan fails);
10. the self-heating lag bound of 07 P0-1 item 2: the per-substep gain of the lagged cross-ring optical heating,
   g = sum_j (1 - exp(-dt/tau_ij)) x_ij G_sh max|dP_abs/dlambda|, must be below 0.05 (computed by
   `mrm.plant.thermal.cross_optical_lag_gain` when the system exists, otherwise from the interim numbers with the
   plant's own crosstalk-pole law `mrm.plant.thermal.crosstalk_tau_s`: nearest neighbour `xt_nearest_frac`, second
   rank `xt_next_frac` (else the floor), every other ring `xt_floor_frac`).

**Seeds and streams.** `seeds` is an int s (element b uses s + b, `mrm.core.rng.batch_seeds`), a (B,) array, or
`{"device": ..., "uncertainty": ...}` with either form. Worlds use `Stream.UNCERTAINTY`, devices `Stream.VARIATION`,
each keyed by (seed_b, stream) with the Philox counter (tick = crc32(entry name) & 0x7fffffff, ring = element
index, slot = component). Consequences: element b of a batch equals a B = 1 run with seed `seeds[b]`; adding or
removing an entry never changes another entry's draws; the same device seed gives the same devices in every world.
Results are identical under `NUMBA_DISABLE_JIT=1`.

**Modes.** `uncertainty`: `nominal | sample | lhs` (default `experiment.uncertainty`, else `sample`). `variation`:
`off | sample` (default `experiment.variation`, else `sample`).

## 10. `PlantParams` reference (for the engine)

`PlantParams` is a frozen dataclass. Every array is read-only, float64 or int64 (complex128 for
`optics.backscatter_rb`), SI. Shapes: die (B, 1), ring (B, N),
line (B, L), sensor (B, S); "global" fields are Python ints or floats. A field marked "absent -> x" uses the neutral
value x when the optional key is missing; the list of such fields is in `diagnostics["defaults_used"]`.
`PlantParams.select(b)` returns the B = 1 view of element b. It cuts every batched array, including those inside
`thermal.system` (the ThermalSystem's (B, M) residues and poles and its float (B,) arrays), `thermal.inputs` /
`thermal_inputs` (gains, interim values, scales, scaling-hypothesis names), `world`, `seeds`, `values` and the
diagnostics `rom_dc_scale`, `geometry_draws_nm` and `geometry_dlambda_res_nm`; element b of a B-run equals a B = 1
run with seed `seeds[b]`.

Top level: `batch, n_rings, n_lines`; groups below; `flags`; `world` {entry: (B,)}; `world_choices`; `seeds`
{device, uncertainty: (B,)}; `config_hash`; `values` {config path or `derived.*`: SI array} (every leaf and
derived value, plus `switch.<path>` codes); `derivations` {path: Derivation(relation, inputs, formula)};
`diagnostics` {relation_residuals, relation_notes, defaults_used, clipped, geometry_draws_nm (as drawn, before a
`width_closes_gaps` correction), geometry_dlambda_res_nm, selfheat_lag_gain, thermal_hook, rom_dc_scale,
uncertainty_mode, variation_mode}; `thermal_inputs`.

### optics (`OpticsParams`)

| field | unit | shape | meaning |
|---|---|---|---|
| `lambda_line_m` | m | B,N | nominal grid wavelength of each ring's assigned line (`channel_plan.ring_to_line`) |
| `lambda_res_cold_m` | m | B,N | cold-cavity resonance = lambda_line - offset, at `t_ref_k`, `bias_ref_v`, heater off, P_in -> 0 |
| `resonance_offset_m` | m | B,N | blue offset (line - cold resonance), after variation |
| `t_ref_k` | K | B,1 | reference temperature of the offset (`resonance_offset_ref_temp_c`) |
| `bias_ref_v` | V | B,1 | reference reverse bias of the offset (absent -> 0) |
| `radius_m`, `length_m` | m | B,N | r_ref of the phase model and L = 2 pi R |
| `n_eff` | 1 | B,N | phase-consistent n_eff at lambda_res_cold, T_ref: n_eff L = order lambda_res exactly |
| `n_eff_configured` | 1 | B,N | n_eff as configured (solver) before the order rounding |
| `n_g` | 1 | B,N | group index |
| `dn_g_dlambda_per_m` | 1/m | B,N | group-index dispersion (absent -> 0) |
| `order` | int64 | B,N | longitudinal order m |
| `fsr_hz` | Hz | B,N | c/(n_g L) |
| `round_trip_time_s` | s | B,N | n_g L/c |
| `q_loaded`, `q_intrinsic`, `q_through`, `q_drop` | 1 | B,N | at reference conditions; 1/Q_L = sum of the others exactly |
| `kappa2_through`, `kappa2_drop` | 1 | B,N | per-pass power coupling at lambda_res_cold, K^2 = 1 - exp(-omega T_rt/Q_c) |
| `round_trip_amplitude` | 1 | B,N | a = exp(-omega T_rt/(2 Q_i)) = exp(-alpha_i L/2) |
| `alpha_i_per_m` | 1/m | B,N | intrinsic power attenuation = 2 pi n_g/(lambda Q_i) |
| `alpha_heat_per_m` | 1/m | B,N | heat-producing part of alpha_i |
| `alpha_fca_per_m` | 1/m | B,N | junction FCA loss at the reference bias (nan without a loss budget) |
| `heat_fraction` | 1 | B,N | eta = alpha_heat/alpha_i (07 P0-2) |
| `dln_kappa2_through_dlambda_per_m`, `dln_kappa2_drop_dlambda_per_m` | 1/m | B,N | coupler dispersion, evaluate at each line (absent -> 0) |
| `dln_alpha_dt_ring_per_k`, `dln_alpha_dt_heater_per_k` | 1/K | B,N | loss vs ring / heater temperature (07 P1-5; absent -> 0) |
| `dn_eff_dt_per_k` | 1/K | B,N | dn_eff/dT at T0 (R1, inverted at lambda_line) |
| `d2n_eff_dt2_per_k2` | 1/K^2 | B,N | (R1) |
| `alpha_l_per_k`, `alpha_l_slope_per_k2` | 1/K, 1/K^2 | B,1 | length expansion at T0 and its slope (slope 0 for order 1) |
| `backscatter_splitting_hz` | Hz | B,N | CW/CCW mode splitting 2\|beta\|/2pi (0 unless `fidelity.backscatter: on`) |
| `backscatter_rb` | 1 (complex128) | B,N,M_b | rb_n = 2 pi beta_n/FSR per longitudinal order (applied as order n mod M_b, M_b = `ring.optics.backscatter_orders`); beta_n complex Gaussian with E\|beta_n\|^2 = (splitting/2)^2, uniform phase, per device; beta_n = splitting/2 with variation off; zeros when off |
| `tcmt_orders` | int | | orders summed by the TCMT model (`ring.optics.tcmt_orders`) |
| `fwhm_hz` | Hz | B,N | c/(lambda_res Q_L) |

Temperature model (R1): the resonance at ring temperature T (heater off, reference bias) is
`lambda_res_cold + F(T) - F(t_ref)`, `F(T) = s0 [(T - T0) + beta (T - T0)^2/2]`, with s0, beta, T0 in `thermal`.
This is the form the plant uses (`mrm.plant.selfheat`). `dn_eff_dt_per_k` and `d2n_eff_dt2_per_k2` are the R1
inversion with lambda = `lambda_line_m` (where a locked ring sits) and lambda/n_g held fixed: a phase-based model
with `n_eff(T) = n_eff + dn_eff_dt (T - T0) + d2n_eff_dt2 (T - T0)^2/2` and
`L(T) = L [1 + alpha_l (T - T0) + alpha_l_slope (T - T0)^2/2]` reproduces s(T) at the line wavelength to first
order. It is not exact: the exact phase relation dlambda/dT = (lambda(T)/n_g(lambda)) (dn_eff/dT + n_eff alpha_L)
also changes through lambda(T) and n_g(lambda) (s0^2/lambda is about 4 % of s0 beta at 67 pm/K, 1.4e-3 /K), and
evaluating it at the cold resonance instead of the line scales it by lambda_res_cold/lambda_line (about 0.6 % for a
7.6 nm offset).

### thermal (`ThermalParams`)

| field | unit | shape | meaning |
|---|---|---|---|
| `dlambda_dt_s0_m_per_k` | m/K | B,N | s0 at T0 (after the hypothesis and the geometry Jacobian) |
| `dlambda_dt_beta_per_k` | 1/K | B,1 | beta (0 for order 1 or H86) |
| `dlambda_dt_t0_k` | K | B,1 | T0 |
| `t_ring_lock_k` | K | B,N | ring temperature at lock when the ambient is `t_ref_k` |
| `s_lock_m_per_k` | m/K | B,N | s(T_lock), the ambient disturbance gain at lock |
| `heater_efficiency_m_per_w` | m/W | B,N | FSR_lambda/P_FSR (R2) |
| `z_heater_ring_k_per_w` | K/W | B,N | Z_heater->ring(0) (R2) |
| `z_ratio_opt_heater` | 1 | B,1 | Z_opt->ring(0)/Z_heater->ring(0) (ROM: from the FEM) |
| `z_opt_ring_k_per_w` | K/W | B,N | Z_opt->ring(0) |
| `self_heating_gain_m_per_w` | m/W | B,N | s(T_lock) Z_opt->ring per absorbed W |
| `system` | object | | `mrm.plant.thermal.ThermalSystem` from the hook, or None |
| `inputs` | `ThermalInputs` | | model, rom_path, interim {SI arrays (B,1); `*_um` keys renamed `*_m`}, z_heater_ring, z_opt_ring, the R1 anchor, family/stack codes and names, scales {name: (B,)}, scaling_hypothesis (B names), sites_um, temp_sensors, ring_pitch_um, dt_s |

### junction (`JunctionParams`)

| field | unit | shape | meaning |
|---|---|---|---|
| `model` | int | | 0 analytic, 1 LUT |
| `fill_fraction` | 1 | B,N | fraction of the circumference with the junction |
| `eff_at_0v_m_per_v` | m/V | B,N | dlambda/dV at 0 V (analytic input; LUT: lambda fill (dn_eff/dV at 0)/n_g, report only) |
| `v_bi_v`, `depletion_exponent` | V, 1 | B,1 | analytic law dlambda = A((V+Vbi)^m - Vbi^m) (nan in LUT mode) |
| `depletion_a_m` | m | B,N | A = eff0/(m Vbi^(m-1)) with V in volts |
| `dqi_dv_frac_per_v` | 1/V | B,N | absent -> 0 |
| `capacitance_at_0v_f`, `series_resistance_ohm` | F, ohm | B,N | absent -> 0 |
| `loss_ratio_q` | 1 | B,1 | q = lambda d_alpha/(4 pi d_n) (absent -> nan) |
| `lut_v_reverse_v`, `lut_dneff`, `lut_alpha_fca_per_m`, `lut_capacitance_f_per_m`, `lut_r_series_ohm_m` | SI | V | junction LUT for a full-circumference junction (empty when analytic) |

### heater (`HeaterParams`)

| field | unit | shape | meaning |
|---|---|---|---|
| `p_fsr_w` | W | B,N | heater power per FSR |
| `r_cold_ohm`, `r_ref_temp_k`, `tcr_per_k` | ohm, K, 1/K | B,N / B,1 / B,1 | R = R0 (1 + tcr (T_heater - T_ref)) |
| `driver_mode` | int64 | B,1 | 0 voltage, 1 current, 2 power, 3 dac_pwm (per world) |
| `full_scale_v`, `full_scale_a`, `full_scale_w` | V, A, W | B,1 | nan when not given |
| `dac_bits`, `code_min`, `code_max` | int | | code range (absent -> 0 and 2^bits - 1) |
| `slew_codes_per_s` | 1/s | B,1 | absent -> inf (07 P1-11) |
| `settle_tau_s` | s | B,1 | absent -> 0 |
| `dac_inl_lsb`, `dac_dnl_lsb` | LSB | B,1 | absent -> 0 |
| `compliance_v` | V | B,1 | absent -> inf |
| `pwm_period_s`, `pwm_bits` | s, int | B,1, int | dac_pwm only (nan / 0 otherwise) |
| `max_update_hz` | Hz | float | absent -> inf |
| `readback_present` | bool | | heater V/I readback |
| `supply_power_max_w`, `supply_current_max_a` | W, A | float | array limits (absent -> inf) |
| `dac_bit_err_z` | 1 | B,N,bits | standard normals for per-bit DAC errors (`dac_bit_errors_from_spec`) |
| `inl_bow_u` | 1 | B,N | uniforms in [-1, 1] for the INL bow |
| `readback_adc_bits` | int | | `ring.heater.readback.adc_bits` (0 when absent) |
| `readback_v_full_scale_v`, `readback_i_full_scale_a` | V, A | B,1 | readback full scales (nan when absent) |
| `readback_noise_rms_codes` | codes | B,1 | readback ADC noise per sample (absent -> 0) |
| `readback_average` | bool | | boxcar over the sample period instead of track-and-hold |

### bias_dac (`BiasDacParams`)

`bits` (int), `v_min_v`, `v_max_v` (V, B,1), `update_period_ticks` (int; 1 when absent), `slew_v_per_s` (V/s, B,1,
absent -> inf), `settle_tau_s` (s, B,1, absent -> 0), `noise_rms_codes` (B,1, absent -> 0).

### monitor (`MonitorParams`)

| field | unit | shape | meaning |
|---|---|---|---|
| `responsivity_a_per_w` | A/W | B,N | drop PD responsivity |
| `gain_scale` | 1 | B,1 | lumped monitor-gain nuisance factor (absent -> 1) |
| `dark_i0_a`, `dark_t0_k`, `dark_activation_j` | A, K, J | B,1 | Arrhenius dark current I_d0 exp(-(E_a/k)(1/T - 1/T0)) (absent -> 0; `dark_t0_c` is required when `dark_i0_na` is given, else nan) |
| `background_fraction` | 1 | B,N | stray light, a fraction of the total bus power at the ring, summed over every line (absent -> 0) |
| `afe_type`, `afe_tau_s` | int, s | | 0 rc, 1 integrate_dump |
| `integration_window_ticks`, `adc_period_ticks` | int | | |
| `adc_bits` | int | | |
| `adc_full_scale_a` | A | B,1 | PD current at ADC full scale |
| `adc_offset_codes`, `adc_gain_error_frac` | codes, 1 | B,N | absent -> 0 |
| `adc_inl_lsb`, `adc_dnl_lsb`, `input_noise_a_per_rt_hz`, `flicker_corner_hz`, `adc_ref_tempco_per_k` | | B,1 | absent -> 0 |
| `pga_gains` | 1 | G | empty when absent |
| `channels_per_adc` | int | | rings per converter (absent -> 1); ring n on converter n // C, sharing its DNL and INL bow; round_robin needs C x window <= ADC period |
| `mux_order` | str | | `ring.monitor.adc_mux.order` (round_robin \| bank) |
| `baseline_a` | A | B,1 | analog baseline subtracted before the ADC (PAT-2; absent -> 0) |
| `stuck_flagged` | bool | | a stuck converter clears `monitor_valid` |
| `flicker_band_min_hz` | Hz | float | lowest 1/f pole of the flicker synthesis |
| `ref_tap_present`, `ref_tap_coupling_frac`, `ref_tap_responsivity_a_per_w`, `ref_tap_dark_i0_a` | -, 1, A/W, A | -, B,1 | reference-tap PD (07 P1-9; nan = absent or the drop PD's value). Carried only: `MonitorHw.from_plant_params` (so the engine) refuses `ref_tap_present` (no channels yet) |
| `adc_ref_site` | str | | `ring.monitor.adc_ref_site`: package_base (the ADC reference at thermal.t_base_k + thermal.xpu_rise_k) \| monitor_pd (at each ring's PD observer; needs channels_per_adc 1) |

Absent `dark_activation_ev` gives 0.37 eV (`mrm.materials.ge_pd_dark_ea_ev`, inferred), not 0, so the dark current
keeps the Arrhenius law of 07 P1-9.

### temp_sensors (`TempSensorParams`)

`names` (S), `type` (S, 0 diode / 1 rtd / 2 thermistor), `site_index` (S, index into `layout.site_names`, -1 when
attached), `attach` (S strings), `sample_period_ticks` (S), `latency_ticks` (B,S, scaled), `noise_rms_k`,
`flicker_corner_hz`, `resolution_k`, `self_heating_k` (B,S), `offset_k` and `gain_error_frac` (B,S, drawn per
device from `offset_sigma_k` and `gain_error_sigma_frac`), `adc_bits` (S, int; 0 when absent) and `code_zero_k`
(S, K; the absolute temperature code 0 reads, from `code_zero_c`; nan when absent).

### datapath (`DatapathParams`)

`bitrate_hz` (B,1), `bitstat_window_bits` (int), `error_count_period_ticks` (int), `transition_density` (B,1),
`driver_output_resistance_ohm`, `driver_static_power_w`, `driver_energy_per_transition_j` (B,1), `swing_pp_v`
(B,N), `level_convention` (B,1 int: 0 centered, 1 one_sided); BER and bit-statistics hardware
`rx_sensitivity_oma_w` (B,N, W at `ber_ref`), `ber_ref` (B,1), `link_transmission` (B,1, exp(-link_loss)),
`ber_floor` (B,1), `bitstat_bw_hz` (B,1), `bitstat_noise_a_per_rt_hz` (B,1). Absent values are nan (or 0 for
powers); `mrm.plant.datapath.DatapathHw.from_plant_params` then uses its module defaults.

### laser (`LaserParams`)

`line_freq_hz` (B,L: grid + static error `laser.line_offset_ghz`), `line_freq_grid_hz` (L), `power_w` (B,L, at the
laser-module output), `input_transmission` (B,1, exp(-chip_input_loss)), `bus_loss_per_m` (B,1), `rin_per_hz` (B,1),
`linewidth_hz` (B,1), `direction` (L strings), `source_laser` (L strings, '' when the lines table does not say),
`pol` (L strings, te default). Lines are `center + (k - (L-1)/2) spacing` for k = 0..L-1 (increasing
frequency) unless `channel_plan.lines` is given.

### layout (`LayoutParams`, not batched)

`site_names`, `site_xy_m` (S,2), `site_kind`, `ring_site_index` (N), `ring_xy_m` (N,2), `ring_distance_m` (N,N),
`bus_order` (N: ring index at each bus position), `ring_to_line` (N), `ring_spacing_on_bus_m`.

### timing (`TimingParams`)

`dt_s`; `controller_period_ticks` (0 when no rate), `adc_period_ticks`, `integration_window_ticks`,
`readback_period_ticks` (ints); `monitor_latency_ticks`, `heater_latency_ticks`, `bias_latency_ticks`,
`readback_latency_ticks`, `error_count_latency_ticks` (B,1 int64, after the latency scale); `latency_scale` (B,1);
the `sim.*` settings the kernel needs: `reset_protocol` (0 laser_ramp, 1 branch_low, 2 branch_high),
`reset_ramp_ticks`, `substep_policy` (0 fixed, 1 auto), `max_detuning_step_m`, `min_samples_per_dither_period`.

### flags (`FidelityFlags`)

Ints `self_heating, thermo_optic_order, thermal_model, junction_model, optics_model, backscatter, shot_noise,
pattern_noise, data_modulation, heater_anchor, heater_pwm_ripple`; `as_array()` gives the int32 array in
`FidelityFlags.ORDER` (07 P0-9).

## 11. `derive(cfg)` (nominal view and diagnostics)

`derive` runs `realize(cfg, 1, uncertainty="nominal", variation="off")` and returns `Derived(params, diagnostics,
inputs, provenance, warnings)`. Diagnostics (per ring unless stated; display units):

| name | meaning |
|---|---|
| `implied_index_weight`, `index_weight_ceiling`, `index_weight_ratio` | dn_eff/dn_Si that s0 needs (oxide term from `ring.optics.index_weight_oxide` when given), the ceiling n_g/n_g,Si, and their ratio (> 1: above the materials ceiling, 07 D2) |
| `self_heating_gain_pm_per_mw` | s(T_lock) Z_opt (= efficiency x z_ratio under lock_slope) |
| `p_in_ring_dbm`, `p_abs_peak_mw` | power at the ring and absorbed power on resonance |
| `bistability_ratio` | G_sh P_abs,peak/(0.77 FWHM); a warning above 1 (07 P0-1) |
| `t_ring_lock_c`, `t_amb_order_hop_c` | lock temperature; ambient at which the heater reaches zero (peak lock, BND-2) |
| `next_order_distance_ghz`, `next_order_distance_fwhm` | other orders of a ring locked on its own line to any other TX line, from the plant's phase model (the lock-shifted cold comb with dispersion; each order's TCMT FWHM): the BND-1 replica's distance (`derive.next_order_distances`; warning below 2 FWHM) |
| `monitor_codes_per_uw_at_drop`, `heater_pm_per_code` (on `heater_code_grid`), `fwhm_pm`, `disturbance_pm_per_k`, `selfheat_pm_per_mw_in` | the identifiable block of 07 P1-9 |
| `heater_efficiency_nm_per_mw`, `z_heater_ring_k_per_mw`, `fsr_thz`, `fsr_nm`, `radius_um`, `rom_dc_scale` | anchors in display units |

Each diagnostic's provenance is the weakest of its inputs (07 P1-14).

## 12. CLI

`mrm params show | table | explain | diff | validate | keys` with `--set PATH=VALUE`, `--alt PATH=NAME`,
`--sources`, `--data-root`, `--no-thermal-hook`:

- `mrm params show g26` prints every leaf with value, unit and provenance;
- `mrm params table g26 --out docs/PARAMETERS.md` writes the generated parameter document (leaves, derived values
  with their weakest-input provenance, diagnostics, uncertainty, variation, materials, textbook disagreements);
  `tests/verification/test_params_cli.py` fails when the committed file is stale;
- `mrm params explain g26 ring.heater.efficiency_nm_per_mw` prints the provenance chain (also diagnostics,
  `derived.*` values and `uncertainty.<entry>`);
- `mrm params diff g26 g26_h86` (or `--alt-b` / `--set-b` on the second config) lists the differences;
- `mrm params validate g26 g26_*` loads, derives and realizes a sampled batch; exit code 1 on any error;
- `mrm params keys` prints the appendix below.

`mrm/cli.py` is a registry: other command groups register with `mrm.cli.register(name, help, setup, run)` from a
provider module listed in `mrm.cli.PROVIDERS` (`mrm.sim.cli`, `mrm.validation.cli`, ... are pre-listed and skipped
while they do not exist).

## 13. Deviations from 07 and doubts

1. **R2 form.** 07 writes Z_heater->ring(0) = efficiency/dlambda/dT(T_lock). With the g26 beta (1.24e-3 /K, the
   secant of the solver curve) and the 7.6 nm offset, T_lock ~ 134 degC, so this Z is 4.0 % lower (5.475 against
   5.704 K/mW) than the Z that makes the integral of s(T) over the P_FSR measurement equal one FSR; with per-ring
   offset variation it also makes a thermal resistance depend on the ring's offset. The integrator made
   `fidelity.heater_anchor: fsr_integral` the default (schema and g26): Z depends only on FSR, P_FSR and s(T), the
   exact inversion of the anchor measurement (07 D16), taken at the P_FSR measurement line
   (`ring.heater.p_fsr_ref_wavelength_nm`), so Z is the same on every line. `lock_slope` (07 as written) stays
   selectable. The two agree when beta = 0 (docs/handoff/integrator.md).
2. `ring.optics.fsr_thz` was added next to 07's `fsr_nm`, because the 07 P1-6 FSR windows are stated in THz.
3. Geometry sigmas are `die_sigma_nm` / `ring_sigma_nm` per dimension (07's `width_nm_sigma` breaks the grammar).
4. Provenance ranks of `private` and `override` are not in 07; see section 2.
5. The order rounding of R3 is absorbed by n_eff (section 8), not by L, so the FSR hypothesis is kept exactly.
6. dB values convert to a natural-log power ratio (section 3), so a `_db` key never becomes a linear factor
   silently; consumers apply `exp(-x)` for a loss.
7. 07 4.8 states the `uncertainty` entries informally; this document fixes the format (section 6).
8. R2 locks each ring on the order reachable by heating: T_lock uses resonance_offset mod FSR_lambda. 07 writes
   the offset; the two agree for the designed 0 <= offset < FSR, and the wrap keeps sampled rings with a negative
   offset (about 11 % of rings for a 7.6 nm mean with the Berkeley 6 nm/2 nm spreads) from getting a lock
   temperature below ambient (and, under lock_slope, a Z from it).
9. An FSR given by a hypothesis is the FSR: `derived.optics.fsr_hz`, `fsr_m`, the round-trip time and the heater
   efficiency inherit the FSR leaf's provenance (07 D16), not that of n_g, which R3 solved the radius from (with a
   sampled geometry a device's FSR also inherits its n_g's). A quantity taken at a ring's line (`fsr_m`, Q_i,
   K^2, dn_eff/dT) also inherits the channel plan's provenance (center, spacing, line count or the lines table,
   recorded as the derivation of `derived.optics.lambda_line_m`; 07 P1-14). `channel_plan.ring_to_line` is not an
   input: it is the design convention per-ring values are stated against (each ring is designed relative to its
   own line) and only permutes the plan's wavelengths.

---

## Appendix: key list

Generated by `mrm params keys` from `mrm/params/schema.py`; `tests/verification/test_params_cli.py` checks that it
is current.

<!-- keys:begin -->
| key | type | level | unit | rules | meaning |
|---|---|---|---|---|---|
| `meta.name` | str |  |  | setting | preset name |
| `meta.kind` | enum |  |  | setting; enum: preset, experiment, corner, private, test; default preset | preset: strict (no bare physical numbers); experiment: bare physical numbers become overrides; corner: an extends: file that only selects alternatives; private: NDA data (provenance private); test: unit-test preset |
| `meta.description` | str |  |  | setting | free text |
| `meta.sources` | map |  |  | setting | extra source keys declared by this file {key: citation} (private and test presets) |
| `meta.generation_mix` | str |  |  | setting | free text: which generation each block comes from (07 4.9 item 6) |
| `meta.inventory` | map |  |  | setting | completeness inventory of docs/research/00_prior_notes_checklist.md: {section: {item: {status: key \| excluded \| unknown, keys: [...], reason: ...}}} (validated by tests/closure/test_presets.py) |
| `fidelity.self_heating` | enum |  |  | setting; enum: on, off; default on | optical self-heating (07 P0-1) |
| `fidelity.thermo_optic_order` | enum |  |  | setting; enum: 1, 2; default 2 | 1: constant dlambda/dT; 2: s0*(1+beta*(T-T0)) (07 P0-3) |
| `fidelity.thermal_model` | enum |  |  | setting; enum: interim, rom; default interim | interim three-pole model (07 P1-2) or a FEM ROM artefact |
| `fidelity.junction_model` | enum |  |  | setting; enum: analytic, lut; default analytic | analytic depletion law or junction LUT artefact (07 P1-7) |
| `fidelity.optics_model` | enum |  |  | setting; enum: exact, tcmt; default exact | exact add-drop ring or TCMT (07 P2-4) |
| `fidelity.backscatter` | enum |  |  | setting; enum: off, on; default off | CW/CCW coupling (07 P2-2) |
| `fidelity.shot_noise` | enum |  |  | setting; enum: on, off; default on | photodiode shot noise |
| `fidelity.pattern_noise` | enum |  |  | setting; enum: off, binomial, prbs7, prbs9, prbs15, prbs23, prbs31; default off | ones-density noise (07 P0-11) |
| `fidelity.data_modulation` | enum |  |  | setting; enum: mean, lut; default mean | mark-density mean or LUT correction (07 P2-6) |
| `fidelity.nonlinear_optics` | enum |  |  | setting; enum: off; default off | TPA/FCA/Kerr excluded with a bound (07 P2-1) |
| `fidelity.heater_anchor` | enum |  |  | setting; enum: lock_slope, fsr_integral; default fsr_integral | R2 form: fsr_integral (default, integrator decision: Z = dT_FSR/P_FSR with the integral of s(T) over the P_FSR sweep equal to one FSR, the exact inversion of the anchor measurement, 07 D16) or lock_slope (07 P0-3 as written: Z = efficiency/s(T_lock); 4 % low at the g26 nominal, and it makes Z depend on each ring's sampled offset; docs/handoff/integrator.md) |
| `fidelity.heater_pwm_ripple` | enum |  |  | setting; enum: off, on; default off | dac_pwm ripple term (07 P1-11) |
| `fidelity.plasma_dispersion` | enum |  |  | setting; enum: nedeljkovic2011, soref_bennett1987; default nedeljkovic2011 | free-carrier model (07 P1-7) |
| `sim.dt_plant_s` | float |  | s | setting; required | plant tick (07 P0-6) |
| `sim.substep_policy` | enum |  |  | setting; enum: fixed, auto; default fixed | 07 P0-10 |
| `sim.min_samples_per_dither_period` | int |  |  | setting; default 8 | 07 P0-10 |
| `sim.max_detuning_step_pm` | float |  | pm | setting; default 0.2 | 07 P0-10 |
| `sim.reset_protocol` | enum |  |  | setting; enum: laser_ramp, branch_low, branch_high; default laser_ramp | 07 P0-1 item 6 |
| `sim.reset_ramp_s` | float |  | s | setting | laser turn-on ramp of laser_ramp (>= 10 tau_slow) |
| `channel_plan.center_thz` | float | global | thz |  | centre of the per-direction grid |
| `channel_plan.spacing_ghz` | float | global | ghz |  | line spacing seen by one ring bank |
| `channel_plan.n_lines` | int | global |  |  | lines per bank |
| `channel_plan.lines` | table (list: freq_thz, direction, source_laser, pol) | global |  |  | explicit line table (overrides centre/spacing) |
| `channel_plan.ring_to_line` | list_int | global |  |  | assigned line of each ring (default: identity) |
| `laser.power_per_line_dbm` | float | line | dbm | required | power per line at the laser-module output (07 P0-1 item 4) |
| `laser.line_offset_ghz` | float | line | ghz |  | static frequency error of each line from the grid |
| `laser.rin_db_per_hz` | float | die | db_per_hz |  | relative intensity noise |
| `laser.linewidth_mhz` | float | die | mhz |  | laser linewidth |
| `laser.tune.present` | bool |  |  | setting; default False | optional laser.tune actuator (07 P0-7, P2-8) |
| `laser.tune.bits` | int | die |  | required if laser.tune.present=True | laser tuning DAC bits |
| `laser.tune.lsb_hz` | float | die | hz | required if laser.tune.present=True | laser tuning step per code |
| `optical_path.chip_input_loss_db` | float | die | db | required | laser output to the first ring (fibre, coupler, splitters) |
| `optical_path.bus_loss_db_per_mm` | float | die | db_per_mm |  | bus waveguide loss between rings |
| `optical_path.ring_spacing_on_bus_um` | float | die | um |  | bus length between adjacent rings |
| `ring.geometry.radius_um` | float | ring | um |  | ring radius = the phase reference radius r_ref of n_eff_bent (07 P1-6); R3 |
| `ring.geometry.width_nm` | float | die | nm |  | rib/strip width (informational; sets the artefact) |
| `ring.geometry.film_nm` | float | die | nm |  | silicon film thickness (D21: 160 or 200) |
| `ring.geometry.slab_nm` | float | die | nm |  | rib slab thickness (0 for a strip) |
| `ring.geometry.xsec` | str |  |  | setting | cross-section hypothesis name (rib160, rib200, strip400x200, ...) |
| `ring.geometry.jacobian` | table | global |  |  | geometry Jacobian {d: [names], of: {output: [per d]}} (07 P1-16) |
| `ring.geometry.jacobian_coupling` | table | global |  |  | second Jacobian block (coupling artefact) |
| `ring.optics.resonance_offset_nm` | float | ring | nm | required | cold-cavity blue offset: assigned line minus resonance, at the reference temperature and bias, heater off, P_in -> 0 (07 P1-6); R3 |
| `ring.optics.resonance_offset_ref_temp_c` | float | die | c | required | temperature at which resonance_offset_nm is defined |
| `ring.optics.resonance_offset_ref_bias_v` | float | die | v |  | reverse bias at which resonance_offset_nm is defined |
| `ring.optics.n_eff` | float | ring |  | required | effective index of the bent mode at the resonance, T_ref (07 P1-6); R3 |
| `ring.optics.n_g` | float | ring |  |  | group index of the bent mode; R1, R3 |
| `ring.optics.dn_g_dlambda_per_nm` | float | ring | per_nm |  | group-index dispersion |
| `ring.optics.fsr_nm` | float | ring | nm |  | FSR in wavelength at the assigned line wavelength, i.e. at the locked resonance (optional R3 member; FSR_lambda = lambda_line^2 FSR/c) |
| `ring.optics.fsr_thz` | float | ring | thz |  | FSR in frequency (optional R3 member; the 07 P1-6 windows are in THz) |
| `ring.optics.order` | int | ring |  | derived only | longitudinal order m = round(n_eff*L/lambda_res) (R3) |
| `ring.optics.q_loaded` | float | ring |  |  | loaded Q at reference conditions; R4 |
| `ring.optics.q_intrinsic` | float | ring |  |  | intrinsic Q; R4 (derived from the loss budget when it is given) |
| `ring.optics.q_through` | float | ring |  |  | through-coupler Q; R4 |
| `ring.optics.q_drop` | float | ring |  |  | drop-coupler Q; R4 |
| `ring.optics.kappa_d_over_kappa` | float | ring |  |  | drop decay rate over total decay rate (sets q_drop = q_loaded/x) |
| `ring.optics.dln_kappa2_through_dlambda_per_nm` | float | ring | per_nm |  | through-coupler dispersion d ln K^2/d lambda (07 P1-5); evaluated at each line |
| `ring.optics.dln_kappa2_drop_dlambda_per_nm` | float | ring | per_nm |  | drop-coupler dispersion |
| `ring.optics.dln_alpha_dt_ring_per_k` | float | ring | per_k |  | loss change with the mode-weighted ring temperature (07 P1-5) |
| `ring.optics.dln_alpha_dt_heater_per_k` | float | ring | per_k |  | loss change with the ROM heater temperature (07 P1-5) |
| `ring.optics.loss_budget.alpha_fca_db_per_cm` | float | ring | db_per_cm |  | free-carrier absorption of the junction at the reference bias (07 P0-2) |
| `ring.optics.loss_budget.alpha_other_abs_db_per_cm` | float | ring | db_per_cm |  | other heat-producing absorption (heater doping, silicide, metal, surface states) |
| `ring.optics.loss_budget.alpha_scatter_db_per_cm` | float | ring | db_per_cm |  | sidewall scattering (no heat) |
| `ring.optics.loss_budget.alpha_bend_radiation_db_per_cm` | float | ring | db_per_cm |  | bend and slab radiation (no heat) |
| `ring.optics.heat_fraction` | float | ring |  |  | eta = heat-producing loss / intrinsic loss (derived from the budget, or given with q_intrinsic when no component is given) |
| `ring.optics.index_weight_oxide` | float | ring |  |  | dn_eff/dn_SiO2 of the bent mode (solver; used only by the implied_index_weight diagnostic) |
| `ring.optics.dn_eff_dt_per_k` | float | ring | per_k | derived only | thermo-optic coefficient of the bent mode at T0 (R1); never set it |
| `ring.optics.alpha_l_per_k` | float | die | per_k |  | ring length expansion at T0, substrate-clamped (07 P0-3) |
| `ring.optics.alpha_l_slope_per_k2` | float | die | per_k2 | required if fidelity.thermo_optic_order=2 | d alpha_L / dT |
| `ring.optics.backscatter_splitting_ghz` | float | ring | ghz | required if fidelity.backscatter=on | CW/CCW mode splitting 2\|beta\|/2pi (not the coupling rate) |
| `ring.optics.backscatter_orders` | int |  |  | setting; default 5 | M_b: independent complex beta per longitudinal order, applied periodically in the order (07 P2-2; plant_optics handoff: >= 5) |
| `ring.optics.tcmt_orders` | int |  |  | setting; default 3 | orders summed by the TCMT model (07 P2-4); use an odd number (symmetric about every resonance, plant_optics handoff) |
| `ring.thermal.dlambda_dt_pm_per_k` | float | ring | pm_per_k | required | s0 = dlambda/dT at the ring temperature T0 (single thermo-optic anchor, 07 P0-3 R1) |
| `ring.thermal.dlambda_dt_ref_ring_temp_c` | float | die | c | required | T0 of s0 |
| `ring.thermal.dlambda_dt_beta_per_k` | float | die | per_k | required if fidelity.thermo_optic_order=2 | beta of s(T) = s0*(1+beta*(T-T0)) |
| `ring.thermal.z_heater_ring_k_per_mw` | float | ring | k_per_mw | derived only | Z_heater->ring(0) (R2) |
| `ring.thermal.t_ring_lock_c` | float | ring | c | derived only | ring temperature at lock when the ambient is the offset reference temperature (diagnostic) |
| `ring.thermal.z_ratio_opt_heater` | float | die |  | required | Z_opt->ring(0)/Z_heater->ring(0) (>= 1 for an external heater, 07 P0-1 item 5) |
| `ring.thermal.reference_rise_frac` | float | die | frac |  | fraction of the ring's mode-weighted temperature rise registered by the thermometer behind Lightmatter's printed degC values (07 D2 H86_sensor: 0.77-0.86; 1 = the ring itself). Not used by the plant; replicas and degC scenario targets read it from PlantParams.values |
| `ring.thermal.family` | enum |  |  | setting; per-world switch; enum: inring_noundercut, silicide_undercut, metal_above; default inring_noundercut | heater family (07 P1-3) |
| `ring.thermal.stack` | enum |  |  | setting; per-world switch; enum: side_by_side, stacked_eic; default side_by_side | stack hypothesis (07 P1-3) |
| `ring.thermal.rom_file` | str |  |  | setting; required if fidelity.thermal_model=rom; forbidden if fidelity.thermal_model=interim | extracted/thermal_rom_<family>_<film>.npz |
| `ring.thermal.tau_fast_s` | float | die | s | required if fidelity.thermal_model=interim; forbidden if fidelity.thermal_model=rom | interim self path, fast pole (07 P1-2) |
| `ring.thermal.share_fast_frac` | float | die | frac | required if fidelity.thermal_model=interim; forbidden if fidelity.thermal_model=rom | interim self path, DC share of the fast pole |
| `ring.thermal.tau_slow_s` | float | die | s | required if fidelity.thermal_model=interim; forbidden if fidelity.thermal_model=rom | interim self path, slow pole |
| `ring.thermal.tail_tau_s` | float | die | s | required if fidelity.thermal_model=interim; forbidden if fidelity.thermal_model=rom | interim die/package tail pole |
| `ring.thermal.tail_share_frac` | float | die | frac | required if fidelity.thermal_model=interim; forbidden if fidelity.thermal_model=rom | DC share of the tail (slow share = 1 - fast - tail) |
| `ring.thermal.selfheat_tau_s` | float | die | s | required if fidelity.thermal_model=interim; forbidden if fidelity.thermal_model=rom | sub-us waveguide-core pole of the optical self-heating path (07 P0-1) |
| `ring.thermal.selfheat_share_frac` | float | die | frac | forbidden if fidelity.thermal_model=rom | DC share of Z_opt in the sub-us pole (rest follows the heater poles); absent: (z - 1)/z, the largest passive share (mrm.plant.thermal.InterimRom) |
| `ring.thermal.heater_self_ratio` | float | die |  | forbidden if fidelity.thermal_model=rom | R_th,H / Z_heater->ring(0): heater resistor observing its own heater |
| `ring.thermal.heater_node_tau_s` | float | die | s | forbidden if fidelity.thermal_model=rom | extra pole of the heater-resistor observer |
| `ring.thermal.xt_next_frac` | float | die | frac | forbidden if fidelity.thermal_model=rom | interim crosstalk to the second rank of neighbours |
| `ring.thermal.site_ref_um` | float | die | um | forbidden if fidelity.thermal_model=rom | interim site coupling: equals xt_nearest_frac at this distance |
| `ring.thermal.site_max_frac` | float | die | frac | forbidden if fidelity.thermal_model=rom | interim site coupling cap |
| `ring.thermal.site_floor_frac` | float | die | frac | forbidden if fidelity.thermal_model=rom | interim site coupling floor at long range (die sensors, far sites), fraction of the self DC: the package term; absent: xt_floor_frac |
| `ring.thermal.pd_offset_um` | list_float | global | um | forbidden if fidelity.thermal_model=rom | monitor PD offset from the ring centre (x, y) |
| `ring.thermal.temp_sensor_offset_um` | list_float | global | um | forbidden if fidelity.thermal_model=rom | ring-attached sensor offset (x, y) |
| `ring.thermal.driver_offset_um` | list_float | global | um | forbidden if fidelity.thermal_model=rom | driver offset from the ring centre (x, y) |
| `ring.thermal.xt_tau_law` | enum |  |  | setting; enum: capped, literal; default capped | capped (monotone) or literal (07 P1-2 as written) |
| `ring.thermal.crosstalk_dynamics` | enum |  |  | setting; enum: imec, self; default imec | imec first-order crosstalk, or the self shape (textbook corner) |
| `ring.thermal.scaling_hypothesis` | enum |  |  | setting; per-world switch; enum: resistance, capacitance, independent; default resistance | how tau scales with the residue (07 P1-2) |
| `ring.thermal.xt_nearest_frac` | float | die | frac | required if fidelity.thermal_model=interim; forbidden if fidelity.thermal_model=rom | interim crosstalk to the nearest neighbour, fraction of the self DC |
| `ring.thermal.xt_floor_frac` | float | die | frac | required if fidelity.thermal_model=interim; forbidden if fidelity.thermal_model=rom | interim crosstalk floor to every other ring |
| `ring.thermal.xt_tau_ref_s` | float | die | s | required if fidelity.thermal_model=interim; forbidden if fidelity.thermal_model=rom | tau_xt = max(xt_tau_min_s, xt_tau_ref_s*(d/xt_dist_ref_um)^2) (07 P1-2), d the heat-source-to-victim distance (centre distance less the ring radii) |
| `ring.thermal.xt_dist_ref_um` | float | die | um | required if fidelity.thermal_model=interim; forbidden if fidelity.thermal_model=rom | reference heat-source-to-victim distance of xt_tau_ref_s (imec: 60 um at a 100 um pitch) |
| `ring.thermal.xt_tau_min_s` | float | die | s | required if fidelity.thermal_model=interim; forbidden if fidelity.thermal_model=rom | floor of the crosstalk time constant |
| `ring.thermal.xt_tau_far_s` | float | die | s | forbidden if fidelity.thermal_model=rom | crosstalk time constant beyond xt_far_um (absent: the printed X2 table of 04 section 2.4) |
| `ring.thermal.xt_far_um` | float | die | um | forbidden if fidelity.thermal_model=rom | distance beyond which xt_tau_far_s applies |
| `ring.thermal.transport_delay_s` | float | die | s |  | optional per-pair transport delay |
| `ring.heater.p_fsr_mw` | float | ring | mw | required | heater power per FSR (the heater anchor, 07 D16, R2) |
| `ring.heater.p_fsr_ref_temp_c` | float | die | c | required if fidelity.heater_anchor=fsr_integral | stage temperature of the P_FSR measurement (fsr_integral form) |
| `ring.heater.p_fsr_ref_wavelength_nm` | float | die | nm |  | wavelength of the line the P_FSR measurement was made on: R2 takes FSR_lambda there, so the efficiency and Z_heater->ring are the same on every line (absent: each ring's line) |
| `ring.heater.efficiency_nm_per_mw` | float | ring | nm_per_mw | derived only | FSR/P_FSR (R2; derived) |
| `ring.heater.r_cold_ohm` | float | ring | ohm | required | heater resistance at r_ref_temp_c |
| `ring.heater.r_ref_temp_c` | float | die | c | required | temperature of r_cold_ohm |
| `ring.heater.tcr_per_k` | float | die | per_k | required | R = R0*(1 + tcr*(T_heater - T_ref)) (07 P1-2, P1-11) |
| `ring.heater.driver_mode` | enum |  |  | setting; per-world switch; enum: voltage, current, power, dac_pwm; default voltage | driver law hypothesis (07 P1-11) |
| `ring.heater.full_scale_v` | float | die | v | required if ring.heater.driver_mode=voltage|dac_pwm | DAC full scale in voltage drive |
| `ring.heater.full_scale_a` | float | die | a | required if ring.heater.driver_mode=current | DAC full scale in current drive |
| `ring.heater.full_scale_w` | float | die | w | required if ring.heater.driver_mode=power | DAC full scale in power drive |
| `ring.heater.dac_bits` | int | die |  | required | heater DAC resolution |
| `ring.heater.code_min` | int | die |  |  | lowest code (07 P1-11: hardware, not measured demand) |
| `ring.heater.code_max` | int | die |  |  | highest code |
| `ring.heater.slew_codes_per_s` | float | die | codes_per_s |  | hardware slew limit (absent: unlimited) |
| `ring.heater.settle_tau_s` | float | die | s |  | analog settle filter |
| `ring.heater.latency_s` | float | die | s | required; whole ticks (period) | DAC latency (one owner, 07 P0-6) |
| `ring.heater.dac_inl_lsb` | float | die | lsb |  | DAC INL |
| `ring.heater.dac_dnl_lsb` | float | die | lsb |  | DAC DNL |
| `ring.heater.dac_max_update_hz` | float | die | hz |  | max DAC update rate |
| `ring.heater.driver_compliance_v` | float | die | v |  | current-drive headroom |
| `ring.heater.pwm_period_s` | float | die | s | required if ring.heater.driver_mode=dac_pwm | dac_pwm period (PAT-2: 2.56 us) |
| `ring.heater.pwm_bits` | int | die |  |  | PWM bits of dac_pwm (PAT-2: 8) |
| `ring.heater.readback.present` | bool |  |  | setting; default False | heater V/I readback (07 P1-10) |
| `ring.heater.readback.adc_bits` | int | die |  | required if ring.heater.readback.present=True | readback ADC bits |
| `ring.heater.readback.v_full_scale_v` | float | die | v |  | readback voltage full scale |
| `ring.heater.readback.i_full_scale_a` | float | die | a |  | readback current full scale |
| `ring.heater.readback.rate_hz` | float | die | hz | whole ticks (rate) | readback rate |
| `ring.heater.readback.latency_s` | float | die | s | whole ticks (period) | readback latency |
| `ring.heater.readback.noise_rms_codes` | float | die | codes |  | readback ADC noise per sample (absent -> 0) |
| `ring.heater.readback.average` | bool |  |  | setting; default False | readback ADC averages V and I over its sample period (boxcar) instead of a track-and-hold of the last tick (mrm.plant.sensing.ReadbackHw) |
| `ring.junction.fill_fraction` | float | ring |  | required | fraction of the circumference with the junction (07 P1-7, CAL-5) |
| `ring.junction.lut_file` | str |  |  | setting; required if fidelity.junction_model=lut; forbidden if fidelity.junction_model=analytic | extracted/junction_<xsec>.json |
| `ring.junction.eff_at_0v_pm_per_v` | float | ring | pm_per_v | required if fidelity.junction_model=analytic; forbidden if fidelity.junction_model=lut | dlambda/dV at 0 V for the analytic law (derived in LUT mode) |
| `ring.junction.v_bi_v` | float | die | v | required if fidelity.junction_model=analytic | built-in voltage |
| `ring.junction.depletion_exponent` | float | die |  | required if fidelity.junction_model=analytic | m of dlambda = A*((V+Vbi)^m - Vbi^m): 1/2 (abrupt) to 2/3 (linearly graded) by charge control, dlambda ~ removed charge (03 section 5.3, plant_electrical note 1); 07 P1-7 wrote 1/3..1/2 (the depletion-width exponents); bounds [1/3, 2/3], the span mrm.plant.electrical.analytic_junction_table accepts |
| `ring.junction.dqi_dv_frac_per_v` | float | ring | frac_per_v | forbidden if fidelity.junction_model=lut | relative Q_i change per volt (derived in LUT mode) |
| `ring.junction.loss_ratio_q` | float | die |  |  | q = lambda*d_alpha/(4 pi d_n) of the plasma effect (03 section 5.4) |
| `ring.junction.capacitance_at_0v_ff` | float | ring | ff |  | junction capacitance |
| `ring.junction.series_resistance_ohm` | float | ring | ohm |  | junction series resistance |
| `ring.monitor.responsivity_a_per_w` | float | ring | a_per_w | required | drop PD responsivity |
| `ring.monitor.gain_scale` | float | die |  |  | lumped monitor-gain nuisance factor (07 P1-9) |
| `ring.monitor.dark_i0_na` | float | die | na |  | dark current at dark_t0_c (07 D20) |
| `ring.monitor.dark_t0_c` | float | die | c |  | reference temperature of dark_i0_na |
| `ring.monitor.dark_activation_ev` | float | die | ev |  | Arrhenius activation energy (07 P1-9) |
| `ring.monitor.background_fraction` | float | ring |  |  | stray light at the PD, as a fraction of the total bus power at the ring (all lines, not one line; i += gain_scale R f P_bus, 07 P1-9) |
| `ring.monitor.afe_type` | enum |  |  | setting; enum: rc, integrate_dump; default integrate_dump | 07 P0-7 |
| `ring.monitor.integration_window_s` | float | die | s | required if ring.monitor.afe_type=integrate_dump; whole ticks (period) | integrate-and-dump boxcar (07 P0-7) |
| `ring.monitor.afe_tau_s` | float | die | s | required if ring.monitor.afe_type=rc | RC front-end time constant |
| `ring.monitor.adc_rate_hz` | float | die | hz | required; whole ticks (rate) | monitor ADC sample rate |
| `ring.monitor.adc_latency_s` | float | die | s | required; whole ticks (period) | conversion latency (one owner, 07 P0-6) |
| `ring.monitor.adc_bits` | int | die |  | required | monitor ADC bits |
| `ring.monitor.adc_full_scale_ua` | float | die | ua | required | PD current at ADC full scale |
| `ring.monitor.adc_offset_codes` | float | ring | codes |  | ADC offset |
| `ring.monitor.adc_gain_error_frac` | float | ring | frac |  | ADC gain error |
| `ring.monitor.adc_inl_lsb` | float | die | lsb |  | ADC INL |
| `ring.monitor.adc_dnl_lsb` | float | die | lsb |  | ADC DNL |
| `ring.monitor.input_noise_pa_per_rt_hz` | float | die | pa_per_rt_hz |  | input-referred current noise density |
| `ring.monitor.flicker_corner_hz` | float | die | hz |  | 1/f corner |
| `ring.monitor.adc_ref_tempco_ppm_per_k` | float | die | ppm_per_k |  | tempco of the ADC reference about 25 degC, at the temperature of ring.monitor.adc_ref_site |
| `ring.monitor.adc_ref_site` | enum |  |  | setting; enum: package_base, monitor_pd; default package_base | where the monitor ADC reference sits, i.e. the temperature its tempco sees: package_base = thermal.t_base_k + thermal.xpu_rise_k (a converter on its own die beside the PIC, 07 P1-3; its own dissipation not modelled); monitor_pd = each ring's PD observer (one converter per ring, next to its PD) |
| `ring.monitor.pga_gains` | list_float | global |  |  | PGA gain steps (07 P1-10) |
| `ring.monitor.adc_mux.channels_per_adc` | int | die |  |  | rings per converter (07 P0-7): ring n on converter n // C; they share its DNL and INL; round_robin needs C x integration window <= the ADC period |
| `ring.monitor.adc_mux.order` | enum |  |  | setting; enum: round_robin, bank; default round_robin | mux order |
| `ring.monitor.baseline_ua` | float | die | ua |  | analog baseline current subtracted before the PGA and ADC (PAT-2 i_os,det, 07 P1-9); absent -> 0 |
| `ring.monitor.stuck_flagged` | bool |  |  | setting; default True | a stuck converter clears monitor_valid (07 P1-10); false = silent |
| `ring.monitor.flicker_band_min_hz` | float |  | hz | setting; default 0.001 | lowest pole of the 1/f noise synthesis (mrm.plant.sensing) |
| `ring.monitor.reference_tap.present` | bool |  |  | setting; default False | reference-tap PD on the bus before the bank (07 P1-9, 06 S8); not built in the engine yet: true is refused when the plant is built (docs/ENGINE.md section 10) |
| `ring.monitor.reference_tap.coupling_frac` | float | die | frac | required if ring.monitor.reference_tap.present=True | bus power fraction tapped to the reference PD |
| `ring.monitor.reference_tap.responsivity_a_per_w` | float | die | a_per_w |  | reference PD responsivity (absent: the drop PD's) |
| `ring.monitor.reference_tap.dark_i0_na` | float | die | na |  | reference PD dark current (absent: the drop PD's) |
| `ring.bias_dac.bits` | int | die |  | required | bias DAC bits (07 P0-7) |
| `ring.bias_dac.v_min_v` | float | die | v | required | bias range low end (printed G26 axis) |
| `ring.bias_dac.v_max_v` | float | die | v | required | bias range high end |
| `ring.bias_dac.update_rate_hz` | float | die | hz | whole ticks (rate) | bias DAC update rate |
| `ring.bias_dac.latency_s` | float | die | s | required; whole ticks (period) | bias DAC latency (one owner) |
| `ring.bias_dac.slew_v_per_s` | float | die | v_per_s |  | bias slew limit |
| `ring.bias_dac.settle_tau_s` | float | die | s |  | bias settle filter |
| `ring.bias_dac.noise_rms_codes` | float | die | codes |  | bias DAC output noise |
| `driver.output_resistance_ohm` | float | die | ohm |  | driver output resistance (07 P1-8) |
| `driver.static_power_mw` | float | die | mw |  | driver static power |
| `driver.energy_per_transition_fj` | float | die | fj |  | driver energy per transition |
| `driver.swing_pp_v` | float | ring | v |  | junction peak-to-peak swing |
| `driver.site` | str |  |  | setting | layout site of the driver (07 P1-8) |
| `driver.level_convention` | enum |  |  | setting; per-world switch; enum: centered, one_sided; default centered | NRZ levels: centered (bias +- Vpp/2, 07 P1-7) or one_sided (bias - Vpp and bias) |
| `temp_sensors` | table (list: name, site, attach, type, sample_rate_hz, latency_s, noise_rms_k, flicker_corner_hz, resolution_k, offset_sigma_k, gain_error_sigma_frac, self_heating_k, adc_bits, code_zero_c) | global |  |  | temperature sensors (07 P1-10) |
| `layout.ring_pitch_um` | float | global | um |  | ring pitch of the default straight bank |
| `layout.sites` | table (map: x_um, y_um, kind, note) | global |  |  | named sites {name: {x_um, y_um, kind}} (07 P1-2); rings default to ring<i> on a line at ring_pitch_um |
| `array.n_rings` | int | global |  | required | rings per bank |
| `array.bus_order` | list_int | global |  |  | order of the rings along the bus (default: identity) |
| `array.heater_total_power_max_w` | float | die | w |  | die heater power limit (07 P1-12) |
| `array.heater_supply_current_max_a` | float | die | a |  | die heater supply current limit |
| `controller_interface.update_rate_hz` | float |  | hz | setting; whole ticks (rate) | controller rate (a whole number of ticks) |
| `controller_interface.bind.tx.heater` | enum |  |  | setting; enum: controller; default controller | 07 P0-7 |
| `controller_interface.bind.tx.bias` | enum |  |  | setting; enum: scenario, controller; default scenario | 07 P0-7 |
| `controller_interface.bind.laser.tune` | enum |  |  | setting; enum: scenario, controller; default scenario | 07 P0-7 |
| `controller_interface.bind.tx.assignment` | enum |  |  | setting; enum: scenario, controller; default scenario | 07 P1-12 |
| `controller_interface.bind.datapath.bitstat_threshold` | enum |  |  | setting; enum: controller, scenario; default controller | 07 P0-11 |
| `controller_interface.bind.monitor.pga_select` | enum |  |  | setting; enum: controller, scenario; default controller | 07 P1-10 |
| `controller_interface.expose` | list_str |  |  | setting | sensor channels the controller may read (default: all bound) |
| `controller_interface.knowledge` | list_str |  |  | setting | what the controller may know (07 P1-1) |
| `controller_interface.schedule` | enum |  |  | setting; enum: simultaneous, round_robin; default simultaneous | 07 P2-7 |
| `controller_interface.on_out_of_range` | enum |  |  | setting; enum: clip, error; default clip | 07 P1-19 |
| `controller_interface.tick_jitter_s` | float |  | s | setting | 07 P2-7 |
| `controller.name` | str |  |  | setting | registry name or module:Class |
| `controller.params` | map |  |  | setting | controller parameters (validated by the controller) |
| `scenario.name` | str |  |  | setting | registry name |
| `scenario.stack` | list_str |  |  | setting | scenario stack |
| `scenario.params` | map |  |  | setting | scenario parameters (validated by the scenario registry) |
| `scenario.time_compression` | float |  |  | setting; default 1.0 | 07 P0-10 |
| `scenario.ambient_c` | float |  | c | setting | constant ambient when no scenario sets it |
| `scenario.bias_v` | float |  | v | setting; forbidden if controller_interface.bind.tx.bias=controller | absolute bias when the scenario owns tx.bias |
| `scenario.bias_offset_v` | float |  | v | setting | additive bias offset (always allowed) |
| `scenario.aggressor` | map |  |  | setting | 07 P1-4 |
| `scenario.target` | map |  |  | setting | 07 P1-4 |
| `scenario.external_reflection` | map |  |  | setting | 07 P2-2 |
| `metrics.reloma_definition` | enum |  |  | setting; enum: oma_over_pin; default oma_over_pin | 07 D9 |
| `metrics.ber_sample_period_s` | float |  | s | setting; default 2.0 | G26 instrument interval |
| `metrics.lock_threshold_pm` | float |  | pm | setting | lock excursion threshold |
| `metrics.hysteresis_pm` | float |  | pm | setting | excursion hysteresis |
| `metrics.allowed_assignments` | enum |  |  | setting; enum: fixed, cyclic, any; default fixed | 07 P1-12 |
| `metrics.lock_reference` | enum |  |  | setting; enum: line, oracle, target; default line | reference of the lock error: the line (07 literal), the oracle operating point, or metrics.lock_target_pm (mrm.sim.metrics; engine handoff 3 item 21) |
| `metrics.lock_target_pm` | float |  | pm | setting | lock-error reference detuning (pm) when metrics.lock_reference is target |
| `experiment.kind` | enum |  |  | setting; enum: run, sweep, mc, compare, robust, acceptance, quasi_static; default run | 07 P1-17 |
| `experiment.duration_s` | float |  | s | setting | simulated duration |
| `experiment.batch` | int |  |  | setting | batch size B |
| `experiment.n_mc` | int |  |  | setting | Monte Carlo worlds |
| `experiment.seeds.device` | int |  |  | setting | device (variation) seed base |
| `experiment.seeds.uncertainty` | int |  |  | setting | world seed base |
| `experiment.seeds.noise` | int |  |  | setting | noise seed base |
| `experiment.seeds.scenario` | int |  |  | setting | scenario seed base |
| `experiment.uncertainty` | enum |  |  | setting; enum: nominal, sample, lhs; default sample | how realize() draws worlds |
| `experiment.variation` | enum |  |  | setting; enum: off, sample; default sample | per-device variation on or off |
| `experiment.grid` | map |  |  | setting | sweep grid {path: [values]} |
| `experiment.controllers` | list_str |  |  | setting | controllers of a compare experiment |
| `experiment.start` | enum |  |  | setting; enum: cold, cold_settled, warm | start state: cold (code_min, reset protocol, cold lags), cold_settled (code_min, settled, branch_low), warm (quasi-static lock codes, settled, branch_high); default: the scenario's ResetSpec |
| `experiment.n_threads` | int |  |  | setting | kernel worker threads (default os.cpu_count()) |
| `experiment.score_from_s` | float |  | s | setting | the metrics accumulate from this time (runner score_from_s) |
| `experiment.convergence_certificate` | str |  |  | setting | run directory whose 07 P0-10 convergence certificate covers this compressed scenario and plant |
| `experiment.compression_factors` | list_float |  |  | setting; default (1.0, 10.0, 100.0) | time-compression factors of the 07 P0-10 convergence experiment |
| `experiment.histogram` | bool |  |  | setting; default False | detuning histogram accumulators (07 P0-8) |
| `experiment.oracle` | bool |  |  | setting; default True | in-kernel oracle (OMA penalty, lock_reference oracle) |
| `experiment.p_in_w` | float |  | w | setting | quasi_static: power per line at the chip (default the plant's own) |
| `experiment.max_worlds_per_plant` | int |  |  | setting; default 512 | largest B in one Plant (Monte Carlo chunk) |
| `experiment.rank_metrics` | list_str |  |  | setting | compare: world metrics to rank by (lower is better; mrm.sim.experiment.RANKABLE) |
| `datapath.bitrate_gbps` | float | die | gbps |  | line rate (G26: 50 Gb/s NRZ) |
| `datapath.bitstat_window_bits` | int | die |  |  | bit-statistics window (07 P0-11) |
| `datapath.error_count_period_s` | float | die | s | whole ticks (period) | error-counter period (G26: 2.0 s) |
| `datapath.error_count_latency_s` | float | die | s | whole ticks (period) | error-counter latency (one owner) |
| `datapath.transition_density` | float | die |  |  | mean transition density of the traffic |
| `datapath.rx_sensitivity_oma_dbm` | float | ring | dbm |  | receiver OMA sensitivity at datapath.ber_ref, per lane |
| `datapath.ber_ref` | float | die |  |  | BER at which rx_sensitivity_oma_dbm is quoted |
| `datapath.link_loss_db` | float | die | db |  | loss from the TX bus output to the RX PD (the OMA reference planes) |
| `datapath.ber_floor` | float | die |  |  | BER floor of the error counter |
| `datapath.bitstat_bw_ghz` | float | die | ghz |  | bandwidth of the bit-rate sampler of the bit statistics (07 P0-11) |
| `datapath.bitstat_noise_pa_per_rt_hz` | float | die | pa_per_rt_hz |  | input noise density of the bit-rate sampler (absent: the monitor TIA density) |
<!-- keys:end -->
