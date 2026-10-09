# Handoff: params machinery (`mrm/params/*`, `mrm/cli.py`)

Status: done. Owner files: `mrm/params/{__init__,units,schema,config,relations,realize,derive,table}.py`,
`mrm/cli.py`, `docs/PARAMETERS_FORMAT.md`, `tests/verification/test_params_{units,schema,config,realize,derive,cli}.py`,
`tests/closure/test_relations.py`, `tests/data/mini.yaml`. After the adversarial review (section 5) 142 tests
pass (1 skipped until `g26.yaml` and `docs/PARAMETERS.md` exist) in about 15 s, about 50 s with
`NUMBA_DISABLE_JIT=1`.

The format and the `PlantParams` field reference are in `docs/PARAMETERS_FORMAT.md` (the contract). This file lists
what other owners need to do, and the doubts.

## 1. Thermal hook (resolved against the published API)

My task text said `mrm.plant.thermal.assemble(rom_or_interim_spec, layout, scales)`. The thermal owner published
`assemble(rom, layout, n_rings, scales, *, temp_sensors, ring_pitch_um, ...)` (docs/handoff/plant_thermal.md).
`realize()` calls exactly that:

- interim: `InterimRom(z_heater_ring_k_per_w=(B, N) from R2, z_ratio_opt_heater=(B,), every ring.thermal.* interim
  key given in the preset as (B,) SI arrays, *_um keys as floats in um, pd/temp_sensor/driver offsets as tuples,
  xt_tau_law, crosstalk_dynamics)`; absent optional keys keep the InterimRom defaults (e.g. `selfheat_share_frac`
  None = (z - 1)/z, as recommended).
- ROM: `load_rom(data_root / ring.thermal.rom_file)` and `ThermalScales.dc_scale` = mean_i Z_heater->ring,i (R2) /
  Z_FEM(heater -> ring_mode_weighted), a global (B,) factor (the `"*"` placement you recommended); warned outside
  [0.5, 2] (07 P1-2 rom_dc_scale). Z_opt then comes from `selfheat_dc_k_per_w(system)`.
- scales: every untargeted `uncertainty.thermal.<field>` draw, where `<field>` must be a ThermalScales field, plus
  `scaling_hypothesis` from the per-world switch `ring.thermal.scaling_hypothesis` ((B,) names).
- layout: the raw `layout.sites` mapping in um, `N`, `temp_sensors` table, `ring_pitch_um`.
- the 07 P0-1 lag bound uses `cross_optical_lag_gain(system, dt, dP_abs/dT max)`.
- `thermal_hook=None` skips the call; `ThermalInputs` always holds every input.

Requests to plant_thermal:
1. ROM mode loses per-ring Z variation (`dc_scale` is per world only). If per-ring heater efficiency spread should
   reach the ROM path, accept a (B, N) `dc_scale` for the heater -> * groups.
2. Your doubt 2 ('capacitance' = 'independent'): the schema keeps the three 07 names; decide and tell me if one goes.

## 2. Counter RNG

`mrm.core.rng` existed when realize was written, so no stand-in is used. Keying: `fill_uniform(out, seeds,
Stream.UNCERTAINTY | Stream.VARIATION, tick = crc32(entry name) & 0x7fffffff, slot = component)`; element index =
the Philox `ring` word. Normals are inverse-CDF (`scipy.special.ndtri`) so truncation is exact. Element b equals a
B = 1 run with seed `seeds[b]` (tested), and results are identical under `NUMBA_DISABLE_JIT=1` (tested).

## 3. Requests

### content agent (presets, `mrm/params/sources.yaml`)
- `sources.yaml` is not mine. The loader accepts a mapping `{KEY: citation}`, a `sources:` sub-mapping, a list of
  keys or a list of `{key: ...}`. Until it exists, the loader warns and skips the check (strict for `kind: test`).
  validation_targets asked for these keys: `BU-45CLO CO22-JLT CO22-TCPMT CO24-JOM DEC19 GF-ECIO HC HOTI26 LM-P LM-T
  LM26-P LM26-S OCP PAT-4 PECC SIB SMTAI YOLE textbook-REF`.
- Write presets from `docs/PARAMETERS_FORMAT.md`; `tests/data/mini.yaml` is a worked example (hypotheses that set
  several leaves, an FSR hypothesis, intervals, world variables, path-keyed variation, geometry Jacobian).
- Anchor choice matters for variation: a geometry Jacobian output can move only a *given* leaf. With Q_L as the
  calibrated anchor (q_through free) the through-coupler gap cannot move K_t^2; give q_through (and leave q_loaded
  free) if the coupling spread should propagate. State the choice in the preset note.
- `H86_sensor` (07 D2) is H_solver at the ring plus a thermometer that sees 0.77 to 0.86 of the ring's rise. The
  plant side is just H_solver; the thermometer fraction belongs to the replicas and scenario targets in degC. It has
  no key yet; propose one (for example `scenario.params`) or ask me for a schema key.
- core_control_api suggests `ring.monitor.adc_bits` as a hypothesis (16 vs about 10); it is a plain int leaf now and
  an alternative + hypothesis entry works for it.
- Run `mrm params validate g26 g26_*` and `mrm params table g26 --out docs/PARAMETERS.md` (the staleness test in
  `tests/verification/test_params_cli.py` activates once both files exist).
- After adding schema keys, regenerate the appendix of PARAMETERS_FORMAT.md: the block between `<!-- keys:begin -->`
  and `<!-- keys:end -->` must equal `mrm params keys` (a test checks it).

### plant electrical (done from your list)
- Added: `ring.heater.pwm_bits`, `fidelity.heater_pwm_ripple`, `ring.junction.loss_ratio_q`,
  `driver.level_convention` (per-world switch: centered | one_sided).
- `HeaterParams` carries `dac_bit_err_z` (B, N, bits) standard normals and `inl_bow_u` (B, N) uniforms in [-1, 1]
  (stream variation) for `dac_bit_errors_from_spec` and the INL bow; `supply_power_max_w`, `supply_current_max_a`;
  `r_ref_temp_k`; `driver_mode` (B, 1) int codes in the schema order voltage, current, power, dac_pwm.
- `OpticsParams.alpha_fca_per_m` (junction FCA at the reference bias, from the loss budget) for
  `analytic_junction_table_from_ring`.
- Schema bounds of `ring.junction.depletion_exponent` are 0.2..0.7, which admits your 1/2..2/3 reading and 07's
  1/3..1/2; the doc string still quotes 07. Please settle it with the content agent.

### core_control_api (done from your list)
- Added `controller_interface.bind.{tx.assignment, datapath.bitstat_threshold, monitor.pga_select}`,
  `controller_interface.expose`, `laser.tune.{present, bits, lsb_hz}`, and the temp-sensor fields `adc_bits` and
  **`code_zero_c`**. `code_offset_c` breaks the suffix grammar (an "offset" in affine degC is a difference); please
  rename your `HARDWARE_KEYS` lookup to `code_zero_c` (the absolute temperature that code 0 reads).
- `Config.fingerprint()` (= `Config.hash()`) and `Config.to_dict()` exist for `ControllerSpec.nominal`.
  `ControllerSpec.from_config(load("tests/data/mini.yaml"), n_batch=2, nominal=cfg)` works.
- The `controller_knowledge` preset and the knowledge filter of 07 D13 are not built here (not in my task).

### plant optics (done from your list)
- `OpticsParams.backscatter_rb` (B, N, M_b) complex128, `rb_n = 2 pi beta_n/FSR`, beta_n complex Gaussian with
  E|beta_n|^2 = (splitting/2)^2 and uniform phase per device (stream variation, entry `ring.optics.backscatter`);
  `ring.optics.backscatter_orders` (default 5) and `ring.optics.tcmt_orders` (default 3) are settings.

### engine (plant.py, sim/kernel.py)
- Import with `from mrm.params.realize import PlantParams` (the attribute `mrm.params.realize` is the function,
  which shadows the submodule name; `import mrm.params.realize as m` gives the function).
- `realize.py` never imports `mrm.params.config` at module level; the 07 P0-4 AST test (nothing under `mrm/plant`
  imports `mrm.params.config`) is in `tests/verification/test_params_realize.py` and passes.
- `PlantParams.values` has every SI value by config path (and `derived.*`), `PlantParams.select(b)` the B = 1 view.
- Line k = 0..L-1 has increasing frequency (`center + (k - (L-1)/2) spacing`) unless `channel_plan.lines` is given.
- dB keys convert to natural-log power ratios (see PARAMETERS_FORMAT section 3); `laser.input_transmission` is
  already linear.

### integrator
- `pyproject.toml`: nothing needed for params (`mrm = "mrm.cli:main"` exists; `params/presets/*.yaml` is in
  package-data).
- `mrm/cli.py` pre-lists providers `mrm.sim.cli`, `mrm.validation.cli`, `mrm.control.cli` (skipped while missing);
  each exposes `register_cli(register)`.

## 4. Contract doubts (also in PARAMETERS_FORMAT.md section 13)

1. **R2 heater form (physically doubtful).** 07: Z_heater->ring(0) = efficiency/s(T_lock). With beta = 1.4e-3 /K
   and a 7.6 nm offset, T_lock is about 130 degC and this Z is 5 to 15 % below the Z that reproduces the P_FSR
   measurement itself (the integral of s(T) over the heater-driven rise equal to one FSR). Worse, because realize
   derives after sampling, a ring with a larger sampled offset gets a higher T_lock and a lower thermal resistance:
   a stack property then depends on the ring's resonance offset. Default kept as 07 wrote (`lock_slope`); the
   alternative `fidelity.heater_anchor: fsr_integral` (needs `ring.heater.p_fsr_ref_temp_c`) depends only on FSR,
   P_FSR and s(T). I recommend `fsr_integral` as the default once the integrator agrees. Both agree when beta = 0.
2. `ring.optics.fsr_thz` added next to `fsr_nm` (the 07 windows are in THz). `fsr_nm` is taken at the line.
3. 07's `variation.geometry.width_nm_sigma` breaks the suffix grammar; the format is `width: {die_sigma_nm,
   ring_sigma_nm, corr_length_um}`.
4. R3 absorbs the order rounding in n_eff (<= lambda/(2L) ~ 0.016), keeping the offset and the FSR hypothesis exact.
5. `private` ranks just below `printed`, `override` below `assumed` (07 does not rank them).
6. Variation with two or more components (die + ring, or correlated) is clipped to the leaf range, not truncated;
   the clipped fraction is reported. A single independent normal component is truncated exactly.
7. Uncertainty and variation are sampled in the units the key is written in (an interval on a dBm key is uniform in
   dBm).
8. `derive()` diagnostics use a peak lock and the reference bias for the self-heating at lock (hop temperature,
   BND-2) and neglect the oxide index weight unless `ring.optics.index_weight_oxide` is given (about 1 %).

## 5. Adversarial review (changes made by the reviewer)

Bugs found and fixed (each has a regression test):

1. `realize`: hypothesis `weights` with numeric keys (`{160: 0.7}`) were silently ignored (choices are strings).
2. `realize`: pinning an absolute interval without a `nominal` to `nominal` (`--alt uncertainty.q_loaded=nominal`)
   set the leaf to nan (q_loaded = nan, and realize did not notice).
3. `realize`: nan residuals and nan arrays passed silently (`isfinite(v) and v > 1e-12`). Now every optics,
   thermal, laser and junction array the plant uses must be finite, and every world's driver law must have its full
   scale (`RealizeError` names the fields).
4. `config`: the fidelity rules were checked only for the nominal value of a per-world switch; a hypothesis choice
   `driver_mode: power` without `full_scale_w` realized nan full scales. Now checked per choice.
5. `config`: an override of a per-world switch (`--set ring.heater.driver_mode=current`) was overwritten by the
   hypothesis draw in every world (even the nominal one). It now pins the hypothesis.
6. `config`: the nominal choice of a hypothesis could differ from the leaves as written, so derive(), the table and
   explain described a world that realize(nominal) does not build. Now a load error.
7. `realize`: a lognormal `dist` was truncated in the wrong space (`truncate: [1, 30]` gave 0.3 to 83).
8. `realize`: untargeted hypotheses under `uncertainty.thermal` (07 4.8's `thermal.family`, `thermal.stack`) were
   passed to `ThermalScales` and raised. Only interval/dist draws are scales now.
9. `realize`: the interim lag-bound fallback (no hook) raised KeyError without the optional `xt_far_um` /
   `xt_tau_far_s`, ignored `xt_tau_law`, the printed X2 table and `xt_next_frac`. It now uses
   `mrm.plant.thermal.crosstalk_tau_s` (the plant's own law).
10. `PlantParams.select(b)` left `thermal.system` (B, M), `thermal.inputs` and the diagnostics at B. It now cuts every
    nested batched array (int64 index arrays of the ThermalSystem are never cut).
11. `Config.hash()` ignored leaf alternatives, so two configs that realize differently shared a hash (run manifests,
    golden tests); a mix of int and str dict keys made `hash()` raise. Drawing a `from:` alternative also mutated
    the config (and its hash). All fixed.
12. `units`: `shift_ghz_per_c` parsed silently as name `shift_ghz_per` in absolute degC, `x_db_per_s` as `x_db` in
    1/s. Both are now UnitErrors (a coefficient per degree is `_per_k`).
13. `config`: scale/lognormal/rel_sigma draws of `_c` or `_dbm` keys were accepted; alternatives were never
    type- or bounds-checked; `range` was not checked to be two numbers; a triangular mode outside [low, high] gave
    nan; absolute interval bounds and corners were not checked against the hard bounds. All are load errors now.
14. `relations` R2: a sampled ring with a negative offset (about 11 % of rings for 7.6 nm with the 6 nm / 2 nm
    Berkeley spreads) got T_lock below ambient and, under lock_slope, a Z from it. T_lock now uses
    offset mod FSR_lambda (`derived.optics.lock_offset_m`, the heater-reachable order); unchanged for 0 <= offset < FSR.
15. `relations` R3 provenance: a given FSR's derived quantities took n_g's weaker provenance; the heater efficiency
    of mini showed `inferred` although FSR and P_FSR are printed (07 D16: the efficiency inherits the FSR
    hypothesis). Fixed.
16. `derive`: `heater_pm_per_code` used 2^bits as full-scale code; the plant (`electrical.dac_frac`) uses 2^bits - 1.
17. CLI: a realize error in `table`/`explain` gave a traceback; now `error: <type>: <message>` and exit 1.
18. `config`: a bare physical number in an experiment file was tagged `override` but did not pin the uncertainty
    entries that target it, so `ring.optics.q_loaded: 4000` in an experiment was redrawn in every world. It now
    behaves exactly like an `overrides=` value.
19. `config`: `{value: null, provenance: derived}` on a key that no relation computes (for example
    `ring.monitor.gain_scale`) silently realized the neutral default; it is now a load error.

Docs corrected: the PARAMETERS_FORMAT claim that a phase model built from `dn_eff_dt_per_k`/`d2n_eff_dt2_per_k2`
has "the same slope and curvature by construction" (true only to first order at the line wavelength; about 4 %
of s0 beta and 0.6 % of s0 are missing), and the `fsr_nm` doc (it is the FSR at the assigned line).

Still open (for the integrator and the content agent):
- R2 default (lock_slope vs fsr_integral), item 1 of section 4: unchanged; I agree with the builder that
  `fsr_integral` is the more physical default (Z is a stack property; under lock_slope it follows each ring's sampled
  offset), but 07 is normative, so the integrator decides.
- `pyproject.toml` package-data lists `params/presets/*.yaml` but not `params/sources.yaml`; add `params/*.yaml` when
  the content agent writes it (wheel installs only; the editable install works).
- 07 4.8 `self_heating.eta` (0.5 to 0.95) has no direct target when the loss budget is given (eta is then derived);
  vary the budget components (for example `alpha_other_abs_db_per_cm` with `mode: scale`) or give `heat_fraction`
  with `q_intrinsic` instead of the budget. The content agent should state which.
- An absolute uncertainty entry on a ring-level leaf written as a per-ring list replaces the stagger with one value
  per world (by design: one draw per world); use `mode: scale` or `offset` to keep a designed stagger.
