# Changelog

Contract changes (ARCHITECTURE v2 and the normative sections of docs/research/07_design_critique.md) and the
decisions that deviate from them. Newest first. Each entry names the reason and where the details are.

## 2026-10-16: calibration and validation-report stage integration

Two tasks, built and reviewed separately, integrated here: calibration (07 P0-5, P1-14, 4.3; details in
docs/handoff/calibration.md, integrator record in its section 11) and the validation report (07 P1-23, 4.9, with the
4.3 identifiability screen and the 4.8 propagation; the report task's handoff could not be written to
docs/handoff/report.md, its record is in docs/handoff/integrator.md section 9.4). Integrator record:
docs/handoff/integrator.md section 9. No engine or njit code changed: tools/perf_scenarios.py against
runs/perf_r2/final3: 0 of 476 signals differ; the nine quick acceptance runs on g26 are bit for bit those of
2026-10-10.

New contracts:

- **Calibration** (`mrm/calibration/`: core, groups, pipeline, preset, report, cli; `python -m mrm.calibration` =
  `mrm calibrate`, now registered in `mrm.cli.PROVIDERS`): least squares + profile likelihood per fit group (heater,
  coupling_er, offset, junction, eo, monitor, modulation_depth, dac, laser), hypotheses compared by Birge-scaled
  likelihood ratios, the 07 4.3 half-prior rule for the `calibrated` tag; reads only targets/calibration.yaml. Fit
  record `mrm/calibration/latest_fit.json` (schema `mrm.calibration.fit/1`). pytest marker `calibration` in
  pyproject.toml (`pytest -m calibration tests/calibration`; not CI-blocking, 07 4.1).
- **Generated presets** (do not edit): `g26_calibrated` (g26 + the fitted leaves; `calibrated` only where the rule
  holds: P_FSR, R0, heater_self_ratio (for the literature TCR), resonance_offset, laser.line_offset and the two
  loss-mechanism coefficients; moved but not calibrated: alpha_other, the coupler dispersion (2.2 %/nm, assumed),
  eff_at_0v (93.6 pm/V at the assumed +4 dBm), R_s (390 ohm, the range top), monitor gain_scale, the line-error
  sigma; the restated uncertainty entries er_mechanism, coupling_regime and thermo_optic (the H86 world selects its
  own heater leaves `cal_h86`)), and **new** `g25_test_vehicle_calibrated` (the G25 bench with the same device
  leaves, written by `preset.write_bench_preset`; the report's G25 companion of g26_calibrated and the before/after
  table's G25 plant). **The default preset stays g26**: the integrator recommends switching to g26_calibrated in a
  separate change, with conditions (calibration.md section 11.4).
- **Validation report** (`mrm/validation/report.py`, `sensitivity.py`, `cli.py`; `mrm report`): the generated
  docs/VALIDATION.md (g26) and the new **docs/VALIDATION_g26_calibrated.md**, from cached parts in
  `docs/validation/<preset>/report_data.json` (+ SVG charts). This replaces 07 4.1's "tests/validation writes
  VALIDATION.md". CI: `tests/verification/test_report.py::test_validation_md_is_current` checks docs/VALIDATION.md and
  every docs/VALIDATION_<preset>.md (a fast re-render with the file's stamp must equal the file; no cached part may be
  stale against its presets, target files or code). A preset or target change therefore needs a recompute
  (predictions ~7-9 min, identifiability ~12-20, sensitivity ~15-24, ranking ~14-28; full ~48-76 min); a new fit
  record or a new commit needs only `--fast`. How to calibrate and regenerate: docs/ENGINE.md section 14.
- Report changes at integration: section 3.1 shows a fit record without a `preset` key only in the reports of g26 and
  g26_calibrated (`report.FIT_RECORD_PRESETS`; before, in every preset's report, test presets included); the
  profile chart shows the written (nominal) hypothesis; a calibrated preset's section 1 says its calibration rows
  are fit residuals and flags where the fit's protocol differs from the replica's (`report.CALIBRATED_FLAGS`).
- Fit record repaired at integration: its `context` (read by `bench_config` and by `--groups` reruns) had kept three
  pre-review values from the merge of two parallel runs (eff_at_0v 93.86 instead of the fitted 93.64 pm/V);
  rebuilt from the leaf decisions (`pipeline.context_from_decisions`, tested) and the before/after table rerun. No fit
  and no preset value changed.

Results (calibration.md section 11): on the held-out targets g26_calibrated scores 5 pass / 9 fail (g26: 2 / 12):
VAL-2a, VAL-2b and VAL-3a turn from fail to pass, none from pass to fail; calibration targets 13 / 7 (9 / 11). The
nine quick acceptance experiments keep their verdicts except A1-ber (passes on g26_calibrated); set-point lock errors
are 7-49 % larger; under A1 the self-heating corners (eta high, z_ratio 2) lose a ring on g26_calibrated only.

## 2026-10-10: experiments, hardware and RL stage integration

Three tasks, built and reviewed separately, integrated here: experiments (07 P0-10, P1-17, P2-12, 4.8; details in
docs/handoff/experiments.md), hardware (07 P1-21, 4.9; docs/handoff/hardware.md) and rl (07 P1-22, P1-19;
docs/handoff/rl.md). Integrator record: docs/handoff/integrator.md section 8. No kernel or leaf njit code changed:
every regression and adversarial scenario of tools/perf_scenarios*.py is bit-identical to runs/perf_r2/final3 and
final3_adv (0 of 476 + 1132 signals differ), and so is the default warm start.

New contracts:

- **Experiments** (`mrm/sim/experiment.py`, `manifest.py`, `cli.py`): experiment files (`meta.kind: experiment`,
  `extends:` a preset; `experiments/acceptance/*.yaml`, `experiments/examples/*.yaml`), the kinds run | sweep | mc |
  compare | robust | acceptance | quasi_static, `mrm.run(config, ...)`, the CLI `mrm run | sweep | mc | compare |
  robust | acceptance | rerun` with `--dry-run`, run directories `runs/<UTC>_<hash8>/` (manifest schema
  `mrm.run/1`, `experiment.resolved.yaml`, `metrics.json`, `trace.h5`, `inputs/`, `report.txt`, `convergence.json`)
  and the bit-for-bit `mrm rerun`. Time compression needs the 07 P0-10 convergence certificate (run first, or
  `experiment.convergence_certificate`). h5py is now a core dependency (07 P1-17; pyproject.toml).
- **Schema**: the experiment-layer keys are mrm.params keys: `experiment.{start, n_threads, score_from_s,
  convergence_certificate, compression_factors, histogram, oracle, p_in_w, max_worlds_per_plant, rank_metrics}`
  and the open subtrees `experiment.{record, corners, self_reference, acceptance}` (validated again by
  mrm.sim.experiment). `mrm.params.load(experiment_file)` and `mrm.run(mrm.params.load(...))` now accept every
  experiment file (before, mrm.sim.experiment stripped these keys). The experiment hash counts them once, as options
  (`mrm.sim.experiment.config_hash_without_options`), so it did not change: recorded run directories keep their
  hash8 and still rerun (checked: every acceptance file). A dotted override refines a map option
  (`--set experiment.self_reference.window_s=...`). mrm.params `--set` maps a YAML 1.1 boolean to an enum word
  (`--set experiment.variation=off` is 'off', not False). PARAMETERS_FORMAT.md key appendix regenerated.
- **Runner**: `run(..., score_from_s=...)` (metrics restart at that time; `MetricsState.restart` keeps the oracle
  lock reference and the previous controller latch); `manifest_inputs["warm_start"]` gains `no_code`,
  `supply_limited`, `mode` and `order_hopped`; `warm_start_codes(..., return_no_code=True)` (the gym env no longer
  imports the private form). Fix: after a scenario `fault.controller_reset` the Python path clips the new held
  command of a CompiledController (it kept clipping the array the reset replaced, so a controller whose commands
  left the range built on an unclipped value and diverged from the kernel path;
  tests/verification/test_runner_fault_reset.py; found by the hardware review).
- **quasi_static checks the array heater supply limit** (07 P1-12, `mrm.plant.electrical.supply_limit_row`): new
  result fields `supply_scale` (A, V, B) and `supply_limited`; at a supply-limited point no ring counts as
  reachable (code_ok False, codes kept), so `code_range_coverage` can drop; `warm_start_codes` warns about such
  worlds. Reason: the experiments handoff's "quasi_static vs settle disagreement" in the g26 stacked_eic world (the
  engine settled 800-1100 pm blue of the lines) is this limit: at 25 degC the stacked_eic lock currents (~155 mA
  at the lock codes) exceed the assumed 150 mA supply (`array.heater_supply_current_max_a`, assumed), the plant
  droops every heater by 3-5 % in voltage (6-9 % in power) and the rings fall to the cold self-heating branch.
  The engine was right; quasi_static did not check the limit. The nominal g26 world never reaches the limit
  (0-105 degC, 1-3.75 V), so A7 and every nominal result are unchanged. The limit binds in ~1/3 of the g26 LHS
  worlds at 25 degC (content owner: the 0.2 W / 0.15 A limits are assumed; integrator.md section 8.3).
- **Reachable warm-start order** (experiments request 10.4b): `quasi_static(..., order="reachable")` and
  `run(..., heater_code="quasi_static_reachable")` lock a ring whose heater-up order needs a negative heater power
  one order bluer instead of starting it at code_min (`runner.WARM_START_MODES`). The default `'quasi_static'` is
  unchanged (bit for bit; worlds without such a ring are identical in both modes). **The experiments' `start: warm`
  now uses the reachable mode**: identical in the nominal world (every acceptance file), different in sampled
  worlds with such rings (g26 seeds 1000, B = 8: 10 -> 3 rings without a code); LHS results recorded before this
  change do not rerun bit for bit.
- **Hardware** (`mrm/sim/bridge.py`, `mrm/sim/vectors.py`, `mrm/control/cli.py`, additions to
  `mrm/control/fxp.py`): binary lockstep bridge protocol 1 (frames generated from the ChannelSpecs, profiles
  minimal / full, layout and ControllerSpec fingerprints in the handshake; C reference controller and header
  generator), golden vectors format `mrm.vectors/1` (`.npz` + `$readmemh` `.hex` + `.layout.vh`), golden traces
  format `mrm.golden/1` (`mrm golden [--update --reason]`), the fixed-point reference PI `FxpPi`
  (`fx_pi_update`, `fxp_pi_init`, `fxp_pi_step`, `pi_float_bound_codes`; registry name `fxp_pi`).
  `mrm.sim.record_vectors` is a lazy export of `mrm.sim.vectors.record_vectors`. **Deviation (07 P1-21)**: the
  Verilator / cocotb closed-loop HDL co-simulation is not built (neither tool is installed); tests/hdl replays the
  golden vectors through a synthesizable Verilog PI under Icarus and checks it with yosys. New optional extra `hdl`
  (cocotb) in pyproject.toml.
- **Golden traces** (07 4.9): static lock, the fast aggressor and the compressed soak. The compressed soak
  (`g26__soak_compressed`, added by the integrator through the hardware task's hook) is the c = 100 case of the A4
  convergence experiment (one soak cycle at compression 100, auto substeps, pi_lock_c only: no convergence
  experiment covers another controller); its metrics equal the certified run's bit for bit (47 of 47 shared
  arrays). `GoldenCase.scenario` may be `'config'` (the scenario section of the case's config, with its signal
  defaults). Its test is marked slow (95 s idle, ~3.5 min under load).
- golden: tests/golden/g26__static.npz updated (data sha256 bb95d4ab4626): first golden traces (static lock and the A1 fast aggressor on g26; hardware task, 07 4.9)
- golden: tests/golden/g26__a1_slide10_aggressor.npz updated (data sha256 2b898a0568ce): first golden traces (static lock and the A1 fast aggressor on g26; hardware task, 07 4.9)
- golden vectors: tests/golden/vectors/fxp_pi__g26__aggressor_square.npz added (sha256 da23b4c547c2): FxpPi on g26 nominal, direct square aggressor, 400 edges (hardware task, 07 P1-21)
- golden: tests/golden/g26__soak_compressed.npz updated (data sha256 b278f2666f89): first compressed-soak golden: one soak cycle at compression 100, pi_lock_c, the c = 100 case of the A4 convergence experiment (integrator, 07 4.9, P0-10)
- **RL** (`mrm/sim/gym_env.py`, `mrm/control/fxp_mlp.py`, `examples/tutorial.ipynb`): `MRMEnv` (a gymnasium
  VectorEnv over the plant batch, registered as `mrm/MRM-v0`), `MRMSingleEnv`, `RewardSpec`; the fixed-point MLP
  inference helper of 07 P1-22 (`quantize_mlp`, `FxpMlp`, `fxp_mlp_eval`, `error_bound`) and its deployment as a
  compiled controller (`MRMEnv.compiled_policy`). **Not met**: 07 P1-22's >= 1e5 env steps/s at B = 256 (measured
  2.5-2.6e4: the plant itself caps it at ~4e4, and the Python-path per-call setup costs ~35 %; rl.md section 4).
  The env's ControllerSpec is `MRMEnv.ctrl_spec` (`spec` is gymnasium's EnvSpec).

## 2026-10-10: engine performance round 2

Details, measurements and the change log R2-1 ... R2-16: docs/handoff/perf.md section 8 (owner: engine). Every
change is bit-identical: every regression and adversarial scenario, the kernel/reference parity and every number of
the controller scorecard are the same as at 6b598f7. B = 1 fixed tick 15.5 -> 9.1 us on AC power: **07 P0-9's
fixed-tick budget (<= 10 us per tick) is met**; B = 64 3.6 -> 2.1 us per world-tick (64 x 1 s Monte Carlo in
2.1-2.5 min).

- **The tick loop runs without the numba runtime**: `mrm.sim.kernel._world_block` keeps the per-block setup and
  calls the new `_world_loop` (`@njit(cache=True, _nrt=False)`); the hot leaves of kernel, optics, phase, electrical,
  sensing, datapath, laser, thermal and metrics are compiled with `_nrt=False` too. Rules for code called from the
  loop (comment above `_world_block`; test_kernel_releases_every_array): no allocation, no call of a function that
  returns an array, and a function with the runtime must not make views of its arguments.
- **Kernel helpers on arrays**: the controller edge, auto policy, oracle and fast kernel run as `_release_a`,
  `_sync_act_a`, `_push_act_a`, `_advance_edge_a`, `_auto_h_a`, `_oracle_a`, `_fast_tick_a`. `release_row`,
  `_auto_h`, `_oracle`, `_fast_tick` and `truth_row` stay as wrappers; `_edge`, `_push_act`, `_advance_edge`,
  `_sync_act_from_registers` and `_edge_tick` (no longer called) are removed. `_heater_split_tables` is replaced by
  `_heater_split_fill`; `_implicit_step_eng` takes per-lag coefficient tables (new `_th_tables`, `_th_coef`,
  `_uu_fill`, `_th_dead_init`, `_th_live`, `_th_wake`); `eng_commit(t, k, args_f, args_i, full)` takes a `full`
  flag.
- **Self-heating adapter** (`mrm.plant.optics`): `sh_fast_outputs` / `sh_pass_on` take `full=True` (False: only the
  outputs a tick reads; the engine passes False at every substep but the last of a block); `bus_finalize_nb` =
  `_bus_pout_nb` + `_bus_refl_nb`; the far-line hold's evaluation moved to `_sh_fast_pabs_hold`. Cache layouts:
  N_FK 6 -> 8 (new FK_K11, FK_A2), N_FC 16 -> 13 (FC_K11, FC_A2, FC_SE removed), so `pack_selfheat_args` rows are
  shorter. New `selfheat_pabs_general`, `selfheat_commit_general`, `selfheat_bounds_general` (the general path, an
  independent implementation).
- **Reference**: `ReferenceEngine(plant, *, oracle=True, airy=True)`; `airy=False` runs the general optics path, so
  the parity test also checks the engine against optics code it does not share
  (test_parity_against_independent_optics). A degenerate ring row (r = t1 t2 a = 1 at resonance) raises
  ZeroDivisionError on the Airy path and in `_airy_line` (it gave inf/NaN under numba's numpy error model).
- `mrm.plant.datapath.errors_tick_ber_row` tests the counting period once per tick (same result).
- `optics_far_hold=True` is now ~10 % slower than the exact engine (it did not get the round-2 changes) and still
  changes closed-loop trajectories: keep it off.
- Independent review (docs/handoff/perf.md section 9): results bit-identical to a fresh 6b598f7 tree, timing
  confirmed (B = 1 fixed 15.4 -> 9.1 us), no engine defect; new test_full_optics_outputs_after_every_block guards
  R2-3 (every optics output complete after every block).

## 2026-10-10: g26 aggressor site moved; the old site is a stress scenario

Project owner's decision (Kenny), implemented by the integrator. Details, the old-vs-new comparison and the open
items: docs/handoff/integrator.md section 7.

- **g26 `layout.sites.aggressor` moved from (0, 30) um to (0, -30) um** (provenance still `assumed`; the note says
  why). Ring i sits at (50 i, 0) um, so every ring-to-site distance is unchanged: the A1/A2 power calibrations (in
  ring units) and every ring and heater-resistor trajectory are the same bit for bit. Ring 0's monitor PD
  (`ring.thermal.pd_offset_um` (0, 25) um) is now 55 um from the site instead of 5 um: at 5 um the calibrated A1
  heats it past ~600 degC (624 degC open loop), which is not physical for a real layout; now 93 degC.
- **New site `aggressor_pd_adjacent`** (0, 30) um, kind aggressor, and **registered stress scenarios**
  `a1_slide10_aggressor_pd_adjacent` (alias `a1_pd_adjacent`) and `a2_808_aggressor_pd_adjacent` (`a2_pd_adjacent`)
  in `mrm.scenarios.library` / `registry` (constants `AGGRESSOR_SITE`, `AGGRESSOR_PD_ADJACENT_SITE`): A1 / A2 with
  the same arguments at that site. A stress test of monitor dark-current runaway, not a realistic layout; they
  reproduce the former default layout bit for bit.
- Every g26-derived config now has two aggressor sources: 25 more thermal lags (945 -> 970 in the interim model,
  13 630 -> 14 180 with the FEM ROM). The second source has zero power unless a stress scenario drives it, so every
  other result is bit-identical (checked: all regression scenarios of tools/perf_scenarios*.py except the two that
  run an aggressor); only the length of the lag-state arrays (Truth `x`) changes.
- **Schema**: `layout.sites` records take an optional free-text `note` (not read by the model). PARAMETERS.md and
  the PARAMETERS_FORMAT.md key appendix regenerated.
- **PD-far overrides removed** (`ring.thermal.pd_offset_um=[0,-25]` in tools/perf_scenarios.py,
  tests/validation/test_controllers_on_engine.py, tests/verification/test_ctrl_b_ramp_aggressor.py): with the new
  site they would put the PD 5 um from it. The dark-current runaway tests run the stress scenario
  (test_ctrl_a_scenarios.py::test_a1_full_on_the_25c_base_monitor_dark_current,
  test_ctrl_c_kalman_lqg.py::test_a1_toward_the_cusp_dark_current_runaway,
  test_ctrl_b_ramp_aggressor.py::test_a2_pd_adjacent_stress_dark_current, renamed from
  test_a2_default_layout_pd_dark_current). Scorecard keys: `a1` (default site) and `a1_pd_adjacent` (stress; the
  former keys were `a1` = old default and `a1_pd_far`); `tools/scorecard_compare.py --rename` compares across the
  change. The test harness helpers `a1_rate_scenario` / `a2_rate_scenario` take `site=`. New
  tests/verification/test_aggressor_layout.py.
- `mrm.validation.replicas` picks the aggressor site named `aggressor` (was: the first aggressor source).
  latest_predictions.json regenerated: every predicted value unchanged (only the config hashes).
- runs/perf_baseline (a1_pi) and the review's adversarial runs (agg_edge_pi) were recorded with the old layout:
  record a new performance baseline before comparing those scenarios (docs/handoff/perf.md section 7).

## 2026-10-09: engine performance round 1

Details, measurements and error bounds: docs/handoff/perf.md (owner: engine).

- **The tick runs in one function** (`mrm.sim.kernel._world_block`): `_prepare_interval`, `_thermal_and_sensors`,
  `_account`, `_auto_substep`, `_heater_step` and `_bias_step` are gone (their code moved into the world block in
  the same order); `truth_row(P, S, b)` stays as a wrapper of `_truth_a` (array arguments). Reason: a numba call
  that takes the KParams/KState tuples costs a reference-count update of every one of their ~270 arrays. No result
  changed (bit-identical, kernel/reference parity and the regression scenarios).
- **Self-heating adapter**: the exact ring without backscatter (g26) evaluates from substep caches (cos/sin of the
  reference phase, rotated; the Airy constants) in `mrm.plant.optics` (`sh_fast_precompute`, `sh_fast_pabs`,
  `sh_fast_outputs`, `sh_bounds`), used by both the engine and `optics.selfheat_pabs` / `selfheat_commit` /
  `selfheat_bounds`. Equal to `_airy_line` up to floating-point rounding (P_abs 2e-13 relative); the optics adapter
  now also fills the slope outputs (RL_DABS_DPHI, R_DABS_DLAM) for these rings. `pack_selfheat_args` layout: new
  blocks SH_O_FK, SH_O_FC, SH_O_FF, SH_O_LE, SH_O_FH and the flag SH_FARHOLD (SH_HDR 18 -> 24); new keyword
  `far_hold`.
- **Phase model** (`mrm.plant.phase`): `ring_optics_row` = `ring_optics_levels_row` + `coupling_row`; the engine
  recomputes K^2 at the lines only when the line wavelengths change (exact).
- `mrm.plant.electrical.heater_target_one` / `bias_target_one`, `mrm.plant.datapath.errors_tick_ber_row`
  (per-channel / given-BER forms of the row kernels; the engine caches the DAC transfer per code and computes the
  BER once per tick).
- `mrm.plant.selfheat.implicit_step_b` (and its kernel mirror) updates the lags in two passes (bit-identical).
- **New fidelity switch** `Plant(optics_far_hold=...)` / `run(..., optics_far_hold=...)`, default False: the
  far-line hold, a bounded approximation (perf.md P5); off by default because the closed loops turn its small P_abs
  error into other trajectories (perf.md 5.4). No schema key yet (params owner: `fidelity.optics_far_hold` if
  wanted). `run(..., sh_tol_k=...)` passes the self-heating tolerance through.
- Second session (perf.md P8-P13, all bit-identical to the first session's engine): the kernel's thermal pass from
  per-block tables (`kernel._th_struct`, `_th_gains`, `_heater_split_all`; `_implicit_step_eng` takes the tables as
  extra arguments); bus finalize without the full view set and without a division for a zero reflection; the junction
  table bracket found once per voltage from the uniform spacing (`mrm.plant.electrical.grid_index`, used by
  `_interp_lin`); the phase model without heap allocations (`mrm.plant.phase.ring_optics_levels_ws`,
  `resonance_shift_ws`, scratch passed in); `kernel.cold_coordinate` stops at an exact fixed point; Truth computes
  er_db / il_db only when a record reads them and at the last step of a block (`_truth_a(..., with_db, dark_known)`);
  the PD dark current once per tick (`mrm.plant.sensing.monitor_tick_row(..., i_dark_in=None)`, new optional
  argument; new state row `RF_I_DARK`).
- Engine accuracy against commit 8646691: every regression scenario and the controller scorecard agree to
  floating-point rounding (largest change 2.2e-6 pm of detuning in the scenarios, 2.6e-6 pm in the scorecard; no ADC
  code, heater code, lock state or line changed; perf.md section 5). B = 1 fixed tick 54-55 -> 15.5 us on AC power;
  07 P0-9's 10 us is not met (perf.md 5.6); its 64 x 1 s Monte Carlo in 5 min is (3.5-3.9 min).
- Review of round 1 (perf.md section 6): `mrm.plant.optics._fast_flag` derives a ring's Airy-path flag from its
  backscatter row at every adapter call (it was kept from the first call, so a caller that rewrote rb through
  `selfheat_views` afterwards ran the ring without its backscatter; the plant never does, results unchanged);
  the incremental cos/sin cache's error bound corrected for constant phase steps (systematic, <= 1.8e-12; ~5e-9 K).

## 2026-10-09: engine and controllers stage integration

Integrator changes (details in docs/handoff/integrator.md section 6 and docs/ENGINE.md):

- **Settled warm start (07 P0-1 item 6).** `Plant.settle()`, `Plant.reset(..., settle=True)` and
  `mrm.sim.runner.run(..., settle=True)` start from the thermal steady state of the warm-start codes and the inputs
  at t = 0 (the kernel run with the DC coefficients of every lag until the ring temperatures move < 2 mK), instead of
  cold lags. It needs reset protocol branch_low or branch_high (laser_ramp starts dark: ValueError).
  `run(..., heater_code='quasi_static')` (`runner.warm_start_codes`) takes the oracle lock codes of `quasi_static`.
  Requested by controllers_a (handoff 2.3), controllers_c (4.3b) and the engine (6a).
- **`run(..., factory_table=table)`** calls `controller.load_factory_cal(table)` before `controller.reset(spec)`
  (controllers_b handoff 2; engine handoff 6b). The runner does not run the factory procedure itself.
- **Baseline registry**: `mrm/control/baselines/__init__.py` imports every baseline module, so all 17 baselines
  resolve by name (controllers_a/b/c handoffs).
- **Schema keys** `metrics.lock_reference` (line | oracle | target, default line) and `metrics.lock_target_pm`
  (engine handoff 3 item 21); `MetricsSpec.from_config` reads both. PARAMETERS_FORMAT.md appendix regenerated.
- **g26 loss budget re-solved for the landed modes_rib160 artefact** (n_g 3.9405 instead of the 4.0 fallback,
  alpha_rad 0.119 instead of 1.0 dB/cm). With the old numbers the G25 notch fell to 26.5 dB (CAL-2: 27.9) and every
  eta hypothesis missed its label by 0.8 % (closure tests). Now alpha_i = 110.08 dB/cm keeps Q_i = 7503 (notch 27.9
  dB under-coupled), and eta = 0.848 at alpha_fca 80: `alpha_other_abs` 14.74 -> 13.35 (over-coupled alternative
  5.97 -> 4.71), `alpha_scatter` 16.0 -> 16.61, the eta_* alternatives of alpha_fca / alpha_scatter re-solved
  (fca + scatter = 96.61 in every eta world), the 200 nm `strip400x200_q_anchor` 10.37 -> 11.09 dB/cm. Recompute
  when modes_rib160 changes (the notes in g26.yaml give the recipe). PARAMETERS.md regenerated.
- tests/validation/test_replicas_static.py::test_other_lines_match_the_control_models_static_point: the per-ring
  bound "other lines add > 3 % heat" became "> 2 % per ring and > 8 % on the first rings" (measured 2.9 % on ring 7,
  last on the bus, whose foreign line was partly dropped by the ring that owns it; 9 % on rings 0-2).
- pytest marker `perf` registered (engine handoff).
- **docs/ENGINE.md** replaces docs/handoff/ENGINE_BRIEF.md as the engine reference (the brief is kept for its
  leaf-signature tables).

## 2026-10-09: foundation-stage integration

Integrator decisions (details in docs/handoff/integrator.md):

- **R2 heater form: `fidelity.heater_anchor` default is now `fsr_integral`** (schema default and g26), not 07 P0-3's
  `lock_slope`. Z_heater->ring = dT_FSR/P_FSR with the integral of s(T) over the P_FSR sweep equal to one FSR is
  the exact inversion of the anchor measurement (07 D16). `lock_slope` puts Z 4-5 % low at beta = 1.4e-3 /K and
  makes a stack property depend on each ring's sampled resonance offset. `lock_slope` stays selectable. To keep the
  G25 droop fit y = 0.00904 /mW (CAL-4), `ring.thermal.heater_self_ratio` moved 1.21 -> 1.158 (in-ring), and its
  alternatives tin 3.38 -> 3.243, textbook 1.69 -> 1.621 (silicide stays at its 1.0 floor; its y is now 23 % high).
- **New plant module `mrm/plant/phase.py`** (ARCHITECTURE section 2): PlantParams + ring/heater temperatures +
  junction levels -> the optics inputs (phase, K^2 at the line, a, eta, t_rt). No leaf module did this.
- **Loss temperature coefficients** (`dln_alpha_dt_ring_per_k`, `dln_alpha_dt_heater_per_k`, 07 P1-5) multiply the
  whole alpha_i (the key is d ln alpha/dT) and the added loss heats (mrm/plant/phase.py). 07 P0-2 writes
  alpha_FCA (1 + c_T dT); the calibration owner must use the same form for the CAL-2 refit.
- **Thermal transport delay** (07 P1-2, 4.8: 0-2 us) is applied as whole extra ticks of heater DAC latency
  (`HeaterDriverParams.from_plant_params`, rounded to the nearest tick), as the thermal owner proposed.
- **Laser drift has one owner: the scenario** (`laser_drift` scenario on `laser.dfreq_hz` / `laser.power_delta_db`,
  07 P1-4 path 4). The drift parameters of `mrm.plant.laser.LaserHw` stay keyword-only (default 0) and get no
  schema keys, so drift cannot be applied twice.
- **g26 aggressor site renamed `agg0` -> `aggressor`**, the default site name of `mrm.scenarios.library` (a1, a2,
  aggressor_direct), so `create('a1')` runs on g26 without arguments.
- Artefact fallback: a missing `from:` artefact FILE falls back to the leaf's `fallback` alternative with a warning
  (`mrm.params.config.ARTEFACT_FALLBACK_ALT`; params_content request). A bad sha256, a missing sidecar or key stays
  an error.
- Schema keys added (plant_sensing requests): `ring.monitor.{baseline_ua, stuck_flagged, flicker_band_min_hz,
  reference_tap.{present, coupling_frac, responsivity_a_per_w, dark_i0_na}}`, `ring.heater.readback.{noise_rms_codes,
  average}`, `datapath.{rx_sensitivity_oma_dbm, ber_ref, link_loss_db, ber_floor, bitstat_bw_ghz,
  bitstat_noise_pa_per_rt_hz}`. PlantParams carries them, the temperature-sensor `adc_bits`/`code_zero_c`, the
  lines' `source_laser`/`pol`, and the `sim.*` settings the kernel needs (TimingParams). An absent
  `ring.monitor.dark_activation_ev` now realizes to 0.37 eV (mrm.materials), not 0.
- The 07 P1-10 `dac_glitch` fault is applied to the heater code leaving the DAC latency FIFO
  (`heater_bank_step_row(..., code_add)`); the register readback does not show it.
- `mrm.core.channels`: the temperature-sensor table field is `code_zero_c` (schema name), not `code_offset_c`.
- `realize()`: per-world values of `ring.thermal.xt_dist_ref_um`, `xt_far_um`, `site_ref_um` are used (they were read
  from the config and a hypothesis that set them was silently ignored); worlds that disagree raise.
- `pyproject.toml` package data: `params/*.yaml` (sources.yaml), `validation/targets/*.yaml` and its README.

Requested by other owners (recorded here as asked):

- 2026-10-09 re-freeze of validation.yaml and bounds.yaml by the review of the target split, before any comparison;
  no validation value or tolerance changed; BND-1 windows corrected, BND-2 margin text corrected, BND-3 split into
  BND-3a/3b; reasons in the file headers and docs/handoff/validation_targets.md.
- 07 P0-9 "parabolic segments" (slide-10 aggressor) replaced by the rectified sine (01 verification log item 5);
  scenario kernels compute in ticks; time-compression reference rate 2127 K/s (docs/handoff/scenarios.md).
- Material coefficients are read from mrm/data/materials.json, the cleaned copy of docs/research/02_materials.json
  with the verification-log corrections; 07 P1-3's thin-film k (56-66 W/(m K), +-25 %) is superseded by
  `mrm.materials.k_si_film_w_per_m_k` (02 log #14; docs/handoff/materials.md).
- FSR window A is 1.47 THz, not 07's 1.32-1.48 THz interval (its interior puts a second order within 2 FWHM of a
  TX line; BND-1 review; docs/handoff/params_content.md section 5).
