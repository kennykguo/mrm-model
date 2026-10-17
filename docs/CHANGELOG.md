# Changelog

Contract changes (ARCHITECTURE v2 and the normative sections of docs/research/07_design_critique.md) and the
decisions that deviate from them. Newest first. Each entry names the reason and where the details are.

## 2026-10-16: experiments review fixes (a lock-independent FSR-hop count, code identity and rerun of convergence certificates, the convergence tolerance on max detuning, corners in the experiment hash, doc corrections)

Eleven findings of the experiments / docs review (experiments_docs-1 to -11), confirmed by an independent verifier.
Tests: tests/verification/test_experiments_review.py (14); all 14 fail on the pre-fix tree
(runs/experiments_docs_fix/new_tests_before.log). Tests updated for the new metric and convergence metric (no
tolerance widened): test_metrics.py (order_changes compared with the full-trace reference), test_kernel_parity.py
(order_changes kernel against kernel_ref), test_scenarios.py (the convergence results carry order_changes),
tests/golden/test_golden.py (the soak check).

- **A lock-independent FSR-hop count** (experiments_docs-1; 07 P0-8, P0-10, BND-2). New metric `order_changes`
  (mrm.sim.metrics, per ring and `world`; also in world_summary): steps at whose end the order nearest the declared
  line differs from the previous step's. The detuning is always measured from that order, so |d| <= FSR/2 by
  construction: A4-hop's former bound (max |d| < 5000 pm) passed an open-loop heater step from cold in which every
  g26 ring hopped one order (max |d| 4884-4989 pm, fsr_hop_events 0, which counts only re-locks). A4-hop
  (experiments/acceptance/a4_soak.yaml) and the compressed-soak golden check now ask order_changes == 0, and
  CONVERGENCE_METRICS compares it (exact). The njit metrics row changed (numba cache cleared); every other metric is
  bit-identical (golden comparison before the update: only the new arrays differ,
  runs/experiments_docs_fix/golden_compare_before_update.log). A hop in A4 also fails A4-ber and A4-line, so the
  overall A4 verdict was never a false PASS. The A4 PASS of 2026-10-10 was judged with the former criterion.
- **A convergence certificate carries the code that made it** (experiments_docs-2; 07 P0-10). New
  `mrm.sim.manifest.code_fingerprint()`: sha256 of the Python sources of mrm/{core, plant, sim, control, scenarios,
  params}, mrm/__init__.py, units.py, materials.py, every file under mrm/data and the python / numpy / scipy / numba /
  llvmlite versions and machine. ConvergenceCertificate.code_fingerprint records it; the gate
  (`require_compression_validated(..., code_fingerprint=)`, always passed by mrm.sim.experiment) refuses a
  certificate of other code, or one that records none. Before, only the config, start and worlds were compared, so a
  certificate made before an engine change kept validating compressed runs of the changed engine. Any edit of those
  files invalidates a certificate (a comment too). The A4 certificate of 2026-10-10
  (runs/acceptance_20261010/20261010T115721Z_d1511028) no longer covers the tree in any case: the g26 plant
  fingerprint changed with the 2026-10-16 fixes (b500c0fd... against 8e979786...), and it records no code
  fingerprint. A4 must run its convergence experiment again (c = 1 took 2.2 h on 2026-10-10, then the soak).
- **A certificate-reuse run reruns from its own directory** (experiments_docs-3; 07 P1-17). rerun takes the reused
  certificate from the run's own convergence.json instead of reloading the recorded path (relative to the working
  directory, and usually git-ignored); the option keeps its recorded value, so the hash is unchanged.
- **Convergence: max detuning per ring, at most 0.5 pm** (experiments_docs-4; 07 P0-10).
  DEFAULT_CONVERGENCE_TOLERANCES['max_abs_detuning_pm'] is (0.2 pm, 5 %, cap 0.5 pm) and mrm.sim.experiment
  compares it ring by ring (CONVERGENCE_PER_RING; assess_convergence takes per-ring vectors). Before: 0.2 pm or 5 % of
  the worst ring's |d|, which for pi_lock_c's ~125 pm side-of-fringe offset allowed +-6.3 pm, more than the 5 pm lock
  budget, and a ring that was not the worst was not compared. Near the line (|d| < 10 pm) the rule is unchanged. The
  cap is a tenth of the lock budget; the mini soak of test_experiment_runs.py needs it: its ring 1 takes one of two
  pi_lock_c set points 0.47 pm apart depending on the start-up ramp (a peak-map quantum). The recorded A4 convergence
  differs by at most 0.046 pm per ring.
- **Corners in the experiment hash and in inputs/** (experiments_docs-5; 07 P1-17). Experiment.hash adds the name
  and alts of every corner a case runs (experiment.corners, the NAMED_CORNERS of compare / robust, a sweep's corner
  axis), only when there are corners, so an experiment without corners keeps its hash. An edit of a corner preset now
  changes hash8 and makes `mrm rerun` refuse ("presets changed") instead of reporting arrays that differ. A corner
  file given by path (not a preset) is copied into inputs/ as `_corner_<i>_<name>` when the experiment loads (manifest
  source.inputs[].corner names it as given), and rerun reads it from there. Experiments with corners (compare,
  robust, A7, sweeps over corner) get a new hash; their recorded runs were already refused by the g26 changes.
- **Doc corrections** (experiments_docs-6 to -11). -6: ENGINE.md section 10 no longer lists `mrm report` as not
  built (already done in the 2026-10-16 engine review, test_engine_docs.py). -7: the `start` option doc names
  heater_code='quasi_static_reachable' and branch_high. -8: mrm/sim/manifest.py's layout lists the `_extends_<i>_<name>`
  and `_corner_<i>_<name>` copies and convergence.json (no absolute paths). -9: mrm/sim/cli.py's usage names
  experiments/acceptance/a1_slide10_aggressor.yaml. -10: experiments.md sections 3 and 9 item 1 describe the
  integrated loader (`mrm.params.load('exp.yaml')` accepts every shipped file; checked on all 21). -11:
  examples/controllers/README.md says its ~2 pm (type 2: 2.8 pm) A1 figure is the 20 K-swing rate test, and that the
  full A1 at the default site loses ring 0 by ~455 pm with its heater at code 0 for ~15 ms (rerun today:
  max_abs_lock_error_pm 454.9 pm on ring 0, rings 1-7 within 14.7 pm; the dither figure is ~120 pm, was ~110).
  ENGINE.md section 11, docs/handoff/experiments.md (sections 3, 6, 7, 9, 10) and scenarios.md section 5 are
  updated; experiments/examples/soak_reuse_certificate.yaml states the code condition.
- golden: tests/golden/g26__static.npz updated (data sha256 db0e87402522): experiments review: new lock-independent metric order_changes (and world/order_changes), the count of changes of the order nearest the declared line; every other metric unchanged (runs/experiments_docs_fix/golden_compare_before_update.log)
- golden: tests/golden/g26__a1_slide10_aggressor.npz updated (data sha256 b4fcdc12a09a): experiments review: new lock-independent metric order_changes (and world/order_changes), the count of changes of the order nearest the declared line; every other metric unchanged (runs/experiments_docs_fix/golden_compare_before_update.log)
- golden: tests/golden/g26__soak_compressed.npz updated (data sha256 cc974ca186b4): experiments review: new lock-independent metric order_changes (and world/order_changes), the count of changes of the order nearest the declared line; every other metric unchanged (runs/experiments_docs_fix/golden_compare_before_update.log)

## 2026-10-16: params review fixes (geometry moves the couplers and the FSR, exponential bend radiation, the rib200 film world, Z at the P_FSR line, secant beta, line provenance, ranges, notes, the calibrated coupling-regime label)

Thirteen findings of the params review (params-1 to -5, -7 to -14), confirmed by an independent verifier. Tests:
tests/verification/test_params_review.py (21) and tests/closure/test_params_notes.py (4). On the pre-fix tree, 17
of the params-1 to -13 tests fail (runs/params_fix/new_tests_before.log). Two tests were added later:
test_calibration_r0_prior_is_the_declared_leaf_range fails with the first version of the groups.py fix
(runs/params_fix/final/r0_prior_before.log), and test_calibrated_geometry_sigma_keeps_the_coupling_jacobian is a
regression check of the CAL-9 writer. Of the 6 params-14 tests, 4 fail on the pre-fix
tree (runs/params_fix/new_tests_before_params14.log) and 2 are regressions that pass before and after (the g26 and
g25_test_vehicle labels were already right). Records: runs/params_fix/ (pre / post regression scenarios and
compare_full.log, golden_update.log and golden_compare.log, vectors/, es_long.log, the before/after tables, the
reports before, final/). The calibration refit started in runs/params_fix/refit/ was stopped and is not used.

- **Not refitted: g26_calibrated, g25_test_vehicle_calibrated and mrm/calibration/latest_fit.json are stale**
  (known issue, docs/KNOWN_ISSUES.md). They were fitted before every fix below and inherit the g26 changes; only
  their coupling_regime label (params-14) was rewritten from the same record. Each carries a STALE header (the
  record a `stale` key) and docs/handoff/calibration.md opens with the note. They must be refit when the real
  device parameters arrive. tests/calibration::test_loss_worlds_hold_the_solver_dispersion is xfail (strict=False):
  its recomputed chi2 is 7.77 against the recorded 7.51, from the secant beta.

- **The geometry moves the couplers** (params-1, params-11; 07 P1-16). `ring.geometry.jacobian_coupling` is wired
  to coupling_rib160's Jacobian (d ln K^2 per nm of width, film, slab and the two gaps). The Jacobian could not be
  wired before: its outputs targeted q_through / q_drop, which are free under the R4 anchors (q_loaded,
  kappa_d_over_kappa), and an output on a free member raised. Now the anchors describe the nominal-geometry device:
  with sampled geometry realize solves the relations once for that device of every world, and each device's
  per-pass couplings are its couplers moved by exp(sum_k J_k dg_k + D d lambda_res) at its own cold resonance (D the
  coupler dispersion, so at the line, where mrm.plant.phase evaluates K^2, the move is exactly J dg: no double count).
  Q_t, Q_d then follow with the device's own omega and T_rt, and Q_L and kappa_d_over_kappa from the rate sum with the
  device's loss. `variation.geometry.width_closes_gaps` (new, true in g26) applies the artefact's own rule (a width
  bias closes both edge-to-edge gaps: total = d/dw - d/dgap). The gaps share one local sigma, 1.5 nm (assumed); the
  stand-in draws of q_loaded (rel 0.015, femwell's strip slope -2 %/nm x 1.5 nm) and kappa_d_over_kappa (rel 0.1,
  -2 %/nm x 5 nm) are gone. g26, 1000 sampled dies (die means): corr(offset, ln K^2 at the line) 0.958 (through) and
  0.983 (drop), sd 0.092 and 0.153 (the verifier's litho-corrected prediction: 0.962 / 0.985, 0.093 / 0.157; before:
  half that amplitude, the model held K^2 at each ring's cold resonance); within a die 3.3 % and 5.0 %; Q_L rel sd
  3.1 % (was 1.5 %), kappa_d/kappa 6.4 % (was 10 %, the strip slope over 5 nm).
- **The FSR follows n_g at the mask radius** (params-2). With sampled geometry a free radius is solved once per world
  from the nominal n_g and the FSR hypothesis and held for every device; the order is the nominal device's (the
  Jacobians are at fixed r_ref and order). Before, R absorbed every n_g draw (FSR exactly 1.8 THz, orders 78-80).
  g26: FSR sd 11.7 GHz (0.65 %), window A 8.7 GHz, the g25_cluster corner 9.0 GHz.
- **The bend radiation is exponential in the geometry** (params-10). The Jacobian's alpha_rad secant is applied as
  d ln alpha_rad/dp from the artefact's stored evaluations (rib160: -0.0485, -0.1785, +0.4104 per nm of width, film,
  slab; new Jacobian target kind `exp`, plus `ln_alpha_rad_per_nm`). 13 % of g26's sampled rings had a negative
  radiation loss (to -0.30 dB/cm); now none (minimum 0.0054 dB/cm in 8000 rings).
- **The 200 nm film world is the rib200 artefact** (params-3). uncertainty.film_thickness '200' selects modes_rib200
  (80 nm slab, n_g 4.0116, s0 66.91 pm/K, its Jacobian) with `rib200_q_anchor` 2.03 dB/cm (the anchor recipe of the
  former strip term, so the measured G25 Q_i holds): eta 0.833 (the strip's fictitious 11.09 dB/cm gave 0.771). The
  strip stays a plain alternative. Its die resonance spread is 4.2 nm with the same fabrication sigmas (6.2 nm on
  rib160): the sigmas stay fabrication errors (params_content.md).
- **Z_heater->ring is the same on every line** (params-4). New key `ring.heater.p_fsr_ref_wavelength_nm` (g26 1309.5,
  inferred: the G25 operating line of CAL-7): R2 takes FSR_lambda at the P_FSR measurement wavelength
  (`derived.heater.p_fsr_span_m`) instead of each ring's line, where Z followed lambda_line^2 (2.3 % across the bank).
  g26: Z 5.704 K/mW and efficiency 0.4046 nm/mW on every ring (were 5.696-5.570 and 0.410-0.400 at the old beta).
- **beta is fitted over the operating range** (params-5). g26's beta is the secant of the modes_rib160 s(T) (its
  quadratic fit, residual 0.003 pm/K) from 25 degC to the nominal lock, 1.2377e-3 /K (rib200: 1.2308e-3); the
  artefact's tangent at 25 degC (1.4723e-3, alternative `tangent_25`) overstated s at the lock by 2.25 %. g26: s at
  the lock 73.89 pm/K (was 75.44; solver 73.85 at the model's lock), T_lock 134.38 degC (was 133.17; the solver
  curve gives 133.98), Z 0.16 % above the solver curve (was 1.2 % below), t_amb_order_hop 113.1 degC (was 111.9).
  Extending the modes table above 125 degC (the P_FSR sweep reaches ~170 degC) is left to the extraction owner.
- **The line wavelengths carry the channel plan's provenance** (params-7; 07 P1-14). realize records how
  `derived.optics.lambda_line_m` (channel_plan center, spacing, line count or the lines table) and
  `lambda_res_cold_m` are made, and fsr_m, Q_i, K^2, dn_eff/dT and d2n_eff/dT2 list them as inputs: the per-ring
  FSR in nm and the efficiency are `inferred`, not `printed`. `channel_plan.ring_to_line` is not an input (it only
  permutes the plan's wavelengths over the rings).
- **Ranges hold every value a world can take** (params-8, params-9, params-13). The loader now checks the corners and
  interval ends of an absolute entry against the target leaf's range and, in strict presets, every alternative
  against its leaf's range (a selected alternative is range-checked, so one outside could never be selected). A
  child preset removes an inherited alternative with `name: null`. Fixed by it: depletion_exponent [0.33, 0.67] ->
  [0.334, 0.666] (schema bounds [1/3, 2/3], the analytic table's span; 0.33 and 0.67 passed load and realize and
  raised in mrm.plant.electrical); chip_input_loss_db [2, 10] -> [1, 10] (holds the OCP +9 dBm corner, 1 dB);
  g25_test_vehicle drops g26's guide_16_lines (8 dBm at the laser module, not at its bench ring); r_cold_ohm
  [42, 58] -> [37, 58] (the G25 hot range over 1 + y P_FSR for every droop the thermo-optic worlds need: 0.00904,
  and 0.0141 / 0.0161 of g26_calibrated's order-2 CAL-4 refits; the calibrated 41.29 ohm was refused as a g26
  override). The calibration's R0 prior now takes its declared width from the leaf (mrm.calibration.groups; it was
  hard-coded [42, 58]). Not changed (the verifier refuted this part): the dispersion leaves keep no range (a prior
  declared after the 0.0222 fit would be chosen after the fact; calibration.md section 3).
- **Notes quote their sources** (params-12). heater_self_ratio 1.158 -> 1.132 keeps y = 0.00904 /mW at the new Z
  (its stated intent; the tin and textbook alternatives move the same way); the Z, heater, bend-radiation and
  variation.geometry notes and PARAMETERS_FORMAT's lock_slope gap (4.0 %, was "5 to 15 %") quote the realized model
  and the loaded artefacts; tests/closure/test_params_notes.py recomputes every quoted number. (a), the self-heating
  pole share, was fixed by the physics_thermal review.
- **The calibrated presets label the coupling regime truthfully** (params-14; 07 P1-5). The CAL-1/CAL-2 fit ties the
  cold 0 V branch to the extinction mechanism (dispersion: under-coupled; ring_t_loss, heater_t_loss: over-coupled,
  their own alpha_other), but g26_calibrated kept g26's independent under | over draw: its 'over' world (the
  dispersion fit on the over branch, chi2 49.2, CAL-2 missed) is under-coupled at 0 V, and the loss worlds were
  labelled 'under' while over-coupled, half of all worlds. New params machinery: a **label entry** (`follows:
  <entry>`, `map: {leader choice: label}`, empty choices; PARAMETERS_FORMAT 6.1): never drawn, its label is the one
  the leader's choice implies; pinning it pins or restricts the leader (the g26 corners: g26_overcoupled now selects
  the over-coupled CAL-2 worlds, g26_low_eta and g26_hot_selfheat the dispersion world); a contradictory pin is a
  load error; mrm.validation.sensitivity treats it as pinned (the leader's axis carries the joint hypothesis). The
  calibration's preset generator (mrm.calibration.preset) writes g26_calibrated's and g25_test_vehicle_calibrated's
  coupling_regime as the label of er_mechanism with the branch of each mechanism's fit from the record. The sampled
  set loses the CAL-2-missing over/dispersion sixth of the worlds.
- **No engine code changed; the plant parameters of g26 did** (beta, Z, heater_self_ratio for every run; the
  coupler, FSR and bend-radiation laws for sampled devices). tools/perf_scenarios.py (runs/params_fix/pre, equal to
  runs/controllers_fix/post, against runs/params_fix/post; compare_full.log): 303 of 476 signals differ. Steady
  states: T_ring +1.2 K at the lock (T_lock), heater codes -290 (ring 7) to +195 (ring 0) (Z is now 5.704 K/mW on
  every ring: +0.1 % at ring 0, +2.4 % at ring 7; T_lock +1.2 K), settled
  detuning within 0.07 pm (steady_pi), 0.12 pm (laser_step), 0.40 pm (idle_traffic). Transients move: steady_pi's
  start-up lock transient by up to 102 pm around 2.5 ms; heater_sweep's fold crossing (up to 10.4 nm instantaneous,
  final within 8.5 pm); a1_pi: ring 0 sits on its low heater rail 0.9 ms longer (14.7 against 13.8 ms) and leaves
  its point by 580 pm (was 472); rings 1-7 peak within 0.5 pm of before, and ring 4's dip at the hot cusp is 9 pm
  deeper (107.6 pm against 116.6, its point 122.4 pm). The A1 target is a resonance trajectory (pm),
  and the model's s at the lock is now 2.1 % lower, so the calibrated aggressor makes a 2.1 % larger ring rise
  (prepare_scenario, ring_t0 'lock'), ~110 pm of the 5.1 nm open-loop swing, which ring 0's heater cannot cancel at
  its rail. mc8_pi: the sampled devices changed (couplers, FSR, film-200 world): 65 of 82 signals, every world.
- golden: tests/golden/g26__static.npz updated (data sha256 4682f10d7792): params review: g26's beta is the secant of the modes_rib160 s(T) from 25 degC to the nominal lock (1.2377e-3 /K, was the 25 degC tangent 1.4723e-3), Z_heater->ring is the same on every line (R2 at the P_FSR line, ring.heater.p_fsr_ref_wavelength_nm 1309.5: 5.704 K/mW) and heater_self_ratio 1.132 keeps the G25 droop y = 0.00904 /mW: T_lock +1.2 K, lock codes -290 to +195, settled detuning within 0.07 pm; A1 ring 0 sits on the low heater rail 0.9 ms longer (peak 468 -> 572 pm)
- golden: tests/golden/g26__a1_slide10_aggressor.npz updated (data sha256 7c830afa52c9): params review: g26's beta is the secant of the modes_rib160 s(T) from 25 degC to the nominal lock (1.2377e-3 /K, was the 25 degC tangent 1.4723e-3), Z_heater->ring is the same on every line (R2 at the P_FSR line, ring.heater.p_fsr_ref_wavelength_nm 1309.5: 5.704 K/mW) and heater_self_ratio 1.132 keeps the G25 droop y = 0.00904 /mW: T_lock +1.2 K, lock codes -290 to +195, settled detuning within 0.07 pm; A1 ring 0 sits on the low heater rail 0.9 ms longer (peak 468 -> 572 pm)
- golden: tests/golden/g26__soak_compressed.npz updated (data sha256 37a9e8b6c23f): params review: g26's beta is the secant of the modes_rib160 s(T) from 25 degC to the nominal lock (1.2377e-3 /K, was the 25 degC tangent 1.4723e-3), Z_heater->ring is the same on every line (R2 at the P_FSR line, ring.heater.p_fsr_ref_wavelength_nm 1309.5: 5.704 K/mW) and heater_self_ratio 1.132 keeps the G25 droop y = 0.00904 /mW: T_lock +1.2 K, lock codes -290 to +195, settled detuning within 0.07 pm; A1 ring 0 sits on the low heater rail 0.9 ms longer (peak 468 -> 572 pm)
- golden vectors: tests/golden/vectors/fxp_pi__g26__aggressor_square.npz updated (sha256 bebb27c774cb): re-recorded with the README command after the params review (secant beta, Z at the P_FSR line, heater_self_ratio 1.132; 13935 of 98400 observation words and 3200 of 9600 action words differ; spec fingerprint e99109b0fca142d3 unchanged)
- Golden metrics (golden_compare.log): static and soak locks within 0.07 pm and 0.08 pm, lock codes as above; A1
  pi_lock_c / fxp_pi ring 0 472 -> 580 / 468 -> 572 pm, rings 1-7 maxima within 0.5 pm; A1 open loop: ring 0
  unchanged (5136 pm), rings 4-7 +4 to +45 pm (s(T) on the way up).
- **Controller scorecard**: regenerated on this plant by the feedforward_pi entry above (its "plant moved the rest"
  list is this change: under A1 the robust locks leave ring 0's point by 454-473 pm, was 347-362, as in a1_pi above;
  es_po no longer falls in the steady run, 4 of 8 locked and 4 still settling; acquire_then_lock 4/8, bitstat_lock
  and wdm_supervisor 8/8). Its reading note on es_po (generated by tests/validation/test_controllers_on_engine.py) and
  docs/ENGINE.md section 5 are updated; the extremum-seeking docstring states the new behaviour: es_po's default
  probes cross the fold only marginally (on the engine, 0.4 s: rings 4-7 fall once before 40 ms, every ring parks at
  the drop peak at 13.0-13.2 dB; runs/params_fix/es_long.log).
- **Tests restated for the new plant** (no tolerance changed): test_ctrl_a_es.py's fold test asserted that es_po's
  default probes fall in the 8 ms harness run; they no longer do (either the tangent beta or the ratio 1.158 alone
  brings the falls back: a 2 % margin), so the fall-and-recapture path is exercised with a +-24-code (~9 pm) probe and
  the default is checked for FAULT. test_report.py's identifiability test takes the chip-loss half-range from the
  leaf (4.5 dB now). test_manifest.py lists the rib200 and coupling_rib160 artefacts.

## 2026-10-16: feedforward_pi gain schedule (no runaway, anti-windup at the applied rails, bumpless exit from LOCK)

The defect the verifier of controllers-7 found (controllers review below, "Not fixed"): `FeedForwardPI.step` set
ff = -K_ff(u + ff) dT, the gain evaluated at the applied code. With the quadratic heater law K_ff(c) c is a constant
k, so the update ff <- -k dT / (u + ff) has no fixed point once u^2 < 4 k dT: a large negative feed-forward raised
its own gain until it ran away, and it then stayed frozen, because ff was updated only while locked. Under A1 (g26
nominal, settled warm start, phase P/2, knowledge dlambda_dt_nominal + heater_law + heater_efficiency) the die sensor
sees ~+20 K of the aggressor (real physics, controllers-7); on the plant of the 19:37 commit (below) ring 0's ff
went to -6.2e3 codes at 54 ms and -7.9e7 at 58 ms, its heater stayed at code 0 and the ring went to FAULT at ~78 ms
with ff at ~-1.1e8, and ring 7 ran away at ~84 ms (runs/ff_runaway_fix/repro_before.log). Tests: three new in
tests/verification/test_ctrl_a_feedforward.py: test_feedforward_gain_cannot_run_away and
test_feedforward_leaves_the_lock_bumpless_and_rereferences (a toy ring with the quadratic heater law, ~2 s each),
test_feedforward_holds_every_ring_under_a1_on_the_engine (slow, 80 ms of A1). All three fail on the pre-fix code
(runs/ff_runaway_fix/new_tests_before.log: ff -3.9e7 with the heater at code 0; ff frozen at -392 codes through the
recapture; ring 0 FAULT) and pass after (new_tests_after.log). The other feed-forward tests (base ramp, staircase,
A1/A2 rates, knowledge gate, per-ring start, the controllers-7 docstring test) pass (ctrl_tests.log). Records:
runs/ff_runaway_fix/.

- **The feed-forward is the heater-law secant from the PI integrator u** (`_ff_codes`): c^2 = u^2 - 2 u K_ff(u) dT,
  ff = c - u for the quadratic law (exactly the heater tuning -dlambda/dT dT; to first order the former -K_ff dT),
  -K_ff dT for a linear law (power or dac_pwm drive). ff no longer depends on itself; u + ff >= 0 and is clipped to
  the DAC range; the PI's u_ref / u gain schedule stays exact (the tuning goes as u^2 - const).
- **Anti-windup at the applied code's rails** (`_u_range`): where u + ff reaches code_lo or code_hi the integrator is
  held there and its type-2 velocity stopped, as PiLock's own clamp of u. Without it ring 0's u wound down to 0
  inside the dead zone u < sqrt(2 u K_ff dT) (~15000 codes at the A1 cusp) and had to climb back before any heat
  flowed (time at code 0 15.9 ms, against 14.9 ms with it). Telemetry integrator_saturated now reports the applied
  code's rails.
- **Leaving LOCK**: a ring that leaves LOCK (a fall: RECAPTURE) takes ff into its integrator and its recapture guard
  (the applied code does not jump; its recapture and fault codes are applied codes); ff is 0 and T_ref is dropped
  while it is not locked, and its next lock takes a new T_ref (ff_codes_per_k keeps its first-lock reference code).
- **Effect, same plant** (the pre-fix controller scored by the scorecard's own code on the same plant,
  runs/ff_runaway_fix/ff_fix_compare_same_plant.log): steady lock unchanged in every discrete field, continuous
  fields within 0.005 pm (max |e|), 0.0002 pm (rms) and 0.0003 pm (d_ref); A1: 8/8 rings locked at the end (was
  0/8), ring 0 454 pm at the hot cusp with its heater at code 0 for 14.9 ms (was 5187 pm and 74.5 ms), rings 1-7
  within 15 pm (was up to 5005 pm, ring 7 at code 0 for 48.6 ms), as pi_lock (454.5 pm, 14.7 pm, 14.7 ms); the
  stress scenario a1_pd_adjacent still loses ring 0 as every controller does (1/8 locked at the end before and
  after), but no ring's heater latches at code 0 any more (rings 2 and 7 were there for 31 and 49 ms).
- **docs/handoff/controller_scorecard.md / .json regenerated** (runs/ff_runaway_fix/scorecard_compare.log, against
  the stored 17:48 scorecard). The stored scorecard predates the params changes of the commit "Final review fixes
  ... (params fix in progress)" (g26.yaml, realize, relations; 18:28-18:56), so every controller moved, and only
  feedforward_pi's move is this fix (above). Both runs (20:02 and the same-plant run, 20:03) are on the plant of
  that commit; the params edits made after it (from 20:05 on, uncommitted) are not in them. The plant moved the rest
  (runs/ff_runaway_fix/plant_change_summary.log): under A1 pi_lock, midranging, kalman_lqg and mpc now leave ring 0's point by 454-473 pm (was 347-362) and hold
  rings 1-7 within 12.5-15.4 pm (was 5.9-8.0); bias_dither holds 8/8 under A1 (was 6/8); in steady state es_po locks
  4/8 (was 0/8, 4 still drifting), acquire_then_lock 4/8 (3), bitstat_lock 8/8 (7) and wdm_supervisor 8/8 (7: it now
  assigns ring 2 to line 0). docs/ENGINE.md section 5 has the new feedforward_pi sentence; its other numbers still
  describe the 17:48 scorecard and are left to the params change.
- docs/handoff/controllers_a.md section 3 states the new law.

## 2026-10-16: controllers review fixes (expose, the D13 knowledge view, round robin, range errors on the compiled path, fast-kernel restart, FxpMlp proof, scorecard stationarity, docstrings)

Eleven findings of the controllers review, confirmed by an independent verifier. Tests:
tests/verification/test_controllers_review.py (17 tests; 14 of the first 15 fail on the pre-fix code, the 15th
is the simultaneous control case, runs/controllers_fix/new_tests_before.log; the scorecard-drift and ES-docstring
tests, added later, need a score field and docstring text the pre-fix code lacks), test_round_robin_parity in
test_kernel_parity.py (Python and compiled controller, through a controller reset), and the ES tests of
test_ctrl_a_es.py (es_po falls; es_sine approaches without a fall; slow: es_sine parks at the drop peak on the
engine). Records: runs/controllers_fix/ (regression scenarios, scorecard before / after and compare, the es_sine /
es_po 0.4 s runs, the no-JIT run).

- **controller_interface.expose is applied** (controllers-1). `ControllerSpec.from_config` never read it, so the
  runner, MRMEnv, the bridge and every other from_config caller exposed every sensor whatever the config said, and
  a misspelled channel was accepted. It now reads the key (None = every sensor; a sample block brings its count;
  unknown names raise ChannelError). Default runs are unchanged (no shipped config sets it).
- **spec.nominal is the knowledge-filtered view of the unsampled design** (controllers-2; 07 D13, P1-1).
  `mrm.control.knowledge.NominalView` (new): `ControllerSpec` wraps any design it is given (also through
  dataclasses.replace); `get` / `si` of a key no granted token releases raise `KnowledgeError`; the experiment
  section (seeds, world choices) is never released, so a controller cannot rebuild the sampled worlds and the spec
  fingerprint no longer depends on `experiment.*` leaves; `model_nominal` releases the whole design but the
  experiment section and the scenario beyond its operating point, and `model_config()` is that design
  (`Config.without`, new) for `mrm.control.models.linearize` (which unwraps a view and raises without the token).
  Both token vocabularies are merged in `knowledge.token_keys()` (`pi_lock.KNOWLEDGE_KEYS` is now
  `knowledge.FINE_TOKENS`); `models.KnowledgeError` is the same class. Every baseline reads the same numbers as
  before (scorecard below unchanged; the Kalman / MPC linearization is identical, tested). Still open: the view is
  of the run's own config, so a corner selected with `alts=` reaches `model_nominal` (no controller_knowledge
  preset). test_control_api.py's fingerprint test stated that any nominal key enters the fingerprint; it now states
  the D13 rule (a released key does, a withheld one does not).
- **Monitor ADC ampere scale is monitor knowledge** (controllers-10): the ChannelSpec scale of `tx.monitor.samples`
  / `tx.monitor.latest` (full scale / (2^bits - 1), A) is set only with `monitor_nominal` or `model_nominal` (as
  `calibration.KNOWLEDGE_TOKENS` documents); without them the specs carry codes only. No controller read it.
- **Engine fix: controller_interface.schedule round_robin is applied** (controllers-3; 07 P2-7). The setting was
  stored and ignored: both paths updated every ring at every edge (8x the per-ring rate of a time-multiplexed
  datapath in g26). Now controller edge j (from the controller phase) updates ring j % N only: the kernel
  (`_active_ring`, `_rr_save_a`, `_rr_apply_a`; KParams `schedule`; `ChannelMaps.act_map` gains a per-ring column)
  packs the ring into the observation header (OBS_ACTIVE_RING) and applies only that ring's element of every (B, N)
  actuator channel, the others keeping the held command of the edge (also through a controller reset); the runner's
  Python path passes `Observation.active` and merges before validating (`api.merge_round_robin`), MRMEnv gets the
  rule from the kernel, the NumPy reference implements it (parity tested), the bridge's serve side and
  `vectors.observation_from_flat` set `active` (`ControllerSpec.active_ring` / `active_mask`, new).
  `CompositeController` refuses the interface round robin (it multiplexes its children with its own `schedule`,
  which its children's specs now carry).
- **Engine fix: on_out_of_range: error on the compiled path** (controllers-4; 07 P1-19). The kernel always clipped
  and counted; it now also records each world's first out-of-range command (`WI_SAT_FIRST_TICK / _ELEM / _VAL`,
  before the clip), and `runner.run` raises ActionRangeError after the block (naming channel, index, value, range and
  tick) when the mode is error. Clip mode is unchanged.
- **Python path restart of a compiled controller after fault.controller_reset** (controllers-6). `ctrl.reset`
  allocated new state arrays while the kernel's `S.c_*` (which the fast kernel updates) kept the old ones, so after
  a reset the fast kernel and the step worked on different memory (heater codes 47 apart in the review's counter
  test; bias_dither's lock-in stopped). `Plant.restart_compiled` (new) copies the new init state into the kernel's
  rows, makes the controller use them again and re-fills the held command from the latched registers, as the
  kernel's own restart does; compiled and Python paths are again bit-identical through the reset.
- **FxpMlp derives its no-overflow proof** (controllers-9). `proven` defaulted to True, so an FxpMlp built from
  layers with a narrower accumulator took the int64 fast path and hid the declared accumulator's overflow (922
  instead of 0 in the review's case). It is now derived from the layers (`fxp_mlp.acc_bits_needed`, recomputed from
  the weights); a claimed True that does not hold raises; False still forces the per-addition path.
  `fxp_mlp_check` returns 14 for a packed proof flag on an accumulator narrower than its worst case.
- **Scorecard: a steady lock needs a stationary operating point** (controllers-5). A ring now counts as locked
  only when its point drifts by at most 0.1 pm/ms (median of the last 10 ms against the 10 ms before: less than the
  1 pm hysteresis over the scoring window); a new column counts the drifting rings. es_sine, reported as 8/8 locked
  at 0.30 dB, drifts at 0.77-0.95 pm/ms toward the drop peak and parks there after 100-160 ms at 12.9-13.0 dB
  (runs/controllers_fix/es_long.log): 0/8 locked, 8 drifting. Also drifting at 40 ms: bitstat_lock ring 5 (locked
  at 23.1 ms) and wdm_supervisor ring 2 (on a foreign line, reached at 26.7 ms). The A1 scoring start is unchanged
  (the steady time to lock). The extremum-seeking docstring said both seekers cycle between approach and fall: es_po
  falls 2-4 times per ring and then parks (at the peak, or on the cold branch for rings 4-6), es_sine never falls.
  The test named for the fold now asserts es_po's falls and es_sine's fall-free approach.
- **Docstrings** (controllers-7, -8, -11): feedforward_pi's die sensor does see a local aggressor (~+20 K under A1,
  open loop; tested), and the docstring now says how that loses every ring; the oracle's g26-nominal optimum is
  +103 to +105 pm and equals the unrestricted optimum (the stability restriction costs no OMA there; tested), and
  `Plant.truth()` does carry the lag states and conditions; the reward's log lock term is -10.6 at 200 s and -13.8
  at 1000 s (tested). docs/handoff/controllers_c.md section 4 items 2-3 updated.
- **Signals**: tools/perf_scenarios.py (runs/controllers_fix/post against runs/engine_fix/post): every signal of
  steady_pi, a1_pi, laser_step, heater_sweep, idle_traffic and mc8_pi bit for bit (default runs: simultaneous
  schedule, clip mode, no reset). Golden traces unchanged. tests/golden/vectors/fxp_pi__g26__aggressor_square.npz
  re-recorded: only its stored spec fingerprint changed (fae458965c7c440f -> e99109b0fca142d3: the spec now hashes
  the knowledge view and the codes-only ADC channels); observations, actions and the data sha256 ef16bffeed9b are
  the same.
- **docs/handoff/controller_scorecard.md / .json regenerated** (runs/controllers_fix/scorecard_compare.log): every
  number of every run unchanged; only the steady `locked` flags of the 10 drifting rings above changed. ENGINE.md
  section 5's summary of it was stale (es_po 4 of 8, midranging 7 of 8, wdm_supervisor all 8) and is rewritten.
- **Not fixed (out of scope, noted):** feedforward_pi's gain is evaluated at u + ff, so a large negative
  feed-forward raises its own gain and runs away to code 0, then is held while the ring is unlocked (the A1 failure
  path; the verifier of controllers-7 measured it); and test_ctrl_b_wdm.py::test_cold_start_locks_every_ring_on_its_line
  still fails on map_ok (the physics_thermal note above; unchanged by these fixes).

## 2026-10-16: engine review fixes (settle at the reset tick, per-world settle stop, iterated order hop, data state of the warm start, kernel blocks on controller edges, reference controller reset)

Seven findings of the engine review, confirmed by an independent verifier. Tests: tests/verification/
test_engine_start_review.py (engine-1, -2, -3, -5), test_engine_blocks.py (engine-4), test_engine_docs.py (engine-8),
two new cases in test_kernel_parity.py (engine-7) and the settled-start replay in test_gym_env.py; each fails on the
pre-fix code (runs/engine_fix/new_tests_before*.log, docs_test_before.log, settle_tests_with_old_settle.log: the
same tests against the former Plant.settle). Records: runs/engine_fix/ (pre / post regression scenarios and
compare.log, golden_check.log, scorecard before / after, the mc_pi_lock rerun in exp/).

- **Engine fix: Plant.settle holds the scenario at the reset tick** (engine-1). Every DC iteration ran one kernel
  tick, and the tick advanced, so iteration i read the scenario of tick k0 + i: the settled state was the steady
  state of the inputs ~20 ticks after the start, not at it. An aggressor switched on at 10 us started ring 0 18.7 K
  hot and +1554 pm red (its lock: +121 pm), reported converged. The settle now runs with every scenario signal held
  (the signal table's component counts set to 0 for the settle, so the kernel keeps the values the reset filled from
  the scenario at k0; no kernel code changed). The noise draws still come from a new tick per iteration, as
  documented. Shipped scenarios at their defaults moved by at most a few pm (the verifier: A1 at its cusp 3 pm, A2
  1 pm, bias_ripple 0.1 pm).
- **Engine fix: each world stops its own settle** (engine-5; rl request 7.5). The batch iterated until its slowest
  world converged, and the extra iterations (noise of new ticks) moved the other worlds' settled states by up to
  ~1 mK, so world b of a batch did not equal the same world alone (07 P1-13; the experiments' start: warm and
  cold_settled). A world now keeps its state of its own first iteration below the 2 mK tolerance; the report gains
  `iterations_per_world` and `converged_per_world` (`iterations` is the largest). Settled gym episodes now replay
  alone bit for bit (test_gym_env.py, warm and cold_settled). B = 1 runs are unchanged.
- **quasi_static(order='reachable') hops until no ring needs a negative heater power** (engine-2). A hopped ring adds
  ~1 FSR of heater power (20-38 mW in g26), whose crosstalk pushed rings that had a heater-up code below zero; the
  single re-solve left them without a code and with fsr_hop_risk False. The hop now repeats (at most N + 1 solves;
  heat only adds, so this is the smallest consistent set of hops); order_hopped marks every hopped ring;
  fsr_hop_risk stays the heater-up order's need. g26 seeds 1000, B = 8: 10 -> 0 rings without a code (ENGINE.md 3
  said "10 -> 3", but none of the 3 was one of the 10); 14 rings hop (world 0: 5, 6, then 1, 4, then 3; world 1:
  1, 6, then 5); worlds 0 and 1 then exceed the array supply limit (droop 0.916, 0.968; world 2 0.933 as before).
  experiments/examples/mc_pi_lock.yaml (start: warm): 1 ring without a code instead of 9, 101 hopped rings, 2
  supply-limited worlds; median worst-lane OMA penalty 1.63 dB, p95 31.0 dB; 27 worlds with a ring > 5 pm off its own
  operating point, the same 10 without any warm-start flag (runs/engine_fix/exp). Two assertions of
  tests/verification/test_warm_start_orders.py stated the single-pass result ("exactly the negative-power rings
  move", "a hopped ring with a code had none before") and now state the iterated one.
- **The warm-start codes use the data path of the reset tick** (engine-3). `quasi_static` gains `data_state`
  (driver_enable, swing_scale, transition_density, ones_density, as the kernel reads the scenario signals);
  `warm_start_codes` passes the scenario's state at the reset tick (`runner.reset_data_state`). The lock target stays
  the traffic-on operating point (the verifier's caveat: with zero swing the relOMA objective is flat); the heat
  balance is solved again in the start state (junction levels with the scaled swing, absorbed power and through
  transmission averaged with its ones density, its data-activity heat). The default (traffic on) is the former
  survey bit for bit. Before, every warm start assumed traffic on: traffic_start_stop(t_start_s=0.01) began 65-98 pm
  blue of the lock (the ~1-1.4 K of data heat missing) and pi_lock_c's start-up dropped rings 4-7 to the cold branch
  (-1228 to -1294 pm); now the codes are 1243-1348 higher, the settled start sits within 1.0 pm of the static start
  and pi_lock_c holds every ring through the traffic start (ones density 0 at t = 0: within 0.5-4.6 pm, before 13-24
  pm; the residual is quasi_static's missing upstream filtering of the other lines).
- **Kernel blocks end on controller edges** (engine-4). Under the auto policy a block end caps the substep in flight
  and the growth limit restarts from it, so chunk_ticks changed the trajectory (13 against 1e5: 11426 against 10582
  substeps, monitor codes up to 9 apart over 20 ms), contrary to perf.md section 9. A substep never crosses a
  controller edge, so the runner (`_block_end`) now ends every block of a run without tick jitter on an edge, on the
  compiled and the open-loop path (which now takes chunk_ticks too); chunk_ticks changes no result there and is
  recorded in `manifest_inputs["chunk_ticks"]`. With tick jitter the dependence remains (documented). The scoring
  split must end a substep at its own tick: under the auto policy it leaves the trajectory unchanged only on a
  controller edge (A4 and the soak golden split at 5 ms, an edge), so the runner warns otherwise
  (`runner.ScoreSplitWarning`); the restarted BER windows also cap substeps at their own ends. ENGINE.md section 2
  ("plant, controller and records are untouched") and perf.md section 9 corrected; experiments.md likewise.
  Default runs are unchanged (g26: 1e5 is a multiple of the 10-tick period from k0 = 0).
- **The NumPy reference applies fault.controller_reset** (engine-7). `ReferenceEngine` defined the latch and never
  used it: with a reset pulse the reference and the kernel were 12684 heater codes apart from the first edge after
  it. It now latches the rising edge in the interval and resets the controller at the next edge before its step (a
  reset in any world resets every world, as the runner's Python path; the kernel restarts per world, the same for a
  die-common pulse). New parity cases: the reset with a compiled controller, and a scenario driving every input the
  kernel reads from the scenario (aggressor, laser failover with a dark gap, fibre loss, traffic start and stop,
  bias step, the four faults, a controller reset) with a Python and a compiled controller. The parity tests had used
  only the config scenario. Also passes under NUMBA_DISABLE_JIT=1.
- **Docs** (engine-8): ENGINE.md section 10 listed `mrm report` / the VALIDATION.md generator as not built (it is
  built and CI-checked, section 14); mrm/cli.py called three built providers "(planned)". Both corrected.
- **Signals that changed** (tools/perf_scenarios.py, runs/engine_fix/pre against post; pre = the physics_sensing post
  tree): steady_pi, laser_step and heater_sweep bit for bit (B = 1, scenarios without components or events before
  the settle sees them). a1_pi (held scenario at the settle; A1 from its cool minimum): the start by up to 0.8 mK
  and 0.06 pm, max |detuning| within 0.004 pm, monitor codes up to 13, heater codes up to 2. idle_traffic (traffic
  off until 10 ms: the warm-start codes now include the missing data heat): heater codes +1243 to +1348, the detuning
  up to 100 pm closer to the lock while traffic is off (the start was 65-98 pm blue), and correspondingly redder
  once traffic starts (open loop). mc8_pi (B = 8 sampled worlds: the per-world settle stop): 56 of 64 rings start
  differently by < 1 mK; the closed-loop traces differ by up to 32 pm in pi_lock_c's start-up transients, every
  final metric within 0.17 pm (one excursion count, 2 first-lock times by 1 us).
- golden: tests/golden/g26__a1_slide10_aggressor.npz updated (data sha256 f1e6cd352409): Plant.settle holds the scenario at the reset tick (it settled the A1 / soak inputs of ticks 1..~20; engine review engine-1): A1 lock metrics within 0.012 pm, soak within 0.002 pm
- golden: tests/golden/g26__soak_compressed.npz updated (data sha256 d88efc616a0d): Plant.settle holds the scenario at the reset tick (it settled the A1 / soak inputs of ticks 1..~20; engine review engine-1): A1 lock metrics within 0.012 pm, soak within 0.002 pm
- g26__static.npz and the golden vectors are unchanged (static scenario, B = 1; the vector run's square aggressor is
  constant over the settle).
- **docs/handoff/controller_scorecard.md / .json regenerated** (B = 1; runs/engine_fix/scorecard_compare.log): the
  steady runs are unchanged number for number; under A1 the start moves with the held scenario: pi_lock(_c) and
  composite rings 1-7 max 5.952 -> 5.935 pm, ring 0 346.835 pm unchanged, kalman_lqg(_c), mpc(_c), midranging
  unchanged to the printed digits; the acquisition and peak-seeking runs (es_po, es_sine, dither_lockin,
  feedforward_pi, acquire_then_lock, bitstat_lock) end on other trajectories, as after every engine change.
- **docs/VALIDATION.md and VALIDATION_g26_calibrated.md: parts sensitivity and ranking recomputed** (`--recompute
  sensitivity,ranking`; they run start: warm under A1 with up to 256 worlds per plant, so the per-world settle stop
  and the held scenario move them; the report's staleness check does not cover engine code; before:
  runs/engine_fix/VALIDATION_*before.md). Sections 1-3 are unchanged (quasi_static's default path is bit for bit the
  former one). Nominal and corner results are unchanged to the printed digits (pi_lock_c worst-lane OMA penalty
  0.3159 dB on g26, 0.3408 dB on g26_calibrated); tornado swings move in the fourth digit; the Saltelli S1 (poorly
  resolved at 128 base rows, as stated) and ST move within their intervals; LHS outcomes in fragile worlds move:
  g26 pi_lock_c median worst-lane OMA penalty 6.13 -> 5.76 dB, mean Kendall tau of oma_penalty_db_worst -0.111 ->
  -0.103 (g26) and -0.257 -> -0.265 (g26_calibrated). Not rerun: the A4 convergence certificate (its plant
  fingerprint is unchanged; the soak's settled start moved by < 0.002 pm in every golden metric).

## 2026-10-16: physics_sensing review fixes (bit-statistics density, ADC reference site, reference tap, adc_mux)

Seven findings of the physics_sensing review, confirmed by an independent verifier. Tests:
tests/verification/test_engine_sensing.py and tests/closure/test_sensing_notes.py (13 of their 16 tests fail on the
pre-fix logic, runs/physics_sensing_fix/new_tests_before.log). Records: runs/physics_sensing_fix/ (pre / post
regression scenarios and compare.log, golden_check.log, scorecard before / after, the VALIDATION reports before).
Details: docs/handoff/plant_sensing.md section 6.

- **Engine fix: the bit-statistics window takes the scenario's ones density** (physics_sensing-1). The kernel
  (`_release_a`) and `mrm.sim.kernel_ref` passed the realized per-tick density n1/nb (`RF_RHO`) to
  `bitstat_window_row`; under a PRBS that is never exactly 0.5 (prbs7 0.504), so the exact PRBS window was never
  taken and every window was Binomial(64, n1/nb): 40 of 480 lane-edges matched the transmitted prbs7 sequence (38
  for prbs9), with std 4.1 instead of 2.8 (prbs7). It now passes `RF_DENS` (the density pattern_tick_row takes), set
  at reset from the scenario at k0 so an edge at k0 has it too: 480 of 480. Binomial mode draws Binomial(64,
  density). The docs claim "prbs7/9/15/23 exact while the density is 0.5" now holds for the window. Default runs
  (pattern_noise off) give the same window except at an edge at k0. Not changed: the window's levels at the first
  edge after a reset are the reset placeholders.
- **Engine fix: the monitor ADC reference sits at the package base** (physics_sensing-2). Every ring's ADC
  reference temperature was its own PD observer (25 um from a ~133 degC ring, ~20 K above the base at the g26 lock,
  ~68 K at PD0 under A1), undocumented, so the ring's heater and the aggressor moved the monitor gain through the
  20 ppm/K tempco (-21 to -23 codes at the lock, more than the +12 to +14 codes of the Arrhenius dark current; 0.14 %
  at PD0 under A1), and with adc_mux one converter had C reference temperatures. New setting
  `ring.monitor.adc_ref_site`: `package_base` (default, set in g26 as an assumed choice: thermal.t_base_k +
  thermal.xpu_rise_k, the converter on its own die beside the PIC, 07 P1-3 side_by_side; its own dissipation is not
  modelled) or `monitor_pd` (the former model, reproduced code for code; needs channels_per_adc 1). Kernel row
  `RF_T_ADC` and `KParams.mo_adc_site`; `MonitorHw.adc_temperature` is the rule for the Python banks
  (`MonitorBank.tick / level_codes`, `SensingBank.tick`, `DatapathBank.bitstat_from_monitor` take `t_base_k`); at
  25 degC ambient the plant now matches `mrm.control.models` (T_ref) exactly.
- **The reference tap is refused, not ignored** (physics_sensing-3). `ring.monitor.reference_tap.present: true`
  was accepted, realized and silently dropped (no channels, and the optics did not take its coupling share).
  `MonitorHw.from_plant_params` (so the Plant, mrm.control.models and the replicas) now raises NotImplementedError.
  integrator.md section 3 said the plant could produce the channels; corrected. ENGINE.md 10 and ENGINE_BRIEF too.
- **background_fraction is a fraction of the TOTAL bus power** (physics_sensing-4). The schema doc said "fraction
  of the line power"; the kernel (unchanged) applies it to the bus power at the ring summed over every line (3.1-8.0
  x the declared line in g26). Doc fixed (schema, PARAMETERS_FORMAT key appendix and section 10). The g26 range
  [0, 0.01] (69-191 uA on a 40 uA scale) is now [0, 3e-4]: the top keeps every nominal drop peak (31.8-32.2 uA) plus
  the background ~2 uA below full scale, with the whole chip input at every ring (bound 4.1e-4). Range metadata only:
  the value stays 0 and no uncertainty entry samples it.
- **Per-world monitor baseline** (physics_sensing-5). `MonitorHw.from_plant_params` flattened `baseline_a` (B, 1)
  to (B,), which `build` reads as per ring when B == N (batch 8 on the 8-ring bank). It now passes (B, 1). Latent:
  no preset varies the baseline per world.
- **adc_mux** (physics_sensing-6). round_robin with channels_per_adc C > 1 now needs C x integration window <= ADC
  period (the C rings take turns on one integrating converter, 07 P0-7; realize and `MonitorHw.build` check it), and
  the rings of one converter (n // C) share its DNL thresholds and INL bow (`PR_ADC_UNIT` keys the DNL;
  `draw_inl_bow_lsb(channels_per_adc=)`); the per-ring offset and gain error stay per ring. C = 1 gives the former
  draws bit for bit. The de-integration time is not modelled (documented). tests/verification/test_ctrl_c_models.py
  ran g26 with C = 4 and the 2 us window (overlapping); it now uses a 1 us window.
- **g26 junction_model comment** (physics_sensing-7): it said the junction LUT was not extracted;
  junction_rib160.json exists. It now gives the reason analytic is kept (no rib200 or lateral LUT while the film is
  sampled; the CAL-5 / CAL-10 decision).
- PARAMETERS.md and the key appendix of PARAMETERS_FORMAT.md regenerated. tools/perf_profile.py passes `RF_T_ADC`.
- **Signals that changed** (tools/perf_scenarios.py, runs/physics_sensing_fix/pre against post; pre = the
  physics_thermal post bit for bit). Only the ADC reference site changes them (no scenario runs a PRBS, the tap or a
  mux). Open loop (laser_step, heater_sweep, idle_traffic): only monitor_code, +17 to +23 codes. steady_pi: monitor
  codes up to 40 codes, every lock metric within 0.004 pm, final heater codes within 1 code, final detuning within
  0.044 pm. a1_pi: max |detuning| within 0.11 pm of ~470 pm, mean within 0.06 pm, the trace up to 54 pm from the
  timing of the transients (first difference at tick 175). mc8_pi (sampled worlds): other trajectories (metrics up
  to 10 pm, 3 of 64 rings change ever_locked).
- golden: tests/golden/g26__static.npz updated (data sha256 90e439c15a3e): the monitor ADC reference sits at the package base temperature (ring.monitor.adc_ref_site: package_base), not at each ring's PD observer (g26 lock: +17 to +24 codes, the 20 ppm/K tempco over the ~20 K PD rise; ring 0 under A1 0.14 %; physics_sensing review)
- golden: tests/golden/g26__a1_slide10_aggressor.npz updated (data sha256 d8ca9253edc5): the monitor ADC reference sits at the package base temperature (ring.monitor.adc_ref_site: package_base), not at each ring's PD observer (g26 lock: +17 to +24 codes, the 20 ppm/K tempco over the ~20 K PD rise; ring 0 under A1 0.14 %; physics_sensing review)
- golden: tests/golden/g26__soak_compressed.npz updated (data sha256 b42c45930d7c): the monitor ADC reference sits at the package base temperature (ring.monitor.adc_ref_site: package_base), not at each ring's PD observer (g26 lock: +17 to +24 codes, the 20 ppm/K tempco over the ~20 K PD rise; ring 0 under A1 0.14 %; physics_sensing review)
- golden vectors: tests/golden/vectors/fxp_pi__g26__aggressor_square.npz updated (sha256 ef16bffeed9b): re-recorded with the README command after the ADC reference site fix (11472 of 98400 observation words and 297 of 9600 action words differ)
- Golden metric changes: static lock metrics <= 0.031 pm; A1 max |detuning| 0.13 pm (fxp_pi) and 0.11 pm
  (pi_lock_c) of ~470 pm; compressed soak <= 0.062 pm.
- **docs/handoff/controller_scorecard.md / .json regenerated** (tools/scorecard_compare.py against the
  physics_thermal file, runs/physics_sensing_fix/scorecard_compare.log): pi_lock(_c) and composite steady rms within
  0.002 pm and max within 0.008 pm, kalman_lqg(_c) within 0.04 pm, mpc(_c) within 0.045 pm; under A1 ring 0 346.81 ->
  346.83 pm (PI), 362.02 -> 361.99 pm (Kalman), 362.06 -> 362.01 pm (MPC), rings 1-7 max 5.89 -> 5.95 pm (PI). As
  after every engine change, the acquisition and peak-seeking outcomes move with the small change (docs/ENGINE.md
  5): es_po's one steady locked ring (ring 0, 16 pm) is lost (1164 pm), and the A1 runs of acquire_then_lock,
  es_po, es_sine, feedforward_pi and bitstat_lock end on other lines or trajectories.
- **docs/VALIDATION.md and VALIDATION_g26_calibrated.md regenerated** (the g26 config hash changed with the
  background_fraction range and the notes; before: runs/physics_sensing_fix/VALIDATION_*before.md). Sections 1-3
  (replica predictions, closure, identifiability) are unchanged: the replicas use the monitor photocurrent, not the
  codes. Sections 4-5 (engine runs) move slightly: nominal worst-lane OMA penalty 0.3154 -> 0.3159 dB (g26) and
  0.3402 -> 0.3408 dB (g26_calibrated), the tornado and Sobol entries within their intervals, the ranking's mean
  Kendall tau within 0.04.

## 2026-10-16: physics_thermal review fixes (ROM time scales, crosstalk distance, package hypothesis, notes)

Eight findings of the physics_thermal review, confirmed by an independent verifier. Records: runs/physics_thermal_fix/
(pre / post regression scenarios and compare.log; post_radius0 and golden_compare_radius0.log: the fixed tree with
the former distance convention; scorecards before / after; the VALIDATION reports before).

- **Engine fix, ROM mode only: the die-level time scales (07 D15) per pair** (physics_thermal-1). `assemble` gave
  every Foster pole the factor of its own layer (tau < 300 us: time_scale_fast). A non-collocated FEM pair (heater ->
  ring mode, every cross pair) has mixed-sign residues up to 6.6 x its DC on both sides of 300 us; with unequal
  factors they stopped cancelling: metal_above_160 heater -> ring at x2 / x0.5 peaked at 1.21 x DC and fell back by
  0.22 x DC (|H| 1.27 x DC near 600 Hz; 1.16 x DC through realize with the g26 0.5-2 interval), silicide_undercut
  1.04-1.08 x DC. Such a pair (mixed signs, poles in both layers) is now time-scaled as a whole by
  s_fast^w s_slow^(1 - w), w its step fraction at tau_split (`thermal._whole_pair_time_scales`; ThermalSystem.meta
  `time_scale_whole_pairs`); one-signed pairs (driving point, every interim lag) keep per-pole factors; equal factors
  give the former result bit for bit. This refines D15 for a fitted ROM: "the fast and slow poles come from
  different layers" holds for physical poles, not for the cancelling poles of one fit. No g26 run changes (interim).
- **Engine fix: the crosstalk tau law's distance** (physics_thermal-3). The X2 table and the 60 um anchor of the d^2
  law are imec's heat-source-to-victim distances (04 2.5: 60 um for the first neighbour at a 100 um pitch), but
  `_interim_lags` passed centre distances: imec's own geometry gave 4.75 ms instead of the printed 4.14 ms. It now
  passes d_cc - r_src - r_obs (`InterimRom.ring_radius_um`: realize passes each ring's r_ref, R3; default 7.5 um, the
  FEM ROM's ring; 0 for point sites), and rings of R = 20 um at a 100 um pitch reproduce Table II. The site coupling
  `site_fraction` (an assumed law) keeps the centre distance. g26 nominal (r_ref 6.73 um): nearest neighbour
  2.847 -> 1.521 ms, second 4.75 -> 4.55 ms, own monitor PD 0.71 -> 0.38 ms, aggressor -> ring 0 1.03 -> 0.62 ms,
  aggressor -> PD 0 and the die sensor unchanged. realize's lag-bound fallback uses the same distances. 07 P1-2 and
  04 2.5 annotated.
- **The package hypothesis** (physics_thermal-2, -5). The interim crosstalk uses imec's natural-convection numbers
  (3 % nearest, 1.5 % to every other ring and every far site, 1.5-9 ms poles); the in-ring FEM ROM's TEC-held package
  (R_pkg 2 K/W) gives 1.0 % nearest reaching 63 % in 0.12 ms at the pitch (13x faster than the corrected interim pole,
  24x the former), 0.17-0.34 % to the farther rings and 0.09 % at 1 mm, and no sampled world came near it
  (time_scale_slow 0.5-2). The handoff called the two "consistent" with wrong arithmetic (extract_thermal.md,
  corrected). New: a third choice of g26 `uncertainty.thermal.stack`, `side_by_side_tec_held` (weight 0, appended
  last: no sampled or Latin-hypercube draw changes), selecting the `tec_held` alternatives of xt_nearest / xt_next /
  xt_floor / xt_tau_ref / xt_tau_min and of the new key `ring.thermal.site_floor_frac` (the far floor of the site law,
  `InterimRom.site_floor_frac`; equal to xt_floor_frac in every other choice, so the default plant is unchanged: the
  die sensor die0 sees each heater with 1.5 % of Z0, 85 K/W, one 9 ms pole, stated in PARAMETERS.md; 0.09 % under
  side_by_side_tec_held), and the corner `g26_tec_package`, added to `mrm.sim.experiment.NAMED_CORNERS` (07 4.8 lists
  12; robust and compare runs and the report band now include it). Weighting the choice in the sampled set is left to
  the content owner (it would change every sampled-world result; docs/handoff/params_content.md). The open conflict is
  the new VALIDATION.md row closure-crosstalk-dynamics with its strict xfail
  `tests/closure/test_thermal_crosstalk.py::test_g26_default_crosstalk_matches_the_fem_rom`.
- **Notes** (physics_thermal-4, -6): `ring.thermal.heater_node_tau_s` is tagged `assumed` (it is the pole of the
  excess (h - 1) Z0 only, 13.6 % of Z_HH; the FEM's 1.47 us is its whole Z_HH t63; the interim Z_HH t63 is 6.7 us,
  set by h and S_H, value unchanged); the `selfheat_tau_s` note quotes the ROM's step fractions (27 % at 100 ns, 54 %
  at 1 us, t63 1.76 us; its "65 % below 1 us" was an older fit's pole share). PARAMETERS.md and the key appendix of
  PARAMETERS_FORMAT.md regenerated.
- **ARCHITECTURE 5.3** (physics_thermal-8): it said the interim model is used "until the FEM artefacts exist"; all
  six exist. It now states the rule g26 follows, which is 07's build order step 3 (an artefact replaces assumed values
  only through the calibration split; `fidelity.thermal_model` interim by default, rom selectable), not a deviation.
- **`DEFAULT_SH_DT_MAX_K` comment** (physics_thermal-7): it said a multi-tick auto substep scales the 50 K bound;
  nothing ever did. Corrected with the measured envelope (ENGINE_BRIEF 4.2): ticks move a ring by mK (~18 K on a
  full-scale heater step from cold), auto substeps by mK; Plant.settle's first DC iterations (~100 / ~240 K) and the
  first tick after a reset to a bracket end (~77 K) exceed it. `Plant.settle` now reports `max_step_k` and
  `sh_dt_max_k`; the g26 settled states equal those with a covering 400 K bound (tested). Engine unchanged.
- **Signals that changed** (tools/perf_scenarios.py, runs/physics_thermal_fix/pre against post; pre = the
  physics_optics post bit for bit; the fixed tree with the old distances reproduces pre in 0 of 476 signals, so every
  change comes from the distance convention): 284 of 476 signals differ. Steady closed loop (steady_pi): max
  |detuning| within 0.035 pm, rms within 0.08 pm, mean within 0.17 pm, the second half of the trace within 0.10 pm;
  the 91 pm trace difference is the timing of the pull-in transient in the first ms. a1_pi: max |detuning| up to
  1.74 pm of ~470 pm, one ring counts one more excursion, A1 calibrated peak power 0.2837 -> 0.2803 W (same mean); PD
  temperatures up to 0.54 K (the faster own-heater -> PD pole). Open loop: laser_step within 0.02 pm, idle_traffic
  within 5.5 pm (0.4 pm in its last quarter). heater_sweep: timing of the fold crossings (10.4 nm transient, 20 pm
  final); mc8_pi (sampled worlds): other trajectories (408 pm, 2 of 64 rings change ever_locked).
- golden: tests/golden/g26__static.npz updated (data sha256 44e63bd0cbf2): the interim crosstalk tau law takes the heat-source-to-victim distance of its imec anchor, the centre distance less both ring radii (g26 nearest neighbour 2.85 -> 1.52 ms, own monitor PD 0.71 -> 0.38 ms, aggressor -> ring 0 1.03 -> 0.62 ms; physics_thermal review, 04 section 2.5)
- golden: tests/golden/g26__a1_slide10_aggressor.npz updated (data sha256 d6cdf972a8ea): the interim crosstalk tau law takes the heat-source-to-victim distance of its imec anchor, the centre distance less both ring radii (g26 nearest neighbour 2.85 -> 1.52 ms, own monitor PD 0.71 -> 0.38 ms, aggressor -> ring 0 1.03 -> 0.62 ms; physics_thermal review, 04 section 2.5)
- golden: tests/golden/g26__soak_compressed.npz updated (data sha256 e7d734180f05): the interim crosstalk tau law takes the heat-source-to-victim distance of its imec anchor, the centre distance less both ring radii (g26 nearest neighbour 2.85 -> 1.52 ms, own monitor PD 0.71 -> 0.38 ms, aggressor -> ring 0 1.03 -> 0.62 ms; physics_thermal review, 04 section 2.5)
- golden vectors: tests/golden/vectors/fxp_pi__g26__aggressor_square.npz updated (sha256 15578ee610e0): re-recorded with the README command after the crosstalk distance fix (9831 of 98400 observation words and 2073 of 9600 action words differ)
- Golden metric changes (against the engine before this fix): static lock metrics <= 0.0012 pm; A1: fxp_pi max
  468.5 -> 470.2 pm, open-loop maxima up to 18 pm of 5136 pm; compressed soak <= 0.074 pm. The compressed-soak golden
  is not the certified A4 run bit for bit (as after the physics_optics fix).
- **docs/handoff/controller_scorecard.md / .json regenerated** (tools/scorecard_compare.py against the physics_optics
  file, runs/physics_thermal_fix/scorecard_compare.log): pi_lock(_c), composite, kalman_lqg(_c), mpc(_c) steady rms and
  max within 0.0006 pm; under A1 ring 0 348.6 -> 346.8 pm (PI) and 363.7 -> 362.0 pm (Kalman / MPC), rings 1-7 max
  5.99 -> 5.89 pm. The acquisition and peak-seeking outcomes flip with the small change (docs/ENGINE.md 5): es_po
  holds 1 of 8 rings steady (4 before), midranging 8 of 8 (7), and the wdm_supervisor cold start (steady case) ends
  with one ring off its declared line (0 before; d_ref up to 4.7 nm, mean OMA penalty 4.88 dB instead of 1.02);
  every A1 locked-at-the-end count is unchanged.
- **docs/VALIDATION.md and VALIDATION_g26_calibrated.md recomputed in full** (every part was stale: g26 changed, the
  named corners and the engine too); the closure table has the new row crosstalk-dynamics.
- Tests: test_thermal.py (ROM time scales on the six artefacts, interim per-pole scaling, imec Table II from the
  source-to-victim distance, per-ring radii); test_params_realize.py (g26 radii); tests/closure/
  test_thermal_crosstalk.py (the corner reaches the FEM within 10 % in DC and 1.3x in t63, the far-site floor, the
  closure xfail); tests/closure/test_thermal_notes.py (every number of the two notes recomputed from the artefact and
  the realized model; ARCHITECTURE 5.3); test_engine_warm_start.py (settle's step against the bound, one bound for
  both substep policies); test_presets.py pins the new corner. test_interim_matches_07_p1_2_numbers now expects the
  30 um pitch less two 7.5 um radii (it encoded the centre distance). test_vectors_equivalence.py: the run-specific
  value of the derived float-vs-fixed bound on the regenerated recording is 1.28 codes (1.19 before; larger errors
  under the faster aggressor -> ring 0 pole); its pin moved from < 1.25 to < 1.32, the measured difference is still at
  most 1 code. **Not fixed, controller fallout:** test_ctrl_b_wdm.py::test_cold_start_locks_every_ring_on_its_line
  fails on `map_ok`: on the corrected plant the WDM supervisor's cold-start map events of rings 1 and 2 move by one
  sweep step and its line-label fit no longer clears its own ambiguity margin (2.0, calibration.label_lines_by_pattern)
  for them (ambiguity of rings 0-3: 3.09 / 2.10 / 5.55 / 5.37 before, 3.85 / 1.93 / 1.98 / 2.06 now: ring 1 was
  already 5 % above the margin); the outcome of that reference-plant run is unchanged (every ring locked on its
  declared line, the same labels, no wrong-line flag, detuning within 0.6 pm). Left to the controllers_b owner (the
  margin is a controller design choice; on the engine scorecard the same supervisor now ends one ring off its line).
  **Not fixed, controller fallout (slow):** test_ctrl_c_kalman_lqg.py::test_a1_toward_the_cusp_dark_current_runaway
  (the stress scenario a1_pd_adjacent, aggressor 5 um from PD0): with the faster aggressor -> ring 0 pole the
  bank-wide cascade that controllers_c.md describes for a longer run (ring 0 at full scale, its next order starving
  rings 1-2) now starts inside the test's 7 ms window: ring 0 ends 16.6 nm off (0.27 nm before), rings 1-7 3.3-5.5 nm
  off, and kalman_lqg still declares rings 3, 4 and 7 locked, which the test's last assertion forbids (a lock flag of
  the controller that the cascade fools). Left to the controllers_c owner. Improved: the non-strict xfail
  test_ctrl_b_wdm_slow.py::test_online_barrel_shift_when_the_chip_cools now passes.

## 2026-10-16: physics_optics review fixes (bus segment loss, derive's BND-1 diagnostic, rigid-comb bound)

Three findings of the physics_optics review, confirmed by an independent verifier. Records of the runs:
runs/physics_optics_fix/ (pre / post regression scenarios, compare.log, scorecards before / after / pre-fix).

- **Engine bug fixed: the bus waveguide loss between rings.** `Plant._build` passed no `bus_t2` to
  `mrm.plant.optics.pack_selfheat_args`, so every bus segment transmitted 1 (and so did kernel_ref, which copies the
  plant's rows), while derive (`p_in_ring_dbm`), the replicas' bus walk and ENGINE_BRIEF 4.2 apply
  exp(-bus_loss x spacing) (g26: 0.073 dB/mm, printed, over the assumed 50 um: 0.99916 per segment, 0.59 % less light
  at ring 7 than at ring 0). New `mrm.plant.phase.bus_t2_from_plant_params(pp)` (B, N) (phase equation 6: 1 for the
  first ring in bus order, the segment loss for every later ring; the output segment is not modelled), used by the
  plant, `runner.quasi_static` (its own-line walk; so `warm_start_codes` and the settled warm start) and
  `mrm.control.models.bank_optics` (static_point, linearize: the controllers' nominal model). No njit code changed;
  the plant with the helper patched to 1 reproduces the former engine bit for bit (the 2026-10-10 scorecard exactly).
  ENGINE_BRIEF 4.2 corrected (it said g26 has no spacing).
- **Signals that changed** (tools/perf_scenarios.py, runs/physics_optics_fix/pre against post; pre = runs/perf_r2/
  final3 bit for bit; 306 of 476 signals differ): the power at each ring falls by 0.084 % per segment (ring 7:
  -0.587 %) in every scenario, with the absorbed and drop power, monitor current and codes (-0.6 % at ring 7);
  quasi-static and warm-start heater codes rise by up to 21 codes (ring 7) to replace the lost self-heating, their
  targets are unchanged (+103.7 to +105.2 pm). Steady closed loop (steady_pi, a1_pi after the peak map): detuning
  within 0.08 / 0.11 pm of before; open loop at the warm-start codes: laser_step within 0.04 pm, idle_traffic within
  1.7 pm (0.08 pm in its second half).
  Larger differences are timing shifts of transients: pi_lock_c's peak map (82-87 pm at t = 2.4 ms), the heater sweep
  across the self-heating fold and the foreign lines (10.4 nm at 18.8 ms, 512 pm in its second half, 4.3 pm final),
  and in mc8_pi (sampled worlds) one ring that takes another trajectory (487 pm; 29 of 64 rings change their time
  above 5 pm).
- golden: tests/golden/g26__static.npz updated (data sha256 3fa3e2ee2fe6): the engine applies the bus waveguide loss between rings (g26 0.073 dB/mm over 50 um, 0.99916 per segment; Plant._build passed no bus_t2), and quasi_static and the warm-start codes include it (physics_optics review, 07 P1-9 bus walk)
- golden: tests/golden/g26__a1_slide10_aggressor.npz updated (data sha256 89c60eedab46): the engine applies the bus waveguide loss between rings (g26 0.073 dB/mm over 50 um, 0.99916 per segment; Plant._build passed no bus_t2), and quasi_static and the warm-start codes include it (physics_optics review, 07 P1-9 bus walk)
- golden: tests/golden/g26__soak_compressed.npz updated (data sha256 6afc636789a8): the engine applies the bus waveguide loss between rings (g26 0.073 dB/mm over 50 um, 0.99916 per segment; Plant._build passed no bus_t2), and quasi_static and the warm-start codes include it (physics_optics review, 07 P1-9 bus walk)
- golden vectors: tests/golden/vectors/fxp_pi__g26__aggressor_square.npz updated (sha256 ed85d48efb39): re-recorded with the README command after the bus segment loss fix (13991 of 98400 observation words and 2838 of 9600 action words differ)
- Golden metric changes: lock and detuning metrics of pi_lock_c, fxp_pi and the soak by <= 0.25 pm (<= 0.2 % of
  their values; pi_lock_c's peak map is scored), mean P_abs -0.6 % at ring 7, heater codes up to +21; the open-loop
  A1 maxima by up to 5.9 pm (of up to 5136 pm); under A1 ring 0 of pi_lock_c counts 2 excursions instead of 1 (its
  350 pm heater-authority excursion at the hot cusp crosses the threshold once more).
  The compressed-soak golden no longer equals the certified A4 convergence run bit for bit (that run, and its
  certificate, are on the engine before this fix; the certificate's config fingerprint is unchanged).
- **docs/handoff/controller_scorecard.md / .json regenerated** (tools/scorecard_compare.py against the 2026-10-10
  file): pi_lock(_c), composite, kalman_lqg(_c) and mpc(_c) change by <= 0.01 pm in steady rms jitter and in the A1
  maxima of rings 1-7 (feedforward_pi still loses every ring under A1); es_po holds 4 of 8 rings in steady state
  (7 before), acquire_then_lock settles on 3 of 8 (8), midranging on 7 of 8 (6): these peak-seeking / acquisition
  outcomes flip with the small change (docs/ENGINE.md 5).
- **docs/VALIDATION.md and VALIDATION_g26_calibrated.md re-rendered (`--fast`)** for the derive change below. Their
  predictions and identifiability parts do not use the engine (the replicas already applied the loss); the
  **sensitivity and ranking parts were computed on the engine before this fix** (pi_lock_c under A1, compare_locks)
  and are not flagged stale by `report.CODE_DEPS` (it tracks replicas.py and sensitivity.py only): recompute them
  (`--recompute sensitivity,ranking`, ~50 min for g26, ~30 min for g26_calibrated) at the next integration.
- **derive's BND-1 diagnostic from the plant's phase model** (`mrm.params.derive.next_order_distances`):
  `next_order_distance_ghz` / `_fwhm` put every order at f_line + k FSR_cold and divided by the cold FWHM (g26 200 GHz,
  3.33-3.37 FWHM). Now: the order positions of the ring locked on its own line from the phase model (the lock-shifted
  cold comb with dispersion), each order's TCMT FWHM at its wavelength, TX lines at the realized frequencies; equal to
  the BND-1 replica (`replicas.adjacent_order_penalty`, 1e-10) whenever that is <= 2 FWHM (derive keeps the orders
  outside the TX span). g26 176.7-181.7 GHz, 2.79-3.01 FWHM; g26_fsr_window_a 1.68-2.51 FWHM (now warns: soft fail;
  was 2.16-2.19 without a warning); g26_fsr_edge 1.33-1.51 (was 1.16-1.18). Inputs and provenance follow (solver
  (assumed-geometry)). PARAMETERS.md regenerated; PARAMETERS_FORMAT.md section 10 and params_content.md item 3 (window A
  was chosen with the idealized estimate) annotated.
- **Not fixed: the phase model's rigid comb** (07 P1-6 deviation, now documented in the mrm.plant.phase docstring,
  equation 2, with its bound; docs/ENGINE.md 10; plant_optics.md; extract_modes.md). Every order of a heated ring moves
  by the same shift; the modes artefact's n_eff(lambda, T) table (with L(T)) moves order m + k by 0.43 % of the shift
  per FSR more or less (kappa 4.1-4.5e-4 /nm: +-33 pm at the adjacent orders and +-66 pm at the second ones at the
  7.6 nm g26 lock; locked order exact). The proper fix needs a thermo-optic dispersion key in g26 and changes every
  consumer of the cold curve (kernel Truth and oracle, quasi_static, replicas, calibration inputs: both reports and
  the reviewed calibration would need recomputing); left for its own change. New
  test_phase.py::test_rigid_comb_against_the_solver_table guards the documented bound. Related: Truth
  `detuning_hwhm` normalises by lambda_line^2 fwhm_hz / (2c), 1.2 % wider than the rigid line at the lock
  (ENGINE.md 7; diagnostic only).
- Tests: new tests/verification/test_bus_loss.py (engine on both optics paths, quasi_static, static_point, derive);
  test_replicas_static.py::test_other_lines_match_the_control_models_static_point now asserts line and absorbed
  power to rel 1e-9 (was 0.01: its "< 1 %" residual was the missing bus loss; measured 1.3e-14);
  test_plant_wiring.py's ReferencePlant passes bus_t2 and its light balance counts the segment loss;
  test_params_derive.py checks the orders against kernel.cold_coordinate and against the BND-1 replica (3 presets).
  test_ctrl_b_robust.py::test_bitstat_ratio_lock_under_a_swing_loss judges its 4 pm p-p bound (unchanged) as the
  median over noise seeds 1-8 instead of seed 7 alone: the p-p is a sample of the ratio loop's wander, 0.9-6.5 pm
  over seeds on the fixed engine and 1.9-6.2 pm on the former one (medians 3.1 / 2.95); seed 7 alone moved 2.98 ->
  5.76 pm with the fix. Every seed must still LOCK on the red side with alpha at or above its floor.

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
