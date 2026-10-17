# Known issues

The open items after the final review of 2026-10-16/17, most important first. One or two lines each; the details
are in the file named on each line. The record of the review is docs/handoff/final_review.md.

## 1. Stale results (read these first)

1. **The calibration is stale.** `g26_calibrated`, `g25_test_vehicle_calibrated` and `mrm/calibration/latest_fit.json`
   were fitted before the params fixes (params-1 to -13) and were not refitted. Refit every group (`python -m
   mrm.calibration`) when the real device parameters arrive (docs/handoff/calibration.md, top note).
2. **Both validation reports are stale.** docs/VALIDATION.md and docs/VALIDATION_g26_calibrated.md predate the
   final-review fixes and carry a STALE banner. Regenerate them after the refit (`python -m mrm.validation.report`);
   until then `test_report.py::test_validation_md_is_current` is `xfail(strict=False)`.
3. **`tests/calibration::test_loss_worlds_hold_the_solver_dispersion` is xfail.** Its recomputed chi2 is 7.77
   against the recorded 7.51. The secant beta (params-5) causes the gap; the refit closes it.
4. **The A4 soak has no current result.** Its 2026-10-10 PASS used the old hop criterion and a certificate that no
   longer covers the plant or the code. Run A4 again, convergence experiment included (4.8 h or more).
5. **The 2026-10-10 result tables are dated records.** experiments.md sections 7 and 12 give A1 ring 0 as 348.7 pm;
   it is now 454.9 pm. The stress-site figures of examples/controllers/README.md were not measured again.

## 2. Model accuracy: validation findings (confirmed, not fixed)

6. **validation-1 (major): CAL-2 and VAL-2a/b use the median G25 device, not the swept device.** So the coupler
   dispersion and the "calibrated" loss coefficients are 10-15 % off, about 1-1.5 dB of ER at the line. Fit CAL-2
   on the swept device (as CAL-4 is), then refit.
7. **validation-2 (major): the heater leaves are refitted per thermo-optic world only.** The thermal.family choices
   silicide_undercut and metal_above (half the family weight) and the g26_undercut corner miss CAL-3b: R_hot is
   5-10 % low and the droop y is 20-35 % off.
8. **validation-3: g26_calibrated samples jointly identified leaves independently** (Q_L with alpha_other on CAL-2,
   gain_scale with chip_input_loss on CAL-8; gain_scale up to 2.0, above its 95 % hull). Some worlds of the
   calibrated report sections 4-5 miss the calibration targets.
9. **validation-7: the FSR world window_a breaks BND-1** (1.68 FWHM on g26, 1.33 on g26_calibrated) but keeps
   weight 1 of 3. It is the largest tornado entry of the calibrated report and widens the robustness spreads.
10. **validation-4: `ring.thermal.heater_self_ratio` is tagged "calibrated", but CAL-4 identifies only
    y = TCR x R_th,H** (TCR held at 1.4e-3). Tag it "inferred" or calibrate y.
11. **validation-5: the reports do not flag VAL-1 as a swept-device target.** VAL-1 shows a 43-53 sigma heater-law
    failure; on the swept device it passes (13.05 nm, z 1.0).
12. **validation-6: g26 calibration passes by construction are not all flagged.** CAL-1, CAL-11, CAL-12a and CAL-12d
    cannot fail; only CAL-5, 8, 10 and 12b are real fit gains of the calibration.
13. **validation-8: the report staleness check is narrow.** It hashes only replicas.py and sensitivity.py; the
    extraction artefacts, mrm/waveform, mrm/scenarios, mrm/control and mrm/sim are not checked, and ranking has no
    entry. `mrm report --check` can pass on a stale report.
14. **validation-9: the calibrated report copies its frozen-targets block from itself,** and its "Frozen-hash check
    now: OK" checks docs/VALIDATION.md only. A stale block in the calibrated report is not detected.

## 3. Model accuracy: other open items

15. **The phase model is a rigid comb** (physics_optics-1). Other orders move +-33 pm (adjacent) and +-66 pm (second)
    at the 7.6 nm lock; the locked order is exact; foreign-line P_abs is off by <= 0.15 % (g26), ~1 % (FSR corners).
16. **Which crosstalk package G26 has is unknown.** The default (imec, natural convection: 3 % nearest, 1.52 ms) and
    the FEM TEC-held package (1.0 %, 0.12 ms; corner g26_tec_package, weight 0) disagree; strict xfail
    `test_g26_default_crosstalk_matches_the_fem_rom` (VALIDATION.md closure-crosstalk-dynamics).
17. **The film-200 world keeps the rib160 couplers and fabrication sigmas** (params-3): die resonance spread 4.2 nm
    instead of about 6 nm. The content owner decides.
18. **The 1.5 nm local gap sigma shared by both couplers has no source** (params-11).
19. **The fitted coupler dispersion is about 3x the solver dispersion** (params-13; calibration.md section 8); the
    dispersion leaves have no declared range, by design.
20. **The secant beta is extrapolated from 125 to 170 degC** (params-5): +0.96 % at the end of the P_FSR sweep. Extend
    the modes table (extraction owner).
21. **s0 is one value for every line** (params-4): about +-0.35 % heater-gain error at the ends of the bank.
22. **The interim model takes a heater self ratio, not an absolute R_th,H**: y follows Z across worlds (0.0075 /mW
    under window_a and H86 against the 0.00904 /mW fit; integrator.md section 3, plant_thermal request c).
23. **The thermal FEM artefact misses the imec targets** VAL-8a/b (time constants 77 % and 69 % off, tolerance 30 %)
    and VAL-8c (efficiency 31 % vs 25 %): the three failing tests of the fast suite (extraction owner).

## 4. Failing targets (as last computed; both reports are stale), one line per family

24. Heater tuning law: CAL-4 fails (z 54 g26, 47 calibrated; median vs swept device), VAL-1 fails (15.6 / 15.2 nm
    against 13 nm).
25. Notch depth against heater voltage: CAL-2 fails (z 82 / 23; the replica has no notch floor); VAL-2a/b fail on g26,
    pass on g26_calibrated.
26. Heater code map: CAL-9a/b/c fail on both presets (z 53-130): heater authority at the hot cusp, the aggressor site
    and the referencing (calibration.md 11.5).
27. Junction and EO: CAL-5, CAL-10 and VAL-3a fail on g26 and pass on g26_calibrated; CAL-10 identifies
    (R_s + 50 ohm) x C_j, not R_s x C_j.
28. Monitor at the lock: CAL-8 fails on g26 (18.6 against 15.5 uA), passes on g26_calibrated.
29. Laser: CAL-12b fails on g26, passes calibrated; CAL-12c fails on both (0.02026 nm against <= 0.02 nm: one
    conversion at 1310 nm for every line, scenarios owner).
30. relOMA: CAL-14 (-6.07 / -5.24 dB against -2.1 dB), VAL-4a/b (no -3 dB contour), VAL-4c and VAL-5 fail on both:
    a drive or definition gap (07 Q4).
31. Published geometry: VAL-7b fails (definition conflict), VAL-7a is unverified, VAL-7c is blocked (no de Cea stack).
32. Thermal FEM vs imec: VAL-8a/b/c fail on both (item 23); VAL-8e would fail.
33. Bounds: BND-3a/b are blocked (the named controllers are not registered).

## 5. Controllers

34. **feedforward_pi gain runaway under A1.** A fix from a separate session (feed-forward as the heater-law secant,
    anti-windup at the applied rails) is in the tree since checkpoint a62f429 (CHANGELOG "feedforward_pi gain
    schedule"). This review did not review it; treat it as unreviewed.
35. **wdm_supervisor cold start: ring 1's line labels are marginal.** Its map is fitted alone in the wide FSR prior
    (the narrowing needs two confident rings), ambiguity 2.15 against the 2.0 threshold. It went to 1.93 after
    physics_thermal-3 (test_cold_start_locks_every_ring_on_its_line failed) and back after the params fixes. The lock
    outcome was right both times.
36. **A1, default site: ring 0 runs out of heater at the hot cusp** (~455 pm, heater at code 0 for ~15 ms, every
    controller); A2 the same; the stress site loses ring 0 for every controller (ENGINE.md section 5).
37. **Sampled worlds lose rings under pi_lock_c from a good warm start** (mc_pi_lock: 10 worlds with no warm-start
    flag). A claim over the uncertainty set needs that analysis (ENGINE.md section 10).
38. **The array heater supply limit binds in about 1/3 of the g26 LHS worlds** (0.2 W, 0.15 A, both assumed). There
    no controller can hold the bank; it needs a source or a range (integrator.md 8.3).
39. **g26_calibrated is fragile to self-heating under A1** (eta high, z_ratio 2: one ring unlocked, one on a foreign
    line) where g26 is not (calibration.md 11.6).
40. **Wrong-line capture at power-up**: each cold g26 ring sits ~2 FWHM blue of line n+3 and can self-heat onto it;
    acquisition controllers need a supervisor (integrator.md section 4).
41. **A named scenario runs without the config's signal defaults**: bias 1.0 V instead of 1.05 V, ~4 heater codes
    and up to ~100 pm of transient apart; name-based and config-based results do not compare number for number.
42. **A corner chosen with `alts=` still reaches `model_nominal`** (controllers-2 part): one knowledge vocabulary and
    a `controller_knowledge` preset are open (controllers_c.md, integrator.md 6.4).
43. **Lock-logic limits** (controllers_a 7.4, controllers_b 7.4, controllers_c 6): a fast red step reads as a fall
    (strict xfail); the WDM DOWN scan can take another line's peak as the set-point; KalmanLQG/MPC fail in some
    reachable worlds (xfail); bias tunes weakly in g26 (0.1-0.5 pm per 0.2 V).
44. **The harnesses use the unshifted phase slope** 2 pi c t_rt / lambda^2 (1-1.2 % of the in-tick thermo-optic phase
    after a 5-8 nm shift); the engine uses the shifted one (integrator.md 6.4).

## 6. Engine, experiments and tooling

45. **With tick_jitter_s > 0, chunk_ticks still changes the auto-policy step sequence**, and a score_from_s split
    off a controller edge changes the trajectory (it warns ScoreSplitWarning) (engine-4 part; ENGINE.md section 2).
46. **The code fingerprint cannot tell a comment edit from a real change** (experiments_docs-2 part): any edit of a
    fingerprinted file, or a numpy/numba change, invalidates every convergence certificate (2 h or more to remake).
47. **The mini soak passes the 0.5 pm convergence cap by 0.03 pm** (experiments_docs-4 part): its ring 1 set point
    moves 0.47 pm between factors; any engine change can flip test_experiment_runs.py.
48. **Not built**: `fidelity.data_modulation: lut` in the kernel, the reference-tap and `datapath.bitstat_ones`
    channels (a reference-tap config is refused), `DacFacts.law`, `run_lock_request`, the Verilator/cocotb HDL
    co-simulation, an in-kernel bridge, per-world reset in the gym env, the 07 P1-22 throughput (ENGINE.md 10).
49. **Sensing details not modelled**: the bit-statistics window levels at the first edge after a reset are reset
    placeholders; a shared integrating ADC's de-integration time is not modelled (ENGINE.md section 10).
