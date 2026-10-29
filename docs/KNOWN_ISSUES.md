# Known issues

This is the list of open issues from the final review on **2026-10-16 and 2026-10-17**. It records
what was known at that review; it is not a fresh validation run. The review record is
[handoff/final_review.md](handoff/final_review.md).

Start with section 1 before using old results. Sections 2–4 describe accuracy problems; section 5
covers controllers; section 6 covers the engine and tools. Issue numbers and review IDs are kept so
that references from other documents still work.

A few terms used below:

- **World:** one choice of uncertain device parameters. LHS means Latin-hypercube sampling of those choices.
- **CAL / VAL / BND:** calibration targets, held-out validation targets and bounds checks.
- **xfail:** a test marked as an expected failure. `strict=False` allows it to pass without failing the suite.
- **z / sigma:** the discrepancy expressed in units of the target's uncertainty.
- **FWHM:** the full width of a resonance at half its peak. FSR is the spacing between resonance orders.

## 1. Stale results (read these first)

1. **The calibration needs to be rerun.** `g26_calibrated`, `g25_test_vehicle_calibrated` and
   `mrm/calibration/latest_fit.json` were fitted at commit `b92c519`, before the final-review fixes,
   including params-1 through params-13. When real device parameters arrive, rerun every fit group with
   `python -m mrm.calibration`. See the top note in `docs/handoff/calibration.md`.

2. **Both validation reports are stale.** `docs/VALIDATION.md` and
   `docs/VALIDATION_g26_calibrated.md` predate the final-review fixes and carry a STALE banner.
   Regenerate them after refitting with `python -m mrm.validation.report`. Until then,
   `test_report.py::test_validation_md_is_current` is marked `xfail(strict=False)`.
   Their section-4 Sobol worlds also predate the seed fix of 2026-10-22 (docs/CHANGELOG.md): the AB rows of
   screened entries 1, 3, 5 and 7 shared the noise seeds of the A rows, so only the total indices of
   entries 2, 4, 6 and 8 carry the noise floor that the section-4 text describes.

3. **The loss-world calibration test has an expected failure.**
   `tests/calibration::test_loss_worlds_hold_the_solver_dispersion` now computes chi2 = 7.77,
   compared with the recorded 7.51. The secant-beta change (params-5) causes the difference;
   refitting is expected to close it.

4. **A4 has no current soak result.** The PASS from 2026-10-10 used the old resonance-hop criterion
   and a convergence certificate that no longer covers the plant or code. Rerun A4, including its
   convergence experiment; this takes at least 4.8 hours.

5. **The result tables from 2026-10-10 are historical records.** Sections 7 and 12 of
   `docs/handoff/experiments.md` report 348.7 pm for A1 ring 0; the updated value is 454.9 pm.
   The stress-site figures in `examples/controllers/README.md` have not been measured again.

## 2. Model accuracy: validation findings (confirmed, not fixed)

6. **validation-1 (major): some fits use the wrong G25 device.** CAL-2 and VAL-2a/b use the median
   device, although the measurements came from a particular swept device. This makes coupler dispersion
   and the “calibrated” loss coefficients 10–15% off, or about 1–1.5 dB of extinction ratio (ER) at
   the line. Use the swept device for CAL-2, as CAL-4 does, then refit.

7. **validation-2 (major): heater parameters are not refitted for every thermal hypothesis.** They
   are refitted only for each thermo-optic world. The `thermal.family` choices `silicide_undercut`
   and `metal_above` (together half the family weight), and the `g26_undercut` corner, miss CAL-3b:
   hot resistance `R_hot` is 5–10% low and the droop parameter `y` is 20–35% off.

8. **validation-3: parameters fitted together are sampled independently.** In `g26_calibrated`,
   this affects `Q_L` with `alpha_other` from CAL-2, and `gain_scale` with `chip_input_loss` from
   CAL-8. `gain_scale` reaches 2.0, outside its 95% fitted hull. Some worlds in sections 4–5 of the
   calibrated report therefore miss the calibration targets.

9. **validation-7: an FSR hypothesis fails a bound but remains in the sampled set.** `window_a`
   fails BND-1 at 1.68 FWHM on `g26` and 1.33 FWHM on `g26_calibrated`, yet retains weight 1 of 3.
   It is the largest entry in the calibrated report's tornado sensitivity chart and widens the
   reported robustness spread.

10. **validation-4: one provenance label overstates what the fit identifies.**
    `ring.thermal.heater_self_ratio` is tagged `calibrated`, but CAL-4 identifies only
    `y = TCR × R_th,H`, with TCR fixed at 1.4e-3. Tag the parameter `inferred`, or calibrate `y`.

11. **validation-5: the reports do not identify VAL-1 as a swept-device target.** They show a
    43–53 sigma heater-law failure. Using the device that was actually swept gives a passing
    result: 13.05 nm, z = 1.0.

12. **validation-6: some calibration passes are guaranteed by construction.** CAL-1, CAL-11,
    CAL-12a and CAL-12d cannot fail, but the reports do not consistently flag this. Only CAL-5,
    CAL-8, CAL-10 and CAL-12b represent real fit gains in the G26 calibration.

13. **validation-8: the staleness check misses important dependencies.** It hashes only
    `replicas.py` and `sensitivity.py`. It does not check extraction artefacts, `mrm/waveform`,
    `mrm/scenarios`, `mrm/control` or `mrm/sim`, and ranking has no dependency entry.
    As a result, `mrm report --check` can pass even when the report is stale.

14. **validation-9: the calibrated report does not independently check its frozen-targets block.**
    It copies that block from itself. Its “Frozen-hash check now: OK” checks only
    `docs/VALIDATION.md`, so it cannot detect a stale block in the calibrated report.

## 3. Model accuracy: other open items

15. **The phase model shifts all resonance orders together** (found in the 2026-10-16 optics review). This “rigid comb”
    approximation is exact for the locked order, but at the 7.6 nm lock the adjacent orders differ
    by ±33 pm and the second orders by ±66 pm. Absorbed power `P_abs` from foreign lines is off by
    at most 0.15% in `g26`, and about 1% in the FSR corners.

16. **G26's thermal package is unknown.** The default uses imec natural-convection data:
    3% nearest-neighbour crosstalk with a 1.52 ms time constant. The finite-element (FEM) package
    held by a thermoelectric cooler (TEC) gives 1.0% and 0.12 ms. That package is the
    `g26_tec_package` corner, with sampling weight 0. The disagreement is recorded by the strict
    xfail `test_g26_default_crosstalk_matches_the_fem_rom`; see `VALIDATION.md`,
    `closure-crosstalk-dynamics`.

17. **The 200 nm film world reuses 160 nm assumptions** (params-3). It keeps the `rib160` couplers
    and fabrication standard deviations, giving a die resonance spread of 4.2 nm instead of about
    6 nm. The parameter-content owner still needs to decide how to handle this.

18. **The shared local coupler-gap variation has no source** (params-11). Both couplers share
    a local gap variation with a 1.5 nm standard deviation.

19. **Fitted coupler dispersion is about three times the solver result** (params-13).
    The dispersion parameters intentionally have no declared range. See section 8 of
    `docs/handoff/calibration.md`.

20. **The temperature fit extends beyond the mode table** (params-5). The secant-beta model is
    extrapolated from 125 to 170 °C, giving a +0.96% discrepancy at the end of the `P_FSR` sweep.
    The extraction owner needs to extend the mode table.

21. **All laser lines use the same reference temperature sensitivity, `s0`** (params-4).
    This introduces about ±0.35% heater-gain error at the ends of the bank.

22. **The interim thermal model accepts a heater self-heating ratio rather than an absolute
    `R_th,H`.** Its droop `y` therefore changes with thermal response `Z` across worlds:
    0.0075 /mW under `window_a` and `H86`, versus the fitted 0.00904 /mW. See section 3 of
    `docs/handoff/integrator.md`, the third thermal request (an optional absolute heater
    self-heating resistance).

23. **The thermal FEM artefact misses the imec targets.** VAL-8a/b time constants are 77% and
    69% off, against a 30% tolerance. VAL-8c efficiency is 31% rather than 25%. These are the three
    failing tests in the fast suite; the extraction owner is responsible for them.

## 4. Failing targets (last computed; both reports are stale)

These are the last recorded results, not current validation evidence. Where two values appear in
`g26 / calibrated` order, “calibrated” means `g26_calibrated`.

24. **Heater tuning law.** CAL-4 fails at z = 54 / 47 because of the median-versus-swept-device
    mismatch. VAL-1 fails at 15.6 / 15.2 nm against the 13 nm target.

25. **Notch depth versus heater voltage.** CAL-2 fails at z = 82 / 23; the replica has no notch
    floor. VAL-2a/b fail on `g26` and pass on `g26_calibrated`.

26. **Heater code map.** CAL-9a/b/c fail on both presets, at z = 53–130. The relevant issues are
    heater tuning range at the hot cusp, the aggressor site and how the disturbance is referenced.
    See section 11.5 of `docs/handoff/calibration.md`.

27. **Junction and electro-optic response.** CAL-5, CAL-10 and VAL-3a fail on `g26` and pass on
    `g26_calibrated`. CAL-10 identifies `(R_s + 50 ohm) × C_j`, rather than `R_s × C_j`.

28. **Monitor current at lock.** CAL-8 fails on `g26`: 18.6 µA against 15.5 µA. It passes on
    `g26_calibrated`.

29. **Laser.** CAL-12b fails on `g26` and passes on the calibrated preset. CAL-12c fails on both:
    0.02026 nm against a limit of 0.02 nm. One conversion at 1310 nm is used for every line;
    this is assigned to the scenarios owner.

30. **Relative optical modulation amplitude (relOMA).** CAL-14 gives −6.07 / −5.24 dB against
    −2.1 dB. VAL-4a/b have no −3 dB contour; VAL-4c and VAL-5 also fail on both presets. There is
    an unresolved drive or definition mismatch; see question 6 (relOMA and the drive) in
    `docs/QUESTIONS_FOR_LIGHTMATTER.md`.

31. **Published geometry.** VAL-7b fails because the definitions conflict. VAL-7a is unverified.
    VAL-7c is blocked because the de Cea stack is unavailable.

32. **Thermal FEM versus imec.** VAL-8a/b/c fail on both presets (issue 23). VAL-8e would also fail.

33. **Bounds checks.** BND-3a/b are blocked because the named controllers are not registered.

## 5. Controllers

34. **The `feedforward_pi` runaway fix has not been reviewed.** Under A1, its gain could run away.
    A fix has been in the working tree since checkpoint `a62f429`: it uses the heater-law secant
    for feed-forward and anti-windup at the applied command limits. See CHANGELOG,
    “feedforward_pi gain schedule”. The final review did not assess this fix.

35. **WDM cold-start results are sensitive to a 1–3% heater-law change.** Before the parameter
    fixes, seed 7 worlds 1 and 3 ended on their assigned lines. They now lose 2 and 4 rings.
    This regression is flagged, not accepted as correct. The test
    `test_cold_start_sampled_worlds_1_and_3_end_on_their_lines` is `xfail(strict=False)`.

36. **The WDM supervisor can still accept the wrong line.** The final check fixed one case
    involving a narrowed label fit at its window boundary. Another remains: with the coupling
    Jacobian off, seed 7 world 3 ring 2 targeted its unfitted first resonance and was accepted
    on a foreign line. This is assigned to the controllers_b owner.

37. **Ring 1's wavelength labels are marginal in the nominal WDM cold start.** Its map is fitted
    alone with the wide FSR prior. The ambiguity score is 2.15 against a 2.0 threshold. It fell
    to 1.93 after the 2026-10-16 fix of the crosstalk distance, failing `test_cold_start_locks_every_ring_on_its_line`, then
    recovered after the parameter fixes. The actual lock was on the correct line.

38. **Ring 0 cannot reduce heater power far enough at the hot end of A1 and A2.** At the default
    site under A1, every controller reaches heater code 0 for about 15 ms, with 454–473 pm of
    lock error. For A2 on line 0, the heater's contribution to the lock temperature (80.8 K)
    equals A2's temperature rise. The full-cycle test therefore runs ring 0 on line 4; the
    line-0 case is xfail. At the stress site, ring 0 is always lost.

39. **`pi_lock_c` loses rings in some sampled worlds despite a good warm start.** In `mc_pi_lock`,
    10 failing worlds have no warm-start flag. Those cases need analysis before claiming
    performance over the uncertainty set. See ENGINE.md section 10.

40. **The assumed heater supply limit is reached in about one-third of G26 LHS worlds.** The
    limits are 0.2 W and 0.15 A, both assumed. No controller can hold the whole bank in those
    conditions. The limits need a source or a range; see `docs/handoff/integrator.md` section 8.3.

41. **The calibrated preset is more fragile to self-heating under A1 in one tested case.** With
    high `eta` and `z_ratio = 2`, `g26_calibrated` has one unlocked ring and one on a foreign line,
    while `g26` does not. See `docs/handoff/calibration.md` section 11.6.

42. **Cold rings can capture a foreign line at power-up.** Each cold G26 ring sits about 2 FWHM
    blue of line `n+3` and can self-heat onto it. Acquisition controllers need a wavelength
    supervisor; see `docs/handoff/integrator.md` section 4.

43. **Named scenarios do not inherit the config's signal defaults.** They use 1.0 V bias rather
    than 1.05 V. Results differ by about 4 heater codes and up to 100 pm during the transient.
    Name-based and config-based runs therefore cannot be compared number for number.

44. **A controller with `model_nominal` can see an alternative selected by `alts=`.** The
    nominal view still comes from the run's own config. Unifying the knowledge-token vocabulary
    and adding a `controller_knowledge` preset remain open (left over from the 2026-10-16 controllers review). See
    `docs/handoff/controllers_c.md` and `docs/handoff/integrator.md` section 6.4.

45. **Several lock-logic limits remain.** A fast red step is mistaken for a branch fall
    (strict xfail). A WDM DOWN scan can use another line's peak as the set-point. KalmanLQG/MPC
    fail in some reachable worlds (xfail). Bias tuning is weak in `g26`: 0.1–0.5 pm per 0.2 V.
    See controllers_a section 7.4, controllers_b section 7.4 and controllers_c section 6 in
    `docs/handoff/`.

46. **The controller harnesses and engine use different phase slopes.** The harnesses use the
    unshifted slope `2 pi c t_rt / lambda^2`; the engine uses the shifted slope. The difference
    is 1–1.2% of the in-tick thermo-optic phase after a 5–8 nm shift. See
    `docs/handoff/integrator.md` section 6.4.

## 6. Engine, experiments and tooling

47. **Automatic substeps can depend on block and scoring boundaries.** With `tick_jitter_s > 0`,
    changing `chunk_ticks` changes the auto-policy step sequence. A `score_from_s` split away
    from a controller edge also changes the trajectory and emits `ScoreSplitWarning`.
    See engine-4 and ENGINE.md section 2.

48. **Even a comment edit invalidates convergence certificates.** The code fingerprint does not
    distinguish comments from executable changes. Editing any fingerprinted file, or changing
    NumPy/Numba versions, invalidates every certificate. Regenerating them takes at least
    2 hours (experiments_docs-2 part).

49. **The mini-soak convergence test has little margin.** Ring 1's set-point moves 0.47 pm between
    compression factors, only 0.03 pm below the 0.5 pm cap. An engine change can flip
    `test_experiment_runs.py` (experiments_docs-4 part).

50. **Several planned features are not built.** These are `fidelity.data_modulation: lut` in the
    kernel; the reference-tap and `datapath.bitstat_ones` channels; `DacFacts.law`;
    `run_lock_request`; Verilator/cocotb HDL co-simulation; an in-kernel bridge; per-world reset
    in the Gym environment; and the RL throughput target (1e5 steps/s). A reference-tap config is
    explicitly refused. See ENGINE.md section 10.

51. **Two sensing details are not modelled.** At the first edge after reset, bit-statistics
    window levels are reset placeholders. A shared integrating ADC's de-integration time is
    also omitted. See ENGINE.md section 10.
