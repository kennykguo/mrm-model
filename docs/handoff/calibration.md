# Handoff: calibration (`mrm/calibration/`, `g26_calibrated.yaml`)

**STALE until the real device parameters arrive (known issue, docs/KNOWN_ISSUES.md).** The fit record
`mrm/calibration/latest_fit.json` (run 2026-10-16T02:43:48Z, commit b92c519) and the presets generated from it,
`g26_calibrated.yaml` and `g25_test_vehicle_calibrated.yaml`, predate the params review fixes params-1 to -13
(docs/CHANGELOG.md, 2026-10-16): the secant beta (params-5), Z_heater->ring at the P_FSR line (params-4), the R0
range [37, 58] that the heater group's prior now reads (params-13), the rib200 film-200 world (params-3), the couplers,
FSR and bend radiation moved by the sampled geometry (params-1, -2, -10, -11), the new ranges (params-8, -9) and the
notes (params-7, -12). Both presets extend g26 and so run with these changes, but their fitted values were fitted on
the model before them. Only their coupling_regime label (params-14) was rewritten, by the generator from the same
record. No refit was run (a refit started for the params review was stopped and is not used). On the present model
38 of the 39 `pytest -m calibration` tests still pass; `test_loss_worlds_hold_the_solver_dispersion` is marked
xfail (strict=False): its recomputed loss-world chi2 is 7.77 against the recorded 7.51, and the tangent beta brings
back the recorded value. Refit every group (`python -m mrm.calibration`, section 1) when the real device parameters
arrive; `--write-preset-only` from this record does not make the presets current. Every number below is from this
stale record.

Owner files: `mrm/calibration/` (`core.py`, `groups.py`, `pipeline.py`, `preset.py`, `report.py`, `cli.py`,
`__init__.py`, `__main__.py`, the generated fit record `latest_fit.json`), `mrm/params/presets/g26_calibrated.yaml`
and `g25_test_vehicle_calibrated.yaml` (generated; the second added by the integrator, section 11), `tests/calibration/` (`conftest.py`, `test_fit_machinery.py`, `test_calibrated_preset.py`,
`a1_g26_calibrated.yaml`) and this file. Contract: 07 P0-5, P1-14, P1-23 and section 4 (4.3 in particular); the
calibration targets and the contamination warnings of `mrm/validation/targets/README.md`. Nothing outside these files
was edited. The calibration package reads only the calibration targets (`load_calibration_targets`); the isolation
scan of `tests/verification/test_validation_isolation.py` passes on it, and `test_every_fitted_residual_is_a_
calibration_target` checks that every target id in the record is a calibration id with `fit: true`. No held-out
target was evaluated by this task.

State of this file: rewritten at the adversarial review (section 10 lists the review's findings); section 11 is the
integrator's record (fixes to the record, the held-out targets and the acceptance experiments on g26_calibrated, and
the default-preset recommendation). Every number below
comes from the record `mrm/calibration/latest_fit.json` of the review's run (heater and coupling_er in one process,
the later groups in a second process on the identical heater and coupling leaves, checked equal to 1e-12, then
merged; the record's `notes` say so).

## 1. How to run

```
MG=".venv/bin/python tools/memguard.py --"
$MG .venv/bin/python -m mrm.calibration                    # every group (~5 h), preset, before/after
$MG .venv/bin/python -m mrm.calibration --groups dac laser # some groups (the other groups' leaves from latest_fit.json)
$MG .venv/bin/python -m mrm.calibration --write-preset-only    # regenerate g26_calibrated.yaml and
                                                             # g25_test_vehicle_calibrated.yaml from latest_fit.json
$MG .venv/bin/python -m mrm.calibration --before-after-only    # the replica table only
$MG .venv/bin/python -m pytest -m calibration tests/calibration  # reported, not CI-blocking
```

`mrm calibrate` is the same command (registered in `mrm.cli.PROVIDERS` by the integrator). Peak RSS 0.2-0.8 GB.

## 2. Public API

```python
# mrm.calibration.core
Param(name, path, lo, hi, start, unit="", log=False, role="leaf"|"nuisance", prior_source="", declared_prior=True,
      declared_range=None)              # declared_range: the leaf's own range when [lo, hi] widens it
Problem(group, hypothesis, params, targets, predict: Callable[[dict], dict], fixed={}, starts=[], constraint=None,
        derived=None, common_sigma={})  # .residuals(theta), .chi2(theta), .predictions(theta) (cached)
fit(prob, *, max_nfev=200, extra_starts=0, seed=0) -> FitResult      # least_squares from every start, all minima
profile(prob, fr, name, *, n_grid=13, max_nfev=60, refine=6) -> Profile   # whole prior range, bisected crossings
restrict(prob, fixed: dict, *, hypothesis=None) -> Problem           # conditional fit (held parameters)
compare_hypotheses(group, fits, note="") -> HypothesisResult          # LR, Birge-scaled LR, AIC
target_residuals(target, prediction, *, sigma_scale=1.0, common_sigma=None) -> np.ndarray  # whitened if common_sigma
half_prior(param, interval) -> float        # width / prior width (log coordinates; the declared width if narrower)
n_data(targets) -> int; calibration_targets() -> {id: Target}; DCHI2_95 = 3.841; INTERVAL_WALL_FRAC = 0.005
# mrm.calibration.groups: Context(device, system, ranges).config(preset, extra=None, alts=None); one builder per group
#   heater_problems(ctx, targets, hypotheses=("same_rth_heater", "same_ratio"), thermo=HEATER_THERMO),
#   coupling_er_problems, offset_problems, junction_problems(ctx, targets, *, lut_variant=True), eo_problems,
#   monitor_problems, modulation_depth_problems, modulation_depth_scan(ctx, targets, *, swings, eff_scales,
#   conventions, mode="oracle", alt_effs=None), dac_problems (+ dac_codes), laser_problems;
#   notch_floor(depth_db, floor_db), CouplingConstants, bench_lut_mapping, widened_child;
#   HEATER_THERMO = ("H_solver", "H86"); CAL4_COMMON_SIGMA_NM = sqrt(0.04^2 - 0.03^2)
# mrm.calibration.pipeline
run(groups=None, *, quick=False, out=RESULTS_PATH, log=print, prior=None) -> CalibrationRun
run_group(name, ctx, targets, *, quick=False, log=print) -> GroupResult
coupling_context(record) -> Context; refresh_coupling_worlds(path=RESULTS_PATH) -> dict
load_results(path) -> dict; write_results(run, path)
GROUP_ORDER = ("heater", "coupling_er", "offset", "junction", "eo", "monitor", "modulation_depth", "dac", "laser")
REJECT_LR = 0.05; TIE_DCHI2 = 0.01; COUPLING_WORLDS; HEATER_WORLDS = (("H86", "thermo_optic", "H86", "cal_h86"),)
LeafDecision, GroupResult, CalibrationRun
# mrm.calibration.preset
write_preset(results, path=PRESET_PATH) -> Path; preset_document(results) -> (dict, header lines)
bench_config(results) -> Config      # g25_test_vehicle + the calibrated device leaves (the G25 bench, calibrated)
bench_overrides(results) -> dict; bank_config(path) -> Config; PRESET_NAME = "g26_calibrated"
# integrator: the same bench as a preset (the report's G25 companion of g26_calibrated)
write_bench_preset(results, path=None) -> Path; bench_preset_document(results) -> (dict, header lines)
BENCH_PRESET_NAME = "g25_test_vehicle_calibrated"; BENCH_PRESET_PATH
# mrm.calibration.pipeline (integrator)
context_from_decisions(record) -> {"device": {...}, "system": {...}}   # what a serial run leaves in its context
# mrm.calibration.report
before_after(results, *, ids=None, preset_path=PRESET_PATH, bench_preset_path=BENCH_PRESET_PATH, log=print)
    -> {id: {"before", "after", "plant_after", "fit"}}; format_table(ba); summary_markdown(results)
# mrm.calibration.cli
register_cli(register)               # "mrm calibrate"
```

## 3. Method (decided before the fits; the review's changes are marked "review"; deviations in section 7)

- **Forward model**: the replicas only, on the preset each target's generation runs on: G25 targets on
  `g25_test_vehicle`, G26 targets on `g26`. Parameters enter as overrides on top of the leaves of earlier groups
  (`Context`). Where a fit needs a leaf outside its own `range` (R0, the swept device's derived ratio and R0), a
  scratch child preset restates the leaf with a wider range (`widened_child`); nothing is written to the package
  except the calibrated preset.
- **Order** (review): heater -> coupling_er -> offset -> junction -> eo -> monitor -> modulation_depth -> dac -> laser.
  CAL-2 is read at heater voltages, so it needs the calibrated heater law; the heater targets do not depend on the
  coupling leaves. One pass is therefore consistent (the first version fitted coupling_er first).
- **Likelihood**: chi2 of standardized residuals with each published sigma (read-off sigmas as printed in the
  targets). CAL-1 (no sigma) is a flat likelihood on [3500, 4200] with a steep wall; it is a constraint, not a datum,
  so it adds no degree of freedom (`core.n_data`). CAL-9a/c compare [lo, hi] elementwise with their sigma. CAL-4
  (review): the 14 shifts are differences against the same 0 V read-off, so they share one error; the residuals are
  whitened with the covariance diag(sigma^2) + 0.026^2 nm^2 off the diagonal (the target's 0.04 nm diagonal with the
  stated 0.03 nm own read-off of each point). A nuisance with an informative prior (the swept device's R0) adds a
  constraint residual, not a datum.
- **Fit**: `scipy.optimize.least_squares` (trf, bounded to the prior, 2-point differences) in normalized prior
  coordinates, from several starts; distinct minima are kept (mirror branches are separate hypotheses).
- **Profiles**: chi2 re-minimized over the other parameters on a grid over the whole prior, threshold crossings
  bisected. 95 % interval: delta chi2 <= 3.84 s^2 with the Birge ratio s^2 = max(1, chi2_min/dof), so a model-form
  misfit widens the interval instead of shrinking it around a wrong value (raw intervals are in the record). Every
  leaf whose tag a profile can decide is profiled at the full resolution (review).
- **Hypotheses**: compared by chi2 (LR, Birge-scaled LR, AIC). A hypothesis is rejected when its Birge-scaled LR
  against the best is < 0.05. The preset keeps the g26 nominal hypothesis of each uncertainty entry unless rejected;
  measurement hypotheses (an instrument floor, the droop transfer) take the best, the first listed among ties
  (delta chi2 < 0.01, review). No alternative is deleted.
- **Half-prior rule (07 4.3)**: `calibrated` only when the hull of the 95 % intervals over all non-rejected
  hypotheses in which the leaf is free (within the leaf's own world, below) is <= half the prior width (log
  coordinates for a log prior; the leaf's declared width when the fit prior widens it, review), closed on both sides
  (not cut by the prior), one connected interval, the prior is a declared one (the leaf's `range` or an uncertainty
  interval, not fit bounds invented for a leaf that declares none), and the fit meets the tolerance of every target
  of the group (a miss is a model-form failure, not a calibration). Otherwise the leaf keeps its provenance and the
  narrowed interval goes in its note; its value is still set to the nominal-hypothesis fit when that fit meets the
  target's tolerance (so the preset reproduces its calibration data where it can), and is NOT changed when the fit
  misses the tolerance (CAL-9, CAL-14).
- **Worlds**: a leaf that only one hypothesis frees (an extinction-mechanism coefficient, the over-coupled branch's
  alpha_other) is written on that hypothesis' alternative, from a conditional fit with the other leaves at the
  nominal fit, i.e. the world the preset actually composes; its provenance follows the rule on the UNCONDITIONAL
  profiles. Likewise (review) the heater leaves are refitted in the thermo-optic world H86 and written as the
  alternatives `cal_h86`, which g26_calibrated's restated `uncertainty.thermo_optic` selects in that world; the leaf
  value and its tag are the H_solver world's.
- **Contamination**: no prior or start value was taken from a research fit that used held-out data (03 key finding 5
  slope, 06 4.3 / V5 Q, 06 M6 swing range). The dispersion's 0-4 %/nm are fit bounds without a declared range, so the
  dispersion is never tagged `calibrated`. Starts are preset values; priors are the leaves' own ranges (listed per
  parameter in `latest_fit.json` and in each leaf's `identifiable.prior`).

## 4. Results (record `mrm/calibration/latest_fit.json`, run 2026-10-16, commit b92c519 plus the review's changes)

Tables: `mrm.calibration.report.summary_markdown(load_results())`. chi2 at the best fit, LR = exp(-dchi2/2) against the
best hypothesis, "Birge" = the same with the best hypothesis' chi2/dof scaling; width = 95 % hull / prior (declared)
width.

### 4.1 CAL-3a/b + CAL-4: heater (calibrated, per thermo-optic world)

| hypothesis (thermo-optic / droop transfer) | chi2 (16 data, CAL-4 correlated) | LR (= Birge) | P_FSR mW | R0 ohm | heater_self_ratio | y /mW |
|---|---|---|---|---|---|---|
| H86 / same_rth_heater | 0.46 | 1 | 25.511 | 46.46 | 1.242 | 0.0082 |
| H86 / same_ratio | 0.46 | 1 | 25.511 | 45.25 | 1.436 | 0.0094 |
| **H_solver / same_rth_heater** (nominal world, written) | 3.27 | 0.25 | 25.506 | 41.29 | 1.796 | 0.0141 |
| H_solver / same_ratio | 3.27 | 0.25 | 25.506 | 39.79 | 2.051 | 0.0161 |

- Written (H_solver world, all `calibrated`): P_FSR 25.506 mW (95 % 25.41-25.60, 0.02 of the prior; the R2 anchor
  with the FSR at the line, +0.22 % over the printed bench median), R0 41.29 ohm (38.76-42.68; 0.24 of the declared
  [42, 58] width, which the fitted R0 lies below; fit prior [35, 58], restated in the preset), heater_self_ratio
  1.796 (1.59-2.19, 0.20): y = TCR x R_th,H = 0.0141 /mW (0.0161 under same_ratio; TCR 1.4e-3 /K held, R_th,H
  10.1 K/mW, Z_heater->ring 5.61 K/mW). Hot R at P_FSR 56.14 ohm (CAL-3b). Swept device 0.349 nm/mW (06 E4: 0.348),
  P_FSR,dev 29.1 mW, median 0.399 nm/mW. CAL-4 max residual 0.036 nm (tolerance 0.12).
- H86 world (alternatives `cal_h86`, selected by `uncertainty.thermo_optic=H86`, all `calibrated`): P_FSR 25.511
  (25.41-25.61), R0 46.46 (44.16-47.73, inside the declared range), ratio 1.242 (1.08-1.57); y 0.0082-0.0094 /mW,
  which brackets the textbook's 0.00904 (a fit without thermo-optic curvature); CAL-4 max residual 0.011 nm.
- **Hypothesis preference**: CAL-4 is fitted better without a thermo-optic curvature (H86's constant dlambda/dT)
  than with H_solver's rising s(T): LR 0.25 for H_solver, weak evidence, not a rejection. (Under the fsr_integral
  anchor the magnitude of dlambda/dT is absorbed by P_FSR, so this compares only the curvature of the droop.) The
  droop transfer (same_rth_heater vs same_ratio) is a reparametrization of the swept device: tied to 1e-5 in chi2;
  the hull covers both. The swept device's R0 is not informed by CAL-4 (fitted R0_dev / R0 = 1.000, its prior
  alone); with the correlated CAL-4 errors it widens the intervals (R0 under same_rth_heater 1.9 -> 2.8 ohm, the R0
  hull over both transfers 3.5 -> 3.9 ohm).
- Why y changed: the textbook's 0.00904 /mW (and g26's R0 45.41, ratio 1.158) assume the droop is TCR droop alone.
  With s(T) rising ~14 %/100 K (H_solver, order 2 under fsr_integral) the same 19 % droop of dlambda/dV^2 needs ~55 %
  more TCR droop (y 0.0141 against 0.00904 /mW). This is why the H86 world keeps a g26-like R0 and the H_solver
  world does not; without per-world leaves the H86 world of the committed preset missed CAL-3b and CAL-4 (finding 2).

### 4.2 CAL-1 + CAL-2: Q_L, coupling branch, extinction mechanism (07 P1-5)

| hypothesis (branch at 0 V / mechanism / floor) | chi2 (6 data + the CAL-1 constraint) | LR | LR Birge | verdict |
|---|---|---|---|---|
| over / ring_t_loss / floor | 7.51 | 1 | 1 | best |
| over / heater_t_loss / floor | 7.51 | 1 | 1 | same (the coefficients differ by the heater/ring rise ratio) |
| **under / dispersion / floor** (g26 nominal, kept) | 10.52 | 0.22 | 0.67 | not rejected |
| over / dispersion / floor | 20.81 | 0.0013 | 0.17 | not rejected; at the branch wall (x0 0.999); misses CAL-2 (1.60 dB) |
| under / ring_t_loss or heater_t_loss / floor | 23.11 | 4e-4 | 0.125 | not rejected; at the branch wall (x0 1.001); misses CAL-2 (1.72 dB) |
| any hypothesis without the floor | 49.8 ... 342 | < 1e-9 | < 0.004 | rejected |

Birge s^2 of the best = 7.51 / 2 dof = 3.75 (the CAL-1 interval is a constraint, not a datum). The heater-first
order lowered every chi2 (nominal 12.45 -> 10.52, best 9.13 -> 7.51).

Findings:
- **No hypothesis fits CAL-2 without an instrument floor**, and the reason is the shape: read as a crossing of
  critical coupling between 0.2 and 0.5 V (which the sampled voltages would not show), ln x of the coupling ratio would
  have to change by ~0.05-0.09 per nm of tuning over the first 1.8 nm and by ~0.02 per nm after, while each mechanism
  of the model moves ln x at a nearly constant rate per nm (dispersion) or per K (loss). With a floor of 29.4-29.9 dB
  (fitted, a measurement nuisance, not written) the 0.2 and 0.5 V points carry information on the floor only, and
  every mechanism fits (the remaining misfit, Birge s^2 3.8-5.3, is spread over the points, |z| <= 1.9). The floor
  was in the measurement model before the first fit (builder; the single commit cannot show the order) and is
  justified by CAL-2's own verification ("near 30 dB a notch depth is also sensitive to the instrument floor"); CAL-2
  alone cannot prove that a floor, and not an unmodelled mechanism, is the cause. A ~30 dB floor is plausible for a
  swept-laser measurement (source spontaneous emission, polarization leakage) but unverified.
- **What the floor implies about the device**: the fitted worlds pass through critical coupling during the heater
  sweep. Without the floor, the nominal world's notch is 36 dB at 0 V, ~75 dB at 0.4 V (critical coupling), 41 dB at
  0.5 V and 21 dB at 1.1 V (the operating line, ~7.6 nm of tuning); the ring_t_loss world is the mirror (35, 71, 41,
  21 dB). The replica itself has no floor, so g26_calibrated fails CAL-2 by 11.4 dB under the replica's own protocol.
  The branch labels (`coupling_regime`, "under" / "over") name the cold 0 V state: the nominal (dispersion) world is
  OVER-coupled at the operating line and the loss worlds are under-coupled there. Drop current, OMA side and the
  ER drift under lock (07 P1-5) differ accordingly, and the through-port data cannot tell them apart.
- **The coupling branch is tied to the mechanism, and not identified.** Dispersion (x rising) fits only from the
  under-coupled side at 0 V, a loss mechanism (x falling) only from the over-coupled side; the mirror fits sit on the
  branch wall x0 = 1 and miss the CAL-2 tolerance. The pairs differ by LR 0.22 (raw) / 0.67 (Birge): the data do not
  decide. The preset keeps g26's nominal (under / dispersion) and writes the loss worlds with their own branch.
- **The dispersion needed is 2.22 %/nm** (95 % 1.53-2.83 %/nm in the nominal hypothesis, 0.88-2.83 %/nm over the two
  non-rejected dispersion hypotheses) against the rib160 coupler artefact's 0.77 %/nm and femwell's 1.3-1.5 %/nm
  (05 t02). The leaf declares no range, so it is written `assumed` (the solver value is the alternative `solver`).
  Under the loss hypotheses the solver dispersion stands and the loss rises by d ln(alpha)/dT = 2.30e-3 /K (ring
  temperature; 95 % 1.22e-3 to 2.66e-3 over both branches, `calibrated`, 0.29 of the prior) or 1.28e-3 /K (heater
  temperature; 0.68e-3 to 1.48e-3, `calibrated`, 0.16). These coefficients are per K of the H_solver world's ring or
  heater temperature; in the H86 world the same CAL-2 would need about 66/86 of them (not refitted, open issue).
- **Q_L is not narrowed beyond CAL-1** (profile 3469-4200, flat inside the printed interval; it trades against
  alpha_other along the CAL-2 valley, and the interval's top coincides with the leaf range's top, so the rule counts
  it open). Q_L stays 3850 (`inferred`); alpha_other moves 13.35 -> 10.72 dB/cm (`assumed`, 95 % 0-24 dB/cm), which
  puts the cold 0 V state at x0 = 0.969 (Q_i 7687) with a 29.67 dB floor.

Composed worlds written (conditional fits, Q_L held at 3850; provenance from the unconditional profiles):

| world (preset selection) | free leaves (alternative, provenance) | chi2 | CAL-2 tolerance |
|---|---|---|---|
| nominal: coupling_regime under, er_mechanism dispersion | alpha_other 10.72 (assumed), dispersion 0.02225 /nm (leaf value, assumed; solver value kept as alternative `solver`) | 10.52 | pass (0.93 dB) |
| er_mechanism=ring_t_loss | alpha_other 7.07 (`cal_ring_t_loss_branch`, assumed: unconditional 0-19.7, open; 6.0-8.5 conditional), d ln a/dT_ring 2.297e-3 (`ring_t_loss`, calibrated), solver dispersion | 7.51 | pass (0.76 dB) |
| er_mechanism=heater_t_loss | alpha_other 7.07 (`cal_heater_t_loss_branch`, assumed), d ln a/dT_heater 1.279e-3 (`heater_t_loss`, calibrated) | 7.51 | pass (0.76 dB) |
| coupling_regime=over (with dispersion) | alpha_other 9.27 (`over`, assumed) | 49.2 | **fail (1.78 dB)**: a model-form failure, kept as a poor world and flagged in the restated `uncertainty.coupling_regime` note |
| ring/heater_t_loss on the under branch (not composed, reported) | - | 42.4 | fail (2.53 dB) |

`er_mechanism` sorts after `coupling_regime`, and realize applies the entries in name order, so a loss world always
takes its own (over-coupled) branch. `Config.get` with explicit `--alt` pairs follows the order the pairs are given
in (it can print the coupling_regime value), while the realized plant follows name order (checked): a
params-machinery quirk, request below.

### 4.3 CAL-7 (calibrated) and CAL-5 (not calibrated)

- resonance_offset_nm 7.600 (95 % 7.40-7.80): R3 closes it by construction; tagged calibrated, nothing changes.
- eff_at_0v_pm_per_v: **93.6 pm/V at +4 dBm** (the replica's primary, 07 P1-7: relief removed at the measurement
  power; profile 85.8 to the prior top 95, open) against **72.5 pm/V at P_in -> 0** (65.7-79.3). LR 0.73 (Birge
  0.75): the data cannot tell the power. Kept `inferred`, value set to the primary fit (it meets the tolerance), the
  P_in -> 0 value is the alternative `cal5_p_in_0`. The +4 dBm fit needs an efficiency at the edge of the prior: the
  self-heating relief at +4 dBm is ~23 % of the shift in this plant. The leaf's provenance stays the one of g26's
  72.5 (which is the P_in -> 0 reading of the same ladder); the written 93.6 rests on the assumed measurement power.
- fill_fraction (LUT mode, reported only; g26 uses the analytic law where the fill cancels): 0.882 at P_in -> 0
  (0.80 to the prior top 0.95); at +4 dBm the fit hits 0.95 (chi2 19.6, fails). If the integrator switches g26 to
  the junction LUT, the G25 power question decides whether rib160 + L-shape reaches CAL-5.

### 4.4 CAL-10, CAL-8, CAL-12b

- **CAL-10**: only the R C is identified. With C_j(0) held at 15 fF, R_s = 390 ohm, the top of its range (a boundary
  solution: 32.2 GHz, inside the 4 GHz tolerance; an exact fit needs R_s above its range at 15 fF, or e.g. R_s 288
  ohm with C_j(0) 20.9 fF, hypothesis both_free). With R_s held at 100 ohm no C_j(0) <= 30 fF reaches it (LR 1e-4,
  i.e. rejected by the C_j prior bound). Written: R_s 390 ohm, `assumed` (open profile). RC pole 35.0 GHz (50 ohm
  driver, C_j(1 V) 10.3 fF) against the 64.0 GHz optical FWHM at 1 V. A larger R_s raises the junction share of the
  CV^2 data heat (R_s/(R_s+R_drv) 0.67 -> 0.89).
- **CAL-8**: gain_scale 0.824 under the half-maximum lock (the replica's primary; 0.77-0.88), 0.532 under an oracle
  lock (0.50-0.57, at the prior bound), a peak lock is rejected within the gain prior (it needs < 0.5). Only gain x
  line power x lock factor is identified (07 D7): kept `assumed`, value = half-max fit, the oracle value as
  alternative `cal8_oracle_lock`.
- **CAL-12b**: laser.line_offset_ghz -2.17 GHz (95 % -3.30 to -1.03, `calibrated`), die/line sigma x0.72 (2.89 / 4.33
  GHz, 95 % x0.44-1.01): the Guide population's median and quartiles exactly. The sigma scale is written but stays
  `inferred`: the variation entry declares no range (fit bounds [0.25, 2] only).

### 4.5 CAL-14: the lumped modulation depth (not closed, not written)

Oracle relOMA at 1.0-1.15 V, 0-105 degC, 0 dBm (the replica's primary): -6.07 dB (g26) -> **-5.24 dB**
(g26_calibrated: the CAL-5 at-power efficiency). Inside the swing prior [0.9, 2.6] V the best is **-2.53 dB at 2.6
Vpp** (target -2.1 +- 0.3: misses by 0.13 dB beyond the tolerance). The scan (`extra.scan`; centred levels about
1.075 V; the junction grid clips at -0.5 V):

| what closes -2.1 dB | value (the -2.4 dB tolerance edge) | lumped depth D = swing_pm/FWHM_pm | plausibility |
|---|---|---|---|
| swing at the calibrated efficiency (93.6 pm/V) | **2.96 Vpp** (2.71 Vpp), levels -0.41 .. +2.56 V | 0.60 | above the prior (2.6) and 0.4 V into forward bias; a stacked or differential CMOS driver; no evidence either way |
| swing at the P_in -> 0 efficiency (72.5 pm/V) | 4.97 Vpp (3.85 Vpp), levels clipped at -0.5 V | 0.60 | implausible (> 2x a 1.8 V IO rail, forward-bias clip) |
| efficiency at 1.3 Vpp | x2.37 (222 pm/V at 0 V) | 0.58 | implausible against G25's measured 52.8 pm/V average (DC) |
| one-sided levels | never (saturates at -3.42 dB) | - | - |

Needed D ~0.60 (~205 pm peak-to-peak at FWHM 342 pm), in line with 06 M6 (227 pm at Q 3500, from the -2.1 dB level
alone). Verdict: with the CAL-5 at-power efficiency the printed level is just out of reach of a plausible drive
(within tolerance 4 % above the swing prior); with the P_in -> 0 efficiency it is far out of reach. It is a
**drive/definition gap, not closable within the stated priors** (07 D9, Q4; 06 R1: "either the reference plane or
the G25 swing and linewidth do not describe G26"). The alternative reading "OMA in dBm at the actual ring power"
gives -5.24 + 4.0 = **-1.24 dBm** at the preset +4 dBm (relOMA does not depend on the ring power in the oracle map),
i.e. it overshoots by 0.9 dB and would match at ~+3.1 dBm at the ring, a plausible power: that reading closes the gap
with the G25 drive, which is why 07 Q4 matters. driver.swing_pp_v stays 1.3 (printed G25); its note records this.
The setpoint-lock mode is worse (-4.17 dB at 2.6 V) and rejected (Birge LR 2e-5).

### 4.6 CAL-9: DAC LSB (not reached, not written)

Fit protocol (deviation from the replica's primary, stated): one die, uniform aggressor (every lane sees the victim's
rise: the slide's ~20 k swings on all lanes), 8 draws of the per-ring variation with the die sigma 0 (lane spread of
one die), the measured die's blue offset as a nuisance (prior [5, 10] nm), voltage law. Free: V_FS, per-ring sigma
scale.

| hypothesis | chi2 (6 data) | V_FS | ring sigma scale | die offset | LR Birge |
|---|---|---|---|---|---|
| H_solver / temperature referencing | 254 | 1.44 V (95 % 1.40-1.54, open) | 0.32 (0.20-0.42) | 10.0 nm (prior top) | 1 |
| H_solver / resonance referencing (CAL-13a default) | 597 | 1.44 | 0.27 | 10.0 | 0.13 |
| H86 / either | 1.3e4-1.5e4 | - | - | - | < 1e-30 |

Best: swing [21.4 k, 25.0 k] vs [19.7 k, 22.3 k]; lane offsets 5.7 k / 9.1 k vs 7.5 k / 9.5 k; range [17.9 k, 49.1 k]
vs [19.4 k, 49.25 k]: 7-9 sigma misses, outside every tolerance. **Every hypothesis fails, so the likelihood ratios
compare failing models and are not evidence for any of them.** What the fits show (reported, nothing written):
- Under resonance referencing (85.86 pm/K x 88.5 K = 7.6 nm, equal to the blue offset) the heater has almost no
  power left at the hot cusp unless the die is ~10 nm blue (the prior edge); the slide shows ~19 k codes there.
  Temperature referencing fits better (LR Birge 0.13, not a rejection).
- Under H86 many rings run out of heater at the hot cusp (code 0; 187-568 infeasible samples), the fits are
  multimodal and under-explored (80 evaluations, 2 starts), and the target itself says the code range cannot confirm
  dlambda/dT (06 E2): the H86 chi2 is a statement about this protocol with the blue-offset prior <= 10 nm, not
  evidence against H86 (the heater group, 4.1, mildly prefers H86's constant dlambda/dT).
- The per-ring geometry sigma that gives the lane offsets is ~0.3x the Berkeley local spread (2.0 / 1.3 nm).
- The replica's primary on g26_calibrated still gives swing [3.9 k, 23.3 k] (aggressor 30 um from ring 0).

### 4.7 Not fitted, reported

CAL-6 (H86 value): H_solver apparent shift 67.4 pm/K on g26_calibrated (66.6 on g26). CAL-11, 12a, 12d, 13a, 13b
pass unchanged. CAL-12c fails by 1.3 % unchanged (the scenario's laser_drift converts 0.02 nm at 1310 nm for every
line; scenarios owner, already requested by the replicas handoff).

## 5. Before / after (the replicas' own protocols; G25 targets on g25_test_vehicle_calibrated)

Integrator: the table was rerun on the repaired record (section 11.1): the G25 rows now run on the preset
g25_test_vehicle_calibrated (values rounded to 6 digits), and the record's context had carried a stale
eff_at_0v_pm_per_v of 93.86 pm/V (fitted 93.64). Changed: CAL-5 6.5 -> 6.6 pm, CAL-3a 4e-6 -> 5e-5 mW, CAL-3b 7e-7 ->
1.4e-6 V, CAL-7 7e-7 -> 5e-6 nm (rounding); every verdict is unchanged.

| id | before | max |res| | after | max |res| | note |
|---|---|---|---|---|---|
| CAL-1 | pass | 0 | pass | 0 | |
| CAL-2 | fail | 41.6 dB | fail | 11.4 dB | the replica has no floor; with the fitted 29.67 dB floor: pass (0.93 dB, test) |
| CAL-3a | pass | 0.056 mW | pass | 5e-5 | |
| CAL-3b | pass | 0.0033 V | pass | 1.4e-6 | the H86 world passes too (test) |
| CAL-4 | fail | 2.21 nm | fail | 1.89 nm | the replica runs the median device; the swept device (P_FSR 29.1 mW): pass, in both worlds (test) |
| CAL-5 | fail | 15.6 pm | pass | 6.6 pm | primary +4 dBm |
| CAL-6 | not scored | | not scored | | 66.6 -> 67.4 pm/K |
| CAL-7 | pass | 5e-6 | pass | 5e-6 | |
| CAL-8 | fail | 3.08 uA | pass | 0.002 uA | |
| CAL-9a/b/c | fail | 16 k / 29 k / 16 k | fail | 16 k / 28 k / 17 k | replica primary geometry; see 4.6 |
| CAL-10 | fail | 13.0 GHz | pass | 1.3 GHz | |
| CAL-11, 12a, 12d, 13a, 13b | pass | | pass | | unchanged |
| CAL-12b | fail | 3.56 GHz | pass | 0.026 GHz | |
| CAL-12c | fail | 0.00026 nm | fail | 0.00026 nm | scenario conversion |
| CAL-14 | fail | 3.97 dB | fail | 3.14 dB | -6.07 -> -5.24 dB; see 4.5 |

Leaves tagged `calibrated` in g26_calibrated: ring.heater.p_fsr_mw, ring.heater.r_cold_ohm,
ring.thermal.heater_self_ratio (each also on its H86 alternative `cal_h86`), ring.optics.resonance_offset_nm,
laser.line_offset_ghz, and the alternatives `ring_t_loss` (dln_alpha_dt_ring) and `heater_t_loss`
(dln_alpha_dt_heater), each conditional on its mechanism and on the CAL-2 instrument floor. Values changed but not
calibrated (provenance kept or `assumed`): alpha_other_abs and its alternatives `over`, `cal_ring_t_loss_branch`,
`cal_heater_t_loss_branch`, the dispersion leaf (assumed; was solver), eff_at_0v (inferred), series_resistance
(assumed), monitor gain_scale (assumed), the variation sigma of laser.line_offset_ghz (inferred). Unchanged with a
note: q_loaded, fill_fraction, swing_pp_v, full_scale_v. The preset restates `uncertainty.er_mechanism`,
`uncertainty.coupling_regime` (note only) and `uncertainty.thermo_optic` (H86 selects `cal_h86`).

## 6. Closed loop on g26_calibrated

`tests/calibration/a1_g26_calibrated.yaml` (A1 settings on g26_calibrated) runs without error (~2 s):

| A1 criterion | g26 | g26_calibrated |
|---|---|---|
| A1-lock, max lock error (<= 5 pm) | 348.7 pm FAIL | 380.5 pm FAIL |
| A1-line, wrong-line rings | 0 PASS | 0 PASS |
| A1-hop, FSR hops | 0 PASS | 0 PASS |
| A1-ber (<= 1e-9) | 1.3e-8 FAIL | 7.3e-10 PASS |
| A1-monitor (BND-3a band) | -69.2 / +2.9 % FAIL | -67.9 / +2.8 % FAIL |
| relOMA mean / oracle | -6.18 / -6.04 dB | -5.40 / -5.22 dB |

The lock failure is ring 0 near the hot cusp (the heater-authority problem of the 30 um aggressor site, scenarios
handoff), not a calibration effect. The H86 world (`alts: {uncertainty.thermo_optic: H86}`, scratch run, not a test)
also runs without error and fails A1 with 2 wrong-line rings and BER 0.24, as the H86 world of g26 does (1227 pm,
2 wrong-line rings, BER 0.25): with 86 pm/K the 88.5 K aggressor uses the whole 7.6 nm blue offset (BND-2).
g26_calibrated also passes tests/closure/test_presets.py (load, relations, every hypothesis realizes, Monte Carlo
batch).

## 7. Deviations (please review)

1. **Instrument floor on CAL-2** (measurement nuisance, not in the replica): decided before the first fit from the two
   equal 29.8 dB readings and the targets' floor warning; reported with and without (without: every hypothesis
   rejected).
2. **CAL-9 fit protocol** differs from the replica's primary (uniform aggressor, one die, die-offset nuisance): the
   primary geometry cannot reach the lower code end with any DAC parameter. Nothing was written from it.
3. **Q_L written at the CAL-1 centre (3850)**: the free best fit sits at an arbitrary point of a flat valley; the
   centred refit is within dchi2 0.01. Profiles are of the free fit.
4. **Rule interpretation**: "calibrated" also requires a closed interval, one component, a declared prior (the
   declared width when widened), a fit within every target's tolerance, and, for a composed world, the
   unconditional profile. q_loaded fails only on "closed" (its upper edge is the CAL-1 edge = the leaf range's top).
5. **Values of non-calibrated leaves are moved to the nominal fit** when it meets the tolerance (alpha_other, eff,
   R_s, gain_scale), not left at the prior value, so the preset reproduces its calibration data; provenance not
   upgraded. R_s sits at the top of its range (a boundary solution).
6. **R0 prior widened** to [35, 58] ohm (restated range in the preset; the rule uses the declared width).
7. **CAL-4 likelihood** uses the shared 0 V read-off error (whitened), and the swept device's R0 is a nuisance with
   the preset's own variation prior: both are likelihood choices of this task, not target changes (the targets'
   sigmas are used as printed).
8. The CAL-5 power, CAL-8 set-point, CAL-10 R/C split, the droop transfer and the thermo-optic world are hypotheses
   carried as alternatives, not resolved.

## 8. Open issues and requests

- **integrator**: add `"mrm.calibration.cli:register_cli"` to `mrm.cli.PROVIDERS`; add the `calibration` marker to
  pyproject (tests/calibration/conftest.py registers it locally and skips the tests unless `-m calibration`);
  CHANGELOG line for g26_calibrated (rerun at the review; heater per thermo-optic world). **Regenerate
  docs/VALIDATION.md** (`python -m mrm.validation.report --fast --preset g26`, and the g26_calibrated report if it is
  kept): `tests/verification/test_report.py::test_validation_md_is_current` (CI-blocking) fails on any fit record,
  because the committed VALIDATION.md was rendered before latest_fit.json existed (its section 3.1 says "No
  calibration output ... was found"); this already failed at b92c519 and is not a calibration file.
- **report owner**: `test_report.py::test_calibrated_leaf_audit_flags_contamination` fails with any fit record present
  (also at b92c519): it takes the first table row that starts with "| `ring.optics.q_loaded`", which is the row of
  section 3.1 (the fit record's decisions), not the section 3.2 audit row of the test preset. Search after the
  "### 3.2" header. Run the held-out targets on g26_calibrated (this task did not look at them); flag that CAL-14
  and CAL-9 are unreached calibration targets, that the CAL-2 fit needs a floor, and that the nominal world is near
  critical coupling at 0-0.6 V heater.
- **params_machinery**: tied hypotheses (branch x mechanism; Q_L with alpha_other) - the data say "loss mechanism <=>
  over-coupled branch at 0 V"; the name-order trick in g26_calibrated is a workaround, and `Config.get` with explicit
  `--alt` pairs follows the given order while realize follows name order. [Branch x mechanism resolved 2026-10-16
  (params review, params-14): a label entry (`follows` + `map`, PARAMETERS_FORMAT 6.1); the generator writes
  g26_calibrated's coupling_regime as the label of er_mechanism, so the label is the realized cold 0 V branch and a
  pinned regime selects its mechanisms; section 12.] An absolute R_th,H key (or a tied ratio)
  so a fitted y holds across worlds; until then the heater leaves carry per-world alternatives (H86 here; the weight-0
  thermo-optic alternatives bench / imec / ocp / textbook_derived keep the H_solver heater leaves and would miss
  CAL-3b / CAL-4 by similar amounts).
- **params_content / extraction (coupling)**: the dispersion world needs 2.2 %/nm against the rib160 artefact's
  0.77 %/nm; re-check the coupler solve (curved gap, rib slab) or accept that the loss mechanisms are the physical
  ones. The R0 leaf range [42, 58] should follow the droop of the thermo-optic world (H_solver 38.8-42.7 ohm).
  [R0 resolved 2026-10-16 (params review, params-13): g26's range is [37, 58], and the heater group takes the
  declared width from the leaf; section 12.]
- **scenarios / layout**: CAL-9 fits better with temperature referencing of the slide-10 degC and a uniform aggressor;
  under the resonance default the heater runs out at the hot cusp (ring 0 at code 0 in A1 as well). The aggressor site
  30 um from ring 0 is the first suspect (replicas handoff 4.2). CAL-12c conversion per line.
- **junction**: decide LUT vs analytic for g26 (4.3); at +4 dBm the rib160 LUT needs fill > 0.95.
- **calibration (open)**: (a) the extinction coefficients, alpha_other and the dispersion are fitted in the H_solver
  world only; the H86 world uses them as they are (the loss coefficients are per K: ~x66/86 there). (b) CAL-3a and
  CAL-3b are medians of two maps, not one die (their 2 % tolerance says so) but their sigmas (0.2 %, 0.4 %) pin
  R_hot to +-0.5 ohm; the R0 interval does not include that map-to-map uncertainty. (c) CAL-1's flat likelihood and
  the Birge widening with 2 dof (coupling) make the coupling intervals coarse statistics: chi2/dof from 2 dof is
  itself uncertain by a factor ~2. (d) A run of `--groups X` keeps the other groups' records; a later group not rerun
  keeps its old fit (rerun it when an earlier group's leaves change).
- **Lightmatter questions** sharpened: G26 junction swing and drive topology (07 Q4: 2.96 Vpp would close CAL-14 at
  the at-power efficiency, or the OMA-in-dBm reading at ~+3 dBm with the G25 drive); the G25 bias-ladder optical
  power (CAL-5: 72.5 vs 93.6 pm/V); the G25 notch-sweep instrument floor (CAL-2); the slide-10 thermometer.

## 9. Tests and runtime

- `pytest -m calibration tests/calibration`: 29 passed (11 machinery tests on known answers: linear recovery and the
  analytic 95 % interval, open profiles, Birge widening, flat interval targets, set/curve sigmas, log priors, LRs,
  restrict, the CAL-4 whitening against d' C^-1 d, the declared-width rule; 18 preset tests: load/realize, every
  written leaf carries targets/residual/identifiable and the rule, no held-out or fit:false id anywhere in the
  record, no `calibrated` tag without tolerance or declared prior, loss worlds hold the solver dispersion, preset ==
  record, CAL-1/3a/3b/5/7/8/10/12b reproduced by the replicas, CAL-4 by the swept device, CAL-2 with the floor, the
  H86 world selects its heater leaves and reproduces CAL-3a/3b/4, A1 closed loop runs). Without `-m calibration`
  they are skipped (not CI-blocking).
- tests/closure/test_presets.py + test_targets_schema.py + tests/verification/test_validation_isolation.py: 205 passed,
  2 skipped (g26_calibrated included). tests/verification/test_kernel_parity.py: 18 passed (no engine or njit code
  touched). `python -m mrm.validation.targets_io check`: targets OK. tests/verification/test_report.py: 19 passed, 2
  failed, both failing at b92c519 as well (section 8).
- Runtime: heater ~1.4 h (4 hypotheses x 3 profiles on a loaded machine), coupling_er ~2.7 h, the later groups ~1 h
  (CAL-9 ~30 min); a serial full run is ~5 h, peak RSS 0.8 GB.

## 10. Adversarial review (2026-10-16): findings and fixes

The checkpoint b92c519 held a fit record and preset generated at 2026-10-11T01:54Z, BEFORE the previous reviewer's
code changes (that reviewer was cut off during its reruns), while this handoff already described the corrected
rules: the committed preset still tagged the dispersion leaf `calibrated` against invented 0-4 %/nm bounds with the
contaminated 07 P1-5 citation, the three alpha_other alternatives `calibrated` from conditional profiles (the `over`
world although it misses CAL-2), and the line-error sigma `calibrated` on invented bounds; two calibration tests
failed. Everything was rerun with the code as reviewed below, and the record, preset and this file regenerated.

| # | severity | finding | fixed |
|---|---|---|---|
| 1 | critical | record and preset stale against the code and this handoff (above); 2 of 23 calibration tests failing | yes: full rerun, preset regenerated |
| 2 | major | the H86 world of g26_calibrated (07 D2's required robustness corner) kept the H_solver heater leaves and missed CAL-3b (1.169 vs 1.195 V, tolerance 2 %) and CAL-4 (0.42 nm, tolerance 0.12 nm): the droop is TCR droop plus the dlambda/dT(T) curvature, so the heater leaves depend on the thermo-optic world | yes: the heater group is fitted per thermo-optic world; the H86 values are alternatives `cal_h86` that the restated `uncertainty.thermo_optic` H86 choice selects (test) |
| 3 | major | group order: coupling_er ran before the heater although CAL-2 reads the notch at heater voltages through the heater law (the previous review measured chi2 12.45 -> 10.62 on a second pass) | yes: heater first (the heater targets do not depend on the coupling leaves) |
| 4 | major | CAL-4's 14 shifts share the 0 V read-off error but were treated as independent (overstating the level information, understating the shape information) | yes: whitened residuals with the shared error (`CAL4_COMMON_SIGMA_NM`) |
| 5 | major | the swept device's cold resistance was silently set equal to the median's, although the voltage sweep identifies y / R0 of that device, so y, R0 and the ratio inherited the assumption | yes: nuisance R0_dev / R0 with a Gaussian log prior from the preset's own variation entry (sigma 0.061) |
| 6 | major | every non-nominal coupling hypothesis was profiled coarsely (5 points, 15 evaluations), and the `calibrated` tags of the extinction coefficients and the branch alpha_other come from those profiles alone | yes: full resolution for every leaf whose tag a profile can decide (Q_L of a non-nominal hypothesis stays coarse: it cannot change its tag). A spot check of the old coarse and full profiles at their 95 % edges with 6 starts and 200 evaluations found them converged to <= 0.22 in chi2 (thresholds 11-16 above the minimum) |
| 7 | minor | R0's prior widened from the declared [42, 58] to [35, 58]: justified by the leaf's own note (the range is R_hot / (1 + y P_FSR) with the superseded y), not hiding a data tension (CAL-3b is met to 1e-6); but the tag was judged on the widened width | yes: the half-prior rule now takes the declared width when narrower (`Param.declared_range`); R0 passes either way. The H86 world's R0 (46.5 ohm) lies inside the declared range: the range assumed the droop without the thermo-optic curvature |
| 8 | minor | the scratch-only heater_self_ratio range [1, 8] is a derived quantity of the swept device (ratio x P_FSR,dev / P_FSR under same_rth_heater), never a prior of a written leaf | no change needed (the swept device's R0 now needs a scratch range too, [25, 80] ohm) |
| 9 | minor | the nominal of tied hypotheses (the droop transfer reparametrizes the swept device, chi2 equal to 1e-5) was chosen by rounding noise | yes: `TIE_DCHI2`, the first listed among ties |
| 10 | minor | `--groups` with a non-contiguous subset (e.g. `heater laser`) fitted the later group without the leaves of the groups between | yes: before each rerun group, the previous record's leaves of every preceding group not rerun |
| 11 | minor | CAL-14's swing hull included the rejected setpoint_lock mode; the "4.95 Vpp at the P_in -> 0 efficiency" and the tolerance-edge swings of this file were not in the record; the closing text was empty whenever the one-sided convention closed | yes: hull over non-rejected modes; the scan adds the swing at the CAL-5 P_in -> 0 efficiency and the -2.4 dB edges; text fixed |
| 12 | minor | the restated `uncertainty.er_mechanism` entry lost its provenance, generation and source | yes |
| 13 | minor | the explanation of why CAL-2 needs a floor was wrong (a critical-coupling crossing between 0.2 and 0.5 V is simply not sampled); see 4.2 for the actual reason and for what the floor implies about the cold device | yes (text) |
| 14 | minor | hypothesis-preference wording on CAL-9 ("H86 strongly disfavoured", "the data prefer temperature referencing") overstated: every hypothesis misses CAL-9 by 7-9 sigma, the fits are multimodal, and the target itself says the code range cannot confirm dlambda/dT (06 E2) | yes (text, 4.6) |
| 15 | minor | `refresh_coupling_worlds` and the loss-world test assumed coupling_er had no earlier context | yes (`coupling_context`) |
| 16 | test gap | nothing checked that the record's residuals trace to calibration ids, that a world other than the nominal reproduces its calibration data, the covariance or the declared-width rule | yes: four tests added |
| 17 | minor | a record made with uncommitted code named only the commit | yes: `-dirty` suffix (this record: b92c519-dirty) |
| 18 | minor | g26's `coupling_regime` note ("the fit of CAL-1/CAL-2 returns both branches") is not true of g26_calibrated's over + dispersion world, which misses CAL-2 | yes: the entry is restated with the finding |
| 19 | minor | (in this review's own change, caught before hand-off) the heater fits' derived `thermo_optic` label was late-bound and read H86 for every hypothesis; the numbers used the right world | yes: code fixed, the record's labels corrected (noted in the record) |

Checked and found sound: no held-out id anywhere in the record (test); every prior is a leaf range, an uncertainty
interval or a stated assumption, none from a contaminated research fit; units and signs of every replica the fits use
(CAL-5/10 bias positive = reverse; CAL-7 line minus resonance; CAL-12b line minus grid, the same sign as
`laser.line_offset_ghz` in realize; alpha in 1/m = dB/cm x 23.03; the x0 and Q_i algebra of `CouplingConstants`);
the least_squares setup (bounded trf in normalized coordinates, several starts, distinct minima kept); the Birge
widening and its use for the rejection LR; the CAL-2 floor nuisance (decided before the fits per the builder, which
the single commit cannot show; justified by CAL-2's own verification note "near 30 dB a notch depth is also
sensitive to the instrument floor"; both variants are reported and every no-floor hypothesis is rejected).

## 11. Integration (calibration stage, 2026-10-16)

The integrator's record. The integrator record of the whole stage is docs/handoff/integrator.md section 9. Nothing was
committed. Order kept: every calibration change below was made, and both presets were final, BEFORE the held-out
targets were evaluated on g26_calibrated; nothing in the calibration was changed after that.

### 11.1 Fixes to the calibration outputs

1. **The record's `context` was stale** (found by the integrator). The review merged two parallel runs and kept the
   pre-review context values of three leaves: `resonance_offset_nm` 7.6000046 (fitted 7.6000052),
   `eff_at_0v_pm_per_v` **93.864** (fitted **93.643**) and `series_resistance_ohm` (equal to 1e-11). The context is
   what `bench_config` (the G25 bench of the before/after table and of the tests) and every `--groups` rerun read,
   so CAL-5's "after" row ran with 93.86 pm/V. The fits themselves were not affected (each process updated its own
   context as it ran; the modulation_depth group ran after junction with 93.64). Fixed: the context was rebuilt from
   the leaf decisions (`pipeline.context_from_decisions`, a note in the record), the before/after table was rerun
   (`--before-after-only`, 47 s), and a test now checks context = decisions. No fit and no preset value changed;
   g26_calibrated.yaml regenerates byte for byte from the record.
2. **The calibrated G25 bench as a preset**: `g25_test_vehicle_calibrated.yaml` (generated by
   `preset.write_bench_preset`, also by `--write-preset-only`): `g25_test_vehicle` plus the device leaves and the three
   uncertainty entries of g26_calibrated, restated exactly. Unlike the overrides of `bench_config`, it keeps the
   alternatives, so the H86 world and the named corners select the calibrated alternatives on the bench too (test).
   It is the report's G25 companion of g26_calibrated (the report builder asked for this name), and the before/after
   table's G25 "after" plant. Its values are rounded to 6 digits like g26_calibrated: CAL-5 at +4 dBm reads the dragged
   minimum on the replica's 0.1 pm fine-sweep grid, so the rounding of P_FSR moves the 1.6 V point by one grid step
   (0.1 pm = 0.02 sigma; test tolerance stated).
3. Before/after after the fixes: CAL-5 6.5 -> 6.6 pm (the stale 93.86 pm/V had given 6.55), CAL-3a 4e-6 -> 5e-5 mW,
   CAL-3b 7e-7 -> 1.4e-6 V, CAL-7 7e-7 -> 5e-6 nm (rounding). Every verdict is unchanged (section 5 updated).
4. Requests of section 8 resolved: `mrm calibrate` is registered in `mrm.cli.PROVIDERS`; the `calibration` marker is
   in pyproject.toml (conftest no longer registers it; it still skips the tests unless `-m calibration`); CHANGELOG
   entry; docs/VALIDATION.md regenerated; the report test's row selection fixed (report owner's item).
5. Tests: `pytest -m calibration tests/calibration`: 39 passed (29 + context = decisions, the bench preset equals the
   record, g26_calibrated and bench_config (H86 world included), and 8 parametrized bench-preset-vs-bench_config
   agreements).

### 11.2 The held-out targets on g26_calibrated (`docs/VALIDATION_g26_calibrated.md`)

Computed once, after 11.1, with the replicas' own protocols (G26 targets on g26_calibrated, G25 targets on
g25_test_vehicle_calibrated). Cost: 2856 s wall in one process under `memguard --heavy` (4 kernel threads, machine
load 5-7 from other work): predictions 420 s (0.66 GB), identifiability 696 s (0.67 GB), sensitivity 910 s
(1.0 GB), ranking 825 s (1.0 GB); peak 1.07 GB. Every calibration-row prediction equals the record's before/after
values to 1e-12 (both presets), and section 3.1 is rendered from latest_fit.json itself.

| role | g26: pass / fail / blocked / not scored | g26_calibrated |
|---|---|---|
| calibration (fit residuals on g26_calibrated) | 9 / 11 / 0 / 1 | 13 / 7 / 0 / 1 |
| validation (held out) | 2 / 12 / 1 / 3 | **5 / 9 / 1 / 3** |
| bounds (held out) | 3 / 0 / 2 / 0 | 3 / 0 / 2 / 0 |

Held-out rows that changed (measured; g26 -> g26_calibrated, z):

| id | measured | g26 | g26_calibrated | note |
|---|---|---|---|---|
| VAL-2a | 18.0 dB at 1.4 V | 36.9 dB, z 37.8, fail | 16.98 dB, z 2.05, **pass** | notch depth above the CAL-2 voltages: the coupling world (dispersion 2.2 %/nm, alpha_other 10.7) extrapolates |
| VAL-2b | 16.9 dB at 1.5 V | 33.3 dB, z 32.8, fail | 15.86 dB, z 2.08, **pass** | same |
| VAL-3a | 35.4 GHz at 2 V | 45.6 GHz, z 14.5, fail | 36.6 GHz, z 1.76, **pass** | the CAL-10 R C (R_s 390 ohm, the RC including the 50 ohm driver) extrapolates to 2 V |
| VAL-1 | 13.0 nm at 1.5 V | 15.63 nm, z 52.7, fail | 15.15 nm, z 43.0, fail | the replica runs the median device; the target is the swept device (as CAL-4). On the frozen fit's swept device (P_FSR,dev 29.1 mW): **13.05 nm, z 1.0** (H86 world 12.98 nm, z 0.45). Computed by the integrator after the primary row was seen; reported, not scored |
| VAL-5 | contour offsets | z 12.9, fail | z 15.2, fail | worse; both far off (the relOMA map family of CAL-14) |
| VAL-4c | 0.9 dB | 3.23, z 23.3 | 3.19, z 22.9 | both fail |
| BND-1 | > 2 FWHM | 2.79 FWHM | 2.38 FWHM | both pass |

Unchanged: VAL-3b pass (weak, flagged), VAL-8d pass, VAL-4a/b fail (no finite prediction: the -3 dB contour is not
reached, the relOMA level gap of CAL-14), VAL-7b fail (definition conflict), VAL-8a/b/c fail (thermal artefact, not
touched by the calibration), VAL-7c and BND-3a/b blocked. **No held-out target went from pass to fail.**

Identifiability at the calibrated point (report section 3): rank 17 of 18, 14 of 18 directions pass. The joint
linearized screen fails `heater_self_ratio` (0.97) and `tcr_per_k` (0.57): only y = TCR x R_th,H is identified, so
the ratio's `calibrated` tag holds for the literature TCR 1.4e-3 /K (its note says so). R_s and C_j fail (58.7 /
31.8), gain_scale 4.4, chip_input_loss 4.4, responsivity 0.92, fill_fraction zero column.

Robustness (report sections 4 and 5, pi_lock_c under A1): the screened top 8 change to chip_input_loss, fsr,
thermal.stack, **self_heating.eta, self_heating.z_ratio**, thermal.crosstalk_scale, thermo_optic, er_mechanism. One at
a time, eta "high" or z_ratio 2 now unlock one ring and put one on a foreign line (worst-lane OMA penalty 32.8 dB,
mean relOMA -13.3 dB; g26: 0.35 dB, no ring lost), although the static bistability ratio is the same (6.02 against
6.05): a closed-loop effect (not diagnosed further; first suspect the hot-cusp heater authority of ring 0 with the
changed heater law). fsr window_a: 42.5 dB (g26 12.7). Over the 256 LHS worlds: pi_lock_c median worst-lane OMA
penalty 10.2 dB (g26 6.1), median wrong-line rings 2 (2); Kendall tau mean of the OMA-penalty ranking -0.28 (g26
-0.13), excursion count 0.19 (0.29), longest excursion 0.15 (0.32); 30.5 % of the Sobol worlds start supply-limited
(29.6 %).

### 11.3 Acceptance experiments: g26 against g26_calibrated

The nine quick files of experiments/acceptance (A4, the 4.8 h soak, not run), each with `extends: g26_calibrated`
instead of g26 (copies in runs/integrator_calib/acc_g26_calibrated/), same seeds and criteria. Runs:
`runs/acceptance_calib_stage_20261016/{g26,g26_calibrated}/` (git-ignored), 63 s and 62 s wall. The g26 runs are bit
for bit the integrator's runs of 2026-10-10 (every array of all nine; no engine change).

| file | g26 | g26_calibrated |
|---|---|---|
| A1 slide-10 aggressor | FAIL (lock 348.7 pm, BER 1.29e-8, monitor -69.2 / +2.9 %) | FAIL (lock 380.5 pm, monitor -67.9 / +2.8 %); **BER 7.3e-10 PASS** |
| A2 808 degC/s | FAIL (lock 103.2 pm, monitor 10.2 uA) | FAIL (lock 110.6 pm, monitor 8.5 uA) |
| A3 laser sweep | FAIL (lock 13.9 pm) | FAIL (lock 18.8 pm) |
| A6 connector re-mate | FAIL (lock 20.7 pm) | FAIL (lock 23.0 pm) |
| A7 relOMA plane | PASS (coverage 1, BND-4 5.4e-9 dB) | PASS (coverage 1, BND-4 5.7e-9 dB) |
| B1 squelch | FAIL (lock 19.2 pm) | FAIL (lock 28.6 pm) |
| B2 laser failover | FAIL (relock 51.6 pm off) | FAIL (relock 56.0 pm off) |
| B3 XPU power step | PASS (0.12 pm) | PASS (0.14 pm) |
| B5 cold start | PASS (all on their lines) | PASS (all on their lines; FSR-hop counter 10 against 5, not judged) |

- Same verdicts on every criterion except A1-ber, which passes on g26_calibrated. No wrong-line ring and no FSR hop
  in either. The set-point lock errors are 7-49 % larger on g26_calibrated (A1 +9 %, A3 +35 %, B1 +49 %).
- relOMA (bank mean) is 0.8 dB higher: A1 -5.40 dB (oracle -5.22) against -6.18 (-6.04), the CAL-5 at-power
  efficiency (93.6 against 72.5 pm/V). A7's oracle plane: -5.17 dB at 1.0 V to -8.36 dB at 3.75 V (g26 -5.99 to
  -9.23): still 3 dB short of the printed -2.1 to -3.0 dB (CAL-14 / 07 D9).
- A7 corners: g26_undercut coverage 1 -> 0.986 (16 points without a code, 2 supply-limited), g26_fsr_window_a 0.833
  -> 0.819, g26_heater_loss BND-4 0.064 -> 0.212 dB (still < 1 dB); the others as in g26.

### 11.4 Should g26_calibrated become the default preset? (recommendation; Kenny decides, nothing was switched)

**Recommendation: yes, in a separate change and with conditions.** Reasons for:

1. It is the only preset with held-out support. On the targets the calibration never read, it turns three fails into
   passes (VAL-2a, VAL-2b, VAL-3a: from 15-38 sigma to about 2 sigma), improves VAL-1, and turns none of the held-out
   passes into a fail. The improvements are in the notch depth and the EO bandwidth, which set the operating point
   and the drive of every controller.
2. It reproduces 13 of 20 scored calibration targets (g26: 9), and the remaining misses are stated and not hidden
   (CAL-2 and CAL-4 pass under the fit's own measurement models; CAL-9, CAL-12c and CAL-14 are model-form or
   definition gaps that g26 has too).
3. Closed loop, it changes no acceptance verdict for the worse; it exposes a self-heating fragility (eta high,
   z_ratio 2) that g26 hides, which is what a model for controller development must show.

Conditions and reasons against switching blindly:

1. Several moved values are not calibrated and rest on assumptions: the coupler dispersion 2.2 %/nm (`assumed`,
   2.9x the coupler artefact's 0.77 %/nm), alpha_other (`assumed`), R_s at the top of its range (a boundary
   solution; only the RC is identified), eff_at_0v 93.6 pm/V (the assumed +4 dBm measurement power; the P_in -> 0
   reading 72.5 pm/V is the alternative `cal5_p_in_0`), gain_scale (only gain x power x lock factor). The nominal
   world is now over-coupled at the operating line (it passes critical coupling during the heater sweep); the
   loss-mechanism worlds (under-coupled at the line) fit CAL-1/CAL-2 as well, so `er_mechanism` must stay a required
   corner of every controller claim.
2. CAL-2 needs a ~30 dB instrument floor that nobody has verified (Lightmatter question), and the heater ratio is
   calibrated only for the literature TCR.
3. The switch is not free: every `extends: g26` file (experiments/acceptance, examples), the golden traces, the
   controller scorecard and the A4 convergence certificate (the plant fingerprint changes, so the 4.8 h soak must be
   rerun) must be re-baselined in the same change; the engine regression scenarios (tools/perf_scenarios.py) should
   stay on g26.
4. Keep g26 as the named uncalibrated reference, and report controller results on both until the open Lightmatter
   questions (07 Q1 heater step response, Q4 drive topology, the G25 bias-ladder power, the notch floor) are
   answered.
5. Add the self-heating corner (eta high or z_ratio 2) to the corner set that controller claims are judged on: it now
   loses a ring under A1.

### 11.5 What still fails, and why

- Calibration targets on g26_calibrated (replica protocols): CAL-2 (the replica has no notch floor; passes with the
  fitted 29.7 dB floor), CAL-4 (the replica runs the median device; the swept device passes), CAL-9a/b/c (no
  hypothesis reaches them: heater authority at the hot cusp, the 30 um aggressor site and the referencing; a
  model-form or definition gap), CAL-12c (the scenario's laser drift converts 0.02 nm at 1310 nm for every line:
  scenarios owner), CAL-14 (relOMA -5.24 against -2.1 dB: a drive / definition gap, 07 Q4).
- Held out: VAL-1 (median against swept device, a protocol question for the replicas owner), VAL-4a/b/c and VAL-5
  (the relOMA map: level and curvature gaps of the CAL-14 family, 07 4.6 reloma-level / reloma-curvature), VAL-7b
  (definition conflict), VAL-8a/b/c (the thermal extraction artefact; the fast suite's three known failures);
  blocked: VAL-7c (no de Cea stack), BND-3a/b (the named controllers are not registered).
- Acceptance: A1, A2 (ring 0 at the hot cusp: heater authority, the aggressor site), A3 (the heater lock lags the
  50 nm/s ramps), A6 and B2 (a non-normalized set-point lock moves with every power step), B1 (data heat and the
  bit-averaged monitor level during squelch): controller and scenario issues, unchanged in kind by the calibration.

### 11.6 Requests (new)

- **replicas owner**: VAL-1 is the swept device (its description says so, as CAL-4's), but the replica runs the
  median device; the swept device's P_FSR is a calibration nuisance (29.1 mW in the record). Decide the protocol
  without looking at the result again (the integrator has seen it, section 11.2). CAL-5's at-power read-off is
  quantized to its 0.1 pm fine-sweep grid (0.02 sigma): harmless for the chi2, but the fit's 2-point differences
  see a staircase; a finer grid or an interpolated minimum would make the CAL-5 Jacobian smooth.
- **calibration owner**: (a) tag y = TCR x R_th,H rather than the ratio, or keep the ratio `inferred` (the joint
  screen fails it, 11.2); (b) when the record is assembled from parallel runs, rebuild its context with
  `pipeline.context_from_decisions` (now tested); (c) CAL-10 identifies (R_s + 50 ohm) x C_j (report reviewer).
- **controllers / scenarios**: the self-heating fragility of g26_calibrated under A1 (eta high, z_ratio 2: one ring
  unlocked, one on a foreign line) needs a diagnosis before the default switch.

