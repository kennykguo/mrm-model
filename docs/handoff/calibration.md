# Handoff: calibration (`mrm/calibration/`, `g26_calibrated.yaml`)

Owner files: `mrm/calibration/` (`core.py`, `groups.py`, `pipeline.py`, `preset.py`, `report.py`, `cli.py`,
`__init__.py`, `__main__.py`, the generated fit record `latest_fit.json`), `mrm/params/presets/g26_calibrated.yaml`
(generated), `tests/calibration/` (`conftest.py`, `test_fit_machinery.py`, `test_calibrated_preset.py`,
`a1_g26_calibrated.yaml`) and this file. Contract: 07 P0-5, P1-14, P1-23 and section 4 (4.3 in particular); the
calibration targets and the contamination warnings of `mrm/validation/targets/README.md`. Nothing outside these files
was edited. The calibration package reads only the calibration targets (`load_calibration_targets`); the isolation
scan of `tests/verification/test_validation_isolation.py` passes on it. No held-out target was evaluated by this task.

## 1. How to run

```
MG=".venv/bin/python tools/memguard.py --"
$MG .venv/bin/python -m mrm.calibration                    # every group (~3 h on a loaded machine), preset, before/after
$MG .venv/bin/python -m mrm.calibration --groups dac laser # later groups only (earlier leaves from latest_fit.json)
$MG .venv/bin/python -m mrm.calibration --write-preset-only    # regenerate g26_calibrated.yaml from latest_fit.json
$MG .venv/bin/python -m mrm.calibration --before-after-only    # the replica table only
$MG .venv/bin/python -m pytest -m calibration tests/calibration  # reported, not CI-blocking
```

`mrm calibrate` is the same command once `"mrm.calibration.cli:register_cli"` is added to `mrm.cli.PROVIDERS`
(integrator: `mrm/cli.py` is not mine). Peak RSS 0.2-0.7 GB.

## 2. Public API

```python
# mrm.calibration.core
Param(name, path, lo, hi, start, unit="", log=False, role="leaf"|"nuisance", prior_source="")
Problem(group, hypothesis, params, targets, predict: Callable[[dict], dict], fixed={}, starts=[], constraint=None,
        derived=None)                       # .residuals(theta), .chi2(theta), .predictions(theta) (cached)
fit(prob, *, max_nfev=200, extra_starts=0, seed=0) -> FitResult      # least_squares from every start, all minima
profile(prob, fr, name, *, n_grid=13, max_nfev=60, refine=6) -> Profile   # whole prior range, bisected crossings
restrict(prob, fixed: dict, *, hypothesis=None) -> Problem           # conditional fit (held parameters)
compare_hypotheses(group, fits, note="") -> HypothesisResult          # LR, Birge-scaled LR, AIC
target_residuals(target, prediction, *, sigma_scale=1.0) -> np.ndarray
half_prior(param, interval) -> float        # width / prior width (log coordinates for a log prior)
calibration_targets() -> {id: Target}; DCHI2_95 = 3.841; INTERVAL_WALL_FRAC = 0.005
# mrm.calibration.groups: Context(device, system, ranges).config(preset, extra=None, alts=None); one builder per group
#   coupling_er_problems, heater_problems, offset_problems, junction_problems, eo_problems, monitor_problems,
#   modulation_depth_problems (+ modulation_depth_scan), dac_problems (+ dac_codes), laser_problems;
#   notch_floor(depth_db, floor_db), CouplingConstants, bench_lut_mapping, widened_child
# mrm.calibration.pipeline
run(groups=None, *, quick=False, out=RESULTS_PATH, log=print, prior=None) -> CalibrationRun
run_group(name, ctx, targets, *, quick=False, log=print) -> GroupResult
refresh_coupling_worlds(path=RESULTS_PATH) -> dict; load_results(path) -> dict; write_results(run, path)
GROUP_ORDER, REJECT_LR = 0.05, LeafDecision, GroupResult, CalibrationRun
# mrm.calibration.preset
write_preset(results, path=PRESET_PATH) -> Path; preset_document(results) -> (dict, header lines)
bench_config(results) -> Config      # g25_test_vehicle + the calibrated device leaves (the G25 bench, calibrated)
bench_overrides(results) -> dict; bank_config(path) -> Config; PRESET_NAME = "g26_calibrated"
# mrm.calibration.report
before_after(results, *, ids=None) -> {id: {"before", "after", ...}}; format_table(ba); summary_markdown(results)
# mrm.calibration.cli
register_cli(register)               # "mrm calibrate"
```

## 3. Method (decided before the fits; deviations in section 6)

- **Forward model**: the replicas only, on the preset each target's generation runs on: G25 targets on
  `g25_test_vehicle`, G26 targets on `g26`. Parameters enter as overrides on top of the leaves of earlier groups
  (`Context`). Where a fit prior leaves a leaf's own `range` (heater R0, the swept device's ratio), a scratch child
  preset restates the leaf with the wider range (`widened_child`); nothing is written to the package except the
  calibrated preset.
- **Likelihood**: chi2 of standardized residuals with each published sigma (read-off sigmas as printed in the
  targets). CAL-1 (no sigma) is a flat likelihood on [3500, 4200] with a steep wall; it is a constraint, not a datum,
  so it adds no degree of freedom (`core.n_data`; fixed at the review: the first run counted it, which understated
  the Birge widening of the coupling group by ~1.2x). CAL-9a/c compare [lo, hi] elementwise with their sigma. The
  14 CAL-4 shifts share the 0 V read-off, so their errors are correlated (covariance 0.03^2 nm^2 off the diagonal);
  the fit treats them as independent (open issue, section 8).
- **Fit**: `scipy.optimize.least_squares` (trf, bounded to the prior, 2-point differences) in normalized prior
  coordinates, from several starts; distinct minima are kept (mirror branches are separate hypotheses).
- **Profiles**: chi2 re-minimized over the other parameters on a grid over the whole prior, threshold crossings
  bisected. 95 % interval: delta chi2 <= 3.84 s^2 with the Birge ratio s^2 = max(1, chi2_min/dof), so a model-form
  misfit widens the interval instead of shrinking it around a wrong value (raw intervals are in the record).
- **Hypotheses**: compared by chi2 (LR, Birge-scaled LR, AIC). A hypothesis is rejected when its Birge-scaled LR
  against the best is < 0.05. The preset keeps the g26 nominal hypothesis of each uncertainty entry unless rejected;
  measurement hypotheses (an instrument floor) take the best. No alternative is deleted.
- **Half-prior rule (07 4.3)**: `calibrated` only when the hull of the 95 % intervals over all non-rejected
  hypotheses in which the leaf is free is <= half the prior width (log coordinates for a log prior), closed on both
  sides (not cut by the prior) and one connected interval, AND (added at the review) the prior is a declared one
  (the leaf's `range` or an uncertainty interval of the preset, not fit bounds the calibration invented for a leaf
  that declares none: then the width of an invented range would decide the tag), AND the fit meets the tolerance of
  every target of the group (the targets' convention: a miss is a model-form failure, not a calibration). Otherwise the leaf keeps its provenance and the narrowed
  interval goes in its note; its value is still set to the nominal-hypothesis fit when that fit meets the target's
  tolerance (so the preset reproduces its calibration data where it can), and is NOT changed when the fit misses
  the tolerance (CAL-9, CAL-14).
- **Composed worlds**: a leaf that only one hypothesis frees (an extinction-mechanism coefficient, the over-coupled
  branch) is written on that hypothesis' alternative, from a conditional fit with the other leaves at the nominal
  fit, i.e. the world the preset actually composes. Its provenance follows the rule on the UNCONDITIONAL profiles
  (Q_L re-minimized too; for alpha_other the base hypothesis only, for a mechanism coefficient every non-rejected
  hypothesis that frees it); the narrower interval conditional on the held Q_L is in the note only (fixed at the
  review: the first version tagged the alternatives from the conditional profiles). The loss hypotheses hold the
  solver dispersion explicitly (fixed at the review: a full run used to leak the nominal world's fitted dispersion
  into the loss worlds through the shared context; the first record was right only because it came from
  `refresh_coupling_worlds`).
- **Contamination**: no prior or start value was taken from a research fit that used held-out data (03 key finding 5
  slope, 06 4.3 / V5 Q, 06 M6 swing range). Review correction: the dispersion fit bound 4 %/nm was first justified
  with 07 P1-5's "~2.9 %/nm needed"; that number is derived from the 17 dB notch at 13 nm of tuning, the held-out
  1.5 V point (VAL-2). The citation is removed, and since the leaf declares no range the dispersion is not tagged
  `calibrated` (rule above). Starts are preset values; priors are the leaves' own ranges (listed per
  parameter in `latest_fit.json` and in each leaf's `identifiable.prior`).

## 4. Results (run 2026-10-11, commit 0fe5b56 + this task; groups coupling_er, dac and laser rerun at the adversarial review with the corrected rules; full record `mrm/calibration/latest_fit.json`)

Tables: `mrm.calibration.report.summary_markdown(load_results())`. chi2 at the best fit, LR = exp(-dchi2/2) against the
best hypothesis, "Birge" = the same with the best hypothesis' chi2/dof scaling; width = 95 % hull / prior.

### 4.1 CAL-1 + CAL-2: Q_L, coupling branch, extinction mechanism (07 P1-5)

| hypothesis (branch / mechanism / floor) | chi2 (6 data + the CAL-1 constraint) | LR | LR Birge | verdict |
|---|---|---|---|---|
| over / ring_t_loss / floor | 9.13 | 1 | 1 | best |
| over / heater_t_loss / floor | 9.13 | 1 | 1 | same (the coefficients differ by the heater/ring rise ratio) |
| **under / dispersion / floor** (g26 nominal, kept) | 12.44 | 0.19 | 0.70 | not rejected |
| over / dispersion / floor | 19.95 | 0.004 | 0.31 | not rejected; pinned at critical coupling (x0 0.999, the branch wall) |
| under / ring_t_loss or heater_t_loss / floor | 21.5 | 0.002 | 0.26 | not rejected; pinned at critical coupling (x0 1.001) |
| any hypothesis without the floor | 46 ... 342 | < 1e-8 | < 0.02 | rejected |

LR Birge uses the best hypothesis' chi2/dof = 9.13/2 (review: the first run counted the sigma-less CAL-1 interval as a
datum, dof 3, and printed 0.58 / 0.17 / 0.13 / < 0.003; the verdicts do not change). The "mirror" fits (dispersion on
the over branch, loss on the under branch) sit on the branch wall x0 = 1, so their LRs compare boundary solutions.

Findings:
- **No hypothesis fits CAL-2 without an instrument floor.** The notch rises 27.9 -> 29.8 -> 29.8 dB and then falls;
  every mechanism of the model moves the coupling ratio x = kappa_e/(kappa_i + kappa_d) monotonically, so the rise
  needs a crossing of critical coupling between 0.2 and 0.8 V, whose infinite notch the read-off does not show. With a
  floor of 29.4-29.8 dB (fitted, a measurement nuisance, not written; the two equal 29.8 dB readings and the targets'
  "floor-sensitive near 30 dB" warning motivated it before the fits) every mechanism fits (chi2 9-12 for 2 dof; the
  remaining misfit, Birge s^2 4.6-6.2, is spread over the points, |z| <= 2). The floor hypothesis was added to the
  measurement model before the first fit; still, CAL-2 alone cannot prove that a floor, and not some unmodelled
  mechanism, is the cause. The replica itself has no floor, so g26_calibrated fails CAL-2 by 10.3 dB under the
  replica's own protocol.
- **The coupling branch is tied to the mechanism, and not identified.** Dispersion (x rising) fits only from the
  under-coupled side; a loss mechanism (x falling) only from the over-coupled side (the 07 P1-5 mirror). The pairs
  differ by LR 0.19 (raw) / 0.70 (Birge): the data do not decide between them.
  The preset keeps g26's nominal (under / dispersion) and writes the loss worlds with their own branch.
- **The dispersion needed is 2.19 %/nm** (95 % 0.88-2.88 %/nm over the two non-rejected dispersion hypotheses, under
  and over branch; the first record gave 1.57-2.76 from the nominal hypothesis alone, against its own hull rule)
  against the rib160 coupler artefact's 0.77 %/nm and femwell's 1.3-1.5 %/nm (05 t02). The leaf declares no range,
  so it is written `assumed` (review; it was `calibrated` against invented 0-4 %/nm bounds whose stated motivation,
  07 P1-5's "~2.9 %/nm", comes from the held-out 1.5 V notch; on those bounds the hull is 0.50 of the width, at the
  rule's edge anyway). Under the dispersion hypothesis the solver under-predicts the coupler dispersion by ~3x
  (best fit); under the loss hypotheses the solver value stands and the loss rises by d ln(alpha)/dT = 2.3e-3 /K
  (ring; 95 % 1.24e-3 to 2.70e-3 over both branches) or 2.0e-3 /K (heater temperature; 1.07e-3 to 2.33e-3).
- **Q_L is not narrowed beyond CAL-1** (profile 3470-4200, flat inside the printed interval): Q_L trades against
  alpha_other along the CAL-2 valley, and the interval's top coincides with the leaf range's top, so the rule counts it
  open. Q_L stays 3850 (`inferred`), alpha_other moves 13.35 -> 10.52 dB/cm (`assumed`, 95 % 0-24 dB/cm), which puts
  the cold 0 V state at x0 = 0.973 (Q_i 7701, under-coupled, Q_L/Q_i 0.50) with the 29.5 dB floor.

Composed worlds written (conditional fits, Q_L held at 3850; provenance from the unconditional profiles):

| world (preset selection) | free leaves (alternative, provenance) | chi2 | CAL-2 tolerance |
|---|---|---|---|
| nominal: coupling_regime under, er_mechanism dispersion | alpha_other 10.52 (assumed), dispersion 0.02192 /nm (leaf value, assumed; solver value kept as alternative `solver`) | 12.45 | pass (1.00 dB) |
| er_mechanism=ring_t_loss | alpha_other 7.232 (`cal_ring_t_loss_branch`, assumed: unconditional 0-20.2 dB/cm, open), d ln a/dT_ring 2.28e-3 (`ring_t_loss`, calibrated), solver dispersion | 9.14 | pass (0.86 dB) |
| er_mechanism=heater_t_loss | alpha_other 7.233 (`cal_heater_t_loss_branch`, assumed), d ln a/dT_heater 1.97e-3 (`heater_t_loss`, calibrated) | 9.14 | pass (0.86 dB) |
| coupling_regime=over (with dispersion) | alpha_other 9.237 (`over`, assumed; was tagged calibrated) | 41.5 | **fail (1.72 dB)**: a model-form failure, kept as a poor world |
| ring/heater_t_loss on the under branch (not composed, reported) | - | 37.2 | fail (2.36 dB) |

Review note: the first version tagged the three alpha_other alternatives `calibrated` from profiles conditional on
the held Q_L (e.g. 6.2-8.7 dB/cm), although Q_L itself is not identified and the unconditional profiles reach the
prior bound (0-20 dB/cm), and tagged `over` although its world misses the CAL-2 tolerance.

Because uncertainty entries apply in name order, `er_mechanism` (written here) sets alpha_other after
`coupling_regime`, so a loss world always takes its own (over-coupled) branch; the `over` choice of coupling_regime
then acts only in the dispersion world. This is the closest the hypothesis machinery can come to the tied pair the
data ask for (request below).

Order of the groups: the coupling group is fitted first, on the uncalibrated heater (CAL-2 depends on the heater law
through V -> P -> T). Refitting the nominal on the calibrated heater (a second pass, review, not written) moves the
dispersion 0.0219 -> 0.0222 /nm and lowers chi2 12.45 -> 10.62 (over/ring_t_loss 9.13 -> 7.87 at fixed parameters):
inside every interval, but the one-pass fit is not converged (open issue).

### 4.2 CAL-3a/b + CAL-4: heater (calibrated)

P_FSR 25.506 mW (was the printed 25.45; the R2 anchor is the FSR at the line, the bench protocol the cold-to-red-order
FSR: +0.22 %), R0 41.52 ohm (was 45.41; hot R at P_FSR 56.14 ohm = CAL-3b), heater_self_ratio 1.758 (was 1.158), i.e.
**y = TCR x R_th,H = 0.0138 /mW** under the written droop-transfer hypothesis same_rth_heater (0.0158 /mW, R0 40.0
ohm, ratio 2.01 under same_ratio: the two tie at LR 1, so y itself is known only to 0.0138-0.0158 /mW; was 0.00904;
TCR 1.4e-3 /K held, R_th,H 9.87 K/mW, Z 5.61 K/mW). The old y was the
textbook's fit without the thermo-optic curvature: with s(T) rising 14 %/100 K (order-2 model under fsr_integral) the
same 19 % droop of CAL-4 needs 53 % more TCR droop. Swept device: 0.349 nm/mW (06 E4: 0.348), median 0.399 nm/mW.
chi2 2.33 for 16 data. The droop-transfer hypothesis (same R_th,H on both devices vs same ratio) is not identified
(LR 1); the hull over both stays within the rule. R0's prior was widened from the leaf range [42, 58] to [35, 58]
(the old range was computed with the old y); the preset restates the range.

### 4.3 CAL-7 (calibrated) and CAL-5 (not calibrated)

- resonance_offset_nm 7.600 (95 % 7.40-7.80): R3 closes it by construction; tagged calibrated, nothing changes.
- eff_at_0v_pm_per_v: **93.9 pm/V at +4 dBm** (the replica's primary, 07 P1-7: relief removed at the measurement
  power; profile 86.1 to the prior top 95, open) against **72.5 pm/V at P_in -> 0** (65.7-79.4). LR 0.74: the data
  cannot tell the power. Kept `inferred`, value set to the primary fit (it meets the tolerance), the P_in -> 0 value is
  the alternative `cal5_p_in_0`. The +4 dBm fit sits at the edge of the prior: the self-heating relief at +4 dBm is
  ~23 % of the shift in this calibrated plant (93.9 vs 72.5 pm/V).
- fill_fraction (LUT mode, reported only; g26 uses the analytic law where the fill cancels): 0.882 at P_in -> 0; at
  +4 dBm the fit hits the prior top 0.95 (chi2 21, fails) - consistent with the junction handoff's 0.90-1.19. If the
  integrator switches g26 to the junction LUT, the G25 power question decides whether rib160 + L-shape reaches CAL-5.

### 4.4 CAL-10, CAL-8, CAL-12b

- **CAL-10**: only the R C is identified. With C_j(0) held at 15 fF, R_s = 390 ohm (the top of its range; 32.2 GHz,
  inside the 4 GHz tolerance; an exact fit needs R_s above its range at 15 fF, or e.g. R_s 288 ohm with C_j(0)
  20.9 fF, hypothesis both_free). With R_s held at
  100 ohm no C_j(0) <= 30 fF reaches it (LR 1e-4). Written: R_s 390 ohm, `assumed` (open profile). The RC pole
  (with the 50 ohm driver, C_j(1 V) 10.3 fF) is 35.0 GHz against the replica's 63.9 GHz optical FWHM at 1 V. Note that a larger R_s raises the junction share of the CV^2 data heat
  (R_s/(R_s+R_drv) 0.67 -> 0.89).
- **CAL-8**: gain_scale 0.823 under the half-maximum lock (the replica's primary), 0.531 under an oracle lock, a peak
  lock is rejected (needs < 0.5). Only gain x line power x lock factor is identified (07 D7): kept `assumed`, value =
  half-max fit, the oracle value as alternative `cal8_oracle_lock`.
- **CAL-12b**: laser.line_offset_ghz -2.17 GHz (95 % -3.3 to -1.0, `calibrated`), die/line sigma x0.72 (2.89 / 4.33
  GHz, 95 % x0.44-1.01): the Guide population's median and quartiles exactly. The sigma scale is written but stays
  `inferred` (review): the variation entry declares no range, and the tag would depend on the invented fit bounds
  [0.25, 2] (0.40 of them; 0.60 of [0.5, 2]).

### 4.5 CAL-14: the lumped modulation depth (not closed, not written)

Oracle relOMA at 1.0-1.15 V, 0-105 degC, 0 dBm (the replica's primary): -6.07 dB (g26) -> **-5.23 dB** (calibrated:
the CAL-5 at-power efficiency). Inside the swing prior [0.9, 2.6] V the best is **-2.52 dB at 2.6 Vpp** (target -2.1
+- 0.3: misses by 0.12 dB beyond the tolerance). Scan (`extra.scan`, centred levels, the junction grid clips at -0.5 V):

| what closes -2.1 dB | value | lumped depth D = swing_pm/FWHM_pm | plausibility |
|---|---|---|---|
| swing at the calibrated efficiency (93.9 pm/V) | **2.95 Vpp** (levels -0.40 .. +2.55 V; the -2.4 dB tolerance edge at ~2.7 Vpp) | 0.60 | above the prior (2.6) and 0.4 V into forward bias; needs a stacked or differential CMOS driver; no evidence either way |
| swing at the P_in -> 0 efficiency (72.5 pm/V) | ~4.95 Vpp (-2.09 dB at 5.0 Vpp; the -2.4 dB tolerance edge at ~3.8 Vpp) | 0.60 | implausible (forward-bias clip at -0.5 V, > 2x a 1.8 V IO rail) |
| efficiency at 1.3 Vpp | x2.36 (221 pm/V at 0 V) | 0.58 | implausible against G25's measured 52.8 pm/V average (DC) |
| one-sided levels | never (saturates at -3.41 dB) | - | - |

Needed D ~ 0.58-0.60 (~200 pm peak-to-peak at FWHM 340 pm), in line with 06 M6 (227 pm at Q 3500). Verdict: with the
CAL-5 at-power efficiency the printed level is just out of reach of a plausible drive; with the P_in -> 0 efficiency it
is far out of reach. It is a **drive/definition gap, not closable within the stated priors** (07 D9, Q4): the G26
swing is unpublished. The alternative reading "OMA in dBm at the actual ring power" gives -5.23 + 4.0 = **-1.23 dBm**
at the preset +4 dBm (the g26 value was -2.07 dBm), i.e. it now overshoots by 0.9 dB: it can match only at ~+3 dBm at
the ring. driver.swing_pp_v stays 1.3 (printed G25); its note records all of this. The setpoint-lock mode is worse
(-4.16 dB at 2.6 V).

### 4.6 CAL-9: DAC LSB (not reached, not written)

Fit protocol (deviation from the replica's primary, stated): one die, uniform aggressor (every lane sees the victim's
rise: the slide's ~20 k swings on all lanes), 8 draws of the per-ring variation with the die sigma 0 (lane spread of
one die), the measured die's blue offset as a nuisance (die sigma 6 nm), voltage law. Free: V_FS, per-ring sigma scale.

| hypothesis | chi2 (6 data) | V_FS | ring sigma scale | die offset | LR Birge |
|---|---|---|---|---|---|
| H_solver / temperature referencing | 250 | 1.44 V (95 % 1.40-1.54, open) | 0.32 (0.20-0.42) | 10.0 nm (prior top) | 1 |
| H_solver / resonance referencing (CAL-13a default) | 591 | 1.44 | 0.27 | 10.0 | 0.13 |
| H86 / either | 1.3e4-1.5e4 (a quick 4-die run found 726-743) | - | - | - | < 1e-30 |

Best: swing [21.4 k, 25.0 k] vs [19.7 k, 22.3 k]; lane offsets 5.7 k / 9.2 k vs 7.5 k / 9.5 k; range [17.9 k, 49.1 k]
vs [19.4 k, 49.25 k]: 7-9 sigma misses, outside every tolerance. Findings (reported, nothing written):
- Under the CAL-13a default (resonance trajectory 85.86 pm/K x 88.5 K = 7.6 nm, equal to the blue offset) the
  heater has no power left at the hot cusp (negative power at the nominal die); the slide shows ~19 k codes there. The
  data prefer the aggressor degC as ring temperature, and need the measured die ~10 nm blue (prior edge) or more heater
  headroom.
- H86 is strongly disfavoured on CAL-9 in every run (with H86 the lock temperature 113 degC sits below the cusp
  temperature, so the ring acquires the next order); its fits are multimodal, so the LR is qualitative only.
- The per-ring geometry sigma that gives the lane offsets is ~0.3x the Berkeley local spread (2.0 / 1.3 nm).
- The replica's primary on g26_calibrated still gives swing [3.9 k, 23.3 k] (aggressor 30 um from ring 0).

### 4.7 Not fitted, reported

CAL-6 (H86 value): H_solver apparent shift 67.4 pm/K on g26_calibrated (66.6 on g26). CAL-11, 12a, 12d, 13a, 13b
pass unchanged. CAL-12c fails by 1.3 % unchanged (the scenario's laser_drift converts 0.02 nm at 1310 nm for every
line; scenarios owner, already requested by the replicas handoff).

## 5. Before / after (the replicas' own protocols; G25 targets on g25_test_vehicle + calibrated device leaves)

| id | before | max |res| | after | max |res| | note |
|---|---|---|---|---|---|
| CAL-1 | pass | 0 | pass | 0 | |
| CAL-2 | fail | 41.6 dB | fail | 10.3 dB | the replica has no floor; with the fitted 29.5 dB floor: pass (1.0 dB, test) |
| CAL-3a | pass | 0.056 mW | pass | 4e-7 | |
| CAL-3b | pass | 0.0033 V | pass | 4e-7 | |
| CAL-4 | fail | 2.21 nm | fail | 1.90 nm | the replica runs the median device; the swept device (P_FSR 29.2 mW): pass (test) |
| CAL-5 | fail | 15.6 pm | pass | 6.8 pm | primary +4 dBm |
| CAL-6 | not scored | | not scored | | 66.6 -> 67.4 pm/K |
| CAL-7 | pass | 5e-6 | pass | 9e-12 | |
| CAL-8 | fail | 3.08 uA | pass | 0.002 uA | |
| CAL-9a/b/c | fail | 16 k / 29 k / 16 k | fail | 16 k / 28 k / 17 k | replica primary geometry; see 4.6 |
| CAL-10 | fail | 13.0 GHz | pass | 1.3 GHz | |
| CAL-11, 12a, 12d, 13a, 13b | pass | | pass | | unchanged |
| CAL-12b | fail | 3.56 GHz | pass | 0.026 GHz | |
| CAL-12c | fail | 0.00026 nm | fail | 0.00026 nm | scenario conversion |
| CAL-14 | fail | 3.97 dB | fail | 3.13 dB | -6.07 -> -5.23 dB; see 4.5 |

Leaves tagged `calibrated` in g26_calibrated (after the review): ring.heater.p_fsr_mw, ring.heater.r_cold_ohm,
ring.thermal.heater_self_ratio, ring.optics.resonance_offset_nm, laser.line_offset_ghz, and the alternatives
`ring_t_loss` (dln_alpha_dt_ring) and `heater_t_loss` (dln_alpha_dt_heater), each conditional on its mechanism and on
the CAL-2 instrument floor. Values changed but not calibrated (provenance kept or `assumed`): alpha_other_abs and its
alternatives `over`, `cal_ring_t_loss_branch`, `cal_heater_t_loss_branch` (assumed), the dispersion leaf (assumed; was
solver), eff_at_0v (inferred), series_resistance (assumed), monitor gain_scale (assumed), the variation sigma of
laser.line_offset_ghz (inferred). The review removed the `calibrated` tag from the dispersion leaf, the three
alpha_other alternatives and the line-error sigma (rule clauses above). Unchanged with a note: q_loaded,
fill_fraction, swing_pp_v, full_scale_v.

## 6. Closed loop on g26_calibrated

`tests/calibration/a1_g26_calibrated.yaml` (A1 settings on g26_calibrated) runs without error (3.1 s): FAIL on the
same criteria as g26 (lock 390.9 pm at ring 0 near the hot cusp vs 348.7 pm on g26 - the heater-authority problem of
the 30 um aggressor site; BER 2.4e-9; monitor -68 / +2.8 %), no wrong line, no FSR hop, worst-lane OMA penalty
0.35 dB, relOMA -5.38 dB (oracle -5.21; g26 -6.18 / -6.04). g26_calibrated also passes tests/closure/test_presets.py
(load, relations, every hypothesis realizes, Monte Carlo batch).

## 7. Deviations (please review)

1. **Instrument floor on CAL-2** (measurement nuisance, not in the replica): decided before the first fit from the two
   equal 29.8 dB readings and the targets' floor warning; reported with and without (without: every hypothesis
   rejected).
2. **CAL-9 fit protocol** differs from the replica's primary (uniform aggressor, one die, die-offset nuisance): the
   primary geometry cannot reach the lower code end with any DAC parameter. Nothing was written from it.
3. **Q_L written at the CAL-1 centre (3850)**: the free best fit sits at an arbitrary point of a flat valley; the
   centred refit is within dchi2 0.01. Profiles are of the free fit.
4. **Rule interpretation**: "calibrated" also requires a closed interval and one component (a prior-truncated or
   hypothesis-split interval would pass the width test spuriously). q_loaded fails only on this (its upper edge is the
   CAL-1 edge, which equals the leaf range's top). Added at the review: a declared prior (not invented fit bounds),
   a fit within every target's tolerance, and, for a composed world, the unconditional profile.
5. **Values of non-calibrated leaves are moved to the nominal fit** when it meets the tolerance (alpha_other, eff, R_s,
   gain_scale), not left at the prior value, so the preset reproduces its calibration data; provenance not upgraded.
6. **R0 prior widened** to [35, 58] ohm (restated range in the preset).
7. The CAL-5 power, CAL-8 set-point and CAL-10 R/C splits are hypotheses carried as alternatives, not resolved.

## 8. Open issues and requests

- **integrator**: add `"mrm.calibration.cli:register_cli"` to `mrm.cli.PROVIDERS`; add the `calibration` marker to
  pyproject (tests/calibration/conftest.py registers it locally and skips the tests unless `-m calibration`);
  CHANGELOG line for g26_calibrated. g26_calibrated is a separate preset; g26 is unchanged.
- **params_machinery**: tied hypotheses (branch x mechanism; Q_L with alpha_other) - the data say "loss mechanism <=>
  over-coupled branch"; the order-of-entries trick in g26_calibrated is a workaround. An absolute R_th,H key (or a tied
  ratio) so the fitted y (0.0138 /mW under same_rth_heater, 0.0158 under same_ratio) holds across FSR / thermo-optic
  worlds (integrator request 3c).
- **params_content / extraction (coupling)**: the dispersion world needs 2.2 %/nm against the rib160 artefact's
  0.77 %/nm; re-check the coupler solve (curved gap, rib slab) or accept that the loss mechanisms are the physical
  ones. The R0 leaf range [42, 58] should follow the droop.
- **scenarios / layout**: CAL-9 prefers temperature referencing of the slide-10 degC and a uniform aggressor; under
  the resonance default the heater runs out at the hot cusp (ring 0 at code 0 in A1 as well). The aggressor site 30 um
  from ring 0 is the first suspect (replicas handoff 4.2). CAL-12c conversion per line.
- **junction**: decide LUT vs analytic for g26 (4.3); at +4 dBm the rib160 LUT needs fill > 0.95.
- **calibration (open, found at the review)**: (a) the groups run once in order, but CAL-2 depends on the heater
  law fitted after it; a second pass moves the dispersion by +1.4 % and chi2 by -1.5 (4.1): iterate coupling <-> heater
  to convergence in a future run. (b) The 14 CAL-4 shifts are differences against one 0 V read-off, so their errors
  are correlated (off-diagonal 0.03^2 nm^2); the likelihood treats them as independent (chi2 2.3 for 12 dof suggests the
  sigmas are generous; the heater intervals are approximate). A full covariance (or a 0 V offset nuisance) would fix
  it. (c) A run of `--groups X` keeps the other groups' records and context leaves (fixed at the review: a subset
  starting at coupling_er used to drop them), but a later group that is not rerun keeps its old fit.
- **report owner**: run the held-out targets on g26_calibrated (this task did not look at them); flag that CAL-14 and
  CAL-9 are unreached calibration targets (model-form / unknown drive), and that the CAL-2 fit needs a floor.
- **Lightmatter questions** sharpened: G26 junction swing and drive topology (07 Q4: 2.95 Vpp would close CAL-14 at the
  at-power efficiency); the G25 bias-ladder optical power (CAL-5: 72.5 vs 93.9 pm/V); the slide-10 thermometer.

## 9. Tests and runtime

- `pytest -m calibration tests/calibration`: 23 passed (9 machinery tests on known answers: linear recovery and the
  analytic 95 % interval, open profiles, Birge widening, flat interval targets, set/curve sigmas, log priors, LRs,
  restrict; 14 preset tests: load/realize, every written leaf carries targets/residual/identifiable and the rule,
  preset == record, CAL-1/3a/3b/5/7/8/10/12b reproduced by the replicas, CAL-4 by the swept device, CAL-2 with the
  floor, A1 closed loop runs). Without `-m calibration` they are skipped (not CI-blocking).
- tests/closure/test_presets.py + test_targets_schema.py + tests/verification/test_validation_isolation.py: 205 passed,
  2 skipped (g26_calibrated included). tests/verification/test_kernel_parity.py: 18 passed (no engine code touched).
- Runtime of a full `python -m mrm.calibration`: ~3 h on the shared machine (coupling profiles ~1.5 h, heater ~0.5 h,
  CAL-9 ~0.5 h, the rest minutes); peak RSS 0.8 GB.

