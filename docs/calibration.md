# Calibration and independent validation

Measured input/output behavior is the strongest route to an accurate controller plant when the proprietary geometry is unavailable. Solver extraction can supply coefficients for a declared geometry, while measured scans/steps supply manufactured-device behavior. Preserve units, timestamps, reference planes and which control loop was active.

## Optical spectra

Use a CSV header:

```text
frequency_hz,through_power_w,drop_power_w
```

`drop_power_w` is optional, but through-only fitting has substantial loss/coupling ambiguity. Record one increasing or otherwise distinct frequency axis, the measured absolute powers, constant optical input power, actual ring temperature, electrical bias, heater condition and port calibration. Do not independently normalize every curve to its minimum or interpret an uncalibrated ADC code as optical power. If the temperature reference is unknown, the fitted resonance is an operating-condition reference, not a zero-heater device parameter.

```sh
.venv/bin/mrm calibrate-spectrum measurements/spectrum.csv \
  --config examples/single_ring.json --input-power 0.001 \
  --output outputs/optical_fit.json
```

The fit defaults to reference resonance, intrinsic loss, bus coupling and drop coupling. Their coordinates are scaled; losses are positive and coupling fractions bounded. Length and group index remain fixed because one narrow spectrum cannot identify them separately. API controls are `fit_fields`, `holdout_fraction`, `training_mask`, `relative_noise_floor` and `max_nfev`; `temperature_k` and `junction_voltage_v` set the known measurement condition.

The result includes fitted parameters, through/drop predictions, training/validation indices, RMSE in watts, covariance/standard errors when locally identifiable, Jacobian condition and warnings. A poor Jacobian declines covariance instead of returning spuriously precise pseudoinverse errors. Intensity-only measurements can retain discrete under/overcoupling branches even when local covariance is small; a different starting branch, phase measurement or independent loss/coupling constraint is necessary to investigate them.

The CLI also writes a `.config.json` with only the fitted ring coefficients updated. It keeps `calibrated=False`: fitting an optical scan does not calibrate thermal dynamics, interfaces, other lanes or different temperatures. Supply voltage tables for measured signed resonance/loss curves; their interpolation refuses out-of-domain voltage extrapolation.

## Actual-power thermal steps

Use a CSV header:

```text
time_s,frequency_shift_hz
```

The step must be in **actual applied heater power**, from an initially equilibrated state. Convert voltage/current into power using synchronized measurements; a commanded DAC edge is not automatically the heater-power edge. Keep package boundary, laser frequency, optical power and junction bias fixed and disable the locking loop. If only closed-loop data exists, retain synchronized controller actions and identify the closed-loop experiment appropriately; do not fit it as an open-loop heater step.

```sh
.venv/bin/mrm calibrate-thermal measurements/heater_step.csv \
  --step-power 0.001 --modes 2 --output outputs/thermal_fit.json
```

`fit_thermal_step()` fits a same-sign sum of exponential modes plus offset. API options include `n_modes` (1–8), `step_power_w`, known `step_time_s`, `holdout_fraction`, `training_mask`, `amplitude_sign` and `max_nfev`. Transport delay is not independently fitted. Align the actual power edge or identify delay separately. It returns pole times in seconds, amplitudes in Hz, gains in Hz/W, fit status, residuals/covariance/condition diagnostics and withheld late-sample error. A brief record may not identify its slow pole or steady gain. A cross-ring spatial response can require signed modal residues; this same-sign fitter is not a universal MIMO identification method.

This fit does not invent a unique C/G geometry from one transfer function. A measured multiport thermal ROM can be imported directly as `ThermalParams`; the repository's FEM export is a worked example of passive C/G/maps extraction. Multiple heater/ring responses, independent boundary perturbations and separate heater temperatures distinguish dynamics that a single scalar heater-to-resonance record cannot.

## What constitutes validation

The test suite's parameter-recovery data is synthetic and explicitly labeled. Recovering its known values verifies fitting machinery. Withheld points in one spectrum or late samples of one step are useful residual checks, but they are weaker than a separate experiment.

Fit across multiple amplitudes/temperatures/biases and reserve separate conditions, traces and preferably devices. Report resonance/linewidth/through/drop residuals, thermal transient timing and gain errors, predicted versus actual ADC codes, actuator limits and saturation, acquisition/recovery and OMA margins. Use experimental uncertainty to define a validity envelope; record correlated corners instead of sampling unrelated intervals as if every combination were physically possible.

The OFC paper's thermal tests and sampled BER are system-level reference evidence. They do not uniquely identify the uncontrolled plant, establish internal control bandwidth or certify a new simulator's BER. A 125 °C scenario is an extrapolation beyond the published 105 °C demonstrations. See [lightmatter-data-request.md](lightmatter-data-request.md) for a prioritized measurement plan.

## Parameter corners through the same interface

```sh
.venv/bin/mrm sweep --config examples/single_ring.json \
  --corners examples/corners.json --controllers pi mpc --output outputs/corners.json
```

Each corner uses dotted JSON paths, for example `optical.0.df_dtemperature_hz_per_k`. The entire resulting configuration is revalidated, including passive C/G constraints. The nominal controller context and assigned detuning target remain fixed across corners; a changed device does not receive its unknown parameters and initial error is not redefined to zero. Reports record each override, metrics and warnings. Summaries are retained, while a corner's full traces are released before the next case. The shipped corners are illustrative stress hypotheses, not sourced production distributions.
