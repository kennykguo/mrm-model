# Calibrating optical spectra without inventing missing information

`ringlab.optical_calibration` fits the exact per-pass optical equations to
normalized measured through/drop spectra. It does not generate measurements or
calibrate a Lightmatter device from link-level BER figures.

CSV headers are explicit: `frequency_hz` plus `through_fraction` and/or
`drop_fraction`. Optional `through_sigma` and `drop_sigma` contain measurement
standard deviations in the same normalized units. The caller supplies
temperature and provenance, including how incident power, dark offsets and path
losses were calibrated. Fractions outside [0,1] are permitted as noisy data and
are not clipped. No unmeasured scale or background is silently fitted.

Every fit must specify which parameters may vary, a finite bounded interval for
each, and a boolean training mask leaving at least three points out. Unvaried
parameters come from the supplied `RingParameters` and are explicitly reported
as fixed assumptions. Supported fitted quantities are reference resonance shift,
intrinsic loss, input power coupling and drop power coupling. Length/group index
are not jointly inferred from one isolated resonance. Several bounded starts
help find competing minima; this is not a proof of a global optimum.

```python
from ringlab.optics import RingParameters
from ringlab.optical_calibration import load_spectrum_csv, fit_spectrum
import numpy as np

data = load_spectrum_csv(
    "measured_scan.csv",
    provenance="Measurement ID, temperature/power/bias, port normalization",
    temperature_k=300.0,
)
# Example only: couplings must come from independent evidence if held fixed.
known = RingParameters(input_coupling_power=0.025, drop_coupling_power=0.005)
result = fit_spectrum(
    data,
    known,
    vary=("resonance_shift_hz", "intrinsic_loss_per_m"),
    bounds={"resonance_shift_hz": (-20e9, 20e9), "intrinsic_loss_per_m": (0, 2000)},
    training_mask=np.arange(data.frequency_hz.size) % 4 != 0,
)
print(result.fitted_values)
print(result.heldout_rmse_by_port)
print(result.identifiability_notes)
```

Holding out interleaved points from one scan checks spectral prediction between
training points. It does not validate a new temperature, bias, optical power,
device or operating trajectory. Such validation requires separately acquired
conditions. If measurement noise scales are omitted, residuals do not establish
confidence intervals. Even supplied noise scales do not account for uncertainty
in the fixed geometry, couplers, normalization or model structure.

## A discrete ambiguity remains even with two power ports

Let `r=sqrt(1-K1)`, `b=a*sqrt(1-K2)`, and `d=K1*K2*a`, with
`a=exp(-alpha*L/2)`. The through-power numerator is symmetric in r and b:

```
|H_through|^2 = (r^2+b^2-2*r*b*cos(phi)) / (1+r^2*b^2-2*r*b*cos(phi))
|H_drop|^2 = d / (1+r^2*b^2-2*r*b*cos(phi)).
```

Swapping r and b preserves both denominators and through power. Keeping d fixed
then yields another passive parameter triple whenever its attenuation is at
most one. `equivalent_power_parameters` computes that branch analytically. For
the illustrative default 10 um radius ring:

| Parameter | Original | Equivalent power branch |
|---|---:|---:|
| Intrinsic loss, Np/m | 250 | 305.4672 |
| Input coupling power | 0.025 | 0.02050731 |
| Drop coupling power | 0.005 | 0.00610602 |

Both port **power** spectra coincide to floating-point accuracy, while complex
through fields differ. This follows algebraically from the model's power
equations; an optimizer returning one branch does not resolve the ambiguity.
The fit returns the alternate parameters and whether they satisfy the supplied
fixed values/bounds. Excluding a branch by an arbitrary bound is an assumption,
not independent evidence. Through-only data have additional ambiguity between
intrinsic loss and drop coupling. A locally full-rank Jacobian cannot detect
every discrete ambiguity, so rank and the analytic alternate branch are both
reported.

Use calibrated complex-field phase, independently measured coupling/loss,
additional electrical/thermal observables, or a known coupling regime to label
physical branches. A physically plausible fit and a calibrated device model are
different levels of evidence.

## Different kinds of bounds must remain separate

The code enforces passivity and admissibility, such as nonnegative total optical
loss, power couplings below one, and a positive group delay. These constraints
make the mathematical model well defined; they do not establish that a chosen
value matches the actual device.

A measured junction's voltage interval is a data-domain boundary. Requests
outside it are rejected because the table contains no evidence there. A fitted
parameter interval is instead a caller-supplied search constraint. A fitted
value touching that interval can indicate inadequate data, a wrong fixed
assumption, a missing effect, or an interval that excluded the device. It is not
a measured physical limit. Inspect both the fitted values and their stated
bounds before using a fit for control design.

Numerical tolerances control optical interpolation, waveform stepping and
optimization. Their checks quantify numerical discrepancies for sampled inputs.
They do not include unknown coupler dispersion, scattering, thermal gradients,
phase splitting, parameter drift or uncertainty in measurement normalization.
The nonlinear screening ratios likewise are not certified omission bounds.

Finally, the currently shipped parameter ranges and defaults are sensitivity
assumptions. They are not confidence/credible intervals or a fabrication-yield
distribution. A defensible statistical interval needs a likelihood/noise model,
data covering the parameters, treatment of correlated nuisance quantities,
branch identification and independent validation. The current fitting helper
reports residuals, local rank and competing branches rather than manufacturing
such an interval from an uncalibrated synthetic dataset.
