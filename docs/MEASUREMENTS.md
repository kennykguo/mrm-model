# Turning an assumed plant into an identified device model

The most useful next accuracy improvement is a small, synchronized **open-loop
dataset**, not more decimal places in assumed coefficients. These are proposed
measurements for discussion with Lightmatter; no hardware experiment has been
performed or authorized by this repository. Use device-owner-approved optical,
electrical, and temperature limits. The public paper's stress tests are not a
safe-operating-area specification for another device.

## Minimum useful release

Raw data can remain separate from proprietary layout. Each dataset should have
an acquisition ID, units, reference planes, timestamp definitions, instrument
bandwidth/uncertainty, initial conditions, operating temperature and bias,
data pattern/rate, and the state of any existing feedback loop. Preserve raw
samples alongside calibrated values; record missing/clipped measurements.

| Dataset | What it identifies | What it cannot identify alone |
|---|---|---|
| Through and drop spectra versus optical frequency at several known temperatures, reverse biases and input powers | Local resonance, FSR, line shape, thermal/bias slopes, evidence of splitting or power dependence | A unique coupling/loss branch from power alone; absorbed versus scattered loss; private geometry |
| Heater terminal V/I and synchronized resonance/monitor responses to multiple small steps | Code-to-power conversion, heater gain, transient poles and headroom | Full cross-coupling matrix from common-mode heating; unexplained sensor filtering versus thermal poles |
| One heater/aggressor excited at a time, all ring responses observed | Cross-ring transfer paths and disturbance entry points | True package geometry or unique high-order internal realization |
| Monitor dark records and illuminated records at several powers/temperatures; known optical modulation or electrical injection | Dark offsets, noise spectrum, filtering, saturation and timing | Receiver BER or an unmeasured local temperature |
| Junction C-V, I-V, series parasitics and complex electro-optic response | Voltage tables, electrical time scales and RF validity | Temperature dependence from a single-temperature sweep; full driver behavior from one bandwidth number |
| Independent closed-loop captures with optical quality measurements | Capture/recovery behavior and model discrepancy under control | A plant fit if all data were already used to tune the model/controller |

For heater identification, distinguish requested DAC code, actual latched code,
terminal voltage/current, optical sample acquisition time, and sample release
time. A single software timestamp is insufficient to separate physical delay,
filtering and transport latency. Measure the boundary/package temperature as a
separate channel; it is not automatically the ring temperature.

## Fit, then challenge the model

1. Establish optical normalization and a low-power regime. Look for doublets,
   hysteresis, asymmetry and power-dependent shifts before fitting a linear
   traveling-wave ring. Do not hide systematic structure in a noise estimate.
2. Fit only identifiable parameters. `fit_spectrum` requires explicit bounds
   and varied fields; it reports an alternate power-equivalent branch.
   Independently measured constraints are preferable to selecting a branch
   merely because it resembles an assumed device.
3. Identify thermal paths with measured **watts**, not voltage alone. The
   bundled fitters assume an equilibrium-to-constant-power step at a fixed
   uniform boundary. Deconvolve or explicitly model instrument/source dynamics;
   do not silently fit them into the thermal plant.
4. Reserve entire independent acquisitions, operating points and disturbances
   for evaluation. A random split of neighboring time samples is usually not
   independent. Report residual versus time, bias, temperature and input power,
   not just a single aggregate error.
5. Evaluate the resulting model under the same quantized/delayed interface used
   by each controller. Compare directly observed signals first. Independent
   truth channels are evaluator outputs, never hidden controller inputs.

`MeasuredJunctionTable` imports bounded one-dimensional voltage relationships;
`ThermalNetwork` imports a conservative operator with an explicit source/node/
observer port contract. Neither API can consume an arbitrary foundry file or
represent every temperature-dependent device effect. The current thermal
fitter changes two global scales of a chosen network; it is not general
multi-input system identification.

The standalone thermal API accepts separate boundary temperatures. The current
controller plant supplies one ambient temperature to every boundary column,
deposits driver heat at its named package node, and uses package temperature
for detector dark current. A replacement network must preserve or explicitly
revise those semantics; matching port names alone is insufficient.

## Acceptance must name an observable and a domain

Before declaring an algorithm successful, agree on the intended wavelength and
fringe, operating range, disturbance envelope, lock-acquisition/recovery time,
allowed optical penalty, actuator headroom/energy, and measurement coverage.
No universal pass threshold is inferred from the OFC plots. A monitor setpoint
may remain constant while the wrong wavelength is selected or OMA worsens.

Numerical refinement measures discretization error for the chosen equations.
Held-out measurements estimate discrepancy in the tested domain. Unknown
geometry, shared calibration errors, correlated fabrication variation and
omitted physics require separate uncertainty statements. Synthetic uniform
parameter sweeps are sensitivity studies, not calibrated confidence intervals
or manufacturing-yield estimates.
