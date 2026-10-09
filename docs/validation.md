# What has been verified

The compact spectrum has been cross-checked against a separate photonic circuit
solver: [SAX](https://github.com/gdsfactory/sax), an open-source scattering-matrix
simulator built on JAX and distributed under the Apache Software License. This verifies the ring equations and their implementation
under the declared component assumptions. It does not establish that the chosen
parameters describe a fabricated Lightmatter device.

The independent model contains two reciprocal, lossless four-port directional
couplers and two lossy waveguide segments. Connecting the segments closes the
ring. The circuit solver resolves that feedback; no component contains the
closed-form ring transfer function. The coupler's self and cross field amplitudes
are `sqrt(1-K)` and `i*sqrt(K)`, so its complete scattering matrix is unitary.
The propagation segments attenuate and phase-shift fields according to their
length, group index and power loss. SAX's feedback-capable **Filipsson-Gunnar**
backend is explicitly selected. Its forward-only/additive solvers must not be
used for a feedback ring, as the [SAX backend documentation](https://gdsfactory.github.io/sax/api/)
explains.

A second independent calculation builds the full 12-port connectivity matrix
and solves `b = S a`, `a = P b + e`, or `(I - P S) a = e`, using NumPy's linear
solver. `S` contains the component scattering matrices, `P` pairs connected
ports, and `e` excites the external bus input. This calculation also tracks the
incident and outgoing fields at each internal segment. Loss is the sum of
power dissipated by those segments, rather than a remainder obtained by
subtracting external through/drop powers. Stored cavity energy integrates the
traveling power in each segment. The absorption fraction splits dissipated
power into heat-producing absorption and unresolved scattering.

## Reproduce the cross-tool check

Install the optional stack into the project environment and run:

```sh
uv pip install --python .venv/bin/python 'sax==0.18.2'
.venv/bin/python scripts/validate_sax.py --solver sax --output results/sax-validation
.venv/bin/python -m pytest tests/test_cross_tool.py -q
```

SAX is optional. `--solver graph` runs the component connectivity solve with the
base NumPy/SciPy environment. `--solver auto` uses SAX when installed. The report
always records which solver ran; absence of SAX cannot silently turn into a
claim that SAX was exercised. GPU preallocation is disabled and JAX uses CPU
float64. The arrays are bounded to at most 4,001 frequency points per validation
case and 10,000 points in the graph helper.

The script writes `report.json`, `curves.csv`, and, when Matplotlib is installed,
`spectrum_comparison.png`. The CSV retains both spectra and independent segment
absorption so differences can be examined without the plotting dependency.
The report records versions, seed, each case, maximum errors, memory use and
the pass criterion. Tests exercise coupler unitarity, randomized feedback
networks, temperature/voltage perturbations and the real SAX backend. The SAX
test is explicitly skipped if that optional package is absent.

## Recorded result

The 9 October 2026 run used NumPy 2.5.3, SciPy 1.18.1, SAX 0.18.2 and JAX/JAXlib
0.9.2, with seed 92. It evaluated 26 cases at 401 frequencies per case and 1 mW
input. Cases include the default, one-bus, lossless, weak-coupling and partial
absorption configurations, a dispersed ring, plus 20 random geometries, losses
and couplings. The scan spans 2.4 reference free spectral ranges and checks
constant-FSR periodicity only for zero-dispersion cases. Nonzero group-delay
dispersion makes the spacing between resonances vary.

| Quantity | Largest absolute difference/residual |
| --- | ---: |
| SAX through power versus compact model | `1.64e-16 W` |
| SAX drop power versus compact model | `2.59e-16 W` |
| Component-segment absorption versus compact model | `2.90e-16 W` |
| Integrated cavity energy versus compact model | `4.02e-28 J` |
| External output plus segment loss minus input | `1.13e-17 W` |
| Graph transmission after one additional FSR | `3.34e-16 W` |

The differences are consistent with floating-point roundoff, and all four
cross-tool tests passed. Peak resident memory was **574 MiB**, below the 1 GiB
budget for this short CPU validation process. The current generated report is
the authoritative result if dependency versions or implementation changes.

## Dynamic, nonlinear and calibration checks

The physics tests establish passive power conservation, lossless limits,
critical coupling, the expected sign of thermal and electrical tuning, and
periodic resonances. With group-delay dispersion, stored energy uses the local
roundtrip delay `dphase/domega`; the local intrinsic decay rate changes
consistently so absorption and passive conservation remain exact. Nonpositive
local delay rejects an unphysical extrapolation. The exact local FWHM accounts
for asymmetric half-height frequencies. The short TCMT waveform uses the
reference delay and remains a local resonance approximation.

An unforced dynamic cavity decays with the derived
logarithmic roundtrip energy rate. The local TCMT spectrum approaches the
periodic spectrum in the high-finesse limit; they are not asserted to be exactly
equal for strong coupling. Reciprocal coupling between counterpropagating modes
preserves total energy while moving it between directions. An instantaneous
energy-balance test retains external port powers, cavity energy change and
TPA/FCA dissipation.

Nonlinear steady-state tests retain all positive-energy polynomial branches,
calculate the optical/carrier Jacobian eigenvalues, and distinguish stable and
unstable solutions. Branch selection requires a previous energy when more
than one stable solution exists. Dynamic carrier generation uses two photons
per electron-hole pair and the supplied effective carrier volume. A nonlinear
transient converges to an independently enumerated stable equilibrium. These
checks do not include the thermal Jacobian and therefore cannot establish
stability of the complete electrothermal system.

NRZ tests check RC/cavity/receiver operation and convergence when temporal
sampling is doubled. OMA and extinction follow the simulated optical levels.
The reported Gaussian BER proxy combines conditional sample spread and declared
noise. It is not a physical prediction of the paper's measured BER and must not
be used as evidence that the 800-Gb/s link demonstration was reproduced.

Calibration tests use **synthetic** spectra and heater steps with known
coefficients, then recover those coefficients and evaluate withheld samples.
That checks the fitting machinery. Power-only spectra may still have discrete
under/overcoupling alternatives; covariance cannot resolve them. Poorly
identifiable fits return no covariance rather than precise errors inferred
from a deficient pseudoinverse. Withheld points from the same scan or trace
also do not replace an independent experiment.

## What remains necessary for device accuracy

Measured through/drop spectra, phase or independently known loss/coupling,
voltage-dependent resonance and loss, actual-power heater step responses,
package boundary conditions, and sensor/actuator timing are still needed.
Final evaluation should use separate temperatures, optical powers, heater
amplitudes, disturbances and devices from those used for fitting. A successful
SAX cross-check cannot supply unknown Lightmatter parameters or validate
extrapolation beyond measured operating conditions.

For controller comparisons, the monitor objective, acquisition task, actuator
authority, available calibration information and disturbance/noise realizations
must match. Peak-drop extremum seeking and a prescribed flank or OMA target are
different tasks; a score comparison is meaningful only after those objectives
are aligned. Likewise, convergence of an initialization fixed-point iteration
does not by itself establish physical electrothermal stability.

## Connected-plant and interface checks

Integration tests exercise all-carrier power accounting, boundary-temperature
memory, electrothermal substep refinement, deterministic independent random
streams, finite controller outputs, and the absence of temperature/detuning
truth in online observations. A one-code perturbation with 15 µs actuator and
25 µs sensor delay agrees with the local delay-augmented model within three
ADC codes. Device-corner tests hold nominal calibration and assigned detuning
fixed rather than silently adapting the controller to the changed device.

Specified nodal and generalized/modal initial states are checked against the
same automatically equilibrated state, including accepted heater bias and PI
response. Compliance derivatives are compared with independent settled-driver
finite differences in interior, current-, power-, voltage-, and minimum-output
regimes. Two-lane tests compare complete DC monitor gain against independently
resolved equilibria, with and without driver settling. These checks distinguish
fixed-voltage TCR from the temperature dependence of a compliant voltage target.
Clipped ADC calibration has no local measurement slope or unreachable target;
open-loop plant runs remain possible while local flank PI rejects that starting
point. Nonsmooth active-set transitions warn and are not globally linearized.

Trace decimation preserves endpoint metrics and the final sample. A synthetic
transient diagnostic explicitly checks that an alias warning survives when the
event is not retained in stored traces. A positive source projected through the
actual FEM heater map is accepted even when its modal coordinates are signed.
The configuration validates trace-allocation bytes before preallocating arrays.

The actual Streamlit app callbacks are exercised with `AppTest`: startup,
controller comparison, waveform simulation, configuration errors/preset loading,
and stale-waveform detection after plant or waveform settings change. The live
HTTP health endpoint was checked. No in-app browser backend was available for
visual verification; these are functional checks, not a screenshot/layout audit.
