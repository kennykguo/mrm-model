# Scope and accuracy audit

This delivery is a measurement-ready, independently implemented **simulation
plant**, not an identified Lightmatter digital twin. Accuracy is the primary
objective: provenance, physically related parameters, conservation tests,
numerical convergence and explicit validity limits take precedence over a
favorable-looking controller plot.

## Requirements and delivered interfaces

| Requirement | Implementation / evidence |
|---|---|
| Start independently of all existing `mrm-model` branches | Empty orphan `v1-3` worktree; no imported files, environment, tests or designs. [Boundary record](INDEPENDENCE.md). |
| Use the supplied paper for accuracy | All three pages read and rendered; source-specific facts, timing ambiguity, measurement resolution and unidentified parameters recorded in [SOURCES.md](SOURCES.md). |
| Account for the textbook, earlier learning work and capstone proposal | Read-only contextual review and [independent textbook comparison](TEXTBOOK_COMPARISON.md); no claims accepted merely because an earlier assistant stated them. |
| One interface for changing the plant, experiment and controller | Strict YAML configuration, CLI and matching Python API; per-path overrides, actual-model serialization, built-in and importable controllers. |
| Test multiple control algorithms | Hold, PID, scan/PI and dither; delayed/quantized observations, no hidden detuning; paired case and uncertainty studies. These are baselines, not a claim of novel or proprietary firmware. |
| Prioritize physical/numerical accuracy | Periodic passive optics; short finite-RC optical dynamics; statistical NRZ reduction; coupled optical/electrical heating; passive thermal network; explicit hardware/event timing; independent analytical and refinement checks. |
| Distinguish known, assumed, inferred and unavailable information | Resolved evidence labels, [parameter/state inventory](PARAMETER_INVENTORY.md), source ledger, validity restrictions and held-out calibration interfaces. |
| Explain intuitively and document disagreements | [Explanation](EXPLAINED.md) connects optical interference, heat flow, sensing and control; [textbook comparison](TEXTBOOK_COMPARISON.md) states reasons and primary references. |
| Use free tools and respect a 16 GB machine | Locked local Python runtime; sparse, guarded thermal extraction; short waveform windows; bounded optical tables; trace preflight and sequential studies. [Tool survey](TOOLS.md) distinguishes actually executed backends from possible alternatives. |
| Prepare for later device identification and team algorithm research | Bounded measured junction input, passive thermal operators, spectral/thermal fitting with independent evaluation, [measurement plan](MEASUREMENTS.md), and a controller plugin contract. |

## Three different accuracy claims

The v1 progress checkpoint passed **166 tests** on 2026-10-09, including the
regression that rejects a clipped scan peak even when the last point is not
clipped. The subsequent interactive control-lab revision passes **245 tests**,
including controller timing, configuration isolation, fixed-model identity,
real paired simulations and local HTTP boundaries. These counts document
checked revisions; they are not percentage accuracy or proof that every
possible operating point has been covered.

Three additional dependency-free JavaScript tests check chart tick formatting
(including fractional negative detuning). Browser checks covered actual runs,
failure warnings, playback, rotation, JSON downloads, light/dark appearance and
320–1400 px layouts. Run the formatting checks with
`node --test src/ringlab/lab_static/tick-format.test.cjs`.

**Equation/implementation verification:** tests check through/drop/loss power
balance, thermal energy balance and passivity, closed-form cavity/RC/thermal
limits, stationary NRZ moments against enumerated histories, shot-noise/filter
statistics, code/sample delay semantics, discontinuity left/right limits,
integrated OMA against independent quadrature, and actual-model replay.

**Numerical evidence on stated examples:**

- [Thermal extraction artifact](../artifacts/thermal_extraction.json): spatial
  refinement to 65,536 cells on an explicitly assumed slab. The finest pair
  changes self-heater gain by about 0.016%, optical-source gain by 0.67%, and
  the full gain matrix by 0.50%. These are differences between meshes, not
  device errors. Reduced-model optical-source errors and small temperature
  undershoots are retained and reported, not hidden.
- [Optical accuracy artifact](../artifacts/optical_accuracy.json): finite-RC
  substep refinement, independently seeded finite data windows, exact/static
  versus local dynamic discrepancies, and a new 10,000-point interpolation
  check. Its finite-window OMA statistics are explicitly not converged.
- [Integration accuracy artifact](../artifacts/integration_accuracy.json)
  compares lookup/direct evaluation and tightened
  ODE tolerances on a specified self-heating transient; source hashes and
  configurations accompany the generated report. Such checks must be repeated
  when changing operating regimes, especially near a thermal instability.
- [Independent solver comparison](../artifacts/solver_comparison.json): four
  fresh-process runs of a 0.2 ms, eight-lane case compare Radau and DOP853 at
  `rtol=1e-6` and `1e-8`. All recorded ADC/DAC codes match. The largest recorded
  temperature discrepancy from the refined Radau reference is under 0.1 µK;
  the reference is not exact truth. DOP853 is faster on this example, but no
  such speed or error guarantee is inferred for other stiffness or operating
  regimes.

The recorded extraction runs used approximately 245 MB peak process RSS for
the finest thermal study and 119 MB for the optical study on this machine.
Memory and runtime vary with model size and numerical environment; these are
observations, not a blanket 16 GB guarantee for arbitrary user configurations.

**Device validation:** not established. There is no open-loop measured
Lightmatter dataset here. Public closed-loop demonstrations do not uniquely
identify geometry, coupling, loss partition, heater efficiency, thermal poles,
sensor dynamics or the controller. Even both optical power ports can leave
multiple physical parameter branches. Synthetic fit recovery and agreement
between two codes are not independent device validation.

## Boundaries that affect controller conclusions

The long-run NRZ reduction assumes ideal electrical edges and independent bits;
short waveforms resolve finite RC. Its WDM cascade propagates mean powers, not
correlated multi-ring fields. In particular, an FSR collision can invalidate an
accurate-eye or precise-OMA interpretation. The supplied eight-lane context is
a test of assumptions, not a reconstruction of the public chip.

The loss-to-heat split, geometry and thermal coefficients remain assumed.
Backscatter/doublets, nonlinear free-carrier feedback, detailed RF/parasitics,
temperature-dependent extraction and a complete photonic link are not silently
substituted by fitted constants with purported universal validity. The full
inventory marks implemented, importable, extraction-only and deferred effects.

The public paper's BER is not reproduced. No calibrated receiver is present;
`ber` is explicitly unavailable. Detuning/temperature extrema and wrong-line
occupancy are sampled diagnostics, not continuous-time safety guarantees.
Monitor lock, correct wavelength identity, OMA and heater headroom are distinct
criteria. The comparison examples include acquisition and differing controller
objectives; they do not establish a universal controller ranking.

No FPGA implementation or synthesis is claimed. The proposal permits a
simulation model; the sampled software contract is a starting point for later
RTL/fixed-point verification. Full Meep, Femwell optical, DEVSIM or commercial
multiphysics extraction has not been run. The actual executed spatial backend
is the documented conservative thermal finite-volume workflow.

## Executed controller examples

### Minimal local-control lab

The new [interactive lab](CONTROL_LAB.md) holds the physical profile fixed and
compares a sign/step controller with a constant-code heater. Its
[complete numerical evidence](../artifacts/control_lab.json) and
[paired response plot](../artifacts/control_lab_response.png) are generated by
`uv run python examples/control_lab_checks.py`. The model's physical-parameter
SHA-256 is `97db6c0630932759fc48659a701c63e6d5691f8cdfcede307b674996efa9b092`;
the implementation, profile and environment fingerprints are recorded separately.

For the default 4 ms synthetic experiment, post-disturbance monitor RMS error
is 7.88 ADC codes with the step controller versus 29.73 with a fixed heater.
The window includes recorded released samples whose physical sample times
are at or after the 2 ms ambient step. The final monitor code is 260 versus
230; the target is 260. This result depends on the explicit known-fringe
prebias, 0.5 ms warmup, four-code step and assumed plant. It is not a general
performance or acquisition claim.

A refined DOP853 run at `rtol=1e-8` reproduces all 201 recorded ADC and DAC
values from Radau at `rtol=1e-6`. The maximum recorded ring-temperature
difference is 0.146 µK, detuning difference 2.19 kHz, and OMA difference
0.0143 nW. Those differences check this numerical example only.

The viewer rotates a **schematic** device and colors its ring, heater and
package from their distinct recorded thermal states. It does not generate
or claim a spatial heat field, extracted geometry, optical mode image or
private chip layout. The four plots explicitly separate interface-level
ADC/DAC values from evaluator-only detuning and OMA. Warmup and the
disturbance are marked; 20 µs recording does not expose every switching edge.

### Earlier acquisition and full-array examples

The [bundled run evidence](../artifacts/example_runs.json) contains complete
resolved configurations, controller settings, metrics, source/environment
fingerprints and the paired comparison results. Numerical traces and
[single-ring](../artifacts/reference_response.png) /
[eight-lane](../artifacts/wdm_response.png) plots accompany it.

The 4 ms single-ring reference acquired a red-side operating point. Its
time-averaged modeled OMA is about 28.87 µW including acquisition, with about
1.016 µJ of heater energy. These are synthetic-example outputs, not target
hardware specifications. Its saved-run replay reproduces all recorded values
bit for bit in the checked environment. An importable custom-controller run
and replay were also exercised.

The supplied paired study completed all 15 runs: three controllers across
three named cases and two uncertainty samples. Hold and dither begin far from
the useful fringe in this cold-start example; dither is a local tracker and
does not replace global acquisition. Therefore, the larger scan/PI mean OMA
is not evidence of general algorithm superiority. The small study demonstrates
the comparison workflow, not statistical robustness.

In the 30 ms eight-lane case with DOP853 at `rtol=1e-8`, every controller reports
acquisition, yet the final nearest-laser identities are `[0,1,2,3,0,1,2,3]`
instead of `[0,1,2,3,4,5,6,7]`. The assumed 15 µm diameter and group index of 4
give a 1.59045 THz FSR, close to four 400 GHz channel spacings. This is a
periodic-resonance identity risk under **assumed geometry and bus topology**,
not a demonstrated flaw in Lightmatter's hardware or a validated BER penalty.
The mean-field WDM approximation itself limits precise modulated-waveform
claims in this collision regime. Monitor acquisition is not wavelength
identification.

The full 30 ms Radau (`rtol=1e-6`) and independent DOP853 (`rtol=1e-8`) runs
have identical recorded ADC and DAC codes. Across all recorded lanes/times,
their maximum differences are 11.8 µK, 176.9 kHz detuning, and 0.738 nW OMA.
Both give the same final nearest-laser identities. This separates the observed
identity risk from this particular solver choice; it does not remove physical
model uncertainty or establish a universal convergence bound. Runtime was
about 844 s versus 218 s under the recorded machine load; the Radau process
peaked at approximately 183 MB RSS.

## Reproduce checks

```sh
uv sync --extra dev
uv run pytest -q
uv run python examples/optical_accuracy_checks.py
uv run python examples/integration_accuracy_checks.py
uv run python examples/solver_comparison.py
uv run python examples/thermal_extraction.py --max-refinement 16
uv run ringlab run examples/models/reference.yaml --dry-run
uv run ringlab run examples/models/reference.yaml
uv run ringlab compare examples/experiments/compare.yaml --dry-run
uv run ringlab compare examples/experiments/compare.yaml
```

The spatial and optical reports include their own relevant source fingerprints.
Saved controller experiments retain complete configuration, actual physical
models, controller specification, numerical environment and source hashes.
Rerunning after changes is a new comparison, not the old result relabeled.
