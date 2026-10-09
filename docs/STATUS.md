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

The full automated suite passed **166 tests** on 2026-10-09, including the
regression that rejects a clipped scan peak even when the last point is not
clipped. This count documents the checked revision; it is not a percentage
accuracy or a proof that every possible operating point has been covered.

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
