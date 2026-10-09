# A fixed plant, a small control problem

The first lab problem is deliberately narrow: keep a **measured drop-port
monitor code** near a specified target after a small ambient-temperature
disturbance, starting in an assumed known operating region. The algorithm does
not find that region, identify a laser wavelength, maximize OMA, or predict BER.
This is a conditional synthetic experiment, not a device-validation result.

Start the local viewer with `uv run ringlab lab` and open
<http://127.0.0.1:8765>. It requires no external JavaScript, cloud service, GPU
solver, or internet connection once the Python dependencies are installed.
Each run executes the existing Python plant twice, sequentially: a step
controller and a constant-code baseline. The browser plots those returned
traces; it does not implement a second approximation of the physics.

## Three separate layers

| Layer | What belongs here | What changing it means |
|---|---|---|
| Fixed model | Optical ring, junction, data averaging, heat network, heater and sensor interfaces, laser; explicitly resolved parameters and evidence | A different physical hypothesis; compare it as a new model revision |
| Controller | Monitor target, heater step, deadband; the rule that consumes permitted measurements | A different proposed solution on the same plant |
| Experiment | Initial heater prebias, warmup, ambient disturbance, duration, initial temperature, noise seed and numerical settings | A different test of that solution, not a new physical model |

The initial-code setting appears with controller setup because the controller
must also command it, but it is simultaneously the experiment's initial heater
condition. Both runs start with that same code, temperature and seed. They
receive the same ambient step. Their resulting temperatures, absorption,
photocurrents and state-dependent noise can diverge as their actions diverge.

The bundled `src/ringlab/lab_model.yaml` freezes all resolved defaults instead
of relying on future library defaults. The interface does not let controller
tuning mutate it. A physical-parameter hash identifies the coefficients; an
implementation fingerprint separately identifies the source revision. A matching
parameter hash alone is **not** proof of identical equations or dependencies.
The downloaded result contains the configuration and provenance needed to
inspect what was used. Keep older results when changing the model.

## The simplest feedback rule

On a known red-side fringe, more heater power lowers the measured drop signal.
Let `y` be the delayed ADC code, `r` the target, `d` the deadband and `s` the
heater-code step:

```text
if y > r + d:  heater_code += s
if y < r - d:  heater_code -= s
otherwise:    hold
clamp heater_code to the DAC range
```

There is one commanded-code state, no fitted plant model in the controller,
no derivative calculation, no scan, and no automatic setpoint selection.
Input/application guards hold the output during warmup, on stale or clipped
measurements, and until a delayed sample is demonstrably newer than the
observed application of the previous command. Those guards do not prove
thermal settling. The controller sees only the public sampled I/O contract;
true ring temperature, detuning and OMA remain evaluator-only.

The default prebias is 2660 on the assumed 12-bit DAC, the target is 260 ADC
codes, the step is four DAC codes, and the deadband is two ADC codes. The
experiment runs for 4 ms, holds the prebias during a 0.5 ms warmup, and changes
the ambient boundary by +0.25 K at 2 ms. These are chosen **synthetic test
settings**, not numbers measured on Lightmatter hardware. A steady-state
calculation on this model at code 2660 and 300 K ambient gives approximately
264 monitor codes on the intended fringe. That calculation informed setup;
it is not available to the feedback algorithm.

A larger step may respond faster but oscillate around the target; a wider
deadband accepts more error but can reduce toggling. The wrong initial fringe,
insufficient actuator headroom, an unreachable target, a large disturbance or
an incorrect feedback sign can make the controller fail. The UI intentionally
permits some failing settings. It must not label every completed run “locked.”
Monitored power is not an unambiguous measure of detuning or wavelength identity.

## Reading the plots and 3D schematic

- Monitor ADC and applied heater code are interface-level values. Monitor
  values are delayed, quantized samples held until another sample is released;
  the simulator's exact sample timestamps accompany the trace. The viewer
  records every 20 µs; transitions between records are not shown at their
  exact event times.
- Detuning and OMA plots are evaluator diagnostics, not controller inputs.
  OMA is a modeled optical modulation amplitude, not an eye diagram or BER.
- Solid and dashed traces compare the controller and fixed-heater baseline
  on shared axes. Startup is part of the experiment, not evidence of feedback
  acquisition. Warmup and the boundary disturbance are marked separately.
- Time playback replays recorded simulation results; it is not real-time
  hardware. Rotating, zooming or separating the schematic changes only the
  camera/presentation, never the model parameters.
- Ring, heater and package colors/readouts represent their respective
  simulated lumped temperatures. The package state is distinct from ambient.
  The geometry, layer spacing, heater placement, waveguide width and visual
  connections are schematic. There is no FEM temperature field, optical-mode
  field, TCAD extraction, physical heat-map interpolation or exact private
  layout hidden behind the rendering.

The 3D view helps connect actuator, ring and heat reservoir to the equations;
it adds no physical accuracy by itself. Accuracy remains limited by the
assumed model and missing measurements documented in
[the inventory](PARAMETER_INVENTORY.md) and [source ledger](SOURCES.md).

## Updating the model without moving the goalposts

1. Keep the current model file, parameter hash, source revision and baseline
   results as a named reference.
2. Record the new measurement or source and whether it identifies a parameter
   uniquely. Separate fitted traces from held-out validation traces.
3. Revise the physical profile, evidence labels and model identifier together;
   update the equations only when required by the evidence. Restart the server
   after any code or profile change.
4. Recheck conservation, numerical convergence and the held-out measurements.
   Then rerun the **same controller and experiment** on old and new models
   before retuning the controller.
5. Only after that comparison should controller tuning change. Record those
   settings separately so improved plots cannot be confused with a more
   favorable physical hypothesis.

For broader experiments and replayable on-disk runs, use the normal Python
API/CLI and `ringlab.experiments.save`; the small web lab deliberately exposes
only bounded single-ring settings. The local server binds to loopback and is
not intended for deployment as a multi-user or network-accessible service.

The default paired traces, configuration, source fingerprints and independent
solver comparison are bundled in [control_lab.json](../artifacts/control_lab.json).
Regenerate those and the [static comparison plot](../artifacts/control_lab_response.png)
with `uv run python examples/control_lab_checks.py` after code edits are complete.
