# RingLab

An independently developed microring-modulator simulator for the Lightmatter
thermal-stabilization capstone. Change the device, disturbances, or controller
through the same YAML configuration and Python API.

Accuracy is the design priority. The optical equations, thermal network,
numerical solver, and controller interface have separate checks. Public
Lightmatter numbers supply context and constraints; unknown device details
remain explicit assumptions. This is a research plant that can be calibrated,
not a validated copy of Lightmatter's chip. It does not invent BER from OMA.

The `v1-3` branch starts from an empty history and contains no files imported
from earlier `mrm-model` branches. See [development boundary](docs/INDEPENDENCE.md).

## Run

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). The committed lockfile
records exact dependency versions and hashes; there are no paid runtime services.

```bash
uv sync --extra dev
uv run ringlab run examples/models/reference.yaml --dry-run
uv run ringlab run examples/models/reference.yaml
uv run ringlab inspect examples/models/reference.yaml
uv run pytest -q
```

A run saves the complete resolved configuration and evidence labels, actual
thermal and junction models, source and dependency fingerprints, metrics,
compressed numerical traces, and a response plot under `runs/`. Replay uses
those physical models, including caller-supplied calibrated replacements:

```bash
uv run ringlab rerun runs/<run-directory>
```

Changing model source, numerical dependencies, Python or platform blocks a
strict replay. Use
`--allow-changed-model` to explicitly record a comparison with changed code.
Editing source during a Python session requires a fresh process and simulation
before saving; a run cannot silently be labeled with newly edited source.

## Try a different model or algorithm

```bash
uv run ringlab run examples/models/reference.yaml \
  --set heater.bits=10 --set sensor.latency_s=4e-6
uv run ringlab run examples/models/reference.yaml \
  --set controller.name=dither \
  --set 'controller.parameters={initial_code: 1500, amplitude_codes: 16}'
uv run ringlab compare examples/experiments/compare.yaml --dry-run
uv run ringlab compare examples/experiments/compare.yaml
```

The comparison uses the same device cases and noise seeds for each controller,
including named corners and stratified uncertainty samples. It runs cases
sequentially and saves each before proceeding. The supplied two-sample study
is a short workflow demonstration, not statistical evidence of robustness.

Python uses the same schema:

```python
from ringlab.config import load
from ringlab.plant import simulate
from ringlab.experiments import save

config = load("examples/models/reference.yaml", {
    "ring.absorption_fraction": 0.6,
    "simulation.seed": 42,
})
result = simulate(config)
directory = save(result)
```

Custom controllers implement `reset(interface)` and `step(observation)` and
return an integer heater-code vector. Configure `name: package.module:Class`,
or pass an object to `simulate`. An observation contains delayed ADC codes,
their timestamp, clipping flags, applied heater codes, and a quantized package
temperature. It contains no actual detuning, optical power, or true ring
temperature. See [controller example](examples/custom_controller.py).

Built-in objects are saved using their actual constructor settings. An arbitrary
injected custom object is usable but cannot be saved without an explicit
`controller_configuration={"name": "package.module:Class", "parameters": {...}}`
declaration. Prefer creating custom controllers through the configuration.
Plugin replay fingerprints the entry/factory source and declared constructor
parameters; imported helpers, external resources and pre-existing object state
are not automatically captured. A plugin's `reset` must establish its complete
initial state. These checks support reproducibility, not universal bitwise
portability across hardware and numerical libraries.

## What is modeled

- Exact passive per-pass through/drop ring optics, periodic resonance orders,
  dispersion, loss, and the heat-producing share of loss.
- Thermal resonance shift and bounded analytic or measured junction response.
  Dependent linewidth and FSR follow from the independent ring parameters.
- Photon storage: short finite-RC waveforms, and statistical NRZ cavity moments
  for slow thermal experiments. WDM uses a documented near/far approximation,
  not an exact broadband modulated circuit solution.
- A passive multi-node thermal network with distinct heaters, optical
  absorption, local aggressors, crosstalk, package, and ambient boundary.
  Optical absorption is recomputed inside the adaptive integration, so
  self-heating feeds back on the resonance.
- Temperature-dependent heater resistance, voltage settling, compliance,
  DAC quantization, latency, photodiode dark/shot noise, readout filtering,
  ADC clipping/quantization, sample latency, and dropped measurements.
- Junction leakage and switching heat, plus explicit assumed driver heat.
- Hold, PID, scan-and-PI, and dither baselines, sharing the same interface.

The eight-lane example combines published architecture with assumed devices.
Its FSR may put unwanted resonance orders close to other lanes. Such failures
are results to investigate, not reasons to silently change the model.

The supplied 30 ms eight-lane example was cross-checked with two adaptive
solvers. The faster checked invocation is:

```bash
uv run ringlab run examples/models/wdm_context.yaml \
  --set simulation.method=DOP853 --set simulation.rtol=1e-8
```

That check applies to this configuration; changed stiffness or operating
regimes need new convergence checks. See the [accuracy audit](docs/STATUS.md)
for full results, runtime, memory, and the observed wavelength-identity risk.

## Read the model and its evidence

- [Intuitive explanation](docs/EXPLAINED.md)
- [Primary sources and what is actually known](docs/SOURCES.md)
- [Optical equations and approximation domains](docs/OPTICAL_MODEL.md)
- [Spectral fitting and identification ambiguities](docs/OPTICAL_CALIBRATION.md)
- [Thermal equations, extraction, and calibration](docs/THERMAL_MODEL.md)
- [Comprehensive parameter/input/state inventory](docs/PARAMETER_INVENTORY.md)
- [Independent comparison with the textbook](docs/TEXTBOOK_COMPARISON.md)
- [Open and commercial tool tradeoffs](docs/TOOLS.md)
- [Current completion and validation audit](docs/STATUS.md)
- [Measurements that would improve device accuracy](docs/MEASUREMENTS.md)

Offline thermal extraction solves an explicitly stated, assumed layered slab
with sparse finite volumes, studies mesh convergence, and checks a reduced
model against the spatial solution. Agreement verifies that numerical workflow;
it does not establish the geometry of a proprietary chip.

For actual device accuracy, replace the synthetic parameters using open-loop
spectra, bias sweeps, and heater/monitor transient traces. Fit and evaluation
traces must be separate. A compact model fitted to representative measurements
can be more relevant to this controller task than a more detailed simulation of
an incorrectly guessed geometry.

## Practical limits

The solver prioritizes accuracy over throughput. Begin with a short dry run;
the paper's 31-hour demonstration is not a duration to launch blindly. Trace
memory is checked before allocation, and repeated cases execute serially.
Waveform mode is for short high-speed windows, not every bit of a thermal run.

No measured Lightmatter transient dataset is included. Backscatter, detailed
semiconductor geometry, temperature-dependent material extraction, packaging,
and receiver BER remain limited by missing data or absent submodels. The
inventory documents each limit rather than treating an unused parameter or an
unsupported effect as modeled. All reported controller outcomes are conditional
on those limits.
