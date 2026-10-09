# Microring control laboratory

A configurable microring modulator plant for the Lightmatter capstone: compare locking algorithms through the same ADC/DAC interface, vary the device and disturbances, and replace assumptions with measured or offline-extracted coefficients.

The implementation uses free, open-source solvers. It is physically verified and **not yet calibrated to Lightmatter hardware**. Public information fixes some operating constraints, but does not identify the complete device, thermal network or electronics. The default profiles state this explicitly; the simulator does not manufacture a reproduction of the paper's BER.

## Start the lab

The environment is already installed in this workspace. Run:

```sh
cd /home/kennykguo/mrm-model
.venv/bin/streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open `http://127.0.0.1:8501`. The lab provides controller comparison, spectra, short NRZ waveforms, calibration uploads and thermal FEM extraction. Sidebar controls create an experiment; the editable JSON contains every shared plant, hardware, disturbance and algorithm setting. Results are tied to the configuration that produced them.

For a fresh checkout, install [uv](https://docs.astral.sh/uv/) and use the checked-in lockfile:

```sh
uv sync --python 3.12 --extra lab --extra physics --extra dev --extra validation
```

Only NumPy/SciPy are required for the compact runtime. The optional full-vector Femwell workflow has a small, separately documented installation in [extraction.md](docs/extraction.md); it avoids unnecessary CAD/VTK dependencies. All solvers have open-source licenses; none of these workflows requires paid solver access.

## One configuration, several algorithms

```sh
.venv/bin/mrm compare --config examples/single_ring.json --output outputs/comparison
.venv/bin/mrm compare --config examples/eight_ring.json --controllers open_loop pi mpc --output outputs/eight_ring
.venv/bin/mrm run --config examples/fem_ring.json --controller pi --output outputs/fem_ring
.venv/bin/mrm sweep --config examples/single_ring.json --corners examples/corners.json --output outputs/corners.json
.venv/bin/mrm waveform --config examples/single_ring.json --bits 511 --rate 5e10 --output outputs/waveform
```

`open_loop`, flank PI, adaptive dither and local MPC share the controller protocol. By default all feedback controllers target the same calibrated flank; dither can optionally target a peak or gradient. Acquisition is a configurable measured scan. Controllers see codes and timestamps, not simulator-only temperature/detuning. The plant logs that truth separately to score tracking, acquisition/excursions, heater energy, ADC clipping, OMA and wrong-channel monitor contributions.

The corner study keeps nominal calibration and assigned detuning fixed as the device changes. Its ranges are explicit illustrative cases, not invented manufacturing statistics. CLI outputs include the full configuration, metrics/warnings, CSV and compressed traces. Recording limits protect memory; decimation does not suppress metric accumulation at controller endpoints.

The same interface accepts your own controller:

```python
import numpy as np
from mrm_model.config import ExperimentConfig
from mrm_model.simulation import run_experiment

config = ExperimentConfig.load("examples/single_ring.json")

class MyController:
    def __init__(self):
        self.codes = np.array(config.run.initial_codes)

    def step(self, observation):
        # Implement your policy using observation.adc_codes and acquisition time.
        return self.codes.copy()

result = run_experiment(config, controller=MyController())
result.save("outputs/my_controller")
```

Use an independently identified `ControllerContext` for sim-to-real studies; the nominal model need not equal the perturbed plant. This is a software boundary for later FPGA integration, not an implemented RTL or deployed Lightmatter controller.

## What is modeled

- Exact periodic complex add-drop optics, with the single-bus limit, integrated cavity energy, absorption/scattering separation, local group-delay dispersion, voltage calibration tables and derived FSR/Q/lifetime.
- Reciprocal passive thermal C/G networks, distinct heater/ring/substrate temperatures, thermal crosstalk, arbitrary passive ROM imports and absorptive optical heating.
- DAC quantization, TCR, settling, slew/compliance and actual transport queues; filtered photodetection, correlated filtered noise, shot noise, ADC quantization/clipping and acquisition/delivery timestamps.
- Every WDM carrier traversing every ring, including adjacent resonance orders and nonassigned monitor light. Long-run data is duty averaged; it is not joint WDM bit simulation.
- Separate short TCMT waveforms with electrical RC, counterpropagating backscatter, optional Kerr/TPA/FCA/FCD/carrier dynamics, branch/stability analysis, receiver filtering and eye metrics.
- Positive/bounded spectral and thermal-step identification with withheld-sample residuals and identifiability diagnostics. Gaussian BER is a clearly labeled proxy; slow-run BER is absent.

The [intuitive model guide](docs/model-guide.md) explains the equations, assumptions and concrete differences from the supplied textbook. The [parameter inventory](docs/parameter-inventory.md) lists inputs, parameters, states and missing conditional physics. [Research and sources](docs/research.md) distinguish the supplied OFC paper, newer Lightmatter slides, generations, tool choices and uncertainty. [Calibration](docs/calibration.md) and the [Lightmatter data request](docs/lightmatter-data-request.md) describe the path to device accuracy.

## Accuracy evidence

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check src app.py scripts tests
.venv/bin/python scripts/validate_sax.py --solver sax --output results/sax-validation
.venv/bin/mrm extract-thermal --modes 32 --output outputs/thermal_fem
```

Recorded independent SAX component-network agreement is about `3e-16 W`. The generic thermal FEM coarse/fine gain change is about 0.5%; its passive ROM transient error is about 0.6% of final temperature rise. Full-vector optical extraction converged in mesh, wavelength derivative and boundary distance, with about 757 MiB measured peak RAM. Reports are included under [results/sax-validation](results/sax-validation/report.json), [thermal_fem.json](examples/thermal_fem.json) and [waveguide_modes.json](examples/waveguide_modes.json).

These checks prove numerical consistency for the stated assumptions. They do not determine undisclosed Lightmatter coefficients or prove hardware accuracy. Coupling, losses, junction response, thermal/package behavior and monitor/driver timing need measured calibration. The remaining limitations and validity of each check are explicit in [validation.md](docs/validation.md).

The [delivery audit](docs/completion-audit.md) maps the original requirements to implemented artifacts and current checks, including the 103-test final run and remaining calibration/visual-verification limits.

`examples/fem_ring.json` demonstrates changing the model through the same interface: it combines the recorded generic thermal ROM and vector-derived group index, with a synthetically aligned resonance. Ring coupling/loss and electronics remain assumptions; its geometry is not a reconstructed G26 device.
