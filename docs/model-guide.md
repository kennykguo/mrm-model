# How the model works

This repository is a configurable physical plant for testing microring locking algorithms. It connects heat flow, periodic optical resonance, heater electronics, an optical monitor, and a sampled controller. The same `ExperimentConfig` is used by Python experiments and the command line, so a controller change need not change the plant or disturbance. The interface is suitable for later firmware/FPGA integration; this repository does not implement a deployed FPGA controller or claim correspondence to proprietary Lightmatter circuitry.

Public evidence determines a few constraints, including the O-band reference, eight transmitting wavelengths per direction on a 400 GHz grid, and approximately 15 GHz/K thermal sensitivity. The supplied presets are **synthetic and uncalibrated**. In particular, geometries, thermal capacities/conductances, electrical bandwidths, coupling, losses, sensor scale, and controller tuning are hypotheses. See [research.md](research.md) for sources and [parameter-inventory.md](parameter-inventory.md) for every configurable field.

## The physical chain

Think of a ring as a loop that repeatedly returns a portion of the light to its coupler. At the right frequency, those returns reinforce each other. Light stored in the loop changes the through-port output and produces a drop-port monitor peak. Heating changes the resonant frequency. A digital controller therefore changes a heater code, waits for electrical and thermal response, reads a filtered/quantized photocurrent, and changes the code again.

```text
DAC code -> delayed heater driver -> actual electrical power -> coupled heat states
                                                              |
lasers + data bias -> periodic ring spectrum <-----------------+
                         |                    |
                    through data        drop monitor -> filter/ADC -> controller
                         |
                  absorbed heat -> heat states
```

Boundary temperature, local external heat, and laser drift enter different points in that chain. A package temperature step must flow through the thermal boundary conductances; it does not instantly change all stored ring temperatures. A laser frequency step changes detuning directly. That distinction matters when comparing feed-forward, observer, or dither controllers.

## Periodic optics for acquisition and thermal tracking

`optics.optical_response()` implements a complex add-drop ring with lossless reciprocal couplers and a uniform ring whose drop coupler is halfway around. With power coupling fractions `K1`, `K2`, self-coupling amplitudes `t1=sqrt(1-K1)`, `t2=sqrt(1-K2)`, round-trip field attenuation `a=exp(-alpha*L/2)`, and phase `phi`, the feedback denominator is:

```text
D = 1 - a*t1*t2*exp(i*phi)
Hthrough = (t1 - a*t2*exp(i*phi)) / D
Hdrop = -sqrt(K1*K2*a)*exp(i*phi/2) / D
```

The returned fields have units sqrt(W), powers W, and integrated cavity energy J. Intrinsic dissipation is split into absorbed heat and nonlocal scattering using `intrinsic_absorption_fraction`. A scattering loss does not automatically heat the modeled ring. Passive energy accounting is a testable constraint: through + drop + dissipation equals input under these static assumptions.

The phase uses frequency offset times round-trip group delay, with optional quadratic phase dispersion. Stored energy uses its local derivative `dphi/domega`, and nonpositive delay rejects an invalid extrapolation. Resonances repeat at constant FSR when dispersion is zero; nonzero dispersion changes the spacing. This is why broad acquisition scans and WDM aliases can be simulated without pretending one Lorentzian represents an entire free spectral range. Independent parameters are length, group index, loss, and coupling. FSR, decay rates, loaded Q and linewidth are derived; users cannot set an inconsistent Q and FSR independently.

Temperature and signed junction voltage move the reference resonance. Voltage-dependent frequency shifts and losses can be supplied as measured tables; the model rejects voltage extrapolation outside a table. The temperature coefficient is currently linear, and couplers/losses are not arbitrary temperature/frequency tables. Calibrating within a bounded operating domain matters more than increasing the number of unsupported coefficients.

`simulation.bus_response()` sends **every carrier through every ring in physical order**, including periodic off-target resonances. The monitor at a ring sums dropped power from all carriers, and `alias_fraction` records the contribution from nonassigned carriers. Laser beats are treated as incoherent at monitor bandwidth. Each ring's NRZ rails are averaged according to `data_one_fraction`; the assigned lane's final two powers retain the actual voltage/bit identity. This is an independent duty-average approximation, not a simultaneous WDM bit-pattern simulation or a coherent reflection network.

## Heat flows through a passive network

`ThermalNetwork` evolves temperature-rise coordinates relative to a **fixed** reference temperature:

```text
C dx/dt = -G*x + B*heater_power + E*(boundary_temperature - Tref) + external_heat
ring_temperature = Tref + H*x
```

`C` stores heat capacity, `G` describes heat leaving a state and flowing between states, `B` deposits heater power, and `H` reconstructs ring temperature. Separate `heater_temperature_map` values let heater resistance respond to the heater strip's temperature rather than forcing it to equal the optical waveguide temperature. The default network has one state per ring, one per heater, and a shared substrate, with nearest-ring conductance and a substrate-to-boundary path.

The nodal-network validation requires positive-definite capacitance/conductance, reciprocal nonpositive off-diagonal conductances, consistent boundary equilibrium, and nonnegative heat/temperature maps. A generalized passive FEM reduced model can instead use full matrices, signed modal maps and `reference_state_map` with `physical_network=False`. Those coordinates are not individual physical-node temperatures.

For frozen forcing, a matrix-exponential step is exact for the configured linear heat network. In the connected experiment, heater resistance and absorbed optical power depend on temperature, so their forcing is refreshed at `physics_substeps` within each control interval. This coupling is a finite-step approximation; increase substeps until the quantities you care about converge. Exact linear stepping does not mean the whole nonlinear electrothermal experiment is exact.

The optional `extraction.thermal_fem_reference()` assembles a **generic** cross-sectional heat problem with scikit-fem and produces a passive reduced model. `thermal_fem_convergence()` checks refinement and full-versus-reduced transients. A finite Robin boundary preserves boundary-temperature memory with consistent FEM heat capacity. Geometry, materials, contact and effective extrusion length are declared assumptions. The optional `mode_extraction` module actually runs Femwell's full-vector Maxwell eigenproblem, tracks the same mode with field overlap, and checks mesh, domain and wavelength-derivative convergence. These are real FEM-to-coefficient workflows, not geometric reconstructions of Lightmatter's device; [extraction.md](extraction.md) records their results and omissions.

## Electronics and sampling are part of the plant

`HeaterDriver` rounds/clips DAC commands, applies gain/offset, schedules changed commands by actual timestamps, and advances a first-order driver with an optional slew limit. Resistance follows a linear TCR law; voltage/current/power compliance limits bound available heat. The driver exposes both endpoint power and the interval-average power obtained from its voltage-squared integral. The thermal solver uses the average, so a settling transition does not act like full endpoint power for the entire interval. Resistance is held fixed within that substep, and per-update voltage noise is a standard deviation, not a white-noise spectral density.

`MonitorADC` converts dropped power to photocurrent, adds dark current, advances a one-pole analog filter, and includes filtered electronics and optional shot noise. Noise density is one-sided A/sqrt(Hz); a one-pole 3 dB bandwidth B has equivalent noise bandwidth pi*B/2. It samples and quantizes the filtered current, records clipping, and delivers it after configured latency. A repeated delayed sample keeps its original acquisition timestamp. This does not include a separate boxcar integration window, ADC aperture jitter, INL/DNL, or arbitrary measured sensor filter.

The controller sees `Observation(time_s, sample_time_s, adc_codes, valid, clipped)`. The simulator's true temperature, detuning and optical margins are logged for evaluation but are absent from the online observation. This separation is what lets a new algorithm be tested through the same eventual hardware boundary.

## Controller comparison and its limits

All built-in algorithms implement `step(observation) -> one DAC code per ring`:

| Algorithm | Objective and knowledge | Important limit |
| --- | --- | --- |
| `open_loop` | Hold initial codes | Establish uncontrolled disturbance response |
| `pi` | Hold a calibrated monitor fraction on a selected flank; slope fixes feedback sign; anti-windup limits integration | Equal readings can occur on either side; power drift can mimic detuning; local controller does not globally acquire a resonance |
| `dither` | Estimate monitor gradient by lock-in demodulation, remove the probe harmonic and track the same flank with adaptive PI; optional peak/slope seeking | Flank still needs branch calibration; probe amplitude, phase and tones need tuning for plant delay and array interference |
| `mpc` | Use a local discrete A/B/C model, sensor-space observer and bounded finite-horizon least-squares optimization | A local linear prediction around calibrated conditions; nonlinear excursions, slew/compliance and unknown disturbances can invalidate it |

`acquisition=True` wraps an algorithm with a sequential measured lane scan and branch-aware target crossing. The scan can still select the wrong WDM alias, or fail if its grid/dwell/range is inadequate. It is a baseline acquisition routine, not a complete industrial recovery state machine.

`nominal_context()` constructs offline calibration from the configured nominal model. It includes heater TCR and absorptive feedback, rejects a locally unstable initial electrothermal Jacobian, and includes local driver settling and analog-filter dynamics. Its actuator derivative respects the active voltage/current/power limit: a strictly limited command has zero local DAC authority; current-limited power rises with resistance while constant-power compliance removes that settled TCR dependence. Driver target-temperature sensitivity is separate from fixed-voltage thermal feedback. ADC clipping similarly gives a clipped target and zero local measurement slope, so PI rejects an unobservable starting point. Explicit nodal/modal initial states retain the accepted heater bias when calibration is constructed. Thresholds and instantaneous output projection are nonsmooth and warn; a local linear model does not predict arbitrary constraint crossings. Command transport is represented by input-history states and an exact fractional-interval split; delivered ADC timing uses discrete output-history states. The bundled dense MPC caps delay at 32 sample periods and the augmented model at 512 states; the plant still supports larger delays with another controller. Slew/compliance remain nonlinear plant constraints rather than exact global MPC predictions. Nominal calibration is allowable preidentified controller knowledge, not online access to evolving truth. For a transfer study, provide a context built from measured data or a different nominal device. Observer covariance constants remain fixed in this baseline controller.

`compare_controllers()` reinitializes physical state and independent random streams for each algorithm. Controllers therefore receive reproducible disturbances and noise draws, although shot-noise amplitude depends on their resulting photocurrent. Feedback controllers target the same initial calibrated flank by default. Selecting dither's optional `peak` or `slope` objective changes the task and emits a warning. Default 10-code dither probes are small for the illustrative tuning gain; a large probe can move the resonance across the entire nonlinear line shape.

`study.corner_study()` perturbs jointly specified parameters while retaining nominal controller calibration and the assigned target. Initial error in a changed device is therefore observable rather than silently redefined to zero. Metrics accumulate at controller endpoints even when stored traces are decimated; they do not certify a continuous-time maximum between samples. The detuning score tracks the specified unwrapped resonance order. A transition to another longitudinal order counts as an excursion in this score even when the same assigned wavelength remains usable; inspect periodic spectra, OMA and nonassigned-carrier contributions separately for channel-level success.

## Short waveform mode for fast physics

Thermal experiments average fast data. `waveform.simulate_nrz()` separately resolves a supplied binary sequence: signed voltage rails -> first-order junction RC -> local cavity fields -> illustrative one-pole receiver power filter. `simulate_cavity()` accepts arbitrary time arrays and supplied frequency/temperature/voltage/power histories. The sample count is bounded; optical carrier oscillations are not stored.

The fast model uses energy-normalized temporal coupled-mode theory. The forward/reverse fields have units sqrt(J); their squared magnitudes sum to cavity energy. Intrinsic and external energy-decay rates follow the static ring parameters, and field decay uses half the energy rate. A complex reciprocal backscatter rate can split modes and produce reverse/reflected power. Linear constant-coefficient intervals have an exact propagator. Time-varying input coefficients are held per interval; NRZ electrical voltage uses an exact midpoint of its RC evolution.

Optional `NonlinearParams` add effective Kerr shift, two-photon absorption, a uniform carrier population, free-carrier absorption and dispersion. Nonlinear intervals use scaled adaptive integration. `nonlinear_steady_states()` enumerates single-mode branches, checks residuals and the full field/carrier Jacobian, and reports stability; `select_nonlinear_branch()` requires history if multiple stable branches exist. This branch analysis does not include thermal states or backscatter.

These coefficients are disabled by default and need wavelength/process/power calibration. Fast waveform mode receives temperature externally; it does not integrate the thermal network. Slow experiments include static absorptive heating but do not couple the optional carrier/Kerr/TPA waveform dynamics back into thermal control. Local TCMT also differs from the exact periodic transfer away from a high-finesse isolated resonance. Check detuning/FSR validity and converge waveform resolution.

The output eye metrics summarize center-sampled levels, OMA, extinction and percentile eye opening. `gaussian_ber_proxy` is explicitly a Gaussian approximation using declared receiver noise and pattern spread; it is neither simulated bit errors nor a validated prediction of the paper's BER. The slow experiment leaves `ber=None`. There is no complete laser phase/RIN/polarization model, SerDes/CDR, equalizer, FEC, fiber channel, interleaver or coherent on-chip reflection topology.

## Running and extending experiments

After installing the package, one editable configuration drives these workflows:

```bash
mrm preset --rings 8 --output examples/eight_ring.json
mrm compare --config examples/eight_ring.json --output outputs/eight-ring-comparison
mrm spectrum --config examples/single_ring.json --output outputs/spectrum
mrm waveform --config examples/single_ring.json --bits 511 --rate 5e10 --samples-per-bit 32 --output outputs/waveform
```

The preset command writes its named target; use a new filename when preserving a modified configuration. Runs export the configuration, traces, metrics and warnings. Resource limits cap control steps/recorded rows; record decimation reduces storage without changing metric accumulation. Short fast runs have a separate sample-count cap. Refinement studies should change only the numerical resolution and confirm physical predictions stabilize.

In Python, a custom controller and independently calibrated context can use the same plant:

```python
from mrm_model.config import ExperimentConfig
from mrm_model.simulation import run_experiment

class MyController:
    def __init__(self, initial_codes):
        self.codes = initial_codes.copy()

    def step(self, observation):
        # Replace this hold with an algorithm using adc_codes and timestamps.
        return self.codes.copy()

config = ExperimentConfig.load("examples/single_ring.json")
# run_experiment(config, controller=MyController(...), context=measured_context)
```

Replace physical coefficients using measured spectra, actual heater power steps, interface timing and crosstalk records. Validate on separate conditions/devices. The exact fit APIs, CSV schemas, covariance and identifiability caveats are described in [calibration.md](calibration.md); [lightmatter-data-request.md](lightmatter-data-request.md) lists the highest-value data to obtain.

## Concrete differences from the textbook

The textbook remains useful for intuition and control tradeoffs. Here the add-drop feedback product is `a*t1*t2`, including both couplers; periodic spectra replace local Lorentzian widgets for alias studies; arbitrary signed fractional powers of carrier change are replaced by calibratable signed voltage-response maps; rail identity is preserved in data waveforms; fixed-reference thermal storage prevents instantaneous ring-temperature jumps from ambient changes; and nominal time constants arise from C/G eigenmodes rather than an assumed pair copied as device facts.

The model also avoids treating a 2.8 THz channel span as a universal minimum FSR: it evaluates every carrier at every resonance order. I also disagree with treating a confinement-weighted bulk thermo-optic estimate as a rigorous bound on `dn_eff/dT`. Our generic vector calculation with explicitly chosen material coefficients yields about `2.04e-4/K`, exceeding the chosen silicon material derivative `1.86e-4/K`; the effective mode derivative has different weighting and normalization. That calculation still does not explain or replace the paper's reported device sensitivity. The [full-vector thermal analysis by Payne et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11897098/) derives the distinction for strongly guiding waveguides. G25 device estimates, the G26 controller demonstration, later presentation results, product generations and capstone extrapolations retain different provenance; see [research.md](research.md).
