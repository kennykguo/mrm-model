# Inputs, parameters, states, and limits

This is a comprehensive engineering checklist for an MRM thermal-control simulator, not a claim that every microscopic effect in a fabricated transceiver is resolved. A parameter can be implemented and still be unknown for Lightmatter. Except for the explicitly identified public context in [SOURCES.md](SOURCES.md), the shipped device values are assumptions. There is no bundled measured Lightmatter calibration dataset.

Status meanings:

- **Modeled:** an executable behavior or parameter in the shipped compact simulation.
- **Importable:** an explicit Python API accepts an identified model/table/operator; the required dataset is not bundled. This does not imply an arbitrary vendor file can be loaded directly.
- **Extraction-only:** the spatial/short-waveform tool resolves the item, but it is not automatically coupled into every long thermal run.
- **Deferred:** no implemented behavior; the reason and evidence needed are stated.

All runtime dimensional fields use SI units. `*_per_m` loss means a **power** attenuation coefficient in Np/m; optical couplings are **power fractions**, while complex optical amplitudes have their own normalization. Temperature states are absolute Kelvin; temperature differences in kelvin equal differences in Celsius. `bias_v` is a nonnegative reverse-bias magnitude. Public rates quoted in GHz/K are converted to Hz/K before use.

## Runtime configuration inventory

The authoritative complete numeric default set is emitted by `Configuration.resolved()` and saved in each run's `resolved.yaml`. The tables below identify every physical input group and its meaning; they avoid duplicating a second numeric default set that could diverge from code.

| Section and keys | Role and units | Status / evidence |
|---|---|---|
| `ring.reference_wavelength_m`, `reference_temperature_k` | Cold/defined reference resonance and temperature. Per-ring cold frequency is constructed from the array grid and offsets. | Modeled. 1310 nm is published band context; exact resonance/reference temperature assumed. |
| `ring.roundtrip_length_m`, `group_index` | Optical length and local group delay; derive FSR and round-trip time. | Modeled. No measured G26 values. The eight-lane example uses a length inferred from the approximate T26 diameter; group index stays assumed. |
| `ring.intrinsic_loss_per_m` | Internal propagation power loss at the reference condition. | Modeled, assumed; includes absorption and scattering, which are not interchangeable. |
| `ring.input_coupling_power`, `drop_coupling_power` | Two independent couplers. | Modeled, assumed; do not replace with one measured loaded Q. |
| `ring.absorption_fraction` | Fraction of internal optical loss deposited as modeled heat. | Modeled, assumed. Drop-port power is not ring absorption. |
| `ring.thermal_frequency_slope_hz_per_k`, `thermal_frequency_curvature_hz_per_k2` | Effective resonance-frequency law about the reference temperature. | Modeled. Slope magnitude constrained by P26; sign inferred; curvature assumed. Neither proves a broad temperature fit. |
| `ring.beta2_s2_per_m` | Second-order optical dispersion in the periodic static phase. | Modeled in exact static optics; not a dispersive time-domain propagation model. |
| `array.count`, `spacing_hz` | Number and spacing of same-direction laser lines/rings. | Modeled; eight and 400 GHz are P26 context. Single-ring default is a reduced experiment. |
| `array.cold_offset_hz`, `resonance_errors_hz` | Common cold detuning and optional per-ring fabrication/calibration offsets. | Modeled, assumed; offsets are relative to each assigned laser line. |
| `array.bus_loss_db` | Lumped through-bus loss after each ring. | Modeled, assumed; location and reference plane are explicit in the bus calculation. |
| `junction.built_in_voltage_v`, `reference_voltage_v` | Reference for assumed square-root depletion behavior. | Modeled, assumed; this is not a Poisson/drift-diffusion solution. |
| `junction.zero_bias_capacitance_f`, `parasitic_capacitance_f`, `series_resistance_ohm` | Lumped voltage-dependent junction C plus parasitic C and series R. | Modeled in finite-RC waveform calculations; stationary NRZ averaging uses ideal electrical switching. |
| `junction.frequency_shift_scale_hz`, `loss_change_scale_per_m` | Voltage-induced resonance and internal-loss changes relative to reference bias. | Modeled, assumed; loss change may be negative, but total loss must remain passive. |
| `junction.leakage_current_a` | Reverse leakage and associated voltage-times-current power. | Modeled, assumed. Temperature dependence is not a TCAD leakage model. |
| `junction.minimum_voltage_v`, `maximum_voltage_v` | Hard validity bounds. | Modeled. Outside-domain requests are errors, not silent extrapolation. |
| `thermal.waveguide_capacity_j_per_k`, `heater_capacity_j_per_k`, `package_capacity_j_per_k` | Thermal inertia of separate local ring/heater nodes and common package. | Modeled, assumed. |
| `thermal.heater_waveguide_conductance_w_per_k`, `waveguide_package_conductance_w_per_k`, `heater_package_conductance_w_per_k` | Reciprocal internal heat-flow paths. | Modeled, assumed. |
| `thermal.neighbor_conductance_w_per_k`, `package_boundary_conductance_w_per_k` | Nearest-neighbor heat transfer and package-to-boundary path. | Modeled, assumed. Longer-range coupling also arises through the shared package. |
| `thermal.aggressor_package_fraction` | Heat split between package and adjacent local heater node. | Modeled, assumed source geometry; not measured by P26's ramp rate. |
| `heater.resistance_ohm`, `resistance_temperature_coefficient_per_k`, `reference_temperature_k` | Electrical resistance versus the heater node's own temperature. | Modeled, assumed. A tuning curve alone cannot identify all three. |
| `heater.voltage_max_v`, `current_max_a`, `power_max_w` | Independent terminal voltage, current, and power limits; heating is one-sided. | Modeled, assumed; headroom is a physical requirement. |
| `heater.bits`, `settling_time_s`, `latency_s` | DAC code range, analog settling, and actuator delay. | Modeled, assumed; no converter precision is inferred from a plotted DAC axis. |
| `data.bias_v`, `swing_v`, `mark_probability` | Binary levels are bias ± swing/2; probability controls average illumination and switching statistics. | Modeled, assumed operating point/probability. |
| `data.bit_rate_hz`, `averaging` | Rate and `stationary_nrz`, `quasistatic`, or `cw` optical reduction. | Modeled. 50 Gb/s is P26 link context. Mode selection changes the physics approximation. |
| `data.junction_heat_fraction`, `driver_power_w` | Fraction of estimated junction switching/leakage heat deposited at each waveguide; separate per-lane driver power deposited at the package. | Modeled, assumed heat placement. Differential capacitance is integrated over the data swing; iid transition statistics determine average electrical heat. |
| `laser.power_per_line_w` | Input at the modeled bus reference plane per line. | Modeled, assumed; not laser-module output before unknown coupling loss. |
| `laser.power_fractional_noise_rms`, `frequency_noise_rms_hz` | Sample-and-hold random disturbances at the sensor cadence. | Modeled scenario noise; these are not an optical linewidth or calibrated continuous RIN/phase-noise spectrum. |
| `sensor.responsivity_a_per_w`, `collection_fraction` | Drop power to monitor photocurrent and separate collection/tap efficiency. | Modeled, assumed. |
| `sensor.dark_current_a`, `dark_activation_energy_ev`, `reference_temperature_k` | Assumed activated dark-current law; default detector temperature is the package node. | Modeled, assumed; detector placement and actual law remain unknown. |
| `sensor.filter_time_s`, `input_noise_a_per_sqrt_hz`, `noise_enabled` | One-pole analog signal filtering, white input current noise and Gaussian shot-noise approximation. | Modeled approximation. Noise variance uses that filter's bandwidth; signal-dependent noise changes are locally approximated. Exact photon counting at very low currents is deferred. |
| `sensor.full_scale_a`, `bits`, `sample_period_s`, `latency_s` | ADC clipping, quantization, sampling and delayed release. | Modeled, assumed. No hidden true detuning is supplied. |
| `sensor.temperature_resolution_k` | Quantization of the package-temperature observation. | Modeled, assumed; not an exact local ring thermometer. |
| `scenario.kind`, `ambient_temperature_k`, `start_s` | Chosen disturbance path, absolute boundary temperature and onset. | Modeled experiment choice. |
| `scenario.ambient_step_k`, `ambient_ramp_k_per_s` | Boundary-temperature step or ramp; internal temperatures evolve continuously. | Modeled; boundary rate is not automatically the ring-temperature rate. |
| `scenario.aggressor_power_w`, `aggressor_period_s`, `aggressor_ring` | Local triangular power waveform and targeted aggressor port. | Modeled, assumed; do not label its temperature trajectory an exact paper waveform. |
| `scenario.laser_drift_hz_per_s`, `power_step_fraction` | Laser-frequency ramp or fractional power step. | Modeled; spectrally or thermally different from an adjacent resistive heater. |
| `scenario.dropout_start_s`, `dropout_duration_s` | Missing monitor samples; controller sees freshness/timestamp behavior. | Modeled scenario, distinct from optical power loss. |
| `simulation.duration_s`, `tick_s`, `control_period_s`, `compute_latency_s`, `record_period_s` | Event lattice, feedback cadence, compute delay and logging cadence. | Modeled experiment choices; logging cadence does not define every ODE step. |
| `simulation.record_memory_limit_mb` | Recording-memory guard for long runs. | Modeled resource choice; increasing it changes storage permission, not physical fidelity. |
| `simulation.method`, `max_step_s`, `rtol`, `temperature_atol_k`, `voltage_atol_v`, `current_atol_a`, `heater_energy_atol_j`, `oma_integral_atol_j` | Adaptive integrator and component-scaled error controls, including accumulated energy and OMA-time integral. | Modeled numerical choices; convergence must be checked for the observable being reported. |
| `simulation.optical_evaluation`, `optical_table_tolerance` | Direct wideband NRZ evaluator or periodically interpolated table and its absolute fraction tolerance. | Numerical acceleration only. Adaptive probes and independent held-out points check interpolation; unsuccessful construction falls back to direct evaluation. The tolerance is not a bound on physical-model error or a proven uniform interpolation bound. |
| `simulation.seed`, `initial_temperature_k`, `initial_heater_code` | Reproducible random streams and initial physical/actuator conditions. | Modeled experiment choices; cold starts differ from a pre-equilibrated device. |
| `controller.name`, `controller.parameters` | Baseline or Python plugin and its own parameters. | Modeled; algorithms are independent baselines, not Lightmatter firmware. |
| `name`, `version`, `evidence` | Experiment identifier, schema and source annotations. | Metadata. An assertion in `evidence` is not a measurement. |

Electrical heat placement is not inferable from total transceiver pJ/bit. The assumed junction-to-waveguide fraction and per-lane driver-to-package power must be calibrated separately.

## Controller contract and state

| Item | Status and meaning |
|---|---|
| Controller inputs | `Observation.time_s`, `sample_time_s`, delayed `monitor_code`, `saturated`, `applied_heater_code`, quantized `temperature_code`, and `fresh`. Device limits/cadence arrive through `Interface`. |
| Controller action | One integer in-range heater DAC code per ring. Compute and actuation latency precede its physical effect. |
| Hold parameters | Constant code. Useful uncontrolled or prebiased baseline. |
| PI/PID parameters | Target/initial code, proportional/integral/derivative gains, fringe sign, integrator state and anti-windup behavior. Correct fringe and operating point must be established. |
| Scan-PI parameters | Scan count, settling frames, target peak fraction, gains and minimum signal threshold; sequential scan state and acquired/failed flags. This baseline does not implement a complete repeated reacquisition firmware state machine. |
| Dither parameters | Initial code, perturbation amplitude, gain and dwell; center and alternating measurement state. A drop-peak objective is not necessarily the best data OMA. |
| Physical long-run states | Absolute thermal-node temperatures; heater analog voltage; filtered monitor current; accumulated heater energy and OMA-time integral. No optical carrier-frequency oscillation is timestepped in this mode. |
| Stochastic/discrete states | Sensor noise history, random generators, held laser disturbances, pending sample/action queues, released sample and consumption timestamps, DAC latch. |
| Short waveform states | Complex cavity envelope and voltage-dependent junction RC state. Temperature is held fixed over the waveform; initialization is explicit. |
| Evaluator-only truth | Resonance/laser detuning, nearest line, exact modeled temperatures, optical powers, OMA, absorbed power and energy. These are recorded for judging a controller, not passed in its observation. |

## Importable and extraction-only quantities

| Input | Status | Interface and restrictions |
|---|---|---|
| Voltage axis, resonance shift, additional loss, C, R, leakage, provenance | Importable | `MeasuredJunctionTable` and `Plant(..., junction_model=...)` / `simulate(..., junction_model=...)`. Piecewise linear, one-dimensional voltage data; no extrapolation. Wavelength/temperature reference and measurement uncertainty must be described. |
| General passive thermal C, G, source map B, boundary conductance H, observer map W | Importable | `ThermalNetwork`. Positive capacities, reciprocal nonpositive off-diagonal conductances, conservative source/observer normalization and boundary balance are checked. |
| Thermal replacement for the controller plant | Importable with port contract | Pass `thermal_network=...`; provide named heater nodes, package node, ring observers and heater/absorption/aggressor source ports expected by `Plant`. Its one scalar ambient drives all boundary columns equally; independent boundary waveforms are only supported by the standalone network API. Junction heat follows absorption ports, driver heat and detector temperature use the named package. A generic reduced spatial network requires an explicit mapping/adaptor; naming alone cannot establish equivalent sensor placement. |
| Arbitrary finite voltage waveform and initial cavity field/junction voltage | Importable | `waveform_response`; supplied times must increase, drive must remain in junction domain. `nrz_drive` accepts a supplied binary sequence; `eye_segments` displays waveforms without BER inference. |
| Waveform timestep/substeps, electrical RC enable | Modeled, short waveform | Verify convergence as voltage and optical coefficients vary. Stationary NRZ moments assume ideal binary switching and independent successive bits. |
| Effective area, TPA coefficient, Kerr nonlinear index | Importable for screening only | `optics.nonlinear_screening` evaluates perturbative estimates on the linear field distribution; inputs have no target-device defaults. It is not a self-consistent nonlinear optical/carrier simulation or a guaranteed omission bound. |
| Layer names/thickness, thermal conductivity, volumetric heat capacity | Extraction-only | `Layer` and layered `SlabGeometry`; explicitly assumed materials unless supplied with measurements. |
| Slab width/extrusion depth; source and observer rectangle positions/extents | Extraction-only | A two-dimensional cross-section with finite depth, normalized source/observer overlap. Does not reconstruct the actual three-dimensional ring/package. |
| Spatial mesh, per-layer vertical resolution, retained modes or Krylov moments | Extraction-only | Sparse finite-volume assembly, generalized modal truncation and symmetric Krylov projection. Projected dynamics retain stability but may have small unphysical temperature undershoots; they are not the default physical-node plant. Compare mesh refinements and reduced/full responses; DC agreement alone does not imply small transient error. |
| Spatial boundary conditions | Extraction-only, restricted | Current example uses an isothermal bottom and insulated top/sides. Arbitrary convection, radiation and thermal interface resistance are deferred. |
| Time-temperature CSV, baseline, known power step, dataset ID/provenance | Importable | `load_step_csv`, `ThermalStepTrace`. Units and zero-to-constant heat-step amplitude are explicit; initial equilibrium and uniform fixed boundary are required. |
| Training/independent validation records, uncertainty scale, fit bounds | Importable | `fit_network_scales` fits two positive global C/G scales of a stated topology and reports residuals/conditioning. It does not identify arbitrary geometry or every conductance. |
| Normalized through/drop spectra, frequency, temperature, uncertainties and provenance | Importable | `SpectrumData`, `load_spectrum_csv`, `fit_spectrum`. Explicit parameter selection/bounds and caller-separated fitting/held-out records; no inferred normalization or reference planes. Local rank and an alternate power-equivalent parameter branch are reported. Even two-port power spectra need not uniquely determine coupling and loss. |
| Named parameter uncertainty ranges, case overrides, sample count and seed | Modeled study inputs | Uniform/log-uniform independent Latin-hypercube marginals with paired controller seeds. Correlations require explicit jointly chosen cases; no manufacturer population prior is implied. |

## Full-device checklist beyond the implemented reduction

The entries below are needed to predict a fabricated MRM from geometry, broaden validity, or reproduce a complete optical link. Grouping related quantities avoids suggesting that every one is an independent knob. Where an effective coefficient already incorporates an effect, adding the underlying effect again would double-count it.

| Parameter/effect to inventory | Status | Why it matters / what is missing |
|---|---|---|
| Ring/racetrack radius, straight-section lengths, bend shape | Effective length modeled; detailed geometry deferred | Needed for full-wave bend loss, phase and coupler extraction. Approximate public diameter is insufficient. |
| Waveguide width, silicon thickness, slab/etch depth, sidewall angle | Deferred | Determine vector modes, dispersion, confinement, scattering and carrier overlap. No exact target cross-section is established here. |
| Coupler gaps, interaction length, lateral/vertical alignment and longitudinal placement | Constant effective couplings modeled; geometry deferred | Needed for wavelength/temperature/bias-dependent complex coupling. Static model assumes two couplers separated by half a round trip. |
| BOX/cladding/substrate/metal/passivation dimensions and material identities | Extraction-only assumed layers; actual stack deferred | Control heat flow, mode tails, strain and electrical parasitics. |
| Index versus wavelength/temperature/doping, complex permittivity and anisotropy | Effective coefficients modeled; material field maps deferred | A bulk index and guessed confinement cannot determine the complete device response. |
| Group index and higher dispersion | Local group index and static β2 modeled; higher orders deferred | Full comb prediction requires measured or mode-solved frequency dependence. |
| Effective mode area, confinement/energy overlap, polarization, vector fields | Deferred | Required to predict nonlinear coefficients and carrier/thermal perturbations from microscopic inputs. |
| Complex, dispersive, lossy and asymmetric coupler S-matrices | Deferred | The current scalar lossless-coupler model cannot predict coupler reflection, phase imbalance or excess loss independently. |
| Distributed absorption versus scattering versus bend/radiation/metal loss | One effective split modeled; spatial separation deferred | The same effective absorption fraction multiplies base and voltage-dependent loss; this cannot independently represent scattering-dominated base loss and fully absorptive carrier loss. Only absorbed power that reaches modeled nodes should become heat; extracted α alone does not locate deposition. |
| Longitudinal inhomogeneity and segmented active/heated fractions | Deferred | Uniform ring reduction cannot predict local hot spots, segmented junctions or local stress directly. |
| Roughness statistics, line-edge correlation and surface-state absorption | Deferred | Need fabrication characterization; can produce both scattering loss and backreflection. |
| CW/CCW coupling, resonance doublets, coherent backscatter, parasitic reflectors | Deferred | Important when splitting approaches linewidth or in bidirectional links; P26 explicitly discusses reflection penalties. |
| Extra spatial modes, TE/TM coupling, polarization rotation and birefringence | Deferred | Single-mode TX model; RX polarization diversity and scrambler demonstration are not reproduced. |
| Kerr coefficient, two-photon absorption, free-carrier generation/lifetime/recombination | Perturbation screening available; self-consistent dynamics deferred | May alter phase/loss/self-heating at high intracavity power. No target mode volume or nonlinear dataset is available. Linear-field screening cannot exclude another nonlinear branch. |
| Surface/trap dynamics, photoconductivity, nonlinear bistability/self-pulsation beyond thermal feedback | Deferred | Thermal absorption feedback is included; omitted carrier/material dynamics may dominate in other regimes. |
| Thermal expansion and stress-optic/photoelastic response | Included only in an effective measured thermal slope; separate geometry/stress model deferred | Separating them requires mechanical constraints, tensors, and temperature fields; avoid double-counting. |
| Temperature-dependent k, heat capacity, boundary conductance | Deferred | Constant passive network is a local reduction; broad 0-105 °C validity is uncalibrated. |
| Thermal interface resistance, die attach, bumps, underfill, heat spreader/cold plate | Effective boundary/network inputs importable; physical package deferred | Dominate slow poles and cross-die paths; public link results do not identify them. |
| Convection/radiation/coolant flow and ambient spatial gradients | Standalone network supports multiple boundaries; controller plant drives them with a shared ambient; nonlinear boundary physics deferred | Actual airflow/cooling maps and interface data are absent. |
| Remote heaters, arbitrary inter-ring matrix and EIC/ASIC aggressor placement | Passive network importable; default nearest-neighbor/shared-package surrogate modeled | Need independent excitation/measurement per thermal path. A common-mode fit cannot determine a complete cross-coupling matrix. |
| Heater dimensions/material/resistivity, contact resistance, nonlinear I-V, thermal runaway | Lumped R(T)/limits modeled; physical design deferred | Changes voltage-to-power conversion and reliability envelope. |
| Heater DAC INL/DNL, glitches, PWM carrier, output impedance, rails/supply noise, slew | Code range/settling/limits/delay modeled; detailed converter deferred | Could dominate small-signal or rapid commands; need electronic transfer measurements. |
| Junction doping concentrations/profiles, diffusion widths, contact locations, mobility, recombination and surface traps | Deferred | Needed for predictive Poisson/drift-diffusion/TCAD; compact coefficients cannot reveal them. |
| Junction temperature-dependent C-V/I-V/R and breakdown/forward injection | Static reverse-bias C/V response modeled/importable; other regimes deferred | Tables are bounded; current model must not be used past its reverse-bias domain. |
| Driver output impedance, pre-emphasis, finite rise/fall asymmetry, supply droop, inductance and distributed RF lines | Lumped RC short waveform modeled; detailed driver deferred | Needed for accurate high-speed eyes, chirp and electrical heating. No driver design is disclosed. |
| Electrical energy deposition between ring, driver, package and off-chip circuit | Configurable assumed heat allocation; spatial circuit extraction deferred | CV² energy is not automatically all deposited in the optical ring. |
| PRBS order, run-length correlations, coding/FEC, data scrambler, modulation format | Supplied binary waveforms importable; iid NRZ long-run reduction modeled; PAM4/FEC deferred | P26 uses PRBS9; iid stationary moments are not an exact PRBS9 link simulation. |
| Timing jitter, duty distortion, lane phase/correlation, clock recovery and equalization | Deferred | Required for receiver eye/BER, not identified by static OMA. |
| Laser linewidth/phase-noise PSD, coherent multitone beats, RIN spectrum, correlated drift | Simple held perturbations modeled; physical noise spectra deferred | RMS per event is not a linewidth and depends on update cadence. |
| Laser wavelength/power versus temperature/current, second feedback loop, failover transients | External drift/step scenarios modeled; laser dynamics deferred | Ring locking and laser control may interact; no coupled laser plant supplied. |
| Optical launch loss, polarization coupling, fiber/connector/packaging return loss | Lumped bus loss modeled; complete path deferred | Source reference planes and direction matter; chip-to-chip 3 dB cannot be assigned to one component. |
| Interleaver transfer matrix and thermal drift, coupled-ring RX filter bank | Deferred | Need separate component models; do not claim full transceiver fidelity from TX-ring tests. |
| Fiber attenuation, dispersion/slope, PMD, nonlinearities and propagation delay | Deferred | Full link waveform/BER is outside current component-control plant. |
| Photodiode geometry/capacitance/bandwidth/saturation, responsivity spectrum/temperature | Scalar collection/responsivity/filter/full scale modeled; richer detector deferred | Sensor and high-speed receiver are different devices and operating planes. |
| TIA transimpedance, poles/zeros, gain changes, offset, drift, 1/f noise and electrical feedthrough | Equivalent one-pole current sensor modeled; detailed front end deferred | Need transfer/noise measurements to establish useful measurement bandwidth and noise floor. |
| ADC INL/DNL, offset/gain error, aperture/jitter, oversampling/decimation and multiplexing | Uniform quantizer/latency/cadence modeled; hardware-specific effects deferred | A bit count alone does not establish effective resolution or control latency. |
| Thermistor/temperature sensor location, offset, noise and self-heating | Quantized package observation modeled; independent sensor dynamics deferred | A package reading is not the local ring state. |
| Startup calibration, wavelength identity, rescan/relock, fault watchdog, saturation recovery | Basic scan/dither/PI modeled; complete firmware deferred | Must be developed for the chosen hardware I/O; nearest-line truth is an evaluator diagnostic. |
| Fixed-point widths, rounding/overflow, FPGA clock/resource budgets, RTL timing closure | Deferred | Software controller interface supports later verification; no FPGA synthesis or bit-exact firmware claim. |
| Process corners, within-die/wafer correlations, aging, drift, reliability limits | Explicit cases/independent ranges modeled; population and aging models deferred | Current arrays share optical/junction parameters except per-ring resonance errors; imported thermal networks can differ by node. General per-ring optical/junction heterogeneity is not implemented. Need device statistics; synthetic range selection is not yield prediction. |
| Parameter covariance, calibration uncertainty and identifiability | Two-scale thermal fit and bounded spectral fit diagnostics, alternate spectral branches, and explicit parameter studies modeled; general statistical inference deferred | Fit residual alone does not resolve multiple physically equivalent parameter sets. |

## Derived outputs and acceptance criteria

FSR, round-trip time, loaded linewidth/Q, decay rates, through/drop/internal-loss fractions, stored optical energy, OMA, heater power and cumulative energy should be derived from the independent parameters, not supplied as mutually inconsistent extra knobs. A measured quantity such as linewidth may instead be a fitting target.

The exact periodic static spectrum and the isolated-resonance temporal coupled-mode approximation are separate calculations. Default `stationary_nrz` long runs use photon-memory moments near the nearest resonance and exact per-pass quasistatic response for distant WDM lines, with a smooth convex blend in between. The default blend spans 0.08-0.12 FSR; this is an approximation choice, not a device law or error bound. The current wideband dynamic helper rejects nonzero β2. Independent-lane mean fields propagate through the bus. It does not solve delayed multi-ring optical fields, coherent interline interference, dispersive pulse propagation, or finite driver RC. Use exact spectra to check operating points, inspect the returned discrepancy diagnostics, vary the blend thresholds where relevant, and use short finite-RC waveforms to assess that reduction. See [OPTICAL_MODEL.md](OPTICAL_MODEL.md).

Reported detuning extrema and wrong-line occupancy are checked at simulation event boundaries; logging and event rates define temporal coverage. They are not proofs that no between-event excursion occurred. Choose and record desired thresholds for intended-fringe capture, OMA penalty, settling/recovery time, maximum detuning, heater energy, rails, sample staleness, wrong-line capture, and independent receiver BER before claiming success. Some baselines intentionally optimize different objectives.

No raw receiver/BER model is available, so BER is explicitly unavailable. The public paper's BER values describe its hardware demonstration. Matching temperature or monitor traces in this simulator cannot independently reproduce those values.
