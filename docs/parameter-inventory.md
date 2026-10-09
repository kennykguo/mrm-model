# Inputs, parameters, states, and remaining physics

This inventory describes the implemented public Python/JSON interfaces, checked against the source on 2026-10-09. All physical quantities use SI. A **parameter** describes the device or algorithm; an **input** varies with an experiment; a **state** stores history. The JSON fields are strict: misspelled or unknown fields raise an error. Listing a field does not establish a Lightmatter value for it.

`default_config()` and the example profiles override some bare dataclass defaults. The tables distinguish those values. All profile values are assumed except the public O-band/grid/thermal-sensitivity constraints recorded in `provenance`; the approximately 56 ohm heater hypothesis resembles an earlier G25 estimate, not a measured G26 interface. Per-ring optics can differ; actuator and sensor parameter sets are currently shared by all lanes.

## Configuration container

`ExperimentConfig` contains the following fields; arrays are serialized as JSON lists.

| Field | Meaning |
| --- | --- |
| `schema_version` | Format version, currently 1 |
| `optical` | One `OpticalParams` per ring; order is bus order |
| `thermal` | `ThermalParams`, matched to ring/heater count |
| `actuator` | Shared `ActuatorParams` |
| `sensor` | Shared `SensorParams` |
| `run` | `RunParams` |
| `disturbance` | `DisturbanceParams` |
| `controller` | `ControllerParams` |
| `provenance` | User metadata dictionary; default source, assumptions and `calibrated=False` |

The integrated experiment supports 1 to 32 rings, one assigned laser and heater per ring, and up to 256 compact thermal states. The standalone thermal API can represent different heater/ring/state counts if explicit maps are supplied. Changing a `calibrated` metadata flag does not perform or prove calibration.

## Optical parameters

Every field below belongs to `OpticalParams` and may vary per ring.

| Field | Units and meaning | Bare default; profile difference |
| --- | --- | --- |
| `reference_frequency_hz` | Hz, reference resonance at reference T and zero voltage | c/(1310 nm); profiles synthetically align each ring to its assigned grid/bias |
| `length_m` | m, round-trip path length | 40e-6; assumed |
| `group_index` | Dimensionless group index | 4.2; assumed |
| `loss_power_np_per_m` | 1/m, total **power** propagation attenuation | 1000; assumed, includes scattering and absorption |
| `coupling_bus_power` | Fraction [0,1), bus-to-ring power coupling | 0.08; assumed |
| `coupling_drop_power` | Fraction [0,1), ring-to-drop power coupling | 0.01; assumed |
| `df_dtemperature_hz_per_k` | Hz/K, signed resonance frequency derivative | -15e9; magnitude is approximate public constraint, linearity/reference condition assumed |
| `reference_temperature_k` | K, optical calibration reference | 298.15; assumed |
| `df_dvoltage_hz_per_v` | Hz/V, signed zero-bias linear response | +8e9; assumed; negative reverse voltage therefore redshifts this hypothesis |
| `loss_voltage_np_per_m_per_v` | 1/(m V), voltage derivative of total power loss | 0; assumed |
| `phase_dispersion_s2` | s^2, quadratic phase coefficient in angular-frequency offset | 0; assumed |
| `intrinsic_absorption_fraction` | Fraction [0,1] of intrinsic dissipation deposited as local heat | 1; strong illustrative assumption, not measured absorption |
| `voltage_points_v` | Increasing voltage calibration axis, at least two entries | Empty; table disabled |
| `resonance_voltage_shifts_hz` | Hz, signed frequency-shift values at voltage points | Empty; supplied table overrides linear voltage shift |
| `loss_voltage_values_np_per_m` | 1/m, **absolute total** power loss at voltage points | Empty; supplied table overrides linear loss law |

At least one of the two response tables can be supplied; each populated table must match `voltage_points_v`. Interpolation is piecewise linear, voltage extrapolation is rejected, and loss must remain nonnegative. A scalar phase-index value is not needed because the measured/reference resonance anchors phase and group delay fixes its local slope. Radius, phase index, Q, FSR, linewidth and lifetime are therefore not extra independent knobs in this compact parameterization.

Derived properties are `roundtrip_time_s`, `fsr_hz`, `kappa_intrinsic_s_inv`, `kappa_bus_s_inv`, `kappa_drop_s_inv`, `kappa_total_s_inv`, `photon_lifetime_s` (energy lifetime), `linewidth_hz` (local pole linewidth), `loaded_q`, and `exact_fwhm_hz` (local periodic FWHM including supplied quadratic dispersion, possibly undefined for very broad resonances). `optical_response()` returns through/drop fields and powers, cavity energy, resonance frequency, absorbed, scattered and total dissipated power.

## Thermal network parameters

Let Ns be state count, Nh heater count, Nr ring count, and Nb boundary count.

| `ThermalParams` field | Units/shape | Meaning and default behavior |
| --- | --- | --- |
| `capacitance_j_per_k` | J/K, vector Ns or matrix Ns x Ns | Positive nodal capacities or full SPD generalized capacity; required |
| `conductance_w_per_k` | W/K, Ns x Ns | Full SPD conductance; required |
| `heater_map` | Dimensionless, Ns x Nh | Deposition map multiplying heater W; required |
| `ring_map` | Dimensionless, Nr x Ns | Ring-temperature reconstruction; required |
| `boundary_map_w_per_k` | W/K, Ns x Nb | Boundary temperature forcing; omitted map becomes `(G @ reference_state_map)[:,None]` |
| `reference_temperature_k` | K | Fixed state reference, default 300 |
| `heater_temperature_map` | Dimensionless, Nh x Ns | Heater-strip temperatures for TCR; defaults to ring map only when counts match |
| `reference_state_map` | Vector Ns | Coordinates of uniform-temperature rise; defaults to ones |
| `physical_network` | Boolean | Default True: validates a reciprocal nodal conductance graph/nonnegative maps; False allows signed reduced coordinates while retaining SPD and equilibrium checks |
| `provenance` | Text | User-specified source/calibration statement |

Default nodal graph: ring capacities 2e-9 J/K, heater capacities 0.5e-9 J/K, substrate capacity 2e-6 J/K; ring-heater conductance 2e-4 W/K, ring-substrate 5e-5 W/K, heater-substrate 2e-5 W/K, nearest ring-ring 5e-6 W/K, and substrate-boundary 2e-3 W/K. All are assumed. Default node order is rings, heaters, shared substrate. Thermal modes/time constants, static self/cross gains, and continuous state-space A/B/E/Q/H are derived from C/G/maps.

Direct thermal inputs are `dt_s`, `heater_power_w` (one W value per heater), `ambient_temperature_k` (scalar or one absolute K value per boundary), and optional `external_heat_w` (Ns generalized heat loads in W). `ThermalNetwork` can initialize from a positive scalar uniform temperature or a physical nodal vector. A reduced modal model needs uniform scalar initialization or explicit state handling; generalized absolute coordinates are not individual nodal temperatures.

## Heater electronics

| `ActuatorParams` field | Units/meaning | Bare default -> profile override |
| --- | --- | --- |
| `dac_bits` | Integer 1..32 | 12 -> 16 |
| `dac_reference_v` | V, full code-to-voltage scale before gain/offset | 3.3 -> 1.6 |
| `min_voltage_v` | V, requested lower output bound | 0 |
| `max_voltage_v` | V, voltage compliance | 3.3 -> 1.6 |
| `heater_resistance_ohm` | ohm at reference heater temperature | 500 -> 56 |
| `resistance_tcr_per_k` | 1/K, linear resistance coefficient | 2e-4 -> 1e-3 |
| `reference_temperature_k` | K, resistance reference | 300 -> 298.15 |
| `settling_time_s` | s, first-order output time constant | 1e-6 -> 2e-6 |
| `latency_s` | s, transport delay before changed code is applied | 0 |
| `slew_rate_v_per_s` | V/s, slope limit | Infinity; JSON null means unlimited |
| `max_current_a` | A, current compliance | 0.02 -> 0.03 |
| `max_power_w` | W, heater power compliance | 0.02 -> 0.04 |
| `dac_gain` | Dimensionless conversion gain | 1 |
| `dac_offset_v` | V, conversion offset | 0 |
| `voltage_noise_std_v` | V RMS, independent per-output-update noise, **not** spectral density | 0 |

Inputs to `HeaterDriver.step()` are elapsed time, commanded code(s), and actual mapped heater temperature(s). State/output includes queued commands/timestamps, active and commanded integer codes, voltage, resistance, current, endpoint power and interval-average power. Compliance can override the lower requested voltage bound. Negative/nonpositive extrapolated resistance is rejected.

## Monitor and ADC

| `SensorParams` field | Units/meaning | Bare default -> profile override |
| --- | --- | --- |
| `responsivity_a_per_w` | A/W, photodiode optical-to-current scale | 0.8; assumed |
| `dark_current_a` | A, mean dark current | 1e-9 |
| `input_noise_density_a_per_sqrt_hz` | A/sqrt(Hz), one-sided electronics density | 2e-12 |
| `bandwidth_hz` | Hz, one-pole analog 3 dB bandwidth | 2e5 |
| `adc_bits` | Integer 1..32 | 14 -> 16 |
| `adc_full_scale_a` | A, maximum ADC input current after gain/offset | 1e-3 -> 100e-6 |
| `adc_gain` | Dimensionless front-end/ADC scale | 1 |
| `adc_offset_a` | A, ADC-input offset | 0 |
| `latency_s` | s, sample transport/conversion delay | 0 |
| `shot_noise` | Boolean, filtered Gaussian shot-noise approximation | True |

Inputs are `dt_s` and drop optical power(s) W; `sample()` acquires at current time. `SensorObservation` contains `codes`, reconstructed `current_a`, acquisition `timestamp_s`, scheduled `delivered_time_s`, `valid` and per-lane `clipped`. Filtered current, noise history and delayed-sample queue are state. There is no configurable separate averaging/aperture window, arbitrary filter transfer function, 1/f noise, RIN or ADC INL/DNL in this version.

## Run setup

| `RunParams` field | Units/meaning | Default |
| --- | --- | --- |
| `duration_s` | s, total slow-run duration | 0.01 |
| `sample_period_s` | s, controller/ADC acquisition interval | 10e-6 |
| `physics_substeps` | Integer 1..100, electrothermal forcing refreshes per controller interval | 4 |
| `record_every` | Integer >=1, trace decimation; metrics still accumulate every step | 1 |
| `seed` | Random seed; separate heater and sensor streams | 7 |
| `input_power_w` | W **per laser**, shared launch power at bus entrance | 100e-6 |
| `laser_frequency_hz` | One Hz value per assigned lane | Single c/(1310 nm); profiles generate 400 GHz grid |
| `initial_codes` | One initial DAC value per ring | Bare (2048,); profiles use 20000 |
| `initial_temperature_k` | Optional nodal K vector, one entry per state | None: solve initial coupled electrothermal equilibrium |
| `initial_state_rise_k` | Optional generalized rise coordinates relative to the fixed thermal reference; supports modal ROM initialization | None; mutually exclusive with nodal temperature |
| `junction_low_v` | V assigned to bit 0 | 0 |
| `junction_high_v` | V assigned to bit 1 | -1 |
| `data_one_fraction` | Fraction [0,1], independent NRZ duty average | 0.5 |
| `optical_heating` | Include static absorbed optical power in heat forcing | True |
| `bus_loss_db_between_rings` | dB, power insertion loss between consecutive rings | 0 |
| `lock_tolerance_hz` | Hz, evaluation budget around initial **unwrapped assigned** detuning; not controller truth | 5e9 |
| `target_detuning_hz` | Optional Hz vector fixing the unwrapped evaluation target, used to retain the same target across device corners | None: initial assigned detuning |
| `max_steps` | Control-step work cap | 1,000,000 |
| `max_recorded_rows` | Stored row cap | 200,000 |
| `max_trace_bytes` | Byte cap on preallocated float64 trace arrays; increase decimation for large arrays | 256,000,000 |

An input frequency shift is common to the configured lasers and optical power modulation is a common fraction in the standard disturbance generator. Independent lane histories, explicit data patterns and arbitrary callbacks are not fields in this JSON generator; use/extend the lower-level APIs or custom experiment code when required.

## Disturbance generator

| `DisturbanceParams` field | Units/meaning | Default |
| --- | --- | --- |
| `kind` | `constant`, `ambient_ramp`, `ambient_step`, `ambient_triangle`, or `local_heat` | `ambient_ramp` |
| `start_s` | s, disturbance start | 0.002 |
| `ambient_temperature_k` | K, baseline thermal boundary | 298.15 |
| `ambient_amplitude_k` | K, nonnegative step/ramp/triangle amplitude | 2 |
| `ramp_rate_k_per_s` | K/s, capped ambient-ramp slope | 808; this is a synthetic boundary scenario, not the paper's local test |
| `period_s` | s, triangle/heat/power modulation period | 0.1 |
| `laser_drift_hz_per_s` | Hz/s, common signed laser drift after start | 0 |
| `laser_power_fraction_amplitude` | Fraction [0,1], common sinusoidal modulation around 1 | 0 |
| `external_heat_peak_w` | Optional W vector per thermal state, multiplied by local triangular envelope; nonnegative nodal heating or signed projected ROM loads | None |

For triangles the slope follows `2*ambient_amplitude_k/period_s`; `ramp_rate_k_per_s` only affects `ambient_ramp`. `local_heat` deposits watts, and its ring temperature ramp is a derived result, not guaranteed by the requested power. For a modal ROM, project a physical source through its deposition map (for example `thermal.heater_map[:, 0] * aggressor_watts`); individual negative modal entries do not mean physical cooling. The generator does not reconstruct the published aggressor waveform; source-matched raw input histories are needed for that claim.

## Controller configuration and online interface

| `ControllerParams` field | Meaning/default |
| --- | --- |
| `kind` | `open_loop`, `pi`, `dither`, or `mpc`; default `pi` |
| `kp` | PI proportional gain on calibrated code-equivalent error; 0.35 |
| `ki_per_s` | PI integral gain per second; 2000 |
| `dither_amplitude_codes` | DAC-code sinusoid amplitude; 10 |
| `dither_frequency_hz` | Base tone Hz; 2000; lane i uses base*(1+0.17*i) |
| `dither_gain_codes_per_s` | DC gradient adaptation scale codes/s; 1e5 |
| `dither_target_gradient_per_code` | Target monitor-fraction gradient per DAC code in `slope` mode; 0 |
| `dither_objective` | `flank` (default, same monitor target as PI/MPC), `peak` or `slope` |
| `dither_phase_rad` | Lock-in reference phase compensation in radians; 0 |
| `mpc_horizon` | Integer prediction steps 1..100; 8 |
| `mpc_move_penalty` | Dimensionless normalized move penalty; 0.1 |
| `acquisition` | Enable sequential scan wrapper; False |
| `scan_min_code` | Lower scan command; 0 |
| `scan_max_code` | Upper scan command; 65535, clipped to actuator range |
| `scan_points` | Number of evenly spaced scan points; 100 |
| `scan_dwell_samples` | Controller calls per dwell; 40 |

`ControllerContext` is offline nominal knowledge: `sample_period_s`, `initial_codes`, `max_code`, `target_fraction` (ADC fraction), `jacobian_per_code` (Nr x Nr monitor fraction/code), `adc_max_code`, and discrete local matrices `a`, `b`, `c`. They represent state deviation dynamics driven by DAC-code deviations and measured output fractions. It is not a JSON `controller` field; pass an independently identified instance to `run_experiment()` or `compare_controllers()` to test model mismatch.

The online `Observation` fields are `time_s` (current invocation), `sample_time_s` (acquisition), `adc_codes`, `valid=True`, and optional `clipped`. A custom controller returns one finite DAC command per ring. It does not receive real-time ring temperatures, resonance or detuning. PI integrator/timestamp, dither means/gradient/tones/DC, MPC observer state/covariance/prior command, and acquisition lane/grid/history are algorithm states. Built-in MPC process/measurement covariance constants, demodulation filter choice and lane-tone spacing are fixed in code, not exposed knobs; customize the controller to vary them.

## Short waveform and nonlinear parameters

`NonlinearParams` is a separate API object, not currently embedded in `ExperimentConfig` JSON or slow-control runs.

| Field | Units/meaning | Default |
| --- | --- | --- |
| `tpa_rate_per_j_s` | 1/(J s), energy-dependent two-photon energy-decay rate coefficient | 0, disabled |
| `carrier_lifetime_s` | s, uniform carrier recombination lifetime | 1e-9, assumed |
| `carrier_volume_m3` | m^3, effective carrier generation volume | 1e-17, assumed |
| `carrier_quantum_yield` | Fraction [0,1], pair-generation yield per absorbed photon pair | 1 |
| `fca_cross_section_m2` | m^2, effective free-carrier absorption cross-section | 0, disabled |
| `fcd_hz_per_m3` | Hz/(m^-3), equivalently Hz*m^3; signed resonance shift per carrier number density | 0, disabled |
| `kerr_hz_per_j` | Hz/J, signed resonance shift per total cavity energy | 0, disabled |

`simulate_cavity()` accepts optical params, strictly increasing `time_s`, scalar or matching-array `laser_frequency_hz`, `temperature_k`, `junction_voltage_v`, and `input_power_w`; optional `nonlinearity`, complex `backscatter_rad_per_s`, `initial_forward_field`/`initial_reverse_field` (sqrt(J)), `initial_carrier_density_m3` (m^-3), `max_samples=250000`, `rtol=1e-7` and scaled-coordinate `atol=1e-10`. Array input j applies during interval j; final input defines final output. Linear intervals use exact propagation; nonlinear intervals integrate adaptively.

`simulate_nrz()` adds a binary `bits` input and these keyword knobs: `bit_rate_bps=50e9`, `samples_per_bit=32`, `voltage_low_v=0`, `voltage_high_v=-1`, `electrical_tau_s=5e-12`, `receiver_bandwidth_hz=25e9` (zero bypasses filter), `receiver_noise_std_w=0`, `discard_bits=8`, and optional `initial_voltage_v`, along with the same optical histories, nonlinear/backscatter and initial field/carrier inputs and `max_samples`. Its initial voltage defaults to bit-0 voltage. The receiver noise standard deviation contributes only to an analytical Gaussian metric, not a random waveform. For this convenience NRZ function, nonlinear tolerances use internal defaults.

`cavity_rhs()` additionally accepts `input_field` and `reverse_input_field` in sqrt(W) for the continuous two-mode derivative API. `nonlinear_steady_states()` takes laser Hz, positive input W, temperature, voltage and nonlinear coefficients; its output fields are `cavity_energy_j`, `carrier_density_m3`, `field`, `stable`, `eigenvalues_s_inv`, `relative_residual`. `select_nonlinear_branch(previous_energy_j=...)` chooses a history-consistent stable branch. That branch solver excludes thermal states and backscatter.

Waveform states are forward/reverse complex cavity fields, carrier density, electrical RC voltage and receiver-filter history. The returned arrays include voltage, through/drop/reflected/reverse-drop power, forward/reverse/total energy, carrier density, absorbed/scattered/TPA/FCA power, fields and actual resonance. Eye metrics report optical levels/OMA/extinction/percentile opening and an explicitly labeled Gaussian proxy. This is not measured BER or a complete link simulation.

## Calibration and extraction controls

These APIs change fitted coefficients or generate a reference model; they are not extra runtime device parameters.

- `fit_optical_spectrum()`: optical initial guess, frequency/through measurements, optional drop measurements, input power, temperature/bias, requested fitted fields, held-out validation fraction or mask and max evaluations. Returns `OpticalFit` with fitted `params`, `success`, `message`, `fitted_fields`, training/validation RMSE, covariance, standard errors, Jacobian condition, identifiability warnings, through/drop predictions and training/validation indices.
- `fit_thermal_step()`: time/frequency-shift records, `n_modes`, actual `step_power_w`, known `step_time_s`, `holdout_fraction`, `training_mask`, `amplitude_sign` and `max_nfev`. It fits offset but does not independently fit delay. `ThermalStepFit` contains pole times/amplitudes/gains, offset, status/message, training/validation RMSE, covariance, condition/warnings, indices and step time. It identifies a measured transfer response, not unique microscopic C/G geometry.
- `thermal_fem_reference()`: `mesh_level`, `n_modes`, `modeled_length_m`, `heater_power_w`, `transient_subdivisions`, and `boundary_contact_w_per_m2_k=1e7`. `thermal_fem_convergence()` accepts mode count, extrusion length and contact coefficient. Generic layer/material constants and the DOF guard are specified in `extraction.py`; they are not a foundry PDK.
- `waveguide_mode_reference()`: `width_m=400e-9`, `height_m=200e-9`, `wavelength_m=1310e-9`, `mesh_scale=1`, `delta_wavelength_m=2e-9`, `padding_m=1.5e-6`, optional `silicon_dn_dt_per_k`/`silica_dn_dt_per_k`, and `delta_temperature_k=.5`. The convergence API refines mesh, wavelength step and outer domain. Outputs include mode overlaps, vector DOFs, material/source/version metadata, neff, ng and optional temperature derivative. Actual silicon dispersion and silica Sellmeier are embedded with source attribution; their data validity range is checked. [extraction.md](extraction.md) explains geometry/material assumptions and memory guards.

Refer to [calibration.md](calibration.md) for exact current fitting options and file schemas. Fits and covariance estimates need identifiability and held-out physical measurements; synthetic fit recovery validates the procedure rather than the target device.

## What a more microscopic model would additionally need

The compact parameterization deliberately does not ask for physical inputs that it cannot use. The following data become necessary when extending the relevant physics and are **not implemented JSON fields**:

| Extension | Additional required inputs |
| --- | --- |
| Optical geometry/mode extraction | Radius/straight lengths, full cross-section and sidewall shape, coupler gap/length/topology, stack/claddings, complex dispersive material index, spatial doping, bend/confinement/modal overlaps, roughness and polarization |
| Junction transport/driver | Absolute dopant/reference carrier distributions, contacts and activation, mobility/recombination/traps, permittivity, DC I-V and C-V, access/contact resistance, breakdown/leakage, transmission-line/parasitic/driver response |
| Spatial/package thermal model | Layer dimensions, density/heat capacity, conductivity tensors/T dependence, heater material/geometry/resistivity, interface resistance, boundary/heat-sink/coolant conditions, ASIC heat-source maps and node locations |
| Nonlinear thermal/optical coupling | Spatial absorption and carrier generation, mode volume/overlap, TPA/FCA/FCD/Kerr calibration at relevant wavelength/power/T, nonuniform carrier transport and thermal feedback of fast nonlinear heat |
| Real monitor/DAC electronics | Per-lane variation, arbitrary measured frequency response, boxcar/aperture integration, 1/f and correlated noise, INL/DNL, drift, jitter, ADC conversion/scheduling and PWM/sigma-delta detail |
| Full WDM/bidirectional link | Coherent complex multiport coupler/reflection network, interleavers, laser linewidth/RIN/phase/polarization, correlated joint data patterns, fiber dispersion/loss/reflections, photoreceiver/TIA/CDR/equalizer and error-counter/FEC definitions |
| Process/aging uncertainty | Correlated per-die/wafer distributions, spatial covariance, fabrication/packaging stress, calibration drift/aging and validity envelope |

There is no finite universal exhaustive parameter list independent of selected physics. This document exhaustively lists current configurable fields and separates conditional physical data from existing capabilities. [lightmatter-data-request.md](lightmatter-data-request.md) prioritizes what is worth requesting before implementing extra detail.
