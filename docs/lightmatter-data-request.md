# Calibration data to request from Lightmatter

Prepared 2026-10-09. This is a draft request for the capstone team to review and send; no message has been sent. The aim is a calibrated controller test plant with a common interface, preferably without requiring proprietary layout. A coefficient set, compact model, or thermal state-space model can be more useful than a complete geometric drawing.

The public references are the supplied [OFC 2026 M4B.3 paper record](https://opg.optica.org/abstract.cfm?origin=search&uri=ofc-2026-M4B.3) and the [official April 2026 presentation](https://lightmatter.co/wp-content/uploads/2026/04/OFC-2026-16-lambda-Bidi_PAM4-Nikhil-Kumar.pdf), accessed 2026-10-09. They demonstrate stability but do not disclose the open-loop dynamics, sensor/actuator transfer functions or full internal loop.

## Minimum useful package

Please identify the hardware generation and intended capstone test platform first. We would appreciate any shareable existing compact model (Python, Verilog-A, FMU, lookup tables, thermal matrices or transfer functions), with its parameter units, operating domain, port definitions, update convention and validation errors. We can use ranges or anonymized traces if detailed measurements cannot be shared.

The highest-priority data are:

| Priority | Requested data | Minimum useful units and fields | What it identifies |
| --- | --- | --- | --- |
| 1 | Controller electrical/measurement interface | DAC code mapping, V/A/W range, update period s, sample/integration duration s, end-to-end delay s, ADC mapping A or V per code, saturation and invalid-data flags | Hardware-transfer boundary and achievable loop bandwidth |
| 2 | Open-loop heater steps at several operating points | Time s, commanded code, actual terminal V and A, laser frequency Hz, resonance frequency Hz, monitor current A/code, boundary temperature K | Actuator dynamics and heater-to-resonance gain/poles/delay |
| 3 | Through and drop spectra vs heater, bias, temperature and input power | Frequency Hz or vacuum wavelength m; power W with reference plane; junction bias V; heater power W; temperature K | Linewidth, coupling/loss, monitor slope, multimode/splitting and power sensitivity |
| 4 | Multiring dynamic response | For each driven heater j, synchronized responses of every ring i, using the fields above | Dynamic crosstalk; a static matrix alone cannot establish transient phase/lag |
| 5 | Known-disturbance records | Local aggressor command/actual power, temperature or calibrated resonance, stabilizer commands, raw monitor and time stamps | Reproduce the 808 K/s paper test and the later >2000 K/s presentation separately |
| 6 | Sensor/actuator characterization | Dark current A; responsivity A/W; gain V/A; electrical/noise bandwidth Hz; noise density A/sqrt(Hz); ADC/DAC nonlinearity, slew and settling | Measurement statistics, quantization and implementation delay |
| 7 | Reference controller trace | Acquisition, normal tracking, saturation, loss-of-light, laser jump and recovery; controller state/mode and commands | Fair baseline and state-machine validation |
| 8 | Device/package variation | Joint parameter sets, device IDs, covariance, operating conditions, measurement uncertainty | Robust-control uncertainty without impossible independent parameter combinations |

For step tests, please state whether the stabilization loop is disabled. If it must remain active, include all stabilizer commands and the reference controller mode; a flat monitor in closed loop does not directly identify the uncontrolled heater plant. Distinguish the heating aggressor from the stabilizer actuator.

Please provide synchronized raw timestamps. To identify a suspected fastest time constant tau, a sampling interval much smaller than tau and a record spanning the slow package dynamics are needed. We can agree a rate after discussing likely time scales; the public data do not establish tau. Supply acquisition rate, dropped-sample behavior, filtering, trigger alignment, timestamp clock source and uncertainty. Long records can be decimated after appropriate anti-alias filtering, while preserving high-rate short transients.

## Fields and conventions to attach to every dataset

Use SI units internally, with unit-bearing column names. Celsius is fine for supplied temperature files if explicitly declared; absolute temperature conversion and temperature increments are different operations. Frequency, angular frequency and wavelength must never share an unlabeled column.

A practical transient CSV could contain:

```text
time_s,device_id,ring_id,heater_command_code,heater_voltage_V,heater_current_A,laser_frequency_Hz,input_power_W,resonance_frequency_Hz,monitor_current_A,monitor_adc_code,boundary_temperature_K,junction_bias_V,controller_enabled,controller_mode,valid
```

Not every field must be available. Mark missing data as missing, not zero. Put experiment metadata in a JSON/YAML sidecar with:

- Generation, die/package identifier, lane mapping, Tx versus Rx topology, firmware revision and sensor location.
- Measurement reference planes, cable/coupling losses, optical polarization, laser settings and all active control loops.
- Ambient, cold-plate, package and local-ring temperature definitions; how each temperature was measured or inferred.
- Time origin, controller clock/update order, ADC integration window, filter response and pipeline delays.
- Electrical polarity, differential versus single-ended junction voltage, commanded versus measured actuator values, and compliance/current limits.
- Data pattern, ones density, squelch state, rate/format and whether average monitor power is pattern averaged.
- Instrument uncertainty, noise bandwidth, repeated trials, calibration date and allowed operating bounds.
- Training/validation experiment labels, preferably with independently held-out devices and operating conditions.

For a spectral CSV, use `frequency_Hz,through_power_W,drop_power_W,input_power_W` plus `temperature_K,heater_power_W,junction_bias_V,device_id,ring_id`. Complex S-parameters, if available, should declare port order, phase convention, reference impedance/normalization, mode/polarization and reference planes. Power-only spectra cannot uniquely identify every coupling/loss/reflection parameter.

For a thermal reduced model, declare `C` in J/K, `G` in W/K, heat-input matrix/normalization, temperature reconstruction matrix, equilibrium reference temperatures and boundary-input definitions. If an arbitrary state-space realization is supplied instead, its states need not be physical kelvins; give the units of all inputs/outputs and the state normalization. State-space matrices are useful only with their port mapping and validated frequency/temperature range.

## Specific questions that close the largest uncertainties

1. Does the capstone use the 2026 eight-MRM Tx bank, an earlier device, or another ring platform? What is the actual Tx monitor topology, and does the control interface expose only photocurrent or also temperature/reference power/bit statistics?
2. At which wavelength, temperature, bias and optical power was approximately 15 GHz/K measured? Is it a local derivative of resonance frequency, an engineering estimate, or another metric? Can a signed frequency-versus-temperature curve be provided?
3. Can we obtain the raw fast-aggressor temperature/command trace? The March paper states 808 K/s and a 100 ms period; the April slide shows a different >2000 K/s nonlinear waveform. Are these different tests/devices/calibrations?
4. Are the stabilizer's codes linear in voltage, current, power, PWM duty, or a calibrated tuning variable? What are its update cadence, resolution, current/voltage compliance, slew, sample/hold and settling behavior?
5. What is the target optical operating point: center of drop peak, one resonance flank, average monitor target, OMA optimum, or a bias-dependent target? How are sign/branch ambiguity and power drift resolved?
6. Can through/drop spectra and junction-bias response maps be shared over temperature and optical power? Do loss/coupling/Q vary materially with heater or bias? What resonance-order spacing and backscatter splitting are measured?
7. Can heater and package dynamic responses be shared independently? Are ring, heater-strip and substrate temperatures distinct in the internal model? Is there a package thermal ROM with an output for each ring?
8. What are the acceptable detuning and optical-margin envelopes, acquisition/recovery time, disturbance maximums and lock-excursion definitions? Is 125 degrees C an allowed hardware condition or only a simulation/extrapolation requirement?
9. For published BER plots, is each point an instantaneous-window or cumulative count? Please include errors, bits observed, integration intervals, confidence limits and overflow/reset behavior. What time resolution is available during a short excursion?
10. Which parameters vary together across wafers/dies/packages? Can per-device joint sets or anonymized fitted transfer functions be supplied rather than independent min/max ranges?

## Additional data if first-principles extraction becomes necessary

For **optical extraction**, request round-trip geometry, bend radii/straight sections, waveguide cross-section and sidewall angles, claddings, complex dispersive refractive indices, coupler gaps/lengths/topology, relevant mode/polarization, doping overlap, propagation/bend/coupling loss, reflection/roughness assumptions, and measured FSR/Q for validation. The public approximately 15 micrometre diameter and 400 by 200 nm illustration are not a complete geometry.

For **electrical/semiconductor extraction**, request doping profiles and activation, junction/contact layout, carrier mobility/recombination laws, electrical permittivity, reference carrier densities, leakage/breakdown, DC I-V, C-V, access/contact resistance, driver impedance and package parasitics. Without these, measured bias-to-frequency/loss maps are preferable to arbitrary TCAD inputs.

For **thermal extraction**, request layer thicknesses, conductivity tensors and temperature dependence, density and heat capacity, heater dimensions/material/resistivity/TCR, interfacial thermal resistances, substrate and package boundaries, heat sinks/coolant/convection assumptions, heat-source locations, neighboring electronics power maps, and optical absorbed-power distribution. A source power in W and a temperature in K do not alone determine a spatial heat problem.

For **nonlinear optical effects**, request power-dependent spectra, absorption fractions, effective mode volume/overlaps, Kerr response, two-photon/free-carrier absorption coefficients and carrier lifetime where relevant. At O-band and the relevant power range these coefficients should be measured or traced to appropriate materials/wavelengths rather than copied from a generic C-band example.

These details are optional if compact measured relationships suffice. Do not delay the control work solely to collect a proprietary geometric stack.

## Acceptance and reporting

Fit on one set of traces; test on independent traces, temperatures, input powers and ideally devices. Report resonance and monitor prediction error, timing/phase error, worst tracking excursion, actuator saturation/headroom, recovery time, and optical margin separately. Use measured error counters to claim BER agreement; label link-noise calculations as estimates.

If only ranges are available, report a robust-controller study over an uncertainty set. If fitted traces are available, report calibration and the validated domain. Reserve the term validated digital twin for independently verified hardware correspondence over the claimed domain. The full source/method discussion is in [research.md](research.md).
