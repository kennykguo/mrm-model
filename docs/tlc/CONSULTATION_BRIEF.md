# Porting a Dither-Based Thermal Locking Controller to a High-Fidelity Microring Transmitter Model
**Consultation brief: project goal, methodology, and open questions**
A. Patel · ECE496 capstone / mrm-model · 27 October 2026

## 1. Context and objective
Microring modulators (MRMs) in WDM transmitters must be thermally locked. Each resonance drifts by about 74 pm/K, so it has to be held on the long-wavelength (red) flank of its laser line, where optical self-heating gives negative feedback.

The ECE496 project designed a ten-block **Thermal Locking Controller (TLC)**:
- **Sensing:** per-ring heater dither tones, lock-in demodulation of the 1st and 2nd harmonics, and the power-invariant ratio ζ = y₂/y₁.
- **Estimation and control:** a dual-rate UKF fusing ζ with the fast DC monitor level, and a horizon-20 fast-gradient MPC.
- **Supervision:** a supervisor FSM, PI fallback, and online RLS calibration.

The TLC was validated on a purpose-built **4-ring plant** (Lorentzian lineshape, two-pole heater, 20 % crosstalk, 12-bit PD). There it met 15 of 17 requirements and beat a PI baseline under fast on-die aggressors.

**Objective.** Run the same TLC, closed loop, against **mrm-model**, an independent 8-ring simulator of the Lightmatter G26 transmitter, and collect performance and locking metrics. mrm-model adds:
- calibrated three-pole thermal dynamics and a 16-bit voltage-mode heater DAC (power ∝ code²);
- a 250 kS/s integrate-and-dump monitor ADC;
- an 88 K on-die aggressor, Monte Carlo device "worlds", and knowledge-gated controller access to plant parameters.

The deliverable for this phase is the **evaluation framework**, not a tuned controller. One command should produce a complete, reproducible metrics report. Poor results are acceptable if they are measured and explained.

## 2. Method
1. **Faithful port.** Transplant the TLC core unchanged, in its native units (GHz, heater power fraction). Only four deviations are allowed: ring count as a parameter, injected parameters, the monitor sample rate, and plant priors drawn from the simulator's nominal data.
2. **Adapter layer.** All simulator-specific translation sits outside the core:
   - ADC codes to normalised level, and heater power to DAC code;
   - a multirate schedule: the simulator calls the controller every 10 µs, and the core runs every 100 µs, emitting 10 dithered heater samples (the original 100 kS/s output stream);
   - telemetry capture.
3. **Fidelity gate.** The ported core must reproduce the original TLC's published metrics on the original 4-ring plant before any G26 run.
4. **Scenario matrix.** Map the ECE496 requirement set (R1–R22, disturbances D1–D6) to the simulator's scenarios: static, thermal ramp, A1/A2 aggressors, laser drift, power step, squelch, cold start, MC campaign. Three cases have no equivalent: common-mode aggressor, +10 K kick, and PD dropout.
5. **Report.** Per-ring p99.9 / peak / rms error, recovery time, lock time, wrong-line locks, OMA penalty, heater power and slew. Present it side by side (TLC on the 4-ring plant vs TLC on G26) and against the simulator's baselines (PI, Kalman-LQG, MPC, dither lock-in).

## 3. Expected discrepancies, to be measured rather than fixed
- **Squelch detection.** The TLC detects squelch from vanishing data-pattern noise, but the G26 monitor sees only the NRZ-averaged level, so the detector should stay silent.
- **Cold-start acquisition.** Rings start about 7.6 nm blue of their lines, and the ramp crosses foreign lines, so wrong-line locks are expected.
- **Dither scoring.** The simulator scores true detuning including the dither, so a dither of ±5 pm or more counts as lock excursions.
- **Heater authority.** Ring 0 exhausts its heater under the A1 aggressor for every controller.
- **Weak ζ signal.** Over a lineshape roughly 3× wider, the ζ channel's SNR may be marginal.

A previous port attempt was withdrawn after review: its lock-in discarded sign, its tone plan collided, and its tests were self-referential.

## 4. Questions for consultation
1. **Validity of comparison.** Is a faithful, untuned port the right scientific baseline for judging the controller on a higher-fidelity plant? Or should a minimal, principled re-parameterisation be allowed first (dither amplitude, model gains)? How should the two be reported?
2. **Fairness of prior knowledge.** The TLC needs nominal lineshape, heater and thermal parameters. Should it receive the same parameter access as the simulator's model-based baselines, or more with disclosure? How should information asymmetry between controllers be reported?
3. **Acceptance criteria.** Should results be judged against the ECE496 thresholds converted to wavelength (2 GHz ≈ 11.4 pm, 6 GHz ≈ 34 pm), against the simulator's 5 pm lock budget, or both? Should excursions caused by the controller's own dither be separated from tracking error?
4. **Unmapped scenarios.** For the common-mode aggressor, temperature kick and PD dropout: omit them, extend the simulator, or inject them at the controller boundary? Which choice threatens validity least?
5. **Statistical design.** Given a Python core at about 3 ms per 100 µs step, what is the minimum scenario duration and number of Monte Carlo worlds for defensible claims, e.g. the wrong-line lock rate (originally 0 of 40 trials)?
6. **Comparators.** Which baselines make the comparison meaningful: the TLC's own PI fallback, the simulator's PI, Kalman-LQG and MPC, or a dither lock-in?
7. **Model trust.** The simulator's thermal poles are literature-derived and its crosstalk package is uncertain. How much weight should G26 results carry relative to the original plant? Which corner worlds should be mandatory?

*Decisions internal to the team (repository layout, code provenance, push target) are not raised here.*
