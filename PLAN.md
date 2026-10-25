# TLC Porting Plan & Open Items

**Target Repository:** `mrm-model`  
**Active Branch:** `feature/tlc-port`  
**Reference Document:** [`docs/TLC_PORT_SPEC_AND_PLAN.md`](docs/TLC_PORT_SPEC_AND_PLAN.md)  
**Last Updated:** Phase 1 & Phase 2 Complete (Commit `5d1cac0`)

---

## 1. Executive Status

| Phase | Description | Status | Verification |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Signal Processing Frontend (B1 Dither, B2 Conditioner, B3 Lock-In) | **COMPLETED** | 38 / 38 unit & stress tests passing |
| **Phase 2** | State Estimation & Setpoints (B4 Thermal ROM, B4 UKF, B6 Setpoint) | **COMPLETED** | 102 / 102 unit & stress tests passing |
| **Phase 3** | Control & Actuation (B7 MPC Solver, B7 PI Fallback, B8 Output Slew Limiter) | **OPEN** | Unit tests planned in `tests/tlc/` |
| **Phase 4** | Supervisory & Top-Level Adapter (B5 Supervisor, B9 Diagnostics, B10 RLS, `api_adapter.py`) | **OPEN** | Integration with `BaseController` |
| **Phase 5** | Closed-Loop Benchmarking on Lightmatter G26 Scenarios | **OPEN** | Comparison against baseline PI & MPC |

**Test Suite Health:**  
- `tests/tlc/`: **140 / 140 passing** (100% green)
- `tests/verification/`: **72 / 72 passing** (100% green baseline regression)

---

## 2. Completed Milestones

### Phase 1: Frontend Signal Processing
- [x] **B1 Dither Generator (`mrm/control/tlc/b1_dither.py`)**: 8 orthogonal tones on 250 Hz grid (`1.5` to `5.0 kHz`) over 4 ms window ($T_s = 100\ \mu\text{s}$) with dynamic amplitude scheduling and cross-channel leakage $< -40\text{ dB}$.
- [x] **B2 Conditioner (`mrm/control/tlc/b2_conditioner.py`)**: Bandpass FIR filter isolating $f_1, f_2$; sample-variance squelch detector rejecting clipped, low-SNR, or dead monitor ADC signals.
- [x] **B3 Lock-In Demodulator (`mrm/control/tlc/b3_lockin.py`)**: Sliding 40-tap Hann windowed I/Q demodulator extracting $y_1$ and $y_2$; computes power-invariant harmonic ratio $\zeta_i = y_{2,i}/y_{1,i}$.

### Phase 2: State Estimation & Setpoint
- [x] **B4 Reduced Thermal Model (`mrm/control/tlc/b4_thermal_model.py`)**: Discrete-time $(A_d, B_d, C_d)$ state-space model capturing G26 self-heating, nearest-neighbor crosstalk, and substrate dynamics across 8 rings.
- [x] **B4 Dual-Rate UKF (`mrm/control/tlc/b4_ukf.py`)**: 24-state filter fusing fast 100 µs DC levels and slow 250 Hz harmonic ratio $\zeta$ with Huber innovation gating and multi-batch broadcasting.
- [x] **B6 Setpoint Generator (`mrm/control/tlc/b6_setpoints.py`)**: Calculates red-flank inflection point ($\Delta\lambda^* \approx +124\text{ pm}$) where negative thermo-optic feedback prevents thermal runaway.

---

## 3. Open Items & Detailed Work Breakdown

### OPEN-1: Phase 3 — Control & Actuation Blocks
- [ ] **B7 MPC Fast Gradient Method Solver (`mrm/control/tlc/b7_mpc.py`)**:
  - Implement QP formulation over horizon $H=20$ steps (2 ms).
  - Cost function: penalize tracking error $(\delta_i - \delta^*)^2$, control effort $\Delta u^2$, and total electrical power $\sum P_i$.
  - Box constraints: $[0, 65535]$ DAC codes; slew constraints $\le 500\text{ codes/step}$.
  - Target execution budget: $< 30\ \mu\text{s}$ per 8-ring solve.
  - Test: `tests/tlc/test_b7_mpc.py` (compare against SciPy QP ground truth).
- [ ] **B7 PI Fallback (`mrm/control/tlc/b7_pi.py`)**:
  - Decentralized anti-windup PI controller for squelched channels or supervisor fallback.
  - Bumpless transfer logic matching integrator state to last MPC output.
- [ ] **B8 Output Limiter & Channel Guard (`mrm/control/tlc/b8_output.py`)**:
  - Combine MPC heater commands with B1 dither signals: $u = u_{\text{mpc}} + u_{\text{dither}}$.
  - Quantize floating power to 16-bit integer DAC codes `[0, 65535]`.
  - Rate limiter: clamp slew rate to $\le 200\text{ codes}/\mu\text{s}$.
  - Channel guard: clamp maximum red detuning to $< +200\text{ GHz}$ to prevent collision with adjacent WDM channels.
  - Test: `tests/tlc/test_b8_output.py`.

### OPEN-2: Phase 4 — Supervisory, Online Adaptation & Top-Level Adapter
- [ ] **B5 Supervisor State Machine (`mrm/control/tlc/b5_supervisor.py`)**:
  - Implement 6-state FSM: `CALIBRATE` $\to$ `SEARCH` $\to$ `LOCK_ACQUIRE` $\to$ `TRACK` $\to$ `FALLBACK_PI` $\to$ `HOLD`.
  - Cold-start heater ramp (2 GHz/ms) with laser comb slope recognition to avoid locking onto wrong comb lines.
  - Squelch transition logic: switch to `HOLD` or `FALLBACK_PI` when monitor photo-current drops below threshold.
- [ ] **B9 Diagnostics & Telemetry (`mrm/control/tlc/b9_diagnostics.py`)**:
  - Format controller diagnostics into `mrm.control.api.TELEMETRY_KEYS` dictionary (`delta_hat`, `rate_hat`, `mode`, `locked`, `dither_amp`, `alarms`, `oma_penalty`).
- [ ] **B10 Online RLS Calibration (`mrm/control/tlc/b10_calibration.py`)**:
  - Recursive Least Squares estimator tracking real heater efficiency (pm/DAC code) and inter-ring thermal coupling matrix $K_{ij}$ from dither harmonics.
  - Periodic model update with bumpless state re-scaling.
- [ ] **Top-Level Controller (`mrm/control/tlc/controller.py` & `api_adapter.py`)**:
  - Subclass `mrm.control.api.BaseController`.
  - Wire all blocks B1–B10 into `step(obs: Observation, spec: SpecNominal) -> dict[str, np.ndarray]`.
  - Register as `@register("tlc")` in controller registry.
  - Verify compliance with `tests/verification/test_control_api.py`.

### OPEN-3: Phase 5 — Closed-Loop Benchmark Execution
- [ ] **Benchmark 1: Steady Lock (`experiments/examples/steady_pi_lock.yaml`)**:
  - 40 ms warm start at 25 °C across 8 rings.
  - Target: rms lock error $< 0.1\text{ pm}$, OMA penalty $< 0.05\text{ dB}$, 0 excursions.
- [ ] **Benchmark 2: Fast Aggressor A1 (`experiments/examples/a1_pi_lock.yaml`)**:
  - Slide-10 aggressor thermal swing (88.5 K step at $+2200 / -2300\text{ K/s}$).
  - Verify fast DC path and MPC feedforward prevent loss-of-lock.
- [ ] **Benchmark 3: Cold-Start Acquisition**:
  - Ramp from DAC code 0 to lock point on 400 GHz comb; verify 0 wrong-line locks.
- [ ] **Benchmark 4: Monte Carlo Robustness (`experiments/examples/robust_pi_lock.yaml`)**:
  - 128 sampled uncertainty worlds; assess lock stability pass rate.
- [ ] **Final Scorecard Report**: Document TLC performance vs. `pi_lock` baseline in `docs/handoff/controller_scorecard.md`.
