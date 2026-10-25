# E2E Test Infra: TLC Port to `mrm-model`

## Test Philosophy
- Opaque-box, requirement-driven derived from `ORIGINAL_REQUEST.md` and `PROJECT.md`.
- Comprehensive multi-tier coverage:
  - **Tier 1**: Feature Coverage (spectral purity, FIR filtering, lock-in demodulation, UKF convergence, inflection setpoints).
  - **Tier 2**: Boundary & Corner Cases (clipping ADC, stuck bits, zero/max heater codes, squelch transitions, NaN handling).
  - **Tier 3**: Cross-Feature Combinations (dither + conditioner + lock-in pipeline, UKF tracking under active dither).
  - **Tier 4**: Real-World Application Scenarios (thermal step disturbance, aggressive heating ramps at 2200 K/s, crosstalk compensation).

## Feature Inventory & Test Mapping
| # | Feature | Source (requirement) | Test Target File | Tier 1 | Tier 2 | Tier 3 |
|---|---------|----------------------|------------------|:------:|:------:|:------:|
| 1 | B1 Dither Generation | ORIGINAL_REQUEST §R1 | `tests/tlc/test_b1_dither.py` | 5 | 5 | ✓ |
| 2 | B1 Adaptive Amplitude Scheduling | ECE496 Ref & G26 Spec | `tests/tlc/test_b1_dither.py` | 5 | 5 | ✓ |
| 3 | B2 FIR Filtering & Imputation | ORIGINAL_REQUEST §R1 | `tests/tlc/test_b2_conditioner.py` | 5 | 5 | ✓ |
| 4 | B2 Squelch Variance Detector | ORIGINAL_REQUEST §R1 | `tests/tlc/test_b2_conditioner.py` | 5 | 5 | ✓ |
| 5 | B3 Lock-In Demodulator (y1, y2) | ORIGINAL_REQUEST §R1 | `tests/tlc/test_b3_lockin.py` | 5 | 5 | ✓ |
| 6 | B3 Harmonic Ratio $\zeta$ | ORIGINAL_REQUEST §R1 | `tests/tlc/test_b3_lockin.py` | 5 | 5 | ✓ |
| 7 | B4 Reduced Thermal Model ($A_d, B_d, C_d$) | ORIGINAL_REQUEST §R2 | `tests/tlc/test_b4_ukf.py` | 5 | 5 | ✓ |
| 8 | B4 Dual-Rate UKF State Estimation | ORIGINAL_REQUEST §R2 | `tests/tlc/test_b4_ukf.py` | 5 | 5 | ✓ |
| 9 | B6 Red-Flank Setpoint Generation | ORIGINAL_REQUEST §R2 | `tests/tlc/test_b6_setpoints.py` | 5 | 5 | ✓ |
| 10 | Regression Invariance | ORIGINAL_REQUEST §R3 | `tests/verification/` | 72 | - | - |

## Test Architecture
- **Runner**: `.venv/bin/pytest tests/tlc/ -v`
- **Regression Runner**: `.venv/bin/pytest tests/verification/test_control_api.py tests/verification/test_plant_wiring.py -v`
- **Format**: Standard pytest test files under `tests/tlc/`
- **Verification Metrics**:
  - B1: Crosstalk rejection $> 40\text{ dB}$, zero harmonic collisions below 5 kHz.
  - B2: Noise gain $\approx 0.52$, squelch assertion on $> 15\text{ dB}$ variance drop, NaN sample handling.
  - B3: Demodulation amplitude and ratio error $\le 2\%$, invariance to laser power.
  - B4: Estimation error bounds satisfied under thermal step and crosstalk disturbances, discrete-time stability of $A_d$ (spectral radius $< 1$).
  - B6: Inflection point detuning $\approx +124\text{ pm}$ matches $d^2L/d\delta^2 = 0$ with negative thermo-optic slope.
