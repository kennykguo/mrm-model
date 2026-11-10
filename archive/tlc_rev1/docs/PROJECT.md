# Project: TLC Port to `mrm-model` (Lightmatter G26)

## Architecture
The Thermal Locking Controller (TLC) ports the reference ECE496 multi-ring thermal stabilization pipeline into `mrm-model` for 8-ring Lightmatter G26 microring modulators.

```
                   +-------------------------------------------------------------+
                   |                 TlcController (BaseController)              |
                   |                                                             |
Plant Sensors      |  +--------------------+      +---------------------------+  |
[tx.monitor.samples|->|   B2 Conditioner   |----->|     B3 Lock-In Demod      |  |
 int32 (B, N, M)]  |  | (FIR + Squelch Det)|      | (Sliding 4ms Hann window) |  |
                   |  +--------------------+      +---------------------------+  |
                   |            |                               |                |
                   |       y0 (DC level)               zeta (y2/y1 ratio)        |
                   |            |                               |                |
                   |            v                               v                |
                   |  +-----------------------------------------------+          |
                   |  |                     B4 UKF                    |          |
                   |  | (Dual-rate 24-state UKF: delta, r, p)         |          |
                   |  | Uses: B4 Reduced Thermal Model (Ad, Bd, Cd)   |          |
                   |  +-----------------------------------------------+          |
                   |            |                               ^                |
                   |            | delta_hat (est detuning)      | sigma_delta    |
                   |            v                               |                |
                   |  +--------------------+      +---------------------------+  |
                   |  |    B6 Setpoints    |      |         B1 Dither         |  |
                   |  | (Red-flank inflect |      | (8 orthogonal tones       |  |
                   |  |  delta* ~ +124 pm) |      |  adaptive amp scheduling) |  |
                   |  +--------------------+      +---------------------------+  |
                   |            |                               |                |
                   |            +---------------+---------------+                |
                   |                            |                                |
Plant Actuators    |                            v                                |
[tx.heater codes   |<-------------------+----------------+                       |
 uint16 (B, N)]    |                    |  Heater DAC    |                       |
                   |                    | Quantization   |                       |
                   |                    +----------------+                       |
                   +-------------------------------------------------------------+
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | B1 Dither Generation | 8 orthogonal dither tones on 250 Hz spacing grid `[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] kHz` over 4 ms window (100 us step rate), zero 2nd-order intermod or harmonic collisions below 5 kHz | M1 | ORIGINAL_REQUEST §R1 |
| 2 | B1 Adaptive Amplitude Scheduling | Dynamic tone amplitude scaling based on UKF uncertainty $\sigma_\delta$ scaled by ring gain $1 / (|H_{ii}| \cdot \hat{\theta}_i)$ | M1 | ECE496 Ref & G26 Spec |
| 3 | B2 FIR Filtering | 31-tap linear-phase FIR lowpass filter, $f_c \le 100\text{ kHz}$ for 250 kS/s ADC, noise gain $\approx 0.52$, NaN sample imputation | M1 | ORIGINAL_REQUEST §R1 |
| 4 | B2 Squelch Variance Detector | 2nd-difference variance detector $\text{std}(d_2)/\sqrt{6}$ with hysteresis flagging modulation collapse $> 15\text{ dB}$ | M1 | ORIGINAL_REQUEST §R1 |
| 5 | B3 Lock-In Demodulator | Sliding 4 ms Hann-windowed I/Q demodulator extracting in-phase and quadrature amplitudes at fundamental ($y_1$) and 2nd harmonic ($y_2$), plus DC ($y_0$) | M1 | ORIGINAL_REQUEST §R1 |
| 6 | B3 Harmonic Ratio $\zeta$ | Power-invariant harmonic ratio $\zeta_i = y_{2,i}/y_{1,i}$ with numerical recovery error $\le 2\%$ | M1 | ORIGINAL_REQUEST §R1 |
| 7 | B4 Reduced Thermal Model | Discrete-time state-space $(A_d, B_d, C_d)$ capturing G26 self-heating, slow heater pole ($\tau_{\text{slow}} \approx 13.01\ \mu\text{s}$), package tail ($\tau_{\text{tail}} \approx 3.0\text{ ms}$), and nearest-neighbor crosstalk matrix $K_{ij}$ at $T_s = 100\ \mu\text{s}$ | M2 | ORIGINAL_REQUEST §R2 |
| 8 | B4 Dual-Rate UKF | Dual-rate UKF fusing fast 100 us DC monitor levels ($y_0$) and slow 250 Hz harmonic ratios ($\zeta$) estimating 24 thermal/detuning states across 8 rings with Huber soft gating | M2 | ORIGINAL_REQUEST §R2 |
| 9 | B6 Red-Flank Setpoints | Solves $d^2 L / d\delta^2 = 0$ on data-averaged resonance lineshape ($\approx +124\text{ pm}$ red of laser line) providing negative thermo-optic feedback stabilization against thermal runaway | M2 | ORIGINAL_REQUEST §R2 |
| 10 | TlcController Integration | Subclass `BaseController` conforming to `mrm.control.api`, orchestrating B1-B6, batched across $(B, N)$ | M3 | G26 Spec & Control API |
| 11 | Comprehensive TLC Test Suite | `tests/tlc/` unit & E2E tests for B1, B2, B3, B4, B6 with $>40\text{ dB}$ crosstalk rejection, $\le 2\%$ demod error, UKF convergence, setpoint inflection | MT | ORIGINAL_REQUEST §Acceptance |
| 12 | Regression Invariance | Maintain 100% pass rate (72/72) on existing `tests/verification/test_control_api.py` and `tests/verification/test_plant_wiring.py` | M3 | ORIGINAL_REQUEST §R3 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| MT | E2E Testing Track | Test infra and comprehensive test suites (`tests/tlc/test_b1_dither.py`, `test_b2_conditioner.py`, `test_b3_lockin.py`, `test_b4_ukf.py`, `test_b6_setpoints.py`), publish `TEST_READY.md` | none | IN_PROGRESS |
| M1 | Signal Processing Frontend | Implement `mrm/control/tlc/b1_dither.py`, `b2_conditioner.py`, `b3_lockin.py` | none | IN_PROGRESS |
| M2 | State Estimation & Setpoints | Implement `mrm/control/tlc/b4_thermal_model.py`, `b4_ukf.py`, `b6_setpoints.py` | M1 (interfaces) | PLANNED |
| M3 | Integration, Regression & Audit | Implement `mrm/control/tlc/__init__.py`, `controller.py`, verify 100% tests pass (TLC + 72 verification), adversarial verification, forensic audit | MT, M1, M2 | PLANNED |

## Interface Contracts

### B1 Dither Generator (`b1_dither.py`)
```python
class DitherConfig:
    frequencies_hz: list[float] = [1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0]
    sample_rate_hz: float = 10000.0  # 1 / 100 us
    window_length_s: float = 0.004   # 4 ms = 40 samples
    nominal_amplitude_pm: float = 25.0
    min_amplitude_pm: float = 5.0
    max_amplitude_pm: float = 80.0

class DitherGenerator:
    def __init__(self, n_rings: int = 8, config: DitherConfig | None = None) -> None: ...
    def reset(self, batch_size: int = 1) -> None: ...
    def step(self, sigma_delta: np.ndarray | None = None) -> np.ndarray:
        """Returns dither detuning offset in pm, shape (B, N)."""
```

### B2 Conditioner (`b2_conditioner.py`)
```python
class ConditionerConfig:
    adc_rate_hz: float = 250000.0
    cutoff_hz: float = 80000.0
    filter_taps: int = 31
    squelch_drop_db: float = 15.0

class SignalConditioner:
    def __init__(self, n_rings: int = 8, config: ConditionerConfig | None = None) -> None: ...
    def reset(self, batch_size: int = 1) -> None: ...
    def process_samples(self, samples: np.ndarray, count: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes ADC monitor samples (B, N, M).
        Returns:
            filtered_dc: (B, N) DC monitor photodiode power / code
            squelch_active: (B, N) bool mask indicating squelch condition
            noise_std: (B, N) high-frequency noise estimate
        """
```

### B3 Lock-In Demodulator (`b3_lockin.py`)
```python
class LockInConfig:
    sample_rate_hz: float = 10000.0
    window_length_s: float = 0.004  # 40 steps
    harmonic_order: int = 2

class LockInDemodulator:
    def __init__(self, n_rings: int = 8, dither_frequencies: list[float] | None = None, config: LockInConfig | None = None) -> None: ...
    def reset(self, batch_size: int = 1) -> None: ...
    def step(self, monitor_dc: np.ndarray, dither_detuning: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            ready: bool (True when 4ms window is complete)
            y1: (B, N) fundamental amplitude
            y2: (B, N) 2nd harmonic amplitude
            zeta: (B, N) harmonic ratio y2 / y1
        """
```

### B4 Reduced Thermal Model (`b4_thermal_model.py`)
```python
class ReducedThermalModel:
    def __init__(self, n_rings: int = 8, dt: float = 1e-4, preset_params: dict | None = None) -> None: ...
    def step(self, x: np.ndarray, u_heater_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Discrete-time state update:
        x[k+1] = Ad @ x[k] + Bd @ u[k]
        y[k]   = Cd @ x[k] + Dd @ u[k]
        Returns: (x_next, delta_lambda_thermal_pm)
        """
```

### B4 Dual-Rate UKF (`b4_ukf.py`)
```python
class DualRateUKF:
    def __init__(self, n_rings: int = 8, dt_fast: float = 1e-4, dt_slow: float = 4e-3) -> None: ...
    def reset(self, batch_size: int = 1) -> None: ...
    def predict(self, u_heater_power_w: np.ndarray) -> None: ...
    def update_fast(self, y0_dc: np.ndarray) -> None: ...
    def update_slow(self, zeta: np.ndarray, squelch_mask: np.ndarray) -> None: ...
    def get_estimates(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns: (delta_hat, sigma_delta) with shapes (B, N)."""
```

### B6 Setpoint Generator (`b6_setpoints.py`)
```python
class SetpointGenerator:
    def __init__(self, n_rings: int = 8, q_loaded: float = 3850.0, wavelength_nm: float = 1310.0) -> None: ...
    def calculate_inflection_setpoint(self) -> float:
        """Computes optimal red-flank inflection point in pm (~ +124 pm)."""
    def get_setpoints(self, batch_size: int = 1) -> np.ndarray:
        """Returns target detunings of shape (B, N)."""
```

## Code Layout
- `mrm/control/tlc/`:
  - `__init__.py`: Package exports (`TlcController`, `DitherGenerator`, `SignalConditioner`, `LockInDemodulator`, `ReducedThermalModel`, `DualRateUKF`, `SetpointGenerator`)
  - `b1_dither.py`: Block B1 Dither Generator
  - `b2_conditioner.py`: Block B2 Signal Conditioner & Squelch
  - `b3_lockin.py`: Block B3 Lock-In Demodulator
  - `b4_thermal_model.py`: Block B4 Reduced Thermal State-Space Model
  - `b4_ukf.py`: Block B4 Dual-Rate Unscented Kalman Filter
  - `b6_setpoints.py`: Block B6 Red-Flank Inflection Setpoint Generator
  - `controller.py`: `TlcController(BaseController)` top-level integration
- `tests/tlc/`:
  - `test_b1_dither.py`: Spectral purity, orthogonality > 40 dB, cycle counts
  - `test_b2_conditioner.py`: FIR response, squelch variance detector, NaN handling
  - `test_b3_lockin.py`: Hann I/Q demodulation, known amplitude recovery, ratio zeta <= 2% error
  - `test_b4_ukf.py`: State estimation convergence, reduced thermal model Ad/Bd/Cd, crosstalk tracking
  - `test_b6_setpoints.py`: Inflection target calculation, red-flank negative thermo-optic slope
- `tests/verification/`:
  - `test_control_api.py`: Baseline regression suite (must remain 100% passing)
  - `test_plant_wiring.py`: Baseline regression suite (must remain 100% passing)
