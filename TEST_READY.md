# TEST_READY — TLC Port to `mrm-model` (Lightmatter G26)

## Test Suite Status: READY

The comprehensive unit and E2E test suite for the Thermal Locking Controller (TLC) Phase 1 (Signal Processing) and Phase 2 (State Estimation & Setpoints) has been authored and verified in `tests/tlc/`.

---

## 1. Test Suite Summary & Verification Metrics

| Test Target | Covered Block | Spec Source | Test Count | Key Verification Metrics & Assertions |
| :--- | :--- | :--- | :---: | :--- |
| `tests/tlc/test_b1_dither.py` | B1 Dither Generator | `PROJECT.md` §Interface Contracts, `ORIGINAL_REQUEST.md` §R1 | 13 | 8-tone grid `[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] kHz`; exact integer cycles over 4 ms `[6, 8, 10, 12, 14, 16, 18, 20]`; pairwise crosstalk rejection $> 40\text{ dB}$; FFT spurious suppression $> 40\text{ dB}$; adaptive amplitude scaling with $\sigma_\delta \in [5, 80]\text{ pm}$; rate limiting; NaN/negative handling; batch vectorization $(B, 8)$. |
| `tests/tlc/test_b2_conditioner.py` | B2 Signal Conditioner | `PROJECT.md` §Interface Contracts, `ORIGINAL_REQUEST.md` §R1 | 13 | 31-tap FIR filter; DC unity gain; stopband attenuation $> 15\text{ dB}$; theoretical noise gain $\approx 0.52$; 2nd-difference noise floor estimator $\text{std}(d_2)/\sqrt{6}$; squelch tripping on $> 15\text{ dB}$ variance drop; hysteresis anti-chatter; NaN sample imputation; stuck bit & rail saturation detection; state continuity across step boundaries. |
| `tests/tlc/test_b3_lockin.py` | B3 Lock-In Demodulator | `PROJECT.md` §Interface Contracts, `ORIGINAL_REQUEST.md` §R1 | 12 | Sliding 4 ms Hann window with priming `ready` flag; fundamental $y_1$ recovery error $\le 2\%$; 2nd harmonic $y_2$ and ratio $\zeta = y_2 / y_1$ recovery error $\le 2\%$; strict optical power invariance ($< 0.5\%$ error across $0.1\times$ to $5\times$ power scaling); red-flank inflection zero-crossing ($\zeta \approx 0$ when $y_2 = 0$); multi-channel cross-tone isolation $> 40\text{ dB}$; quadrature phase robustness; zero-division guard when $y_1 \to 0$. |
| `tests/tlc/test_b4_ukf.py` | B4 Thermal Model & Dual-Rate UKF | `PROJECT.md` §Interface Contracts, `ORIGINAL_REQUEST.md` §R2 | 12 | Reduced thermal model discrete state-space $(A_d, B_d, C_d)$ stability (spectral radius $\rho(A_d) < 1.0$); nearest-neighbor crosstalk matrix symmetry & spatial decay; monotonic step response; 24-state UKF $(\delta, r, p)$; steady-state convergence ($\le 5\text{ pm}$ error); tracking $+50\text{ pm}$ thermal step disturbance; Huber innovation gating rejecting $> 25\text{ NIS}$ glitches; cross-ring outlier decoupling; squelch scale freezing; covariance positive-definiteness. |
| `tests/tlc/test_b6_setpoints.py` | B6 Setpoint Generator | `PROJECT.md` §Interface Contracts, `ORIGINAL_REQUEST.md` §R2 | 10 | Red-flank inflection point solving $d^2L/d\delta^2 = 0$ yielding nominal target $\approx +124\text{ pm}$ (clamped in $[122, 126]\text{ pm}$); negative feedback slope $dL/d\delta < 0$ for thermo-optic stabilization; slope magnitude $|\frac{dL}{d\delta}|$ maximized at inflection; bare Lorentzian fallback $\text{HWHM}/\sqrt{3} \approx 98.22\text{ pm}$; linewidth/Q scaling; target override; vectorized $(B, N)$ output. |
| **Total TLC Test Cases** | | | **60** | **All 60 collected cleanly; ready for M1 and M2 implementations** |

---

## 2. Test Execution Commands & Baseline Verification

### 2.1 Test Collection Check
```bash
.venv/bin/pytest --collect-only tests/tlc/
```
**Result:** 60/60 tests collected in 0.12s without syntax or collection errors.

### 2.2 TLC Execution Check (Pending Implementation)
```bash
.venv/bin/pytest tests/tlc/ -v
```
**Result:** 60 skipped in 0.11s with clear skip reasons until implementation classes in `mrm.control.tlc.*` are implemented.

### 2.3 Baseline Regression Invariance
```bash
.venv/bin/pytest tests/verification/test_control_api.py tests/verification/test_plant_wiring.py -v
```
**Result:** **72 passed in 14.93s (100% green)**. Zero regression introduced.

---

## 3. Implementation Verification Instructions for Implementers

Implementers (M1 Signal Processing and M2 State Estimation & Setpoints) should run targeted tests as each block is finished:
- **Block B1**: `.venv/bin/pytest tests/tlc/test_b1_dither.py -v`
- **Block B2**: `.venv/bin/pytest tests/tlc/test_b2_conditioner.py -v`
- **Block B3**: `.venv/bin/pytest tests/tlc/test_b3_lockin.py -v`
- **Block B4**: `.venv/bin/pytest tests/tlc/test_b4_ukf.py -v`
- **Block B6**: `.venv/bin/pytest tests/tlc/test_b6_setpoints.py -v`
- **Full TLC Suite**: `.venv/bin/pytest tests/tlc/ -v`
