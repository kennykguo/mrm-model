# Adversarial review of the rev 1 TLC port (2026-10-26)

Scope: commits `5d1cac0` (code) and `1387fe6` (PLAN.md), plus rev 1 of `docs/TLC_PORT_SPEC_AND_PLAN.md`.
Source compared: `ECE496_project@f98ce84`.

## Ported blocks
`tests/tlc` passes 140/140 in about 3.7 s on a system Python 3.13. There is no `.venv`, although `TEST_INFRA.md` (now `archive/tlc_rev1/docs/`) references one. The suite emits 639 numpy warnings that nobody has examined.

| Block | Verdict | Why |
|---|---|---|
| B1 dither | REWRITE | The source's collision-free tones (2.0/3.0/3.5/4.5 kHz) were replaced by an even 1.5–5.0 kHz grid. On that grid 2×1.5=3.0, 2×2.0=4.0 and 2×2.5=5.0 fall on other rings' fundamentals, and every f_i±f_j lands on the grid. The docstring at `b1_dither.py:10` claims "zero collisions", and the test meant to check it asserts `2f != f`. Amplitudes (5–80 pm) and `heater_gain=0.05 pm/code` have no source. The output gives one value per 100 µs; the source used a 100 kS/s stream. |
| B2 conditioner | FIX | The FIR, validity and noise logic match the source. Rates are hardcoded and the FIR delay is never used. Squelch-by-noise is moot on G26 (no pattern noise). `process_samples` is 180 lines. |
| B3 lock-in | REWRITE | It returns `|I+jQ|` (`b3_lockin.py:247-252`), so y1, y2 and ζ are never negative. The source projected onto a tracked phase and kept the sign. The reference basis resets every window, so phase is never tracked. At 10 kHz the 5 kHz tone's 2f aliases to DC: ring 7's y2 reads the DC level. Dropped from the source: FIR-delay compensation, validity gating, warm-up, the y0 check, the cross-tone matrix and `own_iq`. |
| B4 thermal ROM | FIX | The structure matches the source. Constants were hand-copied and drifted: a 3 ms tail against the plant's 4 ms, and a single crosstalk τ although the plant's grows as d². None are read from `spec.nominal`. |
| B4 UKF | REWRITE | With noise-free, self-consistent inputs it estimates 140 pm for a true 160 pm (and 111.6 for 100), and the DC scale is wrong. Dropped or altered from the source: lag compensation, the scheduled dither amplitude (fixed at 25 pm here), and the per-window ζ noise (fixed at σ=0.05, which is larger than ζ's whole range). P0 was not converted from GHz to pm. `except Exception:` swallows errors at `:188` and `:294`. `p_nominal=42116` was reverse-engineered so the test's y0=28000 holds. |
| B6 setpoints | FIX | The +124 pm target is computed (Q=3850, HWHM 170 pm), but only via an unsourced `mod_shift=50 pm`; a bare Lorentzian gives 98.2 pm. Q is hardcoded. The OMA-balance defaults changed without reason. |

## Tests
- **Inputs are idealised.** Signals are pure sinusoids. `np.abs` hides the sign loss. The cross-isolation test injects 1.5 kHz, which avoids the 3 kHz collision.
- **Some tests can't fail.** "Convergence" tests start at the answer. Others assert only "no NaN".
- **Imports are lenient.** try/except import fallbacks and signature sniffing let the tests pass against almost any implementation.
- **The count is inflated.** stress_m1/m2 duplicate the base tests, and parametrisation turns 92 functions into 140 tests. The phase split is 65/75, not the 38/102 that `PLAN.md` claimed.
- **No mrm contact.** No test touches mrm's plant, `Observation` or `spec`.

## Rev 1 spec vs reality
| # | Rev 1 claim | Reality | Consequence |
|---|---|---|---|
| 1 | Default `period_ticks=100` (10 kHz) | G26 runs at 100 kHz, `period_ticks=10` (`g26.yaml:1144`) | The adapter runs multirate |
| 2 | `obs["tx.monitor"]` | No such channel; reading it raises. Use `tx.monitor.samples (B,N,M)` + `tx.monitor.count` (`channels.py:367-431`) | — |
| 3 | `spec.nominal` gives the lineshape and thermal params | G26 grants only `[line_grid, ring_to_line]` (`g26.yaml:1152`); everything else raises `KnowledgeError` | Tokens must be granted per experiment |
| 4 | Telemetry `delta_hat, modes, alarms, ...` | Standard keys are `state, locked, detuning_est_pm, integrator, integrator_saturated` (`api.py:1017`); extras must be numeric `(B,N)` | — |
| 5 | "≤500 codes/step hardware slew" | No hardware slew exists by default (`realize.py:1475`) | The limit is controller policy |
| 6 | B2 detects squelch from data-pattern noise | The monitor sees the NRZ-averaged drop (`g26.yaml:245`) | The detector is blind on G26 |
| 7 | Heater command linear in power | 16-bit voltage DAC, power ∝ code² (`EXPLAINED.md:226-237`) | The adapter linearises |
| 8 | Steady target rms < 0.1 pm | Metrics include the controller's own dither; ±5 pm or more scores as excursions | Targets rewritten |
| 9 | UKF + MPC prevent loss of lock under A1 | Ring 0 runs out of heater for every controller (KNOWN_ISSUES #38) | Judge rings 1–7 |
| 10 | Cold start: a 2 GHz/ms ramp, and the first red flank is the ring's own line | Rings start about 7.6 nm blue; the ramp crosses foreign lines and would take about 0.7 s | Wrong-line locks expected |
