# ECE496 TLC on the G26 model: results

The ECE496 thermal locking controller (TLC) run closed loop on mrm-model's 8-ring G26 transmitter, as published
(**faithful**, `tlc`) and with disclosed G26 adaptations (**tuned**, `tlc_arm2` / `tlc_wdm`), next to mrm's own
controllers. Every number below comes from `docs/tlc/results.md` / `.json` (`tools/tlc_run_all.sh`, ~45 min, 151
runs; 2026-11-26). Spec and decisions: `docs/TLC_PORT_SPEC_AND_PLAN.md`; how the code fits: `docs/tlc/WALKTHROUGH.md`;
interactive page: `docs/tlc/tlc_g26_evaluation.html`.

Errors are the true detuning minus mrm's lock reference, in pm (1 GHz ≈ 5.72 pm), world 0, rings 1-7 (ring 0 runs
out of heater under A1 for every controller). Bars: mrm 5 pm, ECE496 2 GHz ≈ 11.4 pm. One world per cell except the
cold-start campaigns (8 worlds).

## 1. Bottom line
- **The port is faithful.** The vendored core (ECE496 `ae9556b`) is bit-identical to the source on the source plant,
  with sine and triangle dither, and reproduces the published R1/R3/R4 numbers.
- **Faithful TLC on G26:** holds slow disturbances at 1.8-2.6 pm rms (6-8 pm peak: inside 11.4 pm, not 5 pm); loses
  lock under the A1 aggressor and the +10 K kick, drifts 34 pm in squelch, and its cold start ends on a foreign line in
  43 of 64 ring trials.
- **Tuned TLC on G26:** four disclosed changes (section 4) bring every slow scenario under 5 pm on every ring
  (0.86-3.0 pm peak, ~0.55 pm rms), make A1 and the kick recoverable, and cut cold-start wrong lines to 1 / 64.
- **Against mrm's controllers:** the tuned TLC beats the comparable dither controller (`dither_lockin_c`) everywhere
  (18-28x in steady state, drift, A1 and the common aggressor; 1.6-2x in squelch and the kick), matches mrm's PI / Kalman-LQG / MPC under A1 and squelch, and stays ~20x behind them in quiet steady state:
  a dithering controller carries its own dither (~0.9 pm peak) as error, while they hold a level calibrated once.
- **Lost by every controller:** the +3 dB power step and the stronger PD-side A1 (R11).

## 2. Results, nominal world (peak / rms, pm, rings 1-7)

| Req | Scenario | TLC faithful | TLC tuned | mrm PI | mrm dither lock-in | mrm Kalman-LQG | mrm MPC |
|---|---|---|---|---|---|---|---|
| R1 | Steady state | 6.3 / 1.8 | **0.86 / 0.53** | 0.050 / 0.013 | 20 / 8.7 | 0.034 / 0.009 | 0.053 / 0.014 |
| R2 | Die ramp, 120 K/s | 8.0 / 2.4 | **1.1 / 0.55** | 0.10 / 0.039 | 20 / 8.8 | 0.039 / 0.010 | 0.051 / 0.015 |
| R5 | Laser drift | 7.5 / 2.6 | **1.1 / 0.53** | 0.096 / 0.035 | 20 / 8.6 | 0.093 / 0.038 | 0.10 / 0.040 |
| R3/R4 | Common aggressor (ext.) | 8.0 / 2.4 | **3.0 / 0.57** | 1.5 / 0.17 | 53 / 10 | 0.56 / 0.071 | 0.39 / 0.047 |
| R3/R6 | Local aggressor A1 | lost (2 wrong lines) | **18.5 / 0.84** | 14.6 / 0.63 | 524 / 18 | 15.3 / 0.58 | 15.3 / 0.57 |
| R12 | Squelch | 34 / 15.5 | **20 / 2.8** | 18.4 / 0.62 | 40 / 10 | 18.3 / 0.88 | 18.3 / 0.88 |
| R10 | +10 K kick (ext.) | 179 / 44, no relock | **179 / 10, relocks** | 2,240 / 33 | 292 / 36 | 153 / 2.4 | 157 / 2.5 |
| R7 | +3 dB power step | lost | lost | lost | 869 / 57 | lost | lost |
| R11 | A1 PD side, 1.5x | lost | lost | lost | 806 / 20 | lost | lost |
| R21 | PD dropout (ext., TLC only) | 5.9 / 2.1 | — | | | | |
| R22 | Forced PI fallback (TLC only) | 26 / 9.9 | — | | | | |

Cold start (R8/R9, 8 sampled worlds, wrong-line ring trials, 95 % Clopper-Pearson): faithful TLC **43 / 64**
[54 %, 78 %], none on its own line; WDM supervisor alone 1 / 64 [0.04 %, 8.4 %]; tuned (`tlc_wdm`: the supervisor
acquires, the tuned TLC tracks) **1 / 64**. The WDM supervisor's 300-world campaign: 12 / 2400 [0.26 %, 0.87 %].

## 3. Why the faithful TLC struggles on G26 (diagnosis)

| # | Cause | Evidence | Tuned arm |
|---|---|---|---|
| 1 | **The dither always runs at its 0.5 GHz maximum.** B1 sizes it from the estimator's uncertainty, which never settles on G26 (see 2). The dither alone is ~1.7 pm rms of scored error on every ring, and its size also drove a slow estimate drift on rings 0-2 | steady: dither part 1.6-1.8 pm rms; tracking part 0.05-0.4 pm on rings 4-7, up to 3.7 pm on ring 0 | dither capped at 0.15 GHz (~0.9 pm) |
| 2 | **Detuning and optical power look alike to the estimator.** G26's line is 1.6x wider than the source plant's, and the lock point sits at 0.6 half-widths, where the power-independent ratio ζ = z2/z1 moves ~1.4e-4 per GHz against ~6e-4 noise (open-loop sweep: `docs/tlc/lockin_diag.md`). A level change can be read either way; in a fast transient the filter reads it as power | +10 K kick: the power scale p jumped 18 % in 0.5 ms, the estimate stayed on target, the ring parked 40 pm off, no loss of lock declared | p held stiff (0.5 % std) once a ring has tracked 30 ms; level changes go to detuning |
| 3 | **Re-acquisition assumes the ring fell blue.** Under A1, ring 0 is lost to the red with its heater already at zero; the source backs off and ramps up, pushing it over the self-heating fold onto the next line, and its light then knocks ring 4 off | probe: ring 0 at u = 0, +335 pm, ramp, +1018 pm within 1.5 ms | a ring lost red at the heater floor holds and re-locks when its level returns to the calibrated operating level |
| 4 | **Cold-start ACQUIRE takes the first fringe.** Every cold G26 ring sits next to a foreign line | 43 / 64 wrong lines | mrm's WDM supervisor acquires (line identity), then hands each world to the TLC |
| 5 | **Squelch detection is blind.** The source detects squelch from data-pattern noise; the G26 monitor is NRZ-averaged | faithful 34 pm | not fixed; the stiff scale alone brings it to 20 pm, level with mrm's controllers (the line narrows when modulation stops) |
| 6 | **A -3 dB power drop knocks rings off their hot branch.** It removes most of the self-heating that holds a G26 ring, and the resonance jumps ~0.9 nm blue within 0.1 ms | every model-based controller loses lock | open: a blue fall-off detector (`arm2.fall_level`, off) halves the wrong lines but no ring re-locks in the run |

Tried without gain: crosstalk priors to every ring (3 % then 1.5 % flat) instead of the nearest three; a rate limit
on p instead of the stiff scale (A1 still lost).

## 4. The tuned arm (second arm, spec 6 Q1)
Every change is a priori (from the nominal design and the diagnosis above, not fitted to a scenario score), named,
and switchable; the vendored core is untouched.

| Change | Where | Setting |
|---|---|---|
| Dither cap | `ControllerParams` via `cp_overrides` | `dither_max = dither_nominal = 0.15 GHz`, `dither_min = 0.05` |
| Stiff DC scale | `mrm/control/tlc/arm2.py` | `stiff_scale_rel = 0.005`, `stiff_settle_s = 0.03` (not at the heater floor) |
| Red-side waiting re-lock | `arm2.py` | `red_wait_s = 0.05` |
| WDM-supervisor acquisition | `mrm/control/tlc/hybrid.py` (`tlc_wdm`) | hand-off per world at the supervisor's TRACK phase; fresh core, warm start |

`tlc_arm2` = the first three as defaults; `tlc_wdm` adds the fourth for cold starts and needs the supervisor's
knowledge tokens (heater, optics, monitor nominal) on top of the TLC's.

## 5. Model trust (corners; worst peak on rings 1-7, pm)

| | | fast thermal | slow thermal | textbook thermal | TEC-held package |
|---|---|---|---|---|---|
| Steady | TLC faithful | 6.5 | 6.1 | 15.9 | 6.8 |
| | TLC tuned | **0.85** | **0.84** | **1.6** | **0.87** |
| | TLC tuned, nominal priors (mismatch) | 0.92 | 2.2 | 2.1 | 0.84 |
| | mrm PI / Kalman-LQG / MPC | 0.04-0.05 | 0.04-0.05 | 0.03-0.04 | 0.05 |
| A1 | TLC faithful | lost | lost | lost | 8.2 |
| | TLC tuned | **15** | lost | lost | **3.5** |
| | mrm PI | 11.8 | lost | lost | 3.2 |
| | mrm Kalman-LQG / MPC | 12.6 | 33 | lost | 3.6 |

The tuned TLC passes 5 pm in steady state on every corner, also with the nominal priors on the corner plant, and
its A1 verdict matches mrm's PI on every corner. Every verdict is the same with corner and with nominal priors.

## 6. Dither waveform (ECE496 `ae9556b`)
The team's sampled waveforms (square, three-level, triangle) need tones at 100 kS/s ÷ even N; no eight distinct ones
avoid each other, so the source's four (3.8-7.1 kHz) repeat every four rings (crosstalk four rings apart ≈ 1e-4).
On the faithful arm the triangle, best on the 4-ring plant, is 1-5 pm worse than sine on G26 and loses A1 harder;
moving to the sampled tones alone costs little (`docs/tlc/dither_compare.md`). Sine stays the default.

## 7. Open
- Power step: blue fall-off detection plus a faster re-acquisition (the source's stuck-sensor check holds the
  fallen rings).
- Squelch: give the TLC the squelch state, or freeze on a collapsed dither response.
- Statistics: 300-world cold-start campaigns for `tlc_wdm`; more worlds per scenario.
- Steady state below the dither floor would need a smaller dither and a better detuning observable than ζ.
