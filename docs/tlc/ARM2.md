# TLC second arm on G26 (disclosed adaptations)

The faithful arm (`core: faithful`, no overrides) stays bit-identical to ECE496 `ae9556b` and is reported as is. This
second arm (spec 6, Q1) changes the TLC for G26 a priori, every change named and switchable. Code:
`mrm/control/tlc/arm2.py` (`Arm2TLC`), `mrm/control/tlc/hybrid.py` (`tlc_wdm`), the adapter params `cp_overrides`,
`arm2`, `pp_overrides`. Comparison arms: `tools/tlc_dither_compare.py` (`arm2_d015`, `arm2_stiff`, `arm2_full`).

## Why the faithful TLC does badly on G26 (diagnosis, 2026-11-26)

| Cause | Evidence | Fix in this arm |
|---|---|---|
| The dither runs at its 0.5 GHz maximum all the time: B1 sizes it from the UKF's uncertainty, which never settles on G26. The dither alone is ~1.7 pm rms of "error" on every ring, and its size also drove a slow drift on rings 0-2 | steady: dither part 1.6-1.8 pm rms; tracking part 0.05-0.4 pm on rings 4-7, up to 3.7 pm on ring 0 | `cp_overrides`: `dither_max = dither_nominal = 0.15 GHz` (~0.9 pm) |
| Detuning and the DC scale p (laser power x monitor gain) are nearly indistinguishable at the G26 lock point (0.6 half-widths): zeta changes ~1.4e-4 per GHz against ~6e-4 noise. In a fast transient the filter puts the level change into p (+18 % after the +10 K kick) and parks the ring ~40 pm off | kick probe: p 0.96 -> 1.13 in 0.5 ms, estimate on target, true error -40 pm, no loss of lock declared | `arm2.stiff_scale_rel = 0.005`: p decorrelated and held to 0.5 % std, so level changes go to detuning. Not applied to a ring at the heater floor or within `stiff_settle_s = 30 ms` of a (re-)lock |
| The source re-acquisition assumes a ring lost to the blue (back off, ramp up). Under A1, ring 0 is lost to the RED with its heater at zero; ramping up pushes it over the self-heating fold onto the next line, and its light then knocks ring 4 off | A1 probe: ring 0 at u = 0, +335 pm, re-acquire ramp, +1018 pm in 1.5 ms | `arm2.red_wait_s = 0.05`: hold the heater and re-lock when the monitor level returns to the level the calibrated p predicts at the operating point |
| The TLC's ACQUIRE takes the first fringe; on the G26 grid that is usually a foreign line | faithful cold campaign: 43 / 64 wrong-line ring trials | `tlc_wdm`: mrm's WDM supervisor acquires, the TLC takes over each world at its TRACK phase (fresh core, warm start) |

## Results (G26 nominal world; worst peak on rings 1-7, pm; one world per cell)

| Scenario | TLC faithful | TLC arm 2 (rms) | mrm PI | mrm dither lock-in | mrm Kalman-LQG | mrm MPC |
|---|---|---|---|---|---|---|
| Steady state (R1) | 6.3 | **0.86** (0.53) | 0.050 | 20 | 0.034 | 0.053 |
| Die-temperature ramp (R2) | 8.0 | **1.1** (0.56) | 0.10 | 20 | 0.039 | 0.051 |
| Laser drift (R5) | 7.5 | **1.1** (0.54) | 0.096 | 20 | 0.093 | 0.10 |
| Common aggressor (R3/R4) | 8.0 | **3.0** (0.58) | 1.5 | 53 | 0.56 | 0.39 |
| Local aggressor A1 (R3/R6) | 1,090, 2 wrong lines | **18.5** (8.9), none | 14.6 | 524 | 15.3 | 15.3 |
| Squelch (R12) | 33.7 | **20.1** (2.8) | 18.4 | 40 | 18.3 | 18.3 |
| +10 K kick (R10) | 179, never relocks (rms 41) | 179, **relocks** (rms 10) | 2,240 (rms 29) | 292 | 153 (rms 2.4) | 157 |
| PD dropout (R21) | 5.9 | **2.4** | — | — | — | — |
| Cold start, 8 worlds (R9), wrong-line ring trials | 43 / 64 | **1 / 64** (`tlc_wdm`) | — | — | — | WDM supervisor alone: 1 / 64 |
| +3 dB power step (R7) | lost | lost | lost | lost | lost | lost |
| A1 PD side x1.5 (R11) | lost | lost | lost | lost | lost | lost |

Reading it:

- **Steady state, die ramp, laser drift, common aggressor and PD dropout now pass the mrm 5 pm budget on every ring
  1-7** (faithful: none did). A1 (18.5 pm, ring 4, light stolen as ring 0 sweeps past its line) and squelch (20 pm, the
  line narrows when modulation stops) do not, for mrm's controllers either.
- **Against the basic dither controller (mrm dither lock-in) the TLC wins everywhere, by 10-30x.**
- **Against mrm PI it ties under A1 and squelch, wins on the +10 K kick, and loses in quiet steady state** (0.9 vs
  0.05 pm): a dithering controller carries its dither (~0.9 pm peak here) as error, while PI holds the monitor at a
  fraction of a peak it measured once, which no dither disturbs and which a laser power change would move.
- **Cold start** matches the WDM supervisor, because the supervisor now does the acquisition.
- **Power step and A1 PD-side x1.5 are lost by every controller.** For the power step the mechanism is physical:
  -3 dB at the ring removes most of the self-heating that holds a G26 ring on its hot branch, and the resonance
  jumps ~0.9 nm blue within 0.1 ms. `arm2.fall_level` (blue fall-off detection, then immediate re-acquisition)
  reduces wrong-line finishes from 6 to 1-2 but no ring re-locks inside the run; the TLC's stuck-sensor check
  (B2) then holds the rings. Left off in `arm2_full`; next step if it matters.

## Tried without gain

- Crosstalk to every ring (`pp_overrides: coupling` = 3 % then 1.5 % flat) instead of the nearest three: no change
  in steady, A1, common aggressor or kick. After the fixes above the model mismatch is no longer the limit; the
  dither is.
- A rate limit on p alone (`scale_rate_max`) instead of `stiff_scale_rel`: A1 still lost.

## Reproduce

```
.venv/bin/python tools/tlc_dither_compare.py --arms arm2_full --experiments <...> --runs-dir runs/arm2 --out docs/tlc/arm2_full.md
.venv/bin/python tools/tlc_run.py experiments/tlc/cold_campaign_hybrid.yaml --set 'controller.params.tlc={core: faithful, cp_overrides: {dither_max: 0.15, dither_nominal: 0.15, dither_min: 0.05}, arm2: {stiff_scale_rel: 0.005, stiff_settle_s: 0.03, red_wait_s: 0.05}}'
```
