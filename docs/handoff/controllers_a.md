# Handoff: controllers A, feedback locks (`controllers_a`)

Owner files: `mrm/control/baselines/{pi_lock,dither_lockin,bias_dither,extremum_seeking,feedforward,midranging}.py`,
`tests/verification/test_ctrl_a_{harness,pi,dither,bias_dither,es,feedforward,midrange,examples,scenarios,engine,corners,nojit}.py`,
`examples/controllers/{README.md,template_python_controller.py,template_per_ring.py,template_compiled_controller.py}`.
Contract: ARCHITECTURE section 6 items 1, 2, 3, 5, 7, 9; 07 P0-1, P0-7, P0-9, P1-1, P1-19, P1-20; 04 section 4
(C1, C2, C3, C5, C11, C13).

## 1. What is built

| registry name | class | form | actuators |
|---|---|---|---|
| `pi_lock` / `pi_lock_c` | `PiLock` / `PiLockCompiled` | drop-port PI at a fraction of the measured peak, red side; MAP (adaptive, stops just past the peak), RETURN / RECAPTURE with a code guard, fall-off detection, driver-law gain schedule, optional type 2 (`k2`, anti-windup at the rails), ADC-clip handling (DECLIP when clipped from the hand-off on), capture start (bistable or not), `start_delay_s` | heater |
| `dither_lockin` / `dither_lockin_c` | `DitherLockin` / `DitherLockinCompiled` | three-level heater dither, drift-free 1f/2f demodulation, `mode='peak'` (1f = 0) or `'inflection'` (2f = 0, power and relief independent) | heater |
| `bias_dither` | `BiasDither` (CompiledController with `fast_kernel`) | bias dither demodulated per ADC sample in the fast kernel (07 P0-7); step: hand-off HOLD (r_set), heater identification, heater integrator on the 1f/DC ratio (or 2f) | heater + bias |
| `es_po`, `es_sine` | `ExtremumSeekingPO`, `ExtremumSeekingSine` | perturb-and-observe (argmax or Newton), sinusoidal ES (normalized demodulation); peak of the drop monitor; fall-off recovery | heater |
| `feedforward_pi` | `FeedForwardPI(PiLock)` | PI + die temperature-sensor feed-forward (first-order hold between sensor deliveries by default) | heater |
| `midranging` | `MidRanging(PiLock)` | PI hand-off, bias identification, inner bias Newton loop, outer heater re-centring; the PI re-settles after a recapture | heater + bias |

Shared helpers (public, in `pi_lock.py`): `KNOWLEDGE_KEYS`, `NominalModel.from_spec(spec)`, `monitor_block_mean`,
`block_inputs`, `law_code`, `sigma_schedule` (njit), `per_ring`, `design_pi_gains`, `make_params`. PI states:
`ST_HOLD 0, ST_MAP 1, ST_RETURN 2, ST_LOCK 3, ST_RECAPTURE 4, ST_FAULT 5, ST_CAPTURE 6, ST_DECLIP 7` (review).

Exact constructor signatures: every Python controller takes `(params: Mapping | Params | None = None, **overrides)`
(BaseController rule: unknown keys raise TypeError); the compiled ones the same (`PiLockCompiled(params=None,
**overrides)` etc.) and `from_params(params)`. Compiled functions: `pi_init(spec_i64, spec_f64)`, `pi_step(p, pf, s,
sf, obs, act)`, `dither_init`, `dither_step`, `bias_init`, `bias_step`, `bias_fast(p, pf, s, sf, fobs, fact)`;
`bias_dither.demod_reference(codes, samples_per_phase, blank, a, start_phase=0) -> [(S1, S2, y0, r)]`.

## 2. Requests

1. **Integrator: register the modules.** `mrm/control/baselines/__init__.py` is empty and not mine. The classes
   register with `@register` / `register(...)` when their module is imported, but `registry.resolve` only imports the
   package. Please add `from . import pi_lock, dither_lockin, bias_dither, extremum_seeking, feedforward,
   midranging  # noqa: F401` (until then `registry.create("pi_lock")` works only after importing the module).
2. **params / content: knowledge tokens.** `controller_interface.knowledge` is a free list and nothing filters
   `spec.nominal` yet (the ENGINE_BRIEF passes the full cfg as `nominal`). `NominalModel` reads `spec.nominal` only
   for the tokens in `spec.knowledge` and defines the token -> key table `pi_lock.KNOWLEDGE_KEYS` (07 P1-1 names
   `heater_r_nominal`, `dlambda_dt_nominal`; proposed: `heater_law`, `heater_efficiency`, `linewidth`,
   `thermal_dynamics`, `bias_efficiency`). Please adopt the token list in the schema doc (or tell me the names), and
   build the knowledge-filtered nominal view (07 D13) so a controller cannot read other keys by accident
   (**done 2026-10-16**, controllers review: `mrm.control.knowledge.NominalView`; the table now lives there as
   `FINE_TOKENS`, re-exported as `pi_lock.KNOWLEDGE_KEYS`; one vocabulary is still open). g26 lists
   only `line_grid, ring_to_line`: every baseline here works without model knowledge (they identify what they need);
   `feedforward_pi` needs `dlambda_dt_nominal + heater_law + heater_efficiency` or a calibrated `ff_codes_per_k`.
3. **engine (runner / plant).**
   - `fast_kernel` semantics I assumed (core_control_api handoff): called once per released monitor sample, before
     `step` on the same tick; the bias/heater offsets are held until rewritten and added to the latched command
     before the DAC FIFO; `tx.<ch>.applied` shows the latched register WITHOUT the offset. My harness implements
     exactly this (`test_ctrl_a_harness.ClosedLoop._fast_release`); please confirm the kernel does the same.
   - Warm start: `run(..., heater_code=..., reset_protocol='branch_high')` starts the thermal state cold (lags at
     zero). My harness settles the lags (`ThermalSystem.steady_state`) before the hand-off; with the engine the
     controllers need `start_delay_s` (PI family) for the slow poles (3 ms tail, ms crosstalk). An option to start
     from the thermal steady state of the warm-start codes would make warm starts honest.
   - First compile of `runner.run` took ~6-7 min per path on this loaded machine (load ~20-26); the engine tests
     here are `slow`.
4. **thermal / scenarios / integrator: the A1 aggressor heats the monitor PD 6x more than the ring.** In the g26
   interim layout a steady 100 mW at the `aggressor` site raises the victim ring 34 K but its monitor PD observer
   205 K (`ring.thermal.pd_offset_um` [0, 25] puts the PD nearer the aggressor). Over the full slide-10 trajectory
   (25 degC base) the PD dark current (Arrhenius 0.37 eV, 5 nA at 25 degC) reaches microamps (thousands of codes):
   the fraction-of-peak set-point then sits at a larger detuning, the lock drifts red from ~25 ms and is lost near
   the cusp (~52 ms); the heater does NOT saturate (lowest code ~22 k). After the loss the PI integrates the dark
   offset upward to full scale (04 C3 one-sided runaway), and the neighbours then fall too (20 falls; probably the
   array supply limit, 0.2 W, once ring 0 sits at full scale: not verified) (`test_a1_full_on_the_25c_base_monitor_dark_current`). Please check whether the PD-to-
   aggressor coupling is intended; a baseline/dark subtraction in the monitor (PAT-2 baseline subtraction,
   `ring.monitor.baseline_ua`) or a controller-side dark estimate would be the fix. A 0 degC base attempt did not
   lock in the harness (heater codes 50-65 k while placing; not diagnosed), so only the 25 degC full run is kept.
   **Resolved 2026-10-10** (docs/handoff/integrator.md section 7): the coupling was not intended. The g26 default
   site moved to (0, -30 um), 55 um from PD0 (every ring distance unchanged); the 5 um layout is now the stress
   scenario `a1_slide10_aggressor_pd_adjacent`, and `test_a1_full_on_the_25c_base_monitor_dark_current` runs it
   (`a1_rate_scenario(..., site=lib.AGGRESSOR_PD_ADJACENT_SITE)`). The a1_rate / a2_rate tests run the default site.
5. **plant / content (bias physics, 07 P1-7).** In g26 at 1.05-1.8 V a 0.2 V bias step moves the resonance only
   0.1-0.5 pm (10 us after the step 0.3-0.5 pm, settling to 0.1-0.3 pm): the FCA absorption change heats/cools the
   ring by about as much as the electro-optic shift, opposite in sign. The loss change modulates the drop monitor
   more than the detuning does (+0.33 % per +0.2 V, measured dy/db > 0 on the stable side). If that is not intended,
   check the sign/size of `alpha_fca(V)` against `eff_at_0v`. It makes bias-based locks (bias_dither, midranging)
   weak in g26; 07 P0-7's "a bias dither can meet the aggressor" does not hold in this plant.
6. **content: monitor full scale.** With `monitor.gain_scale` up to 2 the drop peak exceeds the 40 uA ADC full scale
   (sampled world 0 of seed 3); `pi_lock` then takes the full scale as M_pk. The PGA has no gain below 1.

## 3. Algorithms in one paragraph each (details in the module docstrings)

- **pi_lock.** MAP lowers the heater along the stable branch (8 codes/step on the flank, 1 code/step once the
  windowed slope falls below half the flank slope) and stops when the monitor falls 2e-4 below its maximum: just
  past the peak, before the fold (~9 pm further in g26). M_pk = that maximum; s = dy/dcode is a least-squares fit
  around y = f M_pk (self-heating relief included, ~-10 codes/code in g26). PI in velocity form, zero at the 13 us
  pole (nominal tau_slow with `thermal_dynamics`), k_c = 2 pi f_c T_i, f_c = 2.5 kHz; the type-2 velocity v is reset
  while it pushes into a heater rail (anti-windup). ADC clip: M_pk = full scale; clipped from the hand-off on (no
  unclipped record) -> DECLIP: ramp red by `ramp_step_codes`, recording, until y < f x full scale, slope from the last
  2K+1 unclipped points, LOCK there. Capture start: ramp up until a resonance was seen (max > 2 min + 500 codes) and
  the monitor fell back 2 % from its max (red of the peak after a jump or a smooth passage), then MAP; FAULT back at
  the start code at full scale. Fall-off: drop > 0.35 M_pk to
  below 0.4 M_pk within 3 steps -> RECAPTURE: ramp up, ignore captures below u_fall + guard (half the MAP's code
  travel: rejects line n-4 captured by the next order, which happened to rings 4-7 in g26 when an earlier MAP fell
  off), done when the monitor peaked and fell back 2 % (capture jump or smooth passage); gives up (FAULT, heater
  back to the fall code) after ``recapture_span_codes`` = 8000 codes above the guard, never parking at full scale.
- **dither_lockin.** Phases +A, 0, -A, 0 (6 steps each, 3 blanked): S1 = (y0 - y1 - y2 + y3)/2A, S2 = (y'2 - 2 y'3 +
  2 y0 - 2 y1 + y2)/2A^2 over the previous cycle's last phases (drift-free, and the settling residue of the two
  centre phases cancels; the simpler (y0 + y2 - 2 y1)/A^2 moved the lock by tens of pm); the stored y' are shifted by
  S1 du when the centre moves. Inflection: eps = S2 y0/S1^2 = eps_L/R, W = 0.866 y0/|S1|, du = -gain eps
  W/5.2 (loop gain gain/R per cycle; R = relief). Peak: damped Newton -gain S1/S2.
- **bias_dither.** Same three-level pattern on the bias in the fast kernel, 4 samples per phase (1 blanked); per
  cycle S1, S2, y0, r = S1 A/y0. Step: HOLD 16 cycles (r_set), heater ID +-64 codes (g_r, g_y), LOCK: du = -gain (r -
  r_set)/g_r, the cycle after a heater move skipped (its step response reads as dither). Falls are detected in HOLD,
  ID and LOCK; after the recapture an unfinished HOLD/ID restarts, keeping the hand-off r_set once measured. Ratio
  mode needs a hand-off inside the half width (|r| peaks at x = 1 and r(x) = r(1/x)).
- **extremum seeking.** P&O: dwell at u, u+d, u-d (settle 4, average 4 steps), argmax or Newton. Sine: f_d 2 kHz,
  HPF 200 Hz, phase 0.35 rad, u_hat += gain xi/J_lp.
- **feedforward_pi.** u_ff = c - u, c^2 = u^2 - 2 u K_ff(u) (T_hat - T_ref), K_ff = (dlambda/dT)/(pm per code at u)
  (the secant of the quadratic law from the integrator u, never evaluated at u + u_ff; -K_ff (T_hat - T_ref) for a
  linear law; relief-independent because the relief divides disturbance and correction alike), T_hat a first-order
  hold of the sensor (the staircase is worse than no feed-forward). T_ref is taken per ring at each lock; a ring that
  leaves LOCK takes u_ff into its integrator (bumpless), and the integrator is held where u + u_ff is at a DAC rail.
  Until 2026-10-16 K_ff was evaluated at u + u_ff and ran away under A1 (docs/CHANGELOG.md).
- **midranging.** After the PI settles: bias ID +-300 codes (s_b), then b -= k_b (y - y_set)/s_b (clipped to
  b_mid +- 0.6 V), u += k_o (s_b/s_h)(b - b_mid). After a fall and recapture the heater PI settles at the set-point
  again before the bias loop resumes; telemetry `locked` follows the bias loop in RUN.

## 4. Results (Truth inside the tests; g26 nominal, B = 1, 8 rings unless stated; CTRL_A_METRIC lines)

Hand-off: rings placed by the test at +170 pm on the stable branch (Truth-guided ramp, thermal lags settled).
Victim = ring 0 for the aggressors (A1 rate: 2127 degC/s peak with a 20 K swing; A2 rate: 1616 degC/s edges, 20 K).
Lock error = detuning minus the controller's own settled detuning; excursions = entries into |error| > 5 pm
(1 pm hysteresis); OMA penalty = 10 log10(OMA_ref/OMA) against the pre-disturbance lock; relOMA = OMA/P_in.

| controller, scenario | lock detuning | rms / max lock error (pm) | excursions | OMA penalty mean / max (dB) | time to lock | notes |
|---|---|---|---|---|---|---|
| pi_lock, steady | 128 pm (f = 0.7) | 0.016 / 0.045 | 0 | 0.00 / 0.00 | 3.9 ms incl. MAP | peak ~58.8 k codes, slope -10.0 codes/code, relOMA -6.57 dB; MAP passes the peak by 2-5 pm, never the fold |
| pi_lock, laser -1 dB / +1 dB | shift -57 / +48 pm | (the shift) | 1 (permanent) | 1.44 / 1.45 ; -0.10 | | 04 C3 failure mode, measured |
| pi_lock, capture start (N = L = 1) | 121 pm | 0.015 / 0.04 | 0 | 0 | 5.1 ms | |
| pi_lock, 2 sampled worlds (13 placeable rings) | spread 110 pm | 0.033 / 0.10 | 0 | 0.00 | <= 9.5 ms | 3 rings unplaceable (cold resonance red of the line) |
| pi_lock, A1 rate | | 2.8 / 8.4 | 1 (2.1 ms) | 0.02 / 0.10 | | error peaks at the cusp |
| pi_lock type 2 (k2 = 0.005), A1 rate | | 2.0 / 6.8 | 1 (0.9 ms) | 0.01 / 0.08 | | Python and compiled identical |
| pi_lock, A2 rate | | 2.8 / 7.3 | 1 (2.0 ms) | 0.02 / 0.08 | | |
| pi_lock type 2, A2 rate | | 2.2 / 6.1 | 1 (0.8 ms) | 0.02 / 0.07 | | |
| pi_lock type 2, full A1, 25 degC base | | 1399 / 5251 | 2 | 8.8 / 41 | | holds < 10 pm for 20 ms; error > 50 pm from 41.6 ms (monitor-PD dark current, request 4); 20 neighbour falls after the loss |
| feedforward_pi (type 2), A1 / A2 rate | | 2.0 / 6.8 ; 2.2 / 6.1 | 1 ; 1 | 0.01 / 0.08 | | = PI type 2: the die sensor does not see the local aggressor |
| pi_lock vs feedforward_pi, die ramp 1000 K/s x 6 K | | 0.68 / 0.89 ; 0.53 / 0.83 | 0 | 0.01 | | +-10-20 % depending on settling; staircase FF: max 8.0 pm |
| dither_lockin inflection, steady | 131 pm | 9.2 / 15.5 with the dither; cycle-mean drift 1.4 | 13 (dither) | 0.02 / 0.21 | 4.6 ms from 170 pm | dither swing +-13.6 pm; same point from a 128 pm hand-off |
| dither_lockin inflection, laser -1 dB | shift -9 pm | 26 / 79 | 55 (dither + transient) | 1.0 / 2.3 | | vs -57 pm for the fraction-of-peak PI |
| dither_lockin inflection, A1 / A2 rate | | 52 / 113 ; 50 / 89 | 31 ; 22 | 0.76 / 2.6 ; 0.73 / 1.9 | | cannot follow (one update per 240 us cycle) |
| dither_lockin peak, steady | near 0 | | | | | 8 falls in 6 ms (fold 9 pm blue of the peak) |
| bias_dither ratio (1.8 V), steady | 129 pm | 1.7 / 6.6 with the dither | 9-11 (dither) | 0.00 / 0.38 | ~2.6 ms HOLD + ID | relOMA -7.25 dB (vs -6.57 at 1.05 V) |
| bias_dither ratio, A1 rate | | 12.3 / 28.8 | 100 | 0.06 / 0.73 | | |
| midranging (1.6 V), steady | 125 pm | 0.055 / 0.15 | 0 | 0.01 / 0.03 | PI + 0.4 ms ID | s_b = +0.42 codes/bias code (loss-dominated), relOMA -7.08 dB |
| midranging, A1 rate | | 178 / 473 (builder); 4.2 / 9.7 after the review (saturation fallback) | 1 | 3.4 / 11.5; 0.24 / 0.55 | | lost before the fallback: the bias saturates at once (its authority is tiny in g26) |
| es_po / es_sine, -10 dBm per ring (no bistability) | -1.9 / +3.3 pm (peak) | 4.4 / 7.8 ; 5.5 / 10.5 (probing) | 6 ; 4 | | | relOMA -19 dB: the peak is the wrong point for OMA |
| es_po / es_sine, +4 dBm (nominal) | | | | | | es_po 7 falls in 8 ms; es_sine still approaching after 8 ms |
| es_sine, A1 / A2 rate | | 546 / 1030 ; 712 / 1124 | | | | falls (peak seeking near the fold) |

Engine (`mrm/sim/runner.py`, `test_ctrl_a_engine.py`, slow): open loop from the same warm-start codes the engine
and the harness agree on the thermal state to ~0.01 K (ring 0: 390.48 K at 100 us, 393.67 K at 4 ms in both) and on
the heater power; the engine's `detuning_m` is ~1 % larger in magnitude than the harness Truth (lam_res + shift -
lam_line) at 1-2 nm detuning (-1136.7 vs -1124.9 pm): a conversion difference to settle (engine owner); 159 of 160
points agree within 2 % (the exception is a ring between branches during the warm-up; the open-loop test asserts
>= 98 % and was not re-run after that relaxation). Closed loop
(warm start, `branch_high`, `start_delay_s` = 30 ms, 40 ms): `PiLock` on the Python path and `PiLockCompiled` on the
compiled path give identical engine metrics and states; 7 of 8 rings lock at 126-130 pm; ring 3 ends the cold-lag
warm-up on the cold branch and faults (hand-off precondition not met). Engine wall time is dominated by the first
compile (~8-10 min per path here).

## 5. Deviations and doubts

- Hand-off and acquisition: these are tracking locks. `start='capture'` locks on the first line it meets; in a g26
  bank that is line n+3 (the 7.6 nm offset vs the 2.29 nm grid, integrator handoff section 4). Line identity is the
  supervisor's job (07 P1-12). Tests hand off with a Truth-guided placement (test code only).
- fraction-of-peak normalization: M_pk is measured once (MAP); a laser power step therefore moves the lock point
  (measured). No reference-tap channel exists to normalize by the input power (integrator handoff section 3).
- `bias_dither` uses a DC bias of 1.8 V (+-0.4 V dither) and `midranging` a window around 1.6 V, not the 1.05 V
  relOMA optimum: both need room above the 1.0 V floor. Their relOMA cost is reported.
- `bias_dither` is compiled only (fast-kernel form); `extremum_seeking`, `feedforward`, `midranging` are Python only
  (not in ARCHITECTURE's (c) list).
- The compiled ABI tiles one init state over B, so compiled per-ring tables (init codes, factory peaks/slopes) are
  shared by every world; the Python versions accept (B, N).
- The PI's gain schedule assumes a quadratic driver law unless the `heater_law` token says power/dac_pwm.
- Detuning estimates in telemetry (`detuning_est_pm`) need the `linewidth` token (PI family) and are NaN for the
  dither locks.

## 6. Tests

`-m "not slow"` (41 tests, ~4 min on this machine at load 20-26; each file below ~60 s there):
`test_ctrl_a_harness.py` (harness = ReferencePlant tick, serial noise = library noise, Truth placement, the 07 P0-6
loop-delay probe through the harness), `test_ctrl_a_pi.py` (map and lock, compiled replay bit-identical, toy parity
with falls and recaptures, laser power steps, capture start, knowledge gating, no Truth imports, registry),
`test_ctrl_a_dither.py` (inflection lock, compiled replay, toy parity peak/inflection, peak mode at the fold),
`test_ctrl_a_bias_dither.py` (07 P0-7 fast kernel = Python reference bit for bit, toy lock in both modes, g26),
`test_ctrl_a_es.py`, `test_ctrl_a_midrange.py`, `test_ctrl_a_feedforward.py` (knowledge gate),
`test_ctrl_a_examples.py` (the three templates).

`-m slow`: `test_ctrl_a_scenarios.py` (A1 and A2 at the printed rates for six controllers, the full A1 on 25 degC,
power steps, bias controllers on A1, two sampled worlds; ~10-15 min), `test_ctrl_a_feedforward.py` (the ramp
comparisons), `test_ctrl_a_engine.py` (~20 min, mostly engine compile), and since the review
`test_ctrl_a_corners.py` (every named corner, ~3.5 min) and `test_ctrl_a_nojit.py::test_toy_and_unit_tests_under_
disable_jit` (~40 s). Owner files now also include `tests/verification/test_ctrl_a_{corners,nojit}.py`.

The toy-ring tests also pass with `NUMBA_DISABLE_JIT=1`; the plant-based tests were not run that way (the
ReferencePlant then runs its kernels in Python).

## 7. Adversarial review (second pass)

Numbers in sections 4-6 were measured by the builder before the extracted mode/coupling artefacts landed; on the
current artefacts the nominal PI locks at 124.7 pm (peak 57.5 k codes, slope -9.87 codes/code, rms 0.015 pm, max
0.048 pm, 3.9 ms to lock), the rest of section 4 within a few per cent. The fixes below do not change the nominal
MAP + lock path (identical metrics before and after).

### 7.1 Bugs found and fixed (each with a test that failed before the fix)

| # | file | defect | fix | test |
|---|---|---|---|---|
| 1 | pi_lock.py (Python + compiled) | type-2 velocity v kept integrating while the heater sat at a rail (no anti-windup on the second integrator): the heater stayed at full scale ~80 steps (0.8 ms) after the disturbance reversed | v reset to 0 while it pushes into the clamp | `test_type2_velocity_does_not_wind_up_at_the_rail` |
| 2 | pi_lock.py | `start='capture'` waited for a jump within 3 steps; a ring without bistability rises smoothly, so the heater ramped to full scale and PARKED there in CAPTURE | capture ends when a resonance was seen and the monitor fell back 2 % from its max (after a jump or a smooth passage); gives up at full scale (FAULT, back to the start code) | `test_capture_start_with_and_without_bistability[0.5]`, `test_capture_without_a_resonance_faults_back_to_the_start_code` |
| 3 | pi_lock.py | monitor clipped at the ADC full scale from the hand-off on (no unclipped MAP record): slope NaN -> FAULT. Every ring of the named corner `g26_hot_selfheat` (+8 dBm, peak ~2.5x full scale) ended in FAULT | new state `ST_DECLIP = 7` ('declip'): ramp red by `ramp_step_codes`, recording, until y < f x full scale; slope from the last 2K+1 unclipped points; LOCK | `test_monitor_clipped_at_the_adc_full_scale`, `test_ctrl_a_corners.py` |
| 4 | dither_lockin.py (Python + compiled), bias_dither.py | a fall before the first complete dither cycle (or within two cycles of a recapture) left the reference level at 0, so the recapture ended at the first 2 % dip past the guard, on the COLD branch (toy, 200-code noise: LOCK at x = -3.8) | the reference is set to the pre-fall level at the fall | `test_recapture_after_an_early_fall_waits_for_the_red_side` |
| 5 | bias_dither.py | falls not detected during the heater identification: the ID then measured the cold branch (dy0/du > 0) and the ring ended in FAULT on the cold branch | falls detected in HOLD, ID and LOCK; after the recapture HOLD/ID restart (partial sums cleared) and the hand-off r_set is kept (new int state `rset_ok`, `_SI` 12 -> 13) | `test_fall_during_identification_recaptures_and_keeps_r_set` |
| 6 | midranging.py | after a fall and recapture the bias loop took over at once, just red of the peak and far from the set-point (beyond the small bias range in g26) | the heater PI settles again (`lock_steps` within `lock_tol`) before RUN | `test_toy_fall_in_run_resettles_the_pi_before_midranging` |
| 7 | midranging.py | telemetry `locked` stayed True in RUN (the PI's lock count was frozen at the hand-off) while the bias saturated and the monitor sat off the set-point (07 P1-20) | the lock count follows the bias loop in RUN | `test_toy_locked_telemetry_follows_the_bias_loop` |
| 8 | feedforward.py | T_ref was taken per world once EVERY ring had locked: one ring that never locks switched the feed-forward off for the bank (and later-locking rings started with a bump) | T_ref (and the K_ff reference code) per ring at its first lock | `test_feedforward_starts_per_ring` |
| 9 | pi_lock.py, dither_lockin.py | Python versions used fractional `init_codes` as given, the compiled ports round them: parity broke (209 of 600 codes differed) | Python rounds too | `test_python_init_codes_are_whole_codes_like_the_compiled_port` |
| 10 | examples/template_compiled_controller.py | read `obs[-1 + i]` when `tx.heater.applied` is not exposed | guarded | (template) |
| 11 | midranging.py | no fallback when the bias saturates (04 C13 "limited bias range"): in g26 the bias moves the monitor only ~370 codes over its window, so it saturates within ~0.1 ms of the A1 ramp and the heater then moved at most k_o s_b/s_h b_range ~0.75 codes/step (~10x short): lock lost at the A1 rate (max 470 pm) | `saturation_fallback` (default on): while the bias sits at a rail the heater also runs the PI velocity law on the error. A1 rate now: victim rms 4.2 pm, max 9.7 pm | `test_toy_bias_saturation_falls_back_to_the_heater_pi`, assertion in `test_bias_controllers_on_a1_rate` |

### 7.2 Tests added

- `test_ctrl_a_nojit.py`: the pure-Python bodies (`.py_func`) of `pi_step`, `dither_step`, `bias_step`, `bias_fast`
  equal the jitted ones on the toy ring (fast); the toy/unit tests of all owned files in a subprocess with
  `NUMBA_DISABLE_JIT=1` (slow, ~40 s; 35 passed + 1 strict xfail).
- `test_ctrl_a_corners.py` (slow, ~3.5 min): `pi_lock` MAP + lock on every named corner preset (g26_slow_thermal,
  fast_thermal, hot_selfheat, low_eta, undercut, heater_loss, overcoupled, textbook_thermal, h86) and loop latency x2
  (`uncertainty.timing.latency_scale: high` plus 10 ticks of controller pipeline); type-2 PI under A1 rate on
  slow_thermal, hot_selfheat, heater_loss, latency x2.
- the regression tests of 7.1; `test_fast_red_step_is_not_mistaken_for_a_fall` (strict xfail, 7.4).
- harness: `placed(..., preset=, alts=)`, `load_cfg(overrides, preset, alts)`; `ToyRings` sets `adc_overrange`
  when a sample sits at the 16-bit rail; the plant capture test now also asserts a settled lock (max error < 1 pm).

### 7.3 Robustness measured (Truth in tests only; B = 1, 8 rings, +170 pm hand-off)

| corner | lock detuning (pm) | max lock error steady (pm) | time to lock | A1 rate victim rms / max (pm), type 1 ; type 2 |
|---|---|---|---|---|
| g26 nominal | 121-126 | 0.05 | 3.9 ms | 2.8 / 8.4 ; 2.0 / 6.8 (builder) |
| g26_slow_thermal (x2) | 121-126 | 0.05 | 3.9 ms | 3.4 / 10.4 ; 2.6 / 8.9 |
| g26_fast_thermal (x0.5) | 121-126 | 0.05 | 4.0 ms | 2.5 / 7.4 ; 1.9 / 6.0 |
| g26_hot_selfheat (+8 dBm) | 273-316 (DECLIP, f x full scale; relOMA -10.8 dB vs -6.45) | 0.05 | 4.6 ms | 3.4 / 10.4 ; 2.6 / 8.7 |
| g26_low_eta | 121-126 | 0.11 | 2.3 ms | 4.2 / 11.0 ; 2.0 / 7.3 |
| g26_undercut | 122-126 | 0.03 | 4.0 ms | 2.8 / 8.4 ; 2.0 / 6.8 |
| g26_heater_loss | 130-136 | 0.05 | 3.8 ms | 3.9 / 11.3 ; 3.2 / 9.7 |
| g26_overcoupled | 122-126 (peak 65.4 k: just below the full scale) | 0.05 | 3.8 ms | 2.6 / 7.9 ; 1.9 / 6.3 |
| g26_textbook_thermal | 124-127 | 0.05 | 3.7 ms | 2.6 / 7.0 ; 2.1 / 5.4 |
| g26_h86 | 121-126 | 0.05 | 4.1 ms | 2.2 / 6.2 ; 1.3 / 4.5 |
| latency x2 + 10-tick pipeline | 121-126 | 0.06 | 3.9 ms | - ; 2.0 / 6.8 |

Every corner locks with no falls; the type-2 PI keeps the victim within 4.5-9.7 pm at the A1 peak rate (one
excursion above 5 pm, at the cusp, 0.9-2.3 ms), neighbours within 0.6 pm. The 5 pm budget is met by none of them at
the cusp: the error peaks where the aggressor rate reverses.

Other re-runs on the final code: `test_ctrl_a_scenarios.py` 19 passed (A1/A2 rates within 1 % of section 4; full A1
on 25 degC unchanged: error > 50 pm from 41.5 ms, 19 neighbour falls); `test_ctrl_a_engine.py` 2 passed, including
the relaxed open-loop check the builder had not re-run; the closed loop now locks ALL 8 rings at 126-127 pm on both
engine paths (identical), 0.005 simulated s per wall s on this machine with a warm compile cache.

### 7.4 Limitations confirmed (documented, not fixed)

- Fast RED steps read as falls: every lock here detects a fall as a drop of > 0.35 M_pk within 3 steps to below
  0.4 M_pk. A red step of the line or ring of ~200 pm within ~30 us does the same; RECAPTURE then ramps the heater
  further red, never sees a peak and gives up (FAULT at the fall code). Measured on a toy ring; strict xfail
  `test_fast_red_step_is_not_mistaken_for_a_fall`. A1/A2 rates (~183 pm/ms of resonance target) are far slower. A
  direction-aware recapture (watch the sign of dy/du during the ramp after the fast self-heating transient) would fix
  it; not attempted here.
- Ratio-mode bias dither needs a hand-off inside the half width: |r| is largest at x = 1 and r(x) = r(1/x).
- DECLIP locks at f x full scale: with the peak at ~2.5x full scale that is ~1.6 half widths red (relOMA -10.8 dB in
  g26_hot_selfheat). The monitor PGA has no gain below 1 (content request 6 stands).
- Section 5 items still hold (laser power steps move the fraction-of-peak lock; peak locks fall off the fold;
  bias authority tiny in g26; heater dither cannot follow A1).

## 8. Re-run after the aggressor-site change (2026-10-10, integrator; docs/handoff/integrator.md section 7)

The g26 default aggressor site moved from (0, 30) to (0, -30) um: the same distance to every ring, 55 um instead of
5 um from ring 0's monitor PD. The A1/A2-rate numbers of sections 4 and 7.3 were measured at the old site, where the
PD observer (0.1 ms pole, 2051 K/W) added a fast dark-current transient to the monitor at every aggressor slope
change; at the new site (3.4 ms pole, 186 K/W) it is small. `test_ctrl_a_scenarios.py` (17 passed) and
`test_ctrl_a_corners.py` (14 passed) on the default site, victim ring 0, rms / max lock error (pm), old -> new:

| controller, scenario | old site | new site |
|---|---|---|
| pi_lock, A1 rate ; A2 rate | 2.8 / 8.4 (1 excursion) ; 2.8 / 7.3 | 1.32 / 1.98 (0) ; 1.32 / 1.78 (0) |
| pi_lock type 2 (and _c), A1 rate ; A2 rate | 2.0 / 6.8 (1) ; 2.2 / 6.1 | 0.33 / 2.74 (0) ; 0.29 / 2.23 (0) |
| feedforward_pi (type 2), A1 ; A2 rate | 2.0 / 6.8 ; 2.2 / 6.1 | 0.32 / 2.66 ; 0.29 / 2.23 |
| midranging, A1 rate | 4.2 / 9.7 | 3.23 / 4.44 |
| bias_dither, A1 rate | 12.3 / 28.8 | 12.2 / 28.4 |
| dither_lockin, A1 ; A2 rate | 52 / 113 ; 50 / 89 | 53 / 117 ; 51 / 97 |
| es_sine, A1 ; A2 rate | 546 / 1030 ; 712 / 1124 | 544 / 1034 ; 716 / 1133 |
| type-2 PI on the corners at the A1 rate (slow_thermal, hot_selfheat, heater_loss, latency x2) | 4.5-9.7 max, one excursion at the cusp | 2.0-2.9 max, no excursion; neighbours <= 0.47 |

So the fraction-of-peak PI (and the feed-forward PI, which equals it here) now meets the 5 pm budget at the A1 and
A2 peak rates on every corner tested; the old cusp error was mostly the PD's dark current. Checked with the current
code at the stress site (`a1_rate_scenario(..., site=lib.AGGRESSOR_PD_ADJACENT_SITE)`, a scratch run): pi_lock 2.76 /
8.30 (A1) and 2.82 / 7.21 (A2), type 2 2.01 / 6.76 and 2.20 / 6.04, one excursion each: the old column. The full printed A1 on the
25 degC base (`test_a1_full_on_the_25c_base_monitor_dark_current`) now runs the stress scenario
`a1_slide10_aggressor_pd_adjacent` (the old site): unchanged, error > 50 pm from 41.6 ms, 21 neighbour falls. Request
4 is resolved (the coupling was not intended); the dark/baseline subtraction it suggested is still worth having.
