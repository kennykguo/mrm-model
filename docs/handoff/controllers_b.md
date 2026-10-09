# Handoff: controllers B (`controllers_b`): acquisition, WDM supervision, bit statistics, factory calibration

Owner files: `mrm/control/baselines/{ramp_acquire,wdm_supervisor,bitstat_lock}.py`, `mrm/control/calibration.py`,
`tests/verification/test_ctrl_b_*.py` (`plant` = the closed-loop bench, `ramp`, `ramp_aggressor` (slow), `bitstat`,
`bitstat_density`, `wdm`, `wdm_slow` (slow), `calibration`, `engine`, `robust` (review: sampled worlds, corners,
swing loss)). Section 6 lists what the adversarial review changed. Contract: ARCHITECTURE section 6 items 4, 6,
10; 07 P0-1, P0-11, P1-1, P1-12, P1-19, P1-20; research 04 C1, C6, C14 and 01 PAT-1/PAT-3.

## 1. Public API

```python
# mrm.control.baselines.ramp_acquire
RampParams(alpha=0.5, dw_up=None, dw_down=None, dw_reapproach=None, detect_frac=0.05, drop_frac=0.15, jump_frac=0.08,
           lock_bw_hz=3000.0, lost_frac=0.15, lost_steps=20, max_retries=3, w_start=0.0, w_lo=0.0, w_hi=1.0,
           rings=None, start_delay_s=50e-6, law="voltage", side="red")          # dataclass = controller.params
RampFSM(spec, prm: RampParams, design: NominalDesign | None = None)          # vectorized (B, N) state machine
    .start(mask, w_from=None, w_lo=None, w_hi=None, skip=0); .start_map(mask, w_from=0.0, w_hi=None, n_events=None)
    .park(mask, w); .park_from_below(mask, w); .step(y, valid, clipped=None) -> codes (B, N) int64
    .detuning_est_pm(y); state/y_pk/w_pk/y_bg/setpoint/ki/jumps/bistable/partial/ev_w/ev_y/ev_wfloor/ev_n ...
    .dw_up (B, N) up-sweep step in use (property; ``fsm.dw_up[b] = x`` sets one world), .dw_up0 (configured scalar)
    .sk_w (B, N, E) / .sk_n: drives of the resonances the running UP search passed outside its window
monitor_mean(obs, prev=None) -> (y (B, N) float, valid (B, N) bool, clipped (B, N) bool)
@register("ramp_acquire")      class RampAcquire(BaseController)      # Params = RampParams; writes tx.heater
@register("acquire_then_lock") class AcquireThenLock(BaseController)  # (acquire=None, lock=None, handoff_steps=20,
                                                                     #  lock_controller=None); from_params(params)
    # the lock is built at construction (default mrm.control.baselines.pi_lock:PiLock) and latency_ticks = the
    # lock's, so the runner (which reads latency_ticks before reset) sees it
ST_IDLE, ST_UP, ST_DOWN, ST_REAPPROACH, ST_LOCK, ST_FAIL, ST_HOLD, ST_PARK = 0..7; STATE_NAMES

# mrm.control.baselines.bitstat_lock
BitStatParams(RampParams, alpha=0.5, alpha_eye0=0.75, alpha_min=0.35, alpha_max=0.92, po_step=0.01,
              po_dwell_steps=60, settle_steps=40, rho_min=0.2, rho_max=0.8, eye_min_frac=0.06, one_is_red=True,
              mode="ratio", ratio_gain=0.01, ratio_capture_steps=30, ratio_capture_delay_steps=100,
              step0_frac=0.02, step_min_codes=2.0, step_max_frac=0.1)
@register("bitstat_lock") class BitStatLock(BaseController)   # writes tx.heater, datapath.bitstat_threshold
    # .n_hist = 1 + latency_ticks // period: the window of edge j used the threshold sent at edge j - n_hist
    # .alpha_floor = max(alpha_min, 0.55) in mode 'ratio' (R peaks at alpha 1/2), alpha_min in mode 'po'

# mrm.control.baselines.wdm_supervisor
WdmParams(RampParams, policy="cyclic", map_w_max=0.85, map_dw=None, supply_margin=0.95, excess_code_frac=0.97,
          excess_steps=200, verify_height_frac=0.5, window_frac=0.5, start_delay_s=100e-6, w_min_target=0.0)
@register("wdm_supervisor") class WdmSupervisor(BaseController)
    WdmSupervisor(params=None, *, factory_table: FactoryCalTable | None = None, **overrides)
    .load_factory_cal(table | None)   # warm start (07 P1-1): call before reset
    writes tx.heater and, when bound to the controller, tx.assignment
PH_WAIT, PH_MAP, PH_ASSIGN, PH_ACQUIRE, PH_TRACK = 0..4; PHASE_NAMES

# mrm.control.calibration
KNOWLEDGE_TOKENS, FIELD_TOKENS, has_knowledge(spec, token) -> bool
NominalDesign (frozen dataclass; fields None without their token; fsr_range_hz = the FSR leaf's declared range)
    + fsr_m, fwhm_m, tuning_m_per_w, p_drive_fs_w, tuning_m_per_w_drive, fwhm_w
nominal_design(spec) -> NominalDesign
drive_exponent(law, default=2) -> int; w_of_code(code, code_max, q); code_of_w(w, code_max, q, code_min=0)
heater_power_w(code, d) -> W; heater_current_a(code, d) -> A          # need heater_nominal
predicted_pattern(line_freq_hz, fsr_hz, first_line, k) -> (v, lines, orders)
label_lines_by_pattern(w_events, line_freq_hz, *, fsr_range_hz=(1.3e12, 2.1e12), n_fsr=321,
                       curvature_range=(-0.25, 0.75), n_curv=21, exponents=(1.0,), require_interior=True,
                       amb_min=2.0) -> dict(lines, orders, fsr_hz, curvature, exponent, offset, scale, rms, ambiguity,
                       edge, ok)          # g(w) = w^gamma / (1 + c w^gamma), gamma in exponents
label_bank_by_pattern(events, line_freq_hz, *, same keywords) -> list[dict]   # joint FSR/exponent/curvature
law_exponents(design, q) -> (1, 1/2) for q = 2, (1, 2) for q = 1      # the heater law is a hypothesis (07 P1-11)
fsr_prior_hz(design, rel=0.2, default=(1.3e12, 2.1e12), margin=0.03) -> (lo, hi)   # covers the FSR leaf's range
drop_first(events, n_lines) -> list[array]; with_first(fit, w_all, line_freq_hz) -> dict   # fit without resonance 0
CalBench (Protocol: spec, advance(inputs, n_ticks), sample_sensors() -> (sensors, new))
FactoryProcedure(ambient_c=25.0, settle_s=300e-6, group_size=1, w_max=0.85, dw=None, n_events=None, cool_s=300e-6,
                 line_coding=True, table_noise_codes=0.0, seed=0)  .line_sets(L) .plan(spec, design=None)
                 .scenario(spec, design=None)
FactoryCalTable(code, w, line, order, height, floor, ok, fsr_hz, scale, t_cal_c, q, code_max, ambient_c, notes)
    .events(b, n) -> (w, line); .predict_w(b, n, line, *, t_die_c=None, design=None, w_min=0.0) -> float
factory_cal(plant, procedure=None) -> FactoryCalTable      # plant = CalBench, or bench_factory(scenario) with .spec
```

Telemetry (B, N), all floats: state, locked, detuning_est_pm, integrator, integrator_saturated (the 07 P1-20 keys) plus
per controller: peak_level, peak_code, background, setpoint, jumps, clipped (ramp); eye_codes, eye_center, alpha_eye,
thr_hi, thr_lo, rho, data_ok, eye_ratio, eye_ratio_ref (bitstat); phase, target_line, assigned_line, map_events,
map_ok, wrong_line, line_confident, supply_limited, captured_at_start, shift, p_est_mw (wdm). ``line_confident`` =
locked and map_ok (confident pattern labels, or a factory entry marked ok) and not wrong_line and the lock sits where
the map put it (``_verify_lock``): the only flag that vouches for the line (07 P1-20). A ``locked`` ring with
line_confident 0 holds a lock on a line the supervisor could not verify.

## 2. Requests

- **integrator / baselines package**: the registry decorators run only when the modules are imported, and
  `mrm/control/baselines/__init__.py` is empty. Please add
  `from mrm.control.baselines import ramp_acquire, wdm_supervisor, bitstat_lock` (with the other owners' modules) so
  `registry.create("ramp_acquire" | "acquire_then_lock" | "wdm_supervisor" | "bitstat_lock")` works without an
  explicit import. Until then `"mrm.control.baselines.ramp_acquire:RampAcquire"` paths work.
- **params_content / thermal / scenarios (plant input finding)**: g26 places the aggressor site at (0, 30 um) and
  ring 0's monitor PD at its default offset `ring.thermal.pd_offset_um = (0, 25 um)`: 5 um apart. In the interim
  thermal model a 100 mW aggressor raises PD0 by +205 K and ring 0 by +34 K (steady state). Calibrated to A2 (ring
  +80.8 K), the PD0 observer passes 300 degC after ~25 ms (519 degC max) and its Arrhenius dark current (0.37 eV)
  reaches ~10 uA, half of the half-maximum lock current: every monitor-based lock (ramp, bit statistics, and the
  controllers-A locks) is dragged red and lost after ~35-40 ms of A2. Both positions are `assumed`; please choose
  non-colliding defaults (or clamp the dark-current model at a physical PD temperature). My aggressor tests use
  `ring.thermal.pd_offset_um=[0,-25]`; `test_a2_default_layout_pd_dark_current` (xfail, non-strict) documents it.
- **engine (metrics)**: `locked` / `excursion_count` / `time_to_lock_s` of mrm.sim.metrics are |d| <= 5 pm about the
  LINE. A side-of-fringe lock sits by design 0.3-0.5 FWHM red of the line (G26: half maximum, ~150-190 pm), so
  these metrics read "never locked" for every set-point lock. Suggest measuring the 5 pm budget about the oracle's
  operating point, or about the ring's own steady detuning, and keeping |d| about the line as a separate figure.
  My tests report lock error about the steady detuning (jitter rms, max, excursions with 1 pm hysteresis) plus the
  OMA penalty against the static optimum.
- **engine / runner (warm start)**: when the scenario's ResetSpec says `factory_cal=True`, call
  `ctrl.load_factory_cal(table)` (if the controller has it) before `ctrl.reset(spec)`.
- **core_control_api**: (a) the DAC facts lack the driver law; the drive coordinate w = (code/code_max)^q needs it,
  so without the `heater_nominal` token the controllers assume a voltage DAC (q = 2). A `DacFacts.law` (or `q`) field
  would make the coordinate right without device knowledge. (b) The optional `datapath.bitstat_ones` channel (the
  window's own ones count) would remove BitStatLock's n1 ~ rho W approximation (rho from tx.ones_count/tx.bits).
- **params / integrator (knowledge tokens)**: two vocabularies exist now. Mine (mrm.control.calibration):
  `heater_nominal`, `optics_nominal`, `dlambda_dt_nominal`, `monitor_nominal` (coarse); controllers A
  (mrm.control.baselines.pi_lock.KNOWLEDGE_KEYS): `heater_law`, `heater_r_nominal`, `heater_efficiency`, `linewidth`,
  `dlambda_dt_nominal`, `thermal_dynamics`, `bias_efficiency` (fine). `calibration.FIELD_TOKENS` maps every field
  group I read to the tokens of BOTH vocabularies, so either works; please choose one list for the schema (and for a
  `controller_knowledge` preset, 07 P1-1) and tell both owners. g26's knowledge is only `[line_grid, ring_to_line]`,
  which releases no device numbers; my tests extend it by override.
- **integrator (ReferencePlant)**: `test_plant_wiring.ReferencePlant.release` passes threshold 0 to the bit-statistics
  sampler instead of the `datapath.bitstat_threshold` actuator; my bench subclass wires it. Also, a release must
  happen every controller period (`pop_period`), else the monitor FIFO overflows and `latest` is stale.

## 3. Results (g26 nominal world, B = 1 unless stated; Truth read in the tests only)

Reference bench and engine agree: ring 0's ramp lock sits at 148.3 pm red of line 7 on both (engine 0.9 s for 3 ms
against ~4 s on the bench at low load); the WDM cold start ends at the same detunings on both (to 0.1 pm).

| controller / scenario | lock error (about the steady detuning) | operating point | OMA penalty | excursions | time to lock |
|---|---|---|---|---|---|
| ramp_acquire, steady (ring 0, line 7) | jitter rms 0.12 pm, max 0.30 pm | 148 pm red (half max) | 0.65 dB | 0 | 0.70 ms |
| ramp_acquire, alpha 0.75 | - | 91 pm red | 0.02 dB | - | - |
| ramp_acquire, +10 % laser power | lock point moves +17.7 pm (stale set-point, 04 C3), jitter 0.04 pm after | - | - | 0 | - |
| ramp_acquire, A2 (PD far), bench, 12 ms of the 1616 degC/s ramp | rms 0.88 pm, max 1.79 pm | 172 pm | - | 0 | - |
| ramp_acquire, A2 (PD far), engine, full 100 ms cycle | max deviation 6.4 pm | 173 pm | - | - | - |
| ramp_acquire, A1 (PD far, line 4), engine, through the cusp | max deviation 5.9 pm (at the cusp) | 143 pm | - | - | - |
| ramp_acquire, A2, default g26 layout | lost after ~38 ms (PD0 dark current, section 2) | - | - | - | - |
| acquire_then_lock -> pi_lock(0.7) | jitter 0.8 pm (still converging at 4.5 ms) | 125 pm | - | - | - |
| bitstat_lock (ratio mode), steady | wander 0.3 pm p-p | 90.6 pm (eye max) | 0.02 dB | - | ~1.5 ms |
| bitstat_lock, ones density 1 / 0 plateaus | -0.6 / -1.1 pm | 90.6 pm | - | - | - |
| monitor lock (ramp alpha 0.75), same plateaus | -28.8 / +30.7 pm | 90.5 pm | - | - | - |
| bitstat_lock, squelch 1.5 ms | statistic frozen, lock held; detuning 90.7 -> 99 pm (no swing) | - | - | - | - |
| bitstat_lock, +10 % laser power (engine) | +5.0 pm after 8.5 ms (monitor lock: +20.0 pm) | 91 pm | - | - | - |
| wdm_supervisor, cold start, fixed map (bench and engine) | all 8 rings on their declared lines; jitter 0.02-0.05 pm | 157-183 pm | 0.60-1.15 dB | 0 | 19.3 ms (8 x fresh map + acquisition) |
| wdm_supervisor, warm start from factory_cal (engine) | all 8 on their lines | 157-183 pm | - | - | < 14 ms |
| wdm_supervisor, 0.10 A supply, cyclic (engine) | barrel shift 6, all locked on (n + 6) mod 8, 63 mA | 147-224 pm | - | - | ~27 ms |
| wdm_supervisor, 0.10 A supply, fixed map (engine) | 5 rings locked in bus order, 3 flagged supply_limited | - | - | - | - |
| wdm_supervisor, 0.16 A, ambient -40 K in 5 ms (engine) | online PAT-1: map rotated by one channel, all 8 re-locked, no wrong line | - | - | - | - |
| wdm_supervisor, 18 sampled worlds (seeds 7, 13, 19; review) | 77/144 rings on their declared line, 38 vouched (line_confident), all 38 right; 5 worlds fully right (section 6) | - | - | - | - |

The cold-ring hazard is real and reproduced: at turn-on 7 of 8 nominal rings sit 0.14-0.18 nm red of a foreign
line (n + 3, or the next order for rings 6, 7); the standalone ramp acquisition then puts every ring on a line other
than the declared one and two rings share a line (the downstream one starves). The supervisor flags captured rings
(`captured_at_start`, from the monitor at w = 0), maps each ring right before acquiring it (bus order: the rings
before it are locked on their own lines, the ones after it do not touch its input), labels the lines from the empty
gap of the g26 grid, and ends with every ring on its line.

Two things that broke simpler designs, found in closed loop (for whoever extends this): (1) the resonance comb of a
ring drifts by up to half a spacing between a bank map and its acquisition (crosstalk from the rings locked before
it, ms thermal tails, sweep lag), so a stored window or a count from a park position picks a neighbour; a fresh map
taken right before the acquisition is robust. (2) Under bistability the state at a drive depends on the approach:
a ring dropped from a high drive is held by the hot branch of a line, so every search goes up from code 0 (or from a
park reached from below), as the maps did.

Self-heating jump (07 P0-1), shown on Truth: during a heater-up sweep the detuning goes -310 -> +119 pm in 4 us for
one 97 pm heater step; the up-sweep lands at 0.65 of the true peak (32.5 k vs 49.8 k codes). The DOWN scan measures
the true peak; the set-point is taken from it.

Python per-step cost (B = 1 and 16, load average ~23): RampAcquire 0.4-0.5 ms, BitStatLock 0.5-1.0 ms,
WdmSupervisor 0.5-0.8 ms per controller step. Engine wall times: 3 ms of a ring lock ~1 s; the WDM cold start
(24 ms) ~6 s; a full A2 cycle (110 ms) minutes under heavy load.

## 4. Deviations and doubts (please review)

1. No compiled (numba) versions: none of these controllers is in the (c) list of ARCHITECTURE section 6. The FSM is
   vectorized numpy; a compiled ramp FSM would be the next step for long soak runs.
2. Only the red (self-heating-stable) side is implemented for the set-point (07 P0-1 item 7). Blue-side locks are
   refused at reset.
3. Cyclic policy: the planner keeps the declared map when it fits, else takes the feasible barrel shift with the
   smallest total drive (lock-to-cyclic, [CHOI]); the online PAT-1 step in TRACK is the patent's one channel towards
   lower power, after the excess has persisted for ``excess_steps`` (2 ms).
4. The bit-statistics lock's inner loop is the monitor loop; the bit statistics set its set-point: the eye centre
   (ones-density independent), Sun's ones-density canceller, and (default ``mode='ratio'``) the eye ratio
   D / (c - y_bg), which does not depend on the optical power and is held by an integral law, so a laser power step
   is mostly undone; ``mode='po'`` is Sun's perturb-and-observe on D (wanders ~10 pm on the flat eye top). The
   canceller's sign uses the plant's NRZ convention (level '1' = higher reverse bias = redder resonance, an integrator
   assumption): ``one_is_red``. A squelch removes the swing: the statistic freezes and the lock moves ~8 pm.
5. Factory calibration: labels from the joint pattern fit of the all-lines sweeps (one ring at a time in bus order,
   swept rings parked between lines); the binary line coding is the model-free cross-check, and disagreements are
   listed in ``FactoryCalTable.notes`` (the ms tail and captures at w = 0 make the coding fragile).
6. Warm start: the table's windows (0.75 of the spacing below, 0.35 above, temperature-shifted with the nominal
   dlambda/dT when known) and the drift of the rings already acquired; the supervisor does not wait for the first
   temperature-sensor sample (2 ms at g26), so a warm start far from the factory's ambient is not corrected.
7. Line identification needs one FSR of heater range plus two resonances (w ~ 0.68-0.75 at g26 nominal). In sampled
   worlds with P_FSR up to 41 mW the heater cannot cover it: ``map_ok`` False, and those rings can end on a foreign
   line (reported in telemetry: line_confident 0). A supply-capped map (``_w_cap``) can make a ring
   ``supply_limited``. The cold start is NOT robust over the uncertainty set (section 6: about half the rings of
   sampled worlds end on a foreign line); what holds is that it never vouches for a wrong line.
8. PAT-3 TX-RX pairing dithers (1 kHz-1 MHz, mutually irrational) are not implemented: the wrong-line checks are the
   line-pattern labels, the acquisition window around a fresh map, and the acquired-peak height against the mapped
   height ([HAT24]).
9. The RampFSM also has a COUNT search (pass k resonances, take the next) that the supervisor no longer uses (it
   proved fragile, point (1) of section 3); it is kept for users who start from a known state.

## 5. Tests

(Builder's list; the review added tests, section 6.) Default (`-m "not slow"`), 30 tests in 7 files, 87 s in total at
load ~12 (largest file ~25 s):
`test_ctrl_b_plant` (bench checks), `test_ctrl_b_ramp` (protocol, jump, steady, alpha, laser step, standalone cold
start, partial peak, composition with pi_lock), `test_ctrl_b_bitstat` (registry, trackers, eye max, squelch),
`test_ctrl_b_bitstat_density`, `test_ctrl_b_wdm` (cold start, hazard, planning, online-shift logic, policy fallback,
no-Truth import scan), `test_ctrl_b_calibration` (knowledge gate, drive coordinate, nominal heater model, pattern
labels, factory plan), `test_ctrl_b_engine` (ramp and bitstat on the engine, cross-checked with the bench). Slow
(`-m slow`): `test_ctrl_b_ramp_aggressor` (A2 bench and engine, A1 engine, default-layout xfail),
`test_ctrl_b_wdm_slow` (barrel shift, fixed map under a tight supply, warm start, sampled worlds, online PAT-1),
`test_ctrl_b_engine` (WDM cold start, bitstat laser step), `test_ctrl_b_calibration::test_factory_cal_on_the_
reference_bench` (~90 s). The engine tests skip when `mrm.sim.runner` is not importable. Unit tests also pass with
`NUMBA_DISABLE_JIT=1`.

## 6. Adversarial review (2026-10-09)

The reviewer of this task (same owner files) re-ran every test, tried to break each controller on the engine
(sampled worlds of the g26 uncertainty set, the named corner presets, a swing loss) and fixed what was a bug.

### 6.1 Bugs fixed

1. WDM, B > 1: the bank pattern fit was ONE attribute (`self.fits`), overwritten by each world's `_label`; a world's
   fresh-map labels then used the FSR/curvature window of the last world labelled. Now per world
   (`test_state_is_per_world_and_reset_clears_it`: worlds at 1.8 and 1.47 THz).
2. WDM: `_synced` and `fits` survived `reset` (read with getattr defaults): after a controller reset (scenario
   `fault.controller_reset`, the runner calls `reset`) the supervisor kept stale codes and fits. Reset clears them.
3. `RampFSM.dw_up` was one scalar that the WDM rewrote per world: with B > 1 a world mapped at another world's step
   whenever the map and acquisition steps differ (any `map_dw` / `dw_up` override). Now (B, N) (`dw_up0` = scalar).
4. Start delays compared the ABSOLUTE tick with the delay (RampAcquire, BitStatLock, WdmSupervisor): after a mid-run
   controller reset the delay was skipped. Now counted from the first step after the reset.
5. FSR prior of the pattern fit with the optics knowledge was nominal +- 15 % = 1.53-2.07 THz: it excluded FSR window A
   (1.47 THz, one third of the g26 FSR hypothesis weight). A window-A world was then labelled wrongly AND
   confidently (closed loop, sampled world 4 of seed 7). Now the FSR leaf's declared range (1.4-1.9 THz, "spans every
   FSR hypothesis") + 3 %, joined with nominal +- 20 % (`NominalDesign.fsr_range_hz`).
6. Drive law (07 P1-11, drawn per world): a power-linear driver (power, dac_pwm) read in the q = 2 coordinate gives
   P ~ w^(1/2), which the one-parameter curvature cannot represent: such maps were never labelled right. The fits now
   also try that exponent (`law_exponents`): right in ~70-80 % of synthetic maps, never confidently wrong.
7. Fit acceptance: ambiguity > 1.5 let ~1 % of synthetic maps with line-dependent self-heating jump offsets (+-0.01
   in w) through confidently wrong; now `amb_min = 2.0` (0 of 384, ~80 % of the right labels kept), and a best fit
   on the border of the FSR or curvature grid (`edge`) is not ok.
8. `calibration.__all__` listed `find_resonances`, which does not exist (`import *` raised AttributeError).
9. `nominal_design` with a plain-mapping `spec.nominal` returned config units (mW, THz, nm, degC) as SI; it now
   converts through `mrm.params.units.to_si`.
10. BitStatLock paired each window with the tracker sent at the previous edge, true only for latency_ticks < period;
    now a history of 1 + latency_ticks // period edges (07 P0-6: release, then latch).
11. BitStatLock ratio mode let alpha fall to alpha_min = 0.35, past the maximum of R = 2 x dx / (1 + x^2) at
    alpha = 1/2, where the integral law on R runs away; floor 0.55 in ratio mode.
12. AcquireThenLock built its lock and set latency_ticks at reset, but the runner reads latency_ticks BEFORE reset: a
    lock with latency 3 ran with 0. Built at construction now; default lock `mrm.control.baselines.pi_lock:PiLock`
    (the name `pi_lock` is not registered while `baselines/__init__.py` is empty).
13. WDM `wrong_line` (height check) was never cleared after the retry on the next order succeeded; it now judges the
    latest lock.
14. WDM planner: with no feasible barrel shift it silently kept the declared map (the costliest, possibly); now the
    least-drive shift, and the supply sequencing flags the rings that do not fit.
15. `heater_current_a` raised for a current-law driver without a current full scale.

### 6.2 Added

- WDM telemetry `line_confident` and `_verify_lock` (07 P1-20): a lock is vouched for only when its labels are
  confident (or the table entry ok), the height check passed, and the found resonance AND the last resonance the
  up-search passed below it sit where the map put them (common drift within 0.3 of the local spacing of the drift the
  window assumed). `RampFSM.sk_w / sk_n` record the passed resonances.
- `WdmParams.w_min_target` (PAT-1 [0050] heater headroom, default 0 = off). Tried at 3 nominal linewidths: it broke the
  0.10 A barrel shift, whose lowest-power map sits at w ~ 0.02-0.05.
- Tests: `test_ctrl_b_robust.py` (engine: ramp and bitstat in 8 sampled worlds, ramp on 7 corner presets (slow),
  bitstat ratio lock under a 20 % swing loss), unit tests for every fix above (calibration, wdm, ramp, bitstat), the
  sampled-world WDM test now 6 worlds asserting no vouched wrong line.

### 6.3 Tried and reverted

- Moving the acquisition window by a drift measured from the passed resonances (sequence alignment): on the
  near-periodic g26 pattern a drift of one spacing cannot be told from a search that starts one resonance later (a
  different capture at w = 0); it broke two worlds that the fixed window gets right. Only the verification stayed.
- Excluding the first resonance of a map that began captured: broke the barrel shift (same reason as the floor).

### 6.4 Results (engine; Truth in the tests only)

| case | outcome |
|---|---|
| ramp_acquire ring 0, 8 sampled worlds | all LOCK on the red side of a line, 90-340 pm, p-p <= 0.32 pm; 4 `partial`, 3 with the monitor ADC clipped at the peak (set-point on a clipped amplitude: lock further red) |
| ramp_acquire, corners hot_selfheat, low_eta, undercut, fsr_window_a, textbook_thermal, slow_thermal, overcoupled | all LOCK red, p-p <= 0.2 pm; hot_selfheat 297 pm (partial), the others 139-215 pm |
| ramp_acquire with g26's own knowledge (no device numbers) | 148.1 pm, as with the knowledge |
| bitstat_lock, 8 sampled worlds | all LOCK red, data_ok; p-p <= 2.5 pm (the dim world); the clipped world 5.9 pm (R meaningless when y_bg is near full scale) |
| bitstat_lock (ratio), data swing -20 % | alpha settles at the floor 0.55; lock held, +38 pm red |
| wdm_supervisor cold start, 18 sampled worlds (seeds 7, 13, 19, B = 6, 30 ms) | 77/144 rings on their declared line; 38 vouched, all right (0 vouched wrong; before the review seed 7 alone had 3 rings on foreign lines with map_ok); fully right: 5 worlds |
| wdm_supervisor with g26's own knowledge, nominal | all 8 on their lines |
| wdm_supervisor 0.10 A barrel shift / warm start / online PAT-1 / fixed map | as section 3 (re-run) |

Why the WDM cold start fails in sampled worlds (it is a hard scenario, reported, not fixed): P_FSR ~40 mW (the map
cannot span an FSR + 2 resonances: labels not ok); the monitor ADC clipping at the drop peak (gain 0.5-2 and +0..+8
dBm at the ring against a 40 uA full scale with the nominal peak at 76 % of it); window-A pairs 1 nm apart, closer
than the self-heating jump offsets; strong self-heating with a 10 % ms tail, which leaves a ring warmer after its own
fresh map so its comb drifts by up to one spacing before the acquisition (the window then holds a neighbour). A design
that avoids the last one: after the fresh map, count resonances DOWN from the map's top (a down-scan crosses each
resonance continuously on its stable red branch, with no capture ambiguity), then up-search from just below the
target.

### 6.5 Requests added by the review

- **engine / runner**: `runner._make_controller` calls `registry.create(name, **controller_params)`, but `create`
  takes `params` positionally: any non-empty `controller_params` raises TypeError.
- **params_content / sensing**: `ring.monitor.adc_full_scale_ua = 40` puts the nominal drop peak at ~76 % of full
  scale; over the uncertainty set (monitor gain 0.5-2, power at the ring 0..+8 dBm) the monitor clipped at the peak
  in 3 of 8 sampled worlds (ramp test) and on some rings of 5 of 6 (WDM test), and the PGA gains (1, 2, 4, 8) cannot
  attenuate. Monitor-based set-points then use a clipped amplitude.
  Please add headroom (a gain < 1, or a larger full scale) or state that clipping is intended.

