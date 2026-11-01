# Handoff: engine (compiled kernel, NumPy reference, Plant API, plant I/O, online metrics, minimal runner)

Owner files: `mrm/sim/kernel.py`, `mrm/sim/kernel_ref.py`, `mrm/plant/plant.py`, `mrm/plant/io.py`,
`mrm/sim/metrics.py`, `mrm/sim/runner.py`; tests `tests/verification/test_kernel_parity.py`,
`test_engine_timing.py`, `test_metrics.py`, `test_engine_physics.py`, `tests/perf/test_perf.py`.
Contract: 07 P0-6, P0-7, P0-8, P0-9, P0-10, P1-18, ARCHITECTURE Truth; physics tests of P0-1 (a)-(d), P1-2, P1-7,
P1-8, P1-9, P1-13, P2-1.

## 1. Public API

```python
from mrm.sim.runner import run, RunResult, solve_locked_steady_state, LockedSteadyStateResult
run(cfg_or_params, controller=None, scenario=None, duration_s=1e-3, batch=1, seeds=None, record=None, metrics=None,
    **options) -> RunResult            # options: substep_policy, uncertainty, variation, reset_protocol, heater_code,
                                       # path, n_threads, oracle, histogram, chunk_ticks, controller_params, cfg,
                                       # auto_max_ticks, oracle_period_ticks
RunResult(metrics: dict, trace: dict | None, truth_final: Truth, manifest_inputs: dict, diagnostics: dict,
          wall_s: float, plant: Plant)
solve_locked_steady_state(cfg_or_params, *, ambient_c=None, bias_v=None, batch=1, seeds=None, p_in_w=None,
                          n_scan=301, iterations=4, cfg=None, uncertainty=None, variation=None, order="heater_up",
                          data_state=None) -> LockedSteadyStateResult

from mrm.plant.plant import Plant, Truth, prepare_scenario
Plant(pp, scenario=None, spec=None, *, cfg=None, controller_latency_ticks=0, noise_seed=0, scenario_seed=0,
      metrics=None, record=None, record_ticks=0, substep_policy=None, auto_max_ticks=64, fast_kernel=False,
      oracle=True, histogram=None, calibrate_aggressors=True, sh_tol_k=1e-7)
  .reset(noise_seed=None, *, k0=0, protocol=None, heater_code=None, bias_code=None)
  .advance(inputs: dict | None = None, n_ticks: int = 1)          # inputs: channel -> (B, ...) codes, validated
  .sample_sensors() -> (sensors, new)                             # channels released at the last controller edge
  .truth() -> Truth          .metrics() -> dict          .traces() -> dict          .diagnostics() -> dict
  .run(n_ticks, *, mode=MODE_OPEN|MODE_COMPILED|MODE_PYTHON, step_fn=None, fast_fn=None, resume=False,
       n_threads=None) -> int
  .bind_compiled(ctrl: CompiledController)

from mrm.sim.kernel import get_run_block, KParams, KState, STEP_FT, null_step, MODE_OPEN, MODE_COMPILED, MODE_PYTHON
run_block = get_run_block(P, S)
run_block(step_fn, fast_fn, P, S, k0, n_ticks, mode, resume, done, n_threads=None)
dphi_dlamres(lam_l, shift, t_rt) -> rad/m          # 2 pi c t_rt / (lam_l - shift)^2  [njit]
cold_coordinate(rp_row, x0, target) -> m             # root of n(x) L / x = target on the phase model's cold curve
resonance_detuning(rp_row, lam_l, shift, m) -> m     # exact lambda_res(order m) - lam_l of a ring shifted by `shift`

from mrm.sim.metrics import MetricsSpec, MetricsState, RecordSpec, RecordSet, envelope, summarize,
                            metrics_from_trace, TRUTH_FIELDS, LOCK_REFERENCES
MetricsSpec(..., lock_reference='line'|'oracle'|'target', lock_target_pm=None)   # reference of the lock error
from mrm.sim.kernel_ref import ReferenceEngine      # ReferenceEngine(plant).run(n, controller); .metrics()
from mrm.plant.io import ChannelMaps, ActuatorBank, SensorBank, pack_obs_row, clip_act_row, pipe_push_row, ...
```

`Truth` fields (per ring, (B, N)): `mrm.sim.metrics.TRUTH_FIELDS` (temperatures of ring / heater resistor / PD,
heater power, absorbed power, resonance of the order nearest the declared line, detuning in m and in HWHM, monitor
optical input and current, P_in at the line, bus-output OMA, relOMA = OMA/P_in at the ring's own through port (D9),
ER, IL, applied heater code, bias, order, line the ring sits on, base temperature, latest monitor code, oracle
relOMA, BER, declared line); per line (B, L): chip-input power, bus-output power, wavelength; plus the optics
arrays `out_rl`, `out_r`, `out_l`, every thermal observer temperature, and for Truth-fed controllers (the oracle,
`mrm.validation.oracle.TruthState` duck type) `x` (B, M) thermal lag states, `heater_applied` (B, N) and
`conditions` (one `mrm.control.models.Conditions` per world: line wavelengths and chip powers after the scenario,
NRZ levels, ones and transition density, base temperature, aggressor power, data enable).

Metrics summary keys added by the review: `max_abs_lock_error_pm`, `rms_lock_error_pm`, `lock_reference_pm` (B, N),
`settings.lock_reference`. The lock state, excursions, time above thresholds, time to lock and FSR hops use the lock
error e = d - d_ref; `max_abs_detuning_pm`, mean, rms and the histogram stay about the line.

Record specs: `[(channel, decimation_ticks, reducer)]` with reducer mean | min | max | last over a Truth field
(`envelope(ch, dec)` gives min and max). Traces: `{"ch@dec:reducer": {"t_s": (S,), "value": (S, B, N)}}`.

## 2. How it is built (read before extending)

- `Plant._build` turns PlantParams + Scenario + ControllerSpec into two namedtuples of arrays, `KParams` (178
  fields) and `KState` (90 fields); every array is a writeable C-contiguous copy so that the numba types never
  depend on the config (a read-only view is another type and would recompile the plant; `_check_kernel_arrays`
  raises instead).
- The per-world tick lives in `kernel._world_block` (one world, the whole block of ticks: views, per-block tables,
  scratch) and `kernel._world_loop` (the tick loop, compiled without the numba runtime since performance round 2,
  perf.md R2-11; the rules for what it may call are the comment above `_world_block`), in sections marked
  `# ---` (commands latched at k, scenario k / k+1, laser + laser_ramp, ones density, fault mask and dac_glitch, bias
  DAC with the fast offset, NRZ levels, optics inputs into the self-heating rows and their substep caches, bounds,
  data-activity heat, thermal inputs, substep length, heater TCR split + heater DAC path, implicit thermal +
  self-heating step, sensor inputs, sensors, Truth (`_truth_a`), oracle, metrics and records). Performance round 1
  (docs/handoff/perf.md P3) merged the former `_prepare_interval`, `_thermal_and_sensors`, `_account`,
  `_auto_substep`, `_heater_step`, `_bias_step` into it: a call that takes the KParams/KState tuples costs a
  reference-count update of all ~270 arrays. The event order is the module docstring of `kernel.py` (= ENGINE_BRIEF
  section 3 with the controller edge first). Since round 2 the loop calls array-taking helpers for the controller
  edge, the auto policy, the oracle and the fast kernel too (`_release_a`, `_sync_act_a` / `_push_act_a`,
  `_advance_edge_a`, `_auto_h_a`, `_oracle_a`, `_fast_tick_a`); the tuple-taking forms `release_row`, `_oracle`,
  `_auto_h`, `_fast_tick` and the public `truth_row` remain as wrappers for other callers (`_edge`, `_push_act`,
  `_advance_edge`, `_sync_act_from_registers` and `_edge_tick` were removed).
- Controllers: `step_fn` / `fast_fn` are typed as the first-class `STEP_FT` (the mrm.control.api ABI), so one
  compiled kernel serves every controller and is disk-cached (a Dispatcher argument would recompile the whole
  plant per controller and per process).
- Delays: the controller pipeline (Controller.latency_ticks) is `io.pipe_*` (one push per edge, capacity
  `fifo_capacity(latency, period, 1)`); every analog/converter delay stays in its leaf module (07 P0-6).
- The self-heating adapter `eng_pabs` / `eng_commit` is bit-identical to `optics.selfheat_pabs` /
  `selfheat_commit` (same P_abs and slope summation order; since performance round 1 the exact ring without
  backscatter runs the shared Airy path of `mrm.plant.optics`, `sh_fast_*`, with substep caches: perf.md P1, P4)
  and reuses the last evaluation at the commit when the solver commits the temperature it evaluated last; it saves
  one ring evaluation in three. The rows are `pack_selfheat_args` rows (with the optics adapter's cache blocks)
  extended by `extend_selfheat_rows` (EXT_I ints: the general path's through power, the K^2 key, the Truth FWHM).

## 3. Decisions and deviations (please review)

1. **prange over B is a thread pool over B** (07 P0-9 asks for `@njit(parallel=True, cache=True)` with prange). A
   numba parallel region cannot take the kernel's namedtuple arguments: the parfor tuple limit is 100 fields, nested
   tuples are refused ("scalar type ... given for non scalar argument"), and with `PARFOR_MAX_TUPLE_SIZE` raised a
   trivial prange over a 180-array tuple did not finish compiling in 9 minutes. The chunk function is
   `@njit(nogil=True, cache=True)` with an explicit signature; worlds are split into contiguous chunks on a
   `ThreadPoolExecutor`. Results are bit-identical for any thread count (tested). No reduction across b.
2. **Self-heating tolerance** `sh_tol_k = 1e-7 K` (ENGINE_BRIEF: 1e-9). 7e-6 pm of resonance; saves ~0.3-1 Newton
   iteration per ring and tick. Configurable (`Plant(sh_tol_k=...)`).
3. **Junction swing capacitance**: `kernel.swing_capacitance_row` (cumulative table) replaces
   `electrical.junction_swing_capacitance_row`, which walked every 10 mV grid segment with two binary searches each
   (50 us per tick for 8 rings, a third of the whole tick). Same integral; agreement 1e-12 relative tested. The
   reference engine uses the same function so the parity can be bit for bit.
4. **Bit statistics** use the actual bus background power, PD temperature and PGA setting (ReferencePlant passed 0,
   the dark-current reference temperature and PGA 0).
5. **Reset (07 P0-1 item 6)**: laser_ramp multiplies the chip power by min(1, (k - k_reset + 1)/ramp_ticks) with
   t_prev = -inf; branch_low: lasers on at once, t_prev = -inf; branch_high: t_prev = +inf. Heaters start at
   code_min (or `heater_code`, a warm start); the controller runs from the first edge (it sees the ramp in the
   monitor and must wait or not: a controller choice). The bias register starts at the scenario's code at k0 (no
   one-tick transient from code 0, which ReferencePlant had).
6. **Detuning and Truth** (changed by the review): the resonance of a ring is the mean of its two NRZ level
   resonances (the bias point), of the order m nearest the DECLARED line (tx.assignment when the controller owns it
   and writes a valid index, else channel_plan.ring_to_line), computed EXACTLY on the phase model's cold curve:
   d = x_m - (lambda_line - shift_mean) with x_m the root of n(x) L / x = m (`kernel.resonance_detuning`, five
   Newton steps) and shift_mean the mean level shift of `phase.ring_optics_row` at T_prev plus the R1 increment
   to T_end. The other lines (wrong-line window) keep the first-order d = (phi - 2 pi m)/(dphi/dlambda_res). The
   optics slope is `dphi_dlamres` = 2 pi c t_rt / (lambda_line - shift)^2 (mrm.plant.phase equation 2). The builder
   used 2 pi c t_rt / lambda_line^2, the slope of the UNshifted curve, as ENGINE_BRIEF section 3 step 6 says: that
   is 2 shift / lambda = 1.2 % too small after the 7.6 nm a g26 ring is heated by, so every Truth detuning was 1.2 %
   too large (-3069 vs -3040 pm; 3.6 pm at +300 pm, against a 5 pm lock budget) and the in-tick thermo-optic phase
   of the self-heating solve, the oracle's scan and the quasi-static survey carried the same factor. This was the
   "~1 % conversion difference" docs/handoff/controllers_a.md reported (harness vs engine now agree to 6e-5
   median). HWHM = lambda_line^2 fwhm_hz/(2c) with the realized fwhm_hz (not re-solved for the instantaneous loss). relOMA = |P_thru,1 - P_thru,0|/P_in at the ring's OWN through port (07 D9: "1 mW into the
   MRM"); the OMA for the BER is the bus-output OMA of the declared line (as ReferencePlant).
7. **Wrong line**: a ring sits on line l when its nearest resonance is within `wrong_line_window_fwhm` (1 FWHM) of
   l; wrong when that line is not the declared one (`fixed`), never (`any`), or when its offset (l - declared) mod L
   differs from the bank's most common offset (`cyclic`, the PAT-1 barrel shift).
8. **Oracle relOMA** (07 P1-1, P0-1 item 7): evaluated every `oracle_period_ticks` (1000) by an exact-ring scan of
   the mean detuning over +-1.5 FWHM (151 points) at the present inputs, stable points only (static loop gain
   s(T) Z_opt,self dP_abs/dlambda < 1, the declared line's absorption slope only), and unrestricted. Not a
   controller: the plant never hands it to one. Its best stable point's exact detuning is the lock reference of
   `lock_reference='oracle'` (item 21).
9. **Demanded peak slew** = max |register change| / (ticks since the previous latch) / dt over every controller
   latch; a latch at the reset tick has no interval and is skipped.
10. **Auto substeps (07 P0-10)**: power-of-two substeps h <= `auto_max_ticks` (64) chosen at each tick from: the next
    controller edge and command latch, the laser ramp, quiescent heater and bias DAC paths (no pending write, FIFO
    at the register, slew and settle at target), no dac_glitch, no PWM ripple, no fast kernel, scenario signals
    linear (base temperature, XPU rise, aggressor power: exact FOH) or constant (every other signal), the detuning
    rate (h |dlambda/dt| <= sim.max_detuning_step_pm), the P_abs change of the previous substep (> 1 % falls back to
    1 tick), record windows and the oracle tick. Within a substep: the thermal step is exact (coefficient tables for
    every power of two), the self-heating solve runs once with two Gauss-Seidel sweeps, the bias DAC noise draw,
    laser output and ones density of the first tick are held, and the per-tick sensors (monitor, error counter,
    temperature sensors, readback, ones counts) still run every tick with the monitor input interpolated linearly
    between the substep end values. Stated approximations; tested against the fixed policy (section 5).
    Review additions (the rules above only look backward): (a) a scenario event at k (a held signal changed, or
    an FOH signal's slope changed by more than half, since the previous step) forces a 1-tick step; (b) h at most
    doubles per step (1, 2, 4, ... after any forced tick; the first step after a reset is 1 tick); (c) substeps
    never straddle a BER window, a bias-DAC noise redraw when its period is > 1 tick, or a record window of a record
    that still records. Without (a) the first substep after a +1 dB laser step was 64 ticks and a 10 kHz
    controller's monitor samples showed a linear ramp 43138 -> 44505 codes instead of the jump to 45478 and its
    relaxation. With a per-tick bias-noise redraw (g26: 0.5 codes rms every tick) the first tick's draw is still
    held over the substep (about 0.02 pm rms of resonance; a stated approximation). The auto policy runs two
    Gauss-Seidel sweeps of the bank: the worst-case bound of the explicit cross-ring optical gain is 0.18 at 64 us
    in g26 (0.003 at 1 us; 07 P0-1 asks < 0.05 for the lagged form), the second sweep leaves ~g^2.
11. **Python controller path** needs edges shared by every world (no `tick_jitter_s`); the compiled path supports
    jitter (edges per world from `timing.edge_jitter`).
12. **fault.controller_reset**: compiled controllers get their post-init state back (`c_si0`, `c_sf0`) and the held
    command from the registers; Python controllers get `reset(spec)` from the runner.
13. The fast_kernel needs a CompiledController (it shares the controller's state arrays); its offsets are added to
    the latched registers before the DAC FIFOs and clipped to the channel range, and held between calls.
14. Open-loop plants still release their sensors at the controller edges (the FIFOs are sized for that period);
    `sample_sensors()` returns the last release.
15. **Quasi-static**: per (ambient, bias) the heater-up order (the first order a heater sweep from code_min
    reaches), oracle lock point over stable points, R1 temperature target, then one linear solve of the DC thermal
    map for every heater at once, iterated (loss, coupling and phases at the solved temperatures); the code from the
    driver law with the TCR at the static heater temperature (bisection). Single-pass bus, ones density 0.5,
    transition density of the config. Wrong-line risk = foreign lines (any order) crossed between code_min and the
    lock; FSR-hop risk = the lock needs negative heater power.
16. **Implicit self-heating step is mirrored in the kernel.** `kernel._implicit_step_eng` and `_solve_ring_eng` are
    line-for-line copies of `mrm.plant.selfheat.implicit_step_b` and its ring solver with the `pabs_fn` /
    `commit_fn` arguments replaced by direct calls to `eng_pabs` / `eng_commit`. Passing the adapter as a
    first-class function value made numba embed the dispatcher address ("dynamic globals") and refuse `cache=True`
    for the whole kernel, so every process recompiled the plant (~7 min). Any change to selfheat must be mirrored;
    `test_kernel_parity` (the reference runs `selfheat.implicit_step` with `optics.selfheat_pabs/commit`) fails bit
    for bit if the two drift. `test_kernel_is_disk_cacheable` guards the cache. (Performance round 1 kept the
    mirror: the lag update in two passes went into both, docs/handoff/perf.md P6; since P8 the kernel runs the
    same arithmetic from tables built once per block (lag indices by kind, per-lag gains, the implicit gain per
    observer), bit-identical to `implicit_step_b`, which the reference still runs. A factory that binds the adapter
    at compile time as a closure would remove the copy, but numba's cache index does not distinguish closures of
    one factory, so two adapters could share a cache entry; not done.)
17. **Optics rows lag one tick in temperature.** Loss, coupling, level phases and the junction shift of the
    self-heating rows are prepared from the ring (and heater) temperature at the start of the interval; only the
    thermo-optic shift is inside the implicit solve. The explicit part is ~1 % of dP_abs/dT; near a fold it changes
    the unstable growth rate by ~7 %, which `test_selfheat_divergence_rate` includes (`k_rows`).
18. **Physics-test settings** (test_engine_physics `CLEAN`): besides `ring.thermal.tail_share_frac=0`, the 3 ms 5 %
    package tail lives in `uncertainty.thermal.package_tail_share_frac` (nominal 0.05) and must be zeroed too for
    "steady after 200 us"; RIN is switched off with `laser.rin_db_per_hz=null` (-200 is refused by the key's own
    range). Test settings only.
19. **07 P2-1 literal bounds do not hold** at +9 dBm, Q_L 4200, beta_TPA 2 cm/GW (plant ring, build-up 10.1,
    I = 0.086 GW/cm^2 at A_eff 0.094 um^2): TPA loss 0.19-0.74 dB/cm (07: < 0.2), TPA-carrier shift 17-55 pm at
    1 ns and 2.6-8.1 pm at 0.1 ns (07: < 5 and < 0.5 pm), Kerr 1.9 pm at n2 6.7e-18 m^2/W (07: < 1 pm). 03 section
    7.2 gives the same magnitudes (9-28 pm at 8 mW, 1 ns). The exclusion itself still holds: TPA loss < 1 % of
    alpha_i (109 dB/cm), carrier shift <= 1.1 % of the 5 nm thermal shift at 1 ns and <= 1.2 pm at 10 ps
    (sweep-out), Kerr < 0.1 % of thermal. `test_nonlinear_bounds` asserts these; `test_nonlinear_bounds_07_literal`
    is `xfail(strict=True)` and documents the literal numbers. PARAMETERS.md lines 487-488 repeat 07's numbers.
20. **Quasi-static lock temperature** is set by the line, not by the ambient (fixed laser grid): over 0-105 C the
    g26 lock temperature moves ~0.5 mK while the heater power falls; the survey test asserts both (bound 50 mK).
    The survey uses the plant's optics model (TCMT when fidelity.optics_model is tcmt, backscatter when on) and the
    exact detuning of item 6.
21. **Lock reference** (review; docs/handoff/controllers_b.md request): `MetricsSpec.lock_reference` 'line'
    (default, 07's literal detuning; unchanged numbers), 'oracle' (the oracle's best stable relOMA point,
    refreshed every oracle period) or 'target' (`lock_target_pm`, scalar or (B, N)). A side-of-fringe or OMA lock
    sits 0.3-0.5 FWHM from its line by design, so with 'line' every such controller reads "never locked". No schema
    key yet (`metrics.lock_reference` is read when present): params owner, please add it.
22. **Controller pipeline** (review): sized with the closest edge spacing, period - tick_jitter (it used the
    nominal period: with J = 5, latency 9, 65-71 of 200 commands were silently dropped, counted only in
    diagnostics). `Plant.advance` grows the pipeline when commands are issued faster than the controller period
    (it dropped 3 of 4 commands at latency 3, uncounted).
23. **Plant.reset()** (review) restores a bound compiled controller to its post-init state (it overwrote the saved
    post-init copy with the end-of-episode state, so a second episode and fault.controller_reset used stale state),
    and sets the registers of datapath.bitstat_threshold, monitor.pga_select and laser.tune to their channel init
    values (only tx.assignment was set).
24. **PRBS pattern** (review): `Plant._build` overwrote the PRBS period with the controller period (variable
    shadowing), so `fidelity.pattern_noise: prbs<k>` ran a wrong pattern (ring temperatures 6 K off the reference engine).

## 4. Performance (07 P0-9)

Two performance rounds (docs/handoff/perf.md: change logs P1-P13 and R2-1 ... R2-16, error bounds, tests, raw
numbers). Round 1 took the B = 1 fixed tick from 54-55 us to 15.5 us on AC power (x3.5), with results equal to
commit 8646691 up to floating-point rounding (largest scenario difference 2.2e-6 pm of detuning; no ADC, heater or
bias code, lock state or line changed; the controller scorecard agrees to 2.6e-6 pm). Round 2 took it to 9.1 us
(x1.7) with every change bit-identical: every regression scenario, test and controller-scorecard number is the same
as at 6b598f7 (perf.md section 8). **07 P0-9's fixed-tick budget (<= 10 us per tick, >= 0.1 simulated s per wall s)
is met** on this machine, without approximation. Measured on the 8-core M2 on AC power, Low Power Mode off, load
average in parentheses (`tools/perf_ab.py`: alternating runs, median of 5 x 7 x 3000 ticks; `pytest tests/perf -m
slow -s`):

| case | 8646691 | round 1 (6b598f7) | round 2 | sim-s / wall-s (round 2) | notes |
|---|---|---|---|---|---|
| B = 1, fixed | 54.3-55.4 us (2.5-3.1) | 15.45-16.0 us (1.9-3.7) | **9.06-9.29 us** (3.0-3.7) | 0.108-0.110 | 07 target >= 0.1 (<= 10 us per tick): **met** |
| B = 1, fixed, settled lock codes (b1lock) | 55.1 us (2.5-3.1) | 15.46 us (3.0-3.7) | **9.42 us** (3.0-3.7) | 0.106 | |
| B = 1, auto, 100 kHz controller | 29.2 us (2.5) | 8.4-9.1 us (1.9-2.8) | 4.8-5.1 us (2.8-3.2) | 0.20-0.21 | 0.42 substeps per tick |
| B = 1, auto, 10 kHz controller | 12.2 us (2.5) | 4.5-4.6 us (1.9-2.8) | 2.9-3.0 us (2.8-3.2) | 0.34-0.35 | 0.16 substeps per tick; 07 target >= 20: not met |
| B = 64, fixed, 8 threads | 11.1-12.1 us (2.5-3.1) | 3.3-3.7 us (1.9-3.7) | 2.0-2.3 us (2.8-3.7) | 0.0068-0.0079 (bank) | 64 x 1 s Monte Carlo 2.1-2.5 min: 07 target (5 min) met |
| B = 64, auto | 10.1 us (2.5) | 3.2-3.5 us (1.9-2.8) | 1.9 us (2.8) | 0.0082 | |

The margin to 10 us is ~0.6-0.9 us at load 3-4: other busy processes on the machine can push a B = 1 fixed run
above it. The earlier figures of this section (94 us at load 8-10, 133-143 us at load 15-22) were taken with other
agents running and on battery or Low Power Mode at times. Battery or Low Power Mode makes every number ~1.7x slower:
compare only runs under the same power state (perf_bench records it).

Where the time goes at B = 1 (fixed, settled lock codes, ~9.4 us; tools/perf_tick_prof.py, perf.md 8.6): the
self-heating optics 4.1 us (Newton solves 1.97: 22.5 ring evaluations of 8 lines x 2 NRZ levels per tick at ~1.82
iterations per ring and sh_tol_k = 1e-7 K; the substep caches 0.97; the solver's bounds 0.79; commit 0.41), the
thermal ROM 1.4 us (970 lags with the stress-site source skipped while it is cold, optical lags, heater TCR split),
sensors 1.4 us (mostly the counter-based normal draws of the 14 flicker poles per ring at every ADC sample), the
phase model 0.6 us, Truth 0.4 us, the bias and heater DACs, laser and sensor inputs 0.9 us, the controller edge,
scenario, data, metrics and oracle 0.4 us. ~48 normal draws per tick (Philox + Box-Muller, 32-34 ns each) are
~1.6 us of it.

What round 2 changed (perf.md 8.2): unsigned indices in the hot loops (numba adds a negative-index check to every
signed index it cannot prove non-negative), a light self-heating commit at every substep but the last of a block,
the tick loop and its leaves compiled without the numba runtime (`_world_loop`, `_nrt=False`: no reference-count
updates, ~500 per tick, of the arrays passed to calls), the controller edge, auto policy, oracle and fast kernel
on one world's arrays instead of the KParams/KState tuples, thermal coefficient tables, the Truth Newton iterations
of all rings in lockstep, and inlining of the Airy evaluation and of the per-tick leaves. The round-1 review items are fixed:
`ReferenceEngine(airy=False)` checks the engine against an independent optics implementation, and a degenerate ring
row raises on both optics paths.

Options that trade accuracy for speed (perf.md 8.5): `sh_tol_k=1e-6` gives 8.8 us (b1); `optics_far_hold=True` (the
far-line hold, P5; default off) is now slower than the exact engine (10.1 us) and changes closed-loop trajectories
(up to 0.35 pm, 17 monitor codes in the regression scenarios): there is no reason to turn it on. perf.md 8.7 lists
what is left (each exact lever ~0.05 us; the levers that change results: cached solver bounds or a Newton predictor,
cheaper noise draws in mrm/core/rng.py, a smaller thermal ROM).

Levers not taken (rounds 1 and 2): CW optics when the data is idle (rounding-level, only helps idle scenarios, not
the benchmark), lag merging per (source, tau) (~0.3 us, changes the thermal state representation of the reference
too), evaluating only near lines in the Newton iterations (an approximation; the commit still needs every line), a
reduced-order monitor noise model (changes every noise realization).

First use in a fresh environment compiles the kernel (1.7 min and 2.25 GB peak RSS, measured after round 2 at load
2-4 with an empty numba cache; ~2-5 min after round 1, 5-8 min was measured under load): run it through
`tools/memguard.py --heavy`. Afterwards it is loaded from numba's disk cache (a fresh process runs a short
benchmark in 1.3 s, imports included). Any edit to `mrm/sim/kernel.py` recompiles it; an edit to a leaf module it
calls does NOT (numba keys the cache on the function's own file): run `tools/clear_numba_cache.py` after every
edit of njit code (docs/ENGINE.md section 1).

## 5. Tests

All run with a warm numba cache: `pytest tests/verification/test_engine_timing.py test_metrics.py
test_kernel_parity.py test_engine_physics.py` -> after the review 48 passed, 1 xfailed (strict, documented) in 138 s
at load ~20 (66 s at load ~12); every test under 25 s at that load, no slow marker needed. tests/perf: 1 passed,
3 xfailed (the 07 targets are not met on this machine). Other owners' engine tests (test_ctrl_a/b/c_engine, slow
included): 8 passed. Whole suite without `slow`: 1343 passed, 10 failed, none in engine code or engine tests
(closure presets eta / G25 notch / corner pins, VAL-8 thermal replication, replicas other-lines, PARAMETERS.md
current; the failing tests do not call the engine; probably the extraction agents' changing artefacts, not investigated).
test_replicas_static::test_bank_lock_matches_the_engine_locked_steady_state (an independent static bank solve against
`solve_locked_steady_state`) passes.

- `test_kernel_parity.py`: kernel vs the NumPy reference, bit for bit on every Truth field, every obs vector and the
  metrics: open loop, a Python controller (slow path, latency 2) and a compiled controller (kernel path, clipping);
  the same under NUMBA_DISABLE_JIT=1 in a subprocess (passes in 4 s); swing capacitance vs electrical (1e-12);
  the compiled chunk has no dynamic globals (cacheable). Review: `test_parity_fidelity_variants` (PRBS7 pattern,
  TCMT with backscatter, self-heating off, lock_reference='oracle'; the first two failed before the review) and
  the lock-error metrics in every comparison.
- `test_engine_timing.py`: test_loop_delay_impulse (latency 0 and 3), test_latency_invariant_to_rate (10-200 kHz),
  test_fifo_depth (ADC latency 26 us > period), fast_kernel vs a 1 MHz Python step (bit-identical I/Q), two configs
  in one process (no frozen globals). Review: test_pipeline_with_edge_jitter_drops_nothing,
  test_advance_keeps_every_command, test_reset_repeats_an_episode_with_a_bound_compiled_controller,
  test_run_creates_registered_controllers_with_params (each failed before its fix).
- `test_metrics.py`: accumulators vs full-trace metrics (synthetic and a 10 ms plant run), the 5 us spike in a 1 s
  run decimated to 1 ms (excursion_count == 1; the envelope keeps it, the mean hides it), reducers, BER windows.
  Review: test_lock_reference_moves_the_lock_metrics.
- `test_engine_physics.py` (one ring and line, CLEAN settings of section 3 item 18):
  - P0-1 (a): sweeps of +-2500 codes around the lock code; the up-jump sits 0.07 % of the HWHM and the down-jump
    1.6 % from the exact ring's static folds (y_o coordinate); at -10 dBm swept across the line nothing jumps.
  - P0-1 (b): dt 1 us vs 10 ns on a slow ramp < 1 pm RMS; +9 dBm steady detuning dt vs dt/8 < 1 % FWHM.
  - P0-1 (c): no tick-to-tick reversal on the stable branch at +9 dBm where an explicit coupling multiplier is
    < -1 (asserted); at the unstable middle equilibrium (Z k = 1.2-2) a 10 uK perturbation grows at the
    linearized implicit rate within 2 % (dt 1 us and 0.1 us) and within 5 % of the continuous rate at 0.1 us.
  - P0-1 (d): power step: the plant lands within 0.2 % of the exact static solution at +10 % and +1 %; the 07
    first-order formula is within 0.7 % at +1 % and 7.6 % low at +10 % (its error is O(f); 07 asked 5 % at +10 %,
    which the formula itself cannot meet).
  - P1-2: neighbour step vs the dark counterfactual: thermal system from recorded sources within 0.1 %; the step
    equals the DC cross gains x dP_abs within 1 %, > 90 % from the cross path.
  - P1-7 bias step (instant junction shift within 1 % of the junction table, then the blue self-heating relaxation
    within 5 % of s Z dP); P1-8 traffic step within 5 % of s(Z_j P_j + Z_d P_d)/(1 - G P') after 45 ms, and halving the
    transition density halves P_j; P1-9 monitor identity (1e-9) and dark current vs heater; P1-13 monitor noise
    sigma dt vs dt/2 within 5 %, batch and thread invariance (bit-identical); P2-1 bounds (section 3 item 19);
    quasi-static open-loop consistency and the g26 survey; auto vs fixed within 0.3 pm with < half the substeps.
  - Review: test_truth_detuning_is_the_phase_model_resonance (Truth vs an independent root solve of the phase
    model, 1e-3 pm; failed by 1.2 % before), test_auto_substeps_resolve_disturbance_steps (a laser-power and a
    traffic-enable step: the released monitor samples equal the fixed policy's within 10 codes; failed by ~2300
    codes before), test_truth_fed_oracle_runs_on_the_engine (the oracle on the engine; its target, the engine's
    oracle lock reference and the quasi-static optimum within 1.5 pm), test_metrics_invariant_to_dt (07 4.2:
    metrics within 1 % between dt = 1 us and 0.25 us, self-heating on and off; max |detuning| excluded, see the
    test). The physics-test helpers map detuning to phase exactly (`_phase_at`).

## 6. Requests to other owners

- **plant_electrical**: `_integ_lin` costs ~6 us per ring (two `searchsorted` per 10 mV segment); the engine uses
  `kernel.swing_capacitance_row` (cumulative table). Please make `junction_swing_capacitance_row` use the node
  values directly (or adopt the cumulative form) so both agree by construction.
- **phase owner (integrator)**: `ring_optics_row` recomputes K^2 at every line every tick (2 x N x L exp); a cache
  keyed on the line wavelengths would save ~4 us per tick at N = L = 8.
- **plant_thermal**: the shared-lag compression (x_m = r_m z_(src, tau)) is still the lever for the FEM ROM
  (11 550 lags); with the interim model (945 lags) the thermal pass is ~2.5 us and not the bottleneck.
- **core_control_api / validation**: a truth-fed controller (`truth_fed = True`, the oracle) is called by the runner
  as `step(obs, truth=Truth)` on the Python path (mrm.sim.runner._run_python). Since the review `Truth` carries `x`,
  `heater_applied` and per-world `conditions` (the `TruthState` duck type of mrm.validation.oracle): `Oracle(pp,
  cfg)` runs through `run(pp, oracle, cfg=cfg, ...)` (g26 nominal, warm start: 102.9-104.5 pm against its targets
  103.4-104.9 pm after 3 ms). The reference-tap channels have no plant source in `mrm.plant.io` yet (no
  ChannelSpec). Note for controllers_c: the engine's relOMA (and its oracle) is OMA at the ring's OWN through port
  / P_in (D9 "1 mW into the MRM"); mrm.validation.oracle uses the bus-output OMA of the line / P_in. They differ by
  the downstream rings' off-resonance transmission (the oracle target detunings differ by ~1.5 pm in g26).
- **integrator (ENGINE_BRIEF section 3 step 6) / phase and optics owners**: `dphi_dlamres = 2 pi c t_rt / lam^2`
  is the slope of the UNshifted cold curve; with mrm.plant.phase's rigid translation the slope is
  2 pi c t_rt / (lam - shift)^2 (the phase docstring says so). `optics.dphi_dlambda_res`, `pack_selfheat_args`,
  `test_plant_wiring.ReferencePlant` and the controller harnesses (test_ctrl_a_harness, test_ctrl_b_plant) still
  use the line wavelength: their in-tick self-heating phase is 1-1.2 % off after a 5-8 nm heater shift (their
  Truth, lam_res + shift - lam_line, is exact). The engine uses `kernel.dphi_dlamres`.
- **params owner (schema)**: add `metrics.lock_reference` (line | oracle | target) and, if wanted,
  `metrics.lock_target_pm`; MetricsSpec.from_config already reads the first.
- **plant selfheat owner**: `kernel._implicit_step_eng` / `_solve_ring_eng` mirror `selfheat.implicit_step_b`
  (section 3 item 16). Please either keep them in step when changing the solver (the parity test will fail
  otherwise) or offer a factory that binds pabs/commit at compile time so the kernel can call it without dynamic
  globals.
- **materials / PARAMETERS.md owner**: lines 487-488 state 07 P2-1's TPA bound (< 0.2 dB/cm) and "< 5 pm at 1 ns";
  with the plant ring at +9 dBm, Q 4200, beta 2 cm/GW they are 0.74 dB/cm and 17-55 pm (section 3 item 19). The
  exclusion stands on "< 1 % of alpha_i" and "~1 % of the thermal shift"; please reword the reasons.
- **integrator (pyproject)**: register a `perf` marker; tests/perf uses `slow` meanwhile.
- **next engine agent**: experiments (sweep, mc, compare, robust, acceptance), manifests, CLI, gym env, bridge,
  P1-20 telemetry metrics (missed loss of lock, false alarms), the 07 P0-10 compression convergence runs, and
  `mrm run --dry-run` estimates (use the measured us per world-tick). Still open from other owners' handoffs:
  (a) a warm start from the thermal steady state of the warm-start codes (controllers_a 3, controllers_c 3b; the
  engine starts the lags cold, so `heater_code` + `branch_high` needs a `start_delay_s`): one implicit step with
  h -> infinity (DC coefficients) at held inputs would do it; (b) `ctrl.load_factory_cal(table)` when the
  scenario's ResetSpec says factory_cal (controllers_b); (c) `run_lock_request(LockRunRequest)` for
  mrm.validation.replicas (replicas handoff section 6).

## 7. Adversarial review (2026-10-09): what was found and fixed

Every item below was reproduced first (a probe or a test that failed), then fixed, and the test is now in the suite.

| # | severity | where | defect | fix |
|---|---|---|---|---|
| 1 | major | kernel `_prepare_interval`, `truth_row`, `_oracle`; kernel_ref; runner `solve_locked_steady_state` | d phi / d lambda_res used the line wavelength (2 pi c t_rt / lambda^2) instead of the shifted coordinate of the phase model: every Truth detuning 1.2 % too large after a 7.6 nm heater shift (-3069 vs -3040 pm; 3.6 pm at +300 pm), same factor in the in-tick thermo-optic phase, the oracle scan and the survey | `dphi_dlamres` at x = lambda - shift; the declared line's detuning exact (`resonance_detuning`, Newton on the cold curve) |
| 2 | major | kernel `_auto_h` | the 07 P0-10 auto rules look only backward: the first substep after a laser, traffic, bias or fault step was up to 64 ticks and the controller's monitor samples showed a linear ramp instead of the step (2300 codes wrong) | forced tick at scenario events (held change, FOH kink), step doubling limit, BER-window, bias-noise-period and finished-record handling |
| 3 | major | plant `_build` | PRBS period overwritten by the controller period (shadowed variable): `pattern_noise: prbs<k>` ran a wrong pattern | renamed; parity test variant |
| 4 | major | plant pipeline capacity | sized with the nominal period: with tick jitter 5 and latency 9, 65-71 of 200 commands silently dropped | capacity from period - jitter |
| 5 | major | `Plant.advance` | commands issued faster than the controller period with a latency were dropped (3 of 4), not even counted | the pipeline grows |
| 6 | minor | `Plant.reset` | overwrote the compiled controller's saved post-init state with the end state (second episode and fault.controller_reset wrong); non-DAC registers not set to channel init | restore from the post-init copy; registers from init |
| 7 | minor | runner `_make_controller` | `registry.create(name, **params)`, but create takes a mapping: any `controller_params` raised TypeError | `create(name, params)` |
| 8 | minor | kernel_ref oracle | exact ring even when the plant runs TCMT (parity failed under `optics_model: tcmt`) | the plant's model |
| 9 | minor | solve_locked_steady_state | exact ring under TCMT; optimum quantized to the 3 pm grid with P_abs at the grid point | plant's model; parabola-refined optimum, ring re-evaluated there |
| 10 | feature | metrics | lock state about the LINE only: every side-of-fringe / OMA lock reads "never locked" (controllers_b request) | `lock_reference` line / oracle / target, lock-error metrics |
| 11 | feature | Truth | the Truth-fed oracle could not run on the engine (no lag states, no conditions; controllers_c request) | `Truth.x`, `heater_applied`, `conditions` |

Checked and found correct (no change): the event order and loop delay (first_response_tick at latency 0 and 3);
causality of the fast kernel (samples only after sample tick + ADC latency) and of the Python and compiled paths
(obs from the released FIFOs only; Truth reaches only `truth_fed` controllers and the registry refuses those); the
mirrored implicit solver (`_implicit_step_eng`, `_solve_ring_eng`) is token-for-token `selfheat.implicit_step_b` /
`solve_ring_temperature` with the adapter substituted; the commit reuse (`eng_commit`) only reuses an evaluation of
the same ring at the same temperature and input; every per-world kernel array has B rows under nominal, sampled
and LHS realization (no out-of-bounds reads without boundscheck); backscatter on, self-heating off and binomial
pattern noise are bit for bit with the reference; NUMBA_DISABLE_JIT runs the auto policy, the compiled path, jitter
and the fast kernel. The other owners' engine tests (test_ctrl_a/b/c_engine, slow included) pass with the fixes.

Controllers on the engine (an engine check, not a controller review): `Oracle` (Truth-fed) holds g26 nominal within
1.5 pm of the engine's oracle reference. `PiLockCompiled` (warm start from the quasi-static codes with cold lags,
`start_delay_s` = 5 ms, 10 kHz; too short a delay: in the static scenario this setup ends unlocked at -808 pm, the
controllers_a handoff uses 30 ms) under A1 goes into a growing heater oscillation on the victim ring at ~16 ms
(codes 36285 -> 34916-37521 within 1.2 ms); auto and fixed agree within 0.5 pm until then and then diverge, as an
unstable loop amplifies any difference (static scenario: auto = fixed within 0.15 pm over 25 ms). The controller
owners' handoffs document their A1 results (and the monitor-PD dark current of the interim site model).

