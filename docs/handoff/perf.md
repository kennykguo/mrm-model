# Handoff: engine performance (rounds 1 and 2)

Owner files: `mrm/sim/kernel.py`, `mrm/sim/kernel_ref.py`, `mrm/plant/plant.py` and the leaf kernels the tick calls
(`mrm/plant/{selfheat,optics,phase,thermal,electrical,sensing,datapath,laser}.py`, `mrm/sim/metrics.py`). Starting
point: commit 8646691. Goal: 07 P0-9, >= 0.1 simulated s per wall s at a fixed 1 us tick for B = 1, N = L = 8, g26
(10 us per tick). Rule: accuracy first: every change is EXACT (bit-identical, or algebraically identical up to
floating-point rounding) or an APPROXIMATION with a derived bound, a test of the bound and a fidelity switch.

Round 1 ran in two sessions: P1-P7 by the first optimizer (cut off during its final validation; it ran on battery in
Low Power Mode), P8-P13 and the validation and AC timings of section 5 by the second. Section 5 has its numbers.
Round 2 (section 8, R2-1 ... R2-16) started from 6b598f7 and met the 10 us target: B = 1 fixed 15.5 -> 9.1 us
per tick, bit-identical results. Section 8 has the current numbers.

## 1. Tools (all in `tools/`)

| tool | what it does |
|---|---|
| `clear_numba_cache.py` | deletes the numba cache files of the kernel and every leaf module it calls (numba keys a cached function on its own source file only, so an edited leaf would otherwise keep running stale code inside the cached kernel; docs/ENGINE.md section 1). Run after every edit of njit code, before any timing or comparison. |
| `perf_bench.py` | wall and CPU time (`time.process_time`) per world-tick, median of repeats, load average before and after, Newton iterations per ring and tick. Cases `b1` (B = 1, fixed, open loop, tests/perf setup: sampled world seeds 7, cold rings), `b1lock` (B = 1 nominal world settled at the quasi-static lock codes), `b64` (B = 64, 8 threads). |
| `perf_ab.py` | back-to-back A/B/A/B runs of `perf_bench.py` for two source trees (each with its own numba cache): the before/after numbers in this file. The baseline tree is `git archive 8646691`. |
| `perf_tick_prof.py` | in-context breakdown of the tick: re-compiles `kernel._world_block`, its tick loop `_world_loop` (since R2-11) and `_implicit_step_eng`, `_truth_a` from the kernel source, each with the kernel's own runtime flag, with a read of the arm64 virtual counter (`cntvct_el0`, 24 MHz) at every `# ---` comment line, so the sections follow the kernel source. arm64 only. The instrumented functions get an extra argument `acc`: they must not use a local of that name. The counter reads add ~0.1-0.6 us per tick. |
| `perf_profile.py` | cost of single leaf kernels in compiled loops (out of context). Round 2 fixed its implicit-step case (stale since P8). |
| `perf_scenarios.py` | the fixed accuracy-regression scenarios (section 2), traces + metrics + final Truth to `runs/<dir>/<scenario>.npz`. |
| `perf_compare.py` | largest difference per signal between two `perf_scenarios.py` runs. |
| `perf_summarize.py` | the largest difference per signal over all scenarios of a `perf_compare.py --json` file. |
| `scorecard_compare.py` | largest change per metric between two controller scorecards (test_controllers_on_engine.py). |
| `perf_scenarios_adv.py` | adversarial accuracy-regression scenarios (section 6): fast aggressor edges, moving and switched laser lines, both sides of the self-heating fold, traffic on/off with bias ripple, the auto policy, DAC/ADC faults with the PWM ripple and the er_db / il_db records, backscatter and TCMT, a sampled B = 8 batch under auto; every Truth field at the end. Same format as `perf_scenarios.py`. |

## 2. Baseline (commit 8646691)

Scenarios (`tools/perf_scenarios.py`; saved in `runs/perf_baseline/ref`, git-ignored; a second run `ref2` is
bit-identical, so the engine is deterministic): `steady_pi` (compiled PI lock from the settled warm start, 40 ms),
`a1_pi` (pi_lock_c under the A1 slide-10 aggressor, PD on the far side, one period 130.7 ms), `laser_step` (open
loop at the settled lock codes, +10 % laser power at 5 ms), `heater_sweep` (cold start on branch_low, heater ramp
to 1.15 x the lock code in 40 ms: every ring jumps across self-heating folds, up to 3.4 K in one tick),
`idle_traffic` (open loop, driver off until 10 ms, then traffic), `mc8_pi` (B = 8 sampled worlds, pi_lock_c, 10 ms).
The controller scorecard of 8646691 is saved as `runs/perf_baseline/controller_scorecard_8646691.json`.

Timing at the baseline (load 2.5-2.8, quiet machine): B = 1 fixed 54-55 us per tick (b1 and b1lock), B = 64 fixed
11.1 us per world-tick (8 threads). (docs/handoff/engine.md section 4 reports 94 us at load 8-10: the machine was
busier then.) Newton iterations per ring and tick 1.70-1.82 (2.7-2.8 ring evaluations).

## 3. Log of changes

| # | change | kind | saving (B = 1, us/tick) | error bound | test |
|---|---|---|---|---|---|
| P1 | self-heating evaluation with hoisted substep constants (section 4) | exact, bit-identical | 55 -> 40 (A/B, load 2.5-2.8) | none (bit-identical) | test_engine_adapter (bit for bit vs the optics adapter), test_kernel_parity, all scenarios bit-identical |
| P2 | tuple-free helpers on the hot path (declared line, scenario call), 2 reshapes instead of `selfheat_views` (29 views, 0.44 us), BER once per tick (error counter and Truth) | exact, bit-identical | ~4 (profile, load 4-5) | none | parity, scenarios bit-identical |
| P3 | the tick in one function (`_world_block`): no per-tick call takes the KParams/KState tuples; caches of the DAC transfer, 10^(dB/10) and the Truth FWHM keyed on their inputs | exact, bit-identical | ~15 (35 -> 20, profile, load 4-7); P1-P3 together 54.4 -> 19.3 (A/B, load 3.7-4.1) | none | parity, scenarios bit-identical |
| P4 | Airy path of the self-heating adapter moved into mrm.plant.optics and shared by the engine and the reference: cached cos/sin of the reference phase rotated by the small in-solve angle (Taylor polynomial, truncation < 3e-20 at abs(b) <= 0.05 rad), one reciprocal instead of two divisions, a = a_row and q = sqrt(a) instead of exp(log a) round trips, incremental update of the cached cos/sin from one substep to the next (re-derived with libm every 4096 updates) | exact up to rounding | ring evaluation 0.28 -> ~0.07 us (standalone); tick ~19.3 -> ~18 | P_abs 2e-13, slope 4e-12, outputs 2e-13 relative to the closed forms (the rounding of the absolute phase ~200 rad); cached cos/sin within 6e-15 of libm after 3 x 4096 updates of random phase steps; a constant step (a steady drift) repeats the same rounding, so it drifts systematically: 2.8e-13 measured, <= 4096 x 4 ulp = 1.8e-12 derived (<= 7e-10 relative in P_abs at resonance, ~5e-9 K; section 6) | test_engine_adapter (3 new tests), parity bit for bit (shared code), scenarios: largest differences 2e-6 pm, 3e-5 mK, no controller code changed (section 3.1) |
| P5 | far-line hold of the self-heating optics, `Plant(optics_far_hold=True)` / `run(..., optics_far_hold=True)`, **default off**: a line >= 8 half-widths from every resonance at both levels is held to first order in its phase and loss between refreshes (mrm.plant.optics FAR_*), in the evaluation, the commit, the bounds and the cache update | approximation (fidelity switch) | 31.3 -> 29.1 (A/B on battery, Low Power, load 7.8-9.1); on AC section 5 | P_abs <= 3e-9 W per ring (~1e-3 pm), outputs <= 1e-6 of P_in (derivation at FAR_U); measured 6e-11 W. But the closed loops amplify it: scenario differences up to 0.35 pm, 17 monitor codes, 1 heater code (section 5.4) | test_far_line_hold_error_is_within_its_bound, test_parity_with_far_line_hold |
| P6 | lag update of the implicit step in two passes (every lag with its known input, then the observer sums), in `selfheat.implicit_step_b` and its kernel mirror | exact, bit-identical | 31.3 -> 31.1 (battery, Low Power) | none | parity |
| P7 | rotation of the cached cos/sin by the lowest-degree Taylor polynomial whose truncation stays below 1e-17 (`optics._rot`: degree 2/3 for abs(b) <= 1e-4 rad, 4/5 for <= 3e-3, 8/9 up to 0.05; P4 used degree 8/9 throughout), and the per-(ring, level) memo of the loss factors (`optics._afac`). Reconstructed (section 4) | exact up to rounding | 31.6 -> 30.4 (battery, Low Power, runs/perf_baseline/ab_p6_p7.json) | truncation < 1e-17 per rotation | test_engine_adapter (closed forms, incremental cache) |
| P8 | thermal pass of the kernel from tables: lag indices by kind, the per-lag gain src_scale (c0 + c1), the implicit gain per observer and the heater split's gain per ring (built per block and substep power), the factor src_scale skipped when it is 1.0 | exact, bit-identical | P8-P10 together 17.5 -> 16.2 (A/B on AC, load 5.3-6.6) | none | parity, scenarios bit-identical to the resumed state |
| P9 | bus finalize: only the views it needs (not `_sh_core` / `_sh_outputs`, ~0.15 us), no division or square root for a zero reflection | exact, bit-identical | in P8-P10 | none | parity, scenarios |
| P10 | junction table: one bracket for the four columns of a voltage, found from the uniform spacing and confirmed (`electrical.grid_index`; was four binary searches of the 451-point grid per ring and level); phase model without heap allocations per tick (`phase.ring_optics_levels_ws`); the loss temperature factor (one exp) shared by both NRZ levels | exact, bit-identical | in P8-P10; optics-input section 2.7 -> 0.9 us (profile) | none | parity, test_electrical, scenarios |
| P11 | structure-of-arrays layout of the Airy caches (each field contiguous over level and line) and three-pass evaluation | exact, bit-identical | **reverted**: 16.1 -> 16.7 (slower) | none | (section 4) |
| P12 | Truth: `cold_coordinate` stops at an exact floating-point fixed point (bit-identical to the five iterations); er_db / il_db (two log10 per ring) only when a record reads them and at the last step of a block; the PD dark current computed once per tick for the monitor and Truth | exact, bit-identical | Truth section 1.0 -> 0.8 us (profile); A/B 16.03 -> 16.03 (within noise) | none | parity, test_sensing, scenarios |
| P13 | optical-lag pass with each observer's sum in a register (no store-to-load chain per lag); the eight heater splits in lockstep | exact, bit-identical | 16.0 -> 15.5 (A/B on AC, load 4.0-4.3) | none | parity, scenarios |
| P14 | numba's numpy error model (no zero test before each division) on the division-heavy per-tick leaves (phase model, cold_coordinate, Truth, sh_bounds, bus finalize, dark current, junction lookup) | exact for every run that does not divide by zero | **reverted**: 15.70 -> 15.55 (b1), 15.72 -> 15.85 (b1lock), i.e. no change; a division by zero would have turned into inf/NaN instead of an error | none | parity, scenarios bit-identical |

## 4. Notes on the changes

P1. One ring evaluation (exact ring, no backscatter, NRZ two levels, 8 lines) was ~0.9-2.1 us: per line and
level a log, four square roots, two exponentials, expm1, cos, sin and five divisions, of which only cos and sin
depend on the ring temperature. `kernel.eng_precompute` evaluates the rest once per substep (the loss and the
coupler amplitudes do not change within the implicit solve), with the same floating-point expressions in the same
order as `mrm.plant.optics._airy_line` / `_seg_factors` / `ring_rows_nb`, so P_abs, its slope, every committed
output and therefore every Newton iterate are bit-identical. The coupler amplitudes (sqrt(1 - K^2)) and K^2 itself
(`phase.coupling_row`, 128 exponentials) are recomputed only when the line wavelengths change. TCMT and backscatter
rings keep the general path. The evaluation is now ~0.23 us.

P3. A numba function that receives the KParams/KState namedtuples (~270 arrays) and calls any other compiled
function increments and decrements the reference count of every array in them on entry and exit (numba's
reference-count pruning cannot remove the pairs around a call; measured 0.2-0.4 us per call, in the LLVM IR 210
increfs in the entry block of truth_row). The tick made ~15 such calls. The per-tick code now lives in
`_world_block` (which takes the tuples once per block of ticks); the controller edge, `_auto_h` and the oracle
still take them (once per controller period / auto step / oracle period).

P5 (far-line hold, approximation, default off). Derivation of the bound in the mrm.plant.optics module comment at
`FAR_U`; tests in test_engine_adapter.py (P_abs error against the exact Airy path over 2000 substeps of a random
phase and loss walk, two step sizes) and test_kernel_parity.py (kernel and reference agree bit for bit with the hold
on, since the hold lives in the shared adapter). It stays off by default because its effect on the closed loops is
larger than the bound on P_abs suggests: a 1e-8-relative change of P_abs moves a monitor sample across an ADC code
threshold now and then, and the loops then follow another trajectory (section 5.4: up to 0.35 pm in heater_sweep,
17 monitor codes, 1 heater code; one fast-suite test, test_ctrl_b_robust swing loss, ended outside its p-p bound with
the hold on). The rounding-level changes of this round (P4, P6, P7) never flipped a code in any scenario.

P6 and P7 (reconstructed). The first optimizer's snapshots were lost with its scratch directory. P6 is the two-pass
lag update (CHANGELOG: bit-identical). Between runs/perf_p4 (= runs/perf_p5_off, bit for bit) and runs/perf_p6 the
scenario traces differ at rounding level (largest 1.6e-7 pm, 2.4e-6 mK), and between runs/perf_p6 and
runs/perf_final again (2.6e-7 pm), so rounding-level changes landed in both steps. The code features not described
by P1-P6 are the adaptive-degree rotation polynomial of `optics._rot` (P4's log describes one degree-8/9 polynomial
with truncation < 3e-20) and the per-(ring, level) memo of `optics._afac`; they are logged as P7, the step measured
by runs/perf_baseline/ab_p6_p7.json. Both are covered by test_engine_adapter.py (closed forms to 1e-11 relative;
the incremental cos/sin cache within 2e-14 of libm over 3 x 4096 updates).

P8. The kernel's lag pass branched on the lag kind at every one of the 945 lags; it now runs three loops over index
lists (exogenous, cross-ring, a x only). The per-lag gain src_scale (c0 + c1), the implicit gain of each observer and
the heater split's gain are the same expressions in the same order as `selfheat.implicit_step_b` and
`thermal.observer_split_b`, built once per block and substep power (Plant.settle swaps the coefficient tables between
blocks, so they are not cached in KParams). The reference still runs the original functions; parity is bit for bit.

P10. `electrical.grid_index` returns np.searchsorted(grid, x) - 1 by guessing the bracket from the mean spacing and
confirming grid[j] < x <= grid[j + 1] (the unique bracket on a non-decreasing grid), with the binary search as the
fallback, so `_interp_lin` and `junction_eval_row` return the same values; `junction_eval_row` finds one bracket for
its four columns. The phase model took six heap allocations per tick (np.empty in `ring_optics_levels_row` and
`resonance_shift_row`); the kernel now passes a scratch block. Profile: the optics-input section 2.7 -> 0.9 us.

P11 (reverted). Each field of the Airy cache contiguous over (level, line), with the evaluation split into a rotation
pass, an entry pass and the sums (bit-identical). Standalone the split evaluation was 27 % faster when its field
views were made outside the loop, but numba then addresses every field through a runtime stride (or builds the views
per call), and in the tick the cache update got 25 % slower and the evaluation 5 % slower: 16.1 -> 16.7 us (A/B).

P12. The Newton update of `cold_coordinate` is a function of x alone, so once it returns x unchanged every further
iteration returns x again: stopping there is bit-identical to the five fixed iterations. er_db / il_db are read by no
kernel computation (metrics use reloma; the auto policy lam_res), so `_truth_a` writes them only when a record reads
them and at the last step of a block (k + h = the block end, or the next controller edge on the Python path, where the
block returns), so Plant.truth() is always complete. The PD dark current (one exp per ring) is computed once per tick
and passed to `sensing.monitor_tick_row` (new optional argument) and Truth.

P13. The optical-lag pass added each lag's power step into t_obs[lag_obs[m]] in memory: a store-to-load round trip
per lag on one observer's chain. It now carries the observer's sum in a register while consecutive lags feed it (the
same additions in the same order). The heater split ran ring by ring (one dependent chain of ~80 additions each); the
eight sums now advance in lockstep, each in its own order.

## 5. Round 1 result (validation and timing on AC power)

### 5.1 Resumed state

The second session started from the first optimizer's working tree (P1-P7). Its validation, finished first: the fast
suite passes except the three VAL-8 thermal-replication tests (`tests/validation/test_thermal_replication.py`
VAL-8a/b/c), which fail identically at commit 8646691 (they read the thermal FEM artefacts, not the engine);
`test_kernel_parity.py` (11 tests, `test_parity_without_jit` = NUMBA_DISABLE_JIT=1 included) and
`test_engine_adapter.py` (8) pass; the regression scenarios rerun on it (`runs/perf_r1/ref`) are bit-identical to its
`runs/perf_final`, so the earlier accuracy comparison holds for it.

### 5.2 Timing on AC power

Low Power Mode off; `tools/perf_ab.py`: alternating runs in fresh processes, median of 5 rounds of the per-run median
of 7 x 3000 ticks.

| case | 8646691 | resumed (P1-P7) | final (P1-P13 without P11) | final / 8646691 |
|---|---|---|---|---|
| b1: B = 1, fixed, open loop, cold rings | 54.3 us (cpu 54.3) | 17.6 us (cpu 17.6) | **15.5 us** (cpu 15.5) | x3.50 |
| b1lock: B = 1, fixed, settled lock codes | 55.1 us (cpu 55.0) | 17.6 us (cpu 17.6) | **15.5 us** (cpu 15.5) | x3.55 |
| b64: B = 64, fixed, 8 threads (per world-tick) | 11.1 us (cpu/thread 9.7) | 4.1 us (cpu/thread 3.3) | **3.7 us** (cpu/thread 3.0) | x3.0 |

Load average 2.5-3.1 during these pairs (`runs/perf_r1/ab_base_final_ac_quiet.json`, `ab_r1_final_ac_quiet.json`); a
second session at load 3.4-4.7 gave 54.4 / 17.7 / 15.6 (`ab_base_final_ac.json`, `ab_r1_final_ac.json`). The CPU time
per tick (time.process_time, single thread for B = 1) equals the wall time within 0.1 %, so these B = 1 numbers are
CPU-bound and not inflated by the load. The first optimizer's battery / Low Power Mode figures (92.8 -> 30.3 us) are
1.7-1.9x slower than AC and are not mixed with these. Newton iterations per ring and tick are unchanged (1.727 b1,
1.817 b1lock, 2.008 b64).

tests/perf (`pytest tests/perf -m slow -s -k 'report_table or auto_step'`, load 1.9-2.5, `runs/perf_r1/perf_report_tables.log`):

| case | 8646691 | final |
|---|---|---|
| B = 1, fixed | 55.4 us per tick (0.018 sim-s/wall-s) | 15.6-16.0 us (0.063-0.064) |
| B = 1, auto, 100 kHz controller (0.42 substeps per tick) | 29.2 us (0.034) | 9.0-9.1 us (0.110) |
| B = 1, auto, 10 kHz controller (0.16 substeps per tick) | 12.2 us (0.082) | 4.5 us (0.221) |
| B = 64, fixed, 8 threads | 12.1 us per world-tick | 3.3-3.5 us |
| B = 64, auto | 10.1 us | 3.2-3.5 us |

**07 P0-9 target (>= 0.1 simulated s per wall s at the fixed 1 us tick, i.e. <= 10 us per tick): not met.** Best exact
result 15.5 us per tick (0.065 sim-s/wall-s). With the auto policy the 0.1 sim-s/wall-s mark is passed (0.11 with the
g26 100 kHz controller, 0.22 with a 10 kHz controller); 07's auto target (>= 20 sim-s/wall-s) is not. The 07 Monte
Carlo target (64 seeds x 1 s in 5 min) is met at this load: 3.3-3.7 us per world-tick on 8 threads is 3.5-3.9 min
(8646691: 12-13 min).

Bounded approximations and numerics options (A/B against the final engine, load 4.1-4.6):

| option | b1 | b1lock | effect on results |
|---|---|---|---|
| none (final, exact) | 15.6-15.9 us | 15.4-15.5 us | rounding-level vs 8646691 (5.3) |
| `optics_far_hold=True` (P5) | 14.6 us (x1.09) | 14.8 us (x1.05) | closed-loop divergence up to 0.35 pm, monitor codes up to 17 (5.4) |
| `sh_tol_k=1e-6` (existing option; default 1e-7 K) | 15.2 us (x1.03) | 15.3 us (x1.01) | each solve within 1e-6 K of its root (1.29 instead of 1.73 iterations) |

### 5.3 Accuracy against 8646691 (exact changes)

P8-P13 are bit-identical to the resumed state: every regression scenario of `tools/perf_scenarios.py` (6 scenarios,
every recorded trace, metric and final Truth) has zero differences against `runs/perf_r1/ref` after each step
(`runs/perf_p8`, `perf_p11`, `perf_p12`, `perf_p13`; the final tree, `runs/perf_final_r1`), and `test_kernel_parity`
stays bit for bit (also under NUMBA_DISABLE_JIT=1). Against commit 8646691 the whole round differs at rounding level only (P4 and P7 change the
floating-point path; the Newton iterates then converge to other points inside the 1e-7 K tolerance):
largest differences per signal over all scenarios (`runs/perf_p13/compare_vs_baseline.json`; tools/perf_summarize.py):

| signal | largest difference | relative to the signal's range | scenario |
|---|---|---|---|
| detuning, resonance | 2.2e-6 pm | 1.8e-8 | idle_traffic |
| ring temperature | 2.9e-5 mK | 7e-11 | idle_traffic |
| heater temperature | 1.3e-5 mK | 3e-11 | idle_traffic |
| PD temperature | 8.4e-9 mK | 3e-14 | idle_traffic |
| absorbed power | 1.5e-5 uW | 7e-9 | idle_traffic |
| drop power / monitor current | 2.8e-7 uW / 2.6e-4 nA | 7e-9 | idle_traffic |
| OMA, relOMA, BER | 1.5e-6 uW, 6.5e-10, 2.6e-10 | 3e-9 | idle_traffic, steady_pi |
| heater power | 3.6e-7 uW | 3e-11 | idle_traffic |
| monitor codes, heater codes, bias, line_on, every discrete metric | 0 | 0 | all |
| final Truth: detuning / T_ring / P_abs | 2e-10 pm / 1.6e-9 mK / 6e-10 uW | < 2e-12 | |

Controller scorecard (`pytest tests/validation/test_controllers_on_engine.py -m slow`, 19 passed, written to
`runs/perf_r1/scorecard/`; compared with the copy of 8646691's `docs/handoff/controller_scorecard.json` in
`runs/perf_baseline/controller_scorecard_8646691.json` by tools/scorecard_compare.py): every discrete metric (locked,
lines, excursion counts, settle iterations and convergence, t_lock) unchanged; largest change per metric: steady
rms_pm 3.0e-9 pm, max_pm 1.2e-8 pm, ref_pm 3.9e-10 pm, penalty_db 1.9e-10 dB, t_pd_max_c 2.3e-13 C; a1 max_pm 6.4e-7
pm, rms_pm 9.5e-10 pm; a1_pd_far max_pm 2.6e-6 pm, rms_pm 8.5e-10 pm; engine OMA penalty 1.9e-11 dB; settle
max_delta_k 5.1e-13 K (`runs/perf_r1/scorecard/compare_vs_8646691.txt`). It is identical, number for number, to the
first optimizer's scorecard run (`runs/perf_scorecard/final2`). (It ran on the P13 tree; the final tree differs from
it only by two renamed local variables and gives bit-identical scenarios.)

Fast suite on the final tree (`tools/memguard.py -- pytest -q -m "not slow"`, `runs/perf_r1/suite_final2.log`):
1472 passed, 73 skipped, 3 xfailed; the 3 VAL-8 failures of 5.1 only.

### 5.4 The far-line hold (P5) is not a default

Its bound (P_abs <= 3e-9 W per ring, ~1e-3 pm) is far below the ADC LSB and the noise floor, and its measured P_abs
error is 6e-11 W. But the regression scenarios with the hold on differ from the exact engine by up to 0.35 pm of
detuning (heater_sweep: the fold crossings move), 0.04 pm at the end of steady_pi, 17 monitor codes and 1 heater code
(`runs/perf_final_hold/compare_vs_final.json`): a 1e-8-relative change of P_abs moves an ADC sample across a code
threshold now and then and the closed loop then follows another, equally valid trajectory; one controller test
(test_ctrl_b_robust, swing loss) leaves its p-p bound. A model change whose effect a controller test can see is not
an acceptable default, so the hold stays an opt-in for Monte Carlo throughput. The rounding-level changes of this
round never flipped a code.

### 5.5 Where the time goes now (B = 1, b1lock, final; tools/perf_tick_prof.py, load 3.3, instrumented 16.1 us)

| part | us per tick | share |
|---|---|---|
| self-heating optics: Newton solves (22.5 ring evaluations of 8 lines x 2 levels per tick) | 2.7 | 17 % |
| self-heating optics: commit (outputs of each ring, bus finalize) | 1.5 | 10 % |
| self-heating optics: substep caches (cos/sin rotation and Airy constants, 128 entries) | 1.6 | 10 % |
| self-heating optics: bounds of the solve (128 entries, 2 sqrt + 3 divisions each) | 1.0 | 6 % |
| thermal: 945 lags (advance and observer sums), optical lags, heater TCR split | 1.7 + 0.3 + 0.7 | 17 % |
| sensors (monitor with 14 flicker poles per ring per ADC sample, error counter, temperature, readback) | 1.8 | 11 % |
| phase model (optics inputs), K^2 / slopes / swing capacitance | 0.9 + 0.25 | 7 % |
| Truth | 0.9 | 6 % |
| controller edge (sensor release, bit statistics: 16 binomial draws per edge) | 0.5 | 3 % |
| bias DAC (8 normal draws per tick), laser (RIN draws), heater DAC, data, scenario, metrics | 1.4 | 9 % |

The counter-based normal draws alone (mrm.core.rng: Philox4x32-10 + Box-Muller, 35 ns each; ~44 per tick for the
flicker, white and DAC noise and the laser RIN) take ~1.5 us.

### 5.6 What would be needed for 10 us

Every remaining part is a fixed amount of work per tick, spread over many sections; none is above 17 %. The exact
levers left are small (each <= ~0.5 us): the edge release on arrays instead of the tuples (~0.1), the phase model with
one division instead of two per (ring, line, level) (rounding-level, ~0.15), Philox with 32-bit arithmetic (mrm/core,
not an engine file), and so on. Two attempts that should have helped did not: the structure-of-arrays cache layout
(P11, slower in numba) and numba's numpy error model on the division-heavy leaves (no measurable change; not kept).
Reaching 10 us needs one of:

1. **Far lines nearly free, as an opt-in.** Seven of the eight lines of each ring are 5-27 half-widths from its
   resonance; they carry ~85 % of the optics work (caches, bounds, evaluations, outputs). A lean far-line hold
   (per-ring aggregates in the Newton evaluations, a second-order phase model so that refreshes are rare, stored
   output responses) could take the self-heating optics from 6.8 to ~2 us (~11 us per tick), but like P5 it changes
   closed-loop trajectories (5.4), so it can only be an opt-in for throughput.
2. **A compiled kernel outside numba's scalar model** for the per-line work (explicit SIMD over the 16 line-levels of
   a ring, no-alias arrays, no per-call array bookkeeping): the per-entry costs measured here (~12 ns per cache entry,
   ~5 ns per line-level evaluation, ~2 ns per thermal lag) are 2-4x above what the arithmetic needs.
3. **Cheaper noise draws** (a ziggurat or a sincos-based Box-Muller in mrm/core/rng.py): changes every noise
   realization, so every seeded result; a contract change for the core owner (07 P1-13), ~1 us.
4. **A smaller thermal ROM** (945 lags for 8 rings: 41 per ring observer, one per neighbour source; merging lags per
   (source, pole) saves ~0.3 us exactly up to rounding; dropping cross couplings is an approximation for the thermal
   owner).

Even 1 + 3 together land near 10 us, not clearly below it, on this machine at this load.

## 6. Independent review of round 1

Reviewer's re-run on the final tree (all raw files in `runs/review_r1/`, git-ignored). AC power, Low Power Mode off,
no other pytest of this project running; numba caches cleared with `tools/clear_numba_cache.py` before every
comparison and timing; the baseline is a fresh `git archive 8646691` with its own cache.

**Baseline.** The fresh 8646691 tree reproduces `runs/perf_baseline/ref` bit for bit (all six scenarios), and the
saved scorecard copy equals `git show 8646691:docs/handoff/controller_scorecard.json`.

**Timing** (`tools/perf_ab.py`, 5 alternating rounds, median of 7 x 3000 ticks per run, load 2.7-3.1;
`ab_base_fix.json`):

| case | 8646691 | final | wall (cpu) speed-up |
|---|---|---|---|
| b1 (B = 1, fixed, cold rings) | 53.6 us (cpu 53.6) | **15.45 us** (cpu 15.45) | x3.47 (x3.47) |
| b1lock (B = 1, fixed, settled lock codes) | 54.4 us (cpu 54.2) | **15.43 us** (cpu 15.43) | x3.53 (x3.51) |
| b64 (B = 64, 8 threads, per world-tick) | 11.1 us (cpu/thread 9.7) | **3.56 us** (cpu/thread 2.97) | x3.11 (x3.26) |

A first pair before the review fix (load 2.1-4.8, `ab_base_cur.json`) gave 53.7 / 54.4 / 11.0 -> 15.43 / 15.43 /
3.62. The review fix costs nothing measurable (optimizer's tree vs final: 15.40 vs 15.35 us b1, 15.47 vs 15.43 us
b1lock, `ab_opt_fix.json`). Options (`ab_fix_hold.json`, `ab_fix_tol6.json`, load 2.4-2.6): `optics_far_hold=True`
15.43 -> 14.63 us (b1), 15.42 -> 14.66 us (b1lock), no gain at B = 64 (3.60 -> 3.64); `sh_tol_k=1e-6` 15.37 -> 14.97
(b1), 15.44 -> 15.15 (b1lock). tests/perf (single short runs, load 3.8-4.2, `perf_tables.log`): B = 1 auto at 100 kHz
29.1-29.4 -> 9.2 us, at 10 kHz 12.2-13.8 -> 4.7-6.1 us, B = 64 fixed 10.9-13.7 -> 3.4-3.5 us, B = 64 auto 10.2-10.3 ->
3.6-4.2 us. The numbers of section 5 hold. 07 P0-9 (10 us at the fixed tick) is not met.

**Accuracy.** The final tree reproduces `runs/perf_final_r1` and `runs/perf_r1/ref` bit for bit, so section 5.3
holds as written (largest differences against 8646691: 2.2e-6 pm detuning, 2.9e-5 mK, 1.5e-5 uW absorbed, 2.6e-4 nA
monitor current; every code, line and discrete metric identical). New adversarial scenarios
(`tools/perf_scenarios_adv.py`, 11 cases: a 0.5 W square aggressor whose edges move rings by up to 269 K in one tick,
lines ramped by 5 GHz and stepped by 10 GHz and 30 GHz (beyond ROT_MAX) with a line switched off and on, heater ramps
across both sides of the self-heating fold from the hot branch, traffic on/off with 100 kHz bias ripple, the auto
policy at a 10 kHz controller, DAC glitch and ADC drop faults with the PWM ripple and the er_db / il_db records, the
backscatter (general) path, TCMT, a sampled B = 8 batch under auto) against 8646691 (`adv_cur_vs_base.json`): largest
differences 2.8e-6 pm detuning (fold_up), 3.6e-5 mK ring temperature, 1.3e-5 uW absorbed power, 2.4e-4 nA monitor
current, 8e-6 dB er_db (1.3e-7 relative); every monitor code, heater code, bias, line_on and discrete metric is
identical; the backscatter and TCMT cases are bit-identical; the final Truth (every field, also er_db / il_db, which
the engine now writes only at a block end) agrees to < 3e-10 relative. Chunking (chunk_ticks 1e5 / 777 / 1) and the
thread count (1 vs 8 at B = 8) change nothing, bit for bit, in both trees (for chunking: not under the auto policy
with a compiled controller, see the correction in section 9). The counter RNG streams are unchanged
(mrm/core untouched; every noisy ADC code identical). Controller scorecard rerun on the final tree
(`runs/review_r1/scorecard/`): 19 passed, identical number for number to the optimizer's run, so the comparison with
8646691 of section 5.3 stands (largest: a1_pd_far max_pm 2.6e-6 pm; no discrete metric changed). Fast suite: 1475
passed, 73 skipped, 3 xfailed, 3 failed (VAL-8a/b/c; they fail identically in the fresh 8646691 tree).
`test_kernel_parity` (11, `test_parity_without_jit` included) and `test_engine_adapter` (11) pass.

**The far-line hold (P5) under the adversarial cases** (`adv_hold_vs_cur.json`): up to 0.50 pm of detuning
(fold_up), 40 monitor codes and 4 heater codes (auto_pi10k), and in the violent aggressor case a different
trajectory (167 pm, one more unlock excursion of 4.5 ms). Its P_abs bound holds (6e-11 W measured), but the closed
loops turn any perturbation into another trajectory: it must stay an opt-in.

**Findings and fixes.**
1. `optics._fast_flag` kept the Airy-path flag of a ring from the first call on a packed world (a NaN-keyed memo that
   was never invalidated): a caller that wrote a backscatter coefficient through `selfheat_views` after an
   evaluation kept the Airy path, i.e. ran the ring without its backscatter, against the module's statement that
   the adapter stays correct when a caller rewrites the rows. The plant never rewrites rb after packing, so no
   engine result was affected. Fixed: the flag is derived from the backscatter row at every adapter call and
   refreshed once per substep for the engine (`sh_fast_precompute`); results bit-identical, cost not measurable;
   `test_airy_flag_follows_the_backscatter_row` (fails on the optimizer's tree).
2. The error statement of the incremental cos/sin cache (P4) assumed a random walk (6e-15 to 2e-14 of libm). A
   constant phase step per substep (a steady drift of the ring temperature or of the laser) repeats the same rounding
   at every update, so the cache drifts systematically: 2.8e-13 measured after 4095 updates, derived bound
   ROT_REFRESH x 4 ulp = 1.8e-12, i.e. <= 7e-10 relative in P_abs at resonance and ~5e-9 K of ring temperature,
   20x below the solver's 1e-7 K root tolerance. Still exact up to rounding for every practical purpose; the bound is
   now in the optics module comment, the P4 row above and `test_incremental_phase_cache_constant_step_drift`.
3. The Airy-path comment gave the rotation polynomial's truncation as 1e-19; the code (`_rot`, P7) keeps it below
   1e-17. Fixed.
4. Not changed (noted): the reference engine runs the optimized Airy path (shared code, allowed since P4 is exact up
   to rounding), so `test_kernel_parity` no longer checks the self-heating optics against an independent
   implementation; that check lives in `test_engine_adapter` (closed forms to 2e-13 / 4e-12 relative). The Airy-path
   leaves use numba's numpy error model, so degenerate rows (K^2 = 0 and a = 1 at resonance) give inf/NaN instead of
   ZeroDivisionError, unlike the general path (P14's reason for reverting the same change elsewhere).

## 7. Layout change of 2026-10-10: the baselines above are of the old layout

The g26 default aggressor site moved from (0, 30) to (0, -30) um, and the old site stays in g26 as the stress site
`aggressor_pd_adjacent` (docs/handoff/integrator.md section 7, CHANGELOG). For this file:

- **`runs/perf_baseline` (and `ref2`, `perf_r1`, `perf_final_r1`) recorded `a1_pi` with the old site plus the PD
  moved to the far side** (`ring.thermal.pd_offset_um=[0,-25]`, the former `PD_FAR` of tools/perf_scenarios.py).
  `PD_FAR` is gone: with the new site it would put the PD 5 um from the aggressor. `a1_pi` now runs the default
  layout, whose PD-to-aggressor distances equal the old PD-far ones (mirror image) but whose PDs sit 55 um from
  their own ring's driver site (driver_offset_um (0, -30) um) instead of 5 um. Against `runs/perf_final_r1`
  (`runs/perf_layout_site/compare_vs_final_r1.json`): PD temperatures 8.5-9.0 K lower on every ring (the driver's
  ~4.5 mW of data heat no longer 5 um away), monitor current up to 19 nA lower, monitor codes up to 29, heater codes
  up to 43, detuning up to 0.18 pm, lock-error metrics up to 0.09 pm. **Record a new baseline before any further A/B
  comparison of `a1_pi`.**
- The adversarial `agg_edge_pi` (tools/perf_scenarios_adv.py: a 0.5 W square aggressor at the default site, no PD
  override) ran with the PD 5 um from the site in `runs/review_r1/base_adv`, `cur_adv`, `fix_adv`; it is now a
  different scenario (PD0 up to ~1000 K cooler, another closed-loop trajectory). Re-record it too.
- Every other scenario of both tools is bit-identical to `runs/perf_final_r1` / `runs/review_r1/fix_adv` (the
  second aggressor source has zero power), except the final lag state `final/x`, which has 970 lags instead of 945.
- **Timing**: the 25 extra lags (aggressor_pd_adjacent -> every observer) are in every g26 run, including the
  timing cases of section 5 (`perf_bench.py`). Measured (b1lock method, 5 alternating rounds of 7 x 3000 ticks,
  AC power, load average 4.6 / 2.6 / 2.4): 15.50 us per tick with both sites against 15.38 us with only the
  default site, i.e. +0.1 us (+0.8 %), at the noise level. Timing comparisons with the trees of sections 2-6 therefore compare 970
  against 945 lags; re-time the baseline tree with the same layout, or accept the 0.1 us.

## 8. Round 2 (2026-10-10): 15.5 -> 9.1 us per tick, bit-identical

Owner files as round 1. Starting point: HEAD 6b598f7 (d2328b4, the reviewed round 1, then the g26 aggressor-site
change of section 7). Rule unchanged. Every change of this round is bit-identical: no recorded value of any
regression scenario, test or controller changed. 07 P0-9 (10 us per tick at B = 1, fixed 1 us tick) is met (9.1 us,
8.5) without any approximation.

### 8.1 New baseline at 6b598f7

`runs/perf_baseline_r2/` (git-ignored), recorded in a `git archive HEAD` tree with its own numba cache before any
round-2 change: `ref/` (tools/perf_scenarios.py, 6 scenarios; a1_pi now on the default layout), `adv/`
(tools/perf_scenarios_adv.py, 11 adversarial scenarios), `scorecard/` (tests/validation/test_controllers_on_engine.py:
19 passed, number for number the committed docs/handoff/controller_scorecard.json of 6b598f7), and copies of the
committed scorecards of 8646691 and 6b598f7. The baseline tree reproduces runs/perf_final_r1 bit for bit in every
scenario except a1_pi (the layout change).

### 8.2 Log of changes

After every step: every recorded trace, metric and final Truth (out_rl / out_r / out_l, t_obs and the lag states
included in the adversarial set) of the 6 + 11 scenarios bit-identical to the new baseline (`runs/perf_r2/s1` ...
`s14`, `final`, each with `*_adv`), and the engine tests. Savings: tools/perf_tick_prof.py sections (in context,
instrumented, b1lock) or tools/perf_bench.py / tools/perf_ab.py (b1, b1lock) at the load given; AC power, Low Power
Mode off throughout.

| # | change | kind | saving (B = 1) | error bound | test |
|---|---|---|---|---|---|
| R2-1 | unsigned indices (`_U = np.uint64`) in the hot kernels: Airy-path cache update, evaluation, commit and bounds, thermal lag passes, junction-table lookups, swing capacitance, Truth. numba adds a negative-index wraparound (compare + select) to every access whose signed index it cannot prove non-negative, i.e. every access to the packed rows | exact, bit-identical | with R2-2, R2-3: 16.1 -> 13.5 us instrumented (Newton 2.70 -> 2.27, substep caches 1.52 -> 1.06, bounds 0.98 -> 0.80, lag pass 1.76 -> 1.34); one evaluation 81 -> 60 ns standalone | none | scenarios, parity, test_engine_adapter |
| R2-2 | the far-line hold's evaluation moved out of the evaluation's loop (`optics._sh_fast_pabs_hold`, called only when optics_far_hold is on) | exact, bit-identical | in R2-1 | none | test_parity_with_far_line_hold, test_far_line_hold_error_is_within_its_bound |
| R2-3 | light commit: at every step of a block but its last, the self-heating commit writes only the outputs a tick reads (RL_PIN, RL_THRU, RL_THRU1/0 per line, R_DROP_SUM, R_ABS_SUM, R_DROP_SUM1/0, the walk buffer) and the bus pass only its output part (`optics._bus_pout_nb`; the reflection part `_bus_refl_nb` at the block end); the last step of every block commits in full | exact, bit-identical (every value read in a block and every output after it) | commit 1.48 -> 0.65 us (with R2-1) | none | scenarios (adversarial ones save every final output array), parity |
| R2-4 | review of round 1 (section 6, item 4): `ReferenceEngine(airy=False)` runs independent general-path adapter functions (`optics.selfheat_pabs_general / selfheat_commit_general / selfheat_bounds_general`); degenerate rows (r = t1 t2 a = 1 at resonance) raise ZeroDivisionError explicitly on the Airy path and in `_airy_line` | correctness / test | none | Airy path vs closed forms: P_abs 1e-11, outputs 1e-11 relative; kernel vs independent reference: every Truth field within 1e-9 of its range (measured ~1e-11), codes identical | test_parity_against_independent_optics, test_general_adapter_functions_match_the_airy_path, test_degenerate_row_raises_on_both_paths |
| R2-5 | lags of a source without input whose states are exactly +0.0 skipped (the stress-site lags of every g26 run, 25 of 970; also cold drivers, unheated heaters) until the source gets an input | exact, bit-identical | ~0.05 us | none (x stays +0.0; +0.0 added to a sum leaves it) | scenarios (agg_edge_pi wakes a dead source), test_kernel_releases_every_array |
| R2-6 | error counter: the counting-period test once per tick instead of a modulo and a call per lane | exact, bit-identical | 0.21 -> 0.03 us (standalone) | none | test_datapath, scenarios |
| R2-7 | Truth line search: the division d = num / dpl only for lines that can be inside the wrong-line window (|num| <= lim |dpl| (1 + 1e-12) guarantees fl(|num / dpl|) > lim for the skipped ones) | exact, bit-identical | ~0.05 us | none | scenarios |
| R2-8 | controller edge, auto-policy decision, oracle and fast_kernel on one world's arrays (`_release_a`, `_auto_h_a`, `_oracle_a`, `_fast_tick_a`): a call that passes the KParams/KState tuples copies ~270 array structs (0.44 us measured) | exact, bit-identical | edge 0.48 -> 0.10 us per tick | none | scenarios (auto_pi10k, mc8_auto), parity |
| R2-9 | thermal lag passes from per-lag coefficient tables (a, c0, c1, w side by side), (lag, source) pairs and the inputs (u0, u1) side by side: pair loads | exact, bit-identical | lag pass 1.24 -> 0.94, optical lags 0.27 -> 0.18, heater split 0.56 -> 0.46 us | none | scenarios, parity |
| R2-10 | Truth: the exact-detuning Newton iteration of every ring in lockstep (each ring its own sequence and its own stop at a fixed point) | exact, bit-identical | 11.71 -> 11.24 us b1, 12.03 -> 11.52 b1lock (perf_bench, load 2.2-2.8) | none | scenarios, parity |
| R2-11 | runtime-free compilation (`_nrt=False`) of the hot leaves (kernel, optics, phase, electrical, sensing, datapath, laser, metrics, thermal) and of the tick loop, split from `_world_block` into `_world_loop` (setup with the runtime, loop without): a function with the runtime increfs and decrefs every array it passes to a call (~500 runtime calls per tick) | exact, bit-identical | leaves: 11.24 -> 10.08 b1, 11.52 -> 10.38 b1lock (load 2.3-2.8); loop: -> 9.41 / 9.79 (load 4.9) | none | scenarios, the whole suite, test_kernel_releases_every_array |
| R2-12 | no heap allocation per tick (laser enable mask, the cross-lag power, the heater-split sums) | exact, bit-identical | ~0.1 us | none | scenarios |
| R2-13 | the line constants k1^2, t1^2 kept in the coupler cache (computed when K^2 changes) and the CW energy factor computed where a full commit needs it: 3 stores less per cache entry and substep | exact, bit-identical | substep caches 0.99 -> 0.91 us (1.06 -> 0.82 standalone) | none | test_engine_adapter, scenarios |
| R2-14 | the Airy evaluation inlined into the Newton solver (`inline="always"` on `optics.sh_fast_pabs` and `kernel.eng_pabs`) | exact, bit-identical | A/B 9.57 -> 9.30 b1, 9.88 -> 9.49 b1lock (load 3.1-3.2) | none | scenarios |
| R2-15 | large per-tick leaves inlined into the tick loop (laser, pattern, phase model, heater power, heater split, Truth, temperature sensors, error counter, metrics; monitor_tick_row and _implicit_step_eng cannot be: CPython compiles a call with more than 30 arguments to a *args call, which numba's inliner refuses) | exact, bit-identical | A/B 9.26 -> 9.17 b1, 9.59 -> 9.43 b1lock (load 2.9-3.1) | none | scenarios |
| R2-16 | reference-leak fix of R2-11 (found by the new test): a function with the runtime called from the runtime-free loop must not make views of its arguments (numba 0.68 then leaked one reference to the self-heating rows per block with an oracle call); the oracle and fast kernel take arrays and run runtime-free, the view helpers `_sh_core`, `_sh_outputs`, `selfheat_views` are inlined, the general optics path is runtime-free | exact, bit-identical | none | none | test_kernel_releases_every_array (4 variants: open loop with the oracle, compiled controller with a fast_kernel, Python controller, auto policy, TCMT + backscatter, self-heating off) |

Tried and not kept (each bit-identical, measured slower or no faster): both NRZ levels of a line adjacent in the
Airy cache (no change; reverted), the cache update fused with the first evaluation of each ring (1.55 -> 1.65 us
standalone), the bounds in two passes (0.79 -> 0.88 us), a per-ring evaluation table built per substep (73 -> 99 ns
per evaluation), the observer sums in lockstep (0.19 -> 0.77 us: the sums were never the bottleneck, the advance
is).

### 8.3 Notes

R2-1. The cast `_U(i)` (np.uint64) turns the index into an unsigned one, for which numba emits no wraparound; under
NUMBA_DISABLE_JIT=1 it is a plain index. Every cast index is a non-negative offset by construction.

R2-3. The engine passes `full = last_step` (`k + h` reaches the block end, or the next edge on the Python path) to
`eng_commit` -> `sh_fast_outputs` / `sh_pass_on`. Within a block no kernel computation reads the other fields; after
a block every output is the full set of its last step, which is what Plant.truth(), the reference parity and the
adversarial scenarios' final arrays compare.

R2-5. Per block, `_th_dead_init` marks the sources whose lags are all LAG_EXOGENOUS and exactly +0.0 (sign bit
clear); `_th_live` builds the exogenous lag list without them and trims their trailing lags from each observer's
range; before each lag pass `_th_wake` checks their inputs and rebuilds the lists when one is non-zero. A skipped
lag would have computed a (+0.0) + (c0 0 + c1 0) = +0.0 (the input term is +-0.0, and +0.0 + -0.0 = +0.0 in round to
nearest), and adding +0.0 to an observer sum that starts at the base temperature changes nothing.

R2-11 and R2-16. A function compiled with `_nrt=False` cannot allocate (numba refuses) and cannot return an array it
did not receive (numba refuses). Two hazards numba does not catch: calling a function that returns arrays (the
reference is never released) and calling a function with the runtime that makes views of its arguments (measured:
numba 0.68 leaks one reference per block). The kernel's comment above `_world_block` states the rules;
`test_kernel_releases_every_array` checks the Python reference count of every KParams/KState array over blocks in
six configurations. The tick loop calls functions with the runtime only in other owners' modules (scenario
evaluation, controller pipeline, compiled controllers), none of which makes views of its arguments.

R2-8. The tuple-taking edge helpers that the loop no longer called (`_edge`, `_push_act`, `_advance_edge`,
`_sync_act_from_registers`, `_edge_tick`: a second copy of the edge logic) were removed at the end of the round;
`release_row`, `_auto_h`, `_oracle`, `_fast_tick` and `truth_row` stay as thin wrappers of the array forms. The
scenarios after the removal: `runs/perf_r2/final3`, `final3_adv`, bit-identical.

### 8.4 Accuracy

Bit-identical (8.2). The final tree (`runs/perf_r2/final`, `final_adv`; again `final3`, `final3_adv` after the removal
of the dead edge helpers) against the new baseline: every recorded trace, metric and final Truth of the 6 regression
and 11 adversarial scenarios identical, 0 differing signals in every scenario (`compare_vs_baseline_r2.json` in each
directory). Since 6b598f7 equals the round-1 engine for every scenario but a1_pi (section 7), the comparison with
8646691 of section 5.3 holds unchanged for those (largest 2.2e-6 pm detuning, no code changed).

Controller scorecard (`MRM_SCORECARD_OUT=... pytest -m slow tests/validation/test_controllers_on_engine.py` on the
final tree, `runs/perf_r2/scorecard_final`): 19 passed (181 s; 238 s at 6b598f7), every number identical to the
scorecard of 6b598f7 (`runs/perf_baseline_r2/scorecard`, which equals the committed docs/handoff/
controller_scorecard.json): `compare_vs_6b598f7.txt` reports no change in any metric. Against the copy of 8646691's
scorecard (`tools/scorecard_compare.py ... --rename a1=a1_pd_adjacent a1_pd_far=a1`, `compare_vs_8646691.txt`):
steady and a1_pd_adjacent (the former default a1) differ as in section 5.3 (largest a1_pd_adjacent max_pm 6.4e-7 pm,
steady rms_pm 3.0e-9 pm, penalty_db 1.9e-10 dB, no discrete metric); `a1` against the former `a1_pd_far` is the
layout change of section 7 (another PD-driver distance), not an engine difference.

Tests: fast suite (`tools/memguard.py -- pytest -q -m "not slow"`, `runs/perf_r2/suite_final.log`, again
`suite_final3.log` on the final tree): 1486 passed, 73 skipped, 3 xfailed, 3 failed: VAL-8a/b/c of
tests/validation/test_thermal_replication.py, the known failures of 5.1 (they read the thermal FEM artefacts, not the
engine). test_kernel_parity.py 16 passed (test_parity_without_jit
under NUMBA_DISABLE_JIT=1, the far-line-hold parity, the new test_parity_against_independent_optics and the four
variants of test_kernel_releases_every_array), test_engine_adapter.py 13 passed (two new). tests/perf: the fixed-step
B = 1 target and the Monte Carlo target pass (XPASS); the auto target (>= 20 sim-s/wall-s) does not (xfail). The
numba cache was cleared with tools/clear_numba_cache.py after every edit of njit code and before every timing and
comparison.

### 8.5 Timing (AC power, Low Power Mode off, no other pytest of this project running)

`tools/perf_ab.py` against the `git archive` tree of 6b598f7 (its own numba cache): 5 alternating rounds, per run the
median of 7 x 3000 ticks (B = 64: 7 x 200), load average 3.0-3.7, `runs/perf_r2/ab_base_final.json`. The B = 1
numbers are CPU-bound (CPU time per tick equals the wall time).

| case | 6b598f7 (= round 1) | final | speed-up wall (CPU) |
|---|---|---|---|
| b1: B = 1, fixed, open loop, cold rings | 15.45 us (CPU 15.44) | **9.06 us** (CPU 9.06) | x1.71 (x1.70) |
| b1lock: B = 1, fixed, settled lock codes | 15.46 us (CPU 15.45) | **9.42 us** (CPU 9.42) | x1.64 (x1.64) |
| b64: B = 64, fixed, 8 threads, per world-tick | 3.57 us (CPU/thread 2.98) | **2.14 us** (CPU/thread 1.76) | x1.67 (x1.69) |

Against 8646691 (section 2: 54.3 us b1 at load 2.5-3.1, not taken back to back with this tree) the B = 1 tick is
~6x faster. **07 P0-9 (>= 0.1 simulated s per wall s at the fixed 1 us tick, <= 10 us per tick): met**, with
results bit-identical to 8646691's successors and without any approximation.

tests/perf and the auto policy (`tests/perf/test_perf.py` measure(), single runs, load 2.8-3.2;
`runs/perf_r2/perf_tables.log` and the back-to-back 6b598f7 / final pair of the auto cases):

| case | 6b598f7 | final | sim-s / wall-s (final) |
|---|---|---|---|
| B = 1, fixed | 15.6 us | 9.19-9.29 us | 0.108-0.109 (07 target 0.1: met) |
| B = 1, auto, 100 kHz controller (0.38-0.42 substeps per tick) | 8.42 us | 4.76-5.08 us | 0.20-0.21 |
| B = 1, auto, 10 kHz controller (0.16 substeps per tick) | 4.64 us | 2.86-2.98 us | 0.34-0.35 (07 target 20: not met) |
| B = 64, fixed, 8 threads | 3.56 us per world-tick | 1.97-2.34 us | 64 x 1 s Monte Carlo in 2.1-2.5 min (07: 5 min) |
| B = 64, auto | 3.45 us | 1.90 us | |

(One tests/perf report-table run gave 5.2 us for B = 64 auto, a 200-tick run that includes the first call; the
back-to-back 300-tick pair above is the comparable number.)

Options, A/B against the final engine (3 rounds, load 3.3-3.6; `ab_final_tol6.json`, `ab_final_hold.json`):

| option | b1 | b1lock | B = 64 | effect on results |
|---|---|---|---|---|
| none (final, exact) | 9.11-9.15 us | 9.39-9.45 us | 2.08 us | bit-identical to 6b598f7 |
| `sh_tol_k=1e-6` (default 1e-7 K) | 8.76 us | 9.18 us | | each solve within 1e-6 K of its root (1.29 instead of 1.73 iterations) |
| `optics_far_hold=True` (P5) | 10.07 us | 10.27 us | 2.32 us | now slower than the exact engine: its evaluation (`_sh_fast_pabs_hold`) did not get R2-1 / R2-11 / R2-14; it also changes closed-loop trajectories (5.4, section 6). Kept as it was, off by default; no reason to turn it on. |

The final tree after the removal of the dead edge helpers (`tools/perf_bench.py`, 7 x 3000 ticks, load 2.4;
`runs/perf_r2/bench_final3.log`): b1 9.12 us, b1lock 9.37 us, b64 1.95 us per world-tick (CPU/thread 1.78).

Compile: a cold compile of the final tree (empty numba cache, `tools/memguard.py --heavy`, load 2-4) took 104 s
with a peak RSS of 2.25 GB; a fresh process with the warm cache runs a short benchmark in 1.3 s.

### 8.6 Where the time goes now (B = 1, b1lock; tools/perf_tick_prof.py)

Final tree, `tools/perf_tick_prof.py --locked` (load 2.35; instrumented 9.44 us, sections 9.36 us;
`runs/perf_r2/prof_final_locked.log`):

| part | us per tick | share | round 1 (5.5) |
|---|---|---|---|
| self-heating optics: Newton solves (22.5 ring evaluations of 8 lines x 2 levels, 1.82 iterations per ring) | 1.97 | 21 % | 2.7 |
| self-heating optics: substep caches (cos/sin rotation and Airy constants, 128 entries) | 0.97 | 10 % | 1.6 |
| self-heating optics: bounds of the solve (128 entries, 2 sqrt + 4 divisions each: at the divide/sqrt throughput) | 0.79 | 8 % | 1.0 |
| self-heating optics: commit (light, R2-3) and bus output | 0.41 | 4 % | 1.5 |
| thermal: lag advance and observer sums, optical lags, heater TCR split | 0.88 + 0.16 + 0.38 | 15 % | 2.7 |
| sensors (monitor with 14 flicker poles per ring per ADC sample, error counter, temperature, readback) | 1.39 | 15 % | 1.8 |
| phase model (optics inputs; ~450 divisions), K^2 / slope rows / swing capacitance | 0.49 + 0.12 | 7 % | 1.15 |
| Truth | 0.41 | 4 % | 0.9 |
| bias DAC (8 normal draws), laser (8 RIN draws), heater DAC, sensor inputs (8 exp, 8 erfc) | 0.37 + 0.24 + 0.12 + 0.20 | 10 % | ~1.6 |
| controller edge, scenario, data, metrics, oracle, rest | 0.06 + 0.06 + 0.06 + 0.07 + 0.07 + 0.1 | 4 % | 1.0 |

The counter-based normal draws (Philox4x32-10 + Box-Muller, 32-34 ns each; ~48 per tick: 28 normal pairs of the
monitor flicker, 4 white/shot, 8 bias-DAC, 8 RIN) are ~1.6 us, 17 % of the tick, and fixed by the seeded noise
contract.

### 8.7 What is left

The 10 us target is met with a margin of ~0.6-0.9 us at load 3-4 (b1 9.1, b1lock 9.4). Exact levers left, each
small: the solver evaluation (60 ns per ring evaluation of 16 line-levels standalone after R2-1; 1.97 us / 22.5 =
88 ns in context with the Newton step); the bounds are at the divide/sqrt throughput floor; the per-tick functions of mrm/plant/io.py and mrm/scenarios/generators.py still run
with the numba runtime (other owners; ~0.05 us); monitor_tick_row and _implicit_step_eng cannot be inlined (more
than 30 arguments); the light bus output could compute only the declared lines' POUT (~0.03 us).

Levers that change results (opt-in only, by the rule of 5.4: any per-tick change above rounding moves an ADC code
now and then, and a closed loop then follows another trajectory): cached solver bounds with a margin, or a Newton
predictor from the last step (each solve then lands elsewhere inside the 1e-7 K tolerance; est. 0.5-0.8 us); a
re-optimized far-line hold; a cheaper normal generator (07 P1-13, core owner: every seeded realization changes;
up to ~1 us); a smaller thermal ROM (lags merged per (source, pole): exact up to rounding but another state
representation, thermal owner).

Requests: plant_thermal / integrator: none. Scenario and io owners: their per-tick row functions
(`scen_eval_b`, `pipe_pop_row`, `apply_act_row`, ...) could be compiled with `_nrt=False` (they allocate nothing);
the engine's rule for a function with the runtime called from the runtime-free loop is that it must not make views
of its arguments (8.3).

## 9. Independent review of round 2

Reviewer's re-run of the round-2 tree (raw files in `runs/review_r2/`, git-ignored; the scripts of the extra checks
are described below). AC power, Low Power Mode off, no other pytest of this project running during any timing;
numba caches cleared with `tools/clear_numba_cache.py --all` before the first comparison and again after the
reviewer's edits. The baseline is a fresh `git archive 6b598f7` with its own numba cache.

**Baseline.** The fresh 6b598f7 tree reproduces `runs/perf_baseline_r2/ref` and `adv` bit for bit (6 + 11
scenarios, every array of every npz, sign of zero included), so the optimizer's baseline is genuine.

**Accuracy.** The round-2 tree against the fresh 6b598f7 tree and against `runs/perf_baseline_r2`: 0 differing
arrays in the 6 regression and 11 adversarial scenarios (before and after the reviewer's edits). Against
`runs/perf_baseline` (8646691; a1_pi left out, layout change of section 7): largest differences 2.2e-6 pm detuning
(idle_traffic), 2.9e-5 mK ring temperature, 1.5e-5 uW absorbed power, 2.6e-4 nA monitor current; no monitor code,
heater code or discrete metric changed (round-1 rounding, section 5.3). Seven extra cases aimed at the round-2
changes, run in both trees and compared bit for bit (0 differing arrays): the stress-site aggressor driven by a
0.3 W square wave from 1.5 ms (its source is dead at the block start and wakes mid-block, R2-5; checked with
`_th_dead_init`), both sites driven with a Python controller (blocks end at every edge, R2-3), a Python controller
under the auto policy with lines moving and aggressor edges, `optics_far_hold=True` and `sh_tol_k=1e-6` with lines
moving and aggressor edges, traffic on/off and a bias step with a Python controller, and a sampled B = 2 open loop
run in irregular blocks (1 to 1000 ticks) with `Plant.truth()` out_rl / out_r / out_l saved after every block.
Chunking (chunk_ticks 1e5 / 777 / 13 / 1, also under the auto policy) and the thread count (B = 8: 1, 3 and 8
threads) change nothing, bit for bit. **Correction (2026-10-16, engine review engine-4):** the chunking half is
false for a compiled controller under the auto policy: a block end caps the substep in flight and the growth limit
restarts from it (chunk_ticks 13 against 1e5: 11426 against 10582 substeps, monitor codes up to 9 apart over 20 ms);
the open-loop and Python paths, which this check probably ran, ignore chunk_ticks. Since then the runner ends every
block on a controller edge (no substep crosses one) when the run has no tick jitter, and chunk_ticks changes no
result there (tests/verification/test_engine_blocks.py; docs/ENGINE.md section 4). The counter RNG streams are unchanged (every monitor code, error count and obs
vector identical). Controller scorecard rerun (`runs/review_r2/scorecard`): 19 passed, every one of 4338 numbers
identical to the committed scorecard of 6b598f7; against 8646691 the differences of 8.4 (steady, a1_pd_adjacent at
rounding level; `a1` is the layout change).

**Runtime rules (R2-11, R2-16).** Python reference counts of every KParams / KState array and numba's runtime
allocation statistics (NUMBA_NRT_STATS=1: live allocations and meminfos) are unchanged over 5000 ticks in blocks of
37 with the compiled registry controllers pi_lock_c (square aggressor), dither_lockin_c (traffic on/off),
kalman_lqg_c (auto policy), mpc_c, pi_lock_c with backscatter, and the open loop without self-heating; also with a
compiled controller and fast kernel that make views of their arguments (called through the first-class function
pointer they leak nothing).

**NUMBA_DISABLE_JIT=1.** The shipped no-JIT parity passes; the reviewer added no-JIT kernel/reference parity runs
for the compiled-controller path, TCMT + backscatter, self-heating off and the far-line hold (all pass the
comparison of test_kernel_parity `_compare`), and the auto policy (runs, finite; the reference has no auto policy). agg_edge_pi, auto_pi10k and fold_up without JIT equal the JIT runs except one
Truth sample of agg_edge_pi (detuning of ring 7 at tick 438, 1 ulp of the wavelength: 2.1e-22 m); the 6b598f7 tree
shows the same sample, and its no-JIT run equals the round-2 no-JIT run bit for bit: pre-existing, not from round 2.

**Exactness of R2-1 ... R2-16 (code review).** R2-1: every `_U` cast is of an offset non-negative by construction
(the declared line comes from tx.assignment only when in range, else channel_plan.ring_to_line, validated in
0..L-1; `_U(o_cur)` and `_U(j - 1)` are guarded; `grid_index` returns >= 0 for x strictly inside the grid, and
every caller sends NaN and the ends elsewhere). R2-3: the tick loop reads only RL_PIN, RL_THRU, RL_THRU1/0,
RL_POUT1/0, L_POUT, R_DROP_SUM, R_ABS_SUM and R_DROP_SUM1/0, all written by the light commit; the loop's only
early exit is the Python-path edge, and `last_step` covers it. R2-5: a skipped lag computes a (+0.0) + (+-0.0) =
+0.0 and both wake checks run after the step's inputs are set. R2-7: |num| > fl(fl(lim |dpl|)(1 + 1e-12)) implies
fl(|num / dpl|) > lim (the rounding error is 2 ulp against the 1e-12 margin). R2-10: the same iteration, the same
stop and the same `float(order)` as `cold_coordinate`. R2-13: k1^2 and t1^2 from the same cached k1, t1; the
shared `_afac` memo returns factors of the same a. R2-14 / R2-15: inlining puts the numpy-error-model Airy
evaluation under the caller's Python error model; its only division by a variable (1 / den) is guarded explicitly,
so the values are the same.

**Timing** (`tools/perf_ab.py`, 5 alternating rounds, median of 7 x 3000 ticks per run; `ab_base_vs_final.json`,
load 1.5-2.1; a first pair before the reviewer's edits, load 1.4-2.6, gave 15.48 / 15.46 / 3.55 -> 9.11 / 9.46 /
2.15):

| case | 6b598f7 | round 2 (final) | wall (cpu) speed-up |
|---|---|---|---|
| b1 (B = 1, fixed, cold rings) | 15.36 us (cpu 15.36) | **9.07 us** (cpu 9.08) | x1.69 (x1.69) |
| b1lock (B = 1, fixed, settled lock codes) | 15.47 us (cpu 15.47) | **9.43 us** (cpu 9.43) | x1.64 (x1.64) |
| b64 (B = 64, 8 threads, per world-tick) | 3.67 us (cpu/thread 2.96) | **1.96 us** (cpu/thread 1.78) | x1.87 (x1.66) |

tests/perf (load 2.3): B = 1 fixed 9.24-9.29 us, 0.108 sim-s/wall-s (XPASS: 07 P0-9 met); B = 1 auto 10 kHz 2.85 us
(0.35 sim-s/wall-s, xfail), 100 kHz 5.07 us; B = 64 fixed 2.12-2.33 us (XPASS), auto 1.81 us. Options (3 rounds,
load 1.6-1.8): `optics_far_hold=True` 9.11 -> 9.98 us (b1), 9.46 -> 10.32 (b1lock): slower than the exact engine,
as stated; `sh_tol_k=1e-6` 9.11 -> 8.68 (b1), 9.43 -> 9.21 (b1lock). `tools/perf_tick_prof.py --locked` (load 1.6,
instrumented 10.30 us; the counter reads add more here than in 8.6): Newton 2.13, substep caches 1.05, bounds 0.89,
commit 0.47, lag pass 0.95, sensors 1.45 us, the shares of 8.6. Cold compile after `--all`: ~110 s, peak RSS 2.0-2.1 GB.

**Tests.** Fast suite on the final tree (`suite_final.log`): 1488 passed, 73 skipped, 3 xfailed, 3 failed (VAL-8a/b/c,
the known thermal-artefact failures). test_kernel_parity.py + test_engine_adapter.py: 31 passed.

**Findings and fixes.**
1. No defect in the engine: every change of the round is exact, and no approximation was added.
2. R2-3 had no test of its invariant (every optics output complete after every block: the parity tests compare
   Truth fields, not out_rl / out_r / out_l). Added `test_full_optics_outputs_after_every_block` (Airy path and
   backscatter; blocks of 1 to 64 ticks; the kernel's outputs against the reference's, which commits in full at
   every step). Under NUMBA_DISABLE_JIT=1 with `eng_commit` forced to full = False it fails at the first block. On
   the general path the engine writes RL_DABS_DPHI / R_DABS_DLAM, which the reference leaves at zero (documented
   above EXT_I in kernel.py, since round 1): the test skips those two fields there.
3. Stale references, fixed: optics.py cited R2-12 for the coupler-cache constants (R2-13 in 8.2); the
   `_heater_split_all` docstring named the removed `_heater_split_tables`; docs/handoff/engine.md section 2 listed
   the removed `_edge` and `_push_act` among the remaining wrappers.
4. Noted, not changed: the margin to 10 us is 0.55-0.95 us (b1lock 9.43, b1 9.07 at load 1.5-2.1), so a loaded
   machine can still exceed the budget; the 1-ulp JIT / no-JIT Truth difference above is older than round 2.
