# The engine: how it works and how to use it

The engine advances B worlds of one MRM bank in 1 us plant ticks with a compiled numba kernel, closes the loop with a
controller at its own rate, and returns online metrics, decimated traces and Truth. It implements the normative
sections 07 P0-1 (implicit self-heating), P0-6 (timing), P0-7 (channels), P0-8 (metrics), P0-9 (compiled kernel),
P0-10 (substeps, quasi-static mode) and P1-18 (NumPy reference) of docs/research/07_design_critique.md ("07").

This file replaces docs/handoff/ENGINE_BRIEF.md (written for the engine builder; it is kept for its tables of leaf
kernel signatures and PlantParams consumers, sections 2, 4 and 5). Design decisions and the review record are in
docs/handoff/engine.md; contract changes in docs/CHANGELOG.md.

| file | role |
|---|---|
| `mrm/sim/runner.py` | `run`, `quasi_static`, `load_inputs`, `warm_start_codes`: the user entry points |
| `mrm/plant/plant.py` | `Plant` (builds the kernel arrays, reset, settle, run, advance, Truth), `Truth`, `prepare_scenario` |
| `mrm/sim/kernel.py` | the compiled per-world tick (`get_run_block`), Truth row, oracle, auto substeps |
| `mrm/sim/kernel_ref.py` | `ReferenceEngine`: the same tick in NumPy (07 P1-18), bit for bit against the kernel |
| `mrm/sim/metrics.py` | `MetricsSpec`, streaming accumulators, `RecordSpec`, `summarize`, `TRUTH_FIELDS` |
| `mrm/plant/io.py` | channel maps, controller pipeline FIFO, observation packing, action clipping |
| leaf modules | `mrm/plant/{phase,optics,thermal,selfheat,electrical,sensing,datapath,laser}.py`, `mrm/scenarios` |

## 1. Quick start

```python
from mrm import params
from mrm.sim import runner
from mrm.sim.metrics import RecordSpec, envelope
from mrm.scenarios import library as lib

cfg = params.load("g26")                                    # preset (or a YAML path, overrides=, alts=)

# open loop, nominal world, 2 ms, record the detuning every 10 us (mean and min/max envelope)
r = runner.run(cfg, None, lib.static(), duration_s=2e-3, uncertainty="nominal", variation="off",
               record=[RecordSpec("detuning_m", 10, "mean"), *envelope("detuning_m", 10)])
r.trace["detuning_m@10:mean"]["value"]                      # (S, B, N), metres
r.metrics["max_abs_detuning_pm"]                             # (B, N)

# a registered controller from a settled warm start at the oracle lock codes, under the A1 aggressor
r = runner.run(cfg, "pi_lock", lib.a1_slide10_aggressor(phase_s=lib.A1_PERIOD_S / 2), duration_s=0.1307,
               uncertainty="nominal", variation="off", heater_code="quasi_static", settle=True,
               metrics={"lock_reference": "oracle"})

# Monte Carlo: 64 sampled worlds on 8 threads (compiled controllers run inside the kernel); rings without a
# quasi-static lock code (unreachable oracle point) start at code_min with a warning
r = runner.run(cfg, "pi_lock_c", None, duration_s=10e-3, batch=64, seeds=1000, n_threads=8,
               heater_code="quasi_static", settle=True)

# the locked steady state over ambient and bias (BND-4), no time stepping
qs = runner.quasi_static(cfg, ambient_c=[0, 25, 105], bias_v=[1.05, 2.0], uncertainty="nominal", variation="off")
qs.heater_code                                               # (A, V, B, N)
```

The first call in a fresh environment compiles the kernel (~2 min since performance round 2, 5-8 min before or under
load; ~2.3 GB peak RSS on the shared M2; run it through `tools/memguard.py --heavy`). After that numba loads it
from its disk cache in about a second per process.

**Cache trap.** numba keys a cached function on its OWN source file only (`numba/core/caching.py`,
`get_source_stamp`: mtime and size of `_py_file`). An edit to `mrm/sim/kernel.py` recompiles the kernel; an edit
to an njit function in another file that the kernel calls (`mrm/sim/metrics.py`, `mrm/plant/*.py`,
`mrm/scenarios/generators.py`, `mrm/core/*.py`) does NOT, and the cached kernel silently keeps the old code. After
such an edit delete the kernel's cache files (`rm mrm/sim/__pycache__/kernel.*.nbi mrm/sim/__pycache__/kernel.*.nbc`)
before running anything (docs/handoff/engine.md says the opposite; this is the checked behaviour).
`.venv/bin/python tools/clear_numba_cache.py [module ...]` deletes the cache files of the kernel and of every leaf
module it calls (or of the named modules); run it after every edit of njit code, and before any timing or exact
comparison (docs/handoff/perf.md).

## 2. Inputs

- **Config and worlds.** `run(cfg_or_params, ...)` takes a Config, a preset name or YAML path, or a PlantParams with
  `cfg=` (the ControllerSpec is always built from the UNSAMPLED config, 07 P1-1). Options `uncertainty`
  ('sample' | 'lhs' | 'nominal', default `experiment.uncertainty`) and `variation` ('sample' | 'off') choose the
  worlds; `batch` is B. Seeds: an int, or `{device, uncertainty, noise, scenario}` base seeds; world b uses
  base + b (07 P1-13). Device draws never change with the noise seed.
- **Scenario.** None (the config's `scenario` section), a registry name, or a `mrm.scenarios.generators.Scenario`
  (`mrm.scenarios.library` builders). Aggressor scenarios (A1, A2, `aggressor_direct`) are calibrated against the
  plant before compiling (`prepare_scenario`: the aggressor power that produces the stated victim-ring trajectory,
  recorded in `manifest_inputs["aggressor_calibration"]`). A1 has its hot cusp at t = 0; pass `phase_s = P/2` to
  start at the cool minimum. They heat the layout site `aggressor` (g26: (0, -30) um, 30 um from ring 0 and 55 um
  from its monitor PD). `a1_slide10_aggressor_pd_adjacent` / `a2_808_aggressor_pd_adjacent` (aliases
  `a1_pd_adjacent`, `a2_pd_adjacent`) are STRESS scenarios, not a realistic layout: the same A1 / A2 at the site
  `aggressor_pd_adjacent` (0, 30) um, 5 um from ring 0's monitor PD, a test of monitor dark-current runaway. The
  rings see the same heat on both sites (same distances, same calibrated power); only the PD paths differ.
- **Controller.** None (open loop), a registry name with `controller_params=` (mrm.control.registry.create), a
  `module:Class` path, an object with the Controller protocol (Python path), or a `CompiledController` (compiled
  path). Section 5.
- **Knowledge.** A controller reads device numbers only from `spec.nominal`, filtered by
  `controller_interface.knowledge` (07 P1-1). g26 releases only `line_grid` and `ring_to_line`. Tokens in use (two
  vocabularies exist; `mrm.control.calibration.FIELD_TOKENS` maps both onto config keys): coarse
  `heater_nominal`, `optics_nominal`, `dlambda_dt_nominal`, `monitor_nominal` (controllers B); fine `heater_law`,
  `heater_r_nominal`, `heater_efficiency`, `linewidth`, `thermal_dynamics`, `bias_efficiency` (controllers A,
  `pi_lock.KNOWLEDGE_KEYS`); `model_nominal` (controllers C, the whole nominal model). Grant them with
  `overrides=["controller_interface.knowledge=[line_grid,ring_to_line,model_nominal]"]`. The two vocabularies are
  not unified yet (section 10).
- **Metrics.** `metrics=` a MetricsSpec or a dict of overrides of the config's `metrics` section (schema keys
  `metrics.lock_threshold_pm`, `hysteresis_pm`, `allowed_assignments`, `lock_reference`, `lock_target_pm`, ...).

## 3. Start states

| how | what the plant does at t = 0 |
|---|---|
| default (`reset_protocol` from `sim.reset_protocol`, g26 `laser_ramp`) | heaters at code_min, cold thermal state, laser lines ramp from 0 over `sim.reset_ramp_s` (3 ms) |
| `reset_protocol="branch_low"` | lasers on at once; the self-heating root is met from the cold side |
| `reset_protocol="branch_high"` | lasers on at once; the hot (self-locked) root |
| `heater_code=(B, N) codes` | the heater registers, FIFO and drive start at these codes (a warm start) |
| `heater_code="quasi_static"` | the oracle lock codes of `quasi_static` at each world's base temperature and bias at t = 0 (`warm_start_codes`); no aggressor, laser-offset or unlocked-ring bus effects; a ring without a code (g26 seeds 7: 13 of 64 sampled rings) starts at code_min with a warning |
| `settle=True` | the thermal state starts at the steady state of the held inputs (below); defaults the protocol to branch_high |
| `factory_table=table` | `controller.load_factory_cal(table)` before `controller.reset(spec)` (07 P1-1 warm start; `mrm.control.calibration.factory_cal` makes a table) |

**Settle** (`Plant.settle`, also `Plant.reset(..., settle=True)`). Without it a warm start begins with every thermal
lag at zero: the heater codes of a lock then leave the ring 100 K too cold for the first 10-50 us and several K too
cold for tens of ms (the crosstalk poles are up to 9 ms in g26), so a ring on the hot branch falls off its fold
(measured: every g26 ring at -800 to -1250 pm after 5 ms). The settle runs the kernel itself open loop with the DC
coefficients of every lag (a = 0, c1 = r: the exact FOH step for h -> infinity), one tick per iteration, until the
ring temperatures move by less than 2 mK, then resets everything except the thermal state. Each iteration is the
plant's own implicit self-heating solve, heater TCR split, bias, data heat and scenario inputs at t = 0, so the start
is the engine's own fixed point (tested against a 100 ms open-loop run at code 0: within 5 mK). g26 nominal at the
quasi-static codes settles in ~18 iterations and then holds its detuning within 0.03 pm open loop. The noise draws of
each iteration limit the convergence to ~0.5 mK. `manifest_inputs["warm_start"]` records the codes, protocol and
the settle report.

g26 nominal, settled at the quasi-static codes: the rings sit at +97 to +121 pm (red of their lines) against the
quasi-static targets of +104 to +105 pm. The difference is the quasi-static solve's simplifications (single bus pass,
own-line heat; the engine adds the other lines' absorption, ~9 % of the heat in rings 0-2), not a settle error.

## 4. What happens in one tick (07 P0-6 event order)

Tick k is the instant at the end of the interval [k-1, k). At instant k, for every world:

1. The `fast_kernel` of a compiled controller runs once per monitor sample released at k (before `step`).
2. At a controller edge: every sensor bank is released (monitor block, temperature sensors, readback, error
   counter; the bit-statistics window; ones and bit sums since the last edge), the observation is packed, the
   controller steps, every actuator command is clipped to its channel range and pushed into the controller pipeline
   to latch at k + `Controller.latency_ticks`.

Then the plant interval [k, k+1) (all of it in the tick loop `kernel._world_loop`, called by `_world_block`;
docs/handoff/perf.md P3, R2-11): commands latched at k reach the DAC registers; scenario values at k and k+1 (FOH);
laser lines (and the laser_ramp factor); ones density; bias DAC (+ fast offset, latency FIFO, noise, slew, settle,
scenario offset); NRZ levels; optics inputs at the previous ring temperature (`mrm.plant.phase`; K^2 at the lines
only when the line wavelengths move) and the substep caches of the self-heating adapter; self-heating bounds;
data-activity heat; heater TCR split and heater DAC path (+ fast offset, dac_glitch); the implicit thermal +
self-heating step (`kernel._implicit_step_eng`, a line-for-line mirror of `mrm.plant.selfheat.implicit_step_b`; the
optics walk runs inside it); sensors of the interval (monitor, error counter, temperature sensors, readback); Truth,
metrics and records of the state at k + 1. Every analog or converter delay lives in its leaf module; the engine owns
only the controller pipeline (07 P0-6).

**Substeps** (`substep_policy`, default `sim.substep_policy`, g26 `fixed`): `auto` takes power-of-two steps up to
`auto_max_ticks` (64) while the actuators are quiescent, the scenario is linear, the detuning rate is small and no
edge, record window or oracle tick intervenes; a scenario event forces a 1-tick step and h at most doubles per step.
The thermal step is exact for any h; per-tick sensors still run every tick. Details: engine.md section 3 item 10.

## 5. Controllers on the engine

- **Python path** (any object with `reset(spec)`, `step(obs) -> {channel: codes}`, `latency_ticks`): the kernel
  stops at every controller edge, the runner builds an `Observation` from the released channels, validates the
  actions (shape, dtype, range: 07 P1-19) and resumes. Vectorized over B; needs edges shared by every world (no
  `tick_jitter_s`). Cost: ~100-300 us per edge on top of the plant.
- **Compiled path** (`mrm.control.api.CompiledController`: numba `step(p_i64, p_f64, s_i64, s_f64, obs_i64,
  act_i64)` typed `STEP_FT`): runs inside the kernel, per world, with edge jitter. The optional `fast_kernel` runs
  once per released monitor sample; its heater/bias offsets are held, added to the latched command before the DAC
  FIFOs and clipped. One compiled kernel serves every controller (the step is a first-class function value), so a
  new controller does not recompile the plant.
- **Truth-fed controllers** (`truth_fed = True`, only `mrm.validation.oracle.Oracle`): called as
  `step(obs, truth=plant.truth())`; `Truth` carries the lag states `x`, `heater_applied` and per-world
  `conditions`. The registry refuses them.
- **Composite** (`CompositeController`, registry `composite`, params `{children: [{name, rate_hz, params?}]}`):
  children at their own rates on the Python path.
- **Registered baselines** (`mrm.control.registry.available()`; importing `mrm.control.baselines` registers them):

| family | names | start they expect | needs |
|---|---|---|---|
| feedback locks (A) | `pi_lock`, `pi_lock_c`, `dither_lockin`, `dither_lockin_c`, `es_po`, `es_sine`, `feedforward_pi` | ring on the hot branch red of its line (warm start + settle); they map the peak themselves | `feedforward_pi`: a temperature sensor and `dlambda_dt_nominal` + `heater_law` + `heater_efficiency` (or `ff_codes_per_k`) |
| bias locks (A) | `bias_dither`, `midranging` | as above | `controller_interface.bind.tx.bias=controller`, `scenario.bias_v=null` |
| acquisition / WDM (B) | `ramp_acquire`, `acquire_then_lock`, `bitstat_lock`, `wdm_supervisor` | cold (heaters at 0, `branch_low`); `wdm_supervisor` also a factory table (`factory_table=`) | `heater_nominal`, `optics_nominal`, `dlambda_dt_nominal` tokens help; `wdm_supervisor` writes `tx.assignment` when it is bound to the controller |
| model based (C) | `kalman_lqg`, `kalman_lqg_c`, `mpc`, `mpc_c` | `start='warm'`: at the lock (settled warm start); `start='cold'` ramps (Python versions only) | `model_nominal` token |

docs/handoff/controller_scorecard.md has every one of them on the engine (g26 nominal, steady lock and A1, from
`tests/validation/test_controllers_on_engine.py`), with the conditions. In short (2026-10-10): from the settled warm
start, pi_lock (Python, compiled, composite), feedforward_pi, kalman_lqg and mpc (Python and compiled) hold all 8
rings with < 0.1 pm rms jitter (each compiled version gives the same metrics as its Python version); the dither
locks jitter at the 5 pm level by their own dither; the acquisition controllers without the supervisor lock every
ring on a foreign line; wdm_supervisor locks all 8 on their lines in 23 ms from a cold start. Under A1 (default
site, 55 um from PD0), pi_lock, midranging, kalman_lqg and mpc hold rings 1-7 within 6-8 pm, while ring 0 leaves
its point by 350-365 pm at the hot cusp with its heater at code 0 for 14-15 ms (heater authority); feedforward_pi
loses every ring (its die sensor sees the aggressor; integrator handoff 6.4). In the stress scenario
a1_slide10_aggressor_pd_adjacent every controller loses ring 0 (PD0 above 600 degC).

## 6. Outputs

`RunResult(metrics, trace, truth_final, manifest_inputs, diagnostics, wall_s, plant)`.

- **metrics** (`mrm.sim.metrics.summarize`; per ring (B, N) unless noted): detuning mean / rms / max about the
  line; lock error e = d - d_ref (`max_abs_lock_error_pm`, `rms_lock_error_pm`, `locked_final`, `ever_locked`,
  `excursion_count` with hysteresis, `excursion_max_s`, `time_unlocked_s`, `first_lock_s`, `time_to_lock_s`);
  `time_above_s`; wrong-line time, events and final state (`allowed_assignments` fixed | cyclic | any); FSR hops;
  relOMA and the oracle relOMA (`mean_reloma_db`, `oracle_reloma_db`, `oma_penalty_db`, unrestricted variants);
  BER (expected errors, windows over the limit, worst window); heater energy, time at the rails, peak demanded slew,
  code range used; `world` sums; `settings`. **Lock reference**: `lock_reference='line'` (default, 07's literal
  detuning about the line), `'oracle'` (the oracle's best stable relOMA point, refreshed every 1 ms) or `'target'`
  (`lock_target_pm`, scalar or (B, N)). A side-of-fringe lock sits 0.3-0.5 FWHM from its line by design and reads
  "never locked" about the line.
- **trace**: `{"<field>@<decimation>:<reducer>": {"t_s": (S,), "value": (S, B, N)}}` for each `RecordSpec(field,
  decimation_ticks, reducer)` with reducer mean | min | max | last; `envelope(field, dec)` gives min and max (keeps a
  1-tick spike that a mean hides). Fields: `mrm.sim.metrics.TRUTH_FIELDS` (temperatures of ring / heater / PD,
  heater and absorbed power, resonance, detuning in m and HWHM, monitor input and current, P_in, bus OMA, relOMA, ER,
  IL, heater code, bias, order, line the ring sits on, base temperature, monitor code, oracle relOMA, BER, declared
  line).
- **truth_final** (`Truth`): every field above plus per-line chip and output power and wavelength, the optics arrays,
  every thermal observer temperature, the lag states and per-world conditions. Truth goes to tests, metrics and the
  oracle only, never to a controller.
- **diagnostics**: non-converged self-heating solves, lost samples (FIFO, block), pipeline overflows, clipped
  commands, substeps, largest substep, fast-kernel calls, saturation counts, FIFO capacities.
- **manifest_inputs**: config fingerprint and hash, seeds, B, N, L, dt, ticks, policy, scenario fingerprint and
  aggressor calibration, controller name, path and latency, spec fingerprint, metrics settings, world choices,
  records, diagnostics, `warm_start`.

`quasi_static(...) -> QuasiStaticResult` (07 P0-10): per (ambient, bias, world, ring) the oracle lock detuning,
relOMA (stable and unrestricted), ring temperature, absorbed and heater power, heater code, code-range coverage,
FSR-hop and wrong-line risk; no time stepping (~0.1 s for one point of g26).

## 7. Internals (read before changing the engine)

- `Plant._build` turns PlantParams + Scenario + ControllerSpec into two namedtuples of arrays, `KParams` (~180
  fields) and `KState` (~90 fields). Every array is a writeable C-contiguous copy so the numba types never depend on
  the config; `_check_kernel_arrays` raises on a read-only or strided view (that would be a new type and recompile).
  Per-ring state lives in blocks indexed by named constants (`kernel.RF`, `RI`, `R8`, `RB`, `WI`, `WF`, `LF`).
- The kernel is one `@njit(nogil=True, cache=True)` chunk function with an explicit signature; worlds run in
  contiguous chunks on a thread pool (a numba `prange` cannot take the 180-array namedtuple; engine.md section 3
  item 1). Results are bit-identical for any thread count.
- No closures over config values and no dispatcher arguments (07 P0-9): the self-heating adapter is called
  directly (`eng_pabs` / `eng_commit`), which is why `_implicit_step_eng` mirrors `selfheat.implicit_step_b`. Any
  change to the selfheat solver must be copied there; `tests/verification/test_kernel_parity.py` fails bit for bit
  otherwise, and `test_kernel_is_disk_cacheable` guards the cache.
- The whole tick lives in `_world_block` (per block: views, per-block tables, scratch) and `_world_loop` (the tick
  loop), docs/handoff/perf.md P3, R2-11. The loop and the hot leaves are compiled with `_nrt=False` (without the
  numba runtime: no reference-count update of every array passed to a call, ~500 per tick). Rules, stated in the
  comment above `_world_block`: a runtime-free function cannot allocate or return a new array (numba refuses both);
  it must not call a function that returns an array (the reference is never released), and a function with the
  runtime that it calls must not make views of its arguments (numba 0.68 then leaks one reference per call);
  `tests/verification/test_kernel_parity.py::test_kernel_releases_every_array` checks the reference counts. A call
  that receives the KParams/KState tuples copies ~270 array structs (~0.4 us), so every per-tick helper takes one
  world's arrays: `_truth_a`, the controller edge (`_release_a`, `_sync_act_a`, `_push_act_a`, `_advance_edge_a`),
  `_auto_h_a`, `_oracle_a`, `_fast_tick_a`, the leaf `*_row` kernels. The tuple forms `release_row`, `truth_row`,
  `_auto_h`, `_oracle` and `_fast_tick` are thin wrappers for other callers.
- Hot loops index packed rows through `_U(i)` (`np.uint64`; perf.md R2-1): numba adds a negative-index check to
  every signed index it cannot prove non-negative. Use it only for offsets that are non-negative by construction.
- The self-heating adapter's exact-ring path (no backscatter; the g26 default) lives in `mrm.plant.optics`
  (`sh_fast_precompute`, `sh_fast_pabs`, `sh_fast_outputs`, `sh_bounds`) and is shared by the engine and the
  reference (through `selfheat_pabs` / `selfheat_commit` / `selfheat_bounds`), so the parity stays bit for bit: per
  substep it caches cos/sin of each line's reference phase (updated by rotation) and the constants of the Airy
  closed forms; an evaluation rotates by the small in-solve angle and divides once. Equal to the closed forms up to
  floating-point rounding (tests/verification/test_engine_adapter.py). The caches are keyed on their inputs, so the
  optics adapter stays correct for any caller that rewrites the rows; the engine refreshes them once per substep
  (`kernel.eng_precompute`) and evaluates without checks. At every substep but the last of a block the commit
  writes only the outputs a tick reads (`eng_commit(..., full=False)`, perf.md R2-3); the last step commits in full,
  so every output after a block is complete. `kernel_ref.ReferenceEngine(airy=False)` runs the independent general
  path (`optics.selfheat_pabs_general` / `selfheat_commit_general` / `selfheat_bounds_general`) for a check that
  does not share the Airy path (`test_parity_against_independent_optics`). A degenerate ring row (r = t1 t2 a = 1 at
  resonance) raises ZeroDivisionError on both paths.
- `Plant(optics_far_hold=True)` / `run(..., optics_far_hold=True)` (default False): the far-line hold, a bounded
  approximation (lines >= 8 half-widths from every resonance held to first order in phase and loss between
  refreshes; P_abs error <= 3e-9 W per ring, ~1e-3 pm). Since performance round 2 it is ~10 % slower than the
  exact engine (its evaluation did not get the round-2 changes, perf.md 8.5), and the closed loops amplify its error
  into other trajectories (up to 0.35 pm and 17 monitor codes in the regression scenarios, perf.md 5.4): keep it
  off. `Plant(sh_tol_k=...)` (default 1e-7 K) is the self-heating root tolerance. Both are
  also `run()` options.
- The kernel's thermal pass runs from tables built at the start of each block (lag indices by kind, per-lag gains
  src_scale (c0 + c1), implicit gains per observer, heater-split gains, the lockstep layout of the heater observers;
  per-lag coefficient tables and (lag, source) pairs; perf.md P8, P13, R2-9; the lags of a source whose input and
  states are exactly zero are skipped until it gets an input, R2-5): bit-identical to `selfheat.implicit_step_b` /
  `thermal.observer_split_b`. Tables that depend on KParams are rebuilt per block, never cached across calls
  (Plant.settle swaps the coefficient tables).
- `kernel_ref.ReferenceEngine` is the NumPy tick built from the leaf modules' public functions (07 P1-18). Parity:
  every Truth field, every observation vector and the metrics, bit for bit, open loop and with Python and compiled
  controllers, also under `NUMBA_DISABLE_JIT=1`. With `airy=False` (independent optics, above) the parity is within
  1e-9 of each Truth field's range, codes identical.
- Truth detuning: the resonance of the order nearest the declared line (tx.assignment when the controller owns it),
  computed exactly on the phase model's cold curve (`kernel.resonance_detuning`); the optics slope is
  `dphi_dlamres = 2 pi c t_rt / (lambda - shift)^2` (engine.md section 3 item 6).

## 8. Extending

- **A Truth field / record channel**: append to `TRUTH_FIELDS` (never reorder), write it in `kernel.truth_row` and
  in `kernel_ref`, extend the parity test.
- **A sensor or actuator channel**: declare it in `mrm/core/channels.py` (ids are append-only), produce it in
  `kernel._release_a` (sensor) or consume it in `_world_loop` (the interval inputs) / `apply_act_row` (actuator),
  and in `kernel_ref`. `ChannelMaps.from_spec` maps names to kernel slots.
- **A scenario signal**: add it to `mrm.scenarios.generators` (signal ids append-only), read it in `_world_loop`
  (offset from `_sig_off`, taken once per block in `_world_block`), and decide its auto-substep class (linear FOH
  or held).
- **A cache in the kernel** (a value kept across ticks because its inputs did not change): key it on its inputs and
  invalidate it in `Plant._reset` (as `RI_HE_TCODE`, `LF_PDB_KEY`, the optics adapter's NaN keys); never assume
  that an input is constant.
- **A fidelity switch**: an integer field in `KParams` (from PlantParams flags), branched inside the kernel; never a
  module-level constant.
- **A controller**: copy a template in `examples/controllers/` (Python, per-ring, compiled), register it with
  `@register("name")` in a module imported by `mrm/control/baselines/__init__.py`, and add it to `CASES` in
  `tests/validation/test_controllers_on_engine.py`.
- After any kernel edit, or an edit of an njit function the kernel calls (then delete the kernel's cache files
  first, section 1): `test_kernel_parity.py`, `test_engine_timing.py`, `test_metrics.py`, `test_engine_physics.py`,
  `test_engine_warm_start.py` (the first run recompiles: use `memguard --heavy`).

## 9. Performance (07 P0-9: the fixed-tick and Monte Carlo budgets are met; the auto-substep budget is not)

8-core M2 on AC power (Low Power Mode off), g26, N = L = 8, warm cache, after performance round 2
(docs/handoff/perf.md section 8; engine.md section 4 has the breakdown):

| case | us per world-tick | simulated s per wall s |
|---|---|---|
| B = 1, fixed | 9.1-9.3 (load 3-3.7) | 0.108-0.110 (07 target >= 0.1: met) |
| B = 1, auto, 10 kHz controller | 2.9-3.0 | 0.34-0.35 (07 target >= 20: not met) |
| B = 1, auto, 100 kHz controller | 4.8-5.1 | 0.20-0.21 |
| B = 64, fixed, 8 threads | 2.0-2.3 | 0.0068-0.0079 for the bank: 64 x 1 s in 2.1-2.5 min (07: 5 min) |

Commit 8646691 ran 54-55 us per B = 1 tick under the same conditions, round 1 15.5 us (results equal up to
floating-point rounding, perf.md 5.3); round 2 changed no result (bit-identical, perf.md 8.4). The margin to 10 us is
~0.6-0.9 us at load 3-4, so a busy machine can push a run above it. The self-heating optics (~4.1 us at B = 1)
dominate, then the thermal ROM (~1.4 us) and the sensors (~1.4 us, mostly counter-based noise draws). Battery power
or Low Power Mode is ~1.7x slower. Options: `sh_tol_k=1e-6` ~4 % faster; `Plant(optics_far_hold=True)` (an
approximation that changes closed-loop trajectories, perf.md 5.4) is now ~10 % slower: keep it off.

## 10. Physics to expect, and open items for the next stage

- **Monitor PD next to the A1/A2 aggressor: a stress scenario only (resolved 2026-10-10).** The g26 aggressor site
  moved from (0, 30) to (0, -30) um: 55 um from ring 0's monitor PD (`ring.thermal.pd_offset_um` [0, 25]) instead
  of 5 um, with every ring-to-site distance unchanged. At 5 um the calibrated A1/A2 power heats the PD observer to
  500-600+ degC and its Arrhenius dark current (microamps) moves every monitor-based lock; that layout is kept as the
  stress scenarios `a1_slide10_aggressor_pd_adjacent` / `a2_808_aggressor_pd_adjacent` (section 2). Both positions
  are assumed. docs/handoff/integrator.md section 7.
- **Wrong-line capture at power-up.** Each g26 ring's cold resonance sits ~2 FWHM blue of line n+3; a cold ring can
  self-heat onto it, and a heater ramp crosses foreign lines. Acquisition controllers lock on the first line they
  meet unless a supervisor assigns lines.
- **Bias tunes weakly in g26**: a 0.2 V step moves the resonance 0.1-0.5 pm (the FCA heat cancels most of the
  electro-optic shift); bias locks are weak (controllers_a handoff).
- **Data-activity heat** lifts each ring ~2 K; traffic and squelch steps move rings by this much.
- Not built yet: experiments (sweep, mc, compare, robust, acceptance), run manifests and the CLI `mrm run`, the gym
  environment and the binary bridge, the P1-20 telemetry metrics, the 07 P0-10 compression convergence runs,
  `fidelity.data_modulation: lut` (the waveform LUT exists: docs/handoff/waveform.md section 3), the reference-tap
  and `datapath.bitstat_ones` channels, `DacFacts.law`, one knowledge vocabulary, `run_lock_request` for the
  replicas (docs/handoff/replicas.md section 6).
