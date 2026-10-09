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

The first call in a fresh environment compiles the kernel (5-8 min, ~2.3 GB peak RSS on the shared M2; run it
through `tools/memguard.py --heavy`). After that numba loads it from its disk cache in under a second per process.

**Cache trap.** numba keys a cached function on its OWN source file only (`numba/core/caching.py`,
`get_source_stamp`: mtime and size of `_py_file`). An edit to `mrm/sim/kernel.py` recompiles the kernel; an edit
to an njit function in another file that the kernel calls (`mrm/sim/metrics.py`, `mrm/plant/*.py`,
`mrm/scenarios/generators.py`, `mrm/core/*.py`) does NOT, and the cached kernel silently keeps the old code. After
such an edit delete the kernel's cache files (`rm mrm/sim/__pycache__/kernel.*.nbi mrm/sim/__pycache__/kernel.*.nbc`)
before running anything (docs/handoff/engine.md says the opposite; this is the checked behaviour).

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
  start at the cool minimum.
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

Then the plant interval [k, k+1): commands latched at k reach the DAC registers; scenario values at k and k+1 (FOH);
laser lines (and the laser_ramp factor); ones density; bias DAC (+ fast offset, latency FIFO, noise, slew, settle,
scenario offset); NRZ levels; optics inputs at the previous ring temperature (`mrm.plant.phase`); self-heating
bounds; data-activity heat; heater TCR split and heater DAC path (+ fast offset, dac_glitch); the implicit thermal +
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
`tests/validation/test_controllers_on_engine.py`), with the conditions. In short (2026-10-09): from the settled warm
start, pi_lock (Python, compiled, composite), feedforward_pi, kalman_lqg and mpc (Python and compiled) hold all 8
rings with < 0.1 pm rms jitter (each compiled version gives the same metrics as its Python version); the dither
locks jitter at the 5 pm level by their own dither; the acquisition controllers without the supervisor lock every
ring on a foreign line; wdm_supervisor locks all 8 on their lines in 23 ms from a cold start. Under A1 with the PD
far from the aggressor, pi_lock, midranging, kalman_lqg and mpc hold rings 1-7 within 6-8 pm, while ring 0 leaves
its point by 350-365 pm at the hot cusp with its heater at code 0 for 14-15 ms (heater authority). In the default g26
layout every controller loses ring 0 (PD0 above 600 degC).

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
- `kernel_ref.ReferenceEngine` is the NumPy tick built from the leaf modules' public functions (07 P1-18). Parity:
  every Truth field, every observation vector and the metrics, bit for bit, open loop and with Python and compiled
  controllers, also under `NUMBA_DISABLE_JIT=1`.
- Truth detuning: the resonance of the order nearest the declared line (tx.assignment when the controller owns it),
  computed exactly on the phase model's cold curve (`kernel.resonance_detuning`); the optics slope is
  `dphi_dlamres = 2 pi c t_rt / (lambda - shift)^2` (engine.md section 3 item 6).

## 8. Extending

- **A Truth field / record channel**: append to `TRUTH_FIELDS` (never reorder), write it in `kernel.truth_row` and
  in `kernel_ref`, extend the parity test.
- **A sensor or actuator channel**: declare it in `mrm/core/channels.py` (ids are append-only), produce it in
  `kernel.release_row` (sensor) or consume it in `_prepare_interval` / `apply_act_row` (actuator), and in
  `kernel_ref`. `ChannelMaps.from_spec` maps names to kernel slots.
- **A scenario signal**: add it to `mrm.scenarios.generators` (signal ids append-only), read it in
  `_prepare_interval` with `_sig_off`, and decide its auto-substep class (linear FOH or held).
- **A fidelity switch**: an integer field in `KParams` (from PlantParams flags), branched inside the kernel; never a
  module-level constant.
- **A controller**: copy a template in `examples/controllers/` (Python, per-ring, compiled), register it with
  `@register("name")` in a module imported by `mrm/control/baselines/__init__.py`, and add it to `CASES` in
  `tests/validation/test_controllers_on_engine.py`.
- After any kernel edit, or an edit of an njit function the kernel calls (then delete the kernel's cache files
  first, section 1): `test_kernel_parity.py`, `test_engine_timing.py`, `test_metrics.py`, `test_engine_physics.py`,
  `test_engine_warm_start.py` (the first run recompiles: use `memguard --heavy`).

## 9. Performance (07 P0-9 budget not met on this machine)

Shared 8-core M2, g26, N = L = 8, warm cache (engine.md section 4 has the breakdown):

| case | us per world-tick | simulated s per wall s |
|---|---|---|
| B = 1, fixed | 94-103 (load 8-14), 133-143 (load 15-22) | 0.007-0.011 (07 target >= 0.1) |
| B = 1, auto, 10 kHz controller | 21-39 | 0.026-0.047 (07 target >= 20) |
| B = 1, auto, 100 kHz controller | 53-74 | 0.013-0.019 |
| B = 64, fixed, 8 threads | 27-40 | 0.0004-0.0005 for the bank |
| B = 1, 100 kHz controller, fixed (scorecard runs, load ~3-5) | ~75 (Python path), ~55 (compiled) including the controller | 0.012-0.013 (Python path), 0.017-0.019 (compiled) |

The implicit self-heating solve (~52 us at B = 1) dominates, then `_prepare_interval` (~20 us) and the sensors
(~17 us). A 64 x 1 s Monte Carlo takes ~35-43 min (07: 5 min). Levers not taken are in engine.md section 4.

## 10. Physics to expect, and open items for the next stage

- **Monitor PD next to the A1/A2 aggressor (g26 layout).** The aggressor site (0, 30 um) is 5 um from ring 0's
  monitor PD (`ring.thermal.pd_offset_um` [0, 25]); under the calibrated A1/A2 power the PD observer reaches
  300-500 degC and its Arrhenius dark current (microamps) moves every monitor-based lock. Both positions are
  assumed; the scorecard runs A1 in the default layout and with the PD on the far side. Open for the plant/config
  owner (docs/handoff/integrator.md section 6).
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
