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
| `mrm/sim/experiment.py`, `manifest.py`, `cli.py` | experiment files, run directories, `mrm run ...` (section 11) |
| `mrm/sim/bridge.py`, `vectors.py`, `mrm/control/cli.py` | external controllers, golden vectors and traces (section 12) |
| `mrm/sim/gym_env.py`, `mrm/control/fxp_mlp.py` | the RL environment and fixed-point policies (section 13) |
| `mrm/calibration/`, `mrm/validation/report.py` | the calibration fits, the calibrated presets and the generated validation report (section 14) |

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

Whole experiments (one YAML file: config, controller, scenario, worlds, records, criteria; section 11):

```
mrm run experiments/examples/a1_pi_lock.yaml            # writes runs/<UTC>_<hash8>/; --dry-run estimates first
mrm acceptance --all                                     # the acceptance table (A4 alone takes hours)
```

```python
import mrm
res = mrm.run("experiments/examples/a1_pi_lock.yaml")   # or mrm.run(mrm.params.load(path))
res.report, res.metrics, res.run_dir
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
  base + b (07 P1-13). Device draws never change with the noise seed. A settled start (section 3) holds the rule
  too since 2026-10-16: each world stops its own settle, so world b of a batch starts as it does alone.
- **Scenario.** None (the config's `scenario` section), a registry name, or a `mrm.scenarios.generators.Scenario`
  (`mrm.scenarios.library` builders). Aggressor scenarios (A1, A2, `aggressor_direct`) are calibrated against the
  plant before compiling (`prepare_scenario`: the aggressor power that produces the stated victim-ring trajectory,
  recorded in `manifest_inputs["aggressor_calibration"]`). A1 has its hot cusp at t = 0; pass `phase_s = P/2` to
  start at the cool minimum. They heat the layout site `aggressor` (g26: (0, -30) um, 30 um from ring 0 and 55 um
  from its monitor PD). `a1_slide10_aggressor_pd_adjacent` / `a2_808_aggressor_pd_adjacent` (aliases
  `a1_pd_adjacent`, `a2_pd_adjacent`) are STRESS scenarios, not a realistic layout: the same A1 / A2 at the site
  `aggressor_pd_adjacent` (0, 30) um, 5 um from ring 0's monitor PD, a test of monitor dark-current runaway. The
  rings see the same heat on both sites (same distances, same calibrated power); only the PD paths differ.
  **Signal defaults come from the config only when `scenario` is None** (`mrm.scenarios.registry.from_config`
  applies `scenario.ambient_c`, `scenario.bias_v` and `datapath.transition_density`). A registry name or a Scenario
  object keeps its own defaults: g26 then runs with the bias not driven, i.e. at the bias DAC's initial code
  (1.0 V), not at `scenario.bias_v` 1.05 V. The warm-start codes and every trajectory differ (g26 static: 4 codes;
  pi_lock_c under the soak: up to 100 pm while it maps its peak). Experiment files always build the scenario from
  the config; pass `scenario=None` with `scenario.*` overrides, or `registry.from_config(cfg)`, to get the same
  plant from Python.
- **Controller.** None (open loop), a registry name with `controller_params=` (mrm.control.registry.create), a
  `module:Class` path, an object with the Controller protocol (Python path), or a `CompiledController` (compiled
  path). Section 5.
- **Knowledge.** A controller reads device numbers only from `spec.nominal`, the view of the unsampled config
  filtered by `controller_interface.knowledge` (07 D13, P1-1; `mrm.control.knowledge.NominalView`, since
  2026-10-16: before, `spec.nominal` was the whole Config, experiment section included, and only the helpers below
  applied the gate). `spec.nominal.get` / `.si` of a key no granted token releases raise `KnowledgeError`; the
  experiment section (seeds, world choices) is never released, so the spec fingerprint is the same for every seed
  and world choice, also when they are config leaves. g26 releases only `line_grid` and `ring_to_line`. Tokens in
  use (two vocabularies exist; `mrm.control.knowledge.token_keys()` merges them, and
  `mrm.control.calibration.FIELD_TOKENS` maps both onto the helpers' field groups): coarse `heater_nominal`,
  `optics_nominal`, `dlambda_dt_nominal`, `monitor_nominal` (controllers B); fine `heater_law`,
  `heater_r_nominal`, `heater_efficiency`, `linewidth`, `thermal_dynamics`, `bias_efficiency` (controllers A,
  `pi_lock.KNOWLEDGE_KEYS`); `model_nominal` (controllers C, the whole nominal model: every key but the experiment
  section and the scenario beyond its operating point `scenario.ambient_c` / `bias_v` / `bias_offset_v`;
  `spec.nominal.model_config()` is that design for `mrm.control.models.linearize`). The monitor ADC's ampere scale
  of the `tx.monitor.samples` / `tx.monitor.latest` channel specs is `monitor_nominal` (or `model_nominal`)
  knowledge too; without it they carry codes only. Grant tokens with
  `overrides=["controller_interface.knowledge=[line_grid,ring_to_line,model_nominal]"]`. Still open: the view is of
  the run's own config, so a corner selected with `alts=` reaches `model_nominal` (no `controller_knowledge` preset,
  docs/handoff/controllers_c.md section 4).
- **Exposed sensors.** `controller_interface.expose` (a list of sensor channels; default every channel) limits what
  the controller reads, on every path (runner, MRMEnv, bridge, vectors): a hidden channel raises
  `UndeclaredChannelError` on the Python path and has `obs_offset` -1 in the compiled ABI; a sample block brings its
  count channel. An unknown name is a `ChannelError`. Before 2026-10-16 the key was read by nothing.
- **Metrics.** `metrics=` a MetricsSpec or a dict of overrides of the config's `metrics` section (schema keys
  `metrics.lock_threshold_pm`, `hysteresis_pm`, `allowed_assignments`, `lock_reference`, `lock_target_pm`, ...).
  `score_from_s=t` restarts the metric accumulators at t (a controller's start-up, e.g. pi_lock's peak map in its
  first ~3 ms, is then not scored; `MetricsState.restart` keeps the oracle lock reference and the previous
  controller latch; on the Python path a split on a controller edge moves one tick later). Under the fixed
  substep policy plant, controller and records are untouched. Under `auto` the split ends a substep at t (a kernel
  block ends there, section 4), so they are untouched only when t is a controller edge (g26: a multiple of 10
  ticks after the phase; elsewhere the runner warns, `runner.ScoreSplitWarning`: an off-edge split gave 240 of 320
  detuning samples up to 4.6e-14 m and monitor codes up to 9 apart over 20 ms), and the restarted BER windows
  (`metrics.ber_sample_period_s`, 2 s) cap substeps at their own ends, so a run longer than one window after a
  split that is not a whole number of windows after the reset takes other substeps after the first window end
  (correction 2026-10-16, engine review: this said "untouched" for every split).

## 3. Start states

| how | what the plant does at t = 0 |
|---|---|
| default (`reset_protocol` from `sim.reset_protocol`, g26 `laser_ramp`) | heaters at code_min, cold thermal state, laser lines ramp from 0 over `sim.reset_ramp_s` (3 ms) |
| `reset_protocol="branch_low"` | lasers on at once; the self-heating root is met from the cold side |
| `reset_protocol="branch_high"` | lasers on at once; the hot (self-locked) root |
| `heater_code=(B, N) codes` | the heater registers, FIFO and drive start at these codes (a warm start) |
| `heater_code="quasi_static"` | the oracle lock codes of `quasi_static` at each world's base temperature and bias at t = 0 (`warm_start_codes`), held in the data path of t = 0 (`quasi_static(data_state=)`, `runner.reset_data_state`: driver enable, swing scale, transition and ones density; the target stays the traffic-on operating point); no aggressor, laser-offset or unlocked-ring bus effects; a ring without a code (g26 seeds 7: 13 of 64 sampled rings) starts at code_min with a warning. Every such ring measured so far is one whose heater-up order needs a negative heater power (it is already red of its lock point at code_min) |
| `heater_code="quasi_static_reachable"` | the same, but such a ring locks one order bluer (one FSR of heating) instead of starting at code_min (`quasi_static(order="reachable")`), and so does every ring that the hopped rings' heat then pushes below zero, until none needs a negative power (g26 seeds 1000, B = 8: 10 -> 0 rings without a code; 14 rings hop, the 10 and 4 that had heater-up codes; worlds 0-2 exceed the array supply limit at their codes). Worlds without such a ring get the same codes as `"quasi_static"`. The experiments' `start: warm` uses this mode |
| `settle=True` | the thermal state starts at the steady state of the held inputs (below); defaults the protocol to branch_high |
| `factory_table=table` | `controller.load_factory_cal(table)` before `controller.reset(spec)` (07 P1-1 warm start; `mrm.control.calibration.factory_cal` makes a table) |

**Array heater supply limit.** The plant scales every heater of a world by one droop factor when the sum of the
heater powers or currents exceeds `array.heater_total_power_max_w` / `array.heater_supply_current_max_a` (g26: 0.2 W,
0.15 A, both assumed; 07 P1-12, `mrm.plant.electrical.supply_limit_row`). `quasi_static` checks its lock codes
against the same limit (`supply_scale`, `supply_limited`); a warm start in a supply-limited world keeps its codes and
warns (`manifest_inputs["warm_start"]["supply_limited"]`): no ring of that world can hold its lock there. The
nominal g26 world never reaches the limit (0-105 degC, 1-3.75 V). The `stacked_eic` stack hypothesis (P_FSR 36.9 mW)
does below ~30 degC: at 25 degC the lock currents are ~155 mA, the plant droops the heaters by 3-5 % in voltage, and
every ring settles on its cold self-heating branch, 800-1100 pm blue of its line. Corners `g26_low_eta` and
`g26_fsr_edge` reach the limit at 0 degC (droop 0.985-0.989).

**Settle** (`Plant.settle`, also `Plant.reset(..., settle=True)`). Without it a warm start begins with every thermal
lag at zero: the heater codes of a lock then leave the ring 100 K too cold for the first 10-50 us and several K too
cold for tens of ms (the crosstalk poles are up to 9 ms in g26), so a ring on the hot branch falls off its fold
(measured: every g26 ring at -800 to -1250 pm after 5 ms). The settle runs the kernel itself open loop with the DC
coefficients of every lag (a = 0, c1 = r: the exact FOH step for h -> infinity), one tick per iteration, until the
ring temperatures move by less than 2 mK, then resets everything except the thermal state. Each iteration is the
plant's own implicit self-heating solve, heater TCR split, bias, data heat and scenario inputs at t = 0, so the start
is the engine's own fixed point (tested against a 100 ms open-loop run at code 0: within 5 mK). g26 nominal at the
quasi-static codes settles in ~18 iterations and then holds its detuning within 0.03 pm open loop. The noise draws of
each iteration (a new tick each time) limit the convergence to ~0.5 mK. Every world stops at its own first
iteration below the tolerance (`iterations_per_world`, `converged_per_world` in the report), so its start does not
depend on the other worlds of the batch (07 P1-13). The scenario is held at t = 0 during the settle: every signal
keeps its value of the reset tick (2026-10-16: before, iteration i saw the scenario of tick i, so an input that
changed within the first ~20 ticks was settled as if already there; an aggressor switched on at 10 us started ring
0 18.7 K hot and +1554 pm red instead of at +121 pm, reported converged; the shipped scenarios moved by at most a
few pm). `manifest_inputs["warm_start"]` records the codes, protocol and the settle report.

**Warm start in a data state other than traffic on.** The quasi-static codes put each ring at its traffic-on
operating detuning with the heat of the data path at t = 0. A start with traffic off (`traffic_start_stop` with
t_start > 0) gets 1243-1348 codes more than the static start (the missing data-activity heat, ~1-1.4 K) and settles
within 1.0 pm of the static start's detuning; a start at ones density 0 (`ones_density_steps`, `idle` at t0 = 0)
within 0.5-4.6 pm, the residual growing along the bus (quasi_static leaves out the upstream rings' filtering of the
other lines, which the density changes). Before 2026-10-16 the codes always assumed traffic on: such starts began
65-98 pm (traffic off) and 13-24 pm (ones density 0) blue of the lock, and in `static +
traffic_start_stop(0.01)` pi_lock_c's start-up dropped rings 4-7 to their cold branch (-1228 to -1294 pm; the review
measured ring 6 still at -761 pm at 20 ms). Now it holds every ring through the traffic start (115-118 pm at 20 ms;
engine review, engine-3).

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
optics walk runs inside it, in `layout.bus_order` with the bus segment loss between rings, `mrm.plant.phase`
equation 6: g26 0.073 dB/mm over 50 um, 0.99916 per segment, 0.59 % less light at ring 7 than at ring 0); sensors of
the interval (monitor, error counter, temperature sensors, readback); Truth,
metrics and records of the state at k + 1. Every analog or converter delay lives in its leaf module; the engine owns
only the controller pipeline (07 P0-6).

**Substeps** (`substep_policy`, default `sim.substep_policy`, g26 `fixed`): `auto` takes power-of-two steps up to
`auto_max_ticks` (64) while the actuators are quiescent, the scenario is linear, the detuning rate is small and no
edge, record window or oracle tick intervenes; a scenario event forces a 1-tick step and h at most doubles per step.
The thermal step is exact for any h; per-tick sensors still run every tick. Details: engine.md section 3 item 10.
The end of a kernel block (one `Plant.run` call) also caps the substep in flight, and the growth limit restarts from
the shortened step, so under `auto` the result depends on where the blocks end. A substep never crosses a
controller edge, so a block that ends on an edge changes nothing: the runner ends every block of a run without tick
jitter on a controller edge (`runner._block_end`), and `chunk_ticks` (default 1e5, recorded in
`manifest_inputs["chunk_ticks"]`) then changes no result, on the compiled and the open-loop path (the Python path
stops at every edge anyway). With `tick_jitter_s` > 0 the edges differ per world: there `chunk_ticks` and any block
end change the auto step sequence (another valid sequence of the policy, within its own error). Before 2026-10-16
the blocks ended every `chunk_ticks` ticks: g26 nominal, pi_lock_c, 20 ms, chunk_ticks 13 against 1e5 gave 11426
against 10582 substeps and monitor codes up to 9 apart (engine review, engine-4). Calling `Plant.run` directly
with blocks that do not end on edges still gives a block-dependent auto trajectory.

## 5. Controllers on the engine

- **Python path** (any object with `reset(spec)`, `step(obs) -> {channel: codes}`, `latency_ticks`): the kernel
  stops at every controller edge, the runner builds an `Observation` from the released channels, validates the
  actions (shape, dtype, range: 07 P1-19) and resumes. Vectorized over B; needs edges shared by every world (no
  `tick_jitter_s`). Cost: ~100-300 us per edge on top of the plant. A CompiledController run here (`path='python'`)
  after a scenario `fault.controller_reset` restarts as in the kernel: its new init state goes into the kernel's
  state rows (its `fast_kernel` and its `step` share one state again) and its held command is re-filled from the
  latched DAC registers (`Plant.restart_compiled`; before 2026-10-16 the fast kernel kept updating the old state,
  so bias_dither's lock-in stopped after a reset on this path).
- **Range errors.** `controller_interface.on_out_of_range: error` raises `ActionRangeError` on both paths: the
  Python path at the step (ActionValidator), the compiled path after the kernel block in which the first command
  left its range (the kernel still clips; it records each world's first out-of-range element, tick and value,
  `kernel.WI_SAT_FIRST_*`; the message names them). Before 2026-10-16 the compiled path always clipped and counted.
  `clip` (the default) saturates and counts on both paths (`diagnostics["commands_clipped"]`,
  `saturation_counts`).
- **Schedule** (`controller_interface.schedule`, 07 P2-7). `simultaneous` (default): every ring at every edge.
  `round_robin` (one datapath time-multiplexed over the rings): controller edge j (counted from the controller
  phase, `ControllerSpec.active_ring(tick)`) updates ring j % N only. The observation carries it
  (`Observation.active`, ABI header `OBS_ACTIVE_RING`); of every per-ring (B, N) actuator channel only that ring's
  command is applied and the other rings keep their held command of the edge (also through a controller reset);
  (B, L) and (B, 1) channels are applied whole. Both paths, the NumPy reference and MRMEnv (the kernel applies the
  rule when it resumes) do this; the Python path merges before validating, so only applied commands are
  range-checked and counted (`mrm.control.api.merge_round_robin`). Each ring is then updated at rate / N (g26:
  12.5 kHz per ring at 100 kHz). Before 2026-10-16 the setting was stored and ignored: every ring was updated at
  every edge. A `CompositeController` refuses the interface round robin (it multiplexes its children with its own
  `schedule`); `kalman_lqg` / `mpc` refuse it (their design assumes every ring's command acts each period).
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
| fixed-point reference (hardware task) | `fxp_pi` (mrm.control.fxp.FxpPi, compiled) | at the lock (settled warm start); it holds the operating point it starts at | nothing; g26-nominal gains (section 12) |

docs/handoff/controller_scorecard.md has every one of them on the engine (g26 nominal, steady lock and A1, from
`tests/validation/test_controllers_on_engine.py`), with the conditions. In short (as regenerated 2026-10-16 20:02,
on the plant of the engine, controllers and params reviews and the feedforward_pi fix; the steady lock also needs
a stationary operating point, drift <= 0.1 pm/ms): from the settled warm start, pi_lock (Python, compiled,
composite), feedforward_pi, kalman_lqg and mpc (Python and compiled) hold all 8 rings with <= 0.1 pm rms jitter
(each compiled version gives the same metrics as its Python version), bias_dither and midranging hold all 8; the
dither locks jitter at the 5 pm level by their own dither (dither_lockin 4 of 8 within 5 pm). The peak seekers end
at the drop peak, the wrong point for a modulator (the peak sits at the self-heating fold of a g26 ring, bistability
ratio ~6): es_po's default probes cross the fold only marginally (on this plant rings 4-7 fall once before 40 ms
and recapture, so 4 of 8 count as locked at 40 ms and 4 still settle; every ring parks at the peak at 13.0-13.2 dB
OMA penalty, runs/params_fix/es_long.log), and es_sine's 8 rings are still drifting toward the peak at 0.77-0.95
pm/ms at 40 ms; it parks there after 100-160 ms at ~13 dB (before 2026-10-16 the scorecard counted es_sine as 8 of 8
locked at 0.3 dB: a snapshot of that transient). The acquisition controllers without the supervisor lock every ring
on a foreign line (acquire_then_lock 4 of 8 within 5 pm and 4 still settling; bitstat_lock 8 of 8). These outcomes
flip with small plant changes (the plant fixes of 2026-10-16 moved es_po, midranging, acquire_then_lock, bitstat_lock
and wdm_supervisor; es_po's falls come and go with a 2 % change of the droop or of s(T) at the lock): they are not
robust results either way. wdm_supervisor locks all 8 rings from a cold start on the lines it assigns (a barrel
shift: ring 2 on line 0, tx.assignment). Under A1 (default site, 55 um from PD0), pi_lock, midranging, kalman_lqg
and mpc hold rings 1-7 within 12.5-15.4 pm (pi_lock: ring 4 14.7 pm, ring 5 6.4 pm, the others within 2.5 pm),
while ring 0 leaves its point by 454-473 pm at the hot cusp with its heater at code 0 for
14.7-16.4 ms (heater authority; 347-362 pm before the params review: the A1 target is a resonance trajectory and
the model's s at the lock is 2.1 % lower, so the same pm swing is a 2.1 % larger ring rise for the heater to cancel);
feedforward_pi holds every ring as pi_lock does (its die sensor sees the aggressor, ~+20 K;
until the gain-schedule fix of 2026-10-16 its feed-forward ran away to code 0 and it lost every ring;
mrm/control/baselines/feedforward.py). In the stress scenario
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
  records, diagnostics, `warm_start` (codes, protocol, settle report, factory table, `mode`, `no_code` (B, N),
  `order_hopped` (B, N), `supply_limited` (B,)), `score_from_tick` / `score_from_s`, and a `compression_note` for a
  time-compressed scenario.

`quasi_static(...) -> QuasiStaticResult` (07 P0-10): per (ambient, bias, world, ring) the oracle lock detuning,
relOMA (stable and unrestricted), ring temperature, absorbed and heater power, heater code, code-range coverage,
FSR-hop and wrong-line risk; per (ambient, bias, world) the array supply droop factor `supply_scale` and
`supply_limited`; with `order="reachable"` the rings moved one order bluer (`order_hopped`); no time stepping
(~0.1 s for one point of g26).

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
  controllers, also under `NUMBA_DISABLE_JIT=1`, with the config scenario and with one that drives every scenario
  input the kernel reads (aggressor, laser failover with a dark gap, fibre loss, traffic start and stop, bias step,
  the four faults and a controller reset; `test_parity_with_every_scenario_path`). The reference resets the
  controller at the first edge after a `fault.controller_reset` rising edge (since 2026-10-16; before it ignored
  the fault). With `airy=False` (independent optics, above) the parity is within
  1e-9 of each Truth field's range, codes identical.
- Truth detuning: the resonance of the order nearest the declared line (tx.assignment when the controller owns it),
  computed exactly on the phase model's cold curve (`kernel.resonance_detuning`); the optics slope is
  `dphi_dlamres = 2 pi c t_rt / (lambda - shift)^2` (engine.md section 3 item 6). `detuning_hwhm` divides by
  `lambda_line^2 fwhm_hz / (2 c)`; the rigidly translated line of the phase model keeps the cold width
  `lambda_res^2 fwhm_hz / c` in wavelength, 1.2 % narrower at the 7.6 nm g26 lock (a normalisation of the
  diagnostic only; section 10, phase model).

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
- **Bus segment loss (since 2026-10-16).** Each ring after the first in bus order sees its lines 0.99916 times the
  previous ring's through output (g26: 0.073 dB/mm, printed, over the assumed 50 um spacing; ring 7 gets 0.59 % less
  light than ring 0). Before, `Plant._build` passed no `bus_t2` and the engine, `quasi_static` and
  `mrm.control.models` ran every segment at 1, while derive and the replicas applied the loss.
- **Crosstalk dynamics: two packages (since 2026-10-16).** The interim crosstalk pole takes the heat-source-to-victim
  distance of its imec anchor, the centre distance less both ring radii (r_ref): the g26 nearest neighbour gets
  1.52 ms at the 50 um pitch (2.85 ms before, with the centre distance), the own monitor PD 0.38 ms (0.71 ms), the
  A1/A2 aggressor path to ring 0 0.62 ms (1.03 ms). These are imec's natural-convection numbers (3 % nearest, 1.5 %
  floor, 1.5-9 ms). The in-ring FEM ROM's TEC-held package gives 1.0 % nearest reaching 63 % in 0.12 ms and a 0.09 %
  floor at 1 mm; that is the weight-0 choice `uncertainty.thermal.stack: side_by_side_tec_held`, corner
  `g26_tec_package` (in the named corners, so robust and compare runs include it; sampled worlds do not). Which one
  G26 has is unknown (VALIDATION.md closure-crosstalk-dynamics). The die sensor `die0` sees each heater with the
  far-site floor (`ring.thermal.site_floor_frac`): 1.5 % of Z_heater->ring (85 K/W, 9 ms) by default, 0.09 % in
  the corner.
- **Monitor ADC reference at the package base (since 2026-10-16).** The 20 ppm/K reference tempco
  (`ring.monitor.adc_ref_tempco_ppm_per_k`, assumed) acts at `ring.monitor.adc_ref_site`: `package_base` (default;
  `thermal.t_base_k + thermal.xpu_rise_k`, the converter on its own die beside the PIC, 07 P1-3 side_by_side) or
  `monitor_pd` (each ring's PD observer, the former model; kernel row `RF_T_ADC`). At the g26 lock the PD observers
  sit ~20 K above the base, so the codes read 17-24 codes (0.04 %) higher than before, and ring 0 under A1 (PD0 at
  ~93 degC) loses its former 0.14 % gain dip. At 25 degC ambient the reference sits at its trim temperature, where
  the plant matches `mrm.control.models` (which uses T_ref) exactly. Heater and aggressor heat no longer reach the
  ADC gain.
- **Bit statistics under a PRBS (since 2026-10-16).** The `datapath.bitstat_above` window takes the scenario's ones
  density (kernel row `RF_DENS`, also set at reset for an edge at k0), so under `fidelity.pattern_noise` prbs7 / 9 /
  15 / 23 its ones count is the exact count of the last 64 bits sent (`RI_BS_N1`). Before, the realized per-tick
  density n1/nb (0.504 for prbs7) made every window a Binomial(64, n1/nb) draw (std 4.1 instead of 2.8 for prbs7,
  uncorrelated with the sequence). Binomial mode draws Binomial(64, density). The window's levels at the first edge
  after a reset are still the reset placeholders (no interval has run yet).
- **adc_mux (since 2026-10-16).** `round_robin` with `channels_per_adc` C > 1 needs C x window <= ADC period (the C
  rings take turns on one integrating converter; checked by realize and `MonitorHw.build`), and the C rings share the
  converter's DNL thresholds and INL bow (keyed by converter n // C). The conversion (de-integration) time is not
  modelled: a shared dual-slope converter also needs C (W + T_deint) <= P, which the preset's rate must respect.
- **ROM mode: die-level time scales per pair (since 2026-10-16).** With `fidelity.thermal_model: rom` and unequal
  `time_scale_fast` / `time_scale_slow`, a mixed-sign FEM pair whose poles straddle 300 us is time-scaled as a whole
  (`mrm.plant.thermal._whole_pair_time_scales`); per-pole factors made heater -> ring overshoot by up to 21 %
  (metal_above_160). The interim model is unchanged.
- **The phase model is a rigid comb** (open item, `mrm.plant.phase` equation 2): a heated ring keeps its cold FSR
  in wavelength, while the modes artefact's n_eff(lambda, T) table moves order m + k by 0.43 % of the shift per
  FSR more or less (+-33 pm at the adjacent orders at the 7.6 nm lock, linear in the heater shift). The locked order
  is exact; the foreign-line heat and monitor background move by <= 0.15 % of P_abs in g26 and ~1 % in the FSR
  stress corners. The 07 P1-6 law needs a thermo-optic dispersion key and changes every consumer of the curve.
- **The array heater supply limit binds in some worlds** (section 3): the stacked_eic stack below ~30 degC, the
  g26_low_eta and g26_fsr_edge corners at 0 degC. There no controller can hold the whole bank; LHS quantiles that
  include such worlds measure the supply, not the controller (`warm_start_supply_limited` in the world summaries).
- **Sampled worlds lose rings under pi_lock_c even from a good warm start.** experiments/examples/mc_pi_lock.yaml
  (64 sampled worlds, static, 20 ms, lock judged about each ring's own operating point): 27 worlds have a ring off by
  more than 5 pm, 10 of them with no warm-start flag (no code, order hop or supply limit; worlds 1, 3, 11, 13, 30,
  32, 33, 39, 43, 48); median worst-lane OMA penalty 1.6 dB, p95 31 dB (rerun 2026-10-16 after the iterated order
  hop: 1 ring without a code instead of 9, 101 hopped rings, 2 supply-limited worlds; runs/engine_fix/exp). A controller claim over the uncertainty set needs the controllers owners' analysis of
  those worlds (docs/handoff/experiments.md section 12).
- Acceptance table on g26 nominal with pi_lock_c (docs/handoff/experiments.md section 12): A4 (soak), A7 (relOMA
  plane), B3 (XPU step) and B5 (cold start, wdm_supervisor) pass; A1, A2 (heater authority at the hot end, ring 0),
  A3 (14 pm lag under the 50 nm/s sweep), A6, B2 (set-point moves under power steps) and B1 (squelch) fail.
- Not built yet: the P1-20 telemetry metrics, `fidelity.data_modulation: lut` (the waveform LUT exists:
  docs/handoff/waveform.md section 3), the reference-tap and `datapath.bitstat_ones` channels (a config with
  `ring.monitor.reference_tap.present: true` is refused when the plant is built), `DacFacts.law`, one
  knowledge vocabulary, `run_lock_request` for the replicas (docs/handoff/replicas.md section 6), the Verilator /
  cocotb closed-loop HDL co-simulation (section 12), an in-kernel bridge (bridged runs take the Python path),
  per-world reset in the gym env, and the 07 P1-22 throughput (section 13).

## 11. Experiments, run directories and the CLI (07 P1-17, P0-10, P2-12, 4.8)

An experiment file is an mrm.params config (`meta.kind: experiment`) that `extends:` a preset and adds
`controller`, `scenario`, `metrics` and `experiment` sections. Every `experiment.*` key is a schema key (keys and
defaults: docs/handoff/experiments.md section 3; PARAMETERS_FORMAT.md appendix). Example (abridged from
experiments/acceptance/a1_slide10_aggressor.yaml):

```yaml
extends: g26
meta: {name: acc_a1_slide10_aggressor, kind: experiment}
controller: {name: pi_lock_c}
scenario: {name: a1, params: {phase_s: 0.06535}}
experiment:
  kind: acceptance                 # run | sweep | mc | compare | robust | acceptance | quasi_static
  duration_s: 0.1357
  seeds: {device: 0, uncertainty: 0, noise: 0, scenario: 0}
  start: warm                      # cold | cold_settled | warm (quasi_static_reachable codes, settled)
  score_from_s: 0.005              # leave pi_lock's peak map out of the metrics
  self_reference: {duration_s: 0.02, window_s: 0.005}   # judge the lock about each ring's own operating point
  acceptance:
    - {id: A1-lock, metric: max_abs_lock_error_pm, op: "<=", value: 5.0, source: "04 section 4"}
```

- **Commands.** `mrm run EXP [--set PATH=VALUE] [--alt PATH=NAME] [--threads N] [--runs-dir DIR] [--dry-run]`;
  `mrm sweep | mc | compare | robust EXP` force the kind; `mrm acceptance [EXP ...] [--all]` prints a pass/fail
  table (exit 1 when a criterion fails); `mrm rerun runs/<id>` reruns from the directory and compares every array of
  trace.h5 bit for bit. Python: `mrm.run(path_or_config, overrides=..., options=..., runs_dir=..., write=...)`.
  `--dry-run` prints B, ticks, edges, RAM and wall time per case from the measured engine speeds (Python controllers
  vary ~10x per edge; the soak's auto policy runs at ~6.5-7.8 us per tick).
- **Kinds.** run (one runner.run of `batch` worlds, default records); sweep (Cartesian `experiment.grid` over config
  paths, `alt:<path>`, `corner`, `controller`); mc (`n_mc` worlds, metrics only, chunks of `max_worlds_per_plant`);
  compare / robust (07 4.8: nominal world, the named corners (07's 12 and g26_tec_package), `n_mc` LHS worlds;
  worst corner, quantiles, Kendall-tau rank stability; the first report line is the Q1 caveat); acceptance (criteria
  that must hold in every world of every case); quasi_static (runner.quasi_static over ambient x bias, plus one
  survey per corner).
- **Run directory** `runs/<UTC>_<hash8>/`: `experiment.resolved.yaml` (every leaf with provenance), `manifest.json`
  (schema mrm.run/1: git commit, branch, dirty-diff hash, environment, argv, seeds, source chain, artefact hashes,
  every case's runner manifest), `metrics.json`, `trace.h5` (exact arrays), `inputs/` (the experiment file, its
  non-preset parents and its corner files given by path, copied when loaded: the directory reruns, from any working
  directory, without the original files), `report.txt`, `convergence.json` (compressed runs; a rerun takes a reused
  certificate from it). hash8 is the first 8 hex of the experiment hash (config without the experiment-layer keys,
  plus the resolved options and, when the experiment runs corners, each corner's alts: an edit of a corner preset
  makes `mrm rerun` refuse).
- **Time compression (07 P0-10).** A compressed scenario (`scenario.time_compression` > 1, the soak) runs only with a
  convergence certificate: the convergence experiment (one slow cycle at `compression_factors`, default 1, 10, 100,
  on the same plant, controller, start and worlds) runs first, or `experiment.convergence_certificate` names a run
  directory whose certificate covers the plant fingerprint (resolved config without the experiment section and the
  compression, plus start and worlds) and the code fingerprint (`mrm.sim.manifest.code_fingerprint`: the sources of
  mrm/{core, plant, sim, control, scenarios, params}, mrm/units.py, mrm/materials.py, every file under mrm/data and
  the numerics versions; since 2026-10-16). The factors must agree per world on max detuning (ring by ring: 0.2 pm or
  5 %, at most 0.5 pm), wrong-line events, FSR-hop events and order changes, heater range used and integrated
  errors. The A4 certificate of 2026-10-10 (`runs/acceptance_20261010/20261010T115721Z_d1511028`) no longer covers
  the tree (the g26 plant changed on 2026-10-16 and it records no code fingerprint): A4 runs its convergence
  experiment again.
- **Judging a controller** (07 4.8): world summaries (`mrm.sim.experiment.world_summary`): worst-lane and mean OMA
  penalty against the oracle, excursions (count, longest), time unlocked, rings on a foreign line, time to lock,
  absolute relOMA, BER, heater rails, FSR hops (`fsr_hop_events`: re-locks on another order; `order_changes`:
  changes of the order nearest the declared line, lock-independent, the one to judge a hop by, since |detuning|
  <= FSR/2 holds by construction) and the warm-start flags. A claim holds only if it holds in every world.

## 12. External controllers: the binary bridge, golden vectors and golden traces (07 P1-21, 4.9)

- **Bridge** (`mrm.sim.bridge`, protocol 1): an external controller (Python, C, an RTL simulator or hardware behind a
  socket) runs in lockstep. At every controller edge the plant sends one binary OBS frame (the edge tick and the
  framed channels of every world) and waits for one ACT frame; frames carry ticks, never wall time, so a bridged run
  is bit-identical to the in-process run (tested for Python and C peers). The frame layout is generated from the
  ControllerSpec (profiles `minimal`: monitor, temperature sensors, latched registers, flags -> heater (+ bias);
  `full`: every exposed sensor and bound actuator); the handshake checks the protocol version, the layout and
  ControllerSpec fingerprints and the latency.

  ```python
  from mrm.sim import bridge, runner
  peer = bridge.spawn([".venv/bin/python", "-m", "mrm.sim.bridge", "serve", "--preset", "g26",
                       "--controller", "mrm.control.fxp:FxpPi"])          # or bridge.connect(host, port)
  with bridge.BridgeController(peer, profile="minimal") as ctrl:
      res = runner.run("g26", ctrl, None, duration_s=10e-3, uncertainty="nominal", variation="off",
                       heater_code="quasi_static", settle=True)
  ```

  `mrm bridge c-header --preset g26 --profile minimal` writes `mrm_bridge.h` (with the spec fingerprint);
  `mrm bridge c-example` writes the C twin of FxpPi. A bridged run takes the runner's Python path (~28 us per tick
  of overhead plus the round trip, 11-14 us for a C peer over pipes); `fast_kernel` and Truth-fed controllers are
  refused; over TCP pass `connect(..., timeout_s=)` so a silent peer cannot block a run.
- **Fixed-point reference PI** `FxpPi` (`mrm.control.fxp`, registry name `fxp_pi`): positional PI on
  `tx.monitor.latest` in Q-format integers (gains Q3.20, accumulator Q19.20 by default), the twin of the C and
  Verilog versions. It holds an operating point (start it at a lock); g26-nominal gains, not a tuned controller.
- **Golden vectors** (`mrm.sim.vectors`, format mrm.vectors/1): `mrm vectors record` runs a controller and writes
  the flat int64 ABI streams (`.npz`, `$readmemh` `.hex`, Verilog `.layout.vh`); `mrm vectors replay` / `replay_abi`
  replay them through a compiled controller without a plant (the RTL test-bench view). tests/hdl replays the golden
  vectors through a synthesizable Verilog PI under Icarus. The Verilator / cocotb closed-loop co-simulation of 07
  P1-21 is not built (tools not installed).
- **Golden traces** (`mrm golden`, tests/golden, format mrm.golden/1): the metrics of fixed runs that block CI:
  `g26__static` and `g26__a1_slide10_aggressor` (open loop, pi_lock_c, fxp_pi) and `g26__soak_compressed` (one soak
  cycle at compression 100, pi_lock_c: the certified c = 100 case of the A4 convergence experiment; slow test). A
  change of any engine number fails them; update only with `mrm golden --update --case ... --reason "..."` and the
  printed CHANGELOG line. They were written on arm64 macOS with the .venv versions recorded in each file; another
  platform can follow another valid closed-loop trajectory.

## 13. The RL environment (07 P1-22)

`mrm.sim.gym_env.MRMEnv` is a gymnasium VectorEnv whose B sub-environments are the B worlds of one Plant (no
subprocess workers); `MRMSingleEnv` is the B = 1 view; `gymnasium.make("mrm/MRM-v0")` and
`gymnasium.make_vec("mrm/MRM-v0", num_envs=..., vectorization_mode="vector_entry_point")` work.

```python
from mrm.sim.gym_env import MRMEnv
env = MRMEnv("g26", num_envs=256)                     # warm start, sampled worlds, 1000-step episodes
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
info["truth"]["detuning_m"]                            # Truth only in info, never in obs
policy = env.compiled_policy(mlp)                      # mlp = mrm.control.fxp_mlp.quantize_mlp(...): deploy in-kernel
```

- One step = one controller period (g26: 10 ticks) on the engine's Python-controller path; the observation is the
  last K monitor sample blocks, the applied heater (and bias) codes and the temperature codes, each scaled by a
  power of two (exact in float32 and Q0.15); the action is a bounded delta of heater (and bias, when bound to the
  controller) codes. The reward comes from the kernel's metric accumulators (lock error about the oracle point by
  default, optional relOMA, heater power and wrong-line terms: `RewardSpec`).
- Every reset redraws the worlds (new seeds; `options={'seeds': ...}` pins them); all worlds truncate together. The
  warm start uses the quasi-static codes (`start='warm'`; rings without a code are flagged in
  `info['truth']['warm_start_no_code']`); 'cold' and 'cold_settled' are the other starts.
- `MRMEnv.compiled_policy(mlp)` turns a fixed-point MLP (`mrm.control.fxp_mlp`) into a CompiledController that runs
  inside the kernel bit-identically to the env; `fxp_mlp.error_bound` bounds |fixed - float|.
- Throughput (8-core M2): 2.5-2.6e4 env steps/s at B = 256, not the 1e5 of 07 P1-22. The plant itself caps it at
  ~4e4 (2.3-2.5 us per world-tick), and the Python path's per-call setup costs ~35 % (docs/handoff/rl.md section 4).
  A reset at B = 256 takes 3-5 s (quasi_static is a Python loop over worlds). examples/tutorial.ipynb shows the
  API and the quantize-and-deploy path (no RL library is installed, so it trains nothing).

## 14. Calibration and the validation report (07 P0-5, P1-14, P1-23, 4.3, 4.9)

Two presets are generated from the calibration fit and must not be edited by hand: `g26_calibrated` (g26 with the
leaves fitted to the calibration targets) and `g25_test_vehicle_calibrated` (the G25 bench, `g25_test_vehicle`, with
the same calibrated device leaves; the plant of the G25 calibration targets after the fit). Both load like any
preset (`params.load("g26_calibrated")`, `extends: g26_calibrated` in an experiment file). **The default preset is
still g26** (the calibrated preset is not adopted yet; docs/handoff/calibration.md section 11 gives the
recommendation and its reasons). What each calibrated leaf means: its `targets`, `residual`, `identifiable` and
`note` fields (`mrm params explain g26_calibrated ring.heater.r_cold_ohm`); a leaf is `calibrated` only when the
07 4.3 half-prior rule holds; a value that moved without passing the rule keeps its provenance.

Every command runs under the memory guard (`MG=".venv/bin/python tools/memguard.py --"`; `--heavy` above 1 GB):

```
$MG .venv/bin/python -m mrm.calibration                      # every fit group (~5 h serial), then both presets
                                                              # and the before/after replica table
$MG .venv/bin/python -m mrm.calibration --groups dac laser   # some groups; the other groups' leaves and records come
                                                              # from mrm/calibration/latest_fit.json (rerun every later
                                                              # group when an earlier group's leaves change)
$MG .venv/bin/python -m mrm.calibration --write-preset-only  # regenerate both presets from latest_fit.json
$MG .venv/bin/python -m mrm.calibration --before-after-only  # only the before/after table (~1 min)
$MG .venv/bin/python -m pytest -m calibration tests/calibration   # fit residuals and preset checks (not CI-blocking)
```

`mrm calibrate` is the same command. The calibration reads only `targets/calibration.yaml` (the isolation test
scans `mrm/calibration/`); held-out targets are never fitted. The fit record `mrm/calibration/latest_fit.json`
(schema `mrm.calibration.fit/1`) holds per group the hypotheses, fits, profile likelihoods and leaf decisions, the
`context` (the written leaves, which later groups and `--groups` reruns build on) and the before/after table. Method,
results and deviations: docs/handoff/calibration.md.

The validation report `docs/VALIDATION.md` (g26) is generated by `mrm report` (`python -m mrm.validation.report`)
from cached computed parts in `docs/validation/<preset>/report_data.json` plus live files (presets, target files,
the fit record):

```
$MG .venv/bin/python -m mrm.validation.report --check        # exit 1 when the report is stale (the CI test does this)
$MG .venv/bin/python -m mrm.validation.report --fast         # re-render from the cache (~5 s): after a new fit record,
                                                              # a new commit, or a change of the rendering code
$MG .venv/bin/python -m mrm.validation.report --recompute predictions,identifiability   # stale parts only
$MG --heavy .venv/bin/python -m mrm.validation.report        # every part (~76-80 min, peak ~1.2 GB)
$MG --heavy .venv/bin/python -m mrm.validation.report --preset g26_calibrated --out docs/VALIDATION_g26_calibrated.md
```

Parts and their cost on g26 (4 kernel threads): predictions ~9 min (every target on the nominal world and the
named corners), identifiability ~20 min (the calibration Jacobian), sensitivity ~23-25 min (tornado and Sobol worlds
under A1), ranking ~28 min (the compare_locks experiment). A part goes stale when a preset it ran on, a target file,
or its code (`report.CODE_DEPS`: replicas.py, sensitivity.py) changes; the CI test
`tests/verification/test_report.py::test_validation_md_is_current` then names the part to recompute. A new fit
record changes only the rendering (section 3.1): `--fast` is enough. After a commit, run `--fast` once so the stamp
names the new commit (it recomputes nothing). The G25 bench targets of a preset `g26_X` run on
`g25_test_vehicle_X` when it exists (g26_calibrated: g25_test_vehicle_calibrated). The order of a recalibration:
fit, write the presets, then recompute the calibrated report; never change a calibration after looking at its
held-out rows.
