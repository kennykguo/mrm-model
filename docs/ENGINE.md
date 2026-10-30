# The engine: how it works and how to use it

The engine runs the physical model and its controller together. It advances a batch of **B worlds**
(one device-parameter choice per world) in 1 µs plant ticks, calls the controller at its own rate,
and returns metrics, recorded traces and the internal physical state, called **Truth**.
The main calculation runs in a compiled Numba kernel.

For a first run, start with section 1. Sections 2–6 explain inputs, startup, timing, controllers and
results. Sections 7–10 cover implementation, extensions, performance and limitations. Sections 11–14
cover experiment files, external controllers, the learning archive and calibration.

Throughout this guide, **N** is the number of rings, **L** the number of laser lines, **B** the number
of worlds and **S** the number of recorded samples. A *controller edge* is a tick when the controller
updates. A *block* is one kernel call covering a chunk of simulated time. A *substep* is one thermal
integration step, which can span several plant ticks under the automatic policy.

### Requirements and references

The engine implements these requirements of the design review
([research/07_design_critique.md](research/07_design_critique.md), which also holds the design history):

- Self-heating is on by default and is solved implicitly in every ring.
- Time is counted in integer ticks, each delay has one owner, and the events of a tick have a fixed order.
- The controller reads and writes named channels (heater and bias actuators, ADC sample blocks).
- Metrics are computed online in the plant loop, and traces are stored as min/max envelopes.
- The closed loop runs in a compiled kernel, with a stated compute budget (section 9 says which budgets are met).
- Long runs use exact multi-tick substeps, a quasi-static mode (now `solve_locked_steady_state`, section 6) and
  validated time compression.
- A pure-NumPy reference of the tick gives the same results bit for bit.

This guide replaces `docs/handoff/ENGINE_BRIEF.md`. That brief remains useful for its tables of
physics-kernel signatures and `PlantParams` consumers in sections 2, 4 and 5. Design decisions and
review records are in [handoff/engine.md](handoff/engine.md); contract changes are in
[CHANGELOG.md](CHANGELOG.md).

### Where the code lives

| file | role |
|---|---|
| `mrm/sim/runner.py` | `run`, `solve_locked_steady_state`, `load_inputs`, `warm_start_codes`: the user entry points |
| `mrm/plant/plant.py` | `Plant` (builds the kernel arrays, reset, settle, run, advance, Truth), `Truth`, `prepare_scenario` |
| `mrm/sim/kernel.py` | the compiled per-world tick (`get_run_block`), Truth row, oracle, auto substeps |
| `mrm/sim/state_spec.py` | the registry of every runtime value: KState arrays and block rows, KParams fields, with meaning, unit, shape, kind, writer and inputs (the kernel takes its layout from it) |
| `tools/gen_variable_map.py` | generates docs/VARIABLES.md and the viewer data docs/variables/data.json and symbols.json from the registry and the preset |
| `mrm/sim/kernel_ref.py` | `ReferenceEngine`: the same tick in NumPy, bit for bit against the kernel |
| `mrm/sim/metrics.py` | `MetricsSpec`, streaming accumulators, `RecordSpec`, `summarize`, `TRUTH_FIELDS` |
| `mrm/plant/io.py` | channel maps, controller pipeline FIFO, observation packing, action clipping |
| leaf modules | `mrm/plant/{phase,optics,thermal,selfheat,electrical,sensing,datapath,laser}.py`, `mrm/scenarios` |
| `mrm/sim/experiment.py`, `manifest.py`, `cli.py` | experiment files, run directories, `mrm run ...` (section 11) |
| `mrm/sim/bridge.py`, `vectors.py`, `mrm/control/cli.py` | external controllers, golden vectors and traces (section 12) |
| `archive/rl/` | optional learning tools, excluded from the installed package (section 13) |
| `mrm/calibration/`, `mrm/validation/report.py` | the calibration fits, the calibrated presets and the generated validation report (section 14) |

### Explore the variable map

[VARIABLES.md](VARIABLES.md) lists model values in tables. The interactive view in
`docs/variables/index.html` shows the same information as a graph, from preset constants to values
that change every tick. Serve it locally so it can load its data file:

    cd docs/variables && python3 -m http.server 8000      # then open http://localhost:8000/

Click a box to see its meaning, unit, value or formula, writer, inputs and dependants. Double-click
a group to open it. Append `#ring_float.t_ring` to the URL to open that value directly.

After changing `mrm/sim/state_spec.py`, a preset, a relation or the channel table, regenerate both
the tables and viewer data with `.venv/bin/python tools/gen_variable_map.py`.
`tests/verification/test_variable_map.py` fails until they are current.

## 1. Quick start

Use `runner.run` for a simulation from Python, or `solve_locked_steady_state` to find locked operating
points without stepping through time. The examples below show both. Scenario objects keep their
own signal defaults; see section 2 before comparing them with experiment-file runs.

```python
from mrm import params
from mrm.sim import runner
from mrm.sim.metrics import RecordSpec, envelope
from mrm.scenarios import library

cfg = params.load("g26")                                    # preset (or a YAML path, overrides=, alts=)

# open loop, nominal world, 2 ms, record the detuning every 10 us (mean and min/max envelope)
r = runner.run(cfg, None, library.static(), duration_s=2e-3, uncertainty="nominal", variation="off",
               record=[RecordSpec("detuning_m", 10, "mean"), *envelope("detuning_m", 10)])
r.trace["detuning_m@10:mean"]["value"]                      # (S, B, N), metres
r.metrics["max_abs_detuning_pm"]                             # (B, N)

# a registered controller from a settled warm start at the oracle lock codes, under the A1 aggressor
r = runner.run(cfg, "pi_lock", library.a1_slide10_aggressor(phase_s=library.A1_PERIOD_S / 2), duration_s=0.1307,
               uncertainty="nominal", variation="off", heater_code="locked_steady_state", settle=True,
               metrics={"lock_reference": "oracle"})

# Monte Carlo: 64 sampled worlds on 8 threads (compiled controllers run inside the kernel); rings without a
# steady-state lock code (unreachable oracle point) start at code_min with a warning
r = runner.run(cfg, "pi_lock_c", None, duration_s=10e-3, batch=64, seeds=1000, n_threads=8,
               heater_code="locked_steady_state", settle=True)

# the locked steady state over ambient and bias (BND-4), no time stepping
qs = runner.solve_locked_steady_state(cfg, ambient_c=[0, 25, 105], bias_v=[1.05, 2.0], uncertainty="nominal",
                                      variation="off")
qs.heater_code                                               # (A, V, B, N)
```

An experiment file bundles the config, controller, scenario, worlds, recordings and criteria
(section 11):

```
mrm run experiments/examples/a1_pi_lock.yaml            # writes runs/<UTC>_<hash8>/; --dry-run estimates first
mrm acceptance --all                                     # the acceptance table (A4 alone takes hours)
```

You can also run an experiment file through the top-level Python API:

```python
import mrm
res = mrm.run("experiments/examples/a1_pi_lock.yaml")   # or mrm.run(mrm.params.load(path))
res.report, res.metrics, res.run_dir
```

### First-run compilation

The first call in a fresh environment compiles the kernel. Since performance round 2, this takes
about 2 minutes, or 5–8 minutes under load or before those optimizations. Peak resident memory on
the shared M2 is about 2.3 GB. Run it through `tools/memguard.py --heavy`. Later processes load
Numba's disk cache in about a second.

### Clear the cache after editing compiled code

**Numba may silently reuse old code when a called function changes.** Its cache key checks only
the cached function's own source file (`numba/core/caching.py`, `get_source_stamp`: `_py_file`
modification time and size).

Editing `mrm/sim/kernel.py` therefore recompiles the kernel. Editing an `njit` function it calls in
`mrm/sim/metrics.py`, `mrm/plant/*.py`, `mrm/scenarios/generators.py` or `mrm/core/*.py` does not.
After such an edit, clear the kernel cache before running:

    .venv/bin/python tools/clear_numba_cache.py [module ...]

Without module arguments, the tool clears the kernel and every physics/helper module it calls;
with arguments, it clears the named modules. Run it after every `njit` edit and before timing or
exact comparisons. Deleting `mrm/sim/__pycache__/kernel.*.nbi` and `kernel.*.nbc` is the manual kernel-only
equivalent. The contrary advice in `docs/handoff/engine.md` is outdated; see `docs/handoff/perf.md`.

## 2. Inputs

### Config, worlds and seeds

`run(cfg_or_params, ...)` accepts a `Config`, a preset name, a YAML path, or `PlantParams` accompanied
by `cfg=`. The `ControllerSpec` always comes from the **unsampled** config, so a controller never sees the sampled device of a world.

Choose the batch with these options:

| Option | Meaning |
|---|---|
| `batch` | Number of worlds, B |
| `uncertainty` | `'sample'`, `'lhs'` or `'nominal'`; defaults to `experiment.uncertainty` |
| `variation` | `'sample'` or `'off'` |
| `seeds` | One integer, or base seeds `{device, uncertainty, noise, scenario}` |

World b uses `base + b`. Changing the noise seed does not change the device draws.
Since 2026-10-16, each world also stops its settling process independently, so world b starts the
same way whether run alone or in a batch (section 3).

### Scenario and aggressor calibration

A scenario can be `None` (use the config's `scenario` section), a registry name, or a
`mrm.scenarios.generators.Scenario` built with `mrm.scenarios.library`.

Before compilation, `prepare_scenario` calibrates A1, A2 and `aggressor_direct` against the plant.
It finds the aggressor power that produces the specified victim-ring trajectory and records it in
`manifest_inputs["aggressor_calibration"]`. A1 starts at its hot cusp at t = 0;
use `phase_s = P/2` to start at the cool minimum.

The normal aggressor heats layout site `aggressor`: in G26, (0, −30) µm, 30 µm from ring 0 and
55 µm from its monitor photodiode (PD). The stress scenarios
`a1_slide10_aggressor_pd_adjacent` and `a2_808_aggressor_pd_adjacent`, with aliases
`a1_pd_adjacent` and `a2_pd_adjacent`, instead heat `aggressor_pd_adjacent` at (0, 30) µm,
only 5 µm from that PD. This layout tests dark-current runaway and is not considered realistic.
Both sites have the same distances to the rings and the same calibrated power, so only the
PD thermal paths differ.

### Scenario defaults: Python and experiment files can differ

**The config's signal defaults apply only when `scenario=None`.** In that case,
`mrm.scenarios.registry.from_config` applies `scenario.ambient_c`, `scenario.bias_v` and
`datapath.transition_density`.

A registry name or explicit `Scenario` object keeps its own defaults. In G26, that leaves bias
undriven at the bias DAC's initial code, 1.0 V, rather than the config's 1.05 V. This changes both
warm-start codes and trajectories: about 4 codes in G26 static, and up to 100 pm while
`pi_lock_c` maps its peak under the soak.

Experiment files always build the scenario from the config. To match them from Python, use
`scenario=None` with `scenario.*` overrides, or pass `registry.from_config(cfg)`.

### Controller

Pass `None` for open loop; a registry name with `controller_params=`; a `module:Class` path;
an object implementing the Controller protocol (Python path); or a `CompiledController`
(compiled path). Registry names are resolved by `mrm.control.registry.create`. See section 5.

### What the controller is allowed to know

A controller reads device parameters only through `spec.nominal`. This is a view of the unsampled
config filtered by the tokens in `controller_interface.knowledge`
(`mrm.control.knowledge.NominalView`). Calling `.get` or `.si` for an ungranted key raises
`KnowledgeError`.

The experiment section, including seeds and world choices, is never released. The specification
fingerprint is therefore the same for every seed and world choice, even when they are config leaves.
G26 grants only `line_grid` and `ring_to_line` by default.

Two token vocabularies currently coexist:

| Vocabulary | Tokens |
|---|---|
| Coarse, used by controllers B | `heater_nominal`, `optics_nominal`, `dlambda_dt_nominal`, `monitor_nominal` |
| Fine, used by controllers A | `heater_law`, `heater_r_nominal`, `heater_efficiency`, `linewidth`, `thermal_dynamics`, `bias_efficiency` (`pi_lock.KNOWLEDGE_KEYS`) |
| Model-based, used by controllers C | `model_nominal` |

`mrm.control.knowledge.token_keys()` merges the vocabularies.
`mrm.control.calibration.FIELD_TOKENS` maps both to the helpers' field groups.

`model_nominal` grants the whole nominal model except the experiment section and scenario settings
beyond the operating point: `scenario.ambient_c`, `bias_v` and `bias_offset_v` are retained.
`spec.nominal.model_config()` supplies this design to `mrm.control.models.linearize`.

The ampere scale of `tx.monitor.samples` and `tx.monitor.latest` is also protected knowledge.
It requires `monitor_nominal` or `model_nominal`; without either, the channel specifications carry
codes only. For example, grant model knowledge with:

    overrides=["controller_interface.knowledge=[line_grid,ring_to_line,model_nominal]"]

The filtering has applied at the view itself since 2026-10-16. Previously, `spec.nominal` exposed
the whole `Config`, including experiments, and only helper functions enforced the gate.
One issue remains: the view uses the run's own config, so a corner selected by `alts=` reaches
`model_nominal`. A separate `controller_knowledge` preset is not built; see
`docs/handoff/controllers_c.md` section 4.

### Exposed sensor channels

`controller_interface.expose` lists the sensor channels a controller may read. By default, every
channel is exposed. The restriction applies in the runner, bridge and vectors.

A hidden channel raises `UndeclaredChannelError` on the Python path and has `obs_offset = -1` in
the compiled ABI. Exposing a sample block also exposes its count channel. An unknown channel name
raises `ChannelError`. Before 2026-10-16, this setting was stored but never applied.

### Metrics and when scoring begins

`metrics=` accepts a `MetricsSpec` or a dictionary overriding the config's `metrics` section,
including `lock_threshold_pm`, `hysteresis_pm`, `allowed_assignments`, `lock_reference` and
`lock_target_pm`.

`score_from_s=t` restarts metric accumulators at t. Use it to exclude startup, such as the first
~3 ms of `pi_lock` peak mapping. `MetricsState.restart` preserves the oracle lock reference and
previous controller latch. On the Python path, a split on a controller edge moves one tick later.

With **fixed substeps**, the split leaves the plant, controller and records unchanged. With **auto
substeps**, there are two qualifications:

- Ending the kernel block at t also ends the current substep. This preserves the trajectory only
  when t is a controller edge: in G26, a multiple of 10 ticks after the phase. Otherwise the runner
  emits `runner.ScoreSplitWarning`. In a 20 ms check, an off-edge split changed 240 of 320 detuning
  samples by up to 4.6e-14 m and monitor readings by up to 9 codes.
- Restarted BER windows also cap substeps at their ends. `metrics.ber_sample_period_s` defaults to
  2 s. If the split is not a whole number of windows after reset, a run continuing beyond the first
  restarted window uses different substeps from that point onward.

The engine review corrected the earlier unconditional “untouched” claim on 2026-10-16.

## 3. Start states

A warm start has two parts: **the heater commands** and **the stored thermal state**. Setting only
the commands does not make a cold device thermally settled.

### Reset and heater-code choices

| Option | State at t = 0 |
|---|---|
| Default: `reset_protocol` from `sim.reset_protocol`, G26 `laser_ramp` | Heaters at `code_min`, cold thermal state; lasers ramp from zero over `sim.reset_ramp_s` (3 ms) |
| `reset_protocol="branch_low"` | Lasers turn on immediately; approach the self-heating root from the cold side |
| `reset_protocol="branch_high"` | Lasers turn on immediately; use the hot, self-locked root |
| `heater_code=(B, N) codes` | Initialize heater registers, FIFO and drive to these codes |
| `heater_code="locked_steady_state"` | Use the oracle lock codes; rings with no code start at `code_min` and warn |
| `heater_code="locked_steady_state_reachable"` | Move rings needing negative heater power to a bluer resonance order; described below |
| `settle=True` | Settle the thermal state under the held initial inputs; defaults the protocol to `branch_high` |
| `factory_table=table` | Call `controller.load_factory_cal(table)` before `controller.reset(spec)` |

`mrm.control.calibration.factory_cal` makes the factory table.

### Locked-steady-state codes

`warm_start_codes` calls `solve_locked_steady_state` at each world's base temperature and bias at
t = 0. It uses the initial data-path state through `solve_locked_steady_state(data_state=)` and
`runner.reset_data_state`: driver enable, swing scale, transition density and ones density.
The target detuning remains the traffic-on operating point.

The calculation omits aggressor effects, laser offsets and bus effects from unlocked rings. With
`heater_code="locked_steady_state"`, a ring without a code starts at `code_min` and emits a warning
(G26 seed 7: 13 of 64 sampled rings). Every such ring examined so far already sits red of its
lock point at `code_min`, so its heater-up order would need negative heater power.

`heater_code="locked_steady_state_reachable"` instead uses
`solve_locked_steady_state(order="reachable")`. It moves such a ring one resonance order bluer,
requiring one FSR of heating. The extra heat can push other rings below zero required power, so the
process repeats until none needs negative power. Worlds that need no hop keep the same codes as
`"locked_steady_state"`. **Experiment files use this reachable mode for `start: warm`.**

For G26 seed 1000, B = 8, this reduces rings without a code from 10 to 0. Fourteen rings hop: the
original ten plus four that previously had heater-up codes. Worlds 0–2 then exceed the array supply
limit at those codes.

### Shared heater supply

When a world's total heater power or current exceeds `array.heater_total_power_max_w` or
`array.heater_supply_current_max_a`, the plant applies one shared droop factor to every heater in
that world (`mrm.plant.electrical.supply_limit_row`). G26's limits are 0.2 W and 0.15 A,
both assumed.

`solve_locked_steady_state` checks the same limits and returns `supply_scale` and `supply_limited`.
A supply-limited warm start keeps its codes and warns; the flag is recorded in
`manifest_inputs["warm_start"]["supply_limited"]`. No ring in that world can hold its calculated
lock there.

The nominal G26 world does not reach the limit over 0–105 °C and 1–3.75 V. The `stacked_eic`
hypothesis (`P_FSR = 36.9 mW`) reaches it below about 30 °C. At 25 °C its lock currents total
~155 mA; heater voltages droop 3–5%, and every ring settles on the cold branch, 800–1100 pm blue
of its line. Corners `g26_low_eta` and `g26_fsr_edge` reach the limit at 0 °C, with droop factors
0.985–0.989.

### Settling the thermal state

`Plant.settle`, also available through `Plant.reset(..., settle=True)`, initializes the thermal
state as well as the commands. Without settling, all thermal lags start at zero. Even the correct
lock codes leave the ring about 100 K too cold for the first 10–50 µs and several K too cold for
tens of milliseconds; G26 crosstalk poles extend to 9 ms. A ring intended for the hot branch can
then fall off its stable branch. Measurements without settling put every G26 ring at −800 to
−1250 pm after 5 ms.

Settling runs the kernel open loop, one tick per iteration, using the DC coefficients of every
thermal lag: `a = 0, c1 = r`, the exact first-order-hold (FOH) step as `h → infinity`. It stops
when ring temperatures change by less than 2 mK, then resets everything except the thermal state.
Each iteration includes the plant's own implicit self-heating solve, heater temperature-coefficient
of resistance (TCR) split, bias, data heat and scenario inputs held at t = 0. The result is the
engine's own fixed point: it agrees within 5 mK with a 100 ms open-loop run at code 0.

Nominal G26 settles at the steady-state codes in about 18 iterations, then holds detuning within
0.03 pm open loop. Each iteration uses fresh noise draws, which limit convergence to about 0.5 mK.
Each world stops at its own first iteration below tolerance; the report records
`iterations_per_world` and `converged_per_world`. Its start therefore does not depend on other
worlds in the batch.

The scenario stays at its reset-tick value throughout settling. Before the 2026-10-16 fix,
iteration i used scenario tick i. An aggressor switching on at 10 µs could therefore start ring 0
18.7 K too hot, at +1554 pm rather than +121 pm, despite a “converged” report. The shipped scenarios
were affected by at most a few pm. `manifest_inputs["warm_start"]` records the codes, protocol and
settle report.

### Starting with different traffic

Warm-start codes compensate for the data-path heat actually present at t = 0 while retaining the
traffic-on target detuning.

- With traffic initially off (`traffic_start_stop`, `t_start > 0`), the start needs 1243–1348 more
  codes than static to replace ~1–1.4 K of missing data heat. Settled detuning is within 1.0 pm
  of the static start.
- With ones density initially zero (`ones_density_steps`, `idle` at `t0 = 0`), settled detuning is
  within 0.5–4.6 pm. The residual grows along the bus because `solve_locked_steady_state` omits
  upstream filtering of other lines, which changes with density.

Before 2026-10-16, codes always assumed traffic on. Initial detuning was 65–98 pm too blue with
traffic off and 13–24 pm too blue with zero ones density. In `static + traffic_start_stop(0.01)`,
`pi_lock_c` startup dropped rings 4–7 to their cold branch at −1228 to −1294 pm; ring 6 was still
at −761 pm at 20 ms. After the fix, every ring stays held through traffic startup, at 115–118 pm
at 20 ms (engine review, engine-3).

### Why settled detuning differs slightly from the target

Nominal G26 settled at the steady-state codes sits +97 to +121 pm red of the lines, while targets
are +104 to +105 pm. This comes from the steady-state solver's simplifications—a single bus pass
and own-line heat—not a settling error. The engine includes absorption from other lines, which
contributes about 9% of the heat in rings 0–2.

## 4. What happens in one tick (event order)

Tick k is the instant ending interval `[k-1, k)`. At that instant, in each world:

1. A compiled controller's `fast_kernel` runs once for every monitor sample released at k, before `step`.
2. If k is a controller edge, release all sensor banks: monitor block, temperature sensors,
   readback, error counter, bit-statistics window, and ones/bit sums since the previous edge.
   Pack the observation, call the controller, clip commands to channel ranges, and put them in
   the controller pipeline to latch at `k + Controller.latency_ticks`.

Then advance interval `[k, k+1)` in this order:

1. Apply commands latched at k to the DAC registers.
2. Evaluate scenario inputs at k and k+1 for FOH interpolation, laser lines and the `laser_ramp`
   factor, then ones density.
3. Advance the bias DAC: fast offset, latency FIFO, noise, slew, settling and scenario offset.
   Compute the NRZ levels.
4. Build optics inputs from the previous ring temperature in `mrm.plant.phase`. Recompute coupling
   `K^2` at the laser lines only when their wavelengths move. Prepare the self-heating adapter's
   substep caches and self-heating bounds.
5. Compute data-activity heat, the heater TCR split and heater DAC path, including fast offset
   and `dac_glitch`.
6. Solve the implicit thermal/self-heating step. The optics calculation walks rings in
   `layout.bus_order`, applying bus loss between rings.
7. Advance interval sensors: monitor, error counter, temperature sensors and readback. Produce
   Truth, metrics and records for the state at k+1.

The tick loop is `kernel._world_loop`, called by `_world_block`. The loop is split from its setup so that the loop compiles without the numba runtime
(see `docs/handoff/perf.md`, "Round 2").
`kernel._implicit_step_eng` mirrors `mrm.plant.selfheat.implicit_step_b` line for line.
For G26, bus loss is 0.073 dB/mm across each 50 µm segment: a factor 0.99916 per segment,
or 0.59% less light at ring 7 than at ring 0 (`mrm.plant.phase`, equation 6).

Each physics or converter module owns its analog delays. The engine owns only the controller
pipeline.

### Fixed and automatic substeps

`substep_policy` defaults to `sim.substep_policy`; G26 uses `fixed`. With `auto`, the thermal
integration uses power-of-two steps up to `auto_max_ticks` (64) when actuators are quiet, the
scenario is linear and the detuning rate is small. Steps stop at controller edges, record-window
boundaries and oracle ticks. A scenario event forces a one-tick step, and step size h can at most
double from one step to the next. The thermal step is exact for any h; sensors still run every
plant tick. See `docs/handoff/engine.md` section 3, item 10.

A block boundary—one `Plant.run` call—also ends a substep. Growth then restarts from that shortened
step, so **automatic trajectories can depend on where blocks end**.

A substep never crosses a controller edge. With no tick jitter, `runner._block_end` therefore ends
every block on an edge, making `chunk_ticks` irrelevant to results on the compiled and open-loop
paths. The Python path already stops at every edge. `chunk_ticks` defaults to 1e5 and is saved in
`manifest_inputs["chunk_ticks"]`.

With `tick_jitter_s > 0`, edges differ across worlds. Block boundaries and `chunk_ticks` can then
change the automatic step sequence, producing another valid sequence within the policy's error.
Direct `Plant.run` calls with blocks ending away from controller edges have the same dependence.

Before 2026-10-16, blocks ended every `chunk_ticks` ticks regardless of edges. In nominal G26 with
`pi_lock_c` over 20 ms, chunk sizes 13 versus 1e5 produced 11426 versus 10582 substeps and readings
up to 9 monitor codes apart (engine review, engine-4).

## 5. Controllers on the engine

### Python and compiled execution

The **Python path** accepts an object with `reset(spec)`, `step(obs) -> {channel: codes}` and
`latency_ticks`. At each controller edge, the kernel stops; the runner builds an `Observation`,
validates action shape, dtype and range, then resumes. The controller is vectorized over
B worlds. Edges must be shared by every world, so this path does not support `tick_jitter_s`.
Its overhead is about 100–300 µs per edge, in addition to the plant calculation.

A `CompiledController` can also run here with `path='python'`. Following a
`fault.controller_reset`, `Plant.restart_compiled` puts its new initial state in the kernel's
state rows and refills held commands from latched DAC registers. `fast_kernel` and `step` again
share one state. Before 2026-10-16, the fast kernel kept the old state, stopping `bias_dither`'s
lock-in after a reset on this path.

The **compiled path** runs a `mrm.control.api.CompiledController` inside the kernel, per world,
and supports edge jitter. Its Numba step signature, typed `STEP_FT`, is:

    step(p_i64, p_f64, s_i64, s_f64, obs_i64, act_i64)

An optional `fast_kernel` runs once per released monitor sample. Its heater/bias offsets are held,
added to the latched command before DAC FIFOs, and clipped. The controller step is passed as a
first-class function value, so one compiled plant kernel serves all controllers. Adding a controller
does not recompile the plant.

### Commands outside the allowed range

`controller_interface.on_out_of_range: error` raises `ActionRangeError` on both paths:

- Python raises during the step, through `ActionValidator`.
- Compiled execution raises after the kernel block containing the first invalid command. The
  kernel still clips that command, but records each world's first out-of-range element, tick
  and value in `kernel.WI_SAT_FIRST_*`; the error message names them.

The default, `clip`, saturates and counts on both paths, using
`diagnostics["commands_clipped"]` and `saturation_counts`. Before 2026-10-16, the compiled path
always clipped and counted, even when `error` was requested.

### Simultaneous and round-robin scheduling

`controller_interface.schedule` controls which ring's command takes effect:

- **`simultaneous`**, the default, updates every ring at each edge.
- **`round_robin`** models one datapath time-multiplexed over rings. Edge j, counted from the
  controller phase, updates only ring `j % N`. `ControllerSpec.active_ring(tick)` computes it;
  `Observation.active` and ABI header `OBS_ACTIVE_RING` report it.

For every per-ring `(B, N)` actuator channel, only the active ring's command is applied. Others
keep their held command, including through a controller reset. Channels shaped `(B, L)` or
`(B, 1)` are applied whole. Both engine execution paths and the NumPy reference follow this rule.

The Python path merges commands before validation (`mrm.control.api.merge_round_robin`), so only
applied commands are range-checked and counted. Each ring updates at the controller rate divided
by N: 12.5 kHz per ring for G26's 100 kHz controller.

Before 2026-10-16, the setting was stored but ignored. `CompositeController` refuses interface
round robin because it schedules its children itself. `kalman_lqg` and `mpc` also refuse it:
their designs assume all ring commands act every period. The controllers built on the ramp FSM
(`ramp_acquire`, `acquire_then_lock`, `bitstat_lock`, `wdm_supervisor`) refuse it for the same
reason: their sweeps and lock loops move every ring's heater at every edge.

### Special controller interfaces

**Truth-fed controllers** have `truth_fed = True`; only `mrm.validation.oracle.Oracle` uses this.
They are called as `step(obs, truth=plant.truth())`. Truth includes lag states `x`,
`heater_applied` and per-world `conditions`. The controller registry refuses these controllers.

**Composite controllers** use `CompositeController`, registry name `composite`, with parameters
`{children: [{name, rate_hz, params?}]}`. Children run at their own rates on the Python path.

### Registered baselines

Importing `mrm.control.baselines` registers them. List them with
`mrm.control.registry.available()`.

| family | names | start they expect | needs |
|---|---|---|---|
| feedback locks (A) | `pi_lock`, `pi_lock_c`, `dither_lockin`, `dither_lockin_c`, `es_po`, `es_sine`, `feedforward_pi` | ring on the hot branch red of its line (warm start + settle); they map the peak themselves | `feedforward_pi`: a temperature sensor and `dlambda_dt_nominal` + `heater_law` + `heater_efficiency` (or `ff_codes_per_k`) |
| bias locks (A) | `bias_dither`, `midranging` | as above | `controller_interface.bind.tx.bias=controller`, `scenario.bias_v=null` |
| acquisition / WDM (B) | `ramp_acquire`, `acquire_then_lock`, `bitstat_lock`, `wdm_supervisor` | cold (heaters at 0, `branch_low`); `wdm_supervisor` also a factory table (`factory_table=`) | `heater_nominal`, `optics_nominal`, `dlambda_dt_nominal` tokens help; `wdm_supervisor` writes `tx.assignment` when it is bound to the controller |
| model based (C) | `kalman_lqg`, `kalman_lqg_c`, `mpc`, `mpc_c` | `start='warm'`: at the lock (settled warm start); `start='cold'` ramps (Python versions only) | `model_nominal` token |
| fixed-point reference (hardware task) | `fxp_pi` (mrm.control.fxp.FxpPi, compiled) | at the lock (settled warm start); it holds the operating point it starts at | nothing; g26-nominal gains (section 12) |

### Recorded controller results

[handoff/controller_scorecard.md](handoff/controller_scorecard.md) records all baselines on nominal
G26 for steady lock and A1, using `tests/validation/test_controllers_on_engine.py`.
The results below are from **2026-10-16 20:02**, after the engine, controller and parameter reviews
and the `feedforward_pi` fix. They are conditional on that plant. Steady lock also requires a
stationary operating point, with drift at most 0.1 pm/ms.

From a settled warm start:

- `pi_lock` (Python, compiled and composite), `feedforward_pi`, `kalman_lqg` and `mpc` (Python and
  compiled) hold all eight rings with RMS jitter at most 0.1 pm. Compiled and Python versions
  give identical metrics.
- `bias_dither` and `midranging` hold all eight rings.
- Dither locks create jitter around 5 pm through their own dither. `dither_lockin` keeps four of
  eight rings within 5 pm.

**Peak seeking can reach a stable reading with poor modulation.** The drop peak is the wrong
operating point for a modulator and sits at the self-heating fold of a G26 ring, whose bistability
ratio is about 6. `es_po`'s default probes barely cross that fold: rings 4–7 fall once and recapture
before 40 ms. At 40 ms, four rings count as locked and four are still settling. Eventually every
ring parks at the peak, with 13.0–13.2 dB OMA penalty (`runs/params_fix/es_long.log`).

`es_sine` still drifts toward the peak at 0.77–0.95 pm/ms at 40 ms. It settles there after
100–160 ms, with about 13 dB penalty. The pre-2026-10-16 scorecard reported eight locked rings
and 0.3 dB penalty by counting a transient snapshot.

**Acquisition needs wavelength supervision.** Without the supervisor, acquisition controllers
lock every ring on a foreign line. `acquire_then_lock` has four of eight rings within 5 pm and
four still settling; `bitstat_lock` has eight of eight. `wdm_supervisor` starts cold and locks all
eight rings on its assigned lines, using a barrel shift: ring 2 is on line 0 via `tx.assignment`.

Small plant changes can flip these outcomes. The 2026-10-16 fixes moved results for `es_po`,
`midranging`, `acquire_then_lock`, `bitstat_lock` and `wdm_supervisor`. A 2% change in droop or
`s(T)` at lock can make `es_po`'s falls appear or disappear. These are not robustness results.

**Under A1 at the default site**, 55 µm from PD0, `pi_lock`, `midranging`, `kalman_lqg` and `mpc`
hold rings 1–7 within 12.5–15.4 pm. For `pi_lock`, ring 4 reaches 14.7 pm, ring 5 reaches 6.4 pm
and the others stay within 2.5 pm. Ring 0 departs by 454–473 pm at the hot cusp, with its heater
at zero for 14.7–16.4 ms: it runs out of tuning range.

Before the parameter review, ring 0's error was 347–362 pm. A1 specifies a resonance trajectory;
with the model's temperature sensitivity at lock reduced by 2.1%, the same wavelength swing needs
a 2.1% greater temperature rise for the heater to cancel.

`feedforward_pi` now holds rings as `pi_lock` does. Its die sensor sees about +20 K from the
aggressor. Before the gain-schedule fix on 2026-10-16, its feed-forward ran to code 0 and lost
every ring (`mrm/control/baselines/feedforward.py`). In the stress scenario
`a1_slide10_aggressor_pd_adjacent`, every controller loses ring 0; PD0 exceeds 600 °C.

## 6. Outputs

`runner.run` returns:

    RunResult(metrics, trace, truth_final, manifest_inputs, diagnostics, wall_s, plant)

### Metrics

`mrm.sim.metrics.summarize` produces per-ring arrays shaped `(B, N)` unless otherwise noted.
It reports:

- Detuning mean, RMS and maximum relative to the laser line.
- Lock error `e = d - d_ref`: `max_abs_lock_error_pm`, `rms_lock_error_pm`, `locked_final`,
  `ever_locked`, `excursion_count` with hysteresis, `excursion_max_s`, `time_unlocked_s`,
  `first_lock_s` and `time_to_lock_s`; also `time_above_s`.
- Wrong-line time, events and final state, with `allowed_assignments` set to `fixed`, `cyclic`
  or `any`; and FSR hops.
- relOMA and oracle relOMA: `mean_reloma_db`, `oracle_reloma_db`, `oma_penalty_db`, plus
  unrestricted variants.
- BER: expected errors, windows over the limit and the worst window.
- Heater energy, time at command limits, peak demanded slew and code range used.
- `world` sums and `settings`.

**Choose the lock reference deliberately:**

| `lock_reference` | Reference |
|---|---|
| `'line'` (default) | Literal detuning relative to the laser line |
| `'oracle'` | Oracle's best stable relOMA point, refreshed every 1 ms |
| `'target'` | `lock_target_pm`, a scalar or `(B, N)` array |

A side-of-fringe lock intentionally sits 0.3–0.5 FWHM from its laser line. Judging it against the
line can therefore report “never locked” even while it holds its intended operating point.

### Recorded traces

Each `RecordSpec(field, decimation_ticks, reducer)` produces:

    {"<field>@<decimation>:<reducer>": {"t_s": (S,), "value": (S, B, N)}}

Reducers are `mean`, `min`, `max` and `last`. `envelope(field, dec)` records minimum and maximum,
which preserves a one-tick spike that a mean would hide.

Available fields are listed in `mrm.sim.metrics.TRUTH_FIELDS`: ring/heater/PD temperatures;
heater and absorbed power; resonance; detuning in metres and half-widths at half maximum (HWHM);
monitor input/current; `P_in`; bus OMA; relOMA; extinction ratio (ER); insertion loss (IL);
heater code; bias; resonance order; occupied line; base temperature; monitor code; oracle relOMA;
BER; and declared line.

### Truth, diagnostics and reproducibility information

`truth_final` is a `Truth` object containing the fields above, per-line chip/output powers and
wavelengths, optics arrays, every thermal observer temperature, lag states and per-world conditions.
It is available only to tests, metrics and the oracle, never to an ordinary controller.

`diagnostics` counts non-converged self-heating solves, samples lost to FIFO/block limits,
pipeline overflows, clipped commands, substeps, largest substep, fast-kernel calls, saturation
counts and FIFO capacities.

`manifest_inputs` records what is needed to describe the run:

- Config fingerprint/hash; seeds; B, N, L, dt, ticks and policy.
- Scenario fingerprint and aggressor calibration; controller name, path and latency; spec fingerprint.
- Metric settings, world choices, recordings and diagnostics.
- `warm_start`: codes, protocol, settle report, factory table, `mode`, `no_code (B, N)`,
  `order_hopped (B, N)` and `supply_limited (B,)`.
- `score_from_tick` / `score_from_s`, plus `compression_note` for a time-compressed scenario.

### Locked steady state without a time simulation

`solve_locked_steady_state(...) -> LockedSteadyStateResult` finds settled locked operating points
without time stepping. One nominal G26 point takes about 0.1 s.

For each `(ambient, bias, world, ring)`, it returns oracle lock detuning; stable and unrestricted
relOMA; ring temperature; absorbed and heater power; heater code; code-range coverage; and FSR-hop
and wrong-line risk. For each `(ambient, bias, world)`, it also returns supply droop `supply_scale`
and `supply_limited`. With `order="reachable"`, `order_hopped` identifies rings moved one order bluer.

Before 2026-10-21, this function was named `quasi_static` and returned `QuasiStaticResult`.
The old warm-start names `heater_code="quasi_static"` and `"quasi_static_reachable"`, and experiment
kind `quasi_static`, remain accepted aliases.

## 7. Internals (read before changing the engine)

### Array layout and compilation

`Plant._build` combines `PlantParams`, `Scenario` and `ControllerSpec` into two namedtuples of
arrays: `KParams` (about 180 fields) and `KState` (about 90). Every array is a writable,
C-contiguous copy. This keeps Numba's types independent of the config. `_check_kernel_arrays`
rejects read-only arrays and strided views, which would introduce a new type and trigger recompilation.

Scalar state for rings, worlds and laser lines is packed into seven arrays of named rows:

| State block | Index constants |
|---|---|
| `S.ring_float` | `RF_*` |
| `S.ring_int` | `RI_*` |
| `S.ring_int8` | `R8_*` |
| `S.ring_bool` | `RB_*` |
| `S.world_int` | `WI_*` |
| `S.world_float` | `WF_*` |
| `S.line_float` | `LF_*` |

Mappings such as `kernel.RF` map names to row indices. All row names, constants such as
`RF_T_RING`, field lists for `KParams` / `KState`, and value definitions come from the single
registry in `mrm/sim/state_spec.py`. The kernel imports them from there.

The kernel is one explicitly typed `@njit(nogil=True, cache=True)` chunk function. Worlds run in
contiguous chunks on a thread pool because Numba `prange` cannot accept the 180-array namedtuple
(`docs/handoff/engine.md` section 3, item 1). Results are bit-identical across thread counts.

Do not close over config values: numba freezes them at compile time, so a changed value is silently ignored. Do
not pass dispatcher arguments: they stop numba from keeping the kernel in its disk cache. The self-heating adapter
is called directly through `eng_pabs` / `eng_commit`, which is why `_implicit_step_eng` mirrors
`selfheat.implicit_step_b`. Copy solver changes into both implementations.
`tests/verification/test_kernel_parity.py` catches any bit-level difference, and
`test_kernel_is_disk_cacheable` guards caching.

### Runtime-free hot loops

The full tick is split between `_world_block` (views, tables and scratch created per block) and
`_world_loop` (the tick loop). The loop runs without the numba runtime, which would otherwise count references to
every array passed to a call.

The loop and its hot helper functions use `_nrt=False`: they do not use Numba's runtime, avoiding
roughly 500 array reference-count updates per tick. The comment above `_world_block` gives these rules:

- A runtime-free function cannot allocate or return a new array; Numba rejects both.
- It must not call a function that returns an array, because that reference is never released.
- A runtime-enabled function called from it must not make views of its arguments. In Numba 0.68,
  that leaks one reference per call.

`tests/verification/test_kernel_parity.py::test_kernel_releases_every_array` checks reference counts.

Passing all of `KParams` / `KState` copies about 270 array structs and costs ~0.4 µs per call.
Per-tick helpers therefore take one world's arrays directly: `_truth_a`; controller-edge helpers
`_release_a`, `_sync_act_a`, `_push_act_a`, `_advance_edge_a`; `_auto_substep_a`; `_oracle_a`;
`_fast_tick_a`; and the physics `*_row` kernels.

The tuple forms `release_row`, `truth_row` and `_oracle` are thin wrappers for
`tools/perf_profile.py`. `_auto_substep`, `_fast_tick` and `_declared_line` have no callers;
they document the array forms.

Hot loops use an unsigned index cast, `np.uint64`, through `_as_unsigned_array_index(i)` in `mrm.sim.kernel` and
`mrm.plant.optics`. A signed index that Numba cannot prove non-negative adds a comparison and
selection for negative-index wraparound. Unsigned indices avoid that work. **Use the cast only
for offsets guaranteed non-negative by construction.**

### One optical calculation

Every ring uses the general two-direction field calculation in `mrm.plant.optics`, including when its
backscatter coefficients are zero. `eng_pabs` evaluates absorbed power at a proposed temperature;
`eng_commit` saves outputs and advances the forward bus power. An identical final trial can be reused
at commit because its inputs and result have not changed.

The separate Airy/cache path, polynomial phase rotation, and far-line Taylor hold are retired.
The original implementation is in `archive/optics/retired_cached_approximations.py` for numerical
comparisons. Production does not import it. `optics_far_hold` and the reference engine's `airy`
selector are no longer supported. Reserved packed-array slots retain their positions.

The field equations and physical parameters are unchanged. Configured backscatter is included by the
same calculation for every ring. The default backscatter coefficient setting is unchanged.
A lossless, uncoupled ring at exact resonance is singular and raises `ZeroDivisionError`.

### Solver tolerance

`Plant(sh_tol_k=...)` sets the self-heating root tolerance, default 1e-7 K. It is also a `run()` option.
This is a numerical convergence tolerance, not a selector for a different optical model.

### Thermal tables

At the start of each block, the kernel builds tables of lag indices by kind; per-lag gains
`src_scale (c0 + c1)`; implicit gains per observer; heater-split gains; heater-observer lockstep
layout; coefficient tables; and `(lag, source)` pairs.
Lags whose source input and state are both exactly zero are skipped until an input appears.

The calculation is bit-identical to `selfheat.implicit_step_b` and `thermal.observer_split_b`.
Rebuild tables that depend on `KParams` for every block. Do not cache them across calls:
`Plant.settle` swaps coefficient tables.

### NumPy reference and parity

`kernel_ref.ReferenceEngine` builds the same tick from the physics modules' public functions.
Tests compare every Truth field, observation vector and metric bit for bit, in open
loop and with Python/compiled controllers, including under `NUMBA_DISABLE_JIT=1`.

Coverage includes the config scenario and one driving every scenario input read by the kernel:
aggressor, laser failover with a dark gap, fibre loss, traffic start/stop, bias step, the four faults
and controller reset (`test_parity_with_every_scenario_path`). Since 2026-10-16, the reference resets
the controller at the first edge after a `fault.controller_reset` rising edge; previously it ignored
that fault. With `airy=False`, independent-optics parity is within 1e-9 of each Truth field's range,
with identical codes.

### Detuning reported in Truth

`kernel.resonance_detuning` computes the resonance of the order nearest the declared line, using
the phase model's cold curve exactly. The declared line comes from `tx.assignment` when the
controller owns it. The optics slope is:

    dphi_dlamres = 2 pi c t_rt / (lambda - shift)^2

See `docs/handoff/engine.md` section 3, item 6. `detuning_hwhm` divides detuning by
`lambda_line^2 fwhm_hz / (2 c)`. The rigidly translated phase model retains the cold wavelength
width `lambda_res^2 fwhm_hz / c`, which is 1.2% narrower at G26's 7.6 nm lock.
This affects only diagnostic normalization; see the phase-model limitation in section 10.

## 8. Extending

Use the following change points so that the compiled engine, reference and documentation stay aligned.

| Addition | Required changes |
|---|---|
| State row, `KState` array or `KParams` field | Define it only in `mrm/sim/state_spec.py`, with name, meaning, unit, kind, `updated_by` and `depends_on`. Fill it in `Plant._build` / `Plant._reset` and write it in the kernel. |
| Truth field or record channel | Append to `TRUTH_FIELDS`; never reorder it. Write it in `kernel.truth_row` and `kernel_ref`, and extend the parity test. |
| Sensor or actuator channel | Declare it in `mrm/core/channels.py`; IDs are append-only. Produce sensors in `kernel._release_a`; consume actuators in `_world_loop` (interval inputs) / `apply_act_row`. Add the reference implementation. `ChannelMaps.from_spec` maps names to kernel slots. |
| Scenario signal | Add it in `mrm.scenarios.generators`, with an append-only signal ID. Read it in `_world_loop`, using `_signal_offset` obtained once per block in `_world_block`. Choose its auto-substep class: linear FOH or held. |
| Kernel cache | Key the cached value on its inputs and invalidate it in `Plant._reset`, as with `RI_HE_TCODE`, `LF_PDB_KEY` and the optics adapter's NaN keys. Never assume an input stays constant. |
| Fidelity switch | Add an integer `KParams` field from `PlantParams` flags and branch inside the kernel. Never use a module-level constant. |
| Controller | Copy a Python, per-ring or compiled template from `examples/controllers/`. Register it with `@register("name")` in a module imported by `mrm/control/baselines/__init__.py`. Add it to `CASES` in `tests/validation/test_controllers_on_engine.py`. |

After a state-layout change, run `tools/clear_numba_cache.py` and `tools/gen_variable_map.py`.
A kernel compiled for another layout refuses to run. `tests/verification/test_variable_map.py`
checks that the registry and code agree.

After any kernel edit or change to an `njit` function it calls, clear the kernel cache first
(section 1), then run `test_kernel_parity.py`, `test_engine_timing.py`, `test_metrics.py`,
`test_engine_physics.py` and `test_engine_warm_start.py`. The first run recompiles; use
`memguard --heavy`.

## 9. Performance (the fixed-tick and Monte Carlo budgets are met; the auto-substep budget is not)

These measurements use an eight-core M2 on AC power, Low Power Mode off, G26 with N = L = 8,
a warm cache, and the optimizations from performance round 2. See `docs/handoff/perf.md`
section 8 and `docs/handoff/engine.md` section 4 for the breakdown.

| case | us per world-tick | simulated s per wall s |
|---|---|---|
| B = 1, fixed | 9.1-9.3 (load 3-3.7) | 0.108-0.110 (target >= 0.1: met) |
| B = 1, auto, 10 kHz controller | 2.9-3.0 | 0.34-0.35 (target >= 20: not met) |
| B = 1, auto, 100 kHz controller | 4.8-5.1 | 0.20-0.21 |
| B = 64, fixed, 8 threads | 2.0-2.3 | 0.0068-0.0079 for the bank: 64 x 1 s in 2.1-2.5 min (target: 5 min) |

Commit `8646691` took 54–55 µs per B = 1 tick under the same conditions. Round 1 reduced that to
15.5 µs, with results equal up to floating-point rounding. Round 2 changed no result;
outputs are bit-identical.

The margin below 10 µs is only about 0.6–0.9 µs at system load 3–4. A busy machine can exceed
that budget. The largest costs at B = 1 are self-heating optics (~4.1 µs), thermal ROM (~1.4 µs)
and sensors (~1.4 µs, mostly counter-based noise draws). Battery power or Low Power Mode is
about 1.7 times slower.

These historical timings describe the retired cached implementation. Re-measure the consolidated
general field path before using them as a performance budget. `optics_far_hold` is retired.

## 10. Physics to expect, and open items for the next stage

This section records physical behaviour, fixes and remaining gaps. Its dated results are not a
fresh validation run. [KNOWN_ISSUES.md](KNOWN_ISSUES.md) identifies stale reports and acceptance
results, including A4.

### Aggressor position and PD overheating

On 2026-10-10, the G26 aggressor moved from (0, 30) to (0, −30) µm. Its distance to ring 0's
monitor PD changed from 5 to 55 µm; `ring.thermal.pd_offset_um` is [0, 25]. Every ring-to-site
distance stayed the same. Both positions are assumed.

At 5 µm, calibrated A1/A2 power heats the PD observer to 500–600+ °C. Arrhenius dark current
then reaches microamps and shifts every monitor-based lock. This position remains available only
as the stress scenarios `a1_slide10_aggressor_pd_adjacent` and
`a2_808_aggressor_pd_adjacent` (section 2). See `docs/handoff/integrator.md` section 7.

### Wrong-line capture, bias and data heat

A cold G26 ring sits about 2 FWHM blue of line `n+3` and can self-heat onto it. Heater ramps also
cross foreign lines. Acquisition controllers take the first line they encounter unless a
supervisor assigns wavelengths.

Bias tuning is weak in G26: a 0.2 V step moves resonance only 0.1–0.5 pm. Changes in free-carrier
absorption (FCA) heat cancel most of the direct electro-optic shift, weakening bias locks
(controllers_a handoff). Data-activity heat raises each ring about 2 K, so traffic and squelch
steps move its temperature by that amount.

### Bus loss between rings

Since 2026-10-16, each ring after the first sees 0.99916 times the preceding ring's through output:
G26's printed 0.073 dB/mm over the assumed 50 µm spacing. Ring 7 gets 0.59% less light than ring 0.
Previously, `Plant._build` supplied no `bus_t2`, so the engine, `solve_locked_steady_state` and
`mrm.control.models` used lossless segments, while `derive` and replicas applied the loss.

### Crosstalk: two possible packages

Since 2026-10-16, the interim crosstalk time constant uses the source-to-victim distance from its
imec anchor: centre distance minus both ring radii (`r_ref`). At G26's 50 µm pitch, this gives:

| Path | Current time constant | Before the distance fix |
|---|---|---|
| Nearest ring | 1.52 ms | 2.85 ms |
| Own monitor PD | 0.38 ms | 0.71 ms |
| A1/A2 aggressor to ring 0 | 0.62 ms | 1.03 ms |

These are based on imec natural-convection data: 3% nearest crosstalk, a 1.5% floor, and
1.5–9 ms inter-ring time scales. The in-ring FEM ROM's TEC-held package instead gives 1.0%
nearest crosstalk, reaching 63% of its final value in 0.12 ms, with a 0.09% floor at 1 mm.

The TEC-held package is `uncertainty.thermal.stack: side_by_side_tec_held`, corner
`g26_tec_package`. Its sampling weight is zero: named-corner robust/compare runs include it,
but sampled worlds do not. Which package G26 uses is unknown; see VALIDATION.md,
`closure-crosstalk-dynamics`.

Die sensor `die0` sees each heater through `ring.thermal.site_floor_frac`: by default 1.5% of
`Z_heater->ring`, or 85 K/W with a 9 ms time constant; in the corner it is 0.09%.

### Monitor ADC reference temperature

Since 2026-10-16, the assumed 20 ppm/K ADC reference temperature coefficient
(`ring.monitor.adc_ref_tempco_ppm_per_k`) acts at `ring.monitor.adc_ref_site`:

- **`package_base`**, the default, uses `thermal.t_base_k + thermal.xpu_rise_k`. The converter
  is on its own die beside the PIC (`side_by_side`).
- **`monitor_pd`** uses each ring's PD observer, the previous behaviour. The kernel stores this
  temperature in `RF_T_ADC`.

At G26 lock, PD observers sit about 20 K above the base. Codes now read 17–24 codes, or 0.04%,
higher than before. Ring 0 under A1, with PD0 around 93 °C, no longer has its former 0.14% gain
dip. At 25 °C ambient, the reference sits at its trim temperature, matching
`mrm.control.models` (which uses `T_ref`) exactly. Heater and aggressor heat no longer reach
ADC gain in the default model.

### Bit statistics and ADC multiplexing

Since 2026-10-16, `datapath.bitstat_above` uses the scenario's ones density from `RF_DENS`, also
set at reset for a controller edge at k0. With `fidelity.pattern_noise` set to `prbs7`, `prbs9`,
`prbs15` or `prbs23`, its ones count is the exact count in the last 64 sent bits (`RI_BS_N1`).

Previously, using the realized per-tick density `n1/nb` (0.504 for prbs7) made each window a
`Binomial(64, n1/nb)` draw, uncorrelated with the sequence. For prbs7, its standard deviation was
4.1 instead of 2.8. Binomial mode now draws `Binomial(64, density)`. At the first edge after
reset, window levels remain reset placeholders because no interval has run yet.

For `adc_mux: round_robin` with `channels_per_adc = C > 1`, one integrating converter is shared
by C rings. It requires `C × window <= ADC period`, checked by `realize` and `MonitorHw.build`.
The rings also share converter DNL thresholds and INL bow, keyed by converter `n // C`.
These checks/sharing have applied since 2026-10-16.

Conversion or de-integration time is not modelled. A shared dual-slope converter also requires
`C (W + T_deint) <= P`; the preset's rate must respect that additional constraint.

### Thermal-ROM time scaling

Since 2026-10-16, with `fidelity.thermal_model: rom` and unequal `time_scale_fast` and
`time_scale_slow`, a mixed-sign FEM source–observer pair spanning the 300 µs split is scaled
as a whole (`mrm.plant.thermal._whole_pair_time_scales`). Scaling each pole separately caused
heater-to-ring overshoot of up to 21% in `metal_above_160`. The interim model is unchanged.

### Rigid-comb phase approximation

The phase model keeps a heated ring's cold FSR in wavelength (`mrm.plant.phase`, equation 2).
The mode artefact's `n_eff(lambda, T)` table instead shifts order `m+k` by 0.43% of the resonance
shift per FSR more or less: ±33 pm for adjacent orders at the 7.6 nm lock, scaling linearly
with heater shift.

The locked order is exact. Foreign-line heat and monitor background differ by at most 0.15% of
`P_abs` in G26 and about 1% in FSR stress corners. Implementing a phase law per resonance order requires a
thermo-optic dispersion key and changes every consumer of the curve. This remains open.

### Supply-limited and failing sampled worlds

Some worlds reach the shared heater supply limit (section 3): `stacked_eic` below about 30 °C,
and corners `g26_low_eta` / `g26_fsr_edge` at 0 °C. No controller can hold the whole bank there.
LHS quantiles including those worlds reflect the supply limit; world summaries expose
`warm_start_supply_limited`.

`pi_lock_c` also loses rings in some sampled worlds **without a warm-start problem**.
`experiments/examples/mc_pi_lock.yaml` runs 64 worlds, static for 20 ms, judging lock about each
ring's own operating point. In the 2026-10-16 rerun after iterated order hopping:

- 27 worlds have a ring more than 5 pm from its point.
- Ten have no warm-start flag for missing code, order hop or supply limit: worlds 1, 3, 11, 13,
  30, 32, 33, 39, 43 and 48.
- Median worst-lane OMA penalty is 1.6 dB; the 95th percentile is 31 dB.
- One ring has no code, down from nine; 101 rings hop; two worlds are supply-limited.

The run is recorded under `runs/engine_fix/exp`. Controller owners need to analyse these worlds
before making a claim over the uncertainty set; see `docs/handoff/experiments.md` section 12.

### Recorded acceptance outcomes

The acceptance table in `docs/handoff/experiments.md` section 12 records these outcomes on nominal
G26 with `pi_lock_c`, except B5, which uses `wdm_supervisor`:

| Recorded outcome | Cases |
|---|---|
| Pass | A4 (soak), A7 (relOMA plane), B3 (XPU step), B5 (cold start) |
| Fail | A1 and A2 (ring 0 heater range at the hot end); A3 (14 pm lag under the 50 nm/s sweep); A6 and B2 (set-point movement under power steps); B1 (squelch) |

These are recorded results. In particular, A4's old PASS is not a current result; its certificate
and hop criterion are stale (KNOWN_ISSUES.md).

### Features not built yet

The remaining items are the controller telemetry metrics; `fidelity.data_modulation: lut` (the waveform LUT
exists; see `docs/handoff/waveform.md` section 3); reference-tap and `datapath.bitstat_ones`
channels; `DacFacts.law`; one shared knowledge-token vocabulary; `run_lock_request` for replicas
(`docs/handoff/replicas.md` section 6); Verilator/cocotb closed-loop HDL co-simulation (section 12);
an in-kernel bridge. the archived learning tools retain their separate reset and throughput limitations (section 13).

Building a plant explicitly refuses `ring.monitor.reference_tap.present: true`.
Bridged runs currently take the Python path.

## 11. Experiments, run directories and the CLI

### What an experiment file contains

An experiment file is an `mrm.params` config with `meta.kind: experiment`. It `extends:` a preset
and adds `controller`, `scenario`, `metrics` and `experiment` sections. Every `experiment.*` key
is defined in the schema. Keys and defaults are documented in `docs/handoff/experiments.md`
section 3 and the appendix of [PARAMETERS_FORMAT.md](PARAMETERS_FORMAT.md).

This is an abridged example from `experiments/acceptance/a1_slide10_aggressor.yaml`:

```yaml
extends: g26
meta: {name: acc_a1_slide10_aggressor, kind: experiment}
controller: {name: pi_lock_c}
scenario: {name: a1, params: {phase_s: 0.06535}}
experiment:
  kind: acceptance                 # run | sweep | mc | compare | robust | acceptance | locked_steady_state
  duration_s: 0.1357
  seeds: {device: 0, uncertainty: 0, noise: 0, scenario: 0}
  start: warm                      # cold | cold_settled | warm (locked_steady_state_reachable codes, settled)
  score_from_s: 0.005              # leave pi_lock's peak map out of the metrics
  self_reference: {duration_s: 0.02, window_s: 0.005}   # judge the lock about each ring's own operating point
  acceptance:
    - {id: A1-lock, metric: max_abs_lock_error_pm, op: "<=", value: 5.0, source: "04 section 4"}
```

### Commands and experiment kinds

The main command is:

    mrm run EXP [--set PATH=VALUE] [--alt PATH=NAME] [--threads N] [--runs-dir DIR] [--dry-run]

`mrm sweep EXP`, `mrm mc EXP`, `mrm compare EXP` and `mrm robust EXP` force the experiment kind.
`mrm acceptance [EXP ...] [--all]` prints a pass/fail table and exits with code 1 if a criterion
fails. `mrm rerun runs/<id>` reruns from the saved directory and compares every `trace.h5` array
bit for bit.

From Python, use `mrm.run(path_or_config, overrides=..., options=..., runs_dir=..., write=...)`.
`--dry-run` estimates worlds, ticks, controller edges, memory and wall time per case from measured
engine speeds. Python-controller cost varies by about 10 times per edge; the soak's auto policy
runs at about 6.5–7.8 µs per tick.

| Kind | What it runs |
|---|---|
| `run` | One `runner.run` with `batch` worlds and default recordings |
| `sweep` | Cartesian `experiment.grid` over config paths, `alt:<path>`, `corner` and `controller` |
| `mc` | `n_mc` worlds, metrics only, in chunks of `max_worlds_per_plant` |
| `compare` / `robust` | Nominal world, named corners (the 12 corners of the uncertainty set plus `g26_tec_package`) and `n_mc` LHS worlds; worst corner, quantiles and Kendall-tau rank stability |
| `acceptance` | Criteria that must hold in every world of every case |
| `locked_steady_state` | `runner.solve_locked_steady_state` over ambient × bias, plus one survey per corner; `quasi_static` remains an alias |

For `compare` / `robust`, the first report line carries a caveat: performance is conditional
on assumed thermal dynamics.

### What is saved

Each directory is named `runs/<UTC>_<hash8>/` and contains:

| File or directory | Contents |
|---|---|
| `experiment.resolved.yaml` | Resolved leaves, with provenance |
| `manifest.json` | Schema `mrm.run/1`: Git commit/branch, dirty-diff hash, environment, argv, seeds, source chain, artefact hashes and every case's runner manifest |
| `metrics.json` | Computed metrics |
| `trace.h5` | Exact recorded arrays |
| `inputs/` | Copies of the experiment, non-preset parents and corner files supplied by path, captured when loaded |
| `report.txt` | Human-readable report |
| `convergence.json` | Convergence information for compressed runs; reruns obtain reused certificates here |

Copied inputs let the directory rerun from any working directory without the original input files.
`hash8` is the first eight hexadecimal digits of the experiment hash. That hash includes the config
without experiment-layer keys, resolved options, and each corner's alternatives when corners are
run. Editing a corner preset therefore makes `mrm rerun` refuse the old experiment.

### Time compression needs a convergence certificate

A compressed scenario, such as the soak with `scenario.time_compression > 1`, runs only with a
convergence certificate. Obtain one in either of two ways:

- Run the convergence experiment first: one slow cycle at `compression_factors`, defaulting to
  1, 10 and 100, on the same plant, controller, start and worlds.
- Set `experiment.convergence_certificate` to a run directory whose certificate covers the
  required plant and code fingerprints.

The **plant fingerprint** includes the resolved config except the experiment section and
compression setting, plus the start and worlds. Since 2026-10-16, the **code fingerprint**
(`mrm.sim.manifest.code_fingerprint`) includes sources in
`mrm/{core, plant, sim, control, scenarios, params}`, `mrm/units.py`, `mrm/materials.py`, every file
under `mrm/data`, and numerical-library versions.

Compression factors must agree in each world on maximum detuning ring by ring—0.2 pm or 5%,
with an absolute cap of 0.5 pm—as well as wrong-line events, FSR-hop events, order changes,
heater range used and integrated errors.

The A4 certificate from 2026-10-10 at
`runs/acceptance_20261010/20261010T115721Z_d1511028` no longer covers the working tree.
The G26 plant changed on 2026-10-16, and the old certificate records no code fingerprint.
A4 must run its convergence experiment again.

### Judging a controller

`mrm.sim.experiment.world_summary` reports worst-lane and mean OMA penalty relative to the oracle;
excursion count and longest duration; time unlocked; rings on foreign lines; time to lock;
absolute relOMA; BER; heater limits; FSR hops; and warm-start flags.

Two order-change metrics differ:

- `fsr_hop_events` counts re-locks on another order.
- `order_changes` counts changes in the order nearest the declared line, regardless of lock.
  Use this to judge a hop: `|detuning| <= FSR/2` already holds by construction and cannot rule one out.

A claim holds only if it holds in every world.

## 12. External controllers: the binary bridge, golden vectors and golden traces

### Binary bridge

`mrm.sim.bridge`, protocol 1, lets an external Python or C controller, RTL simulator, or hardware
behind a socket run in lockstep with the plant. At each controller edge, the plant sends one binary
**OBS** frame containing the edge tick and framed channels for every world. It waits for one
**ACT** frame in reply.

Frames carry simulation ticks, never wall time. Bridged runs are therefore bit-identical to
in-process runs, as tested with Python and C peers. The frame layout comes from `ControllerSpec`:

| Profile | Included channels |
|---|---|
| `minimal` | Monitor, temperature sensors, latched registers and flags; heater output, plus bias when used |
| `full` | Every exposed sensor and bound actuator |

The handshake checks protocol version, layout fingerprint, `ControllerSpec` fingerprint and latency.

```python
from mrm.sim import bridge, runner
peer = bridge.spawn([".venv/bin/python", "-m", "mrm.sim.bridge", "serve", "--preset", "g26",
                     "--controller", "mrm.control.fxp:FxpPi"])          # or bridge.connect(host, port)
with bridge.BridgeController(peer, profile="minimal") as ctrl:
    res = runner.run("g26", ctrl, None, duration_s=10e-3, uncertainty="nominal", variation="off",
                     heater_code="locked_steady_state", settle=True)
```

`mrm bridge c-header --preset g26 --profile minimal` writes `mrm_bridge.h`, including the spec
fingerprint. `mrm bridge c-example` writes the C equivalent of `FxpPi`.

A bridged run uses the runner's Python path. Its overhead is about 28 µs per tick plus the round
trip, which is 11–14 µs for a C peer over pipes. `fast_kernel` and Truth-fed controllers are
refused. For TCP, use `connect(..., timeout_s=)` so a silent peer cannot block the run indefinitely.

### Fixed-point reference PI

`FxpPi` in `mrm.control.fxp`, registered as `fxp_pi`, is a positional PI controller on
`tx.monitor.latest`. It uses Q-format integers: Q3.20 gains and a Q19.20 accumulator by default.
It matches the C and Verilog versions. Start it at a lock: it holds the operating point it starts
from. Its gains are G26-nominal, not a tuned controller design.

### Golden vectors: replay controller inputs without a plant

`mrm.sim.vectors` uses format `mrm.vectors/1`. `mrm vectors record` runs a controller and saves
flat `int64` ABI streams as `.npz`, `$readmemh` `.hex`, and Verilog `.layout.vh` files.
`mrm vectors replay` / `replay_abi` feed those streams through a compiled controller without
running the plant—the RTL test-bench view.

`tests/hdl` replays golden vectors through a synthesizable Verilog PI under Icarus.
The Verilator/cocotb closed-loop co-simulation that the design asks for is not built; the tools
are not installed.

### Golden traces: detect changes to fixed runs

`mrm golden` and `tests/golden` use format `mrm.golden/1`. These fixed-run metrics block CI:

- `g26__static` and `g26__a1_slide10_aggressor`, covering open loop, `pi_lock_c` and `fxp_pi`.
- `g26__soak_compressed`, one soak cycle at compression 100 with `pi_lock_c`. This is the certified
  c = 100 case of A4's convergence experiment and is a slow test.

Any changed engine number fails the golden checks. Update only with
`mrm golden --update --case ... --reason "..."` and the printed CHANGELOG line.
The files were produced on arm64 macOS with the `.venv` versions recorded in each file.
Another platform can produce another valid closed-loop trajectory.

## 13. Archived learning tools

The Gymnasium environments and fixed-point neural-policy helper are archived outside the installed package.
See [archive/rl](../archive/rl/README.md) for the preserved code, examples, optional dependencies, and opt-in tests.
The physical plant, ordinary controllers, compiled-controller ABI, and fixed-point PI arithmetic remain active.

## 14. Calibration and the validation report

### Generated presets and provenance

Calibration generates two presets. **Do not edit them by hand.**

| Preset | Meaning |
|---|---|
| `g26_calibrated` | G26 with device parameters fitted to calibration targets |
| `g25_test_vehicle_calibrated` | The G25 bench preset, `g25_test_vehicle`, with the same fitted device parameters; the post-fit plant for G25 calibration targets |

Both load like ordinary presets: `params.load("g26_calibrated")`, or
`extends: g26_calibrated` in an experiment. **The default remains `g26`.** The calibrated preset
has not been adopted; see the recommendation in `docs/handoff/calibration.md` section 11.
The existing fits and reports are stale, as recorded in [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

Each fitted parameter has `targets`, `residual`, `identifiable` and `note` fields explaining it.
Inspect one with `mrm params explain g26_calibrated ring.heater.r_cold_ohm`.
A parameter receives `calibrated` provenance only if it passes the half-prior rule: its 95 % profile interval is at most
half of its prior range.
A value that changes without passing that rule retains its previous provenance.

### Run calibration

Run every command under the memory guard. The commands below use
`MG=".venv/bin/python tools/memguard.py --"` as shorthand; use `--heavy` above 1 GB. `--heavy` goes before
the `--`, so the heavy commands use `MGH=".venv/bin/python tools/memguard.py --heavy --"`.

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

`mrm calibrate` is the same command. Calibration may read only `targets/calibration.yaml`;
an isolation test scans `mrm/calibration/` to enforce this. Held-out targets are never fitted.

The record `mrm/calibration/latest_fit.json`, schema `mrm.calibration.fit/1`, stores each group's
hypotheses, fits, profile likelihoods and parameter decisions. Its `context` contains written
parameters on which later groups and `--groups` reruns build. It also stores the before/after
table. Methods, results and deviations are in `docs/handoff/calibration.md`.

### Generate the validation report

`mrm report`, also `python -m mrm.validation.report`, generates `docs/VALIDATION.md` for G26.
It combines cached computed parts from `docs/validation/<preset>/report_data.json` with live
presets, target files and the fit record.

```
$MG .venv/bin/python -m mrm.validation.report --check        # exit 1 when the report is stale (the CI test does this)
$MG .venv/bin/python -m mrm.validation.report --fast         # re-render from the cache (~5 s): after a new fit record,
                                                              # a new commit, or a change of the rendering code
$MG .venv/bin/python -m mrm.validation.report --recompute predictions,identifiability   # stale parts only
$MGH .venv/bin/python -m mrm.validation.report               # every part (~76-80 min, peak ~1.2 GB)
$MGH .venv/bin/python -m mrm.validation.report --preset g26_calibrated --out docs/VALIDATION_g26_calibrated.md
```

Approximate costs on G26 with four kernel threads:

| Part | Calculation | Time |
|---|---|---|
| `predictions` | Every target on the nominal world and named corners | 9 min |
| `identifiability` | Calibration Jacobian | 20 min |
| `sensitivity` | Tornado and Sobol worlds under A1 | 23–25 min |
| `ranking` | `compare_locks` experiment | 28 min |

A part is marked stale when a preset it used, a target file, or its tracked code changes.
`report.CODE_DEPS` tracks `replicas.py` and `sensitivity.py`; this dependency check is incomplete
(see KNOWN_ISSUES.md). The CI test
`tests/verification/test_report.py::test_validation_md_is_current` names the part to recompute.

A new fit record changes rendering only (report section 3.1), so `--fast` is enough. After a
commit, run `--fast` once to update the commit stamp without recomputing anything.

G25 bench targets for a `g26_X` preset run on `g25_test_vehicle_X` when that preset exists.
For `g26_calibrated`, this is `g25_test_vehicle_calibrated`.

Recalibration follows this order: **fit, write presets, then recompute the calibrated report**.
Never change a calibration after looking at its held-out rows.
