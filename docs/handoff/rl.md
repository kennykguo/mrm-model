# Handoff: batched RL environment, fixed-point MLP helper and tutorial (`rl`)

Scope: 07 P1-22 (the batched RL environment and "a fixed-point MLP inference helper in fxp") and the tutorial
notebook of 07 P1-19. Status: built and tested; the 1e5 env steps/s target is **not met** (section 4). No engine,
runner or other owner's file was changed; requests are in section 7. Adversarial review (2026-10-10): 5 defects
fixed in these files, tests added; section 9.

## 1. Files

| file | what |
|---|---|
| `mrm/sim/gym_env.py` | `MRMEnv` (gymnasium.vector.VectorEnv over the plant batch), `MRMSingleEnv` (B = 1 gymnasium.Env view), `RewardSpec`, `ObsLayout` / `ChannelCodec` / `ActionCodec`, `code_codec`, `fxp_policy_action`, the compiled MLP policy (`MRMEnv.compiled_policy`, `mlp_policy_init`, `mlp_policy_step`), gymnasium registration `mrm/MRM-v0` |
| `mrm/control/fxp_mlp.py` | fixed-point MLP inference: `quantize_mlp`, `FxpMlp` / `FxpLayer`, packed int64 form, `fxp_mlp_eval` (numba, no allocation, no views), `fxp_mlp_forward_batch`, `fxp_mlp_check`, `error_bound` (float vs fixed), `mlp_forward_float`, `memh_lines`. A new module next to `mrm/control/fxp.py` (whose scalar kernels it uses; fxp.py is unchanged) |
| `examples/tutorial.ipynb` | the tutorial (stored with its outputs, ~4 KB of text, ~5-10 s with a warm numba cache) |
| `tests/verification/test_gym_env.py` | 20 tests: spaces, actions, observation, reward, resets, replay from seeds (and of one world alone), start jitter, autoreset, env checker, vector wrappers (EnvSpec kept), parity with the engine's Python path |
| `tests/verification/test_gym_env_fxp.py` | 22 tests: fixed point against a big-integer reference, accumulator proof, error bound, packing and its check, NUMBA_DISABLE_JIT, compiled policy == env (hardtanh, linear output beyond +-1, scenario controller reset), compiled policy releases every array |
| `tests/verification/test_gym_env_perf.py` | throughput (marks `slow`, `perf`; the 1e5 target is `xfail(strict=False)`) |
| `tests/verification/test_gym_env_tutorial.py` | executes the notebook in a subprocess (no nbconvert in the venv: a minimal executor is in the file; `python tests/verification/test_gym_env_tutorial.py --write` refreshes the stored outputs) |
| `docs/handoff/rl.md` | this file |

## 2. Public API

```python
# mrm/sim/gym_env.py
class MRMEnv(gymnasium.vector.VectorEnv):
    metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP, "render_modes": []}
    def __init__(self, config="g26", num_envs: int = 8, *, overrides=None, alts=None, scenario=None,
                 max_episode_steps: int | None = 1000, history_blocks: int = 4, max_delta_heater_codes: int = 256,
                 max_delta_bias_codes: int = 64, action_type: str = "continuous", start: str = "warm",
                 start_jitter_codes: int = 0, uncertainty: str = "sample", variation: str = "sample",
                 reward: RewardSpec | Mapping | None = None, metrics: Mapping | None = None,
                 truth_fields: Sequence[str] | str = DEFAULT_TRUTH_FIELDS, latency_ticks: int = 0,
                 n_threads: int | None = None, substep_policy: str | None = None, oracle: bool = True,
                 record=None, sh_tol_k: float | None = None)
    def reset(self, *, seed: int | None = None, options: Mapping | None = None) -> tuple[np.ndarray, dict]
        # options: {'seeds': int | {device, uncertainty, noise, scenario}, 'start': ..., 'heater_code': (B, N)}
    def step(self, actions) -> tuple[obs (B, D) float32, reward (B,) float64, terminated (B,), truncated (B,), info]
    def compiled_policy(self, mlp: FxpMlp, *, name="mlp_policy", latency_ticks=None) -> CompiledController
    def close(self) -> None
    # attributes: num_envs, single_observation_space (Box float32 (D,)), single_action_space (Box(-1, 1, (A,))
    # or MultiDiscrete), observation_space, action_space, ctrl_spec (ControllerSpec), obs_layout (ObsLayout),
    # action_codec (ActionCodec), frame_skip_ticks, dt_s, step_s, plant (the current Plant), seeds (dict),
    # metrics_spec, reward_spec, cfg; `spec` is gymnasium's (EnvSpec after make / make_vec, else None)
    # info of a step: tick, clipped, controller_reset (B,) bool, truth {...}
class MRMSingleEnv(gymnasium.Env):          # B = 1 view; same keyword arguments without num_envs; .venv, .plant
@dataclass(frozen=True)
class RewardSpec:
    lock_scale_pm: float | None = None      # None: metrics.lock_threshold_pm (5 pm)
    shape: str = "log"                      # 'log': -log1p(u) | 'quadratic': -min(u, clip); u = mean e^2 / s^2
    clip: float = 100.0
    w_lock: float = 1.0
    w_oma: float = 0.0                      # + w_oma * mean relOMA / mean oracle relOMA
    w_heater_per_w: float = 0.0             # - w * mean heater power (W)
    w_wrong_line: float = 0.0               # - w * fraction of the step on a foreign line
    exclude_unreachable: bool = False       # leave rings without a warm-start code out of the env reward
def code_codec(lo: int, hi: int) -> tuple[int, int]          # (c0, s): x = (code - c0) * 2**-s
@dataclass(frozen=True) class ObsLayout: n_rings, history_blocks, samples_per_block, channels, size;
    .channel(name), .slices(), .bounds(), .code_offsets(), .shifts(), .unflatten(obs), .to_codes(obs)
@dataclass(frozen=True) class ChannelCodec: name, offset, size, shape, code_offset, shift, lo, hi
@dataclass(frozen=True) class ActionCodec: channels ((name, a_off, size, max_delta, lo, hi, abi_off), ...), size,
    discrete; .names()
def fxp_policy_action(mlp: FxpMlp, obs) -> np.ndarray        # env action of a fixed-point policy (float32, exact)
mlp_policy_init(spec_i64, spec_f64), mlp_policy_step(p, pf, s, sf, obs, act)   # numba, compiled-controller ABI
ENV_ID = "mrm/MRM-v0"; STARTS = ("warm", "cold", "cold_settled"); DEFAULT_TRUTH_FIELDS; EPISODE_METRICS

# mrm/control/fxp_mlp.py
def quantize_mlp(weights, biases, activations, *, in_fmt="Q0.15", w_bits=16, act_bits=16, out_fmt=None,
                 acc_bits=None, rounding="half_even", overflow="saturate", x_lo=None, x_hi=None,
                 margin=1.25) -> FxpMlp
@dataclass(frozen=True) class FxpMlp: in_fmt, layers, rounding, overflow, proven
    .pack() -> int64 vector, .forward(x_raw) -> raw, .quantize_input(x), .output_value(y_raw), .forward_real(x),
    .describe() -> dict, .resources() -> dict, .memh() -> {name: hex lines}; n_in, n_out, out_fmt, max_width
@dataclass(frozen=True) class FxpLayer: w_raw, b_raw, w_fmt, b_frac, acc_bits, out_fmt, activation, acc_bits_needed
def error_bound(mlp, weights, biases, activations, x_lo=None, x_hi=None, *, input_exact=False) -> MlpErrorBound
    # MlpErrorBound(bound (n_out,), saturation_free, layers)
def mlp_forward_float(weights, biases, activations, x) -> np.ndarray
fxp_mlp_eval(p, base, buf, x_off, y_off, s_off) -> n_out      # numba; no allocation, no views
fxp_mlp_forward_batch(p, x (B, n_in), y (B, n_out))            # numba
fxp_mlp_check(p, base) -> 0 | error code 1..13                  # numba; 0 = fxp_mlp_eval cannot raise (WRAP/SAT)
memh_lines(raw, bits) -> list[str]
ACT_LINEAR, ACT_RELU, ACT_HARDTANH; packed layout constants H_*, L_*, H_SIZE, L_SIZE, MAGIC, VERSION
```

Usage:

```python
from mrm.sim.gym_env import MRMEnv
env = MRMEnv("g26", num_envs=256)                 # or gymnasium.make_vec("mrm/MRM-v0", num_envs=256,
obs, info = env.reset(seed=0)                     #    vectorization_mode="vector_entry_point")
obs, r, term, trunc, info = env.step(env.action_space.sample())
info["truth"]["detuning_m"], info["truth"]["reward_per_ring"]           # Truth only here
mlp = quantize_mlp(weights, biases, ["relu", "relu", "hardtanh"])       # a trained policy
runner.run("g26", env.compiled_policy(mlp), None, duration_s=..., batch=64, heater_code="quasi_static", settle=True)
```

## 3. Design: how each 07 P1-22 item is met

1. **VectorEnv over the plant batch, no subprocess workers.** The B sub-environments are the B worlds of one Plant.
   A step = one call of the compiled kernel on the engine's Python-controller path (`MODE_PYTHON`): the kernel
   stops at the next controller edge after releasing the sensors; the env reads the flat `S.obs` (ABI layout),
   writes `held + delta` into `S.act` and resumes, which pushes the command into the controller pipeline. This is
   exactly `mrm.sim.runner._run_python` with the env as the controller: `test_env_steps_equal_the_engine_python_
   controller_path` (latency 0 and 3 ticks) shows every metric and Truth field bit-identical to `runner.run` with a
   Python controller issuing the same commands.
2. **Spaces from the ChannelSpecs.** Observation: `tx.monitor.samples` (K blocks of M), `tx.heater.applied`,
   `tx.bias.applied` (when exposed) and every `temp.<name>`; each normalized by the power of two that covers the
   channel's [min, max]: x = (code - c0) 2^-s, c0 = min + 2^s, 2^(s+1) >= max - min + 1 (`code_codec`). Bounds are
   the exact images of [min, max]; x is exact in float32 (and in Q0.15 for converters up to 16 bits). Actions: the
   actuator channels bound to the controller among `tx.heater` and `tx.bias`, each with a bounded delta (default 256
   and 64 codes; at most the channel span). A min-max normalization was not used: its 1/(2^16 - 1) scale is not
   exact in binary, and the compiled fixed-point policy could not reproduce it with shifts.
3. **frame_skip = the controller period** (`spec.period_ticks`; g26 10 ticks). No action-repeat option.
4. **Bounded delta codes plus the bias.** Continuous Box(-1, 1): delta = rint(clip(a) max_delta) (half to even);
   discrete MultiDiscrete(2 max_delta + 1). The bias joins the action only when the config binds `tx.bias` to the
   controller (`controller_interface.bind.tx.bias=controller`, `scenario.bias_v=null`); in g26 the scenario owns it.
   The kernel clips `held + delta` to the channel range and counts (07 P1-19), and the clipped value is the next
   held command (`info['clipped']` per step).
5. **Observation = last K sample blocks, applied codes, temperature codes, normalized.** Entries beyond
   `tx.monitor.count` are filled with the held `tx.monitor.latest` (M codes per block always). At a reset the history
   is K copies of the first block, which is empty (code 0) at the first edge, as for any controller at that edge.
   `test_observation_is_the_released_sensor_channels` rebuilds every observation from `Plant.sample_sensors()` (the
   controller's own view) with an independent history.
6. **Reward from Truth, computed in the kernel; Truth only in `info['truth']`.** The per-tick work is the kernel's
   metric accumulators (`mrm.sim.metrics.metrics_tick_row`: sum of squared lock error, relOMA, oracle relOMA, heater
   energy, wrong-line ticks, every tick, from the Truth row); the env takes their differences over the step (B x N
   subtractions) and applies `RewardSpec`. No kernel change was needed. `test_reward_equals_the_reward_of_the_truth_
   trace` recomputes the reward from per-tick Truth records (rtol 1e-9). Lock reference: default `'oracle'` (each
   ring's best open-loop-stable relOMA point, refreshed every 1 ms by the kernel); `metrics={'lock_reference':
   'line' | 'target', ...}` overrides it. `info['truth']` per step: the selected Truth fields (default detuning,
   relOMA, oracle relOMA, heater code, line_on, ring temperature), the lock reference and error, the step rms lock
   error, the per-ring reward; at the truncation step the episode metrics (`EPISODE_METRICS`); at a reset only
   `warm_start_no_code`.
7. **Domain randomization per reset.** Every reset (explicit or automatic) calls `params.realize` with new seeds
   (uncertainty 'sample' | 'lhs' | 'nominal', variation 'sample' | 'off') and builds a new Plant (the kernel types do
   not change, so the cached kernel is reused). Seeds: base = (b0 + r) B for reset r, b0 drawn at a seeded reset
   from the env's np_random; world b uses base + b (07 P1-13); worlds never repeat between resets (2^31 / B / 2
   blocks). `options={'seeds': ...}` pins a reset (evaluation). A scenario callable `(rng) -> Scenario` is called
   per reset with a generator seeded by the scenario seed (scenario randomization; A1 calibrated per plant).
8. **Starts.** 'warm' (default): the quasi-static oracle lock codes of each world (`runner._warm_start_codes`, the
   private form that also returns the no-code mask) and `Plant.settle` on the hot branch: the hand-off contract of
   the feedback baselines. 'cold': the config's reset protocol (g26 laser_ramp), heaters at code_min. 'cold_settled':
   code_min settled on the cold branch (the scorecard's cold start). `start_jitter_codes` adds a uniform offset.
9. **Episodes.** All worlds truncate together after `max_episode_steps`; NEXT_STEP autoreset (the next step resets
   every world and returns reward 0, terminated = truncated = False). No per-world termination (section 6 item 1).
10. **Gymnasium's env checker** (`gymnasium.utils.env_checker.check_env`) passes on `gymnasium.make("mrm/MRM-v0")`
   (with its spec, so the strict determinism checks run) and on `MRMSingleEnv` directly, without a single
   gymnasium warning. Gymnasium 1.4 has no checker for vector envs; the vector API is tested with `make_vec`,
   `RecordEpisodeStatistics` and `DictInfoToList`.
11. **Fixed-point MLP helper.** `quantize_mlp` chooses every format; the accumulator width is the exact worst case
   over every raw input of the format (Python integers), so an overflow is impossible and the evaluation may sum in
   plain int64 (equal bit for bit to the per-addition fitted path, which runs when an accumulator is set narrower by
   hand: WRAP / SATURATE / CHECKED as in fxp.py). Biases are held at the accumulator's fraction bits (no rounding
   when they enter). Activations act on the rounded wide value before the output fit (an RTL datapath's order; it
   differs from `fx.fx_matvec` only under WRAP of the output). `error_bound` gives a rigorous bound on
   |y_fixed - y_float| over an input box (interval propagation; derivation in the module docstring); measured errors
   are ~10x below it. `memh()` / `resources()` serve the RTL side.
12. **Deployment.** `MRMEnv.compiled_policy(mlp)` builds a CompiledController whose numba step keeps the K-block
   history in its state, builds the env observation as raw Q0.f integers by shifts (exact), runs `fxp_mlp_eval` and
   adds round_half_even(clip(y, -1, 1) max_delta) to the held command (the clip is the env's own clip of an action
   to [-1, 1], on the raw output). `test_compiled_policy_on_the_engine_equals_the_env`: the same fixed-point policy
   through the env (Python path) and inside the kernel (`runner.run`, compiled path): every metric and Truth field
   bit-identical, for a hardtanh output, a linear output that reaches |y| > 2, and a scenario controller reset
   mid-run; also under NUMBA_DISABLE_JIT=1 (subprocess test).
13. **Scenario controller reset** (`fault.controller_reset`, 07 P1-10). The env holds the controller state of a delta
   policy (the held command and the K-block history). At the edge where the fault arrives it restarts both, as the
   kernel restarts a compiled controller (held heater / bias commands re-filled from the latched DAC registers,
   `kernel._sync_act_a`; history restarted from that edge's block), and flags the worlds in
   `info['controller_reset']` so that a policy with its own memory (an RNN) can reset that too.

## 4. Measurements (8-core M2, AC power, other agents running; `tests/verification/test_gym_env_perf.py -m perf -s`)

| B | env steps/s | ms per step (kernel / env Python) | same ticks open loop (us per world-tick) | plant-only bound (steps/s) | reset (s) | load |
|---|---|---|---|---|---|---|
| 1 | 3 850 | 0.26 (0.21 / 0.05) | 0.10 ms (9.6) | 10 000 | 0.2 | 3.4 |
| 16 | 10 700 | 1.50 (1.41 / 0.09) | 0.35 ms (2.2) | 46 000 | 0.4 | 3.6 |
| 64 | 19 500 | 3.28 (3.11 / 0.17) | 1.45 ms (2.3) | 44 000 | 1.0 | 1.8 |
| 256 | **24 800-26 200** | 9.8-10.3 (9.5-10.0 / 0.30-0.37) | 6.0-6.4 ms (2.3-2.5) | 40 000-43 000 | 3.4-3.6 | 2.5-5.6 |
| 512 | 27 500 | 18.6 (18.1 / 0.55) | 12.3 ms (2.4) | 41 700 | 6.9 | 4.3 |
| 1024 | 27 800 | 36.9 (35.7 / 1.15) | 24.9 ms (2.4) | 41 000 | 15.2 | 4.6 |

- **07 target (>= 1e5 env steps/s at B = 256): not met; measured 2.5-2.6e4.** Two parts:
  (a) the plant itself: one step is B x 10 world-ticks at ~2.3-2.5 us per world-tick on 8 threads (the engine's B =
  64 figure; 4 performance + 4 efficiency cores), so even with no overhead B = 256 is capped at ~4.0-4.3e4 steps/s.
  1e5 needs ~1 us per world-tick on the batch (2.3-2.5x faster plant).
  (b) the per-call overhead of the Python path: ~3.5 ms of the 9.8 ms step (kernel calls 9.5 ms against 6.0 ms for
  the same ticks in one open-loop call), growing with B (~11 us of wall time per world and call at B = 512). It is
  the per-block setup of `kernel._world_block` (tables rebuilt at every call, i.e. every 10 ticks here) plus the
  unboxing of the ~270 KParams/KState arrays per chunk call (~0.1 ms each, under the GIL; 8 chunks). Requests 7.1.
  The env's own Python work (observation, reward, info) is 3 % of a step (0.3 ms at B = 256).
- More worlds do not help beyond B ~ 256 (overhead and plant both scale with B).
- Earlier checks of the dispatch: 16 chunks instead of 8 over the 8 threads gave 5-10 % (noisy; not used: it
  needs the engine's private chunk function); 4 or 16 threads were not faster than 8; the detuning histogram off
  (the env always turns it off) saved ~25 % at B = 256.
- **Reset cost**: realize 0.09 s + Plant 0.12 s + warm start; at B = 256 the warm start's `quasi_static` (a Python
  loop over worlds and rings: 32 768 `ring_exact` calls) takes 3-5 s of the 3.4-5.3 s. With 1000-step episodes
  (~10 s at B = 256) that is a 25-35 % overhead; `start='cold'` costs 0.2 s. Request 7.2.
- Notebook: ~5 s end to end (warm numba cache), 8 code cells, 3.9 KB of text output.

## 5. Tests (all through tools/memguard.py; warm numba cache; the builder's run, before the review: section 9.3)

- `tests/verification/test_gym_env.py`: 18 passed (~10 s). Includes check_env, the engine-path parity (bit for bit,
  latency 0 and 3), the reward against Truth records, determinism, autoreset, resets, the reward exclusion.
- `tests/verification/test_gym_env_fxp.py`: 18 passed (~10 s). Bit-exact against an independent big-integer
  reference in all 5 rounding modes and WRAP / SATURATE / CHECKED (narrow accumulators that do overflow), the
  accumulator proof, the error bound (5 nets x 2 roundings x 3000 inputs + box vertices), packing / memh /
  resources, NUMBA_DISABLE_JIT subprocess parity of the helper and of env + compiled policy, the compiled policy's
  features equal to the env observation at every edge, env == engine for the same fixed-point policy.
- `tests/verification/test_gym_env_tutorial.py`: 2 passed (~8 s).
- `tests/verification/test_gym_env_perf.py -m perf`: 1 passed, 1 xfailed (the 1e5 target).
- Both functional files also pass with `NUMBA_DISABLE_JIT=1` for the whole file (section 8).

## 6. Deviations and doubts

1. **No per-world termination or partial reset.** Every world truncates at the same step and all are redrawn
   together: a per-world redraw would need a per-world re-realization of the batch arrays inside one Plant (request
   7.3). A lost ring stays in its episode (it can be re-acquired).
2. **Reward split:** per-tick accumulation in the kernel (existing accumulators), per-step shaping (log1p, weights)
   in NumPy on the B x N differences. If 07's "computed in the kernel" is read as "the shaped scalar in numba", a
   kernel hook would be needed; I judged the accumulators to be the in-kernel computation (no per-tick Truth leaves
   the kernel, nothing per tick runs in Python).
3. **Throughput target not met** (section 4).
4. **Warm start in sampled worlds**: rings whose quasi-static lock code is out of [code_min, code_max] (4 of 32 in
   the tutorial's worlds; 13 of 64 for g26 seeds 7 per ENGINE.md) start at code_min, far from any lock; their
   lock error dominates the mean reward unless `RewardSpec(exclude_unreachable=True)`. `info['truth']
   ['warm_start_no_code']` flags them. The hand-off codes (`tx.heater.applied` at the first edge) come from the
   oracle's quasi-static solution, i.e. they carry privileged information into the observation, exactly as for the
   feedback baselines' warm start (their hand-off contract); use `start='cold'` for a task without it.
5. **Warm start under an aggressor**: the quasi-static codes leave out the scenario's aggressor power at t = 0
   (runner.warm_start_codes docstring); `Plant.settle` includes it. With an aggressor that is hot at t = 0 (A1 near
   its cusp, or a scenario callable that randomizes the A1 phase) the ring starts settled but off its lock by the
   aggressor's shift at that phase. Start A1 at its cool minimum (`phase_s = P / 2`, as the tutorial does) for a
   clean hand-off.
6. **The oracle lock reference**: the warm start (quasi-static target) and the engine's oracle point differ by
   ~1-15 pm (nominal; ENGINE.md section 3), more in sampled worlds, so a "hold" policy scores a nonzero lock error;
   a policy must learn to move to the oracle point.
7. **Gymnasium's env checker** runs on the B = 1 view only (no vector checker exists in gymnasium 1.4).
8. **Notebook execution** uses the executor in test_gym_env_tutorial.py (nbconvert / nbclient are not installed; I
   did not install packages). The notebook stores only text stream outputs.
9. `gymnasium.make("mrm/MRM-v0", max_episode_steps=n)` wraps a TimeLimit (gymnasium consumes that keyword); the
   env's own limit (default 1000) still applies. `make_vec` passes it to `MRMEnv`.
10. **Replay of one world alone** (review). An episode replays bit for bit from `env.seeds`
   (`reset(options={'seeds': env.seeds})`), and world b of a batch replays alone at B = 1 with the seeds base + b
   from a COLD start (tested). From a SETTLED start (`start='warm'` or `'cold_settled'`, or the `heater_code`
   option) it does not replay bit for bit: `Plant.settle` iterates the whole batch until the largest ring-temperature
   change over every world is below 2 mK, so a world's iteration count and settled state depend on its batch
   companions (measured: g26 seeds 0, B = 4 vs alone: 16 vs 7-16 iterations, start temperatures differ by up to
   0.9 mK, detuning by 0.014 pm after 50 steps). The same holds for `runner.run(..., settle=True)`. Request 7.5.
   **Resolved 2026-10-16 (engine review, engine-5):** `Plant.settle` stops each world at its own iteration, and
   settled starts replay alone bit for bit (test_gym_env.py::test_an_episode_replays_from_its_seeds_and_a_world_alone
   checks warm and cold_settled).
11. **Scenario randomization is per reset, not per world**: a scenario callable is called once per reset (one
   Plant has one scenario) with a generator seeded by the base scenario seed; every world of the batch shares the
   draw (a world replayed alone gets the draw of its own seed instead).

## 7. Requests

1. **engine / perf** (mrm/sim/kernel.py, mrm/plant/plant.py): the Python-controller path's per-call cost, ~35 % of
   an env step at B = 256 (section 4b): (a) keep the per-block tables of `_world_block` (thermal tables, dead-lag
   init, heater split, ...) across calls when P is unchanged (rebuilt at reset or when `Plant.P` is replaced), or
   let a `resume` call skip the setup; (b) cut the per-call unboxing (one chunk call per thread unboxes ~270
   arrays, ~0.1 ms under the GIL); (c) consider dynamic chunk scheduling for the 4 P + 4 E cores. For the 07 target
   itself the plant needs ~1 us per world-tick at B = 256 (2.3-2.5x); an opt-in RL fidelity would have to be a
   documented approximation (perf.md 5.4 rule).
2. **runner owner** (mrm/sim/runner.py): vectorize `quasi_static` over worlds (3-5 s of every B = 256 reset), and a
   public form of `_warm_start_codes` that returns the no-code mask (the env imports the private one).
3. **engine / params** (optional): a Plant method to replace the worlds of a mask with newly realized ones (same
   array types) would allow per-world termination and autoreset (section 6 item 1).
4. **integrator**: ARCHITECTURE section 2: add `control/fxp_mlp.py` (fixed-point MLP, 07 P1-22) next to fxp.py;
   docs/ENGINE.md section 10 lists the gym environment as not built: it is now; CHANGELOG entry; the fxp.py module
   docstring calls `fx_matvec` "the fixed-point MLP helper of 07 P1-22": the helper is now `fxp_mlp.py` (fx_matvec
   is unchanged and unused by it).
5. **plant owner** (mrm/plant/plant.py, optional, review): `Plant.settle` stops when the largest change over the
   whole batch is below `tol_k`, so a world's settled start depends on the other worlds of its batch (section 6
   item 10). A per-world stop (freeze each world's thermal state once its own change is below `tol_k`) would make a
   settled world replay alone at B = 1; it changes settled starts for B > 1 (the regression scenarios and golden
   traces that use `settle=True` with B > 1 would need a refresh and a CHANGELOG line), so it is a decision for the
   plant owner and the integrator, not a defect of the env. **Done 2026-10-16** (engine review; CHANGELOG): no
   golden trace uses B > 1; the regression scenario mc8_pi moved (docs/CHANGELOG.md).

## 8. Test status at the builder's hand-off (2026-10-10; superseded by section 9.3)

- `pytest tests/verification/test_gym_env.py tests/verification/test_gym_env_fxp.py
  tests/verification/test_gym_env_tutorial.py`: 38 passed in ~20 s.
- The same two functional files with `NUMBA_DISABLE_JIT=1`: 34 passed, 2 skipped (the two subprocess JIT / no-JIT
  comparisons skip themselves when JIT is off) in 136 s.
- `pytest tests/verification/test_gym_env_perf.py -m perf -s` (load 3.1): 1 passed, 1 xfailed; B = 256:
  24 800-25 200 env steps/s (10.2-10.3 ms per step: kernel 9.8-10.0, env Python 0.33-0.37); the same ticks open
  loop 6.3 ms (2.47 us per world-tick; at most ~40 500 steps/s); reset 3.5 s.
- `pytest --collect-only tests`: 1854 tests collected (no collection error). The full fast suite was not run here
  (other agents' files are mid-edit; my modules are imported only by my tests).
- No engine, runner or other owner's file changed; the kernel cache was never invalidated by this task.

## 9. Adversarial review (2026-10-10)

### 9.1 Findings and fixes (all in the files of this task)

1. **`MRMEnv.spec` collided with gymnasium's `VectorEnv.spec`** (major; fixed). The env stored its ControllerSpec
   in `spec`, the attribute gymnasium reserves for the EnvSpec. `gymnasium.make_vec` sets `env.unwrapped.spec` to
   the EnvSpec after construction, so before the first reset `repr(env)` raised AttributeError
   (`'EnvSpec' object has no attribute 'n_rings'`) and `compiled_policy` would too; every reset then put the
   ControllerSpec back, so `env.spec.id` / `env.spec.max_episode_steps` (read by wrappers and RL libraries) raised.
   Fix: the controller interface is now `MRMEnv.ctrl_spec`; `spec` is left to gymnasium. Test:
   `test_vector_api_and_gymnasium_wrappers` checks both before and after a reset.
2. **The compiled MLP policy did not clip its output to [-1, 1]** (major; fixed). The env clips every action to
   [-1, 1] before decoding it; `mlp_policy_step` decoded the raw output unclipped, so any policy whose output can
   leave [-1, 1] (a linear output layer, or a user `out_fmt` with integer bits) gave different commands inside the
   kernel than through the env (measured: final heater codes differed by up to 750 codes after 20 steps), and a
   wide output format could overflow int64 in `y * max_delta`. Fix: the step clips the raw output to
   [-2^f_out, 2^f_out] (exactly the env's clip). Test: the `linear_beyond_unit` case of
   `test_compiled_policy_on_the_engine_equals_the_env` (outputs reach |y| > 2).
3. **A scenario controller reset made the env and the compiled policy diverge** (minor, fault scenarios only;
   fixed). On `fault.controller_reset` the kernel restarts a compiled controller (init state, held command re-filled
   from the DAC registers); the env only reported and cleared the flag, keeping its held command and history. Fix:
   `MRMEnv._controller_reset` restarts both per world (section 3 item 13); a flag raised before the first edge is
   cleared at the reset. Test: the `controller_reset` case of the parity test (reset at 150.5 us, edge 160 us).
4. **`fxp_mlp_check` passed packed vectors that make `fxp_mlp_eval` raise inside the kernel** (minor; fixed). It
   checked sizes and offsets only; a bad rounding / overflow / activation code, an output with more fraction bits
   than its accumulator (negative shift), out-of-range widths or a weight whose product with the largest input
   reaches 2^62 (fx_mul's ValueError) passed it. The compiled policy relies on it at init so that no exception can
   leave the tick loop. Fix: codes 9-13; test `test_packed_check_rejects_vectors_that_could_raise_in_the_kernel`.
   `quantize_mlp` now also refuses `margin < 1` (its docstring already required >= 1).
5. **`start_jitter_codes` was drawn from one batch generator** (minor; fixed): world b's jitter depended on B, so a
   world could not be replayed alone, and the docstring said "seeded by the world seed". Now one generator per world
   (device seed base + b), clipped explicitly to the DAC range (the plant clipped silently before). Tests:
   `test_an_episode_replays_from_its_seeds_and_a_world_alone`,
   `test_start_jitter_is_per_world_and_clipped_to_the_dac_range`.
6. Small: `truth_fields` accepts a single field name; `_build_obs_layout` checks the ABI field sizes it relies on
   (monitor block ring-major, N x M); the tutorial's stated run time (was "about 1 min"; it is ~5-10 s).

Checked and found correct: the reward against per-tick Truth records (log / quadratic, oma, heater terms); units
(m, W, J); the observation codec (exact in float32 and Q0.15, inside the declared Box on g26, g25_test_vehicle,
teaching_basic and g26_h86_sensor for all three starts); NEXT_STEP autoreset; the seed scheme (no repeats; pinned
replay bit for bit); no memory growth over 40 autoresets (B = 16: 219 -> 220 MB); no reference leak of the compiled
policy in the runtime-free tick loop (new test `test_compiled_policy_releases_every_array`); the fixed-point helper
against its big-integer reference and its error bound. Not changed: the reward's wrong-line term has no test
against Truth records (no Truth field holds the kernel's wrong-line flag directly).

Open, not fixed here: the settled-start dependence on the batch (section 6 item 10, request 7.5), the throughput
target (section 4), per-world termination (section 6 item 1).

### 9.2 Performance after the review (load 1.8-4.8; same machine)

`test_gym_env_perf.py -m perf -s`: B = 256: 26 000-26 400 env steps/s (9.7-9.8 ms per step: kernel 9.4-9.5, env
Python 0.29-0.30); the same ticks open loop 5.8 ms (2.27 us per world-tick, at most ~44 000 steps/s); reset
3.4-3.6 s; peak RSS 0.37 GB. The review's changes add one (B,) comparison per step.

### 9.3 Test status after the review (warm numba cache; gym_env and fxp_mlp caches cleared and recompiled)

- `test_gym_env.py` + `test_gym_env_fxp.py`: 42 passed in 15 s (JIT); with `NUMBA_DISABLE_JIT=1`: 40 passed,
  2 skipped (the subprocess JIT / no-JIT comparisons) in 195 s.
- `test_gym_env_tutorial.py`: 2 passed (notebook refreshed with `--write`: 5.2 s, 3.9 KB of text).
- `test_gym_env_perf.py -m perf`: 1 passed, 1 xfailed (the 1e5 target).
- Fast suite (`tools/memguard.py -- pytest -q -m "not slow"`, whole tree with the other agents' uncommitted work):
  1667 passed, 74 skipped, 3 xfailed, 4 failed in 273 s. None of the failures is in this task's files: VAL-8a/b/c
  (the known extraction targets, integrator.md) and `tests/golden/test_golden.py::
  test_every_golden_file_is_in_the_changelog` (the new golden files of the hardware task have no CHANGELOG line yet).
- No engine, runner or other owner's file was changed by the review; the kernel cache was not touched.

## 10. Integration (2026-10-10, integrator; docs/handoff/integrator.md section 8)

Resolved: request 7.4 (ARCHITECTURE section 2 names control/fxp_mlp.py; ENGINE.md section 13 describes the env and
section 10 no longer lists it as not built; CHANGELOG entry; the fxp.py docstring no longer calls fx_matvec the MLP
helper) and the public part of 7.2: `runner.warm_start_codes(plant, cfg, mode=..., return_no_code=True)` returns
(codes, no_code); gym_env uses it instead of the private `_warm_start_codes` (whose return now has four items).

Changes that reach the env through the runner (results of the env unchanged in every test):
- `runner.quasi_static` checks the array heater supply limit (07 P1-12). A warm start in a world whose lock powers
  exceed it warns (`warm_start_codes: ... exceed the array heater supply limit`; the env filters warnings that start
  with `warm_start_codes`) and keeps its codes, but its rings settle below their locks. About a third of the g26 LHS
  worlds are supply-limited at 25 degC (128 worlds, variation off: 42; 29 of the 32 stacked_eic worlds). The env does
  not report it yet: add `info['truth']['warm_start_supply_limited']` (the mask is
  `runner._warm_start_codes(...)[2]`) and consider excluding those worlds from the reward like the no-code rings.
- `heater_code="quasi_static_reachable"` / `quasi_static(order="reachable")`: a ring whose heater-up order needs a
  negative heater power locks one order bluer instead of starting at code_min (g26 seeds 1000, B = 8: 10 -> 3 rings
  without a code; since 2026-10-16 the hop repeats while hopped rings push others below zero: 10 -> 0). The env's
  'warm' start still uses 'quasi_static' (its tests and the tutorial outputs depend on it); switching is a one-line
  change in `MRMEnv._reset_worlds` for the rl owner.

Still open: 7.1 (Python-path per-call cost), 7.2 (vectorized quasi_static), 7.3 (per-world reset), the 1e5 steps/s
target. (7.5, the per-world settle stop, was done on 2026-10-16: engine review.)
