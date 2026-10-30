# archived learning interface

This section describes the preserved learning feature. use archive.rl.gym_env and archive.rl.fxp_mlp for repository-local imports.

## 13. The RL environment

`archive.rl.gym_env.MRMEnv` is a Gymnasium `VectorEnv`: its B sub-environments are the B worlds of
one `Plant`, with no subprocess workers. `MRMSingleEnv` provides the B = 1 view. Both
`gymnasium.make("mrm/MRM-v0")` and
`gymnasium.make_vec("mrm/MRM-v0", num_envs=..., vectorization_mode="vector_entry_point")` work.

```python
from archive.rl.gym_env import MRMEnv
env = MRMEnv("g26", num_envs=256)                     # warm start, sampled worlds, 1000-step episodes
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
info["truth"]["detuning_m"]                            # Truth only in info, never in obs
policy = env.compiled_policy(mlp)                      # mlp = archive.rl.fxp_mlp.quantize_mlp(...): deploy in-kernel
```

### Observations, actions and reward

One environment step is one controller period: 10 ticks in G26, using the Python-controller path.
The observation contains the last K monitor sample blocks, applied heater codes, bias codes when
used, and temperature codes. Each is scaled by a power of two, exactly representable in float32
and Q0.15.

Actions are bounded changes to heater codes, and to bias codes when bias is bound to the controller.
Reward comes from kernel metric accumulators. `RewardSpec` uses lock error relative to the oracle
point by default, with optional relOMA, heater-power and wrong-line terms. Truth appears only in
`info`, never in observations.

### Reset and deployment

Every reset redraws worlds with new seeds; `options={'seeds': ...}` pins them. All worlds truncate
together. A warm start (`start='warm'`) uses steady-state codes and flags rings without a code in
`info['truth']['warm_start_no_code']`. Other choices are `'cold'` and `'cold_settled'`.

`MRMEnv.compiled_policy(mlp)` turns a fixed-point MLP from `archive.rl.fxp_mlp` into a
`CompiledController`. It runs in the kernel bit-identically to the environment's version.
`fxp_mlp.error_bound` bounds `|fixed - float|`.

### Throughput and limitations

On an eight-core M2 at B = 256, throughput is 2.5–2.6e4 environment steps/s, below the 1e5 target
of the design. The plant itself caps throughput near 4e4 at 2.3–2.5 µs per world-tick; Python
per-call setup adds about 35% of the cost (`docs/handoff/rl.md` section 4).

Resetting 256 worlds takes 3–5 s because `solve_locked_steady_state` loops over worlds in Python.
`examples/tutorial.ipynb` demonstrates the API and quantize-and-deploy path. No RL library is
installed, so the notebook does not train a policy.

