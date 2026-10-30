# archived learning tools

## what is kept here?

the optional learning feature was archived on 29 october 2026.

- gym_env.py exposes sensor observations, heater and bias actions, rewards, resets, and compiled-policy deployment.
- fxp_mlp.py quantizes small neural networks and evaluates their fixed-point arithmetic.
- tests/ preserves the environment, inference, throughput, tutorial, and learning-specific review checks.
- examples/tutorial.ipynb preserves the complete controller and learning tutorial.
- research/ preserves the paired toy Gymnasium and PPO smoke tests and the stored toy result.
- docs/ preserves the learning guide and original handoff.

no production trainer or saved learned policy is included. the PPO smoke script trains a separate toy model.

## what remains active?

the physical plant, simulation kernel, controller ABI, standard controllers, fixed-point PI arithmetic, HDL replay, and calibration remain in mrm.
examples/tutorial.ipynb keeps the general controller workflow and its active test is tests/verification/test_tutorial.py.
the installed package has no rl extra and does not include archive/.

## how can the archive be used?

run from the repository root. install the normal project dependencies and pytest, then add the former optional learning dependency if needed.

```sh
.venv/bin/python -m pip install -r archive/rl/requirements.txt
```

```python
from archive.rl.gym_env import MRMEnv
from archive.rl.fxp_mlp import quantize_mlp
```

importing the archived environment registers mrm/MRM-v0 with archive-local entry points.
all simulation calls still use the current shared plant. this is preserved optional tooling, not a frozen snapshot of the entire simulator.

## how are its checks run?

archived tests are excluded from normal recursive collection. select their paths explicitly.

```sh
.venv/bin/python tools/memguard.py --heavy -- .venv/bin/python -m pytest archive/rl/tests/test_gym_env.py archive/rl/tests/test_gym_env_fxp.py archive/rl/tests/test_controllers_learning_review.py
```

throughput tests are opt-in in archive/rl/tests/test_gym_env_perf.py. they include a performance target that was already unmet.
the full learning tutorial has an opt-in executor in archive/rl/tests/test_gym_env_tutorial.py.
research/t14_sb3_ppo.py additionally needs Stable-Baselines3 and PyTorch and imports the adjacent t11 toy model.

## how could this be restored?

restore the two modules to mrm, their dedicated tests to tests/verification, and the rl dependency extra.
update internal imports and Gymnasium registration entry points together.
merge the learning tutorial sections and the two shared review checks back only if the feature is reactivated.

## what was checked during archival?

all 19 active controller registrations resolve with learning imports blocked. normal collection finds 2184 active tests without importing Gymnasium or archive.rl.
the runtime modules retain their original calculations. only learning imports and Gymnasium entry points changed.
both tutorials execute. their HoldLock example uses the current template state names, and the archived notebook adds the repository root to its import path.
the final active-controller and tutorial check passed all 105 selected tests.

the combined controller and archived learning check found 142 passes and three archived failures. both exact-equality failures also occur with the pre-move modules against the current plant.
the randomized inference test can select a network whose output precision exceeds its accumulator precision. the original and archived quantizers reject the same network.
the archived tests retain these assertions. the throughput and toy PPO checks were not run during archival.
