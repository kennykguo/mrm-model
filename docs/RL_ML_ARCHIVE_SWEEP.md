# RL and ML archive sweep

archive completed - see [archive/rl](../archive/rl/README.md). the remainder records the pre-archive inspection.

sweep date - 29 october 2026.
inspection only - no source files were moved or changed.

the learned-controller feature is largely separate from the physical plant and standard controllers.
its two runtime modules are the Gymnasium environment and the fixed-point neural-network helper.
there is no production training entry point or saved policy checkpoint found in this repository.
the PPO research smoke test trains on a separate toy ring model.

## runtime modules

| file | purpose | incoming usage |
|---|---|---|
| mrm/sim/gym_env.py | vector and single-world learning environments, observation/action codecs, reward, reset randomization, compiled MLP deployment | dedicated tests, two mixed-file tests, tutorial, documentation |
| mrm/control/fxp_mlp.py | quantization, integer MLP inference, packed weights, error bounds, resource estimates, memory-image export | gym_env.py, dedicated fixed-point learning tests, one mixed-file test, tutorial |

gym_env.py imports fxp_mlp.py and the existing plant, kernel, controller ABI, metrics, channel, and fixed-point modules.
fxp_mlp.py imports the general fxp.py arithmetic, NumPy, and Numba.
no other production module imports either learning module.
package initializers mention them in comments but do not load them.
Gymnasium registration happens in gym_env.py and names the two classes through mrm.sim.gym_env entry-point strings.
there is no built-in RL or MLP controller registration or learning-specific experiment configuration found.

## archive together

- mrm/sim/gym_env.py.
- mrm/control/fxp_mlp.py.
- tests/verification/test_gym_env.py - environment, resets, observation/action layout, rewards, API, and engine parity.
- tests/verification/test_gym_env_fxp.py - inference, quantization, packed validation, overflow proof, and compiled deployment.
- tests/verification/test_gym_env_perf.py - environment throughput.

archived code that should remain runnable needs its internal import and Gymnasium entry-point strings changed to the archive location.
the project packages mrm* only, so code outside mrm is not installed by the current package configuration.
keep archived tests outside the default tests directory, or exclude them from default collection.

## shared files that need selective edits

| file | learning-specific part | part to retain |
|---|---|---|
| examples/tutorial.ipynb | sections 6 and 7, their two code cells, and learning links in the final cell | sections 1 through 5 on controller templates, runs, metrics, scorecards, and comparisons |
| tests/verification/test_gym_env_tutorial.py | RL wording and execution of learning cells | the notebook executor and tests for the remaining general tutorial, under a general tutorial test name |
| tests/verification/test_controllers_review.py | test_fxp_mlp_derives_its_accumulator_proof and test_reward_docstring_values | the other controller, oracle, reset, permission, and documentation checks |
| pyproject.toml | optional rl extra containing gymnasium | base dependencies and the other extras |
| README.md | rl installation-extra description | installation and normal simulation instructions |
| docs/ENGINE.md | section 13 and its RL examples | general engine, controller, calibration, and experiment documentation |
| docs/ARCHITECTURE.md | gym_env.py and fxp_mlp.py module entries | the core architecture |
| mrm/sim/__init__.py and mrm/control/__init__.py | module-reading-order comments about learning | existing lazy exports and shared control modules |
| mrm/control/fxp.py | comment referring to fxp_mlp | all arithmetic and fixed-point PI functionality |

refresh generated source indexes and diagrams that quote removed module paths after any archival move.
keep historical changelog and handoff references as history, with an archive pointer where helpful.

## research and historical material

- docs/handoff/rl.md - the feature handoff, interface details, measurements, and old verification results.
- docs/research/05_tools_smoke/t11_closed_loop_numba_gym.py - separate simplified ring model and Gymnasium smoke test.
- docs/research/05_tools_smoke/out/t11_closed_loop_numba_gym.json - its stored measurement output.
- docs/research/05_tools_smoke/t14_sb3_ppo.py - PPO trains for 4096 steps on the t11 toy environment, without saving a trained policy.
- docs/research/05_tools.md - library evaluation and historical Gymnasium, Stable-Baselines3, and PyTorch results.
- docs/research/07_design_critique.md - the original learning-interface requirement.

keep the t11 and t14 scripts together if archiving them because t14 imports t11 by its local filename.
PyTorch also appears in t08_photontorch.py, a rejected photonics-library evaluation.
JAX appears in t06_sax_wdm_bus.py for photonics calculation. those framework imports do not establish a production ML dependency.

## keep in the active project

- mrm/control/fxp.py - shared fixed-point arithmetic and the fxp_pi controller.
- tests/hdl and the fixed-point PI replay vectors - ordinary controller verification.
- mrm/control/api.py and the compiled-controller ABI - used by standard controllers and external-controller paths.
- mrm/sim/kernel.py, runner.py, metrics.py, and plant.py - shared simulation and scoring infrastructure.
- mrm/validation/oracle.py - target and scoring calculations used outside learning.
- PI, MPC, Kalman, feed-forward, and other hand-written baseline controllers.
- calibration parameter fitting and mrm/control/models.py - physical parameter estimation and nominal physical models.
- docs/research/scripts/06_thermo_optic_weights.py - optical/thermal weighting, not neural-network weights.
- ring-line quantities such as out_rl and RING_LINE_OUTPUT_* - optical data layout, unrelated to reinforcement learning.

## checks and limits

AST import scans found only gym_env.py importing fxp_mlp.py in production code.
text and notebook scans checked module references, training calls, configurations, optional dependencies, and documentation.
model-artifact searches included ignored local files while excluding the virtual environment, dependency directories, and caches.
no saved .pt, .pth, .onnx, .safetensors, .joblib, .pkl, or .pickle model artifact was found.

in fresh Python processes, imports of gymnasium, mrm.sim.gym_env, and mrm.control.fxp_mlp were deliberately blocked.
the core package, plant, runner, kernel, controller API, CLI, fixed-point arithmetic, and PI controller still imported.
the baseline registry loaded all 19 registered controllers and resolved fxp_pi with those imports blocked.
this verifies the import boundary. it is not a numerical test of an archived version, because no archival changes were made.

recommended boundary - archive the optional learning environment and neural inference stack, plus their dedicated tests and learning examples.
retain the physical simulation, general controller interfaces, fixed-point PI arithmetic, and numerical calibration.
