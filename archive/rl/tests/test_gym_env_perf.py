"""Measures the throughput of the batched RL environment against its target (>= 1e5 env steps/s at B = 256).

Read this first
---------------
Not part of CI (machine dependent, like tests/perf): ``pytest tests/verification/test_gym_env_perf.py -m perf -s``
prints the measured numbers. The target test is ``xfail(strict=False)``: it reports whether this machine meets it.
Breakdown printed with it: the kernel calls of the Python-controller path (one per step), the Python work of the env
(observation, reward, info), and the same plant ticks run open loop in one call (the plant-only floor).
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

from mrm.sim import kernel
from archive.rl.gym_env import MRMEnv

pytestmark = [pytest.mark.slow, pytest.mark.perf]

TARGET_STEPS_PER_S = 1e5   # env steps per second (summed over the B worlds) that the environment aims for


def measure(B: int, n_steps: int = 200, **kw) -> dict:
    """Time n_steps env steps at B worlds (after 20 warm-up steps), split into kernel and Python time, and the same
    plant ticks run open loop in one call. Keyword arguments go to MRMEnv. Returns the timings as a dict."""
    env = MRMEnv("g26", num_envs=B, max_episode_steps=None, **kw)
    t = time.perf_counter()
    env.reset(seed=0)
    t_reset = time.perf_counter() - t
    rng = np.random.default_rng(0)
    acts = rng.uniform(-0.02, 0.02, (8,) + env.action_space.shape).astype(np.float32)
    for i in range(20):                                       # warm-up
        env.step(acts[i % 8])
    adv = env._advance
    t_kernel = [0.0]

    def timed(resume):
        t0 = time.perf_counter()
        adv(resume)
        t_kernel[0] += time.perf_counter() - t0
    env._advance = timed
    t = time.perf_counter()
    for i in range(n_steps):
        env.step(acts[i % 8])
    wall = time.perf_counter() - t
    env._advance = adv
    t = time.perf_counter()
    env.plant.run(n_steps * env.frame_skip_ticks, mode=kernel.MODE_OPEN)
    w_open = time.perf_counter() - t
    out = {"B": B, "env_steps_per_s": B * n_steps / wall, "ms_per_step": wall / n_steps * 1e3,
           "kernel_ms_per_step": t_kernel[0] / n_steps * 1e3,
           "python_ms_per_step": (wall - t_kernel[0]) / n_steps * 1e3,
           "open_loop_ms_per_step": w_open / n_steps * 1e3,
           "open_loop_us_per_world_tick": w_open / (n_steps * B * env.frame_skip_ticks) * 1e6,
           "plant_bound_steps_per_s": B * n_steps / w_open, "reset_s": t_reset, "load": os.getloadavg()[0]}
    env.close()
    return out


def _print_measurement(m: dict) -> None:
    """Print the timings of measure() on one line."""
    print(f"\n[gym_env perf] B = {m['B']}: {m['env_steps_per_s']:,.0f} env steps/s "
          f"({m['ms_per_step']:.2f} ms per step: kernel {m['kernel_ms_per_step']:.2f}, "
          f"env Python {m['python_ms_per_step']:.2f}); the same ticks open loop "
          f"{m['open_loop_ms_per_step']:.2f} ms ({m['open_loop_us_per_world_tick']:.2f} us per world-tick, at most "
          f"{m['plant_bound_steps_per_s']:,.0f} steps/s); reset {m['reset_s']:.1f} s; load {m['load']:.1f}")


@pytest.mark.xfail(strict=False, reason="target 1e5 env steps/s at B = 256: the plant alone (~2-3 us per "
                                        "world-tick on 8 threads, 10 ticks per step) caps it near 4-5e4; "
                                        "docs/handoff/rl.md")
def test_env_steps_per_second_at_b256():
    m = measure(256)
    _print_measurement(m)
    assert m["env_steps_per_s"] >= TARGET_STEPS_PER_S


def test_env_overhead_over_the_plant_is_bounded():
    """The env's own Python work stays a small part of a step at B = 256 (it is O(B N) array work per step)."""
    m = measure(256, n_steps=100)
    _print_measurement(m)
    assert m["python_ms_per_step"] < 0.15 * m["ms_per_step"]
