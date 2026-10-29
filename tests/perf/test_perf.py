"""Compute budget of the compiled kernel: how many simulated seconds the engine runs per wall-clock second.

Read this first. These tests do not block CI (tests/perf is a report, not a gate). Run them with
``pytest tests/perf -m slow -s`` to print the measured numbers.

The targets, for B = 1 world, N = 8 rings, L = 8 lines on the full plant:

- at least 0.1 simulated s per wall s at a fixed 1 us step;
- at least 20 simulated s per wall s with 100 us automatic substeps (substep_policy "auto");
- a Monte Carlo of 64 seeds x 1 simulated s in under 5 min.

The target tests are ``xfail(strict=False)``: they report whether this machine (shared, often loaded) meets them, and
they print the measured number either way. The g26 preset samples its ADC at 250 kHz and runs its controller at
100 kHz. An auto substep stops at the next controller edge (10 us), so the 100 us figure is measured with a 10 kHz
controller.
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pytest

from mrm import params
from mrm.control.api import ControllerSpec
from mrm.plant.plant import Plant
from mrm.scenarios import registry as scenario_registry
from mrm.sim import kernel

pytestmark = pytest.mark.slow

WARMUP_TICKS = 300          # open-loop ticks before the timed run


def _plant(n_worlds, policy, *overrides, oracle=True):
    """g26 Plant with ``n_worlds`` sampled worlds (seeds 7), sim.reset_ramp_s = 1e-4 and the given substep policy."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load("g26", overrides=["sim.reset_ramp_s=1e-4", *overrides])
        pp = params.realize(cfg, n_worlds, seeds=7)
    scenario = scenario_registry.from_config(cfg)
    spec = ControllerSpec.from_config(cfg, n_batch=n_worlds, nominal=cfg, scenario_drives=scenario.drives,
                                      scenario_offsets=scenario.offsets)
    return Plant(pp, scenario, spec, substep_policy=policy, oracle=oracle, histogram=False)


def measure(n_worlds, policy, n_ticks, *overrides, n_threads=None, oracle=True):
    """Simulated seconds per wall second (whole bank) and wall us per world-tick, after a warm-up, open loop."""
    plant = _plant(n_worlds, policy, *overrides, oracle=oracle)
    run_block = plant._fn()
    done = np.zeros(n_worlds, np.int64)          # ticks advanced per world (written by the kernel)

    def advance(n):
        run_block(kernel.null_step, kernel.null_step, plant.P, plant.S, np.int64(plant.tick), np.int64(n),
                  np.int64(kernel.MODE_OPEN), np.int64(0), done, n_threads=n_threads)
        plant.tick += int(done[0])
    advance(WARMUP_TICKS)
    t = time.perf_counter()
    advance(n_ticks)
    wall_s = time.perf_counter() - t
    sim_s = n_ticks * plant.dt
    diag = plant.diagnostics()
    return {"B": n_worlds, "policy": policy, "sim_s_per_wall_s": sim_s / wall_s,
            "us_per_world_tick": wall_s / (n_ticks * n_worlds) * 1e6,
            "substeps_per_tick": float(diag["substeps"].mean()) / max(float(diag["ticks"].mean()), 1.0),
            "load": os.getloadavg()[0]}


def _report(r):
    print(f"\n[perf] B={r['B']} {r['policy']}: {r['sim_s_per_wall_s']:.4f} sim-s/wall-s, "
          f"{r['us_per_world_tick']:.2f} us per world-tick, substeps/tick {r['substeps_per_tick']:.3f}, "
          f"load {r['load']:.1f}")


@pytest.mark.xfail(strict=False, reason="compute-budget target (0.1 sim-s per wall-s); depends on machine load")
def test_fixed_step_b1_target():
    r = measure(1, "fixed", 4000)
    _report(r)
    assert r["sim_s_per_wall_s"] >= 0.1


@pytest.mark.xfail(strict=False, reason="compute-budget target with auto substeps (20 sim-s per wall-s); depends on "
                                        "machine load")
def test_auto_step_b1_target():
    r = measure(1, "auto", 20000, "controller_interface.update_rate_hz=10000.0")
    _report(r)
    assert r["sim_s_per_wall_s"] >= 20.0


@pytest.mark.xfail(strict=False, reason="compute-budget target (64 seeds x 1 s in 5 min); depends on machine load")
def test_monte_carlo_64_target():
    r = measure(64, "fixed", 300)
    _report(r)
    assert 1.0 / r["sim_s_per_wall_s"] <= 300.0


def test_report_table():
    """Prints the timing table: B = 1 and B = 64 (N = L = 8) under the fixed and the auto substep policy."""
    rows = [measure(1, "fixed", 3000), measure(1, "auto", 3000), measure(64, "fixed", 200), measure(64, "auto", 200)]
    for r in rows:
        _report(r)
        assert r["sim_s_per_wall_s"] > 0
