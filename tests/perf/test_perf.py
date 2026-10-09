"""07 P0-9 compute budget of the compiled kernel (owner: engine). Not part of CI (07 section 4.1: tests/perf does
not block); run with ``pytest tests/perf -m slow -s`` to print the measured numbers.

Targets of 07 P0-9 for B = 1, N = 8, L = 8 on the full plant: >= 0.1 simulated s per wall s at a fixed 1 us step;
>= 20 with 100 us adaptive steps (P0-10 auto policy); a 64-seed x 1 s Monte Carlo in under 5 min. The target tests are
``xfail(strict=False)``: they report whether this machine (shared, often loaded) meets them, and the measured number
is printed either way. The g26 preset samples its ADC at 250 kHz and its controller at 100 kHz, which caps an auto
substep at the next controller edge (10 us), so the 100 us figure is measured with a 10 kHz controller.
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
from mrm.scenarios import registry as scen_registry
from mrm.sim import kernel as K

pytestmark = pytest.mark.slow


def _plant(B, policy, *ov, oracle=True):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load("g26", overrides=["sim.reset_ramp_s=1e-4", *ov])
        pp = params.realize(cfg, B, seeds=7)
    sc = scen_registry.from_config(cfg)
    spec = ControllerSpec.from_config(cfg, n_batch=B, nominal=cfg, scenario_drives=sc.drives,
                                      scenario_offsets=sc.offsets)
    pl = Plant(pp, sc, spec, substep_policy=policy, oracle=oracle, histogram=False)
    return pl


def measure(B, policy, n_ticks, *ov, n_threads=None, oracle=True):
    """Simulated seconds per wall second (whole bank) and wall us per world-tick, after a warm-up."""
    pl = _plant(B, policy, *ov, oracle=oracle)
    fn = pl._fn()
    done = np.zeros(B, np.int64)

    def adv(m):
        fn(K.null_step, K.null_step, pl.P, pl.S, np.int64(pl.tick), np.int64(m), np.int64(0), np.int64(0), done,
           n_threads=n_threads)
        pl.tick += int(done[0])
    adv(300)
    t = time.perf_counter()
    adv(n_ticks)
    w = time.perf_counter() - t
    sim_s = n_ticks * pl.dt
    d = pl.diagnostics()
    return {"B": B, "policy": policy, "sim_s_per_wall_s": sim_s / w, "us_per_world_tick": w / (n_ticks * B) * 1e6,
            "substeps_per_tick": float(d["substeps"].mean()) / max(float(d["ticks"].mean()), 1.0),
            "load": os.getloadavg()[0]}


def _report(r):
    print(f"\n[perf] B={r['B']} {r['policy']}: {r['sim_s_per_wall_s']:.4f} sim-s/wall-s, "
          f"{r['us_per_world_tick']:.2f} us per world-tick, substeps/tick {r['substeps_per_tick']:.3f}, "
          f"load {r['load']:.1f}")


@pytest.mark.xfail(strict=False, reason="07 P0-9 target; depends on machine load")
def test_fixed_step_b1_target():
    r = measure(1, "fixed", 4000)
    _report(r)
    assert r["sim_s_per_wall_s"] >= 0.1


@pytest.mark.xfail(strict=False, reason="07 P0-9 / P0-10 target; depends on machine load")
def test_auto_step_b1_target():
    r = measure(1, "auto", 20000, "controller_interface.update_rate_hz=10000.0")
    _report(r)
    assert r["sim_s_per_wall_s"] >= 20.0


@pytest.mark.xfail(strict=False, reason="07 P0-9 target (64 seeds x 1 s in 5 min); depends on machine load")
def test_monte_carlo_64_target():
    r = measure(64, "fixed", 300)
    _report(r)
    assert 1.0 / r["sim_s_per_wall_s"] <= 300.0


def test_report_table():
    """Prints the numbers the engine handoff reports: B = 1 and B = 64 (N = L = 8) under fixed and auto."""
    rows = [measure(1, "fixed", 3000), measure(1, "auto", 3000), measure(64, "fixed", 200), measure(64, "auto", 200)]
    for r in rows:
        _report(r)
        assert r["sim_s_per_wall_s"] > 0
