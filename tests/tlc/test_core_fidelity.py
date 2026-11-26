"""Fidelity gate (spec Phase 2): the vendored core on the original 4-ring plant reproduces the published ECE496
results (``thermal_control/results/metrics.json``) for the steady-state (R1) and common-aggressor (R3, R4)
scenarios, run exactly as ``thermal_control/scenarios.py`` runs them.

The published numbers are rounded to 3 decimals and were produced with a different numpy / scipy build: on this
machine (numpy 2.5.4, scipy 1.18.1) the ORIGINAL core gives steady R1 rms[3] = 0.24948 -> 0.249 where the
published value is 0.25. So the tolerance is 1.5 units in the last published place, and bit-identity is checked
against the original core run here instead (test_core_equivalence.py). Compute-time fields (t_comp_*) are machine
dependent and not compared.
Slow: about 30 s (1.35 s of simulated time on the Python core).

``metrics.json`` was last produced at f98ce84; ae9556b added a monitor noise model (Johnson, shot, per-ring DC
offset) whose defaults change the plant, and did not regenerate it. The published numbers are therefore checked on
the plant with the f98ce84 noise settings (``LEGACY_NOISE``: 4e-4 FS white noise only).
"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pytest

import mrm.control.tlc.core as vendored
from mrm.control.tlc.core import params as vparams

COMPARED = ("p999", "peak", "rms", "heater_mw", "max_slew_pct_per_ms", "cmd_range")
TOL = 1.5e-3
LEGACY_NOISE = dict(pd_noise_fs=4e-4, pd_shot=False, pd_johnson=False, pd_offset_fs=(0.0, 0.0, 0.0, 0.0))


def _published(ece496):
    return json.loads((ece496 / "results" / "metrics.json").read_text())


def _run_vendored(sim, **kw):
    orig = sim.TLC
    sim.TLC = vendored.TLC
    try:
        return sim.run("tlc", pp=vparams.PlantParams(**LEGACY_NOISE), cp=vparams.ControllerParams(), **kw)
    finally:
        sim.TLC = orig


def _assert_close(got: dict, want: dict, where: str):
    for k in COMPARED:
        np.testing.assert_allclose(np.asarray(got[k], dtype=float), np.asarray(want[k], dtype=float),
                                   rtol=0, atol=TOL, err_msg=f"{where}.{k}")


@pytest.mark.slow
def test_steady_state_r1_matches_the_published_result(ece496):
    sim, sc = importlib.import_module("sim"), importlib.import_module("scenarios")
    o = _run_vendored(sim, T=1.0, seed=1)
    _assert_close(sc.metrics(o, 0.1), _published(ece496)["steady"]["tlc"]["R1"], "steady.R1")


@pytest.mark.slow
def test_common_aggressor_r3_r4_match_the_published_result(ece496):
    sim, sc = importlib.import_module("sim"), importlib.import_module("scenarios")
    dist = importlib.import_module("disturbances")
    d2 = dist.D2Aggressor(t0=0.05, n_periods=2)
    o = _run_vendored(sim, T=0.05 + 0.26 + 0.04, dist=dist.Disturbances([d2]), seed=3)
    want = _published(ece496)["d2_common"]["tlc"]
    _assert_close(sc.metrics(o, 0.05), want["R3"], "d2_common.R3")
    assert sc.recovery_ms(o, d2.end_time()) == want["R4_recovery_ms"]
