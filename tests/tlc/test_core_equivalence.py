"""The vendored core (mrm/control/tlc/core) against the original ECE496 core on the original 4-ring plant.

With the source defaults the two must be bit-identical: same plant, same seed, same disturbances, every logged
signal equal. This is what keeps the port faithful while the allowed deviations (spec 2.3) are made. Short runs
(50 ms) of the steady and common-aggressor scenarios; the published metrics are reproduced in
test_core_fidelity.py (slow).
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

import mrm.control.tlc.core as vendored

T_S = 0.05
SIGNALS = ("delta", "delta_hat", "u", "mode", "heater_w", "sigma", "amp", "rate_hat", "eff")


def _run(sim, core_cls, **kw):
    orig = sim.TLC
    sim.TLC = core_cls
    try:
        return sim.run("tlc", **kw)
    finally:
        sim.TLC = orig


def _scenarios(ece496):
    dist = importlib.import_module("disturbances")
    return {
        "steady": dict(T=T_S, seed=1),
        "d2_common": dict(T=T_S, seed=3, dist=dist.Disturbances([dist.D2Aggressor(t0=0.01, n_periods=2)])),
        "cold": dict(T=T_S, seed=7, start="cold"),
    }


@pytest.mark.parametrize("name", ["steady", "d2_common", "cold"])
def test_vendored_core_is_bit_identical_to_the_source(ece496, name):
    sim = importlib.import_module("sim")
    kw = _scenarios(ece496)[name]
    ref = _run(sim, sim.TLC, **kw)
    got = _run(sim, vendored.TLC, **kw)
    assert type(got["ctl"]).__module__.startswith("mrm.control.tlc.core")
    for k in SIGNALS:
        np.testing.assert_array_equal(got[k], ref[k], err_msg=k)
