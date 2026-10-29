"""Checks the controller templates in examples/controllers/ on the controller API (toy ring, no plant).

Each template must import, reset, step, write valid codes and do what its docstring says.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

import test_ctrl_a_harness as harness
from mrm.control.api import ActionValidator, Controller

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "controllers"


def load(name):
    """Import an example by path once (dataclasses need the module in sys.modules; the registry refuses a second
    class under the same name)."""
    import sys
    key = f"ctrl_a_example_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, EXAMPLES_DIR / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
    return sys.modules[key]


def test_python_template_locks_the_toy():
    mod = load("template_python_controller")
    spec = harness.toy_spec(1, 2)
    toy = harness.ToyRings(spec, u0=[30000, 31000], g=0.5, M=58000.0, u_init=[30700, 31700])
    c = mod.MyIntegralLock(peak_codes=58000.0, k_i_codes=150.0)
    assert isinstance(c, Controller)
    c.reset(spec)
    toy.run(c, 1500)
    np.testing.assert_allclose(toy.x, np.sqrt(1 / 0.7 - 1), atol=0.03)


def test_compiled_template_matches_its_python_twin():
    mod = load("template_compiled_controller")
    spec = harness.toy_spec(1, 2)
    runs = []
    for c in (mod.make_my_compiled_lock(k_i_codes=150.0), load("template_python_controller").MyIntegralLock(
            k_i_codes=150.0)):
        toy = harness.ToyRings(spec, u0=[30000, 31000], g=0.5, M=58000.0, u_init=[30700, 31700], noise=0.0, m=1)
        c.reset(spec)
        h, _ = toy.run(c, 800)
        runs.append((h, toy.x.copy()))
    np.testing.assert_allclose(runs[0][1], np.sqrt(1 / 0.7 - 1), atol=0.03)
    # the compiled template uses the held latest code, the Python one the block mean: same with one noiseless sample
    assert np.array_equal(runs[0][0], runs[1][0])


def test_per_ring_template_finds_the_peak():
    mod = load("template_per_ring")
    spec = harness.toy_spec(1, 2)
    toy = harness.ToyRings(spec, u0=[30000, 31000], g=0.3, u_init=[30200, 31200], noise=1.0)
    c = mod.make_po_controller(delta_codes=16)
    c.reset(spec)
    toy.run(c, 2500, validator=ActionValidator(spec))
    assert np.all(np.abs(toy.x) < 0.1), toy.x
