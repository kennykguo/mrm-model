"""Checks that the compiled group-A controllers give the same codes as plain Python.

Rule: the per-tick controller code must also run, with identical results, under NUMBA_DISABLE_JIT=1.

- fast: the pure-Python bodies (``.py_func``) of pi_step, dither_step, bias_step and bias_fast, driven through the
  same CompiledController plumbing on the toy ring, give exactly the jitted codes;
- slow: the toy and unit tests of the group-A controller files run in a subprocess with NUMBA_DISABLE_JIT=1 (every
  helper runs in Python too).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import test_ctrl_a_harness as harness
from mrm.control.api import CompiledController
from mrm.control.baselines import bias_dither, dither_lockin, pi_lock

HERE = Path(__file__).resolve().parent


def _pyfunc(fn):
    return getattr(fn, "py_func", fn)


def _python_twin(ctrl: CompiledController) -> CompiledController:
    """The same controller with the Python bodies of its numba step / fast kernel (init stays compiled: it only lays
    out arrays)."""
    ctrl.step_fn = _pyfunc(ctrl.step_fn)
    if ctrl.fast_fn is not None:
        ctrl.fast_fn = _pyfunc(ctrl.fast_fn)
    return ctrl


def test_pi_and_dither_python_bodies_match_the_jitted_steps():
    spec = harness.toy_spec(2, 3)
    u0 = np.array([[30000, 30500, 31000], [29800, 30200, 30900]], float)
    g = np.array([[4.0, 6.0, 0.8], [3.0, 8.0, 1.5]])

    def disturb(toy, k):
        if 300 <= k < 330:
            toy.u0 -= 20.0
        if 500 <= k < 510:
            toy.u0 += 250.0                     # past the fold: falls and recaptures
        toy.valid[:] = not (200 <= k < 203)
    def make_pi():
        return pi_lock.PiLockCompiled(init_codes=[31400, 31900, 31600], k2=0.002)

    def make_dither():
        return dither_lockin.DitherLockinCompiled(init_codes=[30500, 31000, 31500], dither_codes=40.0, phase_steps=4,
                                                  blank_steps=1)
    for make, init, n in ((make_pi, [31400, 31900, 31600], 900), (make_dither, [30500, 31000, 31500], 900)):
        runs = []
        for twin in (False, True):
            c = make()
            if twin:
                _python_twin(c)
            toy = harness.ToyRings(spec, u0=u0, g=g, u_init=np.array(init), x0=3.0)
            c.reset(spec)
            runs.append(toy.run(c, n, on_step=disturb)[0])
        assert np.array_equal(runs[0], runs[1])


def test_bias_dither_python_bodies_match_the_jitted_kernels():
    import test_ctrl_a_bias_dither as bias_dither_tests
    spec = harness.toy_spec(1, 2, bias=True)
    u0 = np.array([[30000.0, 31000.0]])
    init = u0[0] + 360.0
    xs = []
    for twin in (False, True):
        toy = harness.ToyRings(spec, u0=u0, g=0.5, W=450.0, noise=1.0, bias_gain=0.5, u_init=init, bias_init=1191)
        c = bias_dither.BiasDither(dither_codes=120.0, samples_per_phase=2, hold_cycles=8, id_cycles=8, id_codes=30.0,
                                   gain=0.5, init_codes=init)
        if twin:
            _python_twin(c)
        c.reset(spec)
        saved = bias_dither_tests.bias_fast
        bias_dither_tests.bias_fast = c.fast_fn                    # the toy loop calls the module-level kernel
        try:
            xs.append(bias_dither_tests.toy_fast_loop(c, spec, toy, 400))
        finally:
            bias_dither_tests.bias_fast = saved
    assert np.array_equal(xs[0], xs[1])


@pytest.mark.slow
def test_toy_and_unit_tests_under_disable_jit():
    """~40 s with the JIT off on this machine (35 tests + 1 strict xfail); the plant-based tests are not run this way
    (the ReferencePlant kernels would run in Python: minutes per test)."""
    files = ["test_ctrl_a_pi.py", "test_ctrl_a_dither.py", "test_ctrl_a_bias_dither.py", "test_ctrl_a_es.py",
             "test_ctrl_a_midrange.py", "test_ctrl_a_feedforward.py", "test_ctrl_a_examples.py"]
    sel = ("toy or wind or capture_start_with or capture_without or whole or clipped or fast_kernel or early_fall or "
           "identification or per_ring or registry or checks or template or knowledge or truth or fast_red")
    env = dict(os.environ, NUMBA_DISABLE_JIT="1")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "not slow", "-k", sel,
                        *[str(HERE / f) for f in files]], env=env, capture_output=True, text=True, timeout=1800,
                       cwd=str(HERE.parents[1]))
    tail = r.stdout[-3000:]
    assert r.returncode == 0, tail
    assert re.search(r"\b\d+ passed", tail) and not re.search(r"\b\d+ failed", tail), tail
