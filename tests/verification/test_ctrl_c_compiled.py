"""Checks the compiled versions (numba, the flat ABI of the compiled kernel) of the model-based baselines against
their Python versions.

Read this first
---------------
The Python controller runs the closed loop on the ReferencePlant. Every Observation it receives is recorded and
replayed into the compiled controller, which starts from the same applied codes. The commands must agree code for
code, and the compiled estimator must track the Python one to rounding. The compiled step functions also run under
NUMBA_DISABLE_JIT=1: a subprocess replays synthetic observations and compares with the jitted result.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mrm.control import registry
from mrm.control.api import ActionValidator, Observation
from mrm.control.baselines import kalman_lqg, mpc

from test_ctrl_c_harness import build_rig, hold, load_cfg, run_loop

ROOT = Path(__file__).resolve().parents[2]


class Recorder:
    """Wraps a Python controller and keeps a copy of every Observation it sees."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.latency_ticks = inner.latency_ticks
        self.obs = []

    def reset(self, spec):
        self.inner.reset(spec)

    def step(self, obs):
        self.obs.append(Observation(obs.tick, obs.dt_s, {k: np.array(v) for k, v in obs.sensors.items()},
                                    {k: np.array(v) for k, v in obs.new.items()}))
        return self.inner.step(obs)

    def telemetry(self):
        return self.inner.telemetry()


@pytest.fixture(scope="module")
def cfg():
    return load_cfg()


def _replay(cfg, py_ctrl, c_ctrl, n_ticks=1200, B=2):
    rig = build_rig(cfg, B=B)
    hold(rig, 30)
    rec = Recorder(py_ctrl)
    tr = run_loop(rig, rec, n_ticks)
    c_ctrl.reset(rig.spec)
    val = ActionValidator(rig.spec)
    codes_c = []
    for obs in rec.obs:
        codes_c.append(val.validate(c_ctrl.step(obs))["tx.heater"].astype(np.int64))
    return tr, np.array(codes_c), rec


def test_compiled_lqg_matches_python(cfg):
    py = kalman_lqg.KalmanLQG()
    cc = registry.create("kalman_lqg_c")
    assert isinstance(cc, kalman_lqg.CompiledLqg) and cc.latency_ticks == py.latency_ticks
    tr, codes_c, _ = _replay(cfg, py, cc)
    np.testing.assert_array_equal(codes_c, tr.heater_codes)
    # estimator state: the compiled filter equals the Python one to rounding
    d = cc.design
    _, _, s_i, s_f = cc.state
    B, N = s_f.shape[0], d.Phi.shape[0]
    z_c = s_f[:, :N * (d.nz + 2)].reshape(B, N, d.nz + 2)[..., :d.nz]
    np.testing.assert_allclose(z_c, py.z, rtol=1e-7, atol=1e-9 * np.max(np.abs(py.z)))
    tel = cc.telemetry()
    np.testing.assert_allclose(tel["detuning_est_pm"], py.telemetry()["detuning_est_pm"], atol=1e-6)
    assert np.all(s_i[:, 1:] == kalman_lqg.ST_LOCK)


def test_compiled_mpc_matches_python(cfg):
    py = mpc.LinearMPC()
    cc = registry.create("mpc_c")
    tr, codes_c, _ = _replay(cfg, py, cc, n_ticks=1000)
    diff = np.abs(codes_c - tr.heater_codes)
    # same ADMM, per-ring stopping, float64; the summation order differs, so a command may round differently when
    # it sits within rounding of a half code: allow it, but rarely and by one code
    assert diff.max() <= 1 and np.mean(diff > 0) < 0.01, (diff.max(), np.mean(diff > 0))
    tel = cc.telemetry()
    np.testing.assert_allclose(tel["detuning_est_pm"], py.telemetry()["detuning_est_pm"], atol=1e-3)


def test_compiled_controllers_lock_in_closed_loop(cfg):
    """The compiled controllers on their own close the loop (CompiledController Python path)."""
    for name in ("kalman_lqg_c", "mpc_c"):
        rig = build_rig(cfg)
        hold(rig, 30)
        tr = run_loop(rig, registry.create(name), 800)
        err = (tr.detuning_m[200:] - rig.targets_m[None]) * 1e12
        assert np.max(np.abs(err)) < 2.0, (name, np.max(np.abs(err)))


# Run in a subprocess, with and without NUMBA_DISABLE_JIT: 40 synthetic observations through each compiled controller.
_NO_JIT_SCRIPT = r"""
import json, sys, numpy as np, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, "tests/verification")
from test_ctrl_c_harness import load_cfg
from mrm.control.api import ControllerSpec, Observation
from mrm.control.baselines import kalman_lqg, mpc
cfg = load_cfg()
spec = ControllerSpec.from_config(cfg, n_batch=1, nominal=cfg)
out = {}
for name, ctrl in (("lqg", kalman_lqg.CompiledLqg()), ("mpc", mpc.CompiledMpc())):
    ctrl.reset(spec)
    d = ctrl.design if name == "lqg" else ctrl.mpc.lqg
    N, M = spec.n_rings, spec.samples_per_step
    rng = np.random.default_rng(5)
    codes = []
    for k in range(40):
        y = d.y0[None, :] + 300.0 * np.sin(0.3 * k) + rng.normal(0, 3.0, (1, N))
        smp = np.zeros((1, N, M), np.int32); smp[..., :2] = np.rint(y)[..., None].astype(np.int32)
        sens = {"tx.monitor.samples": smp, "tx.monitor.sample_ticks": np.zeros((1, N, M), np.int64),
                "tx.monitor.count": np.full((1, N), 2, np.int32), "tx.monitor.latest": smp[..., 1],
                "monitor_valid": np.ones((1, N), bool), "adc_overrange": np.zeros((1, N), bool),
                "tx.heater.applied": np.rint(d.u0)[None].astype(np.uint16),
                "tx.bias.applied": np.zeros((1, N), np.uint16)}
        sens = {k: v for k, v in sens.items() if k in spec.sensors}
        for s in spec.sensors:
            if s not in sens:
                ch = spec.channels[s]
                sens[s] = np.zeros(spec.shape_of(s), ch.np_dtype)
        new = {s: np.ones((1,) + spec.channels[s].flag_shape(spec.dims), bool) for s in spec.sensors}
        act = ctrl.step(Observation(10 * (k + 1), 1e-6, sens, new))
        codes.append(np.asarray(act["tx.heater"]).tolist())
    out[name] = codes
print(json.dumps(out))
"""


def test_compiled_steps_run_without_jit():
    """NUMBA_DISABLE_JIT=1 gives the same commands as the jitted step functions (synthetic observations)."""
    env_j = dict(os.environ)
    env_j.pop("NUMBA_DISABLE_JIT", None)
    env_n = dict(env_j, NUMBA_DISABLE_JIT="1")
    res = []
    for env in (env_j, env_n):
        p = subprocess.run([sys.executable, "-c", _NO_JIT_SCRIPT], cwd=ROOT, env=env, capture_output=True, text=True,
                           timeout=600)
        assert p.returncode == 0, p.stderr[-2000:]
        res.append(json.loads(p.stdout.strip().splitlines()[-1]))
    assert res[0]["lqg"] == res[1]["lqg"]
    a, b = np.array(res[0]["mpc"]), np.array(res[1]["mpc"])
    assert np.max(np.abs(a - b)) <= 1
