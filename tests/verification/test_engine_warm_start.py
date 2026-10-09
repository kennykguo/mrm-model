"""Integration features of the engine stage (integrator, 2026-10-09): the settled warm start (Plant.settle,
``run(..., settle=True)``), the quasi-static warm-start codes (``heater_code='quasi_static'``), the factory-table hand-off
(``run(..., factory_table=...)``) and the baseline registry.

Requests resolved: controllers_a handoff 2.3 and controllers_c handoff 4.3b (a warm start from the thermal steady
state), engine handoff 6 (a)/(b), controllers_b handoff 2 (load_factory_cal before reset), and the baselines package
imports (controllers_a/b/c). Physics: 07 P0-1 item 6 (defined start states).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mrm import params
from mrm.control import registry
from mrm.control.api import ControllerSpec
from mrm.plant.plant import Plant
from mrm.sim import kernel as K
from mrm.sim import runner
from mrm.sim.metrics import RecordSpec

BASELINES = ("pi_lock", "pi_lock_c", "dither_lockin", "dither_lockin_c", "bias_dither", "es_po", "es_sine",
             "feedforward_pi", "midranging", "ramp_acquire", "acquire_then_lock", "bitstat_lock", "wdm_supervisor",
             "kalman_lqg", "kalman_lqg_c", "mpc", "mpc_c")


@pytest.fixture(scope="module")
def cfg():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load("g26")


def _plant(cfg, record=None, ticks=0):
    c, pp, _ = runner.load_inputs(cfg, uncertainty="nominal", variation="off")
    sc = runner._make_scenario(None, c)
    spec = ControllerSpec.from_config(c, n_batch=1, nominal=c, scenario_drives=sc.drives, scenario_offsets=sc.offsets)
    return Plant(pp, sc, spec, record=record, record_ticks=ticks)


def test_registry_finds_every_baseline_without_an_import():
    """The registry imports mrm.control.baselines once; every baseline module registers itself there."""
    names = registry.available()
    missing = [n for n in BASELINES if n not in names]
    assert not missing, missing
    assert "composite" in names


def test_settled_warm_start_is_a_steady_lock(cfg):
    """quasi-static oracle codes + settle: the plant starts in its own steady state (every ring on the hot branch,
    red side of its line) and stays there open loop; with cold lags the same codes drop every ring off the line
    (measured -800 to -1250 pm after 5 ms), which is why controllers needed start_delay_s before."""
    rec = [RecordSpec("detuning_m", 100, "last")]
    out = {}
    for settle in (True, False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = runner.run(cfg, None, None, duration_s=3e-3, uncertainty="nominal", variation="off",
                           heater_code="quasi_static", reset_protocol="branch_high", settle=settle, record=rec)
        out[settle] = r.trace["detuning_m@100:last"]["value"][:, 0, :] * 1e12
        if settle:
            info = r.manifest_inputs["warm_start"]["settle"]
            assert info["converged"] and info["iterations"] < 100, info
            assert int(np.sum(info["selfheat_not_converged_last"])) == 0, info
    d = out[True]
    assert np.all((d[0] > 50.0) & (d[0] < 200.0)), d[0]               # red side of the line, near the oracle point
    assert np.max(np.abs(d - d[0])) < 0.3, np.max(np.abs(d - d[0]))   # steady (measured 0.03 pm)
    assert np.all(out[False][-1] < -300.0), out[False][-1]            # cold lags: off the hot branch


def test_settle_equals_the_long_open_loop_limit(cfg):
    """At code 0 (cold branch, no bistability in play) the settled state equals the plant's own state after 100 ms of
    open loop from cold lags (11 x the slowest pole of g26, 9 ms): ring, heater and PD temperatures within 5 mK
    (after 30 ms the 9 ms lags still miss ~4 %: 7-21 mK)."""
    a = _plant(cfg)
    a.reset(protocol="branch_low", settle=True)
    assert a.settle_info["converged"], a.settle_info
    t_settled = a.S.rf[:, [K.RF_T_RING, K.RF_T_HEATER, K.RF_T_PD], :].copy()
    b = _plant(cfg)
    b.reset(protocol="branch_low")
    b.run(100_000, mode=K.MODE_OPEN)
    t_long = b.S.rf[:, [K.RF_T_RING, K.RF_T_HEATER, K.RF_T_PD], :]
    assert np.max(np.abs(t_settled - t_long)) < 5e-3, np.max(np.abs(t_settled - t_long))
    assert np.max(t_settled[:, 0] - a.S.wf[:, K.WF_T_BASE0, None]) > 0.5        # data heat and absorption warm it


def test_settle_needs_the_lasers_on(cfg):
    p = _plant(cfg)
    with pytest.raises(ValueError, match="laser_ramp"):
        p.reset(protocol="laser_ramp", settle=True)
    with pytest.raises(ValueError, match="heater_code mode"):
        runner.warm_start_codes(p, cfg, mode="nope")


class _HoldWithTable:
    """Minimal Python controller with a factory-table hook: holds the heater registers."""
    name = "hold_with_table"
    latency_ticks = 0

    def __init__(self):
        self.calls = []

    def load_factory_cal(self, table):
        self.calls.append(("load", table))

    def reset(self, spec):
        self.calls.append(("reset", None))

    def step(self, obs):
        return {"tx.heater": np.asarray(obs.sensors["tx.heater.applied"], np.int64)}


def test_factory_table_is_loaded_before_the_reset(cfg):
    ctrl = _HoldWithTable()
    table = object()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = runner.run(cfg, ctrl, None, duration_s=50e-6, uncertainty="nominal", variation="off",
                       factory_table=table)
    assert ctrl.calls[:2] == [("load", table), ("reset", None)]
    assert r.manifest_inputs["warm_start"]["factory_table"] is True
    with pytest.raises(TypeError, match="load_factory_cal"):
        runner.run(cfg, registry.create("pi_lock"), None, duration_s=20e-6, uncertainty="nominal",
                   variation="off", factory_table=table)


def test_quasi_static_codes_fall_back_to_code_min_in_sampled_worlds(cfg):
    """Sampled worlds: rings whose oracle point the heater cannot reach have no quasi-static code; they start at
    code_min with a warning (measured for g26, seeds 7, B = 8: 13 of 64 rings), and the settle still converges."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = runner.run(cfg, None, None, duration_s=50e-6, batch=8, seeds=7, heater_code="quasi_static", settle=True)
    msg = [str(x.message) for x in w if "warm_start_codes" in str(x.message)]
    assert len(msg) == 1, msg
    ws = r.manifest_inputs["warm_start"]
    assert ws["settle"]["converged"], ws["settle"]
    codes = np.asarray(ws["heater_code"])
    assert codes.shape == (8, 8) and np.any(codes == 0) and np.any(codes > 20000)
