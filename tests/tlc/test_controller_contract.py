"""Contract of the ``tlc`` adapter (docs/TLC_PORT_SPEC_AND_PLAN.md 2.1-2.2) on a G26 ControllerSpec, with hand-built
observations: action shape, dtype and range; the multirate schedule (one core step per 10 edges, every planned
sample emitted exactly once and in order); warm hand-off; telemetry keys and shapes; the telemetry log."""
from __future__ import annotations

import numpy as np
import pytest

import mrm.params as P
from mrm.control import registry
from mrm.control.api import TELEMETRY_KEYS, ControllerSpec, Observation
from mrm.control.tlc import controller as tlc_controller
from mrm.control.tlc.controller import TlcController
from mrm.control.tlc.stub_core import StubCore

B = 2
N_OUT = 10


@pytest.fixture(scope="module")
def spec():
    cfg = P.load("g26")
    return ControllerSpec.from_config(cfg, n_batch=B, nominal=cfg)


def _obs(spec, edge, applied, level_code=30000):
    """The observation of controller edge ``edge``: 2-3 monitor samples per ring, as at 250 kS/s and 100 kHz."""
    n, m = spec.n_rings, spec.samples_per_step
    count = np.full((B, n), 3 if edge % 2 == 0 else 2, dtype=np.int64)
    samples = np.full((B, n, m), level_code, dtype=np.int64)
    sensors = {"tx.monitor.samples": samples, "tx.monitor.count": count,
               "tx.heater.applied": np.asarray(applied, dtype=np.int64),
               "monitor_valid": np.ones((B, n), dtype=bool)}
    return Observation(tick=spec.phase_ticks + edge * spec.period_ticks, dt_s=spec.tick_s, sensors=sensors)


class RecordingCore:
    """A core whose planned samples are unique numbers, so the test can see which sample went out when."""
    calls: list = []

    def __init__(self, spec, *, ts, n_out, params):
        self.n, self.n_out, self.k = spec.n_rings, n_out, 0
        self.last = StubCore(self.n, ts=ts, n_out=n_out).last

    def warm_start(self, u0):
        self.u0 = np.array(u0)

    def first_plan(self, t0):
        return self._seq()

    def cold_start(self):
        raise AssertionError("not used")

    def step(self, pd, t_block_start):
        RecordingCore.calls.append((pd.shape, t_block_start))
        return self._seq()

    def _seq(self):
        seq = (self.k * self.n_out + np.arange(self.n_out))[:, None] * 1e-4 + np.zeros((1, self.n))
        self.k += 1
        return seq


def _run_edges(ctrl, spec, n_edges, applied):
    actions = []
    for e in range(n_edges):
        actions.append(ctrl.step(_obs(spec, e, applied))["tx.heater"])
    return np.stack(actions)


def test_tlc_is_registered():
    assert isinstance(registry.create("tlc", {}), TlcController)


def test_action_is_int64_bn_within_the_dac_range(spec):
    ctrl = TlcController(dither_amp_u=0.0)
    ctrl.reset(spec)
    applied = np.full((B, spec.n_rings), 40000)
    acts = _run_edges(ctrl, spec, 30, applied)
    assert acts.dtype == np.int64
    assert acts.shape == (30, B, spec.n_rings)
    assert acts.min() >= spec.heater_dac.code_min and acts.max() <= spec.heater_dac.code_max


def test_stub_without_dither_holds_the_warm_code(spec):
    ctrl = TlcController(dither_amp_u=0.0)
    ctrl.reset(spec)
    applied = np.arange(B * spec.n_rings).reshape(B, spec.n_rings) * 1000 + 20000
    acts = _run_edges(ctrl, spec, 25, applied)
    assert np.array_equal(acts, np.broadcast_to(applied, acts.shape))


def test_core_runs_every_tenth_edge_and_each_planned_sample_goes_out_once_in_order(spec, monkeypatch):
    monkeypatch.setitem(tlc_controller.CORES, "recording", RecordingCore)
    RecordingCore.calls = []
    ctrl = TlcController(core="recording")
    ctrl.reset(spec)
    applied = np.zeros((B, spec.n_rings), dtype=np.int64)
    acts = _run_edges(ctrl, spec, 100, applied)
    assert len(RecordingCore.calls) == 9 * B                     # edges 10, 20, ..., 90; one core per world
    law = ctrl.heater_law
    want = law.code_from_u(np.arange(100) * 1e-4)                # sample j of the stream is planned value j
                                                                 # (block 0 from first_plan)
    assert np.array_equal(acts[:, 0, 0], want)


def test_core_blocks_hold_the_samples_of_the_previous_ten_edges(spec, monkeypatch):
    monkeypatch.setitem(tlc_controller.CORES, "recording", RecordingCore)
    RecordingCore.calls = []
    ctrl = TlcController(core="recording")
    ctrl.reset(spec)
    _run_edges(ctrl, spec, 21, np.zeros((B, spec.n_rings), dtype=np.int64))
    shapes = [c[0] for c in RecordingCore.calls[::B]]
    # the samples delivered at an edge were taken in the 10 us before it, so the block at edge e holds the samples
    # of edges e-9 .. e (10 edges of 2, 3, 2, 3, ... samples); edge 0's samples predate the start and are dropped
    assert shapes[0] == (25, spec.n_rings)
    assert shapes[1] == (25, spec.n_rings)
    starts = [c[1] for c in RecordingCore.calls[::B]]
    assert starts[0] == pytest.approx(spec.phase_ticks * spec.tick_s)     # the block that began at the first edge
    assert starts[1] - starts[0] == pytest.approx(10 * spec.period_s)


def test_core_period_must_be_whole_edges(spec):
    with pytest.raises(ValueError):
        TlcController(core_every_n_edges=0).reset(spec)


def test_dither_beyond_the_source_tone_plan_is_refused(spec):
    ctrl = TlcController(dither_amp_u=0.01)                       # 8 rings, 4 source tones
    ctrl.reset(spec)
    with pytest.raises(ValueError, match="tone"):
        ctrl.step(_obs(spec, 0, np.zeros((B, spec.n_rings), dtype=np.int64)))


def test_telemetry_has_the_standard_keys_as_numeric_bn_arrays(spec):
    ctrl = TlcController()
    ctrl.reset(spec)
    for when in ("before", "after"):
        if when == "after":
            _run_edges(ctrl, spec, 11, np.full((B, spec.n_rings), 30000))
        tel = ctrl.telemetry()
        assert set(TELEMETRY_KEYS) <= set(tel)
        for k, v in tel.items():
            v = np.asarray(v)
            assert v.shape == (B, spec.n_rings), k
            assert v.dtype.kind in "biuf", k


def test_telemetry_log_has_one_row_per_core_step(spec):
    ctrl = TlcController()
    ctrl.reset(spec)
    _run_edges(ctrl, spec, 35, np.full((B, spec.n_rings), 30000))
    arrays = ctrl.telemetry_log.to_arrays()
    assert arrays["t_s"].shape == (4,)                           # core steps at edges 0, 10, 20, 30
    assert arrays["state"].shape == (4, B, spec.n_rings)


def test_reset_starts_a_fresh_run(spec):
    ctrl = TlcController()
    ctrl.reset(spec)
    _run_edges(ctrl, spec, 15, np.full((B, spec.n_rings), 30000))
    ctrl.reset(spec)
    assert ctrl.telemetry_log.to_arrays()["t_s"].shape == (0,)
    acts = _run_edges(ctrl, spec, 3, np.full((B, spec.n_rings), 12345))
    assert np.all(acts == 12345)


def test_tlc_pi_is_registered_with_the_source_baseline_core():
    ctrl = registry.create("tlc_pi", {})
    assert ctrl.params.core == "faithful_pi"


def test_invalid_monitor_samples_reach_the_core_as_missing(spec, monkeypatch):
    monkeypatch.setitem(tlc_controller.CORES, "recording", RecordingCore)
    seen = []
    orig = RecordingCore.step
    monkeypatch.setattr(RecordingCore, "step", lambda self, pd, t: (seen.append(pd.copy()), orig(self, pd, t))[1])
    ctrl = TlcController(core="recording")
    ctrl.reset(spec)
    applied = np.zeros((B, spec.n_rings), dtype=np.int64)
    for e in range(11):
        obs = _obs(spec, e, applied)
        valid = np.ones((B, spec.n_rings), dtype=bool)
        valid[:, 3] = e != 5                                     # ring 3 invalid at edge 5
        sensors = {k: obs.sensors[k] for k in obs.sensors}
        sensors["monitor_valid"] = valid
        ctrl.step(Observation(tick=obs.tick, dt_s=obs.dt_s, sensors=sensors))
    pd = seen[0]                                                 # world 0, block of edges 1..10
    assert np.isnan(pd[:, 3]).sum() == 2                         # edge 5 delivered 2 samples
    assert not np.isnan(np.delete(pd, 3, axis=1)).any()


@pytest.fixture(scope="module")
def nominal_spec():
    cfg = P.load("g26", overrides=["controller_interface.knowledge=[line_grid,ring_to_line,model_nominal]"])
    return ControllerSpec.from_config(cfg, n_batch=B, nominal=cfg)


@pytest.mark.parametrize("core", ["faithful", "faithful_pi"])
def test_faithful_cores_run_in_the_adapter(nominal_spec, core):
    ctrl = TlcController(core=core)
    ctrl.reset(nominal_spec)
    acts = _run_edges(ctrl, nominal_spec, 31, np.full((B, nominal_spec.n_rings), 34500))
    assert acts.dtype == np.int64 and acts.min() >= 0 and acts.max() <= 65535
    assert len(ctrl.telemetry_log) == 4
    assert np.all(np.isfinite(ctrl.telemetry()["detuning_est_pm"]))


def test_faithful_core_needs_model_nominal(spec):
    ctrl = TlcController(core="faithful")
    ctrl.reset(spec)
    with pytest.raises(Exception, match="model_nominal"):
        ctrl.step(_obs(spec, 0, np.zeros((B, spec.n_rings), dtype=np.int64)))


def test_forced_fault_switches_the_faithful_core_to_its_fallback(nominal_spec):
    ctrl = TlcController(core="faithful", force_fault_at_s=0.0002)
    ctrl.reset(nominal_spec)
    _run_edges(ctrl, nominal_spec, 41, np.full((B, nominal_spec.n_rings), 34500))
    assert all(c.ctl.fault for c in ctrl.cores)
    assert np.all(ctrl.telemetry()["fault"])
