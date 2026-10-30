"""Regression tests for the fixes of the controllers review (docs/CHANGELOG.md, "2026-10-16: controllers review
fixes"). Each test failed on the code before its fix.

What the tests check:

- ``controller_interface.expose`` reaches the ControllerSpec built from a config (runner, bridge);
- ``spec.nominal`` is the knowledge-filtered view of the unsampled design, never the experiment section. So a
  controller knows only what its knowledge list grants, and the spec fingerprint does not depend on the experiment's
  seeds or world choices;
- the monitor ADC's code-to-ampere scale reaches the channel specs only with monitor knowledge;
- ``controller_interface.schedule: round_robin`` updates one ring per controller edge on both engine paths;
- ``controller_interface.on_out_of_range: error`` raises on the compiled path too;
- after a scenario controller reset, the Python path re-binds a compiled controller's new state to the kernel (its
  fast_kernel and its step share one state again);
- the numbers that the feed-forward, oracle and extremum-seeking docstrings quote hold in the model;
- the scorecard does not count a drifting operating point as locked.

Run time about 1 min (the runs are B = 1, 0.05-0.5 ms).
"""
from __future__ import annotations

import inspect
import re
import warnings

import numba as nb
import numpy as np
import pytest

from mrm import params
from mrm.control import api
from mrm.control.api import (ActionRangeError, BaseController, CompiledController, ControllerSpec, PerRing,
                             UndeclaredChannelError)
from mrm.core.channels import OBS_ACTIVE_RING, ActId, ChannelError, ObsId
from mrm.sim import runner
from mrm.sim.metrics import RecordSpec

_HEATER = int(ActId.TX_HEATER)
_N = 8                                      # g26 rings
_RUN = dict(uncertainty="nominal", variation="off", seeds=0)


def _cfg(*overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load("g26", overrides=list(overrides))


def _run(cfg, ctrl, duration_s, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return runner.run(cfg, ctrl, None, duration_s=duration_s, **_RUN, **kw)


# ------------------------------------------------------------------------------------ expose limits the sensor channels
class _Probe(BaseController):
    name = "probe"

    def __init__(self):
        super().__init__()
        self.seen = []
        self.hidden_raises = None

    def step(self, obs):
        self.seen.append(tuple(sorted(obs.sensors)))
        try:
            obs.sensors["tx.heater.applied"]
            self.hidden_raises = False
        except UndeclaredChannelError:
            self.hidden_raises = True
        return {}


def test_config_expose_limits_the_sensor_channels():
    cfg = _cfg("controller_interface.expose=[tx.monitor.samples,temp.die0]")
    spec = ControllerSpec.from_config(cfg, n_batch=1, nominal=cfg)
    want = ("temp.die0", "tx.monitor.count", "tx.monitor.samples")      # a block exposes its count
    assert tuple(sorted(spec.sensors)) == want
    si, _ = api.spec_arrays(spec)
    assert api.obs_offset(si, int(ObsId.TX_HEATER_APPLIED)) == -1
    assert api.obs_offset(si, int(ObsId.TX_MONITOR_SAMPLES)) >= 0
    # the runner's controller sees exactly those channels
    probe = _Probe()
    _run(cfg, probe, 50e-6)
    assert probe.seen and set(probe.seen) == {want} and probe.hidden_raises
    # the default (no expose) still exposes every sensor channel
    full = ControllerSpec.from_config(_cfg(), n_batch=1, nominal=None)
    assert "tx.heater.applied" in full.sensors and len(full.sensors) > len(want)
    with pytest.raises(ChannelError, match="no.such.channel"):
        c = _cfg("controller_interface.expose=[no.such.channel]")
        ControllerSpec.from_config(c, n_batch=1, nominal=c)


# ----------------------------------------------------------- the nominal design a controller sees is knowledge-filtered
def test_spec_nominal_is_the_knowledge_filtered_design():
    from mrm.control.models import KnowledgeError
    cfg = _cfg()                                                     # g26: line_grid, ring_to_line
    spec = ControllerSpec.from_config(cfg, n_batch=1, nominal=cfg)
    nom = spec.nominal
    assert nom.get("channel_plan.spacing_ghz") == cfg.get("channel_plan.spacing_ghz")         # line_grid
    assert nom.get("channel_plan.ring_to_line") == cfg.get("channel_plan.ring_to_line")       # ring_to_line
    for key in ("ring.optics.q_loaded", "ring.thermal.dlambda_dt_pm_per_k", "ring.monitor.adc_full_scale_ua",
                "experiment.seeds.uncertainty", "experiment.uncertainty", "scenario.ambient_c"):
        with pytest.raises(KnowledgeError):
            nom.si(key)
        with pytest.raises(KnowledgeError):
            nom.get(key)
    with pytest.raises(KnowledgeError):
        nom.model_config()
    # a granted token releases its keys
    c2 = _cfg("controller_interface.knowledge=[line_grid,ring_to_line,linewidth]")
    assert ControllerSpec.from_config(c2, n_batch=1, nominal=c2).nominal.si("ring.optics.q_loaded") == 3850.0
    # model_nominal: the whole design (for linearize), but never the experiment section (seeds, world choices)
    c3 = _cfg("controller_interface.knowledge=[line_grid,ring_to_line,model_nominal]",
              "experiment.seeds.uncertainty=1000")
    n3 = ControllerSpec.from_config(c3, n_batch=1, nominal=c3).nominal
    assert n3.si("ring.thermal.dlambda_dt_pm_per_k") == c3.si("ring.thermal.dlambda_dt_pm_per_k")
    assert n3.si("scenario.ambient_c") == c3.si("scenario.ambient_c")
    with pytest.raises(KnowledgeError):
        n3.get("experiment.seeds.uncertainty")
    mc = n3.model_config()
    assert not any(k.startswith("experiment.") for k in (*mc.leaves, *mc.settings, *mc.open_values))
    assert mc.get("ring.optics.q_loaded") == c3.get("ring.optics.q_loaded")


def test_spec_fingerprint_does_not_depend_on_the_experiment_section():
    """The spec is identical across seeds and uncertainty worlds, also when they are config leaves."""
    for know in ((), ("controller_interface.knowledge=[line_grid,ring_to_line,model_nominal]",)):
        fps = set()
        for extra in (("experiment.seeds.uncertainty=1000",), ("experiment.seeds.uncertainty=2000",),
                      ("experiment.uncertainty=lhs", "experiment.variation=off", "experiment.seeds.device=7")):
            c = _cfg(*know, *extra)
            fps.add(ControllerSpec.from_config(c, n_batch=1, nominal=c).fingerprint())
        assert len(fps) == 1, (know, fps)


def test_model_based_design_unchanged_by_the_filtered_view():
    from mrm.control import models
    c = _cfg("controller_interface.knowledge=[line_grid,ring_to_line,model_nominal]")
    spec = ControllerSpec.from_config(c, n_batch=1, nominal=c)
    a = models.linearize_bank(spec.nominal, None, spec.period_s)
    b = models.linearize_bank(c, None, spec.period_s)
    for ma, mb in zip(a, b):
        for f in ("A", "B", "C", "D"):
            np.testing.assert_array_equal(getattr(ma, f), getattr(mb, f))
        assert ma.delay_ticks == mb.delay_ticks
    with pytest.raises(models.KnowledgeError):
        g = _cfg()
        models.linearize_bank(ControllerSpec.from_config(g, n_batch=1, nominal=g).nominal, None, spec.period_s)


# ------------------------------------------------------------------------- monitor ampere scale needs monitor knowledge
def test_monitor_ampere_scale_needs_monitor_knowledge():
    for know, scaled in (("line_grid,ring_to_line", False), ("line_grid,ring_to_line,monitor_nominal", True),
                         ("line_grid,ring_to_line,model_nominal", True)):
        c = _cfg(f"controller_interface.knowledge=[{know}]")
        spec = ControllerSpec.from_config(c, n_batch=1, nominal=c)
        for ch in ("tx.monitor.samples", "tx.monitor.latest"):
            cs = spec.channels[ch]
            if scaled:
                assert cs.phys_unit == "A" and cs.scale == pytest.approx(40e-6 / 65535), (know, ch)
            else:
                assert cs.scale is None and cs.phys_unit is None, (know, ch)


# ------------------------------------------------------------------------------------------------- round-robin schedule
class _RingCounter:
    def __init__(self, ctx, counts):
        self.key = (ctx.b, ctx.n)
        self.counts = counts
        counts[self.key] = 0

    def step(self, robs):
        self.counts[self.key] += 1
        return {}


# The toy compiled controllers below keep p = [n_rings, offset of tx.heater in the action vector].
@nb.njit
def _rr_init(si, sf):
    p = np.zeros(2, np.int64)
    p[0] = si[api.SPEC_I_N_RINGS]
    p[1] = api.act_offset(si, _HEATER)
    return p, np.zeros(0), np.zeros(1 + p[0], np.int64), np.zeros(0)


@nb.njit
def _rr_step(p, pf, s, sf, obs, act):
    """Counts the edges per active ring (s[1 + ring]; s[0] when every ring is active) and adds 1 code to EVERY
    ring's heater command: the engine must apply only the active ring's."""
    ar = obs[OBS_ACTIVE_RING]
    if ar >= 0:
        s[1 + ar] += 1
    else:
        s[0] += 1
    for i in range(p[0]):
        act[p[1] + i] = act[p[1] + i] + 1


def test_round_robin_steps_one_ring_per_edge_python_path():
    counts = {}
    ctrl = PerRing(lambda ctx: _RingCounter(ctx, counts))
    _run(_cfg("controller_interface.schedule=round_robin"), ctrl, 400e-6)      # 40 edges, 8 rings
    assert sorted(counts) == [(0, n) for n in range(_N)]
    assert set(counts.values()) == {5}, counts
    counts.clear()
    _run(_cfg(), PerRing(lambda ctx: _RingCounter(ctx, counts)), 400e-6)       # simultaneous: every ring, every edge
    assert set(counts.values()) == {40}, counts


@pytest.mark.parametrize("schedule,per_ring", [("round_robin", 5), ("simultaneous", 40)])
def test_round_robin_applies_only_the_active_rings_command(schedule, per_ring):
    """Compiled and Python paths: obs carries the active ring (edge j: ring j % N), only that ring's (B, N) command
    is applied (the others hold), and both paths give the same trace."""
    cfg = _cfg(f"controller_interface.schedule={schedule}")
    out = {}
    for path in ("compiled", "python"):
        ctrl = CompiledController("rr", _rr_init, _rr_step)
        r = _run(cfg, ctrl, 400e-6, path=path, heater_code="locked_steady_state", settle=True,
                 record=[RecordSpec("heater_code", 1, "last")])
        start = np.asarray(r.manifest_inputs["warm_start"]["heater_code"])[0]
        off = int(ctrl.spec.layout.act_field("tx.heater").offset)
        held = ctrl.act[0, off:off + _N]
        s = ctrl.state[2][0]
        if schedule == "round_robin":
            assert s[0] == 0 and s[1:].tolist() == [per_ring] * _N, s
        else:
            assert s[0] == 40 and not s[1:].any(), s
        assert (held - start).tolist() == [per_ring] * _N, (held, start)
        out[path] = np.asarray(r.trace["heater_code@1:last"]["value"])
    assert np.array_equal(out["compiled"], out["python"])


def test_composite_refuses_an_interface_round_robin():
    """A composite time-multiplexes its children with its own ``schedule``; the interface schedule would multiplex
    its output a second time."""
    cfg = _cfg("controller_interface.schedule=round_robin")
    spec = ControllerSpec.from_config(cfg, n_batch=1, nominal=cfg)
    comp = api.CompositeController([(_Probe(), 100e3)])
    with pytest.raises(ValueError, match="schedule"):
        comp.reset(spec)


# ------------------------------------------------------------------------------ out-of-range error on the compiled path
@nb.njit
def _big_step(p, pf, s, sf, obs, act):
    for i in range(p[0]):
        act[p[1] + i] = 70000 + i


def test_on_out_of_range_error_on_the_compiled_path():
    cfg = _cfg("controller_interface.on_out_of_range=error")
    for path in ("compiled", "python"):
        with pytest.raises(ActionRangeError, match=r"tx\.heater\[0, 0\] = 70000 outside \[0, 65535\]"):
            _run(cfg, CompiledController("big", _rr_init, _big_step), 50e-6, path=path)
    with pytest.raises(ActionRangeError, match="tick 0"):
        _run(cfg, CompiledController("big", _rr_init, _big_step), 50e-6, path="compiled")
    # clip (the default): saturate and count, no error
    r = _run(_cfg(), CompiledController("big", _rr_init, _big_step), 50e-6, path="compiled")
    assert int(r.diagnostics["commands_clipped"][0]) == 5 * _N


# ---------------------------------------------------------------------------- a reset re-binds a fast-kernel controller
@nb.njit
def _cnt_init(si, sf):
    p = np.zeros(2, np.int64)
    p[0] = si[api.SPEC_I_N_RINGS]
    p[1] = api.act_offset(si, _HEATER)
    return p, np.zeros(0), np.zeros(2, np.int64), np.zeros(0)


@nb.njit
def _cnt_fast(p, pf, s, sf, fobs, fact):
    s[0] += 1                                   # one count per released monitor sample block


@nb.njit
def _cnt_step(p, pf, s, sf, obs, act):
    if s[1] == 0:
        s[1] = act[p[1]]                        # the held command of the first step after (re)start
    for i in range(p[0]):
        act[p[1] + i] = s[1] + s[0] % 50


def test_python_path_rebinds_a_fast_kernel_controller_after_a_reset():
    from mrm.scenarios import generators
    from mrm.scenarios import registry as scenario_registry
    cfg = _cfg()
    t = 200.5e-6
    reset_pulse = generators.pwl("fault.controller_reset", [0.0, t, t, t + 3e-6, t + 3e-6, 1.0], [0, 0, 1, 1, 0, 0],
                                 absolute=True)                      # a 3 us reset pulse at t = 200.5 us
    reset = generators.Scenario(name="controller_reset", components=(reset_pulse,))
    sc = scenario_registry.from_config(cfg) + reset
    out = {}
    for path in ("compiled", "python"):
        ctrl = CompiledController("cnt", _cnt_init, _cnt_step, fast_fn=_cnt_fast)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = runner.run(cfg, ctrl, sc, duration_s=0.5e-3, path=path, heater_code="locked_steady_state", settle=True,
                           record=[RecordSpec("heater_code", 1, "last")], **_RUN)
        out[path] = np.asarray(r.trace["heater_code@1:last"]["value"])
        # the controller's state is the kernel's (its fast kernel and its step update one array)
        assert ctrl.state[2] is r.plant.S.ctrl_s_i64, path
    assert np.array_equal(out["compiled"], out["python"])


# -------------------------------------------------------------------------- docstring numbers checked against the model


def test_oracle_docstring_optimum():
    """The oracle docstring: the g26-nominal optimum it quotes, and its claim about the unrestricted optimum."""
    from mrm.validation import oracle
    doc = inspect.getcomments(oracle)
    m = re.search(r"in g26 nominal the optimum is \+(\d+) to \+(\d+) pm", doc, re.IGNORECASE)
    assert m, "the docstring must quote the g26-nominal optimum range"
    lo, hi = float(m.group(1)), float(m.group(2))
    c = _cfg("controller_interface.knowledge=[line_grid,ring_to_line,model_nominal]")
    pp = params.realize(c, 1, uncertainty="nominal", variation="off")
    t = oracle.oracle_target(pp, None, cfg=c)
    d = np.asarray(t.detuning_m) * 1e12
    assert lo - 0.5 <= d.min() and d.max() <= hi + 0.5, d
    np.testing.assert_allclose(t.detuning_unrestricted_m, t.detuning_m)          # the restriction costs nothing
    assert "costs no OMA" in doc and "does not carry the lag states" not in doc


def test_feedforward_docstring_die_sensor_sees_the_aggressor():
    """The feed-forward docstring: the die sensor sees the A1 aggressor (open loop, heaters held at the warm-start
    codes). The docstring must not say that it does not, and the size it quotes must hold."""
    from mrm.control.baselines import feedforward
    from mrm.scenarios import library
    doc = inspect.getcomments(feedforward)
    assert "sees neither a local aggressor" not in doc
    assert re.search(r"die sensor sees a local aggressor", doc)
    c = _cfg()
    t_die = {}
    for name, sc in (("a1", library.a1_slide10_aggressor()), ("static", library.static())):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = runner.run(c, None, sc, duration_s=2e-3, heater_code="locked_steady_state", settle=True, **_RUN)
        tr = r.truth_final
        i = [k for k, nm in enumerate(tr.obs_names) if "die0" in str(nm)]
        assert i, tr.obs_names
        t_die[name] = float(tr.t_obs[0, i[0]])
    assert t_die["a1"] - t_die["static"] > 15.0, t_die                # A1 hot cusp at t = 0: ~ +22 K at 2 ms


def test_extremum_seeking_docstring_matches_the_engine():
    from mrm.control.baselines import extremum_seeking
    doc = inspect.getcomments(extremum_seeking)
    assert "cycle between approach and fall" not in doc
    assert "never falls" in doc and "parks at the drop peak" in doc


# ------------------------------------------------------------------------------ scorecard lock needs a stationary point
def test_scorecard_does_not_count_a_drifting_point_as_locked():
    """The scorecard's steady lock needs a stationary operating point: a ring that drifts at 0.8 pm/ms (es_sine on
    g26) stays within 5 pm of its last-5-ms median for the last ~6 ms of a 40 ms run, which passed as a lock."""
    import importlib.util
    import types
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "validation" / "test_controllers_on_engine.py"
    spec_ = importlib.util.spec_from_file_location("_scorecard_mod", path)
    scorecard = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(scorecard)
    t = np.arange(1, 4001) * 10e-6                                  # 40 ms at the 10 us record decimation
    drifting = 90.0 - 0.8 * t * 1e3                                 # pm, ring 0
    steady = np.full_like(t, 120.0)                                 # pm, ring 1
    d = np.stack([drifting, steady], axis=1)[:, None, :] * 1e-12    # (S, B, N) m

    def rec(v):
        return {"t_s": t, "value": v}
    ones = np.ones_like(d)
    res = types.SimpleNamespace(trace={
        "detuning_m@10:mean": rec(d), "detuning_m@10:min": rec(d), "detuning_m@10:max": rec(d),
        "reloma@10:mean": rec(ones), "oracle_reloma@10:mean": rec(ones), "line_on@10:last": rec(0 * ones),
        "declared_line@10:last": rec(0 * ones), "t_pd_k@1000:max": rec(300.0 * ones)})
    s = scorecard._score_steady(res)
    assert s["within_5pm"].tolist() == [True, True]
    assert s["drift_pm_per_ms"][0] == pytest.approx(-0.8, abs=0.01) and s["drift_pm_per_ms"][1] == 0.0
    assert s["locked"].tolist() == [False, True]
