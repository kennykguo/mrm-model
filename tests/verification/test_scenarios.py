"""Checks the scenario package mrm.scenarios: the disturbance and stimulus signals that drive the plant.

Read this first
---------------
A Scenario is a list of components (step, pulse, trapezoid, piecewise-linear, random walk, ...) on named signals
(thermal.t_base_k, laser.dfreq_hz, data.ones_density, ...). compile() turns it into numba arrays for the engine;
render() samples it. The tests check:

- the numba generators against closed forms;
- the counter-based random generators (batch invariance, order independence);
- next_break: between two breaks a signal is linear, so the engine can take exact first-order-hold multi-tick steps;
- time compression: only slow exogenous signals may be compressed, with a load-time rule and a convergence
  certificate;
- the aggressor calibration: the target is given in resonance units at the victim ring, and the solver finds the
  non-negative heater power that, through the thermal path, reproduces it (dlambda/dT applied once);
- the numbers of the named library scenarios against the Lightmatter notes
  (docs/research/01_lightmatter_public.md) and the thermal notes (docs/research/04_thermal_and_control.md);
- the registry, the config entry point and stacking, and identical results with NUMBA_DISABLE_JIT=1.

"The design review" below means docs/research/07_design_critique.md.
"""
import json
import math
import os
import subprocess
import sys

import numba as nb
import numpy as np
import pytest

import mrm.scenarios as scenarios
from mrm.scenarios import calibrate
from mrm.scenarios import generators
from mrm.scenarios import library

JIT_OFF = bool(nb.config.DISABLE_JIT)
DT = 1e-6   # s, plant tick


def _render(sc, n_ticks=None, ticks=None, stride=1, dt=DT, n_rings=2, n_lines=2, batch=1, seed=0, sites=None):
    """Compile scenario sc and render it: returns (compiled arrays, DisturbanceFrame)."""
    arr = sc.compile(dt_s=dt, n_rings=n_rings, n_lines=n_lines, batch=batch, seed=seed, aggressor_sites=sites)
    return arr, arr.render(n_ticks=n_ticks, ticks=ticks, stride_ticks=stride)


def _signal(fr, name, b=0, e=0):
    """The rendered values of signal name for world b and element e (ring or line), one per rendered tick."""
    return fr.signals[name][:, b, e]


# ------------------------------------------------------------------------------------------ Signal table and data model

def test_signal_table_is_the_contract():
    assert [s.id for s in generators.SIGNALS] == list(range(len(generators.SIGNALS)))
    assert len({s.name for s in generators.SIGNALS}) == len(generators.SIGNALS)
    assert all(s.level in generators.LEVELS for s in generators.SIGNALS)
    # ids the engine compiles against (append-only)
    assert generators.SIGNALS[generators.SIG_T_BASE_K].name == "thermal.t_base_k"
    assert generators.SIGNALS[generators.SIG_AGGRESSOR_W].name == "thermal.aggressor_w"
    assert generators.SIGNALS[generators.SIG_XPU_RISE_K].name == "thermal.xpu_rise_k"
    assert generators.SIGNALS[generators.SIG_LASER_DFREQ_HZ].name == "laser.dfreq_hz"
    assert generators.SIGNALS[generators.SIG_BIAS_V].name == "tx.bias_v"
    # only slow exogenous signals may be time-compressed; the aggressor, XPU and data paths may not
    slow = {s.name for s in generators.SIGNALS if s.slow_eligible}
    assert slow == {"thermal.t_base_k", "laser.dfreq_hz", "laser.power_delta_db", "optical.input_loss_delta_db"}
    with pytest.raises(scenarios.CompressionError):
        generators.const("thermal.aggressor_w", 1.0, site="a", slow=True)
    with pytest.raises(scenarios.ScenarioError, match="did you mean"):
        generators.const("thermal.t_base", 300.0)


def test_layout_widths_and_defaults():
    arr = scenarios.Scenario("x").compile(dt_s=DT, n_rings=3, n_lines=5)
    assert arr.layout["thermal.t_base_k"][1] == 1
    assert arr.layout["data.ones_density"][1] == 3
    assert arr.layout["laser.dfreq_hz"][1] == 5
    assert arr.layout["thermal.aggressor_w"][1] == 0
    v = arr.evaluate(0)
    assert v["thermal.t_base_k"][0, 0] == pytest.approx(298.15)
    assert np.all(v["laser.line_enable"] == 1.0) and np.all(v["data.transition_density"] == 0.5)
    assert np.all(np.isnan(v["tx.bias_v"]))                      # not driven
    assert scenarios.Scenario("x").drives == ()


# -------------------------------------------------------------------------------------- Generators against closed forms

def test_const_step_pulse_and_holds():
    sc = scenarios.Scenario("w", components=(
        generators.step("laser.power_delta_db", 2e-6, -1.0, label="s"),
        generators.pulse("optical.input_loss_delta_db", 3e-6, 5e-6, 2.0, label="p"),
        generators.pwl("thermal.xpu_rise_k", [0.0, 4e-6], [1.0, 5.0], t0_s=2e-6, hold_before=True, hold_after=True,
                       t1_s=4e-6, label="h")))
    _, fr = _render(sc, n_ticks=8)
    assert np.allclose(_signal(fr, "laser.power_delta_db"), [0, 0, -1, -1, -1, -1, -1, -1])
    assert np.allclose(_signal(fr, "optical.input_loss_delta_db"), [0, 0, 0, 2, 2, 0, 0, 0])
    # held before t0 at the t0 value (1), linear 1 -> 3 on [2, 4) us, held at the left limit 3 after t1
    assert np.allclose(_signal(fr, "thermal.xpu_rise_k"), [1, 1, 1, 2, 3, 3, 3, 3], atol=1e-6)


@pytest.mark.parametrize("rise, high, fall", [(0.3, 0.2, 0.1), (0.0, 0.5, 0.0), (0.5, 0.0, 0.5), (1.0, 0.0, 0.0)])
def test_trapezoid_family_against_numpy(rise, high, fall):
    P = 1e-3
    c = generators.trapezoid("thermal.xpu_rise_k", P, rise * P, high * P, fall * P, amp=2.5, offset=-1.0,
                             phase_s=0.17e-3, t0_s=0.05e-3, label="t")
    _, fr = _render(scenarios.Scenario("t", components=(c,)), n_ticks=3000)
    k = np.arange(3000)                                   # reference in whole ticks (edges fall on ticks exactly)
    n = round(P / DT)
    want = np.where(k >= 50, -1.0 + 2.5 * generators.np_trapezoid(k - 50 + 170, n, round(rise * n), round(high * n),
                                                                  round(fall * n)), 0.0)
    assert np.max(np.abs(_signal(fr, "thermal.xpu_rise_k") - want)) < 1e-9


def test_square_triangle_sawtooth_builders():
    sq = generators.square("thermal.xpu_rise_k", 1e-4, 0.25, amp=1.0, label="a")
    assert sq.params[:3] == (0.0, 0.25e-4, 0.0)
    tri = generators.triangle("thermal.xpu_rise_k", 1e-4, rise_frac=0.5, label="b")
    assert tri.params[:3] == (0.5e-4, 0.0, 0.5e-4)
    saw = generators.sawtooth("thermal.xpu_rise_k", 1e-4, fall_s=1e-6, label="c")
    assert saw.params[:3] == pytest.approx((0.99e-4, 0.0, 1e-6))
    with pytest.raises(scenarios.ScenarioError):
        generators.trapezoid("thermal.xpu_rise_k", 1e-4, 0.6e-4, 0.0, 0.6e-4)


def test_rect_sine_sine_smooth_step():
    P = 0.1307
    sc = scenarios.Scenario("r", components=(
        generators.rect_sine("thermal.xpu_rise_k", P, amp=-88.5, offset=113.6, label="rs"),
        generators.sine("tx.bias_offset_v", 1e-5, amp=0.005, phase_s=2e-6, label="s"),
        generators.smooth_step("laser.power_delta_db", 1e-3, 2e-3, amp=3.0, label="ss")))
    _, fr = _render(sc, n_ticks=2000, stride=137)
    t = fr.times_s()
    assert np.max(np.abs(_signal(fr, "thermal.xpu_rise_k") - (113.6 - 88.5 * np.abs(np.sin(np.pi * t / P))))) < 1e-9
    assert np.max(np.abs(_signal(fr, "tx.bias_offset_v") - 0.005 * np.sin(2 * np.pi * (t + 2e-6) / 1e-5))) < 1e-12
    assert np.max(np.abs(_signal(fr, "laser.power_delta_db") - 3.0 * generators.np_smooth_step(t - 1e-3, 2e-3))) < 1e-12


def test_pwl_periodic_steps_and_per_element_per_world():
    # periodic state table with jumps (duplicate times), right-continuous
    c = generators.pwl("data.ones_density", [0, 1e-5, 1e-5, 2e-5, 2e-5, 3e-5], [0, 0, 0.5, 0.5, 1, 1], periodic=True,
                       absolute=True, label="p")
    _, fr = _render(scenarios.Scenario("p", components=(c,)), n_ticks=70)
    v = _signal(fr, "data.ones_density")
    assert v[0] == 0 and v[9] == 0 and v[10] == 0.5 and v[19] == 0.5 and v[20] == 1 and v[29] == 1 and v[30] == 0
    assert v[65] == 0.0 and v[55] == 1.0                      # 65 mod 30 = 5, 55 mod 30 = 25
    # per world (B = 3) and per element (2 rings) values
    vals = np.arange(3 * 2 * 2, dtype=float).reshape(3, 2, 2)
    c2 = generators.pwl("tx.bias_offset_v", [0.0, 1e-5], vals, label="q")
    _, fr = _render(scenarios.Scenario("q", components=(c2,)), n_ticks=11, batch=3)
    got = fr.signals["tx.bias_offset_v"]
    assert np.allclose(got[0], vals[:, :, 0]) and np.allclose(got[10], vals[:, :, 1])
    assert np.allclose(got[5], 0.5 * (vals[:, :, 0] + vals[:, :, 1]))
    with pytest.raises(scenarios.ScenarioError, match="per-world"):
        scenarios.Scenario("q", components=(c2,)).compile(dt_s=DT, n_rings=2, n_lines=1, batch=2)


def test_sampled_zoh_linear_periodic():
    v = np.array([0.0, 1.0, 4.0, 9.0])
    lin = generators.sampled("thermal.xpu_rise_k", 1e-5, v, periodic=True, label="a")
    zoh = generators.sampled("laser.power_delta_db", 1e-5, v, linear=False, label="b")
    _, fr = _render(scenarios.Scenario("s", components=(lin, zoh)), n_ticks=60)
    t = fr.times_s()
    want_lin = np.interp(np.mod(t, 4e-5), np.arange(5) * 1e-5, np.append(v, v[0]))
    assert np.max(np.abs(_signal(fr, "thermal.xpu_rise_k") - want_lin)) < 1e-9
    z = _signal(fr, "laser.power_delta_db")
    assert z[0] == 0 and z[9] == 0 and z[10] == 1 and z[25] == 4 and z[59] == 9          # held after the end


def test_burst_envelope_gates_a_component():
    c = generators.with_burst(generators.const("thermal.aggressor_w", 0.2, site="agg", label="b"), on_s=3e-6,
                              off_s=2e-6, phase_s=1e-6)
    _, fr = _render(scenarios.Scenario("b", components=(c,)), n_ticks=12)
    got = _signal(fr, "thermal.aggressor_w")
    want = np.array([0.2 if ((k - 1) % 5) < 3 else 0.0 for k in range(12)])
    assert np.allclose(got, want)


def test_absolute_components_and_conflicts():
    a = generators.const("thermal.t_base_k", 350.0, absolute=True, label="a")
    d = generators.const("thermal.t_base_k", 5.0, label="d")
    _, fr = _render(scenarios.Scenario("a", components=(a, d)), n_ticks=2)
    assert _signal(fr, "thermal.t_base_k")[0] == pytest.approx(355.0)                   # default dropped, delta added
    b = generators.const("thermal.t_base_k", 300.0, absolute=True, label="b")
    with pytest.raises(scenarios.ScenarioError, match="absolutely by both"):
        scenarios.Scenario("c", components=(a, b)).compile(dt_s=DT, n_rings=1, n_lines=1)
    # absolute on some elements only: the others keep the default
    e = generators.const("data.ones_density", 0.9, absolute=True, elements=(1,), label="e")
    _, fr = _render(scenarios.Scenario("e", components=(e,)), n_ticks=1, n_rings=3)
    assert np.allclose(fr.signals["data.ones_density"][0, 0], [0.5, 0.9, 0.5])


def test_clipping_and_defaults_override():
    sc = scenarios.Scenario("c", components=(generators.const("laser.line_enable", 0.7, label="x"),
                                             generators.const("thermal.aggressor_w", -1.0, site="s", label="y")),
                            defaults={"thermal.t_base_k": 310.0, "data.ones_density": [0.1, 0.2]})
    _, fr = _render(sc, n_ticks=1)
    assert np.all(fr.signals["laser.line_enable"] == 1.0)            # 1 + 0.7 clipped to 1
    assert np.all(fr.signals["thermal.aggressor_w"] == 0.0)          # a heater only heats
    assert fr.signals["thermal.t_base_k"][0, 0, 0] == 310.0
    assert np.allclose(fr.signals["data.ones_density"][0, 0], [0.1, 0.2])


# ----------------------------------------------------------------------------------------------- Counter RNG generators

def _rw_scenario(common=False):
    """A random walk on the laser power and random ones-density levels, both from the counter-based RNG."""
    return scenarios.Scenario("rw", components=(
        generators.random_walk("laser.power_delta_db", knot_s=1e-5, sigma_per_rt_s=2.0, drift_per_s=50.0,
                               common=common, label="rw", stream="scenario_test_rw"),
        generators.random_levels("data.ones_density", interval_s=3e-5, dist="uniform", a=0.2, b=0.8, common=False,
                                 absolute=True, label="rl", stream="scenario_test_rw")))


def test_batch_invariance_element_b_equals_seed_plus_b():
    sc = _rw_scenario()
    _, frB = _render(sc, n_ticks=300, batch=4, seed=11, n_lines=3, n_rings=2)
    for b in range(4):
        _, fr1 = _render(sc, n_ticks=300, batch=1, seed=11 + b, n_lines=3, n_rings=2)
        for k in ("laser.power_delta_db", "data.ones_density"):
            assert np.array_equal(frB.signals[k][:, b], fr1.signals[k][:, 0])


def test_random_walk_is_order_independent_and_linear_between_knots():
    sc = _rw_scenario()
    arr = sc.compile(dt_s=DT, n_rings=2, n_lines=3, batch=2, seed=3)
    seq = arr.render(n_ticks=400).signals["laser.power_delta_db"]
    rng = np.random.default_rng(0)
    cache = arr.new_cache()
    for k in rng.permutation(400)[:120]:                          # random order: cache misses, backwards jumps
        v = arr.evaluate(int(k), cache)["laser.power_delta_db"]
        assert np.array_equal(v, seq[k])
    # linear between knots (every 10 ticks)
    x = seq[:, 0, 0]
    knots = x[::10]
    assert np.allclose(x[:391], np.interp(np.arange(391), np.arange(0, 400, 10), knots), atol=1e-12)
    # per-line walks differ; a common walk is identical across lines
    assert not np.allclose(seq[:, 0, 0], seq[:, 0, 1])
    com = scenarios.Scenario("c", components=(
        generators.random_walk("laser.power_delta_db", knot_s=1e-5, sigma_per_rt_s=2.0, common=True, label="rw",
                               stream="scenario_test_rw"),))
    _, fr = _render(com, n_ticks=100, n_lines=3)
    v = fr.signals["laser.power_delta_db"][:, 0]
    assert np.array_equal(v[:, 0], v[:, 2])


def test_random_walk_statistics():
    sig, mu, T = 2.0, 50.0, 1e-3
    c = generators.random_walk("laser.dfreq_hz", knot_s=1e-5, sigma_per_rt_s=sig, drift_per_s=mu, common=False,
                               label="rw", stream="scenario_test_stats")
    arr = scenarios.Scenario("s", components=(c,)).compile(dt_s=DT, n_rings=1, n_lines=64, batch=16, seed=5)
    v = arr.evaluate(int(round(T / DT)))["laser.dfreq_hz"].ravel()         # 1024 independent walks at t = T
    assert abs(v.mean() - mu * T) < 4 * sig * math.sqrt(T) / math.sqrt(v.size)
    assert v.std() == pytest.approx(sig * math.sqrt(T), rel=0.1)


def test_random_levels_distributions_and_bernoulli_link():
    n = generators.random_levels("optical.input_loss_delta_db", interval_s=1e-6, dist="normal", a=0.1, b=0.35, lo=-0.43,
                                 label="n", stream="scenario_test_rl")
    arr, fr = _render(scenarios.Scenario("n", components=(n,)), n_ticks=4000)
    v = _signal(fr, "optical.input_loss_delta_db")
    assert v.min() >= -0.43 and abs(np.median(v) - 0.1) < 0.03 and v.std() == pytest.approx(0.33, rel=0.1)
    sc = library.ones_density_steps(mode="random", levels=(0.0, 0.25, 0.5, 1.0), step_s=1e-5, common=False)
    _, fr = _render(sc, n_ticks=2000, n_rings=3)
    p = fr.signals["data.ones_density"][:, 0]
    td = fr.signals["data.transition_density"][:, 0]
    assert set(np.unique(p)) <= {0.0, 0.25, 0.5, 1.0} and len(np.unique(p)) == 4
    assert np.allclose(td, 2 * p * (1 - p))                                   # linked draws (Bernoulli data)
    assert not np.array_equal(p[:, 0], p[:, 1])                              # per-ring statistics
    assert np.all(np.diff(p[:, 0])[np.arange(1999) % 10 != 9] == 0)          # levels change only every 10 ticks


# -------------------------------------------------------------------------- Next break: exact FOH over multi-tick steps

def test_next_break_gives_linear_spans():
    comps = (generators.trapezoid("thermal.xpu_rise_k", 7.3e-5, 1.1e-5, 2.0e-5, 0.7e-5, amp=3.0, phase_s=4e-6,
                                  label="a"),
             generators.random_walk("laser.dfreq_hz", knot_s=1.3e-5, sigma_per_rt_s=1e9, common=False, label="b",
                                    stream="scenario_nb"),
             generators.random_levels("data.ones_density", interval_s=2.1e-5, absolute=True, label="c",
                                      stream="scenario_nb"),
             generators.pwl("tx.bias_offset_v", [0, 1.7e-5, 1.7e-5, 4e-5], [0, 1, -1, 0.5], t0_s=5e-6, label="d"),
             generators.rect_sine("laser.power_delta_db", 2e-2, amp=1.0, label="e"),
             generators.with_burst(generators.const("thermal.aggressor_w", 1.0, site="s", label="f"), 9e-6, 6e-6))
    arr = scenarios.Scenario("nb", components=comps).compile(dt_s=DT, n_rings=2, n_lines=2)
    fr = arr.render(n_ticks=400)
    for name in ("thermal.xpu_rise_k", "laser.dfreq_hz", "data.ones_density", "tx.bias_offset_v",
                 "laser.power_delta_db", "thermal.aggressor_w"):
        x = fr.signals[name][:, 0, 0]
        k = 0
        spans = 0
        while k < 380:
            kb = arr.next_break(name, k)
            assert kb > k
            kb = min(kb, 399)
            lin = np.interp(np.arange(k, kb + 1), [k, kb], [x[k], x[kb]])
            tol = 1.2 * generators.SIGNAL_BY_NAME[name].foh_tol if name == "laser.power_delta_db" else \
                1e-9 * max(1.0, np.max(np.abs(x)))
            assert np.max(np.abs(x[k:kb + 1] - lin)) <= tol, (name, k, kb)
            spans += 1
            k = kb
        assert spans < 380                                       # multi-tick steps happen
    const = scenarios.Scenario("k", components=(generators.const("thermal.xpu_rise_k", 1.0, label="k"),)).compile(
        dt_s=DT, n_rings=1, n_lines=1)
    assert const.next_break("thermal.xpu_rise_k", 5) == generators.NO_BREAK


# ----------------------------------------------------------------------------------------------------- Time compression

def test_compression_speeds_up_only_slow_components():
    base = library.soak(n_cycles=3)
    fast = generators.square("thermal.xpu_rise_k", 10.0, 0.5, amp=1.0, label="fast")
    sc1 = base + scenarios.Scenario("f", components=(fast,))
    sc50 = sc1.with_compression(50.0)
    dt = 1.0
    _, f1 = _render(sc1, ticks=np.arange(0, 3000, 50), dt=dt, n_rings=1, n_lines=1)
    _, f50 = _render(sc50, ticks=np.arange(0, 60, 1), dt=dt, n_rings=1, n_lines=1)
    # T_base(50 t) = T_base(t)
    assert np.allclose(_signal(f50, "thermal.t_base_k"), _signal(f1, "thermal.t_base_k"))
    _, fx = _render(sc1, ticks=np.arange(0, 60, 1), dt=dt, n_rings=1, n_lines=1)
    assert np.array_equal(_signal(f50, "thermal.xpu_rise_k"), _signal(fx, "thermal.xpu_rise_k"))  # fast part unchanged


@pytest.mark.filterwarnings("ignore:.*no K/s equivalent")
def test_compression_load_time_rule():
    rep = scenarios.check_time_compression(library.soak().with_compression(100))
    assert rep["components"]["soak"]["max_compression"] == pytest.approx(
        0.01 * generators.FAST_RATE_REF_K_PER_S / (75.0 / 609.0), rel=1e-9)                    # about 173
    with pytest.raises(scenarios.CompressionError, match="time_compression rule"):
        library.soak().with_compression(1000).compile(dt_s=1.0, n_rings=1, n_lines=1)
    # laser frequency rates are compared through 15 GHz/degC; the 24 h drift allows huge compression
    rep = scenarios.check_time_compression(library.laser_drift().with_compression(1000))
    assert rep["components"]["laser_freq_drift"]["max_compression"] > 1e6
    with pytest.raises(scenarios.CompressionError):
        scenarios.Scenario("x", time_compression=0.5)


def test_convergence_plan_assessment_and_certificate():
    sc = library.soak()
    plan = scenarios.convergence_plan(sc)
    assert sorted(plan) == [1.0, 10.0, 100.0]
    assert plan[100.0][1] == pytest.approx(library.SOAK_CYCLE_S / 100) and plan[100.0][0].time_compression == 100.0
    ref = {"max_abs_detuning_pm": 3.0, "wrong_line_events": 0, "fsr_hop_events": 0, "order_changes": 0,
           "heater_range_used_codes": 21000, "integrated_errors": 50.0}
    ok = {1.0: ref, 10.0: dict(ref, max_abs_detuning_pm=3.1), 100.0: dict(ref, heater_range_used_codes=21100)}
    rep = scenarios.assess_convergence(sc, ok, plant_fingerprint="abc")
    assert rep.passed and rep.certificate.max_validated_factor == 100.0
    scenarios.require_compression_validated(sc.with_compression(100), rep.certificate, plant_fingerprint="abc")
    cert = scenarios.ConvergenceCertificate.from_dict(json.loads(json.dumps(rep.certificate.to_dict())))
    scenarios.require_compression_validated(sc.with_compression(50), cert, plant_fingerprint="abc")
    with pytest.raises(scenarios.CompressionError, match="different plant"):
        scenarios.require_compression_validated(sc.with_compression(10), cert, plant_fingerprint="xyz")
    with pytest.raises(scenarios.CompressionError, match="without a convergence"):
        scenarios.require_compression_validated(sc.with_compression(10), None)
    bad = {**ok, 100.0: dict(ref, fsr_hop_events=1)}
    rep = scenarios.assess_convergence(sc, bad, plant_fingerprint="abc")
    assert not rep.passed and rep.certificate.max_validated_factor == 10.0
    with pytest.raises(scenarios.CompressionError, match="exceeds the validated"):
        scenarios.require_compression_validated(sc.with_compression(100), rep.certificate, plant_fingerprint="abc")
    scenarios.require_compression_validated(sc, None)                                 # uncompressed: nothing needed


# ------------------------------------------------------------------ DisturbanceFrame (recorded profiles) and JIT parity

@pytest.mark.filterwarnings("ignore:.*no K/s equivalent")
def test_disturbance_frame_roundtrip():
    sc = (library.soak(n_cycles=2) + library.ones_density_steps(step_s=0.5)
          + library.laser_drift(mode="random_walk", knot_s=1.0))
    _, fr = _render(sc, n_ticks=200, stride=25, dt=1.0, n_rings=2, n_lines=2, batch=2, seed=4)
    replay = scenarios.Scenario.from_frame(fr)
    _, fr2 = _render(replay, ticks=fr.ticks, dt=1.0, n_rings=2, n_lines=2, batch=2)
    for k in ("thermal.t_base_k", "data.ones_density", "laser.dfreq_hz", "laser.power_delta_db"):
        assert np.allclose(fr2.signals[k], fr.signals[k], rtol=0, atol=1e-9 * max(1, np.abs(fr.signals[k]).max()))


_NO_JIT_PARITY_SCRIPT = r"""
import json, numpy as np
from mrm.scenarios import library
sc = (library.soak(n_cycles=2) + library.ones_density_steps(mode="random", step_s=0.01)
      + library.laser_drift(mode="random_walk", knot_s=0.5) + library.connector_remate(interval_s=0.05)
      + library.xpu_power_step() + library.bias_ripple(freq_hz=500.0) + library.laser_sweep_emulation()
      + library.squelch(t_on_s=0.01, t_off_s=0.02) + library.laser_failover(gap_s=0.003))
arr = sc.compile(dt_s=1e-4, n_rings=2, n_lines=2, batch=2, seed=9)
fr = arr.render(n_ticks=300, stride_ticks=7)
out = {k: v.tolist() for k, v in fr.signals.items()}
out["breaks"] = [[arr.next_break(s, k) for k in (0, 5, 99, 250, 1777)] for s in arr.active_signals]
print(json.dumps(out))
"""


def test_numba_disable_jit_parity():
    """Identical signals with NUMBA_DISABLE_JIT=1 (pure NumPy scalars) and compiled."""
    env = dict(os.environ)
    a = subprocess.run([sys.executable, "-c", _NO_JIT_PARITY_SCRIPT], capture_output=True, text=True, env=env,
                       check=True)
    env["NUMBA_DISABLE_JIT"] = "0" if JIT_OFF else "1"
    b = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", _NO_JIT_PARITY_SCRIPT],
                       capture_output=True, text=True, env=env, check=True)
    assert json.loads(a.stdout) == json.loads(b.stdout)


# -------------------------------------- Aggressor calibration: a target in resonance units, power into the thermal path

# a 4-pole thermal step response for the calibration tests: time constants (s) and residues (K/W)
FOSTER = calibrate.FosterResponse([2e-6, 1.3e-5, 4.5e-4, 4e-3], [100.0, 150.0, 300.0, 50.0])


def test_foster_ramp_is_integral_of_step():
    t = np.concatenate([[0.0, 1e-12, 1e-9], np.geomspace(1e-7, 0.1, 50)])
    from scipy.integrate import quad
    for ti in t[::7]:
        want = quad(lambda s: float(FOSTER.step(s)), 0, ti, limit=200, points=[2e-6, 1.3e-5])[0] if ti > 0 else 0.0
        assert FOSTER.ramp(ti) == pytest.approx(want, rel=1e-8, abs=1e-18)


def test_response_matrices_closed_form_equals_numerical():
    A1, _ = calibrate.periodic_response_matrix(FOSTER, 0.1307, 48, 4)
    A2, info = calibrate.periodic_response_matrix(FOSTER.step, 0.1307, 48, 4)
    assert np.max(np.abs(A1 - A2)) < 1e-9 * np.max(np.abs(A1))
    assert np.allclose(A1.sum(axis=1), FOSTER.dc())                  # sum of all hats = 1 W -> DC gain
    B1 = calibrate.transient_response_matrix(FOSTER, 0.05, 48, 4)
    B2 = calibrate.transient_response_matrix(FOSTER.step, 0.05, 48, 4)
    assert np.max(np.abs(B1 - B2)) < 1e-9 * np.max(np.abs(B1))


def test_periodic_matrix_equals_long_foh_simulation():
    P, n, os_ = 0.1307, 64, 4
    D = P / n
    A, _ = calibrate.periodic_response_matrix(FOSTER, P, n, os_)
    p = np.abs(np.sin(np.arange(n) * 0.3)) + 0.1
    h = D / os_
    tt = np.arange(0, 3 * P + 0.5 * h, h)                            # slowest pole 4 ms << P
    u = np.interp(np.mod(tt, P), np.append(np.arange(n) * D, P), np.append(p, p[0]))
    r, _ = calibrate.simulate_foster_foh(FOSTER, u, h)
    last = slice(tt.size - n * os_ - 1, tt.size - 1)
    idx = np.round(np.mod(tt[last], P) / h).astype(int) % (n * os_)
    assert np.max(np.abs(r[last] - (A @ p)[idx])) < 1e-9 * np.max(np.abs(r))


def test_calibration_recovers_a_known_power():
    P, n = 0.1, 64
    true = 0.05 + 0.04 * np.sin(2 * np.pi * np.arange(n) / n) ** 2
    A, _ = calibrate.periodic_response_matrix(FOSTER, P, n, 4)
    rise = A @ true
    t_s = np.arange(n * 4) * (P / n / 4)
    tgt = calibrate.AggressorTarget(units="temperature", form="table", period_s=P, t_base_k=300.0,
                                    table_t_s=t_s, table_k=300.0 + rise)
    cw = calibrate.calibrate_aggressor_power(FOSTER, tgt, n_knots=n, oversample=4, reg=1e-9)
    assert cw.residual_rms_k[0] < 1e-6 * rise.max()
    assert np.max(np.abs(cw.power_w[0] - true)) < 1e-4 * true.max()
    assert cw.provenance == "calibrated" and cw.record()["assumed_thermal_response"][0]["form"] == "foster"


def test_resonance_referencing_applies_dlambda_dt_once():
    sc = library.a1_slide10_aggressor()
    tgt = sc.aggressors[0].target
    t = np.linspace(0, tgt.period_s, 400, endpoint=False)
    for s0, beta in ((66e-12, 1.4e-3), (85.86e-12, 0.0), (70e-12, 1.0e-3)):
        x = tgt.ring_rise_k(t, s0_m_per_k=s0, beta_per_k=beta, t_ref_k=298.15)
        back = calibrate.rise_to_resonance_m(x, s0_m_per_k=s0, beta_per_k=beta, t_ref_k=298.15, t_base_k=tgt.t_base_k)
        assert np.allclose(back, tgt.resonance_m(t), rtol=1e-12, atol=1e-20)    # plant shift == target shift
    # under H86 with beta = 0 the rise equals the printed excursion; under a lower dlambda/dT it is larger
    x86 = tgt.ring_rise_k(t, s0_m_per_k=calibrate.LM_DLAMBDA_DT_M_PER_K)
    assert np.allclose(x86, tgt.excursion_k(t))
    x66 = tgt.ring_rise_k(t, s0_m_per_k=66e-12)
    assert np.max(x66) / np.max(x86) == pytest.approx(85.8644 / 66.0, rel=1e-4)    # 1.2-1.3x (design review)
    # temperature referencing: no dlambda/dT, divided by the thermometer fraction
    tt = library.a1_slide10_aggressor(referencing="temperature", thermometer_fraction=0.8).aggressors[0].target
    assert np.allclose(tt.ring_rise_k(t), tgt.excursion_k(t) / 0.8)
    with pytest.raises(ValueError, match="dlambda/dT"):
        tgt.ring_rise_k(t)


def test_nonnegativity_is_enforced_and_reported():
    """A target that dives below the base faster than the path can cool needs negative power: the solver keeps
    p >= 0 and reports the residual instead of inventing a cooler."""
    tgt = calibrate.AggressorTarget(units="temperature", form="trapezoid", period_s=0.1, t_base_k=300.0, t_low_k=300.0,
                                    delta_k=50.0, rise_s=0.0995, high_s=0.0, fall_s=0.0005)
    slow = calibrate.FosterResponse([0.02], [500.0])                          # 20 ms pole: cannot fall 50 K in 0.5 ms
    cw = calibrate.calibrate_aggressor_power(slow, tgt, n_knots=200, reg=1e-6)
    assert np.all(cw.power_w >= 0) and cw.lower_bound_fraction[0] > 0
    assert cw.residual_max_k[0] > 5.0                                 # honest: the return edge is not reachable
    unc = np.linalg.lstsq(calibrate.periodic_response_matrix(slow, 0.1, 200, 4)[0],
                          tgt.ring_rise_k(np.arange(800) * 0.1 / 800), rcond=None)[0]
    assert unc.min() < 0                                              # the unconstrained answer would cool


def test_a1_end_to_end_through_the_kernel():
    sc = scenarios.create("a1", victim_ring=1, site="agg")
    with pytest.raises(scenarios.ScenarioError, match="calibrat"):
        sc.compile(dt_s=DT, n_rings=2, n_lines=2)
    resp = calibrate.FosterResponse([4.5e-4, 4e-3], [500.0, 50.0], info={"note": "test path"})
    model = {"s0_m_per_k": [70e-12, 85.86e-12], "beta_per_k": 1.4e-3, "t_ref_k": 298.15}
    sc2, cal = scenarios.calibrate_scenario(sc, {"agg": [resp, resp]}, model=model, n_knots=256)
    cw = cal["agg"]
    assert cw.power_w.shape == (2, 256) and np.all(cw.power_w >= 0)
    assert cw.residual_rms_k.max() < 0.05 and cw.residual_rms_m.max() < 4e-12
    assert cw.peak_power_w[0] > cw.peak_power_w[1]                    # lower dlambda/dT -> more heat needed
    rec = sc2.aggressors[0].record
    assert rec["provenance"] == "calibrated" and rec["assumed_thermal_response"][0]["taus_s"] == [4.5e-4, 4e-3]
    # the kernel's piecewise-linear power through an exact FOH reproduces the target in periodic steady state
    P = cw.period_s
    arr = sc2.compile(dt_s=DT, n_rings=2, n_lines=2, batch=2, aggressor_sites=["other", "agg"])
    assert arr.layout["thermal.aggressor_w"][1] == 2
    stride = 20
    fr = arr.render(n_ticks=int(4 * P / (stride * DT)), stride_ticks=stride)
    assert np.all(fr.signals["thermal.aggressor_w"][:, :, 0] == 0.0)         # the other site stays off
    t = fr.times_s()
    for b in range(2):
        u = fr.signals["thermal.aggressor_w"][:, b, 1]
        r, _ = calibrate.simulate_foster_foh(resp, u, stride * DT)
        last = t >= 3 * P
        y = sc.aggressors[0].target.ring_rise_k(t[last], s0_m_per_k=model["s0_m_per_k"][b], beta_per_k=1.4e-3,
                                                t_ref_k=298.15)
        err = r[last] - y
        assert np.sqrt(np.mean(err ** 2)) < 1.5 * cw.residual_rms_k[b] + 1e-3
    assert fr.signals["thermal.t_base_k"][0, 0, 0] == pytest.approx(298.15)   # package at 25 degC


def test_transient_target_from_rest():
    tgt = calibrate.AggressorTarget(units="temperature", form="trapezoid", period_s=None, horizon_s=0.02,
                                    t_base_k=300.0, t_low_k=300.0, delta_k=10.0, rise_s=0.01, high_s=1.0, fall_s=0.0)
    cw = calibrate.calibrate_aggressor_power(FOSTER, tgt, n_knots=100, reg=1e-7)
    assert cw.residual_rms_k[0] < 0.05
    sc = scenarios.Scenario("t", aggressors=(generators.AggressorSpec("agg", target=tgt),))
    sc2, _ = scenarios.calibrate_scenario(sc, {"agg": FOSTER}, n_knots=100, reg=1e-7)
    fr = sc2.render(dt_s=1e-5, n_ticks=2001)
    u = fr.signals["thermal.aggressor_w"][:, 0, 0]
    assert u[0] == 0.0
    r, _ = calibrate.simulate_foster_foh(FOSTER, u, 1e-5)
    assert np.max(np.abs(r[200:] - tgt.ring_rise_k(fr.times_s()[200:]))) < 0.3


@pytest.mark.parametrize("periodic", [True, False])
def test_mean_power_is_the_time_mean_of_the_waveform(periodic):
    """mean_power_w is the time mean of power_at over one period (periodic) or over [0, horizon] (one-shot, which
    starts from 0 W at t = 0), not the plain knot mean."""
    if periodic:
        tgt = calibrate.AggressorTarget(units="temperature", form="rectified_sine", period_s=0.01, t_base_k=300.0,
                                        t_max_k=310.0, delta_k=10.0)
        t_end = 0.01
    else:
        tgt = calibrate.AggressorTarget(units="temperature", form="trapezoid", period_s=None, horizon_s=0.02,
                                        t_base_k=300.0, t_low_k=300.0, delta_k=10.0, rise_s=0.01, high_s=1.0,
                                        fall_s=0.0)
        t_end = 0.02
    cw = calibrate.calibrate_aggressor_power(FOSTER, tgt, n_knots=64, reg=1e-7)
    t = np.linspace(0.0, t_end, 64 * 50 + 1)                   # 50 samples per knot: exact for a piecewise-linear p
    p = cw.power_at(t)
    time_mean_w = np.sum(0.5 * (p[1:] + p[:-1]) * np.diff(t)) / t_end
    assert cw.mean_power_w[0] == pytest.approx(time_mean_w, rel=1e-9)
    assert cw.record()["mean_power_w"] == pytest.approx([time_mean_w], rel=1e-9)


def test_foster_from_system_duck_typed_with_the_thermal_module():
    thermal = pytest.importorskip("mrm.plant.thermal")
    lay = thermal.LayoutSites.line(4, 30.0, aggressor_xy_um=[[0.0, 20.0]], aggressor_names=("aggressor",))
    sysm = thermal.assemble(thermal.InterimRom(), lay, 4, sources=("heater", "aggressor"),
                            observers=("ring_mode_weighted",))
    resp = scenarios.foster_from_system(sysm, victim_ring=0, site="aggressor")
    assert len(resp) == sysm.batch
    g = thermal.step_response(sysm, [1e-4, 1e-2], obs=[sysm.observer_index("ring_mode_weighted", ring=0)],
                              src=[sysm.source_index("aggressor", name="aggressor")])[0, 0, 0]
    assert np.allclose(resp[0].step([1e-4, 1e-2]), g)


# ------------------------------ Library numbers (Lightmatter notes with their corrections, thermal notes, textbook REF)

def test_a1_numbers_follow_the_corrected_slide10_fit():
    tgt = scenarios.create("a1").aggressors[0].target
    assert tgt.form == "rectified_sine"
    assert tgt.peak_rate_k_per_s() == pytest.approx(2127.2, abs=0.5)        # pi dT / P (CAL-13a derived)
    assert tgt.mean_rate_k_per_s() == pytest.approx(1354.3, abs=0.5)        # 2 dT / P
    assert tgt.printed_k(0.0) - 273.15 == pytest.approx(113.6)              # cusp at t = 0
    assert tgt.printed_k(tgt.period_s / 2) - 273.15 == pytest.approx(25.1)
    assert tgt.s_ref_m_per_k * 1e12 == pytest.approx(85.86, abs=0.01)       # 15 GHz/degC at 1310 nm
    tri = scenarios.create("a1", waveform="symmetric_triangle").aggressors[0].target
    assert tri.printed_k(0.0) == pytest.approx(tgt.printed_k(0.0))
    assert tri.peak_rate_k_per_s() == pytest.approx(2 * 88.5 / 0.1307)


def test_a2_readings():
    tri = scenarios.create("a2").aggressors[0].target
    assert tri.delta_k == pytest.approx(80.8) and tri.peak_rate_k_per_s() == pytest.approx(1616.0)
    saw = scenarios.create("a2", reading="one_sided_sawtooth").aggressors[0].target
    assert saw.rise_s == pytest.approx(80.0 / 808.0) and saw.fall_s == pytest.approx(0.1 - 80.0 / 808.0)
    assert saw.delta_k / saw.rise_s == pytest.approx(808.0)
    rs = scenarios.create("a2", reading="rectified_sine").aggressors[0].target
    # ~2538 degC/s (Lightmatter notes, "The 808 degC/s definition")
    assert rs.peak_rate_k_per_s() == pytest.approx(math.pi * 80.8 / 0.1)


def test_pd_adjacent_stress_scenarios_are_a1_a2_at_the_stress_site():
    """The stress scenarios of monitor dark-current runaway (not a realistic layout) are A1 / A2 unchanged except the
    site: aggressor_pd_adjacent instead of the default aggressor."""
    assert (library.AGGRESSOR_SITE, library.AGGRESSOR_PD_ADJACENT_SITE) == ("aggressor", "aggressor_pd_adjacent")
    t = np.linspace(0.0, 0.3, 61)
    for short, full, base in (("a1_pd_adjacent", "a1_slide10_aggressor_pd_adjacent", "a1"),
                              ("a2_pd_adjacent", "a2_808_aggressor_pd_adjacent", "a2")):
        assert full in scenarios.available() and scenarios.resolve(short) is scenarios.resolve(full)
        kw = {"victim_ring": 1, **({"phase_s": 0.01} if base == "a1" else {})}
        sc, ref = scenarios.create(short, **kw), scenarios.create(base, **kw)
        assert sc.name == full and "STRESS" in sc.description
        assert sc.aggressor_sites == (library.AGGRESSOR_PD_ADJACENT_SITE,) and \
            ref.aggressor_sites == (library.AGGRESSOR_SITE,)
        a, b = sc.aggressors[0], ref.aggressors[0]
        assert a.victim_ring == b.victim_ring == 1 and a.pending and b.pending
        assert np.array_equal(a.target.printed_k(t), b.target.printed_k(t))
        assert a.target.period_s == b.target.period_s and a.target.units == b.target.units
        assert sc.defaults == ref.defaults
        with pytest.raises(scenarios.ScenarioError, match="site"):        # the stress builders fix the site
            scenarios.create(short, site="aggressor")
        assert "site" not in scenarios.describe(short).split("\n")[0]


def test_soak_profile():
    sc = scenarios.create("a4")
    assert sc.duration_s == pytest.approx(90 * 20.3 * 60) and sc.duration_s / 3600 == pytest.approx(30.45)
    _, fr = _render(sc, ticks=np.array([0, 609, 1218, 1218 + 304, 90 * 1218, 90 * 1218 + 1000]), dt=1.0,
                    n_rings=1, n_lines=1)
    v = _signal(fr, "thermal.t_base_k") - 273.15
    assert np.allclose(v[:3], [33.0, 108.0, 33.0]) and v[3] == pytest.approx(33.0 + 75.0 * 304 / 609)
    assert v[4] == pytest.approx(33.0, abs=1e-6) and v[5] == pytest.approx(33.0, abs=1e-6)   # held after the end
    rate = generators.component_max_rate(sc.components[0])
    assert rate == pytest.approx(0.123, abs=0.001)                  # ~0.124 degC/s (Lightmatter notes, section 2.5)


def test_xpu_sweep_drift_failover():
    _, fr = _render(scenarios.create("b3"), n_ticks=1500, dt=1e-4, n_rings=1, n_lines=1)
    v = _signal(fr, "thermal.xpu_rise_k")
    r = np.gradient(v, fr.times_s())
    assert r.max() == pytest.approx(120.0, rel=1e-3) and fr.times_s()[np.argmax(r)] == pytest.approx(0.04)
    assert v[-1] == pytest.approx(4.0 * 120.0 * 0.03 / math.pi)
    _, fr = _render(scenarios.create("a3"), n_ticks=4400, dt=1e-4, n_rings=1, n_lines=2)
    v = _signal(fr, "laser.dfreq_hz", e=1)
    lam = 1310e-9
    assert v.max() == pytest.approx(299792458.0 * 4e-9 / lam ** 2, rel=1e-9)     # 4 nm -> 699 GHz
    assert np.max(np.gradient(v, fr.times_s())) == pytest.approx(299792458.0 * 50e-9 / lam ** 2, rel=1e-6)
    assert np.isclose(v, v.max()).sum() > 0.3 * 4400                        # plateaus (0.14 s of each 0.44 s)
    _, fr = _render(scenarios.create("laser_drift"), ticks=np.array([0, 43200, 86400]), dt=1.0, n_rings=1, n_lines=2)
    assert fr.signals["laser.dfreq_hz"][-1, 0] == pytest.approx(-299792458.0 * 0.02e-9 / lam ** 2)  # -3.49 GHz
    assert fr.signals["laser.power_delta_db"][-1, 0] == pytest.approx(10 * math.log10(0.975))
    _, fr = _render(scenarios.create("b2", line=1, t_fail_s=1e-5, gap_s=5e-6), n_ticks=30, n_lines=3)
    le = fr.signals["laser.line_enable"][:, 0]
    assert np.all(le[10:15, 1] == 0) and np.all(le[:, [0, 2]] == 1) and le[15, 1] == 1
    assert fr.signals["laser.dfreq_hz"][15, 0, 1] == 10e9 and fr.signals["laser.dfreq_hz"][14, 0, 1] == 0
    assert fr.signals["laser.power_delta_db"][20, 0].tolist() == [0.0, -1.0, 0.0]


def test_connector_remate_levels():
    sc = scenarios.create("a6", interval_s=1e-3, t0_s=5e-4, dark_s=1e-4)
    _, fr = _render(sc, n_ticks=1300, stride=1, dt=1e-5, n_rings=1, n_lines=2, batch=8, seed=2)
    v = fr.signals["optical.input_loss_delta_db"][:, :, 0]
    assert np.all(v[:50] == 0)
    ch = np.flatnonzero(np.any(np.diff(v, axis=0) != 0, axis=1)) + 1
    assert set(ch) <= {50 + 100 * j for j in range(10)}
    assert np.all(v >= -0.43) and np.all(v[1060] == v[-1])                    # held after the tenth re-mate
    assert not np.allclose(v[-1, 0], v[-1, 1])                                 # each world its own draws
    le = fr.signals["laser.line_enable"][:, 0, 0]
    assert np.all(le[50:60] == 0) and np.all(le[60:150] == 1) and np.all(le[1060:] == 1)
    levels = fr.signals["optical.input_loss_delta_db"][50 + 100 * np.arange(10)][:, :, 0].ravel()
    assert 0.15 < np.std(levels) < 0.6                                        # sigma 0.35 dB (80 draws)


def test_traffic_squelch_idle_and_ones_density():
    sc = scenarios.create("traffic", t_start_s=1e-5, t_stop_s=3e-5)
    _, fr = _render(sc, n_ticks=40)
    for k, on in (("data.driver_enable", 1), ("data.swing_scale", 1), ("data.transition_density", 0.5)):
        x = _signal(fr, k)
        assert np.all(x[:10] == 0) and np.all(x[10:30] == on) and np.all(x[30:] == 0)
    _, fr = _render(scenarios.create("b1", t_on_s=1e-5, t_off_s=2e-5), n_ticks=30)
    x = _signal(fr, "data.swing_scale")
    assert np.all(x[:10] == 1) and np.all(x[10:20] == 0) and np.all(x[20:] == 1)
    _, fr = _render(scenarios.create("idle", t0_s=1e-5, level=1), n_ticks=20)
    assert _signal(fr, "data.ones_density")[15] == 1 and _signal(fr, "data.transition_density")[15] == 0
    assert _signal(fr, "data.ones_density")[5] == 0.5
    _, fr = _render(scenarios.create("ones_density_steps"), ticks=np.array([0, 199999, 200000, 400000, 600000, 650000]))
    assert _signal(fr, "data.ones_density").tolist() == [0.0, 0.0, 0.5, 1.0, 0.0, 0.0]          # Sun et al. 200 ms
    assert _signal(fr, "data.transition_density").tolist() == [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]


def test_traffic_description_names_a_stop_at_zero():
    """A stop at t = 0.0 s is a stop: the description names it (only None means no stop)."""
    assert library.traffic_start_stop(t_start_s=0.0, t_stop_s=0.0).description == "traffic on at 0.0 s, off at 0.0 s"
    assert library.traffic_start_stop(t_start_s=0.0).description == "traffic on at 0.0 s"


def test_bias_cold_warm_reloma_line_enable():
    ripple = scenarios.create("bias_ripple")
    assert ripple.offsets == ("tx.bias",) and ripple.drives == ()
    rp = scenarios.create("a7", sweep_s=1e-5)
    assert rp.drives == ("tx.bias",)
    _, fr = _render(rp, n_ticks=90, n_rings=2)
    assert _signal(fr, "tx.bias_v")[0] == 1.0 and _signal(fr, "tx.bias_v")[75] == 3.75
    assert _signal(fr, "thermal.t_base_k")[5] == pytest.approx(273.15 + 52.5)
    cold = scenarios.create("b5")
    assert cold.reset.start == "cold" and not cold.reset.factory_cal and cold.reset.heater_init == "zero"
    _, fr = _render(cold, n_ticks=3, batch=16, seed=1)
    amb = fr.signals["thermal.t_base_k"][:, :, 0] - 273.15
    assert np.all((amb >= 0) & (amb <= 105)) and np.all(amb[0] == amb[2]) and np.ptp(amb[0]) > 10
    warm = scenarios.create("warm_start")
    assert warm.reset.start == "warm" and warm.reset.factory_cal
    _, fr = _render(scenarios.create("line_enable", events=[(1e-5, [0, 2], False), (2e-5, 0, True)]), n_ticks=30,
                    n_lines=3)
    le = fr.signals["laser.line_enable"][:, 0]
    assert le[5].tolist() == [1, 1, 1] and le[15].tolist() == [0, 1, 0] and le[25].tolist() == [1, 1, 0]


# ------------------------------------------------------------------------------- Registry, config entry point, stacking

def test_registry_names_aliases_and_errors():
    assert {"a1_slide10_aggressor", "soak", "laser_sweep_emulation", "connector_remate"} <= set(scenarios.available())
    assert scenarios.resolve("A1") is library.a1_slide10_aggressor and scenarios.resolve("b3") is library.xpu_power_step
    assert scenarios.resolve("mrm.scenarios.library:soak") is library.soak
    with pytest.raises(scenarios.ScenarioError, match="did you mean"):
        scenarios.resolve("soack")
    with pytest.raises(scenarios.ScenarioError, match="period_ss"):
        scenarios.create("soak", period_ss=3)
    assert "rectified" in scenarios.describe("a1")
    with pytest.raises(scenarios.ScenarioError, match="parabolic"):
        scenarios.create("aggressor_direct", waveform="parabolic")
    d = scenarios.create("aggressor_direct", site="h", waveform="square", power_w=0.2, period_s=1e-5, duty=0.5)
    _, fr = _render(d, n_ticks=10)
    assert _signal(fr, "thermal.aggressor_w").tolist() == [0.2] * 5 + [0.0] * 5


@pytest.mark.filterwarnings("ignore:.*no K/s equivalent")
def test_from_config_with_the_params_machinery():
    from mrm.params import config
    cfg = config.load("tests/data/mini.yaml", overrides={
        "scenario.stack": ["soak", "laser_drift"], "scenario.params.soak.n_cycles": 2,
        "scenario.params.laser_drift.mode": "random_walk", "scenario.time_compression": 50,
        "scenario.ambient_c": 30.0, "scenario.bias_offset_v": 0.01})
    sc = scenarios.from_config(cfg)
    assert sc.time_compression == 50.0 and sc.duration_s == pytest.approx(2 * library.SOAK_CYCLE_S)
    assert sc.defaults["thermal.t_base_k"] == pytest.approx(303.15) and "tx.bias" in sc.offsets
    arr = sc.compile(dt_s=1e-3, n_rings=cfg.n_rings, n_lines=cfg.n_lines)
    assert arr.compression == 50.0
    cfg2 = config.load("tests/data/mini.yaml", overrides={"scenario.aggressor.site": "agg0",
                                                          "scenario.target.units": "temperature",
                                                          "scenario.target.trajectory": "a2",
                                                          "scenario.target.reading": "one_sided_sawtooth"})
    sc2 = scenarios.from_config(cfg2)
    assert sc2.pending_aggressors == ("agg0",) and sc2.aggressors[0].target.units == "temperature"
    assert scenarios.from_config(
        {"scenario": {"name": "a1", "params": {"period_s": 0.13}}}).aggressors[0].target.period_s == 0.13
    assert scenarios.from_config({}).name == "static"
    with pytest.raises(scenarios.ScenarioError):
        scenarios.from_config({"scenario": {"name": "a1", "stack": ["soak"]}})


def test_stack_combines_and_detects_conflicts():
    sc = scenarios.create("a1") + scenarios.create("laser_drift") + scenarios.create("b1")
    assert sc.pending_aggressors == ("aggressor",) and len(sc.components) == 3    # A1's base is a default
    assert sc.defaults["thermal.t_base_k"] == pytest.approx(298.15)
    clash = scenarios.create("a4") + scenarios.create("b5")          # two absolute base temperatures
    with pytest.raises(scenarios.ScenarioError, match="absolutely by both"):
        clash.compile(dt_s=1.0, n_rings=1, n_lines=1)
    with pytest.raises(scenarios.ScenarioError, match="defined by two"):
        scenarios.create("a1") + scenarios.create("a2")
    assert scenarios.create("a1").fingerprint() != scenarios.create("a1", period_s=0.13).fingerprint()
    assert scenarios.create("a4").fingerprint(include_compression=False) == \
        scenarios.create("a4").with_compression(10).fingerprint(include_compression=False)


# ----------------------------------------------------------------------------------------------------- Review additions

def test_unlabeled_duplicates_are_detected():
    a = generators.const("thermal.t_base_k", 300.0, absolute=True)
    b = generators.const("thermal.t_base_k", 310.0, absolute=True)
    with pytest.raises(scenarios.ScenarioError, match="absolutely by both"):
        scenarios.Scenario("x", components=(a, b)).compile(dt_s=DT, n_rings=1, n_lines=1)
    r1 = generators.random_levels("data.ones_density", interval_s=1e-5, stream="scenario_dup")
    r2 = generators.random_levels("data.ones_density", interval_s=1e-5, stream="scenario_dup")
    with pytest.raises(scenarios.ScenarioError, match="identical draws"):
        scenarios.Scenario("y", components=(r1, r2)).compile(dt_s=DT, n_rings=1, n_lines=1)


def test_scen_eval_b_skips_constant_signals_after_init():
    arr = (library.xpu_power_step() + library.squelch()).compile(dt_s=1e-4, n_rings=2, n_lines=2)
    assert set(arr.active_signals) == {"thermal.xpu_rise_k", "data.swing_scale"}
    out = arr.new_out()[0]
    assert np.array_equal(out, arr.elem_default, equal_nan=True)
    cf, tb = arr.row(0)
    out[arr.layout["laser.line_enable"][0]] = -7.0                  # a constant signal is never rewritten
    generators.scen_eval_b(np.int64(500), arr.compression, arr.sig_i64, arr.sig_f64, arr.elem_default, arr.comp_i64, cf,
                           tb, arr.keys[0], arr.new_cache()[0], out)
    assert out[arr.layout["laser.line_enable"][0]] == -7.0
    o, _ = arr.layout["thermal.xpu_rise_k"]
    assert out[o] == pytest.approx(arr.evaluate(500)["thermal.xpu_rise_k"][0, 0])


def test_frame_replay_holds_first_sample_and_strictest_compression_reference():
    fr = generators.DisturbanceFrame(dt_s=1.0, tick0=10, stride_ticks=5,
                                     signals={"thermal.t_base_k": [300.0, 310.0, 320.0]})
    _, out = _render(scenarios.Scenario.from_frame(fr), ticks=np.array([0, 10, 12, 20, 40]), dt=1.0,
                     n_rings=1, n_lines=1)
    assert np.allclose(_signal(out, "thermal.t_base_k"), [300.0, 300.0, 304.0, 320.0, 320.0])
    loose = scenarios.Scenario("l", fast_rate_ref_k_per_s=1e5)
    assert generators.stack([library.soak(), loose]).fast_rate_ref_k_per_s == pytest.approx(
        generators.FAST_RATE_REF_K_PER_S)
    sc = library.soak() + library.a2_808_aggressor(reading="one_sided_sawtooth")
    # the strictest reference, not the 80,800 K/s return edge
    assert sc.fast_rate_ref_k_per_s == pytest.approx(generators.FAST_RATE_REF_K_PER_S)


def test_ring_t0_changes_the_rise_by_the_second_order_term_only():
    tgt = library.a1_slide10_aggressor().aggressors[0].target
    t = np.linspace(0, tgt.period_s, 200, endpoint=False)
    kw = dict(s0_m_per_k=70e-12, beta_per_k=1.4e-3, t_ref_k=298.15)
    cold = tgt.ring_rise_k(t, **kw)
    hot = tgt.ring_rise_k(t, ring_t0_k=398.15, **kw)
    assert np.all(hot <= cold + 1e-12)                                      # s(T) is larger when hot
    assert np.max(cold) / np.max(hot) == pytest.approx((1 + 1.4e-3 * 100 + 1.4e-3 * np.max(hot) / 2)
                                                       / (1 + 1.4e-3 * np.max(cold) / 2), rel=1e-9)
    cw = calibrate.calibrate_aggressor_power([FOSTER, FOSTER], tgt, s0_m_per_k=70e-12, beta_per_k=1.4e-3,
                                             t_ref_k=[298.15, 298.15], ring_t0_k=[298.15, 398.15], n_knots=128)
    assert cw.peak_power_w[1] < cw.peak_power_w[0] and cw.record()["model"]["ring_t0_k"] == [298.15, 398.15]


# ------------------------------------------------------------------ Adversarial review additions (bugs fixed in review)

def _assert_linear_spans(arr, fr, name, k_end, tol):
    """Between consecutive next_break ticks (from tick 0 to k_end) the rendered signal is linear within tol."""
    x = fr.signals[name][:, 0, 0]
    k = 0
    while k < k_end:
        kb = arr.next_break(name, k)
        assert kb > k
        kb = min(kb, x.size - 1)
        lin = np.interp(np.arange(k, kb + 1), [k, kb], [x[k], x[kb]])
        assert np.max(np.abs(x[k:kb + 1] - lin)) <= tol, (name, k, kb)
        k = kb


def test_next_break_periodic_sampled_with_negative_phase():
    """A periodic table has knots at negative local time too; the break must not jump to x = 0."""
    c = generators.sampled("thermal.xpu_rise_k", 1e-5, [0.0, 1.0, 4.0, 9.0], periodic=True, phase_s=-1e-4, label="a")
    arr = scenarios.Scenario("s", components=(c,)).compile(dt_s=DT, n_rings=1, n_lines=1)
    fr = arr.render(n_ticks=200)
    _assert_linear_spans(arr, fr, "thermal.xpu_rise_k", 190, 1e-9)
    assert arr.next_break("thermal.xpu_rise_k", 0) == 9                    # next knot at tick 10, not tick 100


def test_next_break_is_per_tick_where_the_sum_may_clip():
    """A ramp that runs into a clip limit has a kink the parametric breaks cannot see: such a signal is flagged at
    compile (SI_CLIP) and stepped per tick; library scenarios that stay inside their limits are not flagged."""
    c = generators.pwl("laser.line_enable", [0, 1e-4], [0.0, -2.0], label="r")    # 1 + ramp crosses 0 at tick 50
    arr = scenarios.Scenario("s", components=(c,)).compile(dt_s=DT, n_rings=1, n_lines=1)
    assert arr.clipped_signals == ("laser.line_enable",)
    fr = arr.render(n_ticks=120)
    _assert_linear_spans(arr, fr, "laser.line_enable", 110, 1e-12)
    resp = calibrate.FosterResponse([4.5e-4], [500.0])
    a1, _ = scenarios.calibrate_scenario(scenarios.create("a1"), {"aggressor": resp}, model={"s0_m_per_k": 70e-12},
                                         n_knots=64)
    for sc in (a1, scenarios.create("a6", dark_s=0.1), scenarios.create("b2", gap_s=1e-3),
               scenarios.create("b5", laser_turn_on_s=1e-3), scenarios.create("traffic", t_stop_s=1.0),
               scenarios.create("b1"), scenarios.create("a4"), scenarios.create("ones_density_steps"),
               scenarios.create("ones_density_steps", mode="random"), scenarios.create("training")):
        assert sc.compile(dt_s=1e-4, n_rings=2, n_lines=2).clipped_signals == (), sc.name
    # traffic + squelch: swing 0 - 1 is clipped to 0 while traffic is off, so the swing is flagged (and exact)
    both = (scenarios.create("traffic", t_start_s=1e-5) + scenarios.create("b1", t_on_s=2e-5, t_off_s=3e-5))
    arr = both.compile(dt_s=DT, n_rings=1, n_lines=1)
    assert arr.clipped_signals == ("data.swing_scale",)
    x = arr.render(n_ticks=40).signals["data.swing_scale"][:, 0, 0]
    assert np.all(x[:10] == 0) and np.all(x[10:20] == 1) and np.all(x[20:30] == 0) and np.all(x[30:] == 1)


def test_smooth_kinds_meet_foh_tol_with_map_and_several_components():
    """CF_HSMOOTH accounts for the bernoulli_td map (2 r (1 - r) curves more than r) and for several smooth
    components sharing one signal's foh_tol."""
    tol_td = generators.SIGNAL_BY_NAME["data.transition_density"].foh_tol
    c = generators.rect_sine("data.transition_density", 2e-3, amp=1.0, map="bernoulli_td", absolute=True, label="m")
    arr = scenarios.Scenario("m", components=(c,)).compile(dt_s=DT, n_rings=1, n_lines=1)
    assert arr.clipped_signals == ()
    fr = arr.render(n_ticks=4000)
    _assert_linear_spans(arr, fr, "data.transition_density", 3900, 1.0001 * tol_td)
    tol_x = generators.SIGNAL_BY_NAME["thermal.xpu_rise_k"].foh_tol
    cs = (generators.sine("thermal.xpu_rise_k", 3e-3, amp=2.0, label="s1"),
          generators.sine("thermal.xpu_rise_k", 3e-3, amp=2.0, label="s2"))
    arr = scenarios.Scenario("two", components=cs).compile(dt_s=DT, n_rings=1, n_lines=1)
    fr = arr.render(n_ticks=6000)
    _assert_linear_spans(arr, fr, "thermal.xpu_rise_k", 5900, 1.0001 * tol_x)


def test_additive_component_on_undriven_bias_is_an_error():
    with pytest.raises(scenarios.ScenarioError, match="NaN"):
        scenarios.Scenario("b", components=(generators.const("tx.bias_v", 0.1, label="x"),)).compile(
            dt_s=DT, n_rings=2, n_lines=1)
    ok = scenarios.Scenario("b", components=(generators.const("tx.bias_v", 0.1, label="x"),),
                            defaults={"tx.bias_v": 2.0})
    assert ok.render(dt_s=DT, n_ticks=1, n_rings=2).signals["tx.bias_v"][0, 0].tolist() == [2.1, 2.1]
    assert ok.drives == ("tx.bias",)


def test_a1_and_warm_start_stack_with_a_soak():
    """A1's 25 degC base and warm_start's ambient are signal defaults: a stacked soak (absolute) replaces them."""
    resp = calibrate.FosterResponse([4.5e-4], [500.0])
    sc, _ = scenarios.calibrate_scenario(
        scenarios.create("a1") + scenarios.create("a4", time_compression=100) + scenarios.create("warm_start"),
        {"aggressor": resp}, model={"s0_m_per_k": 70e-12}, n_knots=64)
    fr = sc.render(dt_s=1e-3, ticks=np.array([0, 3045]), n_rings=1, n_lines=1)          # 3.045 s x 100 = 304.5 s
    assert fr.signals["thermal.t_base_k"][:, 0, 0] - 273.15 == pytest.approx([33.0, 33.0 + 75.0 * 304.5 / 609.0])
    assert sc.reset.start == "warm"
    alone = scenarios.create("a1", set_base=True)
    assert alone.defaults["thermal.t_base_k"] == pytest.approx(298.15)
    with pytest.raises(scenarios.ScenarioError, match="disagree"):
        scenarios.create("a1") + scenarios.create("static", ambient_c=40.0)


def test_reloma_base_temperature_is_continuous():
    rp = scenarios.create("a7", sweep_s=1e-4)
    _, fr = _render(rp, n_ticks=800, n_rings=1)
    t = _signal(fr, "thermal.t_base_k") - 273.15
    assert np.max(np.abs(np.diff(t))) <= 105.0 / 100 + 1e-9                  # never a 105 K jump between biases
    assert t[100] == pytest.approx(105.0) and t[200] == pytest.approx(0.0)   # up for bias 0, down for bias 1
    assert t.min() >= -1e-9 and t.max() <= 105.0 + 1e-9


def test_training_pattern_densities():
    for bits, p, td in (("10", 0.5, 1.0), ("1100", 0.5, 0.5), ("1111100000", 0.5, 0.2), ("1110", 0.75, 0.5)):
        sc = scenarios.create("training", bits=bits, t0_s=1e-5, duration_s=1e-5)
        _, fr = _render(sc, n_ticks=30)
        o, d = _signal(fr, "data.ones_density"), _signal(fr, "data.transition_density")
        assert o[15] == pytest.approx(p) and d[15] == pytest.approx(td)
        assert o[5] == 0.5 and d[5] == 0.5 and o[25] == 0.5 and d[25] == 0.5
    with pytest.raises(scenarios.ScenarioError):
        scenarios.create("training", bits="10a")


def test_frame_replays_aggressor_power_and_keeps_the_callers_dict():
    d = scenarios.create("aggressor_direct", site="h", waveform="triangle", power_w=0.2, period_s=4e-5)
    arr = d.compile(dt_s=DT, n_rings=1, n_lines=1, aggressor_sites=["g", "h"])
    fr = arr.render(n_ticks=100)
    assert fr.aggressor_sites == ("g", "h")
    rep = scenarios.Scenario.from_frame(fr)
    fr2 = rep.render(dt_s=DT, n_ticks=100, aggressor_sites=["g", "h"])
    assert np.allclose(fr2.signals["thermal.aggressor_w"], fr.signals["thermal.aggressor_w"], atol=1e-15)
    raw = {"thermal.t_base_k": [300.0, 310.0]}
    generators.DisturbanceFrame(dt_s=1.0, tick0=0, stride_ticks=1, signals=raw)
    assert isinstance(raw["thermal.t_base_k"], list)                          # not replaced by a (T, 1, 1) array
    bad = generators.DisturbanceFrame(dt_s=1.0, tick0=0, stride_ticks=1,
                                      signals={"thermal.aggressor_w": np.ones((3, 1, 2))})
    with pytest.raises(scenarios.ScenarioError, match="aggressor_sites"):
        bad.to_components()


@pytest.mark.filterwarnings("ignore:.*no K/s equivalent")
def test_from_config_applies_the_compression_rule_at_load():
    with pytest.raises(scenarios.CompressionError, match="time_compression rule"):
        scenarios.from_config({"scenario": {"name": "soak", "time_compression": 1000}})
    assert scenarios.from_config({"scenario": {"name": "soak", "time_compression": 100}}).time_compression == 100.0
    with pytest.raises(scenarios.ScenarioError, match="period"):
        generators.pwl("thermal.xpu_rise_k", [0.0, 2e-3], [0.0, 1.0], periodic=True, period_s=1e-3)


def test_calibrate_scenario_per_site_model_and_offset_warning():
    resp = calibrate.FosterResponse([4.5e-4], [500.0])
    two = scenarios.Scenario("two", aggressors=(
        generators.AggressorSpec("a", victim_ring=0, target=library.a1_slide10_aggressor().aggressors[0].target),
        generators.AggressorSpec("b", victim_ring=1, target=library.a1_slide10_aggressor().aggressors[0].target)))
    _, cal = scenarios.calibrate_scenario(two, {"a": resp, "b": resp}, n_knots=64,
                                          model={"a": {"s0_m_per_k": 70e-12}, "b": {"s0_m_per_k": 85.86e-12}})
    assert cal["a"].peak_power_w[0] == pytest.approx(cal["b"].peak_power_w[0] * 85.86 / 70.0, rel=1e-3)
    with pytest.raises(scenarios.ScenarioError, match="per site"):
        scenarios.calibrate_scenario(two, {"a": resp, "b": resp}, model={"a": {"s0_m_per_k": 70e-12}})
    with pytest.warns(RuntimeWarning, match="not applied"):
        scenarios.calibrate_scenario(scenarios.create("a1"), {"aggressor": resp}, model={"s0_m_per_k": 70e-12},
                                     n_knots=64, fit_offset=True)


def test_a2_ramp_per_period_reading():
    """The thermal notes, disturbance row "On-die heater aggressor next to one MRM" (verified note): the printed
    'period' may be one ramp, so both ramps run at 808 degC/s for 100 ms."""
    tgt = scenarios.create("a2", reading="ramp_per_period").aggressors[0].target
    assert tgt.period_s == pytest.approx(0.2) and tgt.delta_k == pytest.approx(80.8)
    assert tgt.peak_rate_k_per_s() == pytest.approx(808.0) and tgt.mean_rate_k_per_s() == pytest.approx(808.0, rel=1e-3)
    assert tgt.printed_k(0.1) - 273.15 == pytest.approx(105.8)


def test_mapped_ramps_are_curved_for_next_break():
    """bernoulli_td turns a linear ramp r(t) into the parabola 2 r (1 - r): next_break must not treat it as linear."""
    tol = generators.SIGNAL_BY_NAME["data.transition_density"].foh_tol
    for c in (generators.trapezoid("data.transition_density", 2e-3, 0.5e-3, 0.3e-3, 0.7e-3, map="bernoulli_td",
                                   absolute=True, label="t"),
              generators.pwl("data.transition_density", [0, 1e-3, 3e-3], [0, 1, 0.2], map="bernoulli_td", absolute=True,
                             label="p"),
              generators.sampled("data.transition_density", 0.4e-3, [0, 0.3, 1.0, 0.6], periodic=True,
                                 map="bernoulli_td", absolute=True, label="s")):
        arr = scenarios.Scenario("m", components=(c,)).compile(dt_s=DT, n_rings=1, n_lines=1)
        fr = arr.render(n_ticks=5000)
        _assert_linear_spans(arr, fr, "data.transition_density", 4900, 1.0001 * tol)
    with pytest.raises(scenarios.ScenarioError, match="unbounded"):
        generators.random_walk("data.transition_density", knot_s=1e-3, sigma_per_rt_s=1.0, map="bernoulli_td")


def test_calibration_with_mixed_sign_residues_matches_exact_foh():
    """A non-collocated path (mixed-sign residues, a delayed rise; docs/ARCHITECTURE.md, "5.3 Thermal ROM") is
    calibrated and reproduced by an exact FOH simulation of the rendered power to the reported residual."""
    # two-stage cascade (zero initial slope, delayed rise): r = A tau1/(tau1 - tau2), -A tau2/(tau1 - tau2); + a tail
    A, t1, t2 = 200.0, 2e-3, 3e-4
    resp = calibrate.FosterResponse([t2, t1, 2e-2], [-A * t2 / (t1 - t2), A * t1 / (t1 - t2), 30.0])
    g = resp.step(np.geomspace(1e-7, 1.0, 400))
    assert np.all(g >= -1e-12) and np.all(np.diff(g) >= -1e-12)        # a physical (monotone) delayed rise
    sc, cal = scenarios.calibrate_scenario(scenarios.create("a1"), {"aggressor": resp},
                                           model={"s0_m_per_k": 70e-12, "beta_per_k": 1.4e-3, "t_ref_k": 298.15},
                                           n_knots=256)
    cw = cal["aggressor"]
    P = cw.period_s
    arr = sc.compile(dt_s=1e-5, n_rings=1, n_lines=1)
    fr = arr.render(n_ticks=int(round(8 * P / 1e-5)))
    r, _ = calibrate.simulate_foster_foh(resp, fr.signals["thermal.aggressor_w"][:, 0, 0], 1e-5)
    t = fr.times_s()
    last = t >= 7 * P - 1e-9                                         # slowest pole 20 ms: settled after 7 periods
    y = sc.aggressors[0].target.ring_rise_k(t[last], s0_m_per_k=70e-12, beta_per_k=1.4e-3, t_ref_k=298.15)
    assert np.sqrt(np.mean((r[last] - y) ** 2)) < 1.2 * cw.residual_rms_k[0] + 1e-3
    assert np.all(cw.power_w >= 0)
