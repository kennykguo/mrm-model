"""Heater-dither lock-in (mrm/control/baselines/dither_lockin.py): closed loop on the ReferencePlant (through
test_ctrl_a_harness), parity of the compiled port, peak and inflection modes on a toy bistable ring.

Truth is read by the TEST only. CTRL_A_METRIC lines are the reported numbers (run with -s).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import test_ctrl_a_harness as H
from mrm.control import registry
from mrm.control.api import Observation
from mrm.control.baselines import dither_lockin as dl
from mrm.control.baselines.dither_lockin import DitherLockin, DitherLockinCompiled


def report(tag, m: H.LockMetrics, **extra):
    s = m.summary()
    s.update(extra)
    print("CTRL_A_METRIC", tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})


def cycle_mean(x, period_ticks: int):
    """Mean over whole dither cycles (removes the dither from a Truth trace)."""
    n = (x.shape[0] // period_ticks) * period_ticks
    return x[:n].reshape((n // period_ticks, period_ticks) + x.shape[1:]).mean(axis=1)


@pytest.fixture(scope="module")
def inflection():
    cfg, pp, spec, plant = H.placed()
    ctrl = DitherLockin()
    loop = H.ClosedLoop(plant, spec, ctrl)
    seen = []
    orig = ctrl.step

    def step(obs):
        out = orig(obs)
        if len(seen) < 300:
            seen.append((obs.tick, {k: np.array(v) for k, v in obs.sensors.items()},
                         {k: np.array(v) for k, v in obs.new.items()}, out["tx.heater"].copy()))
        return out
    ctrl.step = step
    rec = loop.run(9000)
    ctrl.step = orig
    return dict(cfg=cfg, pp=pp, spec=spec, plant=plant, ctrl=ctrl, rec=rec, seen=seen)


def test_inflection_lock_converges_on_the_red_side(inflection):
    rec, ctrl, spec = inflection["rec"], inflection["ctrl"], inflection["spec"]
    tel = ctrl.telemetry()
    assert np.all(tel["state"] == dl.D_LOCK)
    p = ctrl.params
    cyc = 4 * p.phase_steps * spec.period_ticks                        # ticks per dither cycle (240)
    d = rec["delta_pm"].astype(float)
    dc = cycle_mean(d, cyc)
    final = dc[-8:].mean(axis=0)
    # the 2f zero is the inflection of the (NRZ-broadened) drop line: red of the line, the same on every ring
    assert np.all((final > 100.0) & (final < 165.0)), final
    assert np.ptp(final) < 10.0, final
    assert np.all(np.abs(tel["eps"]) < 0.3), tel["eps"]
    assert np.all(tel["s1"] < 0.0)
    swing = (d[-cyc * 4:].max(axis=0) - d[-cyc * 4:].min(axis=0)) / 2.0
    m = H.lock_metrics(rec, slice(-3000, -1500), slice(-1500, None))
    report("dither_lockin inflection steady g26 nominal", m, lock_delta_pm=float(final.mean()),
           dither_swing_pm=float(swing.mean()), cycle_mean_drift_pm=float(np.abs(dc[-8:] - final).max()),
           time_to_lock_us=float(np.nanmax(H.time_to_lock_us({"delta_pm": dc}, dt_us=cyc))))


def test_compiled_replay_is_bit_identical(inflection):
    spec = inflection["spec"]
    cc = DitherLockinCompiled()
    cc.reset(spec)
    for tick, sens, new, want in inflection["seen"]:
        got = cc.step(Observation(tick, spec.tick_s, sens, new))["tx.heater"]
        assert np.array_equal(got, want), (tick, got, want)


@pytest.mark.parametrize("mode", ["peak", "inflection"])
def test_toy_parity_and_lock_points(mode):
    """Python and compiled agree code for code on a toy bistable ring (with a fold-crossing kick); the 1f zero is
    the peak (x = 0) and the 2f zero the inflection x = 1/sqrt(3) on the red side."""
    spec = H.toy_spec(2, 3)
    u0 = np.array([[30000, 30500, 31000], [29800, 30200, 30900]], float)
    g = np.array([[1.0, 2.0, 0.5], [3.0, 0.8, 1.5]]) if mode == "inflection" else \
        np.array([[0.5, 0.8, 1.0], [0.3, 0.6, 1.2]])
    init = u0 + (500 if mode == "inflection" else 150)
    kw = dict(mode=mode, dither_codes=40.0 if mode == "inflection" else 12.0, phase_steps=4, blank_steps=1,
              max_step_codes=60.0, init_codes=init[0])

    def disturb(toy, k):
        toy.valid[:] = not (300 <= k < 303)
        if 900 <= k < 910:
            toy.u0 += 60.0
    runs = []
    for ctrl in (DitherLockin(**kw), DitherLockinCompiled(**kw)):
        toy = H.ToyRings(spec, u0=u0, g=g, u_init=init[0], x0=3.0, noise=1.0)
        ctrl.reset(spec)
        h, _ = toy.run(ctrl, 2000, on_step=disturb)
        runs.append((h, ctrl.telemetry(), toy.x.copy()))
    (ha, ta, xa), (hb, tb, xb) = runs
    assert np.array_equal(ha, hb)
    np.testing.assert_array_equal(ta["state"], tb["state"])
    assert np.all(ta["state"] == dl.D_LOCK), ta["state"]
    # x at the end of a cycle's centre phase is the centre detuning (the dither returns to the centre)
    want = 0.0 if mode == "peak" else 1.0 / math.sqrt(3.0)
    tol = 0.08 if mode == "peak" else 0.08
    np.testing.assert_allclose(xa, want, atol=tol)


def test_peak_mode_on_g26_sits_next_to_the_fold():
    """Peak lock (1f -> 0) on g26 nominal: the fold is ~9 pm blue of the peak (bistability ratio ~6). Measured: how
    close the lock gets, and how often the ring falls off and is recaptured (07 P0-1, 04 C5)."""
    cfg, pp, spec, plant = H.placed(delta_pm=60.0, n_ticks=2500)
    ctrl = DitherLockin(mode="peak", dither_codes=16.0, max_step_codes=48.0)
    loop = H.ClosedLoop(plant, spec, ctrl)
    rec = loop.run(6000)
    tel = ctrl.telemetry()
    assert np.all(tel["state"] != dl.D_FAULT)
    d = rec["delta_pm"].astype(float)
    cyc = 4 * ctrl.params.phase_steps * spec.period_ticks
    dc = cycle_mean(d, cyc)
    falls = tel["fall_count"]
    near = np.abs(dc[-5:].mean(axis=0))
    # every ring either sits within 25 pm of the peak or has fallen off and been recaptured
    assert np.all((near < 25.0) | (falls > 0)), (near, falls)
    m = H.lock_metrics(rec, slice(-2400, -1200), slice(-1200, None))
    report("dither_lockin peak g26 nominal", m, centre_abs_detuning_pm=float(near.mean()),
           min_detuning_pm=float(d.min()), falls=float(falls.sum()))


def test_registry_and_checks():
    assert registry.resolve("dither_lockin") is DitherLockin
    assert isinstance(registry.create("dither_lockin_c", {"mode": "peak"}), DitherLockinCompiled)
    with pytest.raises(ValueError):
        DitherLockin(mode="third")
    with pytest.raises(ValueError):
        DitherLockin(phase_steps=3, blank_steps=3)


def test_recapture_after_an_early_fall_waits_for_the_red_side():
    """A fall before the first complete dither cycle (no DC reference yet). The recapture level used to be
    recapture_level x 0 then, so any 2 % dip of a noisy monitor past the guard ended the recapture on the COLD branch
    (measured with 200-code noise: LOCK at x = -3.8). The reference is now the pre-fall level: the recapture ends red of
    the peak. Python and compiled identical."""
    spec = H.toy_spec(1, 1)
    u0 = np.array([[30000.0]])
    init = u0[0] + 500
    kw = dict(mode="inflection", dither_codes=40.0, phase_steps=4, blank_steps=1, max_step_codes=60.0,
              init_codes=init)
    runs = []
    for cls in (DitherLockin, DitherLockinCompiled):
        c = cls(**kw)
        toy = H.ToyRings(spec, u0=u0, g=3.0, u_init=init, x0=3.0, noise=200.0)
        c.reset(spec)
        trans, prev = [], [dl.D_LOCK]

        def hook(t, k, c=c, trans=trans, prev=prev):
            if k == 2:
                t.u0 += 3000.0                         # far past the fold: the ring falls to the cold branch
            st = int(c.telemetry()["state"][0, 0])
            if prev[0] == dl.D_RECAPTURE and st == dl.D_LOCK:
                trans.append(float(t.x[0, 0]))
            prev[0] = st
        h, _ = toy.run(c, 1500, on_step=hook)
        runs.append((h, trans, c.telemetry()))
    (ha, ta, tela), (hb, tb, telb) = runs
    assert np.array_equal(ha, hb)
    assert tela["fall_count"][0, 0] >= 1 and len(ta) >= 1
    assert all(x > 0.0 for x in ta), ta                   # every recapture ended red of the peak
    assert tela["state"][0, 0] == dl.D_LOCK
