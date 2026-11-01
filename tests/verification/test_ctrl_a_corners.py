"""Checks pi_lock on the named corners of the uncertainty set (slow tests).

Read this first
---------------
A controller is judged on the nominal result and on the worst named corner of the uncertainty set. On every corner
preset the test runs MAP and lock from the +170 pm hand-off. It adds one more corner, latency_x2: the plant converter
latencies doubled (uncertainty.timing.latency_scale high) with one extra controller period of pipeline. Then it runs
the slide-10 aggressor at its printed peak rate (2127 degC/s, 20 K swing, victim ring 0) with the type-2 PI on the
hardest corners. Only the test reads Truth; the CTRL_A_METRIC lines report the numbers (run with -s -m slow).

Measured: every corner locks. On g26_hot_selfheat (+8 dBm at the ring) the drop peak is at ~2.5x the ADC full scale,
so the PI locks at f x full scale (DECLIP state; ~275-315 pm red, a large OMA cost). Before a fix, every ring of that
corner ended in FAULT. At the A1 rate with the type-2 PI the victim's maximum error is 4.5-9.8 pm over the corners
(type 1: 6.2-11.3 pm).
"""
from __future__ import annotations

import numpy as np
import pytest

import test_ctrl_a_harness as harness
from mrm.control.baselines.pi_lock import ST_LOCK, STATE_NAMES, PiLock

pytestmark = pytest.mark.slow

LATENCY_X2 = {"uncertainty.timing.latency_scale": "high"}
CORNERS = ["g26_slow_thermal", "g26_fast_thermal", "g26_hot_selfheat", "g26_low_eta", "g26_undercut", "g26_heater_loss",
           "g26_overcoupled", "g26_textbook_thermal", "g26_h86", "latency_x2"]


def report(tag, m: harness.LockMetrics, **extra):
    s = m.summary()
    s.update(extra)
    print("CTRL_A_METRIC", tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})


def _placed(corner):
    if corner == "latency_x2":
        return harness.placed(alts=LATENCY_X2)
    return harness.placed(preset=corner)


def _pi(corner, **kw):
    return PiLock(latency_ticks=10 if corner == "latency_x2" else 0, **kw)


@pytest.mark.parametrize("corner", CORNERS)
def test_map_and_lock_on_the_corner(corner):
    cfg, pp, spec, plant = _placed(corner)
    pi = _pi(corner)
    loop = harness.ClosedLoop(plant, spec, pi)
    rec = loop.run(10000)
    tel = pi.telemetry()
    assert np.all(tel["state"] == ST_LOCK), [STATE_NAMES[int(s)] for s in tel["state"].ravel()]
    assert np.all(tel["fall_count"] == 0)
    m = harness.lock_metrics(rec, slice(-1500, -800), slice(-800, None))
    assert np.all(m.max_err_pm < 0.5), m.max_err_pm
    final = np.median(rec["delta_pm"][-300:].astype(float), axis=0)
    assert np.all(final > 60.0), final                        # red (stable) side
    report(f"pi_lock map+lock corner {corner}", m, lock_delta_min_pm=float(final.min()),
           lock_delta_max_pm=float(final.max()), time_to_lock_us=float(np.nanmax(harness.time_to_lock_us(rec))),
           peak_clipped=int(np.sum(tel["peak_codes"] >= spec.adc.full_scale_codes)),
           min_delta_pm=float(rec["delta_pm"].min()))


@pytest.mark.parametrize("corner", ["g26_slow_thermal", "g26_hot_selfheat", "g26_heater_loss", "latency_x2"])
def test_type2_pi_on_a1_rate_at_the_corner(corner):
    cfg, pp, spec, plant = _placed(corner)
    pi = _pi(corner)
    harness.ClosedLoop(plant, spec, pi).run(10000)
    tel = pi.telemetry()
    assert np.all(tel["state"] == ST_LOCK)
    c = _pi(corner, start="lock", peak_codes=tel["peak_codes"], slope_per_code=tel["slope_per_code"], k2=0.005,
            init_codes=plant.lat_heater.copy())
    loop = harness.ClosedLoop(plant, spec, c)
    loop.run(300)
    t_min = (plant.k_next + 1000) * plant.dt
    sc, period = harness.a1_rate_scenario(cfg, pp, t_min)
    plant.set_scenario(sc)
    rec = loop.run(int((period / 2 + 0.004) / plant.dt) + 1000)
    m = harness.lock_metrics(rec, slice(0, 900), slice(1000, None))
    ct = c.telemetry()
    assert np.all(ct["state"] == ST_LOCK) and np.all(ct["fall_count"] == 0)
    assert m.max_err_pm[0, 0] < 15.0, m.max_err_pm[0, 0]       # victim; measured 4.5-9.8 pm
    assert np.all(m.max_err_pm[0, 1:] < 3.0), m.max_err_pm      # neighbours (crosstalk only)
    report(f"pi_lock_type2 a1_rate corner {corner} (victim ring 0)",
           harness.LockMetrics(*(np.asarray(getattr(m, f))[..., :1] for f in m.__dataclass_fields__)),
           others_max_err_pm=float(m.max_err_pm[0, 1:].max()), heater_min_code=float(rec["heater"][:, 0, 0].min()))
