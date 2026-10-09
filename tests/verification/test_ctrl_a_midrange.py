"""Dual-actuator mid-ranging (mrm/control/baselines/midranging.py): toy ring where the bias tunes the detuning, and the
g26 nominal plant (CTRL_A_METRIC lines; run with -s)."""
from __future__ import annotations

import numpy as np
import pytest

import test_ctrl_a_harness as H
from mrm.control import registry
from mrm.control.baselines.midranging import M_RUN, MidRanging
from mrm.control.baselines.pi_lock import ST_LOCK


def report(tag, m: H.LockMetrics, **extra):
    s = m.summary()
    s.update(extra)
    print("CTRL_A_METRIC", tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})


def test_toy_bias_holds_fast_heater_recentres():
    """Toy ring, bias tunes the detuning (0.5 heater codes per bias code): after MAP and the bias identification the
    inner loop holds y = f M_pk, a fast disturbance is taken by the bias, and the heater drives the bias back to
    mid-range."""
    spec = H.toy_spec(1, 2, bias=True)
    u0 = np.array([[30000.0, 31000.0]])
    toy = H.ToyRings(spec, u0=u0, g=np.array([[0.5, 2.0]]), W=450.0, noise=1.0, bias_gain=0.5,
                     u_init=u0[0] + 450.0, bias_init=873, x0=3.0)
    c = MidRanging(init_codes=u0[0] + 450.0, id_steps=20, id_bias_codes=200.0)
    c.reset(spec)
    offs = []

    def kick(t, k):
        if k == 1500:
            t.u0 += 150.0                         # a fast blue step of the line (heater-code equivalent)
        offs.append(c.b - c.b_mid if hasattr(c, "b") else 0)
    toy.run(c, 4000, on_step=kick)
    tel = c.telemetry()
    assert np.all(tel["state"] == ST_LOCK) and np.all(tel["mid_state"] == M_RUN)
    assert np.all(tel["bias_slope_per_code"] < 0.0)          # bias up -> red -> monitor down on the stable side
    np.testing.assert_allclose(toy.x, np.sqrt(1 / 0.7 - 1), atol=0.03)
    offs = np.array(offs)[:, 0]
    assert np.max(np.abs(offs[1500:1520])) > 50.0                # the bias took the step first ...
    assert np.all(np.abs(offs[-50:]) < 15.0)                     # ... and was re-centred by the heater


def test_g26_midranging_steady():
    """g26 nominal, bias window 1.0-2.2 V around 1.6 V: MAP + PI settle, bias identification, mid-ranging. The bias
    response is loss-dominated there (s_b > 0 on the stable side: a higher reverse bias lowers the FCA loss and
    raises the drop power more than its ~0.1-0.5 pm shift lowers it), so the inner loop holds the monitor partly
    through the loss; measured here."""
    cfg, pp, spec, plant = H.placed(bias_v=1.6)
    c = MidRanging()
    loop = H.ClosedLoop(plant, spec, c)
    rec = loop.run(7000)
    tel = c.telemetry()
    assert np.all(tel["state"] == ST_LOCK) and np.all(tel["mid_state"] == M_RUN)
    m = H.lock_metrics(rec, slice(-1500, -1000), slice(-1000, None))
    assert np.all(m.max_err_pm < 3.0), m.max_err_pm
    report("midranging g26 nominal steady (1.6 V mid)", m, bias_slope_per_code=float(np.mean(tel["bias_slope_per_code"])),
           bias_offset_codes=float(np.mean(np.abs(tel["bias_offset_codes"]))),
           lock_delta_pm=float(np.mean(np.median(rec["delta_pm"][-500:], axis=0))))


def test_checks():
    with pytest.raises(ValueError, match="bind.tx.bias"):
        MidRanging().reset(H.toy_spec(1, 2))
    assert isinstance(registry.create("midranging", {"outer_gain": 0.01}), MidRanging)


def test_toy_fall_in_run_resettles_the_pi_before_midranging():
    """A kick past the fold while mid-ranging runs (bistable toy ring, g = 4): the fall is detected, the heater
    recaptures, and the heater PI settles at the set-point again (bias held at mid-range) before the inner bias loop
    takes over; it used to hand straight back to the bias at the end of the recapture (just red of the peak, far from
    the set-point, which the small bias range of g26 cannot bridge)."""
    spec = H.toy_spec(1, 2, bias=True)
    u0 = np.array([[30000.0, 31000.0]])
    toy = H.ToyRings(spec, u0=u0, g=np.array([[4.0, 4.0]]), W=450.0, noise=1.0, bias_gain=0.5,
                     u_init=u0[0] + 1400.0, bias_init=873, x0=3.0)
    c = MidRanging(init_codes=u0[0] + 1400.0, id_steps=20, id_bias_codes=200.0)
    c.reset(spec)
    log = []

    def kick(t, k):
        if k == 2000:
            t.u0[0, 0] += 900.0
        log.append((c.rings[0][0].st, float(c.b[0, 0] - c.b_mid), float(c.rings[0][0].lock_cnt)))
    toy.run(c, 4000, on_step=kick)
    tel = c.telemetry()
    assert tel["fall_count"][0, 0] == 1 and tel["fall_count"][0, 1] == 0
    assert np.all(tel["state"] == ST_LOCK) and np.all(tel["mid_state"] == M_RUN)
    np.testing.assert_allclose(toy.x, np.sqrt(1 / 0.7 - 1), atol=0.03)
    # after the fall: the bias stays at mid-range until the PI has held the set-point for lock_steps steps
    after = log[2000:]
    fell = next(i for i, (st, _, _) in enumerate(after) if st != ST_LOCK)
    first_lock = next(i for i, (st, _, _) in enumerate(after) if i > fell and st == ST_LOCK)
    held = [bo for st, bo, lc in after[first_lock:first_lock + c.params.lock_steps]]
    assert np.all(np.abs(held) < 1e-9), held


def test_toy_locked_telemetry_follows_the_bias_loop():
    """07 P1-20: 'locked' must report the loop that runs. In RUN the PI is bypassed, and its lock count used to stay
    frozen at the hand-off: a red step larger than the bias range (the bias saturates and the slow heater loop has
    to catch up) kept reporting lock while the monitor sat far from the set-point. Now the count follows the
    mid-ranging loop."""
    spec = H.toy_spec(1, 1, bias=True)
    toy = H.ToyRings(spec, u0=[[30000.0]], g=0.5, W=450.0, noise=1.0, bias_gain=0.5, u_init=[30450], bias_init=873,
                     x0=3.0)
    c = MidRanging(init_codes=[30450], id_steps=20, id_bias_codes=200.0)
    c.reset(spec)
    seen = []

    def kick(t, k):
        if 1500 <= k < 1600:
            t.u0 -= 15.0                  # red move of the line, faster than the heater re-centring follows
        seen.append((int(c.mstate[0, 0]), bool(c.telemetry()["locked"][0, 0])))
    toy.run(c, 3500, on_step=kick)
    assert all(ms == M_RUN for ms, _ in seen[1400:])
    assert seen[1499][1]                                   # locked before the step
    assert not all(lk for _, lk in seen[1502:1800])        # the bias saturated: not locked for a while
    assert seen[-1][1] and c.telemetry()["fall_count"][0, 0] == 0
    np.testing.assert_allclose(toy.x, np.sqrt(1 / 0.7 - 1), atol=0.03)


def test_toy_bias_saturation_falls_back_to_the_heater_pi():
    """04 C13 failure mode "limited bias range": a red move faster than the heater re-centring saturates the bias.
    Without the fallback the heater moves only k_o |s_b/s_h| b_range per step (the g26 A1 run lost lock: 470 pm);
    with it the heater runs the PI on the error while the bias sits at a rail (g26 A1 rate: 9.7 pm max)."""
    spec = H.toy_spec(1, 1, bias=True)
    err = {}
    for fb in (False, True):
        toy = H.ToyRings(spec, u0=[[30000.0]], g=0.5, W=450.0, noise=1.0, bias_gain=0.5, u_init=[30450],
                         bias_init=873, x0=3.0)
        c = MidRanging(init_codes=[30450], id_steps=20, id_bias_codes=200.0, saturation_fallback=fb)
        c.reset(spec)
        xs = []

        def kick(t, k, xs=xs):
            if 1500 <= k < 1600:
                t.u0 -= 15.0
            xs.append(float(t.x[0, 0]))
        toy.run(c, 3500, on_step=kick)
        assert c.telemetry()["fall_count"][0, 0] == 0
        err[fb] = float(np.abs(np.array(xs[1500:]) - np.sqrt(1 / 0.7 - 1)).max())
    assert err[True] < 0.1 and err[False] > 4 * err[True], err
