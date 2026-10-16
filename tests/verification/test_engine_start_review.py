"""Engine review fixes of the start states (2026-10-16; docs/CHANGELOG.md "engine review fixes"; ENGINE.md section 3).

1. engine-1: ``Plant.settle`` iterates at the scenario inputs of the reset tick. Before, every DC iteration advanced
   the scenario clock by one tick, so an input that changed inside the first ~20 ticks (an aggressor switched on at
   10 us) was settled as if it were already on: ring 0 started 18.7 K hot and +1554 pm red instead of at its lock
   (+121 pm), and the settle still reported converged.
2. engine-5: each world stops its own settle (07 P1-13: element b of a batch equals the B = 1 run with seed b).
   Before, the batch iterated until its slowest world converged, and the extra iterations (each with the noise of a
   new tick) moved every other world's settled state by up to ~1 mK.
3. engine-2: ``order='reachable'`` hops until no ring needs a negative heater power. Hopping a ring adds about one
   FSR of heater power, whose crosstalk pushed rings with a valid heater-up code below zero; before, the single
   re-solve left them without a code (g26 seeds 1000, B = 8: rings [0, 1], [0, 4], [1, 5]).
4. engine-3: the warm-start codes use the data path of the reset tick (driver enable, swing scale, transition and
   ones density). Before, they always assumed traffic on: a scenario that starts with traffic off
   (traffic_start_stop) began 65-98 pm blue of the lock (the ~1-1.4 K of data heat was missing).
~1 min with a warm numba cache.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mrm import params
from mrm.control.api import ControllerSpec
from mrm.plant.plant import Plant
from mrm.scenarios import library as lib
from mrm.sim import kernel as K
from mrm.sim import runner
from mrm.sim.metrics import RecordSpec

_KEPT = ("th_x", "th_t_obs", "th_g", "th_u0", "th_u1")
_ROWS = [K.RF_P_PREV, K.RF_T_RING, K.RF_T_HEATER, K.RF_T_PD, K.RF_T_NEW, K.RF_P_NEW, K.RF_HE_P, K.RF_P_ABS]


@pytest.fixture(scope="module")
def cfg():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load("g26")


def _run(cfg, scenario, **kw):
    rec = [RecordSpec("t_ring_k", 1, "last"), RecordSpec("detuning_m", 1, "last")]
    opts = dict(duration_s=2e-6, uncertainty="nominal", variation="off", heater_code="quasi_static", settle=True,
                record=rec)
    opts.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return runner.run(cfg, None, scenario, **opts)


# ------------------------------------------------------------------------------------------------ engine-1
def test_settle_holds_the_scenario_inputs_of_the_reset_tick(cfg):
    """The settled start depends only on the scenario at the reset tick: an aggressor that switches on at 10 us
    settles exactly like one that switches on at 5 ms (both off at t = 0), and one that switches off at 10 us exactly
    like one that stays on for 5 ms (both on at t = 0)."""
    def agg(t0_s, period_s):
        return lib.aggressor_direct(power_w=0.1, period_s=period_s, duty=0.5, t0_s=t0_s)
    off_early, off_late = _run(cfg, agg(10e-6, 0.02)), _run(cfg, agg(5e-3, 0.02))
    on_early, on_late = _run(cfg, agg(0.0, 20e-6)), _run(cfg, agg(0.0, 10e-3))
    for a, b in ((off_early, off_late), (on_early, on_late)):
        for key in a.trace:
            np.testing.assert_array_equal(a.trace[key]["value"], b.trace[key]["value"], err_msg=key)
        sa, sb = a.manifest_inputs["warm_start"]["settle"], b.manifest_inputs["warm_start"]["settle"]
        assert sa["converged"] and sa["iterations"] == sb["iterations"], (sa, sb)
    # the aggressor at t = 0 is what the settle sees: 0.1 W, 30 um from ring 0, heats it by ~19 K
    t_off = off_early.trace["t_ring_k@1:last"]["value"][0, 0, 0]
    t_on = on_early.trace["t_ring_k@1:last"]["value"][0, 0, 0]
    assert 10.0 < t_on - t_off < 30.0, (t_off, t_on)
    d = off_early.trace["detuning_m@1:last"]["value"][0, 0] * 1e12
    assert np.all((d > 80.0) & (d < 140.0)), d                 # ring 0 at its lock point (+121 pm), not +1554 pm


# ------------------------------------------------------------------------------------------------ engine-5
def _nominal_plant(cfg, B):
    c, pp, _ = runner.load_inputs(cfg, batch=B, uncertainty="nominal", variation="off")
    sc = runner._make_scenario(None, c)
    spec = ControllerSpec.from_config(c, n_batch=B, nominal=c, scenario_drives=sc.drives, scenario_offsets=sc.offsets)
    return Plant(pp, sc, spec, noise_seed=11)


def test_a_settled_world_does_not_depend_on_its_batch(cfg):
    """07 P1-13 for the settled start: world 0 of a B = 2 batch whose other world needs more settle iterations
    (heater codes 2000 lower: 22 against 18 iterations) equals the same world settled alone (noise seed base + 0),
    bit for bit in every thermal state the settle keeps."""
    one = _nominal_plant(cfg, 1)
    one.reset(protocol="branch_high")
    codes = runner.warm_start_codes(one, cfg)
    one.reset(protocol="branch_high", heater_code=codes, settle=True)
    two = _nominal_plant(cfg, 2)
    two.reset(protocol="branch_high", heater_code=np.stack([codes[0], codes[0] - 2000]), settle=True)
    for name in _KEPT:
        np.testing.assert_array_equal(getattr(two.S, name)[0], getattr(one.S, name)[0], err_msg=name)
    np.testing.assert_array_equal(two.S.rf[0][_ROWS], one.S.rf[0][_ROWS])
    i1, i2 = one.settle_info, two.settle_info
    assert i1["converged"] and i2["converged"], (i1, i2)
    it1, it2 = list(i1["iterations_per_world"]), list(i2["iterations_per_world"])
    assert it2[0] == it1[0] and it2[1] > it2[0], (it1, it2)   # the worlds stop at their own iteration
    assert i2["iterations"] == max(it2) and list(i2["converged_per_world"]) == [True, True]
    # and the next ticks of the plant are the same plant
    one.run(50, mode=K.MODE_OPEN)
    two.run(50, mode=K.MODE_OPEN)
    np.testing.assert_array_equal(two.S.rf[0, K.RF_T_RING], one.S.rf[0, K.RF_T_RING])


# ------------------------------------------------------------------------------------------------ engine-2
def test_reachable_order_hops_until_no_ring_needs_negative_power(cfg):
    """g26 seeds 1000, B = 8, sampled: the heater-up order leaves 10 rings without a code (they need a negative heater
    power). Hopping them adds ~1 FSR of heater power each, whose crosstalk pushed three rings that had heater-up codes
    (world 0 rings 1 and 4, world 1 ring 5: p_h 1.29, 3.00, 0.67 mW) below zero; the single re-solve left them at
    code_min. They are hopped too now, and so is world 0 ring 3, which their own heat pushes below zero in turn
    (world 0 hops 5, 6, then 1, 4, then 3; world 1 hops 1, 6, then 5), so every ring of the reachable solve has a
    non-negative heater power."""
    kw = dict(ambient_c=[25.0], bias_v=[1.05], batch=8, seeds=1000, uncertainty="sample", variation="sample")
    up = runner.quasi_static(cfg, **kw)
    re = runner.quasi_static(cfg, order="reachable", **kw)
    pushed = [(0, 1), (0, 3), (0, 4), (1, 5)]
    for b, n in pushed:
        assert np.isfinite(up.heater_code[0, 0, b, n]) and up.p_heater_w[0, 0, b, n] > 0.0   # heater-up: a code
        assert re.order_hopped[0, 0, b, n] and not re.fsr_hop_risk[0, 0, b, n], (b, n)
        assert np.isfinite(re.heater_code[0, 0, b, n]), (b, n)
    assert not np.any(re.p_heater_w < 0.0), np.argwhere(re.p_heater_w < 0.0)
    hop = re.order_hopped
    assert np.all(hop[up.fsr_hop_risk])                            # every negative-power ring of heater-up hops
    assert np.all(np.isfinite(re.heater_code))                     # every ring of this batch gets a code
    # the extra heat of the hopped rings exceeds the array supply in worlds 0 and 1 (droop 0.916 / 0.968; world 2,
    # 0.933, was supply-limited after one re-solve already): no ring of those worlds can hold its lock there, which
    # the warm start flags (supply_limited)
    assert list(np.nonzero(re.supply_limited[0, 0])[0]) == [0, 1, 2], re.supply_scale[0, 0]
    assert not up.supply_limited.any()


def test_runner_reachable_warm_start_leaves_no_ring_at_code_min(cfg):
    kw = dict(duration_s=1e-6, batch=8, seeds=1000, uncertainty="sample", variation="sample", settle=False,
              reset_protocol="branch_high")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = runner.run(cfg, None, None, heater_code="quasi_static", **kw)
        b = runner.run(cfg, None, None, heater_code="quasi_static_reachable", **kw)
    wa, wb = a.manifest_inputs["warm_start"], b.manifest_inputs["warm_start"]
    na, nb = np.asarray(wa["no_code"]), np.asarray(wb["no_code"])
    hop = np.asarray(wb["order_hopped"])
    assert na.sum() == 10 and nb.sum() == 0, (na.sum(), nb.sum())
    assert np.all(hop[na])                                         # every ring without a heater-up code hopped
    secondary = np.argwhere(hop & ~na).tolist()                    # rings that had a heater-up code and hopped
    assert secondary == [[0, 1], [0, 3], [0, 4], [1, 5]], secondary
    assert wb["supply_limited"] == [True, True, True] + [False] * 5, wb["supply_limited"]


# ------------------------------------------------------------------------------------------------ engine-3
def _det(res):
    return np.asarray(res.trace["detuning_m@100:last"]["value"])[:, 0, :] * 1e12


def test_warm_start_codes_take_the_data_state_of_the_reset_tick(cfg):
    """A scenario that starts with traffic off (driver disabled, no swing; traffic from 10 ms) gets codes that hold
    the same traffic-on operating detuning as the static scenario: the settled open-loop detuning agrees with the
    static run within 2 pm on every ring (measured 0.2-1.0 pm, the codes 1243-1348 higher; before: 65-98 pm blue,
    the data heat missing). A start at ones density 0 (Sun et al. steps from t = 0): within 6 pm (measured 0.5 pm at
    the first ring of the bus to 4.6 pm at the last; before 13-24 pm). The residual grows along the bus: quasi_static
    does not model the upstream rings' filtering of the other lines, which the ones density changes (ENGINE.md 3)."""
    rec = [RecordSpec("detuning_m", 100, "last")]
    kw = dict(duration_s=3e-3, record=rec)
    st = _run(cfg, lib.static(), **kw)
    tr = _run(cfg, lib.traffic_start_stop(t_start_s=0.01), **kw)
    od = _run(cfg, lib.ones_density_steps(), **kw)
    d0 = _det(st)
    for res, bound in ((tr, 2.0), (od, 6.0)):
        dev = np.max(np.abs(_det(res) - d0))
        assert dev < bound, (res.manifest_inputs["scenario"], dev, _det(res)[0], d0[0])
    c0 = np.asarray(st.manifest_inputs["warm_start"]["heater_code"])
    c1 = np.asarray(tr.manifest_inputs["warm_start"]["heater_code"])
    assert np.all(c1 - c0 > 500), c1 - c0                     # traffic off: ~1-1.4 K more heater heat


def test_quasi_static_data_state_default_is_the_traffic_on_state(cfg):
    """quasi_static(data_state=...) with the traffic-on state (driver on, nominal swing, the config's transition
    density, ones density 0.5) gives the survey without it bit for bit; traffic off keeps the traffic-on target and
    needs more heater power."""
    kw = dict(ambient_c=[25.0], bias_v=[1.05], uncertainty="nominal", variation="off")
    base = runner.quasi_static(cfg, **kw)
    on = runner.quasi_static(cfg, data_state={"driver_enable": 1.0, "swing_scale": 1.0, "transition_density": 0.5,
                                              "ones_density": 0.5}, **kw)
    for name in ("heater_code", "p_heater_w", "t_ring_k", "p_abs_w", "detuning_target_m", "reloma_db",
                 "fsr_hop_risk", "wrong_line_risk", "supply_scale"):
        np.testing.assert_array_equal(np.asarray(getattr(on, name)), np.asarray(getattr(base, name)), err_msg=name)
    off = runner.quasi_static(cfg, data_state={"driver_enable": 0.0, "swing_scale": 0.0, "transition_density": 0.0,
                                               "ones_density": 0.5}, **kw)
    np.testing.assert_array_equal(off.detuning_target_m, base.detuning_target_m)
    np.testing.assert_array_equal(off.reloma_db, base.reloma_db)
    assert np.all(off.p_heater_w > base.p_heater_w + 0.1e-3)
    with pytest.raises(ValueError, match="data_state"):
        runner.quasi_static(cfg, data_state={"swing": 1.0}, **kw)


def test_pi_lock_holds_every_ring_through_a_traffic_start_from_the_warm_start(cfg):
    """07 P0-11 traffic start from the documented warm start (static + traffic_start_stop at 10 ms), pi_lock_c, 20 ms:
    no ring falls off its hot branch during the controller's start-up and every ring ends at its operating point
    (measured: lowest detuning -1.1 pm during pi_lock's peak map, final 115-118 pm). With the former traffic-on codes
    rings 4-7 start 1-14 pm red of their lines and fall to the cold branch during the start-up (this test: down to
    -1228 to -1294 pm; the review measured ring 6 still at -761 pm at 20 ms)."""
    rec = [RecordSpec("detuning_m", 100, "mean")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = runner.run(cfg, "pi_lock_c", lib.static() + lib.traffic_start_stop(t_start_s=0.01), duration_s=20e-3,
                       uncertainty="nominal", variation="off", heater_code="quasi_static", settle=True, record=rec)
    d = np.asarray(r.trace["detuning_m@100:mean"]["value"])[:, 0, :] * 1e12
    assert np.all(d.min(axis=0) > -50.0), d.min(axis=0)
    assert np.all((d[-1] > 100.0) & (d[-1] < 130.0)), d[-1]
