"""Checks the rules of the start states: the settle, the batch rule, the reachable order and the data state.

Read this first
---------------
Each group of tests pins one rule of docs/ENGINE.md, "Start states", that an engine review found broken:

1. ``Plant.settle`` iterates at the scenario inputs of the reset tick. If every DC iteration advanced the scenario
   clock by one tick, an input that changes inside the first ~20 ticks (an aggressor switched on at 10 us) would be
   settled as if it were already on: ring 0 would start 18.7 K hot and +1554 pm red instead of at its lock
   (+121 pm), and the settle would still report converged.
2. Each world stops its own settle, so element b of a batch equals the B = 1 run with seed b. If the batch iterated
   until its slowest world converged, the extra iterations (each with the noise of a new tick) would move every
   other world's settled state by up to ~1 mK.
3. ``order='reachable'`` hops until no ring needs a negative heater power. Hopping a ring adds about one FSR of
   heater power, and its crosstalk can push rings with a valid heater-up code below zero. A single re-solve would
   leave them without a code (g26 seeds 1000, B = 8: rings [0, 1], [0, 4], [1, 5]).
4. The warm-start codes use the data path of the reset tick (driver enable, swing scale, transition and ones
   density). Codes that always assume traffic on would start a scenario with traffic off (traffic_start_stop)
   65-98 pm blue of the lock (the ~1-1.4 K of data heat would be missing).

~1 min with a warm numba cache.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mrm import params
from mrm.control.api import ControllerSpec
from mrm.plant.plant import Plant
from mrm.scenarios import library
from mrm.sim import kernel
from mrm.sim import runner
from mrm.sim.metrics import RecordSpec

# the thermal state arrays and the ring_float rows that the settle writes (compared world by world)
_SETTLE_STATE_ARRAYS = ("th_x", "th_t_obs", "th_g", "th_u0", "th_u1")
_SETTLE_RING_ROWS = [kernel.RF_P_PREV, kernel.RF_T_RING, kernel.RF_T_HEATER, kernel.RF_T_PD, kernel.RF_T_NEW,
                     kernel.RF_P_NEW, kernel.RF_HE_P, kernel.RF_P_ABS]


@pytest.fixture(scope="module")
def cfg():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load("g26")


def _run(cfg, scenario, **kw):
    """A 2 us open-loop nominal run from the settled locked steady state, recording T_ring and the detuning every
    tick. Keyword arguments override the run options."""
    rec = [RecordSpec("t_ring_k", 1, "last"), RecordSpec("detuning_m", 1, "last")]
    opts = dict(duration_s=2e-6, uncertainty="nominal", variation="off", heater_code="locked_steady_state", settle=True,
                record=rec)
    opts.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return runner.run(cfg, None, scenario, **opts)


# ----------------------------------------------------------------------- the settle uses the scenario at the reset tick

def test_settle_holds_the_scenario_inputs_of_the_reset_tick(cfg):
    """The settled start depends only on the scenario at the reset tick: an aggressor that switches on at 10 us
    settles exactly like one that switches on at 5 ms (both off at t = 0), and one that switches off at 10 us exactly
    like one that stays on for 5 ms (both on at t = 0)."""
    def agg(t0_s, period_s):
        return library.aggressor_direct(power_w=0.1, period_s=period_s, duty=0.5, t0_s=t0_s)
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


# -------------------------------------------------------------------------------------- each world stops its own settle

def _nominal_plant(cfg, B):
    """A Plant of B nominal worlds (no variation) with the config's scenario and noise seed 11, not reset."""
    c, pp, _ = runner.load_inputs(cfg, batch=B, uncertainty="nominal", variation="off")
    sc = runner._make_scenario(None, c)
    spec = ControllerSpec.from_config(c, n_batch=B, nominal=c, scenario_drives=sc.drives, scenario_offsets=sc.offsets)
    return Plant(pp, sc, spec, noise_seed=11)


def test_a_settled_world_does_not_depend_on_its_batch(cfg):
    """The batch rule for the settled start: world 0 of a B = 2 batch whose other world needs more settle iterations
    (heater codes 2000 lower: 22 against 18 iterations) equals the same world settled alone (noise seed base + 0),
    bit for bit in every thermal state the settle keeps."""
    one = _nominal_plant(cfg, 1)
    one.reset(protocol="branch_high")
    codes = runner.warm_start_codes(one, cfg)
    one.reset(protocol="branch_high", heater_code=codes, settle=True)
    two = _nominal_plant(cfg, 2)
    two.reset(protocol="branch_high", heater_code=np.stack([codes[0], codes[0] - 2000]), settle=True)
    for name in _SETTLE_STATE_ARRAYS:
        np.testing.assert_array_equal(getattr(two.S, name)[0], getattr(one.S, name)[0], err_msg=name)
    np.testing.assert_array_equal(two.S.ring_float[0][_SETTLE_RING_ROWS], one.S.ring_float[0][_SETTLE_RING_ROWS])
    i1, i2 = one.settle_info, two.settle_info
    assert i1["converged"] and i2["converged"], (i1, i2)
    it1, it2 = list(i1["iterations_per_world"]), list(i2["iterations_per_world"])
    assert it2[0] == it1[0] and it2[1] > it2[0], (it1, it2)   # the worlds stop at their own iteration
    assert i2["iterations"] == max(it2) and list(i2["converged_per_world"]) == [True, True]
    # and the next ticks of the plant are the same plant
    one.run(50, mode=kernel.MODE_OPEN)
    two.run(50, mode=kernel.MODE_OPEN)
    np.testing.assert_array_equal(two.S.ring_float[0, kernel.RF_T_RING], one.S.ring_float[0, kernel.RF_T_RING])


# ------------------------------------------------------------------ the reachable order hops until no power is negative

def test_reachable_order_hops_until_no_ring_needs_negative_power(cfg):
    """g26 seeds 1000, B = 8, sampled: the heater-up order leaves 10 rings without a code (they need a negative heater
    power). Hopping them adds ~1 FSR of heater power each, whose crosstalk pushed three rings that had heater-up codes
    (world 0 rings 1 and 4, world 1 ring 5: p_h 1.29, 3.00, 0.67 mW) below zero; a single re-solve would leave them
    at code_min. They are hopped too, and so is world 0 ring 3, which their own heat pushes below zero in turn
    (world 0 hops 5, 6, then 1, 4, then 3; world 1 hops 1, 6, then 5), so every ring of the reachable solve has a
    non-negative heater power."""
    kw = dict(ambient_c=[25.0], bias_v=[1.05], batch=8, seeds=1000, uncertainty="sample", variation="sample")
    up = runner.solve_locked_steady_state(cfg, **kw)
    reachable = runner.solve_locked_steady_state(cfg, order="reachable", **kw)
    pushed = [(0, 1), (0, 3), (0, 4), (1, 5)]
    for b, n in pushed:
        assert np.isfinite(up.heater_code[0, 0, b, n]) and up.p_heater_w[0, 0, b, n] > 0.0   # heater-up: a code
        assert reachable.order_hopped[0, 0, b, n] and not reachable.fsr_hop_risk[0, 0, b, n], (b, n)
        assert np.isfinite(reachable.heater_code[0, 0, b, n]), (b, n)
    assert not np.any(reachable.p_heater_w < 0.0), np.argwhere(reachable.p_heater_w < 0.0)
    hop = reachable.order_hopped
    assert np.all(hop[up.fsr_hop_risk])                            # every negative-power ring of heater-up hops
    assert np.all(np.isfinite(reachable.heater_code))              # every ring of this batch gets a code
    # the extra heat of the hopped rings exceeds the array supply in worlds 0 and 1 (droop 0.916 / 0.968; world 2,
    # 0.933, was supply-limited after one re-solve already): no ring of those worlds can hold its lock there, which
    # the warm start flags (supply_limited)
    assert list(np.nonzero(reachable.supply_limited[0, 0])[0]) == [0, 1, 2], reachable.supply_scale[0, 0]
    assert not up.supply_limited.any()


def test_runner_reachable_warm_start_leaves_no_ring_at_code_min(cfg):
    kw = dict(duration_s=1e-6, batch=8, seeds=1000, uncertainty="sample", variation="sample", settle=False,
              reset_protocol="branch_high")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = runner.run(cfg, None, None, heater_code="locked_steady_state", **kw)
        b = runner.run(cfg, None, None, heater_code="locked_steady_state_reachable", **kw)
    wa, wb = a.manifest_inputs["warm_start"], b.manifest_inputs["warm_start"]
    na, nb = np.asarray(wa["no_code"]), np.asarray(wb["no_code"])
    hop = np.asarray(wb["order_hopped"])
    assert na.sum() == 10 and nb.sum() == 0, (na.sum(), nb.sum())
    assert np.all(hop[na])                                         # every ring without a heater-up code hopped
    secondary = np.argwhere(hop & ~na).tolist()                    # rings that had a heater-up code and hopped
    assert secondary == [[0, 1], [0, 3], [0, 4], [1, 5]], secondary
    assert wb["supply_limited"] == [True, True, True] + [False] * 5, wb["supply_limited"]


# ---------------------------------------------------------------- warm-start codes use the data state of the reset tick

def _detuning_pm(res):
    """The recorded detuning of world 0 in pm, shape (records, rings)."""
    return np.asarray(res.trace["detuning_m@100:last"]["value"])[:, 0, :] * 1e12


def test_warm_start_codes_take_the_data_state_of_the_reset_tick(cfg):
    """A scenario that starts with traffic off (driver disabled, no swing; traffic from 10 ms) gets codes that hold
    the same traffic-on operating detuning as the static scenario: the settled open-loop detuning agrees with the
    static run within 2 pm on every ring (measured 0.2-1.0 pm, the codes 1243-1348 higher; traffic-on codes would be
    65-98 pm blue, the data heat missing). A start at ones density 0 (Sun et al. steps from t = 0): within 6 pm
    (measured 0.5 pm at the first ring of the bus to 4.6 pm at the last; traffic-on codes: 13-24 pm). The residual
    grows along the bus: solve_locked_steady_state does not model the upstream rings' filtering of the other lines,
    which the ones density changes (docs/ENGINE.md, "Start states")."""
    rec = [RecordSpec("detuning_m", 100, "last")]
    kw = dict(duration_s=3e-3, record=rec)
    st = _run(cfg, library.static(), **kw)
    tr = _run(cfg, library.traffic_start_stop(t_start_s=0.01), **kw)
    od = _run(cfg, library.ones_density_steps(), **kw)
    d0 = _detuning_pm(st)
    for res, bound in ((tr, 2.0), (od, 6.0)):
        dev = np.max(np.abs(_detuning_pm(res) - d0))
        assert dev < bound, (res.manifest_inputs["scenario"], dev, _detuning_pm(res)[0], d0[0])
    c0 = np.asarray(st.manifest_inputs["warm_start"]["heater_code"])
    c1 = np.asarray(tr.manifest_inputs["warm_start"]["heater_code"])
    assert np.all(c1 - c0 > 500), c1 - c0                     # traffic off: ~1-1.4 K more heater heat


def test_solve_locked_steady_state_data_state_default_is_the_traffic_on_state(cfg):
    """solve_locked_steady_state(data_state=...) with the traffic-on state (driver on, nominal swing, the config's
    transition density, ones density 0.5) gives the survey without it bit for bit; traffic off keeps the traffic-on
    target and needs more heater power."""
    kw = dict(ambient_c=[25.0], bias_v=[1.05], uncertainty="nominal", variation="off")
    base = runner.solve_locked_steady_state(cfg, **kw)
    on = runner.solve_locked_steady_state(cfg, data_state={"driver_enable": 1.0, "swing_scale": 1.0,
                                                           "transition_density": 0.5, "ones_density": 0.5}, **kw)
    for name in ("heater_code", "p_heater_w", "t_ring_k", "p_abs_w", "detuning_target_m", "reloma_db",
                 "fsr_hop_risk", "wrong_line_risk", "supply_scale"):
        np.testing.assert_array_equal(np.asarray(getattr(on, name)), np.asarray(getattr(base, name)), err_msg=name)
    off = runner.solve_locked_steady_state(cfg, data_state={"driver_enable": 0.0, "swing_scale": 0.0,
                                                            "transition_density": 0.0, "ones_density": 0.5}, **kw)
    np.testing.assert_array_equal(off.detuning_target_m, base.detuning_target_m)
    np.testing.assert_array_equal(off.reloma_db, base.reloma_db)
    assert np.all(off.p_heater_w > base.p_heater_w + 0.1e-3)
    with pytest.raises(ValueError, match="data_state"):
        runner.solve_locked_steady_state(cfg, data_state={"swing": 1.0}, **kw)


def test_solve_locked_steady_state_notes_name_the_worlds_in_the_given_data_state(cfg):
    """The notes say which worlds hold their lock in the given data_state (not always the traffic-on state); the
    traffic-on state keeps the plain note."""
    kw = dict(ambient_c=[25.0], bias_v=[1.05], uncertainty="nominal", variation="off")
    res = runner.solve_locked_steady_state(cfg, data_state={"driver_enable": [[1.0], [0.0]]}, batch=2, **kw)
    (note,) = res.notes
    assert "of worlds [1] are those of the given data_state" in note
    (plain,) = runner.solve_locked_steady_state(cfg, **kw).notes
    assert "given data_state" not in plain and "data averaged with ones density 0.5" in plain


def test_pi_lock_holds_every_ring_through_a_traffic_start_from_the_warm_start(cfg):
    """Traffic start from the documented warm start (static + traffic_start_stop at 10 ms), pi_lock_c, 20 ms: no
    ring falls off its hot branch during the controller's start-up, and every ring ends at its operating point
    (measured: lowest detuning -1.1 pm during pi_lock's peak map, final 115-118 pm). With traffic-on codes, rings 4-7
    would start 1-14 pm red of their lines and fall to the cold branch during the start-up (down to -1228 to
    -1294 pm; the review measured ring 6 still at -761 pm at 20 ms)."""
    rec = [RecordSpec("detuning_m", 100, "mean")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = runner.run(cfg, "pi_lock_c", library.static() + library.traffic_start_stop(t_start_s=0.01),
                       duration_s=20e-3, uncertainty="nominal", variation="off", heater_code="locked_steady_state",
                       settle=True, record=rec)
    d = np.asarray(r.trace["detuning_m@100:mean"]["value"])[:, 0, :] * 1e12
    assert np.all(d.min(axis=0) > -50.0), d.min(axis=0)
    assert np.all((d[-1] > 100.0) & (d[-1] < 130.0)), d[-1]
