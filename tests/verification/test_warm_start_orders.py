"""Quasi-static warm start: the array heater supply limit and the reachable resonance order (integrator, 2026-10-10;
docs/handoff/experiments.md section 10.4, ENGINE.md section 3).

1. mrm.sim.runner.quasi_static checks the bank's lock powers against the array heater supply limit (07 P1-12,
   mrm.plant.electrical.supply_limit_row). In the g26 stacked_eic world (P_FSR 36.9 mW) the lock currents exceed the
   assumed 150 mA supply below ~30 degC: the engine droops every heater and the rings settle on the cold branch,
   800-1100 pm blue of their lines. This was the "quasi_static vs settle disagreement" of the experiments handoff:
   the engine was right, quasi_static did not check the limit.
2. ``order='reachable'`` / ``heater_code='quasi_static_reachable'``: a ring whose heater-up order needs a negative
   heater power locks one order bluer instead of starting at code_min; worlds without such a ring are unchanged bit
   for bit, and the default mode is unchanged. Since the engine review of 2026-10-16 the hop is repeated: a ring that
   the hopped rings' extra heater power pushes below zero hops too (tests/verification/test_engine_start_review.py).
~10 s with a warm numba cache.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from mrm import params
from mrm.sim import runner


def _cfg(**alts):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load("g26", alts=alts or None)


def test_supply_limit_is_checked_in_quasi_static():
    amb = [0.0, 25.0, 60.0, 105.0]
    nom = runner.quasi_static(_cfg(), ambient_c=amb, bias_v=[1.0, 1.05, 3.75], uncertainty="nominal",
                              variation="off")
    assert not nom.supply_limited.any() and np.all(nom.supply_scale == 1.0)
    assert nom.code_range_coverage[0] == 1.0
    eic = runner.quasi_static(_cfg(**{"uncertainty.thermal.stack": "stacked_eic"}), ambient_c=amb,
                              bias_v=[1.05], uncertainty="nominal", variation="off")
    s = eic.supply_scale[:, 0, 0]
    assert s[0] < s[1] < 1.0 and s[2] == s[3] == 1.0            # limited at 0 and 25 degC, free at 60 and 105
    assert 0.95 < s[1] < 0.99                                      # 25 degC: ~0.969 (current limit)
    assert np.all(np.isfinite(eic.heater_code[1, 0, 0]))           # the codes are kept ...
    assert not eic.code_ok[1, 0, 0].any() and eic.code_ok[2, 0, 0].all()   # ... but no ring is reachable there
    assert eic.code_range_coverage[0] == pytest.approx(0.5)


def test_supply_limited_warm_start_warns_and_the_engine_droops():
    """stacked_eic at 25 degC: the warm start is flagged, and the settled engine really sits below the lock (the
    plant applies the same limit); side_by_side holds its lock point."""
    cfg = _cfg(**{"uncertainty.thermal.stack": "stacked_eic"})
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        r = runner.run(cfg, None, "static", duration_s=1e-3, uncertainty="nominal", variation="off",
                       heater_code="quasi_static", settle=True)
    assert any("supply limit" in str(w.message) for w in wl)
    assert r.manifest_inputs["warm_start"]["supply_limited"] == [True]
    det = np.asarray(r.truth_final.fields["detuning_m"])[0] * 1e12
    assert np.all(det < -500.0)                                    # cold branch, blue of the lines
    r0 = runner.run(_cfg(), None, "static", duration_s=1e-3, uncertainty="nominal", variation="off",
                    heater_code="quasi_static", settle=True)
    assert r0.manifest_inputs["warm_start"]["supply_limited"] == [False]
    det0 = np.asarray(r0.truth_final.fields["detuning_m"])[0] * 1e12
    assert np.all((det0 > 80.0) & (det0 < 130.0))                  # the hot-branch lock point (ENGINE.md 3)


def test_reachable_order_moves_only_the_rings_that_need_negative_power():
    cfg = _cfg()
    kw = dict(ambient_c=[25.0, 80.0], bias_v=[1.05], batch=8, seeds=1000, uncertainty="sample", variation="sample")
    up = runner.quasi_static(cfg, **kw)
    re = runner.quasi_static(cfg, order="reachable", **kw)
    assert up.order_hopped is not None and not up.order_hopped.any()
    hop = re.order_hopped
    neg = up.fsr_hop_risk                                           # the heater-up order's negative-power rings
    assert hop.any() and np.all(hop[neg])                          # every one of them moves ...
    extra = hop & ~neg                                              # ... and the rings their heat pushes below zero
    assert np.all(extra.any(axis=3) <= neg.any(axis=3))             # (only in worlds with a first hop)
    assert np.isfinite(up.heater_code[extra]).all() and np.all(up.p_heater_w[extra] >= 0.0)
    assert np.array_equal(re.fsr_hop_risk, up.fsr_hop_risk)       # still reported for the heater-up order
    assert np.isnan(up.heater_code[neg]).all()
    assert not np.any(re.p_heater_w < 0.0)                         # no ring left needing a negative power
    assert np.isfinite(re.heater_code[hop]).mean() > 0.6           # most of them get a code one order bluer
    assert np.all(re.p_heater_w[hop] > 0.0)
    moved = hop.any(axis=3)                                         # (A, V, B) worlds with a hopped ring
    for name in ("heater_code", "p_heater_w", "t_ring_k", "detuning_target_m", "reloma_db", "wrong_line_risk",
                 "supply_scale"):
        a, b = np.asarray(getattr(up, name)), np.asarray(getattr(re, name))
        assert np.array_equal(a[~moved], b[~moved], equal_nan=True), name   # other worlds bit for bit
    # a hopped ring's heater-up sweep from code_min crosses about one FSR of lines more
    assert np.median(re.wrong_line_risk[hop]) > np.median(re.wrong_line_risk[~hop])
    with pytest.raises(ValueError, match="order"):
        runner.quasi_static(cfg, ambient_c=[25.0], bias_v=[1.05], order="nearest")


def test_runner_warm_start_reachable_mode():
    cfg = _cfg()
    kw = dict(duration_s=2e-4, batch=8, seeds=1000, uncertainty="sample", variation="sample", settle=False,
              reset_protocol="branch_high")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = runner.run(cfg, None, "static", heater_code="quasi_static", **kw)
        b = runner.run(cfg, None, "static", heater_code="quasi_static_reachable", **kw)
    wa, wb = a.manifest_inputs["warm_start"], b.manifest_inputs["warm_start"]
    assert wa["mode"] == "quasi_static" and wb["mode"] == "quasi_static_reachable"
    na, nb = np.asarray(wa["no_code"]), np.asarray(wb["no_code"])
    hop = np.asarray(wb["order_hopped"])
    assert nb.sum() < na.sum() and not np.asarray(wa["order_hopped"]).any()
    assert np.all(hop[na & ~nb])                                   # a ring that gained a code hopped
    assert not np.any(hop & ~na & nb)                              # a secondary hop (it had a code) keeps one
    ca, cb = np.asarray(wa["heater_code"]), np.asarray(wb["heater_code"])
    rows = ~hop.any(axis=1)
    assert np.array_equal(ca[rows], cb[rows])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        codes, no_code = runner.warm_start_codes(b.plant, cfg, mode="quasi_static", return_no_code=True)
    assert codes.shape == (8, 8) and no_code.dtype == bool
    with pytest.raises(ValueError, match="heater_code mode"):
        runner.warm_start_codes(b.plant, cfg, mode="nearest")
