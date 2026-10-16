"""The fitting machinery of mrm.calibration on problems with a known answer (fast; marker ``calibration``)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from mrm.calibration import core
from mrm.validation import targets_io as tio

pytestmark = pytest.mark.calibration

T = core.calibration_targets()


def _linear_problem(slope: float = 2.0, offset: float = 1.0, lo: float = 0.0, hi: float = 10.0) -> core.Problem:
    """CAL-7 (7.6 +- 0.1 nm) predicted as offset + slope * x: the answer is x = (7.6 - offset) / slope and the 95 %
    profile half-width 1.96 sigma / slope."""
    t7 = T["CAL-7"]
    par = core.Param("x", None, lo, hi, 5.0, prior_source="test")
    return core.Problem(group="test", hypothesis="linear", params=[par], targets=[t7],
                        predict=lambda th: {"CAL-7": offset + slope * th["x"]})


def test_only_the_calibration_role_is_read():
    """07 P0-5: the package reaches no held-out file (the same scan as tests/verification)."""
    assert tio.held_out_reference_violations(tio.CALIBRATION_DIR) == []
    assert all(t.role == "calibration" for t in T.values())


def test_least_squares_and_profile_recover_a_linear_parameter():
    p = _linear_problem()
    fr = core.fit(p)
    assert fr.theta["x"] == pytest.approx(3.3, abs=1e-5)
    assert fr.chi2 < 1e-8
    pr = core.profile(p, fr, "x", n_grid=11)
    half = math.sqrt(core.DCHI2_95) * 0.1 / 2.0
    assert pr.interval[0] == pytest.approx(3.3 - half, abs=2e-3)
    assert pr.interval[1] == pytest.approx(3.3 + half, abs=2e-3)
    assert pr.unique and not pr.open_lo and not pr.open_hi
    assert pr.width_frac == pytest.approx(2 * half / 10.0, rel=1e-2)
    assert pr.half_prior_ok


def test_profile_is_open_when_the_data_do_not_constrain_the_parameter():
    p = _linear_problem(slope=1e-4, offset=7.6)
    fr = core.fit(p)
    pr = core.profile(p, fr, "x", n_grid=7)
    assert pr.open_lo and pr.open_hi and not pr.unique
    assert pr.width_frac == pytest.approx(1.0)


def test_birge_ratio_widens_the_interval_of_a_poor_fit():
    """Two inconsistent data on one parameter: chi2_min / dof > 1 widens the interval by sqrt(birge2)."""
    t4 = T["CAL-4"]
    y = np.asarray(t4.value["y"])
    par = core.Param("s", None, 0.5, 2.0, 1.0)
    pred = lambda th: {"CAL-4": (y * th["s"] + np.where(np.arange(y.size) % 2, 0.3, -0.3)).tolist()}  # noqa: E731
    p = core.Problem(group="test", hypothesis="misfit", params=[par], targets=[t4], predict=pred)
    fr = core.fit(p)
    assert fr.birge2 > 1.0
    pr = core.profile(p, fr, "s", n_grid=9)
    w = pr.interval[1] - pr.interval[0]
    w_raw = pr.interval_raw[1] - pr.interval_raw[0]
    assert w == pytest.approx(w_raw * math.sqrt(fr.birge2), rel=0.05)


def test_interval_target_without_sigma_is_a_flat_likelihood():
    t1 = T["CAL-1"]
    assert np.all(core.target_residuals(t1, 3850.0) == 0.0)
    assert np.all(core.target_residuals(t1, 3500.0) == 0.0)
    r_out = core.target_residuals(t1, 3400.0)
    assert r_out[0] < -10.0                       # 100 below the edge, a 3.5-unit wall per unit of residual


def test_set_and_curve_residuals_use_each_point_sigma():
    t9b = T["CAL-9b"]
    r = core.target_residuals(t9b, {"at_code_maxima": 7700.0, "at_code_minima": 9100.0})
    assert sorted(np.round(np.abs(r), 6).tolist()) == [1.0, 1.0]
    t4 = T["CAL-4"]
    y = np.asarray(t4.value["y"]) + np.asarray(t4.sigma)
    assert np.allclose(core.target_residuals(t4, y.tolist()), 1.0)


def test_log_parameter_and_half_prior_in_log_coordinates():
    par = core.Param("g", None, 0.5, 2.0, 1.0, log=True)
    assert par.from_u(par.to_u(0.8)) == pytest.approx(0.8)
    assert core.half_prior(par, [0.5, 1.0]) == pytest.approx(0.5)
    assert core.half_prior(par, [0.7, 1.4]) == pytest.approx(0.5)


def test_hypothesis_likelihood_ratios():
    a = core.fit(_linear_problem())
    b = core.fit(_linear_problem(slope=2.0, offset=1.0, lo=4.0, hi=10.0))   # constrained away from the optimum
    cmp = core.compare_hypotheses("test", [a, b])
    assert cmp.best == a.hypothesis or a.chi2 <= b.chi2
    row = next(r for r in cmp.rows if r["chi2"] == b.chi2)
    assert row["likelihood_ratio"] == pytest.approx(math.exp(-0.5 * (b.chi2 - a.chi2)))


def test_restrict_holds_parameters():
    p = _linear_problem()
    two = core.Problem(group="t", hypothesis="h", params=p.params + [core.Param("y", None, -1, 1, 0.0)],
                       targets=p.targets, predict=lambda th: {"CAL-7": 1.0 + 2.0 * th["x"] + th["y"]})
    r = core.restrict(two, {"y": 0.5})
    fr = core.fit(r)
    assert set(fr.theta) == {"x"}
    assert fr.theta["x"] == pytest.approx(3.05, abs=1e-5)
