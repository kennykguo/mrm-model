"""Test the fitting machinery of mrm.calibration on small problems with a known answer (fast; marker ``calibration``).

Read this first. Each test builds a core.Problem from one or two real calibration targets and a made-up prediction
(for example a straight line), so the best fit, the profile interval and the chi2 are known in closed form. The
tests cover the least-squares fit, the profile likelihood (and when it is open), the Birge-ratio widening of a poor
fit, the residuals of interval, set and curve targets, log parameters, the half-prior rule, hypothesis comparison,
held parameters, the common-mode covariance and the prediction cache.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from mrm.calibration import core
from mrm.validation import targets_io

pytestmark = pytest.mark.calibration

TARGETS = core.calibration_targets()        # target id -> calibration target (calibration.yaml)


def _linear_problem(slope: float = 2.0, offset: float = 1.0, lo: float = 0.0, hi: float = 10.0) -> core.Problem:
    """CAL-7 (7.6 +- 0.1 nm) predicted as offset + slope * x: the answer is x = (7.6 - offset) / slope and the 95 %
    profile half-width 1.96 sigma / slope."""
    t7 = TARGETS["CAL-7"]
    par = core.Param("x", None, lo, hi, 5.0, prior_source="test")
    return core.Problem(group="test", hypothesis="linear", params=[par], targets=[t7],
                        predict=lambda th: {"CAL-7": offset + slope * th["x"]})


def test_only_the_calibration_role_is_read():
    """The calibration package reads no held-out (validation or bounds) file, and every target it loads has the
    calibration role (the same scan as tests/verification)."""
    assert targets_io.held_out_reference_violations(targets_io.CALIBRATION_DIR) == []
    assert all(t.role == "calibration" for t in TARGETS.values())


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
    t4 = TARGETS["CAL-4"]
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
    t1 = TARGETS["CAL-1"]
    assert np.all(core.target_residuals(t1, 3850.0) == 0.0)
    assert np.all(core.target_residuals(t1, 3500.0) == 0.0)
    r_out = core.target_residuals(t1, 3400.0)
    assert r_out[0] < -10.0                       # 100 below the edge, a 3.5-unit wall per unit of residual


def test_set_and_curve_residuals_use_each_point_sigma():
    t9b = TARGETS["CAL-9b"]
    r = core.target_residuals(t9b, {"at_code_maxima": 7700.0, "at_code_minima": 9100.0})
    assert sorted(np.round(np.abs(r), 6).tolist()) == [1.0, 1.0]
    t4 = TARGETS["CAL-4"]
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


def test_common_mode_covariance_whitening():
    """CAL-4's shifts share one read-off error: the whitened residuals give the correlated chi2 d' C^-1 d, a common
    offset of all points costs much less than the same offset on one point, and without it the residuals are the
    independent ones."""
    t4 = TARGETS["CAL-4"]
    y = np.asarray(t4.value["y"])
    s = np.asarray(t4.sigma)
    c = 0.025
    cov = np.full((y.size, y.size), c ** 2)
    np.fill_diagonal(cov, s ** 2)
    d = np.random.default_rng(1).normal(0.0, 0.05, y.size)
    w = core.target_residuals(t4, (y + d).tolist(), common_sigma=c)
    assert float(w @ w) == pytest.approx(float(d @ np.linalg.solve(cov, d)), rel=1e-10)
    common = core.target_residuals(t4, (y + 0.04).tolist(), common_sigma=c)
    one = core.target_residuals(t4, (y + np.eye(y.size)[5] * 0.04 * math.sqrt(y.size)).tolist(), common_sigma=c)
    assert float(common @ common) < 0.5 * float(one @ one)
    assert np.allclose(core.target_residuals(t4, (y + s).tolist()), 1.0)
    p = core.Problem(group="t", hypothesis="h", params=[core.Param("a", None, -1.0, 1.0, 0.0)], targets=[t4],
                     predict=lambda th: {"CAL-4": (y + th["a"]).tolist()}, common_sigma={"CAL-4": c})
    assert core.restrict(p, {}).common_sigma == {"CAL-4": c}
    assert p.chi2({"a": 0.04}) == pytest.approx(float(common @ common))


def test_half_prior_uses_the_declared_width_when_the_prior_is_widened():
    """A fit prior that widens a leaf's declared range never makes the half-prior rule easier to meet."""
    par = core.Param("r", None, 35.0, 58.0, 45.0, declared_range=(42.0, 58.0))
    assert core.half_prior(par, [39.0, 43.0]) == pytest.approx(4.0 / 16.0)
    assert core.half_prior(core.Param("r", None, 35.0, 58.0, 45.0), [39.0, 43.0]) == pytest.approx(4.0 / 23.0)


def test_prediction_cache_returns_the_new_prediction_when_it_clears(monkeypatch):
    """The prediction cache is cleared when it holds more than _CACHE_MAX_ENTRIES entries. The evaluation that
    fills it past the limit still returns its own prediction (it raised KeyError: the clear removed it)."""
    monkeypatch.setattr(core, "_CACHE_MAX_ENTRIES", 3)
    p = _linear_problem()
    for i in range(10):
        assert p.predictions({"x": float(i)}) == {"CAL-7": 1.0 + 2.0 * i}
    assert p.n_evals == 10
    assert len(p._cache) <= 3
