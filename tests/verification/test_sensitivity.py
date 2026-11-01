"""Checks mrm.validation.sensitivity: the report's own Sobol / Saltelli estimator, the tornado table, the
identifiability helpers and the u-space map of the uncertainty set.

These feed sections 3 (identifiability) and 4 (uncertainty propagation) of the validation report. SALib is not
installed, so the estimator is checked against the analytic Ishigami indices and against scipy.stats.sobol_indices.
The u-space map turns a point of the unit cube into one world of the uncertainty set; the tests check that it
matches what realize draws.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mrm.validation import sensitivity


# ---------------------------------------------------------------------------------------------------------------- Sobol

def _ishigami_run(n, seed=0, n_boot=200):
    """Sobol indices of the Ishigami function from a Saltelli design with n base rows (n_boot bootstrap resamples)."""
    A, B = sensitivity.sobol_design(3, n, seed=seed)
    y = sensitivity.ishigami(sensitivity.saltelli_stack(A, B))
    fA, fB, fAB = sensitivity.split_stack(y, n, 3)
    return sensitivity.sobol_indices(fA, fB, fAB, names=["x1", "x2", "x3"], n_boot=n_boot, seed=seed)


def test_ishigami_analytic_values():
    ref = sensitivity.ishigami_indices()
    # textbook values (Saltelli et al. 2008, a = 7, b = 0.1)
    np.testing.assert_allclose(ref["s1"], [0.3139, 0.4424, 0.0], atol=1e-4)
    np.testing.assert_allclose(ref["st"], [0.5576, 0.4424, 0.2437], atol=1e-4)
    # the variance against a large plain Monte Carlo
    u = np.random.default_rng(1).random((400_000, 3))
    assert abs(np.var(sensitivity.ishigami(u)) - ref["variance"]) / ref["variance"] < 0.01


def test_sobol_estimator_reproduces_ishigami():
    ref = sensitivity.ishigami_indices()
    r = _ishigami_run(2 ** 14)
    np.testing.assert_allclose(r.s1, ref["s1"], atol=0.02)
    np.testing.assert_allclose(r.st, ref["st"], atol=0.02)
    # the bootstrap intervals cover the analytic values (95 %; a loose check over 6 indices)
    cover = [(lo <= v <= hi) for (lo, hi), v in zip(np.vstack([r.s1_ci, r.st_ci]), np.r_[ref["s1"], ref["st"]])]
    assert sum(cover) >= 5


def test_sobol_estimator_converges():
    ref = sensitivity.ishigami_indices()
    errs = []
    for n in (2 ** 8, 2 ** 12):
        r = _ishigami_run(n, n_boot=0)
        errs.append(np.max(np.abs(np.r_[r.s1 - ref["s1"], r.st - ref["st"]])))
    assert errs[1] < errs[0]


def test_sobol_matches_scipy():
    from scipy.stats import sobol_indices, uniform
    rng = np.random.default_rng(7)

    def f(x):                                   # scipy convention: x is (d, n), returns (s, n)
        return sensitivity.ishigami(((x + np.pi) / (2 * np.pi)).T)[None, :]
    res = sobol_indices(func=f, n=2 ** 13, dists=[uniform(loc=-np.pi, scale=2 * np.pi)] * 3, random_state=rng)
    r = _ishigami_run(2 ** 13, n_boot=0)
    np.testing.assert_allclose(r.s1, np.ravel(res.first_order), atol=0.03)
    np.testing.assert_allclose(r.st, np.ravel(res.total_order), atol=0.03)


def test_sobol_indices_edge_cases():
    n, d = 16, 2
    with pytest.raises(ValueError, match="finite"):
        sensitivity.sobol_indices(np.r_[np.inf, np.zeros(n - 1)], np.zeros(n), np.zeros((d, n)))
    r = sensitivity.sobol_indices(np.ones(n), np.ones(n), np.ones((d, n)), n_boot=10)
    assert np.all(np.isnan(r.s1)) and r.notes
    # an additive model: S1 = ST, sum = 1
    A, B = sensitivity.sobol_design(2, 2 ** 12, seed=3)
    y = (lambda X: 2.0 * X[:, 0] + X[:, 1])(sensitivity.saltelli_stack(A, B))
    r = sensitivity.sobol_indices(*sensitivity.split_stack(y, 2 ** 12, 2), n_boot=0)
    np.testing.assert_allclose(r.s1, [0.8, 0.2], atol=0.01)
    np.testing.assert_allclose(r.st, [0.8, 0.2], atol=0.01)


def test_saltelli_stack_layout():
    A = np.arange(6.0).reshape(3, 2)
    B = -A
    X = sensitivity.saltelli_stack(A, B)
    assert X.shape == (12, 2)
    np.testing.assert_array_equal(X[:3], A)
    np.testing.assert_array_equal(X[3:6], B)
    np.testing.assert_array_equal(X[6:9], np.c_[B[:, 0], A[:, 1]])
    np.testing.assert_array_equal(X[9:12], np.c_[A[:, 0], B[:, 1]])


# -------------------------------------------------------------------------------------------------------------- tornado

def test_tornado_and_screening():
    base = {"m": 1.0, "k": 0.0}
    pts = [("a", "low", {"m": 0.5, "k": 0.0}), ("a", "high", {"m": 3.0, "k": 0.0}),
           ("b", "x", {"m": 1.2, "k": np.inf}), ("c", "y", {"m": 1.0, "k": 1.0})]
    t = sensitivity.tornado_table(base, pts, ["m", "k"])
    assert [r["entry"] for r in t["m"]] == ["a", "b", "c"]
    assert t["m"][0]["swing"] == pytest.approx(2.5) and t["m"][0]["lo_label"] == "low"
    assert t["m"][1]["lo_label"] == "nominal"                # the base is always inside the bar
    assert np.isinf(t["k"][0]["swing"]) and t["k"][0]["entry"] == "b"
    assert sensitivity.screen_top(t, 2) == ["a", "b"]


def test_tornado_all_nan_entry_is_undefined_not_zero():
    """An entry whose metric is NaN at every point keeps lo = hi = swing = NaN (it is not 'no change')."""
    t = sensitivity.tornado_table({"m": 2.0}, [("e2", "x", {"m": np.nan}), ("e1", "y", {"m": 3.0})], ["m"])
    nan_row = next(r for r in t["m"] if r["entry"] == "e2")
    assert np.isnan(nan_row["lo"]) and np.isnan(nan_row["hi"]) and np.isnan(nan_row["swing"])
    assert nan_row["lo_label"] == "x"
    assert [r["entry"] for r in t["m"]] == ["e1", "e2"]                  # NaN last
    assert sensitivity.screen_top(t, 1) == ["e1"]


# ------------------------------------------------------------------------------------------------------ identifiability

def test_identifiability_rule_and_null_space():
    # p1 well determined (s = 10), p2 weak (s = 1), p3 only through p1 + p3 (null direction p1 - p3 absent)
    J = np.array([[10.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    r = sensitivity.identifiability(J, ["p1", "p2", "p3"])
    assert r["rank"] == 2
    assert r["half_prior_rule"] == [True, False, False]
    assert r["half_width_95"][0] == pytest.approx(0.196, rel=1e-3)
    assert r["half_width_95"][1] == pytest.approx(1.96, rel=1e-3)
    assert np.isinf(r["half_width_95"][2])
    assert np.isinf(r["condition"])
    assert [d["identified"] for d in r["directions"]] == [True, False, False]
    # a correlated pair: only the sum is identified
    J2 = np.array([[5.0, 5.0], [5.0, 5.0]])
    r2 = sensitivity.identifiability(J2, ["a", "b"])
    assert r2["rank"] == 1 and r2["half_prior_rule"] == [False, False]
    top = r2["directions"][0]["loadings"]
    assert abs(abs(top[0][1]) - 2 ** -0.5) < 1e-9


# ------------------------------------------------------------------------------------------------------ uncertainty set

@pytest.fixture(scope="module")
def g26_space():
    from mrm import params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load("g26")
    return sensitivity.UncertaintySpace(cfg)


def test_forward_map_matches_realize_formulas():
    """sensitivity._interval_x is a copy of realize._interval_x: keep them in step."""
    import importlib
    realize_module = importlib.import_module("mrm.params.realize")
    u = np.linspace(0.001, 0.999, 41)

    class E:                                    # the two fields of an uncertainty entry that _interval_x reads
        def __init__(self, kind, f):
            self.kind, self.fields = kind, f
    cases = [("interval", {"low": 1.0, "high": 3.0}), ("interval", {"low": 0.5, "high": 2.0, "dist": "loguniform"}),
             ("interval", {"low": 0.0, "high": 4.0, "dist": "triangular", "nominal": 1.0}),
             ("dist", {"mean": 2.0, "sigma": 0.3}), ("dist", {"mean": 2.0, "sigma": 0.3, "truncate": [1.5, 2.8]}),
             ("dist", {"mean": 2.0, "sigma": 0.3, "dist": "lognormal", "truncate": [1.0, 3.0]})]
    for kind, f in cases:
        mine = sensitivity._interval_x(f, kind, u)
        theirs = realize_module._interval_x(E("interval" if kind == "interval" else "dist", f), u)
        np.testing.assert_allclose(mine, theirs, rtol=1e-14)
        back = np.array([sensitivity._interval_u(f, kind, float(x)) for x in mine])
        np.testing.assert_allclose(back, u, atol=1e-9)


def test_space_nominal_and_points(g26_space):
    sp = g26_space
    assert sp.d == 25 and sp.names == sorted(sp.names)
    u0 = sp.nominal_u()
    for i, n in enumerate(sp.names):
        ax = sp.axes[n]
        v = ax.value_of_u(u0[i])
        if ax.kind == "hypothesis":
            assert v == ax.nominal
            assert all(c != ax.nominal for c, _ in ax.tornado_points())
            assert all(ax.value_of_u(u) == c for c, u in ax.tornado_points())
        else:
            assert v == pytest.approx(ax.nominal, rel=1e-12)
    with pytest.raises(ValueError, match="zero weight"):
        sp.axes["thermo_optic"].u_of_value("textbook_derived")


def test_injection_reaches_realize(g26_space):
    """The injected design is what realize draws, row by row (nominal, every tornado point, random rows)."""
    sp = g26_space
    u0 = sp.nominal_u()
    rows = [u0]
    for i, n in enumerate(sp.names):
        for _, u in sp.axes[n].tornado_points():
            r = u0.copy()
            r[i] = u
            rows.append(r)
    rows += list(np.random.default_rng(5).random((8, sp.d)))
    U = np.vstack(rows)
    chk = sensitivity.check_injection(sp, U)
    assert chk["ok"], chk["mismatches"]
    # and the nominal row realizes the nominal world: same draws as uncertainty='nominal'
    from mrm import params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nom = params.realize(sp.cfg, 1, seeds=0, uncertainty="nominal", variation="off")
        with sensitivity.injected_lhs(U[:1]):
            inj = params.realize(sp.cfg, 1, seeds=0, uncertainty="lhs", variation="off")
    for n in sp.names:
        # realize clips u to [1e-12, 1 - 1e-12]: a nominal at an interval end (transport delay 0 s) becomes 2e-18 s
        np.testing.assert_allclose(np.asarray(inj.world[n], float), np.asarray(nom.world[n], float), rtol=1e-12,
                                   atol=1e-15)
    # the patch is removed afterwards, and a wrong shape is loud
    from scipy.stats import qmc
    assert qmc.LatinHypercube.__name__ == "LatinHypercube"
    with pytest.raises(RuntimeError, match="rows"):
        with sensitivity.injected_lhs(U[:2]), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params.realize(sp.cfg, 3, seeds=0, uncertainty="lhs", variation="off")


def test_given_data_first_order():
    """The binned S1 against Ishigami (biased low by binning a smooth input) and an exact discrete case."""
    ref = sensitivity.ishigami_indices()
    u = np.random.default_rng(3).random((2 ** 14, 3))
    r = sensitivity.given_data_first_order(u, sensitivity.ishigami(u), n_bins=16, n_boot=50)
    assert np.all(r["s1"] <= ref["s1"] + 0.01)
    np.testing.assert_allclose(r["s1"], ref["s1"], atol=0.03)
    assert np.all(r["s1_ci"][:, 1] - r["s1_ci"][:, 0] < 0.05) and r["s1_ci"][0, 0] <= ref["s1"][0] <= r["s1_ci"][0, 1]
    # a hypothesis with weights 0.25 / 0.75 (bins = its cumulative weights) and an additive uniform input
    x = np.random.default_rng(4).random((8000, 2))
    y = 3.0 * (x[:, 0] >= 0.25) + x[:, 1]
    v = 0.1875 * 9 + 1 / 12
    g = sensitivity.given_data_first_order(x, y, edges=[[0.25, 1.0], None], n_boot=0)
    np.testing.assert_allclose(g["s1"], [0.1875 * 9 / v, (1 / 12) / v * (1 - 1 / 64)], atol=0.01)
    assert g["n_bins"] == [2, 8]
