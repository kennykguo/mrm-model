"""Variance-based sensitivity (Sobol / Saltelli), one-at-a-time screening (tornado) over the 07 4.8 uncertainty set,
and local identifiability (Jacobian SVD and the half-prior rule of 07 4.3) for the validation report (07 P1-23,
section 4.9 items 3 and 4).

SALib is not installed, so the estimators are implemented here and verified against the analytic Ishigami indices
(tests/verification/test_sensitivity.py; also against ``scipy.stats.sobol_indices``).

Sobol / Saltelli
    :func:`sobol_design` draws the two base matrices A, B (n x d, scrambled Sobol points on [0, 1)^d);
    :func:`saltelli_stack` stacks [A; B; AB_1; ...; AB_d] (AB_i = A with column i from B), n (d + 2) rows; the
    model is evaluated on every row; :func:`sobol_indices` returns the first-order indices (Saltelli et al. 2010,
    Comput. Phys. Commun. 181, 259, table 2 (b)) S_i = mean(f_B (f_AB_i - f_A)) / V and the total indices
    (Jansen 1999) ST_i = mean((f_A - f_AB_i)^2) / (2 V), V = Var([f_A; f_B]), with bootstrap confidence intervals
    (rows resampled jointly). :func:`given_data_first_order` re-estimates S1 from the independent A and B rows by
    binning (better resolved than the Saltelli S1 at small n for threshold-like outputs; biased low by the binning
    of a continuous input). Indices are with respect to the uniform u-space of the inputs; a discrete input
    (a hypothesis) is a piecewise-constant function of its u, so the indices measure the variance its prior weights
    induce.

Uncertainty set
    :class:`UncertaintySpace` maps every ``uncertainty`` entry of a Config to the u in [0, 1] that
    ``mrm.params.realize(..., uncertainty="lhs")`` turns into a draw (same ordering, distributions, hypothesis
    weights and cumulative bins as realize), and back. :func:`injected_lhs` makes realize use a given (B, D) u
    matrix instead of its own Latin hypercube, so any design (tornado points, Saltelli rows) runs as one batched
    Plant through the unchanged engine. :func:`check_injection` realizes the design and compares every world's
    draws with the intended values (no silent mismatch).

Identifiability
    :func:`identifiability` takes a Jacobian normalized to residual sigmas per prior half-width and returns its SVD,
    the linearized 95 % intervals and the half-prior rule: a parameter may be tagged ``calibrated`` only when its
    95 % interval is at most half its prior range (07 4.3). In normalized units the prior range is [-1, 1] (width
    2), so the rule is 1.96 sigma <= 0.5, and a direction of the SVD is identified by the rule when its singular
    value is at least 1.96 / 0.5 = 3.92. The linearized (Fisher) interval is a necessary screen, not the profile
    likelihood 07 asks for; the profiles are the calibration task's (the report reads them when present).

This module is pure numerics plus ``mrm.params`` (no target files, no engine); ``mrm/calibration`` may import it.
"""
from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from scipy.special import ndtr, ndtri

__all__ = ["SobolResult", "sobol_design", "saltelli_stack", "split_stack", "sobol_indices",
           "given_data_first_order", "ishigami",
           "ishigami_indices", "EntryAxis", "UncertaintySpace", "injected_lhs", "check_injection",
           "tornado_table", "screen_top", "identifiability", "HALF_PRIOR_SINGULAR_VALUE", "LHS_CLIP"]

#: realize() clips the Latin-hypercube columns to [LHS_CLIP, 1 - LHS_CLIP]
LHS_CLIP = 1e-12
#: in prior-half-width units, a direction with singular value >= this has a 95 % half-width <= 0.5 (07 4.3 rule)
HALF_PRIOR_SINGULAR_VALUE = 1.959963984540054 / 0.5


# ================================================================================================ Sobol
@dataclass
class SobolResult:
    """First-order (``s1``) and total (``st``) Sobol indices, (d,) each, with bootstrap intervals (d, 2)."""
    names: list[str]
    s1: np.ndarray
    st: np.ndarray
    s1_ci: np.ndarray
    st_ci: np.ndarray
    variance: float
    mean: float
    n: int
    n_boot: int
    conf: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        def f(x):
            return [None if not np.isfinite(v) else float(v) for v in np.asarray(x, float).ravel()]
        return {"names": list(self.names), "s1": f(self.s1), "st": f(self.st),
                "s1_ci": [f(r) for r in np.asarray(self.s1_ci)], "st_ci": [f(r) for r in np.asarray(self.st_ci)],
                "variance": float(self.variance), "mean": float(self.mean), "n": int(self.n),
                "n_boot": int(self.n_boot), "conf": float(self.conf), "notes": list(self.notes)}


def sobol_design(d: int, n: int, *, seed: int = 0, scramble: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Base matrices A, B (n, d) from one scrambled Sobol sequence of dimension 2d (n should be a power of 2)."""
    from scipy.stats import qmc
    if d < 1 or n < 2:
        raise ValueError("sobol_design needs d >= 1 and n >= 2")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)          # 'balance properties' when n is not 2^m
        x = qmc.Sobol(d=2 * d, scramble=scramble, seed=seed).random(n)
    return x[:, :d], x[:, d:]


def saltelli_stack(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Rows [A; B; AB_1; ...; AB_d] (n (d + 2), d); AB_i is A with column i taken from B."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if A.shape != B.shape or A.ndim != 2:
        raise ValueError("A and B must be (n, d) of the same shape")
    n, d = A.shape
    blocks = [A, B]
    for i in range(d):
        ab = A.copy()
        ab[:, i] = B[:, i]
        blocks.append(ab)
    return np.vstack(blocks)


def split_stack(y: np.ndarray, n: int, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(f_A (n,), f_B (n,), f_AB (d, n)) from the model values on :func:`saltelli_stack` rows."""
    y = np.asarray(y, float).ravel()
    if y.size != n * (d + 2):
        raise ValueError(f"expected {n * (d + 2)} values, got {y.size}")
    return y[:n], y[n:2 * n], y[2 * n:].reshape(d, n)


def _indices(fA, fB, fAB) -> tuple[np.ndarray, np.ndarray, float]:
    v = float(np.var(np.concatenate([fA, fB])))
    if not v > 0.0:
        d = fAB.shape[0]
        return np.full(d, np.nan), np.full(d, np.nan), v
    s1 = np.mean(fB[None, :] * (fAB - fA[None, :]), axis=1) / v
    st = 0.5 * np.mean((fA[None, :] - fAB) ** 2, axis=1) / v
    return s1, st, v


def sobol_indices(fA: np.ndarray, fB: np.ndarray, fAB: np.ndarray, *, names: Sequence[str] | None = None,
                  n_boot: int = 500, conf: float = 0.95, seed: int = 0) -> SobolResult:
    """Saltelli (2010) first-order and Jansen (1999) total indices with percentile bootstrap intervals.

    ``fA``, ``fB``: (n,) model values on A and B; ``fAB``: (d, n) on AB_i. Values must be finite (transform a
    metric that can be infinite before calling). A zero output variance gives NaN indices and a note."""
    fA = np.asarray(fA, float).ravel()
    fB = np.asarray(fB, float).ravel()
    fAB = np.asarray(fAB, float)
    n = fA.size
    if fB.size != n or fAB.ndim != 2 or fAB.shape[1] != n:
        raise ValueError("fA, fB must be (n,) and fAB (d, n)")
    if not (np.all(np.isfinite(fA)) and np.all(np.isfinite(fB)) and np.all(np.isfinite(fAB))):
        raise ValueError("sobol_indices needs finite model values (transform inf/NaN first)")
    d = fAB.shape[0]
    names = list(names) if names is not None else [f"x{i + 1}" for i in range(d)]
    s1, st, v = _indices(fA, fB, fAB)
    notes = []
    if not v > 0.0:
        notes.append("zero output variance: the metric does not vary over the design")
    rng = np.random.default_rng(seed)
    bs1 = np.full((n_boot, d), np.nan)
    bst = np.full((n_boot, d), np.nan)
    if v > 0.0 and n_boot > 0:
        for k in range(n_boot):
            j = rng.integers(0, n, n)
            a, b_, c = _indices(fA[j], fB[j], fAB[:, j])
            bs1[k], bst[k] = a, b_
    lo, hi = (1.0 - conf) / 2.0 * 100.0, (1.0 + conf) / 2.0 * 100.0
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            s1_ci = np.stack([np.nanpercentile(bs1, lo, axis=0), np.nanpercentile(bs1, hi, axis=0)], axis=1) \
                if n_boot else np.full((d, 2), np.nan)
            st_ci = np.stack([np.nanpercentile(bst, lo, axis=0), np.nanpercentile(bst, hi, axis=0)], axis=1) \
                if n_boot else np.full((d, 2), np.nan)
    return SobolResult(names=names, s1=s1, st=st, s1_ci=s1_ci, st_ci=st_ci, variance=v,
                       mean=float(np.mean(np.concatenate([fA, fB]))), n=n, n_boot=int(n_boot), conf=float(conf),
                       notes=notes)


def given_data_first_order(X: np.ndarray, y: np.ndarray, *, edges: Sequence[Sequence[float] | None] | None = None,
                           n_bins: int = 8, n_boot: int = 500, conf: float = 0.95, seed: int = 0) -> dict:
    """First-order indices from independent rows by binning (the 'given data' estimator):
    S1_i = (sum_k p_k (mean_k - mean)^2 - sum_k s_k^2 / N) / V, with the bias of the bin means removed.

    ``X`` (N, d) u in [0, 1] (independent rows, e.g. the A and B rows of a Saltelli design), ``y`` (N,) finite.
    ``edges[i]``: the u bin edges of column i (a hypothesis: its cumulative weights, so a bin is a choice); None:
    ``n_bins`` equal-probability bins. Binning a continuous input misses the variation of the conditional mean
    inside a bin, so S1 of a smooth input is biased low: about 6 % for one period of a sine over 8 bins, about
    20 % for two periods (Ishigami x1 and x2 at 256 rows); a monotone or step-like effect loses less. Returns
    {s1 (d,), s1_ci (d, 2), n, n_bins (d,)}; percentile bootstrap over rows (at 256 rows it covers Ishigami x1 and x3
    in 95 and 98 % of repeats; x2 less, through the binning bias)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    N, d = X.shape
    if y.size != N or not np.all(np.isfinite(y)):
        raise ValueError("given_data_first_order needs one finite y per row")
    idx = np.empty((N, d), dtype=np.int64)
    nb = []
    for i in range(d):
        e = None if edges is None else edges[i]
        if e is None:
            inner = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        else:
            inner = np.asarray(e, float)[:-1]                  # cdf without its final 1.0
        idx[:, i] = np.searchsorted(inner, X[:, i], side="right")
        nb.append(len(inner) + 1)

    def est(rows: np.ndarray) -> np.ndarray:
        yy = y[rows]
        v = float(np.var(yy))
        out = np.full(d, np.nan)
        if not v > 0.0:
            return out
        m, n = float(np.mean(yy)), yy.size
        for i in range(d):
            b = idx[rows, i]
            cnt = np.bincount(b, minlength=nb[i]).astype(float)
            s = np.bincount(b, weights=yy, minlength=nb[i])
            s2 = np.bincount(b, weights=yy * yy, minlength=nb[i])
            k = cnt > 0
            mk = s[k] / cnt[k]
            between = float(np.sum(cnt[k] * (mk - m) ** 2)) / n
            var_k = np.where(cnt[k] > 1, (s2[k] - cnt[k] * mk ** 2) / np.maximum(cnt[k] - 1, 1), 0.0)
            out[i] = (between - float(np.sum(var_k)) / n) / v
        return out

    s1 = est(np.arange(N))
    rng = np.random.default_rng(seed)
    bs = np.array([est(rng.integers(0, N, N)) for _ in range(int(n_boot))]) if n_boot else np.full((0, d), np.nan)
    lo, hi = (1.0 - conf) / 2.0 * 100.0, (1.0 + conf) / 2.0 * 100.0
    with warnings_ignored():
        ci = np.stack([np.nanpercentile(bs, lo, axis=0), np.nanpercentile(bs, hi, axis=0)], axis=1) if n_boot \
            else np.full((d, 2), np.nan)
    return {"s1": s1, "s1_ci": ci, "n": int(N), "n_bins": nb}


@contextlib.contextmanager
def warnings_ignored() -> Iterator[None]:
    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        yield


def ishigami(u: np.ndarray, a: float = 7.0, b: float = 0.1) -> np.ndarray:
    """Ishigami function of u in [0, 1]^3 (x = -pi + 2 pi u): sin x1 + a sin^2 x2 + b x3^4 sin x1."""
    x = -math.pi + 2.0 * math.pi * np.asarray(u, float)
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])


def ishigami_indices(a: float = 7.0, b: float = 0.1) -> dict:
    """Analytic first-order and total Sobol indices of :func:`ishigami` (uniform inputs on [-pi, pi])."""
    p4, p8 = math.pi ** 4, math.pi ** 8
    v1 = 0.5 * (1.0 + b * p4 / 5.0) ** 2
    v2 = a * a / 8.0
    v13 = 8.0 * b * b * p8 / 225.0
    v = a * a / 8.0 + b * p4 / 5.0 + b * b * p8 / 18.0 + 0.5
    return {"s1": np.array([v1 / v, v2 / v, 0.0]), "st": np.array([(v1 + v13) / v, v2 / v, v13 / v]),
            "variance": v}


# ================================================================================================ uncertainty set
@dataclass
class EntryAxis:
    """One ``uncertainty`` entry as an axis of u in [0, 1] (the mapping realize applies in lhs mode)."""
    name: str
    kind: str                      # hypothesis | interval | dist
    target: str | None
    pinned: Any
    fields: dict
    choices: tuple = ()            # hypothesis: every choice, in realize's order
    weights: tuple = ()            # hypothesis: normalized weights (0 outside the allowed set)
    cdf: tuple = ()                # hypothesis: cumulative weights
    nominal: Any = None            # hypothesis: choice name; interval/dist: value (config units) or None

    @property
    def support(self) -> tuple:
        """Hypothesis choices that the sampled set can reach (positive weight)."""
        return tuple(c for c, w in zip(self.choices, self.weights) if w > 0)

    def value_of_u(self, u: float) -> Any:
        """The draw realize makes from u (a choice name, or a value in config units)."""
        u = float(np.clip(u, LHS_CLIP, 1.0 - LHS_CLIP))
        if self.kind == "hypothesis":
            idx = min(int(np.searchsorted(np.asarray(self.cdf), u, side="right")), len(self.choices) - 1)
            return self.choices[idx]
        return float(_interval_x(self.fields, self.kind, np.array([u]))[0])

    def u_of_value(self, value: Any) -> float:
        """A u that realize maps to ``value`` (bin midpoint for a choice; the inverse CDF for a number)."""
        if self.kind == "hypothesis":
            c = str(value)
            if c not in self.choices:
                raise ValueError(f"uncertainty.{self.name}: {c!r} is not a choice of {self.choices}")
            i = self.choices.index(c)
            if self.weights[i] <= 0:
                raise ValueError(f"uncertainty.{self.name}: choice {c!r} has zero weight (a stress corner); the "
                                 "sampled set cannot reach it")
            lo = self.cdf[i - 1] if i > 0 else 0.0
            return 0.5 * (lo + self.cdf[i])
        return float(_interval_u(self.fields, self.kind, float(value)))

    def tornado_points(self) -> list[tuple[str, float]]:
        """(label, u) of the one-at-a-time points: the interval ends (2.5 / 97.5 % for an unbounded
        distribution), or every reachable choice other than the nominal."""
        if self.kind == "hypothesis":
            return [(c, self.u_of_value(c)) for c in self.support if c != self.nominal]
        if self.kind == "interval":
            return [("low", 0.0), ("high", 1.0)]
        return [("p2.5", 0.025), ("p97.5", 0.975)]


def _interval_x(f: Mapping, kind: str, u: np.ndarray) -> np.ndarray:
    """realize._interval_x (the same formulas; kept in step by tests/verification/test_sensitivity.py)."""
    if kind == "interval":
        lo, hi = float(f["low"]), float(f["high"])
        d = f.get("dist", "uniform")
        if d == "uniform":
            return lo + (hi - lo) * u
        if d == "loguniform":
            return np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * u)
        c = float(f.get("nominal", 0.5 * (lo + hi))) if isinstance(f.get("nominal"), (int, float)) else 0.5 * (lo + hi)
        fc = (c - lo) / (hi - lo) if hi > lo else 0.5
        return np.where(u < fc, lo + np.sqrt(u * (hi - lo) * (c - lo)), hi - np.sqrt((1 - u) * (hi - lo) * (hi - c)))
    mean, sigma = float(f["mean"]), float(f["sigma"])
    lognormal = f.get("dist", "normal") == "lognormal"
    tr = f.get("truncate")
    if tr:
        a, b = _trunc_z(f, mean, sigma, lognormal)
        u = ndtr(a) + u * (ndtr(b) - ndtr(a))
    z = ndtri(np.clip(u, 1e-300, 1 - 1e-16))
    return mean * np.exp(sigma * z) if lognormal else mean + sigma * z


def _trunc_z(f: Mapping, mean: float, sigma: float, lognormal: bool) -> tuple[float, float]:
    lo_hi = np.asarray(f["truncate"], float)
    if lognormal:
        with np.errstate(divide="ignore"):
            zb = np.log(np.maximum(lo_hi, 0.0) / mean) / max(sigma, 1e-300)
    else:
        zb = (lo_hi - mean) / max(sigma, 1e-300)
    return float(zb[0]), float(zb[1])


def _interval_u(f: Mapping, kind: str, x: float) -> float:
    """Inverse of :func:`_interval_x`."""
    if kind == "interval":
        lo, hi = float(f["low"]), float(f["high"])
        if not lo <= x <= hi:
            raise ValueError(f"value {x} outside [{lo}, {hi}]")
        if hi == lo:
            return 0.5
        d = f.get("dist", "uniform")
        if d == "uniform":
            return (x - lo) / (hi - lo)
        if d == "loguniform":
            return (math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo))
        c = float(f.get("nominal", 0.5 * (lo + hi))) if isinstance(f.get("nominal"), (int, float)) else 0.5 * (lo + hi)
        if x < c:
            return (x - lo) ** 2 / ((hi - lo) * (c - lo))
        return 1.0 - (hi - x) ** 2 / ((hi - lo) * (hi - c)) if hi > c else 1.0
    mean, sigma = float(f["mean"]), float(f["sigma"])
    lognormal = f.get("dist", "normal") == "lognormal"
    z = math.log(x / mean) / sigma if lognormal else (x - mean) / sigma
    u = float(ndtr(z))
    if f.get("truncate"):
        a, b = _trunc_z(f, mean, sigma, lognormal)
        u = (u - float(ndtr(a))) / (float(ndtr(b)) - float(ndtr(a)))
    return u


class UncertaintySpace:
    """The ``uncertainty`` entries of a Config as the u-columns of realize's Latin hypercube (sorted names)."""

    def __init__(self, cfg):
        from mrm.params.config import _hyp_choice_map
        self.cfg = cfg
        self.names: list[str] = sorted(cfg.uncertainty)
        self.axes: dict[str, EntryAxis] = {}
        for name in self.names:
            e = cfg.uncertainty[name]
            f = dict(e.fields)
            if e.kind == "hypothesis":
                cm = _hyp_choice_map(e)
                choices = tuple(cm)
                allowed = e.restrict or list(choices)
                w = {str(k): v for k, v in (f.get("weights") or {}).items()}
                wv = np.array([float(w.get(c, 1.0)) if c in allowed else 0.0 for c in choices])
                if wv.sum() <= 0:
                    raise ValueError(f"uncertainty.{name}: weights sum to zero")
                wv = wv / wv.sum()
                nom = str(f.get("nominal"))
                nom = nom if nom in allowed else allowed[0]
                self.axes[name] = EntryAxis(name, "hypothesis", e.target, e.pinned, f, choices, tuple(wv),
                                            tuple(np.cumsum(wv)), nom)
            else:
                kind = "interval" if e.kind == "interval" else "dist"
                self.axes[name] = EntryAxis(name, kind, e.target, e.pinned, f, nominal=self._numeric_nominal(e))

    def _numeric_nominal(self, e) -> float | None:
        nom = e.fields.get("nominal")
        if isinstance(nom, (int, float)):
            return float(nom)
        if e.target is None:
            return None
        mode = e.fields.get("mode", "absolute")
        if mode == "scale":
            return 1.0
        if mode == "offset":
            return 0.0
        v = np.asarray(self.cfg.get(e.target), float).ravel()     # absolute without a nominal: the leaf keeps it
        if v.size and np.allclose(v, v[0], rtol=0, atol=0):
            return float(v[0])
        return None

    @property
    def d(self) -> int:
        return len(self.names)

    @property
    def free(self) -> list[str]:
        """Entries realize draws (not pinned)."""
        return [n for n in self.names if self.axes[n].pinned is None]

    def nominal_u(self) -> np.ndarray:
        """(D,) u of the nominal world (pinned entries: 0.5, ignored by realize). Raises for an entry without a
        nominal that a u can express."""
        out = np.full(self.d, 0.5)
        for i, n in enumerate(self.names):
            ax = self.axes[n]
            if ax.pinned is not None:
                continue
            if ax.nominal is None:
                raise ValueError(f"uncertainty.{n} has no nominal value a u can express")
            out[i] = ax.u_of_value(ax.nominal)
        return out

    def design(self, base: np.ndarray, columns: Mapping[str, np.ndarray]) -> np.ndarray:
        """(B, D) u matrix: ``base`` (D,) in every row, then the given columns ({name: (B,)}) replaced."""
        sizes = {np.asarray(v).size for v in columns.values()} or {1}
        if len(sizes) != 1:
            raise ValueError("every column must have the same length")
        B = sizes.pop()
        U = np.tile(np.asarray(base, float)[None, :], (B, 1))
        for n, col in columns.items():
            U[:, self.names.index(n)] = np.asarray(col, float).ravel()
        return U

    def expected_world(self, U: np.ndarray) -> dict[str, list]:
        """{name: [draw per row]} that realize should produce from ``U`` (choice names or values)."""
        return {n: [self.axes[n].value_of_u(u) for u in U[:, i]] for i, n in enumerate(self.names)
                if self.axes[n].pinned is None}

    def describe(self, name: str, u: float) -> str:
        """Readable value of entry ``name`` at u (a choice name, or a number with 4 significant digits)."""
        v = self.axes[name].value_of_u(u)
        return v if isinstance(v, str) else f"{v:.4g}"


@contextlib.contextmanager
def injected_lhs(U: np.ndarray) -> Iterator[None]:
    """Inside this context ``mrm.params.realize(..., uncertainty='lhs')`` takes its Latin-hypercube matrix from
    ``U`` (B, D) instead of ``scipy.stats.qmc.LatinHypercube(d=D, seed).random(B)``. A call with another d or B
    raises (so an unexpected realize inside the context is loud, not silently wrong)."""
    from scipy.stats import qmc
    U = np.array(U, float)
    if U.ndim != 2:
        raise ValueError("U must be (B, D)")
    original = qmc.LatinHypercube

    class _Fixed:
        def __init__(self, d, *args, **kwargs):
            if int(d) != U.shape[1]:
                raise RuntimeError(f"injected_lhs: realize asked for d = {d}, the design has {U.shape[1]} columns")

        def random(self, n=1, *args, **kwargs):
            if int(n) != U.shape[0]:
                raise RuntimeError(f"injected_lhs: realize asked for {n} rows, the design has {U.shape[0]}")
            return U.copy()

    qmc.LatinHypercube = _Fixed
    try:
        yield
    finally:
        qmc.LatinHypercube = original


def check_injection(space: UncertaintySpace, U: np.ndarray, *, rtol: float = 1e-9) -> dict:
    """Realize ``space.cfg`` with ``U`` injected and compare each world's draws with the intended values.

    Returns {"ok": bool, "mismatches": [str], "worlds": B}. Uses ``PlantParams.world`` (hypotheses as choice
    indices, intervals as values)."""
    import warnings
    from mrm import params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with injected_lhs(U):
            pp = params.realize(space.cfg, U.shape[0], seeds=0, uncertainty="lhs", variation="off")
    exp = space.expected_world(U)
    bad = []
    for n, vals in exp.items():
        got = np.asarray(pp.world[n])
        ax = space.axes[n]
        for b, v in enumerate(vals):
            if ax.kind == "hypothesis":
                g = pp.world_choices[n][int(got[b])]
                if g != v:
                    bad.append(f"{n}[{b}]: realize drew {g}, intended {v}")
            elif not np.isclose(float(got[b]), float(v), rtol=rtol, atol=0.0):
                bad.append(f"{n}[{b}]: realize drew {float(got[b])!r}, intended {v!r}")
    return {"ok": not bad, "mismatches": bad[:20], "n_mismatches": len(bad), "worlds": int(U.shape[0])}


# ================================================================================================ tornado
def tornado_table(base: Mapping[str, float], points: Sequence[tuple[str, str, Mapping[str, float]]],
                  metrics: Sequence[str]) -> dict[str, list[dict]]:
    """One-at-a-time swings. ``points``: (entry, label, {metric: value}); ``base``: {metric: nominal value}.

    Returns {metric: [{entry, lo, hi, lo_label, hi_label, swing}] sorted by swing, largest first}; lo/hi are the
    smallest and largest values over the entry's points and the base (the base is always inside the bar).
    Non-finite values are kept as labels (``inf``) and make the swing infinite."""
    out: dict[str, list[dict]] = {}
    for m in metrics:
        rows: dict[str, dict] = {}
        for entry, label, vals in points:
            v = float(vals.get(m, np.nan))
            r = rows.setdefault(entry, {"entry": entry, "values": [], "labels": []})
            r["values"].append(v)
            r["labels"].append(label)
        b = float(base.get(m, np.nan))
        res = []
        for entry, r in rows.items():
            vs = np.asarray(r["values"], float)
            labs = list(r["labels"])
            ilo = int(np.nanargmin(np.where(np.isnan(vs), np.inf, vs))) if np.any(~np.isnan(vs)) else 0
            ihi = int(np.nanargmax(np.where(np.isnan(vs), -np.inf, vs))) if np.any(~np.isnan(vs)) else 0
            lo, hi = float(vs[ilo]), float(vs[ihi])
            lo_l, hi_l = labs[ilo], labs[ihi]
            if np.isfinite(b) and not lo <= b:
                lo, lo_l = b, "nominal"
            if np.isfinite(b) and not hi >= b:
                hi, hi_l = b, "nominal"
            with np.errstate(invalid="ignore"):
                sw = hi - lo if not (np.isnan(lo) or np.isnan(hi)) else np.nan
            res.append({"entry": entry, "lo": lo, "hi": hi, "lo_label": lo_l, "hi_label": hi_l,
                        "swing": float(sw), "values": [float(x) for x in vs], "labels": labs})
        res.sort(key=lambda r: (-(r["swing"] if np.isfinite(r["swing"]) else (np.inf if r["swing"] > 0 else -1)),
                                r["entry"]))
        out[m] = res
    return out


def screen_top(tornado: Mapping[str, Sequence[Mapping]], k: int = 8) -> list[str]:
    """The k entries with the largest normalized swing: per metric, swing / largest finite swing (an infinite
    swing counts as 1); the score of an entry is its largest normalized swing over the metrics (ties by name)."""
    score: dict[str, float] = {}
    for m, rows in tornado.items():
        fin = [r["swing"] for r in rows if np.isfinite(r["swing"]) and r["swing"] > 0]
        ref = max(fin) if fin else 0.0
        for r in rows:
            s = r["swing"]
            x = 1.0 if (np.isinf(s) and s > 0) else (s / ref if ref > 0 and np.isfinite(s) else 0.0)
            score[r["entry"]] = max(score.get(r["entry"], 0.0), x)
    return [e for e, _ in sorted(score.items(), key=lambda t: (-t[1], t[0]))[:k]]


# ================================================================================================ identifiability
def identifiability(J: np.ndarray, names: Sequence[str], *, rcond: float = 1e-10, top: int = 4) -> dict:
    """SVD of a normalized Jacobian J (residuals in sigmas x parameters in prior half-widths) and the linearized
    half-prior rule of 07 4.3.

    Returns: singular values ``s`` (descending), ``condition`` (s_max/s_min, inf when singular), ``rank``,
    ``directions`` [{s, identified (s >= 3.92), loadings: [(name, v)] (largest |v| first, ``top`` of them)}],
    per parameter ``half_width_95`` (1.96 sqrt((J^T J)^+_jj), inf when the parameter loads on a null direction),
    ``interval_frac_of_prior`` (95 % interval width / prior width = half_width_95 / 1) and ``half_prior_rule``
    (interval <= half the prior range), plus ``column_norm`` (sensitivity of each parameter alone)."""
    J = np.asarray(J, float)
    names = list(names)
    if J.ndim != 2 or J.shape[1] != len(names):
        raise ValueError("J must be (n_residuals, n_params) with one name per column")
    good_rows = np.all(np.isfinite(J), axis=1)
    Jf = J[good_rows]
    U, s, Vt = np.linalg.svd(Jf, full_matrices=False) if Jf.size else (None, np.zeros(0), np.zeros((0, J.shape[1])))
    p = J.shape[1]
    s_full = np.zeros(p)
    s_full[:s.size] = s
    if s.size < p:                                       # fewer residuals than parameters: extra null directions
        _, _, Vt_full = np.linalg.svd(np.vstack([Jf, np.zeros((p - Jf.shape[0], p))]), full_matrices=True)
        Vt = Vt_full
    smax = float(s_full.max()) if s_full.size else 0.0
    null = s_full <= rcond * max(smax, 1e-300)
    inv_s2 = np.where(null, 0.0, 1.0 / np.where(null, 1.0, s_full) ** 2)
    cov = (Vt.T * inv_s2[None, :]) @ Vt
    hw = 1.959963984540054 * np.sqrt(np.maximum(np.diag(cov), 0.0))
    if null.any():
        loads_null = np.max(np.abs(Vt[null, :]), axis=0)
        hw = np.where(loads_null > 1e-6, np.inf, hw)
    dirs = []
    for k in range(p):
        v = Vt[k]
        order = np.argsort(-np.abs(v))[:top]
        dirs.append({"s": float(s_full[k]), "identified": bool(s_full[k] >= HALF_PRIOR_SINGULAR_VALUE),
                     "loadings": [(names[j], float(v[j])) for j in order]})
    with np.errstate(divide="ignore"):
        cond = float(smax / s_full.min()) if s_full.size and s_full.min() > 0 else float("inf")
    return {"names": names, "s": [float(x) for x in s_full], "condition": cond,
            "rank": int(np.sum(~null)), "n_residuals_used": int(good_rows.sum()),
            "n_residuals_dropped": int((~good_rows).sum()), "directions": dirs,
            "half_width_95": [float(x) for x in hw], "interval_frac_of_prior": [float(x) for x in hw],
            "half_prior_rule": [bool(x <= 0.5) for x in hw],
            "column_norm": [float(x) for x in np.linalg.norm(np.where(np.isfinite(J), J, 0.0), axis=0)],
            "threshold_singular_value": HALF_PRIOR_SINGULAR_VALUE}
