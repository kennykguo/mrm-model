# measure which uncertain inputs move outputs and which parameter combinations data constrain.
# `tornado_table` changes one uncertainty entry at a time.
# Sobol analysis then splits output variance among selected inputs, including their interactions.
# `identifiability` instead uses a normalized Jacobian to screen parameters that calibration data may constrain.
# this module reads no target files and runs no engine.
# callers supply model evaluations.
# `sobol_design` draws A and B.
# `saltelli_stack` adds AB_i, which is A with column i from B.
# evaluate all rows, then use `split_stack` and `sobol_indices`.
# with V = Var([f_A, f_B]), the estimators are S_i = mean(f_B (f_AB_i - f_A)) / V and ST_i = mean((f_A - f_AB_i)^2) / (2 V).
# bootstrap resampling keeps corresponding rows together.
# `given_data_first_order` provides a binned check that can miss smooth variation within each bin.
# indices depend on the declared input distributions.
# hypothesis weights set how often discrete choices occur.
# UncertaintySpace maps uniforms to the same draws as realize.
# injected_lhs temporarily supplies a chosen design, and check_injection verifies the actual draws.
# Jacobian residuals use sigmas and parameters use prior half-widths.
# a singular value of about 3.92 gives a linearized 95 percent interval half the prior width.
# this is only a necessary screen.
# full profile likelihoods and the calibration decision gates are still required.
from __future__ import annotations

import contextlib
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from scipy.special import ndtr, ndtri

__all__ = ["SobolResult", "sobol_design", "saltelli_stack", "split_stack", "sobol_indices",
           "given_data_first_order", "ishigami",
           "ishigami_indices", "EntryAxis", "UncertaintySpace", "injected_lhs", "check_injection",
           "tornado_table", "screen_top", "identifiability", "HALF_PRIOR_SINGULAR_VALUE", "LHS_CLIP"]

# realize() clips every Latin-hypercube column to [LHS_CLIP, 1 - LHS_CLIP] (u is dimensionless)
LHS_CLIP = 1e-12
# 97.5 % quantile of the standard normal - the half-width of a two-sided 95 % interval, in sigmas
_Z_95 = 1.959963984540054
# singular value (sigmas per prior half-width) at which a direction's 95 % half-width is 0.5 prior half-widths, that is, its 95 % interval is half the prior range (the half-prior rule)
HALF_PRIOR_SINGULAR_VALUE = _Z_95 / 0.5


# Sobol indices
# first-order (`s1`) and total (`st`) Sobol indices, (d,) each, with bootstrap intervals (d, 2) each.
# `variance` and `mean` are of the model output over the A and B rows.
# `n` is the number of base rows, `n_boot` the number of bootstrap resamples and `conf` the interval level (0.95 for 95 %).
@dataclass
class SobolResult:
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

    # jSON-ready form (a non-finite index becomes None).
    def to_dict(self) -> dict:
        def finite_list(x):
            return [None if not np.isfinite(v) else float(v) for v in np.asarray(x, float).ravel()]
        return {"names": list(self.names), "s1": finite_list(self.s1), "st": finite_list(self.st),
                "s1_ci": [finite_list(r) for r in np.asarray(self.s1_ci)],
                "st_ci": [finite_list(r) for r in np.asarray(self.st_ci)],
                "variance": float(self.variance), "mean": float(self.mean), "n": int(self.n),
                "n_boot": int(self.n_boot), "conf": float(self.conf), "notes": list(self.notes)}


# base matrices A and B, (n, d) each, from one scrambled Sobol sequence of dimension 2 d.
# a is the first d columns and B the last d.
# `n` should be a power of 2 (scipy warns otherwise. the warning is silenced here).
def sobol_design(d: int, n: int, *, seed: int = 0, scramble: bool = True) -> tuple[np.ndarray, np.ndarray]:
    from scipy.stats import qmc
    if d < 1 or n < 2:
        raise ValueError("sobol_design needs d >= 1 and n >= 2")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)          # 'balance properties' when n is not 2^m
        x = qmc.Sobol(d=2 * d, scramble=scramble, seed=seed).random(n)
    return x[:, :d], x[:, d:]


# rows [A. B. AB_1. .... AB_d], shape (n (d + 2), d).
# AB_i is A with column i taken from B.
def saltelli_stack(A: np.ndarray, B: np.ndarray) -> np.ndarray:
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


# split the model values on the `saltelli_stack` rows into f_A (n,), f_B (n,) and f_AB (d, n).
def split_stack(y: np.ndarray, n: int, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, float).ravel()
    if y.size != n * (d + 2):
        raise ValueError(f"expected {n * (d + 2)} values, got {y.size}")
    return y[:n], y[n:2 * n], y[2 * n:].reshape(d, n)


# saltelli (2010) first-order and Jansen (1999) total indices, with percentile bootstrap intervals.
# `fA`, `fB` — (n,) model values on the A and B rows.
# `fAB` — (d, n) model values on the AB_i rows.
# the values must be finite (transform a metric that can be infinite before the call).
# a zero output variance gives NaN indices and a note.
def sobol_indices(fA: np.ndarray, fB: np.ndarray, fAB: np.ndarray, *, names: Sequence[str] | None = None,
                  n_boot: int = 500, conf: float = 0.95, seed: int = 0) -> SobolResult:
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
    s1, st, variance = _saltelli_jansen(fA, fB, fAB)
    notes = []
    if not variance > 0.0:
        notes.append("zero output variance: the metric does not vary over the design")
    rng = np.random.default_rng(seed)
    boot_s1 = np.full((n_boot, d), np.nan)
    boot_st = np.full((n_boot, d), np.nan)
    if variance > 0.0 and n_boot > 0:
        for k in range(n_boot):
            rows = rng.integers(0, n, n)
            boot_s1[k], boot_st[k], _ = _saltelli_jansen(fA[rows], fB[rows], fAB[:, rows])
    lo_pct, hi_pct = (1.0 - conf) / 2.0 * 100.0, (1.0 + conf) / 2.0 * 100.0
    with _warnings_ignored():
        s1_ci = np.stack([np.nanpercentile(boot_s1, lo_pct, axis=0), np.nanpercentile(boot_s1, hi_pct, axis=0)],
                         axis=1) if n_boot else np.full((d, 2), np.nan)
        st_ci = np.stack([np.nanpercentile(boot_st, lo_pct, axis=0), np.nanpercentile(boot_st, hi_pct, axis=0)],
                         axis=1) if n_boot else np.full((d, 2), np.nan)
    return SobolResult(names=names, s1=s1, st=st, s1_ci=s1_ci, st_ci=st_ci, variance=variance,
                       mean=float(np.mean(np.concatenate([fA, fB]))), n=n, n_boot=int(n_boot), conf=float(conf),
                       notes=notes)


# estimate first-order variance effects by binning independent input rows.
# X has shape (N, d) in uniform coordinates and y has shape (N,).
# `edges` can use hypothesis cumulative weights so each choice is a bin.
# otherwise use equal-probability bins.
# subtract estimated bin-mean sampling noise from the between-bin variance, then divide by total output variance.
# continuous effects within bins are lost, so smooth-input indices can be biased low.
# bootstrap intervals describe sampling uncertainty, not this bias.
# return indices (d,), intervals (d, 2), row count, and bin counts.
def given_data_first_order(X: np.ndarray, y: np.ndarray, *, edges: Sequence[Sequence[float] | None] | None = None,
                           n_bins: int = 8, n_boot: int = 500, conf: float = 0.95, seed: int = 0) -> dict:
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    N, d = X.shape
    if y.size != N or not np.all(np.isfinite(y)):
        raise ValueError("given_data_first_order needs one finite y per row")
    bin_of_row = np.empty((N, d), dtype=np.int64)          # bin index of every row, per input column
    n_bins_of_col = []
    for i in range(d):
        col_edges = None if edges is None else edges[i]
        if col_edges is None:
            inner = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        else:
            inner = np.asarray(col_edges, float)[:-1]          # cdf without its final 1.0
        bin_of_row[:, i] = np.searchsorted(inner, X[:, i], side="right")
        n_bins_of_col.append(len(inner) + 1)

    def estimate_s1(rows: np.ndarray) -> np.ndarray:
        yy = y[rows]
        v = float(np.var(yy))
        out = np.full(d, np.nan)
        if not v > 0.0:
            return out
        m, n = float(np.mean(yy)), yy.size
        for i in range(d):
            b = bin_of_row[rows, i]
            cnt = np.bincount(b, minlength=n_bins_of_col[i]).astype(float)
            s = np.bincount(b, weights=yy, minlength=n_bins_of_col[i])
            s2 = np.bincount(b, weights=yy * yy, minlength=n_bins_of_col[i])
            k = cnt > 0
            mk = s[k] / cnt[k]
            between = float(np.sum(cnt[k] * (mk - m) ** 2)) / n
            var_k = np.where(cnt[k] > 1, (s2[k] - cnt[k] * mk ** 2) / np.maximum(cnt[k] - 1, 1), 0.0)
            out[i] = (between - float(np.sum(var_k)) / n) / v
        return out

    s1 = estimate_s1(np.arange(N))
    rng = np.random.default_rng(seed)
    boot = (np.array([estimate_s1(rng.integers(0, N, N)) for _ in range(int(n_boot))]) if n_boot
            else np.full((0, d), np.nan))
    lo_pct, hi_pct = (1.0 - conf) / 2.0 * 100.0, (1.0 + conf) / 2.0 * 100.0
    with _warnings_ignored():
        ci = np.stack([np.nanpercentile(boot, lo_pct, axis=0), np.nanpercentile(boot, hi_pct, axis=0)], axis=1) \
            if n_boot else np.full((d, 2), np.nan)
    return {"s1": s1, "s1_ci": ci, "n": int(N), "n_bins": n_bins_of_col}


# ishigami test function of u in [0, 1]^3, with x = -pi + 2 pi u — sin x1 + a sin^2 x2 + b x3^4 sin x1.
def ishigami(u: np.ndarray, a: float = 7.0, b: float = 0.1) -> np.ndarray:
    x = -math.pi + 2.0 * math.pi * np.asarray(u, float)
    return np.sin(x[:, 0]) + a * np.sin(x[:, 1]) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])


# analytic first-order and total Sobol indices of `ishigami` (uniform inputs on [-pi, pi]).
def ishigami_indices(a: float = 7.0, b: float = 0.1) -> dict:
    p4, p8 = math.pi ** 4, math.pi ** 8
    v1 = 0.5 * (1.0 + b * p4 / 5.0) ** 2
    v2 = a * a / 8.0
    v13 = 8.0 * b * b * p8 / 225.0
    v = a * a / 8.0 + b * p4 / 5.0 + b * b * p8 / 18.0 + 0.5
    return {"s1": np.array([v1 / v, v2 / v, 0.0]), "st": np.array([(v1 + v13) / v, v2 / v, v13 / v]),
            "variance": v}


# (S1 (d,), ST (d,), V) by the formulas of the module comments.
# a zero variance gives NaN indices.
def _saltelli_jansen(fA, fB, fAB) -> tuple[np.ndarray, np.ndarray, float]:
    variance = float(np.var(np.concatenate([fA, fB])))
    if not variance > 0.0:
        d = fAB.shape[0]
        return np.full(d, np.nan), np.full(d, np.nan), variance
    s1 = np.mean(fB[None, :] * (fAB - fA[None, :]), axis=1) / variance
    st = 0.5 * np.mean((fA[None, :] - fAB) ** 2, axis=1) / variance
    return s1, st, variance


# silence numpy floating-point warnings and RuntimeWarnings (an all-NaN bootstrap column, for example).
@contextlib.contextmanager
def _warnings_ignored() -> Iterator[None]:
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        yield


# the uncertainty set as u-columns
# one `uncertainty` entry as an axis of u in [0, 1] (the map that realize applies in lhs mode).
# `kind` is "hypothesis", "interval" or "dist".
# `target` is the entry's `target` field — the config path it sets, or None (for example a hypothesis whose choices name their own paths).
# `pinned` is the pinned value or choice ("follows <leader>" for a label entry, which realize sets from its leader), None when realize draws the entry.
# `fields` are the entry's fields as in the preset.
@dataclass
class EntryAxis:
    name: str
    kind: str
    target: str | None
    pinned: Any
    fields: dict
    choices: tuple = ()            # hypothesis - every choice, in realize's order
    weights: tuple = ()            # hypothesis - normalized weights (0 outside the allowed set)
    cdf: tuple = ()                # hypothesis - cumulative weights (the u bin edges of the choices)
    nominal: Any = None            # hypothesis - choice name - interval/dist - value (config units) or None

    # the hypothesis choices that the sampled set can reach (positive weight).
    @property
    def support(self) -> tuple:
        return tuple(c for c, w in zip(self.choices, self.weights) if w > 0)

    # the draw realize makes from u — a choice name, or a value in config units.
    def value_of_u(self, u: float) -> Any:
        u = float(np.clip(u, LHS_CLIP, 1.0 - LHS_CLIP))
        if self.kind == "hypothesis":
            idx = min(int(np.searchsorted(np.asarray(self.cdf), u, side="right")), len(self.choices) - 1)
            return self.choices[idx]
        return float(_interval_x(self.fields, self.kind, np.array([u]))[0])

    # a u that realize maps to `value` — the middle of the choice's bin, or the inverse CDF of a number.
    def u_of_value(self, value: Any) -> float:
        if self.kind == "hypothesis":
            choice = str(value)
            if choice not in self.choices:
                raise ValueError(f"uncertainty.{self.name}: {choice!r} is not a choice of {self.choices}")
            i = self.choices.index(choice)
            if self.weights[i] <= 0:
                raise ValueError(f"uncertainty.{self.name}: choice {choice!r} has zero weight (a stress corner); "
                                 "the sampled set cannot reach it")
            lo = self.cdf[i - 1] if i > 0 else 0.0
            return 0.5 * (lo + self.cdf[i])
        return float(_interval_u(self.fields, self.kind, float(value)))

    # (label, u) of the one-at-a-time points — u = 0 and 1 (the ends) for an interval.
    # u = 0.025 and 0.975 for a dist entry, truncated or not.
    # every reachable choice other than the nominal for a hypothesis.
    def tornado_points(self) -> list[tuple[str, float]]:
        if self.kind == "hypothesis":
            return [(c, self.u_of_value(c)) for c in self.support if c != self.nominal]
        if self.kind == "interval":
            return [("low", 0.0), ("high", 1.0)]
        return [("p2.5", 0.025), ("p97.5", 0.975)]


# the `uncertainty` entries of a Config as the u-columns of realize's Latin hypercube.
# Column i of a u matrix belongs to `names[i]` (the entry names, sorted, as in realize).
# `axes` maps each name to its `EntryAxis`.
# the property `d` is the number of columns (D in the comments).
class UncertaintySpace:

    def __init__(self, cfg):
        from mrm.params.config import _hypothesis_choice_map
        self.cfg = cfg
        self.names: list[str] = sorted(cfg.uncertainty)
        self.axes: dict[str, EntryAxis] = {}
        for name in self.names:
            entry = cfg.uncertainty[name]
            fields = dict(entry.fields)
            if entry.kind == "hypothesis":
                choices = tuple(_hypothesis_choice_map(entry))
                allowed = entry.restrict or list(choices)
                weight_of = {str(k): v for k, v in (fields.get("weights") or {}).items()}
                weights = np.array([float(weight_of.get(c, 1.0)) if c in allowed else 0.0 for c in choices])
                if weights.sum() <= 0:
                    raise ValueError(f"uncertainty.{name}: weights sum to zero")
                weights = weights / weights.sum()
                nominal = str(fields.get("nominal"))
                nominal = nominal if nominal in allowed else allowed[0]
                # a label entry (follows - + map:) is not drawn - realize sets it from its leader.
                # so it is not an axis of u - the leader's axis carries the joint hypothesis.
                pinned = entry.pinned if entry.follows is None else (entry.pinned or f"follows {entry.follows}")
                self.axes[name] = EntryAxis(name, "hypothesis", entry.target, pinned, fields, choices,
                                            tuple(weights), tuple(np.cumsum(weights)), nominal)
            else:
                kind = "interval" if entry.kind == "interval" else "dist"
                self.axes[name] = EntryAxis(name, kind, entry.target, entry.pinned, fields,
                                            nominal=self._numeric_nominal(entry))

    # nominal value of an interval or dist entry (config units) — its `nominal` field.
    # else None for an entry without a target.
    # else 1 (mode scale), 0 (mode offset), or, for mode absolute, the target leaf's value when all its elements are equal.
    # else None.
    def _numeric_nominal(self, entry) -> float | None:
        nominal = entry.fields.get("nominal")
        if isinstance(nominal, (int, float)):
            return float(nominal)
        if entry.target is None:
            return None
        mode = entry.fields.get("mode", "absolute")
        if mode == "scale":
            return 1.0
        if mode == "offset":
            return 0.0
        values = np.asarray(self.cfg.get(entry.target), float).ravel()   # absolute without a nominal - the leaf keeps it
        if values.size and np.allclose(values, values[0], rtol=0, atol=0):
            return float(values[0])
        return None

    # number of entries (columns of a u matrix).
    @property
    def d(self) -> int:
        return len(self.names)

    # the entries that realize draws (not pinned).
    @property
    def free(self) -> list[str]:
        return [n for n in self.names if self.axes[n].pinned is None]

    # (D,) u of the nominal world.
    # a pinned entry gets 0.5 (realize ignores it).
    # raises for a drawn entry without a nominal that a u can express.
    def nominal_u(self) -> np.ndarray:
        uniforms = np.full(self.d, 0.5)
        for i, n in enumerate(self.names):
            axis = self.axes[n]
            if axis.pinned is not None:
                continue
            if axis.nominal is None:
                raise ValueError(f"uncertainty.{n} has no nominal value a u can express")
            uniforms[i] = axis.u_of_value(axis.nominal)
        return uniforms

    # (B, D) u matrix — `base` (D,) in every row, then the given columns ({name — (B,)}) replaced.
    def design(self, base: np.ndarray, columns: Mapping[str, np.ndarray]) -> np.ndarray:
        sizes = {np.asarray(v).size for v in columns.values()} or {1}
        if len(sizes) != 1:
            raise ValueError("every column must have the same length")
        n_rows = sizes.pop()
        U = np.tile(np.asarray(base, float)[None, :], (n_rows, 1))
        for n, col in columns.items():
            U[:, self.names.index(n)] = np.asarray(col, float).ravel()
        return U

    # {name — [draw per row]} that realize should produce from `U` (choice names or values).
    def expected_world(self, U: np.ndarray) -> dict[str, list]:
        return {n: [self.axes[n].value_of_u(u) for u in U[:, i]] for i, n in enumerate(self.names)
                if self.axes[n].pinned is None}

    # readable value of entry `name` at u — a choice name, or a number with 4 significant digits.
    def describe(self, name: str, u: float) -> str:
        v = self.axes[name].value_of_u(u)
        return v if isinstance(v, str) else f"{v:.4g}"


# make `mrm.params.realize(..., uncertainty='lhs')` take its Latin-hypercube matrix from `U` (B, D).
# Inside the context, realize gets `U` in place of `scipy.stats.qmc.LatinHypercube(d=D, seed).random(B)`.
# a call with another d or B raises, so an unexpected realize inside the context is loud, not silently wrong.
@contextlib.contextmanager
def injected_lhs(U: np.ndarray) -> Iterator[None]:
    from scipy.stats import qmc
    U = np.array(U, float)
    if U.ndim != 2:
        raise ValueError("U must be (B, D)")
    original = qmc.LatinHypercube

    class _FixedDesign:
        def __init__(self, d, *args, **kwargs):
            if int(d) != U.shape[1]:
                raise RuntimeError(f"injected_lhs: realize asked for d = {d}, the design has {U.shape[1]} columns")

        def random(self, n=1, *args, **kwargs):
            if int(n) != U.shape[0]:
                raise RuntimeError(f"injected_lhs: realize asked for {n} rows, the design has {U.shape[0]}")
            return U.copy()

    qmc.LatinHypercube = _FixedDesign
    try:
        yield
    finally:
        qmc.LatinHypercube = original


# realize `space.cfg` with `U` injected and compare each world's draws with the intended values.
# returns {"ok" — bool, "mismatches" — [str] (the first 20), "n_mismatches" — int, "worlds" — B}.
# it reads `PlantParams.world`, which holds a hypothesis as a choice index and an interval as a value.
def check_injection(space: UncertaintySpace, U: np.ndarray, *, rtol: float = 1e-9) -> dict:
    from mrm import params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with injected_lhs(U):
            plant = params.realize(space.cfg, U.shape[0], seeds=0, uncertainty="lhs", variation="off")
    expected = space.expected_world(U)
    mismatches = []
    for n, values in expected.items():
        draws = np.asarray(plant.world[n])
        axis = space.axes[n]
        for b, v in enumerate(values):
            if axis.kind == "hypothesis":
                drawn = plant.world_choices[n][int(draws[b])]
                if drawn != v:
                    mismatches.append(f"{n}[{b}]: realize drew {drawn}, intended {v}")
            elif not np.isclose(float(draws[b]), float(v), rtol=rtol, atol=0.0):
                mismatches.append(f"{n}[{b}]: realize drew {float(draws[b])!r}, intended {v!r}")
    return {"ok": not mismatches, "mismatches": mismatches[:20], "n_mismatches": len(mismatches),
        "worlds": int(U.shape[0])}


# values of an interval or dist entry with fields `f` at the uniforms `u`.
# a copy of `mrm.params.realize._interval_x` (same formulas. tests/verification/test_sensitivity.py keeps the two in step).
# Interval — uniform, loguniform, or triangular with the mode at the nominal (or the midpoint).
# Dist — normal or lognormal, optionally truncated exactly through the inverse CDF.
def _interval_x(f: Mapping, kind: str, u: np.ndarray) -> np.ndarray:
    if kind == "interval":
        lo, hi = float(f["low"]), float(f["high"])
        distribution = f.get("dist", "uniform")
        if distribution == "uniform":
            return lo + (hi - lo) * u
        if distribution == "loguniform":
            return np.exp(np.log(lo) + (np.log(hi) - np.log(lo)) * u)
        mode = float(f.get("nominal", 0.5 * (lo + hi))) if isinstance(f.get("nominal"), (int,
            float)) else 0.5 * (lo + hi)
        mode_cdf = (mode - lo) / (hi - lo) if hi > lo else 0.5
        return np.where(u < mode_cdf, lo + np.sqrt(u * (hi - lo) * (mode - lo)),
            hi - np.sqrt((1 - u) * (hi - lo) * (hi - mode)))
    mean, sigma = float(f["mean"]), float(f["sigma"])
    lognormal = f.get("dist", "normal") == "lognormal"
    truncation = f.get("truncate")
    if truncation:
        a, b = _truncation_z(f, mean, sigma, lognormal)
        u = ndtr(a) + u * (ndtr(b) - ndtr(a))
    z = ndtri(np.clip(u, 1e-300, 1 - 1e-16))
    return mean * np.exp(sigma * z) if lognormal else mean + sigma * z


# the `truncate: [lo, hi]` bounds of a dist entry in z units.
# Normal — x = mean + sigma z.
# Lognormal — x = mean exp(sigma z), so z = ln(x / mean) / sigma (a lower bound <= 0 is no bound).
def _truncation_z(f: Mapping, mean: float, sigma: float, lognormal: bool) -> tuple[float, float]:
    lo_hi = np.asarray(f["truncate"], float)
    if lognormal:
        with np.errstate(divide="ignore"):
            zb = np.log(np.maximum(lo_hi, 0.0) / mean) / max(sigma, 1e-300)
    else:
        zb = (lo_hi - mean) / max(sigma, 1e-300)
    return float(zb[0]), float(zb[1])


# inverse of `_interval_x` — the u that gives the value `x` (config units).
def _interval_u(f: Mapping, kind: str, x: float) -> float:
    if kind == "interval":
        lo, hi = float(f["low"]), float(f["high"])
        if not lo <= x <= hi:
            raise ValueError(f"value {x} outside [{lo}, {hi}]")
        if hi == lo:
            return 0.5
        distribution = f.get("dist", "uniform")
        if distribution == "uniform":
            return (x - lo) / (hi - lo)
        if distribution == "loguniform":
            return (math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo))
        mode = float(f.get("nominal", 0.5 * (lo + hi))) if isinstance(f.get("nominal"), (int,
            float)) else 0.5 * (lo + hi)
        if x < mode:
            return (x - lo) ** 2 / ((hi - lo) * (mode - lo))
        return 1.0 - (hi - x) ** 2 / ((hi - lo) * (hi - mode)) if hi > mode else 1.0
    mean, sigma = float(f["mean"]), float(f["sigma"])
    lognormal = f.get("dist", "normal") == "lognormal"
    z = math.log(x / mean) / sigma if lognormal else (x - mean) / sigma
    u = float(ndtr(z))
    if f.get("truncate"):
        a, b = _truncation_z(f, mean, sigma, lognormal)
        u = (u - float(ndtr(a))) / (float(ndtr(b)) - float(ndtr(a)))
    return u


# tornado (one entry at a time)
# one-at-a-time swings of each metric.
# `points` — (entry, label, {metric — value}) for every tornado world.
# `base` — {metric — value in the nominal world}.
# returns {metric — [{entry, lo, hi, lo_label, hi_label, swing, values, labels}]}, sorted by swing, largest first.
# lo and hi are the smallest and largest values over the entry's points and the base, so the base is always inside the bar.
# a non-finite value is kept (its label, for example `inf`) and makes the swing infinite.
# an entry
# whose values are all NaN (the metric is undefined at each of its points) keeps lo = hi = swing = NaN.
def tornado_table(base: Mapping[str, float], points: Sequence[tuple[str, str, Mapping[str, float]]],
                  metrics: Sequence[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for metric in metrics:
        rows: dict[str, dict] = {}
        for entry, label, vals in points:
            v = float(vals.get(metric, np.nan))
            r = rows.setdefault(entry, {"entry": entry, "values": [], "labels": []})
            r["values"].append(v)
            r["labels"].append(label)
        nominal = float(base.get(metric, np.nan))
        metric_rows = []
        for entry, r in rows.items():
            values = np.asarray(r["values"], float)
            labels = list(r["labels"])
            i_lo = int(np.nanargmin(np.where(np.isnan(values), np.inf, values))) if np.any(~np.isnan(values)) else 0
            i_hi = int(np.nanargmax(np.where(np.isnan(values), -np.inf, values))) if np.any(~np.isnan(values)) else 0
            lo, hi = float(values[i_lo]), float(values[i_hi])
            lo_label, hi_label = labels[i_lo], labels[i_hi]
            # widen the bar to the base - an all-NaN entry (lo = hi = NaN) stays NaN
            if np.isfinite(nominal) and lo > nominal:
                lo, lo_label = nominal, "nominal"
            if np.isfinite(nominal) and hi < nominal:
                hi, hi_label = nominal, "nominal"
            with np.errstate(invalid="ignore"):
                swing = hi - lo if not (np.isnan(lo) or np.isnan(hi)) else np.nan
            metric_rows.append({"entry": entry, "lo": lo, "hi": hi, "lo_label": lo_label, "hi_label": hi_label,
                        "swing": float(swing), "values": [float(x) for x in values], "labels": labels})
        # largest swing first - +inf before every finite swing, NaN last - ties by entry name
        metric_rows.sort(key=lambda r: (-(r["swing"] if np.isfinite(r["swing"]) else (np.inf if r["swing"] > 0 else -1)),
                                r["entry"]))
        out[metric] = metric_rows
    return out


# the k entries with the largest normalized swing.
# Per metric, the normalized swing is swing / (largest finite swing of that metric).
# an infinite swing counts as 1.
# the score of an entry is its largest normalized swing over the metrics (ties by name).
def screen_top(tornado: Mapping[str, Sequence[Mapping]], k: int = 8) -> list[str]:
    score: dict[str, float] = {}
    for m, rows in tornado.items():
        finite_swings = [r["swing"] for r in rows if np.isfinite(r["swing"]) and r["swing"] > 0]
        ref = max(finite_swings) if finite_swings else 0.0
        for r in rows:
            s = r["swing"]
            x = 1.0 if (np.isinf(s) and s > 0) else (s / ref if ref > 0 and np.isfinite(s) else 0.0)
            score[r["entry"]] = max(score.get(r["entry"], 0.0), x)
    return [e for e, _ in sorted(score.items(), key=lambda t: (-t[1], t[0]))[:k]]


# identifiability
# sVD of a normalized Jacobian and the linearized half-prior rule.
# `J` (n_residuals, n_params) — d(residual / sigma) / d(parameter / prior half-width).
# Rows with a non-finite entry are dropped.
# a singular value at or below `rcond` times the largest is a null direction.
# returns —
# `s` — singular values, descending.
# `condition` — s_max / s_min (inf when singular).
# `rank`.
# `directions` — [{s, identified (s >= `HALF_PRIOR_SINGULAR_VALUE`), loadings — the `top` (name, v) with the largest |v| first}].
# Per parameter — `half_width_95` = 1.96 sqrt((J^T J)^+_jj), inf when the parameter loads on a null direction.
# `interval_frac_of_prior` = 95 % interval width / prior width (equal to half_width_95, because the prior half-width is 1).
# `half_prior_rule` (the interval is at most half the prior range).
# and `column_norm` (the sensitivity of each parameter alone).
def identifiability(J: np.ndarray, names: Sequence[str], *, rcond: float = 1e-10, top: int = 4) -> dict:
    J = np.asarray(J, float)
    names = list(names)
    if J.ndim != 2 or J.shape[1] != len(names):
        raise ValueError("J must be (n_residuals, n_params) with one name per column")
    good_rows = np.all(np.isfinite(J), axis=1)
    J_good = J[good_rows]
    _, s, Vt = (np.linalg.svd(J_good, full_matrices=False) if J_good.size
                else (None, np.zeros(0), np.zeros((0, J.shape[1]))))
    n_params = J.shape[1]
    s_full = np.zeros(n_params)
    s_full[:s.size] = s
    if s.size < n_params:                                # fewer residuals than parameters - extra null directions
        _, _, Vt_full = np.linalg.svd(np.vstack([J_good, np.zeros((n_params - J_good.shape[0], n_params))]),
                                      full_matrices=True)
        Vt = Vt_full
    s_max = float(s_full.max()) if s_full.size else 0.0
    null_directions = s_full <= rcond * max(s_max, 1e-300)
    inv_s2 = np.where(null_directions, 0.0, 1.0 / np.where(null_directions, 1.0, s_full) ** 2)
    covariance = (Vt.T * inv_s2[None, :]) @ Vt                  # pseudo-inverse of J^T J
    half_width = _Z_95 * np.sqrt(np.maximum(np.diag(covariance), 0.0))
    if null_directions.any():
        null_loading = np.max(np.abs(Vt[null_directions, :]), axis=0)
        half_width = np.where(null_loading > 1e-6, np.inf, half_width)
    directions = []
    for k in range(n_params):
        v = Vt[k]
        order = np.argsort(-np.abs(v))[:top]
        directions.append({"s": float(s_full[k]), "identified": bool(s_full[k] >= HALF_PRIOR_SINGULAR_VALUE),
                           "loadings": [(names[j], float(v[j])) for j in order]})
    with np.errstate(divide="ignore"):
        condition = float(s_max / s_full.min()) if s_full.size and s_full.min() > 0 else float("inf")
    return {"names": names, "s": [float(x) for x in s_full], "condition": condition,
            "rank": int(np.sum(~null_directions)), "n_residuals_used": int(good_rows.sum()),
            "n_residuals_dropped": int((~good_rows).sum()), "directions": directions,
            "half_width_95": [float(x) for x in half_width], "interval_frac_of_prior": [float(x) for x in half_width],
            "half_prior_rule": [bool(x <= 0.5) for x in half_width],
            "column_norm": [float(x) for x in np.linalg.norm(np.where(np.isfinite(J), J, 0.0), axis=0)],
            "threshold_singular_value": HALF_PRIOR_SINGULAR_VALUE}
