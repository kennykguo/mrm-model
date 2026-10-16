"""Fitting machinery of the calibration stage (07 P0-5, section 4.3): least squares, profile likelihoods, the
half-prior rule and hypothesis likelihood ratios.

Owner: calibration. Contract: docs/research/07_design_critique.md ("07") P0-5 and 4.3; the calibration targets and
their README (``mrm/validation/targets``). This module reads targets only through
:func:`mrm.validation.targets_io.load_calibration_targets` and predicts them only through the replicas of
:mod:`mrm.validation.replicas` (no other forward model).

Statistics
----------
For one hypothesis (a fixed choice of every discrete alternative) a :class:`Problem` holds the free parameters
(:class:`Param`, each with a declared prior range) and a prediction function theta -> {target id: prediction}. The
residual vector stacks, per target, ``(prediction - value) / sigma`` (:func:`target_residuals`):

- scalar, curve and set targets with a sigma: one standardized residual per published number;
- an interval target with ``sigma: null`` (CAL-1, CAL-11) is a flat likelihood: 0 inside the interval and a steep
  wall outside it (:data:`INTERVAL_WALL_FRAC` of the interval width per unit of residual), so its profile is the
  interval itself;
- an interval target with a sigma (CAL-9a, CAL-9c) compares the predicted [lo, hi] with the printed [lo, hi].

chi2 = sum of squared residuals. The degrees of freedom (Birge ratio below) count only residuals that carry a
sigma: a sigma-less interval target is a constraint (a flat prior on the prediction), not a datum. The fit is ``scipy.optimize.least_squares`` (trust region, bounded to the prior
range, finite differences) in normalized coordinates u in [0, 1] of each prior range (linear, or logarithmic for a
scale parameter), from every listed start; distinct local minima are kept (a mirror branch is a second minimum).

Profile likelihood of parameter i: chi2 minimized over the other parameters on a grid of u_i covering the whole
prior range (warm-started along the grid). The 95 % profile interval is {theta_i : chi2_prof - chi2_min <= 3.84 s^2}
with the Birge ratio s^2 = max(1, chi2_min / dof): when the model cannot reach the data within their sigmas (a
model-form misfit), the interval is widened by the misfit instead of shrinking around a wrong value. Edges are
interpolated linearly between grid points; an interval that reaches a prior bound is open there.

Half-prior rule (07 4.3): a leaf may be tagged ``calibrated`` only when the width of its 95 % profile interval is at
most half the width of its prior range, measured in the coordinate of the prior (log for a log-scale parameter);
otherwise it keeps ``assumed`` (or its own weaker provenance), with the narrowed interval recorded in its note.
``unique`` means one connected interval, closed on both sides, and no second local minimum within the threshold.

Hypotheses (discrete alternatives, e.g. the coupling regime or the extinction mechanism) are compared by their best
chi2: likelihood ratio LR = exp(-(chi2_a - chi2_best)/2), also with the Birge-scaled chi2 of the best hypothesis and
with AIC (2k + chi2) when the number of free parameters differs. A ratio is evidence between the listed hypotheses
only; no alternative is deleted.
"""
from __future__ import annotations

import dataclasses
import math
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy import optimize

from mrm.validation.targets_io import Target, evaluate, load_calibration_targets

__all__ = ["Param", "Problem", "FitResult", "Profile", "HypothesisResult", "calibration_targets",
           "target_residuals", "n_data", "fit", "profile", "half_prior", "compare_hypotheses", "INTERVAL_WALL_FRAC",
           "DCHI2_95", "jsonable"]

#: chi2 increase of a 95 % profile interval (one parameter of interest)
DCHI2_95 = 3.841458820694124
#: a value outside a sigma-less interval target costs one unit of residual per this fraction of the interval width
INTERVAL_WALL_FRAC = 0.005


def calibration_targets() -> dict[str, Target]:
    """Every calibration target by id (the only target file this package reads, 07 P0-5)."""
    return {t.id: t for t in load_calibration_targets()}


def jsonable(x: Any) -> Any:
    """Plain JSON types (numpy scalars and arrays, tuples, dataclasses, non-finite floats as None)."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return jsonable(dataclasses.asdict(x))
    if isinstance(x, Mapping):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# =====================================================================================================================
# Parameters
# =====================================================================================================================
@dataclasses.dataclass(frozen=True)
class Param:
    """One free parameter of a fit.

    name: short name used in results. path: the preset leaf it sets (an override path), or None for a nuisance that
    is not a preset leaf (a measurement floor, the swept device's own efficiency). lo, hi: the prior range (also the
    fit bounds), in the leaf's own unit; ``prior_source`` says where the range comes from (the leaf's ``range``, an
    uncertainty interval, or a stated assumption). log: fit and judge the half-prior rule in log coordinates (a
    positive scale parameter with a multiplicative prior). role: ``leaf`` (written to the calibrated preset) or
    ``nuisance`` (fitted alongside, never written). declared_prior: the range is a prior declared before the fit
    (the leaf's own ``range`` or an uncertainty interval of the preset); False when the calibration had to invent
    fit bounds for a leaf that declares none, in which case the half-prior rule cannot be applied (the width of an
    invented range would decide the tag) and the leaf is never tagged ``calibrated``."""

    name: str
    path: str | None
    lo: float
    hi: float
    start: float
    unit: str = ""
    log: bool = False
    role: str = "leaf"
    prior_source: str = ""
    declared_prior: bool = True

    def __post_init__(self):
        if not self.hi > self.lo:
            raise ValueError(f"{self.name}: prior range must have hi > lo")
        if self.log and self.lo <= 0:
            raise ValueError(f"{self.name}: a log parameter needs a positive range")
        if self.role not in ("leaf", "nuisance"):
            raise ValueError(f"{self.name}: role must be leaf or nuisance")

    def to_u(self, x: float) -> float:
        """Normalized coordinate in [0, 1] of the prior range."""
        if self.log:
            return (math.log(x) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (x - self.lo) / (self.hi - self.lo)

    def from_u(self, u: float) -> float:
        u = min(max(float(u), 0.0), 1.0)
        if self.log:
            x = math.exp(math.log(self.lo) + u * (math.log(self.hi) - math.log(self.lo)))
        else:
            x = self.lo + u * (self.hi - self.lo)
        return float(min(max(x, self.lo), self.hi))


# =====================================================================================================================
# Residuals of one target
# =====================================================================================================================
def _as_float_array(x: Any) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=float))


def n_data(targets: Sequence[Target]) -> int:
    """Number of Gaussian data in a likelihood (the residuals that carry a sigma). A sigma-less interval target
    (CAL-1, CAL-11) is a flat constraint: it contributes a residual (zero inside) but no degree of freedom."""
    n = 0
    for t in targets:
        if t.kind == "interval" and t.sigma is None:
            continue
        if t.kind == "curve":
            n += len(t.value["y"])
        elif t.kind == "set":
            n += len(t.value)
        elif t.kind == "interval":
            n += 2
        else:
            n += 1
    return n


def target_residuals(target: Target, prediction: Any, *, sigma_scale: float = 1.0) -> np.ndarray:
    """Standardized residuals (prediction - value) / sigma of one calibration target (module docstring).

    ``prediction`` is shaped as :func:`mrm.validation.targets_io.evaluate` expects (the replica's value). A
    non-finite prediction gives a large finite residual (1e3 per point) so that the optimizer moves away from it."""
    kind = target.kind
    if kind == "curve":
        y = _as_float_array(target.value["y"])
        p = _as_float_array(prediction)
        s = _as_float_array(target.sigma) * np.ones_like(y)
    elif kind == "scalar":
        y = _as_float_array(target.value)
        p = _as_float_array(prediction)
        if target.sigma is None:
            raise ValueError(f"{target.id}: a scalar target without sigma cannot be fitted")
        s = _as_float_array(target.sigma)
    elif kind == "set":
        keys = sorted(target.value)
        y = np.array([float(target.value[k]) for k in keys])
        p = np.array([float(prediction[k]) for k in keys])
        sg = target.sigma
        s = np.array([float(sg[k]) for k in keys]) if isinstance(sg, Mapping) else np.full(len(keys), float(sg))
    elif kind == "interval":
        lo, hi = (float(v) for v in target.value)
        p = _as_float_array(prediction)
        if target.sigma is None:
            # flat likelihood inside the interval, a steep wall outside; a predicted range (CAL-11) must lie inside
            wall = INTERVAL_WALL_FRAC * (hi - lo)
            r = np.where(p < lo, (p - lo) / wall, np.where(p > hi, (p - hi) / wall, 0.0))
            return np.where(np.isfinite(r), r, 1e3) / sigma_scale
        y = np.array([lo, hi])
        s = np.full(2, float(target.sigma))
    else:
        raise ValueError(f"{target.id}: kind {kind!r} is not fitted here")
    r = (p - y) / s
    return np.where(np.isfinite(r), r, 1e3) / sigma_scale


# =====================================================================================================================
# Problem, fit, profile
# =====================================================================================================================
@dataclasses.dataclass
class Problem:
    """One hypothesis of one fit group.

    params: the free parameters. targets: the targets in the likelihood. predict(theta) -> {target id: prediction}
    with theta = {param name: value}. fixed: description of everything held fixed (overrides, alternatives) for the
    record. measurement: {target id: callable(prediction, theta) -> prediction} applied after predict (a measurement
    model such as an instrument floor)."""

    group: str
    hypothesis: str
    params: list[Param]
    targets: list[Target]
    predict: Callable[[dict[str, float]], dict[str, Any]]
    fixed: dict[str, Any] = dataclasses.field(default_factory=dict)
    starts: list[dict[str, float]] = dataclasses.field(default_factory=list)
    constraint: Callable[[dict[str, float]], np.ndarray] | None = None
    derived: Callable[[dict[str, float]], dict[str, Any]] | None = None
    n_evals: int = 0
    _cache: dict = dataclasses.field(default_factory=dict, repr=False)

    def theta(self, u: Sequence[float], fixed_u: Mapping[int, float] | None = None) -> dict[str, float]:
        u = list(u)
        if fixed_u:
            full, k = [], 0
            for i in range(len(self.params)):
                if i in fixed_u:
                    full.append(fixed_u[i])
                else:
                    full.append(u[k])
                    k += 1
            u = full
        return {p.name: p.from_u(ui) for p, ui in zip(self.params, u)}

    def predictions(self, theta: Mapping[str, float]) -> dict[str, Any]:
        key = tuple(round(float(theta[p.name]), 12) for p in self.params)
        if key not in self._cache:
            self.n_evals += 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._cache[key] = self.predict(dict(theta))
            if len(self._cache) > 20000:
                self._cache.clear()
        return self._cache[key]

    def residuals(self, theta: Mapping[str, float], *, with_constraint: bool = True) -> np.ndarray:
        pred = self.predictions(theta)
        r = [target_residuals(t, pred[t.id]) for t in self.targets]
        if with_constraint and self.constraint is not None:
            r.append(np.atleast_1d(np.asarray(self.constraint(dict(theta)), dtype=float)))
        return np.concatenate(r)

    def chi2(self, theta: Mapping[str, float]) -> float:
        r = self.residuals(theta)
        return float(r @ r)


@dataclasses.dataclass
class FitResult:
    """Best fit of one hypothesis: theta, chi2, residual vector, per-target summaries and every local minimum."""

    group: str
    hypothesis: str
    theta: dict[str, float]
    chi2: float
    dof: int
    birge2: float
    residuals: list[float]
    targets: dict[str, dict]
    minima: list[dict]
    n_evals: int
    runtime_s: float
    fixed: dict[str, Any]
    derived: dict[str, Any] = dataclasses.field(default_factory=dict)
    profiles: dict[str, "Profile"] = dataclasses.field(default_factory=dict)

    def as_dict(self) -> dict:
        return jsonable(self)


def _least_squares(prob: Problem, u0: np.ndarray, fixed_u: Mapping[int, float] | None = None,
                   max_nfev: int = 200) -> tuple[np.ndarray, float]:
    free = [i for i in range(len(prob.params)) if not fixed_u or i not in fixed_u]
    if not free:
        th = prob.theta([], fixed_u)
        return np.array([]), prob.chi2(th)

    def fun(z):
        return prob.residuals(prob.theta(np.asarray(z) - 1.0, fixed_u))
    z0 = np.clip(np.asarray(u0, dtype=float), 0.0, 1.0) + 1.0
    try:
        res = optimize.least_squares(fun, z0, bounds=(np.ones(len(free)), 2.0 * np.ones(len(free))),
                                     method="trf", diff_step=2e-3, x_scale=1.0, max_nfev=max_nfev,
                                     xtol=1e-6, ftol=1e-8, gtol=1e-8)
        z = res.x
    except Exception:  # noqa: BLE001 - a failed local fit returns the start (reported by its chi2)
        z = z0
    u = z - 1.0
    return u, prob.chi2(prob.theta(u, fixed_u))


def fit(prob: Problem, *, max_nfev: int = 200, extra_starts: int = 0, seed: int = 0) -> FitResult:
    """Least-squares fit of one hypothesis from every start in ``prob.starts`` (default: each parameter's
    ``start``), plus ``extra_starts`` Latin-hypercube starts. Distinct local minima (normalized distance > 0.02) are
    kept in ``minima``, best first."""
    t0 = time.time()
    n = len(prob.params)
    starts = [np.array([p.to_u(s.get(p.name, p.start)) for p in prob.params]) for s in prob.starts] or \
        [np.array([p.to_u(p.start) for p in prob.params])]
    if extra_starts:
        from scipy.stats import qmc
        starts += list(qmc.LatinHypercube(d=n, seed=seed).random(extra_starts))
    found: list[tuple[float, np.ndarray]] = []
    for u0 in starts:
        u, c = _least_squares(prob, u0, max_nfev=max_nfev)
        if not any(np.max(np.abs(u - v)) < 0.02 for _, v in found):
            found.append((c, u))
        else:
            j = next(k for k, (_, v) in enumerate(found) if np.max(np.abs(u - v)) < 0.02)
            if c < found[j][0]:
                found[j] = (c, u)
    found.sort(key=lambda cu: cu[0])
    c_best, u_best = found[0]
    th = prob.theta(u_best)
    pred = prob.predictions(th)
    res = prob.residuals(th, with_constraint=False)
    dof = max(n_data(prob.targets) - n, 1)
    tsum = {}
    for t in prob.targets:
        r = target_residuals(t, pred[t.id])
        ev = evaluate(t, pred[t.id])
        tsum[t.id] = {"prediction": jsonable(pred[t.id]), "chi2": float(r @ r), "z_max": float(np.max(np.abs(r))),
                      "max_abs_residual": ev.max_abs_residual, "tolerance_level": ev.level,
                      "within_tolerance": ev.passed}
    minima = [{"chi2": c, "theta": prob.theta(u)} for c, u in found]
    der = prob.derived(dict(th)) if prob.derived is not None else {}
    return FitResult(group=prob.group, hypothesis=prob.hypothesis, theta=th, chi2=float(c_best), dof=int(dof),
                     birge2=max(1.0, float(c_best) / dof), residuals=res.tolist(), targets=tsum, minima=minima,
                     n_evals=prob.n_evals, runtime_s=time.time() - t0, fixed=dict(prob.fixed), derived=der)


@dataclasses.dataclass
class Profile:
    """Profile likelihood of one parameter: grid (values), chi2 (profile), the 95 % interval (Birge-widened and
    raw), and the half-prior verdict."""

    name: str
    path: str | None
    grid: list[float]
    chi2: list[float]
    chi2_min: float
    threshold: float
    interval: list[float | None]
    interval_raw: list[float | None]
    open_lo: bool
    open_hi: bool
    n_components: int
    prior: list[float]
    log: bool
    width_frac: float
    half_prior_ok: bool
    unique: bool
    best: float


def _interval_from_profile(grid: np.ndarray, chi2: np.ndarray, thr: float) -> tuple[list, bool, bool, int]:
    ok = chi2 <= thr
    if not np.any(ok):
        i = int(np.argmin(chi2))
        return [float(grid[i]), float(grid[i])], False, False, 1
    idx = np.flatnonzero(ok)
    comps = 1 + int(np.sum(np.diff(idx) > 1))
    i0, i1 = int(idx[0]), int(idx[-1])
    open_lo, open_hi = i0 == 0, i1 == len(grid) - 1

    def cross(a, b):
        ca, cb = chi2[a] - thr, chi2[b] - thr
        if not (np.isfinite(ca) and np.isfinite(cb)) or ca == cb:
            return float(grid[b])
        return float(grid[a] + (grid[b] - grid[a]) * ca / (ca - cb))
    lo = float(grid[0]) if open_lo else cross(i0 - 1, i0)
    hi = float(grid[-1]) if open_hi else cross(i1 + 1, i1)
    return [lo, hi], open_lo, open_hi, comps


def profile(prob: Problem, fr: FitResult, name: str, *, n_grid: int = 13, max_nfev: int = 60,
            refine: int = 6) -> Profile:
    """Profile likelihood of parameter ``name`` over its whole prior range (module docstring). The coarse grid is
    walked outwards from the best fit (warm starts follow the valley), re-walked from the other local minima, and
    each threshold crossing is then located by ``refine`` bisection steps (each a re-optimization of the others)."""
    i = next(k for k, p in enumerate(prob.params) if p.name == name)
    par = prob.params[i]
    u_best = np.array([p.to_u(fr.theta[p.name]) for p in prob.params])
    others = [k for k in range(len(prob.params)) if k != i]
    pts: dict[float, tuple[float, np.ndarray]] = {}        # u_i -> (chi2, u_others)

    def solve(ui: float, u_start: np.ndarray) -> tuple[float, np.ndarray]:
        u, c = _least_squares(prob, u_start, fixed_u={i: float(ui)}, max_nfev=max_nfev)
        key = round(float(ui), 12)
        if key not in pts or c < pts[key][0]:
            pts[key] = (c, u.copy() if u.size else u)
        return pts[key]
    # the best fit itself is a point of the profile (so the interval can never collapse below it when a
    # re-optimization with the inner budget stops short of it)
    pts[round(float(u_best[i]), 12)] = (float(fr.chi2), u_best[others].copy())
    grid_u = np.unique(np.concatenate([np.linspace(0.0, 1.0, n_grid), [u_best[i]]]))
    j0 = int(np.argmin(np.abs(grid_u - u_best[i])))
    for order in (range(j0, grid_u.size), range(j0, -1, -1)):
        u_prev = u_best[others].copy()
        for j in order:
            _, u = solve(grid_u[j], u_prev)
            if u.size:
                u_prev = u
    for m in fr.minima[1:3]:
        um = np.array([p.to_u(m["theta"][p.name]) for p in prob.params])
        for j in range(grid_u.size):
            solve(grid_u[j], um[others])
    chi_min = min(min(c for c, _ in pts.values()), fr.chi2)
    thr = chi_min + DCHI2_95 * fr.birge2
    thr_raw = chi_min + DCHI2_95
    # bisection of every crossing of either threshold between neighbouring grid points
    for level in (thr, thr_raw):
        for _ in range(refine):
            us = sorted(pts)
            cs = [pts[u][0] for u in us]
            new = []
            for a, b in zip(range(len(us) - 1), range(1, len(us))):
                if (cs[a] <= level) != (cs[b] <= level) and us[b] - us[a] > 1e-4:
                    inside = a if cs[a] <= level else b
                    new.append((0.5 * (us[a] + us[b]), pts[us[inside]][1]))
            if not new:
                break
            for um, ustart in new:
                solve(um, ustart if ustart.size else u_best[others])
    us = np.array(sorted(pts))
    chi = np.array([pts[u][0] for u in us])
    grid = np.array([par.from_u(u) for u in us])
    iv, olo, ohi, comps = _interval_from_profile(grid, chi, thr)
    iv_raw, _, _, _ = _interval_from_profile(grid, chi, thr_raw)
    wf = half_prior(par, iv)
    second = [m for m in fr.minima[1:] if m["chi2"] <= thr]
    unique = comps == 1 and not olo and not ohi and not second
    return Profile(name=name, path=par.path, grid=grid.tolist(), chi2=chi.tolist(), chi2_min=chi_min,
                   threshold=thr, interval=iv, interval_raw=iv_raw, open_lo=olo, open_hi=ohi, n_components=comps,
                   prior=[par.lo, par.hi], log=par.log, width_frac=wf, half_prior_ok=wf <= 0.5, unique=unique,
                   best=fr.theta[name])


def half_prior(par: Param, interval: Sequence[float]) -> float:
    """Width of ``interval`` as a fraction of the prior range of ``par`` (log coordinates for a log parameter)."""
    lo, hi = float(interval[0]), float(interval[1])
    if par.log:
        return (math.log(max(hi, par.lo)) - math.log(max(lo, par.lo))) / (math.log(par.hi) - math.log(par.lo))
    return (hi - lo) / (par.hi - par.lo)


# =====================================================================================================================
# Hypotheses
# =====================================================================================================================
@dataclasses.dataclass
class HypothesisResult:
    """Comparison of the hypotheses of one group: best chi2 per hypothesis, likelihood ratios against the best
    (raw and with the best hypothesis' Birge ratio), AIC differences."""

    group: str
    best: str
    rows: list[dict]
    note: str


def compare_hypotheses(group: str, fits: Sequence[FitResult], note: str = "") -> HypothesisResult:
    best = min(fits, key=lambda f: f.chi2)
    s2 = best.birge2
    rows = []
    for f in sorted(fits, key=lambda f: f.chi2):
        d = f.chi2 - best.chi2
        k = len(f.theta)
        aic = 2 * k + f.chi2
        rows.append({"hypothesis": f.hypothesis, "chi2": f.chi2, "dof": f.dof, "n_params": k,
                     "delta_chi2": d, "likelihood_ratio": math.exp(-0.5 * d),
                     "likelihood_ratio_birge": math.exp(-0.5 * d / s2), "aic": aic,
                     "delta_aic": aic - (2 * len(best.theta) + best.chi2)})
    return HypothesisResult(group=group, best=best.hypothesis, rows=rows, note=note)


def restrict(prob: Problem, fixed: Mapping[str, float], *, hypothesis: str | None = None) -> Problem:
    """The same likelihood with some parameters held at given values (a conditional fit: e.g. one alternative's
    own leaf refitted with every other leaf at the nominal fit, as the calibrated preset composes its worlds)."""
    fixed = {k: float(v) for k, v in fixed.items()}
    params = [p for p in prob.params if p.name not in fixed]

    def predict(th):
        return prob.predict({**fixed, **th})
    cons = None if prob.constraint is None else (lambda th: prob.constraint({**fixed, **th}))
    der = None if prob.derived is None else (lambda th: prob.derived({**fixed, **th}))
    return Problem(group=prob.group, hypothesis=hypothesis or f"{prob.hypothesis} | held {sorted(fixed)}",
                   params=params, targets=prob.targets, predict=predict, fixed={**prob.fixed, **fixed},
                   starts=[{k: v for k, v in s.items() if k not in fixed} for s in prob.starts],
                   constraint=cons, derived=der)
