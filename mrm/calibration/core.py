# fit continuous parameters and measure how strongly calibration data constrain them.
# `Problem` holds one fixed hypothesis, its free parameters, targets, and replica prediction function.
# `fit` minimizes squared residuals from several starts.
# parameters use normalized coordinates in [0, 1], with log coordinates for positive scale parameters.
# distinct local minima are retained.
# with a sigma, it is (prediction - value) / sigma.
# curve points can share a read-off error, so their residuals are whitened with the full covariance.
# a sigma-less interval instead has zero cost inside and a steep wall outside.
# it constrains the prediction but adds no Gaussian data to the degrees of freedom.
# `profile` holds one parameter fixed and re-fits all others across its range.
# accept points with chi2 - chi2_min <= 3.84 x max(1, chi2_min / dof).
# the Birge factor widens intervals when residuals exceed their stated sigmas.
# interpolate threshold crossings and mark intervals open at prior bounds.
# `half_prior` compares interval width with prior width, using log widths when needed and the narrower declared width if bounds were widened.
# this width test is only part of the calibration policy in pipeline.py.
# connectedness, endpoints, a declared prior, and target tolerances also matter.
# `compare_hypotheses` reports likelihood ratios and AIC among the listed hypotheses.
# these comparisons do not establish that any hypothesis is correct.
# this package loads calibration targets only and predicts them through replicas.
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

# chi2 increase that bounds a 95 % profile interval of one parameter of interest (chi-square quantile, 1 dof)
DCHI2_95 = 3.841458820694124
# wall of a sigma-less interval target - a value outside the interval costs one unit of residual per this fraction of the interval width
INTERVAL_WALL_FRAC = 0.005
# residual given to each point of a non-finite prediction (large but finite, so the optimizer moves away from it)
_NONFINITE_RESIDUAL = 1e3
# two local minima are distinct when their normalized coordinates u differ by more than this in some parameter
_DISTINCT_MINIMUM_DU = 0.02
# the prediction cache of a Problem is cleared when it holds more entries than this
_CACHE_MAX_ENTRIES = 20000


# every calibration target by id.
# the calibration file is the only target file this package reads.
def calibration_targets() -> dict[str, Target]:
    return {t.id: t for t in load_calibration_targets()}


# plain JSON types (numpy scalars and arrays, tuples, dataclasses, non-finite floats as None).
def jsonable(x: Any) -> Any:
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


# Parameters
# one free fit parameter and its prior information.
# `name` identifies it in the fit.
# `path` names a preset leaf, or is None for a nuisance.
# lo, hi, and start use the leaf's own unit.
# `log` selects log coordinates for fitting and interval widths.
# `declared_prior=False` means bounds were invented for fitting, so their width cannot justify a calibrated label.
# `declared_range` retains a narrower original range when fit bounds were widened.
# `role` separates written leaves from nuisances fitted only to describe the measurement.
@dataclasses.dataclass(frozen=True)
class Param:

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
    declared_range: tuple[float, float] | None = None

    def __post_init__(self):
        if not self.hi > self.lo:
            raise ValueError(f"{self.name}: prior range must have hi > lo")
        if self.declared_range is not None and not self.declared_range[1] > self.declared_range[0]:
            raise ValueError(f"{self.name}: declared range must have hi > lo")
        if self.log and self.lo <= 0:
            raise ValueError(f"{self.name}: a log parameter needs a positive range")
        if self.role not in ("leaf", "nuisance"):
            raise ValueError(f"{self.name}: role must be leaf or nuisance")

    # normalized coordinate in [0, 1] of the prior range (log coordinates for a log parameter).
    def to_u(self, x: float) -> float:
        if self.log:
            return (math.log(x) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (x - self.lo) / (self.hi - self.lo)

    # the parameter value (its own unit) at normalized coordinate u, clipped to the prior range.
    def from_u(self, u: float) -> float:
        u = min(max(float(u), 0.0), 1.0)
        if self.log:
            x = math.exp(math.log(self.lo) + u * (math.log(self.hi) - math.log(self.lo)))
        else:
            x = self.lo + u * (self.hi - self.lo)
        return float(min(max(x, self.lo), self.hi))


# Residuals of one target
def _as_float_array(x: Any) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=float))


# number of Gaussian data in a likelihood — the residuals that carry a sigma.
# a sigma-less interval target (CAL-1, CAL-11) is a flat constraint — it gives a residual (zero inside) but no degree of freedom.
def n_data(targets: Sequence[Target]) -> int:
    count = 0
    for target in targets:
        if target.kind == "interval" and target.sigma is None:
            continue
        if target.kind == "curve":
            count += len(target.value["y"])
        elif target.kind == "set":
            count += len(target.value)
        elif target.kind == "interval":
            count += 2
        else:
            count += 1
    return count


# standardized residuals (prediction - value) / sigma of one calibration target (see the module comments).
# `prediction` has the shape that `mrm.validation.targets_io.evaluate` expects (the replica's value).
# a non-finite prediction gives a large finite residual (`_NONFINITE_RESIDUAL` per point), so the optimizer moves away from it.
# `sigma_scale` divides every residual.
# `common_sigma` — the points of a curve target share one read-off error of this size.
# Example — in CAL-4 every shift is a difference against the same 0 V read-off.
# the covariance is then diag(sigma^2) with common_sigma^2 off
# the diagonal.
# the residuals are whitened, w = L^-1 (prediction - value) with L L^T the covariance, so w.w is the correlated chi2.
# (Treating the points as independent would count the one shared 0 V read-off error as 14 independent errors.)
def target_residuals(target: Target, prediction: Any, *, sigma_scale: float = 1.0,
                     common_sigma: float | None = None) -> np.ndarray:
    kind = target.kind
    # y - the published values, p - the prediction, s - the sigma of each point (target unit)
    if common_sigma and kind == "curve":
        observed = _as_float_array(target.value["y"])
        difference = _as_float_array(prediction) - observed
        point_sigma = _as_float_array(target.sigma) * np.ones_like(observed)
        if np.all(np.isfinite(difference)):
            covariance = np.full((observed.size, observed.size), float(common_sigma) ** 2)
            np.fill_diagonal(covariance, point_sigma ** 2)
            return np.linalg.solve(np.linalg.cholesky(covariance), difference) / sigma_scale
    if kind == "curve":
        observed = _as_float_array(target.value["y"])
        predicted = _as_float_array(prediction)
        point_sigma = _as_float_array(target.sigma) * np.ones_like(observed)
    elif kind == "scalar":
        observed = _as_float_array(target.value)
        predicted = _as_float_array(prediction)
        if target.sigma is None:
            raise ValueError(f"{target.id}: a scalar target without sigma cannot be fitted")
        point_sigma = _as_float_array(target.sigma)
    elif kind == "set":
        keys = sorted(target.value)
        observed = np.array([float(target.value[k]) for k in keys])
        predicted = np.array([float(prediction[k]) for k in keys])
        sigma = target.sigma
        point_sigma = (np.array([float(sigma[k]) for k in keys]) if isinstance(sigma, Mapping)
             else np.full(len(keys), float(sigma)))
    elif kind == "interval":
        lo, hi = (float(v) for v in target.value)
        predicted = _as_float_array(prediction)
        if target.sigma is None:
            # flat likelihood inside the interval, a steep wall outside - a predicted range (CAL-11) must lie inside
            wall = INTERVAL_WALL_FRAC * (hi - lo)
            residual = np.where(predicted < lo, (predicted - lo) / wall, np.where(predicted > hi,
                (predicted - hi) / wall, 0.0))
            return np.where(np.isfinite(residual), residual, _NONFINITE_RESIDUAL) / sigma_scale
        observed = np.array([lo, hi])
        point_sigma = np.full(2, float(target.sigma))
    else:
        raise ValueError(f"{target.id}: kind {kind!r} is not fitted here")
    residual = (predicted - observed) / point_sigma
    return np.where(np.isfinite(residual), residual, _NONFINITE_RESIDUAL) / sigma_scale


# Problem, fit and profile
# one hypothesis with parameters, targets, and a prediction function.
# `predict` maps parameter names to target predictions.
# `constraint` adds penalty or prior residuals.
# `derived` records other quantities at the fit.
# `starts` lists initial values, with missing entries taken from Param.start.
# `common_sigma` specifies a shared error for each affected curve target.
# `fixed` records assumptions.
# n_evals counts prediction cache misses.
@dataclasses.dataclass
class Problem:

    group: str
    hypothesis: str
    params: list[Param]
    targets: list[Target]
    predict: Callable[[dict[str, float]], dict[str, Any]]
    fixed: dict[str, Any] = dataclasses.field(default_factory=dict)
    starts: list[dict[str, float]] = dataclasses.field(default_factory=list)
    constraint: Callable[[dict[str, float]], np.ndarray] | None = None
    derived: Callable[[dict[str, float]], dict[str, Any]] | None = None
    common_sigma: dict[str, float] = dataclasses.field(default_factory=dict)
    n_evals: int = 0
    _cache: dict = dataclasses.field(default_factory=dict, repr=False)

    # {param name — value} from normalized coordinates.
    # with `fixed_u` ({param index — u}), `u` holds only the other parameters, in order (a profile holds one parameter fixed).
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

    # predict(theta), cached on the parameter values rounded to 12 decimal places (round(x, 12)).
    def predictions(self, theta: Mapping[str, float]) -> dict[str, Any]:
        key = tuple(round(float(theta[p.name]), 12) for p in self.params)
        if key in self._cache:
            return self._cache[key]
        self.n_evals += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = self.predict(dict(theta))
        self._cache[key] = prediction
        if len(self._cache) > _CACHE_MAX_ENTRIES:
            self._cache.clear()           # return the new prediction from pred - the clear removed it from the cache
        return prediction

    # the stacked residuals of every target (and of the constraint, unless `with_constraint` is False).
    def residuals(self, theta: Mapping[str, float], *, with_constraint: bool = True) -> np.ndarray:
        pred = self.predictions(theta)
        r = [target_residuals(t, pred[t.id], common_sigma=self.common_sigma.get(t.id)) for t in self.targets]
        if with_constraint and self.constraint is not None:
            r.append(np.atleast_1d(np.asarray(self.constraint(dict(theta)), dtype=float)))
        return np.concatenate(r)

    # sum of squares of every residual, the constraint residuals included.
    def chi2(self, theta: Mapping[str, float]) -> float:
        r = self.residuals(theta)
        return float(r @ r)


# best values, fit quality, and local minima for one hypothesis.
# chi2 includes target and constraint residuals.
# `residuals` stores target residuals only.
# dof counts Gaussian data minus free parameters, with a floor
# of one.
# birge2 = max(1, chi2 / dof), including constraint contributions.
# field names are saved record keys.
@dataclasses.dataclass
class FitResult:

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

    # the fit as plain JSON types (`jsonable`).
    def as_dict(self) -> dict:
        return jsonable(self)


# one local least-squares fit from the normalized start `u0` (the free parameters only).
# returns (u, chi2).
# the optimizer works on z = u + 1 in [1, 2] rather than on u in [0, 1] — the relative options (the finite-difference step diff_step, the step tolerance xtol) then act as absolute ones of about the same size over the whole range.
# a local fit that fails returns its start, which its chi2 then reports.
def _least_squares(prob: Problem, u0: np.ndarray, fixed_u: Mapping[int, float] | None = None,
                   max_nfev: int = 200) -> tuple[np.ndarray, float]:
    free = [i for i in range(len(prob.params)) if not fixed_u or i not in fixed_u]
    if not free:
        theta = prob.theta([], fixed_u)
        return np.array([]), prob.chi2(theta)

    def residuals_of_z(z):
        return prob.residuals(prob.theta(np.asarray(z) - 1.0, fixed_u))
    z0 = np.clip(np.asarray(u0, dtype=float), 0.0, 1.0) + 1.0
    try:
        result = optimize.least_squares(residuals_of_z, z0, bounds=(np.ones(len(free)), 2.0 * np.ones(len(free))),
                                     method="trf", diff_step=2e-3, x_scale=1.0, max_nfev=max_nfev,
                                     xtol=1e-6, ftol=1e-8, gtol=1e-8)
        z = result.x
    except Exception:  # noqa: BLE001 - a failed local fit returns the start (reported by its chi2)
        z = z0
    u = z - 1.0
    return u, prob.chi2(prob.theta(u, fixed_u))


# least-squares fit of one hypothesis.
# it starts from every start in `prob.starts` (default — each parameter's `start`), plus `extra_starts` Latin-hypercube starts.
# Distinct local minima (normalized distance > `_DISTINCT_MINIMUM_DU`) are kept in `minima`, best first.
# `max_nfev` — the budget of residual evaluations of each local fit.
def fit(prob: Problem, *, max_nfev: int = 200, extra_starts: int = 0, seed: int = 0) -> FitResult:
    started = time.time()
    n_params = len(prob.params)
    starts = [np.array([p.to_u(s.get(p.name, p.start)) for p in prob.params]) for s in prob.starts] or \
        [np.array([p.to_u(p.start) for p in prob.params])]
    if extra_starts:
        from scipy.stats import qmc
        starts += list(qmc.LatinHypercube(d=n_params, seed=seed).random(extra_starts))
    found: list[tuple[float, np.ndarray]] = []          # (chi2, u) of each distinct local minimum
    for u0 in starts:
        u, c = _least_squares(prob, u0, max_nfev=max_nfev)
        if not any(np.max(np.abs(u - v)) < _DISTINCT_MINIMUM_DU for _, v in found):
            found.append((c, u))
        else:
            j = next(k for k, (_, v) in enumerate(found) if np.max(np.abs(u - v)) < _DISTINCT_MINIMUM_DU)
            if c < found[j][0]:
                found[j] = (c, u)
    found.sort(key=lambda cu: cu[0])
    c_best, u_best = found[0]
    theta = prob.theta(u_best)
    predictions = prob.predictions(theta)
    residuals = prob.residuals(theta, with_constraint=False)
    dof = max(n_data(prob.targets) - n_params, 1)
    target_summaries = {}
    for t in prob.targets:
        r = target_residuals(t, predictions[t.id], common_sigma=prob.common_sigma.get(t.id))
        z = target_residuals(t, predictions[t.id])          # each point against its own sigma (the reported z)
        evaluation = evaluate(t, predictions[t.id])
        target_summaries[t.id] = {"prediction": jsonable(predictions[t.id]), "chi2": float(r @ r),
                                  "z_max": float(np.max(np.abs(z))), "max_abs_residual": evaluation.max_abs_residual,
                                  "tolerance_level": evaluation.level, "within_tolerance": evaluation.passed}
    minima = [{"chi2": c, "theta": prob.theta(u)} for c, u in found]
    derived = prob.derived(dict(theta)) if prob.derived is not None else {}
    return FitResult(group=prob.group, hypothesis=prob.hypothesis, theta=theta, chi2=float(c_best), dof=int(dof),
                     birge2=max(1.0, float(c_best) / dof), residuals=residuals.tolist(), targets=target_summaries,
                     minima=minima, n_evals=prob.n_evals, runtime_s=time.time() - started, fixed=dict(prob.fixed),
                     derived=derived)


# profile likelihood of one parameter, and its half-prior verdict.
# the field names are keys of the fit record.
# grid — the parameter values (its unit).
# chi2 — the profile chi2 at each grid value.
# chi2_min — the lowest chi2.
# threshold — chi2_min + 3.84 x Birge ratio.
# interval — the 95 % interval at that threshold.
# interval_raw — the interval at chi2_min + 3.84 (no Birge widening).
# open_lo, open_hi — the interval reaches that prior bound.
# n_components — connected pieces below the threshold.
# prior — [lo, hi].
# log — log coordinates.
# width_frac — `half_prior` of the interval.
# half_prior_ok —
# width_frac <= 0.5.
# unique — one closed interval and no second local minimum below the threshold.
# best — the best-fit value.
@dataclasses.dataclass
class Profile:

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


# ([lo, hi], open_lo, open_hi, n_components) of the grid points with chi2 <= threshold.
# the edges are linear interpolations of the threshold crossings.
# with no point below the threshold, the interval is the best point.
def _interval_from_profile(grid: np.ndarray, chi2: np.ndarray, threshold: float) -> tuple[list, bool, bool, int]:
    inside = chi2 <= threshold
    if not np.any(inside):
        i = int(np.argmin(chi2))
        return [float(grid[i]), float(grid[i])], False, False, 1
    idx = np.flatnonzero(inside)
    n_components = 1 + int(np.sum(np.diff(idx) > 1))
    i0, i1 = int(idx[0]), int(idx[-1])
    open_lo, open_hi = i0 == 0, i1 == len(grid) - 1

    def crossing(a, b):
        ca, cb = chi2[a] - threshold, chi2[b] - threshold
        if not (np.isfinite(ca) and np.isfinite(cb)) or ca == cb:
            return float(grid[b])
        return float(grid[a] + (grid[b] - grid[a]) * ca / (ca - cb))
    lo = float(grid[0]) if open_lo else crossing(i0 - 1, i0)
    hi = float(grid[-1]) if open_hi else crossing(i1 + 1, i1)
    return [lo, hi], open_lo, open_hi, n_components


# profile likelihood of parameter `name` over its whole prior range (see the module comments).
# `fr` is the best fit of `prob`.
# the coarse grid (`n_grid` points) is walked outwards from the best fit, so the warm starts follow the valley.
# every grid point is also solved once from the second and third local minima (each started at that minimum, without warm starts).
# then each threshold crossing is located by up to `refine` bisection steps.
# each step is a re-optimization of the other parameters (`max_nfev` evaluations each).
def profile(prob: Problem, fr: FitResult, name: str, *, n_grid: int = 13, max_nfev: int = 60,
            refine: int = 6) -> Profile:
    i = next(k for k, p in enumerate(prob.params) if p.name == name)
    parameter = prob.params[i]
    u_best = np.array([p.to_u(fr.theta[p.name]) for p in prob.params])
    others = [k for k in range(len(prob.params)) if k != i]
    points: dict[float, tuple[float, np.ndarray]] = {}        # u_i -> (chi2, u of the other parameters)

    def solve(ui: float, u_start: np.ndarray) -> tuple[float, np.ndarray]:
        u, c = _least_squares(prob, u_start, fixed_u={i: float(ui)}, max_nfev=max_nfev)
        key = round(float(ui), 12)
        if key not in points or c < points[key][0]:
            points[key] = (c, u.copy() if u.size else u)
        return points[key]
    # the best fit itself is a point of the profile, so the interval can never collapse below it when a re-optimization with the smaller inner budget stops short of it
    points[round(float(u_best[i]), 12)] = (float(fr.chi2), u_best[others].copy())
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
    chi_min = min(min(c for c, _ in points.values()), fr.chi2)
    threshold = chi_min + DCHI2_95 * fr.birge2
    threshold_raw = chi_min + DCHI2_95
    # bisect every crossing of either threshold between neighbouring grid points
    for level in (threshold, threshold_raw):
        for _ in range(refine):
            us = sorted(points)
            cs = [points[u][0] for u in us]
            midpoints = []
            for a, b in zip(range(len(us) - 1), range(1, len(us))):
                if (cs[a] <= level) != (cs[b] <= level) and us[b] - us[a] > 1e-4:
                    inside = a if cs[a] <= level else b
                    midpoints.append((0.5 * (us[a] + us[b]), points[us[inside]][1]))
            if not midpoints:
                break
            for um, ustart in midpoints:
                solve(um, ustart if ustart.size else u_best[others])
    us = np.array(sorted(points))
    chi = np.array([points[u][0] for u in us])
    grid = np.array([parameter.from_u(u) for u in us])
    interval, open_lo, open_hi, n_components = _interval_from_profile(grid, chi, threshold)
    interval_raw, _, _, _ = _interval_from_profile(grid, chi, threshold_raw)
    width_fraction = half_prior(parameter, interval)
    second_minima = [m for m in fr.minima[1:] if m["chi2"] <= threshold]
    unique = n_components == 1 and not open_lo and not open_hi and not second_minima
    return Profile(name=name, path=parameter.path, grid=grid.tolist(), chi2=chi.tolist(), chi2_min=chi_min,
                   threshold=threshold, interval=interval, interval_raw=interval_raw, open_lo=open_lo,
                   open_hi=open_hi, n_components=n_components, prior=[parameter.lo, parameter.hi], log=parameter.log,
                   width_frac=width_fraction, half_prior_ok=width_fraction <= 0.5, unique=unique, best=fr.theta[name])


# width of `interval` as a fraction of the prior range of `par` (log coordinates for a log parameter).
# when the fit prior is wider than the leaf's declared range (`par.declared_range`), the fraction of the narrower, declared width is returned if it is larger.
# so the widening never makes the rule easier to meet.
def half_prior(par: Param, interval: Sequence[float]) -> float:
    lo, hi = float(interval[0]), float(interval[1])
    ranges = [(par.lo, par.hi)] + ([tuple(par.declared_range)] if par.declared_range is not None else [])
    fractions = []
    for a, b in ranges:
        if par.log:
            fractions.append((math.log(max(hi, par.lo)) - math.log(max(lo, par.lo))) / (math.log(b) - math.log(a)))
        else:
            fractions.append((hi - lo) / (b - a))
    return max(fractions)


# Hypotheses
# comparison of the hypotheses of one group — the best chi2 of each hypothesis, the likelihood ratios against the best one (raw, and with the best hypothesis' Birge ratio), and the AIC differences.
# the field names are keys of the fit record.
@dataclasses.dataclass
class HypothesisResult:

    group: str
    best: str
    rows: list[dict]
    note: str


# one row per fit, best chi2 first — delta chi2 to the best, LR = exp(-delta / 2), the Birge-scaled
# LR = exp(-delta / (2 s^2)) with s^2 the best fit's Birge ratio, and AIC = 2 k + chi2 (k free parameters).
def compare_hypotheses(group: str, fits: Sequence[FitResult], note: str = "") -> HypothesisResult:
    best = min(fits, key=lambda f: f.chi2)
    birge2_best = best.birge2
    rows = []
    for f in sorted(fits, key=lambda f: f.chi2):
        delta_chi2 = f.chi2 - best.chi2
        n_params = len(f.theta)
        aic = 2 * n_params + f.chi2
        rows.append({"hypothesis": f.hypothesis, "chi2": f.chi2, "dof": f.dof, "n_params": n_params,
                     "delta_chi2": delta_chi2, "likelihood_ratio": math.exp(-0.5 * delta_chi2),
                     "likelihood_ratio_birge": math.exp(-0.5 * delta_chi2 / birge2_best), "aic": aic,
                     "delta_aic": aic - (2 * len(best.theta) + best.chi2)})
    return HypothesisResult(group=group, best=best.hypothesis, rows=rows, note=note)


# the same likelihood with some parameters held at given values (a conditional fit).
# Example — one alternative's own leaf refitted with every other leaf at the nominal fit, as the calibrated preset composes its worlds.
def restrict(prob: Problem, fixed: Mapping[str, float], *, hypothesis: str | None = None) -> Problem:
    fixed = {k: float(v) for k, v in fixed.items()}
    params = [p for p in prob.params if p.name not in fixed]

    def predict(th):
        return prob.predict({**fixed, **th})
    constraint = None if prob.constraint is None else (lambda th: prob.constraint({**fixed, **th}))
    derived = None if prob.derived is None else (lambda th: prob.derived({**fixed, **th}))
    return Problem(group=prob.group, hypothesis=hypothesis or f"{prob.hypothesis} | held {sorted(fixed)}",
                   params=params, targets=prob.targets, predict=predict, fixed={**prob.fixed, **fixed},
                   starts=[{k: v for k, v in s.items() if k not in fixed} for s in prob.starts],
                   constraint=constraint, derived=derived, common_sigma=dict(prob.common_sigma))
