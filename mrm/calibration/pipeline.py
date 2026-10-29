# fit calibration targets, compare physical hypotheses, and decide preset updates.
# `run` fits groups in GROUP_ORDER.
# each group inherits values fitted earlier.
# the heater comes before coupling because the notch data use heater voltage, which must first be converted to power and temperature.
# device fits precede system fits.
# `write_results` saves the record.
# preset.py applies its decisions.
# compare chi-squared, likelihood ratios, Birge-scaled ratios, and AIC.
# retain the preset's nominal physical hypothesis unless its Birge-scaled likelihood ratio is below REJECT_LR.
# measurement-model choices take the best fit, with the first listed choice retained when chi-squared differs by less than TIE_DCHI2.
# record rejected hypotheses without deleting their preset alternatives.
# combine the 95 percent profile intervals of all non-rejected hypotheses that free the leaf.
# the resulting hull must be no wider than half the declared prior, closed at both ends, and connected.
# use log widths for log priors and the declared width when fitting requires wider bounds.
# invented fit bounds do not count as a declared prior.
# the fit must also meet every target tolerance.
# a successful fit can update a value without earning the calibrated label.
# a failed target match is evidence of a model limit, not a calibration.
# fit their free leaves with shared leaves held at the nominal fit.
# use those conditional values in alternatives.
# assign provenance from unconditional profiles, where the other parameters are re-fitted.
# record the narrower conditional intervals separately.
# heater thermo-optic worlds follow the same rule within their own hypotheses.
# see docs/handoff/calibration.md.
from __future__ import annotations

import dataclasses
import datetime
import json
import math
import subprocess
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from mrm.calibration import core
from mrm.calibration import groups as fit_groups
from mrm.calibration.core import FitResult, Problem, Profile, jsonable

__all__ = ["GROUP_ORDER", "REJECT_LR", "LeafDecision", "GroupResult", "CalibrationRun", "run", "RESULTS_PATH",
           "run_group", "write_results", "load_results", "coupling_context", "refresh_coupling_worlds",
           "context_from_decisions"]

# the fit record of the last run (JSON, schema mrm.calibration.fit/1)
RESULTS_PATH = Path(__file__).resolve().parent / "latest_fit.json"
# the order of the fit groups (module comments, "Group order")
GROUP_ORDER = ("heater", "coupling_er", "offset", "junction", "eo", "monitor", "modulation_depth", "dac", "laser")
# rule 1 - a hypothesis is rejected when its Birge-scaled likelihood ratio against the best hypothesis is below this
REJECT_LR = 0.05
# two hypotheses whose chi2 differ by less than this are a tie (a reparametrization such as the droop transfer fits the data identically) - the first listed one is taken, so the record does not depend on rounding noise
TIE_DCHI2 = 0.01


# Records
# value, provenance, and reason for updating one preset leaf.
# field names are saved record keys.
# `write_value=False` keeps the current value while allowing notes and alternatives to be written.
@dataclasses.dataclass
class LeafDecision:

    path: str
    value: Any
    provenance: str
    previous_value: Any
    previous_provenance: str
    scope: str                          # device (both presets) | system (the G26 bank only)
    targets: list[str]
    residual: dict[str, Any]            # target id -> max |prediction - value| (target unit) at the nominal fit
    identifiable: dict[str, Any]        # profile interval, prior, width fraction, uniqueness, rule outcome
    hypothesis: str
    note: str
    alternatives: dict[str, dict] = dataclasses.field(default_factory=dict)
    range: list[float] | None = None    # restated plausible range when the fit needs a wider one (stated in note)
    write_value: bool = True            # False - only alternatives / note are written, the value stays


# the result of one fit group — its fits (by hypothesis), the comparison, the profiles ({hypothesis — {param name — Profile}}), the leaf decisions, group-specific extras, and the wall time (s).
@dataclasses.dataclass
class GroupResult:

    group: str
    targets: list[str]
    nominal: str
    fits: dict[str, FitResult]
    comparison: core.HypothesisResult
    profiles: dict[str, dict[str, Profile]]
    decisions: list[LeafDecision]
    extra: dict[str, Any]
    runtime_s: float


# one calibration run — the group results, the context (leaves for later groups) and the notes.
@dataclasses.dataclass
class CalibrationRun:

    created_utc: str
    git_commit: str | None
    groups: dict[str, GroupResult]
    context: dict[str, dict]
    before_after: dict[str, Any] = dataclasses.field(default_factory=dict)
    notes: list[str] = dataclasses.field(default_factory=list)
    previous_groups: dict[str, dict] = dataclasses.field(default_factory=dict)   # records of groups not rerun

    def decisions(self) -> list[LeafDecision]:
        return [d for g in self.groups.values() for d in g.decisions]


# the run
# fit all groups, or a subset in GROUP_ORDER, and optionally save the record.
# for a subset, load earlier groups from `prior` or the previous output file.
# carry over groups that are not run.
# only preceding groups supply fit context.
# a later group that is not run retains its old fit, so rerun it if its inputs changed.
# `quick` uses coarser profiles and fewer starting points.
def run(groups: Sequence[str] | None = None, *, quick: bool = False, out: str | Path | None = RESULTS_PATH,
        log: Callable[[str], None] = print, prior: CalibrationRun | dict | None = None) -> CalibrationRun:
    targets = core.calibration_targets()
    ctx = fit_groups.Context()
    names = list(GROUP_ORDER) if not groups else [g for g in GROUP_ORDER if g in groups]
    subset = names != list(GROUP_ORDER)
    previous = prior
    if subset and previous is None:
        source_path = Path(out) if out is not None and Path(out).exists() else RESULTS_PATH
        previous = load_results(source_path) if source_path.exists() else None
    previous_context = {}
    if previous is not None:
        previous_context = previous["context"] if isinstance(previous, dict) else previous.context
    previous_groups: dict = {}
    if previous is not None:
        previous_groups = previous.get("groups", {}) if isinstance(previous, dict) else \
            {g: {"decisions": [jsonable(d) for d in gr.decisions]} for g, gr in previous.groups.items()}
    if subset and previous is not None:
        ctx.ranges.update({k: list(v) for k, v in previous_context.get("ranges", {}).items()})
        if not previous_groups:
            # a bare context (no group records) - every leaf of it is an earlier group's
            ctx.device.update(previous_context.get("device", {}))
            ctx.system.update(previous_context.get("system", {}))

    # before rerun group g — load the previous record's leaves of every group that precedes g and is not rerun.
    # this holds for any subset, also a non-contiguous one such as `heater laser` (laser then sees the coupling ... dac leaves).
    # a later group's leaf never enters an earlier group's fit, as in a full run.
    def preceding_context(g: str) -> None:
        if not subset or previous is None:
            return
        for h in GROUP_ORDER[:GROUP_ORDER.index(g)]:
            if h in names or h not in previous_groups:
                continue
            paths = {d.get("path") for d in previous_groups[h].get("decisions", [])}
            previous_count = len(ctx.device) + len(ctx.system)
            ctx.device.update({k: v for k, v in previous_context.get("device", {}).items() if k in paths})
            ctx.system.update({k: v for k, v in previous_context.get("system", {}).items() if k in paths})
            if len(ctx.device) + len(ctx.system) > previous_count:
                log(f"context from the previous run of group {h}: "
                    f"{len(ctx.device) + len(ctx.system) - previous_count} leaves")
    calibration_run = CalibrationRun(
        created_utc=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        git_commit=_git_commit(), groups={}, context={})
    if subset and isinstance(previous, dict):
        calibration_run.previous_groups = {g: v for g, v in previous.get("groups", {}).items() if g not in names}
        if calibration_run.previous_groups:
            calibration_run.notes.append(f"groups {', '.join(calibration_run.previous_groups)} from the run of "
                                         f"{previous.get('created_utc')}")

    def merged_context() -> dict:
        out_ctx = {k: dict(previous_context.get(k, {})) for k in ("device", "system", "ranges")} if subset else \
            {"device": {}, "system": {}, "ranges": {}}
        out_ctx["device"].update(ctx.device)
        out_ctx["system"].update(ctx.system)
        out_ctx["ranges"].update(ctx.ranges)
        return out_ctx
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for g in names:
            log(f"group {g}")
            preceding_context(g)
            calibration_run.groups[g] = run_group(g, ctx, targets, quick=quick, log=log)
            calibration_run.context = merged_context()
            if out is not None:
                write_results(calibration_run, out)
    return calibration_run


# fit one group, compare its hypotheses, profile, decide its leaves and update `ctx`.
# `targets` — the calibration targets by id.
# `quick` — a coarser profile grid (7 instead of 9 points) and a smaller inner budget (25 instead of 30 evaluations per profile point).
# coupling_er also skips its 4 extra Latin-hypercube starts, and dac uses 4 dies instead of 8.
def run_group(name: str, ctx: fit_groups.Context, targets: Mapping[str, Any], *, quick: bool = False,
              log=print) -> GroupResult:
    started = time.time()
    n_grid = 7 if quick else 9
    inner = 25 if quick else 30
    group_fn = {"coupling_er": _group_coupling_er, "heater": _group_heater, "offset": _group_offset,
                "junction": _group_junction, "eo": _group_eo, "monitor": _group_monitor,
                "modulation_depth": _group_modulation_depth, "dac": _group_dac, "laser": _group_laser}[name]
    result = group_fn(ctx, targets, n_grid=n_grid, inner=inner, quick=quick, log=log)
    result.runtime_s = time.time() - started
    for d in result.decisions:
        if d.write_value and d.path and not d.path.startswith("variation.") and d.path != "uncertainty":
            (ctx.device if d.scope == "device" else ctx.system)[d.path] = d.value
    return result


# write the fit record of `res` (JSON, schema `mrm.calibration.fit/1`), groups in `GROUP_ORDER`.
def write_results(res: CalibrationRun, path: str | Path = RESULTS_PATH) -> Path:
    path = Path(path)
    record = {"schema": "mrm.calibration.fit/1", "created_utc": res.created_utc, "git_commit": res.git_commit,
         "reject_lr": REJECT_LR, "context": res.context, "notes": res.notes, "before_after": res.before_after,
         "groups": {}}
    for g in GROUP_ORDER:
        if g in res.previous_groups and g not in res.groups:
            record["groups"][g] = res.previous_groups[g]
    for g, group_result in res.groups.items():
        record["groups"][g] = {
            "targets": group_result.targets, "nominal": group_result.nominal, "runtime_s": group_result.runtime_s,
            "comparison": jsonable(group_result.comparison),
            "fits": {h: {k: v for k, v in jsonable(f).items() if k != "profiles"} for h,
                f in group_result.fits.items()},
            "profiles": {h: {n: jsonable(p) for n, p in ps.items()} for h, ps in group_result.profiles.items()},
            "decisions": [jsonable(x) for x in group_result.decisions],
            "extra": jsonable(group_result.extra)}
    record["groups"] = {g: record["groups"][g] for g in GROUP_ORDER if g in record["groups"]}
    path.write_text(json.dumps(record, indent=1, allow_nan=False, default=str))
    return path


# the fit record as a dict.
def load_results(path: str | Path = RESULTS_PATH) -> dict:
    return json.loads(Path(path).read_text())


# the device and system leaves that a full serial run leaves in its context, rebuilt from a record's leaf decisions.
# `run_group` writes every decision with `write_value`.
# coupling_er also hands the fitted dispersion to the later groups.
# a record's `context` must equal this, because `bench_config` and every `--groups` rerun read it.
# (If a record's context disagrees with its decisions, the before/after table and every rerun use stale leaves.)
def context_from_decisions(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {"device": {}, "system": {}}
    groups = record.get("groups") or {}
    for g in GROUP_ORDER:
        for d in (groups.get(g) or {}).get("decisions") or []:
            p = d.get("path")
            if d.get("write_value", True) and p and not str(p).startswith("variation.") and p != "uncertainty":
                out["device" if d.get("scope") == "device" else "system"][p] = d["value"]
            if g == "coupling_er" and p == fit_groups.LEAF_COUPLER_DISPERSION:
                out["device"][p] = d["value"]
    return out


# the short HEAD commit, with `-dirty` when the calibration code or its inputs (mrm/, tests/calibration/) differ from it, so a record made with uncommitted code does not name only the commit.
def _git_commit() -> str | None:
    root = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=root, timeout=10).stdout.strip() or None
        if head is None:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "mrm", "tests/calibration",
                                ":!mrm/calibration/latest_fit.json", ":!mrm/params/presets/g26_calibrated.yaml"],
                               capture_output=True, text=True, cwd=root, timeout=10).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return None


# rules 1 and 2 - shared group helpers
# rule 1 — the hypotheses whose Birge-scaled likelihood ratio is below `REJECT_LR`.
def _rejected_hypotheses(comparison: core.HypothesisResult) -> set[str]:
    return {r["hypothesis"] for r in comparison.rows if r["likelihood_ratio_birge"] < REJECT_LR}


# the best of `hyps` by chi2, the first listed one among ties (`TIE_DCHI2`).
def _first_best(hyps: Sequence[str], fits: Mapping[str, FitResult]) -> str | None:
    if not hyps:
        return None
    c_min = min(fits[h].chi2 for h in hyps)
    return next(h for h in hyps if fits[h].chi2 <= c_min + TIE_DCHI2)


# [lowest lo, highest hi] of the intervals.
def _interval_hull(intervals: Sequence[Sequence[float]]) -> list[float]:
    return [min(float(i[0]) for i in intervals), max(float(i[1]) for i in intervals)]


# (value, provenance) of a leaf of `cfg`.
# (None, "absent") for a leaf absent from the preset.
def _leaf_value_and_provenance(cfg, path: str) -> tuple[Any, str]:
    try:
        leaf = cfg.leaf(path)
        return jsonable(leaf.value), str(leaf.provenance)
    except Exception:  # noqa: BLE001 - a leaf absent from the preset
        return None, "absent"


# apply the interval and declared-prior requirements to one leaf.
# combine profiles from non-rejected hypotheses.
# detect open endpoints and gaps, then compare the hull width with half the prior.
# the target-tolerance gate is applied separately.
# a unique optimum is recorded separately from calibration.
def _identifiability(par: core.Param, profiles: Mapping[str, Profile], fits: Mapping[str, FitResult],
                     rejected: set[str]) -> dict[str, Any]:
    used = {h: p for h, p in profiles.items() if h not in rejected}
    if not used:
        return {"rule": "no profile", "calibrated": False}
    hull = _interval_hull([p.interval for p in used.values()])
    has_open_endpoint = any(p.open_lo or p.open_hi for p in used.values())
    # components - gaps inside one hypothesis' profile, plus gaps between the hypotheses' intervals
    n_components = 1 + sum(p.n_components - 1 for p in used.values())
    intervals = sorted((list(p.interval) for p in used.values()), key=lambda iv: iv[0])
    reach = intervals[0][1]
    for iv in intervals[1:]:
        if iv[0] > reach:
            n_components += 1
        reach = max(reach, iv[1])
    width_fraction = core.half_prior(par, hull)
    declared = bool(getattr(par, "declared_prior", True))
    passes_interval_rule = width_fraction <= 0.5 and not has_open_endpoint and n_components == 1 and declared
    best_by_h = {h: fits[h].theta.get(par.name) for h in used}
    return {"profile_95": hull, "prior": [par.lo, par.hi], "prior_source": par.prior_source,
            "log_prior": par.log, "width_frac": width_fraction, "open": has_open_endpoint, "components": n_components,
            "per_hypothesis": {h: {"interval": p.interval, "interval_raw": p.interval_raw, "best": best_by_h[h],
                                   "open_lo": p.open_lo, "open_hi": p.open_hi, "unique": p.unique,
                                   "chi2_min": p.chi2_min, "threshold": p.threshold}
                               for h, p in used.items()},
            "unique": bool(passes_interval_rule and all(p.unique for p in used.values()) and len(used) == 1),
            "calibrated": bool(passes_interval_rule),
            "rule": ("calibrated: 95 % profile hull within half the prior, closed, one interval" if passes_interval_rule else
                     "kept: " + "; ".join(x for x in (
                         f"width {width_fraction:.2f} of the prior > 0.5" if width_fraction > 0.5 else "",
                         "interval reaches the prior bound (open)" if has_open_endpoint else "",
                         f"{n_components} disjoint intervals across hypotheses" if n_components > 1 else "",
                         "no declared prior range (fit bounds only), so the half-prior rule cannot apply"
                         if not declared else "") if x))}


# the per-target summaries of a fit (a FitResult or its record dict).
def _target_summaries(fr) -> Mapping[str, Any]:
    return fr.targets if isinstance(fr, FitResult) else fr["targets"]


# prevent a calibrated label when any group target misses its tolerance.
# return a copied identifiability record with the tolerance result.
# a narrow parameter interval cannot compensate for a model that misses its data.
def _tolerance_gate(ident: dict, fr) -> dict:
    missed = [t for t, v in _target_summaries(fr).items() if not v.get("within_tolerance")]
    ident = dict(ident, within_tolerance=not missed)
    if missed and ident.get("calibrated"):
        ident.update(calibrated=False, unique=False,
                     rule="kept: the fit misses the tolerance of " + ", ".join(missed)
                          + " (a model-form failure is not a calibration)")
    return ident


# per target of a fit — max |prediction - value| (target unit), z_max and the tolerance verdict.
def _residual_summary(fr: FitResult) -> dict[str, Any]:
    return {t: {"max_abs_residual": v["max_abs_residual"], "z_max": v["z_max"],
                "tolerance_level": v["tolerance_level"]} for t, v in fr.targets.items()}


# `x` to `n` significant digits.
def _fmt_sig(x: float, n: int = 4) -> str:
    return f"{x:.{n}g}"


# the LeafDecision of parameter `par` from the fit `fr` and its identifiability `ident` (rule 2, with the tolerance gate).
# `cfg` — the config whose leaf is the previous value.
def _leaf_decision(cfg, par: core.Param, fr: FitResult, ident: dict, *, scope: str, hypothesis: str,
                   targets: list[str], extra_note: str = "", range_: list[float] | None = None,
                   write_value: bool = True) -> LeafDecision:
    prev_value, prev_provenance = _leaf_value_and_provenance(cfg, par.path)
    # pseudo-paths - a scale on a preset variation entry, previously 1 (written by preset._variation_update)
    if par.path in ("variation.geometry.ring_sigma_scale", "variation.laser.line_offset_ghz.sigma_scale"):
        prev_value, prev_provenance = 1.0, "assumed" if "geometry" in par.path else "inferred"
    value = float(fr.theta[par.name])
    ident = _tolerance_gate(ident, fr)
    provenance = "calibrated" if ident.get("calibrated") else prev_provenance
    hull = ident.get("profile_95")
    note = (f"calibration ({', '.join(targets)}, hypothesis {hypothesis}): best fit {_fmt_sig(value)}"
            + (f", 95 % profile {_fmt_sig(hull[0])}-{_fmt_sig(hull[1])} over the non-rejected hypotheses"
               f" ({ident['width_frac']:.2f} of the "
               + (f"declared range {_fmt_sig(par.declared_range[0])}-{_fmt_sig(par.declared_range[1])}; fit prior "
                  if getattr(par, "declared_range", None) else "prior ")
               + f"{_fmt_sig(par.lo)}-{_fmt_sig(par.hi)}, {par.prior_source})"
               if hull else "")
            + f"; {ident.get('rule', '')}"
            + (f"; was {prev_value!r} ({prev_provenance})" if prev_value is not None else "")
            + (f". {extra_note}" if extra_note else ""))
    return LeafDecision(path=par.path, value=value, provenance=provenance, previous_value=prev_value,
                        previous_provenance=prev_provenance, scope=scope, targets=list(targets),
                        residual=_residual_summary(fr), identifiable=ident, hypothesis=hypothesis, note=note,
                        range=range_, write_value=write_value)


# {hypothesis — FitResult} of every problem of a group.
def _fit_hypotheses(probs: Sequence[Problem], *, extra_starts: int = 0, max_nfev: int = 150, log=print) -> dict:
    fits = {}
    for problem in probs:
        started = time.time()
        fits[problem.hypothesis] = core.fit(problem, max_nfev=max_nfev, extra_starts=extra_starts)
        log(f"  fit {problem.group} {problem.hypothesis}: chi2 {fits[problem.hypothesis].chi2:.4g} "
            f"theta {{{', '.join(f'{k}: {_fmt_sig(v)}' for k, v in fits[problem.hypothesis].theta.items())}}} "
            f"({time.time() - started:.0f} s)")
    return fits


# {param name — Profile} of every leaf parameter of `prob` (or of `names`).
# also stored in `fr.profiles`.
def _profile_leaves(prob: Problem, fr: FitResult, *, n_grid: int, max_nfev: int, log=print,
                    names: Sequence[str] | None = None) -> dict[str, Profile]:
    profiles = {}
    for par in prob.params:
        if par.role != "leaf" or (names is not None and par.name not in names):
            continue
        started = time.time()
        profiles[par.name] = core.profile(prob, fr, par.name, n_grid=n_grid, max_nfev=max_nfev)
        profile_result = profiles[par.name]
        log(f"    profile {prob.hypothesis} {par.name}: [{_fmt_sig(profile_result.interval[0])}, {_fmt_sig(profile_result.interval[1])}] "
            f"frac {profile_result.width_frac:.2f} open {profile_result.open_lo}/{profile_result.open_hi} ({time.time() - started:.0f} s)")
    fr.profiles.update(profiles)
    return profiles


# Group heater (CAL-3a, CAL-3b, CAL-4)
# thermo-optic worlds of the heater group other than the nominal H_solver - (hypothesis prefix, uncertainty entry, choice, name of the alternative written on the heater leaves)
HEATER_WORLDS = (("H86", "thermo_optic", "H86", "cal_h86"),)


def _group_heater(ctx, targets, *, n_grid, inner, quick, log):
    problems = fit_groups.heater_problems(ctx, targets)
    problem_by_hyp = {p.hypothesis: p for p in problems}
    fits = _fit_hypotheses(problems, log=log)
    comparison = core.compare_hypotheses("heater", list(fits.values()),
                                         note="thermo-optic world x droop transfer from the swept device to the "
                                              "median device (CAL-4 vs CAL-3)")
    rejected = _rejected_hypotheses(comparison)
    world_of = {h: next((w for w in HEATER_WORLDS if h.startswith(w[0] + "/")), None) for h in problem_by_hyp}
    nominal_set = [h for h in problem_by_hyp if world_of[h] is None]
    # the nominal world is H_solver (rule 1 - the g26 nominal of uncertainty.thermo_optic unless rejected) - within it the droop-transfer hypothesis (a measurement model) takes the best
    nominal = _first_best([h for h in nominal_set if h not in rejected], fits) or comparison.best
    profiles = {h: _profile_leaves(p, fits[h], n_grid=n_grid, max_nfev=inner, log=log)
                for h, p in problem_by_hyp.items() if h not in rejected}
    cfg = ctx.config(fit_groups.BENCH)
    nominal_fit = fits[nominal]
    target_ids = ["CAL-3a", "CAL-3b", "CAL-4"]
    decisions = []
    derived = nominal_fit.derived
    for par in problem_by_hyp[nominal].params:
        if par.role != "leaf":
            continue
        # the leaf value is the H_solver world's - its tag follows the profiles of that world's hypotheses - each other thermo-optic world gets its own alternative, tagged from its own profiles (rule 3)
        ident = _identifiability(par, {h: pr[par.name] for h, pr in profiles.items() if h in nominal_set}, fits,
                                 rejected)
        extra, rng = "", None
        if par.path == fit_groups.LEAF_R_COLD:
            dlo, dhi = par.declared_range if par.declared_range is not None else (par.lo, par.hi)
            extra = (f"hot resistance at P_FSR {_fmt_sig(derived['r_hot_at_p_fsr_ohm'])} ohm (CAL-3b: "
                     f"1.195^2/25.45 = 56.1); the prior widens the leaf range [{_fmt_sig(dlo)}, {_fmt_sig(dhi)}] "
                     f"(R0 = R_hot/(1 + y P_FSR) over the droops of the thermo-optic worlds, the leaf's own note) to "
                     f"[{_fmt_sig(par.lo)}, {_fmt_sig(par.hi)}], as R0 trades against y; the half-prior fraction is "
                     f"taken on the declared {_fmt_sig(dhi - dlo)} ohm width")
            rng = [float(par.lo), float(par.hi)]
        if par.path == fit_groups.LEAF_HEATER_SELF_RATIO:
            extra = (f"y = TCR x R_th,H = {_fmt_sig(derived['y_per_mw'])} /mW at TCR "
                     f"{_fmt_sig(derived['tcr_per_k'])} /K (R_th,H {_fmt_sig(derived['r_th_heater_k_per_mw'])} K/mW, "
                     f"Z_heater->ring {_fmt_sig(derived['z_heater_ring_k_per_mw'])} K/mW); only y is identified "
                     f"(06 E1): the ratio "
                     f"is calibrated for the family's TCR. Swept device "
                     f"{_fmt_sig(derived['efficiency_swept_device_nm_per_mw'])} nm/mW (06 E4: 0.348), median "
                     f"{_fmt_sig(derived['efficiency_median_nm_per_mw'])} nm/mW")
        if par.path == fit_groups.LEAF_P_FSR:
            extra = ("the R2 anchor with the FSR at the line, set so the bench protocol (FSR between the cold order "
                     "and the next red order) gives the printed 25.45 mW median")
        extra += (f". Conditional on: the thermo-optic world H_solver (dlambda/dT rising with T; the H86 world has "
                  f"its own alternative 'cal_h86'), the droop transfer from the swept device (hull over "
                  f"same_rth_heater and same_ratio), the swept device's own R0 (a nuisance with the variation "
                  f"entry's log prior; fitted R0_dev {_fmt_sig(derived.get('r_cold_swept_device_ohm', float('nan')))} "
                  f"ohm), and the CAL-3a/3b medians read as one device")
        d = _leaf_decision(cfg, par, nominal_fit, ident, scope="device", hypothesis=nominal, targets=target_ids,
                           extra_note=extra, range_=rng)
        for prefix, entry, choice, alt in HEATER_WORLDS:
            world_hyps = [h for h in problem_by_hyp if world_of[h] and world_of[h][0] == prefix]
            world_ok = [h for h in world_hyps if h not in rejected]
            if not world_ok:
                d.note += f". World {entry}={choice}: rejected (no alternative written)"
                continue
            world_best = _first_best(world_ok, fits)
            world_ident = _tolerance_gate(_identifiability(par, {h: profiles[h][par.name] for h in world_ok}, fits,
                                                           rejected), fits[world_best])
            row = next(r for r in comparison.rows if r["hypothesis"] == world_best)
            hull = world_ident.get("profile_95")
            d.alternatives[alt] = {
                "value": float(fits[world_best].theta[par.name]),
                "calibrated": bool(world_ident.get("calibrated")),
                "provenance": "calibrated" if world_ident.get("calibrated") else d.previous_provenance,
                "select": {"entry": entry, "choice": choice},
                "identifiable": world_ident,
                "note": (f"world uncertainty.{entry}={choice} ({world_best}): chi2 {fits[world_best].chi2:.3g} "
                         f"against {nominal_fit.chi2:.3g} in the nominal world (LR {row['likelihood_ratio']:.2g}, "
                         f"Birge {row['likelihood_ratio_birge']:.2g}); 95 % profile "
                         + (f"{_fmt_sig(hull[0])}-{_fmt_sig(hull[1])} over {', '.join(world_ok)}" if hull else "none")
                         + f"; {world_ident.get('rule', '')}; y "
                           f"{_fmt_sig(fits[world_best].derived.get('y_per_mw', float('nan')))} /mW")}
        decisions.append(d)
    return GroupResult("heater", target_ids, nominal, fits, comparison, profiles, decisions,
                       {"derived": derived, "worlds": {w[0]: {"entry": w[1], "choice": w[2], "alternative": w[3]}
                                                       for w in HEATER_WORLDS}}, 0.0)


# Group coupling_er (CAL-1, CAL-2)
# the worlds the calibrated preset composes from the CAL-1/CAL-2 hypotheses (rule 3) - (world, base hypothesis, leaves held at the nominal fit, {free leaf name: alternative name}).
# the loss mechanisms fit only on the over-coupled branch (the mirror of the dispersion fit), so their worlds carry their own alpha_other.
COUPLING_WORLDS = (
    ("coupling_regime=over", "over/dispersion/floor", ("q_loaded", "dln_kappa2_through"),
     {"alpha_other_abs": "over"}),
    ("er_mechanism=ring_t_loss", "over/ring_t_loss/floor", ("q_loaded",),
     {"alpha_other_abs": "cal_ring_t_loss_branch", "dln_alpha_dt_ring": "ring_t_loss"}),
    ("er_mechanism=heater_t_loss", "over/heater_t_loss/floor", ("q_loaded",),
     {"alpha_other_abs": "cal_heater_t_loss_branch", "dln_alpha_dt_heater": "heater_t_loss"}),
)
# the same loss worlds held on the nominal (under-coupled) branch - reported only, to show why the branch is tied
COUPLING_WORLDS_INFO = (
    ("er_mechanism=ring_t_loss on the nominal branch", "under/ring_t_loss/floor", ("q_loaded", "alpha_other_abs")),
    ("er_mechanism=heater_t_loss on the nominal branch", "under/heater_t_loss/floor",
     ("q_loaded", "alpha_other_abs")),
)
# leaves whose value belongs to one branch - their unconditional interval is the base hypothesis' alone
BRANCH_LEAVES = ("alpha_other_abs",)


def _group_coupling_er(ctx, targets, *, n_grid, inner, quick, log):
    problems = fit_groups.coupling_er_problems(ctx, targets)
    problem_by_hyp = {p.hypothesis: p for p in problems}
    fits = _fit_hypotheses(problems, extra_starts=0 if quick else 4, log=log)
    comparison = core.compare_hypotheses("coupling_er", list(fits.values()),
                                         note="coupling branch x extinction mechanism x instrument floor")
    rejected = _rejected_hypotheses(comparison)
    nominal = "under/dispersion/floor"
    if nominal in rejected:
        nominal = comparison.best
    log(f"  coupling_er: best {comparison.best}, nominal {nominal}, rejected {sorted(rejected)}")
    profiles: dict[str, dict[str, Profile]] = {}
    for h, p in problem_by_hyp.items():
        if h in rejected:
            continue
        # every leaf whose tag a hypothesis' profile can decide is profiled at the full resolution - the mechanism coefficients and the branch alpha_other of the composed worlds are tagged from those profiles alone.
        # only Q_L of a non-nominal hypothesis stays coarse - it can only widen the Q_L hull, which the nominal's profile
        # already opens at the CAL-1 edge (= the top of the leaf range), so its tag cannot depend on it.
        full_names = [q.name for q in p.params if q.role == "leaf" and (h == nominal or q.name != "q_loaded")]
        profiles[h] = _profile_leaves(p, fits[h], n_grid=n_grid, max_nfev=inner, log=log, names=full_names)
        if h != nominal:
            profiles[h].update(_profile_leaves(p, fits[h], n_grid=max(5, n_grid - 4), max_nfev=max(12, inner // 2),
                                               log=log, names=["q_loaded"]))
    nominal_fit = fits[nominal]
    # CAL-1 is a flat interval, and Q_L trades against alpha_other along the CAL-2 fit valley.
    # use the CAL-1 interval centre if its fit is within delta_chi2<0.5 of the free optimum.
    # retain profile intervals from the free fit.
    q_mid = 0.5 * sum(float(x) for x in targets["CAL-1"].value)
    centred = core.fit(core.restrict(problem_by_hyp[nominal], {"q_loaded": q_mid}, hypothesis=nominal), max_nfev=150)
    log(f"  coupling_er nominal at Q_L {q_mid:g}: chi2 {centred.chi2:.4g} (free {nominal_fit.chi2:.4g})")
    if centred.chi2 - nominal_fit.chi2 < 0.5:
        centred.theta = {"q_loaded": q_mid, **centred.theta}
        centred.profiles = nominal_fit.profiles
        fits[nominal + " (Q_L at the CAL-1 centre)"] = centred
        nominal_fit = centred
    decisions, composed = _coupling_decisions(ctx, problem_by_hyp, fits, profiles, rejected, nominal, nominal_fit,
                                              n_grid=n_grid, inner=inner, log=log)
    # only now does the nominal world take the fitted dispersion (the loss worlds above hold the solver value)
    ctx.device[fit_groups.LEAF_COUPLER_DISPERSION] = next(
        d for d in decisions if d.path == fit_groups.LEAF_COUPLER_DISPERSION).value
    extra = {"composed_worlds": composed,
             "topology_note": "the add-drop ring of the plant (kappa_d/kappa 0.007); the all-pass reading of the G25 "
                              "paper differs by the drop coupler only and is absorbed by alpha_other"}
    return GroupResult("coupling_er", ["CAL-1", "CAL-2"], nominal, fits, comparison, profiles, decisions, extra, 0.0)


# the leaf decisions of the coupling_er group (rules 2 and 3) from its fits and profiles.
# returns (decisions, the record of the composed worlds).
def _coupling_decisions(ctx, problem_by_hyp, fits, profiles, rejected, nominal, nominal_fit, *, n_grid, inner,
                        log=print):
    cfg = ctx.config(fit_groups.BENCH)
    target_ids = ["CAL-1", "CAL-2"]
    decisions = []
    for par in problem_by_hyp[nominal].params:
        if par.role != "leaf" or par.path == fit_groups.LEAF_COUPLER_DISPERSION:
            continue
        sub = {h: pr[par.name] for h, pr in profiles.items() if par.name in pr}
        ident = _identifiability(par, sub, fits, rejected)
        decisions.append(_leaf_decision(cfg, par, nominal_fit, ident, scope="device", hypothesis=nominal,
                                        targets=target_ids, extra_note=_coupling_note(nominal_fit)))
    # the nominal mechanism's own leaf (dispersion) - the hull of every non-rejected hypothesis that frees it (both branches) - written by the preset as the leaf value, the solver value kept as the alternative 'solver'
    disp_param = next(p for p in problem_by_hyp[nominal].params if p.path == fit_groups.LEAF_COUPLER_DISPERSION)
    disp_profiles = {h: pr[disp_param.name] for h, pr in profiles.items() if disp_param.name in pr}
    disp_decision = _leaf_decision(cfg, disp_param, nominal_fit,
                                   _identifiability(disp_param, disp_profiles, fits, rejected), scope="device",
                                   hypothesis=nominal, targets=target_ids, write_value=False,
                                   extra_note="g26_calibrated writes this fit as the leaf value (the nominal world, "
                                              "uncertainty.er_mechanism 'dispersion') and keeps the solver value as "
                                              "the alternative 'solver', which the loss-mechanism worlds select (they "
                                              "take the solver dispersion). The value is conditional on "
                                              "the dispersion mechanism, which the data do not prefer over the loss "
                                              "mechanisms (docs/handoff/calibration.md)")
    if disp_decision.provenance != "calibrated":
        disp_decision.provenance = "assumed"           # a fitted value is no longer the solver's
    disp_decision.alternatives["cal_dispersion"] = {"value": disp_decision.value,
                                                    "provenance": disp_decision.provenance,
                                                    "note": disp_decision.note}
    decisions.append(disp_decision)
    composed, alt_by_path = coupling_worlds(problem_by_hyp, nominal_fit, profiles=profiles, fits=fits,
                                            rejected=rejected, n_grid=n_grid, inner=inner, log=log)
    _merge_world_alternatives(decisions, alt_by_path, cfg, target_ids)
    return decisions, composed


# return composed-world records and alternatives grouped by preset path.
# conditional fits hold shared leaves at nominal values.
# provenance instead uses unconditional profiles from all relevant non-rejected hypotheses.
# a branch-specific leaf uses its own base hypothesis.
# also apply the world's target-tolerance gate.
# retain conditional intervals as additional information.
def coupling_worlds(problem_by_hyp: Mapping[str, Problem], nominal_fit: FitResult, *,
                    profiles: Mapping[str, Mapping[str, Profile]], fits: Mapping[str, Any], rejected: set[str],
                    n_grid: int, inner: int, log=print) -> tuple[dict, dict]:
    composed: dict = {}
    alts: dict = {}
    for world, h, held, free in COUPLING_WORLDS:
        base = problem_by_hyp[h]
        world_problem = core.restrict(base, {k: nominal_fit.theta[k] for k in held}, hypothesis=f"composed {world}")
        world_fit = core.fit(world_problem, max_nfev=150)
        world_profiles = _profile_leaves(world_problem, world_fit, n_grid=n_grid, max_nfev=inner, log=log,
                                         names=list(free))
        log(f"  composed {world}: chi2 {world_fit.chi2:.4g} "
            f"{{{', '.join(f'{k}: {_fmt_sig(v)}' for k, v in world_fit.theta.items())}}}")
        rec = {"chi2": world_fit.chi2, "nominal_chi2": nominal_fit.chi2, "theta": world_fit.theta,
               "held": {k: nominal_fit.theta[k] for k in held}, "targets": world_fit.targets,
               "derived": world_fit.derived, "identifiable": {}}
        for leaf_name, alt in free.items():
            par = next(p for p in world_problem.params if p.name == leaf_name)
            hyps = [h] if leaf_name in BRANCH_LEAVES else [x for x in profiles if leaf_name in profiles[x]]
            sub = {x: profiles[x][leaf_name] for x in hyps if x in profiles and leaf_name in profiles[x]}
            ident = (_identifiability(par, sub, fits, rejected) if sub else
                     {"rule": f"kept: the base hypothesis {h} is rejected (no unconditional profile)",
                      "calibrated": False})
            ident = _tolerance_gate(ident, world_fit)
            conditional = world_profiles[leaf_name]
            ident["conditional_profile_95"] = list(conditional.interval)
            ident["conditional_on"] = {k: nominal_fit.theta[k] for k in held}
            rec["identifiable"][leaf_name] = ident
            hull = ident.get("profile_95")
            alts.setdefault(par.path, {})[alt] = {
                "value": float(world_fit.theta[leaf_name]), "calibrated": bool(ident["calibrated"]),
                "note": f"world {world} ({h}, held at the nominal fit: {', '.join(held)}): chi2 {world_fit.chi2:.3g} "
                        f"against {nominal_fit.chi2:.3g} for the nominal world; CAL-2 "
                        f"{'within' if ident.get('within_tolerance') else 'OUTSIDE'} tolerance; 95 % profile "
                        + (f"{_fmt_sig(hull[0])}-{_fmt_sig(hull[1])} (unconditional, over {', '.join(sub)})"
                           if hull else "none")
                        + f", {_fmt_sig(conditional.interval[0])}-{_fmt_sig(conditional.interval[1])} conditional on "
                          f"the held leaves; {ident['rule']}"}
        composed[world] = rec
    for world, h, held in COUPLING_WORLDS_INFO:
        world_problem = core.restrict(problem_by_hyp[h], {k: nominal_fit.theta[k] for k in held},
                                      hypothesis=f"composed {world}")
        world_fit = core.fit(world_problem, max_nfev=150)
        log(f"  composed {world} (reported): chi2 {world_fit.chi2:.4g}")
        composed[world] = {"chi2": world_fit.chi2, "nominal_chi2": nominal_fit.chi2, "theta": world_fit.theta,
                           "held": {k: nominal_fit.theta[k] for k in held}, "targets": world_fit.targets,
                           "reported_only": True}
    return composed, alts


# add the world alternatives ({leaf path — {name — ...}}) to the decisions (LeafDecision or record dicts).
# a leaf without a decision gets one that keeps its value and carries only the alternatives.
def _merge_world_alternatives(decisions: list, alt_by_path: Mapping[str, Mapping[str, dict]], cfg,
                              target_ids: list[str]) -> None:
    for path, alts in alt_by_path.items():
        d = next((x for x in decisions if (x.path if isinstance(x, LeafDecision) else x["path"]) == path), None)
        if d is None:
            prev_value, prev_provenance = _leaf_value_and_provenance(cfg, path)
            d = LeafDecision(path=path, value=prev_value, provenance=prev_provenance, previous_value=prev_value,
                             previous_provenance=prev_provenance, scope="device", targets=list(target_ids),
                             residual={}, identifiable={"rule": "the leaf value is not fitted; see its alternatives"},
                             hypothesis="composed worlds", note="value unchanged: the calibration writes the "
                             "alternatives the extinction-mechanism worlds select", write_value=False)
            decisions.append(d)
        prev_provenance = d.previous_provenance if isinstance(d, LeafDecision) else d["previous_provenance"]
        alt_store = d.alternatives if isinstance(d, LeafDecision) else d.setdefault("alternatives", {})
        for an, a in alts.items():
            alt_store[an] = {"value": a["value"], "note": a["note"],
                             "provenance": "calibrated" if a["calibrated"] else
                             (prev_provenance if prev_provenance not in ("absent", "derived", "override")
                              else "inferred")}


# the note text on the cold 0 V coupling ratio x0, Q_i and the notch floor of a coupling_er fit.
def _coupling_note(fr: FitResult) -> str:
    d = fr.derived or {}
    th = fr.theta
    return (f"cold 0 V coupling ratio x0 {_fmt_sig(d.get('x0_cold_0v', float('nan')))} ({d.get('branch_at_0v')}), "
            f"Q_i {_fmt_sig(d.get('q_intrinsic', float('nan')))}, instrument notch floor "
            f"{_fmt_sig(th.get('notch_floor_db', float('nan')))} dB (measurement nuisance, not written)")


# the context on which the coupling_er problems of a fit record were built — the record's leaves of every group before coupling_er in `GROUP_ORDER` (and the scratch ranges).
def coupling_context(record: Mapping[str, Any]) -> fit_groups.Context:
    ctx = fit_groups.Context(ranges={k: list(v) for k, v in record.get("context", {}).get("ranges", {}).items()})
    device_values = record.get("context", {}).get("device", {})
    for h in GROUP_ORDER[:GROUP_ORDER.index("coupling_er")]:
        paths = {d.get("path") for d in record.get("groups", {}).get(h, {}).get("decisions", [])}
        ctx.device.update({k: v for k, v in device_values.items() if k in paths})
    return ctx


# recompute the leaf decisions and composed worlds of the coupling_er group of a fit record, and rewrite the record.
# it uses the record's stored fits and profiles — the same code as a full run, without refitting the hypotheses or their profiles.
# the problems see the leaves of the groups before coupling_er in `GROUP_ORDER` (the heater), taken from the record.
def refresh_coupling_worlds(path: str | Path = RESULTS_PATH, *, log=print) -> dict:
    record = load_results(path)
    group = record["groups"]["coupling_er"]
    nominal = group["nominal"]
    fits = {h: _fit_from_dict(f) for h, f in group["fits"].items()}
    nominal_key = next((k for k in group["fits"] if k.startswith(nominal) and "CAL-1 centre" in k), nominal)
    profiles = {h: {n: _profile_from_dict(p) for n, p in ps.items()} for h, ps in group["profiles"].items()}
    rejected = {r["hypothesis"] for r in group["comparison"]["rows"] if r["likelihood_ratio_birge"] < REJECT_LR}
    targets = core.calibration_targets()
    ctx = coupling_context(record)
    problem_by_hyp = {p.hypothesis: p for p in fit_groups.coupling_er_problems(ctx, targets)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        decisions, composed = _coupling_decisions(ctx, problem_by_hyp, fits, profiles, rejected, nominal,
                                                  fits[nominal_key], n_grid=9, inner=30, log=log)
    group["decisions"] = [jsonable(d) for d in decisions]
    group["extra"]["composed_worlds"] = jsonable(composed)
    Path(path).write_text(json.dumps(record, indent=1, allow_nan=False, default=str))
    return record


# a Profile rebuilt from its record dict.
def _profile_from_dict(d: Mapping[str, Any]) -> Profile:
    return Profile(**{f.name: d[f.name] for f in dataclasses.fields(Profile)})


# a FitResult rebuilt from its record dict (without its profiles).
def _fit_from_dict(d: Mapping[str, Any]) -> FitResult:
    keys = {f.name for f in dataclasses.fields(FitResult)} - {"profiles"}
    return FitResult(**{k: d[k] for k in keys if k in d})


# Group offset (CAL-7)
def _group_offset(ctx, targets, *, n_grid, inner, quick, log):
    problems = fit_groups.offset_problems(ctx, targets)
    fits = _fit_hypotheses(problems, log=log)
    comparison = core.compare_hypotheses("offset", list(fits.values()))
    problem = problems[0]
    profiles = _profile_leaves(problem, fits[problem.hypothesis], n_grid=n_grid, max_nfev=inner, log=log)
    ident = _identifiability(problem.params[0], {problem.hypothesis: profiles[problem.params[0].name]}, fits, set())
    cfg = ctx.config(fit_groups.BENCH)
    decisions = [_leaf_decision(cfg, problem.params[0], fits[problem.hypothesis], ident, scope="device",
                                hypothesis=problem.hypothesis, targets=["CAL-7"],
                                extra_note="R3 closes it by construction: the fit confirms the cold-cavity "
                                           "definition of the phase model; the reference temperature is unknown")]
    return GroupResult("offset", ["CAL-7"], problem.hypothesis, fits, comparison, {problem.hypothesis: profiles},
                       decisions, {}, 0.0)


# Group junction (CAL-5)
def _group_junction(ctx, targets, *, n_grid, inner, quick, log):
    problems = fit_groups.junction_problems(ctx, targets)
    problem_by_hyp = {p.hypothesis: p for p in problems}
    fits = _fit_hypotheses(problems, log=log)
    analytic_fits = {h: f for h, f in fits.items() if h.startswith("analytic/")}
    lut_fits = {h: f for h, f in fits.items() if h.startswith("lut/")}
    comparison = core.compare_hypotheses("junction", list(analytic_fits.values()),
                                         note="measurement power of the G25 bias ladder")
    comparison_lut = core.compare_hypotheses("junction_lut", list(lut_fits.values()),
                                             note="LUT mode (reported variant)")
    rejected = _rejected_hypotheses(comparison)
    nominal = "analytic/p_in_4dbm"
    profiles = {h: _profile_leaves(problem_by_hyp[h], fits[h], n_grid=n_grid, max_nfev=inner, log=log) for h in fits
                if h not in rejected}
    cfg = ctx.config(fit_groups.BENCH)
    par = problem_by_hyp[nominal].params[0]
    ident = _identifiability(par, {h: profiles[h][par.name] for h in analytic_fits if h in profiles}, fits, rejected)
    decision = _leaf_decision(cfg, par, fits[nominal], ident, scope="device", hypothesis=nominal, targets=["CAL-5"],
                              extra_note="self-heating relief removed at the measurement power; the G25 power is "
                                         "unpublished, so the P_in -> 0 fit is kept as the alternative "
                                         "'cal5_p_in_0'. In the analytic mode the fill fraction cancels from the DC "
                                         "shift (the ring-level efficiency is this leaf)")
    alt = fits["analytic/p_in_0"]
    decision.alternatives["cal5_p_in_0"] = {"value": float(alt.theta[par.name]), "provenance": decision.provenance,
                                            "note": f"CAL-5 fitted at P_in -> 0 (no self-heating relief), chi2 "
                                                    f"{alt.chi2:.3g}"}
    fill = problem_by_hyp["lut/p_in_0"].params[0]
    fill_by_hyp = {h: (fits[h].theta[fill.name], profiles.get(h, {}).get(fill.name)) for h in lut_fits}
    fill_note = ("LUT mode (junction_rib160.json, reported variant; g26 runs the analytic law, where the fill "
                 "cancels): CAL-5 gives fill " + "; ".join(
                     f"{h.split('/')[1]}: {_fmt_sig(v)}"
                     + (f" (95 % {_fmt_sig(p.interval[0])}-{_fmt_sig(p.interval[1])}"
                        f"{', at the prior bound' if p.open_hi or p.open_lo else ''}"
                        f", chi2 {fits[h].chi2:.3g})" if p else "")
                     for h, (v, p) in fill_by_hyp.items()))
    fill_value, fill_provenance = _leaf_value_and_provenance(cfg, fit_groups.LEAF_FILL_FRACTION)
    fill_decision = LeafDecision(path=fit_groups.LEAF_FILL_FRACTION, value=fill_value, provenance=fill_provenance,
                                 previous_value=fill_value, previous_provenance=fill_provenance, scope="device",
                                 targets=["CAL-5"], residual=_residual_summary(fits["lut/p_in_0"]),
                                 identifiable=_identifiability(fill, {h: profiles[h][fill.name] for h in lut_fits
                                                                      if h in profiles}, fits, set()),
                                 hypothesis="lut (variant)", note=fill_note, write_value=False)
    return GroupResult("junction", ["CAL-5"], nominal, fits, comparison, profiles, [decision, fill_decision],
                       {"lut_comparison": comparison_lut}, 0.0)


# Group eo (CAL-10)
def _group_eo(ctx, targets, *, n_grid, inner, quick, log):
    problems = fit_groups.eo_problems(ctx, targets)
    problem_by_hyp = {p.hypothesis: p for p in problems}
    fits = _fit_hypotheses(problems, log=log)
    comparison = core.compare_hypotheses("eo", list(fits.values()), note="which of R_s and C_j carries the R C")
    rejected = _rejected_hypotheses(comparison)
    profiles = {h: _profile_leaves(problem_by_hyp[h], fits[h], n_grid=n_grid, max_nfev=inner, log=log) for h in fits
                if h not in rejected}
    nominal = "rs_free" if "rs_free" not in rejected else comparison.best
    cfg = ctx.config(fit_groups.BENCH)
    par = problem_by_hyp[nominal].params[0]
    ident = _identifiability(par, {h: profiles[h][par.name] for h in profiles if par.name in profiles[h]}, fits,
                             rejected)
    derived = fits[nominal].derived
    decision = _leaf_decision(
        cfg, par, fits[nominal], ident, scope="device", hypothesis=nominal, targets=["CAL-10"],
        extra_note=f"only the junction R C is identified (RC pole {_fmt_sig(derived.get('rc_pole_ghz', 0))} GHz "
                   f"with the 50 ohm driver; FWHM {_fmt_sig(derived.get('fwhm_ghz', 0))} GHz): C_j(0) is held at "
                   f"its assumed 15 fF; with R_s held at 100 ohm no C_j(0) in [6, 30] fF reaches 30.9 GHz "
                   f"(hypothesis c_free). The rib160 junction LUT gives C_j(1 V) 37.7 fF and R_s 82 ohm "
                   f"(extract_junction handoff), a similar R C with the opposite split")
    return GroupResult("eo", ["CAL-10"], nominal, fits, comparison, profiles, [decision], {"derived": derived}, 0.0)


# Group monitor (CAL-8)
def _group_monitor(ctx, targets, *, n_grid, inner, quick, log):
    problems = fit_groups.monitor_problems(ctx, targets)
    problem_by_hyp = {p.hypothesis: p for p in problems}
    fits = _fit_hypotheses(problems, log=log)
    comparison = core.compare_hypotheses("monitor", list(fits.values()), note="lock set-point of the G26 loop")
    rejected = _rejected_hypotheses(comparison)
    profiles = {h: _profile_leaves(problem_by_hyp[h], fits[h], n_grid=n_grid, max_nfev=inner, log=log) for h in fits
                if h not in rejected}
    nominal = "setpoint=half_max" if "setpoint=half_max" not in rejected else comparison.best
    cfg = ctx.config(fit_groups.BANK)
    par = problem_by_hyp[nominal].params[0]
    ident = _identifiability(par, {h: profiles[h][par.name] for h in profiles}, fits, rejected)
    decision = _leaf_decision(cfg, par, fits[nominal], ident, scope="system", hypothesis=nominal, targets=["CAL-8"],
                              extra_note="CAL-8 identifies only gain x line power x lock factor (the weak drop "
                                         "coupler); the lock factor is the unpublished set-point, so the gain is "
                                         "conditional on it")
    for h, f in fits.items():
        if h != nominal and h not in rejected:
            decision.alternatives[f"cal8_{h.split('=')[1]}_lock"] = {
                "value": float(f.theta[par.name]), "provenance": decision.provenance,
                "note": f"CAL-8 under the {h} lock (chi2 {f.chi2:.3g})"}
    return GroupResult("monitor", ["CAL-8"], nominal, fits, comparison, profiles, [decision], {}, 0.0)


# Group modulation_depth (CAL-14)
def _group_modulation_depth(ctx, targets, *, n_grid, inner, quick, log):
    problems = fit_groups.modulation_depth_problems(ctx, targets)
    problem_by_hyp = {p.hypothesis: p for p in problems}
    fits = _fit_hypotheses(problems, log=log)
    comparison = core.compare_hypotheses("modulation_depth", list(fits.values()), note="lock mode of the relOMA map")
    profiles = {h: _profile_leaves(problem_by_hyp[h], fits[h], n_grid=n_grid, max_nfev=inner, log=log) for h in fits}
    # the swing that closes CAL-14 also at the CAL-5 alternative efficiency (P_in -> 0 - a refit, a few seconds)
    junction_problem_by_hyp = {p.hypothesis: p for p in fit_groups.junction_problems(ctx, targets, lut_variant=False)}
    eff_p_in_0_pm_per_v = float(core.fit(junction_problem_by_hyp["analytic/p_in_0"]).theta["eff_at_0v_pm_per_v"])
    scan = fit_groups.modulation_depth_scan(ctx, targets, alt_effs={"cal5_p_in_0": eff_p_in_0_pm_per_v})
    dbm_reading = _reloma_dbm_reading(ctx, targets)
    nominal = "mode=oracle"
    nominal_fit = fits[nominal]
    cal14 = targets["CAL-14"]
    tolerance_db = float(cal14.tolerance["value"])
    reached = abs(nominal_fit.targets["CAL-14"]["prediction"] - float(cal14.value)) <= tolerance_db
    cfg = ctx.config(fit_groups.BANK)
    par = problem_by_hyp[nominal].params[0]
    # the hull over the lock modes that the data do not reject
    ident = _identifiability(par, {h: profiles[h][par.name] for h in profiles}, fits, _rejected_hypotheses(comparison))
    closing = _cal14_closing_drive(scan, float(cal14.value))
    note = (f"CAL-14 (oracle, the replica's primary) cannot be reached inside the swing prior: best "
            f"{_fmt_sig(nominal_fit.targets['CAL-14']['prediction'])} dB at {_fmt_sig(nominal_fit.theta[par.name])} V "
            f"(target -2.1 +- 0.3). Closing it needs a lumped depth (peak-to-peak resonance swing / FWHM) of about "
            f"{_fmt_sig(closing.get('depth', float('nan')), 3)}: {closing.get('text', '')}. The value is not "
            f"changed: the closing drive lies outside the swing prior (plausibility, the efficiency dependence and "
            f"the 'OMA in dBm' reading: docs/handoff/calibration.md)") if not reached else ""
    decision = _leaf_decision(cfg, par, nominal_fit, ident, scope="system", hypothesis=nominal, targets=["CAL-14"],
                              extra_note=note, write_value=bool(reached))
    if not reached:
        decision.provenance = decision.previous_provenance
        decision.value = decision.previous_value
    return GroupResult("modulation_depth", ["CAL-14"], nominal, fits, comparison, profiles, [decision],
                       {"scan": scan, "closing": closing, "dbm_reading": dbm_reading}, 0.0)


# where the relOMA of the rows, sorted by `key`, first rises through `target_db` (dB) — (`key` there, the depth there, the largest relOMA), by linear interpolation.
# (None, None, max) when it never reaches it.
def _interpolate_crossing(rows: list, key: str, target_db: float):
    rows = sorted(rows, key=lambda r: r[key])
    x = np.array([r[key] for r in rows])
    y = np.array([r["reloma_db"] for r in rows])
    depth = np.array([r["depth"] for r in rows])
    crossings = np.flatnonzero((y[:-1] < target_db) & (y[1:] >= target_db))
    if not crossings.size:
        return None, None, (float(np.max(y)) if y.size else float("nan"))
    i = int(crossings[0])
    fraction = (target_db - y[i]) / (y[i + 1] - y[i])
    return float(x[i] + fraction * (x[i + 1] - x[i])), float(depth[i] + fraction * (depth[i + 1] - depth[i])), float(np.max(y))


# the drive that reaches `target_db` (dB) in the CAL-14 scan of `groups.modulation_depth_scan`.
# by linear interpolation — the swing at the preset efficiency, the efficiency scale at a 1.3 Vpp swing, and the centered swing at each alternative efficiency of the scan (`alt_effs`).
# Also the swings at the tolerance edge (`target_db` - 0.3 dB, the CAL-14 tolerance), the median depth over the closing points (`depth`) and a summary sentence (`text`).
def _cal14_closing_drive(scan: dict, target_db: float) -> dict:
    out = {}
    context_rows = [r for r in scan["rows"] if r.get("eff_base", "context") == "context"]
    for name in scan.get("alt_effs", {}):
        sw, dp, mx = _interpolate_crossing([r for r in scan["rows"] if r.get("eff_base") == name], "swing_pp_v",
                                           target_db)
        out[f"centered_swing_v_at_{name}"] = sw
        out[f"centered_depth_at_{name}"] = dp
        out[f"centered_max_db_at_{name}"] = mx
        swing_tol_edge, _, _ = _interpolate_crossing([r for r in scan["rows"] if r.get("eff_base") == name],
                                                     "swing_pp_v", target_db - 0.3)
        out[f"centered_swing_v_tolerance_edge_at_{name}"] = swing_tol_edge
    swing_tol_edge, _, _ = _interpolate_crossing(
        [r for r in context_rows if r["level_convention"] == "centered" and r["eff_scale"] == 1.0], "swing_pp_v",
        target_db - 0.3)
    out["centered_swing_v_tolerance_edge"] = swing_tol_edge
    for conv in ("centered", "one_sided"):
        # the swing scan at scale 1
        rows = [r for r in context_rows if r["level_convention"] == conv and r["eff_scale"] == 1.0]
        rows.sort(key=lambda r: r["swing_pp_v"])
        x = np.array([r["swing_pp_v"] for r in rows])
        y = np.array([r["reloma_db"] for r in rows])
        dpt = np.array([r["depth"] for r in rows])
        k = np.flatnonzero((y[:-1] < target_db) & (y[1:] >= target_db))
        if k.size:
            i = int(k[0])
            f = (target_db - y[i]) / (y[i + 1] - y[i])
            out[f"{conv}_swing_v"] = float(x[i] + f * (x[i + 1] - x[i]))
            out[f"{conv}_depth"] = float(dpt[i] + f * (dpt[i + 1] - dpt[i]))
        else:
            out[f"{conv}_swing_v"] = None
            out[f"{conv}_max_db"] = float(np.max(y))
        # the efficiency scan at a 1.3 Vpp swing
        eff_rows = [r for r in context_rows if r["level_convention"] == conv and r["swing_pp_v"] == 1.3]
        eff_rows.sort(key=lambda r: r["eff_scale"])
        xs = np.array([r["eff_scale"] for r in eff_rows])
        ys = np.array([r["reloma_db"] for r in eff_rows])
        ds = np.array([r["depth"] for r in eff_rows])
        k = np.flatnonzero((ys[:-1] < target_db) & (ys[1:] >= target_db))
        if k.size:
            i = int(k[0])
            f = (target_db - ys[i]) / (ys[i + 1] - ys[i])
            out[f"{conv}_eff_scale_at_1p3v"] = float(xs[i] + f * (xs[i + 1] - xs[i]))
            out[f"{conv}_depth_eff"] = float(ds[i] + f * (ds[i + 1] - ds[i]))
    depths = [v for k, v in out.items() if (k.endswith("_depth") or k.endswith("_depth_eff")
                                            or k.startswith("centered_depth_at_")) and v is not None]
    out["depth"] = float(np.median(depths)) if depths else float("nan")
    centered_swing = out.get("centered_swing_v")
    centered_eff_scale = out.get("centered_eff_scale_at_1p3v")
    max_swing = max(r["swing_pp_v"] for r in context_rows) if context_rows else float("nan")
    bias_v = fit_groups.CAL14_BIAS_V
    parts = [(f"a {_fmt_sig(centered_swing, 3)} Vpp centered swing (levels "
              f"{_fmt_sig(bias_v - centered_swing / 2, 3)} to {_fmt_sig(bias_v + centered_swing / 2, 3)} V; the "
              f"junction grid clips at -0.5 V forward)"
              if centered_swing else f"no centered swing up to {_fmt_sig(max_swing, 3)} Vpp"),
             (f"x{_fmt_sig(centered_eff_scale, 3)} the junction efficiency at 1.3 Vpp" if centered_eff_scale
              else "more than x3 efficiency")]
    for name in scan.get("alt_effs", {}):
        alt_swing = out.get(f"centered_swing_v_at_{name}")
        parts.append(f"at the {name} efficiency ({_fmt_sig(scan['alt_effs'][name], 3)} pm/V) "
                     + (f"a {_fmt_sig(alt_swing, 3)} Vpp centered swing" if alt_swing else
                        f"no centered swing up to {_fmt_sig(max_swing, 3)} Vpp (best "
                        f"{_fmt_sig(out.get(f'centered_max_db_at_{name}', float('nan')), 3)} dB)"))
    if out.get("one_sided_swing_v") is None:
        parts.append(f"the one-sided convention saturates at "
                     f"{_fmt_sig(out.get('one_sided_max_db', float('nan')), 3)} dB")
    else:
        parts.append(f"a {_fmt_sig(out['one_sided_swing_v'], 3)} Vpp one-sided swing")
    out["text"] = "; ".join(parts)
    return out


# the other reading of CAL-14, 'OMA in dBm at the actual input power' (not OMA / 1 mW) — the CAL-14 replica at the preset line power, plus that power in dBm.
# Reported, not fitted (the ring power is unpublished).
def _reloma_dbm_reading(ctx, targets) -> dict:
    from mrm.validation import replicas
    cfg = ctx.config(fit_groups.BANK)
    r = replicas.relOMA_map(cfg, targets["CAL-14"], p_in_dbm="preset", with_variants=False)
    _, plant = replicas.resolve_plant(cfg)
    ring = replicas.StaticRing(plant)
    p_dbm = 10.0 * math.log10(ring.p_line_w / 1e-3)
    return {"reloma_db_at_preset_power": r.value, "p_ring_dbm": p_dbm, "oma_dbm": r.value + p_dbm,
            "target_db": float(targets["CAL-14"].value),
            "note": "relOMA re 1 mW (the printed definition) vs OMA in dBm at the actual ring power (an open "
                    "question); the ring power is unpublished (uncertainty 0 to +8 dBm), so this reading is not fitted"}


# Group dac (CAL-9a, CAL-9b, CAL-9c)
def _group_dac(ctx, targets, *, n_grid, inner, quick, log):
    problems = fit_groups.dac_problems(ctx, targets, n_dies=4 if quick else 8)
    problem_by_hyp = {p.hypothesis: p for p in problems}
    fits = _fit_hypotheses(problems, log=log, max_nfev=80)
    comparison = core.compare_hypotheses("dac", list(fits.values()),
                                         note="thermo-optic hypothesis x aggressor referencing")
    rejected = _rejected_hypotheses(comparison)
    nominal = comparison.best           # a reported fit (nothing is written unless it reaches the targets)
    # each CAL-9 evaluation locks 8 banks over an aggressor period (~2 s) - profile the best hypothesis only (the values are written only if a fit reaches the targets, which none does - the profiles are informative)
    profiles = {h: _profile_leaves(problem_by_hyp[h], fits[h], n_grid=max(5, n_grid - 4), max_nfev=max(15, inner // 2),
                                   log=log) for h in fits if h == comparison.best}
    cfg = ctx.config(fit_groups.BANK)
    nominal_fit = fits[nominal]
    decisions = []
    for par in problem_by_hyp[nominal].params:
        if par.role != "leaf":
            continue
        ident = _identifiability(par, {h: profiles[h][par.name] for h in profiles}, fits, rejected)
        reaches_targets = all(v["tolerance_level"] == "pass" for v in nominal_fit.targets.values())
        note = (f"CAL-9 under the voltage law, uniform aggressor, {nominal}; die offset nuisance "
                f"{_fmt_sig(nominal_fit.theta.get('die_resonance_offset_nm', float('nan')))} nm (not written). "
                + ("" if reaches_targets else "The fit does not reach CAL-9 within tolerance; the value is not "
                                              "changed."))
        path = par.path
        if par.name == "ring_sigma_scale":
            path = "variation.geometry.ring_sigma_scale"
        d = _leaf_decision(cfg, dataclasses.replace(par, path=path), nominal_fit, ident, scope="system",
                           hypothesis=nominal, targets=["CAL-9a", "CAL-9b", "CAL-9c"], extra_note=note,
                           write_value=reaches_targets)
        if not reaches_targets:
            d.provenance = d.previous_provenance
            d.value = d.previous_value
        decisions.append(d)
    return GroupResult("dac", ["CAL-9a", "CAL-9b", "CAL-9c"], nominal, fits, comparison, profiles, decisions, {}, 0.0)


# Group laser (CAL-12b)
def _group_laser(ctx, targets, *, n_grid, inner, quick, log):
    problems = fit_groups.laser_problems(ctx, targets)
    fits = _fit_hypotheses(problems, log=log)
    comparison = core.compare_hypotheses("laser", list(fits.values()))
    problem = problems[0]
    nominal_fit = fits[problem.hypothesis]
    profiles = _profile_leaves(problem, nominal_fit, n_grid=n_grid, max_nfev=inner, log=log)
    cfg = ctx.config(fit_groups.BANK)
    decisions = []
    for par in problem.params:
        ident = _identifiability(par, {problem.hypothesis: profiles[par.name]}, fits, set())
        decisions.append(_leaf_decision(cfg, par, nominal_fit, ident, scope="system", hypothesis=problem.hypothesis,
                                        targets=["CAL-12b"],
                                        extra_note="the Guide line-error population (median and quartiles); the "
                                                   "+-15 GHz is a yield limit, not a sigma (CAL-12b)"))
    return GroupResult("laser", ["CAL-12b"], problem.hypothesis, fits, comparison, {problem.hypothesis: profiles},
                       decisions, {}, 0.0)
