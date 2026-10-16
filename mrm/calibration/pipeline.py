"""The calibration run: every fit group in order, hypothesis comparison, profiles, the leaf decisions of the
half-prior rule, and the before/after replica residuals (07 P0-5, section 4.3).

Owner: calibration. ``python -m mrm.calibration`` or ``mrm calibrate`` runs it (:mod:`mrm.calibration.cli`). The
result is written to ``mrm/calibration/latest_fit.json`` and the calibrated preset to
``mrm/params/presets/g26_calibrated.yaml`` (:mod:`mrm.calibration.preset`).

Order (each group sees the leaves fitted by the groups before it, :class:`mrm.calibration.groups.Context`):
coupling_er -> heater -> offset -> junction -> eo -> monitor -> modulation_depth -> dac -> laser. The device groups
(CAL-1..5, 7, 10 on the G25 bench) come first because the G26 groups (CAL-8, 9, 14) run on a plant that takes its
device physics from them.

Decision rules (stated before the fits were run, see docs/handoff/calibration.md):

1. Hypotheses of a group are compared by chi2 (likelihood ratio, Birge-scaled ratio, AIC). The preset keeps the
   g26 nominal hypothesis of each uncertainty entry unless the data reject it (Birge-scaled likelihood ratio below
   :data:`REJECT_LR` against the best hypothesis); measurement-model hypotheses (an instrument floor, the swept
   device's droop transfer) take the best one. Rejected hypotheses are reported, never deleted from the preset.
2. A leaf is tagged ``calibrated`` only when its 95 % profile interval, taken as the hull of the intervals of every
   non-rejected hypothesis in which the leaf is free, is at most half its prior range (log coordinates for a log
   prior), is closed on both sides (not cut by the prior) and is one connected interval, the prior is a declared
   one (the leaf's ``range`` or an uncertainty interval, not fit bounds invented for a leaf that declares none) and
   the fit reproduces every target of the group within its tolerance (a fit that misses it is a model-form
   failure, not a calibration). Otherwise the leaf keeps its provenance (assumed, inferred, ...) and its note
   records the narrowed interval; its value is still set to the nominal-hypothesis fit when that fit reproduces the
   target within its tolerance, so that the preset reproduces its calibration data where it can.
3. A leaf that only one hypothesis uses (an extinction-mechanism coefficient) is written on that hypothesis'
   alternative, from a conditional fit with every other leaf at the nominal fit (the world the preset composes).
   Its provenance follows rule 2 applied to the UNCONDITIONAL profiles (every other parameter re-minimized, Q_L
   included); the narrower interval conditional on the held leaves is recorded in its note only.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
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
from mrm.calibration import groups as G
from mrm.calibration.core import FitResult, Problem, Profile, jsonable

__all__ = ["GROUP_ORDER", "REJECT_LR", "LeafDecision", "GroupResult", "CalibrationRun", "run", "RESULTS_PATH",
           "run_group", "write_results", "load_results"]

RESULTS_PATH = Path(__file__).resolve().parent / "latest_fit.json"
GROUP_ORDER = ("coupling_er", "heater", "offset", "junction", "eo", "monitor", "modulation_depth", "dac", "laser")
#: a hypothesis is rejected when its Birge-scaled likelihood ratio against the best hypothesis is below this
REJECT_LR = 0.05


@dataclasses.dataclass
class LeafDecision:
    """What the calibration writes for one preset leaf (or keeps, with the reason)."""

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
    write_value: bool = True            # False: only alternatives / note are written, the value stays


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


# =====================================================================================================================
# Helpers
# =====================================================================================================================
def _rejected(cmp: core.HypothesisResult) -> set[str]:
    return {r["hypothesis"] for r in cmp.rows if r["likelihood_ratio_birge"] < REJECT_LR}


def _hull(intervals: Sequence[Sequence[float]]) -> list[float]:
    return [min(float(i[0]) for i in intervals), max(float(i[1]) for i in intervals)]


def _leaf_info(cfg, path: str) -> tuple[Any, str]:
    try:
        leaf = cfg.leaf(path)
        return jsonable(leaf.value), str(leaf.provenance)
    except Exception:  # noqa: BLE001 - a leaf absent from the preset
        return None, "absent"


def _identifiable(par: core.Param, profs: Mapping[str, Profile], fits: Mapping[str, FitResult],
                  rejected: set[str]) -> dict[str, Any]:
    """Rule 2 of the module docstring for one leaf, from the profiles of every non-rejected hypothesis."""
    used = {h: p for h, p in profs.items() if h not in rejected}
    if not used:
        return {"rule": "no profile", "calibrated": False}
    hull = _hull([p.interval for p in used.values()])
    open_ = any(p.open_lo or p.open_hi for p in used.values())
    # components: gaps inside one hypothesis' profile, plus gaps between the hypotheses' intervals
    comps = 1 + sum(p.n_components - 1 for p in used.values())
    ivs = sorted((list(p.interval) for p in used.values()), key=lambda iv: iv[0])
    reach = ivs[0][1]
    for iv in ivs[1:]:
        if iv[0] > reach:
            comps += 1
        reach = max(reach, iv[1])
    wf = core.half_prior(par, hull)
    declared = bool(getattr(par, "declared_prior", True))
    ok = wf <= 0.5 and not open_ and comps == 1 and declared
    best_by_h = {h: fits[h].theta.get(par.name) for h in used}
    return {"profile_95": hull, "prior": [par.lo, par.hi], "prior_source": par.prior_source,
            "log_prior": par.log, "width_frac": wf, "open": open_, "components": comps,
            "per_hypothesis": {h: {"interval": p.interval, "interval_raw": p.interval_raw, "best": best_by_h[h],
                                   "open_lo": p.open_lo, "open_hi": p.open_hi, "unique": p.unique,
                                   "chi2_min": p.chi2_min, "threshold": p.threshold}
                               for h, p in used.items()},
            "unique": bool(ok and all(p.unique for p in used.values()) and len(used) == 1),
            "calibrated": bool(ok),
            "rule": ("calibrated: 95 % profile hull within half the prior, closed, one interval" if ok else
                     "kept: " + "; ".join(x for x in (
                         f"width {wf:.2f} of the prior > 0.5" if wf > 0.5 else "",
                         "interval reaches the prior bound (open)" if open_ else "",
                         f"{comps} disjoint intervals across hypotheses" if comps > 1 else "",
                         "no declared prior range (fit bounds only), so the half-prior rule cannot apply"
                         if not declared else "") if x))}


def _fit_targets(fr) -> Mapping[str, Any]:
    return fr.targets if isinstance(fr, FitResult) else fr["targets"]


def _gate(ident: dict, fr) -> dict:
    """Rule 2, last clause: a fit that misses the tolerance of any of its targets cannot calibrate a leaf (the
    targets' convention: the tolerance is the largest residual accepted before a model-form failure)."""
    miss = [t for t, v in _fit_targets(fr).items() if not v.get("within_tolerance")]
    ident = dict(ident, within_tolerance=not miss)
    if miss and ident.get("calibrated"):
        ident.update(calibrated=False, unique=False,
                     rule="kept: the fit misses the tolerance of " + ", ".join(miss)
                          + " (a model-form failure is not a calibration)")
    return ident


def _residuals(fr: FitResult) -> dict[str, Any]:
    return {t: {"max_abs_residual": v["max_abs_residual"], "z_max": v["z_max"],
                "tolerance_level": v["tolerance_level"]} for t, v in fr.targets.items()}


def _fmt(x: float, n: int = 4) -> str:
    return f"{x:.{n}g}"


def _decision(cfg, par: core.Param, fr: FitResult, ident: dict, *, scope: str, hypothesis: str, targets: list[str],
              extra_note: str = "", range_: list[float] | None = None, write_value: bool = True) -> LeafDecision:
    prev_v, prev_p = _leaf_info(cfg, par.path)
    if par.path in ("variation.geometry.ring_sigma_scale", "variation.laser.line_offset_ghz.sigma_scale"):
        prev_v, prev_p = 1.0, "assumed" if "geometry" in par.path else "inferred"   # a scale on the preset entry
    value = float(fr.theta[par.name])
    ident = _gate(ident, fr)
    prov = "calibrated" if ident.get("calibrated") else prev_p
    hull = ident.get("profile_95")
    note = (f"calibration ({', '.join(targets)}, hypothesis {hypothesis}): best fit {_fmt(value)}"
            + (f", 95 % profile {_fmt(hull[0])}-{_fmt(hull[1])} over the non-rejected hypotheses"
               f" ({ident['width_frac']:.2f} of the prior {_fmt(par.lo)}-{_fmt(par.hi)}, {par.prior_source})"
               if hull else "")
            + f"; {ident.get('rule', '')}"
            + (f"; was {prev_v!r} ({prev_p})" if prev_v is not None else "")
            + (f". {extra_note}" if extra_note else ""))
    return LeafDecision(path=par.path, value=value, provenance=prov, previous_value=prev_v, previous_provenance=prev_p,
                        scope=scope, targets=list(targets), residual=_residuals(fr), identifiable=ident,
                        hypothesis=hypothesis, note=note, range=range_, write_value=write_value)


def _fit_all(probs: Sequence[Problem], *, extra_starts: int = 0, max_nfev: int = 150, log=print) -> dict:
    fits = {}
    for p in probs:
        t = time.time()
        fits[p.hypothesis] = core.fit(p, max_nfev=max_nfev, extra_starts=extra_starts)
        log(f"  fit {p.group} {p.hypothesis}: chi2 {fits[p.hypothesis].chi2:.4g} "
            f"theta {{{', '.join(f'{k}: {_fmt(v)}' for k, v in fits[p.hypothesis].theta.items())}}} "
            f"({time.time() - t:.0f} s)")
    return fits


def _profile_leaves(prob: Problem, fr: FitResult, *, n_grid: int, max_nfev: int, log=print,
                    names: Sequence[str] | None = None) -> dict[str, Profile]:
    out = {}
    for par in prob.params:
        if par.role != "leaf" or (names is not None and par.name not in names):
            continue
        t = time.time()
        out[par.name] = core.profile(prob, fr, par.name, n_grid=n_grid, max_nfev=max_nfev)
        pr = out[par.name]
        log(f"    profile {prob.hypothesis} {par.name}: [{_fmt(pr.interval[0])}, {_fmt(pr.interval[1])}] "
            f"frac {pr.width_frac:.2f} open {pr.open_lo}/{pr.open_hi} ({time.time() - t:.0f} s)")
    fr.profiles.update(out)
    return out


# =====================================================================================================================
# Groups
# =====================================================================================================================
def run_group(name: str, ctx: G.Context, T: Mapping[str, Any], *, quick: bool = False, log=print) -> GroupResult:
    """Fit one group, compare its hypotheses, profile, decide its leaves and update ``ctx``."""
    t0 = time.time()
    n_grid = 7 if quick else 9
    inner = 25 if quick else 30
    fn = {"coupling_er": _group_coupling, "heater": _group_heater, "offset": _group_offset,
          "junction": _group_junction, "eo": _group_eo, "monitor": _group_monitor,
          "modulation_depth": _group_modulation_depth, "dac": _group_dac, "laser": _group_laser}[name]
    res = fn(ctx, T, n_grid=n_grid, inner=inner, quick=quick, log=log)
    res.runtime_s = time.time() - t0
    for d in res.decisions:
        if d.write_value and d.path and not d.path.startswith("variation.") and d.path != "uncertainty":
            (ctx.device if d.scope == "device" else ctx.system)[d.path] = d.value
    return res


def _group_coupling(ctx, T, *, n_grid, inner, quick, log):
    probs = G.coupling_er_problems(ctx, T)
    byname = {p.hypothesis: p for p in probs}
    fits = _fit_all(probs, extra_starts=0 if quick else 4, log=log)
    cmp = core.compare_hypotheses("coupling_er", list(fits.values()),
                                  note="branch x extinction mechanism x instrument floor (07 P1-5)")
    rej = _rejected(cmp)
    nominal = "under/dispersion/floor"
    if nominal in rej:
        nominal = cmp.best
    log(f"  coupling_er: best {cmp.best}, nominal {nominal}, rejected {sorted(rej)}")
    profs: dict[str, dict[str, Profile]] = {}
    for h, p in byname.items():
        if h in rej:
            continue
        # the nominal hypothesis at the full resolution, the others (they only widen the hull) coarser
        full = h == nominal
        profs[h] = _profile_leaves(p, fits[h], n_grid=n_grid if full else max(5, n_grid - 4),
                                   max_nfev=inner if full else max(12, inner // 2), log=log)
    fr = fits[nominal]
    # CAL-1 is a flat interval and Q_L trades against alpha_other along the CAL-2 valley, so the free best fit sits at
    # an arbitrary point of the valley: write the point at the centre of the CAL-1 interval when it fits as well
    # (delta chi2 < 0.5), keeping the profile intervals of the free fit
    q_mid = 0.5 * sum(float(x) for x in T["CAL-1"].value)
    centred = core.fit(core.restrict(byname[nominal], {"q_loaded": q_mid}, hypothesis=nominal), max_nfev=150)
    log(f"  coupling_er nominal at Q_L {q_mid:g}: chi2 {centred.chi2:.4g} (free {fr.chi2:.4g})")
    if centred.chi2 - fr.chi2 < 0.5:
        centred.theta = {"q_loaded": q_mid, **centred.theta}
        centred.profiles = fr.profiles
        fits[nominal + " (Q_L at the CAL-1 centre)"] = centred
        fr = centred
    decisions, composed = _coupling_decisions(ctx, byname, fits, profs, rej, nominal, fr, n_grid=n_grid,
                                              inner=inner, log=log)
    # only now does the nominal world take the fitted dispersion (the loss worlds above hold the solver value)
    ctx.device[G.P_DISP] = next(d for d in decisions if d.path == G.P_DISP).value
    extra = {"composed_worlds": composed,
             "topology_note": "the add-drop ring of the plant (kappa_d/kappa 0.007); the all-pass reading of 01 D14 "
                              "differs by the drop coupler only and is absorbed by alpha_other"}
    return GroupResult("coupling_er", ["CAL-1", "CAL-2"], nominal, fits, cmp, profs, decisions, extra, 0.0)


def _coupling_decisions(ctx, byname, fits, profs, rej, nominal, fr, *, n_grid, inner, log=print):
    """The leaf decisions of the coupling_er group (rules 2 and 3) from its fits and profiles."""
    cfg = ctx.config(G.BENCH)
    tg = ["CAL-1", "CAL-2"]
    decisions = []
    for par in byname[nominal].params:
        if par.role != "leaf" or par.path == G.P_DISP:
            continue
        sub = {h: pr[par.name] for h, pr in profs.items() if par.name in pr}
        ident = _identifiable(par, sub, fits, rej)
        decisions.append(_decision(cfg, par, fr, ident, scope="device", hypothesis=nominal, targets=tg,
                                   extra_note=_coupling_note(fr)))
    # the nominal mechanism's own leaf (dispersion): the hull of every non-rejected hypothesis that frees it (both
    # branches); written by the preset as the leaf value, the solver value kept as the alternative 'solver'
    disp = next(p for p in byname[nominal].params if p.path == G.P_DISP)
    dsub = {h: pr[disp.name] for h, pr in profs.items() if disp.name in pr}
    dd = _decision(cfg, disp, fr, _identifiable(disp, dsub, fits, rej), scope="device", hypothesis=nominal,
                   targets=tg, write_value=False,
                   extra_note="g26_calibrated writes this fit as the leaf value (the nominal world, "
                              "uncertainty.er_mechanism 'dispersion') and keeps the solver value as the alternative "
                              "'solver', which the loss-mechanism worlds select (07 P1-5: they take the solver "
                              "dispersion). The value is conditional on the dispersion mechanism, which the data do "
                              "not prefer over the loss mechanisms (docs/handoff/calibration.md)")
    if dd.provenance != "calibrated":
        dd.provenance = "assumed"           # a fitted value is no longer the solver's
    dd.alternatives["cal_dispersion"] = {"value": dd.value, "provenance": dd.provenance, "note": dd.note}
    decisions.append(dd)
    composed, alt_by_path = coupling_worlds(byname, fr, profs=profs, fits=fits, rejected=rej, n_grid=n_grid,
                                            inner=inner, log=log)
    _merge_world_alternatives(decisions, alt_by_path, cfg, tg)
    return decisions, composed


#: the worlds the calibrated preset composes from the CAL-1/CAL-2 hypotheses (rule 3): (world, base hypothesis,
#: leaves held at the nominal fit, {free leaf name: alternative name}). The loss mechanisms fit only on the
#: over-coupled branch (the mirror of the dispersion fit, 07 P1-5), so their worlds carry their own alpha_other.
COUPLING_WORLDS = (
    ("coupling_regime=over", "over/dispersion/floor", ("q_loaded", "dln_kappa2_through"),
     {"alpha_other_abs": "over"}),
    ("er_mechanism=ring_t_loss", "over/ring_t_loss/floor", ("q_loaded",),
     {"alpha_other_abs": "cal_ring_t_loss_branch", "dln_alpha_dt_ring": "ring_t_loss"}),
    ("er_mechanism=heater_t_loss", "over/heater_t_loss/floor", ("q_loaded",),
     {"alpha_other_abs": "cal_heater_t_loss_branch", "dln_alpha_dt_heater": "heater_t_loss"}),
)
#: the same loss worlds held on the nominal (under-coupled) branch: reported only, to show why the branch is tied
COUPLING_WORLDS_INFO = (
    ("er_mechanism=ring_t_loss on the nominal branch", "under/ring_t_loss/floor", ("q_loaded", "alpha_other_abs")),
    ("er_mechanism=heater_t_loss on the nominal branch", "under/heater_t_loss/floor",
     ("q_loaded", "alpha_other_abs")),
)
#: leaves whose value belongs to one branch: their unconditional interval is the base hypothesis' alone
BRANCH_LEAVES = ("alpha_other_abs",)


def coupling_worlds(byname: Mapping[str, Problem], fr: FitResult, *, profs: Mapping[str, Mapping[str, Profile]],
                    fits: Mapping[str, Any], rejected: set[str], n_grid: int, inner: int,
                    log=print) -> tuple[dict, dict]:
    """Conditional fits of the worlds the preset composes (COUPLING_WORLDS). Each free leaf's provenance follows the
    half-prior rule on its UNCONDITIONAL profiles (``profs``: every other parameter re-minimized, Q_L included;
    the base hypothesis alone for a branch leaf, every non-rejected hypothesis that frees it otherwise) and the
    tolerance gate on the world's own fit; the interval conditional on the held leaves is recorded in the note.
    Returns (record {world: ...}, {leaf path: {alternative name: {value, calibrated, note}}})."""
    composed: dict = {}
    alts: dict = {}
    for world, h, held, free in COUPLING_WORLDS:
        base = byname[h]
        cp = core.restrict(base, {k: fr.theta[k] for k in held}, hypothesis=f"composed {world}")
        cf = core.fit(cp, max_nfev=150)
        cprof = _profile_leaves(cp, cf, n_grid=n_grid, max_nfev=inner, log=log, names=list(free))
        log(f"  composed {world}: chi2 {cf.chi2:.4g} {{{', '.join(f'{k}: {_fmt(v)}' for k, v in cf.theta.items())}}}")
        rec = {"chi2": cf.chi2, "nominal_chi2": fr.chi2, "theta": cf.theta, "held": {k: fr.theta[k] for k in held},
               "targets": cf.targets, "derived": cf.derived, "identifiable": {}}
        for leaf_name, alt in free.items():
            par = next(p for p in cp.params if p.name == leaf_name)
            hyps = [h] if leaf_name in BRANCH_LEAVES else [x for x in profs if leaf_name in profs[x]]
            sub = {x: profs[x][leaf_name] for x in hyps if x in profs and leaf_name in profs[x]}
            ident = (_identifiable(par, sub, fits, rejected) if sub else
                     {"rule": f"kept: the base hypothesis {h} is rejected (no unconditional profile)",
                      "calibrated": False})
            ident = _gate(ident, cf)
            cond = cprof[leaf_name]
            ident["conditional_profile_95"] = list(cond.interval)
            ident["conditional_on"] = {k: fr.theta[k] for k in held}
            rec["identifiable"][leaf_name] = ident
            hull = ident.get("profile_95")
            alts.setdefault(par.path, {})[alt] = {
                "value": float(cf.theta[leaf_name]), "calibrated": bool(ident["calibrated"]),
                "note": f"world {world} ({h}, held at the nominal fit: {', '.join(held)}): chi2 {cf.chi2:.3g} "
                        f"against {fr.chi2:.3g} for the nominal world; CAL-2 "
                        f"{'within' if ident.get('within_tolerance') else 'OUTSIDE'} tolerance; 95 % profile "
                        + (f"{_fmt(hull[0])}-{_fmt(hull[1])} (unconditional, over {', '.join(sub)})"
                           if hull else "none")
                        + f", {_fmt(cond.interval[0])}-{_fmt(cond.interval[1])} conditional on the held leaves; "
                        f"{ident['rule']}"}
        composed[world] = rec
    for world, h, held in COUPLING_WORLDS_INFO:
        cp = core.restrict(byname[h], {k: fr.theta[k] for k in held}, hypothesis=f"composed {world}")
        cf = core.fit(cp, max_nfev=150)
        log(f"  composed {world} (reported): chi2 {cf.chi2:.4g}")
        composed[world] = {"chi2": cf.chi2, "nominal_chi2": fr.chi2, "theta": cf.theta,
                           "held": {k: fr.theta[k] for k in held}, "targets": cf.targets, "reported_only": True}
    return composed, alts


def _merge_world_alternatives(decisions: list, alt_by_path: Mapping[str, Mapping[str, dict]], cfg,
                              tg: list[str]) -> None:
    for path, alts in alt_by_path.items():
        d = next((x for x in decisions if (x.path if isinstance(x, LeafDecision) else x["path"]) == path), None)
        if d is None:
            prev_v, prev_p = _leaf_info(cfg, path)
            d = LeafDecision(path=path, value=prev_v, provenance=prev_p, previous_value=prev_v,
                             previous_provenance=prev_p, scope="device", targets=list(tg), residual={},
                             identifiable={"rule": "the leaf value is not fitted; see its alternatives"},
                             hypothesis="composed worlds", note="value unchanged: the calibration writes the "
                             "alternatives the extinction-mechanism worlds select", write_value=False)
            decisions.append(d)
        prev_p = d.previous_provenance if isinstance(d, LeafDecision) else d["previous_provenance"]
        target = d.alternatives if isinstance(d, LeafDecision) else d.setdefault("alternatives", {})
        for an, a in alts.items():
            target[an] = {"value": a["value"], "note": a["note"],
                          "provenance": "calibrated" if a["calibrated"] else
                          (prev_p if prev_p not in ("absent", "derived", "override") else "inferred")}


def _profile_from_dict(d: Mapping[str, Any]) -> Profile:
    return Profile(**{f.name: d[f.name] for f in dataclasses.fields(Profile)})


def _fit_from_dict(d: Mapping[str, Any]) -> FitResult:
    keys = {f.name for f in dataclasses.fields(FitResult)} - {"profiles"}
    return FitResult(**{k: d[k] for k in keys if k in d})


def refresh_coupling_worlds(path: str | Path = RESULTS_PATH, *, log=print) -> dict:
    """Recompute the leaf decisions and composed worlds of the coupling_er group of a fit record from its stored
    fits and profiles (the same code as a full run, without refitting the hypotheses or their profiles) and rewrite
    the record. Coupling is the first group, so its problems need no earlier context."""
    res = load_results(path)
    g = res["groups"]["coupling_er"]
    nom = g["nominal"]
    fits = {h: _fit_from_dict(f) for h, f in g["fits"].items()}
    key = next((k for k in g["fits"] if k.startswith(nom) and "CAL-1 centre" in k), nom)
    profs = {h: {n: _profile_from_dict(p) for n, p in ps.items()} for h, ps in g["profiles"].items()}
    rej = {r["hypothesis"] for r in g["comparison"]["rows"] if r["likelihood_ratio_birge"] < REJECT_LR}
    T = core.calibration_targets()
    ctx = G.Context()
    byname = {p.hypothesis: p for p in G.coupling_er_problems(ctx, T)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        decisions, composed = _coupling_decisions(ctx, byname, fits, profs, rej, nom, fits[key], n_grid=9,
                                                  inner=30, log=log)
    g["decisions"] = [jsonable(d) for d in decisions]
    g["extra"]["composed_worlds"] = jsonable(composed)
    Path(path).write_text(json.dumps(res, indent=1, allow_nan=False, default=str))
    return res


def _coupling_note(fr: FitResult) -> str:
    d = fr.derived or {}
    th = fr.theta
    return (f"cold 0 V coupling ratio x0 {_fmt(d.get('x0_cold_0v', float('nan')))} ({d.get('branch_at_0v')}), "
            f"Q_i {_fmt(d.get('q_intrinsic', float('nan')))}, instrument notch floor "
            f"{_fmt(th.get('notch_floor_db', float('nan')))} dB (measurement nuisance, not written)")


def _group_heater(ctx, T, *, n_grid, inner, quick, log):
    probs = G.heater_problems(ctx, T)
    byname = {p.hypothesis: p for p in probs}
    fits = _fit_all(probs, log=log)
    cmp = core.compare_hypotheses("heater", list(fits.values()),
                                  note="droop transfer from the swept device to the median device (CAL-4 vs CAL-3)")
    rej = _rejected(cmp)
    nominal = cmp.best
    profs = {h: _profile_leaves(p, fits[h], n_grid=n_grid, max_nfev=inner, log=log)
             for h, p in byname.items() if h not in rej}
    cfg = ctx.config(G.BENCH)
    fr = fits[nominal]
    tg = ["CAL-3a", "CAL-3b", "CAL-4"]
    dec = []
    der = fr.derived
    for par in byname[nominal].params:
        if par.role != "leaf":
            continue
        ident = _identifiable(par, {h: pr[par.name] for h, pr in profs.items()}, fits, rej)
        extra, rng = "", None
        if par.path == G.P_R0:
            extra = (f"hot resistance at P_FSR {_fmt(der['r_hot_at_p_fsr_ohm'])} ohm (CAL-3b: 1.195^2/25.45 = 56.1); "
                     f"the prior was widened from the leaf range [42, 58] because that range was computed with the "
                     f"earlier droop y = 0.00904 /mW (R0 = R_hot/(1 + y P_FSR))")
            rng = [35.0, 58.0]
        if par.path == G.P_RATIO:
            extra = (f"y = TCR x R_th,H = {_fmt(der['y_per_mw'])} /mW at TCR {_fmt(der['tcr_per_k'])} /K "
                     f"(R_th,H {_fmt(der['r_th_heater_k_per_mw'])} K/mW, Z_heater->ring "
                     f"{_fmt(der['z_heater_ring_k_per_mw'])} K/mW); only y is identified (06 E1): the ratio is "
                     f"calibrated for the family's TCR. Swept device {_fmt(der['efficiency_swept_device_nm_per_mw'])} "
                     f"nm/mW (06 E4: 0.348), median {_fmt(der['efficiency_median_nm_per_mw'])} nm/mW")
        if par.path == G.P_PFSR:
            extra = ("the R2 anchor with the FSR at the line, set so the bench protocol (FSR between the cold order "
                     "and the next red order) gives the printed 25.45 mW median")
        dec.append(_decision(cfg, par, fr, ident, scope="device", hypothesis=nominal, targets=tg, extra_note=extra,
                             range_=rng))
    return GroupResult("heater", tg, nominal, fits, cmp, profs, dec, {"derived": der}, 0.0)


def _group_offset(ctx, T, *, n_grid, inner, quick, log):
    probs = G.offset_problems(ctx, T)
    fits = _fit_all(probs, log=log)
    cmp = core.compare_hypotheses("offset", list(fits.values()))
    p = probs[0]
    prof = _profile_leaves(p, fits[p.hypothesis], n_grid=n_grid, max_nfev=inner, log=log)
    ident = _identifiable(p.params[0], {p.hypothesis: prof[p.params[0].name]}, fits, set())
    cfg = ctx.config(G.BENCH)
    dec = [_decision(cfg, p.params[0], fits[p.hypothesis], ident, scope="device", hypothesis=p.hypothesis,
                     targets=["CAL-7"], extra_note="R3 closes it by construction: the fit confirms the cold-cavity "
                                                   "definition (07 P1-6); the reference temperature is unknown")]
    return GroupResult("offset", ["CAL-7"], p.hypothesis, fits, cmp, {p.hypothesis: prof}, dec, {}, 0.0)


def _group_junction(ctx, T, *, n_grid, inner, quick, log):
    probs = G.junction_problems(ctx, T)
    byname = {p.hypothesis: p for p in probs}
    fits = _fit_all(probs, log=log)
    ana = {h: f for h, f in fits.items() if h.startswith("analytic/")}
    lut = {h: f for h, f in fits.items() if h.startswith("lut/")}
    cmp = core.compare_hypotheses("junction", list(ana.values()), note="measurement power of the G25 bias ladder")
    cmp_lut = core.compare_hypotheses("junction_lut", list(lut.values()), note="LUT mode (reported variant)")
    rej = _rejected(cmp)
    nominal = "analytic/p_in_4dbm"
    profs = {h: _profile_leaves(byname[h], fits[h], n_grid=n_grid, max_nfev=inner, log=log) for h in fits
             if h not in rej}
    cfg = ctx.config(G.BENCH)
    par = byname[nominal].params[0]
    ident = _identifiable(par, {h: profs[h][par.name] for h in ana if h in profs}, fits, rej)
    d = _decision(cfg, par, fits[nominal], ident, scope="device", hypothesis=nominal, targets=["CAL-5"],
                  extra_note="07 P1-7: the relief removed at the measurement power; the G25 power is unpublished, so "
                             "the P_in -> 0 fit is kept as the alternative 'cal5_p_in_0'. In the analytic mode the "
                             "fill fraction cancels from the DC shift (the ring-level efficiency is this leaf)")
    alt = fits["analytic/p_in_0"]
    d.alternatives["cal5_p_in_0"] = {"value": float(alt.theta[par.name]), "provenance": d.provenance,
                                     "note": f"CAL-5 fitted at P_in -> 0 (no self-heating relief), chi2 "
                                             f"{alt.chi2:.3g}"}
    fill = byname["lut/p_in_0"].params[0]
    fl = {h: (fits[h].theta[fill.name], profs.get(h, {}).get(fill.name)) for h in lut}
    fill_note = ("LUT mode (junction_rib160.json, reported variant; g26 runs the analytic law, where the fill "
                 "cancels): CAL-5 gives fill " + "; ".join(
                     f"{h.split('/')[1]}: {_fmt(v)}" + (f" (95 % {_fmt(p.interval[0])}-{_fmt(p.interval[1])}"
                                                        f"{', at the prior bound' if p.open_hi or p.open_lo else ''}"
                                                        f", chi2 {fits[h].chi2:.3g})" if p else "")
                     for h, (v, p) in fl.items()))
    pv, pp_ = _leaf_info(cfg, G.P_FILL)
    dfill = LeafDecision(path=G.P_FILL, value=pv, provenance=pp_, previous_value=pv, previous_provenance=pp_,
                         scope="device", targets=["CAL-5"], residual=_residuals(fits["lut/p_in_0"]),
                         identifiable=_identifiable(fill, {h: profs[h][fill.name] for h in lut if h in profs}, fits,
                                                    set()),
                         hypothesis="lut (variant)", note=fill_note, write_value=False)
    return GroupResult("junction", ["CAL-5"], nominal, fits, cmp, profs, [d, dfill],
                       {"lut_comparison": cmp_lut}, 0.0)


def _group_eo(ctx, T, *, n_grid, inner, quick, log):
    probs = G.eo_problems(ctx, T)
    byname = {p.hypothesis: p for p in probs}
    fits = _fit_all(probs, log=log)
    cmp = core.compare_hypotheses("eo", list(fits.values()), note="which of R_s and C_j carries the R C")
    rej = _rejected(cmp)
    profs = {h: _profile_leaves(byname[h], fits[h], n_grid=n_grid, max_nfev=inner, log=log) for h in fits
             if h not in rej}
    nominal = "rs_free" if "rs_free" not in rej else cmp.best
    cfg = ctx.config(G.BENCH)
    par = byname[nominal].params[0]
    ident = _identifiable(par, {h: profs[h][par.name] for h in profs if par.name in profs[h]}, fits, rej)
    der = fits[nominal].derived
    d = _decision(cfg, par, fits[nominal], ident, scope="device", hypothesis=nominal, targets=["CAL-10"],
                  extra_note=f"only the junction R C is identified (RC pole {_fmt(der.get('rc_pole_ghz', 0))} GHz "
                             f"with the 50 ohm driver; FWHM {_fmt(der.get('fwhm_ghz', 0))} GHz): C_j(0) is held at "
                             f"its assumed 15 fF; with R_s held at 100 ohm no C_j(0) in [6, 30] fF reaches 30.9 GHz "
                             f"(hypothesis c_free). The rib160 junction LUT gives C_j(1 V) 37.7 fF and R_s 82 ohm "
                             f"(extract_junction handoff), a similar R C with the opposite split")
    return GroupResult("eo", ["CAL-10"], nominal, fits, cmp, profs, [d], {"derived": der}, 0.0)


def _group_monitor(ctx, T, *, n_grid, inner, quick, log):
    probs = G.monitor_problems(ctx, T)
    byname = {p.hypothesis: p for p in probs}
    fits = _fit_all(probs, log=log)
    cmp = core.compare_hypotheses("monitor", list(fits.values()), note="lock set-point of the G26 loop")
    rej = _rejected(cmp)
    profs = {h: _profile_leaves(byname[h], fits[h], n_grid=n_grid, max_nfev=inner, log=log) for h in fits
             if h not in rej}
    nominal = "setpoint=half_max" if "setpoint=half_max" not in rej else cmp.best
    cfg = ctx.config(G.BANK)
    par = byname[nominal].params[0]
    ident = _identifiable(par, {h: profs[h][par.name] for h in profs}, fits, rej)
    d = _decision(cfg, par, fits[nominal], ident, scope="system", hypothesis=nominal, targets=["CAL-8"],
                  extra_note="CAL-8 identifies only gain x line power x lock factor (07 D7, P1-9); the lock factor "
                             "is the unpublished set-point, so the gain is conditional on it")
    for h, f in fits.items():
        if h != nominal and h not in rej:
            d.alternatives[f"cal8_{h.split('=')[1]}_lock"] = {
                "value": float(f.theta[par.name]), "provenance": d.provenance,
                "note": f"CAL-8 under the {h} lock (chi2 {f.chi2:.3g})"}
    return GroupResult("monitor", ["CAL-8"], nominal, fits, cmp, profs, [d], {}, 0.0)


def _group_modulation_depth(ctx, T, *, n_grid, inner, quick, log):
    probs = G.modulation_depth_problems(ctx, T)
    byname = {p.hypothesis: p for p in probs}
    fits = _fit_all(probs, log=log)
    cmp = core.compare_hypotheses("modulation_depth", list(fits.values()), note="lock mode of the relOMA map")
    profs = {h: _profile_leaves(byname[h], fits[h], n_grid=n_grid, max_nfev=inner, log=log) for h in fits}
    scan = G.modulation_depth_scan(ctx, T)
    alt_read = _reloma_dbm_reading(ctx, T)
    nominal = "mode=oracle"
    fr = fits[nominal]
    t14 = T["CAL-14"]
    tol = float(t14.tolerance["value"])
    reached = abs(fr.targets["CAL-14"]["prediction"] - float(t14.value)) <= tol
    cfg = ctx.config(G.BANK)
    par = byname[nominal].params[0]
    ident = _identifiable(par, {h: profs[h][par.name] for h in profs}, fits, set())
    closing = _closing_rows(scan, float(t14.value))
    note = (f"CAL-14 (oracle, the replica's primary) cannot be reached inside the swing prior: best "
            f"{_fmt(fr.targets['CAL-14']['prediction'])} dB at {_fmt(fr.theta[par.name])} V (target -2.1 +- 0.3). "
            f"Closing it needs a lumped depth (peak-to-peak resonance swing / FWHM) of about "
            f"{_fmt(closing.get('depth', float('nan')), 3)}: {closing.get('text', '')}. The value is not changed: "
            f"the closing drive lies outside the swing prior (plausibility, the efficiency dependence and the "
            f"'OMA in dBm' reading: docs/handoff/calibration.md)") if not reached else ""
    d = _decision(cfg, par, fr, ident, scope="system", hypothesis=nominal, targets=["CAL-14"],
                  extra_note=note, write_value=bool(reached))
    if not reached:
        d.provenance = d.previous_provenance
        d.value = d.previous_value
    return GroupResult("modulation_depth", ["CAL-14"], nominal, fits, cmp, profs, [d],
                       {"scan": scan, "closing": closing, "dbm_reading": alt_read}, 0.0)


def _closing_rows(scan: dict, target_db: float) -> dict:
    """Interpolated swing (at the preset efficiency) and efficiency scale (at 1.3 V) that reach ``target_db``."""
    out = {}
    for conv in ("centered", "one_sided"):
        rows = [r for r in scan["rows"] if r["level_convention"] == conv and r["eff_scale"] == 1.0]
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
        er = [r for r in scan["rows"] if r["level_convention"] == conv and r["swing_pp_v"] == 1.3]
        er.sort(key=lambda r: r["eff_scale"])
        xs = np.array([r["eff_scale"] for r in er])
        ys = np.array([r["reloma_db"] for r in er])
        ds = np.array([r["depth"] for r in er])
        k = np.flatnonzero((ys[:-1] < target_db) & (ys[1:] >= target_db))
        if k.size:
            i = int(k[0])
            f = (target_db - ys[i]) / (ys[i + 1] - ys[i])
            out[f"{conv}_eff_scale_at_1p3v"] = float(xs[i] + f * (xs[i + 1] - xs[i]))
            out[f"{conv}_depth_eff"] = float(ds[i] + f * (ds[i + 1] - ds[i]))
    depths = [v for k, v in out.items() if k.endswith("_depth") or k.endswith("_depth_eff")]
    out["depth"] = float(np.median(depths)) if depths else float("nan")
    c = out.get("centered_swing_v")
    e = out.get("centered_eff_scale_at_1p3v")
    out["text"] = (f"{'a ' + _fmt(c, 3) + ' Vpp centered swing (the levels clip at the -0.5 V forward edge of the '
                               'junction grid)' if c else 'no centered swing up to 4.5 Vpp'}, or "
                   f"{'x' + _fmt(e, 3) + ' the junction efficiency at 1.3 Vpp' if e else 'more than x3 efficiency'}"
                   f"; the one-sided convention saturates at {_fmt(out.get('one_sided_max_db', float('nan')), 3)} dB"
                   if out.get("one_sided_swing_v") is None else "")
    return out


def _reloma_dbm_reading(ctx, T) -> dict:
    """07 D9 / Q4: the alternative reading 'OMA in dBm at the actual input power' (not OMA / 1 mW): the CAL-14
    replica at the preset line power, plus that power in dBm."""
    from mrm.validation import replicas as R
    cfg = ctx.config(G.BANK)
    r = R.relOMA_map(cfg, T["CAL-14"], p_in_dbm="preset", with_variants=False)
    _, pp = R.resolve_plant(cfg)
    ring = R.StaticRing(pp)
    p_dbm = 10.0 * math.log10(ring.p_line_w / 1e-3)
    return {"reloma_db_at_preset_power": r.value, "p_ring_dbm": p_dbm, "oma_dbm": r.value + p_dbm,
            "target_db": float(T["CAL-14"].value),
            "note": "relOMA re 1 mW (07 D9, printed) vs OMA in dBm at the actual ring power (07 Q4, a question); "
                    "the ring power is unpublished (uncertainty 0 to +8 dBm), so this reading is not fitted"}


def _group_dac(ctx, T, *, n_grid, inner, quick, log):
    probs = G.dac_problems(ctx, T, n_dies=4 if quick else 8)
    byname = {p.hypothesis: p for p in probs}
    fits = _fit_all(probs, log=log, max_nfev=80)
    cmp = core.compare_hypotheses("dac", list(fits.values()),
                                  note="thermo-optic hypothesis x aggressor referencing (07 D2, D6)")
    rej = _rejected(cmp)
    nominal = cmp.best           # a reported fit (nothing is written unless it reaches the targets)
    # each CAL-9 evaluation locks 8 banks over an aggressor period (~2 s): profile the best hypothesis only (the
    # values are written only if a fit reaches the targets, which none does; the profiles are informative)
    profs = {h: _profile_leaves(byname[h], fits[h], n_grid=max(5, n_grid - 4), max_nfev=max(15, inner // 2),
                                log=log) for h in fits if h == cmp.best}
    cfg = ctx.config(G.BANK)
    fr = fits[nominal]
    dec = []
    for par in byname[nominal].params:
        if par.role != "leaf":
            continue
        ident = _identifiable(par, {h: profs[h][par.name] for h in profs}, fits, rej)
        ok = all(v["tolerance_level"] == "pass" for v in fr.targets.values())
        note = (f"CAL-9 under the voltage law, uniform aggressor, {nominal}; die offset nuisance "
                f"{_fmt(fr.theta.get('die_resonance_offset_nm', float('nan')))} nm (not written). "
                + ("" if ok else "The fit does not reach CAL-9 within tolerance; the value is not changed."))
        path = par.path
        if par.name == "ring_sigma_scale":
            path = "variation.geometry.ring_sigma_scale"
        d = _decision(cfg, dataclasses.replace(par, path=path), fr, ident, scope="system", hypothesis=nominal,
                      targets=["CAL-9a", "CAL-9b", "CAL-9c"], extra_note=note, write_value=ok)
        if not ok:
            d.provenance = d.previous_provenance
            d.value = d.previous_value
        dec.append(d)
    return GroupResult("dac", ["CAL-9a", "CAL-9b", "CAL-9c"], nominal, fits, cmp, profs, dec, {}, 0.0)


def _group_laser(ctx, T, *, n_grid, inner, quick, log):
    probs = G.laser_problems(ctx, T)
    fits = _fit_all(probs, log=log)
    cmp = core.compare_hypotheses("laser", list(fits.values()))
    p = probs[0]
    fr = fits[p.hypothesis]
    prof = _profile_leaves(p, fr, n_grid=n_grid, max_nfev=inner, log=log)
    cfg = ctx.config(G.BANK)
    dec = []
    for par in p.params:
        ident = _identifiable(par, {p.hypothesis: prof[par.name]}, fits, set())
        dec.append(_decision(cfg, par, fr, ident, scope="system", hypothesis=p.hypothesis, targets=["CAL-12b"],
                             extra_note="the Guide line-error population (median and quartiles); the +-15 GHz is a "
                                        "yield limit, not a sigma (CAL-12b)"))
    return GroupResult("laser", ["CAL-12b"], p.hypothesis, fits, cmp, {p.hypothesis: prof}, dec, {}, 0.0)


# =====================================================================================================================
# Run
# =====================================================================================================================
def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[2], timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def run(groups: Sequence[str] | None = None, *, quick: bool = False, out: str | Path | None = RESULTS_PATH,
        log: Callable[[str], None] = print, prior: CalibrationRun | dict | None = None) -> CalibrationRun:
    """Run the fit groups in :data:`GROUP_ORDER` (or the listed subset). For a subset, the previous record
    (``prior``, else the file ``out``, else :data:`RESULTS_PATH`) supplies the leaves of the earlier groups as
    context, and the records and context leaves of every group not rerun are carried over, so a partial rerun never
    drops the other groups from the record (a later group not rerun keeps its old fit; rerun it when an earlier
    group's leaves changed). Writes ``out`` (JSON) unless None."""
    T = core.calibration_targets()
    ctx = G.Context()
    names = list(GROUP_ORDER) if not groups else [g for g in GROUP_ORDER if g in groups]
    subset = names != list(GROUP_ORDER)
    prev = prior
    if subset and prev is None:
        src = Path(out) if out is not None and Path(out).exists() else RESULTS_PATH
        prev = load_results(src) if src.exists() else None
    prev_ctx = {}
    if prev is not None:
        prev_ctx = prev["context"] if isinstance(prev, dict) else prev.context
    if subset and names and names[0] != GROUP_ORDER[0] and prev is not None:
        done = GROUP_ORDER[:GROUP_ORDER.index(names[0])]
        # only the leaves of the groups BEFORE the first rerun group: a later group's leaf (e.g. the laser line
        # offset) must not enter an earlier group's fit, as it could not in a full run
        later: set[str] = set()
        if isinstance(prev, dict):
            for g, gv in prev.get("groups", {}).items():
                if g not in done:
                    later.update(d.get("path") for d in gv.get("decisions", []))
        ctx.device.update({k: v for k, v in prev_ctx.get("device", {}).items() if k not in later})
        ctx.system.update({k: v for k, v in prev_ctx.get("system", {}).items() if k not in later})
        ctx.ranges.update({k: list(v) for k, v in prev_ctx.get("ranges", {}).items()})
        log(f"context from the previous run (groups {', '.join(done)}): {len(ctx.device)} device, "
            f"{len(ctx.system)} system leaves")
    res = CalibrationRun(created_utc=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         git_commit=_git_commit(), groups={}, context={})
    if subset and isinstance(prev, dict):
        res.previous_groups = {g: v for g, v in prev.get("groups", {}).items() if g not in names}
        if res.previous_groups:
            res.notes.append(f"groups {', '.join(res.previous_groups)} from the run of {prev.get('created_utc')}")

    def merged_context() -> dict:
        out_ctx = {k: dict(prev_ctx.get(k, {})) for k in ("device", "system", "ranges")} if subset else \
            {"device": {}, "system": {}, "ranges": {}}
        out_ctx["device"].update(ctx.device)
        out_ctx["system"].update(ctx.system)
        out_ctx["ranges"].update(ctx.ranges)
        return out_ctx
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for g in names:
            log(f"group {g}")
            res.groups[g] = run_group(g, ctx, T, quick=quick, log=log)
            res.context = merged_context()
            if out is not None:
                write_results(res, out)
    return res


def write_results(res: CalibrationRun, path: str | Path = RESULTS_PATH) -> Path:
    path = Path(path)
    d = {"schema": "mrm.calibration.fit/1", "created_utc": res.created_utc, "git_commit": res.git_commit,
         "reject_lr": REJECT_LR, "context": res.context, "notes": res.notes, "before_after": res.before_after,
         "groups": {}}
    for g in GROUP_ORDER:
        if g in res.previous_groups and g not in res.groups:
            d["groups"][g] = res.previous_groups[g]
    for g, gr in res.groups.items():
        d["groups"][g] = {
            "targets": gr.targets, "nominal": gr.nominal, "runtime_s": gr.runtime_s,
            "comparison": jsonable(gr.comparison),
            "fits": {h: {k: v for k, v in jsonable(f).items() if k != "profiles"} for h, f in gr.fits.items()},
            "profiles": {h: {n: jsonable(p) for n, p in ps.items()} for h, ps in gr.profiles.items()},
            "decisions": [jsonable(x) for x in gr.decisions],
            "extra": jsonable(gr.extra)}
    d["groups"] = {g: d["groups"][g] for g in GROUP_ORDER if g in d["groups"]}
    path.write_text(json.dumps(d, indent=1, allow_nan=False, default=str))
    return path


def load_results(path: str | Path = RESULTS_PATH) -> dict:
    return json.loads(Path(path).read_text())
