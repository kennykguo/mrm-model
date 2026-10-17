"""Write ``mrm/params/presets/g26_calibrated.yaml`` from the leaf decisions of a calibration run.

Owner: calibration. The preset ``extends: g26`` and restates only what the calibration changed (07 P1-14: a child
that changes a value restates provenance and generation): each fitted leaf carries ``targets`` (the CAL ids),
``residual`` (per target, the largest |prediction - value| in the target's unit, its z and the tolerance verdict at
the nominal fit) and ``identifiable`` (the 95 % profile interval, the prior, the width fraction, whether the interval
is closed and unique, and the half-prior rule outcome). A leaf that fails the half-prior rule keeps its provenance
(assumed, inferred, ...), with the narrowed interval in its note. Alternatives of hypotheses that the data do not
reject are kept and refitted (rule 3 of :mod:`mrm.calibration.pipeline`); nothing is deleted.

The G25 bench device for the replicas of the G25 targets is ``g25_test_vehicle`` with the device leaves of this
preset applied (:func:`bench_overrides`): ``g25_test_vehicle`` extends ``g26`` and a preset cannot extend two
chains that share a parent without the second one overwriting the first. The same bench is also written as a preset,
``g25_test_vehicle_calibrated`` (:func:`write_bench_preset`, a child of ``g25_test_vehicle`` that restates the device
leaves and uncertainty entries of this preset), for users and for the report.
"""
from __future__ import annotations

import copy
import datetime as _dt
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from mrm.calibration import groups as G
from mrm.calibration.core import calibration_targets, jsonable

__all__ = ["PRESET_NAME", "PRESET_PATH", "write_preset", "preset_document", "bench_overrides", "bank_config",
           "bench_config", "BENCH_PRESET_NAME", "BENCH_PRESET_PATH", "bench_preset_document", "write_bench_preset"]

PRESET_NAME = "g26_calibrated"


def _preset_dir() -> Path:
    from mrm.params import config as C
    return Path(C.PRESET_DIR)


PRESET_PATH = _preset_dir() / f"{PRESET_NAME}.yaml"
#: the calibrated G25 bench (the report's G25 companion of g26_calibrated)
BENCH_PRESET_NAME = "g25_test_vehicle_calibrated"
BENCH_PRESET_PATH = _preset_dir() / f"{BENCH_PRESET_NAME}.yaml"

#: pseudo-paths of the pipeline that map onto variation entries
_VARIATION_PSEUDO = {"variation.geometry.ring_sigma_scale", "variation.laser.line_offset_ghz.sigma_scale"}


def _round(x: Any, sig: int = 6) -> Any:
    if isinstance(x, float) and math.isfinite(x) and x != 0.0:
        return float(f"{x:.{sig}g}")
    if isinstance(x, list):
        return [_round(v, sig) for v in x]
    if isinstance(x, dict):
        return {k: _round(v, sig) for k, v in x.items()}
    return x


def _raw_g26() -> dict:
    from mrm.params import config as C
    return C.yaml_load((_preset_dir() / "g26.yaml").read_text())


def _raw_leaf(raw: dict, path: str) -> dict | None:
    node: Any = raw
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node if isinstance(node, dict) else None


def _get(tree: dict, path: str) -> Any:
    node: Any = tree
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


def _set(tree: dict, path: str, value: Any) -> None:
    keys = path.split(".")
    node = tree
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _target_meta() -> dict[str, tuple[str, str]]:
    """(generation, first source key) of every calibration target."""
    return {t.id: (t.generation, t.source[0] if t.source else "VAL-TARGETS") for t in calibration_targets().values()}


def _residual_block(res: Mapping[str, Any]) -> dict:
    out = {}
    for tid, r in res.items():
        out[tid] = {"max_abs": _round(r.get("max_abs_residual")), "z_max": _round(r.get("z_max"), 3),
                    "tolerance": r.get("tolerance_level")}
    return out


def _ident_block(ident: Mapping[str, Any], hypothesis: str) -> dict:
    keys = ("profile_95", "prior", "width_frac", "open", "components", "unique", "calibrated")
    out = {k: _round(jsonable(ident.get(k)), 5) for k in keys if k in ident}
    out["hypothesis"] = hypothesis
    out["rule"] = ident.get("rule")
    return out


def preset_document(results: Mapping[str, Any]) -> tuple[dict, list[str]]:
    """The preset as a mapping (from a results dict of :func:`mrm.calibration.pipeline.load_results`), plus the
    header comment lines."""
    raw = _raw_g26()
    tmeta = _target_meta()
    doc: dict = {"extends": "g26"}
    groups = results["groups"]
    created = results.get("created_utc", _dt.datetime.now(_dt.timezone.utc).isoformat())
    doc["meta"] = {
        "name": PRESET_NAME, "kind": "preset",
        "description": ("g26 with the leaves fitted to the calibration targets CAL-1..CAL-14 (07 P0-5, 4.3) through "
                        "the replicas; provenance calibrated only where the half-prior rule holds"),
        "generation_mix": raw.get("meta", {}).get("generation_mix", "") + (
            " Calibrated (mrm.calibration, " + str(created) + "): G25 Q_L, coupling and extinction mechanism, heater "
            "P_FSR / R / droop, junction efficiency, R C (G25 bench targets); G26 monitor gain; Guide line-error "
            "population (see each leaf's targets)."),
    }
    variation: dict = {}
    uncertainty: dict = {}
    selections: dict = {}           # uncertainty entry -> choice -> {leaf path: alternative name}
    for gname, g in groups.items():
        for d in g["decisions"]:
            path = d["path"]
            if path in _VARIATION_PSEUDO:
                if not d.get("write_value", True):
                    continue
                _variation_update(variation, raw, d)
                continue
            parent = _raw_leaf(raw, path)
            if parent is None:
                continue
            gen, src = tmeta.get(d["targets"][0], ("none", "VAL-TARGETS")) if d["targets"] else ("none", "D07")
            leaf: dict = {}
            if d.get("write_value", True) and d["value"] is not None:
                leaf["value"] = _round(float(d["value"]))
                if d["provenance"] == "calibrated":
                    leaf.update(provenance="calibrated", generation=gen, source=src)
                else:
                    leaf.update(provenance=parent.get("provenance", d["provenance"]),
                                generation=parent.get("generation", gen), source=parent.get("source", src))
            else:
                leaf["value"] = parent.get("value")          # unchanged (needed for a leaf mapping)
            leaf["note"] = d["note"]
            leaf["targets"] = list(d["targets"])
            leaf["residual"] = _residual_block(d["residual"])
            leaf["identifiable"] = _ident_block(d["identifiable"], d["hypothesis"])
            if d.get("range"):
                leaf["range"] = [float(x) for x in d["range"]]
            if d.get("alternatives"):
                alts = {}
                for an, a in d["alternatives"].items():
                    rec = {"value": _round(float(a["value"])), "provenance": a.get("provenance", "assumed"),
                           "note": a.get("note", "")}
                    if rec["provenance"] == "calibrated":
                        rec.update(generation=gen, source=src)
                    alts[an] = rec
                    sel = a.get("select")
                    if sel:
                        # a world of an uncertainty entry selects this alternative (e.g. thermo_optic=H86 its own
                        # heater leaves): restate the parent's entry with the selection added
                        ent = selections.setdefault(sel["entry"], {})
                        ent.setdefault(sel["choice"], {})[path] = an
                leaf["alternatives"] = alts
            if path == G.P_DISP and "cal_dispersion" in (d.get("alternatives") or {}):
                # the nominal world must be the config as written: the calibrated dispersion becomes the leaf value
                # and the solver value an alternative that the loss-mechanism worlds select
                cal = d["alternatives"]["cal_dispersion"]
                leaf["value"] = _round(float(cal["value"]))
                if cal.get("provenance") == "calibrated":
                    leaf.update(provenance="calibrated", generation=gen, source=src)
                else:
                    leaf.update(provenance="assumed", generation=gen, source=src)
                leaf["alternatives"] = {"solver": {
                    "value": parent.get("value"), "provenance": parent.get("provenance"),
                    "generation": parent.get("generation"), "source": parent.get("source"),
                    "note": "the coupler artefact's dispersion (the g26 value), selected by the loss-mechanism "
                            "worlds of uncertainty.er_mechanism"}}
                # a child's uncertainty entry replaces the parent's: restate its provenance, generation and source
                pu = (raw.get("uncertainty") or {}).get("er_mechanism") or {}
                uncertainty["er_mechanism"] = {"kind": "hypothesis", "nominal": "dispersion",
                                               "choices": {"dispersion": {}},
                                               **{k: pu[k] for k in ("provenance", "generation", "source") if k in pu}}
            _set(doc, path, leaf)
    if "er_mechanism" in uncertainty:
        coupling_alts = set((_get(doc, G.P_COUPLING) or {}).get("alternatives", {}))
        ch = uncertainty["er_mechanism"]["choices"]
        for mech, leaf_path in (("ring_t_loss", G.P_CT_RING), ("heater_t_loss", G.P_CT_HEATER)):
            ch[mech] = {leaf_path: mech, G.P_DISP: "solver"}
            if f"cal_{mech}_branch" in coupling_alts:
                ch[mech][G.P_COUPLING] = f"cal_{mech}_branch"
        uncertainty["er_mechanism"]["note"] = (
            "07 P1-5, calibrated on CAL-1/CAL-2: 'dispersion' (nominal) is the coupler dispersion fitted with the "
            "under-coupled branch (the leaf value); the loss mechanisms select the solver dispersion (alternative "
            "'solver') and fit "
            "only on the over-coupled branch (the mirror solution), so each selects its own coefficient AND its own "
            "alpha_other (cal_<mechanism>_branch, Q_L at the nominal): the branch is tied to the mechanism, so this "
            "entry is the joint (mechanism, regime) hypothesis and uncertainty.coupling_regime is its label "
            "(follows: er_mechanism). Likelihood ratios: docs/handoff/calibration.md")
    worlds = ((groups.get("coupling_er") or {}).get("extra") or {}).get("composed_worlds") or {}
    if "er_mechanism" in uncertainty and (raw.get("uncertainty") or {}).get("coupling_regime") is not None:
        uncertainty["coupling_regime"] = _coupling_regime_label(raw, groups["coupling_er"], worlds,
                                                                uncertainty["er_mechanism"])
    for entry, choices in selections.items():
        pu = copy.deepcopy(uncertainty.get(entry) or (raw.get("uncertainty") or {}).get(entry))
        if pu is None:
            raise ValueError(f"uncertainty.{entry} is not in g26: cannot select calibrated alternatives in it")
        for choice, sel in choices.items():
            if choice not in pu["choices"]:
                raise ValueError(f"uncertainty.{entry} has no choice {choice!r}")
            pu["choices"][choice] = {**(pu["choices"][choice] or {}), **sel}
        pu["note"] = (str(pu.get("note", "")) + f" [g26_calibrated: the calibration adds its own leaves to "
                      f"{', '.join(sorted(choices))} ({', '.join(sorted({p for v in choices.values() for p in v}))}), "
                      f"refitted in that world]")
        uncertainty[entry] = pu
    if uncertainty:
        doc["uncertainty"] = uncertainty
    if variation:
        doc["variation"] = variation
    header = _header(results)
    return doc, header


def _coupling_regime_label(raw: dict, g: Mapping[str, Any], worlds: Mapping[str, Any], er: dict) -> dict:
    """uncertainty.coupling_regime of the calibrated presets: a label entry of uncertainty.er_mechanism
    (``follows`` + ``map``, PARAMETERS_FORMAT 6.1), not an independent hypothesis (params review 2026-10-16,
    params-14). The CAL-1/CAL-2 fit ties the cold 0 V branch to the extinction mechanism: the dispersion world fits
    only under-coupled, each loss world only over-coupled (its own alpha_other), and the crossed worlds miss the CAL-2
    tolerance. So each world's label is the branch its mechanism's fit lands on (``derived.branch_at_0v`` of the
    record), and pinning the label (the g26 corners g26_overcoupled, g26_low_eta, g26_hot_selfheat) selects the
    mechanisms that give it. Before, g26's independent under | over draw was kept: the 'over' world was the
    dispersion fit pushed to the branch wall (under-coupled at 0 V, CAL-2 missed) and the loss worlds were labelled
    'under' while over-coupled, half of all worlds."""
    branch = {}
    nominal = g.get("nominal")
    for mech in er["choices"]:
        rec = (g["fits"].get(nominal) if mech == er.get("nominal") else worlds.get(f"er_mechanism={mech}")) or {}
        b = (rec.get("derived") or {}).get("branch_at_0v")
        if b not in ("under", "over"):
            raise ValueError(f"coupling_er: the record gives no cold 0 V branch for er_mechanism={mech} "
                             f"(derived.branch_at_0v {b!r}); cannot label its coupling regime")
        branch[mech] = b
    labels = [c for c in ("under", "over") if c in branch.values()]
    cr = copy.deepcopy((raw.get("uncertainty") or {}).get("coupling_regime")) or {}
    out = {"kind": "hypothesis", "follows": "er_mechanism", "map": dict(branch), "choices": {c: {} for c in labels},
           "nominal": branch[er["nominal"]]}
    out.update({k: cr[k] for k in ("provenance", "generation", "source") if k in cr})
    over = worlds.get("coupling_regime=over") or {}
    od = over.get("derived") or {}
    ok = all(t.get("within_tolerance") for t in (over.get("targets") or {}).values()) if over else None
    x0 = {m: (worlds.get(f"er_mechanism={m}") or {}).get("derived", {}).get("x0_cold_0v") for m in branch}
    x0[er["nominal"]] = ((g["fits"].get(nominal) or {}).get("derived") or {}).get("x0_cold_0v")
    out["note"] = (
        "07 P1-5 under | over coupled at the 27.9 dB G25 notch. In the calibrated presets the regime is the second "
        "axis of the joint hypothesis uncertainty.er_mechanism (a label entry: follows er_mechanism, never drawn): "
        "the CAL-1/CAL-2 fit lands each extinction mechanism on one branch of the cold 0 V state (the G25 bench "
        "ring, x0 = kappa_e / (kappa_i + kappa_d) of the record), "
        + ", ".join(f"{m} {b}" + (f" (x0 {_round(float(x0[m]), 4)})" if x0.get(m) is not None else "")
                    for m, b in branch.items())
        + ", and the crossed worlds miss the CAL-2 tolerance (docs/handoff/calibration.md 4.2). Pinning the label "
        "selects the mechanisms that give it (g26_overcoupled: the loss mechanisms; g26_low_eta, g26_hot_selfheat: "
        "dispersion). The label holds at the nominal Q_L; a q_loaded draw can move a world across critical coupling "
        "(see uncertainty.q_loaded)."
        + (f" g26's independent 'over' choice (alpha_other alternative 'over', the dispersion fit on the over branch) "
           f"is not a world here: chi2 {_round(float(over['chi2']), 3)} against "
           f"{_round(float(over['nominal_chi2']), 3)} for the nominal world, "
           + ("within" if ok else "outside") + " the CAL-2 tolerance, and it lands "
           + (f"{od['branch_at_0v']}-coupled at 0 V (x0 {_round(float(od['x0_cold_0v']), 4)})"
              if od.get("branch_at_0v") else "on an unknown branch")
           + "; the leaf alternative keeps its value" if over else ""))
    return out


def _variation_update(variation: dict, raw: dict, d: Mapping[str, Any]) -> None:
    s = float(d["value"])
    rv = raw.get("variation", {})
    if d["path"] == "variation.geometry.ring_sigma_scale":
        geo = rv.get("geometry", {})
        # a child's variation entry replaces each top key of the parent's (a shallow update): restate the whole
        # width / thickness mappings, or their die sigmas would be lost
        variation["geometry"] = {
            "width": {**geo["width"], "ring_sigma_nm": _round(float(geo["width"]["ring_sigma_nm"]) * s)},
            "thickness": {**geo["thickness"], "ring_sigma_nm": _round(float(geo["thickness"]["ring_sigma_nm"]) * s)},
            "provenance": d["provenance"] if d["provenance"] == "calibrated" else geo.get("provenance", "assumed"),
            "note": f"per-ring sigmas x{_round(s, 4)} ({', '.join(d['targets'])}): {d['note']}"}
    else:
        e = rv.get("laser.line_offset_ghz", {})
        variation["laser.line_offset_ghz"] = {
            "die_sigma": _round(float(e["die_sigma"]) * s), "line_sigma": _round(float(e["line_sigma"]) * s),
            "provenance": "calibrated" if d["provenance"] == "calibrated" else e.get("provenance", "inferred"),
            "generation": e.get("generation", "Guide"), "source": e.get("source", "YOLE"),
            "note": f"die and line sigma x{_round(s, 4)} ({', '.join(d['targets'])}): {d['note']}"}


def _header(results: Mapping[str, Any]) -> list[str]:
    lines = [
        "g26_calibrated: g26 fitted to the calibration targets CAL-1 ... CAL-14 and only to them (07 P0-5, 4.3).",
        "GENERATED by mrm.calibration (mrm calibrate --write-preset); edit the code or rerun, not this file.",
        f"Run {results.get('created_utc')} at commit {results.get('git_commit')}; fit record "
        "mrm/calibration/latest_fit.json; method and findings docs/handoff/calibration.md.",
        "Rules: least squares + profile likelihood; a leaf is 'calibrated' only when the hull of its 95 % profile",
        "intervals over the non-rejected hypotheses is at most half its declared prior range, closed and one",
        "interval, and the fit meets every target's tolerance; otherwise it keeps its provenance and the narrowed",
        "interval is in its note. Every calibrated value is conditional on the hypotheses named in its",
        "identifiable.hypothesis and note (CAL-2: an instrument notch floor). Not fitted: CAL-6 (the H86",
        "hypothesis), CAL-11 / 12a / 12c / 12d / 13a / 13b (populations, scenario bounds and scenario calibrations,",
        "residuals reported), CAL-14 (not reached inside the swing prior, see the swing note) and CAL-9 (no",
        "hypothesis reaches it within tolerance, see the full-scale note): their leaves keep their values.",
    ]
    return lines


def write_preset(results: Mapping[str, Any], path: str | Path = PRESET_PATH) -> Path:
    """Write the calibrated preset (YAML with a header comment) and check that it loads."""
    doc, header = preset_document(results)
    text = "\n".join("# " + h for h in header) + "\n" + yaml.safe_dump(doc, sort_keys=False, width=118,
                                                                     allow_unicode=False)
    path = Path(path)
    path.write_text(text)
    from mrm import params
    params.load(str(path))
    return path


def bench_preset_document(results: Mapping[str, Any]) -> tuple[dict, list[str]]:
    """The calibrated G25 bench as a preset (integrator, calibration stage): ``g25_test_vehicle`` plus, restated
    exactly as in g26_calibrated, every leaf of decision scope ``device`` and the uncertainty entries g26_calibrated
    restates (their choices select device alternatives: the loss-mechanism worlds, the coupling branch, the heater
    leaves of the H86 world). The system leaves (monitor gain, DAC, drive, laser population) are G26 leaves and are
    not applied to the bench, as in :func:`bench_config`. Unlike the overrides of :func:`bench_config`, a preset keeps
    the alternatives, so the named corners and the H86 world select the calibrated alternatives on the bench too."""
    from mrm.params import config as C
    doc, _ = preset_document(results)
    bench_raw = C.yaml_load((_preset_dir() / f"{G.BENCH}.yaml").read_text())
    device = [d["path"] for g in results["groups"].values() for d in g["decisions"] if d.get("scope") == "device"]
    out: dict = {"extends": G.BENCH, "meta": {
        "name": BENCH_PRESET_NAME, "kind": "preset",
        "description": ("g25_test_vehicle with the device leaves of g26_calibrated (the calibrated G25 bench: the "
                        "plant of the G25 calibration targets after the fit)"),
        "generation_mix": str((bench_raw.get("meta") or {}).get("generation_mix", "")).strip() + (
            " Calibrated device leaves (mrm.calibration, " + str(results.get("created_utc")) + "): Q_L, coupling "
            "and extinction mechanism, heater P_FSR / R / droop, offset, junction efficiency, R_s, as in "
            "g26_calibrated.")}}
    for path in device:
        leaf = _get(doc, path)
        if leaf is None:
            continue
        if _raw_leaf(bench_raw, path) is not None:
            raise ValueError(f"{G.BENCH} restates {path}: the calibrated bench cannot take g26_calibrated's leaf")
        _set(out, path, copy.deepcopy(leaf))
    unc = copy.deepcopy(doc.get("uncertainty") or {})
    clash = sorted(set(unc) & set(bench_raw.get("uncertainty") or {}))
    if clash:
        raise ValueError(f"{G.BENCH} restates uncertainty entries {clash} that g26_calibrated restates too")
    if unc:
        out["uncertainty"] = unc
    header = [
        f"{BENCH_PRESET_NAME}: the G25 bench ({G.BENCH}) with the calibrated device leaves of {PRESET_NAME}.",
        "GENERATED by mrm.calibration (mrm calibrate --write-preset-only); edit the code or rerun, not this file.",
        f"Run {results.get('created_utc')} at commit {results.get('git_commit')}; fit record "
        "mrm/calibration/latest_fit.json; leaves, notes and rules as in g26_calibrated.yaml.",
    ]
    return out, header


def write_bench_preset(results: Mapping[str, Any], path: str | Path | None = None) -> Path:
    """Write the calibrated G25 bench preset (:func:`bench_preset_document`) and check that it loads."""
    doc, header = bench_preset_document(results)
    text = "\n".join("# " + h for h in header) + "\n" + yaml.safe_dump(doc, sort_keys=False, width=118,
                                                                     allow_unicode=False)
    path = Path(path) if path is not None else BENCH_PRESET_PATH
    path.write_text(text)
    from mrm import params
    params.load(str(path))
    return path


def bench_overrides(results: Mapping[str, Any]) -> dict[str, Any]:
    """The device leaves of a calibration run as overrides of ``g25_test_vehicle`` (the G25 bench with the
    calibrated device physics; module docstring)."""
    return dict(results.get("context", {}).get("device", {}))


def bench_config(results: Mapping[str, Any]):
    """``g25_test_vehicle`` with the calibrated device leaves (and the plausible ranges the calibration widened)."""
    ctx = G.Context(device=bench_overrides(results),
                    ranges={k: list(v) for k, v in results.get("context", {}).get("ranges", {}).items()})
    cfg = ctx.config(G.BENCH)
    cfg._calibration_context = ctx          # keeps the scratch directory of the widened child alive
    return cfg


def bank_config(path: str | Path = PRESET_PATH):
    from mrm.validation import replicas as R
    return R.load_config(Path(path))
