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
chains that share a parent without the second one overwriting the first.
"""
from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from mrm.calibration import groups as G
from mrm.calibration.core import calibration_targets, jsonable

__all__ = ["PRESET_NAME", "PRESET_PATH", "write_preset", "preset_document", "bench_overrides", "bank_config",
           "bench_config"]

PRESET_NAME = "g26_calibrated"


def _preset_dir() -> Path:
    from mrm.params import config as C
    return Path(C.PRESET_DIR)


PRESET_PATH = _preset_dir() / f"{PRESET_NAME}.yaml"

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
                uncertainty["er_mechanism"] = {"kind": "hypothesis", "nominal": "dispersion",
                                               "choices": {"dispersion": {}}}
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
            "alpha_other (cal_<mechanism>_branch, Q_L at the nominal): the branch is tied to the mechanism and "
            "overrides uncertainty.coupling_regime in those worlds (entries apply in name order). Likelihood ratios: "
            "docs/handoff/calibration.md")
    if uncertainty:
        doc["uncertainty"] = uncertainty
    if variation:
        doc["variation"] = variation
    header = _header(results)
    return doc, header


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
        "intervals over the non-rejected hypotheses is at most half its prior range, closed and one interval;",
        "otherwise it keeps its provenance and the narrowed interval is in its note. Not fitted: CAL-6 (the H86",
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
