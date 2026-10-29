# turn calibration decisions into bank and bench child presets.
# `preset_document` builds g26_calibrated from a saved fit record.
# `write_preset` writes it and verifies that it loads.
# changed leaves retain target IDs, residuals, profile intervals, and the reason for their provenance.
# a fitted value can keep its previous provenance when calibration rules fail.
# alternative hypotheses retain their fitted values and selection rules.
# coupling regime follows the extinction mechanism because those choices are tied by the fits.
# they must not be drawn as independent uncertainties.
# `bench_preset_document` applies only device decisions to the G25 bench.
# system leaves belong to G26.
# the bench uses its own child preset because combining two inheritance chains with the same parent can overwrite leaves.
from __future__ import annotations

import copy
import datetime
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from mrm.calibration import groups as fit_groups
from mrm.calibration.core import calibration_targets, jsonable

__all__ = ["PRESET_NAME", "PRESET_PATH", "write_preset", "preset_document", "bench_overrides", "bank_config",
           "bench_config", "BENCH_PRESET_NAME", "BENCH_PRESET_PATH", "bench_preset_document", "write_bench_preset"]

PRESET_NAME = "g26_calibrated"


def _preset_dir() -> Path:
    from mrm.params import config
    return Path(config.PRESET_DIR)


PRESET_PATH = _preset_dir() / f"{PRESET_NAME}.yaml"
# the calibrated G25 bench (the report's G25 companion of g26_calibrated)
BENCH_PRESET_NAME = "g25_test_vehicle_calibrated"
BENCH_PRESET_PATH = _preset_dir() / f"{BENCH_PRESET_NAME}.yaml"

# pseudo-paths of the pipeline that map onto variation entries (a scale on the entry's sigmas)
_VARIATION_PSEUDO_PATHS = {"variation.geometry.ring_sigma_scale", "variation.laser.line_offset_ghz.sigma_scale"}


# g26_calibrated.yaml
# write the calibrated preset (YAML with a header comment) and check that it loads.
def write_preset(results: Mapping[str, Any], path: str | Path = PRESET_PATH) -> Path:
    doc, header = preset_document(results)
    text = "\n".join("# " + h for h in header) + "\n" + yaml.safe_dump(doc, sort_keys=False, width=118,
                                                                     allow_unicode=False)
    path = Path(path)
    path.write_text(text)
    from mrm import params
    params.load(str(path))
    return path


# the calibrated preset as a mapping, plus the header comment lines.
# `results` — the fit record (the dict of `mrm.calibration.pipeline.load_results`).
def preset_document(results: Mapping[str, Any]) -> tuple[dict, list[str]]:
    g26_raw = _raw_g26()
    target_meta = _target_meta()
    doc: dict = {"extends": "g26"}
    groups = results["groups"]
    created = results.get("created_utc", datetime.datetime.now(datetime.timezone.utc).isoformat())
    doc["meta"] = {
        "name": PRESET_NAME, "kind": "preset",
        "description": ("g26 with the leaves fitted to the calibration targets CAL-1..CAL-14 (and only to them) "
                        "through the replicas; provenance calibrated only where the half-prior rule holds"),
        "generation_mix": g26_raw.get("meta", {}).get("generation_mix", "") + (
            " Calibrated (mrm.calibration, " + str(created) + "): G25 Q_L, coupling and extinction mechanism, heater "
            "P_FSR / R / droop, junction efficiency, R C (G25 bench targets); G26 monitor gain; Guide line-error "
            "population (see each leaf's targets)."),
    }
    variation: dict = {}
    uncertainty: dict = {}
    selections: dict = {}           # uncertainty entry -> choice -> {leaf path: alternative name}
    for gname, group in groups.items():
        for decision in group["decisions"]:
            path = decision["path"]
            if path in _VARIATION_PSEUDO_PATHS:
                if not decision.get("write_value", True):
                    continue
                _variation_update(variation, g26_raw, decision)
                continue
            g26_leaf = _raw_leaf(g26_raw, path)
            if g26_leaf is None:
                continue
            generation, source = target_meta.get(decision["targets"][0], ("none",
                "VAL-TARGETS")) if decision["targets"] else ("none", "D07")
            leaf: dict = {}
            if decision.get("write_value", True) and decision["value"] is not None:
                leaf["value"] = _round_sig(float(decision["value"]))
                if decision["provenance"] == "calibrated":
                    leaf.update(provenance="calibrated", generation=generation, source=source)
                else:
                    leaf.update(provenance=g26_leaf.get("provenance", decision["provenance"]),
                                generation=g26_leaf.get("generation", generation), source=g26_leaf.get("source",
                                    source))
            else:
                leaf["value"] = g26_leaf.get("value")          # unchanged (needed for a leaf mapping)
            leaf["note"] = decision["note"]
            leaf["targets"] = list(decision["targets"])
            leaf["residual"] = _residual_block(decision["residual"])
            leaf["identifiable"] = _identifiable_block(decision["identifiable"], decision["hypothesis"])
            if decision.get("range"):
                leaf["range"] = [float(x) for x in decision["range"]]
            if decision.get("alternatives"):
                alts = {}
                for an, a in decision["alternatives"].items():
                    alternative = {"value": _round_sig(float(a["value"])), "provenance": a.get("provenance", "assumed"),
                           "note": a.get("note", "")}
                    if alternative["provenance"] == "calibrated":
                        alternative.update(generation=generation, source=source)
                    alts[an] = alternative
                    sel = a.get("select")
                    if sel:
                        # a world of an uncertainty entry selects this alternative (e.g. thermo_optic=H86 its own heater leaves) - restate the parent's entry with the selection added
                        entry_selections = selections.setdefault(sel["entry"], {})
                        entry_selections.setdefault(sel["choice"], {})[path] = an
                leaf["alternatives"] = alts
            if path == fit_groups.LEAF_COUPLER_DISPERSION and "cal_dispersion" in (decision.get("alternatives") or {}):
                # the nominal world must be the config as written - the calibrated dispersion becomes the leaf value and the solver value an alternative that the loss-mechanism worlds select
                cal = decision["alternatives"]["cal_dispersion"]
                leaf["value"] = _round_sig(float(cal["value"]))
                if cal.get("provenance") == "calibrated":
                    leaf.update(provenance="calibrated", generation=generation, source=source)
                else:
                    leaf.update(provenance="assumed", generation=generation, source=source)
                leaf["alternatives"] = {"solver": {
                    "value": g26_leaf.get("value"), "provenance": g26_leaf.get("provenance"),
                    "generation": g26_leaf.get("generation"), "source": g26_leaf.get("source"),
                    "note": "the coupler artefact's dispersion (the g26 value), selected by the loss-mechanism "
                            "worlds of uncertainty.er_mechanism"}}
                # a child's uncertainty entry replaces the parent's - restate its provenance, generation and source
                unc_entry = (g26_raw.get("uncertainty") or {}).get("er_mechanism") or {}
                uncertainty["er_mechanism"] = {"kind": "hypothesis", "nominal": "dispersion",
                                               "choices": {"dispersion": {}},
                                               **{k: unc_entry[k] for k in ("provenance", "generation", "source")
                                                  if k in unc_entry}}
            _set_path(doc, path, leaf)
    if "er_mechanism" in uncertainty:
        # each loss mechanism selects its own coefficient, the solver dispersion and its own branch alpha_other
        coupling_alts = set((_get_path(doc, fit_groups.LEAF_ALPHA_OTHER) or {}).get("alternatives", {}))
        er_choices = uncertainty["er_mechanism"]["choices"]
        for mech, leaf_path in (("ring_t_loss", fit_groups.LEAF_DLN_ALPHA_DT_RING),
                                ("heater_t_loss", fit_groups.LEAF_DLN_ALPHA_DT_HEATER)):
            er_choices[mech] = {leaf_path: mech, fit_groups.LEAF_COUPLER_DISPERSION: "solver"}
            if f"cal_{mech}_branch" in coupling_alts:
                er_choices[mech][fit_groups.LEAF_ALPHA_OTHER] = f"cal_{mech}_branch"
        uncertainty["er_mechanism"]["note"] = (
            "Extinction-droop mechanism, calibrated on CAL-1/CAL-2: 'dispersion' (nominal) is the coupler dispersion "
            "fitted with the under-coupled branch (the leaf value); the loss mechanisms select the solver dispersion "
            "(alternative 'solver') and fit only on the over-coupled branch (the mirror solution), so each selects its "
            "own coefficient AND its own alpha_other (cal_<mechanism>_branch, Q_L at the nominal): the branch is tied "
            "to the mechanism, so this entry is the joint (mechanism, regime) hypothesis and "
            "uncertainty.coupling_regime is its label (follows: er_mechanism). Likelihood ratios: "
            "docs/handoff/calibration.md")
    worlds = ((groups.get("coupling_er") or {}).get("extra") or {}).get("composed_worlds") or {}
    if "er_mechanism" in uncertainty and (g26_raw.get("uncertainty") or {}).get("coupling_regime") is not None:
        uncertainty["coupling_regime"] = _coupling_regime_label(g26_raw, groups["coupling_er"], worlds,
                                                                uncertainty["er_mechanism"])
    for entry, choices in selections.items():
        unc_entry = copy.deepcopy(uncertainty.get(entry) or (g26_raw.get("uncertainty") or {}).get(entry))
        if unc_entry is None:
            raise ValueError(f"uncertainty.{entry} is not in g26: cannot select calibrated alternatives in it")
        for choice, sel in choices.items():
            if choice not in unc_entry["choices"]:
                raise ValueError(f"uncertainty.{entry} has no choice {choice!r}")
            unc_entry["choices"][choice] = {**(unc_entry["choices"][choice] or {}), **sel}
        selected_paths = ", ".join(sorted({p for v in choices.values() for p in v}))
        unc_entry["note"] = (str(unc_entry.get("note", "")) + f" [g26_calibrated: the calibration adds its own leaves "
                             f"to {', '.join(sorted(choices))} ({selected_paths}), refitted in that world]")
        uncertainty[entry] = unc_entry
    if uncertainty:
        doc["uncertainty"] = uncertainty
    if variation:
        doc["variation"] = variation
    header = _preset_header(results)
    return doc, header


# label each extinction-mechanism world with its fitted cold coupling branch.
# the nominal dispersion fit is under-coupled.
# the loss mechanisms fit on the over-coupled branch.
# crossed combinations miss target tolerance, so regime follows er_mechanism through a label entry instead of an independent draw.
# pinning a regime selects compatible mechanisms.
# the label describes the nominal Q_L state.
# later Q_L variation can cross critical coupling.
def _coupling_regime_label(raw: dict, g: Mapping[str, Any], worlds: Mapping[str, Any], er: dict) -> dict:
    branch_of = {}
    nominal = g.get("nominal")
    for mech in er["choices"]:
        rec = (g["fits"].get(nominal) if mech == er.get("nominal") else worlds.get(f"er_mechanism={mech}")) or {}
        b = (rec.get("derived") or {}).get("branch_at_0v")
        if b not in ("under", "over"):
            raise ValueError(f"coupling_er: the record gives no cold 0 V branch for er_mechanism={mech} "
                             f"(derived.branch_at_0v {b!r}); cannot label its coupling regime")
        branch_of[mech] = b
    labels = [c for c in ("under", "over") if c in branch_of.values()]
    g26_regime = copy.deepcopy((raw.get("uncertainty") or {}).get("coupling_regime")) or {}
    out = {"kind": "hypothesis", "follows": "er_mechanism", "map": dict(branch_of), "choices": {c: {} for c in labels},
           "nominal": branch_of[er["nominal"]]}
    out.update({k: g26_regime[k] for k in ("provenance", "generation", "source") if k in g26_regime})
    over = worlds.get("coupling_regime=over") or {}
    over_derived = over.get("derived") or {}
    over_within_tol = all(t.get("within_tolerance") for t in (over.get("targets") or {}).values()) if over else None
    x0 = {m: (worlds.get(f"er_mechanism={m}") or {}).get("derived", {}).get("x0_cold_0v") for m in branch_of}
    x0[er["nominal"]] = ((g["fits"].get(nominal) or {}).get("derived") or {}).get("x0_cold_0v")
    out["note"] = (
        "Coupling regime: under | over coupled at the 27.9 dB G25 notch. In the calibrated presets the regime is the "
        "second axis of the joint hypothesis uncertainty.er_mechanism (a label entry: follows er_mechanism, never "
        "drawn): the CAL-1/CAL-2 fit lands each extinction mechanism on one branch of the cold 0 V state (the G25 "
        "bench ring, x0 = kappa_e / (kappa_i + kappa_d) of the record), "
        + ", ".join(f"{m} {b}" + (f" (x0 {_round_sig(float(x0[m]), 4)})" if x0.get(m) is not None else "")
                    for m, b in branch_of.items())
        + ", and the crossed worlds miss the CAL-2 tolerance (docs/handoff/calibration.md 4.2). Pinning the label "
        "selects the mechanisms that give it (g26_overcoupled: the loss mechanisms; g26_low_eta, g26_hot_selfheat: "
        "dispersion). The label holds at the nominal Q_L; a q_loaded draw can move a world across critical coupling "
        "(see uncertainty.q_loaded)."
        + (f" g26's independent 'over' choice (alpha_other alternative 'over', the dispersion fit on the over branch) "
           f"is not a world here: chi2 {_round_sig(float(over['chi2']), 3)} against "
           f"{_round_sig(float(over['nominal_chi2']), 3)} for the nominal world, "
           + ("within" if over_within_tol else "outside") + " the CAL-2 tolerance, and it lands "
           + (f"{over_derived['branch_at_0v']}-coupled at 0 V (x0 {_round_sig(float(over_derived['x0_cold_0v']), 4)})"
              if over_derived.get("branch_at_0v") else "on an unknown branch")
           + "; the leaf alternative keeps its value" if over else ""))
    return out


# write the variation entry of a pseudo-path decision `d` (a scale on the entry's sigmas) into `variation`.
# `raw` — the g26 preset mapping.
def _variation_update(variation: dict, raw: dict, d: Mapping[str, Any]) -> None:
    scale = float(d["value"])
    raw_variation = raw.get("variation", {})
    if d["path"] == "variation.geometry.ring_sigma_scale":
        geometry = raw_variation.get("geometry", {})
        # a child's variation entry replaces each top key of the parent's (a shallow update) - restate the whole width / thickness mappings, or their die sigmas would be lost
        variation["geometry"] = {
            "width": {**geometry["width"], "ring_sigma_nm": _round_sig(float(geometry["width"]["ring_sigma_nm"]) * scale)},
            "thickness": {**geometry["thickness"],
                          "ring_sigma_nm": _round_sig(float(geometry["thickness"]["ring_sigma_nm"]) * scale)},
            "provenance": d["provenance"] if d["provenance"] == "calibrated" else geometry.get("provenance", "assumed"),
            "note": f"per-ring sigmas x{_round_sig(scale, 4)} ({', '.join(d['targets'])}): {d['note']}"}
    else:
        line_entry = raw_variation.get("laser.line_offset_ghz", {})
        variation["laser.line_offset_ghz"] = {
            "die_sigma": _round_sig(float(line_entry["die_sigma"]) * scale),
            "line_sigma": _round_sig(float(line_entry["line_sigma"]) * scale),
            "provenance": "calibrated" if d["provenance"] == "calibrated" else line_entry.get("provenance", "inferred"),
            "generation": line_entry.get("generation", "Guide"), "source": line_entry.get("source", "YOLE"),
            "note": f"die and line sigma x{_round_sig(scale, 4)} ({', '.join(d['targets'])}): {d['note']}"}


# the header comment lines of g26_calibrated.yaml.
def _preset_header(results: Mapping[str, Any]) -> list[str]:
    lines = [
        "g26_calibrated: g26 fitted to the calibration targets CAL-1 ... CAL-14 and only to them.",
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


# g25_test_vehicle_calibrated.yaml
# write the calibrated G25 bench preset (`bench_preset_document`) and check that it loads.
def write_bench_preset(results: Mapping[str, Any], path: str | Path | None = None) -> Path:
    doc, header = bench_preset_document(results)
    text = "\n".join("# " + h for h in header) + "\n" + yaml.safe_dump(doc, sort_keys=False, width=118,
                                                                     allow_unicode=False)
    path = Path(path) if path is not None else BENCH_PRESET_PATH
    path.write_text(text)
    from mrm import params
    params.load(str(path))
    return path


# return the calibrated G25 bench mapping and header comment lines.
# copy device leaves and their uncertainty choices from the bank child preset.
# keep bench generation settings and omit G26 system leaves.
# alternatives remain available, so named corners and thermo-optic worlds select calibrated values.
# raise if the bench already restates a copied leaf or uncertainty entry.
def bench_preset_document(results: Mapping[str, Any]) -> tuple[dict, list[str]]:
    from mrm.params import config
    doc, _ = preset_document(results)
    bench_raw = config.yaml_load((_preset_dir() / f"{fit_groups.BENCH}.yaml").read_text())
    device_paths = [d["path"] for g in results["groups"].values() for d in g["decisions"]
                    if d.get("scope") == "device"]
    out: dict = {"extends": fit_groups.BENCH, "meta": {
        "name": BENCH_PRESET_NAME, "kind": "preset",
        "description": ("g25_test_vehicle with the device leaves of g26_calibrated (the calibrated G25 bench: the "
                        "plant of the G25 calibration targets after the fit)"),
        "generation_mix": str((bench_raw.get("meta") or {}).get("generation_mix", "")).strip() + (
            " Calibrated device leaves (mrm.calibration, " + str(results.get("created_utc")) + "): Q_L, coupling "
            "and extinction mechanism, heater P_FSR / R / droop, offset, junction efficiency, R_s, as in "
            "g26_calibrated.")}}
    for path in device_paths:
        leaf = _get_path(doc, path)
        if leaf is None:
            continue
        if _raw_leaf(bench_raw, path) is not None:
            raise ValueError(f"{fit_groups.BENCH} restates {path}: the calibrated bench cannot take g26_calibrated's "
                             f"leaf")
        _set_path(out, path, copy.deepcopy(leaf))
    uncertainty = copy.deepcopy(doc.get("uncertainty") or {})
    conflicts = sorted(set(uncertainty) & set(bench_raw.get("uncertainty") or {}))
    if conflicts:
        raise ValueError(f"{fit_groups.BENCH} restates uncertainty entries {conflicts} that g26_calibrated restates too")
    if uncertainty:
        out["uncertainty"] = uncertainty
    header = [
        f"{BENCH_PRESET_NAME}: the G25 bench ({fit_groups.BENCH}) with the calibrated device leaves of {PRESET_NAME}.",
        "GENERATED by mrm.calibration (mrm calibrate --write-preset-only); edit the code or rerun, not this file.",
        f"Run {results.get('created_utc')} at commit {results.get('git_commit')}; fit record "
        "mrm/calibration/latest_fit.json; leaves, notes and rules as in g26_calibrated.yaml.",
    ]
    return out, header


# Configs for the replicas
# the device leaves of a calibration run as overrides of `g25_test_vehicle` (the G25 bench with the calibrated device physics. see the module comments).
def bench_overrides(results: Mapping[str, Any]) -> dict[str, Any]:
    return dict(results.get("context", {}).get("device", {}))


# `g25_test_vehicle` with the calibrated device leaves (and the plausible ranges that the calibration widened).
def bench_config(results: Mapping[str, Any]):
    context = fit_groups.Context(device=bench_overrides(results),
                             ranges={k: list(v) for k, v in results.get("context", {}).get("ranges", {}).items()})
    config = context.config(fit_groups.BENCH)
    config._calibration_context = context          # keeps the scratch directory of the widened child alive
    return config


# the Config of the preset file `path` (default g26_calibrated.yaml. the report also loads g25_test_vehicle_calibrated.yaml with it).
def bank_config(path: str | Path = PRESET_PATH):
    from mrm.validation import replicas
    return replicas.load_config(Path(path))


# Mapping helpers
# `x` rounded to `sig` significant digits (also inside lists and dicts). 0 and non-finite values as they are.
def _round_sig(x: Any, sig: int = 6) -> Any:
    if isinstance(x, float) and math.isfinite(x) and x != 0.0:
        return float(f"{x:.{sig}g}")
    if isinstance(x, list):
        return [_round_sig(v, sig) for v in x]
    if isinstance(x, dict):
        return {k: _round_sig(v, sig) for k, v in x.items()}
    return x


# the g26 preset file as a mapping (as written, no inheritance resolved).
def _raw_g26() -> dict:
    from mrm.params import config
    return config.yaml_load((_preset_dir() / "g26.yaml").read_text())


# the mapping at dotted `path` of a preset mapping, or None (also when the node is not a mapping).
def _raw_leaf(raw: dict, path: str) -> dict | None:
    node: Any = raw
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node if isinstance(node, dict) else None


# the node at dotted `path` of a nested mapping, or None.
def _get_path(tree: dict, path: str) -> Any:
    node: Any = tree
    for k in path.split("."):
        if not isinstance(node, dict) or k not in node:
            return None
        node = node[k]
    return node


# set the node at dotted `path` of a nested mapping (making the parents).
def _set_path(tree: dict, path: str, value: Any) -> None:
    keys = path.split(".")
    node = tree
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


# (generation, first source key) of every calibration target.
def _target_meta() -> dict[str, tuple[str, str]]:
    return {t.id: (t.generation, t.source[0] if t.source else "VAL-TARGETS") for t in calibration_targets().values()}


# the `residual` block of a leaf — per target, max |residual| (target unit), z_max and the tolerance verdict.
def _residual_block(res: Mapping[str, Any]) -> dict:
    block = {}
    for target_id, residual in res.items():
        block[target_id] = {"max_abs": _round_sig(residual.get("max_abs_residual")),
            "z_max": _round_sig(residual.get("z_max"), 3),
                    "tolerance": residual.get("tolerance_level")}
    return block


# the `identifiable` block of a leaf — the profile interval, the prior, the half-prior verdict and the rule.
def _identifiable_block(ident: Mapping[str, Any], hypothesis: str) -> dict:
    keys = ("profile_95", "prior", "width_frac", "open", "components", "unique", "calibrated")
    out = {k: _round_sig(jsonable(ident.get(k)), 5) for k in keys if k in ident}
    out["hypothesis"] = hypothesis
    out["rule"] = ident.get("rule")
    return out
