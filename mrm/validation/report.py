# Generate the validation report docs/VALIDATION.md from the replicas, the target files and the preset.
# `generate()` computes cached parts and writes the report.
# `render()` builds Markdown and SVG charts from that cache plus the current preset and target files.
# `main()` runs the command line - `mrm report` or `python -m mrm.validation.report [--preset g26]`.
# `PARTS` names the four expensive computations stored in `docs/validation/<preset>/report_data.json`.
# `--fast` renders from that cache.
# `--recompute a,b` computes only the named parts.
# each part records its source hashes, presets, computation time and memory use.
# `check()` compares a fast render with the saved report.
# `_section_*` functions build the seven report sections.
# they cover targets, closure conflicts, identifiability, uncertainty, controller rankings, generation mix and textbook disagreements.
# this file does not fit parameters.
# the Jacobian uses calibration targets at the preset values.
# the renderer copies the existing frozen-targets block byte for byte - `targets_io` owns that block.
# read constants and cache helpers first, then the four `compute_*` functions, chart helpers, section renderers and the command-line driver.
# see docs/research/07_design_critique.md, 'Reports, golden traces and reproducibility'.
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import math
import os
import re
import resource
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from mrm.validation import sensitivity
from mrm.validation import targets_io

__all__ = ["SCHEMA", "PARTS", "DOC_PATH", "DATA_ROOT", "CAL_PARAMETERS", "SENS_METRICS", "companion_g25",
           "data_path", "load_data", "save_data", "compute_predictions", "compute_identifiability",
           "compute_sensitivity", "compute_ranking", "render", "generate", "read_stamp", "build_parser", "main"]

# files
SCHEMA = "mrm.validation.report/1"                  # the ``schema`` of report_data.json
REPO = targets_io.REPO_DIR
DOC_PATH = targets_io.DOC_VALIDATION                # docs/VALIDATION.md, the report of the default preset
DATA_ROOT = REPO / "docs" / "validation"            # report_data.json and the charts, one folder per preset
PRESETS_DIR = REPO / "mrm" / "params" / "presets"
COMPARE_FILE = REPO / "experiments" / "examples" / "compare_locks.yaml"    # the section-5 compare experiment
# computed (cached) parts of the report data
PARTS = ("predictions", "identifiability", "sensitivity", "ranking")
G25_DEFAULT = "g25_test_vehicle"                    # G25 bench of g26, and of a preset without a G25 companion
# JSON report stamp - commit, date, dirty, g25_preset and preset. read_stamp reads it back.
STAMP_RE = re.compile(r"<!-- report-stamp (\{.*?\}) -->")

# protocol sources whose changes make a cached part stale.
# kernel parity tests keep the engine bit-identical, so engine sources are not listed.
CODE_DEPS: dict[str, tuple[str, ...]] = {
    "predictions": ("mrm/validation/replicas.py",),
    "identifiability": ("mrm/validation/replicas.py",),
    "sensitivity": ("mrm/validation/sensitivity.py",),
    "ranking": (),
}

# report contents
# scenario calibration replicas - exclude these from the plant-parameter Jacobian.
SCENARIO_REPLICAS = ("laser_line_statistics", "aggressor_trajectory")

# candidate parameters - (config path, calibration ids the parameter may explain).
# each leaf's ``range`` supplies its prior. See docs/research/07_design_critique.md, 'Calibration targets'.
CAL_PARAMETERS: tuple[tuple[str, str], ...] = (
    ("ring.optics.q_loaded", "CAL-1"),
    ("ring.optics.loss_budget.alpha_other_abs_db_per_cm", "CAL-2 (coupling branch)"),
    ("ring.optics.kappa_d_over_kappa", "CAL-2 (coupling branch)"),
    ("ring.optics.dln_alpha_dt_heater_per_k", "CAL-2 (ER mechanism)"),
    ("ring.heater.p_fsr_mw", "CAL-3a"),
    ("ring.heater.r_cold_ohm", "CAL-3b"),
    ("ring.heater.tcr_per_k", "CAL-4 (droop)"),
    ("ring.thermal.heater_self_ratio", "CAL-4 (droop)"),
    ("ring.junction.fill_fraction", "CAL-5"),
    ("ring.junction.eff_at_0v_pm_per_v", "CAL-5"),
    ("ring.optics.resonance_offset_nm", "CAL-7"),
    ("ring.monitor.gain_scale", "CAL-8"),
    ("optical_path.chip_input_loss_db", "CAL-8, CAL-14 (power at the ring)"),
    ("ring.heater.full_scale_v", "CAL-9 (DAC LSB under the voltage law)"),
    ("ring.junction.series_resistance_ohm", "CAL-10"),
    ("ring.junction.capacitance_at_0v_ff", "CAL-10"),
    ("driver.swing_pp_v", "CAL-14"),
    ("ring.monitor.responsivity_a_per_w", "CAL-11, CAL-8"),
)

# sensitivity metrics - (world_summary key, unit, transform applied before computing Sobol indices).
SENS_METRICS: tuple[tuple[str, str, str | None], ...] = (
    ("oma_penalty_db_worst", "dB", None),
    ("excursion_count", "", "log10(1+x)"),
    ("excursion_max_s", "s", None),
    ("wrong_line_rings", "rings", None),
    ("unlocked_rings", "rings", None),
    ("mean_reloma_db", "dB", None),
)
SENS_CONTROLLER = "pi_lock_c"           # the controller that runs in every section-4 world
SENS_TOP = 8                            # entries selected by the largest normalized tornado swing across metrics

# fixed caveats for target-table flags. Preserve these report strings.
# see docs/handoff/replicas.md, 'Findings the predictions reveal' and 'Decisions and deviations'.
# the target split also requires caveats about shared evidence - docs/handoff/validation_targets.md.
TARGET_FLAGS: dict[str, str] = {
    "CAL-3a": "R2 anchors P_FSR: close to the target by construction (replicas 4.6)",
    "CAL-3b": "R2 anchors V_FSR: close to the target by construction",
    "CAL-7": "R3 sets the cold offset: close to the target by construction",
    "CAL-13a": "scenario calibration residual (small by construction), not a prediction",
    "CAL-13b": "scenario calibration residual (small by construction), not a prediction",
    "CAL-14": "relOMA = one MRM's through port over the power into it; mrm.control.models reports OMA at "
              "the bus output: 0.35 dB lower (replicas 4.7)",
    "VAL-3b": "replica method changed after the comparison was visible (replicas handoff 5): treat a pass as weak",
    "VAL-4c": "not independent of VAL-4a/b (the map maximum is the CAL-14 level): not counted as a third "
              "independent pass; shares its printed spread with BND-4",
    "BND-4": "shares the printed '< 1 dB' spread with VAL-4c",
    "VAL-7a": "unverified: n_g 2.73 not re-read from de Cea 2019 (reported, not scored)",
    "VAL-7b": "definition conflict: energy fraction (primary) vs Robinson dn_eff/dn_Si (variant) (replicas 5)",
    "VAL-6": "deferred until the DAC law and full scale are known; its data are CAL-9",
    "VAL-8e": "cross-check (solver against imec's FE fit), never counted as validation",
    "VAL-8a": "a match of the interim thermal model with VAL-8 would be tautological; the replica reads the FEM "
              "extraction artefact",
    "BND-1": "a soft_fail (1 to 2 FWHM) is marginal, not a pass",
}

# closure rows - (anchor slug, quantity, conflicting published evidence). Preserve this transcribed evidence.
# see docs/research/07_design_critique.md, 'Closure and conflict table', and the thermal physics review.
# _closure_model_column computes the model column for each preset.
CLOSURE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("dlambda-dt", "dlambda/dT", "86 pm/K (printed aside) . 62.5 (Lightmatter bench, inferred) . 62.54 (imec, "
     "measured, O-band) . 66 to 69 at 25 degC and 74 to 77 at 105 degC (FEM plus expansion) . materials ceiling "
     "71 to 80 . 50 (textbook, refuted)"),
    ("fsr", "FSR", "10.3 nm (1.8 THz design) . < 11.44 nm (patent) . 8.7 to 8.8 nm (15 um diameter, wafer maps) . "
     "9.0 to 9.3 nm (R = 7.5 um at the 160 nm film's n_g) . 11.2 nm (P_FSR x 0.44) . >= 14.1 nm (unsupported "
     "read-off)"),
    ("thickness", "Waveguide thickness", "200 nm (SMTAI render) . 160 nm (45SPCLO photonics layer)"),
    ("q", "Q", "3500 (talk) . 3900 to 4200 (linewidth read-off)"),
    ("heater-efficiency", "Heater efficiency", "0.44 nm/mW (printed) . 0.405 (median at 10.3 nm) . 0.348 (swept "
     "device at 1.195 V)"),
    ("rth", "R_th", "4.7 K/mW under H86 . about 6.1 K/mW under H_solver . 2.9 to 11.4 K/mW across heater families"),
    ("reloma-level", "relOMA level", "-2.1 to -3.0 dB printed . -5.9 to -8.1 dB for a G25-like drive . requires "
     "170 to 230 pm at Q 3500"),
    ("reloma-curvature", "relOMA curvature", "k about 0.67 dB/degC^2 implies FWHM/(dlambda/dT) about 6 to 8 K only, "
     "in tension with Q 3500 and every dlambda/dT hypothesis unless the swing or Q differ (k is a secant, EXC-4)"),
    ("aggressor-rate", "Aggressor rate", "808 degC/s x 100 ms = 80.8 K (a mean rate) . slide-10 peak about 2200 "
     "degC/s (rectified sine: 2127) . a symmetric triangle would need 1600 degC/s"),
    ("g25-topology", "G25 TX topology", "all-pass in the paper schematic . input/thru/drop micrograph on later "
     "slides"),
    ("eo-bandwidth", "EO bandwidth", "35.4 against 34.1 GHz at -2 V . 66.8 GHz (peaked, single device, generation "
     "unclear)"),
    ("rom-dc-scale", "rom_dc_scale", "reported per dlambda/dT hypothesis"),
    ("crosstalk-dynamics", "Ring-to-ring crosstalk and far floor",
     "imec, natural convection and an EIC on top (04 X1, X2): 3 % nearest, 1.3-2.6 % floor, first-order 4.1-9.0 ms "
     "poles at a 100 um pitch . in-ring FEM ROM, TEC-held package (R_pkg 2 K/W): 1.0 % at 50 um with t63 0.12 ms "
     "(0.36 ms at 100 um; not first order, 90 % at 2.7 ms), 0.17-0.34 % at 150-350 um, 0.09 % at 1 mm . the G26 "
     "package is unpublished (physics_thermal review 2026-10-16)"),
)


# presets that accept fit records with no ``preset`` field, including mrm/calibration/latest_fit.json.
# the fit starts from g26 and writes g26_calibrated. Other presets require a record that names them.
FIT_RECORD_PRESETS = ("g26", "g26_calibrated")

# calibrated-preset flags - preserve the report text about protocol differences and unmet targets.
# see docs/handoff/calibration.md, 'Results' and 'Deviations (please review)'.
CALIBRATED_FLAGS: dict[str, str] = {
    "CAL-2": "the fit adds an instrument notch-floor nuisance that the replica protocol does not model (calibration "
             "handoff 4.2): fails here, passes with the fitted floor; the fitted nominal world passes critical "
             "coupling during the heater sweep (over-coupled at the operating line)",
    "CAL-4": "fitted on the swept device (its own P_FSR, calibration handoff 4.1); the replica runs the median device",
    "CAL-9a": "not reached by any hypothesis; nothing written (calibration handoff 4.6)",
    "CAL-9b": "not reached by any hypothesis; nothing written (calibration handoff 4.6)",
    "CAL-9c": "not reached by any hypothesis; nothing written (calibration handoff 4.6)",
    "CAL-14": "not reached inside the swing prior; nothing written (calibration handoff 4.5)",
}

# chart colours - blue for below nominal, pass, identified, S1 and profile curves.
# orange marks above nominal, failed identification and ST. Grey marks references and missing nominal values.
# ink marks text, nominal values, thresholds and error bars.
C_BLUE, C_ORANGE, C_GREY, C_INK = "#2a78d6", "#eb6834", "#a8a7a0", "#0b0b0b"


# small helpers
# return the current UTC time as an ISO 8601 string, to the second.
def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# return the commit, dirty flag and diff hash for the current repository.
# commit and diff_sha256 use 12 and 16 hex digits, respectively.
def _git_stamp() -> dict:
    from mrm.sim.manifest import git_state
    git = git_state(REPO)
    return {"commit": (git.get("commit") or "")[:12] or None, "dirty": git.get("dirty"),
            "diff_sha256": (git.get("diff_sha256") or "")[:16] or None}


# return peak resident memory so far in GB.
# ru_maxrss uses bytes on macOS and kB on Linux.
def _peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2), 3)


# copy `x` into JSON-compatible values.
# keep NaN and infinities as 'nan', 'inf' and '-inf'.
def _json_safe(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return _json_safe(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(x, Path):
        return str(x)
    return x


# restore 'nan', 'inf' and '-inf' strings as floats, recursively.
def _json_restore(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {k: _json_restore(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_restore(v) for v in x]
    if isinstance(x, str) and x in ("nan", "inf", "-inf"):
        return float(x)
    return x


# format a table value with `sig` significant digits.
# use '-' for None.
# use e-notation below 1e-3 and from 1e5 up.
# Otherwise use Python's `g` format, which also selects e-notation from 10**sig up.
# for example, 12345 with sig 4 gives '1.234e+04'.
def _fmt_num(x: float, sig: int = 4) -> str:
    if x is None:
        return "-"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if v == 0:
        return "0"
    if abs(v) >= 1e5 or abs(v) < 1e-3:
        return f"{v:.{sig - 1}e}"
    return f"{v:.{sig}g}"


# format a scalar, list or mapping as compact text for a table cell.
def _fmt_value(x: Any, max_items: int = 8, sig: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, Mapping):
        if "t_s" in x and "y" in x:
            return f"<trajectory, {len(x['y'])} samples>"
        items = [f"{k}: {_fmt_value(v, max_items, sig)}" for k, v in list(x.items())[:max_items]]
        return "{" + ", ".join(items) + (", ..." if len(x) > max_items else "") + "}"
    if isinstance(x, (list, tuple, np.ndarray)):
        flat = list(np.asarray(x, dtype=object).ravel()) if not (len(x) and isinstance(x[0], Mapping)) else list(x)
        s = ", ".join(_fmt_value(v, max_items, sig) for v in flat[:max_items])
        return "[" + s + (", ..." if len(flat) > max_items else "") + "]"
    return _fmt_num(x, sig)


# escape pipe characters and remove line breaks for a Markdown table cell.
def _md_cell(s: Any) -> str:
    return "" if s is None else str(s).replace("|", "\\|").replace("\n", " ")


# return prediction values as a flat numeric array.
# visit mapping keys in sorted order so Jacobian rows stay aligned.
# Represent None or invalid data as [nan].
def _flatten(x: Any) -> np.ndarray:
    if x is None:
        return np.array([np.nan])
    if isinstance(x, Mapping):
        parts = [_flatten(x[k]) for k in sorted(x)]
        return np.concatenate(parts) if parts else np.zeros(0)
    try:
        return np.asarray(x, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.array([np.nan])


# return one label per element of `_flatten(x)`.
# mapping labels use `key.subkey`.
# Array labels use zero-based indices.
def _flat_keys(x: Any, prefix: str = "") -> list[str]:
    if isinstance(x, Mapping):
        out = []
        for k in sorted(x):
            out += _flat_keys(x[k], f"{prefix}{k}.")
        return out
    n = 1 if x is None or np.ndim(x) == 0 else int(np.asarray(x, dtype=object).size)
    if n == 1:
        return [prefix.rstrip(".") or "value"]
    return [f"{prefix}{i}" for i in range(n)]


# load a preset with `mrm.params.load` and silence its warnings.
def _load_cfg(name: str, *, alts: Mapping | None = None, overrides: Mapping | None = None):
    from mrm import params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load(name, overrides=dict(overrides) if overrides else None,
                           alts=dict(alts) if alts else None)


# select the G25 bench preset for `preset`.
# use `g25` when supplied.
# Otherwise replace the first g26 in the name with g25_test_vehicle if that preset exists.
# Fall back to g25_test_vehicle.
def companion_g25(preset: str, g25: str | None = None) -> str:
    if g25:
        return g25
    if preset == "g26":
        return G25_DEFAULT
    candidate = preset.replace("g26", G25_DEFAULT, 1)
    if candidate != preset and (PRESETS_DIR / f"{candidate}.yaml").exists():
        return candidate
    return G25_DEFAULT


# return the named corner presets used for the prediction band in section 1.
def _named_corners() -> tuple[str, ...]:
    from mrm.sim.experiment import NAMED_CORNERS
    return tuple(NAMED_CORNERS)


# read the alternatives selected by a corner preset.
# a corner may select alternatives but may not set leaf values.
def _corner_alts(name: str) -> dict:
    import yaml
    doc = yaml.safe_load((PRESETS_DIR / f"{name}.yaml").read_text()) or {}
    extra = set(doc) - {"extends", "meta", "alts"}
    if extra:
        raise ValueError(f"corner preset {name} sets {sorted(extra)}: only alts are supported in the report band")
    return {str(k): str(v) for k, v in (doc.get("alts") or {}).items()}


# return the sha256 hash for each calibration, validation and bounds target file.
def _target_files_sha() -> dict:
    return {r: targets_io.file_sha256(r) for r in targets_io.TARGET_ROLES}


# return the source hashes for one cached part.
# use 16 hex digits per file in `CODE_DEPS`.
def _code_sha(part: str) -> dict:
    return {f: hashlib.sha256((REPO / f).read_bytes()).hexdigest()[:16] for f in CODE_DEPS.get(part, ())}


# build metadata for a computed part.
# record time, commit, presets, config and target hashes, and `extra` fields.
# Wall time since `t0` is in s.
# Peak resident memory is in GB.
def _part_meta(presets: Mapping[str, str], t0: float, **extra) -> dict:
    config_hash = {}
    for k, name in presets.items():
        try:
            config_hash[name] = _load_cfg(name).hash()[:16]
        except Exception as e:                                          # noqa: BLE001
            config_hash[name] = f"error: {e.__class__.__name__}"
    metadata = {"computed_utc": _utc_now(), "git": _git_stamp(), "presets": dict(presets), "config_hash": config_hash,
                "targets_sha256": _target_files_sha(), "wall_s": round(time.perf_counter() - t0, 1),
                "peak_rss_gb": _peak_rss_gb()}
    metadata.update(extra)
    return metadata


# write a compute-progress message to standard error.
def _log_to_stderr(s: str) -> None:
    print(s, file=sys.stderr, flush=True)


# the data file (report_data.json)
# return docs/validation/<preset>/, the report cache and chart folder.
def data_dir(preset: str) -> Path:
    return DATA_ROOT / preset


# return the path to docs/validation/<preset>/report_data.json.
def data_path(preset: str) -> Path:
    return data_dir(preset) / "report_data.json"


# load cached report data for `preset`.
# return an empty cache header when the file is absent.
def load_data(preset: str) -> dict:
    path = data_path(preset)
    if not path.exists():
        return {"schema": SCHEMA, "preset": preset}
    data = _json_restore(json.loads(path.read_text()))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {data.get('schema')!r}, expected {SCHEMA}")
    return data


# write report data with NaN and infinities encoded as strings.
# return the file path.
def save_data(preset: str, data: Mapping) -> Path:
    path = data_path(preset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(data)), separators=(",", ":"), allow_nan=False) + "\n")
    return path


# computed part - predictions (section 1)
# build the stored nominal prediction for target `t` from its ReplicaResult `res`.
def _prediction_record(t, res) -> dict:
    from mrm.validation.replicas import _nonfinite
    fallbacks = (res.conditions or {}).get("artefact_fallbacks") if isinstance(res.conditions, Mapping) else None
    return {"id": t.id, "replica_status": res.status, "predicted": res.value,
            "predicted_nonfinite": bool(res.status == "ok" and _nonfinite(res.value)),
            "variants": res.variants or {}, "notes": [str(n) for n in (res.notes or [])],
            "artefact_fallbacks": fallbacks}


# return the preset key for a target - "G25" for a G25 bench measurement, otherwise "G26".
def _plant_key(t) -> str:
    from mrm.validation.replicas import preset_for
    return "G25" if preset_for(t) == G25_DEFAULT else "G26"


# compute nominal and corner predictions for calibration, validation and bounds targets.
# `presets` maps G26 and G25 to preset names.
# apply each corner's alternatives to those same presets.
# replica failures are stored as blocked or error results with a reason.
def compute_predictions(presets: Mapping[str, str], *, corners: Sequence[str] | None = None,
                        log: Callable[[str], None] = _log_to_stderr) -> dict:
    from mrm.validation import replicas
    t0 = time.perf_counter()
    corners = _named_corners() if corners is None else tuple(corners)
    targets = [t for role in targets_io.TARGET_ROLES for t in targets_io.load_targets(role)]
    records = []
    for t in targets:
        t1 = time.perf_counter()
        res = replicas.run_target(t, presets[_plant_key(t)])
        rec = _prediction_record(t, res)
        rec["runtime_s"] = round(time.perf_counter() - t1, 2)
        records.append(rec)
        log(f"  {t.id:8s} {res.status:8s} {rec['runtime_s']:6.1f} s")
    corner_preds: dict[str, dict] = {}            # {corner: {target id: {status, predicted, note}}}
    alts_of: dict[str, dict] = {}
    for corner in corners:
        alts = _corner_alts(corner)
        alts_of[corner] = alts
        t1 = time.perf_counter()
        cfgs: dict[str, Any] = {}                 # {"G26" | "G25": Config with the corner's alts, or the error}
        for k, name in presets.items():
            try:
                cfgs[k] = replicas.load_config(name, alts=alts)
            except Exception as e:                                      # noqa: BLE001
                cfgs[k] = e
        preds = {}
        for t in targets:
            cfg = cfgs[_plant_key(t)]
            if isinstance(cfg, Exception):
                preds[t.id] = {"status": "error", "predicted": None, "note": f"{cfg.__class__.__name__}: {cfg}"}
                continue
            res = replicas.run_target(t, cfg)
            preds[t.id] = {"status": res.status, "predicted": res.value,
                           "note": (res.notes or [""])[0] if res.status != "ok" else ""}
        corner_preds[corner] = preds
        log(f"  corner {corner}: {time.perf_counter() - t1:.1f} s")
    return {"meta": _part_meta(presets, t0, corners=list(corners), code_sha256=_code_sha("predictions")),
            "corner_alts": alts_of, "records": records,
            "corners": corner_preds}


# computed part - identifiability (section 3)
# return one normalization scale per flattened prediction element, in the target units.
# prefer a positive target sigma, supplied per point, per key or for the whole target.
# Otherwise use a quarter of an interval tolerance, half an absolute tolerance, or half a relative tolerance times the largest absolute target value.
# for curves, use y values.
# return NaN where no scale is available so the identifiability calculation can drop that row.
def _residual_scales(t, pred: Any) -> np.ndarray:
    n = _flatten(pred).size
    sigma = t.sigma
    if isinstance(sigma, Mapping) and isinstance(pred, Mapping):
        out = []
        for k in sorted(pred):
            n_elem = _flatten(pred[k]).size
            s = sigma.get(k)
            out += [float(s) if isinstance(s, (int, float)) else np.nan] * n_elem
        scale = np.asarray(out, float)
    elif isinstance(sigma, (list, tuple)) and len(sigma) == n:
        scale = np.asarray(sigma, float)
    elif isinstance(sigma, (int, float)) and not isinstance(sigma, bool):
        scale = np.full(n, float(sigma))
    else:
        scale = np.full(n, np.nan)
    if np.any(~np.isfinite(scale) | (scale <= 0)):
        tol = t.tolerance
        k = tol.get("kind")
        v = _flatten(t.value["y"] if t.kind == "curve" else t.value)
        if k == "interval":
            fallback = (float(tol["hi"]) - float(tol["lo"])) / 4.0
        elif k == "abs":
            fallback = float(tol["value"]) / 2.0
        elif k == "rel":
            fallback = float(tol["value"]) * float(np.nanmax(np.abs(v))) / 2.0
        else:
            fallback = np.nan
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, fallback)
    return scale


# select scored calibration targets for the plant Jacobian.
# Exclude scenario and never-fitted targets.
def _jacobian_targets() -> list:
    return [t for t in targets_io.load_calibration_targets() if t.scored and t.replica not in SCENARIO_REPLICAS
            and t.raw.get("fit", True) is not False]


# compute the calibration Jacobian at the preset values and its SVD.
# `params` contains config paths and the target ids each parameter explains.
# Move each parameter x_j to x_j +/- h, where h is `rel_step` times its prior range.
# clip each step to the range, which makes the difference one-sided at a range end.
# Express derivatives in residual sigmas per prior half-width -
# J[i, j] = (y_i(x_j+) - y_i(x_j-)) / (x_j+ - x_j-) * (prior half-width of j) / scale_i
# `sensitivity.identifiability` applies the SVD and half-prior rule.
# Move a leaf on another plant only when its value matches G26.
# the `kept` record names plants that retain their own value.
def compute_identifiability(presets: Mapping[str, str], *, params: Sequence[tuple[str, str]] = CAL_PARAMETERS,
                            rel_step: float = 0.05, log: Callable[[str], None] = _log_to_stderr) -> dict:
    from mrm.validation import replicas
    t0 = time.perf_counter()
    targets = _jacobian_targets()
    base_cfg = {k: _load_cfg(n) for k, n in presets.items()}

    # return replica outputs in target order, with None for failed replicas.
    # `cfgs` maps G26 and G25 to configs.
    def predict(cfgs) -> list:
        out = []
        for t in targets:
            cfg = cfgs[_plant_key(t)]
            res = replicas.run_target(t, cfg) if not isinstance(cfg, Exception) else None
            out.append(res.value if res is not None and res.status == "ok" else None)
        return out

    y0 = predict(base_cfg)                   # base predictions, one per target

    # recompute targets on the perturbed plants.
    # keep y0 for targets on other plants.
    def predict_on(cfgs, perturbed) -> list:
        out = []
        for t, y in zip(targets, y0):
            k = _plant_key(t)
            if k not in perturbed:
                out.append(y)
                continue
            res = replicas.run_target(t, cfgs[k]) if not isinstance(cfgs[k], Exception) else None
            out.append(res.value if res is not None and res.status == "ok" else None)
        return out

    # align each flattened base prediction with its normalization scales and row labels.
    base_rows, scales, labels = [], [], []
    for t, p in zip(targets, y0):
        f = _flatten(p)
        base_rows.append(f)
        scales.append(_residual_scales(t, p))
        labels += [f"{t.id}[{k}]" if len(f) > 1 else t.id for k in _flat_keys(p)][:f.size]
    y0_flat = np.concatenate(base_rows)
    scale = np.concatenate(scales)
    jac_cols, param_info = [], []
    for path, why in params:
        g26 = base_cfg["G26"]
        leaf = g26.leaves.get(path)
        info = {"path": path, "identifies": why}
        if leaf is None or leaf.range is None or leaf.is_free:
            info["skipped"] = "not a leaf with a prior range in this preset"
            param_info.append(info)
            jac_cols.append(np.full(y0_flat.size, np.nan))
            continue
        lo, hi = float(leaf.range[0]), float(leaf.range[1])
        h = rel_step * (hi - lo)
        info.update(prior=[lo, hi], provenance=leaf.provenance, targets=leaf.targets, residual=leaf.residual,
                    identifiable=leaf.identifiable)
        sides = {}                           # {-1 | +1 - ({plant key: moved value}, predictions)}
        try:
            g26_value = np.asarray(leaf.value, float).ravel()
            if g26_value.size != 1:
                raise ValueError(f"{path} is not a scalar leaf in {presets['G26']}")
            # move the same physical parameter only where its value matches G26.
            # a plant-specific value, such as the G25 optical path, stays fixed and gives zero target derivatives.
            # use the G26 prior range and step for every moved plant.
            moved, kept = ["G26"], {}
            for k, name in presets.items():
                if k == "G26":
                    continue
                other_leaf = base_cfg[k].leaves.get(path)
                other_value = (None if other_leaf is None or other_leaf.is_free
                               else np.asarray(other_leaf.value, float).ravel())
                if other_value is not None and other_value.size == 1 and float(other_value[0]) == float(g26_value[0]):
                    moved.append(k)
                else:
                    kept[k] = (f"{name} has {_fmt_value(None if other_leaf is None else other_leaf.value)} "
                               f"(G26 {_fmt_num(float(g26_value[0]))}): not moved")
            info["moved_on"] = moved
            if kept:
                info["kept"] = kept
            for sign in (-1, 1):
                v = float(np.clip(g26_value[0] + sign * h, lo, hi))
                vals = {k: v for k in moved}
                cfgs = {k: _load_cfg(presets[k], overrides={path: v}) for k in moved}
                sides[sign] = (vals, predict_on(cfgs, moved))
        except Exception as e:                                          # noqa: BLE001
            info["skipped"] = f"{e.__class__.__name__}: {e}"
            param_info.append(info)
            jac_cols.append(np.full(y0_flat.size, np.nan))
            continue
        dx = sides[1][0]["G26"] - sides[-1][0]["G26"]            # actual step (smaller at a range end)
        info["value"] = float(np.asarray(g26.leaves[path].value, float).ravel()[0])
        info["step"] = [sides[-1][0]["G26"], sides[1][0]["G26"]]
        # mark size-changing predictions as NaN because their elements no longer align with the base rows.
        y_plus = np.concatenate([_flatten(p) if _flatten(p).size == r.size else np.full(r.size, np.nan)
                                 for p, r in zip(sides[1][1], base_rows)])
        y_minus = np.concatenate([_flatten(p) if _flatten(p).size == r.size else np.full(r.size, np.nan)
                                  for p, r in zip(sides[-1][1], base_rows)])
        jac_cols.append((y_plus - y_minus) / dx * (0.5 * (hi - lo)) / scale)
        param_info.append(info)
        log(f"  {path}: {time.perf_counter() - t0:.0f} s")
    J = np.stack(jac_cols, axis=1)
    names = [p for p, _ in params]
    used = [i for i, inf in enumerate(param_info) if "skipped" not in inf]
    ident = sensitivity.identifiability(J[:, used], [names[i] for i in used]) if used else None
    target_values = []
    for t in targets:
        target_values.append({"id": t.id, "replica": t.replica, "value": t.value, "sigma": t.sigma})
    base_eval = []
    for t, p in zip(targets, y0):
        ev = targets_io.evaluate(t, p) if p is not None else None
        base_eval.append({"id": t.id, "level": ev.level if ev else "blocked", "z": ev.z if ev else None})
    return {"meta": _part_meta(presets, t0, rel_step=rel_step, n_evaluations=1 + 2 * len(used),
                               code_sha256=_code_sha("identifiability")),
            "targets": target_values, "residual_labels": labels, "scales": scale, "base_prediction": y0_flat,
            "jacobian": J, "params": param_info, "svd": ident, "base_evaluation": base_eval}


# computed part - sensitivity (section 4)
# build experiment YAML for `B` sensitivity worlds on `preset`.
# run pi_lock_c with scenario A1, a warm start, self reference and metrics only.
# Score one aggressor period after 5 ms of start-up.
# this is the compare_locks protocol for one controller.
# `first_world` is the first row in the full design.
# offset every seed base by that row so design row i always uses noise seed 300 + i, for any chunk size.
def _sensitivity_experiment_yaml(preset: str, B: int, first_world: int = 0) -> str:
    s = int(first_world)
    return "\n".join([
        f"extends: {preset}",
        "meta: {name: report_sensitivity, kind: experiment}",
        f"controller: {{name: {SENS_CONTROLLER}, params: {{}}}}",
        "scenario: {name: a1, params: {phase_s: 0.06535}}",
        "metrics: {lock_threshold_pm: 5.0, hysteresis_pm: 1.0}",
        "experiment:",
        "  kind: mc",
        "  duration_s: 0.1357",
        f"  n_mc: {int(B)}",
        f"  seeds: {{device: {100 + s}, uncertainty: {200 + s}, noise: {300 + s}, scenario: {400 + s}}}",
        "  uncertainty: lhs",
        '  variation: "off"',
        "  start: warm",
        "  score_from_s: 0.005",
        "  self_reference: {duration_s: 0.02, window_s: 0.005}",
        ""])


# return world-summary arrays for the sensitivity design `U` with shape (B, D).
# B is the number of worlds.
# D is the number of uncertainty entries.
# each returned metric has shape (B,).
# run at most `chunk` worlds per Plant.
# check each injected design against realize's draws before running it.
# each chunk is a separate experiment.
# offset its seed bases by its first row index so every design row keeps the same seeds for any chunk size.
def evaluate_worlds(preset: str, U: np.ndarray, *, chunk: int = 256, n_threads: int = 4,
                    log: Callable[[str], None] = _log_to_stderr) -> dict[str, np.ndarray]:
    from mrm.sim import experiment
    space = sensitivity.UncertaintySpace(_load_cfg(preset))
    out: dict[str, list] = {}
    with tempfile.TemporaryDirectory(prefix="mrm_report_") as td:
        for i0 in range(0, U.shape[0], chunk):
            U_chunk = U[i0:i0 + chunk]
            check = sensitivity.check_injection(space, U_chunk)
            if not check["ok"]:
                raise RuntimeError(f"injected design does not reach realize: {check['mismatches'][:3]}")
            f = Path(td) / f"sens_{i0}.yaml"
            f.write_text(_sensitivity_experiment_yaml(preset, U_chunk.shape[0], first_world=i0))
            t1 = time.perf_counter()
            with sensitivity.injected_lhs(U_chunk):
                res = experiment.run(str(f), write=False, log=lambda s: None,
                                     options={"n_threads": int(n_threads),
                                              "max_worlds_per_plant": max(512, U_chunk.shape[0])})
            for k, v in res.summary["world_summary"].items():
                out.setdefault(k, []).append(np.asarray(v, float))
            log(f"  worlds {i0}..{i0 + U_chunk.shape[0] - 1}: {time.perf_counter() - t1:.0f} s")
    return {k: np.concatenate(v) for k, v in out.items()}


# apply the metric transform from `SENS_METRICS` - None or "log10(1+x)".
def _apply_transform(x: np.ndarray, tr: str | None) -> np.ndarray:
    if tr == "log10(1+x)":
        return np.log10(1.0 + np.asarray(x, float))
    return np.asarray(x, float)


# compute tornado responses, then Sobol indices for the most sensitive uncertainty entries.
# move one free entry at a time to its `EntryAxis.tornado_points`.
# keep the other entries nominal.
# `sensitivity.screen_top` selects `top` entries by their largest normalized metric swing.
# Saltelli first-order and Jansen total indices then use those entries' prior coordinates in u-space.
# keep the other entries nominal.
# `n_base` base rows give (top + 2) * n_base worlds.
# the binned first-order estimate uses the 2 * n_base independent A and B rows.
# append `n_rep` nominal replicates to the tornado design to measure metric noise variance.
# Row indices set noise seeds in `evaluate_worlds`, so a Saltelli row and its AB_i partner have different noise.
# retain the raw world values so the indices can be computed again.
def compute_sensitivity(preset: str, *, n_base: int = 128, top: int = SENS_TOP, seed: int = 20261010,
                        chunk: int = 256, n_threads: int = 4, n_boot: int = 500, n_rep: int = 8,
                        log: Callable[[str], None] = _log_to_stderr) -> dict:
    t0 = time.perf_counter()
    space = sensitivity.UncertaintySpace(_load_cfg(preset))
    u0 = space.nominal_u()
    labels, rows = [("nominal", "nominal")], [u0]          # (entry, tornado point label) per world
    for i, n in enumerate(space.names):
        if space.axes[n].pinned is not None:
            continue
        for point_label, u in space.axes[n].tornado_points():
            row = u0.copy()
            row[i] = u
            rows.append(row)
            labels.append((n, point_label))
    n_tornado = len(rows)
    rows += [u0] * int(n_rep)                          # nominal replicates vary noise through the world index
    U_tornado = np.vstack(rows)
    log(f"tornado: {n_tornado} worlds + {n_rep} nominal replicates")
    tornado_worlds = evaluate_worlds(preset, U_tornado, chunk=chunk, n_threads=n_threads, log=log)
    metrics = [m for m, _, _ in SENS_METRICS]
    base = {m: float(tornado_worlds[m][0]) for m in metrics}
    tornado_pts = []
    for j, (entry, point_label) in enumerate(labels[1:], start=1):
        if space.axes[entry].kind == "hypothesis":
            desc = point_label
        else:
            desc = f"{point_label} ({space.describe(entry, U_tornado[j, space.names.index(entry)])})"
        tornado_pts.append((entry, desc, {m: float(tornado_worlds[m][j]) for m in metrics}))
    tornado = sensitivity.tornado_table(base, tornado_pts, metrics)
    top_entries = sensitivity.screen_top(tornado, top)
    log(f"Sobol over {top_entries}")
    A, B = sensitivity.sobol_design(len(top_entries), n_base, seed=seed)
    X_stack = sensitivity.saltelli_stack(A, B)
    U_sobol = np.tile(u0[None, :], (X_stack.shape[0], 1))
    for c, n in enumerate(top_entries):
        U_sobol[:, space.names.index(n)] = X_stack[:, c]
    t2 = time.perf_counter()
    sobol_worlds = evaluate_worlds(preset, U_sobol, chunk=chunk, n_threads=n_threads, log=log)
    sobol_by_metric = {}
    d = len(top_entries)
    X_ab = np.vstack([A, B])                           # the independent rows, for the binned estimate
    edges = [space.axes[n].cdf if space.axes[n].kind == "hypothesis" else None for n in top_entries]
    replicate_rows = [0] + list(range(n_tornado, n_tornado + int(n_rep)))     # nominal world and its replicates
    for m, unit, tr in SENS_METRICS:
        y = _apply_transform(sobol_worlds[m], tr)
        fA, fB, fAB = sensitivity.split_stack(y, n_base, d)
        good = np.isfinite(fA) & np.isfinite(fB) & np.all(np.isfinite(fAB), axis=0)
        entry = {"transform": tr, "n_rows_dropped_nonfinite": int((~good).sum())}
        if good.sum() >= 8:
            res = sensitivity.sobol_indices(fA[good], fB[good], fAB[:, good], names=top_entries, n_boot=n_boot,
                                            seed=seed)
            entry.update(res.to_dict())
            y_ab = np.r_[fA, fB]
            finite = np.isfinite(y_ab)
            binned = sensitivity.given_data_first_order(X_ab[finite], y_ab[finite], edges=edges, n_boot=n_boot,
                                                        seed=seed)
            entry["s1_binned"] = [float(x) for x in binned["s1"]]
            entry["s1_binned_ci"] = [[float(a), float(b)] for a, b in binned["s1_ci"]]
            entry["s1_binned_n_bins"] = list(binned["n_bins"])
            replicate_y = _apply_transform(tornado_worlds[m][replicate_rows], tr)
            noise_var = (float(np.var(replicate_y, ddof=1)) if n_rep > 0 and np.all(np.isfinite(replicate_y))
                         else float("nan"))
            entry["noise_variance"] = noise_var
            entry["noise_share"] = noise_var / float(entry["variance"]) if entry["variance"] > 0 else float("nan")
        else:
            entry["notes"] = ["fewer than 8 finite base rows: not computed"]
        sobol_by_metric[m] = entry
    supply_limited = sobol_worlds.get("warm_start_supply_limited")
    return {"meta": _part_meta({"G26": preset}, t0, controller=SENS_CONTROLLER, scenario="a1 (one 130.7 ms period, "
                               "5 ms start-up)", n_tornado_worlds=int(n_tornado),
                               n_sobol_worlds=int(U_sobol.shape[0]), n_base=int(n_base), seed=int(seed),
                               chunk=int(chunk), n_threads=int(n_threads),
                               sobol_wall_s=round(time.perf_counter() - t2, 1), n_boot=int(n_boot),
                               n_noise_replicates=int(n_rep), code_sha256=_code_sha("sensitivity")),
            "metrics": [{"key": m, "unit": u, "transform": tr} for m, u, tr in SENS_METRICS],
            "entries": {n: {"kind": space.axes[n].kind, "pinned": space.axes[n].pinned,
                            "nominal": space.axes[n].nominal,
                            "support": list(space.axes[n].support) if space.axes[n].kind == "hypothesis" else None,
                            "low": space.axes[n].fields.get("low"), "high": space.axes[n].fields.get("high"),
                            "dist": space.axes[n].fields.get("dist", "uniform" if space.axes[n].kind == "interval"
                                                             else None)}
                        for n in space.names},
            "base": base, "base_world": {k: float(v[0]) for k, v in tornado_worlds.items()},
            "tornado_points": [{"entry": e, "label": lab, "values": v} for e, lab, v in tornado_pts],
            "noise_replicates": {k: [float(x) for x in np.asarray(v, float)[[0] + list(range(n_tornado,
                                                                                           n_tornado + n_rep))]]
                                 for k, v in tornado_worlds.items()},
            "sobol_worlds": {k: [float(x) for x in np.asarray(v, float)] for k, v in sobol_worlds.items()
                             if k in {m for m, _, _ in SENS_METRICS} | {"warm_start_supply_limited"}},
            "tornado": tornado, "top": top_entries, "sobol": sobol_by_metric,
            "sobol_supply_limited_frac": float(np.mean(supply_limited)) if supply_limited is not None else None,
            "tornado_supply_limited": {labels[j][0] + ":" + labels[j][1]:
                                       float(tornado_worlds["warm_start_supply_limited"][j])
                                       for j in range(len(labels))
                                       if "warm_start_supply_limited" in tornado_worlds
                                       and tornado_worlds["warm_start_supply_limited"][j] > 0}}


# computed part - ranking (section 5)
# compute controller rank stability across worlds with the compare experiment.
# run experiments/examples/compare_locks.yaml on `preset` with `n_mc` Latin-hypercube worlds.
# if `compare_run` is supplied, read that existing run's summary instead.
def compute_ranking(preset: str, *, n_mc: int = 256, compare_run: str | Path | None = None, n_threads: int = 4,
                    log: Callable[[str], None] = _log_to_stderr) -> dict:
    t0 = time.perf_counter()
    if compare_run is not None:
        run_dir = Path(compare_run)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if metrics.get("kind") != "compare":
            raise ValueError(f"{run_dir} is not a compare run")
        files = [str(x) for x in (manifest.get("files") or [])]
        summary = metrics["summary"]
        run_git = manifest.get("git") or {}
        source = {"run_dir": str(run_dir), "run_id": manifest.get("run_id"),
                  "experiment_hash": manifest.get("experiment_hash"),
                  "run_git": {"commit": run_git.get("commit"), "dirty": run_git.get("dirty")},
                  "run_wall_s": manifest.get("wall_s"), "files": files}
    else:
        import yaml
        from mrm.sim import experiment
        doc = yaml.safe_load(COMPARE_FILE.read_text())
        doc["extends"] = preset
        doc.setdefault("meta", {})["name"] = f"report_compare_{preset}"
        doc["experiment"]["n_mc"] = int(n_mc)
        with tempfile.TemporaryDirectory(prefix="mrm_report_") as td:
            f = Path(td) / "compare.yaml"
            f.write_text(yaml.safe_dump(doc, sort_keys=False))
            res = experiment.run(str(f), write=False, log=log, options={"n_threads": int(n_threads)})
        summary = res.summary
        source = {"experiment": "experiments/examples/compare_locks.yaml re-based on the preset", "n_mc": int(n_mc),
                  "experiment_hash": res.experiment_hash[:16]}
    controllers = {}
    for label, ctrl in summary.get("controllers", {}).items():
        controllers[label] = {"nominal": ctrl.get("nominal"), "worst_corner_overall": ctrl.get("worst_corner_overall"),
                              "worst_corner": {k: v for k, v in (ctrl.get("worst_corner") or {}).items()},
                              "quantiles": {k: v for k, v in (ctrl.get("quantiles") or {}).items()}}
    return {"meta": _part_meta({"G26": preset}, t0, source=source), "rank_metrics": summary.get("rank_metrics"),
            "rank_stability": summary.get("rank_stability"), "controllers": controllers}


# charts
# load matplotlib.pyplot with the Agg backend and report style.
# keep SVG text as text.
# Fix svg.hashsalt so element ids stay stable.
# `_save_svg` removes date metadata.
def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "mrm-validation-report", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    return plt


# save and close an SVG chart.
# Omit date and creator metadata so repeated renders are byte-identical.
def _save_svg(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


# plot scored target z values on a log axis for section 1.
# `rows` holds (target id, z, result).
def _chart_z(rows: list[tuple[str, float, str]], path: Path) -> None:
    plt = _pyplot()
    rows = [r for r in rows if r[1] is not None and np.isfinite(r[1])]
    fig, ax = plt.subplots(figsize=(6.0, 0.22 * len(rows) + 0.9))
    y = np.arange(len(rows))
    z = np.array([max(r[1], 1e-3) for r in rows])
    cols = [C_BLUE if r[2] == "pass" else C_ORANGE for r in rows]
    ax.barh(y, z, color=cols, height=0.6)
    for yi, r, zi in zip(y, rows, z):
        ax.text(zi * 1.08, yi, r[2], va="center", fontsize=7, color=C_INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.axvline(1.0, color=C_GREY, lw=0.8)
    ax.set_xlabel("z = max |residual| / sigma (log); bar label = result against the frozen tolerance")
    _save_svg(fig, path)


# plot normalized Jacobian singular values against the half-prior threshold for section 3.
def _chart_svd(s: Sequence[float], threshold: float, path: Path) -> None:
    plt = _pyplot()
    s = np.asarray(s, float)
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    k = np.arange(1, s.size + 1)
    ax.bar(k, np.maximum(s, 1e-6), color=[C_BLUE if v >= threshold else C_ORANGE for v in s], width=0.6)
    ax.axhline(threshold, color=C_INK, lw=0.8, ls="--")
    ax.text(s.size + 0.4, threshold, f" half-prior rule\n s >= {threshold:.2f}", va="center", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(k)
    ax.set_xlabel("SVD direction")
    ax.set_ylabel("singular value (sigmas per prior half-width)")
    _save_svg(fig, path)


# plot the first `n` tornado rows for one metric.
# use blue below `base` and orange above.
# draw infinite ends 15 % of the finite span beyond the finite values.
def _chart_tornado(rows: list[dict], base: float, title: str, path: Path, n: int = 12) -> None:
    plt = _pyplot()
    rows = rows[:n]
    fig, ax = plt.subplots(figsize=(6.0, 0.3 * len(rows) + 0.9))
    y = np.arange(len(rows))
    finite_vals = ([v for r in rows for v in (r["lo"], r["hi"]) if np.isfinite(v)]
                   + ([base] if np.isfinite(base) else []))
    span = (max(finite_vals) - min(finite_vals)) if finite_vals else 1.0
    cap_hi = (max(finite_vals) + 0.15 * span) if finite_vals else 1.0
    cap_lo = (min(finite_vals) - 0.15 * span) if finite_vals else 0.0
    for yi, r in zip(y, rows):
        lo = r["lo"] if np.isfinite(r["lo"]) else cap_lo
        hi = r["hi"] if np.isfinite(r["hi"]) else cap_hi
        if np.isfinite(base):
            ax.barh(yi, base - lo, left=lo, color=C_BLUE, height=0.6)
            ax.barh(yi, hi - base, left=base, color=C_ORANGE, height=0.6)
        else:
            ax.barh(yi, hi - lo, left=lo, color=C_GREY, height=0.6)
        lab = (f"  {r['lo_label']} to {r['hi_label']}" + (" (inf)" if not np.isfinite(r["hi"]) else "")
               if r["swing"] != 0 else "  no effect")
        ax.text(max(hi, base) if np.isfinite(base) else hi, yi, lab, va="center", fontsize=6.5, color=C_INK)
    if np.isfinite(base):
        ax.axvline(base, color=C_INK, lw=0.8)
    ax.set_yticks(y, [r["entry"] for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(title)
    ax.margins(x=0.45)
    _save_svg(fig, path)


# plot total ST and first-order S1 indices with one panel per metric.
# show the ST confidence interval.
# use binned S1 when available.
def _chart_sobol(sobol: Mapping[str, Mapping], names: Sequence[str], path: Path) -> None:
    plt = _pyplot()
    metrics = [m for m in sobol if "st" in sobol[m]]
    if not metrics:
        return
    ncol = min(3, len(metrics))
    nrow = int(math.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol + 1.6, (0.3 * len(names) + 1.0) * nrow), sharey=True,
                             squeeze=False)
    for ax in axes.ravel()[len(metrics):]:
        ax.set_visible(False)
    axes = axes.ravel()
    y = np.arange(len(names))
    for ax, m in zip(axes, metrics):
        entry = sobol[m]
        st = np.array([np.nan if v is None else v for v in entry["st"]], float)
        s1 = np.array([np.nan if v is None else v for v in entry.get("s1_binned", entry["s1"])], float)
        ax.barh(y + 0.18, np.clip(st, 0, None), height=0.34, color=C_ORANGE, label="total ST")
        ax.barh(y - 0.18, np.clip(s1, 0, None), height=0.34, color=C_BLUE,
                label="first order S1" + (" (binned)" if "s1_binned" in entry else ""))
        ci = np.array([[np.nan if v is None else v for v in r] for r in entry["st_ci"]], float)
        ax.errorbar(np.clip(st, 0, None), y + 0.18, xerr=np.abs(np.vstack([st - ci[:, 0], ci[:, 1] - st])),
                    fmt="none", ecolor=C_INK, lw=0.6)
        ax.set_title(m + (f"\n({entry['transform']})" if entry.get("transform") else ""), fontsize=7)
        ax.set_xlim(0, 1.05)
    for r in range(nrow):
        axes[r * ncol].set_yticks(y, list(names), fontsize=7)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=6.5, loc="lower right", frameon=False)
    fig.tight_layout()
    _save_svg(fig, path)


# render - header and staleness
# return a path relative to docs/ with forward slashes for links from docs/VALIDATION.md.
def _docs_relpath(path: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), (REPO / "docs").resolve())).as_posix()


# copy the frozen-targets block from report text, including both markers.
def _frozen_block(doc_text: str) -> str:
    a = doc_text.find(targets_io.FROZEN_BEGIN)
    b = doc_text.find(targets_io.FROZEN_END)
    if a < 0 or b < 0 or b < a:
        raise ValueError("docs/VALIDATION.md has no frozen-targets block: restore it from git (the report never "
                         "writes it; mrm/validation/targets/README.md)")
    return doc_text[a:b + len(targets_io.FROZEN_END)]


# read the report-stamp record - preset, g25_preset, date, commit and dirty.
def read_stamp(text: str) -> dict | None:
    m = STAMP_RE.search(text)
    return json.loads(m.group(1)) if m else None


# format one header-table row with a cached part's computation time, commit, cost and presets.
def _part_table_row(name: str, part: Mapping | None) -> str:
    if not part:
        return f"| {name} | not computed | | | | |"
    meta = part.get("meta", {})
    git = meta.get("git") or {}
    return (f"| {name} | {meta.get('computed_utc', '')} | `{git.get('commit')}`"
            f"{' (dirty)' if git.get('dirty') else ''} | "
            f"{_fmt_num(meta.get('wall_s'))} | {_fmt_num(meta.get('peak_rss_gb'))} | "
            f"{', '.join(f'{k} `{v}`' for k, v in (meta.get('config_hash') or {}).items())} |")


# list cache mismatches with current presets, target files and protocol source files.
# compare config hashes and the source hashes in `CODE_DEPS`.
# return an empty list when all parts match.
def _staleness(data: Mapping, presets: Mapping[str, str]) -> list[str]:
    out = []
    target_sha_now = _target_files_sha()
    config_hash_now = {}
    for name in set(presets.values()):
        try:
            config_hash_now[name] = _load_cfg(name).hash()[:16]
        except Exception as e:                                          # noqa: BLE001
            config_hash_now[name] = f"error: {e.__class__.__name__}"
    for p in PARTS:
        part = data.get(p)
        if not part:
            continue
        meta = part.get("meta", {})
        for name, h in (meta.get("config_hash") or {}).items():
            if name in config_hash_now and config_hash_now[name] != h:
                out.append(f"{p}: preset {name} changed since it was computed (config hash {h} -> "
                           f"{config_hash_now[name]})")
        for r, h in (meta.get("targets_sha256") or {}).items():
            if target_sha_now.get(r) != h:
                out.append(f"{p}: targets file {r}.yaml changed since it was computed")
        code_sha_now = _code_sha(p)
        for f, h in (meta.get("code_sha256") or {}).items():
            if f in code_sha_now and code_sha_now[f] != h:
                out.append(f"{p}: {f} changed since it was computed")
    return out


# section 1 - targets
# format the tolerance column in the target units.
def _tolerance_text(tol: Mapping) -> str:
    k = tol.get("kind")
    if k in ("abs", "rel"):
        return f"{k} {_fmt_num(tol.get('value'))}"
    if k == "interval":
        return f"in [{_fmt_num(tol.get('lo'))}, {_fmt_num(tol.get('hi'))}]"
    if k == "inequality":
        s = f"{tol.get('op')} {_fmt_num(tol.get('value'))}"
        return s + (f" (hard {_fmt_num(tol['hard_value'])})" if tol.get("hard_value") is not None else "")
    if k == "band":
        s = f"band [{_fmt_num(tol.get('lo'))}, {_fmt_num(tol.get('hi'))}] slack {_fmt_num(tol.get('slack', 0))}"
        return s + (" either sign" if tol.get("either_sign") else "")
    if k == "rms_abs":
        return f"rms <= {_fmt_num(tol.get('value'))}"
    if k == "contour_asymmetry":
        return f"offsets abs {_fmt_num(tol.get('abs'))}, {tol.get('require')}"
    return str(k)


# format the measured target value with sigma and units for the measured column.
def _measured_text(t) -> str:
    v = t.value
    if isinstance(v, Mapping) and "y" in v and "x" in v:
        s = f"{_fmt_value(v['y'])} at {v.get('x_name')} {_fmt_value(v['x'])}"
    elif isinstance(v, Mapping) and v.get("form") == "rectified_sine":
        s = f"rectified sine {_fmt_value(v.get('params'))}"
    else:
        s = _fmt_value(v)
    if t.sigma is not None:
        s += f" +- {_fmt_value(t.sigma)}"
    return s + (f" {t.unit}" if t.unit not in ("1", "mixed") else "")


# return the prediction band text and the corner pass count for target `t`.
# include the nominal prediction and successful corner predictions that match the first prediction's size.
# report scalar bounds, a trajectory marker, or the largest element deviation from nominal.
# Without a successful nominal prediction, report the widest element range across corners.
def _corner_band(t, rec: Mapping, corners: Mapping[str, Mapping]) -> tuple[str, str]:
    vals = []
    if rec.get("replica_status") == "ok":
        vals.append(rec.get("predicted"))
    passes, n = 0, 0
    for corner, preds in corners.items():
        pred = preds.get(t.id) or {}
        if pred.get("status") != "ok":
            continue
        n += 1
        vals.append(pred.get("predicted"))
        if t.scored:
            try:
                if targets_io.evaluate(t, pred.get("predicted")).level == "pass":
                    passes += 1
            except Exception:                                           # noqa: BLE001
                pass
    if not vals:
        return "-", "-"
    flats = [_flatten(v) for v in vals]
    size = flats[0].size
    flats = [f for f in flats if f.size == size]
    flat_vals = np.vstack(flats)                       # one row per world, the nominal first when it ran ok
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lo, hi = np.nanmin(flat_vals, axis=0), np.nanmax(flat_vals, axis=0)
    if size == 1:
        txt = f"[{_fmt_num(lo[0])}, {_fmt_num(hi[0])}]"
    elif isinstance(vals[0], Mapping) and "t_s" in vals[0]:
        txt = "(trajectory)"
    elif rec.get("replica_status") != "ok":
        txt = f"corners only: element ranges up to {_fmt_num(float(np.nanmax(hi - lo)))} wide"
    else:
        dev = (np.nanmax(np.abs(flat_vals - flat_vals[0]), axis=0) if np.all(np.isfinite(flat_vals[0]))
               else np.full(size, np.nan))
        txt = f"nominal +- up to {_fmt_num(np.nanmax(dev)) if np.any(np.isfinite(dev)) else 'nan'}"
    return txt, (f"{passes}/{n}" if t.scored and n else "-")


# build section 1 Markdown - role counts, the z chart and one row per target.
def _section_targets(data: Mapping, presets: Mapping[str, str], chart_dir: Path, charts: bool) -> list[str]:
    pred_part = data.get("predictions")
    out = ["## 1. Targets: measured, predicted, scored", "", '<a id="targets"></a>', ""]
    if not pred_part:
        return out + ["Not computed yet: run `python -m mrm.validation.report --recompute predictions`.", ""]
    recs = {r["id"]: r for r in pred_part["records"]}
    corners = pred_part.get("corners") or {}
    out += [f"Replica protocols (`mrm.validation.replicas`) on the nominal world (every uncertainty entry at its "
            f"nominal, no device variation): G26 targets on `{presets['G26']}`, G25 bench targets on "
            f"`{presets['G25']}`. Scored with `targets_io.evaluate` against the frozen tolerances only; sigma enters "
            "only z = max |residual_i| / sigma_i. **Band** = the range of the prediction over the nominal world and "
            f"the {len(corners)} named corners of the uncertainty set (each corner's alternatives applied to the same "
            "presets): a hypothesis range, not a probability interval. **Corners pass** counts the corners whose "
            "prediction passes the tolerance.", ""]
    if presets["G26"] != "g26" and presets["G25"] == G25_DEFAULT:
        out += [f"Warning: no G25 companion of `{presets['G26']}` exists, so the G25 bench targets run on the "
                f"uncalibrated `{G25_DEFAULT}`.", ""]
    unapplied: dict[str, set] = {}                   # {"<corner> on the <plant> plant (...): <note>" - {target ids}}
    target_of = {t.id: t for role in targets_io.TARGET_ROLES for t in targets_io.load_targets(role)}
    for corner, preds in corners.items():
        for tid, pred in preds.items():
            if pred.get("status") == "error" and tid in target_of:
                plant = _plant_key(target_of[tid])
                unapplied.setdefault(f"{corner} on the {plant} plant "
                                     f"(`{presets[plant]}`): {pred.get('note', '')[:160]}", set()).add(tid)
    if unapplied:
        out += ["Corners that could not be applied (left out of the band and of the corner count): "
                + "; ".join(f"{k} ({len(v)} targets)" for k, v in sorted(unapplied.items())) + ".", ""]
    calibrated = _calibrated(presets["G26"])
    if presets["G26"] == "g26":
        out += ["The preset is uncalibrated: no parameter was fitted to any target, so the calibration rows are "
                "prior-predictive residuals, not fit residuals.", ""]
    elif calibrated:
        out += [f"Preset `{presets['G26']}` (and its G25 bench `{presets['G25']}`) was fitted to the calibration "
                "targets only (section 3.1): the calibration rows are fit residuals under the replicas' own "
                "protocols, not predictions. The validation and bounds rows are held-out predictions: the "
                "calibration never read them. Where the fit's protocol differs from the replica's, or the fit did "
                "not reach a target, the row is flagged.", ""]
    # score every target and build its table row
    rows, zrows = {}, []                             # rows by role and (id, z, result) chart rows
    counts: dict[str, dict[str, int]] = {}
    for role in targets_io.TARGET_ROLES:
        for t in targets_io.load_targets(role):
            rec = recs.get(t.id, {"replica_status": "missing", "predicted": None, "notes": []})
            status = rec.get("replica_status")
            if status == "ok":
                pred = rec.get("predicted")
                ev = targets_io.evaluate(t, float("nan") if rec.get("predicted_nonfinite") and pred is None else pred)
                lvl = ev.level
                if not t.scored:
                    res = (f"not scored ({t.status}; would be {lvl})" if lvl != "not_scored"
                           else f"not scored ({t.status})")
                else:
                    res = {"soft_fail": "marginal (soft_fail)"}.get(lvl, lvl)
                z = ev.z
            else:
                ev, lvl, z, res = None, status, None, status
                if not t.scored:
                    res = f"not scored ({t.status}); {status}"
            count = counts.setdefault(role, {"scored": 0, "pass": 0, "marginal": 0, "fail": 0, "blocked/error": 0,
                                             "not scored": 0})
            if not t.scored:
                count["not scored"] += 1
            else:
                count["scored"] += 1
                if status != "ok":
                    count["blocked/error"] += 1
                elif lvl == "pass":
                    count["pass"] += 1
                elif lvl == "soft_fail":
                    count["marginal"] += 1
                else:
                    count["fail"] += 1
            band, corners_pass = _corner_band(t, rec, corners)
            flags = [TARGET_FLAGS[t.id]] if t.id in TARGET_FLAGS else []
            if calibrated and t.id in CALIBRATED_FLAGS:
                flags.append(CALIBRATED_FLAGS[t.id])
            if rec.get("predicted_nonfinite"):
                flags.append("prediction not finite")
            if rec.get("artefact_fallbacks"):
                flags.append(f"artefact fallback: {_fmt_value(rec['artefact_fallbacks'])}")
            pred_txt = ("nan" if rec.get("predicted_nonfinite") and rec.get("predicted") is None
                        else _fmt_value(rec.get("predicted")))
            rows.setdefault(role, []).append(
                f"| {t.id} | {t.status} | {_md_cell(t.quantity)} | {_md_cell(_measured_text(t))} | "
                f"{_md_cell(pred_txt)} | {_md_cell(band)} | "
                f"{_fmt_num(z, 3)} | {_md_cell(_tolerance_text(t.tolerance))} | {_md_cell(res)} | {corners_pass} | "
                f"{_md_cell('; '.join(flags))} |")
            if t.scored and status == "ok" and z is not None:
                zrows.append((t.id, z, "pass" if lvl == "pass" else ("marginal" if lvl == "soft_fail" else "fail")))
    out += ["| role | scored | pass | marginal | fail | blocked or error | not scored |",
            "|---|---|---|---|---|---|---|"]
    for role, count in counts.items():
        out.append(f"| {role} | {count['scored']} | {count['pass']} | {count['marginal']} | {count['fail']} | "
                   f"{count['blocked/error']} | {count['not scored']} |")
    out += ["", "VAL-4c is not an independent third pass of the VAL-4 group, and VAL-4c and BND-4 rest on the same "
            "printed spread (target split handoff, request 4).", ""]
    if charts and zrows:
        _chart_z(zrows, chart_dir / "targets_z.svg")
    if zrows:
        out += [f"![z-scores of the scored targets]({_docs_relpath(chart_dir / 'targets_z.svg')})", ""]
    head = ["| id | status | quantity | measured +- sigma | predicted (nominal) | band (corners) | z | tolerance | "
            "result | corners pass | flags |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for role in targets_io.TARGET_ROLES:
        out += ([f"### 1.{targets_io.TARGET_ROLES.index(role) + 1} {role.capitalize()} targets", ""] + head
                + rows.get(role, []) + [""])
    notes = [(r["id"], n) for r in pred_part["records"] for n in r.get("notes") or []]
    if notes:
        out += ["<details><summary>Replica notes (assumptions and blocked reasons, per target)</summary>", ""]
        for i, n in notes:
            out.append(f"- **{i}**: {_md_cell(n[:400])}{'...' if len(n) > 400 else ''}")
        out += ["", "</details>", ""]
    return out


# section 2 - closure and conflict table
# find closure-anchor links in xfail reasons.
# return test ids grouped by anchor.
def _closure_tests() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    anchor_pattern = re.compile(r"VALIDATION\.md#closure-([a-z0-9-]+)")

    # read (reason, strict) from an xfail decorator call.
    # return None for other expressions.
    def xfail_reason(dec) -> tuple[str, bool] | None:
        if not isinstance(dec, ast.Call):
            return None
        fn = dec.func
        name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else None)
        if name != "xfail":
            return None
        kw = {k.arg: k.value for k in dec.keywords if k.arg}
        reason_node = kw.get("reason")
        reason = (reason_node.value if isinstance(reason_node, ast.Constant) and isinstance(reason_node.value, str)
                  else None)
        strict = kw.get("strict")
        return (reason, isinstance(strict, ast.Constant) and strict.value is True) if reason else None

    # inspect decorators so an anchor string inside a test body does not count as a closure test.
    # only xfail(reason=...) on a test function establishes the link.
    for test_path in sorted((REPO / "tests").rglob("test_*.py")):
        text = test_path.read_text(errors="replace")
        if "xfail" not in text or "closure-" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.name.startswith("test"):
                continue
            for dec in node.decorator_list:
                xfail = xfail_reason(dec)
                if not xfail:
                    continue
                reason, strict = xfail
                for m in anchor_pattern.finditer(reason):
                    found.setdefault(m.group(1), []).append(
                        f"`{test_path.relative_to(REPO).as_posix()}::{node.name}`" + ("" if strict else " (not strict)"))
    return found


# describe nominal thermal coupling for the closure-crosstalk-dynamics row.
# for the interim model, report heater 1 to ring 0 DC gain as a share of heater 0 to ring 0 gain.
# report the residue-weighted mean time constant in ms and the far-site floor.
# `stack_hypothesis` is the nominal choice for the thermal.stack uncertainty entry.
def _crosstalk_model(cfg, derived, stack_hypothesis) -> str:
    system = getattr(getattr(derived.params, "thermal", None), "system", None)
    if system is None or system.n_rings < 2:
        return "no ring-to-ring path in this preset"
    if system.meta.get("model") != "interim":
        return "the FEM ROM's own cross residues (fidelity.thermal_model rom)"
    ring0_obs, heater0_src, heater1_src = system.ring_obs[0], system.heater_src[0], system.heater_src[1]
    self_lags = np.flatnonzero((system.lag_obs == ring0_obs) & (system.lag_src == heater0_src))
    cross_lags = np.flatnonzero((system.lag_obs == ring0_obs) & (system.lag_src == heater1_src))
    z_self_k_per_w = float(system.res_k_per_w[0, self_lags].sum())      # DC gain heater 0 -> ring 0 (world 0)
    frac = float(system.res_k_per_w[0, cross_lags].sum()) / z_self_k_per_w
    tau_s = (float(system.res_k_per_w[0, cross_lags] @ system.taus_s[0, cross_lags])
             / float(system.res_k_per_w[0, cross_lags].sum()))
    pitch_um = float(derived.params.layout.ring_distance_m[0, 1]) * 1e6
    floor = cfg.leaves.get("ring.thermal.site_floor_frac")
    floor_txt = f"; far-site floor {_fmt_num(100 * floor.value)} % of Z_heater->ring" if floor is not None else ""
    return (f"interim, stack hypothesis {stack_hypothesis}: ring 1 -> ring 0 at {_fmt_num(pitch_um)} um: "
            f"{_fmt_num(100 * frac)} % with "
            f"a {_fmt_num(tau_s * 1e3)} ms pole (heat-source-to-victim distance law){floor_txt}; corner "
            f"g26_tec_package: the FEM-like numbers")


# build the preset's model cell for each conflict in `CLOSURE_ROWS`.
# `derived` is `derive(cfg)`.
# `preds` maps target ids to successful nominal predictions or None.
def _closure_model_column(cfg, derived, preds: Mapping[str, Any]) -> dict[str, str]:
    diag = derived.diagnostics
    leaves = cfg.leaves

    def leaf(path):
        x = leaves.get(path)
        return None if x is None else x.value

    def diag_value(k, i=0):
        v = diag.get(k)
        return None if v is None else float(np.asarray(v, float).ravel()[i])

    # return the pinned choice of an uncertainty entry, or its nominal choice when unpinned.
    def entry_nominal(name):
        e = cfg.uncertainty.get(name)
        return None if e is None else (e.pinned if e.pinned is not None else e.fields.get("nominal"))

    def pred_text(tid):
        v = preds.get(tid)
        return _fmt_value(v) if v is not None else "-"
    fwhm_pm, disturbance_pm_per_k = diag_value("fwhm_pm"), diag_value("disturbance_pm_per_k")
    return {
        "dlambda-dt": f"{_fmt_num(leaf('ring.thermal.dlambda_dt_pm_per_k'))} pm/K at "
                      f"{_fmt_num(leaf('ring.thermal.dlambda_dt_ref_ring_temp_c'))} degC (hypothesis "
                      f"{entry_nominal('thermo_optic')}), {_fmt_num(disturbance_pm_per_k)} pm/K at the lock "
                      f"temperature {_fmt_num(diag_value('t_ring_lock_c'))} degC; CAL-6 apparent sensitivity "
                      f"{pred_text('CAL-6')} pm/K",
        "fsr": f"{_fmt_num(diag_value('fsr_thz'))} THz = {_fmt_num(diag_value('fsr_nm'))} nm (hypothesis "
               f"{entry_nominal('fsr')})",
        "thickness": f"{_fmt_num(leaf('ring.geometry.film_nm'))} nm (hypothesis {entry_nominal('film_thickness')})",
        "q": f"Q_L {_fmt_num(leaf('ring.optics.q_loaded'))}; CAL-1 replica {pred_text('CAL-1')}",
        "heater-efficiency": f"{_fmt_num(diag_value('heater_efficiency_nm_per_mw'))} nm/mW (R2 from P_FSR "
                             f"{_fmt_num(leaf('ring.heater.p_fsr_mw'))} mW and the FSR)",
        "rth": f"{_fmt_num(diag_value('z_heater_ring_k_per_mw'))} K/mW (heater to ring, nominal world)",
        "reloma-level": f"CAL-14 replica {pred_text('CAL-14')} dB",
        "reloma-curvature": (f"FWHM / (dlambda/dT at lock) = {_fmt_num(fwhm_pm)} pm / "
                             f"{_fmt_num(disturbance_pm_per_k)} pm/K = "
                             f"{_fmt_num(fwhm_pm / disturbance_pm_per_k if fwhm_pm and disturbance_pm_per_k else None)}"
                             " K"),
        "aggressor-rate": f"CAL-13b replica {pred_text('CAL-13b')} K/s (scenario calibration)",
        "g25-topology": f"add-drop, kappa_d/kappa {_fmt_num(leaf('ring.optics.kappa_d_over_kappa'))}; the replicas "
                        "also report the all-pass reading as a variant",
        "eo-bandwidth": f"CAL-10 replica {pred_text('CAL-10')} GHz at 1 V, VAL-3a {pred_text('VAL-3a')} GHz at 2 V",
        "rom-dc-scale": (f"{_fmt_value(diag['rom_dc_scale'])}" if "rom_dc_scale" in diag else
                         "not computed: the preset uses the interim thermal model (no FEM ROM)"),
        "crosstalk-dynamics": _crosstalk_model(cfg, derived, entry_nominal("thermal.stack")),
    }


# build section 2 Markdown - one row per conflict with evidence, model choice and closure tests.
def _section_closure(cfg, derived, data: Mapping) -> list[str]:
    preds = {r["id"]: (r.get("predicted") if r.get("replica_status") == "ok" else None)
             for r in (data.get("predictions") or {}).get("records", [])}
    model = _closure_model_column(cfg, derived, preds)
    tests = _closure_tests()
    out = ["## 2. Closure and conflict table", "", '<a id="closure"></a>', "",
           "Published numbers that do not close through `derive()`. Each row is an open conflict; the model column "
           f"shows what preset `{cfg.name}` uses. CI keeps them visible through `xfail(strict=True, "
           "reason='VALIDATION.md#closure-<id>')` tests in `tests/closure/`; a row without a test is "
           "listed as such.", "", "| id | quantity | conflicting evidence | model (this preset) | closure test |",
           "|---|---|---|---|---|"]
    missing = []
    for slug, quantity, evidence in CLOSURE_ROWS:
        test_ids = tests.get(slug)
        if not test_ids:
            missing.append(slug)
        out.append(f'| <a id="closure-{slug}"></a>closure-{slug} | {_md_cell(quantity)} | {_md_cell(evidence)} | '
                   f"{_md_cell(model.get(slug))} | {', '.join(test_ids) if test_ids else 'none yet'} |")
    out.append("")
    if missing:
        out += [f"{len(missing)} of {len(CLOSURE_ROWS)} rows have no xfail(strict=True) closure test yet "
                "(requested in docs/handoff/integrator.md section 9.4).", ""]
    return out


# section 3 - identifiability and the calibration record
# return True when a non-base preset has a calibration output record.
def _calibrated(preset: str) -> bool:
    return preset != "g26" and bool(_calibration_outputs(preset))


# read calibration result records that apply to `preset`.
# scan JSON files under mrm/calibration/ and docs/calibration/.
# Accept schemas that start with `mrm.calibration` and name this preset.
# Records without a preset apply only to `FIT_RECORD_PRESETS`.
# this includes latest_fit.json, which describes the g26 and G25 fit that writes g26_calibrated.
def _calibration_outputs(preset: str) -> list[dict]:
    out = []
    for root in (REPO / "mrm" / "calibration", REPO / "docs" / "calibration"):
        if not root.is_dir():
            continue
        for f in sorted(root.rglob("*.json")):
            try:
                d = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(d, dict) and str(d.get("schema", "")).startswith("mrm.calibration") and \
                    (d.get("preset") == preset or (d.get("preset") is None and preset in FIT_RECORD_PRESETS)):
                d["_file"] = f.relative_to(REPO).as_posix()
                out.append(d)
    return out


# list violations of the target split rules for the ids used by a fit.
# validation and bounds ids indicate contamination.
# Excluded ids, unknown ids and calibration ids with `fit: false` also violate the split.
# CAL-6 checks apparent thermal sensitivity and may not be fitted.
# see mrm/validation/targets/README.md.
# return an empty list when all ids may be fitted.
def _fit_target_audit(ids: Sequence[str]) -> list[str]:
    cal_targets = {t.id: t for t in targets_io.load_calibration_targets()}
    held_out_ids = {t.id for r in targets_io.HELD_OUT_ROLES for t in targets_io.load_targets(r)}
    try:
        excluded_ids = {e.id for e in targets_io.load_excluded()}
    except Exception:                                                   # noqa: BLE001
        excluded_ids = set()
    out = []
    held_out = sorted({i for i in ids if i in held_out_ids})
    if held_out:
        out.append(f"CONTAMINATION: fitted to held-out {held_out}")
    excluded = sorted({i for i in ids if i in excluded_ids})
    if excluded:
        out.append(f"fitted to excluded {excluded}")
    unknown = sorted({i for i in ids if i not in cal_targets and i not in held_out_ids and i not in excluded_ids})
    if unknown:
        out.append(f"unknown target ids {unknown}")
    not_fittable = sorted({i for i in ids if i in cal_targets and cal_targets[i].raw.get("fit", True) is False})
    if not_fittable:
        out.append(f"fitted to {not_fittable}, which calibration.yaml marks fit: false")
    return out


# plot profile likelihoods for one calibration group.
# show chi2 - chi2_min against each parameter, with the 95 % threshold dashed and prior ends grey.
def _chart_profiles(profiles: Mapping[str, Mapping], path: Path) -> None:
    plt = _pyplot()
    names = [n for n, p in profiles.items() if p.get("grid") and p.get("chi2")]
    if not names:
        return
    ncol = min(4, len(names))
    nrow = int(math.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol + 0.6, 1.6 * nrow + 0.4), squeeze=False)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    for ax, n in zip(axes.ravel(), names):
        p = profiles[n]
        x = np.asarray([np.nan if v is None else v for v in p["grid"]], float)
        y = np.asarray([np.nan if v is None else v for v in p["chi2"]], float) - float(p.get("chi2_min") or 0.0)
        ax.plot(x, y, color=C_BLUE, lw=1.0, marker="o", ms=2)
        threshold = p.get("threshold")
        if threshold is not None:
            ax.axhline(float(threshold) - float(p.get("chi2_min") or 0.0), color=C_INK, lw=0.6, ls="--")
        prior = p.get("prior") or []
        for v in prior[:2]:
            if v is not None:
                ax.axvline(float(v), color=C_GREY, lw=0.6)
        if p.get("log"):
            ax.set_xscale("log")
        ax.set_title(str(n).split(".")[-1], fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0][0].set_ylabel("chi2 - min", fontsize=7)
    fig.tight_layout()
    _save_svg(fig, path)


# build section 3.1 from a `mrm.calibration.fit/1` record.
# for each group, show the target audit, hypothesis comparison, leaf decisions and profile chart.
def _section_calibration_fit(record: Mapping, chart_dir: Path, charts: bool) -> list[str]:
    groups = record.get("groups") or {}
    sha, _, suffix = str(record.get("git_commit") or "").partition("-")          # "<sha>[-dirty]"
    out = [f"From `{record['_file']}` (schema `{record.get('schema')}`, created {record.get('created_utc')}, commit "
           f"`{sha[:12]}`{' with uncommitted changes' if suffix == 'dirty' else ''}). Every target id the fit used "
           "(group targets, per-hypothesis residual targets, leaf decisions) is checked against the frozen split.", ""]
    for group, grp in groups.items():
        fits = grp.get("fits") or {}
        ids = set(map(str, grp.get("targets") or []))
        for fit in fits.values():
            ids |= set(map(str, (fit.get("targets") or {}).keys()))
        for decision in grp.get("decisions") or []:
            ids |= set(map(str, decision.get("targets") or [])) | set(map(str, (decision.get("residual") or {}).keys()))
        audit = _fit_target_audit(sorted(ids))
        comparison = grp.get("comparison") or {}
        out += [f"**Group `{group}`** (nominal hypothesis `{grp.get('nominal')}`, best `{comparison.get('best')}`). "
                f"Targets: {', '.join(sorted(ids)) or 'none'}. Audit: "
                + ("**" + "; ".join(audit) + "**" if audit else "ok (calibration ids only)") + ".", ""]
        rows = comparison.get("rows") or []
        if rows:
            out += ["| hypothesis | chi2 | dof | Birge ratio^2 | delta chi2 | likelihood ratio (Birge) | delta AIC |",
                    "|---|---|---|---|---|---|---|"]
            for r in rows:
                fit = fits.get(r.get("hypothesis")) or {}
                out.append(f"| {_md_cell(r.get('hypothesis'))} | {_fmt_num(r.get('chi2'))} | {r.get('dof')} | "
                           f"{_fmt_num(fit.get('birge2'))} | {_fmt_num(r.get('delta_chi2'))} | "
                           f"{_fmt_num(r.get('likelihood_ratio_birge'))} | {_fmt_num(r.get('delta_aic'))} |")
            if comparison.get("note"):
                out += ["", _md_cell(comparison["note"])]
            out.append("")
        decisions = grp.get("decisions") or []
        if decisions:
            out += ["| leaf | scope | value (was) | provenance (was) | 95 % profile hull | prior | width / prior | "
                    "rule | unique | targets |", "|---|---|---|---|---|---|---|---|---|---|"]
            for decision in decisions:
                ident = decision.get("identifiable") or {}
                out.append(f"| `{decision.get('path')}` | {decision.get('scope')} | "
                           f"{_md_cell(_fmt_value(decision.get('value')))} "
                           f"({_md_cell(_fmt_value(decision.get('previous_value')))}) | {decision.get('provenance')} "
                           f"({decision.get('previous_provenance')}) | "
                           f"{_md_cell(_fmt_value(ident.get('profile_95')))} | "
                           f"{_md_cell(_fmt_value(ident.get('prior')))} | {_fmt_num(ident.get('width_frac'), 3)} | "
                           f"{_md_cell(ident.get('rule'))} | {_md_cell(_fmt_value(ident.get('unique')))} | "
                           f"{_md_cell(_fmt_value(decision.get('targets')))} |")
            out.append("")
        # show profiles for the written nominal hypothesis to match the leaf decisions above.
        # use the best hypothesis when the nominal has no profiles.
        hyp = grp.get("nominal") if (grp.get("profiles") or {}).get(grp.get("nominal")) else comparison.get("best")
        profiles = (grp.get("profiles") or {}).get(hyp) or {}
        if profiles:
            f = chart_dir / f"profiles_{re.sub(r'[^a-z0-9_]+', '_', str(group).lower())}.svg"
            if charts:
                _chart_profiles(profiles, f)
            out += [f"![profile likelihoods, group {group}, hypothesis {hyp}]({_docs_relpath(f)})", ""]
    if record.get("notes"):
        out += ["Calibration notes: " + _md_cell("; ".join(map(str, record["notes"]))), ""]
    return out


# build section 3 Markdown - the Jacobian screen, calibration records and calibrated-leaf audit.
def _section_identifiability(cfg, data: Mapping, preset: str, chart_dir: Path, charts: bool) -> list[str]:
    ident_part = data.get("identifiability")
    out = ["## 3. Identifiability of the calibration parameters", "", '<a id="identifiability"></a>', ""]
    if not ident_part:
        out += ["Not computed yet: run `python -m mrm.validation.report --recompute identifiability`.", ""]
    else:
        svd = ident_part.get("svd") or {}
        out += [f"Jacobian of the calibration replicas (calibration targets only; the scenario calibrations "
                f"{', '.join(SCENARIO_REPLICAS)} and the never-fitted CAL-6 left out) with respect to "
                f"{len(ident_part['params'])} candidate parameters of the calibration targets, by central differences "
                f"of {_fmt_num(100 * ident_part['meta'].get('rel_step', 0.05))} % of each prior range (one-sided at a "
                "range end), "
                "at the preset's values. Entries are normalized: residual sigmas (the target sigma; a quarter of an "
                "interval tolerance or half an abs/rel tolerance where no sigma is printed) per prior half-width "
                "(the leaf's `range`). In these units the half-prior rule (95 % interval at most half the "
                "prior range) is 1.96 sigma <= 0.5, i.e. a direction is identified when its singular value is at "
                f"least {sensitivity.HALF_PRIOR_SINGULAR_VALUE:.2f}.", "",
                "This is a local, linearized screen at a point that is not a fit (the preset is "
                + ("uncalibrated" if preset == "g26" else "the calibrated one") + "): a parameter that fails it "
                "cannot be tagged `calibrated` from these targets; one that passes still needs the profile "
                "likelihood (below). The screen works in linear coordinates of each range and fits all "
                "candidates jointly; the calibration task applies the rule to profile intervals per group, in log "
                "coordinates for a log-scale parameter, so its verdicts can differ.", ""]
        if svd:
            out += [f"Residuals used: {svd.get('n_residuals_used')} (dropped as non-finite: "
                    f"{svd.get('n_residuals_dropped')}); rank {svd.get('rank')} of {len(svd.get('names', []))}; "
                    f"condition number {_fmt_num(svd.get('condition'))}; directions passing the rule: "
                    f"{sum(1 for x in svd.get('directions', []) if x['identified'])} of "
                    f"{len(svd.get('directions', []))}.", ""]
            if charts:
                _chart_svd(svd["s"], svd["threshold_singular_value"], chart_dir / "identifiability_svd.svg")
            out += [f"![singular values]({_docs_relpath(chart_dir / 'identifiability_svd.svg')})", ""]
            out += ["| k | singular value | rule | largest loadings |", "|---|---|---|---|"]
            for k, direction in enumerate(svd["directions"], start=1):
                loadings = ", ".join(f"{n.split('.')[-1]} {v:+.2f}" for n, v in direction["loadings"])
                out.append(f"| {k} | {_fmt_num(direction['s'], 3)} | "
                           f"{'identified' if direction['identified'] else 'not identified'} | "
                           f"{_md_cell(loadings)} |")
            out.append("")
        svd_index = {n: i for i, n in enumerate((svd or {}).get("names", []))}
        out += ["| parameter | meant to explain | prior range | value | column norm | linearized 95 % interval / "
                "prior range | half-prior rule | provenance |", "|---|---|---|---|---|---|---|---|"]
        for p in ident_part["params"]:
            if "skipped" in p:
                out.append(f"| `{p['path']}` | {_md_cell(p['identifies'])} | | | | skipped: {_md_cell(p['skipped'])} "
                           "| | |")
                continue
            i = svd_index.get(p["path"])
            half_width = svd["half_width_95"][i] if i is not None else None
            out.append(f"| `{p['path']}` | {_md_cell(p['identifies'])} | "
                       f"[{_fmt_num(p['prior'][0])}, {_fmt_num(p['prior'][1])}] | "
                       f"{_fmt_num(p.get('value'))} | {_fmt_num(svd['column_norm'][i], 3) if i is not None else '-'} | "
                       f"{_fmt_num(half_width, 3)} | "
                       f"{'pass' if half_width is not None and half_width <= 0.5 else 'fail'} "
                       f"| {p.get('provenance')} |")
        kept = [p for p in ident_part["params"] if p.get("kept")]
        if kept:
            out += ["", "Moved on the G26 plant only (the other plant's leaf has another value, so it is a different "
                    "physical quantity there; its targets get a zero column): " + "; ".join(
                        f"`{p['path']}` ({_md_cell('; '.join(p['kept'].values()))})" for p in kept) + "."]
        out += ["", "Base point (the preset's values): " + ", ".join(
            f"{r['id']} {r['level']}" + (f" (z {_fmt_num(r['z'], 3)})" if r.get("z") is not None else "")
            for r in ident_part.get("base_evaluation", [])) + ".", ""]
        if svd:
            col_norm = np.asarray(svd["column_norm"], float)
            zero = [n for n, c in zip(svd["names"], col_norm) if c <= 1e-9 * max(col_norm.max(), 1e-300)]
            fail = [n for n, h in zip(svd["names"], svd["half_width_95"]) if not h <= 0.5]
            if zero:
                out += ["No calibration target responds to " + ", ".join(f"`{n}`" for n in zero) + " in this preset "
                        "(a zero Jacobian column): it cannot be calibrated from these targets at all.", ""]
            if fail:
                out += [f"Failing the linearized half-prior rule (all {len(svd['names'])} fitted jointly): " +
                        ", ".join(f"`{n}`" for n in fail) + ". They keep `assumed` (or their current provenance) with "
                        "a range unless another target is added.", ""]
            big = [r["id"] for r in ident_part.get("base_evaluation", []) if r.get("z") is not None and r["z"] > 10]
            if big:
                out += ["Caution: " + ", ".join(big) + " miss by more than 10 sigma at this point, so the largest "
                        "singular values describe the slope far from any fit (CAL-2 also crosses critical coupling, "
                        "where the notch depth is not smooth). The linearization says which combinations the targets "
                        "constrain, not the values a fit would reach; a model-form error (a target no parameter "
                        "value can reach) shows up as a large residual that survives the fit, not here.", ""]
    # profile likelihoods (calibration task) and the calibrated-leaf audit
    out += ["### 3.1 Profile likelihoods (calibration task)", ""]
    cal_records = _calibration_outputs(preset)
    if not cal_records:
        out += ["No calibration output for this preset was found (`mrm/calibration/latest_fit.json`, schema "
                "`mrm.calibration.fit/1`, or a JSON file under `mrm/calibration/` or `docs/calibration/` with "
                "`schema: mrm.calibration...`, `preset` and `parameters: [{path, prior, estimate, profile_ci95, "
                "half_prior_rule, unique, targets, residual}]`). The profile likelihoods are therefore not "
                "shown.", ""]
    for record in cal_records:
        if isinstance(record.get("groups"), Mapping):
            out += _section_calibration_fit(record, chart_dir, charts)
            continue
        fitted_params = record.get("parameters") or []
        ids = sorted({str(x) for p in fitted_params for x in (p.get("targets") or [])})
        audit = _fit_target_audit(ids)
        out += [f"From `{record['_file']}`. Fitted targets: {', '.join(ids) or 'none listed'}; audit: "
                + ("; ".join(audit) if audit else "every fitted id is a calibration.yaml id with fit allowed") + ".",
                "",
                "| parameter | prior | estimate | 95 % profile interval | half-prior rule | unique | targets | "
                "residual |",
                "|---|---|---|---|---|---|---|---|"]
        for p in fitted_params:
            out.append(f"| `{p.get('path')}` | {_md_cell(_fmt_value(p.get('prior')))} | "
                       f"{_md_cell(_fmt_value(p.get('estimate')))} | "
                       f"{_md_cell(_fmt_value(p.get('profile_ci95')))} | "
                       f"{_md_cell(_fmt_value(p.get('half_prior_rule')))} | "
                       f"{_md_cell(_fmt_value(p.get('unique')))} | {_md_cell(_fmt_value(p.get('targets')))} | "
                       f"{_md_cell(_fmt_value(p.get('residual')))} |")
        out.append("")
    calibrated_leaves = [(p, leaf) for p, leaf in sorted(cfg.leaves.items()) if leaf.provenance == "calibrated"]
    out += ["### 3.2 Leaves tagged `calibrated` in this preset", ""]
    if not calibrated_leaves:
        out += [f"None: preset `{cfg.name}` has no `calibrated` leaf.", ""]
    else:
        out += ["| leaf | value | targets | residual | identifiable | audit |", "|---|---|---|---|---|---|"]
        jacobian_params = {p["path"]: p for p in (ident_part or {}).get("params", [])}
        svd_index = {n: i for i, n in enumerate(((ident_part or {}).get("svd") or {}).get("names", []))}
        for p, leaf in calibrated_leaves:
            audit = []
            if not leaf.targets:
                audit.append("no targets listed (every calibrated leaf must name its targets)")
            audit += _fit_target_audit([str(x) for x in (leaf.targets or [])])
            if leaf.residual is None or leaf.identifiable is None:
                audit.append("residual or identifiability missing (the half-prior rule needs both)")
            if p in svd_index and ident_part and (ident_part.get("svd") or {}).get("half_width_95"):
                if ident_part["svd"]["half_width_95"][svd_index[p]] > 0.5:
                    audit.append("fails the linearized half-prior rule above")
            elif p not in jacobian_params:
                audit.append("not in the Jacobian parameter set")
            out.append(f"| `{p}` | {_fmt_value(leaf.value)} | {_fmt_value(leaf.targets)} | "
                       f"{_fmt_value(leaf.residual)} | "
                       f"{_fmt_value(leaf.identifiable)} | {_md_cell('; '.join(audit) or 'ok')} |")
        out.append("")
    return out


# section 4 - uncertainty propagation
# build section 4 Markdown - the protocol, cost, tornado responses and screened Sobol indices.
def _section_uncertainty(data: Mapping, chart_dir: Path, charts: bool) -> list[str]:
    sens_part = data.get("sensitivity")
    out = ["## 4. Uncertainty propagation over the uncertainty set", "", '<a id="uncertainty"></a>', ""]
    if not sens_part:
        return out + ["Not computed yet: run `python -m mrm.validation.report --recompute sensitivity`.", ""]
    meta = sens_part["meta"]
    n_top = len(sens_part["top"])
    out += ["Performance is conditional on the assumed thermal dynamics (heater, crosstalk and aggressor poles) "
            "until question 1 for Lightmatter (the heater step response at the lock point) is answered.", "",
            f"Protocol: controller `{meta['controller']}` under {meta['scenario']}, warm start, self reference, "
            "metrics only (the compare_locks protocol for one controller), variation off (epistemic worlds "
            "only). "
            f"Every world is one row of a u-design injected into `realize(..., uncertainty='lhs')` and checked "
            "against realize's draws (`mrm.validation.sensitivity`). The tornado moves one entry at a time to its "
            "interval ends or to each reachable hypothesis choice (stress choices with zero weight are corners, not "
            "part of the sampled set), all other entries at nominal; the base world is the nominal world (a "
            "nominal at an interval end becomes 1e-12 of the range from it). The top "
            f"{n_top} entries by normalized swing (largest swing over the metrics, each metric normalized by "
            "its largest finite swing) then get Saltelli (2010) first-order and Jansen total Sobol indices over "
            "their prior (u-space; hypotheses by their weights), the other entries held at nominal, so the indices "
            "are conditional on those. Bootstrap intervals: "
            f"{int(100 * 0.95)} %, {meta.get('n_boot')} resamples. "
            + ("Noise seeds differ by world index (world b uses base + b), so a Saltelli row and its AB_i partner "
               f"also differ in noise: the nominal world rerun with {meta['n_noise_replicates']} other noise seeds "
               "gives the noise variance of each metric below; its share of the output variance is added to every "
               "ST (and is the smallest ST that means anything)." if meta.get("n_noise_replicates") else
               "Noise seeds differ by world index (world b uses base + b); the noise variance was not measured in "
               "this computation."), "",
            f"Cost: {meta['n_tornado_worlds']} tornado worlds + {meta['n_sobol_worlds']} Sobol worlds "
            f"({meta['n_base']} base rows x ({n_top} + 2)), at most {meta['chunk']} worlds per Plant, "
            f"{meta['n_threads']} kernel threads; wall {_fmt_num(meta['wall_s'])} s (Sobol part "
            f"{_fmt_num(meta.get('sobol_wall_s'))} s), peak RSS {_fmt_num(meta['peak_rss_gb'])} GB of the report "
            "process.", ""]
    if sens_part.get("tornado_supply_limited"):
        out += ["Tornado worlds that start supply-limited (the bank cannot hold every lock at the warm start, "
                "experiments handoff): " + ", ".join(f"`{k}`" for k in sens_part["tornado_supply_limited"]) + ".", ""]
    if sens_part.get("sobol_supply_limited_frac") is not None:
        out += ["Fraction of Sobol worlds that start supply-limited: "
                f"{_fmt_num(sens_part['sobol_supply_limited_frac'], 3)}.", ""]
    base = sens_part["base"]
    out += ["Nominal world: " + ", ".join(f"{k} {_fmt_num(v)}" for k, v in base.items()) + ".", ""]

    # show at most 12 moving entries per metric in the tornado table and chart.
    out += ["### 4.1 Tornado (one entry at a time)", ""]
    for metric in sens_part["metrics"]:
        key = metric["key"]
        rows = sens_part["tornado"].get(key, [])
        title = f"{key}" + (f" ({metric['unit']})" if metric["unit"] else "") + "; blue: below nominal, orange: above"
        f = chart_dir / f"tornado_{key}.svg"
        # a NaN swing means the metric is undefined at every tornado point for this entry.
        moving = [r for r in rows if r["swing"] != 0 and not math.isnan(r["swing"])]
        no_effect = [r["entry"] for r in rows if r["swing"] == 0]
        undefined = [r["entry"] for r in rows if math.isnan(r["swing"])]
        if charts and rows:
            _chart_tornado(moving or rows[:1], base.get(key, float("nan")), title, f)
        out += [f"**{key}** (nominal {_fmt_num(base.get(key))} {metric['unit']})", "",
                f"![tornado {key}]({_docs_relpath(f)})", "",
                "| entry | low end | high end | swing |", "|---|---|---|---|"]
        for r in moving[:12]:
            out.append(f"| `{r['entry']}` | {_fmt_num(r['lo'])} ({_md_cell(r['lo_label'])}) | {_fmt_num(r['hi'])} "
                       f"({_md_cell(r['hi_label'])}) | {_fmt_num(r['swing'])} |")
        more = moving[12:]
        if more:
            out.append(f"| ({len(more)} more entries) | | | <= {_fmt_num(max(r['swing'] for r in more))} |")
        out.append("")
        if no_effect:
            out += [f"No change of {key} when moved alone (the others at nominal): "
                    + ", ".join(f"`{e}`" for e in no_effect) + ".", ""]
        if undefined:
            out += [f"{key} is not defined (NaN) at every tornado point of: "
                    + ", ".join(f"`{e}`" for e in undefined) + ".", ""]

    # Sobol indices for the screened entries
    out += ["### 4.2 Sobol indices of the top entries", "",
            "Screened entries: " + ", ".join(f"`{e}`" for e in sens_part["top"]) + ".", ""]
    if charts:
        _chart_sobol(sens_part["sobol"], sens_part["top"], chart_dir / "sobol.svg")
    out += [f"![Sobol indices]({_docs_relpath(chart_dir / 'sobol.svg')})", ""]
    for metric in sens_part["metrics"]:
        key = metric["key"]
        sob = sens_part["sobol"].get(key, {})
        header = f"**{key}**" + (f" (indices of {sob['transform']})" if sob.get("transform") else "")
        if "st" not in sob:
            out += [header + ": " + "; ".join(sob.get("notes", ["not computed"])), ""]
            continue
        has_binned = "s1_binned" in sob
        noise = (f", noise variance {_fmt_num(sob['noise_variance'])} ({_fmt_num(100 * sob['noise_share'], 2)} % of it)"
                 if sob.get("noise_share") is not None and np.isfinite(sob.get("noise_share", np.nan)) else "")
        sum_s1 = sum(x for x in sob["s1"] if x is not None)
        sum_st = sum(x for x in sob["st"] if x is not None)
        sum_st_lower = sum(r[0] for r in sob["st_ci"] if r[0] is not None)
        out += [header + f": variance {_fmt_num(sob['variance'])}{noise}, rows dropped as non-finite "
                f"{sob.get('n_rows_dropped_nonfinite', 0)}, sum of S1 {_fmt_num(sum_s1, 3)}"
                + (f" (binned: {_fmt_num(sum(x for x in sob['s1_binned'] if x is not None), 3)})" if has_binned
                   else "")
                + f", sum of ST {_fmt_num(sum_st, 3)}"
                + (f" (sum of the ST lower 95 % bounds {_fmt_num(sum_st_lower, 3)})")
                + ("; " + "; ".join(sob["notes"]) if sob.get("notes") else ""), "",
                "| entry | S1 | S1 95 % |" + (" S1 binned | S1 binned 95 % |" if has_binned else "")
                + " ST | ST 95 % |",
                "|---|---|---|" + ("---|---|" if has_binned else "") + "---|---|"]
        for i, n in enumerate(sob["names"]):
            binned_cells = (f" {_fmt_num(sob['s1_binned'][i], 3)} | [{_fmt_num(sob['s1_binned_ci'][i][0], 2)}, "
                            f"{_fmt_num(sob['s1_binned_ci'][i][1], 2)}] |" if has_binned else "")
            out.append(f"| `{n}` | {_fmt_num(sob['s1'][i], 3)} | "
                       f"[{_fmt_num(sob['s1_ci'][i][0], 2)}, {_fmt_num(sob['s1_ci'][i][1], 2)}] |"
                       + binned_cells + f" {_fmt_num(sob['st'][i], 3)} | "
                       f"[{_fmt_num(sob['st_ci'][i][0], 2)}, {_fmt_num(sob['st_ci'][i][1], 2)}] |")
        out.append("")
    n_base = sens_part["meta"]["n_base"]
    n_other_free = sum(1 for x in sens_part["entries"].values() if x.get("pinned") is None) - n_top
    any_binned = any("s1_binned" in (sens_part["sobol"].get(m["key"]) or {}) for m in sens_part["metrics"])
    out += ["Reading: ST - S1 is the share of variance through interactions; a sum of ST above 1 (judge it by the "
            "sum of the lower bounds) means the metric responds to combinations of entries, typically a threshold "
            "such as a lock loss or a supply limit. The Saltelli S1 (from the A, B and AB_i rows) is poorly resolved "
            f"with {n_base} base rows for these threshold-like metrics (a negative S1 or an interval "
            "that spans 0 is estimator noise, not a negative effect). "
            + ("'S1 binned' re-estimates S1 from the 2 x "
               f"{n_base} independent A and B rows by binning each entry (a hypothesis by its choices, "
               "an interval in 8 equal-probability bins, bias of the bin means removed; "
               "`sensitivity.given_data_first_order`): better resolved, slightly low for a smooth interval effect. "
               if any_binned else "")
            + f"All indices are conditional on the other {n_other_free} "
            "free entries at nominal.", ""]
    return out


# section 5 - controller rank stability
# build section 5 Markdown - Kendall tau stability per metric and one row per controller.
def _section_ranking(data: Mapping) -> list[str]:
    rank_part = data.get("ranking")
    out = ["## 5. Controller rank stability across worlds", "", '<a id="ranking"></a>', "",
           "Performance is conditional on the assumed thermal dynamics (heater, crosstalk and aggressor poles) until "
           "question 1 for Lightmatter (the heater step response at the lock point) is answered.", ""]
    if not rank_part:
        return out + ["Not computed yet: run `python -m mrm.validation.report --recompute ranking`.", ""]
    source = rank_part["meta"].get("source", {})
    rank_metrics = rank_part.get("rank_metrics") or []
    out += [f"Source: `mrm.sim.experiment` kind compare ({_md_cell(json.dumps(source, sort_keys=True))}); wall "
            f"{_fmt_num(rank_part['meta'].get('wall_s'))} s. For each rank metric (lower is better), Kendall's tau-b "
            "between every world's ranking (Latin-hypercube worlds and named corners) and the nominal ranking; "
            "'defined' is the fraction of worlds where tau exists (a ranking of all ties has none).", "",
            "| metric | nominal order (best first) | median order | tau mean | tau median | tau min | same order | "
            "defined |", "|---|---|---|---|---|---|---|---|"]
    for k, stab in (rank_part.get("rank_stability") or {}).items():
        out.append(f"| {k} | {' < '.join(stab.get('nominal_order') or [])} | "
                   f"{' < '.join(stab.get('median_order') or [])} | "
                   f"{_fmt_num(stab.get('tau_mean'), 3)} | {_fmt_num(stab.get('tau_median'), 3)} | "
                   f"{_fmt_num(stab.get('tau_min'), 3)} | "
                   f"{_fmt_num(stab.get('frac_same_order'), 3)} | {_fmt_num(stab.get('frac_defined'), 3)} |")
    out += ["", "| controller | nominal | worst named corner | " + " | ".join(
        f"{k} median [p5, p95]" for k in rank_metrics) + " | supply-limited worlds |",
            "|---|---|---|" + "---|" * len(rank_metrics) + "---|"]
    for label, ctrl in rank_part["controllers"].items():
        nominal = ctrl.get("nominal") or {}
        quantiles = ctrl.get("quantiles") or {}
        cells = []
        for k in rank_metrics:
            q = quantiles.get(k) or {}
            cells.append(f"{_fmt_num(q.get('p50'))} [{_fmt_num(q.get('p5'))}, {_fmt_num(q.get('p95'))}]")
        worst = ctrl.get("worst_corner_overall") or {}
        supply_limited = (quantiles.get("warm_start_supply_limited") or {}).get("mean")
        out.append(f"| {label} | " + _md_cell(", ".join(f"{k} {_fmt_num(nominal.get(k))}" for k in rank_metrics))
                   + f" | {_md_cell(worst.get('corner'))}: {_md_cell(worst.get('why'))} | " + " | ".join(cells)
                   + f" | {_fmt_num(supply_limited, 3)} |")
    out += ["", "A ranking claim holds only if it holds over the whole uncertainty set. The worlds mix controller "
            "robustness with warm-start failures (supply-limited worlds start with rings the bank cannot hold; "
            "experiments handoff 9.7 and 12).", ""]
    return out


# sections 6, 7 and the contamination warnings
# trace a derived quantity to its config leaves.
# return leaf paths grouped by generation.
# `seen` prevents a cycle in the derivation graph from causing infinite recursion.
def _leaf_generations(path: str, cfg, derivations: Mapping, seen: frozenset = frozenset()) -> dict[str, set]:
    if path in seen:
        return {}
    seen = seen | {path}
    leaf = cfg.leaves.get(path)
    if leaf is not None and not leaf.is_free:
        return {str(leaf.generation or "none"): {path}}
    derivation = derivations.get(path)
    out: dict[str, set] = {}
    if derivation is None:
        return out
    for input_path in derivation.inputs:
        for generation, leaf_paths in _leaf_generations(input_path, cfg, derivations, seen).items():
            out.setdefault(generation, set()).update(leaf_paths)
    return out


# build section 6 Markdown - input generations and weakest provenance for derived quantities.
# include diagnostics and counts of config leaves by generation and provenance.
def _section_generation(cfg, derived) -> list[str]:
    from mrm.params.derive import effective_provenance
    derivations = derived.params.derivations
    out = ["## 6. Generation mix of the derived quantities", "", '<a id="generation"></a>', "",
           f"Preset `{cfg.name}`: " + _md_cell((cfg.meta or {}).get("generation_mix", "(no meta.generation_mix)")), "",
           "Each derived quantity (the relations of mrm.params.relations and the diagnostics of `derive()`) with the "
           "generation of every config leaf it rests on (recursively) and its weakest input provenance.", "",
           "| quantity | generations (leaves) | weakest provenance |", "|---|---|---|"]
    items = [(p, [p]) for p in sorted(derivations) if not p.startswith("derived.timing")]
    items += [(f"diagnostic `{k}`", list(derived.inputs.get(k) or [])) for k in derived.diagnostics
              if derived.inputs.get(k)]
    for name, inputs in items:
        generations: dict[str, set] = {}
        for input_path in inputs:
            for generation, leaf_paths in _leaf_generations(input_path, cfg, derivations).items():
                generations.setdefault(generation, set()).update(leaf_paths)
        generation_text = "; ".join(
            f"**{generation}**: " + ", ".join(sorted(p[5:] if p.startswith("ring.") else p for p in leaf_paths))
            for generation, leaf_paths in sorted(generations.items()))
        provenance = (effective_provenance(name, cfg, derivations) if not name.startswith("diagnostic")
                      else derived.provenance.get(name.split("`")[1], ""))
        out.append(
            f"| {('`' + name + '`') if not name.startswith('diagnostic') else name} | {_md_cell(generation_text)} | "
            f"{provenance} |")
    counts: dict[tuple[str, str], int] = {}          # leaf counts by (generation, provenance)
    for p, leaf in cfg.leaves.items():
        if leaf.is_free:
            continue
        key = (str(leaf.generation or "none"), leaf.provenance)
        counts[key] = counts.get(key, 0) + 1
    generations = sorted({generation for generation, _ in counts})
    provenances = sorted({p for _, p in counts})
    out += ["", "Leaves by generation and provenance:", "", "| generation | " + " | ".join(provenances) + " | total |",
            "|---|" + "---|" * (len(provenances) + 1)]
    for generation in generations:
        row = [counts.get((generation, p), 0) for p in provenances]
        out.append(f"| {generation} | " + " | ".join(str(x) for x in row) + f" | {sum(row)} |")
    out.append("")
    return out


# build section 7 Markdown for leaves with `textbook:` or `disagrees:` fields.
def _section_textbook(cfg) -> list[str]:
    from mrm.params.table import format_value
    out = ["## 7. Disagreements with the textbook", "", '<a id="textbook"></a>', "",
           f"Generated from the `textbook:` and `disagrees:` fields of preset `{cfg.name}`. Where the "
           "leaf carries a `textbook` alternative, its value is shown (select it with `--alt <path>=textbook`).", "",
           "| key | value here | textbook field | textbook alternative | disagreement |", "|---|---|---|---|---|"]
    disagreement_count = 0
    for path, leaf in sorted(cfg.leaves.items()):
        if not (leaf.textbook or leaf.disagrees):
            continue
        disagreement_count += 1
        alternative = leaf.alternatives.get("textbook") if leaf.alternatives else None
        alt_value = (format_value(alternative.get("value"))
                     if isinstance(alternative, Mapping) and "value" in alternative else "")
        value = "(derived)" if leaf.is_free else format_value(leaf.value)
        out.append(f"| `{path}` | {_md_cell(value)} | {_md_cell(leaf.textbook)} | {_md_cell(alt_value)} | "
                   f"{_md_cell(leaf.disagrees)} |")
    out += ["", f"{disagreement_count} leaves disagree with the textbook.", ""]
    return out


# copy contamination warnings from the targets README into Markdown.
def _section_contamination() -> list[str]:
    text = (targets_io.TARGETS_DIR / "README.md").read_text()
    heading_start = text.find("## Contamination warnings")
    body = (text[heading_start:].split("\n", 1)[1].strip()
            if heading_start >= 0 else "(section not found in the targets README)")
    return ["## Contamination warnings", "", '<a id="contamination"></a>', "",
            "Copied from `mrm/validation/targets/README.md`: research-report fits that used held-out data; their "
            "results must not become defaults or priors.", "", body, ""]


# render - the whole report
# render docs/VALIDATION.md from cached `data` and current preset and target files.
# `stamp` supplies date, commit and dirty fields.
# by default, use today and the current git state.
# `doc_text` supplies the frozen-targets block to copy verbatim.
# it defaults to docs/VALIDATION.md.
# write SVG charts when `charts` is True.
# store them in `chart_dir`, or docs/validation/<preset>/.
def render(data: Mapping, *, preset: str, g25_preset: str | None = None, stamp: Mapping | None = None,
           doc_text: str | None = None, charts: bool = True, chart_dir: Path | None = None) -> str:
    from mrm.params.derive import derive
    g25 = companion_g25(preset, g25_preset)
    presets = {"G26": preset, "G25": g25}
    chart_dir = chart_dir or data_dir(preset)
    doc_text = DOC_PATH.read_text(encoding="utf-8") if doc_text is None else doc_text
    block = _frozen_block(doc_text)
    if stamp is None:
        git = _git_stamp()
        stamp = {"date": datetime.date.today().isoformat(), "commit": git["commit"], "dirty": git["dirty"]}
    stamp_record = {"preset": preset, "g25_preset": g25, "date": stamp.get("date"), "commit": stamp.get("commit"),
                    "dirty": stamp.get("dirty")}        # the report-stamp record (read back by read_stamp)
    cfg = _load_cfg(preset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        derived = derive(cfg)
    header_errs = targets_io.header_errors(None)
    out = ["# Validation report", "", block, "",
           f"<!-- report-stamp {json.dumps(stamp_record, sort_keys=True)} -->",
           f"Generated by `python -m mrm.validation.report --preset {preset}` (also `mrm report`) on "
           f"{stamp_record['date']} from commit `{stamp_record['commit']}`"
           f"{' with uncommitted changes' if stamp_record['dirty'] else ''}. Preset `{preset}` "
           f"(config hash `{cfg.hash()[:16]}`; files {', '.join(f'`{f}`' for f in cfg.files)}); G25 bench targets on "
           f"`{g25}`. Do not edit by hand: every section below is generated.", "",
           "The two held-out target files above were frozen before any calibration fit or controller comparison; "
           "the block is owned by the target split and copied verbatim. The hashes are of the file "
           "bytes with CRLF read as LF; any other edit is a re-freeze (`mrm/validation/targets/README.md`). "
           "Frozen-hash check now: " + ("**OK**" if not header_errs else "**FAILED**: " + "; ".join(header_errs))
           + ".", "",
           "Computed parts (`docs/validation/" + preset + "/report_data.json`; `--fast` re-renders from it):", "",
           "| part | computed (UTC) | commit | wall (s) | peak RSS (GB) | presets (config hash) |",
           "|---|---|---|---|---|---|"]
    out += [_part_table_row(p, data.get(p)) for p in PARTS]
    stale = _staleness(data, presets)
    out += [""]
    if stale:
        out += ["**Stale parts** (recompute them): " + "; ".join(stale) + ".", ""]
    out += ["Sections: [1 targets](#targets) . [2 closure](#closure) . [3 identifiability](#identifiability) . "
            "[4 uncertainty](#uncertainty) . [5 ranking](#ranking) . [6 generation mix](#generation) . "
            "[7 textbook](#textbook) . [contamination warnings](#contamination)", ""]
    out += _section_targets(data, presets, chart_dir, charts)
    out += _section_closure(cfg, derived, data)
    out += _section_identifiability(cfg, data, preset, chart_dir, charts)
    out += _section_uncertainty(data, chart_dir, charts)
    out += _section_ranking(data)
    out += _section_generation(cfg, derived)
    out += _section_textbook(cfg)
    out += _section_contamination()
    return "\n".join(out).rstrip("\n") + "\n"


# driver and command line
# compute requested parts, save the cache and write the report.
# return the report path.
# `recompute` defaults to all `PARTS`.
# an empty sequence selects fast mode.
# save the cache to docs/validation/<preset>/report_data.json after each part so later failures retain completed results.
# write the report to `out`, or docs/VALIDATION.md by default.
# `n_base` sets Sobol base rows.
# `compare_worlds` sets Latin-hypercube worlds for the compare.
# `compare_run` reads an existing compare run instead of starting one.
def generate(preset: str = "g26", *, g25_preset: str | None = None, recompute: Sequence[str] | None = None,
             out: str | Path | None = None, n_base: int = 128, compare_worlds: int = 256,
             compare_run: str | Path | None = None, n_threads: int = 4, stamp: Mapping | None = None,
             log: Callable[[str], None] = _log_to_stderr) -> Path:
    recompute = PARTS if recompute is None else tuple(recompute)
    bad = set(recompute) - set(PARTS)
    if bad:
        raise ValueError(f"unknown parts {sorted(bad)}; choose from {PARTS}")
    g25 = companion_g25(preset, g25_preset)
    presets = {"G26": preset, "G25": g25}
    for name in presets.values():
        _load_cfg(name)                                    # a missing or invalid preset fails here, before any work
    data = load_data(preset)
    data["schema"], data["preset"] = SCHEMA, preset
    for p in recompute:
        log(f"report: computing {p} on {presets}")
        if p == "predictions":
            data[p] = compute_predictions(presets, log=log)
        elif p == "identifiability":
            data[p] = compute_identifiability(presets, log=log)
        elif p == "sensitivity":
            data[p] = compute_sensitivity(preset, n_base=n_base, n_threads=n_threads, log=log)
        elif p == "ranking":
            data[p] = compute_ranking(preset, n_mc=compare_worlds, compare_run=compare_run, n_threads=n_threads,
                                      log=log)
        save_data(preset, data)                    # after every part, so a later failure keeps the earlier parts
    out = Path(out) if out is not None else DOC_PATH
    doc_text = out.read_text(encoding="utf-8") if out.exists() else DOC_PATH.read_text(encoding="utf-8")
    text = render(data, preset=preset, g25_preset=g25_preset, stamp=stamp, doc_text=doc_text)
    out.write_text(text, encoding="utf-8")
    log(f"report: wrote {out}")
    return out


# add report arguments to `p`, or create a new parser.
# `p` can be the `mrm report` subparser.
def build_parser(p: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    p = p or argparse.ArgumentParser(prog="python -m mrm.validation.report", description='Generate the validation report docs/VALIDATION.md from the replicas, the target files and the preset.')
    p.add_argument("--preset", default="g26", help="preset the report is about (g26, g26_calibrated, ...)")
    p.add_argument("--g25-preset", default=None, help="G25 bench preset (default: the companion of --preset)")
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument("--fast", action="store_true", help="recompute nothing; re-render from the cached data")
    mode_group.add_argument("--recompute", default=None,
                            help=f"comma-separated parts to recompute (others from the cache): {','.join(PARTS)}")
    p.add_argument("--check", action="store_true",
                   help="fast render in memory and compare with the report file (keeps its stamp); exit 1 if stale")
    p.add_argument("--out", default=None, help="report file (default docs/VALIDATION.md)")
    p.add_argument("--sobol-n", type=int, default=128, help="Sobol base rows (worlds = n (top + 2))")
    p.add_argument("--compare-worlds", type=int, default=256, help="Latin-hypercube worlds of the section-5 compare")
    p.add_argument("--compare-run", default=None, help="reuse an existing compare run directory for section 5")
    p.add_argument("--threads", type=int, default=4, help="kernel threads")
    return p


# list reasons why the report file `out` is stale.
# return an empty list when it is current.
# check for text differences from a fast render with the saved stamp, missing charts and stale cached parts.
def check(preset: str | None = None, out: str | Path | None = None) -> list[str]:
    out = Path(out) if out is not None else DOC_PATH
    text = out.read_text(encoding="utf-8")
    stamp_record = read_stamp(text)
    if stamp_record is None:
        return [f"{out} has no report stamp: it was not generated by mrm.validation.report"]
    preset = preset or stamp_record["preset"]
    data = load_data(preset)
    problems = []
    rendered_text = render(data, preset=preset, g25_preset=stamp_record.get("g25_preset"), stamp=stamp_record,
                           doc_text=text, charts=False, chart_dir=data_dir(preset))
    if rendered_text != text:
        stored_lines, rendered_lines = text.splitlines(), rendered_text.splitlines()
        first_difference = next(
            (k for k in range(min(len(stored_lines), len(rendered_lines))) if stored_lines[k] != rendered_lines[k]),
            min(len(stored_lines), len(rendered_lines)))
        problems.append(
            f"{out} differs from a re-render at line {first_difference + 1}: run python -m mrm.validation.report --fast "
            f"--preset {preset}")
    for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", text):
        if not (out.parent / m.group(1)).exists():
            problems.append(f"missing chart {m.group(1)}")
    problems += _staleness(data, {"G26": preset, "G25": companion_g25(preset, stamp_record.get("g25_preset"))})
    return problems


# run parsed report arguments for both `python -m` and `mrm report`.
def run_args(a: argparse.Namespace) -> int:
    if a.check:
        probs = check(a.preset if a.preset != "g26" else None, a.out)
        for p in probs:
            print(p, file=sys.stderr)
        return 1 if probs else 0
    parts = () if a.fast else (None if a.recompute is None
                               else tuple(s for s in (x.strip() for x in a.recompute.split(",")) if s))
    try:
        generate(a.preset, g25_preset=a.g25_preset, recompute=parts, out=a.out, n_base=a.sobol_n,
                 compare_worlds=a.compare_worlds, compare_run=a.compare_run, n_threads=a.threads)
    except Exception as e:                                              # noqa: BLE001
        print(f"mrm report: {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1
    return 0


# parse `argv`, run the report command and return its exit code.
def main(argv: Sequence[str] | None = None) -> int:
    return run_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
