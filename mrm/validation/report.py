"""Generated validation report ``docs/VALIDATION.md`` (07 P1-23 and section 4.9).

``python -m mrm.validation.report [--preset g26]`` (or ``mrm report``) regenerates the report with the fixed
sections of 07 4.9:

1. the target table (id, role, measured +- sigma, predicted nominal and band, z-score, tolerance, pass/fail) of the
   calibration, validation and bounds targets, from the replicas (``mrm.validation.replicas``);
2. the closure and conflict table of 07 4.6, with the ``xfail(strict=True)`` closure tests that link to it;
3. identifiability: the Jacobian of the calibration replicas, its SVD, the linearized half-prior rule, and the
   profile likelihoods of the calibration task when they exist (a hook otherwise);
4. uncertainty propagation over the 07 4.8 set: one tornado chart per controller metric and Sobol indices for the
   top 8 entries (``mrm.validation.sensitivity``; SALib is not installed);
5. controller rank stability across worlds (Kendall tau; ``mrm.sim.experiment`` kind compare);
6. the generation mix of each derived quantity (G25 vs G26 provenance);
7. the textbook disagreements, from the ``disagrees:`` fields of the preset.

The frozen-targets block at the top of the existing ``docs/VALIDATION.md`` (owned by the target split) is copied
byte for byte; the report never writes it.

Data flow. The expensive parts (replica predictions over the nominal world and the 12 named corners, the
calibration Jacobian, the tornado / Sobol worlds and the controller compare) are computed into
``docs/validation/<preset>/report_data.json``; the Markdown and the charts (``docs/validation/<preset>/*.svg``) are
rendered from that file plus the cheap live parts (target files, frozen header, preset leaves, derive()). ``--fast``
re-renders without recomputing (the CI check ``tests/verification/test_report.py`` does this and compares with the
committed file); ``--recompute a,b`` recomputes only the named parts. Every computed part records when, at which
commit, on which presets and at what cost it was computed, and the report prints that.

Nothing here fits a parameter: the Jacobian is evaluated at the preset's values, on calibration targets only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import re
import resource
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from mrm.validation import sensitivity as sens
from mrm.validation import targets_io as tio

__all__ = ["SCHEMA", "PARTS", "DOC_PATH", "DATA_ROOT", "CAL_PARAMETERS", "SENS_METRICS", "companion_g25",
           "data_path", "load_data", "save_data", "compute_predictions", "compute_identifiability",
           "compute_sensitivity", "compute_ranking", "render", "generate", "read_stamp", "build_parser", "main"]

SCHEMA = "mrm.validation.report/1"
REPO = tio.REPO_DIR
DOC_PATH = tio.DOC_VALIDATION
DATA_ROOT = REPO / "docs" / "validation"
PRESETS_DIR = REPO / "mrm" / "params" / "presets"
COMPARE_FILE = REPO / "experiments" / "examples" / "compare_locks.yaml"
#: computed (cached) parts of the report data
PARTS = ("predictions", "identifiability", "sensitivity", "ranking")
G25_DEFAULT = "g25_test_vehicle"
STAMP_RE = re.compile(r"<!-- report-stamp (\{.*?\}) -->")

#: replicas whose targets are scenario calibrations (not plant parameters): left out of the plant Jacobian
SCENARIO_REPLICAS = ("laser_line_statistics", "aggressor_trajectory")

#: calibration candidates (07 4.3 "what it may identify"): config path -> the calibration ids it is meant to explain.
#: The prior range is the leaf's ``range``.
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

#: controller metrics of section 4: (world_summary key, unit, transform for the Sobol indices)
SENS_METRICS: tuple[tuple[str, str, str | None], ...] = (
    ("oma_penalty_db_worst", "dB", None),
    ("excursion_count", "", "log10(1+x)"),
    ("excursion_max_s", "s", None),
    ("wrong_line_rings", "rings", None),
    ("unlocked_rings", "rings", None),
    ("mean_reloma_db", "dB", None),
)
SENS_CONTROLLER = "pi_lock_c"
SENS_TOP = 8

#: fixed flags of the replicas handoff and the target split (docs/handoff/replicas.md 5, validation_targets.md 4)
TARGET_FLAGS: dict[str, str] = {
    "CAL-3a": "R2 anchors P_FSR: close to the target by construction (replicas 4.6)",
    "CAL-3b": "R2 anchors V_FSR: close to the target by construction",
    "CAL-7": "R3 sets the cold offset: close to the target by construction",
    "CAL-13a": "scenario calibration residual (small by construction), not a prediction",
    "CAL-13b": "scenario calibration residual (small by construction), not a prediction",
    "CAL-14": "relOMA = one MRM's through port over the power into it (07 D9); mrm.control.models reports OMA at "
              "the bus output: 0.35 dB lower (replicas 4.7)",
    "VAL-3b": "replica method changed after the comparison was visible (replicas handoff 5): treat a pass as weak",
    "VAL-4c": "not independent of VAL-4a/b (the map maximum is the CAL-14 level): not counted as a third "
              "independent pass; shares its printed spread with BND-4",
    "BND-4": "shares the printed '< 1 dB' spread with VAL-4c",
    "VAL-7a": "unverified: n_g 2.73 not re-read from de Cea 2019 (reported, not scored)",
    "VAL-7b": "definition conflict: energy fraction (primary) vs Robinson dn_eff/dn_Si (variant) (replicas 5)",
    "VAL-6": "deferred until the DAC law and full scale are known (07 Q3); its data are CAL-9",
    "VAL-8e": "cross-check (solver against imec's FE fit), never counted as validation",
    "VAL-8a": "a match of the interim thermal model with VAL-8 would be tautological; the replica reads the FEM "
              "extraction artefact",
    "BND-1": "a soft_fail (1 to 2 FWHM) is marginal, not a pass",
}

#: 07 4.6 closure and conflict table (evidence transcribed from 07; the model column is computed per preset)
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
    ("rom-dc-scale", "rom_dc_scale", "reported per dlambda/dT hypothesis (07 P1-2)"),
)


# ================================================================================================ small helpers
def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git() -> dict:
    from mrm.sim.manifest import git_state
    g = git_state(REPO)
    return {"commit": (g.get("commit") or "")[:12] or None, "dirty": g.get("dirty"),
            "diff_sha256": (g.get("diff_sha256") or "")[:16] or None}


def _peak_rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2), 3)


def _enc(x: Any) -> Any:
    """JSON form that keeps NaN and inf ('nan', 'inf', '-inf')."""
    if isinstance(x, Mapping):
        return {str(k): _enc(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_enc(v) for v in x]
    if isinstance(x, np.ndarray):
        return _enc(x.tolist())
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


def _dec(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {k: _dec(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_dec(v) for v in x]
    if isinstance(x, str) and x in ("nan", "inf", "-inf"):
        return float(x)
    return x


def _num(x: float, sig: int = 4) -> str:
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


def _fmt(x: Any, max_items: int = 8, sig: int = 4) -> str:
    """Compact text of a value (scalar, list, mapping) for a table cell."""
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, Mapping):
        if "t_s" in x and "y" in x:
            return f"<trajectory, {len(x['y'])} samples>"
        items = [f"{k}: {_fmt(v, max_items, sig)}" for k, v in list(x.items())[:max_items]]
        return "{" + ", ".join(items) + (", ..." if len(x) > max_items else "") + "}"
    if isinstance(x, (list, tuple, np.ndarray)):
        flat = list(np.asarray(x, dtype=object).ravel()) if not (len(x) and isinstance(x[0], Mapping)) else list(x)
        s = ", ".join(_fmt(v, max_items, sig) for v in flat[:max_items])
        return "[" + s + (", ..." if len(flat) > max_items else "") + "]"
    return _num(x, sig)


def _md(s: Any) -> str:
    return "" if s is None else str(s).replace("|", "\\|").replace("\n", " ")


def _flatten(x: Any) -> np.ndarray:
    """Numbers of a prediction or value in a stable order (mappings by sorted key; None -> [nan])."""
    if x is None:
        return np.array([np.nan])
    if isinstance(x, Mapping):
        parts = [_flatten(x[k]) for k in sorted(x)]
        return np.concatenate(parts) if parts else np.zeros(0)
    try:
        return np.asarray(x, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.array([np.nan])


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


def _load_cfg(name: str, *, alts: Mapping | None = None, overrides: Mapping | None = None):
    from mrm import params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load(name, overrides=dict(overrides) if overrides else None,
                           alts=dict(alts) if alts else None)


def companion_g25(preset: str, g25: str | None = None) -> str:
    """The G25 bench preset used with ``preset``: ``g25`` when given; ``g25_test_vehicle`` for g26; for another
    preset, the same name with g26 replaced by g25_test_vehicle when such a preset exists (for example
    ``g25_test_vehicle_calibrated`` for ``g26_calibrated``), else ``g25_test_vehicle``."""
    if g25:
        return g25
    if preset == "g26":
        return G25_DEFAULT
    cand = preset.replace("g26", G25_DEFAULT, 1)
    if cand != preset and (PRESETS_DIR / f"{cand}.yaml").exists():
        return cand
    return G25_DEFAULT


def _named_corners() -> tuple[str, ...]:
    from mrm.sim.experiment import NAMED_CORNERS
    return tuple(NAMED_CORNERS)


def _corner_alts(name: str) -> dict:
    """The ``alts`` of a corner preset file (corner presets select alternatives only, 07 4.8)."""
    import yaml
    doc = yaml.safe_load((PRESETS_DIR / f"{name}.yaml").read_text()) or {}
    extra = set(doc) - {"extends", "meta", "alts"}
    if extra:
        raise ValueError(f"corner preset {name} sets {sorted(extra)}: only alts are supported in the report band")
    return {str(k): str(v) for k, v in (doc.get("alts") or {}).items()}


def _target_files_sha() -> dict:
    return {r: tio.file_sha256(r) for r in tio.TARGET_ROLES}


#: source files whose change makes a cached part stale (the protocol code the part's numbers come from; the
#: engine is held bit-identical by tests/verification/test_kernel_parity.py and is not listed)
CODE_DEPS: dict[str, tuple[str, ...]] = {
    "predictions": ("mrm/validation/replicas.py",),
    "identifiability": ("mrm/validation/replicas.py",),
    "sensitivity": ("mrm/validation/sensitivity.py",),
    "ranking": (),
}


def _code_sha(part: str) -> dict:
    import hashlib
    return {f: hashlib.sha256((REPO / f).read_bytes()).hexdigest()[:16] for f in CODE_DEPS.get(part, ())}


def _meta(presets: Mapping[str, str], t0: float, **extra) -> dict:
    cfg_hash = {}
    for k, name in presets.items():
        try:
            cfg_hash[name] = _load_cfg(name).hash()[:16]
        except Exception as e:                                          # noqa: BLE001
            cfg_hash[name] = f"error: {e.__class__.__name__}"
    m = {"computed_utc": _now(), "git": _git(), "presets": dict(presets), "config_hash": cfg_hash,
         "targets_sha256": _target_files_sha(), "wall_s": round(time.perf_counter() - t0, 1),
         "peak_rss_gb": _peak_rss_gb()}
    m.update(extra)
    return m


def _log_default(s: str) -> None:
    print(s, file=sys.stderr, flush=True)


# ================================================================================================ data file
def data_dir(preset: str) -> Path:
    return DATA_ROOT / preset


def data_path(preset: str) -> Path:
    return data_dir(preset) / "report_data.json"


def load_data(preset: str) -> dict:
    """The cached report data of ``preset`` ({} parts when absent)."""
    p = data_path(preset)
    if not p.exists():
        return {"schema": SCHEMA, "preset": preset}
    d = _dec(json.loads(p.read_text()))
    if d.get("schema") != SCHEMA:
        raise ValueError(f"{p}: schema {d.get('schema')!r}, expected {SCHEMA}")
    return d


def save_data(preset: str, data: Mapping) -> Path:
    p = data_path(preset)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_enc(dict(data)), separators=(",", ":"), allow_nan=False) + "\n")
    return p


# ================================================================================================ 1. predictions
def _record(t, res) -> dict:
    from mrm.validation.replicas import _nonfinite
    fb = (res.conditions or {}).get("artefact_fallbacks") if isinstance(res.conditions, Mapping) else None
    return {"id": t.id, "replica_status": res.status, "predicted": res.value,
            "predicted_nonfinite": bool(res.status == "ok" and _nonfinite(res.value)),
            "variants": res.variants or {}, "notes": [str(n) for n in (res.notes or [])],
            "artefact_fallbacks": fb}


def _plant_key(t) -> str:
    from mrm.validation.replicas import preset_for
    return "G25" if preset_for(t) == G25_DEFAULT else "G26"


def compute_predictions(presets: Mapping[str, str], *, corners: Sequence[str] | None = None,
                        log: Callable[[str], None] = _log_default) -> dict:
    """Replica predictions of every calibration, validation and bounds target on the nominal world of
    ``presets`` ({"G26": name, "G25": name}) and on each named corner (the corner's alternatives applied to the
    same presets). Never raises for a replica: blocked / error results carry their reason."""
    from mrm.validation import replicas as rp
    t0 = time.perf_counter()
    corners = _named_corners() if corners is None else tuple(corners)
    targets = [t for role in tio.TARGET_ROLES for t in tio.load_targets(role)]
    recs = []
    for t in targets:
        t1 = time.perf_counter()
        res = rp.run_target(t, presets[_plant_key(t)])
        r = _record(t, res)
        r["runtime_s"] = round(time.perf_counter() - t1, 2)
        recs.append(r)
        log(f"  {t.id:8s} {res.status:8s} {r['runtime_s']:6.1f} s")
    cor: dict[str, dict] = {}
    alts_of: dict[str, dict] = {}
    for cn in corners:
        alts = _corner_alts(cn)
        alts_of[cn] = alts
        t1 = time.perf_counter()
        cfgs: dict[str, Any] = {}
        for k, name in presets.items():
            try:
                cfgs[k] = rp.load_config(name, alts=alts)
            except Exception as e:                                      # noqa: BLE001
                cfgs[k] = e
        out = {}
        for t in targets:
            c = cfgs[_plant_key(t)]
            if isinstance(c, Exception):
                out[t.id] = {"status": "error", "predicted": None, "note": f"{c.__class__.__name__}: {c}"}
                continue
            res = rp.run_target(t, c)
            out[t.id] = {"status": res.status, "predicted": res.value,
                         "note": (res.notes or [""])[0] if res.status != "ok" else ""}
        cor[cn] = out
        log(f"  corner {cn}: {time.perf_counter() - t1:.1f} s")
    return {"meta": _meta(presets, t0, corners=list(corners), code_sha256=_code_sha("predictions")),
            "corner_alts": alts_of, "records": recs,
            "corners": cor}


# ================================================================================================ 3. identifiability
def _scales(t, pred: Any) -> np.ndarray:
    """One scale per flattened prediction element: the target sigma (per point / key when given), else a quarter of
    an interval tolerance, else half an abs / rel tolerance."""
    n = _flatten(pred).size
    sig = t.sigma
    if isinstance(sig, Mapping) and isinstance(pred, Mapping):
        out = []
        for k in sorted(pred):
            m = _flatten(pred[k]).size
            s = sig.get(k)
            out += [float(s) if isinstance(s, (int, float)) else np.nan] * m
        s_arr = np.asarray(out, float)
    elif isinstance(sig, (list, tuple)) and len(sig) == n:
        s_arr = np.asarray(sig, float)
    elif isinstance(sig, (int, float)) and not isinstance(sig, bool):
        s_arr = np.full(n, float(sig))
    else:
        s_arr = np.full(n, np.nan)
    if np.any(~np.isfinite(s_arr) | (s_arr <= 0)):
        tol = t.tolerance
        k = tol.get("kind")
        v = _flatten(t.value)
        if k == "interval":
            fb = (float(tol["hi"]) - float(tol["lo"])) / 4.0
        elif k == "abs":
            fb = float(tol["value"]) / 2.0
        elif k == "rel":
            fb = float(tol["value"]) * float(np.nanmax(np.abs(v))) / 2.0
        else:
            fb = np.nan
        s_arr = np.where(np.isfinite(s_arr) & (s_arr > 0), s_arr, fb)
    return s_arr


def _jacobian_targets() -> list:
    return [t for t in tio.load_calibration_targets() if t.scored and t.replica not in SCENARIO_REPLICAS
            and t.raw.get("fit", True) is not False]


def compute_identifiability(presets: Mapping[str, str], *, params: Sequence[tuple[str, str]] = CAL_PARAMETERS,
                            rel_step: float = 0.05, log: Callable[[str], None] = _log_default) -> dict:
    """Central-difference Jacobian of the calibration replicas at the presets' values, normalized to residual
    sigmas per prior half-width (steps of ``rel_step`` of the prior range, one-sided at a range end), and its SVD
    (:func:`mrm.validation.sensitivity.identifiability`)."""
    from mrm.validation import replicas as rp
    t0 = time.perf_counter()
    targets = _jacobian_targets()
    base_cfg = {k: _load_cfg(n) for k, n in presets.items()}

    def predict(cfgs) -> list:
        out = []
        for t in targets:
            c = cfgs[_plant_key(t)]
            res = rp.run_target(t, c) if not isinstance(c, Exception) else None
            out.append(res.value if res is not None and res.status == "ok" else None)
        return out

    y0 = predict(base_cfg)

    def predict_on(cfgs, perturbed) -> list:
        """Replica outputs with the presets in ``perturbed`` replaced; targets on the other plant keep y0."""
        out = []
        for t, y in zip(targets, y0):
            k = _plant_key(t)
            if k not in perturbed:
                out.append(y)
                continue
            res = rp.run_target(t, cfgs[k]) if not isinstance(cfgs[k], Exception) else None
            out.append(res.value if res is not None and res.status == "ok" else None)
        return out
    rows, scales, labels, tvals = [], [], [], []
    for t, p in zip(targets, y0):
        f = _flatten(p)
        rows.append(f)
        scales.append(_scales(t, p))
        labels += [f"{t.id}[{k}]" if len(f) > 1 else t.id for k in _flat_keys(p)][:f.size]
    Y0 = np.concatenate(rows)
    S = np.concatenate(scales)
    cols, pinfo = [], []
    for path, why in params:
        g26 = base_cfg["G26"]
        lf = g26.leaves.get(path)
        info = {"path": path, "identifies": why}
        if lf is None or lf.range is None or lf.is_free:
            info["skipped"] = "not a leaf with a prior range in this preset"
            pinfo.append(info)
            cols.append(np.full(Y0.size, np.nan))
            continue
        lo, hi = float(lf.range[0]), float(lf.range[1])
        h = rel_step * (hi - lo)
        info.update(prior=[lo, hi], provenance=lf.provenance, targets=lf.targets, residual=lf.residual,
                    identifiable=lf.identifiable)
        sides = {}
        try:
            g0 = np.asarray(lf.value, float).ravel()
            if g0.size != 1:
                raise ValueError(f"{path} is not a scalar leaf in {presets['G26']}")
            # The same physical parameter is moved on every plant whose leaf has the G26 value; a plant whose
            # leaf differs (a plant-specific leaf, e.g. the G25 bench's own optical path) keeps its value, so its
            # targets have a zero column (the prior range and step are G26's).
            moved, kept = ["G26"], {}
            for k, name in presets.items():
                if k == "G26":
                    continue
                lk = base_cfg[k].leaves.get(path)
                vk = None if lk is None or lk.is_free else np.asarray(lk.value, float).ravel()
                if vk is not None and vk.size == 1 and float(vk[0]) == float(g0[0]):
                    moved.append(k)
                else:
                    kept[k] = (f"{name} has {_fmt(None if lk is None else lk.value)} (G26 {_num(float(g0[0]))}): "
                               "not moved")
            info["moved_on"] = moved
            if kept:
                info["kept"] = kept
            for sgn in (-1, 1):
                v = float(np.clip(g0[0] + sgn * h, lo, hi))
                vals = {k: v for k in moved}
                cfgs = {k: _load_cfg(presets[k], overrides={path: v}) for k in moved}
                sides[sgn] = (vals, predict_on(cfgs, moved))
        except Exception as e:                                          # noqa: BLE001
            info["skipped"] = f"{e.__class__.__name__}: {e}"
            pinfo.append(info)
            cols.append(np.full(Y0.size, np.nan))
            continue
        dv = sides[1][0]["G26"] - sides[-1][0]["G26"]
        info["value"] = float(np.asarray(g26.leaves[path].value, float).ravel()[0])
        info["step"] = [sides[-1][0]["G26"], sides[1][0]["G26"]]
        yp = np.concatenate([_flatten(p) if _flatten(p).size == r.size else np.full(r.size, np.nan)
                             for p, r in zip(sides[1][1], rows)])
        ym = np.concatenate([_flatten(p) if _flatten(p).size == r.size else np.full(r.size, np.nan)
                             for p, r in zip(sides[-1][1], rows)])
        cols.append((yp - ym) / dv * (0.5 * (hi - lo)) / S)
        pinfo.append(info)
        log(f"  {path}: {time.perf_counter() - t0:.0f} s")
    J = np.stack(cols, axis=1)
    names = [p for p, _ in params]
    ok = [i for i, inf in enumerate(pinfo) if "skipped" not in inf]
    ident = sens.identifiability(J[:, ok], [names[i] for i in ok]) if ok else None
    tv = []
    for t in targets:
        tv.append({"id": t.id, "replica": t.replica, "value": t.value, "sigma": t.sigma})
    resid = []
    for t, p in zip(targets, y0):
        ev = tio.evaluate(t, p) if p is not None else None
        resid.append({"id": t.id, "level": ev.level if ev else "blocked", "z": ev.z if ev else None})
    return {"meta": _meta(presets, t0, rel_step=rel_step, n_evaluations=1 + 2 * len(ok),
                          code_sha256=_code_sha("identifiability")),
            "targets": tv, "residual_labels": labels, "scales": S, "base_prediction": Y0, "jacobian": J,
            "params": pinfo, "svd": ident, "base_evaluation": resid}


# ================================================================================================ 4. sensitivity
def _sens_experiment(preset: str, B: int) -> str:
    """YAML of the section-4 world evaluation: pi_lock_c under A1 (one aggressor period after 5 ms of start-up),
    warm start, self reference, metrics only (the compare_locks protocol for one controller)."""
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
        "  seeds: {device: 100, uncertainty: 200, noise: 300, scenario: 400}",
        "  uncertainty: lhs",
        '  variation: "off"',
        "  start: warm",
        "  score_from_s: 0.005",
        "  self_reference: {duration_s: 0.02, window_s: 0.005}",
        ""])


def evaluate_worlds(preset: str, U: np.ndarray, *, chunk: int = 256, n_threads: int = 4,
                    log: Callable[[str], None] = _log_default) -> dict[str, np.ndarray]:
    """World summaries (:func:`mrm.sim.experiment.world_summary`) of the section-4 protocol on the worlds of the
    u-design ``U`` (B, D), in chunks of at most ``chunk`` worlds per Plant. Each chunk's design is checked
    against realize's draws before it runs (:func:`mrm.validation.sensitivity.check_injection`)."""
    from mrm.sim import experiment as ex
    space = sens.UncertaintySpace(_load_cfg(preset))
    out: dict[str, list] = {}
    with tempfile.TemporaryDirectory(prefix="mrm_report_") as td:
        for i0 in range(0, U.shape[0], chunk):
            Uc = U[i0:i0 + chunk]
            chk = sens.check_injection(space, Uc)
            if not chk["ok"]:
                raise RuntimeError(f"injected design does not reach realize: {chk['mismatches'][:3]}")
            f = Path(td) / f"sens_{i0}.yaml"
            f.write_text(_sens_experiment(preset, Uc.shape[0]))
            t1 = time.perf_counter()
            with sens.injected_lhs(Uc):
                res = ex.run(str(f), write=False, log=lambda s: None,
                             options={"n_threads": int(n_threads), "max_worlds_per_plant": max(512, Uc.shape[0])})
            ws = res.summary["world_summary"]
            for k, v in ws.items():
                out.setdefault(k, []).append(np.asarray(v, float))
            log(f"  worlds {i0}..{i0 + Uc.shape[0] - 1}: {time.perf_counter() - t1:.0f} s")
    return {k: np.concatenate(v) for k, v in out.items()}


def _transform(x: np.ndarray, tr: str | None) -> np.ndarray:
    if tr == "log10(1+x)":
        return np.log10(1.0 + np.asarray(x, float))
    return np.asarray(x, float)


def compute_sensitivity(preset: str, *, n_base: int = 128, top: int = SENS_TOP, seed: int = 20261010,
                        chunk: int = 256, n_threads: int = 4, n_boot: int = 500, n_rep: int = 8,
                        log: Callable[[str], None] = _log_default) -> dict:
    """Section 4: one-at-a-time tornado over every free ``uncertainty`` entry (others at nominal), screening of the
    top ``top`` entries by normalized swing, then Saltelli/Jansen Sobol indices over those entries (u-space of
    their prior; the other entries at nominal), ``n_base`` base rows, (top + 2) n_base worlds, plus the binned
    first-order estimate from the 2 n_base independent A and B rows.

    ``n_rep`` extra copies of the nominal world (appended after the tornado rows, so they run with other noise
    seeds: world b uses noise seed base + b) measure the noise variance of each metric, which every Sobol index
    carries (a Saltelli row and its AB_i partner are different worlds, so different noise). The raw world values
    are stored so the indices can be re-derived."""
    t0 = time.perf_counter()
    space = sens.UncertaintySpace(_load_cfg(preset))
    u0 = space.nominal_u()
    labels, rows = [("nominal", "nominal")], [u0]
    for i, n in enumerate(space.names):
        if space.axes[n].pinned is not None:
            continue
        for lab, u in space.axes[n].tornado_points():
            r = u0.copy()
            r[i] = u
            rows.append(r)
            labels.append((n, lab))
    n_torn = len(rows)
    rows += [u0] * int(n_rep)                          # nominal replicates: noise only (other world index)
    U1 = np.vstack(rows)
    log(f"tornado: {n_torn} worlds + {n_rep} nominal replicates")
    W1 = evaluate_worlds(preset, U1, chunk=chunk, n_threads=n_threads, log=log)
    metrics = [m for m, _, _ in SENS_METRICS]
    base = {m: float(W1[m][0]) for m in metrics}
    pts = []
    for j, (entry, lab) in enumerate(labels[1:], start=1):
        desc = lab if space.axes[entry].kind == "hypothesis" else f"{lab} ({space.describe(entry, U1[j, space.names.index(entry)])})"
        pts.append((entry, desc, {m: float(W1[m][j]) for m in metrics}))
    torn = sens.tornado_table(base, pts, metrics)
    top_entries = sens.screen_top(torn, top)
    log(f"Sobol over {top_entries}")
    A, B = sens.sobol_design(len(top_entries), n_base, seed=seed)
    X = sens.saltelli_stack(A, B)
    U2 = np.tile(u0[None, :], (X.shape[0], 1))
    for c, n in enumerate(top_entries):
        U2[:, space.names.index(n)] = X[:, c]
    t2 = time.perf_counter()
    W2 = evaluate_worlds(preset, U2, chunk=chunk, n_threads=n_threads, log=log)
    sob = {}
    d = len(top_entries)
    XAB = np.vstack([A, B])
    edges = [space.axes[n].cdf if space.axes[n].kind == "hypothesis" else None for n in top_entries]
    for m, unit, tr in SENS_METRICS:
        y = _transform(W2[m], tr)
        fA, fB, fAB = sens.split_stack(y, n_base, d)
        good = np.isfinite(fA) & np.isfinite(fB) & np.all(np.isfinite(fAB), axis=0)
        entry = {"transform": tr, "n_rows_dropped_nonfinite": int((~good).sum())}
        if good.sum() >= 8:
            r = sens.sobol_indices(fA[good], fB[good], fAB[:, good], names=top_entries, n_boot=n_boot, seed=seed)
            entry.update(r.to_dict())
            yab = np.r_[fA, fB]
            ok = np.isfinite(yab)
            g = sens.given_data_first_order(XAB[ok], yab[ok], edges=edges, n_boot=n_boot, seed=seed)
            entry["s1_binned"] = [float(x) for x in g["s1"]]
            entry["s1_binned_ci"] = [[float(a), float(b)] for a, b in g["s1_ci"]]
            entry["s1_binned_n_bins"] = list(g["n_bins"])
            rep_y = _transform(W1[m][[0] + list(range(n_torn, n_torn + int(n_rep)))], tr)
            nv = float(np.var(rep_y, ddof=1)) if n_rep > 0 and np.all(np.isfinite(rep_y)) else float("nan")
            entry["noise_variance"] = nv
            entry["noise_share"] = nv / float(entry["variance"]) if entry["variance"] > 0 else float("nan")
        else:
            entry["notes"] = ["fewer than 8 finite base rows: not computed"]
        sob[m] = entry
    sup = W2.get("warm_start_supply_limited")
    return {"meta": _meta({"G26": preset}, t0, controller=SENS_CONTROLLER, scenario="a1 (one 130.7 ms period, "
                          "5 ms start-up)", n_tornado_worlds=int(n_torn), n_sobol_worlds=int(U2.shape[0]),
                          n_base=int(n_base), seed=int(seed), chunk=int(chunk), n_threads=int(n_threads),
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
            "base": base, "base_world": {k: float(v[0]) for k, v in W1.items()},
            "tornado_points": [{"entry": e, "label": lab, "values": v} for e, lab, v in pts],
            "noise_replicates": {k: [float(x) for x in np.asarray(v, float)[[0] + list(range(n_torn, n_torn + n_rep))]]
                                 for k, v in W1.items()},
            "sobol_worlds": {k: [float(x) for x in np.asarray(v, float)] for k, v in W2.items()
                             if k in {m for m, _, _ in SENS_METRICS} | {"warm_start_supply_limited"}},
            "tornado": torn, "top": top_entries, "sobol": sob,
            "sobol_supply_limited_frac": float(np.mean(sup)) if sup is not None else None,
            "tornado_supply_limited": {labels[j][0] + ":" + labels[j][1]: float(W1["warm_start_supply_limited"][j])
                                       for j in range(len(labels))
                                       if "warm_start_supply_limited" in W1
                                       and W1["warm_start_supply_limited"][j] > 0}}


# ================================================================================================ 5. ranking
def compute_ranking(preset: str, *, n_mc: int = 256, compare_run: str | Path | None = None, n_threads: int = 4,
                    log: Callable[[str], None] = _log_default) -> dict:
    """Section 5 from ``mrm.sim.experiment`` kind compare: experiments/examples/compare_locks.yaml re-based on
    ``preset`` with ``n_mc`` Latin-hypercube worlds, or the summary of an existing compare run directory."""
    t0 = time.perf_counter()
    if compare_run is not None:
        d = Path(compare_run)
        m = json.loads((d / "metrics.json").read_text())
        man = json.loads((d / "manifest.json").read_text())
        if m.get("kind") != "compare":
            raise ValueError(f"{d} is not a compare run")
        files = [str(x) for x in (man.get("files") or [])]
        summary = m["summary"]
        src = {"run_dir": str(d), "run_id": man.get("run_id"), "experiment_hash": man.get("experiment_hash"),
               "run_git": {"commit": (man.get("git") or {}).get("commit"), "dirty": (man.get("git") or {}).get(
                   "dirty")}, "run_wall_s": man.get("wall_s"), "files": files}
    else:
        import yaml
        from mrm.sim import experiment as ex
        doc = yaml.safe_load(COMPARE_FILE.read_text())
        doc["extends"] = preset
        doc.setdefault("meta", {})["name"] = f"report_compare_{preset}"
        doc["experiment"]["n_mc"] = int(n_mc)
        with tempfile.TemporaryDirectory(prefix="mrm_report_") as td:
            f = Path(td) / "compare.yaml"
            f.write_text(yaml.safe_dump(doc, sort_keys=False))
            res = ex.run(str(f), write=False, log=log, options={"n_threads": int(n_threads)})
        summary = res.summary
        src = {"experiment": "experiments/examples/compare_locks.yaml re-based on the preset", "n_mc": int(n_mc),
               "experiment_hash": res.experiment_hash[:16]}
    ctrl = {}
    for lab, d in summary.get("controllers", {}).items():
        ctrl[lab] = {"nominal": d.get("nominal"), "worst_corner_overall": d.get("worst_corner_overall"),
                     "worst_corner": {k: v for k, v in (d.get("worst_corner") or {}).items()},
                     "quantiles": {k: v for k, v in (d.get("quantiles") or {}).items()}}
    return {"meta": _meta({"G26": preset}, t0, source=src), "rank_metrics": summary.get("rank_metrics"),
            "rank_stability": summary.get("rank_stability"), "controllers": ctrl}


# ================================================================================================ charts
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "mrm-validation-report", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    return plt


C_BLUE, C_ORANGE, C_GREY, C_INK = "#2a78d6", "#eb6834", "#a8a7a0", "#0b0b0b"


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _chart_z(rows: list[tuple[str, float, str]], path: Path) -> None:
    plt = _mpl()
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
    _save(fig, path)


def _chart_svd(s: Sequence[float], thr: float, path: Path) -> None:
    plt = _mpl()
    s = np.asarray(s, float)
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    k = np.arange(1, s.size + 1)
    ax.bar(k, np.maximum(s, 1e-6), color=[C_BLUE if v >= thr else C_ORANGE for v in s], width=0.6)
    ax.axhline(thr, color=C_INK, lw=0.8, ls="--")
    ax.text(s.size + 0.4, thr, f" half-prior rule\n s >= {thr:.2f}", va="center", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(k)
    ax.set_xlabel("SVD direction")
    ax.set_ylabel("singular value (sigmas per prior half-width)")
    _save(fig, path)


def _chart_tornado(rows: list[dict], base: float, title: str, path: Path, n: int = 12) -> None:
    plt = _mpl()
    rows = rows[:n]
    fig, ax = plt.subplots(figsize=(6.0, 0.3 * len(rows) + 0.9))
    y = np.arange(len(rows))
    fin = [v for r in rows for v in (r["lo"], r["hi"]) if np.isfinite(v)] + ([base] if np.isfinite(base) else [])
    span = (max(fin) - min(fin)) if fin else 1.0
    cap_hi = (max(fin) + 0.15 * span) if fin else 1.0
    cap_lo = (min(fin) - 0.15 * span) if fin else 0.0
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
    _save(fig, path)


def _chart_sobol(sob: Mapping[str, Mapping], names: Sequence[str], path: Path) -> None:
    plt = _mpl()
    ms = [m for m in sob if "st" in sob[m]]
    if not ms:
        return
    ncol = min(3, len(ms))
    nrow = int(math.ceil(len(ms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol + 1.6, (0.3 * len(names) + 1.0) * nrow), sharey=True,
                             squeeze=False)
    for ax in axes.ravel()[len(ms):]:
        ax.set_visible(False)
    axes = axes.ravel()
    y = np.arange(len(names))
    for ax, m in zip(axes, ms):
        e = sob[m]
        st = np.array([np.nan if v is None else v for v in e["st"]], float)
        s1 = np.array([np.nan if v is None else v for v in e.get("s1_binned", e["s1"])], float)
        ax.barh(y + 0.18, np.clip(st, 0, None), height=0.34, color=C_ORANGE, label="total ST")
        ax.barh(y - 0.18, np.clip(s1, 0, None), height=0.34, color=C_BLUE,
                label="first order S1" + (" (binned)" if "s1_binned" in e else ""))
        ci = np.array([[np.nan if v is None else v for v in r] for r in e["st_ci"]], float)
        ax.errorbar(np.clip(st, 0, None), y + 0.18, xerr=np.abs(np.vstack([st - ci[:, 0], ci[:, 1] - st])),
                    fmt="none", ecolor=C_INK, lw=0.6)
        ax.set_title(m + (f"\n({e['transform']})" if e.get("transform") else ""), fontsize=7)
        ax.set_xlim(0, 1.05)
    for r in range(nrow):
        axes[r * ncol].set_yticks(y, list(names), fontsize=7)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=6.5, loc="lower right", frameon=False)
    fig.tight_layout()
    _save(fig, path)


# ================================================================================================ render
def _rel(path: Path) -> str:
    import os
    return Path(os.path.relpath(Path(path).resolve(), (REPO / "docs").resolve())).as_posix()


def _frozen_block(doc_text: str) -> str:
    a = doc_text.find(tio.FROZEN_BEGIN)
    b = doc_text.find(tio.FROZEN_END)
    if a < 0 or b < 0 or b < a:
        raise ValueError("docs/VALIDATION.md has no frozen-targets block: restore it from git (the report never "
                         "writes it; mrm/validation/targets/README.md)")
    return doc_text[a:b + len(tio.FROZEN_END)]


def read_stamp(text: str) -> dict | None:
    """The ``report-stamp`` of a rendered report ({preset, g25_preset, date, commit, dirty})."""
    m = STAMP_RE.search(text)
    return json.loads(m.group(1)) if m else None


def _part_line(name: str, part: Mapping | None) -> str:
    if not part:
        return f"| {name} | not computed | | | | |"
    m = part.get("meta", {})
    g = m.get("git") or {}
    return (f"| {name} | {m.get('computed_utc', '')} | `{g.get('commit')}`{' (dirty)' if g.get('dirty') else ''} | "
            f"{_num(m.get('wall_s'))} | {_num(m.get('peak_rss_gb'))} | "
            f"{', '.join(f'{k} `{v}`' for k, v in (m.get('config_hash') or {}).items())} |")


def _staleness(data: Mapping, presets: Mapping[str, str]) -> list[str]:
    """Reasons why cached parts no longer match the tree (config hashes of the presets, target files)."""
    out = []
    cur_t = _target_files_sha()
    cur_h = {}
    for name in set(presets.values()):
        try:
            cur_h[name] = _load_cfg(name).hash()[:16]
        except Exception as e:                                          # noqa: BLE001
            cur_h[name] = f"error: {e.__class__.__name__}"
    for p in PARTS:
        part = data.get(p)
        if not part:
            continue
        m = part.get("meta", {})
        for name, h in (m.get("config_hash") or {}).items():
            if name in cur_h and cur_h[name] != h:
                out.append(f"{p}: preset {name} changed since it was computed (config hash {h} -> {cur_h[name]})")
        for r, h in (m.get("targets_sha256") or {}).items():
            if cur_t.get(r) != h:
                out.append(f"{p}: targets file {r}.yaml changed since it was computed")
        cur_c = _code_sha(p)
        for f, h in (m.get("code_sha256") or {}).items():
            if f in cur_c and cur_c[f] != h:
                out.append(f"{p}: {f} changed since it was computed")
    return out


def _tolerance_text(tol: Mapping) -> str:
    k = tol.get("kind")
    if k in ("abs", "rel"):
        return f"{k} {_num(tol.get('value'))}"
    if k == "interval":
        return f"in [{_num(tol.get('lo'))}, {_num(tol.get('hi'))}]"
    if k == "inequality":
        s = f"{tol.get('op')} {_num(tol.get('value'))}"
        return s + (f" (hard {_num(tol['hard_value'])})" if tol.get("hard_value") is not None else "")
    if k == "band":
        s = f"band [{_num(tol.get('lo'))}, {_num(tol.get('hi'))}] slack {_num(tol.get('slack', 0))}"
        return s + (" either sign" if tol.get("either_sign") else "")
    if k == "rms_abs":
        return f"rms <= {_num(tol.get('value'))}"
    if k == "contour_asymmetry":
        return f"offsets abs {_num(tol.get('abs'))}, {tol.get('require')}"
    return str(k)


def _measured(t) -> str:
    v = t.value
    if isinstance(v, Mapping) and "y" in v and "x" in v:
        s = f"{_fmt(v['y'])} at {v.get('x_name')} {_fmt(v['x'])}"
    elif isinstance(v, Mapping) and v.get("form") == "rectified_sine":
        s = f"rectified sine {_fmt(v.get('params'))}"
    else:
        s = _fmt(v)
    if t.sigma is not None:
        s += f" +- {_fmt(t.sigma)}"
    return s + (f" {t.unit}" if t.unit not in ("1", "mixed") else "")


def _band(t, rec: Mapping, corners: Mapping[str, Mapping]) -> tuple[str, str]:
    """(band text, 'k/n corners pass') from the nominal and the corner predictions."""
    vals = []
    if rec.get("replica_status") == "ok":
        vals.append(rec.get("predicted"))
    passes, n = 0, 0
    for cn, d in corners.items():
        c = d.get(t.id) or {}
        if c.get("status") != "ok":
            continue
        n += 1
        vals.append(c.get("predicted"))
        if t.scored:
            try:
                if tio.evaluate(t, c.get("predicted")).level == "pass":
                    passes += 1
            except Exception:                                           # noqa: BLE001
                pass
    if not vals:
        return "-", "-"
    flats = [_flatten(v) for v in vals]
    size = flats[0].size
    flats = [f for f in flats if f.size == size]
    F = np.vstack(flats)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lo, hi = np.nanmin(F, axis=0), np.nanmax(F, axis=0)
    if size == 1:
        txt = f"[{_num(lo[0])}, {_num(hi[0])}]"
    elif isinstance(vals[0], Mapping) and "t_s" in vals[0]:
        txt = "(trajectory)"
    elif rec.get("replica_status") != "ok":
        txt = f"corners only: element ranges up to {_num(float(np.nanmax(hi - lo)))} wide"
    else:
        dev = np.nanmax(np.abs(F - F[0]), axis=0) if np.all(np.isfinite(F[0])) else np.full(size, np.nan)
        txt = f"nominal +- up to {_num(np.nanmax(dev)) if np.any(np.isfinite(dev)) else 'nan'}"
    return txt, (f"{passes}/{n}" if t.scored and n else "-")


def _section_targets(data: Mapping, presets: Mapping[str, str], ddir: Path, charts: bool) -> list[str]:
    P = data.get("predictions")
    out = ["## 1. Targets: measured, predicted, scored", "", '<a id="targets"></a>', ""]
    if not P:
        return out + ["Not computed yet: run `python -m mrm.validation.report --recompute predictions`.", ""]
    recs = {r["id"]: r for r in P["records"]}
    corners = P.get("corners") or {}
    out += [f"Replica protocols (`mrm.validation.replicas`) on the nominal world (every uncertainty entry at its "
            f"nominal, no device variation): G26 targets on `{presets['G26']}`, G25 bench targets on "
            f"`{presets['G25']}`. Scored with `targets_io.evaluate` against the frozen tolerances only; sigma enters "
            "only z = max |residual_i| / sigma_i. **Band** = the range of the prediction over the nominal world and "
            f"the {len(corners)} named corners of 07 4.8 (each corner's alternatives applied to the same presets): "
            "a hypothesis range, not a probability interval. **Corners pass** counts the corners whose prediction "
            "passes the tolerance.", ""]
    if presets["G26"] != "g26" and presets["G25"] == G25_DEFAULT:
        out += [f"Warning: no G25 companion of `{presets['G26']}` exists, so the G25 bench targets run on the "
                f"uncalibrated `{G25_DEFAULT}`.", ""]
    unapplied: dict[str, set] = {}
    tmap = {t.id: t for role in tio.TARGET_ROLES for t in tio.load_targets(role)}
    for cn, dd in corners.items():
        for tid, c in dd.items():
            if c.get("status") == "error" and tid in tmap:
                unapplied.setdefault(f"{cn} on the {_plant_key(tmap[tid])} plant "
                                     f"(`{presets[_plant_key(tmap[tid])]}`): {c.get('note', '')[:160]}", set()).add(tid)
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
    # evaluation
    rows, zrows = {}, []
    counts: dict[str, dict[str, int]] = {}
    for role in tio.TARGET_ROLES:
        for t in tio.load_targets(role):
            r = recs.get(t.id, {"replica_status": "missing", "predicted": None, "notes": []})
            st = r.get("replica_status")
            if st == "ok":
                pred = r.get("predicted")
                ev = tio.evaluate(t, float("nan") if r.get("predicted_nonfinite") and pred is None else pred)
                lvl = ev.level
                if not t.scored:
                    res = f"not scored ({t.status}; would be {lvl})" if lvl != "not_scored" else f"not scored ({t.status})"
                else:
                    res = {"soft_fail": "marginal (soft_fail)"}.get(lvl, lvl)
                z = ev.z
            else:
                ev, lvl, z, res = None, st, None, st
                if not t.scored:
                    res = f"not scored ({t.status}); {st}"
            c = counts.setdefault(role, {"scored": 0, "pass": 0, "marginal": 0, "fail": 0, "blocked/error": 0,
                                         "not scored": 0})
            if not t.scored:
                c["not scored"] += 1
            else:
                c["scored"] += 1
                if st != "ok":
                    c["blocked/error"] += 1
                elif lvl == "pass":
                    c["pass"] += 1
                elif lvl == "soft_fail":
                    c["marginal"] += 1
                else:
                    c["fail"] += 1
            band, cp = _band(t, r, corners)
            flags = [TARGET_FLAGS[t.id]] if t.id in TARGET_FLAGS else []
            if calibrated and t.id in CALIBRATED_FLAGS:
                flags.append(CALIBRATED_FLAGS[t.id])
            if r.get("predicted_nonfinite"):
                flags.append("prediction not finite")
            if r.get("artefact_fallbacks"):
                flags.append(f"artefact fallback: {_fmt(r['artefact_fallbacks'])}")
            pred_txt = "nan" if r.get("predicted_nonfinite") and r.get("predicted") is None else _fmt(r.get("predicted"))
            rows.setdefault(role, []).append(
                f"| {t.id} | {t.status} | {_md(t.quantity)} | {_md(_measured(t))} | {_md(pred_txt)} | {_md(band)} | "
                f"{_num(z, 3)} | {_md(_tolerance_text(t.tolerance))} | {_md(res)} | {cp} | {_md('; '.join(flags))} |")
            if t.scored and st == "ok" and z is not None:
                zrows.append((t.id, z, "pass" if lvl == "pass" else ("marginal" if lvl == "soft_fail" else "fail")))
    out += ["| role | scored | pass | marginal | fail | blocked or error | not scored |", "|---|---|---|---|---|---|---|"]
    for role, c in counts.items():
        out.append(f"| {role} | {c['scored']} | {c['pass']} | {c['marginal']} | {c['fail']} | {c['blocked/error']} | "
                   f"{c['not scored']} |")
    out += ["", "VAL-4c is not an independent third pass of the VAL-4 group, and VAL-4c and BND-4 rest on the same "
            "printed spread (target split handoff, request 4).", ""]
    if charts and zrows:
        _chart_z(zrows, ddir / "targets_z.svg")
    if zrows:
        out += [f"![z-scores of the scored targets]({_rel(ddir / 'targets_z.svg')})", ""]
    head = ["| id | status | quantity | measured +- sigma | predicted (nominal) | band (corners) | z | tolerance | "
            "result | corners pass | flags |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for role in tio.TARGET_ROLES:
        out += [f"### 1.{tio.TARGET_ROLES.index(role) + 1} {role.capitalize()} targets", ""] + head + rows.get(role, []) + [""]
    notes = [(r["id"], n) for r in P["records"] for n in r.get("notes") or []]
    if notes:
        out += ["<details><summary>Replica notes (assumptions and blocked reasons, per target)</summary>", ""]
        for i, n in notes:
            out.append(f"- **{i}**: {_md(n[:400])}{'...' if len(n) > 400 else ''}")
        out += ["", "</details>", ""]
    return out


def _closure_tests() -> dict[str, list[str]]:
    """{anchor: [test ids]} of xfail tests whose reason links to VALIDATION.md#closure-<anchor>."""
    import ast
    found: dict[str, list[str]] = {}
    pat = re.compile(r"VALIDATION\.md#closure-([a-z0-9-]+)")

    def xfail_reason(dec) -> tuple[str, bool] | None:
        """(reason, strict) of a ``pytest.mark.xfail(...)`` decorator (or ``mark.xfail`` / ``xfail``), else None."""
        if not isinstance(dec, ast.Call):
            return None
        fn = dec.func
        name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else None)
        if name != "xfail":
            return None
        kw = {k.arg: k.value for k in dec.keywords if k.arg}
        r = kw.get("reason")
        reason = r.value if isinstance(r, ast.Constant) and isinstance(r.value, str) else None
        st = kw.get("strict")
        return (reason, isinstance(st, ast.Constant) and st.value is True) if reason else None

    # Parsed, not grepped: only a test function decorated with xfail(reason=...) that names the anchor counts (a
    # string in a test body, such as the fixture of tests/verification/test_report.py, is not a closure test).
    for f in sorted((REPO / "tests").rglob("test_*.py")):
        text = f.read_text(errors="replace")
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
                xr = xfail_reason(dec)
                if not xr:
                    continue
                for m in pat.finditer(xr[0]):
                    found.setdefault(m.group(1), []).append(
                        f"`{f.relative_to(REPO).as_posix()}::{node.name}`" + ("" if xr[1] else " (not strict)"))
    return found


def _closure_model(cfg, d, preds: Mapping[str, Any]) -> dict[str, str]:
    diag = d.diagnostics
    lv = cfg.leaves

    def leaf(p):
        x = lv.get(p)
        return None if x is None else x.value

    def dg(k, i=0):
        v = diag.get(k)
        return None if v is None else float(np.asarray(v, float).ravel()[i])

    def unom(n):
        e = cfg.uncertainty.get(n)
        return None if e is None else (e.pinned if e.pinned is not None else e.fields.get("nominal"))

    def pr(i):
        v = preds.get(i)
        return _fmt(v) if v is not None else "-"
    fw, dist = dg("fwhm_pm"), dg("disturbance_pm_per_k")
    return {
        "dlambda-dt": f"{_num(leaf('ring.thermal.dlambda_dt_pm_per_k'))} pm/K at "
                      f"{_num(leaf('ring.thermal.dlambda_dt_ref_ring_temp_c'))} degC (hypothesis "
                      f"{unom('thermo_optic')}), {_num(dist)} pm/K at the lock temperature "
                      f"{_num(dg('t_ring_lock_c'))} degC; CAL-6 apparent sensitivity {pr('CAL-6')} pm/K",
        "fsr": f"{_num(dg('fsr_thz'))} THz = {_num(dg('fsr_nm'))} nm (hypothesis {unom('fsr')})",
        "thickness": f"{_num(leaf('ring.geometry.film_nm'))} nm (hypothesis {unom('film_thickness')})",
        "q": f"Q_L {_num(leaf('ring.optics.q_loaded'))}; CAL-1 replica {pr('CAL-1')}",
        "heater-efficiency": f"{_num(dg('heater_efficiency_nm_per_mw'))} nm/mW (R2 from P_FSR "
                             f"{_num(leaf('ring.heater.p_fsr_mw'))} mW and the FSR)",
        "rth": f"{_num(dg('z_heater_ring_k_per_mw'))} K/mW (heater to ring, nominal world)",
        "reloma-level": f"CAL-14 replica {pr('CAL-14')} dB",
        "reloma-curvature": (f"FWHM / (dlambda/dT at lock) = {_num(fw)} pm / {_num(dist)} pm/K = "
                             f"{_num(fw / dist if fw and dist else None)} K"),
        "aggressor-rate": f"CAL-13b replica {pr('CAL-13b')} K/s (scenario calibration)",
        "g25-topology": f"add-drop, kappa_d/kappa {_num(leaf('ring.optics.kappa_d_over_kappa'))}; the replicas also "
                        "report the all-pass reading as a variant",
        "eo-bandwidth": f"CAL-10 replica {pr('CAL-10')} GHz at 1 V, VAL-3a {pr('VAL-3a')} GHz at 2 V",
        "rom-dc-scale": (f"{_fmt(diag['rom_dc_scale'])}" if "rom_dc_scale" in diag else
                         "not computed: the preset uses the interim thermal model (no FEM ROM)"),
    }


def _section_closure(cfg, d, data: Mapping) -> list[str]:
    preds = {r["id"]: (r.get("predicted") if r.get("replica_status") == "ok" else None)
             for r in (data.get("predictions") or {}).get("records", [])}
    model = _closure_model(cfg, d, preds)
    tests = _closure_tests()
    out = ["## 2. Closure and conflict table (07 4.6)", "", '<a id="closure"></a>', "",
           "Published numbers that do not close through `derive()`. Each row is an open conflict; the model column "
           f"shows what preset `{cfg.name}` uses. CI keeps them visible through `xfail(strict=True, "
           "reason='VALIDATION.md#closure-<id>')` tests in `tests/closure/` (07 4.1); a row without a test is "
           "listed as such.", "", "| id | quantity | conflicting evidence | model (this preset) | closure test |",
           "|---|---|---|---|---|"]
    missing = []
    for slug, q, ev in CLOSURE_ROWS:
        tl = tests.get(slug)
        if not tl:
            missing.append(slug)
        out.append(f'| <a id="closure-{slug}"></a>closure-{slug} | {_md(q)} | {_md(ev)} | {_md(model.get(slug))} | '
                   f"{', '.join(tl) if tl else 'none yet'} |")
    out.append("")
    if missing:
        out += [f"{len(missing)} of {len(CLOSURE_ROWS)} rows have no xfail(strict=True) closure test yet "
                "(requested in docs/handoff/integrator.md section 9.4).", ""]
    return out


#: presets whose report shows a fit record that names no ``preset`` (``mrm/calibration/latest_fit.json``): g26, the
#: preset the fit starts from, and g26_calibrated, the preset it writes. Any other preset (a corner, a test preset)
#: shows only records that name it.
FIT_RECORD_PRESETS = ("g26", "g26_calibrated")

#: flags of the calibration rows in the report of a calibrated preset (where the fit's protocol differs from the
#: replica's primary, or the fit did not reach the target; docs/handoff/calibration.md sections 4 and 7)
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


def _calibrated(preset: str) -> bool:
    """True for a preset written by the calibration (a fit record belongs to it and it is not the base g26)."""
    return preset != "g26" and bool(_calibration_outputs(preset))


def _calibration_outputs(preset: str) -> list[dict]:
    """Calibration-task results for ``preset``: JSON files under mrm/calibration/ (any depth) or
    docs/calibration/ whose ``schema`` starts with ``mrm.calibration`` and whose ``preset`` is this preset, or absent
    for a preset of :data:`FIT_RECORD_PRESETS` (``mrm/calibration/latest_fit.json`` names no preset: it is the fit on
    the g26 / G25 bench presets that g26_calibrated is written from, shown in the report of either)."""
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


def _fit_target_audit(ids: Sequence[str]) -> list[str]:
    """Problems with target ids a fit used (07 P0-5, targets README): a held-out id (validation or bounds) is
    CONTAMINATION; an excluded id, an unknown id or a calibration id with ``fit: false`` (CAL-6) is a violation too.
    Empty when every id is a calibration.yaml id that may be fitted."""
    cal = {t.id: t for t in tio.load_calibration_targets()}
    held = {t.id for r in tio.HELD_OUT_ROLES for t in tio.load_targets(r)}
    try:
        exc = {e.id for e in tio.load_excluded()}
    except Exception:                                                   # noqa: BLE001
        exc = set()
    out = []
    bad_h = sorted({i for i in ids if i in held})
    if bad_h:
        out.append(f"CONTAMINATION: fitted to held-out {bad_h}")
    bad_e = sorted({i for i in ids if i in exc})
    if bad_e:
        out.append(f"fitted to excluded {bad_e}")
    unk = sorted({i for i in ids if i not in cal and i not in held and i not in exc})
    if unk:
        out.append(f"unknown target ids {unk}")
    nofit = sorted({i for i in ids if i in cal and cal[i].raw.get("fit", True) is False})
    if nofit:
        out.append(f"fitted to {nofit}, which calibration.yaml marks fit: false")
    return out


def _chart_profiles(profs: Mapping[str, Mapping], path: Path) -> None:
    """Profile likelihoods (chi2 - chi2_min against the parameter, with the 95 % threshold) of one group."""
    plt = _mpl()
    names = [n for n, p in profs.items() if p.get("grid") and p.get("chi2")]
    if not names:
        return
    ncol = min(4, len(names))
    nrow = int(math.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol + 0.6, 1.6 * nrow + 0.4), squeeze=False)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    for ax, n in zip(axes.ravel(), names):
        p = profs[n]
        x = np.asarray([np.nan if v is None else v for v in p["grid"]], float)
        y = np.asarray([np.nan if v is None else v for v in p["chi2"]], float) - float(p.get("chi2_min") or 0.0)
        ax.plot(x, y, color=C_BLUE, lw=1.0, marker="o", ms=2)
        thr = p.get("threshold")
        if thr is not None:
            ax.axhline(float(thr) - float(p.get("chi2_min") or 0.0), color=C_INK, lw=0.6, ls="--")
        pr = p.get("prior") or []
        for v in pr[:2]:
            if v is not None:
                ax.axvline(float(v), color=C_GREY, lw=0.6)
        if p.get("log"):
            ax.set_xscale("log")
        ax.set_title(str(n).split(".")[-1], fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0][0].set_ylabel("chi2 - min", fontsize=7)
    fig.tight_layout()
    _save(fig, path)


def _section_calibration_fit(c: Mapping, ddir: Path, charts: bool) -> list[str]:
    """Section 3.1 from a ``mrm.calibration.fit/1`` result (mrm/calibration/latest_fit.json)."""
    groups = c.get("groups") or {}
    sha, _, suffix = str(c.get("git_commit") or "").partition("-")          # "<sha>[-dirty]"
    out = [f"From `{c['_file']}` (schema `{c.get('schema')}`, created {c.get('created_utc')}, commit "
           f"`{sha[:12]}`{' with uncommitted changes' if suffix == 'dirty' else ''}). Every target id the fit used "
           "(group targets, per-hypothesis residual targets, leaf decisions) is checked against the frozen split.", ""]
    for g, gr in groups.items():
        fits = gr.get("fits") or {}
        ids = set(map(str, gr.get("targets") or []))
        for f in fits.values():
            ids |= set(map(str, (f.get("targets") or {}).keys()))
        for dcs in gr.get("decisions") or []:
            ids |= set(map(str, dcs.get("targets") or [])) | set(map(str, (dcs.get("residual") or {}).keys()))
        aud = _fit_target_audit(sorted(ids))
        cmp_ = gr.get("comparison") or {}
        out += [f"**Group `{g}`** (nominal hypothesis `{gr.get('nominal')}`, best `{cmp_.get('best')}`). Targets: "
                f"{', '.join(sorted(ids)) or 'none'}. Audit: "
                + ("**" + "; ".join(aud) + "**" if aud else "ok (calibration ids only)") + ".", ""]
        rows = cmp_.get("rows") or []
        if rows:
            out += ["| hypothesis | chi2 | dof | Birge ratio^2 | delta chi2 | likelihood ratio (Birge) | delta AIC |",
                    "|---|---|---|---|---|---|---|"]
            for r in rows:
                f = fits.get(r.get("hypothesis")) or {}
                out.append(f"| {_md(r.get('hypothesis'))} | {_num(r.get('chi2'))} | {r.get('dof')} | "
                           f"{_num(f.get('birge2'))} | {_num(r.get('delta_chi2'))} | "
                           f"{_num(r.get('likelihood_ratio_birge'))} | {_num(r.get('delta_aic'))} |")
            if cmp_.get("note"):
                out += ["", _md(cmp_["note"])]
            out.append("")
        dec = gr.get("decisions") or []
        if dec:
            out += ["| leaf | scope | value (was) | provenance (was) | 95 % profile hull | prior | width / prior | "
                    "rule | unique | targets |", "|---|---|---|---|---|---|---|---|---|---|"]
            for dcs in dec:
                idf = dcs.get("identifiable") or {}
                out.append(f"| `{dcs.get('path')}` | {dcs.get('scope')} | {_md(_fmt(dcs.get('value')))} "
                           f"({_md(_fmt(dcs.get('previous_value')))}) | {dcs.get('provenance')} "
                           f"({dcs.get('previous_provenance')}) | {_md(_fmt(idf.get('profile_95')))} | "
                           f"{_md(_fmt(idf.get('prior')))} | {_num(idf.get('width_frac'), 3)} | "
                           f"{_md(idf.get('rule'))} | {_md(_fmt(idf.get('unique')))} | {_md(_fmt(dcs.get('targets')))} |")
            out.append("")
        # the profiles of the written (nominal) hypothesis, whose leaves the decisions above describe; the best
        # hypothesis when the nominal has none
        hyp = gr.get("nominal") if (gr.get("profiles") or {}).get(gr.get("nominal")) else cmp_.get("best")
        profs = (gr.get("profiles") or {}).get(hyp) or {}
        if profs:
            f = ddir / f"profiles_{re.sub(r'[^a-z0-9_]+', '_', str(g).lower())}.svg"
            if charts:
                _chart_profiles(profs, f)
            out += [f"![profile likelihoods, group {g}, hypothesis {hyp}]({_rel(f)})", ""]
    if c.get("notes"):
        out += ["Calibration notes: " + _md("; ".join(map(str, c["notes"]))), ""]
    return out


def _section_identifiability(cfg, data: Mapping, preset: str, ddir: Path, charts: bool) -> list[str]:
    I = data.get("identifiability")
    out = ["## 3. Identifiability of the calibration parameters", "", '<a id="identifiability"></a>', ""]
    if not I:
        out += ["Not computed yet: run `python -m mrm.validation.report --recompute identifiability`.", ""]
    else:
        svd = I.get("svd") or {}
        out += [f"Jacobian of the calibration replicas (calibration targets only; the scenario calibrations "
                f"{', '.join(SCENARIO_REPLICAS)} and the never-fitted CAL-6 left out) with respect to "
                f"{len(I['params'])} candidate parameters of 07 4.3, by central differences of "
                f"{_num(100 * I['meta'].get('rel_step', 0.05))} % of each prior range (one-sided at a range end), "
                "at the preset's values. Entries are normalized: residual sigmas (the target sigma; a quarter of an "
                "interval tolerance or half an abs/rel tolerance where no sigma is printed) per prior half-width "
                "(the leaf's `range`). In these units the 07 4.3 half-prior rule (95 % interval at most half the "
                "prior range) is 1.96 sigma <= 0.5, i.e. a direction is identified when its singular value is at "
                f"least {sens.HALF_PRIOR_SINGULAR_VALUE:.2f}.", "",
                "This is a local, linearized screen at a point that is not a fit (the preset is "
                + ("uncalibrated" if preset == "g26" else "the calibrated one") + "): a parameter that fails it "
                "cannot be tagged `calibrated` from these targets; one that passes still needs the profile "
                "likelihood of 07 4.3 (below). The screen works in linear coordinates of each range and fits all "
                "candidates jointly; the calibration task applies the rule to profile intervals per group, in log "
                "coordinates for a log-scale parameter, so its verdicts can differ.", ""]
        if svd:
            out += [f"Residuals used: {svd.get('n_residuals_used')} (dropped as non-finite: "
                    f"{svd.get('n_residuals_dropped')}); rank {svd.get('rank')} of {len(svd.get('names', []))}; "
                    f"condition number {_num(svd.get('condition'))}; directions passing the rule: "
                    f"{sum(1 for x in svd.get('directions', []) if x['identified'])} of "
                    f"{len(svd.get('directions', []))}.", ""]
            if charts:
                _chart_svd(svd["s"], svd["threshold_singular_value"], ddir / "identifiability_svd.svg")
            out += [f"![singular values]({_rel(ddir / 'identifiability_svd.svg')})", ""]
            out += ["| k | singular value | rule | largest loadings |", "|---|---|---|---|"]
            for k, dd in enumerate(svd["directions"], start=1):
                lo = ", ".join(f"{n.split('.')[-1]} {v:+.2f}" for n, v in dd["loadings"])
                out.append(f"| {k} | {_num(dd['s'], 3)} | {'identified' if dd['identified'] else 'not identified'} | "
                           f"{_md(lo)} |")
            out.append("")
        idx = {n: i for i, n in enumerate((svd or {}).get("names", []))}
        out += ["| parameter | meant to explain | prior range | value | column norm | linearized 95 % interval / "
                "prior range | half-prior rule | provenance |", "|---|---|---|---|---|---|---|---|"]
        for p in I["params"]:
            if "skipped" in p:
                out.append(f"| `{p['path']}` | {_md(p['identifies'])} | | | | skipped: {_md(p['skipped'])} | | |")
                continue
            i = idx.get(p["path"])
            hw = svd["half_width_95"][i] if i is not None else None
            out.append(f"| `{p['path']}` | {_md(p['identifies'])} | [{_num(p['prior'][0])}, {_num(p['prior'][1])}] | "
                       f"{_num(p.get('value'))} | {_num(svd['column_norm'][i], 3) if i is not None else '-'} | "
                       f"{_num(hw, 3)} | {'pass' if hw is not None and hw <= 0.5 else 'fail'} | {p.get('provenance')} |")
        kept = [p for p in I["params"] if p.get("kept")]
        if kept:
            out += ["", "Moved on the G26 plant only (the other plant's leaf has another value, so it is a different "
                    "physical quantity there; its targets get a zero column): " + "; ".join(
                        f"`{p['path']}` ({_md('; '.join(p['kept'].values()))})" for p in kept) + "."]
        out += ["", "Base point (the preset's values): " + ", ".join(
            f"{r['id']} {r['level']}" + (f" (z {_num(r['z'], 3)})" if r.get("z") is not None else "")
            for r in I.get("base_evaluation", [])) + ".", ""]
        if svd:
            cn = np.asarray(svd["column_norm"], float)
            zero = [n for n, c in zip(svd["names"], cn) if c <= 1e-9 * max(cn.max(), 1e-300)]
            fail = [n for n, h in zip(svd["names"], svd["half_width_95"]) if not h <= 0.5]
            if zero:
                out += ["No calibration target responds to " + ", ".join(f"`{n}`" for n in zero) + " in this preset "
                        "(a zero Jacobian column): it cannot be calibrated from these targets at all.", ""]
            if fail:
                out += [f"Failing the linearized half-prior rule (all {len(svd['names'])} fitted jointly): " +
                        ", ".join(f"`{n}`" for n in fail) + ". They keep `assumed` (or their current provenance) with "
                        "a range unless another target is added.", ""]
            big = [r["id"] for r in I.get("base_evaluation", []) if r.get("z") is not None and r["z"] > 10]
            if big:
                out += ["Caution: " + ", ".join(big) + " miss by more than 10 sigma at this point, so the largest "
                        "singular values describe the slope far from any fit (CAL-2 also crosses critical coupling, "
                        "where the notch depth is not smooth). The linearization says which combinations the targets "
                        "constrain, not the values a fit would reach; a model-form error (a target no parameter "
                        "value can reach) shows up as a large residual that survives the fit, not here.", ""]
    # profile likelihoods (calibration task) and the calibrated-leaf audit
    out += ["### 3.1 Profile likelihoods (calibration task)", ""]
    cal = _calibration_outputs(preset)
    if not cal:
        out += ["No calibration output for this preset was found (`mrm/calibration/latest_fit.json`, schema "
                "`mrm.calibration.fit/1`, or a JSON file under `mrm/calibration/` or `docs/calibration/` with "
                "`schema: mrm.calibration...`, `preset` and `parameters: [{path, prior, estimate, profile_ci95, "
                "half_prior_rule, unique, targets, residual}]`). The profile likelihoods of 07 4.3 are therefore not "
                "shown.", ""]
    for c in cal:
        if isinstance(c.get("groups"), Mapping):
            out += _section_calibration_fit(c, ddir, charts)
            continue
        params_ = c.get("parameters") or []
        ids = sorted({str(x) for p in params_ for x in (p.get("targets") or [])})
        aud = _fit_target_audit(ids)
        out += [f"From `{c['_file']}`. Fitted targets: {', '.join(ids) or 'none listed'}; audit: "
                + ("; ".join(aud) if aud else "every fitted id is a calibration.yaml id with fit allowed") + ".", "",
                "| parameter | prior | estimate | 95 % profile interval | half-prior rule | unique | targets | residual |",
                "|---|---|---|---|---|---|---|---|"]
        for p in params_:
            out.append(f"| `{p.get('path')}` | {_md(_fmt(p.get('prior')))} | {_md(_fmt(p.get('estimate')))} | "
                       f"{_md(_fmt(p.get('profile_ci95')))} | {_md(_fmt(p.get('half_prior_rule')))} | "
                       f"{_md(_fmt(p.get('unique')))} | {_md(_fmt(p.get('targets')))} | {_md(_fmt(p.get('residual')))} |")
        out.append("")
    calib = [(p, lf) for p, lf in sorted(cfg.leaves.items()) if lf.provenance == "calibrated"]
    out += ["### 3.2 Leaves tagged `calibrated` in this preset", ""]
    if not calib:
        out += [f"None: preset `{cfg.name}` has no `calibrated` leaf.", ""]
    else:
        out += ["| leaf | value | targets | residual | identifiable | audit |", "|---|---|---|---|---|---|"]
        jac = {p["path"]: p for p in (I or {}).get("params", [])}
        idx = {n: i for i, n in enumerate(((I or {}).get("svd") or {}).get("names", []))}
        for p, lf in calib:
            audit = []
            if not lf.targets:
                audit.append("no targets listed (07 P0-5)")
            audit += _fit_target_audit([str(x) for x in (lf.targets or [])])
            if lf.residual is None or lf.identifiable is None:
                audit.append("residual or identifiability missing (07 4.3)")
            if p in idx and I and (I.get("svd") or {}).get("half_width_95"):
                if I["svd"]["half_width_95"][idx[p]] > 0.5:
                    audit.append("fails the linearized half-prior rule above")
            elif p not in jac:
                audit.append("not in the Jacobian parameter set")
            out.append(f"| `{p}` | {_fmt(lf.value)} | {_fmt(lf.targets)} | {_fmt(lf.residual)} | "
                       f"{_fmt(lf.identifiable)} | {_md('; '.join(audit) or 'ok')} |")
        out.append("")
    return out


def _section_uncertainty(data: Mapping, ddir: Path, charts: bool) -> list[str]:
    S = data.get("sensitivity")
    out = ["## 4. Uncertainty propagation (07 4.8 set)", "", '<a id="uncertainty"></a>', ""]
    if not S:
        return out + ["Not computed yet: run `python -m mrm.validation.report --recompute sensitivity`.", ""]
    m = S["meta"]
    out += ["Performance is conditional on the assumed thermal dynamics (heater, crosstalk and aggressor poles) "
            "until Q1 (the heater step response at the lock point, 07 section 5) is answered.", "",
            f"Protocol: controller `{m['controller']}` under {m['scenario']}, warm start, self reference, metrics "
            "only (the compare_locks protocol of 07 4.8 for one controller), variation off (epistemic worlds only). "
            f"Every world is one row of a u-design injected into `realize(..., uncertainty='lhs')` and checked "
            "against realize's draws (`mrm.validation.sensitivity`). The tornado moves one entry at a time to its "
            "interval ends or to each reachable hypothesis choice (stress choices with zero weight are corners, not "
            "part of the sampled set), all other entries at nominal; the base world is the nominal world (a "
            "nominal at an interval end becomes 1e-12 of the range from it). The top "
            f"{len(S['top'])} entries by normalized swing (largest swing over the metrics, each metric normalized by "
            "its largest finite swing) then get Saltelli (2010) first-order and Jansen total Sobol indices over "
            "their prior (u-space; hypotheses by their weights), the other entries held at nominal, so the indices "
            "are conditional on those. Bootstrap intervals: "
            f"{int(100 * 0.95)} %, {m.get('n_boot')} resamples. "
            + ("Noise seeds differ by world index (world b uses base + b), so a Saltelli row and its AB_i partner "
               f"also differ in noise: the nominal world rerun with {m['n_noise_replicates']} other noise seeds "
               "gives the noise variance of each metric below; its share of the output variance is added to every "
               "ST (and is the smallest ST that means anything)." if m.get("n_noise_replicates") else
               "Noise seeds differ by world index (world b uses base + b); the noise variance was not measured in "
               "this computation."), "",
            f"Cost: {m['n_tornado_worlds']} tornado worlds + {m['n_sobol_worlds']} Sobol worlds ({m['n_base']} base "
            f"rows x ({len(S['top'])} + 2)), at most {m['chunk']} worlds per Plant, {m['n_threads']} kernel "
            f"threads; wall {_num(m['wall_s'])} s (Sobol part {_num(m.get('sobol_wall_s'))} s), peak RSS "
            f"{_num(m['peak_rss_gb'])} GB of the report process.", ""]
    if S.get("tornado_supply_limited"):
        out += ["Tornado worlds that start supply-limited (the bank cannot hold every lock at the warm start, "
                "experiments handoff): " + ", ".join(f"`{k}`" for k in S["tornado_supply_limited"]) + ".", ""]
    if S.get("sobol_supply_limited_frac") is not None:
        out += [f"Fraction of Sobol worlds that start supply-limited: {_num(S['sobol_supply_limited_frac'], 3)}.", ""]
    base = S["base"]
    out += ["Nominal world: " + ", ".join(f"{k} {_num(v)}" for k, v in base.items()) + ".", ""]
    out += ["### 4.1 Tornado (one entry at a time)", ""]
    for md in S["metrics"]:
        k = md["key"]
        rows = S["tornado"].get(k, [])
        title = f"{k}" + (f" ({md['unit']})" if md["unit"] else "") + "; blue: below nominal, orange: above"
        f = ddir / f"tornado_{k}.svg"
        if charts and rows:
            _chart_tornado([r for r in rows if r["swing"] != 0] or rows[:1], base.get(k, float("nan")), title, f)
        out += [f"**{k}** (nominal {_num(base.get(k))} {md['unit']})", "",
                f"![tornado {k}]({_rel(f)})", "",
                "| entry | low end | high end | swing |", "|---|---|---|---|"]
        moving = [r for r in rows if r["swing"] != 0]
        for r in moving[:12]:
            out.append(f"| `{r['entry']}` | {_num(r['lo'])} ({_md(r['lo_label'])}) | {_num(r['hi'])} "
                       f"({_md(r['hi_label'])}) | {_num(r['swing'])} |")
        rest = [r["entry"] for r in rows if r["swing"] == 0]
        nz = moving[12:]
        if nz:
            out.append(f"| ({len(nz)} more entries) | | | <= {_num(max(r['swing'] for r in nz))} |")
        out.append("")
        if rest:
            out += [f"No change of {k} when moved alone (the others at nominal): " + ", ".join(f"`{e}`" for e in rest)
                    + ".", ""]
    out += ["### 4.2 Sobol indices of the top entries", "",
            "Screened entries: " + ", ".join(f"`{e}`" for e in S["top"]) + ".", ""]
    if charts:
        _chart_sobol(S["sobol"], S["top"], ddir / "sobol.svg")
    out += [f"![Sobol indices]({_rel(ddir / 'sobol.svg')})", ""]
    for md in S["metrics"]:
        k = md["key"]
        e = S["sobol"].get(k, {})
        hdr = f"**{k}**" + (f" (indices of {e['transform']})" if e.get("transform") else "")
        if "st" not in e:
            out += [hdr + ": " + "; ".join(e.get("notes", ["not computed"])), ""]
            continue
        has_b = "s1_binned" in e
        noise = (f", noise variance {_num(e['noise_variance'])} ({_num(100 * e['noise_share'], 2)} % of it)"
                 if e.get("noise_share") is not None and np.isfinite(e.get("noise_share", np.nan)) else "")
        out += [hdr + f": variance {_num(e['variance'])}{noise}, rows dropped as non-finite "
                f"{e.get('n_rows_dropped_nonfinite', 0)}, sum of S1 {_num(sum(x for x in e['s1'] if x is not None), 3)}"
                + (f" (binned: {_num(sum(x for x in e['s1_binned'] if x is not None), 3)})" if has_b else "")
                + f", sum of ST {_num(sum(x for x in e['st'] if x is not None), 3)}"
                + (f" (sum of the ST lower 95 % bounds {_num(sum(r[0] for r in e['st_ci'] if r[0] is not None), 3)})")
                + ("; " + "; ".join(e["notes"]) if e.get("notes") else ""), "",
                "| entry | S1 | S1 95 % |" + (" S1 binned | S1 binned 95 % |" if has_b else "") + " ST | ST 95 % |",
                "|---|---|---|" + ("---|---|" if has_b else "") + "---|---|"]
        for i, n in enumerate(e["names"]):
            b = (f" {_num(e['s1_binned'][i], 3)} | [{_num(e['s1_binned_ci'][i][0], 2)}, "
                 f"{_num(e['s1_binned_ci'][i][1], 2)}] |" if has_b else "")
            out.append(f"| `{n}` | {_num(e['s1'][i], 3)} | [{_num(e['s1_ci'][i][0], 2)}, {_num(e['s1_ci'][i][1], 2)}] |"
                       + b + f" {_num(e['st'][i], 3)} | [{_num(e['st_ci'][i][0], 2)}, {_num(e['st_ci'][i][1], 2)}] |")
        out.append("")
    out += ["Reading: ST - S1 is the share of variance through interactions; a sum of ST above 1 (judge it by the "
            "sum of the lower bounds) means the metric responds to combinations of entries, typically a threshold "
            "such as a lock loss or a supply limit. The Saltelli S1 (from the A, B and AB_i rows) is poorly resolved "
            f"with {S['meta']['n_base']} base rows for these threshold-like metrics (a negative S1 or an interval "
            "that spans 0 is estimator noise, not a negative effect). "
            + ("'S1 binned' re-estimates S1 from the 2 x "
               f"{S['meta']['n_base']} independent A and B rows by binning each entry (a hypothesis by its choices, "
               "an interval in 8 equal-probability bins, bias of the bin means removed; "
               "`sensitivity.given_data_first_order`): better resolved, slightly low for a smooth interval effect. "
               if any("s1_binned" in (S["sobol"].get(md["key"]) or {}) for md in S["metrics"]) else "")
            + f"All indices are conditional on the other {sum(1 for x in S['entries'].values() if x.get('pinned') is None) - len(S['top'])} "
            "free entries at nominal.", ""]
    return out


def _section_ranking(data: Mapping) -> list[str]:
    R = data.get("ranking")
    out = ["## 5. Controller rank stability across worlds", "", '<a id="ranking"></a>', "",
           "Performance is conditional on the assumed thermal dynamics (heater, crosstalk and aggressor poles) until "
           "Q1 (the heater step response at the lock point, 07 section 5) is answered.", ""]
    if not R:
        return out + ["Not computed yet: run `python -m mrm.validation.report --recompute ranking`.", ""]
    src = R["meta"].get("source", {})
    out += [f"Source: `mrm.sim.experiment` kind compare ({_md(json.dumps(src, sort_keys=True))}); wall "
            f"{_num(R['meta'].get('wall_s'))} s. For each rank metric (lower is better), Kendall's tau-b between "
            "every world's ranking (Latin-hypercube worlds and named corners) and the nominal ranking; 'defined' is "
            "the fraction of worlds where tau exists (a ranking of all ties has none).", "",
            "| metric | nominal order (best first) | median order | tau mean | tau median | tau min | same order | "
            "defined |", "|---|---|---|---|---|---|---|---|"]
    for k, rs in (R.get("rank_stability") or {}).items():
        out.append(f"| {k} | {' < '.join(rs.get('nominal_order') or [])} | {' < '.join(rs.get('median_order') or [])} | "
                   f"{_num(rs.get('tau_mean'), 3)} | {_num(rs.get('tau_median'), 3)} | {_num(rs.get('tau_min'), 3)} | "
                   f"{_num(rs.get('frac_same_order'), 3)} | {_num(rs.get('frac_defined'), 3)} |")
    out += ["", "| controller | nominal | worst named corner | " + " | ".join(
        f"{k} median [p5, p95]" for k in (R.get("rank_metrics") or [])) + " | supply-limited worlds |",
            "|---|---|---|" + "---|" * len(R.get("rank_metrics") or []) + "---|"]
    for lab, c in R["controllers"].items():
        nom = c.get("nominal") or {}
        q = c.get("quantiles") or {}
        cells = []
        for k in R.get("rank_metrics") or []:
            qq = q.get(k) or {}
            cells.append(f"{_num(qq.get('p50'))} [{_num(qq.get('p5'))}, {_num(qq.get('p95'))}]")
        w = c.get("worst_corner_overall") or {}
        sl = (q.get("warm_start_supply_limited") or {}).get("mean")
        out.append(f"| {lab} | " + _md(", ".join(f"{k} {_num(nom.get(k))}" for k in (R.get('rank_metrics') or [])))
                   + f" | {_md(w.get('corner'))}: {_md(w.get('why'))} | " + " | ".join(cells) + f" | {_num(sl, 3)} |")
    out += ["", "A ranking claim holds only if it holds over the set (07 4.8). The worlds mix controller robustness "
            "with warm-start failures (supply-limited worlds start with rings the bank cannot hold; experiments "
            "handoff 9.7 and 12).", ""]
    return out


def _leaf_generations(path: str, cfg, derivations: Mapping, seen: frozenset = frozenset()) -> dict[str, set]:
    """{generation: {leaf paths}} of the config leaves a derived quantity rests on (recursively)."""
    if path in seen:
        return {}
    seen = seen | {path}
    lf = cfg.leaves.get(path)
    if lf is not None and not lf.is_free:
        return {str(lf.generation or "none"): {path}}
    d = derivations.get(path)
    out: dict[str, set] = {}
    if d is None:
        return out
    for i in d.inputs:
        for g, ps in _leaf_generations(i, cfg, derivations, seen).items():
            out.setdefault(g, set()).update(ps)
    return out


def _section_generation(cfg, d) -> list[str]:
    from mrm.params.derive import effective_provenance
    der = d.params.derivations
    out = ["## 6. Generation mix of the derived quantities", "", '<a id="generation"></a>', "",
           f"Preset `{cfg.name}`: " + _md((cfg.meta or {}).get("generation_mix", "(no meta.generation_mix)")), "",
           "Each derived quantity (relations of 07 P0-3 and the diagnostics of `derive()`) with the generation of "
           "every config leaf it rests on (recursively) and its weakest input provenance (07 P1-14).", "",
           "| quantity | generations (leaves) | weakest provenance |", "|---|---|---|"]
    items = [(p, [p]) for p in sorted(der) if not p.startswith("derived.timing")]
    items += [(f"diagnostic `{k}`", list(d.inputs.get(k) or [])) for k in d.diagnostics if d.inputs.get(k)]
    for name, inputs in items:
        gens: dict[str, set] = {}
        for i in inputs:
            for g, ps in _leaf_generations(i, cfg, der).items():
                gens.setdefault(g, set()).update(ps)
        txt = "; ".join(f"**{g}**: " + ", ".join(sorted(p[5:] if p.startswith("ring.") else p for p in ps))
                        for g, ps in sorted(gens.items()))
        prov = (effective_provenance(name, cfg, der) if not name.startswith("diagnostic")
                else d.provenance.get(name.split("`")[1], ""))
        out.append(f"| {('`' + name + '`') if not name.startswith('diagnostic') else name} | {_md(txt)} | {prov} |")
    counts: dict[tuple[str, str], int] = {}
    for p, lf in cfg.leaves.items():
        if lf.is_free:
            continue
        counts[(str(lf.generation or "none"), lf.provenance)] = counts.get((str(lf.generation or "none"),
                                                                           lf.provenance), 0) + 1
    gens = sorted({g for g, _ in counts})
    provs = sorted({p for _, p in counts})
    out += ["", "Leaves by generation and provenance:", "", "| generation | " + " | ".join(provs) + " | total |",
            "|---|" + "---|" * (len(provs) + 1)]
    for g in gens:
        row = [counts.get((g, p), 0) for p in provs]
        out.append(f"| {g} | " + " | ".join(str(x) for x in row) + f" | {sum(row)} |")
    out.append("")
    return out


def _section_textbook(cfg) -> list[str]:
    from mrm.params.table import format_value
    out = ["## 7. Disagreements with the textbook", "", '<a id="textbook"></a>', "",
           f"Generated from the `textbook:` and `disagrees:` fields of preset `{cfg.name}` (07 P1-14). Where the "
           "leaf carries a `textbook` alternative, its value is shown (select it with `--alt <path>=textbook`).", "",
           "| key | value here | textbook field | textbook alternative | disagreement |", "|---|---|---|---|---|"]
    n = 0
    for p, lf in sorted(cfg.leaves.items()):
        if not (lf.textbook or lf.disagrees):
            continue
        n += 1
        alt = lf.alternatives.get("textbook") if lf.alternatives else None
        altv = format_value(alt.get("value")) if isinstance(alt, Mapping) and "value" in alt else ""
        val = "(derived)" if lf.is_free else format_value(lf.value)
        out.append(f"| `{p}` | {_md(val)} | {_md(lf.textbook)} | {_md(altv)} | {_md(lf.disagrees)} |")
    out += ["", f"{n} leaves disagree with the textbook.", ""]
    return out


def _contamination() -> list[str]:
    text = (tio.TARGETS_DIR / "README.md").read_text()
    i = text.find("## Contamination warnings")
    body = text[i:].split("\n", 1)[1].strip() if i >= 0 else "(section not found in the targets README)"
    return ["## Contamination warnings", "", '<a id="contamination"></a>', "",
            "Copied from `mrm/validation/targets/README.md`: research-report fits that used held-out data; their "
            "results must not become defaults or priors.", "", body, ""]


def render(data: Mapping, *, preset: str, g25_preset: str | None = None, stamp: Mapping | None = None,
           doc_text: str | None = None, charts: bool = True, chart_dir: Path | None = None) -> str:
    """The Markdown of docs/VALIDATION.md from the cached ``data`` and the live preset and target files.

    ``stamp``: {date, commit, dirty} to print (default: now and the current git state); ``doc_text``: the current
    report, from which the frozen-targets block is copied verbatim (default: docs/VALIDATION.md); ``charts``:
    write the SVG charts (False only checks the text); ``chart_dir``: where they go (default
    docs/validation/<preset>/)."""
    from mrm.params.derive import derive
    g25 = companion_g25(preset, g25_preset)
    presets = {"G26": preset, "G25": g25}
    ddir = chart_dir or data_dir(preset)
    doc_text = DOC_PATH.read_text(encoding="utf-8") if doc_text is None else doc_text
    block = _frozen_block(doc_text)
    if stamp is None:
        g = _git()
        stamp = {"date": _dt.date.today().isoformat(), "commit": g["commit"], "dirty": g["dirty"]}
    st = {"preset": preset, "g25_preset": g25, "date": stamp.get("date"), "commit": stamp.get("commit"),
          "dirty": stamp.get("dirty")}
    cfg = _load_cfg(preset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = derive(cfg)
    herr = tio.header_errors(None)
    out = ["# Validation report", "", block, "",
           f"<!-- report-stamp {json.dumps(st, sort_keys=True)} -->",
           f"Generated by `python -m mrm.validation.report --preset {preset}` (also `mrm report`) on {st['date']} "
           f"from commit `{st['commit']}`{' with uncommitted changes' if st['dirty'] else ''}. Preset `{preset}` "
           f"(config hash `{cfg.hash()[:16]}`; files {', '.join(f'`{f}`' for f in cfg.files)}); G25 bench targets on "
           f"`{g25}`. Do not edit by hand: every section below is generated (07 P1-23, 4.9).", "",
           "The two held-out target files above were frozen before any calibration fit or controller comparison "
           "(07 P0-5); the block is owned by the target split and copied verbatim. The hashes are of the file "
           "bytes with CRLF read as LF; any other edit is a re-freeze (`mrm/validation/targets/README.md`). "
           "Frozen-hash check now: " + ("**OK**" if not herr else "**FAILED**: " + "; ".join(herr)) + ".", "",
           "Computed parts (`docs/validation/" + preset + "/report_data.json`; `--fast` re-renders from it):", "",
           "| part | computed (UTC) | commit | wall (s) | peak RSS (GB) | presets (config hash) |",
           "|---|---|---|---|---|---|"]
    out += [_part_line(p, data.get(p)) for p in PARTS]
    stale = _staleness(data, presets)
    out += [""]
    if stale:
        out += ["**Stale parts** (recompute them): " + "; ".join(stale) + ".", ""]
    out += ["Sections: [1 targets](#targets) . [2 closure](#closure) . [3 identifiability](#identifiability) . "
            "[4 uncertainty](#uncertainty) . [5 ranking](#ranking) . [6 generation mix](#generation) . "
            "[7 textbook](#textbook) . [contamination warnings](#contamination)", ""]
    out += _section_targets(data, presets, ddir, charts)
    out += _section_closure(cfg, d, data)
    out += _section_identifiability(cfg, data, preset, ddir, charts)
    out += _section_uncertainty(data, ddir, charts)
    out += _section_ranking(data)
    out += _section_generation(cfg, d)
    out += _section_textbook(cfg)
    out += _contamination()
    return "\n".join(out).rstrip("\n") + "\n"


# ================================================================================================ driver
def generate(preset: str = "g26", *, g25_preset: str | None = None, recompute: Sequence[str] | None = None,
             out: str | Path | None = None, n_base: int = 128, compare_worlds: int = 256,
             compare_run: str | Path | None = None, n_threads: int = 4, stamp: Mapping | None = None,
             log: Callable[[str], None] = _log_default) -> Path:
    """Recompute the parts in ``recompute`` (default: all of :data:`PARTS`; ``()`` = none, the fast mode), save
    them to ``docs/validation/<preset>/report_data.json`` and write the report to ``out`` (default
    docs/VALIDATION.md)."""
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
        save_data(preset, data)                            # after every part: a later failure keeps the earlier
    out = Path(out) if out is not None else DOC_PATH
    doc_text = out.read_text(encoding="utf-8") if out.exists() else DOC_PATH.read_text(encoding="utf-8")
    text = render(data, preset=preset, g25_preset=g25_preset, stamp=stamp, doc_text=doc_text)
    out.write_text(text, encoding="utf-8")
    log(f"report: wrote {out}")
    return out


def build_parser(p: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    p = p or argparse.ArgumentParser(prog="python -m mrm.validation.report", description=__doc__.split("\n\n")[0])
    p.add_argument("--preset", default="g26", help="preset the report is about (g26, g26_calibrated, ...)")
    p.add_argument("--g25-preset", default=None, help="G25 bench preset (default: the companion of --preset)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--fast", action="store_true", help="recompute nothing; re-render from the cached data")
    g.add_argument("--recompute", default=None,
                   help=f"comma-separated parts to recompute (others from the cache): {','.join(PARTS)}")
    p.add_argument("--check", action="store_true",
                   help="fast render in memory and compare with the report file (keeps its stamp); exit 1 if stale")
    p.add_argument("--out", default=None, help="report file (default docs/VALIDATION.md)")
    p.add_argument("--sobol-n", type=int, default=128, help="Sobol base rows (worlds = n (top + 2))")
    p.add_argument("--compare-worlds", type=int, default=256, help="Latin-hypercube worlds of the compare (07 4.8)")
    p.add_argument("--compare-run", default=None, help="reuse an existing compare run directory for section 5")
    p.add_argument("--threads", type=int, default=4, help="kernel threads")
    return p


def check(preset: str | None = None, out: str | Path | None = None) -> list[str]:
    """Problems that make the report file stale: text that differs from a fast re-render with the file's own stamp,
    missing charts, stale cached parts. Empty when current."""
    out = Path(out) if out is not None else DOC_PATH
    text = out.read_text(encoding="utf-8")
    st = read_stamp(text)
    if st is None:
        return [f"{out} has no report stamp: it was not generated by mrm.validation.report"]
    preset = preset or st["preset"]
    data = load_data(preset)
    probs = []
    new = render(data, preset=preset, g25_preset=st.get("g25_preset"), stamp=st, doc_text=text, charts=False,
                 chart_dir=data_dir(preset))
    if new != text:
        a, b = text.splitlines(), new.splitlines()
        i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        probs.append(f"{out} differs from a re-render at line {i + 1}: run python -m mrm.validation.report --fast "
                     f"--preset {preset}")
    for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", text):
        if not (out.parent / m.group(1)).exists():
            probs.append(f"missing chart {m.group(1)}")
    probs += _staleness(data, {"G26": preset, "G25": companion_g25(preset, st.get("g25_preset"))})
    return probs


def run_args(a: argparse.Namespace) -> int:
    """Run the parsed arguments of :func:`build_parser` (shared by ``python -m`` and ``mrm report``)."""
    if a.check:
        probs = check(a.preset if a.preset != "g26" else None, a.out)
        for p in probs:
            print(p, file=sys.stderr)
        return 1 if probs else 0
    rec = () if a.fast else (None if a.recompute is None else tuple(x.strip() for x in a.recompute.split(",") if x))
    try:
        generate(a.preset, g25_preset=a.g25_preset, recompute=rec, out=a.out, n_base=a.sobol_n,
                 compare_worlds=a.compare_worlds, compare_run=a.compare_run, n_threads=a.threads)
    except Exception as e:                                              # noqa: BLE001
        print(f"mrm report: {e.__class__.__name__}: {e}", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
