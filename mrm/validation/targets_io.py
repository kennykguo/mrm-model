# load declared targets, check their separation, and score replica predictions.
# the target files have four roles — calibration values may be fitted, validation values are held out, bounds are necessary operating conditions, and excluded values are neither fitted nor scored.
# `load_all` checks all four.
# calibration code must use `load_calibration_targets` only.
# held-out files have frozen hashes in docs/VALIDATION.md.
# the isolation scan checks for accidental routes from calibration code to held-out data.
# this is a static guard, not a security sandbox.
# see targets/README.md for fields.
# `evaluate` applies the target's declared tolerance in the source's printed unit.
# sigma describes measurement uncertainty and supplies z scores.
# it does not replace the acceptance tolerance.
# non-active entries can be evaluated without counting toward the scored result.
# `to_si` is available for replica calculations.
# it handles linear scales, absolute degC, logarithmic power ratios, and dBm separately.
# run `python -m mrm.validation.targets_io check` to check the files and split.
# `header` prints a new hash block and is used only for an intentional re-freeze.
from __future__ import annotations

import ast
import dataclasses
import hashlib
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mrm.units import Q_E, T0_K

__all__ = [
    "TARGETS_DIR", "DOC_VALIDATION", "ROLES", "TARGET_ROLES", "HELD_OUT_ROLES", "SCHEMA_ID", "REPLICA_NAMES",
    "Target", "ExcludedItem", "Evaluation", "TargetSchemaError",
    "targets_path", "read_raw", "schema_errors", "load_targets", "load_calibration_targets", "load_excluded",
    "load_all", "split_errors", "file_sha256", "frozen_header_block", "read_frozen_header", "header_errors",
    "held_out_reference_violations", "parse_unit", "unit_factor", "to_si", "rectified_sine", "evaluate", "main",
]

# locations
PACKAGE_DIR = Path(__file__).resolve().parents[1]          # .../mrm
REPO_DIR = PACKAGE_DIR.parent                              # repository root
TARGETS_DIR = Path(__file__).resolve().parent / "targets"
DOC_VALIDATION = REPO_DIR / "docs" / "VALIDATION.md"
CALIBRATION_DIR = PACKAGE_DIR / "calibration"
SOURCES_YAML = PACKAGE_DIR / "params" / "sources.yaml"

# closed sets
SCHEMA_ID = "mrm.validation.targets/1"              # the ``schema:`` value of every target file
ROLES = ("calibration", "validation", "bounds", "excluded")     # one file per role - <role>.yaml
TARGET_ROLES = ("calibration", "validation", "bounds")          # files whose entries are Targets
HELD_OUT_ROLES = ("validation", "bounds")                       # files that mrm/calibration must never read
FROZEN_ROLES = HELD_OUT_ROLES                     # files whose sha256 is in the docs/VALIDATION.md header
ID_PREFIX = {"calibration": "CAL-", "validation": "VAL-", "bounds": "BND-", "excluded": "EXC-"}
ID_PATTERN = re.compile(r"^(CAL|VAL|BND|EXC)-(\d+)([a-z]?)$")   # for example CAL-3a - prefix, number, sub-letter

# Provenance words, ranked from strongest to weakest (docs/ARCHITECTURE.md, 'Record where every number came from'), then derived, override and private.
# GENERATIONS - the product generation or source family of a number.
PROVENANCE_WORDS = ("printed", "read-off", "calibrated", "inferred", "solver", "literature", "assumed",
                    "derived", "override", "private")
GENERATIONS = ("G25", "G26", "EVK100", "L20", "M1000", "Guide", "OCP", "OCI", "GF", "PAT", "none")
STATUSES = ("active", "deferred", "unverified", "cross_check")
KINDS = ("scalar", "interval", "curve", "set", "parametric", "contour_offsets", "inequality", "band")
TOLERANCE_KINDS = ("abs", "rel", "interval", "inequality", "band", "rms_abs", "contour_asymmetry", "none",
                   "pending")
EXCLUDED_CATEGORIES = ("excluded", "closure_only", "hypothesis_only", "scenario_input", "prior_only",
                       "not_a_limit", "controller_specific")
# the measurement conditions an entry may state (temperature, drive, bias, optical power, port, loop, ...).
CONDITION_KEYS = frozenset({
    "t_c", "t_reference", "heater_v", "heater_code", "bias_v", "p_in_dbm", "port", "loop", "detuning",
    "lambda_nm", "window_s", "period_s", "grid",
})
# every target entry states the temperature (t_c), the heater drive (heater_v or heater_code), the bias, the input power and the port.
# null means "not printed" or "not applicable", but the key must be present, so that a missing condition is a decision, not an omission.
REQUIRED_CONDITION_KEYS = ("t_c", "bias_v", "p_in_dbm", "port")
HEATER_CONDITION_KEYS = ("heater_v", "heater_code")
# a target read off a figure (provenance read-off) records who digitized it, the image, the method and the crop.
# "not recorded" is an allowed, honest value.
READOFF_KEYS = ("by", "image", "method", "crop")
PORTS = ("through", "drop", "output", "none")
LOOPS = ("open", "closed", "none")
T_REFERENCES = ("stage", "tec", "ring", "on_die_sensor", "aggressor_trajectory", "unknown")

# the replica protocol functions that mrm/validation/replicas.py must provide, one per published measurement protocol.
# each takes the plant configuration and a Target and returns a prediction shaped like Target.value, in Target.unit (see evaluate()).
REPLICA_NAMES = (
    "notch_er_vs_heater_v", "loaded_q_from_3db_width", "tuning_range_at_v", "relOMA_map",
    "monitor_current_at_lock", "stabilizer_code_vs_T", "apparent_shift_fixed_code", "bistability_sweep",
    "heater_power_per_fsr", "resonance_vs_heater_v", "resonance_shift_vs_bias", "cold_resonance_offset",
    "eo_bandwidth_at_bias", "pd_responsivity", "laser_line_statistics", "aggressor_trajectory",
    "mode_solve_published_geometry", "self_heating_time_constant", "thermal_step_published_stack",
    "adjacent_order_penalty", "order_hop_margin", "aggressor_feasibility",
)

REQUIRED_TARGET_FIELDS = ("id", "quantity", "description", "kind", "value", "unit", "sigma", "conditions",
                          "generation", "source", "locator", "provenance", "replica", "tolerance",
                          "tolerance_basis", "data_points", "status", "verification")
# the free-text field that says what an entry is for - what it identifies, tests or bounds
PURPOSE_FIELD = {"calibration": "identifies", "validation": "tests", "bounds": "bounds"}
REQUIRED_EXCLUDED_FIELDS = ("id", "quantity", "value", "unit", "generation", "source", "locator", "provenance",
                            "category", "origin", "reason", "data_points")
# Closed field sets - a misspelt optional field (`use:` for `uses:`, `blocked-by:`) would otherwise be ignored silently and drop a split check.
OPTIONAL_TARGET_FIELDS = ("components", "population", "conditions_note", "readoff", "rechecked", "replica_args",
                          "replica_args_note", "fit", "identifies", "tests", "bounds", "uses", "blocked_by",
                          "absolute_positions_nm", "inputs_needed", "caveats")
OPTIONAL_EXCLUDED_FIELDS = ("used_in", "description", "readoff")
# Allowed keys per tolerance kind (required ones first) - an unknown key such as `hardvalue` is an error.
TOLERANCE_KEYS = {
    "abs": ("value",), "rel": ("value",), "rms_abs": ("value",), "interval": ("lo", "hi"),
    "inequality": ("op", "value", "hard_value"), "band": ("lo", "hi", "slack", "either_sign"),
    "contour_asymmetry": ("abs", "require"), "none": (), "pending": (),
}
# the required keys per tolerance kind (the leading keys of TOLERANCE_KEYS).
_REQUIRED_TOLERANCE_KEYS = {
    "abs": ("value",), "rel": ("value",), "rms_abs": ("value",), "interval": ("lo", "hi"),
    "inequality": ("op", "value"), "band": ("lo", "hi", "slack"), "contour_asymmetry": ("abs", "require"),
    "none": (), "pending": (),
}

# units
# Unit tokens of the unit-suffix grammar, kept identical to mrm/params/units.py (tests/closure/test_targets_schema.py cross-checks the two).
# the values are the SI scales of exact SI prefixes (definitions, not measurements) - eV uses the exact elementary charge.
# "codes" (converter counts) and the targets-only "1" are dimensionless.
# "c" (absolute degC, affine), "db" and "dbm" are special and allowed only alone - a temperature rate or difference uses "k" (for example "k_per_s").
_UNIT_FACTORS: dict[str, float] = {
    "pm": 1e-12, "nm": 1e-9, "um": 1e-6, "mm": 1e-3, "cm": 1e-2, "m": 1.0,
    "ps": 1e-12, "ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0,
    "hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "thz": 1e12, "gbps": 1e9,
    "w": 1.0, "mw": 1e-3, "uw": 1e-6, "nw": 1e-9,
    "a": 1.0, "ma": 1e-3, "ua": 1e-6, "na": 1e-9, "pa": 1e-12,
    "v": 1.0, "mv": 1e-3, "ohm": 1.0, "f": 1.0, "pf": 1e-12, "ff": 1e-15,
    "k": 1.0, "ev": Q_E, "j": 1.0, "pj": 1e-12, "fj": 1e-15,
    "pct": 1e-2, "frac": 1.0, "ppm": 1e-6, "codes": 1.0, "lsb": 1.0,
}
_SPECIAL_UNITS = ("c", "db", "dbm")           # not linear - allowed only alone, converted by to_si()
_DIMENSIONLESS = "1"
# values of the unit field that are not units.
# "mixed" - a set whose keys end in their own units (kind set only).
# "none" - no unit (excluded.yaml only).
_NON_UNITS = ("mixed", "none")
_LN10_OVER_10 = math.log(10.0) / 10.0       # dB -> natural-log power ratio (definition of the decibel)
_TOKEN_RE = re.compile(r"^([a-z]+?)([23])?$")  # a unit token with an optional exponent digit, for example "um2"


# raised when a target file violates the schema or the calibration / held-out split.
class TargetSchemaError(ValueError):
    pass


# sI scale of one unit token (`"um2"` -> 1e-12), or None for an unknown token.
def _token_scale(tok: str) -> float | None:
    match = _TOKEN_RE.match(tok)
    if not match or match.group(1) not in _UNIT_FACTORS:
        return None
    return _UNIT_FACTORS[match.group(1)] ** (int(match.group(2)) if match.group(2) else 1)


# split a unit string of the suffix grammar into numerator and denominator factors.
# Grammar (as mrm/params/units.py) — factors are tokens with an optional exponent digit (`um2`), optionally preceded by `rt` (square root).
# every factor after the first `per` is in the denominator.
# `"pm_per_k"` -> (("pm",), ("k",)).
# `"pa_per_rt_hz"` -> (("pa",), ("rt_hz",)).
# `"db"` -> (("db",), ()).
# `"1"` (dimensionless), `"mixed"` and `"none"` parse to themselves.
# raises ValueError otherwise.
def parse_unit(unit: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(unit, str) or not unit:
        raise ValueError(f"unit must be a non-empty string, got {unit!r}")
    if unit in _SPECIAL_UNITS or unit in _NON_UNITS or unit == _DIMENSIONLESS:
        return (unit,), ()
    raw = unit.split("_")
    numerator: list[str] = []
    denominator: list[str] = []
    target = numerator
    i = 0
    while i < len(raw):
        token = raw[i]
        if token == "per":
            if i == len(raw) - 1 or i == 0:
                raise ValueError(f"unit {unit!r}: 'per' needs factors on both sides")
            target = denominator
            i += 1
            continue
        prefix = ""
        if token == "rt":
            if i == len(raw) - 1:
                raise ValueError(f"unit {unit!r}: 'rt' must precede a unit token")
            prefix, i = "rt_", i + 1
            token = raw[i]
        if token in _SPECIAL_UNITS:
            raise ValueError(f"unit {unit!r}: {token!r} is only allowed alone (use 'k' for a temperature rate)")
        if _token_scale(token) is None:
            raise ValueError(f"unit {unit!r}: unknown token {token!r}")
        target.append(prefix + token)
        i += 1
    if not numerator:
        raise ValueError(f"unit {unit!r}: no numerator factor")
    return tuple(numerator), tuple(denominator)


# sI scale of a linear unit — value_SI = factor * value (`"1"` -> 1).
# raises ValueError for the special units `c` (affine), `db` and `dbm` (logarithmic. use `to_si`) and for `mixed`/`none`.
def unit_factor(unit: str) -> float:
    if unit in _SPECIAL_UNITS:
        raise ValueError(f"{unit!r} is not linear; use to_si()")
    if unit in _NON_UNITS:
        raise ValueError(f"{unit!r} is not a unit")
    if unit == _DIMENSIONLESS:
        return 1.0
    numerator, denominator = parse_unit(unit)

    def _scale(tok: str) -> float:
        if tok.startswith("rt_"):
            return math.sqrt(_token_scale(tok[3:]))
        return _token_scale(tok)

    factor = 1.0
    for t in numerator:
        factor *= _scale(t)
    for t in denominator:
        factor /= _scale(t)
    return factor


# convert a value (scalar, sequence or ndarray) from `unit` to SI, with the conventions of mrm.params.units.
# Equations — linear units, x_SI = scale * x.
# absolute degC (`c`), T_K = T_C + 273.15.
# `db` becomes the natural-log power ratio x * ln(10) / 10.
# `dbm` becomes watts, P_W = 1e-3 * 10**(P_dBm / 10).
# Targets are compared in their printed unit (`evaluate`).
# this is a convenience for replicas.
def to_si(value: Any, unit: str) -> Any:
    arr = np.asarray(value, dtype=float)
    if unit == "db":
        out = arr * _LN10_OVER_10
    elif unit == "dbm":
        out = 1e-3 * 10.0 ** (arr / 10.0)
    elif unit == "c":
        out = arr + T0_K
    else:
        out = arr * unit_factor(unit)
    return out.item() if out.ndim == 0 else out


# records
# one calibration, validation or bound entry.
# field meanings — README.md in mrm/validation/targets.
# `value`/`sigma` are in `unit` as printed by the source.
# `raw` keeps the full YAML mapping, including the free-text fields (verification, identifies/tests/bounds, notes).
@dataclasses.dataclass(frozen=True)
class Target:

    id: str
    role: str
    quantity: str
    description: str
    kind: str
    value: Any
    unit: str
    sigma: Any
    conditions: Mapping[str, Any]
    generation: str
    source: tuple[str, ...]
    locator: str
    provenance: str
    replica: str
    replica_args: Mapping[str, Any]
    tolerance: Mapping[str, Any]
    status: str
    data_points: tuple[str, ...]
    uses: tuple[str, ...]
    rechecked: bool | None
    raw: Mapping[str, Any]

    # what the entry identifies (calibration), tests (validation) or bounds (bounds).
    @property
    def purpose(self) -> str:
        return str(self.raw.get(PURPOSE_FIELD[self.role], ""))

    # true if a comparison counts as a pass or a failure (status active and a real tolerance).
    @property
    def scored(self) -> bool:
        return self.status == "active" and self.tolerance.get("kind") not in ("none", "pending")

    # for calibration entries — whether the value is fitted (CAL-6 is not).
    @property
    def fit(self) -> bool:
        return bool(self.raw.get("fit", self.role == "calibration"))


# one entry of excluded.yaml — a published number that is not fitted and not scored, with its reason.
@dataclasses.dataclass(frozen=True)
class ExcludedItem:

    id: str
    quantity: str
    value: Any
    unit: str
    generation: str
    source: tuple[str, ...]
    locator: str
    provenance: str
    category: str
    origin: str
    reason: str
    data_points: tuple[str, ...]
    raw: Mapping[str, Any]


# comparison result with a target's acceptance rule.
# `level` is pass, soft_fail, fail, or not_scored.
# soft_fail lies between an inequality's preferred and hard limits.
# `scored` is false for non-active entries, even when their pass result can be calculated.
# `max_abs_residual` reports the miss in the target's unit.
# for an interval, band, or inequality, it is zero inside the allowed region.
# `z` is the largest sigma-normalized residual when uncertainty aligns with the prediction.
@dataclasses.dataclass(frozen=True)
class Evaluation:

    target_id: str
    scored: bool
    passed: bool | None
    level: str
    max_abs_residual: float | None
    z: float | None
    detail: str


# reading
# path of the YAML file of one role ("calibration", "validation", "bounds", "excluded").
def targets_path(role: str, root: Path | None = None) -> Path:
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    return (Path(root) if root is not None else TARGETS_DIR) / f"{role}.yaml"


# parse one target file with `yaml.safe_load` and return the document mapping.
def read_raw(role: str, root: Path | None = None) -> dict[str, Any]:
    with targets_path(role, root).open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if not isinstance(doc, dict):
        raise TargetSchemaError(f"{role}.yaml: top level must be a mapping")
    return doc


# a YAML string-or-list field as a tuple of strings (None -> ()).
def _as_tuple(x: Any) -> tuple[str, ...]:
    if x is None:
        return ()
    if isinstance(x, str):
        return (x,)
    return tuple(str(v) for v in x)


# true for a finite int or float (a bool is not a number here).
def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


# flatten the numeric leaves of a value or sigma (scalars, lists, mappings).
def _numbers(x: Any) -> list[float]:
    if _is_number(x):
        return [float(x)]
    if isinstance(x, Mapping):
        out: list[float] = []
        for v in x.values():
            out.extend(_numbers(v))
        return out
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out.extend(_numbers(v))
        return out
    return []


# errors in the shape of `value` (and `sigma`) for the entry kind.
# only a deferred entry may omit value.
def _check_value_shape(tid: str, kind: str, value: Any, sigma: Any, status: str) -> list[str]:
    errors: list[str] = []
    if value is None:
        if status != "deferred":
            errors.append(f"{tid}: value is null but status is {status!r} (only deferred entries may omit it)")
        return errors
    if kind in ("scalar", "inequality"):
        if not _is_number(value):
            errors.append(f"{tid}: kind {kind} needs a numeric value")
    elif kind in ("interval", "band"):
        if not (isinstance(value, list) and len(value) == 2 and all(_is_number(v) for v in value)
                and value[0] <= value[1]):
            errors.append(f"{tid}: kind {kind} needs value [lo, hi] with lo <= hi")
    elif kind == "curve":
        if not isinstance(value, Mapping) or not {"x", "y", "x_name", "x_unit"} <= set(value):
            errors.append(f"{tid}: curve needs value {{x_name, x_unit, x, y}}")
        else:
            x, y = value["x"], value["y"]
            if not (isinstance(x, list) and isinstance(y, list) and len(x) == len(y) and len(x) > 0):
                errors.append(f"{tid}: curve x and y must be equal-length non-empty lists")
            elif not all(_is_number(v) for v in x + y):
                errors.append(f"{tid}: curve x and y must be numeric")
            elif any(b <= a for a, b in zip(x, x[1:])):
                errors.append(f"{tid}: curve x must be strictly increasing")
            if isinstance(sigma, list) and isinstance(y, list) and len(sigma) != len(y):
                errors.append(f"{tid}: curve sigma list must match y")
    elif kind == "set":
        if not isinstance(value, Mapping) or not all(_is_number(v) for v in value.values()):
            errors.append(f"{tid}: set needs a mapping of named numbers")
        if isinstance(sigma, Mapping) and isinstance(value, Mapping) and set(sigma) != set(value):
            errors.append(f"{tid}: set sigma keys must match value keys")
    elif kind == "parametric":
        if not isinstance(value, Mapping) or "form" not in value or "params" not in value:
            errors.append(f"{tid}: parametric needs value {{form, params}}")
        elif value["form"] != "rectified_sine":
            errors.append(f"{tid}: unknown parametric form {value['form']!r}")
        elif not {"t_max_c", "delta_t_k", "period_s"} <= set(value["params"]):
            errors.append(f"{tid}: rectified_sine needs params t_max_c, delta_t_k, period_s")
    elif kind == "contour_offsets":
        keys = ("levels_db", "below_peak_k", "above_peak_k")
        ok = (isinstance(value, Mapping) and set(keys) <= set(value)
              and all(isinstance(value[k], list) and value[k] and all(_is_number(v) for v in value[k]) for k in keys)
              and len(value["levels_db"]) == len(value["below_peak_k"]) == len(value["above_peak_k"]))
        if not ok:
            errors.append(f"{tid}: contour_offsets needs equal-length numeric levels_db, below_peak_k, above_peak_k")
    return errors


# errors in one tolerance mapping — kind, required and unknown keys, numeric values, ordering and sign rules.
def _check_tolerance(tid: str, tol: Any, status: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(tol, Mapping) or tol.get("kind") not in TOLERANCE_KINDS:
        return [f"{tid}: tolerance must be a mapping with kind in {TOLERANCE_KINDS}"]
    kind = tol["kind"]
    for key in _REQUIRED_TOLERANCE_KEYS[kind]:
        if key not in tol:
            errors.append(f"{tid}: tolerance kind {kind} needs {key!r}")
    unknown = sorted(set(tol) - {"kind"} - set(TOLERANCE_KEYS[kind]))
    if unknown:
        errors.append(f"{tid}: tolerance kind {kind} does not take {unknown}")
    for key in ("value", "lo", "hi", "slack", "abs", "hard_value"):
        if key in tol and key != "require" and not _is_number(tol[key]):
            errors.append(f"{tid}: tolerance {key} must be numeric")
    if kind in ("abs", "rel", "rms_abs") and _is_number(tol.get("value")) and tol["value"] <= 0:
        errors.append(f"{tid}: tolerance value must be positive")
    if kind in ("interval", "band") and _is_number(tol.get("lo")) and _is_number(tol.get("hi")) and tol["lo"] > tol["hi"]:
        errors.append(f"{tid}: tolerance lo must not exceed hi")
    if kind == "band" and _is_number(tol.get("slack")) and tol["slack"] < 0:
        errors.append(f"{tid}: band slack must be non-negative")
    if "either_sign" in tol and (kind != "band" or not isinstance(tol["either_sign"], bool)):
        errors.append(f"{tid}: either_sign is a boolean allowed only for kind band")
    if kind == "inequality" and tol.get("op") not in (">=", "<="):
        errors.append(f"{tid}: inequality op must be '>=' or '<='")
    if kind == "inequality" and "hard_value" in tol and _is_number(tol.get("value")) and _is_number(tol["hard_value"]):
        worse_ok = tol["hard_value"] <= tol["value"] if tol.get("op") == ">=" else tol["hard_value"] >= tol["value"]
        if not worse_ok:
            errors.append(f"{tid}: hard_value must be on the failing side of value")
    if kind == "contour_asymmetry" and tol.get("require") not in ("cold_side_wider", "hot_side_wider"):
        errors.append(f"{tid}: contour_asymmetry require must be cold_side_wider or hot_side_wider")
    if kind == "pending" and status != "deferred":
        errors.append(f"{tid}: tolerance 'pending' is allowed only for deferred entries")
    return errors


# a quantity name that ends in a unit token must end in its own unit (the unit-suffix grammar).
def _quantity_unit_errors(tid: str, quantity: str, unit: str) -> list[str]:
    if unit in ("1", "mixed", "none") or quantity.endswith("_" + unit):
        return []
    last = quantity.rsplit("_", 1)[-1]
    if last in _UNIT_FACTORS or last in _SPECIAL_UNITS:
        return [f"{tid}: quantity {quantity!r} ends in unit token {last!r} but unit is {unit!r}"]
    return []


# check one parsed target file against the schema.
# returns a list of messages (empty when valid).
def schema_errors(role: str, doc: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if doc.get("schema") != SCHEMA_ID:
        errors.append(f"{role}: schema must be {SCHEMA_ID!r}")
    if doc.get("role") != role:
        errors.append(f"{role}: role field is {doc.get('role')!r}")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(doc.get("frozen_on", ""))):
        errors.append(f"{role}: frozen_on must be an ISO date string")
    list_key = "excluded" if role == "excluded" else "targets"
    entries = doc.get(list_key)
    if not isinstance(entries, list) or not entries:
        return errors + [f"{role}: {list_key} must be a non-empty list"]
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            errors.append(f"{role}[{i}]: entry must be a mapping")
            continue
        target_id = str(entry.get("id", f"{role}[{i}]"))
        required = REQUIRED_EXCLUDED_FIELDS if role == "excluded" else REQUIRED_TARGET_FIELDS
        for f in required:
            if f not in entry:
                errors.append(f"{target_id}: missing field {f!r}")
        known = set(required) | set(OPTIONAL_EXCLUDED_FIELDS if role == "excluded" else OPTIONAL_TARGET_FIELDS)
        unknown_fields = sorted(set(entry) - known)
        if unknown_fields:
            errors.append(f"{target_id}: unknown fields {unknown_fields} (misspelt?)")
        if not ID_PATTERN.match(target_id) or not target_id.startswith(ID_PREFIX[role]):
            errors.append(f"{target_id}: id must look like {ID_PREFIX[role]}<n>[a-z]")
        if target_id in seen:
            errors.append(f"{target_id}: duplicate id")
        seen.add(target_id)
        if entry.get("provenance") not in PROVENANCE_WORDS:
            errors.append(f"{target_id}: provenance {entry.get('provenance')!r} not in {PROVENANCE_WORDS}")
        if entry.get("generation") not in GENERATIONS:
            errors.append(f"{target_id}: generation {entry.get('generation')!r} not in {GENERATIONS}")
        if not _as_tuple(entry.get("source")):
            errors.append(f"{target_id}: source must name at least one source key")
        unit = entry.get("unit")
        try:
            parse_unit(unit)
        except ValueError as exc:
            errors.append(f"{target_id}: {exc}")
        data_points = entry.get("data_points")
        if not isinstance(data_points, list) or not all(isinstance(d, str) and "/" in d for d in data_points):
            errors.append(f"{target_id}: data_points must be a list of 'dataset/item' strings")
        elif len(set(data_points)) != len(data_points):
            errors.append(f"{target_id}: duplicate data_points")
        if role == "excluded":
            if entry.get("category") not in EXCLUDED_CATEGORIES:
                errors.append(f"{target_id}: category {entry.get('category')!r} not in {EXCLUDED_CATEGORIES}")
            if not str(entry.get("reason", "")).strip():
                errors.append(f"{target_id}: reason must not be empty")
            continue
        # calibration / validation / bounds entries
        if PURPOSE_FIELD[role] not in entry:
            errors.append(f"{target_id}: missing field {PURPOSE_FIELD[role]!r}")
        status = entry.get("status")
        if status not in STATUSES:
            errors.append(f"{target_id}: status {status!r} not in {STATUSES}")
        kind = entry.get("kind")
        if kind not in KINDS:
            errors.append(f"{target_id}: kind {kind!r} not in {KINDS}")
        else:
            errors.extend(_check_value_shape(target_id, kind, entry.get("value"), entry.get("sigma"), str(status)))
        if isinstance(unit, str):
            errors.extend(_quantity_unit_errors(target_id, str(entry.get("quantity", "")), unit))
        if unit == "none":
            errors.append(f"{target_id}: unit 'none' is allowed only in excluded.yaml")
        if unit == "mixed":
            val = entry.get("value")
            if kind != "set" or not isinstance(val, Mapping):
                errors.append(f"{target_id}: unit 'mixed' is allowed only for kind set")
            else:
                for key in val:
                    last = str(key).rsplit("_", 1)[-1]
                    if last not in _UNIT_FACTORS and last not in _SPECIAL_UNITS:
                        errors.append(f"{target_id}: with unit 'mixed' every key must end in a unit token ({key!r})")
        sigma = entry.get("sigma")
        if sigma is not None and (not _numbers(sigma) or any(s < 0 for s in _numbers(sigma))):
            errors.append(f"{target_id}: sigma must be null or non-negative numbers")
        conditions = entry.get("conditions")
        if not isinstance(conditions, Mapping):
            errors.append(f"{target_id}: conditions must be a mapping")
        else:
            bad = set(conditions) - CONDITION_KEYS
            if bad:
                errors.append(f"{target_id}: unknown condition keys {sorted(bad)}")
            if "port" in conditions and conditions["port"] not in PORTS:
                errors.append(f"{target_id}: port {conditions['port']!r} not in {PORTS}")
            if "loop" in conditions and conditions["loop"] not in LOOPS:
                errors.append(f"{target_id}: loop {conditions['loop']!r} not in {LOOPS}")
            if "t_reference" in conditions and conditions["t_reference"] not in T_REFERENCES:
                errors.append(f"{target_id}: t_reference {conditions['t_reference']!r} not in {T_REFERENCES}")
            missing = [k for k in REQUIRED_CONDITION_KEYS if k not in conditions]
            if not any(k in conditions for k in HEATER_CONDITION_KEYS):
                missing.append("heater_v|heater_code")
            if missing:
                errors.append(f"{target_id}: conditions lack {missing} (use null when not printed)")
        if entry.get("provenance") == "read-off":
            readoff = entry.get("readoff")
            if not isinstance(readoff, Mapping) or any(not str(readoff.get(k, "")).strip() for k in READOFF_KEYS):
                errors.append(f"{target_id}: a read-off entry needs readoff {{{', '.join(READOFF_KEYS)}}} (who digitized "
                            "it, the image, the method and the crop)")
        if entry.get("replica") not in REPLICA_NAMES:
            errors.append(f"{target_id}: replica {entry.get('replica')!r} not in REPLICA_NAMES")
        if not isinstance(entry.get("replica_args", {}), Mapping):
            errors.append(f"{target_id}: replica_args must be a mapping")
        errors.extend(_check_tolerance(target_id, entry.get("tolerance"), str(status)))
        if role == "bounds" and kind not in ("inequality", "band"):
            errors.append(f"{target_id}: a bound must be of kind inequality or band")
        if status == "deferred" and not entry.get("blocked_by"):
            errors.append(f"{target_id}: a deferred entry must list blocked_by")
        if role == "calibration" and status == "cross_check":
            errors.append(f"{target_id}: cross_check entries belong in validation.yaml")
        if not isinstance(entry.get("rechecked", True), bool):
            errors.append(f"{target_id}: rechecked must be true or false")
    return errors


# a `Target` from one YAML entry of a target file.
def _target_from_entry(role: str, e: Mapping[str, Any]) -> Target:
    return Target(
        id=str(e["id"]), role=role, quantity=str(e["quantity"]), description=str(e["description"]),
        kind=str(e["kind"]), value=e["value"], unit=str(e["unit"]), sigma=e.get("sigma"),
        conditions=dict(e.get("conditions") or {}), generation=str(e["generation"]),
        source=_as_tuple(e["source"]), locator=str(e["locator"]), provenance=str(e["provenance"]),
        replica=str(e["replica"]), replica_args=dict(e.get("replica_args") or {}),
        tolerance=dict(e["tolerance"]), status=str(e["status"]), data_points=_as_tuple(e["data_points"]),
        uses=_as_tuple(e.get("uses")), rechecked=e.get("rechecked"), raw=dict(e),
    )


# load the entries of one target file ("calibration", "validation" or "bounds").
# with `check` the file is validated with `schema_errors` and a `TargetSchemaError` lists every problem.
# Code under `mrm/calibration` must call this only with `role="calibration"` (or use `load_calibration_targets`).
# the isolation test enforces that statically.
def load_targets(role: str, *, root: Path | None = None, check: bool = True) -> tuple[Target, ...]:
    if role not in TARGET_ROLES:
        raise ValueError(f"load_targets: role must be one of {TARGET_ROLES}, got {role!r}")
    doc = read_raw(role, root)
    if check:
        errs = schema_errors(role, doc)
        if errs:
            raise TargetSchemaError(f"{role}.yaml:\n  " + "\n  ".join(errs))
    return tuple(_target_from_entry(role, e) for e in doc["targets"])


# the calibration targets — the only target file that mrm/calibration may read.
# `active_only` drops entries whose status is not "active" (for example CAL-12d, unverified).
def load_calibration_targets(root: Path | None = None, *, active_only: bool = False) -> tuple[Target, ...]:
    targets = load_targets("calibration", root=root)
    return tuple(t for t in targets if t.status == "active") if active_only else targets


# load excluded.yaml (published numbers that are neither fitted nor scored, with reasons).
def load_excluded(root: Path | None = None, *, check: bool = True) -> tuple[ExcludedItem, ...]:
    doc = read_raw("excluded", root)
    if check:
        errs = schema_errors("excluded", doc)
        if errs:
            raise TargetSchemaError("excluded.yaml:\n  " + "\n  ".join(errs))
    return tuple(
        ExcludedItem(id=str(e["id"]), quantity=str(e["quantity"]), value=e["value"], unit=str(e["unit"]),
                     generation=str(e["generation"]), source=_as_tuple(e["source"]), locator=str(e["locator"]),
                     provenance=str(e["provenance"]), category=str(e["category"]), origin=str(e["origin"]),
                     reason=str(e["reason"]), data_points=_as_tuple(e["data_points"]), raw=dict(e))
        for e in doc["excluded"])


# load all four files (for the report and the tests. never for mrm/calibration).
# with `check` the cross-file split rules of `split_errors` are enforced as well.
def load_all(root: Path | None = None, *, check: bool = True) -> dict[str, tuple]:
    out: dict[str, tuple] = {r: load_targets(r, root=root, check=check) for r in TARGET_ROLES}
    out["excluded"] = load_excluded(root, check=check)
    if check:
        errs = split_errors(out)
        if errs:
            raise TargetSchemaError("target split:\n  " + "\n  ".join(errs))
    return out


# errors against the cross-file rules of the split (empty when the split is clean).
# 1.
# Ids are unique across all four files.
# 2.
# no data point of a calibration entry is tested by a validation or bound entry (deferred entries are exempt, but must list the calibration ids that block them in `blocked_by`).
# 3.
# no excluded data point appears in any other file.
# 4.
# `uses` and `blocked_by` name existing calibration ids.
def split_errors(sets: Mapping[str, Sequence[Any]]) -> list[str]:
    errs: list[str] = []
    ids: dict[str, str] = {}
    for role in ROLES:
        for t in sets.get(role, ()):
            if t.id in ids:
                errs.append(f"{t.id}: id appears in both {ids[t.id]} and {role}")
            ids[t.id] = role
    cal = {t.id: t for t in sets.get("calibration", ())}
    cal_points: dict[str, str] = {}
    for t in cal.values():
        for d in t.data_points:
            cal_points.setdefault(d, t.id)
    for role in HELD_OUT_ROLES:
        for t in sets.get(role, ()):
            blocked = _as_tuple(t.raw.get("blocked_by"))
            for d in t.data_points:
                if d in cal_points:
                    if t.status == "deferred":
                        if cal_points[d] not in blocked:
                            errs.append(f"{t.id}: shares {d} with {cal_points[d]}, which is not in blocked_by")
                    else:
                        errs.append(f"{t.id}: data point {d} is also calibration data ({cal_points[d]})")
            for u in t.uses + blocked:
                if u not in cal:
                    errs.append(f"{t.id}: uses/blocked_by {u!r} is not a calibration id")
    for t in cal.values():
        for u in t.uses:
            if u not in cal:
                errs.append(f"{t.id}: uses {u!r} is not a calibration id")
    other_points: dict[str, str] = {}
    for role in TARGET_ROLES:
        for t in sets.get(role, ()):
            for d in t.data_points:
                other_points.setdefault(d, t.id)
    seen_exc: dict[str, str] = {}
    for x in sets.get("excluded", ()):
        for d in x.data_points:
            if d in other_points:
                errs.append(f"{x.id}: excluded data point {d} is used by {other_points[d]}")
            if d in seen_exc:
                errs.append(f"{x.id}: data point {d} also excluded by {seen_exc[d]}")
            seen_exc[d] = x.id
    return errs


# freezing
FROZEN_BEGIN = "<!-- frozen-targets:begin"
FROZEN_END = "<!-- frozen-targets:end -->"
# one table row of the frozen-targets block - | <file> | `<sha256>` | <ISO date> |
_HEADER_ROW = re.compile(r"^\|\s*(\S+?)\s*\|\s*`?([0-9a-f]{64})`?\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*$")


# sha256 (hex) of the bytes of one target file, with CRLF line ends normalized to LF.
# Any other byte change, a comment included, is a re-freeze.
# the CRLF normalization only keeps a checkout with `core.autocrlf` (Windows) from reporting a false change.
# for an LF file the hash equals the raw-byte hash.
def file_sha256(role: str, root: Path | None = None) -> str:
    return hashlib.sha256(targets_path(role, root).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# the machine-readable header block of docs/VALIDATION.md for the held-out files, frozen on `date`.
def frozen_header_block(date: str, root: Path | None = None) -> str:
    lines = [
        f"{FROZEN_BEGIN} (owned by the target split; the report generator must copy this block verbatim) -->",
        "| file | sha256 | frozen on |",
        "|---|---|---|",
    ]
    for role in FROZEN_ROLES:
        lines.append(f"| mrm/validation/targets/{role}.yaml | `{file_sha256(role, root)}` | {date} |")
    lines.append(FROZEN_END)
    return "\n".join(lines)


# parse the frozen-targets block of docs/VALIDATION.md — {file basename — (sha256, date)}.
def read_frozen_header(path: Path | None = None) -> dict[str, tuple[str, str]]:
    text = (Path(path) if path is not None else DOC_VALIDATION).read_text(encoding="utf-8")
    start = text.find(FROZEN_BEGIN)
    end = text.find(FROZEN_END)
    if start < 0 or end < 0 or end < start:
        raise TargetSchemaError("docs/VALIDATION.md has no frozen-targets block")
    out: dict[str, tuple[str, str]] = {}
    for line in text[start:end].splitlines():
        m = _HEADER_ROW.match(line.strip())
        if m:
            out[Path(m.group(1)).name] = (m.group(2), m.group(3))
    return out


# compare the sha256 of the frozen files with the header of docs/VALIDATION.md.
def header_errors(doc_path: Path | None = None, root: Path | None = None) -> list[str]:
    try:
        header = read_frozen_header(doc_path)
    except (OSError, TargetSchemaError) as exc:
        return [str(exc)]
    errs = []
    for role in FROZEN_ROLES:
        name = f"{role}.yaml"
        if name not in header:
            errs.append(f"docs/VALIDATION.md header has no row for {name}")
            continue
        actual = file_sha256(role, root)
        if header[name][0] != actual:
            errs.append(f"{name}: sha256 {actual} differs from the frozen {header[name][0]} "
                        f"({header[name][1]}); a change to a held-out file is a re-freeze (README.md)")
    return errs


# isolation scan
# Text in any file under mrm/calibration that points at a held-out file or the report.
_FORBIDDEN_TEXT = re.compile(r"(?:validation|bounds)\.ya?ml|VALIDATION\.md|targets[/\\](?:validation|bounds)"
                             r"|mrm[./\\]validation[./\\]report")
# Functions that take a role - calibration code may call them only with the literal role "calibration".
_ROLE_FUNCS = frozenset({"load_targets", "read_raw", "targets_path", "file_sha256", "schema_errors"})
# Names that reach a file other than calibration.yaml (every role, the held-out or excluded files, the frozen header of docs/VALIDATION.md) - calibration code must not use them at all.
_FORBIDDEN_NAMES = frozenset({"load_all", "HELD_OUT_ROLES", "FROZEN_ROLES", "TARGETS_DIR", "DOC_VALIDATION",
                              "read_frozen_header", "header_errors", "frozen_header_block", "load_excluded"})
_FORBIDDEN_MODULES = ("mrm.validation.report",)


# true for `mrm.validation.targets_io`, for the package `mrm.validation` (it re-exports load_targets), and for relative spellings of both (`..validation.targets_io`, `..validation`).
def _is_targets_io_module(mod: str | None, level: int) -> bool:
    if not mod:
        return False
    if mod in ("mrm.validation.targets_io", "mrm.validation"):
        return True
    return level > 0 and (mod.endswith("targets_io") or mod.split(".")[-1] == "validation")


# find accidental references from calibration code to held-out targets.
# scan text for held-out filenames and the validation report.
# in Python, also check imports, aliases, dynamic name strings, and role-taking calls.
# those calls must state the literal calibration role.
# return path and line messages.
# the guard rejects indirect references and broad imports that could bypass the intended loader.
# deliberately obfuscated code can evade this static scan.
def held_out_reference_violations(directory: Path | None = None) -> list[str]:
    base = Path(directory) if directory is not None else CALIBRATION_DIR
    out: list[str] = []
    if not base.exists():
        return out
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(base.parent) if base.parent in path.parents else path
        for n, line in enumerate(text.splitlines(), 1):
            if _FORBIDDEN_TEXT.search(line):
                out.append(f"{rel}:{n}: references a held-out file: {line.strip()[:80]}")
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            out.append(f"{rel}:{exc.lineno}: cannot parse ({exc.msg}); isolation not verifiable")
            continue
        # Import aliases of the role functions (from mrm.validation.targets_io import load_targets as lt - also from the package mrm.validation, which re-exports load_targets).
        role_alias: dict[str, str] = {f: f for f in _ROLE_FUNCS}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and _is_targets_io_module(node.module, node.level):
                for a in node.names:
                    if a.name in _ROLE_FUNCS and a.asname:
                        role_alias[a.asname] = a.name
        call_funcs = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fname = (role_alias.get(node.func.id) if isinstance(node.func, ast.Name)
                         else node.func.attr if isinstance(node.func, ast.Attribute) else None)
                if fname in _ROLE_FUNCS:
                    role_node = node.args[0] if node.args else next(
                        (k.value for k in node.keywords if k.arg == "role"), None)
                    if role_node is None:
                        out.append(f"{rel}:{node.lineno}: {fname}() without an explicit role")
                    elif not (isinstance(role_node, ast.Constant) and role_node.value == "calibration"):
                        out.append(f"{rel}:{node.lineno}: {fname}() with a role other than the literal "
                                   f"'calibration'")
            elif isinstance(node, ast.Name):
                if node.id in _FORBIDDEN_NAMES:
                    out.append(f"{rel}:{node.lineno}: uses {node.id}")
                elif node.id in role_alias and id(node) not in call_funcs and isinstance(node.ctx, ast.Load):
                    out.append(f"{rel}:{node.lineno}: refers to {role_alias[node.id]} outside a direct call")
            elif isinstance(node, ast.Attribute):
                if node.attr in _FORBIDDEN_NAMES:
                    out.append(f"{rel}:{node.lineno}: uses .{node.attr}")
                elif node.attr in _ROLE_FUNCS and id(node) not in call_funcs and isinstance(node.ctx, ast.Load):
                    out.append(f"{rel}:{node.lineno}: refers to .{node.attr} outside a direct call")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value in _FORBIDDEN_NAMES or node.value in _ROLE_FUNCS or node.value in _FORBIDDEN_MODULES:
                    out.append(f"{rel}:{node.lineno}: string literal {node.value!r} names a held-out route")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod in _FORBIDDEN_MODULES or any(a.name in _FORBIDDEN_NAMES for a in node.names):
                    out.append(f"{rel}:{node.lineno}: imports a held-out route from {mod}")
                if mod == "mrm.validation" and any(a.name == "report" for a in node.names):
                    out.append(f"{rel}:{node.lineno}: imports mrm.validation.report")
                if _is_targets_io_module(node.module, node.level) and any(a.name == "*" for a in node.names):
                    out.append(f"{rel}:{node.lineno}: star import from {node.module}")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name in _FORBIDDEN_MODULES:
                        out.append(f"{rel}:{node.lineno}: imports {a.name}")
    return out


# evaluation
# slide-10 aggressor temperature in degC — T(t) = T_max - dT * |sin(pi * (t - t0) / P)|.
# inputs — `t_s` time (s), `t_max_c` peak temperature (degC), `delta_t_k` swing dT (K), `period_s` P (s), `t0_s` time of a cusp (s).
# Peak |dT/dt| = pi * dT / P and mean |dT/dt| = 2 * dT / P.
# Source — docs/research/01_lightmatter_public.md, item D7 (the 808 degC/s definition), and its 'Verification log', row 5 (a fit to the LM26-S slide-10 trace, 0.34 degC rms).
def rectified_sine(t_s: Any, t_max_c: float, delta_t_k: float, period_s: float, t0_s: float = 0.0) -> np.ndarray:
    t = np.asarray(t_s, dtype=float)
    return t_max_c - delta_t_k * np.abs(np.sin(np.pi * (t - t0_s) / period_s))


# the target value as an array (a curve's y list), or None for a kind that has no single array.
def _target_array(target: Target) -> np.ndarray | None:
    v = target.value
    if target.kind == "curve":
        return np.asarray(v["y"], dtype=float)
    if target.kind in ("scalar", "inequality"):
        return np.asarray(v, dtype=float)
    if target.kind in ("interval", "band"):
        return np.asarray(v, dtype=float)
    return None


# |residual| / sigma for a single numeric sigma.
# None when sigma is absent, non-positive or not a scalar.
def _z_score(residual: float | None, sigma: Any) -> float | None:
    if residual is None or not _is_number(sigma) or float(sigma) <= 0:
        return None
    return abs(residual) / float(sigma)


# largest |r_i| / sigma_i over the elements, with sigma a scalar, a list matching `resid`, or a mapping matching `keys` (kind `set`).
# None when sigma cannot be aligned with the residuals or is not positive.
def _max_z_score(resid: np.ndarray, sigma: Any, keys: Sequence[str] | None = None) -> float | None:
    if sigma is None:
        return None
    if _is_number(sigma):
        s = np.full(resid.shape, float(sigma))
    elif keys is not None and isinstance(sigma, Mapping) and set(sigma) == set(keys):
        s = np.array([float(sigma[k]) for k in keys])
    elif isinstance(sigma, (list, tuple)) and len(sigma) == resid.size and all(_is_number(v) for v in sigma):
        s = np.asarray(sigma, dtype=float).reshape(resid.shape)
    else:
        return None
    if np.any(s <= 0):
        return None
    return float(np.max(np.abs(resid) / s))


# true if every numeric leaf of a prediction (scalar, sequence, ndarray, mapping) is finite.
def _all_finite(x: Any) -> bool:
    if isinstance(x, Mapping):
        return all(_all_finite(v) for v in x.values())
    try:
        arr = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return True                     # non-numeric leaves are checked by the shape logic of evaluate()
    return bool(np.all(np.isfinite(arr)))


# score one prediction against one target with the tolerance fixed in the target file.
# Prediction shapes by tolerance kind (units are the target's `unit`) —
# `abs`/`rel` — same shape as the value (scalar, the curve's y list, or a mapping for kind `set`).
# pass if |pred - value| <= tol (abs) or <= tol * |value| (rel), element-wise.
# `interval` — a scalar or an array (for example a sampled range).
# pass if every element is in [lo, hi].
# `inequality` — a scalar or array.
# pass if every element satisfies `op value`.
# with `hard_value`, a prediction between value and hard_value is a "soft_fail".
# `band` — [min, max] of the predicted signal (min <= max).
# pass if lo - slack <= min and max <= hi + slack.
# with `either_sign: true` (the source's sign convention is unpublished) the mirrored band [-hi, -lo] is accepted as well.
# `rms_abs` — for kind `parametric` a mapping {"t_s" — times, "y" — predicted degC-equivalent}, with t_s = 0 at a cusp of the rectified sine (t0_s).
# for a curve, the y list.
# Pass if rms(pred - target) <= value.
# `contour_asymmetry` — a mapping {"below_peak_k" — [...], "above_peak_k" — [...]}.
# pass if every offset is within `abs` and every level shows the required asymmetry.
# `none`/`pending` — not scored.
# `max_abs_residual` is the size of the miss in the target's unit — the largest |pred - value| (abs, rel), the distance outside the allowed range (interval, band, inequality. 0 when inside), the rms (rms_abs) or the largest offset error (contour_asymmetry).
# `z` is the largest |residual_i| / sigma_i where sigma aligns with the value (scalar, list or mapping), else None.
# a non-finite prediction (NaN or inf) is a "fail", never a pass.
def evaluate(target: Target, prediction: Any) -> Evaluation:
    tolerance = target.tolerance
    kind = tolerance.get("kind")
    scored = target.scored
    if kind in ("none", "pending"):
        return Evaluation(target.id, False, None, "not_scored", None, None, f"tolerance {kind}")
    if not _all_finite(prediction):
        return Evaluation(target.id, scored, False, "fail", None, None, "non-finite prediction (replica failed?)")

    def _result(passed: bool, resid: float | None, detail: str, level: str | None = None,
                z: float | None = None) -> Evaluation:
        result_level = level if level is not None else ("pass" if passed else "fail")
        return Evaluation(target.id, scored, passed, result_level, resid,
                          z if z is not None else _z_score(resid, target.sigma), detail)

    if kind in ("abs", "rel"):
        keys: list[str] | None = None
        if target.kind == "set":
            keys = list(target.value)
            if not isinstance(prediction, Mapping) or set(prediction) != set(keys):
                raise ValueError(f"{target.id}: prediction must be a mapping with keys {keys}")
            expected = np.array([float(target.value[k]) for k in keys])
            predicted = np.array([float(prediction[k]) for k in keys])
        else:
            expected = _target_array(target)
            if expected is None:
                raise ValueError(f"{target.id}: kind {target.kind} cannot be scored with {kind}")
            predicted = np.asarray(prediction, dtype=float)
            if predicted.shape != expected.shape:
                raise ValueError(f"{target.id}: prediction shape {predicted.shape} != target shape {expected.shape}")
        resid = np.atleast_1d(predicted - expected)
        allowed = float(tolerance["value"]) * (np.abs(np.atleast_1d(expected)) if kind == "rel" else 1.0)
        worst = float(np.max(np.abs(resid)))
        passed = bool(np.all(np.abs(resid) <= allowed))
        return _result(passed, worst, f"max |pred - value| = {worst:.4g} {target.unit}",
                       z=_max_z_score(resid, target.sigma, keys))

    if kind == "interval":
        predicted = np.atleast_1d(np.asarray(prediction, dtype=float))
        if predicted.size == 0:
            raise ValueError(f"{target.id}: empty prediction")
        lo, hi = float(tolerance["lo"]), float(tolerance["hi"])
        excess = float(np.max(np.maximum(lo - predicted, predicted - hi)))
        passed = bool(np.all((predicted >= lo) & (predicted <= hi)))
        return _result(passed, max(excess, 0.0), f"prediction range [{predicted.min():.4g}, {predicted.max():.4g}] "
                                                 f"against [{lo:.4g}, {hi:.4g}]")

    if kind == "inequality":
        predicted = np.atleast_1d(np.asarray(prediction, dtype=float))
        if predicted.size == 0:
            raise ValueError(f"{target.id}: empty prediction")
        op, v = tolerance["op"], float(tolerance["value"])
        worst = float(predicted.min() if op == ">=" else predicted.max())
        ok = worst >= v if op == ">=" else worst <= v
        level = "pass" if ok else "fail"
        if not ok and "hard_value" in tolerance:
            hard_value = float(tolerance["hard_value"])
            level = "soft_fail" if (worst >= hard_value if op == ">=" else worst <= hard_value) else "fail"
        margin = (worst - v) if op == ">=" else (v - worst)          # > 0 - inside, < 0 - violated
        return Evaluation(target.id, scored, ok, level, max(-margin, 0.0), None,
                          f"worst prediction {worst:.4g} {op} {v:.4g} (margin {margin:+.4g} {target.unit})")

    if kind == "band":
        predicted = np.asarray(prediction, dtype=float).ravel()
        if predicted.size != 2 or predicted[0] > predicted[1]:
            raise ValueError(f"{target.id}: band prediction must be [min, max] with min <= max")
        lo = float(tolerance["lo"]) - float(tolerance["slack"])
        hi = float(tolerance["hi"]) + float(tolerance["slack"])

        def _excess(p0: float, p1: float) -> float:
            return float(max(lo - p0, p1 - hi, 0.0))

        excess, mirrored = _excess(predicted[0], predicted[1]), False
        if excess > 0.0 and tolerance.get("either_sign", False):
            mirrored_excess = _excess(-predicted[1], -predicted[0])
            if mirrored_excess < excess:
                excess, mirrored = mirrored_excess, True
        passed = excess == 0.0
        return _result(passed, excess,
                       f"predicted band [{predicted[0]:.4g}, {predicted[1]:.4g}]{' (mirrored)' if mirrored else ''} "
                       f"within [{lo:.4g}, {hi:.4g}]")

    if kind == "rms_abs":
        if target.kind == "parametric":
            if not isinstance(prediction, Mapping) or not {"t_s", "y"} <= set(prediction):
                raise ValueError(f"{target.id}: parametric prediction must be a mapping with t_s and y")
            p = target.value["params"]
            reference = rectified_sine(prediction["t_s"], float(p["t_max_c"]), float(p["delta_t_k"]),
                                 float(p["period_s"]), float(p.get("t0_s", 0.0)))
            predicted = np.asarray(prediction["y"], dtype=float)
        else:
            reference = _target_array(target)
            predicted = np.asarray(prediction, dtype=float)
        if reference is None or predicted.shape != reference.shape:
            raise ValueError(f"{target.id}: prediction shape does not match the target")
        rms = float(np.sqrt(np.mean((predicted - reference) ** 2)))
        return _result(rms <= float(tolerance["value"]), rms, f"rms residual {rms:.4g} {target.unit}")

    if kind == "contour_asymmetry":
        if not isinstance(prediction, Mapping) or not {"below_peak_k", "above_peak_k"} <= set(prediction):
            raise ValueError(f"{target.id}: prediction must map below_peak_k and above_peak_k")
        # contour offsets (K) below and above the peak temperature, one per level - target and prediction
        target_below = np.asarray(target.value["below_peak_k"], dtype=float)
        target_above = np.asarray(target.value["above_peak_k"], dtype=float)
        pred_below = np.asarray(prediction["below_peak_k"], dtype=float)
        pred_above = np.asarray(prediction["above_peak_k"], dtype=float)
        if pred_below.shape != target_below.shape or pred_above.shape != target_above.shape:
            raise ValueError(f"{target.id}: prediction must have one offset per level")
        resid = np.concatenate([pred_below - target_below, pred_above - target_above])
        worst = float(np.max(np.abs(resid)))
        sign_ok = (bool(np.all(pred_below > pred_above)) if tolerance["require"] == "cold_side_wider"
                   else bool(np.all(pred_above > pred_below)))
        passed = worst <= float(tolerance["abs"]) and sign_ok
        return _result(passed, worst, f"max offset error {worst:.3g} K; asymmetry {'ok' if sign_ok else 'WRONG'}")

    raise ValueError(f"{target.id}: unknown tolerance kind {kind!r}")


# CLI
# `check` — schema, split, frozen hashes and isolation scan (exit 1 on any error).
# `header [date]` — print the frozen-targets block for docs/VALIDATION.md (a re-freeze. see README.md).
def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args else "check"
    if cmd == "header":
        date = args[1] if len(args) > 1 else None
        if date is None:
            date = str(read_raw("validation").get("frozen_on"))
        print(frozen_header_block(date))
        return 0
    if cmd != "check":
        print('``check`` — schema, split, frozen hashes and isolation scan (exit 1 on any error).\n    ``header [date]`` — print the frozen-targets block for docs/VALIDATION.md (a re-freeze. see README.md).')
        return 2
    errs: list[str] = []
    for role in ROLES:
        errs.extend(schema_errors(role, read_raw(role)))
    if not errs:
        errs.extend(split_errors(load_all(check=False)))
    errs.extend(header_errors())
    errs.extend(held_out_reference_violations())
    for e in errs:
        print("ERROR", e)
    if not errs:
        counts = {r: len(v) for r, v in load_all().items()}
        print("targets OK:", ", ".join(f"{k} {v}" for k, v in counts.items()))
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
