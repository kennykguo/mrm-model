# read units from config-key suffixes and convert values between config units and SI.
# parse_key selects the longest valid trailing unit expression that leaves a nonempty name.
# radius_um means micrometres.
# dlambda_dt_pm_per_k means pm/K.
# unrecognized suffixes are unitless, including n_g, q_loaded, and kappa2_per_pass.
# adjacent factors multiply.
# per puts subsequent factors in the denominator.
# rt takes a square root.
# tokens can end in 2 or 3 for powers.
# w_per_m_k is W/(m*K).
# pm_per_k_per_mw is pm/(K*mW).
# absolute Celsius uses x+273.15 K.
# temperature differences must use k, not affine c.
# dBm is absolute power.
# dB is a power ratio, stored in SI form as a natural logarithm.
# dB per length is an exponential power-loss coefficient.
# dB per Hz is a linear spectral density.
# see to_si for conversion equations.
# special logarithmic/affine tokens cannot form arbitrary compounds.
# TOKENS gives scales and dimensions.
# _parse_expr validates candidates.
# parse_key chooses the suffix.
# to_si converts scalars/arrays.
# from_si serves display.
# describe labels units for reports.
# frequency tokens mean cycles per second, never angular frequency.
# pa means picoampere in this model.
# k means kelvin, not kilo.
# see docs/ARCHITECTURE.md, section 1.
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from mrm.units import Q_E, T0_K

__all__ = [
    "TOKENS", "UnitSpec", "UnitError", "parse_key", "unit_of", "to_si", "from_si", "describe", "CELSIUS_OFFSET_K",
]

CELSIUS_OFFSET_K = T0_K     # 0 degC in K (definition of the Celsius scale)
_LN10_OVER_10 = math.log(10.0) / 10.0      # x dB of a power ratio is ln(P2/P1) = x * ln(10)/10 (-)

# Base dimensions - exponents of (m, kg, s, A, K).
# a parsed unit holds floats, so that rt (square root) gives 0.5.
_DIM_NAMES = ("m", "kg", "s", "A", "K")
_DIM_M = (1, 0, 0, 0, 0)
_DIM_S = (0, 0, 1, 0, 0)
_DIM_NONE = (0, 0, 0, 0, 0)
_DIM_W = (2, 1, -3, 0, 0)
_DIM_A = (0, 0, 0, 1, 0)
_DIM_V = (2, 1, -3, -1, 0)
_DIM_OHM = (2, 1, -3, -2, 0)
_DIM_F = (-2, -1, 4, 2, 0)
_DIM_J = (2, 1, -2, 0, 0)
_DIM_K = (0, 0, 0, 0, 1)
_DIM_HZ = (0, 0, -1, 0, 0)

# token -> (SI scale, dimension exponents, note). Exact SI prefixes - eV from the exact elementary charge.
TOKENS: dict[str, tuple[float, tuple[int, ...], str]] = {
    # length
    "pm": (1e-12, _DIM_M, "picometre"), "nm": (1e-9, _DIM_M, "nanometre"), "um": (1e-6, _DIM_M, "micrometre"),
    "mm": (1e-3, _DIM_M, "millimetre"), "cm": (1e-2, _DIM_M, "centimetre (added: loss per cm)"),
    "m": (1.0, _DIM_M, "metre"),
    # time
    "ps": (1e-12, _DIM_S, "picosecond"), "ns": (1e-9, _DIM_S, "nanosecond"), "us": (1e-6, _DIM_S, "microsecond"),
    "ms": (1e-3, _DIM_S, "millisecond"), "s": (1.0, _DIM_S, "second"),
    # frequency (cycles per second)
    "hz": (1.0, _DIM_HZ, "hertz"), "khz": (1e3, _DIM_HZ, "kilohertz"), "mhz": (1e6, _DIM_HZ, "megahertz"),
    "ghz": (1e9, _DIM_HZ, "gigahertz"), "thz": (1e12, _DIM_HZ, "terahertz"),
    "gbps": (1e9, _DIM_HZ, "gigabit per second (added: data rates)"),
    # power
    "w": (1.0, _DIM_W, "watt"), "mw": (1e-3, _DIM_W, "milliwatt"), "uw": (1e-6, _DIM_W, "microwatt"),
    "nw": (1e-9, _DIM_W, "nanowatt"),
    # current
    "a": (1.0, _DIM_A, "ampere"), "ma": (1e-3, _DIM_A, "milliampere"), "ua": (1e-6, _DIM_A, "microampere"),
    "na": (1e-9, _DIM_A, "nanoampere"), "pa": (1e-12, _DIM_A, "picoampere (added: noise densities)"),
    # voltage, resistance, capacitance
    "v": (1.0, _DIM_V, "volt"), "mv": (1e-3, _DIM_V, "millivolt"), "ohm": (1.0, _DIM_OHM, "ohm"),
    "f": (1.0, _DIM_F, "farad"), "pf": (1e-12, _DIM_F, "picofarad (added)"), "ff": (1e-15, _DIM_F, "femtofarad"),
    # temperature (k = kelvin - c is affine and handled separately)
    "k": (1.0, _DIM_K, "kelvin"),
    # energy
    "ev": (Q_E, _DIM_J, "electronvolt"), "j": (1.0, _DIM_J, "joule (added)"),
    "pj": (1e-12, _DIM_J, "picojoule (added)"), "fj": (1e-15, _DIM_J, "femtojoule (added)"),
    # dimensionless multipliers and counts
    "pct": (1e-2, _DIM_NONE, "percent"), "frac": (1.0, _DIM_NONE, "fraction"),
    "ppm": (1e-6, _DIM_NONE, "parts per million (added)"),
    "codes": (1.0, _DIM_NONE, "converter codes (added)"), "lsb": (1.0, _DIM_NONE, "least significant bits (added)"),
}
_SPECIAL = {"c", "db", "dbm"}           # units that are not a scale factor (affine or logarithmic)
_OPERATORS = {"per", "rt"}              # "per" starts a denominator group, "rt" takes the square root
# qualifiers that mark a temperature difference - such keys must use ``_k``, not the affine ``_c``.
_DIFF_QUALIFIERS = ("sigma", "delta", "rise", "offset", "span", "diff", "swing", "error", "noise", "resolution", "step")

_TOKEN_RE = re.compile(r"^([a-z]+?)([23])?$")      # a unit token with an optional exponent 2 or 3


# a key violates the unit-suffix grammar.
class UnitError(ValueError):
    pass


# hold a parsed suffix, conversion kind, SI scale, and base-dimension exponents.
# kinds are none, linear, celsius, db, dbm, db_per_len, and db_per_hz.
# dims is a tuple of exponents for (m, kg, s, A, K).
# fractional exponents represent roots.
# scale is a multiplier for linear units or length scale for db_per_len.
# see to_si for special conversions.
@dataclass(frozen=True)
class UnitSpec:
    expr: str | None
    kind: str
    scale: float
    dims: tuple[float, ...]

    # Human-readable SI unit, e.g. 'm K^-1'.
    @property
    def si_unit(self) -> str:
        if self.kind == "celsius":
            return "K"
        if self.kind == "dbm":
            return "W"
        if self.kind == "db":
            return "ln(P2/P1)"
        if self.kind == "db_per_len":
            return "m^-1 (ln power)"
        if self.kind == "db_per_hz":
            return "Hz^-1"
        parts = []
        for name, exponent in zip(_DIM_NAMES, self.dims):
            if exponent == 0:
                continue
            if exponent == 1:
                parts.append(name)
            else:
                exponent_text = f"{exponent:g}"
                parts.append(f"{name}^{exponent_text}")
        return " ".join(parts) if parts else "1"


_UNITLESS = UnitSpec(None, "none", 1.0, (0, 0, 0, 0, 0))     # the spec of a key without a unit suffix


# (SI scale, dimensions) of one factor token such as `um` or `m2` (a trailing 2 or 3 is an exponent) - None when the token is not in `TOKENS`.
def _factor(tok: str) -> tuple[float, tuple[float, ...]] | None:
    match = _TOKEN_RE.match(tok)
    if not match:
        return None
    base, exponent_text = match.group(1), match.group(2)
    if base not in TOKENS:
        return None
    scale, dims, _ = TOKENS[base]
    exponent = int(exponent_text) if exponent_text else 1
    return scale ** exponent, tuple(d * exponent for d in dims)


# parse one candidate trailing token sequence.
# return None for unsupported expressions.
# handle affine/logarithmic cases first.
# ordinary factors accumulate scale and dimension exponents.
# per changes the exponent sign to negative.
# rt halves the next factor's exponent.
def _parse_expr(tokens: list[str]) -> UnitSpec | None:
    if not tokens:
        return None
    expr = "_".join(tokens)
    # specials
    if tokens == ["c"]:
        return UnitSpec(expr, "celsius", 1.0, _DIM_K)
    if tokens == ["dbm"]:
        return UnitSpec(expr, "dbm", 1e-3, _DIM_W)
    if tokens == ["db"]:
        return UnitSpec(expr, "db", _LN10_OVER_10, _DIM_NONE)
    if tokens[0] == "db" and len(tokens) >= 3 and tokens[1] == "per":
        denominator_spec = _parse_expr(tokens[2:])
        if denominator_spec is None or denominator_spec.kind != "linear":
            return None
        if denominator_spec.dims == _DIM_M or denominator_spec.dims == tuple(float(x) for x in _DIM_M):
            return UnitSpec(expr, "db_per_len", denominator_spec.scale,
                tuple(-factor_dims for factor_dims in denominator_spec.dims))
        if tuple(denominator_spec.dims) == tuple(float(x) for x in _DIM_HZ) and tokens[2:] == ["hz"]:
            return UnitSpec(expr, "db_per_hz", 1.0, tuple(-factor_dims for factor_dims in denominator_spec.dims))
        return None
    if any(unit_token in _SPECIAL for unit_token in tokens):
        return None
    scale = 1.0
    dims = [0.0] * 5
    sign = 1.0
    n_factors = 0
    i = 0
    while i < len(tokens):
        unit_token = tokens[i]
        if unit_token == "per":
            if i == len(tokens) - 1:
                return None
            sign = -1.0
            i += 1
            continue
        root_exponent = 1.0
        if unit_token == "rt":
            if i == len(tokens) - 1:
                return None
            root_exponent = 0.5
            i += 1
            unit_token = tokens[i]
        factor = _factor(unit_token)
        if factor is None:
            return None
        factor_scale, factor_dims = factor
        scale *= factor_scale ** (sign * root_exponent)
        for j in range(5):
            dims[j] += sign * root_exponent * factor_dims[j]
        n_factors += 1
        i += 1
    if n_factors == 0:
        return None
    # a 'per' that is not followed by factors was rejected above - a trailing operator is invalid
    return UnitSpec(expr, "linear", scale, tuple(dims))


# split the final dotted path segment into (quantity name, UnitSpec).
# try suffixes from longest to shortest while retaining at least one name token.
# reject affine Celsius for known difference qualifiers and dBm for spreads/steps.
# also reject special-unit compounds that could otherwise be misread as part of the quantity name.
# return a unitless spec when no valid suffix exists.
@lru_cache(maxsize=4096)
def parse_key(key: str) -> tuple[str, UnitSpec]:
    segment = key.rsplit(".", 1)[-1]
    tokens = segment.split("_")
    for i in range(1, len(tokens)):
        spec = _parse_expr(tokens[i:])
        if spec is not None:
            name = "_".join(tokens[:i])
            qualifier = tokens[i - 1]      # the qualifier right before the unit decides whether it is a difference
            if spec.kind == "celsius" and qualifier in _DIFF_QUALIFIERS:
                raise UnitError(f"{key!r}: '_c' is an absolute (affine) temperature; a difference or spread must "
                                f"use '_k'")
            if spec.kind == "dbm" and qualifier in ("sigma", "delta", "offset", "step", "swing"):
                raise UnitError(f"{key!r}: a spread or step of a power is a ratio; use '_db', not '_dbm'")
            # a special token inside a compound unit cannot be parsed, and the longest-match rule would then pick a shorter unit and leave the special token in the name - 'shift_ghz_per_c' would read as name 'shift_ghz_per' in absolute degC, 'x_db_per_s' as name 'x_db' in 1/s.
            # Refuse both.
            if qualifier == "per" and spec.kind in ("celsius", "dbm", "db"):
                raise UnitError(f"{key!r}: '_per_{spec.expr}' is not a unit; a coefficient per degree is '_per_k' "
                                f"('_c' is affine, '_db'/'_dbm' are logarithmic)")
            if qualifier in _SPECIAL and tokens[i] == "per":
                raise UnitError(f"{key!r}: '_{qualifier}_{spec.expr}' is not a supported unit (only dB per length "
                                f"and dB per Hz are)")
            return name, spec
    return segment, _UNITLESS


# UnitSpec of `key` (see `parse_key`).
def unit_of(key: str) -> UnitSpec:
    return parse_key(key)[1]


# convert a scalar or array from suffix units to SI, using float64 arithmetic.
# linear — x*scale.
# Celsius — x+273.15 K.
# dB — x*ln(10)/10, a natural-log power ratio.
# a positive loss gives transmission exp(-converted_loss).
# dBm — 1e-3*10^(x/10) W.
# dB per length — x*ln(10)/(10*length_scale), in 1/m for P(z)=P(0)*exp(-alpha*z).
# dB per Hz — 10^(x/10), in 1/Hz for RIN density.
# return a Python float for scalar input, otherwise an array.
# unitless input may share its array.
# this is the common config conversion used by the parameter layer.
def to_si(key: str | UnitSpec, value):
    spec = key if isinstance(key, UnitSpec) else unit_of(key)
    x = np.asarray(value, dtype=np.float64)
    if spec.kind in ("none",):
        out = x
    elif spec.kind == "linear":
        out = x * spec.scale
    elif spec.kind == "celsius":
        out = x + CELSIUS_OFFSET_K
    elif spec.kind == "db":
        out = x * _LN10_OVER_10
    elif spec.kind == "dbm":
        out = 1e-3 * np.power(10.0, x / 10.0)
    elif spec.kind == "db_per_len":
        out = x * _LN10_OVER_10 / spec.scale
    elif spec.kind == "db_per_hz":
        out = np.power(10.0, x / 10.0)
    else:  # pragma: no cover - kinds are closed
        raise UnitError(f"unknown unit kind {spec.kind}")
    return out if out.ndim else float(out)


# convert SI values back to suffix units for display.
# scalar input returns a Python float.
# logarithmic power/density display floors values at 1e-300 before log10.
def from_si(key: str | UnitSpec, value):
    spec = key if isinstance(key, UnitSpec) else unit_of(key)
    x = np.asarray(value, dtype=np.float64)
    if spec.kind == "none":
        out = x
    elif spec.kind == "linear":
        out = x / spec.scale
    elif spec.kind == "celsius":
        out = x - CELSIUS_OFFSET_K
    elif spec.kind == "db":
        out = x / _LN10_OVER_10
    elif spec.kind == "dbm":
        out = 10.0 * np.log10(np.maximum(x, 1e-300) / 1e-3)
    elif spec.kind == "db_per_len":
        out = x * spec.scale / _LN10_OVER_10
    elif spec.kind == "db_per_hz":
        out = 10.0 * np.log10(np.maximum(x, 1e-300))
    else:  # pragma: no cover
        raise UnitError(f"unknown unit kind {spec.kind}")
    return out if out.ndim else float(out)


# one-line description of a key's unit, e.g. `'pm_per_k -> m K^-1 (x 1e-12)'`.
def describe(key: str) -> str:
    name, spec = parse_key(key)
    if spec.kind == "none":
        return "unitless"
    if spec.kind == "linear":
        return f"{spec.expr} -> {spec.si_unit} (x {spec.scale:g})"
    return f"{spec.expr} -> {spec.si_unit} ({spec.kind})"
