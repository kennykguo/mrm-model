"""Unit-suffix grammar of config keys and the single conversion to SI (07 P1-15, ARCHITECTURE section 1).

Grammar
-------
A config key is ``<name>[_<qualifier>]_<unit>``. The unit is the longest run of trailing ``_``-separated tokens that
forms a valid unit expression::

    unit     := [factor ...] [per factor ...]*        (at least one factor in total)
    factor   := [rt] TOKEN[2|3]                       (rt = square root; a digit is an exponent)

Adjacent factors multiply; every ``per`` starts a denominator group, so ``w_per_m_k`` is W/(m K) and
``pm_per_k_per_mw`` is (pm/K)/mW. A key whose trailing tokens do not form a unit (``n_g``, ``q_loaded``,
``kappa2_per_pass``) is unitless.

Special units (07 P1-15: "``_c`` is affine; ``_dbm`` and ``_db`` are nonlinear"):

- ``c``    absolute temperature in degC, affine: K = degC + 273.15. Only allowed alone. A temperature *difference*
           must use ``k`` (the parser rejects ``_c`` after qualifiers such as ``sigma``, ``offset``, ``rise``).
- ``dbm``  absolute power: W = 1e-3 * 10**(x/10). Only allowed alone.
- ``db``   a power ratio. SI form: the natural-log power ratio ln(P2/P1) = x*ln(10)/10 (dimensionless). A loss of
           x dB therefore becomes a transmission ``exp(-ln_ratio)``; a sigma in dB becomes a sigma of ln(P).
- ``db_per_<length>``  power attenuation coefficient alpha in 1/m with P(z) = P(0)*exp(-alpha*z),
           alpha = x*ln(10)/10 / length_scale.
- ``db_per_hz``  a spectral density relative to 1/Hz (RIN): SI value 10**(x/10) in 1/Hz.

Token table (ARCHITECTURE section 1 plus documented additions marked "added"): see :data:`TOKENS`.
``hz``/``khz``/.../``thz`` are always cycles per second; angular rates are never written with these tokens.
``pa`` is the picoampere (there is no pressure in this model). ``k`` is the kelvin, never "kilo".
"""
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
_LN10_OVER_10 = math.log(10.0) / 10.0

# Base dimensions: (m, kg, s, A, K). Exponents are floats so that rt gives 0.5.
_DIM_NAMES = ("m", "kg", "s", "A", "K")
_L = (1, 0, 0, 0, 0)
_T = (0, 0, 1, 0, 0)
_NONE = (0, 0, 0, 0, 0)
_W = (2, 1, -3, 0, 0)
_A = (0, 0, 0, 1, 0)
_V = (2, 1, -3, -1, 0)
_OHM = (2, 1, -3, -2, 0)
_F = (-2, -1, 4, 2, 0)
_J = (2, 1, -2, 0, 0)
_K = (0, 0, 0, 0, 1)
_HZ = (0, 0, -1, 0, 0)

#: token -> (SI scale, dimension exponents, note). Exact SI prefixes; eV from the exact elementary charge.
TOKENS: dict[str, tuple[float, tuple[int, ...], str]] = {
    # length
    "pm": (1e-12, _L, "picometre"), "nm": (1e-9, _L, "nanometre"), "um": (1e-6, _L, "micrometre"),
    "mm": (1e-3, _L, "millimetre"), "cm": (1e-2, _L, "centimetre (added: loss per cm)"), "m": (1.0, _L, "metre"),
    # time
    "ps": (1e-12, _T, "picosecond"), "ns": (1e-9, _T, "nanosecond"), "us": (1e-6, _T, "microsecond"),
    "ms": (1e-3, _T, "millisecond"), "s": (1.0, _T, "second"),
    # frequency (cycles per second)
    "hz": (1.0, _HZ, "hertz"), "khz": (1e3, _HZ, "kilohertz"), "mhz": (1e6, _HZ, "megahertz"),
    "ghz": (1e9, _HZ, "gigahertz"), "thz": (1e12, _HZ, "terahertz"),
    "gbps": (1e9, _HZ, "gigabit per second (added: data rates)"),
    # power
    "w": (1.0, _W, "watt"), "mw": (1e-3, _W, "milliwatt"), "uw": (1e-6, _W, "microwatt"), "nw": (1e-9, _W, "nanowatt"),
    # current
    "a": (1.0, _A, "ampere"), "ma": (1e-3, _A, "milliampere"), "ua": (1e-6, _A, "microampere"),
    "na": (1e-9, _A, "nanoampere"), "pa": (1e-12, _A, "picoampere (added: noise densities)"),
    # voltage, resistance, capacitance
    "v": (1.0, _V, "volt"), "mv": (1e-3, _V, "millivolt"), "ohm": (1.0, _OHM, "ohm"),
    "f": (1.0, _F, "farad"), "pf": (1e-12, _F, "picofarad (added)"), "ff": (1e-15, _F, "femtofarad"),
    # temperature (k = kelvin; c is affine and handled separately)
    "k": (1.0, _K, "kelvin"),
    # energy
    "ev": (Q_E, _J, "electronvolt"), "j": (1.0, _J, "joule (added)"), "pj": (1e-12, _J, "picojoule (added)"),
    "fj": (1e-15, _J, "femtojoule (added)"),
    # dimensionless multipliers and counts
    "pct": (1e-2, _NONE, "percent"), "frac": (1.0, _NONE, "fraction"), "ppm": (1e-6, _NONE, "parts per million (added)"),
    "codes": (1.0, _NONE, "converter codes (added)"), "lsb": (1.0, _NONE, "least significant bits (added)"),
}
_SPECIAL = {"c", "db", "dbm"}
_OPERATORS = {"per", "rt"}
#: qualifiers that mark a temperature difference; such keys must use ``_k``, not the affine ``_c``.
_DIFF_QUALIFIERS = ("sigma", "delta", "rise", "offset", "span", "diff", "swing", "error", "noise", "resolution", "step")

_TOKEN_RE = re.compile(r"^([a-z]+?)([23])?$")


class UnitError(ValueError):
    """A key violates the unit-suffix grammar."""


@dataclass(frozen=True)
class UnitSpec:
    """Parsed unit of a config key.

    kind: 'none' (unitless), 'linear' (si = x*scale), 'celsius' (si = x + 273.15),
    'db' (si = x*ln(10)/10, a natural-log power ratio), 'dbm' (si = 1e-3*10**(x/10) W),
    'db_per_len' (si = x*ln(10)/10/scale, 1/m), 'db_per_hz' (si = 10**(x/10) 1/Hz).
    """
    expr: str | None
    kind: str
    scale: float
    dims: tuple[float, ...]

    @property
    def si_unit(self) -> str:
        """Human-readable SI unit, e.g. 'm K^-1'."""
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
        for name, e in zip(_DIM_NAMES, self.dims):
            if e == 0:
                continue
            if e == 1:
                parts.append(name)
            else:
                es = f"{e:g}"
                parts.append(f"{name}^{es}")
        return " ".join(parts) if parts else "1"


_UNITLESS = UnitSpec(None, "none", 1.0, (0, 0, 0, 0, 0))


def _factor(tok: str) -> tuple[float, tuple[float, ...]] | None:
    m = _TOKEN_RE.match(tok)
    if not m:
        return None
    base, exp = m.group(1), m.group(2)
    if base not in TOKENS:
        return None
    scale, dims, _ = TOKENS[base]
    e = int(exp) if exp else 1
    return scale ** e, tuple(d * e for d in dims)


def _parse_expr(tokens: list[str]) -> UnitSpec | None:
    """Parse a candidate unit expression (list of tokens). Returns None if it is not a valid unit."""
    if not tokens:
        return None
    expr = "_".join(tokens)
    # specials
    if tokens == ["c"]:
        return UnitSpec(expr, "celsius", 1.0, _K)
    if tokens == ["dbm"]:
        return UnitSpec(expr, "dbm", 1e-3, _W)
    if tokens == ["db"]:
        return UnitSpec(expr, "db", _LN10_OVER_10, _NONE)
    if tokens[0] == "db" and len(tokens) >= 3 and tokens[1] == "per":
        rest = _parse_expr(tokens[2:])
        if rest is None or rest.kind != "linear":
            return None
        if rest.dims == _L or rest.dims == tuple(float(x) for x in _L):
            return UnitSpec(expr, "db_per_len", rest.scale, tuple(-d for d in rest.dims))
        if tuple(rest.dims) == tuple(float(x) for x in _HZ) and tokens[2:] == ["hz"]:
            return UnitSpec(expr, "db_per_hz", 1.0, tuple(-d for d in rest.dims))
        return None
    if any(t in _SPECIAL for t in tokens):
        return None
    scale = 1.0
    dims = [0.0] * 5
    sign = 1.0
    n_factors = 0
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "per":
            if i == len(tokens) - 1:
                return None
            sign = -1.0
            i += 1
            continue
        root = 1.0
        if t == "rt":
            if i == len(tokens) - 1:
                return None
            root = 0.5
            i += 1
            t = tokens[i]
        f = _factor(t)
        if f is None:
            return None
        s, d = f
        scale *= s ** (sign * root)
        for j in range(5):
            dims[j] += sign * root * d[j]
        n_factors += 1
        i += 1
    if n_factors == 0:
        return None
    # a 'per' that is not followed by factors was rejected above; a trailing operator is invalid
    return UnitSpec(expr, "linear", scale, tuple(dims))


@lru_cache(maxsize=4096)
def parse_key(key: str) -> tuple[str, UnitSpec]:
    """Split the last path segment of ``key`` into (name, UnitSpec).

    The unit is the longest valid trailing token run that leaves a non-empty name. Raises UnitError when the key
    uses ``_c`` for a temperature difference or ``_dbm`` for a spread (use ``_k`` and ``_db``).
    """
    seg = key.rsplit(".", 1)[-1]
    tokens = seg.split("_")
    for i in range(1, len(tokens)):
        spec = _parse_expr(tokens[i:])
        if spec is not None:
            name = "_".join(tokens[:i])
            last = tokens[i - 1]      # the qualifier right before the unit decides whether it is a difference
            if spec.kind == "celsius" and last in _DIFF_QUALIFIERS:
                raise UnitError(f"{key!r}: '_c' is an absolute (affine) temperature; a difference or spread must "
                                f"use '_k' (07 P1-15)")
            if spec.kind == "dbm" and last in ("sigma", "delta", "offset", "step", "swing"):
                raise UnitError(f"{key!r}: a spread or step of a power is a ratio; use '_db', not '_dbm'")
            # A special token inside a compound unit cannot be parsed, and the longest-match rule would then pick a
            # shorter unit and leave the special token in the name: 'shift_ghz_per_c' would read as name
            # 'shift_ghz_per' in absolute degC, 'x_db_per_s' as name 'x_db' in 1/s. Refuse both.
            if last == "per" and spec.kind in ("celsius", "dbm", "db"):
                raise UnitError(f"{key!r}: '_per_{spec.expr}' is not a unit; a coefficient per degree is '_per_k' "
                                f"(07 P1-15: '_c' is affine, '_db'/'_dbm' are logarithmic)")
            if last in _SPECIAL and tokens[i] == "per":
                raise UnitError(f"{key!r}: '_{last}_{spec.expr}' is not a supported unit (only dB per length "
                                f"and dB per Hz are)")
            return name, spec
    return seg, _UNITLESS


def unit_of(key: str) -> UnitSpec:
    """UnitSpec of ``key`` (see :func:`parse_key`)."""
    return parse_key(key)[1]


def to_si(key: str | UnitSpec, value):
    """Convert ``value`` (scalar or array, in the unit named by the key suffix) to SI. Vectorized; returns float64.

    This is the only place where config units become SI (ARCHITECTURE section 1: "mrm.params converts to SI exactly
    once").
    """
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


def from_si(key: str | UnitSpec, value):
    """Inverse of :func:`to_si` (for display only)."""
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


def describe(key: str) -> str:
    """One-line description of a key's unit, e.g. ``'pm_per_k -> m K^-1 (x 1e-12)'``."""
    name, spec = parse_key(key)
    if spec.kind == "none":
        return "unitless"
    if spec.kind == "linear":
        return f"{spec.expr} -> {spec.si_unit} (x {spec.scale:g})"
    return f"{spec.expr} -> {spec.si_unit} ({spec.kind})"
