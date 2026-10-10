"""Fixed-point toolkit for controllers that must become RTL (07 P1-21).

Python integers never overflow and numba's int64 wraps at 64 bits, not at the 16- to 48-bit widths of an FPGA
datapath (07 P1-21, "Why"). A controller written with this module states every format and every overflow and
rounding decision, so its numba form, its Python golden model and its Verilog agree bit for bit (textbook Ch17: the
integrator's fraction bits are the usual failure).

Formats
-------
``QFormat(int_bits, frac_bits, signed)`` in TI notation: ``Qm.n`` is signed with 1 sign bit, m integer bits and n
fraction bits (total 1 + m + n); ``UQm.n`` is unsigned with m + n bits. A raw value r (int64) represents r * 2**-n.
Example (docs/research/05_tools.md, check C12): Kp = 0.37512 in Q7.8 is raw 96 = 0.375, and raw 40000 saturates to
32767. Widths are limited to 1..62 bits and fraction bits to 0..62, so every intermediate fits an int64.

Operations
----------
Raw-integer kernels, all ``@njit`` and usable inside compiled controllers (arguments are int64 scalars; the mode
arguments are the integer values of :class:`Rounding` and :class:`Overflow`):

- ``fx_overflow(x, bits, signed, ovf)``: WRAP (two's complement, like RTL), SATURATE (clip), CHECKED (raise
  OverflowError);
- ``fx_round_shift(x, s, rnd)``: x * 2**-s rounded with FLOOR (arithmetic shift, what ``>>>`` does), HALF_UP
  (add half then floor), HALF_EVEN (convergent), TOWARD_ZERO (truncate) or HALF_AWAY (away from zero);
- ``fx_from_float``, ``fx_to_float``, ``fx_add``, ``fx_sub``, ``fx_neg``, ``fx_mul`` (exact int64 product, then one
  rounding, then one overflow step), ``fx_cast`` (re-quantize between formats), ``fx_matvec`` (one dense layer
  with a declared accumulator width; the fixed-point MLP inference helper of 07 P1-22 is mrm.control.fxp_mlp,
  which uses the scalar kernels of this module but not fx_matvec).

``fx_mul`` raises ValueError if the exact product does not fit an int64 (|a*b| >= 2**62); that is a limitation of
the toolkit, not an RTL overflow, so it raises in every mode. Keep total_a + total_b <= 62 bits.

The :class:`QFormat` class wraps the kernels for NumPy arrays and scalars in Python.
"""
from __future__ import annotations

import enum
import math
import re
from dataclasses import dataclass

import numba as nb
import numpy as np

__all__ = [
    "Rounding", "Overflow", "QFormat", "MAX_BITS",
    "fx_raw_min", "fx_raw_max", "fx_overflow", "fx_round_shift", "fx_from_float", "fx_to_float", "fx_add", "fx_sub",
    "fx_neg", "fx_mul", "fx_cast", "fx_matvec",
    "quantize", "to_float", "add", "sub", "mul", "cast",
    # reference fixed-point PI controller (added by the hardware task; 07 P1-21)
    "fx_pi_update", "fxp_pi_init", "fxp_pi_step", "FxpPi", "pi_float_bound_codes",
]

MAX_BITS = 62   # widest format: keeps every sum and every exact product of the allowed operands inside an int64


class Rounding(enum.IntEnum):
    """Rounding mode of a right shift (drop of fraction bits)."""
    FLOOR = 0        # toward -inf; an arithmetic right shift (Verilog >>> on a signed value)
    HALF_UP = 1      # round half toward +inf: (x + half) >>> s
    HALF_EVEN = 2    # convergent rounding (round half to even)
    TOWARD_ZERO = 3  # truncate toward zero
    HALF_AWAY = 4    # round half away from zero


class Overflow(enum.IntEnum):
    """What happens when a result does not fit its format."""
    WRAP = 0         # two's complement wrap (what RTL does without a saturator)
    SATURATE = 1     # clip to the format range
    CHECKED = 2      # raise OverflowError (use in tests to prove no overflow occurs)


_FLOOR, _HALF_UP, _HALF_EVEN, _TOWARD_ZERO, _HALF_AWAY = 0, 1, 2, 3, 4
_WRAP, _SATURATE, _CHECKED = 0, 1, 2
_I64_ONE = np.int64(1)


# ---- scalar kernels ---------------------------------------------------------------------------------------------

@nb.njit(cache=True)
def fx_raw_min(bits, signed):
    """Smallest raw value of a ``bits``-wide format."""
    if signed:
        return -(_I64_ONE << (bits - 1))
    return np.int64(0)


@nb.njit(cache=True)
def fx_raw_max(bits, signed):
    """Largest raw value of a ``bits``-wide format."""
    if signed:
        return (_I64_ONE << (bits - 1)) - 1
    return (_I64_ONE << bits) - 1


@nb.njit(cache=True)
def fx_overflow(x, bits, signed, ovf):
    """Fit raw x into ``bits`` (signed or not) by WRAP, SATURATE or CHECKED (raises OverflowError)."""
    lo = fx_raw_min(bits, signed)
    hi = fx_raw_max(bits, signed)
    if x >= lo and x <= hi:
        return np.int64(x)
    if ovf == _SATURATE:
        return lo if x < lo else hi
    if ovf == _CHECKED:
        raise OverflowError("fixed-point overflow (Overflow.CHECKED)")
    m = _I64_ONE << bits
    y = np.int64(x) & (m - 1)
    if signed and y >= (m >> 1):
        y -= m
    return y


@nb.njit(cache=True)
def fx_round_shift(x, s, rnd):
    """Round x * 2**-s to an integer (0 <= s <= 62) with rounding mode ``rnd``."""
    if s < 0 or s > MAX_BITS:
        raise ValueError("fx_round_shift: shift must be in 0..62")
    x = np.int64(x)
    if s == 0:
        return x
    if rnd == _FLOOR:
        return x >> s
    half = _I64_ONE << (s - 1)
    if rnd == _HALF_UP:
        return (x + half) >> s
    if rnd == _HALF_AWAY:
        if x >= 0:
            return (x + half) >> s
        return -((half - x) >> s)
    q = x >> s
    r = x - (q << s)          # 0 <= r < 2**s
    if rnd == _HALF_EVEN:
        if r > half or (r == half and (q & 1) == 1):
            q += 1
        return q
    if rnd == _TOWARD_ZERO:
        if x < 0 and r != 0:
            q += 1
        return q
    raise ValueError("fx_round_shift: unknown rounding mode")


@nb.njit(cache=True)
def _shift_left_checked(x, s, bits, signed, ovf):
    """x * 2**s fitted into ``bits`` (0 < s <= 62); overflow detected before the shift."""
    lo = fx_raw_min(bits, signed)
    hi = fx_raw_max(bits, signed)
    hi_q = hi >> s
    lo_q = -((-lo) >> s)      # ceil(lo / 2**s)
    if x > hi_q or x < lo_q:
        if ovf == _SATURATE:
            return lo if x < lo_q else hi
        if ovf == _CHECKED:
            raise OverflowError("fixed-point overflow (Overflow.CHECKED)")
        m = _I64_ONE << bits
        y = (np.int64(x) & (m - 1)) << s          # exact modulo 2**64, then wrap to bits
        return fx_overflow(y & (m - 1), bits, signed, _WRAP)
    return np.int64(x) << s


@nb.njit(cache=True)
def fx_cast(x, frac_in, bits_out, frac_out, signed_out, rnd, ovf):
    """Re-quantize raw x (frac_in fraction bits) into the format (bits_out, frac_out, signed_out)."""
    s = frac_in - frac_out
    if s >= 0:
        return fx_overflow(fx_round_shift(x, s, rnd), bits_out, signed_out, ovf)
    return _shift_left_checked(x, -s, bits_out, signed_out, ovf)


@nb.njit(cache=True)
def fx_from_float(v, bits, frac, signed, rnd, ovf):
    """Quantize the real value v into raw int64 (exact power-of-two scaling, then one rounding, then overflow).

    All steps are exact in float64: scaling by 2**frac, floor (np.floor, a float), the fraction d, and the integer
    comparison, which is done in int64 whenever |r| < 2**63 (float(hi) would round for formats wider than 53 bits).
    NaN gives 0 in WRAP and SATURATE and raises in CHECKED. Infinities saturate (WRAP has no finite answer).
    Values so large that 2**frac * |v| >= 2**115 wrap to 0 (every bit below 2**62 is zero).
    """
    if math.isnan(v):
        if ovf == _CHECKED:
            raise OverflowError("fixed-point conversion of NaN (Overflow.CHECKED)")
        return np.int64(0)
    lo = fx_raw_min(bits, signed)
    hi = fx_raw_max(bits, signed)
    if math.isinf(v) or abs(v) >= 2.0 ** (115 - frac):
        if ovf == _CHECKED:
            raise OverflowError("fixed-point overflow (Overflow.CHECKED)")
        if ovf == _SATURATE or math.isinf(v):
            return lo if v < 0.0 else hi
        return np.int64(0)
    scaled = v * (2.0 ** frac)
    f = np.floor(scaled)
    d = scaled - f                  # exact: 0 <= d < 1
    if rnd == _FLOOR:
        r = f
    elif rnd == _HALF_UP:
        r = f + 1.0 if d >= 0.5 else f
    elif rnd == _HALF_EVEN:
        odd = (f - 2.0 * np.floor(0.5 * f)) != 0.0
        r = f + 1.0 if (d > 0.5 or (d == 0.5 and odd)) else f
    elif rnd == _TOWARD_ZERO:
        r = f + 1.0 if (scaled < 0.0 and d > 0.0) else f
    elif rnd == _HALF_AWAY:
        if scaled >= 0.0:
            r = f + 1.0 if d >= 0.5 else f
        else:
            r = f + 1.0 if d > 0.5 else f
    else:
        raise ValueError("fx_from_float: unknown rounding mode")
    if abs(r) < 9.2e18:             # below 2**63: exact conversion, then the integer overflow rule
        return fx_overflow(np.int64(r), bits, signed, ovf)
    if ovf == _CHECKED:
        raise OverflowError("fixed-point overflow (Overflow.CHECKED)")
    if ovf == _SATURATE:
        return lo if r < 0.0 else hi
    m = 2.0 ** bits                 # |r| >= 2**63 > m: r - m*floor(r/m) is exact (Sterbenz)
    y = r - m * np.floor(r / m)
    if signed and y >= 0.5 * m:
        y -= m
    return np.int64(y)


@nb.njit(cache=True)
def fx_to_float(raw, frac):
    """Real value of raw (exact while |raw| < 2**53)."""
    return float(raw) * (2.0 ** (-frac))


@nb.njit(cache=True)
def fx_add(a, b, bits, signed, ovf):
    """a + b in one format (same fraction bits), fitted by ``ovf``."""
    return fx_overflow(np.int64(a) + np.int64(b), bits, signed, ovf)


@nb.njit(cache=True)
def fx_sub(a, b, bits, signed, ovf):
    """a - b in one format (same fraction bits), fitted by ``ovf``."""
    return fx_overflow(np.int64(a) - np.int64(b), bits, signed, ovf)


@nb.njit(cache=True)
def fx_neg(a, bits, signed, ovf):
    """-a fitted by ``ovf`` (-min overflows a signed format)."""
    return fx_overflow(-np.int64(a), bits, signed, ovf)


@nb.njit(cache=True)
def fx_mul(a, frac_a, b, frac_b, bits_out, frac_out, signed_out, rnd, ovf):
    """a * b: exact int64 product at frac_a + frac_b fraction bits, one rounding to frac_out, one overflow step.

    Raises ValueError (in every mode) if |a * b| would reach 2**62, which this int64-backed toolkit cannot hold.
    """
    a = np.int64(a)
    b = np.int64(b)
    if a != 0 and abs(b) > ((_I64_ONE << MAX_BITS) - 1) // abs(a):
        raise ValueError("fx_mul: exact product exceeds 2**62; use narrower operand formats")
    return fx_cast(a * b, frac_a + frac_b, bits_out, frac_out, signed_out, rnd, ovf)


@nb.njit(cache=True)
def fx_matvec(w, x, bias, frac_w, frac_x, frac_b, acc_bits, bits_out, frac_out, signed_out, rnd, ovf, relu, out):
    """Fixed-point dense layer: out = act(cast(W @ x + bias)). In place into ``out`` (int64, shape (O,)).

    w: (O, I) raw int64 at frac_w; x: (I,) at frac_x; bias: (O,) at frac_b. Products are exact; the accumulator is
    ``acc_bits`` wide at frac_w + frac_x fraction bits and is fitted by ``ovf`` after every addition (as an RTL MAC
    without guard bits would be). The bias is aligned to the accumulator by ``fx_cast`` (same rounding). The result is
    cast once to (bits_out, frac_out, signed_out); ``relu`` != 0 clamps negatives to 0 after the cast.
    """
    frac_acc = frac_w + frac_x
    for o in range(w.shape[0]):
        acc = fx_cast(bias[o], frac_b, acc_bits, frac_acc, True, rnd, ovf)
        for i in range(w.shape[1]):
            p = fx_mul(w[o, i], frac_w, x[i], frac_x, acc_bits, frac_acc, True, rnd, ovf)
            acc = fx_add(acc, p, acc_bits, True, ovf)
        y = fx_cast(acc, frac_acc, bits_out, frac_out, signed_out, rnd, ovf)
        if relu != 0 and y < 0:
            y = np.int64(0)
        out[o] = y


# ---- array kernels (used by the Python wrappers) ----------------------------------------------------------------

@nb.njit(cache=True)
def _quantize_arr(v, bits, frac, signed, rnd, ovf, out):
    for i in range(v.size):
        out[i] = fx_from_float(v[i], bits, frac, signed, rnd, ovf)


@nb.njit(cache=True)
def _overflow_arr(x, bits, signed, ovf, out):
    for i in range(x.size):
        out[i] = fx_overflow(x[i], bits, signed, ovf)


@nb.njit(cache=True)
def _cast_arr(x, frac_in, bits_out, frac_out, signed_out, rnd, ovf, out):
    for i in range(x.size):
        out[i] = fx_cast(x[i], frac_in, bits_out, frac_out, signed_out, rnd, ovf)


@nb.njit(cache=True)
def _mul_arr(a, frac_a, b, frac_b, bits_out, frac_out, signed_out, rnd, ovf, out):
    for i in range(a.size):
        out[i] = fx_mul(a[i], frac_a, b[i], frac_b, bits_out, frac_out, signed_out, rnd, ovf)


# ---- Python wrappers --------------------------------------------------------------------------------------------

_QRE = re.compile(r"^\s*(U?)Q(\d+)\.(\d+)\s*$", re.IGNORECASE)


def _mode(value, enum_cls) -> int:
    if isinstance(value, str):
        try:
            return int(enum_cls[value.upper()])
        except KeyError:
            raise ValueError(f"unknown {enum_cls.__name__} {value!r}; use one of {[m.name.lower() for m in enum_cls]}")
    return int(enum_cls(value))


@dataclass(frozen=True)
class QFormat:
    """A declared fixed-point format in TI notation (Qm.n signed: 1 + m + n bits; UQm.n unsigned: m + n bits)."""
    int_bits: int
    frac_bits: int
    signed: bool = True

    def __post_init__(self):
        if self.int_bits < 0 or self.frac_bits < 0:
            raise ValueError(f"{self}: int_bits and frac_bits must be >= 0")
        if not 1 <= self.bits <= MAX_BITS:
            raise ValueError(f"{self}: total width {self.bits} outside 1..{MAX_BITS} bits")

    @classmethod
    def parse(cls, text: str) -> "QFormat":
        """Parse 'Q7.8' (signed, 16 bits) or 'UQ8.8' (unsigned, 16 bits)."""
        m = _QRE.match(text)
        if not m:
            raise ValueError(f"cannot parse Q format {text!r}; expected 'Qm.n' or 'UQm.n'")
        return cls(int(m.group(2)), int(m.group(3)), signed=not m.group(1))

    def __str__(self) -> str:
        return f"{'' if self.signed else 'U'}Q{self.int_bits}.{self.frac_bits}"

    @property
    def bits(self) -> int:
        """Total width in bits."""
        return self.int_bits + self.frac_bits + (1 if self.signed else 0)

    @property
    def raw_min(self) -> int:
        return -(1 << (self.bits - 1)) if self.signed else 0

    @property
    def raw_max(self) -> int:
        return (1 << (self.bits - 1)) - 1 if self.signed else (1 << self.bits) - 1

    @property
    def lsb(self) -> float:
        """Value of one LSB, 2**-frac_bits."""
        return 2.0 ** -self.frac_bits

    @property
    def min_value(self) -> float:
        return self.raw_min * self.lsb

    @property
    def max_value(self) -> float:
        return self.raw_max * self.lsb

    def quantize(self, v, rounding="half_even", overflow="saturate"):
        """Real value(s) -> raw int64 (scalar in, int out; array in, int64 array out)."""
        return quantize(v, self, rounding, overflow)

    def to_float(self, raw):
        """Raw value(s) -> real value(s)."""
        return to_float(raw, self)

    def fit(self, raw, overflow="saturate"):
        """Fit raw integer(s) of any size into this format by wrap, saturate or checked."""
        arr = np.atleast_1d(np.asarray(raw, dtype=np.int64))
        out = np.empty_like(arr)
        _overflow_arr(arr.ravel(), self.bits, self.signed, _mode(overflow, Overflow), out.ravel())
        return _scalar_or(out.reshape(arr.shape), raw)


def _scalar_or(arr: np.ndarray, like):
    return int(arr.reshape(-1)[0]) if np.ndim(like) == 0 else arr


def quantize(v, fmt: QFormat, rounding="half_even", overflow="saturate"):
    """Real value(s) -> raw int64 in ``fmt``."""
    arr = np.atleast_1d(np.asarray(v, dtype=np.float64))
    out = np.empty(arr.shape, dtype=np.int64)
    _quantize_arr(arr.ravel(), fmt.bits, fmt.frac_bits, fmt.signed, _mode(rounding, Rounding),
                  _mode(overflow, Overflow), out.ravel())
    return _scalar_or(out, v)


def to_float(raw, fmt: QFormat):
    """Raw value(s) in ``fmt`` -> real value(s)."""
    r = np.asarray(raw, dtype=np.int64).astype(np.float64) * fmt.lsb
    return float(r) if np.ndim(raw) == 0 else r


def add(a, b, fmt: QFormat, overflow="saturate"):
    """Raw a + b in one format."""
    s = np.asarray(a, dtype=np.int64) + np.asarray(b, dtype=np.int64)
    return fmt.fit(s if np.ndim(s) else int(s), overflow)


def sub(a, b, fmt: QFormat, overflow="saturate"):
    """Raw a - b in one format."""
    s = np.asarray(a, dtype=np.int64) - np.asarray(b, dtype=np.int64)
    return fmt.fit(s if np.ndim(s) else int(s), overflow)


def mul(a, fa: QFormat, b, fb: QFormat, out: QFormat, rounding="half_even", overflow="saturate"):
    """Raw a (in fa) times raw b (in fb), rounded and fitted into ``out``."""
    if fa.bits + fb.bits > MAX_BITS:
        raise ValueError(f"{fa} x {fb}: operand widths {fa.bits} + {fb.bits} exceed {MAX_BITS}; the exact product "
                         "would not fit an int64")
    aa, bb = np.broadcast_arrays(np.atleast_1d(np.asarray(a, dtype=np.int64)),
                                 np.atleast_1d(np.asarray(b, dtype=np.int64)))
    aa, bb = np.ascontiguousarray(aa), np.ascontiguousarray(bb)
    res = np.empty(aa.shape, dtype=np.int64)
    _mul_arr(aa.ravel(), fa.frac_bits, bb.ravel(), fb.frac_bits, out.bits, out.frac_bits, out.signed,
             _mode(rounding, Rounding), _mode(overflow, Overflow), res.ravel())
    return int(res.reshape(-1)[0]) if (np.ndim(a) == 0 and np.ndim(b) == 0) else res


def cast(x, fin: QFormat, fout: QFormat, rounding="half_even", overflow="saturate"):
    """Re-quantize raw x from ``fin`` to ``fout``."""
    arr = np.atleast_1d(np.asarray(x, dtype=np.int64))
    res = np.empty(arr.shape, dtype=np.int64)
    _cast_arr(np.ascontiguousarray(arr).ravel(), fin.frac_bits, fout.bits, fout.frac_bits, fout.signed,
              _mode(rounding, Rounding), _mode(overflow, Overflow), res.ravel())
    return _scalar_or(res, x)


# =================================================================================================================
# Reference fixed-point PI controller (07 P1-21). Added by the hardware task: new functions only, nothing above
# changed. It is the controller of the golden vectors (mrm.sim.vectors), of the binary bridge reference
# (mrm.sim.bridge, whose C twin is C_FXP_PI_SOURCE) and of the Verilog twin in tests/hdl/.
# =================================================================================================================

from mrm.control import api as _api                        # noqa: E402  (api never imports fxp: no cycle)
from mrm.core.channels import ActId as _ActId, ObsId as _ObsId  # noqa: E402

_E_BITS = 17          # e = y - y_set for a 16-bit ADC: [-65535, 65535] needs 17 signed bits


@nb.njit(cache=True)
def fx_pi_update(e, integ, kp, ki, u0, frac, acc_bits, lo, hi, rnd):
    """One positional PI update in declared fixed-point formats. Returns ``(u, integ_new)`` (int64).

    Raw operands (int64): ``e`` = y - y_set in ADC codes (Q16.0: 17-bit signed for a 16-bit ADC); ``kp``, ``ki``
    gains at ``frac`` fraction bits (heater codes per ADC code, per step for ki); ``integ`` the integrator at
    ``frac`` fraction bits in an ``acc_bits``-wide signed register; ``u0`` the hand-off heater code; ``lo``, ``hi``
    the heater code range. With F = frac:

        a      = sat_acc(integ + ki * e)                              exact product and sum, one saturation
        integ' = clamp(a, (lo - u0) 2^F, (hi - u0) 2^F)               integrator clamping (anti-windup, 04 C3)
        v      = u0 2^F + kp * e + integ'                             exact
        u      = clamp(round_F(v), lo, hi)                            one rounding (``rnd``), one clamp

    Every product and sum is exact in int64 (the FxpPi constructor checks 17 + gain bits <= 62 and
    acc_bits + 1 <= 62); only the accumulator saturation, the integrator clamp, the output rounding and the
    output clamp lose information, and the C and Verilog twins do exactly these four steps.
    """
    a = fx_overflow(np.int64(integ) + np.int64(e) * np.int64(ki), acc_bits, True, _SATURATE)
    i_lo = (np.int64(lo) - np.int64(u0)) << frac
    i_hi = (np.int64(hi) - np.int64(u0)) << frac
    if a < i_lo:
        a = i_lo
    elif a > i_hi:
        a = i_hi
    v = (np.int64(u0) << frac) + np.int64(e) * np.int64(kp) + a
    u = fx_round_shift(v, frac, rnd)
    if u < lo:
        u = np.int64(lo)
    elif u > hi:
        u = np.int64(hi)
    return u, a


# p_i64 of the compiled controller (indices)
_PI_N, _PI_LATEST, _PI_NEW, _PI_VALID, _PI_APPLIED, _PI_ACT = 0, 1, 2, 3, 4, 5
_PI_KP, _PI_KI, _PI_FRAC, _PI_ACC, _PI_LO, _PI_HI, _PI_RND, _PI_NSET = 6, 7, 8, 9, 10, 11, 12, 13
_PI_SP = 14                                    # N set-points follow (filled from the user list)
# s_i64: [started] then blocks of N: captured, y_set, u0, integrator, last output
_SI_CAP, _SI_YSET, _SI_U0, _SI_INT, _SI_U = 0, 1, 2, 3, 4
_LATEST_ID, _VALID_ID = int(_ObsId.TX_MONITOR_LATEST), int(_ObsId.MONITOR_VALID)
_APPLIED_ID, _HEATER_ID = int(_ObsId.TX_HEATER_APPLIED), int(_ActId.TX_HEATER)


@nb.njit(cache=True)
def fxp_pi_init(spec_i64, spec_f64):
    """Compiled-ABI init of :class:`FxpPi` (07 P0-9). User integers: [kp, ki, frac, acc_bits, rnd, n_set, sp...]."""
    n = spec_i64[_api.SPEC_I_N_RINGS]
    p = np.zeros(_PI_SP + n, dtype=np.int64)
    p[_PI_N] = n
    p[_PI_LATEST] = _api.obs_offset(spec_i64, _LATEST_ID)
    p[_PI_NEW] = _api.new_offset(spec_i64, _LATEST_ID)
    p[_PI_VALID] = _api.obs_offset(spec_i64, _VALID_ID)
    p[_PI_APPLIED] = _api.obs_offset(spec_i64, _APPLIED_ID)
    p[_PI_ACT] = _api.act_offset(spec_i64, _HEATER_ID)
    if p[_PI_LATEST] < 0 or p[_PI_NEW] < 0 or p[_PI_ACT] < 0:
        raise ValueError("FxpPi needs tx.monitor.latest (with its new-sample flags) and tx.heater")
    p[_PI_KP] = _api.user_i64(spec_i64, 0)
    p[_PI_KI] = _api.user_i64(spec_i64, 1)
    p[_PI_FRAC] = _api.user_i64(spec_i64, 2)
    p[_PI_ACC] = _api.user_i64(spec_i64, 3)
    p[_PI_RND] = _api.user_i64(spec_i64, 4)
    n_set = _api.user_i64(spec_i64, 5)
    if n_set != 0 and n_set != 1 and n_set != n:
        raise ValueError("FxpPi: setpoint_codes must be None, one value or one per ring")
    p[_PI_NSET] = n_set
    for i in range(n):
        if n_set == 1:
            p[_PI_SP + i] = _api.user_i64(spec_i64, 6)
        elif n_set == n:
            p[_PI_SP + i] = _api.user_i64(spec_i64, 6 + i)
    p[_PI_LO] = spec_i64[_api.SPEC_I_HEATER_MIN]
    p[_PI_HI] = spec_i64[_api.SPEC_I_HEATER_MAX]
    s = np.zeros(1 + 5 * n, dtype=np.int64)
    return p, np.zeros(0), s, np.zeros(0)


@nb.njit(cache=True)
def fxp_pi_step(p, pf, s, sf, obs, act):
    """Compiled-ABI step of :class:`FxpPi`: per ring, one :func:`fx_pi_update` per new valid monitor sample.

    First step: u0 = the latched heater register (``tx.heater.applied``, else the held command), the integrator 0,
    the set-point from the user list or (capture mode) the first new valid sample, which is not used for an update.
    A ring without a new sample on ``tx.monitor.latest``, or with ``monitor_valid`` false, holds its command. The
    heater command of every ring is written at every step (the held output register, like the C and Verilog twins),
    so the actions do not depend on the ABI's pre-fill of the held command.
    """
    n = p[_PI_N]
    if s[0] == 0:
        for i in range(n):
            u0 = obs[p[_PI_APPLIED] + i] if p[_PI_APPLIED] >= 0 else act[p[_PI_ACT] + i]
            s[1 + _SI_U0 * n + i] = u0
            s[1 + _SI_U * n + i] = u0
            s[1 + _SI_INT * n + i] = 0
            if p[_PI_NSET] > 0:
                s[1 + _SI_CAP * n + i] = 1
                s[1 + _SI_YSET * n + i] = p[_PI_SP + i]
        s[0] = 1
    for i in range(n):
        upd = obs[p[_PI_NEW] + i] != 0 and (p[_PI_VALID] < 0 or obs[p[_PI_VALID] + i] != 0)
        if upd:
            y = obs[p[_PI_LATEST] + i]
            if s[1 + _SI_CAP * n + i] == 0:
                s[1 + _SI_YSET * n + i] = y
                s[1 + _SI_CAP * n + i] = 1
            else:
                e = y - s[1 + _SI_YSET * n + i]
                u, a = fx_pi_update(e, s[1 + _SI_INT * n + i], p[_PI_KP], p[_PI_KI], s[1 + _SI_U0 * n + i],
                                    p[_PI_FRAC], p[_PI_ACC], p[_PI_LO], p[_PI_HI], p[_PI_RND])
                s[1 + _SI_INT * n + i] = a
                s[1 + _SI_U * n + i] = u
        act[p[_PI_ACT] + i] = s[1 + _SI_U * n + i]      # the output register is driven every step (as in C / RTL)


def pi_float_bound_codes(abs_e_sum: float, abs_e: float, frac_bits: int) -> float:
    """Largest |u_fixed - u_float| (heater codes) of :class:`FxpPi` against its float64 twin on the SAME inputs.

    The fixed-point PI quantizes only its gains (|kp_q - kp|, |ki_q - ki| <= 2^-(F+1) with round-to-nearest) and
    its output; its products and sums are exact, and the saturations and clamps are 1-Lipschitz. Before the output
    rounding the two outputs therefore differ by at most d = 2^-(F+1) (sum_j |e_j| + |e_k|) (sum over the updates
    so far); the two roundings add at most 1 code: |u_fixed - u_float| <= 1 + d. ``abs_e_sum`` is sum_j |e_j|
    including step k, ``abs_e`` = |e_k|, both in ADC codes. Premises: the same u0 and set-point; a float twin with
    the same integrator clamp and no accumulator width, which is equivalent as long as the clamp range
    (hi - lo) 2^F lies inside the accumulator (true for the default Q19.20 with a 16-bit heater; a narrower
    accumulator saturates first and the bound no longer applies); float64 rounding of the twin (~1e-11 code)
    neglected. (Derived here; the docs/research files state no
    numeric float-vs-fixed rule; 07 P1-21 cites Ch17 only for the failure mode of too few integrator fraction
    bits, which tests/verification/test_vectors_equivalence.py demonstrates.)
    """
    return 1.0 + 2.0 ** (-(int(frac_bits) + 1)) * (float(abs_e_sum) + float(abs_e))


class FxpPi(_api.CompiledController):
    """Reference fixed-point set-point PI heater lock (07 P1-21), compiled ABI + Python path (same functions).

    Error e = y - y_set (monitor ADC codes, ``tx.monitor.latest``), positional PI of :func:`fx_pi_update` per ring,
    output the heater DAC code. Positive gains lock on the red (open-loop stable) side of the drop peak, where
    dy/dcode < 0 (mrm.control.baselines.pi_lock docstring). It holds the operating point it starts at (capture mode)
    or the given set-points; it does NOT acquire, map the peak or recapture after a fall-off: start it at a lock
    (``run(..., heater_code='quasi_static', settle=True)``).

    Parameters
    ----------
    kp, ki: gains in heater codes per ADC code (ki per controller step). Defaults (design choice, assumed, for g26
        nominal only): pi_lock's loop (crossover 2.5 kHz, PI zero 13 us: k_c = 2 pi 2500 13e-6 = 0.204 codes per
        code of heater error, T/T_i = 10/13) divided by the measured g26 nominal lock slope |dy/dcode| = 10.6 ADC
        codes per heater code (static, with the self-heating relief): kp = 0.0193, ki = 0.0148.
    gain_format: Q format of kp and ki ('Q3.20': 24-bit signed, LSB 2^-20); acc_format: the integrator register
        ('Q19.20': 40-bit signed); both must have the same fraction bits F.
    rounding: output rounding ('half_up' = (v + 2^(F-1)) >>> F, what RTL does); setpoint_codes: None (capture the
        first valid sample of each ring), one code, or one per ring; latency_ticks: digital pipeline (07 P0-6).
    """

    def __init__(self, kp: float = 0.0193, ki: float = 0.0148, *, gain_format: str = "Q3.20",
                 acc_format: str = "Q19.20", rounding: str = "half_up", setpoint_codes=None, latency_ticks: int = 0,
                 name: str = "fxp_pi"):
        gf, af = QFormat.parse(gain_format), QFormat.parse(acc_format)
        if not (gf.signed and af.signed):
            raise ValueError("FxpPi: gain and accumulator formats must be signed (Qm.n)")
        if gf.frac_bits != af.frac_bits:
            raise ValueError(f"FxpPi: gain {gf} and accumulator {af} must have the same fraction bits")
        if _E_BITS + gf.bits > MAX_BITS or af.bits + 1 > MAX_BITS or gf.frac_bits > 40:
            raise ValueError(f"FxpPi: {gf} x 17-bit error or {af} would not stay exact in int64")
        self.gain_format, self.acc_format = gf, af
        self.frac_bits = gf.frac_bits
        self.rounding = Rounding[rounding.upper()] if isinstance(rounding, str) else Rounding(rounding)
        self.kp, self.ki = float(kp), float(ki)
        # CHECKED: a gain outside its format is a design error, not something to saturate silently
        self.kp_raw = int(quantize(self.kp, gf, "half_even", "checked"))
        self.ki_raw = int(quantize(self.ki, gf, "half_even", "checked"))
        if setpoint_codes is None:
            sp: list[int] = []
        else:
            sp = [int(x) for x in np.atleast_1d(np.asarray(setpoint_codes)).ravel()]
        self.setpoint_codes = None if setpoint_codes is None else tuple(sp)
        user = [self.kp_raw, self.ki_raw, self.frac_bits, af.bits, int(self.rounding), len(sp), *sp]
        super().__init__(name, fxp_pi_init, fxp_pi_step, params_i64=user, latency_ticks=latency_ticks)

    @classmethod
    def from_params(cls, params) -> "FxpPi":
        """Registry entry point (``mrm.control.registry.create('mrm.control.fxp:FxpPi', params)``)."""
        return cls(**dict(params or {}))

    def reset(self, spec) -> None:
        """CompiledController.reset, after checking the exactness premise of :func:`fx_pi_update` on this spec:
        the monitor codes span < 2^16, so e = y - y_set fits the 17 signed bits the constructor sized the int64
        products for (an 18-bit ADC would need a narrower gain format)."""
        mon = spec.channels.get("tx.monitor.latest")
        if mon is None or "tx.heater" not in spec.channels:
            raise ValueError("FxpPi needs tx.monitor.latest exposed and tx.heater bound to the controller")
        if int(mon.max) - int(mon.min) >= 1 << (_E_BITS - 1):
            raise ValueError(f"FxpPi: monitor range [{mon.min}, {mon.max}] needs errors wider than {_E_BITS} bits")
        super().reset(spec)

    @property
    def kp_q(self) -> float:
        """The quantized kp (real value)."""
        return self.kp_raw * 2.0 ** -self.frac_bits

    @property
    def ki_q(self) -> float:
        """The quantized ki (real value)."""
        return self.ki_raw * 2.0 ** -self.frac_bits

    def describe(self) -> dict:
        """Formats and raw parameters (for the vectors' metadata and the C / Verilog twins)."""
        return {"controller": "fxp_pi", "kp": self.kp, "ki": self.ki, "kp_raw": self.kp_raw, "ki_raw": self.ki_raw,
                "frac_bits": self.frac_bits, "gain_format": str(self.gain_format),
                "acc_format": str(self.acc_format), "acc_bits": self.acc_format.bits,
                "rounding": self.rounding.name.lower(), "setpoint_codes": self.setpoint_codes,
                "latency_ticks": self.latency_ticks}
