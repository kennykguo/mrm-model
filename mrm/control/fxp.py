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
  rounding, then one overflow step), ``fx_cast`` (re-quantize between formats), ``fx_matvec`` (dense layer with a
  declared accumulator width: the fixed-point MLP helper of 07 P1-22).

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
