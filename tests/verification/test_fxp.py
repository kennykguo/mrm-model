"""Verification of mrm.control.fxp and mrm.control.lfsr (07 P1-21): Q formats, rounding modes and overflow against an
exact big-integer reference, the 05 C12 golden PI, the fixed-point dense layer, maximal-length LFSRs, and identical
results with NUMBA_DISABLE_JIT=1."""
import json
import math
import os
import subprocess
import sys
from fractions import Fraction

import numba as nb
import numpy as np
import pytest

from mrm.control import fxp, lfsr
from mrm.control.fxp import Overflow, QFormat, Rounding

JIT_OFF = bool(nb.config.DISABLE_JIT)


# ---- exact reference (Python integers and fractions) ---------------------------------------------------------------

def ref_round(q: Fraction, mode: Rounding) -> int:
    f = math.floor(q)
    d = q - f
    if mode == Rounding.FLOOR:
        return f
    if mode == Rounding.HALF_UP:
        return f + 1 if d >= Fraction(1, 2) else f
    if mode == Rounding.HALF_EVEN:
        if d > Fraction(1, 2) or (d == Fraction(1, 2) and f % 2 == 1):
            return f + 1
        return f
    if mode == Rounding.TOWARD_ZERO:
        return math.trunc(q)
    if mode == Rounding.HALF_AWAY:
        a = abs(q)
        r = math.floor(a + Fraction(1, 2))
        return r if q >= 0 else -r
    raise ValueError(mode)


def ref_fit(x: int, bits: int, signed: bool, ovf: Overflow):
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    if lo <= x <= hi:
        return x
    if ovf == Overflow.SATURATE:
        return lo if x < lo else hi
    if ovf == Overflow.CHECKED:
        return "overflow"
    y = x % (1 << bits)
    return y - (1 << bits) if signed and y >= (1 << (bits - 1)) else y


ROUNDINGS = list(Rounding)
OVERFLOWS = list(Overflow)


# ---- formats --------------------------------------------------------------------------------------------------------

def test_qformat_parse_and_ranges():
    q = QFormat.parse("Q7.8")
    assert (q.bits, q.raw_min, q.raw_max, q.lsb) == (16, -32768, 32767, 1 / 256)
    u = QFormat.parse("UQ8.8")
    assert (u.bits, u.raw_min, u.raw_max) == (16, 0, 65535)
    assert str(q) == "Q7.8" and str(u) == "UQ8.8"
    assert QFormat.parse("q0.15").bits == 16
    for bad in ("Q7", "Q.8", "X7.8", "Q7.8.1"):
        with pytest.raises(ValueError):
            QFormat.parse(bad)
    with pytest.raises(ValueError):
        QFormat(40, 30)          # 71 bits > 62


def test_c12_reference_values():
    """docs/research/05_tools.md check C12: Kp = 0.37512 in Q7.8 -> 0.375 (raw 96); 40000 saturates to 32767."""
    q = QFormat.parse("Q7.8")
    assert q.quantize(0.37512) == 96 and q.to_float(96) == 0.375
    assert q.fit(40000, "saturate") == 32767
    assert q.fit(40000, "wrap") == 40000 - 65536
    with pytest.raises(OverflowError):
        q.fit(40000, "checked")
    assert list(q.quantize([0.37512, -1000.0, 1000.0])) == [96, -32768, 32767]


# ---- rounding and overflow against the exact reference ---------------------------------------------------------------

@pytest.mark.parametrize("mode", ROUNDINGS)
def test_round_shift_exact(mode):
    for x in range(-70, 71):
        for s in range(0, 6):
            assert fxp.fx_round_shift(x, s, int(mode)) == ref_round(Fraction(x, 1 << s), mode), (x, s, mode)
    for x in (2**61 - 1, -(2**61), 12345678901234567):
        for s in (1, 17, 40, 62):
            assert fxp.fx_round_shift(x, s, int(mode)) == ref_round(Fraction(x, 1 << s), mode)


@pytest.mark.parametrize("ovf", OVERFLOWS)
def test_overflow_exact(ovf):
    for bits in (1, 2, 5, 16, 33, 62):
        for signed in (True, False):
            for x in (-(1 << 62), -(1 << bits), -(1 << (bits - 1)) - 1, -1, 0, 1, (1 << (bits - 1)),
                      (1 << bits) + 3, (1 << 62) - 1):
                want = ref_fit(x, bits, signed, ovf)
                if want == "overflow":
                    with pytest.raises(OverflowError):
                        fxp.fx_overflow(x, bits, signed, int(ovf))
                else:
                    assert fxp.fx_overflow(x, bits, signed, int(ovf)) == want, (x, bits, signed)


def test_from_float_matches_reference():
    r = np.random.default_rng(0)
    vals = list(r.normal(0, 40, 300)) + [0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 127.99, -128.0, 1e300, -1e300]
    for fmt in (QFormat(7, 8), QFormat(0, 15), QFormat(8, 8, signed=False), QFormat(20, 30)):
        for mode in ROUNDINGS:
            for ovf in (Overflow.WRAP, Overflow.SATURATE):
                got = fxp.quantize(np.array(vals), fmt, mode.name, ovf.name)
                for v, g in zip(vals, got):
                    raw = ref_round(Fraction(v) * (1 << fmt.frac_bits), mode)
                    assert g == ref_fit(raw, fmt.bits, fmt.signed, ovf), (v, fmt, mode, ovf)
    wide = QFormat(55, 6)                                  # 62 bits: wider than a float64 significand
    big = list(r.normal(0, 2.0**54, 200)) + [2.0**55, -2.0**55, 2.0**55 - 64.0, 3.0 * 2**60]
    for mode in ROUNDINGS:
        for ovf in (Overflow.WRAP, Overflow.SATURATE):
            got = fxp.quantize(np.array(big), wide, mode.name, ovf.name)
            for v, g in zip(big, got):
                assert g == ref_fit(ref_round(Fraction(v) * 64, mode), wide.bits, True, ovf), (v, mode, ovf)
    assert fxp.quantize(2.0**61, QFormat(61, 0)) == 2**61 - 1          # float(hi) would round up to 2**61
    assert fxp.quantize(2.0**61, QFormat(61, 0), overflow="wrap") == -2**61
    assert fxp.quantize(float("nan"), QFormat(7, 8)) == 0
    with pytest.raises(OverflowError):
        fxp.quantize(float("nan"), QFormat(7, 8), overflow="checked")
    assert fxp.quantize(float("inf"), QFormat(7, 8), overflow="wrap") == 32767


def test_mul_and_cast_match_reference():
    r = np.random.default_rng(1)
    cases = [(QFormat(7, 8), QFormat(1, 14), QFormat(7, 8)), (QFormat(15, 16), QFormat(3, 12), QFormat(19, 12)),
             (QFormat(8, 8, False), QFormat(0, 16, False), QFormat(4, 4)), (QFormat(3, 4), QFormat(3, 4), QFormat(0, 12))]
    for fa, fb, fo in cases:
        a = r.integers(fa.raw_min, fa.raw_max + 1, 200)
        b = r.integers(fb.raw_min, fb.raw_max + 1, 200)
        for mode in ROUNDINGS:
            for ovf in (Overflow.WRAP, Overflow.SATURATE):
                got = fxp.mul(a, fa, b, fb, fo, mode.name, ovf.name)
                for x, y, g in zip(a, b, got):
                    p = Fraction(int(x) * int(y), 1 << (fa.frac_bits + fb.frac_bits)) * (1 << fo.frac_bits)
                    assert g == ref_fit(ref_round(p, mode), fo.bits, fo.signed, ovf)
                cin = fxp.cast(a, fa, fo, mode.name, ovf.name)
                for x, g in zip(a, cin):
                    want = ref_round(Fraction(int(x) * (1 << fo.frac_bits), 1 << fa.frac_bits), mode)
                    assert g == ref_fit(want, fo.bits, fo.signed, ovf)
    with pytest.raises(OverflowError):
        fxp.mul(32767, QFormat(7, 8), 32767, QFormat(7, 8), QFormat(7, 8), overflow="checked")


def test_cast_left_shift_overflow():
    fin, fout = QFormat(15, 0), QFormat(5, 10)       # 16-bit integer into 16 bits with 10 fraction bits
    for x in (-40000, -33, -32, -31, 0, 31, 32, 40000, 2**40):
        want = Fraction(x) * (1 << 10)
        for ovf in (Overflow.WRAP, Overflow.SATURATE):
            assert fxp.fx_cast(x, 0, fout.bits, 10, True, 0, int(ovf)) == ref_fit(int(want), fout.bits, True, ovf)
        if ref_fit(int(want), fout.bits, True, Overflow.CHECKED) == "overflow":
            with pytest.raises(OverflowError):
                fxp.fx_cast(x, 0, fout.bits, 10, True, 0, int(Overflow.CHECKED))
    assert fin.bits == 16


def test_mul_refuses_products_beyond_int64():
    with pytest.raises(ValueError):
        fxp.fx_mul(2**40, 0, 2**30, 0, 62, 0, True, 0, 0)
    with pytest.raises(ValueError):
        fxp.mul(1, QFormat(40, 0), 1, QFormat(30, 0), QFormat(40, 0))


def test_add_sub_neg():
    q = QFormat(7, 8)
    assert fxp.add(32767, 1, q, "wrap") == -32768
    assert fxp.add(32767, 1, q, "saturate") == 32767
    assert fxp.sub(-32768, 1, q, "saturate") == -32768
    assert list(fxp.add(np.array([1, 2]), np.array([3, 4]), q)) == [4, 6]
    assert fxp.fx_neg(-32768, 16, True, int(Overflow.WRAP)) == -32768
    assert fxp.fx_neg(-32768, 16, True, int(Overflow.SATURATE)) == 32767
    with pytest.raises(OverflowError):
        fxp.fx_neg(-32768, 16, True, int(Overflow.CHECKED))


# ---- a controller written with the toolkit reproduces the cocotb-verified golden PI ----------------------------------

KP, KI, SH = 96, 5, 8          # docs/research/05_tools_smoke/t12_fixed_point/test_pi.py


def golden_pi(state, sp, me):
    """Plain-integer golden model of t12 (verified bit-exact against Icarus and Verilator RTL, 05 C12)."""
    e = sp - me
    acc = max(-2**31, min(2**31 - 1, state["acc"] + KI * e))
    v = (KP * e + acc) >> SH
    state["acc"] = acc
    return 0 if v < 0 else 4095 if v > 4095 else v


@nb.njit(cache=True)
def fxp_pi(sp, me, acc):
    """The same PI written with declared formats: e in 13 bits, acc Q31.0 saturating, output UQ12.0 saturating."""
    e = fxp.fx_sub(sp, me, 14, True, 1)
    acc = fxp.fx_add(acc, fxp.fx_mul(5, 0, e, 0, 32, 0, True, 0, 1), 32, True, 1)
    v = fxp.fx_add(fxp.fx_mul(96, 0, e, 0, 40, 0, True, 0, 1), acc, 40, True, 1)
    out = fxp.fx_cast(v, 8, 12, 0, False, 0, 1)            # >> 8 (floor), then clamp to 0..4095
    return out, acc


def test_fxp_pi_matches_golden_model():
    r = np.random.default_rng(1)
    st = {"acc": 0}
    acc = 0
    for _ in range(2000):
        sp, me = int(r.integers(-3000, 3001)), int(r.integers(-3000, 3001))
        want = golden_pi(st, sp, me)
        got, acc = fxp_pi(sp, me, acc)
        assert got == want and acc == st["acc"]


def test_matvec_dense_layer():
    r = np.random.default_rng(2)
    fw, fx, fb, fo = QFormat(1, 14), QFormat(3, 12), QFormat(7, 8), QFormat(7, 8)
    w = r.integers(fw.raw_min, fw.raw_max + 1, (5, 7)).astype(np.int64)
    x = r.integers(fx.raw_min, fx.raw_max + 1, 7).astype(np.int64)
    bias = r.integers(-2000, 2000, 5).astype(np.int64)
    out = np.zeros(5, dtype=np.int64)
    fxp.fx_matvec(w, x, bias, 14, 12, 8, 48, fo.bits, 8, True, int(Rounding.HALF_EVEN), int(Overflow.SATURATE),
                  1, out)
    for o in range(5):
        acc = bias[o] * (1 << (26 - 8)) + sum(int(w[o, i]) * int(x[i]) for i in range(7))
        y = ref_fit(ref_round(Fraction(acc, 1 << (26 - 8)), Rounding.HALF_EVEN), 16, True, Overflow.SATURATE)
        assert out[o] == max(y, 0)


# ---- LFSR --------------------------------------------------------------------------------------------------------------

def test_every_tabulated_mask_is_maximal():
    for width, mask in lfsr.MAXIMAL_MASKS.items():
        assert lfsr.is_maximal(width, mask), width
    assert set(lfsr.MAXIMAL_MASKS) == set(range(2, 64))


@pytest.mark.parametrize("width", range(2, 13 if JIT_OFF else 19))
def test_bruteforce_period(width):
    period = lfsr.lfsr_period_bruteforce(lfsr.MAXIMAL_MASKS[width], 1, 2**width)
    assert period == 2**width - 1


def test_order_test_rejects_non_maximal_masks():
    # width 4: x^4 + x^2 + 1 = (x^2 + x + 1)^2 is reducible; check against brute force for every 4-bit mask
    for mask in range(8, 16):
        brute = lfsr.lfsr_period_bruteforce(mask, 1, 64) == 15
        assert lfsr.is_maximal(4, mask) == brute
    for mask in range(1 << 7, 1 << 8):
        brute = lfsr.lfsr_period_bruteforce(mask, 1, 600) == 255
        assert lfsr.is_maximal(8, mask) == brute


def test_lfsr_recurrence_and_wrapper():
    w, m = 16, lfsr.MAXIMAL_MASKS[16]
    s = 0xACE1
    g = lfsr.Lfsr(w, seed=0xACE1)
    ref_bits = []
    for _ in range(200):                     # the Verilog form of the module docstring, in plain integers
        b = s & 1
        s = (s >> 1) ^ (m if b else 0)
        ref_bits.append(b)
    assert [g.step() for _ in range(200)] == ref_bits
    g2 = lfsr.Lfsr(w, seed=0xACE1)
    v = g2.bits(16)
    assert v == sum(b << i for i, b in enumerate(ref_bits[:16]))
    with pytest.raises(ValueError):
        lfsr.Lfsr(8, seed=0)
    with pytest.raises(ValueError):
        lfsr.Lfsr(8, mask=0x0F)


def test_dither_balance_over_one_period():
    w = 10
    st = np.array([1, 5, 77], dtype=np.int64)
    acc = np.zeros(3, dtype=np.int64)
    out = np.zeros(3, dtype=np.int64)
    for _ in range(2**w - 1):
        lfsr.lfsr_fill_pm1(st, lfsr.MAXIMAL_MASKS[w], out)
        acc += out
    assert list(acc) == [1, 1, 1]            # 2**(w-1) ones and 2**(w-1) - 1 zeros per period
    assert list(st) == [1, 5, 77]


_PARITY_SCRIPT = r"""
import json, numpy as np
from mrm.control import fxp, lfsr
r = np.random.default_rng(5)
a = r.integers(-30000, 30000, 50); b = r.integers(-30000, 30000, 50)
out = {
  "mul": [int(fxp.fx_mul(int(x), 8, int(y), 12, 20, 10, True, m, 1)) for x, y in zip(a, b) for m in range(5)],
  "rs": [int(fxp.fx_round_shift(int(x), s, m)) for x in a for s in (0, 3, 9) for m in range(5)],
  "q": [int(fxp.fx_from_float(float(v), 16, 8, True, m, 0)) for v in r.normal(0, 200, 40) for m in range(5)],
  "wrap": [int(fxp.fx_cast(int(x), 0, 12, 4, True, 2, 0)) for x in a],
  "lfsr": int(lfsr.lfsr_bits(np.int64(0x1234), np.int64(lfsr.MAXIMAL_MASKS[31]), 40)[1]),
}
print(json.dumps(out))
"""


def test_identical_results_without_jit():
    env = dict(os.environ)
    here = subprocess.run([sys.executable, "-c", _PARITY_SCRIPT], capture_output=True, text=True, env=env, check=True)
    env["NUMBA_DISABLE_JIT"] = "0" if JIT_OFF else "1"
    other = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", _PARITY_SCRIPT], capture_output=True,
                           text=True, env=env, check=True)
    assert json.loads(here.stdout) == json.loads(other.stdout)
