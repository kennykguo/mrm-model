"""Maximal-length linear-feedback shift registers for controller randomness and dithers (07 P1-21).

An FPGA controller draws its dither signs and its exploration noise from an LFSR, not from a floating-point RNG.
This module gives the exact register a student can also write in Verilog, so the Python, numba and RTL versions
stay bit-exact.

Recurrence (Galois form, right shift)
-------------------------------------
    out   = s & 1
    s_new = (s >> 1) ^ (mask if out else 0)

with an n-bit state s != 0 and an n-bit toggle mask whose bit n-1 is set. Its characteristic polynomial is
p(x) = x^n + sum_j mask_j x^(n-1-j) (derived from the companion form; checked against brute-force periods for every
width 2..16 in tests/verification/test_fxp.py). The register has period 2^n - 1 (maximal length) iff p is
primitive over GF(2), i.e. iff x^(2^n-1) = 1 mod p and x^((2^n-1)/q) != 1 mod p for every prime q dividing 2^n - 1.

Verilog equivalent (n = 16, mask = 16'hD008):
    always @(posedge clk) s <= {1'b0, s[15:1]} ^ (s[0] ? 16'hD008 : 16'h0000);

Tap table
---------
:data:`MAXIMAL_MASKS` holds one mask per width 2..63 (int64-backed state). Provenance: computed, not copied from a
published table: for each width the first primitive trinomial x^n + x^k + 1 in increasing k, else the first
primitive pentanomial in lexicographic order. :func:`is_maximal` re-checks every entry in the tests by the order
test above (factorization of 2^n - 1 by trial division, Miller-Rabin and Pollard rho).
"""
from __future__ import annotations

import math
import random

import numba as nb
import numpy as np

__all__ = [
    "MAXIMAL_MASKS", "Lfsr", "lfsr_step", "lfsr_bits", "lfsr_pm1", "lfsr_fill_pm1", "lfsr_period_bruteforce",
    "is_maximal", "charpoly", "default_mask",
]

# Galois toggle masks, width -> mask (computed and verified, see module docstring).
MAXIMAL_MASKS: dict[int, int] = {
    2: 0x3, 3: 0x6, 4: 0xC, 5: 0x14, 6: 0x30, 7: 0x60, 8: 0xE1, 9: 0x110, 10: 0x240, 11: 0x500, 12: 0xE08,
    13: 0x1C80, 14: 0x3802, 15: 0x6000, 16: 0xD008, 17: 0x12000, 18: 0x20400, 19: 0x72000, 20: 0x90000,
    21: 0x140000, 22: 0x300000, 23: 0x420000, 24: 0xE10000, 25: 0x1200000, 26: 0x3880000, 27: 0x7200000,
    28: 0x9000000, 29: 0x14000000, 30: 0x38000040, 31: 0x48000000, 32: 0xE0000200, 33: 0x100080000,
    34: 0x380000040, 35: 0x500000000, 36: 0x801000000, 37: 0x1C08000000, 38: 0x3401000000, 39: 0x4400000000,
    40: 0xE000000010, 41: 0x12000000000, 42: 0x38000001000, 43: 0x70040000000, 44: 0xD0000000020,
    45: 0x1B0000000000, 46: 0x341000000000, 47: 0x420000000000, 48: 0xD00000080000, 49: 0x1008000000000,
    50: 0x3800200000000, 51: 0x7000000400000, 52: 0x9000000000000, 53: 0x1C400000000000, 54: 0x38001000000000,
    55: 0x40000040000000, 56: 0xE0000000002000, 57: 0x102000000000000, 58: 0x200004000000000,
    59: 0x700000400000000, 60: 0xC00000000000000, 61: 0x1C80000000000000, 62: 0x3400000200000000,
    63: 0x6000000000000000,
}


def default_mask(width: int) -> int:
    """The tabulated maximal-length mask for ``width`` bits (2..63)."""
    if width not in MAXIMAL_MASKS:
        raise ValueError(f"LFSR width must be in 2..63, got {width}")
    return MAXIMAL_MASKS[width]


# ---- per-step kernels (numba; int64 state, mask < 2**63) -------------------------------------------------------

@nb.njit(cache=True)
def lfsr_step(state, mask):
    """One Galois right-shift step. Returns (new_state, output_bit). state and mask are int64, state != 0."""
    out = state & 1
    s = state >> 1
    if out:
        s ^= mask
    return s, out


@nb.njit(cache=True)
def lfsr_bits(state, mask, k):
    """Clock the register k times (1 <= k <= 62) and collect the output bits LSB-first into an int64.

    Returns (new_state, value) with 0 <= value < 2**k. Use it for a uniform k-bit dither code.
    """
    v = np.int64(0)
    s = state
    for i in range(k):
        s, b = lfsr_step(s, mask)
        v |= np.int64(b) << i
    return s, v


@nb.njit(cache=True)
def lfsr_pm1(state, mask):
    """One step, mapped to a dither sign: returns (new_state, +1 or -1). Output bit 1 -> +1, 0 -> -1."""
    s, b = lfsr_step(state, mask)
    return s, 2 * b - 1


@nb.njit(cache=True)
def lfsr_fill_pm1(states, mask, out):
    """Vector form: for each i, advance states[i] one step and write the sign to out[i]. In place."""
    for i in range(states.shape[0]):
        s, sgn = lfsr_pm1(states[i], mask)
        states[i] = s
        out[i] = sgn


@nb.njit(cache=True)
def lfsr_period_bruteforce(mask, seed, max_steps):
    """Steps until the state returns to ``seed`` (or -1 after max_steps). Test helper for small widths."""
    s = seed
    for k in range(1, max_steps + 1):
        s, _b = lfsr_step(s, mask)
        if s == seed:
            return k
    return -1


class Lfsr:
    """Python convenience wrapper around the kernels above (same bits as the numba form).

    Parameters: width (2..63), seed (non-zero, < 2**width), mask (default: :func:`default_mask`).
    """

    def __init__(self, width: int, seed: int = 1, mask: int | None = None):
        self.width = int(width)
        self.mask = default_mask(self.width) if mask is None else int(mask)
        if not (self.mask >> (self.width - 1)) & 1 or self.mask >= (1 << self.width):
            raise ValueError(f"mask 0x{self.mask:X} must have bit {self.width - 1} set and fit in {self.width} bits")
        seed = int(seed)
        if not 0 < seed < (1 << self.width):
            raise ValueError(f"LFSR seed must be in 1..2**{self.width}-1, got {seed}")
        self.state = seed

    @property
    def period(self) -> int:
        """2**width - 1 when the mask is maximal (see :func:`is_maximal`)."""
        return (1 << self.width) - 1

    def step(self) -> int:
        """Advance one step and return the output bit."""
        s, b = lfsr_step(np.int64(self.state), np.int64(self.mask))
        self.state = int(s)
        return int(b)

    def bits(self, k: int) -> int:
        """Advance k steps (1 <= k <= 62) and return the k output bits, LSB-first."""
        if not 1 <= int(k) <= 62:
            raise ValueError(f"Lfsr.bits: k must be in 1..62 (an int64 holds the result), got {k}")
        s, v = lfsr_bits(np.int64(self.state), np.int64(self.mask), int(k))
        self.state = int(s)
        return int(v)

    def pm1(self) -> int:
        """Advance one step and return a dither sign (+1 or -1)."""
        return 2 * self.step() - 1


# ---- verification utilities (plain Python integers) -----------------------------------------------------------

def charpoly(width: int, mask: int) -> int:
    """Characteristic polynomial of the Galois register as an int (bit k = coefficient of x^k)."""
    rev = 0
    for j in range(width):
        if (mask >> j) & 1:
            rev |= 1 << (width - 1 - j)
    return (1 << width) | rev


def _is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)   # deterministic Miller-Rabin bases for n < 3.3e24
    for p in small:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _pollard_brent(n: int) -> int:
    if n % 2 == 0:
        return 2
    rnd = random.Random(n)
    while True:
        y, c, m = rnd.randrange(1, n), rnd.randrange(1, n), 128
        g = r = q = 1
        x = ys = y
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
        if g != n:
            return g


def _prime_factors(n: int) -> set[int]:
    out: set[int] = set()
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if _is_probable_prime(m):
            out.add(m)
            continue
        for p in range(2, 1000):
            if m % p == 0:
                out.add(p)
                while m % p == 0:
                    m //= p
                stack.append(m)
                break
        else:
            d = _pollard_brent(m)
            stack += [d, m // d]
    return out


def _x_pow_mod(e: int, p: int, n: int) -> int:
    """x^e mod p over GF(2), p of degree n."""
    def mulmod(a: int, b: int) -> int:
        r = 0
        while b:
            if b & 1:
                r ^= a
            b >>= 1
            a <<= 1
            if (a >> n) & 1:
                a ^= p
        return r
    r, base = 1, 2 if n > 1 else 2 ^ p
    while e:
        if e & 1:
            r = mulmod(r, base)
        base = mulmod(base, base)
        e >>= 1
    return r


def is_maximal(width: int, mask: int) -> bool:
    """True iff the Galois register (width, mask) has period 2**width - 1 (order test on its characteristic poly)."""
    if not (mask >> (width - 1)) & 1 or mask >= (1 << width):
        return False
    p = charpoly(width, mask)
    order = (1 << width) - 1
    if _x_pow_mod(order, p, width) != 1:
        return False
    return all(_x_pow_mod(order // q, p, width) != 1 for q in _prime_factors(order))
