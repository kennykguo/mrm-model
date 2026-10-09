"""Counter-based random numbers for the plant kernel (07 P1-13).

Every random number in the model is a pure function of a counter. Nothing carries a generator state, so a draw
does not depend on how many draws happened before it, on the order in which threads ran, or on the batch size.
This is what 07 P1-13 asks for: "a counter-based RNG in numba (Philox or splitmix64), keyed by (seed, stream_id,
b, ring, tick)", with the tests "element b of a B = 64 run equals the B = 1 run with seed b, for any thread count",
"toggling fidelity.shot_noise leaves the ADC-noise draws unchanged" and "two controllers on one seed see the same
noise (common random numbers)".

Generator
---------
Philox4x32-10 (Salmon, Moraes, Dror, Shaw, "Parallel random numbers: as easy as 1, 2, 3", SC'11, Random123). It
maps a 128-bit counter and a 64-bit key to 128 random bits, passes TestU01 BigCrush, and needs only 32 x 32 -> 64
bit products. Those products never overflow a uint64, so the code gives identical bits under numba and under
NUMBA_DISABLE_JIT=1 (plain NumPy scalars), with no overflow warnings. The implementation is checked against the
Random123 known-answer vectors in tests/verification/test_core_rng.py.

Keying convention (how (seed, stream_id, b, ring, tick) map onto Philox)
-----------------------------------------------------------------------
- Batch element b uses its own seed, ``seed_b = base_seed + b`` (see :func:`batch_seeds`). The pair (seed, b)
  therefore enters as ``seed_b``. This is what makes element b of a batch bit-identical to a B = 1 run with seed
  ``base_seed + b``. It also means that the runs (base 0, element 1) and (base 1, element 0) share a stream; that is
  intended (common random numbers) and must be kept in mind when two Monte Carlo sets are pooled.
- ``key = stream_key(seed_b, stream_id)``: one Philox call on the counter (stream_lo, stream_hi, 0, KEYGEN_TAG)
  under the 64-bit seed as key. Streams are therefore independent even for adjacent seeds.
- Counter of a draw: ``(tick_lo32, tick_hi32, ring, slot << 12 | inner)``.
  ``tick`` is the int64 plant tick (07 P0-6) and must be >= 0; ``ring`` is any per-element index below 2**32 (ring,
  laser line, sensor, sample index); ``slot`` (< 2**20) separates several draws at the same (ring, tick), e.g. the
  M ADC samples of one tick or the two quadratures of a noise source; ``inner`` (< 2**12) is used internally by the
  rejection samplers and is never set by callers.

Distributions
-------------
uniform (0, 1) with 53 random bits; standard normal by Box-Muller; Poisson by inversion for lam < 10 and by PTRS
(Hoermann 1993, "The transformed rejection method for generating Poisson random variables", Insurance: Mathematics
and Economics 12, 39-45) otherwise; binomial by inversion (BINV, Kachitvichyanukul and Schmeiser 1988, CACM 31) for
n*min(p, 1-p) < 10 and by BTRS (Hoermann 1993, "The generation of binomial random variates", J. Statist. Comput.
Simul. 46, 101-110) otherwise. The PTRS and BTRS constants are the published ones (also used by NumPy and
TensorFlow).

Stream ids
----------
A fixed enum (:class:`Stream`). Parametric streams (``adc_<channel>``, ``scenario_<name>``, ``dac_<channel>``) get
``category | crc24(name)``. crc32 is stable across processes and platforms (unlike ``hash()``), and
:func:`stream_id` refuses a second name that collides with one already issued in the process.

All per-draw functions are ``@njit`` with explicit signatures (key uint64; ring, tick, slot int64) and take only
scalars. The explicit signatures matter: a key returned to Python is an int above 2**63, and a lazily compiled
int64 specialization would otherwise be picked and overflow. The ``fill_*`` helpers take arrays and use ``prange``
over the batch only, so results do not depend on the thread count.
"""
from __future__ import annotations

import enum
import math
import zlib

import numba as nb
import numpy as np

__all__ = [
    "Stream", "stream_id", "batch_seeds", "philox4x32_10", "stream_key", "random_u32x4", "uniform", "uniform2",
    "normal", "normal2", "poisson", "binomial", "fill_uniform", "fill_normal", "fill_poisson", "fill_binomial",
    "SLOT_MAX", "INNER_MAX",
]

# ---- Philox4x32-10 constants (Salmon et al. SC'11; Random123 philox.h) -----------------------------------------
_M32 = np.uint64(0xFFFFFFFF)
_SH32 = np.uint64(32)
_PHILOX_M0 = np.uint64(0xD2511F53)    # Random123 PHILOX_M4x32_0
_PHILOX_M1 = np.uint64(0xCD9E8D57)    # Random123 PHILOX_M4x32_1
_PHILOX_W0 = np.uint64(0x9E3779B9)    # Random123 PHILOX_W32_0 (golden ratio)
_PHILOX_W1 = np.uint64(0xBB67AE85)    # Random123 PHILOX_W32_1 (sqrt(3) - 1)
_KEYGEN_TAG = np.uint64(0x5EED5EED)   # arbitrary constant that separates key derivation from draws (design choice)
_SLOT_SHIFT = np.uint64(12)
_INNER_MASK = np.uint64(0xFFF)
SLOT_MAX = (1 << 20) - 1              # largest slot index: 32 - 12 bits of counter word 3
INNER_MAX = (1 << 12) - 1             # rejection-sampler iterations before the documented fallback

_TWO_PI = 2.0 * math.pi
_INV_2_53 = 1.0 / 9007199254740992.0  # 2**-53
_TWO_26 = 67108864.0                  # 2**26
_ONE_MINUS_ULP = 1.0 - 2.0**-53       # largest float64 below 1.0
_TWO_62 = 4611686018427387904.0       # 2**62: Poisson saturation level (fits an int64 with margin)
_I_TWO_62 = 4611686018427387904


class Stream(enum.IntEnum):
    """Fixed stream-id enum (07 P1-13). Values never change; new streams are only appended.

    The top byte is the category. Parametric names (``adc_<channel>``, ``scenario_<name>``, ``dac_<channel>``)
    combine their category with the 24-bit crc of the name, see :func:`stream_id`.
    """
    VARIATION = 0x01000000      # per-ring/per-line device variation (realize)
    UNCERTAINTY = 0x02000000    # epistemic world draws (realize)
    LASER_RIN = 0x03000000      # laser relative intensity noise
    LASER_DRIFT = 0x04000000    # laser wavelength/power random walk
    SHOT_NOISE = 0x05000000     # photodiode shot noise (incl. dark-current shot noise)
    CONTROLLER = 0x06000000     # randomness owned by a Python controller (dithers, exploration)
    ADC = 0x07000000            # category of adc_<channel>: AFE + quantizer noise of one converter channel
    SCENARIO = 0x08000000       # category of scenario_<name>
    DATAPATH = 0x09000000       # bit-statistics binomials and error-count Poisson draws (07 P0-11); added here
    PATTERN = 0x0A000000        # ones-density pattern noise (fidelity.pattern_noise, 07 P0-11); added here
    DAC = 0x0B000000            # category of dac_<channel>: DAC output noise (ring.bias_dac.noise_rms_codes); added
    SCHEDULE = 0x0C000000       # controller tick jitter (controller_interface.tick_jitter_s, 07 P2-7); added


_CATEGORY_BY_PREFIX = {"adc_": Stream.ADC, "scenario_": Stream.SCENARIO, "dac_": Stream.DAC}
_FIXED_BY_NAME = {m.name.lower(): int(m) for m in Stream if m not in _CATEGORY_BY_PREFIX.values()}
_ISSUED: dict[int, str] = {}


def stream_id(name: str) -> int:
    """Return the fixed integer stream id for a stream name.

    ``variation``, ``uncertainty``, ``laser_rin``, ``laser_drift``, ``shot_noise``, ``controller``, ``datapath``,
    ``pattern`` and ``schedule`` map to the enum. ``adc_<channel>``, ``scenario_<name>`` and ``dac_<channel>`` map to
    ``category | (crc32(rest) & 0xFFFFFF)``. Raises ValueError for an unknown name or for a crc collision between
    two different parametric names issued in this process.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"stream name must be a non-empty string, got {name!r}")
    key = name.lower()
    if key in _FIXED_BY_NAME:
        return _FIXED_BY_NAME[key]
    for prefix, cat in _CATEGORY_BY_PREFIX.items():
        if key.startswith(prefix):
            rest = key[len(prefix):]           # case-insensitive like the prefix (channel names are lower case)
            if not rest:
                raise ValueError(f"stream name {name!r} needs a suffix after {prefix!r}")
            sid = int(cat) | (zlib.crc32(rest.encode("utf-8")) & 0xFFFFFF)
            prev = _ISSUED.setdefault(sid, key)
            if prev != key:
                raise ValueError(f"stream ids of {prev!r} and {name!r} collide (0x{sid:08x}); rename one of them")
            return sid
    known = sorted(_FIXED_BY_NAME) + [p + "<name>" for p in _CATEGORY_BY_PREFIX]
    raise ValueError(f"unknown stream {name!r}; known: {known}")


def batch_seeds(base_seed: int, n_batch: int) -> np.ndarray:
    """Per-element seeds ``base_seed + b`` for b in 0..n_batch-1, as int64 (the convention of this module).

    Seeds must lie in [0, 2**63 - n_batch]: a negative seed has no uint64 value under NUMBA_DISABLE_JIT.
    """
    base = int(base_seed)
    if base < 0 or base + int(n_batch) > 2**63 - 1:
        raise ValueError(f"seed {base_seed} out of range [0, 2**63 - B]")
    return base + np.arange(int(n_batch), dtype=np.int64)


# ---- Philox core ----------------------------------------------------------------------------------------------

@nb.njit("UniTuple(uint64, 4)(uint64, uint64, uint64, uint64, uint64, uint64)", cache=True)
def _round(c0, c1, c2, c3, k0, k1):
    p0 = _PHILOX_M0 * c0              # 32 x 32 -> 64 bit product: never overflows a uint64
    p1 = _PHILOX_M1 * c2
    hi0 = p0 >> _SH32
    lo0 = p0 & _M32
    hi1 = p1 >> _SH32
    lo1 = p1 & _M32
    return hi1 ^ c1 ^ k0, lo1, hi0 ^ c3 ^ k1, lo0


@nb.njit("UniTuple(uint64, 4)(uint64, uint64, uint64, uint64, uint64, uint64)", cache=True)
def philox4x32_10(c0, c1, c2, c3, k0, k1):
    """Philox4x32 with 10 rounds. Inputs and outputs are uint64 values holding 32-bit words.

    Counter (c0, c1, c2, c3), key (k0, k1). Returns the four output words. Reference: Random123 philox.h.
    """
    c0 = np.uint64(c0) & _M32
    c1 = np.uint64(c1) & _M32
    c2 = np.uint64(c2) & _M32
    c3 = np.uint64(c3) & _M32
    k0 = np.uint64(k0) & _M32
    k1 = np.uint64(k1) & _M32
    for r in range(10):
        if r > 0:
            k0 = (k0 + _PHILOX_W0) & _M32
            k1 = (k1 + _PHILOX_W1) & _M32
        c0, c1, c2, c3 = _round(c0, c1, c2, c3, k0, k1)
    return c0, c1, c2, c3


@nb.njit("uint64(int64, int64)", cache=True)
def stream_key(seed, stream):
    """64-bit Philox key (as uint64) for one (per-element seed, stream id) pair. seed >= 0, stream >= 0."""
    s = np.uint64(seed)
    st = np.uint64(stream)
    x0, x1, _x2, _x3 = philox4x32_10(st & _M32, st >> _SH32, np.uint64(0), _KEYGEN_TAG, s & _M32, s >> _SH32)
    return (x1 << _SH32) | x0


@nb.njit("UniTuple(uint64, 4)(uint64, int64, int64, int64, int64)", cache=True)
def random_u32x4(key, ring, tick, slot, inner):
    """Four random 32-bit words (held in uint64) for counter (tick, ring, slot, inner) under ``key``."""
    k = np.uint64(key)
    t = np.uint64(tick)
    c3 = (np.uint64(slot) << _SLOT_SHIFT) | (np.uint64(inner) & _INNER_MASK)
    return philox4x32_10(t & _M32, t >> _SH32, np.uint64(ring), c3, k & _M32, k >> _SH32)


@nb.njit("float64(uint64, uint64)", cache=True)
def _u53(a, b):
    """Uniform in the open interval (0, 1) from two 32-bit words: 27 + 26 = 53 random bits, offset by half an LSB.

    x + 0.5 with x = 2**53 - 1 is a float64 tie that rounds to 2**53, i.e. exactly 1.0 (probability 2**-53); that
    one value is clamped to the largest float64 below 1, so the interval is really open. (A 1.0 would give
    us = 0 and a division by zero in PTRS/BTRS.) Every other value is unchanged.
    """
    hi = np.float64(a >> np.uint64(5))
    lo = np.float64(b >> np.uint64(6))
    u = (hi * _TWO_26 + lo + 0.5) * _INV_2_53
    return u if u < 1.0 else _ONE_MINUS_ULP


@nb.njit("UniTuple(float64, 2)(uint64, int64, int64, int64, int64)", cache=True)
def uniform2(key, ring, tick, slot, inner):
    """Two independent uniforms in (0, 1) from one Philox call."""
    x0, x1, x2, x3 = random_u32x4(key, ring, tick, slot, inner)
    return _u53(x0, x1), _u53(x2, x3)


@nb.njit("float64(uint64, int64, int64, int64)", cache=True)
def uniform(key, ring, tick, slot):
    """One uniform in (0, 1) for (ring, tick, slot) under ``key``."""
    u, _v = uniform2(key, ring, tick, slot, 0)
    return u


@nb.njit("UniTuple(float64, 2)(uint64, int64, int64, int64)", cache=True)
def normal2(key, ring, tick, slot):
    """Two independent standard normals (Box-Muller on one Philox call)."""
    u1, u2 = uniform2(key, ring, tick, slot, 0)
    r = math.sqrt(-2.0 * math.log(u1))
    return r * math.cos(_TWO_PI * u2), r * math.sin(_TWO_PI * u2)


@nb.njit("float64(uint64, int64, int64, int64)", cache=True)
def normal(key, ring, tick, slot):
    """One standard normal N(0, 1) for (ring, tick, slot) under ``key``."""
    z, _w = normal2(key, ring, tick, slot)
    return z


@nb.njit("int64(uint64, int64, int64, int64, float64)", cache=True)
def poisson(key, ring, tick, slot, lam):
    """Poisson(lam) as int64. lam <= 0 (or NaN) gives 0; lam above 2**62 (or +inf) saturates at 2**62, because the
    count would not fit an int64 (such a lam is a bug upstream, but a kernel must not crash on it).

    lam < 10: inversion by sequential search on one uniform. lam >= 10: PTRS (Hoermann 1993) with its published
    constants; each rejection iteration uses one Philox call (inner counter). After INNER_MAX rejections (probability
    far below 1e-100) the normal approximation is returned.
    """
    if not (lam > 0.0):
        return np.int64(0)
    if lam >= _TWO_62:
        return np.int64(_I_TWO_62)
    if lam < 10.0:
        u = uniform(key, ring, tick, slot)
        p = math.exp(-lam)
        f = p
        k = 0
        kmax = int(10.0 * lam + 100.0)
        while u > f and k < kmax:
            k += 1
            p *= lam / k
            f += p
        return np.int64(k)
    slam = math.sqrt(lam)
    loglam = math.log(lam)
    b = 0.931 + 2.53 * slam                    # PTRS constants, Hoermann 1993 (as in NumPy random_poisson_ptrs)
    a = -0.059 + 0.02483 * b
    invalpha = 1.1239 + 1.1328 / (b - 3.4)
    vr = 0.9277 - 3.6224 / (b - 2.0)
    for inner in range(INNER_MAX + 1):
        uu, v = uniform2(key, ring, tick, slot, inner)
        u = uu - 0.5
        us = 0.5 - abs(u)
        k = math.floor((2.0 * a / us + b) * u + lam + 0.43)
        if us >= 0.07 and v <= vr:
            return np.int64(k)
        if k < 0.0 or (us < 0.013 and v > us):
            continue
        if (math.log(v) + math.log(invalpha) - math.log(a / (us * us) + b)) <= (-lam + k * loglam - math.lgamma(k + 1.0)):
            return np.int64(k)
    z = normal(key, ring, tick, slot)
    return np.int64(max(0.0, math.floor(lam + slam * z + 0.5)))


@nb.njit("int64(uint64, int64, int64, int64, int64, float64)", cache=True)
def _binomial_core(key, ring, tick, slot, n, p):
    """Binomial(n, p) for 0 < p <= 0.5 and n >= 1."""
    q = 1.0 - p
    if n * p < 10.0:
        # BINV inversion (Kachitvichyanukul and Schmeiser 1988) on one uniform
        u = uniform(key, ring, tick, slot)
        s = p / q
        a = (n + 1) * s
        r = math.exp(n * math.log1p(-p))       # P(X = 0) = q**n; log1p keeps full precision when p << 1
        x = 0
        while u > r and x < n:
            u -= r
            x += 1
            r *= (a / x - s)
        return np.int64(x)
    spq = math.sqrt(n * p * q)
    b = 1.15 + 2.53 * spq                      # BTRS constants, Hoermann 1993
    a = -0.0873 + 0.0248 * b + 0.01 * p
    c = n * p + 0.5
    vr = 0.92 - 4.2 / b
    alpha = (2.83 + 5.1 / b) * spq
    lpq = math.log(p / q)
    m = math.floor((n + 1) * p)
    h = math.lgamma(m + 1.0) + math.lgamma(n - m + 1.0)
    for inner in range(INNER_MAX + 1):
        uu, v = uniform2(key, ring, tick, slot, inner)
        u = uu - 0.5
        us = 0.5 - abs(u)
        k = math.floor((2.0 * a / us + b) * u + c)
        if k < 0.0 or k > n:
            continue
        if us >= 0.07 and v <= vr:
            return np.int64(k)
        lv = math.log(v * alpha / (a / (us * us) + b))
        if lv <= h - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0) + (k - m) * lpq:
            return np.int64(k)
    z = normal(key, ring, tick, slot)
    return np.int64(min(float(n), max(0.0, math.floor(n * p + spq * z + 0.5))))


@nb.njit("int64(uint64, int64, int64, int64, int64, float64)", cache=True)
def binomial(key, ring, tick, slot, n, p):
    """Binomial(n, p) as int64. n <= 0 or p <= 0 gives 0; p >= 1 gives n. Uses the symmetry p -> 1 - p."""
    if n <= 0 or not (p > 0.0):
        return np.int64(0)
    if p >= 1.0:
        return np.int64(n)
    if p > 0.5:
        return np.int64(n) - _binomial_core(key, ring, tick, slot, n, 1.0 - p)
    return _binomial_core(key, ring, tick, slot, n, p)


# ---- batched helpers (prange over b only) ---------------------------------------------------------------------

@nb.njit(cache=True, parallel=True)
def fill_uniform(out, seeds, stream, tick, slot):
    """out[b, i] = uniform(stream_key(seeds[b], stream), i, tick, slot). out: (B, K) float64; seeds: (B,) int64."""
    for b in nb.prange(out.shape[0]):
        key = stream_key(seeds[b], stream)
        for i in range(out.shape[1]):
            out[b, i] = uniform(key, i, tick, slot)


@nb.njit(cache=True, parallel=True)
def fill_normal(out, seeds, stream, tick, slot):
    """out[b, i] = normal(stream_key(seeds[b], stream), i, tick, slot). out: (B, K) float64."""
    for b in nb.prange(out.shape[0]):
        key = stream_key(seeds[b], stream)
        for i in range(out.shape[1]):
            out[b, i] = normal(key, i, tick, slot)


@nb.njit(cache=True, parallel=True)
def fill_poisson(out, seeds, stream, tick, slot, lam):
    """out[b, i] = Poisson(lam[b, i]). out: (B, K) int64; lam: (B, K) float64."""
    for b in nb.prange(out.shape[0]):
        key = stream_key(seeds[b], stream)
        for i in range(out.shape[1]):
            out[b, i] = poisson(key, i, tick, slot, lam[b, i])


@nb.njit(cache=True, parallel=True)
def fill_binomial(out, seeds, stream, tick, slot, n, p):
    """out[b, i] = Binomial(n[b, i], p[b, i]). out: (B, K) int64; n: (B, K) int64; p: (B, K) float64."""
    for b in nb.prange(out.shape[0]):
        key = stream_key(seeds[b], stream)
        for i in range(out.shape[1]):
            out[b, i] = binomial(key, i, tick, slot, n[b, i], p[b, i])
