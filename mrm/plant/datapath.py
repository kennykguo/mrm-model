"""Data-path observables: ones count, bit-statistics sampler, BER and the error counter (07 P0-11).

Scope (docs/ARCHITECTURE.md section 2; 07 = docs/research/07_design_critique.md, P0-11, P1-13):
- the ones density of each plant tick, ``fidelity.pattern_noise: off | binomial | prbs<k>`` ("generates the ones
  density per substep, so the averaged monitor carries the ones-count random walk"), and the channels
  ``tx.ones_count`` / ``tx.bits`` (the transmitter knows its own data);
- ``datapath.bitstat_above``: the number of monitor samples above ``datapath.bitstat_threshold`` in a window of
  ``datapath.bitstat_window_bits`` bits, drawn as Binomial(n1, p_above|1) + Binomial(n0, p_above|0) from the per-level
  monitor levels and the sampler noise (mrm.plant.sensing.monitor_level_codes_row);
- ``rx.error_count``: Poisson(sum_t BER(OMA(t)) * bits(t)) over each ``datapath.error_count_period_s`` (2.0 s, G26),
  delivered after ``datapath.error_count_latency_s``.

Units: W, s, bits (int64 counts). BER model (Gaussian, thermal-noise-limited receiver; Q linear in OMA):
    BER(OMA_rx) = 0.5 erfc(Q / sqrt 2),   Q = Q_ref * OMA_rx / OMA_sens,   Q_ref = Q^-1(BER_ref) = sqrt2 erfcinv(2 BER_ref),
with OMA_sens the receiver sensitivity at BER_ref (G26: 1e-9 at -14.0 (best lane) to -12.8 (worst lane) dBm OMA,
read-off, textbook REF.rxSensG26DbmOma; a re-read gives -14.7 / -12.5, 01 D12) and OMA_rx = OMA_tx * 10^(-link_loss/10).
The sensitivity was measured at 56 Gb/s PRBS9 (01); using it at 50 Gb/s is slightly pessimistic.

Pattern modes. off: n1 = round(rho n) and the optics sees rho exactly; binomial: n1 ~ Binomial(n, rho) per tick;
prbs<k>: the ITU-T O.150 sequence (x^k + x^t + 1: k/t = 7/6, 9/5, 15/14, 23/18, 31/28; Fibonacci register with
feedback from stages t and k, all-ones seed), counted exactly from a packed table for k <= 23 while the scenario
density is 0.5 (normal traffic). prbs31 (period 2^31 - 1 bits = 256 MB packed) and any other density fall back to
Binomial(n, rho): for windows far shorter than the period an m-sequence's ones count is close to binomial (stated
approximation). Densities 0 and 1 (idle, training) are exact in every mode.

Tick convention as in mrm/core/timing.py: ``*_tick_row(k, ...)`` covers plant interval [k, k+1). Bits in interval k:
floor((k+1) R dt) - floor(k R dt) (exactly R dt when that is an integer, e.g. 50 Gb/s at 1 us = 50 000).

Randomness (07 P1-13): counter RNG keyed by (noise seed_b, stream, ring, tick, slot); ``Stream.PATTERN`` for the
per-tick ones counts and the PRBS start phases, ``Stream.DATAPATH`` for the bit-statistics binomials (tick = edge
tick) and the error-count Poisson draws (tick = end of the counting period).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.special import erfcinv

from mrm.core import rng as _rng
from mrm.core.timing import fifo_capacity, fifo_pop_ready, fifo_push, rate_to_period_ticks, seconds_to_ticks

__all__ = [
    "PATTERN_OFF", "PATTERN_BINOMIAL", "PATTERN_PRBS", "PATTERN_NAMES", "PRBS_TAPS", "PRBS_TABLE_MAX_K",
    "BER_REF_G26", "G26_RX_SENS_OMA_DBM", "G26_RX_SENS_OMA_DBM_REREAD", "G26_LINK_LOSS_DB",
    "ERROR_COUNT_PERIOD_S_G26", "BITSTAT_WINDOW_BITS_DEFAULT",
    "prbs_bits", "prbs_table", "popcount32", "prbs_ones_before", "prbs_window_ones",
    "q_from_ber", "ber_from_oma", "p_above", "pattern_tick_row", "pattern_tick", "errors_tick_row", "errors_tick_ber_row", "errors_tick",
    "errors_release_row", "errors_release", "bitstat_window_row", "bitstat_window",
    "DatapathHw", "DatapathBank", "pattern_mode_from_name",
]

PATTERN_OFF = 0
PATTERN_BINOMIAL = 1
PATTERN_PRBS = 2
#: schema enum order of fidelity.pattern_noise (= mrm.params.realize FidelityFlags.pattern_noise index)
PATTERN_NAMES = ("off", "binomial", "prbs7", "prbs9", "prbs15", "prbs23", "prbs31")
#: ITU-T O.150 PRBS polynomials x^k + x^t + 1 (printed in the standard): k -> t
PRBS_TAPS = {7: 6, 9: 5, 15: 14, 23: 18, 31: 28}
PRBS_TABLE_MAX_K = 23        # design choice: 2^23 bits = 1 MB packed; prbs31 would need 256 MB

BER_REF_G26 = 1e-9           # printed, G26 sensitivity curves are read at BER 1e-9 (LM26-P Fig. 2(b))
G26_RX_SENS_OMA_DBM = {"best": -14.0, "worst": -12.8}          # read-off, textbook REF.rxSensG26DbmOma
G26_RX_SENS_OMA_DBM_REREAD = {"best": -14.7, "worst": -12.5}   # read-off, 01 D12 (+-0.3 dB)
G26_LINK_LOSS_DB = 3.0       # printed, LM26-P section 3.1: ~3 dB chip-to-chip (6 connectors, 1 km SMF28)
ERROR_COUNT_PERIOD_S_G26 = 2.0  # printed, G26 BER sampled every 2 s (LM26-P Fig. 3(b); 07 P0-8, P0-11)
BITSTAT_WINDOW_BITS_DEFAULT = 64  # assumed: inside the 16-128 bits of Sun et al. (07 P0-11)

_SLOT_PATTERN = 0
_SLOT_PRBS_PHASE = 1
_SLOT_BS_N1 = 1
_SLOT_BS_A1 = 2
_SLOT_BS_A0 = 3
_SLOT_ERR = 4
_SQRT2 = math.sqrt(2.0)


def pattern_mode_from_name(name: str) -> tuple[int, int]:
    """(mode, k) for a fidelity.pattern_noise name: off -> (PATTERN_OFF, 0), binomial -> (PATTERN_BINOMIAL, 0),
    prbs<k> -> (PATTERN_PRBS, k)."""
    if name == "off":
        return PATTERN_OFF, 0
    if name == "binomial":
        return PATTERN_BINOMIAL, 0
    if name.startswith("prbs"):
        k = int(name[4:])
        if k not in PRBS_TAPS:
            raise ValueError(f"unknown PRBS order {k}; known {sorted(PRBS_TAPS)}")
        return PATTERN_PRBS, k
    raise ValueError(f"unknown pattern_noise {name!r}; known {PATTERN_NAMES}")


# =============================================================================================== PRBS tables
@njit(cache=True)
def prbs_bits(k, t):
    """One period (2^k - 1 bits, uint8) of the PRBS a_m = a_{m-t} XOR a_{m-k} from the all-ones seed."""
    n = (1 << k) - 1
    a = np.empty(n + k, np.uint8)
    for i in range(k):
        a[i] = 1
    for m in range(k, n + k):
        a[m] = a[m - t] ^ a[m - k]
    return a[:n].copy()


@njit(cache=True)
def popcount32(x):
    """Number of set bits of a non-negative int64 below 2^32 (SWAR)."""
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return ((x * 0x01010101) & 0xFFFFFFFF) >> 24


@njit(cache=True)
def _pack32(bits):
    n = bits.shape[0]
    nw = (n + 31) // 32
    words = np.zeros(nw, np.int64)
    pre = np.zeros(nw + 1, np.int64)
    for i in range(n):
        if bits[i]:
            words[i >> 5] |= np.int64(1) << (i & 31)
    for w in range(nw):
        pre[w + 1] = pre[w] + popcount32(words[w])
    return words, pre


_TABLES: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}


def prbs_table(k: int) -> tuple[np.ndarray, np.ndarray, int]:
    """(words, prefix, period) of PRBS-k (k <= PRBS_TABLE_MAX_K): bit i of the sequence is bit (i & 31) of
    words[i >> 5] (32 bits per int64 word), prefix[w] = ones in words[:w]. Cached per process."""
    if k not in PRBS_TAPS:
        raise ValueError(f"unknown PRBS order {k}")
    if k > PRBS_TABLE_MAX_K:
        raise ValueError(f"PRBS{k} is not tabulated (period 2^{k} - 1 bits); it falls back to the binomial model")
    if k not in _TABLES:
        words, pre = _pack32(prbs_bits(k, PRBS_TAPS[k]))
        _TABLES[k] = (words, pre, (1 << k) - 1)
    return _TABLES[k]


@njit(cache=True)
def prbs_ones_before(words, pre, x):
    """Ones among the first x bits (0 <= x <= period) of the tabulated sequence."""
    w = x >> 5
    r = x & 31
    if r == 0:
        return pre[w]
    return pre[w] + popcount32(words[w] & ((np.int64(1) << r) - 1))


@njit(cache=True)
def prbs_window_ones(words, pre, period, start, nbits):
    """Ones in the nbits consecutive bits starting at position ``start`` of the periodic sequence (start >= 0)."""
    if nbits <= 0:
        return np.int64(0)
    total_ones = pre[pre.shape[0] - 1]
    full = nbits // period
    rem = nbits - full * period
    s = start % period
    e = s + rem
    out = full * total_ones
    if e <= period:
        out += prbs_ones_before(words, pre, e) - prbs_ones_before(words, pre, s)
    else:
        out += (total_ones - prbs_ones_before(words, pre, s)) + prbs_ones_before(words, pre, e - period)
    return out


# ================================================================================================ BER, sampler
def q_from_ber(ber: float) -> float:
    """Q such that 0.5 erfc(Q/sqrt2) = ber: Q = sqrt2 erfcinv(2 ber) (Q(1e-9) = 5.998)."""
    if not 0.0 < ber < 0.5:
        raise ValueError("ber must be in (0, 0.5)")
    return float(math.sqrt(2.0) * erfcinv(2.0 * ber))


@njit(cache=True, _nrt=False)
def ber_from_oma(oma_rx_w, oma_sens_w, q_ref, ber_floor):
    """BER of an NRZ lane, Gaussian model with Q linear in OMA: 0.5 erfc(Q/sqrt2), Q = q_ref OMA_rx/OMA_sens; never
    below ber_floor (an error floor, default 0). OMA <= 0 gives 0.5."""
    if not (oma_rx_w > 0.0) or not (oma_sens_w > 0.0):
        return 0.5
    q = q_ref * oma_rx_w / oma_sens_w
    ber = 0.5 * math.erfc(q / _SQRT2)
    return ber if ber > ber_floor else ber_floor


@njit(cache=True, _nrt=False)
def p_above(mu, sig, thr):
    """P(code > thr) for a monitor level with mean mu and rms sig (codes): P(x >= thr + 0.5), x ~ N(mu, sig^2)."""
    edge = thr + 0.5
    if sig > 0.0:
        return 0.5 * math.erfc((edge - mu) / (sig * _SQRT2))
    return 1.0 if mu >= edge else 0.0


@njit(cache=True)
def _bits_in_tick(k, bpt_i, bpt_f):
    if bpt_i >= 0:
        return np.int64(bpt_i)
    return np.int64(math.floor((k + 1) * bpt_f)) - np.int64(math.floor(k * bpt_f))


@njit(cache=True, _nrt=False)
def _ones(mode, n, rho, words, pre, period, start, key, ring, tick, slot):
    """Ones among n bits of density rho under the pattern mode (exact PRBS window when tabulated and rho = 0.5)."""
    if n <= 0:
        return np.int64(0)
    if rho <= 0.0:
        return np.int64(0)
    if rho >= 1.0:
        return np.int64(n)
    if mode == PATTERN_OFF:
        return np.int64(math.floor(rho * n + 0.5))
    if mode == PATTERN_PRBS and period > 0 and abs(rho - 0.5) < 1e-12:
        return prbs_window_ones(words, pre, period, start, n)
    return _rng.binomial(key, ring, tick, slot, n, rho)


# =============================================================================================== tick kernels
@njit(cache=True, _nrt=False, inline="always")
def pattern_tick_row(k, mode, bpt_i, bpt_f, density, traffic_on, words, pre, period, key, bitpos,
                     out_n1, out_nb, out_rho):
    """Ones count and ones density of interval [k, k+1) for the N lanes of one world (07 P0-11).

    density (N,): the scenario's ones density (0.5 normal traffic; 0, 1 idle/training; Sun et al. steps);
    traffic_on (N,) bool: data is being sent and counted (False: no bits counted, the optics sees ``density``).
    mode: PATTERN_*; bpt_i: bits per tick if integral (else -1) and bpt_f the float value; words/pre/period: PRBS
    table (period 0 = none); bitpos (N,) int64 position in the PRBS (state). Outputs (N,): out_n1, out_nb (int64),
    out_rho = n1/n (or ``density`` in mode off or without bits), the density the optics should average with.
    """
    for n in range(density.shape[0]):
        rho = density[n]
        if rho < 0.0:
            rho = 0.0
        elif rho > 1.0:
            rho = 1.0
        if not traffic_on[n]:
            out_n1[n] = 0
            out_nb[n] = 0
            out_rho[n] = rho
            continue
        nb = _bits_in_tick(k, bpt_i, bpt_f)
        n1 = _ones(mode, nb, rho, words, pre, period, bitpos[n], key, n, k, _SLOT_PATTERN)
        out_n1[n] = n1
        out_nb[n] = nb
        out_rho[n] = rho if (mode == PATTERN_OFF or nb == 0) else n1 / nb
        if period > 0:
            bitpos[n] = (bitpos[n] + nb) % period
        else:
            bitpos[n] += nb


@njit(cache=True)
def pattern_tick(k, mode, bpt_i, bpt_f, density, traffic_on, words, pre, period, key, bitpos, out_n1, out_nb,
                 out_rho):
    """Batched :func:`pattern_tick_row` ((B, N) arrays, key (B,) uint64)."""
    for b in range(density.shape[0]):
        pattern_tick_row(k, mode, bpt_i, bpt_f, density[b], traffic_on[b], words, pre, period, key[b], bitpos[b],
                         out_n1[b], out_nb[b], out_rho[b])


@njit(cache=True)
def errors_tick_row(k, oma_tx_w, nbits, link_t, oma_sens_w, q_ref, ber_floor, period, lat, key, lam, fv, fr, fm,
                    out_new):
    """Accumulate the expected errors of interval [k, k+1) and close a counting period at instant k + 1.

    lam (N,) += BER(OMA_tx * link_t) * nbits (07 P0-11). When (k + 1) % period == 0 the count
    Poisson(lam) (keyed by tick k + 1) is queued for release at k + 1 + lat and lam restarts at 0. oma_tx_w (N,) W at
    the transmitter bus output; oma_sens_w (N,) per-lane sensitivity (W OMA at BER_ref); link_t linear transmission
    TX bus -> RX PD. period 0 disables the counter. out_new (N,) True when a count was queued.
    """
    for n in range(oma_tx_w.shape[0]):
        out_new[n] = False
        if nbits[n] > 0:
            lam[n] += ber_from_oma(oma_tx_w[n] * link_t, oma_sens_w[n], q_ref, ber_floor) * nbits[n]
        _errors_close(k, n, period, lat, key, lam, fv, fr, fm, out_new)


@njit(cache=True, _nrt=False)
def _errors_close(k, n, period, lat, key, lam, fv, fr, fm, out_new):
    if period > 0 and (k + 1) % period == 0:
        cnt = _rng.poisson(key, n, k + 1, _SLOT_ERR, lam[n])
        lam[n] = 0.0
        fifo_push(fv, fr, fm, n, cnt, k + 1 + lat)
        out_new[n] = True


@njit(cache=True, _nrt=False, inline="always")
def errors_tick_ber_row(k, ber, nbits, period, lat, key, lam, fv, fr, fm, out_new):
    """:func:`errors_tick_row` with the BER of each lane given (ber (N,) = ber_from_oma(oma_tx_w * link_t,
    oma_sens_w, q_ref, ber_floor), computed once by the caller for the error counter and the Truth BER)."""
    # whether instant k + 1 closes a counting period is the same for every lane: tested once per tick (the modulo and
    # the call of _errors_close per lane were ~0.2 us per tick; same values, docs/handoff/perf.md R2-6)
    close = period > 0 and (k + 1) % period == 0
    for n in range(ber.shape[0]):
        out_new[n] = False
        if nbits[n] > 0:
            lam[n] += ber[n] * nbits[n]
        if close:
            _errors_close(k, n, period, lat, key, lam, fv, fr, fm, out_new)


@njit(cache=True)
def errors_tick(k, oma_tx_w, nbits, link_t, oma_sens_w, q_ref, ber_floor, period, lat, key, lam, fv, fr, fm,
                out_new):
    """Batched :func:`errors_tick_row` (link_t, lat, key (B,))."""
    for b in range(oma_tx_w.shape[0]):
        errors_tick_row(k, oma_tx_w[b], nbits[b], link_t[b], oma_sens_w[b], q_ref, ber_floor, period, lat[b],
                        key[b], lam[b], fv[b], fr[b], fm[b], out_new[b])


@njit(cache=True, _nrt=False)
def errors_release_row(now, fv, fr, fm, latest, out_valid):
    """Release error counts ready at ``now``; latest (N,) is held, out_valid (N,) marks a delivery (the channel
    rx.error_count_valid)."""
    for n in range(fm.shape[0]):
        out_valid[n] = False
        while True:
            ok, val, rel = fifo_pop_ready(fv, fr, fm, n, now)
            if not ok:
                break
            latest[n] = val
            out_valid[n] = True


@njit(cache=True)
def errors_release(now, fv, fr, fm, latest, out_valid):
    """Batched :func:`errors_release_row`."""
    for b in range(fm.shape[0]):
        errors_release_row(now, fv[b], fr[b], fm[b], latest[b], out_valid[b])


@njit(cache=True, _nrt=False)
def bitstat_window_row(edge, n_win, mode, density, mu1, mu0, sig1, sig0, thr, words, pre, period, bitpos, key,
                       out_above, out_n1, out_a1, out_a0):
    """Bit-statistics sampler of one world at controller edge ``edge`` (07 P0-11, Sun et al.).

    The window holds the last n_win bits: n1 ones (pattern mode as in :func:`pattern_tick_row`; the PRBS window ends
    at the lane's current bit position), n0 = n_win - n1 zeros. Each '1' bit reads above the threshold with
    p1 = P(N(mu1, sig1^2) >= thr + 0.5), each '0' bit with p0 likewise, so
        above = Binomial(n1, p1) + Binomial(n0, p0),   E[above | n1] = n1 p1 + n0 p0.
    mu/sig (N,) in monitor codes from mrm.plant.sensing.monitor_level_codes_row; thr (N,) the threshold code.
    Outputs (N,): out_above (datapath.bitstat_above), out_n1 (the window's ones, known to the TX), out_a1, out_a0.
    """
    for n in range(density.shape[0]):
        rho = density[n]
        if rho < 0.0:
            rho = 0.0
        elif rho > 1.0:
            rho = 1.0
        start = bitpos[n] - n_win
        if period > 0:
            start = start % period
            if start < 0:
                start += period
        n1 = _ones(mode, n_win, rho, words, pre, period, start, key, n, edge, _SLOT_BS_N1)
        n0 = n_win - n1
        a1 = _rng.binomial(key, n, edge, _SLOT_BS_A1, n1, p_above(mu1[n], sig1[n], thr[n]))
        a0 = _rng.binomial(key, n, edge, _SLOT_BS_A0, n0, p_above(mu0[n], sig0[n], thr[n]))
        out_above[n] = a1 + a0
        out_n1[n] = n1
        out_a1[n] = a1
        out_a0[n] = a0


@njit(cache=True)
def bitstat_window(edge, n_win, mode, density, mu1, mu0, sig1, sig0, thr, words, pre, period, bitpos, key,
                   out_above, out_n1, out_a1, out_a0):
    """Batched :func:`bitstat_window_row` ((B, N) arrays, key (B,))."""
    for b in range(density.shape[0]):
        bitstat_window_row(edge, n_win, mode, density[b], mu1[b], mu0[b], sig1[b], sig0[b], thr[b], words, pre,
                           period, bitpos[b], key[b], out_above[b], out_n1[b], out_a1[b], out_a0[b])


# ================================================================================================ Python API
def _arr_b(x, B, name, dtype=np.float64):
    a = np.asarray(x, dtype=dtype).reshape(-1)
    if a.size == 1:
        return np.full(B, a[0], dtype=dtype)
    if a.size != B:
        raise ValueError(f"{name}: shape {np.shape(x)} is not scalar or (B={B},)")
    return a.copy()


def _arr_bn(x, B, N, name, dtype=np.float64):
    a = np.asarray(x, dtype=dtype)
    if a.ndim == 1:
        a = a.reshape(1, -1) if a.shape[0] == N else a.reshape(-1, 1)
    try:
        return np.ascontiguousarray(np.broadcast_to(a, (B, N))).copy()
    except ValueError:
        raise ValueError(f"{name}: shape {np.shape(x)} does not broadcast to (B={B}, N={N})") from None


def _keys(seeds, B, stream) -> np.ndarray:
    seeds = _arr_b(seeds, B, "seeds", np.int64)
    sid = _rng.stream_id(stream) if isinstance(stream, str) else int(stream)
    return np.array([_rng.stream_key(int(s), sid) for s in seeds], dtype=np.uint64)


@dataclass
class DatapathHw:
    """Realized data-path settings for B worlds and N lanes (07 P0-11). Build with ``build`` or
    ``from_plant_params``."""

    B: int
    N: int
    dt_s: float
    bitrate_hz: float
    bpt_i: int                  # bits per tick when integral, else -1
    bpt_f: float
    mode: int                   # PATTERN_*
    prbs_k: int                 # 0 unless mode == PATTERN_PRBS
    window_bits: int
    error_period_ticks: int     # 0: error counter off
    error_latency_ticks: np.ndarray  # (B,) int64
    link_t: np.ndarray          # (B,) linear transmission TX bus output -> RX PD
    oma_sens_w: np.ndarray      # (B, N) sensitivity OMA at BER_ref, W
    q_ref: float
    ber_floor: float
    bitstat_tau_s: float        # sampler front-end time constant, s (1/(2 pi f_3dB))
    bitstat_noise_a_per_rt_hz: float | None = None  # sampler input noise density (None: the monitor's own)

    @classmethod
    def build(cls, B: int, N: int, dt_s: float, *, bitrate_hz: float, pattern_noise: str = "off",
              bitstat_window_bits: int = BITSTAT_WINDOW_BITS_DEFAULT,
              error_count_period_s: float | None = ERROR_COUNT_PERIOD_S_G26, error_count_latency_s=0.0,
              rx_sensitivity_oma_dbm=G26_RX_SENS_OMA_DBM["worst"], ber_ref: float = BER_REF_G26,
              link_loss_db=G26_LINK_LOSS_DB, ber_floor: float = 0.0, bitstat_bw_hz: float = 16e9,
              bitstat_noise_a_per_rt_hz: float | None = None) -> "DatapathHw":
        """bitrate_hz: line rate (G26 50 Gb/s, printed). pattern_noise: a fidelity.pattern_noise name.
        rx_sensitivity_oma_dbm: per lane (N,) or per world; default the worst G26 lane (-12.8 dBm, read-off).
        link_loss_db: TX bus output -> RX PD (default 3 dB printed chip-to-chip; on-chip TX/RX path losses are not
        included, assumed 0). bitstat_bw_hz: 3 dB bandwidth of the bit-rate monitor sampler (no Lightmatter value;
        default 16 GHz, the G25 RX AFE bandwidth (printed, simulated, 01), assumed to carry over).
        bitstat_noise_a_per_rt_hz: one-sided input noise density of that sampler; None (default) reuses the slow
        monitor's ring.monitor.input_noise (assumed; a bit-rate TIA is usually noisier per root hertz, no data).
        error_count_period_s None or 0 disables the error counter. error_count_latency_s defaults to 0 here, while
        mrm.core.channels.ChannelHardware assumes 1 ms for the ChannelSpec: presets should set the key."""
        dt = float(dt_s)
        r = float(bitrate_hz)
        if not (math.isfinite(r) and r > 0.0):
            raise ValueError("bitrate_hz must be finite and > 0")
        bpt = r * dt
        bpt_i = int(round(bpt)) if abs(bpt - round(bpt)) <= 1e-9 * max(bpt, 1.0) else -1
        mode, k = pattern_mode_from_name(pattern_noise)
        if error_count_period_s:
            per = seconds_to_ticks(float(error_count_period_s), dt, "datapath.error_count_period_s",
                                   allow_zero=False)
        else:
            per = 0
        lat = _arr_b(error_count_latency_s, B, "error_count_latency_s")
        lat_t = np.array([seconds_to_ticks(float(x), dt, "datapath.error_count_latency_s") for x in lat], np.int64)
        sens = _arr_bn(rx_sensitivity_oma_dbm, B, N, "rx_sensitivity_oma_dbm")
        link = 10.0 ** (-_arr_b(link_loss_db, B, "link_loss_db") / 10.0)
        if int(bitstat_window_bits) < 1:
            raise ValueError("bitstat_window_bits must be >= 1")
        if not 0.0 <= ber_floor < 0.5:
            raise ValueError("ber_floor must be in [0, 0.5)")
        if not (math.isfinite(float(bitstat_bw_hz)) and float(bitstat_bw_hz) > 0.0):
            raise ValueError("bitstat_bw_hz must be finite and > 0")
        if bitstat_noise_a_per_rt_hz is not None and not float(bitstat_noise_a_per_rt_hz) >= 0.0:
            raise ValueError("bitstat_noise_a_per_rt_hz must be >= 0")
        return cls(B=int(B), N=int(N), dt_s=dt, bitrate_hz=r, bpt_i=bpt_i, bpt_f=bpt, mode=mode, prbs_k=k,
                   window_bits=int(bitstat_window_bits), error_period_ticks=int(per), error_latency_ticks=lat_t,
                   link_t=link, oma_sens_w=1e-3 * 10.0 ** (sens / 10.0), q_ref=q_from_ber(ber_ref),
                   ber_floor=float(ber_floor), bitstat_tau_s=1.0 / (2.0 * math.pi * float(bitstat_bw_hz)),
                   bitstat_noise_a_per_rt_hz=(None if bitstat_noise_a_per_rt_hz is None
                                              else float(bitstat_noise_a_per_rt_hz)))

    @classmethod
    def from_plant_params(cls, pp, **kw) -> "DatapathHw":
        """From PlantParams: datapath.bitrate_hz (B,1; the first world's value is used: the line rate is a setting),
        bitstat_window_bits (0 -> default), error_count_period_ticks (0 -> G26 2.0 s), timing.error_count_latency_ticks,
        flags.pattern_noise, and the integrator's DatapathParams fields rx_sensitivity_oma_w (B,N), ber_ref,
        link_transmission (B,1), ber_floor, bitstat_bw_hz, bitstat_noise_a_per_rt_hz (datapath.rx_sensitivity_oma_dbm,
        ber_ref, link_loss_db, ber_floor, bitstat_bw_ghz, bitstat_noise_pa_per_rt_hz). A field left nan (key absent)
        keeps the module default of :meth:`build`; ber_ref, ber_floor, the sampler bandwidth and noise are settings of
        one instrument, so the first world's value is used. Keyword arguments override everything."""
        d, t = pp.datapath, pp.timing
        dt = float(t.dt_s)
        br = np.asarray(d.bitrate_hz, dtype=float).reshape(-1)
        if not np.all(np.isfinite(br)):
            raise ValueError("datapath.bitrate_gbps is needed for the data path")
        args = dict(bitrate_hz=float(br[0]), pattern_noise=PATTERN_NAMES[int(pp.flags.pattern_noise)],
                    bitstat_window_bits=int(d.bitstat_window_bits) or BITSTAT_WINDOW_BITS_DEFAULT,
                    error_count_period_s=(int(d.error_count_period_ticks) * dt if d.error_count_period_ticks
                                          else ERROR_COUNT_PERIOD_S_G26),
                    error_count_latency_s=np.asarray(t.error_count_latency_ticks).reshape(-1) * dt)

        def finite(name):
            v = getattr(d, name, None)
            if v is None:
                return None
            v = np.asarray(v, dtype=float)
            return v if np.all(np.isfinite(v)) else None
        sens = finite("rx_sensitivity_oma_w")
        if sens is not None:
            args["rx_sensitivity_oma_dbm"] = 10.0 * np.log10(sens / 1e-3)      # W -> dBm
        link = finite("link_transmission")
        if link is not None:
            args["link_loss_db"] = -10.0 * np.log10(link.reshape(-1))          # transmission -> dB loss
        for name, arg in (("ber_ref", "ber_ref"), ("ber_floor", "ber_floor"), ("bitstat_bw_hz", "bitstat_bw_hz"),
                          ("bitstat_noise_a_per_rt_hz", "bitstat_noise_a_per_rt_hz")):
            v = finite(name)
            if v is not None:
                args[arg] = float(v.reshape(-1)[0])
        args.update(kw)
        return cls.build(int(pp.batch), int(pp.n_rings), dt, **args)

    def ber(self, oma_tx_w) -> np.ndarray:
        """BER (B, N) at a transmitter OMA (W, broadcast to (B, N)) (vectorized :func:`ber_from_oma`)."""
        from scipy.special import erfc
        o = _arr_bn(oma_tx_w, self.B, self.N, "oma_tx_w") * self.link_t[:, None]
        q = self.q_ref * o / self.oma_sens_w
        b = np.where(o > 0, 0.5 * erfc(q / math.sqrt(2.0)), 0.5)
        return np.maximum(b, self.ber_floor)


class DatapathBank:
    """State of the data path for B worlds and N lanes (Python slow path).

    Per plant interval: ``rho = bank.pattern(k, density, traffic_on)`` before the optics (rho is the ones density the
    optics averages the two levels with), then ``bank.errors(k, oma_tx_w)`` after it. At each controller edge:
    ``bank.bitstat(edge, mu1, mu0, sig1, sig0, thr, density)`` (levels from MonitorBank.level_codes), then
    ``sensors, new = bank.release(edge)``.
    """

    def __init__(self, hw: DatapathHw, noise_seeds, pop_period_ticks: int = 1):
        self.hw = hw
        B, N = hw.B, hw.N
        if hw.mode == PATTERN_PRBS and hw.prbs_k <= PRBS_TABLE_MAX_K:
            self.words, self.pre, self.period = prbs_table(hw.prbs_k)
        else:
            self.words, self.pre, self.period = np.zeros(1, np.int64), np.zeros(2, np.int64), 0
        per = hw.error_period_ticks
        cap = fifo_capacity(int(hw.error_latency_ticks.max()), max(per, 1), max(int(pop_period_ticks), 1))
        self.fv = np.zeros((B, N, cap), np.int64)
        self.fr = np.zeros((B, N, cap), np.int64)
        self.fm = np.zeros((B, N, 2), np.int64)
        self.lam = np.zeros((B, N))
        self.bitpos = np.zeros((B, N), np.int64)
        self.ones_acc = np.zeros((B, N), np.int64)
        self.bits_acc = np.zeros((B, N), np.int64)
        self.err_latest = np.zeros((B, N), np.int64)
        self.err_valid = np.zeros((B, N), np.bool_)
        self.err_new_tick = np.zeros((B, N), np.bool_)
        self.n1 = np.zeros((B, N), np.int64)
        self.nb = np.zeros((B, N), np.int64)
        self.rho = np.zeros((B, N))
        self.bs_above = np.zeros((B, N), np.int64)
        self.bs_n1 = np.zeros((B, N), np.int64)
        self.bs_a1 = np.zeros((B, N), np.int64)
        self.bs_a0 = np.zeros((B, N), np.int64)
        self.bs_new = False
        self._on = np.ones((B, N), np.bool_)
        self.reset(noise_seeds)

    def reset(self, noise_seeds=None) -> None:
        """Clear the state; PRBS start phases are drawn per lane from the pattern stream (tick 0, its own slot)."""
        hw = self.hw
        if noise_seeds is not None:
            self.key_pattern = _keys(noise_seeds, hw.B, "pattern")
            self.key_dp = _keys(noise_seeds, hw.B, "datapath")
        self.lam[:] = 0.0
        self.fm[:] = 0
        self.ones_acc[:] = 0
        self.bits_acc[:] = 0
        self.err_latest[:] = 0
        self.err_valid[:] = False
        self.err_new_tick[:] = False
        self.n1[:] = 0
        self.nb[:] = 0
        self.rho[:] = 0.0
        self.bs_above[:] = 0
        self.bs_n1[:] = 0
        self.bs_a1[:] = 0
        self.bs_a0[:] = 0
        self.bs_new = False
        self.bitpos[:] = 0
        if self.period > 0:
            for b in range(hw.B):
                for n in range(hw.N):
                    u = _rng.uniform(self.key_pattern[b], n, 0, _SLOT_PRBS_PHASE)
                    self.bitpos[b, n] = int(u * self.period)

    def pattern(self, k: int, density=0.5, traffic_on=None) -> np.ndarray:
        """Interval [k, k+1): returns the ones density (B, N) the optics should use; accumulates tx.ones_count and
        tx.bits. density: scenario ``data.ones_density``; traffic_on: bool, or a float on at >= 0.5 (the scenario
        ``data.driver_enable`` convention)."""
        hw = self.hw
        d = _arr_bn(density, hw.B, hw.N, "density")
        if traffic_on is None:
            on = self._on
        else:
            t = np.asarray(traffic_on)
            on = (_arr_bn(t, hw.B, hw.N, "traffic_on", np.bool_) if t.dtype == np.bool_
                  else _arr_bn(t, hw.B, hw.N, "traffic_on") >= 0.5)
        pattern_tick(np.int64(k), hw.mode, hw.bpt_i, hw.bpt_f, d, on, self.words, self.pre, self.period,
                     self.key_pattern, self.bitpos, self.n1, self.nb, self.rho)
        self.ones_acc += self.n1
        self.bits_acc += self.nb
        return self.rho

    def errors(self, k: int, oma_tx_w) -> None:
        """Interval [k, k+1) after the optics: accumulate BER(OMA) * bits (uses the bits of the last pattern call)."""
        hw = self.hw
        o = _arr_bn(oma_tx_w, hw.B, hw.N, "oma_tx_w")
        errors_tick(np.int64(k), o, self.nb, hw.link_t, hw.oma_sens_w, hw.q_ref, hw.ber_floor,
                    hw.error_period_ticks, hw.error_latency_ticks, self.key_dp, self.lam, self.fv, self.fr, self.fm,
                    self.err_new_tick)

    def bitstat(self, edge: int, mu1, mu0, sig1, sig0, thr, density=0.5) -> np.ndarray:
        """Draw datapath.bitstat_above (B, N) at controller edge ``edge`` (07 P0-11)."""
        hw = self.hw
        B, N = hw.B, hw.N
        a = [_arr_bn(x, B, N, nm) for x, nm in ((mu1, "mu1"), (mu0, "mu0"), (sig1, "sig1"), (sig0, "sig0"),
                                                (thr, "thr"), (density, "density"))]
        bitstat_window(np.int64(edge), hw.window_bits, hw.mode, a[5], a[0], a[1], a[2], a[3], a[4], self.words,
                       self.pre, self.period, self.bitpos, self.key_dp, self.bs_above, self.bs_n1, self.bs_a1,
                       self.bs_a0)
        self.bs_new = True
        return self.bs_above

    def bitstat_from_monitor(self, edge: int, monitor, p1_w, p0_w, thr, density=0.5, p_bg_w=0.0, t_pd_k=None,
                             t_adc_k=None, pga_select=None, rin_per_hz=0.0) -> np.ndarray:
        """:meth:`bitstat` with the level codes computed by a mrm.plant.sensing.MonitorBank from the per-level PD
        powers p1_w / p0_w (W; optics R_DROP_SUM1 / R_DROP_SUM0) through this sampler's bandwidth. rin_per_hz:
        the laser RIN (pp.laser.rin_per_hz, (B,) or scalar), which the sampler sees over its whole bandwidth."""
        mu1, mu0, s1, s0 = monitor.level_codes(p1_w, p0_w, p_bg_w, t_pd_k, t_adc_k, pga_select,
                                               tau_s=self.hw.bitstat_tau_s,
                                               noise_a_per_rt_hz=self.hw.bitstat_noise_a_per_rt_hz,
                                               rin_per_hz=rin_per_hz)
        return self.bitstat(edge, mu1, mu0, s1, s0, thr, density)

    def release(self, now: int) -> tuple[dict, dict]:
        """Channels at controller edge ``now``: tx.ones_count and tx.bits (sums since the last release), the latest
        datapath.bitstat_above, rx.error_count (held) and rx.error_count_valid (07 P0-11)."""
        errors_release(np.int64(now), self.fv, self.fr, self.fm, self.err_latest, self.err_valid)
        B, N = self.hw.B, self.hw.N
        always = np.ones((B, N), np.bool_)
        sensors = {"tx.ones_count": self.ones_acc.copy(), "tx.bits": self.bits_acc.copy(),
                   "datapath.bitstat_above": self.bs_above.astype(np.int32),
                   "rx.error_count": self.err_latest.copy(), "rx.error_count_valid": self.err_valid.copy()}
        new = {"tx.ones_count": always, "tx.bits": always,
               "datapath.bitstat_above": np.full((B, N), self.bs_new, np.bool_),
               "rx.error_count": self.err_valid.copy(), "rx.error_count_valid": always}
        self.ones_acc[:] = 0
        self.bits_acc[:] = 0
        self.bs_new = False
        return sensors, new
