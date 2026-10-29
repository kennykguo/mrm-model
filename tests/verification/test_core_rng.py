"""Checks the counter-based random numbers of mrm.core.rng.

Read this first
---------------
Every random draw is a pure function of (seed, stream id, ring, tick, slot), computed with the Philox4x32-10
counter-based generator. There is no hidden state. So element b of a batch with base seed s equals the single run
with seed s + b, the thread count does not matter, and switching one noise source on or off leaves the other streams
unchanged.

The tests check the Philox known answers, batch and thread-count invariance, the stream ids, purity, the
distributions (uniform, normal, Poisson, binomial), edge cases, and identical bits with NUMBA_DISABLE_JIT=1.
"""
import json
import os
import subprocess
import sys

import numba as nb
import numpy as np
import pytest
from scipy import stats

from mrm.core import rng

JIT_OFF = bool(nb.config.DISABLE_JIT)
NS = 20_000 if JIT_OFF else 200_000        # sample count for distribution checks (pure Python is ~100x slower)

# Random123 kat_vectors, philox4x32 with 10 rounds: counter (4 words), key (2 words) -> output (4 words)
KAT = [
    ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
    ((0xFFFFFFFF,) * 4, (0xFFFFFFFF, 0xFFFFFFFF), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
    ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0),
     (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
]


@pytest.mark.parametrize("ctr, key, expected", KAT)
def test_philox_known_answers(ctr, key, expected):
    out = rng.philox4x32_10(*[np.uint64(x) for x in ctr], *[np.uint64(x) for x in key])
    assert tuple(int(x) for x in out) == expected


def test_stream_ids_fixed_and_parametric():
    assert rng.stream_id("variation") == rng.Stream.VARIATION == 0x01000000
    assert rng.stream_id("shot_noise") == rng.Stream.SHOT_NOISE
    assert rng.stream_id("controller") == rng.Stream.CONTROLLER
    a = rng.stream_id("adc_tx.monitor")
    assert a >> 24 == rng.Stream.ADC >> 24 and a == rng.stream_id("adc_tx.monitor")     # stable
    assert a != rng.stream_id("adc_temp.die")
    assert rng.stream_id("scenario_aggressor") >> 24 == rng.Stream.SCENARIO >> 24
    assert rng.stream_id("dac_tx.bias") >> 24 == rng.Stream.DAC >> 24
    # crc32 is process independent: this exact value must never change (it keys golden traces)
    import zlib
    assert a == 0x07000000 | (zlib.crc32(b"tx.monitor") & 0xFFFFFF)
    for bad in ("nonsense", "adc_", "", None):
        with pytest.raises(ValueError):
            rng.stream_id(bad)
    ids = [int(m) for m in rng.Stream]
    assert len(set(ids)) == len(ids)


def test_batch_seeds():
    assert list(rng.batch_seeds(7, 4)) == [7, 8, 9, 10]
    with pytest.raises(ValueError):
        rng.batch_seeds(-1, 2)


def test_batch_seeds_reaches_the_largest_int64_seed():
    """The last seed base + B - 1 may be 2**63 - 1 (it fits an int64); one more is refused."""
    B = 4
    seeds = rng.batch_seeds(2**63 - B, B)
    assert seeds.dtype == np.int64 and int(seeds[-1]) == 2**63 - 1
    assert int(rng.batch_seeds(2**63 - 1, 1)[0]) == 2**63 - 1
    with pytest.raises(ValueError):
        rng.batch_seeds(2**63 - B + 1, B)
    with pytest.raises(ValueError):
        rng.batch_seeds(2**63, 0)


def test_element_b_equals_b1_run_with_seed_b():
    """Element b of a B = 64 run (base seed 1000) equals the B = 1 run with seed 1000 + b."""
    base, B, K = 1000, 64, 8
    sid = rng.stream_id("adc_tx.monitor")
    seeds = rng.batch_seeds(base, B)
    for fill, extra in ((rng.fill_normal, ()), (rng.fill_uniform, ())):
        big = np.empty((B, K))
        fill(big, seeds, sid, 123456789, 3, *extra)
        for b in range(B):
            one = np.empty((1, K))
            fill(one, rng.batch_seeds(base + b, 1), sid, 123456789, 3, *extra)
            assert np.array_equal(one[0], big[b])
    lam = np.full((B, K), 25.0)
    big = np.empty((B, K), dtype=np.int64)
    rng.fill_poisson(big, seeds, sid, 5, 0, lam)
    for b in (0, 17, 63):
        one = np.empty((1, K), dtype=np.int64)
        rng.fill_poisson(one, rng.batch_seeds(base + b, 1), sid, 5, 0, lam[:1])
        assert np.array_equal(one[0], big[b])


def test_thread_count_invariance():
    B, K = 64, 16
    seeds = rng.batch_seeds(0, B)
    sid = rng.stream_id("shot_noise")
    outs = []
    n_max = nb.config.NUMBA_NUM_THREADS
    for n_threads in sorted({1, min(2, n_max), n_max}):
        try:
            nb.set_num_threads(n_threads)
        except Exception:             # pragma: no cover - threading layer unavailable
            pass
        a = np.empty((B, K))
        rng.fill_normal(a, seeds, sid, 42, 0)
        n = np.empty((B, K), dtype=np.int64)
        rng.fill_binomial(n, seeds, sid, 42, 1, np.full((B, K), 128, dtype=np.int64), np.full((B, K), 0.3))
        outs.append((a, n))
    nb.set_num_threads(n_max)
    for a, n in outs[1:]:
        assert np.array_equal(a, outs[0][0]) and np.array_equal(n, outs[0][1])


def test_draws_are_pure_functions_of_the_counter():
    """No hidden state: drawing from another stream in between changes nothing. So toggling shot noise leaves the
    ADC-noise draws unchanged, and two controllers on one seed see the same noise."""
    k_adc = rng.stream_key(5, rng.stream_id("adc_tx.monitor"))
    k_shot = rng.stream_key(5, rng.stream_id("shot_noise"))
    first = [rng.normal(k_adc, r, t, 0) for r in range(4) for t in range(50)]
    _ = [rng.normal(k_shot, r, t, 0) for r in range(4) for t in range(50)]
    again = [rng.normal(k_adc, r, t, 0) for r in range(4) for t in range(50)]
    assert first == again
    assert k_adc != k_shot
    assert rng.stream_key(5, 1) != rng.stream_key(6, 1)
    # different ring / tick / slot give different numbers
    vals = {rng.uniform(k_adc, r, t, s) for r in range(3) for t in range(3) for s in range(3)}
    assert len(vals) == 27


def test_large_ticks_and_rings():
    k = rng.stream_key(1, 2)
    big_tick = 109_440_000_000 * 10          # beyond 2**32: uses both counter words
    u1 = rng.uniform(k, 0, big_tick, 0)
    u2 = rng.uniform(k, 0, big_tick & 0xFFFFFFFF, 0)
    assert u1 != u2 and 0.0 < u1 < 1.0
    assert 0.0 < rng.uniform(k, 2**32 - 1, 0, rng.SLOT_MAX) < 1.0


def _uniforms(n, seed=3, stream=7):
    out = np.empty((1, n))
    rng.fill_uniform(out, rng.batch_seeds(seed, 1), stream, 99, 0)
    return out[0]


def test_uniform_distribution():
    u = _uniforms(NS)
    assert 0.0 < u.min() and u.max() < 1.0
    assert stats.kstest(u, "uniform").pvalue > 1e-4
    # serial correlation between neighbouring counters
    assert abs(np.corrcoef(u[:-1], u[1:])[0, 1]) < 5.0 / np.sqrt(NS)


def test_normal_distribution():
    z = np.empty((1, NS))
    rng.fill_normal(z, rng.batch_seeds(11, 1), 9, 0, 0)
    z = z[0]
    assert abs(z.mean()) < 5.0 / np.sqrt(NS)
    assert abs(z.var() - 1.0) < 5.0 * np.sqrt(2.0 / NS)
    assert stats.kstest(z, "norm").pvalue > 1e-4


def _chi2_pvalue(samples, pmf, support):
    counts = np.array([np.sum(samples == k) for k in support], dtype=float)
    expected = pmf(np.array(support)) * samples.size
    keep = expected >= 5.0
    o = np.append(counts[keep], samples.size - counts[keep].sum())
    e = np.append(expected[keep], samples.size - expected[keep].sum())
    if e[-1] < 5.0:
        o, e = o[:-1], e[:-1]
        e *= o.sum() / e.sum()
    return stats.chisquare(o, e).pvalue


@pytest.mark.parametrize("lam", [0.3, 2.5, 9.9, 10.0, 37.5, 1000.0])
def test_poisson_distribution(lam):
    n = NS // 4
    x = np.empty((1, n), dtype=np.int64)
    rng.fill_poisson(x, rng.batch_seeds(21, 1), 17, 7, 0, np.full((1, n), lam))
    x = x[0]
    assert abs(x.mean() - lam) < 5.0 * np.sqrt(lam / n)
    lo, hi = int(max(0, lam - 8 * np.sqrt(lam) - 2)), int(lam + 8 * np.sqrt(lam) + 3)
    assert _chi2_pvalue(x, lambda k: stats.poisson.pmf(k, lam), range(lo, hi + 1)) > 1e-4


@pytest.mark.parametrize("n_trials, p", [(20, 0.1), (100, 0.3), (1000, 0.02), (64, 0.5), (200, 0.93), (5000, 0.4)])
def test_binomial_distribution(n_trials, p):
    n = NS // 4
    x = np.empty((1, n), dtype=np.int64)
    rng.fill_binomial(x, rng.batch_seeds(31, 1), 19, 7, 0, np.full((1, n), n_trials, dtype=np.int64),
                      np.full((1, n), p))
    x = x[0]
    mu, var = n_trials * p, n_trials * p * (1 - p)
    assert x.min() >= 0 and x.max() <= n_trials
    assert abs(x.mean() - mu) < 5.0 * np.sqrt(var / n)
    lo, hi = int(max(0, mu - 8 * np.sqrt(var) - 2)), int(min(n_trials, mu + 8 * np.sqrt(var) + 3))
    assert _chi2_pvalue(x, lambda k: stats.binom.pmf(k, n_trials, p), range(lo, hi + 1)) > 1e-4


def test_edge_cases():
    k = rng.stream_key(0, 0)
    assert rng.poisson(k, 0, 0, 0, 0.0) == 0
    assert rng.poisson(k, 0, 0, 0, -3.0) == 0
    assert rng.poisson(k, 0, 0, 0, float("nan")) == 0
    assert rng.binomial(k, 0, 0, 0, 0, 0.5) == 0
    assert rng.binomial(k, 0, 0, 0, 10, 0.0) == 0
    assert rng.binomial(k, 0, 0, 0, 10, 1.0) == 10
    assert rng.binomial(k, 0, 0, 0, -4, 0.5) == 0


_PARITY_SCRIPT = r"""
import json, numpy as np
from mrm.core import rng
k = rng.stream_key(12345, rng.stream_id("adc_tx.monitor"))
out = {
  "key": int(k),
  "u": [rng.uniform(k, r, t, s) for r in range(3) for t in (0, 7, 2**40) for s in (0, 5)],
  "z": [rng.normal(k, r, t, 1) for r in range(3) for t in (1, 2)],
  "p": [int(rng.poisson(k, 0, t, 0, lam)) for t in range(4) for lam in (0.5, 9.0, 12.0, 400.0)],
  "b": [int(rng.binomial(k, 0, t, 0, n, p)) for t in range(4) for (n, p) in ((10, .2), (100, .3), (500, .7))],
}
print(json.dumps(out))
"""


def test_identical_bits_without_jit():
    """The same draws with NUMBA_DISABLE_JIT=1 (pure NumPy scalars) as compiled."""
    env = dict(os.environ)
    here = subprocess.run([sys.executable, "-c", _PARITY_SCRIPT], capture_output=True, text=True, env=env,
                          check=True)
    env["NUMBA_DISABLE_JIT"] = "0" if JIT_OFF else "1"
    other = subprocess.run([sys.executable, "-W", "error::RuntimeWarning", "-c", _PARITY_SCRIPT],
                           capture_output=True, text=True, env=env, check=True)
    assert json.loads(here.stdout) == json.loads(other.stdout)


# --------------------------------------------------------------------------------------------------- numeric edge cases

def test_uniform_interval_is_open_at_both_ends():
    """All-ones words used to give exactly 1.0 (x + 0.5 with x = 2**53 - 1 rounds to 2**53), which would make
    us = 0 and divide by zero in PTRS / BTRS."""
    top = rng._uniform53(np.uint64(0xFFFFFFFF), np.uint64(0xFFFFFFFF))
    assert top < 1.0 and top == 1.0 - 2.0**-53
    assert rng._uniform53(np.uint64(0), np.uint64(0)) == 0.5 * 2.0**-53
    assert rng._uniform53(np.uint64(0xFFFFFFFF), np.uint64(0xFFFFFFBF)) < 1.0      # the next value below the top


def test_binomial_small_p_large_n_is_accurate():
    """BINV starts from P(X = 0) = q**n. With log(1 - p) in float64 that was off by about 11 % in the exponent at
    p = 1e-16 (1 - p rounds); log1p fixes it. n = 1e16, p = 1e-16: P(X = 0) = e**-1."""
    n_draws = 4000 if JIT_OFF else 40_000
    x = np.empty((1, n_draws), dtype=np.int64)
    rng.fill_binomial(x, rng.batch_seeds(41, 1), 23, 3, 0, np.full((1, n_draws), 10**16, dtype=np.int64),
                      np.full((1, n_draws), 1e-16))
    p0 = np.mean(x[0] == 0)
    assert abs(p0 - np.exp(-1.0)) < 5.0 * np.sqrt(np.exp(-1.0) * (1 - np.exp(-1.0)) / n_draws)
    assert abs(x[0].mean() - 1.0) < 5.0 / np.sqrt(n_draws)


def test_poisson_saturates_instead_of_overflowing():
    k = rng.stream_key(0, 0)
    assert rng.poisson(k, 0, 0, 0, float("inf")) == 2**62
    assert rng.poisson(k, 0, 0, 0, 1e30) == 2**62
    big = rng.poisson(k, 0, 0, 0, 1e15)
    assert abs(big - 1e15) < 10 * np.sqrt(1e15)


def test_parametric_stream_names_are_case_insensitive():
    assert rng.stream_id("ADC_TX.MONITOR") == rng.stream_id("adc_tx.monitor")
    assert rng.stream_id("schedule") == rng.Stream.SCHEDULE
