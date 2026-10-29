"""Checks the data-path model of mrm.plant.datapath: bit counts, the bit-statistics sampler and the error counter.

Read this first
---------------
The data path gives a controller the observables of the data stream that a real chip could give it. For each lane
and tick the model counts the bits sent (nb) and the ones among them (n1). The pattern_noise mode sets how n1 is
drawn (rho is the ones density):

- off: rho * nb rounded to the nearest integer;
- binomial: Binomial(nb, rho);
- prbs<k>: at rho = 0.5, the ones in the next nb bits of an ITU-T PRBS-k sequence. At other densities, or for a k
  without a table (such as prbs31), it falls back to binomial.

From these counts the model makes three observables:

- the bit-statistics sampler: the number of bits in a window whose monitor sample reads above a threshold code,
  drawn as Binomial(n1, p1) + Binomial(n0, p0);
- the receiver error counter: Poisson(sum of BER(OMA) * bits) over each counting period, delivered after a latency;
- BER(OMA): a Gaussian Q model scaled from the receiver sensitivity, BER = 0.5 erfc(Q / sqrt(2)).

The tests are verification tests: they check the code against hand calculations, not against hardware. The last
group checks the batch rule of the random streams (element b of a batch equals the B = 1 run with seed base + b),
the adapter from plant parameters, and that the kernels give the same numbers with numba on and off.
"""
import math
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import erfc

from mrm.core import rng
from mrm.plant import datapath

DT = 1e-6   # s, plant tick


def _datapath_hw(B=1, N=1, **kw):
    """A DatapathHw for B worlds and N lanes at 50 Gb/s and the 1 us tick. Keyword arguments override build inputs."""
    base = dict(bitrate_hz=50e9)
    base.update(kw)
    return datapath.DatapathHw.build(B, N, DT, **base)


# ---------------------------------------------------------------------------------------------------------- PRBS tables

@pytest.mark.parametrize("k", [7, 9, 15])
def test_prbs_tables_are_maximal_sequences(k):
    """ITU-T O.150 PRBS-k: period 2^k - 1, 2^(k-1) ones, the recurrence a_m = a_{m-t} ^ a_{m-k}, and (k <= 9)
    every non-zero k-bit window exactly once per period."""
    t = datapath.PRBS_TAPS[k]
    bits = datapath.prbs_bits(k, t)
    n = (1 << k) - 1
    assert bits.size == n and int(bits.sum()) == 1 << (k - 1)
    ext = np.concatenate([bits, bits])
    m = np.arange(k, n + k)
    np.testing.assert_array_equal(ext[m], ext[m - t] ^ ext[m - k])
    words, pre, period = datapath.prbs_table(k)
    assert period == n and pre[-1] == 1 << (k - 1)
    if k <= 9:
        windows = {int("".join(map(str, ext[i:i + k])), 2) for i in range(n)}
        assert len(windows) == n and 0 not in windows


def test_prbs_window_ones_matches_bruteforce():
    rs = np.random.default_rng(0)
    for k in (7, 9):
        words, pre, period = datapath.prbs_table(k)
        bits = datapath.prbs_bits(k, datapath.PRBS_TAPS[k])
        tiled = np.tile(bits, 40)
        for _ in range(300):
            start = int(rs.integers(0, 5 * period))
            n = int(rs.integers(0, 3 * period))
            want = int(tiled[start % period:start % period + n].sum())
            assert datapath.prbs_window_ones(words, pre, period, start, n) == want
    assert datapath.popcount32(np.int64(0xFFFFFFFF)) == 32 and datapath.popcount32(np.int64(0x80000001)) == 2


# -------------------------------------------------------------------------------------------------- ones count per tick

def test_pattern_off_and_extreme_densities_are_exact():
    hw = _datapath_hw(1, 3, pattern_noise="off")
    bank = datapath.DatapathBank(hw, [0])
    rho = bank.pattern(0, [0.5, 0.0, 1.0])
    np.testing.assert_array_equal(bank.nb, [[50000] * 3])
    np.testing.assert_array_equal(bank.n1, [[25000, 0, 50000]])
    np.testing.assert_array_equal(rho, [[0.5, 0.0, 1.0]])
    rho = bank.pattern(1, 0.3, traffic_on=[[True, False, True]])
    np.testing.assert_array_equal(bank.nb, [[50000, 0, 50000]])
    np.testing.assert_array_equal(rho, [[0.3, 0.3, 0.3]])
    for mode in ("binomial", "prbs7"):
        bank = datapath.DatapathBank(_datapath_hw(1, 2, pattern_noise=mode), [0])
        bank.pattern(0, [0.0, 1.0])
        np.testing.assert_array_equal(bank.n1, [[0, 50000]])


def test_pattern_binomial_statistics():
    """pattern_noise binomial: n1 ~ Binomial(n, rho) per tick; mean n rho, variance n rho (1 - rho)."""
    B, N = 200, 4
    hw = _datapath_hw(B, N, pattern_noise="binomial")
    bank = datapath.DatapathBank(hw, rng.batch_seeds(1, B))
    dens = np.array([0.5, 0.25, 0.9, 0.5])
    n1 = []
    for k in range(20):
        bank.pattern(k, dens)
        n1.append(bank.n1.copy())
    n1 = np.array(n1).reshape(-1, N)
    n = 50000
    np.testing.assert_allclose(n1.mean(0), n * dens, rtol=2e-3)
    np.testing.assert_allclose(n1.var(0), n * dens * (1 - dens), rtol=0.08)


def test_pattern_prbs_counts_follow_the_sequence():
    """prbs7 at density 0.5: n1 of each tick is the ones count of the next 50 000 bits of the sequence from the
    lane's random start phase; the ones density is nearly constant (period 127)."""
    hw = _datapath_hw(2, 3, pattern_noise="prbs7")
    bank = datapath.DatapathBank(hw, [5, 6])
    bits = datapath.prbs_bits(7, 6)
    start = bank.bitpos.copy()
    tiled = np.tile(bits, 50000 // 127 + 3)
    for k in range(3):
        bank.pattern(k, 0.5)
        for b in range(2):
            for n in range(3):
                s = (start[b, n] + k * 50000) % 127
                assert bank.n1[b, n] == int(tiled[s:s + 50000].sum())
    assert np.all(np.abs(bank.rho - 0.5) < 0.005)
    # prbs31 is not tabulated: binomial fallback
    hw31 = _datapath_hw(1, 1, pattern_noise="prbs31")
    assert datapath.DatapathBank(hw31, [0]).period == 0


def test_non_integral_bits_per_tick():
    """53.125 Gb/s at 0.1 us is 5312.5 bits per tick: the count alternates between 5312 and 5313, exactly
    floor((k + 1) R dt) - floor(k R dt)."""
    hw = datapath.DatapathHw.build(1, 1, 0.1e-6, bitrate_hz=53.125e9)
    assert hw.bpt_i == -1
    bank = datapath.DatapathBank(hw, [0])
    nb = []
    for k in range(10):
        bank.pattern(k, 0.5)
        nb.append(int(bank.nb[0, 0]))
    assert set(nb) == {5312, 5313} and sum(nb) == 53125


def test_ones_count_and_bits_channels_sum_the_window():
    hw = _datapath_hw(1, 2, pattern_noise="binomial")
    bank = datapath.DatapathBank(hw, [3])
    n1 = np.zeros((1, 2), np.int64)
    for k in range(10):
        bank.pattern(k, 0.5)
        n1 += bank.n1
    s, new = bank.release(10)
    np.testing.assert_array_equal(s["tx.ones_count"], n1)
    np.testing.assert_array_equal(s["tx.bits"], [[500000, 500000]])
    s, _ = bank.release(10)
    assert not s["tx.bits"].any()                    # 'sum' channels report only what happened since the last step


# ------------------------------------------------------------------------------------- bit error rate and error counter

def test_ber_from_oma_gaussian_q():
    """BER = 0.5 erfc(Q/sqrt2), Q = Q_ref OMA_rx/OMA_sens: BER(sensitivity) = BER_ref = 1e-9, Q_ref = 5.998."""
    q = datapath.q_from_ber(1e-9)
    assert q == pytest.approx(5.997807, rel=1e-6)
    sens = 1e-3 * 10 ** (-12.8 / 10)
    # abs=0: pytest.approx adds abs=1e-12 by default, which would accept any BER below 1e-12
    assert datapath.ber_from_oma(sens, sens, q, 0.0) == pytest.approx(1e-9, rel=1e-9, abs=0)
    assert datapath.ber_from_oma(2 * sens, sens, q, 0.0) == pytest.approx(0.5 * erfc(2 * q / math.sqrt(2)), rel=1e-9,
                                                                          abs=0)
    assert 1e-34 < datapath.ber_from_oma(2 * sens, sens, q, 0.0) < 1e-32
    assert datapath.ber_from_oma(0.0, sens, q, 0.0) == 0.5
    assert datapath.ber_from_oma(10 * sens, sens, q, 1e-15) == 1e-15
    hw = _datapath_hw(1, 2, rx_sensitivity_oma_dbm=[-14.0, -12.8], link_loss_db=3.0)
    oma_tx = 1e-3 * 10 ** ((-12.8 + 3.0) / 10)       # exactly the sensitivity of lane 1 after the 3 dB link
    b = hw.ber(oma_tx)
    assert b[0, 1] == pytest.approx(1e-9, rel=1e-9, abs=0) and b[0, 0] < 1e-9


def test_error_count_poisson_mean_and_delivery():
    """rx.error_count = Poisson(sum of BER(OMA) * bits) per counting period: its mean and variance equal lambda.
    A count is delivered at the period boundary plus the latency and then held. rx.error_count_valid is set on
    delivery."""
    B, N = 400, 2
    period_s, lat_s = 1e-3, 50e-6
    hw = _datapath_hw(B, N, error_count_period_s=period_s, error_count_latency_s=lat_s, rx_sensitivity_oma_dbm=-12.8,
                      link_loss_db=0.0)
    lam_target = np.array([3.7, 0.4])
    # choose OMA so that BER * 50 000 bits * 1000 ticks = lam_target
    from scipy.special import erfcinv
    ber = lam_target / (50000 * 1000)
    oma = math.sqrt(2) * erfcinv(2 * ber) / hw.q_ref * hw.oma_sens_w[0]
    bank = datapath.DatapathBank(hw, rng.batch_seeds(17, B))
    counts, deliveries = [], []
    for k in range(3000 + 60):
        bank.pattern(k, 0.5)
        bank.errors(k, oma)
        if (k + 1) % 10 == 0:
            s, new = bank.release(k + 1)
            if s["rx.error_count_valid"].any():
                assert s["rx.error_count_valid"].all() and new["rx.error_count"].all()
                deliveries.append(k + 1)
                counts.append(s["rx.error_count"].copy())
            else:
                assert not new["rx.error_count"].any()
    assert deliveries == [1050, 2050, 3050]
    c = np.array(counts).reshape(-1, N)
    np.testing.assert_allclose(c.mean(0), lam_target, rtol=0.08)
    np.testing.assert_allclose(c.var(0), lam_target, rtol=0.15)


# ----------------------------------------------------------------------------------------------- bit-statistics sampler

@pytest.mark.parametrize("mode", ["binomial", "off", "prbs9"])
def test_bitstat_expectation_equals_analytic(mode):
    """E[bitstat_above] = n (rho p1 + (1 - rho) p0), with p = P(N(mu, sig^2) >= thr + 1/2) for each level.
    For binomial data, Var = n rho p1 q1 + n (1 - rho) p0 q0 + n rho (1 - rho) (p1 - p0)^2, with q = 1 - p."""
    B, N = 3000, 3
    n_win = 64
    hw = _datapath_hw(B, N, pattern_noise=mode, bitstat_window_bits=n_win)
    bank = datapath.DatapathBank(hw, rng.batch_seeds(2, B))
    bank.pattern(0, 0.5)
    mu1 = np.array([1000.0, 1000.0, 520.0])
    mu0 = np.array([200.0, 600.0, 480.0])
    sig1 = np.array([30.0, 150.0, 25.0])
    sig0 = np.array([20.0, 100.0, 25.0])
    thr = np.array([600.0, 800.0, 500.0])
    rho = 0.5
    above = bank.bitstat(1, mu1, mu0, sig1, sig0, thr, rho).astype(float)
    p1 = 0.5 * erfc((thr + 0.5 - mu1) / (sig1 * math.sqrt(2)))
    p0 = 0.5 * erfc((thr + 0.5 - mu0) / (sig0 * math.sqrt(2)))
    mean = n_win * (rho * p1 + (1 - rho) * p0)
    var = n_win * (rho * p1 * (1 - p1) + (1 - rho) * p0 * (1 - p0)) + n_win * rho * (1 - rho) * (p1 - p0) ** 2
    if mode == "off":
        var = n_win * (rho * p1 * (1 - p1) + (1 - rho) * p0 * (1 - p0))
    std_err = np.sqrt(var / B)
    assert np.all(np.abs(above.mean(0) - mean) < 4 * std_err + 1e-9)
    if mode != "prbs9":       # a PRBS window has its own (smaller) ones-count spread
        np.testing.assert_allclose(above.var(0), var, rtol=0.12, atol=1e-6)
    np.testing.assert_array_equal(bank.bs_a1 + bank.bs_a0, bank.bs_above)
    assert np.all(bank.bs_n1 <= n_win)
    s, new = bank.release(1)
    assert s["datapath.bitstat_above"].dtype == np.int32 and new["datapath.bitstat_above"].all()


def test_bitstat_prbs_window_is_the_last_bits():
    hw = _datapath_hw(1, 2, pattern_noise="prbs7", bitstat_window_bits=100)
    bank = datapath.DatapathBank(hw, [4])
    bank.pattern(0, 0.5)
    bits = np.tile(datapath.prbs_bits(7, 6), 3)
    bank.bitstat(1, np.full(2, 1e4), np.full(2, -1e4), np.ones(2), np.ones(2), np.zeros(2), 0.5)
    for n in range(2):
        end = int(bank.bitpos[0, n]) + 127
        assert bank.bs_n1[0, n] == int(bits[end - 100:end].sum())
        assert bank.bs_above[0, n] == bank.bs_n1[0, n]            # p1 = 1, p0 = 0


def test_bitstat_from_monitor_levels():
    """The sampler levels come from the monitor's ADC transfer: with a threshold half-way between the two level codes
    and a quiet sampler, every '1' bit reads above and every '0' below."""
    from mrm.plant import sensing
    mon = sensing.MonitorBank(sensing.MonitorHw.build(1, 2, DT, responsivity_a_per_w=1.0, adc_full_scale_a=40e-6,
                                            input_noise_a_per_rt_hz=1e-15, shot_noise=False), [0])
    hw = _datapath_hw(1, 2, pattern_noise="binomial", bitstat_window_bits=128)
    bank = datapath.DatapathBank(hw, [1])
    bank.pattern(0, 0.5)
    thr = mon.hw.ideal_code(12.5e-6)[0]
    above = bank.bitstat_from_monitor(1, mon, 20e-6, 5e-6, thr, 0.5)
    np.testing.assert_array_equal(above, bank.bs_n1)
    assert 0 < bank.bs_n1.min() < 128


def test_traffic_on_accepts_the_scenario_driver_enable_float():
    """data.driver_enable is a float flag, on at >= 0.5 (mrm.scenarios): 0.3 must mean no traffic."""
    bank = datapath.DatapathBank(_datapath_hw(1, 3), [0])
    bank.pattern(0, 0.5, traffic_on=[[1.0, 0.3, 0.5]])
    np.testing.assert_array_equal(bank.nb, [[50000, 0, 50000]])


def test_bitstat_sampler_noise_override_and_reset():
    """bitstat_noise_a_per_rt_hz replaces the monitor's noise density in the sampler sigma (a wider p_above
    transition); reset() clears the window state and the delivery flags."""
    from mrm.plant import sensing
    B = 4000

    def monitor(b):
        return sensing.MonitorBank(sensing.MonitorHw.build(b, 1, DT, responsivity_a_per_w=1.0, adc_full_scale_a=40e-6,
                                                           input_noise_a_per_rt_hz=1e-12, shot_noise=False),
                                   rng.batch_seeds(0, b))
    hw = _datapath_hw(B, 1, pattern_noise="off", bitstat_window_bits=64, bitstat_noise_a_per_rt_hz=20e-12)
    assert hw.bitstat_noise_a_per_rt_hz == 20e-12
    mon = monitor(1)
    thr = float(mon.hw.ideal_code(12.5e-6)[0, 0])
    mu1, _, s1, _ = mon.level_codes(14e-6, 5e-6, tau_s=hw.bitstat_tau_s, noise_a_per_rt_hz=20e-12)
    p1 = 0.5 * erfc((thr + 0.5 - mu1[0, 0]) / (s1[0, 0] * math.sqrt(2)))
    assert 0.6 < p1 < 0.99                                   # the 20 pA/rtHz sampler blurs the '1' level
    bank = datapath.DatapathBank(hw, rng.batch_seeds(0, B))
    bank.pattern(0, 0.5)
    above = bank.bitstat_from_monitor(1, monitor(B), 14e-6, 5e-6, thr, 0.5).astype(float)
    assert above.mean() == pytest.approx(32 * p1, rel=0.02)   # the 32 zeros never read above
    bank.reset()
    assert not bank.bs_above.any() and not bank.bs_new and not bank.err_valid.any() and not bank.nb.any()
    with pytest.raises(ValueError):
        _datapath_hw(1, 1, bitstat_bw_hz=0.0)


# ----------------------------------------------------------------------- batch invariance, parameter adapter, numba off

def test_batch_invariance():
    """Element b of a B = 3 run equals the B = 1 run with seed base + b: each world has its own random stream."""
    def run(B, base):
        hw = _datapath_hw(B, 2, pattern_noise="binomial", error_count_period_s=20e-6)
        bank = datapath.DatapathBank(hw, rng.batch_seeds(base, B))
        out = []
        for k in range(60):
            out.append(bank.pattern(k, 0.4).copy())
            bank.errors(k, 1e-5)
            if (k + 1) % 10 == 0:
                bank.bitstat(k + 1, np.full(2, 500.0), np.full(2, 300.0), np.full(2, 50.0), np.full(2, 50.0),
                             np.full(2, 400.0), 0.4)
                s, _ = bank.release(k + 1)
                out.append(s["datapath.bitstat_above"].astype(float))
                out.append(s["rx.error_count"].astype(float))
        return np.array(out)

    big = run(3, 10)
    for b in range(3):
        np.testing.assert_array_equal(big[:, b], run(1, 10 + b)[:, 0])


def test_from_plant_params_mini():
    from mrm.params import load, realize
    pp = realize(load(os.path.join(os.path.dirname(__file__), "..", "data", "mini.yaml")), batch=2, seeds=0)
    hw = datapath.DatapathHw.from_plant_params(pp)
    assert (hw.B, hw.N, hw.bpt_i, hw.mode, hw.window_bits) == (2, 4, 50000, datapath.PATTERN_OFF, 64)
    assert hw.error_period_ticks == 2_000_000
    np.testing.assert_allclose(hw.oma_sens_w, 1e-3 * 10 ** (-12.8 / 10))


# A short binomial data path with error counts and bit statistics. The test below runs it with numba on and off.
_NOJIT_SCRIPT = r"""
import numpy as np
from mrm.core import rng
from mrm.plant import datapath
hw = datapath.DatapathHw.build(2, 3, 1e-6, bitrate_hz=50e9, pattern_noise="binomial", error_count_period_s=20e-6)
bank = datapath.DatapathBank(hw, rng.batch_seeds(9, 2))
out = []
for k in range(80):
    out.append(bank.pattern(k, 0.45).copy())
    bank.errors(k, 3e-5)
    if (k + 1) % 10 == 0:
        bank.bitstat(k + 1, np.full(3, 900.0), np.full(3, 300.0), np.full(3, 80.0), np.full(3, 60.0),
                     np.full(3, 600.0), 0.45)
        s, _ = bank.release(k + 1)
        out.append(s["datapath.bitstat_above"].astype(float))
        out.append(s["rx.error_count"].astype(float))
np.save(OUT, np.array(out))
"""


def test_kernels_identical_without_jit(tmp_path):
    res = {}
    for mode in ("0", "1"):
        out = tmp_path / f"out{mode}.npy"
        env = dict(os.environ, NUMBA_DISABLE_JIT=mode)
        subprocess.run([sys.executable, "-c", f"OUT = {str(out)!r}\n" + _NOJIT_SCRIPT], env=env, check=True,
                       timeout=120)
        res[mode] = np.load(out)
    np.testing.assert_array_equal(res["0"], res["1"])
