"""Verification tests for mrm/plant/sensing.py (07 section 4.1 class: verification; the code against analytic
limits, not against hardware).

Covers 07 P1-9 (PD, Arrhenius dark current, background, reference tap), P1-10 (AFE, ADC offset/gain/INL/DNL/PGA/
tempco/saturation, temperature sensors, heater readback, faults), P1-13 (exact noise discretization: sigma at the ADC
invariant to dt; batch and thread invariance; shot-noise toggle leaves the ADC draws unchanged), P0-7 (sample blocks
with sample ticks, latency, adc_mux phases).
"""
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import scipy.linalg

from mrm import materials
from mrm.core import rng
from mrm.plant import sensing as se
from mrm.units import K_B, Q_E

T25 = 298.15
I_MON = 15e-6      # A, read-off G26 monitor current (01: 15.5 +- 0.5 uA); used as a representative operating point
P_MON = 15e-6      # W at R = 1 A/W


def _hw(B=1, N=1, dt=1e-6, **kw):
    base = dict(responsivity_a_per_w=1.0, adc_full_scale_a=40e-6, adc_bits=16, afe_type="integrate_dump",
                integration_window_s=2e-6, adc_rate_hz=250e3, adc_latency_s=2e-6, shot_noise=False)
    base.update(kw)
    return se.MonitorHw.build(B, N, dt, **base)


def _run(hw, n_ticks, p_pd, seeds=0, m_max=None, release_every=None, t_pd=T25, fault=None, pga=None, k0=0,
         bank=None, device_seeds=None):
    """Tick a MonitorBank and collect every released sample: returns (codes (n, B, N), ticks (n, B, N), bank)."""
    P = hw.period_ticks
    rel_every = P if release_every is None else release_every
    if bank is None:
        bank = se.MonitorBank(hw, rng.batch_seeds(seeds, hw.B), device_seeds=device_seeds,
                              m_max=m_max or max(1, -(-rel_every // P)), pop_period_ticks=rel_every)
    codes, ticks = [], []
    for k in range(k0, k0 + n_ticks):
        p = p_pd(k) if callable(p_pd) else p_pd
        f = fault(k) if callable(fault) else fault
        bank.tick(k, p, 0.0, t_pd, None, pga, f)
        if (k + 1) % rel_every == 0:
            r = bank.release(k + 1)
            for m in range(int(r.count.max())):
                codes.append(r.samples[:, :, m].astype(np.int64).copy())
                ticks.append(r.sample_ticks[:, :, m].copy())
    return np.array(codes), np.array(ticks), bank


# ============================================================================================ photodiode
def test_dark_current_arrhenius_matches_materials():
    """I_d = I_d0 exp[-(E_a/k)(1/T - 1/T0)] (07 P1-9), equal to mrm.materials.ge_pd_dark_current_a."""
    i0 = 5e-9
    for ea_ev in (0.35, 0.37, 0.38):
        for t in (273.15, 298.15, 358.15, 378.15):
            got = se.dark_current_a(i0, t, T25, ea_ev * Q_E)
            want = i0 * math.exp(-(ea_ev * Q_E / K_B) * (1 / t - 1 / T25))
            # abs=0 everywhere below: pytest.approx adds abs=1e-12 by default, which swamps SI-scale values
            assert got == pytest.approx(want, rel=1e-13, abs=0)
            assert got == pytest.approx(float(materials.ge_pd_dark_current_a(i0, t, T25, ea_ev)), rel=1e-12, abs=0)
    # x per 10 K from 25 C: 1.56 (0.35 eV) to 1.62 (0.38 eV)
    r35 = se.dark_current_a(1.0, T25 + 10, T25, 0.35 * Q_E)
    r38 = se.dark_current_a(1.0, T25 + 10, T25, 0.38 * Q_E)
    assert 1.55 < r35 < 1.57 and 1.61 < r38 < 1.63
    assert se.dark_current_a(0.0, 350.0, T25, 0.37 * Q_E) == 0.0


def test_dark_current_reaches_the_codes():
    """With no light, the monitor reads the Arrhenius dark current (07 P1-9 test_dark_current_vs_heater analogue)."""
    hw = _hw(B=1, N=3, adc_bits=24, dark_i0_a=30e-9, dark_activation_ev=0.37)
    t_pd = np.array([[T25, 330.0, 378.15]])
    codes, _, _ = _run(hw, 40, 0.0, t_pd=t_pd)
    want = [se.dark_current_a(30e-9, t, T25, 0.37 * Q_E) for t in t_pd[0]]
    np.testing.assert_allclose(codes[-1, 0] * hw.lsb_a[0], want, rtol=1e-5)


# ============================================================================================ noise (07 P1-13)
@pytest.mark.parametrize("afe", ["integrate_dump", "rc"])
def test_monitor_noise_sigma_invariant_to_dt(afe):
    """07 P1-13: the monitor-noise sigma at the ADC stays within 5 % for dt from 0.25 to 2 us (exact Gramian /
    S/(2 T_w)); the mean equals the photocurrent plus the dark current."""
    B, N = 16, 2
    sig_ref = None
    for dt in (0.25e-6, 0.5e-6, 1e-6, 2e-6):
        hw = _hw(B, N, dt, adc_bits=24, adc_full_scale_a=40e-6, dark_i0_a=5e-9, input_noise_a_per_rt_hz=20e-12,
                 afe_type=afe, afe_tau_s=1e-6, shot_noise=True)
        P = hw.period_ticks
        codes, _, _ = _run(hw, 205 * P, P_MON, seeds=11)
        x = codes[5:].astype(float) * hw.lsb_a[0]
        i_sig, i_d = hw.photocurrent_a(P_MON, 0.0, T25)
        sig = hw.white_sigma_a(i_sig, i_d)[0, 0]
        assert x.mean() == pytest.approx(i_sig[0, 0] + i_d[0, 0], rel=2e-5, abs=0)
        assert x.std() == pytest.approx(sig, rel=0.05, abs=0), (afe, dt)
        sig_ref = x.std() if sig_ref is None else sig_ref
        assert x.std() == pytest.approx(sig_ref, rel=0.05, abs=0)
    # the analytic sigma itself: shot + TIA over the 2 us window or through the 1 us RC
    s = 2 * Q_E * (I_MON + 5e-9) + (20e-12) ** 2
    want = math.sqrt(s / (2 * 2e-6)) if afe == "integrate_dump" else math.sqrt(s / (4 * 1e-6))
    assert sig == pytest.approx(want, rel=1e-12, abs=0)


def test_rc_noise_starts_stationary():
    """The rc noise state starts in its stationary distribution: the first sample already has sigma^2 = S/(4 tau)."""
    B = 2000
    hw = _hw(B, 1, 1e-6, adc_bits=24, afe_type="rc", afe_tau_s=5e-6, input_noise_a_per_rt_hz=50e-12)
    codes, ticks, _ = _run(hw, 2 * hw.period_ticks, 5e-6, seeds=3)   # 5 uA keeps the codes off the rail
    assert ticks[0, 0, 0] == hw.period_ticks                 # the first sample after the start
    x = codes[0, :, 0] * hw.lsb_a[0]
    assert x.mean() == pytest.approx(5e-6, rel=1e-3, abs=0)
    assert x.std() == pytest.approx(math.sqrt((50e-12) ** 2 / (4 * 5e-6)), rel=0.05, abs=0)


def test_shot_noise_toggle_leaves_adc_draws_unchanged():
    """07 P1-13: shot noise has its own stream. (a) With no current at all (zero shot PSD) the codes with shot on and
    off are identical; (b) with light, (on - off) is the shot noise alone: rms 2 q I/(2 W) and uncorrelated with the
    TIA residual of the shot-off run."""
    B = 64
    kw = dict(adc_bits=24, input_noise_a_per_rt_hz=30e-12, dark_i0_a=0.0)
    on = _hw(B, 1, shot_noise=True, **kw)
    off = _hw(B, 1, shot_noise=False, **kw)
    c_on, _, _ = _run(on, 200, 0.0, seeds=5)
    c_off, _, _ = _run(off, 200, 0.0, seeds=5)
    np.testing.assert_array_equal(c_on, c_off)
    c_on, _, _ = _run(on, 400, P_MON, seeds=5)
    c_off, _, _ = _run(off, 400, P_MON, seeds=5)
    lsb = on.lsb_a[0]
    d = (c_on - c_off).astype(float).ravel() * lsb
    resid = (c_off.astype(float).ravel() * lsb) - I_MON
    assert d.std() == pytest.approx(math.sqrt(2 * Q_E * I_MON / (2 * 2e-6)), rel=0.05, abs=0)
    assert abs(np.corrcoef(d, resid)[0, 1]) < 0.06


def test_batch_and_thread_invariance():
    """07 P1-13: element b of a B = 4 run equals the B = 1 run with seed base + b (bitwise), and the prange kernel
    gives the same bits as the sequential one for any thread count."""
    kw = dict(adc_bits=20, input_noise_a_per_rt_hz=30e-12, dark_i0_a=5e-9, shot_noise=True, flicker_corner_hz=1e5,
              adc_dnl_lsb=0.3)
    big = _hw(4, 3, **kw)
    c_big, _, _ = _run(big, 120, P_MON, seeds=40, device_seeds=np.arange(4) + 7)
    for b in range(4):
        one = _hw(1, 3, **kw)
        c_one, _, _ = _run(one, 120, P_MON, seeds=40 + b, device_seeds=[7 + b])
        np.testing.assert_array_equal(c_big[:, b], c_one[:, 0])

    import numba
    results = []
    for threads in sorted({1, min(4, numba.config.NUMBA_NUM_THREADS)}):
        numba.set_num_threads(threads)
        hw = _hw(8, 3, **kw)
        bank = se.MonitorBank(hw, rng.batch_seeds(40, 8), m_max=1, pop_period_ticks=4)
        out = []
        for k in range(80):
            bank.tick(k, P_MON, 0.0, T25, parallel=True)
            if (k + 1) % 4 == 0:
                r = bank.release(k + 1)
                if r.count.max() > 0:
                    out.append(r.samples.copy())
        results.append(np.array(out))
    numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
    hw = _hw(8, 3, **kw)
    seq, _, _ = _run(hw, 80, P_MON, seeds=40, release_every=4)
    for r in results:
        np.testing.assert_array_equal(r[:, :, :, 0], seq)


# ============================================================================================ AFE (deterministic)
def test_integrate_dump_window_mean_is_exact():
    """A sample with sample tick s is the mean current over the intervals s - W .. s - 1 (07 P0-6, textbook 12.42)."""
    dt = 0.5e-6
    hw = _hw(1, 1, dt, adc_bits=24, integration_window_s=1.5e-6, adc_rate_hz=1 / 3e-6, adc_latency_s=0.0)
    W, P = hw.window_ticks, hw.period_ticks
    assert (W, P) == (3, 6)

    def p_of(k):
        return 1e-7 * (1.0 + 0.37 * k + 0.01 * k * k)        # stays below the 40 uA full scale

    codes, ticks, _ = _run(hw, 60, p_of, release_every=1)
    for c, s in zip(codes[:, 0, 0], ticks[:, 0, 0]):
        want = np.mean([p_of(k) for k in range(s - W, s)])
        assert c == round(want / hw.lsb_a[0])


def test_rc_deterministic_response_is_exact_zoh():
    """rc: y_{k+1} = a y_k + (1 - a) i_k exactly (ZOH input); a step from I0 to I1 at tick 20 gives
    y(s) = I1 + (I0 - I1) exp(-(s - 20) dt / tau), independent of dt."""
    for dt in (0.25e-6, 1e-6):
        hw = _hw(1, 1, dt, adc_bits=24, afe_type="rc", afe_tau_s=3e-6, adc_rate_hz=250e3, adc_latency_s=0.0)
        k_step = int(round(20e-6 / dt))
        codes, ticks, _ = _run(hw, int(round(60e-6 / dt)), lambda k: 5e-6 if k < k_step else 25e-6,
                               release_every=1)
        for c, s in zip(codes[:, 0, 0], ticks[:, 0, 0]):
            t_rel = max(s - k_step, 0) * dt
            want = 25e-6 + (5e-6 - 25e-6) * math.exp(-t_rel / 3e-6) if s > k_step else 5e-6
            assert abs(c * hw.lsb_a[0] - want) <= hw.lsb_a[0]


def test_window_and_rc_validation():
    with pytest.raises(ValueError, match="exceeds the ADC period"):
        _hw(integration_window_s=5e-6, adc_rate_hz=250e3)
    with pytest.raises(ValueError, match="afe_tau_s"):
        _hw(afe_type="rc")
    with pytest.raises(ValueError):
        _hw(integration_window_s=2.5e-6)            # not a whole number of 1 us ticks


# ============================================================================================ sample blocks
def test_latency_sample_ticks_block_and_hold():
    """07 P0-7 / P0-6: samples at s = j P (+ phase), released at s + latency, in sample-tick order; the block keeps
    the newest M if more arrive; latest is held between deliveries; new flags mark deliveries."""
    hw = _hw(1, 2, adc_bits=16, adc_latency_s=3e-6, adc_rate_hz=250e3)
    bank = se.MonitorBank(hw, [0], m_max=3, pop_period_ticks=10)
    seen = []
    for k in range(50):
        bank.tick(k, P_MON, 0.0, T25)
        if (k + 1) % 10 == 0:
            r = bank.release(k + 1)
            seen.append((k + 1, r.count[0, 0], list(r.sample_ticks[0, 0, :r.count[0, 0]])))
            assert np.all(r.sample_ticks[0, :, r.count[0, 0]:] == 0)
            assert np.all(r.sample_ticks[0, 0, :r.count[0, 0]] + 3 <= k + 1)
    # sample ticks 4, 8, 12, ...; released at s + 3
    assert seen[0] == (10, 1, [4])          # 8 + 3 = 11 > 10
    assert seen[1] == (20, 3, [8, 12, 16])
    assert seen[2] == (30, 2, [20, 24])
    # overflow: release once after 40 ticks with M = 3 keeps the newest 3
    bank2 = se.MonitorBank(hw, [0], m_max=3, pop_period_ticks=40)
    for k in range(40):
        bank2.tick(k, P_MON, 0.0, T25)
    r = bank2.release(40)
    assert r.count[0, 0] == 3 and list(r.sample_ticks[0, 0]) == [28, 32, 36] and r.lost == 2 * 6
    assert r.latest[0, 0] == r.samples[0, 0, 2]
    held = r.latest.copy()
    r = bank2.release(40)                   # nothing new: latest held, new False
    assert r.count[0, 0] == 0 and not r.new.any()
    np.testing.assert_array_equal(r.latest, held)
    sensors, new = bank2.channels(r)
    assert set(sensors) == {"tx.monitor.samples", "tx.monitor.sample_ticks", "tx.monitor.count",
                            "tx.monitor.latest", "monitor_valid", "adc_overrange"}
    assert sensors["tx.monitor.samples"].dtype == np.int32 and sensors["tx.monitor.count"].dtype == np.int32


def test_adc_mux_round_robin_phases():
    """07 P0-7 adc_mux: C rings per converter are sampled at staggered phases floor((n mod C) P / C)."""
    hw = _hw(1, 8, adc_rate_hz=125e3, channels_per_adc=4)
    assert hw.period_ticks == 8
    np.testing.assert_array_equal(hw.phase, [0, 2, 4, 6, 0, 2, 4, 6])
    _, ticks, _ = _run(hw, 40, P_MON, release_every=1)
    for n in range(8):
        t = ticks[:, 0, n]
        t = t[t > 0]
        assert np.all((t - hw.phase[n]) % 8 == 0)
    assert np.all(_hw(1, 8, adc_rate_hz=125e3, channels_per_adc=4, mux_order="bank").phase == 0)


# ============================================================================================ ADC
def test_adc_offset_gain_pga_tempco_and_saturation():
    """code = (i - i_base) g / LSB (1 + gain_err)/(1 + tc (T - T_ref)) + offset, clipped with the overrange flag."""
    hw = _hw(1, 3, adc_bits=16, adc_offset_codes=[0.0, 100.0, -50.0], adc_gain_error_frac=[0.0, 0.02, -0.01],
             baseline_a=[0.0, 1e-6, 0.0], pga_gains=[1.0, 2.0, 4.0], adc_ref_tempco_per_k=50e-6)
    lsb = hw.lsb_a[0]
    for sel, t_adc in ((0, T25), (1, T25 + 40.0), (2, T25)):
        bank = se.MonitorBank(hw, [0], m_max=1, pop_period_ticks=4)
        for k in range(8):
            bank.tick(k, 5e-6, 0.0, T25, np.full((1, 3), t_adc), np.full((1, 3), sel))
        r = bank.release(8)
        g = [1.0, 2.0, 4.0][sel]
        ref = 1 + 50e-6 * (t_adc - T25)
        want = (np.array([5e-6, 4e-6, 5e-6]) * g / lsb * np.array([1.0, 1.02, 0.99]) / ref
                + np.array([0.0, 100.0, -50.0]))
        np.testing.assert_array_equal(r.samples[0, :, 0], np.round(want).astype(int))
        np.testing.assert_allclose(hw.ideal_code(5e-6, sel, t_adc)[0], want, rtol=1e-12)
        assert not r.overrange.any()
    # saturation: 50 uA > 40 uA full scale -> code max and overrange; negative -> 0 and overrange
    bank = se.MonitorBank(hw, [0], m_max=1, pop_period_ticks=4)
    for k in range(8):
        bank.tick(k, 50e-6, 0.0, T25)
    r = bank.release(8)
    assert r.samples[0, 0, 0] == 65535 and r.overrange[0, 0]
    code, over = se.adc_quantize(-3.0, 0.0, 0.0, 65535, np.uint64(1), 0, 0)
    assert (code, over) == (0, 1)
    code, over = se.adc_quantize(float("nan"), 0.0, 0.0, 65535, np.uint64(1), 0, 0)
    assert (code, over) == (0, 1)


def test_dnl_code_widths_and_monotonicity():
    """07 P1-10 DNL: code widths are 1 + d (u_{c+1} - u_c), so |DNL| < d, thresholds monotone, no missing codes for
    d < 1; d = 0 gives exact rounding."""
    key = rng.stream_key(3, rng.stream_id("adc_monitor_dnl"))
    v = np.arange(-0.4, 250.0, 0.004)
    for d in (0.0, 0.5, 0.95):
        codes = np.array([se.adc_quantize(x, 0.0, d, 65535, np.uint64(key), 2, 0)[0] for x in v])
        assert np.all(np.diff(codes) >= 0)
        assert set(np.unique(codes)) == set(range(codes[0], codes[-1] + 1))      # no missing codes
        widths = np.bincount(codes)[1:-1] * 0.004
        assert np.all(np.abs(widths - 1.0) <= d + 0.0081)
        if d == 0.0:
            np.testing.assert_array_equal(codes, np.floor(v + 0.5).astype(int))
        else:
            assert np.abs(widths - 1.0).max() > 0.5 * d                           # the spread is really there


def test_inl_bow_and_device_draw():
    """INL bow: v -> v + bow 4u(1 - u), so mid-scale shifts by bow and the end points not at all; the per-device
    bow is uniform in +-inl from the device seed (reproducible, independent of the noise seed)."""
    cmax = 4095
    for bow, want in ((3.0, 2050), (-2.0, 2045)):           # v = 2047 is u = 0.4999, 4u(1 - u) = 0.99999994
        assert se.adc_quantize(2047.0, bow, 0.0, cmax, np.uint64(1), 0, 0)[0] == want
        assert se.adc_quantize(0.2, bow, 0.0, cmax, np.uint64(1), 0, 0)[0] == 0
    a = se.draw_inl_bow_lsb([1, 2, 3], 8, 2.0)
    b = se.draw_inl_bow_lsb([1, 2, 3], 8, 2.0)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (3, 8) and np.all(np.abs(a) <= 2.0) and np.std(a) > 0.5


# ============================================================================================ faults
def test_fault_drop_stuck_saturate():
    """07 P1-10 masks: drop -> no sample and monitor_valid False; stuck -> the last good code repeated with
    monitor_valid False (or silent); saturate -> full scale with adc_overrange."""
    hw = _hw(1, 3, input_noise_a_per_rt_hz=100e-12, adc_bits=20)

    def fault(k):
        f = np.zeros((1, 3), np.int8)
        if 8 <= k < 16:
            f[0] = [se.FAULT_ADC_DROP, se.FAULT_ADC_STUCK, se.FAULT_ADC_SATURATE]
        return f

    bank = se.MonitorBank(hw, [0], m_max=1, pop_period_ticks=4)
    out = []
    for k in range(24):
        bank.tick(k, P_MON, 0.0, T25, fault=fault(k))
        if (k + 1) % 4 == 0:
            r = bank.release(k + 1)
            out.append((r.count[0].copy(), r.samples[0, :, 0].copy(), r.valid[0].copy(), r.overrange[0].copy()))
    # samples at 4, 8, ... released at s + 2: the faulted samples are those with sample tick 12, 16
    good_before = out[2][1][1]                               # ring 1's code at sample tick 8 (released at 12)
    for i in (3, 4):                                         # releases at 16 and 20
        cnt, code, valid, over = out[i]
        assert cnt[0] == 0 and not valid[0]                  # dropped
        assert code[1] == good_before and not valid[1]       # stuck at the last good code
        assert code[2] == (1 << 20) - 1 and over[2] and valid[2]
    assert out[5][2].all() and out[5][0][0] == 1             # recovered
    silent = _hw(1, 1, stuck_flagged=False)
    bank = se.MonitorBank(silent, [0], m_max=1, pop_period_ticks=4)
    for k in range(12):
        bank.tick(k, P_MON, 0.0, T25, fault=se.FAULT_ADC_STUCK if k >= 4 else 0)
    assert bank.release(12).valid.all()


def test_dac_glitch():
    codes = np.array([[100, 200, 65530]], np.int64)
    fault = np.array([[0, se.FAULT_DAC_GLITCH, se.FAULT_DAC_GLITCH]], np.int8)
    out = np.zeros_like(codes)
    se.dac_glitch(codes, fault, np.array([[512]], np.int64), 0, 65535, out)
    np.testing.assert_array_equal(out, [[100, 712, 65535]])


# ============================================================================================ flicker
def test_flicker_pole_set_psd_is_one_over_f():
    """The OU sum approximates S f_c / f within 0.05 dB more than two decades inside the band and within 0.3 dB one
    decade inside it (2 poles per decade; the band edges roll off by design)."""
    s_w, fc = 1e-22, 1e5
    lam, var = se.flicker_pole_set(s_w, fc, 1e-2, 1e7, per_decade=2)
    for lo, hi, tol in ((1.0, 1e5, 0.05), (1e-1, 1e6, 0.3)):
        f = np.logspace(np.log10(lo), np.log10(hi), 500)
        err_db = 10 * np.log10(se.flicker_psd(f, lam, var) / (s_w * fc / f))
        assert np.abs(err_db).max() < tol
    assert se.flicker_pole_set(0.0, fc)[0].size == 0


def _van_loan(a, q, t):
    n = a.shape[0]
    m = np.zeros((2 * n, 2 * n))
    m[:n, :n] = -a
    m[:n, n:] = q
    m[n:, n:] = a.T
    e = scipy.linalg.expm(m * t)
    phi = e[n:, n:].T
    return phi, phi @ e[:n, n:]


def test_ou_coefficients_match_van_loan():
    """The closed-form transition and Gramian equal Van Loan's matrix-exponential method (07 P1-13)."""
    for lam in (2e3, 3e5, 1.1e6):
        var = 2.0
        q = np.array([[2 * lam * var, 0.0], [0.0, 0.0]])
        # rc: state (x, y), dy = mu (x - y)
        mu, P = 1 / 1.5e-6, 4e-6
        row = se.ou_rc_coefficients([lam], [var], P, 1.5e-6)[0]
        phi, Q = _van_loan(np.array([[-lam, 0.0], [mu, -mu]]), q, P)
        np.testing.assert_allclose(row[[0, 1, 2, 3]], phi.ravel(), rtol=1e-9, atol=1e-14)
        L = np.array([[row[4], 0.0], [row[5], row[6]]])
        np.testing.assert_allclose(L @ L.T, Q, rtol=1e-8, atol=1e-14 * var)
        # integrate-and-dump: free run over D = P - W, then (x, integral) over W, contribution = integral / W
        W = 2e-6
        row = se.ou_window_coefficients([lam], [var], P, W)[0]
        phi_d, q_d = _van_loan(np.array([[-lam]]), np.array([[2 * lam * var]]), P - W)
        phi_w, q_w = _van_loan(np.array([[-lam, 0.0], [1.0, 0.0]]), q, W)
        sc = np.diag([1.0, 1.0 / W])
        mean = sc @ phi_w @ np.array([phi_d[0, 0], 0.0])
        cov = sc @ (phi_w @ np.diag([q_d[0, 0], 0.0]) @ phi_w.T + q_w) @ sc
        np.testing.assert_allclose([row[0], row[2]], mean, rtol=1e-9)
        L = np.array([[row[4], 0.0], [row[5], row[6]]])
        np.testing.assert_allclose(L @ L.T, cov, rtol=1e-7, atol=1e-13 * var)


def _window_mean_var(lam, var, w_s):
    """Stationary variance of the W-window mean of a sum of OU processes: sum_j 2 var_j (u - 1 + e^-u)/u^2,
    u = lam_j W (= m_a^2 var + the conditional part 2 var g(u)/u^2 of ou_window_coefficients)."""
    u = np.asarray(lam) * w_s
    return float(np.sum(2.0 * np.asarray(var) * (u + np.expm1(-u)) / u ** 2))


def test_flicker_sample_variance_is_stationary():
    """The ensemble variance of the flicker part of a sample equals the analytic stationary value at the first sample
    and after 40 propagation steps (the transition and Gramian preserve stationarity). [review: the original check
    compared against the conditional variance 2 var g(u)/u^2 only, and passed because pytest.approx's default
    abs=1e-12 swamped values of 1e-15 A^2.]"""
    B, N = 400, 4
    s_w = (40e-12) ** 2
    hw = _hw(B, N, adc_bits=24, input_noise_a_per_rt_hz=40e-12, flicker_corner_hz=2e5, flicker_f_min_hz=10.0)
    codes, _, _ = _run(hw, 42 * hw.period_ticks, 5e-6, seeds=1)
    lsb = hw.lsb_a[0]
    lam, var = se.flicker_pole_set(s_w, 2e5, 10.0, 10 * min(2e5, 1 / (2 * 2e-6)))
    v_flicker = _window_mean_var(lam, var, 2e-6)
    v_white = s_w / (2 * 2e-6)
    assert v_flicker > 5 * v_white                           # the flicker part dominates, so the check has teeth
    for i in (0, 40):
        x = codes[i].astype(float).ravel() * lsb
        assert x.var() == pytest.approx(v_white + v_flicker, rel=0.12, abs=0)


def test_rc_flicker_sample_variance_is_stationary():
    """rc AFE: the sampled flicker variance is the stationary RC-output variance sum_j var_j mu/(lam_j + mu), plus
    the white S/(4 tau), at the first sample and after 40 steps (ou_rc_coefficients preserve stationarity)."""
    B, N, tau = 400, 4, 3e-6
    s_w = (40e-12) ** 2
    hw = _hw(B, N, adc_bits=24, afe_type="rc", afe_tau_s=tau, input_noise_a_per_rt_hz=40e-12,
             flicker_corner_hz=2e5, flicker_f_min_hz=10.0)
    codes, _, _ = _run(hw, 42 * hw.period_ticks, 5e-6, seeds=2)
    mu = 1.0 / tau
    lam, var = se.flicker_pole_set(s_w, 2e5, 10.0, 10 * min(2e5, mu / (2 * math.pi)))
    v_flicker = float(np.sum(var * mu / (lam + mu)))
    v_white = s_w / (4 * tau)
    assert v_flicker > 5 * v_white
    for i in (0, 40):
        x = codes[i].astype(float).ravel() * hw.lsb_a[0]
        assert x.var() == pytest.approx(v_white + v_flicker, rel=0.12, abs=0)


# ============================================================================================ level codes
def test_monitor_level_codes():
    """mu = ADC transfer of the level current, sigma^2 = (2 q i + S_tia)/(4 tau_s) in codes (07 P0-11 sampler)."""
    hw = _hw(1, 2, adc_bits=16, input_noise_a_per_rt_hz=10e-12, dark_i0_a=0.0, shot_noise=True)
    bank = se.MonitorBank(hw, [0])
    tau = 1 / (2 * math.pi * 16e9)
    mu1, mu0, s1, s0 = bank.level_codes(20e-6, 5e-6, tau_s=tau)
    lsb = hw.lsb_a[0]
    assert mu1[0, 0] == pytest.approx(20e-6 / lsb) and mu0[0, 1] == pytest.approx(5e-6 / lsb)
    assert s1[0, 0] == pytest.approx(math.sqrt((2 * Q_E * 20e-6 + 1e-22) / (4 * tau)) / lsb)
    assert s0[0, 0] < s1[0, 0]


# ============================================================================================ temperature sensors
def test_temperature_sensor_errors_quantization_latency():
    """07 P1-10: T_meas = T_ref + (1 + g)(T_avg - T_ref) + offset + self_heating; code = round((T - T_zero)/res);
    boxcar over the conversion period; released after the latency and held."""
    dt = 1e-6
    sensors = [dict(name="die0", type="diode", sample_rate_hz=10e3, latency_s=50e-6, noise_rms_k=0.0,
                    resolution_k=0.0625, offset_k=1.5, gain_error_frac=0.01, self_heating_k=0.2),
               dict(name="ring3", type="rtd", sample_rate_hz=20e3, latency_s=0.0, noise_rms_k=0.0,
                    resolution_k=0.25, offset_k=-0.5, gain_error_frac=0.0, self_heating_k=0.0)]
    hw = se.TempSensorsHw.build(1, dt, sensors)
    bank = se.TempSensorBank(hw, [0])
    t_of = lambda k: np.array([[330.0 + 1e-3 * k, 310.0]])
    rels = {}
    for k in range(400):
        bank.tick(k, t_of(k))
        s, new = bank.release(k + 1)
        for nm in s:
            if new[nm][0, 0]:
                rels.setdefault(nm, []).append((k + 1, int(s[nm][0, 0])))
    # die0: period 100, latency 50 -> first delivery at 150, value of the window 0..99
    first_tick, code = rels["temp.die0"][0]
    assert first_tick == 150
    t_avg = np.mean([330.0 + 1e-3 * k for k in range(100)])
    t_meas = T25 + 1.01 * (t_avg - T25) + 1.5 + 0.2
    assert code == round((t_meas - 233.15) / 0.0625)
    assert rels["temp.ring3"][0] == (50, round((310.0 - 0.5 - 233.15) / 0.25))
    assert [t for t, _ in rels["temp.die0"]] == [150, 250, 350]


def test_temperature_sensor_noise_sigma():
    sensors = [dict(name="d", sample_rate_hz=100e3, latency_s=0.0, noise_rms_k=0.3, resolution_k=0.01,
                    adc_bits=16)]
    hw = se.TempSensorsHw.build(200, 1e-6, sensors)
    bank = se.TempSensorBank(hw, rng.batch_seeds(0, 200))
    vals = []
    for k in range(200):
        bank.tick(k, 330.0)
        s, new = bank.release(k + 1)
        if new["temp.d"][0, 0]:
            vals.append(s["temp.d"][:, 0] * 0.01 + 233.15)
    x = np.array(vals)
    assert x.std() == pytest.approx(0.3, rel=0.05) and x.mean() == pytest.approx(330.0, abs=0.01)


def test_temperature_sensor_flicker_variance():
    """Temperature-sensor flicker (S_white = 2 sigma^2 T_s, boxcar over the conversion period): the ensemble variance
    of a sample is sigma^2 plus the window-mean variance of the OU set, at the first sample and later."""
    B, T_s, sig, fc = 1500, 10e-6, 0.2, 2e4
    sensors = [dict(name="d", sample_rate_hz=1 / T_s, latency_s=0.0, noise_rms_k=sig, resolution_k=1e-4,
                    adc_bits=24, flicker_corner_hz=fc)]
    hw = se.TempSensorsHw.build(B, 1e-6, sensors)
    assert hw.fl_coef.shape[2] > 0
    bank = se.TempSensorBank(hw, rng.batch_seeds(4, B))
    lam, var = se.flicker_pole_set(2 * sig ** 2 * T_s, fc, 1e-3, 10 * min(fc, 1 / (2 * T_s)))
    v_fl = _window_mean_var(lam, var, T_s)
    got = []
    for k in range(300):
        bank.tick(k, 330.0)
        s, new = bank.release(k + 1)
        if new["temp.d"][0, 0]:
            got.append(s["temp.d"][:, 0] * 1e-4)
    for i in (0, 25):
        assert got[i].var() == pytest.approx(sig ** 2 + v_fl, rel=0.1)


# ============================================================================================ heater readback
def test_heater_readback_codes_and_latency():
    hw = se.ReadbackHw.build(1, 2, 1e-6, adc_bits=12, v_full_scale_v=2.0, i_full_scale_a=0.032, rate_hz=10e3,
                             latency_s=20e-6)
    bank = se.ReadbackBank(hw, [0])
    got = []
    for k in range(300):
        v = np.array([[1.2, 0.5 + 1e-3 * k]])
        bank.tick(k, v, v / 50.0)
        s, new = bank.release(k + 1)
        if new["heater.readback_v"][0, 0]:
            got.append((k + 1, s["heater.readback_v"][0].copy(), s["heater.readback_i"][0].copy()))
    assert [g[0] for g in got] == [120, 220]
    _, cv, ci = got[0]
    assert cv[0] == round(1.2 / (2.0 / 4095)) and ci[0] == round(1.2 / 50 / (0.032 / 4095))
    assert cv[1] == round((0.5 + 1e-3 * 99) / (2.0 / 4095))     # track-and-hold of the last interval before 100


# ============================================================================================ reference tap
def test_reference_tap_has_independent_noise():
    """07 P1-9 reference tap: same AFE/ADC, own responsivity and dark current, own noise stream and slots."""
    hw = _hw(200, 1, adc_bits=24, input_noise_a_per_rt_hz=50e-12, shot_noise=True)
    ref = hw.reference_tap(responsivity_a_per_w=0.5, dark_i0_a=0.0)
    seeds = rng.batch_seeds(9, 200)
    main = se.MonitorBank(hw, seeds, m_max=1, pop_period_ticks=4)
    tap = se.MonitorBank(ref, seeds, m_max=1, pop_period_ticks=4, prefix="tx.monitor_ref",
                         valid_name="monitor_ref_valid", overrange_name="adc_ref_overrange")
    sb = se.SensingBank(main, ref_tap=tap)
    xs, xr = [], []
    for k in range(80):
        sb.tick(k, p_pd_w=P_MON, p_ref_w=20e-6, t_pd_k=T25)
        if (k + 1) % 4 == 0:
            s, _ = sb.release(k + 1)
            xs.append(s["tx.monitor.latest"][:, 0].astype(float))
            xr.append(s["tx.monitor_ref.latest"][:, 0].astype(float))
    xs, xr = np.array(xs[1:]).ravel(), np.array(xr[1:]).ravel()
    assert xr.mean() * hw.lsb_a[0] == pytest.approx(0.5 * 20e-6, rel=1e-4, abs=0)
    assert abs(np.corrcoef(xs, xr)[0, 1]) < 0.05


# ============================================================================================ params adapter
def test_from_plant_params_mini():
    """MonitorHw / TempSensorsHw build from realize() output of tests/data/mini.yaml."""
    from mrm.params import load, realize
    pp = realize(load(os.path.join(os.path.dirname(__file__), "..", "data", "mini.yaml")), batch=2, seeds=0)
    hw = se.MonitorHw.from_plant_params(pp)
    assert (hw.B, hw.N, hw.window_ticks, hw.period_ticks, hw.adc_bits) == (2, 4, 2, 4, 16)
    np.testing.assert_array_equal(hw.latency_ticks, pp.timing.monitor_latency_ticks.reshape(-1))
    np.testing.assert_allclose(hw.pr[:, :, se.PR_RESP], pp.monitor.responsivity_a_per_w)
    assert hw.pw[0, se.PW_DARK_EA] == pytest.approx(pp.monitor.dark_activation_j[0, 0], rel=1e-12, abs=0)
    codes, _, _ = _run(hw, 20, 15e-6)
    i_sig, i_d = hw.photocurrent_a(15e-6, 0.0, T25)
    np.testing.assert_allclose(codes[-1] * hw.lsb_a[:, None], i_sig + i_d, rtol=2e-3)
    ts = se.TempSensorsHw.from_plant_params(pp)
    assert ts.names == ("die0",) and ts.period_ticks[0] == 1000
    np.testing.assert_allclose(ts.tp[:, 0, 2], pp.temp_sensors.offset_k[:, 0])


# ============================================================================================ review additions
def test_rc_noise_sample_autocorrelation_is_exact_and_dt_invariant():
    """rc AFE: consecutive samples of the noise are an AR(1) sequence with coefficient exp(-P/tau) (the exact decay
    of the discrete Gramian), for any dt; the integrate-and-dump samples with W <= P are uncorrelated."""
    tau, P_s = 6e-6, 4e-6
    rho_want = math.exp(-P_s / tau)
    for dt in (0.5e-6, 2e-6):
        hw = _hw(256, 1, dt, adc_bits=24, afe_type="rc", afe_tau_s=tau, input_noise_a_per_rt_hz=60e-12)
        codes, _, _ = _run(hw, 60 * hw.period_ticks, 5e-6, seeds=21)
        x = codes[:, :, 0].astype(float)
        x -= x.mean()
        r = np.mean(x[1:] * x[:-1]) / np.mean(x * x)
        assert r == pytest.approx(rho_want, abs=0.03), dt
    hw = _hw(256, 1, 1e-6, adc_bits=24, input_noise_a_per_rt_hz=60e-12)
    codes, _, _ = _run(hw, 60 * hw.period_ticks, 5e-6, seeds=22)
    x = codes[:, :, 0].astype(float)
    x -= x.mean()
    assert abs(np.mean(x[1:] * x[:-1]) / np.mean(x * x)) < 0.03


def test_fault_flags_follow_the_reported_code():
    """A stuck converter reports its old code, so the clip state of the hidden conversion must not raise
    adc_overrange; a dropped sample carries neither a code nor an overrange flag (07 P1-10)."""
    hw = _hw(1, 2, adc_bits=16)
    bank = se.MonitorBank(hw, [0], m_max=1, pop_period_ticks=4)
    for k in range(4):                                          # a good in-range sample first (s = 4)
        bank.tick(k, 5e-6, 0.0, T25)
    flt = np.array([[se.FAULT_ADC_STUCK, se.FAULT_ADC_DROP]], np.int8)
    for k in range(4, 8):                                       # input far above full scale while faulted (s = 8)
        bank.tick(k, 80e-6, 0.0, T25, fault=flt)
        if k == 5:
            assert bank.release(4 + 2).count.all()             # s = 4 released at 4 + latency 2
    r = bank.release(8 + 2)
    assert not r.overrange.any() and not r.valid.any()
    assert r.count[0, 0] == 1 and r.count[0, 1] == 0
    assert r.samples[0, 0, 0] == round(5e-6 / hw.lsb_a[0])      # the frozen code, not full scale


def test_fault_mask_from_scenario_signals():
    """Scenario fault channels (floats, on at >= 0.5; dac glitch codes non-zero) map to the FAULT_* bits."""
    m = se.fault_mask(2, 3, adc_drop=[[1.0, 0.0, 0.4], [0.0, 0.0, 0.0]], adc_stuck=[0.0, 0.6, 0.0],
                      adc_saturate=[0.0, 1.0], dac_glitch_codes=[[0.0, 0.0, -3.0], [0.0, 0.0, 0.0]])
    want = np.array([[se.FAULT_ADC_DROP, se.FAULT_ADC_STUCK, se.FAULT_DAC_GLITCH],
                     [se.FAULT_ADC_SATURATE, se.FAULT_ADC_STUCK | se.FAULT_ADC_SATURATE, se.FAULT_ADC_SATURATE]])
    np.testing.assert_array_equal(m, want)
    assert m.dtype == np.int8


def test_default_block_size_holds_every_sample():
    """With m_max omitted the block holds ceil(release period / ADC period) samples, so nothing is lost."""
    hw = _hw(1, 1, adc_rate_hz=250e3)
    bank = se.MonitorBank(hw, [0], pop_period_ticks=10)
    assert bank.m_max == 3
    lost = 0
    for k in range(200):
        bank.tick(k, P_MON, 0.0, T25)
        if (k + 1) % 10 == 0:
            lost += bank.release(k + 1).lost
    assert lost == 0 and bank.fifo_lost == 0


def test_from_plant_params_device_seeds_drive_dnl_and_dark_activation_default():
    """MonitorHw.from_plant_params keys the DNL thresholds by pp.seeds['device'] (different devices get different
    thresholds, the same device the same, whatever the noise seed), and an absent dark_activation_ev takes the
    inferred 0.37 eV instead of realize's 0."""
    import dataclasses
    from mrm.params import load, realize
    pp = realize(load(os.path.join(os.path.dirname(__file__), "..", "data", "mini.yaml"),
                      overrides={"ring.monitor.adc_dnl_lsb": 0.9}), batch=2, seeds=0)
    hw = se.MonitorHw.from_plant_params(pp)
    np.testing.assert_array_equal(hw.device_seeds, pp.seeds["device"])
    a = se.MonitorBank(hw, [5, 5])
    b = se.MonitorBank(hw, [6, 6])
    np.testing.assert_array_equal(a.key_dnl, b.key_dnl)
    assert a.key_dnl[0] != a.key_dnl[1]
    v = np.arange(100.0, 300.0, 0.01)
    c0 = [se.adc_quantize(x, 0.0, 0.9, hw.code_max, a.key_dnl[0], 0, 0)[0] for x in v]
    c1 = [se.adc_quantize(x, 0.0, 0.9, hw.code_max, a.key_dnl[1], 0, 0)[0] for x in v]
    assert c0 != c1
    vals = {k: v for k, v in pp.values.items() if k != "ring.monitor.dark_activation_ev"}
    hw2 = se.MonitorHw.from_plant_params(dataclasses.replace(pp, values=vals))
    assert hw2.pw[0, se.PW_DARK_EA] == pytest.approx(0.37 * Q_E, rel=1e-12, abs=0)


def test_sensing_bank_merges_every_channel():
    """SensingBank.release returns the monitor, temperature and readback channels with their dtypes and flags."""
    mon = se.MonitorBank(_hw(2, 3), rng.batch_seeds(0, 2), pop_period_ticks=10)
    temps = se.TempSensorBank(se.TempSensorsHw.build(2, 1e-6, [dict(name="die0", sample_rate_hz=100e3,
                                                                    latency_s=0.0)]), [0, 1], pop_period_ticks=10)
    rb = se.ReadbackBank(se.ReadbackHw.build(2, 3, 1e-6, rate_hz=100e3, latency_s=0.0), [0, 1], pop_period_ticks=10)
    sb = se.SensingBank(mon, temps, rb)
    for k in range(20):
        sb.tick(k, p_pd_w=P_MON, t_pd_k=T25, t_site_k=np.full((2, 1), 320.0), heater_v=np.full((2, 3), 1.0),
                heater_i=np.full((2, 3), 0.02))
    s, new = sb.release(20)
    assert set(s) == set(new) == {"tx.monitor.samples", "tx.monitor.sample_ticks", "tx.monitor.count",
                                  "tx.monitor.latest", "monitor_valid", "adc_overrange", "temp.die0",
                                  "heater.readback_v", "heater.readback_i"}
    assert s["temp.die0"].shape == (2, 1) and s["temp.die0"][0, 0] == round((320.0 - 233.15) / 0.0625)
    assert s["heater.readback_v"][0, 0] == round(1.0 / (2.0 / 4095)) and new["heater.readback_i"].all()
    assert s["tx.monitor.count"].max() <= mon.m_max and s["tx.monitor.samples"].shape == (2, 3, mon.m_max)


def test_readback_bit_limit_and_from_plant_params():
    """The V and I codes share one int64 FIFO word, so more than 31 bits must be refused (a 32-bit current code
    would wrap negative); from_plant_params reads the realized rate, latency and full scales."""
    with pytest.raises(ValueError, match="1..31"):
        se.ReadbackHw.build(1, 1, 1e-6, adc_bits=32)
    hw = se.ReadbackHw.build(1, 1, 1e-6, adc_bits=31, i_full_scale_a=0.032, rate_hz=1e6, latency_s=0.0)
    bank = se.ReadbackBank(hw, [0])
    bank.tick(0, 1.99, 0.0319)
    s, _ = bank.release(1)
    assert s["heater.readback_i"][0, 0] == round(0.0319 / hw.lsb_i[0]) > 0
    from mrm.params import load, realize
    path = os.path.join(os.path.dirname(__file__), "..", "data", "mini.yaml")
    with pytest.raises(ValueError, match="present"):
        se.ReadbackHw.from_plant_params(realize(load(path), batch=1, seeds=0))
    pp = realize(load(path, overrides={"ring.heater.readback.present": True, "ring.heater.readback.adc_bits": 10,
                                       "ring.heater.readback.v_full_scale_v": 1.8,
                                       "ring.heater.readback.rate_hz": 20e3,
                                       "ring.heater.readback.latency_s": 30e-6}), batch=2, seeds=0)
    rb = se.ReadbackHw.from_plant_params(pp, adc_bits=10)
    assert rb.period_ticks == pp.timing.readback_period_ticks == 50 and rb.code_max == 1023
    np.testing.assert_array_equal(rb.latency_ticks, pp.timing.readback_latency_ticks.reshape(-1))
    np.testing.assert_allclose(rb.lsb_v, 1.8 / 1023)


def test_level_codes_sampler_noise_override():
    """A bit-rate sampler has its own front-end noise density: the override replaces the monitor's S_tia."""
    hw = _hw(1, 1, adc_bits=16, input_noise_a_per_rt_hz=1e-12, shot_noise=False)
    bank = se.MonitorBank(hw, [0])
    tau = 1 / (2 * math.pi * 16e9)
    _, _, s_mon, _ = bank.level_codes(20e-6, 5e-6, tau_s=tau)
    _, _, s_ovr, _ = bank.level_codes(20e-6, 5e-6, tau_s=tau, noise_a_per_rt_hz=20e-12)
    assert s_mon[0, 0] == pytest.approx(math.sqrt(1e-24 / (4 * tau)) / hw.lsb_a[0])
    assert s_ovr[0, 0] == pytest.approx(20 * s_mon[0, 0])
    # laser RIN over the sampler bandwidth: sigma^2 += (R P)^2 RIN / (4 tau)
    _, _, s_rin, s0_rin = bank.level_codes(20e-6, 5e-6, tau_s=tau, rin_per_hz=1e-14)
    assert s_rin[0, 0] == pytest.approx(math.sqrt((1e-24 + (20e-6) ** 2 * 1e-14) / (4 * tau)) / hw.lsb_a[0])
    assert s0_rin[0, 0] == pytest.approx(math.sqrt((1e-24 + (5e-6) ** 2 * 1e-14) / (4 * tau)) / hw.lsb_a[0])


def test_codes_per_w_is_the_small_signal_gain():
    """codes_per_w (07 P1-9 identifiable monitor_codes_per_uw_at_drop x 1e6) equals the slope of the kernel's
    noiseless codes against PD power, PGA and gain error included."""
    hw = _hw(1, 2, adc_bits=24, gain_scale=1.3, responsivity_a_per_w=[0.9, 1.0], adc_gain_error_frac=[0.0, 0.02],
             pga_gains=[1.0, 2.0])
    k = hw.codes_per_w(pga_select=1)
    np.testing.assert_allclose(k[0], np.array([1.3 * 0.9, 1.3 * 1.0 * 1.02]) * 2.0 / hw.lsb_a[0], rtol=1e-12)
    c = []
    for p in (2e-6, 6e-6):
        bank = se.MonitorBank(hw, [0], pop_period_ticks=4)
        for t in range(8):
            bank.tick(t, p, 0.0, T25, pga_select=1)
        c.append(bank.release(8).latest[0].astype(float))
    np.testing.assert_allclose((c[1] - c[0]) / 4e-6, k[0], rtol=1e-5)


def test_flicker_corner_below_the_band_is_off():
    """A corner so low that 10 f_c <= f_min leaves no pole in the band (no exception, no flicker rows)."""
    c = se.flicker_coefficients(1e-24, 5e-5, afe_type=se.AFE_INTEGRATE_DUMP, period_s=4e-6, window_s=2e-6)
    assert c.shape == (0, 10)
    hw = _hw(1, 1, input_noise_a_per_rt_hz=1e-12, flicker_corner_hz=5e-5)
    assert hw.fl_coef.shape[1] == 0


# ============================================================================================ NUMBA_DISABLE_JIT
_NOJIT_SCRIPT = r"""
import numpy as np
from mrm.core import rng
from mrm.plant import sensing as se
hw = se.MonitorHw.build(2, 3, 1e-6, responsivity_a_per_w=1.0, adc_full_scale_a=40e-6, adc_bits=20,
                        input_noise_a_per_rt_hz=30e-12, dark_i0_a=5e-9, flicker_corner_hz=1e5, adc_dnl_lsb=0.4,
                        adc_inl_bow_lsb=1.5)
bank = se.MonitorBank(hw, rng.batch_seeds(3, 2), device_seeds=[1, 2], m_max=1, pop_period_ticks=4)
out = []
for k in range(64):
    flt = se.fault_mask(2, 3, adc_stuck=1.0 if 20 <= k < 28 else 0.0, adc_drop=[0.0, 0.0, 1.0 if k > 50 else 0.0])
    bank.tick(k, 15e-6 + 1e-7 * k, 0.0, 300.0, fault=flt)
    if (k + 1) % 4 == 0:
        r = bank.release(k + 1)
        out.append(np.concatenate([r.samples[:, :, 0], r.valid, r.overrange], axis=1).astype(float))
mu1, mu0, s1, s0 = bank.level_codes(20e-6, 5e-6, tau_s=1e-11, noise_a_per_rt_hz=15e-12, rin_per_hz=1e-14)
out.append(np.concatenate([mu1, mu0, s1, s0], axis=1)[:, :3])
rc = se.MonitorHw.build(2, 3, 1e-6, responsivity_a_per_w=1.0, adc_full_scale_a=40e-6, adc_bits=20, afe_type="rc",
                        afe_tau_s=3e-6, input_noise_a_per_rt_hz=30e-12, dark_i0_a=5e-9, flicker_corner_hz=1e5)
bank = se.MonitorBank(rc, rng.batch_seeds(3, 2), m_max=1, pop_period_ticks=4)
for k in range(32):
    bank.tick(k, 15e-6, 0.0, 300.0)
    if (k + 1) % 4 == 0:
        out.append(bank.release(k + 1).samples[:, :, 0].astype(float))
np.save(OUT, np.concatenate([np.ravel(o) for o in out]))
"""


def test_kernels_identical_without_jit(tmp_path):
    """The kernels give the same codes with numba and under NUMBA_DISABLE_JIT=1 (ARCHITECTURE kernel rule)."""
    res = {}
    for mode in ("0", "1"):
        out = tmp_path / f"out{mode}.npy"
        env = dict(os.environ, NUMBA_DISABLE_JIT=mode)
        subprocess.run([sys.executable, "-c", f"OUT = {str(out)!r}\n" + _NOJIT_SCRIPT], env=env, check=True,
                       timeout=120)
        res[mode] = np.load(out)
    np.testing.assert_array_equal(res["0"], res["1"])
