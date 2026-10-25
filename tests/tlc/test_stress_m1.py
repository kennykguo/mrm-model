"""Adversarial and empirical stress test suite for Block B1 and Block B2.

Challenger test suite verifying:
1. Block B1 Dither Generator:
   - Full 8-channel orthogonality (pairwise cross-correlation < 1e-12)
   - Frequency accuracy, SFDR (> 100 dB), full 50.0 pm peak-to-peak / 25.0 pm nominal detuning unattenuated across all channels
   - Long sequence phase continuity across integer periods (100 ms and beyond)
   - Batch dimension handling (B=1, B=4, B=16)
2. Block B2 Signal Conditioner:
   - Step 1 squelch tripping on ADC rail codes (0, 65535), flatline/stuck codes (std < 0.5 LSB), and non-finite values (NaN, +inf, -inf)
   - Squelch activation on hardware faults (monitor_valid=False, adc_overrange=True)
   - Squelch release on valid active noise signals (with 2-stage hysteresis)
   - 1D count array broadcasting across batched inputs (B > 1)
   - FIR noise gain (~0.52) and stopband attenuation (> 50 dB)
"""

import numpy as np
import pytest

from mrm.control.tlc.b1_dither import DitherConfig, DitherGenerator
from mrm.control.tlc.b2_conditioner import ConditionerConfig, SignalConditioner


# ==============================================================================
# Block B1 Stress Tests
# ==============================================================================

def test_b1_empirical_orthogonality_precision():
    """Verify full 8-channel pairwise normalized cross-correlation is < 1e-12."""
    gen = DitherGenerator(n_rings=8)
    gen.reset(batch_size=1)

    # 40 samples (4 ms window at 10 kHz)
    samples = []
    for _ in range(40):
        samples.append(gen.step()[0])
    samples = np.array(samples)  # (40, 8)

    # Normalized cross-correlation matrix rho_ij = |R_ij| / sqrt(R_ii * R_jj)
    R = samples.T @ samples
    diag = np.diag(R)
    norm = np.sqrt(np.outer(diag, diag))
    rho = np.abs(R) / norm
    np.fill_diagonal(rho, 0.0)

    max_cross = np.max(rho)
    assert max_cross < 1e-12, f"Cross-correlation {max_cross:.2e} exceeds 1e-12 threshold"


def test_b1_frequency_accuracy_and_sfdr():
    """Verify frequency accuracy and SFDR > 100 dB across all 8 channels."""
    gen = DitherGenerator(n_rings=8)
    gen.reset(batch_size=1)

    n_samples = 200  # 5 periods = 20 ms
    samples = []
    for _ in range(n_samples):
        samples.append(gen.step()[0])
    samples = np.array(samples)  # (200, 8)

    freqs = np.fft.rfftfreq(n_samples, d=1.0 / gen.config.sample_rate_hz)

    for ch in range(8):
        f_target = gen.config.frequencies_hz[ch]
        x = samples[:, ch]
        spectrum = np.abs(np.fft.rfft(x))
        peak_idx = np.argmax(spectrum)
        assert freqs[peak_idx] == pytest.approx(f_target, abs=1e-3), (
            f"Channel {ch} frequency mismatch: expected {f_target} Hz, got {freqs[peak_idx]} Hz"
        )

        peak_mag = spectrum[peak_idx]
        mask = np.ones(len(spectrum), dtype=bool)
        mask[peak_idx] = False
        spurious_mag = np.max(spectrum[mask])
        sfdr_db = 20.0 * np.log10(peak_mag / max(spurious_mag, 1e-18))

        assert sfdr_db > 100.0, f"Channel {ch} SFDR {sfdr_db:.2f} dB is below 100 dB"


def test_b1_unattenuated_channel_amplitudes():
    """Verify recovered detuning amplitude is exactly nominal 25.0 pm (50.0 pm p-p) across all channels."""
    gen = DitherGenerator(n_rings=8)
    gen.reset(batch_size=1)

    n_samples = 40
    samples = []
    for _ in range(n_samples):
        samples.append(gen.step()[0])
    samples = np.array(samples)

    # Coherent DFT demodulation over exact integer cycles
    k = np.arange(n_samples)
    for ch in range(8):
        f = gen.config.frequencies_hz[ch]
        m = int(round(f * gen.config.window_length_s))
        ref_cos = np.cos(2 * np.pi * m * k / n_samples)
        ref_sin = np.sin(2 * np.pi * m * k / n_samples)
        I = np.sum(samples[:, ch] * ref_cos)
        Q = np.sum(samples[:, ch] * ref_sin)

        if m == n_samples // 2:
            recovered_amp = I / n_samples
        else:
            recovered_amp = 2.0 * np.sqrt(I ** 2 + Q ** 2) / n_samples

        assert recovered_amp == pytest.approx(25.0, abs=1e-9), (
            f"Channel {ch} ({f} Hz): recovered amplitude {recovered_amp:.6f} pm != 25.0 pm"
        )


def test_b1_long_sequence_phase_continuity():
    """Verify long-term phase continuity and zero drift over 100 ms (1000 steps)."""
    gen = DitherGenerator(n_rings=8)
    gen.reset(batch_size=1)

    outputs = [gen.step()[0] for _ in range(1001)]
    outputs = np.array(outputs)

    # Every 4 ms window (40 steps) must return exactly to t=0 state
    initial = outputs[0]
    for w in range(1, 26):
        step_k = w * 40
        diff = np.max(np.abs(outputs[step_k] - initial))
        assert diff < 1e-9, f"Phase drift {diff:.2e} pm detected at window {w} (step {step_k})"


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_b1_batch_dimension_handling(batch_size):
    """Verify B1 operates cleanly across batch dimensions B=1, 4, 16."""
    gen = DitherGenerator(n_rings=8)
    gen.reset(batch_size=batch_size)

    # Step without sigma_delta
    out_default = gen.step()
    assert out_default.shape == (batch_size, 8)

    # Step with 2D sigma_delta (B, 8)
    sigma_2d = np.full((batch_size, 8), 20.0)
    out_2d = gen.step(sigma_delta=sigma_2d)
    assert out_2d.shape == (batch_size, 8)

    # Step with 1D sigma_delta (8,)
    sigma_1d = np.full(8, 30.0)
    out_1d = gen.step(sigma_delta=sigma_1d)
    assert out_1d.shape == (batch_size, 8)

    # Step with scalar sigma_delta
    out_scalar = gen.step(sigma_delta=15.0)
    assert out_scalar.shape == (batch_size, 8)


# ==============================================================================
# Block B2 Stress Tests
# ==============================================================================

@pytest.mark.parametrize("fault_desc, samples_func", [
    ("Upper rail 65535", lambda: np.full((1, 8, 25), 65535.0)),
    ("Lower rail 0", lambda: np.full((1, 8, 25), 0.0)),
    ("Flatline stuck code 32000", lambda: np.full((1, 8, 25), 32000.0)),
    ("Sub-threshold noise std < 0.5 LSB", lambda: 32000.0 + np.random.default_rng(10).normal(0, 0.35, (1, 8, 25))),
    ("All NaN buffer", lambda: np.full((1, 8, 25), np.nan)),
    ("Positive Infinity buffer", lambda: np.full((1, 8, 25), np.inf)),
    ("Negative Infinity buffer", lambda: np.full((1, 8, 25), -np.inf)),
])
def test_b2_step1_squelch_tripping(fault_desc, samples_func):
    """Verify squelch trips on Step 1 for rail codes, stuck codes, and non-finite values."""
    cond = SignalConditioner(n_rings=8)
    samples = samples_func()
    dc, sq, nz = cond.process_samples(samples)

    assert np.all(sq), f"Step 1 squelch failed to trip on {fault_desc}"
    assert np.all(np.isfinite(dc)), f"Filtered DC contains non-finite values on {fault_desc}"


def test_b2_hardware_fault_activation():
    """Verify hardware fault masks (monitor_valid=False, adc_overrange=True) trip squelch immediately."""
    cond = SignalConditioner(n_rings=8)
    rng = np.random.default_rng(20)
    valid_samples = 32000.0 + rng.normal(0, 30.0, (1, 8, 25))

    # Channel 2 sensor invalid
    mv = np.ones((1, 8), dtype=bool)
    mv[0, 2] = False
    _, sq_mv, _ = cond.process_samples(valid_samples, monitor_valid=mv)
    assert sq_mv[0, 2] == True, "Channel 2 failed to trip on monitor_valid=False"
    assert not np.any(sq_mv[0, [0, 1, 3, 4, 5, 6, 7]]), "Spurious squelch asserted on valid channels"

    # Channel 5 ADC overrange
    ao = np.zeros((1, 8), dtype=bool)
    ao[0, 5] = True
    _, sq_ao, _ = cond.process_samples(valid_samples, adc_overrange=ao)
    assert sq_ao[0, 5] == True, "Channel 5 failed to trip on adc_overrange=True"
    assert not np.any(sq_ao[0, [0, 1, 2, 3, 4, 6, 7]]), "Spurious squelch asserted on normal channels"


def test_b2_squelch_release_on_active_noise():
    """Verify squelch releases after 2 consecutive steps of valid active noise following squelch."""
    cond = SignalConditioner(n_rings=8)
    rng = np.random.default_rng(30)

    # 1. Establish high noise baseline (active modulation std = 60.0)
    for _ in range(10):
        cond.process_samples(30000.0 + rng.normal(0, 60.0, (1, 8, 25)))
    assert not np.any(cond.squelch_active), "Should be unsquelched during initial active modulation"

    # 2. Quiet period (20 dB drop in noise, std = 6.0) -> squelch trips
    for _ in range(5):
        cond.process_samples(30000.0 + rng.normal(0, 6.0, (1, 8, 25)))
    assert np.all(cond.squelch_active), "Squelch must trip after quiet period"

    # 3. Active modulation resumes (std = 60.0)
    # Step 1: Hysteresis holds squelch active
    _, sq_step1, _ = cond.process_samples(30000.0 + rng.normal(0, 60.0, (1, 8, 25)))
    assert np.all(sq_step1), "Hysteresis must keep squelch active on first loud step"

    # Step 2: Squelch releases
    _, sq_step2, _ = cond.process_samples(30000.0 + rng.normal(0, 60.0, (1, 8, 25)))
    assert not np.any(sq_step2), "Squelch must release on second consecutive loud step"


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
def test_b2_count_broadcasting_batched(batch_size):
    """Verify 1D and 2D count array broadcasting across batch sizes B >= 1."""
    cond = SignalConditioner(n_rings=8)
    rng = np.random.default_rng(batch_size)
    samples = 30000.0 + rng.normal(0, 40.0, (batch_size, 8, 25))

    # 1D count of shape (8,) [per-ring]
    c_ring = np.array([18, 19, 20, 21, 22, 23, 24, 25])
    dc1, sq1, nz1 = cond.process_samples(samples, count=c_ring)
    assert dc1.shape == (batch_size, 8)

    # 1D count of shape (B,) [per-batch]
    c_batch = np.full(batch_size, 22)
    dc2, sq2, nz2 = cond.process_samples(samples, count=c_batch)
    assert dc2.shape == (batch_size, 8)

    # Scalar count
    dc3, sq3, nz3 = cond.process_samples(samples, count=20)
    assert dc3.shape == (batch_size, 8)


def test_b2_fir_noise_gain_and_stopband_attenuation():
    """Verify FIR filter noise gain is ~0.52 and stopband attenuation > 50 dB."""
    cond = SignalConditioner(n_rings=8)

    # 1. Theoretical noise gain
    noise_gain = cond.noise_gain
    assert noise_gain == pytest.approx(0.5227, abs=0.01)

    # 2. Empirical noise gain with white Gaussian noise
    rng = np.random.default_rng(42)
    total_blocks = 100
    M = 25
    white_noise = rng.normal(0, 100.0, (1, 8, total_blocks * M))
    cond.reset(batch_size=1)
    filtered = []
    for b in range(total_blocks):
        cond.process_samples(30000.0 + white_noise[:, :, b * M : (b + 1) * M])
        filtered.append(cond.filtered_samples)
    filt_arr = np.concatenate(filtered, axis=-1)[0, 0, 50:]
    empirical_gain = np.std(filt_arr) / 100.0
    assert empirical_gain == pytest.approx(noise_gain, rel=0.05)

    # 3. Stopband attenuation > 50 dB above 60 kHz
    fs = cond.config.adc_rate_hz
    stopband_freqs = [60000.0, 75000.0, 80000.0, 100000.0, 115000.0]
    for f in stopband_freqs:
        cond.reset(batch_size=1)
        t = np.arange(2500) / fs
        sig_in = 5000.0 * np.sin(2.0 * np.pi * f * t)
        out_filt = []
        for b in range(100):
            blk = (30000.0 + sig_in[b * M : (b + 1) * M])[None, None, :].repeat(8, axis=1)
            cond.process_samples(blk)
            out_filt.append(cond.filtered_samples[0, 0])
        settled_out = np.concatenate(out_filt)[200:]
        out_amp = np.sqrt(2.0) * np.std(settled_out)
        gain_db = 20.0 * np.log10(out_amp / 5000.0)
        attenuation_db = -gain_db
        assert attenuation_db > 50.0, f"Stopband attenuation at {f} Hz is {attenuation_db:.2f} dB <= 50 dB"
