"""Comprehensive unit and E2E tests for TLC Block B3 Lock-In Demodulator.

Tests cover:
- Sliding 4 ms Hann-windowed I/Q demodulation and buffer priming / ready flag
- Recovery of known fundamental (y1) amplitude within <= 2% error
- Recovery of known second harmonic (y2) amplitude and ratio zeta = y2 / y1 within <= 2% error
- Optical power invariance: scaling input power scales y1 and y2 equally, leaving zeta invariant
- Red-flank inflection point zero-crossing: y2 = 0, zeta = 0
- Cross-tone rejection > 40 dB in multi-channel environment
- Quadrature phase robustness
- Zero-division safety when y1 -> 0
- Vectorization across batch dimension B and channel count N=8
"""
import numpy as np
import pytest

try:
    from mrm.control.tlc.b3_lockin import LockInConfig, LockInDemodulator
except ImportError:
    try:
        from mrm.control.tlc import LockInConfig, LockInDemodulator
    except ImportError:
        try:
            from mrm.control.tlc.b3_lockin import B3LockIn as LockInDemodulator
            LockInConfig = None
        except ImportError:
            LockInConfig = None
            LockInDemodulator = None

requires_b3 = pytest.mark.skipif(
    LockInDemodulator is None,
    reason="mrm.control.tlc.b3_lockin (LockInDemodulator / B3LockIn) is not yet implemented",
)

FREQUENCIES_HZ = [1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0]
WINDOW_DURATION_S = 0.004  # 4 ms
SAMPLE_RATE_HZ = 10000.0   # 10 kHz
ADC_RATE_HZ = 250000.0     # 250 kS/s plant ADC rate
SAMPLES_PER_STEP = 25      # M = 25 at 100 us step
WINDOW_STEPS = 40          # 40 steps at 100 us
N_RINGS = 8


def _instantiate_lockin(n_rings=8, **kwargs):
    """Helper to instantiate LockInDemodulator across supported signatures."""
    if LockInConfig is not None:
        try:
            cfg = LockInConfig(**kwargs)
            return LockInDemodulator(n_rings=n_rings, config=cfg)
        except (TypeError, ValueError):
            pass
    try:
        return LockInDemodulator(n_rings=n_rings, dither_frequencies=FREQUENCIES_HZ, **kwargs)
    except TypeError:
        return LockInDemodulator(n_rings=n_rings)


def _step_lockin(demod, monitor_dc, dither_detuning=None, step_idx=0, ts=1e-4):
    """Helper to step demodulator across supported interfaces."""
    if hasattr(demod, "step"):
        if dither_detuning is None:
            dither_detuning = np.zeros_like(monitor_dc)
        return demod.step(monitor_dc, dither_detuning)
    if hasattr(demod, "push") and hasattr(demod, "demod"):
        t_now = step_idx * ts
        B, N = monitor_dc.shape
        invalid = np.zeros((B, N), dtype=bool)
        demod.push(monitor_dc, t_now, invalid)
        y0, y1, y2, zeta, valid = demod.demod()
        return valid, y1, y2, zeta
    raise AttributeError("Demodulator has neither step nor push/demod")


# ==============================================================================
# Tier 1: Sliding Hann Window & Harmonic Extraction Accuracy
# ==============================================================================

@requires_b3
def test_b3_default_configuration():
    """Verify default lock-in configuration parameters."""
    if LockInConfig is not None:
        cfg = LockInConfig()
        assert cfg.sample_rate_hz == pytest.approx(SAMPLE_RATE_HZ)
        assert cfg.window_length_s == pytest.approx(WINDOW_DURATION_S)
        assert cfg.harmonic_order >= 2


@requires_b3
def test_b3_window_priming_and_ready_flag():
    """Verify ready flag remains False during initial priming and becomes True when buffer fills."""
    demod = _instantiate_lockin(n_rings=N_RINGS)
    if hasattr(demod, "reset"):
        demod.reset(batch_size=1)

    dc_val = np.full((1, N_RINGS), 25000.0)

    # First few steps (< 30 steps) should definitely not be ready
    for step in range(30):
        ready, y1, y2, zeta = _step_lockin(demod, dc_val, step_idx=step)
        if isinstance(ready, np.ndarray):
            assert not np.all(ready), f"Buffer prematurely marked ready at step {step}"
        else:
            assert not ready, f"Buffer prematurely marked ready at step {step}"

    # Feed enough steps to complete the window (> 40-60 steps)
    ready_achieved = False
    for step in range(30, 80):
        ready, y1, y2, zeta = _step_lockin(demod, dc_val, step_idx=step)
        if (isinstance(ready, np.ndarray) and np.all(ready)) or (isinstance(ready, (bool, np.bool_)) and ready):
            ready_achieved = True
            break

    assert ready_achieved, "Lock-in demodulator must assert ready flag once window duration elapses"


@requires_b3
def test_b3_fundamental_y1_recovery_accuracy():
    """Verify fundamental harmonic amplitude y1 is recovered within <= 2% error."""
    demod = _instantiate_lockin(n_rings=N_RINGS)
    if hasattr(demod, "reset"):
        demod.reset(batch_size=1)

    y0 = 30000.0
    known_amp_y1 = 50.0  # Known dither fundamental amplitude
    ts = 1e-4
    dt = ts / SAMPLES_PER_STEP

    # Run for 100 steps (10 ms = 2.5 windows) to settle sliding Hann window
    for step in range(100):
        t = step * ts + np.arange(SAMPLES_PER_STEP) * dt
        # Signal: y0 + known_amp_y1 * cos(2*pi*f_i*t)
        tones = known_amp_y1 * np.cos(2 * np.pi * np.array(FREQUENCIES_HZ)[:, None] * t[None, :])
        dc_sample = (y0 + tones)[None, :, :]
        ready, y1, y2, zeta = _step_lockin(demod, dc_sample, step_idx=step, ts=ts)

    # Check recovered amplitude
    y1_recovered = np.asarray(y1).reshape(-1, N_RINGS)[0]
    # Check error <= 2% across all 8 channels
    for ch in range(N_RINGS):
        err_pct = np.abs(np.abs(y1_recovered[ch]) - known_amp_y1) / known_amp_y1 * 100.0
        assert err_pct <= 2.0, (
            f"Channel {ch} ({FREQUENCIES_HZ[ch]} Hz): y1 error {err_pct:.2f}% exceeds 2% limit "
            f"(recovered={y1_recovered[ch]:.3f}, expected={known_amp_y1})"
        )


@requires_b3
def test_b3_second_harmonic_and_ratio_zeta_accuracy():
    """Verify recovery of second harmonic y2 and ratio zeta = y2 / y1 within <= 2% error."""
    demod = _instantiate_lockin(n_rings=N_RINGS)
    if hasattr(demod, "reset"):
        demod.reset(batch_size=1)

    y0 = 30000.0
    known_y1 = 60.0
    known_zeta = 0.25  # Target ratio
    known_y2 = known_y1 * known_zeta  # 15.0
    ts = 1e-4
    dt = ts / SAMPLES_PER_STEP

    for step in range(100):
        t = step * ts + np.arange(SAMPLES_PER_STEP) * dt
        f = np.array(FREQUENCIES_HZ)[:, None]
        signal = y0 + known_y1 * np.cos(2 * np.pi * f * t[None, :]) + known_y2 * np.cos(4 * np.pi * f * t[None, :])
        dc_sample = signal[None, :, :]
        ready, y1, y2, zeta = _step_lockin(demod, dc_sample, step_idx=step, ts=ts)

    zeta_recovered = np.asarray(zeta).reshape(-1, N_RINGS)[0]
    for ch in range(N_RINGS):
        rel_err = np.abs(np.abs(zeta_recovered[ch]) - known_zeta) / known_zeta * 100.0
        assert rel_err <= 2.0, (
            f"Channel {ch}: zeta error {rel_err:.2f}% exceeds 2% limit "
            f"(recovered={zeta_recovered[ch]:.4f}, expected={known_zeta})"
        )


# ==============================================================================
# Tier 2: Invariance, Inflection Zero-Crossing & Quadrature Tests
# ==============================================================================

@requires_b3
def test_b3_optical_power_invariance():
    """Verify ratio zeta = y2 / y1 is strictly invariant to input laser power scaling."""
    power_scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    recovered_zetas = []

    y0_base = 20000.0
    y1_base = 50.0
    y2_base = 12.5  # true zeta = 0.25
    ts = 1e-4

    for scale in power_scales:
        demod = _instantiate_lockin(n_rings=N_RINGS)
        if hasattr(demod, "reset"):
            demod.reset(batch_size=1)

        y0 = y0_base * scale
        known_y1 = y1_base * scale
        known_y2 = y2_base * scale
        dt = ts / SAMPLES_PER_STEP

        for step in range(100):
            t = step * ts + np.arange(SAMPLES_PER_STEP) * dt
            f = np.array(FREQUENCIES_HZ)[:, None]
            signal = y0 + known_y1 * np.cos(2 * np.pi * f * t[None, :]) + known_y2 * np.cos(4 * np.pi * f * t[None, :])
            ready, y1, y2, zeta = _step_lockin(demod, signal[None, :, :], step_idx=step, ts=ts)

        recovered_zetas.append(np.asarray(zeta).reshape(-1, N_RINGS)[0])

    # Compare zetas across all power scales against baseline (scale=1.0)
    baseline_zeta = recovered_zetas[2]
    for i, scale in enumerate(power_scales):
        np.testing.assert_allclose(
            recovered_zetas[i],
            baseline_zeta,
            rtol=0.005,
            err_msg=f"Zeta violated power invariance at power scale {scale}",
        )


@requires_b3
def test_b3_inflection_point_zero_crossing():
    """Verify that at the inflection point where y2 = 0, demodulator recovers zeta = 0."""
    demod = _instantiate_lockin(n_rings=N_RINGS)
    if hasattr(demod, "reset"):
        demod.reset(batch_size=1)

    y0 = 30000.0
    y1 = 50.0
    y2 = 0.0  # Zero curvature at inflection point
    ts = 1e-4
    dt = ts / SAMPLES_PER_STEP

    for step in range(100):
        t = step * ts + np.arange(SAMPLES_PER_STEP) * dt
        f = np.array(FREQUENCIES_HZ)[:, None]
        signal = y0 + y1 * np.cos(2 * np.pi * f * t[None, :])
        ready, y1_out, y2_out, zeta_out = _step_lockin(demod, signal[None, :, :], step_idx=step, ts=ts)

    zeta_recovered = np.asarray(zeta_out).reshape(-1, N_RINGS)[0]
    np.testing.assert_allclose(zeta_recovered, 0.0, atol=1e-3)


@requires_b3
def test_b3_quadrature_phase_invariance():
    """Verify amplitude recovery is robust across arbitrary phase lags."""
    phase_lags = [0.0, np.pi / 6, np.pi / 4, np.pi / 2, np.pi]
    y1_known = 40.0
    ts = 1e-4
    dt = ts / SAMPLES_PER_STEP

    for phi in phase_lags:
        demod = _instantiate_lockin(n_rings=N_RINGS)
        if hasattr(demod, "reset"):
            demod.reset(batch_size=1)

        for step in range(100):
            t = step * ts + np.arange(SAMPLES_PER_STEP) * dt
            f = np.array(FREQUENCIES_HZ)[:, None]
            signal = 25000.0 + y1_known * np.cos(2 * np.pi * f * t[None, :] + phi)
            ready, y1_out, y2_out, zeta_out = _step_lockin(demod, signal[None, :, :], step_idx=step, ts=ts)

        recovered = np.asarray(y1_out).reshape(-1, N_RINGS)[0]
        # Quadrature magnitude or phase projection should recover y1 within 2%
        np.testing.assert_allclose(np.abs(recovered), y1_known, rtol=0.02)


# ==============================================================================
# Tier 3: Multi-Channel Isolation & Zero Division Safety
# ==============================================================================

@requires_b3
def test_b3_cross_channel_isolation_gt_40db():
    """Verify each channel rejects other channels' dither tones by > 40 dB."""
    demod = _instantiate_lockin(n_rings=N_RINGS)
    if hasattr(demod, "reset"):
        demod.reset(batch_size=1)

    # Inject only channel 0's tone (1500 Hz) into ALL channels
    f0 = FREQUENCIES_HZ[0]
    y1_in = 50.0
    ts = 1e-4

    for step in range(100):
        t = step * ts
        # All rings receive channel 0's tone
        signal = np.full((1, N_RINGS), 25000.0 + y1_in * np.cos(2 * np.pi * f0 * t))
        ready, y1_out, y2_out, zeta_out = _step_lockin(demod, signal, step_idx=step, ts=ts)

    y1_recovered = np.abs(np.asarray(y1_out).reshape(-1, N_RINGS)[0])

    # Channel 0 should see ~50.0
    assert y1_recovered[0] == pytest.approx(y1_in, rel=0.02)

    # Channels 1 through 7 should reject channel 0 by > 40 dB (leakage < 0.5)
    for ch in range(1, N_RINGS):
        rejection_db = 20.0 * np.log10(y1_recovered[0] / max(y1_recovered[ch], 1e-12))
        assert rejection_db > 40.0, (
            f"Channel {ch} cross-talk rejection from channel 0 is only {rejection_db:.1f} dB <= 40 dB"
        )


@requires_b3
def test_b3_zero_division_guard():
    """Verify demodulator guards against division by zero when y1 -> 0 without crashing."""
    demod = _instantiate_lockin(n_rings=N_RINGS)
    if hasattr(demod, "reset"):
        demod.reset(batch_size=1)

    # Pure DC, zero dither amplitude => y1 = 0
    signal_flat = np.full((1, N_RINGS), 25000.0)
    for step in range(80):
        ready, y1_out, y2_out, zeta_out = _step_lockin(demod, signal_flat, step_idx=step)

    # Output should not contain NaN or Inf
    assert not np.any(np.isnan(zeta_out)), "Zeta must not be NaN when y1 -> 0"
    assert not np.any(np.isinf(zeta_out)), "Zeta must not be Inf when y1 -> 0"


@requires_b3
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_b3_vectorized_batch_execution(batch_size):
    """Verify lock-in demodulator supports arbitrary batch sizes (B, N)."""
    demod = _instantiate_lockin(n_rings=N_RINGS)
    if hasattr(demod, "reset"):
        demod.reset(batch_size=batch_size)

    dc_sample = np.full((batch_size, N_RINGS), 28000.0)
    ready, y1, y2, zeta = _step_lockin(demod, dc_sample)

    assert y1.shape == (batch_size, N_RINGS)
    assert y2.shape == (batch_size, N_RINGS)
    assert zeta.shape == (batch_size, N_RINGS)
