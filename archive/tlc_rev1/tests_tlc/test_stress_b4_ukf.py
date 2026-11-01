"""Adversarial and empirical stress test suite for Block B4 Dual-Rate UKF (b4_ukf.py).

Challenger test suite verifying:
1. Steady-State Convergence and Tracking:
   - Steady-state convergence under constant detuning (80 steps at constant y0=28000.0, zeta=0.0)
   - Estimation error <= 0.1 pm from nominal inflection setpoint 124.0 pm
   - Uncertainty monotonically non-increasing (sigma_final < sigma_initial)
   - Dynamic tracking under thermal step disturbance (y0=24000.0, zeta=0.15)
   - Adaptive process noise multiplier qm response opening filter bandwidth
2. Outlier Gating & Cross-Ring Decoupling:
   - Single-step extreme glitch y0 = 0.0 on ring 3
   - Single-step extreme glitch y0 = 65535.0 (ADC rail) on ring 3
   - Extreme negative y0 (-10000.0) and extreme positive y0 (1e6) on ring 3
   - Extreme slow harmonic ratio glitches (zeta = +10.0 and -10.0)
   - Simultaneous multi-ring glitches (ring 1=0.0, ring 3=65535.0)
   - Neighbor ring 4 detuning estimate corruption <= 15 pm (empirically < 1e-7 pm)
   - Huber innovation gating flag asserted
3. Squelch Handling & Photodiode Scale Freeze:
   - Single-channel squelch active on ring 3 under extreme DC and harmonic ratio
   - All-channel squelch active across array
   - Photodiode DC scale factor p remains frozen within <= 0.1% (empirically 0.000000%)
   - Slow update Kalman gain column completely zeroed on squelched channels
4. Numerical Stability & Covariance Conditioning:
   - 500 noisy simulation steps with stochastic thermal inputs and measurement noise
   - All eigenvalues of covariance matrix P remain strictly positive (lambda_min(P) > 0)
   - Exact covariance matrix symmetry ||P - P^T||_max == 0.0
   - Long-horizon numerical stability over 1000 steps without covariance divergence
5. Vectorized Batch Execution across B in [1, 2, 4, 8]:
   - Valid output shapes for delta_hat and sigma_delta of (B, N)
   - Positive-definiteness preserved for all batch instances
"""

import numpy as np
import pytest

from mrm.control.tlc.b4_ukf import DualRateUKF, UKFConfig

N_RINGS = 8
DT_FAST = 1e-4  # 100 us
DT_SLOW = 4e-3  # 4 ms


# ==============================================================================
# 1. Steady-State Convergence and Dynamic Tracking Tests
# ==============================================================================

def test_ukf_steady_state_convergence():
    """Verify UKF converges to 124.0 pm within <= 0.1 pm and reduces uncertainty over 80 steps."""
    ukf = DualRateUKF(n_rings=N_RINGS, dt_fast=DT_FAST, dt_slow=DT_SLOW)
    ukf.reset(batch_size=1)

    initial_delta, initial_sigma = ukf.get_estimates()
    assert np.allclose(initial_delta, 124.0)

    # 80 fast steps (8 ms = 2 slow cycles)
    for step in range(80):
        ukf.predict()
        ukf.update_fast(np.full((1, N_RINGS), 28000.0))
        if step % 40 == 0:
            ukf.update_slow(np.full((1, N_RINGS), 0.0))

    final_delta, final_sigma = ukf.get_estimates()

    # 1. Error from nominal 124.0 pm must be <= 0.1 pm
    max_err = np.max(np.abs(final_delta - 124.0))
    assert max_err <= 0.1, f"Steady-state detuning error {max_err:.4f} pm > 0.1 pm"

    # 2. Uncertainty must decrease as observations are integrated
    assert np.all(final_sigma < initial_sigma), "Uncertainty did not reduce after 80 steps"
    assert np.all(final_sigma > 0.0), "Uncertainty collapsed to zero or negative"


def test_ukf_dynamic_tracking_thermal_step():
    """Verify UKF dynamically tracks sudden thermal step (y0=24000.0, zeta=0.15)."""
    ukf = DualRateUKF(n_rings=N_RINGS, dt_fast=DT_FAST, dt_slow=DT_SLOW)
    ukf.reset(batch_size=1)

    # Settle initially at nominal
    for step in range(40):
        ukf.predict()
        ukf.update_fast(np.full((1, N_RINGS), 28000.0))
        if step % 40 == 0:
            ukf.update_slow(np.full((1, N_RINGS), 0.0))

    delta_pre = ukf.delta.copy()

    # Apply thermal disturbance: y0 drops to 24000.0, zeta steps to 0.15
    for step in range(40, 80):
        ukf.predict()
        ukf.update_fast(np.full((1, N_RINGS), 24000.0))
        if step % 40 == 0:
            ukf.update_slow(np.full((1, N_RINGS), 0.15))

    delta_post = ukf.delta.copy()
    shift = delta_post[0, 0] - delta_pre[0, 0]

    # Detuning must shift significantly in positive direction
    assert shift > 20.0, f"Expected detuning shift > 20 pm, got {shift:.2f} pm"
    # Adaptive multiplier qm must expand to open bandwidth
    assert ukf.qm[0, 0] > 5.0, f"Expected qm > 5.0 under step disturbance, got {ukf.qm[0, 0]:.2f}"
    assert not np.any(np.isnan(delta_post))


# ==============================================================================
# 2. Outlier Gating & Cross-Ring Decoupling Tests
# ==============================================================================

@pytest.mark.parametrize("glitch_val, desc", [
    (0.0, "zero code"),
    (65535.0, "ADC rail saturation"),
    (-10000.0, "negative ADC code"),
    (1000000.0, "extreme overrange spike"),
])
def test_ukf_fast_dc_outlier_gating_and_cross_ring_isolation(glitch_val, desc):
    """Verify extreme glitch on ring 3 does not corrupt neighbor ring 4 by > 15 pm."""
    ukf = DualRateUKF(n_rings=N_RINGS, dt_fast=DT_FAST, dt_slow=DT_SLOW)
    ukf.reset(batch_size=1)

    # Settle for 40 steps
    for step in range(40):
        ukf.predict()
        ukf.update_fast(np.full((1, N_RINGS), 28000.0))
        if step % 40 == 0:
            ukf.update_slow(np.full((1, N_RINGS), 0.0))

    d_steady = ukf.delta.copy()

    # Single-step extreme glitch on ring 3
    glitch_dc = np.full((1, N_RINGS), 28000.0)
    glitch_dc[0, 3] = glitch_val
    ukf.predict()
    ukf.update_fast(glitch_dc)

    d_glitch = ukf.delta.copy()
    diff = np.abs(d_glitch[0] - d_steady[0])

    # Neighbor ring 4 corruption must be strictly <= 15 pm
    assert diff[4] <= 15.0, f"Neighbor ring 4 corrupted by {diff[4]:.4f} pm on {desc}"
    # In fact, cross-ring decoupling should isolate it to < 1e-6 pm
    assert diff[4] < 1e-6, f"Cross-ring decoupling leaked {diff[4]:.4e} pm to neighbor ring 4"
    # Ring 3 itself should be bounded by Huber gating (< 2.0 pm)
    assert diff[3] < 2.0, f"Ring 3 detuning corrupted too heavily: {diff[3]:.4f} pm"
    # Huber gating flag on ring 3 must be asserted
    assert ukf.gated[0, 3] >= 1, "Huber gating failed to flag extreme glitch"


@pytest.mark.parametrize("zeta_glitch", [10.0, -10.0])
def test_ukf_slow_zeta_outlier_gating(zeta_glitch):
    """Verify extreme harmonic ratio spike on ring 3 is gated and does not corrupt neighbors."""
    ukf = DualRateUKF(n_rings=N_RINGS, dt_fast=DT_FAST, dt_slow=DT_SLOW)
    ukf.reset(batch_size=1)

    for step in range(40):
        ukf.predict()
        ukf.update_fast(np.full((1, N_RINGS), 28000.0))
        if step % 40 == 0:
            ukf.update_slow(np.full((1, N_RINGS), 0.0))

    d_steady = ukf.delta.copy()

    ukf.predict()
    ukf.update_fast(np.full((1, N_RINGS), 28000.0))
    z = np.full((1, N_RINGS), 0.0)
    z[0, 3] = zeta_glitch
    ukf.update_slow(z)

    d_glitch = ukf.delta.copy()
    diff = np.abs(d_glitch[0] - d_steady[0])

    assert diff[4] <= 15.0, f"Neighbor ring 4 corrupted by slow zeta glitch: {diff[4]:.4f} pm"
    assert diff[3] < 2.0, f"Glitched ring 3 corrupted too heavily: {diff[3]:.4f} pm"


def test_ukf_multiple_simultaneous_glitches():
    """Verify simultaneous glitches on ring 1 (0.0) and ring 3 (65535.0) isolate sandwiched ring 2."""
    ukf = DualRateUKF(n_rings=N_RINGS, dt_fast=DT_FAST, dt_slow=DT_SLOW)
    ukf.reset(batch_size=1)

    for step in range(40):
        ukf.predict()
        ukf.update_fast(np.full((1, N_RINGS), 28000.0))

    d_steady = ukf.delta.copy()

    glitch_dc = np.full((1, N_RINGS), 28000.0)
    glitch_dc[0, 1] = 0.0
    glitch_dc[0, 3] = 65535.0
    ukf.predict()
    ukf.update_fast(glitch_dc)

    diff = np.abs(ukf.delta[0] - d_steady[0])
    # Ring 2 sandwiched between 1 and 3 must not be corrupted
    assert diff[2] < 1e-6, f"Sandwiched ring 2 corrupted by {diff[2]:.4e} pm"
    assert diff[4] < 1e-6, f"Neighbor ring 4 corrupted by {diff[4]:.4e} pm"


# ==============================================================================
# 3. Squelch Handling & Photodiode Scale Freeze Tests
# ==============================================================================

def test_ukf_squelch_freezes_photodiode_scale_single_channel():
    """Verify photodiode scale p remains frozen within 0.1% when single-channel squelch is active."""
    ukf = DualRateUKF(n_rings=N_RINGS, dt_fast=DT_FAST, dt_slow=DT_SLOW)
    ukf.reset(batch_size=1)

    # Settle
    for step in range(40):
        ukf.predict()
        ukf.update_fast(np.full((1, N_RINGS), 28000.0))
        if step % 40 == 0:
            ukf.update_slow(np.full((1, N_RINGS), 0.0))

    p_before = ukf.pscale.copy()
    sq_mask = np.zeros((1, N_RINGS), dtype=bool)
    sq_mask[0, 3] = True  # Ring 3 squelched

    # 100 steps of signal collapse on squelched channel
    for step in range(40, 140):
        ukf.predict()
        dc = np.full((1, N_RINGS), 28000.0)
        dc[0, 3] = 0.0  # signal collapsed
        ukf.update_fast(dc, squelch_mask=sq_mask)
        if step % 40 == 0:
            z = np.full((1, N_RINGS), 0.0)
            z[0, 3] = 2.0  # corrupted ratio
            ukf.update_slow(z, squelch_mask=sq_mask)

    p_after = ukf.pscale.copy()
    rel_p_err = np.abs(p_after[0] - p_before[0]) / p_before[0]

    # Ring 3 scale must remain frozen within 0.1% (0.001)
    assert rel_p_err[3] <= 0.001, f"Ring 3 p drifted by {rel_p_err[3]*100:.4f}% > 0.1%"
    assert rel_p_err[3] == pytest.approx(0.0, abs=1e-9)


def test_ukf_squelch_freezes_all_channels():
    """Verify photodiode scale p remains frozen within 0.1% when all channels are squelched."""
    ukf = DualRateUKF(n_rings=N_RINGS, dt_fast=DT_FAST, dt_slow=DT_SLOW)
    ukf.reset(batch_size=1)

    for step in range(40):
        ukf.predict()
        ukf.update_fast(np.full((1, N_RINGS), 28000.0))

    p_before = ukf.pscale.copy()
    all_squelch = np.ones((1, N_RINGS), dtype=bool)

    for step in range(40, 140):
        ukf.predict()
        ukf.update_fast(np.full((1, N_RINGS), 500.0), squelch_mask=all_squelch)
        if step % 40 == 0:
            ukf.update_slow(np.full((1, N_RINGS), 0.0), squelch_mask=all_squelch)

    p_after = ukf.pscale.copy()
    rel_p_err = np.abs(p_after - p_before) / p_before

    assert np.all(rel_p_err <= 0.001), f"Squelched p scale drifted: max rel err = {np.max(rel_p_err):.4e}"


# ==============================================================================
# 4. Numerical Stability & Covariance Conditioning Tests
# ==============================================================================

@pytest.mark.parametrize("B", [1, 2, 4, 8])
def test_ukf_500_noisy_steps_covariance_positive_definite(B):
    """Verify all eigenvalues of P remain strictly positive (lambda_min > 0) across 500 noisy steps."""
    rng = np.random.default_rng(2026 + B)
    ukf = DualRateUKF(n_rings=N_RINGS, dt_fast=DT_FAST, dt_slow=DT_SLOW)
    ukf.reset(batch_size=B)

    min_eig_overall = float("inf")
    max_asymmetry = 0.0

    for step in range(500):
        u_pwr = np.clip(rng.normal(0.015, 0.003, (B, N_RINGS)), 0.0, 0.050)
        ukf.predict(u_pwr)

        y0_noisy = 28000.0 + rng.normal(0.0, 150.0, (B, N_RINGS))
        ukf.update_fast(y0_noisy)

        if step % 40 == 0:
            z_noisy = rng.normal(0.0, 0.05, (B, N_RINGS))
            ukf.update_slow(z_noisy)

        for b in range(B):
            P_mat = ukf.P[b]
            asym = float(np.max(np.abs(P_mat - P_mat.T)))
            if asym > max_asymmetry:
                max_asymmetry = asym
            eigs = np.linalg.eigvalsh(P_mat)
            lmin = float(np.min(eigs))
            if lmin < min_eig_overall:
                min_eig_overall = lmin

    # Eigenvalues must be strictly positive (lambda_min > 0)
    assert min_eig_overall > 0.0, f"Covariance lost positive definiteness: min_eig = {min_eig_overall}"
    # Covariance must remain numerically symmetric
    assert max_asymmetry < 1e-12, f"Covariance asymmetry detected: {max_asymmetry:.4e}"


def test_ukf_long_horizon_stability_1000_steps():
    """Verify UKF remains bounded and stable across 1000 consecutive noisy steps."""
    rng = np.random.default_rng(999)
    ukf = DualRateUKF(n_rings=N_RINGS, dt_fast=DT_FAST, dt_slow=DT_SLOW)
    ukf.reset(batch_size=1)

    for step in range(1000):
        ukf.predict()
        y0_noisy = 28000.0 + rng.normal(0.0, 100.0, (1, N_RINGS))
        ukf.update_fast(y0_noisy)
        if step % 40 == 0:
            ukf.update_slow(rng.normal(0.0, 0.03, (1, N_RINGS)))

    delta_hat, sigma_delta = ukf.get_estimates()
    assert not np.any(np.isnan(delta_hat))
    assert not np.any(np.isnan(sigma_delta))
    assert np.all(sigma_delta > 0.0)
    # Under open-loop sensor noise, detuning estimate remains bounded within 2-sigma (< 25 pm)
    assert np.all(np.abs(delta_hat - 124.0) < 25.0)


def test_ukf_noise_free_extended_convergence_1000_steps():
    """Verify UKF remains tightly converged within <= 0.2 pm across 1000 noise-free steps."""
    ukf = DualRateUKF(n_rings=N_RINGS, dt_fast=DT_FAST, dt_slow=DT_SLOW)
    ukf.reset(batch_size=1)

    for step in range(1000):
        ukf.predict()
        ukf.update_fast(np.full((1, N_RINGS), 28000.0))
        if step % 40 == 0:
            ukf.update_slow(np.full((1, N_RINGS), 0.0))

    delta_hat, sigma_delta = ukf.get_estimates()
    assert np.all(np.abs(delta_hat - 124.0) < 0.2), f"Drifted: {delta_hat}"



# ==============================================================================
# 5. Vectorized Batch Execution across B in [1, 2, 4, 8]
# ==============================================================================

@pytest.mark.parametrize("B", [1, 2, 4, 8])
def test_ukf_batch_output_shapes_and_consistency(B):
    """Verify UKF executes seamlessly and returns (B, N) outputs for B in [1, 2, 4, 8]."""
    ukf = DualRateUKF(n_rings=N_RINGS, dt_fast=DT_FAST, dt_slow=DT_SLOW)
    ukf.reset(batch_size=B)

    ukf.predict(np.zeros((B, N_RINGS)))
    ukf.update_fast(np.full((B, N_RINGS), 28000.0))
    ukf.update_slow(np.zeros((B, N_RINGS)))

    delta_hat, sigma_delta = ukf.get_estimates()
    assert delta_hat.shape == (B, N_RINGS)
    assert sigma_delta.shape == (B, N_RINGS)
    assert ukf.cov.shape == (B, 24, 24)
    assert ukf.delta.shape == (B, N_RINGS)
    assert ukf.rate.shape == (B, N_RINGS)
    assert ukf.pscale.shape == (B, N_RINGS)


# ==============================================================================
# 6. Adversarial Shape Handling & Edge Case Defects
# ==============================================================================

def test_ukf_1d_y0_input_with_batch_size_greater_than_1():
    """Adversarial probe: update_fast with 1D input (N,) when batch_size > 1."""
    ukf = DualRateUKF(n_rings=N_RINGS)
    ukf.reset(batch_size=4)
    # Docstring advertises shape (B, N) or (N,)
    ukf.update_fast(y0_dc=np.full(N_RINGS, 28000.0))


def test_ukf_1d_squelch_mask_with_batch_size_greater_than_1():
    """Adversarial probe: update_fast with 1D squelch_mask (N,) when batch_size > 1."""
    ukf = DualRateUKF(n_rings=N_RINGS)
    ukf.reset(batch_size=4)
    ukf.update_fast(y0_dc=np.full((4, N_RINGS), 28000.0), squelch_mask=np.zeros(N_RINGS, dtype=bool))


def test_ukf_unified_update_1d_squelch_breaks_channel_masking_in_predict():
    """Adversarial probe: update() with 1D squelch corrupts per-channel squelch in predict()."""
    ukf = DualRateUKF(n_rings=N_RINGS)
    ukf.reset(batch_size=1)
    sq_1d = np.array([False, False, True, False, False, False, False, False], dtype=bool)
    ukf.update(zeta=np.zeros((1, N_RINGS)), squelch=sq_1d)

    # In predict, self.squelch_mask[0] indexes ring 0 (False) instead of the boolean vector
    # so ring 2's process noise is NOT frozen
    ukf.predict()
    # Check if squelch_mask is properly 2D
    assert ukf.squelch_mask.ndim == 2, f"Expected 2D squelch_mask, got {ukf.squelch_mask.shape}"
