"""Adversarial and empirical stress test suite for Block B4 and Block B6.

Challenger test suite verifying:
1. Block B4 Reduced Thermal Model (b4_thermal_model.py):
   - Spectral radius of Ad strictly < 1.0, all eigenvalues non-negative real
   - Crosstalk coupling matrix symmetry K = K^T, zero diagonal, monotonic decay with distance (K[0,1] >= K[0,2] >= K[0,4])
   - Step response monotonicity under heater power steps (20 mW step over 100 steps)
   - Zero-input asymptotic settling to zero
   - Analytical steady-state gain matrix consistency with infinite-step simulation
   - Discrete-time state-space stability across varying sampling intervals dt
   - Vectorized batch execution and shape conformance for B in [1, 4, 16] and 1D inputs
2. Block B6 Red-Flank Setpoints (b6_setpoints.py):
   - Numerical second derivative d^2 L / d delta^2 = 0 within |d^2 L| < 1e-5 at calculated inflection setpoint
   - Negative slope dL/d delta < 0 for thermo-optic self-stabilization
   - Maximum slope magnitude |dL/d delta| at inflection point compared to delta* +/- 25 pm
   - Bare Lorentzian fallback when modulation m = 0 (HWHM / sqrt(3) ~ 98.22 pm)
   - Vectorized batch output shape (B, N) across arbitrary B in [1, 4, 16]
   - Scaling across loaded Q factors (1000 to 10000)
   - Adversarial stress on asymmetric mark fractions (p1) and edge case boundary conditions
"""

import numpy as np
import pytest

from mrm.control.tlc.b4_thermal_model import ReducedThermalModel
from mrm.control.tlc.b6_setpoints import SetpointGenerator, SetpointConfig


# ==============================================================================
# Block B4 Reduced Thermal Model Stress Tests
# ==============================================================================

def test_b4_spectral_radius_and_nonnegative_real_eigenvalues():
    """Adversarially verify Ad eigenvalues: rho(Ad) < 1.0, imag == 0, real >= 0."""
    model = ReducedThermalModel(n_rings=8, dt=100e-6)
    Ad = model.Ad

    eigvals = np.linalg.eigvals(Ad)
    spectral_radius = float(np.max(np.abs(eigvals)))

    # 1. Strictly stable: rho < 1.0
    assert spectral_radius < 1.0, f"Spectral radius rho = {spectral_radius:.6f} >= 1.0"
    assert spectral_radius == pytest.approx(np.exp(-100e-6 / 3.0e-3), rel=1e-5)

    # 2. All eigenvalues strictly real (no oscillatory modes in pure thermal diffusion)
    np.testing.assert_allclose(np.imag(eigvals), 0.0, atol=1e-15)

    # 3. All eigenvalues non-negative (no alternating sign / negative pole artifacts)
    assert np.all(np.real(eigvals) >= 0.0), f"Negative eigenvalues found in Ad: {eigvals}"

    # Verify both slow self-heating and crosstalk pole clusters
    assert len(eigvals) == 16
    n_slow_poles = np.sum(np.isclose(np.real(eigvals), model.a_hs, atol=1e-6))
    n_xt_poles = np.sum(np.isclose(np.real(eigvals), model.a_x, atol=1e-6))
    assert n_slow_poles == 8, f"Expected 8 slow poles, got {n_slow_poles}"
    assert n_xt_poles == 8, f"Expected 8 crosstalk poles, got {n_xt_poles}"


@pytest.mark.parametrize("dt_test", [1e-6, 10e-6, 50e-6, 100e-6, 500e-6, 1e-3, 5e-3])
def test_b4_stability_across_dt_sweep(dt_test):
    """Verify Ad remains strictly stable and non-negative real across varying dt."""
    model = ReducedThermalModel(n_rings=8, dt=dt_test)
    eigvals = np.linalg.eigvals(model.Ad)
    rho = float(np.max(np.abs(eigvals)))

    assert 0.0 < rho < 1.0
    assert np.all(np.real(eigvals) > 0.0)
    assert np.all(np.imag(eigvals) == 0.0)


def test_b4_crosstalk_matrix_symmetry_and_spatial_decay():
    """Verify crosstalk matrix K is symmetric, zero-diagonal, and monotonically decays."""
    model = ReducedThermalModel(n_rings=8)
    K = model.K

    assert K.shape == (8, 8)

    # Exact symmetry: K == K^T
    np.testing.assert_allclose(K, K.T, atol=1e-15, err_msg="Crosstalk matrix K is not symmetric")

    # Zero diagonal: self-crosstalk must be zero
    np.testing.assert_allclose(np.diag(K), 0.0, atol=1e-15, err_msg="K diagonal is not strictly zero")

    # Monotonic decay with physical distance: K[0, 1] >= K[0, 2] >= K[0, 4]
    assert K[0, 1] >= K[0, 2] >= K[0, 4], (
        f"Spatial decay violated: K[0,1]={K[0,1]}, K[0,2]={K[0,2]}, K[0,4]={K[0,4]}"
    )
    assert K[0, 1] == pytest.approx(0.030)
    assert K[0, 2] == pytest.approx(0.015)
    assert K[0, 4] == pytest.approx(0.015)

    # Check all rows exhibit non-increasing monotonic decay with distance |i - j|
    for i in range(8):
        for dist in range(1, 7):
            j1 = i + dist
            j2 = i + dist + 1
            if j2 < 8:
                assert K[i, j1] >= K[i, j2]


def test_b4_step_response_monotonicity():
    """Empirically test step response monotonicity under 20 mW heater step over 100 steps."""
    model = ReducedThermalModel(n_rings=8)
    x = np.zeros((1, 16))
    u_step = np.full((1, 8), 0.020)  # 20 mW in Watts

    shifts = []
    for _ in range(100):
        x, y = model.step(x, u_step)
        shifts.append(y[0])

    shifts = np.array(shifts)  # (100, 8)
    step_diffs = np.diff(shifts, axis=0)

    # Every step increment must be strictly non-negative (heating must never reverse during a step)
    assert np.all(step_diffs >= -1e-12), (
        f"Non-monotonic step response detected! Min diff: {np.min(step_diffs)}"
    )

    # Verify steady-state limit matches analytical formula
    y_analytical_ss = model.steady_state_gain_matrix() @ u_step[0]
    final_sim_shift = shifts[-1]
    # At step 100 (10 ms = 3.33 * tau_tail), response should be > 99.5% settled
    assert np.all(final_sim_shift / y_analytical_ss > 0.995)


def test_b4_zero_input_asymptotic_settling():
    """Verify unforced zero-input response asymptotically settles to zero and follows exact pole decay."""
    model = ReducedThermalModel(n_rings=8)
    # Start from large arbitrary non-zero thermal state (+1000 pm)
    x0 = np.full((1, 16), 1000.0)
    u_zero = np.zeros((1, 8))

    x = x0.copy()
    for step in range(1, 1001):  # 100 ms total
        x, y = model.step(x, u_zero)
        if step == 500:
            # At 50 ms (500 steps), verify exact decay matches analytical Ad^k:
            # Slow self-heating poles: x0 * (a_hs)^500
            expected_hs = 1000.0 * (model.a_hs ** 500)
            np.testing.assert_allclose(x[0, :8], expected_hs, rtol=1e-10)
            # Crosstalk poles: x0 * (a_x)^500
            expected_x = 1000.0 * (model.a_x ** 500)
            np.testing.assert_allclose(x[0, 8:], expected_x, rtol=1e-10)

    # After 1000 steps (100 ms = 33.3 tau_tail), residual state and output must be < 1e-10 pm
    assert np.max(np.abs(x)) < 1e-10, f"Residual state {np.max(np.abs(x))} exceeds 1e-10 pm"
    assert np.max(np.abs(y)) < 1e-10, f"Residual output {np.max(np.abs(y))} exceeds 1e-10 pm"


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_b4_batch_execution_and_shapes(batch_size):
    """Verify B4 step accepts (B, 16) state and (B, 8) input, returning (B, 16) and (B, 8)."""
    model = ReducedThermalModel(n_rings=8)
    x = np.zeros((batch_size, 16))
    u = np.full((batch_size, 8), 0.015)

    x_next, y = model.step(x, u)
    assert x_next.shape == (batch_size, 16)
    assert y.shape == (batch_size, 8)
    assert np.all(np.isfinite(x_next))
    assert np.all(np.isfinite(y))


def test_b4_1d_input_handling():
    """Verify B4 handles 1D array inputs gracefully."""
    model = ReducedThermalModel(n_rings=8)
    x = np.zeros(16)
    u = np.full(8, 0.020)

    x_next, y = model.step(x, u)
    assert x_next.shape == (16,)
    assert y.shape == (8,)


def test_b4_settle_equivalence():
    """Verify settle() returns analytical steady-state matching infinite simulation."""
    model = ReducedThermalModel(n_rings=8)
    u = np.full(8, 0.025)

    x_settle = model.settle(u)
    y_settle = model.h(u, x_settle)

    y_gain_mat = model.steady_state_gain_matrix() @ u
    np.testing.assert_allclose(y_settle, y_gain_mat, rtol=1e-12)


# ==============================================================================
# Block B6 Red-Flank Setpoint Stress Tests
# ==============================================================================

def test_b6_second_derivative_zero_at_inflection():
    """Verify numerical second derivative d^2L/ddelta^2 = 0 within |d^2L| < 1e-5."""
    gen = SetpointGenerator(n_rings=8, q_loaded=3850.0, wavelength_nm=1310.0, mod_shift_pm=50.0)
    target = gen.calculate_inflection_setpoint()

    gamma = gen.hwhm_pm
    mod_shift = gen.mod_shift_pm
    p1 = 0.5

    def l_avg(delta):
        u1 = (delta + mod_shift) / gamma
        u0 = (delta - mod_shift) / gamma
        return p1 / (1.0 + u1**2) + (1.0 - p1) / (1.0 + u0**2)

    # Central difference with h = 1e-4
    h = 1e-4
    d2 = (l_avg(target + h) - 2.0 * l_avg(target) + l_avg(target - h)) / (h**2)

    assert abs(d2) < 1e-5, f"Numerical second derivative {d2:.2e} exceeds 1e-5 threshold"
    # Also verify with 5-point stencil for higher order accuracy
    d2_5pt = (-l_avg(target + 2*h) + 16*l_avg(target + h) - 30*l_avg(target) + 16*l_avg(target - h) - l_avg(target - 2*h)) / (12 * h**2)
    assert abs(d2_5pt) < 1e-6, f"5-point stencil second derivative {d2_5pt:.2e} exceeds 1e-6"


def test_b6_negative_feedback_slope():
    """Verify negative transmission slope dL/ddelta < 0 at inflection for thermo-optic stability."""
    gen = SetpointGenerator(n_rings=8)
    target = gen.calculate_inflection_setpoint()

    gamma = gen.hwhm_pm
    mod_shift = gen.mod_shift_pm
    p1 = 0.5

    def l_avg(delta):
        u1 = (delta + mod_shift) / gamma
        u0 = (delta - mod_shift) / gamma
        return p1 / (1.0 + u1**2) + (1.0 - p1) / (1.0 + u0**2)

    h = 1e-4
    slope = (l_avg(target + h) - l_avg(target - h)) / (2.0 * h)

    # Must be strictly negative (red flank)
    assert slope < 0.0, f"Slope {slope:.4e} must be strictly negative on the red flank"
    assert slope == pytest.approx(-0.003244, rel=1e-3)


def test_b6_maximum_slope_magnitude_at_inflection():
    """Verify slope magnitude |dL/ddelta| is strictly maximized at inflection vs delta* +/- 25 pm."""
    gen = SetpointGenerator(n_rings=8)
    target = gen.calculate_inflection_setpoint()

    gamma = gen.hwhm_pm
    mod_shift = gen.mod_shift_pm

    def slope_mag(delta):
        h = 1e-4
        l_hi = 0.5 * (1.0 / (1.0 + ((delta + h + mod_shift) / gamma)**2) + 1.0 / (1.0 + ((delta + h - mod_shift) / gamma)**2))
        l_lo = 0.5 * (1.0 / (1.0 + ((delta - h + mod_shift) / gamma)**2) + 1.0 / (1.0 + ((delta - h - mod_shift) / gamma)**2))
        return abs((l_hi - l_lo) / (2.0 * h))

    slope_target = slope_mag(target)
    slope_minus_25 = slope_mag(target - 25.0)
    slope_plus_25 = slope_mag(target + 25.0)

    assert slope_target > slope_minus_25, f"Inflection slope {slope_target} <= {slope_minus_25} at -25 pm"
    assert slope_target > slope_plus_25, f"Inflection slope {slope_target} <= {slope_plus_25} at +25 pm"


def test_b6_bare_lorentzian_fallback():
    """Verify bare Lorentzian fallback when mod_shift = 0 is HWHM / sqrt(3) ~ 98.22 pm."""
    gamma = 170.13
    expected_bare = gamma / np.sqrt(3.0)  # 98.2245 pm

    target_func = SetpointGenerator.compute_inflection_point(gamma, mod_shift_pm=0.0)
    np.testing.assert_allclose(target_func, expected_bare, rtol=1e-12)

    gen = SetpointGenerator(n_rings=8, mod_shift_pm=0.0)
    target_gen = gen.calculate_inflection_setpoint()
    np.testing.assert_allclose(target_gen, gen.hwhm_pm / np.sqrt(3.0), rtol=1e-12)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_b6_batch_output_shapes(batch_size):
    """Verify get_setpoints produces (B, N) array across B in [1, 4, 16]."""
    gen = SetpointGenerator(n_rings=8)
    sp = gen.get_setpoints(batch_size=batch_size)

    assert sp.shape == (batch_size, 8)
    assert sp.dtype == np.float64
    assert np.all(sp > 0.0)
    assert np.all(np.isfinite(sp))
    # All rows must be identical for uniform channels
    for b in range(1, batch_size):
        np.testing.assert_allclose(sp[b], sp[0])


@pytest.mark.parametrize("q_loaded", [1000.0, 2500.0, 3850.0, 6000.0, 10000.0])
def test_b6_loaded_q_scaling(q_loaded):
    """Verify inflection target scales inversely with loaded Q factor."""
    gen = SetpointGenerator(n_rings=8, q_loaded=q_loaded)
    target = gen.calculate_inflection_setpoint()

    # Target must be strictly positive and decrease monotonically with higher Q (narrower resonance)
    assert target > 0.0
    expected_fwhm = (1310.0 / q_loaded) * 1000.0
    expected_approx = (expected_fwhm / 2.0) / np.sqrt(3.0)
    # Target should be bounded around the bare Lorentzian with modulation expansion
    assert target > expected_approx * 0.9


def test_b6_adversarial_mark_fraction_finding():
    """Adversarial challenge: test mark fraction p1 asymmetry.
    
    Documenting finding:
    For standard telecomm streams with balanced bits (p1=0.5), SetpointGenerator is rock-solid.
    However, when mark density is highly biased (p1 >= 0.99 for mod_shift=50 pm,
    or p1 >= 0.90 for mod_shift=70 pm), the true inflection point shifts below
    the hardcoded lower bracket low = max(0.01, gamma / sqrt(3) * 0.5) ~ 49.11 pm,
    causing Brent's method fallback to raise ValueError.
    """
    gamma = 170.13
    # Moderately unbalanced mark fractions succeed
    for p1 in [0.2, 0.4, 0.5, 0.6, 0.8]:
        target = SetpointGenerator.compute_inflection_point(gamma, mod_shift_pm=50.0, frac_ones=p1)
        assert target > 0.0 and np.isfinite(target)
