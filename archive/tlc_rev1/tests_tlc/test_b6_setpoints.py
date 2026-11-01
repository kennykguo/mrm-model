"""Comprehensive unit and E2E tests for TLC Block B6 Setpoint Generator.

Tests cover:
- Red-flank inflection point calculation solving d^2L/d(delta)^2 = 0
- Verification of inflection target ~ +124 pm (within 122 - 126 pm range for G26)
- Negative thermo-optic feedback slope (dL/d(delta) < 0) ensuring stability against thermal runaway
- Maximum slope magnitude (|dL/d(delta)|) at inflection point maximizing lock-in SNR
- Fallback to bare Lorentzian inflection point HWHM / sqrt(3) ~ 98.2 pm when mod_shift = 0
- Proportional scaling of inflection target with optical resonance FWHM / loaded Q
- Target override support
- Vectorization across batch dimension B and channel count N=8
"""
import numpy as np
import pytest

try:
    from mrm.control.tlc.b6_setpoints import SetpointGenerator
except ImportError:
    try:
        from mrm.control.tlc import SetpointGenerator
    except ImportError:
        try:
            from mrm.control.tlc.b6_setpoints import B6Setpoints as SetpointGenerator
        except ImportError:
            SetpointGenerator = None

requires_b6 = pytest.mark.skipif(
    SetpointGenerator is None,
    reason="mrm.control.tlc.b6_setpoints (SetpointGenerator / B6Setpoints) is not yet implemented",
)

NOMINAL_Q_LOADED = 3850.0
NOMINAL_WAVELENGTH_NM = 1310.0
NOMINAL_FWHM_PM = NOMINAL_WAVELENGTH_NM / NOMINAL_Q_LOADED * 1000.0  # 340.26 pm
NOMINAL_HWHM_PM = NOMINAL_FWHM_PM / 2.0                            # 170.13 pm
N_RINGS = 8


def _instantiate_setpoints(n_rings=8, q_loaded=NOMINAL_Q_LOADED, wavelength_nm=NOMINAL_WAVELENGTH_NM, **kwargs):
    """Helper to instantiate SetpointGenerator across supported signatures."""
    try:
        return SetpointGenerator(n_rings=n_rings, q_loaded=q_loaded, wavelength_nm=wavelength_nm, **kwargs)
    except TypeError:
        fwhm_pm = wavelength_nm / q_loaded * 1000.0
        return SetpointGenerator(n_rings=n_rings, fwhm_pm=fwhm_pm, **kwargs)


def _get_inflection_point(gen):
    """Helper to extract inflection target."""
    if hasattr(gen, "calculate_inflection_setpoint"):
        return gen.calculate_inflection_setpoint()
    if hasattr(gen, "targets_pm"):
        targets = gen.targets_pm
        return float(np.mean(targets))
    if hasattr(gen, "get_setpoints"):
        sp = gen.get_setpoints(batch_size=1)
        return float(np.mean(sp))
    raise AttributeError("Setpoint generator has no inflection accessor")


def _get_setpoints_array(gen, batch_size=1):
    """Helper to extract (B, N) setpoints."""
    if hasattr(gen, "get_setpoints"):
        return gen.get_setpoints(batch_size=batch_size)
    if hasattr(gen, "targets_pm"):
        targets = np.asarray(gen.targets_pm).reshape(1, -1)
        return np.repeat(targets, batch_size, axis=0)
    raise AttributeError("Setpoint generator has no setpoint array accessor")


def _lorentzian_avg(delta, hwhm, delta_mod):
    """Data-averaged Lorentzian drop transmission under NRZ modulation."""
    l_plus = 1.0 / (1.0 + ((delta + delta_mod) / hwhm) ** 2)
    l_minus = 1.0 / (1.0 + ((delta - delta_mod) / hwhm) ** 2)
    return 0.5 * (l_plus + l_minus)


# ==============================================================================
# Tier 1: Inflection Target & Mathematical Optimality Tests
# ==============================================================================

@requires_b6
def test_b6_inflection_target_range_nominal():
    """Verify calculated inflection target is in nominal range [122.0, 126.0] pm (~ +124 pm)."""
    gen = _instantiate_setpoints(n_rings=N_RINGS)
    target = _get_inflection_point(gen)
    assert 122.0 <= target <= 126.0, (
        f"Calculated inflection target {target:.3f} pm outside expected [122.0, 126.0] pm range"
    )
    assert target == pytest.approx(124.0, abs=2.0)


@requires_b6
def test_b6_second_derivative_zero_at_inflection():
    """Verify the second derivative d^2L/d(delta)^2 vanishes at the calculated setpoint."""
    gen = _instantiate_setpoints(n_rings=N_RINGS)
    target = _get_inflection_point(gen)

    # Numerical second derivative using central differences
    hwhm = NOMINAL_HWHM_PM
    mod_shift = getattr(gen, "mod_shift_pm", 50.0)
    eps = 1e-4

    l_plus = _lorentzian_avg(target + eps, hwhm, mod_shift)
    l_mid = _lorentzian_avg(target, hwhm, mod_shift)
    l_minus = _lorentzian_avg(target - eps, hwhm, mod_shift)

    d2L = (l_plus - 2.0 * l_mid + l_minus) / (eps ** 2)
    # At the inflection point, second derivative must be extremely close to 0
    assert abs(d2L) < 1e-5, f"Second derivative at inflection target is non-zero: d2L = {d2L:.2e}"


@requires_b6
def test_b6_negative_feedback_slope():
    """Verify first derivative dL/d(delta) < 0 at the red-flank setpoint ensuring self-stabilization."""
    gen = _instantiate_setpoints(n_rings=N_RINGS)
    target = _get_inflection_point(gen)

    hwhm = NOMINAL_HWHM_PM
    mod_shift = getattr(gen, "mod_shift_pm", 50.0)
    eps = 1e-4

    # Central difference first derivative
    dL = (_lorentzian_avg(target + eps, hwhm, mod_shift) - _lorentzian_avg(target - eps, hwhm, mod_shift)) / (2.0 * eps)

    # Transmission slope must be strictly negative on the red flank
    assert dL < 0.0, (
        f"Red-flank slope must be negative for thermo-optic stability, but got dL = {dL:.4e}"
    )


@requires_b6
def test_b6_maximum_slope_magnitude_at_inflection():
    """Verify slope magnitude |dL/d(delta)| is maximized at the inflection point."""
    gen = _instantiate_setpoints(n_rings=N_RINGS)
    target = _get_inflection_point(gen)

    hwhm = NOMINAL_HWHM_PM
    mod_shift = getattr(gen, "mod_shift_pm", 50.0)
    eps = 1e-3

    def slope_at(d):
        return abs((_lorentzian_avg(d + eps, hwhm, mod_shift) - _lorentzian_avg(d - eps, hwhm, mod_shift)) / (2.0 * eps))

    slope_inflection = slope_at(target)
    slope_left = slope_at(target - 25.0)
    slope_right = slope_at(target + 25.0)

    # Inflection slope magnitude must exceed neighboring slopes
    assert slope_inflection >= slope_left, "Slope at inflection must be greater than red flank shoulder"
    assert slope_inflection >= slope_right, "Slope at inflection must be greater than outer red flank"


# ==============================================================================
# Tier 2: Boundary, Parametric Scaling & Fallback Tests
# ==============================================================================

@requires_b6
def test_b6_bare_lorentzian_fallback():
    """Verify inflection point collapses to HWHM / sqrt(3) ~ 98.22 pm when mod_shift is 0."""
    gen = _instantiate_setpoints(n_rings=N_RINGS, mod_shift_pm=0.0)
    if hasattr(gen, "compute_inflection_point"):
        target_bare = gen.compute_inflection_point(NOMINAL_HWHM_PM, 0.0)
        expected_bare = NOMINAL_HWHM_PM / np.sqrt(3.0)
        np.testing.assert_allclose(target_bare, expected_bare, rtol=1e-3)


@requires_b6
def test_b6_fwhm_scaling():
    """Verify inflection point scales with optical resonance linewidth (FWHM)."""
    gen_narrow = _instantiate_setpoints(n_rings=N_RINGS, q_loaded=5000.0)  # narrower resonance
    gen_wide = _instantiate_setpoints(n_rings=N_RINGS, q_loaded=2500.0)    # wider resonance

    target_narrow = _get_inflection_point(gen_narrow)
    target_wide = _get_inflection_point(gen_wide)

    # Wider resonance (lower Q) must have a larger detuning setpoint
    assert target_wide > target_narrow


@requires_b6
def test_b6_target_override():
    """Verify explicit target override is honored by generator."""
    custom_target = 135.0
    gen = _instantiate_setpoints(n_rings=N_RINGS, target_override_pm=custom_target)
    target = _get_inflection_point(gen)
    assert target == pytest.approx(custom_target)


# ==============================================================================
# Tier 3: Vectorized Setpoints & Shape Tests
# ==============================================================================

@requires_b6
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_b6_vectorized_setpoints_shape(batch_size):
    """Verify get_setpoints produces (B, N) array with N=8 channels."""
    gen = _instantiate_setpoints(n_rings=N_RINGS)
    sp = _get_setpoints_array(gen, batch_size=batch_size)

    assert sp.shape == (batch_size, N_RINGS)
    # All rings should be initialized near nominal inflection target
    np.testing.assert_allclose(sp, 124.0, atol=2.0)
