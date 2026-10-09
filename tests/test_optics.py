from dataclasses import replace

import numpy as np
import pytest

from ringlab.optics import C0, RingParameters, decay_rates, loaded_linewidth_hz, nonlinear_screening, optical_response, resonance_frequency
from ringlab.waveform import tcmt_response


def test_add_drop_energy_conservation_across_one_fsr_and_loss_levels():
    p = RingParameters(input_coupling_power=0.18, drop_coupling_power=0.07)
    frequencies = p.reference_frequency_hz + np.linspace(-0.5, 0.5, 2001) * p.fsr_hz
    for alpha in (0, 1e-6, 250, 10000):
        r = optical_response(replace(p, intrinsic_loss_per_m=alpha), frequencies, input_power_w=0.003)
        np.testing.assert_allclose(r.through_fraction + r.drop_fraction + r.intrinsic_loss_fraction, 1, atol=3e-14)
        np.testing.assert_allclose(r.absorbed_power_w, p.absorption_fraction * r.intrinsic_loss_power_w)
        assert np.all(r.stored_energy_j >= 0)


def test_single_bus_closed_form_and_lossless_limit():
    p = RingParameters(drop_coupling_power=0)
    frequency = p.reference_frequency_hz + np.linspace(-1, 1, 41) * 20e9
    phi = 2 * np.pi * (frequency - p.reference_frequency_hz) * p.roundtrip_time_s
    a, r = np.exp(-p.intrinsic_loss_per_m * p.roundtrip_length_m / 2), np.sqrt(1 - p.input_coupling_power)
    expected = (r**2 + a**2 - 2 * a * r * np.cos(phi)) / (1 + (r * a)**2 - 2 * a * r * np.cos(phi))
    response = optical_response(p, frequency)
    np.testing.assert_allclose(response.through_fraction, expected, atol=2e-12)
    np.testing.assert_allclose(response.drop_fraction, 0)
    lossless = optical_response(replace(p, intrinsic_loss_per_m=0), frequency)
    np.testing.assert_allclose(lossless.through_fraction, 1, atol=1e-14)
    assert np.all(lossless.stored_energy_j > 0)


def test_generalized_critical_coupling_null():
    p = RingParameters(drop_coupling_power=0.03)
    critical_k1 = 1 - np.exp(-p.intrinsic_loss_per_m * p.roundtrip_length_m) * (1 - p.drop_coupling_power)
    p = replace(p, input_coupling_power=critical_k1)
    r = optical_response(p, p.reference_frequency_hz)
    assert r.through_fraction < 1e-26
    np.testing.assert_allclose(r.drop_fraction + r.intrinsic_loss_fraction, 1, atol=2e-14)


def test_stored_energy_matches_distributed_loss_and_zero_loss_integral():
    p = RingParameters()
    r = optical_response(p, p.reference_frequency_hz, input_power_w=2e-3)
    np.testing.assert_allclose(r.stored_energy_j * p.intrinsic_loss_per_m * C0 / p.group_index, r.intrinsic_loss_power_w)
    p = replace(p, intrinsic_loss_per_m=0)
    r = optical_response(p, p.reference_frequency_hz, input_power_w=2e-3)
    denominator = (1 - np.sqrt((1 - p.input_coupling_power) * (1 - p.drop_coupling_power)))**2
    energy = 2e-3 * p.input_coupling_power / denominator * p.roundtrip_time_s * (1 - p.drop_coupling_power / 2)
    np.testing.assert_allclose(r.stored_energy_j, energy)


def test_linewidth_half_maximum_and_fsr():
    p = RingParameters()
    width = loaded_linewidth_hz(p)
    r = optical_response(p, p.reference_frequency_hz + np.array([-0.5, 0, 0.5]) * width)
    np.testing.assert_allclose(r.drop_fraction[[0, 2]], 0.5 * r.drop_fraction[1], rtol=2e-11)
    periodic = optical_response(p, p.reference_frequency_hz + p.fsr_hz)
    np.testing.assert_allclose(periodic.drop_fraction, r.drop_fraction[1], rtol=1e-11)
    ki, k1, k2 = decay_rates(p)
    assert abs(width / ((ki + k1 + k2) / (2 * np.pi)) - 1) < 1e-4


def test_thermal_and_voltage_shift_signs_and_broadcasting():
    p = RingParameters()
    temperatures = np.array([[300], [301], [310]])
    frequency = p.reference_frequency_hz + np.array([0, -15e9])
    r = optical_response(p, frequency, temperature_k=temperatures)
    assert r.through_fraction.shape == (3, 2)
    assert resonance_frequency(p, 301) == p.reference_frequency_hz - 15e9
    np.testing.assert_allclose(r.drop_fraction[0, 0], r.drop_fraction[1, 1])
    assert resonance_frequency(p, 300, -20e9) < resonance_frequency(p, 300)


def test_dispersion_changes_phase_and_energy_consistently():
    p = RingParameters(beta2_s2_per_m=2e-21)
    delta = 3e9
    r = optical_response(p, p.reference_frequency_hz + delta)
    expected = 2 * np.pi * delta * p.roundtrip_time_s + 0.5 * p.beta2_s2_per_m * p.roundtrip_length_m * (2 * np.pi * delta)**2
    assert r.roundtrip_phase_rad == pytest.approx(expected)
    delay = p.roundtrip_time_s + p.beta2_s2_per_m * p.roundtrip_length_m * 2 * np.pi * delta
    np.testing.assert_allclose(r.stored_energy_j * p.intrinsic_loss_per_m * p.roundtrip_length_m / delay, r.intrinsic_loss_power_w)


def test_high_q_tcmt_converges_to_exact_per_pass():
    p = RingParameters(intrinsic_loss_per_m=20, input_coupling_power=0.002, drop_coupling_power=0.001)
    frequency = p.reference_frequency_hz + np.linspace(-3, 3, 301) * loaded_linewidth_hz(p)
    exact, local = optical_response(p, frequency), tcmt_response(p, frequency)
    np.testing.assert_allclose(local.through_power_w, exact.through_power_w, atol=2e-6)
    np.testing.assert_allclose(local.drop_power_w, exact.drop_power_w, atol=2e-6)
    np.testing.assert_allclose(local.absorbed_power_w, exact.absorbed_power_w, atol=2e-6)
    np.testing.assert_allclose(local.through_power_w + local.drop_power_w + local.intrinsic_loss_power_w, 1, atol=1e-14)


@pytest.mark.parametrize("kwargs", [{"input_coupling_power": 0}, {"drop_coupling_power": 1}, {"group_index": -1}, {"intrinsic_loss_per_m": -1}, {"absorption_fraction": 2}, {"beta2_s2_per_m": np.nan}])
def test_parameter_bounds(kwargs):
    with pytest.raises(ValueError):
        RingParameters(**kwargs)


def test_passivity_and_finite_input_checks():
    p = RingParameters()
    for kwargs in ({"additional_loss_per_m": -251}, {"input_power_w": -1}, {"temperature_k": 0}):
        with pytest.raises(ValueError):
            optical_response(p, p.reference_frequency_hz, **kwargs)
    with pytest.raises(ValueError):
        optical_response(p, np.nan)


def test_nonlinear_screening_power_scaling_and_sign():
    # Arbitrary explicitly supplied coefficients test equations, not hardware.
    p = RingParameters()
    supplied = dict(effective_mode_area_m2=1e-13, tpa_coefficient_m_per_w=1e-11, kerr_index_m2_per_w=4e-18)
    a = nonlinear_screening(p, p.reference_frequency_hz, input_power_w=1e-3, **supplied)
    b = nonlinear_screening(p, p.reference_frequency_hz, input_power_w=2e-3, **supplied)
    np.testing.assert_allclose(b.estimated_tpa_power_w, 4 * a.estimated_tpa_power_w)
    np.testing.assert_allclose(b.estimated_kerr_frequency_shift_hz, 2 * a.estimated_kerr_frequency_shift_hz)
    np.testing.assert_allclose(b.tpa_decay_to_loaded_decay_ratio, 2 * a.tpa_decay_to_loaded_decay_ratio)
    assert a.estimated_kerr_frequency_shift_hz < 0
    assert a.estimated_tpa_pair_generation_per_s > 0


def test_nonlinear_screening_integrates_distributed_squared_power():
    from scipy.integrate import simpson
    p = RingParameters(intrinsic_loss_per_m=900, drop_coupling_power=0.2)
    area, beta = 2e-13, 7e-12
    r = nonlinear_screening(p, p.reference_frequency_hz + 3e9, effective_mode_area_m2=area,
                            tpa_coefficient_m_per_w=beta, kerr_index_m2_per_w=0, input_power_w=1e-3)
    z = np.linspace(0, p.roundtrip_length_m / 2, 1001)
    first = r.peak_circulating_power_w * np.exp(-p.intrinsic_loss_per_m * z)
    second = first * np.exp(-p.intrinsic_loss_per_m * p.roundtrip_length_m / 2) * (1 - p.drop_coupling_power)
    numerical = beta / area * (simpson(first**2, x=z) + simpson(second**2, x=z))
    np.testing.assert_allclose(r.estimated_tpa_power_w, numerical, rtol=1e-12)
    assert r.estimated_kerr_frequency_shift_hz == 0


def test_nonlinear_screening_zero_power_lossless_limit_and_input_bounds():
    p = replace(RingParameters(), intrinsic_loss_per_m=0)
    supplied = dict(effective_mode_area_m2=1e-13, tpa_coefficient_m_per_w=1e-11, kerr_index_m2_per_w=4e-18)
    zero = nonlinear_screening(p, p.reference_frequency_hz, input_power_w=0, **supplied)
    assert zero.estimated_tpa_power_w == zero.estimated_kerr_frequency_shift_hz == 0
    assert zero.tpa_to_linear_intrinsic_loss_ratio == 0
    light = nonlinear_screening(p, p.reference_frequency_hz, input_power_w=1e-3, **supplied)
    assert np.isinf(light.tpa_to_linear_intrinsic_loss_ratio)
    assert np.isfinite(light.tpa_decay_to_loaded_decay_ratio)
    for key, value in (("effective_mode_area_m2", 0), ("tpa_coefficient_m_per_w", -1), ("kerr_index_m2_per_w", np.nan)):
        with pytest.raises(ValueError):
            nonlinear_screening(p, p.reference_frequency_hz, **(supplied | {key: value}))
