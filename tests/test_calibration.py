"""Synthetic data recovery: these are algorithm checks, not hardware validation."""

from dataclasses import replace

import numpy as np
import pytest

from mrm_model.calibration import fit_optical_spectrum, fit_thermal_step
from mrm_model.optics import OpticalParams, optical_response


def test_synthetic_both_port_spectrum_parameter_recovery_and_holdout():
    truth = OpticalParams(
        reference_frequency_hz=OpticalParams().reference_frequency_hz + 7e9,
        loss_power_np_per_m=700,
        coupling_bus_power=0.045,
        coupling_drop_power=0.012,
    )
    f = truth.reference_frequency_hz + np.linspace(-150e9, 150e9, 501)
    spectrum = optical_response(truth, f, input_power_w=0.002)
    initial = replace(
        truth,
        reference_frequency_hz=truth.reference_frequency_hz - 4e9,
        loss_power_np_per_m=1100,
        coupling_bus_power=0.06,
        coupling_drop_power=0.008,
    )
    result = fit_optical_spectrum(
        initial, f, spectrum.through_power_w, drop_power_w=spectrum.drop_power_w, input_power_w=0.002
    )
    assert result.success
    assert result.params.reference_frequency_hz == pytest.approx(truth.reference_frequency_hz, abs=100)
    assert result.params.loss_power_np_per_m == pytest.approx(truth.loss_power_np_per_m, rel=1e-5)
    assert result.params.coupling_bus_power == pytest.approx(truth.coupling_bus_power, rel=1e-5)
    assert result.params.coupling_drop_power == pytest.approx(truth.coupling_drop_power, rel=1e-5)
    assert result.training_rmse_w < 1e-10
    assert result.validation_rmse_w < 1e-10
    assert not set(result.training_indices) & set(result.validation_indices)
    assert len(result.validation_indices) > 50
    assert result.covariance.shape == (4, 4)
    assert any("not an independent" in warning for warning in result.identifiability_warnings)


def test_through_only_fit_warns_of_loss_coupling_ambiguity():
    p = OpticalParams()
    f = p.reference_frequency_hz + np.linspace(-150e9, 150e9, 101)
    r = optical_response(p, f)
    fit = fit_optical_spectrum(p, f, r.through_power_w)
    assert any("Through-only" in warning for warning in fit.identifiability_warnings)
    assert fit.covariance is None
    assert all(value is None for value in fit.standard_errors.values())


def test_synthetic_two_mode_thermal_step_recovery_with_late_holdout():
    t = np.linspace(0, 180e-6, 500)
    true_tau = np.array([4e-6, 30e-6])
    amplitude = np.array([-8e9, -3e9])
    y = 2e8 + np.sum(amplitude * -np.expm1(-t[:, None] / true_tau), axis=1)
    fit = fit_thermal_step(t, y, n_modes=2, step_power_w=0.002)
    assert fit.success
    np.testing.assert_allclose(fit.tau_s, true_tau, rtol=1e-6)
    np.testing.assert_allclose(fit.amplitudes_hz, amplitude, rtol=1e-6)
    np.testing.assert_allclose(fit.gains_hz_per_w, amplitude / 0.002, rtol=1e-6)
    np.testing.assert_allclose(fit.predict(t), y, rtol=1e-6, atol=100)
    assert fit.validation_rmse_hz < 100
    assert np.max(fit.training_indices) < np.min(fit.validation_indices)


def test_noisy_synthetic_thermal_fit_has_nonzero_covariance_and_holdout():
    rng = np.random.default_rng(98)
    t = np.linspace(0, 100e-6, 300)
    y = -10e9 * -np.expm1(-t / 8e-6) + rng.normal(0, 10e6, len(t))
    fit = fit_thermal_step(t, y, n_modes=1, step_power_w=0.001)
    assert fit.tau_s[0] == pytest.approx(8e-6, rel=0.02)
    assert fit.training_rmse_hz > 1e6
    assert fit.validation_rmse_hz > 1e6
    assert np.all(np.diag(fit.covariance) > 0)


def test_invalid_calibration_data_rejected():
    with pytest.raises(ValueError):
        fit_thermal_step([0, 1], [0, 1])
    with pytest.raises(ValueError):
        fit_thermal_step(np.arange(30), np.zeros(30), step_power_w=-1)
    p = OpticalParams()
    with pytest.raises(ValueError):
        fit_optical_spectrum(p, np.arange(100), -np.ones(100))
