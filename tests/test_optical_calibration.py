from dataclasses import replace

import numpy as np
import pytest

from ringlab.optics import RingParameters, optical_response
from ringlab.optical_calibration import SpectrumData, equivalent_power_parameters, fit_spectrum


def test_two_port_power_spectra_have_exact_discrete_ambiguity():
    p = RingParameters()
    other = equivalent_power_parameters(p)
    assert other is not None
    assert abs(other.intrinsic_loss_per_m - p.intrinsic_loss_per_m) > 50
    frequency = p.reference_frequency_hz + np.linspace(-0.5, 0.5, 1001) * p.fsr_hz
    a, b = optical_response(p, frequency), optical_response(other, frequency)
    np.testing.assert_allclose(a.through_fraction, b.through_fraction, atol=3e-14)
    np.testing.assert_allclose(a.drop_fraction, b.drop_fraction, atol=3e-14)
    assert np.max(np.abs(a.through_field - b.through_field)) > 0.1


def test_fit_recovers_loss_and_center_when_couplings_are_independently_fixed():
    p = RingParameters()
    frequency = p.reference_frequency_hz + np.linspace(-35, 35, 141) * 1e9
    truth = optical_response(replace(p, intrinsic_loss_per_m=330), frequency, resonance_shift_hz=3.2e9)
    data = SpectrumData(frequency, "Synthetic exact fit test; independently fixed couplings", through_fraction=truth.through_fraction, drop_fraction=truth.drop_fraction)
    train = np.arange(frequency.size) % 4 != 0
    fit = fit_spectrum(data, p, vary=("resonance_shift_hz", "intrinsic_loss_per_m"),
                       bounds={"resonance_shift_hz": (-10e9, 10e9), "intrinsic_loss_per_m": (10, 1000)},
                       training_mask=train, starts=3)
    assert fit.optimizer_success
    assert fit.fitted_values["intrinsic_loss_per_m"] == pytest.approx(330, rel=1e-5)
    assert fit.fitted_values["resonance_shift_hz"] == pytest.approx(3.2e9, rel=1e-5)
    assert max(fit.heldout_rmse_by_port.values()) < 1e-7
    assert fit.jacobian_rank == 2
    assert not fit.alternate_satisfies_fit_constraints
    assert "input_coupling_power" in fit.fixed_parameter_names


def test_fit_requires_explicit_holdout_bounds_and_data_provenance():
    p = RingParameters()
    frequency = np.linspace(p.reference_frequency_hz - 1e10, p.reference_frequency_hz + 1e10, 20)
    with pytest.raises(ValueError):
        SpectrumData(frequency, "", through_fraction=np.ones(20))
    data = SpectrumData(frequency, "Synthetic test", through_fraction=np.ones(20))
    with pytest.raises(ValueError, match="heldout"):
        fit_spectrum(data, p, vary=("intrinsic_loss_per_m",), bounds={"intrinsic_loss_per_m": (0, 1000)}, training_mask=np.ones(20, dtype=bool))
    with pytest.raises(ValueError, match="bounds"):
        fit_spectrum(data, p, vary=("intrinsic_loss_per_m",), bounds={}, training_mask=np.arange(20) % 2 == 0)
