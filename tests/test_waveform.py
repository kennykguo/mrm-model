from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import simpson

from ringlab.junction import DepletionJunction, MeasuredJunctionTable
from ringlab.optics import RingParameters, decay_rates, optical_response
from ringlab.waveform import eye_segments, nrz_average_response, nrz_drive, tcmt_response, waveform_response, wdm_nrz_average_response


def test_constant_optical_transient_has_correct_photon_lifetime():
    p, junction = RingParameters(), DepletionJunction()
    time = np.linspace(0, 300e-12, 1001)
    voltage = np.ones(time.shape)
    j = junction.evaluate(1)
    frequency = p.reference_frequency_hz + float(j.resonance_shift_hz)
    r = waveform_response(p, time, voltage, junction, frequency_hz=frequency, initial_field_sqrt_j=0)
    steady = tcmt_response(p, frequency, resonance_shift_hz=j.resonance_shift_hz, additional_loss_per_m=j.additional_loss_per_m, input_power_w=1e-3)
    k = sum(decay_rates(p, j.additional_loss_per_m))
    expected = steady.cavity_field_sqrt_j * (1 - np.exp(-0.5 * k * time))
    np.testing.assert_allclose(r.cavity_field_sqrt_j, expected, atol=1e-20)


def test_constant_rc_reaches_exact_voltage_step():
    junction = MeasuredJunctionTable(
        [0, 3], [0, 0], [0, 0], [40e-15, 40e-15], [100, 100], [0, 0], "synthetic constant RC validation"
    )
    p = RingParameters()
    time = np.linspace(0, 30e-12, 101)
    r = waveform_response(p, time, np.full(time.shape, 3), junction, frequency_hz=p.reference_frequency_hz, initial_voltage_v=0)
    np.testing.assert_allclose(r.junction_voltage_v, 3 * (1 - np.exp(-time / 4e-12)), atol=3e-14)
    np.testing.assert_allclose(r.series_resistor_heat_w, (3 - r.junction_voltage_v)**2 / 100)


def test_nrz_average_conserves_power_and_recovers_constant_level():
    p = RingParameters()
    frequency = p.reference_frequency_hz + np.linspace(-50, 50, 101) * 1e9
    for probability in (0, 0.2, 0.5, 1):
        r = nrz_average_response(p, frequency, resonance_shift_hz=(-20e9, 0), additional_loss_per_m=(-20, 10), mark_probability=probability)
        np.testing.assert_allclose(r.through_power_w + r.drop_power_w + r.intrinsic_loss_power_w, 1, atol=2e-14)
    r = nrz_average_response(p, frequency, resonance_shift_hz=(-20e9, -20e9), additional_loss_per_m=(-20, -20))
    steady = tcmt_response(p, frequency, resonance_shift_hz=-20e9, additional_loss_per_m=-20)
    np.testing.assert_allclose(r.through_power_w, steady.through_power_w, atol=2e-14)
    np.testing.assert_allclose(r.absorbed_power_w, steady.absorbed_power_w, atol=2e-14)
    np.testing.assert_allclose(r.oma_w, 0, atol=2e-14)


def test_nrz_stationary_moments_against_enumerated_bit_histories():
    # Enumerate all 12-bit histories independently, then quadrature the next bit.
    # Finite history has forgotten the empty cavity because each bit is long.
    p = RingParameters()
    rate, pin = 8e9, 1e-3
    frequency = p.reference_frequency_hz - 8e9
    shifts, losses = np.array([-20e9, 0]), np.array([-10, 15])
    ki, k1, k2 = decay_rates(p, losses)
    total = ki + k1 + k2
    detuning = 2 * np.pi * (frequency - p.reference_frequency_hz - shifts)
    fields = np.zeros(1, dtype=complex)
    steady = np.sqrt(k1 * pin) / (0.5 * total + 1j * detuning)
    for _ in range(12):
        fields = np.concatenate([steady[j] + (fields - steady[j]) * np.exp((-0.5 * total[j] - 1j * detuning[j]) / rate) for j in (0, 1)])
    fractions = np.linspace(0, 1, 257)
    through, drop, heat = [], [], []
    for j in (0, 1):
        a = steady[j] + (fields[:, None] - steady[j]) * np.exp((-0.5 * total[j] - 1j * detuning[j]) * fractions / rate)
        energy_mean = np.mean(np.abs(a)**2, axis=0)
        through.append(simpson(np.mean(np.abs(np.sqrt(pin) - np.sqrt(k1) * a)**2, axis=0), x=fractions))
        drop.append(simpson(k2 * energy_mean, x=fractions))
        heat.append(simpson(p.absorption_fraction * ki[j] * energy_mean, x=fractions))
    r = nrz_average_response(p, frequency, resonance_shift_hz=tuple(shifts), additional_loss_per_m=tuple(losses), bit_rate_hz=rate, input_power_w=pin)
    np.testing.assert_allclose(r.through_power_w, np.mean(through), rtol=3e-6)
    np.testing.assert_allclose(r.drop_power_w, np.mean(drop), rtol=3e-6)
    np.testing.assert_allclose(r.absorbed_power_w, np.mean(heat), rtol=3e-6)


def test_finite_rc_waveform_refines_with_substeps():
    p, junction = RingParameters(), DepletionJunction()
    time, drive = nrz_drive([0, 1, 0, 1, 1, 0], bit_rate_hz=50e9, samples_per_bit=8)
    kwargs = dict(frequency_hz=p.reference_frequency_hz - 12e9)
    coarse = waveform_response(p, time, drive, junction, substeps=1, **kwargs)
    middle = waveform_response(p, time, drive, junction, substeps=4, **kwargs)
    fine = waveform_response(p, time, drive, junction, substeps=16, **kwargs)
    assert np.max(np.abs(middle.through_power_w - fine.through_power_w)) < 0.15 * np.max(np.abs(coarse.through_power_w - fine.through_power_w))
    assert np.all(fine.absorbed_power_w >= 0)
    assert np.all(fine.junction_leakage_heat_w >= 0)


def test_nrz_drive_and_eye_shapes():
    time, drive = nrz_drive([0, 1, 1, 0], bit_rate_hz=50e9, samples_per_bit=16)
    assert time.shape == drive.shape == (65,)
    np.testing.assert_allclose(drive[:16], 0.5)
    np.testing.assert_allclose(drive[16:48], 2.5)
    assert eye_segments(drive, samples_per_bit=16, discard_bits=0).shape == (3, 33)


def test_waveform_input_validation():
    p = RingParameters()
    with pytest.raises(ValueError):
        waveform_response(p, [0, 0], [0, 1], DepletionJunction(), frequency_hz=p.reference_frequency_hz)
    with pytest.raises(ValueError):
        nrz_average_response(p, p.reference_frequency_hz, bit_rate_hz=-1)


def test_wdm_hybrid_conserves_power_and_has_correct_near_far_limits():
    p = RingParameters()
    shifts, losses = (-20e9, 0), (-10, 10)
    frequency = p.reference_frequency_hz - 10e9 + np.linspace(-1.5, 1.5, 1001) * p.fsr_hz
    r = wdm_nrz_average_response(p, frequency, resonance_shift_hz=shifts, additional_loss_per_m=losses)
    np.testing.assert_allclose(r.through_power_w + r.drop_power_w + r.intrinsic_loss_power_w, 1, atol=5e-13)
    assert np.all(r.absorbed_power_w >= 0)
    near_f = p.reference_frequency_hz - 10e9
    near = wdm_nrz_average_response(p, near_f, resonance_shift_hz=shifts, additional_loss_per_m=losses)
    dynamic = nrz_average_response(p, near_f, resonance_shift_hz=shifts, additional_loss_per_m=losses)
    np.testing.assert_allclose(near.absorbed_power_w, dynamic.absorbed_power_w)
    far_f = near_f + 0.4 * p.fsr_hz
    far = wdm_nrz_average_response(p, far_f, resonance_shift_hz=shifts, additional_loss_per_m=losses)
    exact = [optical_response(p, far_f, resonance_shift_hz=shifts[j], additional_loss_per_m=losses[j]) for j in (0, 1)]
    np.testing.assert_allclose(far.absorbed_power_w, np.mean([e.absorbed_power_w for e in exact]))
    assert near.diagnostics["dynamic_weight"] == 1
    assert far.diagnostics["dynamic_weight"] == 0


def test_wdm_hybrid_transition_is_continuous_and_periodic():
    p = RingParameters()
    for edge in (0.08, 0.12, 0.5):
        frequency = p.reference_frequency_hz + np.array([edge - 1e-9, edge, edge + 1e-9]) * p.fsr_hz
        r = wdm_nrz_average_response(p, frequency, resonance_shift_hz=(-10e9, 10e9))
        assert np.max(np.abs(np.diff(r.absorbed_power_w))) < 1e-8
    a = wdm_nrz_average_response(p, p.reference_frequency_hz)
    b = wdm_nrz_average_response(p, p.reference_frequency_hz + p.fsr_hz)
    np.testing.assert_allclose(a.absorbed_power_w, b.absorbed_power_w, atol=1e-12)


def test_wdm_hybrid_rejects_unmodeled_dispersive_comb():
    p = replace(RingParameters(), beta2_s2_per_m=1e-24)
    with pytest.raises(ValueError, match="dispersive comb"):
        wdm_nrz_average_response(p, p.reference_frequency_hz)
