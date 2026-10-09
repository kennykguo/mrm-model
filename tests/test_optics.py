"""Physical invariants, not assertions that illustrative parameters match a device."""

from dataclasses import replace

import numpy as np
import pytest

from mrm_model.optics import (
    C,
    NonlinearParams,
    OpticalParams,
    cavity_rhs,
    nonlinear_steady_states,
    optical_response,
    select_nonlinear_branch,
)


def test_add_drop_conserves_energy_and_absorption_split():
    rng = np.random.default_rng(714)
    for _ in range(12):
        p = OpticalParams(
            coupling_bus_power=rng.uniform(0.001, 0.8),
            coupling_drop_power=rng.uniform(0, 0.6),
            loss_power_np_per_m=rng.uniform(0, 30000),
            intrinsic_absorption_fraction=0.37,
        )
        f = p.reference_frequency_hz + np.linspace(-p.fsr_hz, p.fsr_hz, 301)
        r = optical_response(p, f, input_power_w=0.002)
        np.testing.assert_allclose(
            r.through_power_w + r.drop_power_w + r.absorbed_power_w + r.scattered_power_w,
            0.002,
            atol=2e-16,
            rtol=1e-11,
        )
        assert np.all(r.cavity_energy_j >= 0)
        np.testing.assert_allclose(r.absorbed_power_w, 0.37 * r.dissipated_power_w)


def test_lossless_all_pass_and_drop_ring():
    p = OpticalParams(loss_power_np_per_m=0, coupling_drop_power=0)
    f = p.reference_frequency_hz + np.linspace(-p.fsr_hz / 2, p.fsr_hz / 2, 151)
    r = optical_response(p, f)
    np.testing.assert_allclose(r.through_power_w, 1, atol=1e-13)
    np.testing.assert_allclose(r.drop_power_w, 0)
    p = replace(p, coupling_drop_power=p.coupling_bus_power)
    r = optical_response(p, p.reference_frequency_hz)
    assert r.drop_power_w == pytest.approx(1, abs=1e-12)
    assert r.through_power_w == pytest.approx(0, abs=1e-20)


def test_exact_periodicity_and_thermal_voltage_sign():
    p = OpticalParams()
    f = p.reference_frequency_hz + np.array([-40e9, -12e9, 11e9, 45e9])
    r = optical_response(p, f)
    shifted = optical_response(p, f + p.fsr_hz)
    np.testing.assert_allclose(shifted.through_power_w, r.through_power_w, atol=1e-12)
    heated = optical_response(p, f, temperature_k=p.reference_temperature_k + 2)
    assert np.all(heated.resonance_frequency_hz == p.reference_frequency_hz - 30e9)
    biased = optical_response(p, f, junction_voltage_v=-1)
    assert np.all(biased.resonance_frequency_hz == p.reference_frequency_hz - p.df_dvoltage_hz_per_v)


def test_critical_coupling_and_exact_cavity_energy():
    p = OpticalParams(coupling_drop_power=0, loss_power_np_per_m=2000)
    p = replace(p, coupling_bus_power=-np.expm1(-p.loss_power_np_per_m * p.length_m))
    r = optical_response(p, p.reference_frequency_hz, input_power_w=0.003)
    assert r.through_power_w == pytest.approx(0, abs=1e-27)
    assert r.absorbed_power_w == pytest.approx(0.003, rel=1e-12)
    assert r.cavity_energy_j == pytest.approx(0.003 / p.kappa_intrinsic_s_inv)
    assert p.linewidth_hz == pytest.approx(p.kappa_total_s_inv / (2 * np.pi))
    assert p.loaded_q == pytest.approx(2 * np.pi * p.reference_frequency_hz * p.photon_lifetime_s)


def test_isolated_ring_is_not_singular():
    p = OpticalParams(coupling_bus_power=0, coupling_drop_power=0, loss_power_np_per_m=0)
    r = optical_response(p, p.reference_frequency_hz)
    assert r.through_power_w == 1
    assert r.cavity_energy_j == 0


def test_voltage_calibration_tables_reject_extrapolation():
    p = OpticalParams(
        voltage_points_v=(-2, -1, 0),
        resonance_voltage_shifts_hz=(-20e9, -8e9, 0),
        loss_voltage_values_np_per_m=(800, 950, 1000),
    )
    r = optical_response(p, p.reference_frequency_hz, junction_voltage_v=-1.5)
    assert r.resonance_frequency_hz == p.reference_frequency_hz - 14e9
    with pytest.raises(ValueError, match="outside"):
        optical_response(p, p.reference_frequency_hz, junction_voltage_v=-3)


def test_dynamic_backscatter_is_conservative_and_reciprocal():
    p = OpticalParams(loss_power_np_per_m=0, coupling_bus_power=0, coupling_drop_power=0)
    a, b = 3e-8 + 1e-8j, 2e-8 - 4e-8j
    da, db, _ = cavity_rhs(
        p, a, b, 0, laser_frequency_hz=p.reference_frequency_hz, backscatter_rad_per_s=1e11 + 3e10j
    )
    energy_rate = 2 * np.real(np.conj(a) * da + np.conj(b) * db)
    assert energy_rate == pytest.approx(0, abs=1e-19)


def test_nonlinear_branches_expose_stability_and_require_history():
    p = OpticalParams(loss_power_np_per_m=0, coupling_bus_power=0.002, coupling_drop_power=0.001)
    power = 0.001
    k = p.kappa_total_s_inv
    scale = p.kappa_bus_s_inv * power / (k / 2) ** 2
    nl = NonlinearParams(kerr_hz_per_j=-10 * k / (2 * np.pi * scale))
    f = p.reference_frequency_hz - 3 * k / (2 * np.pi)
    branches = nonlinear_steady_states(p, f, power, nonlinearity=nl)
    assert len(branches) == 3
    assert [branch.stable for branch in branches] == [True, False, True]
    assert max(branch.relative_residual for branch in branches) < 1e-10
    with pytest.raises(ValueError, match="multiple stable"):
        select_nonlinear_branch(branches)
    assert select_nonlinear_branch(branches, previous_energy_j=0) == branches[0]
    assert select_nonlinear_branch(branches, previous_energy_j=branches[-1].cavity_energy_j) == branches[-1]


def test_tpa_carrier_generation_uses_two_photons_and_volume():
    p = OpticalParams()
    nl = NonlinearParams(tpa_rate_per_j_s=1e24, carrier_volume_m3=3e-18)
    a = 1e-8 + 2e-8j
    _, _, dn = cavity_rhs(p, a, 0j, 0, laser_frequency_hz=p.reference_frequency_hz, nonlinearity=nl)
    from mrm_model.optics import H_PLANCK

    expected = (
        nl.tpa_rate_per_j_s * abs(a) ** 4 / (2 * H_PLANCK * p.reference_frequency_hz * nl.carrier_volume_m3)
    )
    assert dn == pytest.approx(expected)


def test_nonlinear_tcmt_instantaneous_energy_balance_with_both_modes():
    p = OpticalParams(intrinsic_absorption_fraction=0.2)
    nl = NonlinearParams(
        tpa_rate_per_j_s=1e25, fca_cross_section_m2=1e-21, fcd_hz_per_m3=1e-13, kerr_hz_per_j=-1e21
    )
    a, b, s, density = 2e-8 + 1e-8j, 1e-8 - 2e-8j, 0.04 + 0.01j, 1e20
    da, db, _ = cavity_rhs(
        p,
        a,
        b,
        density,
        laser_frequency_hz=p.reference_frequency_hz + 9e9,
        input_field=s,
        backscatter_rad_per_s=3e10 + 1e10j,
        nonlinearity=nl,
    )
    U = abs(a) ** 2 + abs(b) ** 2
    through = abs(s - np.sqrt(p.kappa_bus_s_inv) * a) ** 2
    reverse = p.kappa_bus_s_inv * abs(b) ** 2
    both_drops = p.kappa_drop_s_inv * U
    dissipation = (
        p.kappa_intrinsic_s_inv
        + nl.tpa_rate_per_j_s * U
        + nl.fca_cross_section_m2 * C / p.group_index * density
    ) * U
    dU = 2 * np.real(np.conj(a) * da + np.conj(b) * db)
    assert through + reverse + both_drops + dissipation + dU == pytest.approx(abs(s) ** 2, rel=1e-12)


def test_invalid_loss_and_inputs_are_rejected():
    with pytest.raises(ValueError):
        OpticalParams(loss_power_np_per_m=-1)
    with pytest.raises(ValueError):
        OpticalParams(coupling_bus_power=1)
    p = OpticalParams(loss_voltage_np_per_m_per_v=2000)
    with pytest.raises(ValueError, match="loss"):
        optical_response(p, p.reference_frequency_hz, junction_voltage_v=-1)
    with pytest.raises(ValueError):
        optical_response(p, p.reference_frequency_hz, input_power_w=-1)


def test_dispersive_stored_energy_uses_local_group_delay_and_preserves_passivity():
    p = OpticalParams(phase_dispersion_s2=2e-26)
    f = p.reference_frequency_hz + np.array([-400e9, 0, 400e9, 2e12])
    r = optical_response(p, f, input_power_w=0.002)
    local_tau = p.roundtrip_time_s + p.phase_dispersion_s2 * 2 * np.pi * (f - p.reference_frequency_hz)
    np.testing.assert_allclose(
        r.absorbed_power_w, p.loss_power_np_per_m * p.length_m / local_tau * r.cavity_energy_j, rtol=1e-13
    )
    np.testing.assert_allclose(r.through_power_w + r.drop_power_w + r.dissipated_power_w, 0.002, atol=1e-16)
    # Holding phase fixed isolates the energy's frequency-dependent group delay.
    phi = (
        2 * np.pi * (f - p.reference_frequency_hz) * p.roundtrip_time_s
        + 0.5 * p.phase_dispersion_s2 * (2 * np.pi * (f - p.reference_frequency_hz)) ** 2
    )
    equivalent_frequency = p.reference_frequency_hz + phi / (2 * np.pi * p.roundtrip_time_s)
    reference = optical_response(replace(p, phase_dispersion_s2=0), equivalent_frequency, input_power_w=0.002)
    np.testing.assert_allclose(
        r.cavity_energy_j / reference.cavity_energy_j, local_tau / p.roundtrip_time_s, rtol=2e-10
    )


def test_dispersion_rejects_nonphysical_extrapolation():
    p = OpticalParams(phase_dispersion_s2=2e-26)
    with pytest.raises(ValueError, match="local group delay"):
        optical_response(p, p.reference_frequency_hz - 5e12)


def test_dispersive_exact_fwhm_matches_sampled_half_height():
    p = OpticalParams(phase_dispersion_s2=2e-25)
    rho = np.exp(-p.kappa_total_s_inv * p.roundtrip_time_s / 2)
    half_phase = 2 * np.arcsin((1 - rho) / (2 * np.sqrt(rho)))
    roots = [
        2 * phi / (p.roundtrip_time_s + np.sqrt(p.roundtrip_time_s**2 + 2 * p.phase_dispersion_s2 * phi))
        for phi in (-half_phase, half_phase)
    ]
    frequencies = p.reference_frequency_hz + np.asarray(roots) / (2 * np.pi)
    half = optical_response(p, frequencies).drop_power_w
    peak = optical_response(p, p.reference_frequency_hz).drop_power_w
    np.testing.assert_allclose(half, peak / 2, rtol=1e-10)
    assert p.exact_fwhm_hz == pytest.approx((roots[1] - roots[0]) / (2 * np.pi))
