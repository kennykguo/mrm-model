import numpy as np
import pytest

from mrm_model.optics import NonlinearParams, OpticalParams, nonlinear_steady_states, optical_response
from mrm_model.waveform import simulate_cavity, simulate_nrz


def test_unforced_cavity_energy_decays_at_logarithmic_roundtrip_rate():
    p = OpticalParams()
    time = np.linspace(0, 8 * p.photon_lifetime_s, 150)
    energy = 2e-15
    r = simulate_cavity(p, time, input_power_w=0, initial_forward_field=np.sqrt(energy))
    np.testing.assert_allclose(r["cavity_energy_j"], energy * np.exp(-time / p.photon_lifetime_s), rtol=2e-13)


def test_static_dynamic_agreement_in_high_finesse_limit():
    p = OpticalParams(coupling_bus_power=0.0008, coupling_drop_power=0.0004, loss_power_np_per_m=2.5)
    time = np.linspace(0, 20 * p.photon_lifetime_s, 150)
    for detuning in (-p.linewidth_hz, 0, p.linewidth_hz):
        f = p.reference_frequency_hz + detuning
        dynamic = simulate_cavity(p, time, laser_frequency_hz=f, input_power_w=0.001)
        static = optical_response(p, f, input_power_w=0.001)
        assert dynamic["through_power_w"][-1] == pytest.approx(
            float(static.through_power_w), rel=5e-4, abs=1e-10
        )
        assert dynamic["drop_power_w"][-1] == pytest.approx(float(static.drop_power_w), rel=5e-4)
        assert dynamic["cavity_energy_j"][-1] == pytest.approx(float(static.cavity_energy_j), rel=5e-4)


def test_backscattering_preserves_total_decay_and_creates_reverse_mode():
    p = OpticalParams()
    time = np.linspace(0, 4 * p.photon_lifetime_s, 120)
    energy = 1e-15
    r = simulate_cavity(
        p, time, input_power_w=0, initial_forward_field=np.sqrt(energy), backscatter_rad_per_s=3e11 + 1e11j
    )
    np.testing.assert_allclose(r["cavity_energy_j"], energy * np.exp(-time / p.photon_lifetime_s), rtol=3e-13)
    assert np.max(r["reverse_energy_j"]) > energy / 10
    assert np.max(r["reflected_power_w"]) > 0


def test_nrz_rc_cavity_receiver_and_proxy_metrics_are_explicit():
    p = OpticalParams()
    bits = np.random.default_rng(7).integers(0, 2, 100)
    r = simulate_nrz(
        p,
        bits,
        samples_per_bit=16,
        laser_frequency_hz=p.reference_frequency_hz + 20e9,
        receiver_noise_std_w=1e-6,
    )
    assert len(r["time_s"]) == 1601
    assert np.all(r["voltage_v"] >= -1) and np.all(r["voltage_v"] <= 0)
    assert np.all(np.isfinite(r["through_power_w"]))
    assert r["metrics"]["oma_w"] > 0
    assert 0 <= r["metrics"]["gaussian_ber_proxy"] <= 0.5
    assert "not predicted" in r["metrics"]["ber_interpretation"]
    assert not np.array_equal(r["receiver_power_w"], r["through_power_w"])


def test_nrz_time_step_convergence_is_nontrivial():
    p = OpticalParams()
    bits = np.random.default_rng(74).integers(0, 2, 50)
    coarse = simulate_nrz(p, bits, samples_per_bit=16, laser_frequency_hz=p.reference_frequency_hz + 20e9)
    fine = simulate_nrz(p, bits, samples_per_bit=32, laser_frequency_hz=p.reference_frequency_hz + 20e9)
    np.testing.assert_allclose(coarse["through_power_w"], fine["through_power_w"][::2], rtol=0.001, atol=3e-7)
    assert coarse["metrics"]["oma_w"] == pytest.approx(fine["metrics"]["oma_w"], rel=0.01)


def test_nonlinear_dynamic_state_generates_carriers_and_positive_absorption():
    p = OpticalParams()
    nl = NonlinearParams(
        tpa_rate_per_j_s=1e25, carrier_lifetime_s=20e-12, fca_cross_section_m2=1e-21, fcd_hz_per_m3=1e-13
    )
    time = np.linspace(0, 50e-12, 100)
    r = simulate_cavity(p, time, nonlinearity=nl, input_power_w=0.003)
    assert r["carrier_density_m3"][-1] > 0
    assert np.all(r["carrier_density_m3"] >= 0)
    assert np.all(r["tpa_absorbed_power_w"] >= 0)
    assert np.all(r["fca_absorbed_power_w"] >= 0)
    assert r["cavity_energy_j"][-1] < simulate_cavity(p, time, input_power_w=0.003)["cavity_energy_j"][-1]


def test_waveform_memory_limit_precedes_allocating_large_waveform():
    p = OpticalParams()
    with pytest.raises(ValueError, match="max_samples"):
        simulate_nrz(p, np.zeros(100), samples_per_bit=32, max_samples=1000)
    with pytest.raises(ValueError):
        simulate_cavity(p, [0, 0, 1e-12])


def test_nonlinear_dynamic_converges_to_stable_carrier_equilibrium():
    p = OpticalParams()
    nl = NonlinearParams(
        tpa_rate_per_j_s=1e25, carrier_lifetime_s=20e-12, fca_cross_section_m2=1e-21, fcd_hz_per_m3=1e-13
    )
    states = nonlinear_steady_states(p, p.reference_frequency_hz, 0.003, nonlinearity=nl)
    assert len(states) == 1 and states[0].stable
    r = simulate_cavity(p, np.linspace(0, 400e-12, 101), input_power_w=0.003, nonlinearity=nl)
    assert r["cavity_energy_j"][-1] == pytest.approx(states[0].cavity_energy_j, rel=1e-5)
    assert r["carrier_density_m3"][-1] == pytest.approx(states[0].carrier_density_m3, rel=1e-5)
