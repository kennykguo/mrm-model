"""Known synthetic systems test fitting; these are not device validation data."""

import numpy as np
import pytest

from scipy.integrate import solve_ivp

from ringlab.calibration import ThermalStepTrace, fit_network_scales, fit_single_pole, load_step_csv, predict_step
from ringlab.thermal import ThermalNetwork, default_ring_network


def synthetic_trace(dataset_id, power, gain=1700.0, tau=17e-6):
    times = np.r_[0, np.geomspace(1e-7, 2e-4, 80)]
    rise = power * gain * (-np.expm1(-times / tau))
    return ThermalStepTrace(times, rise, [power], dataset_id,
                            "synthetic analytic first-order thermal system; test only")


def test_measured_data_path_recovers_known_synthetic_rc_and_checks_heldout():
    training = synthetic_trace("synthetic training run", 1e-3)
    validation = synthetic_trace("synthetic heldout run", 0.4e-3)
    result = fit_single_pole(training, validation)
    assert result.solver_success
    assert result.training_rmse_k < 1e-10
    assert result.validation_rmse_k < 1e-10
    np.testing.assert_allclose(1 / result.network.conductance_w_per_k[0, 0], 1700, rtol=1e-8)
    tau = result.network.capacity_j_per_k[0] / result.network.conductance_w_per_k[0, 0]
    np.testing.assert_allclose(tau, 17e-6, rtol=1e-8)
    np.testing.assert_allclose(predict_step(result.network, validation), validation.temperature_rise_k,
                               atol=1e-10)


def test_validation_is_not_used_to_choose_the_fit():
    training = synthetic_trace("training", 1e-3)
    validation = synthetic_trace("heldout inconsistent synthetic process", 0.4e-3, gain=2000)
    result = fit_single_pole(training, validation)
    assert result.training_rmse_k < 1e-10
    assert result.validation_rmse_k > 0.03
    np.testing.assert_allclose(1 / result.network.conductance_w_per_k[0, 0], 1700, rtol=1e-8)


def test_reusing_training_acquisition_as_validation_is_rejected():
    trace = synthetic_trace("same acquisition", 1e-3)
    with pytest.raises(ValueError, match="separate"):
        fit_single_pole(trace, trace)


def test_csv_requires_explicit_baseline_and_source_provenance(tmp_path):
    path = tmp_path / "synthetic.csv"
    path.write_text("time_s,temperature_k\n0,300\n1,301\n2,302\n3,303\n4,304\n")
    trace = load_step_csv(path, power_step_w=[1e-3], baseline_temperature_k=300,
                          dataset_id="synthetic CSV fixture", provenance="synthetic test only")
    np.testing.assert_array_equal(trace.temperature_rise_k[:, 0], [0, 1, 2, 3, 4])


def test_calibrated_named_network_runs_in_the_full_plant():
    # Independent time integration generates synthetic observations. No device
    # dataset is invented: this verifies the calibration-to-plant interface.
    from ringlab.config import load
    from ringlab.plant import Plant, simulate

    template = default_ring_network(1)
    true_network = ThermalNetwork(template.capacity_j_per_k * 1.8,
                                   template.conductance_w_per_k * 0.7, template.source_map,
                                   template.boundary_conductance_w_per_k * 0.7,
                                   template.observer_matrix, template.node_names,
                                   template.source_names, template.observer_names)
    times = np.r_[0, np.geomspace(1e-7, 2e-4, 80)]

    def acquisition(name, power):
        powers = np.array([power, 0, 0])
        solution = solve_ivp(lambda _t, state: true_network.derivative(state, powers, 300),
                              (0, times[-1]), np.full(3, 300.0), t_eval=times,
                              rtol=1e-11, atol=1e-11)
        observations = (true_network.observer_matrix @ solution.y).T - 300
        return ThermalStepTrace(times, observations, powers, name,
                                 "synthetic RK45 trajectory used solely to verify integration")

    result = fit_network_scales(template, acquisition("synthetic training", 0.3e-3),
                                 acquisition("synthetic heldout", 0.1e-3))
    np.testing.assert_allclose([result.capacity_scale, result.conductance_scale], [1.8, 0.7], rtol=1e-6)
    config = load(overrides={"simulation.duration_s": 20e-6,
                             "simulation.record_period_s": 5e-6,
                             "controller.name": "hold", "controller.parameters": {"code": 1000},
                             "sensor.noise_enabled": False, "data.averaging": "cw"})
    plant = Plant(config, thermal_network=result.network)
    assert plant.thermal is result.network
    calibrated = simulate(config, thermal_network=result.network)
    known = simulate(config, thermal_network=true_network)
    np.testing.assert_allclose(calibrated.trace["ring_temperature_k"],
                               known.trace["ring_temperature_k"], rtol=0, atol=1e-6)
