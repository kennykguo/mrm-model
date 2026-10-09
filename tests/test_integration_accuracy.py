"""Independent physical checks of event integration, heat accounting, and noise.

All configurations are synthetic verification cases, not target-device data.
"""

import numpy as np
from scipy.integrate import quad

from ringlab.config import load
from ringlab.plant import ELECTRON_CHARGE_C, Plant, simulate


def quiet_configuration(**overrides):
    base = {
        "simulation.duration_s": 20e-6,
        "simulation.record_period_s": 20e-6,
        "sensor.sample_period_s": 20e-6,
        "simulation.control_period_s": 20e-6,
        "controller.name": "hold",
        "controller.parameters": {"code": 0},
        "sensor.noise_enabled": False,
        "data.averaging": "cw",
        "data.junction_heat_fraction": 0,
        "laser.power_per_line_w": 0,
    }
    base.update(overrides)
    return load(overrides=base)


def test_scheduled_scenario_has_right_limit_at_its_integer_tick():
    config = quiet_configuration(**{
        "scenario.kind": "ambient_step", "scenario.start_s": 10e-6,
        "scenario.ambient_step_k": 80,
    })
    plant = Plant(config)
    event_time = config.ticks(config.scenario.start_s) * config.simulation.tick_s
    # Literal 10e-6 and 10*1e-6 differ by one representable float. Scheduling
    # and scenario evaluation must share the same canonical event boundary.
    assert plant.environment(np.nextafter(event_time, -np.inf))[0] == 300
    assert plant.environment(event_time)[0] == 380


def test_endpoint_discontinuity_does_not_heat_the_preceding_interval():
    config = quiet_configuration(**{
        "scenario.kind": "ambient_step", "scenario.start_s": 8e-6,
        "scenario.ambient_step_k": 80,
    })
    plant = Plant(config)
    plant.advance(0, config.ticks(config.scenario.start_s))
    np.testing.assert_allclose(plant.state[:plant.thermal.n_nodes], 300, rtol=0, atol=1e-10)
    plant.advance(config.ticks(config.scenario.start_s), config.ticks(12e-6))
    assert plant.state[plant.package_node] > 300


def test_discontinuous_optical_power_is_not_smeared_into_interval_averages():
    config = quiet_configuration(**{
        "data.averaging": "quasistatic", "laser.power_per_line_w": 0.5e-3,
        "ring.absorption_fraction": 0, "scenario.kind": "power_step",
        "scenario.start_s": 10e-6, "scenario.power_step_fraction": -1,
    })
    plant = Plant(config)
    before = plant.optical(0, plant.state[:plant.thermal.n_nodes]).oma_w
    result = simulate(config)
    assert before[0] > 0
    np.testing.assert_allclose(result.metrics["mean_oma_w"], before / 2, rtol=1e-6, atol=0)


def test_oma_integral_matches_independent_quadrature_during_resonance_sweep():
    config = quiet_configuration(**{
        "data.averaging": "quasistatic", "laser.power_per_line_w": 0.5e-3,
        "ring.absorption_fraction": 0, "array.cold_offset_hz": 100e9,
        "scenario.kind": "laser_drift", "scenario.start_s": 0,
        "scenario.laser_drift_hz_per_s": 1e16,
    })
    static_thermal_plant = Plant(config)
    temperature = static_thermal_plant.state[:static_thermal_plant.thermal.n_nodes].copy()
    duration = config.simulation.duration_s
    integral, _ = quad(lambda time: static_thermal_plant.optical(time, temperature).oma_w[0],
                       0, duration, epsabs=1e-18, epsrel=1e-9, limit=200)
    coarse_records = simulate(config)
    dense_records = simulate(load(config, {"simulation.record_period_s": 1e-6}))
    np.testing.assert_allclose(coarse_records.metrics["mean_oma_w"], [integral / duration], rtol=1e-5)
    np.testing.assert_allclose(dense_records.metrics["mean_oma_w"], [integral / duration], rtol=1e-5)


def test_dropout_interval_includes_start_and_excludes_end_integer_ticks():
    config = quiet_configuration(**{
        "sensor.sample_period_s": 5e-6, "sensor.latency_s": 0,
        "simulation.record_period_s": 1e-6,
        "scenario.dropout_start_s": 10e-6, "scenario.dropout_duration_s": 5e-6,
    })
    result = simulate(config)
    ticks = np.rint(result.trace["time_s"] / config.simulation.tick_s).astype(int)
    sample_ticks = np.rint(result.trace["sample_time_s"] / config.simulation.tick_s).astype(int)
    assert sample_ticks[ticks == 5].item() == 5
    assert sample_ticks[ticks == 10].item() == 5
    assert sample_ticks[ticks == 15].item() == 15


def test_coupled_rhs_accounts_for_every_deposited_heat_source_once():
    config = quiet_configuration(**{
        "array.count": 2, "data.averaging": "quasistatic",
        "laser.power_per_line_w": 0.5e-3, "data.junction_heat_fraction": 0.6,
        "data.driver_power_w": 0.2e-3, "scenario.kind": "local_triangle",
        "scenario.start_s": 0, "scenario.aggressor_period_s": 40e-6,
        "scenario.aggressor_power_w": 0.7e-3,
    })
    plant = Plant(config)
    n, nt = plant.n, plant.thermal.n_nodes
    state = plant.state.copy()
    state[:nt] = np.linspace(300, 310, nt)
    state[nt:nt + n] = [0.2, 0.3]
    time = 10e-6
    optical = plant.optical(time, state[:nt])
    boundary, _, _, external = plant.environment(time)
    expected = (plant.heater_power(state[:nt], state[nt:nt + n]).sum()
                + optical.absorbed_power_w.sum() + n * plant.junction_heat_w
                + n * config.data.driver_power_w + external.sum()
                + np.sum(plant.thermal.boundary_conductance_w_per_k[:, 0] * (boundary - state[:nt])))
    actual = plant.thermal.capacity_j_per_k @ plant.rhs(time, state)[:nt]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)


def test_accumulated_heater_energy_matches_settling_voltage_integral():
    config = quiet_configuration(**{
        "heater.resistance_temperature_coefficient_per_k": 0,
        "simulation.rtol": 1e-9, "simulation.temperature_atol_k": 1e-10,
        "simulation.voltage_atol_v": 1e-11,
    })
    plant = Plant(config)
    plant.action(np.array([2**config.heater.bits - 1], dtype=np.int64))
    end = 8e-6
    plant.advance(0, config.ticks(end))
    tau = config.heater.settling_time_s
    final_power = config.heater.voltage_max_v**2 / config.heater.resistance_ohm
    expected = final_power * (end - 2 * tau * (-np.expm1(-end / tau))
                              + tau / 2 * (-np.expm1(-2 * end / tau)))
    # This integral is independent of the thermal transient when TCR=0.
    energy_index = plant.thermal.n_nodes + 2 * plant.n
    np.testing.assert_allclose(plant.state[energy_index], expected, rtol=1e-7, atol=1e-15)


def test_stationary_sensor_noise_matches_one_sided_psd_and_filter_correlation():
    config = quiet_configuration(**{
        "sensor.noise_enabled": True, "sensor.sample_period_s": 5e-6,
        "sensor.filter_time_s": 20e-6, "sensor.dark_current_a": 100e-6,
        "sensor.input_noise_a_per_sqrt_hz": 1e-9,
    })
    plant = Plant(config)
    values = []
    sample_ticks = config.ticks(config.sensor.sample_period_s)
    for i in range(21000):
        plant.sample((i + 1) * sample_ticks)
        if i >= 1000:  # Remove deterministic zero-noise startup.
            values.append(plant.noise_current_a[0])
    values = np.asarray(values)
    expected_psd = (2 * ELECTRON_CHARGE_C * config.sensor.dark_current_a
                    + config.sensor.input_noise_a_per_sqrt_hz**2)
    expected_variance = expected_psd / (4 * config.sensor.filter_time_s)
    expected_rho = np.exp(-config.sensor.sample_period_s / config.sensor.filter_time_s)
    np.testing.assert_allclose(np.var(values), expected_variance, rtol=0.08)
    measured_rho = np.corrcoef(values[:-1], values[1:])[0, 1]
    np.testing.assert_allclose(measured_rho, expected_rho, rtol=0, atol=0.025)
