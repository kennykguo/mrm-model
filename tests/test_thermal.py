import numpy as np
import pytest
from scipy.integrate import solve_ivp

from mrm_model.thermal import ThermalNetwork, ThermalParams, default_thermal_params


def test_one_node_matches_closed_form_and_ambient_retains_memory():
    params = ThermalParams([2.0], [[0.5]], [[1.0]], [[1.0]])
    network = ThermalNetwork(params, 300.0)
    result = network.step(1.0, [2.0], 310.0)
    np.testing.assert_allclose(result, [314 + (300 - 314) * np.exp(-0.25)])
    previous = result.copy()
    np.testing.assert_allclose(network.step(0.0, [0], 280), previous)
    result = network.step(0.5, [0.0], 280.0)
    np.testing.assert_allclose(result, [280 + (previous[0] - 280) * np.exp(-0.125)])


def test_coupled_graph_matches_independent_ode_and_zoh_composes():
    params = default_thermal_params(3)
    network = ThermalNetwork(params)
    power = np.array([0.001, 0.0002, 0.0])
    forcing = params.heater_map @ power
    capacitance = params.capacitance_matrix
    reference = solve_ivp(
        lambda t, x: np.linalg.solve(capacitance, -params.conductance_w_per_k @ x + forcing),
        (0, 70e-6),
        np.zeros(params.n_states),
        rtol=1e-10,
        atol=1e-12,
    )
    network.step(70e-6, power, 300.0)
    np.testing.assert_allclose(network.state_rise_k, reference.y[:, -1], rtol=1e-8, atol=1e-10)
    repeated = ThermalNetwork(params)
    for dt in [1e-6, 2e-6, 7e-6, 20e-6, 40e-6]:
        repeated.step(dt, power, 300.0)
    np.testing.assert_allclose(repeated.state_rise_k, network.state_rise_k, rtol=1e-11)
    assert network.ring_temperatures_k[1] > 300
    assert network.ring_temperatures_k[2] > 300


def test_uniform_boundary_equilibrium_and_reciprocal_transfer():
    params = default_thermal_params(3)
    network = ThermalNetwork(params)
    np.testing.assert_allclose(network.equilibrium([0, 0, 0], 345), 345)
    thermal_impedance = np.linalg.inv(params.conductance_w_per_k)
    np.testing.assert_allclose(thermal_impedance, thermal_impedance.T, rtol=1e-12)
    assert np.all(network.modes()["time_constants_s"] > 0)


def test_unforced_energy_metric_decays():
    network = ThermalNetwork(default_thermal_params(2), 330)
    previous = network.stored_energy_metric()
    for dt in [1e-7, 1e-6, 1e-5, 1e-3]:
        network.step(dt, [0, 0], 300)
        energy = network.stored_energy_metric()
        assert energy < previous
        previous = energy


def test_generalized_congruence_rom_matches_physical_model():
    physical = default_thermal_params(1)
    transform = np.array([[1.0, 0.2, -0.3], [0.1, 1.2, 0.1], [0.0, -0.2, 0.9]])
    uniform = np.linalg.solve(transform, np.ones(3))
    rom_params = ThermalParams(
        transform.T @ physical.capacitance_matrix @ transform,
        transform.T @ physical.conductance_w_per_k @ transform,
        transform.T @ physical.heater_map,
        physical.ring_map @ transform,
        transform.T @ physical.boundary_map_w_per_k,
        heater_temperature_map=physical.heater_temperature_map @ transform,
        reference_state_map=uniform,
        physical_network=False,
    )
    nodal = ThermalNetwork(physical)
    rom = ThermalNetwork(rom_params)
    for ambient, power in [(320, 0.001), (310, 0.002), (290, 0)]:
        np.testing.assert_allclose(
            rom.step(20e-6, [power], ambient), nodal.step(20e-6, [power], ambient), rtol=1e-12
        )
    reconstructed = ThermalParams(**rom_params.to_dict())
    np.testing.assert_allclose(reconstructed.conductance_w_per_k, rom_params.conductance_w_per_k)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capacitance_j_per_k": [-1.0]},
        {"conductance_w_per_k": [[-1.0]]},
        {"boundary_map_w_per_k": [[0.0]]},
        {"ring_map": [[0.5]]},
        {"heater_map": [[2.0]]},
    ],
)
def test_invalid_passive_physics_rejected(kwargs):
    base = {
        "capacitance_j_per_k": [1.0],
        "conductance_w_per_k": [[1.0]],
        "heater_map": [[1.0]],
        "ring_map": [[1.0]],
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        ThermalParams(**base)


def test_multiple_independent_boundaries():
    params = ThermalParams([1.0], [[3.0]], [[1.0]], [[1.0]], [[1.0, 2.0]])
    network = ThermalNetwork(params)
    np.testing.assert_allclose(network.equilibrium([0], [300.0, 330.0]), [320.0])
