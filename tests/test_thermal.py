"""Analytic physics checks; fixtures are synthetic, never target measurements."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from ringlab.thermal import ThermalNetwork, default_ring_network


def one_pole():
    return ThermalNetwork([2e-6], [[0.02]], [[1.0]], [[0.02]], [[1.0]])


def test_exact_zoh_matches_analytic_one_pole_and_semigroup():
    network = one_pole()
    duration = 7e-5
    expected = 300 + 0.01 / 0.02 * (-np.expm1(-duration / 1e-4))
    result = network.step_zoh([300], [0.01], 300, duration)
    np.testing.assert_allclose(result, [expected], rtol=1e-14)
    half = network.step_zoh([300], [0.01], 300, duration / 2)
    np.testing.assert_allclose(network.step_zoh(half, [0.01], 300, duration / 2), result)


def test_boundary_change_has_no_artificial_temperature_jump():
    network = one_pole()
    np.testing.assert_array_equal(network.step_zoh([300], [0], 310, 0), [300])
    np.testing.assert_allclose(network.derivative([300], [0], 310), [1e5])
    np.testing.assert_allclose(network.step_zoh([300], [0], 310, 1e-4),
                               [310 - 10 * np.exp(-1)])


def test_continuously_varying_boundary_matches_analytic_ramp():
    network = one_pole()
    rate = 808.0
    tau = 1e-4
    end = 5e-4
    solution = solve_ivp(lambda time, temperature: network.derivative(
        temperature, [0], 300 + rate * time), (0, end), [300], rtol=1e-10, atol=1e-10)
    expected = 300 + rate * (end - tau * (-np.expm1(-end / tau)))
    np.testing.assert_allclose(solution.y[:, -1], [expected], rtol=1e-10, atol=2e-8)


def test_insulated_network_conserves_internal_energy_and_total_input_heat():
    network = ThermalNetwork([2, 3], [[1, -1], [-1, 1]], [[1], [0]],
                              np.zeros((2, 0)), [[0.5, 0.5]])
    assert not network.is_asymptotically_stable
    initial = np.array([300, 310])
    np.testing.assert_allclose(network.energy_rate_w(initial, [4], []), 4)
    final = network.step_zoh(initial, [4], [], 7)
    np.testing.assert_allclose(network.capacity_j_per_k @ (final - initial), 28, atol=1e-11)
    with pytest.raises(ValueError, match="not unique"):
        network.steady_state([0], [])


def test_default_network_is_passive_positive_stable_and_cross_coupled():
    network = default_ring_network(3)
    assert network.is_asymptotically_stable
    assert np.max(np.linalg.eigvals(network.state_matrix).real) < 0
    np.testing.assert_allclose(network.source_map.sum(axis=0), 1)
    np.testing.assert_allclose(network.conductance_w_per_k.sum(axis=1),
                               network.boundary_conductance_w_per_k.sum(axis=1), atol=1e-18)
    powers = np.zeros(9)
    powers[0] = 1e-3
    result = network.step_zoh(np.full(network.n_nodes, 300), powers, 300, 1e-5)
    assert np.all(result >= 300)
    observed = network.observe(result)
    assert observed[0] > observed[1] > observed[2] > 300
    steady = network.steady_state(powers, 300)
    assert abs(network.energy_rate_w(steady, powers, 300)) < 1e-15
    boundary_out = (network.boundary_conductance_w_per_k[:, 0] * (steady - 300)).sum()
    np.testing.assert_allclose(boundary_out, powers.sum(), rtol=1e-12)


@pytest.mark.parametrize("bad", [
    dict(capacity_j_per_k=[-1], conductance_w_per_k=[[1]], source_map=[[1]],
         boundary_conductance_w_per_k=[[1]], observer_matrix=[[1]]),
    dict(capacity_j_per_k=[1], conductance_w_per_k=[[1]], source_map=[[0.5]],
         boundary_conductance_w_per_k=[[1]], observer_matrix=[[1]]),
    dict(capacity_j_per_k=[1], conductance_w_per_k=[[1]], source_map=[[1]],
         boundary_conductance_w_per_k=[[2]], observer_matrix=[[1]]),
])
def test_nonphysical_network_inputs_rejected(bad):
    with pytest.raises(ValueError):
        ThermalNetwork(**bad)


def test_multiple_boundaries_have_correct_weighted_equilibrium():
    network = ThermalNetwork([1], [[3]], [[1]], [[1, 2]], [[1]])
    np.testing.assert_allclose(network.steady_state([0], [300, 315]), [310])

