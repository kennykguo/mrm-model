import numpy as np
import pytest

from ringlab.optics import RingParameters
from ringlab.optical_tables import build_nrz_average_table


def test_adaptive_table_matches_fresh_dense_grid_and_conserves_power():
    p = RingParameters()
    table = build_nrz_average_table(p, resonance_shift_hz=(-20e9, 0), additional_loss_per_m=(-10, 10), absolute_tolerance=3e-6)
    assert table.diagnostics["validated"], table.diagnostics
    # A validation grid not used during construction, extending across periods.
    detuning = p.fsr_hz * np.linspace(-1.43, 1.59, 10003)
    a, b = table.evaluate(detuning), table.direct(detuning)
    for name in ("through_power_w", "drop_power_w", "absorbed_power_w", "conditional_through_power_w"):
        assert np.max(np.abs(getattr(a, name) - getattr(b, name))) < 3e-6
    assert np.max(np.abs(a.stored_energy_j - b.stored_energy_j)) * table.energy_rate_per_s < 3e-6
    np.testing.assert_allclose(a.through_power_w + a.drop_power_w + a.intrinsic_loss_power_w, 1, atol=8e-13)
    assert np.all(a.absorbed_power_w >= 0)
    assert table.diagnostics["memory_bytes"] < 2_000_000
    assert not table.detuning_grid_hz.flags.writeable


def test_table_power_scaling_and_periodic_boundary_continuity():
    p = RingParameters()
    table = build_nrz_average_table(p, absolute_tolerance=1e-5)
    fraction = table.evaluate(np.array([-p.fsr_hz/2, p.fsr_hz/2, 0]))
    powers = table.evaluate(np.array([-p.fsr_hz/2, p.fsr_hz/2, 0]), np.array([1e-3, 2e-3, 0]))
    np.testing.assert_allclose(powers.absorbed_power_w, fraction.absorbed_power_w * [1e-3, 2e-3, 0])
    np.testing.assert_allclose(fraction.absorbed_power_w[0], fraction.absorbed_power_w[1])


def test_node_budget_failure_uses_exact_direct_fallback():
    p = RingParameters()
    table = build_nrz_average_table(p, absolute_tolerance=1e-12, initial_nodes=5, max_nodes=5)
    assert not table.diagnostics["validated"]
    frequency = np.array([0, 7e9, -13e9])
    a, b = table.evaluate(frequency), table.direct(frequency)
    np.testing.assert_array_equal(a.absorbed_power_w, b.absorbed_power_w)


def test_table_requires_positive_tolerance():
    with pytest.raises(ValueError):
        build_nrz_average_table(RingParameters(), absolute_tolerance=-1)
