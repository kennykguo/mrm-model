import json

import numpy as np
import pytest

from mrm_model.extraction import _assemble, _full_transient, _reduced_params, thermal_fem_reference
from mrm_model.thermal import ThermalNetwork, ThermalParams


def test_optional_sparse_fem_exports_passive_compatible_rom():
    pytest.importorskip("skfem")
    result = thermal_fem_reference(mesh_level=0, n_modes=12, transient_subdivisions=4)
    assert result["mesh"]["total_dofs"] < 25000
    assert result["mesh"]["sparse_matrix_bytes"] < 10_000_000
    assert result["rom_dc_relative_error"] < 1e-8
    assert result["calibration_status"] == "illustrative_geometry_unvalidated"
    params = ThermalParams(**result["thermal_params"])
    network = ThermalNetwork(params)
    np.testing.assert_allclose(network.equilibrium([0], 320), [320])
    assert np.all(network.modes()["time_constants_s"] > 0)
    assert result["steady_ring_temperature_rise_k_per_w"] > 0
    assert result["rom_transient_max_relative_to_full_scale"] < 0.15
    assert json.loads(json.dumps(result))["schema"] == result["schema"]


def test_fem_mesh_guard_before_large_allocation():
    pytest.importorskip("skfem")
    with pytest.raises(ValueError, match="memory guard"):
        thermal_fem_reference(mesh_level=3)


def test_robin_boundary_memory_matches_independent_full_mesh_step_and_ramp():
    pytest.importorskip("skfem")
    system = _assemble(0, 100e-6, boundary_contact_w_per_m2_k=1e7)
    np.testing.assert_allclose(
        system.conductance @ np.ones(len(system.free_dofs)), system.boundary_load, atol=1e-15
    )
    params, _ = _reduced_params(system, 24)
    times = np.r_[0.0, np.geomspace(1e-9, 2e-3, 64)]
    for boundary in [lambda time: 10.0, lambda time: min(time * 5000, 10.0)]:
        full = _full_transient(system, times, 0, 32, boundary)
        network = ThermalNetwork(params)
        np.testing.assert_allclose(network.step(0, [0], 310), [300])
        rom = []
        previous = 0.0
        for time in times:
            dt = (time - previous) / 32
            for substep in range(32):
                midpoint = previous + (substep + 0.5) * dt
                network.step(dt, [0], 300 + boundary(midpoint))
            rom.append(network.ring_temperatures_k[0] - 300)
            previous = time
        assert np.max(abs(np.asarray(rom) - full)) / max(full) < 0.015
    np.testing.assert_allclose(network.equilibrium([0], 330), [330], atol=1e-9)
