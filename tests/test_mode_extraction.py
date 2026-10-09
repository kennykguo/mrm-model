import json

import pytest

from mrm_model.mode_extraction import material_indices, waveguide_mode_reference


def test_material_dispersion_public_reference_points_and_range():
    silicon, silica = material_indices(1.3e-6)
    assert silicon == pytest.approx(3.5016)
    assert silica == pytest.approx(1.44691753, rel=1e-7)
    lower = material_indices(1.29e-6)
    upper = material_indices(1.31e-6)
    assert lower[0] > upper[0]
    assert lower[1] > upper[1]
    with pytest.raises(ValueError, match="material data"):
        material_indices(1.1e-6)


def test_full_vector_mode_and_same_branch_group_derivative():
    pytest.importorskip("femwell")
    result = waveguide_mode_reference(mesh_scale=1)
    assert 1.45 < result["neff"] < 3.51
    assert result["ng"] > result["neff"]
    assert result["mode"]["te_fraction"] > 0.7
    assert result["wavelength_step_relative_ng_change"] < 0.002
    assert all(sample["overlap_with_reference"] > 0.99 for sample in result["wavelength_samples"])
    assert result["mesh"]["mixed_dofs"] < 42000
    assert result["thermal_sensitivity"] is None
    assert "NOT_Lightmatter" in result["calibration_status"]
    assert json.loads(json.dumps(result))["ng"] == result["ng"]


def test_mode_memory_guard_and_input_validation():
    pytest.importorskip("femwell")
    with pytest.raises(ValueError, match="memory guard"):
        waveguide_mode_reference(mesh_scale=0.1)
    with pytest.raises(ValueError, match="delta_wavelength"):
        waveguide_mode_reference(delta_wavelength_m=0)


def test_vector_effective_temperature_derivative_is_not_bulk_weight_bound():
    pytest.importorskip("femwell")
    result = waveguide_mode_reference(silicon_dn_dt_per_k=1.86e-4, silica_dn_dt_per_k=1e-5)
    derivative = result["thermal_sensitivity"]
    assert derivative is not None
    # Inspect exact report names rather than fitting this reference into textbook confinement weighting.
    assert derivative["dn_eff_per_k"] > 1.86e-4
