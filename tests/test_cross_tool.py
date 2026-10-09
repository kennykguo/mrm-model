"""Independent network solves; optional SAX verifies the real circuit backend."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mrm_model.optics import OpticalParams, optical_response

spec = importlib.util.spec_from_file_location(
    "validate_sax", Path(__file__).parents[1] / "scripts" / "validate_sax.py"
)
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


def test_component_coupler_is_unitary_and_reciprocal():
    for coupling in (0, 0.002, 0.08, 0.5, 0.999):
        s = validation.coupler_smatrix(coupling)
        np.testing.assert_allclose(s.conj().T @ s, np.eye(4), atol=3e-16)
        np.testing.assert_array_equal(s.T, s)


def test_independent_connected_component_graph_randomized_spectra():
    report, curve = validation.validate(solver="graph", random_cases=12, samples=151)
    assert report["passed"], report["max_errors"]
    assert report["case_count"] == 18
    assert report["max_errors"]["max_abs_absorption_error_w"] < 1e-14
    assert report["max_errors"]["max_abs_energy_error_j"] < 1e-23
    assert np.max(np.abs(curve["network_through_w"] - curve["compact_through_w"])) < 1e-14


def test_component_network_temperature_and_voltage_perturbations():
    p = OpticalParams()
    f = p.reference_frequency_hz + np.linspace(-100e9, 100e9, 151)
    graph = validation.port_graph_response(p, f, temperature_k=310, junction_voltage_v=-1.2)
    compact = optical_response(p, f, temperature_k=310, junction_voltage_v=-1.2, input_power_w=0.001)
    np.testing.assert_allclose(graph["through_power_w"], compact.through_power_w, atol=1e-15)
    np.testing.assert_allclose(graph["drop_power_w"], compact.drop_power_w, atol=1e-15)


@pytest.mark.skipif(
    importlib.util.find_spec("sax") is None, reason="optional SAX/JAX circuit stack not installed"
)
def test_sax_feedback_solver_independently_matches_exact_periodic_ports():
    report, _ = validation.validate(solver="sax", random_cases=4, samples=101)
    assert report["passed"], report["max_errors"]
    assert "SAX" in report["solver"]
    assert "sax" in report["versions"]
