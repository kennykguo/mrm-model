import numpy as np
import pytest

from ringlab.junction import DepletionJunction, MeasuredJunctionTable


def measured_table():
    return MeasuredJunctionTable(
        voltage_v=[0, 1, 3], resonance_shift_hz=[0, -4e9, -10e9],
        additional_loss_per_m=[0, -10, -20], capacitance_f=[40e-15, 30e-15, 20e-15],
        resistance_ohm=[60, 70, 90], leakage_current_a=[1e-9, 2e-9, 4e-9],
        provenance="Synthetic unit-test table, not measured hardware",
    )


def test_depletion_sign_and_square_root_capacitance():
    j = DepletionJunction()
    r = j.evaluate(np.array([0, 0.8, 2.4]))
    np.testing.assert_allclose(r.capacitance_f - j.parasitic_capacitance_f, j.zero_bias_capacitance_f / np.sqrt([1, 2, 4]))
    assert r.resonance_shift_hz[0] == 0
    assert np.all(np.diff(r.resonance_shift_hz) < 0)
    assert np.all(np.diff(r.additional_loss_per_m) < 0)
    np.testing.assert_allclose(r.electrical_heat_w, np.array([0, 0.8, 2.4]) * j.leakage_current_a)


def test_measured_table_interpolates_and_owns_immutable_copy():
    j = measured_table()
    r = j.evaluate(np.array([[0, 2], [3, 0.5]]))
    np.testing.assert_allclose(r.resonance_shift_hz, [[0, -7e9], [-10e9, -2e9]])
    np.testing.assert_allclose(r.capacitance_f[0, 1], 25e-15)
    assert not j.voltage_v.flags.writeable
    assert r.resistance_ohm.shape == (2, 2)


@pytest.mark.parametrize("voltage", [-0.1, 5.1, np.nan, np.inf])
def test_analytic_bounds(voltage):
    with pytest.raises(ValueError):
        DepletionJunction().evaluate(voltage)


@pytest.mark.parametrize("voltage", [-0.1, 3.0001, np.nan, np.inf])
def test_measured_table_never_extrapolates(voltage):
    with pytest.raises(ValueError):
        measured_table().evaluate(voltage)


def test_table_rejects_duplicate_grid_and_invalid_physics():
    j = measured_table()
    from dataclasses import replace
    for kwargs in ({"voltage_v": [0, 0, 3]}, {"capacitance_f": [1, 0, 2]}, {"provenance": ""}, {"leakage_current_a": [-1, 0, 1]}):
        with pytest.raises(ValueError):
            replace(j, **kwargs)
