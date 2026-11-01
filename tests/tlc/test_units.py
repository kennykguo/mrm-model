"""Adapter unit laws (docs/TLC_PORT_SPEC_AND_PLAN.md 2.1): heater power fraction <-> voltage-DAC code, ADC code ->
normalised level, GHz <-> pm. Expected values come from the closed-form laws, never from the code under test."""
from __future__ import annotations

import numpy as np
import pytest

from mrm.control.tlc.units import HeaterLaw, level_from_codes, pm_per_ghz

C_M_PER_S = 299_792_458.0


def test_code_to_u_to_code_is_exact_for_every_code():
    law = HeaterLaw(code_min=0, code_max=65535)
    codes = np.arange(65536)
    assert np.array_equal(law.code_from_u(law.u_from_code(codes)), codes)


def test_u_to_code_is_within_half_an_lsb_of_the_square_root_law():
    law = HeaterLaw(code_min=0, code_max=65535)
    u = np.linspace(0.0, 1.0, 10_001)
    assert np.max(np.abs(law.code_from_u(u) - 65535 * np.sqrt(u))) <= 0.5


def test_power_is_quadratic_in_code():
    law = HeaterLaw(code_min=0, code_max=1000)
    assert law.u_from_code(np.array([500]))[0] == pytest.approx(0.25)
    assert law.u_from_code(np.array([1000]))[0] == pytest.approx(1.0)


def test_codes_are_int64_and_clamped_to_the_dac_range():
    law = HeaterLaw(code_min=100, code_max=1000)
    codes = law.code_from_u(np.array([-1.0, 0.0, 2.0, np.nan]))
    assert codes.dtype == np.int64
    assert codes.tolist() == [100, 100, 1000, 100]


def test_saturation_flags_commands_outside_the_reachable_power_range():
    law = HeaterLaw(code_min=100, code_max=1000)
    u_min = (100 / 1000) ** 2
    assert law.saturated(np.array([u_min / 2, u_min, 0.5, 1.0, 1.5])).tolist() == [True, False, False, False, True]


def test_heater_law_rejects_a_bad_range():
    with pytest.raises(ValueError):
        HeaterLaw(code_min=10, code_max=10)


def test_level_is_code_over_full_scale():
    assert level_from_codes(np.array([0, 32767.5, 65535]), 65535).tolist() == [0.0, 0.5, 1.0]


def test_pm_per_ghz_is_lambda_squared_over_c():
    f_hz = C_M_PER_S / 1310e-9
    want = (1310e-9) ** 2 / C_M_PER_S * 1e9 * 1e12      # pm per GHz at 1310 nm, about 5.72
    assert pm_per_ghz(np.array([f_hz]))[0] == pytest.approx(want, rel=1e-12)
    assert want == pytest.approx(5.72, abs=0.01)
