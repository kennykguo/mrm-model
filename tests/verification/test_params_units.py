"""Checks the unit-suffix grammar of config keys and the single SI conversion (mrm.params.units).

A key ends with its unit (``dlambda_dt_pm_per_k``: pm/K). parse_key splits the name from the unit, to_si converts a
value once at load (dB to a natural-log power ratio, dBm to W, degC to K) and from_si inverts it. The token list
follows docs/ARCHITECTURE.md, "Use SI inside the code".
"""
import math

import numpy as np
import pytest

from mrm.params import schema
from mrm.params.units import TOKENS, UnitError, from_si, parse_key, to_si, unit_of


@pytest.mark.parametrize("key, name, expr, kind, scale", [
    ("dlambda_dt_pm_per_k", "dlambda_dt", "pm_per_k", "linear", 1e-12),
    ("bus_loss_db_per_mm", "bus_loss", "db_per_mm", "db_per_len", 1e-3),
    ("power_per_line_dbm", "power_per_line", "dbm", "dbm", 1e-3),
    ("alpha_l_per_k", "alpha_l", "per_k", "linear", 1.0),
    ("input_noise_pa_per_rt_hz", "input_noise", "pa_per_rt_hz", "linear", 1e-12),
    ("eff_at_0v_pm_per_v", "eff_at_0v", "pm_per_v", "linear", 1e-12),
    ("capacitance_at_0v_ff", "capacitance_at_0v", "ff", "linear", 1e-15),
    ("t_ring_lock_c", "t_ring_lock", "c", "celsius", 1.0),
    ("resonance_offset_ref_temp_c", "resonance_offset_ref_temp", "c", "celsius", 1.0),
    ("rin_db_per_hz", "rin", "db_per_hz", "db_per_hz", 1.0),
    ("alpha_l_slope_per_k2", "alpha_l_slope", "per_k2", "linear", 1.0),
    ("slew_codes_per_s", "slew", "codes_per_s", "linear", 1.0),
    ("v_min_v", "v_min", "v", "linear", 1.0),
    ("tail_share_frac", "tail_share", "frac", "linear", 1.0),
    ("z_heater_ring_k_per_mw", "z_heater_ring", "k_per_mw", "linear", 1e3),
    ("bitrate_gbps", "bitrate", "gbps", "linear", 1e9),
])
def test_parse_known_keys(key, name, expr, kind, scale):
    n, spec = parse_key(key)
    assert n == name
    assert spec.expr == expr
    assert spec.kind == kind
    assert spec.scale == pytest.approx(scale, rel=1e-15)


@pytest.mark.parametrize("key", ["n_g", "q_loaded", "kappa2_per_pass", "channels_per_adc", "fill_fraction",
                                 "adc_bits", "order", "bitstat_window_bits", "heat_fraction"])
def test_unitless_keys(key):
    assert unit_of(key).kind == "none"
    assert to_si(key, 3.5) == 3.5


def test_dimensions():
    assert unit_of("dlambda_dt_pm_per_k").si_unit == "m K^-1"
    assert unit_of("r_cold_ohm").dims == (2, 1, -3, -2, 0)
    assert unit_of("input_noise_pa_per_rt_hz").dims == (0, 0, 0.5, 1, 0)     # A s^0.5 = A/sqrt(Hz)
    assert unit_of("dark_activation_ev").scale == pytest.approx(1.602176634e-19)


def test_conversions_and_inverse():
    assert to_si("x_c", 25.0) == pytest.approx(298.15)
    assert to_si("p_dbm", 10.0) == pytest.approx(10e-3)
    assert to_si("p_dbm", 0.0) == pytest.approx(1e-3)
    # dB -> natural-log power ratio: 10 dB is ln(10)
    assert to_si("loss_db", 10.0) == pytest.approx(math.log(10.0))
    assert math.exp(-to_si("loss_db", 3.0)) == pytest.approx(10 ** -0.3)
    # 1 dB/cm -> alpha with P(z) = P0 exp(-alpha z): 23.03 /m
    assert to_si("a_db_per_cm", 1.0) == pytest.approx(math.log(10) / 10 / 0.01)
    assert to_si("rin_db_per_hz", -150.0) == pytest.approx(1e-15)
    for key, v in [("x_c", 37.0), ("p_dbm", -3.0), ("l_db", 2.5), ("a_db_per_cm", 110.0), ("r_db_per_hz", -140.0),
                   ("s_pm_per_k", 67.5), ("n_pa_per_rt_hz", 6.4)]:
        assert from_si(key, to_si(key, v)) == pytest.approx(v)
    arr = to_si("t_us", np.array([[1.0, 2.0]]))
    assert arr.shape == (1, 2) and np.allclose(arr, [[1e-6, 2e-6]])


def test_affine_and_log_misuse_rejected():
    with pytest.raises(UnitError, match="_k"):
        parse_key("t_offset_c")
    with pytest.raises(UnitError, match="_k"):
        parse_key("ambient_sigma_c")
    with pytest.raises(UnitError, match="_db"):
        parse_key("power_sigma_dbm")
    # Celsius is fine for an absolute temperature even with 'offset' earlier in the name
    assert unit_of("resonance_offset_ref_temp_c").kind == "celsius"
    # c, db and dbm do not combine with other tokens; the longest-match rule must not leave one in the name and
    # silently pick a shorter unit ('shift_ghz_per_c' would read as name 'shift_ghz_per' in absolute degC)
    for bad in ("x_c_per_s", "shift_ghz_per_c", "x_db_per_s", "drift_dbm_per_s", "gain_per_db"):
        with pytest.raises(UnitError):
            parse_key(bad)
    assert unit_of("rate_k_per_s").expr == "k_per_s"
    assert unit_of("shift_ghz_per_k").expr == "ghz_per_k"


def test_every_schema_key_parses():
    assert schema.check_units() == []


def test_token_table_matches_architecture():
    arch = "nm um mm m pm s ms us ns ps hz khz mhz ghz thz w mw uw nw a ma ua na v mv ohm f ff k pct frac ev".split()
    for t in arch:
        assert t in TOKENS, t
    assert TOKENS["hz"][0] == 1.0 and TOKENS["ghz"][0] == 1e9      # cycles per second, never angular
