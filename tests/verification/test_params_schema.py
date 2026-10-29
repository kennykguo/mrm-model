"""Checks the strict schema mrm.params.schema: the key list format, did-you-mean suggestions, deleted keys and the
fidelity rules (a switch that requires or forbids other keys).

A typo in an override or a preset must be an error that names the right key: before the strict schema, an override
typo such as ``r_cold_ohms`` silently ran the nominal model.
"""
import copy
from pathlib import Path

import pytest

from mrm.params import schema
from mrm.params.config import ConfigError, load, yaml_load

MINI = Path(__file__).resolve().parents[1] / "data" / "mini.yaml"   # the small 4-ring test config


def _mini_dict():
    """A fresh copy of tests/data/mini.yaml as a dict (safe to modify)."""
    return copy.deepcopy(yaml_load(MINI.read_text()))


def test_archived_tcmt_is_rejected_in_config():
    with pytest.raises(ConfigError, match="fidelity.optics_model"):
        load(_mini_dict(), overrides=["fidelity.optics_model=tcmt"])


def test_key_format():
    types = {"float", "int", "bool", "str", "enum", "list_float", "list_int", "list_str", "table", "map"}
    for k in schema.SCHEMA.values():
        assert k.type in types, k.path
        assert k.level in ("die", "ring", "line", "global"), k.path
        if k.physical:
            assert k.default is None, f"{k.path}: a physical number never has a schema default"
        if k.type == "enum":
            assert k.enum, k.path
            if k.default is not None:
                assert k.default in k.enum, k.path
        if k.per_world:
            assert k.type == "enum" and not k.physical, k.path
        for path, _ in k.required_if + k.forbidden_if:
            assert path in schema.SCHEMA, f"{k.path}: condition on unknown key {path}"
        assert k.path.split(".")[0] in schema.TOP_SECTIONS, k.path


def test_extend_rejects_duplicates():
    with pytest.raises(ValueError, match="twice"):
        schema.extend([schema.Key("sim.dt_plant_s")])


def test_did_you_mean():
    assert "ring.heater.r_cold_ohm" in schema.suggest("ring.heater.r_cold_ohms")
    assert "ring.optics.q_loaded" in schema.suggest("ring.optics.q_load")


def test_typo_override_is_an_error_with_suggestion():
    # the failure the strict schema prevents: "an override typo (r_cold_ohms) silently runs the nominal model"
    with pytest.raises(ConfigError, match="r_cold_ohm"):
        load(_mini_dict(), overrides=["ring.heater.r_cold_ohms=50"])


def test_typo_in_preset_is_an_error():
    d = _mini_dict()
    d["ring"]["optics"]["q_loadd"] = {"value": 3800, "provenance": "printed", "generation": "G25", "source": "LM-T"}
    with pytest.raises(ConfigError, match="did you mean ring.optics.q_loaded"):
        load(d)


def test_deleted_key_message():
    d = _mini_dict()
    d["ring"]["optics"]["absorbed_fraction"] = {"value": 0.3, "provenance": "assumed", "generation": "none",
                                                "source": "TEST"}
    with pytest.raises(ConfigError, match="loss_budget"):
        load(d)


def test_thermal_model_rom_requires_rom_file_and_forbids_tau_fast():
    d = _mini_dict()
    d["fidelity"]["thermal_model"] = "rom"
    with pytest.raises(ConfigError) as e:
        load(d)
    msg = str(e.value)
    assert "rom_file" in msg and "required" in msg
    assert "ring.thermal.tau_fast_s" in msg and "forbidden" in msg


def test_driver_mode_requires_its_full_scale():
    d = _mini_dict()
    d["ring"]["heater"]["driver_mode"] = "power"
    with pytest.raises(ConfigError, match="full_scale_w"):
        load(d)


def test_lut_forbids_analytic_slope():
    d = _mini_dict()
    d["fidelity"]["junction_model"] = "lut"
    with pytest.raises(ConfigError) as e:
        load(d)
    assert "lut_file" in str(e.value) and "eff_at_0v_pm_per_v" in str(e.value)


def test_enum_value_checked():
    d = _mini_dict()
    d["fidelity"]["thermal_model"] = "romm"
    with pytest.raises(ConfigError, match="did you mean 'rom'"):
        load(d)


def test_bias_binding_forbids_scenario_bias():
    d = _mini_dict()
    d["controller_interface"]["bind"] = {"tx.bias": "controller"}
    d["scenario"] = {"bias_v": 2.0}
    with pytest.raises(ConfigError, match="scenario.bias_v"):
        load(d)


def test_enum_code():
    assert schema.enum_code("ring.heater.driver_mode", "current") == 1
    assert schema.relation_members("R3_geometry")
