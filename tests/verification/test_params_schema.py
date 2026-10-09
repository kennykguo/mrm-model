"""Strict schema: key list format, did-you-mean, deleted keys, fidelity rules (07 P1-15)."""
import copy
from pathlib import Path

import pytest

from mrm.params import schema as sch
from mrm.params.config import ConfigError, load, yaml_load

MINI = Path(__file__).resolve().parents[1] / "data" / "mini.yaml"


def mini():
    return copy.deepcopy(yaml_load(MINI.read_text()))


def test_key_format():
    types = {"float", "int", "bool", "str", "enum", "list_float", "list_int", "list_str", "table", "map"}
    for k in sch.SCHEMA.values():
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
            assert path in sch.SCHEMA, f"{k.path}: condition on unknown key {path}"
        assert k.path.split(".")[0] in sch.TOP_SECTIONS, k.path


def test_extend_rejects_duplicates():
    with pytest.raises(ValueError, match="twice"):
        sch.extend([sch.Key("sim.dt_plant_s")])


def test_did_you_mean():
    assert "ring.heater.r_cold_ohm" in sch.suggest("ring.heater.r_cold_ohms")
    assert "ring.optics.q_loaded" in sch.suggest("ring.optics.q_load")


def test_typo_override_is_an_error_with_suggestion():
    # 07 P1-15: "an override typo (r_cold_ohms) silently runs the nominal model"
    with pytest.raises(ConfigError, match="r_cold_ohm"):
        load(mini(), overrides=["ring.heater.r_cold_ohms=50"])


def test_typo_in_preset_is_an_error():
    d = mini()
    d["ring"]["optics"]["q_loadd"] = {"value": 3800, "provenance": "printed", "generation": "G25", "source": "LM-T"}
    with pytest.raises(ConfigError, match="did you mean ring.optics.q_loaded"):
        load(d)


def test_deleted_key_message():
    d = mini()
    d["ring"]["optics"]["absorbed_fraction"] = {"value": 0.3, "provenance": "assumed", "generation": "none",
                                                "source": "TEST"}
    with pytest.raises(ConfigError, match="07 P0-2"):
        load(d)


def test_thermal_model_rom_requires_rom_file_and_forbids_tau_fast():
    d = mini()
    d["fidelity"]["thermal_model"] = "rom"
    with pytest.raises(ConfigError) as e:
        load(d)
    msg = str(e.value)
    assert "rom_file" in msg and "required" in msg
    assert "ring.thermal.tau_fast_s" in msg and "forbidden" in msg


def test_driver_mode_requires_its_full_scale():
    d = mini()
    d["ring"]["heater"]["driver_mode"] = "power"
    with pytest.raises(ConfigError, match="full_scale_w"):
        load(d)


def test_lut_forbids_analytic_slope():
    d = mini()
    d["fidelity"]["junction_model"] = "lut"
    with pytest.raises(ConfigError) as e:
        load(d)
    assert "lut_file" in str(e.value) and "eff_at_0v_pm_per_v" in str(e.value)


def test_enum_value_checked():
    d = mini()
    d["fidelity"]["thermal_model"] = "romm"
    with pytest.raises(ConfigError, match="did you mean 'rom'"):
        load(d)


def test_bias_binding_forbids_scenario_bias():
    d = mini()
    d["controller_interface"]["bind"] = {"tx.bias": "controller"}
    d["scenario"] = {"bias_v": 2.0}
    with pytest.raises(ConfigError, match="scenario.bias_v"):
        load(d)


def test_enum_code():
    assert sch.enum_code("ring.heater.driver_mode", "current") == 1
    assert sch.relation_members("R3_geometry")
