"""Loader: leaf format, provenance enforcement, extends, alternatives, overrides, artefacts, timing rule
(07 P1-14, P1-15, P0-6; ARCHITECTURE sections 1 and 5)."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from mrm.params.config import (ArtefactError, ConfigError, load, provenance_rank, weakest, yaml_load)

MINI = Path(__file__).resolve().parents[1] / "data" / "mini.yaml"


def mini():
    return copy.deepcopy(yaml_load(MINI.read_text()))


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def leafv(v, prov="assumed", gen="none", src="TEST"):
    return {"value": v, "provenance": prov, "generation": gen, "source": src}


# ------------------------------------------------------------------------------------------------ basics

def test_mini_loads_and_reports():
    cfg = load(MINI)
    assert cfg.n_rings == 4 and cfg.n_lines == 4
    assert cfg.leaf("ring.heater.p_fsr_mw").provenance == "printed"
    assert cfg.si("ring.heater.p_fsr_mw") == pytest.approx(25.45e-3)
    assert cfg.si("ring.optics.resonance_offset_ref_temp_c") == pytest.approx(298.15)
    assert cfg.get("fidelity.optics_model") == "exact"           # schema default of a switch
    assert cfg.hash() == load(MINI).hash()
    assert yaml.safe_load(cfg.dump_yaml())["ring"]["heater"]["p_fsr_mw"]["provenance"] == "printed"


def test_yaml_reads_exponent_without_sign():
    assert yaml_load("a: 1.0e6\nb: 1e6\nc: 2") == {"a": 1e6, "b": 1e6, "c": 2}


def test_provenance_ranking():
    assert provenance_rank("printed") < provenance_rank("solver") < provenance_rank("assumed")
    assert provenance_rank("solver(assumed-geometry)") == provenance_rank("assumed")
    assert weakest(["printed", "literature", "inferred"]) == "literature"
    assert weakest(["override", "printed"]) == "override"


# ------------------------------------------------------------------------------------------------ provenance rules

def test_bare_number_in_preset_rejected():
    d = mini()
    d["ring"]["heater"]["p_fsr_mw"] = 25.45
    with pytest.raises(ConfigError, match="bare value in a preset"):
        load(d)


def test_bare_number_in_experiment_is_override(tmp_path):
    write(tmp_path, "base.yaml", mini())
    exp = {"extends": "base", "meta": {"kind": "experiment", "name": "e1"},
           "ring": {"optics": {"q_loaded": 4000}}, "experiment": {"duration_s": 0.01}}
    cfg = load(write(tmp_path, "e1.yaml", exp))
    assert cfg.leaf("ring.optics.q_loaded").value == 4000
    assert cfg.leaf("ring.optics.q_loaded").provenance == "override"
    assert cfg.get("experiment.duration_s") == 0.01
    # like an overrides= value it holds in every world: the q_loaded interval is pinned (it was drawn before)
    from mrm.params.realize import realize
    assert cfg.uncertainty["q_loaded"].pinned == "override"
    assert np.all(realize(cfg, 8, 0, variation="off", thermal_hook=None).optics.q_loaded == 4000)


def test_leaf_needs_provenance_generation_source():
    d = mini()
    d["ring"]["heater"]["r_cold_ohm"] = {"value": 45.4}
    with pytest.raises(ConfigError) as e:
        load(d)
    assert "no provenance" in str(e.value) and "no generation" in str(e.value) and "no source" in str(e.value)


def test_closed_enums_and_unknown_fields():
    d = mini()
    d["ring"]["heater"]["r_cold_ohm"] = {"value": 45.4, "provenance": "guessed", "generation": "G27",
                                         "source": "NOWHERE", "units": "ohm"}
    with pytest.raises(ConfigError) as e:
        load(d)
    msg = str(e.value)
    assert "guessed" in msg and "G27" in msg and "NOWHERE" in msg and "unknown leaf field 'units'" in msg


def test_calibrated_needs_targets_and_private_needs_private_dir():
    d = mini()
    d["ring"]["heater"]["r_cold_ohm"]["provenance"] = "calibrated"
    with pytest.raises(ConfigError, match="targets"):
        load(d)
    d["ring"]["heater"]["r_cold_ohm"]["targets"] = ["CAL-3"]
    load(d)
    d["ring"]["heater"]["r_cold_ohm"]["provenance"] = "private"
    with pytest.raises(ConfigError, match="private"):
        load(d)


def test_range_and_bounds():
    d = mini()
    d["ring"]["optics"]["q_loaded"]["value"] = 5000
    with pytest.raises(ConfigError, match="outside its own range"):
        load(d)
    d = mini()
    d["ring"]["monitor"]["responsivity_a_per_w"] = leafv(3.0)
    with pytest.raises(ConfigError, match="physical bounds"):
        load(d)


def test_derived_only_key_rejected():
    d = mini()
    d["ring"]["optics"]["dn_eff_dt_per_k"] = leafv(2e-4)
    with pytest.raises(ConfigError, match="R1_thermo_optic"):
        load(d)


def test_ring_list_length_checked():
    d = mini()
    d["ring"]["optics"]["resonance_offset_nm"]["value"] = [7.6, 7.7, 7.8]
    with pytest.raises(ConfigError, match="expected 4"):
        load(d)


def test_sources_checked_when_known():
    d = mini()
    d["meta"]["kind"] = "preset"
    srcs = d["meta"].pop("sources")
    with pytest.raises(ConfigError, match="LM26-P"):
        load(d, sources={"OTHER"})
    load(d, sources=set(srcs))


# ------------------------------------------------------------------------------------------------ timing rule

def test_latency_must_be_whole_ticks():
    d = mini()
    d["ring"]["monitor"]["adc_latency_s"]["value"] = 1.5e-6
    with pytest.raises(ConfigError, match="ring.monitor.adc_latency_s.*whole number of ticks"):
        load(d)
    d = mini()
    d["ring"]["monitor"]["adc_rate_hz"]["value"] = 300e3          # 3.33 us period
    with pytest.raises(ConfigError, match="adc_rate_hz"):
        load(d)
    d = mini()
    d["controller_interface"]["update_rate_hz"] = 30e3
    with pytest.raises(ConfigError, match="update_rate_hz"):
        load(d)


# ------------------------------------------------------------------------------------------------ extends

def test_extends_child_must_restate_provenance(tmp_path):
    write(tmp_path, "base.yaml", mini())
    child = {"extends": "base", "meta": {"name": "child", "kind": "test", "sources": {"TEST": "x"}},
             "ring": {"heater": {"p_fsr_mw": {"value": 30.0}}}}
    with pytest.raises(ConfigError, match="without restating provenance and generation"):
        load(write(tmp_path, "child.yaml", child))
    child["ring"]["heater"]["p_fsr_mw"] = {"value": 30.0, "provenance": "assumed", "generation": "none"}
    cfg = load(write(tmp_path, "child.yaml", child))
    lf = cfg.leaf("ring.heater.p_fsr_mw")
    assert lf.value == 30.0 and lf.provenance == "assumed" and lf.source == "LM-T"   # source inherited
    assert any("value changed" in o for o in lf.origin) and len(cfg.files) == 2


def test_extends_note_only_change_is_free(tmp_path):
    write(tmp_path, "base.yaml", mini())
    child = {"extends": "base", "meta": {"name": "c2", "kind": "test"},
             "ring": {"heater": {"p_fsr_mw": {"note": "new note"}}}}
    # a mapping without value/from is a section, not a leaf: p_fsr_mw.note is then an unknown key
    with pytest.raises(ConfigError):
        load(write(tmp_path, "c2.yaml", child))


def test_extends_cycle(tmp_path):
    write(tmp_path, "a.yaml", {"extends": "b", "meta": {"name": "a"}})
    write(tmp_path, "b.yaml", {"extends": "a", "meta": {"name": "b"}})
    with pytest.raises(ConfigError, match="cycle"):
        load(tmp_path / "a.yaml")


def test_corner_preset_only_selects(tmp_path):
    write(tmp_path, "base.yaml", mini())
    corner = {"extends": "base", "meta": {"name": "h86", "kind": "corner"},
              "alts": {"uncertainty.thermo_optic": "H86"}}
    cfg = load(write(tmp_path, "h86.yaml", corner))
    assert cfg.leaf("ring.thermal.dlambda_dt_pm_per_k").value == 86.0
    assert cfg.leaf("ring.thermal.dlambda_dt_pm_per_k").provenance == "printed"
    assert cfg.leaf("ring.thermal.dlambda_dt_beta_per_k").value == 0.0
    assert cfg.uncertainty["thermo_optic"].pinned == "H86"
    corner["ring"] = {"optics": {"q_loaded": {"value": 4000, "provenance": "assumed", "generation": "none"}}}
    with pytest.raises(ConfigError, match="corner preset may contain only"):
        load(write(tmp_path, "bad.yaml", corner))


# ------------------------------------------------------------------------------------------------ alts and overrides

def test_leaf_alternative_selection_pins_hypothesis():
    cfg = load(MINI, alts={"ring.optics.fsr_thz": "window_a"})
    lf = cfg.leaf("ring.optics.fsr_thz")
    assert lf.value == 1.40 and lf.selected_alt == "window_a" and lf.provenance == "inferred"
    assert cfg.uncertainty["fsr"].pinned == "window_a"
    # selecting a dlambda/dT alternative restricts or pins the thermo_optic hypothesis consistently
    cfg = load(MINI, alts=["ring.thermal.dlambda_dt_pm_per_k=bench"])
    assert cfg.uncertainty["thermo_optic"].pinned == "bench"


def test_unknown_alternative():
    with pytest.raises(ConfigError, match="no alternative 'H99'"):
        load(MINI, alts={"ring.thermal.dlambda_dt_pm_per_k": "H99"})
    with pytest.raises(ConfigError, match="no choice"):
        load(MINI, alts={"uncertainty.thermo_optic": "H99"})


def test_interval_corner_pin():
    cfg = load(MINI, alts={"uncertainty.q_loaded": "low_q"})
    assert cfg.uncertainty["q_loaded"].pinned == 2600


def test_overrides_dict_and_strings():
    cfg = load(MINI, overrides={"ring.optics.q_loaded": 4100, "sim.dt_plant_s": 1e-6})
    lf = cfg.leaf("ring.optics.q_loaded")
    assert lf.value == 4100 and lf.provenance == "override" and lf.base_provenance == "read-off"
    assert cfg.uncertainty["q_loaded"].pinned == "override"
    cfg = load(MINI, overrides=["ring.optics.q_loaded=4100", "uncertainty.q_loaded.high=4150",
                                "variation.ring.heater.r_cold_ohm.ring_sigma=1.0", "controller.params.kp=0.3"])
    assert cfg.uncertainty["q_loaded"].fields["high"] == 4150
    with pytest.raises(ConfigError, match="outside the range"):     # a world's value must lie in the leaf's range
        load(MINI, overrides=["uncertainty.q_loaded.high=4500"])
    assert cfg.variation["ring.heater.r_cold_ohm"].fields["ring_sigma"] == 1.0
    assert cfg.get("controller.params.kp") == 0.3
    with pytest.raises(ConfigError, match="must be 'path=value'"):
        load(MINI, overrides=["ring.optics.q_loaded"])


# ------------------------------------------------------------------------------------------------ uncertainty / variation format

def test_uncertainty_entry_validation():
    d = mini()
    d["uncertainty"]["thermo_optic"]["nominal"] = "H99"
    with pytest.raises(ConfigError, match="nominal 'H99'"):
        load(d)
    d = mini()
    d["uncertainty"]["thermo_optic"]["choices"]["H86"]["ring.thermal.dlambda_dt_pm_per_k"] = "H87"
    with pytest.raises(ConfigError, match="no alternative 'H87'"):
        load(d)
    d = mini()
    d["uncertainty"]["q_loaded"]["low"] = 5000
    with pytest.raises(ConfigError, match="low <= high"):
        load(d)
    d = mini()
    d["uncertainty"]["bad"] = {"kind": "interval", "target": "ring.heater.efficiency_nm_per_mw", "low": 0.3,
                               "high": 0.5, "provenance": "assumed", "generation": "none", "source": "TEST"}
    with pytest.raises(ConfigError, match="derived"):
        load(d)
    d = mini()
    d["uncertainty"]["thermal"]["time_scale_fast"]["kind"] = "range"
    with pytest.raises(ConfigError, match="kind 'range'"):
        load(d)


def test_variation_entry_validation():
    d = mini()
    d["variation"]["ring.heater.r_cold_ohms"] = {"ring_sigma": 1.0, "provenance": "assumed", "generation": "none",
                                                 "source": "TEST"}
    with pytest.raises(ConfigError, match="did you mean ring.heater.r_cold_ohm"):
        load(d)
    d = mini()
    d["variation"]["ring.geometry.radius_um"] = {"ring_sigma": 0.01, "provenance": "assumed", "generation": "none",
                                                 "source": "TEST"}
    with pytest.raises(ConfigError, match="free relation member"):
        load(d)
    d = mini()
    d["variation"]["ring.heater.r_cold_ohm"]["line_sigma"] = 1.0
    with pytest.raises(ConfigError, match="line_sigma needs level line"):
        load(d)


# ------------------------------------------------------------------------------------------------ artefacts

def _artefact(root: Path, name="modes_test.json", data=None, good_sha=True, inputs=None):
    ext = root / "extracted"
    ext.mkdir(parents=True, exist_ok=True)
    p = ext / name
    data = data or {"n_g": 4.05, "dlambda_dt_pm_per_k": {"t_c": [25.0, 105.0], "value": [67.0, 75.0]},
                    "jacobian": {"d": ["width_nm"], "of": {"lambda_res_nm_per_nm": [1.1]}}}
    p.write_text(json.dumps(data))
    sha = hashlib.sha256(p.read_bytes()).hexdigest() if good_sha else "0" * 64
    side = {"schema": "modes/1", "sha256": sha}
    if inputs:
        side["inputs"] = inputs
    (ext / (name + ".sidecar.json")).write_text(json.dumps(side))
    return p


def test_from_artefact(tmp_path):
    _artefact(tmp_path, inputs={"geometry": {"provenance": "assumed", "width_nm": 400}})
    d = mini()
    d["ring"]["optics"]["n_g"] = {"from": "extracted/modes_test.json:n_g", "generation": "none", "source": "R06"}
    d["ring"]["thermal"]["dlambda_dt_pm_per_k"] = {
        "value": {"from": "extracted/modes_test.json:dlambda_dt_pm_per_k", "at": {"t_c": 65.0}},
        "provenance": "solver", "generation": "none", "source": "R06"}
    del d["uncertainty"]["thermo_optic"]            # it selects alternatives that this leaf no longer has
    cfg = load(d, data_root=tmp_path)
    ng = cfg.leaf("ring.optics.n_g")
    assert ng.value == 4.05 and ng.provenance == "solver(assumed-geometry)"
    assert provenance_rank(ng.provenance) == provenance_rank("assumed")
    assert ng.artefact["sha256"] and ng.artefact["key"] == "n_g"
    assert cfg.leaf("ring.thermal.dlambda_dt_pm_per_k").value == pytest.approx(71.0)


def test_from_artefact_sha_mismatch_and_missing(tmp_path):
    _artefact(tmp_path, good_sha=False)
    d = mini()
    d["ring"]["optics"]["n_g"] = {"from": "extracted/modes_test.json:n_g", "generation": "none", "source": "R06"}
    with pytest.raises(ArtefactError, match="sha256"):
        load(d, data_root=tmp_path)
    d["ring"]["optics"]["n_g"]["from"] = "extracted/nothing.json:n_g"
    with pytest.raises(ArtefactError, match="not found"):
        load(d, data_root=tmp_path)
    _artefact(tmp_path)
    d["ring"]["optics"]["n_g"]["from"] = "extracted/modes_test.json:n_gg"
    with pytest.raises(ArtefactError, match="n_gg"):
        load(d, data_root=tmp_path)
    d["ring"]["optics"]["n_g"] = {"from": "extracted/modes_test.json:n_g", "provenance": "printed",
                                  "generation": "none", "source": "R06"}
    with pytest.raises(ConfigError, match="provenance 'solver'"):
        load(d, data_root=tmp_path)


def test_from_artefact_in_alternative(tmp_path):
    _artefact(tmp_path)
    d = mini()
    d["ring"]["optics"]["n_g"]["alternatives"] = {
        "t200": {"from": "extracted/modes_test.json:n_g", "provenance": "solver"}}
    cfg = load(d, data_root=tmp_path, alts={"ring.optics.n_g": "t200"})
    assert cfg.leaf("ring.optics.n_g").value == 4.05
    assert cfg.leaf("ring.optics.n_g").artefact["file"] == "extracted/modes_test.json"


# ------------------------------------------------------------------------------------------------ review additions

def test_fidelity_rules_hold_in_every_hypothesis_world():
    """A per-world switch reached by a hypothesis must find its required keys (it gave nan full scales before)."""
    d = mini()
    d["uncertainty"]["heater_driver"]["choices"]["power"] = {"ring.heater.driver_mode": "power"}
    with pytest.raises(ConfigError, match=r"heater_driver=power.*full_scale_w"):
        load(d)
    d["ring"]["heater"]["full_scale_w"] = leafv(0.05)
    load(d)


def test_multiplicative_draws_of_affine_or_log_keys_rejected():
    d = mini()
    d["uncertainty"]["tref"] = {"kind": "interval", "target": "ring.optics.resonance_offset_ref_temp_c",
                                "mode": "scale", "low": 0.9, "high": 1.1, "provenance": "assumed",
                                "generation": "none", "source": "TEST"}
    with pytest.raises(ConfigError, match="affine"):
        load(d)
    d = mini()
    d["variation"]["laser.power_per_line_dbm"] = {"level": "line", "rel_sigma": 0.05, "provenance": "assumed",
                                                  "generation": "none", "source": "TEST"}
    with pytest.raises(ConfigError, match="logarithmic"):
        load(d)
    d["variation"]["laser.power_per_line_dbm"] = {"level": "line", "line_sigma": 0.5, "provenance": "assumed",
                                                  "generation": "none", "source": "TEST"}
    load(d)                                                         # an additive spread in dB is fine


def test_interval_and_dist_entry_checks():
    def entry(**kw):
        d = mini()
        d["uncertainty"]["x"] = dict(kw, provenance="assumed", generation="none", source="TEST")
        return d
    with pytest.raises(ConfigError, match="triangular"):
        load(entry(kind="interval", target="ring.optics.q_loaded", low=3500, high=4200, dist="triangular",
                   nominal=5000))
    with pytest.raises(ConfigError, match="physical bounds"):
        load(entry(kind="interval", target="ring.optics.q_loaded", low=10, high=4200))
    with pytest.raises(ConfigError, match="truncate"):
        load(entry(kind="dist", target="ring.monitor.dark_i0_na", mean=5.0, sigma=1.0, truncate=[30.0, 1.0]))
    with pytest.raises(ConfigError, match="lognormal needs mean > 0"):
        load(entry(kind="dist", target="ring.monitor.dark_i0_na", dist="lognormal", mean=0.0, sigma=1.0))


def test_alternative_values_and_ranges_are_checked():
    d = mini()
    d["ring"]["optics"]["q_loaded"]["alternatives"]["bad"] = {"value": "high", "provenance": "assumed"}
    with pytest.raises(ConfigError, match="alternatives.bad"):
        load(d)
    d = mini()
    d["ring"]["thermal"]["dlambda_dt_pm_per_k"]["alternatives"]["H86"]["value"] = 0.086    # nm/K typed as pm/K
    with pytest.raises(ConfigError, match="physical bounds"):
        load(d)
    d = mini()
    d["ring"]["optics"]["q_loaded"]["range"] = [3000]
    with pytest.raises(ConfigError, match="range must be"):
        load(d)


def test_hash_covers_alternatives_and_is_stable():
    """Two configs that realize differently must not share Config.hash(); drawing an alternative must not change
    it."""
    from mrm.params.realize import realize
    a = load(MINI)
    d = mini()
    d["ring"]["thermal"]["dlambda_dt_pm_per_k"]["alternatives"]["H86"]["value"] = 80.0
    assert load(d).hash() != a.hash()
    h0 = a.hash()
    realize(a, 8, 0, thermal_hook=None)
    assert a.hash() == h0 == load(MINI).hash()


def test_hash_stable_after_drawing_an_artefact_alternative(tmp_path):
    from mrm.params.realize import realize
    _artefact(tmp_path)
    d = mini()
    d["ring"]["optics"]["n_g"]["alternatives"] = {"t200": {"from": "extracted/modes_test.json:n_g",
                                                           "provenance": "solver"}}
    d["uncertainty"]["xsec"] = {"kind": "hypothesis", "target": "ring.optics.n_g", "choices": ["default", "t200"],
                                "nominal": "default", "provenance": "assumed", "generation": "none", "source": "TEST"}
    cfg = load(d, data_root=tmp_path)
    h0 = cfg.hash()
    p = realize(cfg, 16, 0, variation="off", thermal_hook=None)
    assert set(np.unique(p.optics.n_g)) == {4.0, 4.05}
    assert cfg.hash() == h0


def test_override_of_a_per_world_switch_holds_in_every_world():
    """--set ring.heater.driver_mode=current used to be overwritten by the hypothesis draw in every world."""
    from mrm.params.realize import realize
    cfg = load(MINI, overrides=["ring.heater.driver_mode=current"])
    assert cfg.uncertainty["heater_driver"].pinned == "current"
    for mode in ("nominal", "sample"):
        p = realize(cfg, 16, 0, uncertainty=mode, thermal_hook=None)
        assert np.all(p.heater.driver_mode == 1), mode
    # a value no choice gives: the hypothesis is pinned to its nominal and the switch keeps the override
    with pytest.warns(UserWarning, match="not a choice"):
        cfg = load(MINI, overrides={"ring.heater.driver_mode": "power", "ring.heater.full_scale_w": 0.05})
    p = realize(cfg, 8, 0, thermal_hook=None)
    assert np.all(p.heater.driver_mode == 2) and np.all(p.heater.full_scale_w == 0.05)


def test_hypothesis_nominal_must_be_the_config_as_written():
    """derive(), the table and explain show the leaves; the nominal world must therefore select exactly them."""
    d = mini()
    d["uncertainty"]["thermo_optic"]["nominal"] = "H86"
    with pytest.raises(ConfigError, match="nominal choice 'H86'"):
        load(d)
    d = mini()
    d["uncertainty"]["heater_driver"]["nominal"] = "current"
    with pytest.raises(ConfigError, match="nominal choice 'current'"):
        load(d)
    load(MINI, alts={"uncertainty.thermo_optic": "H86"})        # a pinned entry applies its choice: consistent


def test_derived_tag_only_on_relation_members():
    d = mini()
    d["ring"]["monitor"]["gain_scale"] = {"value": None, "provenance": "derived"}
    with pytest.raises(ConfigError, match="no relation computes it"):
        load(d)
