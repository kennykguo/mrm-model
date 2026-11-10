"""tools/tlc_matrix.py: which (experiment, controller) pairs a matrix file expands to, without running them."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("tlc_matrix", ROOT / "tools" / "tlc_matrix.py")
tlc_matrix = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tlc_matrix)


def test_shipped_matrix_expands_restricted_entries_and_restricted_controllers():
    matrix = yaml.safe_load((ROOT / "experiments/tlc/matrix.yaml").read_text())
    pairs = tlc_matrix.expand(matrix)
    names = {c["name"] for c in matrix["controllers"]}
    plain = [e for e in matrix["experiments"] if isinstance(e, str)]
    assert all((e, c) in [(p[0], p[1]["name"]) for p in pairs] for e in plain for c in names - {"wdm_supervisor"})
    assert ("experiments/tlc/cold_wdm.yaml", "wdm_supervisor") in [(p[0], p[1]["name"]) for p in pairs]
    assert not any(p[1]["name"] == "wdm_supervisor" and p[0] in plain for p in pairs)
    assert all(p[2] == [] for p in pairs if p[0] == "experiments/tlc/cold_tlc.yaml")   # keeps its own params


def test_only_filter():
    matrix = {"experiments": ["a.yaml", {"path": "b.yaml", "controllers": ["tlc"]}],
              "controllers": [{"name": "tlc", "params": {"core": "faithful"}}, {"name": "pi_lock_c"}]}
    pairs = tlc_matrix.expand(matrix, only={"pi_lock_c"})
    assert [(p[0], p[1]["name"]) for p in pairs] == [("a.yaml", "pi_lock_c")]
    assert tlc_matrix.expand(matrix)[0][2] == ['controller.name=tlc', 'controller.params={"core": "faithful"}']


def test_corners_section_expands_each_preset():
    matrix = yaml.safe_load((ROOT / "experiments/tlc/matrix.yaml").read_text())
    runs = tlc_matrix.expand_corners(matrix)
    variants = tlc_matrix.corner_variants(matrix)
    assert len(runs) == sum(len(v["presets"]) * len(v["experiments"]) * len(v["controllers"]) for v in variants)
    assert {r[3] for r in runs} == {p for v in variants for p in v["presets"]}


# the corner runs of the first pass (docs/tlc/SUMMARY.md, section 5) must stay as they were
CORNER_PRESETS = ["g26_fast_thermal", "g26_slow_thermal", "g26_textbook_thermal", "g26_tec_package"]
CORNER_EXPERIMENTS = ["experiments/tlc/steady_tlc_faithful.yaml", "experiments/tlc/a1_tlc.yaml"]


def test_shipped_corner_priors_runs_keep_the_first_pass_and_add_the_model_based_comparators():
    matrix = yaml.safe_load((ROOT / "experiments/tlc/matrix.yaml").read_text())
    runs = [(r[0], r[1]["name"], r[3]) for r in tlc_matrix.expand_corners(matrix) if r[4] == "corner"]
    first_pass = [(e, c, p) for p in CORNER_PRESETS for e in CORNER_EXPERIMENTS for c in ("tlc", "pi_lock_c")]
    assert [r for r in runs if r[1] in ("tlc", "pi_lock_c")] == first_pass
    names = ("tlc", "pi_lock_c", "kalman_lqg_c", "mpc_c")
    assert set(runs) == {(e, c, p) for p in CORNER_PRESETS for e in CORNER_EXPERIMENTS for c in names}
    assert len(runs) == len(set(runs))


def test_shipped_mismatch_runs_cover_the_corner_table_with_the_model_based_comparators():
    matrix = yaml.safe_load((ROOT / "experiments/tlc/matrix.yaml").read_text())
    runs = [r for r in tlc_matrix.expand_corners(matrix) if r[4] == "nominal"]
    names = ("tlc", "pi_lock_c", "kalman_lqg_c", "mpc_c")
    want = {(e, c, p) for p in CORNER_PRESETS for e in CORNER_EXPERIMENTS for c in names}
    assert {(r[0], r[1]["name"], r[3]) for r in runs} == want
    assert len(runs) == len(want)


def test_corner_mapping_without_knowledge_defaults_to_corner_priors():
    matrix = {"controllers": [{"name": "tlc", "params": {"core": "faithful"}}, {"name": "pi_lock_c"}],
              "corners": {"presets": ["g26_fast_thermal"], "experiments": ["a.yaml"], "controllers": ["tlc"]}}
    runs = tlc_matrix.expand_corners(matrix)
    assert [(r[0], r[1]["name"], r[3], r[4]) for r in runs] == [("a.yaml", "tlc", "g26_fast_thermal", "corner")]


def test_corner_variants_list_and_knowledge_filter():
    matrix = {"controllers": [{"name": "tlc"}, {"name": "mpc_c"}],
              "corners": [{"presets": ["c1"], "experiments": ["a.yaml"], "controllers": ["tlc"]},
                          {"knowledge": "nominal", "presets": ["c1", "c2"], "experiments": ["a.yaml"],
                           "controllers": ["tlc", "mpc_c"]}]}
    assert len(tlc_matrix.expand_corners(matrix)) == 5
    nominal = tlc_matrix.expand_corners(matrix, knowledge="nominal")
    assert [(r[1]["name"], r[3], r[4]) for r in nominal] == [("tlc", "c1", "nominal"), ("mpc_c", "c1", "nominal"),
                                                             ("tlc", "c2", "nominal"), ("mpc_c", "c2", "nominal")]
    assert [r[1]["name"] for r in tlc_matrix.expand_corners(matrix, only={"mpc_c"})] == ["mpc_c", "mpc_c"]


def test_unknown_corner_knowledge_is_refused():
    matrix = {"controllers": [{"name": "tlc"}],
              "corners": {"knowledge": "truth", "presets": ["c1"], "experiments": ["a.yaml"], "controllers": ["tlc"]}}
    with pytest.raises(ValueError, match="knowledge"):
        tlc_matrix.expand_corners(matrix)


def test_corner_alts_come_from_the_preset():
    assert tlc_matrix.corner_alts("g26_fast_thermal") == {"uncertainty.thermal.time_scale_fast": "low",
                                                          "uncertainty.thermal.time_scale_slow": "low"}
