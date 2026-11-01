"""tools/tlc_matrix.py: which (experiment, controller) pairs a matrix file expands to, without running them."""
from __future__ import annotations

import importlib.util
from pathlib import Path

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
