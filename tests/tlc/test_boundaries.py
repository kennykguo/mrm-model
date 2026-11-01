"""Boundary rules for ``mrm/control/tlc`` (docs/TLC_PORT_SPEC_AND_PLAN.md 2.1): a controller sees only what the
hardware shows it, so nothing under the package may import the plant, the simulator or the validation oracle."""
from __future__ import annotations

import ast
from pathlib import Path

import mrm

FORBIDDEN = ("mrm.plant", "mrm.sim", "mrm.validation")
PKG = Path(mrm.__file__).parent / "control" / "tlc"


def _imports(path: Path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_package_exists():
    assert (PKG / "controller.py").is_file()


def test_no_plant_or_simulator_imports():
    bad = [(p.name, m) for p in PKG.rglob("*.py") for m in _imports(p) if m.startswith(FORBIDDEN)]
    assert bad == []
