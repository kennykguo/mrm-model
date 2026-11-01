"""Shared fixtures for the TLC tests: access to the ECE496 source checkout (plant, simulator, original core).

The source lives beside this repository (``../ECE496_project``) or at ``$ECE496_PATH``; tests that need it are
skipped when it is absent. The source uses top-level module names (``params``, ``plant``, ``sim``, ``tlc``), so the
fixture puts ``thermal_control`` on ``sys.path`` for the test session.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("ECE496_PATH", ROOT.parent / "ECE496_project")) / "thermal_control"
SOURCE_COMMIT = "f98ce84"


@pytest.fixture(scope="session")
def ece496():
    """The ECE496 ``thermal_control`` directory on sys.path; skips the test if the checkout is missing."""
    if not (SOURCE / "sim.py").is_file():
        pytest.skip(f"ECE496 source not found at {SOURCE} (set ECE496_PATH)")
    if str(SOURCE) not in sys.path:
        sys.path.insert(0, str(SOURCE))
    return SOURCE
