"""tests/calibration: fit residuals and the calibrated preset (07 4.1: reported, not CI-blocking).

Every test here carries the ``calibration`` marker and is skipped unless the marker is selected
(``pytest -m calibration tests/calibration``), so a plain ``pytest`` run (CI) does not block on calibration
results. The marker is registered in pyproject.toml.
"""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    expr = config.getoption("-m") or ""
    if "calibration" in expr:
        return
    skip = pytest.mark.skip(reason="calibration tests are reported, not CI-blocking: run pytest -m calibration")
    for item in items:
        if "calibration" in item.keywords:
            item.add_marker(skip)
