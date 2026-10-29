"""Skip the calibration tests unless the ``calibration`` marker is selected.

Read this first. tests/calibration checks the fit residuals and the calibrated preset. Its results are reported;
they do not block CI. Every test here carries the ``calibration`` marker and is skipped unless the marker is selected
(``pytest -m calibration tests/calibration``), so a plain ``pytest`` run (CI) does not block on calibration
results. The marker is registered in pyproject.toml.
"""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    """Add a skip marker to every ``calibration`` test when the -m expression does not name the marker."""
    expr = config.getoption("-m") or ""
    if "calibration" in expr:
        return
    skip = pytest.mark.skip(reason="calibration tests are reported, not CI-blocking: run pytest -m calibration")
    for item in items:
        if "calibration" in item.keywords:
            item.add_marker(skip)
