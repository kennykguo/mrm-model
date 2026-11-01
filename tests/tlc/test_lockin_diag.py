"""Pieces of tools/tlc_lockin_diag.py that do not need a plant run, checked against closed-form cases."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("tlc_lockin_diag", ROOT / "tools" / "tlc_lockin_diag.py")
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)


def test_window_average_of_a_constant_is_the_constant():
    t = np.arange(1000) * 1e-4
    out = diag.window_average(t, np.full(1000, 3.0), np.array([0.05, 0.09]), 4e-3)
    np.testing.assert_allclose(out, 3.0)


def test_window_average_of_a_ramp_is_its_value_at_the_window_centre():
    t = np.arange(1000) * 1e-4
    out = diag.window_average(t, 2.0 * t, np.array([0.05]), 4e-3)       # Hann is symmetric: centre = t - 2 ms
    assert out[0] == pytest.approx(2.0 * (0.05 - 2e-3 + 0.5e-4), abs=1e-6)


def test_quality_summary():
    rows = [{"z1_mean": -0.01, "z1_std": 1e-4, "z1_model": -0.0125, "zeta_mean": 0.0, "zeta_std": 1e-3,
             "zeta_model": z} for z in (0.004, 0.0, -0.002)]
    q = diag.quality({"rows": rows})
    assert q["z1_ratio_median"] == pytest.approx(0.8)
    assert q["zeta_span_over_noise"] == pytest.approx(6.0)
    assert q["zeta_bias_max"] == pytest.approx(0.004)
