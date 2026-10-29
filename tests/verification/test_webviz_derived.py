"""The plain scenario paragraphs of the webviz site (webviz/derived.py, describe) quote the numbers of the ring they
name.

describe() is called with small hand-made inputs here (no engine run): only the fields that it reads.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def derived():
    """webviz/derived.py as a module (webviz is not a package; generate.py imports it as `derived`)."""
    spec = importlib.util.spec_from_file_location("webviz_derived", ROOT / "webviz" / "derived.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _a1_inputs(distance_um, max_err_pm, rail_ms, first_ms):
    """describe() inputs for an a1_aggressor run of len(distance_um) rings (per-ring lists in ring order)."""
    n = len(distance_um)
    R = {"sd": {"id": "a1_aggressor"}}
    sj = {"summary": {"per_ring": {"max_abs_lock_error_pm": max_err_pm}, "controller": "pi",
                      "duration_s": 0.1357, "lock_threshold_pm": 20.0, "score_from_s": 0.002},
          "drives": {"t_base_k": [318.15, 318.15], "aggressor_w": {"aggressor": [0.0, 0.5, 0.1]}},
          "t_s": [0.0, 0.066, 0.135]}
    stats = {"quantities": [],
             "limits": [{"key": "heater_lo", "scored": {"time_at_ms": rail_ms, "first_ms": first_ms}}]}
    devj = {"laser": {"p_chip_mw": [[2.5] * n, [2.5] * n]}}
    meta_geo = {"ring_to_site_um": {"aggressor": distance_um}}
    return R, sj, stats, devj, meta_geo


def test_a1_paragraph_quotes_the_ring_nearest_the_aggressor(derived):
    """The aggressor is nearest ring 2: the paragraph names ring 2 and quotes ring 2's rail time, first rail time
    and lock error; "the other rings" are rings 0 and 1."""
    text = derived.describe(*_a1_inputs(distance_um=[80.0, 55.0, 30.0], max_err_pm=[11.0, 22.0, 333.0],
                                        rail_ms=[0.0, 0.0, 4.5], first_ms=[None, None, 61.25]), period_s=0.27)
    assert "sits 30 µm from ring 2" in text
    assert ("Ring 2 gets the most heat: its heater reaches code 0 at 61.25 ms and stays at that rail for 4.5 ms in "
            "total, so the ring leaves its operating point by up to 333 pm. The other rings stay within 22 pm.") in text


def test_a1_paragraph_ring_0_nearest(derived):
    """The g26 layout: the aggressor is nearest ring 0, which is quoted; the other rings are rings 1 to N-1."""
    text = derived.describe(*_a1_inputs(distance_um=[30.0, 55.0, 80.0], max_err_pm=[455.0, 12.0, 7.0],
                                        rail_ms=[9.75, 0.0, 0.0], first_ms=[58.5, None, None]), period_s=0.27)
    assert ("Ring 0 gets the most heat: its heater reaches code 0 at 58.5 ms and stays at that rail for 9.75 ms in "
            "total, so the ring leaves its operating point by up to 455 pm. The other rings stay within 12 pm.") in text
