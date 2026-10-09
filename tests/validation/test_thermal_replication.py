"""07 VAL-8 (frozen as VAL-8a..e in mrm/validation/targets/validation.yaml): third-party replication of the thermal
pipeline on a published stack, the imec metal-heater ring of Coenen et al., JLT 2022.

Targets: heater-node tau 5.0 us (VAL-8a) and waveguide tau 10.0 us (VAL-8b), 63 % times, +/-30 %; heater efficiency
222 pm/mW without undercut (VAL-8c, +/-25 %) and 843 pm/mW with full undercut (VAL-8d, +/-30 %); the two-pole fit
3.18/13.01 us of a different imec device is a cross-check only (VAL-8e). The replication uses OUR materials
(mrm.materials) on the printed geometry and boundary conditions; dimensions imec does not print are assumptions,
swept in the artefact. Validation tests report the outcome; they do not block CI (07 section 4.1).
"""
from __future__ import annotations

import json

import pytest

from mrm.extraction import thermal_fem as tf

PATH = tf.OUT_DIR / "thermal_val8_imec.json"


@pytest.fixture(scope="module")
def val8():
    if not PATH.exists():
        pytest.skip("thermal_val8_imec.json not produced yet")
    return json.loads(PATH.read_text())


def test_val8_schema(val8):
    assert val8["schema"] == "thermal_val8/1"
    assert set(val8["target_ids"]) == {"VAL-8a", "VAL-8b", "VAL-8c", "VAL-8d", "VAL-8e"}
    assert val8["nominal"]["materials"] == "mrm"          # our materials, not imec's calibrated set
    for k in ("tau_heater_us", "tau_wg_us", "eta_h_c_per_mw", "r_h_c_per_mw"):
        assert val8["nominal"][k] > 0


def _report(val8, tid):
    c = val8["comparison"][tid]
    print(f"{tid}: predicted {c['predicted']:.3g}, target {c['target']:.3g}, rel {c['rel_err']:+.1%}, "
          f"tolerance {c['tolerance']:.0%}")
    return c


@pytest.mark.parametrize("tid", ["VAL-8a", "VAL-8b"])
def test_val8_time_constants(val8, tid):
    """Heater-node and waveguide 63 % times within the frozen tolerance."""
    c = _report(val8, tid)
    assert abs(c["rel_err"]) <= c["tolerance"]


@pytest.mark.parametrize("tid", ["VAL-8c", "VAL-8d"])
def test_val8_efficiency(val8, tid):
    """FEM DC (K/mW) times our own dlambda/dT of the imec rib (femwell) against the measured pm/mW."""
    c = _report(val8, tid)
    assert abs(c["rel_err"]) <= c["tolerance"]


def test_val8e_cross_check_reported(val8):
    c = val8["comparison"].get("VAL-8e")
    if c is None:
        pytest.skip("quick run")
    assert c["status"] == "cross_check"
    assert len(c["predicted"]["tau_us"]) == 2


def test_val8_mesh_converged(val8):
    ch = val8["comparison"]["mesh_change_last_level"]
    if ch is None:
        pytest.skip("single mesh level (quick run)")
    for k, v in ch.items():
        assert abs(v) < 0.01, k


@pytest.mark.slow
def test_val8_reruns(tmp_path):
    comp = tf.stage_val8([], [], tf.RunOptions.quick_options(), tmp_path)
    assert "VAL-8a" in comp
    d = json.loads((tmp_path / "thermal_val8_imec.json").read_text())
    assert d["nominal"]["tau_wg_us"] > d["nominal"]["tau_heater_us"]
