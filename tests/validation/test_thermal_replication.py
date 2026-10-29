"""VAL-8: replicate a published thermal measurement with our own thermal FEM pipeline (a third-party check).

Read this first. The stack is the imec metal-heater ring of Coenen et al., JLT 2022. The targets are frozen as
VAL-8a..e in mrm/validation/targets/validation.yaml. The FEM results come from the artefact thermal_val8_imec.json
(mrm.extraction.thermal_fem); these tests read it and are skipped when it is missing.

Targets: heater-node tau 5.0 us (VAL-8a) and waveguide tau 10.0 us (VAL-8b), 63 % times, +/-30 %; heater efficiency
222 pm/mW without undercut (VAL-8c, +/-25 %) and 843 pm/mW with full undercut (VAL-8d, +/-30 %); the two-pole fit
3.18/13.01 us of a different imec device is a cross-check only (VAL-8e). The replication uses OUR materials
(mrm.materials) on the printed geometry and boundary conditions; dimensions imec does not print are assumptions,
swept in the artefact. Validation tests report the outcome; they do not block CI. The three tests of VAL-8a, VAL-8b
and VAL-8c are known failures (docs/KNOWN_ISSUES.md, "The thermal FEM artefact misses the imec targets").
"""
from __future__ import annotations

import json

import pytest

from mrm.extraction import thermal_fem

PATH = thermal_fem.OUT_DIR / "thermal_val8_imec.json"


@pytest.fixture(scope="module")
def val8():
    if not PATH.exists():
        pytest.skip("thermal_val8_imec.json not produced yet")
    return json.loads(PATH.read_text())


def test_val8_schema(val8):
    """The artefact has the val8 schema, all five target ids, our own materials and positive nominal results."""
    assert val8["schema"] == "thermal_val8/1"
    assert set(val8["target_ids"]) == {"VAL-8a", "VAL-8b", "VAL-8c", "VAL-8d", "VAL-8e"}
    assert val8["nominal"]["materials"] == "mrm"          # our materials, not imec's calibrated set
    for k in ("tau_heater_us", "tau_wg_us", "eta_h_c_per_mw", "r_h_c_per_mw"):
        assert val8["nominal"][k] > 0


def _report(val8, tid):
    """Print and return the comparison row of target ``tid`` (predicted, target, relative error, tolerance)."""
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
    """VAL-8e (the two-pole fit of another imec device) is reported as a cross-check with two time constants."""
    c = val8["comparison"].get("VAL-8e")
    if c is None:
        pytest.skip("quick run")
    assert c["status"] == "cross_check"
    assert len(c["predicted"]["tau_us"]) == 2


def test_val8_mesh_converged(val8):
    """Every result changed by less than 1 % at the last mesh refinement."""
    ch = val8["comparison"]["mesh_change_last_level"]
    if ch is None:
        pytest.skip("single mesh level (quick run)")
    for k, v in ch.items():
        assert abs(v) < 0.01, k


@pytest.mark.slow
def test_val8_reruns(tmp_path):
    """A quick rerun of the VAL-8 stage writes the artefact, and the waveguide is slower than the heater node."""
    comp = thermal_fem.stage_val8([], [], thermal_fem.RunOptions.quick_options(), tmp_path)
    assert "VAL-8a" in comp
    d = json.loads((tmp_path / "thermal_val8_imec.json").read_text())
    assert d["nominal"]["tau_wg_us"] > d["nominal"]["tau_heater_us"]
