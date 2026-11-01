"""Checks run directories and manifests (mrm.sim.manifest).

Read this first
---------------
Every run writes a directory with a manifest that records what ran: the git state, the environment (the numerics
stack), and the sha256 of every artefact the config reads. These tests check those records, the strict JSON and the
bit-exact HDF5 writers, the run-directory names, and the bit-for-bit run comparison that ``mrm rerun`` uses.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from mrm.sim import manifest


def test_git_state_is_deterministic_and_complete():
    a, b = manifest.git_state(), manifest.git_state()
    if not a["available"]:
        pytest.skip("not a git work tree")
    assert len(a["commit"]) == 40 and a["commit"] == b["commit"]
    assert len(a["diff_sha256"]) == 64 and a["diff_sha256"] == b["diff_sha256"]
    assert isinstance(a["dirty"], bool)


def test_git_state_outside_a_repository(tmp_path):
    g = manifest.git_state(tmp_path)
    assert g["available"] is False and g["commit"] is None


def test_environment_records_the_numerics_stack():
    e = manifest.environment()
    for k in ("python", "numpy", "scipy", "numba", "llvmlite", "h5py", "platform", "machine", "cpu_count"):
        assert k in e
    assert e["numba"] and e["llvmlite"] and e["numpy"]


def test_artefacts_used_hash_every_file_the_g26_config_reads():
    from mrm import params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load("g26")
    arts = {a["file"]: a for a in manifest.artefacts_used(cfg)}
    # the leaves that take their value from the rib160 mode solve, the 200 nm alternative of film_thickness (the
    # rib200 artefact) and the strip stand-in that stays a plain alternative
    assert "extracted/modes_rib160.json" in arts and arts["extracted/modes_rib160.json"]["selected"]
    assert "extracted/modes_rib200.json" in arts and not arts["extracted/modes_rib200.json"]["selected"]
    assert "extracted/modes_strip400x200.json" in arts and not arts["extracted/modes_strip400x200.json"]["selected"]
    assert "extracted/coupling_rib160.json" in arts and arts["extracted/coupling_rib160.json"]["selected"]
    assert "materials.json" in arts
    for a in arts.values():
        assert a["exists"] and len(a["sha256"]) == 64
        assert not a.get("mismatch"), a
        if a["recorded_sha256"]:
            assert a["recorded_sha256"] == a["sha256"]
    m = arts["extracted/modes_rib160.json"]
    assert m["sidecar"] and len(m["sidecar_sha256"]) == 64


def test_to_jsonable_is_strict_json():
    obj = {"a": np.array([1.0, np.nan, np.inf, -np.inf]), "b": np.int64(3), "c": (1, 2), "d": np.bool_(True),
           "e": {"f": np.zeros((2, 2))}, "g": Path("x/y")}
    j = manifest.to_jsonable(obj)
    assert j["a"] == [1.0, None, "inf", "-inf"] and j["b"] == 3 and j["c"] == [1, 2] and j["d"] is True
    assert j["e"]["f"] == [[0.0, 0.0], [0.0, 0.0]] and j["g"] == "x/y"
    json.dumps(j, allow_nan=False)                   # no NaN token survives


def test_to_jsonable_takes_0d_float_arrays():
    """A 0-d float or complex array becomes its scalar (it must not raise 'iteration over a 0-d array')."""
    assert manifest.to_jsonable(np.array(1.5)) == 1.5 and manifest.to_jsonable(np.array(3)) == 3
    assert manifest.to_jsonable(np.array(np.nan)) is None and manifest.to_jsonable(np.array(-np.inf)) == "-inf"
    assert manifest.to_jsonable(np.array(1 + 2j)) == str(1 + 2j)
    assert manifest.to_jsonable({"x": np.float64(2.0) * np.ones(())}) == {"x": 2.0}


def test_h5_round_trip_keeps_every_bit(tmp_path):
    x = np.array([0.1, -0.0, np.nan, np.inf, 1e-308, 5e-324])
    tree = {"metrics": {"case|a": {"x": x, "i": np.arange(5, dtype=np.int64), "flag": np.array([True, False]),
                                   "nested": {"y": np.ones((2, 3))}, "note": "text", "lst": [1, 2, 3],
                                   "none": None}}}
    p = tmp_path / "t.h5"
    manifest.write_trace_h5(p, tree, attrs={"run_id": "r"})
    back = manifest.read_trace_h5(p)
    y = back["metrics"]["case|a"]["x"]
    assert y.tobytes() == x.tobytes()                # NaN payload and the sign of zero
    assert back["metrics"]["case|a"]["i"].dtype == np.int64
    assert np.array_equal(back["metrics"]["case|a"]["nested"]["y"], np.ones((2, 3)))
    assert back["metrics"]["case|a"]["@note"] == "text"
    assert back["@run_id"] == "r" and back["@schema"] == manifest.TRACE_SCHEMA


def _write_fake_run(d: Path, x: np.ndarray, run_id: str) -> Path:
    """A minimal run directory d (manifest.json and a trace.h5 holding metrics/main/x); returns d."""
    d.mkdir(parents=True)
    manifest.write_json(d / "manifest.json", {"schema": manifest.MANIFEST_SCHEMA, "run_id": run_id})
    manifest.write_trace_h5(d / "trace.h5", {"metrics": {"main": {"x": x}}}, attrs={"run_id": run_id})
    return d


def test_compare_runs_is_bit_exact(tmp_path):
    x = np.array([1.0, 2.0, np.nan])
    a = _write_fake_run(tmp_path / "a", x, "a")
    b = _write_fake_run(tmp_path / "b", x.copy(), "b")
    c = _write_fake_run(tmp_path / "c", np.array([1.0, 2.0 + 2e-16 * 2, np.nan]), "c")
    r = manifest.compare_runs(a, b)
    assert r["identical"] and r["n_compared"] == 2      # x and the schema attribute; the run ids are not compared
    r = manifest.compare_runs(a, c)
    assert not r["identical"] and r["differing"] == ["metrics/main/x"]
    assert math.isclose(r["max_abs_diff"]["metrics/main/x"], 4e-16, rel_tol=0.5)
    e = _write_fake_run(tmp_path / "e", np.array([1.0, 2.0, np.nan, 4.0]), "e")
    assert not manifest.compare_runs(a, e)["identical"]


def test_new_run_dir_names_and_collisions(tmp_path):
    import datetime as dt
    t = dt.datetime(2026, 10, 10, 12, 0, 0, tzinfo=dt.timezone.utc)
    a = manifest.new_run_dir(tmp_path, "1a2b3c4d", now=t)
    b = manifest.new_run_dir(tmp_path, "1a2b3c4d", now=t)
    assert a.name == "20261010T120000Z_1a2b3c4d" and b.name == "20261010T120000Z_1a2b3c4d_2"
    assert (a / "inputs").is_dir()


def test_runs_root_default_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MRM_RUNS_DIR", str(tmp_path / "r"))
    assert manifest.runs_root(None) == tmp_path / "r"
    assert manifest.runs_root(tmp_path / "x") == tmp_path / "x"
    monkeypatch.delenv("MRM_RUNS_DIR")
    assert manifest.runs_root(None) == Path.cwd() / "runs"


def test_artefacts_used_include_the_thermal_rom_in_rom_mode():
    """A config in ROM mode (no g26_rom preset exists yet, so a stand-in config object) hashes the ROM it names."""
    class RomCfg:
        leaves: dict = {}
        data_root = manifest.DATA_ROOT

        def get(self, key, default=None):
            return {"fidelity.thermal_model": "rom", "fidelity.junction_model": "analytic",
                    "ring.thermal.rom_file": "extracted/thermal_rom_inring_noundercut_160.npz"}.get(key, default)
    arts = {a["file"]: a for a in manifest.artefacts_used(RomCfg())}
    rom = arts["extracted/thermal_rom_inring_noundercut_160.npz"]
    assert rom["selected"] and rom["exists"] and len(rom["sha256"]) == 64
    assert rom["sidecar"] == "thermal_rom_inring_noundercut_160.json" and rom["used_by"] == ["ring.thermal.rom_file"]
