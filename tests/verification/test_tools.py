"""The helper tools in tools/ report what their docstrings promise: the run comparisons (perf_compare.py,
perf_summarize.py, scorecard_compare.py), the heavy-job lock of memguard.py and the profile rows of perf_profile.py.

Read this first. Each tool is loaded from its file (tools/ is not a package) and run on small hand-made inputs in a
temporary directory. The memguard tests use their own lock directory, never the machine-wide /tmp lock.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _tool(name):
    """The module tools/<name>.py, loaded from its file."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _line(out, start):
    """The first output line that starts with ``start``."""
    return next(line for line in out.splitlines() if line.startswith(start))


# source descriptions survive changes between leading and inline constant comments.
def test_variable_map_reads_constant_descriptions(tmp_path):
    source = tmp_path / "rows.py"
    source.write_text(
        "# power leaving the ring (W)\n"
        "THROUGH_POWER = 0\n"
        "DROP_POWER: int = 1  # monitor power (W)\n"
        "UNCOMMENTED_AFTER_INLINE = 2\n"
        "NEXT_ROW = (\n"
        "    DROP_POWER + 1  # next storage position (1)\n"
        ")\n"
        "UNCOMMENTED_ROW = 3\n"
        "LABEL = 'this # stays inside the string'\n"
    )
    assert _tool("gen_variable_map")._constant_comments(source) == {
        "THROUGH_POWER": "power leaving the ring (W)",
        "DROP_POWER": "monitor power (W)",
        "UNCOMMENTED_AFTER_INLINE": "",
        "NEXT_ROW": "next storage position (1)",
        "UNCOMMENTED_ROW": "",
        "LABEL": "",
    }


# ------------------------------------------------------------------------------------------------------ perf_compare.py

def test_perf_compare_reports_missing_scenarios_and_signals(tmp_path, capsys):
    """A scenario file or a compared key that is in only one run is printed as MISSING, counts as a difference and
    makes the exit code 1. A crashed scenario or a dropped trace must not pass a bit-identity check."""
    perf_compare = _tool("perf_compare")
    ref, new = tmp_path / "ref", tmp_path / "new"
    ref.mkdir()
    new.mkdir()
    trace = "trace/t_ring_k@1:mean/value"
    np.savez(ref / "a.npz", **{"metric/x": [1.0], trace: [0.0, 1.0, 2.0]})
    np.savez(ref / "only_ref.npz", **{"metric/x": [1.0]})
    np.savez(new / "a.npz", **{"metric/x": [1.0], "metric/y": [2.0]})
    np.savez(new / "only_new.npz", **{"metric/x": [1.0]})
    assert perf_compare.main([str(ref), str(new)]) == 1
    out = capsys.readouterr().out
    assert "== a: 2 of 3 signals differ" in out
    assert f"{trace}: MISSING in the new run" in out
    assert "metric/y: MISSING in the reference run" in out
    assert "== only_ref: MISSING in the new run" in out
    assert "== only_new: MISSING in the reference run" in out


def test_perf_compare_identical_runs_exit_0(tmp_path, capsys):
    """Two identical runs show "0 of N signals differ" and exit with code 0."""
    perf_compare = _tool("perf_compare")
    for run in ("ref", "new"):
        (tmp_path / run).mkdir()
        np.savez(tmp_path / run / "a.npz", **{"metric/x": [1.0], "trace/t_ring_k@1:mean/value": [0.0, np.nan]})
    assert perf_compare.main([str(tmp_path / "ref"), str(tmp_path / "new")]) == 0
    assert "== a: 0 of 2 signals differ" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------------- perf_summarize.py

@pytest.mark.parametrize("order", [("s1", "s2"), ("s2", "s1")])
def test_perf_summarize_keeps_the_largest_relative_difference(tmp_path, capsys, order):
    """The relative difference printed for a signal is the largest over all rows, also when the row with the largest
    absolute difference has a smaller relative one."""
    perf_summarize = _tool("perf_summarize")
    key = "trace/t_ring_k@1:mean/value"
    rows = {"s1": {key: {"max_abs": 1e-3, "max_abs_scaled": 1.0, "unit": "mK", "rel_to_max": 0.5}},
            "s2": {key: {"max_abs": 2e-3, "max_abs_scaled": 2.0, "unit": "mK", "rel_to_max": 0.1}}}
    path = tmp_path / "compare.json"
    path.write_text(json.dumps({s: rows[s] for s in order}))
    perf_summarize.main([str(path)])
    line = _line(capsys.readouterr().out, "trace t_ring_k")
    assert "2.000e+00 mK" in line and "rel  5.00e-01" in line and "(s2)" in line


# ------------------------------------------------------------------------------------------------- scorecard_compare.py

def test_scorecard_compare_counts_a_value_that_becomes_null(tmp_path, capsys):
    """The scorecard JSON stores NaN and inf as null. A number that becomes null (a ring that stops locking) is an
    infinite change, and null against null is no change."""
    scorecard_compare = _tool("scorecard_compare")
    ref = {"pi": {"steady": {"t_lock_s": [0.001, 0.002], "locked": [True, True], "rms_pm": [None]}}}
    new = {"pi": {"steady": {"t_lock_s": [0.001, None], "locked": [True, None], "rms_pm": [None]}}}
    (tmp_path / "ref.json").write_text(json.dumps(ref))
    (tmp_path / "new.json").write_text(json.dumps(new))
    scorecard_compare.main([str(tmp_path / "ref.json"), str(tmp_path / "new.json")])
    out = capsys.readouterr().out
    assert _line(out, "steady/t_lock_s").split()[1:3] == ["inf", "1/2"]
    assert _line(out, "steady/locked").split()[1:3] == ["(discrete)", "1/2"]
    assert _line(out, "steady/rms_pm").split()[1:3] == ["0", "0/1"]


# ---------------------------------------------------------------------------------------------------------- memguard.py

def test_heavy_lock_appears_with_its_owner_file(tmp_path, monkeypatch):
    """The lock directory never exists without its owner file. A waiter that reads the lock while another process
    takes it would otherwise see no owner, call the lock stale and remove it, and two heavy jobs would run."""
    memguard = _tool("memguard")
    lock = tmp_path / "heavy.lock"
    owner_seen = []                         # one entry each time the lock directory may have appeared
    real_mkdir, real_rename = os.mkdir, os.rename

    def record():
        if lock.is_dir():
            owner_seen.append((lock / "owner").is_file())

    def mkdir(path, *args, **kwargs):
        real_mkdir(path, *args, **kwargs)
        record()

    def rename(src, dst, *args, **kwargs):
        real_rename(src, dst, *args, **kwargs)
        record()

    monkeypatch.setattr(memguard.os, "mkdir", mkdir)
    monkeypatch.setattr(memguard.os, "rename", rename)
    memguard.take_heavy_lock(["job"], str(lock))
    assert owner_seen and all(owner_seen)
    assert (lock / "owner").read_text().split()[0] == str(os.getpid())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["heavy.lock"]     # no temporary directory left behind


def test_heavy_lock_replaces_a_stale_lock(tmp_path):
    """A lock whose owner process is dead is removed and taken."""
    memguard = _tool("memguard")
    lock = tmp_path / "heavy.lock"
    lock.mkdir()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (lock / "owner").write_text(f"{dead.pid} old job\n")
    memguard.take_heavy_lock(["job"], str(lock))
    assert (lock / "owner").read_text().split()[0] == str(os.getpid())


def test_heavy_lock_waits_for_a_live_owner(tmp_path, monkeypatch):
    """A lock whose owner is alive is left alone: the waiter sleeps and tries again."""
    memguard = _tool("memguard")
    lock = tmp_path / "heavy.lock"
    lock.mkdir()
    (lock / "owner").write_text(f"{os.getpid()} other job\n")

    class Slept(Exception):
        pass

    def sleep(_seconds):
        raise Slept

    monkeypatch.setattr(memguard.time, "sleep", sleep)
    with pytest.raises(Slept):
        memguard.take_heavy_lock(["job"], str(lock))
    assert (lock / "owner").read_text() == f"{os.getpid()} other job\n"


# ------------------------------------------------------------------------------------------------------ perf_profile.py

@pytest.mark.slow
def test_perf_profile_prints_newton_iterations_of_the_implicit_step():
    """The implicit self-heating row prints its Newton iterations per ring per tick (p_implicit counts them; the
    kernel counter world_int WI_SH_ITER that the other rows read does not move in this loop)."""
    cmd = [sys.executable, str(ROOT / "tools" / "perf_profile.py"), "--only", "implicit", "--n", "50"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=ROOT, check=True).stdout
    row = _line(out, "implicit self-heating step")
    assert "newton it/ring/tick" in row
    assert float(row.split("newton it/ring/tick")[1]) >= 1.0
