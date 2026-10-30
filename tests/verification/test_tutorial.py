"""Checks that the tutorial notebook examples/tutorial.ipynb runs end to end.

Read this first
---------------
nbconvert / nbclient are not installed in this environment, so this file carries a minimal executor: every code cell
runs in order in one namespace of a fresh interpreter (a subprocess, as a notebook kernel), its stdout and stderr are
captured as stream outputs, and nothing else (no rich outputs; the notebook prints text only). Run it as a script to
refresh the stored outputs::

    .venv/bin/python tools/memguard.py -- .venv/bin/python tests/verification/test_tutorial.py --write

The test checks that the notebook runs without an error in under 5 min (the requirement: "runs end to end in under
~5 min"), that its stored outputs are small, and that the stored outputs come from an execution (every code cell has
an execution count). About 10 s with a warm numba cache (the first run in a fresh environment compiles the
plant kernel: minutes).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "examples" / "tutorial.ipynb"
MAX_OUTPUT_BYTES = 24_000   # bytes, upper limit of the stored text outputs of all cells
TIME_LIMIT_S = 300.0        # s, the notebook must run in less time than this


def execute(path: Path, write: bool = False) -> dict:
    """Run every code cell of the notebook at ``path`` in order (cwd = the repository root); return {'wall_s',
    'cells', 'output_bytes'}; raise RuntimeError naming the cell on an error. ``write`` stores the outputs."""
    nb = json.loads(path.read_text())
    ns: dict = {"__name__": "__main__"}
    count = 0
    t0 = time.perf_counter()
    out_bytes = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        src = "".join(cell["source"])
        so, se = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
                exec(compile(src, f"<tutorial cell {count}>", "exec"), ns)
        except Exception:                                    # noqa: BLE001 - report the cell and re-raise
            raise RuntimeError(f"cell {count} failed:\n{src}\n\n{traceback.format_exc()}") from None
        outs = []
        for name, buf in (("stdout", so), ("stderr", se)):
            text = buf.getvalue()
            if text:
                lines = text.splitlines(keepends=True)
                outs.append({"name": name, "output_type": "stream", "text": lines})
                out_bytes += len(text.encode())
        cell["execution_count"] = count
        cell["outputs"] = outs
    wall = time.perf_counter() - t0
    if write:
        path.write_text(json.dumps(nb, indent=1) + "\n")
    return {"wall_s": wall, "cells": count, "output_bytes": out_bytes}


def test_tutorial_notebook_runs_end_to_end():
    env = dict(os.environ)
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, __file__, "--run", str(NOTEBOOK)], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=2 * TIME_LIMIT_S)
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    res = json.loads(p.stdout.strip().splitlines()[-1])
    assert res["wall_s"] < TIME_LIMIT_S and time.perf_counter() - t0 < TIME_LIMIT_S + 60
    assert res["output_bytes"] < MAX_OUTPUT_BYTES


def test_stored_notebook_is_executed_and_small():
    nb = json.loads(NOTEBOOK.read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code and all(c["execution_count"] == i + 1 for i, c in enumerate(code))
    size = sum(len("".join(o.get("text", ""))) for c in code for o in c["outputs"])
    assert size < MAX_OUTPUT_BYTES
    assert all(o["output_type"] == "stream" for c in code for o in c["outputs"])
    assert not any(o["name"] == "stderr" for c in code for o in c["outputs"]), "warnings leaked into the outputs"


if __name__ == "__main__":
    args = sys.argv[1:]
    write = "--write" in args
    paths = [a for a in args if not a.startswith("--")]
    target = Path(paths[0]) if paths else NOTEBOOK
    os.chdir(ROOT)
    result = execute(target, write=write)
    print(json.dumps(result))
