"""Run the TLC scenario matrix and write the report (docs/TLC_PORT_SPEC_AND_PLAN.md 3, Phase 4).

    .venv/bin/python tools/tlc_matrix.py [experiments/tlc/matrix.yaml] [--only tlc,pi_lock_c] [--out docs/tlc/results.md]

Every experiment of the matrix file runs once per controller (overriding controller.name and controller.params);
TLC runs keep their telemetry (tools/tlc_run.py). The run directories are listed in ``<out>.runs.txt`` so the
report can be rebuilt with tools/tlc_report.py without rerunning.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def overrides_for(ctrl: dict) -> list[str]:
    return [f"controller.name={ctrl['name']}", f"controller.params={json.dumps(ctrl.get('params') or {})}"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("matrix", nargs="?", default="experiments/tlc/matrix.yaml")
    ap.add_argument("--only", default="", help="comma-separated controller names to run (default: all)")
    ap.add_argument("--out", default="docs/tlc/results.md")
    ap.add_argument("--runs-dir", default=None)
    args = ap.parse_args(argv)
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    only = {s for s in args.only.split(",") if s}
    run_tool, report = _tool("tlc_run"), _tool("tlc_report")
    runs = []
    for entry in matrix["experiments"]:
        exp, names = (entry, None) if isinstance(entry, str) else (entry["path"], set(entry["controllers"]))
        for ctrl in matrix["controllers"]:
            if (only and ctrl["name"] not in only) or (names is not None and ctrl["name"] not in names):
                continue
            print(f"== {exp} :: {ctrl['name']}", file=sys.stderr, flush=True)
            ov = overrides_for(ctrl) if names is None else []      # restricted entries keep their own controller
            runs.append(str(run_tool.run_with_telemetry(exp, args.runs_dir, ov)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}.runs.txt").write_text("\n".join(runs) + "\n")
    return report.main([*runs, "--out", str(out)])


if __name__ == "__main__":
    raise SystemExit(main())
