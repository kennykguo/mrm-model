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


def corner_alts(name: str) -> dict:
    """The alternatives a corner preset pins (mrm/params/presets/<name>.yaml, ``alts``)."""
    import mrm.params
    path = Path(mrm.params.__file__).parent / "presets" / f"{name}.yaml"
    return dict(yaml.safe_load(path.read_text()).get("alts") or {})


def expand_corners(matrix: dict, only: set | None = None) -> list[tuple[str, dict, list[str], str]]:
    """(experiment, controller, overrides, corner) for the matrix's ``corners`` section: each listed experiment and
    controller on each corner preset (spec 6, Q7: model trust)."""
    sec = matrix.get("corners") or {}
    sub = {"experiments": sec.get("experiments", []), "controllers": [
        c for c in matrix["controllers"] if c["name"] in set(sec.get("controllers", []))]}
    return [(e, c, ov, corner) for corner in sec.get("presets", []) for e, c, ov in expand(sub, only)]


def expand(matrix: dict, only: set | None = None) -> list[tuple[str, dict, list[str]]]:
    """(experiment path, controller entry, overrides) for every run of the matrix."""
    out = []
    for entry in matrix["experiments"]:
        exp, names = (entry, None) if isinstance(entry, str) else (entry["path"], set(entry["controllers"]))
        for ctrl in matrix["controllers"]:
            if only and ctrl["name"] not in only:
                continue
            if (names is None and ctrl.get("restricted_only")) or (names is not None and ctrl["name"] not in names):
                continue
            out.append((exp, ctrl, overrides_for(ctrl) if names is None else []))   # restricted: own controller
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("matrix", nargs="?", default="experiments/tlc/matrix.yaml")
    ap.add_argument("--only", default="", help="comma-separated controller names to run (default: all)")
    ap.add_argument("--out", default="docs/tlc/results.md")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--no-corners", action="store_true", help="skip the corners section")
    args = ap.parse_args(argv)
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    only = {s for s in args.only.split(",") if s}
    run_tool, report = _tool("tlc_run"), _tool("tlc_report")
    runs = []
    for exp, ctrl, ov in expand(matrix, only):
        print(f"== {exp} :: {ctrl['name']}", file=sys.stderr, flush=True)
        runs.append(str(run_tool.run_with_telemetry(exp, args.runs_dir, ov)))
    if not args.no_corners:
        for exp, ctrl, ov, corner in expand_corners(matrix, only):
            print(f"== {exp} :: {ctrl['name']} @ {corner}", file=sys.stderr, flush=True)
            runs.append(str(run_tool.run_with_telemetry(exp, args.runs_dir, ov, corner_alts(corner),
                                                        {"corner": corner})))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}.runs.txt").write_text("\n".join(runs) + "\n")
    return report.main([*runs, "--out", str(out)])


if __name__ == "__main__":
    raise SystemExit(main())
