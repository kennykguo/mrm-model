"""Compare two controller scorecards (tests/validation/test_controllers_on_engine.py JSON) number by number.

    .venv/bin/python tools/scorecard_compare.py runs/perf_baseline/controller_scorecard_8646691.json \
        runs/perf_scorecard/controller_scorecard.json [--json out.json]

For every metric (key path ending in a number or a list of numbers; wall times and notes excluded) prints the
largest absolute change over the controllers, scenarios and rings, and where it occurs; booleans and integers
(locked, line, excursion counts) are reported as the number of entries that changed.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

SKIP = ("wall", "note", "errors", "path", "name", "start")


def _walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k in b and not any(s in k for s in SKIP):
                _walk(a[k], b[k], path + (k,), out)
        return
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, path + (i,), out)
        return
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, int) and isinstance(b, int):
        out.append((path, a, b, "discrete"))
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        out.append((path, float(a), float(b), "float"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    ref = json.loads(Path(a.ref).read_text())
    new = json.loads(Path(a.new).read_text())
    rows = []
    _walk(ref, new, (), rows)
    by_metric = defaultdict(lambda: {"max_abs": 0.0, "where": None, "n": 0, "n_changed": 0, "kind": None})
    for path, x, y, kind in rows:
        # metric key: scenario (steady / a1_*) + metric name, without controller and ring index
        metric = "/".join(str(p) for p in path[1:] if not isinstance(p, int))
        m = by_metric[metric]
        m["kind"] = kind
        m["n"] += 1
        if kind == "discrete":
            if x != y:
                m["n_changed"] += 1
                m["where"] = m["where"] or f"{path[0]} {path[1:]}: {x} -> {y}"
            continue
        d = abs(y - x) if (x == x and y == y) else (0.0 if (x != x and y != y) else float("inf"))
        if d > 0:
            m["n_changed"] += 1
        if d > m["max_abs"]:
            m["max_abs"] = d
            m["where"] = f"{path[0]} {path[1:]}: {x:.6g} -> {y:.6g}"
    print(f"{'metric':45s} {'largest change':>16s}  changed/total  where")
    for metric, m in sorted(by_metric.items()):
        if m["kind"] == "discrete":
            print(f"{metric:45s} {'(discrete)':>16s}  {m['n_changed']:5d}/{m['n']:<5d}  {m['where'] or ''}")
        else:
            print(f"{metric:45s} {m['max_abs']:16.4g}  {m['n_changed']:5d}/{m['n']:<5d}  {m['where'] or ''}")
    missing = sorted(set(ref) - set(new))
    if missing:
        print("controllers missing in the new scorecard:", missing)
    if a.json:
        Path(a.json).write_text(json.dumps(by_metric, indent=1, default=str))


if __name__ == "__main__":
    main()
