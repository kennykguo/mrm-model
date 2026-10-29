"""Compare two controller scorecards (the JSON of tests/validation/test_controllers_on_engine.py) number by number.

    .venv/bin/python tools/scorecard_compare.py runs/perf_baseline/controller_scorecard_8646691.json \
        runs/perf_scorecard/controller_scorecard.json [--json out.json] [--rename OLD=NEW ...]

Read this first. A scorecard is a nested JSON: controller -> scenario -> metric -> per-ring values. For every metric
(a key path that ends in a number or a list of numbers; wall times, notes and the keys in SKIP are left out) the
tool prints the largest absolute change over the controllers, scenarios and rings, and where it occurs. Booleans
and integers (locked, line, excursion counts) are reported as the number of entries that changed.

``--rename OLD=NEW`` renames a scenario key of the REFERENCE scorecard before the comparison (keys without a partner
are not compared). Example: the scorecards before 2026-10-10 hold ``a1`` (aggressor 5 um from ring 0's monitor PD)
and ``a1_pd_far``. The later ones hold ``a1`` (default site, 55 um) and ``a1_pd_adjacent`` (the stress site). So
``--rename a1=a1_pd_adjacent a1_pd_far=a1`` compares like with like (docs/handoff/integrator.md, "Aggressor site
moved; the old site is a stress scenario").
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

#: a key that contains one of these words is not compared (timings, text and identifiers)
SKIP = ("wall", "note", "errors", "path", "name", "start")


def _walk(a, b, path, out):
    """Append (key path, value in a, value in b, "discrete" | "float") for every number that both trees hold. The
    scorecard JSON stores NaN and inf as null, so null is read as NaN: a number that becomes null is a change."""
    a = float("nan") if a is None else a
    b = float("nan") if b is None else b
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
    ap.add_argument("--rename", nargs="*", default=[], metavar="OLD=NEW",
                    help="rename scenario keys of the reference scorecard (e.g. a1_pd_far=a1)")
    args = ap.parse_args(argv)
    ref = json.loads(Path(args.ref).read_text())
    new = json.loads(Path(args.new).read_text())
    if args.rename:
        renames = dict(r.split("=", 1) for r in args.rename)
        ref = {name: {renames.get(k, k): v for k, v in r.items()} for name, r in ref.items()}
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
        # x == x is False only for NaN (or null): NaN against NaN counts as no change, NaN against a number as
        # infinite.
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
    if args.json:
        Path(args.json).write_text(json.dumps(by_metric, indent=1, default=str))


if __name__ == "__main__":
    main()
