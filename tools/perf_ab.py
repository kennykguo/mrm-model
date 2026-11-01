"""Time two source trees back to back: A/B/A/B... runs of tools/perf_bench.py in fresh processes.

    .venv/bin/python tools/perf_ab.py <tree_a> <tree_b> [--rounds 3] [--cases b1,b1lock] [--reps 5] [--ticks 3000]
                                      [--set-b key=value ...] [--json out.json]

Read this first. Each tree is a checkout of the repository with its own numba cache under its own mrm/ (compile
each tree once before timing). The two trees run alternately, so that both see the same machine load. For each
case and tree the report gives the median over rounds of the per-run medians (wall time and CPU time per
world-tick, us), the self-heating Newton iterations per ring and tick, and the range of the load average.
``--set-a`` / ``--set-b`` pass Plant keyword overrides (``--set`` of tools/perf_bench.py) to one tree only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

PY = sys.executable
#: the summary line that tools/perf_bench.py prints for each case (groups: label, case, wall median, min, max,
#: cpu, cpu per thread, newton iterations, load before, load after, power state)
BENCH_LINE = re.compile(r"\[perf_bench\] (\S*) (\w+): wall ([\d.]+) us/world-tick \(min ([\d.]+), max ([\d.]+)\), "
                        r"cpu ([\d.]+) us \(per thread ([\d.]+)\), newton it/ring/tick ([\d.]+), load ([\d.]+) -> "
                        r"([\d.]+)(?:, power (.*))?")


def run_bench(tree, cases, reps, ticks, sets, label):
    """Run tools/perf_bench.py of ``tree`` once in a fresh process; {case: numbers} parsed from its output."""
    cmd = [PY, str(Path(tree) / "tools" / "perf_bench.py"), "--cases", cases, "--reps", str(reps), "--ticks",
           str(ticks), "--label", label]
    for s in sets:
        cmd += ["--set", s]
    out = subprocess.run(cmd, cwd=tree, capture_output=True, text=True).stdout
    rows = {}
    for m in BENCH_LINE.finditer(out):
        rows[m.group(2)] = {"wall": float(m.group(3)), "wall_min": float(m.group(4)), "cpu": float(m.group(6)),
                            "cpu_thread": float(m.group(7)), "newton": float(m.group(8)),
                            "load": float(m.group(9)), "power": (m.group(11) or "?").strip()}
    if not rows:
        raise RuntimeError(out)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--cases", default="b1,b1lock")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--set-a", action="append", default=[])
    ap.add_argument("--set-b", action="append", default=[])
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)
    res = {"A": [], "B": []}                     # tree tag -> one {case: numbers} per round
    for r in range(args.rounds):
        for tag, tree, sets in (("A", args.a, args.set_a), ("B", args.b, args.set_b)):
            rows = run_bench(tree, args.cases, args.reps, args.ticks, sets, tag)
            res[tag].append(rows)
            print(f"round {r} {tag}: " + ", ".join(f"{c} {v['wall']:.2f} us (cpu {v['cpu_thread']:.2f}, load "
                                                   f"{v['load']:.2f})" for c, v in rows.items()), flush=True)
    summary = {}
    for case in args.cases.split(","):
        s = {}
        for tag in ("A", "B"):
            wall = [x[case]["wall"] for x in res[tag]]
            cpu = [x[case]["cpu_thread"] for x in res[tag]]
            load = [x[case]["load"] for x in res[tag]]
            newton = [x[case]["newton"] for x in res[tag]]
            s[tag] = {"wall_us": float(np.median(wall)), "cpu_us": float(np.median(cpu)),
                      "load": [min(load), max(load)], "newton": float(np.median(newton)),
                      "power": sorted({x[case]["power"] for x in res[tag]})}
        s["speedup_wall"] = s["A"]["wall_us"] / s["B"]["wall_us"]
        s["speedup_cpu"] = s["A"]["cpu_us"] / s["B"]["cpu_us"]
        summary[case] = s
        print(f"[perf_ab] {case}: A {s['A']['wall_us']:.2f} us (cpu {s['A']['cpu_us']:.2f}) | B "
              f"{s['B']['wall_us']:.2f} us (cpu {s['B']['cpu_us']:.2f}) | x{s['speedup_wall']:.2f} wall, "
              f"x{s['speedup_cpu']:.2f} cpu | load A {s['A']['load'][0]:.1f}-{s['A']['load'][1]:.1f}, B "
              f"{s['B']['load'][0]:.1f}-{s['B']['load'][1]:.1f} | newton it A {s['A']['newton']:.3f} B "
              f"{s['B']['newton']:.3f} | power {'; '.join(s['A']['power'] + s['B']['power'])}")
    if args.json:
        p = Path(args.json)
        old = json.loads(p.read_text()) if p.exists() else []
        old.append({"a": args.a, "b": args.b, "set_a": args.set_a, "set_b": args.set_b, "summary": summary,
                    "runs": res})
        p.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
