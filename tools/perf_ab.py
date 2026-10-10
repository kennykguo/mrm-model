"""Back-to-back timing of two source trees (docs/handoff/perf.md): A/B/A/B... runs of tools/perf_bench.py.

    .venv/bin/python tools/perf_ab.py <tree_a> <tree_b> [--rounds 3] [--cases b1,b1lock] [--reps 5] [--ticks 3000]
                                      [--set-b key=value ...] [--json out.json]

Each tree is a checkout of the repository (its own numba cache under its own mrm/; compile each once first). The
two trees run alternately in fresh processes under the same conditions; the report gives, per case and tree, the
median over rounds of the per-run medians (wall and CPU time per world-tick) and the load average of each run.
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
LINE = re.compile(r"\[perf_bench\] (\S*) (\w+): wall ([\d.]+) us/world-tick \(min ([\d.]+), max ([\d.]+)\), cpu "
                  r"([\d.]+) us \(per thread ([\d.]+)\), newton it/ring/tick ([\d.]+), load ([\d.]+) -> ([\d.]+)"
                  r"(?:, power (.*))?")


def run(tree, cases, reps, ticks, sets, label):
    cmd = [PY, str(Path(tree) / "tools" / "perf_bench.py"), "--cases", cases, "--reps", str(reps), "--ticks",
           str(ticks), "--label", label]
    for s in sets:
        cmd += ["--set", s]
    out = subprocess.run(cmd, cwd=tree, capture_output=True, text=True).stdout
    rows = {}
    for m in LINE.finditer(out):
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
    a = ap.parse_args(argv)
    res = {"A": [], "B": []}
    for r in range(a.rounds):
        for tag, tree, sets in (("A", a.a, a.set_a), ("B", a.b, a.set_b)):
            rows = run(tree, a.cases, a.reps, a.ticks, sets, tag)
            res[tag].append(rows)
            print(f"round {r} {tag}: " + ", ".join(f"{c} {v['wall']:.2f} us (cpu {v['cpu_thread']:.2f}, load "
                                                   f"{v['load']:.2f})" for c, v in rows.items()), flush=True)
    summary = {}
    for c in a.cases.split(","):
        s = {}
        for tag in ("A", "B"):
            w = [x[c]["wall"] for x in res[tag]]
            u = [x[c]["cpu_thread"] for x in res[tag]]
            ld = [x[c]["load"] for x in res[tag]]
            nw = [x[c]["newton"] for x in res[tag]]
            s[tag] = {"wall_us": float(np.median(w)), "cpu_us": float(np.median(u)), "load": [min(ld), max(ld)],
                      "newton": float(np.median(nw)), "power": sorted({x[c]["power"] for x in res[tag]})}
        s["speedup_wall"] = s["A"]["wall_us"] / s["B"]["wall_us"]
        s["speedup_cpu"] = s["A"]["cpu_us"] / s["B"]["cpu_us"]
        summary[c] = s
        print(f"[perf_ab] {c}: A {s['A']['wall_us']:.2f} us (cpu {s['A']['cpu_us']:.2f}) | B {s['B']['wall_us']:.2f} "
              f"us (cpu {s['B']['cpu_us']:.2f}) | x{s['speedup_wall']:.2f} wall, x{s['speedup_cpu']:.2f} cpu | "
              f"load A {s['A']['load'][0]:.1f}-{s['A']['load'][1]:.1f}, B {s['B']['load'][0]:.1f}-"
              f"{s['B']['load'][1]:.1f} | newton it A {s['A']['newton']:.3f} B {s['B']['newton']:.3f} | power "
              f"{'; '.join(s['A']['power'] + s['B']['power'])}")
    if a.json:
        p = Path(a.json)
        old = json.loads(p.read_text()) if p.exists() else []
        old.append({"a": a.a, "b": a.b, "set_a": a.set_a, "set_b": a.set_b, "summary": summary, "runs": res})
        p.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
