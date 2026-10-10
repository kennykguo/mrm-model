"""Largest difference per signal over every scenario of a tools/perf_compare.py JSON (docs/handoff/perf.md).

    .venv/bin/python tools/perf_summarize.py runs/perf_x/compare.json [--rel]

Groups trace signals by field (detuning_m, t_ring_k, ...) over decimations and reducers, and summary metrics and final
Truth by name; prints the largest absolute difference (scaled units of tools/perf_compare.py), the largest difference
relative to the signal's largest magnitude, and the scenario where it occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    a = ap.parse_args(argv)
    d = json.loads(Path(a.json).read_text())
    best = {}
    for sc, rows in d.items():
        for key, r in rows.items():
            if "max_abs" not in r:
                continue
            if key.startswith("trace/"):
                name = "trace " + key.split("/")[1].split("@")[0]
            else:
                name = key.split("/")[0] + " " + "/".join(key.split("/")[1:])
            cur = best.get(name)
            if cur is None or r["max_abs_scaled"] > cur[0]:
                best[name] = (r["max_abs_scaled"], r["unit"], r["rel_to_max"], sc)
            elif cur is not None and r["rel_to_max"] > cur[2]:
                best[name] = (cur[0], cur[1], max(cur[2], r["rel_to_max"]), cur[3])
    for name in sorted(best):
        v, u, rel, sc = best[name]
        print(f"{name:45s} {v:10.3e} {u:3s}  rel {rel:9.2e}  ({sc})")


if __name__ == "__main__":
    main()
