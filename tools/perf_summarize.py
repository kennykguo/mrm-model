"""Print the largest difference per signal, over every scenario, from the JSON that tools/perf_compare.py writes.

    .venv/bin/python tools/perf_summarize.py runs/perf_x/compare.json

Read this first. tools/perf_compare.py --json gives one row per scenario and per recorded key. This tool groups the
rows by signal: a trace by its field name (detuning_m, t_ring_k, ...) over all decimations and reducers, and a
summary metric or a final Truth value by its name. For each signal it prints:
- the largest absolute difference (in the scaled units of tools/perf_compare.py, for example pm or mK);
- the largest relative difference (a difference divided by the largest magnitude of the signal in the reference
  run). It can come from another row than the largest absolute difference;
- the scenario where the largest absolute difference occurs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    args = ap.parse_args(argv)
    compare_rows = json.loads(Path(args.json).read_text())
    # signal name -> (largest scaled |difference|, unit, largest relative difference, scenario of the first)
    worst = {}
    for scenario, rows in compare_rows.items():
        for key, row in rows.items():
            if "max_abs" not in row:
                continue                                   # a SHAPE or MISSING row has no numbers
            if key.startswith("trace/"):
                name = "trace " + key.split("/")[1].split("@")[0]      # trace/<field>@<dec>:<reducer>/value
            else:
                name = key.split("/")[0] + " " + "/".join(key.split("/")[1:])
            cur = worst.get(name)
            if cur is None:
                worst[name] = (row["max_abs_scaled"], row["unit"], row["rel_to_max"], scenario)
            elif row["max_abs_scaled"] > cur[0]:
                worst[name] = (row["max_abs_scaled"], row["unit"], max(cur[2], row["rel_to_max"]), scenario)
            elif row["rel_to_max"] > cur[2]:
                worst[name] = (cur[0], cur[1], row["rel_to_max"], cur[3])
    for name in sorted(worst):
        diff, unit, rel, scenario = worst[name]
        print(f"{name:45s} {diff:10.3e} {unit:3s}  rel {rel:9.2e}  ({scenario})")


if __name__ == "__main__":
    main()
