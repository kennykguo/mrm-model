"""Compare two runs of tools/perf_scenarios.py signal by signal (docs/handoff/perf.md).

    .venv/bin/python tools/perf_compare.py runs/perf_baseline/ref runs/perf_x [--json out.json] [--all]

For every scenario present in both directories and every recorded trace (``<field>@<dec>:<reducer>``), prints the
largest absolute difference, the largest difference relative to the signal's range in the baseline, and the first
sample where they differ; then the summary metrics. Units: detuning / resonance in pm, temperatures in K (mK in the
table), powers in W, currents in A, codes as codes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCALE = {"detuning_m": (1e12, "pm"), "lam_res_m": (1e12, "pm"), "t_ring_k": (1e3, "mK"), "t_heater_k": (1e3, "mK"),
         "t_pd_k": (1e3, "mK"), "p_abs_w": (1e6, "uW"), "p_heater_w": (1e6, "uW"), "p_drop_w": (1e6, "uW"),
         "i_monitor_a": (1e9, "nA"), "p_in_w": (1e6, "uW"), "oma_w": (1e6, "uW")}


def compare(ref_dir: Path, new_dir: Path, show_all=False):
    out = {}
    for f in sorted(ref_dir.glob("*.npz")):
        g = new_dir / f.name
        if not g.exists():
            continue
        a = np.load(f)
        b = np.load(g)
        rows = {}
        for key in a.files:
            if key not in b.files or not key.endswith("/value") and not key.startswith(("metric/", "final/")):
                continue
            x = np.asarray(a[key], float)
            y = np.asarray(b[key], float)
            if x.shape != y.shape:
                rows[key] = {"shape": [list(x.shape), list(y.shape)]}
                continue
            fin = np.isfinite(x) & np.isfinite(y)
            same_nan = (np.isnan(x) & np.isnan(y)) | (np.isinf(x) & np.isinf(y) & (np.sign(x) == np.sign(y)))
            with np.errstate(invalid="ignore"):
                d = np.where(fin, np.abs(x - y), np.where(same_nan, 0.0, np.inf))
            mx = float(d.max()) if d.size else 0.0
            rng = float(np.nanmax(np.abs(x[np.isfinite(x)]))) if np.isfinite(x).any() else 0.0
            first = -1
            if key.startswith("trace/") and mx > 0:
                nz = np.argwhere(d > 0)
                first = int(nz[0][0]) if nz.size else -1
            field = key.split("/")[1].split("@")[0] if key.startswith("trace/") else key.split("/")[-1]
            sc, unit = SCALE.get(field, (1.0, ""))
            rows[key] = {"max_abs": mx, "max_abs_scaled": mx * sc, "unit": unit, "rel_to_max": mx / rng if rng
                         else (0.0 if mx == 0 else np.inf), "first_diff_sample": first, "n": int(d.size),
                         "n_diff": int((d > 0).sum())}
        out[f.stem] = rows
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--json", default="")
    ap.add_argument("--all", action="store_true", help="also print identical signals")
    a = ap.parse_args(argv)
    res = compare(Path(a.ref), Path(a.new), a.all)
    for sc, rows in res.items():
        nd = sum(1 for r in rows.values() if r.get("max_abs", 1) != 0)
        print(f"== {sc}: {nd} of {len(rows)} signals differ")
        for k, r in rows.items():
            if "shape" in r:
                print(f"   {k}: SHAPE {r['shape']}")
                continue
            if r["max_abs"] == 0 and not a.all:
                continue
            print(f"   {k:45s} max|d| {r['max_abs_scaled']:.3e} {r['unit']:3s} rel {r['rel_to_max']:.2e} "
                  f"first {r['first_diff_sample']} ({r['n_diff']}/{r['n']})")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
