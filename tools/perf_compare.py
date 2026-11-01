"""Compare two runs of tools/perf_scenarios.py signal by signal.

    .venv/bin/python tools/perf_compare.py runs/perf_baseline/ref runs/perf_x [--json out.json] [--all]

Read this first. Each run directory holds one ``<scenario>.npz`` per scenario. For every recorded trace (key
``trace/<field>@<dec>:<reducer>/value``), every summary metric (``metric/...``) and every final Truth value
(``final/...``) of a scenario in both directories it prints:

- the largest absolute difference, in the display unit of the field (see SCALE);
- that difference relative to the largest magnitude of the signal in the reference run;
- for a trace, the first sample index where the two runs differ, and how many samples differ.

A scenario file or a signal that is in only one of the two runs is printed as MISSING and counts as a difference
(a crashed scenario or a dropped trace). The exit code is then 1, otherwise 0.

A refactor that must not change the engine shows "0 of N signals differ" for every scenario. Display units:
detuning and resonance in pm, temperatures in mK, powers in uW, currents in nA, codes as codes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

#: field name -> (factor from SI to the display unit, display unit). Fields not listed print in SI with no unit.
SCALE = {"detuning_m": (1e12, "pm"), "lam_res_m": (1e12, "pm"), "t_ring_k": (1e3, "mK"), "t_heater_k": (1e3, "mK"),
         "t_pd_k": (1e3, "mK"), "p_abs_w": (1e6, "uW"), "p_heater_w": (1e6, "uW"), "p_drop_w": (1e6, "uW"),
         "i_monitor_a": (1e9, "nA"), "p_in_w": (1e6, "uW"), "oma_w": (1e6, "uW")}

#: row key of a scenario whose .npz file is in only one of the two run directories
SCENARIO_FILE = "(scenario file)"
#: value of a row's "missing" -> the run it is missing from, as printed
MISSING_FROM = {"new": "new run", "ref": "reference run"}


def compare(ref_dir: Path, new_dir: Path, show_all=False):
    """{scenario: {key: row}} for every scenario file in either directory. A row holds max_abs (SI), max_abs_scaled
    (display unit), unit, rel_to_max, first_diff_sample (-1 when none or not a trace), n and n_diff; a row with
    different array shapes holds only ``shape``. A key in only one run gives the row {"missing": "new" | "ref"} (the
    run that lacks it), and a scenario file in only one directory gives the single row SCENARIO_FILE: {"missing":
    ...}. NaN against NaN and the same infinity count as equal. ``show_all`` is not used here (the printing in main
    uses it)."""
    out = {}
    ref_names = {f.name for f in ref_dir.glob("*.npz")}
    new_names = {f.name for f in new_dir.glob("*.npz")}
    for name in sorted(ref_names | new_names):
        scenario = Path(name).stem
        if name not in new_names or name not in ref_names:
            out[scenario] = {SCENARIO_FILE: {"missing": "new" if name not in new_names else "ref"}}
            continue
        ref_npz = np.load(ref_dir / name)
        new_npz = np.load(new_dir / name)
        ref_keys, new_keys = _compared_keys(ref_npz), _compared_keys(new_npz)
        rows = {}
        for key in ref_keys:
            if key not in new_keys:
                rows[key] = {"missing": "new"}
                continue
            x = np.asarray(ref_npz[key], float)
            y = np.asarray(new_npz[key], float)
            if x.shape != y.shape:
                rows[key] = {"shape": [list(x.shape), list(y.shape)]}
                continue
            both_finite = np.isfinite(x) & np.isfinite(y)
            same_nonfinite = (np.isnan(x) & np.isnan(y)) | (np.isinf(x) & np.isinf(y) & (np.sign(x) == np.sign(y)))
            with np.errstate(invalid="ignore"):
                diff = np.where(both_finite, np.abs(x - y), np.where(same_nonfinite, 0.0, np.inf))
            max_diff = float(diff.max()) if diff.size else 0.0
            ref_max = float(np.nanmax(np.abs(x[np.isfinite(x)]))) if np.isfinite(x).any() else 0.0
            first = -1
            if key.startswith("trace/") and max_diff > 0:
                nonzero = np.argwhere(diff > 0)
                first = int(nonzero[0][0]) if nonzero.size else -1
            field = key.split("/")[1].split("@")[0] if key.startswith("trace/") else key.split("/")[-1]
            scale, unit = SCALE.get(field, (1.0, ""))
            rows[key] = {"max_abs": max_diff, "max_abs_scaled": max_diff * scale, "unit": unit,
                         "rel_to_max": max_diff / ref_max if ref_max else (0.0 if max_diff == 0 else np.inf),
                         "first_diff_sample": first, "n": int(diff.size), "n_diff": int((diff > 0).sum())}
        for key in new_keys:
            if key not in ref_keys:
                rows[key] = {"missing": "ref"}
        out[scenario] = rows
    return out


def _compared_keys(npz):
    """The keys of a run file that are compared: the trace values (not their time axes), the metrics and the final
    Truth."""
    return [key for key in npz.files if key.endswith("/value") or key.startswith(("metric/", "final/"))]


def main(argv=None):
    """Print the comparison. Returns the exit code: 1 when a scenario or a signal is missing from one run, else 0."""
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("new")
    ap.add_argument("--json", default="")
    ap.add_argument("--all", action="store_true", help="also print identical signals")
    args = ap.parse_args(argv)
    res = compare(Path(args.ref), Path(args.new), args.all)
    n_missing = 0
    for scenario, rows in res.items():
        if SCENARIO_FILE in rows:
            n_missing += 1
            print(f"== {scenario}: MISSING in the {MISSING_FROM[rows[SCENARIO_FILE]['missing']]}")
            continue
        n_differ = sum(1 for r in rows.values() if r.get("max_abs", 1) != 0)    # a shape mismatch or MISSING too
        print(f"== {scenario}: {n_differ} of {len(rows)} signals differ")
        for key, r in rows.items():
            if "missing" in r:
                n_missing += 1
                print(f"   {key}: MISSING in the {MISSING_FROM[r['missing']]}")
                continue
            if "shape" in r:
                print(f"   {key}: SHAPE {r['shape']}")
                continue
            if r["max_abs"] == 0 and not args.all:
                continue
            print(f"   {key:45s} max|d| {r['max_abs_scaled']:.3e} {r['unit']:3s} rel {r['rel_to_max']:.2e} "
                  f"first {r['first_diff_sample']} ({r['n_diff']}/{r['n']})")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, default=float))
    return 1 if n_missing else 0


if __name__ == "__main__":
    sys.exit(main())
