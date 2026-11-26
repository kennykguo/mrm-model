"""Extract compact plot data from mrm-model TLC runs into one JSON for the catch-up report (docs/tlc/catchup.html).

    .venv/bin/python docs/tlc/catchup_src/extract.py && .venv/bin/python docs/tlc/catchup_src/build.py

Needs the run directories listed in docs/tlc/results.md.runs.txt, runs/mismatch, runs/corner_mb, runs/campaign300
(local, not in git); data.json is written beside this file.
"""
import json
from pathlib import Path

import h5py
import numpy as np

import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
from tlc_report import _detuning_trace  # noqa: E402

OUT = Path(__file__).with_name("data.json")
N_POINTS = 360
TRACES = [  # (experiment, controller) pairs to plot as time series
    ("steady_tlc_faithful", "tlc"), ("steady_tlc_faithful", "kalman_lqg_c"),
    ("tlc_a1", "tlc"), ("tlc_a1", "mpc_c"),
    ("tlc_squelch", "tlc"), ("tlc_squelch", "pi_lock_c"),
    ("tlc_ext_kick", "tlc"), ("tlc_ext_kick", "kalman_lqg_c"),
    ("tlc_cold", "tlc"), ("tlc_cold_wdm", "wdm_supervisor"),
]


def r3(x):
    return None if x is None else float(f"{x:.3g}")


def ring_rows(run: dict) -> dict:
    rs = run["rings"]
    return {"rms": [r3(r["rms_err_pm"]) for r in rs], "peak": [r3(r["peak_err_pm"]) for r in rs],
            "wrong": [bool(r["wrong_line_final"]) for r in rs]}


def trace(run_dir: Path) -> dict:
    with h5py.File(run_dir / "trace.h5", "r") as f:
        t, det = _detuning_trace(f)
        det = det[:, 0, :]
    ref = np.asarray(json.loads((run_dir / "metrics.json").read_text())["cases"]["main"]["lock_reference_pm"], float)[0]
    err = det - ref[None, :]
    idx = np.linspace(0, t.size - 1, min(N_POINTS, t.size)).astype(int)
    return {"t_ms": [round(float(v) * 1e3, 3) for v in t[idx]],
            "err": [[round(float(v), 2) for v in err[idx, i]] for i in range(err.shape[1])]}


def main() -> None:
    res = json.loads((REPO / "docs/tlc/results.json").read_text())
    dirs = (REPO / "docs/tlc/results.md.runs.txt").read_text().split()
    assert len(dirs) == len(res["runs"]), "run index does not match results.json"
    matrix = [{"exp": r["experiment"], "ctrl": r["controller"], "corner": r["corner"], **ring_rows(r)}
              for r in res["runs"] if not r["campaign"]]
    traces = []
    for exp, ctrl in TRACES:
        i = next(k for k, r in enumerate(res["runs"]) if r["experiment"] == exp and r["controller"] == ctrl
                 and r["corner"] is None)
        traces.append({"exp": exp, "ctrl": ctrl, "score_ms": res["runs"][i]["score_from_s"] * 1e3, **ring_rows(res["runs"][i]),
                       **trace(Path(dirs[i]))})
    mism = json.loads((REPO / "runs/mismatch/results_mismatch.json").read_text())["runs"]
    cmb = json.loads((REPO / "runs/corner_mb/results_corner_mb.json").read_text())["runs"]
    corners = ([{"exp": r["experiment"].split("@")[0], "ctrl": r["controller"], "corner": r["corner"], "know": "corner",
                 **ring_rows(r)} for r in res["runs"] + cmb if r["corner"]]
               + [{"exp": r["experiment"].split("@")[0], "ctrl": r["controller"], "corner": r["corner"],
                   "know": "nominal", **ring_rows(r)} for r in mism])
    camp = json.loads((REPO / "runs/campaign300/20261110T025137Z_ecf03629/metrics.json").read_text())["cases"]["main"]
    wrong = np.asarray(camp["wrong_line_final"], bool)
    ttl = np.asarray(camp["time_to_lock_s"], float)
    campaign300 = {"wrong_per_ring": wrong.sum(0).tolist(), "worlds": int(wrong.shape[0]),
                   "ttl_ms": [round(float(v) * 1e3, 2) for v in ttl[np.isfinite(ttl)].ravel()]}
    diag = json.loads((REPO / "docs/tlc/lockin_diag.json").read_text())["rows"]
    lockin = [{k: r3(r[k]) for k in ("detuning_pm", "z1_mean", "z1_std", "z1_model", "zeta_mean", "zeta_std",
                                      "zeta_model")} for r in diag]
    ref = [{"sc": r["scenario"], "ctrl": r["controller"], "ring": r["ring"], "rms": r3(r["rms_err_pm"]),
            "peak": r3(r["peak_err_pm"])} for r in res["ece496_reference"]]
    dc = json.loads((REPO / "docs/tlc/dither_compare.json").read_text())
    dither = [{"exp": r["experiment"], "arm": r["arm"], **ring_rows(r),
               "camp": (r.get("campaign") or {}).get("wrong_line", {}).get("ring_trials")} for r in dc["runs"]]
    OUT.write_text(json.dumps({"matrix": matrix, "traces": traces, "corners": corners, "campaign300": campaign300,
                               "lockin": lockin, "ece496_ref": ref, "dither": dither}, separators=(",", ":")))
    print(OUT, OUT.stat().st_size // 1024, "KiB")


if __name__ == "__main__":
    main()
