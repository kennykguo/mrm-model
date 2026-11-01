"""Metrics report for TLC runs (docs/TLC_PORT_SPEC_AND_PLAN.md 2.4).

    .venv/bin/python tools/tlc_report.py runs/<id> [runs/<id> ...] [--out docs/tlc/results.md]

For each run directory: mrm's own per-ring metrics (metrics.json), the ECE496 per-ring statistics of the true lock
error from the recorded detuning trace (p99.9, peak, rms; pm and GHz; scored from experiment.score_from_s), the
reference thresholds (mrm's 5 pm budget and ECE496's 2 GHz), and a summary of the TLC telemetry log
(tlc_telemetry.npz, written by tools/tlc_run.py) when there is one. Writes Markdown and a JSON twin. World 0 only.

The report reads Truth (it is scoring, not control); nothing here may be imported by a controller.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

C_M_PER_S = 299_792_458.0
MRM_BUDGET_PM = 5.0           # metrics.lock_threshold_pm of the shipped experiments
ECE496_TIGHT_GHZ = 2.0        # ECE496 tight lock threshold (docs/TLC_PORT_SPEC_AND_PLAN.md 6, Q8)
MODE_TRACK = 2
ECE496_LAMBDA_M = 1310e-9    # the ECE496 plant's band (thermal_control/params.py: 86 pm/K at 1310 nm)
ECE496_REFERENCE = {          # published scenario -> requirement block holding the error statistics
    "steady": "R1", "d2_common": "R3"}
DEFAULT_ECE496_METRICS = Path(__file__).resolve().parents[2] / "ECE496_project/thermal_control/results/metrics.json"
TLC_CONTROLLERS = ("tlc", "tlc_pi")
DITHER_BLOCK_S = 4e-3         # one lock-in window: every tone of the plan is a whole number of cycles in it
MRM_COLUMNS = ("max_abs_lock_error_pm", "rms_lock_error_pm", "time_to_lock_s", "locked_final", "wrong_line_final",
               "excursion_count", "oma_penalty_db", "peak_slew_codes_per_s")


def _detuning_trace(f: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    """(t_s (S,), detuning_pm (S, B, N)) of the finest recorded mean detuning trace."""
    keys = [k for k in f["trace/main"] if k.startswith("detuning_m@") and k.endswith(":mean")]
    if not keys:
        raise ValueError("trace.h5 has no detuning_m@<dec>:mean record")
    key = min(keys, key=lambda k: int(k.split("@")[1].split(":")[0]))
    g = f["trace/main"][key]
    return np.asarray(g["t_s"]), np.asarray(g["value"]) * 1e12


def _telemetry_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    z = np.load(path)
    return {"core_steps": int(z["t_s"].size),
            "core_step_us_mean": float(np.mean(z["core_step_us"][..., 0])),
            "track_fraction": float(np.mean(z["state"] == MODE_TRACK)),
            "dither_amp_pm_max": float(np.max(np.abs(z["dither_amp_pm"]))) if z["t_s"].size else 0.0}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided 1 - alpha interval of a binomial rate k / n (spec 6, Q10)."""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def campaign_rates(m: dict) -> dict | None:
    """Rates over every world of a run with B > 1: per ring-trial and per world (any ring), with 95 % intervals."""
    locked = np.asarray(m.get("locked_final"), dtype=object)
    if locked.ndim != 2 or locked.shape[0] < 2:
        return None
    out = {"worlds": int(locked.shape[0]), "rings": int(locked.shape[1])}
    events = {"unlocked": locked.astype(object) == False}                               # noqa: E712
    if "wrong_line_final" in m:
        events["wrong_line"] = np.asarray(m["wrong_line_final"], dtype=object).astype(bool)
    for name, ev in events.items():
        ev = np.asarray(ev, dtype=bool)
        k_ring, n_ring = int(ev.sum()), int(ev.size)
        k_world, n_world = int(ev.any(axis=1).sum()), int(ev.shape[0])
        out[name] = {"ring_trials": [k_ring, n_ring, *clopper_pearson(k_ring, n_ring)],
                     "worlds_any_ring": [k_world, n_world, *clopper_pearson(k_world, n_world)]}
    return out


def _cell(values, ring: int):
    """World 0, ring ``ring`` of a (B, N) metric from metrics.json; None (not defined, e.g. never locked) stays None."""
    v = np.asarray(values, dtype=object)[0, ring]
    return None if v is None else (v.item() if hasattr(v, "item") else v)


def _ring_stats(err_pm: np.ndarray, pm_per_ghz: float) -> dict:
    a = np.abs(err_pm)
    peak, p999, rms = float(a.max()), float(np.quantile(a, 0.999)), float(np.sqrt(np.mean(err_pm ** 2)))
    return {"p999_err_pm": p999, "peak_err_pm": peak, "rms_err_pm": rms,
            "p999_err_ghz": p999 / pm_per_ghz, "peak_err_ghz": peak / pm_per_ghz, "rms_err_ghz": rms / pm_per_ghz,
            "within_5pm": bool(peak <= MRM_BUDGET_PM), "within_2ghz": bool(peak <= ECE496_TIGHT_GHZ * pm_per_ghz)}


def remove_tones(t: np.ndarray, x: np.ndarray, tones_hz, block_s: float = DITHER_BLOCK_S) -> np.ndarray:
    """x (S, N) minus its components at every tone and 2nd harmonic: per block, a least-squares fit of a line plus
    cos/sin at each frequency, of which only the tone part is subtracted (Q8: dither-removed tracking error)."""
    f = np.r_[np.asarray(tones_hz, float), 2 * np.asarray(tones_hz, float)]
    out = np.array(x, dtype=float)
    edges = np.arange(t[0], t[-1] + block_s, block_s)
    for lo in edges:
        sel = (t >= lo) & (t < lo + block_s)
        if sel.sum() < 2 * f.size + 4:
            continue
        tb = t[sel]
        w = 2 * np.pi * f[None, :] * tb[:, None]
        tone = np.hstack([np.cos(w), np.sin(w)])
        X = np.hstack([np.ones((tb.size, 1)), (tb - tb.mean())[:, None], tone])
        coef, *_ = np.linalg.lstsq(X, x[sel], rcond=None)
        out[sel] = x[sel] - tone @ coef[2:]
    return out


def _tlc_tones(controller: str | None, n_rings: int):
    """The TLC's tone plan for this run, or None for a controller whose tones the report does not know."""
    if controller not in TLC_CONTROLLERS:
        return None
    from mrm.control.tlc.tone_plan import EIGHT_RING_BINS, search, to_hz
    return to_hz(EIGHT_RING_BINS if n_rings == len(EIGHT_RING_BINS) else search(n_rings))


def summarize_run(run_dir: str | Path) -> dict:
    run = Path(run_dir)
    cfg = yaml.safe_load((run / "experiment.resolved.yaml").read_text())
    exp = cfg.get("experiment", {})
    score_from = float(exp.get("score_from_s") or 0.0)
    metrics = json.loads((run / "metrics.json").read_text())
    m = metrics["cases"]["main"]
    with h5py.File(run / "trace.h5", "r") as f:
        t, det_pm = _detuning_trace(f)
        lam_m = np.asarray(f["truth_final/main/lam_m"])[0]
    ref = np.asarray(m["lock_reference_pm"], dtype=float)[0]
    scored = t >= score_from
    err = det_pm[scored, 0, :] - ref[None, :]
    ppg = lam_m ** 2 / C_M_PER_S * 1e21
    controller = cfg.get("controller", {}).get("name")
    tones = _tlc_tones(controller, err.shape[1])
    track = remove_tones(t[scored], err, tones) if tones else None
    rings = []
    for i in range(err.shape[1]):
        row = {"ring": i, **_ring_stats(err[:, i], float(ppg[min(i, ppg.size - 1)]))}
        if track is not None:
            row.update(track_peak_err_pm=float(np.abs(track[:, i]).max()),
                       track_rms_err_pm=float(np.sqrt(np.mean(track[:, i] ** 2))))
        row.update({k: _cell(m[k], i) for k in MRM_COLUMNS if k in m})
        rings.append(row)
    return {"run": run.name, "experiment": metrics.get("name") or run.name,
            "controller": controller, "scenario": cfg.get("scenario", {}).get("name"),
            "score_from_s": score_from, "duration_s": exp.get("duration_s"), "rings": rings,
            "telemetry": _telemetry_summary(run / "tlc_telemetry.npz"), "campaign": campaign_rates(m)}


def ece496_reference(metrics_path: str | Path) -> list[dict]:
    """Per-ring error rows of the published ECE496 4-ring results (GHz, and pm at 1310 nm). The vendored core
    reproduces them (tests/tlc/test_core_fidelity.py), so they are the report's 4-ring reference column."""
    pub = json.loads(Path(metrics_path).read_text())
    ppg = ECE496_LAMBDA_M ** 2 / C_M_PER_S * 1e21
    rows = []
    for scenario, req in ECE496_REFERENCE.items():
        for kind, res in pub.get(scenario, {}).items():
            m = res[req]
            for i in range(len(m["peak"])):
                rows.append({"scenario": scenario, "req": req, "controller": kind, "ring": i,
                             **{f"{k}_err_ghz": m[k][i] for k in ("p999", "peak", "rms")},
                             **{f"{k}_err_pm": m[k][i] * ppg for k in ("p999", "peak", "rms")}})
    return rows


def _reference_markdown(rows: list[dict]) -> list[str]:
    cols = ["scenario", "req", "controller", "ring", "p999_err_ghz", "peak_err_ghz", "rms_err_ghz", "peak_err_pm"]
    out = ["## ECE496 4-ring reference (published; reproduced by the vendored core)", "",
           "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows]
    return out + [""]


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def overview_rows(summaries: list[dict]) -> list[dict]:
    """One row per run: the worst and typical ring numbers (ring 0 also left out: it is A1's heater-limited victim)."""
    rows = []
    for s in summaries:
        r = s["rings"]
        peak = [x["peak_err_pm"] for x in r]
        rows.append({"experiment": s["experiment"], "controller": s["controller"],
                     "peak_err_pm_max": max(peak), "peak_err_pm_max_rings_1_up": max(peak[1:]) if len(peak) > 1 else None,
                     "rms_err_pm_mean": sum(x["rms_err_pm"] for x in r) / len(r),
                     "track_peak_err_pm_max": max(x["track_peak_err_pm"] for x in r) if "track_peak_err_pm" in r[0]
                     else None,
                     "rings_within_5pm": sum(x["within_5pm"] for x in r),
                     "excursions": sum(int(x.get("excursion_count") or 0) for x in r),
                     "unlocked_final": sum(not x.get("locked_final", True) for x in r),
                     "wrong_line_final": sum(bool(x.get("wrong_line_final")) for x in r)})
    return rows


def _campaign_markdown(summaries: list[dict]) -> list[str]:
    rows = [s for s in summaries if s.get("campaign")]
    if not rows:
        return []
    out = ["## Monte Carlo rates (95 % Clopper-Pearson)", "",
           "| experiment | controller | worlds | event | ring trials k/n | interval | worlds with any k/n | interval |",
           "|---|---|---|---|---|---|---|---|"]
    for s in rows:
        c = s["campaign"]
        for ev in ("wrong_line", "unlocked"):
            if ev not in c:
                continue
            r, w = c[ev]["ring_trials"], c[ev]["worlds_any_ring"]
            out.append(f"| {s['experiment']} | {s['controller']} | {c['worlds']} | {ev} | {r[0]}/{r[1]} | "
                       f"[{r[2]:.3g}, {r[3]:.3g}] | {w[0]}/{w[1]} | [{w[2]:.3g}, {w[3]:.3g}] |")
    return out + [""]


def _overview_markdown(rows: list[dict]) -> list[str]:
    cols = ["experiment", "controller", "peak_err_pm_max", "peak_err_pm_max_rings_1_up", "rms_err_pm_mean",
            "track_peak_err_pm_max",
            "rings_within_5pm", "excursions", "unlocked_final", "wrong_line_final"]
    out = ["## Overview", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows]
    return out + [""]


def to_markdown(summaries: list[dict], reference: list[dict] | None = None) -> str:
    cols = ["ring", "p999_err_pm", "peak_err_pm", "rms_err_pm", "track_peak_err_pm", "track_rms_err_pm",
            "peak_err_ghz", "within_5pm", "within_2ghz",
            *MRM_COLUMNS]
    out = ["# TLC results", "",
           "Generated by `tools/tlc_report.py`. Errors are the true lock error (detuning minus mrm's lock reference), "
           "world 0, scored from `score_from_s`. The p99.9 / peak / rms columns come from the recorded 10 us-mean "
           "detuning trace; `track_*` is the same error with the TLC's dither tones removed (least squares per 4 ms "
           "block; TLC runs only). The mrm columns (`max_abs_lock_error_pm` onwards) are mrm's own per-tick metrics. "
           "Thresholds: mrm 5 pm budget, ECE496 2 GHz.", ""]
    out += _overview_markdown(overview_rows(summaries))
    out += _campaign_markdown(summaries)
    for s in summaries:
        out += [f"## {s['experiment']}: `{s['controller']}` on `{s['scenario']}`", "",
                f"Run `{s['run']}`, {s['duration_s']} s simulated, scored from {s['score_from_s']} s.", ""]
        tel = s["telemetry"]
        if tel:
            out += [f"Telemetry: {tel['core_steps']} core steps, {tel['core_step_us_mean']:.1f} us per core step per "
                    f"world, TRACK {100 * tel['track_fraction']:.0f} % of ring-steps, dither up to "
                    f"{tel['dither_amp_pm_max']:.3g} pm.", ""]
        present = [c for c in cols if c in s["rings"][0]]
        out += ["| " + " | ".join(present) + " |", "|" + "---|" * len(present)]
        out += ["| " + " | ".join(_fmt(r[c]) for c in present) + " |" for r in s["rings"]]
        out.append("")
    if reference:
        out += _reference_markdown(reference)
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--out", default="docs/tlc/results.md", help="Markdown output; a .json twin is written beside it")
    ap.add_argument("--ece496-metrics", default=str(DEFAULT_ECE496_METRICS),
                    help="published ECE496 results/metrics.json for the 4-ring reference column ('' to skip)")
    args = ap.parse_args(argv)
    summaries = [summarize_run(r) for r in args.runs]
    ref_path = Path(args.ece496_metrics) if args.ece496_metrics else None
    reference = ece496_reference(ref_path) if ref_path and ref_path.is_file() else None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(summaries, reference))
    out.with_suffix(".json").write_text(json.dumps({"runs": summaries, "ece496_reference": reference}, indent=2))
    print(f"wrote {out} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
