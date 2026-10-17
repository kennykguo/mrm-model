"""Every registered controller closed loop on the real engine (mrm.sim.runner.run), g26 nominal world, two scenarios:
steady lock (static, 25 degC) and A1 (the slide-10 on-die aggressor, one full period). The results go to
docs/handoff/controller_scorecard.md (results only, with the conditions).

Slow: 2-4 s (steady, 40 ms) and 7-33 s (A1, 130.7 ms) of wall time per controller on the shared M2 at load 3-5
(B = 1, fixed 1 us ticks; the Python path stops the kernel at every 10 us controller edge); ~8 min for the file.
Regenerate the scorecard with

    .venv/bin/python tools/memguard.py --heavy -- .venv/bin/python -m pytest -m slow -s \
        tests/validation/test_controllers_on_engine.py

Conditions (the same for every controller; they are written into the scorecard):

- g26 preset, nominal world (uncertainty nominal, variation off), seeds 0, B = 1, N = L = 8, dt 1 us, substep policy
  fixed, controller at the preset rate (100 kHz). Controller knowledge: every token any baseline reads
  (KNOWLEDGE below; g26 itself releases only line_grid and ring_to_line). Bias owners (bias_dither, midranging):
  ``controller_interface.bind.tx.bias=controller``. wdm_supervisor: ``bind.tx.assignment=controller``.
- Start. ``warm``: heaters at the quasi-static oracle lock codes of each ring (mrm.sim.runner.warm_start_codes) with
  the thermal state settled (Plant.settle, reset protocol branch_high): a steady lock, not a cold-lag start.
  ``cold``: heaters at code 0, lasers on at t = 0, thermal state settled at code 0 from the cold branch (branch_low).
  Each controller runs its own start-up from there (map, capture, ramp, probe, or lock at once).
- Lock error e = d - d_ref, with d the engine Truth detuning of the ring's resonance (order nearest its DECLARED
  line) from the 10-tick mean / min / max records. Steady: d_ref = the ring's own final lock point (median of the
  last 5 ms); a ring counts as locked when |e| <= 5 pm from some time on (time to lock = that time) AND its
  operating point is stationary: the drift (median d of the last 10 ms minus that of the 10 ms before, per ms) is
  at most DRIFT_PM_PER_MS = 0.1 pm/ms, i.e. the point moves by less than the 1 pm hysteresis over the 10 ms
  scoring window (controllers review 2026-10-16: es_sine drifted at ~0.8 pm/ms toward the drop peak and passed as a
  lock within 40 ms); jitter rms and max |e| over the last 10 ms. A1: d_ref = the same ring's steady lock point; scored from the bank's steady time to
  lock + 1 ms (10 ms when the steady run never locked) to the end; excursions = exits beyond 5 pm with 1 pm
  hysteresis (min/max envelope, so a 1-tick spike counts). OMA penalty = 10 log10(mean oracle relOMA / mean relOMA)
  over the scored window (engine oracle: best stable relOMA point, refreshed every 1 ms; 07 P1-1). Lines: the line
  each ring ends on (engine line_on, -1 = none) against the declared line.
- A1: mrm.scenarios.library.a1_slide10_aggressor with phase_s = P/2, so t = 0 is the rounded minimum (aggressor
  near zero power, the warm start is honest) and the hot cusp is at t = 65.35 ms; duration one period (130.7 ms).
  The aggressor power is calibrated by the engine (resonance referencing at ring 0). Two runs: ``a1``, the g26
  default site ``aggressor`` (0, -30) um, 55 um from ring 0's monitor PD; and ``a1_pd_adjacent``, the STRESS
  scenario a1_slide10_aggressor_pd_adjacent at the site ``aggressor_pd_adjacent`` (0, 30) um, 5 um from that PD
  (the former default: the PD dark current runs away, controllers_a/b/c handoffs; not a realistic layout). Both
  sites are 30 um from ring 0 and the same distance from every ring, so the calibrated power and the rings' heat
  are identical; only the monitor-PD paths differ (the die sensor sees both sites alike in the interim model).

The asserts check that every controller ran closed loop on the engine (it moved the heater, the engine counted no
pipeline overflow, the metrics are finite); lock quality is recorded, not asserted.
"""
from __future__ import annotations

import json
import math
import os
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pytest

runner = pytest.importorskip("mrm.sim.runner")

from mrm import params                                    # noqa: E402
from mrm.control import registry                          # noqa: E402
from mrm.scenarios import library as lib                  # noqa: E402
from mrm.sim.metrics import RecordSpec, envelope          # noqa: E402

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("MRM_SCORECARD_OUT", ROOT / "docs" / "handoff" / "controller_scorecard.md"))
JSON_OUT = OUT.with_suffix(".json")

KNOWLEDGE = ("line_grid", "ring_to_line", "heater_nominal", "optics_nominal", "dlambda_dt_nominal", "monitor_nominal",
             "heater_law", "heater_r_nominal", "heater_efficiency", "linewidth", "thermal_dynamics",
             "bias_efficiency", "model_nominal")
BIAS = ("controller_interface.bind.tx.bias=controller", "scenario.bias_v=null")
ASSIGN = ("controller_interface.bind.tx.assignment=controller",)
LOCK_PM = 5.0
HYST_PM = 1.0
DEC = 10                                     # record decimation, ticks (= the g26 controller period)
STEADY_S = float(os.environ.get("MRM_SCORECARD_STEADY_S", 40e-3))
A1_S = float(os.environ.get("MRM_SCORECARD_A1_S", lib.A1_PERIOD_S))
SCORE_LAST_S = 10e-3
DRIFT_PM_PER_MS = 0.1                        # stationary operating point: < the 1 pm hysteresis per 10 ms window

# name -> (controller params, start, extra overrides, note)
CASES: dict[str, tuple[dict, str, tuple, str]] = {
    "pi_lock": ({}, "warm", (), "defaults: MAP from the hand-off code, PI at 0.7 of the peak (red side)"),
    "pi_lock_c": ({}, "warm", (), "compiled PiLock (in-kernel), same parameters"),
    "dither_lockin": ({}, "warm", (), "defaults: heater dither, inflection mode"),
    "dither_lockin_c": ({}, "warm", (), "compiled DitherLockin"),
    "bias_dither": ({}, "warm", BIAS, "defaults (bias 1.8 V); bias owned by the controller"),
    "es_po": ({}, "warm", (), "defaults: perturb-and-observe extremum seeking (peak)"),
    "es_sine": ({}, "warm", (), "defaults: sine-dither extremum seeking (peak)"),
    "feedforward_pi": ({}, "warm", (), "defaults: PiLock + die-sensor feed-forward from nominal knowledge"),
    "midranging": ({}, "warm", BIAS, "defaults: bias inner loop, heater re-centring"),
    "ramp_acquire": ({}, "cold", (), "defaults: every ring ramps up from w = 0 and locks on the first line found"),
    "acquire_then_lock": ({}, "cold", (), "defaults: ramp acquisition, then pi_lock"),
    "bitstat_lock": ({}, "cold", (), "defaults: ramp acquisition, then the bit-statistics ratio lock"),
    "wdm_supervisor": ({}, "cold", ASSIGN, "defaults (policy cyclic): map, assign, acquire, track; tx.assignment"),
    "kalman_lqg": ({}, "warm", (), "defaults: start warm (lock at once), target oma_opt from the nominal model"),
    "kalman_lqg_c": ({}, "warm", (), "compiled LQG (lock only)"),
    "mpc": ({}, "warm", (), "defaults: start warm, horizon 20, 4 moves"),
    "mpc_c": ({}, "warm", (), "compiled MPC (lock only)"),
    "composite": ({"children": [{"name": "pi_lock", "rate_hz": 100e3}]}, "warm", (),
                  "CompositeController with one child, pi_lock at 100 kHz (exercises the composite path)"),
}
_RESULTS: dict[str, dict] = {}


def _cfg(extra=()):
    ov = ["controller_interface.knowledge=[" + ",".join(KNOWLEDGE) + "]", *extra]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load("g26", overrides=ov)


RECORD = [RecordSpec("detuning_m", DEC, "mean"), *envelope("detuning_m", DEC), RecordSpec("reloma", DEC, "mean"),
          RecordSpec("oracle_reloma", DEC, "mean"), RecordSpec("line_on", DEC, "last"),
          RecordSpec("declared_line", DEC, "last"), RecordSpec("heater_code", DEC, "last"),
          RecordSpec("t_pd_k", 100 * DEC, "max")]


def _run(name: str, scenario, duration_s: float, extra=()):
    prm, start, ov, _ = CASES[name]
    cfg = _cfg((*ov, *extra))
    ctrl = registry.create(name, prm)
    kw = dict(settle=True)
    if start == "warm":
        kw.update(heater_code="quasi_static", reset_protocol="branch_high")
    else:
        kw.update(reset_protocol="branch_low")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = runner.run(cfg, controller=ctrl, scenario=scenario, duration_s=duration_s, batch=1,
                         uncertainty="nominal", variation="off", seeds=0, record=RECORD, **kw)
    return res, ctrl


def _tr(res, key):
    return np.asarray(res.trace[key]["value"])[:, 0, :]


def _first_lock_index(e_abs: np.ndarray) -> np.ndarray:
    """(N,) first sample index from which |e| stays <= LOCK_PM to the end (-1 if never)."""
    S, N = e_abs.shape
    out = np.full(N, -1)
    for n in range(N):
        bad = np.flatnonzero(e_abs[:, n] > LOCK_PM)
        if bad.size == 0:
            out[n] = 0
        elif bad[-1] < S - 1:
            out[n] = bad[-1] + 1
    return out


def _excursions(e_abs: np.ndarray) -> np.ndarray:
    """(N,) exits beyond LOCK_PM after being within LOCK_PM - HYST_PM (07 P0-8 hysteresis), from the envelope."""
    S, N = e_abs.shape
    cnt = np.zeros(N, np.int64)
    for n in range(N):
        locked = e_abs[0, n] <= LOCK_PM
        for s in range(1, S):
            if locked and e_abs[s, n] > LOCK_PM:
                cnt[n] += 1
                locked = False
            elif not locked and e_abs[s, n] <= LOCK_PM - HYST_PM:
                locked = True
    return cnt


def _penalty_db(res, sl) -> np.ndarray:
    rel = _tr(res, f"reloma@{DEC}:mean")[sl]
    orc = _tr(res, f"oracle_reloma@{DEC}:mean")[sl]
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(orc.mean(axis=0) / rel.mean(axis=0))


def _score_steady(res) -> dict:
    t = np.asarray(res.trace[f"detuning_m@{DEC}:mean"]["t_s"])
    d = _tr(res, f"detuning_m@{DEC}:mean") * 1e12
    lo = _tr(res, f"detuning_m@{DEC}:min") * 1e12
    hi = _tr(res, f"detuning_m@{DEC}:max") * 1e12
    last5 = t >= t[-1] - 5e-3
    ref = np.median(d[last5], axis=0)
    e_abs = np.maximum(np.abs(lo - ref), np.abs(hi - ref))
    i0 = _first_lock_index(e_abs)
    win = t >= t[-1] - SCORE_LAST_S
    e = d[win] - ref
    t_lock = np.where(i0 >= 0, t[np.maximum(i0, 0)], np.nan)
    line = _tr(res, f"line_on@{DEC}:last")[-1].astype(int)
    decl = _tr(res, f"declared_line@{DEC}:last")[-1].astype(int)
    # drift of the operating point: median of the last SCORE_LAST_S against the window before it (shorter windows
    # for a run shorter than two of them), pm per ms
    w = min(SCORE_LAST_S, 0.5 * float(t[-1] - t[0]))
    a_win = t >= t[-1] - w
    b_win = (t >= t[-1] - 2.0 * w) & ~a_win
    drift = (np.median(d[a_win], axis=0) - np.median(d[b_win], axis=0)) / (w * 1e3)
    within = i0 >= 0
    stationary = np.abs(drift) <= DRIFT_PM_PER_MS
    return {"ref_pm": ref, "locked": within & stationary, "within_5pm": within, "stationary": stationary,
            "drift_pm_per_ms": drift, "t_lock_s": t_lock, "rms_pm": np.sqrt((e ** 2).mean(axis=0)),
            "max_pm": e_abs[win].max(axis=0), "exc": _excursions(e_abs[win]), "penalty_db": _penalty_db(res, win),
            "line": line, "declared": decl, "t_pd_max_c": _tr(res, f"t_pd_k@{100 * DEC}:max").max(axis=0) - 273.15}


def _score_a1(res, ref, t0) -> dict:
    t = np.asarray(res.trace[f"detuning_m@{DEC}:mean"]["t_s"])
    lo = _tr(res, f"detuning_m@{DEC}:min") * 1e12
    hi = _tr(res, f"detuning_m@{DEC}:max") * 1e12
    d = _tr(res, f"detuning_m@{DEC}:mean") * 1e12
    win = t >= min(t0, t[-1])                 # a run shorter than t0 (smoke runs) scores its last sample
    e_abs = np.maximum(np.abs(lo - ref), np.abs(hi - ref))[win]
    e = (d - ref)[win]
    line = _tr(res, f"line_on@{DEC}:last")[-1].astype(int)
    return {"max_pm": e_abs.max(axis=0), "rms_pm": np.sqrt((e ** 2).mean(axis=0)), "exc": _excursions(e_abs),
            "locked_end": e_abs[-max(1, int(1e-3 / (DEC * 1e-6))):].max(axis=0) <= LOCK_PM,
            "penalty_db": _penalty_db(res, win), "line": line,
            "t_pd_max_c": _tr(res, f"t_pd_k@{100 * DEC}:max").max(axis=0) - 273.15,
            "low_rail_ms": np.asarray(res.metrics["heater_low_rail_s"])[0] * 1e3,
            "full_scale_ms": np.asarray(res.metrics["heater_sat_s"])[0] * 1e3}


def _closed_loop_checks(res, name):
    m = res.metrics
    rng = np.asarray(m["heater_range_used_codes"])
    assert np.any(rng > 0), f"{name}: the heater never moved (not closed loop?)"
    assert int(np.sum(res.diagnostics["pipeline_overflow"])) == 0, name
    assert np.all(np.isfinite(np.asarray(m["mean_detuning_pm"]))), name


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (np.ndarray, list, tuple)):
        return [_jsonable(v) for v in np.asarray(x).tolist()] if not isinstance(x, (list, tuple)) else \
            [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


@pytest.mark.parametrize("name", list(CASES))
def test_controller_runs_closed_loop_on_the_engine(name):
    assert name in registry.available(), f"{name} is not registered"
    out = {"name": name, "start": CASES[name][1], "note": CASES[name][3], "errors": []}
    try:
        t_w = time.perf_counter()
        res, ctrl = _run(name, lib.static(), STEADY_S)
        _closed_loop_checks(res, name)
        st = _score_steady(res)
        out["steady"] = st
        out["steady_wall_s"] = time.perf_counter() - t_w
        out["path"] = res.manifest_inputs["controller_path"]
        out["engine_oma_penalty_db_run"] = np.asarray(res.metrics["oma_penalty_db"])[0]
        out["settle"] = res.manifest_inputs["warm_start"]["settle"]
        # the A1 scoring start: the steady time to lock (|e| <= 5 pm from then on), stationary or not
        lk = st["t_lock_s"][st["within_5pm"]]
        t0 = (float(np.max(lk)) + 1e-3) if st["within_5pm"].any() else 10e-3
        out["a1_t0_s"] = t0
        for key, a1 in (("a1", lib.a1_slide10_aggressor(phase_s=lib.A1_PERIOD_S / 2)),
                        ("a1_pd_adjacent", lib.a1_slide10_aggressor_pd_adjacent(phase_s=lib.A1_PERIOD_S / 2))):
            t_w = time.perf_counter()
            r1, _ = _run(name, a1, A1_S)
            _closed_loop_checks(r1, name)
            out[key] = _score_a1(r1, st["ref_pm"], t0)
            out[key + "_wall_s"] = time.perf_counter() - t_w
    except Exception as exc:                                                   # noqa: BLE001
        out["errors"].append("".join(traceback.format_exception_only(type(exc), exc)).strip())
        out["traceback"] = traceback.format_exc()
        _RESULTS[name] = out
        raise
    _RESULTS[name] = out
    print(f"\n[scorecard] {name}: " + json.dumps(_jsonable({k: out[k] for k in ("steady", "a1")}))[:800])


def _fmt(x, f="{:.1f}"):
    if x is None:
        return "-"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "nan" if not math.isfinite(xf) else f.format(xf)


def _write_scorecard(results: dict) -> str:
    lines = ["# Controller scorecard: every registered controller on the engine (g26 nominal)", "",
             "Generated by `tests/validation/test_controllers_on_engine.py` (slow) on "
             f"{time.strftime('%Y-%m-%d %H:%M')}. Results only; the conditions are in that file's docstring and "
             "summarized here. Nothing below is a pass/fail judgement of a controller.", "",
             "## Conditions", "",
             "- g26 preset, nominal world (uncertainty nominal, variation off), seeds 0, B = 1, N = L = 8, dt = 1 us, "
             "substep policy fixed, controller edges every 10 us (100 kHz), default controller parameters.",
             "- Knowledge: every token a baseline reads (" + ", ".join(KNOWLEDGE) + "); g26 alone releases only "
             "line_grid and ring_to_line.",
             "- Start `warm`: heaters at the quasi-static oracle lock codes (mrm.sim.runner.warm_start_codes), thermal "
             "state settled (Plant.settle, branch_high). Start `cold`: heaters at code 0, lasers on at t = 0, "
             "thermal state settled at code 0 (branch_low). Each controller then runs its own start-up.",
             f"- Steady: static scenario, 25 degC, {STEADY_S * 1e3:.0f} ms. Lock point d_ref = median detuning of "
             "the last 5 ms (to the order nearest the declared line); locked = |e| <= 5 pm from some time on AND a "
             f"stationary operating point (drift = median d of the last 10 ms minus that of the 10 ms before, per ms, "
             f"at most {DRIFT_PM_PER_MS} pm/ms: less than the 1 pm hysteresis over the scoring window); a ring "
             "within 5 pm whose point still drifts is counted under 'drifting' and not as locked (its d_ref is a "
             "snapshot of a transient); jitter rms / max |e| / excursions over the last 10 ms; time to lock from "
             "t = 0 (includes the start-up; the A1 runs are scored from the latest time to lock + 1 ms, drifting "
             "rings included).",
             f"- A1: slide-10 aggressor (rectified sine, 113.6 - 88.5 |sin| degC at ring 0, resonance referencing), "
             f"phase P/2 (t = 0 at the minimum, hot cusp at 65.35 ms), {A1_S * 1e3:.1f} ms. Lock error about the "
             "steady d_ref, scored from the steady time to lock + 1 ms. `a1` = the g26 default site `aggressor` "
             "(0, -30) um, 55 um from ring 0's monitor PD; `a1_pd_adjacent` = the STRESS scenario "
             "a1_slide10_aggressor_pd_adjacent at the site `aggressor_pd_adjacent` (0, 30) um, 5 um from that PD "
             "(the default site until 2026-10-10; not a realistic layout). Both sites are 30 um from ring 0 and the "
             "same distance from every ring: the calibrated power and the rings' heat are the same, only the "
             "monitor-PD paths differ.",
             "- Excursions: exits beyond 5 pm with 1 pm hysteresis on the 10 us min/max envelope. OMA penalty: "
             "10 log10(mean oracle relOMA / mean relOMA) over the scored window (positive = OMA lost against the "
             "engine's oracle). Lines: rings whose final line (engine line_on) is not the declared line.",
             "", "## Steady lock", "",
             "| controller | path | start | locked rings | drifting rings | drift, max abs (pm/ms) | time to lock, max "
             "(ms) | jitter rms, median / max (pm) | max abs e (pm) | excursions | OMA penalty, mean (dB) | d_ref "
             "range (pm) | off declared line |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        if "steady" not in r:
            lines.append(f"| {name} | - | {r.get('start')} | ERROR: {r['errors'][0][:120] if r['errors'] else ''} "
                         "| | | | | | | | | |")
            continue
        s = r["steady"]
        lk = np.asarray(s["locked"], bool)
        drifting = np.asarray(s["within_5pm"], bool) & ~np.asarray(s["stationary"], bool)
        tl = np.asarray(s["t_lock_s"], float)
        off = int(np.sum(np.asarray(s["line"]) != np.asarray(s["declared"])))
        lines.append(
            f"| {name} | {r.get('path')} | {r['start']} | {int(lk.sum())}/8 | {int(drifting.sum())} | "
            f"{_fmt(np.max(np.abs(s['drift_pm_per_ms'])), '{:.3f}')} | "
            f"{_fmt(np.nanmax(tl[lk]) * 1e3 if lk.any() else None, '{:.2f}')} | "
            f"{_fmt(np.median(s['rms_pm']), '{:.2f}')} / {_fmt(np.max(s['rms_pm']), '{:.2f}')} | "
            f"{_fmt(np.max(s['max_pm']), '{:.1f}')} | {int(np.sum(s['exc']))} | "
            f"{_fmt(np.nanmean(s['penalty_db']), '{:.2f}')} | "
            f"{_fmt(np.min(s['ref_pm']), '{:.0f}')} to {_fmt(np.max(s['ref_pm']), '{:.0f}')} | {off} |")
    for key, title in (("a1", "A1 aggressor, default layout (site 55 um from ring 0's monitor PD)"),
                       ("a1_pd_adjacent", "A1 STRESS scenario a1_slide10_aggressor_pd_adjacent (site 5 um from ring "
                                          "0's monitor PD; not a realistic layout)")):
        lines += ["", f"## {title}", "",
                  "| controller | max abs e ring 0 (pm) | max abs e rings 1-7 (pm) | rms e ring 0 (pm) | excursions "
                  "ring 0 / others | locked at the end | OMA penalty, mean (dB) | off declared line | PD0 max "
                  "(degC) | ring 0 heater at code 0 / full scale (ms) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in results.items():
            if key not in r:
                lines.append(f"| {name} | ERROR or not run | | | | | | | | |")
                continue
            a = r[key]
            mx = np.asarray(a["max_pm"], float)
            ex = np.asarray(a["exc"], int)
            le = np.asarray(a["locked_end"], bool)
            off = int(np.sum(np.asarray(a["line"]) != np.asarray(r["steady"]["declared"])))
            lines.append(f"| {name} | {_fmt(mx[0])} | {_fmt(np.max(mx[1:]))} | {_fmt(a['rms_pm'][0], '{:.2f}')} | "
                         f"{int(ex[0])} / {int(ex[1:].sum())} | {int(le.sum())}/8 | "
                         f"{_fmt(np.nanmean(a['penalty_db']), '{:.2f}')} | {off} | "
                         f"{_fmt(a['t_pd_max_c'][0], '{:.0f}')} | {_fmt(a['low_rail_ms'][0])} / "
                         f"{_fmt(a['full_scale_ms'][0])} |")
    lines += ["", "## Reading notes (true for every run above)", "",
              f"- A steady run of {STEADY_S * 1e3:.0f} ms shows a controller's operating point, not where it ends: a "
              "ring whose point still drifts (column 'drifting') is not counted as locked, and its d_ref, jitter and "
              "OMA penalty describe a transient. es_sine (default parameters) is the case: it drifts toward the drop "
              "peak at 0.77-0.95 pm/ms, without a single fall off the fold, and parks at the peak (-0.6 to -1.1 pm) "
              "after 100-160 ms, where its OMA penalty is 12.9-13.0 dB on every ring (0.4 s run, "
              "runs/controllers_fix/es_long.log, runs/params_fix/es_long.log); es_po falls off the fold once on "
              "rings 4-7 before 40 ms (rings 0-3 never) and every ring parks at the drop peak (-0.3 to -1.4 pm, "
              "13.0-13.2 dB) by 40 ms (0.4 s run, runs/params_fix/es_long.log; before the params review of "
              "2026-10-16 it fell 2-4 times per ring and rings 4-6 parked on the cold branch). The drop peak is the "
              "wrong point for a modulator (04 C5). A ring that locks late is also counted as drifting while its "
              "point still settles (in the scorecard before the params review: wdm_supervisor ring 2, locked at "
              "26.7 ms, -0.24 pm/ms by a least-squares fit over the last 10 ms, and bitstat_lock ring 5, locked at "
              "23.1 ms; runs/controllers_fix/drift_defs.log).",
              "- The detuning envelope includes any deliberate modulation: heater or bias dithers (dither_lockin, "
              "bias_dither, midranging, es_sine, es_po probes) show as jitter and as 5 pm excursions when their "
              "resonance swing is that large.",
              "- The acquisition controllers without a supervisor (ramp_acquire, acquire_then_lock, bitstat_lock) lock "
              "each ring on the first line its ramp meets, not on its declared line (g26: the cold resonance sits "
              "~2 FWHM blue of line n+3). Their lock error is about their own lock point, but their OMA penalty is "
              "against the oracle of the DECLARED line, so it is large by construction.",
              "- The lock error is about each ring's own steady operating point, so a side-of-fringe lock and an "
              "OMA-optimal lock score alike; the OMA penalty column shows what the operating point costs.",
              "- A1 is calibrated by resonance referencing (ring 0's resonance follows the printed 88.5 K swing at "
              "85.86 pm/K). Near the hot cusp ring 0 then needs less than zero heater power: in the default-layout "
              "runs its heater sits at code 0 for several ms (column 'at code 0'), and the error there is set by "
              "heater authority, not by the controller (the replicas' CAL-9 finding, docs/handoff/replicas.md).",
              "- In the stress scenario the monitor PD of ring 0 passes 600 degC near the cusp; its Arrhenius dark "
              "current reads as light on the monitor, so monitor-based locks are dragged red and then lost; a ring "
              "at full-scale heater then disturbs its neighbours. The default site (55 um from the PD) removes that "
              "path. The die temperature sensor (feedforward_pi) sees the same aggressor in both runs: it sits "
              "~1 mm away, where the interim site law is at its long-range floor (1.5 % of Z_heater->ring) for "
              "both sites, so the aggressor alone moves its reading by ~20 K over A1 (open loop) in both.",
              "- Until 2026-10-10 the g26 default site was (0, 30) um (the stress table, unchanged) and this file "
              "ran an `a1_pd_far` case (`ring.thermal.pd_offset_um=[0,-25]`). `a1` now has the same PD-to-site "
              "distances, but that override also put every PD 5 um from its own ring's driver site (driver_offset_um "
              "(0, -30) um, ~4.5 mW of data heat), so the old PD-far PDs ran 8-10 K warmer. The controllers that "
              "hold their lock agree with the old PD-far numbers within 0.09 pm; the ones that lose a ring diverge "
              "after the loss. docs/handoff/integrator.md section 7 compares the two.",
              "", "## Per-controller notes", ""]
    for name, r in results.items():
        w = [f"steady {r.get('steady_wall_s', float('nan')):.0f} s"]
        for key in ("a1", "a1_pd_adjacent"):
            if key + "_wall_s" in r:
                w.append(f"{key} {r[key + '_wall_s']:.0f} s")
        lines.append(f"- `{name}` ({r.get('start')}): {r.get('note')}. Wall time: {', '.join(w)}."
                     + (f" Errors: {'; '.join(r['errors'])}" if r.get("errors") else ""))
    lines.append("")
    text = "\n".join(lines)
    OUT.write_text(text)
    JSON_OUT.write_text(json.dumps(_jsonable(results), indent=1))
    return text


def test_write_scorecard():
    """Runs last (definition order): writes docs/handoff/controller_scorecard.md from the runs above."""
    if not _RESULTS:
        pytest.skip("no controller ran in this session")
    text = _write_scorecard(_RESULTS)
    assert "## Steady lock" in text
