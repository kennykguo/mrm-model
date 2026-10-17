#!/usr/bin/env python
"""Regenerate every data file of the webviz site (webviz/site/data/) from the mrm plant model, g26 preset.

    cd ~/mrm-model
    .venv/bin/python tools/memguard.py -- .venv/bin/python webviz/generate.py --date 2026-10-17

Options: --only steady,a1_aggressor,laser_step,cold_start,bistability (a subset; meta.json and checks.json keep the
entries of the files not regenerated), --quick (short runs for a smoke test, written to runs/webviz_quick/ instead of
the site), --out DIR.

Everything written here is model output, never a sketch:
  scenario_<id>.json   engine runs through mrm.sim.experiment's own case pipeline (_plan / _execute: the code path
                       of `mrm run <file>`), so the summary metrics are those of the experiment files; per-frame
                       traces come from the engine's RecordSpec records.
  spectra_<id>.json    the model's phase model (mrm.plant.phase.ring_optics_row) and exact ring
                       (mrm.plant.optics.ring_exact) evaluated at each ring's state of a key frame (Truth: ring and
                       heater-resistor temperature, the two NRZ junction levels, ones density).
  thermal_<id>.json    the model's thermal system (mrm.plant.thermal.assemble with the InterimRom and ThermalScales
                       that mrm.params.realize built for the run) with one extra point observer per grid node,
                       stepped by the model's own kernel (thermal_step_b) with the source powers the engine used
                       (its th_u1 row, read every 10 ticks: the aggressor linear in between, the other sources held);
                       dynamic, not quasi-static.
  bistability.json     slow open-loop heater sweeps of ring 0 on the engine, and the static S-curve of the same ring
                       from the model's static laws (phase model, exact ring, DC thermal gains, heater driver law).
  checks.json          the validation numbers (webviz/README.md, "How the data was checked").
The engine is run through two read-only taps (a Plant subclass): a run with them is checked to give bit-identical
metrics to a run without them.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import subprocess
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SITE_DATA = ROOT / "webviz" / "site" / "data"

from mrm import params  # noqa: E402
from mrm.control import registry as creg  # noqa: E402
from mrm.plant import electrical as el  # noqa: E402
from mrm.plant import optics as op  # noqa: E402
from mrm.plant import phase as ph  # noqa: E402
from mrm.plant import thermal as th  # noqa: E402
from mrm.plant.plant import Plant  # noqa: E402
from mrm.scenarios import generators as gen  # noqa: E402
from mrm.scenarios import registry as sreg  # noqa: E402
from mrm.sim import experiment as ex  # noqa: E402
from mrm.sim import kernel as K  # noqa: E402
from mrm.sim import runner  # noqa: E402
from mrm.sim.metrics import RecordSpec  # noqa: E402
from mrm.units import C0  # noqa: E402

# ============================================================================================ settings
#: thermal source powers are read every SAMPLE ticks (one controller period of g26: 10 us)
SAMPLE = 10
#: spectra: per-ring window of +-HALF_NM around its centre, N_SPEC points (12.5 pm step)
HALF_NM = 1.5
N_SPEC = 241
#: thermal map grid (um): every ring centre, PD, driver and aggressor site sits 1.0 um (x) and 1.25 um (y) from
#: its four nearest nodes, so all of them are drawn alike
GRID_X = -61.0 + 4.0 * np.arange(120)          # -61 .. 415 um
GRID_Y = -61.25 + 2.5 * np.arange(50)          # -61.25 .. 61.25 um
#: display geometry not modelled by the plant (the model has no waveguide layout): rib width, coupling gaps,
#: in-ring heater placement, all from the model's own extraction inputs (assumed values there)
RIB_W_UM = 0.400                                # ring.geometry.width_nm
GAP_THROUGH_UM = 0.1831                         # coupling_rib160.json through.gap_nm (edge to edge)
GAP_DROP_UM = 0.3819                            # coupling_rib160.json drop.gap_nm
HEATER_GAP_UM = 2.0                             # mrm.extraction.thermal_fem.ThermalConfig.heater_gap_um (assumed)
HEATER_W_UM = 1.0                               # ThermalConfig.heater_width_um (assumed)

#: the per-ring trace records of every scenario (field, reducer, export name, scale, unit)
TRACE = (
    ("t_ring_k", "mean", "t_ring_k", 1.0, "K"),
    ("t_heater_k", "mean", "t_heater_k", 1.0, "K"),
    ("t_pd_k", "mean", "t_pd_k", 1.0, "K"),
    ("detuning_m", "mean", "detuning_pm", 1e12, "pm"),
    ("detuning_m", "min", "detuning_min_pm", 1e12, "pm"),
    ("detuning_m", "max", "detuning_max_pm", 1e12, "pm"),
    ("lam_res_m", "mean", "lam_res_nm", 1e9, "nm"),
    ("heater_code", "last", "heater_code", 1.0, "code"),
    ("p_heater_w", "mean", "p_heater_mw", 1e3, "mW"),
    ("p_abs_w", "mean", "p_abs_mw", 1e3, "mW"),
    ("monitor_code", "last", "monitor_code", 1.0, "code"),
    ("i_monitor_a", "mean", "i_monitor_ua", 1e6, "uA"),
    ("p_in_w", "mean", "p_in_mw", 1e3, "mW"),
    ("p_drop_w", "mean", "p_drop_uw", 1e6, "uW"),
    ("reloma", "mean", "reloma_db", None, "dB"),
    ("oracle_reloma", "mean", "oracle_reloma_db", None, "dB"),
    ("line_on", "last", "line_on", 1.0, "index"),
    ("declared_line", "last", "declared_line", 1.0, "index"),
    ("order", "last", "order", 1.0, "index"),
    ("t_base_k", "mean", "t_base_k", 1.0, "K"),
)
INT_FIELDS = {"heater_code", "monitor_code", "line_on", "declared_line", "order"}


def scenario_defs(quick: bool):
    """The four scenarios. dec: ticks per frame; key: key-frame rule [(t0_ms, t1_ms, every_n_frames), ...]."""
    q = (lambda full, short: short if quick else full)
    a1_period_ms = 130.7
    return [
        dict(id="steady", file="experiments/examples/steady_pi_lock.yaml",
             overrides=[] if not quick else ["experiment.duration_s=0.01"],
             title="Steady lock", dec=100, key=[(0, 1e9, q(6, 4))],
             description="pi_lock_c holds all 8 rings at 25 °C from a settled warm start, no disturbance (40 ms)."),
        dict(id="a1_aggressor", file="experiments/examples/a1_pi_lock.yaml",
             overrides=[] if not quick else ["experiment.duration_s=0.02"],
             title="A1 on-die aggressor", dec=250, key=[(0, 1e9, 12), (56.0, 76.0, 2)],
             description=("pi_lock_c under the A1 slide-10 aggressor 30 µm from ring 0, one 130.7 ms period from its "
                          "cool minimum (hot cusp at 65.35 ms). Ring 0 is heated most; near the cusp its heater "
                          "runs down to code 0 and the ring leaves its operating point.")),
        dict(id="laser_step", file="experiments/examples/steady_pi_lock.yaml",
             overrides=["experiment.duration_s=" + ("0.03" if not quick else "0.012")],
             step_s=0.010, title="+10 % laser power step", dec=100,
             key=[(0, 1e9, 6), (9.9, 11.0, 1), (11.0, 13.0, 2)],
             description=("pi_lock_c at 25 °C; at 10 ms every laser line steps +10 % in power (0.414 dB), the "
                          "rings heat themselves more and the lock moves its set point.")),
        dict(id="cold_start", file="experiments/acceptance/b5_cold_start.yaml",
             overrides=["experiment.kind=run", "experiment.batch=1", "experiment.uncertainty=nominal",
                        "experiment.variation=off", "scenario.params.ambient_c_range=[25.0, 25.0]",
                        "experiment.duration_s=" + ("0.035" if not quick else "0.01")],
             title="Cold start (WDM supervisor)", dec=100, key=[(0, 25.0, 4), (25.0, 1e9, 20)],
             description=("wdm_supervisor from a cold start at 25 °C: heaters at 0, lasers ramp on over 3 ms; it "
                          "sweeps each ring to map the lines, assigns lines (a barrel shift is allowed) and locks "
                          "the rings one by one, in bus order.")),
    ], a1_period_ms


# ============================================================================================ helpers
def rnd(a, sig: int = 4, dec: int | None = None) -> np.ndarray:
    """Round to `sig` significant digits (and to `dec` decimals when given, whichever is coarser)."""
    x = np.array(a, dtype=float)
    fin = np.isfinite(x) & (x != 0.0)
    if np.any(fin):
        mag = np.floor(np.log10(np.abs(x[fin])))
        f = 10.0 ** (sig - 1 - mag)
        x[fin] = np.round(x[fin] * f) / f
    if dec is not None:
        f2 = np.isfinite(x)
        x[f2] = np.round(x[f2], dec)
    return x


def _none(o):
    if isinstance(o, list):
        return [_none(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def js(a, sig: int = 4, dec: int | None = None):
    """JSON-ready nested lists, rounded, NaN -> null."""
    return _none(rnd(a, sig, dec).tolist())


def jint(a):
    x = np.asarray(a, dtype=float)
    return _none(np.where(np.isfinite(x), np.rint(x), np.nan).tolist()) if np.any(~np.isfinite(x)) else \
        np.rint(x).astype(np.int64).tolist()


def jnum(v, sig=6):
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    v = float(v)
    return float(rnd(v, sig)) if math.isfinite(v) else None


def write_json(path: Path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(obj, separators=(",", ":"), allow_nan=False)
    path.write_text(txt)
    return len(txt)


def db(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(np.asarray(x, float))


def git_state():
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "--short", "HEAD"),
            "tracked_files_modified": bool(git("status", "--porcelain", "--untracked-files=no"))}


# ============================================================================================ engine taps
class _Tap:
    """What the tapped plants collect (one entry per Plant the runner builds: the self-reference pre-run first)."""

    def __init__(self, snap_ticks):
        self.snap_ticks = set(int(t) for t in snap_ticks)
        self.plants: list = []


class _TapPlant(Plant):
    """runner.run's Plant with two read-only taps, active after reset: the thermal source powers (th_u1, the input
    of the last interval) every SAMPLE ticks, and Truth plus th_u1 at the tap's snapshot ticks. On the compiled and
    open-loop paths a kernel block is cut into SAMPLE-tick blocks; every cut is a controller edge (g26: period 10,
    phase 0), and under the fixed substep policy a block end changes nothing (docs/ENGINE.md section 4; checked
    here: bit-identical metrics). The Python path stops at every controller edge already."""
    tap: _Tap | None = None

    def reset(self, *a, **kw):
        self._armed = False
        out = super().reset(*a, **kw)
        if int(self.P.policy) != K.POLICY_FIXED:
            raise RuntimeError("the taps need the fixed substep policy (block ends would change an auto run)")
        if int(self.P.jitter) != 0 or int(self.P.period) % SAMPLE or int(self.P.phase) % SAMPLE:
            raise RuntimeError("the taps need controller edges on multiples of SAMPLE ticks")
        self.u_ticks = [int(self.tick)]
        self.u_rows = [self.S.th_u1[0].copy()]
        self.x_reset = self.S.th_x[0].copy()
        self.snaps = {}
        if int(self.tick) in self.tap.snap_ticks:
            self.snaps[int(self.tick)] = (self.truth(), self.S.th_u1[0].copy())
        self.tap.plants.append(self)
        self._armed = True
        return out

    def _cap(self):
        k = int(self.tick)
        if k != self.u_ticks[-1]:
            self.u_ticks.append(k)
            self.u_rows.append(self.S.th_u1[0].copy())
        if k in self.tap.snap_ticks and k not in self.snaps:
            self.snaps[k] = (self.truth(), self.S.th_u1[0].copy())

    def run(self, n_ticks, *, mode=K.MODE_OPEN, step_fn=None, fast_fn=None, resume=False, n_threads=None):
        if not getattr(self, "_armed", False):
            return super().run(n_ticks, mode=mode, step_fn=step_fn, fast_fn=fast_fn, resume=resume,
                               n_threads=n_threads)
        if mode == K.MODE_PYTHON:
            d = super().run(n_ticks, mode=mode, step_fn=step_fn, fast_fn=fast_fn, resume=resume, n_threads=n_threads)
            self._cap()
            return d
        end = self.tick + int(n_ticks)
        done = 0
        while self.tick < end:
            nxt = min(end, (self.tick // SAMPLE + 1) * SAMPLE)
            done += super().run(nxt - self.tick, mode=mode, step_fn=step_fn, fast_fn=fast_fn, resume=resume,
                                n_threads=n_threads)
            resume = False
            self._cap()
        return done


@contextlib.contextmanager
def tapped(snap_ticks):
    tap = _Tap(snap_ticks)
    old = runner.Plant
    _TapPlant.tap = tap
    runner.Plant = _TapPlant
    try:
        yield tap
    finally:
        runner.Plant = old
        _TapPlant.tap = None


def flat_metrics(m, pre=""):
    out = {}
    for k, v in m.items():
        if isinstance(v, dict):
            out.update(flat_metrics(v, pre + k + "."))
        else:
            a = np.asarray(v)
            if a.dtype.kind in "fiub":
                out[pre + k] = a
    return out


def same_metrics(m0, m1):
    f0, f1 = flat_metrics(m0), flat_metrics(m1)
    bad = [k for k in f0 if k not in f1 or not np.array_equal(f0[k], f1[k], equal_nan=True)]
    return not bad, bad


# ============================================================================================ scenarios
def load_case(sd):
    exp = ex.load(str(ROOT / sd["file"]), overrides=list(sd["overrides"]) or None)
    case = ex._plan(exp)[0]
    sc = None
    if sd["id"] == "laser_step":
        base = sreg.from_config(exp.cfg)
        step = gen.Scenario(name="laser_step", components=(gen.step(
            "laser.power_delta_db", sd["step_s"], 10.0 * math.log10(1.1), label="laser_step_10pct",
            provenance="assumed", source="", note="webviz: +10 % on every laser line (as tools/perf_scenarios.py)"),),
            description=f"+10 % laser power on every line at {sd['step_s']} s")
        sc = base + step
    return exp, case, sc


def records_for(dec):
    return [RecordSpec(f, dec, red) for f, red, *_ in TRACE]


def key_frames(t_s, rules):
    t_ms = np.asarray(t_s) * 1e3
    idx = {0, len(t_ms) - 1}
    for t0, t1, every in rules:
        sel = np.flatnonzero((t_ms >= t0 - 1e-9) & (t_ms <= t1 + 1e-9))
        if sel.size:
            idx.update(sel[(sel - sel[0]) % every == every - 1].tolist())
            idx.add(int(sel[0]))
    return np.array(sorted(idx), dtype=np.int64)


def run_scenario(sd, log):
    """Two runs of the experiment's main case: without taps (frame times, reference metrics) and with them."""
    exp, case, sc = load_case(sd)
    dt = float(exp.cfg.get("sim.dt_plant_s"))
    case = dataclasses.replace(case, record=records_for(sd["dec"]), record_default=False, scenario=sc)
    t0 = time.perf_counter()
    plain = ex._execute(exp, case)
    w0 = time.perf_counter() - t0
    tkey = next(iter(plain.trace))
    t_s = np.asarray(plain.trace[tkey]["t_s"])
    kidx = key_frames(t_s, sd["key"])
    ticks = np.rint(t_s[kidx] / dt).astype(np.int64)
    t0 = time.perf_counter()
    with tapped(ticks) as tap:
        cr = ex._execute(exp, case)
    w1 = time.perf_counter() - t0
    plant = tap.plants[-1]
    ok, bad = same_metrics(plain.metrics, cr.metrics)
    tr_ok = all(np.array_equal(plain.trace[k]["value"], cr.trace[k]["value"], equal_nan=True) for k in plain.trace)
    missing = sorted(set(ticks.tolist()) - set(plant.snaps))
    if missing:
        raise RuntimeError(f"{sd['id']}: no snapshot at ticks {missing[:5]}")
    log(f"  {sd['id']}: {len(t_s)} frames, {len(kidx)} key frames; wall {w0:.1f} s plain, {w1:.1f} s tapped; "
        f"taps change nothing: metrics {ok}, traces {tr_ok}")
    return dict(sd=sd, exp=exp, case=case, cr=cr, plain=plain, plant=plant, t_s=t_s, kidx=kidx, ticks=ticks,
                dt=dt, tap_identical=bool(ok and tr_ok), tap_bad=bad)


def official_metrics(sd):
    """`mrm.run(<file>, overrides, write=False)`: the experiment pipeline itself (default records)."""
    res = ex.run(str(ROOT / sd["file"]), overrides=list(sd["overrides"]) or None, write=False, log=lambda s: None)
    return res.cases["main"].metrics


# ============================================================================================ scenario arrays
def scenario_json(R):
    sd, cr, plant, t_s, dt = R["sd"], R["cr"], R["plant"], R["t_s"], R["dt"]
    tr = cr.trace
    S = len(t_s)
    out = {"id": sd["id"], "title": sd["title"], "description": sd["description"], "preset": "g26",
           "experiment_file": sd["file"], "overrides": sd["overrides"],
           "controller": (R["case"].controller or {}).get("name"),
           "t_s": js(t_s, 6), "frame_s": sd["dec"] * dt, "frame_reducer_note":
               "each frame is the window (t_s - frame_s, t_s]: means over it unless the unit list says 'last' "
               "(value at t_s) or 'min'/'max' (envelope over the window)",
           "key_frames": R["kidx"].tolist(), "rings": {}, "units": {}, "reducers": {}}
    rings = {}
    for f, red, name, scale, unit in TRACE:
        v = np.asarray(tr[RecordSpec(f, sd["dec"], red).key]["value"])[:, 0, :]          # (S, N)
        if name in ("reloma_db", "oracle_reloma_db"):
            arr = js(db(v), 4, 3)
        elif name in INT_FIELDS:
            arr = jint(v)
        elif name == "lam_res_nm":
            arr = js(v * scale, 8)
        elif name.startswith("detuning"):
            arr = js(v * scale, 6, 3)
        elif name in ("t_ring_k", "t_heater_k", "t_pd_k", "t_base_k"):
            arr = js(v * scale, 6, 3)
        else:
            arr = js(v * scale, 4, 4)
        rings[name] = arr
        out["units"][name] = unit
        out["reducers"][name] = red
    out["rings"] = rings
    # lock reference (self-referenced runs) and the lock error
    m = cr.metrics
    ref = cr.reference
    d_ref = np.asarray(ref["d_ref_pm"])[0] if ref is not None else np.zeros(plant.N)
    out["lock_reference"] = {"kind": "self_reference" if ref is not None else "line",
                             "d_ref_pm": js(d_ref, 5),
                             "note": ("each ring's own steady operating point (median detuning over the last 5 ms "
                                      "of a 20 ms static pre-run, experiment.self_reference)" if ref is not None else
                                      "the declared line (lock error = detuning)")}
    # scenario drives (window means of the compiled scenario's signals, every 10 ticks)
    arr = plant.scen_arrays
    n_end = int(round(t_s[-1] / dt))
    ticks = np.arange(0, n_end + 1, SAMPLE, dtype=np.int64)
    sig = [s for s in ("thermal.aggressor_w", "thermal.t_base_k", "laser.power_delta_db", "laser.line_enable")
           if s in arr.layout]
    fr = arr.render(ticks=ticks, signals=sig)
    drives = {}
    dec = sd["dec"]

    def wmean(v):                                     # (T, W) sampled every SAMPLE ticks -> (S, W) window means
        per = dec // SAMPLE
        v = v[1:1 + S * per]                          # samples at ticks SAMPLE .. end (window (t - frame, t])
        return v.reshape(S, per, -1).mean(axis=1)
    if "thermal.aggressor_w" in fr.signals:
        a = wmean(fr.signals["thermal.aggressor_w"][:, 0, :])
        names = list(arr.aggressor_sites) or [f"site{i}" for i in range(a.shape[1])]
        drives["aggressor_w"] = {nm: js(a[:, i], 4, 6) for i, nm in enumerate(names)}
    pdb = wmean(fr.signals["laser.power_delta_db"][:, 0, :]) if "laser.power_delta_db" in fr.signals else None
    en = wmean(fr.signals["laser.line_enable"][:, 0, :]) if "laser.line_enable" in fr.signals else None
    L = plant.L
    fac = np.ones((S, L))
    if pdb is not None:
        fac = fac * 10.0 ** (np.broadcast_to(pdb, (S, L)) / 10.0)
    if en is not None:
        fac = fac * np.clip(np.broadcast_to(en, (S, L)), 0, 1)
    drives["laser_power_factor"] = js(fac, 5)
    drives["t_base_k"] = js(np.asarray(tr[RecordSpec("t_base_k", dec, "mean").key]["value"])[:, 0, 0], 6, 3)
    drives["note"] = ("aggressor_w: power into each aggressor site (window mean); laser_power_factor: chip power of "
                      "each line relative to nominal (scenario power step x line enable); the cold start's 3 ms "
                      "laser turn-on ramp is a plant reset protocol, visible in p_in_mw, not in this factor; "
                      "t_base_k: the die base temperature the thermal model adds every rise to")
    out["drives"] = drives
    # event markers for a timeline (from the data, not typed in)
    ev = []
    if "aggressor_w" in drives and sd["id"] == "a1_aggressor":
        a = np.array(drives["aggressor_w"].get("aggressor", []), float)
        if a.size and np.nanmax(a) > 0:
            i = int(np.nanargmax(a))
            ev.append({"t_s": jnum(t_s[i] - 0.5 * sd["dec"] * dt), "label": "aggressor power at its peak",
                       "value": jnum(a[i]), "unit": "W"})
            tgt = [x for x in sc_of(R).aggressors if x.site == "aggressor"]
            if tgt:
                tg = tgt[0].target
                if tg is not None and tg.form == "rectified_sine":
                    t_cusp = (tg.period_s - tg.phase_s) % tg.period_s
                    ev.append({"t_s": jnum(t_cusp), "label": "ring 0 target temperature at its hot cusp (A1)"})
    if sd.get("step_s") is not None:
        ev.append({"t_s": sd["step_s"], "label": "+10 % laser power on every line"})
    if sd["id"] == "cold_start":
        ev.append({"t_s": 0.0, "label": "lasers start their 3 ms turn-on ramp; heaters at code 0"})
    sf = cr.manifest_inputs.get("score_from_s")
    if sf:
        ev.append({"t_s": jnum(sf), "label": "metrics start (the lock's start-up peak map ends before this)"})
    ttl = np.asarray(cr.metrics.get("time_to_lock_s", np.full((1, plant.N), np.nan)), float)[0]
    if sd["id"] == "cold_start":
        for k in range(plant.N):
            if np.isfinite(ttl[k]):
                ev.append({"t_s": jnum(ttl[k]), "label": f"ring {k} locked", "ring": k})
    out["events"] = sorted(ev, key=lambda e: e["t_s"])
    out["units"].update({"aggressor_w": "W", "laser_power_factor": "1", "t_s": "s", "frame_s": "s",
                         "d_ref_pm": "pm"})
    # summary metrics (the run's own, after score_from_s)
    per_ring = {}
    for k in ("max_abs_lock_error_pm", "rms_lock_error_pm", "oma_penalty_db", "mean_reloma_db", "oracle_reloma_db",
              "max_abs_detuning_pm", "rms_detuning_pm", "excursion_count", "excursion_max_s", "time_unlocked_s",
              "first_lock_s", "time_to_lock_s", "wrong_line_s", "heater_low_rail_s", "heater_sat_s",
              "code_min_used", "code_max_used", "monitor_error_pct_min", "monitor_error_pct_max"):
        if k in m:
            per_ring[k] = js(np.asarray(m[k], float)[0], 5)
    for k in ("locked_final", "ever_locked", "wrong_line_final"):
        if k in m:
            per_ring[k] = [bool(x) for x in np.asarray(m[k])[0]]
    locked = np.asarray(m["locked_final"])[0]
    wrong = np.asarray(m.get("wrong_line_final", np.zeros_like(locked)))[0]
    pen = np.asarray(m["oma_penalty_db"], float)[0]
    settings = m.get("settings", {})
    mi = cr.manifest_inputs
    out["summary"] = {
        "per_ring": per_ring,
        "max_abs_lock_error_pm": jnum(np.nanmax(np.asarray(m["max_abs_lock_error_pm"], float)[0])),
        "rms_lock_error_pm_worst": jnum(np.nanmax(np.asarray(m["rms_lock_error_pm"], float)[0])),
        "oma_penalty_db_worst": jnum(np.nanmax(pen)),
        "unlocked_rings": int(np.sum(~locked.astype(bool))),
        "wrong_line_rings": int(np.sum(wrong.astype(bool))),
        "excursions_total": int(np.sum(np.asarray(m["excursion_count"])[0])),
        "lock_threshold_pm": jnum(settings.get("lock_threshold_pm")),
        "hysteresis_pm": jnum(settings.get("hysteresis_pm")),
        "lock_reference": settings.get("lock_reference"),
        "allowed_assignments": settings.get("allowed_assignments"),
        "score_from_s": jnum(mi.get("score_from_s")),
        "duration_s": jnum(mi.get("duration_s")),
        "controller": mi.get("controller"),
        "controller_path": mi.get("controller_path"),
        "aggressor_calibration": _agg_cal(mi.get("aggressor_calibration")),
        "warnings": list(cr.warnings),
        "note": "metrics are the engine's own accumulators (every 1 us tick after score_from_s), as `mrm run` "
                "reports them",
    }
    return out


def sc_of(R):
    """The scenario object the run used (the case's own, else the config's)."""
    return R["case"].scenario if R["case"].scenario is not None else sreg.from_config(R["exp"].cfg)


def _agg_cal(cal):
    if not cal:
        return None
    out = {}
    for site, rec in cal.items():
        r = {}
        for k, v in dict(rec).items():
            if isinstance(v, (int, float, np.floating, np.integer)) and math.isfinite(float(v)):
                r[k] = jnum(v)
            elif isinstance(v, str):
                r[k] = v
        out[site] = r
    return out


# ============================================================================================ spectra
class Spectra:
    """Per-ring through and drop transmission from the model's phase model and exact ring at a Truth state."""

    def __init__(self, pp):
        self.pp = pp
        self.phs = ph.PhaseParams.from_plant_params(pp)
        if int(pp.flags.optics_model) != 0:
            raise RuntimeError("the spectra assume the exact-ring optics model (g26 default)")
        rb = np.asarray(pp.optics.backscatter_rb)
        if np.any(rb != 0):
            raise RuntimeError("the spectra assume no backscatter (g26 nominal)")
        self.offsets_m = np.linspace(-HALF_NM, HALF_NM, N_SPEC) * 1e-9

    def levels(self, truth):
        f = truth.fields
        c = truth.conditions[0]
        return (np.ascontiguousarray(f["t_ring_k"][0]), np.ascontiguousarray(f["t_heater_k"][0]),
                np.ascontiguousarray(c.v_lo), np.ascontiguousarray(c.v_hi), np.asarray(c.rho, float))

    def optics(self, truth, lam):
        """phase-model rows of every ring at wavelengths lam (W,): dict of (N, W) arrays."""
        t_ring, t_heat, v0, v1, _ = self.levels(truth)
        phs, r = self.phs, self.phs.table_row(0)
        N, W = self.pp.n_rings, lam.size
        o = {k: np.zeros((N, W)) for k in ("phi0", "phi1", "a0", "a1", "e0", "e1", "k2t", "k2d", "trt")}
        shift, cj, rs, fl = np.zeros((N, 2)), np.zeros((N, 2)), np.zeros((N, 2)), np.zeros(N, np.int8)
        ph.ring_optics_row(phs.ring[0], np.ascontiguousarray(lam, float), t_ring, t_heat, v0, v1, phs.grid,
                           phs.dneff[r], phs.alpha_fca[r], phs.capacitance[r], phs.r_series[r], o["phi0"], o["phi1"],
                           o["a0"], o["a1"], o["e0"], o["e1"], o["k2t"], o["k2d"], o["trt"], shift, cj, rs, fl)
        o["shift"] = shift
        return o

    def ring_response(self, o, k, sl):
        r0 = op.ring_exact(o["phi0"][k, sl], o["k2t"][k, sl], o["k2d"][k, sl], o["a0"][k, sl], o["trt"][k, sl])
        r1 = op.ring_exact(o["phi1"][k, sl], o["k2t"][k, sl], o["k2d"][k, sl], o["a1"][k, sl], o["trt"][k, sl])
        return r0, r1

    def frame(self, truth, centers_m):
        """(thru0, thru1, drop_avg) (N, G) power transmissions on the windows centred at centers_m (N,)."""
        N, G = self.pp.n_rings, self.offsets_m.size
        lam = (np.asarray(centers_m)[:, None] + self.offsets_m[None, :]).ravel()
        o = self.optics(truth, lam)
        rho = self.levels(truth)[4]
        t0 = np.zeros((N, G))
        t1 = np.zeros((N, G))
        dr = np.zeros((N, G))
        for k in range(N):
            sl = slice(k * G, (k + 1) * G)
            r0, r1 = self.ring_response(o, k, sl)
            t0[k], t1[k] = r0.p_thru, r1.p_thru
            dr[k] = rho[k] * np.abs(r1.drop) ** 2 + (1.0 - rho[k]) * np.abs(r0.drop) ** 2
        return t0, t1, dr

    def at_lines(self, truth):
        """Each ring at every line wavelength: thru0/thru1/drop0/drop1/loss0/loss1 (N, L) and eta (N, L, 2)."""
        lam = np.asarray(truth.lam_m[0], float)
        o = self.optics(truth, lam)
        N, L = self.pp.n_rings, lam.size
        out = {k: np.zeros((N, L)) for k in ("t0", "t1", "d0", "d1", "l0", "l1")}
        for k in range(N):
            r0, r1 = self.ring_response(o, k, slice(0, L))
            out["t0"][k], out["t1"][k] = r0.p_thru, r1.p_thru
            out["d0"][k], out["d1"][k] = np.abs(r0.drop) ** 2, np.abs(r1.drop) ** 2
            out["l0"][k], out["l1"][k] = r0.p_loss, r1.p_loss
        out["e0"], out["e1"] = o["e0"], o["e1"]
        return out

    def resonance_exact(self, truth, k, lam_line):
        """Exact level-0 and level-1 resonance (m) of ring k's order nearest lam_line, and the notch transmission
        there (the model's closed form: the exact ring at phase 2 pi m)."""
        o = self.optics(truth, np.array([lam_line]))
        rp = self.phs.ring[0][k]
        res, notch = [], []
        for lvl in (0, 1):
            phi = o[f"phi{lvl}"][k, 0]
            m0 = round(phi / (2 * math.pi))
            # the order whose resonance is nearest the line in wavelength (as Truth's lam_res)
            d, m = min(((K.resonance_detuning(rp, lam_line, o["shift"][k, lvl], float(mm)), mm)
                        for mm in (m0 - 1, m0, m0 + 1)), key=lambda dm: abs(dm[0]))
            lam_r = lam_line + d
            o2 = self.optics(truth, np.array([lam_r]))
            rr = op.ring_exact(np.array([2 * math.pi * m]), o2["k2t"][k, :1], o2["k2d"][k, :1],
                               o2[f"a{lvl}"][k, :1], o2["trt"][k, :1])
            res.append(lam_r)
            notch.append(float(rr.p_thru[0]))
        return res, notch


def spectra_json(R, spec: Spectra, checks):
    sd, plant, t_s, kidx, ticks = R["sd"], R["plant"], R["t_s"], R["kidx"], R["ticks"]
    N = plant.N
    center_kind = "resonance" if sd["id"] == "cold_start" else "declared_line"
    T0, T1, DR, CEN, LRES, LINES, VLO, VHI, RHO, TR, TH = [], [], [], [], [], [], [], [], [], [], []
    RESL, NOTCH = [], []
    BUS_IN, BUS_THRU = [], []
    err_res_pm, err_notch_db, err_thru_rel, err_reloma, err_drop_rel, in_step = [], [], [], [], [], []
    G = spec.offsets_m.size
    step_m = spec.offsets_m[1] - spec.offsets_m[0]
    for tick in ticks:
        truth = plant.snaps[int(tick)][0]
        f = truth.fields
        lam_l = np.asarray(truth.lam_m[0], float)
        decl = np.rint(f["declared_line"][0]).astype(int)
        lam_res = np.asarray(f["lam_res_m"][0], float)
        cen = lam_l[decl] if center_kind == "declared_line" else lam_res
        t0, t1, dr = spec.frame(truth, cen)
        T0.append(t0), T1.append(t1), DR.append(dr), CEN.append(cen), LRES.append(lam_res), LINES.append(lam_l)
        t_ring, t_heat, v0, v1, rho = spec.levels(truth)
        VLO.append(v0), VHI.append(v1), RHO.append(rho), TR.append(t_ring), TH.append(t_heat)
        # checks against the engine's own outputs at this instant
        lines = spec.at_lines(truth)
        p_in = truth.out_rl[0, op.RL_PIN]                                 # (N, L) W
        BUS_IN.append(np.array(p_in, float)), BUS_THRU.append(np.array(truth.out_rl[0, op.RL_THRU], float))
        resl, notl = np.zeros((N, 2)), np.zeros((N, 2))
        for k in range(N):
            l = decl[k]
            res, notch = spec.resonance_exact(truth, k, lam_l[l])
            resl[k], notl[k] = res, notch
            mean_res = 0.5 * (res[0] + res[1])
            err_res_pm.append((mean_res - lam_res[k]) * 1e12)
            # the NRZ-average through spectrum's minimum, on the exported grid, against lam_res (grid steps)
            avg = rho[k] * t1[k] + (1 - rho[k]) * t0[k]
            lam_grid = cen[k] + spec.offsets_m
            i = int(np.argmin(avg))
            if abs(lam_res[k] - cen[k]) < HALF_NM * 1e-9 - step_m:
                in_step.append(abs(lam_grid[i] - lam_res[k]) / step_m)
            # notch depth: grid minimum of each level against the closed form at its exact resonance
            for lvl, tt in ((0, t0[k]), (1, t1[k])):
                if abs(res[lvl] - cen[k]) < HALF_NM * 1e-9 - step_m:
                    err_notch_db.append(float(db(tt.min()) - db(notch[lvl])))
            # the spectra at the line reproduce the engine's through power, relOMA and monitor input
            thr = rho[k] * lines["t1"][k, l] + (1 - rho[k]) * lines["t0"][k, l]
            eng = truth.out_rl[0, op.RL_THRU, k, l] / p_in[k, l]
            err_thru_rel.append(abs(thr - eng) / max(eng, 1e-30))
            err_reloma.append(abs(abs(lines["t1"][k, l] - lines["t0"][k, l]) - f["reloma"][0, k]))
            drop = float(np.sum(p_in[k] * (rho[k] * lines["d1"][k] + (1 - rho[k]) * lines["d0"][k])))
            err_drop_rel.append(abs(drop - f["p_drop_w"][0, k]) / max(f["p_drop_w"][0, k], 1e-30))
        RESL.append(resl), NOTCH.append(notl)
    checks[sd["id"]]["spectra"] = {
        "key_frames": int(len(ticks)),
        "resonance_vs_trace_pm_max_abs": jnum(np.max(np.abs(err_res_pm))),
        "nrz_avg_minimum_vs_lam_res_grid_steps_max": jnum(np.max(in_step)) if in_step else None,
        "grid_step_pm": jnum(step_m * 1e12),
        "notch_grid_minus_exact_db_range": [jnum(np.min(err_notch_db)), jnum(np.max(err_notch_db))]
        if err_notch_db else None,
        "thru_at_line_vs_engine_rel_max": jnum(np.max(err_thru_rel)),
        "reloma_at_line_vs_engine_abs_max": jnum(np.max(err_reloma)),
        "drop_sum_vs_engine_p_drop_rel_max": jnum(np.max(err_drop_rel)),
    }
    out = {
        "id": sd["id"], "frames": R["kidx"].tolist(), "t_s": js(t_s[kidx], 6),
        "grid": {"mode": "per_ring_window", "center_kind": center_kind,
                 "offset_nm": js(spec.offsets_m * 1e9, 6), "half_width_nm": HALF_NM, "points": int(G),
                 "step_pm": jnum(step_m * 1e12),
                 "note": ("wavelength = center_nm[frame][ring] + offset_nm. declared_line: the window is centred on "
                          "the ring's own (declared) laser line, so offset = wavelength minus line. resonance: "
                          "centred on the ring's resonance of the order nearest its declared line (the cold start, "
                          "where the resonance starts nm away from any line)")},
        "center_nm": js(np.array(CEN) * 1e9, 9),
        "lam_res_nm": js(np.array(LRES) * 1e9, 9),
        "line_nm": js(np.array(LINES) * 1e9, 9),
        "thru0_db": js(db(np.array(T0)), 4, 2),
        "thru1_db": js(db(np.array(T1)), 4, 2),
        "drop_db": js(db(np.array(DR)), 4, 2),
        "resonance_level_nm": js(np.array(RESL) * 1e9, 9),
        "notch_level_db": js(db(np.array(NOTCH)), 4, 2),
        "bus": {"p_in_mw": js(np.array(BUS_IN) * 1e3, 4), "p_thru_mw": js(np.array(BUS_THRU) * 1e3, 4),
                "note": ("the engine's own optics outputs at t_s (Truth out_rl), [frame][ring][line] in mW: p_in = "
                         "light of each line arriving at the ring's input after the bus segment (RL_PIN), p_thru = "
                         "light of each line leaving its through port, data-averaged (RL_THRU); the bus between ring "
                         "k and k+1 carries p_thru[k] going in and p_in[k+1] arriving (the difference is bus loss)")},
        "levels": {"v_lo_v": js(np.array(VLO), 4), "v_hi_v": js(np.array(VHI), 4), "ones_density": js(np.array(RHO), 3),
                   "t_ring_k": js(np.array(TR), 6, 3), "t_heater_k": js(np.array(TH), 6, 3)},
        "units": {"offset_nm": "nm", "center_nm": "nm", "lam_res_nm": "nm", "line_nm": "nm", "thru0_db": "dB",
                  "thru1_db": "dB", "drop_db": "dB", "v_lo_v": "V", "v_hi_v": "V", "ones_density": "1",
                  "t_ring_k": "K", "t_heater_k": "K", "t_s": "s", "resonance_level_nm": "nm",
                  "notch_level_db": "dB"},
        "exact_note": ("resonance_level_nm [frame][ring][level]: the exact resonance of each NRZ level (order nearest "
                       "the declared line); notch_level_db: the through transmission exactly there (the grid's "
                       "12.5 pm step does not resolve the bottom of a deep notch; draw the bottom from these)"),
        "note": ("power transmission of each ring alone (input 1 W on the bus at that wavelength): thru0_db / thru1_db "
                 "the through port with the junction at NRZ level 0 (v_lo, lower reverse bias) / level 1 (v_hi); "
                 "drop_db the drop port (to the monitor PD) averaged over the data, ones_density x level 1 + (1 - "
                 "ones_density) x level 0, which is what the monitor and the lock see. The NRZ-averaged through port "
                 "is the same average of thru0 and thru1 in linear units. Computed with the model's phase model "
                 "(mrm.plant.phase) and exact ring (mrm.plant.optics.ring_exact) at the ring's state at t_s (ring "
                 "and heater-resistor temperature, bias levels): rigid-comb phase model, no backscatter (g26)."),
    }
    return out


# ============================================================================================ thermal maps
def capture_thermal_inputs(cfg, uncertainty="nominal", variation="off", seeds=None):
    """The InterimRom, ThermalScales and assembly arguments that mrm.params.realize passes to
    mrm.plant.thermal.assemble for this config and world, through realize's own thermal_hook."""
    sd = runner._seeds(seeds)
    cap = {}

    def assemble(rom, layout, n, scales, **kw):
        cap.update(rom=rom, layout=layout, n=n, scales=scales, kw=kw)
        return th.assemble(rom, layout, n, scales, **kw)
    hook = types.SimpleNamespace(__name__="webviz_capture", ThermalScales=th.ThermalScales, InterimRom=th.InterimRom,
                                 load_rom=th.load_rom, assemble=assemble, selfheat_dc_k_per_w=th.selfheat_dc_k_per_w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp = params.realize(cfg, 1, seeds={"device": sd["device"], "uncertainty": sd["uncertainty"]},
                            uncertainty=uncertainty, variation=variation, thermal_hook=hook)
    if not isinstance(cap.get("rom"), th.InterimRom):
        raise RuntimeError("the thermal map assumes the interim thermal model (g26 default)")
    return cap, pp


class ThermalMap:
    """The model's thermal system with one extra point observer per grid node (and probes at the ring centres and
    monitor PDs): the site law of the interim model (mrm.plant.thermal._interim_lags: c(d) Z0 with one pole
    tau_xt(d) per source), assembled by mrm.plant.thermal.assemble with the run's own InterimRom and scales."""

    def __init__(self, cap, sys_engine, checks):
        self.cap = cap
        sites = dict(cap["layout"])
        self.n = int(cap["n"])
        lay = sys_engine.meta["layout"]
        self.ring_xy = np.asarray(lay.ring_xy_um, float)
        pd_off = np.asarray(cap["rom"].pd_offset_um, float)
        self.pd_xy = self.ring_xy + pd_off[None, :] if lay.pd_xy_um is None else np.asarray(lay.pd_xy_um, float)
        ext = dict(sites)
        self.grid_names = []
        for j, y in enumerate(GRID_Y):
            for i, x in enumerate(GRID_X):
                nm = f"webviz_g{j}_{i}"
                ext[nm] = {"x_um": float(x), "y_um": float(y), "kind": "temp_sensor"}
                self.grid_names.append(nm)
        for k in range(self.n):
            ext[f"webviz_ring{k}"] = {"x_um": float(self.ring_xy[k, 0]), "y_um": float(self.ring_xy[k, 1]),
                                      "kind": "temp_sensor"}
            ext[f"webviz_pd{k}"] = {"x_um": float(self.pd_xy[k, 0]), "y_um": float(self.pd_xy[k, 1]),
                                    "kind": "temp_sensor"}
        t0 = time.perf_counter()
        sysx = th.assemble(cap["rom"], ext, self.n, cap["scales"], **cap["kw"])
        self.wall_assemble = time.perf_counter() - t0
        # the engine's own observers keep exactly their lags in the extended system
        same = sysx.src_names == sys_engine.src_names
        for o, name in enumerate(sys_engine.obs_names):
            ox = sysx.obs_names.index(name)
            a = slice(sys_engine.obs_ptr[o], sys_engine.obs_ptr[o + 1])
            b = slice(sysx.obs_ptr[ox], sysx.obs_ptr[ox + 1])
            same &= (np.array_equal(sys_engine.lag_src[a], sysx.lag_src[b])
                     and np.array_equal(sys_engine.taus_s[:, a], sysx.taus_s[:, b])
                     and np.array_equal(sys_engine.res_k_per_w[:, a], sysx.res_k_per_w[:, b]))
        checks["thermal_system"] = {"engine_observers_identical_in_extended_system": bool(same),
                                    "lags_extended": int(sysx.n_lags), "observers_extended": int(sysx.n_obs),
                                    "assemble_wall_s": jnum(self.wall_assemble, 3)}
        if not same:
            raise RuntimeError("the extended thermal system does not reproduce the engine's lags")
        self.sys = sysx
        self.sys_engine = sys_engine
        self.idx_grid = np.array([sysx.obs_names.index(n) for n in self.grid_names])
        self.idx_ring_probe = np.array([sysx.obs_names.index(f"webviz_ring{k}") for k in range(self.n)])
        self.idx_pd_probe = np.array([sysx.obs_names.index(f"webviz_pd{k}") for k in range(self.n)])
        self.idx_eng = {nm: sysx.obs_names.index(nm) for nm in sys_engine.obs_names}
        self.eng_names = tuple(sys_engine.obs_names)

    def reconstruct(self, plant, ticks):
        """Observer rises (K above the base temperature) at the snapshot ticks: the lags stepped every SAMPLE ticks
        with the exact first-order hold between the source-power samples the engine used (th_u1)."""
        u_t = np.asarray(plant.u_ticks)
        U = np.asarray(plant.u_rows)
        if not np.all(np.diff(u_t) == SAMPLE) or u_t[0] != 0:
            raise RuntimeError("thermal source samples are not every SAMPLE ticks from 0")
        sysx = self.sys
        # initial state: settled starts sit at the steady state of the inputs at reset; a cold start at zero
        eng_x0 = plant.x_reset
        eng_ss = plant.sys.steady_state(U[0][None, :])[0]
        if np.max(np.abs(eng_x0)) == 0.0:
            x = np.zeros(sysx.n_lags)
            init = "zero (cold start: every lag at the base temperature)"
        elif np.max(np.abs(eng_x0 - eng_ss)) < 5e-3:
            x = sysx.steady_state(U[0][None, :])[0].copy()
            init = (f"steady state of the reset inputs (settled start; the engine's settled lags are within "
                    f"{np.max(np.abs(eng_x0 - eng_ss)) * 1e3:.2f} mK of it)")
        else:
            raise RuntimeError("unrecognized initial thermal state")
        a, c0, c1 = sysx.coefficients(SAMPLE * float(plant.dt))
        a, c0, c1 = (np.ascontiguousarray(v[0]) for v in (a, c0, c1))
        ss = np.ones(sysx.n_src)
        # hold between samples: the aggressor waveforms are piecewise linear in the engine (first-order hold); every
        # other source is held per tick and changes at controller edges, so the value read at the end of a 10-tick
        # interval is held over it (checked against the engine's PD and die-sensor observers: README)
        agg = np.zeros(sysx.n_src, bool)
        agg[np.asarray(sysx.aggressor_src, np.int64)] = True
        want = {int(t): i for i, t in enumerate(ticks)}
        out = np.zeros((len(ticks), sysx.n_obs))
        tmp = np.zeros(sysx.n_obs)
        lag_src = np.ascontiguousarray(sysx.lag_src)
        obs_ptr = np.ascontiguousarray(sysx.obs_ptr)
        if 0 in want:
            th.thermal_outputs_b(x, obs_ptr, 0.0, tmp)
            out[want[0]] = tmp
        for k in range(len(u_t) - 1):
            u0 = np.where(agg, U[k], U[k + 1])
            th.thermal_step_b(x, a, c0, c1, lag_src, np.ascontiguousarray(u0), np.ascontiguousarray(U[k + 1]), ss)
            t = int(u_t[k + 1])
            if t in want:
                th.thermal_outputs_b(x, obs_ptr, 0.0, tmp)
                out[want[t]] = tmp
        return out, init


def thermal_json(R, tm: ThermalMap, checks):
    sd, plant, t_s, kidx, ticks = R["sd"], R["plant"], R["t_s"], R["kidx"], R["ticks"]
    t0 = time.perf_counter()
    rise, init = tm.reconstruct(plant, ticks)
    wall = time.perf_counter() - t0
    N = tm.n
    grid = rise[:, tm.idx_grid].reshape(len(ticks), GRID_Y.size, GRID_X.size)
    t_base = np.array([plant.snaps[int(t)][0].fields["t_base_k"][0, 0] for t in ticks])
    eng_obs = np.array([plant.snaps[int(t)][0].t_obs[0] for t in ticks]) - t_base[:, None]     # (F, O_engine)
    names = tm.eng_names
    pd_e = eng_obs[:, [names.index(f"monitor_pd{k}") for k in range(N)]]
    ring_e = eng_obs[:, [names.index(f"ring_mode_weighted{k}") for k in range(N)]]
    heat_e = eng_obs[:, [names.index(f"heater_resistor{k}") for k in range(N)]]
    die_e = eng_obs[:, names.index("die0")]
    pd_rec = rise[:, [tm.idx_eng[f"monitor_pd{k}"] for k in range(N)]]
    ring_rec = rise[:, [tm.idx_eng[f"ring_mode_weighted{k}"] for k in range(N)]]
    die_rec = rise[:, tm.idx_eng["die0"]]
    pd_probe = rise[:, tm.idx_pd_probe]
    ring_probe = rise[:, tm.idx_ring_probe]
    checks[sd["id"]]["thermal"] = {
        "method": "dynamic: every lag stepped from the reset state with the engine's source powers, read every "
                  f"{SAMPLE} ticks",
        "initial_state": init,
        "reconstruct_wall_s": jnum(wall, 3),
        "pd_engine_observer_reconstructed_minus_engine_k_max_abs": jnum(np.max(np.abs(pd_rec - pd_e))),
        "die0_reconstructed_minus_engine_k_max_abs": jnum(np.max(np.abs(die_rec - die_e))),
        "ring_observer_reconstructed_minus_engine_k_max_abs": jnum(np.max(np.abs(ring_rec - ring_e))),
        "map_point_at_pd_minus_engine_pd_k_max_abs": jnum(np.max(np.abs(pd_probe - pd_e))),
        "map_point_at_ring_centre_minus_engine_ring_k": [jnum(np.min(ring_probe - ring_e)),
                                                         jnum(np.max(ring_probe - ring_e))],
        "engine_ring_rise_k_range": [jnum(np.min(ring_e)), jnum(np.max(ring_e))],
        "engine_pd_rise_k_range": [jnum(np.min(pd_e)), jnum(np.max(pd_e))],
    }
    out = {
        "id": sd["id"], "frames": kidx.tolist(), "t_s": js(t_s[kidx], 6),
        "grid": {"x_um": js(GRID_X, 6), "y_um": js(GRID_Y, 6), "nx": int(GRID_X.size), "ny": int(GRID_Y.size)},
        "rise_k": js(grid, 4, 2),
        "t_base_k": js(t_base, 6, 3),
        "method": ("dynamic: the model's interim thermal law for a point site (the law the engine applies to the "
                   "monitor PDs, the die sensor and every aggressor path): each source s adds a lag with residue "
                   "c(d) Z0 and one pole tau_xt(d) at the node's distance d, c(d) = clip(0.03 x 60 um / d, 0.015, 1). "
                   "Lags stepped by mrm.plant.thermal.thermal_step_b from the reset state with the source powers "
                   "the engine used (heaters, absorbed light, junction and driver heat, aggressor), read every "
                   f"{SAMPLE} us (aggressor: linear in between, as the engine's waveform; the other sources: held at "
                   "the value read at the end of each interval, as the engine holds them per tick). " + init),
        "observers": {
            "ring_rise_k": js(ring_e, 5, 3), "heater_rise_k": js(heat_e, 5, 3), "pd_rise_k": js(pd_e, 5, 3),
            "die0_rise_k": js(die_e, 5, 3),
            "map_at_ring_centre_k": js(ring_probe, 5, 3), "map_at_pd_k": js(pd_probe, 5, 3),
            "note": ("ring_rise_k / heater_rise_k / pd_rise_k / die0_rise_k: the engine's own observers at t_s "
                     "(ring = mode-weighted ring temperature, heater = heater resistor). map_at_*: the map's law at "
                     "the exact ring centre and PD position. The map equals the engine at the PDs (same law); at a "
                     "ring it does not: the ring is its own lumped observer (self path with three poles, light "
                     "heats it 1.76 x more per mW than the heater), which the point law does not contain. Draw the "
                     "rings with ring_rise_k."),
        },
        "display_hint_k": {"p99": jnum(np.percentile(grid, 99)), "max": jnum(grid.max()),
                           "max_at_engine_observers": jnum(max(ring_e.max(), pd_e.max(), heat_e.max())),
                           "note": "the law is clipped at c = 1 below 1.8 um and peaks at point sources (the "
                                   "aggressor site, drivers): clip the colour scale (p99 or the ring maximum)"},
        "units": {"x_um": "um", "y_um": "um", "rise_k": "K", "t_base_k": "K", "t_s": "s"},
    }
    return out


# ============================================================================================ bistability
class SweepController:
    """Python open-loop heater sweep (as tools/perf_scenarios.RampController): ring `ring` follows a piecewise-linear
    code profile, every other ring holds its start code."""
    name = "webviz_open_loop_sweep"
    latency_ticks = 0

    def __init__(self, codes0, ring, t_knots_s, c_knots):
        self.codes0 = np.asarray(codes0, np.int64)
        self.ring = int(ring)
        self.t_k = np.asarray(t_knots_s, float)
        self.c_k = np.asarray(c_knots, float)

    def reset(self, spec):
        self.spec = spec

    def step(self, obs):
        t = obs.tick * 1e-6
        c = self.codes0.copy()
        c[self.ring] = int(round(float(np.interp(t, self.t_k, self.c_k))))
        return {"tx.heater": c[None, :]}


def static_s_curve(pp, sysm, truth, u_now, ring=0, t_below_k=75.0, t_above_k=20.0, n=9501):
    """The static self-heating S-curve of one ring, the other sources held at their present powers: scan the ring
    temperature T, solve the heater power that holds it (DC gains of the thermal system: T_obs = T_base + G u),
    absorbed light and drop from the phase model and exact ring at every line that is on (all orders), and the code
    from the heater driver law with TCR at the heater-resistor temperature. Stable where the heater power rises with
    T (self-heating loop gain below 1)."""
    G = th.dc_gain(sysm)[0]                                                 # (O, S) K/W
    ro, ho = int(sysm.ring_obs[ring]), int(sysm.heater_obs[ring])
    hs, ws = int(sysm.heater_src[ring]), int(sysm.wg_src[ring])
    t_base = float(truth.fields["t_base_k"][0, 0])
    u_rest = np.array(u_now, float)
    u_rest[hs] = 0.0
    u_rest[ws] = 0.0
    rest_r = float(G[ro] @ u_rest)
    rest_h = float(G[ho] @ u_rest)
    phs = ph.PhaseParams.from_plant_params(pp)
    r = phs.table_row(0)
    rp_all = phs.ring[0]
    lam_l = np.asarray(truth.lam_m[0], float)
    p_in = np.asarray(truth.out_rl[0, op.RL_PIN, ring], float)               # (L,) light reaching the ring
    c = truth.conditions[0]
    N = pp.n_rings
    v0, v1, rho = np.asarray(c.v_lo, float), np.asarray(c.v_hi, float), float(c.rho[ring])
    t_lock = float(truth.fields["t_ring_k"][0, ring])
    T = np.linspace(t_lock - t_below_k, t_lock + t_above_k, n)
    hp = el.HeaterDriverParams.from_plant_params(pp)
    out = {k: np.zeros(n) for k in ("p_h", "p_abs", "drop", "det", "code")}
    t_h = T.copy()
    line = int(np.rint(truth.fields["declared_line"][0, ring]))
    for it in range(4):
        for i in range(n):
            o = {k: np.zeros((N, lam_l.size)) for k in ("phi0", "phi1", "a0", "a1", "e0", "e1", "k2t", "k2d", "trt")}
            shift, cj, rs, fl = np.zeros((N, 2)), np.zeros((N, 2)), np.zeros((N, 2)), np.zeros(N, np.int8)
            ph.ring_optics_row(rp_all, lam_l, np.full(N, T[i]), np.full(N, t_h[i]), v0, v1, phs.grid, phs.dneff[r],
                               phs.alpha_fca[r], phs.capacitance[r], phs.r_series[r], o["phi0"], o["phi1"], o["a0"],
                               o["a1"], o["e0"], o["e1"], o["k2t"], o["k2d"], o["trt"], shift, cj, rs, fl)
            k = ring
            r0 = op.ring_exact(o["phi0"][k], o["k2t"][k], o["k2d"][k], o["a0"][k], o["trt"][k])
            r1 = op.ring_exact(o["phi1"][k], o["k2t"][k], o["k2d"][k], o["a1"][k], o["trt"][k])
            p_abs = float(np.sum(p_in * (rho * o["e1"][k] * r1.p_loss + (1 - rho) * o["e0"][k] * r0.p_loss)))
            drop = float(np.sum(p_in * (rho * np.abs(r1.drop) ** 2 + (1 - rho) * np.abs(r0.drop) ** 2)))
            p_h = (T[i] - t_base - G[ro, ws] * p_abs - rest_r) / G[ro, hs]
            t_h[i] = t_base + G[ho, hs] * p_h + G[ho, ws] * p_abs + rest_h
            out["p_h"][i], out["p_abs"][i], out["drop"][i] = p_h, p_abs, drop
            if it == 3:
                m = round(0.5 * (o["phi0"][k, line] + o["phi1"][k, line]) / (2 * math.pi))
                out["det"][i] = K.resonance_detuning(rp_all[k], lam_l[line], 0.5 * (shift[k, 0] + shift[k, 1]),
                                                     float(m))
    out["t_h"] = t_h
    out["T"] = T
    # heater code: the driver law inverted (bisection on the integer codes, then linear in power between them)
    for i in range(n):
        p = out["p_h"][i]
        if p <= 0:
            out["code"][i] = np.nan
            continue
        cc = runner._code_for_power(hp, 0, ring, p, out["t_h"][i])
        if math.isfinite(cc) and cc > hp.code_lo:
            p_hi = runner._heater_drive(hp, 0, ring, cc, out["t_h"][i])[0]
            p_lo = runner._heater_drive(hp, 0, ring, cc - 1, out["t_h"][i])[0]
            cc = cc - 1 + (p - p_lo) / max(p_hi - p_lo, 1e-30)
        out["code"][i] = cc
    return out


def folds_of(p_h, det, fwhm_pm):
    """Fold points of the main loop (the extrema of the heater power over T, where the self-heating loop gain is 1):
    the local maximum nearest blue of the line ends the cold branch (the up-sweep jump), the local minimum nearest
    the line ends the hot branch (the down-sweep fall)."""
    dp = np.diff(p_h)
    ext = np.flatnonzero(np.sign(dp[1:]) * np.sign(dp[:-1]) < 0) + 1
    ups = [i for i in ext if dp[i - 1] > 0 > dp[i] and -12 * fwhm_pm < det[i] * 1e12 < 0]
    dns = [i for i in ext if dp[i - 1] < 0 < dp[i] and -12 * fwhm_pm < det[i] * 1e12 < 0.5 * fwhm_pm]
    if not ups or not dns:
        return None
    return max(ups, key=lambda i: det[i]), max(dns, key=lambda i: det[i])


def _bist_scenario(cfg, ring_line, all_lines):
    sc = sreg.from_config(cfg)
    if all_lines:
        return sc
    off = tuple(l for l in range(int(cfg.get("array.n_rings"))) if l != ring_line)
    return sc + gen.Scenario(name="one_line", components=(gen.const(
        "laser.line_enable", -1.0, elements=off, label="webviz_one_line", provenance="assumed", source="",
        note="webviz bistability: only the swept ring's own line on"),), description="only one line on")


def bistability(log, quick, checks):
    exp = ex.load(str(ROOT / "experiments/examples/steady_pi_lock.yaml"))
    ring = 0
    runs = [(10.0, False), (6.0, False), (2.0, False), (6.0, True)]      # chip input loss (dB), all lines on
    out = {"ring": ring, "line": 0, "powers": [], "units": {
        "code": "code", "p_drop_uw": "uW", "detuning_pm": "pm", "t_ring_k": "K", "monitor_code": "code", "t_ms": "ms",
        "chip_loss_db": "dB", "p_ring_dbm": "dBm", "p_heater_mw": "mW", "p_abs_mw": "mW", "fwhm_pm": "pm"}}
    for loss_db, all_lines in runs:
        cfg = params.load(str(ROOT / "experiments/examples/steady_pi_lock.yaml"),
                          overrides=[f"optical_path.chip_input_loss_db={loss_db}"])
        r2l = cfg.get("channel_plan.ring_to_line", None)
        line = int(np.asarray(r2l)[ring]) if r2l is not None else ring
        sc = _bist_scenario(cfg, line, all_lines)
        # the warm start (every ring at its quasi-static lock code, settled): the state the S-curve holds the other
        # sources in, and the codes the other rings keep during the sweep
        with tapped([SAMPLE]) as tap:                       # Truth is filled from the first tick on
            w = runner.run(cfg, None, sc, duration_s=SAMPLE * 1e-6, batch=1, seeds=exp.seeds, uncertainty="nominal",
                           variation="off", heater_code="quasi_static_reachable", settle=True,
                           reset_protocol="branch_high")
        plant0 = tap.plants[-1]
        truth0, u0 = plant0.snaps[SAMPLE]
        codes0 = np.asarray(w.manifest_inputs["warm_start"]["heater_code"])[0]
        pp = plant0.pp
        lam0 = float(np.asarray(truth0.lam_m[0])[line])
        fwhm_pm = float(lam0 ** 2 * np.asarray(pp.optics.fwhm_hz)[0, ring] / C0 * 1e12)
        t0 = time.perf_counter()
        s = static_s_curve(pp, plant0.sys, truth0, u0, ring=ring, n=2501 if quick else 9501)
        ws = time.perf_counter() - t0
        f = folds_of(s["p_h"], s["det"], fwhm_pm)
        if f is None:
            raise RuntimeError(f"no fold pair found at chip loss {loss_db} dB")
        iu, idn = f
        c_up, c_dn = float(s["code"][iu]), float(s["code"][idn])
        span = c_up - c_dn
        c_lo, c_hi = c_dn - 0.6 * span, c_up + 0.6 * span
        # engine: settled start with ring 0 at c_lo (the cold branch is the only root there), up, hold, down
        t_hold, t_ramp = 2e-3, (10e-3 if quick else 40e-3)
        tk = [0.0, t_hold, t_hold + t_ramp, 2 * t_hold + t_ramp, 2 * t_hold + 2 * t_ramp]
        ck = [c_lo, c_lo, c_hi, c_hi, c_lo]
        start = codes0.copy()
        start[ring] = int(round(c_lo))
        ctrl = SweepController(codes0, ring, tk, ck)
        dec = 20
        rec = [RecordSpec("p_drop_w", dec, "mean"), RecordSpec("heater_code", dec, "mean"),
               RecordSpec("detuning_m", dec, "mean"), RecordSpec("monitor_code", dec, "last"),
               RecordSpec("t_ring_k", dec, "mean")]
        t0 = time.perf_counter()
        res = runner.run(cfg, ctrl, sc, duration_s=tk[-1], batch=1, seeds=exp.seeds, uncertainty="nominal",
                         variation="off", heater_code=start[None, :], settle=True, reset_protocol="branch_high",
                         record=rec)
        we = time.perf_counter() - t0
        tr = res.trace
        t_ms = np.asarray(tr[rec[0].key]["t_s"]) * 1e3
        get = lambda r: np.asarray(tr[r.key]["value"])[:, 0, ring]               # noqa: E731
        drop_e, code_e, det_e, mon_e, tring_e = (get(r) for r in rec)
        up = (t_ms > tk[1] * 1e3) & (t_ms <= tk[2] * 1e3 + 1e-9)
        dn = (t_ms > tk[3] * 1e3) & (t_ms <= tk[4] * 1e3 + 1e-9)
        jumps = []
        for leg, sel in (("up", up), ("down", dn)):
            ii = np.flatnonzero(sel)
            dd = np.diff(det_e[ii])
            j = int(np.argmax(np.abs(dd)))
            jumps.append({"leg": leg, "code": jnum(0.5 * (code_e[ii[j]] + code_e[ii[j + 1]])),
                          "t_ms": jnum(t_ms[ii[j + 1]]), "from_pm": jnum(det_e[ii[j]] * 1e12),
                          "to_pm": jnum(det_e[ii[j + 1]] * 1e12),
                          "p_drop_from_uw": jnum(drop_e[ii[j]] * 1e6), "p_drop_to_uw": jnum(drop_e[ii[j + 1]] * 1e6)})
        # the static curve inside the swept code range, every 5th point (plus the folds)
        keep = np.zeros(s["T"].size, bool)
        keep[::5] = True
        keep[[iu, idn]] = True
        inside = np.isfinite(s["code"]) & (s["code"] >= c_lo - 0.1 * span) & (s["code"] <= c_hi + 0.1 * span)
        lo_i, hi_i = min(iu, idn), max(iu, idn)                  # the contiguous stretch of T around the loop
        while lo_i > 0 and inside[lo_i - 1]:
            lo_i -= 1
        while hi_i < inside.size - 1 and inside[hi_i + 1]:
            hi_i += 1
        seg = np.zeros_like(inside)
        seg[lo_i:hi_i + 1] = True
        sel = keep & seg
        stable = np.r_[np.diff(s["p_h"]) > 0, True]
        p_line = float(truth0.out_rl[0, op.RL_PIN, ring, line])
        mon_fs = int(np.nanmax(mon_e))
        entry = {
            "chip_loss_db": loss_db, "laser_dbm": jnum(cfg.get("laser.power_per_line_dbm")),
            "lines_on": "all" if all_lines else [line], "p_ring_dbm": round(10 * math.log10(p_line * 1e3), 2) + 0.0,
            "fwhm_pm": jnum(fwhm_pm, 4),
            "static": {"code": js(s["code"][sel], 6), "p_drop_uw": js(s["drop"][sel] * 1e6, 5),
                       "detuning_pm": js(s["det"][sel] * 1e12, 5), "t_ring_k": js(s["T"][sel], 6, 3),
                       "p_heater_mw": js(s["p_h"][sel] * 1e3, 5), "p_abs_mw": js(s["p_abs"][sel] * 1e3, 5),
                       "stable": [bool(x) for x in stable[sel]],
                       "folds": [{"kind": "cold branch ends (up-sweep jump)", "code": jnum(c_up),
                                  "detuning_pm": jnum(s["det"][iu] * 1e12), "p_drop_uw": jnum(s["drop"][iu] * 1e6)},
                                 {"kind": "hot branch ends (down-sweep fall)", "code": jnum(c_dn),
                                  "detuning_pm": jnum(s["det"][idn] * 1e12),
                                  "p_drop_uw": jnum(s["drop"][idn] * 1e6)}]},
            "sweep": {"t_knots_ms": js(np.array(tk) * 1e3, 6), "code_knots": js(ck, 6),
                      "up": {"t_ms": js(t_ms[up], 6), "code": js(code_e[up], 6), "p_drop_uw": js(drop_e[up] * 1e6, 5),
                             "detuning_pm": js(det_e[up] * 1e12, 5), "monitor_code": jint(mon_e[up]),
                             "t_ring_k": js(tring_e[up], 6, 3)},
                      "down": {"t_ms": js(t_ms[dn], 6), "code": js(code_e[dn], 6),
                               "p_drop_uw": js(drop_e[dn] * 1e6, 5), "detuning_pm": js(det_e[dn] * 1e12, 5),
                               "monitor_code": jint(mon_e[dn]), "t_ring_k": js(tring_e[dn], 6, 3)},
                      "jumps": jumps, "monitor_code_max": mon_fs},
        }
        out["powers"].append(entry)
        cj = {j["leg"]: j["code"] for j in jumps}
        key = f"ring_{round(10 * math.log10(p_line * 1e3), 1) + 0.0:+.1f}dBm" + ("_all_lines" if all_lines else "")
        checks.setdefault("bistability", {})[key] = {
            "static_fold_codes": {"up_jump": jnum(c_up), "down_fall": jnum(c_dn)},
            "engine_jump_codes": cj,
            "engine_minus_static_codes": {"up": jnum(cj["up"] - c_up), "down": jnum(cj["down"] - c_dn)},
            "engine_jump_detuning_pm": {j["leg"]: [j["from_pm"], j["to_pm"]] for j in jumps},
            "static_fold_detuning_pm": {"up": jnum(s["det"][iu] * 1e12), "down": jnum(s["det"][idn] * 1e12)},
            "loop_width_codes_static": jnum(span), "wall_s": {"static": jnum(ws, 3), "engine": jnum(we, 3)}}
        log(f"  bistability {key}: static folds {c_up:.0f} / {c_dn:.0f}, engine jumps {cj['up']:.0f} / "
            f"{cj['down']:.0f} (static {ws:.1f} s, engine {we:.1f} s)")
    out["note"] = (f"ring {ring} alone is swept open loop while the other rings hold their warm-start codes: 2 ms at "
                   "the lowest code, a linear ramp up, 2 ms at the top, a linear ramp down (40 ms each). Only line 0 "
                   "is on (lines_on), so the loop is the ring's own self-heating; the light at the ring is set by "
                   "the chip input loss (10, 6, 2 dB: the model's 0 to +8 dBm world range); the last entry repeats "
                   "the nominal +4 dBm with all eight lines on. static: the S-curve of the model's static laws (other sources held "
                   "at the start); stable = the heater power rises with the ring temperature (self-heating loop gain "
                   "below 1); the middle branch cannot be held open loop. folds: where the stable branches end. "
                   "p_ring_dbm: light reaching the ring on its line. monitor_code saturates at the ADC full scale.")
    return out


# ============================================================================================ meta
def meta_json(date, results, cap, pp, sysm, specs):
    lay = sysm.meta["layout"]
    ring_xy = np.asarray(lay.ring_xy_um, float)
    rom = cap["rom"]
    r_um = float(np.asarray(pp.optics.radius_m)[0, 0] * 1e6)
    pd_xy = ring_xy + np.asarray(rom.pd_offset_um, float)[None, :]
    drv_xy = ring_xy + np.asarray(rom.driver_offset_um, float)[None, :] if rom.driver_offset_um is not None else None
    y_bus = -(r_um + RIB_W_UM / 2 + GAP_THROUGH_UM + RIB_W_UM / 2)
    y_drop = r_um + RIB_W_UM / 2 + GAP_DROP_UM + RIB_W_UM / 2
    h_out = r_um - RIB_W_UM / 2 - HEATER_GAP_UM
    lam = C0 / np.asarray(pp.laser.line_freq_hz, float)[0]
    r2l = np.asarray(pp.layout.ring_to_line)
    fsr_nm = (lam[r2l] ** 2 * np.asarray(pp.optics.fsr_hz, float)[0] / C0) * 1e9
    fwhm_pm = (lam[r2l] ** 2 * np.asarray(pp.optics.fwhm_hz, float)[0] / C0) * 1e12
    agg = {nm: js(xy, 6) for nm, xy in zip(lay.aggressor_names, np.asarray(lay.aggressor_xy_um, float))}
    ts = {nm: js(xy, 6) for nm, xy in zip(lay.temp_sensor_names, np.asarray(lay.temp_sensor_xy_um, float))}
    g = git_state()
    scen = []
    for R in results:
        sd = R["sd"]
        scen.append({"id": sd["id"], "title": sd["title"], "description": sd["description"],
                     "duration_s": jnum(R["cr"].manifest_inputs["duration_s"]), "frames": int(len(R["t_s"])),
                     "frame_s": jnum(sd["dec"] * R["dt"]), "key_frames": int(len(R["kidx"])),
                     "controller": (R["case"].controller or {}).get("name"), "experiment_file": sd["file"],
                     "files": {"scenario": f"scenario_{sd['id']}.json", "spectra": f"spectra_{sd['id']}.json",
                               "thermal": f"thermal_{sd['id']}.json"}})
    return {
        "preset": "g26", "git_commit": g["commit"], "tracked_files_modified": g["tracked_files_modified"],
        "generated": date, "generator": "webviz/generate.py",
        "n_rings": int(pp.n_rings), "n_lines": int(pp.n_lines),
        "geometry": {
            "units": "um; x along the bus, y across it; ring 0 at the origin",
            "ring_centers_um": js(ring_xy, 6),
            "ring_radius_um": jnum(r_um), "ring_radius_note": "r_ref of the model (R3: the radius that gives the 1.8 "
                                                               "THz FSR with the solver's group index)",
            "rib_width_um": RIB_W_UM,
            "heater_annulus_um": {"inner": jnum(h_out - HEATER_W_UM), "outer": jnum(h_out),
                                  "note": "in-ring doped-silicon heater inside the rib, placed as the model's FEM "
                                          "extraction assumes (heater_gap 2 um, width 1 um; assumed); the plant "
                                          "itself treats each ring's heater as a lumped source at its centre"},
            "bus_waveguide_um": [[float(GRID_X[0]), jnum(y_bus)], [float(GRID_X[-1]), jnum(y_bus)]],
            "drop_waveguides_um": [[[jnum(x + 9.0), jnum(y_drop)], [jnum(x), jnum(y_drop)], [jnum(x), 25.0]]
                                   for x in ring_xy[:, 0]],
            "waveguide_note": ("drawn, not modelled: the plant has no waveguide layout (the bus loss uses the 50 um "
                               "ring spacing); bus and drop centre lines from r_ref, the rib width and the coupling "
                               "gaps of the coupler extraction (183 nm through, 382 nm drop); light enters at the "
                               "left (x < 0) and passes ring 0 first"),
            "monitor_pd_um": js(pd_xy, 6), "driver_um": js(drv_xy, 6) if drv_xy is not None else None,
            "aggressor_sites_um": agg,
            "aggressor_note": "'aggressor' is the default A1/A2 site (30 um from ring 0, 55 um from its PD, the "
                              "same point as ring 0's driver site); 'aggressor_pd_adjacent' is a stress site only",
            "temp_sensors_um": ts, "temp_sensor_note": "die0 sits 1 mm from the bank centroid, outside the maps",
            "positions_provenance": "assumed (layout.sites, ring.thermal.pd_offset_um, driver_offset_um, ring_pitch)",
            "bus_order": np.asarray(pp.layout.bus_order).tolist(),
        },
        "lasers": {"line_nm": js(lam * 1e9, 9), "line_thz": js(np.asarray(pp.laser.line_freq_hz)[0] * 1e-12, 7),
                   "ring_to_line": r2l.tolist(), "grid_ghz": 400.0,
                   "power_per_line_dbm_at_laser": jnum(float(params.load("g26").get("laser.power_per_line_dbm"))),
                   "note": "line 0 is the reddest; ring n is designed for line n (ring_to_line); the cold start's "
                           "supervisor may assign another line (declared_line in the traces)"},
        "rings": {"fsr_nm": js(fsr_nm, 6), "fwhm_pm": js(fwhm_pm, 5),
                  "cold_resonance_nm": js(np.asarray(pp.optics.lambda_res_cold_m)[0] * 1e9, 9),
                  "cold_resonance_note": "resonance at 25 degC, 0 V bias, heater off (the design is 7.6 nm blue of "
                                         "its line)"},
        "scenarios": scen,
        "spectra_grid": {"mode": "per_ring_window", "half_width_nm": HALF_NM, "points": N_SPEC,
                         "step_pm": jnum(2 * HALF_NM / (N_SPEC - 1) * 1e3)},
        "thermal_grid": {"x_um": [float(GRID_X[0]), float(GRID_X[-1])], "nx": int(GRID_X.size),
                         "y_um": [float(GRID_Y[0]), float(GRID_Y[-1])], "ny": int(GRID_Y.size)},
        "files": {"bistability": "bistability.json", "checks": "checks.json"},
        "units": {
            "t_s": "s", "t_ring_k": "K (absolute)", "t_heater_k": "K (absolute)", "t_pd_k": "K (absolute)",
            "detuning_pm": "pm, resonance minus its declared line (positive = red of the line)",
            "lam_res_nm": "nm", "heater_code": "DAC code 0-65535", "p_heater_mw": "mW", "p_abs_mw": "mW",
            "monitor_code": "ADC code 0-65535", "i_monitor_ua": "uA", "p_in_mw": "mW (light at the ring input on its "
                                                                             "declared line)",
            "p_drop_uw": "uW (monitor optical input)", "reloma_db": "dB (OMA / P_in at the ring's through port)",
            "oracle_reloma_db": "dB (best stable relOMA at that instant)", "line_on": "line index (-1: none)",
            "rise_k": "K above t_base_k", "thru_db": "dB", "drop_db": "dB", "aggressor_w": "W",
        },
        "limits": [
            "Every number is the g26 preset's nominal world: many inputs are assumed, inferred or from imec and G25 "
            "devices, not from a measured G26 die (docs/EXPLAINED.md).",
            "The thermal map is the model's lumped point-site law (one pole per source and distance), not a 3-D "
            "field solve; it is exact at the monitor PDs (same law) and under-reads near and inside a ring, whose "
            "temperature is its own lumped observer (ring_rise_k).",
            "The heat dynamics come from imec measurements and assumptions (natural-convection package); G26's own "
            "time constants and crosstalk are unknown.",
            "Spectra use the model's rigid-comb phase model: a heated ring keeps its cold FSR (the locked order is "
            "exact; adjacent orders are off by up to about 33 pm at the lock).",
            "Waveguides, heater annuli and coupling gaps are drawn from the model's extraction inputs; the plant "
            "itself has no layout beyond the site positions.",
        ],
    }


# ============================================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True, help="generation date written into meta.json (YYYY-MM-DD)")
    ap.add_argument("--only", default="", help="comma list of scenario ids and/or 'bistability'")
    ap.add_argument("--quick", action="store_true", help="short runs into runs/webviz_quick/ (smoke test)")
    ap.add_argument("--out", default=None, help="output directory (default webviz/site/data)")
    a = ap.parse_args(argv)
    out_dir = Path(a.out) if a.out else (ROOT / "runs" / "webviz_quick" if a.quick else SITE_DATA)
    only = {s for s in a.only.split(",") if s}
    T0 = time.perf_counter()
    log = lambda s: print(s, flush=True)                               # noqa: E731
    warnings.simplefilter("ignore")
    defs, _ = scenario_defs(a.quick)
    checks: dict = {"generated": a.date, "git": git_state()}
    chk_path = out_dir / "checks.json"
    if only and chk_path.exists():
        old = json.loads(chk_path.read_text())
        checks = {**old, "generated": a.date, "git": git_state()}
    sizes = {}
    results = []
    cap, pp_nom = capture_thermal_inputs(ex.load(str(ROOT / defs[0]["file"])).cfg)
    tmap = None
    spec = None
    for sd in defs:
        if only and sd["id"] not in only:
            continue
        log(f"scenario {sd['id']}")
        R = run_scenario(sd, log)
        results.append(R)
        c = checks[sd["id"]] = {"taps_bit_identical": R["tap_identical"], "taps_differing_metrics": R["tap_bad"]}
        if sd["id"] in ("steady", "a1_aggressor", "cold_start"):
            om = official_metrics(sd)
            ok, bad = same_metrics(om, R["plain"].metrics)
            c["metrics_equal_mrm_run_of_the_file"] = {"equal": bool(ok), "differing": bad}
            log(f"  metrics equal to mrm.run(<file>): {ok}")
        cs = R["case"]
        if (cs.uncertainty, cs.variation) != ("nominal", "off"):
            raise RuntimeError(f"{sd['id']}: the site shows the nominal world, got {cs.uncertainty}/{cs.variation}")
        cap_s, _ = capture_thermal_inputs(R["exp"].cfg, cs.uncertainty, cs.variation, R["exp"].seeds)
        if tmap is None:
            tmap = ThermalMap(cap_s, R["plant"].sys, checks)
            log(f"  thermal system with {GRID_X.size * GRID_Y.size} grid observers: {tmap.sys.n_lags} lags "
                f"({tmap.wall_assemble:.1f} s)")
        else:
            s1, s0 = R["plant"].sys, tmap.sys_engine
            same = (s1.obs_names == s0.obs_names and s1.src_names == s0.src_names
                    and np.array_equal(s1.lag_src, s0.lag_src) and np.array_equal(s1.taus_s, s0.taus_s)
                    and np.array_equal(s1.res_k_per_w, s0.res_k_per_w))
            if not same:
                raise RuntimeError("scenario thermal system differs from the first one")
        c["thermal_system_same_as_first_scenario"] = True
        if spec is None:
            spec = Spectra(R["plant"].pp)
        sj = scenario_json(R)
        c["arrays_reproduce_metrics"] = arrays_check(sj, R)
        sizes[f"scenario_{sd['id']}.json"] = write_json(out_dir / f"scenario_{sd['id']}.json", sj)
        sizes[f"spectra_{sd['id']}.json"] = write_json(out_dir / f"spectra_{sd['id']}.json",
                                                       spectra_json(R, spec, checks))
        sizes[f"thermal_{sd['id']}.json"] = write_json(out_dir / f"thermal_{sd['id']}.json",
                                                       thermal_json(R, tmap, checks))
        log(f"  wrote {sd['id']}: " + ", ".join(f"{k} {v / 1e6:.2f} MB" for k, v in sizes.items()
                                                if sd["id"] in k))
        R["plant"].snaps = {}                                     # free the snapshots
    if not only or "bistability" in only:
        log("bistability")
        sizes["bistability.json"] = write_json(out_dir / "bistability.json", bistability(log, a.quick, checks))
    if results:
        sys0 = results[0]["plant"].sys
        meta_res = results
        if only and (out_dir / "meta.json").exists():
            prev = json.loads((out_dir / "meta.json").read_text())
            keep = [s for s in prev.get("scenarios", []) if s["id"] not in {R["sd"]["id"] for R in results}]
        else:
            keep = []
        meta = meta_json(a.date, meta_res, cap, results[0]["plant"].pp, sys0, spec)
        order = [d["id"] for d in defs]
        meta["scenarios"] = sorted(keep + meta["scenarios"], key=lambda s: order.index(s["id"]))
        sizes["meta.json"] = write_json(out_dir / "meta.json", meta)
    checks["sizes_mb"] = {q.name: round(q.stat().st_size / 1e6, 3) for q in sorted(out_dir.glob("*.json"))
                          if q.name != "checks.json"}
    checks["total_wall_s"] = round(time.perf_counter() - T0, 1)
    write_json(out_dir / "checks.json", checks)
    total = sum(p.stat().st_size for p in out_dir.glob("*.json"))
    log(f"done in {time.perf_counter() - T0:.0f} s; {out_dir}: {total / 1e6:.1f} MB of JSON")


def arrays_check(sj, R):
    """Do the exported (rounded) arrays reproduce the run's summary metrics? Uses only what the site reads."""
    sd = R["sd"]
    m = R["cr"].metrics
    t = np.array(sj["t_s"], float)
    fs = float(sj["frame_s"])
    sf = float(sj["summary"]["score_from_s"] or 0.0)
    sel = (t - fs) >= sf - 1e-9                                       # frames entirely after the scoring start
    d_ref = np.array(sj["lock_reference"]["d_ref_pm"], float)
    lo = np.array(sj["rings"]["detuning_min_pm"], float)[sel] - d_ref
    hi = np.array(sj["rings"]["detuning_max_pm"], float)[sel] - d_ref
    mx = np.maximum(np.abs(lo), np.abs(hi)).max(axis=0)
    rel = 10 ** (np.array(sj["rings"]["reloma_db"], float)[sel] / 10)
    orc = 10 ** (np.array(sj["rings"]["oracle_reloma_db"], float)[sel] / 10)
    pen = 10 * np.log10(orc.sum(axis=0) / rel.sum(axis=0))
    mean = np.array(sj["rings"]["detuning_pm"], float)[sel] - d_ref
    rms = np.sqrt((mean ** 2).mean(axis=0))
    m_mx = np.asarray(m["max_abs_lock_error_pm"], float)[0]
    m_pen = np.asarray(m["oma_penalty_db"], float)[0]
    m_rms = np.asarray(m["rms_lock_error_pm"], float)[0]
    covered = (t[sel][-1] - (t[sel][0] - fs)) if np.any(sel) else 0.0
    return {
        "frames_used": int(np.sum(sel)), "scored_time_covered_s": jnum(covered),
        "scored_time_s": jnum(float(R["cr"].manifest_inputs["duration_s"]) - sf),
        "max_abs_lock_error_pm_from_envelopes_minus_metric": js(mx - m_mx, 3),
        "oma_penalty_db_from_frame_means_minus_metric": js(pen - m_pen, 3),
        "rms_lock_error_pm_from_frame_means_vs_metric": {"frames": js(rms, 4), "metric": js(m_rms, 4)},
    }


if __name__ == "__main__":
    main()
