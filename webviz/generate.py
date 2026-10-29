#!/usr/bin/env python
"""Regenerate every data file of the webviz site (webviz/site/data/) from the mrm plant model, g26 preset.

    cd ~/mrm-model
    .venv/bin/python tools/memguard.py -- .venv/bin/python webviz/generate.py --date 2026-10-17

Options:
  --only steady,a1_aggressor,laser_step,cold_start,bistability   regenerate a subset. meta.json and checks.json keep
                                                                  the entries of the files that are not regenerated.
  --quick                                                         short runs for a smoke test, written to
                                                                  runs/webviz_quick/ instead of the site.
  --out DIR                                                       another output directory.

Read this first. main() runs the four scenarios of scenario_defs() one after the other. For each scenario,
run_scenario() runs the experiment's main case twice: once plain and once through two read-only taps (_TapPlant, a
Plant subclass). checks.json records whether the two runs give bit-identical metrics and traces (taps_bit_identical).
For steady, a1_aggressor and cold_start, main() also runs the file through experiment.run (the code path of
`mrm run <file>`) and compares its metrics with the plain run. Five writers then turn the tapped run into the JSON
files of that scenario: scenario_json, spectra_json, thermal_json, devices_json and derived.stats_json. After the
scenarios, bistability() writes bistability.json (the heater sweeps and the static S-curve of ring 0), meta_json()
writes meta.json (shared by every scenario), and checks.json is written last. The JSON helpers js, jint and jnum
(section "JSON helpers") round every number before it is written.

Everything written here is model output, never a sketch:
  scenario_<id>.json   Engine runs through the case pipeline of mrm.sim.experiment (_plan and _execute, the code path
                       of `mrm run <file>`). So the summary metrics are those of the experiment files. The per-frame
                       traces come from the engine's RecordSpec records.
  spectra_<id>.json    The model's phase model (mrm.plant.phase.ring_optics_row) and exact ring
                       (mrm.plant.optics.ring_exact), evaluated at the state of each ring at a key frame (Truth: ring
                       and heater-resistor temperature, the two NRZ junction levels, ones density).
  thermal_<id>.json    The model's thermal system with one extra point observer per grid node. It is built by
                       mrm.plant.thermal.assemble with the InterimRom and ThermalScales that mrm.params.realize built
                       for the run. The model's own kernel (thermal_step_b) steps it with the source powers that the
                       engine used: its th_u1 row, read every 10 ticks. The aggressor power is linear between two
                       reads; the other sources are held. The map is dynamic, not quasi-static.
  devices_<id>.json    The kernel state of world 0 (heater DAC and driver, monitor PD, laser lines) read by the tap at
                       every controller step, and the running metrics at every frame end.
  stats_<id>.json      Statistics, time in each state and the limits reached, from 10 us samples (derived.py).
  bistability.json     Slow open-loop heater sweeps of ring 0 on the engine, and the static S-curve of the same ring
                       from the model's static laws (phase model, exact ring, DC thermal gains, heater driver law).
  checks.json          The validation numbers (webviz/README.md, "How the data was checked").
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import subprocess
import time
import types
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SITE_DATA = ROOT / "webviz" / "site" / "data"

from mrm import params  # noqa: E402
from mrm.plant import electrical  # noqa: E402
from mrm.plant import optics  # noqa: E402
from mrm.plant import phase  # noqa: E402
from mrm.plant import sensing  # noqa: E402
from mrm.plant import thermal  # noqa: E402
from mrm.plant.plant import Plant  # noqa: E402
from mrm.scenarios import generators  # noqa: E402
from mrm.scenarios import registry as scenario_registry  # noqa: E402
from mrm.sim import experiment  # noqa: E402
from mrm.sim import kernel  # noqa: E402
from mrm.sim import metrics  # noqa: E402
from mrm.sim import runner  # noqa: E402
from mrm.sim.metrics import RecordSpec  # noqa: E402
from mrm.units import C0  # noqa: E402

import derived  # noqa: E402  (webviz/derived.py: statistics, limits, crosstalk, scenario inputs)

# ------------------------------------------------------------------------------------------------------------- settings
#: the tap reads the engine every SAMPLE ticks: 10 ticks of 1 us = 10 us, one controller period of g26
SAMPLE = derived.SAMPLE
#: spectra: a window of +-SPECTRUM_HALF_NM (nm) around each ring's centre, SPECTRUM_POINTS points (12.5 pm step)
SPECTRUM_HALF_NM = 1.5
SPECTRUM_POINTS = 241
#: thermal map grid (um): every ring centre, PD, driver and aggressor site sits 1.0 um (x) and 1.25 um (y) from
#: its four nearest nodes, so all of them are drawn alike
GRID_X = -61.0 + 4.0 * np.arange(120)          # -61 .. 415 um
GRID_Y = -61.25 + 2.5 * np.arange(50)          # -61.25 .. 61.25 um
#: display geometry that the plant does not model (the model has no waveguide layout): rib width, coupling gaps and
#: in-ring heater placement. All come from the model's own extraction inputs (assumed values there).
RIB_W_UM = 0.400                                # ring.geometry.width_nm
GAP_THROUGH_UM = 0.1831                         # coupling_rib160.json through.gap_nm (edge to edge)
GAP_DROP_UM = 0.3819                            # coupling_rib160.json drop.gap_nm
HEATER_GAP_UM = 2.0                             # mrm.extraction.thermal_fem.ThermalConfig.heater_gap_um (assumed)
HEATER_W_UM = 1.0                               # ThermalConfig.heater_width_um (assumed)

#: the per-ring trace records of every scenario: (Truth field, reducer over the frame, export name, scale, unit).
#: scale None: the value is converted to dB instead (relOMA).
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
    # for the device panels: code envelopes, the data path and the junction
    ("heater_code", "min", "heater_code_min", 1.0, "code"),
    ("heater_code", "max", "heater_code_max", 1.0, "code"),
    ("monitor_code", "min", "monitor_code_min", 1.0, "code"),
    ("monitor_code", "max", "monitor_code_max", 1.0, "code"),
    ("ber", "mean", "ber", 1.0, "1"),
    ("oma_w", "mean", "oma_uw", 1e6, "uW"),
    ("bias_v", "mean", "bias_v", 1.0, "V"),
    ("er_db", "mean", "er_db", 1.0, "dB"),
    ("il_db", "mean", "il_db", 1.0, "dB"),
)
#: export names that are integer codes or indices (written as integers)
INT_FIELDS = {"heater_code", "monitor_code", "line_on", "declared_line", "order", "heater_code_min",
              "heater_code_max", "monitor_code_min", "monitor_code_max"}
#: kernel state that the device tap reads every SAMPLE ticks (world 0): rows of the KState arrays ring_int,
#: ring_float and line_float, named as in kernel.RI, kernel.RF and kernel.LF
DEVICE_RING_INT_ROWS = ("he_cmd", "he_reg", "he_code", "pga", "mo_count")
DEVICE_RING_FLOAT_ROWS = ("he_v", "he_i", "he_r", "he_p", "i_dark", "rho")
DEVICE_LINE_FLOAT_ROWS = ("f", "p", "pchip")
#: running metrics exported per frame (mrm.sim.metrics.summarize of the accumulators at each frame end)
RUNNING_METRICS = ("max_abs_lock_error_pm", "rms_lock_error_pm", "oma_penalty_db", "mean_reloma_db",
                   "oracle_reloma_db", "excursion_count", "excursion_max_s", "time_unlocked_s", "wrong_line_s",
                   "wrong_line_events", "fsr_hop_events", "order_changes", "heater_energy_j", "heater_sat_s",
                   "heater_low_rail_s", "ber_mean", "expected_errors", "peak_slew_codes_per_s", "code_min_used",
                   "code_max_used", "time_to_lock_s", "first_lock_s")


def scenario_defs(quick: bool):
    """The four scenarios of the site, and the A1 aggressor period (ms).

    Each scenario is a dict: id, experiment file, overrides, title, description, and
      dec      ticks per frame (the RecordSpec decimation of the traces)
      key      key-frame rule [(t0_ms, t1_ms, every_n_frames), ...]: inside [t0, t1], every n-th frame is a key frame
      step_s   laser_step only: time of the +10 % laser power step (s)
    quick: shorter runs for a smoke test."""
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


# --------------------------------------------------------------------------------------------------------- JSON helpers
# The writers use three short helpers on almost every line:
#   js(a, sig, dec)   array -> nested list, rounded to sig significant digits (and dec decimals), NaN -> null
#   jint(a)           array -> nested list of integers, NaN and inf -> null
#   jnum(v, sig)      one number -> float rounded to sig digits, a bool stays a bool; None, NaN and inf -> null
def round_sig(a, sig: int = 4, dec: int | None = None) -> np.ndarray:
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


def _nan_to_null(o):
    """Nested lists with every non-finite float replaced by None (JSON null)."""
    if isinstance(o, list):
        return [_nan_to_null(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def js(a, sig: int = 4, dec: int | None = None):
    """JSON-ready nested lists, rounded, NaN -> null."""
    return _nan_to_null(round_sig(a, sig, dec).tolist())


def jint(a):
    """JSON-ready nested lists of integers (rounded to the nearest integer), NaN and inf -> null.

    When the array holds a NaN or inf, the other values stay floats (12.0, not 12)."""
    x = np.asarray(a, dtype=float)
    return _nan_to_null(np.where(np.isfinite(x), np.rint(x), np.nan).tolist()) if np.any(~np.isfinite(x)) else \
        np.rint(x).astype(np.int64).tolist()


def jnum(v, sig=6):
    """One JSON-ready number: a bool stays a bool, a finite value is rounded to `sig` digits, else None."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    v = float(v)
    return float(round_sig(v, sig)) if math.isfinite(v) else None


def write_json(path: Path, obj) -> int:
    """Write obj as compact JSON (no NaN allowed); return the file size in bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    txt = json.dumps(obj, separators=(",", ":"), allow_nan=False)
    path.write_text(txt)
    return len(txt)


def to_db(x):
    """10 log10(x) of a linear power ratio (0 -> -inf, no warning)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(np.asarray(x, float))


def git_state():
    """The short commit hash of the repository and whether tracked files are modified."""
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "--short", "HEAD"),
            "tracked_files_modified": bool(git("status", "--porcelain", "--untracked-files=no"))}


# ---------------------------------------------------------------------------------------------------------- engine taps
class _Tap:
    """What the tapped plants collect (one entry per Plant the runner builds: the self-reference pre-run first)."""

    def __init__(self, snap_ticks, dec=None):
        self.snap_ticks = set(int(t) for t in snap_ticks)
        self.dec = None if dec is None else int(dec)        # frame length: metric snapshots at every frame end
        self.plants: list = []


class _TapPlant(Plant):
    """The Plant of runner.run with two read-only taps. Both are active after reset.

    Tap 1: every SAMPLE ticks, the thermal source powers (th_u1, the input of the last interval), a device row and
    the crosstalk lag split (xt_sum, xt_n); at every frame end also the metric accumulators (msnaps).
    Tap 2: at the snapshot ticks of the tap, Truth and th_u1.
    On the compiled and open-loop paths a kernel block is cut into SAMPLE-tick blocks. Every cut is a controller
    edge (g26: period 10, phase 0). Under the fixed substep policy a block end changes nothing (docs/ENGINE.md,
    "Fixed and automatic substeps"). run_scenario compares the metrics and traces with a plain run, and checks.json
    records the result (taps_bit_identical). The Python path stops at every controller edge already."""
    tap: _Tap | None = None

    def reset(self, *a, **kw):
        self._armed = False
        out = super().reset(*a, **kw)
        if int(self.P.policy) != kernel.POLICY_FIXED:
            raise RuntimeError("the taps need the fixed substep policy (block ends would change an auto run)")
        if int(self.P.jitter) != 0 or int(self.P.period) % SAMPLE or int(self.P.phase) % SAMPLE:
            raise RuntimeError("the taps need controller edges on multiples of SAMPLE ticks")
        self.u_ticks = [int(self.tick)]
        self.u_rows = [self.S.th_u1[0].copy()]
        self.x_reset = self.S.th_x[0].copy()
        self.snaps = {}
        self.dev = []                                    # device rows, one per u_ticks entry
        self.msnaps = {}                                 # tick -> metric accumulators (frame ends)
        self._xt = derived.xt_index(self.sys)            # lag-state split of the ring and PD observers by source
        self.xt_sum, self.xt_n = {}, {}                  # frame -> sum of the splits at its controller edges, count
        self.dev.append(self._device_row())
        if int(self.tick) in self.tap.snap_ticks:
            self.snaps[int(self.tick)] = (self.truth(), self.S.th_u1[0].copy())
        self.tap.plants.append(self)
        self._armed = True
        return out

    def _device_row(self):
        """Read-only copy of the device state of world 0 (heater DAC path and driver law outputs, monitor PD, laser
        lines, clip counters): the state at the end of the last plant interval."""
        S = self.S
        row = {n: S.ring_int[0, kernel.RI[n]].copy() for n in DEVICE_RING_INT_ROWS}
        row.update({n: S.ring_float[0, kernel.RF[n]].copy() for n in DEVICE_RING_FLOAT_ROWS})
        row.update({"lf_" + n: S.line_float[0, kernel.LF[n]].copy() for n in DEVICE_LINE_FLOAT_ROWS})
        row["mo_over"] = S.ring_bool[0, kernel.RB["mo_over"]].copy()
        row["he_scale"] = float(S.world_float[0, kernel.WF["he_scale"]])
        o, n = _heater_field(self)
        sat = S.sat[0, o:o + n].astype(np.int64)
        vc = getattr(self.validator, "counts", {}).get("tx.heater")
        if vc is not None:
            sat = sat + np.asarray(vc)[0].astype(np.int64)
        row["sat"] = sat
        k, dec = int(self.tick), self.tap.dec
        if dec and k > 0:                                # crosstalk: lag states by heat source, summed per frame
            lag, idx, shp = self._xt
            v = np.bincount(idx, weights=S.th_x[0][lag], minlength=shp[0] * shp[1]).reshape(shp)
            f = (k - 1) // dec
            if f in self.xt_sum:
                self.xt_sum[f] += v
                self.xt_n[f] += 1
            else:
                self.xt_sum[f], self.xt_n[f] = v, 1
        return row

    def _capture(self):
        """Record what the taps read at the present tick (called after every block that run() hands to Plant.run)."""
        k = int(self.tick)
        if k != self.u_ticks[-1]:
            self.u_ticks.append(k)
            self.u_rows.append(self.S.th_u1[0].copy())
            self.dev.append(self._device_row())
            dec = self.tap.dec
            if dec and k % dec == 0 and k not in self.msnaps:
                S = self.S
                self.msnaps[k] = (S.m_af[0].copy(), S.m_ai[0].copy(), S.m_runhist[0].copy(),
                                  None if S.m_ref is None else S.m_ref[0].copy())
        if k in self.tap.snap_ticks and k not in self.snaps:
            self.snaps[k] = (self.truth(), self.S.th_u1[0].copy())

    def run(self, n_ticks, *, mode=kernel.MODE_OPEN, step_fn=None, fast_fn=None, resume=False, n_threads=None):
        """Plant.run, cut into blocks that end on multiples of SAMPLE ticks once the taps are armed."""
        if not getattr(self, "_armed", False):
            return super().run(n_ticks, mode=mode, step_fn=step_fn, fast_fn=fast_fn, resume=resume,
                               n_threads=n_threads)
        if mode == kernel.MODE_PYTHON:
            d = super().run(n_ticks, mode=mode, step_fn=step_fn, fast_fn=fast_fn, resume=resume, n_threads=n_threads)
            self._capture()
            return d
        end = self.tick + int(n_ticks)
        done = 0
        while self.tick < end:
            nxt = min(end, (self.tick // SAMPLE + 1) * SAMPLE)
            done += super().run(nxt - self.tick, mode=mode, step_fn=step_fn, fast_fn=fast_fn, resume=resume,
                                n_threads=n_threads)
            resume = False
            self._capture()
        return done


@contextlib.contextmanager
def tapped(snap_ticks, dec=None):
    """Inside the block, runner.run builds _TapPlant instead of Plant; yields the _Tap that collects the plants.

    snap_ticks: ticks at which Truth is copied (the key frames). dec: frame length in ticks (metric snapshots at
    every frame end), or None."""
    tap = _Tap(snap_ticks, dec)
    old = runner.Plant
    _TapPlant.tap = tap
    runner.Plant = _TapPlant
    try:
        yield tap
    finally:
        runner.Plant = old
        _TapPlant.tap = None


def flat_metrics(m, pre=""):
    """The numeric arrays of a nested metrics dict, keyed by dotted path ("settings.x" for m["settings"]["x"])."""
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
    """(True if every numeric metric of m0 is bit-identical in m1, the list of metrics that are not)."""
    f0, f1 = flat_metrics(m0), flat_metrics(m1)
    bad = [k for k in f0 if k not in f1 or not np.array_equal(f0[k], f1[k], equal_nan=True)]
    return not bad, bad


# ------------------------------------------------------------------------------------------------------------ scenarios
def load_case(sd):
    """(experiment, its main case, the scenario to run): the scenario is None (use the file's), except for the
    laser step, which adds a +10 % power step on every line to the file's scenario."""
    exp = experiment.load(str(ROOT / sd["file"]), overrides=list(sd["overrides"]) or None)
    case = experiment._plan(exp)[0]
    sc = None
    if sd["id"] == "laser_step":
        base = scenario_registry.from_config(exp.cfg)
        step = generators.Scenario(name="laser_step", components=(generators.step(
            "laser.power_delta_db", sd["step_s"], 10.0 * math.log10(1.1), label="laser_step_10pct",
            provenance="assumed", source="", note="webviz: +10 % on every laser line (as tools/perf_scenarios.py)"),),
            description=f"+10 % laser power on every line at {sd['step_s']} s")
        sc = base + step
    return exp, case, sc


def records_for(dec):
    """The RecordSpec list of the TRACE records, every `dec` ticks."""
    return [RecordSpec(f, dec, red) for f, red, *_ in TRACE]


def key_frames(t_s, rules):
    """Indices of the key frames: the first and last frame, and for each rule (t0_ms, t1_ms, every) the first frame
    inside [t0, t1] and every `every`-th frame of that window, counting the first frame as number 1 (offsets
    every-1, 2*every-1, ... from it). t_s: frame end times (s)."""
    t_ms = np.asarray(t_s) * 1e3
    idx = {0, len(t_ms) - 1}
    for t0, t1, every in rules:
        sel = np.flatnonzero((t_ms >= t0 - 1e-9) & (t_ms <= t1 + 1e-9))
        if sel.size:
            idx.update(sel[(sel - sel[0]) % every == every - 1].tolist())
            idx.add(int(sel[0]))
    return np.array(sorted(idx), dtype=np.int64)


def run_scenario(sd, log):
    """Two runs of the experiment's main case: without taps (frame times, reference metrics) and with them.

    Returns the run record R that every writer reads:
      sd, exp, case    the scenario definition, the loaded experiment and its main case (with the site's records)
      plain, cr        the case results of the plain and of the tapped run
      plant            the tapped Plant of the main run (device rows, snapshots, thermal inputs)
      t_s, dt          frame end times (s) and the plant tick (s)
      kidx, ticks      key-frame indices and their ticks
      tap_identical    True when the taps changed no metric and no trace (tap_bad: the metrics that differ)"""
    exp, case, sc = load_case(sd)
    dt = float(exp.cfg.get("sim.dt_plant_s"))
    case = dataclasses.replace(case, record=records_for(sd["dec"]) + derived.stats_records(), record_default=False,
                               scenario=sc)
    t0 = time.perf_counter()
    plain = experiment._execute(exp, case)
    w0 = time.perf_counter() - t0
    tkey = next(iter(plain.trace))
    t_s = np.asarray(plain.trace[tkey]["t_s"])
    kidx = key_frames(t_s, sd["key"])
    ticks = np.rint(t_s[kidx] / dt).astype(np.int64)
    t0 = time.perf_counter()
    with tapped(ticks, sd["dec"]) as tap:
        cr = experiment._execute(exp, case)
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
    res = experiment.run(str(ROOT / sd["file"]), overrides=list(sd["overrides"]) or None, write=False,
                         log=lambda s: None)
    return res.cases["main"].metrics


# ------------------------------------------------------------------------------------------------------ scenario arrays
def scenario_json(R):
    """scenario_<id>.json: the per-frame traces of every ring, the lock reference, the scenario drives, the events
    and the run's summary metrics."""
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
            arr = js(to_db(v), 4, 3)
        elif name == "ber":
            arr = js(v, 4)                            # no decimal rounding: BER spans many decades
        elif name in ("er_db", "il_db"):
            arr = js(v, 4, 3)
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
            tgt = [x for x in scenario_of(R).aggressors if x.site == "aggressor"]
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
    out["inputs"] = derived.inputs_json(R, S)
    out["controller_timing"] = derived.controller_timing(R)
    out["units"].update({"aggressor_w": "W", "laser_power_factor": "1", "t_s": "s", "frame_s": "s",
                         "d_ref_pm": "pm"})
    # summary metrics (the run's own, after score_from_s)
    per_ring = {}
    for k in ("max_abs_lock_error_pm", "rms_lock_error_pm", "oma_penalty_db", "mean_reloma_db", "oracle_reloma_db",
              "max_abs_detuning_pm", "rms_detuning_pm", "excursion_count", "excursion_max_s", "time_unlocked_s",
              "first_lock_s", "time_to_lock_s", "wrong_line_s", "heater_low_rail_s", "heater_sat_s",
              "code_min_used", "code_max_used", "monitor_error_pct_min", "monitor_error_pct_max",
              "wrong_line_events", "fsr_hop_events", "order_changes", "heater_energy_j", "peak_slew_codes_per_s",
              "ber_mean", "max_window_ber", "expected_errors", "bits", "ber_windows", "ber_windows_over_limit",
              "mean_p_abs_w", "oma_penalty_unrestricted_db", "oracle_reloma_unrestricted_db",
              "heater_range_used_codes", "mean_detuning_pm"):
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
        "aggressor_calibration": _aggressor_calibration_json(mi.get("aggressor_calibration")),
        "ber_limit": jnum(settings.get("ber_limit")), "ber_sample_period_s": jnum(settings.get("ber_sample_period_s")),
        "wrong_line_window_fwhm": jnum(settings.get("wrong_line_window_fwhm")),
        "controller_latency_ticks": mi.get("controller_latency_ticks"),
        "start": mi.get("start"), "warm_start": mi.get("start") == "warm",
        "saturation_counts": _sat_counts(cr, R["plant"]),
        "warnings": list(cr.warnings),
        "note": "metrics are the engine's own accumulators (every 1 us tick after score_from_s), as `mrm run` "
                "reports them",
    }
    return out


def _heater_field(plant):
    """(offset, size) of the tx.heater channel in the plant's action vector."""
    f = plant.spec.layout.act_field("tx.heater")
    return int(f.offset), int(f.size)


def _sat_counts(cr, plant):
    """Clipped heater commands per ring over the whole run (start-up included): the engine's own counters
    (Plant.diagnostics saturation_counts; on the Python controller path the ActionValidator counts, which clip
    before the kernel sees the command)."""
    diag = cr.manifest_inputs.get("diagnostics") or {}
    o, n = _heater_field(plant)
    out = {"note": "number of controller steps whose heater command was outside [code_min, code_max] and was "
                   "clipped, whole run (start-up included)"}
    sat = diag.get("saturation_counts")
    tot = np.zeros(n)
    if sat is not None:
        tot = tot + np.asarray(sat)[0, o:o + n]
    py = diag.get("saturation_counts_python")
    if py and "tx.heater" in py:
        tot = tot + np.asarray(py["tx.heater"])[0]
        out["source"] = "ActionValidator (Python controller path) + kernel clip counter"
    else:
        out["source"] = "kernel clip counter (S.sat, Plant.diagnostics)"
    out["tx_heater"] = [int(x) for x in tot]
    return out


def scenario_of(R):
    """The scenario object the run used (the case's own, else the config's)."""
    return R["case"].scenario if R["case"].scenario is not None else scenario_registry.from_config(R["exp"].cfg)


def _aggressor_calibration_json(cal):
    """The run's aggressor calibration record per site, with only its finite numbers and its strings."""
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


def _site_distances(R):
    """Ring centre to each aggressor site (um), from the run's thermal layout (as meta.geometry.distances)."""
    lay = R["plant"].sys.meta["layout"]
    ring_xy = np.asarray(lay.ring_xy_um, float)
    return {"ring_to_site_um": {nm: np.hypot(*(ring_xy - np.asarray(xy, float)[None, :]).T).tolist()
                                for nm, xy in zip(lay.aggressor_names, np.asarray(lay.aggressor_xy_um, float))}}


def _a1_period(R):
    """Period of the A1 aggressor's rectified-sine target (s), or None."""
    tg = [x for x in getattr(scenario_of(R), "aggressors", ()) if x.site == "aggressor"]
    if tg and tg[0].target is not None and tg[0].target.form == "rectified_sine":
        return float(tg[0].target.period_s)
    return None


def arrays_check(sj, R):
    """Do the exported (rounded) arrays reproduce the run's summary metrics? Uses only what the site reads.

    sj: the scenario_<id>.json dict. Returns the differences for checks.json (README, "How the data was checked",
    item 4)."""
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


# -------------------------------------------------------------------------------------------------------------- spectra
class Spectra:
    """Per-ring through and drop transmission from the model's phase model and exact ring at a Truth state.

    The phase model (mrm.plant.phase.ring_optics_row) gives, for each ring and wavelength: the round-trip phase
    phi0 / phi1 (rad) and amplitude a0 / a1 of NRZ level 0 / 1, the fraction e0 / e1 of the ring loss that is
    absorbed (heat), the power couplings k2t (through) and k2d (drop) and the round-trip time trt (s). The exact ring
    (mrm.plant.optics.ring_exact) turns them into the through, drop and loss of each level."""

    def __init__(self, pp):
        self.pp = pp
        self.phs = phase.PhaseParams.from_plant_params(pp)
        if int(pp.flags.optics_model) != 0:
            raise RuntimeError("the spectra assume the exact-ring optics model (g26 default)")
        rb = np.asarray(pp.optics.backscatter_rb)
        if np.any(rb != 0):
            raise RuntimeError("the spectra assume no backscatter (g26 nominal)")
        self.offsets_m = np.linspace(-SPECTRUM_HALF_NM, SPECTRUM_HALF_NM, SPECTRUM_POINTS) * 1e-9

    def levels(self, truth):
        """(ring temperature K, heater-resistor temperature K, level-0 voltage V, level-1 voltage V, ones density),
        each (N,), of world 0 at a Truth state."""
        f = truth.fields
        c = truth.conditions[0]
        return (np.ascontiguousarray(f["t_ring_k"][0]), np.ascontiguousarray(f["t_heater_k"][0]),
                np.ascontiguousarray(c.v_lo), np.ascontiguousarray(c.v_hi), np.asarray(c.rho, float))

    def phase_rows(self, truth, lam, t_ring=None):
        """Phase-model rows of every ring at the wavelengths lam (W,) in m: dict of (N, W) arrays.

        t_ring (N,) in K replaces the ring temperatures of the Truth state (for the self-heating slope)."""
        t_ring0, t_heat, v0, v1, _ = self.levels(truth)
        t_ring = t_ring0 if t_ring is None else np.ascontiguousarray(t_ring, float)
        phs, r = self.phs, self.phs.table_row(0)
        N, W = self.pp.n_rings, lam.size
        o = {k: np.zeros((N, W)) for k in ("phi0", "phi1", "a0", "a1", "e0", "e1", "k2t", "k2d", "trt")}
        shift, cj, rs, fl = np.zeros((N, 2)), np.zeros((N, 2)), np.zeros((N, 2)), np.zeros(N, np.int8)
        phase.ring_optics_row(phs.ring[0], np.ascontiguousarray(lam, float), t_ring, t_heat, v0, v1, phs.grid,
                              phs.dneff[r], phs.alpha_fca[r], phs.capacitance[r], phs.r_series[r], o["phi0"], o["phi1"],
                              o["a0"], o["a1"], o["e0"], o["e1"], o["k2t"], o["k2d"], o["trt"], shift, cj, rs, fl)
        o["shift"] = shift
        return o

    def ring_response(self, o, k, sl):
        """The exact-ring response of ring k at NRZ level 0 and level 1, on the wavelength slice sl of the rows o."""
        r0 = optics.ring_exact(o["phi0"][k, sl], o["k2t"][k, sl], o["k2d"][k, sl], o["a0"][k, sl], o["trt"][k, sl])
        r1 = optics.ring_exact(o["phi1"][k, sl], o["k2t"][k, sl], o["k2d"][k, sl], o["a1"][k, sl], o["trt"][k, sl])
        return r0, r1

    def frame(self, truth, centers_m):
        """(thru0, thru1, drop_avg) (N, G) power transmissions on the windows centred at centers_m (N,)."""
        N, G = self.pp.n_rings, self.offsets_m.size
        lam = (np.asarray(centers_m)[:, None] + self.offsets_m[None, :]).ravel()
        o = self.phase_rows(truth, lam)
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
        o = self.phase_rows(truth, lam)
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
        o = self.phase_rows(truth, np.array([lam_line]))
        rp = self.phs.ring[0][k]
        res, notch = [], []
        for lvl in (0, 1):
            phi = o[f"phi{lvl}"][k, 0]
            m0 = round(phi / (2 * math.pi))
            # the order whose resonance is nearest the line in wavelength (as Truth's lam_res)
            d, m = min(((kernel.resonance_detuning(rp, lam_line, o["shift"][k, lvl], float(mm)), mm)
                        for mm in (m0 - 1, m0, m0 + 1)), key=lambda dm: abs(dm[0]))
            lam_r = lam_line + d
            o2 = self.phase_rows(truth, np.array([lam_r]))
            rr = optics.ring_exact(np.array([2 * math.pi * m]), o2["k2t"][k, :1], o2["k2d"][k, :1],
                                   o2[f"a{lvl}"][k, :1], o2["trt"][k, :1])
            res.append(lam_r)
            notch.append(float(rr.p_thru[0]))
        return res, notch


def spectra_json(R, spec: Spectra, checks):
    """spectra_<id>.json: each ring's through and drop spectrum at every key frame, the exact resonances and notch
    depths, and the engine's bus powers. Also writes the spectra checks into checks[id]["spectra"]."""
    sd, plant, t_s, kidx, ticks = R["sd"], R["plant"], R["t_s"], R["kidx"], R["ticks"]
    N = plant.N
    center_kind = "resonance" if sd["id"] == "cold_start" else "declared_line"
    # one entry per key frame: spectra (T0, T1, DR), window centres, resonances, lines, junction levels, temperatures
    T0, T1, DR, CEN, LRES, LINES, VLO, VHI, RHO, TR, TH = [], [], [], [], [], [], [], [], [], [], []
    RESL, NOTCH = [], []
    BUS_IN, BUS_THRU, BUS_DROP, BUS_ABS = [], [], [], []
    # differences against the engine's own outputs, for checks.json
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
        T0.append(t0), T1.append(t1), DR.append(dr), CEN.append(cen), LRES.append(lam_res), LINES.append(
            lam_l
        )
        t_ring, t_heat, v0, v1, rho = spec.levels(truth)
        VLO.append(v0), VHI.append(v1), RHO.append(rho), TR.append(t_ring), TH.append(t_heat)
        # checks against the engine's own outputs at this instant
        lines = spec.at_lines(truth)
        p_in = truth.out_rl[0, optics.RING_LINE_OUTPUT_ROW_INPUT_POWER]  # (N, L) W
        BUS_IN.append(np.array(p_in, float)), BUS_THRU.append(
            np.array(truth.out_rl[0, optics.RING_LINE_OUTPUT_ROW_THROUGH_POWER], float)
        )
        BUS_DROP.append(np.array(truth.out_rl[0, optics.RING_LINE_OUTPUT_ROW_CLOCKWISE_DROP_POWER], float))
        BUS_ABS.append(np.array(truth.out_rl[0, optics.RING_LINE_OUTPUT_ROW_ABSORBED_POWER], float))
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
            if abs(lam_res[k] - cen[k]) < SPECTRUM_HALF_NM * 1e-9 - step_m:
                in_step.append(abs(lam_grid[i] - lam_res[k]) / step_m)
            # notch depth: grid minimum of each level against the closed form at its exact resonance
            for lvl, tt in ((0, t0[k]), (1, t1[k])):
                if abs(res[lvl] - cen[k]) < SPECTRUM_HALF_NM * 1e-9 - step_m:
                    err_notch_db.append(float(to_db(tt.min()) - to_db(notch[lvl])))
            # the spectra at the line reproduce the engine's through power, relOMA and monitor input
            thr = rho[k] * lines["t1"][k, l] + (1 - rho[k]) * lines["t0"][k, l]
            eng = truth.out_rl[0, optics.RING_LINE_OUTPUT_ROW_THROUGH_POWER, k, l] / p_in[k, l]
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
        "notch_grid_minus_exact_db_range": (
            [jnum(np.min(err_notch_db)), jnum(np.max(err_notch_db))] if err_notch_db else None
        ),
        "thru_at_line_vs_engine_rel_max": jnum(np.max(err_thru_rel)),
        "reloma_at_line_vs_engine_abs_max": jnum(np.max(err_reloma)),
        "drop_sum_vs_engine_p_drop_rel_max": jnum(np.max(err_drop_rel)),
    }
    out = {
        "id": sd["id"],
        "frames": R["kidx"].tolist(),
        "t_s": js(t_s[kidx], 6),
        "grid": {
            "mode": "per_ring_window",
            "center_kind": center_kind,
            "offset_nm": js(spec.offsets_m * 1e9, 6),
            "half_width_nm": SPECTRUM_HALF_NM,
            "points": int(G),
            "step_pm": jnum(step_m * 1e12),
            "note": (
                "wavelength = center_nm[frame][ring] + offset_nm. declared_line: the window is centred on "
                "the ring's own (declared) laser line, so offset = wavelength minus line. resonance: "
                "centred on the ring's resonance of the order nearest its declared line (the cold start, "
                "where the resonance starts nm away from any line)"
            ),
        },
        "center_nm": js(np.array(CEN) * 1e9, 9),
        "lam_res_nm": js(np.array(LRES) * 1e9, 9),
        "line_nm": js(np.array(LINES) * 1e9, 9),
        "thru0_db": js(to_db(np.array(T0)), 4, 2),
        "thru1_db": js(to_db(np.array(T1)), 4, 2),
        "drop_db": js(to_db(np.array(DR)), 4, 2),
        "resonance_level_nm": js(np.array(RESL) * 1e9, 9),
        "notch_level_db": js(to_db(np.array(NOTCH)), 4, 2),
        "bus": {
            "p_in_mw": js(np.array(BUS_IN) * 1e3, 4),
            "p_thru_mw": js(np.array(BUS_THRU) * 1e3, 4),
            "p_drop_uw": js(np.array(BUS_DROP) * 1e6, 4),
            "p_abs_uw": js(np.array(BUS_ABS) * 1e6, 4),
            "optical_note": (
                "p_drop_uw / p_abs_uw [frame][ring][line]: the engine's own drop-port power (to the "
                "monitor PD, RING_LINE_OUTPUT_ROW_CLOCKWISE_DROP_POWER) and absorbed power (RING_LINE_OUTPUT_ROW_ABSORBED_POWER) of each line in each ring, "
                "data-averaged: the light of the other lines in a ring's monitor and heat. The bus "
                "output OMA of a line (BER) averages the data of every other ring (RL_POUT), so the "
                "model has no data crosstalk between channels."
            ),
            "note": (
                "the engine's own optics outputs at t_s (Truth out_rl), [frame][ring][line] in mW: p_in = "
                "light of each line arriving at the ring's input after the bus segment (RING_LINE_OUTPUT_ROW_INPUT_POWER), p_thru = "
                "light of each line leaving its through port, data-averaged (RING_LINE_OUTPUT_ROW_THROUGH_POWER); the bus between ring "
                "k and k+1 carries p_thru[k] going in and p_in[k+1] arriving (the difference is bus loss)"
            ),
        },
        "levels": {
            "v_lo_v": js(np.array(VLO), 4),
            "v_hi_v": js(np.array(VHI), 4),
            "ones_density": js(np.array(RHO), 3),
            "t_ring_k": js(np.array(TR), 6, 3),
            "t_heater_k": js(np.array(TH), 6, 3),
        },
        "units": {
            "offset_nm": "nm",
            "center_nm": "nm",
            "lam_res_nm": "nm",
            "line_nm": "nm",
            "thru0_db": "dB",
            "thru1_db": "dB",
            "drop_db": "dB",
            "v_lo_v": "V",
            "v_hi_v": "V",
            "ones_density": "1",
            "t_ring_k": "K",
            "t_heater_k": "K",
            "t_s": "s",
            "resonance_level_nm": "nm",
            "notch_level_db": "dB",
        },
        "exact_note": (
            "resonance_level_nm [frame][ring][level]: the exact resonance of each NRZ level (order nearest "
            "the declared line); notch_level_db: the through transmission exactly there (the grid's "
            "12.5 pm step does not resolve the bottom of a deep notch; draw the bottom from these)"
        ),
        "note": (
            "power transmission of each ring alone (input 1 W on the bus at that wavelength): thru0_db / thru1_db "
            "the through port with the junction at NRZ level 0 (v_lo, lower reverse bias) / level 1 (v_hi); "
            "drop_db the drop port (to the monitor PD) averaged over the data, ones_density x level 1 + (1 - "
            "ones_density) x level 0, which is what the monitor and the lock see. The NRZ-averaged through port "
            "is the same average of thru0 and thru1 in linear units. Computed with the model's phase model "
            "(mrm.plant.phase) and exact ring (mrm.plant.optics.ring_exact) at the ring's state at t_s (ring "
            "and heater-resistor temperature, bias levels): rigid-comb phase model, no backscatter (g26)."
        ),
    }
    return out


# -------------------------------------------------------------------------------------------------------- device panels
def _device_frame_reducer(plant, S, dec):
    """Reducers of the device rows (one row every SAMPLE ticks from the reset tick) over the frame windows.

    Returns (get, per): get(key, how) gives the (S, ...) frame values of one device-row key, with how = "mean",
    "min", "max", "last" (the last sample of the frame) or "step" (the largest change between consecutive rows);
    per is the number of samples per frame. S: number of frames, dec: ticks per frame."""
    u_t = np.asarray(plant.u_ticks)
    if u_t[0] != 0 or not np.all(np.diff(u_t) == SAMPLE):
        raise RuntimeError("device rows are not every SAMPLE ticks from 0")
    per = dec // SAMPLE
    if len(plant.dev) < 1 + S * per:
        raise RuntimeError("device rows do not cover every frame")

    def get(key, how="mean"):
        v = np.array([r[key] for r in plant.dev[:1 + S * per]], float)
        if how == "step":                                  # largest change between consecutive rows (one per step)
            d = np.abs(np.diff(v, axis=0))
            return d.reshape((S, per) + v.shape[1:]).max(axis=1)
        w = v[1:].reshape((S, per) + v.shape[1:])
        return {"mean": w.mean, "min": w.min, "max": w.max}[how](axis=1) if how != "last" else w[:, -1]
    return get, per


def _per_ring_row(a, N):
    """A per-ring parameter as a contiguous (N,) float64 row: its last N values, or its one value repeated."""
    return np.ascontiguousarray(np.broadcast_to(np.asarray(a, float).reshape(-1)[-N:] if np.size(a) >= N
                                                else np.asarray(a, float).reshape(-1)[:1], (N,)), dtype=np.float64)


def running_metrics(R):
    """The engine's own metrics (mrm.sim.metrics.summarize) of the accumulators copied at every frame end: the
    value of each metric over the scored time up to that frame. Frames that end at or before the scoring start are
    null (the accumulators restart there)."""
    sd, plant, t_s = R["sd"], R["plant"], R["t_s"]
    dec, dt = sd["dec"], R["dt"]
    ms = plant.metrics_state
    sf = float(R["cr"].manifest_inputs.get("score_from_s") or 0.0)
    S, N = len(t_s), plant.N
    out = {k: np.full((S, N), np.nan) for k in RUNNING_METRICS}
    locked = np.full((S, N), np.nan)
    scored = np.zeros(S, bool)
    for f in range(S):
        k = (f + 1) * dec
        if k not in plant.msnaps:
            raise RuntimeError(f"no metric snapshot at tick {k}")
        if f * dec * dt < sf - 1e-12:
            continue
        af, ai, rh, ref = plant.msnaps[k]
        ns = types.SimpleNamespace(af=af[None], ai=ai[None], runhist=rh[None], dt_s=ms.dt_s, spec=ms.spec,
                                   ref=None if ref is None else ref[None], hist_edges_m=None)
        m = metrics.summarize(ns)
        for key in RUNNING_METRICS:
            out[key][f] = np.asarray(m[key], float)[0]
        locked[f] = ai[:, metrics.AI_LOCKED]
        scored[f] = True
    return out, locked, scored


def selfheat_loop_gain(spec, truth, z_self, dT=0.01):
    """Static self-heating loop gain of each ring at a Truth state: Z_opt,self(0) x dP_abs/dT_ring, with P_abs
    from the phase model and exact ring at every line (as static_s_curve), the light reaching each ring (Truth
    out_rl RING_LINE_OUTPUT_ROW_INPUT_POWER) held. Below 1 the ring's own heating is a stable (negative-feedback-like) loop at this point;
    above 1 it runs away (the unstable middle branch of the S-curve). Also P_abs at the state itself (a check
    against Truth p_abs_w).

    z_self (N,): DC self-heating gain of each ring (K/W). dT: half step of the central difference (K).
    Returns (loop gain (N,), P_abs at the state (N,) in W)."""
    N = spec.pp.n_rings
    lam = np.asarray(truth.lam_m[0], float)
    p_in = np.asarray(truth.out_rl[0, optics.RING_LINE_OUTPUT_ROW_INPUT_POWER], float)
    rho = spec.levels(truth)[4]
    t0 = np.asarray(truth.fields["t_ring_k"][0], float)

    def pabs(t):
        o = spec.phase_rows(truth, lam, t)
        out = np.zeros(N)
        for k in range(N):
            r0, r1 = spec.ring_response(o, k, slice(0, lam.size))
            out[k] = float(
                np.sum(p_in[k] * (rho[k] * o["e1"][k] * r1.p_loss + (1 - rho[k]) * o["e0"][k] * r0.p_loss))
            )
        return out

    p_hi, p_lo, p_0 = pabs(t0 + dT), pabs(t0 - dT), pabs(t0)
    return z_self * (p_hi - p_lo) / (2 * dT), p_0


def devices_json(R, spec: Spectra, checks):
    """devices_<id>.json: what the device panels, the bank table and the metrics section read (README)."""
    sd, plant, t_s, kidx, ticks = R["sd"], R["plant"], R["t_s"], R["kidx"], R["ticks"]
    dec, dt = sd["dec"], R["dt"]
    N, L, S = plant.N, plant.L, len(t_s)
    tr = R["cr"].trace
    rec = lambda f, red="mean": np.asarray(tr[RecordSpec(f, dec, red).key]["value"])[:, 0, :]   # noqa: E731
    get, per = _device_frame_reducer(plant, S, dec)
    hp, mh, lhw = plant.heater_params, plant.monitor_hw, plant.laser_hw
    # ---- heater: DAC path and driver law (kernel rows), supply limit
    reg = get("he_reg")
    heater = {
        "cmd": jint(get("he_cmd", "last")), "reg": jint(get("he_reg", "last")), "dac_code": jint(get("he_code", "last")),
        "v_v": js(get("he_v"), 5), "i_ma": js(get("he_i") * 1e3, 5), "r_ohm": js(get("he_r"), 6),
        "p_mw": js(get("he_p") * 1e3, 5), "step_max_codes": jint(get("he_reg", "step")),
        "cmd_minus_dac_max_abs": jint(np.abs(get("he_cmd", "last") - get("he_code", "last"))),
        "sat_count": jint(get("sat", "last")),
    }
    rows = plant.dev[1:1 + S * per]
    p_tot = np.array([np.sum(r["he_p"]) for r in rows]).reshape(S, per)
    i_tot = np.array([np.sum(r["he_i"]) for r in rows]).reshape(S, per)
    scale = np.array([r["he_scale"] for r in rows]).reshape(S, per)
    pmax, imax = float(np.ravel(hp.supply_power_max_w)[0]), float(np.ravel(hp.supply_current_max_a)[0])
    heater["bank_p_mw_max"] = js(p_tot.max(axis=1) * 1e3, 5)
    heater["bank_i_ma_max"] = js(i_tot.max(axis=1) * 1e3, 5)
    heater["supply_p_frac_max"] = js(p_tot.max(axis=1) / pmax, 4)
    heater["supply_i_frac_max"] = js(i_tot.max(axis=1) / imax, 4)
    heater["supply_scale_min"] = js(scale.min(axis=1), 6)
    # the heater power the full-scale code would give at the present heater temperature (driver law, no further
    # self-heating within the tick: g_self = 0)
    law = int(np.ravel(hp.law)[0])
    t_h = rec("t_heater_k")
    pfull = np.zeros((S, N))
    tmp = [np.zeros(N) for _ in range(4)]
    fl = np.zeros(N, np.int8)
    for f in range(S):
        electrical.heater_power_row(np.ones(N), law, np.ascontiguousarray(t_h[f]), np.zeros(N),
                                    _per_ring_row(hp.r_cold_ohm, N), _per_ring_row(hp.tcr_per_k, N),
                                    _per_ring_row(hp.tcr_ref_k, N), _per_ring_row(hp.v_full_scale_v, N),
                                    _per_ring_row(hp.i_full_scale_a, N), _per_ring_row(hp.p_full_scale_w, N),
                                    _per_ring_row(hp.compliance_v, N), tmp[0], tmp[1], tmp[2], tmp[3], fl)
        pfull[f] = tmp[0]
    heater["p_full_code_mw"] = js(pfull * 1e3, 5)
    # ---- monitor PD: the model's PD law, noise and ADC transfer (mrm.plant.sensing.MonitorHw)
    p_drop, t_pd, i_mon = rec("p_drop_w"), rec("t_pd_k"), rec("i_monitor_a")
    t_base = rec("t_base_k")
    pga = get("pga", "last").astype(np.int64)
    i_sig, i_d, sig, cpa, i_fs, ideal = (np.zeros((S, N)) for _ in range(6))
    cmax = float(mh.code_max)
    for f in range(S):
        a, b = mh.photocurrent_a(p_drop[f][None], 0.0, t_pd[f][None])
        i_sig[f], i_d[f] = a[0], b[0]
        sig[f] = mh.white_sigma_a(a, b)[0]
        t_adc = mh.adc_temperature(t_pd[f][None], t_base[f][None])
        z = mh.ideal_code(0.0, pga[f], t_adc)[0]
        cpa[f] = mh.ideal_code(1.0, pga[f], t_adc)[0] - z
        i_fs[f] = (cmax - z) / cpa[f]
        ideal[f] = mh.ideal_code(i_mon[f][None], pga[f], t_adc)[0]
    resp = float(np.ravel(mh.pr[:, :, sensing.PR_RESP])[0]) * float(mh.pw[0, sensing.PW_GAIN_SCALE])
    pd = {"i_signal_ua": js(i_sig * 1e6, 5), "i_dark_na": js(i_d * 1e9, 5),
          "i_dark_sampled_na": js(get("i_dark") * 1e9, 5),
          "noise_rms_na": js(sig * 1e9, 4), "noise_rms_codes": js(sig * cpa, 4),
          "codes_per_ua": js(cpa * 1e-6, 6), "full_scale_ua": js(i_fs * 1e6, 5),
          "full_scale_drop_uw": js(i_fs / resp * 1e6, 5), "ideal_code": js(ideal, 6),
          "pga": jint(pga), "samples_per_step": js(get("mo_count"), 4),
          "overrange_frac": js(get("mo_over"), 3)}
    # ---- laser lines
    f0 = np.asarray(lhw.f0_hz, float)[0]
    lf_f, lf_p, lf_pc = get("lf_f"), get("lf_p"), get("lf_pchip")
    p_mn, p_mx = get("lf_p", "min"), get("lf_p", "max")
    laser = {"f_dev_mhz": js((lf_f - f0) * 1e-6, 5), "f_dev_mhz_min": js((get("lf_f", "min") - f0) * 1e-6, 5),
             "f_dev_mhz_max": js((get("lf_f", "max") - f0) * 1e-6, 5),
             "p_mw": js(lf_p * 1e3, 7), "p_chip_mw": js(lf_pc * 1e3, 7),
             "p_ppm_min": js((p_mn / lf_p - 1) * 1e6, 4), "p_ppm_max": js((p_mx / lf_p - 1) * 1e6, 4)}
    # ---- data path
    datapath = {"ones_density": js(get("rho"), 5)}
    # ---- running metrics
    run, locked, scored = running_metrics(R)
    running = {k: js(v, 5) for k, v in run.items()}
    running["locked"] = _nan_to_null(np.where(np.isfinite(locked), locked, np.nan).tolist())
    running["unlocked_rings"] = [None if not scored[f] else int(np.sum(locked[f] == 0)) for f in range(S)]
    running["scored"] = scored.tolist()
    # ---- key frames: self-heating loop gain and the source decomposition of every ring and PD temperature
    sysm = plant.sys
    z_self = thermal.selfheat_dc_k_per_w(sysm)[0]
    LG, PAB, XR, XP = [], [], [], []
    err_obs, err_pabs = 0.0, 0.0
    lag_src, obs_ptr = np.asarray(sysm.lag_src), np.asarray(sysm.obs_ptr)
    pd_obs = [int(o) for o in np.asarray(sysm.pd_obs).ravel()] if sysm.pd_obs is not None else []

    def decompose(x, o):
        m = np.arange(obs_ptr[o], obs_ptr[o + 1])
        return np.bincount(lag_src[m], weights=x[m], minlength=sysm.n_src)
    for tick in ticks:
        truth = plant.snaps[int(tick)][0]
        lg, p0 = selfheat_loop_gain(spec, truth, z_self)
        LG.append(lg), PAB.append(p0)
        pe = np.asarray(truth.fields["p_abs_w"][0], float)
        err_pabs = max(err_pabs, float(np.max(np.abs(p0 - pe) / np.maximum(pe, 1e-30))))
        x = np.asarray(truth.x[0], float)
        tb = float(truth.fields["t_base_k"][0, 0])
        xr = np.array([decompose(x, int(sysm.ring_obs[i])) for i in range(N)])
        xp = np.array([decompose(x, o) for o in pd_obs]) if pd_obs else np.zeros((0, sysm.n_src))
        err_obs = max(err_obs, float(np.max(np.abs(xr.sum(axis=1) + tb - truth.t_obs[0][np.asarray(sysm.ring_obs)]))))
        XR.append(xr), XP.append(xp)
    checks[sd["id"]]["devices"] = {
        "ring_rise_by_source_sum_minus_engine_observer_k_max_abs": jnum(err_obs),
        "p_abs_phase_model_vs_engine_rel_max": jnum(err_pabs),
        "device_rows": int(len(plant.dev)), "metric_snapshots": int(len(plant.msnaps)),
        "running_final_minus_summary": _running_check(run, scored, R["cr"].metrics),
    }
    out = {
        "id": sd["id"], "t_s": js(t_s, 6), "frame_s": dec * dt, "sample_s": SAMPLE * dt, "samples_per_frame": per,
        "note": ("device rows: the engine's kernel state of world 0 read every SAMPLE ticks (each controller step, "
                 "at the step's edge, before that edge's command) by a read-only tap, reduced over each frame window "
                 "(mean of the samples unless the name or 'reducers' says otherwise). Unlike the scenario file's "
                 "records (every 1 us tick), these are samples every 10 us."),
        "heater": heater, "pd": pd, "laser": laser, "datapath": datapath, "running": running,
        "xt": derived.xt_json(R, t_s, checks),
        "reducers": {"heater.cmd": "last", "heater.reg": "last", "heater.dac_code": "last",
                     "heater.step_max_codes": "max |register change| between consecutive controller steps",
                     "heater.sat_count": "last (cumulative)", "heater.*_max": "max", "heater.supply_scale_min": "min",
                     "pd.pga": "last", "pd.i_signal_ua / i_dark_na / noise / full_scale / ideal_code":
                         "from the frame means of the records through the model's PD law and ADC transfer",
                     "laser.*_min / *_max": "envelope of the samples", "running.*": "value at the frame end"},
        "key": {"frames": kidx.tolist(), "t_s": js(t_s[kidx], 6),
                "selfheat_loop_gain": js(np.array(LG), 4), "p_abs_phase_model_mw": js(np.array(PAB) * 1e3, 5),
                "src_names": list(sysm.src_names), "ring_rise_by_source_k": js(np.array(XR), 4, 5),
                "pd_rise_by_source_k": js(np.array(XP), 4, 5),
                "note": ("selfheat_loop_gain: Z_opt,self(0) x dP_abs/dT_ring at the key frame (phase model + exact "
                         "ring at every line, light at the ring held): below 1 the ring's own heating is stable at "
                         "this point. ring_rise_by_source_k [frame][ring][source]: the ring's temperature rise above "
                         "the base split by heat source: the sum of the engine's thermal lag states of that ring "
                         "observer, grouped by the lag's source (exact: the observer is the base plus the sum of its "
                         "lags, mrm.plant.thermal.thermal_outputs_b). pd_rise_by_source_k: the same for each monitor "
                         "PD.")},
        "units": {"heater.v_v": "V", "heater.i_ma": "mA", "heater.r_ohm": "ohm", "heater.p_mw": "mW",
                  "heater.cmd / reg / dac_code / step_max_codes": "code", "pd.i_*_ua": "uA", "pd.*_na": "nA",
                  "pd.noise_rms_codes": "code", "pd.codes_per_ua": "code per uA", "pd.full_scale_drop_uw": "uW",
                  "laser.f_dev_mhz": "MHz from the nominal grid frequency", "laser.p_mw": "mW at the laser output",
                  "laser.p_chip_mw": "mW into the chip", "laser.p_ppm_*": "ppm of the frame mean",
                  "running.*_s": "s", "running.*_pm": "pm", "running.*_db": "dB", "running.heater_energy_j": "J"},
    }
    return out


def _running_check(run, scored, m):
    """The running metrics at the last scored frame against the run's summary (same accumulators; the last frame
    ends at or before the run's end)."""
    if not np.any(scored):
        return None
    f = int(np.flatnonzero(scored)[-1])
    out = {}
    for k in ("max_abs_lock_error_pm", "excursion_count", "heater_energy_j", "oma_penalty_db"):
        a = np.asarray(m[k], float)[0]
        b = run[k][f]
        out[k] = jnum(np.nanmax(np.abs(b - a) / np.maximum(np.abs(a), 1e-30)), 3)
    return out


# --------------------------------------------------------------------------------------------------------- thermal maps
def capture_thermal_inputs(cfg, uncertainty="nominal", variation="off", seeds=None):
    """The InterimRom, ThermalScales and assembly arguments that mrm.params.realize passes to
    mrm.plant.thermal.assemble for this config and world, through realize's own thermal_hook."""
    sd = runner._seeds(seeds)
    cap = {}

    def assemble(rom, layout, n, scales, **kw):
        cap.update(rom=rom, layout=layout, n=n, scales=scales, kw=kw)
        return thermal.assemble(rom, layout, n, scales, **kw)
    hook = types.SimpleNamespace(__name__="webviz_capture", ThermalScales=thermal.ThermalScales,
                                 InterimRom=thermal.InterimRom, load_rom=thermal.load_rom, assemble=assemble,
                                 selfheat_dc_k_per_w=thermal.selfheat_dc_k_per_w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp = params.realize(cfg, 1, seeds={"device": sd["device"], "uncertainty": sd["uncertainty"]},
                            uncertainty=uncertainty, variation=variation, thermal_hook=hook)
    if not isinstance(cap.get("rom"), thermal.InterimRom):
        raise RuntimeError("the thermal map assumes the interim thermal model (g26 default)")
    return cap, pp


class ThermalMap:
    """The model's thermal system with one extra point observer per grid node, for the thermal map.

    It adds probes at the ring centres and the monitor PDs. The observers follow the site law of the interim model
    (mrm.plant.thermal._interim_lags: c(d) Z0 with one pole tau_xt(d) per source), and mrm.plant.thermal.assemble
    builds the system with the run's own InterimRom and scales."""

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
        sysx = thermal.assemble(cap["rom"], ext, self.n, cap["scales"], **cap["kw"])
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
            thermal.thermal_outputs_b(x, obs_ptr, 0.0, tmp)
            out[want[0]] = tmp
        for k in range(len(u_t) - 1):
            u0 = np.where(agg, U[k], U[k + 1])
            thermal.thermal_step_b(x, a, c0, c1, lag_src, np.ascontiguousarray(u0), np.ascontiguousarray(U[k + 1]), ss)
            t = int(u_t[k + 1])
            if t in want:
                thermal.thermal_outputs_b(x, obs_ptr, 0.0, tmp)
                out[want[t]] = tmp
        return out, init


def thermal_json(R, tm: ThermalMap, checks):
    """thermal_<id>.json: the temperature-rise map (K above the base) at every key frame, the engine's own
    observers, and the colour-scale hint. Also writes the map checks into checks[id]["thermal"]."""
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


# ---------------------------------------------------------------------------------------------------------- bistability
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
    """The static self-heating S-curve of one ring, with the other sources held at their present powers.

    It scans the ring temperature T and, at each T, solves the heater power that holds it (DC gains of the thermal
    system: T_obs = T_base + G u). The absorbed light and the drop come from the phase model and the exact ring at
    every line that is on (all orders), and the code from the heater driver law with TCR at the heater-resistor
    temperature. The curve is stable where the heater power rises with T (self-heating loop gain below 1)."""
    G = thermal.dc_gain(sysm)[0]  # (O, S) K/W
    ro, ho = int(sysm.ring_obs[ring]), int(sysm.heater_obs[ring])
    hs, ws = int(sysm.heater_src[ring]), int(sysm.wg_src[ring])
    t_base = float(truth.fields["t_base_k"][0, 0])
    u_rest = np.array(u_now, float)
    u_rest[hs] = 0.0
    u_rest[ws] = 0.0
    rest_r = float(G[ro] @ u_rest)
    rest_h = float(G[ho] @ u_rest)
    phs = phase.PhaseParams.from_plant_params(pp)
    r = phs.table_row(0)
    rp_all = phs.ring[0]
    lam_l = np.asarray(truth.lam_m[0], float)
    p_in = np.asarray(
        truth.out_rl[0, optics.RING_LINE_OUTPUT_ROW_INPUT_POWER, ring], float
    )  # (L,) light reaching the ring
    c = truth.conditions[0]
    N = pp.n_rings
    v0, v1, rho = np.asarray(c.v_lo, float), np.asarray(c.v_hi, float), float(c.rho[ring])
    t_lock = float(truth.fields["t_ring_k"][0, ring])
    T = np.linspace(t_lock - t_below_k, t_lock + t_above_k, n)
    hp = electrical.HeaterDriverParams.from_plant_params(pp)
    out = {k: np.zeros(n) for k in ("p_h", "p_abs", "drop", "det", "code")}
    t_h = T.copy()
    line = int(np.rint(truth.fields["declared_line"][0, ring]))
    for it in range(4):
        for i in range(n):
            o = {
                k: np.zeros((N, lam_l.size))
                for k in ("phi0", "phi1", "a0", "a1", "e0", "e1", "k2t", "k2d", "trt")
            }
            shift, cj, rs, fl = np.zeros((N, 2)), np.zeros((N, 2)), np.zeros((N, 2)), np.zeros(N, np.int8)
            phase.ring_optics_row(
                rp_all,
                lam_l,
                np.full(N, T[i]),
                np.full(N, t_h[i]),
                v0,
                v1,
                phs.grid,
                phs.dneff[r],
                phs.alpha_fca[r],
                phs.capacitance[r],
                phs.r_series[r],
                o["phi0"],
                o["phi1"],
                o["a0"],
                o["a1"],
                o["e0"],
                o["e1"],
                o["k2t"],
                o["k2d"],
                o["trt"],
                shift,
                cj,
                rs,
                fl,
            )
            k = ring
            r0 = optics.ring_exact(o["phi0"][k], o["k2t"][k], o["k2d"][k], o["a0"][k], o["trt"][k])
            r1 = optics.ring_exact(o["phi1"][k], o["k2t"][k], o["k2d"][k], o["a1"][k], o["trt"][k])
            p_abs = float(np.sum(p_in * (rho * o["e1"][k] * r1.p_loss + (1 - rho) * o["e0"][k] * r0.p_loss)))
            drop = float(np.sum(p_in * (rho * np.abs(r1.drop) ** 2 + (1 - rho) * np.abs(r0.drop) ** 2)))
            p_h = (T[i] - t_base - G[ro, ws] * p_abs - rest_r) / G[ro, hs]
            t_h[i] = t_base + G[ho, hs] * p_h + G[ho, ws] * p_abs + rest_h
            out["p_h"][i], out["p_abs"][i], out["drop"][i] = p_h, p_abs, drop
            if it == 3:
                m = round(0.5 * (o["phi0"][k, line] + o["phi1"][k, line]) / (2 * math.pi))
                out["det"][i] = kernel.resonance_detuning(
                    rp_all[k], lam_l[line], 0.5 * (shift[k, 0] + shift[k, 1]), float(m)
                )
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
    """Fold points of the main loop of the S-curve.

    The folds are the extrema of the heater power over T, where the self-heating loop gain is 1. The local maximum
    nearest blue of the line ends the cold branch (the up-sweep jump). The local minimum nearest the line ends the hot
    branch (the down-sweep fall)."""
    dp = np.diff(p_h)
    ext = np.flatnonzero(np.sign(dp[1:]) * np.sign(dp[:-1]) < 0) + 1
    ups = [i for i in ext if dp[i - 1] > 0 > dp[i] and -12 * fwhm_pm < det[i] * 1e12 < 0]
    dns = [i for i in ext if dp[i - 1] < 0 < dp[i] and -12 * fwhm_pm < det[i] * 1e12 < 0.5 * fwhm_pm]
    if not ups or not dns:
        return None
    return max(ups, key=lambda i: det[i]), max(dns, key=lambda i: det[i])


def _bistability_scenario(cfg, ring_line, all_lines):
    """The config's scenario, with every laser line off except ring_line unless all_lines is True."""
    sc = scenario_registry.from_config(cfg)
    if all_lines:
        return sc
    off = tuple(l for l in range(int(cfg.get("array.n_rings"))) if l != ring_line)
    return sc + generators.Scenario(name="one_line", components=(generators.const(
        "laser.line_enable", -1.0, elements=off, label="webviz_one_line", provenance="assumed", source="",
        note="webviz bistability: only the swept ring's own line on"),), description="only one line on")


def bistability(log, quick, checks):
    """bistability.json: for each light level, the static S-curve of ring 0 and an open-loop engine sweep of its
    heater code up and down. Also writes the fold-against-jump checks into checks["bistability"]."""
    exp = experiment.load(str(ROOT / "experiments/examples/steady_pi_lock.yaml"))
    ring = 0
    runs = [(10.0, False), (6.0, False), (2.0, False), (6.0, True)]  # chip input loss (dB), all lines on
    out = {
        "ring": ring,
        "line": 0,
        "powers": [],
        "units": {
            "code": "code",
            "p_drop_uw": "uW",
            "detuning_pm": "pm",
            "t_ring_k": "K",
            "monitor_code": "code",
            "t_ms": "ms",
            "chip_loss_db": "dB",
            "p_ring_dbm": "dBm",
            "p_heater_mw": "mW",
            "p_abs_mw": "mW",
            "fwhm_pm": "pm",
        },
    }
    for loss_db, all_lines in runs:
        cfg = params.load(
            str(ROOT / "experiments/examples/steady_pi_lock.yaml"),
            overrides=[f"optical_path.chip_input_loss_db={loss_db}"],
        )
        r2l = cfg.get("channel_plan.ring_to_line", None)
        line = int(np.asarray(r2l)[ring]) if r2l is not None else ring
        sc = _bistability_scenario(cfg, line, all_lines)
        # the warm start (every ring at its locked-steady-state lock code, settled): the state the S-curve holds the
        # other sources in, and the codes the other rings keep during the sweep
        with tapped([SAMPLE]) as tap:  # Truth is filled from the first tick on
            w = runner.run(
                cfg,
                None,
                sc,
                duration_s=SAMPLE * 1e-6,
                batch=1,
                seeds=exp.seeds,
                uncertainty="nominal",
                variation="off",
                heater_code="locked_steady_state_reachable",
                settle=True,
                reset_protocol="branch_high",
            )
        plant0 = tap.plants[-1]
        truth0, u0 = plant0.snaps[SAMPLE]
        codes0 = np.asarray(w.manifest_inputs["warm_start"]["heater_code"])[0]
        pp = plant0.pp
        lam0 = float(np.asarray(truth0.lam_m[0])[line])
        fwhm_pm = float(lam0**2 * np.asarray(pp.optics.fwhm_hz)[0, ring] / C0 * 1e12)
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
        rec = [
            RecordSpec("p_drop_w", dec, "mean"),
            RecordSpec("heater_code", dec, "mean"),
            RecordSpec("detuning_m", dec, "mean"),
            RecordSpec("monitor_code", dec, "last"),
            RecordSpec("t_ring_k", dec, "mean"),
        ]
        t0 = time.perf_counter()
        res = runner.run(
            cfg,
            ctrl,
            sc,
            duration_s=tk[-1],
            batch=1,
            seeds=exp.seeds,
            uncertainty="nominal",
            variation="off",
            heater_code=start[None, :],
            settle=True,
            reset_protocol="branch_high",
            record=rec,
        )
        we = time.perf_counter() - t0
        tr = res.trace
        t_ms = np.asarray(tr[rec[0].key]["t_s"]) * 1e3
        get = lambda r: np.asarray(tr[r.key]["value"])[:, 0, ring]  # noqa: E731
        drop_e, code_e, det_e, mon_e, tring_e = (get(r) for r in rec)
        up = (t_ms > tk[1] * 1e3) & (t_ms <= tk[2] * 1e3 + 1e-9)
        dn = (t_ms > tk[3] * 1e3) & (t_ms <= tk[4] * 1e3 + 1e-9)
        jumps = []
        for leg, sel in (("up", up), ("down", dn)):
            ii = np.flatnonzero(sel)
            dd = np.diff(det_e[ii])
            j = int(np.argmax(np.abs(dd)))
            jumps.append(
                {
                    "leg": leg,
                    "code": jnum(0.5 * (code_e[ii[j]] + code_e[ii[j + 1]])),
                    "t_ms": jnum(t_ms[ii[j + 1]]),
                    "from_pm": jnum(det_e[ii[j]] * 1e12),
                    "to_pm": jnum(det_e[ii[j + 1]] * 1e12),
                    "p_drop_from_uw": jnum(drop_e[ii[j]] * 1e6),
                    "p_drop_to_uw": jnum(drop_e[ii[j + 1]] * 1e6),
                }
            )
        # the static curve inside the swept code range, every 5th point (plus the folds)
        keep = np.zeros(s["T"].size, bool)
        keep[::5] = True
        keep[[iu, idn]] = True
        inside = np.isfinite(s["code"]) & (s["code"] >= c_lo - 0.1 * span) & (s["code"] <= c_hi + 0.1 * span)
        lo_i, hi_i = min(iu, idn), max(iu, idn)  # the contiguous stretch of T around the loop
        while lo_i > 0 and inside[lo_i - 1]:
            lo_i -= 1
        while hi_i < inside.size - 1 and inside[hi_i + 1]:
            hi_i += 1
        seg = np.zeros_like(inside)
        seg[lo_i : hi_i + 1] = True
        sel = keep & seg
        stable = np.r_[np.diff(s["p_h"]) > 0, True]
        p_line = float(truth0.out_rl[0, optics.RING_LINE_OUTPUT_ROW_INPUT_POWER, ring, line])
        mon_fs = int(np.nanmax(mon_e))
        entry = {
            "chip_loss_db": loss_db,
            "laser_dbm": jnum(cfg.get("laser.power_per_line_dbm")),
            "lines_on": "all" if all_lines else [line],
            "p_ring_dbm": round(10 * math.log10(p_line * 1e3), 2) + 0.0,
            "fwhm_pm": jnum(fwhm_pm, 4),
            "static": {
                "code": js(s["code"][sel], 6),
                "p_drop_uw": js(s["drop"][sel] * 1e6, 5),
                "detuning_pm": js(s["det"][sel] * 1e12, 5),
                "t_ring_k": js(s["T"][sel], 6, 3),
                "p_heater_mw": js(s["p_h"][sel] * 1e3, 5),
                "p_abs_mw": js(s["p_abs"][sel] * 1e3, 5),
                "stable": [bool(x) for x in stable[sel]],
                "folds": [
                    {
                        "kind": "cold branch ends (up-sweep jump)",
                        "code": jnum(c_up),
                        "detuning_pm": jnum(s["det"][iu] * 1e12),
                        "p_drop_uw": jnum(s["drop"][iu] * 1e6),
                    },
                    {
                        "kind": "hot branch ends (down-sweep fall)",
                        "code": jnum(c_dn),
                        "detuning_pm": jnum(s["det"][idn] * 1e12),
                        "p_drop_uw": jnum(s["drop"][idn] * 1e6),
                    },
                ],
            },
            "sweep": {
                "t_knots_ms": js(np.array(tk) * 1e3, 6),
                "code_knots": js(ck, 6),
                "up": {
                    "t_ms": js(t_ms[up], 6),
                    "code": js(code_e[up], 6),
                    "p_drop_uw": js(drop_e[up] * 1e6, 5),
                    "detuning_pm": js(det_e[up] * 1e12, 5),
                    "monitor_code": jint(mon_e[up]),
                    "t_ring_k": js(tring_e[up], 6, 3),
                },
                "down": {
                    "t_ms": js(t_ms[dn], 6),
                    "code": js(code_e[dn], 6),
                    "p_drop_uw": js(drop_e[dn] * 1e6, 5),
                    "detuning_pm": js(det_e[dn] * 1e12, 5),
                    "monitor_code": jint(mon_e[dn]),
                    "t_ring_k": js(tring_e[dn], 6, 3),
                },
                "jumps": jumps,
                "monitor_code_max": mon_fs,
            },
        }
        out["powers"].append(entry)
        cj = {j["leg"]: j["code"] for j in jumps}
        key = f"ring_{round(10 * math.log10(p_line * 1e3), 1) + 0.0:+.1f}dBm" + (
            "_all_lines" if all_lines else ""
        )
        checks.setdefault("bistability", {})[key] = {
            "static_fold_codes": {"up_jump": jnum(c_up), "down_fall": jnum(c_dn)},
            "engine_jump_codes": cj,
            "engine_minus_static_codes": {"up": jnum(cj["up"] - c_up), "down": jnum(cj["down"] - c_dn)},
            "engine_jump_detuning_pm": {j["leg"]: [j["from_pm"], j["to_pm"]] for j in jumps},
            "static_fold_detuning_pm": {"up": jnum(s["det"][iu] * 1e12), "down": jnum(s["det"][idn] * 1e12)},
            "loop_width_codes_static": jnum(span),
            "wall_s": {"static": jnum(ws, 3), "engine": jnum(we, 3)},
        }
        log(
            f"  bistability {key}: static folds {c_up:.0f} / {c_dn:.0f}, engine jumps {cj['up']:.0f} / "
            f"{cj['down']:.0f} (static {ws:.1f} s, engine {we:.1f} s)"
        )
    out["note"] = (
        f"ring {ring} alone is swept open loop while the other rings hold their warm-start codes: 2 ms at "
        "the lowest code, a linear ramp up, 2 ms at the top, a linear ramp down (40 ms each). Only line 0 "
        "is on (lines_on), so the loop is the ring's own self-heating; the light at the ring is set by "
        "the chip input loss (10, 6, 2 dB: the model's 0 to +8 dBm world range); the last entry repeats "
        "the nominal +4 dBm with all eight lines on. static: the S-curve of the model's static laws (other sources held "
        "at the start); stable = the heater power rises with the ring temperature (self-heating loop gain "
        "below 1); the middle branch cannot be held open loop. folds: where the stable branches end. "
        "p_ring_dbm: light reaching the ring on its line. monitor_code saturates at the ADC full scale."
    )
    return out


# ----------------------------------------------------------------------------------------------------------------- meta
def _provenance(cfg, key):
    """The preset's provenance tag of a config key, or None when the preset has no such key."""
    try:
        return cfg.provenance(key)
    except Exception:                       # noqa: BLE001 - a key absent from this preset
        return None


def _finite_num(v):
    """jnum of float(v): a finite number rounded to 6 digits, else None."""
    v = float(v)
    return jnum(v) if math.isfinite(v) else None


def layout_extras(R0, pp, sysm, ring_xy, pd_xy, drv_xy, agg, ts, r_um):
    """Heater structure of the preset's heater family (the FEM extraction's geometry), the extraction's site sizes,
    and the distances between the model's site positions (all from the preset; every position is assumed)."""
    from mrm.extraction import thermal_fem
    cfg = R0["exp"].cfg
    tin = getattr(pp.thermal, "inputs", pp.thermal)            # realize's ThermalInputs
    fam = tuple(tin.family_names)[int(np.ravel(tin.family)[0])]
    film = float(cfg.get("ring.geometry.film_nm"))
    fc = thermal_fem.default_config(fam, film, radius_um=r_um, width_nm=RIB_W_UM * 1e3)
    rects = [{"r_inner_um": jnum(a), "r_outer_um": jnum(b), "z_bottom_um": jnum(z0), "z_top_um": jnum(z1),
              "material": mat} for a, b, z0, z1, mat in fc.heater_rects()]
    sites = {nm: np.asarray(xy, float) for nm, xy in agg.items()}
    die = {nm: np.asarray(xy, float) for nm, xy in ts.items()}
    dist = lambda a, b: np.hypot(*(np.asarray(a, float) - np.asarray(b, float)).T)   # noqa: E731
    d = {
        "ring_pitch_um": js(np.diff(ring_xy[:, 0]), 5), "ring_radius_um": jnum(r_um),
        "ring_to_own_pd_um": js(dist(ring_xy, pd_xy), 5),
        "ring_to_own_driver_um": js(dist(ring_xy, drv_xy), 5) if drv_xy is not None else None,
        "ring_to_site_um": {nm: js(dist(ring_xy, xy[None, :]), 5) for nm, xy in sites.items()},
        "pd_to_site_um": {nm: js(dist(pd_xy, xy[None, :]), 5) for nm, xy in sites.items()},
        "ring_to_temp_sensor_um": {nm: js(dist(ring_xy, xy[None, :]), 5) for nm, xy in die.items()},
        "ring_edge_to_own_pd_um": js(dist(ring_xy, pd_xy) - (r_um + RIB_W_UM / 2), 5),
        "note": "centre to centre unless the name says edge; from the preset's site positions (all assumed)",
    }
    prov = {k: _provenance(cfg, k) for k in ("layout.ring_pitch_um", "layout.sites", "ring.thermal.pd_offset_um",
                                             "ring.thermal.driver_offset_um", "ring.geometry.width_nm",
                                             "ring.geometry.film_nm", "ring.geometry.radius_um")}
    return {
        "heater_structure": {
            "family": fam, "rects": rects, "film_um": jnum(film * 1e-3),
            "note": ("the heater body of the preset's heater family (ring.thermal.family) as the model's FEM "
                     "extraction draws it (mrm.extraction.thermal_fem.ThermalConfig.heater_rects: axisymmetric "
                     "annuli, r from the ring centre, z from the bottom of the silicon film), at the model's ring "
                     "radius r_ref. Assumed: the G26 heater type and position are unpublished (preset "
                     "coverage 'metal/heater geometry'). The plant treats the heater as a lumped source.")},
        "site_sizes_um": {"monitor_pd_radius": jnum(fc.pd_radius_um), "driver_radius": jnum(fc.driver_radius_um),
                          "aggressor_radius": jnum(fc.aggressor_radius_um),
                          "temp_sensor_radius": jnum(fc.sensor_radius_um),
                          "note": ("the radii the FEM extraction gives these sites (ThermalConfig; assumed); the "
                                   "plant itself uses point sites")},
        "distances": d, "provenance": prov,
    }


def device_constants(R0):
    """Fixed device settings of the realized plant (nominal world), for the device panels' labels and rails."""
    plant, cfg, dt = R0["plant"], R0["exp"].cfg, R0["dt"]
    hp, mh, lhw, dph, spec, pp = (plant.heater_params, plant.monitor_hw, plant.laser_hw, plant.datapath_hw,
                                  plant.spec, plant.pp)
    laws = {v: k for k, v in electrical.DRIVER_LAWS.items()}
    N = plant.N
    heater = {
        "law": laws.get(int(np.ravel(hp.law)[0])), "bits": int(hp.bits), "code_lo": int(hp.code_lo),
        "code_hi": int(hp.code_hi), "v_full_scale_v": _finite_num(np.ravel(hp.v_full_scale_v)[0]),
        "r_cold_ohm": js(_per_ring_row(hp.r_cold_ohm, N), 6), "tcr_per_k": _finite_num(np.ravel(hp.tcr_per_k)[0]),
        "tcr_ref_k": _finite_num(np.ravel(hp.tcr_ref_k)[0]), "compliance_v": _finite_num(np.ravel(hp.compliance_v)[0]),
        "supply_power_max_w": _finite_num(np.ravel(hp.supply_power_max_w)[0]),
        "supply_current_max_a": _finite_num(np.ravel(hp.supply_current_max_a)[0]),
        "dac_latency_s": jnum(float(np.ravel(hp.latency_ticks)[0]) * dt),
        "min_update_s": jnum(float(hp.min_update_ticks) * dt),
        "slew_codes_per_s": _finite_num(np.ravel(hp.slew_u_per_s)[0] * hp.code_full_scale),
        "settle_tau_s": _finite_num(np.ravel(hp.settle_tau_s)[0]),
        "provenance": {k: _provenance(cfg, k) for k in (
            "ring.heater.driver_mode", "ring.heater.dac_bits", "ring.heater.full_scale_v", "ring.heater.r_cold_ohm",
            "ring.heater.tcr_per_k", "ring.heater.driver_compliance_v", "array.heater_total_power_max_w",
            "array.heater_supply_current_max_a", "ring.heater.latency_s", "ring.heater.code_max")},
    }
    lsb = float(mh.pw[0, sensing.PW_LSB])
    s_tia = float(mh.pw[0, sensing.PW_S_TIA])
    monitor = {
        "responsivity_a_per_w": jnum(float(np.ravel(mh.pr[:, :, sensing.PR_RESP])[0])),
        "gain_scale": jnum(float(mh.pw[0, sensing.PW_GAIN_SCALE])), "adc_bits": int(mh.adc_bits),
        "code_max": int(mh.code_max), "lsb_na": jnum(lsb * 1e9), "full_scale_ua_gain1": jnum(mh.code_max * lsb * 1e6),
        "pga_gains": [float(g) for g in np.ravel(mh.pga_gains)] or [1.0],
        "afe": "integrate_dump" if int(mh.afe_type) == int(sensing.AFE_INTEGRATE_DUMP) else "rc",
        "integration_window_s": jnum(mh.window_ticks * dt), "sample_period_s": jnum(mh.period_ticks * dt),
        "latency_s": jnum(float(np.ravel(mh.latency_ticks)[0]) * dt),
        "dark_i0_na": jnum(float(mh.pw[0, sensing.PW_DARK_I0]) * 1e9),
        "dark_t0_k": jnum(float(mh.pw[0, sensing.PW_DARK_T0])),
        "dark_ea_ev": jnum(float(mh.pw[0, sensing.PW_DARK_EA]) / 1.602176634e-19),
        "input_noise_pa_per_rt_hz": jnum(math.sqrt(s_tia) * 1e12) if s_tia > 0 else 0.0,
        "shot_noise": bool(mh.shot_on), "adc_ref_tempco_per_k": jnum(float(mh.pw[0, sensing.PW_TEMPCO])),
        "provenance": {k: _provenance(cfg, k) for k in (
            "ring.monitor.responsivity_a_per_w", "ring.monitor.dark_i0_na", "ring.monitor.dark_activation_ev",
            "ring.monitor.adc_bits", "ring.monitor.adc_full_scale_ua", "ring.monitor.integration_window_s",
            "ring.monitor.adc_rate_hz", "ring.monitor.input_noise_pa_per_rt_hz")},
    }
    controller = {"period_s": jnum(int(spec.period_ticks) * dt), "samples_per_step_capacity": int(spec.samples_per_step),
                  "adc_samples_per_step_nominal": jnum(int(spec.period_ticks) / int(mh.period_ticks)),
                  "schedule": str(spec.schedule)}
    rin = float(np.ravel(lhw.rin_per_hz)[0])
    laser = {"f0_thz": js(np.asarray(lhw.f0_hz, float)[0] * 1e-12, 8),
             "p0_mw": js(np.asarray(lhw.p0_w, float)[0] * 1e3, 5),
             "input_transmission": jnum(float(np.ravel(pp.laser.input_transmission)[0])),
             "rin_db_per_hz": jnum(10 * math.log10(rin)) if rin > 0 else None,
             "f_rate_hz_per_s": js(np.asarray(lhw.f_rate_hz_per_s, float)[0], 4),
             "f_rw_hz_per_rt_s": js(np.asarray(lhw.f_rw_hz_per_rt_s, float)[0], 4),
             "p_rate_per_s": js(np.asarray(lhw.p_rate_per_s, float)[0], 4),
             "provenance": {k: _provenance(cfg, k) for k in ("laser.power_per_line_dbm", "laser.rin_db_per_hz",
                                                             "optical_path.chip_input_loss_db")}}
    datapath = {"bitrate_gbps": jnum(dph.bitrate_hz * 1e-9), "link_transmission": jnum(float(np.ravel(dph.link_t)[0])),
                "oma_sensitivity_uw": jnum(float(np.ravel(dph.oma_sens_w)[0]) * 1e6), "q_ref": jnum(dph.q_ref),
                "ber_floor": jnum(dph.ber_floor)}
    return {"dt_s": dt, "sample_s": SAMPLE * dt, "heater": heater, "monitor": monitor, "controller": controller,
            "laser": laser, "datapath": datapath,
            "note": "the realized plant's settings (nominal world), read from the engine's hardware objects"}


def thermal_coupling(sysm):
    """Static (DC) thermal transfer of the model: K of rise at each ring, heater resistor and PD per mW into each
    source (mrm.plant.thermal.dc_gain)."""
    G = thermal.dc_gain(sysm)[0] * 1e-3                        # K/mW
    pd = [int(o) for o in np.asarray(sysm.pd_obs).ravel()] if sysm.pd_obs is not None else []
    return {"src_names": list(sysm.src_names),
            "ring_k_per_mw": js(G[np.asarray(sysm.ring_obs)], 4),
            "heater_resistor_k_per_mw": js(G[np.asarray(sysm.heater_obs)], 4),
            "pd_k_per_mw": js(G[pd], 4) if pd else None,
            "note": "DC gains Z(0) = sum of the lag residues (mrm.plant.thermal.dc_gain), K per mW into the source"}


def _junction_of(results):
    """The junction levels of the steady run (else of the first scenario regenerated) for the limits table."""
    for R in results:
        if R["sd"]["id"] == "steady":
            return R.get("junction"), "steady"
    return results[0].get("junction"), results[0]["sd"]["id"]


def meta_json(date, results, cap, pp, sysm, specs):
    """meta.json: the geometry (site positions, drawn waveguides and heater), the laser lines, the ring FSR and FWHM,
    the scenario list, the device settings, the thermal coupling, the hardware limits, units and model limits."""
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
                               "thermal": f"thermal_{sd['id']}.json", "devices": f"devices_{sd['id']}.json",
                               "stats": f"stats_{sd['id']}.json"}})
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
            **layout_extras(results[0], pp, sysm, ring_xy, pd_xy, drv_xy, agg, ts, r_um),
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
        "spectra_grid": {"mode": "per_ring_window", "half_width_nm": SPECTRUM_HALF_NM, "points": SPECTRUM_POINTS,
                         "step_pm": jnum(2 * SPECTRUM_HALF_NM / (SPECTRUM_POINTS - 1) * 1e3)},
        "thermal_grid": {"x_um": [float(GRID_X[0]), float(GRID_X[-1])], "nx": int(GRID_X.size),
                         "y_um": [float(GRID_Y[0]), float(GRID_Y[-1])], "ny": int(GRID_Y.size)},
        "files": {"bistability": "bistability.json", "checks": "checks.json"},
        "devices": {**device_constants(results[0]), "timing": derived.controller_timing(results[0])},
        "thermal_coupling": {**thermal_coupling(sysm), "full": derived.coupling_json(sysm)},
        "hardware_limits": {**derived.hardware_limits(results[0], _junction_of(results)[0]),
                            "junction_from": _junction_of(results)[1]},
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


# ----------------------------------------------------------------------------------------------------------------- main
def main(argv=None):
    """Run every scenario and write the data files (see the module docstring for the order)."""
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
    cap, pp_nom = capture_thermal_inputs(experiment.load(str(ROOT / defs[0]["file"])).cfg)
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
        cfg = R["exp"].cfg
        R["bias_range_v"] = (float(cfg.get("ring.bias_dac.v_min_v")), float(cfg.get("ring.bias_dac.v_max_v")))
        specj = spectra_json(R, spec, checks)
        thermj = thermal_json(R, tmap, checks)
        devj = devices_json(R, spec, checks)
        stj = derived.stats_json(R, sj, devj, specj, checks)
        R["junction"] = stj["junction"]
        sj["plain_description"] = derived.describe(R, sj, stj, devj, _site_distances(R), _a1_period(R))
        sizes[f"scenario_{sd['id']}.json"] = write_json(out_dir / f"scenario_{sd['id']}.json", sj)
        sizes[f"spectra_{sd['id']}.json"] = write_json(out_dir / f"spectra_{sd['id']}.json", specj)
        sizes[f"thermal_{sd['id']}.json"] = write_json(out_dir / f"thermal_{sd['id']}.json", thermj)
        sizes[f"devices_{sd['id']}.json"] = write_json(out_dir / f"devices_{sd['id']}.json", devj)
        sizes[f"stats_{sd['id']}.json"] = write_json(out_dir / f"stats_{sd['id']}.json", stj)
        del specj, thermj, devj, stj
        log(f"  wrote {sd['id']}: " + ", ".join(f"{k} {v / 1e6:.2f} MB" for k, v in sizes.items()
                                                if sd["id"] in k))
        R["plant"].snaps = {}                                     # free the snapshots
        R["plant"].dev, R["plant"].msnaps = [], {}
        R["plant"].xt_sum, R["plant"].xt_n = {}, {}
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


if __name__ == "__main__":
    main()
