"""Run the adversarial engine scenarios: the cases most likely to expose an inexact engine optimization.

    .venv/bin/python tools/memguard.py --heavy -- .venv/bin/python tools/perf_scenarios_adv.py runs/x
    .venv/bin/python tools/perf_scenarios_adv.py runs/x --only laser_move_pi,auto_pi10k

Read this first. These scenarios add to the fixed set of tools/perf_scenarios.py. They target three kinds of error:
a cache that is not invalidated, a restructuring that is not exact, and an approximation that leaks into the
default engine. The output format is the same (``<out>/<name>.npz``, compared with tools/perf_compare.py). Here the
final Truth holds every Truth field, also er_db / il_db, which the engine writes only when a record reads them or
at the end of a block.

Scenarios (g26 nominal world, B = 1 unless stated, fixed 1 us ticks unless stated, seeds 0):
  agg_edge_pi      pi_lock_c under a direct square aggressor (0.5 W, 4 ms period: instantaneous edges), 16 ms, at
                   the default g26 site (0, -30) um. The old outputs in runs/review_r1/*_adv were recorded with the
                   earlier site (0, 30) um, 5 um from ring 0's PD: record a new baseline before you compare
                   agg_edge_pi again (docs/handoff/perf.md, "Layout change of 2026-10-10")
  laser_move_pi    pi_lock_c while the lines move: a 5 GHz ramp of every line over 8 ms, then a 10 GHz step of
                   line 0 (in-solve rotation up to PHASE_ROTATION_POLYNOMIAL_MAX_ANGLE_RAD), a 30 GHz step of line 3 (beyond PHASE_ROTATION_POLYNOMIAL_MAX_ANGLE_RAD: the libm path)
                   and a line switched off and on again (K^2 cache, Truth FWHM cache, incremental cos/sin cache)
  laser_move_open  the same line moves, open loop at the settled steady-state codes
  fold_up          from the settled hot branch: heater ramp from the lock code to 1.35 x (Python controller; the
                   ring falls off the self-heating fold)
  fold_down        from the settled hot branch: heater ramp down to 0.6 x and back
  traffic_idle_pi  pi_lock_c, traffic on at 4 ms and off (idle, no swing) at 12 ms, 100 kHz bias ripple of 5 mV
  auto_pi10k       pi_lock_c at a 10 kHz controller rate under substep_policy auto (multi-tick substeps), 30 ms
  faults_pi        pi_lock_c with a +40-code heater DAC glitch on rings 0-3 (6-9 ms), an ADC drop on ring 5, the
                   PWM ripple on, the er_db / il_db records on
  backscatter_pi   pi_lock_c with fidelity.backscatter on (the general optics path of the engine adapter), 10 ms
  mc8_auto         B = 8 sampled worlds (seeds 2000), pi_lock_c at 10 kHz, substep_policy auto, 8 ms
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from perf_scenarios import FIELDS, load_g26, record_specs  # noqa: E402

from mrm import params  # noqa: E402
from mrm.control import registry  # noqa: E402
from mrm.scenarios import generators  # noqa: E402
from mrm.scenarios import library  # noqa: E402
from mrm.sim import runner  # noqa: E402
from mrm.sim.metrics import TRUTH_FIELDS  # noqa: E402

ADV = ("agg_edge_pi", "laser_move_pi", "laser_move_open", "fold_up", "fold_down", "traffic_idle_pi", "auto_pi10k",
       "faults_pi", "backscatter_pi", "mc8_auto")
#: provenance keywords of every scenario component built here
PROVENANCE_KW = dict(provenance="assumed", source="", note="perf review adversarial case")


class Ramp2:
    """Python open-loop heater ramp from c0 to c1 over [t0, t1] (then held), optionally back to c0 over [t1, t2].

    ``c0``, ``c1``: (N,) heater codes. ``t0``, ``t1``, ``t2``: times, s (the controller assumes 1 us ticks)."""
    name = "ramp2"
    latency_ticks = 0

    def __init__(self, c0, c1, t0, t1, t2=None):
        self.c0 = np.asarray(c0, float)
        self.c1 = np.asarray(c1, float)
        self.t0, self.t1, self.t2 = float(t0), float(t1), t2

    def reset(self, spec):
        self.spec = spec

    def step(self, obs):
        t = obs.tick * 1e-6
        if t <= self.t0:
            frac = 0.0
        elif t <= self.t1:
            frac = (t - self.t0) / (self.t1 - self.t0)
        elif self.t2 is None:
            frac = 1.0
        else:
            frac = max(0.0, 1.0 - (t - self.t1) / (self.t2 - self.t1))
        code = np.rint(self.c0 + (self.c1 - self.c0) * frac).astype(np.int64)
        return {"tx.heater": code[None, :]}


def _lock_codes(cfg):
    """(N,) steady-state lock codes of the nominal world of ``cfg``."""
    from mrm.control.api import ControllerSpec
    from mrm.plant.plant import Plant
    _, pp, _ = runner.load_inputs(cfg, batch=1, seeds=0, uncertainty="nominal", variation="off")
    spec = ControllerSpec.from_config(cfg, n_batch=1, nominal=cfg)
    return np.asarray(runner.warm_start_codes(Plant(pp, None, spec), cfg, mode="locked_steady_state"), float)[0]


def _line_moves():
    """Static scenario plus line moves: a ramp of every line, steps of lines 0 and 3, line 5 dark for 1 ms and a
    0.5 dB laser power drop."""
    return library.static() + generators.Scenario(name="line_moves", components=(
        generators.pwl("laser.dfreq_hz", [0.0, 2e-3, 10e-3, 30e-3], [0.0, 0.0, 5e9, 5e9], label="ramp",
                       **PROVENANCE_KW),
        generators.step("laser.dfreq_hz", 12e-3, 10e9, elements=(0,), label="step0", **PROVENANCE_KW),
        generators.step("laser.dfreq_hz", 14e-3, 30e9, elements=(3,), label="step3", **PROVENANCE_KW),
        generators.pulse("laser.line_enable", 16e-3, 17e-3, -1.0, elements=(5,), label="dark5", **PROVENANCE_KW),
        generators.step("laser.power_delta_db", 18e-3, -0.5, label="pdrop", **PROVENANCE_KW)))


def run_one(name: str, plant_kw: dict):
    """Run scenario ``name``; returns (runner result, wall time in s). ``plant_kw`` overrides the runner keywords."""
    t0 = time.perf_counter()
    common = dict(uncertainty="nominal", variation="off", seeds=0, batch=1)
    common.update(plant_kw)
    warm = dict(heater_code="locked_steady_state", reset_protocol="branch_high", settle=True)
    if name == "agg_edge_pi":
        cfg = load_g26()
        scenario = library.static() + library.aggressor_direct(waveform="square", power_w=0.5, period_s=4e-3,
                                                               duty=0.5, t0_s=1e-3)
        res = runner.run(cfg, registry.create("pi_lock_c", {}), scenario, duration_s=16e-3, record=record_specs(1),
                         **warm, **common)
    elif name in ("laser_move_pi", "laser_move_open"):
        cfg = load_g26()
        ctrl = registry.create("pi_lock_c", {}) if name == "laser_move_pi" else None
        res = runner.run(cfg, ctrl, _line_moves(), duration_s=22e-3, record=record_specs(1), **warm, **common)
    elif name in ("fold_up", "fold_down"):
        cfg = load_g26()
        codes = _lock_codes(cfg)
        if name == "fold_up":
            ctrl = Ramp2(codes, np.minimum(codes * 1.35, 65535.0), 1e-3, 15e-3)      # 65535: g26 code_max
        else:
            ctrl = Ramp2(codes, codes * 0.6, 1e-3, 9e-3, 17e-3)
        res = runner.run(cfg, ctrl, library.static(), duration_s=18e-3, record=record_specs(1),
                         heater_code=codes[None, :], reset_protocol="branch_high", settle=True, **common)
    elif name == "traffic_idle_pi":
        cfg = load_g26()
        scenario = (library.static() + library.traffic_start_stop(t_start_s=4e-3, t_stop_s=12e-3)
                    + library.bias_ripple())
        res = runner.run(cfg, registry.create("pi_lock_c", {}), scenario, duration_s=16e-3, record=record_specs(1),
                         **warm, **common)
    elif name == "auto_pi10k":
        cfg = load_g26(("controller_interface.update_rate_hz=10000.0",))
        res = runner.run(cfg, registry.create("pi_lock_c", {}), library.static(), duration_s=30e-3,
                         record=record_specs(10), substep_policy="auto", **warm, **common)
    elif name == "faults_pi":
        cfg = load_g26(('fidelity.heater_pwm_ripple="on"',))
        scenario = library.static() + generators.Scenario(name="faults", components=(
            generators.pulse("fault.dac_glitch_codes", 6e-3, 9e-3, 40.0, elements=(0, 1, 2, 3), label="glitch",
                             **PROVENANCE_KW),
            generators.pulse("fault.adc_drop", 7e-3, 8e-3, 1.0, elements=(5,), label="drop", **PROVENANCE_KW)))
        res = runner.run(cfg, registry.create("pi_lock_c", {}), scenario, duration_s=12e-3,
                         record=record_specs(1, fields=FIELDS + ("er_db", "il_db")), **warm, **common)
    elif name == "backscatter_pi":
        cfg = load_g26(('fidelity.backscatter="on"',))
        res = runner.run(cfg, registry.create("pi_lock_c", {}), library.static(), duration_s=10e-3,
                         record=record_specs(1), **warm, **common)
    elif name == "mc8_auto":
        cfg = load_g26(("controller_interface.update_rate_hz=10000.0",))
        common.update(uncertainty="sample", variation="sample", seeds=2000, batch=8)
        res = runner.run(cfg, registry.create("pi_lock_c", {}), library.static(), duration_s=8e-3,
                         record=record_specs(10, "mean"), substep_policy="auto", **warm,
                         n_threads=int(os.environ.get("MRM_PERF_THREADS", "4")), **common)
    else:
        raise KeyError(name)
    return res, time.perf_counter() - t0


def save(res, path: Path, wall: float):
    """Write the traces, the numeric metrics, every final Truth field (plus the optics outputs and the thermal
    state of the final Truth when present) and three solver diagnostics to the npz ``path``; returns the
    summary.json entry {wall_s, diagnostics}."""
    out = {}
    for key, tr in (res.trace or {}).items():
        out[f"trace/{key}/t_s"] = np.asarray(tr["t_s"])
        out[f"trace/{key}/value"] = np.asarray(tr["value"])
    for key, v in res.metrics.items():
        items = v.items() if isinstance(v, dict) else [(None, v)]
        for k2, v2 in items:
            arr = np.asarray(v2)
            if arr.dtype.kind in "fiub":               # numbers and booleans only
                out[f"metric/{key}" + ("" if k2 is None else f"/{k2}")] = arr
    truth_final = res.truth_final
    for f in TRUTH_FIELDS:
        out[f"final/{f}"] = np.asarray(truth_final.fields[f])
    for f in ("out_rl", "out_r", "out_l", "t_obs", "x"):
        arr = getattr(truth_final, f, None)
        if arr is not None:
            out[f"final/{f}"] = np.asarray(arr)
    diag = res.diagnostics
    for k in ("n_substeps", "selfheat_not_converged", "selfheat_iterations"):
        if k in diag:
            out[f"metric/diag_{k}"] = np.asarray(diag[k])
    np.savez_compressed(path, **out)
    return {"wall_s": wall, "diagnostics": {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                                            for k, v in diag.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--only", default="")
    ap.add_argument("--set", action="append", default=[], help="Plant/runner keyword key=value (python literal)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [s for s in args.only.split(",") if s] or list(ADV)
    plant_kw = {}
    for s in args.set:
        k, v = s.split("=", 1)
        plant_kw[k] = eval(v, {}, {})       # noqa: S307 (local tool, literal values)
    summary_file = out / "summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
    for name in names:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res, wall = run_one(name, plant_kw)
        info = save(res, out / f"{name}.npz", wall)
        info["load"] = os.getloadavg()[0]
        summary[name] = info
        print(f"[perf_scenarios_adv] {name}: wall {wall:.1f} s (load {info['load']:.1f})", flush=True)
        summary_file.write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
