"""Adversarial accuracy-regression scenarios for engine performance work (docs/handoff/perf.md, review round 1).

The cases that would expose a cache that is not invalidated, an inexact restructuring or an approximation that leaks
into the default engine, beyond the fixed set of tools/perf_scenarios.py. Same output format (``<out>/<name>.npz``,
compared with tools/perf_compare.py); the final Truth here holds every Truth field (also er_db / il_db, which the
engine writes only when a record reads them or at the end of a block).

    .venv/bin/python tools/memguard.py --heavy -- .venv/bin/python tools/perf_scenarios_adv.py runs/x
    .venv/bin/python tools/perf_scenarios_adv.py runs/x --only laser_move_pi,auto_pi10k

Scenarios (g26 nominal world, B = 1 unless stated, fixed 1 us ticks unless stated, seeds 0):
  agg_edge_pi      pi_lock_c under a direct square aggressor (0.5 W, 4 ms period: instantaneous edges), 16 ms, at
                   the default g26 site (0, -30) um. runs/review_r1/*_adv were recorded with the former site
                   (0, 30) um, 5 um from ring 0's PD: record a new baseline before comparing agg_edge_pi again
                   (docs/handoff/perf.md section 7)
  laser_move_pi    pi_lock_c while the lines move: a 5 GHz ramp of every line over 8 ms, then a 10 GHz step of
                   line 0 (in-solve rotation up to ROT_MAX), a 30 GHz step of line 3 (beyond ROT_MAX: the libm path)
                   and a line switched off and on again (K^2 cache, Truth FWHM cache, incremental cos/sin cache)
  laser_move_open  the same line moves open loop at the settled quasi-static codes
  fold_up          from the settled hot branch: heater ramp from the lock code to 1.35 x (Python controller; the
                   ring falls off the self-heating fold)
  fold_down        from the settled hot branch: heater ramp down to 0.6 x and back
  traffic_idle_pi  pi_lock_c, traffic on at 4 ms and off (idle, no swing) at 12 ms, 100 kHz bias ripple of 5 mV
  auto_pi10k       pi_lock_c at a 10 kHz controller rate under substep_policy auto (multi-tick substeps), 30 ms
  faults_pi        pi_lock_c with a +40-code heater DAC glitch on rings 0-3 (6-9 ms), an ADC drop on ring 5, the
                   PWM ripple on, the er_db / il_db records on
  backscatter_pi   pi_lock_c with fidelity.backscatter on (the general optics path of the engine adapter), 10 ms
  tcmt_open        open loop, fidelity.optics_model tcmt, 5 ms
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

from perf_scenarios import FIELDS, _cfg, _rec  # noqa: E402

from mrm import params  # noqa: E402
from mrm.control import registry  # noqa: E402
from mrm.scenarios import generators as g  # noqa: E402
from mrm.scenarios import library as lib  # noqa: E402
from mrm.sim import runner  # noqa: E402
from mrm.sim.metrics import TRUTH_FIELDS  # noqa: E402

ADV = ("agg_edge_pi", "laser_move_pi", "laser_move_open", "fold_up", "fold_down", "traffic_idle_pi", "auto_pi10k",
       "faults_pi", "backscatter_pi", "tcmt_open", "mc8_auto")
KW = dict(provenance="assumed", source="", note="perf review adversarial case")


class Ramp2:
    """Python open-loop heater ramp from c0 to c1 over [t0, t1] (then held), optionally back to c0 over [t1, t2]."""
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
            f = 0.0
        elif t <= self.t1:
            f = (t - self.t0) / (self.t1 - self.t0)
        elif self.t2 is None:
            f = 1.0
        else:
            f = max(0.0, 1.0 - (t - self.t1) / (self.t2 - self.t1))
        c = np.rint(self.c0 + (self.c1 - self.c0) * f).astype(np.int64)
        return {"tx.heater": c[None, :]}


def _lock_codes(cfg):
    from mrm.control.api import ControllerSpec
    from mrm.plant.plant import Plant
    _, pp, _ = runner.load_inputs(cfg, batch=1, seeds=0, uncertainty="nominal", variation="off")
    spec = ControllerSpec.from_config(cfg, n_batch=1, nominal=cfg)
    return np.asarray(runner.warm_start_codes(Plant(pp, None, spec), cfg, mode="quasi_static"), float)[0]


def _line_moves():
    return lib.static() + g.Scenario(name="line_moves", components=(
        g.pwl("laser.dfreq_hz", [0.0, 2e-3, 10e-3, 30e-3], [0.0, 0.0, 5e9, 5e9], label="ramp", **KW),
        g.step("laser.dfreq_hz", 12e-3, 10e9, elements=(0,), label="step0", **KW),
        g.step("laser.dfreq_hz", 14e-3, 30e9, elements=(3,), label="step3", **KW),
        g.pulse("laser.line_enable", 16e-3, 17e-3, -1.0, elements=(5,), label="dark5", **KW),
        g.step("laser.power_delta_db", 18e-3, -0.5, label="pdrop", **KW)))


def run_one(name: str, plant_kw: dict):
    t0 = time.perf_counter()
    common = dict(uncertainty="nominal", variation="off", seeds=0, batch=1)
    common.update(plant_kw)
    warm = dict(heater_code="quasi_static", reset_protocol="branch_high", settle=True)
    if name == "agg_edge_pi":
        cfg = _cfg()
        sc = lib.static() + lib.aggressor_direct(waveform="square", power_w=0.5, period_s=4e-3, duty=0.5,
                                                 t0_s=1e-3)
        res = runner.run(cfg, registry.create("pi_lock_c", {}), sc, duration_s=16e-3, record=_rec(1), **warm,
                         **common)
    elif name in ("laser_move_pi", "laser_move_open"):
        cfg = _cfg()
        ctrl = registry.create("pi_lock_c", {}) if name == "laser_move_pi" else None
        res = runner.run(cfg, ctrl, _line_moves(), duration_s=22e-3, record=_rec(1), **warm, **common)
    elif name in ("fold_up", "fold_down"):
        cfg = _cfg()
        c = _lock_codes(cfg)
        if name == "fold_up":
            ctrl = Ramp2(c, np.minimum(c * 1.35, 65535.0), 1e-3, 15e-3)
        else:
            ctrl = Ramp2(c, c * 0.6, 1e-3, 9e-3, 17e-3)
        res = runner.run(cfg, ctrl, lib.static(), duration_s=18e-3, record=_rec(1), heater_code=c[None, :],
                         reset_protocol="branch_high", settle=True, **common)
    elif name == "traffic_idle_pi":
        cfg = _cfg()
        sc = lib.static() + lib.traffic_start_stop(t_start_s=4e-3, t_stop_s=12e-3) + lib.bias_ripple()
        res = runner.run(cfg, registry.create("pi_lock_c", {}), sc, duration_s=16e-3, record=_rec(1), **warm,
                         **common)
    elif name == "auto_pi10k":
        cfg = _cfg(("controller_interface.update_rate_hz=10000.0",))
        res = runner.run(cfg, registry.create("pi_lock_c", {}), lib.static(), duration_s=30e-3, record=_rec(10),
                         substep_policy="auto", **warm, **common)
    elif name == "faults_pi":
        cfg = _cfg(('fidelity.heater_pwm_ripple="on"',))
        sc = lib.static() + g.Scenario(name="faults", components=(
            g.pulse("fault.dac_glitch_codes", 6e-3, 9e-3, 40.0, elements=(0, 1, 2, 3), label="glitch", **KW),
            g.pulse("fault.adc_drop", 7e-3, 8e-3, 1.0, elements=(5,), label="drop", **KW)))
        res = runner.run(cfg, registry.create("pi_lock_c", {}), sc, duration_s=12e-3,
                         record=_rec(1, fields=FIELDS + ("er_db", "il_db")), **warm, **common)
    elif name == "backscatter_pi":
        cfg = _cfg(('fidelity.backscatter="on"',))
        res = runner.run(cfg, registry.create("pi_lock_c", {}), lib.static(), duration_s=10e-3, record=_rec(1),
                         **warm, **common)
    elif name == "tcmt_open":
        cfg = _cfg(("fidelity.optics_model=tcmt",))
        res = runner.run(cfg, None, lib.static(), duration_s=5e-3, record=_rec(1), **warm, **common)
    elif name == "mc8_auto":
        cfg = _cfg(("controller_interface.update_rate_hz=10000.0",))
        common.update(uncertainty="sample", variation="sample", seeds=2000, batch=8)
        res = runner.run(cfg, registry.create("pi_lock_c", {}), lib.static(), duration_s=8e-3,
                         record=_rec(10, "mean"), substep_policy="auto", **warm,
                         n_threads=int(os.environ.get("MRM_PERF_THREADS", "4")), **common)
    else:
        raise KeyError(name)
    return res, time.perf_counter() - t0


def save(res, path: Path, wall: float):
    out = {}
    for key, tr in (res.trace or {}).items():
        out[f"trace/{key}/t_s"] = np.asarray(tr["t_s"])
        out[f"trace/{key}/value"] = np.asarray(tr["value"])
    for key, v in res.metrics.items():
        items = v.items() if isinstance(v, dict) else [(None, v)]
        for k2, v2 in items:
            a = np.asarray(v2)
            if a.dtype.kind in "fiub":
                out[f"metric/{key}" + ("" if k2 is None else f"/{k2}")] = a
    tf = res.truth_final
    for f in TRUTH_FIELDS:
        out[f"final/{f}"] = np.asarray(tf.fields[f])
    for f in ("out_rl", "out_r", "out_l", "t_obs", "x"):
        a = getattr(tf, f, None)
        if a is not None:
            out[f"final/{f}"] = np.asarray(a)
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
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [s for s in a.only.split(",") if s] or list(ADV)
    kw = {}
    for s in a.set:
        k, v = s.split("=", 1)
        kw[k] = eval(v, {}, {})       # noqa: S307 (local tool, literal values)
    sfile = out / "summary.json"
    summary = json.loads(sfile.read_text()) if sfile.exists() else {}
    for n in names:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res, wall = run_one(n, kw)
        info = save(res, out / f"{n}.npz", wall)
        info["load"] = os.getloadavg()[0]
        summary[n] = info
        print(f"[perf_scenarios_adv] {n}: wall {wall:.1f} s (load {info['load']:.1f})", flush=True)
        sfile.write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
