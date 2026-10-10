"""Fixed accuracy-regression scenarios for engine performance work (docs/handoff/perf.md).

Runs a fixed set of closed- and open-loop scenarios on the engine and saves every recorded trace, the summary
metrics and the final Truth to ``<out>/<scenario>.npz`` (+ ``<out>/summary.json``), so an optimized engine can be
compared with the baseline signal by signal (tools/perf_compare.py).

    .venv/bin/python tools/memguard.py --heavy -- .venv/bin/python tools/perf_scenarios.py runs/perf_baseline/ref
    .venv/bin/python tools/perf_scenarios.py runs/perf_x --only steady_pi,laser_step

Scenarios (g26, B = 1 nominal world unless stated, fixed 1 us ticks, seeds 0):
  steady_pi       compiled PI lock (pi_lock_c) from the settled warm start at the quasi-static codes, static, 40 ms
  a1_pi           pi_lock_c under the A1 slide-10 aggressor (phase P/2, PD on the far side), one period (130.7 ms)
  laser_step      open loop at the settled quasi-static codes, +10 % laser power on every line at 5 ms, 25 ms
  heater_sweep    cold start (branch_low, settled at code 0), heater ramp 0 -> 1.15 x lock code over 40 ms per
                  ring (Python ramp controller): crosses the self-heating fold (and foreign lines)
  idle_traffic    open loop at the quasi-static codes, driver off (idle, no swing) until 10 ms, then traffic, 30 ms
  mc8_pi          B = 8 sampled worlds (uncertainty + variation sampled, seeds 1000), pi_lock_c, 10 ms
Fidelity overrides for the optimized engine can be passed with --set key=value (Plant keyword, e.g. sh_tol_k).
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

from mrm import params  # noqa: E402
from mrm.control import registry  # noqa: E402
from mrm.scenarios import generators as g  # noqa: E402
from mrm.scenarios import library as lib  # noqa: E402
from mrm.sim import runner  # noqa: E402
from mrm.sim.metrics import RecordSpec, envelope  # noqa: E402

FIELDS = ("t_ring_k", "t_heater_k", "t_pd_k", "p_heater_w", "p_abs_w", "lam_res_m", "detuning_m", "p_drop_w",
          "i_monitor_a", "p_in_w", "oma_w", "reloma", "heater_code", "bias_v", "monitor_code", "oracle_reloma",
          "ber", "line_on")
PD_FAR = ("ring.thermal.pd_offset_um=[0.0,-25.0]",)
SCENARIOS = ("steady_pi", "a1_pi", "laser_step", "heater_sweep", "idle_traffic", "mc8_pi")


def _cfg(extra=()):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load("g26", overrides=list(extra))


def _rec(dec, reducer="last", fields=FIELDS, env=("detuning_m", "t_ring_k")):
    r = [RecordSpec(f, dec, reducer) for f in fields]
    if dec > 1:
        for f in env:
            r += envelope(f, dec)
    return r


class RampController:
    """Python open-loop heater ramp: code(t) = c_end * min(t / t_ramp, 1) on every ring (a cold-start sweep)."""
    name = "ramp_sweep"
    latency_ticks = 0

    def __init__(self, c_end, t_ramp_s):
        self.c_end = np.asarray(c_end, float)
        self.t_ramp_s = float(t_ramp_s)

    def reset(self, spec):
        self.spec = spec

    def step(self, obs):
        t = obs.tick * obs.dt_s if hasattr(obs, "dt_s") else obs.tick * 1e-6
        f = min(max(t / self.t_ramp_s, 0.0), 1.0)
        c = np.rint(self.c_end * f).astype(np.int64)
        return {"tx.heater": c[None, :]}


def run_one(name: str, plant_kw: dict):
    t0 = time.perf_counter()
    common = dict(uncertainty="nominal", variation="off", seeds=0, batch=1)
    common.update(plant_kw)
    if name == "steady_pi":
        cfg = _cfg()
        res = runner.run(cfg, registry.create("pi_lock_c", {}), lib.static(), duration_s=40e-3,
                         record=_rec(1), heater_code="quasi_static", reset_protocol="branch_high", settle=True,
                         **common)
    elif name == "a1_pi":
        cfg = _cfg(PD_FAR)
        res = runner.run(cfg, registry.create("pi_lock_c", {}), lib.a1_slide10_aggressor(phase_s=lib.A1_PERIOD_S / 2),
                         duration_s=lib.A1_PERIOD_S, record=_rec(10, "mean"), heater_code="quasi_static",
                         reset_protocol="branch_high", settle=True, **common)
    elif name == "laser_step":
        cfg = _cfg()
        sc = lib.static() + g.Scenario(name="pstep", components=(
            g.step("laser.power_delta_db", 5e-3, 10.0 * math.log10(1.1), label="pstep", provenance="assumed",
                   source="", note="perf baseline +10 % laser power"),))
        res = runner.run(cfg, None, sc, duration_s=25e-3, record=_rec(1), heater_code="quasi_static",
                         reset_protocol="branch_high", settle=True, **common)
    elif name == "heater_sweep":
        cfg = _cfg()
        # the lock codes of the nominal world (quasi-static), only to size the ramp end
        from mrm.plant.plant import Plant
        from mrm.control.api import ControllerSpec
        _, pp, _ = runner.load_inputs(cfg, batch=1, seeds=0, uncertainty="nominal", variation="off")
        spec = ControllerSpec.from_config(cfg, n_batch=1, nominal=cfg)
        codes = runner.warm_start_codes(Plant(pp, None, spec), cfg, mode="quasi_static")
        c_end = np.minimum(np.asarray(codes, float)[0] * 1.15, 65535.0)
        res = runner.run(cfg, RampController(c_end, 40e-3), lib.static(), duration_s=42e-3, record=_rec(1),
                         reset_protocol="branch_low", settle=True, **common)
    elif name == "idle_traffic":
        cfg = _cfg()
        res = runner.run(cfg, None, lib.traffic_start_stop(t_start_s=10e-3), duration_s=30e-3, record=_rec(1),
                         heater_code="quasi_static", reset_protocol="branch_high", settle=True, **common)
    elif name == "mc8_pi":
        cfg = _cfg()
        common.update(uncertainty="sample", variation="sample", seeds=1000, batch=8)
        res = runner.run(cfg, registry.create("pi_lock_c", {}), lib.static(), duration_s=10e-3,
                         record=_rec(10, "mean"), heater_code="quasi_static", reset_protocol="branch_high",
                         settle=True, n_threads=int(os.environ.get("MRM_PERF_THREADS", "4")), **common)
    else:
        raise KeyError(name)
    wall = time.perf_counter() - t0
    return res, wall


def save(res, path: Path, wall: float):
    out = {}
    for key, tr in (res.trace or {}).items():
        out[f"trace/{key}/t_s"] = np.asarray(tr["t_s"])
        out[f"trace/{key}/value"] = np.asarray(tr["value"])
    for key, v in res.metrics.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                a = np.asarray(v2)
                if a.dtype.kind in "fiub":
                    out[f"metric/{key}/{k2}"] = a
        else:
            a = np.asarray(v)
            if a.dtype.kind in "fiub":
                out[f"metric/{key}"] = a
    tf = res.truth_final
    for f in ("t_ring_k", "detuning_m", "p_abs_w", "heater_code", "i_monitor_a", "monitor_code"):
        try:
            out[f"final/{f}"] = np.asarray(getattr(tf, f))
        except Exception:
            pass
    diag = {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in res.diagnostics.items()}
    np.savez_compressed(path, **out)
    return {"wall_s": wall, "diagnostics": diag}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--only", default="")
    ap.add_argument("--set", action="append", default=[], help="Plant/runner keyword key=value (python literal)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [s for s in a.only.split(",") if s] or list(SCENARIOS)
    kw = {}
    for s in a.set:
        k, v = s.split("=", 1)
        kw[k] = eval(v, {}, {})       # noqa: S307 (local tool, literal values)
    summary = {}
    sfile = out / "summary.json"
    if sfile.exists():
        summary = json.loads(sfile.read_text())
    for n in names:
        res, wall = run_one(n, kw)
        info = save(res, out / f"{n}.npz", wall)
        info["load"] = os.getloadavg()[0]
        summary[n] = info
        print(f"[perf_scenarios] {n}: wall {wall:.1f} s (load {info['load']:.1f})", flush=True)
        sfile.write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
