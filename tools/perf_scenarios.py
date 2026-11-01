"""Run a fixed set of engine scenarios and save every signal, so that two engine versions can be compared exactly.

    .venv/bin/python tools/memguard.py --heavy -- .venv/bin/python tools/perf_scenarios.py runs/perf_baseline/ref
    .venv/bin/python tools/perf_scenarios.py runs/perf_x --only steady_pi,laser_step

Read this first. This is the accuracy check for engine performance work and for refactors. Each scenario runs
closed or open loop and writes ``<out>/<scenario>.npz``: every recorded trace, the summary metrics and some fields
of the final Truth. ``<out>/summary.json`` gets the wall time and the diagnostics. tools/perf_compare.py then
compares two output directories signal by signal; a change that must not alter the engine shows a difference of
exactly 0 everywhere.

Scenarios (g26 preset, B = 1 nominal world unless stated, fixed 1 us ticks, seeds 0):
  steady_pi       compiled PI lock (pi_lock_c) from the settled warm start at the steady-state codes, static, 40 ms
  a1_pi           pi_lock_c under the A1 slide-10 aggressor (phase P/2, default g26 site: 55 um from ring 0's PD),
                  one period (130.7 ms). The old baseline in runs/perf_baseline was recorded with the earlier site
                  (0, 30) um and the PD on the far side (ring.thermal.pd_offset_um=[0,-25]): record a new baseline
                  before you compare a1_pi again (docs/handoff/perf.md, "Layout change of 2026-10-10")
  laser_step      open loop at the settled steady-state codes, +10 % laser power on every line at 5 ms, 25 ms
  heater_sweep    cold start (branch_low, settled at code 0), heater ramp 0 -> 1.15 x lock code over 40 ms per
                  ring (Python ramp controller): crosses the self-heating fold (and foreign lines)
  idle_traffic    open loop at the steady-state codes, driver off (idle, no swing) until 10 ms, then traffic, 30 ms
  mc8_pi          B = 8 sampled worlds (uncertainty + variation sampled, seeds 1000), pi_lock_c, 10 ms

Fidelity overrides for an optimized engine can be passed with --set key=value (a Plant keyword, e.g. sh_tol_k).
MRM_PERF_THREADS sets the thread count of the multi-world scenario (default 4).
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
from mrm.scenarios import generators  # noqa: E402
from mrm.scenarios import library  # noqa: E402
from mrm.sim import runner  # noqa: E402
from mrm.sim.metrics import RecordSpec, envelope  # noqa: E402

#: the per-ring fields every scenario records (Truth, sensor and command values)
FIELDS = ("t_ring_k", "t_heater_k", "t_pd_k", "p_heater_w", "p_abs_w", "lam_res_m", "detuning_m", "p_drop_w",
          "i_monitor_a", "p_in_w", "oma_w", "reloma", "heater_code", "bias_v", "monitor_code", "oracle_reloma",
          "ber", "line_on")
SCENARIOS = ("steady_pi", "a1_pi", "laser_step", "heater_sweep", "idle_traffic", "mc8_pi")


def load_g26(extra=()):
    """The g26 config with the override strings ``extra`` (load warnings silenced)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load("g26", overrides=list(extra))


def record_specs(dec, reducer="last", fields=FIELDS, env=("detuning_m", "t_ring_k")):
    """RecordSpecs of ``fields`` every ``dec`` ticks with ``reducer``; when dec > 1, also the min/max envelope of
    the ``env`` fields (so a decimated trace still shows the extremes)."""
    specs = [RecordSpec(f, dec, reducer) for f in fields]
    if dec > 1:
        for f in env:
            specs += envelope(f, dec)
    return specs


class RampController:
    """Python open-loop heater ramp: code(t) = c_end * min(t / t_ramp, 1) on every ring (a cold-start sweep).

    ``c_end``: (N,) final heater codes. ``t_ramp_s``: ramp duration, s."""
    name = "ramp_sweep"
    latency_ticks = 0

    def __init__(self, c_end, t_ramp_s):
        self.c_end = np.asarray(c_end, float)
        self.t_ramp_s = float(t_ramp_s)

    def reset(self, spec):
        self.spec = spec

    def step(self, obs):
        t = obs.tick * obs.dt_s if hasattr(obs, "dt_s") else obs.tick * 1e-6
        frac = min(max(t / self.t_ramp_s, 0.0), 1.0)
        code = np.rint(self.c_end * frac).astype(np.int64)
        return {"tx.heater": code[None, :]}


def run_one(name: str, plant_kw: dict):
    """Run scenario ``name``; returns (runner result, wall time in s). ``plant_kw`` overrides the runner keywords."""
    t0 = time.perf_counter()
    common = dict(uncertainty="nominal", variation="off", seeds=0, batch=1)
    common.update(plant_kw)
    if name == "steady_pi":
        cfg = load_g26()
        res = runner.run(cfg, registry.create("pi_lock_c", {}), library.static(), duration_s=40e-3,
                         record=record_specs(1), heater_code="locked_steady_state", reset_protocol="branch_high",
                         settle=True, **common)
    elif name == "a1_pi":
        cfg = load_g26()
        res = runner.run(cfg, registry.create("pi_lock_c", {}),
                         library.a1_slide10_aggressor(phase_s=library.A1_PERIOD_S / 2), duration_s=library.A1_PERIOD_S,
                         record=record_specs(10, "mean"), heater_code="locked_steady_state",
                         reset_protocol="branch_high", settle=True, **common)
    elif name == "laser_step":
        cfg = load_g26()
        scenario = library.static() + generators.Scenario(name="pstep", components=(
            generators.step("laser.power_delta_db", 5e-3, 10.0 * math.log10(1.1), label="pstep",
                            provenance="assumed", source="", note="perf baseline +10 % laser power"),))
        res = runner.run(cfg, None, scenario, duration_s=25e-3, record=record_specs(1),
                         heater_code="locked_steady_state", reset_protocol="branch_high", settle=True, **common)
    elif name == "heater_sweep":
        cfg = load_g26()
        # the steady-state lock codes of the nominal world, only to size the end of the ramp
        from mrm.plant.plant import Plant
        from mrm.control.api import ControllerSpec
        _, pp, _ = runner.load_inputs(cfg, batch=1, seeds=0, uncertainty="nominal", variation="off")
        spec = ControllerSpec.from_config(cfg, n_batch=1, nominal=cfg)
        codes = runner.warm_start_codes(Plant(pp, None, spec), cfg, mode="locked_steady_state")
        c_end = np.minimum(np.asarray(codes, float)[0] * 1.15, 65535.0)        # 65535: g26 ring.heater.code_max
        res = runner.run(cfg, RampController(c_end, 40e-3), library.static(), duration_s=42e-3,
                         record=record_specs(1), reset_protocol="branch_low", settle=True, **common)
    elif name == "idle_traffic":
        cfg = load_g26()
        res = runner.run(cfg, None, library.traffic_start_stop(t_start_s=10e-3), duration_s=30e-3,
                         record=record_specs(1), heater_code="locked_steady_state", reset_protocol="branch_high",
                         settle=True, **common)
    elif name == "mc8_pi":
        cfg = load_g26()
        common.update(uncertainty="sample", variation="sample", seeds=1000, batch=8)
        res = runner.run(cfg, registry.create("pi_lock_c", {}), library.static(), duration_s=10e-3,
                         record=record_specs(10, "mean"), heater_code="locked_steady_state",
                         reset_protocol="branch_high", settle=True,
                         n_threads=int(os.environ.get("MRM_PERF_THREADS", "4")), **common)
    else:
        raise KeyError(name)
    wall = time.perf_counter() - t0
    return res, wall


def save(res, path: Path, wall: float):
    """Write the traces (``trace/<key>/t_s`` and ``/value``), the numeric metrics (``metric/...``) and six final
    Truth fields (``final/...``) to the npz ``path``; returns the summary.json entry {wall_s, diagnostics}."""
    out = {}
    for key, tr in (res.trace or {}).items():
        out[f"trace/{key}/t_s"] = np.asarray(tr["t_s"])
        out[f"trace/{key}/value"] = np.asarray(tr["value"])
    for key, v in res.metrics.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                arr = np.asarray(v2)
                if arr.dtype.kind in "fiub":           # numbers and booleans only
                    out[f"metric/{key}/{k2}"] = arr
        else:
            arr = np.asarray(v)
            if arr.dtype.kind in "fiub":
                out[f"metric/{key}"] = arr
    truth_final = res.truth_final
    for f in ("t_ring_k", "detuning_m", "p_abs_w", "heater_code", "i_monitor_a", "monitor_code"):
        try:
            out[f"final/{f}"] = np.asarray(getattr(truth_final, f))
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
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [s for s in args.only.split(",") if s] or list(SCENARIOS)
    plant_kw = {}
    for s in args.set:
        k, v = s.split("=", 1)
        plant_kw[k] = eval(v, {}, {})       # noqa: S307 (local tool, literal values)
    summary = {}
    summary_file = out / "summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
    for name in names:
        res, wall = run_one(name, plant_kw)
        info = save(res, out / f"{name}.npz", wall)
        info["load"] = os.getloadavg()[0]
        summary[name] = info
        print(f"[perf_scenarios] {name}: wall {wall:.1f} s (load {info['load']:.1f})", flush=True)
        summary_file.write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
