"""Per-tick timing of the compiled engine (docs/handoff/perf.md): wall and CPU time per world-tick, median of repeats.

    .venv/bin/python tools/perf_bench.py                       # B = 1 fixed (the 07 P0-9 case) + locked + B = 64
    .venv/bin/python tools/perf_bench.py --cases b1 --reps 7 --ticks 3000
    .venv/bin/python tools/perf_bench.py --set sh_tol_k=1e-7   # Plant keyword overrides (python literals)

Cases:
  b1      B = 1, fixed, open loop, g26 sampled world seeds 7, sim.reset_ramp_s = 1e-4 (tests/perf/test_perf.py),
          heaters at code_min (rings cold), 300 warm-up ticks
  b1lock  B = 1, fixed, open loop, g26 nominal world settled at the quasi-static lock codes (a locked bank)
  b64     B = 64, fixed, 8 threads (us per world-tick), seeds 7
Each case reports the median (and min, max) of ``--reps`` repeats of ``--ticks`` ticks, the CPU time per tick
(time.process_time, all threads of the process; for B = 1 the kernel runs in the calling thread), the load average
before and after, and the self-heating evaluations per ring and tick.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mrm import params  # noqa: E402
from mrm.control.api import ControllerSpec  # noqa: E402
from mrm.plant.plant import Plant  # noqa: E402
from mrm.scenarios import registry as scen_registry  # noqa: E402
from mrm.sim import kernel as K  # noqa: E402
from mrm.sim import runner  # noqa: E402


def _plant(B, kw, *, locked=False, policy="fixed"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if locked:
            cfg = params.load("g26")
            _, pp, _ = runner.load_inputs(cfg, batch=B, seeds=0, uncertainty="nominal", variation="off")
        else:
            cfg = params.load("g26", overrides=["sim.reset_ramp_s=1e-4"])
            pp = params.realize(cfg, B, seeds=7)
        sc = scen_registry.from_config(cfg)
        spec = ControllerSpec.from_config(cfg, n_batch=B, nominal=cfg, scenario_drives=sc.drives,
                                          scenario_offsets=sc.offsets)
        pl = Plant(pp, sc, spec, substep_policy=policy, histogram=False, **kw)
        if locked:
            codes = runner.warm_start_codes(pl, cfg, mode="quasi_static")
            pl.reset(protocol="branch_high", heater_code=codes, settle=True)
    return pl


def power_state() -> str:
    """'AC' / 'battery' and macOS Low Power Mode (it throttles the CPU: compare only runs under the same state)."""
    import subprocess
    try:
        batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        lpm = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
        src = "AC" if "AC Power" in batt else ("battery" if "Battery Power" in batt else "?")
        low = any(ln.split()[-1] == "1" for ln in lpm.splitlines() if "lowpowermode" in ln)
        return src + (", low power mode" if low else "")
    except Exception:
        return "?"


def measure(case, reps, n_ticks, kw):
    B = 64 if case == "b64" else 1
    nthr = 8 if B > 1 else 1
    pl = _plant(B, kw, locked=(case == "b1lock"))
    fn = pl._fn()
    done = np.zeros(B, np.int64)

    def adv(m):
        fn(K.null_step, K.null_step, pl.P, pl.S, np.int64(pl.tick), np.int64(m), np.int64(0), np.int64(0), done,
           n_threads=nthr)
        pl.tick += int(done[0])
    adv(300)
    walls, cpus = [], []
    it0 = pl.S.wi[:, K.WI_SH_ITER].sum()
    t_tot = 0
    load0 = os.getloadavg()
    for _ in range(reps):
        c = time.process_time()
        t = time.perf_counter()
        adv(n_ticks)
        w = time.perf_counter() - t
        c = time.process_time() - c
        walls.append(w / (n_ticks * B) * 1e6)
        cpus.append(c / (n_ticks * B) * 1e6)
        t_tot += n_ticks
    load1 = os.getloadavg()
    evals = (pl.S.wi[:, K.WI_SH_ITER].sum() - it0) / (t_tot * B * pl.N)
    return {"case": case, "B": B, "threads": nthr, "ticks": n_ticks, "reps": reps,
            "wall_us_median": float(np.median(walls)), "wall_us_min": float(np.min(walls)),
            "wall_us_max": float(np.max(walls)), "cpu_us_median": float(np.median(cpus)),
            "cpu_us_per_thread_median": float(np.median(cpus)) / (nthr if B > 1 else 1),
            "load_before": load0[0], "load_after": load1[0], "newton_iter_per_ring_tick": float(evals),
            "n_fail": int(pl.S.wi[:, K.WI_N_FAIL].sum()), "power": power_state()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="b1,b1lock,b64")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--json", default="")
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    kw = {}
    for s in a.set:
        k, v = s.split("=", 1)
        kw[k] = eval(v, {}, {})       # noqa: S307 (local tool, literal values)
    rows = []
    for case in a.cases.split(","):
        ticks = a.ticks if case != "b64" else max(a.ticks // 15, 100)
        r = measure(case, a.reps, ticks, kw)
        r["label"] = a.label
        r["set"] = a.set
        rows.append(r)
        print(f"[perf_bench] {a.label} {case}: wall {r['wall_us_median']:.2f} us/world-tick "
              f"(min {r['wall_us_min']:.2f}, max {r['wall_us_max']:.2f}), cpu {r['cpu_us_median']:.2f} us "
              f"(per thread {r['cpu_us_per_thread_median']:.2f}), newton it/ring/tick "
              f"{r['newton_iter_per_ring_tick']:.3f}, load {r['load_before']:.2f} -> {r['load_after']:.2f}, "
              f"power {r['power']}",
              flush=True)
    if a.json:
        p = Path(a.json)
        old = json.loads(p.read_text()) if p.exists() else []
        p.write_text(json.dumps(old + rows, indent=1))


if __name__ == "__main__":
    main()
