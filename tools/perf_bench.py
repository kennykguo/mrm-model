"""Time the compiled engine per tick: wall and CPU time per world-tick, median of repeats.

    .venv/bin/python tools/perf_bench.py                       # cases b1, b1lock and b64
    .venv/bin/python tools/perf_bench.py --cases b1 --reps 7 --ticks 3000
    .venv/bin/python tools/perf_bench.py --set sh_tol_k=1e-7   # Plant keyword overrides (python literals)

Read this first. Each case builds one g26 plant, runs it open loop (no controller) for 300 warm-up ticks, then
times ``--reps`` repeats of ``--ticks`` ticks with the compiled kernel. The cases:

  b1      B = 1 world, fixed substeps, g26 sampled world with seed 7, sim.reset_ramp_s = 1e-4, heaters at code_min
          (rings cold). This is the compute-budget case of tests/perf/test_perf.py.
  b1lock  B = 1, fixed substeps, g26 nominal world settled at the steady-state lock codes (a locked bank)
  b64     B = 64 worlds, fixed substeps, 8 threads, seeds 7; the time is per world-tick (us per tick / 64)

Each case reports the median (and min, max) wall time per world-tick in us, the CPU time per world-tick
(time.process_time, all threads of the process; for B = 1 the kernel runs in the calling thread), the load average
before and after, and the self-heating Newton iterations per ring and tick. tools/perf_ab.py parses the printed
line: keep its format.
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
from mrm.scenarios import registry as scenario_registry  # noqa: E402
from mrm.sim import kernel  # noqa: E402
from mrm.sim import runner  # noqa: E402

WARMUP_TICKS = 300          # open-loop ticks before the timed repeats


def bench_plant(n_worlds, plant_kw, *, locked=False, policy="fixed"):
    """The g26 Plant of a case (also used by tools/perf_profile.py and tools/perf_tick_prof.py).

    locked=False: ``n_worlds`` sampled worlds (seeds 7), sim.reset_ramp_s = 1e-4, heaters at code_min.
    locked=True: the nominal world, reset on the hot branch and settled at the steady-state lock codes.
    ``plant_kw``: extra Plant keywords (fidelity overrides such as sh_tol_k)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if locked:
            cfg = params.load("g26")
            _, pp, _ = runner.load_inputs(cfg, batch=n_worlds, seeds=0, uncertainty="nominal", variation="off")
        else:
            cfg = params.load("g26", overrides=["sim.reset_ramp_s=1e-4"])
            pp = params.realize(cfg, n_worlds, seeds=7)
        scenario = scenario_registry.from_config(cfg)
        spec = ControllerSpec.from_config(cfg, n_batch=n_worlds, nominal=cfg, scenario_drives=scenario.drives,
                                          scenario_offsets=scenario.offsets)
        plant = Plant(pp, scenario, spec, substep_policy=policy, histogram=False, **plant_kw)
        if locked:
            codes = runner.warm_start_codes(plant, cfg, mode="locked_steady_state")
            plant.reset(protocol="branch_high", heater_code=codes, settle=True)
    return plant


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


def measure(case, reps, n_ticks, plant_kw):
    """Time one case: {case, B, threads, ticks, reps, wall_us_*, cpu_us_*, load_*, newton_iter_per_ring_tick,
    n_fail, power}. Times are us per world-tick."""
    n_worlds = 64 if case == "b64" else 1
    n_threads = 8 if n_worlds > 1 else 1
    plant = bench_plant(n_worlds, plant_kw, locked=(case == "b1lock"))
    run_block = plant._fn()
    done = np.zeros(n_worlds, np.int64)          # ticks advanced per world (written by the kernel)

    def advance(n):
        """n open-loop ticks with no controller (both step functions are the kernel's null_step)."""
        run_block(kernel.null_step, kernel.null_step, plant.P, plant.S, np.int64(plant.tick), np.int64(n),
                  np.int64(kernel.MODE_OPEN), np.int64(0), done, n_threads=n_threads)
        plant.tick += int(done[0])
    advance(WARMUP_TICKS)
    walls, cpus = [], []
    newton_iter0 = plant.S.world_int[:, kernel.WI_SH_ITER].sum()
    ticks_total = 0
    load0 = os.getloadavg()
    for _ in range(reps):
        cpu = time.process_time()
        t = time.perf_counter()
        advance(n_ticks)
        wall = time.perf_counter() - t
        cpu = time.process_time() - cpu
        walls.append(wall / (n_ticks * n_worlds) * 1e6)
        cpus.append(cpu / (n_ticks * n_worlds) * 1e6)
        ticks_total += n_ticks
    load1 = os.getloadavg()
    newton_per_ring_tick = ((plant.S.world_int[:, kernel.WI_SH_ITER].sum() - newton_iter0)
                            / (ticks_total * n_worlds * plant.N))
    return {"case": case, "B": n_worlds, "threads": n_threads, "ticks": n_ticks, "reps": reps,
            "wall_us_median": float(np.median(walls)), "wall_us_min": float(np.min(walls)),
            "wall_us_max": float(np.max(walls)), "cpu_us_median": float(np.median(cpus)),
            "cpu_us_per_thread_median": float(np.median(cpus)) / (n_threads if n_worlds > 1 else 1),
            "load_before": load0[0], "load_after": load1[0], "newton_iter_per_ring_tick": float(newton_per_ring_tick),
            "n_fail": int(plant.S.world_int[:, kernel.WI_N_FAIL].sum()), "power": power_state()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="b1,b1lock,b64")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--json", default="")
    ap.add_argument("--label", default="")
    args = ap.parse_args(argv)
    plant_kw = {}
    for s in args.set:
        k, v = s.split("=", 1)
        plant_kw[k] = eval(v, {}, {})       # noqa: S307 (local tool, literal values)
    rows = []
    for case in args.cases.split(","):
        ticks = args.ticks if case != "b64" else max(args.ticks // 15, 100)     # b64 does 64 worlds per tick
        r = measure(case, args.reps, ticks, plant_kw)
        r["label"] = args.label
        r["set"] = args.set
        rows.append(r)
        print(f"[perf_bench] {args.label} {case}: wall {r['wall_us_median']:.2f} us/world-tick "
              f"(min {r['wall_us_min']:.2f}, max {r['wall_us_max']:.2f}), cpu {r['cpu_us_median']:.2f} us "
              f"(per thread {r['cpu_us_per_thread_median']:.2f}), newton it/ring/tick "
              f"{r['newton_iter_per_ring_tick']:.3f}, load {r['load_before']:.2f} -> {r['load_after']:.2f}, "
              f"power {r['power']}",
              flush=True)
    if args.json:
        p = Path(args.json)
        old = json.loads(p.read_text()) if p.exists() else []
        p.write_text(json.dumps(old + rows, indent=1))


if __name__ == "__main__":
    main()
