"""Break one plant tick down into the sections of the kernel source, timed with the CPU's virtual counter.

    .venv/bin/python tools/perf_tick_prof.py [--locked] [--n 20000] [--set key=value]

Read this first. The tool runs mrm.sim.kernel._world_block (open loop, the plant's substep policy) on the g26
B = 1 plant of tools/perf_bench.py. It adds up the time of every section of the kernel with the arm64 virtual
counter cntvct_el0 (24 MHz on Apple silicon: 41.7 ns resolution; over many ticks the quantization averages out).

The sections are the kernel's own source. _world_block and its tick loop _world_loop are compiled again from
mrm/sim/kernel.py (with the kernel's own runtime flag), with a counter read inserted at every comment line that
starts with ``# ---``. So the breakdown follows the kernel as it changes. arm64 only. The instrumented copy is
compiled on every run (about 1-2 min); the kernel itself comes from its cache. The instrumented functions receive
the accumulator as an extra argument named ``acc``: the kernel functions must not use a local of that name.
"""
from __future__ import annotations

import argparse
import inspect
import os
import platform
import re
import subprocess
import sys
import textwrap
import time
import warnings
from pathlib import Path

import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.extending import intrinsic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from mrm.sim import kernel  # noqa: E402
import perf_bench  # noqa: E402


@intrinsic
def cntvct(typingctx):
    """Read the arm64 virtual counter register cntvct_el0 (int64 counts; hw.tbfrequency counts per second)."""
    sig = types.int64()

    def codegen(context, builder, signature, args):
        mod = builder.module
        fnty = ir.FunctionType(ir.IntType(64), [ir.MetaDataType()])
        fn = mod.declare_intrinsic("llvm.read_volatile_register", [ir.IntType(64)], fnty)
        md = mod.add_metadata([ir.MetaDataString(mod, "cntvct_el0")])
        return builder.call(fn, [md])
    return sig, codegen


#: section labels; LABELS[i] names the accumulator slot acc[i] (filled by instrument, in compile order)
LABELS: list[str] = []


def _add_acc_to_calls(src: str, callee: str, new_name: str) -> str:
    """Rename calls of ``callee(`` to ``new_name(`` and append ``, acc`` to their argument lists."""
    out = []
    i = 0
    pat = re.compile(r"(?<![\w.])" + re.escape(callee) + r"\(")
    while True:
        m = pat.search(src, i)
        if not m:
            out.append(src[i:])
            break
        out.append(src[i:m.start()])
        j = m.end()
        depth = 1
        while depth:
            c = src[j]
            depth += (c == "(") - (c == ")")
            j += 1
        out.append(new_name + "(" + src[m.end():j - 1] + ", acc)")
        i = j
    return "".join(out)


def instrument(name: str, ns: dict, callees=()):
    """Compile kernel function ``name`` with a counter read at every comment line that starts with ``# ---`` (any
    indentation): the time from one marker to the next goes to the section the first one opens (``_cur``), so
    loops and branches are attributed correctly. ``callees``: (kernel name, instrumented function) pairs whose calls
    are redirected (their time is then itemized and not counted in the caller's section; the caller resumes its
    own clock after the call). Returns an njit function with an extra trailing ``acc`` argument."""
    src = textwrap.dedent(inspect.getsource(getattr(kernel, name).py_func))
    nrt = "_nrt=False" not in src.split("def ", 1)[0]      # compile as the kernel does (with or without the runtime)
    ns = dict(ns)
    for cname, cfun in callees:
        src = _add_acc_to_calls(src, cname, cname + "_i")
        ns[cname + "_i"] = cfun
    out = []
    LABELS.append(f"{name}: (setup)")
    cur0 = len(LABELS) - 1
    in_sig = False
    pending_resume = False
    for ln in src.split("\n"):
        if ln.startswith("@njit"):
            continue
        if ln.startswith("def "):
            in_sig = True
        if in_sig:
            if ln.rstrip().endswith("):"):
                out.append(ln[:ln.rstrip().rindex(")")] + ", acc):")
                out.append(f"    _tl = cntvct()\n    _cur = {cur0}")
                in_sig = False
            else:
                out.append(ln)
            continue
        mc = re.match(r"^(\s*)# ---\s*(.*)$", ln)
        if mc:
            ind = mc.group(1)
            LABELS.append(f"{name[:14]}: {mc.group(2)[:60]}")
            out.append(f"{ind}_t = cntvct(); acc[_cur] += _t - _tl; _tl = _t; _cur = {len(LABELS) - 1}")
        mr = re.match(r"^(\s*)return\b", ln)
        if mr:
            out.append(f"{mr.group(1)}_t = cntvct(); acc[_cur] += _t - _tl")
        out.append(ln)
    src2 = "\n".join(out)
    # an itemized callee's time must not count in the caller: restart the caller's clock after each such call
    for cname, _ in callees:
        lines = src2.split("\n")
        res = []
        k = 0
        while k < len(lines):
            ln = lines[k]
            res.append(ln)
            if cname + "_i(" in ln:
                ind = re.match(r"^(\s*)", ln).group(1)
                depth = ln.count("(") - ln.count(")")
                while depth > 0:
                    k += 1
                    res.append(lines[k])
                    depth += lines[k].count("(") - lines[k].count(")")
                if not ln.strip().startswith(("if ", "return")):
                    res.append(f"{ind}_tl = cntvct()")
            k += 1
        src2 = "\n".join(res)
    ns["cntvct"] = cntvct
    exec(compile(src2, f"<instrumented {name}>", "exec"), ns)
    return njit(ns[name]) if nrt else njit(_nrt=False)(ns[name])


def build():
    """The instrumented tick runner prof_ticks(P, S, b, k0, n, acc) -> next tick: n open-loop ticks of world b from
    tick k0, with the section times (counter counts) added to ``acc``. The implicit self-heating step and the Truth
    update are itemized separately from the tick loop."""
    ns = dict(kernel.__dict__)
    implicit_step = instrument("_implicit_step_eng", ns)
    truth = instrument("_truth_a", ns)
    loop = instrument("_world_loop", ns, callees=(("_implicit_step_eng", implicit_step), ("_truth_a", truth)))
    world_block = instrument("_world_block", ns, callees=(("_world_loop", loop),))

    @njit
    def prof_ticks(P, S, b, k0, n, acc):
        return k0 + world_block(kernel.null_step, kernel.null_step, b, k0, n, kernel.MODE_OPEN, 0, P, S, acc)
    return prof_ticks


def main(argv=None):
    if platform.machine() != "arm64":
        raise SystemExit("perf_tick_prof: arm64 only (cntvct_el0)")
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked", action="store_true")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--set", action="append", default=[])
    opts = ap.parse_args(argv)
    plant_kw = {}
    for s in opts.set:
        k, v = s.split("=", 1)
        plant_kw[k] = eval(v, {}, {})       # noqa: S307 (local tool, literal values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plant = perf_bench.bench_plant(1, plant_kw, locked=opts.locked)
    run_block = plant._fn()
    done = np.zeros(1, np.int64)             # ticks advanced per world (written by the kernel)
    run_block(kernel.null_step, kernel.null_step, plant.P, plant.S, np.int64(plant.tick), np.int64(300),
              np.int64(kernel.MODE_OPEN), np.int64(0), done)        # 300 warm-up ticks with the real kernel
    plant.tick += int(done[0])
    t_c = time.perf_counter()
    prof_ticks = build()
    # counter frequency, counts per s
    freq = float(subprocess.run(["sysctl", "-n", "hw.tbfrequency"], capture_output=True, text=True).stdout)
    # one short call compiles the instrumented copy (this is the slow part)
    acc = np.zeros(len(LABELS), np.int64)
    k = prof_ticks(plant.P, plant.S, np.int64(0), np.int64(plant.tick), np.int64(10), acc)
    t_c = time.perf_counter() - t_c
    res = []                                 # per repeat: us per tick of each section
    walls = []
    it = 0.0
    for _ in range(opts.reps):
        acc[:] = 0
        it0 = int(plant.S.world_int[0, kernel.WI_SH_ITER])
        t = time.perf_counter()
        k = prof_ticks(plant.P, plant.S, np.int64(0), np.int64(k), np.int64(opts.n), acc)
        walls.append((time.perf_counter() - t) / opts.n * 1e6)
        res.append(acc / freq / opts.n * 1e6)
        it = (int(plant.S.world_int[0, kernel.WI_SH_ITER]) - it0) / (opts.n * plant.N)
    res = np.median(np.array(res), axis=0)
    tot = res.sum()
    print(f"[perf_tick_prof] locked={opts.locked} n={opts.n} reps={opts.reps} load {os.getloadavg()[0]:.2f}; wall "
          f"{np.median(walls):.2f} us/tick, sections sum {tot:.2f} us, newton it/ring/tick {it:.3f} "
          f"(instrumented compile {t_c:.0f} s)")
    for name, v in zip(LABELS, res):
        if v > 0.0005:
            print(f"   {name:72s} {v:8.3f} us  {100 * v / tot:5.1f} %")


if __name__ == "__main__":
    main()
