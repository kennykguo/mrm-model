"""HDL check of the golden vectors and the fixed-point reference (07 P1-21: "fixed point against RTL, bit-exact"),
with Icarus Verilog only.

07 P1-21 names cocotb (with Icarus or Verilator) and a Verilator closed-loop co-simulation. Neither cocotb nor
Verilator is installed here, and this task installs nothing, so that part is NOT built (documented stretch,
docs/handoff/hardware.md). What runs instead, when ``iverilog`` and ``vvp`` are on the PATH (skipped otherwise):

- tests/hdl/tb_fxp_pi_replay.v reads the golden vectors with ``$readmemh`` (the .hex files and the
  ``.layout.vh`` include that mrm.sim.vectors.record_vectors writes), sequences FxpPi per ring and world (capture,
  hold, update on a new valid sample) around the synthesizable tests/hdl/fxp_pi_update.v, and compares every heater
  action with the recorded one: 0 mismatches;
- tests/hdl/tb_fxp_pi_update_random.v checks fxp_pi_update on random operands (the accumulator saturation, the
  integrator clamp and the output clamp included) against mrm.control.fxp.fx_pi_update, for RND = floor and
  half_up;
- negative controls: one edited expected word gives exactly one mismatch;
- with ``yosys`` on the PATH: fxp_pi_update.v synthesizes to combinational logic (no latch, no flip-flop).
"""
from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mrm.control import api, fxp
from mrm.control.fxp import QFormat, Rounding
from mrm.sim import vectors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GOLDEN_VEC = ROOT / "tests" / "golden" / "vectors" / "fxp_pi__g26__aggressor_square.npz"
IVERILOG, VVP = shutil.which("iverilog"), shutil.which("vvp")
pytestmark = pytest.mark.skipif(IVERILOG is None or VVP is None, reason="Icarus Verilog (iverilog, vvp) not found")


def _sim(tmp: Path, top: str, sources, defines: dict, includes=()) -> str:
    exe = tmp / f"{top}.vvp"
    cmd = [IVERILOG, "-g2012", "-Wall", "-s", top, "-o", str(exe)]
    for d in includes:
        cmd += ["-I", str(d)]
    for k, v in defines.items():
        cmd.append(f"-D{k}={v}")
    cmd += [str(s) for s in sources]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run([VVP, "-n", str(exe)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def _replay(tmp: Path, vec: vectors.VectorStream) -> tuple[int, int, str]:
    paths = vectors.record_vectors(vec, tmp / "vec")
    p = vec.meta["controller_params"]
    si = vec.spec_i64
    defines = {"OBS_HEX": f'"{paths["obs_hex"]}"', "ACT_HEX": f'"{paths["act_hex"]}"', "KP": p["kp_raw"],
               "KI": p["ki_raw"], "FRAC": p["frac_bits"], "ACC": p["acc_bits"],
               "KW": QFormat.parse(p["gain_format"]).bits,
               "RND": int(Rounding[p["rounding"].upper()]), "LO": int(si[api.SPEC_I_HEATER_MIN]),
               "HI": int(si[api.SPEC_I_HEATER_MAX])}
    out = _sim(tmp, "tb_fxp_pi_replay", [HERE / "tb_fxp_pi_replay.v", HERE / "fxp_pi_update.v"], defines,
               includes=[tmp])
    line = [x for x in out.splitlines() if x.startswith("CHECKED")][-1].split()
    return int(line[1]), int(line[3]), out


def test_golden_vectors_replay_on_icarus(tmp_path):
    vec = vectors.load_vectors(GOLDEN_VEC)
    assert np.flatnonzero(vec.reset_before).tolist() == [0]           # the test bench resets at k = 0 only
    assert vec.meta["controller_params"]["setpoint_codes"] is None    # capture mode (what the bench sequences)
    checked, mism, out = _replay(tmp_path, vec)
    assert checked == vec.n_records * vec.n_batch * 8
    assert mism == 0, out[-3000:]


def test_replay_bench_detects_one_wrong_word(tmp_path):
    vec = vectors.load_vectors(GOLDEN_VEC)
    off = vec.field("tx.heater", "act")["offset"]
    vec.act = vec.act.copy()
    vec.act[123, 0, off + 5] += 1
    checked, mism, _ = _replay(tmp_path, vec)
    assert mism == 1


@pytest.mark.parametrize("rnd", [Rounding.FLOOR, Rounding.HALF_UP])
def test_pi_update_random_operands_on_icarus(tmp_path, rnd):
    rng = random.Random(1234 + int(rnd))
    frac, acc, kw = 20, 30, 24                       # a 30-bit accumulator so its saturation is reached often
    stim, exp = [], []
    for _ in range(3000):
        lo = rng.choice((0, rng.randint(0, 30000)))
        hi = rng.choice((65535, rng.randint(lo, 65535)))
        u0 = rng.choice((lo, hi, rng.randint(lo, hi)))
        e = rng.randint(-65535, 65535) if rng.random() < 0.6 else rng.randint(-40, 40)
        integ = rng.randint(-(1 << (acc - 1)), (1 << (acc - 1)) - 1)
        kp = rng.randint(-(1 << (kw - 1)), (1 << (kw - 1)) - 1)
        ki = rng.randint(-(1 << (kw - 1)), (1 << (kw - 1)) - 1)
        u, a = fxp.fx_pi_update(e, integ, kp, ki, u0, frac, acc, lo, hi, int(rnd))
        stim += [e, integ, kp, ki, u0, lo, hi]
        exp += [int(u), int(a)]
    sp = vectors.write_hex(tmp_path / "stim.hex", np.array(stim, dtype=np.int64), ["stimulus"])
    ep = vectors.write_hex(tmp_path / "exp.hex", np.array(exp, dtype=np.int64), ["expected"])
    out = _sim(tmp_path, "tb_fxp_pi_update_random", [HERE / "tb_fxp_pi_update_random.v", HERE / "fxp_pi_update.v"],
               {"STIM_HEX": f'"{sp}"', "EXP_HEX": f'"{ep}"', "NCASE": 3000, "FRAC": frac, "ACC": acc, "KW": kw,
                "RND": int(rnd)})
    line = [x for x in out.splitlines() if x.startswith("CASES")][-1].split()
    assert line[1] == "3000" and line[3] == "0", out[-3000:]
    a = np.array(exp[1::2])
    assert (a == (1 << (acc - 1)) - 1).any() and (a == -(1 << (acc - 1))).any()   # accumulator saturation hit
    assert (np.array(exp[0::2]) == 0).any() and (np.array(exp[0::2]) == 65535).any()


@pytest.mark.skipif(shutil.which("yosys") is None, reason="yosys not found")
def test_pi_update_is_synthesizable(tmp_path):
    """fxp_pi_update.v synthesizes to combinational logic (generic gates, no latch, no flip-flop). Measured
    2026-10-10 with the default parameters: ~7500 generic cells (Yosys `synth`, no technology mapping; the
    64-bit multipliers dominate; a real design would size them to 17 x 24 bits)."""
    stat = tmp_path / "stat.txt"
    r = subprocess.run([shutil.which("yosys"), "-q", "-p",
                        f"read_verilog -sv {HERE / 'fxp_pi_update.v'}; synth -top fxp_pi_update; tee -o {stat} stat"],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    text = stat.read_text()
    assert "LATCH" not in text.upper() and "_DFF" not in text.upper(), text
    cells = int([ln for ln in text.splitlines() if ln.strip().endswith("cells")][0].split()[0])
    assert cells > 100
    print(f"\nfxp_pi_update: {cells} generic cells (yosys synth)")
