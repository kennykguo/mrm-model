"""Checks the C reference controller that talks to the plant over the binary bridge.

Read this first
---------------
mrm.sim.bridge.C_FXP_PI_SOURCE is a C twin of mrm.control.fxp.FxpPi. The tests compile it against the header that
mrm.sim.bridge.c_header generates from the ChannelSpecs, and then check:

- a closed loop over pipes is bit-identical to FxpPi in the kernel (both frame profiles);
- the committed golden vectors, replayed through the C program, give every heater action bit-exact;
- a synthetic stress stream gives bit-exact actions in the C program and in FxpPi's compiled ABI. The stream has
  random monitor codes, missing and invalid samples, large gains, a 26-bit accumulator and every rounding mode. It
  drives the accumulator saturation, the integrator clamp and the output clamp;
- the C side refuses a header for another frame layout or another ControllerSpec.

The round-trip cost of the binary protocol (C peer over pipes) is printed. The file is skipped when no C compiler
(``cc``) is on the PATH.
"""
from __future__ import annotations

import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

from mrm import params
from mrm.control import api, fxp
from mrm.control.fxp import FxpPi, Rounding
from mrm.scenarios import library
from mrm.sim import bridge, runner, vectors
from mrm.sim.metrics import RecordSpec

CC = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
pytestmark = pytest.mark.skipif(CC is None, reason="no C compiler on the PATH")
ROOT = Path(__file__).resolve().parents[2]
GOLDEN_VEC = ROOT / "tests" / "golden" / "vectors" / "fxp_pi__g26__aggressor_square.npz"
RUN = dict(uncertainty="nominal", variation="off", seeds=0, heater_code="locked_steady_state", settle=True)
REC = [RecordSpec("detuning_m", 1, "last"), RecordSpec("heater_code", 10, "last")]


def scenario():
    return library.aggressor_direct(waveform="square", power_w=0.05, period_s=2e-3, duty=0.5, phase_s=1e-3)


@pytest.fixture(scope="module")
def cfg():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load("g26")


@pytest.fixture(scope="module")
def spec(cfg):
    return bridge.spec_for(cfg, scenario(), 1)


@pytest.fixture(scope="module")
def build(tmp_path_factory, spec):
    """Compile the C controller once per profile: {profile: executable}."""
    out = {}
    for profile in ("full", "minimal"):
        d = tmp_path_factory.mktemp(f"c_{profile}")
        (d / "mrm_bridge.h").write_text(bridge.c_header(bridge.FrameLayout.from_spec(spec, profile),
                                                        spec_fingerprint=spec.fingerprint()))
        (d / "fxp_pi.c").write_text(bridge.C_FXP_PI_SOURCE)
        exe = d / "fxp_pi"
        r = subprocess.run([CC, "-O2", "-std=c99", "-Wall", "-Wextra", "-Werror", "-o", str(exe),
                            str(d / "fxp_pi.c"), "-I", str(d)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        out[profile] = exe
    return out


def c_args(c: FxpPi, setpoint=None):
    """Command-line arguments of the C program for the same gains, formats and rounding as ``c``."""
    a = [str(c.kp_raw), str(c.ki_raw), str(c.frac_bits), str(c.acc_format.bits), str(int(c.rounding))]
    return a + ([str(setpoint)] if setpoint is not None else [])


@pytest.mark.parametrize("profile", ["full", "minimal"])
def test_c_controller_closed_loop_is_bit_identical(cfg, build, profile):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = runner.run(cfg, FxpPi(), scenario(), duration_s=3e-3, record=REC, **RUN)
    c = FxpPi()
    with bridge.BridgeController(bridge.spawn([str(build[profile]), *c_args(c)]), profile=profile,
                                 keep_times=True) as bc:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = runner.run(cfg, bc, scenario(), duration_s=3e-3, record=REC, **RUN)
    assert bc.remote_name == "fxp_pi_c"
    for k in ref.trace:
        assert np.array_equal(ref.trace[k]["value"], res.trace[k]["value"]), k
    for k in ("max_abs_detuning_pm", "rms_detuning_pm", "heater_range_used_codes", "heater_energy_j"):
        assert np.array_equal(ref.metrics[k], res.metrics[k]), k


def _drive(exe_args, spec, obs, ticks, profile="full"):
    """Feed obs_i64 records (K, B, obs_size) to a C controller over the bridge; returns its heater codes (K, B, N)
    (held codes where it wrote none) and the BridgeController."""
    K = obs.shape[0]
    out = np.zeros((K, spec.n_batch, spec.n_rings), dtype=np.int64)
    bc = bridge.BridgeController(bridge.spawn(exe_args), profile=profile, keep_times=True)
    try:
        bc.reset(spec)
        held = None
        for k in range(K):
            acts = bc.step(vectors.observation_from_flat(spec, obs[k], int(ticks[k])))
            if "tx.heater" in acts:
                held = np.asarray(acts["tx.heater"], dtype=np.int64)
            out[k] = held
    finally:
        bc.close()
    return out, bc


def test_golden_vectors_replay_through_c(build, spec):
    v = vectors.load_vectors(GOLDEN_VEC)
    assert v.spec_fingerprint == spec.fingerprint()
    got, bc = _drive([str(build["full"]), *c_args(FxpPi())], spec, v.obs, v.ticks)
    exp = v.channel("tx.heater", "act")
    n = int((got != exp).sum())
    assert n == 0, f"{n} heater actions differ from the golden vectors"
    rt = np.median(bc.rt_samples[10:]) * 1e6
    print(f"\nC peer over pipes: median round trip {rt:.1f} us per controller edge "
          f"(encode {1e6 * bc.stats['encode_s'] / bc.stats['steps']:.1f} us, "
          f"decode {1e6 * bc.stats['decode_s'] / bc.stats['steps']:.1f} us per edge on the plant side)")


def _synthetic_obs(spec, K, rng):
    """obs_i64 records with random monitor codes (sometimes extreme), ~20 % missing and ~5 % invalid samples, and
    the latched heater register at a random start code."""
    lay = spec.layout
    obs = np.zeros((K, 1, lay.obs_size), dtype=np.int64)
    f_lat, f_new = lay.obs_field("tx.monitor.latest"), lay.new_field("tx.monitor.latest")
    f_val, f_app = lay.obs_field("monitor_valid"), lay.obs_field("tx.heater.applied")
    u0 = rng.integers(0, 65536, size=8)
    u0[:4] = (0, 3, 65532, 65535)                    # rings whose integrator clamp is tighter than the accumulator
    y = 30000 + np.cumsum(rng.integers(-3000, 3001, size=(K, 8)), axis=0)
    y[rng.random((K, 8)) < 0.05] = 0
    y[rng.random((K, 8)) < 0.05] = 65535
    y = np.clip(y, 0, 65535)
    for k in range(K):
        obs[k, 0, 0] = 10 * k
        obs[k, 0, f_lat.offset:f_lat.offset + 8] = y[k]
        obs[k, 0, f_new.offset:f_new.offset + 8] = rng.random(8) > 0.2
        obs[k, 0, f_val.offset:f_val.offset + 8] = rng.random(8) > 0.05
        obs[k, 0, f_app.offset:f_app.offset + 8] = u0
    return obs, 10 * np.arange(K)


@pytest.mark.parametrize("rnd", list(Rounding))
def test_synthetic_stress_stream_c_equals_python(build, spec, rnd):
    rng = np.random.default_rng(int(rnd) + 11)
    obs, ticks = _synthetic_obs(spec, 1500, rng)
    c = FxpPi(kp=3.7, ki=1.9, gain_format="Q3.20", acc_format="Q5.20", rounding=rnd.name.lower())
    c.reset(spec)
    p, pf, s, sf = fxp.fxp_pi_init(c.spec_i64, c.spec_f64)
    act = c.initial_act(spec)[0].copy()
    f_act = spec.layout.act_field("tx.heater")
    exp = np.zeros((obs.shape[0], 1, 8), dtype=np.int64)
    # FxpPi's state array: a start flag, then blocks of one value per ring (8 rings); one block holds the integrators
    integrators = slice(fxp._S_BLOCKS + fxp._S_INTEGRATOR * 8, fxp._S_BLOCKS + (fxp._S_INTEGRATOR + 1) * 8)
    integ_seen = []
    for k in range(obs.shape[0]):
        fxp.fxp_pi_step(p, pf, s, sf, np.ascontiguousarray(obs[k, 0]), act)
        exp[k, 0] = act[f_act.offset:f_act.offset + 8]
        integ_seen.append(s[integrators].copy())
    integ_seen = np.array(integ_seen)
    got, _ = _drive([str(build["full"]), *c_args(c)], spec, obs, ticks)
    assert np.array_equal(got, exp), int((got != exp).sum())
    # the paths were exercised: output clamps, the 26-bit accumulator saturation and the integrator clamp
    assert (exp == 0).any() and (exp == 65535).any()
    acc_hi = (1 << 25) - 1                             # largest value of the 26-bit signed accumulator
    mid = integ_seen[:, 4:]                            # rings 4-7 start at random codes: the accumulator limit binds
    assert (mid == acc_hi).any() and (mid == -acc_hi - 1).any()
    u0 = obs[0, 0, spec.layout.obs_field("tx.heater.applied").offset:][:4]
    lim_lo, lim_hi = (0 - u0) << 20, (65535 - u0) << 20     # integrator clamp: output range, 20 fraction bits
    edge = integ_seen[:, :4]                           # rings 0-3 start within 3 codes of a rail: the clamp binds
    assert ((edge == lim_lo) | (edge == lim_hi)).any(axis=0).all()


def test_c_setpoint_argument(build, spec):
    """A given set-point (no capture) behaves like FxpPi(setpoint_codes=...)."""
    rng = np.random.default_rng(3)
    obs, ticks = _synthetic_obs(spec, 300, rng)
    c = FxpPi(setpoint_codes=31000)
    c.reset(spec)
    p, pf, s, sf = fxp.fxp_pi_init(c.spec_i64, c.spec_f64)
    act = c.initial_act(spec)[0].copy()
    exp = np.zeros((300, 1, 8), dtype=np.int64)
    for k in range(300):
        fxp.fxp_pi_step(p, pf, s, sf, np.ascontiguousarray(obs[k, 0]), act)
        exp[k, 0] = act[:8]
    got, _ = _drive([str(build["minimal"]), *c_args(c, 31000)], spec, obs, ticks, profile="minimal")
    assert np.array_equal(got, exp)


def test_header_matches_the_python_layout(spec, tmp_path):
    """Compile a probe against the generated header and compare every offset and size with the Python layout."""
    lay = bridge.FrameLayout.from_spec(spec, "full")
    (tmp_path / "mrm_bridge.h").write_text(bridge.c_header(lay))
    lines = ['#include <stdio.h>', '#include "mrm_bridge.h"', "int main(void) {",
             'printf("%d %d %d %d\\n", MRMB_OBS_RECORD_BYTES, MRMB_ACT_RECORD_BYTES, MRMB_OBS_PAYLOAD_BYTES, '
             'MRMB_ACT_PAYLOAD_BYTES);']
    for f in lay.obs_fields:
        n = bridge._cname(f.name)
        lines.append(f'printf("{f.name} %d %d %d\\n", MRMB_OBS_{n}_OFF, MRMB_OBS_{n}_COUNT, MRMB_OBS_{n}_BYTES);')
    lines += ['printf("%s\\n", MRMB_LAYOUT_FP);', "return 0; }"]
    (tmp_path / "probe.c").write_text("\n".join(lines))
    exe = tmp_path / "probe"
    r = subprocess.run([CC, "-std=c99", "-Wall", "-Werror", "-o", str(exe), str(tmp_path / "probe.c"), "-I",
                        str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = subprocess.run([str(exe)], capture_output=True, text=True).stdout.splitlines()
    assert out[0].split() == [str(lay.obs_record_bytes), str(lay.act_record_bytes), str(lay.obs_payload_bytes),
                              str(lay.act_payload_bytes)]
    for f, line in zip(lay.obs_fields, out[1:]):
        assert line.split() == [f.name, str(f.offset), str(f.count), str(np.dtype(f.dtype).itemsize)]
    assert out[-1] == lay.fingerprint()
    # every value field also reports its channel range
    h = (tmp_path / "mrm_bridge.h").read_text()
    assert "#define MRMB_ACT_TX_HEATER_MAX INT64_C(65535)" in h and "#define MRMB_ACT_TX_HEATER_MASK_BIT 0" in h


def test_c_refuses_another_layout(build, spec):
    """The minimal-profile binary against a full-profile plant: the C side checks the layout fingerprint."""
    bc = bridge.BridgeController(bridge.spawn([str(build["minimal"]), *c_args(FxpPi())]), profile="full")
    with pytest.raises(bridge.BridgeError, match="fingerprint"):
        bc.reset(spec)
    bc.close()


def test_c_refuses_another_spec(spec, tmp_path):
    """A header generated with MRMB_SPEC_FP (as `mrm bridge c-header` does) for another interface with the SAME
    frame layout (another knowledge list): the C side compares the HELLO's spec fingerprint and refuses."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c3 = params.load("g26", overrides=["controller_interface.knowledge=[line_grid,ring_to_line,heater_law]"])
    s3 = bridge.spec_for(c3, scenario(), 1)
    lay = bridge.FrameLayout.from_spec(s3, "minimal")
    assert lay.fingerprint() == bridge.FrameLayout.from_spec(spec, "minimal").fingerprint()
    assert s3.fingerprint() != spec.fingerprint()
    (tmp_path / "mrm_bridge.h").write_text(bridge.c_header(lay, spec_fingerprint=s3.fingerprint()))
    (tmp_path / "fxp_pi.c").write_text(bridge.C_FXP_PI_SOURCE)
    exe = tmp_path / "fxp_pi"
    r = subprocess.run([CC, "-O2", "-std=c99", "-Wall", "-Wextra", "-Werror", "-o", str(exe),
                        str(tmp_path / "fxp_pi.c"), "-I", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    bc = bridge.BridgeController(bridge.spawn([str(exe), *c_args(FxpPi())], stderr=subprocess.DEVNULL),
                                 profile="minimal")
    with pytest.raises(bridge.BridgeError, match="ControllerSpec fingerprint"):
        bc.reset(spec)
    bc.close()
    assert f'#define MRMB_SPEC_FP "{s3.fingerprint()}"' in (tmp_path / "mrm_bridge.h").read_text()
