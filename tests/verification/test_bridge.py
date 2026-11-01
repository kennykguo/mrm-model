"""Checks the binary lockstep bridge (mrm.sim.bridge) that lets an external process act as the controller.

Read this first
---------------
The bridge sends one observation frame per controller step to an external controller and reads back one action
frame. The frame layout is generated from the ChannelSpecs. This file checks:

- the frame layouts (the minimal and full profiles and custom frames) and the codec;
- the handshake and its checks (protocol, magic, ControllerSpec, latency);
- closed loops with an external controller: a thread over a socketpair, a Python process over pipes and over TCP.
  Each one must be bit-identical to the same controller run in-process;
- loud failures: a dead or silent peer, a lost lockstep tick, an actuator that the frame does not carry.

The C reference endpoint is checked in test_bridge_c.py.

Closed-loop runs: g26 nominal world, settled warm start at the steady-state lock codes, controller
mrm.control.fxp.FxpPi, a direct square aggressor (0.05 W, period 2 ms). The heaters then move by 200-2900 codes.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import threading
import warnings
from pathlib import Path

import numba as nb
import numpy as np
import pytest

from mrm import params
from mrm.control import api
from mrm.control.api import BaseController, Observation
from mrm.core.channels import ActId
from mrm.control.fxp import FxpPi
from mrm.scenarios import library
from mrm.sim import bridge, runner, vectors
from mrm.sim.metrics import RecordSpec

RUN = dict(uncertainty="nominal", variation="off", seeds=0, heater_code="locked_steady_state", settle=True)
DURATION_S = 3e-3
REC = [RecordSpec("detuning_m", 1, "last"), RecordSpec("heater_code", 10, "last"),
       RecordSpec("monitor_code", 10, "last")]
SERVE = [sys.executable, "-m", "mrm.sim.bridge", "serve", "--preset", "g26", "--controller", "mrm.control.fxp:FxpPi"]


def scenario():
    return library.aggressor_direct(waveform="square", power_w=0.05, period_s=2e-3, duty=0.5, phase_s=1e-3)


SCEN_ARGS = ["--scenario", "aggressor_direct", "--scenario-params",
             json.dumps({"waveform": "square", "power_w": 0.05, "period_s": 2e-3, "duty": 0.5, "phase_s": 1e-3})]


@pytest.fixture(scope="module")
def cfg():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load("g26")


@pytest.fixture(scope="module")
def cfg_bias():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load("g26", overrides=["controller_interface.bind.tx.bias=controller", "scenario.bias_v=null"])


@pytest.fixture(scope="module")
def spec(cfg):
    return bridge.spec_for(cfg, scenario(), 1)


@pytest.fixture(scope="module")
def reference(cfg):
    """FxpPi in-process (compiled path, in the kernel)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return runner.run(cfg, FxpPi(), scenario(), duration_s=DURATION_S, record=REC, **RUN)


def _same_metrics(a, b, prefix=""):
    """Every metric, nested dicts (world, settings, histogram) included, bit for bit (NaN = NaN)."""
    assert set(a) == set(b), (prefix, set(a) ^ set(b))
    for k, v in a.items():
        if isinstance(v, dict):
            _same_metrics(v, b[k], f"{prefix}{k}/")
        elif v is None or isinstance(v, str):
            assert v == b[k], prefix + k
        else:
            va, vb = np.asarray(v), np.asarray(b[k])
            assert va.dtype == vb.dtype and np.array_equal(va, vb, equal_nan=va.dtype.kind == "f"), prefix + k


def assert_same_run(a, b):
    assert set(a.trace) == set(b.trace)
    for k in a.trace:
        assert np.array_equal(a.trace[k]["t_s"], b.trace[k]["t_s"]), k
        assert np.array_equal(a.trace[k]["value"], b.trace[k]["value"]), k
    _same_metrics(a.metrics, b.metrics)


def run_bridged(cfg, ctrl):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return runner.run(cfg, ctrl, scenario(), duration_s=DURATION_S, record=REC, **RUN)


# --------------------------------------------------------------------------------------------------------- frame layout

def _name_dtype_count(fields):
    """(name, wire dtype, element count) of each frame field, in frame order."""
    return [(f.name, f.dtype, f.count) for f in fields]


def test_minimal_profile_is_the_07_frame(spec, cfg_bias):
    """The minimal profile is the frame that the design review asked for (fixed-point toolkit and HDL co-simulation):
    {u64 tick, u16 adc[N], i16 temp[S], u8 flags} -> {u64 tick, u16 dac[N], u16 bias[N]}.

    The generated frame maps it as follows. adc = tx.monitor.latest. temp = temp.* (12-bit codes, so u16). Then come
    the latched registers of the framed actions, flags[N] (u8) and dflags (u8). dac = tx.heater. bias is in the frame
    only when tx.bias is bound to the controller."""
    lay = bridge.FrameLayout.from_spec(spec, "minimal")
    assert _name_dtype_count(lay.obs_fields) == [("tx.monitor.latest", "<u2", 8), ("temp.die0", "<u2", 1),
                                                 ("tx.heater.applied", "<u2", 8), ("flags", "<u1", 8),
                                                 ("dflags", "<u1", 1)]
    assert _name_dtype_count(lay.act_fields) == [("tx.heater", "<u2", 8)]
    assert [(b.channel, b.what, b.group, b.bit) for b in lay.flag_bits] == [
        ("tx.monitor.latest", "new", "flags", 0), ("monitor_valid", "value", "flags", 1),
        ("monitor_valid", "new", "flags", 2), ("adc_overrange", "value", "flags", 3),
        ("adc_overrange", "new", "flags", 4), ("tx.heater.applied", "new", "flags", 5),
        ("temp.die0", "new", "dflags", 0)]
    assert lay.obs_record_bytes == 16 + 2 + 16 + 8 + 1 and lay.obs_payload_bytes == 8 + 43
    assert lay.act_payload_bytes == 16 + 16
    lb = bridge.FrameLayout.from_spec(bridge.spec_for(cfg_bias, scenario(), 1), "minimal")
    assert _name_dtype_count(lb.act_fields) == [("tx.heater", "<u2", 8), ("tx.bias", "<u2", 8)]
    assert ("tx.bias.applied", "<u2", 8) in _name_dtype_count(lb.obs_fields)


def test_full_profile_covers_the_spec(spec):
    lay = bridge.FrameLayout.from_spec(spec, "full")
    sensors = [f.name for f in spec.layout.obs]
    assert lay.sensors == sensors
    assert lay.actuators == [f.name for f in spec.layout.act]
    types = {f.name: (f.dtype, f.count) for f in lay.obs_fields}
    assert types["tx.monitor.samples"] == ("<u2", 8 * spec.samples_per_step)
    assert types["tx.monitor.sample_ticks"] == ("<u8", 8 * spec.samples_per_step)
    assert types["tx.monitor.count"] == ("<u4", 8)              # 0 .. INT32_MAX
    assert types["datapath.bitstat_above"] == ("<u1", 8)        # 0 .. bitstat window (64)
    assert types["flags"] == ("<u2", 8)                         # 3 bool channels x 2 bits + 10 new bits = 16
    assert types["dflags"] == ("<u1", 1)
    assert {f.name: f.dtype for f in lay.act_fields} == {"tx.heater": "<u2", "datapath.bitstat_threshold": "<u2",
                                                         "monitor.pga_select": "<u1"}
    assert len({b.bit for b in lay.flag_bits if b.group == "flags"}) == 16
    assert lay.fingerprint() != bridge.FrameLayout.from_spec(spec, "minimal").fingerprint()


def test_a_block_channel_brings_its_count(spec):
    """A sample block brings its count channel, a framed action its latched register (the warm-start held command
    of a CompiledController; without it FxpPi's u0 would be the channel init, not the plant's register)."""
    lay = bridge.FrameLayout.from_spec(spec, sensors=["tx.monitor.samples"], actuators=["tx.heater"])
    order = [f.name for f in spec.layout.obs]
    want = {"tx.monitor.samples", "tx.monitor.count", "tx.heater.applied"}
    assert lay.profile == "custom" and lay.sensors == [n for n in order if n in want]
    lay2 = bridge.FrameLayout.from_spec(spec, sensors=["tx.monitor.latest"], actuators=["monitor.pga_select"])
    assert lay2.sensors == ["tx.monitor.latest"]                 # pga_select has no latched readback
    with pytest.raises(bridge.BridgeError, match="not sensors"):
        bridge.FrameLayout.from_spec(spec, sensors=["tx.monitor.nope"])
    with pytest.raises(bridge.BridgeError, match="profile"):
        bridge.FrameLayout.from_spec(spec, "tiny")


def _random_obs(spec, rng):
    B = spec.n_batch
    sensors, new = {}, {}
    for f in spec.layout.obs:
        ch = spec.channels[f.name]
        shape = (B,) + f.shape
        if ch.np_dtype.kind == "b":
            sensors[f.name] = rng.random(shape) < 0.5
        else:
            hi = min(int(ch.max), 2**62)
            v = rng.integers(int(ch.min), hi, size=shape, endpoint=True)
            v.flat[0], v.flat[-1] = int(ch.min), int(ch.max)            # the range ends
            sensors[f.name] = v.astype(ch.dtype)
    for f in spec.layout.new:
        new[f.name] = rng.random((B,) + f.shape) < 0.5
    return sensors, new


@pytest.mark.parametrize("profile", ["minimal", "full"])
def test_codec_round_trip(cfg, profile):
    spec3 = bridge.spec_for(cfg, scenario(), 3)
    lay = bridge.FrameLayout.from_spec(spec3, profile)
    rng = np.random.default_rng(7)
    for _ in range(20):
        sensors, new = _random_obs(spec3, rng)
        tick = int(rng.integers(0, 2**62))
        p = lay.encode_obs(tick, sensors, new)
        assert len(p) == lay.obs_payload_bytes
        t2, s2, n2 = lay.decode_obs(p)
        assert t2 == tick
        assert sorted(s2) == sorted(lay.sensors)
        for k, v in s2.items():
            assert v.dtype == spec3.channels[k].np_dtype and np.array_equal(v, sensors[k]), k
        for k, v in n2.items():
            assert np.array_equal(v, new[k]), k
        assert set(n2) == set(lay.sensors)
    acts = {f.name: rng.integers(spec3.channels[f.name].min, spec3.channels[f.name].max, size=(3,) + f.shape,
                                 endpoint=True) for f in spec3.layout.act if f.name in lay.actuators}
    sub = dict(list(acts.items())[:1])
    t, mask, back = lay.decode_act(lay.encode_act(99, sub))
    assert t == 99 and mask == 1 and list(back) == list(sub)
    assert np.array_equal(back["tx.heater"], sub["tx.heater"]) and back["tx.heater"].dtype == np.uint16
    t, mask, back = lay.decode_act(lay.encode_act(5, acts))
    assert mask == (1 << len(acts)) - 1


def test_codec_refuses_out_of_range_and_bad_sizes(spec):
    lay = bridge.FrameLayout.from_spec(spec, "minimal")
    rng = np.random.default_rng(1)
    sensors, new = _random_obs(spec, rng)
    sensors["tx.monitor.latest"] = sensors["tx.monitor.latest"].astype(np.int64)
    sensors["tx.monitor.latest"][0, 3] = 70000
    with pytest.raises(bridge.BridgeError, match="tx.monitor.latest"):
        lay.encode_obs(0, sensors, new)
    with pytest.raises(bridge.BridgeError, match="tx.heater"):
        lay.encode_act(0, {"tx.heater": np.full((1, 8), -5)})
    with pytest.raises(bridge.BridgeError, match="integer"):
        lay.encode_act(0, {"tx.heater": np.full((1, 8), 5.0)})
    with pytest.raises(bridge.BridgeError, match="payload"):
        lay.decode_obs(b"\0" * 10)
    bad = struct.pack("<QII", 0, 0b10, 0) + b"\0" * lay.act_record_bytes
    with pytest.raises(bridge.BridgeError, match="write_mask"):
        lay.decode_act(bad)


def test_frames_are_checked(spec):
    a, b = socket.socketpair()
    ta, tb = bridge.SocketTransport(a), bridge.SocketTransport(b)
    ta.send(struct.pack("<2sBBI", b"MB", 2, 3, 0))
    with pytest.raises(bridge.BridgeError, match="protocol 2"):
        bridge.read_frame(tb)
    ta.send(struct.pack("<2sBBI", b"XX", 1, 3, 0))
    with pytest.raises(bridge.BridgeError, match="magic"):
        bridge.read_frame(tb)
    bridge.write_frame(ta, bridge.KIND["ERROR"], b"boom")
    with pytest.raises(bridge.BridgeError, match="peer error: boom"):
        bridge.read_frame(tb)
    ta.close()
    assert bridge.read_frame(tb, eof_ok=True) is None
    tb.close()


def test_layout_fingerprint_covers_the_channel_ranges(spec):
    """Same wire types, another heater range (code_max 60000 instead of 65535): another layout fingerprint, so a C
    twin compiled for one range (it clamps to MRMB_ACT_TX_HEATER_MAX) refuses a plant with the other."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c2 = params.load("g26", overrides=["ring.heater.code_max=60000"])
    s2 = bridge.spec_for(c2, scenario(), 1)
    assert s2.channels["tx.heater"].max == 60000
    for prof in bridge.PROFILES:
        a, b = bridge.FrameLayout.from_spec(spec, prof), bridge.FrameLayout.from_spec(s2, prof)
        assert [f.dtype for f in a.act_fields] == [f.dtype for f in b.act_fields]
        assert a.fingerprint() != b.fingerprint(), prof
    row = bridge.FrameLayout.from_spec(s2, "minimal").describe()["act"][0]
    assert (row["name"], row["min"], row["max"]) == ("tx.heater", 0, 60000)


def test_hello_round_trip(spec):
    lay = bridge.FrameLayout.from_spec(spec, "full")
    h = bridge._hello_for(spec, lay, 3)
    h2 = bridge.Hello.unpack(h.pack())
    assert h2 == h
    assert struct.unpack_from("<I", h.pack(), 48)[0] == 1 and struct.unpack_from("<i", h.pack(), 88)[0] == 3


# --------------------------------------------------------------------------------------------------------- closed loops

def _thread_endpoint(ctrl, spec_ext, sock):
    out = {}

    def target():
        try:
            out["stats"] = bridge.serve(ctrl, spec_ext, bridge.SocketTransport(sock))
        except Exception as e:                        # noqa: BLE001 - reported to the test
            out["error"] = e
    th = threading.Thread(target=target, daemon=True)
    th.start()
    return th, out


@pytest.mark.parametrize("profile", ["full", "minimal"])
def test_thread_endpoint_is_bit_identical(cfg, spec, reference, profile):
    a, b = socket.socketpair()
    th, out = _thread_endpoint(FxpPi(), spec, b)
    with bridge.BridgeController(bridge.SocketTransport(a), profile=profile, keep_times=True) as bc:
        res = run_bridged(cfg, bc)
    th.join(30)
    assert "error" not in out, out.get("error")
    assert out["stats"]["steps"] == bc.stats["steps"] == 300 and bc.remote_name == "fxp_pi"
    assert_same_run(reference, res)
    assert res.diagnostics["saturation_counts_python"]["tx.heater"].sum() == 0


@pytest.mark.parametrize("profile", ["full", "minimal"])
def test_python_process_over_pipes_is_bit_identical(cfg, reference, profile):
    with bridge.BridgeController(bridge.spawn(SERVE + SCEN_ARGS), profile=profile, keep_times=True) as bc:
        res = run_bridged(cfg, bc)
    assert_same_run(reference, res)
    rt = np.median(bc.rt_samples[10:]) * 1e6
    print(f"\nPython endpoint over pipes ({profile}): median round trip {rt:.0f} us per controller edge "
          f"({rt / 10:.1f} us per plant tick at the g26 10-tick period)")


def test_python_process_over_tcp_is_bit_identical(cfg, reference):
    srv = bridge.listen("127.0.0.1", 0)
    port = srv.getsockname()[1]
    proc = subprocess.Popen(SERVE + SCEN_ARGS + ["--connect", f"127.0.0.1:{port}"])
    try:
        srv.settimeout(120)
        conn, _ = srv.accept()
        with bridge.BridgeController(bridge.SocketTransport(conn, timeout_s=300), keep_times=True) as bc:
            res = run_bridged(cfg, bc)
        assert proc.wait(60) == 0
    finally:
        srv.close()
        if proc.poll() is None:
            proc.kill()
    assert_same_run(reference, res)
    print(f"\nPython endpoint over TCP: median round trip {np.median(bc.rt_samples[10:]) * 1e6:.0f} us")


def test_custom_frame_is_bit_identical(cfg, reference):
    """An explicit two-channel frame (monitor sample + its valid flag -> heater): the heater's latched register comes
    along (the warm-start u0 of FxpPi), so even this frame reproduces the in-kernel run."""
    a, b = socket.socketpair()
    th, out = _thread_endpoint(FxpPi(), bridge.spec_for(cfg, scenario(), 1), b)
    with bridge.BridgeController(bridge.SocketTransport(a), sensors=["tx.monitor.latest", "monitor_valid"],
                                 actuators=["tx.heater"]) as bc:
        res = run_bridged(cfg, bc)
    th.join(30)
    assert "error" not in out, out.get("error")
    assert bc.layout.profile == "custom" and "tx.heater.applied" in bc.layout.sensors
    assert_same_run(reference, res)


_HEATER_ID = int(ActId.TX_HEATER)


# A CompiledController is two njit functions. init(spec_i64, spec_f64) returns its four arrays (int params, float
# params, int state, float state). step(p, pf, s, sf, obs, act) updates the state and writes the commands into act.
@nb.njit
def _swing_init(spec_i64, spec_f64):
    p = np.zeros(2, np.int64)
    p[0] = spec_i64[api.SPEC_I_N_RINGS]
    p[1] = api.act_offset(spec_i64, _HEATER_ID)
    return p, np.zeros(0), np.zeros(1, np.int64), np.zeros(0)


@nb.njit
def _swing_step(p, pf, s, sf, obs, act):
    """Drive the heater 30000 codes up per step for 3 steps (far past the 65535 rail), then down 1000 per step:
    the held command it builds on is the plant's clip of the previous one."""
    d = 30000 if s[0] < 3 else -1000
    for i in range(p[0]):
        act[p[1] + i] = act[p[1] + i] + d
    s[0] += 1


def test_out_of_range_compiled_controller_is_bit_identical(cfg):
    """A CompiledController whose commands leave the heater range: the plant clips them and the next step builds on
    the clipped held command (kernel, runner Python path). The external endpoint mirrors that clip, so the bridged
    run equals the in-kernel run; the endpoint's saturation counts equal the in-process ones."""
    def ctrl():
        return api.CompiledController("swing", _swing_init, _swing_step)
    rec = [RecordSpec("heater_code", 1, "last")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = runner.run(cfg, ctrl(), scenario(), duration_s=0.3e-3, record=rec, **RUN)
        py = runner.run(cfg, ctrl(), scenario(), duration_s=0.3e-3, record=rec, path="python", **RUN)
    a, b = socket.socketpair()
    th, out = _thread_endpoint(ctrl(), bridge.spec_for(cfg, scenario(), 1), b)
    with bridge.BridgeController(bridge.SocketTransport(a), profile="minimal") as bc:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = runner.run(cfg, bc, scenario(), duration_s=0.3e-3, record=rec, **RUN)
    th.join(30)
    assert "error" not in out, out.get("error")
    h = np.asarray(ref.trace["heater_code@1:last"]["value"])[:, 0, :]
    assert h.max() == 65535 and h[-1].max() < 65535 - 20000          # it hit the rail and came back down
    assert_same_run(ref, py)
    assert_same_run(ref, res)
    f = bridge.spec_for(cfg, scenario(), 1).layout.act_field("tx.heater")
    sat_kernel = int(np.asarray(ref.diagnostics["saturation_counts"])[:, f.offset:f.offset + f.size].sum())
    sat_python = int(np.sum(py.diagnostics["saturation_counts_python"]["tx.heater"]))
    sat_ext = int(np.sum(out["stats"]["saturation_counts"]["tx.heater"]))
    assert sat_ext > 0 and sat_ext == sat_python == sat_kernel, (sat_ext, sat_python, sat_kernel)
    assert int(np.sum(res.diagnostics["saturation_counts_python"]["tx.heater"])) == 0     # documented: plant side 0


def test_bias_actuator_over_the_bridge(cfg_bias):
    """A controller that owns tx.bias: the minimal frame carries bias[N] and the plant applies it."""
    class BiasStep(BaseController):
        name = "bias_step"

        def step(self, obs):
            code = 1000 if obs.tick < 1000 else 3000
            return {"tx.bias": np.full((1, 8), code, dtype=np.int64)}

    spec_b = bridge.spec_for(cfg_bias, scenario(), 1)
    a, b = socket.socketpair()
    th, out = _thread_endpoint(BiasStep(), spec_b, b)
    rec = [RecordSpec("bias_v", 100, "last")]
    with bridge.BridgeController(bridge.SocketTransport(a), profile="minimal") as bc:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = runner.run(cfg_bias, bc, scenario(), duration_s=2e-3, record=rec, **RUN)
    th.join(30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = runner.run(cfg_bias, BiasStep(), scenario(), duration_s=2e-3, record=rec, **RUN)
    assert "error" not in out, out.get("error")
    assert np.array_equal(ref.trace["bias_v@100:last"]["value"], res.trace["bias_v@100:last"]["value"])
    v = np.asarray(res.trace["bias_v@100:last"]["value"])[:, 0, 0]
    assert v[2] < 1.7 < 2.9 < v[-1]                 # 1000 and 3000 of 4095 codes over 1.0-3.75 V


# ---------------------------------------------------------------------------------------------------- failures are loud

def test_spec_mismatch_is_refused(cfg, spec):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c3 = params.load("g26", overrides=["controller_interface.knowledge=[line_grid,ring_to_line,heater_law]"])
    a, b = socket.socketpair()
    th, out = _thread_endpoint(FxpPi(), bridge.spec_for(c3, scenario(), 1), b)
    bc = bridge.BridgeController(bridge.SocketTransport(a))
    with pytest.raises(bridge.BridgeError, match="ControllerSpec"):
        bc.reset(spec)
    bc.close()
    th.join(30)
    assert isinstance(out.get("error"), bridge.BridgeError)


def test_latency_mismatch_is_refused(spec):
    a, b = socket.socketpair()
    th, out = _thread_endpoint(FxpPi(latency_ticks=2), spec, b)
    bc = bridge.BridgeController(bridge.SocketTransport(a), latency_ticks=0)
    with pytest.raises(bridge.BridgeError, match="latency_ticks = 2"):
        bc.reset(spec)
    bc.close()
    th.join(30)


def test_controller_error_reaches_the_plant(spec):
    class Boom(BaseController):
        name = "boom"

        def step(self, obs):
            raise ZeroDivisionError("controller bug at tick %d" % obs.tick)

    a, b = socket.socketpair()
    th, out = _thread_endpoint(Boom(), spec, b)
    with bridge.BridgeController(bridge.SocketTransport(a)) as bc:
        bc.reset(spec)
        sensors, new = _random_obs(spec, np.random.default_rng(0))
        with pytest.raises(bridge.BridgeError, match="ZeroDivisionError: controller bug at tick 7"):
            bc.step(Observation(7, spec.tick_s, sensors, new))
    th.join(30)
    assert isinstance(out.get("error"), ZeroDivisionError)


def test_lockstep_tick_is_checked(spec):
    """An ACT for another tick (a peer that lost a frame) is an error, never a silent shift."""
    a, b = socket.socketpair()
    lay = bridge.FrameLayout.from_spec(spec, "full")

    def fake():
        tb = bridge.SocketTransport(b)
        bridge.read_frame(tb)
        bridge.write_frame(tb, bridge.KIND["READY"], bridge._READY.pack(lay.fingerprint().encode(), 0, 0, b"x"))
        bridge.read_frame(tb)
        bridge.write_frame(tb, bridge.KIND["ACT"], lay.encode_act(12, {}))
    th = threading.Thread(target=fake, daemon=True)
    th.start()
    with bridge.BridgeController(bridge.SocketTransport(a)) as bc:
        bc.reset(spec)
        sensors, new = _random_obs(spec, np.random.default_rng(0))
        with pytest.raises(bridge.BridgeError, match="lockstep lost"):
            bc.step(Observation(11, spec.tick_s, sensors, new))
    th.join(10)


def test_a_dead_peer_is_reported(spec):
    tr = bridge.spawn([sys.executable, "-c", "import sys; sys.exit(3)"])
    bc = bridge.BridgeController(tr)
    with pytest.raises(bridge.BridgeError, match="closed the stream"):
        bc.reset(spec)
    bc.close()
    assert tr.proc.returncode == 3


def test_changing_an_unframed_actuator_is_refused(spec):
    """Minimal frame without monitor.pga_select: a controller that changes it would silently diverge."""
    class Pga(BaseController):
        name = "pga"

        def step(self, obs):
            return {"monitor.pga_select": np.full((1, 8), 1 if obs.tick > 5 else 0, dtype=np.int64)}

    a, b = socket.socketpair()
    th, out = _thread_endpoint(Pga(), spec, b)
    with bridge.BridgeController(bridge.SocketTransport(a), profile="minimal") as bc:
        bc.reset(spec)
        sensors, new = _random_obs(spec, np.random.default_rng(0))
        bc.step(Observation(0, spec.tick_s, sensors, new))
        with pytest.raises(bridge.BridgeError, match="monitor.pga_select"):
            bc.step(Observation(10, spec.tick_s, sensors, new))
    th.join(30)


def test_unframed_actuator_must_equal_the_plant_register(spec):
    """A controller that writes an unframed actuator with a value other than the plant's register from the FIRST
    step on: the plant would never see it (it holds the channel init), so the endpoint refuses it instead of
    dropping it silently."""
    class PgaOne(BaseController):
        name = "pga_one"

        def step(self, obs):
            return {"monitor.pga_select": np.ones((1, 8), dtype=np.int64)}

    assert spec.channels["monitor.pga_select"].init == 0
    a, b = socket.socketpair()
    th, out = _thread_endpoint(PgaOne(), spec, b)
    with bridge.BridgeController(bridge.SocketTransport(a), profile="minimal") as bc:
        bc.reset(spec)
        sensors, new = _random_obs(spec, np.random.default_rng(0))
        with pytest.raises(bridge.BridgeError, match="monitor.pga_select"):
            bc.step(Observation(0, spec.tick_s, sensors, new))
    th.join(30)


@pytest.mark.filterwarnings("ignore:bridge serve")
def test_unframed_warm_start_register_is_refused(spec):
    """The heater outside the frame: its plant register is a warm-start code this side cannot know, so writing it
    is refused (a CompiledController writes every actuator, so it needs the heater framed)."""
    a, b = socket.socketpair()
    th, out = _thread_endpoint(FxpPi(), spec, b)
    with bridge.BridgeController(bridge.SocketTransport(a), sensors=["tx.monitor.latest"],
                                 actuators=["monitor.pga_select"]) as bc:
        bc.reset(spec)
        sensors, new = _random_obs(spec, np.random.default_rng(0))
        with pytest.raises(bridge.BridgeError, match="warm-start register"):
            bc.step(Observation(0, spec.tick_s, sensors, new))
    th.join(30)


def test_fast_kernel_and_truth_fed_controllers_are_refused(spec):
    """Their behaviour is not in the edge frame: refused at the handshake, never run without it."""
    fast = FxpPi()
    fast.fast_fn = _swing_step                     # any function: the bridge must refuse a fast kernel outright

    class TruthFed(BaseController):
        name, truth_fed = "tf", True

        def step(self, obs, truth=None):
            return {}
    for ctrl, msg in ((fast, "fast_kernel"), (TruthFed(), "Truth-fed")):
        a, b = socket.socketpair()
        th, out = _thread_endpoint(ctrl, spec, b)
        bc = bridge.BridgeController(bridge.SocketTransport(a))
        with pytest.raises(bridge.BridgeError, match=msg):
            bc.reset(spec)
        bc.close()
        th.join(30)
        assert isinstance(out.get("error"), bridge.BridgeError)


def test_silent_socket_peer_times_out(spec):
    """timeout_s: a peer that never answers raises BridgeError instead of blocking the run (wall clock only guards
    liveness; the lockstep itself runs on ticks)."""
    a, b = socket.socketpair()
    bc = bridge.BridgeController(bridge.SocketTransport(a, timeout_s=0.2))
    with pytest.raises(bridge.BridgeError, match="silent"):
        bc.reset(spec)
    bc.close()
    b.close()


def test_stdio_endpoint_keeps_prints_off_the_frame_stream(cfg, reference, tmp_path):
    """The external Python side may print: stdio() points fd 1 at stderr, frames use a private descriptor."""
    mod = tmp_path / "noisy_ctrl.py"
    mod.write_text("from mrm.control.fxp import FxpPi\n"
                   "class Noisy(FxpPi):\n"
                   "    def step(self, obs):\n"
                   "        print('debug tick', obs.tick)\n"
                   "        return super().step(obs)\n"
                   "    @classmethod\n"
                   "    def from_params(cls, p):\n"
                   "        return cls()\n")
    cmd = [sys.executable, "-m", "mrm.sim.bridge", "serve", "--preset", "g26", "--controller", "noisy_ctrl:Noisy",
           *SCEN_ARGS]
    env = dict(os.environ, PYTHONPATH=f"{tmp_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
    with bridge.BridgeController(bridge.spawn(cmd, env=env, stderr=subprocess.DEVNULL)) as bc:
        res = run_bridged(cfg, bc)
    assert_same_run(reference, res)


def test_batch_of_worlds_over_the_bridge(cfg):
    """B = 2 sampled worlds (device variation on, distinct noise): one frame carries both world records."""
    kw = dict(RUN, uncertainty="sample", variation="sample", seeds=11, batch=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = runner.run(cfg, FxpPi(), scenario(), duration_s=1e-3, record=REC, **kw)
    spec2 = bridge.spec_for(cfg, scenario(), 2)
    a, b = socket.socketpair()
    th, out = _thread_endpoint(FxpPi(), spec2, b)
    with bridge.BridgeController(bridge.SocketTransport(a), profile="minimal") as bc:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = runner.run(cfg, bc, scenario(), duration_s=1e-3, record=REC, **kw)
    th.join(30)
    assert "error" not in out, out.get("error")
    assert bc.layout.n_batch == 2 and bc.layout.obs_payload_bytes == 8 + 2 * bc.layout.obs_record_bytes
    assert_same_run(ref, res)
    h = np.asarray(res.trace["heater_code@10:last"]["value"])
    assert not np.array_equal(h[:, 0], h[:, 1])                      # two different worlds


def test_cli_commands(tmp_path, spec):
    from mrm import cli
    p = tmp_path / "mrm_bridge.h"
    assert cli.main(["bridge", "c-header", "--profile", "minimal", "--scenario", "aggressor_direct", "--out",
                     str(p)]) == 0
    assert bridge.FrameLayout.from_spec(spec, "minimal").fingerprint() in p.read_text()
    q = tmp_path / "fxp_pi.c"
    assert cli.main(["bridge", "c-example", "--out", str(q)]) == 0 and q.read_text() == bridge.C_FXP_PI_SOURCE
    gv = str(Path(__file__).resolve().parents[2] / "tests" / "golden" / "vectors" / "fxp_pi__g26__aggressor_square.npz")
    assert cli.main(["vectors", "replay", gv]) == 0
    assert cli.main(["vectors", "replay", gv, "--params", '{"ki": 0.0149}']) == 1      # other gains: mismatches
    assert cli.main(["vectors", "hex", gv, "--out", str(tmp_path / "gv")]) == 0
    assert (tmp_path / "gv.obs.hex").exists() and (tmp_path / "gv.layout.vh").exists()
