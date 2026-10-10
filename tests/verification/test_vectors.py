"""Golden vectors (07 P1-21; mrm.sim.vectors): recording is transparent, the files round-trip, and replaying the
recorded observations through the same fixed-point controller reproduces the actions bit for bit (numba, Python
protocol, compiled ABI, NUMBA_DISABLE_JIT=1, and the committed golden vector file).

Run: g26 nominal world, settled warm start at the quasi-static lock codes, mrm.control.fxp.FxpPi, a direct square
aggressor (0.05 W at the g26 aggressor site, period 2 ms, phase 1 ms) for 4 ms: 400 controller edges, the heater
codes of rings 0-7 move by 2867 ... 203 codes (the integrators work; measured 2026-10-10).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numba as nb
import numpy as np
import pytest

from mrm import params
from mrm.control import api, fxp
from mrm.control.api import BaseController, CompiledController
from mrm.control.fxp import FxpPi
from mrm.core.channels import ActId
from mrm.scenarios import library as lib
from mrm.sim import bridge, runner, vectors
from mrm.sim.metrics import RecordSpec

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_VEC = ROOT / "tests" / "golden" / "vectors" / "fxp_pi__g26__aggressor_square.npz"
RUN = dict(uncertainty="nominal", variation="off", seeds=0, heater_code="quasi_static", settle=True)
DURATION_S = 4e-3
REC = [RecordSpec("detuning_m", 10, "mean"), RecordSpec("heater_code", 10, "last"),
       RecordSpec("monitor_code", 10, "last")]


def scenario():
    return lib.aggressor_direct(waveform="square", power_w=0.05, period_s=2e-3, duty=0.5, phase_s=1e-3)


@pytest.fixture(scope="module")
def cfg():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load("g26")


@pytest.fixture(scope="module")
def recorded(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vectors.run_recorded(cfg, FxpPi(), scenario(), duration_s=DURATION_S, record=REC, **RUN)


@pytest.fixture(scope="module")
def spec(cfg):
    return bridge.spec_for(cfg, scenario(), 1)


def test_recording_is_transparent(cfg, recorded):
    """The recorder changes nothing: the recorded run equals the in-kernel compiled run bit for bit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = runner.run(cfg, FxpPi(), scenario(), duration_s=DURATION_S, record=REC, **RUN)
    assert ref.manifest_inputs["controller_path"] == "compiled"
    assert recorded.manifest_inputs["controller_path"] == "python"
    for k in ref.trace:
        assert np.array_equal(ref.trace[k]["value"], recorded.trace[k]["value"]), k
    _same_metrics(ref.metrics, recorded.metrics)


def _same_metrics(a, b, prefix=""):
    """Every metric (nested dicts included) bit for bit, NaN = NaN."""
    assert set(a) == set(b), (prefix, set(a) ^ set(b))
    for k, v in a.items():
        if isinstance(v, dict):
            _same_metrics(v, b[k], f"{prefix}{k}/")
        elif v is None or isinstance(v, str):
            assert v == b[k], prefix + k
        else:
            va, vb = np.asarray(v), np.asarray(b[k])
            assert np.array_equal(va, vb, equal_nan=va.dtype.kind == "f"), prefix + k


_HEATER_ID = int(ActId.TX_HEATER)


@nb.njit
def _swing_init(spec_i64, spec_f64):
    p = np.zeros(2, np.int64)
    p[0] = spec_i64[api.SPEC_I_N_RINGS]
    p[1] = api.act_offset(spec_i64, _HEATER_ID)
    return p, np.zeros(0), np.zeros(1, np.int64), np.zeros(0)


@nb.njit
def _swing_step(p, pf, s, sf, obs, act):
    """+30000 heater codes per step for 3 steps (past the 65535 rail), then -1000 per step."""
    d = 30000 if s[0] < 3 else -1000
    for i in range(p[0]):
        act[p[1] + i] = act[p[1] + i] + d
    s[0] += 1


def test_recording_is_transparent_when_commands_leave_the_range(cfg, spec):
    """The recorder mirrors the plant's clip of a CompiledController's held command (FxpPi never leaves the range,
    so the main test cannot see it): the recorded run equals the in-kernel run, the recorded act holds the raw
    out-of-range commands, act_in the clipped held ones, and both replays reproduce the actions."""
    def ctrl():
        return CompiledController("swing", _swing_init, _swing_step)
    rec = [RecordSpec("heater_code", 1, "last")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = runner.run(cfg, ctrl(), scenario(), duration_s=0.3e-3, record=rec, **RUN)
        rr = vectors.run_recorded(cfg, ctrl(), scenario(), duration_s=0.3e-3, record=rec, **RUN)
    assert np.array_equal(ref.trace["heater_code@1:last"]["value"], rr.trace["heater_code@1:last"]["value"])
    _same_metrics(ref.metrics, rr.metrics)
    v = rr.vectors
    h, h_in = v.channel("tx.heater", "act")[:, 0, :], v.act_in[:, 0, :8]
    assert h.max() > 65535 and h_in.max() == 65535                     # raw command vs the clipped held one
    assert np.array_equal(h_in[1:], np.clip(h[:-1], 0, 65535))
    assert vectors.replay(ctrl(), v, spec).ok
    assert vectors.replay_abi(_swing_init, _swing_step, v).ok


def test_stream_content(recorded, spec):
    v = recorded.vectors
    K = int(round(DURATION_S / (spec.period_ticks * spec.tick_s)))
    assert v.n_records == K == 400
    assert v.obs.shape == (K, 1, spec.layout.obs_size) and v.act.shape == (K, 1, spec.layout.act_size)
    assert v.act_in.shape == v.act.shape
    assert np.array_equal(v.ticks, np.arange(K) * spec.period_ticks)
    assert np.array_equal(v.obs[:, 0, 0], v.ticks)                      # obs header: tick
    assert v.reset_before[0] and not v.reset_before[1:].any()
    assert v.written.all()                                              # a CompiledController returns every channel
    assert v.layout_fingerprint == spec.layout.fingerprint() and v.spec_fingerprint == spec.fingerprint()
    # held command on entry = the previous output (FxpPi never leaves the channel ranges, so no clip)
    assert np.array_equal(v.act_in[1:], v.act[:-1])
    # the first held heater command is the latched register (warm start)
    assert np.array_equal(v.act_in[0, 0, :8], v.channel("tx.heater.applied")[0, 0])
    span = np.ptp(v.channel("tx.heater", "act")[:, 0, :], axis=0)
    assert span.min() > 100 and span.max() > 2000, span                # the integrators really move
    assert v.meta["controller_params"]["kp_raw"] == FxpPi().kp_raw
    assert v.meta["scenario"] == "aggressor_direct"


def test_files_round_trip(recorded, tmp_path):
    paths = vectors.record_vectors(recorded, tmp_path / "vec" / "fxp_pi.npz")
    assert set(paths) == {"npz", "obs_hex", "act_hex", "act_in_hex", "ticks_hex", "written_hex", "vh"}
    v = recorded.vectors
    w = vectors.load_vectors(paths["npz"])
    for k, a in v.arrays().items():
        assert np.array_equal(a, getattr(w, k)), k
    assert w.layout_rows == json.loads(json.dumps(v.layout_rows)) and w.meta["controller"] == "fxp_pi"
    assert np.array_equal(vectors.read_hex(paths["obs_hex"]).reshape(v.obs.shape), v.obs)
    assert np.array_equal(vectors.read_hex(paths["act_hex"]).reshape(v.act.shape), v.act)
    assert np.array_equal(vectors.read_hex(paths["act_in_hex"]).reshape(v.act.shape), v.act_in)
    assert np.array_equal(vectors.read_hex(paths["ticks_hex"]), v.ticks)
    assert np.array_equal(vectors.read_hex(paths["written_hex"]), np.full(v.n_records, 0b111))
    lines = Path(paths["obs_hex"]).read_text().splitlines()
    assert lines[0].startswith("// mrm.vectors/1") and all(len(x) == 16 for x in lines if not x.startswith("//"))
    vh = Path(paths["vh"]).read_text()
    for r in v.layout_rows:
        if r["id"] >= 0:
            pre = {"obs": "OBS", "new": "NEW", "act": "ACT"}[r["vector"]]
            nm = r["name"].replace(".", "_").upper()
            assert f"localparam integer MRMV_{pre}_{nm} = {r['offset']};" in vh
    assert f"localparam integer MRMV_K = {v.n_records};" in vh


def test_negative_words_are_twos_complement(tmp_path):
    p = vectors.write_hex(tmp_path / "x.hex", np.array([-1, -2**63, 2**63 - 1, 0, 5]), ["c"])
    assert Path(p).read_text().splitlines()[1:] == ["ffffffffffffffff", "8000000000000000", "7fffffffffffffff",
                                                   "0000000000000000", "0000000000000005"]
    assert np.array_equal(vectors.read_hex(p), [-1, -2**63, 2**63 - 1, 0, 5])


def test_load_rejects_edited_or_foreign_files(recorded, tmp_path):
    paths = vectors.record_vectors(recorded, tmp_path / "v", hex=False)
    with np.load(paths["npz"]) as z:
        d = {k: np.array(z[k]) for k in z.files}
    d["act"][3, 0, 2] += 1
    np.savez(tmp_path / "edited.npz", **d)
    with pytest.raises(vectors.VectorError, match="sha256"):
        vectors.load_vectors(tmp_path / "edited.npz")
    head = json.loads(str(d["meta_json"]))
    head["format"] = "mrm.vectors/0"
    d["meta_json"] = np.array(json.dumps(head))
    np.savez(tmp_path / "old.npz", **d)
    with pytest.raises(vectors.VectorError, match="format"):
        vectors.load_vectors(tmp_path / "old.npz")


def test_replay_python_protocol_is_bit_exact(recorded, spec):
    res = vectors.replay(FxpPi(), recorded.vectors, spec)
    assert res.ok, res.first
    assert np.array_equal(res.act, recorded.vectors.act)


def test_replay_compiled_abi_is_bit_exact(recorded):
    res = vectors.replay_abi(fxp.fxp_pi_init, fxp.fxp_pi_step, recorded.vectors)
    assert res.ok, res.first


def test_replay_detects_a_different_controller(recorded, spec):
    """The replay is a real check: other gains (one LSB of ki) give other actions."""
    c = FxpPi()
    other = FxpPi(ki=(c.ki_raw + 1) * 2.0 ** -c.frac_bits)
    res = vectors.replay(other, recorded.vectors, spec)
    assert res.mismatches > 0 and res.first["channel"] == "tx.heater"
    si = recorded.vectors.spec_i64.copy()
    si[si[api.SPEC_I_USER] + 1] += 1                                   # ki is FxpPi's user integer 1
    assert not vectors.replay_abi(fxp.fxp_pi_init, fxp.fxp_pi_step, recorded.vectors, spec_i64=si).ok


def test_replay_needs_the_same_interface(recorded, cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c2 = params.load("g26", overrides=["controller_interface.bind.tx.bias=controller", "scenario.bias_v=null"])
    with pytest.raises(vectors.VectorError, match="layout"):
        vectors.replay(FxpPi(), recorded.vectors, bridge.spec_for(c2, scenario(), 1))
    with pytest.raises(vectors.VectorError, match="layout"):          # B is part of the layout (dims)
        vectors.replay(FxpPi(), recorded.vectors, bridge.spec_for(cfg, scenario(), 2), check_spec=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c3 = params.load("g26", overrides=["controller_interface.knowledge=[line_grid,ring_to_line,heater_law]"])
    s3 = bridge.spec_for(c3, scenario(), 1)                            # same layout, other knowledge
    with pytest.raises(vectors.VectorError, match="ControllerSpec"):
        vectors.replay(FxpPi(), recorded.vectors, s3)
    assert vectors.replay(FxpPi(), recorded.vectors, s3, check_spec=False).ok


def test_replay_without_jit_is_bit_exact(recorded, tmp_path):
    """The same functions executed by the Python interpreter (NUMBA_DISABLE_JIT=1): numba's int64 arithmetic and
    NumPy's agree on every action."""
    paths = vectors.record_vectors(recorded, tmp_path / "v", hex=False)
    code = ("import sys; from mrm.sim import vectors; from mrm.control import fxp\n"
            "v = vectors.load_vectors(sys.argv[1])\n"
            "r = vectors.replay_abi(fxp.fxp_pi_init, fxp.fxp_pi_step, v)\n"
            "print('MISMATCHES', r.mismatches, r.first)\n")
    env = dict(os.environ, NUMBA_DISABLE_JIT="1")
    out = subprocess.run([sys.executable, "-c", code, str(paths["npz"])], capture_output=True, text=True, env=env,
                         timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "MISMATCHES 0 None" in out.stdout, out.stdout


class _HoldPlus(BaseController):
    """A Python (non-compiled) controller that writes tx.heater every other step only: +1 code per new sample."""
    name = "hold_plus"

    def reset(self, spec):
        super().reset(spec)
        self.k = 0
        self.u = None

    def step(self, obs):
        self.k += 1
        if self.u is None:
            self.u = np.array(obs["tx.heater.applied"], dtype=np.int64)
        self.u = self.u + obs.new["tx.monitor.latest"].astype(np.int64)
        return {"tx.heater": self.u.copy()} if self.k % 2 else {}


def test_python_controller_vectors_and_written_mask(cfg, spec):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rr = vectors.run_recorded(cfg, _HoldPlus(), scenario(), duration_s=0.4e-3, **RUN)
    v = rr.vectors
    assert v.act_in is None and v.spec_i64 is None
    names = [r["name"] for r in v.act_fields]
    w = v.written[:, names.index("tx.heater")]
    assert w[::2].all() and not w[1::2].any()
    assert not v.written[:, names.index("monitor.pga_select")].any()
    res = vectors.replay(_HoldPlus(), v, spec)
    assert res.ok, res.first
    with pytest.raises(vectors.VectorError, match="spec arrays"):
        vectors.replay_abi(fxp.fxp_pi_init, fxp.fxp_pi_step, v)


def test_recorder_refuses_fast_kernel_and_truth_fed():
    class Fast:
        name, latency_ticks, fast_kernel = "f", 0, (lambda *a: None)

    class Oracle:
        name, latency_ticks, truth_fed = "o", 0, True
    with pytest.raises(TypeError, match="fast_kernel"):
        vectors.VectorRecorder(Fast())
    with pytest.raises(TypeError, match="Truth-fed"):
        vectors.VectorRecorder(Oracle())


def test_reset_marks_and_replay_resets(recorded, spec):
    """A controller reset mid-run (scenario fault.controller_reset) is recorded as reset_before and replayed."""
    v = recorded.vectors
    rec = vectors.VectorRecorder(FxpPi())
    rec.reset(spec)
    for k in range(150):
        if k == 100:
            rec.reset(spec)
        rec.step(vectors.observation_from_flat(spec, v.obs[k], int(v.ticks[k])))
    w = rec.stream()
    assert np.flatnonzero(w.reset_before).tolist() == [0, 100]
    assert not np.array_equal(w.act[100:], v.act[100:150])          # the reset re-captured the set-point
    assert vectors.replay(FxpPi(), w, spec).ok
    assert vectors.replay_abi(fxp.fxp_pi_init, fxp.fxp_pi_step, w).ok


# ---- the committed golden vectors ----------------------------------------------------------------------------------

def test_golden_vector_file_replays_bit_exact():
    """tests/golden/vectors/: the committed recording (regenerate only with the command in tests/golden/README.md)
    replays bit for bit through FxpPi's compiled ABI, independent of the plant."""
    if not GOLDEN_VEC.exists():
        pytest.fail(f"{GOLDEN_VEC} is missing")
    v = vectors.load_vectors(GOLDEN_VEC)
    assert v.meta["controller"] == "fxp_pi" and v.n_records == 400
    c = FxpPi()
    assert v.meta["controller_params"]["kp_raw"] == c.kp_raw and v.meta["controller_params"]["ki_raw"] == c.ki_raw
    res = vectors.replay_abi(fxp.fxp_pi_init, fxp.fxp_pi_step, v)
    assert res.ok, res.first


def test_golden_vector_file_matches_todays_plant(recorded):
    """The plant still produces the committed observations (a plant change shows up here, like the golden traces;
    a failure means: regenerate tests/golden/vectors with a CHANGELOG line, after checking why)."""
    v = vectors.load_vectors(GOLDEN_VEC)
    assert v.spec_fingerprint == recorded.vectors.spec_fingerprint
    n = int((v.obs != recorded.vectors.obs).sum())
    assert n == 0, f"{n} observation words differ from the committed golden vectors"
    assert np.array_equal(v.act, recorded.vectors.act)
