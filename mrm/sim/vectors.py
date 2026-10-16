"""Golden vectors: the observation and action streams of a closed-loop run, as versioned ``.npz`` and ``.hex`` files
for ``$readmemh`` (07 P1-21).

A controller that will become RTL is developed against the plant in Python or numba, then frozen as vectors: every
observation vector it received and every action vector it produced, in the flat int64 ABI of
:class:`mrm.core.channels.AbiLayout` (07 P0-7, P0-9). Replaying the observations through the same fixed-point
controller (its numba form, its Python form under ``NUMBA_DISABLE_JIT=1``, its C twin or its Verilog twin) must
reproduce the actions bit for bit (07 P1-21: "fixed point against RTL, bit-exact"). The plant is not needed for a
replay: the vectors are the whole interface.

Recording
---------
Wrap the controller in a :class:`VectorRecorder` (a Controller-protocol wrapper: the run takes the Python path, which
for a :class:`mrm.control.api.CompiledController` runs the same numba ``step`` per world, bit-identical to the
in-kernel path) or call :func:`run_recorded`, then :func:`record_vectors`::

    from mrm.control.fxp import FxpPi
    from mrm.sim import vectors
    rr = vectors.run_recorded("g26", FxpPi(), "static", duration_s=2e-3, uncertainty="nominal", variation="off",
                              heater_code="quasi_static", settle=True)
    paths = vectors.record_vectors(rr, "runs/vec/fxp_pi")      # fxp_pi.npz, fxp_pi.obs.hex, ..., fxp_pi.layout.vh
    vec = vectors.load_vectors(paths["npz"])
    res = vectors.replay(FxpPi(), vec, spec)                    # Python protocol, needs the run's ControllerSpec
    res = vectors.replay_abi(fxp.fxp_pi_init, fxp.fxp_pi_step, vec)   # compiled ABI, no spec needed (RTL view)

What is recorded per controller edge k (K edges, B worlds)
----------------------------------------------------------
- ``ticks[k]``: the edge tick (int64);
- ``obs[k, b]``: the observation vector obs_i64 (``spec.layout.pack_obs``: header [tick, active_ring, 0, 0], the
  sensor values, the new-sample flags), exactly what a compiled controller's ``step`` receives;
- ``act[k, b]``: the controller's output act_i64 after the step, BEFORE the plant's range check (07 P1-19), as int64
  (for a channel the controller did not return: the held value, or 0 when it is unknown);
- ``written[k, j]``: True if the controller returned action channel j at edge k (Python protocol: a channel left out
  holds; a CompiledController returns every channel);
- ``act_in[k, b]`` (CompiledController only): the held command act_i64 on entry of ``step`` (07 P0-9 ABI: "on entry
  act_i64 holds the controller's previous command"), after the plant's clip of the previous command;
- ``reset_before[k]``: True at the first edge after a controller reset (the start, and any scenario
  ``fault.controller_reset``); a replay resets the controller there;
- ``spec_i64``, ``spec_f64`` (CompiledController only): the spec arrays its ``init`` was called with (07 P0-9; they
  include the controller's own parameters), so :func:`replay_abi` needs no ControllerSpec.

Files written by :func:`record_vectors` (stem ``P``)
-----------------------------------------------------
- ``P.npz``: every array above plus ``meta_json`` (format ``mrm.vectors/1``, ABI version, layout and spec
  fingerprints, the layout rows, the controller's description, the run's config fingerprint, seeds, scenario, and a
  sha256 of the arrays, checked by :func:`load_vectors`);
- ``P.obs.hex``, ``P.act.hex`` (+ ``P.act_in.hex``): one 64-bit two's-complement word per line (16 lower-case hex
  digits), in the order record k, world b, element i (row-major (K, B, size)); ``//`` comment lines at the top
  (Verilog readmem files allow comments). Load with ``reg [63:0] mem [0:K*B*SIZE-1]; $readmemh("P.obs.hex", mem);``;
- ``P.ticks.hex`` (K words), ``P.written.hex`` (K words, bit j = action channel j written);
- ``P.layout.vh``: Verilog ``localparam`` declarations of K, B, the vector sizes and every channel's offset and size
  (``MRMV_OBS_<CHANNEL>``, ``MRMV_NEW_<CHANNEL>``, ``MRMV_ACT_<CHANNEL>``, ``..._SIZE``).

Limits: a controller with a ``fast_kernel`` cannot be recorded (its per-sample inputs are not in the edge
observation; 07 P0-7), nor a Truth-fed one. Memory: K x B x (obs_size + 2 act_size) x 8 bytes (g26: 246 + 48 words,
2.4 kB per world-edge; one A1 period at 100 kHz is 31 MB at B = 1); ``max_steps`` caps it.
"""
from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from mrm.control.api import CompiledController, ControllerSpec, Observation
from mrm.core.channels import ABI_VERSION, OBS_ACTIVE_RING, OBS_TICK

__all__ = ["FORMAT", "VectorStream", "VectorRecorder", "RecordedRun", "ReplayResult", "VectorError", "run_recorded",
           "record_vectors", "load_vectors", "replay", "replay_abi", "compare_actions", "write_hex", "read_hex",
           "layout_vh", "observation_from_flat", "stream_of", "clip_held_command", "hidden_warm_start_registers"]

FORMAT = "mrm.vectors/1"
_ARRAYS = ("ticks", "obs", "act", "written", "reset_before", "act_in", "spec_i64", "spec_f64")


class VectorError(ValueError):
    """A vector file of another format or version, a corrupted file, or a layout that does not match."""


# ---------------------------------------------------------------------------------------------------- the stream
@dataclass
class VectorStream:
    """The recorded streams of one run (module docstring). Arrays are int64 except ``written`` / ``reset_before``
    (bool) and ``spec_f64`` (float64)."""
    ticks: np.ndarray
    obs: np.ndarray
    act: np.ndarray
    written: np.ndarray
    reset_before: np.ndarray
    layout_rows: list
    layout_fingerprint: str
    spec_fingerprint: str
    act_in: np.ndarray | None = None
    spec_i64: np.ndarray | None = None
    spec_f64: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_records(self) -> int:
        return int(self.ticks.shape[0])

    @property
    def n_batch(self) -> int:
        return int(self.obs.shape[1])

    @property
    def obs_size(self) -> int:
        return int(self.obs.shape[2])

    @property
    def act_size(self) -> int:
        return int(self.act.shape[2])

    @property
    def act_fields(self) -> list[dict]:
        """Rows of the action vector, in order (name, offset, size, shape, dtype, id)."""
        return [r for r in self.layout_rows if r["vector"] == "act"]

    def field(self, name: str, vector: str = "obs") -> dict:
        """The layout row of ``name`` in ``vector`` ('obs', 'new' or 'act')."""
        for r in self.layout_rows:
            if r["vector"] == vector and r["name"] == name:
                return r
        raise KeyError(f"{vector} channel {name!r} is not in these vectors")

    def channel(self, name: str, vector: str = "obs") -> np.ndarray:
        """(K, B, size) slice of one channel ('obs' and 'new' read obs, 'act' reads act)."""
        r = self.field(name, vector)
        src = self.act if vector == "act" else self.obs
        return src[:, :, r["offset"]:r["offset"] + r["size"]]

    def arrays(self) -> dict[str, np.ndarray]:
        """The arrays that the sha256 covers, in a fixed order (absent optional arrays left out)."""
        out = {}
        for k in _ARRAYS:
            v = getattr(self, k)
            if v is not None:
                out[k] = np.ascontiguousarray(v)
        return out

    def sha256(self) -> str:
        """sha256 of every array (name, dtype, shape and bytes), hex."""
        h = hashlib.sha256()
        for k, v in self.arrays().items():
            h.update(f"{k}|{v.dtype.str}|{v.shape}|".encode())
            h.update(v.tobytes())
        return h.hexdigest()


def stream_of(run) -> VectorStream:
    """The VectorStream of a VectorStream, a VectorRecorder, a RecordedRun or any object with ``.vectors``."""
    if isinstance(run, VectorStream):
        return run
    if isinstance(run, VectorRecorder):
        return run.stream()
    v = getattr(run, "vectors", None)
    if isinstance(v, VectorStream):
        return v
    raise TypeError(f"no recorded vectors in {type(run).__name__}: record the run with mrm.sim.vectors."
                    f"run_recorded(...) or wrap the controller in VectorRecorder")


# ---------------------------------------------------------------------------------------------------- recording
def clip_held_command(ctrl: CompiledController, spec: ControllerSpec) -> None:
    """Clip a CompiledController's held command ``ctrl.act`` (B, act_size) in place to each actuator's channel range.

    This is what the engine does after a compiled step: the kernel clips the held vector itself
    (mrm.sim.kernel, event-order step 4) and the runner's Python path copies the validated (clipped) command back
    into it. A controller run outside the runner (VectorRecorder, :func:`replay`, the bridge's ``serve``) must do
    the same, or its next step would see an unclipped held command. The returned action dict of ``step`` is a copy
    and is not touched (the validator still sees and counts the out-of-range values)."""
    for f in spec.layout.act:
        ch = spec.channels[f.name]
        np.clip(ctrl.act[:, f.offset:f.offset + f.size], ch.min, ch.max, out=ctrl.act[:, f.offset:f.offset + f.size])


def hidden_warm_start_registers(spec: ControllerSpec) -> list[str]:
    """Controller-bound DAC actuators (heater, bias) whose latched readback ``<name>.applied`` the spec does not
    expose. In-process the runner pre-fills a bound CompiledController's held command from the plant's registers
    (the warm-start codes); a CompiledController run outside the runner can only pre-fill from the readback, and
    falls back to the channel init without it, so its first held command (and a controller that uses it, such as
    FxpPi's u0) then differs from the in-process run."""
    acts = {f.name for f in spec.layout.act}
    return [n for n in ("tx.heater", "tx.bias") if n in acts and n + ".applied" not in spec.channels]


class VectorRecorder:
    """Controller-protocol wrapper that records every edge's obs_i64 and act_i64 (module docstring).

    The wrapped controller runs unchanged; the run takes the Python path (the runner sees a Python controller), where
    a CompiledController executes the same numba ``step`` per world as in the kernel. Two details of the runner's
    handling of a bare CompiledController are mirrored so the run stays bit-identical to it: the held command starts
    from the latched DAC registers (CompiledController's own ``.applied`` rule) and, after each step, the held
    command is clipped to the channel ranges as the runner's ActionValidator would (clip mode). Other attributes
    (``telemetry``, ``load_factory_cal``, parameters) are forwarded to the wrapped controller.
    """

    def __init__(self, controller: Any, *, max_steps: int | None = None):
        if getattr(controller, "truth_fed", False):
            raise TypeError("a Truth-fed controller (the oracle) cannot be recorded: it reads Truth, not the "
                            "observation (07 P1-1)")
        if getattr(controller, "fast_kernel", None) is not None:
            raise TypeError(f"{getattr(controller, 'name', controller)!r} has a fast_kernel: its per-sample inputs "
                            f"are not in the edge observation, so its vectors would not replay (07 P0-7)")
        self.inner = controller
        self.name = str(getattr(controller, "name", type(controller).__name__))
        self.latency_ticks = int(getattr(controller, "latency_ticks", 0) or 0)
        self.max_steps = None if max_steps is None else int(max_steps)
        self.spec: ControllerSpec | None = None
        self._clear()

    def __getattr__(self, item):            # forwards telemetry, load_factory_cal, params, ...
        if item == "inner":
            raise AttributeError(item)
        return getattr(self.inner, item)

    def _clear(self) -> None:
        self._ticks: list[int] = []
        self._obs: list[np.ndarray] = []
        self._act: list[np.ndarray] = []
        self._act_in: list[np.ndarray] = []
        self._written: list[np.ndarray] = []
        self._reset: list[bool] = []
        self._pending_reset = True
        self.truncated = False
        self._spec_arrays = None

    def reset(self, spec: ControllerSpec) -> None:
        if self.spec is None or spec.layout.fingerprint() != self.spec.layout.fingerprint():
            if self.spec is not None and self._ticks:
                raise VectorError("the controller was reset with a different ABI layout mid-run")
        self.spec = spec
        self.inner.reset(spec)
        self._pending_reset = True
        if isinstance(self.inner, CompiledController):
            hidden = hidden_warm_start_registers(spec)
            if hidden:
                warnings.warn(f"VectorRecorder: {hidden} are bound to the controller but their '.applied' readbacks "
                              f"are not exposed: the recorded CompiledController starts its held command from the "
                              f"channel init, not from the plant's warm-start registers, so the recorded run can "
                              f"differ from the in-process one", stacklevel=2)
            si = np.array(self.inner.spec_i64, dtype=np.int64, copy=True)
            sf = np.array(self.inner.spec_f64, dtype=np.float64, copy=True)
            if self._spec_arrays is not None and not (np.array_equal(si, self._spec_arrays[0])
                                                      and np.array_equal(sf, self._spec_arrays[1])):
                raise VectorError("spec arrays changed at a controller reset")
            self._spec_arrays = (si, sf)

    def _held_before(self, obs: Observation) -> np.ndarray:
        """The CompiledController's held command on entry of step (its own .applied rule on the first step)."""
        c = self.inner
        held = np.array(c.act, dtype=np.int64, copy=True)
        if not c._act_synced:
            for f in self.spec.layout.act:
                rb = f.name + ".applied"
                if rb in obs.sensors:
                    held[:, f.offset:f.offset + f.size] = np.asarray(obs.sensors[rb]).reshape(held.shape[0], -1)
        return held

    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        spec = self.spec
        if spec is None:
            raise RuntimeError("VectorRecorder: call reset(spec) first")
        lay = spec.layout
        B = spec.n_batch
        record = self.max_steps is None or len(self._ticks) < self.max_steps
        o = lay.pack_obs(obs.tick, obs.sensors, obs.new, obs.active_ring) if record else None
        compiled = isinstance(self.inner, CompiledController)
        act_in = self._held_before(obs) if compiled else None
        actions = self.inner.step(obs)
        if compiled:
            # the runner validates a bare CompiledController's command into its held vector (clip mode); mirror it
            clip_held_command(self.inner, spec)
        if record:
            a = np.zeros((B, lay.act_size), dtype=np.int64) if act_in is None else act_in.copy()
            wr = np.zeros(len(lay.act), dtype=bool)
            for j, f in enumerate(lay.act):
                if f.name in actions:
                    v = np.asarray(actions[f.name])
                    if v.dtype == np.uint64:
                        v = np.minimum(v, np.uint64(2**63 - 1))
                    a[:, f.offset:f.offset + f.size] = v.astype(np.int64).reshape(B, f.size)
                    wr[j] = True
            self._ticks.append(int(obs.tick))
            self._obs.append(o)
            self._act.append(a)
            if act_in is not None:
                self._act_in.append(act_in)
            self._written.append(wr)
            self._reset.append(self._pending_reset)
            self._pending_reset = False
        else:
            self.truncated = True
        return actions

    def stream(self, meta: Mapping[str, Any] | None = None) -> VectorStream:
        """The recorded streams (copies)."""
        spec = self.spec
        if spec is None:
            raise RuntimeError("nothing recorded: the controller was never reset")
        K, B = len(self._ticks), spec.n_batch
        lay = spec.layout
        obs = np.stack(self._obs) if K else np.zeros((0, B, lay.obs_size), np.int64)
        act = np.stack(self._act) if K else np.zeros((0, B, lay.act_size), np.int64)
        act_in = (np.stack(self._act_in) if self._act_in else None) if isinstance(self.inner, CompiledController) \
            else None
        if isinstance(self.inner, CompiledController) and act_in is None:
            act_in = np.zeros((0, B, lay.act_size), np.int64)
        m = {"controller": self.name, "controller_class": f"{type(self.inner).__module__}."
                                                          f"{type(self.inner).__qualname__}",
             "latency_ticks": self.latency_ticks, "truncated": bool(self.truncated), "n_batch": B,
             "n_rings": spec.n_rings, "tick_s": spec.tick_s, "period_ticks": spec.period_ticks,
             "samples_per_step": spec.samples_per_step}
        desc = getattr(self.inner, "describe", None)
        if callable(desc):
            m["controller_params"] = desc()
        m.update(dict(meta or {}))
        si, sf = (self._spec_arrays if self._spec_arrays is not None else (None, None))
        return VectorStream(
            ticks=np.asarray(self._ticks, dtype=np.int64), obs=obs, act=act,
            written=(np.stack(self._written) if K else np.zeros((0, len(lay.act)), bool)),
            reset_before=np.asarray(self._reset, dtype=bool), layout_rows=_json_rows(lay.describe()),
            layout_fingerprint=lay.fingerprint(), spec_fingerprint=spec.fingerprint(), act_in=act_in,
            spec_i64=si, spec_f64=sf, meta=m)


def _json_rows(rows: list[dict]) -> list[dict]:
    return [dict(r, shape=list(r["shape"])) for r in rows]


@dataclass
class RecordedRun:
    """A RunResult and the vectors recorded during it. Attribute access falls through to the RunResult."""
    result: Any
    vectors: VectorStream

    def __getattr__(self, item):
        if item in ("result", "vectors"):
            raise AttributeError(item)
        return getattr(self.result, item)


def run_recorded(cfg_or_params, controller, scenario=None, duration_s: float = 1e-3, *,
                 max_steps: int | None = None, **run_kwargs) -> RecordedRun:
    """``mrm.sim.runner.run`` with the controller wrapped in a :class:`VectorRecorder` (Python path).

    ``controller`` is an object or a registry name (``controller_params=`` as in run). The stream's ``meta`` gets
    the run's config fingerprint and hash, seeds, scenario name and fingerprint, duration and ticks.
    """
    from mrm.sim import runner
    if isinstance(controller, str):
        from mrm.control import registry
        controller = registry.create(controller, dict(run_kwargs.pop("controller_params", None) or {}))
    rec = VectorRecorder(controller, max_steps=max_steps)
    if run_kwargs.get("path", "auto") == "compiled":
        raise ValueError("vectors are recorded on the Python path (path='python' or 'auto')")
    res = runner.run(cfg_or_params, rec, scenario, duration_s, **run_kwargs)
    mi = res.manifest_inputs
    meta = {k: mi.get(k) for k in ("config_fingerprint", "config_hash", "seeds", "scenario", "scenario_fingerprint",
                                   "duration_s", "n_ticks", "dt_s", "spec_fingerprint")}
    meta["warm_start"] = {k: mi.get("warm_start", {}).get(k) for k in ("reset_protocol", "factory_table")}
    return RecordedRun(result=res, vectors=rec.stream(meta=_jsonable(meta)))


def _jsonable(x):
    return json.loads(json.dumps(x, default=_json_default))


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return repr(o)


# ---------------------------------------------------------------------------------------------------- files
def _stem(path) -> Path:
    p = Path(path)
    return p.with_suffix("") if p.suffix == ".npz" else p


def record_vectors(run, path, *, hex: bool = True, meta: Mapping[str, Any] | None = None) -> dict[str, Path]:
    """Write the vectors of ``run`` (a RecordedRun, VectorRecorder or VectorStream) to ``path`` (a stem; ``.npz``
    is stripped). Returns {'npz', 'obs_hex', 'act_hex', ['act_in_hex'], 'ticks_hex', 'written_hex', 'vh'} paths.
    The parent directory is created."""
    vec = stream_of(run)
    stem = _stem(path)
    stem.parent.mkdir(parents=True, exist_ok=True)
    m = dict(vec.meta)
    m.update(dict(meta or {}))
    head = {"format": FORMAT, "abi_version": ABI_VERSION, "layout_fingerprint": vec.layout_fingerprint,
            "spec_fingerprint": vec.spec_fingerprint, "layout": vec.layout_rows, "n_records": vec.n_records,
            "n_batch": vec.n_batch, "obs_size": vec.obs_size, "act_size": vec.act_size,
            "act_channels": [r["name"] for r in vec.act_fields], "sha256": vec.sha256(), "meta": m}
    arrays = vec.arrays()
    arrays["meta_json"] = np.array(json.dumps(head, sort_keys=True, default=_json_default))
    out: dict[str, Path] = {"npz": stem.with_suffix(".npz")}
    np.savez_compressed(out["npz"], **arrays)
    if hex:
        note = [f"{FORMAT} layout {vec.layout_fingerprint} spec {vec.spec_fingerprint} sha256 {head['sha256'][:16]}"]
        K, B = vec.n_records, vec.n_batch
        out["obs_hex"] = write_hex(stem.with_name(stem.name + ".obs.hex"), vec.obs,
                                   note + [f"obs_i64: K={K} B={B} OBS_SIZE={vec.obs_size}, order (k, b, i)"])
        out["act_hex"] = write_hex(stem.with_name(stem.name + ".act.hex"), vec.act,
                                   note + [f"act_i64 after step: K={K} B={B} ACT_SIZE={vec.act_size}, order (k, b, i)"])
        if vec.act_in is not None:
            out["act_in_hex"] = write_hex(stem.with_name(stem.name + ".act_in.hex"), vec.act_in,
                                          note + [f"act_i64 held on entry: K={K} B={B} ACT_SIZE={vec.act_size}"])
        out["ticks_hex"] = write_hex(stem.with_name(stem.name + ".ticks.hex"), vec.ticks, note + [f"ticks: K={K}"])
        bits = np.zeros(K, dtype=np.int64)
        for j in range(vec.written.shape[1]):
            bits |= vec.written[:, j].astype(np.int64) << j
        out["written_hex"] = write_hex(stem.with_name(stem.name + ".written.hex"), bits,
                                       note + ["bit j = action channel j written: "
                                               + ", ".join(f"{j}={r['name']}" for j, r in enumerate(vec.act_fields))])
        vh = stem.with_name(stem.name + ".layout.vh")
        vh.write_text(layout_vh(vec))
        out["vh"] = vh
    return out


def load_vectors(path) -> VectorStream:
    """Read a ``.npz`` written by :func:`record_vectors`; checks the format, the ABI version and the sha256."""
    p = Path(path)
    if p.suffix != ".npz":
        p = p.with_name(p.name + ".npz")
    with np.load(p, allow_pickle=False) as z:
        if "meta_json" not in z.files:
            raise VectorError(f"{p}: not a {FORMAT} file (no meta_json)")
        head = json.loads(str(z["meta_json"]))
        arrs = {k: np.array(z[k]) for k in z.files if k != "meta_json"}
    if head.get("format") != FORMAT:
        raise VectorError(f"{p}: format {head.get('format')!r}, this reader understands {FORMAT!r}")
    if int(head.get("abi_version", -1)) != ABI_VERSION:
        raise VectorError(f"{p}: ABI version {head.get('abi_version')} != {ABI_VERSION}")
    vec = VectorStream(ticks=arrs["ticks"], obs=arrs["obs"], act=arrs["act"], written=arrs["written"],
                       reset_before=arrs["reset_before"], layout_rows=head["layout"],
                       layout_fingerprint=head["layout_fingerprint"], spec_fingerprint=head["spec_fingerprint"],
                       act_in=arrs.get("act_in"), spec_i64=arrs.get("spec_i64"), spec_f64=arrs.get("spec_f64"),
                       meta=head.get("meta", {}))
    if vec.sha256() != head.get("sha256"):
        raise VectorError(f"{p}: sha256 of the arrays does not match the recorded one (file corrupted or edited)")
    return vec


def write_hex(path, words: np.ndarray, comments=()) -> Path:
    """Write int64 ``words`` (any shape, row-major) as 64-bit two's-complement hex, one word per line, after ``//``
    comment lines. Returns the path."""
    p = Path(path)
    w = np.ascontiguousarray(np.asarray(words, dtype=np.int64)).ravel().view(np.uint64)
    head = "".join(f"// {c}\n" for c in comments)
    body = "\n".join(f"{int(x):016x}" for x in w.tolist())
    p.write_text(head + body + ("\n" if w.size else ""))
    return p


def read_hex(path) -> np.ndarray:
    """Read a file of :func:`write_hex` (or any $readmemh file of <= 64-bit words, no @address lines) as int64."""
    vals = []
    for line in Path(path).read_text().splitlines():
        line = line.split("//", 1)[0].strip()
        if not line:
            continue
        for tok in line.split():
            if tok.startswith("@"):
                raise VectorError(f"{path}: @address records are not supported")
            vals.append(int(tok.replace("_", ""), 16))
    u = np.array(vals, dtype=np.uint64)
    return u.view(np.int64).copy()


def _vname(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", name).upper()


def layout_vh(vec: VectorStream) -> str:
    """Verilog localparams of the vectors' layout (offsets in 64-bit words within one record of one world)."""
    lines = [f"// generated by mrm.sim.vectors ({FORMAT}); layout {vec.layout_fingerprint}, spec {vec.spec_fingerprint}",
             "// word index of element i of channel X, record k, world b: (k*MRMV_B + b)*MRMV_<VEC>_SIZE + "
             "MRMV_<VEC>_<X> + i",
             f"localparam integer MRMV_ABI_VERSION = {ABI_VERSION};",
             f"localparam integer MRMV_K = {vec.n_records};",
             f"localparam integer MRMV_B = {vec.n_batch};",
             f"localparam integer MRMV_OBS_SIZE = {vec.obs_size};",
             f"localparam integer MRMV_ACT_SIZE = {vec.act_size};",
             f"localparam integer MRMV_OBS_TICK = {OBS_TICK};",
             f"localparam integer MRMV_OBS_ACTIVE_RING = {OBS_ACTIVE_RING};"]
    for r in vec.layout_rows:
        if r["id"] < 0:
            continue
        pre = {"obs": "OBS", "new": "NEW", "act": "ACT"}[r["vector"]]
        nm = _vname(r["name"])
        lines.append(f"localparam integer MRMV_{pre}_{nm} = {r['offset']};")
        lines.append(f"localparam integer MRMV_{pre}_{nm}_SIZE = {r['size']};")
    for j, r in enumerate(vec.act_fields):
        lines.append(f"localparam integer MRMV_WRITTEN_BIT_{_vname(r['name'])} = {j};")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------- replay
def observation_from_flat(spec: ControllerSpec, obs_i64: np.ndarray, tick: int | None = None) -> Observation:
    """An Observation (B worlds) from flat obs_i64 rows (B, obs_size) in ``spec.layout`` (channel dtypes), with the
    round-robin active ring of the header (07 P2-7) as ``active``."""
    o = np.asarray(obs_i64, dtype=np.int64)
    if o.ndim == 1:
        o = o[None, :]
    t, active_ring, sensors, new = spec.layout.unpack_obs(o)
    active = None
    if active_ring >= 0:
        active = np.zeros(spec.n_rings, dtype=bool)
        active[active_ring] = True
    return Observation(int(t if tick is None else tick), spec.tick_s, sensors, new, active)


@dataclass
class ReplayResult:
    """Actions of a replay and their comparison with the recording."""
    act: np.ndarray                 # (K, B, act_size) int64
    written: np.ndarray             # (K, n_act_fields) bool
    mismatches: int                 # number of (k, b, element) differences on written channels
    first: dict | None              # the first mismatch {k, tick, b, channel, index, recorded, replayed}

    @property
    def ok(self) -> bool:
        return self.mismatches == 0


def _check_layout(spec: ControllerSpec, vec: VectorStream, check_spec: bool) -> None:
    if spec.layout.fingerprint() != vec.layout_fingerprint:
        raise VectorError(f"layout {spec.layout.fingerprint()} of this spec != {vec.layout_fingerprint} of the "
                          f"vectors")
    if check_spec and spec.fingerprint() != vec.spec_fingerprint:
        raise VectorError(f"ControllerSpec {spec.fingerprint()} != {vec.spec_fingerprint} of the vectors (pass "
                          f"check_spec=False to replay against another spec with the same layout)")
    if spec.n_batch != vec.n_batch:
        raise VectorError(f"spec B = {spec.n_batch}, vectors B = {vec.n_batch}")


def replay(controller, vec: VectorStream, spec: ControllerSpec, *, check_spec: bool = True) -> ReplayResult:
    """Feed the recorded observations to ``controller`` through the Python protocol (reset at every
    ``reset_before``) and compare its actions with the recorded ones on the written channels.

    ``spec`` must have the vectors' layout (and, unless ``check_spec=False``, their spec fingerprint). For a
    CompiledController the held command is clipped to the channel ranges after each step, as the recording did
    (the runner's clip of a bare CompiledController's command).
    """
    _check_layout(spec, vec, check_spec)
    lay = spec.layout
    K, B = vec.n_records, vec.n_batch
    out = np.zeros((K, B, lay.act_size), dtype=np.int64)
    wr = np.zeros((K, len(lay.act)), dtype=bool)
    compiled = isinstance(controller, CompiledController)
    for k in range(K):
        if vec.reset_before[k]:
            controller.reset(spec)
        obs = observation_from_flat(spec, vec.obs[k], int(vec.ticks[k]))
        actions = controller.step(obs)
        if compiled:                      # the plant's clip of the held command, as in the recording
            clip_held_command(controller, spec)
        held = vec.act_in[k] if (vec.act_in is not None and vec.act_in.shape[0] == K) else None
        a = np.zeros((B, lay.act_size), np.int64) if held is None else held.copy()
        for j, f in enumerate(lay.act):
            if f.name in actions:
                a[:, f.offset:f.offset + f.size] = np.asarray(actions[f.name]).astype(np.int64).reshape(B, f.size)
                wr[k, j] = True
        out[k] = a
    return _result(vec, out, wr)


def replay_abi(init_fn: Callable, step_fn: Callable, vec: VectorStream, *, spec_i64: np.ndarray | None = None,
               spec_f64: np.ndarray | None = None) -> ReplayResult:
    """Replay at the compiled ABI, without a ControllerSpec (the RTL test-bench view, 07 P0-9):
    ``p, pf, s, sf = init_fn(spec_i64, spec_f64)`` per world; at each record ``act = act_in[k]`` (the held command)
    and ``step_fn(p, pf, s, sf, obs[k, b], act)``. Spec arrays default to the recorded ones (CompiledController
    recordings). A ``reset_before`` record calls ``init_fn`` again."""
    si = vec.spec_i64 if spec_i64 is None else np.asarray(spec_i64, np.int64)
    sf = vec.spec_f64 if spec_f64 is None else np.asarray(spec_f64, np.float64)
    if si is None or sf is None:
        raise VectorError("no spec arrays in these vectors (recorded from a Python controller): pass spec_i64 and "
                          "spec_f64")
    if vec.act_in is None or vec.act_in.shape[0] != vec.n_records:
        raise VectorError("replay_abi needs the held commands act_in (a CompiledController recording)")
    K, B = vec.n_records, vec.n_batch
    out = np.zeros_like(vec.act)
    state = None
    for k in range(K):
        if state is None or vec.reset_before[k]:
            state = []
            for _b in range(B):
                p, pf, s, sfs = init_fn(si, sf)
                state.append((np.array(p, np.int64), np.array(pf, np.float64), np.array(s, np.int64),
                              np.array(sfs, np.float64)))
        for b in range(B):
            a = np.array(vec.act_in[k, b], dtype=np.int64, copy=True)
            o = np.ascontiguousarray(vec.obs[k, b])
            p, pf, s, sfs = state[b]
            step_fn(p, pf, s, sfs, o, a)
            out[k, b] = a
    wr = np.ones((K, len(vec.act_fields)), dtype=bool)
    return _result(vec, out, wr)


def compare_actions(vec: VectorStream, act: np.ndarray, written: np.ndarray | None = None) -> tuple[int, dict | None]:
    """(number of differing elements on written channels, first mismatch or None) of ``act`` against ``vec``."""
    act = np.asarray(act, dtype=np.int64)
    if act.shape != vec.act.shape:
        raise VectorError(f"action shape {act.shape} != recorded {vec.act.shape}")
    wr = vec.written if written is None else np.asarray(written, bool)
    mask = np.zeros(vec.act.shape, dtype=bool)
    for j, r in enumerate(vec.act_fields):
        mask[:, :, r["offset"]:r["offset"] + r["size"]] = wr[:, j][:, None, None]
    diff = (act != vec.act) & mask
    n = int(diff.sum())
    if not n:
        return 0, None
    k, b, i = (int(x) for x in np.argwhere(diff)[0])
    ch = next(r for r in vec.act_fields if r["offset"] <= i < r["offset"] + r["size"])
    return n, {"k": k, "tick": int(vec.ticks[k]), "b": b, "channel": ch["name"], "index": i - ch["offset"],
               "recorded": int(vec.act[k, b, i]), "replayed": int(act[k, b, i])}


def _result(vec: VectorStream, out: np.ndarray, wr: np.ndarray) -> ReplayResult:
    if not np.array_equal(wr, vec.written):
        warnings.warn("the replayed controller returned other channels than the recorded one; only channels "
                      "written in the recording are compared", stacklevel=3)
    n, first = compare_actions(vec, out)
    return ReplayResult(act=out, written=wr, mismatches=n, first=first)
