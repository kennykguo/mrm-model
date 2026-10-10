"""Binary lockstep bridge for external controllers (07 P1-21).

An external controller process (Python, C, an RTL simulator, or real controller hardware behind a socket) drives
the plant in lockstep: at every controller edge the plant sends one binary observation frame and waits for one
action frame. Frames carry the edge's plant tick, never wall-clock time, so a bridged run is bit-identical to the same
controller run in-process (tests/verification/test_bridge.py checks this for a Python and a C controller).

Frame layout (generated from the ChannelSpecs, 07 P0-7)
------------------------------------------------------
Every frame starts with an 8-byte header ``{u8 'M', u8 'B', u8 version, u8 kind, u32 payload_bytes}``
(little-endian, like every number below). Kinds: HELLO 1 and BYE 5 (plant -> controller), READY 2 and ACT 4
(controller -> plant), OBS 3 (plant -> controller), ERROR 6 (either way; payload = utf-8 message).

    OBS payload: u64 tick, then B world records of the observation fields
    ACT payload: u64 tick (echo of the OBS tick), u32 write_mask (bit j: action field j written this step; a field
                 not written holds its last command, like a channel left out of a Python controller's dict),
                 u32 reserved = 0, then B world records of the action fields

A world record is the fields packed without padding, in order. :class:`FrameLayout` builds them from a
ControllerSpec: one field per non-bool channel, its elements row-major, on the narrowest little-endian integer type
that holds the channel's [min, max] (u16 for 16-bit codes, u64 for tick and count channels); bool channels and the
new-sample flags are packed as bits into ``flags`` (one word per ring, channels of shape (B, N)) and ``dflags``
(one word per world, channels of shape (B, 1)), each the narrowest of u8/u16/u32/u64 that holds its bits. For each
included channel: a bool channel gives a value bit then a new bit, any other channel its new bit, in channel order.
Profiles:

- ``minimal`` (07 P1-21's frame ``{u64 tick, u16 adc[N], i16 temp[S], u8 flags} -> {u64 tick, u16 dac[N],
  u16 bias[N]}``): ``tx.monitor.latest`` (adc), every ``temp.*`` (temp; 12-bit codes, so u16), the latched registers
  ``<actuator>.applied`` of the actions in the frame, and the flags (new sample, ``monitor_valid``,
  ``adc_overrange``); actions ``tx.heater`` (dac) and ``tx.bias`` (only when bound to the controller);
- ``full``: every sensor channel the ControllerSpec exposes and every actuator bound to the controller;
- explicit ``sensors=`` / ``actuators=`` lists (profile ``custom``).

In every profile a sample block brings its count channel and a framed action brings its latched register
``<actuator>.applied`` when the spec exposes it (the held command of a warm start). The layout fingerprint covers
every field's wire type, offset, count and channel range.

Handshake and versioning
------------------------
HELLO carries the protocol version, the frame layout fingerprint, the ControllerSpec fingerprint, the profile and
the dimensions (B, N, L, S, M, period, record sizes, tick_s, the controller latency the plant was built with) and the
layout as JSON. The external side rebuilds the layout from its own ControllerSpec and answers READY with the layout
fingerprint and its ``latency_ticks``; any difference is an error before the first tick (:class:`BridgeError`). The
runner reads ``latency_ticks`` before the handshake, so it is declared on :class:`BridgeController` and checked.

Use
---
Plant side::

    from mrm.sim import bridge, runner
    with bridge.BridgeController(bridge.spawn([".venv/bin/python", "-m", "mrm.sim.bridge", "serve", "--preset",
                                                "g26", "--controller", "mrm.control.fxp:FxpPi"])) as ctrl:
        res = runner.run("g26", ctrl, "static", duration_s=10e-3, ...)
    ctrl.stats   # round trips, wall time per round trip

External side (Python): :func:`serve` (``python -m mrm.sim.bridge serve ...`` builds the config, the scenario and
the spec exactly as the runner does: :func:`spec_for`); in C: :func:`c_header` generates ``mrm_bridge.h`` from a
layout, :data:`C_FXP_PI_SOURCE` is a complete C controller (the twin of mrm.control.fxp.FxpPi).

Bit-identity and its conditions: the controller sees only the channels in the frame. A Python controller that
reads another channel gets UndeclaredChannelError on the external side, never a silent zero; a CompiledController
finds offset -1 for it (``obs_offset``), exactly as when ``controller_interface.expose`` hides the channel, and
behaves as it would in-process with that channel hidden, so use the ``full`` profile unless the controller reads
only the framed channels. The external endpoint applies the same ActionValidator as the runner before encoding
(codes out of range are clipped or raise per ``controller_interface.on_out_of_range``), so the plant receives the
same commands, and it clips a CompiledController's held command as the runner and the kernel do. The saturation
counts are then the endpoint's (``serve`` returns them); the runner's own count of a bridged run is 0. An action the
controller writes outside the frame must equal the plant's register (else BridgeError). Controllers with a
``fast_kernel`` (per-sample I/O) and Truth-fed controllers are refused. A tick mismatch, a short read, a peer that
exits or (with ``timeout_s``) a silent socket peer raises BridgeError.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import struct
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Mapping, Sequence

import numpy as np

from mrm.control.api import ActionValidator, ControllerSpec, Observation
from mrm.core.channels import ChannelSpec

__all__ = ["PROTOCOL_VERSION", "MAGIC", "KIND", "BridgeError", "WireField", "FlagBit", "FrameLayout", "Transport",
           "StreamTransport", "SocketTransport", "spawn", "stdio", "connect", "listen", "write_frame", "read_frame",
           "Hello", "BridgeController", "serve", "spec_for", "restrict_spec", "c_header", "C_FXP_PI_SOURCE",
           "add_arguments", "run_command", "main", "PROFILES"]

PROTOCOL_VERSION = 1
MAGIC = b"MB"
KIND = {"HELLO": 1, "READY": 2, "OBS": 3, "ACT": 4, "BYE": 5, "ERROR": 6}
_KIND_NAME = {v: k for k, v in KIND.items()}
_HDR = struct.Struct("<2sBBI")                 # magic, version, kind, payload bytes
_HELLO = struct.Struct("<16s16s16sIIIIIIIIIIidI")
_READY = struct.Struct("<16siI32s")
PROFILES = ("minimal", "full")
_MAX_PAYLOAD = 1 << 24          # 16 MiB: far above any frame (g26 full profile: 539 bytes per world), so a corrupted
                                # length word fails at once instead of allocating up to 1 GiB


class BridgeError(RuntimeError):
    """Protocol violation, version or layout mismatch, a remote error, or a lost peer."""


# ================================================================================================ frame layout
def _wire_int(lo: int, hi: int, name: str) -> str:
    """Narrowest little-endian numpy integer type holding [lo, hi]."""
    if lo >= 0:
        for t in ("<u1", "<u2", "<u4", "<u8"):
            if hi <= np.iinfo(np.dtype(t)).max:
                return t
    else:
        for t in ("<i1", "<i2", "<i4", "<i8"):
            info = np.iinfo(np.dtype(t))
            if lo >= info.min and hi <= info.max:
                return t
    raise BridgeError(f"{name}: range [{lo}, {hi}] fits no 64-bit integer")


def _flag_word(nbits: int, group: str) -> str:
    for t, n in (("<u1", 8), ("<u2", 16), ("<u4", 32), ("<u8", 64)):
        if nbits <= n:
            return t
    raise BridgeError(f"{nbits} flag bits in {group}: more than 64")


@dataclass(frozen=True)
class WireField:
    """One field of a world record: ``name`` (the channel name, or 'flags' / 'dflags'), ``dtype`` (little-endian
    numpy type string), ``count`` elements, byte ``offset`` within the record, element ``shape`` (without B)."""
    name: str
    dtype: str
    count: int
    offset: int
    shape: tuple

    @property
    def nbytes(self) -> int:
        return np.dtype(self.dtype).itemsize * self.count


@dataclass(frozen=True)
class FlagBit:
    """Bit ``bit`` of the ``group`` word ('flags' per ring, 'dflags' per world) carries ``what`` ('value' of a bool
    channel, or 'new': its new-sample flag) of ``channel``."""
    channel: str
    what: str
    group: str
    bit: int


@dataclass(frozen=True)
class FrameLayout:
    """Observation and action records of the bridge, generated from a ControllerSpec (module docstring)."""
    profile: str
    n_batch: int
    n_rings: int
    obs_fields: tuple
    act_fields: tuple
    flag_bits: tuple
    obs_record_bytes: int
    act_record_bytes: int
    channels: dict = field(repr=False, compare=False, hash=False, default_factory=dict)

    # -- construction --
    @classmethod
    def from_spec(cls, spec: ControllerSpec, profile: str = "full", *, sensors: Sequence[str] | None = None,
                  actuators: Sequence[str] | None = None) -> "FrameLayout":
        """Layout for ``spec``: a profile ('minimal' | 'full'), or explicit channel lists (profile 'custom')."""
        obs_order = [f.name for f in spec.layout.obs]
        act_order = [f.name for f in spec.layout.act]
        if sensors is not None or actuators is not None:
            prof = "custom"
            sens = list(sensors if sensors is not None else obs_order)
            acts = list(actuators if actuators is not None else act_order)
        elif profile == "full":
            prof, sens, acts = "full", obs_order, act_order
        elif profile == "minimal":
            prof = "minimal"
            acts = [a for a in ("tx.heater", "tx.bias") if a in act_order]
            want = {"tx.monitor.latest", "monitor_valid", "adc_overrange"} | {a + ".applied" for a in acts}
            sens = [n for n in obs_order if n in want or n.startswith("temp.")]
        else:
            raise BridgeError(f"unknown bridge profile {profile!r}; use one of {PROFILES} or sensors=/actuators=")
        unknown = [n for n in sens if n not in obs_order] + [n for n in acts if n not in act_order]
        if unknown:
            raise BridgeError(f"channels {unknown} are not sensors exposed to / actuators bound to the controller "
                              f"in this spec")
        for n in list(sens):                     # a sample block is meaningless without its valid-entry count
            cc = spec.channels[n].count_channel if n in spec.channels else None
            if cc is not None and cc not in sens:
                sens.append(cc)
        for a in acts:                           # a framed action brings its latched register (the held command of
            rb = a + ".applied"                  # a warm start; CompiledController pre-fills act_i64 from it)
            if rb in obs_order and rb not in sens:
                sens.append(rb)
        sens = [n for n in obs_order if n in set(sens)]            # spec order, whatever the request order
        acts = [n for n in act_order if n in set(acts)]
        if len(acts) > 32:
            raise BridgeError("more than 32 action fields (write_mask is 32 bits)")
        dims = spec.dims
        chans: dict[str, ChannelSpec] = {n: spec.channels[n] for n in sens + acts}
        fields: list[WireField] = []
        bits: dict[str, list[FlagBit]] = {"flags": [], "dflags": []}
        off = 0

        def group_of(shape: tuple, name: str) -> str:
            if shape == (int(spec.n_rings),) and "N" in spec.channels[name].shape:
                return "flags"
            if shape == (1,):
                return "dflags"
            raise BridgeError(f"{name}: flag shape {shape} is neither per ring (N,) nor per world (1,)")

        for n in sens:
            ch = chans[n]
            es = ch.element_shape(dims)
            fs = ch.flag_shape(dims)
            if ch.np_dtype.kind == "b":
                g = group_of(es, n)
                bits[g].append(FlagBit(n, "value", g, len(bits[g])))
            else:
                cnt = int(np.prod(es)) if es else 1
                fields.append(WireField(n, _wire_int(int(ch.min), int(ch.max), n), cnt, off, tuple(es)))
                off += fields[-1].nbytes
            g = group_of(fs if fs else (1,), n)
            bits[g].append(FlagBit(n, "new", g, len(bits[g])))
        for g, cnt in (("flags", int(spec.n_rings)), ("dflags", 1)):
            if bits[g]:
                fields.append(WireField(g, _flag_word(len(bits[g]), g), cnt, off, (cnt,)))
                off += fields[-1].nbytes
        obs_bytes = off
        afields: list[WireField] = []
        off = 0
        for n in acts:
            ch = chans[n]
            es = ch.element_shape(dims)
            cnt = int(np.prod(es)) if es else 1
            afields.append(WireField(n, _wire_int(int(ch.min), int(ch.max), n), cnt, off, tuple(es)))
            off += afields[-1].nbytes
        if max(8 + int(spec.n_batch) * obs_bytes, 16 + int(spec.n_batch) * off) > _MAX_PAYLOAD:
            raise BridgeError(f"B = {spec.n_batch} worlds of {obs_bytes} bytes exceed the {_MAX_PAYLOAD}-byte frame "
                              f"limit; bridge fewer worlds or a smaller profile")
        return cls(profile=prof, n_batch=int(spec.n_batch), n_rings=int(spec.n_rings), obs_fields=tuple(fields),
                   act_fields=tuple(afields), flag_bits=tuple(bits["flags"] + bits["dflags"]),
                   obs_record_bytes=obs_bytes, act_record_bytes=off, channels=chans)

    # -- description --
    @property
    def sensors(self) -> list[str]:
        """Sensor channels carried (value fields and bool flag channels), in order."""
        out = [f.name for f in self.obs_fields if f.name not in ("flags", "dflags")]
        out += [b.channel for b in self.flag_bits if b.what == "value"]
        return [n for n in self.channels if n in set(out)]

    @property
    def actuators(self) -> list[str]:
        return [f.name for f in self.act_fields]

    @property
    def obs_payload_bytes(self) -> int:
        return 8 + self.n_batch * self.obs_record_bytes

    @property
    def act_payload_bytes(self) -> int:
        return 16 + self.n_batch * self.act_record_bytes

    def _row(self, f: WireField) -> dict:
        d = dict(name=f.name, dtype=f.dtype, count=f.count, offset=f.offset, shape=list(f.shape))
        if f.name in self.channels:          # the channel range: both sides range-check and clamp to it, so it is
            ch = self.channels[f.name]       # part of the contract (a C twin compiled for a 12-bit heater must not
            d.update(min=int(ch.min), max=int(ch.max))   # accept a 16-bit plant with the same u16 wire type)
        return d

    def describe(self) -> dict:
        """JSON-serializable description (sent in HELLO; hashed by :meth:`fingerprint`): every field's wire type,
        count, offset, element shape and channel range [min, max], the flag bits and the channel lists."""
        return {"protocol": PROTOCOL_VERSION, "profile": self.profile, "B": self.n_batch, "N": self.n_rings,
                "obs": [self._row(f) for f in self.obs_fields],
                "act": [self._row(f) for f in self.act_fields],
                "flag_bits": [dict(channel=b.channel, what=b.what, group=b.group, bit=b.bit) for b in self.flag_bits],
                "obs_record_bytes": self.obs_record_bytes, "act_record_bytes": self.act_record_bytes,
                "sensors": self.sensors, "actuators": self.actuators}

    def fingerprint(self) -> str:
        """sha256 (16 hex) of :meth:`describe`."""
        return hashlib.sha256(json.dumps(self.describe(), sort_keys=True).encode()).hexdigest()[:16]

    def _np_dtype(self, fields: tuple) -> np.dtype:
        return np.dtype({"names": [f.name for f in fields],
                         "formats": [(f.dtype, (f.count,)) if f.count > 1 or f.name in ("flags",) else f.dtype
                                     for f in fields],
                         "offsets": [f.offset for f in fields],
                         "itemsize": sum(f.nbytes for f in fields)})

    @property
    def obs_dtype(self) -> np.dtype:
        return self._np_dtype(self.obs_fields)

    @property
    def act_dtype(self) -> np.dtype:
        return self._np_dtype(self.act_fields)

    # -- codec --
    @cached_property
    def _plan(self) -> dict:
        """Per-field codec tables, built once (the layout is immutable)."""
        B = self.n_batch
        vals = []
        for f in self.obs_fields:
            if f.name in ("flags", "dflags"):
                continue
            ch = self.channels[f.name]
            vals.append((f.name, np.dtype(f.dtype), f.count, f.offset, f.nbytes, (B,) + f.shape, ch.np_dtype))
        lo = np.concatenate([np.full(v[2], int(self.channels[v[0]].min), np.int64) for v in vals]) if vals \
            else np.zeros(0, np.int64)
        hi = np.concatenate([np.full(v[2], int(self.channels[v[0]].max), np.int64) for v in vals]) if vals \
            else np.zeros(0, np.int64)
        groups = []
        for f in self.obs_fields:
            if f.name not in ("flags", "dflags"):
                continue
            bits = []
            for b in self.flag_bits:
                if b.group != f.name:
                    continue
                shp = (B, f.count)
                if b.what == "value":
                    shp = (B,) + self.channels[b.channel].element_shape({"B": B, "N": self.n_rings, "M": 1, "L": 1,
                                                                         "S": 1})
                bits.append((b.what, b.channel, np.uint64(b.bit), shp))
            shifts = np.array([int(b[2]) for b in bits], dtype=np.uint64)
            groups.append((f.name, f.count, np.dtype(f.dtype), f.offset, f.nbytes, bits, shifts))
        acts = []
        for f in self.act_fields:
            ch = self.channels[f.name]
            acts.append((f.name, np.dtype(f.dtype), f.count, f.offset, f.nbytes, (B,) + f.shape, ch.np_dtype,
                         int(ch.min), int(ch.max)))
        return {"vals": vals, "lo": lo, "hi": hi, "groups": groups, "acts": acts}

    def _range_error(self, names_counts, a: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> BridgeError:
        bad = np.flatnonzero(((a < lo) | (a > hi)).any(axis=0))[0]
        pos = 0
        for name, cnt in names_counts:
            if bad < pos + cnt:
                ch = self.channels[name]
                return BridgeError(f"{name}: value outside the channel range [{ch.min}, {ch.max}] cannot be framed")
            pos += cnt
        return BridgeError("value outside its channel range")              # pragma: no cover

    def encode_obs(self, tick: int, sensors: Mapping[str, np.ndarray], new: Mapping[str, np.ndarray]) -> bytes:
        """OBS payload from (B, ...) sensor and new-flag arrays (only the channels of this layout are read). Every
        value is checked against its channel range before the cast (a cast would wrap it)."""
        B = self.n_batch
        pl = self._plan
        buf = np.empty((B, self.obs_record_bytes), dtype=np.uint8)
        vals = pl["vals"]
        if vals:
            cols = []
            for name, _wdt, cnt, *_ in vals:
                a = np.asarray(sensors[name])
                if a.dtype == np.uint64:
                    a = np.minimum(a, np.uint64(2**63 - 1))
                cols.append(a.reshape(B, cnt).astype(np.int64, copy=False))
            allv = np.concatenate(cols, axis=1)
            if (allv < pl["lo"]).any() or (allv > pl["hi"]).any():
                raise self._range_error([(v[0], v[2]) for v in vals], allv, pl["lo"], pl["hi"])
            for (name, wdt, cnt, off, nb, _shp, _cdt), c in zip(vals, cols):
                buf[:, off:off + nb] = c.astype(wdt).view(np.uint8).reshape(B, nb)
        for _g, cnt, wdt, off, nb, bits, shifts in pl["groups"]:
            m = np.stack([np.asarray(sensors[ch] if what == "value" else new[ch]).reshape(B, cnt)
                          for what, ch, _bit, _shp in bits], axis=-1)
            w = np.bitwise_or.reduce(m.astype(np.uint64) << shifts, axis=-1)
            buf[:, off:off + nb] = w.astype(wdt).view(np.uint8).reshape(B, nb)
        return struct.pack("<Q", int(tick)) + buf.tobytes()

    def decode_obs(self, payload: bytes) -> tuple[int, dict[str, np.ndarray], dict[str, np.ndarray]]:
        """(tick, sensors, new) from an OBS payload; arrays (B, ...) in the channel dtypes."""
        if len(payload) != self.obs_payload_bytes:
            raise BridgeError(f"OBS payload of {len(payload)} bytes, layout needs {self.obs_payload_bytes}")
        B = self.n_batch
        pl = self._plan
        tick = struct.unpack_from("<Q", payload, 0)[0]
        buf = np.frombuffer(payload, dtype=np.uint8, offset=8).reshape(B, self.obs_record_bytes)
        sensors: dict[str, np.ndarray] = {}
        new: dict[str, np.ndarray] = {}
        for name, wdt, cnt, off, nb, shp, cdt in pl["vals"]:
            sensors[name] = np.ascontiguousarray(buf[:, off:off + nb]).view(wdt).reshape(shp).astype(cdt)
        for _g, cnt, wdt, off, nb, bits, shifts in pl["groups"]:
            w = np.ascontiguousarray(buf[:, off:off + nb]).view(wdt).reshape(B, cnt, 1).astype(np.uint64)
            allb = ((w >> shifts) & np.uint64(1)).astype(bool)                  # (B, cnt, bits)
            for j, (what, ch, _bit, shp) in enumerate(bits):
                v = np.ascontiguousarray(allb[:, :, j])
                if what == "value":
                    sensors[ch] = v.reshape(shp)
                else:
                    new[ch] = v
        return int(tick), sensors, new

    def encode_act(self, tick: int, actions: Mapping[str, np.ndarray]) -> bytes:
        """ACT payload: the channels of this layout present in ``actions`` set their write_mask bit."""
        B = self.n_batch
        buf = np.zeros((B, self.act_record_bytes), dtype=np.uint8)
        mask = 0
        for j, (name, wdt, cnt, off, nb, _shp, _cdt, lo, hi) in enumerate(self._plan["acts"]):
            if name not in actions:
                continue
            a = np.asarray(actions[name])
            if a.dtype.kind not in "biu":
                raise BridgeError(f"{name}: integer codes expected, got {a.dtype}")
            if a.dtype == np.uint64:
                a = np.minimum(a, np.uint64(2**63 - 1))
            a = a.reshape(B, cnt).astype(np.int64, copy=False)
            if a.size and (a.min() < lo or a.max() > hi):
                raise BridgeError(f"{name}: value outside the channel range [{lo}, {hi}] cannot be framed")
            buf[:, off:off + nb] = a.astype(wdt).view(np.uint8).reshape(B, nb)
            mask |= 1 << j
        return struct.pack("<QII", int(tick), mask, 0) + buf.tobytes()

    def decode_act(self, payload: bytes) -> tuple[int, int, dict[str, np.ndarray]]:
        """(tick, write_mask, {written channel: (B, ...) array in the channel dtype}) from an ACT payload."""
        if len(payload) != self.act_payload_bytes:
            raise BridgeError(f"ACT payload of {len(payload)} bytes, layout needs {self.act_payload_bytes}")
        B = self.n_batch
        tick, mask, _res = struct.unpack_from("<QII", payload, 0)
        if mask >> len(self.act_fields):
            raise BridgeError(f"write_mask 0x{mask:x} sets bits beyond the {len(self.act_fields)} action fields")
        buf = np.frombuffer(payload, dtype=np.uint8, offset=16).reshape(B, self.act_record_bytes)
        out = {}
        for j, (name, wdt, cnt, off, nb, shp, cdt, lo, hi) in enumerate(self._plan["acts"]):
            if mask >> j & 1:
                a = np.ascontiguousarray(buf[:, off:off + nb]).view(wdt).reshape(shp).astype(np.int64)
                if a.size and (a.min() < lo or a.max() > hi):
                    raise BridgeError(f"{name}: value outside the channel range [{lo}, {hi}] in an ACT frame")
                out[name] = a.astype(cdt)
        return int(tick), int(mask), out


# ================================================================================================ transports
class Transport:
    """Byte stream with exact reads. Subclasses implement ``_send`` / ``_recv_into`` / ``close``."""

    def send(self, data: bytes) -> None:
        raise NotImplementedError

    def recv_exact(self, n: int, eof_ok: bool = False) -> bytes:
        """Exactly n bytes; at an end of stream before the first byte: b"" if ``eof_ok``, else BridgeError."""
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class StreamTransport(Transport):
    """Unbuffered pipe ends (file descriptors). ``proc``: a child process to wait for on close."""

    def __init__(self, rfd: int, wfd: int, *, proc: subprocess.Popen | None = None, own_fds: bool = False):
        self.rfd, self.wfd, self.proc, self.own_fds = int(rfd), int(wfd), proc, own_fds
        self._closed = False

    def send(self, data: bytes) -> None:
        mv = memoryview(data)
        while mv:
            try:
                k = os.write(self.wfd, mv)
            except BrokenPipeError:
                raise BridgeError(self._lost("write")) from None
            mv = mv[k:]

    def recv_exact(self, n: int, eof_ok: bool = False) -> bytes:
        buf = bytearray(n)
        mv = memoryview(buf)
        got = 0
        while got < n:
            chunk = os.read(self.rfd, n - got)
            if not chunk:
                if eof_ok and got == 0:
                    return b""
                raise BridgeError(self._lost("read") + f" ({got} of {n} bytes)")
            mv[got:got + len(chunk)] = chunk
            got += len(chunk)
        return bytes(buf)

    def _lost(self, what: str) -> str:
        rc = None if self.proc is None else self.proc.poll()
        return f"bridge peer closed the stream during {what}" + ("" if rc is None else f" (exit code {rc})")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.proc is not None:
            for f in (self.proc.stdin,):
                try:
                    f.close()
                except OSError:
                    pass
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            if self.proc.stdout is not None:
                self.proc.stdout.close()
        elif self.own_fds:
            for fd in {self.rfd, self.wfd}:
                try:
                    os.close(fd)
                except OSError:
                    pass


class SocketTransport(Transport):
    """A connected stream socket (TCP_NODELAY set for TCP; a socketpair works too). ``timeout_s``: a lost-peer
    guard on every read and write (wall clock; it never affects the lockstep, which runs on ticks): a hung hardware
    peer raises BridgeError after that long instead of blocking the run. None (default) keeps the socket's own
    setting (blocking unless set)."""

    def __init__(self, sock: socket.socket, *, timeout_s: float | None = None):
        self.sock = sock
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if timeout_s is not None:
            sock.settimeout(float(timeout_s))

    def _timeout(self, what: str) -> BridgeError:
        return BridgeError(f"bridge peer silent for {self.sock.gettimeout()} s during {what} (timeout_s)")

    def send(self, data: bytes) -> None:
        try:
            self.sock.sendall(data)
        except (BrokenPipeError, ConnectionResetError):
            raise BridgeError("bridge peer closed the socket during write") from None
        except TimeoutError:
            raise self._timeout("write") from None

    def recv_exact(self, n: int, eof_ok: bool = False) -> bytes:
        buf = bytearray(n)
        mv = memoryview(buf)
        got = 0
        while got < n:
            try:
                k = self.sock.recv_into(mv[got:], n - got)
            except TimeoutError:
                raise self._timeout(f"read ({got} of {n} bytes)") from None
            except ConnectionResetError:
                raise BridgeError(f"bridge peer reset the connection during read ({got} of {n} bytes)") from None
            if k == 0:
                if eof_ok and got == 0:
                    return b""
                raise BridgeError(f"bridge peer closed the socket during read ({got} of {n} bytes)")
            got += k
        return bytes(buf)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def spawn(cmd: Sequence[str], *, env: Mapping[str, str] | None = None, cwd: str | None = None,
          stderr=None) -> StreamTransport:
    """Start an external controller process; frames go over its stdin / stdout (unbuffered)."""
    proc = subprocess.Popen(list(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, bufsize=0,
                            env=None if env is None else dict(env), cwd=cwd)
    return StreamTransport(proc.stdout.fileno(), proc.stdin.fileno(), proc=proc)


def stdio() -> StreamTransport:
    """The external side's transport over its own stdin / stdout. File descriptor 1 is then pointed at stderr, so a
    stray print cannot corrupt the frame stream (frames use a private duplicate of the original stdout)."""
    sys.stdout.flush()
    wfd = os.dup(1)
    os.dup2(2, 1)
    return StreamTransport(0, wfd, own_fds=False)


def connect(host: str, port: int, *, timeout_s: float | None = None) -> SocketTransport:
    """Connect to a listening peer (TCP). ``timeout_s`` bounds connect and every read (a lost-peer guard; it never
    affects the lockstep, which runs on ticks)."""
    s = socket.create_connection((host, int(port)), timeout=timeout_s)
    s.settimeout(timeout_s)
    return SocketTransport(s, timeout_s=timeout_s)


def listen(host: str = "127.0.0.1", port: int = 0) -> socket.socket:
    """A listening TCP socket (port 0 picks a free port: ``sock.getsockname()[1]``). Accept with
    ``SocketTransport(sock.accept()[0])``."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, int(port)))
    s.listen(1)
    return s


def write_frame(tr: Transport, kind: int, payload: bytes = b"") -> None:
    """Send one frame."""
    tr.send(_HDR.pack(MAGIC, PROTOCOL_VERSION, int(kind), len(payload)) + payload)


def read_frame(tr: Transport, *, eof_ok: bool = False) -> tuple[int, bytes] | None:
    """Read one frame: (kind, payload). An ERROR frame raises BridgeError with the peer's message. With ``eof_ok``,
    an end of stream at a frame boundary returns None."""
    head = tr.recv_exact(_HDR.size, eof_ok=eof_ok)
    if not head:
        return None
    magic, ver, kind, n = _HDR.unpack(head)
    if magic != MAGIC:
        raise BridgeError(f"bad frame magic {magic!r} (expected {MAGIC!r}); the stream is out of sync")
    if ver != PROTOCOL_VERSION:
        raise BridgeError(f"peer speaks bridge protocol {ver}, this side {PROTOCOL_VERSION}")
    if kind not in _KIND_NAME:
        raise BridgeError(f"unknown frame kind {kind}")
    if n > _MAX_PAYLOAD:
        raise BridgeError(f"frame payload of {n} bytes is implausible")
    payload = tr.recv_exact(n) if n else b""
    if kind == KIND["ERROR"]:
        raise BridgeError("peer error: " + payload.decode("utf-8", "replace"))
    return kind, payload


def _expect(tr: Transport, kind: str) -> bytes:
    k, p = read_frame(tr)
    if k != KIND[kind]:
        raise BridgeError(f"expected a {kind} frame, got {_KIND_NAME[k]}")
    return p


def _send_error(tr: Transport, msg: str) -> None:
    try:
        write_frame(tr, KIND["ERROR"], msg.encode("utf-8")[:4096])
    except (BridgeError, OSError):
        pass


# ================================================================================================ handshake
@dataclass(frozen=True)
class Hello:
    """The HELLO frame's content."""
    layout_fingerprint: str
    spec_fingerprint: str
    profile: str
    n_batch: int
    n_rings: int
    n_lines: int
    n_temp: int
    samples_per_step: int
    period_ticks: int
    obs_record_bytes: int
    act_record_bytes: int
    obs_payload_bytes: int
    act_payload_bytes: int
    latency_ticks: int
    tick_s: float
    layout: dict

    def pack(self) -> bytes:
        js = json.dumps(self.layout, sort_keys=True).encode()
        return _HELLO.pack(self.layout_fingerprint.encode(), self.spec_fingerprint.encode(),
                           self.profile.encode()[:16], self.n_batch, self.n_rings, self.n_lines, self.n_temp,
                           self.samples_per_step, self.period_ticks, self.obs_record_bytes, self.act_record_bytes,
                           self.obs_payload_bytes, self.act_payload_bytes, self.latency_ticks, self.tick_s,
                           len(js)) + js

    @classmethod
    def unpack(cls, payload: bytes) -> "Hello":
        if len(payload) < _HELLO.size:
            raise BridgeError("HELLO frame too short")
        v = _HELLO.unpack_from(payload, 0)
        js = payload[_HELLO.size:_HELLO.size + v[-1]]
        if len(js) != v[-1]:
            raise BridgeError("HELLO frame truncated")

        def s(b):
            return b.rstrip(b"\0").decode()
        return cls(s(v[0]), s(v[1]), s(v[2]), *v[3:13], v[13], v[14], json.loads(js))


def _hello_for(spec: ControllerSpec, layout: FrameLayout, latency_ticks: int) -> Hello:
    return Hello(layout.fingerprint(), spec.fingerprint(), layout.profile, spec.n_batch, spec.n_rings, spec.n_lines,
                 len(spec.temp_sensors), spec.samples_per_step, spec.period_ticks, layout.obs_record_bytes,
                 layout.act_record_bytes, layout.obs_payload_bytes, layout.act_payload_bytes, int(latency_ticks),
                 float(spec.tick_s), layout.describe())


# ================================================================================================ plant side
class BridgeController:
    """The plant side: a Controller-protocol object whose ``step`` sends the observation frame and returns the
    external controller's actions (module docstring). Use it as ``controller=`` of ``mrm.sim.runner.run`` (Python
    path). ``latency_ticks`` must equal the external controller's (checked at the handshake).

    ``stats``: steps, round trips and their wall time (``rt_s``: from the OBS write to the complete ACT read,
    i.e. transport + the external side's decode, step and encode), encode / decode time on this side.
    ``keep_times=True`` also keeps every round trip's wall time in ``rt_samples`` (s), for medians.
    """
    truth_fed = False

    def __init__(self, transport: Transport, *, latency_ticks: int = 0, profile: str = "full",
                 sensors: Sequence[str] | None = None, actuators: Sequence[str] | None = None,
                 name: str = "bridge", keep_times: bool = False):
        self.transport = transport
        self.latency_ticks = int(latency_ticks)
        self.profile = profile
        self._sensors = None if sensors is None else list(sensors)
        self._actuators = None if actuators is None else list(actuators)
        self.name = name
        self.layout: FrameLayout | None = None
        self.spec: ControllerSpec | None = None
        self.remote_name: str | None = None
        self.stats = {"steps": 0, "resets": 0, "rt_s": 0.0, "rt_max_s": 0.0, "encode_s": 0.0, "decode_s": 0.0}
        self.rt_samples: list[float] | None = [] if keep_times else None
        self._closed = False

    def reset(self, spec: ControllerSpec) -> None:
        if self._closed:
            raise BridgeError("bridge already closed")
        lay = FrameLayout.from_spec(spec, self.profile, sensors=self._sensors, actuators=self._actuators)
        hello = _hello_for(spec, lay, self.latency_ticks)
        write_frame(self.transport, KIND["HELLO"], hello.pack())
        p = _expect(self.transport, "READY")
        if len(p) < _READY.size:
            raise BridgeError("READY frame too short")
        fp, lat, _flags, nm = _READY.unpack_from(p, 0)
        fp = fp.rstrip(b"\0").decode()
        if fp != lay.fingerprint():
            raise BridgeError(f"external controller built frame layout {fp}, the plant {lay.fingerprint()}")
        if int(lat) != self.latency_ticks:
            raise BridgeError(f"external controller latency_ticks = {lat}, BridgeController was built with "
                              f"{self.latency_ticks} (the plant's controller pipeline is sized from it)")
        self.remote_name = nm.rstrip(b"\0").decode()
        self.layout, self.spec = lay, spec
        self.stats["resets"] += 1

    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        lay = self.layout
        if lay is None:
            raise RuntimeError("BridgeController: call reset(spec) first")
        t0 = time.perf_counter()
        payload = lay.encode_obs(obs.tick, obs.sensors, obs.new)
        hdr = _HDR.pack(MAGIC, PROTOCOL_VERSION, KIND["OBS"], len(payload))
        t1 = time.perf_counter()
        self.transport.send(hdr + payload)
        p = _expect(self.transport, "ACT")
        t2 = time.perf_counter()
        tick, _mask, acts = lay.decode_act(p)
        if tick != obs.tick:
            raise BridgeError(f"lockstep lost: ACT for tick {tick}, OBS was tick {obs.tick}")
        t3 = time.perf_counter()
        st = self.stats
        st["steps"] += 1
        st["encode_s"] += t1 - t0
        st["rt_s"] += t2 - t1
        st["rt_max_s"] = max(st["rt_max_s"], t2 - t1)
        st["decode_s"] += t3 - t2
        if self.rt_samples is not None:
            self.rt_samples.append(t2 - t1)
        return acts

    @property
    def round_trip_us(self) -> float:
        """Mean wall time of one round trip (OBS write to ACT read), us."""
        return 1e6 * self.stats["rt_s"] / max(1, self.stats["steps"])

    def close(self) -> None:
        """Send BYE and close the transport (waits for a spawned process)."""
        if self._closed:
            return
        self._closed = True
        try:
            write_frame(self.transport, KIND["BYE"])
        except (BridgeError, OSError):
            pass
        self.transport.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ================================================================================================ controller side
def _check_bridgeable(controller: Any) -> None:
    """Refuse a controller whose behaviour the edge frame cannot carry (raises BridgeError)."""
    from mrm.control.api import controller_fast_kernel
    if getattr(controller, "truth_fed", False):
        raise BridgeError(f"{getattr(controller, 'name', controller)!r} is Truth-fed (the oracle): Truth never "
                          f"crosses the bridge (07 P1-1)")
    if controller_fast_kernel(controller) is not None:
        raise BridgeError(f"{getattr(controller, 'name', controller)!r} has a fast_kernel: its per-sample inputs and "
                          f"offsets are not in the edge frame, so a bridged run would silently drop it (07 P0-7); "
                          f"run it in-process")


def _warm_start_registers() -> frozenset:
    """Actuators whose plant register starts at a warm-start code instead of the channel init (mrm.plant.plant
    reset: the heater and bias DAC registers); the other controller-bound actuators start at ``ChannelSpec.init``."""
    from mrm.plant.io import DST_BIAS, DST_HEATER, DST_NAMES
    return frozenset(n for n, d in DST_NAMES.items() if d in (DST_HEATER, DST_BIAS))


def serve(controller: Any, spec: ControllerSpec, transport: Transport, *, max_steps: int | None = None) -> dict:
    """The external side: answer HELLO with READY (after ``controller.reset(spec)``; every HELLO resets), every OBS
    with one ACT, until BYE or the end of the stream. Returns stats {steps, resets, step_s, saturation_counts}.

    The layout is rebuilt from ``spec`` with the channel lists of the HELLO and must have the same fingerprint; the
    HELLO's spec fingerprint must equal ``spec.fingerprint()`` (both sides built the same interface). When the frame
    carries fewer sensors than ``spec`` exposes, the controller is reset with :func:`restrict_spec` (only the
    framed sensors exposed, channel ids kept). Actions pass the runner's ActionValidator (clip | error) before they
    are framed, and a CompiledController's held command is clipped as the runner and the kernel clip it
    (:func:`mrm.sim.vectors.clip_held_command`), so its next step sees what it would see in-process.

    A channel the controller writes that is not in the frame never reaches the plant, so it must equal the plant's
    register at every step: its latched readback ``<channel>.applied`` when that is framed, else its
    ``ChannelSpec.init`` (the plant's start value of every register except the heater and bias DACs, whose warm-start
    codes this side cannot know: writing one of them unframed is refused). Anything else raises BridgeError.
    Controllers with a ``fast_kernel`` and Truth-fed controllers are refused at the handshake.
    """
    from mrm.control.api import CompiledController
    from mrm.sim.vectors import clip_held_command, hidden_warm_start_registers
    stats = {"steps": 0, "resets": 0, "step_s": 0.0, "saturation_counts": {}}
    lay: FrameLayout | None = None
    cspec = spec
    validator: ActionValidator | None = None
    outside: dict[str, np.ndarray] = {}
    warm = _warm_start_registers()
    compiled = isinstance(controller, CompiledController)
    try:
        while True:
            fr = read_frame(transport, eof_ok=True)
            if fr is None:                                  # the plant closed the stream at a frame boundary
                break
            kind, p = fr
            if kind == KIND["BYE"]:
                break
            if kind == KIND["HELLO"]:
                h = Hello.unpack(p)
                if h.spec_fingerprint != spec.fingerprint():
                    raise BridgeError(f"plant ControllerSpec {h.spec_fingerprint} != this side's "
                                      f"{spec.fingerprint()}: build both from the same config, scenario and batch")
                lay = FrameLayout.from_spec(spec, sensors=h.layout["sensors"], actuators=h.layout["actuators"]) \
                    if h.profile == "custom" else FrameLayout.from_spec(spec, h.profile)
                if lay.fingerprint() != h.layout_fingerprint:
                    raise BridgeError(f"frame layout {lay.fingerprint()} != the plant's {h.layout_fingerprint}")
                _check_bridgeable(controller)
                lat = int(getattr(controller, "latency_ticks", 0) or 0)
                cspec = restrict_spec(spec, lay.sensors)
                if compiled and hidden_warm_start_registers(cspec):
                    warnings.warn(f"bridge serve: {hidden_warm_start_registers(cspec)} have no exposed '.applied' "
                                  f"readback: the external CompiledController starts its held command from the "
                                  f"channel init, not from the plant's warm-start registers as in-process",
                                  stacklevel=2)
                controller.reset(cspec)
                validator = ActionValidator(cspec)
                outside = {}
                nm = str(getattr(controller, "name", type(controller).__name__)).encode()[:32]
                write_frame(transport, KIND["READY"], _READY.pack(lay.fingerprint().encode(), lat, 0, nm))
                stats["resets"] += 1
                continue
            if kind != KIND["OBS"] or lay is None:
                raise BridgeError(f"unexpected {_KIND_NAME[kind]} frame" + ("" if lay else " before HELLO"))
            tick, sensors, new = lay.decode_obs(p)
            t0 = time.perf_counter()
            acts = validator.validate(controller.step(Observation(tick, cspec.tick_s, sensors, new)))
            if compiled:
                clip_held_command(controller, cspec)
            stats["step_s"] += time.perf_counter() - t0
            for name, v in acts.items():
                if name in lay.actuators:
                    continue
                if name not in outside:
                    v64 = np.asarray(v, dtype=np.int64)
                    rb = sensors.get(name + ".applied")
                    if rb is not None:
                        ref, what = np.asarray(rb, dtype=np.int64).reshape(v64.shape), "its framed latched register"
                    elif name in warm:
                        raise BridgeError(f"the controller wrote {name}, which is not in the frame (profile "
                                          f"{lay.profile!r}), and the plant's warm-start register of it is unknown "
                                          f"on this side; add {name} to actuators=")
                    else:
                        ref, what = np.full(v64.shape, int(spec.channels[name].init), np.int64), "its init value"
                    if not np.array_equal(v64, ref):
                        raise BridgeError(f"the controller wrote {name} != {what} (the plant's register), but "
                                          f"{name} is not in the frame (profile {lay.profile!r}): the plant would "
                                          f"never see it; use profile='full' or add it to actuators=")
                    outside[name] = np.array(v, copy=True)
                elif not np.array_equal(outside[name], v):
                    raise BridgeError(f"the controller changed {name}, which is not in the frame (profile "
                                      f"{lay.profile!r}); use profile='full' or add it to actuators=")
            write_frame(transport, KIND["ACT"], lay.encode_act(tick, acts))
            stats["steps"] += 1
            if max_steps is not None and stats["steps"] >= max_steps:
                break
    except BridgeError as e:
        _send_error(transport, str(e))
        raise
    except Exception as e:                                  # report a controller crash to the plant, then re-raise
        _send_error(transport, f"{type(e).__name__}: {e}")
        raise
    if validator is not None:
        stats["saturation_counts"] = {k: v.tolist() for k, v in validator.saturation_counts().items()}
    return stats


def restrict_spec(spec: ControllerSpec, sensors: Sequence[str]) -> ControllerSpec:
    """``spec`` with only ``sensors`` exposed (the actuators unchanged; channel ids kept, as
    ``controller_interface.expose`` does). The external side's controller is reset with it when the frame carries
    fewer sensors than the spec exposes, so a CompiledController packs only what it receives."""
    import dataclasses
    from mrm.core.channels import make_abi_layout
    keep = set(sensors)
    chans = {n: c for n, c in spec.channels.items() if c.kind == "actuator" or n in keep}
    if len(chans) == len(spec.channels):
        return spec
    lay = make_abi_layout(chans, spec.dims, spec.bindings, obs_ids={f.name: f.ch_id for f in spec.layout.obs},
                          act_ids={f.name: f.ch_id for f in spec.layout.act})
    return dataclasses.replace(spec, channels=chans, layout=lay)


def spec_for(cfg, scenario=None, batch: int = 1) -> ControllerSpec:
    """The ControllerSpec exactly as ``mrm.sim.runner.run`` builds it: from the UNSAMPLED config (``nominal=cfg``),
    with the scenario's actuator drives and offsets (07 P0-7, P1-1). ``scenario``: None (the config's), a registry
    name, or a Scenario."""
    from mrm import params
    from mrm.scenarios import registry as scen_registry
    from mrm.scenarios.generators import Scenario
    c = params.load(cfg) if isinstance(cfg, str) else cfg
    if scenario is None:
        sc = scen_registry.from_config(c)
    elif isinstance(scenario, str):
        sc = scen_registry.create(scenario)
    elif isinstance(scenario, Scenario):
        sc = scenario
    else:
        raise TypeError(f"scenario must be None, a name or a Scenario, got {type(scenario).__name__}")
    return ControllerSpec.from_config(c, n_batch=int(batch), nominal=c, scenario_drives=sc.drives,
                                      scenario_offsets=sc.offsets)


# ================================================================================================ C side
def _cname(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).upper()


def c_header(layout: FrameLayout, *, guard: str = "MRM_BRIDGE_H", spec_fingerprint: str | None = None) -> str:
    """``mrm_bridge.h`` for ``layout``: protocol constants, record sizes, every field's byte offset, element count,
    element bytes, signedness and channel range, every flag bit, the write-mask bits, and little-endian accessors.
    ``spec_fingerprint`` (``ControllerSpec.fingerprint()``, 16 hex digits; ``mrm bridge c-header`` passes it) adds
    ``MRMB_SPEC_FP``, which C_FXP_PI_SOURCE compares with the HELLO's spec fingerprint (bytes 16..31): the C side
    then refuses a plant built from another interface (timing, knowledge, ranges), not only another frame."""
    L = [f"/* generated by mrm.sim.bridge: protocol {PROTOCOL_VERSION}, layout {layout.fingerprint()} "
         f"(profile {layout.profile}). Do not edit; regenerate with `mrm bridge c-header`. */",
         f"#ifndef {guard}", f"#define {guard}", "#include <stdint.h>", "#include <string.h>", "",
         f"#define MRMB_VERSION {PROTOCOL_VERSION}", "#define MRMB_MAGIC0 0x4D /* 'M' */",
         "#define MRMB_MAGIC1 0x42 /* 'B' */", "#define MRMB_HEADER_BYTES 8"]
    L += [f"#define MRMB_KIND_{k} {v}" for k, v in KIND.items()]
    if spec_fingerprint is not None:
        sfp = str(spec_fingerprint)
        if len(sfp.encode()) > 16 or not sfp.isalnum():
            raise BridgeError(f"spec fingerprint {sfp!r}: expected at most 16 hex digits")
        L.append(f'#define MRMB_SPEC_FP "{sfp}"')
    L += [f'#define MRMB_LAYOUT_FP "{layout.fingerprint()}"', f"#define MRMB_B {layout.n_batch}",
          f"#define MRMB_N {layout.n_rings}", f"#define MRMB_OBS_RECORD_BYTES {layout.obs_record_bytes}",
          f"#define MRMB_ACT_RECORD_BYTES {layout.act_record_bytes}",
          f"#define MRMB_OBS_PAYLOAD_BYTES {layout.obs_payload_bytes} /* u64 tick + B records */",
          f"#define MRMB_ACT_PAYLOAD_BYTES {layout.act_payload_bytes} /* u64 tick, u32 mask, u32 0, B records */",
          f"#define MRMB_HELLO_FIXED_BYTES {_HELLO.size}", f"#define MRMB_READY_BYTES {_READY.size}", "",
          "/* observation fields: byte offset in a world record, element count, bytes per element, signed */"]
    for f in layout.obs_fields:
        n = _cname(f.name)
        dt = np.dtype(f.dtype)
        L += [f"#define MRMB_OBS_{n}_OFF {f.offset}", f"#define MRMB_OBS_{n}_COUNT {f.count}",
              f"#define MRMB_OBS_{n}_BYTES {dt.itemsize}", f"#define MRMB_OBS_{n}_SIGNED {int(dt.kind == 'i')}"]
        if f.name in layout.channels:
            ch = layout.channels[f.name]
            L += [f"#define MRMB_OBS_{n}_MIN INT64_C({int(ch.min)})", f"#define MRMB_OBS_{n}_MAX INT64_C({int(ch.max)})"]
    L += ["", "/* flag bits (group FLAGS: one word per ring; DFLAGS: one word per world) */"]
    for b in layout.flag_bits:
        L.append(f"#define MRMB_{b.group.upper()}_{_cname(b.channel)}_{b.what.upper()}_BIT {b.bit}")
    L += ["", "/* action fields; MRMB_ACT_<X>_MASK_BIT is its bit in write_mask */"]
    for j, f in enumerate(layout.act_fields):
        n = _cname(f.name)
        dt = np.dtype(f.dtype)
        L += [f"#define MRMB_ACT_{n}_OFF {f.offset}", f"#define MRMB_ACT_{n}_COUNT {f.count}",
              f"#define MRMB_ACT_{n}_BYTES {dt.itemsize}", f"#define MRMB_ACT_{n}_SIGNED {int(dt.kind == 'i')}",
              f"#define MRMB_ACT_{n}_MIN INT64_C({int(layout.channels[f.name].min)})",
              f"#define MRMB_ACT_{n}_MAX INT64_C({int(layout.channels[f.name].max)})",
              f"#define MRMB_ACT_{n}_MASK_BIT {j}"]
    L += ["",
          "static inline uint64_t mrmb_get_u(const uint8_t *p, int nbytes) {",
          "    uint64_t v = 0; for (int i = nbytes - 1; i >= 0; --i) v = (v << 8) | p[i]; return v; }",
          "static inline int64_t mrmb_get(const uint8_t *p, int nbytes, int is_signed) {",
          "    uint64_t v = mrmb_get_u(p, nbytes);",
          "    if (is_signed && nbytes < 8 && (v >> (8 * nbytes - 1)) & 1u) v |= ~((UINT64_C(1) << (8 * nbytes)) - 1u);",
          "    return (int64_t)v; }",
          "static inline void mrmb_put(uint8_t *p, int nbytes, int64_t x) {",
          "    uint64_t v = (uint64_t)x; for (int i = 0; i < nbytes; ++i) { p[i] = (uint8_t)(v & 0xFFu); v >>= 8; } }",
          "/* element i of field X of a world record: mrmb_get(rec + MRMB_OBS_X_OFF + i * MRMB_OBS_X_BYTES, "
          "MRMB_OBS_X_BYTES, MRMB_OBS_X_SIGNED) */",
          f"#endif /* {guard} */", ""]
    return "\n".join(L)


#: A complete external controller in C99: the twin of mrm.control.fxp.FxpPi (fx_pi_update, capture mode or one
#: set-point), speaking the bridge protocol on stdin / stdout. Build: ``cc -O2 -std=c99 -I<dir of mrm_bridge.h>``.
#: Arguments: kp_raw ki_raw frac_bits acc_bits rounding [setpoint]; rounding as mrm.control.fxp.Rounding (0 floor,
#: 1 half_up, 2 half_even, 3 toward_zero, 4 half_away). Needs tx.monitor.latest (+ its new bit), tx.heater.applied
#: and tx.heater in the frame (profiles minimal and full have them); monitor_valid is used when present.
C_FXP_PI_SOURCE = r"""/* Fixed-point PI external controller: C twin of mrm.control.fxp.FxpPi (07 P1-21). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include "mrm_bridge.h"

#ifndef MRMB_FLAGS_MONITOR_VALID_VALUE_BIT
#define MRMB_FLAGS_MONITOR_VALID_VALUE_BIT (-1)
#endif

static int read_exact(void *buf, size_t n) {
    uint8_t *p = (uint8_t *)buf; size_t got = 0;
    while (got < n) { ssize_t k = read(0, p + got, n - got); if (k <= 0) return -1; got += (size_t)k; }
    return 0;
}
static int write_all(const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t *)buf; size_t put = 0;
    while (put < n) { ssize_t k = write(1, p + put, n - put); if (k <= 0) return -1; put += (size_t)k; }
    return 0;
}
static int send_frame(int kind, const uint8_t *payload, uint32_t n) {
    uint8_t h[8] = {MRMB_MAGIC0, MRMB_MAGIC1, MRMB_VERSION, (uint8_t)kind, 0, 0, 0, 0};
    mrmb_put(h + 4, 4, (int64_t)n);
    return write_all(h, 8) || (n && write_all(payload, n));
}
static void fail(const char *msg) {
    send_frame(MRMB_KIND_ERROR, (const uint8_t *)msg, (uint32_t)strlen(msg));
    fprintf(stderr, "fxp_pi_c: %s\n", msg); exit(2);
}
/* arithmetic shift right without implementation-defined behaviour */
static int64_t asr(int64_t x, int s) { return x >= 0 ? (x >> s) : ~((~x) >> s); }
static int64_t sat_bits(int64_t x, int bits) {
    int64_t hi = (int64_t)((UINT64_C(1) << (bits - 1)) - 1), lo = -hi - 1;
    return x < lo ? lo : (x > hi ? hi : x);
}
/* mrm.control.fxp.fx_round_shift */
static int64_t round_shift(int64_t x, int s, int rnd) {
    if (s == 0) return x;
    int64_t half = (int64_t)1 << (s - 1), q, r;
    switch (rnd) {
    case 0: return asr(x, s);
    case 1: return asr(x + half, s);
    case 4: return x >= 0 ? asr(x + half, s) : -asr(half - x, s);
    default:
        q = asr(x, s); r = x - q * ((int64_t)1 << s);
        if (rnd == 2) { if (r > half || (r == half && (q & 1))) q += 1; return q; }
        if (rnd == 3) { if (x < 0 && r != 0) q += 1; return q; }
        fail("unknown rounding mode"); return 0;
    }
}
/* mrm.control.fxp.fx_pi_update */
static int64_t pi_update(int64_t e, int64_t *integ, int64_t kp, int64_t ki, int64_t u0, int frac, int acc_bits,
                         int64_t lo, int64_t hi, int rnd) {
    int64_t a = sat_bits(*integ + e * ki, acc_bits);
    int64_t ilo = (lo - u0) * ((int64_t)1 << frac), ihi = (hi - u0) * ((int64_t)1 << frac);
    if (a < ilo) a = ilo; else if (a > ihi) a = ihi;
    int64_t v = u0 * ((int64_t)1 << frac) + e * kp + a;
    int64_t u = round_shift(v, frac, rnd);
    *integ = a;
    return u < lo ? lo : (u > hi ? hi : u);
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: fxp_pi_c kp_raw ki_raw frac acc_bits rounding [setpoint]\n"); return 1; }
    int64_t kp = strtoll(argv[1], 0, 10), ki = strtoll(argv[2], 0, 10);
    int frac = atoi(argv[3]), acc_bits = atoi(argv[4]), rnd = atoi(argv[5]);
    int have_sp = argc > 6; int64_t sp_arg = have_sp ? strtoll(argv[6], 0, 10) : 0;
    const int64_t lo = MRMB_ACT_TX_HEATER_MIN, hi = MRMB_ACT_TX_HEATER_MAX;   /* the heater code range */
    static uint8_t obs[MRMB_OBS_PAYLOAD_BYTES], act[MRMB_ACT_PAYLOAD_BYTES], buf[65536];
    int64_t cap[MRMB_B][MRMB_N], yset[MRMB_B][MRMB_N], u0[MRMB_B][MRMB_N], integ[MRMB_B][MRMB_N],
            u[MRMB_B][MRMB_N];
    int started = 0;
    for (;;) {
        uint8_t h[8];
        if (read_exact(h, 8)) return 0;                          /* plant closed the stream */
        if (h[0] != MRMB_MAGIC0 || h[1] != MRMB_MAGIC1) fail("bad magic");
        if (h[2] != MRMB_VERSION) fail("protocol version mismatch");
        uint32_t n = (uint32_t)mrmb_get_u(h + 4, 4);
        int kind = h[3];
        if (kind == MRMB_KIND_BYE) return 0;
        if (kind == MRMB_KIND_HELLO) {
            if (n > sizeof buf || read_exact(buf, n)) fail("bad HELLO");
            if (memcmp(buf, MRMB_LAYOUT_FP, 16) != 0) fail("frame layout fingerprint differs from mrm_bridge.h");
#ifdef MRMB_SPEC_FP
            if (memcmp(buf + 16, MRMB_SPEC_FP, strlen(MRMB_SPEC_FP)) != 0)
                fail("ControllerSpec fingerprint differs from mrm_bridge.h (regenerate it for this config)");
#endif
            if (mrmb_get_u(buf + 48, 4) != MRMB_B || mrmb_get_u(buf + 52, 4) != MRMB_N) fail("B or N differs");
            if ((int32_t)mrmb_get_u(buf + 88, 4) != 0) fail("this controller has latency_ticks = 0");
            uint8_t ready[MRMB_READY_BYTES];
            memset(ready, 0, sizeof ready);
            memcpy(ready, MRMB_LAYOUT_FP, 16);
            memcpy(ready + 24, "fxp_pi_c", 8);
            if (send_frame(MRMB_KIND_READY, ready, sizeof ready)) return 3;
            started = 0;
            continue;
        }
        if (kind != MRMB_KIND_OBS || n != MRMB_OBS_PAYLOAD_BYTES) fail("expected an OBS frame");
        if (read_exact(obs, n)) return 3;
        memset(act, 0, sizeof act);
        memcpy(act, obs, 8);                                     /* echo the tick */
        mrmb_put(act + 8, 4, (int64_t)1 << MRMB_ACT_TX_HEATER_MASK_BIT);
        for (int b = 0; b < MRMB_B; ++b) {
            const uint8_t *rec = obs + 8 + (size_t)b * MRMB_OBS_RECORD_BYTES;
            uint8_t *arec = act + 16 + (size_t)b * MRMB_ACT_RECORD_BYTES;
            for (int i = 0; i < MRMB_N; ++i) {
                if (!started) {
                    u0[b][i] = mrmb_get(rec + MRMB_OBS_TX_HEATER_APPLIED_OFF + i * MRMB_OBS_TX_HEATER_APPLIED_BYTES,
                                        MRMB_OBS_TX_HEATER_APPLIED_BYTES, MRMB_OBS_TX_HEATER_APPLIED_SIGNED);
                    u[b][i] = u0[b][i]; integ[b][i] = 0; cap[b][i] = have_sp; yset[b][i] = sp_arg;
                }
                uint64_t fl = mrmb_get_u(rec + MRMB_OBS_FLAGS_OFF + i * MRMB_OBS_FLAGS_BYTES, MRMB_OBS_FLAGS_BYTES);
                int fresh = (int)((fl >> MRMB_FLAGS_TX_MONITOR_LATEST_NEW_BIT) & 1u);
                int valid = MRMB_FLAGS_MONITOR_VALID_VALUE_BIT < 0 ? 1
                            : (int)((fl >> (MRMB_FLAGS_MONITOR_VALID_VALUE_BIT < 0 ? 0 : MRMB_FLAGS_MONITOR_VALID_VALUE_BIT)) & 1u);
                if (fresh && valid) {
                    int64_t y = mrmb_get(rec + MRMB_OBS_TX_MONITOR_LATEST_OFF + i * MRMB_OBS_TX_MONITOR_LATEST_BYTES,
                                         MRMB_OBS_TX_MONITOR_LATEST_BYTES, MRMB_OBS_TX_MONITOR_LATEST_SIGNED);
                    if (!cap[b][i]) { yset[b][i] = y; cap[b][i] = 1; }
                    else u[b][i] = pi_update(y - yset[b][i], &integ[b][i], kp, ki, u0[b][i], frac, acc_bits, lo, hi,
                                             rnd);
                }
                mrmb_put(arec + MRMB_ACT_TX_HEATER_OFF + i * MRMB_ACT_TX_HEATER_BYTES, MRMB_ACT_TX_HEATER_BYTES,
                         u[b][i]);
            }
        }
        started = 1;
        if (send_frame(MRMB_KIND_ACT, act, MRMB_ACT_PAYLOAD_BYTES)) return 3;
    }
}
"""


# ================================================================================================ command line
def _load_cfg(preset: str, overrides: Sequence[str]):
    from mrm import params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return params.load(preset, overrides=list(overrides) or None)


def _scenario(name: str | None, params_json: str | None):
    if name is None:
        return None
    from mrm.scenarios import registry as scen_registry
    return scen_registry.create(name, **(json.loads(params_json) if params_json else {}))


def add_arguments(ap: argparse.ArgumentParser) -> None:
    """Arguments of ``mrm bridge`` / ``python -m mrm.sim.bridge``."""
    sub = ap.add_subparsers(dest="bridge_cmd", required=True)
    s = sub.add_parser("serve", help="serve a controller over the bridge (stdin/stdout, or --connect host:port)")
    s.add_argument("--preset", default="g26", help="preset name or YAML path (as given to the plant side)")
    s.add_argument("--override", action="append", default=[], help="config override key=value (repeatable)")
    s.add_argument("--scenario", default=None, help="scenario registry name (default: the config's)")
    s.add_argument("--scenario-params", default=None, help="JSON keyword arguments of the scenario builder")
    s.add_argument("--batch", type=int, default=1)
    s.add_argument("--controller", required=True, help="registry name or package.module:Class")
    s.add_argument("--params", default=None, help="JSON controller parameters")
    s.add_argument("--connect", default=None, help="host:port of a plant waiting with bridge.listen()")
    for nm, what in (("c-header", "write mrm_bridge.h for a config"), ("c-example", "write the C FxpPi twin")):
        c = sub.add_parser(nm, help=what)
        c.add_argument("--preset", default="g26")
        c.add_argument("--override", action="append", default=[])
        c.add_argument("--scenario", default=None)
        c.add_argument("--scenario-params", default=None)
        c.add_argument("--batch", type=int, default=1)
        c.add_argument("--profile", default="full", choices=PROFILES)
        c.add_argument("--out", default="-", help="output file (default stdout)")


def run_command(args: argparse.Namespace) -> int:
    """Execute a parsed ``mrm bridge ...`` command."""
    cfg = _load_cfg(args.preset, args.override)
    sc = _scenario(args.scenario, args.scenario_params)
    spec = spec_for(cfg, sc, args.batch)
    if args.bridge_cmd == "serve":
        from mrm.control import registry
        ctrl = registry.create(args.controller, json.loads(args.params) if args.params else {})
        if args.connect:
            host, _, port = args.connect.rpartition(":")
            tr: Transport = connect(host or "127.0.0.1", int(port))
        else:
            tr = stdio()
        st = serve(ctrl, spec, tr)
        tr.close()
        print(json.dumps({"bridge_serve": {k: v for k, v in st.items() if k != "saturation_counts"}}),
              file=sys.stderr)
        return 0
    lay = FrameLayout.from_spec(spec, args.profile)
    text = c_header(lay, spec_fingerprint=spec.fingerprint()) if args.bridge_cmd == "c-header" else C_FXP_PI_SOURCE
    if args.out == "-":
        sys.stdout.write(text)
    else:
        with open(args.out, "w") as f:
            f.write(text)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m mrm.sim.bridge serve | c-header | c-example``."""
    ap = argparse.ArgumentParser(prog="python -m mrm.sim.bridge", description=__doc__.split("\n")[0])
    add_arguments(ap)
    return run_command(ap.parse_args(argv))


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
