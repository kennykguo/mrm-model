"""Named I/O channels, the v1 channel list, port binding and the flat int64 ABI layout (07 P0-7, D4, P0-9).

Every signal between the plant and a controller is a named channel described by a :class:`ChannelSpec`. Python
controllers exchange ``dict[str, ndarray]``; compiled controllers exchange one flat int64 vector per batch element
whose layout (:class:`AbiLayout`) is generated from the same specs (07 D4: "named channels in Python; for compiled
controllers, realize() generates a flat int64 layout from the ChannelSpecs"). New actuators (laser tuning,
interleaver or polarization-controller phases) then add channels and break nothing.

Shapes are symbolic: ``B`` batch, ``N`` rings, ``L`` laser lines, ``S`` temperature sensors, ``M`` the maximum number
of ADC samples delivered per controller step (``ceil(controller period / ADC period)``). The first symbol is always
``B``. Codes are integers end to end ("codes in and codes out", 07 Summary).

The v1 channel list (:data:`V1_CHANNEL_TABLE`) is data. :func:`build_v1_channels` fills its hardware-dependent
fields (bits, ranges, rates, latencies) from a :class:`ChannelHardware` record, which the params layer builds from
the config (:meth:`ChannelHardware.from_config`). The defaults of :class:`ChannelHardware` are fallbacks for tests and
carry their provenance; presets override them.

Binding (07 P0-7): every actuator channel is owned by the controller or by the scenario, never both. A scenario may
still add an offset (ripple, droop) to any actuator. :func:`resolve_bindings` enforces this and names the channel.
"""
from __future__ import annotations

import enum
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "SYMBOLS", "ChannelSpec", "ChannelError", "BindingError", "ChannelHardware", "TempSensorHw",
    "V1_CHANNEL_TABLE", "DEFAULT_BINDINGS", "build_v1_channels", "resolve_bindings", "resolve_shape",
    "samples_per_step", "ObsId", "ActId", "TEMP_ID_BASE", "Field", "AbiLayout", "make_abi_layout",
    "OBS_HEADER", "OBS_TICK", "OBS_ACTIVE_RING", "ABI_VERSION", "INT64_MAX", "OBS_ID_BY_NAME",
    "ACT_ID_BY_NAME", "HARDWARE_KEYS", "assign_channel_ids",
]

SYMBOLS = ("B", "N", "L", "S", "M")
KINDS = ("actuator", "sensor")
AGGS = ("block", "count", "latest", "sum", "all", "any")
OWNERS = ("controller", "scenario")
INT64_MAX = 2**63 - 1
INT32_MAX = 2**31 - 1
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)*$")


class ChannelError(ValueError):
    """An invalid channel definition, an unknown channel name or a shape/dtype mismatch."""


class BindingError(ChannelError):
    """An actuator channel bound to both the controller and the scenario, or an invalid binding entry."""


@dataclass(frozen=True)
class ChannelSpec:
    """One named signal (07 P0-7: {name, shape, dtype, bits, min, max, unit, rate_hz, latency_s}).

    Attributes
    ----------
    name: dotted lower-case name, e.g. ``tx.heater``.
    kind: ``actuator`` (controller or scenario -> plant) or ``sensor`` (plant -> controller).
    shape: symbolic shape, first symbol ``B`` (e.g. ``('B', 'N')``, ``('B', 'N', 'M')``, ``('B', 1)``).
    dtype: NumPy dtype name of the values (``uint16``, ``int32``, ``int64``, ``bool``).
    bits: converter width in bits, or None when the channel is not a converter code.
    min, max: inclusive integer range of valid values (codes, counts, indices, ticks; bool is 0..1).
    unit: ``code``, ``count``, ``index``, ``tick`` or ``flag``.
    rate_hz: sensor sample rate, or actuator maximum update rate; None = once per controller step.
    latency_s: converter latency owned by the plant's io layer (nominal, so the controller may know it).
    agg: how a slower consumer combines several steps (CompositeController): ``block`` (concatenate samples, with
        ``count_channel``), ``count`` (sum, capped at the consumer's M), ``latest``, ``sum``, ``all``, ``any``.
        Contract with the plant: a ``latest`` sensor HOLDS its last delivered value between deliveries (its
        new-sample flag marks a delivery), and a ``sum`` sensor reports only what happened since the last step.
    count_channel: for ``block`` channels, the channel that holds the number of valid entries along M.
    init: value before the first write (initial actuator command / initial sensor value).
    scale, offset, phys_unit: nominal design conversion ``physical = offset + scale * code`` (None if not a
        documented linear converter, e.g. the heater DAC, whose power law is an uncertainty, 07 P1-11).
    optional: True if the channel exists only when its hardware is present.
    doc: one-line description with its source.
    """
    name: str
    kind: str
    shape: tuple
    dtype: str
    bits: int | None
    min: int
    max: int
    unit: str
    rate_hz: float | None = None
    latency_s: float = 0.0
    agg: str = "latest"
    count_channel: str | None = None
    init: int = 0
    scale: float | None = None
    offset: float | None = None
    phys_unit: str | None = None
    optional: bool = False
    doc: str = ""

    def __post_init__(self):
        if not _NAME_RE.match(self.name):
            raise ChannelError(f"channel name {self.name!r} must be dotted lower-case identifiers")
        if self.kind not in KINDS:
            raise ChannelError(f"{self.name}: kind {self.kind!r} not in {KINDS}")
        shape = tuple(self.shape)
        object.__setattr__(self, "shape", shape)
        if not shape or shape[0] != "B":
            raise ChannelError(f"{self.name}: shape {shape} must start with 'B'")
        for s in shape:
            if not ((isinstance(s, str) and s in SYMBOLS) or (isinstance(s, int) and not isinstance(s, bool) and s >= 1)):
                raise ChannelError(f"{self.name}: shape entry {s!r} must be one of {SYMBOLS} or a positive int")
        if shape.count("B") != 1 or shape.count("M") > 1:
            raise ChannelError(f"{self.name}: shape {shape} has a repeated B or M")
        try:
            dt = np.dtype(self.dtype)
        except TypeError:
            raise ChannelError(f"{self.name}: unknown dtype {self.dtype!r}") from None
        if dt.kind not in "biu":
            raise ChannelError(f"{self.name}: dtype {self.dtype} must be integer or bool (codes in, codes out)")
        if self.agg not in AGGS:
            raise ChannelError(f"{self.name}: agg {self.agg!r} not in {AGGS}")
        if (self.agg == "block") != (self.count_channel is not None):
            raise ChannelError(f"{self.name}: a 'block' channel needs count_channel, and only it")
        if self.agg == "block" and "M" not in shape:
            raise ChannelError(f"{self.name}: a 'block' channel needs an 'M' axis")
        if int(self.min) > int(self.max):
            raise ChannelError(f"{self.name}: min {self.min} > max {self.max}")
        lo, hi = (0, 1) if dt.kind == "b" else (int(np.iinfo(dt).min), int(np.iinfo(dt).max))
        if int(self.min) < lo or int(self.max) > hi:
            raise ChannelError(f"{self.name}: range [{self.min}, {self.max}] does not fit dtype {self.dtype}")
        if not int(self.min) <= int(self.init) <= int(self.max):
            raise ChannelError(f"{self.name}: init {self.init} outside [{self.min}, {self.max}]")
        if self.bits is not None and not 1 <= int(self.bits) <= 63:
            raise ChannelError(f"{self.name}: bits {self.bits} outside 1..63")
        if self.rate_hz is not None and not (math.isfinite(self.rate_hz) and self.rate_hz > 0):
            raise ChannelError(f"{self.name}: rate_hz {self.rate_hz!r} must be positive or None")
        if not (math.isfinite(self.latency_s) and self.latency_s >= 0):
            raise ChannelError(f"{self.name}: latency_s {self.latency_s!r} must be >= 0")

    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    def resolve_shape(self, dims: Mapping[str, int]) -> tuple[int, ...]:
        """Concrete shape for dims, e.g. {'B': 4, 'N': 8, 'M': 5}."""
        return resolve_shape(self.shape, dims, self.name)

    def element_shape(self, dims: Mapping[str, int]) -> tuple[int, ...]:
        """Shape of one batch element (the ABI shape): the resolved shape without B."""
        return self.resolve_shape(dims)[1:]

    def flag_shape(self, dims: Mapping[str, int]) -> tuple[int, ...]:
        """Shape of the new-sample flag of one batch element: without B and without M."""
        return tuple(d for s, d in zip(self.shape[1:], self.element_shape(dims)) if s != "M")


def resolve_shape(shape: Sequence, dims: Mapping[str, int], name: str = "?") -> tuple[int, ...]:
    """Replace each symbol of ``shape`` by ``dims[symbol]``; raise ChannelError naming a missing symbol."""
    out = []
    for s in shape:
        if isinstance(s, str):
            if s not in dims:
                raise ChannelError(f"{name}: dimension {s!r} not given (have {sorted(dims)})")
            v = int(dims[s])
            if v < 0 or (v == 0 and s != "S"):
                raise ChannelError(f"{name}: dimension {s} = {v} must be >= 1")
            out.append(v)
        else:
            out.append(int(s))
    return tuple(out)


def samples_per_step(controller_period_ticks: int, adc_period_ticks: int) -> int:
    """M = ceil(controller period / per-ring ADC period), at least 1: the most samples one step can receive.

    Between two edges P ticks apart, a sample grid of period A puts floor(P/A) or ceil(P/A) samples (constant ADC
    latency keeps them in order), so ceil is exact.
    """
    p, a = int(controller_period_ticks), int(adc_period_ticks)
    if p < 1 or a < 1:
        raise ChannelError(f"samples_per_step: periods must be >= 1 tick (got {p}, {a})")
    return max(1, -(-p // a))


# ---- hardware record ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TempSensorHw:
    """One on-die temperature sensor as seen through its converter (07 P1-10 ``temp_sensors`` entry).

    code = round((T_measured_c - offset_c) / lsb_k), clipped to 0 .. 2**bits - 1.
    """
    name: str = "die"
    bits: int = 12            # assumed: typical on-die sensor converter width (no Lightmatter data)
    lsb_k: float = 0.0625     # assumed: config key temp_sensors[].resolution_k (07 P1-10)
    offset_c: float = -40.0   # assumed: code 0 at -40 degC, so 12 bits x 0.0625 K span -40 .. 216 degC
    rate_hz: float = 1e3      # assumed: 07 P1-10 "on-die sensors are slow converters, with millisecond conversion"
    latency_s: float = 1e-3   # assumed: same source (millisecond conversion)
    site: str = "die"
    type: str = "diode"

    def __post_init__(self):
        if not re.match(r"^[a-z][a-z0-9_]*$", self.name):
            raise ChannelError(f"temperature sensor name {self.name!r} must be a lower-case identifier")


@dataclass(frozen=True)
class ChannelHardware:
    """Hardware facts that set the channel ranges, rates and latencies. Each field maps to one config key
    (:data:`HARDWARE_KEYS`); the defaults are fallbacks with the provenance noted, and presets override them."""
    heater_dac_bits: int = 16              # inferred: G26 codes reach ~49.3 k (LM26-S slide 10) -> >= 16 bit; PAT-2 "e.g., 16 bits" (01 table)
    heater_code_min: int = 0               # 07 P1-11: hardware range 0..65535, not REF.heaterDacCodeRange
    heater_code_max: int | None = None     # None -> 2**bits - 1 (07 P1-11)
    heater_update_rate_hz: float | None = None   # None: one update per controller step (07 P1-10 dac_max_update_hz)
    heater_latency_s: float = 0.0          # assumed (ring.heater.latency_s; unpublished)
    bias_dac_bits: int = 12                # assumed (07 P0-7: everything but the 1.0-3.75 V range is assumed)
    bias_v_min_v: float = 1.0              # printed: G26 relOMA bias axis (07 P0-7)
    bias_v_max_v: float = 3.75             # printed: G26 relOMA bias axis (07 P0-7)
    bias_update_rate_hz: float | None = None
    bias_latency_s: float = 0.0            # assumed (ring.bias_dac.latency_s)
    # adc_bits/rate/window are not jointly checked: a dual-slope ADC clocked at 5 ns with a 400-clock window
    # (PAT-2) resolves ~400 counts per conversion, not 2**16 (see docs/handoff/core_control_api.md).
    adc_bits: int = 16                     # inferred: from RX polarization-monitor codes; MRM monitor width unpublished (01 D17)
    # inferred: a dual-slope conversion is the 2 us integration (PAT-2 T_int example) plus up to 2 us of
    # de-integration (D <= N = 400 clocks), so one converter delivers at most 250 kHz
    adc_rate_hz: float = 250e3
    adc_latency_s: float = 2e-6            # assumed: de-integration takes up to one window (D <= N, PAT-2)
    adc_integration_window_s: float = 2e-6  # printed example: PAT-2 "2 us = 400 x 5 ns" (07 P0-7 default, tagged inferred)
    adc_full_scale_a: float | None = None  # nominal monitor full-scale current; None = not a documented linear scale
    afe_type: str = "integrate_dump"       # 07 P0-7 default (PAT-2 dual-slope integrating ADC)
    adc_channels_per_adc: int = 1          # assumed (adc_mux.channels_per_adc)
    pga_gains: tuple = (1.0,)              # assumed: one gain unless ring.monitor.pga_gains is set (07 P1-10)
    heater_readback_present: bool = False  # 07 P1-10 ring.heater.readback.present (schema default False)
    readback_adc_bits: int = 12            # assumed
    readback_v_full_scale_v: float = 2.0   # assumed: above the 1.5-1.6 V heater drive (REF.heaterVFsr range, textbook)
    readback_i_full_scale_a: float = 0.032  # assumed: above 1.55 V / 52.8 ohm = 29 mA (REF.heaterOhmRange, textbook)
    readback_rate_hz: float = 10e3         # assumed
    readback_latency_s: float = 1e-4       # assumed
    temp_sensors: tuple = (TempSensorHw(),)
    bitstat_window_bits: int = 64          # assumed: inside the 16-128 bits of Sun et al. (07 P0-11)
    error_count_period_s: float = 2.0      # printed: G26 BER sample period (07 P0-8, P0-11)
    error_count_latency_s: float = 1e-3    # assumed (datapath.error_count_latency_s)
    laser_tune: bool = False               # optional channel (07 P0-7)
    laser_tune_bits: int = 16              # assumed
    laser_tune_lsb_hz: float = 1e6         # assumed: 1 MHz per code

    @property
    def heater_max(self) -> int:
        return (1 << self.heater_dac_bits) - 1 if self.heater_code_max is None else int(self.heater_code_max)

    @classmethod
    def from_config(cls, cfg) -> "ChannelHardware":
        """Build from a config. ``cfg`` is an ``mrm.params.Config`` (numbers read with ``cfg.si(key)``, settings with
        ``cfg.get(key)``) or a plain mapping of dotted keys in config units (converted by key suffix with
        ``mrm.params.units.to_si``). Keys are those of
        :data:`HARDWARE_KEYS` (the names of mrm/params/schema.py); a missing key keeps the default above.
        ``temp_sensors`` is the 07 P1-10 table (name, site|attach, type, sample_rate_hz, latency_s, resolution_k),
        plus two optional fields: ``adc_bits`` and ``code_zero_c`` (the absolute degC that code 0 reads; schema name,
        docs/handoff/params_machinery.md section 3).
        """
        has_si = callable(getattr(cfg, "si", None))
        to_si = None
        if not has_si:
            try:                                   # plain mapping in config units: convert by key suffix
                from mrm.params.units import to_si
            except ImportError:                    # pragma: no cover - params layer absent
                to_si = None

        def get(key, numeric):
            if has_si and numeric:
                return cfg.si(key, None)
            v = cfg.get(key, None)
            v = v.get("value") if isinstance(v, Mapping) and "value" in v else v
            if v is not None and numeric and to_si is not None:
                v = to_si(key, v)
            return v

        kw: dict[str, object] = {}
        for fld, key in HARDWARE_KEYS.items():
            numeric = fld not in _NON_NUMERIC_FIELDS
            v = get(key, numeric)
            if v is None:
                continue
            if fld in _INT_FIELDS:
                v = int(v)
            elif fld in _BOOL_FIELDS:
                v = bool(v)
            elif fld == "pga_gains":
                v = tuple(float(g) for g in np.atleast_1d(v))
            elif fld == "afe_type":
                v = str(v)
            else:
                v = float(v)
            kw[fld] = v
        table = cfg.get("temp_sensors", None)
        if table:
            sensors = []
            for i, s in enumerate(table):
                s = {k: (v.get("value") if isinstance(v, Mapping) and "value" in v else v) for k, v in s.items()}
                d = TempSensorHw()

                def pick(k, default, s=s):
                    v = s.get(k)
                    return default if v is None else v
                sensors.append(TempSensorHw(
                    name=str(pick("name", f"t{i}")), bits=int(pick("adc_bits", d.bits)),
                    lsb_k=float(pick("resolution_k", d.lsb_k)), offset_c=float(pick("code_zero_c", d.offset_c)),
                    rate_hz=float(pick("sample_rate_hz", d.rate_hz)), latency_s=float(pick("latency_s", d.latency_s)),
                    site=str(pick("site", pick("attach", d.site))), type=str(pick("type", d.type))))
            kw["temp_sensors"] = tuple(sensors)
        return cls(**kw)


_INT_FIELDS = frozenset({"heater_dac_bits", "heater_code_min", "heater_code_max", "bias_dac_bits", "adc_bits",
                         "adc_channels_per_adc", "readback_adc_bits", "bitstat_window_bits", "laser_tune_bits"})
_BOOL_FIELDS = frozenset({"heater_readback_present", "laser_tune"})
_NON_NUMERIC_FIELDS = _INT_FIELDS | _BOOL_FIELDS | {"afe_type", "pga_gains"}


# field -> config key (the keys of mrm/params/schema.py; the laser.tune.* keys are proposed here, see
# docs/handoff/core_control_api.md).
HARDWARE_KEYS: dict[str, str] = {
    "heater_dac_bits": "ring.heater.dac_bits",
    "heater_code_min": "ring.heater.code_min",
    "heater_code_max": "ring.heater.code_max",
    "heater_update_rate_hz": "ring.heater.dac_max_update_hz",
    "heater_latency_s": "ring.heater.latency_s",
    "bias_dac_bits": "ring.bias_dac.bits",
    "bias_v_min_v": "ring.bias_dac.v_min_v",
    "bias_v_max_v": "ring.bias_dac.v_max_v",
    "bias_update_rate_hz": "ring.bias_dac.update_rate_hz",
    "bias_latency_s": "ring.bias_dac.latency_s",
    "adc_bits": "ring.monitor.adc_bits",
    "adc_rate_hz": "ring.monitor.adc_rate_hz",
    "adc_latency_s": "ring.monitor.adc_latency_s",
    "adc_integration_window_s": "ring.monitor.integration_window_s",
    "adc_full_scale_a": "ring.monitor.adc_full_scale_ua",     # read with cfg.si -> A
    "afe_type": "ring.monitor.afe_type",
    "adc_channels_per_adc": "ring.monitor.adc_mux.channels_per_adc",
    "pga_gains": "ring.monitor.pga_gains",
    "heater_readback_present": "ring.heater.readback.present",
    "readback_adc_bits": "ring.heater.readback.adc_bits",
    "readback_v_full_scale_v": "ring.heater.readback.v_full_scale_v",
    "readback_i_full_scale_a": "ring.heater.readback.i_full_scale_a",
    "readback_rate_hz": "ring.heater.readback.rate_hz",
    "readback_latency_s": "ring.heater.readback.latency_s",
    "bitstat_window_bits": "datapath.bitstat_window_bits",
    "error_count_period_s": "datapath.error_count_period_s",
    "error_count_latency_s": "datapath.error_count_latency_s",
    "laser_tune": "laser.tune.present",
    "laser_tune_bits": "laser.tune.bits",
    "laser_tune_lsb_hz": "laser.tune.lsb_hz",
}


# ---- the v1 channel list, as data -------------------------------------------------------------------------------
# Columns: name, kind, shape, dtype, bits, range, unit, rate, latency, agg, extra. dtype "u:bits" is the smallest
# unsigned type that holds the code (uint16 for <= 16 bits), "i:bits" is int32 (int64 above 31 bits).
# bits/rate/latency name a ChannelHardware attribute (or are literals / None). range is a rule resolved by
# build_v1_channels: 'heater' | 'unsigned' (0..2**bits-1) | 'signed' | 'line_index' (-1..L-1) | 'pga_index' |
# 'tick' | 'count' | 'nonneg' | 'bitstat' | 'flag'. 'present' names a ChannelHardware flag that must be true.
# "{t}" in a name expands to one channel per temperature sensor.

V1_CHANNEL_TABLE: tuple[dict, ...] = (
    # --- actuators (07 P0-7)
    dict(name="tx.heater", kind="actuator", shape=("B", "N"), dtype="u:bits", bits="heater_dac_bits",
         range="heater", unit="code", rate="heater_update_rate_hz", latency="heater_latency_s", agg="latest",
         doc="Heater DAC code; heaters only add heat (PAT-1); P(code) law is an uncertainty (07 P1-11)."),
    dict(name="tx.bias", kind="actuator", shape=("B", "N"), dtype="u:bits", bits="bias_dac_bits",
         range="unsigned", unit="code", rate="bias_update_rate_hz", latency="bias_latency_s", agg="latest",
         scale="bias", doc="Bias DAC code over ring.bias_dac.v_min_v..v_max_v (G26 axis 1.0-3.75 V; 07 P0-7, P1-7)."),
    dict(name="tx.assignment", kind="actuator", shape=("B", "N"), dtype="int32", bits=None, range="line_index",
         unit="index", rate=None, latency=0.0, agg="latest", init=-1,
         doc="Logical ring-to-line map; -1 = unassigned (07 P1-12, PAT-1 barrel shift)."),
    dict(name="datapath.bitstat_threshold", kind="actuator", shape=("B", "N"), dtype="u:bits", bits="adc_bits",
         range="unsigned", unit="code", rate=None, latency=0.0, agg="latest",
         doc="Comparator threshold of the bit-statistics sampler, in monitor ADC codes (07 P0-11)."),
    dict(name="monitor.pga_select", kind="actuator", shape=("B", "N"), dtype="int32", bits=None, range="pga_index",
         unit="index", rate=None, latency=0.0, agg="latest",
         doc="Index into ring.monitor.pga_gains (07 P1-10)."),
    dict(name="laser.tune", kind="actuator", shape=("B", "L"), dtype="i:bits", bits="laser_tune_bits",
         range="signed", unit="code", rate=None, latency=0.0, agg="latest", present="laser_tune", scale="laser",
         doc="Optional per-line laser frequency trim (07 P0-7, P2-8)."),
    # --- sensors (07 P0-7, P0-11, P1-10)
    dict(name="tx.monitor.samples", kind="sensor", shape=("B", "N", "M"), dtype="i:bits", bits="adc_bits",
         range="unsigned", unit="code", rate="adc_rate_hz", latency="adc_latency_s", agg="block",
         count="tx.monitor.count", scale="adc", doc="Every monitor ADC sample released since the last step."),
    dict(name="tx.monitor.sample_ticks", kind="sensor", shape=("B", "N", "M"), dtype="int64", bits=None,
         range="tick", unit="tick", rate="adc_rate_hz", latency="adc_latency_s", agg="block",
         count="tx.monitor.count", doc="Sample tick of each entry of tx.monitor.samples (end of its window)."),
    dict(name="tx.monitor.count", kind="sensor", shape=("B", "N"), dtype="int32", bits=None, range="count",
         unit="count", rate=None, latency=0.0, agg="count", doc="Number of valid entries along M (<= M)."),
    dict(name="tx.monitor.latest", kind="sensor", shape=("B", "N"), dtype="i:bits", bits="adc_bits",
         range="unsigned", unit="code", rate="adc_rate_hz", latency="adc_latency_s", agg="latest", scale="adc",
         doc="Most recent released monitor sample (held)."),
    dict(name="temp.{t}", kind="sensor", shape=("B", 1), dtype="i:bits", bits="temp", range="unsigned",
         unit="code", rate="temp", latency="temp", agg="latest", scale="temp",
         doc="On-die temperature sensor code (07 P1-10)."),
    dict(name="heater.readback_v", kind="sensor", shape=("B", "N"), dtype="i:bits", bits="readback_adc_bits",
         range="unsigned", unit="code", rate="readback_rate_hz", latency="readback_latency_s", agg="latest",
         present="heater_readback_present", scale="readback_v", doc="Heater voltage readback (07 P1-10)."),
    dict(name="heater.readback_i", kind="sensor", shape=("B", "N"), dtype="i:bits", bits="readback_adc_bits",
         range="unsigned", unit="code", rate="readback_rate_hz", latency="readback_latency_s", agg="latest",
         present="heater_readback_present", scale="readback_i", doc="Heater current readback (07 P1-10)."),
    dict(name="tx.ones_count", kind="sensor", shape=("B", "N"), dtype="int64", bits=None, range="nonneg",
         unit="count", rate=None, latency=0.0, agg="sum", doc="Ones transmitted in the controller window (07 P0-11)."),
    dict(name="tx.bits", kind="sensor", shape=("B", "N"), dtype="int64", bits=None, range="nonneg", unit="count",
         rate=None, latency=0.0, agg="sum", doc="Bits transmitted in the controller window (07 P0-11)."),
    dict(name="datapath.bitstat_above", kind="sensor", shape=("B", "N"), dtype="int32", bits=None,
         range="bitstat", unit="count", rate=None, latency=0.0, agg="latest",
         doc="Monitor samples above datapath.bitstat_threshold in the latest window (07 P0-11)."),
    dict(name="rx.error_count", kind="sensor", shape=("B", "N"), dtype="int64", bits=None, range="nonneg",
         unit="count", rate="error_count", latency="error_count_latency_s", agg="latest",
         doc="Errors counted in the last datapath.error_count_period_s (07 P0-11); held until the next delivery."),
    dict(name="rx.error_count_valid", kind="sensor", shape=("B", "N"), dtype="bool", bits=None, range="flag",
         unit="flag", rate=None, latency=0.0, agg="any",
         doc="True when a new rx.error_count was delivered since the last step."),
    dict(name="monitor_valid", kind="sensor", shape=("B", "N"), dtype="bool", bits=None, range="flag",
         unit="flag", rate=None, latency=0.0, agg="all", init=1,
         doc="False while a fault masks the monitor (adc_drop_mask, adc_stuck; 07 P1-10)."),
    dict(name="adc_overrange", kind="sensor", shape=("B", "N"), dtype="bool", bits=None, range="flag",
         unit="flag", rate=None, latency=0.0, agg="any", doc="A monitor sample clipped at an ADC rail (07 P1-10)."),
    dict(name="tx.heater.applied", kind="sensor", shape=("B", "N"), dtype="u:bits", bits="heater_dac_bits",
         range="heater", unit="code", rate=None, latency=0.0, agg="latest",
         doc="Latched heater DAC register (07 P0-6 applied codes; never the slewed analog value)."),
    dict(name="tx.bias.applied", kind="sensor", shape=("B", "N"), dtype="u:bits", bits="bias_dac_bits",
         range="unsigned", unit="code", rate=None, latency=0.0, agg="latest", scale="bias",
         doc="Latched bias DAC register (07 P0-6 applied codes)."),
)

# Default owners (07 P0-7 example binding {tx.heater: controller, tx.bias: scenario | controller, laser.tune: scenario}).
DEFAULT_BINDINGS: dict[str, str] = {
    "tx.heater": "controller",
    "tx.bias": "scenario",
    "tx.assignment": "scenario",
    "datapath.bitstat_threshold": "controller",
    "monitor.pga_select": "controller",
    "laser.tune": "scenario",
}


def _udtype(bits: int) -> str:
    """Smallest unsigned dtype for a ``bits``-wide code (uint16 for the 16-bit v1 converters)."""
    return "uint16" if bits <= 16 else "uint32" if bits <= 32 else "int64"


def _idtype(magnitude_bits: int) -> str:
    """Signed dtype for a code with ``magnitude_bits`` value bits (sign excluded): int32 (07 P0-7) up to 31."""
    return "int32" if magnitude_bits <= 31 else "int64"


def build_v1_channels(hw: ChannelHardware | None = None, *, n_lines: int = 1) -> dict[str, ChannelSpec]:
    """The v1 channel set for ``hw``, in table order (temperature channels in sensor order).

    ``n_lines`` (L) bounds ``tx.assignment``. Optional channels whose hardware is absent are left out.
    """
    hw = ChannelHardware() if hw is None else hw
    if n_lines < 1:
        raise ChannelError(f"n_lines must be >= 1, got {n_lines}")
    out: dict[str, ChannelSpec] = {}
    for row in V1_CHANNEL_TABLE:
        if "present" in row and not getattr(hw, row["present"]):
            continue
        if "{t}" in row["name"]:
            for ts in hw.temp_sensors:
                spec = _build_row(row, hw, n_lines, ts)
                out[spec.name] = spec
        else:
            spec = _build_row(row, hw, n_lines, None)
            out[spec.name] = spec
    names = [s.name for s in out.values()]
    if len(set(names)) != len(names):
        raise ChannelError(f"duplicate channel names in {names}")
    return out


def _attr(hw: ChannelHardware, ref, ts: TempSensorHw | None, what: str):
    if ref is None or isinstance(ref, (int, float)):
        return ref
    if ref == "temp":
        return {"bits": ts.bits, "rate": ts.rate_hz, "latency": ts.latency_s}[what]
    if ref == "error_count":
        return 1.0 / hw.error_count_period_s
    return getattr(hw, ref)


def _build_row(row: dict, hw: ChannelHardware, n_lines: int, ts: TempSensorHw | None) -> ChannelSpec:
    name = row["name"].replace("{t}", ts.name) if ts is not None else row["name"]
    bits = _attr(hw, row["bits"], ts, "bits")
    rule = row["range"]
    if rule == "heater":
        lo, hi = int(hw.heater_code_min), int(hw.heater_max)
        if hi > (1 << hw.heater_dac_bits) - 1:
            raise ChannelError(f"ring.heater.code_max {hi} exceeds the {hw.heater_dac_bits}-bit DAC")
    elif rule == "unsigned":
        lo, hi = 0, (1 << int(bits)) - 1
    elif rule == "signed":
        lo, hi = -(1 << (int(bits) - 1)), (1 << (int(bits) - 1)) - 1
    elif rule == "line_index":
        lo, hi = -1, n_lines - 1
    elif rule == "pga_index":
        lo, hi = 0, len(hw.pga_gains) - 1
    elif rule == "tick":
        lo, hi = 0, INT64_MAX
    elif rule == "count":
        lo, hi = 0, INT32_MAX
    elif rule == "nonneg":
        lo, hi = 0, INT64_MAX
    elif rule == "bitstat":
        lo, hi = 0, int(hw.bitstat_window_bits)
    elif rule == "flag":
        lo, hi = 0, 1
    else:
        raise ChannelError(f"{name}: unknown range rule {rule!r}")
    if row["dtype"] == "u:bits":
        dtype = _udtype(int(bits))
    elif row["dtype"] == "i:bits":
        # a signed w-bit code has w - 1 magnitude bits; an unsigned w-bit code needs w (int32 holds 31)
        dtype = _idtype(int(bits) - (1 if rule == "signed" else 0))
    else:
        dtype = row["dtype"]
    scale = offset = phys = None
    sk = row.get("scale")
    if sk == "bias":
        scale = (hw.bias_v_max_v - hw.bias_v_min_v) / ((1 << hw.bias_dac_bits) - 1)
        offset, phys = hw.bias_v_min_v, "V"
    elif sk == "laser":
        scale, offset, phys = hw.laser_tune_lsb_hz, 0.0, "Hz"
    elif sk == "adc" and hw.adc_full_scale_a is not None:
        scale, offset, phys = hw.adc_full_scale_a / ((1 << hw.adc_bits) - 1), 0.0, "A"
    elif sk == "temp":
        scale, offset, phys = ts.lsb_k, ts.offset_c, "degC"
    elif sk == "readback_v":
        scale, offset, phys = hw.readback_v_full_scale_v / ((1 << hw.readback_adc_bits) - 1), 0.0, "V"
    elif sk == "readback_i":
        scale, offset, phys = hw.readback_i_full_scale_a / ((1 << hw.readback_adc_bits) - 1), 0.0, "A"
    rate = _attr(hw, row["rate"], ts, "rate")
    latency = _attr(hw, row["latency"], ts, "latency")
    return ChannelSpec(
        name=name, kind=row["kind"], shape=row["shape"], dtype=dtype, bits=None if bits is None else int(bits),
        min=lo, max=hi, unit=row["unit"], rate_hz=None if rate is None else float(rate),
        latency_s=float(latency or 0.0), agg=row["agg"], count_channel=row.get("count"),
        init=int(row.get("init", max(lo, 0) if lo <= 0 <= hi else lo)), scale=scale, offset=offset, phys_unit=phys,
        optional="present" in row, doc=row["doc"])


# ---- binding -----------------------------------------------------------------------------------------------------

def resolve_bindings(channels: Mapping[str, ChannelSpec], bind: Mapping[str, str] | None = None, *,
                     scenario_drives: Iterable[str] = (), scenario_offsets: Iterable[str] = ()) -> dict[str, str]:
    """Owner of every actuator channel: ``controller`` or ``scenario`` (07 P0-7).

    ``bind`` is ``controller_interface.bind`` (channel -> owner); unspecified actuators take
    :data:`DEFAULT_BINDINGS` (or ``scenario`` for channels without a default). ``scenario_drives`` lists the actuator
    channels the scenario sets ABSOLUTELY; ``scenario_offsets`` those it adds an offset to. Raises BindingError if a
    channel is bound to the controller and also driven absolutely by the scenario, if a name is unknown, if it is a
    sensor, or if an owner is not 'controller' / 'scenario'.
    """
    bind = dict(bind or {})
    actuators = {n: c for n, c in channels.items() if c.kind == "actuator"}
    errors: list[str] = []
    for name, owner in bind.items():
        if name not in channels:
            errors.append(f"controller_interface.bind: unknown channel {name!r} (actuators: {sorted(actuators)})")
        elif name not in actuators:
            errors.append(f"controller_interface.bind: {name!r} is a sensor; only actuators are bound")
        elif owner not in OWNERS:
            errors.append(f"controller_interface.bind.{name} = {owner!r}: must be one of {OWNERS}")
    out = {n: bind.get(n, DEFAULT_BINDINGS.get(n, "scenario")) for n in actuators}
    for name in scenario_drives:
        if name not in actuators:
            errors.append(f"scenario sets {name!r}, which is not an actuator channel (actuators: {sorted(actuators)})")
        elif out.get(name) == "controller":
            errors.append(f"{name} is bound to the controller and the scenario also sets it absolutely; bind it to "
                          f"one of them, or make the scenario add an offset instead (07 P0-7)")
    for name in scenario_offsets:
        if name not in actuators:
            errors.append(f"scenario offsets {name!r}, which is not an actuator channel")
    if errors:
        raise BindingError("; ".join(errors))
    return out


# ---- flat int64 ABI (07 P0-9) ------------------------------------------------------------------------------------

ABI_VERSION = 1
OBS_HEADER = 4            # obs_i64[0:4] = [tick, active_ring (-1 = all rings), reserved, reserved]
OBS_TICK = 0
OBS_ACTIVE_RING = 1


class ObsId(enum.IntEnum):
    """Fixed ids of the v1 sensor channels in the ABI offset tables. Values never change; append only."""
    TX_MONITOR_SAMPLES = 0
    TX_MONITOR_SAMPLE_TICKS = 1
    TX_MONITOR_COUNT = 2
    TX_MONITOR_LATEST = 3
    HEATER_READBACK_V = 4
    HEATER_READBACK_I = 5
    TX_ONES_COUNT = 6
    TX_BITS = 7
    DATAPATH_BITSTAT_ABOVE = 8
    RX_ERROR_COUNT = 9
    RX_ERROR_COUNT_VALID = 10
    MONITOR_VALID = 11
    ADC_OVERRANGE = 12
    TX_HEATER_APPLIED = 13
    TX_BIAS_APPLIED = 14


TEMP_ID_BASE = 15         # temp.<name> channels get TEMP_ID_BASE + s, in temp_sensors order


class ActId(enum.IntEnum):
    """Fixed ids of the v1 actuator channels in the ABI offset table. Values never change; append only."""
    TX_HEATER = 0
    TX_BIAS = 1
    TX_ASSIGNMENT = 2
    DATAPATH_BITSTAT_THRESHOLD = 3
    MONITOR_PGA_SELECT = 4
    LASER_TUNE = 5


OBS_ID_BY_NAME: dict[str, int] = {
    "tx.monitor.samples": ObsId.TX_MONITOR_SAMPLES, "tx.monitor.sample_ticks": ObsId.TX_MONITOR_SAMPLE_TICKS,
    "tx.monitor.count": ObsId.TX_MONITOR_COUNT, "tx.monitor.latest": ObsId.TX_MONITOR_LATEST,
    "heater.readback_v": ObsId.HEATER_READBACK_V, "heater.readback_i": ObsId.HEATER_READBACK_I,
    "tx.ones_count": ObsId.TX_ONES_COUNT, "tx.bits": ObsId.TX_BITS,
    "datapath.bitstat_above": ObsId.DATAPATH_BITSTAT_ABOVE, "rx.error_count": ObsId.RX_ERROR_COUNT,
    "rx.error_count_valid": ObsId.RX_ERROR_COUNT_VALID, "monitor_valid": ObsId.MONITOR_VALID,
    "adc_overrange": ObsId.ADC_OVERRANGE, "tx.heater.applied": ObsId.TX_HEATER_APPLIED,
    "tx.bias.applied": ObsId.TX_BIAS_APPLIED,
}
ACT_ID_BY_NAME: dict[str, int] = {
    "tx.heater": ActId.TX_HEATER, "tx.bias": ActId.TX_BIAS, "tx.assignment": ActId.TX_ASSIGNMENT,
    "datapath.bitstat_threshold": ActId.DATAPATH_BITSTAT_THRESHOLD, "monitor.pga_select": ActId.MONITOR_PGA_SELECT,
    "laser.tune": ActId.LASER_TUNE,
}


@dataclass(frozen=True)
class Field:
    """One channel's place in a flat vector: offset, element shape (without B), size, id and dtype."""
    name: str
    offset: int
    shape: tuple
    size: int
    ch_id: int
    dtype: str


@dataclass(frozen=True)
class AbiLayout:
    """Flat int64 layout of one batch element's observation and action vectors (07 P0-7, P0-9).

    obs_i64 = [header (OBS_HEADER)] [sensor values, in channel order] [new-sample flags, in channel order]
    act_i64 = [actuator values bound to the controller, in channel order]

    Values are row-major (C order) over the element shape; bools are 0/1. Offsets are also published as tables
    indexed by channel id (:class:`ObsId`, :class:`ActId`, TEMP_ID_BASE + s) so a compiled controller can find a
    channel without knowing N or M at compile time (``mrm.control.api.obs_offset``).
    """
    version: int
    dims: tuple
    obs: tuple
    new: tuple
    act: tuple
    obs_size: int
    act_size: int
    n_obs_ids: int
    n_act_ids: int

    def _find(self, fields: tuple, name: str) -> Field:
        for f in fields:
            if f.name == name:
                return f
        raise ChannelError(f"channel {name!r} is not in this layout (have {[f.name for f in fields]})")

    def obs_field(self, name: str) -> Field:
        return self._find(self.obs, name)

    def new_field(self, name: str) -> Field:
        return self._find(self.new, name)

    def act_field(self, name: str) -> Field:
        return self._find(self.act, name)

    def _table(self, fields: tuple, n: int) -> np.ndarray:
        t = np.full(n, -1, dtype=np.int64)
        for f in fields:
            t[f.ch_id] = f.offset
        return t

    def obs_table(self) -> np.ndarray:
        """Offset of every sensor id in obs_i64 (-1 if absent)."""
        return self._table(self.obs, self.n_obs_ids)

    def new_table(self) -> np.ndarray:
        """Offset of every sensor id's new-sample flags in obs_i64 (-1 if absent)."""
        return self._table(self.new, self.n_obs_ids)

    def act_table(self) -> np.ndarray:
        """Offset of every actuator id in act_i64 (-1 if absent or not bound to the controller)."""
        return self._table(self.act, self.n_act_ids)

    def describe(self) -> list[dict]:
        """Rows {vector, name, id, offset, size, shape, dtype} for docs, the binary bridge and golden vectors."""
        rows = [dict(vector="obs", name=n, id=-1, offset=o, size=1, shape=(), dtype="int64")
                for n, o in (("tick", OBS_TICK), ("active_ring", OBS_ACTIVE_RING))]
        for vec, fields in (("obs", self.obs), ("new", self.new), ("act", self.act)):
            rows += [dict(vector=vec, name=f.name, id=f.ch_id, offset=f.offset, size=f.size, shape=f.shape,
                          dtype=f.dtype) for f in fields]
        return rows

    def fingerprint(self) -> str:
        """sha256 (16 hex) of the layout; changes whenever any offset, shape, dtype or id changes."""
        blob = json.dumps({"v": self.version, "rows": self.describe(), "dims": list(self.dims)}, sort_keys=True,
                          default=list)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    # -- packing (Python slow path and tests; the kernel writes the same layout directly) --

    def pack_obs(self, tick: int, sensors: Mapping[str, np.ndarray], new: Mapping[str, np.ndarray],
                 active_ring: int = -1, out: np.ndarray | None = None) -> np.ndarray:
        """Write (B, obs_size) int64 from batched sensor arrays (B, ...) and new flags (B, ...)."""
        b = int(dict(self.dims)["B"])
        if out is None:
            out = np.zeros((b, self.obs_size), dtype=np.int64)
        out[:, OBS_TICK] = int(tick)
        out[:, OBS_ACTIVE_RING] = int(active_ring)
        for f in self.obs:
            out[:, f.offset:f.offset + f.size] = np.asarray(sensors[f.name]).reshape(b, f.size)
        for f in self.new:
            out[:, f.offset:f.offset + f.size] = np.asarray(new[f.name]).reshape(b, f.size)
        return out

    def unpack_obs(self, obs_i64: np.ndarray) -> tuple[int, int, dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Inverse of pack_obs: (tick, active_ring, sensors, new) with channel dtypes and (B, ...) shapes."""
        b = obs_i64.shape[0]
        sensors = {f.name: obs_i64[:, f.offset:f.offset + f.size].reshape((b,) + f.shape).astype(f.dtype)
                   for f in self.obs}
        new = {f.name: obs_i64[:, f.offset:f.offset + f.size].reshape((b,) + f.shape).astype(bool)
               for f in self.new}
        return int(obs_i64[0, OBS_TICK]), int(obs_i64[0, OBS_ACTIVE_RING]), sensors, new

    def pack_act(self, actions: Mapping[str, np.ndarray], out: np.ndarray) -> np.ndarray:
        """Write the given actuator arrays (B, ...) into act_i64 (B, act_size); channels not given are left as is."""
        b = out.shape[0]
        for name, val in actions.items():
            f = self.act_field(name)
            out[:, f.offset:f.offset + f.size] = np.asarray(val).reshape(b, f.size)
        return out

    def unpack_act(self, act_i64: np.ndarray, *, cast: bool = True) -> dict[str, np.ndarray]:
        """Actuator arrays (B, ...) from act_i64 (B, act_size): in the channel dtypes (``cast=True``) or as int64
        copies (``cast=False``). Casting WRAPS out-of-range values (70000 -> 4464 in uint16, -5 -> 65531), so
        unvalidated controller output must be unpacked with ``cast=False`` and range-checked first
        (``mrm.control.api.ActionValidator``)."""
        b = act_i64.shape[0]
        return {f.name: (act_i64[:, f.offset:f.offset + f.size].reshape((b,) + f.shape).astype(f.dtype) if cast
                         else act_i64[:, f.offset:f.offset + f.size].reshape((b,) + f.shape).copy())
                for f in self.act}


def assign_channel_ids(channels: Mapping[str, ChannelSpec]) -> tuple[dict[str, int], dict[str, int]]:
    """(sensor ids, actuator ids) of a FULL channel set (07 P0-9 offset tables).

    Fixed channels take their :class:`ObsId` / :class:`ActId`. Temperature channels take TEMP_ID_BASE + s in the
    order they appear (the order of ``temp_sensors``, so s is the index into ``ControllerSpec.temp_sensors``). Any
    other channel takes the next free id in order of appearance. Compute the ids on the full configured set, before
    ``expose`` hides channels, so hiding a channel never renumbers another one.
    """
    sensors = [c for c in channels.values() if c.kind == "sensor"]
    next_obs = TEMP_ID_BASE + sum(1 for c in sensors if c.name.startswith("temp."))
    temp_i = 0
    obs_ids: dict[str, int] = {}
    for c in sensors:
        if c.name in OBS_ID_BY_NAME:
            obs_ids[c.name] = int(OBS_ID_BY_NAME[c.name])
        elif c.name.startswith("temp."):
            obs_ids[c.name] = TEMP_ID_BASE + temp_i
            temp_i += 1
        else:
            obs_ids[c.name] = next_obs
            next_obs += 1
    next_act = len(ActId)
    act_ids: dict[str, int] = {}
    for c in channels.values():
        if c.kind != "actuator":
            continue
        if c.name in ACT_ID_BY_NAME:
            act_ids[c.name] = int(ACT_ID_BY_NAME[c.name])
        else:
            act_ids[c.name] = next_act
            next_act += 1
    return obs_ids, act_ids


def make_abi_layout(channels: Mapping[str, ChannelSpec], dims: Mapping[str, int],
                    bindings: Mapping[str, str], *, obs_ids: Mapping[str, int] | None = None,
                    act_ids: Mapping[str, int] | None = None) -> AbiLayout:
    """Generate the flat layout from channel specs (07 P0-7: "an offset per channel").

    All sensor channels in ``channels`` enter obs_i64 (with a new-sample flag block each); actuator channels enter
    act_i64 only if ``bindings[name] == 'controller'``. ``dims`` must give B, N, L, M (S optional). Channel ids come
    from ``obs_ids`` / ``act_ids`` when given (:func:`assign_channel_ids` of the full configured set; required when
    ``channels`` is a subset, e.g. after ``controller_interface.expose``), else from :func:`assign_channel_ids` of
    ``channels`` itself. Append new channels to the enums to keep ids stable across configs.
    """
    dims = {k: int(v) for k, v in dims.items()}
    obs_fields: list[Field] = []
    new_fields: list[Field] = []
    act_fields: list[Field] = []
    sensors = [c for c in channels.values() if c.kind == "sensor"]
    actuators = [c for c in channels.values() if c.kind == "actuator" and bindings.get(c.name) == "controller"]
    own_obs, own_act = assign_channel_ids(channels)
    ids = dict(own_obs if obs_ids is None else obs_ids)
    aids = dict(own_act if act_ids is None else act_ids)
    missing = [c.name for c in sensors if c.name not in ids] + [c.name for c in actuators if c.name not in aids]
    if missing:
        raise ChannelError(f"make_abi_layout: no channel id for {missing}")
    used = [ids[c.name] for c in sensors]
    if len(set(used)) != len(used) or len({aids[c.name] for c in actuators}) != len(actuators):
        raise ChannelError("make_abi_layout: two channels share an id")
    off = OBS_HEADER
    for c in sensors:
        shp = c.element_shape(dims)
        size = int(np.prod(shp)) if shp else 1
        obs_fields.append(Field(c.name, off, shp, size, int(ids[c.name]), c.dtype))
        off += size
    for c in sensors:
        shp = c.flag_shape(dims)
        size = int(np.prod(shp)) if shp else 1
        new_fields.append(Field(c.name, off, shp, size, int(ids[c.name]), "bool"))
        off += size
    obs_size = off
    off = 0
    for c in actuators:
        shp = c.element_shape(dims)
        size = int(np.prod(shp)) if shp else 1
        act_fields.append(Field(c.name, off, shp, size, int(aids[c.name]), c.dtype))
        off += size
    n_obs_ids = max([TEMP_ID_BASE] + [int(v) + 1 for v in ids.values()])
    n_act_ids = max([len(ActId)] + [int(v) + 1 for v in aids.values()])
    return AbiLayout(version=ABI_VERSION, dims=tuple(sorted(dims.items())), obs=tuple(obs_fields),
                     new=tuple(new_fields), act=tuple(act_fields), obs_size=obs_size, act_size=off,
                     n_obs_ids=n_obs_ids, n_act_ids=n_act_ids)
