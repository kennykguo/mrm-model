"""Controller API: ControllerSpec, Observation, the Controller protocol, adapters, validation, telemetry, composition
and the compiled-controller ABI (07 P1-1, P1-19, P1-20, P2-7, P0-7, P0-9).

What a controller sees
----------------------
- ``reset(spec)`` once, with a :class:`ControllerSpec`: typed facts about the hardware interface (tick, controller
  period, ADC, heater DAC, bias DAC, temperature sensors, channel specs, bus order) and ``nominal``, the view of the
  UNSAMPLED design filtered by ``controller_interface.knowledge`` (:class:`mrm.control.knowledge.NominalView`: a key
  no granted token releases raises KnowledgeError; the experiment section is never released). The spec is
  therefore identical across seeds and uncertainty worlds (07 D13, ``ControllerSpec.fingerprint``). There is no
  ``extras`` field.
- ``step(obs)`` at every controller edge, with an :class:`Observation`: ``tick`` (int64), ``t_s`` (derived),
  ``sensors`` (channel -> (B, ...) array, read-only) and ``new`` (channel -> new-sample flags). Access to an
  undeclared channel raises :class:`UndeclaredChannelError` (07 P1-1 ``test_no_truth_leak``).
- ``step`` returns ``dict[str, ndarray]`` of actuator codes (B, ...). A channel left out holds its last command.
  The runner checks shape, dtype and range with :class:`ActionValidator` (``on_out_of_range: clip | error`` plus a
  saturation count, 07 P1-19).

Timing (07 P0-6, ``mrm.core.timing``): the command of edge k latches in the DAC register at
``k + controller.latency_ticks``, then passes the plant's DAC latency, slew limit and settle filter.

Compiled controllers (07 P0-9)
------------------------------
Two numba functions with flat arrays only::

    init(spec_i64, spec_f64) -> (p_i64, p_f64, s_i64, s_f64)
    step(p_i64, p_f64, s_i64, s_f64, obs_i64, act_i64) -> None        # writes act_i64 in place

``spec_i64``/``spec_f64`` come from :func:`spec_arrays` (header constants ``SPEC_I_*``/``SPEC_F_*``, offset tables of
the :class:`mrm.core.channels.AbiLayout`, then the controller's own integer and float parameters). ``obs_i64`` and
``act_i64`` are ONE batch element's vectors in that layout. On entry ``act_i64`` holds the controller's previous
command (initially each channel's ``init``), so a step writes only what it changes. ``p_*`` are parameters (never
written by step), ``s_*`` state. The kernel calls ``step`` per batch element inside ``prange`` over B.

An optional ``fast_kernel`` with the same signature runs once per released monitor sample block inside the plant
loop (07 P0-7): ``fobs`` = [tick, N, new[N], code[N], sample_tick[N]] and ``fact`` = [heater offset[N],
bias offset[N]] in codes, added to the latched command before the DAC latency FIFO. It shares ``s_*`` with ``step``.
:class:`CompiledController` packages these functions and also implements the Python protocol (slow path), so the
same controller runs in both paths.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

import numba as nb
import numpy as np
from numba.core.dispatcher import Dispatcher

from mrm.control.knowledge import knowledge_view
from mrm.core.channels import (
    INT64_MAX, V1_CHANNEL_TABLE, AbiLayout, BindingError, ChannelError, ChannelHardware, ChannelSpec,
    assign_channel_ids, build_v1_channels, make_abi_layout, resolve_bindings, samples_per_step,
)
from mrm.core.timing import TimingError, check_tick_table, is_edge, edge_index, rate_to_period_ticks

__all__ = [
    "DacFacts", "AdcFacts", "TempSensorFacts", "ControllerSpec", "Observation", "StrictChannels",
    "UndeclaredChannelError", "ActionRangeError", "Controller", "BaseController", "RingContext", "DieContext",
    "RingObservation", "DieObservation", "PerRing", "PerDie", "ActionValidator", "validate_telemetry",
    "TELEMETRY_KEYS", "controller_telemetry", "controller_fast_kernel", "CompositeController", "CompiledController",
    "spec_arrays", "obs_offset", "new_offset", "act_offset", "user_i64", "user_f64", "call_step_batch",
    "fast_obs_size", "fast_act_size", "fo_new", "fo_code", "fo_sample_tick", "fa_heater", "fa_bias",
    "SCHEDULES", "ON_OUT_OF_RANGE", "nominal_line_grid_hz", "merge_round_robin",
]

SCHEDULES = ("simultaneous", "round_robin")
ON_OUT_OF_RANGE = ("clip", "error")
#: the sensor channels whose ChannelSpec scale converts monitor ADC codes to amperes (monitor_nominal knowledge)
_ADC_SCALED_CHANNELS = frozenset(r["name"] for r in V1_CHANNEL_TABLE if r.get("scale") == "adc")


class UndeclaredChannelError(KeyError):
    """Access to a channel the controller interface does not declare (07 P1-1)."""

    def __str__(self) -> str:      # KeyError would print the repr of the message
        return str(self.args[0]) if self.args else ""


class ActionRangeError(ValueError):
    """An action value outside its channel range with on_out_of_range = error (07 P1-19)."""


# ---- typed facts ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class DacFacts:
    """What a controller may know about a DAC (07 P1-1 "heater-DAC facts", "bias-DAC facts").

    bits; code_min/code_max: the hardware code range (07 P1-11); latency_ticks: DAC latency owned by the plant
    (ring.heater.latency_s / ring.bias_dac.latency_s); update_period_ticks: minimum update period (None = every
    controller step); phys_min/phys_max/phys_unit: nominal output range when documented (bias: 1.0-3.75 V).
    """
    bits: int
    code_min: int
    code_max: int
    latency_ticks: int
    update_period_ticks: int | None = None
    phys_min: float | None = None
    phys_max: float | None = None
    phys_unit: str | None = None


@dataclass(frozen=True)
class AdcFacts:
    """Monitor ADC facts (07 P1-1: adc_rate_hz, adc_full_scale_codes, adc_latency_ticks, samples per step).

    rate_hz and period_ticks are PER RING (with an ADC mux, channels_per_adc rings share one converter and are
    sampled at staggered phases); integration_window_ticks is the integrate-and-dump boxcar (07 P0-7), 0 for an
    rc AFE.
    """
    bits: int
    full_scale_codes: int
    rate_hz: float
    period_ticks: int
    latency_ticks: int
    integration_window_ticks: int
    afe_type: str
    channels_per_adc: int
    pga_gains: tuple


@dataclass(frozen=True)
class TempSensorFacts:
    """One temperature sensor (07 P1-10): its channel, converter and timing. T = offset_c + lsb_k * code."""
    name: str
    channel: str
    bits: int
    lsb_k: float
    offset_c: float
    rate_hz: float
    period_ticks: int
    latency_ticks: int
    site: str
    type: str


# ---- ControllerSpec ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ControllerSpec:
    """Everything a controller is allowed to know at reset (07 P1-1). Build it with :meth:`build`.

    Fields: n_batch (B), n_rings (N), n_lines (L), tick_s (sim.dt_plant_s, s), period_ticks (controller period),
    phase_ticks (first edge), samples_per_step (M), adc, heater_dac, bias_dac, temp_sensors, channels (exposed
    sensors + actuators bound to the controller), bindings (owner of every actuator), layout (flat ABI),
    bus_order (ring index at each bus position along the light path, N; array.bus_order, as in realize),
    line_grid_hz (nominal grid in Hz, L, or None if not in knowledge), ring_to_line (nominal assignment, N, or
    None), knowledge (controller_interface.knowledge), nominal (the unsampled design filtered by knowledge: any
    design given here is wrapped in a :class:`mrm.control.knowledge.NominalView`, 07 D13; None = no design),
    schedule (simultaneous | round_robin: the engine updates ring j % N only at controller edge j, 07 P2-7),
    on_out_of_range (clip | error, on both engine paths), heater_readback (tuple of
    (name, value) readback facts, or None), error_count_period_ticks, tick_jitter_ticks
    (largest delay of a controller edge, 07 P2-7; the runner draws each edge's delay with
    ``mrm.core.timing.edge_jitter``; 0 = no jitter).
    """
    n_batch: int
    n_rings: int
    n_lines: int
    tick_s: float
    period_ticks: int
    phase_ticks: int
    samples_per_step: int
    adc: AdcFacts
    heater_dac: DacFacts
    bias_dac: DacFacts
    temp_sensors: tuple
    channels: dict
    bindings: dict
    layout: AbiLayout
    bus_order: tuple
    line_grid_hz: tuple | None = None
    ring_to_line: tuple | None = None
    knowledge: tuple = ()
    nominal: object = None
    schedule: str = "simultaneous"
    on_out_of_range: str = "clip"
    heater_readback: tuple | None = None
    error_count_period_ticks: int | None = None
    tick_jitter_ticks: int = 0

    def __post_init__(self):
        # 07 D13: a controller reads the design only through the knowledge filter (also after dataclasses.replace)
        object.__setattr__(self, "knowledge", tuple(self.knowledge or ()))
        object.__setattr__(self, "nominal", knowledge_view(self.nominal, self.knowledge))

    # -- derived --
    @property
    def rate_hz(self) -> float:
        """Controller update rate, 1 / (period_ticks * tick_s)."""
        return 1.0 / (self.period_ticks * self.tick_s)

    @property
    def period_s(self) -> float:
        return self.period_ticks * self.tick_s

    @property
    def samples_per_tick(self) -> int:
        """Alias of samples_per_step (07 P1-1 calls it samples_per_tick: samples per controller tick)."""
        return self.samples_per_step

    @property
    def adc_rate_hz(self) -> float:
        """07 P1-1 name: per-ring monitor ADC sample rate (= adc.rate_hz)."""
        return self.adc.rate_hz

    @property
    def adc_full_scale_codes(self) -> int:
        """07 P1-1 name: largest monitor ADC code (= adc.full_scale_codes)."""
        return self.adc.full_scale_codes

    @property
    def adc_latency_ticks(self) -> int:
        """07 P1-1 name: monitor ADC conversion latency in ticks (= adc.latency_ticks)."""
        return self.adc.latency_ticks

    @property
    def dims(self) -> dict[str, int]:
        return {"B": self.n_batch, "N": self.n_rings, "L": self.n_lines, "M": self.samples_per_step,
                "S": len(self.temp_sensors)}

    def shape_of(self, channel: str) -> tuple[int, ...]:
        """Concrete (B, ...) shape of a declared channel."""
        if channel not in self.channels:
            raise UndeclaredChannelError(f"channel {channel!r} is not declared for this controller "
                                         f"(declared: {sorted(self.channels)})")
        return self.channels[channel].resolve_shape(self.dims)

    def active_ring(self, tick: int) -> int:
        """The ring that the controller edge at ``tick`` updates under ``schedule = round_robin`` (07 P2-7): j % N
        with j = (tick - phase_ticks) // period_ticks the edge index (a jittered edge keeps its index, as in the
        kernel); -1 (every ring) when simultaneous."""
        if self.schedule != "round_robin":
            return -1
        return int((int(tick) - self.phase_ticks) // self.period_ticks) % self.n_rings

    def active_mask(self, tick: int) -> np.ndarray | None:
        """:meth:`active_ring` as the (N,) bool mask of ``Observation.active`` (None when simultaneous)."""
        r = self.active_ring(tick)
        if r < 0:
            return None
        m = np.zeros(self.n_rings, dtype=bool)
        m[r] = True
        return m

    @property
    def actuators(self) -> dict[str, ChannelSpec]:
        """Actuator channels bound to the controller."""
        return {n: c for n, c in self.channels.items() if c.kind == "actuator"}

    @property
    def sensors(self) -> dict[str, ChannelSpec]:
        return {n: c for n, c in self.channels.items() if c.kind == "sensor"}

    @classmethod
    def build(cls, *, n_batch: int, n_rings: int, n_lines: int, tick_s: float, update_rate_hz: float,
              hw: ChannelHardware | None = None, channels: Mapping[str, ChannelSpec] | None = None,
              bind: Mapping[str, str] | None = None, scenario_drives: Iterable[str] = (),
              scenario_offsets: Iterable[str] = (), expose: Iterable[str] | None = None,
              bus_order: Sequence[int] | None = None, line_grid_hz: Sequence[float] | None = None,
              ring_to_line: Sequence[int] | None = None, knowledge: Iterable[str] = (), nominal: object = None,
              schedule: str = "simultaneous", on_out_of_range: str = "clip", phase_ticks: int = 0,
              tick_jitter_s: float = 0.0) -> "ControllerSpec":
        """Build a spec from design values. Every period and latency is converted to whole ticks; ONE TimingError
        names every config key that is not a whole number of ticks (07 P0-6).

        ``hw`` defaults to :class:`ChannelHardware()`; ``channels`` defaults to ``build_v1_channels(hw, n_lines)``.
        ``bind``/``scenario_drives``/``scenario_offsets`` go to :func:`resolve_bindings`. ``expose`` limits the
        sensor channels the controller may read (default: all; exposing a sample block also exposes its count
        channel). Channel ids in the ABI come from the full channel set, so ``expose`` never renumbers a channel.
        ``line_grid_hz`` / ``ring_to_line`` are kept only if ``line_grid`` / ``ring_to_line`` are in ``knowledge``
        (07 P1-1, P1-12); so is the monitor ADC's code-to-ampere scale of the ``tx.monitor.samples`` /
        ``tx.monitor.latest`` channel specs (``ring.monitor.adc_full_scale_ua``) for ``monitor_nominal`` or
        ``model_nominal`` (without them those specs carry no scale: codes only, the full scale in codes is in
        ``adc``). ``nominal`` (the unsampled design) is kept as its knowledge-filtered view (07 D13).
        ``tick_jitter_s`` (07 P2-7) must be whole ticks and below the controller period.
        """
        hw = ChannelHardware() if hw is None else hw
        if schedule not in SCHEDULES:
            raise ValueError(f"controller_interface.schedule = {schedule!r}: must be one of {SCHEDULES}")
        if on_out_of_range not in ON_OUT_OF_RANGE:
            raise ValueError(f"controller_interface.on_out_of_range = {on_out_of_range!r}: one of {ON_OUT_OF_RANGE}")
        for nm, v in (("n_batch", n_batch), ("n_rings", n_rings), ("n_lines", n_lines)):
            if int(v) < 1:
                raise ValueError(f"{nm} must be >= 1, got {v}")
        table: list[tuple[str, float, str]] = [
            ("controller_interface.update_rate_hz", update_rate_hz, "hz"),
            ("controller_interface.tick_jitter_s", tick_jitter_s, "s"),
            ("ring.monitor.adc_rate_hz", hw.adc_rate_hz, "hz"),
            ("ring.monitor.adc_latency_s", hw.adc_latency_s, "s"),
            ("ring.heater.latency_s", hw.heater_latency_s, "s"),
            ("ring.bias_dac.latency_s", hw.bias_latency_s, "s"),
            ("datapath.error_count_period_s", hw.error_count_period_s, "s+"),
            ("datapath.error_count_latency_s", hw.error_count_latency_s, "s"),
        ]
        if hw.afe_type == "integrate_dump":     # the boxcar exists (and must be whole ticks) only for this AFE
            table.append(("ring.monitor.integration_window_s", hw.adc_integration_window_s, "s+"))
        if hw.heater_update_rate_hz is not None:
            table.append(("ring.heater.dac_max_update_hz", hw.heater_update_rate_hz, "hz"))
        if hw.bias_update_rate_hz is not None:
            table.append(("ring.bias_dac.update_rate_hz", hw.bias_update_rate_hz, "hz"))
        if hw.heater_readback_present:
            table += [("ring.heater.readback.rate_hz", hw.readback_rate_hz, "hz"),
                      ("ring.heater.readback.latency_s", hw.readback_latency_s, "s")]
        for i, ts in enumerate(hw.temp_sensors):
            table += [(f"temp_sensors[{i}].sample_rate_hz", ts.rate_hz, "hz"),
                      (f"temp_sensors[{i}].latency_s", ts.latency_s, "s")]
        t = check_tick_table(table, tick_s)
        period = t["controller_interface.update_rate_hz"]
        if not 0 <= int(phase_ticks) < period:
            raise TimingError(f"controller phase {phase_ticks} ticks must be in 0..{period - 1}")
        jitter = t["controller_interface.tick_jitter_s"]
        if jitter >= period:
            raise TimingError(f"controller_interface.tick_jitter_s = {jitter} ticks must be below the controller "
                              f"period ({period} ticks), or edges could reorder")
        # with jitter J, two actual edges can be up to period + J ticks apart (07 P2-7)
        m = samples_per_step(period + jitter, t["ring.monitor.adc_rate_hz"])
        all_ch = dict(build_v1_channels(hw, n_lines=n_lines) if channels is None else channels)
        know = tuple(knowledge)
        if not ({"monitor_nominal", "model_nominal"} & set(know)):
            # the monitor full scale (A per code) is monitor_nominal knowledge (mrm.control.calibration), not an
            # interface fact: without the token the ADC channels carry codes only
            for n in _ADC_SCALED_CHANNELS & set(all_ch):
                if all_ch[n].scale is not None:
                    all_ch[n] = dataclasses.replace(all_ch[n], scale=None, offset=None, phys_unit=None)
        bindings = resolve_bindings(all_ch, bind, scenario_drives=scenario_drives, scenario_offsets=scenario_offsets)
        expose_set = None if expose is None else set(expose)
        if expose_set is not None:
            unknown = expose_set - {n for n, c in all_ch.items() if c.kind == "sensor"}
            if unknown:
                raise ChannelError(f"controller_interface.expose: unknown sensor channels {sorted(unknown)}")
            # a sample block is meaningless without its valid-entry count: expose the count with the block
            expose_set |= {all_ch[n].count_channel for n in list(expose_set) if all_ch[n].agg == "block"}
        spec_ch = {n: c for n, c in all_ch.items()
                   if (c.kind == "sensor" and (expose_set is None or n in expose_set))
                   or (c.kind == "actuator" and bindings[n] == "controller")}
        dims = {"B": int(n_batch), "N": int(n_rings), "L": int(n_lines), "M": m, "S": len(hw.temp_sensors)}
        obs_ids, act_ids = assign_channel_ids(all_ch)     # ids of the FULL set: expose never renumbers a channel
        layout = make_abi_layout(spec_ch, dims, bindings, obs_ids=obs_ids, act_ids=act_ids)
        heater = DacFacts(bits=hw.heater_dac_bits, code_min=int(hw.heater_code_min), code_max=int(hw.heater_max),
                          latency_ticks=t["ring.heater.latency_s"],
                          update_period_ticks=t.get("ring.heater.dac_max_update_hz"))
        bias = DacFacts(bits=hw.bias_dac_bits, code_min=0, code_max=(1 << hw.bias_dac_bits) - 1,
                        latency_ticks=t["ring.bias_dac.latency_s"],
                        update_period_ticks=t.get("ring.bias_dac.update_rate_hz"),
                        phys_min=hw.bias_v_min_v, phys_max=hw.bias_v_max_v, phys_unit="V")
        adc = AdcFacts(bits=hw.adc_bits, full_scale_codes=(1 << hw.adc_bits) - 1, rate_hz=float(hw.adc_rate_hz),
                       period_ticks=t["ring.monitor.adc_rate_hz"], latency_ticks=t["ring.monitor.adc_latency_s"],
                       integration_window_ticks=t.get("ring.monitor.integration_window_s", 0), afe_type=hw.afe_type,
                       channels_per_adc=int(hw.adc_channels_per_adc), pga_gains=tuple(hw.pga_gains))
        temps = tuple(TempSensorFacts(name=ts.name, channel=f"temp.{ts.name}", bits=ts.bits, lsb_k=ts.lsb_k,
                                      offset_c=ts.offset_c, rate_hz=float(ts.rate_hz),
                                      period_ticks=t[f"temp_sensors[{i}].sample_rate_hz"],
                                      latency_ticks=t[f"temp_sensors[{i}].latency_s"], site=ts.site, type=ts.type)
                      for i, ts in enumerate(hw.temp_sensors))
        readback = None
        if hw.heater_readback_present:
            readback = (("adc_bits", hw.readback_adc_bits), ("v_full_scale_v", hw.readback_v_full_scale_v),
                        ("i_full_scale_a", hw.readback_i_full_scale_a),
                        ("period_ticks", t["ring.heater.readback.rate_hz"]),
                        ("latency_ticks", t["ring.heater.readback.latency_s"]))
        bus = tuple(range(int(n_rings))) if bus_order is None else tuple(int(x) for x in bus_order)
        if sorted(bus) != list(range(int(n_rings))):
            raise ValueError(f"bus_order {bus} must be a permutation of 0..{int(n_rings) - 1}")
        grid = tuple(float(x) for x in line_grid_hz) if (line_grid_hz is not None and "line_grid" in know) else None
        if grid is not None and len(grid) != int(n_lines):
            raise ValueError(f"line_grid_hz has {len(grid)} entries for L = {n_lines}")
        r2l = tuple(int(x) for x in ring_to_line) if (ring_to_line is not None and "ring_to_line" in know) else None
        if r2l is not None and (len(r2l) != int(n_rings) or not all(0 <= x < int(n_lines) for x in r2l)):
            raise ValueError(f"ring_to_line {r2l} must hold N = {n_rings} line indices in 0..{int(n_lines) - 1}")
        return cls(n_batch=int(n_batch), n_rings=int(n_rings), n_lines=int(n_lines), tick_s=float(tick_s),
                   period_ticks=period, phase_ticks=int(phase_ticks), samples_per_step=m, adc=adc, heater_dac=heater,
                   bias_dac=bias, temp_sensors=temps, channels=spec_ch, bindings=bindings, layout=layout,
                   bus_order=bus, line_grid_hz=grid, ring_to_line=r2l, knowledge=know, nominal=nominal,
                   schedule=schedule, on_out_of_range=on_out_of_range, heater_readback=readback,
                   error_count_period_ticks=t["datapath.error_count_period_s"], tick_jitter_ticks=jitter)

    @classmethod
    def from_config(cls, cfg, *, n_batch: int, nominal: object = None, scenario_drives: Iterable[str] = (),
                    scenario_offsets: Iterable[str] = (), expose: Iterable[str] | None = None,
                    line_grid_hz: Sequence[float] | None = None) -> "ControllerSpec":
        """Build from an ``mrm.params.Config`` (duck-typed: ``get``, ``si``). Reads sim.dt_plant_s, array.n_rings,
        the line count, controller_interface.{update_rate_hz, tick_jitter_s, bind.<channel>, expose, knowledge,
        schedule, on_out_of_range}, array.bus_order, channel_plan.ring_to_line (default: identity, as in realize),
        the nominal grid (:func:`nominal_line_grid_hz`, unless ``line_grid_hz`` is given) and the hardware keys of
        :meth:`ChannelHardware.from_config`. The grid and the assignment reach the spec only if ``line_grid`` /
        ``ring_to_line`` are in ``controller_interface.knowledge``. ``expose`` (default: the config's
        ``controller_interface.expose``; None there = every sensor channel) limits the sensor channels, as in
        :meth:`build`. ``nominal`` is the unsampled design (the Config); the spec keeps only its knowledge-filtered
        view (:class:`mrm.control.knowledge.NominalView`, 07 D13). ``scenario_drives`` / ``scenario_offsets`` come
        from the scenario (07 P0-7).
        """
        def val(key, default=None):
            v = cfg.get(key, None)
            v = v.get("value") if isinstance(v, Mapping) and "value" in v else v
            return default if v is None else v

        hw = ChannelHardware.from_config(cfg)
        n_rings = int(val("array.n_rings"))
        n_lines = getattr(cfg, "n_lines", None)
        if n_lines is None:
            lines = val("channel_plan.lines")
            n_lines = len(lines) if lines else int(val("channel_plan.n_lines", 1))
        bind = {}
        for ch in build_v1_channels(hw, n_lines=int(n_lines)):
            owner = val(f"controller_interface.bind.{ch}")
            if isinstance(owner, str):
                bind[ch] = owner
        if line_grid_hz is None:
            line_grid_hz = nominal_line_grid_hz(cfg, int(n_lines))
        has_si = callable(getattr(cfg, "si", None))
        rate = cfg.si("controller_interface.update_rate_hz", None) if has_si \
            else val("controller_interface.update_rate_hz")
        dt = cfg.si("sim.dt_plant_s", None) if has_si else val("sim.dt_plant_s")
        jitter = (cfg.si("controller_interface.tick_jitter_s", None) if has_si
                  else val("controller_interface.tick_jitter_s"))
        if rate is None or dt is None:
            raise ValueError("config needs sim.dt_plant_s and controller_interface.update_rate_hz")
        # channel_plan.ring_to_line defaults to the identity, as in mrm.params.realize (schema: "default: identity")
        r2l = val("channel_plan.ring_to_line")
        r2l = list(range(n_rings)) if r2l is None else r2l
        if expose is None:
            # controller_interface.expose (07 P0-7): None = every sensor channel; build() refuses unknown names
            expose = val("controller_interface.expose")
            if isinstance(expose, str):
                expose = [expose]
        return cls.build(n_batch=n_batch, n_rings=n_rings, n_lines=int(n_lines), tick_s=float(dt),
                         update_rate_hz=float(rate), hw=hw, bind=bind, scenario_drives=scenario_drives,
                         scenario_offsets=scenario_offsets, expose=expose, bus_order=val("array.bus_order"),
                         line_grid_hz=line_grid_hz, ring_to_line=r2l,
                         knowledge=tuple(val("controller_interface.knowledge", ())), nominal=nominal,
                         schedule=val("controller_interface.schedule", "simultaneous"),
                         on_out_of_range=val("controller_interface.on_out_of_range", "clip"),
                         tick_jitter_s=float(jitter or 0.0))

    def with_period(self, period_ticks: int, phase_ticks: int = 0) -> "ControllerSpec":
        """The same interface at another controller period (M and the layout follow; channel ids are kept). Used
        for composite children."""
        period_ticks = int(period_ticks)
        if period_ticks < 1:
            raise TimingError(f"controller period must be >= 1 tick, got {period_ticks}")
        if int(phase_ticks) < 0:
            raise TimingError(f"controller phase must be >= 0 ticks, got {phase_ticks}")
        if self.tick_jitter_ticks >= period_ticks:
            raise TimingError(f"tick jitter {self.tick_jitter_ticks} ticks must be below the period {period_ticks}")
        m = samples_per_step(period_ticks + self.tick_jitter_ticks, self.adc.period_ticks)
        dims = dict(self.dims, M=m)
        layout = make_abi_layout(self.channels, dims, self.bindings,
                                 obs_ids={f.name: f.ch_id for f in self.layout.obs},
                                 act_ids={f.name: f.ch_id for f in self.layout.act})
        return dataclasses.replace(self, period_ticks=period_ticks, phase_ticks=int(phase_ticks),
                                   samples_per_step=m, layout=layout)

    def to_dict(self) -> dict:
        """Canonical, JSON-serializable content (nominal via its own fingerprint/to_dict, else repr)."""
        d = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.name == "nominal":
                d[f.name] = _canon_nominal(v)
            elif f.name == "channels":
                d[f.name] = {k: dataclasses.asdict(c) for k, c in v.items()}
            elif f.name == "layout":
                d[f.name] = v.describe()
            elif dataclasses.is_dataclass(v):
                d[f.name] = dataclasses.asdict(v)
            elif f.name == "temp_sensors":
                d[f.name] = [dataclasses.asdict(x) for x in v]
            else:
                d[f.name] = v
        return d

    def fingerprint(self) -> str:
        """sha256 (16 hex) of :meth:`to_dict`. 07 P1-1 test_spec_independent_of_seed_and_uncertainty compares it."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=_json_default)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return repr(o)


def nominal_line_grid_hz(cfg, n_lines: int) -> list[float] | None:
    """Nominal laser grid (L,) in Hz, by the rule of ``mrm.params.realize.line_frequencies_hz`` (re-implemented here
    because mrm.control must not import realize, 07 P1-1): the ``channel_plan.lines`` table (``freq_thz``) when
    given, else ``center + (k - (L - 1) / 2) * spacing`` for k = 0..L-1. None when the config has neither.
    ``cfg`` is an mrm.params Config (``si``) or a plain mapping in config units."""
    from mrm.params.units import to_si

    def raw(key):
        v = cfg.get(key, None)
        return v.get("value") if isinstance(v, Mapping) and "value" in v else v

    lines = raw("channel_plan.lines")
    if lines:
        out = []
        for ln in lines:
            x = ln.get("freq_thz")
            x = x.get("value") if isinstance(x, Mapping) else x
            out.append(float(to_si("freq_thz", x)))
        return out
    fc, df = raw("channel_plan.center_thz"), raw("channel_plan.spacing_ghz")
    if fc is None or df is None:
        return None
    fc, df = float(to_si("channel_plan.center_thz", fc)), float(to_si("channel_plan.spacing_ghz", df))
    return [fc + (k - (n_lines - 1) / 2.0) * df for k in range(int(n_lines))]


def _canon_nominal(v):
    if v is None:
        return None
    if hasattr(v, "fingerprint") and callable(v.fingerprint):
        return {"fingerprint": v.fingerprint()}
    if hasattr(v, "to_dict") and callable(v.to_dict):
        return v.to_dict()
    if isinstance(v, Mapping):
        return dict(v)
    return repr(v)


# ---- Observation ---------------------------------------------------------------------------------------------------

class StrictChannels(Mapping):
    """Read-only channel mapping that raises :class:`UndeclaredChannelError` for an undeclared name.

    Arrays are exposed as non-writeable views, so a controller cannot modify the plant's buffers.
    """

    __slots__ = ("_d", "_what")

    def __init__(self, data: Mapping[str, Any], what: str = "sensor"):
        d = {}
        for k, v in data.items():
            a = np.asarray(v)
            if a.flags.writeable:
                a = a.view()
                a.flags.writeable = False
            d[k] = a
        self._d = d
        self._what = what

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            return self._d[name]
        except KeyError:
            raise UndeclaredChannelError(
                f"{self._what} channel {name!r} is not declared for this controller (declared: {sorted(self._d)})"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __repr__(self) -> str:
        return f"StrictChannels({sorted(self._d)})"


@dataclass(frozen=True)
class Observation:
    """What a controller receives at an edge (07 P0-6, P0-7).

    tick: int64 plant tick of the edge; dt_s: sim.dt_plant_s; sensors: channel -> (B, ...) array; new: channel ->
    new-sample flags of shape (B, ...) without the M axis (True if at least one new value since the last step);
    active: (N,) bool mask of the rings this step updates (round-robin schedule) or None for all rings.

    The arrays are read-only VIEWS of the caller's buffers and are valid during ``step`` only: a runner may reuse
    its buffers at the next edge, so a controller that keeps data across steps must copy it (``np.array(x)``).
    """
    tick: int
    dt_s: float
    sensors: Mapping
    new: Mapping = dataclasses.field(default_factory=dict)
    active: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "tick", int(self.tick))
        if not isinstance(self.sensors, StrictChannels):
            object.__setattr__(self, "sensors", StrictChannels(self.sensors, "sensor"))
        if not isinstance(self.new, StrictChannels):
            object.__setattr__(self, "new", StrictChannels(self.new, "new-sample flag of"))

    @property
    def t_s(self) -> float:
        """Derived time in seconds, tick * dt_s (never accumulated)."""
        return self.tick * self.dt_s

    def __getitem__(self, name: str) -> np.ndarray:
        return self.sensors[name]

    @property
    def active_ring(self) -> int:
        """Index of the single active ring under round_robin, or -1 when all rings are active."""
        if self.active is None:
            return -1
        idx = np.flatnonzero(self.active)
        return int(idx[0]) if idx.size == 1 else -1


# ---- protocol and base class ---------------------------------------------------------------------------------------

@runtime_checkable
class Controller(Protocol):
    """The controller protocol (07 P0-7, P1-19, P1-20).

    Required: ``name: str``, ``latency_ticks: int`` (digital pipeline depth in plant ticks, 07 P0-6),
    ``reset(spec) -> None`` and ``step(obs) -> dict[str, ndarray]``. Optional: ``telemetry() -> dict[str, (B, N)
    array]`` and ``fast_kernel`` (a numba function with the flat ABI, or None); read them with
    :func:`controller_telemetry` and :func:`controller_fast_kernel`.
    """
    name: str
    latency_ticks: int

    def reset(self, spec: ControllerSpec) -> None: ...

    def step(self, obs: Observation) -> dict[str, np.ndarray]: ...


def controller_telemetry(ctrl: Any) -> dict[str, np.ndarray]:
    """``ctrl.telemetry()`` if it exists, else {}."""
    fn = getattr(ctrl, "telemetry", None)
    return dict(fn()) if callable(fn) else {}


def controller_fast_kernel(ctrl: Any):
    """``ctrl.fast_kernel`` if it exists, else None."""
    return getattr(ctrl, "fast_kernel", None)


class BaseController:
    """Convenience base class (07 P1-19): a params dataclass filled from ``controller.params``, a no-op reset,
    state arrays allocated by symbolic shape, empty telemetry.

    Subclass it, declare ``Params`` as a dataclass, override ``reset`` (call ``super().reset(spec)``) and ``step``::

        class MyPI(BaseController):
            name = "my_pi"
            @dataclass
            class Params:
                kp: float = 0.1
            def reset(self, spec):
                super().reset(spec)
                self.integ = self.zeros("B", "N")
    """
    name: ClassVar[str] = "base"
    latency_ticks: int = 0
    fast_kernel = None

    @dataclass
    class Params:
        pass

    def __init__(self, params: Mapping[str, Any] | Any | None = None, **overrides: Any):
        self.params = self._make_params(params, overrides)
        self.spec: ControllerSpec | None = None

    @classmethod
    def _make_params(cls, params, overrides):
        pcls = cls.Params
        if params is not None and isinstance(params, pcls):
            return dataclasses.replace(params, **overrides) if overrides else params
        values = dict(params or {})
        values.update(overrides)
        names = {f.name for f in dataclasses.fields(pcls)}
        unknown = set(values) - names
        if unknown:
            raise TypeError(f"{cls.__name__}: unknown controller.params {sorted(unknown)}; allowed: {sorted(names)}")
        return pcls(**values)

    def reset(self, spec: ControllerSpec) -> None:
        self.spec = spec

    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        raise NotImplementedError(f"{type(self).__name__}.step")

    def telemetry(self) -> dict[str, np.ndarray]:
        return {}

    # -- helpers --
    def _require_spec(self) -> ControllerSpec:
        if self.spec is None:
            raise RuntimeError(f"{type(self).__name__}: call reset(spec) first")
        return self.spec

    def shape(self, *symbols) -> tuple[int, ...]:
        """Concrete shape of symbols such as ('B', 'N') from the spec."""
        dims = self._require_spec().dims
        return tuple(dims[s] if isinstance(s, str) else int(s) for s in symbols)

    def zeros(self, *symbols, dtype=np.float64) -> np.ndarray:
        return np.zeros(self.shape(*symbols), dtype=dtype)

    def full(self, *symbols, fill_value, dtype=np.float64) -> np.ndarray:
        return np.full(self.shape(*symbols), fill_value, dtype=dtype)


# ---- PerRing / PerDie adapters -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RingContext:
    """Given to a PerRing factory: the spec and the (b, n) this scalar controller instance serves."""
    spec: ControllerSpec
    b: int
    n: int


@dataclass(frozen=True)
class DieContext:
    """Given to a PerDie factory: the spec and the batch element b."""
    spec: ControllerSpec
    b: int


class _SliceView(Mapping):
    """Lazy per-ring / per-die slices of a StrictChannels mapping."""

    __slots__ = ("_src", "_spec", "_b", "_n", "_flags")

    def __init__(self, src: Mapping, spec: ControllerSpec, b: int, n: int | None, flags: bool):
        self._src, self._spec, self._b, self._n, self._flags = src, spec, b, n, flags

    def __getitem__(self, name: str):
        arr = self._src[name]                     # raises UndeclaredChannelError for undeclared names
        ch = self._spec.channels[name]
        shape = [s for s in ch.shape if not (self._flags and s == "M")]
        sel = [self._b]
        if self._n is not None:
            for s in shape[1:]:
                if s == "N":
                    sel.append(self._n)
                elif s == "M":
                    sel.append(slice(None))
                elif s == 1:
                    sel.append(0)
                else:
                    sel.append(slice(None))
        out = arr[tuple(sel)]
        if self._n is not None and not self._flags and ch.agg == "block":
            cnt = int(self._src[ch.count_channel][self._b, self._n])
            out = out[:cnt]
        return out

    def __iter__(self):
        return iter(self._src)

    def __len__(self):
        return len(self._src)


@dataclass(frozen=True)
class RingObservation:
    """One ring's view of an Observation: (B, N) channels become scalars, (B, N, M) blocks become the valid samples
    (length = count), (B, 1) become scalars, (B, L) become (L,) arrays. ``new`` follows the same rule."""
    tick: int
    dt_s: float
    b: int
    n: int
    sensors: Mapping
    new: Mapping

    @property
    def t_s(self) -> float:
        return self.tick * self.dt_s


@dataclass(frozen=True)
class DieObservation:
    """One batch element's view: every channel indexed by [b] ((N,), (N, M), (1,), (L,)); blocks are not trimmed,
    use the count channel. ``active`` is the (N,) round-robin mask or None."""
    tick: int
    dt_s: float
    b: int
    sensors: Mapping
    new: Mapping
    active: np.ndarray | None

    @property
    def t_s(self) -> float:
        return self.tick * self.dt_s


def _is_integral(v) -> bool:
    a = np.asarray(v)
    return a.dtype.kind in "biu"


class _HoldMixin:
    """Keeps the last command of each actuator channel, so partial writes hold the rest (07 P0-7 hold semantics)."""

    def _hold_init(self):
        self._hold: dict[str, np.ndarray] = {}

    def _hold_get(self, spec: ControllerSpec, obs: Observation, ch: str) -> np.ndarray:
        if ch not in self._hold:
            if ch not in spec.channels or spec.channels[ch].kind != "actuator":
                raise UndeclaredChannelError(f"{ch!r} is not an actuator bound to this controller "
                                             f"(bound: {sorted(spec.actuators)})")
            readback = ch + ".applied"
            if readback in spec.channels:
                self._hold[ch] = np.array(obs.sensors[readback], dtype=np.int64)
            else:
                self._hold[ch] = np.full(spec.shape_of(ch), spec.channels[ch].init, dtype=np.int64)
        return self._hold[ch]


class PerRing(BaseController, _HoldMixin):
    """Vectorize a scalar per-ring algorithm over (B, N) (07 P1-19).

    ``factory(ctx: RingContext)`` returns an object with ``step(robs: RingObservation) -> Mapping[str, int]`` (codes
    for (B, N) actuator channels) and optionally ``telemetry() -> Mapping[str, float]``. One instance per (b, n) is
    created at reset. Under a round-robin schedule only the active ring's instance is stepped. A ring that does not
    write a channel holds its previous command.
    """
    name = "per_ring"

    def __init__(self, factory: Callable[[RingContext], Any], *, name: str | None = None, latency_ticks: int = 0):
        super().__init__()
        self.factory = factory
        if name is not None:
            self.name = name
        self.latency_ticks = int(latency_ticks)
        self._inst: list[list[Any]] = []

    def reset(self, spec: ControllerSpec) -> None:
        super().reset(spec)
        self._hold_init()
        self._inst = [[self.factory(RingContext(spec, b, n)) for n in range(spec.n_rings)]
                      for b in range(spec.n_batch)]

    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        spec = self._require_spec()
        written: set[str] = set()
        for b in range(spec.n_batch):
            for n in range(spec.n_rings):
                if obs.active is not None and not obs.active[n]:
                    continue
                robs = RingObservation(obs.tick, obs.dt_s, b, n, _SliceView(obs.sensors, spec, b, n, False),
                                       _SliceView(obs.new, spec, b, n, True))
                out = self._inst[b][n].step(robs) or {}
                for ch, v in out.items():
                    hold = self._hold_get(spec, obs, ch)
                    if spec.channels[ch].shape != ("B", "N"):
                        raise ChannelError(f"PerRing can only write (B, N) channels; {ch} is {spec.channels[ch].shape}")
                    if np.ndim(v) != 0 or not _is_integral(v):
                        raise TypeError(f"PerRing ring ({b}, {n}) wrote {ch} = {v!r}; expected one integer code")
                    hold[b, n] = int(v)
                    written.add(ch)
        return {ch: self._hold[ch].copy() for ch in written}

    def telemetry(self) -> dict[str, np.ndarray]:
        spec = self._require_spec()
        out: dict[str, np.ndarray] = {}
        for b in range(spec.n_batch):
            for n in range(spec.n_rings):
                for k, v in controller_telemetry(self._inst[b][n]).items():
                    if k not in out:
                        out[k] = np.full((spec.n_batch, spec.n_rings), np.nan)
                    out[k][b, n] = float(v)
        return out


class PerDie(BaseController, _HoldMixin):
    """Vectorize a per-die algorithm over B (07 P1-19).

    ``factory(ctx: DieContext)`` returns an object with ``step(dobs: DieObservation) -> Mapping[str, array]`` whose
    arrays have the channel shape without B ((N,) or (L,)), and optionally ``telemetry() -> Mapping[str, (N,)]``.
    """
    name = "per_die"

    def __init__(self, factory: Callable[[DieContext], Any], *, name: str | None = None, latency_ticks: int = 0):
        super().__init__()
        self.factory = factory
        if name is not None:
            self.name = name
        self.latency_ticks = int(latency_ticks)
        self._inst: list[Any] = []

    def reset(self, spec: ControllerSpec) -> None:
        super().reset(spec)
        self._hold_init()
        self._inst = [self.factory(DieContext(spec, b)) for b in range(spec.n_batch)]

    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        spec = self._require_spec()
        written: set[str] = set()
        for b in range(spec.n_batch):
            dobs = DieObservation(obs.tick, obs.dt_s, b, _SliceView(obs.sensors, spec, b, None, False),
                                  _SliceView(obs.new, spec, b, None, True), obs.active)
            out = self._inst[b].step(dobs) or {}
            for ch, v in out.items():
                hold = self._hold_get(spec, obs, ch)
                a = np.asarray(v)
                if a.shape != hold.shape[1:]:
                    raise ChannelError(f"PerDie element {b} wrote {ch} with shape {a.shape}; expected {hold.shape[1:]}")
                if not _is_integral(a):
                    raise TypeError(f"PerDie element {b} wrote {ch} with dtype {a.dtype}; expected integer codes")
                hold[b] = a
                written.add(ch)
        return {ch: self._hold[ch].copy() for ch in written}

    def telemetry(self) -> dict[str, np.ndarray]:
        spec = self._require_spec()
        out: dict[str, np.ndarray] = {}
        for b in range(spec.n_batch):
            for k, v in controller_telemetry(self._inst[b]).items():
                if k not in out:
                    out[k] = np.full((spec.n_batch, spec.n_rings), np.nan)
                out[k][b] = np.asarray(v, dtype=np.float64)
        return out


# ---- action validation and telemetry -------------------------------------------------------------------------------

class ActionValidator:
    """Shape, dtype and range checks of a controller's actions on every step (07 P1-19).

    ``validate(actions)`` returns new arrays cast to each channel's dtype. Out-of-range values are clipped and
    counted (``on_out_of_range = clip``) or raise :class:`ActionRangeError` (``error``). Float arrays are rejected:
    channels carry integer codes, and silent truncation would hide quantization from the controller designer.
    """

    def __init__(self, spec: ControllerSpec, on_out_of_range: str | None = None):
        self.spec = spec
        self.mode = spec.on_out_of_range if on_out_of_range is None else on_out_of_range
        if self.mode not in ON_OUT_OF_RANGE:
            raise ValueError(f"on_out_of_range = {self.mode!r}: must be one of {ON_OUT_OF_RANGE}")
        self.counts = {n: np.zeros(spec.shape_of(n), dtype=np.int64) for n in spec.actuators}

    def reset(self) -> None:
        for c in self.counts.values():
            c[...] = 0

    def saturation_counts(self) -> dict[str, np.ndarray]:
        """Per channel, per element: number of steps in which the command was clipped."""
        return {n: c.copy() for n, c in self.counts.items()}

    def validate(self, actions: Mapping[str, Any]) -> dict[str, np.ndarray]:
        if not isinstance(actions, Mapping):
            raise TypeError(f"controller step() must return a dict of channel -> array (use {{}} to hold every "
                            f"actuator), got {type(actions).__name__}")
        spec = self.spec
        out: dict[str, np.ndarray] = {}
        for name, val in actions.items():
            if name not in spec.bindings:
                if name in spec.channels:
                    raise ChannelError(f"controller wrote {name!r}, which is a sensor channel")
                raise UndeclaredChannelError(f"controller wrote unknown channel {name!r}; actuators bound to the "
                                             f"controller: {sorted(spec.actuators)}")
            if spec.bindings[name] != "controller":
                raise BindingError(f"controller wrote {name!r}, which is bound to the {spec.bindings[name]} "
                                   f"(controller_interface.bind)")
            ch = spec.channels[name]
            arr = np.asarray(val)
            want = spec.shape_of(name)
            if arr.shape != want:
                raise ChannelError(f"{name}: expected shape {want} {ch.shape}, got {arr.shape}")
            if arr.dtype.kind not in "biu":
                raise TypeError(f"{name}: expected integer codes ({ch.dtype}), got dtype {arr.dtype}; round "
                                f"explicitly, e.g. np.rint(x).astype(np.int64)")
            if arr.dtype == np.uint64:
                arr = np.minimum(arr, np.uint64(INT64_MAX))
            a64 = arr.astype(np.int64)
            bad = (a64 < ch.min) | (a64 > ch.max)
            if bad.any():
                if self.mode == "error":
                    idx = tuple(int(i) for i in np.argwhere(bad)[0])
                    raise ActionRangeError(f"{name}{list(idx)} = {int(a64[idx])} outside [{ch.min}, {ch.max}] "
                                           f"({ch.unit}); set controller_interface.on_out_of_range: clip to "
                                           f"saturate instead")
                self.counts[name] += bad
                a64 = np.clip(a64, ch.min, ch.max)
            out[name] = a64.astype(ch.np_dtype)
        return out


def merge_round_robin(spec: ControllerSpec, actions: Mapping[str, Any], held: np.ndarray,
                      active_ring: int) -> Mapping[str, Any]:
    """What the engine applies of a step's actions under ``schedule = round_robin`` (07 P2-7): every per-ring
    (B, N) actuator channel keeps only the ``active_ring`` element; the other rings take the held command of the edge
    (``held``: the (B, act_size) int64 act vector before the step). Other channels pass unchanged, and so does a
    channel the validator will refuse (unknown, a sensor, wrong shape or dtype). ``active_ring`` < 0: ``actions``
    unchanged. The kernel applies the same rule (mrm.sim.kernel._rr_apply_a); the Python path merges before
    :class:`ActionValidator`, so only applied commands are range-checked and counted, as in the kernel."""
    if active_ring < 0 or not isinstance(actions, Mapping):
        return actions
    out = dict(actions)
    others = np.ones(spec.n_rings, dtype=bool)
    others[int(active_ring)] = False
    for name, v in actions.items():
        ch = spec.channels.get(name)
        if ch is None or ch.kind != "actuator" or tuple(ch.shape) != ("B", "N"):
            continue
        a = np.asarray(v)
        if a.shape != spec.shape_of(name) or a.dtype.kind not in "biu":
            continue
        a = a.astype(np.int64) if a.dtype != np.uint64 else np.minimum(a, np.uint64(INT64_MAX)).astype(np.int64)
        f = spec.layout.act_field(name)
        a[:, others] = np.asarray(held)[:, f.offset:f.offset + f.size][:, others]
        out[name] = a
    return out


TELEMETRY_KEYS: dict[str, str] = {
    "state": "int: state-machine state (controller-defined enumeration)",
    "locked": "bool: the controller declares lock (scored against Truth: missed loss of lock, false alarms)",
    "detuning_est_pm": "float: estimated ring-minus-laser detuning, pm",
    "integrator": "float: integrator value, in the controller's own units",
    "integrator_saturated": "bool: the integrator sits at a clamp (integrator saturation time, 07 P1-20)",
}


def validate_telemetry(tel: Mapping[str, Any], spec: ControllerSpec) -> dict[str, np.ndarray]:
    """Check telemetry (07 P1-20): every value numeric or bool with shape (B, N). Returns float64 copies.

    Standard keys are in :data:`TELEMETRY_KEYS`; other keys are allowed and recorded as is.
    """
    out = {}
    want = (spec.n_batch, spec.n_rings)
    for k, v in tel.items():
        a = np.asarray(v)
        if a.dtype.kind not in "biuf":
            raise TypeError(f"telemetry {k!r}: dtype {a.dtype} is not numeric")
        if a.shape != want:
            raise ChannelError(f"telemetry {k!r}: expected shape (B, N) = {want}, got {a.shape}")
        out[k] = a.astype(np.float64)
    return out


# ---- composite controller (07 P2-7) --------------------------------------------------------------------------------

class _Accumulator:
    """Combines base-period observations into one child observation by each channel's ``agg`` rule."""

    def __init__(self, spec_child: ControllerSpec):
        self.spec = spec_child
        self.m = spec_child.samples_per_step
        self.vals: dict[str, np.ndarray] = {}
        self.new: dict[str, np.ndarray] = {}
        self.blocks = {n: c for n, c in spec_child.sensors.items() if c.agg == "block"}
        self.count_channels = {c.count_channel for c in self.blocks.values()}
        self.started = False

    def _reset_window(self):
        for n, c in self.spec.sensors.items():
            if c.agg == "block":
                self.vals[n][...] = 0
            elif c.agg in ("sum", "count") and n not in self.count_channels:
                self.vals[n][...] = 0
            elif c.agg == "all":
                self.vals[n][...] = True
            elif c.agg == "any":
                self.vals[n][...] = False
        for cnt in self.count_channels:
            self.vals[cnt][...] = 0
        for f in self.new.values():
            f[...] = False

    def add(self, obs: Observation) -> None:
        spec = self.spec
        if not self.started:
            for n, c in spec.sensors.items():
                self.vals[n] = np.zeros(c.resolve_shape(spec.dims), dtype=c.np_dtype)
                self.new[n] = np.zeros((spec.n_batch,) + c.flag_shape(spec.dims), dtype=bool)
            self.started = True
            self._reset_window()
        counts_in = {cnt: np.asarray(obs.sensors[cnt]) for cnt in self.count_channels}
        for n, c in spec.sensors.items():
            x = np.asarray(obs.sensors[n])
            if c.agg == "block":
                continue
            if n in self.count_channels:
                continue
            if c.agg == "latest":
                self.vals[n][...] = x
            elif c.agg in ("sum", "count"):
                self.vals[n] += x.astype(self.vals[n].dtype)
            elif c.agg == "all":
                self.vals[n] &= x.astype(bool)
            elif c.agg == "any":
                self.vals[n] |= x.astype(bool)
            self.new[n] |= np.asarray(obs.new[n], dtype=bool)
        for cnt in self.count_channels:
            members = [n for n, c in self.blocks.items() if c.count_channel == cnt]
            k_in = counts_in[cnt]
            k_acc = self.vals[cnt]
            for b in range(spec.n_batch):
                for i in range(spec.n_rings):
                    k = int(k_in[b, i])
                    if k <= 0:
                        continue
                    have = int(k_acc[b, i])
                    for n in members:
                        src = np.asarray(obs.sensors[n])[b, i, :k]
                        joined = np.concatenate([self.vals[n][b, i, :have], src])[-self.m:]
                        self.vals[n][b, i, :joined.size] = joined
                    k_acc[b, i] = min(have + k, self.m)
            for n in members + [cnt]:
                self.new[n] |= np.asarray(obs.new[n], dtype=bool)

    def emit(self, tick: int, dt_s: float, active: np.ndarray | None) -> Observation:
        obs = Observation(tick, dt_s, {n: v.copy() for n, v in self.vals.items()},
                          {n: f.copy() for n, f in self.new.items()}, active)
        self._reset_window()
        return obs


class CompositeController(_HoldMixin):
    """Several controllers at their own rates in one controller (07 P2-7).

    ``CompositeController([(fast_lock, 100e3), (supervisor, 1e3)], schedule="simultaneous")``. Each child entry is
    ``(controller, rate_hz)`` or ``(controller, rate_hz, phase_ticks)``. The runner calls the composite at
    :meth:`base_period_ticks` (the gcd of the child periods; ``spec.period_ticks`` must divide every child period and
    phase). A child is stepped at ticks where ``(tick - phase) % period == 0`` and sees all samples since its own last
    step (channels combined by their ``agg`` rule). When several children write one channel in the same base step,
    the later child in the list wins.

    ``schedule = round_robin`` time-multiplexes one datapath across the rings: at its j-th edge a child gets
    ``obs.active`` with only ring ``j % N`` set, and only that ring's (B, N) commands are applied; (B, L) channels
    are applied whole. A child's spec carries the composite's ``schedule``. The composite itself needs the interface
    schedule ``simultaneous`` (``controller_interface.schedule: round_robin`` would multiplex its output a second
    time: :meth:`reset` refuses it; use this ``schedule`` instead). ``latency_ticks`` is the largest child latency
    (equal pipelines are the common case; a shorter child pipeline is modelled conservatively). Compiled composition
    is not provided: a composite runs on the Python path.
    """

    def __init__(self, children: Sequence[tuple], schedule: str = "simultaneous", name: str = "composite"):
        if schedule not in SCHEDULES:
            raise ValueError(f"schedule {schedule!r} must be one of {SCHEDULES}")
        if not children:
            raise ValueError("CompositeController needs at least one child")
        self.children = []
        for i, entry in enumerate(children):
            if len(entry) not in (2, 3):
                raise ValueError(f"child {i}: expected (controller, rate_hz[, phase_ticks]), got {entry!r}")
            ctrl, rate = entry[0], float(entry[1])
            phase = int(entry[2]) if len(entry) == 3 else 0
            if not isinstance(ctrl, Controller):
                raise TypeError(f"child {i} ({ctrl!r}) does not implement the Controller protocol")
            if controller_fast_kernel(ctrl) is not None:
                # the plant would never call it (a composite runs on the Python path and has no fast_kernel), so
                # a dither written as a fast_kernel would silently vanish
                raise TypeError(f"child {i} ({getattr(ctrl, 'name', ctrl)!r}) declares a fast_kernel, which a "
                                f"CompositeController cannot run; run that controller alone")
            self.children.append((ctrl, rate, phase))
        self.schedule = schedule
        self.name = name
        self.latency_ticks = max(int(c.latency_ticks) for c, _, _ in self.children)
        self.fast_kernel = None
        self.spec: ControllerSpec | None = None
        self._child_specs: list[ControllerSpec] = []
        self._acc: list[_Accumulator] = []

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "CompositeController":
        """From ``controller.params``: {children: [{name, rate_hz, params?, phase_ticks?}], schedule?}."""
        from mrm.control import registry
        kids = []
        for i, c in enumerate(params.get("children", ())):
            if "name" not in c or "rate_hz" not in c:
                raise ValueError(f"controller.params.children[{i}] needs name and rate_hz")
            ctrl = registry.create(c["name"], c.get("params"))
            kids.append((ctrl, float(c["rate_hz"]), int(c.get("phase_ticks", 0))))
        return cls(kids, schedule=params.get("schedule", "simultaneous"), name=params.get("name", "composite"))

    def child_periods(self, tick_s: float) -> list[int]:
        return [rate_to_period_ticks(r, tick_s, f"controller.params.children[{i}].rate_hz")
                for i, (_, r, _) in enumerate(self.children)]

    def base_period_ticks(self, tick_s: float) -> int:
        """gcd of the child periods (and phases): the rate at which the runner must call the composite."""
        g = 0
        for p, (_, _, ph) in zip(self.child_periods(tick_s), self.children):
            g = math.gcd(g, p)
            g = math.gcd(g, ph) if ph else g
        return g

    def reset(self, spec: ControllerSpec) -> None:
        if spec.schedule != "simultaneous":
            raise ValueError(f"CompositeController: controller_interface.schedule = {spec.schedule!r}; the engine "
                             f"would multiplex the composite's output over the rings on top of its own schedule "
                             f"({self.schedule!r}). Set the interface schedule to simultaneous and the composite's "
                             f"schedule (controller.params.schedule) to round_robin")
        self.spec = spec
        self._hold_init()
        periods = self.child_periods(spec.tick_s)
        errs = []
        for i, (p, (_, _, ph)) in enumerate(zip(periods, self.children)):
            if p % spec.period_ticks or (ph - spec.phase_ticks) % spec.period_ticks:
                errs.append(f"child {i}: period {p} / phase {ph} ticks is not on the composite's base grid of "
                            f"{spec.period_ticks} ticks (set controller_interface.update_rate_hz to "
                            f"1 / ({self.base_period_ticks(spec.tick_s)} * dt))")
        if errs:
            raise TimingError("; ".join(errs))
        self._child_specs = [dataclasses.replace(spec.with_period(p, ph), schedule=self.schedule)
                             for p, (_, _, ph) in zip(periods, self.children)]
        self._acc = [_Accumulator(cs) for cs in self._child_specs]
        for (ctrl, _, _), cs in zip(self.children, self._child_specs):
            ctrl.reset(cs)

    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        spec = self.spec
        if spec is None:
            raise RuntimeError("CompositeController: call reset(spec) first")
        written: set[str] = set()
        # the nominal base edge of this call: the actual tick may be late by up to tick_jitter_ticks (07 P2-7)
        j = int(edge_index(obs.tick, spec.period_ticks, spec.phase_ticks))
        nominal = spec.phase_ticks + j * spec.period_ticks
        if j < 0 or obs.tick - nominal > spec.tick_jitter_ticks:
            raise TimingError(f"CompositeController called at tick {obs.tick}, which is not on its base grid "
                              f"(period {spec.period_ticks}, phase {spec.phase_ticks}, jitter <= "
                              f"{spec.tick_jitter_ticks} ticks)")
        for (ctrl, _, _), cs, acc in zip(self.children, self._child_specs, self._acc):
            acc.add(obs)
            if not is_edge(nominal, cs.period_ticks, cs.phase_ticks):
                continue
            active = None
            k = -1
            if self.schedule == "round_robin":
                k = int(edge_index(nominal, cs.period_ticks, cs.phase_ticks)) % spec.n_rings
                active = np.zeros(spec.n_rings, dtype=bool)
                active[k] = True
            out = ctrl.step(acc.emit(obs.tick, obs.dt_s, active)) or {}
            for ch, v in out.items():
                hold = self._hold_get(spec, obs, ch)
                a = np.asarray(v)
                if a.shape != hold.shape:
                    raise ChannelError(f"child {ctrl.name!r} wrote {ch} with shape {a.shape}; expected {hold.shape}")
                if a.dtype.kind not in "biu":
                    raise TypeError(f"child {ctrl.name!r} wrote {ch} with dtype {a.dtype}; expected integer codes")
                if k >= 0 and spec.channels[ch].shape[:2] == ("B", "N"):
                    hold[:, k] = a[:, k]
                else:
                    hold[...] = a
                written.add(ch)
        return {ch: self._hold[ch].copy() for ch in written}

    def telemetry(self) -> dict[str, np.ndarray]:
        out = {}
        names = [c.name for c, _, _ in self.children]
        for i, (ctrl, _, _) in enumerate(self.children):
            prefix = ctrl.name if names.count(ctrl.name) == 1 else f"{ctrl.name}{i}"
            for k, v in controller_telemetry(ctrl).items():
                out[f"{prefix}.{k}"] = v
        return out


# ---- compiled-controller ABI (07 P0-9) -----------------------------------------------------------------------------
# spec_i64 header (indices). Values never change; append only.
SPEC_I_ABI_VERSION = 0
SPEC_I_N_BATCH = 1
SPEC_I_N_RINGS = 2
SPEC_I_N_LINES = 3
SPEC_I_M = 4
SPEC_I_PERIOD_TICKS = 5
SPEC_I_LATENCY_TICKS = 6
SPEC_I_OBS_SIZE = 7
SPEC_I_ACT_SIZE = 8
SPEC_I_N_OBS_IDS = 9
SPEC_I_N_ACT_IDS = 10
SPEC_I_OBS_TABLE = 11      # value = start index of the obs offset table (n_obs_ids entries, -1 = absent)
SPEC_I_NEW_TABLE = 12      # value = start index of the new-flag offset table
SPEC_I_ACT_TABLE = 13      # value = start index of the act offset table (n_act_ids entries)
SPEC_I_USER = 14           # value = start index of the controller's integer parameters
SPEC_I_N_USER = 15
SPEC_I_N_USER_F = 16
SPEC_I_ADC_PERIOD_TICKS = 17
SPEC_I_ADC_LATENCY_TICKS = 18
SPEC_I_ADC_BITS = 19
SPEC_I_HEATER_BITS = 20
SPEC_I_HEATER_MIN = 21
SPEC_I_HEATER_MAX = 22
SPEC_I_BIAS_BITS = 23
SPEC_I_N_TEMP = 24
SPEC_I_SCHEDULE = 25       # 0 simultaneous, 1 round_robin
SPEC_I_PHASE_TICKS = 26
SPEC_I_BIAS_MAX = 27             # largest bias DAC code (2**bits - 1)
SPEC_I_HEATER_LATENCY_TICKS = 28 # heater DAC latency owned by the plant (07 P0-6), for model-based controllers
SPEC_I_BIAS_LATENCY_TICKS = 29
SPEC_I_ADC_WINDOW_TICKS = 30     # integrate-and-dump window (0 for an rc AFE)
SPEC_I_TICK_JITTER_TICKS = 31    # largest controller-edge delay (07 P2-7)
SPEC_I_HEADER = 32
# spec_f64 header
SPEC_F_TICK_S = 0
SPEC_F_PERIOD_S = 1
SPEC_F_ADC_RATE_HZ = 2
SPEC_F_HEADER = 8          # the controller's float parameters start here
# fast-kernel vectors
FO_TICK = 0
FO_N = 1
FO_HEADER = 2


def spec_arrays(spec: ControllerSpec, params_i64: Sequence[int] = (), params_f64: Sequence[float] = (),
                latency_ticks: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Build (spec_i64, spec_f64) for ``init`` from a spec and the controller's own parameters."""
    lay = spec.layout
    obs_t, new_t, act_t = lay.obs_table(), lay.new_table(), lay.act_table()
    pi = np.asarray(params_i64, dtype=np.int64).ravel()
    pf = np.asarray(params_f64, dtype=np.float64).ravel()
    start_obs = SPEC_I_HEADER
    start_new = start_obs + obs_t.size
    start_act = start_new + new_t.size
    start_user = start_act + act_t.size
    si = np.zeros(start_user + pi.size, dtype=np.int64)
    si[SPEC_I_ABI_VERSION] = lay.version
    si[SPEC_I_N_BATCH] = spec.n_batch
    si[SPEC_I_N_RINGS] = spec.n_rings
    si[SPEC_I_N_LINES] = spec.n_lines
    si[SPEC_I_M] = spec.samples_per_step
    si[SPEC_I_PERIOD_TICKS] = spec.period_ticks
    si[SPEC_I_LATENCY_TICKS] = int(latency_ticks)
    si[SPEC_I_OBS_SIZE] = lay.obs_size
    si[SPEC_I_ACT_SIZE] = lay.act_size
    si[SPEC_I_N_OBS_IDS] = lay.n_obs_ids
    si[SPEC_I_N_ACT_IDS] = lay.n_act_ids
    si[SPEC_I_OBS_TABLE] = start_obs
    si[SPEC_I_NEW_TABLE] = start_new
    si[SPEC_I_ACT_TABLE] = start_act
    si[SPEC_I_USER] = start_user
    si[SPEC_I_N_USER] = pi.size
    si[SPEC_I_N_USER_F] = pf.size
    si[SPEC_I_ADC_PERIOD_TICKS] = spec.adc.period_ticks
    si[SPEC_I_ADC_LATENCY_TICKS] = spec.adc.latency_ticks
    si[SPEC_I_ADC_BITS] = spec.adc.bits
    si[SPEC_I_HEATER_BITS] = spec.heater_dac.bits
    si[SPEC_I_HEATER_MIN] = spec.heater_dac.code_min
    si[SPEC_I_HEATER_MAX] = spec.heater_dac.code_max
    si[SPEC_I_BIAS_BITS] = spec.bias_dac.bits
    si[SPEC_I_N_TEMP] = len(spec.temp_sensors)
    si[SPEC_I_SCHEDULE] = SCHEDULES.index(spec.schedule)
    si[SPEC_I_PHASE_TICKS] = spec.phase_ticks
    si[SPEC_I_BIAS_MAX] = spec.bias_dac.code_max
    si[SPEC_I_HEATER_LATENCY_TICKS] = spec.heater_dac.latency_ticks
    si[SPEC_I_BIAS_LATENCY_TICKS] = spec.bias_dac.latency_ticks
    si[SPEC_I_ADC_WINDOW_TICKS] = spec.adc.integration_window_ticks
    si[SPEC_I_TICK_JITTER_TICKS] = spec.tick_jitter_ticks
    si[start_obs:start_new] = obs_t
    si[start_new:start_act] = new_t
    si[start_act:start_user] = act_t
    si[start_user:] = pi
    sf = np.zeros(SPEC_F_HEADER + pf.size, dtype=np.float64)
    sf[SPEC_F_TICK_S] = spec.tick_s
    sf[SPEC_F_PERIOD_S] = spec.period_s
    sf[SPEC_F_ADC_RATE_HZ] = spec.adc.rate_hz
    sf[SPEC_F_HEADER:] = pf
    return si, sf


@nb.njit(cache=True)
def obs_offset(spec_i64, ch_id):
    """Offset of sensor channel ``ch_id`` (ObsId / TEMP_ID_BASE + s) in obs_i64, or -1 if absent."""
    if ch_id < 0 or ch_id >= spec_i64[SPEC_I_N_OBS_IDS]:
        return np.int64(-1)
    return spec_i64[spec_i64[SPEC_I_OBS_TABLE] + ch_id]


@nb.njit(cache=True)
def new_offset(spec_i64, ch_id):
    """Offset of the new-sample flags of sensor channel ``ch_id`` in obs_i64, or -1 if absent."""
    if ch_id < 0 or ch_id >= spec_i64[SPEC_I_N_OBS_IDS]:
        return np.int64(-1)
    return spec_i64[spec_i64[SPEC_I_NEW_TABLE] + ch_id]


@nb.njit(cache=True)
def act_offset(spec_i64, ch_id):
    """Offset of actuator channel ``ch_id`` (ActId) in act_i64, or -1 if absent / not bound to the controller."""
    if ch_id < 0 or ch_id >= spec_i64[SPEC_I_N_ACT_IDS]:
        return np.int64(-1)
    return spec_i64[spec_i64[SPEC_I_ACT_TABLE] + ch_id]


@nb.njit(cache=True)
def user_i64(spec_i64, k):
    """The controller's k-th integer parameter."""
    return spec_i64[spec_i64[SPEC_I_USER] + k]


@nb.njit(cache=True)
def user_f64(spec_f64, k):
    """The controller's k-th float parameter."""
    return spec_f64[SPEC_F_HEADER + k]


def fast_obs_size(n_rings: int) -> int:
    """Length of the fast-kernel observation vector: [tick, N, new[N], code[N], sample_tick[N]]."""
    return FO_HEADER + 3 * int(n_rings)


def fast_act_size(n_rings: int) -> int:
    """Length of the fast-kernel action vector: [heater offset[N], bias offset[N]] (codes)."""
    return 2 * int(n_rings)


@nb.njit(cache=True)
def fo_new(i, n):
    return FO_HEADER + i


@nb.njit(cache=True)
def fo_code(i, n):
    return FO_HEADER + n + i


@nb.njit(cache=True)
def fo_sample_tick(i, n):
    return FO_HEADER + 2 * n + i


@nb.njit(cache=True)
def fa_heater(i, n):
    return i


@nb.njit(cache=True)
def fa_bias(i, n):
    return n + i


@nb.njit
def call_step_batch(step_fn, p_i64, p_f64, s_i64, s_f64, obs_i64, act_i64):
    """Call a compiled ``step`` for every batch element (row) of the 2-D state and I/O arrays."""
    for b in range(obs_i64.shape[0]):
        step_fn(p_i64[b], p_f64[b], s_i64[b], s_f64[b], obs_i64[b], act_i64[b])


def _as_1d(a, dtype, what: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 1 or arr.dtype != dtype:
        raise TypeError(f"compiled init must return 1-D {np.dtype(dtype).name} arrays; {what} is "
                        f"{arr.dtype} with shape {arr.shape}")
    return arr


class CompiledController:
    """Container for a compiled controller (07 P0-9) that also implements the Python protocol.

    Example (a per-ring integer integrator; the full version is in tests/verification/test_control_api.py)::

        LATEST, HEATER = int(ObsId.TX_MONITOR_LATEST), int(ActId.TX_HEATER)

        @nb.njit
        def init(si, sf):
            p = np.zeros(4, np.int64)
            p[0] = si[SPEC_I_N_RINGS]; p[1] = obs_offset(si, LATEST); p[2] = act_offset(si, HEATER)
            p[3] = user_i64(si, 0)                                  # set-point in ADC codes
            return p, np.zeros(0), np.zeros(p[0], np.int64), np.zeros(0)

        @nb.njit
        def step(p, pf, s, sf, obs, act):
            for i in range(p[0]):
                s[i] += p[3] - obs[p[1] + i]
                act[p[2] + i] = min(max(s[i] >> 4, 0), 65535)

        ctrl = CompiledController("my_i", init, step, params_i64=[30000])

    ``init_fn(spec_i64, spec_f64) -> (p_i64, p_f64, s_i64, s_f64)`` and
    ``step_fn(p_i64, p_f64, s_i64, s_f64, obs_i64, act_i64)`` are numba functions (or plain Python functions with
    the same behaviour under NUMBA_DISABLE_JIT). ``params_i64`` / ``params_f64`` are appended to the spec arrays.
    After ``reset(spec)``, ``state`` holds (p_i64, p_f64, s_i64, s_f64) as (B, size) arrays (row b = element b),
    ``spec_i64``/``spec_f64`` the spec arrays and ``act`` the held command vector (B, act_size); the kernel uses
    them directly. ``step(obs)`` runs the same functions on the Python path and returns int64 arrays (validate them
    with :class:`ActionValidator`, which clips, counts and casts). Before the first step the held command
    is taken from the ``<channel>.applied`` readbacks when they are exposed (the latched DAC registers, e.g. a warm
    start), else from each channel's ``init``; the kernel must pre-fill act_i64 the same way.
    """

    def __init__(self, name: str, init_fn: Callable, step_fn: Callable, *, params_i64: Sequence[int] = (),
                 params_f64: Sequence[float] = (), latency_ticks: int = 0, fast_fn: Callable | None = None):
        self.name = name
        self.init_fn = init_fn
        self.step_fn = step_fn
        self.fast_fn = fast_fn
        self.params_i64 = np.asarray(params_i64, dtype=np.int64).ravel()
        self.params_f64 = np.asarray(params_f64, dtype=np.float64).ravel()
        self.latency_ticks = int(latency_ticks)
        self.spec: ControllerSpec | None = None
        self.spec_i64 = self.spec_f64 = None
        self.state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        self.act: np.ndarray | None = None
        self._act_synced = False

    @property
    def fast_kernel(self):
        return self.fast_fn

    def initial_act(self, spec: ControllerSpec) -> np.ndarray:
        """(B, act_size) int64 filled with each controller-bound actuator's init value."""
        act = np.zeros((spec.n_batch, spec.layout.act_size), dtype=np.int64)
        for f in spec.layout.act:
            act[:, f.offset:f.offset + f.size] = spec.channels[f.name].init
        return act

    def reset(self, spec: ControllerSpec) -> None:
        self.spec = spec
        self.spec_i64, self.spec_f64 = spec_arrays(spec, self.params_i64, self.params_f64, self.latency_ticks)
        res = self.init_fn(self.spec_i64, self.spec_f64)
        if not isinstance(res, tuple) or len(res) != 4:
            raise TypeError(f"{self.name}: init must return (p_i64, p_f64, s_i64, s_f64)")
        p_i, p_f, s_i, s_f = (_as_1d(res[0], np.int64, "p_i64"), _as_1d(res[1], np.float64, "p_f64"),
                              _as_1d(res[2], np.int64, "s_i64"), _as_1d(res[3], np.float64, "s_f64"))
        b = spec.n_batch
        self.state = (np.tile(p_i, (b, 1)), np.tile(p_f, (b, 1)), np.tile(s_i, (b, 1)), np.tile(s_f, (b, 1)))
        self.act = self.initial_act(spec)
        self._act_synced = False

    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        spec = self.spec
        if spec is None or self.state is None:
            raise RuntimeError(f"{self.name}: call reset(spec) first")
        if not self._act_synced:            # start from the latched DAC registers when they are observable
            for f in spec.layout.act:
                rb = f.name + ".applied"
                if rb in obs.sensors:
                    self.act[:, f.offset:f.offset + f.size] = np.asarray(obs.sensors[rb]).reshape(spec.n_batch, -1)
            self._act_synced = True
        obs_i64 = spec.layout.pack_obs(obs.tick, obs.sensors, obs.new, obs.active_ring)
        p_i, p_f, s_i, s_f = self.state
        if isinstance(self.step_fn, Dispatcher):
            call_step_batch(self.step_fn, p_i, p_f, s_i, s_f, obs_i64, self.act)
        else:                                  # a plain Python step function (or NUMBA_DISABLE_JIT=1)
            for b in range(spec.n_batch):
                self.step_fn(p_i[b], p_f[b], s_i[b], s_f[b], obs_i64[b], self.act[b])
        # int64, not the channel dtype: a cast would wrap an out-of-range command (-5 -> 65531 for a uint16 heater)
        # before ActionValidator could clip and count it
        return spec.layout.unpack_act(self.act, cast=False)
