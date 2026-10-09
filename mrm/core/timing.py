"""Integer-tick time, latency FIFOs and the order of events at a controller edge (07 P0-6).

Time is an int64 count of plant ticks of ``sim.dt_plant_s`` (default 1 us, 0.1 us allowed; ARCHITECTURE.md section
3). Float time drifts: ``t += 1e-6`` in float64 is 1.9 ms off after 1e10 steps (07 P0-6, measured by [CTRL]), which
on the 30.4 h soak drops or duplicates ADC samples. Every hardware period and latency is therefore converted to
ticks once, at load time, by :func:`seconds_to_ticks` / :func:`rate_to_period_ticks`, and a value that is not a whole
number of ticks is rejected with the config key named (07 P0-6: ADC period, controller period, conversion latency,
DAC latency, integration window). ``t_s`` exists only as a derived quantity (``tick * dt_s``).

One owner per delay (07 P0-6)
-----------------------------
- The plant's SensorBank / ActuatorBank (``mrm/plant/io.py``) own the analog and converter delays:
  ``ring.monitor.adc_latency_s``, ``ring.monitor.integration_window_s``, ``ring.heater.latency_s``,
  ``slew_codes_per_s``, ``settle_tau_s`` and the bias-DAC equivalents.
- The controller owns only its digital pipeline depth, ``Controller.latency_ticks``.
- The runner only schedules ticks.

Order of events at a controller edge
------------------------------------
A controller edge is a tick k with ``(k - phase) % period == 0`` (:func:`is_edge`); with
``controller_interface.tick_jitter_s`` (07 P2-7) edge j runs instead at its nominal tick plus
:func:`edge_jitter` (0..J ticks, J < period), and everything below happens at that actual tick. Tick k is the instant
t = k * dt at the END of plant interval [k-1, k)::

      plant interval [k-1, k)                          | edge at tick k
    --------------------------------------------------+-------------------------------------------------------
      substeps: optics, self-heating, thermal ROM,     |
      AFE: integrate-and-dump running sums             |
      (a sample with sample tick s covers (s - W, s],  |
       W = integration window in ticks, divided once)  |
                                                       | (1) plant has finished its last substep: state at k
                                                       |     the ADC samples with sample tick k are taken
                                                       | (2) SensorBank releases every sample whose
                                                       |     sample_tick + adc_latency_ticks <= k
                                                       |     (fast_kernel, if declared, runs once per
                                                       |      released sample, in sample-tick order)
                                                       | (3) controller.step(obs), obs.tick == k
                                                       | (4) command enters the controller pipeline and
                                                       |     latches in the DAC register at
                                                       |     k_latch = k + latency_ticks
                                                       |     (applied codes = this latched register value)
                                                       | (5) DAC latency FIFO (dac_latency_ticks), then the
                                                       |     slew limit, then the settle filter:
                                                       |     the analog value first acts on the plant over
                                                       |     [k_act, k_act + 1), k_act = k_latch + dac_latency
    --------------------------------------------------+-------------------------------------------------------
      plant interval [k, k+1) uses the actuator values whose k_act <= k (zero-order hold within the tick)

Consequences, used by the timing tests of the plant (07 P0-6 ``test_loop_delay_impulse``):
- with all latencies 0, a command computed at edge k acts on interval [k, k+1), and the first ADC sample that can
  see it has sample tick k + 1; :func:`first_response_tick` gives the tick at which a controller first observes it;
- "applied codes" are the latched DAC register (step 4), never the slewed analog value;
- at one tick, a FIFO is always pushed before it is popped (sample taken in step 1, released in step 2; command
  latched in step 4, applied in step 5), which is what :func:`fifo_capacity` assumes.

Latency FIFOs
-------------
Timed ring buffers, one independent queue per element (ring, sensor, ...), numba-compatible: three arrays
(``vals (W, C)``, ``rel (W, C)`` release ticks, ``meta (W, 2)`` = [head, count]) and scalar indices. Entries leave in
push order once their release tick is <= now. With a constant latency per channel, order is preserved.
"""
from __future__ import annotations

import enum
import math
from collections.abc import Iterable, Mapping

import numba as nb
import numpy as np

from mrm.core import rng as _rng

__all__ = [
    "TimingError", "EventPhase", "seconds_to_ticks", "rate_to_period_ticks", "ticks_to_seconds", "check_tick_table",
    "is_edge", "next_edge", "edge_index", "first_response_tick", "edge_jitter",
    "fifo_alloc", "fifo_capacity", "fifo_push", "fifo_pop_ready", "fifo_peek", "fifo_count", "fifo_clear",
    "fifo_pop_all_ready",
    "TICK_ATOL", "MAX_TICKS",
]

TICK_ATOL = 1e-9   # largest |value/dt - round(value/dt)| accepted as "a whole number of ticks" (design choice)
MAX_TICKS = 2**62  # largest tick count accepted: keeps tick + latency sums inside an int64 (design choice)


class TimingError(ValueError):
    """A period or latency is not a whole number of plant ticks (or is negative / not finite)."""


class EventPhase(enum.IntEnum):
    """The fixed order of events at a controller edge (module docstring)."""
    PLANT_SUBSTEPS_DONE = 1
    SENSOR_RELEASE = 2
    CONTROLLER_STEP = 3
    COMMAND_LATCH = 4
    DAC_PIPELINE = 5


def _check_dt(dt_s: float) -> float:
    dt = float(dt_s)
    if not (math.isfinite(dt) and dt > 0.0):
        raise TimingError(f"sim.dt_plant_s must be a positive finite number of seconds, got {dt_s!r}")
    return dt


def seconds_to_ticks(value_s: float, dt_s: float, key: str, *, allow_zero: bool = True) -> int:
    """Convert a duration in seconds to a whole number of plant ticks, or raise TimingError naming ``key``.

    Accepts |value_s/dt_s - n| <= TICK_ATOL + 4 * eps * |value_s/dt_s| for an integer n >= 0 (or >= 1 if
    ``allow_zero`` is False). Example: 2e-6 s at dt = 1e-6 s -> 2; 2.5e-6 s at dt = 1e-6 s -> TimingError.
    """
    dt = _check_dt(dt_s)
    try:
        v = float(value_s)
    except (TypeError, ValueError):
        raise TimingError(f"{key}: expected a duration in seconds, got {value_s!r}") from None
    if not math.isfinite(v) or v < 0.0:
        raise TimingError(f"{key} = {value_s!r} s: must be finite and >= 0")
    r = v / dt
    if r > MAX_TICKS:
        raise TimingError(f"{key} = {v!r} s is {r:.3g} ticks of {dt!r} s, beyond the int64 tick range (2**62)")
    n = round(r)
    if abs(r - n) > TICK_ATOL + 4.0 * np.finfo(np.float64).eps * abs(r):
        raise TimingError(
            f"{key} = {v!r} s is {r:.9g} plant ticks of sim.dt_plant_s = {dt!r} s, not a whole number; "
            f"use {math.floor(r) * dt!r} s or {math.ceil(r) * dt!r} s, or change sim.dt_plant_s")
    if n == 0 and not allow_zero:
        raise TimingError(f"{key} = {v!r} s rounds to 0 ticks of {dt!r} s; it must be at least one tick")
    return int(n)


def rate_to_period_ticks(rate_hz: float, dt_s: float, key: str) -> int:
    """Convert a rate in Hz to its period in whole plant ticks (>= 1), or raise TimingError naming ``key``.

    Example: 100 kHz at dt = 1 us -> 10 ticks; 3 MHz at dt = 1 us -> TimingError (0.333 ticks).
    """
    try:
        f = float(rate_hz)
    except (TypeError, ValueError):
        raise TimingError(f"{key}: expected a rate in Hz, got {rate_hz!r}") from None
    if not (math.isfinite(f) and f > 0.0):
        raise TimingError(f"{key} = {rate_hz!r} Hz: must be a positive finite rate")
    return seconds_to_ticks(1.0 / f, dt_s, f"{key} (period 1/{f!r} Hz)", allow_zero=False)


def ticks_to_seconds(ticks, dt_s: float):
    """Seconds of ``ticks`` (int or int array): ticks * dt_s, as float64. Derived quantity only."""
    return np.asarray(ticks, dtype=np.float64) * _check_dt(dt_s) if np.ndim(ticks) else float(ticks) * _check_dt(dt_s)


def check_tick_table(entries: Iterable[tuple[str, float, str]] | Mapping[str, tuple[float, str]], dt_s: float
                     ) -> dict[str, int]:
    """Convert many (key, value, kind) entries at once; kind is 's' (duration, may be 0), 's+' (duration >= 1 tick)
    or 'hz' (rate). Raises ONE TimingError that lists every offending key, so a config is fixed in one pass.
    """
    items = entries.items() if isinstance(entries, Mapping) else ((k, (v, kind)) for k, v, kind in entries)
    out: dict[str, int] = {}
    errors: list[str] = []
    for key, (value, kind) in items:
        try:
            if kind == "hz":
                out[key] = rate_to_period_ticks(value, dt_s, key)
            elif kind in ("s", "s+"):
                out[key] = seconds_to_ticks(value, dt_s, key, allow_zero=(kind == "s"))
            else:
                raise TimingError(f"{key}: unknown kind {kind!r} (use 's', 's+' or 'hz')")
        except TimingError as e:
            errors.append(str(e))
    if errors:
        raise TimingError("timing check failed:\n  " + "\n  ".join(errors))
    return out


# ---- edges (numba) -----------------------------------------------------------------------------------------------

@nb.njit(cache=True)
def is_edge(tick, period, phase):
    """True if ``tick`` is a controller (or ADC) edge: (tick - phase) % period == 0 and tick >= phase."""
    return tick >= phase and (tick - phase) % period == 0


@nb.njit(cache=True)
def next_edge(tick, period, phase):
    """Smallest edge tick >= ``tick``."""
    if tick <= phase:
        return phase
    r = (tick - phase) % period
    return tick if r == 0 else tick + (period - r)


@nb.njit(cache=True)
def edge_index(tick, period, phase):
    """Index j of the edge tick = phase + j * period (floor for ticks between edges; -1 before the first)."""
    if tick < phase:
        return -1
    return (tick - phase) // period


@nb.njit(cache=True)
def first_response_tick(k, ctrl_latency, dac_latency, adc_period, adc_phase, adc_latency, ctrl_period, ctrl_phase):
    """Tick at which a controller first OBSERVES a command it computed at edge k (event order above).

    The command acts from k_act = k + ctrl_latency + dac_latency; the first ADC sample that sees it has the first
    sample tick s >= k_act + 1 on the ADC grid (adc_phase + j * adc_period); it is released at s + adc_latency and
    observed at the first controller edge >= that tick. Assumes a settle filter that responds within one tick and
    no slew limit; the plant tests add their own analog delays.
    """
    k_act = k + ctrl_latency + dac_latency
    s = next_edge(k_act + 1, adc_period, adc_phase)
    return next_edge(s + adc_latency, ctrl_period, ctrl_phase)


@nb.njit(cache=True)
def edge_jitter(key, edge_idx, max_jitter_ticks):
    """Delay d_j in 0..max_jitter_ticks (whole ticks) of controller edge j (07 P2-7 ``tick_jitter_s``).

    The controller step of nominal edge e_j = phase + j * period runs at tick e_j + d_j: steps (2)-(4) of the
    event order move with it, and ``obs.tick`` is the actual tick. d_j = floor(u * (J + 1)) with u the counter-RNG
    uniform of (key, ring 0, tick j, slot 0), so it is a pure function of (seed, edge index): reproducible, batch-
    and thread-invariant. ``key = rng.stream_key(seed_b, rng.Stream.SCHEDULE)``. Jitter only delays (never
    advances) an edge, and ``ControllerSpec`` requires J < period, so edges stay in order. edge_idx >= 0.
    """
    if max_jitter_ticks <= 0:
        return np.int64(0)
    u = _rng.uniform(key, 0, edge_idx, 0)
    return np.int64(u * (max_jitter_ticks + 1))


# ---- latency FIFOs (numba) ---------------------------------------------------------------------------------------

def fifo_capacity(latency_ticks: int, push_period_ticks: int, pop_period_ticks: int = 1) -> int:
    """Entries in flight at most for one element, pushed every ``push_period_ticks`` with a constant latency and
    drained (every ready entry, :func:`fifo_pop_all_ready`) every ``pop_period_ticks``; push before pop at a tick.

    An entry pushed at t0 leaves at the first pop tick >= t0 + latency, i.e. at most t0 + latency + pop - 1, so
    just before a pop at tick t the queue holds the pushes with t0 in [t - latency - pop + 1, t]:
    ``(latency + pop_period - 1) // push_period + 1``. With pops at every tick (the default) this is
    ``latency // push_period + 1``. Tight for the worst phase between the push and pop grids: a FIFO with one entry
    fewer overflows (tested by brute force over phases). A SensorBank that drains its latency FIFO only at controller
    edges must pass the controller period as ``pop_period_ticks``.
    """
    lat, per, pop = int(latency_ticks), int(push_period_ticks), int(pop_period_ticks)
    if lat < 0 or per < 1 or pop < 1:
        raise TimingError(f"fifo_capacity: latency {lat} must be >= 0, push period {per} and pop period {pop} "
                          f">= 1 tick")
    return (lat + pop - 1) // per + 1


def fifo_alloc(width: int, capacity: int, dtype=np.int64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Allocate ``width`` independent FIFOs of ``capacity`` entries: (vals (W, C) dtype, rel (W, C) int64,
    meta (W, 2) int64 = [head, count])."""
    if width < 1 or capacity < 1:
        raise ValueError(f"fifo_alloc: width {width} and capacity {capacity} must be >= 1")
    return (np.zeros((width, capacity), dtype=dtype), np.zeros((width, capacity), dtype=np.int64),
            np.zeros((width, 2), dtype=np.int64))


@nb.njit(cache=True)
def fifo_push(vals, rel, meta, i, value, release_tick):
    """Append (value, release_tick) to queue i. Returns False (and drops the entry) if the queue is full."""
    cap = vals.shape[1]
    cnt = meta[i, 1]
    if cnt >= cap:
        return False
    tail = (meta[i, 0] + cnt) % cap
    vals[i, tail] = value
    rel[i, tail] = release_tick
    meta[i, 1] = cnt + 1
    return True


@nb.njit(cache=True)
def fifo_count(meta, i):
    """Entries in queue i."""
    return meta[i, 1]


@nb.njit(cache=True)
def fifo_peek(vals, rel, meta, i):
    """(has_entry, value, release_tick) of the oldest entry of queue i, without removing it."""
    h = meta[i, 0]
    return meta[i, 1] > 0, vals[i, h], rel[i, h]


@nb.njit(cache=True)
def fifo_pop_ready(vals, rel, meta, i, now):
    """Pop the oldest entry of queue i if its release tick <= now. Returns (popped, value, release_tick)."""
    h = meta[i, 0]
    if meta[i, 1] == 0 or rel[i, h] > now:
        return False, vals[i, h], rel[i, h]
    v = vals[i, h]
    t = rel[i, h]
    meta[i, 0] = (h + 1) % vals.shape[1]
    meta[i, 1] -= 1
    return True, v, t


@nb.njit(cache=True)
def fifo_pop_all_ready(vals, rel, meta, i, now, out_vals, out_rel):
    """Pop every ready entry of queue i into out_vals / out_rel (in order). Returns how many were popped.

    Stops early, leaving entries queued, when the output buffers are full.
    """
    n = 0
    while n < out_vals.shape[0]:
        ok, v, t = fifo_pop_ready(vals, rel, meta, i, now)
        if not ok:
            break
        out_vals[n] = v
        out_rel[n] = t
        n += 1
    return n


@nb.njit(cache=True)
def fifo_clear(meta):
    """Empty every queue (head = count = 0)."""
    for i in range(meta.shape[0]):
        meta[i, 0] = 0
        meta[i, 1] = 0
