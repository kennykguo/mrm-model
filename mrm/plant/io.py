"""Plant I/O: the channel <-> plant mapping, the controller pipeline and the sensor release (07 P0-6, P0-7).

Owner: engine. 07 P0-6 gives each delay one owner. The analog and converter delays (ADC window and latency, DAC
latency, slew, settle) live in the leaf modules (mrm.plant.sensing, mrm.plant.datapath, mrm.plant.electrical); this
module owns what is left between them and a controller:

- the ActuatorBank: the controller's own digital pipeline (``Controller.latency_ticks``): a command computed at edge
  k latches in the DAC register at k + latency_ticks (07 P0-6 step 4), one entry per controller edge; the clip of
  every actuator to its channel range [min, max] BEFORE any cast (docs/handoff/core_control_api.md item 1);
  the registers of the actuators that are not DACs (tx.assignment, datapath.bitstat_threshold, monitor.pga_select,
  laser.tune);
- the SensorBank: which bank feeds which sensor channel, with the 07 P0-7 new-sample flag of every channel
  (``latest`` channels hold their value, ``sum`` channels report only what happened since the last release);
- the flat ABI maps of :class:`ChannelMaps`: the compiled kernel packs obs_i64 and unpacks act_i64 through them, in
  the AbiLayout of mrm.core.channels, so both controller paths see the same numbers.

Compiled helpers (@njit(cache=True), arrays only, run under NUMBA_DISABLE_JIT=1): :func:`pack_obs_row`,
:func:`clip_act_row`, :func:`pipe_push_row`, :func:`pipe_pop_row`, :func:`apply_act_row`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from mrm.core.channels import OBS_ACTIVE_RING, OBS_TICK
from mrm.core.timing import fifo_capacity

__all__ = [
    "SRC_NAMES", "DST_NAMES", "ChannelMaps", "pack_obs_row", "clip_act_row", "pipe_push_row", "pipe_pop_row",
    "apply_act_row", "pipe_capacity", "ActuatorBank", "SensorBank",
]

# sensor sources (obs_map column 0)
SRC_MON_SAMPLES = 0
SRC_MON_TICKS = 1
SRC_MON_COUNT = 2
SRC_MON_LATEST = 3
SRC_MON_VALID = 4
SRC_ADC_OVER = 5
SRC_TEMP = 6            # aux = sensor index
SRC_RB_V = 7
SRC_RB_I = 8
SRC_ONES = 9
SRC_BITS = 10
SRC_BITSTAT = 11
SRC_ERR = 12
SRC_ERR_VALID = 13
SRC_HEATER_APPLIED = 14
SRC_BIAS_APPLIED = 15
SRC_NAMES = {"tx.monitor.samples": SRC_MON_SAMPLES, "tx.monitor.sample_ticks": SRC_MON_TICKS,
             "tx.monitor.count": SRC_MON_COUNT, "tx.monitor.latest": SRC_MON_LATEST, "monitor_valid": SRC_MON_VALID,
             "adc_overrange": SRC_ADC_OVER, "heater.readback_v": SRC_RB_V, "heater.readback_i": SRC_RB_I,
             "tx.ones_count": SRC_ONES, "tx.bits": SRC_BITS, "datapath.bitstat_above": SRC_BITSTAT,
             "rx.error_count": SRC_ERR, "rx.error_count_valid": SRC_ERR_VALID,
             "tx.heater.applied": SRC_HEATER_APPLIED, "tx.bias.applied": SRC_BIAS_APPLIED}
# actuator destinations (act_map column 0)
DST_HEATER = 0
DST_BIAS = 1
DST_ASSIGN = 2
DST_BS_THR = 3
DST_PGA = 4
DST_TUNE = 5
DST_NAMES = {"tx.heater": DST_HEATER, "tx.bias": DST_BIAS, "tx.assignment": DST_ASSIGN,
             "datapath.bitstat_threshold": DST_BS_THR, "monitor.pga_select": DST_PGA, "laser.tune": DST_TUNE}


@dataclass(frozen=True)
class ChannelMaps:
    """Kernel tables of one ControllerSpec layout.

    obs_map (F, 6) int64: [source SRC_*, value offset, value size, new-flag offset, new-flag size, aux] for every
    sensor channel the controller may read (``spec.layout.obs``); act_map (A, 6) int64: [destination DST_*, offset,
    size, min, max, per_ring] for every actuator bound to the controller (per_ring 1 for a (B, N) channel: under
    ``controller_interface.schedule: round_robin`` only the active ring's element is applied, 07 P2-7); obs_size,
    act_size; act_init (act_size,) the channel init values.
    """
    obs_map: np.ndarray
    act_map: np.ndarray
    obs_size: int
    act_size: int
    act_init: np.ndarray
    m: int

    @classmethod
    def from_spec(cls, spec) -> "ChannelMaps":
        lay = spec.layout
        temp_idx = {ts.channel: i for i, ts in enumerate(spec.temp_sensors)}
        rows = []
        for f in lay.obs:
            nf = lay.new_field(f.name)
            if f.name in SRC_NAMES:
                src, aux = SRC_NAMES[f.name], 0
            elif f.name in temp_idx:
                src, aux = SRC_TEMP, temp_idx[f.name]
            else:
                raise ValueError(f"sensor channel {f.name!r} has no plant source in mrm.plant.io (optional channels "
                                 "such as the reference tap need an io entry first)")
            rows.append((src, f.offset, f.size, nf.offset, nf.size, aux))
        arows = []
        init = np.zeros(lay.act_size, np.int64)
        for f in lay.act:
            ch = spec.channels[f.name]
            if f.name not in DST_NAMES:
                raise ValueError(f"actuator channel {f.name!r} has no plant destination in mrm.plant.io")
            arows.append((DST_NAMES[f.name], f.offset, f.size, int(ch.min), int(ch.max),
                          int(tuple(ch.shape[:2]) == ("B", "N") and len(ch.shape) == 2)))
            init[f.offset:f.offset + f.size] = int(ch.init)
        return cls(obs_map=np.array(rows, np.int64).reshape(-1, 6), act_map=np.array(arows, np.int64).reshape(-1, 6),
                   obs_size=int(lay.obs_size), act_size=int(lay.act_size), act_init=init,
                   m=int(spec.samples_per_step))


def pipe_capacity(latency_ticks: int, period_ticks: int) -> int:
    """Entries of the controller pipeline: one push per edge, pops every tick (mrm.core.timing.fifo_capacity)."""
    return fifo_capacity(int(latency_ticks), max(int(period_ticks), 1), 1)


@njit(cache=True)
def pack_obs_row(obs, obs_map, tick, active_ring, mon_samples, mon_ticks, mon_count, mon_latest, mon_valid, mon_over,
                 mon_new, temp_latest, temp_new, rb_v, rb_i, rb_new, ones, bits, bs_above, bs_new, err_latest,
                 err_valid, heater_reg, bias_reg):
    """Write one world's obs_i64 (AbiLayout of mrm.core.channels) from the released sensor values (07 P0-7).

    New-sample flags follow mrm.plant.sensing / mrm.plant.datapath: samples, sample_ticks and latest are new when a
    sample arrived; count, monitor_valid, adc_overrange, ones, bits, error_count_valid and the applied registers are
    always new; a temperature or readback channel when it was delivered; bitstat_above when its window ran;
    rx.error_count when a count was delivered. Bools are 0/1.
    """
    obs[OBS_TICK] = tick
    obs[OBS_ACTIVE_RING] = active_ring
    n_r = mon_count.shape[0]
    m = mon_samples.shape[1]
    for i in range(obs_map.shape[0]):
        src = obs_map[i, 0]
        vo = obs_map[i, 1]
        vs = obs_map[i, 2]
        no = obs_map[i, 3]
        ns = obs_map[i, 4]
        aux = obs_map[i, 5]
        if src == SRC_MON_SAMPLES or src == SRC_MON_TICKS:
            for n in range(n_r):
                for j in range(m):
                    obs[vo + n * m + j] = mon_samples[n, j] if src == SRC_MON_SAMPLES else mon_ticks[n, j]
                obs[no + n] = 1 if mon_new[n] else 0
        elif src == SRC_TEMP:
            obs[vo] = temp_latest[aux]
            obs[no] = 1 if temp_new[aux] else 0
        else:
            for n in range(vs):
                if src == SRC_MON_COUNT:
                    v, f = mon_count[n], 1
                elif src == SRC_MON_LATEST:
                    v, f = mon_latest[n], 1 if mon_new[n] else 0
                elif src == SRC_MON_VALID:
                    v, f = 1 if mon_valid[n] else 0, 1
                elif src == SRC_ADC_OVER:
                    v, f = 1 if mon_over[n] else 0, 1
                elif src == SRC_RB_V:
                    v, f = rb_v[n], 1 if rb_new[n] else 0
                elif src == SRC_RB_I:
                    v, f = rb_i[n], 1 if rb_new[n] else 0
                elif src == SRC_ONES:
                    v, f = ones[n], 1
                elif src == SRC_BITS:
                    v, f = bits[n], 1
                elif src == SRC_BITSTAT:
                    v, f = bs_above[n], 1 if bs_new else 0
                elif src == SRC_ERR:
                    v, f = err_latest[n], 1 if err_valid[n] else 0
                elif src == SRC_ERR_VALID:
                    v, f = 1 if err_valid[n] else 0, 1
                elif src == SRC_HEATER_APPLIED:
                    v, f = heater_reg[n], 1
                else:
                    v, f = bias_reg[n], 1
                obs[vo + n] = v
                if n < ns:
                    obs[no + n] = f


@njit(cache=True)
def clip_act_row(act, act_map, sat):
    """Clip every actuator value of one world's act_i64 to its channel [min, max] in place (never a cast first:
    -5 must not wrap to 65531, docs/handoff/core_control_api.md item 1) and count the clipped elements in sat
    (act_size,) (07 P1-19 saturation counts). Returns the number clipped."""
    total = 0
    for i in range(act_map.shape[0]):
        o = act_map[i, 1]
        lo = act_map[i, 3]
        hi = act_map[i, 4]
        for j in range(act_map[i, 2]):
            v = act[o + j]
            if v < lo:
                act[o + j] = lo
                sat[o + j] += 1
                total += 1
            elif v > hi:
                act[o + j] = hi
                sat[o + j] += 1
                total += 1
    return total


@njit(cache=True)
def pipe_push_row(vals, ticks, meta, act, latch_tick):
    """Queue one command vector (act_size,) to latch at latch_tick (07 P0-6 step 4). vals (D, act_size), ticks (D,),
    meta (2,) = [head, count]. Returns False when the pipeline is full (capacity from :func:`pipe_capacity`)."""
    cap = vals.shape[0]
    if meta[1] >= cap:
        return False
    tail = (meta[0] + meta[1]) % cap
    for j in range(act.shape[0]):
        vals[tail, j] = act[j]
    ticks[tail] = latch_tick
    meta[1] += 1
    return True


@njit(cache=True)
def pipe_pop_row(vals, ticks, meta, now, out):
    """Pop every queued command whose latch tick <= now into out (act_size,) (the newest wins). Returns True when
    at least one command latched at this tick."""
    hit = False
    cap = vals.shape[0]
    while meta[1] > 0:
        h = meta[0]
        if ticks[h] > now:
            break
        for j in range(out.shape[0]):
            out[j] = vals[h, j]
        meta[0] = (h + 1) % cap
        meta[1] -= 1
        hit = True
    return hit


@njit(cache=True)
def apply_act_row(act, act_map, cmd_heater, valid_heater, cmd_bias, valid_bias, assign, bs_thr, pga, tune):
    """Route one latched command vector to the plant: the heater and bias DAC commands (valid flags set), and the
    registers of tx.assignment, datapath.bitstat_threshold, monitor.pga_select and laser.tune (held until the next
    command)."""
    for i in range(act_map.shape[0]):
        dst = act_map[i, 0]
        o = act_map[i, 1]
        for j in range(act_map[i, 2]):
            v = act[o + j]
            if dst == DST_HEATER:
                cmd_heater[j] = v
                valid_heater[j] = True
            elif dst == DST_BIAS:
                cmd_bias[j] = v
                valid_bias[j] = True
            elif dst == DST_ASSIGN:
                assign[j] = v
            elif dst == DST_BS_THR:
                bs_thr[j] = v
            elif dst == DST_PGA:
                pga[j] = v
            else:
                tune[j] = v


class ActuatorBank:
    """Python form of the controller pipeline and the non-DAC actuator registers for B worlds (the NumPy reference
    engine uses it; the compiled kernel uses the same njit helpers on its own arrays).

    ``push(edge_tick, act)`` queues a validated (B, act_size) command vector to latch at edge + latency_ticks;
    ``pop(tick)`` returns (heater_cmd, heater_valid, bias_cmd, bias_valid, popped (B,)) for the plant interval
    [tick, tick + 1) and updates ``assign``, ``bs_thr``, ``pga``, ``tune``.
    """

    def __init__(self, maps: ChannelMaps, batch: int, n_rings: int, n_lines: int, latency_ticks: int,
                 period_ticks: int, *, assign0=None, init_heater=0, init_bias=0):
        self.maps = maps
        B, N, L = int(batch), int(n_rings), int(n_lines)
        self.B, self.N, self.L = B, N, L
        self.latency_ticks = int(latency_ticks)
        cap = pipe_capacity(latency_ticks, period_ticks)
        self.vals = np.zeros((B, cap, max(maps.act_size, 1)), np.int64)
        self.ticks = np.zeros((B, cap), np.int64)
        self.meta = np.zeros((B, 2), np.int64)
        self.act = np.tile(maps.act_init, (B, 1)) if maps.act_size else np.zeros((B, 1), np.int64)
        self.sat = np.zeros_like(self.act)
        self.assign = (np.full((B, N), -1, np.int64) if assign0 is None
                       else np.broadcast_to(np.asarray(assign0, np.int64), (B, N)).copy())
        self.bs_thr = np.zeros((B, N), np.int64)
        self.pga = np.zeros((B, N), np.int64)
        self.tune = np.zeros((B, L), np.int64)
        for i in range(maps.act_map.shape[0]):           # registers start at the channel init values
            dst, o, sz = (int(x) for x in maps.act_map[i, :3])
            if dst == DST_HEATER:
                self.act[:, o:o + sz] = init_heater
            elif dst == DST_BIAS:
                self.act[:, o:o + sz] = init_bias
        self._scratch = np.zeros_like(self.act[0])

    def push(self, edge_tick: int, act: np.ndarray) -> None:
        act = np.asarray(act, np.int64).reshape(self.B, -1)
        for b in range(self.B):
            if not pipe_push_row(self.vals[b], self.ticks[b], self.meta[b], act[b],
                                 np.int64(edge_tick + self.latency_ticks)):
                raise RuntimeError("controller pipeline overflow (more edges in flight than its capacity)")

    def pop(self, tick: int):
        B, N = self.B, self.N
        hc = np.zeros((B, N), np.int64)
        hv = np.zeros((B, N), np.bool_)
        bc = np.zeros((B, N), np.int64)
        bv = np.zeros((B, N), np.bool_)
        popped = np.zeros(B, np.bool_)
        for b in range(B):
            if pipe_pop_row(self.vals[b], self.ticks[b], self.meta[b], np.int64(tick), self._scratch):
                popped[b] = True
                apply_act_row(self._scratch, self.maps.act_map, hc[b], hv[b], bc[b], bv[b], self.assign[b],
                              self.bs_thr[b], self.pga[b], self.tune[b])
        return hc, hv, bc, bv, popped


class SensorBank:
    """The sensor side of the plant for the Python reference engine: monitor, temperature sensors, heater readback
    and the data path, released as the 07 P0-7 channel dictionaries (with their new-sample flags)."""

    def __init__(self, monitor, temps=None, readback=None, datapath=None):
        self.monitor, self.temps, self.readback, self.datapath = monitor, temps, readback, datapath

    def release(self, now: int, *, heater_applied, bias_applied, bitstat=None) -> tuple[dict, dict]:
        """Every channel at controller edge ``now`` (07 P0-6 step 2). ``bitstat``: a callable run before the data
        path release (the bit-statistics window of the edge), or None."""
        sensors, new = self.monitor.channels(self.monitor.release(now))
        for bank in (self.temps, self.readback):
            if bank is not None:
                s, nw = bank.release(now)
                sensors.update(s)
                new.update(nw)
        if self.datapath is not None:
            if bitstat is not None:
                bitstat()
            s, nw = self.datapath.release(now)
            sensors.update(s)
            new.update(nw)
        B, N = heater_applied.shape
        sensors["tx.heater.applied"] = np.asarray(heater_applied)
        sensors["tx.bias.applied"] = np.asarray(bias_applied)
        new["tx.heater.applied"] = np.ones((B, N), bool)
        new["tx.bias.applied"] = np.ones((B, N), bool)
        return sensors, new
