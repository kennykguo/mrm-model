"""Streaming metric accumulators, record specs (channel x decimation x reducer) and summaries (07 P0-8).

Contract: docs/research/07_design_critique.md P0-8 (normative), D9 (relOMA = OMA / P_in), P1-12 (wrong line against
the declared assignment), P1-11 (demanded peak slew and code range used), P0-1 item 7 (best achievable OMA over
open-loop-stable points, the unrestricted value next to it). Owner: engine.

Why streaming. B = 256 worlds x N = 8 rings x 10 signals x 1 s at 1 us is 164 GB (07 P0-8, checked), and plain
decimation hides exactly the short excursions the acceptance tests count. So the compiled kernel calls
:func:`metrics_tick_row` once per plant step for every world, and the summary only reduces the accumulators. A
decimated record keeps min/max envelopes (``reducer = min | max``) so a 5 us spike survives a 1 ms decimation.

Definitions (per ring n of world b; detuning d = lambda_res - lambda_line of the resonance order nearest the DECLARED
line of the ring, positive = resonance red of the line; FWHM = lambda_line^2 fwhm_hz / c of the ring):

- lock error e = d - d_ref with the lock reference d_ref of ``MetricsSpec.lock_reference``: 'line' (d_ref = 0, the
  detuning from the line itself; the default, 07's literal "detuning"), 'oracle' (d_ref = the detuning of the
  oracle's best open-loop-stable relOMA point, re-evaluated with the oracle every ``oracle_period_ticks``; the OMA
  operating point a modulator is held at, 0.3-0.5 FWHM red of its line in g26) or 'target' (d_ref =
  ``lock_target_pm`` per ring, e.g. a controller's declared set-point detuning). A side-of-fringe or OMA lock sits
  hundreds of pm from its line by design, so the 5 pm lock budget (04 section 4: "lock" = the detuning held within
  5 pm of its operating point) only means something with 'oracle' or 'target'.
- lock state, with hysteresis: locked when |e| <= thr - hyst after being unlocked (or |e| <= thr for the first
  lock), unlocked when |e| > thr. thr = metrics.lock_threshold_pm, hyst = metrics.hysteresis_pm (07 P0-8).
- excursion: a transition locked -> unlocked after the first lock; its run length is the number of ticks until the
  ring relocks. ``excursion_count``, ``excursion_max_s``, ``time_unlocked_s`` (after the first lock) and a log2
  histogram of run lengths (bin j holds runs of 2^j .. 2^(j+1) - 1 ticks). ``max_abs_lock_error_pm`` and
  ``rms_lock_error_pm`` summarize e.
- time above each threshold: ticks with |e| > t_i for every t_i in ``time_above_pm`` (no hysteresis).
- time to lock: ``first_lock_s`` (first entry into lock) and ``time_to_lock_s`` (start of the locked run that is
  still on at the end of the run; NaN when the ring ends unlocked). Times are relative to the reset tick.
- max |d|, mean d and rms d over the run.
- relOMA = |P_thru,1 - P_thru,0| / P_in at the ring's own through port for its declared line (07 D9: "1 mW of laser
  into the MRM"; the per-level powers are the NRZ levels of the junction swing). Time-integrated. The oracle relOMA
  is the best relOMA over the detuning at the same instant among open-loop-stable points (G_sh dP_abs/dlambda < 1,
  07 P0-1 item 7) and also without that restriction; the OMA penalty is 10 log10(sum oracle / sum relOMA) (dB).
- expected errors sum BER(OMA_rx) bits over the run (07 P0-8, P0-11), with BER from mrm.plant.datapath (Gaussian Q
  linear in the bus-output OMA times the link transmission). BER windows of ``ber_sample_period_s`` (G26: 2.0 s):
  number of complete windows, number above ``ber_limit`` (1e-9, the G26 criterion) and the largest window BER.
- heater energy sum P_heater dt; saturation time: ticks with the latched register at code_max (``heater_sat_s``) and
  at code_min (``heater_low_rail_s``); code range used (min / max register); demanded peak slew: the largest
  |register change| / (ticks since the previous latch) / dt over every controller latch (07 P1-11: report demand).
- wrong line (07 P1-12): the ring sits within ``wrong_line_window_fwhm`` x FWHM of a line that is not its declared
  line (the kernel decides per tick, honouring metrics.allowed_assignments); ticks, entries (events) and the final
  state. FSR hop: the ring relocks on a resonance order other than the one of its previous lock (event count).
- detuning histogram (optional, 07 P0-8): 0.1 pm bins within +-2 FWHM and 10 pm bins out to +-FSR/2, plus one
  under- and one overflow bin; the fine half-width uses the largest FWHM of the bank and the coarse one the largest FSR
  (one bin layout for every world and ring). int64 counts of ticks.

All per-tick kernels are numba @njit(cache=True), take arrays only, write in place and run under
NUMBA_DISABLE_JIT=1. A multi-tick step of h ticks (07 P0-10 ``substep_policy: auto``) counts every tick with the
end-of-step value (``h`` argument).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numba import njit

__all__ = [
    "MetricsSpec", "MetricsState", "RecordSpec", "RecordSet", "TRUTH_FIELDS", "TR", "N_TR",
    "metrics_tick_row", "metrics_reset_row", "record_tick_row", "summarize", "metrics_from_trace",
    "envelope", "NAF", "NAI", "N_RUNHIST", "REDUCERS", "LOCK_REFERENCES",
]

# ------------------------------------------------------------------------------------------- truth fields
#: Per-ring Truth fields the kernel writes every tick (ARCHITECTURE 5.1 Truth; index = position). Units in the name.
TRUTH_FIELDS: tuple[str, ...] = (
    "t_ring_k",            # 0  mode-weighted ring temperature (ROM observer ring_mode_weighted), K
    "t_heater_k",          # 1  heater-resistor observer temperature, K
    "t_pd_k",              # 2  monitor PD observer temperature, K
    "p_heater_w",          # 3  heater electrical power (tick average), W
    "p_abs_w",             # 4  absorbed (heat) optical power, every line, order and direction, W
    "lam_res_m",           # 5  resonance of the order nearest the declared line (mean of the two NRZ levels), m
    "detuning_m",          # 6  lam_res - lam_line of the declared line, m (positive = resonance red of the line)
    "detuning_hwhm",       # 7  detuning in half-linewidths (HWHM = lam_line^2 fwhm_hz / (2 c))
    "p_drop_w",            # 8  monitor optical input (CW drop, every line and order, NRZ averaged), W
    "i_monitor_a",         # 9  monitor photocurrent incl. dark current at the PD temperature, A
    "p_in_w",              # 10 power at the ring input at its declared line, W
    "oma_w",               # 11 bus-output OMA of the declared line (|P_out,1 - P_out,0|), W
    "reloma",              # 12 relOMA = |P_thru,1 - P_thru,0| / P_in at the ring's own through port (07 D9), linear
    "er_db",               # 13 extinction ratio at the ring's through port, 10 log10(max/min level), dB
    "il_db",               # 14 insertion loss of the high level at the ring's through port, dB
    "heater_code",         # 15 latched heater DAC register (applied code)
    "bias_v",              # 16 tick-average bias voltage at the junction (reverse positive), V
    "order",               # 17 longitudinal order index of the resonance nearest the declared line
    "line_on",             # 18 line the ring sits on within the wrong-line window (-1 none)
    "t_base_k",            # 19 base temperature incl. the XPU rise (die level, replicated per ring), K
    "monitor_code",        # 20 latest released monitor ADC code (sensor, for records)
    "oracle_reloma",       # 21 oracle relOMA (stable points) at the last oracle evaluation, linear
    "ber",                 # 22 instantaneous BER of the declared line from oma_w
    "declared_line",       # 23 declared line index of the ring (tx.assignment or channel_plan.ring_to_line)
)
TR = {name: i for i, name in enumerate(TRUTH_FIELDS)}
N_TR = len(TRUTH_FIELDS)

# ------------------------------------------------------------------------------------------- accumulator layout
# float accumulators (B, N, NAF)
AF_SUM_DET = 0          # sum of d * h (m ticks)
AF_SUM_DET2 = 1         # sum of d^2 * h
AF_MAX_ABS_DET = 2      # m
AF_SUM_RELOMA = 3       # sum relOMA * h
AF_SUM_ORACLE = 4       # sum oracle relOMA (stable) * h
AF_SUM_ORACLE_UNR = 5   # sum oracle relOMA (unrestricted) * h
AF_HEATER_ENERGY = 6    # J
AF_EXP_ERRORS = 7       # expected errors
AF_BITS = 8             # bits sent
AF_WIN_ERRORS = 9       # expected errors in the current BER window
AF_WIN_BITS = 10        # bits in the current BER window
AF_MAX_WIN_BER = 11     # largest complete-window BER
AF_PEAK_SLEW = 12       # codes/s
AF_SUM_PABS = 13        # sum P_abs * h (W ticks)
AF_MAX_ABS_ERR = 14     # max |lock error|, m
AF_SUM_ERR2 = 15        # sum of lock error^2 * h (m^2 ticks)
NAF = 16
# int accumulators (B, N, NAI)
AI_TICKS = 0
AI_LOCKED = 1           # hysteresis state 0/1
AI_EVER_LOCKED = 2
AI_EXC_COUNT = 3
AI_RUN_CUR = 4          # current excursion length (ticks)
AI_RUN_MAX = 5
AI_TICKS_UNLOCKED = 6   # after the first lock
AI_FIRST_LOCK = 7       # tick of the first lock entry (-1)
AI_LAST_LOCK = 8        # tick at which the current / last locked run started (-1)
AI_WRONG_TICKS = 9
AI_WRONG_EVENTS = 10
AI_WRONG_STATE = 11
AI_ORDER_LOCK = 12      # order of the previous lock
AI_FSR_HOPS = 13
AI_SAT_HI = 14
AI_SAT_LO = 15
AI_CODE_MIN = 16
AI_CODE_MAX = 17
AI_PREV_POP_REG = 18
AI_PREV_POP_TICK = 19
AI_BER_WINDOWS = 20
AI_BER_OVER = 21
AI_WIN_TICKS = 22
AI_T0 = 23              # reset tick
AI_ABOVE0 = 24          # ticks above time_above threshold i at AI_ABOVE0 + i (up to N_ABOVE_MAX)
N_ABOVE_MAX = 4
NAI = AI_ABOVE0 + N_ABOVE_MAX
N_RUNHIST = 48          # log2 run-length bins (2^47 ticks = 4.5 years at 1 us)

# metric parameter vectors
MP_DT = 0               # s
MP_THR = 1              # lock threshold, m
MP_HYST = 2             # hysteresis, m
MP_BER_LIMIT = 3
MP_FINE_W = 4           # histogram fine half-width, m
MP_FINE_BIN = 5         # m
MP_COARSE_W = 6         # m
MP_COARSE_BIN = 7       # m
NMP_F = 8 + N_ABOVE_MAX  # time_above thresholds (m) at 8 + i
MPI_HIST = 0            # 1: histogram on
MPI_NFINE = 1
MPI_NCOARSE = 2         # coarse bins per side
MPI_BER_WIN = 3         # ticks per BER window (0: none)
MPI_N_ABOVE = 4
MPI_CODE_LO = 5
MPI_CODE_HI = 6
NMP_I = 8

_INT_MIN = -(2 ** 62)


@njit(cache=True)
def metrics_reset_row(k0, reg, af, ai, hist, runhist):
    """Clear one world's accumulators at reset tick k0; reg (N,) the initial heater registers."""
    for n in range(af.shape[0]):
        for j in range(af.shape[1]):
            af[n, j] = 0.0
        for j in range(ai.shape[1]):
            ai[n, j] = 0
        ai[n, AI_FIRST_LOCK] = -1
        ai[n, AI_LAST_LOCK] = -1
        ai[n, AI_ORDER_LOCK] = _INT_MIN
        ai[n, AI_CODE_MIN] = reg[n]
        ai[n, AI_CODE_MAX] = reg[n]
        ai[n, AI_PREV_POP_REG] = reg[n]
        ai[n, AI_PREV_POP_TICK] = k0
        ai[n, AI_T0] = k0
        for j in range(hist.shape[1]):
            hist[n, j] = 0
        for j in range(runhist.shape[1]):
            runhist[n, j] = 0


@njit(cache=True)
def _hist_bin(d, mpf, mpi):
    """Bin of detuning d (m) in the two-resolution histogram (module docstring)."""
    nf = mpi[MPI_NFINE]
    nc = mpi[MPI_NCOARSE]
    wf = mpf[MP_FINE_W]
    if d >= -wf and d < wf:
        j = int(math.floor((d + wf) / mpf[MP_FINE_BIN]))
        if j >= nf:
            j = nf - 1
        return 1 + nc + j
    if d < -wf:
        j = int(math.floor((-wf - d) / mpf[MP_COARSE_BIN]))      # 0 next to the fine region
        if j >= nc:
            return 0                                            # underflow
        return nc - j
    j = int(math.floor((d - wf) / mpf[MP_COARSE_BIN]))
    if j >= nc:
        return 1 + nc + nf + nc                                 # overflow
    return 1 + nc + nf + j


@njit(cache=True)
def _log2_bin(n):
    j = 0
    while n > 1 and j < N_RUNHIST - 1:
        n >>= 1
        j += 1
    return j


@njit(cache=True)
def metrics_tick_row(k, h, mpf, mpi, det, ref, order, wrong, reloma, oracle, oracle_unr, p_heater, p_abs, reg,
                     popped, ber, bits, af, ai, hist, runhist):
    """Update one world's accumulators with the state at the end of the step [k, k + h) (07 P0-8).

    k: first tick of the step; h: ticks it covers (1 at the fixed policy). mpf (NMP_F,), mpi (NMP_I,): parameters
    (MP_*, MPI_*). Per ring (N,): det detuning (m), ref lock reference detuning (m; the lock error is det - ref,
    module docstring), order (int64), wrong (bool: on a foreign line), reloma, oracle, oracle_unr (linear), p_heater
    (W), p_abs (W), reg (latched heater code), popped (bool: a controller command latched in this step), ber
    (instantaneous BER), bits (bits sent in the step). State af (N, NAF), ai (N, NAI), hist (N, NH) int64, runhist
    (N, N_RUNHIST) int64.
    """
    dt = mpf[MP_DT]
    thr = mpf[MP_THR]
    hyst = mpf[MP_HYST]
    n_above = mpi[MPI_N_ABOVE]
    hist_on = mpi[MPI_HIST] != 0
    win = mpi[MPI_BER_WIN]
    code_lo = mpi[MPI_CODE_LO]
    code_hi = mpi[MPI_CODE_HI]
    hf = float(h)
    k_end = k + h
    for n in range(det.shape[0]):
        d = det[n]
        ad = abs(d)
        ai[n, AI_TICKS] += h
        af[n, AF_SUM_DET] += d * hf
        af[n, AF_SUM_DET2] += d * d * hf
        if ad > af[n, AF_MAX_ABS_DET]:
            af[n, AF_MAX_ABS_DET] = ad
        e = d - ref[n]                       # lock error about the lock reference
        ae = abs(e)
        af[n, AF_SUM_ERR2] += e * e * hf
        if ae > af[n, AF_MAX_ABS_ERR]:
            af[n, AF_MAX_ABS_ERR] = ae
        for i in range(n_above):
            if ae > mpf[8 + i]:
                ai[n, AI_ABOVE0 + i] += h
        # lock state with hysteresis; excursions counted after the first lock
        if ai[n, AI_LOCKED] == 0:
            lim = thr if ai[n, AI_EVER_LOCKED] == 0 else thr - hyst
            if ae <= lim:
                ai[n, AI_LOCKED] = 1
                ai[n, AI_LAST_LOCK] = k_end
                if ai[n, AI_EVER_LOCKED] == 0:
                    ai[n, AI_EVER_LOCKED] = 1
                    ai[n, AI_FIRST_LOCK] = k_end
                else:
                    run = ai[n, AI_RUN_CUR]
                    if run > ai[n, AI_RUN_MAX]:
                        ai[n, AI_RUN_MAX] = run
                    runhist[n, _log2_bin(run)] += 1
                    ai[n, AI_RUN_CUR] = 0
                prev = ai[n, AI_ORDER_LOCK]
                if prev != _INT_MIN and prev != order[n]:
                    ai[n, AI_FSR_HOPS] += 1
                ai[n, AI_ORDER_LOCK] = order[n]
            elif ai[n, AI_EVER_LOCKED] != 0:
                ai[n, AI_RUN_CUR] += h
                ai[n, AI_TICKS_UNLOCKED] += h
        else:
            if ae > thr:
                ai[n, AI_LOCKED] = 0
                ai[n, AI_EXC_COUNT] += 1
                ai[n, AI_RUN_CUR] = h
                ai[n, AI_TICKS_UNLOCKED] += h
        # wrong line (decided by the kernel against the declared assignment)
        if wrong[n]:
            ai[n, AI_WRONG_TICKS] += h
            if ai[n, AI_WRONG_STATE] == 0:
                ai[n, AI_WRONG_EVENTS] += 1
            ai[n, AI_WRONG_STATE] = 1
        else:
            ai[n, AI_WRONG_STATE] = 0
        # OMA
        af[n, AF_SUM_RELOMA] += reloma[n] * hf
        af[n, AF_SUM_ORACLE] += oracle[n] * hf
        af[n, AF_SUM_ORACLE_UNR] += oracle_unr[n] * hf
        af[n, AF_SUM_PABS] += p_abs[n] * hf
        # errors and BER windows
        e = ber[n] * bits[n]
        af[n, AF_EXP_ERRORS] += e
        af[n, AF_BITS] += bits[n]
        if win > 0:
            af[n, AF_WIN_ERRORS] += e
            af[n, AF_WIN_BITS] += bits[n]
            ai[n, AI_WIN_TICKS] += h
            if ai[n, AI_WIN_TICKS] >= win:
                ai[n, AI_BER_WINDOWS] += 1
                wb = af[n, AF_WIN_ERRORS] / af[n, AF_WIN_BITS] if af[n, AF_WIN_BITS] > 0.0 else 0.0
                if wb > af[n, AF_MAX_WIN_BER]:
                    af[n, AF_MAX_WIN_BER] = wb
                if wb > mpf[MP_BER_LIMIT]:
                    ai[n, AI_BER_OVER] += 1
                af[n, AF_WIN_ERRORS] = 0.0
                af[n, AF_WIN_BITS] = 0.0
                ai[n, AI_WIN_TICKS] = 0
        # heater
        af[n, AF_HEATER_ENERGY] += p_heater[n] * hf * dt
        r = reg[n]
        if r >= code_hi:
            ai[n, AI_SAT_HI] += h
        if r <= code_lo:
            ai[n, AI_SAT_LO] += h
        if r < ai[n, AI_CODE_MIN]:
            ai[n, AI_CODE_MIN] = r
        if r > ai[n, AI_CODE_MAX]:
            ai[n, AI_CODE_MAX] = r
        if popped[n]:
            span = k - ai[n, AI_PREV_POP_TICK]
            if span > 0:
                sl = abs(r - ai[n, AI_PREV_POP_REG]) / (span * dt)
                if sl > af[n, AF_PEAK_SLEW]:
                    af[n, AF_PEAK_SLEW] = sl
            ai[n, AI_PREV_POP_REG] = r
            ai[n, AI_PREV_POP_TICK] = k
        if hist_on:
            hist[n, _hist_bin(d, mpf, mpi)] += h


# ------------------------------------------------------------------------------------------- record specs
REDUCERS = ("mean", "min", "max", "last")
#: references of the lock error (module docstring)
LOCK_REFERENCES = ("line", "oracle", "target")


@dataclass(frozen=True)
class RecordSpec:
    """One recorded signal (07 P0-8): a Truth field of TRUTH_FIELDS, a decimation in ticks and a reducer.

    reducer: mean (window average), min / max (envelope), last (value at the end of the window). Use
    :func:`envelope` for a min/max pair.
    """
    channel: str
    decimation: int = 1
    reducer: str = "mean"

    def __post_init__(self):
        if self.channel not in TR:
            raise ValueError(f"record channel {self.channel!r} is not a Truth field; known: {TRUTH_FIELDS}")
        if int(self.decimation) < 1:
            raise ValueError("record decimation must be >= 1 tick")
        if self.reducer not in REDUCERS:
            raise ValueError(f"record reducer {self.reducer!r} not in {REDUCERS}")

    @property
    def key(self) -> str:
        return f"{self.channel}@{int(self.decimation)}:{self.reducer}"


def envelope(channel: str, decimation: int) -> list[RecordSpec]:
    """A min/max envelope of one channel at one decimation (07 P0-8)."""
    return [RecordSpec(channel, decimation, "min"), RecordSpec(channel, decimation, "max")]


def _as_record_specs(record) -> list[RecordSpec]:
    if record is None:
        return []
    out = []
    for r in record:
        if isinstance(r, RecordSpec):
            out.append(r)
        elif isinstance(r, str):
            out.append(RecordSpec(r))
        elif isinstance(r, Mapping):
            out.append(RecordSpec(r["channel"], int(r.get("decimation", 1)), r.get("reducer", "mean")))
        else:
            ch, dec, *red = r
            out.append(RecordSpec(ch, int(dec), red[0] if red else "mean"))
    keys = [r.key for r in out]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate record specs: {keys}")
    return out


@dataclass
class RecordSet:
    """Kernel arrays of a list of RecordSpec for a run of n_ticks ticks and B worlds (trace buffer included).

    rec_i (R, 5) int64: [field, decimation, reducer, first slot, slots]; trace (B, S_total, N) float64 window values;
    tick_end (S_total,) int64 relative end tick of each slot (filled at summary time); acc (B, R, N), cnt (B, R),
    idx (B, R) running window state.
    """
    specs: list
    rec_i: np.ndarray
    trace: np.ndarray
    acc: np.ndarray
    cnt: np.ndarray
    idx: np.ndarray

    @classmethod
    def build(cls, record, batch: int, n_rings: int, n_ticks: int) -> "RecordSet":
        specs = _as_record_specs(record)
        R = len(specs)
        rec_i = np.zeros((max(R, 1), 5), np.int64)
        slot = 0
        for r, s in enumerate(specs):
            n_slots = int(n_ticks) // int(s.decimation)
            rec_i[r] = (TR[s.channel], int(s.decimation), REDUCERS.index(s.reducer), slot, n_slots)
            slot += n_slots
        mem = batch * max(slot, 1) * n_rings * 8
        if mem > 2 * 1024 ** 3:
            raise MemoryError(f"record buffers need {mem / 1e9:.1f} GB; decimate more or record fewer channels "
                              "(Monte Carlo runs record metrics only, 07 P0-8)")
        return cls(specs=specs, rec_i=rec_i[:R] if R else np.zeros((0, 5), np.int64),
                   trace=np.zeros((batch, max(slot, 1), n_rings)), acc=np.zeros((batch, max(R, 1), n_rings)),
                   cnt=np.zeros((batch, max(R, 1)), np.int64), idx=np.zeros((batch, max(R, 1)), np.int64))

    def reset(self) -> None:
        self.trace[:] = 0.0
        self.acc[:] = 0.0
        self.cnt[:] = 0
        self.idx[:] = 0

    def traces(self, dt_s: float, t0_tick: int = 0) -> dict[str, dict[str, np.ndarray]]:
        """{key: {'t_s': (S,) window end times, 'value': (S, B, N)}} of the completed windows."""
        out = {}
        for r, s in enumerate(self.specs):
            _, dec, _, s0, ns = (int(x) for x in self.rec_i[r])
            done = int(self.idx[:, r].min()) if self.idx.shape[0] else 0
            done = min(done, ns)
            t = (t0_tick + dec * (np.arange(done) + 1)) * float(dt_s)
            out[s.key] = {"t_s": t, "value": np.moveaxis(self.trace[:, s0:s0 + done, :], 1, 0).copy()}
        return out


@njit(cache=True)
def record_tick_row(h, tv, rec_i, acc, cnt, idx, trace):
    """Feed one world's Truth vector tv (N_TR, N) of a step of h ticks into every record (07 P0-8 reducers).

    A step longer than the remaining window length is attributed to the current window (multi-tick steps never
    straddle a record window when the auto policy breaks substeps at record windows; at the fixed policy h = 1).
    """
    for r in range(rec_i.shape[0]):
        f = rec_i[r, 0]
        dec = rec_i[r, 1]
        red = rec_i[r, 2]
        if idx[r] >= rec_i[r, 4]:
            continue
        c = cnt[r]
        for n in range(tv.shape[1]):
            v = tv[f, n]
            if red == 0:
                if c == 0:
                    acc[r, n] = v * h
                else:
                    acc[r, n] += v * h
            elif red == 1:
                if c == 0 or v < acc[r, n]:
                    acc[r, n] = v
            elif red == 2:
                if c == 0 or v > acc[r, n]:
                    acc[r, n] = v
            else:
                acc[r, n] = v
        c += h
        if c >= dec:
            s = rec_i[r, 3] + idx[r]
            for n in range(tv.shape[1]):
                trace[s, n] = acc[r, n] / c if red == 0 else acc[r, n]
            idx[r] += 1
            c = 0
        cnt[r] = c


# ------------------------------------------------------------------------------------------- Python side
@dataclass
class MetricsSpec:
    """Acceptance-metric settings (07 P0-8 ``metrics:`` section, plus engine options).

    From the config: lock_threshold_pm, hysteresis_pm (metrics.*), ber_sample_period_s (2.0 s, G26),
    reloma_definition (oma_over_pin only, D9), allowed_assignments (fixed | cyclic | any, P1-12). Engine options
    (no schema key; documented defaults): time_above_pm (thresholds of the time-above counters; default the lock
    threshold), histogram (on/off), wrong_line_window_fwhm (1 FWHM: a ring within one linewidth of a foreign line
    carries that line's modulation), ber_limit (1e-9, the G26 continuous-BER criterion), oracle_period_ticks (how
    often the oracle relOMA is re-evaluated; 1000 ticks = 1 ms at 1 us; it changes only with slow quantities: bias,
    line power, ring loss), oracle_points (detuning scan points over +-1.5 FWHM), lock_reference ('line' | 'oracle' |
    'target', the reference of the lock error, module docstring; 'oracle' needs oracle=True) and lock_target_pm (the
    'target' reference: a scalar or a (B, N) array of detunings in pm).
    """
    lock_threshold_pm: float = 5.0
    hysteresis_pm: float = 1.0
    ber_sample_period_s: float = 2.0
    reloma_definition: str = "oma_over_pin"
    allowed_assignments: str = "fixed"
    time_above_pm: tuple = ()
    histogram: bool = True
    fine_bin_pm: float = 0.1
    coarse_bin_pm: float = 10.0
    wrong_line_window_fwhm: float = 1.0
    ber_limit: float = 1e-9
    oracle_period_ticks: int = 1000
    oracle_points: int = 151
    oracle: bool = True
    lock_reference: str = "line"
    lock_target_pm: Any = None

    def __post_init__(self):
        if self.reloma_definition != "oma_over_pin":
            raise ValueError("metrics.reloma_definition: only oma_over_pin is defined (07 D9)")
        if self.allowed_assignments not in ("fixed", "cyclic", "any"):
            raise ValueError("metrics.allowed_assignments must be fixed, cyclic or any (07 P1-12)")
        if not (self.lock_threshold_pm > 0 and 0 <= self.hysteresis_pm < self.lock_threshold_pm):
            raise ValueError("need lock_threshold_pm > 0 and 0 <= hysteresis_pm < lock_threshold_pm")
        if self.lock_reference not in LOCK_REFERENCES:
            raise ValueError(f"metrics.lock_reference must be one of {LOCK_REFERENCES}")
        if self.lock_reference == "oracle" and not self.oracle:
            raise ValueError("lock_reference='oracle' needs the oracle (oracle=True)")
        if self.lock_reference == "target" and self.lock_target_pm is None:
            raise ValueError("lock_reference='target' needs lock_target_pm (pm, scalar or (B, N))")
        ta = tuple(float(x) for x in (self.time_above_pm or (self.lock_threshold_pm,)))
        if len(ta) > N_ABOVE_MAX:
            raise ValueError(f"at most {N_ABOVE_MAX} time_above thresholds")
        object.__setattr__(self, "time_above_pm", ta)

    @property
    def lock_reference_mode(self) -> int:
        """0: a fixed reference (line or target, from MetricsState.ref); 1: the oracle's operating point."""
        return 1 if self.lock_reference == "oracle" else 0

    def reference_m(self, batch: int, n_rings: int) -> np.ndarray:
        """The fixed lock reference (B, N) in m: zeros for 'line' and 'oracle' (the kernel fills the oracle's),
        lock_target_pm for 'target'."""
        if self.lock_reference != "target":
            return np.zeros((batch, n_rings))
        return np.ascontiguousarray(np.broadcast_to(np.asarray(self.lock_target_pm, float) * 1e-12,
                                                    (batch, n_rings)), dtype=np.float64).copy()

    @classmethod
    def from_config(cls, cfg, **overrides) -> "MetricsSpec":
        """Read metrics.* from a Config (or a mapping); keyword arguments override."""
        def val(key, default):
            try:
                v = cfg.get(key, None) if hasattr(cfg, "get") else None
            except Exception:              # noqa: BLE001 - a Config raises for unknown keys in some modes
                v = None
            if isinstance(v, Mapping) and "value" in v:
                v = v["value"]
            return default if v is None else v
        kw = dict(lock_threshold_pm=float(val("metrics.lock_threshold_pm", 5.0)),
                  hysteresis_pm=float(val("metrics.hysteresis_pm", 1.0)),
                  ber_sample_period_s=float(val("metrics.ber_sample_period_s", 2.0)),
                  reloma_definition=str(val("metrics.reloma_definition", "oma_over_pin")),
                  allowed_assignments=str(val("metrics.allowed_assignments", "fixed")),
                  lock_reference=str(val("metrics.lock_reference", "line")),
                  lock_target_pm=val("metrics.lock_target_pm", None))
        kw.update(overrides)
        return cls(**kw)

    @property
    def assignment_mode(self) -> int:
        return ("fixed", "cyclic", "any").index(self.allowed_assignments)


@dataclass
class MetricsState:
    """Accumulator arrays for B worlds and N rings plus their parameter vectors (built by :meth:`build`)."""
    spec: MetricsSpec
    mpf: np.ndarray
    mpi: np.ndarray
    af: np.ndarray
    ai: np.ndarray
    hist: np.ndarray
    runhist: np.ndarray
    dt_s: float
    hist_edges_m: np.ndarray | None = None
    ref: np.ndarray | None = None            # (B, N) lock reference detuning (m); the kernel updates it ('oracle')

    @classmethod
    def build(cls, spec: MetricsSpec, batch: int, n_rings: int, dt_s: float, *, fwhm_max_m: float,
              fsr_max_m: float, code_lo: int, code_hi: int, histogram: bool | None = None) -> "MetricsState":
        hist_on = spec.histogram if histogram is None else bool(histogram)
        mpf = np.zeros(NMP_F)
        mpi = np.zeros(NMP_I, np.int64)
        mpf[MP_DT] = dt_s
        mpf[MP_THR] = spec.lock_threshold_pm * 1e-12
        mpf[MP_HYST] = spec.hysteresis_pm * 1e-12
        mpf[MP_BER_LIMIT] = spec.ber_limit
        fine_bin = spec.fine_bin_pm * 1e-12
        coarse_bin = spec.coarse_bin_pm * 1e-12
        wf = math.ceil(2.0 * fwhm_max_m / fine_bin) * fine_bin
        wc = max(0.5 * fsr_max_m, wf + coarse_bin)
        nf = int(round(2.0 * wf / fine_bin))
        nc = int(math.ceil((wc - wf) / coarse_bin))
        mpf[MP_FINE_W] = wf
        mpf[MP_FINE_BIN] = fine_bin
        mpf[MP_COARSE_W] = wf + nc * coarse_bin
        mpf[MP_COARSE_BIN] = coarse_bin
        for i, t in enumerate(spec.time_above_pm):
            mpf[8 + i] = t * 1e-12
        mpi[MPI_HIST] = int(hist_on)
        mpi[MPI_NFINE] = nf
        mpi[MPI_NCOARSE] = nc
        win = spec.ber_sample_period_s / dt_s
        mpi[MPI_BER_WIN] = int(round(win)) if spec.ber_sample_period_s > 0 else 0
        mpi[MPI_N_ABOVE] = len(spec.time_above_pm)
        mpi[MPI_CODE_LO] = int(code_lo)
        mpi[MPI_CODE_HI] = int(code_hi)
        nh = (nf + 2 * nc + 2) if hist_on else 1
        mem = batch * n_rings * nh * 8
        if hist_on and mem > 1024 ** 3:
            raise MemoryError(f"detuning histogram needs {mem / 1e9:.1f} GB; pass histogram=False")
        edges = None
        if hist_on:
            left = -wf - coarse_bin * np.arange(nc, 0, -1)
            fine = -wf + fine_bin * np.arange(nf + 1)
            right = wf + coarse_bin * np.arange(1, nc + 1)
            edges = np.concatenate([[-np.inf], left, fine, right, [np.inf]])
        return cls(spec=spec, mpf=mpf, mpi=mpi, af=np.zeros((batch, n_rings, NAF)),
                   ai=np.zeros((batch, n_rings, NAI), np.int64), hist=np.zeros((batch, n_rings, nh), np.int64),
                   runhist=np.zeros((batch, n_rings, N_RUNHIST), np.int64), dt_s=float(dt_s), hist_edges_m=edges,
                   ref=spec.reference_m(batch, n_rings))

    def reset(self, k0: int, reg) -> None:
        reg = np.asarray(reg, np.int64)
        if self.ref is not None:
            self.ref[...] = self.spec.reference_m(*self.ref.shape)
        for b in range(self.af.shape[0]):
            metrics_reset_row(np.int64(k0), reg[b], self.af[b], self.ai[b], self.hist[b], self.runhist[b])


def _ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(a, float) / np.asarray(b, float)


def _db(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(x)


def summarize(state: MetricsState, *, k_now: int | None = None) -> dict[str, Any]:
    """Reduce the accumulators to the acceptance metrics (07 P0-8). Per-ring values are (B, N) arrays; the
    world-level values used by the compression convergence check (mrm.scenarios.assess_convergence:
    max_abs_detuning_pm, wrong_line_events, fsr_hop_events, heater_range_used_codes, integrated_errors) are (B,)
    arrays under ``world``."""
    af, ai, dt, spec = state.af, state.ai, state.dt_s, state.spec
    ticks = ai[:, :, AI_TICKS].astype(float)
    tk = np.where(ticks > 0, ticks, np.nan)
    t0 = ai[:, :, AI_T0]
    out: dict[str, Any] = {}
    out["duration_s"] = ticks * dt
    out["max_abs_detuning_pm"] = af[:, :, AF_MAX_ABS_DET] * 1e12
    out["mean_detuning_pm"] = _ratio(af[:, :, AF_SUM_DET], tk) * 1e12
    out["rms_detuning_pm"] = np.sqrt(_ratio(af[:, :, AF_SUM_DET2], tk)) * 1e12
    out["max_abs_lock_error_pm"] = af[:, :, AF_MAX_ABS_ERR] * 1e12
    out["rms_lock_error_pm"] = np.sqrt(_ratio(af[:, :, AF_SUM_ERR2], tk)) * 1e12
    if state.ref is not None:
        out["lock_reference_pm"] = state.ref * 1e12
    out["time_above_s"] = {f"{t:g}pm": ai[:, :, AI_ABOVE0 + i] * dt for i, t in enumerate(spec.time_above_pm)}
    out["locked_final"] = ai[:, :, AI_LOCKED].astype(bool)
    out["ever_locked"] = ai[:, :, AI_EVER_LOCKED].astype(bool)
    out["excursion_count"] = ai[:, :, AI_EXC_COUNT].copy()
    run_max = np.maximum(ai[:, :, AI_RUN_MAX], np.where(ai[:, :, AI_LOCKED] == 0, ai[:, :, AI_RUN_CUR], 0))
    out["excursion_max_s"] = run_max * dt
    out["time_unlocked_s"] = ai[:, :, AI_TICKS_UNLOCKED] * dt
    out["excursion_run_hist_log2"] = state.runhist.copy()
    fl = ai[:, :, AI_FIRST_LOCK]
    ll = ai[:, :, AI_LAST_LOCK]
    out["first_lock_s"] = np.where(fl >= 0, (fl - t0) * dt, np.nan)
    out["time_to_lock_s"] = np.where((ll >= 0) & (ai[:, :, AI_LOCKED] == 1), (ll - t0) * dt, np.nan)
    out["wrong_line_s"] = ai[:, :, AI_WRONG_TICKS] * dt
    out["wrong_line_events"] = ai[:, :, AI_WRONG_EVENTS].copy()
    out["wrong_line_final"] = ai[:, :, AI_WRONG_STATE].astype(bool)
    out["fsr_hop_events"] = ai[:, :, AI_FSR_HOPS].copy()
    s_rel, s_or, s_oru = af[:, :, AF_SUM_RELOMA], af[:, :, AF_SUM_ORACLE], af[:, :, AF_SUM_ORACLE_UNR]
    out["mean_reloma"] = _ratio(s_rel, tk)
    out["mean_reloma_db"] = _db(_ratio(s_rel, tk))
    out["oracle_reloma_db"] = _db(_ratio(s_or, tk))
    out["oracle_reloma_unrestricted_db"] = _db(_ratio(s_oru, tk))
    out["oma_penalty_db"] = _db(_ratio(s_or, s_rel))
    out["oma_penalty_unrestricted_db"] = _db(_ratio(s_oru, s_rel))
    out["mean_p_abs_w"] = _ratio(af[:, :, AF_SUM_PABS], tk)
    bits = af[:, :, AF_BITS]
    out["expected_errors"] = af[:, :, AF_EXP_ERRORS].copy()
    out["bits"] = bits.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["ber_mean"] = np.where(bits > 0, af[:, :, AF_EXP_ERRORS] / bits, np.nan)
    out["ber_windows"] = ai[:, :, AI_BER_WINDOWS].copy()
    out["ber_windows_over_limit"] = ai[:, :, AI_BER_OVER].copy()
    out["max_window_ber"] = af[:, :, AF_MAX_WIN_BER].copy()
    out["heater_energy_j"] = af[:, :, AF_HEATER_ENERGY].copy()
    out["heater_sat_s"] = ai[:, :, AI_SAT_HI] * dt
    out["heater_low_rail_s"] = ai[:, :, AI_SAT_LO] * dt
    out["peak_slew_codes_per_s"] = af[:, :, AF_PEAK_SLEW].copy()
    out["code_min_used"] = ai[:, :, AI_CODE_MIN].copy()
    out["code_max_used"] = ai[:, :, AI_CODE_MAX].copy()
    out["heater_range_used_codes"] = ai[:, :, AI_CODE_MAX] - ai[:, :, AI_CODE_MIN]
    if state.hist_edges_m is not None:
        out["detuning_histogram"] = {"edges_pm": state.hist_edges_m * 1e12, "counts": state.hist.copy()}
    out["world"] = {
        "max_abs_detuning_pm": out["max_abs_detuning_pm"].max(axis=1),
        "wrong_line_events": out["wrong_line_events"].sum(axis=1),
        "fsr_hop_events": out["fsr_hop_events"].sum(axis=1),
        "heater_range_used_codes": out["heater_range_used_codes"].max(axis=1),
        "integrated_errors": out["expected_errors"].sum(axis=1),
        "excursion_count": out["excursion_count"].sum(axis=1),
        "time_unlocked_s": out["time_unlocked_s"].sum(axis=1),
    }
    out["settings"] = {"lock_threshold_pm": spec.lock_threshold_pm, "hysteresis_pm": spec.hysteresis_pm,
                       "ber_sample_period_s": spec.ber_sample_period_s, "ber_limit": spec.ber_limit,
                       "allowed_assignments": spec.allowed_assignments,
                       "wrong_line_window_fwhm": spec.wrong_line_window_fwhm,
                       "reloma_definition": spec.reloma_definition, "lock_reference": spec.lock_reference}
    return out


def metrics_from_trace(spec: MetricsSpec, dt_s: float, *, det_m, order, wrong, reloma, oracle, oracle_unr,
                       p_heater_w, p_abs_w, reg, popped, ber, bits, code_lo: int, code_hi: int,
                       reg0, k0: int = 0, lock_ref_m=None) -> dict[str, np.ndarray]:
    """NumPy reference: the same metrics from full per-tick traces (T, B, N) (07 P0-8 test: the accumulators must
    equal the metrics computed from a full trace). Written independently of :func:`metrics_tick_row`. lock_ref_m:
    the lock reference detuning per tick (T, B, N) or broadcastable (m); None = 0 (the line)."""
    det = np.asarray(det_m, float)
    T, B, N = det.shape
    thr = spec.lock_threshold_pm * 1e-12
    hyst = spec.hysteresis_pm * 1e-12
    err = det - (0.0 if lock_ref_m is None else np.broadcast_to(np.asarray(lock_ref_m, float), det.shape))
    ad = np.abs(err)
    out = {"max_abs_detuning_pm": np.abs(det).max(axis=0) * 1e12, "mean_detuning_pm": det.mean(axis=0) * 1e12,
           "rms_detuning_pm": np.sqrt((det ** 2).mean(axis=0)) * 1e12,
           "max_abs_lock_error_pm": ad.max(axis=0) * 1e12, "rms_lock_error_pm": np.sqrt((err ** 2).mean(axis=0)) * 1e12}
    out["time_above_s"] = {f"{t:g}pm": (ad > t * 1e-12).sum(axis=0) * dt_s for t in spec.time_above_pm}
    exc = np.zeros((B, N), np.int64)
    unl = np.zeros((B, N), np.int64)
    run_max = np.zeros((B, N), np.int64)
    first = np.full((B, N), -1, np.int64)
    last = np.full((B, N), -1, np.int64)
    locked_final = np.zeros((B, N), bool)
    hops = np.zeros((B, N), np.int64)
    for b in range(B):
        for n in range(N):
            locked, ever, run, prev_order = False, False, 0, None
            for t in range(T):
                x = ad[t, b, n]
                k = k0 + t
                if not locked:
                    if x <= (thr if not ever else thr - hyst):
                        locked = True
                        last[b, n] = k + 1
                        if not ever:
                            ever = True
                            first[b, n] = k + 1
                        else:
                            run_max[b, n] = max(run_max[b, n], run)
                            run = 0
                        if prev_order is not None and prev_order != order[t, b, n]:
                            hops[b, n] += 1
                        prev_order = order[t, b, n]
                    elif ever:
                        run += 1
                        unl[b, n] += 1
                else:
                    if x > thr:
                        locked = False
                        exc[b, n] += 1
                        run = 1
                        unl[b, n] += 1
            if not locked:
                run_max[b, n] = max(run_max[b, n], run)
            locked_final[b, n] = locked
    out["excursion_count"] = exc
    out["time_unlocked_s"] = unl * dt_s
    out["excursion_max_s"] = run_max * dt_s
    out["first_lock_s"] = np.where(first >= 0, (first - k0) * dt_s, np.nan)
    out["time_to_lock_s"] = np.where((last >= 0) & locked_final, (last - k0) * dt_s, np.nan)
    out["locked_final"] = locked_final
    out["fsr_hop_events"] = hops
    w = np.asarray(wrong, bool)
    out["wrong_line_s"] = w.sum(axis=0) * dt_s
    entries = w.copy()
    entries[1:] &= ~w[:-1]
    out["wrong_line_events"] = entries.sum(axis=0)
    out["wrong_line_final"] = w[-1]
    rel, orc, oru = (np.asarray(x, float) for x in (reloma, oracle, oracle_unr))
    out["mean_reloma"] = rel.mean(axis=0)
    out["oma_penalty_db"] = _db(_ratio(orc.sum(axis=0), rel.sum(axis=0)))
    out["oma_penalty_unrestricted_db"] = _db(_ratio(oru.sum(axis=0), rel.sum(axis=0)))
    out["mean_p_abs_w"] = np.asarray(p_abs_w, float).mean(axis=0)
    e = np.asarray(ber, float) * np.asarray(bits, float)
    out["expected_errors"] = e.sum(axis=0)
    out["heater_energy_j"] = np.asarray(p_heater_w, float).sum(axis=0) * dt_s
    r = np.asarray(reg, np.int64)
    out["heater_sat_s"] = (r >= code_hi).sum(axis=0) * dt_s
    out["heater_low_rail_s"] = (r <= code_lo).sum(axis=0) * dt_s
    out["code_min_used"] = r.min(axis=0)
    out["code_max_used"] = r.max(axis=0)
    out["heater_range_used_codes"] = r.max(axis=0) - r.min(axis=0)
    pk = np.zeros((B, N))
    pop = np.asarray(popped, bool)
    r0 = np.asarray(reg0, np.int64)
    for b in range(B):
        for n in range(N):
            prev_t, prev_r = k0, int(r0[b, n])
            for t in np.flatnonzero(pop[:, b, n]):
                rr = int(r[t, b, n])
                span = (k0 + t) - prev_t
                if span > 0:
                    pk[b, n] = max(pk[b, n], abs(rr - prev_r) / (span * dt_s))
                prev_t, prev_r = k0 + t, rr
    out["peak_slew_codes_per_s"] = pk
    return out

