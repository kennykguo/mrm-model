"""Ramp-and-maximize acquisition (Lightmatter PAT-3) with a red-side lock hand-off, for every ring at once.

Sources: PAT-3 (US12470304B2 claims 7-10: a linear ramp drives the ring heater, the controller parks at the value
that maximizes the monitor; docs/research/01 section 3.2), the acquisition FSM of docs/research/04 C1 (init ->
search -> map -> return -> hand-off, "approach from the same direction because of bistability", [SUN15], [KRA24]),
and 07 P0-1 (self-heating bistability: rings start blue of their line, so a heater-up sweep approaches on the
positive-feedback side and jumps).

Physics that sets the algorithm (07 P0-1; checked on the g26 reference plant, tests/verification/test_ctrl_b_ramp.py)
-------------------------------------------------------------------------------------------------------------------
Let delta = lambda_laser - lambda_res (positive = resonance blue of the laser) and delta0 the value without
self-heating. The ring sits where delta + G P_abs(delta) = delta0. When G P_abs,peak exceeds 0.77 FWHM (g26 nominal:
about 3x), the map has a cold branch (far blue), an unstable middle branch, and a hot branch that runs from just blue
of the line through the line to the red side. A heater-up sweep (delta0 decreasing) follows the cold branch until
its fold and then JUMPS to the hot branch, which in the strong case lands on the RED side of the line, past the
drop-port peak: the up-sweep never sees the true peak (g26 nominal, measured: a 0.4 nm jump within 4 us, landing
0.1-0.2 nm red at about half of the peak; the hot branch spans up to ~1.2 nm of heater tuning). The red side is
self-stabilizing (dP_abs/dlambda_res < 0), so a heater-DOWN scan from there moves the ring continuously back
through the true peak; just blue of the peak it falls off the fold. Hence:

    UP          ramp w up (w = (code/code_max)^q, q = 2 for a voltage/current DAC: uniform in heater power) until a
                resonance is passed: the monitor rose by >= detect above its floor (often in one step: a jump) and
                fell again by drop_frac of the rise;
    DOWN        scan w down slowly: the monitor climbs the stable red branch to the TRUE peak y_pk (measured here),
                then falls (fold or plain blue flank); record (w, y) for the slope at the set-point;
    REAPPROACH  ramp up again past the peak (jump or continuous) until the ring is on the red side;
    LOCK        integral loop on the block-mean monitor y at the set-point r = y_bg + alpha (y_pk - y_bg) on the red
                side (heater up when y > r). Gain Ki = 2 pi f_c T_s / |dy/dcode|, with dy/dcode measured at r on the
                DOWN scan (it includes the self-heating stiffness 1 + G|P'|, 07 P0-1). Loss of lock (y below
                y_bg + lost_frac (y_pk - y_bg) for lost_steps steps, or the heater at a rail) restarts the search.

Two cases need care. Bistable re-approach: the DOWN scan's fall-off can leave the ring on the hot branch of the next
bluer line, whose red flank must not be taken for this one; REAPPROACH therefore waits for the jump near the drive
where the UP sweep jumped. Heaters only heat: when the red branch reaches w = 0 before its peak (a line whose hot
branch extends below zero drive), the highest point seen is taken as a partial peak (``partial``) and the ring locks
there. Supervisors also use two more modes: MAP (record every resonance an up-sweep passes, no hand-off) and PARK
(move to a drive from below, so the hysteresis branch reached is the one an up-sweep recorded).

The lock point is reached along the stable red branch, so its approach direction does not matter; what the bistability
forbids is reading the peak (and so the set-point) from the up-sweep, and locking on the blue flank, which is the
unstable middle branch. ``side='red'`` is the only side implemented (doc: 07 P0-1 item 7, the open-loop-stable side).

Search windows ``w_lo``/``w_hi`` per ring restrict which resonance is taken (the WDM supervisor or a factory table
sets them, mrm.control.baselines.wdm_supervisor); the default takes the first resonance approached from the blue side
(a ring already captured by self-heating at code 0 sits red of that line and moves away from it).

Knowledge: only the DAC facts are needed. With ``heater_nominal`` + ``optics_nominal`` (+ ``dlambda_dt_nominal``) in
``controller_interface.knowledge`` the sweep steps follow the nominal linewidth in drive units and the telemetry
reports a detuning estimate in pm; otherwise the documented defaults below are used (mrm.control.calibration).
The algorithm never reads Truth or sampled parameters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mrm.control.api import BaseController, ControllerSpec, Observation
from mrm.control.calibration import NominalDesign, code_of_w, drive_exponent, nominal_design, w_of_code
from mrm.control.registry import register

__all__ = ["ST_IDLE", "ST_UP", "ST_DOWN", "ST_REAPPROACH", "ST_LOCK", "ST_FAIL", "ST_HOLD", "ST_PARK", "STATE_NAMES",
           "RampParams", "RampFSM", "RampAcquire", "AcquireThenLock", "monitor_mean"]

ST_IDLE = 0          # waiting (start delay); holds the code
ST_UP = 1            # search: ramp up until a resonance is passed
ST_DOWN = 2          # measure the true peak on the stable red branch, scanning down
ST_REAPPROACH = 3    # ramp up past the peak to the red side
ST_LOCK = 4          # red-side set-point loop
ST_FAIL = 5          # no resonance in the window (holds the park code)
ST_HOLD = 6          # parked at a commanded drive by a supervisor
ST_PARK = 7          # ramping up to a park drive from below (same hysteresis branch as an up-sweep)
STATE_NAMES = ("idle", "up", "down", "reapproach", "lock", "fail", "hold", "park")

_BUF = 128           # (w, y) points kept per ring on the DOWN scan


@dataclass
class RampParams:
    """Tuning of the acquisition and the hand-off lock. Drive steps are in w = (code/code_max)^q (module docstring);
    None = derived from the nominal linewidth when the knowledge allows, else the default in brackets.

    alpha: set-point fraction of the peak above background (0.5: the G26 half-maximum lock, CAL-8 15-16 uA);
    dw_up [0.004], dw_down [0.0015], dw_reapproach [0.003]: steps per controller period (up/down/re-approach;
        derived as fwhm_w / 3.5, fwhm_w / 10, fwhm_w / 5);
    detect_frac: minimum rise of a resonance above its floor, fraction of the ADC full scale (0.05);
    drop_frac: fall after the maximum that confirms it, fraction of the rise (0.15, at least 0.01 FS);
    jump_frac: one-step rise counted as a bistability jump, fraction of full scale (0.08);
    lock_bw_hz: crossover of the integral lock (3 kHz: ~70 deg margin over the 3-13 us thermal poles and ~10 us of
        ADC/DAC/period delay; type-1 lag ramp / (2 pi f_c), ~6 pm at 1616 degC/s and 70 pm/K: measured 6.4 pm
        max over a full A2 cycle, tests/verification/test_ctrl_b_ramp_aggressor.py);
    lost_frac, lost_steps: loss-of-lock test; max_retries: searches before FAIL;
    w_start, w_lo, w_hi: start drive and search window (scalar, or one value per ring);
    rings: the rings this controller acquires (None = all); the others keep their applied heater code;
    start_delay_s: wait before the first sweep, counted from the first step after a reset (50 us: the turn-on
        self-heating capture settles; set it to the laser turn-on ramp when there is one, 07 P0-1 item 6);
    law: heater driver law assumed when the heater_nominal knowledge is missing (voltage: q = 2; 07 P1-11).
    """
    alpha: float = 0.5
    dw_up: float | None = None
    dw_down: float | None = None
    dw_reapproach: float | None = None
    detect_frac: float = 0.05
    drop_frac: float = 0.15
    jump_frac: float = 0.08
    lock_bw_hz: float = 3000.0
    lost_frac: float = 0.15
    lost_steps: int = 20
    max_retries: int = 3
    w_start: float | list = 0.0
    w_lo: float | list = 0.0
    w_hi: float | list = 1.0
    rings: list | None = None
    start_delay_s: float = 50e-6
    law: str = "voltage"
    side: str = "red"


def monitor_mean(obs: Observation, prev: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block mean of the monitor samples released since the last step, (B, N) float, with the valid mask (at least
    one sample and monitor_valid) and the clip flag. Rings without a valid sample keep ``prev`` (or the latest)."""
    s = np.asarray(obs.sensors["tx.monitor.samples"], dtype=np.float64)
    cnt = np.asarray(obs.sensors["tx.monitor.count"], dtype=np.int64)
    m = s.shape[-1]
    sel = np.arange(m)[None, None, :] < cnt[:, :, None]
    tot = np.where(sel, s, 0.0).sum(axis=2)
    valid = cnt > 0
    if "monitor_valid" in obs.sensors:
        valid &= np.asarray(obs.sensors["monitor_valid"], dtype=bool)
    y = np.where(valid, tot / np.maximum(cnt, 1), np.nan)
    fallback = np.asarray(obs.sensors["tx.monitor.latest"], dtype=np.float64) if prev is None else prev
    y = np.where(valid, y, fallback)
    clip = np.asarray(obs.sensors["adc_overrange"], dtype=bool) if "adc_overrange" in obs.sensors \
        else np.zeros_like(valid)
    return y, valid, clip


class RampFSM:
    """The per-ring acquisition state machine of the module docstring, vectorized over (B, N). Shared by
    :class:`RampAcquire`, the bit-statistics lock and the WDM supervisor.

    A supervisor drives it with :meth:`start` (search in a window), :meth:`park` (hold a drive) and reads
    ``state``, ``y_pk``, ``w_pk``, ``y_bg``, ``code``. :meth:`step` takes the block-mean monitor (B, N) and returns the
    heater codes (B, N) int64. Rings whose ``active`` is False keep their code and state.
    """

    def __init__(self, spec: ControllerSpec, prm: RampParams, design: NominalDesign | None = None):
        self.spec = spec
        self.prm = prm
        d = nominal_design(spec) if design is None else design
        self.design = d
        B, N = spec.n_batch, spec.n_rings
        self.B, self.N = B, N
        self.code_min = int(spec.heater_dac.code_min)
        self.code_max = int(spec.heater_dac.code_max)
        self.q = d.drive_q if d.drive_q is not None else drive_exponent(prm.law)
        fs = float(spec.adc.full_scale_codes)
        self.fs = fs
        fw = d.fwhm_w
        #: configured up-sweep step (scalar); ``dw_up`` is the (B, N) step in use, which a supervisor may set per world
        #: or ring (the WDM supervisor maps at one step and acquires at another, world by world)
        self.dw_up0 = float(prm.dw_up if prm.dw_up is not None else (fw / 3.5 if fw else 0.004))
        self._dw_up = np.full((B, N), self.dw_up0)
        self.dw_down = prm.dw_down if prm.dw_down is not None else (fw / 10.0 if fw else 0.0015)
        self.dw_re = prm.dw_reapproach if prm.dw_reapproach is not None else (fw / 5.0 if fw else 0.003)
        self.detect = prm.detect_frac * fs
        self.jump = prm.jump_frac * fs
        self.drop_min = 0.01 * fs
        self.g_loop = 2.0 * math.pi * prm.lock_bw_hz * spec.period_s
        if self.g_loop >= 1.0:
            raise ValueError(f"lock_bw_hz = {prm.lock_bw_hz} is too high for the {spec.rate_hz:.4g} Hz controller")
        z = lambda v=0.0, dt=np.float64: np.full((B, N), v, dtype=dt)
        self.state = z(ST_IDLE, np.int64)
        self.active = z(True, bool)
        if prm.rings is not None:
            self.active[:] = False
            self.active[:, list(prm.rings)] = True
        per_ring = lambda v: np.broadcast_to(np.asarray(v, dtype=np.float64), (N,))[None, :].repeat(B, axis=0)
        self.w = per_ring(prm.w_start).copy()
        self.w_meas = self.w.copy()           # drive that the current measurement reflects (one period of lag)
        self.code = code_of_w(self.w, self.code_max, self.q, self.code_min)
        self.u = self.code.astype(np.float64)  # lock integrator (codes)
        self.w_lo = per_ring(prm.w_lo).copy()
        self.w_hi = per_ring(prm.w_hi).copy()
        self.w_up0 = self.w.copy()
        self.park_w = self.w.copy()
        self.dw_down_max = 6.0 * d.fwhm_w if d.fwhm_w else 0.1   # longest red branch scanned down (strong
        #                                                           self-heating: G P_peak up to ~5 FWHM)
        self.y_prev = z(np.nan)
        self.y_floor = z(np.inf)
        self.floor_at_max = z(np.inf)
        self.y_max = z(-np.inf)
        self.w_max = z(np.nan)
        self.y_pk = z(np.nan)
        self.w_pk = z(np.nan)
        self.y_bg = z(np.nan)
        self.w_event = z(np.nan)
        self.y_land = z(np.nan)
        self.setpoint = z(np.nan)
        self.slope_code = z(np.nan)           # dy/dcode at the set-point, red side (< 0)
        self.ki = z(0.0)
        self.jumps = z(0, np.int64)
        self.w_jump = z(np.nan)
        self.jumped = z(False, bool)
        self.bistable = z(False, bool)        # the resonance was found by a jump (07 P0-1 bistability)
        self.partial = z(False, bool)         # the peak lies below w = 0: y_pk is a lower bound
        self.w_jump_ev = z(np.nan)
        self.lost = z(0, np.int64)
        self.retries = z(0, np.int64)
        self.steps = z(0, np.int64)
        self.clipped = z(False, bool)
        self.locked = z(False, bool)
        self.saturated = z(False, bool)
        # MAP mode (supervisors, factory calibration): record every resonance passed by the up-sweep
        E = 2 * spec.n_lines + 4
        self.map_mode = z(False, bool)
        self.map_done = z(False, bool)
        self.skip = z(0, np.int64)               # COUNT search: resonances to pass before the one to take
        self.map_target = z(spec.n_lines + 1, np.int64)
        self.ev_w = np.full((B, N, E), np.nan)
        self.ev_y = np.full((B, N, E), np.nan)
        self.ev_floor = np.full((B, N, E), np.nan)
        self.ev_wfloor = np.full((B, N, E), np.nan)   # drive of the floor before each resonance (between lines)
        self.ev_n = z(0, np.int64)
        self.w_floor = z(np.nan)
        self.w_floor_at_max = z(np.nan)
        self.y_start = z(np.nan)
        self.buf_w = np.full((B, N, _BUF), np.nan)
        self.buf_y = np.full((B, N, _BUF), np.nan)
        self.buf_n = z(0, np.int64)
        # resonances an UP search passed outside its window (drive of each maximum), since the search started: a
        # supervisor aligns them with a stored map to measure the comb drift and re-centre the window
        self.sk_w = np.full((B, N, E), np.nan)
        self.sk_n = z(0, np.int64)

    @property
    def dw_up(self) -> np.ndarray:
        """Up-sweep (and park) step per controller period in drive units, (B, N). Assigning a scalar or an array sets
        every element; ``fsm.dw_up[b] = x`` sets one world."""
        return self._dw_up

    @dw_up.setter
    def dw_up(self, v) -> None:
        self._dw_up = np.broadcast_to(np.asarray(v, dtype=np.float64), (self.B, self.N)).copy()

    # ---- supervisor interface ---------------------------------------------------------------------------------
    def start(self, mask, w_from=None, w_lo=None, w_hi=None, skip=0):
        """Begin a search (UP) on the rings in ``mask`` from drive ``w_from`` (default: their current drive) inside
        [w_lo, w_hi] (default: the params' window), passing ``skip`` resonances before taking one (COUNT search:
        robust to a drift of the whole resonance comb, as long as the start state is the one the count refers to)."""
        m = np.asarray(mask, dtype=bool) & self.active
        self.skip = np.where(m, skip, self.skip)
        if w_from is not None:
            self.w = np.where(m, w_from, self.w)
        if w_lo is not None:
            self.w_lo = np.where(m, w_lo, self.w_lo)
        if w_hi is not None:
            self.w_hi = np.where(m, w_hi, self.w_hi)
        self.w = np.where(m, np.maximum(self.w, 0.0), self.w)
        if not getattr(self, "_mapping_call", False):
            self.map_mode = np.where(m, False, self.map_mode)
        self._enter_up(m)
        self.retries = np.where(m, 0, self.retries)
        self.jumps = np.where(m, 0, self.jumps)
        self._apply_code(m)

    def start_map(self, mask, w_from=0.0, w_hi=None, n_events=None):
        """Begin a MAP sweep on the rings in ``mask``: ramp up from ``w_from`` and record every resonance passed
        (drive ``ev_w``, landing level ``ev_y``, floor before it ``ev_floor``) until ``n_events`` (default L + 1: one
        full FSR) are recorded or the drive passes ``w_hi``; then HOLD with ``map_done``."""
        m = np.asarray(mask, dtype=bool) & self.active
        self.map_mode = np.where(m, True, self.map_mode)
        self.map_done = np.where(m, False, self.map_done)
        if n_events is not None:
            self.map_target = np.where(m, int(n_events), self.map_target)
        self.ev_w[m] = np.nan
        self.ev_y[m] = np.nan
        self.ev_floor[m] = np.nan
        self.ev_n = np.where(m, 0, self.ev_n)
        self.y_start = np.where(m, np.nan, self.y_start)
        self._mapping_call = True
        try:
            self.start(m, w_from=w_from, w_lo=0.0, w_hi=1.0 if w_hi is None else w_hi)
        finally:
            self._mapping_call = False

    def park(self, mask, w):
        """Hold the rings in ``mask`` at drive ``w`` (state HOLD)."""
        m = np.asarray(mask, dtype=bool)
        self.w = np.where(m, w, self.w)
        self.state = np.where(m, ST_HOLD, self.state)
        self.locked = np.where(m, False, self.locked)
        self._apply_code(m)

    def park_from_below(self, mask, w):
        """Move the rings in ``mask`` to drive ``w`` approaching from below: drop to 0 if above, then ramp up by
        dw_up per step (state PARK, then HOLD). Under self-heating bistability the state reached depends on the
        approach direction; from below it is the state an up-sweep recorded at ``w``."""
        m = np.asarray(mask, dtype=bool) & self.active
        self.park_w = np.where(m, w, self.park_w)
        self.w = np.where(m & (self.w > w), 0.0, self.w)
        self.state = np.where(m, ST_PARK, self.state)
        self.locked = np.where(m, False, self.locked)
        self.map_mode = np.where(m, False, self.map_mode)
        self._apply_code(m)

    def _apply_code(self, m):
        c = code_of_w(self.w, self.code_max, self.q, self.code_min)
        self.code = np.where(m, c, self.code)
        self.u = np.where(m, self.code.astype(np.float64), self.u)

    def _enter_up(self, m):
        self.state = np.where(m, ST_UP, self.state)
        self.partial = np.where(m, False, self.partial)
        self.w_up0 = np.where(m, self.w, self.w_up0)
        self.y_floor = np.where(m, np.inf, self.y_floor)
        self.floor_at_max = np.where(m, np.inf, self.floor_at_max)
        self.y_max = np.where(m, -np.inf, self.y_max)
        self.w_max = np.where(m, np.nan, self.w_max)
        self.jumped = np.where(m, False, self.jumped)
        self.locked = np.where(m, False, self.locked)
        self.steps = np.where(m, 0, self.steps)
        self.sk_n = np.where(m, 0, self.sk_n)

    def _enter_down(self, m):
        self.state = np.where(m, ST_DOWN, self.state)
        self.bistable = np.where(m, self.jumped, self.bistable)
        self.w_jump_ev = np.where(m, np.where(self.jumped, self.w_jump, self.w_max), self.w_jump_ev)
        self.w_event = np.where(m, self.w_max, self.w_event)
        self.y_land = np.where(m, self.y_max, self.y_land)
        self.y_bg = np.where(m, np.where(np.isfinite(self.floor_at_max), self.floor_at_max, self.y_floor),
                             self.y_bg)
        self.y_max = np.where(m, -np.inf, self.y_max)
        self.w_max = np.where(m, np.nan, self.w_max)
        self.buf_n = np.where(m, 0, self.buf_n)
        self.steps = np.where(m, 0, self.steps)

    def _enter_reapproach(self, m):
        self._set_peak(m)
        self.state = np.where(m, ST_REAPPROACH, self.state)

    def _set_peak(self, m):
        """The DOWN scan's maximum is the peak: set y_pk, w_pk, the background, the set-point and the slope."""
        self.y_pk = np.where(m, self.y_max, self.y_pk)
        self.w_pk = np.where(m, self.w_max, self.w_pk)
        self.y_bg = np.where(m, np.minimum(self.y_bg, self.y_floor), self.y_bg)
        self.setpoint = np.where(m, self.y_bg + self.prm.alpha * (self.y_pk - self.y_bg), self.setpoint)
        if np.any(m):
            self._measure_slope(m)
        self.jumped = np.where(m, False, self.jumped)
        self.y_max = np.where(m, -np.inf, self.y_max)
        self.steps = np.where(m, 0, self.steps)

    def _measure_slope(self, m):
        """dy/dcode at the set-point from the DOWN scan points on the red side (before the maximum)."""
        for b, n in zip(*np.nonzero(m)):
            k = int(min(self.buf_n[b, n], _BUF))
            w = self.buf_w[b, n, :k]
            y = self.buf_y[b, n, :k]
            r = self.setpoint[b, n]
            s_code = np.nan
            if k >= 3 and np.isfinite(r):
                imax = int(np.argmax(y))
                wr, yr = w[:imax + 1], y[:imax + 1]          # red branch: w decreasing, y increasing
                idx = np.flatnonzero((yr[:-1] <= r) & (yr[1:] > r))
                if idx.size:
                    i = int(idx[-1])
                    lo, hi = max(i - 2, 0), min(i + 3, wr.size - 1)
                    dw = wr[hi] - wr[lo]
                    if dw < 0:
                        s_w = (yr[hi] - yr[lo]) / dw              # dy/dw < 0 on the red side
                        wc = 0.5 * (wr[hi] + wr[lo])
                        x = max(wc, 1e-9) ** (1.0 / self.q)
                        dwdc = self.q * x ** (self.q - 1) / self.code_max
                        s_code = s_w * dwdc
                elif imax >= 2:
                    # the scan started nearer the peak than the set-point: use the mean slope of the red branch
                    dw = wr[imax] - wr[0]
                    if dw < 0:
                        s_w = (yr[imax] - yr[0]) / dw
                        x = max(0.5 * (wr[0] + wr[imax]), 1e-9) ** (1.0 / self.q)
                        s_code = s_w * self.q * x ** (self.q - 1) / self.code_max
            if not (np.isfinite(s_code) and s_code < 0):
                # fallback: Lorentzian slope at x = 1 over the measured half-width of the red branch in codes
                hw = max(abs(self.w_event[b, n] - self.w_max[b, n]), 4 * self.dw_down)
                x = max(self.w_max[b, n], 1e-9) ** (1.0 / self.q)
                hw_code = hw / (self.q * x ** (self.q - 1) / self.code_max)
                s_code = -0.5 * (self.y_max[b, n] - self.y_bg[b, n]) / max(hw_code, 1.0)
            self.slope_code[b, n] = s_code
            self.ki[b, n] = self.g_loop / abs(s_code)

    def _enter_lock(self, m):
        self.state = np.where(m, ST_LOCK, self.state)
        self.u = np.where(m, self.code.astype(np.float64), self.u)
        self.lost = np.where(m, 0, self.lost)
        self.locked = np.where(m, True, self.locked)
        self.steps = np.where(m, 0, self.steps)

    def _retry(self, m):
        """Search again from the current drive (a lost peak), FAIL after max_retries."""
        retry = m & (self.retries < self.prm.max_retries)
        fail = m & ~retry
        self.retries = np.where(retry, self.retries + 1, self.retries)
        self.w = np.where(retry, np.maximum(self.w - 4.0 * self.dw_up, self.w_lo), self.w)
        self._enter_up(retry)
        self.state = np.where(fail, ST_FAIL, self.state)
        self.locked = np.where(fail, False, self.locked)

    def _fail_or_retry(self, m):
        retry = m & (self.retries < self.prm.max_retries)
        fail = m & ~retry
        self.retries = np.where(retry, self.retries + 1, self.retries)
        self.w = np.where(retry, self.w_lo, self.w)
        self._enter_up(retry)
        self.state = np.where(fail, ST_FAIL, self.state)
        self.locked = np.where(fail, False, self.locked)
        self.w = np.where(fail, self.w_lo, self.w)

    # ---- one controller step ----------------------------------------------------------------------------------
    def step(self, y: np.ndarray, valid: np.ndarray, clipped: np.ndarray | None = None) -> np.ndarray:
        """Advance every active ring by one controller period with the block-mean monitor ``y`` (codes)."""
        prm = self.prm
        go = self.active & np.asarray(valid, dtype=bool)
        if clipped is not None:
            self.clipped |= go & np.asarray(clipped, dtype=bool) & (self.state != ST_LOCK)
        y = np.where(go, y, self.y_prev)
        dy = y - self.y_prev
        self.steps = np.where(go, self.steps + 1, self.steps)
        new_jump = go & np.isfinite(dy) & (dy > self.jump) & np.isin(self.state, (ST_UP, ST_REAPPROACH))
        self.jumps = np.where(new_jump, self.jumps + 1, self.jumps)
        self.w_jump = np.where(new_jump, self.w_meas, self.w_jump)
        self.jumped |= new_jump
        wm = self.w_meas

        # ---- UP: search
        m = go & (self.state == ST_UP)
        if np.any(m):
            # a resonance is a maximum that FOLLOWS a floor: a new minimum restarts the maximum
            newmin = m & (y < self.y_floor)
            self.y_floor = np.where(newmin, y, self.y_floor)
            self.w_floor = np.where(newmin, wm, self.w_floor)
            self.y_max = np.where(newmin, y, self.y_max)
            self.w_max = np.where(newmin, wm, self.w_max)
            self.floor_at_max = np.where(newmin, y, self.floor_at_max)
            self.w_floor_at_max = np.where(newmin, wm, self.w_floor_at_max)
            newmax = m & (y > self.y_max)
            self.y_max = np.where(newmax, y, self.y_max)
            self.w_max = np.where(newmax, wm, self.w_max)
            self.floor_at_max = np.where(newmax, self.y_floor, self.floor_at_max)
            self.w_floor_at_max = np.where(newmax, self.w_floor, self.w_floor_at_max)
            rise = self.y_max - self.floor_at_max
            drop = np.maximum(self.drop_min, prm.drop_frac * rise)
            # the maximum must come from the sweep, not from a transient at its start
            inwin = ((self.w_max >= self.w_lo) & (self.w_max <= self.w_hi)
                     & (self.w_max >= self.w_up0 + 1.5 * self.dw_up))
            found = m & (rise >= self.detect) & (y <= self.y_max - drop) & inwin
            self.y_start = np.where(m & np.isnan(self.y_start), y, self.y_start)
            # COUNT search: pass this resonance and keep going
            passed = found & ~self.map_mode & (self.skip > 0)
            if np.any(passed):
                self.skip = np.where(passed, self.skip - 1, self.skip)
                self.y_floor = np.where(passed, y, self.y_floor)
                self.w_floor = np.where(passed, wm, self.w_floor)
                self.y_max = np.where(passed, y, self.y_max)
                self.w_max = np.where(passed, wm, self.w_max)
                self.floor_at_max = np.where(passed, y, self.floor_at_max)
                self.jumped = np.where(passed, False, self.jumped)
                found = found & ~passed
            mapped = found & self.map_mode
            if np.any(mapped):
                bi, ni = np.nonzero(mapped)
                k = np.minimum(self.ev_n[bi, ni], self.ev_w.shape[2] - 1)
                self.ev_w[bi, ni, k] = self.w_max[bi, ni]
                self.ev_y[bi, ni, k] = self.y_max[bi, ni]
                self.ev_floor[bi, ni, k] = self.floor_at_max[bi, ni]
                self.ev_wfloor[bi, ni, k] = self.w_floor_at_max[bi, ni]
                self.ev_n = np.where(mapped, self.ev_n + 1, self.ev_n)
                # keep searching from here: the next resonance must follow a new floor
                self.y_floor = np.where(mapped, y, self.y_floor)
                self.w_floor = np.where(mapped, wm, self.w_floor)
                self.y_max = np.where(mapped, y, self.y_max)
                self.w_max = np.where(mapped, wm, self.w_max)
                self.floor_at_max = np.where(mapped, y, self.floor_at_max)
                full = mapped & (self.ev_n >= self.map_target)
                self.state = np.where(full, ST_HOLD, self.state)
                self.map_done |= full
                found = found & ~mapped
            # a passed resonance outside the window: forget it and keep searching
            skip = m & (rise >= self.detect) & (y <= self.y_max - drop) & ~inwin
            rec = skip & (self.w_max >= self.w_up0 + 1.5 * self.dw_up)    # not a transient at the search start
            if np.any(rec):
                bi, ni = np.nonzero(rec)
                kk = np.minimum(self.sk_n[bi, ni], self.sk_w.shape[2] - 1)
                self.sk_w[bi, ni, kk] = self.w_max[bi, ni]
                self.sk_n = np.where(rec, self.sk_n + 1, self.sk_n)
            self.y_max = np.where(skip, y, self.y_max)
            self.w_max = np.where(skip, wm, self.w_max)
            self.floor_at_max = np.where(skip, y, self.floor_at_max)
            self.y_floor = np.where(skip, y, self.y_floor)
            self._enter_down(found)
            cont = m & ~found & (self.state == ST_UP)
            self.w = np.where(cont, self.w + self.dw_up, self.w)
            # a maximum inside the window needs a few more steps for its fall to be confirmed
            over = cont & (self.w > self.w_hi + np.where(self.map_mode, 0.0, 3.0 * self.dw_up))
            map_over = over & self.map_mode
            self.state = np.where(map_over, ST_HOLD, self.state)
            self.map_done |= map_over
            self._fail_or_retry(over & ~self.map_mode)

        # ---- DOWN: true peak on the red branch
        m = go & (self.state == ST_DOWN) & (self.steps > 1)
        m0 = go & (self.state == ST_DOWN) & (self.steps <= 1)
        self.w = np.where(m0, self.w - self.dw_down, self.w)
        if np.any(m):
            k = np.minimum(self.buf_n, _BUF - 1)
            bi, ni = np.nonzero(m)
            self.buf_w[bi, ni, k[bi, ni]] = wm[bi, ni]
            self.buf_y[bi, ni, k[bi, ni]] = y[bi, ni]
            self.buf_n = np.where(m, self.buf_n + 1, self.buf_n)
            self.y_floor = np.where(m, np.minimum(self.y_floor, y), self.y_floor)
            newmax = m & (y > self.y_max)
            self.y_max = np.where(newmax, y, self.y_max)
            self.w_max = np.where(newmax, wm, self.w_max)
            rise = self.y_max - self.y_bg
            fall = np.maximum(self.drop_min, prm.drop_frac * rise)
            peak = m & (rise >= self.detect) & (y <= self.y_max - fall)
            self._enter_reapproach(peak)
            cont = m & ~peak
            # the heater cannot cool: when the red branch reaches w = 0 (or runs longer than dw_down_max) before its
            # peak, the ring is still on the red side; take the highest point seen as a partial peak and lock
            end = cont & ((self.w - self.dw_down < 0.0) | (self.w - self.dw_down < self.w_event - self.dw_down_max))
            rising = end & (y >= self.y_max - self.drop_min) & (self.y_max - self.y_bg >= self.detect)
            if np.any(rising):
                self.partial = np.where(rising, True, self.partial)
                self._set_peak(rising)
                self._enter_lock(rising)
            self._retry(end & ~rising)
            cont = cont & ~end
            self.w = np.where(cont, self.w - self.dw_down, self.w)

        # ---- REAPPROACH: back to the red side of the SAME resonance. Bistable case: the jump recurs near the drive
        # where the UP sweep jumped (the fall-off of the DOWN scan may have left the ring on the hot branch of the
        # next bluer line, whose red flank must not be taken for this one). Continuous case: a high maximum near
        # w_pk, then a fall.
        m = go & (self.state == ST_REAPPROACH)
        if np.any(m):
            tol = 4.0 * self.dw_up
            jb = m & self.bistable & new_jump & (wm >= self.w_jump_ev - tol)
            near = m & ~self.bistable & (wm >= self.w_pk - 2.0 * self.dw_re)
            newmax = near & (y > self.y_max)
            self.y_max = np.where(newmax, y, self.y_max)
            high = self.y_max >= self.y_bg + 0.6 * (self.y_pk - self.y_bg)
            jc = near & high & (y < self.y_max - self.drop_min)
            red = (jb | jc) & (self.steps > 1)
            self._enter_lock(red)
            cont = m & ~red
            self.w = np.where(cont, self.w + self.dw_re, self.w)
            limit = np.where(self.bistable, self.w_jump_ev, self.w_pk) + tol
            over = cont & (self.w > limit)
            self._retry(over)

        # ---- LOCK: integral loop on the red side
        m = go & (self.state == ST_LOCK)
        if np.any(m):
            e = y - self.setpoint
            self.u = np.where(m, self.u + self.ki * e, self.u)
            lo, hi = float(self.code_min), float(self.code_max)
            self.saturated = np.where(m, (self.u <= lo) | (self.u >= hi), self.saturated)
            self.u = np.where(m, np.clip(self.u, lo, hi), self.u)
            self.code = np.where(m, np.rint(self.u).astype(np.int64), self.code)
            self.w = np.where(m, w_of_code(self.code, self.code_max, self.q), self.w)
            low = m & (y < self.y_bg + prm.lost_frac * (self.y_pk - self.y_bg))
            self.lost = np.where(low, self.lost + 1, np.where(m, 0, self.lost))
            gone = m & ((self.lost >= prm.lost_steps) | (self.saturated & (self.steps > prm.lost_steps)))
            if np.any(gone):
                # restart the search a little blue of the peak, in a window around it
                self.locked = np.where(gone, False, self.locked)
                lo_w = np.maximum(self.w_pk - 6.0 * self.dw_up, 0.0)
                self.w = np.where(gone, lo_w, self.w)
                self.state = np.where(gone, ST_UP, self.state)
                self.retries = np.where(gone, self.retries + 1, self.retries)
                self._enter_up(gone)

        # ---- PARK: ramp up to the park drive
        m = self.active & (self.state == ST_PARK)
        if np.any(m):
            self.w = np.where(m, np.minimum(self.w + self.dw_up, self.park_w), self.w)
            self.state = np.where(m & (self.w >= self.park_w), ST_HOLD, self.state)

        # ---- drive -> code for the sweeping states
        sweep = (go & np.isin(self.state, (ST_UP, ST_DOWN, ST_REAPPROACH, ST_FAIL))) | m
        sweep |= self.active & (self.state == ST_HOLD) & m
        self.w = np.clip(self.w, 0.0, 1.0)
        c = code_of_w(self.w, self.code_max, self.q, self.code_min)
        self.code = np.where(sweep, c, self.code)
        self.w_meas = np.where(go, self.w, self.w_meas)
        self.y_prev = np.where(go, y, self.y_prev)
        return self.code.copy()

    # ---- estimates for telemetry ------------------------------------------------------------------------------
    def detuning_est_pm(self, y: np.ndarray) -> np.ndarray:
        """Resonance-minus-line detuning estimate (pm) of locked rings on the red side, from the Lorentzian:
        (y - y_bg)/(y_pk - y_bg) = 1/(1 + x^2), x = 2 delta / FWHM; needs the nominal linewidth (else NaN)."""
        fw = self.design.fwhm_m
        if fw is None:
            return np.full((self.B, self.N), np.nan)
        a = np.clip((y - self.y_bg) / np.maximum(self.y_pk - self.y_bg, 1.0), 1e-3, 1.0)
        x = np.sqrt(1.0 / a - 1.0)
        return np.where(self.state == ST_LOCK, 0.5 * x * fw * 1e12, np.nan)


@register("ramp_acquire")
class RampAcquire(BaseController):
    """PAT-3 ramp-and-maximize acquisition with the red-side lock hand-off (module docstring), every ring at once.

    ``controller.params`` = the fields of :class:`RampParams`. Writes ``tx.heater`` only. Telemetry (B, N): state,
    locked, detuning_est_pm, integrator (lock integrator, codes), integrator_saturated, peak_level (codes),
    peak_code, background (codes), setpoint (codes), jumps (bistability jumps seen), clipped (a monitor sample
    clipped during the search).
    """
    name = "ramp_acquire"
    Params = RampParams

    def reset(self, spec: ControllerSpec) -> None:
        super().reset(spec)
        if self.params.side != "red":
            raise ValueError("RampAcquire: only side='red' (the self-heating-stable side) is implemented")
        self.fsm = RampFSM(spec, self.params)
        self.delay_ticks = int(round(self.params.start_delay_s / spec.tick_s))
        self.t0: int | None = None          # tick of the first step after reset (the start delay counts from it)
        self.started = False
        self.y = np.full((spec.n_batch, spec.n_rings), np.nan)

    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        f = self.fsm
        y, valid, clip = monitor_mean(obs, None if np.all(np.isnan(self.y)) else self.y)
        self.y = y
        if self.t0 is None:
            self.t0 = int(obs.tick)
        if not self.started:
            # rings outside params.rings keep their latched register
            app = np.asarray(obs.sensors["tx.heater.applied"], dtype=np.int64)
            f.code = np.where(f.active, f.code, app)
            if obs.tick - self.t0 < self.delay_ticks:
                return {"tx.heater": f.code.copy()}
            f.start(np.ones((f.B, f.N), bool))
            self.started = True
        return {"tx.heater": f.step(y, valid, clip)}

    def telemetry(self) -> dict[str, np.ndarray]:
        f = self.fsm
        pk_code = np.where(np.isfinite(f.w_pk), code_of_w(np.nan_to_num(f.w_pk), f.code_max, f.q), -1)
        return {"state": f.state.astype(float), "locked": f.locked.astype(float),
                "detuning_est_pm": f.detuning_est_pm(self.y), "integrator": f.u.copy(),
                "integrator_saturated": f.saturated.astype(float), "peak_level": f.y_pk.copy(),
                "peak_code": pk_code.astype(float), "background": f.y_bg.copy(), "setpoint": f.setpoint.copy(),
                "jumps": f.jumps.astype(float), "clipped": f.clipped.astype(float)}


@register("acquire_then_lock")
class AcquireThenLock(BaseController):
    """PAT-3 acquisition followed by another lock controller (composition with the controllers of section 6).

    Runs :class:`RampAcquire` until every acquired ring has held its red-side lock for ``handoff_steps`` steps, then
    resets the lock controller with the same spec and forwards every later step to it. The hand-off contract is
    the one of mrm.control.baselines.pi_lock: at the lock controller's reset the ring sits on the stable red branch
    of its line and ``tx.heater.applied`` holds its code. A plain CompositeController cannot do this (its children
    all run from the start and the later child wins), hence this small sequencer.

    ``controller.params`` = {acquire: RampParams fields, lock: {name, params} (default: controllers A's PI lock,
    ``mrm.control.baselines.pi_lock:PiLock``), handoff_steps}. The lock controller is built here, not at reset: the
    runner reads ``latency_ticks`` once, before reset, and the pipeline depth is one for the whole run, so this
    sequencer takes the lock's (the acquisition FSM tolerates the extra delay: it measures one period behind).
    """
    name = "acquire_then_lock"

    def __init__(self, acquire=None, lock=None, handoff_steps: int = 20, lock_controller=None):
        super().__init__()
        self.acquire = RampAcquire(dict(acquire or {}))
        self.lock_spec = dict(lock or {"name": "mrm.control.baselines.pi_lock:PiLock"})
        if lock_controller is None:
            from mrm.control import registry
            lock_controller = registry.create(self.lock_spec["name"], self.lock_spec.get("params"))
        self.lock_ctrl = lock_controller
        self.handoff_steps = int(handoff_steps)
        self.latency_ticks = max(int(getattr(self.lock_ctrl, "latency_ticks", 0) or 0), 0)

    @classmethod
    def from_params(cls, params):
        params = dict(params or {})
        return cls(params.get("acquire"), params.get("lock"), int(params.get("handoff_steps", 20)))

    def reset(self, spec: ControllerSpec) -> None:
        super().reset(spec)
        self.acquire.reset(spec)
        self.handed = False
        self.held = 0

    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        if self.handed:
            return self.lock_ctrl.step(obs)
        out = self.acquire.step(obs)
        f = self.acquire.fsm
        done = np.all((f.state == ST_LOCK) | ~f.active)
        self.held = self.held + 1 if done else 0
        if self.held >= self.handoff_steps:
            self.lock_ctrl.reset(self.spec)
            self.handed = True
        return out

    def telemetry(self) -> dict[str, np.ndarray]:
        from mrm.control.api import controller_telemetry
        tel = controller_telemetry(self.lock_ctrl) if self.handed else self.acquire.telemetry()
        tel = dict(tel)
        tel["handed_off"] = np.full((self.spec.n_batch, self.spec.n_rings), float(self.handed))
        return tel
