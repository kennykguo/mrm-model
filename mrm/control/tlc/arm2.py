"""Second arm (docs/TLC_PORT_SPEC_AND_PLAN.md 6, Q1): disclosed G26 adaptations of the TLC core.

The vendored core (``mrm.control.tlc.core``) stays bit-identical to the ECE496 source; every change for G26 lives
here as a subclass, switched on per run through the adapter's ``arm2`` params, and reported as a separate arm.

Options (all off by default; ``Arm2Options``):

- ``stiff_scale_rel``: the UKF's DC-scale state p (laser power x monitor gain) may only move slowly. On G26 the lock
  point sits at about 0.6 half-widths, where the normalised lock-in signals hardly depend on detuning (zeta varies
  ~1.4e-4 per GHz against ~6e-4 noise), so a level change can be explained either by detuning or by p. In a fast
  thermal transient the filter put it into p (+18 % after the +10 K kick) and parked the ring ~40 pm off. With this
  option p is decorrelated from the other states before every update and its standard deviation capped at
  ``stiff_scale_rel`` x p, so level changes go to detuning; p still follows slow drifts. A ring whose heater sits at
  the floor (``stiff_min_u``) or is flagged heater-limited keeps the source behaviour: it is out of control anyway,
  and a free p stops light from a neighbouring line (A1 pushes ring 0 that far) being read as a blue shift. A ring
  that (re-)entered TRACK less than ``stiff_settle_s`` ago is also left free: acquisition initialises p from one
  level reading, and a stiff p would freeze that guess.
- ``red_wait_s``: the source re-acquires a lost ring by backing off and ramping the heater up, which assumes the ring
  fell to the blue. Under A1, ring 0 is lost to the RED with its heater already at zero: ramping up pushes it over the
  self-heating fold onto the next line (+1 nm). With this option a ring lost red at the floor holds its heater and
  waits for the resonance to come back blue as the aggressor cools. It re-locks when the monitor level reaches the
  level its last calibrated scale p predicts at the operating point (on the red flank, z1 < 0), with that p, so the
  re-locked ring is calibrated at once (the source's acquisition estimate assumes a ring moving red). After
  ``red_wait_s`` it falls back to the source re-acquisition.
- ``fall_level``: a laser power drop (R7: -3 dB) removes most of the self-heating that holds a G26 ring on its hot
  branch, and the resonance jumps ~0.9 nm blue within 0.1 ms, off its line. The source waits for its loss-of-lock
  timer and meanwhile reads the falling level as a red shift, lowering the heater the wrong way. With this option a
  tracking ring whose block-mean monitor level (100 us, not the 4 ms lock-in window, which lags) falls below
  ``fall_level`` x the level its scale p predicts at the operating point for ``fall_steps`` steps, while its heater is
  not at the floor (so it cannot be a red loss; a red excursion as large as the +10 K kick still leaves ~55 % of the
  level), restarts the source acquisition at once from its current heater command: ramp up from the blue side, the
  direction the source acquisition is built for.
- ``scale_rate_max``: a softer version: p may change by at most this fraction per second (clipped after each
  update). A thermal transient (the kick moved p 18 % in 0.5 ms) cannot be absorbed into p, while a slow change of
  a ring's input power (another ring crossing its laser line under the A1 aggressor) still can.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from mrm.control.tlc.core import TLC
from mrm.control.tlc.core.common import Mode
from mrm.control.tlc.core.optics import drop_avg


@dataclass(frozen=True)
class Arm2Options:
    stiff_scale_rel: float = 0.0        # 0: off; else cap on std(p) / p before each UKF update
    scale_rate_max: float = 0.0         # 0: off; else max relative change of p per second (clip after each update)
    stiff_min_u: float = 0.0            # stiffen only rings whose heater command is above this (not at the floor)
    stiff_settle_s: float = 0.0         # stiffen only rings that have tracked this long (p re-learnt after a re-lock)
    red_wait_s: float = 0.0             # 0: off; else max wait (s) for a ring lost to the red at the heater floor
    red_floor_u: float = 0.02           # heater command treated as "at the floor" for red_wait
    fall_level: float = 0.0             # 0: off; else blue fall-off when the monitor drops below this x expected
    fall_steps: int = 2                 # ... (block mean, 100 us) for this many control steps

    @classmethod
    def from_dict(cls, d: dict | None) -> "Arm2Options":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        bad = sorted(set(d) - known)
        if bad:
            raise ValueError(f"arm2: unknown options {bad}; available: {sorted(known)}")
        return cls(**d)


def _stiffen_scale(ukf, rel: float, rings: np.ndarray) -> None:
    """Decorrelate the DC-scale states of ``rings`` (bool mask) from the rest and cap their standard deviation at
    ``rel`` x p."""
    for i in np.where(rings)[0]:
        s = 2 * ukf.N + i
        var = min(ukf.P[s, s], (rel * abs(ukf.x[s])) ** 2)
        ukf.P[s, :] = 0.0
        ukf.P[:, s] = 0.0
        ukf.P[s, s] = var


class Arm2TLC(TLC):
    """The vendored TLC with the selected G26 adaptations (``Arm2Options``)."""

    def __init__(self, pp, cp, *, options: Arm2Options, **kw):
        super().__init__(pp, cp, **kw)
        self.arm2 = options
        self._track_since = np.full(self.N, -np.inf)   # time each ring last entered TRACK (warm start: settled)
        self._fall_count = np.zeros(self.N)
        if options.stiff_scale_rel > 0 or options.scale_rate_max > 0:
            update, ukf, n = self.ukf.update, self.ukf, self.N
            step = options.scale_rate_max * cp.ts

            def arm2_update(m):
                if options.stiff_scale_rel > 0:
                    settled = (self.t - self._track_since) >= options.stiff_settle_s
                    free = (self.u_cmd <= options.stiff_min_u) | self.limit_flag | ~settled
                    _stiffen_scale(ukf, options.stiff_scale_rel, ~free)
                p0 = ukf.x[2 * n:].copy()
                out = update(m)
                if step > 0:
                    ukf.x[2 * n:] = np.clip(ukf.x[2 * n:], p0 * (1 - step), p0 * (1 + step))
                return out

            self.ukf.update = arm2_update

    def _supervise(self, z1, y0, valid, invalid):
        self.y0_block = np.array(y0)                 # block-mean monitor level (diagnostics)
        before = self.modes.copy()
        self._supervise_arm2(z1, y0, valid, invalid, before)
        entered = (self.modes == Mode.TRACK) & (before != Mode.TRACK)
        self._track_since = np.where(entered, self.t, self._track_since)

    def _held_off(self, i):
        return self.modes[i] == Mode.REACQUIRE and self.t < self.acq[i].get("holdoff", -np.inf)

    def _supervise_arm2(self, z1, y0, valid, invalid, before):
        opt = self.arm2
        held = [i for i in range(self.N) if self._held_off(i)]
        for i in held:
            self.modes[i] = Mode.INIT                # no detection while the lock-in window is stale
        try:
            self._supervise_arm2_inner(z1, y0, valid, invalid, before)
        finally:
            for i in held:
                self.modes[i] = Mode.REACQUIRE

    def _supervise_arm2_inner(self, z1, y0, valid, invalid, before):
        opt = self.arm2
        if opt.red_wait_s <= 0:
            super()._supervise(z1, y0, valid, invalid)
            self._fall_check(z1, y0, valid)
            return
        red = (self.ukf.delta - self.sp.targets) > 0
        floor = self.u_cmd <= opt.red_floor_u
        waiting = [i for i in range(self.N) if self.modes[i] == Mode.REACQUIRE and self.acq[i].get("red_wait")]
        for i in waiting:
            self.modes[i] = Mode.INIT                # the source supervisor leaves INIT alone
        super()._supervise(z1, y0, valid, invalid)
        for i in waiting:
            self.modes[i] = Mode.REACQUIRE
            self._red_wait_step(i, z1, y0, valid)
        self._fall_check(z1, y0, valid)
        lost = (before == Mode.TRACK) & (self.modes == Mode.REACQUIRE) & red & floor
        for i in np.where(lost)[0]:
            p_i = self.ukf.x[2 * self.N + i]
            level = p_i * drop_avg(self.sp.targets[i], self.cal.hwhm[i], self.cal.mod)
            self.acq[i] = dict(red_wait=True, t0=self.t, p=p_i, level=level)
            self.diag.alarm(self.t, i, "red_wait")

    def _fall_check(self, z1, y0, valid):
        opt = self.arm2
        if opt.fall_level <= 0:
            return
        n = self.N
        expect = self.ukf.x[2 * n:] * drop_avg(self.sp.targets, self.cal.hwhm, self.cal.mod)
        fallen = ((self.modes == Mode.TRACK) & valid & (self.u_cmd > opt.red_floor_u)
                  & (y0 < opt.fall_level * expect))
        self._fall_count = np.where(fallen, self._fall_count + 1, 0)
        for i in np.where(self._fall_count >= opt.fall_steps)[0]:
            self.diag.alarm(self.t, i, "blue_fall")
            self.ukf.init_ring(i, self.sp.targets[i], var=100.0, p=self.ukf.x[2 * n + i])
            # ramp at once, but arm the detector only once the lock-in window holds no pre-fall samples
            self.acq[i] = dict(phase="ramp", zmax=0.0, zf=0.0, n=0, holdoff=self.t + 1.25 * self.cp.lockin_window)
            self.modes[i] = Mode.REACQUIRE
            self.lol_count[i] = 0
            self._fall_count[i] = 0

    def _red_wait_step(self, i, z1, y0, valid):
        """Hold the heater; the ring is back at its operating point when the monitor level, falling from the red
        tail as the aggressor cools, reaches the level the calibrated scale p predicts there, on the red flank."""
        st = self.acq[i]
        if self.t - st["t0"] > self.arm2.red_wait_s:          # out of time: the source re-acquisition
            back = self.cp.reacquire_backoff / self.cal.dc_gain_open()[i]
            self.acq[i] = dict(phase="back", u_floor=max(self.u_cmd[i] - back, 0.0))
            return
        if valid[i] and z1[i] < 0 and y0[i] >= st["level"]:
            self.ukf.init_ring(i, self.sp.targets[i], var=4.0, p=st["p"])
            self.ukf.hist.clear()
            self.modes[i] = Mode.TRACK
            self.lol_count[i] = 0
            self.diag.alarm(self.t, i, "red_wait_locked")

    def _planned_command(self, i):
        if self.modes[i] == Mode.REACQUIRE and self.acq[i].get("red_wait"):
            return self.u_cmd[i]                     # hold: the resonance comes back as the aggressor cools
        return super()._planned_command(i)          # (a held-off ring ramps as in the source "ramp" phase)
