"""Extremum seeking on the drop monitor (ARCHITECTURE section 6 item 5; 04 C2 perturb-and-observe, C5/C15 sinusoidal
extremum-seeking control). Both maximize the drop-port monitor, i.e. they seek the peak of the drop line.

Perturb and observe (:class:`ExtremumSeekingPO`, 04 C2, [HAT24])
--------------------------------------------------------------
Per round, three dwell periods of ``settle_steps + avg_steps`` steps at u, u + delta and u - delta (the last
``avg_steps`` block means averaged: y0, y+, y-), then u moves to the best of the three (``mode='argmax'``) or by a
gradient step u += gain * (y+ - y-)/(2 delta) * delta^2 / max(|y+ + y- - 2 y0|, floor) (``mode='newton'``, a damped
Newton step from the same three points), clamped to ``max_step_codes``. Tracking rate <= max_step / round time
(04 C2 failure modes); a disturbance inside a round reads as a gradient.

Sinusoidal ES (:class:`ExtremumSeekingSine`, Krstic-Wang form)
-----------------------------------------------------------
u = u_hat + a sin(w k) with w = 2 pi f_d T; J = monitor block mean; high-pass J_hp = J - J_lp (first-order low-pass,
corner ``hpf_hz``); demodulation xi = J_hp sin(w k - phi), phi the phase lag of the plant at f_d (thermal poles and
the sample delay); u_hat += gain_codes * xi / J_lp (normalized by the DC level, so the gain does not depend on the
laser power). Bandwidth << f_d (04 constraint 1).

Under self-heating (07 P0-1)
---------------------------
The peak is open-loop stable, but at g26 nominal (bistability ratio ~6) the fold sits only ~9 pm blue of it: every
probe or dither that reaches past it makes the ring fall onto the cold branch, the monitor collapses and the
controller would chase the wrong side. Both controllers detect the fall (monitor drop by ``fall_frac`` of the recent
DC to below ``fall_level``), ramp the heater up past a guard until the monitor peaked and fell back (as pi_lock), and
resume. Measured at g26 nominal with the default parameters (controllers review 2026-10-16):

- es_po: its +-delta probes (~6 pm at the peak) reach the fold only marginally. On the plant of the params review
  (2026-10-16: secant beta, heater_self_ratio 1.132) it does not fall in the 8 ms harness run of
  tests/verification/test_ctrl_a_es.py; on the engine (settled warm start at the oracle lock codes, 0.4 s) rings 4-7
  fall once before 40 ms and every ring then parks at the drop peak (-0.3 to -1.4 pm, OMA penalty 13.0-13.2 dB;
  runs/params_fix/es_long.log). Before the review (tangent beta, ratio 1.158) it fell once per ring in the harness
  and 2-4 times per ring on the engine, and rings 4-6 parked on the cold branch ~1 nm blue of the line: whether the
  probes cross the fold moves with the droop and with s(T) at the lock at the 2 % level. It does not keep cycling.
- es_sine: its dither does not reach the fold and it never falls. It approaches the peak slowly, at 0.77-0.95
  pm/ms from the warm start (+97 to +121 pm), and parks at the drop peak (-0.6 to -1.1 pm) after 100-160 ms, where
  the OMA penalty is ~13 dB; a run shorter than that sees a transient (docs/handoff/controller_scorecard.md counts
  its rings as drifting, not locked).

The peak is the wrong point for a modulator (04 C5). Without bistability (low laser power) both converge to it.

Hand-off contract as pi_lock (stable branch; heater code from ``init_codes`` or ``tx.heater.applied``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mrm.control.api import BaseController, ControllerSpec, Observation
from mrm.control.baselines.pi_lock import block_inputs, per_ring
from mrm.control.registry import register

__all__ = ["ExtremumSeekingPO", "ExtremumSeekingSine", "E_RUN", "E_RECAPTURE", "E_FAULT"]

E_RUN, E_RECAPTURE, E_FAULT = 3, 4, 5      # LOCK / RECAPTURE / FAULT numbering of pi_lock


@dataclass
class _Common:
    init_codes: object = None
    fall_frac: float = 0.35
    fall_level: float = 0.4
    jump_steps: int = 3
    guard_codes: float = 300.0
    recapture_step_codes: float = 32.0
    recapture_span_codes: float = 8000.0
    recapture_level: float = 0.3
    recapture_drop: float = 0.02
    latency_ticks: int = 0


class _Ring:
    """Fall-off detection and recapture shared by both ES forms (state per ring)."""

    def __init__(self, J):
        self.st = E_RUN
        self.u = 0.0
        self.y_dc = 0.0
        self.hist = []
        self.J = J
        self.u_guard = 0.0
        self.y_seen = 0.0
        self.n_fall = 0

    def push(self, y):
        self.hist.insert(0, y)
        del self.hist[self.J:]

    def fell(self, y, p) -> bool:
        if not self.hist or self.y_dc <= 0.0:
            return False
        return (max(self.hist) - y) > p.fall_frac * self.y_dc and y < p.fall_level * self.y_dc

    def start_recapture(self, p):
        self.st = E_RECAPTURE
        self.n_fall += 1
        self.u_guard = self.u + p.guard_codes
        self.y_seen = 0.0

    def recapture(self, y, p, hi) -> bool:
        """One RECAPTURE step; True when the ring is red of the peak again."""
        if self.u >= self.u_guard and y > self.y_seen:
            self.y_seen = y
        if self.y_seen > p.recapture_level * self.y_dc and y < (1.0 - p.recapture_drop) * self.y_seen:
            self.st = E_RUN
            return True
        if self.u >= hi or self.u >= self.u_guard + p.recapture_span_codes:
            self.st = E_FAULT                    # give up: back to the code at the fall (see pi_lock)
            self.u = self.u_guard - p.guard_codes
        else:
            self.u = min(self.u + p.recapture_step_codes, hi)
        return False


class _ESBase(BaseController):
    """Shared plumbing: hand-off code, block means, per-ring loop, telemetry."""

    def __init__(self, params=None, **overrides):
        super().__init__(params, **overrides)
        self.latency_ticks = int(self.params.latency_ticks)

    def reset(self, spec: ControllerSpec) -> None:
        super().reset(spec)
        sh = (spec.n_batch, spec.n_rings)
        self.code_lo, self.code_hi = float(spec.heater_dac.code_min), float(spec.heater_dac.code_max)
        self.init_codes = per_ring(self.params.init_codes, sh, "init_codes")
        self.rings = [[self._new_ring() for _ in range(sh[1])] for _ in range(sh[0])]
        self.first = True
        self.y = np.zeros(sh)

    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        spec = self._require_spec()
        y, ok = block_inputs(obs, self.y)
        self.y = y
        if self.first:
            applied = (np.asarray(obs.sensors["tx.heater.applied"], dtype=float) if "tx.heater.applied" in obs.sensors
                       else np.full((spec.n_batch, spec.n_rings), float(spec.channels["tx.heater"].init)))
            u0 = applied if self.init_codes is None else self.init_codes
            for b in range(spec.n_batch):
                for n in range(spec.n_rings):
                    self._start(self.rings[b][n], float(u0[b, n]), n)
            self.first = False
        out = np.zeros((spec.n_batch, spec.n_rings), dtype=np.int64)
        for b in range(spec.n_batch):
            for n in range(spec.n_rings):
                r = self.rings[b][n]
                if ok[b, n]:
                    yv = float(y[b, n])
                    if r.st == E_RUN and r.fell(yv, self.params):
                        r.start_recapture(self.params)
                    elif r.st == E_RECAPTURE:
                        if r.recapture(yv, self.params, self.code_hi):
                            self._restart(r)
                    elif r.st == E_RUN:
                        self._update(r, yv)
                    r.push(yv)
                out[b, n] = int(np.rint(min(max(self._command(r), self.code_lo), self.code_hi)))
        return {"tx.heater": out}

    def ring_values(self, attr):
        return np.array([[float(getattr(r, attr)) for r in row] for row in self.rings])

    def telemetry(self) -> dict[str, np.ndarray]:
        st = self.ring_values("st")
        u = self.ring_values("u")
        run = st == E_RUN
        return {"state": st, "locked": run & (self.ring_values("conv") > 0), "detuning_est_pm": np.full(st.shape,
                np.nan), "integrator": u, "integrator_saturated": run & ((u <= self.code_lo) | (u >= self.code_hi)),
                "dc_codes": self.ring_values("y_dc"), "fall_count": self.ring_values("n_fall")}


# ---- perturb and observe ------------------------------------------------------------------------------------------

class _PORing(_Ring):
    def __init__(self, J):
        super().__init__(J)
        self.k = 0             # step within the round
        self.acc = [0.0, 0.0, 0.0]
        self.conv = 0


@register("es_po")
class ExtremumSeekingPO(_ESBase):
    """Perturb-and-observe peak seeking on the drop monitor (module docstring). Writes ``tx.heater``."""
    name = "es_po"

    @dataclass
    class Params(_Common):
        delta_codes: float = 16.0        # probe offset (g26: ~6 pm at the peak, where there is no relief)
        settle_steps: int = 4            # discarded steps after each probe change (thermal settling)
        avg_steps: int = 4               # averaged steps per probe
        mode: str = "newton"             # argmax | newton
        gain: float = 0.5                # newton damping
        max_step_codes: float = 32.0
        curv_floor_frac: float = 2e-4    # |y+ + y- - 2 y0| floor, fraction of the DC level
        conv_codes: float = 4.0          # |move| below this -> telemetry 'locked'

    def __init__(self, params=None, **overrides):
        super().__init__(params, **overrides)
        if self.params.mode not in ("argmax", "newton"):
            raise ValueError(f"{self.name}: mode must be argmax | newton")

    def _new_ring(self):
        return _PORing(int(self.params.jump_steps))

    def _start(self, r, u0, n):
        r.u = u0

    def _restart(self, r):
        r.k = 0
        r.acc = [0.0, 0.0, 0.0]

    def _command(self, r):
        if r.st != E_RUN:
            return r.u
        p = self.params
        phase = r.k // (p.settle_steps + p.avg_steps)
        return r.u + (0.0, p.delta_codes, -p.delta_codes)[min(phase, 2)]

    def _update(self, r, y):
        p = self.params
        dwell = p.settle_steps + p.avg_steps
        phase, kk = divmod(r.k, dwell)
        if kk >= p.settle_steps:
            r.acc[phase] += y
        if phase == 0 and kk == dwell - 1:
            r.y_dc = r.acc[0] / p.avg_steps
        r.k += 1
        if r.k == 3 * dwell:
            y0, yp, ym = (a / p.avg_steps for a in r.acc)
            if p.mode == "argmax":
                move = (0.0, p.delta_codes, -p.delta_codes)[int(np.argmax([y0, yp, ym]))]
            else:
                curv = yp + ym - 2.0 * y0
                floor = p.curv_floor_frac * max(y0, 1.0)
                if curv < -floor:
                    move = p.gain * (yp - ym) * p.delta_codes / (2.0 * -curv)
                else:
                    move = p.max_step_codes * (1.0 if yp > ym else -1.0)
            move = min(max(move, -p.max_step_codes), p.max_step_codes)
            r.conv = 1 if abs(move) < p.conv_codes else 0
            r.u = min(max(r.u + move, self.code_lo), self.code_hi)
            r.k = 0
            r.acc = [0.0, 0.0, 0.0]


# ---- sinusoidal extremum seeking -----------------------------------------------------------------------------------

class _SineRing(_Ring):
    def __init__(self, J):
        super().__init__(J)
        self.k = 0
        self.y_lp = 0.0
        self.xi_lp = 0.0
        self.conv = 0


@register("es_sine")
class ExtremumSeekingSine(_ESBase):
    """Sinusoidal extremum seeking on the drop monitor (module docstring). Writes ``tx.heater``."""
    name = "es_sine"

    @dataclass
    class Params(_Common):
        amplitude_codes: float = 16.0     # a
        dither_hz: float = 2000.0         # f_d (PAT-3 band 1 kHz - 1 MHz; << the heater's thermal corner)
        phase_rad: float = 0.35           # phi: plant lag at f_d (13 us pole ~9 deg + sample/ADC delay ~10 deg)
        hpf_hz: float = 200.0             # high-pass corner (J_lp)
        gain_codes: float = 100.0         # u_hat step per unit normalized demodulated signal (loop gain
        #                                   ~gain a/W^2 per step near the peak: 0.008 for W = 450 codes)
        stagger: bool = True              # ring n starts at phase n * 2 pi / N
        conv_tol: float = 2e-4            # |filtered xi| below this -> telemetry 'locked'

    def reset(self, spec: ControllerSpec) -> None:
        super().reset(spec)
        p = self.params
        T = spec.period_s
        self.w = 2.0 * math.pi * p.dither_hz * T
        if not 0.0 < self.w < math.pi / 2:
            raise ValueError(f"{self.name}: dither_hz {p.dither_hz} needs at least 4 controller steps per period")
        self.a_lp = 1.0 - math.exp(-2.0 * math.pi * p.hpf_hz * T)

    def _new_ring(self):
        return _SineRing(int(self.params.jump_steps))

    def _start(self, r, u0, n):
        r.u = u0
        if self.params.stagger:
            r.k = int(round(n * (2.0 * math.pi / self.w) / max(self.spec.n_rings, 1)))

    def _restart(self, r):
        r.y_lp = 0.0

    def _command(self, r):
        if r.st != E_RUN:
            return r.u
        return r.u + self.params.amplitude_codes * math.sin(self.w * r.k)

    def _update(self, r, y):
        p = self.params
        if r.y_lp <= 0.0:
            r.y_lp = y
        r.y_lp += self.a_lp * (y - r.y_lp)
        r.y_dc = r.y_lp
        # the block mean of this step answers the command sent at the last step, sin(w k) (plus the plant lag phi)
        xi = (y - r.y_lp) * math.sin(self.w * r.k - p.phase_rad) / max(r.y_lp, 1.0)
        r.xi_lp += self.a_lp * (xi - r.xi_lp)
        r.conv = 1 if abs(r.xi_lp) < p.conv_tol else 0
        r.u = min(max(r.u + p.gain_codes * xi, self.code_lo), self.code_hi)
        r.k += 1
