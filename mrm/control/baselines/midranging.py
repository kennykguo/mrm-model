"""Dual-actuator mid-ranging: the bias is the fast actuator that holds the lock, the heater slowly drives the bias
back to the middle of its range (ARCHITECTURE section 6 item 9; 04 C13; 07 P0-7, P1-7).

Error and hand-off as :mod:`mrm.control.baselines.pi_lock` (fraction-of-peak drop-port error on the stable red side;
MAP / RETURN / fall-off and recapture on the heater, with the bias held at mid-range; after a recapture the heater PI
settles at the set-point again, ``lock_steps`` steps within ``lock_tol``, before the bias takes over: the recapture
ends just red of the peak, far from the set-point, which the small bias range cannot bridge). Once the heater PI has
settled at the set-point for the first time (``lock_steps`` steps within ``lock_tol``) the controller identifies the
bias: +``id_bias_codes`` for ``id_steps`` steps, then -``id_bias_codes`` (second halves averaged): s_b = dy/db (monitor
codes per bias code, sign included). The heater slope s_h is the MAP's.
Then, per controller step:

    inner (fast):  b[k] = clip(b[k-1] - k_b (y - f M_pk)/s_b, b_mid - b_range, b_mid + b_range)
    outer (slow):  u[k] = u[k-1] + k_o (s_b / s_h) (b[k] - b_mid)

The inner law is a Newton step on the measured sensitivity (k_b = ``inner_gain`` per step: a deadbeat-like loop on the
bias, whose electro-optic path has no thermal lag); the outer one moves the heater by the code that would give the same
monitor change as the bias offset, at ``outer_gain`` per step (default 1/50: an outer bandwidth ~1/50 of the
controller rate, >= 10x below the inner, 04 C13). Both are velocity forms: nothing winds up at the clamps.

Saturation fallback (``saturation_fallback``, default on; 04 C13 failure mode "limited bias range"): while the bias
sits at a rail of its window the heater also runs the PI velocity law of pi_lock on the error, so the loop never
does worse than the plain PI. Without it the heater moves at most k_o |s_b/s_h| b_range per step (~0.75 codes per
step in g26, ~10x short of the slide-10 aggressor's need) and the lock was lost at the A1 rate (max 470 pm).

What limits it (measured in tests/verification/test_ctrl_a_midrange.py, g26 nominal): the bias authority over the
detuning is tiny there (a 0.2 V step moves the resonance 0.1-0.5 pm, because the FCA self-heating change cancels most
of the electro-optic shift, 07 P1-7), and the bias also changes the loss, so the measured s_b mixes detuning and loss
(for some rings it is positive: the inner loop then holds the monitor by changing the loss, not the detuning). The
default bias window (1.0-2.2 V around 1.6 V) also costs OMA against the 1.05 V relOMA optimum. Requires
``controller_interface.bind.tx.bias: controller``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mrm.control.api import ControllerSpec, Observation
from mrm.control.baselines.pi_lock import (ST_LOCK, PiLock, _pi_error, _pi_ring_step, block_inputs, _START)
from mrm.control.registry import register

__all__ = ["MidRanging", "M_ID_PLUS", "M_ID_MINUS", "M_RUN"]

M_ID_PLUS, M_ID_MINUS, M_RUN = 0, 1, 2


@register("midranging")
class MidRanging(PiLock):
    """Bias fast / heater slow mid-ranging lock (module docstring). Writes ``tx.heater`` and ``tx.bias``.

    Telemetry: PiLock's, plus bias_codes, bias_slope_per_code, mid_state (0/1 identification, 2 running) and
    bias_offset_codes (b - b_mid).
    """
    name = "midranging"

    @dataclass
    class Params(PiLock.Params):
        bias_mid_v: float = 1.6            # middle of the bias window (V, printed axis 1.0-3.75 V)
        bias_range_v: float = 0.6          # half width of the window the inner loop may use
        id_bias_codes: float = 300.0       # identification step (0.2 V)
        id_steps: int = 20                 # steps per identification half (second half averaged)
        inner_gain: float = 0.5            # bias Newton gain per step
        outer_gain: float = 0.02           # heater re-centring gain per step
        saturation_fallback: bool = True   # bias at a rail: the heater also runs the PI on the error (docstring)

    def reset(self, spec: ControllerSpec) -> None:
        if spec.bindings.get("tx.bias") != "controller":
            raise ValueError(f"{self.name}: needs controller_interface.bind.tx.bias: controller")
        super().reset(spec)
        p = self.params
        bd = spec.bias_dac
        v_lo, v_hi = (bd.phys_min or 1.0), (bd.phys_max or 3.75)
        lsb = (v_hi - v_lo) / bd.code_max
        self.b_mid = float(round((p.bias_mid_v - v_lo) / lsb))
        half = p.bias_range_v / lsb
        self.b_lo = max(0.0, self.b_mid - half)
        self.b_hi = min(float(bd.code_max), self.b_mid + half)
        if self.b_mid - p.id_bias_codes < self.b_lo or self.b_mid + p.id_bias_codes > self.b_hi:
            raise ValueError(f"{self.name}: id_bias_codes {p.id_bias_codes} exceeds the bias window")
        sh = (spec.n_batch, spec.n_rings)
        self.b = np.full(sh, self.b_mid)
        self.mstate = np.full(sh, M_ID_PLUS, np.int64)
        self.mcount = np.zeros(sh, np.int64)
        self.macc = np.zeros(sh)
        self.y_plus = np.zeros(sh)
        self.s_b = np.full(sh, np.nan)
        self.resettle = np.zeros(sh, bool)     # after a fall: the PI settles at the set-point before RUN again

    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        spec = self._require_spec()
        p = self.params
        y, ok = block_inputs(obs, self.y)
        self.y = y
        u0 = None
        if self.first:
            applied = (np.asarray(obs.sensors["tx.heater.applied"], dtype=float) if "tx.heater.applied" in obs.sensors
                       else np.full((spec.n_batch, spec.n_rings), float(spec.channels["tx.heater"].init)))
            u0 = applied if self.init_codes is None else self.init_codes
        out_h = np.zeros((spec.n_batch, spec.n_rings), dtype=np.int64)
        start = _START[p.start]
        over = (np.asarray(obs.sensors["adc_overrange"], dtype=bool) if "adc_overrange" in obs.sensors
                else np.zeros((spec.n_batch, spec.n_rings), bool))
        f = p.fraction
        half = p.id_steps // 2
        for bb in range(spec.n_batch):
            for n in range(spec.n_rings):
                r = self.rings[bb][n]
                settling = (r.st == ST_LOCK and (np.isnan(self.s_b[bb, n]) or self.resettle[bb, n])
                            and self.mcount[bb, n] == 0 and r.lock_cnt < p.lock_steps)
                if r.st != ST_LOCK or self.first or not ok[bb, n] or settling:
                    out_h[bb, n] = int(_pi_ring_step(r, float(y[bb, n]), bool(ok[bb, n]), self.first,
                                                     0.0 if u0 is None else float(u0[bb, n]), start, self.cfg,
                                                     bool(over[bb, n])))
                    if r.st != ST_LOCK:
                        self.b[bb, n] = self.b_mid
                        self.mstate[bb, n] = M_ID_PLUS if np.isnan(self.s_b[bb, n]) else M_RUN
                        self.mcount[bb, n] = 0
                        self.macc[bb, n] = 0.0
                        self.resettle[bb, n] = True
                    continue
                self.resettle[bb, n] = False            # the PI has settled (or never left): mid-ranging runs
                yv = float(y[bb, n])
                # the PI's fall-off detection still applies; while running, the PI integrator is replaced
                hmax = max(r.hist[:r.hist_n]) if r.hist_n else yv
                if r.hist_n >= 1 and (hmax - yv) > p.fall_frac * r.peak and yv < p.fall_level * r.peak:
                    out_h[bb, n] = int(_pi_ring_step(r, yv, True, False, 0.0, start, self.cfg))
                    self.b[bb, n] = self.b_mid
                    continue
                ms = self.mstate[bb, n]
                if ms in (M_ID_PLUS, M_ID_MINUS):
                    self.mcount[bb, n] += 1
                    if self.mcount[bb, n] > half:
                        self.macc[bb, n] += yv
                    if self.mcount[bb, n] >= p.id_steps:
                        avg = self.macc[bb, n] / (p.id_steps - half)
                        if ms == M_ID_PLUS:
                            self.y_plus[bb, n] = avg
                            self.b[bb, n] = self.b_mid - p.id_bias_codes
                            self.mstate[bb, n] = M_ID_MINUS
                        else:
                            self.s_b[bb, n] = (self.y_plus[bb, n] - avg) / (2.0 * p.id_bias_codes)
                            self.b[bb, n] = self.b_mid
                            self.mstate[bb, n] = M_RUN
                        self.mcount[bb, n] = 0
                        self.macc[bb, n] = 0.0
                    elif ms == M_ID_PLUS and self.mcount[bb, n] == 1:
                        self.b[bb, n] = self.b_mid + p.id_bias_codes
                else:
                    sb = self.s_b[bb, n]
                    if sb != 0.0 and math.isfinite(sb):
                        db = -p.inner_gain * (yv - f * r.peak) / sb
                        self.b[bb, n] = min(max(self.b[bb, n] + db, self.b_lo), self.b_hi)
                        sh_ = r.slope if (r.slope < 0.0 and math.isfinite(r.slope)) else -1.0
                        du = p.outer_gain * (sb / sh_) * (self.b[bb, n] - self.b_mid)
                        r.y = yv
                        e = _pi_error(r, f, self.law, p.sigma_min, p.sigma_max)
                        if p.saturation_fallback and (self.b[bb, n] <= self.b_lo or self.b[bb, n] >= self.b_hi):
                            du += self.k_c * ((e - r.e_prev) + self.t_ratio * e)
                        r.e_prev = e
                        r.u = min(max(r.u + du, self.code_lo), self.code_hi)
                    # telemetry 'locked' (07 P1-20) follows the mid-ranging loop, not the PI's count frozen at the
                    # hand-off (only in RUN: during the identification the count must stay, see `settling`)
                    if r.peak > 0.0 and abs(yv / r.peak - f) < p.lock_tol:
                        r.lock_cnt += 1
                    else:
                        r.lock_cnt = 0
                # keep the PI's history (fall detection) current
                for j in range(len(r.hist) - 1, 0, -1):
                    r.hist[j] = r.hist[j - 1]
                r.hist[0] = yv
                r.y = yv
                if r.hist_n < len(r.hist):
                    r.hist_n += 1
                out_h[bb, n] = int(np.rint(r.u))
        self.first = False
        return {"tx.heater": out_h, "tx.bias": np.rint(self.b).astype(np.int64)}

    def telemetry(self) -> dict[str, np.ndarray]:
        t = super().telemetry()
        t.update(bias_codes=self.b.copy(), bias_slope_per_code=self.s_b.copy(),
                 mid_state=self.mstate.astype(float), bias_offset_codes=self.b - self.b_mid)
        return t
