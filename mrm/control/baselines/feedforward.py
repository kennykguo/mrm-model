"""Temperature-sensor feed-forward plus the drop-port PI (ARCHITECTURE section 6 item 7; 04 C11).

The heater command is the PI's integrator (:mod:`mrm.control.baselines.pi_lock`, which still closes the loop on the
monitor) plus a feed-forward from an on-die temperature sensor:

    T_hat[k] = T_prev + (T_s - T_prev) min(1, t_since/T_period)             (predict='interpolate', default)
             = T_s + slope (latency + period/2 + t_since)                    (predict='extrapolate')
    u_ff[k]  = c - u,  c^2 = u^2 - 2 u K_ff(u) (T_hat[k] - T_ref)            (heater codes; quadratic law)
             = -K_ff (T_hat[k] - T_ref)                                      (linear law: power or dac_pwm drive)
    K_ff(u)  = (dlambda/dT, pm/K) / (heater pm per code at u)                (codes per K of die temperature)

T_s = offset_c + lsb_k * code (TempSensorFacts of the spec); T_ref is the reading when each ring's PI locks (per ring,
at each lock, so the sensor's absolute offset cancels and every ring's feed-forward starts at 0, bumpless; a ring that
never locks does not hold the others back); slope[k] is a first-order filtered finite difference of the sensor
readings (per new reading, time constant ``slope_tau_s``). A staircase (``predict='hold'``) steps the heater every
sensor period, which the PI sees as disturbance steps (measured: worse than no feed-forward); the first-order hold
adds one sensor period of lag but no steps. K_ff needs model knowledge: ``ff_codes_per_k`` (a calibration constant
at ``ff_ref_code``, default the first lock code, scheduled as 1/code: the quadratic law) or the knowledge tokens
dlambda_dt_nominal + heater_law + heater_efficiency (07 P1-1, the nominal heater law), scaled by ``ff_scale``.
Without either the feed-forward is off (the controller is then the PI) and telemetry ``ff_active`` says so.

The gain schedule. A quadratic law (voltage or current drive) tunes as code^2, so u K_ff(u) is a constant and the
step from u to c tunes the ring by exactly -dlambda/dT (T_hat - T_ref): the secant of the law from the PI integrator u
(to first order u_ff = -K_ff(u) (T_hat - T_ref)). u_ff depends on u and the sensor only, not on itself; u + u_ff
is >= 0 and is held in the DAC range, and the PI's gain schedule u_ref / u stays exact (the tuning goes as
u^2 - const). Where u + u_ff reaches a DAC rail the integrator is held (anti-windup, as the PI's own clamp of u). A
ring that leaves LOCK (a fall: RECAPTURE) takes u_ff into its integrator, so the applied code does not jump and its
recapture and fault codes are applied codes; its next lock takes a new T_ref. Until 2026-10-16 K_ff was evaluated at
the applied code u + u_ff, so a large negative feed-forward raised its own gain (~1/code) and ran away, and u_ff was
held while a ring was not locked: under A1 ring 0's u_ff went to -7e3 codes at 55 ms and -8e7 at 58 ms, its heater
stayed at code 0 and the ring went to FAULT (ring 7 the same at ~85 ms), and the scorecard lost every ring.

Honest about the sensor (07 P1-10, 04 C11 failure modes): g26's die sensor converts at 1 kHz with 1 ms latency and
0.0625 K LSB (assumed preset values) and sits at the die site, ~1 mm from the bank, not at the ring. It sees neither
laser drift nor self-heating, but the die sensor sees a local aggressor: through the long-range floor of the
interim thermal site law (1.5 % of Z_heater->ring, ~84 K/W from the aggressor site) the calibrated A1 aggressor
(~0.26 W at its peak) moves its reading by ~+20 K (open loop, heaters held: +22.9 K 2 ms after the hot cusp,
+19.6 K peak when A1 starts at its minimum), about a quarter of the victim ring's own rise. The feed-forward reads
that as ambient and shifts every ring's heater (under A1 by -3000 to -4300 codes on rings 1-7 at the hot cusp, most
of what they need: they are heated too), and the PI takes up the difference: under A1 this controller holds every
ring about as pi_lock does (docs/handoff/controller_scorecard.md; ring 0 leaves its point at the hot cusp with its
heater at code 0, which is heater authority, not the feed-forward). The prediction over the latency amplifies
quantization noise (slope from 0.0625 K steps). The self-heating relief divides the base disturbance and the heater
correction alike, so K_ff = s/kappa needs no relief model; but the nominal dlambda/dT is quoted at 25 degC while the
locked ring runs ~100 K hotter (s grows with beta, ~15 % in g26) and the heater law is the cold one, so K_ff is a few
to ~15 % low (``ff_scale`` trims it); the PI removes the residual. The die sensor also sees the bank's own heater
power (the heaters warm the die): while the heaters still move (right after a MAP, or under a local aggressor that
makes the PI back the heaters off) the feed-forward reads that as ambient and double-counts what the PI already
corrects. The tests (tests/verification/test_ctrl_a_feedforward.py) measure a die-wide base ramp: a small,
settling-dependent change (+-10-20 %) of an error the PI alone already holds below 1 pm; and the gain schedule and
the exit from LOCK on toy rings with the quadratic heater law and on the engine under A1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mrm.control.api import ControllerSpec, Observation
from mrm.control.baselines.pi_lock import ST_LOCK, PiLock
from mrm.control.registry import register

__all__ = ["FeedForwardPI"]


@register("feedforward_pi")
class FeedForwardPI(PiLock):
    """Temperature-sensor feed-forward + PI (module docstring). Writes ``tx.heater``.

    Telemetry: PiLock's, plus ff_codes (the feed-forward term), sensor_k (T_hat - T_ref, K) and ff_active;
    integrator_saturated is the applied code u + ff at a DAC rail.
    """
    name = "feedforward_pi"

    @dataclass
    class Params(PiLock.Params):
        sensor: str | None = None          # temperature sensor name (temp.<name>); None = the first one
        ff_codes_per_k: float | None = None  # calibrated K_ff at ff_ref_code (codes/K); None: from nominal knowledge
        ff_ref_code: float | None = None   # code at which ff_codes_per_k holds (None: the lock code)
        ff_scale: float = 1.0              # multiplies K_ff (e.g. 1/relief if known; 1 = open-loop dlambda/dT)
        predict: str = "interpolate"       # hold | interpolate (first-order hold between deliveries, one period
        #                                    more lag, no steps) | extrapolate (filtered slope over the latency)
        slope_tau_s: float = 2e-3          # filter of the sensor slope (s)

    def reset(self, spec: ControllerSpec) -> None:
        super().reset(spec)
        p = self.params
        if not spec.temp_sensors:
            raise ValueError(f"{self.name}: the spec declares no temperature sensor")
        ts = spec.temp_sensors[0] if p.sensor is None else next(
            (t for t in spec.temp_sensors if t.name == p.sensor), None)
        if ts is None:
            raise ValueError(f"{self.name}: no temperature sensor named {p.sensor!r}")
        if ts.channel not in spec.channels:
            raise ValueError(f"{self.name}: sensor channel {ts.channel} is not exposed to the controller")
        self.ts = ts
        nm = self.nominal
        self.k_nominal = None
        if p.ff_codes_per_k is None and nm.dlambda_dt_pm_per_k is not None:
            self.k_nominal = nm.dlambda_dt_pm_per_k          # pm/K; divided by pm/code at run time
        self.ff_active = (p.ff_codes_per_k is not None) or (self.k_nominal is not None and
                                                            nm.pm_per_code(30000.0) is not None)
        sh = (spec.n_batch, 1)                         # one die sensor per world
        self.t_ref = np.full((spec.n_batch, spec.n_rings), np.nan)
        self.t_last = np.full(sh, np.nan)
        self.t_meas = np.full(sh, np.nan)
        self.slope = np.zeros(sh)
        self.tick_last = np.full(sh, -1, np.int64)
        self.t_prev = np.full(sh, np.nan)
        self.dt_prev = np.zeros(sh)
        if p.predict not in ("hold", "interpolate", "extrapolate"):
            raise ValueError(f"{self.name}: predict must be hold | interpolate | extrapolate, got {p.predict!r}")
        self.ff = np.zeros((spec.n_batch, spec.n_rings))
        self.k_ref_code = np.full((spec.n_batch, spec.n_rings), np.nan)
        self.was_locked = np.zeros((spec.n_batch, spec.n_rings), bool)
        self.horizon_s = (ts.latency_ticks + 0.5 * ts.period_ticks) * spec.tick_s
        # the law _k_ff schedules: 1/code (quadratic) for the calibrated constant, the nominal law otherwise
        self.ff_quadratic = p.ff_codes_per_k is not None or nm.law not in ("power", "dac_pwm")

    def _k_ff(self, code: np.ndarray) -> np.ndarray:
        """Feed-forward gain (codes/K) at the heater codes ``code``, quadratic-law scheduled."""
        p = self.params
        c = np.maximum(code, 1.0)
        if p.ff_codes_per_k is not None:
            ref = p.ff_ref_code if p.ff_ref_code is not None else np.where(np.isfinite(self.k_ref_code),
                                                                           self.k_ref_code, c)
            return p.ff_scale * p.ff_codes_per_k * ref / c
        ppc = self.nominal.pm_per_code(c)
        return p.ff_scale * self.k_nominal / np.maximum(ppc, 1e-9)

    def _ff_codes(self, u: np.ndarray, dT: np.ndarray) -> np.ndarray:
        """Feed-forward (codes) for the die change ``dT`` (K) from the PI integrator ``u``: the heater tuning change
        -K_ff(u) u dT taken through the heater law from u (the secant, not the tangent at u or at u + ff). Quadratic
        law: c K_ff(c) is a constant, so c^2 = u^2 - 2 c K_ff(c) dT, ff = c - u >= -u; linear law: ff = -K_ff dT.
        Bounded so that u + ff stays in the DAC range."""
        if self.ff_quadratic:
            c = np.maximum(u, 1.0)
            ff = np.sqrt(np.maximum(u * u - 2.0 * c * self._k_ff(c) * dT, 0.0)) - u
        else:
            ff = -self._k_ff(u) * dT
        return np.clip(ff, self.code_lo - u, self.code_hi - u)

    def _u_range(self, u: np.ndarray, dT: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The integrator range in which u + ff (:meth:`_ff_codes`) stays inside [code_lo, code_hi]."""
        lo, hi = self.code_lo, self.code_hi
        if self.ff_quadratic:
            c = np.maximum(u, 1.0)
            b = 2.0 * c * self._k_ff(c) * dT
            return np.sqrt(np.maximum(lo * lo + b, 0.0)), np.sqrt(np.maximum(hi * hi + b, 0.0))
        k = self._k_ff(u) * dT
        return lo + k, hi + k

    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        spec = self._require_spec()
        p = self.params
        # sensor reading (held between deliveries; the new flag marks a delivery)
        code = np.asarray(obs.sensors[self.ts.channel], dtype=float).reshape(spec.n_batch, -1)[:, :1]
        new = np.asarray(obs.new[self.ts.channel], dtype=bool).reshape(spec.n_batch, -1)[:, :1]
        t_s = self.ts.offset_c + self.ts.lsb_k * code
        fresh = new & np.isfinite(t_s)
        if fresh.any():
            self.t_prev = np.where(fresh & np.isfinite(self.t_meas), self.t_meas, np.where(fresh, t_s, self.t_prev))
            self.dt_prev = np.where(fresh & (self.tick_last >= 0), (obs.tick - self.tick_last) * spec.tick_s,
                                    self.dt_prev)
            dt = (obs.tick - self.tick_last) * spec.tick_s
            have = fresh & (self.tick_last >= 0) & (dt > 0)
            raw = np.where(have, (t_s - self.t_meas) / np.where(dt > 0, dt, 1.0), 0.0)
            a = np.where(have, dt / (p.slope_tau_s + dt), 0.0)
            self.slope = np.where(have, self.slope + a * (raw - self.slope), self.slope)
            self.t_meas = np.where(fresh, t_s, self.t_meas)
            self.tick_last = np.where(fresh, obs.tick, self.tick_last)
        # the PI part (on the integrator u; the feed-forward is added to the output below)
        out = super().step(obs)["tx.heater"].astype(float)
        st = self.ring_values("st")
        locked = (st == ST_LOCK)
        if self.ff_active:
            # a ring that left LOCK (a fall: RECAPTURE) takes its feed-forward into the integrator, bumpless (its
            # recapture and fault codes are then applied codes), and takes a new reference at its next lock
            for b, n in zip(*np.nonzero(self.was_locked & ~locked)):
                r = self.rings[b][n]
                r.u += self.ff[b, n]
                r.u_guard += self.ff[b, n]
            self.t_ref = np.where(locked, self.t_ref, np.nan)
            self.ff = np.where(locked, self.ff, 0.0)
        self.was_locked = locked
        u = self.ring_values("u")
        if self.ff_active:
            # predict to now: latency + half a period + the time since the delivery (a ramp between deliveries,
            # not a staircase of 1 kHz steps that the PI would see as disturbance steps)
            since = np.where(self.tick_last >= 0, (obs.tick - self.tick_last) * spec.tick_s, 0.0)
            if p.predict == "extrapolate":
                t_hat = self.t_meas + self.slope * (self.horizon_s + since)
            elif p.predict == "interpolate":
                frac = np.clip(since / np.where(self.dt_prev > 0, self.dt_prev, np.inf), 0.0, 1.0)
                t_hat = np.where(np.isfinite(self.t_prev), self.t_prev + (self.t_meas - self.t_prev) * frac,
                                 self.t_meas)
            else:
                t_hat = self.t_meas
            start = locked & np.isnan(self.t_ref) & np.isfinite(self.t_meas)        # per ring, at each lock
            self.t_ref = np.where(start, t_hat, self.t_ref)
            self.k_ref_code = np.where(start & np.isnan(self.k_ref_code), u, self.k_ref_code)   # the first lock
            dT = np.where(np.isfinite(self.t_ref), t_hat - self.t_ref, 0.0)
            # anti-windup at the applied code's rails, as the PI's own clamp of u (its type-2 velocity stops there)
            u_lo, u_hi = self._u_range(u, dT)
            for b, n in zip(*np.nonzero(locked & ((u < u_lo) | (u > u_hi)))):
                r = self.rings[b][n]
                if r.u < u_lo[b, n]:
                    r.u = float(u_lo[b, n])
                    r.v = max(r.v, 0.0)
                else:
                    r.u = float(u_hi[b, n])
                    r.v = min(r.v, 0.0)
                u[b, n] = r.u
            self.ff = np.where(locked, self._ff_codes(u, dT), 0.0)
            # the PI keeps the integrator; the feed-forward shifts the applied code (bumpless: ff starts at 0)
            out = np.clip(np.rint(u + self.ff), self.code_lo, self.code_hi)
            # the PI integrator u keeps closing the loop on the monitor (it integrates the residual); its gain
            # schedule u_ref / u stays exact: with the quadratic law the tuning goes as c^2 = u^2 - const dT
        return {"tx.heater": out.astype(np.int64)}

    def telemetry(self) -> dict[str, np.ndarray]:
        t = super().telemetry()
        sh = t["state"].shape
        dT = np.broadcast_to(np.where(np.isfinite(self.t_ref), self.t_meas - self.t_ref, np.nan), sh)
        t.update(ff_codes=self.ff.copy(), sensor_k=np.array(dT, dtype=float),
                 ff_active=np.full(sh, float(self.ff_active)))
        if self.ff_active:              # the clamp is on the applied code u + ff (where the anti-windup holds u)
            c = np.rint(t["integrator"] + self.ff)
            t["integrator_saturated"] = (t["state"] == ST_LOCK) & ((c <= self.code_lo) | (c >= self.code_hi))
        return t
