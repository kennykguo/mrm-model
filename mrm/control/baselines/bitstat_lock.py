"""Bit-statistics lock (Sun et al., VLSI 2015 / JSSC 2016) on ``datapath.bitstat_above`` with the threshold actuator
``datapath.bitstat_threshold`` (07 P0-11; docs/research/04 C6).

What the hardware gives (mrm.plant.datapath, 07 P0-11): at every controller edge, one window of W =
``datapath.bitstat_window_bits`` monitor samples taken at the bit rate through a fast sampler; each '1' bit reads
above the threshold thr with p1 = P(N(mu1, s1^2) > thr), each '0' bit with p0, so
    above = Binomial(n1, p1) + Binomial(n0, p0),     n1 ~ rho W,
where mu1, mu0 are the monitor codes of the two data levels (the eye) and rho the ones density, which the
transmitter knows (``tx.ones_count`` / ``tx.bits``). The window's own n1 is not exposed (the optional
``datapath.bitstat_ones`` channel does not exist yet), so the controller uses rho W.

Algorithm (per ring, vectorized over (B, N))
--------------------------------------------
1. Acquisition: the PAT-3 ramp FSM of mrm.control.baselines.ramp_acquire (UP -> DOWN for the true peak -> back to
   the red side), which hands off to a fast integral loop on the block-mean monitor y at a set-point r_y (red side:
   heater up when y > r_y; 3 kHz crossover from the measured slope).
2. Level estimation (Sun's statistic, threshold form): two threshold trackers, time-multiplexed on the one comparator
   (even edges: upper level, odd edges: lower level). The upper tracker seeks the threshold where half of the upper
   class reads above, i.e. E[above] = n_hi / 2, the lower tracker E[above] = n_hi + n_lo / 2; at those points the
   thresholds are the class medians mu_hi, mu_lo (Gaussian levels). Update: sign of (above - target) times an
   adaptive step (x1.5 after three agreeing signs, x0.5 on a sign change, bounded), robust to the unknown sampler
   noise. Eye opening D = mu_hi - mu_lo (Sun's L_diff = L1 - L0 up to the sign), eye centre c = (mu_hi + mu_lo)/2.
   Which class is upper: on the red side the redder level is dimmer; level '1' is the higher reverse bias
   (``one_is_red``, the plant's NRZ convention, ENGINE_BRIEF section 3 step 5), so the zeros are the upper class.
3. Eye-centre set-point with Sun's ones-density canceller: the monitor averages the levels, y = rho mu1 + (1 - rho)
   mu0 = c + (rho - 1/2)(mu1 - mu0), so the inner set-point is r_y = r_c + (rho - 1/2)(mu1 - mu0) with
   mu1 - mu0 = -D (one_is_red, red side). The loop then holds the EYE CENTRE c at r_c whatever the data density;
   a plain monitor lock moves by (rho - 1/2) D / |dy/dlambda| on a density step (04 C3, C6 failure modes).
4. Eye set-point: r_c = y_bg + alpha (y_pk - y_bg), starting at alpha = 3/4, the eye maximum of a Lorentzian with a
   small swing (x = 1/sqrt 3, the maximum slope). Two outer laws (``mode``):
   - 'ratio' (default): the eye ratio R = D / (c - y_bg) = 2 x dx / (1 + x^2) (Lorentzian, small swing) depends on the
     detuning only, not on the optical power; it is captured once the trackers have settled and then held by an
     integral law on alpha (alpha kept above 0.55: R has its maximum at x = 1, alpha = 1/2). R is proportional to
     the swing dx, so a swing change (data.swing_scale, a bias step) moves this lock; a swing loss of more than
     ~13 % at alpha 3/4 cannot be compensated (R above its reachable maximum) and alpha settles at the floor. A laser power step scales every level but not R, so this lock returns to its detuning,
     where any absolute set-point (and the monitor lock) is pushed by (dP/P) y / |dy/ddelta| (04 C3).
   - 'po': Sun's perturb-and-observe ("perturb, keep the move if L_diff increases"): every ``po_dwell_steps`` steps
     alpha moves by +-``po_step`` and the direction reverses when the dwell-mean D fell (it finds the eye maximum of
     a distorted line too, but wanders ~10 pm p-p on the flat top and does not follow a power step in ms).
5. No data (Sun: "needs data, fails in squelch / idle"): when rho is outside [rho_min, rho_max] or no bits were
   sent, the trackers freeze; when the eye is below eye_min_frac (y_pk - y_bg) (squelch: with one level the trackers
   settle +-0.67 sigma around it, D ~ 1.35 sigma ~ 2.5 % of the peak in g26, hence the 6 % default) the eye P&O
   freezes. The canceller keeps using the last good D, so a density step to all-0 / all-1 does not move the lock.

Tuning comes from measurement (the acquisition's peak, background and slope) and the channel facts (W from the
``datapath.bitstat_above`` range); no Truth, no sampled parameters (07 P1-1). Speed: the trackers settle in about
10-20 updates per level (0.2-0.4 ms at 100 kHz) and the eye P&O moves alpha by po_step per dwell: the eye loop is
slow and data-aware, in the spirit of [SUN15]'s 76 kHz decisions and <= 7 ms lock (04 section 4.0 constraint 2);
fast disturbances are rejected by the inner monitor loop, as in the ramp controller.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mrm.control.api import BaseController, ControllerSpec, Observation
from mrm.control.baselines.ramp_acquire import ST_LOCK, RampFSM, RampParams, monitor_mean
from mrm.control.registry import register

__all__ = ["BitStatParams", "BitStatLock"]


@dataclass
class BitStatParams(RampParams):
    """:class:`RampParams` (acquisition and inner loop) plus the bit-statistics loop.

    alpha_eye0: initial eye-centre fraction of the peak above background (0.75: Lorentzian maximum slope);
    mode: 'ratio' (default: hold the eye ratio D / (c - y_bg), captured over ratio_capture_steps after
    ratio_capture_delay_steps of good data (the inner loop has moved to alpha_eye0 and the trackers followed),
    integral gain ratio_gain per step on alpha) or 'po' (Sun's perturb-and-observe on D: po_step per po_dwell_steps,
    the first half of a dwell not scored);
    alpha_min, alpha_max: bounds of alpha (mode 'ratio' keeps alpha >= max(alpha_min, 0.55): R peaks at the half
    maximum, alpha = 1/2, so below it the integral law on R would run away); settle_steps: tracker settling after
    the hand-off;
    rho_min, rho_max: ones density needed to estimate both levels; eye_min_frac: minimum eye opening (fraction of the
    peak above background) for the eye loop; one_is_red: data level '1' is the redder resonance (plant convention);
    step0_frac, step_min_codes, step_max_frac: adaptive tracker step (initial and maximum as fractions of the peak
    above background).
    """
    alpha: float = 0.5
    alpha_eye0: float = 0.75
    alpha_min: float = 0.35
    alpha_max: float = 0.92
    po_step: float = 0.01
    po_dwell_steps: int = 60
    settle_steps: int = 40
    rho_min: float = 0.2
    rho_max: float = 0.8
    eye_min_frac: float = 0.06
    one_is_red: bool = True
    mode: str = "ratio"
    ratio_gain: float = 0.01
    ratio_capture_steps: int = 30
    ratio_capture_delay_steps: int = 100
    step0_frac: float = 0.02
    step_min_codes: float = 2.0
    step_max_frac: float = 0.1


@register("bitstat_lock")
class BitStatLock(BaseController):
    """Sun-style bit-statistics lock (module docstring). Writes ``tx.heater`` and ``datapath.bitstat_threshold``.

    Telemetry (B, N): state, locked, detuning_est_pm, integrator, integrator_saturated (inner loop), eye_codes (D),
    eye_center, alpha_eye, thr_hi, thr_lo, rho, data_ok.
    """
    name = "bitstat_lock"
    Params = BitStatParams

    def reset(self, spec: ControllerSpec) -> None:
        super().reset(spec)
        if "datapath.bitstat_threshold" not in spec.actuators:
            raise ValueError("BitStatLock needs datapath.bitstat_threshold bound to the controller")
        for ch in ("datapath.bitstat_above", "tx.ones_count", "tx.bits"):
            if ch not in spec.sensors:
                raise ValueError(f"BitStatLock needs the {ch} sensor channel")
        p = self.params
        self.fsm = RampFSM(spec, p)
        B, N = spec.n_batch, spec.n_rings
        self.W = float(spec.channels["datapath.bitstat_above"].max)
        self.thr_max = float(spec.channels["datapath.bitstat_threshold"].max)
        z = lambda v=0.0, dt=np.float64: np.full((B, N), v, dtype=dt)
        self.t_hi = z(np.nan)
        self.t_lo = z(np.nan)
        self.s_hi = z(0.0)
        self.s_lo = z(0.0)
        self.run_hi = z(0, np.int64)            # signed run length of agreeing signs
        self.run_lo = z(0, np.int64)
        self.which = z(0, np.int64)             # tracker whose threshold is applied for the next window: 0 hi, 1 lo
        self.sent = z(-1, np.int64)             # tracker of the threshold sent at the last edge (-1 none)
        # the threshold sent at edge j latches at j + latency_ticks and is used by the window released at the first
        # edge after the latch (event order 07 P0-6: release, then latch): that is the window of edge
        # j + 1 + latency_ticks // period. hist[0] is the tracker in effect for the window released now.
        self.n_hist = 1 + int(getattr(self, "latency_ticks", 0) or 0) // int(spec.period_ticks)
        self.hist = np.full((self.n_hist, B, N), -1, np.int64)
        self.alpha_eye = z(p.alpha_eye0)
        self.po_dir = z(1.0)
        self.po_count = z(0, np.int64)
        self.po_sum = z(0.0)
        self.po_n = z(0, np.int64)
        self.po_prev = z(np.nan)
        self.eye = z(np.nan)
        self.eye_last = z(np.nan)
        self.r_ref = z(np.nan)
        self.r_acc = z(0.0)
        self.r_n = z(0, np.int64)
        self.r_wait = z(0, np.int64)
        self.ratio = z(np.nan)
        if p.mode not in ("ratio", "po"):
            raise ValueError(f"BitStatLock: mode {p.mode!r} must be 'ratio' or 'po'")
        self.center = z(np.nan)
        self.rho = z(0.5)
        self.data_ok = z(False, bool)
        self.lock_steps = z(0, np.int64)
        self.delay_ticks = int(round(p.start_delay_s / spec.tick_s))
        self.t0: int | None = None              # tick of the first step after reset (start delay origin)
        self.started = False
        self.y = z(np.nan)
        self.thr = z(0.0)
        # the eye ratio R = 2 x dx / (1 + x^2) rises with the detuning x only up to x = 1 (alpha = 1/2) and falls
        # beyond: an integral law on R is stable only for alpha > 1/2, so the ratio mode keeps alpha above this floor
        self.alpha_floor = max(p.alpha_min, 0.55) if p.mode == "ratio" else p.alpha_min

    # ---- threshold trackers
    def _update_tracker(self, sel, which, above, target, amp):
        t = self.t_hi if which == 0 else self.t_lo
        s = self.s_hi if which == 0 else self.s_lo
        run = self.run_hi if which == 0 else self.run_lo
        p = self.params
        sgn = np.sign(above - target)                 # above > target: threshold too low -> raise it
        same = (np.sign(run) == sgn) & (sgn != 0)
        new_run = np.where(sgn == 0, 0, np.where(same, run + sgn.astype(np.int64), sgn.astype(np.int64)))
        grow = same & (np.abs(new_run) >= 3)
        flip = (np.sign(run) != sgn) & (run != 0) & (sgn != 0)
        s_new = np.where(grow, s * 1.5, np.where(flip, s * 0.5, s))
        s_new = np.clip(s_new, p.step_min_codes, p.step_max_frac * amp)
        t_new = np.clip(t + sgn * s_new, 0.0, self.thr_max)
        t[...] = np.where(sel, t_new, t)
        s[...] = np.where(sel, s_new, s)
        run[...] = np.where(sel, new_run, run)

    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        p = self.params
        f = self.fsm
        y, valid, clip = monitor_mean(obs, None if np.all(np.isnan(self.y)) else self.y)
        self.y = y
        if self.t0 is None:
            self.t0 = int(obs.tick)
        if not self.started:
            app = np.asarray(obs.sensors["tx.heater.applied"], dtype=np.int64)
            f.code = np.where(f.active, f.code, app)
            if obs.tick - self.t0 < self.delay_ticks:
                return {"tx.heater": f.code.copy(), "datapath.bitstat_threshold": self._thr_codes()}
            f.start(np.ones((f.B, f.N), bool))
            self.started = True
        ones = np.asarray(obs.sensors["tx.ones_count"], dtype=np.float64)
        bits = np.asarray(obs.sensors["tx.bits"], dtype=np.float64)
        self.rho = np.where(bits > 0, ones / np.maximum(bits, 1.0), self.rho)
        above = np.asarray(obs.sensors["datapath.bitstat_above"], dtype=np.float64)
        new_bs = np.asarray(obs.new["datapath.bitstat_above"], dtype=bool)

        locked = f.active & (f.state == ST_LOCK)
        amp = np.maximum(f.y_pk - f.y_bg, 1.0)
        # hand-off: initialize the trackers around the measured monitor level (y ~ eye centre)
        first = locked & np.isnan(self.t_hi)
        if np.any(first):
            half = 0.15 * amp
            self.t_hi = np.where(first, y + half, self.t_hi)
            self.t_lo = np.where(first, np.maximum(y - half, 0.0), self.t_lo)
            self.s_hi = np.where(first, p.step0_frac * amp, self.s_hi)
            self.s_lo = np.where(first, p.step0_frac * amp, self.s_lo)
            self.lock_steps = np.where(first, 0, self.lock_steps)
            self.sent = np.where(first, -1, self.sent)
            self.hist[:, first] = -1
        self.lock_steps = np.where(locked, self.lock_steps + 1, 0)
        lost = ~locked & ~np.isnan(self.t_hi)                 # the inner loop lost lock: re-initialize later
        self.t_hi = np.where(lost, np.nan, self.t_hi)
        self.t_lo = np.where(lost, np.nan, self.t_lo)
        self.r_ref = np.where(lost, np.nan, self.r_ref)
        self.r_acc = np.where(lost, 0.0, self.r_acc)
        self.r_n = np.where(lost, 0, self.r_n)
        self.r_wait = np.where(lost, 0, self.r_wait)

        # data needed for the level estimate (Sun: no data, no statistic)
        rho_ok = (self.rho >= p.rho_min) & (self.rho <= p.rho_max) & (bits > 0)
        # class sizes in a window: on the red side with one_is_red the zeros are the upper (brighter) class
        upper_is_zero = p.one_is_red
        n_hi = self.W * ((1.0 - self.rho) if upper_is_zero else self.rho)
        n_lo = self.W - n_hi
        meas = self.hist[0]                                    # tracker whose threshold this window used
        upd = locked & rho_ok & new_bs & (meas >= 0)
        self._update_tracker(upd & (meas == 0), 0, above, 0.5 * n_hi, amp)
        self._update_tracker(upd & (meas == 1), 1, above, n_hi + 0.5 * n_lo, amp)
        # eye estimate
        d = self.t_hi - self.t_lo
        self.eye = np.where(locked, d, np.nan)
        self.center = np.where(locked, 0.5 * (self.t_hi + self.t_lo), np.nan)
        settled = locked & (self.lock_steps > p.settle_steps)
        self.data_ok = settled & rho_ok & (d >= p.eye_min_frac * amp)
        self.eye_last = np.where(self.data_ok, d, self.eye_last)

        # eye ratio R = D / (c - y_bg): a function of the detuning only (both scale with the optical power), captured
        # once the trackers have settled at the start alpha and then held by moving alpha (mode 'ratio')
        ratio = np.where(self.data_ok, d / np.maximum(self.center - f.y_bg, 1.0), np.nan)
        if p.mode == "ratio":
            # capture only after the inner loop has moved to alpha_eye0 and the trackers have followed it
            self.r_wait = np.where(self.data_ok, self.r_wait + 1, self.r_wait)
            cap = self.data_ok & np.isnan(self.r_ref) & (self.r_wait > p.ratio_capture_delay_steps)
            self.r_acc = np.where(cap, self.r_acc + ratio, self.r_acc)
            self.r_n = np.where(cap, self.r_n + 1, self.r_n)
            ready = cap & (self.r_n >= p.ratio_capture_steps)
            self.r_ref = np.where(ready, self.r_acc / np.maximum(self.r_n, 1), self.r_ref)
            hold = self.data_ok & np.isfinite(self.r_ref)
            err = np.where(hold, (ratio - self.r_ref) / self.r_ref, 0.0)
            # R rises with |x| on the red flank up to the half maximum: R above the reference = too far from the
            # line = a larger fraction alpha (d alpha = 0.75 dR/R at x = 1/sqrt 3, Lorentzian, small swing)
            self.alpha_eye = np.where(hold, np.clip(self.alpha_eye + p.ratio_gain * 0.75 * err, self.alpha_floor,
                                                    p.alpha_max), self.alpha_eye)
        self.ratio = ratio
        # eye P&O on alpha (mode 'po', Sun's lock)
        po = self.data_ok & (p.mode == "po")
        self.po_count = np.where(po, self.po_count + 1, self.po_count)
        score = po & (self.po_count > p.po_dwell_steps // 2)
        self.po_sum = np.where(score, self.po_sum + d, self.po_sum)
        self.po_n = np.where(score, self.po_n + 1, self.po_n)
        done = po & (self.po_count >= p.po_dwell_steps) & (self.po_n > 0)
        if np.any(done):
            j = self.po_sum / np.maximum(self.po_n, 1)
            worse = done & np.isfinite(self.po_prev) & (j < self.po_prev)
            self.po_dir = np.where(worse, -self.po_dir, self.po_dir)
            self.po_prev = np.where(done, j, self.po_prev)
            self.alpha_eye = np.where(done, np.clip(self.alpha_eye + p.po_step * self.po_dir, p.alpha_min,
                                                    p.alpha_max), self.alpha_eye)
            self.po_count = np.where(done, 0, self.po_count)
            self.po_sum = np.where(done, 0.0, self.po_sum)
            self.po_n = np.where(done, 0, self.po_n)

        # inner set-point: eye centre target plus the ones-density canceller (mu1 - mu0 = -D on the red side)
        settled_eye = settled & np.isfinite(self.eye_last)
        r_c = f.y_bg + self.alpha_eye * (f.y_pk - f.y_bg)
        sign = -1.0 if p.one_is_red else 1.0
        r_y = r_c + (self.rho - 0.5) * sign * np.nan_to_num(self.eye_last)
        f.setpoint = np.where(settled_eye, r_y, f.setpoint)

        f.auto_rearm()
        code = f.step(y, valid, clip)

        # threshold for the next window: alternate the trackers
        nxt = np.where(self.sent == 0, 1, 0)
        self.thr = np.where(locked, np.where(nxt == 0, self.t_hi, self.t_lo), self.thr)
        self.sent = np.where(locked, nxt, -1)
        self.hist = np.concatenate([self.hist[1:], self.sent[None]], axis=0)
        return {"tx.heater": code, "datapath.bitstat_threshold": self._thr_codes()}

    def _thr_codes(self) -> np.ndarray:
        return np.clip(np.rint(np.nan_to_num(self.thr)), 0, self.thr_max).astype(np.int64)

    def telemetry(self) -> dict[str, np.ndarray]:
        f = self.fsm
        return {"state": f.state.astype(float), "locked": f.locked.astype(float),
                "detuning_est_pm": f.detuning_est_pm(self.y), "integrator": f.u.copy(),
                "integrator_saturated": f.saturated.astype(float), "eye_codes": self.eye.copy(),
                "eye_center": self.center.copy(), "alpha_eye": self.alpha_eye.copy(), "thr_hi": self.t_hi.copy(),
                "thr_lo": self.t_lo.copy(), "rho": self.rho.copy(), "data_ok": self.data_ok.astype(float),
                "eye_ratio": np.nan_to_num(self.ratio, nan=np.nan), "eye_ratio_ref": self.r_ref.copy()}
