"""Heater-dither lock-in: 1f (slope) and 2f (curvature) demodulation of the drop monitor, peak lock or offset lock at
the inflection point (ARCHITECTURE section 6 item 2; 04 C5; 07 P0-7). Python reference and compiled (numba, flat
ABI) versions with identical arithmetic.

Dither and demodulation
-----------------------
A three-level square dither rides on the heater centre code u_c: phases p = 0, 1, 2, 3 apply u_c + A, u_c, u_c - A,
u_c, each for ``phase_steps`` controller steps (default 6 steps = 60 us at 100 kHz: a 240 us cycle, 4.2 kHz,
inside PAT-3's 1 kHz - 1 MHz band). The first ``blank_steps`` block means of each phase are discarded (the heater
step response: 3.2 us and 13 us poles, 07 P1-2), the rest averaged into y_p (integrate-and-dump of the
``tx.monitor.samples`` blocks). After every complete cycle:

    S1 = (y_0 - y_1 - y_2 + y_3) / (2 A)                              dy/du, codes per code (1f)
    S2 = (y'_2 - 2 y'_3 + 2 y_0 - 2 y_1 + y_2) / (2 A^2)                d2y/du2, codes per code^2 (2f)
    y0 = (y_1 + y_3) / 2                                               DC level

(y'_q: the previous cycle). With y_q = Y + d q + S1 D_q + S2 D_q^2/2 (D = A, 0, -A, 0; d a drift per phase: a
disturbance ramp) both are exact: drift and curvature cancel in S1, drift in S2. The settling residue eps of each
phase change (the blanked steps do not remove all of the 13 us pole) only scales S1, and cancels in this S2 because
the two centre phases that follow +A and -A enter with equal weights; the shorter (y_0 + y_2 - 2 y_1)/A^2 carries a
bias -2 S1 A eps/A^2 that, under the self-heating relief (the true 2f signal falls as 1/R^3), moved the lock point by
tens of pm (measured). The three-level pattern gives the curvature directly; a two-level square wave has no centre
sample. The first cycle after a (re)start only primes y'. When the centre moves by du at the end of a cycle, the
stored y'_2, y'_3 are shifted by S1 du (the move's expected effect), else the move reads as curvature
(S1 du / (2 A^2): a large bias under the self-heating relief). Rings start at
phase n mod 4 when ``stagger`` is on (the total heater power then carries no common dither).

Lock points and update laws (once per cycle; u_c += du, |du| <= ``max_step_codes``)
------------------------------------------------------------------------------------
On the stable branch (07 P0-1) delta = u + G P_abs(delta) gives y_u = L'/R and y_uu = L''/R^3 with the self-heating
relief R = 1 + G |P_abs'| (pure-Lorentzian algebra, x = delta/w, the absorbed power has the monitor's shape):

- ``mode='peak'`` (1f -> 0): damped Newton step du = gain * S1 / (-S2) while S2 < 0, else du = max_step sign(S1).
  The peak is open-loop stable (P' = 0) but in g26 the fold sits only ~9 pm blue of it (bistability ratio ~6): the
  dither, noise or a disturbance crosses it and the ring falls (04 C5 "the wrong point for a modulator"; tests).
- ``mode='inflection'`` (2f -> 0, the default offset lock): the zero of S2 is the inflection of the drop line,
  x = 1/sqrt(3) for a Lorentzian (the NRZ-averaged line of g26 nominal: ~131 pm, measured), on the red side, for any
  laser power, relief or dither amplitude (y_uu = L''/R^3): the power-independent lock of Grimaldi et al. 2023
  (04 C5). With eps = S2 y0 / S1^2 = eps_L / R (eps_L = (6x^2 - 2)/(4x^2),
  d eps_L/dx = 5.2 at the inflection) and the half width in heater codes W = 0.866 y0/|S1| (= W_u R at the
  inflection), du = -gain * eps * W / 5.2 moves x by -gain (x - x_i) / R per cycle: the relief divides the loop gain
  (R ~ 7 in g26, unknown to the controller) so ``gain`` is set for R = 1 and the loop is slow under strong
  self-heating. If S1 >= 0 the ring is blue of the peak (near the fold): du = +max_step.

Bandwidth: at most one update per cycle, so the loop is below ~1/(10 cycle) = 400 Hz (04 constraint 1): it cannot
follow the slide-10 aggressor (183 pm/ms of resonance target; tests report the error). The dither costs OMA: the
amplitude in detuning is A/(W_u R), here ~13 pm at A = 256 codes; the 2f signal falls as (A/W_u)^2/R^3, which is why
A is large in codes.

Fall-off (crossing the fold, 07 P0-1): the monitor drops by more than ``fall_frac`` of the DC level within
``jump_steps`` steps; RECAPTURE ramps the heater up, and above u_fall + ``guard_codes`` (a capture by another line or
order is ignored, see pi_lock) waits until the monitor has peaked above ``recapture_level`` of the old DC level and
fallen back by ``recapture_drop`` (red of the peak again, after a capture jump or a smooth passage), then the dither
restarts.

Hand-off contract as in :mod:`mrm.control.baselines.pi_lock` (stable branch, heater code from ``init_codes`` or
``tx.heater.applied``). Units: codes and seconds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numba as nb
import numpy as np

from mrm.control import api
from mrm.control.api import BaseController, CompiledController, ControllerSpec, Observation
from mrm.control.baselines.pi_lock import NominalModel, block_inputs, per_ring, _ring_list
from mrm.control.registry import register
from mrm.core.channels import ActId, ObsId

__all__ = ["DitherLockin", "DitherLockinCompiled", "dither_init", "dither_step", "MODES", "D_LOCK", "D_RECAPTURE",
           "D_FAULT", "INFLECTION_SLOPE", "INFLECTION_WIDTH"]

MODES = {"peak": 0, "inflection": 1}
D_LOCK, D_RECAPTURE, D_FAULT = 3, 4, 5          # same numbers as pi_lock's LOCK / RECAPTURE / FAULT
INFLECTION_SLOPE = 3.0 * math.sqrt(3.0)          # d eps_L/dx at x = 1/sqrt(3): 1/x^3 = 5.196
INFLECTION_WIDTH = 2.0 * (1.0 / math.sqrt(3.0)) / (1.0 + 1.0 / 3.0)   # |L'|/L at the inflection = 0.866
_W_FACTOR = INFLECTION_WIDTH


def _check(p, name):
    if p.mode not in MODES:
        raise ValueError(f"{name}: mode must be one of {sorted(MODES)}, got {p.mode!r}")
    if p.phase_steps < 2 or not 0 <= p.blank_steps < p.phase_steps:
        raise ValueError(f"{name}: need phase_steps >= 2 and 0 <= blank_steps < phase_steps")
    if p.dither_codes <= 0 or p.max_step_codes <= 0 or p.gain <= 0:
        raise ValueError(f"{name}: dither_codes, max_step_codes and gain must be > 0")


@dataclass
class _DParams:
    mode: str = "inflection"        # peak (1f -> 0) | inflection (2f -> 0, red side): see the module docstring
    dither_codes: float = 256.0     # A: ~13 pm of detuning at R ~ 7 (g26 inflection); peak mode wants ~16
    phase_steps: int = 6            # steps per dither phase (4 phases per cycle)
    blank_steps: int = 3            # blocks discarded at the start of each phase (thermal settling)
    gain: float = 1.0               # loop gain per cycle at R = 1 (deadbeat; divided by the relief R, docstring)
    max_step_codes: float = 128.0   # largest centre move per cycle
    stagger: bool = True            # ring n starts at phase n mod 4
    init_codes: object = None       # hand-off heater code(s); None = tx.heater.applied
    fall_frac: float = 0.35         # fall-off: drop > fall_frac * DC within jump_steps steps
    recapture_level: float = 0.3    # RECAPTURE ends once the monitor peaked above level * DC (before the fall) ...
    recapture_drop: float = 0.02    # ... and fell back by this fraction (red of the peak again)
    guard_codes: float = 500.0
    recapture_step_codes: float = 32.0
    recapture_span_codes: float = 8000.0   # RECAPTURE gives up this far above the guard (-> FAULT, back to u_fall)
    jump_steps: int = 3
    eps_tol: float = 0.3            # inflection: |eps| below this for lock_cycles cycles -> 'locked'
    lock_cycles: int = 5
    latency_ticks: int = 0


class _DRing:
    __slots__ = ("st", "uc", "y", "acc", "yph", "y0", "s1", "s2", "eps", "w", "u_guard", "y_ref", "y_seen", "p2",
                 "p3", "hist", "phase", "kph", "nacc", "nph", "cycles", "lock_cnt", "n_fall", "hist_n")

    def __init__(self, J: int):
        self.st = D_LOCK
        self.uc = self.y = self.acc = self.y0 = self.s1 = self.s2 = self.eps = self.w = self.u_guard = 0.0
        self.y_ref = self.y_seen = self.p2 = self.p3 = 0.0
        self.yph = [0.0, 0.0, 0.0, 0.0]
        self.hist = [0.0] * J
        self.phase = self.kph = self.nacc = self.nph = self.cycles = self.lock_cnt = self.n_fall = self.hist_n = 0


def _dither_offset(phase: int, a: float) -> float:
    if phase == 0:
        return a
    if phase == 2:
        return -a
    return 0.0


def _d_ring_step(r: _DRing, y_blk: float, ok: bool, first: bool, u0: float, ring: int, cfg: tuple) -> float:
    """One step of one ring (module docstring); returns the heater code (rint, clipped)."""
    (mode, a, h, blank, gain, max_step, stagger, fall_frac, rec_level, guard, recap, eps_tol, rec_drop, lo, hi,
     span) = cfg
    if first:
        r.uc = u0
        r.st = D_LOCK
        r.phase = ring % 4 if stagger else 0
        r.kph = 0
        r.nph = 0
    if ok:
        y = y_blk
        r.y = y
        J = len(r.hist)
        hmax = y
        hmin = y
        if r.hist_n > 0:
            hmax = r.hist[0]
            hmin = r.hist[0]
            for j in range(1, r.hist_n):
                if r.hist[j] > hmax:
                    hmax = r.hist[j]
                if r.hist[j] < hmin:
                    hmin = r.hist[j]
        if r.st == D_LOCK:
            yr = r.y_ref if r.y_ref > 0.0 else hmax
            if r.hist_n >= 1 and (hmax - y) > fall_frac * yr:
                r.st = D_RECAPTURE
                r.n_fall += 1
                r.lock_cnt = 0
                r.u_guard = r.uc + guard
                r.y_seen = 0.0
                r.y_ref = yr          # the recapture level refers to the pre-fall level (never 0: see RECAPTURE)
            else:
                if r.kph >= blank:
                    r.acc += y
                    r.nacc += 1
                r.kph += 1
                if r.kph == h:
                    r.yph[r.phase] = r.acc / r.nacc if r.nacc > 0 else y
                    r.acc = 0.0
                    r.nacc = 0
                    r.kph = 0
                    r.nph += 1
                    if r.phase == 3 and r.nph >= 4 and r.nph < 8:
                        r.p2 = r.yph[2]                  # prime the previous cycle
                        r.p3 = r.yph[3]
                    elif r.phase == 3 and r.nph >= 8:
                        r.cycles += 1
                        r.s1 = (r.yph[0] - r.yph[1] - r.yph[2] + r.yph[3]) / (2.0 * a)
                        r.y0 = 0.5 * (r.yph[1] + r.yph[3])
                        r.s2 = (r.p2 - 2.0 * r.p3 + 2.0 * r.yph[0] - 2.0 * r.yph[1] + r.yph[2]) / (2.0 * a * a)
                        r.p2 = r.yph[2]
                        r.p3 = r.yph[3]
                        r.y_ref = r.y0
                        if mode == 1:
                            if r.s1 < 0.0:
                                r.eps = r.s2 * r.y0 / (r.s1 * r.s1)
                                r.w = _W_FACTOR * r.y0 / (-r.s1)
                                du = -gain * r.eps * r.w / INFLECTION_SLOPE
                            else:
                                r.eps = math.nan
                                du = max_step
                            if abs(r.eps) < eps_tol:
                                r.lock_cnt += 1
                            else:
                                r.lock_cnt = 0
                        else:
                            if r.s2 < 0.0:
                                du = gain * r.s1 / (-r.s2)
                            elif r.s1 > 0.0:
                                du = max_step
                            elif r.s1 < 0.0:
                                du = -max_step
                            else:
                                du = 0.0
                            r.eps = r.s1 * a / r.y0 if r.y0 > 0.0 else math.nan
                            if abs(r.eps) < eps_tol:
                                r.lock_cnt += 1
                            else:
                                r.lock_cnt = 0
                        if du > max_step:
                            du = max_step
                        if du < -max_step:
                            du = -max_step
                        un = r.uc + du
                        un = lo if un < lo else (hi if un > hi else un)
                        # the centre move shifts every later sample by S1 du: shift the stored previous cycle too,
                        # so the cross-cycle S2 does not read the move as curvature
                        r.p2 += r.s1 * (un - r.uc)
                        r.p3 += r.s1 * (un - r.uc)
                        r.uc = un
                    r.phase = (r.phase + 1) % 4
        elif r.st == D_RECAPTURE:
            # as pi_lock: past the guard, ramp until the monitor peaked (capture jump or smooth passage) and fell back
            if r.uc >= r.u_guard and y > r.y_seen:
                r.y_seen = y
            if r.y_seen > rec_level * r.y_ref and y < (1.0 - rec_drop) * r.y_seen:
                r.st = D_LOCK
                r.kph = 0
                r.nph = 0
                r.acc = 0.0
                r.nacc = 0
                r.y_ref = 0.0
            elif r.uc >= hi or r.uc >= r.u_guard + span:
                r.st = D_FAULT                    # give up: back to the code at the fall (see pi_lock)
                r.uc = r.u_guard - guard
            else:
                r.uc = r.uc + recap if r.uc + recap < hi else hi
        for j in range(J - 1, 0, -1):
            r.hist[j] = r.hist[j - 1]
        r.hist[0] = y
        if r.hist_n < J:
            r.hist_n += 1
    off = _dither_offset(r.phase, a) if r.st == D_LOCK else 0.0
    c = r.uc + off
    c = lo if c < lo else (hi if c > hi else c)
    return float(np.rint(c))


@register("dither_lockin")
class DitherLockin(BaseController):
    """Heater-dither lock-in, 1f/2f, peak or inflection lock (module docstring); writes ``tx.heater``.

    Telemetry (B, N): state, locked, detuning_est_pm (NaN: the dither alone does not give the absolute detuning),
    integrator (centre code), integrator_saturated, s1 (codes/code), s2 (codes/code^2), eps, dc_codes, fall_count.
    """
    name = "dither_lockin"
    Params = _DParams

    def __init__(self, params=None, **overrides):
        super().__init__(params, **overrides)
        _check(self.params, self.name)
        self.latency_ticks = int(self.params.latency_ticks)

    def _cfg(self, spec):
        p = self.params
        self.nominal = NominalModel.from_spec(spec)
        self.code_lo, self.code_hi = float(spec.heater_dac.code_min), float(spec.heater_dac.code_max)
        return (MODES[p.mode], float(p.dither_codes), int(p.phase_steps), int(p.blank_steps), float(p.gain),
                float(p.max_step_codes), bool(p.stagger), float(p.fall_frac), float(p.recapture_level),
                float(p.guard_codes), float(p.recapture_step_codes), float(p.eps_tol), float(p.recapture_drop),
                self.code_lo, self.code_hi, float(p.recapture_span_codes))

    def reset(self, spec: ControllerSpec) -> None:
        super().reset(spec)
        sh = (spec.n_batch, spec.n_rings)
        self.cfg = self._cfg(spec)
        init = per_ring(self.params.init_codes, sh, "init_codes")
        self.init_codes = None if init is None else np.rint(init)      # whole codes, as the compiled port
        self.rings = [[_DRing(int(self.params.jump_steps)) for _ in range(sh[1])] for _ in range(sh[0])]
        self.first = True
        self.y = np.zeros(sh)

    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        spec = self._require_spec()
        y, ok = block_inputs(obs, self.y)
        self.y = y
        u0 = None
        if self.first:
            applied = (np.asarray(obs.sensors["tx.heater.applied"], dtype=float) if "tx.heater.applied" in obs.sensors
                       else np.full((spec.n_batch, spec.n_rings), float(spec.channels["tx.heater"].init)))
            u0 = applied if self.init_codes is None else self.init_codes
        out = np.zeros((spec.n_batch, spec.n_rings), dtype=np.int64)
        for b in range(spec.n_batch):
            for n in range(spec.n_rings):
                out[b, n] = int(_d_ring_step(self.rings[b][n], float(y[b, n]), bool(ok[b, n]), self.first,
                                             0.0 if u0 is None else float(u0[b, n]), n, self.cfg))
        self.first = False
        return {"tx.heater": out}

    def ring_values(self, attr: str) -> np.ndarray:
        return np.array([[float(getattr(r, attr)) for r in row] for row in self.rings])

    def telemetry(self) -> dict[str, np.ndarray]:
        return _d_telemetry(self.params, self.code_lo, self.code_hi, self.ring_values("st"), self.ring_values("uc"),
                            self.ring_values("s1"), self.ring_values("s2"), self.ring_values("eps"),
                            self.ring_values("y0"), self.ring_values("lock_cnt"), self.ring_values("n_fall"))


def _d_telemetry(p, lo, hi, st, uc, s1, s2, eps, y0, lock_cnt, n_fall):
    lock = st == D_LOCK
    return {"state": st, "locked": lock & (lock_cnt >= p.lock_cycles), "detuning_est_pm": np.full(st.shape, np.nan),
            "integrator": uc, "integrator_saturated": lock & ((uc <= lo) | (uc >= hi)), "s1": s1, "s2": s2,
            "eps": eps, "dc_codes": y0, "fall_count": n_fall}


# ---- compiled port ------------------------------------------------------------------------------------------------
# p_i64: [0 N, 1 M, 2 off_samples, 3 off_count, 4 off_valid, 5 off_applied, 6 off_heater, 7 mode, 8 h, 9 blank,
#         10 stagger, 11 J, 12 has_init, 13.. init codes (N)]
# p_f64: [0 A, 1 gain, 2 max_step, 3 fall_frac, 4 recapture_level, 5 guard, 6 recapture_step, 7 eps_tol,
#         8 recapture_drop, 9 lo, 10 hi, 11 recapture_span]
# s_i64: [0 first] + per ring at 1 + 9 n: [st, phase, kph, nacc, nph, cycles, lock_cnt, n_fall, hist_n]
# s_f64: per ring at n R (R = 17 + J): [uc, y, acc, yph0, yph1, yph2, yph3, y0, s1, s2, eps, w, u_guard, y_ref, y_seen,
#        p2, p3, hist (J)]
_D_NI = 13
_D_NUF = 9
_D_SI = 9
_D_SF = 17
_SAMPLES, _COUNT = int(ObsId.TX_MONITOR_SAMPLES), int(ObsId.TX_MONITOR_COUNT)
_VALID, _APPLIED, _HEATER = int(ObsId.MONITOR_VALID), int(ObsId.TX_HEATER_APPLIED), int(ActId.TX_HEATER)


@nb.njit(cache=True)
def dither_init(spec_i64, spec_f64):
    n = spec_i64[api.SPEC_I_N_RINGS]
    p = np.zeros(_D_NI + n, dtype=np.int64)
    p[0] = n
    p[1] = spec_i64[api.SPEC_I_M]
    p[2] = api.obs_offset(spec_i64, _SAMPLES)
    p[3] = api.obs_offset(spec_i64, _COUNT)
    p[4] = api.obs_offset(spec_i64, _VALID)
    p[5] = api.obs_offset(spec_i64, _APPLIED)
    p[6] = api.act_offset(spec_i64, _HEATER)
    for k in range(6):
        p[7 + k] = api.user_i64(spec_i64, k)          # mode, h, blank, stagger, J, has_init
    for i in range(n):
        p[_D_NI + i] = api.user_i64(spec_i64, 6 + i)
    pf = np.zeros(_D_NUF + 3)
    for k in range(_D_NUF):
        pf[k] = api.user_f64(spec_f64, k)
    pf[9] = spec_i64[api.SPEC_I_HEATER_MIN]
    pf[10] = spec_i64[api.SPEC_I_HEATER_MAX]
    pf[11] = api.user_f64(spec_f64, _D_NUF)
    J = p[11]
    s_i = np.zeros(1 + _D_SI * n, dtype=np.int64)
    s_i[0] = 1
    s_f = np.zeros((_D_SF + J) * n)
    for i in range(n):
        s_i[1 + _D_SI * i] = D_LOCK
    return p, pf, s_i, s_f


@nb.njit(cache=True)
def dither_step(p, pf, s, sf, obs, act):
    """Compiled step: the per-ring algorithm of :func:`_d_ring_step`, operation for operation."""
    n_r = p[0]
    m_max = p[1]
    mode = p[7]
    h = p[8]
    blank = p[9]
    stagger = p[10]
    J = p[11]
    a = pf[0]
    gain = pf[1]
    max_step = pf[2]
    fall_frac = pf[3]
    rec_level = pf[4]
    guard = pf[5]
    recap = pf[6]
    eps_tol = pf[7]
    rec_drop = pf[8]
    lo = pf[9]
    hi = pf[10]
    R = _D_SF + J
    first = s[0] == 1
    active_ring = obs[1]
    for i in range(n_r):
        si = 1 + _D_SI * i
        b = i * R
        if first:
            if p[12] == 1:
                sf[b] = float(p[_D_NI + i])
            elif p[5] >= 0:
                sf[b] = float(obs[p[5] + i])
            else:
                sf[b] = 0.0
            s[si] = D_LOCK
            s[si + 1] = i % 4 if stagger == 1 else 0
            s[si + 2] = 0
            s[si + 4] = 0
        cnt = obs[p[3] + i]
        ok = cnt > 0
        if p[4] >= 0 and obs[p[4] + i] == 0:
            ok = False
        if active_ring >= 0 and active_ring != i:
            ok = False
        if ok:
            acc_y = 0.0
            for m in range(cnt):
                acc_y += float(obs[p[2] + i * m_max + m])
            y = acc_y / cnt
            sf[b + 1] = y
            hb = b + _D_SF
            hn = s[si + 8]
            hmax = y
            hmin = y
            if hn > 0:
                hmax = sf[hb]
                hmin = sf[hb]
                for j in range(1, hn):
                    if sf[hb + j] > hmax:
                        hmax = sf[hb + j]
                    if sf[hb + j] < hmin:
                        hmin = sf[hb + j]
            st = s[si]
            if st == D_LOCK:
                yr = sf[b + 13] if sf[b + 13] > 0.0 else hmax
                if hn >= 1 and (hmax - y) > fall_frac * yr:
                    s[si] = D_RECAPTURE
                    s[si + 7] += 1
                    s[si + 6] = 0
                    sf[b + 12] = sf[b] + guard
                    sf[b + 14] = 0.0
                    sf[b + 13] = yr
                else:
                    if s[si + 2] >= blank:
                        sf[b + 2] += y
                        s[si + 3] += 1
                    s[si + 2] += 1
                    if s[si + 2] == h:
                        ph = s[si + 1]
                        sf[b + 3 + ph] = sf[b + 2] / s[si + 3] if s[si + 3] > 0 else y
                        sf[b + 2] = 0.0
                        s[si + 3] = 0
                        s[si + 2] = 0
                        s[si + 4] += 1
                        if ph == 3 and s[si + 4] >= 4 and s[si + 4] < 8:
                            sf[b + 15] = sf[b + 5]
                            sf[b + 16] = sf[b + 6]
                        elif ph == 3 and s[si + 4] >= 8:
                            s[si + 5] += 1
                            s1 = (sf[b + 3] - sf[b + 4] - sf[b + 5] + sf[b + 6]) / (2.0 * a)
                            y0 = 0.5 * (sf[b + 4] + sf[b + 6])
                            s2 = (sf[b + 15] - 2.0 * sf[b + 16] + 2.0 * sf[b + 3] - 2.0 * sf[b + 4] + sf[b + 5]) / (
                                2.0 * a * a)
                            sf[b + 15] = sf[b + 5]
                            sf[b + 16] = sf[b + 6]
                            sf[b + 8] = s1
                            sf[b + 7] = y0
                            sf[b + 9] = s2
                            sf[b + 13] = y0
                            if mode == 1:
                                if s1 < 0.0:
                                    eps = s2 * y0 / (s1 * s1)
                                    w = _W_FACTOR * y0 / (-s1)
                                    sf[b + 11] = w
                                    du = -gain * eps * w / INFLECTION_SLOPE
                                else:
                                    eps = math.nan
                                    du = max_step
                                sf[b + 10] = eps
                                if abs(eps) < eps_tol:
                                    s[si + 6] += 1
                                else:
                                    s[si + 6] = 0
                            else:
                                if s2 < 0.0:
                                    du = gain * s1 / (-s2)
                                elif s1 > 0.0:
                                    du = max_step
                                elif s1 < 0.0:
                                    du = -max_step
                                else:
                                    du = 0.0
                                eps = s1 * a / y0 if y0 > 0.0 else math.nan
                                sf[b + 10] = eps
                                if abs(eps) < eps_tol:
                                    s[si + 6] += 1
                                else:
                                    s[si + 6] = 0
                            if du > max_step:
                                du = max_step
                            if du < -max_step:
                                du = -max_step
                            un = sf[b] + du
                            un = lo if un < lo else (hi if un > hi else un)
                            sf[b + 15] += s1 * (un - sf[b])
                            sf[b + 16] += s1 * (un - sf[b])
                            sf[b] = un
                        s[si + 1] = (ph + 1) % 4
            elif st == D_RECAPTURE:
                if sf[b] >= sf[b + 12] and y > sf[b + 14]:
                    sf[b + 14] = y
                if sf[b + 14] > rec_level * sf[b + 13] and y < (1.0 - rec_drop) * sf[b + 14]:
                    s[si] = D_LOCK
                    s[si + 2] = 0
                    s[si + 4] = 0
                    sf[b + 2] = 0.0
                    s[si + 3] = 0
                    sf[b + 13] = 0.0
                elif sf[b] >= hi or sf[b] >= sf[b + 12] + pf[11]:
                    s[si] = D_FAULT
                    sf[b] = sf[b + 12] - guard
                else:
                    sf[b] = sf[b] + recap if sf[b] + recap < hi else hi
            for j in range(J - 1, 0, -1):
                sf[hb + j] = sf[hb + j - 1]
            sf[hb] = y
            if hn < J:
                s[si + 8] = hn + 1
        ph = s[si + 1]
        off = 0.0
        if s[si] == D_LOCK:
            if ph == 0:
                off = a
            elif ph == 2:
                off = -a
        c = sf[b] + off
        c = lo if c < lo else (hi if c > hi else c)
        act[p[6] + i] = np.int64(np.rint(c))
    s[0] = 0


class DitherLockinCompiled(CompiledController):
    """Compiled :class:`DitherLockin` (07 P0-9 flat ABI). init_codes: scalar or (N,) shared by every world."""
    name = "dither_lockin_c"

    def __init__(self, params=None, **overrides):
        self.params = DitherLockin._make_params(params, overrides)
        _check(self.params, self.name)
        super().__init__(self.name, dither_init, dither_step, latency_ticks=int(self.params.latency_ticks))

    @classmethod
    def from_params(cls, params):
        return cls(params)

    def reset(self, spec: ControllerSpec) -> None:
        p = self.params
        cfg = DitherLockin._cfg(self, spec)
        init = _ring_list(p.init_codes, spec.n_rings, "init_codes", 0.0, self.name)
        self.params_i64 = np.concatenate([[cfg[0], cfg[2], cfg[3], int(cfg[6]), int(p.jump_steps),
                                           0 if p.init_codes is None else 1], np.rint(init)]).astype(np.int64)
        self.params_f64 = np.array([cfg[1], cfg[4], cfg[5], cfg[7], cfg[8], cfg[9], cfg[10], cfg[11], cfg[12],
                                    cfg[15]], dtype=float)
        super().reset(spec)

    def ring_values(self, k: int, ints: bool = False) -> np.ndarray:
        p_i, _, s_i, s_f = self.state
        n = self.spec.n_rings
        if ints:
            return np.stack([s_i[:, 1 + _D_SI * i + k] for i in range(n)], axis=1).astype(float)
        R = _D_SF + int(p_i[0, 11])
        return np.stack([s_f[:, i * R + k] for i in range(n)], axis=1)

    def telemetry(self) -> dict[str, np.ndarray]:
        return _d_telemetry(self.params, self.code_lo, self.code_hi, self.ring_values(0, True), self.ring_values(0),
                            self.ring_values(8), self.ring_values(9), self.ring_values(10), self.ring_values(7),
                            self.ring_values(6, True), self.ring_values(7, True))


register("dither_lockin_c", DitherLockinCompiled)
