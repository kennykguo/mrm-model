"""Bias-dither lock-in with the heater integrating its error (ARCHITECTURE section 6 item 3; 07 P0-7 fast_kernel;
04 C5 on the junction, C10). Compiled only: the dither and the demodulation run in the ``fast_kernel`` (once per
released monitor sample, inside the plant loop, like FPGA DSP fabric), the heater law in ``step``.

Why the bias
------------
A heater dither is slow: the thermal poles (3.2 us, 13 us) force dither periods of ~100 us and loop bandwidths of a
few hundred Hz (04 constraint 1; dither_lockin). The junction bias moves the resonance electro-optically, without
the thermal lag (07 P0-7: ~50 pm/V, 0.14 nm over the printed 1.0-3.75 V), so a bias dither can be demodulated at the
ADC sample rate. The bias is used here ONLY to sense; the correction is the heater's (an integrator on the lock-in
error). That is the difference from :mod:`mrm.control.baselines.midranging`, where the bias is the fast correcting
actuator and the heater re-centres it.

fast_kernel (per released monitor sample of ring i)
---------------------------------------------------
A three-level square dither on the bias offset: phases q = 0..3 apply +A, 0, -A, 0 bias codes for ``samples_per_phase``
samples each (4 samples = 16 us at 250 kHz: a 64 us cycle, 15.6 kHz, inside PAT-3's 1 kHz - 1 MHz band). The first
``blank_samples`` samples of each phase are discarded (their integrate-and-dump window straddles the offset change:
the offset written at release tick r acts from r + bias DAC latency), the rest averaged into y_q. Per complete cycle:

    S1 = (y_0 - y_1 - y_2 + y_3)/(2 A)   S2 = (y'_2 - 2 y'_3 + 2 y_0 - 2 y_1 + y_2)/(2 A^2)
    y0 = (y_1 + y_3)/2                   r = S1 A / y0

(drift-free and settling-symmetric estimators, y' the previous cycle; see dither_lockin).

r (the 1f/DC ratio, the dither's relative modulation) is independent of the laser power and of the monitor gain. On
the red side r < 0 and |r| grows with the detuning up to x = 1 (HWHM): r = -(A_delta/w) 2x/(1 + x^2) for a
Lorentzian, A_delta the dither amplitude in detuning (divided by the dynamic self-heating relief).

step (controller rate): hand-off, identification, lock
------------------------------------------------------
HOLD: ``hold_cycles`` cycles with the heater still; r_set = their mean r (or ``r_set`` if given): the lock holds the
hand-off detuning (e.g. a PI's fraction-of-peak point). Ratio mode needs a hand-off inside the half width (x < 1): for
a Lorentzian |r| is largest at x = 1 and takes the same value at x and 1/x, so beyond the half width the lock point
has an alias nearer the peak (and g_r changes sign at x = 1). ID: heater +``id_codes`` then -``id_codes``, ``id_cycles``
cycles each (the second half averaged): g_r = dr/du and g_y = dy0/du in heater codes, the loop gains (model-free,
self-heating relief and the driver law included). LOCK, once per new cycle:

- ``mode='ratio'`` (default): du = -gain (r - r_set)/g_r (a Newton step on the measured sensitivity);
- ``mode='inflection'``: the 2f zero (power and relief independent, see dither_lockin): eps = S2 y0/S1^2,
  W = 0.866 y0/|g_y| (half width in heater codes), du = -gain eps W/5.2; +max_step if S1 >= 0 (blue of the peak).

|du| <= ``max_step_codes``; the cycle after a heater move is skipped (it carries the heater's step response, which
would read as a dither response). A fall-off (monitor block mean drops by ``fall_frac`` of the DC to below ``fall_level``,
07 P0-1) stops the dither and ramps the heater up past a guard until the monitor peaked and fell back (as pi_lock),
then LOCK resumes with the same r_set and gains (a fall during HOLD or the identification restarts from HOLD).

The DC bias is ``bias_v`` (default 1.8 V +- 0.4 V, not the 1.05 V relOMA optimum: the dither needs room above the
1.0 V floor of the printed axis; the OMA cost is reported by the tests). Requires ``controller_interface.bind.tx.bias:
controller``. Known limits: r_set holds a detuning only while the dynamic relief and the bias efficiency are
unchanged (a laser power step changes the relief, so the lock point moves; tests measure it); the 2f signal is weak
(the bias moves the resonance only ~0.03 pm/code), so 'inflection' is noisy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numba as nb
import numpy as np

from mrm.control import api
from mrm.control.api import CompiledController, ControllerSpec
from mrm.control.baselines.pi_lock import NominalModel, _ring_list, make_params
from mrm.control.baselines.dither_lockin import INFLECTION_SLOPE, INFLECTION_WIDTH
from mrm.control.registry import register
from mrm.core.channels import ActId, ObsId

__all__ = ["BiasDither", "bias_init", "bias_step", "bias_fast", "demod_reference", "B_HOLD", "B_IDP", "B_IDM",
           "B_LOCK", "B_RECAP", "B_FAULT"]

B_HOLD, B_IDP, B_IDM, B_LOCK, B_RECAP, B_FAULT = 0, 1, 2, 3, 4, 5

# p_i64: [0 N, 1 off_bias_act, 2 off_heater_act, 3 off_samples, 4 off_count, 5 off_valid, 6 off_h_applied, 7 M,
#         8 samples_per_phase, 9 blank, 10 J, 11 has_init, 12 bias_dc_code, 13 hold_cycles, 14 id_cycles, 15 mode,
#         16 stagger, 17.. init codes (N)]
# p_f64: [0 A, 1 gain, 2 max_step, 3 id_codes, 4 fall_frac, 5 fall_level, 6 rec_level, 7 rec_drop, 8 guard,
#         9 recapture_step, 10 r_set (nan: from HOLD), 11 code_lo, 12 code_hi, 13 recapture_span]
# s_i64: [0 first] + per ring at 1 + 13 n: [st, phase, kph, nacc, nph, cycles, seen, st_count, n_fall, hist_n,
#         identified, skip, rset_ok]
# s_f64: per ring at n R (R = 26 + J): [uc, acc, y_q0, y_q1, y_q2, y_q3, s1, s2, y0, r, eps, r_set, g_r, g_y, acc_r,
#        acc_y, r_plus, y_plus, y_blk, y_ref, u_guard, y_seen, u_c0, w, p2, p3, hist (J)]
_NI = 17
_NUF = 11
_SI = 13
_SF = 26
_SAMPLES, _COUNT = int(ObsId.TX_MONITOR_SAMPLES), int(ObsId.TX_MONITOR_COUNT)
_VALID, _HAPPLIED = int(ObsId.MONITOR_VALID), int(ObsId.TX_HEATER_APPLIED)
_HEATER, _BIAS = int(ActId.TX_HEATER), int(ActId.TX_BIAS)


@nb.njit(cache=True)
def bias_init(spec_i64, spec_f64):
    n = spec_i64[api.SPEC_I_N_RINGS]
    p = np.zeros(_NI + n, dtype=np.int64)
    p[0] = n
    p[1] = api.act_offset(spec_i64, _BIAS)
    p[2] = api.act_offset(spec_i64, _HEATER)
    p[3] = api.obs_offset(spec_i64, _SAMPLES)
    p[4] = api.obs_offset(spec_i64, _COUNT)
    p[5] = api.obs_offset(spec_i64, _VALID)
    p[6] = api.obs_offset(spec_i64, _HAPPLIED)
    p[7] = spec_i64[api.SPEC_I_M]
    for k in range(9):
        p[8 + k] = api.user_i64(spec_i64, k)      # sp, blank, J, has_init, bias_dc, hold, id, mode, stagger
    for i in range(n):
        p[_NI + i] = api.user_i64(spec_i64, 9 + i)
    pf = np.zeros(_NUF + 3)
    for k in range(_NUF):
        pf[k] = api.user_f64(spec_f64, k)
    pf[11] = spec_i64[api.SPEC_I_HEATER_MIN]
    pf[12] = spec_i64[api.SPEC_I_HEATER_MAX]
    pf[13] = api.user_f64(spec_f64, _NUF)
    J = p[10]
    s_i = np.zeros(1 + _SI * n, dtype=np.int64)
    s_i[0] = 1
    s_f = np.zeros((_SF + J) * n)
    for i in range(n):
        s_i[1 + _SI * i] = B_HOLD
        s_i[1 + _SI * i + 1] = i % 4 if p[16] == 1 else 0
    return p, pf, s_i, s_f


@nb.njit(cache=True)
def _cycle_update(sf, b, a):
    """S1, S2, y0, r (and eps) of the cycle just completed (sf[b + 2 .. b + 5] = y_q)."""
    s1 = (sf[b + 2] - sf[b + 3] - sf[b + 4] + sf[b + 5]) / (2.0 * a)
    y0 = 0.5 * (sf[b + 3] + sf[b + 5])
    s2 = (sf[b + 24] - 2.0 * sf[b + 25] + 2.0 * sf[b + 2] - 2.0 * sf[b + 3] + sf[b + 4]) / (2.0 * a * a)
    sf[b + 24] = sf[b + 4]
    sf[b + 25] = sf[b + 5]
    sf[b + 6] = s1
    sf[b + 7] = s2
    sf[b + 8] = y0
    sf[b + 9] = s1 * a / y0 if y0 > 0.0 else 0.0
    sf[b + 10] = s2 * y0 / (s1 * s1) if s1 < 0.0 else math.nan


@nb.njit(cache=True)
def bias_fast(p, pf, s, sf, fobs, fact):
    """fast_kernel (07 P0-7): per released sample, the three-level bias dither and its demodulation. Writes the bias
    offset (codes) into fact[N + i]; the heater offset fact[i] stays 0."""
    n_r = fobs[1]
    h = p[8]
    blank = p[9]
    J = p[10]
    a = pf[0]
    R = _SF + J
    for i in range(n_r):
        si = 1 + _SI * i
        b = i * R
        st = s[si]
        active = st == B_HOLD or st == B_IDP or st == B_IDM or st == B_LOCK
        if fobs[api.FO_HEADER + i] == 1 and active:
            code = float(fobs[api.FO_HEADER + n_r + i])
            if s[si + 2] >= blank:
                sf[b + 1] += code
                s[si + 3] += 1
            s[si + 2] += 1
            if s[si + 2] == h:
                q = s[si + 1]
                sf[b + 2 + q] = sf[b + 1] / s[si + 3] if s[si + 3] > 0 else code
                sf[b + 1] = 0.0
                s[si + 3] = 0
                s[si + 2] = 0
                s[si + 4] += 1
                if q == 3 and s[si + 4] >= 4 and s[si + 4] < 8:
                    sf[b + 24] = sf[b + 4]          # prime the previous cycle
                    sf[b + 25] = sf[b + 5]
                elif q == 3 and s[si + 4] >= 8:
                    _cycle_update(sf, b, a)
                    s[si + 5] += 1
                s[si + 1] = (q + 1) % 4
        q = s[si + 1]
        off = 0
        if active:
            if q == 0:
                off = np.int64(np.rint(a))
            elif q == 2:
                off = -np.int64(np.rint(a))
        fact[n_r + i] = off
        fact[i] = 0


@nb.njit(cache=True)
def bias_step(p, pf, s, sf, obs, act):
    """Controller step: hand-off HOLD, heater identification, heater integrator on the lock-in error, fall-off and
    recapture (module docstring)."""
    n_r = p[0]
    m_max = p[7]
    J = p[10]
    hold_cycles = p[13]
    id_cycles = p[14]
    mode = p[15]
    gain = pf[1]
    max_step = pf[2]
    id_codes = pf[3]
    fall_frac = pf[4]
    fall_level = pf[5]
    rec_level = pf[6]
    rec_drop = pf[7]
    guard = pf[8]
    recap = pf[9]
    r_set_p = pf[10]
    lo = pf[11]
    hi = pf[12]
    R = _SF + J
    first = s[0] == 1
    for i in range(n_r):
        si = 1 + _SI * i
        b = i * R
        if first:
            if p[11] == 1:
                sf[b] = float(p[_NI + i])
            elif p[6] >= 0:
                sf[b] = float(obs[p[6] + i])
        cnt = obs[p[4] + i]
        ok = cnt > 0
        if p[5] >= 0 and obs[p[5] + i] == 0:
            ok = False
        if ok:
            acc = 0.0
            for m in range(cnt):
                acc += float(obs[p[3] + i * m_max + m])
            y = acc / cnt
            sf[b + 18] = y
            hb = b + _SF
            hn = s[si + 9]
            hmax = y
            if hn > 0:
                hmax = sf[hb]
                for j in range(1, hn):
                    if sf[hb + j] > hmax:
                        hmax = sf[hb + j]
            st = s[si]
            yref = sf[b + 19] if sf[b + 19] > 0.0 else hmax
            if (st == B_LOCK or st == B_HOLD or st == B_IDP or st == B_IDM) and hn >= 1 and \
                    (hmax - y) > fall_frac * yref and y < fall_level * yref:
                # a fall during the identification too (the +-id_codes heater step can push a ring off the fold):
                # it then restarts from HOLD after the recapture
                s[si] = B_RECAP
                s[si + 8] += 1
                sf[b + 20] = sf[b] + guard
                sf[b + 21] = 0.0
                sf[b + 19] = yref        # recapture level against the pre-fall level (never 0 in RECAP)
            elif st == B_RECAP:
                if sf[b] >= sf[b + 20] and y > sf[b + 21]:
                    sf[b + 21] = y
                if sf[b + 21] > rec_level * yref and y < (1.0 - rec_drop) * sf[b + 21]:
                    s[si] = B_LOCK if s[si + 10] > 0 else B_HOLD
                    s[si + 2] = 0
                    s[si + 3] = 0
                    s[si + 4] = 0
                    sf[b + 1] = 0.0
                    s[si + 6] = s[si + 5]
                    s[si + 7] = 0
                    sf[b + 14] = 0.0              # drop a partial HOLD / identification average
                    sf[b + 15] = 0.0
                elif sf[b] >= hi or sf[b] >= sf[b + 20] + pf[13]:
                    s[si] = B_FAULT                   # give up: back to the code at the fall (see pi_lock)
                    sf[b] = sf[b + 20] - guard
                else:
                    sf[b] = sf[b] + recap if sf[b] + recap < hi else hi
            for j in range(J - 1, 0, -1):
                sf[hb + j] = sf[hb + j - 1]
            sf[hb] = y
            if hn < J:
                s[si + 9] = hn + 1
        # one FSM update per newly completed dither cycle (the fast kernel counts them)
        st = s[si]
        if s[si + 5] != s[si + 6] and (st == B_HOLD or st == B_IDP or st == B_IDM or st == B_LOCK):
            s[si + 6] = s[si + 5]
            if s[si + 11] > 0:           # the cycle after a heater move carries its step response: skip it
                s[si + 11] -= 1
                act[p[2] + i] = np.int64(np.rint(sf[b]))
                act[p[1] + i] = p[12]
                continue
            r = sf[b + 9]
            y0 = sf[b + 8]
            if st == B_HOLD:
                sf[b + 14] += r
                s[si + 7] += 1
                if s[si + 7] >= hold_cycles:
                    if not math.isnan(r_set_p):
                        sf[b + 11] = r_set_p
                    elif s[si + 12] == 0:          # the hand-off r_set; a HOLD after a recapture keeps it (the
                        sf[b + 11] = sf[b + 14] / s[si + 7]   # recapture ends just red of the peak, near the fold)
                        s[si + 12] = 1
                    sf[b + 19] = y0
                    sf[b + 22] = sf[b]
                    sf[b] = sf[b + 22] + id_codes
                    s[si] = B_IDP
                    s[si + 7] = 0
                    sf[b + 14] = 0.0
                    sf[b + 15] = 0.0
            elif st == B_IDP or st == B_IDM:
                s[si + 7] += 1
                if s[si + 7] > id_cycles // 2:
                    sf[b + 14] += r
                    sf[b + 15] += y0
                if s[si + 7] >= id_cycles:
                    nu = id_cycles - id_cycles // 2
                    if st == B_IDP:
                        sf[b + 16] = sf[b + 14] / nu
                        sf[b + 17] = sf[b + 15] / nu
                        sf[b] = sf[b + 22] - id_codes
                        s[si] = B_IDM
                    else:
                        r_m = sf[b + 14] / nu
                        y_m = sf[b + 15] / nu
                        sf[b + 12] = (sf[b + 16] - r_m) / (2.0 * id_codes)
                        sf[b + 13] = (sf[b + 17] - y_m) / (2.0 * id_codes)
                        sf[b] = sf[b + 22]
                        s[si] = B_LOCK if sf[b + 12] != 0.0 and sf[b + 13] < 0.0 else B_FAULT
                        s[si + 10] = 1
                    s[si + 7] = 0
                    sf[b + 14] = 0.0
                    sf[b + 15] = 0.0
            else:
                sf[b + 19] = y0
                if mode == 0:
                    du = -gain * (r - sf[b + 11]) / sf[b + 12]
                else:
                    if sf[b + 6] < 0.0:
                        w = INFLECTION_WIDTH * y0 / (-sf[b + 13])
                        sf[b + 23] = w
                        du = -gain * sf[b + 10] * w / INFLECTION_SLOPE
                    else:
                        du = max_step
                if du > max_step:
                    du = max_step
                if du < -max_step:
                    du = -max_step
                un = sf[b] + du
                sf[b] = lo if un < lo else (hi if un > hi else un)
                if np.rint(un) != np.rint(un - du):
                    # skip the cycle carrying the heater step (two in inflection mode: the cross-cycle S2 also
                    # uses the previous cycle's last phases)
                    s[si + 11] = 2 if mode == 1 else 1
        act[p[2] + i] = np.int64(np.rint(sf[b]))
        act[p[1] + i] = p[12]
    s[0] = 0


def demod_reference(codes, samples_per_phase: int, blank: int, a: float, start_phase: int = 0):
    """Python reference of the fast-kernel demodulator (07 P0-7 test: bit-identical accumulators): the per-cycle
    (S1, S2, y0, r) of one ring's sample sequence, same operations in the same order."""
    acc, nacc, kph, nph, q = 0.0, 0, 0, 0, start_phase
    yq = [0.0, 0.0, 0.0, 0.0]
    p2 = p3 = 0.0
    out = []
    for c in codes:
        c = float(c)
        if kph >= blank:
            acc += c
            nacc += 1
        kph += 1
        if kph == samples_per_phase:
            yq[q] = acc / nacc if nacc > 0 else c
            acc, nacc, kph = 0.0, 0, 0
            nph += 1
            if q == 3 and 4 <= nph < 8:
                p2, p3 = yq[2], yq[3]
            elif q == 3 and nph >= 8:
                s1 = (yq[0] - yq[1] - yq[2] + yq[3]) / (2.0 * a)
                y0 = 0.5 * (yq[1] + yq[3])
                s2 = (p2 - 2.0 * p3 + 2.0 * yq[0] - 2.0 * yq[1] + yq[2]) / (2.0 * a * a)
                p2, p3 = yq[2], yq[3]
                out.append((s1, s2, y0, s1 * a / y0 if y0 > 0.0 else 0.0))
            q = (q + 1) % 4
    return out


@dataclass
class _BParams:
    mode: str = "ratio"              # ratio (hold r_set) | inflection (2f zero)
    bias_v: float = 1.8              # DC bias (V) on the printed 1.0-3.75 V axis; room for the dither above 1.0 V
    dither_codes: float = 600.0      # A, bias codes (0.4 V at 12 bits over 2.75 V)
    samples_per_phase: int = 4       # 4 phases per cycle: 16 samples = 64 us at 250 kHz (15.6 kHz)
    blank_samples: int = 1           # samples discarded at each phase change (window straddles the offset change)
    stagger: bool = True
    hold_cycles: int = 16            # hand-off: cycles averaged into r_set
    id_cycles: int = 12              # heater identification: cycles per +/- step (second half averaged)
    id_codes: float = 64.0           # heater identification step
    gain: float = 0.3                # per cycle (Newton step on the measured sensitivity)
    max_step_codes: float = 64.0
    r_set: float | None = None       # fixed 1f/DC set-point (None: the hand-off value)
    init_codes: object = None        # hand-off heater code(s), scalar or (N,); None = tx.heater.applied
    fall_frac: float = 0.35
    fall_level: float = 0.4
    recapture_level: float = 0.3
    recapture_drop: float = 0.02
    guard_codes: float = 500.0
    recapture_step_codes: float = 32.0
    recapture_span_codes: float = 8000.0
    jump_steps: int = 3
    latency_ticks: int = 0


class BiasDither(CompiledController):
    """Bias-dither lock-in, fast_kernel form, heater integrating the lock-in error (module docstring).

    Writes ``tx.heater`` and ``tx.bias`` (DC; the dither is the fast kernel's bias offset). Telemetry (B, N): state,
    locked, detuning_est_pm (NaN), integrator (heater code), integrator_saturated, ratio (r), ratio_set, eps,
    g_ratio_per_code, g_dc_per_code, dc_codes, fall_count.
    """
    name = "bias_dither"
    Params = _BParams

    def __init__(self, params=None, **overrides):
        self.params = make_params(_BParams, params, overrides, self.name)
        p = self.params
        if p.mode not in ("ratio", "inflection"):
            raise ValueError(f"{self.name}: mode must be ratio | inflection, got {p.mode!r}")
        if p.samples_per_phase < 1 or not 0 <= p.blank_samples < p.samples_per_phase:
            raise ValueError(f"{self.name}: need 0 <= blank_samples < samples_per_phase")
        if p.id_cycles < 2 or p.hold_cycles < 1 or p.dither_codes <= 0 or p.id_codes <= 0:
            raise ValueError(f"{self.name}: id_cycles >= 2, hold_cycles >= 1, dither_codes and id_codes > 0")
        super().__init__(self.name, bias_init, bias_step, latency_ticks=int(p.latency_ticks), fast_fn=bias_fast)

    @classmethod
    def from_params(cls, params):
        return cls(params)

    def reset(self, spec: ControllerSpec) -> None:
        p = self.params
        if spec.bindings.get("tx.bias") != "controller":
            raise ValueError(f"{self.name}: needs controller_interface.bind.tx.bias: controller (the scenario owns "
                             f"the bias in this config)")
        self.nominal = NominalModel.from_spec(spec)
        bd = spec.bias_dac
        v_lo, v_hi = (bd.phys_min or 1.0), (bd.phys_max or 3.75)
        code = int(round((p.bias_v - v_lo) / (v_hi - v_lo) * bd.code_max))
        a = float(p.dither_codes)
        if code - a < 0 or code + a > bd.code_max:
            raise ValueError(f"{self.name}: bias_v {p.bias_v} V +- {a} codes leaves the bias DAC range")
        self.bias_dc_code = code
        self.code_lo, self.code_hi = float(spec.heater_dac.code_min), float(spec.heater_dac.code_max)
        init = _ring_list(p.init_codes, spec.n_rings, "init_codes", 0.0, self.name)
        modes = {"ratio": 0, "inflection": 1}
        self.params_i64 = np.concatenate([[p.samples_per_phase, p.blank_samples, p.jump_steps,
                                           0 if p.init_codes is None else 1, code, p.hold_cycles, p.id_cycles,
                                           modes[p.mode], 1 if p.stagger else 0], np.rint(init)]).astype(np.int64)
        self.params_f64 = np.array([a, p.gain, p.max_step_codes, p.id_codes, p.fall_frac, p.fall_level,
                                    p.recapture_level, p.recapture_drop, p.guard_codes, p.recapture_step_codes,
                                    math.nan if p.r_set is None else p.r_set, p.recapture_span_codes], dtype=float)
        super().reset(spec)

    def ring_values(self, k: int, ints: bool = False) -> np.ndarray:
        p_i, _, s_i, s_f = self.state
        n = self.spec.n_rings
        if ints:
            return np.stack([s_i[:, 1 + _SI * i + k] for i in range(n)], axis=1).astype(float)
        R = _SF + int(p_i[0, 10])
        return np.stack([s_f[:, i * R + k] for i in range(n)], axis=1)

    def telemetry(self) -> dict[str, np.ndarray]:
        st = self.ring_values(0, True)
        u = self.ring_values(0)
        lock = st == B_LOCK
        r, rs = self.ring_values(9), self.ring_values(11)
        return {"state": st, "locked": lock & (np.abs(r - rs) < 0.1 * np.abs(rs) + 1e-12), "detuning_est_pm":
                np.full(st.shape, np.nan), "integrator": u,
                "integrator_saturated": lock & ((u <= self.code_lo) | (u >= self.code_hi)), "ratio": r,
                "ratio_set": rs, "eps": self.ring_values(10), "g_ratio_per_code": self.ring_values(12),
                "g_dc_per_code": self.ring_values(13), "dc_codes": self.ring_values(8),
                "fall_count": self.ring_values(8, True), "cycles": self.ring_values(5, True)}


register("bias_dither", BiasDither)
