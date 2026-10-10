"""Sensing plant: monitor photodiode, analog front end (AFE), ADC, temperature sensors, heater readback, faults.

Scope (docs/ARCHITECTURE.md section 2; 07 = docs/research/07_design_critique.md): 07 P1-9 (monitor PD, Arrhenius dark
current, background light, optional reference tap), P1-10 (converters and sensors as hardware, fault masks), P1-13
(exact noise discretization, counter RNG), P0-7 (monitor sample blocks with sample ticks), P0-6 (one owner per
converter delay: the AFE window and the ADC latency live here).

Units are SI everywhere: W, A, A^2/Hz (one-sided power spectral densities), K, s, Hz. Codes are int64 inside the
kernels and are cast to the ChannelSpec dtype only by the Python classes.

Signal chain of one monitor channel (ring n)::

    optical power at the PD   P_pd (drop port: every line and order, 07 P1-9) + background_fraction * P_bus
      -> photocurrent         i_sig = gain_scale * R * (P_pd + f_bg * P_bus)                      (07 P1-9)
      +  dark current         i_d = I_d0 exp[-(E_a/k)(1/T_pd - 1/T_0)]                            (07 D20, P1-9)
      -> AFE  integrate_dump: boxcar mean of the current over the window (s - W, s]               (07 P0-7, PAT-2)
              rc:             first-order low pass y' = (i - y)/tau, sampled at s
         + white noise: shot 2 q (i_sig + i_d) (fidelity.shot_noise) and TIA input noise S_tia   (one-sided)
         + flicker: S_tia * f_c / f, as a sum of Ornstein-Uhlenbeck processes                     (07 P1-10)
      -> baseline subtraction (the patent's analog baseline subtraction, PAT-2)
      -> PGA gain (monitor.pga_select)                                                           (07 P1-10)
      -> ADC: code = x g / LSB (1 + gain_err) / (1 + tc (T_adc - T_ref)) + offset + INL bow, then quantized with
              per-code threshold errors (DNL); clipped to 0 .. 2^bits - 1 with the overrange flag
      -> faults: adc_saturate, adc_stuck, adc_drop                                               (07 P1-10)
      -> latency FIFO (ring.monitor.adc_latency_s), released at controller edges as a sample block (07 P0-7)

Exact noise discretization (07 P1-13: "the discrete-time Gramian of the filter, and a variance of S/(2 T_w) for an
integrate-and-dump window. Never use PSD x bandwidth.")
- integrate_dump: the window mean of white noise with one-sided PSD S(t) is Gaussian with variance
  (1/T_w^2) * integral of S(t)/2 dt. The kernel accumulates sum_k S_k/2 over the window ticks and draws ONE normal at
  the dump: Var = sum_k (S_k/2) dt / (n dt)^2, which is S/(2 T_w) for a constant S, independent of dt.
- rc: the noise state obeys dy = -mu y dt + mu dn, mu = 1/tau, white n of two-sided intensity S/2. Over one tick
  the exact transition is y <- a y + w, a = exp(-mu dt), Var(w) = (S mu/4)(1 - a^2) (the discrete Gramian); the
  stationary variance is S/(4 tau). The kernel keeps the decayed variance accumulated since the last sample and draws
  one normal per sample, so the sampled noise is exact for piecewise-constant S and any dt.
- flicker: J OU processes with log-spaced corners f_j (``per_decade`` per decade, equal variances
  sigma_j^2 = S f_c ln r, r = 10^(1/per_decade)) sum to the one-sided PSD S f_c / f between f_min and f_max. Each
  process is advanced from sample to sample with its exact 2 x 2 transition and Gramian (closed forms in
  :func:`ou_window_coefficients` / :func:`ou_rc_coefficients`), jointly with its contribution to the sample (the
  window mean, or the RC output). Flicker is therefore generated at the sample rate and is dt-invariant.
- laser RIN is applied to the line power per tick by mrm/plant/laser.py (exact tick averages of white RIN), so it
  reaches the monitor through the optics and is correlated across rings, as it is on the chip.

Tick convention (mrm/core/timing.py): tick k is the instant k*dt; ``*_tick_row(k, ...)`` integrates the plant
interval [k, k+1). A sample with sample tick s covers the intervals s - W .. s - 1, i.e. the instants (s - W, s];
it is taken when the kernel processes interval s - 1, and it is released by ``*_release_row(now, ...)`` at the first
call with ``now >= s + latency``.

Randomness (07 P1-13): counter RNG of mrm/core/rng.py, keyed by (seed_b, stream, ring, sample tick, slot):
- TIA white noise and flicker: stream ``adc_monitor`` (``adc_monitor_ref`` for a reference tap), noise seed;
- shot noise: ``Stream.SHOT_NOISE``, noise seed (a separate stream, so toggling fidelity.shot_noise leaves the ADC
  noise draws unchanged);
- DNL thresholds and the INL bow: streams ``adc_monitor_dnl`` / ``adc_monitor_inl``, DEVICE seed (a device property;
  Plant.reset(noise_seed) never resamples it);
- temperature sensors ``adc_temp`` (ring index = sensor index), heater readback ``adc_readback``.
A second monitor bank (the reference tap) uses its own stream names and a slot offset (``slot_base``).

Kernels: every per-tick function is numba @njit, takes arrays and scalars only, and runs under NUMBA_DISABLE_JIT=1.
``*_row`` works on one world (arrays (N,) or (S,)) for use inside a ``prange`` over B; the batched form loops over b
(``*_par`` uses prange; results do not depend on the thread count). The Python classes hold the state.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import NamedTuple

import numba as nb
import numpy as np
from numba import njit

from mrm.core import rng as _rng
from mrm.core.timing import (fifo_capacity, fifo_pop_ready, fifo_push, next_edge, rate_to_period_ticks,
                             seconds_to_ticks)
from mrm.units import K_B, Q_E

__all__ = [
    # enums, flags, constants
    "AFE_RC", "AFE_INTEGRATE_DUMP", "AFE_TYPES", "FAULT_ADC_DROP", "FAULT_ADC_STUCK", "FAULT_ADC_SATURATE",
    "FAULT_DAC_GLITCH", "SAMPLE_OVERRANGE", "SAMPLE_INVALID", "SAMPLE_DROPPED", "TEMP_SENSOR_TYPES",
    "PR_RESP", "PR_BG", "PR_IBASE", "PR_GAIN_ERR", "PR_OFFSET", "PR_INL_BOW", "NPR",
    "PW_GAIN_SCALE", "PW_DARK_I0", "PW_DARK_T0", "PW_DARK_EA", "PW_S_TIA", "PW_LSB", "PW_TEMPCO", "PW_T_REF",
    "PW_DNL", "PW_A_TICK", "PW_MU", "PW_DT", "NPW",
    "REF_TAP_SLOT_BASE", "PAT2_T_INT_S", "PAT2_C_INT_F", "PAT2_V_REF_V", "ADC_REF_T_K_DEFAULT",
    # numba kernels
    "dark_current_a", "adc_quantize", "flicker_step", "monitor_tick_row", "monitor_tick", "monitor_tick_par",
    "monitor_release_row", "monitor_release", "monitor_level_codes_row",
    "temp_tick_row", "temp_tick", "temp_release_row", "temp_release",
    "readback_tick_row", "readback_tick", "readback_release_row", "readback_release",
    "dac_glitch_row", "dac_glitch", "fault_mask_row", "fault_mask",
    # Python helpers
    "flicker_pole_set", "flicker_psd", "ou_window_coefficients", "ou_rc_coefficients", "flicker_coefficients",
    "draw_inl_bow_lsb", "adc_channel_dtype", "integrator_full_scale_a",
    # Python API
    "MonitorHw", "MonitorBank", "MonitorRelease", "TempSensorsHw", "TempSensorBank", "ReadbackHw", "ReadbackBank",
    "SensingBank",
]

# --------------------------------------------------------------------------------------------- enums and flags
AFE_RC = 0                 # same codes as mrm.params.realize.MonitorParams.afe_type (0 rc, 1 integrate_dump)
AFE_INTEGRATE_DUMP = 1
AFE_TYPES = {"rc": AFE_RC, "integrate_dump": AFE_INTEGRATE_DUMP}

# fault mask bits (07 P1-10: scenario-bound channel masks), one int8 per (B, N) per tick
FAULT_ADC_DROP = 1         # the sample is lost: not delivered; monitor_valid goes False
FAULT_ADC_STUCK = 2        # the converter repeats its last good code; monitor_valid goes False if flagged
FAULT_ADC_SATURATE = 4     # the converter output is forced to full scale; adc_overrange goes True
FAULT_DAC_GLITCH = 8       # a DAC output code glitch for that tick (see dac_glitch_row)

# flags carried with each sample through the latency FIFO (packed above bit 40 of the int64 FIFO value)
SAMPLE_OVERRANGE = 1
SAMPLE_INVALID = 2
SAMPLE_DROPPED = 4
_FLAG_SHIFT = 40
_CODE_MASK = (1 << 40) - 1

TEMP_SENSOR_TYPES = {"diode": 0, "rtd": 1, "thermistor": 2}

# packed per-ring float parameters of a monitor bank, pr (B, N, NPR)
PR_RESP = 0       # responsivity, A/W
PR_BG = 1         # background (stray light) fraction of the bus power, 1
PR_IBASE = 2      # analog baseline current subtracted before the PGA, A
PR_GAIN_ERR = 3   # ADC gain error, fraction
PR_OFFSET = 4     # ADC offset, codes
PR_INL_BOW = 5    # INL bow amplitude, LSB (per device)
NPR = 6
# packed per-world float parameters, pw (B, NPW)
PW_GAIN_SCALE = 0  # lumped monitor-gain nuisance factor (07 P1-9), 1
PW_DARK_I0 = 1     # dark current at T0, A
PW_DARK_T0 = 2     # reference temperature of the dark current, K
PW_DARK_EA = 3     # activation energy, J
PW_S_TIA = 4       # input-referred white current noise, one-sided A^2/Hz
PW_LSB = 5         # ADC LSB at PGA gain 1, A per code
PW_TEMPCO = 6      # ADC reference tempco, 1/K
PW_T_REF = 7       # temperature at which the ADC reference is exact, K
PW_DNL = 8         # DNL threshold spread, LSB (peak |DNL| < this value)
PW_A_TICK = 9      # rc AFE: exp(-dt/tau) (unused for integrate_dump)
PW_MU = 10         # rc AFE: 1/tau, 1/s (0 for integrate_dump)
PW_DT = 11         # plant tick, s
NPW = 12

# per-monitor-ring float state sf (B, N, NSF) and int state si (B, N, NSI)
_SF_ACC_I = 0      # integrate_dump: sum of i_k over the window ticks, A
_SF_ACC_VS = 1     # shot noise: int-dump sum of S_k/2 (A^2/Hz); rc decayed variance since the last sample (A^2)
_SF_ACC_VT = 2     # TIA white noise: same as above
_SF_ACC_N = 3      # integrate_dump: ticks accumulated in the window
_SF_YDET = 4       # rc: deterministic output, A
_SF_YNS = 5        # rc: shot-noise output at the last sample, A
_SF_YNT = 6        # rc: TIA-noise output at the last sample, A
_SF_DEC = 7        # rc: exp(-mu (t - t_last_sample))
NSF = 8
_SI_PRIMED = 0
_SI_LAST_CODE = 1
NSI = 2

# counter-RNG slots (07 P1-13): offsets inside one bank; a bank adds its slot_base
_SLOT_WHITE = 0
_SLOT_FLICKER = 16          # + pole index j (j < 1024)
_SLOT_FL_INIT = 2048        # + pole index j
_SLOT_INL = 0
REF_TAP_SLOT_BASE = 1 << 16  # slot offset of a reference-tap bank (design choice; < rng.SLOT_MAX)

# --------------------------------------------------------------------- hardware constants (provenance stated)
PAT2_T_INT_S = 2e-6          # printed example, PAT-2 (US11409045): T_int = 2 us = 400 x 5 ns (range 1-10 us)
PAT2_C_INT_F = 16e-12        # printed example, PAT-2: integration capacitor 16 pF (range 10-20 pF)
PAT2_V_REF_V = 0.5           # printed, PAT-2: Vref about 500 mV
ADC_REF_T_K_DEFAULT = 298.15  # assumed: the ADC reference is trimmed at 25 C (07 CAL-3 wafer maps are at 25 C)
_DEFAULT_DARK_EA_EV = 0.37   # inferred, 02 section 10 / mrm.materials.ge_pd_dark_ea_ev() (range 0.35-0.38 eV)
_DEFAULT_T0_K = 298.15       # printed, GF: dark current specified at 25 C (07 D20)
_FLICKER_F_MIN_HZ = 1e-3     # assumed: lowest flicker pole (~17 min); below it 1/f noise acts as a static offset
_FLICKER_PER_DECADE = 2      # design choice: 2 per decade: < 0.05 dB ripple in band, -0.27 dB a decade from an edge (tested)
_TEMP_GAIN_REF_K = 298.15    # assumed: a temperature sensor's gain error pivots about its 25 C trim point
_RC_NUDGE = 1e-4             # design choice: an OU corner within 1e-4 of the RC corner is moved by 1e-4 (closed
#                              forms are singular at lambda = mu; the model error is ~1e-4 of one pole's variance)


# =============================================================================================== small kernels
@njit(cache=True)
def dark_current_a(i0_a, t_k, t0_k, ea_j):
    """Arrhenius dark current, A: I_d0 exp[-(E_a/k)(1/T - 1/T_0)] (07 P1-9, D20; 02 section 10).

    i0_a: dark current at t0_k (A); t_k, t0_k in K; ea_j: activation energy in J (E_a 0.35-0.38 eV, inferred, 02).
    Returns 0 for i0 <= 0 (no dark current) and for non-positive temperatures.
    """
    if not (i0_a > 0.0) or not (t_k > 0.0) or not (t0_k > 0.0):
        return 0.0
    return i0_a * math.exp(-(ea_j / K_B) * (1.0 / t_k - 1.0 / t0_k))


@njit(cache=True)
def _round_clip(x, cmax):
    """round(x) clipped to 0 .. cmax as int64 (NaN -> 0; the clip happens before the float -> int conversion)."""
    if not (x == x) or x < 0.0:
        return np.int64(0)
    if x > cmax:
        return np.int64(cmax)
    return np.int64(math.floor(x + 0.5))


@njit(cache=True)
def _dnl_threshold(c, dnl, key, ring, slot):
    """Lower decision threshold of code c (in code units): c - 0.5 + dnl (u_c - 0.5), u_c uniform per device."""
    return c - 0.5 + dnl * (_rng.uniform(key, ring, c, slot) - 0.5)


@njit(cache=True)
def adc_quantize(v, inl_bow_lsb, dnl_lsb, code_max, key, ring, slot):
    """Quantize an ideal (real-valued) code v into an integer code with INL and DNL. Returns (code, overrange).

    INL: a smooth bow v += bow * 4 u (1 - u), u = v/code_max (07 P1-10 adc_inl_lsb; per device). DNL: the decision
    threshold of code c is T_c = c - 0.5 + d (u_c - 0.5), u_c uniform from the device stream, so code widths are
    1 + d (u_{c+1} - u_c) and |DNL| < d; d <= 1 keeps the thresholds monotone (d = 1 allows missing codes).
    Saturation: v below -0.5 or at/above code_max + 0.5 (or NaN) clips to 0 / code_max with overrange = 1.
    """
    if not (v == v):
        return np.int64(0), 1
    over = 0
    if v < -0.5 or v >= code_max + 0.5:
        over = 1
    if inl_bow_lsb != 0.0:
        u = v / code_max
        if u < 0.0:
            u = 0.0
        elif u > 1.0:
            u = 1.0
        v = v + inl_bow_lsb * 4.0 * u * (1.0 - u)
    if v < -0.5:
        return np.int64(0), 1
    if v >= code_max + 0.5:
        return np.int64(code_max), 1
    c = np.int64(math.floor(v + 0.5))
    if c < 0:
        c = np.int64(0)
    elif c > code_max:
        c = np.int64(code_max)
    if dnl_lsb > 0.0:
        while c > 0 and v < _dnl_threshold(c, dnl_lsb, key, ring, slot):
            c -= 1
        while c < code_max and v >= _dnl_threshold(c + 1, dnl_lsb, key, ring, slot):
            c += 1
    return c, over


@njit(cache=True)
def flicker_step(key, ring, tick, slot0, coef, fx, fc):
    """Advance the J flicker OU processes of one channel by one sample period; return the summed contribution.

    State per pole j: fx[j] = OU value at the last sample instant, fc[j] = its contribution to the last sample (the
    window mean for integrate_dump, the RC output for rc). coef[j] = [P00, P01, P10, P11, L00, L10, L11, S00, S10,
    S11]: new (x, c) = P (x, c) + L z, L lower-triangular (z = two normals keyed by (ring, tick, slot0 + j)).
    """
    tot = 0.0
    for j in range(coef.shape[0]):
        z1, z2 = _rng.normal2(key, ring, tick, slot0 + j)
        x0 = fx[j]
        c0 = fc[j]
        xn = coef[j, 0] * x0 + coef[j, 1] * c0 + coef[j, 4] * z1
        cn = coef[j, 2] * x0 + coef[j, 3] * c0 + coef[j, 5] * z1 + coef[j, 6] * z2
        fx[j] = xn
        fc[j] = cn
        tot += cn
    return tot


@njit(cache=True)
def _flicker_init(key, ring, tick, slot0, coef, fx, fc):
    """Draw the flicker states from their stationary joint distribution (Cholesky S of coef[:, 7:10])."""
    for j in range(coef.shape[0]):
        z1, z2 = _rng.normal2(key, ring, tick, slot0 + j)
        fx[j] = coef[j, 7] * z1
        fc[j] = coef[j, 8] * z1 + coef[j, 9] * z2


# ============================================================================================= monitor kernels
@njit(cache=True)
def monitor_tick_row(k, p_pd_w, p_bg_w, t_pd_k, t_adc_k, pga_sel, fault,
                     pr, pw, lat, phase, pga_gains, fl_coef,
                     afe_type, window_ticks, period_ticks, code_max, shot_on, stuck_flagged, slot_base,
                     key_adc, key_shot, key_dnl,
                     sf, si, fx, fc, fv, fr, fm, out_sampled, i_dark_in=None):
    """Integrate plant interval [k, k+1) for the N monitor channels of one world; take, convert and queue the samples
    whose sample tick is k + 1. i_dark_in (N,), optional: the PD dark currents dark_current_a(pw[PW_DARK_I0], t_pd_k,
    pw[PW_DARK_T0], pw[PW_DARK_EA]) already computed by the caller (the engine shares them with Truth).

    Inputs (N,): p_pd_w optical power at the PD (W; the drop port sums every line and order, 07 P1-9), p_bg_w the
    bus power that the background fraction applies to (W), t_pd_k PD temperature (K, ROM observer monitor_pd),
    t_adc_k ADC reference temperature (K), pga_sel PGA index (int64, clipped), fault mask (int8, FAULT_*).
    Parameters: pr (N, NPR), pw (NPW,), lat (ADC latency, ticks), phase (N,) sample phases (ticks, adc_mux),
    pga_gains (G,) (empty = gain 1), fl_coef (J, 10) flicker coefficients (J = 0: no flicker).
    Switches: afe_type (AFE_*), window_ticks W, period_ticks P (W <= P), code_max = 2^bits - 1, shot_on, stuck_flagged
    (a stuck converter marks its samples invalid), slot_base (counter-RNG slot offset of this bank).
    Keys (uint64): key_adc (TIA + flicker, noise seed), key_shot (shot noise, noise seed), key_dnl (DNL, device seed).
    State: sf (N, NSF), si (N, NSI), fx/fc (N, J), FIFO fv/fr/fm (N, C) (mrm.core.timing).
    out_sampled (N,) int8: 0 no sample, 1 sample queued, 2 sample lost because the FIFO was full.
    """
    n_rings = p_pd_w.shape[0]
    n_pga = pga_gains.shape[0]
    gs = pw[PW_GAIN_SCALE]
    i0 = pw[PW_DARK_I0]
    t0 = pw[PW_DARK_T0]
    ea = pw[PW_DARK_EA]
    s_tia = pw[PW_S_TIA]
    a = pw[PW_A_TICK]
    mu = pw[PW_MU]
    dt = pw[PW_DT]
    one_m_a2 = 1.0 - a * a
    for n in range(n_rings):
        out_sampled[n] = 0
        i_sig = gs * pr[n, PR_RESP] * (p_pd_w[n] + pr[n, PR_BG] * p_bg_w[n])
        if i_dark_in is None:
            i_dark = dark_current_a(i0, t_pd_k[n], t0, ea)
        else:
            i_dark = i_dark_in[n]
        i_tot = i_sig + i_dark
        s_shot = 2.0 * Q_E * (abs(i_sig) + i_dark) if shot_on else 0.0    # one-sided shot-noise PSD, A^2/Hz
        s = next_edge(k + 1, period_ticks, phase[n])                       # next sample tick (>= k + 1)
        if si[n, _SI_PRIMED] == 0:
            sf[n, _SF_ACC_I] = 0.0
            sf[n, _SF_ACC_N] = 0.0
            sf[n, _SF_YDET] = i_tot                     # rc: start in steady state
            sf[n, _SF_YNS] = 0.0
            sf[n, _SF_YNT] = 0.0
            sf[n, _SF_DEC] = 1.0
            if afe_type == AFE_RC:                      # stationary start: accumulated variance = S/(4 tau)
                sf[n, _SF_ACC_VS] = 0.25 * s_shot * mu
                sf[n, _SF_ACC_VT] = 0.25 * s_tia * mu
            else:
                sf[n, _SF_ACC_VS] = 0.0
                sf[n, _SF_ACC_VT] = 0.0
            if fl_coef.shape[0] > 0:
                _flicker_init(key_adc, n, k, slot_base + _SLOT_FL_INIT, fl_coef, fx[n], fc[n])
            si[n, _SI_PRIMED] = 1
        if afe_type == AFE_INTEGRATE_DUMP:
            if k >= s - window_ticks:
                sf[n, _SF_ACC_I] += i_tot
                sf[n, _SF_ACC_VS] += 0.5 * s_shot
                sf[n, _SF_ACC_VT] += 0.5 * s_tia
                sf[n, _SF_ACC_N] += 1.0
        else:
            sf[n, _SF_YDET] = a * sf[n, _SF_YDET] + (1.0 - a) * i_tot       # exact for a ZOH input
            sf[n, _SF_DEC] *= a
            sf[n, _SF_ACC_VS] = a * a * sf[n, _SF_ACC_VS] + 0.25 * s_shot * mu * one_m_a2   # discrete Gramian
            sf[n, _SF_ACC_VT] = a * a * sf[n, _SF_ACC_VT] + 0.25 * s_tia * mu * one_m_a2
        if k + 1 != s:
            continue
        # ---- take the sample with sample tick s
        z_t = _rng.normal(key_adc, n, s, slot_base + _SLOT_WHITE)
        z_s = _rng.normal(key_shot, n, s, slot_base) if shot_on else 0.0
        if afe_type == AFE_INTEGRATE_DUMP:
            m = sf[n, _SF_ACC_N]
            if m < 1.0:
                m = 1.0
            x = sf[n, _SF_ACC_I] / m
            x += math.sqrt(sf[n, _SF_ACC_VS] / (m * m * dt)) * z_s + math.sqrt(sf[n, _SF_ACC_VT] / (m * m * dt)) * z_t
            sf[n, _SF_ACC_I] = 0.0
            sf[n, _SF_ACC_VS] = 0.0
            sf[n, _SF_ACC_VT] = 0.0
            sf[n, _SF_ACC_N] = 0.0
        else:
            ys = sf[n, _SF_DEC] * sf[n, _SF_YNS] + math.sqrt(sf[n, _SF_ACC_VS]) * z_s
            yt = sf[n, _SF_DEC] * sf[n, _SF_YNT] + math.sqrt(sf[n, _SF_ACC_VT]) * z_t
            sf[n, _SF_YNS] = ys
            sf[n, _SF_YNT] = yt
            sf[n, _SF_ACC_VS] = 0.0
            sf[n, _SF_ACC_VT] = 0.0
            sf[n, _SF_DEC] = 1.0
            x = sf[n, _SF_YDET] + ys + yt
        if fl_coef.shape[0] > 0:
            x += flicker_step(key_adc, n, s, slot_base + _SLOT_FLICKER, fl_coef, fx[n], fc[n])
        # ---- baseline subtraction, PGA, ADC
        x -= pr[n, PR_IBASE]
        g = 1.0
        if n_pga > 0:
            sel = pga_sel[n]
            if sel < 0:
                sel = 0
            elif sel > n_pga - 1:
                sel = n_pga - 1
            g = pga_gains[sel]
        ref = 1.0 + pw[PW_TEMPCO] * (t_adc_k[n] - pw[PW_T_REF])
        v = x * g / pw[PW_LSB] * (1.0 + pr[n, PR_GAIN_ERR]) / ref + pr[n, PR_OFFSET]
        code, over = adc_quantize(v, pr[n, PR_INL_BOW], pw[PW_DNL], code_max, key_dnl, n, slot_base)
        flags = SAMPLE_OVERRANGE if over else 0
        f = fault[n]
        if f & FAULT_ADC_SATURATE:
            code = np.int64(code_max)
            flags |= SAMPLE_OVERRANGE
        if f & FAULT_ADC_STUCK:
            # the frozen converter reports its last good code; the clip state of the hidden conversion is not seen
            code = si[n, _SI_LAST_CODE]
            flags = SAMPLE_INVALID if stuck_flagged else 0
        if f & FAULT_ADC_DROP:
            flags |= SAMPLE_INVALID | SAMPLE_DROPPED
        if not (f & (FAULT_ADC_STUCK | FAULT_ADC_DROP)):
            si[n, _SI_LAST_CODE] = code
        ok = fifo_push(fv, fr, fm, n, code | (np.int64(flags) << _FLAG_SHIFT), s + lat)
        out_sampled[n] = 1 if ok else 2


@njit(cache=True)
def monitor_tick(k, p_pd_w, p_bg_w, t_pd_k, t_adc_k, pga_sel, fault, pr, pw, lat, phase, pga_gains, fl_coef,
                 afe_type, window_ticks, period_ticks, code_max, shot_on, stuck_flagged, slot_base,
                 key_adc, key_shot, key_dnl, sf, si, fx, fc, fv, fr, fm, out_sampled):
    """Batched :func:`monitor_tick_row`: (B, N) inputs, pr (B, N, NPR), pw (B, NPW), lat (B,), fl_coef (B, J, 10),
    keys (B,) uint64, state (B, ...). Sequential loop over b."""
    for b in range(p_pd_w.shape[0]):
        monitor_tick_row(k, p_pd_w[b], p_bg_w[b], t_pd_k[b], t_adc_k[b], pga_sel[b], fault[b], pr[b], pw[b],
                         lat[b], phase, pga_gains, fl_coef[b], afe_type, window_ticks, period_ticks, code_max,
                         shot_on, stuck_flagged, slot_base, key_adc[b], key_shot[b], key_dnl[b], sf[b], si[b],
                         fx[b], fc[b], fv[b], fr[b], fm[b], out_sampled[b])


@njit(cache=True, parallel=True)
def monitor_tick_par(k, p_pd_w, p_bg_w, t_pd_k, t_adc_k, pga_sel, fault, pr, pw, lat, phase, pga_gains, fl_coef,
                     afe_type, window_ticks, period_ticks, code_max, shot_on, stuck_flagged, slot_base,
                     key_adc, key_shot, key_dnl, sf, si, fx, fc, fv, fr, fm, out_sampled):
    """:func:`monitor_tick` with ``prange`` over b (07 P0-9). Bit-identical to the sequential form for any thread
    count, because every draw is a pure function of (seed_b, stream, ring, tick, slot)."""
    for b in nb.prange(p_pd_w.shape[0]):
        monitor_tick_row(k, p_pd_w[b], p_bg_w[b], t_pd_k[b], t_adc_k[b], pga_sel[b], fault[b], pr[b], pw[b],
                         lat[b], phase, pga_gains, fl_coef[b], afe_type, window_ticks, period_ticks, code_max,
                         shot_on, stuck_flagged, slot_base, key_adc[b], key_shot[b], key_dnl[b], sf[b], si[b],
                         fx[b], fc[b], fv[b], fr[b], fm[b], out_sampled[b])


@njit(cache=True)
def monitor_release_row(now, lat, fv, fr, fm, out_samples, out_ticks, out_count, latest, out_valid, out_over,
                        out_new):
    """Release every queued sample of one world whose release tick (sample tick + latency) is <= now (07 P0-6 step 2).

    Writes the block channels: out_samples (N, M) codes and out_ticks (N, M) sample ticks, valid in [:count] and in
    sample-tick order, zeros after; out_count (N,); latest (N,) is HELD (updated only when a sample arrives);
    out_valid (N,) False if any released sample was dropped or stuck (07 P1-10 monitor_valid); out_over (N,) True if
    any released sample clipped or was saturated (adc_overrange); out_new (N,) True if at least one sample arrived.
    More than M samples (only with a release period above the declared one): the newest M are kept. Returns the
    number of samples dropped that way.
    """
    n_rings = fm.shape[0]
    m_max = out_samples.shape[1]
    lost = 0
    for n in range(n_rings):
        cnt = 0
        valid = True
        over = False
        while True:
            ok, val, rel = fifo_pop_ready(fv, fr, fm, n, now)
            if not ok:
                break
            flags = val >> _FLAG_SHIFT
            code = val & _CODE_MASK
            if flags & SAMPLE_INVALID:
                valid = False
            if flags & SAMPLE_DROPPED:      # a lost sample: no code and no clip flag reach the controller
                continue
            if flags & SAMPLE_OVERRANGE:
                over = True
            if cnt == m_max:
                for m in range(m_max - 1):
                    out_samples[n, m] = out_samples[n, m + 1]
                    out_ticks[n, m] = out_ticks[n, m + 1]
                cnt = m_max - 1
                lost += 1
            out_samples[n, cnt] = code
            out_ticks[n, cnt] = rel - lat
            cnt += 1
        for m in range(cnt, m_max):
            out_samples[n, m] = 0
            out_ticks[n, m] = 0
        out_count[n] = cnt
        if cnt > 0:
            latest[n] = out_samples[n, cnt - 1]
        out_valid[n] = valid
        out_over[n] = over
        out_new[n] = cnt > 0
    return lost


@njit(cache=True)
def monitor_release(now, lat, fv, fr, fm, out_samples, out_ticks, out_count, latest, out_valid, out_over, out_new):
    """Batched :func:`monitor_release_row` (lat (B,)). Returns the total number of samples lost to the M limit."""
    lost = 0
    for b in range(fm.shape[0]):
        lost += monitor_release_row(now, lat[b], fv[b], fr[b], fm[b], out_samples[b], out_ticks[b], out_count[b],
                                    latest[b], out_valid[b], out_over[b], out_new[b])
    return lost


@njit(cache=True)
def monitor_level_codes_row(p1_w, p0_w, p_bg_w, t_pd_k, t_adc_k, pga_sel, pr, pw, pga_gains, shot_on, tau_s,
                            out_mu1, out_mu0, out_sig1, out_sig0, s_tia_a2_per_hz=-1.0, rin_per_hz=0.0):
    """Ideal monitor codes of the two NRZ levels and their noise for a bit-rate sampler (07 P0-11 bit statistics).

    mu_x = the ADC transfer of i_x = gain_scale R (P_x + f_bg P_bus) + i_d (without quantization, INL or DNL);
    sigma_x^2 = (2 q (i_sig,x + i_d) [shot_on] + S_tia) / (4 tau_s) in A^2: the exact variance of an instantaneous
    sample of white noise through a single-pole sampler front end of time constant tau_s (= 1/(2 pi f_3dB); 1/(4 tau)
    is the noise bandwidth of one pole), converted to codes with the same gain as mu. tau_s is a property of the
    sampler hardware (datapath.bitstat_bw_hz), not of the plant step. Flicker is not included (it is far slower than
    one bit-statistics window). S_tia is the monitor's input noise density pw[PW_S_TIA] unless s_tia_a2_per_hz >= 0
    is given: a bit-rate front end is a different, wider-band amplifier (its noise density is assumed, no data).
    rin_per_hz (one-sided laser RIN, 1/Hz; default 0) adds the RIN of the line light at the sampler bandwidth,
    (gain_scale R P_x)^2 RIN / (4 tau_s): the per-tick RIN of mrm/plant/laser.py only moves the levels at the tick
    rate, while a bit-rate sampler sees the whole RIN band up to its own bandwidth (background light excluded).
    """
    n_rings = p1_w.shape[0]
    n_pga = pga_gains.shape[0]
    s_tia = pw[PW_S_TIA] if s_tia_a2_per_hz < 0.0 else s_tia_a2_per_hz
    for n in range(n_rings):
        g = 1.0
        if n_pga > 0:
            sel = pga_sel[n]
            if sel < 0:
                sel = 0
            elif sel > n_pga - 1:
                sel = n_pga - 1
            g = pga_gains[sel]
        ref = 1.0 + pw[PW_TEMPCO] * (t_adc_k[n] - pw[PW_T_REF])
        k_code = g / pw[PW_LSB] * (1.0 + pr[n, PR_GAIN_ERR]) / ref
        i_d = dark_current_a(pw[PW_DARK_I0], t_pd_k[n], pw[PW_DARK_T0], pw[PW_DARK_EA])
        for lvl in range(2):
            p = p1_w[n] if lvl == 0 else p0_w[n]
            i_sig = pw[PW_GAIN_SCALE] * pr[n, PR_RESP] * (p + pr[n, PR_BG] * p_bg_w[n])
            mu = (i_sig + i_d - pr[n, PR_IBASE]) * k_code + pr[n, PR_OFFSET]
            s = s_tia + (2.0 * Q_E * (abs(i_sig) + i_d) if shot_on else 0.0)
            if rin_per_hz > 0.0:
                i_line = pw[PW_GAIN_SCALE] * pr[n, PR_RESP] * p
                s += i_line * i_line * rin_per_hz
            sig = math.sqrt(s / (4.0 * tau_s)) * abs(k_code) if tau_s > 0.0 else 0.0
            if lvl == 0:
                out_mu1[n] = mu
                out_sig1[n] = sig
            else:
                out_mu0[n] = mu
                out_sig0[n] = sig


# ===================================================================================== temperature sensor kernels
_TP_NOISE = 0      # white noise per sample, K rms (temp_sensors[].noise_rms_k)
_TP_RES = 1        # quantization step, K (resolution_k)
_TP_OFFSET = 2     # offset, K (drawn per device)
_TP_GAIN_ERR = 3   # gain error, fraction (drawn per device)
_TP_SELF = 4       # self-heating, K
_TP_ZERO_K = 5     # temperature read by code 0, K
_TP_GAIN_REF_K = 6  # pivot of the gain error, K
NTP = 7


@njit(cache=True)
def temp_tick_row(k, t_site_k, tp, period, lat, code_max, fl_coef, key, st, fx, fc, primed, fv, fr, fm,
                  out_sampled):
    """Integrate interval [k, k+1) for the S temperature sensors of one world; take, convert and queue the samples
    whose sample tick (a multiple of the sensor's period) is k + 1 (07 P1-10).

    t_site_k (S,): true temperature at the sensor site (K; ROM temp_sensor observer). A conversion averages the
    temperature over its whole sample period (boxcar; assumed for an integrating / sigma-delta sensor converter), then
        T_meas = T_gref + (1 + gain_err)(T_avg - T_gref) + offset + self_heating + noise + flicker,
        code = round((T_meas - T_zero) / resolution), clipped to 0 .. code_max[s].
    noise: white, temp_sensors[].noise_rms_k per sample (one normal per sample: exact, dt-invariant); flicker: OU set
    (fl_coef (S, J, 10)). tp (S, NTP); period (S,) and lat (S,) in ticks; code_max (S,).
    State st (S, 2) [sum of T over the window, ticks], fx/fc (S, J), primed (S,), FIFO (S, C).
    """
    n_s = t_site_k.shape[0]
    for i in range(n_s):
        out_sampled[i] = 0
        if primed[i] == 0:
            st[i, 0] = 0.0
            st[i, 1] = 0.0
            if fl_coef.shape[1] > 0:
                _flicker_init(key, i, k, _SLOT_FL_INIT, fl_coef[i], fx[i], fc[i])
            primed[i] = 1
        st[i, 0] += t_site_k[i]
        st[i, 1] += 1.0
        if (k + 1) % period[i] != 0:
            continue
        s = k + 1
        t_avg = st[i, 0] / st[i, 1]
        st[i, 0] = 0.0
        st[i, 1] = 0.0
        noise = tp[i, _TP_NOISE] * _rng.normal(key, i, s, _SLOT_WHITE)
        if fl_coef.shape[1] > 0:
            noise += flicker_step(key, i, s, _SLOT_FLICKER, fl_coef[i], fx[i], fc[i])
        gref = tp[i, _TP_GAIN_REF_K]
        t_meas = gref + (1.0 + tp[i, _TP_GAIN_ERR]) * (t_avg - gref) + tp[i, _TP_OFFSET] + tp[i, _TP_SELF] + noise
        c = _round_clip((t_meas - tp[i, _TP_ZERO_K]) / tp[i, _TP_RES], code_max[i])
        ok = fifo_push(fv, fr, fm, i, c, s + lat[i])
        out_sampled[i] = 1 if ok else 2


@njit(cache=True)
def temp_tick(k, t_site_k, tp, period, lat, code_max, fl_coef, key, st, fx, fc, primed, fv, fr, fm, out_sampled):
    """Batched :func:`temp_tick_row`: t_site_k (B, S), tp (B, S, NTP), lat (B, S), fl_coef (B, S, J, 10), key (B,)."""
    for b in range(t_site_k.shape[0]):
        temp_tick_row(k, t_site_k[b], tp[b], period, lat[b], code_max, fl_coef[b], key[b], st[b], fx[b], fc[b],
                      primed[b], fv[b], fr[b], fm[b], out_sampled[b])


@njit(cache=True)
def temp_release_row(now, fv, fr, fm, latest, out_new):
    """Release the temperature samples whose release tick is <= now; latest (S,) is held, out_new (S,) marks a
    delivery since the last call (07 P0-7 'latest' channel contract)."""
    for i in range(fm.shape[0]):
        out_new[i] = False
        while True:
            ok, val, rel = fifo_pop_ready(fv, fr, fm, i, now)
            if not ok:
                break
            latest[i] = val
            out_new[i] = True


@njit(cache=True)
def temp_release(now, fv, fr, fm, latest, out_new):
    """Batched :func:`temp_release_row`."""
    for b in range(fm.shape[0]):
        temp_release_row(now, fv[b], fr[b], fm[b], latest[b], out_new[b])


# ===================================================================================== heater readback kernels
@njit(cache=True)
def readback_tick_row(k, v_v, i_a, lsb_v, lsb_i, noise_codes, period, lat, code_max, average, key, st, fv, fr, fm,
                      out_sampled):
    """Heater V/I readback of one world (07 P1-10 ring.heater.readback): interval [k, k+1) and the samples at tick
    k + 1 when it is a multiple of ``period``.

    v_v, i_a (N,): heater voltage and current of the interval (V, A; HeaterTickOut.v_v / i_a, rms for dac_pwm).
    average = 0: track-and-hold of the last interval (assumed default); 1: boxcar mean over the readback period.
    code = round(x / lsb + noise_codes z), clipped to 0 .. code_max; lsb = full scale / code_max. The two codes are
    packed into one FIFO value (v | i << 32). State st (N, 3) [sum v, sum i, ticks].
    """
    n_rings = v_v.shape[0]
    for n in range(n_rings):
        out_sampled[n] = 0
        if average:
            st[n, 0] += v_v[n]
            st[n, 1] += i_a[n]
            st[n, 2] += 1.0
        if (k + 1) % period != 0:
            continue
        s = k + 1
        if average and st[n, 2] > 0.0:
            vv = st[n, 0] / st[n, 2]
            ii = st[n, 1] / st[n, 2]
        else:
            vv = v_v[n]
            ii = i_a[n]
        st[n, 0] = 0.0
        st[n, 1] = 0.0
        st[n, 2] = 0.0
        zv = 0.0
        zi = 0.0
        if noise_codes > 0.0:
            zv, zi = _rng.normal2(key, n, s, _SLOT_WHITE)
        cv = _round_clip(vv / lsb_v + noise_codes * zv, code_max)
        ci = _round_clip(ii / lsb_i + noise_codes * zi, code_max)
        ok = fifo_push(fv, fr, fm, n, cv | (ci << 32), s + lat)
        out_sampled[n] = 1 if ok else 2


@njit(cache=True)
def readback_tick(k, v_v, i_a, lsb_v, lsb_i, noise_codes, period, lat, code_max, average, key, st, fv, fr, fm,
                  out_sampled):
    """Batched :func:`readback_tick_row`: (B, N) inputs; lsb_v, lsb_i, noise_codes, lat, key (B,)."""
    for b in range(v_v.shape[0]):
        readback_tick_row(k, v_v[b], i_a[b], lsb_v[b], lsb_i[b], noise_codes[b], period, lat[b], code_max, average,
                          key[b], st[b], fv[b], fr[b], fm[b], out_sampled[b])


@njit(cache=True)
def readback_release_row(now, fv, fr, fm, latest_v, latest_i, out_new):
    """Release ready readback samples; latest_v / latest_i (N,) are held, out_new (N,) marks a delivery."""
    mask = (np.int64(1) << 32) - 1
    for n in range(fm.shape[0]):
        out_new[n] = False
        while True:
            ok, val, rel = fifo_pop_ready(fv, fr, fm, n, now)
            if not ok:
                break
            latest_v[n] = val & mask
            latest_i[n] = val >> 32
            out_new[n] = True


@njit(cache=True)
def readback_release(now, fv, fr, fm, latest_v, latest_i, out_new):
    """Batched :func:`readback_release_row`."""
    for b in range(fm.shape[0]):
        readback_release_row(now, fv[b], fr[b], fm[b], latest_v[b], latest_i[b], out_new[b])


# ===================================================================================== DAC glitch fault
@njit(cache=True)
def dac_glitch_row(codes, fault, glitch_codes, lo, hi, out):
    """07 P1-10 ``dac_glitch``: where fault & FAULT_DAC_GLITCH, out = clip(code + glitch_codes, lo, hi) for that tick;
    elsewhere out = code. Apply it to the code that LEAVES the DAC latency FIFO (an output glitch), never to the
    latched register (the register readback must not show it). codes, fault, glitch_codes (N,) (glitch may be
    (1,)); lo, hi the DAC code range."""
    n_g = glitch_codes.shape[0]
    for n in range(codes.shape[0]):
        c = codes[n]
        if fault[n] & FAULT_DAC_GLITCH:
            c = c + glitch_codes[n if n_g > 1 else 0]
            if c < lo:
                c = lo
            elif c > hi:
                c = hi
        out[n] = c


@njit(cache=True)
def dac_glitch(codes, fault, glitch_codes, lo, hi, out):
    """Batched :func:`dac_glitch_row` ((B, N) codes and fault; glitch_codes (B|1, N|1))."""
    n_b = glitch_codes.shape[0]
    for b in range(codes.shape[0]):
        dac_glitch_row(codes[b], fault[b], glitch_codes[b if n_b > 1 else 0], lo, hi, out[b])


@njit(cache=True)
def fault_mask_row(adc_drop, adc_stuck, adc_saturate, dac_glitch_codes, out):
    """FAULT_* mask (N,) int8 from the scenario fault signals of one world (mrm.scenarios.generators: fault.adc_drop,
    fault.adc_stuck, fault.adc_saturate are flags, on at >= 0.5; fault.dac_glitch_codes is an additive code, on when
    non-zero). Inputs (N,) float64 (07 P1-10 scenario-bound channel masks)."""
    for n in range(out.shape[0]):
        m = 0
        if adc_drop[n] >= 0.5:
            m |= FAULT_ADC_DROP
        if adc_stuck[n] >= 0.5:
            m |= FAULT_ADC_STUCK
        if adc_saturate[n] >= 0.5:
            m |= FAULT_ADC_SATURATE
        if dac_glitch_codes[n] != 0.0:
            m |= FAULT_DAC_GLITCH
        out[n] = m


def fault_mask(B: int, N: int, *, adc_drop=0.0, adc_stuck=0.0, adc_saturate=0.0, dac_glitch_codes=0.0
               ) -> np.ndarray:
    """(B, N) int8 FAULT_* mask from scenario fault signals (scalars, (N,), (B,) or (B, N); see
    :func:`fault_mask_row`). Use it to turn the float scenario channels into the ``fault`` argument of
    :meth:`MonitorBank.tick` and :func:`dac_glitch_row`."""
    a = [_arr_bn(x, B, N, nm) for x, nm in ((adc_drop, "adc_drop"), (adc_stuck, "adc_stuck"),
                                            (adc_saturate, "adc_saturate"), (dac_glitch_codes, "dac_glitch_codes"))]
    out = np.zeros((B, N), np.int8)
    for b in range(B):
        fault_mask_row(a[0][b], a[1][b], a[2][b], a[3][b], out[b])
    return out


# ============================================================================== flicker (Python, build time)
def flicker_pole_set(s_white: float, f_corner_hz: float, f_min_hz: float = _FLICKER_F_MIN_HZ,
                     f_max_hz: float | None = None, per_decade: int = _FLICKER_PER_DECADE
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Rates (rad/s) and variances of J OU processes whose summed PSD approximates S_f(f) = s_white f_c / f.

    One OU process with rate lambda and variance sigma^2 has the one-sided PSD 4 sigma^2 lambda/(lambda^2 + w^2).
    Corners f_j = f_min r^(j + 1/2), r = 10^(1/per_decade), j = 0..J-1, J = ceil(per_decade log10(f_max/f_min)), and
    equal variances sigma^2 = s_white f_c ln(r) give sum_j ~ sigma^2/(f ln r) = s_white f_c / f (a log-uniform
    mixture of Lorentzians). The PSD flattens below f_min and falls as 1/f^2 above f_max (band limits).
    s_white: one-sided white PSD in the channel's unit^2/Hz; f_corner_hz: frequency where flicker equals white.
    Returns empty arrays when s_white or f_corner is not positive.
    """
    if not (s_white > 0.0 and f_corner_hz > 0.0):
        return np.zeros(0), np.zeros(0)
    if f_max_hz is None:
        f_max_hz = 10.0 * f_corner_hz
    if not (0.0 < f_min_hz < f_max_hz):
        raise ValueError(f"flicker band needs 0 < f_min ({f_min_hz}) < f_max ({f_max_hz})")
    if int(per_decade) < 1:
        raise ValueError("per_decade must be >= 1")
    r = 10.0 ** (1.0 / per_decade)
    n_poles = max(1, int(math.ceil(per_decade * math.log10(f_max_hz / f_min_hz) - 1e-9)))
    f = f_min_hz * r ** (np.arange(n_poles) + 0.5)
    lam = 2.0 * math.pi * f
    var = np.full(n_poles, s_white * f_corner_hz * math.log(r))
    return lam, var


def flicker_psd(f_hz, lam, var) -> np.ndarray:
    """One-sided PSD of the OU sum at f_hz: sum_j 4 var_j lam_j / (lam_j^2 + (2 pi f)^2)."""
    w = 2.0 * math.pi * np.atleast_1d(np.asarray(f_hz, dtype=float))[:, None]
    lam = np.asarray(lam, dtype=float)[None, :]
    var = np.asarray(var, dtype=float)[None, :]
    return np.sum(4.0 * var * lam / (lam * lam + w * w), axis=1)


def _g_window(u: float) -> float:
    """g(u) = u - 2(1 - e^-u) + (1 - e^-2u)/2 (variance kernel of the integral of an OU process); series below 1e-2."""
    if u < 1e-2:
        return u ** 3 / 3.0 - u ** 4 / 4.0 + 7.0 * u ** 5 / 60.0 - u ** 6 / 24.0
    return u + 2.0 * math.expm1(-u) - 0.5 * math.expm1(-2.0 * u)


def _chol2(vx: float, cxc: float, vc: float) -> tuple[float, float, float]:
    l00 = math.sqrt(max(vx, 0.0))
    l10 = cxc / l00 if l00 > 0.0 else 0.0
    l11 = math.sqrt(max(vc - l10 * l10, 0.0))
    return l00, l10, l11


def ou_window_coefficients(lam, var, period_s: float, window_s: float) -> np.ndarray:
    """Exact sample-to-sample coefficients (J, 10) of OU processes seen through an integrate-and-dump window.

    Between samples (period P) the process runs freely for D = P - W, then the window (W) averages it. Given x at the
    previous sample instant x0 (dx = -lam x dt + sqrt(2 lam) sigma dW):
        x(s - W) = a_D x0 + eta,  Var eta = sigma^2 (1 - a_D^2),  a_D = e^{-lam D}
        x(s)     = a_W x(s-W) + e1,  Var e1 = sigma^2 (1 - a_W^2)
        Y        = integral over W = x(s-W)(1 - a_W)/lam + e2,  Var e2 = 2 sigma^2 g(lam W)/lam^2,
        Cov(e1, e2) = sigma^2 (1 - a_W)^2 / lam,
    and the contribution to the sample is the window mean Y/W. Rows: [P00, P01, P10, P11, L00, L10, L11, S00, S10,
    S11] (see :func:`flicker_step`); the window mean is not carried (P01 = P11 = 0); S = stationary start (x only).
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    var = np.atleast_1d(np.asarray(var, dtype=float))
    if window_s <= 0.0 or period_s < window_s:
        raise ValueError(f"need 0 < window ({window_s}) <= period ({period_s})")
    out = np.zeros((lam.size, 10))
    d = period_s - window_s
    for j, (lj, vj) in enumerate(zip(lam, var)):
        a_d = math.exp(-lj * d)
        v_eta = vj * (-math.expm1(-2.0 * lj * d))
        u = lj * window_s
        a_w = math.exp(-u)
        oma = -math.expm1(-u)
        v1 = vj * (-math.expm1(-2.0 * u))
        c12a = vj * oma * oma / lj / window_s
        v2a = 2.0 * vj * _g_window(u) / (u * u)
        m_a = oma / u
        vx = a_w * a_w * v_eta + v1
        va = m_a * m_a * v_eta + v2a
        cxa = a_w * m_a * v_eta + c12a
        l00, l10, l11 = _chol2(vx, cxa, va)
        out[j] = [a_w * a_d, 0.0, m_a * a_d, 0.0, l00, l10, l11, math.sqrt(vj), 0.0, 0.0]
    return out


def ou_rc_coefficients(lam, var, period_s: float, tau_s: float) -> np.ndarray:
    """Exact sample-to-sample coefficients (J, 10) of OU processes seen through a first-order RC front end.

    State (x, y): dx = -lam x dt + sqrt(2 lam) sigma dW, dy = mu (x - y) dt, mu = 1/tau. Over the sample period P:
        Phi = [[a, 0], [kappa (a - b), b]],  a = e^{-lam P}, b = e^{-mu P}, kappa = mu/(mu - lam);
        Q_xx = sigma^2 (1 - a^2),  Q_xy = 2 lam sigma^2 kappa [E(2 lam) - E(lam + mu)],
        Q_yy = 2 lam sigma^2 kappa^2 [E(2 lam) - 2 E(lam + mu) + E(2 mu)],  E(alpha) = (1 - e^{-alpha P})/alpha.
    Stationary covariance: Var x = sigma^2, Cov = Var y = sigma^2 mu/(lam + mu). A pole within _RC_NUDGE of mu is
    moved by that fraction (the closed form is singular there). Rows as in :func:`ou_window_coefficients`.
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    var = np.atleast_1d(np.asarray(var, dtype=float))
    if not (tau_s > 0.0 and period_s > 0.0):
        raise ValueError("need tau_s > 0 and period_s > 0")
    mu = 1.0 / tau_s
    out = np.zeros((lam.size, 10))

    def e(alpha):
        return -math.expm1(-alpha * period_s) / alpha

    for j, (lj, vj) in enumerate(zip(lam, var)):
        if abs(mu - lj) < _RC_NUDGE * mu:
            lj = mu * (1.0 - _RC_NUDGE)
        a = math.exp(-lj * period_s)
        b = math.exp(-mu * period_s)
        kap = mu / (mu - lj)
        b2 = 2.0 * lj * vj
        qxx = vj * (-math.expm1(-2.0 * lj * period_s))
        qxy = b2 * kap * (e(2.0 * lj) - e(lj + mu))
        qyy = b2 * kap * kap * (e(2.0 * lj) - 2.0 * e(lj + mu) + e(2.0 * mu))
        l00, l10, l11 = _chol2(qxx, qxy, qyy)
        sxy = vj * mu / (lj + mu)
        s00, s10, s11 = _chol2(vj, sxy, sxy)
        out[j] = [a, 0.0, kap * (a - b), b, l00, l10, l11, s00, s10, s11]
    return out


def flicker_coefficients(s_white: float, f_corner_hz: float, *, afe_type: int, period_s: float,
                         window_s: float = 0.0, tau_s: float = 0.0, f_min_hz: float = _FLICKER_F_MIN_HZ,
                         f_max_hz: float | None = None, per_decade: int = _FLICKER_PER_DECADE) -> np.ndarray:
    """(J, 10) flicker coefficients of one channel (zero rows when flicker is off).

    Default f_max = 10 min(f_corner, f_afe), f_afe = 1/(2 W) (integrate_dump) or 1/(2 pi tau) (rc): above f_corner
    the white floor dominates, and above f_afe the front end removes the flicker anyway.
    """
    if not (s_white > 0.0 and f_corner_hz > 0.0):
        return np.zeros((0, 10))
    f_afe = 1.0 / (2.0 * window_s) if afe_type == AFE_INTEGRATE_DUMP else 1.0 / (2.0 * math.pi * tau_s)
    if f_max_hz is None:
        f_max_hz = 10.0 * min(f_corner_hz, f_afe)
        if f_max_hz <= f_min_hz:
            # a corner this low puts the whole 1/f band below the lowest pole, where it acts as a static offset
            return np.zeros((0, 10))
    lam, var = flicker_pole_set(s_white, f_corner_hz, f_min_hz, f_max_hz, per_decade)
    if lam.size >= _SLOT_FL_INIT - _SLOT_FLICKER:
        raise ValueError(f"{lam.size} flicker poles exceed the counter-RNG slot range ({_SLOT_FL_INIT - _SLOT_FLICKER})")
    if afe_type == AFE_INTEGRATE_DUMP:
        return ou_window_coefficients(lam, var, period_s, window_s)
    return ou_rc_coefficients(lam, var, period_s, tau_s)


def _stack_coefs(coefs: list[np.ndarray]) -> np.ndarray:
    """Stack per-world (J_b, 10) coefficient arrays into (B, J, 10), zero-padding (a zero row contributes 0)."""
    jmax = max((c.shape[0] for c in coefs), default=0)
    out = np.zeros((len(coefs), jmax, 10))
    for b, c in enumerate(coefs):
        out[b, :c.shape[0]] = c
    return out


# ================================================================================================ Python API
def _arr_bn(x, B, N, name, dtype=np.float64) -> np.ndarray:
    """Broadcast a scalar, (N,), (B,), (B, 1), (1, N) or (B, N) value to a C-contiguous (B, N) array.

    A 1-D array of length N is per ring; of length B (!= N) per world (the convention of mrm/plant/electrical.py)."""
    a = np.asarray(x, dtype=dtype)
    if a.ndim == 1:
        a = a.reshape(1, -1) if a.shape[0] == N else a.reshape(-1, 1)
    try:
        return np.ascontiguousarray(np.broadcast_to(a, (B, N))).copy()
    except ValueError:
        raise ValueError(f"{name}: shape {np.shape(x)} does not broadcast to (B={B}, N={N})") from None


def _arr_b(x, B, name, dtype=np.float64) -> np.ndarray:
    a = np.asarray(x, dtype=dtype).reshape(-1)
    if a.size == 1:
        return np.full(B, a[0], dtype=dtype)
    if a.size != B:
        raise ValueError(f"{name}: shape {np.shape(x)} is not scalar or (B={B},)")
    return a.copy()


def _keys(seeds, B, stream_name) -> np.ndarray:
    """Philox keys (B,) uint64 for per-element seeds (B,) and a stream name (mrm.core.rng.stream_id)."""
    seeds = _arr_b(seeds, B, "seeds", np.int64)
    sid = _rng.stream_id(stream_name)
    return np.array([_rng.stream_key(int(s), sid) for s in seeds], dtype=np.uint64)


def adc_channel_dtype(bits: int) -> np.dtype:
    """dtype of an unsigned ADC-code sensor channel of ``bits`` (int32 up to 31 bits, else int64; mrm/core/channels)."""
    return np.dtype(np.int32) if int(bits) <= 31 else np.dtype(np.int64)


def integrator_full_scale_a(t_int_s: float = PAT2_T_INT_S, c_int_f: float = PAT2_C_INT_F,
                            v_swing_v: float = PAT2_V_REF_V) -> float:
    """Largest DC input current of a dual-slope integrator before it saturates: I = C V / T_int (A).

    With the PAT-2 example values (16 pF, 0.5 V, 2 us) this is 4 uA, about 4 times below the 15.5 uA G26 monitor
    current (read-off, 01). A preset with a 2 us window must therefore assume a current divider (mirror) ahead of the
    integrator, a larger capacitor or a shorter window; see docs/handoff/plant_sensing.md.
    """
    return c_int_f * v_swing_v / t_int_s


def draw_inl_bow_lsb(device_seeds, N: int, inl_lsb, stream_name: str = "adc_monitor_inl") -> np.ndarray:
    """Per-device INL bow amplitudes (B, N) uniform in [-inl, +inl] LSB from the DEVICE seed (07 P1-10, P1-13).

    The bow peaks at mid-scale with |INL| = |bow| (adc_quantize). inl_lsb: scalar or (B,) spec value.
    """
    seeds = np.atleast_1d(np.asarray(device_seeds, dtype=np.int64))
    B = seeds.size
    inl = _arr_b(inl_lsb, B, "inl_lsb")
    keys = _keys(seeds, B, stream_name)
    out = np.zeros((B, N))
    for b in range(B):
        for n in range(N):
            out[b, n] = inl[b] * (2.0 * _rng.uniform(keys[b], n, 0, _SLOT_INL) - 1.0)
    return out


@dataclass
class MonitorHw:
    """Realized monitor-channel hardware for B worlds and N rings, SI (07 P1-9, P1-10). Build with ``build`` or
    ``from_plant_params``; never sample inside (realize owns the uncertainty draws, except the INL bow and DNL
    thresholds, which are device-keyed functions of the device seed)."""

    B: int
    N: int
    dt_s: float
    afe_type: int
    window_ticks: int
    period_ticks: int
    adc_bits: int
    code_max: int
    shot_on: int
    stuck_flagged: int
    phase: np.ndarray            # (N,) int64 sample phases in ticks (adc_mux)
    latency_ticks: np.ndarray    # (B,) int64
    pga_gains: np.ndarray        # (G,) float64 (empty = gain 1)
    pr: np.ndarray               # (B, N, NPR)
    pw: np.ndarray               # (B, NPW)
    fl_coef: np.ndarray          # (B, J, 10)
    slot_base: int = 0
    stream: str = "adc_monitor"  # noise stream of the TIA/flicker draws; DNL/INL use stream + '_dnl' / '_inl'
    afe_tau_s: float = 0.0
    device_seeds: np.ndarray | None = None  # (B,) int64 device seeds (DNL thresholds); set by from_plant_params

    @property
    def lsb_a(self) -> np.ndarray:
        """ADC LSB at PGA gain 1, A per code, (B,)."""
        return self.pw[:, PW_LSB].copy()

    @classmethod
    def build(cls, B: int, N: int, dt_s: float, *, responsivity_a_per_w, adc_full_scale_a, gain_scale=1.0,
              dark_i0_a=0.0, dark_t0_k=_DEFAULT_T0_K, dark_activation_ev=_DEFAULT_DARK_EA_EV,
              background_fraction=0.0, baseline_a=0.0, afe_type="integrate_dump",
              integration_window_s: float | None = PAT2_T_INT_S, afe_tau_s: float | None = None,
              adc_rate_hz: float = 250e3, adc_latency_s=2e-6, adc_bits: int = 16, adc_offset_codes=0.0,
              adc_gain_error_frac=0.0, adc_inl_bow_lsb=0.0, adc_dnl_lsb=0.0, input_noise_a_per_rt_hz=0.0,
              flicker_corner_hz=0.0, flicker_f_min_hz: float = _FLICKER_F_MIN_HZ,
              flicker_f_max_hz: float | None = None, flicker_per_decade: int = _FLICKER_PER_DECADE,
              adc_ref_tempco_per_k=0.0, adc_ref_t_k=ADC_REF_T_K_DEFAULT, pga_gains=(), channels_per_adc: int = 1,
              mux_order: str = "round_robin", shot_noise: bool = True, stuck_flagged: bool = True,
              slot_base: int = 0, stream: str = "adc_monitor") -> "MonitorHw":
        """Build from SI values. Arrays broadcast: per ring (B, N) for responsivity, background_fraction, baseline_a,
        adc_offset_codes, adc_gain_error_frac, adc_inl_bow_lsb; per world (B,) for the rest.

        Defaults are fallbacks for tests; presets own the real values. Their provenance: integration window 2 us
        (printed example, PAT-2), adc_rate_hz 250 kHz (inferred: a dual-slope conversion is the 2 us integration plus
        up to 2 us of de-integration, mrm.core.channels.ChannelHardware), adc_latency_s 2 us (assumed: one
        de-integration), adc_bits 16 (inferred from the RX polarization-monitor codes, 01 D17; disputed, see
        docs/handoff/core_control_api.md), dark current off, E_a 0.37 eV (inferred, 02 section 10), T0 25 C (printed,
        GF dark-current spec temperature), every converter error 0 and no PGA (neutral).

        afe_type: 'integrate_dump' (default, 07 P0-7; window integration_window_s, whole ticks, <= the ADC period) or
        'rc' (afe_tau_s). adc_rate_hz is the rate PER RING (docs/handoff/core_control_api.md item 7); the ADC period
        must be whole ticks. adc_mux (07 P0-7): with channels_per_adc = C, 'round_robin' staggers ring n by
        floor((n mod C) P / C) ticks (one converter serves C rings in turn); 'bank' samples every ring at phase 0
        (simultaneous sample-and-hold; assumed meaning). adc_full_scale_a: input current at code 2^bits - 1 (PGA 1).
        dark_activation_ev: E_a in eV (07 D20: 0.35-0.38). adc_dnl_lsb: peak |DNL| (clipped to 1).
        input_noise_a_per_rt_hz: one-sided input-referred white current noise density; flicker_corner_hz its 1/f
        corner. shot_noise = fidelity.shot_noise. stuck_flagged: a stuck converter marks its samples invalid
        (monitor_valid False, as the channel doc says); False models a silent stuck fault.
        """
        dt = float(dt_s)
        afe = AFE_TYPES[afe_type] if isinstance(afe_type, str) else int(afe_type)
        if afe not in (AFE_RC, AFE_INTEGRATE_DUMP):
            raise ValueError(f"afe_type must be one of {tuple(AFE_TYPES)}")
        period = rate_to_period_ticks(adc_rate_hz, dt, "ring.monitor.adc_rate_hz")
        if afe == AFE_INTEGRATE_DUMP:
            if integration_window_s is None:
                raise ValueError("afe_type integrate_dump needs ring.monitor.integration_window_s")
            window = seconds_to_ticks(integration_window_s, dt, "ring.monitor.integration_window_s",
                                      allow_zero=False)
            if window > period:
                raise ValueError(f"integration window ({window} ticks) exceeds the ADC period ({period} ticks): one "
                                 "integrate-and-dump channel cannot overlap its windows")
            tau = 0.0
            a_tick, mu = 1.0, 0.0
        else:
            if afe_tau_s is None or not afe_tau_s > 0.0:
                raise ValueError("afe_type rc needs ring.monitor.afe_tau_s > 0")
            tau = float(afe_tau_s)
            window = 0
            a_tick, mu = math.exp(-dt / tau), 1.0 / tau
        lat = _arr_b(adc_latency_s, B, "adc_latency_s")
        lat_ticks = np.array([seconds_to_ticks(float(x), dt, "ring.monitor.adc_latency_s") for x in lat],
                             dtype=np.int64)
        bits = int(adc_bits)
        if not 1 <= bits <= 32:
            raise ValueError("adc_bits must be in 1..32")
        code_max = (1 << bits) - 1
        fs = _arr_b(adc_full_scale_a, B, "adc_full_scale_a")
        if not np.all(np.isfinite(fs) & (fs > 0)):
            raise ValueError("adc_full_scale_a must be finite and > 0")
        dnl = _arr_b(adc_dnl_lsb, B, "adc_dnl_lsb")
        if np.any(dnl < 0):
            raise ValueError("adc_dnl_lsb must be >= 0")
        if np.any(dnl > 1.0):
            warnings.warn("adc_dnl_lsb > 1 LSB clipped to 1 (thresholds must stay monotone)", stacklevel=2)
            dnl = np.minimum(dnl, 1.0)
        cpa = int(channels_per_adc)
        if cpa < 1:
            raise ValueError("channels_per_adc must be >= 1")
        if mux_order == "round_robin":
            phase = np.array([((n % cpa) * period) // cpa for n in range(N)], dtype=np.int64)
        elif mux_order == "bank":
            phase = np.zeros(N, dtype=np.int64)
        else:
            raise ValueError("mux_order must be 'round_robin' or 'bank'")
        pr = np.zeros((B, N, NPR))
        pr[:, :, PR_RESP] = _arr_bn(responsivity_a_per_w, B, N, "responsivity_a_per_w")
        pr[:, :, PR_BG] = _arr_bn(background_fraction, B, N, "background_fraction")
        pr[:, :, PR_IBASE] = _arr_bn(baseline_a, B, N, "baseline_a")
        pr[:, :, PR_GAIN_ERR] = _arr_bn(adc_gain_error_frac, B, N, "adc_gain_error_frac")
        pr[:, :, PR_OFFSET] = _arr_bn(adc_offset_codes, B, N, "adc_offset_codes")
        pr[:, :, PR_INL_BOW] = _arr_bn(adc_inl_bow_lsb, B, N, "adc_inl_bow_lsb")
        noise = _arr_b(input_noise_a_per_rt_hz, B, "input_noise_a_per_rt_hz")
        pw = np.zeros((B, NPW))
        pw[:, PW_GAIN_SCALE] = _arr_b(gain_scale, B, "gain_scale")
        pw[:, PW_DARK_I0] = _arr_b(dark_i0_a, B, "dark_i0_a")
        pw[:, PW_DARK_T0] = _arr_b(dark_t0_k, B, "dark_t0_k")
        pw[:, PW_DARK_EA] = _arr_b(dark_activation_ev, B, "dark_activation_ev") * Q_E
        pw[:, PW_S_TIA] = noise ** 2
        pw[:, PW_LSB] = fs / code_max
        pw[:, PW_TEMPCO] = _arr_b(adc_ref_tempco_per_k, B, "adc_ref_tempco_per_k")
        pw[:, PW_T_REF] = _arr_b(adc_ref_t_k, B, "adc_ref_t_k")
        pw[:, PW_DNL] = dnl
        pw[:, PW_A_TICK] = a_tick
        pw[:, PW_MU] = mu
        pw[:, PW_DT] = dt
        fc_hz = _arr_b(flicker_corner_hz, B, "flicker_corner_hz")
        coefs = [flicker_coefficients(float(noise[b] ** 2), float(fc_hz[b]), afe_type=afe,
                                      period_s=period * dt, window_s=window * dt, tau_s=tau,
                                      f_min_hz=flicker_f_min_hz, f_max_hz=flicker_f_max_hz,
                                      per_decade=flicker_per_decade) for b in range(B)]
        gains = np.asarray(pga_gains, dtype=np.float64).reshape(-1)
        if gains.size and not np.all(np.isfinite(gains) & (gains > 0)):
            raise ValueError("pga_gains must be finite and > 0")
        return cls(B=int(B), N=int(N), dt_s=dt, afe_type=afe, window_ticks=int(window), period_ticks=int(period),
                   adc_bits=bits, code_max=code_max, shot_on=int(bool(shot_noise)),
                   stuck_flagged=int(bool(stuck_flagged)), phase=phase, latency_ticks=lat_ticks, pga_gains=gains,
                   pr=pr, pw=pw, fl_coef=_stack_coefs(coefs), slot_base=int(slot_base), stream=str(stream),
                   afe_tau_s=tau)

    @classmethod
    def from_plant_params(cls, pp, *, baseline_a=None, mux_order: str | None = None, stuck_flagged: bool | None = None,
                          flicker_f_min_hz: float | None = None, flicker_f_max_hz: float | None = None,
                          flicker_per_decade: int = _FLICKER_PER_DECADE, adc_ref_t_k=ADC_REF_T_K_DEFAULT
                          ) -> "MonitorHw":
        """Build from ``mrm.params.realize`` output (``PlantParams``; fields of docs/PARAMETERS_FORMAT.md section 10).

        Uses pp.monitor, pp.timing (dt, ADC period, window, monitor latency ticks after the latency scale),
        pp.flags.shot_noise and pp.seeds['device'] (INL bow draw from ring.monitor.adc_inl_lsb; DNL thresholds via
        hw.device_seeds). An absent ring.monitor.dark_activation_ev takes 0.37 eV (inferred, 02 section 10; realize
        now defaults it the same way). baseline_a, mux_order, stuck_flagged and flicker_f_min_hz default to the
        MonitorParams fields (ring.monitor.baseline_ua, adc_mux.order, stuck_flagged, flicker_band_min_hz; integrator
        additions); a keyword argument overrides them.
        """
        m, t = pp.monitor, pp.timing
        B, N = int(pp.batch), int(pp.n_rings)
        dt = float(t.dt_s)
        if baseline_a is None:
            bl = getattr(m, "baseline_a", None)
            baseline_a = 0.0 if bl is None else np.asarray(bl, dtype=float).reshape(-1)
        if mux_order is None:
            mux_order = str(getattr(m, "mux_order", "round_robin"))
        if stuck_flagged is None:
            stuck_flagged = bool(getattr(m, "stuck_flagged", True))
        if flicker_f_min_hz is None:
            flicker_f_min_hz = float(getattr(m, "flicker_band_min_hz", _FLICKER_F_MIN_HZ))
        inl = np.asarray(m.adc_inl_lsb, dtype=float).reshape(-1)
        bow = draw_inl_bow_lsb(pp.seeds["device"], N, inl) if np.any(inl > 0) else 0.0
        afe = int(m.afe_type)
        ea_ev = m.dark_activation_j.reshape(-1) / Q_E
        if "ring.monitor.dark_activation_ev" not in (getattr(pp, "values", None) or {}):
            # realize fills an absent key with 0, which would make the dark current temperature-independent;
            # 07 P1-9 / D20 fix the law, so an absent E_a takes the inferred default (02 section 10)
            ea_ev = np.full(B, _DEFAULT_DARK_EA_EV)
        hw = cls.build(
            B, N, dt, responsivity_a_per_w=m.responsivity_a_per_w, adc_full_scale_a=m.adc_full_scale_a.reshape(-1),
            gain_scale=m.gain_scale.reshape(-1), dark_i0_a=m.dark_i0_a.reshape(-1),
            dark_t0_k=np.nan_to_num(m.dark_t0_k.reshape(-1), nan=_DEFAULT_T0_K),
            dark_activation_ev=ea_ev, background_fraction=m.background_fraction,
            baseline_a=baseline_a, afe_type=afe,
            integration_window_s=(t.integration_window_ticks * dt if afe == AFE_INTEGRATE_DUMP else None),
            afe_tau_s=(m.afe_tau_s if afe == AFE_RC else None), adc_rate_hz=1.0 / (m.adc_period_ticks * dt),
            adc_latency_s=np.asarray(t.monitor_latency_ticks).reshape(-1) * dt, adc_bits=m.adc_bits,
            adc_offset_codes=m.adc_offset_codes, adc_gain_error_frac=m.adc_gain_error_frac, adc_inl_bow_lsb=bow,
            adc_dnl_lsb=m.adc_dnl_lsb.reshape(-1), input_noise_a_per_rt_hz=m.input_noise_a_per_rt_hz.reshape(-1),
            flicker_corner_hz=m.flicker_corner_hz.reshape(-1), flicker_f_min_hz=flicker_f_min_hz,
            flicker_f_max_hz=flicker_f_max_hz, flicker_per_decade=flicker_per_decade,
            adc_ref_tempco_per_k=m.adc_ref_tempco_per_k.reshape(-1), adc_ref_t_k=adc_ref_t_k,
            pga_gains=m.pga_gains, channels_per_adc=m.channels_per_adc, mux_order=mux_order,
            shot_noise=bool(pp.flags.shot_noise), stuck_flagged=stuck_flagged)
        hw.device_seeds = _arr_b(pp.seeds["device"], B, "pp.seeds['device']", np.int64)
        return hw

    def reference_tap(self, *, responsivity_a_per_w=None, dark_i0_a=None, adc_full_scale_a=None, baseline_a=0.0,
                      stream: str = "adc_monitor_ref") -> "MonitorHw":
        """A reference-tap channel (07 P1-9, 06 S8) with this channel's AFE and ADC but its own responsivity, dark
        current, full scale and analog baseline (default none: the drop channel's baseline is set for the drop
        current, not for the tap), its own noise stream and the slot offset REF_TAP_SLOT_BASE. Its optical input is
        the tap coupling times the bus power (the caller computes it; no background term)."""
        pr = self.pr.copy()
        pw = self.pw.copy()
        if responsivity_a_per_w is not None:
            pr[:, :, PR_RESP] = _arr_bn(responsivity_a_per_w, self.B, self.N, "responsivity_a_per_w")
        pr[:, :, PR_BG] = 0.0
        pr[:, :, PR_IBASE] = _arr_bn(baseline_a, self.B, self.N, "baseline_a")
        if dark_i0_a is not None:
            pw[:, PW_DARK_I0] = _arr_b(dark_i0_a, self.B, "dark_i0_a")
        if adc_full_scale_a is not None:
            pw[:, PW_LSB] = _arr_b(adc_full_scale_a, self.B, "adc_full_scale_a") / self.code_max
        return MonitorHw(B=self.B, N=self.N, dt_s=self.dt_s, afe_type=self.afe_type, window_ticks=self.window_ticks,
                         period_ticks=self.period_ticks, adc_bits=self.adc_bits, code_max=self.code_max,
                         shot_on=self.shot_on, stuck_flagged=self.stuck_flagged, phase=self.phase.copy(),
                         latency_ticks=self.latency_ticks.copy(), pga_gains=self.pga_gains.copy(), pr=pr, pw=pw,
                         fl_coef=self.fl_coef.copy(), slot_base=self.slot_base + REF_TAP_SLOT_BASE, stream=stream,
                         afe_tau_s=self.afe_tau_s,
                         device_seeds=None if self.device_seeds is None else self.device_seeds.copy())

    def ideal_code(self, i_a, pga_select=0, t_adc_k=None) -> np.ndarray:
        """Ideal (unquantized, noiseless) ADC code of a PD current i_a (A, broadcast to (B, N)): the transfer used by
        the kernel without INL, DNL or clipping. For calibration replicas and tests."""
        i = _arr_bn(i_a, self.B, self.N, "i_a")
        g = 1.0
        if self.pga_gains.size:
            g = self.pga_gains[np.clip(np.asarray(pga_select), 0, self.pga_gains.size - 1)]
        t = self.pw[:, PW_T_REF][:, None] if t_adc_k is None else _arr_bn(t_adc_k, self.B, self.N, "t_adc_k")
        ref = 1.0 + self.pw[:, PW_TEMPCO][:, None] * (t - self.pw[:, PW_T_REF][:, None])
        return ((i - self.pr[:, :, PR_IBASE]) * g / self.pw[:, PW_LSB][:, None] * (1.0 + self.pr[:, :, PR_GAIN_ERR])
                / ref + self.pr[:, :, PR_OFFSET])

    def codes_per_w(self, pga_select=0, t_adc_k=None) -> np.ndarray:
        """Small-signal monitor gain, ADC codes per W of optical power at the PD, (B, N): gain_scale R g (1 +
        gain_err) / (LSB (1 + tc (T_adc - T_ref))). Times 1e-6 it is the 07 P1-9 identifiable quantity
        ``monitor_codes_per_uw_at_drop`` (before the drop-port transfer, which the optics owns)."""
        one = self.ideal_code(1.0, pga_select, t_adc_k) - self.ideal_code(0.0, pga_select, t_adc_k)   # codes per A
        return one * self.pw[:, PW_GAIN_SCALE][:, None] * self.pr[:, :, PR_RESP]

    def photocurrent_a(self, p_pd_w, p_bg_w=0.0, t_pd_k=None) -> tuple[np.ndarray, np.ndarray]:
        """(i_sig, i_dark) in A, (B, N): the PD model of the kernel (for tests, replicas and the identifiable block)."""
        p = _arr_bn(p_pd_w, self.B, self.N, "p_pd_w")
        pb = _arr_bn(p_bg_w, self.B, self.N, "p_bg_w")
        t = (np.broadcast_to(self.pw[:, PW_DARK_T0][:, None], (self.B, self.N)) if t_pd_k is None
             else _arr_bn(t_pd_k, self.B, self.N, "t_pd_k"))
        i_sig = self.pw[:, PW_GAIN_SCALE][:, None] * self.pr[:, :, PR_RESP] * (p + self.pr[:, :, PR_BG] * pb)
        i0 = self.pw[:, PW_DARK_I0][:, None]
        t0 = self.pw[:, PW_DARK_T0][:, None]
        ea = self.pw[:, PW_DARK_EA][:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            i_d = np.where(i0 > 0, i0 * np.exp(-(ea / K_B) * (1.0 / t - 1.0 / t0)), 0.0)
        return i_sig, i_d

    def white_sigma_a(self, i_sig_a, i_dark_a=0.0) -> np.ndarray:
        """Analytic rms of the white noise (shot + TIA) of one sample, A (B, N): S/(2 W) for integrate_dump, S/(4 tau)
        for rc, S = 2 q (i_sig + i_d) [shot on] + S_tia (07 P1-13; the dt-invariant reference of the tests)."""
        i = _arr_bn(i_sig_a, self.B, self.N, "i_sig_a") + _arr_bn(i_dark_a, self.B, self.N, "i_dark_a")
        s = self.pw[:, PW_S_TIA][:, None] + (2.0 * Q_E * np.abs(i) if self.shot_on else 0.0)
        if self.afe_type == AFE_INTEGRATE_DUMP:
            return np.sqrt(s / (2.0 * self.window_ticks * self.dt_s))
        return np.sqrt(s / (4.0 * self.afe_tau_s))


class MonitorRelease(NamedTuple):
    """Monitor channels released at one controller edge (07 P0-7). Arrays are copies (safe to keep)."""
    samples: np.ndarray       # (B, N, M) codes, valid in [:count]
    sample_ticks: np.ndarray  # (B, N, M) int64
    count: np.ndarray         # (B, N) int32
    latest: np.ndarray        # (B, N) held code
    valid: np.ndarray         # (B, N) bool (monitor_valid)
    overrange: np.ndarray     # (B, N) bool (adc_overrange)
    new: np.ndarray           # (B, N) bool: at least one sample since the last release
    lost: int                 # samples dropped because more than M arrived


class MonitorBank:
    """State and stepping of N monitor channels in B worlds (07 P0-7, P1-9, P1-10, P1-13).

    Usage::
        hw = MonitorHw.build(B, N, 1e-6, responsivity_a_per_w=0.95, adc_full_scale_a=40e-6, ...)
        bank = MonitorBank(hw, noise_seeds=rng.batch_seeds(7, B), device_seeds=pp.seeds['device'], m_max=M,
                           pop_period_ticks=controller_period)
        bank.tick(k, p_pd_w, p_bg_w, t_pd_k)   # once per plant interval [k, k+1), in order
        rel = bank.release(edge_tick)          # at each controller edge (event order step 2)
        sensors, new = bank.channels(rel)      # tx.monitor.* / monitor_valid / adc_overrange dicts

    m_max defaults to ceil(pop_period_ticks / ADC period), the most samples one release can see. device_seeds
    defaults to hw.device_seeds (set by MonitorHw.from_plant_params), else 0 for every world.
    """

    def __init__(self, hw: MonitorHw, noise_seeds, device_seeds=None, m_max: int | None = None,
                 pop_period_ticks: int = 1, prefix: str = "tx.monitor", valid_name: str = "monitor_valid",
                 overrange_name: str = "adc_overrange"):
        self.hw = hw
        B, N = hw.B, hw.N
        self.prefix, self.valid_name, self.overrange_name = prefix, valid_name, overrange_name
        pop = max(int(pop_period_ticks), 1)
        # default M: the most samples one release can see (ceil of the release period over the ADC period)
        self.m_max = int(m_max) if m_max is not None else max(1, -(-pop // hw.period_ticks))
        if self.m_max < 1:
            raise ValueError("m_max must be >= 1")
        if device_seeds is None:
            device_seeds = hw.device_seeds        # from_plant_params carries pp.seeds['device']
        dev = np.zeros(B, np.int64) if device_seeds is None else _arr_b(device_seeds, B, "device_seeds", np.int64)
        self.key_dnl = _keys(dev, B, hw.stream + "_dnl")
        cap = fifo_capacity(int(hw.latency_ticks.max()), hw.period_ticks, pop)
        self.fv = np.zeros((B, N, cap), np.int64)
        self.fr = np.zeros((B, N, cap), np.int64)
        self.fm = np.zeros((B, N, 2), np.int64)
        J = hw.fl_coef.shape[1]
        self.sf = np.zeros((B, N, NSF))
        self.si = np.zeros((B, N, NSI), np.int64)
        self.fx = np.zeros((B, N, J))
        self.fc = np.zeros((B, N, J))
        self.out_sampled = np.zeros((B, N), np.int8)
        dtype = adc_channel_dtype(hw.adc_bits)
        self._samples = np.zeros((B, N, self.m_max), np.int64)
        self._ticks = np.zeros((B, N, self.m_max), np.int64)
        self._count = np.zeros((B, N), np.int64)
        self._latest = np.zeros((B, N), np.int64)
        self._valid = np.ones((B, N), np.bool_)
        self._over = np.zeros((B, N), np.bool_)
        self._new = np.zeros((B, N), np.bool_)
        self._dtype = dtype
        self._zeros_i8 = np.zeros((B, N), np.int8)
        self._zeros_i64 = np.zeros((B, N), np.int64)
        self.fifo_lost = 0               # samples lost because release() was called less often than pop_period_ticks
        self.reset(noise_seeds)

    def reset(self, noise_seeds=None) -> None:
        """Clear every state and FIFO; optionally re-key the noise (07 P1-13: reset never resamples the device)."""
        B = self.hw.B
        if noise_seeds is not None:
            self.key_adc = _keys(noise_seeds, B, self.hw.stream)
            self.key_shot = _keys(noise_seeds, B, "shot_noise")
        self.sf[:] = 0.0
        self.si[:] = 0
        self.fx[:] = 0.0
        self.fc[:] = 0.0
        self.fm[:] = 0
        self._samples[:] = 0
        self._ticks[:] = 0
        self._latest[:] = 0
        self._count[:] = 0
        self._valid[:] = True
        self._over[:] = False
        self._new[:] = False
        self.out_sampled[:] = 0
        self.fifo_lost = 0

    def tick(self, k: int, p_pd_w, p_bg_w=0.0, t_pd_k=None, t_adc_k=None, pga_select=None, fault=None,
             parallel: bool = False) -> np.ndarray:
        """Integrate interval [k, k+1). p_pd_w, p_bg_w in W; t_pd_k (default: the dark-current reference
        temperature), t_adc_k (default: t_pd_k) in K; pga_select int; fault int8 FAULT_* masks. All broadcast to
        (B, N). Returns out_sampled (B, N) int8 (a view)."""
        hw = self.hw
        B, N = hw.B, hw.N
        p = _arr_bn(p_pd_w, B, N, "p_pd_w")
        pb = _arr_bn(p_bg_w, B, N, "p_bg_w")
        t_pd = (np.ascontiguousarray(np.broadcast_to(hw.pw[:, PW_DARK_T0][:, None], (B, N))) if t_pd_k is None
                else _arr_bn(t_pd_k, B, N, "t_pd_k"))
        t_adc = t_pd if t_adc_k is None else _arr_bn(t_adc_k, B, N, "t_adc_k")
        sel = self._zeros_i64 if pga_select is None else _arr_bn(pga_select, B, N, "pga_select", np.int64)
        flt = self._zeros_i8 if fault is None else _arr_bn(fault, B, N, "fault", np.int8)
        fn = monitor_tick_par if parallel else monitor_tick
        fn(np.int64(k), p, pb, t_pd, t_adc, sel, flt, hw.pr, hw.pw, hw.latency_ticks, hw.phase, hw.pga_gains,
           hw.fl_coef, hw.afe_type, hw.window_ticks, hw.period_ticks, hw.code_max, hw.shot_on, hw.stuck_flagged,
           hw.slot_base, self.key_adc, self.key_shot, self.key_dnl, self.sf, self.si, self.fx, self.fc, self.fv,
           self.fr, self.fm, self.out_sampled)
        if self.out_sampled.max() > 1:
            self.fifo_lost += int(np.count_nonzero(self.out_sampled == 2))
        return self.out_sampled

    def release(self, now: int) -> MonitorRelease:
        """Release the samples ready at tick ``now`` (07 P0-6 step 2)."""
        lost = monitor_release(np.int64(now), self.hw.latency_ticks, self.fv, self.fr, self.fm, self._samples,
                               self._ticks, self._count, self._latest, self._valid, self._over, self._new)
        # copies: a controller may keep an Observation across steps, and the next release reuses the buffers
        return MonitorRelease(self._samples.astype(self._dtype), self._ticks.copy(), self._count.astype(np.int32),
                              self._latest.astype(self._dtype), self._valid.copy(), self._over.copy(),
                              self._new.copy(), int(lost))

    def channels(self, rel: MonitorRelease) -> tuple[dict, dict]:
        """(sensors, new) dicts with the 07 P0-7 channel names (prefix ``tx.monitor`` for the main bank)."""
        p = self.prefix
        always = np.ones_like(rel.new)
        sensors = {f"{p}.samples": rel.samples, f"{p}.sample_ticks": rel.sample_ticks, f"{p}.count": rel.count,
                   f"{p}.latest": rel.latest, self.valid_name: rel.valid, self.overrange_name: rel.overrange}
        new = {f"{p}.samples": rel.new, f"{p}.sample_ticks": rel.new, f"{p}.count": always, f"{p}.latest": rel.new,
               self.valid_name: always, self.overrange_name: always}
        return sensors, new

    def level_codes(self, p1_w, p0_w, p_bg_w=0.0, t_pd_k=None, t_adc_k=None, pga_select=None, tau_s: float = 0.0,
                    noise_a_per_rt_hz: float | None = None, rin_per_hz=0.0):
        """(mu1, mu0, sig1, sig0) (B, N) codes of the two NRZ levels for the bit-statistics sampler
        (:func:`monitor_level_codes_row`). noise_a_per_rt_hz: input noise density of the bit-rate sampler front end
        (one-sided, A/sqrt(Hz)); None uses the monitor's own input_noise_a_per_rt_hz. rin_per_hz: laser RIN (1/Hz,
        scalar or (B,)) seen at the sampler bandwidth."""
        hw = self.hw
        B, N = hw.B, hw.N
        p1 = _arr_bn(p1_w, B, N, "p1_w")
        p0 = _arr_bn(p0_w, B, N, "p0_w")
        pb = _arr_bn(p_bg_w, B, N, "p_bg_w")
        t_pd = (np.ascontiguousarray(np.broadcast_to(hw.pw[:, PW_DARK_T0][:, None], (B, N))) if t_pd_k is None
                else _arr_bn(t_pd_k, B, N, "t_pd_k"))
        t_adc = t_pd if t_adc_k is None else _arr_bn(t_adc_k, B, N, "t_adc_k")
        sel = self._zeros_i64 if pga_select is None else _arr_bn(pga_select, B, N, "pga_select", np.int64)
        s_tia = -1.0 if noise_a_per_rt_hz is None else float(noise_a_per_rt_hz) ** 2
        rin = _arr_b(rin_per_hz, B, "rin_per_hz")
        out = [np.zeros((B, N)) for _ in range(4)]
        for b in range(B):
            monitor_level_codes_row(p1[b], p0[b], pb[b], t_pd[b], t_adc[b], sel[b], hw.pr[b], hw.pw[b],
                                    hw.pga_gains, hw.shot_on, float(tau_s), out[0][b], out[1][b], out[2][b],
                                    out[3][b], s_tia, float(rin[b]))
        return tuple(out)


# ------------------------------------------------------------------------------------------ temperature sensors
@dataclass
class TempSensorsHw:
    """Realized temperature sensors (07 P1-10 ``temp_sensors``) for B worlds and S sensors, SI (K, s)."""

    B: int
    S: int
    dt_s: float
    names: tuple
    types: np.ndarray        # (S,) int64 TEMP_SENSOR_TYPES (recorded; the model is the same for every type)
    period_ticks: np.ndarray  # (S,) int64
    latency_ticks: np.ndarray  # (B, S) int64
    code_max: np.ndarray     # (S,) int64
    tp: np.ndarray           # (B, S, NTP)
    fl_coef: np.ndarray      # (B, S, J, 10)

    @classmethod
    def build(cls, B: int, dt_s: float, sensors) -> "TempSensorsHw":
        """sensors: a sequence of mappings, one per sensor, with SI fields (scalars or (B,) arrays):
        name, type ('diode' | 'rtd' | 'thermistor'), sample_rate_hz (whole ticks), latency_s, noise_rms_k (white,
        per sample), flicker_corner_hz, resolution_k (code LSB), offset_k and gain_error_frac (already drawn per
        device), self_heating_k, adc_bits (default 12, assumed, mrm.core.channels.TempSensorHw), code_zero_k
        (temperature of code 0; default 233.15 K = -40 C, assumed, same source), gain_ref_k (default 298.15 K)."""
        dt = float(dt_s)
        S = len(sensors)
        names, types, per, cmax = [], [], [], []
        lat = np.zeros((B, S), np.int64)
        tp = np.zeros((B, S, NTP))
        coefs = [[None] * S for _ in range(B)]
        for i, s in enumerate(sensors):
            names.append(str(s.get("name", f"t{i}")))
            types.append(TEMP_SENSOR_TYPES[str(s.get("type", "diode"))])
            p = rate_to_period_ticks(float(s.get("sample_rate_hz", 1e3)), dt, f"temp_sensors[{i}].sample_rate_hz")
            per.append(p)
            bits = int(s.get("adc_bits", 12))
            cmax.append((1 << bits) - 1)
            lat_s = _arr_b(s.get("latency_s", 1e-3), B, "latency_s")
            lat[:, i] = [seconds_to_ticks(float(x), dt, f"temp_sensors[{i}].latency_s") for x in lat_s]
            noise = _arr_b(s.get("noise_rms_k", 0.0), B, "noise_rms_k")
            res = _arr_b(s.get("resolution_k", 0.0625), B, "resolution_k")
            if np.any(res <= 0):
                raise ValueError(f"temp_sensors[{i}].resolution_k must be > 0")
            tp[:, i, _TP_NOISE] = noise
            tp[:, i, _TP_RES] = res
            tp[:, i, _TP_OFFSET] = _arr_b(s.get("offset_k", 0.0), B, "offset_k")
            tp[:, i, _TP_GAIN_ERR] = _arr_b(s.get("gain_error_frac", 0.0), B, "gain_error_frac")
            tp[:, i, _TP_SELF] = _arr_b(s.get("self_heating_k", 0.0), B, "self_heating_k")
            tp[:, i, _TP_ZERO_K] = _arr_b(s.get("code_zero_k", 233.15), B, "code_zero_k")
            tp[:, i, _TP_GAIN_REF_K] = _arr_b(s.get("gain_ref_k", _TEMP_GAIN_REF_K), B, "gain_ref_k")
            fcn = _arr_b(s.get("flicker_corner_hz", 0.0), B, "flicker_corner_hz")
            t_s = p * dt
            for b in range(B):
                # white PSD equivalent to sigma^2 per boxcar sample of length T_s: S = 2 sigma^2 T_s
                coefs[b][i] = flicker_coefficients(2.0 * noise[b] ** 2 * t_s, float(fcn[b]),
                                                   afe_type=AFE_INTEGRATE_DUMP, period_s=t_s, window_s=t_s)
        jmax = max((c.shape[0] for row in coefs for c in row), default=0)
        fl = np.zeros((B, S, jmax, 10))
        for b in range(B):
            for i in range(S):
                fl[b, i, :coefs[b][i].shape[0]] = coefs[b][i]
        return cls(B=int(B), S=S, dt_s=dt, names=tuple(names), types=np.array(types, np.int64),
                   period_ticks=np.array(per, np.int64), latency_ticks=lat, code_max=np.array(cmax, np.int64),
                   tp=tp, fl_coef=fl)

    @classmethod
    def from_plant_params(cls, pp, *, adc_bits=12, code_zero_k=233.15) -> "TempSensorsHw":
        """From ``PlantParams.temp_sensors`` (TempSensorParams) and pp.timing.dt_s. Each sensor's adc_bits and
        code_zero_k come from TempSensorParams (temp_sensors[].adc_bits, code_zero_c); a sensor that leaves them
        out takes the keyword defaults, those of mrm.core.channels.TempSensorHw (12 bits, -40 degC; assumed)."""
        ts = pp.temp_sensors
        dt = float(pp.timing.dt_s)
        B = int(pp.batch)
        inv = {v: k for k, v in TEMP_SENSOR_TYPES.items()}
        sensors = []
        for i, name in enumerate(ts.names):
            if int(ts.sample_period_ticks[i]) < 1:
                raise ValueError(f"temp_sensors[{i}] ({name}): sample_rate_hz is missing or gives a period below one "
                                 "tick")
            sensors.append(dict(
                name=name, type=inv[int(ts.type[i])], sample_rate_hz=1.0 / (int(ts.sample_period_ticks[i]) * dt),
                latency_s=np.asarray(ts.latency_ticks)[:, i] * dt, noise_rms_k=np.asarray(ts.noise_rms_k)[:, i],
                flicker_corner_hz=np.nan_to_num(np.asarray(ts.flicker_corner_hz)[:, i]),
                resolution_k=np.asarray(ts.resolution_k)[:, i], offset_k=np.asarray(ts.offset_k)[:, i],
                gain_error_frac=np.asarray(ts.gain_error_frac)[:, i],
                self_heating_k=np.asarray(ts.self_heating_k)[:, i], adc_bits=_ts_bits(ts, i, adc_bits),
                code_zero_k=_ts_zero(ts, i, code_zero_k)))
        return cls.build(B, dt, sensors)


def _ts_bits(ts, i: int, default: int) -> int:
    """temp_sensors[i].adc_bits from TempSensorParams (0 or absent -> default)."""
    a = getattr(ts, "adc_bits", None)
    v = int(np.asarray(a)[i]) if a is not None and np.asarray(a).size > i else 0
    return v if v > 0 else int(default)


def _ts_zero(ts, i: int, default: float) -> float:
    """temp_sensors[i].code_zero_c (as K) from TempSensorParams (nan or absent -> default)."""
    a = getattr(ts, "code_zero_k", None)
    v = float(np.asarray(a)[i]) if a is not None and np.asarray(a).size > i else math.nan
    return v if math.isfinite(v) else float(default)


class TempSensorBank:
    """State of the temperature sensors (channels ``temp.<name>``, held, with new-sample flags; 07 P1-10)."""

    def __init__(self, hw: TempSensorsHw, noise_seeds, pop_period_ticks: int = 1):
        self.hw = hw
        B, S = hw.B, hw.S
        cap = max([fifo_capacity(int(hw.latency_ticks[:, i].max()), int(hw.period_ticks[i]),
                                 max(int(pop_period_ticks), 1)) for i in range(S)] or [1])
        J = hw.fl_coef.shape[2]
        self.fv = np.zeros((B, S, cap), np.int64)
        self.fr = np.zeros((B, S, cap), np.int64)
        self.fm = np.zeros((B, S, 2), np.int64)
        self.st = np.zeros((B, S, 2))
        self.fx = np.zeros((B, S, J))
        self.fc = np.zeros((B, S, J))
        self.primed = np.zeros((B, S), np.int64)
        self.latest = np.zeros((B, S), np.int64)
        self.new = np.zeros((B, S), np.bool_)
        self.out_sampled = np.zeros((B, S), np.int8)
        self.reset(noise_seeds)

    def reset(self, noise_seeds=None) -> None:
        """Clear the state and FIFOs; optionally re-key the noise (device offsets and gains are kept, 07 P1-13)."""
        if noise_seeds is not None:
            self.key = _keys(noise_seeds, self.hw.B, "adc_temp")
        self.st[:] = 0.0
        self.fx[:] = 0.0
        self.fc[:] = 0.0
        self.primed[:] = 0
        self.fm[:] = 0
        self.latest[:] = 0
        self.new[:] = False
        self.out_sampled[:] = 0

    def tick(self, k: int, t_site_k) -> np.ndarray:
        """Integrate interval [k, k+1); t_site_k (B, S) in K. Returns out_sampled (B, S) int8."""
        hw = self.hw
        t = np.ascontiguousarray(np.broadcast_to(np.asarray(t_site_k, dtype=np.float64), (hw.B, hw.S)))
        temp_tick(np.int64(k), t, hw.tp, hw.period_ticks, hw.latency_ticks, hw.code_max, hw.fl_coef, self.key,
                  self.st, self.fx, self.fc, self.primed, self.fv, self.fr, self.fm, self.out_sampled)
        return self.out_sampled

    def release(self, now: int) -> tuple[dict, dict]:
        """({'temp.<name>': (B, 1) codes}, {'temp.<name>': (B, 1) new flags}) at tick ``now``."""
        temp_release(np.int64(now), self.fv, self.fr, self.fm, self.latest, self.new)
        sensors, new = {}, {}
        for i, name in enumerate(self.hw.names):
            bits = int(self.hw.code_max[i]).bit_length()
            sensors[f"temp.{name}"] = self.latest[:, i:i + 1].astype(adc_channel_dtype(bits))
            new[f"temp.{name}"] = self.new[:, i:i + 1].copy()
        return sensors, new


# ------------------------------------------------------------------------------------------ heater readback
@dataclass
class ReadbackHw:
    """Heater V/I readback converter (07 P1-10 ring.heater.readback), realized."""

    B: int
    N: int
    dt_s: float
    adc_bits: int
    code_max: int
    period_ticks: int
    latency_ticks: np.ndarray   # (B,) int64
    lsb_v: np.ndarray           # (B,) V per code
    lsb_i: np.ndarray           # (B,) A per code
    noise_codes: np.ndarray     # (B,) rms codes
    average: int

    @classmethod
    def build(cls, B: int, N: int, dt_s: float, *, adc_bits: int = 12, v_full_scale_v=2.0, i_full_scale_a=0.032,
              rate_hz: float = 10e3, latency_s=1e-4, noise_rms_codes=0.0, average: bool = False) -> "ReadbackHw":
        """Defaults are the assumed fallbacks of mrm.core.channels.ChannelHardware (12 bits, 2.0 V, 32 mA, 10 kHz,
        100 us). noise_rms_codes has no schema key (assumed 0). average=False: track-and-hold of the last tick.
        adc_bits must be 1..31: the V and I codes share one int64 FIFO word (v | i << 32)."""
        dt = float(dt_s)
        bits = int(adc_bits)
        if not 1 <= bits <= 31:
            raise ValueError(f"ring.heater.readback.adc_bits = {bits}: the readback model supports 1..31 bits")
        code_max = (1 << bits) - 1
        per = rate_to_period_ticks(rate_hz, dt, "ring.heater.readback.rate_hz")
        lat = _arr_b(latency_s, B, "latency_s")
        lat_t = np.array([seconds_to_ticks(float(x), dt, "ring.heater.readback.latency_s") for x in lat], np.int64)
        return cls(B=int(B), N=int(N), dt_s=dt, adc_bits=bits, code_max=code_max, period_ticks=per,
                   latency_ticks=lat_t, lsb_v=_arr_b(v_full_scale_v, B, "v_full_scale_v") / code_max,
                   lsb_i=_arr_b(i_full_scale_a, B, "i_full_scale_a") / code_max,
                   noise_codes=_arr_b(noise_rms_codes, B, "noise_rms_codes"), average=int(bool(average)))

    @classmethod
    def from_plant_params(cls, pp, *, adc_bits: int | None = None, noise_rms_codes=None, average: bool | None = None
                          ) -> "ReadbackHw":
        """From PlantParams: pp.timing.readback_period_ticks / readback_latency_ticks (after the latency scale) and
        the ring.heater.readback.{v_full_scale_v, i_full_scale_a} values; a key the preset leaves out takes the
        assumed fallback of mrm.core.channels.ChannelHardware (2.0 V, 32 mA, 10 kHz). adc_bits, noise_rms_codes and
        average default to HeaterParams.readback_adc_bits / readback_noise_rms_codes / readback_average
        (ring.heater.readback.adc_bits, noise_rms_codes, average; 12 bits assumed when absent); keyword arguments
        override them. Raises if ring.heater.readback.present is False."""
        if not bool(getattr(pp.heater, "readback_present", False)):
            raise ValueError("ring.heater.readback.present is False: no heater readback channel")
        dt = float(pp.timing.dt_s)
        vals = getattr(pp, "values", {}) or {}

        def first(key, default):
            v = vals.get(key)
            return default if v is None else float(np.asarray(v, dtype=float).reshape(-1)[0])

        def per_world(key, default):
            v = vals.get(key)
            return default if v is None else np.asarray(v, dtype=float).reshape(-1)
        per = int(pp.timing.readback_period_ticks)
        rate = 1.0 / (per * dt) if per > 0 else 10e3      # assumed fallback, mrm.core.channels.ChannelHardware
        h = pp.heater
        if adc_bits is None:
            adc_bits = int(getattr(h, "readback_adc_bits", 0) or 0) or int(first("ring.heater.readback.adc_bits", 12))
            # 12: assumed, ChannelHardware
        if noise_rms_codes is None:
            nz = getattr(h, "readback_noise_rms_codes", None)
            noise_rms_codes = 0.0 if nz is None else np.asarray(nz, dtype=float).reshape(-1)
        if average is None:
            average = bool(getattr(h, "readback_average", False))
        # fallbacks 2.0 V and 32 mA: assumed, mrm.core.channels.ChannelHardware (above the 1.5-1.6 V, ~29 mA drive)
        return cls.build(int(pp.batch), int(pp.n_rings), dt, adc_bits=int(adc_bits),
                         v_full_scale_v=per_world("ring.heater.readback.v_full_scale_v", 2.0),
                         i_full_scale_a=per_world("ring.heater.readback.i_full_scale_a", 0.032), rate_hz=rate,
                         latency_s=np.asarray(pp.timing.readback_latency_ticks).reshape(-1) * dt,
                         noise_rms_codes=noise_rms_codes, average=average)


class ReadbackBank:
    """State of the heater readback channels ``heater.readback_v`` / ``heater.readback_i`` (held, with new flags)."""

    def __init__(self, hw: ReadbackHw, noise_seeds, pop_period_ticks: int = 1):
        self.hw = hw
        B, N = hw.B, hw.N
        cap = fifo_capacity(int(hw.latency_ticks.max()), hw.period_ticks, max(int(pop_period_ticks), 1))
        self.fv = np.zeros((B, N, cap), np.int64)
        self.fr = np.zeros((B, N, cap), np.int64)
        self.fm = np.zeros((B, N, 2), np.int64)
        self.st = np.zeros((B, N, 3))
        self.latest_v = np.zeros((B, N), np.int64)
        self.latest_i = np.zeros((B, N), np.int64)
        self.new = np.zeros((B, N), np.bool_)
        self.out_sampled = np.zeros((B, N), np.int8)
        self.reset(noise_seeds)

    def reset(self, noise_seeds=None) -> None:
        if noise_seeds is not None:
            self.key = _keys(noise_seeds, self.hw.B, "adc_readback")
        self.st[:] = 0.0
        self.fm[:] = 0
        self.latest_v[:] = 0
        self.latest_i[:] = 0
        self.new[:] = False
        self.out_sampled[:] = 0

    def tick(self, k: int, v_v, i_a) -> np.ndarray:
        """Interval [k, k+1) with heater voltage v_v and current i_a (B, N) (HeaterTickOut.v_v / i_a)."""
        hw = self.hw
        v = _arr_bn(v_v, hw.B, hw.N, "v_v")
        i = _arr_bn(i_a, hw.B, hw.N, "i_a")
        readback_tick(np.int64(k), v, i, hw.lsb_v, hw.lsb_i, hw.noise_codes, hw.period_ticks, hw.latency_ticks,
                      hw.code_max, hw.average, self.key, self.st, self.fv, self.fr, self.fm, self.out_sampled)
        return self.out_sampled

    def release(self, now: int) -> tuple[dict, dict]:
        readback_release(np.int64(now), self.fv, self.fr, self.fm, self.latest_v, self.latest_i, self.new)
        dt = adc_channel_dtype(self.hw.adc_bits)
        return ({"heater.readback_v": self.latest_v.astype(dt), "heater.readback_i": self.latest_i.astype(dt)},
                {"heater.readback_v": self.new.copy(), "heater.readback_i": self.new.copy()})


# ------------------------------------------------------------------------------------------ composite
class SensingBank:
    """Every sensor of the plant for the Python slow path: monitor (+ optional reference tap), temperature sensors and
    heater readback, with one ``tick`` per plant interval and one ``release`` per controller edge.

    ``release(now)`` returns (sensors, new): the channel dicts of 07 P0-7 / P1-10 (``tx.monitor.*``,
    ``monitor_valid``, ``adc_overrange``, ``temp.<name>``, ``heater.readback_v/i``, and ``tx.monitor_ref.*`` /
    ``monitor_ref_valid`` / ``adc_ref_overrange`` when a reference tap is present; the latter have no ChannelSpec yet).
    """

    def __init__(self, monitor: MonitorBank, temps: TempSensorBank | None = None,
                 readback: ReadbackBank | None = None, ref_tap: MonitorBank | None = None):
        self.monitor, self.temps, self.readback, self.ref_tap = monitor, temps, readback, ref_tap

    def reset(self, noise_seeds=None) -> None:
        for bank in (self.monitor, self.temps, self.readback, self.ref_tap):
            if bank is not None:
                bank.reset(noise_seeds)

    def tick(self, k: int, *, p_pd_w, p_bg_w=0.0, t_pd_k=None, t_adc_k=None, pga_select=None, fault=None,
             t_site_k=None, heater_v=None, heater_i=None, p_ref_w=None) -> None:
        self.monitor.tick(k, p_pd_w, p_bg_w, t_pd_k, t_adc_k, pga_select, fault)
        if self.ref_tap is not None:
            self.ref_tap.tick(k, 0.0 if p_ref_w is None else p_ref_w, 0.0, t_pd_k, t_adc_k, pga_select, None)
        if self.temps is not None and self.temps.hw.S > 0:
            if t_site_k is None:
                raise ValueError("temperature sensors need t_site_k (B, S)")
            self.temps.tick(k, t_site_k)
        if self.readback is not None:
            if heater_v is None or heater_i is None:
                raise ValueError("heater readback needs heater_v and heater_i (B, N)")
            self.readback.tick(k, heater_v, heater_i)

    def release(self, now: int) -> tuple[dict, dict]:
        sensors, new = self.monitor.channels(self.monitor.release(now))
        for bank in (self.temps, self.readback):
            if bank is not None:
                s, nw = bank.release(now)
                sensors.update(s)
                new.update(nw)
        if self.ref_tap is not None:
            s, nw = self.ref_tap.channels(self.ref_tap.release(now))
            sensors.update(s)
            new.update(nw)
        return sensors, new
