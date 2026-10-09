"""Normalized-coordinate data-modulation LUT with a validity domain (07 P2-6; fidelity.data_modulation = lut).

Owner: waveform. Contract: 07 P2-6 ("corrections keyed by (detuning/HWHM, kappa_e/kappa, swing in linewidths, mark and
transition density, rise time); store the domain, raise an error outside it, record the config hash; the correction
to the mean drop power is below 3 % for PRBS at tau_p ~ 3 ps and a 20 ps UI"); docs/research/03 sections 9.2-9.3.
Artefact: mrm/data/extracted/waveform_lut_<name>.npz (float32, compressed) with the sidecar waveform_lut_<name>.json
(schema waveform_lut/1, sha256 of the npz, build config and its hash, validation numbers, tool versions).

What the quasi-static plant does, and what the LUT corrects
-----------------------------------------------------------
The control plant averages the optics over the data (03 section 9.2, mrm.plant.optics):
P_bar = p1 P(level 1) + (1 - p1) P(level 0) with P the STATIC ring response at each NRZ level. The true time average
differs because the field cannot follow the junction instantly (photon lifetime) and the junction edge is finite
(driver + RC): during an edge the resonance sweeps through intermediate (possibly resonant) detunings. The LUT gives
the time-domain value relative to the static two-level value of the same quantity in the SAME normalized model, so
the engine applies the ratio (or the difference) to its own exact static numbers.

Normalized model (exact reduction of the single-mode TCMT, no backscatter, no loss modulation)
-------------------------------------------------------------------------------------------
With b = a1 kappa/(2 sqrt(kappa_e) s_in) and time in UI, da1/dt = -(i Delta + kappa/2) a1 - i sqrt(kappa_e) s_in becomes

    db/dt = -(1 + i x(t)) b/(2 tau_p) - i/(2 tau_p),     tau_p = 1/kappa (photon energy lifetime, in UI),

with x = 2 Delta/kappa the detuning in half-widths (03 section 2.1). The static solution is b = -i/(1 + i x), so
u = |b|^2 = L(x) = 1/(1 + x^2). Every port follows from b EXACTLY (03 sections 2.3-2.5):

    stored energy  U = (4 kappa_e P_in/kappa^2) u           -> drop kappa_d U, loss kappa_i U, heat eta kappa_i U
    through        P_thru/P_in = 1 + 4 r_e Im(b) + 4 r_e^2 u,   r_e = kappa_e/kappa

so kappa_e/kappa needs no axis for the means: the mean of u corrects drop, loss and heat by the same factor c_mean,
and the through power by energy conservation (P_thru = P_in - P_drop - P_loss - dU/dt, and dU/dt averages to zero
over a period): P_thru' = P_thru - (c_mean - 1) (P_drop + P_loss). kappa_e/kappa enters only the through-port OMA
(axis r_e). The resonance trajectory is x(t) = x0 + (x1 - x0) e(t): x0, x1 the static detunings of the two levels
(from the plant's phase model), e(t) in [0, 1] the normalized edge, CANONICALLY the NRZ pattern through a single pole
whose 10-90 % rise time is t_r (tau_e = t_r/ln 9). Physically e(t) is the normalized removed junction charge (the
resonance shift is proportional to it, 03 section 5.3), so t_r should be the 10-90 % rise of that charge for the real
driver + junction (DriverJunction.step_rise_time_s(quantity="charge")); the single-pole edge shape is an
approximation, quantified against full driver + nonlinear-junction runs in tests/verification/test_waveform.py.
Patterns: cyclic two-state Markov chains with mark density p1 and transition density td (driver.markov_bits; p1 = td =
0.5 is i.i.d. data, close to PRBS).

Definitions (identical in eye.py): mean = time average over the period; OMA = the largest |P1(phi) - P0(phi)| over
the sampling phase phi in [0, 2 UI), P1/P0 the means over the ones/zeros of bit k sampled at k UI + phi (a smooth
function of phi, refined between samples by the parabola vertex: it converges quadratically in the sample spacing).
Eye heights are not tabulated (a min/max over bits: kinked in phi and in the parameters); use waveform mode.

Keys, nodes, symmetry and interpolation
---------------------------------------
Query (x0, x1, p1, td, t_r/UI, tau_p/UI, r_e) -> canonical (x_c, w, p) with x_c = (x0 + x1)/2 (detuning/HWHM at the
level midpoint), w = |x1 - x0|/2 (= resonance swing / FWHM: "swing in linewidths") and p the mark density of the
level with the larger x. The model is invariant under x(t) -> -x(t) (b -> -conj(b): u and Im b unchanged), which with
relabelling the levels maps (x_c, w, p) -> (-x_c, w, 1 - p); the table stores x_c >= 0 only. Stored per node (float32):
U_TD (time-average u, rescaled to the nominal p1 of the node: c_node x U_ST(nominal)), OMA_D_TD (drop OMA in u units)
and, per r_e, OMA_T_TD (through OMA in P_in units). The static values are ANALYTIC (U_ST = p L(x1) + (1 - p) L(x0),
OMA_D_ST = |L(x1) - L(x0)|, OMA_T_ST = 4 r_e (1 - r_e) OMA_D_ST) and are evaluated at the same interpolation corners,
so the outputs are RATIOS OF INTERPOLANTS (c_mean = interp(U_TD)/interp(U_ST), ...): the static parts cancel most of
the interpolation error and the ratio stays finite where the static OMA vanishes. Multilinear in all axes.

Validity domain (stored in the artefact; outside it lut_eval_nb returns a status and the Python/strict paths raise)
-----------------------------------------------------------------------------------------------------------------
|x_c| <= max(ax_xc), w <= max(ax_w), p in [min, max](ax_p1), 0 <= td <= max(ax_td) and td <= 2 min(p, 1 - p),
t_r/UI and tau_p/UI within their axes, r_e within its axis (only the through OMA uses it). Not represented (the
correction is then an approximation, docs/handoff/waveform.md): backscatter g, loss modulation (q), multi-order
leakage, edges far from single-pole shape, PAM4. Interpolation accuracy: sidecar ``validation.interpolation``.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numba import njit, prange

from mrm.waveform.driver import markov_bits, pattern_stats
from mrm.waveform.eye import _best_phase
from mrm.waveform.tcmt_time import _mode_step

__all__ = [
    "SCHEMA", "EXTRACTED_DIR", "DEFAULT_AXES", "F6_U_TD", "F6_OMA_D_TD", "N_F6", "F7_OMA_T_TD", "N_F7",
    "OUT_C_MEAN", "OUT_C_OMA_THRU", "OUT_D_OMA_THRU", "OUT_C_OMA_DROP", "OUT_D_OMA_DROP", "OUT_D_MEAN_U", "N_OUT",
    "ST_XC", "ST_W", "ST_P1", "ST_TD", "ST_TR", "ST_TAU", "ST_RE", "ST_NONFINITE",
    "samples_per_bit_for", "normalized_node", "normalized_correction", "lut_eval_nb", "lut_eval_strict_nb",
    "lut_query_nb", "WaveformLUT", "build_lut", "load_lut", "LutDomainError", "p2_6_prbs_check",
    "interpolation_check",
]

SCHEMA = "waveform_lut/1"
EXTRACTED_DIR = Path(__file__).resolve().parents[1] / "data" / "extracted"

#: Default node axes (design choices; ranges in docs/handoff/waveform.md). tau_p/UI 0.07-0.25 covers Q_L 2020-4200 at
#: 50-56 Gb/s and 1310 nm (tau_p = Q/w0 = 1.4-2.9 ps; the g25/g26 presets run 56 and 50 Gb/s), but at 25 Gb/s only
#: Q_L >= 4030; t_r/UI up to 0.7; swings up to 1 FWHM (07 D9 needs ~0.5-0.7);
#: mark density 0.3-0.7 with transition density up to 0.6 (= 2 x 0.3, so every node is a feasible Markov pattern; PRBS
#: has td = 2^(k-1)/(2^k - 1), slightly above 0.5).
DEFAULT_AXES = {
    "xc": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0, 6.0],
    "w": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
    "p1": [0.3, 0.5, 0.7],
    "td": [0.0, 0.15, 0.3, 0.45, 0.6],
    "tr": [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7],
    "tau": [0.07, 0.085, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25],
    "re": [0.3, 0.4, 0.5, 0.6, 0.7],
}
_AX6 = ("xc", "w", "p1", "td", "tr", "tau")

# stored fields: tab6 over (xc, w, p1, td, tr, tau), tab7 over (..., re)
F6_U_TD, F6_OMA_D_TD = 0, 1
N_F6 = 2
F7_OMA_T_TD = 0
N_F7 = 1

# outputs of lut_eval_nb
OUT_C_MEAN = 0       # mean stored energy: time domain / static (drop, loss, heat; through by energy conservation)
OUT_C_OMA_THRU = 1   # through-port OMA: time domain / static
OUT_D_OMA_THRU = 2   # through-port OMA: time domain - static, in units of P_in
OUT_C_OMA_DROP = 3   # drop-port OMA ratio
OUT_D_OMA_DROP = 4   # drop-port OMA difference, in units of the drop peak 4 kappa_e kappa_d P_in / kappa^2
OUT_D_MEAN_U = 5     # mean u difference, in units of the peak energy 4 kappa_e P_in / kappa^2
N_OUT = 6

# status bits of lut_eval_nb (0 = inside the domain)
ST_XC, ST_W, ST_P1, ST_TD, ST_TR, ST_TAU, ST_RE, ST_NONFINITE = 1, 2, 4, 8, 16, 32, 64, 128
_ST_NAMES = {ST_XC: "|x_c| (detuning/HWHM)", ST_W: "w (swing/FWHM)", ST_P1: "mark density",
             ST_TD: "transition density (incl. td <= 2 min(p1, 1 - p1))", ST_TR: "rise time/UI",
             ST_TAU: "tau_p/UI", ST_RE: "kappa_e/kappa", ST_NONFINITE: "non-finite input"}

_LN9 = math.log(9.0)


class LutDomainError(ValueError):
    """A query outside the stored validity domain of the waveform LUT (07 P2-6)."""


def samples_per_bit_for(tau_ui: float, base: int = 32) -> int:
    """Samples per UI of a normalized node: at least ``base`` and h <= tau_p/2 (rounded up to a multiple of 8)."""
    return max(int(base), 8 * int(math.ceil(2.0 / (8.0 * tau_ui))))


# ============================================================================================ normalized node
@njit(cache=True)
def _lor(x):
    return 1.0 / (1.0 + x * x)


@njit(cache=True)
def _oma_max(dif):
    """Largest |dif| over the phases, refined by the parabola vertex (eye.py definition of the OMA)."""
    nph = dif.shape[0]
    ad = np.empty(nph)
    for j in range(nph):
        ad[j] = abs(dif[j])
    jb, d, v = _best_phase(ad)
    return v


@njit(cache=True)
def _node_nb(x0, x1, bits, n_warm, s, tau_ui, tr_ui, re_grid, out6, out7):
    """One normalized node (module docstring): simulate the cyclic pattern ``bits`` (warm-up n_warm bits, record
    n + 1 bits) at s samples per UI. out6 (4,) = (u_td, u_st with the pattern's p1, oma_drop_td, oma_drop_st);
    out7 (2, n_re) = (oma_thru_td, oma_thru_st) per r_e."""
    n = bits.shape[0]
    n_tot = n_warm + n + 1
    h = 1.0 / s
    kh = 0.5 / tau_ui
    beta = -1j * kh
    dx = x1 - x0
    q = math.exp(-0.5 * h * _LN9 / tr_ui) if tr_ui > 0.0 else 0.0
    rec = np.empty((n + 1) * s, np.complex128)
    e = float(bits[(-n_warm) % n])
    b = -1j / (1.0 + 1j * (x0 + dx * e))
    for i in range(n_tot):
        bk = float(bits[(i - n_warm) % n])
        if tr_ui <= 0.0:
            e = bk
        rec_on = i >= n_warm
        base = (i - n_warm) * s
        for j in range(s):
            if rec_on:
                rec[base + j] = b
            e0 = e
            em = bk + (e0 - bk) * q
            e1 = bk + (em - bk) * q
            l0 = -kh * (1.0 + 1j * (x0 + dx * e0))
            lm = -kh * (1.0 + 1j * (x0 + dx * em))
            l1 = -kh * (1.0 + 1j * (x0 + dx * e1))
            b = _mode_step(b, beta, l0, lm, l1, h)
            e = e1
    su = 0.0
    for k in range(n * s):
        z = rec[k]
        su += z.real * z.real + z.imag * z.imag
    n1 = 0
    for k in range(n):
        n1 += int(bits[k])
    p1e = n1 / n
    l0s = _lor(x0)
    l1s = _lor(x1)
    out6[0] = su / (n * s)
    out6[1] = p1e * l1s + (1.0 - p1e) * l0s
    out6[3] = abs(l1s - l0s)
    nre = re_grid.shape[0]
    for r in range(nre):
        out7[1, r] = 4.0 * re_grid[r] * (1.0 - re_grid[r]) * abs(l1s - l0s)
    if n1 == 0 or n1 == n:
        out6[2] = 0.0
        for r in range(nre):
            out7[0, r] = 0.0
        return
    # conditional means over phases j in [0, 2s), sample k s + j labelled with bit k
    nph = 2 * s
    su_l = np.zeros((2, nph))
    si_l = np.zeros((2, nph))
    for k in range(n):
        lab = bits[k]
        for j in range(nph):
            z = rec[k * s + j]
            su_l[lab, j] += z.real * z.real + z.imag * z.imag
            si_l[lab, j] += z.imag
    c1 = float(n1)
    c0 = float(n - n1)
    dif = np.empty(nph)
    for j in range(nph):
        dif[j] = su_l[1, j] / c1 - su_l[0, j] / c0
    out6[2] = _oma_max(dif)
    for r in range(nre):
        re_ = re_grid[r]
        for j in range(nph):
            dif[j] = 4.0 * re_ * (si_l[1, j] / c1 - si_l[0, j] / c0) + 4.0 * re_ * re_ * (su_l[1, j] / c1
                                                                                            - su_l[0, j] / c0)
        out7[0, r] = _oma_max(dif)


@njit(cache=True, parallel=True)
def _build_group_nb(ax_xc, ax_w, ax_tr, ax_tau, s_tau, re_grid, bits, n_warm, out6, out7):
    """All (xc, w, tr, tau) nodes of one pattern; out6 (nxc, nw, ntr, ntau, 4), out7 (nxc, nw, ntr, ntau, 2, nre)."""
    nxc = ax_xc.shape[0]
    nw = ax_w.shape[0]
    ntr = ax_tr.shape[0]
    ntau = ax_tau.shape[0]
    total = nxc * nw * ntr * ntau
    for flat in prange(total):
        it = flat % ntau
        r1 = flat // ntau
        ir = r1 % ntr
        r2 = r1 // ntr
        iw = r2 % nw
        ix = r2 // nw
        xc = ax_xc[ix]
        w = ax_w[iw]
        if w <= 0.0:
            ll = _lor(xc)
            out6[ix, iw, ir, it, 0] = ll
            out6[ix, iw, ir, it, 1] = ll
            out6[ix, iw, ir, it, 2] = 0.0
            out6[ix, iw, ir, it, 3] = 0.0
            for f in range(2):
                for r in range(out7.shape[5]):
                    out7[ix, iw, ir, it, f, r] = 0.0
            continue
        _node_nb(xc - w, xc + w, bits, n_warm, s_tau[it], ax_tau[it], ax_tr[ir], re_grid,
                 out6[ix, iw, ir, it], out7[ix, iw, ir, it])


def normalized_node(x0: float, x1: float, bits, tau_ui: float, tr_ui: float, re_grid=(0.5,), *,
                    samples_per_bit: int | None = None, n_warm: int | None = None) -> dict:
    """Run one normalized node on an arbitrary cyclic pattern (e.g. a PRBS) without the table.

    Returns {"u_td", "u_st", "c_mean", "oma_drop_td", "oma_drop_st", "oma_thru_td" (n_re,), "oma_thru_st" (n_re,),
    "re", "p1", "td", "samples_per_bit"}; OMA in u units (drop) and P_in units (through)."""
    bits = np.ascontiguousarray(np.asarray(bits, dtype=np.uint8))
    s = samples_per_bit_for(tau_ui) if samples_per_bit is None else int(samples_per_bit)
    nw = min(bits.size, 32) if n_warm is None else int(n_warm)
    re_g = np.ascontiguousarray(np.asarray(re_grid, dtype=float))
    o6 = np.zeros(4)
    o7 = np.zeros((2, re_g.size))
    _node_nb(float(x0), float(x1), bits, nw, s, float(tau_ui), float(tr_ui), re_g, o6, o7)
    p1, td = pattern_stats(bits)
    return {"u_td": o6[0], "u_st": o6[1], "c_mean": o6[0] / o6[1], "oma_drop_td": o6[2], "oma_drop_st": o6[3],
            "oma_thru_td": o7[0].copy(), "oma_thru_st": o7[1].copy(), "re": re_g, "p1": p1, "td": td,
            "samples_per_bit": s}


def normalized_correction(x0, x1, bits, tau_ui, tr_ui, **kw) -> float:
    """Mean-energy correction c = <u>_time / <u>_static of one normalized node (drop, loss, heat)."""
    return normalized_node(x0, x1, bits, tau_ui, tr_ui, **kw)["c_mean"]


# ============================================================================================ lookup kernels
@njit(cache=True)
def _locate(ax, v):
    n = ax.shape[0]
    if v < ax[0]:
        return 0, 0.0, True
    if v > ax[n - 1]:
        return n - 2, 1.0, True
    i = np.searchsorted(ax, v, side="right") - 1
    if i > n - 2:
        i = n - 2
    if i < 0:
        i = 0
    return i, (v - ax[i]) / (ax[i + 1] - ax[i]), False


@njit(cache=True)
def lut_eval_nb(ax_xc, ax_w, ax_p1, ax_td, ax_tr, ax_tau, ax_re, tab6, tab7, x0, x1, p1, td, tr_ui, tau_ui, r_e,
                out):
    """Evaluate the LUT (module docstring) at static level detunings x0, x1 (= 2 Delta/kappa of NRZ levels 0 and 1),
    mark density p1 (of level 1), transition density td, rise time t_r/UI, photon lifetime tau_p/UI and
    r_e = kappa_e/kappa (floats). Writes out (N_OUT,) (OUT_* indices) and returns a status: 0 inside the validity
    domain, else ST_* bits (the query is then clamped to the domain; NaN outputs for a non-finite input).
    Allocation-free apart from two small work vectors; for the engine kernel (about 1-2 us per call).

    tab6 (N_F6, n6) and tab7 (N_F7, n6 n_re) are the C-ordered flattened tables over (xc, w, p1, td, tr, tau[, re])."""
    st = 0
    if not (math.isfinite(x0) and math.isfinite(x1) and math.isfinite(p1) and math.isfinite(td)
            and math.isfinite(tr_ui) and math.isfinite(tau_ui) and math.isfinite(r_e)):
        for k in range(out.shape[0]):
            out[k] = np.nan
        return ST_NONFINITE
    p = p1
    if x1 < x0:
        p = 1.0 - p
    xc = 0.5 * (x0 + x1)
    w = 0.5 * abs(x1 - x0)
    if xc < 0.0:
        xc = -xc
        p = 1.0 - p
    if td > 2.0 * min(p, 1.0 - p) + 1e-9 or td < 0.0:
        st |= ST_TD
    i0, f0, o0 = _locate(ax_xc, xc)
    i1, f1, o1 = _locate(ax_w, w)
    i2, f2, o2 = _locate(ax_p1, p)
    i3, f3, o3 = _locate(ax_td, td)
    i4, f4, o4 = _locate(ax_tr, tr_ui)
    i5, f5, o5 = _locate(ax_tau, tau_ui)
    i6, f6, o6 = _locate(ax_re, r_e)
    if o0:
        st |= ST_XC
    if o1:
        st |= ST_W
    if o2:
        st |= ST_P1
    if o3:
        st |= ST_TD
    if o4:
        st |= ST_TR
    if o5:
        st |= ST_TAU
    if o6:
        st |= ST_RE
    n1 = ax_w.shape[0]
    n2 = ax_p1.shape[0]
    n3 = ax_td.shape[0]
    n4 = ax_tr.shape[0]
    n5 = ax_tau.shape[0]
    n6 = ax_re.shape[0]
    s4 = n5
    s3 = n4 * s4
    s2 = n3 * s3
    s1 = n2 * s2
    s0 = n1 * s1
    re_lo = ax_re[i6]
    re_hi = ax_re[i6 + 1]
    g_lo = 4.0 * re_lo * (1.0 - re_lo)
    g_hi = 4.0 * re_hi * (1.0 - re_hi)
    u_td = 0.0
    u_st = 0.0
    od_td = 0.0
    od_st = 0.0
    ot_td = 0.0
    ot_st = 0.0
    for c in range(64):
        b0 = c & 1
        b1 = (c >> 1) & 1
        b2 = (c >> 2) & 1
        b3 = (c >> 3) & 1
        b4 = (c >> 4) & 1
        b5 = (c >> 5) & 1
        wt = ((f0 if b0 else 1.0 - f0) * (f1 if b1 else 1.0 - f1) * (f2 if b2 else 1.0 - f2)
              * (f3 if b3 else 1.0 - f3) * (f4 if b4 else 1.0 - f4) * (f5 if b5 else 1.0 - f5))
        if wt == 0.0:
            continue
        idx = (i0 + b0) * s0 + (i1 + b1) * s1 + (i2 + b2) * s2 + (i3 + b3) * s3 + (i4 + b4) * s4 + (i5 + b5)
        xcc = ax_xc[i0 + b0]
        wc = ax_w[i1 + b1]
        pc = ax_p1[i2 + b2]
        la = 1.0 / (1.0 + (xcc - wc) * (xcc - wc))
        lb = 1.0 / (1.0 + (xcc + wc) * (xcc + wc))
        dl = abs(lb - la)
        u_td += wt * tab6[F6_U_TD, idx]
        u_st += wt * (pc * lb + (1.0 - pc) * la)
        od_td += wt * tab6[F6_OMA_D_TD, idx]
        od_st += wt * dl
        j7 = idx * n6 + i6
        ot_td += wt * ((1.0 - f6) * tab7[F7_OMA_T_TD, j7] + f6 * tab7[F7_OMA_T_TD, j7 + 1])
        ot_st += wt * ((1.0 - f6) * g_lo + f6 * g_hi) * dl
    out[OUT_C_MEAN] = u_td / u_st
    out[OUT_D_MEAN_U] = u_td - u_st
    out[OUT_C_OMA_DROP] = od_td / od_st if od_st > 0.0 else 1.0
    out[OUT_D_OMA_DROP] = od_td - od_st
    out[OUT_C_OMA_THRU] = ot_td / ot_st if ot_st > 0.0 else 1.0
    out[OUT_D_OMA_THRU] = ot_td - ot_st
    return st


@njit(cache=True)
def lut_eval_strict_nb(ax_xc, ax_w, ax_p1, ax_td, ax_tr, ax_tau, ax_re, tab6, tab7, x0, x1, p1, td, tr_ui, tau_ui,
                       r_e, out):
    """:func:`lut_eval_nb` that raises ValueError outside the validity domain (07 P2-6 "raise an error outside
    it"); the Python wrapper WaveformLUT.eval names the offending keys."""
    st = lut_eval_nb(ax_xc, ax_w, ax_p1, ax_td, ax_tr, ax_tau, ax_re, tab6, tab7, x0, x1, p1, td, tr_ui, tau_ui,
                     r_e, out)
    if st != 0:
        raise ValueError("waveform LUT query outside its validity domain (07 P2-6)")
    return st


@njit(cache=True)
def lut_query_nb(phi0, phi1, k2_th, k2_dr, a0, a1, t_rt_s, bitrate_hz, rise_time_s, out):
    """LUT keys from the plant's optics inputs of one (ring, line) (mrm.plant.phase.ring_optics_row outputs):
    round-trip phases phi0/phi1 (rad) and amplitudes a0/a1 of the two NRZ levels, per-pass K^2 of the through and
    drop couplers, group delay t_rt (s), bit rate (1/s) and the 10-90 % rise time of the junction-charge edge (s).
    Rates by the logarithmic mapping (mrm.plant.optics.tcmt_rates); Delta_l = (phi_l - 2 pi m)/t_rt with m the order
    nearest the mean phase; normalization by the mean kappa of the two levels. Writes out (5,) = (x0, x1,
    t_r/UI, tau_p/UI, r_e)."""
    m = math.floor(0.5 * (phi0 + phi1) / (2.0 * math.pi) + 0.5)
    ke = -math.log1p(-k2_th) / t_rt_s
    kd = -math.log1p(-k2_dr) / t_rt_s
    ki0 = -2.0 * math.log(a0) / t_rt_s
    ki1 = -2.0 * math.log(a1) / t_rt_s
    kap = 0.5 * (ki0 + ki1) + ke + kd
    out[0] = 2.0 * (phi0 - 2.0 * math.pi * m) / t_rt_s / kap
    out[1] = 2.0 * (phi1 - 2.0 * math.pi * m) / t_rt_s / kap
    out[2] = rise_time_s * bitrate_hz
    out[3] = bitrate_hz / kap
    out[4] = ke / kap


# ============================================================================================ table object
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tool_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in ("numpy", "numba", "scipy"):
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def _config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class WaveformLUT:
    """The table (axes, tab6, tab7), its build config (and hash) and the validity domain; see the module docstring."""

    axes: dict
    tab6: np.ndarray
    tab7: np.ndarray
    config: dict
    config_hash: str
    validation: dict = field(default_factory=dict)
    path: str | None = None
    sha256: str | None = None

    def __post_init__(self):
        self.axes = {k: np.ascontiguousarray(np.asarray(v, dtype=float)) for k, v in self.axes.items()}
        for k in _AX6 + ("re",):
            a = self.axes[k]
            if a.ndim != 1 or a.size < 2 or np.any(np.diff(a) <= 0):
                raise ValueError(f"axis {k} must be strictly increasing with >= 2 nodes")
        n6 = int(np.prod([self.axes[k].size for k in _AX6]))
        self.tab6 = np.ascontiguousarray(np.asarray(self.tab6, dtype=np.float64).reshape(N_F6, n6))
        self.tab7 = np.ascontiguousarray(np.asarray(self.tab7, dtype=np.float64).reshape(N_F7,
                                                                                        n6 * self.axes["re"].size))

    @property
    def domain(self) -> dict:
        """Validity domain {key: [lo, hi]} (plus the feasibility td <= 2 min(p1, 1 - p1))."""
        a = self.axes
        return {"abs_xc": [0.0, float(a["xc"][-1])], "w": [0.0, float(a["w"][-1])],
                "p1": [float(a["p1"][0]), float(a["p1"][-1])], "td": [0.0, float(a["td"][-1])],
                "tr_ui": [float(a["tr"][0]), float(a["tr"][-1])], "tau_ui": [float(a["tau"][0]), float(a["tau"][-1])],
                "r_e": [float(a["re"][0]), float(a["re"][-1])]}

    def kernel_args(self) -> tuple:
        """(ax_xc, ax_w, ax_p1, ax_td, ax_tr, ax_tau, ax_re, tab6, tab7): the leading arguments of lut_eval_nb."""
        a = self.axes
        return (a["xc"], a["w"], a["p1"], a["td"], a["tr"], a["tau"], a["re"], self.tab6, self.tab7)

    def eval(self, x0, x1, p1, td, tr_ui, tau_ui, r_e=0.5, *, strict: bool = True) -> dict:
        """Python lookup; raises LutDomainError outside the domain (strict) naming the keys, else returns the
        clamped values with "status". Keys: c_mean, c_oma_thru, d_oma_thru, c_oma_drop, d_oma_drop, d_mean_u,
        status."""
        out = np.zeros(N_OUT)
        st = lut_eval_nb(*self.kernel_args(), float(x0), float(x1), float(p1), float(td), float(tr_ui),
                         float(tau_ui), float(r_e), out)
        if st and strict:
            bad = [name for bit, name in _ST_NAMES.items() if st & bit]
            raise LutDomainError(f"waveform LUT query outside the validity domain: {', '.join(bad)} "
                                 f"(query x0={x0}, x1={x1}, p1={p1}, td={td}, tr/UI={tr_ui}, tau_p/UI={tau_ui}, "
                                 f"r_e={r_e}; domain {self.domain}; LUT {self.config_hash[:12]})")
        keys = ("c_mean", "c_oma_thru", "d_oma_thru", "c_oma_drop", "d_oma_drop", "d_mean_u")
        d = {k: float(out[i]) for i, k in enumerate(keys)}
        d["status"] = int(st)
        return d

    # ---------------------------------------------------------------- artefact
    def save(self, name: str, out_dir: Path | None = None) -> tuple[Path, Path]:
        """Write waveform_lut_<name>.npz (float32 tables, compressed) and its sidecar waveform_lut_<name>.json."""
        d = Path(out_dir) if out_dir is not None else EXTRACTED_DIR
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"waveform_lut_{name}.npz"
        arrays = {f"ax_{k}": v for k, v in self.axes.items()}
        np.savez_compressed(path, tab6=self.tab6.astype(np.float32), tab7=self.tab7.astype(np.float32),
                            config_json=np.array(json.dumps(self.config, sort_keys=True)),
                            config_hash=np.array(self.config_hash), **arrays)
        side = {
            "schema": SCHEMA, "file": path.name, "sha256": _sha256(path),
            "description": "Normalized-coordinate data-modulation LUT (07 P2-6): time-domain corrections, relative to "
                           "the two-level static average, of the mean stored energy (drop, loss, heat; through by "
                           "energy conservation) and of the through and drop OMA, from the single-mode TCMT in "
                           "normalized time (mrm/waveform/lut.py docstring).",
            "fields6": ["u_td"], "fields6_extra": ["oma_drop_td"], "fields7": ["oma_thru_td"],
            "static_parts": "analytic: u_st = p L(x1) + (1 - p) L(x0), oma_drop_st = |L(x1) - L(x0)|, "
                            "oma_thru_st = 4 r_e (1 - r_e) oma_drop_st, L(x) = 1/(1 + x^2)",
            "storage": "float32, C order over (xc, w, p1, td, tr, tau[, re])",
            "axes": {k: v.tolist() for k, v in self.axes.items()},
            "axes_meaning": {"xc": "(x0 + x1)/2, detuning/HWHM at the level midpoint (x_c >= 0 stored; symmetry)",
                             "w": "|x1 - x0|/2 = resonance swing / FWHM", "p1": "mark density of the higher-x level",
                             "td": "transition density (cyclic)", "tr": "10-90 % rise of the junction-charge edge / UI",
                             "tau": "photon energy lifetime 1/kappa / UI", "re": "kappa_e/kappa"},
            "domain": self.domain, "config": self.config, "config_hash": self.config_hash,
            "validation": self.validation, "inputs": {"model": "normalized single-mode TCMT, g = 0, q = 0",
                                                      "code": "mrm/waveform/lut.py"},
            "tool_versions": _tool_versions(),
            "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        sp = d / f"waveform_lut_{name}.json"
        sp.write_text(json.dumps(side, indent=1, sort_keys=False))
        self.path, self.sha256 = str(path), side["sha256"]
        return path, sp


def load_lut(name_or_path: str | Path = "nrz", *, verify: bool = True) -> WaveformLUT:
    """Load waveform_lut_<name>.npz (or a path) and verify its sha256 against the sidecar (ARCHITECTURE section 5)
    and its config hash."""
    p = Path(name_or_path)
    if p.suffix != ".npz":
        p = EXTRACTED_DIR / f"waveform_lut_{name_or_path}.npz"
    side_path = p.with_suffix(".json")
    if not p.exists():
        raise FileNotFoundError(f"{p} not found (build it with python -m mrm.waveform.lut --name <name>)")
    if not side_path.exists():
        raise FileNotFoundError(f"sidecar {side_path.name} missing")
    side = json.loads(side_path.read_text())
    if side.get("schema") != SCHEMA:
        raise ValueError(f"{side_path.name}: schema {side.get('schema')!r} is not {SCHEMA}")
    digest = _sha256(p)
    if verify and side.get("sha256") != digest:
        raise ValueError(f"{p.name}: sha256 does not match its sidecar")
    with np.load(p, allow_pickle=False) as z:
        axes = {k[3:]: z[k] for k in z.files if k.startswith("ax_")}
        config = json.loads(str(z["config_json"]))
        lut = WaveformLUT(axes, z["tab6"], z["tab7"], config, str(z["config_hash"]),
                          validation=side.get("validation", {}), path=str(p), sha256=digest)
    if lut.config_hash != side.get("config_hash") or _config_hash(config) != lut.config_hash:
        raise ValueError(f"{p.name}: config hash mismatch")
    return lut


# ============================================================================================ builder
def build_lut(axes: dict | None = None, *, n_bits: int = 1024, n_warm: int = 32, base_samples: int = 32,
              seed: int = 2026, progress: bool = False) -> WaveformLUT:
    """Build the table on ``axes`` (default DEFAULT_AXES): one cyclic Markov pattern of n_bits per (p1, td) (seed
    seed + 1000 i_p1 + 100 i_td; the same pattern for every (xc, w, tr, tau) node so the table is smooth across them),
    nodes in parallel (numba prange). td = 0 and w = 0 nodes are exact without simulation (no transitions: the static
    levels; no swing: u = L(x_c), OMA 0). U_TD is stored as c_node x U_ST(nominal p1), so the analytic static of the
    lookup carries the node's own correction exactly."""
    import time

    ax = {k: np.asarray(v, dtype=float) for k, v in (DEFAULT_AXES if axes is None else axes).items()}
    nxc, nw, np1, ntd, ntr, ntau, nre = (ax[k].size for k in _AX6 + ("re",))
    s_tau = np.array([samples_per_bit_for(t, base_samples) for t in ax["tau"]], np.int64)
    t6 = np.zeros((nxc, nw, np1, ntd, ntr, ntau, N_F6))
    t7 = np.zeros((nxc, nw, np1, ntd, ntr, ntau, N_F7, nre))
    xc = ax["xc"][:, None]
    w = ax["w"][None, :]
    l0 = 1.0 / (1.0 + (xc - w) ** 2)
    l1 = 1.0 / (1.0 + (xc + w) ** 2)
    dl = np.abs(l1 - l0)
    gre = 4.0 * ax["re"] * (1.0 - ax["re"])
    patterns = {}
    t_start = time.time()
    for ip, p1 in enumerate(ax["p1"]):
        ust_nom = p1 * l1 + (1 - p1) * l0                   # (nxc, nw)
        for it, td in enumerate(ax["td"]):
            if td <= 0.0:
                t6[:, :, ip, it, :, :, F6_U_TD] = ust_nom[:, :, None, None]
                t6[:, :, ip, it, :, :, F6_OMA_D_TD] = dl[:, :, None, None]
                t7[:, :, ip, it, :, :, F7_OMA_T_TD, :] = dl[:, :, None, None, None] * gre
                continue
            bits = markov_bits(n_bits, p1, td, seed=seed + 1000 * ip + 100 * it)
            patterns[f"{p1:g}_{td:g}"] = list(pattern_stats(bits))
            o6 = np.zeros((nxc, nw, ntr, ntau, 4))
            o7 = np.zeros((nxc, nw, ntr, ntau, 2, nre))
            _build_group_nb(ax["xc"], ax["w"], ax["tr"], ax["tau"], s_tau, ax["re"], bits, int(n_warm), o6, o7)
            t6[:, :, ip, it, :, :, F6_U_TD] = o6[..., 0] / o6[..., 1] * ust_nom[:, :, None, None]
            t6[:, :, ip, it, :, :, F6_OMA_D_TD] = o6[..., 2]
            t7[:, :, ip, it, :, :, F7_OMA_T_TD, :] = o7[..., 0, :]
            if progress:
                print(f"waveform LUT: p1={p1:g} td={td:g} done ({time.time() - t_start:.1f} s)", flush=True)
    tab6 = np.moveaxis(t6, -1, 0).reshape(N_F6, -1)
    tab7 = np.moveaxis(t7, -2, 0).reshape(N_F7, -1)
    config = {"schema": SCHEMA, "axes": {k: v.tolist() for k, v in ax.items()}, "n_bits": int(n_bits),
              "n_warm": int(n_warm), "base_samples": int(base_samples), "samples_per_bit": s_tau.tolist(),
              "seed": int(seed), "pattern": "cyclic two-state Markov (driver.markov_bits, best of 64 seeds)",
              "edge": "single pole, tau_e = t_r/ln 9", "integrator": "exponential, Simpson exponent + first-order "
              "forcing correction (tcmt_time._mode_step)", "model": "normalized single-mode TCMT, g = 0, q = 0",
              "oma": "max over phases [0, 2 UI) of |P1 - P0|, parabola-refined", "storage": "float32"}
    lut = WaveformLUT(ax, tab6.astype(np.float32), tab7.astype(np.float32), config, _config_hash(config),
                      validation={"build_seconds": round(time.time() - t_start, 1), "pattern_stats": patterns})
    return lut


# ============================================================================================ validation
def p2_6_prbs_check(tau_ui: float = 0.15, *, order: int = 9, xc=None, w=None, tr=None,
                    limit: float = 0.03) -> dict:
    """07 P2-6 test quantity: |c_mean - 1| (the correction to the mean drop power) for PRBS-``order`` data at
    tau_p/UI = tau_ui (3 ps at a 20 ps UI is 0.15), computed directly (no table) over a grid of detuning x_c, swing
    w and edge rise time t_r/UI. Returns the maximum, where it occurs, the largest swing below which the correction
    stays under ``limit`` for every x_c and t_r, per-swing maxima and the full grid ("c_mean")."""
    from mrm.waveform.driver import prbs
    xc = np.asarray(DEFAULT_AXES["xc"] if xc is None else xc, float)
    w = np.asarray([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0] if w is None else w, float)
    tr = np.asarray(DEFAULT_AXES["tr"] if tr is None else tr, float)
    bits = prbs(order)
    c = np.zeros((xc.size, w.size, tr.size))
    for i, a in enumerate(xc):
        for j, b in enumerate(w):
            for k, t in enumerate(tr):
                c[i, j, k] = normalized_correction(a - b, a + b, bits, tau_ui, t, re_grid=(0.5,))
    dev = np.abs(c - 1.0)
    imax = np.unravel_index(int(np.argmax(dev)), dev.shape)
    ok_w = [float(b) for j, b in enumerate(w) if np.all(dev[:, : j + 1, :] < limit)]
    return {"tau_ui": float(tau_ui), "pattern": f"PRBS{order}", "max_abs_correction": float(dev.max()),
            "at": {"xc": float(xc[imax[0]]), "w": float(w[imax[1]]), "tr_ui": float(tr[imax[2]])},
            "max_swing_w_below_limit": max(ok_w) if ok_w else 0.0, "limit": float(limit),
            "max_abs_correction_by_w": {f"{b:g}": float(dev[:, j, :].max()) for j, b in enumerate(w)},
            "max_abs_correction_by_tr_at_w0p3": ({f"{t:g}": float(dev[:, list(w).index(0.3), k].max())
                                                  for k, t in enumerate(tr)} if 0.3 in list(w) else {}),
            "grid": {"xc": xc.tolist(), "w": w.tolist(), "tr": tr.tolist()}, "c_mean": c.tolist()}


def interpolation_check(lut: WaveformLUT, n_random: int = 200, seed: int = 7, n_bits: int = 1024,
                        practical: bool = False) -> dict:
    """Off-node accuracy: random queries inside the domain against direct normalized runs (fresh Markov patterns
    with the queried p1, td, so pattern sampling is included). Errors: |dc_mean| (absolute), the through-OMA
    difference |d(oma_td - oma_st)| relative to the largest static through OMA at that swing (4 r_e (1 - r_e)
    max_x |L(x + w) - L(x - w)|), and the OMA ratio error where the static OMA exceeds 25 % of that maximum.
    practical=True restricts to x_c 0.3-3, w 0.05-0.6, p1 = td = 0.5, t_r <= 0.5, tau 0.1-0.2, r_e 0.4-0.6."""
    rng = np.random.default_rng(seed)
    a = lut.axes
    dm, da, dr = [], [], []
    xs = np.linspace(0.0, 6.0, 1201)
    for _ in range(int(n_random)):
        if practical:
            xc, w, p1, td = rng.uniform(0.3, 3.0), rng.uniform(0.05, 0.6), 0.5, 0.5
            tr, tau, re_ = rng.uniform(0, 0.5), rng.uniform(0.1, 0.2), rng.uniform(0.4, 0.6)
        else:
            xc = rng.uniform(0, a["xc"][-1])
            w = rng.uniform(0.02, a["w"][-1])
            p1 = float(rng.choice([0.5, rng.uniform(a["p1"][0], a["p1"][-1])]))
            td = rng.uniform(0.05, min(a["td"][-1], 2 * min(p1, 1 - p1)))
            tr = rng.uniform(a["tr"][0], a["tr"][-1])
            tau = rng.uniform(a["tau"][0], a["tau"][-1])
            re_ = rng.uniform(a["re"][0], a["re"][-1])
        bits = markov_bits(n_bits, p1, td, seed=int(rng.integers(1 << 30)))
        d = normalized_node(xc - w, xc + w, bits, tau, tr, re_grid=(re_,))
        r = lut.eval(xc - w, xc + w, p1, td, tr, tau, re_)
        dm.append(abs(r["c_mean"] - d["c_mean"]))
        scale = 4 * re_ * (1 - re_) * np.max(np.abs(1 / (1 + (xs + w) ** 2) - 1 / (1 + (xs - w) ** 2)))
        da.append(abs(r["d_oma_thru"] - (d["oma_thru_td"][0] - d["oma_thru_st"][0])) / scale)
        if d["oma_thru_st"][0] > 0.25 * scale:
            dr.append(abs(r["c_oma_thru"] - d["oma_thru_td"][0] / d["oma_thru_st"][0]))
    dm, da, dr = np.array(dm), np.array(da), np.array(dr)
    q = lambda x, p: float(np.percentile(x, p)) if x.size else None  # noqa: E731
    return {"n": int(n_random), "practical": bool(practical),
            "c_mean_abs_err_max": float(dm.max()), "c_mean_abs_err_p95": q(dm, 95),
            "oma_thru_diff_err_rel_scale_max": float(da.max()), "oma_thru_diff_err_rel_scale_p95": q(da, 95),
            "c_oma_thru_abs_err_max": float(dr.max()) if dr.size else None, "c_oma_thru_abs_err_p95": q(dr, 95),
            "n_ratio": int(dr.size)}


def main(argv=None) -> int:
    """``python -m mrm.waveform.lut [--name nrz] [--quick]``: build, validate and save the artefact."""
    import argparse
    import time
    ap = argparse.ArgumentParser(description="build the waveform LUT artefact (07 P2-6)")
    ap.add_argument("--name", default=None, help="artefact name (default nrz; quick with --quick)")
    ap.add_argument("--quick", action="store_true", help="small axes (smoke test), not for the engine")
    ap.add_argument("--n-random", type=int, default=300)
    args = ap.parse_args(argv)
    name = args.name or ("quick" if args.quick else "nrz")
    if args.quick and name == "nrz":
        ap.error("--quick builds a smoke-test table: it must not replace the engine artefact waveform_lut_nrz")
    axes = None
    if args.quick:
        axes = {"xc": [0, 1, 2, 6], "w": [0, 0.3, 1.0], "p1": [0.3, 0.5, 0.7], "td": [0, 0.3, 0.6],
                "tr": [0, 0.7], "tau": [0.07, 0.25], "re": [0.3, 0.7]}
    t0 = time.time()
    lut = build_lut(axes, progress=True)
    print(f"built in {time.time() - t0:.1f} s", flush=True)
    t1 = time.time()
    # the tables as stored (float32) are what the engine sees: validate those
    lut = WaveformLUT(lut.axes, lut.tab6.astype(np.float32), lut.tab7.astype(np.float32), lut.config,
                      lut.config_hash, validation=lut.validation)
    lut.validation["p2_6_prbs9_tau_0p15"] = {k: v for k, v in p2_6_prbs_check(0.15).items() if k != "c_mean"}
    lut.validation["interpolation"] = interpolation_check(lut, n_random=args.n_random)
    lut.validation["interpolation_practical"] = interpolation_check(lut, n_random=args.n_random // 2, seed=8,
                                                                    practical=True)
    lut.validation["validation_seconds"] = round(time.time() - t1, 1)
    print(json.dumps({k: v for k, v in lut.validation.items() if k != "pattern_stats"}, indent=1), flush=True)
    p, s = lut.save(name)
    print(f"wrote {p} ({p.stat().st_size / 1e6:.2f} MB) and {s}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
