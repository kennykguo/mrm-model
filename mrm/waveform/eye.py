"""End-to-end waveform runs, eye diagrams (height, width, OMA, ER, levels) and the small-signal EO response.

Owner: waveform. Contract: 07 P2-6 (waveform mode outputs), D9 (relOMA = OMA/P_in), 4.3 CAL-10 and 4.4 VAL-3
(EO bandwidth replica ``eo_bandwidth_at_bias``); docs/research/03 sections 5.5, 5.6, 9.3.

Chain of one run (:func:`simulate`): bit pattern -> NRZ levels (mrm.plant.electrical.nrz_levels_row; bit 1 = the
higher reverse voltage) -> Gaussian-edged driver through R_drv into the junction network with C_j(V), R_s(V)
(driver.py) -> instantaneous Delta(V_j), kappa_i(V_j) -> CW/CCW TCMT (tcmt_time.py) -> port powers. The pattern is
cyclic: the run starts n_warm_bits before t = 0 in the DC state and the recorded window [0, n_bits T) is the periodic
steady state (the slowest time constant, a few ps, is far shorter than the warm-up).

Eye metrics (:func:`eye_metrics`; noise-free, so "inner" eye)
-------------------------------------------------------------
Samples are folded by bit: sample k S + j (k the bit, S samples per bit) is labelled with bit k, for phases
j in [0, 2S) (the optical response may lag the electrical bit by up to one UI). At each phase: means P1(j), P0(j) and
extremes over the ones and the zeros. Polarity: the sign of P1 - P0 at the phase where |P1 - P0| is largest.
OMA = the largest |P1(j) - P0(j)| over the phase (a smooth function of the phase: the maximum is refined between
samples by the vertex of the parabola through the three samples around it and converges quadratically in 1/S); the
levels P1, P0 (interpolated to that phase) and ER = 10 log10(max(P1, P0)/min(P1, P0)) dB are reported there, with
relOMA = OMA/P_in (07 D9). Eye height H(j) = min over ones - max over zeros (polarity-adjusted) at its own best phase
(a min/max over bits, kinked in the phase: first-order convergent in 1/S); eye width = the interval of phases around
that phase where every one stays above and every zero below the mid level (linear interpolation at the ends; capped
at one UI). The time-average power is reported as ``mean``. The same OMA definition is used by the LUT (lut.py).

Small-signal EO response (:func:`eo_response`, :func:`eo_bandwidth`; 03 section 5.6)
------------------------------------------------------------------------------------
H_EO(f) = H_E(f) H_opt(f), with H_E = V_j/V_src of the junction network (and the Gaussian driver edge when
include_driver) at the bias, and H_opt the exact TCMT linearization in W per V of junction voltage (resonance shift
dDelta/dV, loss modulation dkappa_i/dV, backscatter). The 3 dB bandwidth is the first frequency where
|H(f)/H(0)|^2 = 1/2 (the "electrical 3 dB point" of 03 section 5.6, i.e. -3 dB in a VNA S21 of the detected
signal); the peaking is max 20 log10 |H(f)/H(0)| over 0 < f < f_3dB.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from mrm.waveform.driver import DriverJunction, JunctionRun
from mrm.waveform.tcmt_time import RingTCMT, port_powers, tcmt_run_tables_nb

__all__ = ["WaveformResult", "EyeMetrics", "simulate", "eye_metrics", "eo_response", "eo_bandwidth",
           "bandwidth_3db"]


@dataclass
class EyeMetrics:
    """Eye of one port (module docstring). Powers in W (or in the unit of the waveform given), times in s."""

    p1: float
    p0: float
    oma: float
    er_db: float
    eye_height: float
    eye_width_s: float
    eye_width_ui: float
    phase_s: float
    mean: float
    polarity: int
    rel_oma: float = float("nan")


@dataclass
class WaveformResult:
    """One waveform run over the recorded window [0, n_bits T) (n_bits S samples, sample i at i T/S).

    v_src, v_j (V); powers: port powers (W) of tcmt_time.port_powers; static: the two-level static references
    {port: {"p1": P(V_hi), "p0": P(V_lo), "mean": p1_emp P(V_hi) + (1 - p1_emp) P(V_lo)}} with p1_emp the pattern's
    mark density; levels (V_lo, V_hi, flags); x_levels (x0, x1) = 2 Delta/kappa at the two static levels."""

    t_s: np.ndarray
    v_src: np.ndarray
    v_j: np.ndarray
    bits: np.ndarray
    t_bit_s: float
    samples_per_bit: int
    p_in_w: float
    powers: dict
    static: dict
    levels: tuple
    x_levels: tuple
    meta: dict = field(default_factory=dict)

    def mean(self, port: str) -> float:
        """Time-average power of a port over the period (W)."""
        return float(np.mean(self.powers[port]))

    def mean_correction(self, port: str) -> float:
        """Time-domain mean / two-level static mean (the 07 P2-6 correction factor)."""
        return self.mean(port) / self.static[port]["mean"]

    def eye(self, port: str = "thru") -> EyeMetrics:
        """Eye metrics of a port (:func:`eye_metrics`)."""
        em = eye_metrics(self.powers[port], self.bits, self.samples_per_bit, self.t_bit_s)
        em.rel_oma = em.oma / self.p_in_w
        return em


def simulate(ring: RingTCMT, drv: DriverJunction, bits, bitrate_hz: float, *, v_bias: float, swing_vpp: float,
             level_mode: int = 0, samples_per_bit: int = 64, p_in_w: float = 1e-3,
             n_warm_bits: int | None = None, junction: JunctionRun | None = None) -> WaveformResult:
    """Run the driver + junction + TCMT chain on the cyclic pattern ``bits`` at ``bitrate_hz`` (module docstring).

    v_bias (V reverse) and swing_vpp (V) give the NRZ levels (level_mode 0 centered, 1 one-sided); p_in_w is the
    power at the ring's input port (W; the TCMT is linear in it). samples_per_bit S sets the optical step T/S and
    the half-step electrical grid. A precomputed ``junction`` run (same pattern and grid) may be passed."""
    bits = np.ascontiguousarray(np.asarray(bits, dtype=np.uint8))
    t_bit = 1.0 / float(bitrate_hz)
    v_lo, v_hi, flags = drv.levels(v_bias, swing_vpp, level_mode)
    if junction is not None:
        # the static references below use (v_lo, v_hi) of this call: the run must be of the same pattern and drive
        jl = getattr(junction, "levels", None)
        if (junction.bits.shape != bits.shape or np.any(junction.bits != bits)
                or not math.isclose(junction.t_bit_s, t_bit, rel_tol=1e-12)
                or (jl is not None and not np.allclose(jl, (v_lo, v_hi), rtol=0.0, atol=1e-12))):
            raise ValueError("the junction run was computed for another pattern, bit rate or NRZ levels")
    jr = junction if junction is not None else drv.run(bits, t_bit, samples_per_bit, v_lo, v_hi,
                                                       n_warm_bits=n_warm_bits)
    s = jr.samples_per_bit
    n = bits.size
    h = t_bit / s
    kk = (jr.v_j.size - 1) // 2
    a1 = np.empty(kk + 1, complex)
    a2 = np.empty(kk + 1, complex)
    wd = np.empty(jr.v_j.size)
    wk = np.empty(jr.v_j.size)
    s_in = complex(math.sqrt(p_in_w))
    tcmt_run_tables_nb(h, s_in, ring.kappa_e, ring.kappa_d, ring.g, np.ascontiguousarray(jr.v_j), ring.v_grid,
                       ring.delta_tab, ring.kappa_i_tab, a1, a2, wd, wk)
    i0 = jr.n_warm_bits * s
    sl = slice(i0, i0 + n * s)
    vj = jr.v_j[::2][sl]
    pw = port_powers(a1[sl], a2[sl], s_in, wk[::2][sl], ring.kappa_e, ring.kappa_d, ring.eta(vj))
    p1 = float(bits.mean())
    st_hi = ring.steady(v_hi, p_in_w)
    st_lo = ring.steady(v_lo, p_in_w)
    static = {k: {"p1": float(st_hi[k][0]), "p0": float(st_lo[k][0]),
                  "mean": p1 * float(st_hi[k][0]) + (1.0 - p1) * float(st_lo[k][0])} for k in st_hi}
    x_lv = (float(ring.x(v_lo)), float(ring.x(v_hi)))
    return WaveformResult(jr.t_s[::2][sl], jr.v_src[::2][sl], vj, bits, t_bit, s, float(p_in_w), pw, static,
                          (v_lo, v_hi, flags), x_lv,
                          meta={"rise_time_s": drv.rise_time_s, "r_drv_ohm": drv.r_drv_ohm, "c_pad_f": drv.c_pad_f,
                                "n_warm_bits": jr.n_warm_bits, "v_bias": float(v_bias),
                                "swing_vpp": float(swing_vpp), "level_mode": int(level_mode)})


def eye_metrics(p, bits, samples_per_bit: int, t_bit_s: float = 1.0) -> EyeMetrics:
    """Noise-free eye of the cyclic waveform p (n_bits S samples, sample k S + j in bit k) (module docstring)."""
    p = np.asarray(p, dtype=float)
    bits = np.asarray(bits).astype(bool)
    s = int(samples_per_bit)
    n = bits.size
    if p.size != n * s:
        raise ValueError("p must have n_bits * samples_per_bit samples")
    if bits.all() or not bits.any():
        raise ValueError("the pattern needs both ones and zeros")
    ph = np.arange(2 * s)
    idx = (np.arange(n)[:, None] * s + ph[None, :]) % (n * s)      # (n, 2S)
    w = p[idx]
    ones, zeros = w[bits], w[~bits]
    m1, m0 = ones.mean(axis=0), zeros.mean(axis=0)
    j_pol = int(np.argmax(np.abs(m1 - m0)))
    pol = 1 if m1[j_pol] >= m0[j_pol] else -1
    if pol > 0:
        height = ones.min(axis=0) - zeros.max(axis=0)
    else:
        height = zeros.min(axis=0) - ones.max(axis=0)
    jh, dh, hv = _best_phase(np.ascontiguousarray(height, dtype=float))
    jo, do, _ = _best_phase(np.ascontiguousarray(np.abs(m1 - m0), dtype=float))
    js = jo + do
    p1 = float(_at_phase(m1, jo, do))
    p0 = float(_at_phase(m0, jo, do))
    thr = 0.5 * (float(_at_phase(m1, jh, dh)) + float(_at_phase(m0, jh, dh)))
    if pol > 0:
        margin = np.minimum(ones.min(axis=0) - thr, thr - zeros.max(axis=0))
    else:
        margin = np.minimum(zeros.min(axis=0) - thr, thr - ones.max(axis=0))
    if margin[jh] <= 0:
        width = 0.0
    else:
        lo = jh
        while lo > 0 and margin[lo - 1] > 0:
            lo -= 1
        hi = jh
        while hi < 2 * s - 1 and margin[hi + 1] > 0:
            hi += 1
        left = lo - (margin[lo] / (margin[lo] - margin[lo - 1]) if lo > 0 else 0.0)
        right = hi + (margin[hi] / (margin[hi] - margin[hi + 1]) if hi < 2 * s - 1 else 0.0)
        width = min(float(right - left) / s, 1.0)
    hi_l, lo_l = max(p1, p0), min(p1, p0)
    er = 10.0 * math.log10(hi_l / lo_l) if lo_l > 0 else float("inf")
    return EyeMetrics(p1, p0, abs(p1 - p0), er, float(hv), width * t_bit_s, width, js * t_bit_s / s,
                      float(p.mean()), pol)


@njit(cache=True)
def _best_phase(hgt):
    """(j_b, d, h_v): discrete maximum j_b of hgt over the phases, the offset d in (-1/2, 1/2) (samples) of the
    vertex of the parabola through hgt[j_b - 1 .. j_b + 1], and the parabola's value there. d = 0 and h_v = hgt[j_b]
    when the maximum is at the end of the window, the parabola does not open downward or the vertex leaves the
    half-sample interval."""
    nph = hgt.shape[0]
    jb = 0
    for j in range(1, nph):
        if hgt[j] > hgt[jb]:
            jb = j
    h0 = hgt[jb]
    d = 0.0
    hv = h0
    if 0 < jb < nph - 1:
        hm = hgt[jb - 1]
        hp = hgt[jb + 1]
        den = hm - 2.0 * h0 + hp
        if den < 0.0:
            dd = 0.5 * (hm - hp) / den
            if -0.5 < dd < 0.5:
                d = dd
                hv = h0 - 0.125 * (hm - hp) * (hm - hp) / den
    return jb, d, hv


@njit(cache=True)
def _at_phase(y, jb, d):
    """y at the fractional phase jb + d by the quadratic through y[jb - 1 .. jb + 1] (|d| < 1/2), so that the
    difference of two such interpolants at the vertex of :func:`_best_phase` equals the vertex value."""
    if d == 0.0:
        return y[jb]
    ym = y[jb - 1]
    y0 = y[jb]
    yp = y[jb + 1]
    return y0 + 0.5 * d * (yp - ym) + 0.5 * d * d * (yp - 2.0 * y0 + ym)


# ============================================================================================ small-signal EO
def eo_response(ring: RingTCMT, drv: DriverJunction | None, v_bias: float, f_hz, *, port: str = "through",
                p_in_w: float = 1e-3, include_driver: bool = False, loss_modulation: bool = True):
    """Small-signal EO response (W per V of driver source voltage; per V of junction voltage when drv is None) at
    the reverse bias v_bias, frequencies f_hz (module docstring)."""
    h = ring.optical_response(v_bias, f_hz, port=port, p_in_w=p_in_w, loss_modulation=loss_modulation)
    if drv is not None:
        h = h * drv.h_electrical(v_bias, f_hz, include_driver=include_driver)
    return h


def bandwidth_3db(f_hz, h) -> tuple[float, float]:
    """(f_3dB, peaking dB) from a sampled response h(f) on an increasing grid starting near DC: f_3dB is the first
    crossing of |h/h[0]|^2 = 1/2 (linear interpolation in |h|^2), peaking = max 20 log10 |h/h[0]| before it.
    Returns (nan, nan) when h[0] = 0 and inf for f_3dB if the response never drops below 1/2."""
    f = np.asarray(f_hz, dtype=float)
    a = np.abs(np.asarray(h)) ** 2
    if a[0] == 0:
        return float("nan"), float("nan")
    r = a / a[0]
    below = np.nonzero(r < 0.5)[0]
    if below.size == 0:
        return float("inf"), float(10 * np.log10(r.max()))
    i = int(below[0])
    f3 = f[i - 1] + (0.5 - r[i - 1]) * (f[i] - f[i - 1]) / (r[i] - r[i - 1])
    return float(f3), float(10 * np.log10(max(r[:i].max(), 1.0)))


def eo_bandwidth(ring: RingTCMT, drv: DriverJunction | None, v_bias: float, *, port: str = "through",
                 include_driver: bool = False, loss_modulation: bool = True, f_max_hz: float | None = None,
                 n: int = 4001) -> dict:
    """3 dB bandwidth and peaking of the small-signal EO response at v_bias (module docstring), refined by brentq.

    Returns {"f3db_hz", "peaking_db", "f_hz", "h_rel" (complex h/h(f->0)), "x_bias", "fwhm_hz", "rc_pole_hz"};
    f3db_hz and peaking_db are nan (h_rel nan) when the DC response vanishes (x = 0 without loss modulation)."""
    from scipy.optimize import brentq

    fwhm = float(ring.fwhm_hz(v_bias))
    if f_max_hz is None:
        f_max_hz = 8.0 * fwhm
    f = np.linspace(f_max_hz / n * 1e-3, f_max_hz, n)

    def resp(ff):
        return eo_response(ring, drv, v_bias, ff, port=port, include_driver=include_driver,
                           loss_modulation=loss_modulation)

    h0 = resp(np.array([1e3]))[0]
    # natural scale of the response (W/V at the 1 mW of eo_response): P_in |dDelta/dV, dkappa_i/dV| / kappa. A DC
    # gain below 1e-9 of it (laser on resonance without loss modulation: pure index modulation gives no first-order
    # intensity modulation, 03 section 5.6) leaves no 3 dB point to define.
    scale = 1e-3 * (abs(ring.ddelta_dv(v_bias)) + (abs(ring.dkappa_i_dv(v_bias)) if loss_modulation else 0.0)) \
        / float(ring.kappa(v_bias))
    if not abs(h0) > 1e-9 * scale:
        nan = float("nan")
        return {"f3db_hz": nan, "peaking_db": nan, "f_hz": f, "h_rel": np.full(f.shape, complex(nan, nan)),
                "x_bias": float(ring.x(v_bias)), "fwhm_hz": fwhm,
                "rc_pole_hz": drv.rc_pole_hz(v_bias) if drv is not None else float("inf")}
    hr = resp(f) / h0
    f3, pk = bandwidth_3db(np.concatenate([[1e3], f]), np.concatenate([[1.0 + 0j], hr]))
    if np.isfinite(f3):
        g = lambda x: abs(resp(np.array([x]))[0] / h0) ** 2 - 0.5  # noqa: E731
        i = int(np.searchsorted(f, f3))
        lo = f[max(i - 1, 0)] if i > 0 else 1e3
        hi = f[min(i, n - 1)]
        if g(lo) > 0 > g(hi):
            f3 = float(brentq(g, lo, hi, xtol=1e-3 * fwhm * 1e-6, rtol=1e-12))
    return {"f3db_hz": f3, "peaking_db": pk, "f_hz": f, "h_rel": hr, "x_bias": float(ring.x(v_bias)),
            "fwhm_hz": fwhm, "rc_pole_hz": drv.rc_pole_hz(v_bias) if drv is not None else float("inf")}
