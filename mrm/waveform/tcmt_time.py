"""Time-domain TCMT of one add-drop ring with CW/CCW backscatter, driven by the junction voltage (waveform mode).

Owner: waveform. Contract: docs/ARCHITECTURE.md section 2 (``waveform/``: short time-domain TCMT + driver + junction
RC runs) and section 3 (photon lifetime ~3 ps, junction RC ~4 ps: waveform mode only); 07 P2-6. Physics:
docs/research/03_mrm_compact_models.md ("03") sections 2.1-2.5 (TCMT, conventions, steady state), 5.4 (loss
modulation), 5.6 (small-signal response) and 9.3 (fast layer, exponential integrator), including the [corrected]
notes and the 03 verification log rows 1, 2 and 6.

Conventions (03 section 2.1, identical to mrm.plant.optics)
----------------------------------------------------------
Time dependence e^{+i w t}; slowly varying envelopes in the frame of the laser at w_L. |a|^2 is stored energy (J),
|s|^2 is power (W). kappa_i, kappa_e, kappa_d are ENERGY decay rates (1/s); Delta = w_L - w_r (rad/s; Delta > 0: laser
blue of the resonance); g is the CW<->CCW coupling rate (rad/s, complex). The field equations (03 section 2.2, with
no add-port or backward inputs) are

    da1/dt = -(i Delta(t) + kappa(t)/2) a1 - i g a2 - i sqrt(kappa_e) s_in
    da2/dt = -(i Delta(t) + kappa(t)/2) a2 - i g* a1,          kappa(t) = kappa_i(t) + kappa_e + kappa_d

and the ports (03 section 2.3): s_thru = s_in - i sqrt(kappa_e) a1 (through), s_drop = -i sqrt(kappa_d) a1 (CW drop:
the monitor), s_refl = -i sqrt(kappa_e) a2 (back toward the laser), s_add = -i sqrt(kappa_d) a2 (CCW drop at the add
port). Loss power kappa_i (|a1|^2 + |a2|^2); heat eta kappa_i (|a1|^2 + |a2|^2) (07 P0-2). Only the resonance
order nearest the line is carried: at the reference ring single-order TCMT matches the exact ring to 0.005 dB in the
through port near resonance (03 section 0.1; 07 P2-4 limits in docs/handoff/plant_optics.md section 5).

The junction voltage enters only through the tables Delta(V) and kappa_i(V) (and eta(V)), evaluated at the
INSTANTANEOUS junction voltage V_j(t): the depletion edge follows V_j on the dielectric-relaxation time scale (fs),
so the index and the free-carrier loss are instantaneous functions of V_j (03 sections 5.2-5.5). The tables come from
the plant's own phase model (mrm.plant.phase.ring_optics_row: R1 thermal shift + junction dneff(V) + loss alpha(V, T))
in :meth:`RingTCMT.from_plant_params`, so the waveform and the quasi-static plant describe the same ring.

Integrator (03 section 9.3, extended to time-varying rates)
-----------------------------------------------------------
The coefficient matrix is M(t) = lambda(t) I - i G with lambda = -(i Delta + kappa/2) scalar and G = [[0, g], [g*, 0]]
CONSTANT, so all M(t) commute. In the eigenbasis of G (eigenvalues +-|g|, g = |g| e^{i theta})

    u_pm = (a1 +- e^{i theta} a2)/sqrt(2),   du_pm/dt = lambda_pm(t) u_pm + beta,   lambda_pm = lambda -+ i|g|,
    beta = -i sqrt(kappa_e) s_in / sqrt(2),   a1 = (u_+ + u_-)/sqrt(2),   a2 = e^{-i theta} (u_+ - u_-)/sqrt(2).

Each mode is a scalar linear ODE, solved per step h from the rates at t, t + h/2 and t + h (lambda_0, lambda_m,
lambda_1, from V_j at those instants) by

    Lambda = h (lambda_0 + 4 lambda_m + lambda_1)/6                 (Simpson: the exact exponent to O(h^5))
    u(t+h) = e^Lambda u(t) + beta [ (e^Lambda - 1)/(Lambda/h) + h^2 (lambda_1 - lambda_0) e^{Lambda/2} / 12 ]

The bracket is the forcing integral int_0^h exp(Lambda(h) - Lambda(s)) ds with exp(delta(s)) expanded to first
order in delta(s) = s Lambda(h)/h - Lambda(s) (zero at both ends) and that term integrated by Simpson (for a linear
lambda(t), delta(s) = (lambda_1 - lambda_0) s (h - s)/(2h) and the unweighted integral of delta is exactly
h^2 (lambda_1 - lambda_0)/12; the exponential weight makes the correction approximate, with a local error O(h^5)).
The step is EXACT for constant rates (steady states and the piecewise-constant inputs of an ideal NRZ edge at a sample
boundary come out to rounding, at any h). Against scipy DOP853 (rtol 1e-11 to 1e-12) with detuning swings of
1.2 kappa in 2 ps, a 37 GHz detuning tone, loss modulation and backscatter, the observed global error falls ~16x
per halving of h (fourth order): 3.6e-5, 2.4e-6, 1.5e-7, 9.5e-9 of max |a| at h = 1, 0.5, 0.25, 0.125 ps
(tests/verification/test_waveform.py). One complex exponential per mode and step.

Small-signal response (03 section 5.6, Sacher & Poon 2008, Karimelahi & Sheikholeslami 2016)
---------------------------------------------------------------------------------------------
Linearizing about the steady state a0 with a real input u(t) = dV(t): d(da)/dt = M da + f u, f = -(i Delta' +
kappa_i'/2) a0 (Delta' = dDelta/dV, kappa_i' = dkappa_i/dV). With F(s) = (sI - M)^{-1} f and a port power
P = P0 + 2 Re(K da_j) (through: K = conj(s_thru,0) (-i sqrt(kappa_e)), j = 1; drop: K = kappa_d conj(a1,0)), the
response to cos(w t) is

    H(i w) = K F_j(i w) + conj(K F_j(-i w))          (W per V of junction voltage)

For g = 0, kappa_i' = 0 and the input delta w_r = -delta Delta it reduces exactly to the closed form of 03 section
5.6 (:func:`h_opt_closed_form`): H/P_in = -(dT/dDelta) (1 + s/(kappa_i + kappa_d)) w_n^2/(s^2 + kappa s + w_n^2),
w_n^2 = kappa^2/4 + Delta^2: poles -kappa/2 +- i Delta (Sacher & Poon's "modulation resonance") and a zero at
-(kappa_i + kappa_d). Tested to 1e-10 against it, against every row of the 03 section 5.6 bandwidth/peaking table to
its printed digits, and to 3e-3 against a lock-in on the time-domain integrator (small-signal amplitude effects).

Kernels are numba @njit(cache=True) and run unchanged under NUMBA_DISABLE_JIT=1. Units SI throughout.
"""
from __future__ import annotations

import cmath
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from mrm.units import C0

__all__ = [
    "tcmt_steady_nb", "tcmt_run_nb", "tables_at_nb", "tcmt_run_tables_nb",
    "RingTCMT", "TcmtTrace", "port_powers", "optical_response_rates", "h_opt_closed_form",
]

_TWO_PI = 2.0 * math.pi


# ============================================================================================ kernels
@njit(cache=True)
def tcmt_steady_nb(delta, kappa_i, kappa_e, kappa_d, g, s_in):
    """Steady state (a1, a2) of 03 section 2.5 for constant rates (1/s, rad/s) and input amplitude s_in (sqrt(W)):
    a1 = -i sqrt(ke) D s_in/(D^2 + |g|^2), a2 = -sqrt(ke) g* s_in/(D^2 + |g|^2), D = kappa/2 + i Delta."""
    d = 0.5 * (kappa_i + kappa_e + kappa_d) + 1j * delta
    gg = g.real * g.real + g.imag * g.imag
    den = d * d + gg
    sq = math.sqrt(kappa_e)
    a1 = -1j * sq * d * s_in / den
    a2 = -sq * g.conjugate() * s_in / den
    return a1, a2


@njit(cache=True)
def _mode_step(u, beta, l0, lm, l1, h):
    """One exponential step of du/dt = lambda(t) u + beta (module docstring): exact for constant lambda."""
    lam_int = h * (l0 + 4.0 * lm + l1) / 6.0
    eh = cmath.exp(0.5 * lam_int)
    e = eh * eh
    lbar = lam_int / h
    forcing = (e - 1.0) / lbar + h * h * (l1 - l0) * eh / 12.0
    return e * u + beta * forcing


@njit(cache=True)
def tcmt_run_nb(h, s_in, kappa_e, kappa_d, g, delta_half, kappa_i_half, a1_0, a2_0, out_a1, out_a2):
    """Integrate the CW/CCW TCMT over K = out_a1.shape[0] - 1 steps of h (s).

    delta_half, kappa_i_half (2K+1,): Delta (rad/s) and kappa_i (1/s) at the half-step instants t0 + j h/2
    (even j: grid points, odd j: midpoints). kappa_e, kappa_d (1/s) and g (rad/s, complex) are constant; s_in
    (sqrt(W), complex) is the constant input amplitude. a1_0, a2_0: state at t0 (use :func:`tcmt_steady_nb` for
    a start in steady state). Writes out_a1, out_a2 (K+1,) complex (J^0.5) at the grid points.
    """
    kap_c = kappa_e + kappa_d
    ga = abs(g)
    if ga > 0.0:
        eth = g / ga                      # e^{i theta}
    else:
        eth = 1.0 + 0j
    beta = -1j * math.sqrt(kappa_e) * s_in / math.sqrt(2.0)
    r2 = 1.0 / math.sqrt(2.0)
    up = (a1_0 + eth * a2_0) * r2
    um = (a1_0 - eth * a2_0) * r2
    out_a1[0] = a1_0
    out_a2[0] = a2_0
    nk = out_a1.shape[0] - 1
    for k in range(nk):
        j = 2 * k
        l0 = -(1j * delta_half[j] + 0.5 * (kappa_i_half[j] + kap_c))
        lm = -(1j * delta_half[j + 1] + 0.5 * (kappa_i_half[j + 1] + kap_c))
        l1 = -(1j * delta_half[j + 2] + 0.5 * (kappa_i_half[j + 2] + kap_c))
        ig = 1j * ga
        up = _mode_step(up, beta, l0 - ig, lm - ig, l1 - ig, h)
        um = _mode_step(um, beta, l0 + ig, lm + ig, l1 + ig, h)
        out_a1[k + 1] = (up + um) * r2
        out_a2[k + 1] = (up - um) * r2 * eth.conjugate()


@njit(cache=True)
def tables_at_nb(v, grid, delta_tab, kappa_i_tab, out_delta, out_kappa_i):
    """Delta(V) (rad/s) and kappa_i(V) (1/s) at junction voltages v (V, reverse positive) by linear interpolation of
    the tables on the increasing grid (end values held outside it, as mrm.plant.electrical does)."""
    for j in range(v.shape[0]):
        out_delta[j] = np.interp(v[j], grid, delta_tab)
        out_kappa_i[j] = np.interp(v[j], grid, kappa_i_tab)


@njit(cache=True)
def tcmt_run_tables_nb(h, s_in, kappa_e, kappa_d, g, v_half, grid, delta_tab, kappa_i_tab, out_a1, out_a2,
                       work_delta, work_kappa_i):
    """Junction-voltage-driven run: v_half (2K+1,) is V_j at the half-step instants; the tables give Delta and
    kappa_i at the instantaneous V_j; the run starts in the steady state of v_half[0]. Work arrays (2K+1,)."""
    tables_at_nb(v_half, grid, delta_tab, kappa_i_tab, work_delta, work_kappa_i)
    a1, a2 = tcmt_steady_nb(work_delta[0], work_kappa_i[0], kappa_e, kappa_d, g, s_in)
    tcmt_run_nb(h, s_in, kappa_e, kappa_d, g, work_delta, work_kappa_i, a1, a2, out_a1, out_a2)


# ============================================================================================ outputs
def port_powers(a1, a2, s_in, kappa_i, kappa_e, kappa_d, eta=1.0) -> dict:
    """Port powers (W) from the mode amplitudes (03 section 2.3): through, drop (CW, monitor), refl (back to the
    laser), add (CCW drop), loss = kappa_i U and abs = eta kappa_i U (heat), with U = |a1|^2 + |a2|^2 (J).
    kappa_i and eta may be arrays aligned with a1 (time-varying)."""
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)
    u1 = np.abs(a1) ** 2
    u2 = np.abs(a2) ** 2
    loss = np.asarray(kappa_i) * (u1 + u2)
    return {
        "thru": np.abs(s_in - 1j * math.sqrt(kappa_e) * a1) ** 2,
        "drop": kappa_d * u1,
        "refl": kappa_e * u2,
        "add": kappa_d * u2,
        "loss": loss,
        "abs": np.asarray(eta) * loss,
        "u_cw": u1,
        "u_ccw": u2,
    }


def optical_response_rates(kappa_i, kappa_e, kappa_d, delta, f_hz, *, ddelta=1.0, dkappa_i=0.0, g=0j,
                           s_in=1.0, port="through"):
    """Small-signal optical response H(i 2 pi f) (module docstring) of a ring at constant rates.

    Input: a real perturbation u(t) with dDelta = ddelta u and dkappa_i = dkappa_i u (per unit of u; for
    resonance modulation delta w_r, ddelta = -1). Output: the port power change (W per unit u), complex, for
    port in {"through", "drop", "refl"}. f_hz: array of frequencies (Hz). Rates in 1/s, Delta and g in rad/s,
    s_in in sqrt(W). Exact linearization of the two-mode TCMT including backscatter and loss modulation."""
    f = np.atleast_1d(np.asarray(f_hz, dtype=float))
    lam = -(1j * delta + 0.5 * (kappa_i + kappa_e + kappa_d))
    gc = complex(g)
    a1, a2 = tcmt_steady_nb(float(delta), float(kappa_i), float(kappa_e), float(kappa_d), gc, complex(s_in))
    f1 = -(1j * ddelta + 0.5 * dkappa_i) * a1
    f2 = -(1j * ddelta + 0.5 * dkappa_i) * a2
    if port == "through":
        s0 = s_in - 1j * math.sqrt(kappa_e) * a1
        kk, j = np.conj(s0) * (-1j * math.sqrt(kappa_e)), 0
    elif port == "drop":
        kk, j = kappa_d * np.conj(a1), 0
    elif port == "refl":
        kk, j = kappa_e * np.conj(a2), 1
    else:
        raise ValueError("port must be 'through', 'drop' or 'refl'")
    gg = abs(gc) ** 2

    def resp(s):
        # (sI - M)^{-1} f with M = [[lam, -i g], [-i g*, lam]]: det = (s - lam)^2 + |g|^2
        sl = s - lam
        det = sl * sl + gg
        if j == 0:
            return (sl * f1 - 1j * gc * f2) / det
        return (-1j * gc.conjugate() * f1 + sl * f2) / det

    s = 1j * _TWO_PI * f
    return kk * resp(s) + np.conj(kk * resp(-s))


def h_opt_closed_form(s, kappa_i, kappa_e, kappa_d, delta):
    """03 section 5.6 closed form, per unit input power and per unit resonance modulation delta w_r (rad/s):

        H(s) = -(dT/dDelta) (1 + s/(kappa_i + kappa_d)) w_n^2/(s^2 + kappa s + w_n^2),
        w_n^2 = kappa^2/4 + Delta^2,  dT/dDelta = 2 Delta kappa_e (kappa_i + kappa_d)/w_n^4.

    Pure resonance modulation, no backscatter (Sacher & Poon 2008; Karimelahi & Sheikholeslami 2016; 03
    verification log row 1). s = i 2 pi f (complex array)."""
    s = np.asarray(s, dtype=complex)
    kap = kappa_i + kappa_e + kappa_d
    wn2 = 0.25 * kap * kap + delta * delta
    dtd = 2.0 * delta * kappa_e * (kappa_i + kappa_d) / (wn2 * wn2)
    return -dtd * (1.0 + s / (kappa_i + kappa_d)) * wn2 / (s * s + kap * s + wn2)


# ============================================================================================ ring model
def _table_slope(v: float, grid: np.ndarray, tab: np.ndarray) -> float:
    """Derivative of the piecewise-linear interpolant of tab at v (no cancellation: segment slopes); on a node the
    mean of the two adjacent segments; 0 outside the grid (end-held)."""
    if v < grid[0] or v > grid[-1]:
        return 0.0
    i = int(np.searchsorted(grid, v, side="right")) - 1
    i = min(max(i, 0), grid.size - 2)
    sl = (tab[i + 1] - tab[i]) / (grid[i + 1] - grid[i])
    tol = 1e-9 * (grid[-1] - grid[0])
    if abs(v - grid[i]) <= tol and i > 0:
        sl = 0.5 * (sl + (tab[i] - tab[i - 1]) / (grid[i] - grid[i - 1]))
    elif abs(v - grid[i + 1]) <= tol and i + 2 < grid.size:
        sl = 0.5 * (sl + (tab[i + 2] - tab[i + 1]) / (grid[i + 2] - grid[i + 1]))
    return float(sl)


@dataclass
class TcmtTrace:
    """Result of :meth:`RingTCMT.run`: grid times t_s (K+1,), mode amplitudes a1/a2 (J^0.5), the junction voltage
    v_j and the rates at the grid points, and the port powers (W) of :func:`port_powers`."""

    t_s: np.ndarray
    v_j: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    delta: np.ndarray
    kappa_i: np.ndarray
    powers: dict
    p_in_w: float


@dataclass
class RingTCMT:
    """One ring for waveform mode: constant coupling rates and the junction-voltage tables at one laser line.

    kappa_e, kappa_d: energy decay rates into the through and drop buses at the LINE (1/s; 07 P1-5).
    g: CW<->CCW coupling (rad/s, complex; 0 without backscatter). v_grid (G,) reverse junction voltage (V);
    delta_tab (G,) Delta(V) = w_L - w_r(V) (rad/s); kappa_i_tab (G,) intrinsic loss rate (1/s); eta_tab (G,) heat
    fraction. omega_l: laser angular frequency (rad/s, informational). meta: provenance of the tables.
    """

    kappa_e: float
    kappa_d: float
    g: complex
    v_grid: np.ndarray
    delta_tab: np.ndarray
    kappa_i_tab: np.ndarray
    eta_tab: np.ndarray
    omega_l: float = float("nan")
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.v_grid = np.ascontiguousarray(np.asarray(self.v_grid, dtype=float))
        g_ = self.v_grid
        if g_.ndim != 1 or g_.size < 2 or np.any(np.diff(g_) <= 0):
            raise ValueError("v_grid must be strictly increasing")
        for name in ("delta_tab", "kappa_i_tab", "eta_tab"):
            a = np.asarray(getattr(self, name), dtype=float)
            if a.ndim == 0:
                a = np.full(g_.shape, float(a))
            if a.shape != g_.shape or not np.all(np.isfinite(a)):
                raise ValueError(f"{name} must be finite with the shape of v_grid")
            setattr(self, name, np.ascontiguousarray(a))
        if np.any(self.kappa_i_tab < 0) or not (self.kappa_e > 0) or not (self.kappa_d >= 0):
            raise ValueError("need kappa_i >= 0, kappa_e > 0, kappa_d >= 0")
        self.kappa_e = float(self.kappa_e)
        self.kappa_d = float(self.kappa_d)
        self.g = complex(self.g)

    # ---------------------------------------------------------------- constructors
    @classmethod
    def from_rates(cls, kappa_i, kappa_e, kappa_d, delta_bias, *, v_bias=0.0, ddelta_dv=0.0, dkappa_i_dv=0.0,
                   g=0j, eta=1.0, v_span=(-0.5, 4.0), n_grid=451) -> "RingTCMT":
        """Linear tables: Delta(V) = delta_bias + ddelta_dv (V - v_bias), kappa_i(V) = kappa_i + dkappa_i_dv
        (V - v_bias) (floored at 0). For analytic checks; physical rings come from :meth:`from_plant_params`."""
        v = np.linspace(float(v_span[0]), float(v_span[1]), int(n_grid))
        d = delta_bias + ddelta_dv * (v - v_bias)
        ki = np.maximum(kappa_i + dkappa_i_dv * (v - v_bias), 0.0)
        return cls(kappa_e, kappa_d, g, v, d, ki, np.full(v.shape, float(eta)), meta={"kind": "linear"})

    @classmethod
    def from_plant_params(cls, pp, b: int = 0, ring: int = 0, *, v_bias: float, x_bias: float | None = None,
                          t_ring_k: float | None = None, t_heater_k: float | None = None, line: int | None = None,
                          lambda_line_m: float | None = None, backscatter: bool | None = None,
                          v_grid=None, order: int | None = None) -> "RingTCMT":
        """Tables from PlantParams through the plant's phase model (mrm.plant.phase.ring_optics_row).

        World b, ring ``ring``, at the line ``line`` (default pp.layout.ring_to_line[ring]) or the vacuum
        wavelength ``lambda_line_m``. The ring temperature is ``t_ring_k`` (K) or, with ``x_bias`` given (not
        both), the one that puts the laser at the normalized detuning x_bias = 2 Delta/kappa at the reverse bias
        v_bias (V) (the heater and self-heating state that holds the lock). The lock order is ``order`` (absolute
        longitudinal order) or by default: with x_bias, the order of the cold resonance (n_eff L = m lambda_res,
        the convention of mrm.control.models; a warning is raised when it needs a comb shift outside -0.5..1.5
        FSR, i.e. an implausible ring temperature, as in sampled worlds with a large lambda_res spread); with
        t_ring_k, the order nearest the line. t_heater_k
        defaults to the ring temperature (it enters only dln_alpha_dt_heater). On each voltage of v_grid (default:
        the junction table grid) the phase model gives phi, a, eta, K^2 and t_rt at the line, mapped to TCMT by
        mrm.plant.optics.tcmt_rates (logarithmic mapping, 03 section 2.6) with Delta = (phi - 2 pi m)/t_rt.
        kappa_e, kappa_d are taken at v_bias (they depend on V only through t_rt, by < 1e-4). Backscatter g =
        r_b,m/t_rt of the order m (pp.optics.backscatter_rb) when ``backscatter`` (default: pp.flags.backscatter).
        """
        from scipy.optimize import brentq

        from mrm.plant import phase as ph
        from mrm.plant.optics import tcmt_rates

        P = ph.PhaseParams.from_plant_params(pp)
        rp = P.ring[b]
        r = P.table_row(b)
        n_r = rp.shape[0]
        if lambda_line_m is None:
            li = int(pp.layout.ring_to_line[ring]) if line is None else int(line)
            lambda_line_m = C0 / float(np.asarray(pp.laser.line_freq_hz)[b, li])
        lam = np.array([float(lambda_line_m)])
        grid = P.grid if v_grid is None else np.asarray(v_grid, dtype=float)

        def optics_at(t_ring, v):
            o = {k: np.zeros((n_r, 1)) for k in ("p0", "p1", "a0", "a1", "e0", "e1", "kt", "kd", "trt")}
            sh, cj, rs = np.zeros((n_r, 2)), np.zeros((n_r, 2)), np.zeros((n_r, 2))
            fl = np.zeros(n_r, np.int8)
            tr = np.full(n_r, float(t_ring))
            th = np.full(n_r, float(t_ring if t_heater_k is None else t_heater_k))
            vv = np.full(n_r, float(v))
            ph.ring_optics_row(rp, lam, tr, th, vv, vv, P.grid, P.dneff[r], P.alpha_fca[r], P.capacitance[r],
                               P.r_series[r], o["p0"], o["p1"], o["a0"], o["a1"], o["e0"], o["e1"], o["kt"],
                               o["kd"], o["trt"], sh, cj, rs, fl)
            return (o["p0"][ring, 0], o["a0"][ring, 0], o["e0"][ring, 0], o["kt"][ring, 0], o["kd"][ring, 0],
                    o["trt"][ring, 0])

        if t_ring_k is not None and x_bias is not None:
            raise ValueError("give either t_ring_k or x_bias, not both")
        if t_ring_k is None:
            if x_bias is None:
                raise ValueError("give t_ring_k or x_bias")
            # the lock order is by default the order of the cold resonance (n_eff L = m lambda_res, R3; the
            # convention of mrm.control.models.resonance_detuning_m): phi = 2 pi m at the line once the comb is
            # shifted by lambda_line - lambda_res. ``order`` selects another one; order m_cold - k sits k cold FSRs
            # red of lambda_res. First guess: R1 inverted for the shift that brings the chosen order to the line.
            t_ref = rp[ring, ph.RP_T_REF]
            s0, beta, t0 = rp[ring, ph.RP_S0], rp[ring, ph.RP_BETA], rp[ring, ph.RP_T0]
            lam_r = rp[ring, ph.RP_LAM_RES]
            m_cold = int(round(rp[ring, ph.RP_N_EFF] * rp[ring, ph.RP_LEN] / lam_r))
            m_line = m_cold if order is None else int(order)
            fsr_m = lam_r * lam_r / (rp[ring, ph.RP_N_G] * rp[ring, ph.RP_LEN])
            need = float(lambda_line_m) - lam_r - (m_cold - m_line) * fsr_m
            if order is None and not -0.5 <= need / fsr_m <= 1.5:
                warnings.warn(f"RingTCMT.from_plant_params: the cold-resonance order {m_cold} of ring {ring} (world "
                              f"{b}) needs a comb shift of {need * 1e9:.2f} nm ({need / fsr_m:.2f} FSR) to reach the "
                              f"line: the solved ring temperature is far from T_ref = {t_ref:.1f} K. Pass order= "
                              f"(order m_cold - k sits k FSR red of the cold resonance) or t_ring_k to choose another "
                              f"lock order.", stacklevel=2)

            def t_for_shift(sh):
                # s0 [(T - Tref) + beta ((T - T0)^2 - (Tref - T0)^2)/2] = sh
                if beta == 0.0:
                    return t_ref + sh / s0
                a_ = 0.5 * beta
                b_ = 1.0 - beta * t0
                c_ = -(t_ref + 0.5 * beta * ((t_ref - t0) ** 2 - t0 * t0)) - sh / s0
                # a T^2 + b T + c = 0 written in T; take the root continuous with beta -> 0
                disc = b_ * b_ - 4 * a_ * c_
                if disc < 0.0:                 # no real root (extreme beta): the linear law as a first guess
                    return t_ref + sh / s0
                return (-b_ + math.sqrt(disc)) / (2 * a_)

            def f_detune(t_ring):
                phi, a_rt, _, kt, kd, trt = optics_at(t_ring, v_bias)
                ki, ke, kdd = tcmt_rates(kt, kd, a_rt, trt)
                dl = (phi - _TWO_PI * m_line) / trt
                return dl - 0.5 * x_bias * (ki + ke + kdd)

            t_guess = t_for_shift(need)
            lo, hi = t_guess - 30.0, t_guess + 30.0
            if f_detune(lo) * f_detune(hi) > 0:
                raise ValueError("could not bracket the ring temperature for x_bias (check the line and order)")
            t_ring_k = brentq(f_detune, lo, hi, xtol=1e-12, rtol=1e-14, maxiter=200)
        else:
            phi_b = optics_at(t_ring_k, v_bias)[0]
            m_line = int(round(phi_b / _TWO_PI)) if order is None else int(order)

        phi_b, a_b, eta_b, kt_b, kd_b, trt_b = optics_at(t_ring_k, v_bias)
        _, ke, kd = tcmt_rates(kt_b, kd_b, a_b, trt_b)
        d_tab = np.empty(grid.size)
        ki_tab = np.empty(grid.size)
        e_tab = np.empty(grid.size)
        for i, v in enumerate(grid):
            phi, a_rt, eta, kt, kdd, trt = optics_at(t_ring_k, v)
            ki_tab[i] = -2.0 * math.log(a_rt) / trt
            d_tab[i] = (phi - _TWO_PI * m_line) / trt
            e_tab[i] = eta
        use_bs = bool(pp.flags.backscatter) if backscatter is None else bool(backscatter)
        g = 0j
        if use_bs:
            rb = np.asarray(pp.optics.backscatter_rb)[b, ring]
            if rb.size:
                g = complex(rb[m_line % rb.size]) / trt_b
        meta = {"kind": "plant_params", "world": int(b), "ring": int(ring), "lambda_line_m": float(lambda_line_m),
                "t_ring_k": float(t_ring_k), "order": int(m_line), "v_bias": float(v_bias),
                "x_bias": None if x_bias is None else float(x_bias), "t_rt_s": float(trt_b),
                "junction_model": int(pp.junction.model), "config_hash": getattr(pp, "config_hash", None)}
        return cls(ke, kd, g, grid, d_tab, ki_tab, e_tab, omega_l=_TWO_PI * C0 / float(lambda_line_m), meta=meta)

    # ---------------------------------------------------------------- static quantities
    def delta(self, v):
        """Delta(V) = w_L - w_r(V) (rad/s)."""
        return np.interp(v, self.v_grid, self.delta_tab)

    def kappa_i(self, v):
        """Intrinsic loss rate kappa_i(V) (1/s)."""
        return np.interp(v, self.v_grid, self.kappa_i_tab)

    def eta(self, v):
        """Heat fraction eta(V)."""
        return np.interp(v, self.v_grid, self.eta_tab)

    def kappa(self, v):
        """Total energy decay rate kappa(V) (1/s)."""
        return self.kappa_i(v) + self.kappa_e + self.kappa_d

    def x(self, v):
        """Normalized detuning x = 2 Delta/kappa = detuning/HWHM at junction voltage v (03 section 2.1)."""
        return 2.0 * self.delta(v) / self.kappa(v)

    def fwhm_hz(self, v):
        """Linewidth kappa/(2 pi) (Hz)."""
        return self.kappa(v) / _TWO_PI

    def tau_p_s(self, v):
        """Photon (energy) lifetime 1/kappa (s); the field decays at kappa/2."""
        return 1.0 / self.kappa(v)

    def r_e(self, v):
        """kappa_e/kappa."""
        return self.kappa_e / self.kappa(v)

    def ddelta_dv(self, v):
        """dDelta/dV (rad/s/V): slope of the piecewise-linear table at v (mean of the two segments on a node)."""
        return _table_slope(float(v), self.v_grid, self.delta_tab)

    def dkappa_i_dv(self, v):
        """dkappa_i/dV (1/s/V): slope of the piecewise-linear table at v."""
        return _table_slope(float(v), self.v_grid, self.kappa_i_tab)

    def steady(self, v, p_in_w=1.0) -> dict:
        """Steady-state port powers (W) at constant junction voltage(s) v (V)."""
        v = np.atleast_1d(np.asarray(v, dtype=float))
        s_in = math.sqrt(p_in_w)
        a1 = np.empty(v.shape, complex)
        a2 = np.empty(v.shape, complex)
        for i, vi in enumerate(v):
            a1[i], a2[i] = tcmt_steady_nb(float(self.delta(vi)), float(self.kappa_i(vi)), self.kappa_e,
                                          self.kappa_d, self.g, complex(s_in))
        return port_powers(a1, a2, s_in, self.kappa_i(v), self.kappa_e, self.kappa_d, self.eta(v))

    # ---------------------------------------------------------------- dynamics
    def run(self, v_half, h_s, p_in_w=1.0, t0_s=0.0) -> TcmtTrace:
        """Integrate with the junction voltage v_half (2K+1,) given at the half-step instants t0 + j h/2 (V);
        starts in the steady state of v_half[0]. Returns a :class:`TcmtTrace` at the K+1 grid points."""
        v_half = np.ascontiguousarray(np.asarray(v_half, dtype=float))
        if v_half.ndim != 1 or v_half.size < 3 or v_half.size % 2 == 0:
            raise ValueError("v_half must have 2K+1 >= 3 samples")
        k = (v_half.size - 1) // 2
        a1 = np.empty(k + 1, complex)
        a2 = np.empty(k + 1, complex)
        wd = np.empty(v_half.size)
        wk = np.empty(v_half.size)
        s_in = complex(math.sqrt(p_in_w))
        tcmt_run_tables_nb(float(h_s), s_in, self.kappa_e, self.kappa_d, self.g, v_half, self.v_grid,
                           self.delta_tab, self.kappa_i_tab, a1, a2, wd, wk)
        vj = v_half[::2]
        pw = port_powers(a1, a2, s_in, wk[::2], self.kappa_e, self.kappa_d, self.eta(vj))
        return TcmtTrace(t0_s + h_s * np.arange(k + 1), vj, a1, a2, wd[::2], wk[::2], pw, float(p_in_w))

    def optical_response(self, v_bias, f_hz, *, port="through", p_in_w=1.0, loss_modulation=True):
        """Small-signal optical response at junction bias v_bias (V): port power change per volt of JUNCTION
        voltage (W/V), complex, at frequencies f_hz (module docstring). Includes the resonance shift dDelta/dV, the
        loss modulation dkappa_i/dV (unless loss_modulation=False) and the backscatter g."""
        return optical_response_rates(float(self.kappa_i(v_bias)), self.kappa_e, self.kappa_d,
                                      float(self.delta(v_bias)), f_hz, ddelta=float(self.ddelta_dv(v_bias)),
                                      dkappa_i=float(self.dkappa_i_dv(v_bias)) if loss_modulation else 0.0,
                                      g=self.g, s_in=math.sqrt(p_in_w), port=port)
