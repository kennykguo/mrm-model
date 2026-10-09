"""Bit patterns, the NRZ driver and the junction RC network for waveform mode.

Owner: waveform. Contract: 07 P1-7 (junction C(V), R_s), P1-8 (driver output resistance), P2-6; docs/research/03
section 5.5 (drive network, charge-based form) and section 5.6 (H_E(s)).

Patterns
--------
``prbs(order, n_bits)``: the ITU-T O.150 sequences x^k + x^t + 1 (k/t = 7/6, 9/5, 15/14, 23/18, 31/28), recurrence
a_m = a_{m-t} XOR a_{m-k} from the all-ones seed: the same sequence as mrm.plant.datapath.prbs_bits (tested), so
the waveform and the data-path statistics use one PRBS. ``markov_bits``: a cyclic two-state Markov pattern with a given
mark density p1 and transition density td (P(1->0) = td/(2 p1), P(0->1) = td/(2 (1 - p1)); needs td <= 2 min(p1,
1 - p1)), the canonical pattern of the normalized LUT (lut.py). Transition density is counted cyclically.

Driver (assumed form; no G25/G26 driver waveform is published)
---------------------------------------------------------------
An ideal NRZ source between the junction reverse voltages V_lo (bit 0) and V_hi (bit 1) (mrm.plant.electrical
.nrz_levels_row; level 1 = the higher reverse voltage, integrator decision 6), with Gaussian edges: the ideal NRZ
filtered by a Gaussian of standard deviation sigma, which is exact superposition of erf edges,

    V_src(t) = V_lo + (V_hi - V_lo) sum_k b_k [Phi((t - k T)/sigma) - Phi((t - (k+1) T)/sigma)],

with 10-90 % rise time t_r = 2 * 1.28155 sigma = 2.5631 sigma (Phi the standard normal CDF) and transfer function
exp(-(2 pi f sigma)^2 / 2). The source drives the junction through the driver output resistance R_drv
(driver.output_resistance_ohm, 07 P1-8).

Junction network (03 section 5.5, charge form; C_j(V) differential capacitance, R_s(V) from the junction table)
-----------------------------------------------------------------------------------------------------------
    one node (C_pad = 0):   C_j(V_j) dV_j/dt = (V_src - V_j)/(R_drv + R_s(V_j))
    two nodes (C_pad > 0):  C_pad dV_p/dt = (V_src - V_p)/R_drv - (V_p - V_j)/R_s(V_j)
                            C_j(V_j) dV_j/dt = (V_p - V_j)/R_s(V_j)
C_j dV/dt is dQ_j/dt for the differential capacitance C_j = dQ_j/dV, so the nonlinear C(V) is handled exactly
(the integration variable is V_j; the charge follows). C_j and R_s are ring-level values (F, ohm) from
mrm.plant.electrical.junction_eval_row (fill f, length L: C_j = f L c'(V), R_s = r'(V)/(f L)). Integrated by RK4 with
substeps of at most 0.1 of the fastest RC time constant and, for Gaussian edges, of sigma/2: a linear step response
matches the exponential to 1e-7 V, and the nonlinear two-node network with Gaussian edges matches scipy DOP853 to
2e-6 V on a 1.3 V swing (tests/verification/test_waveform.py).
Small signal: V_j/V_src = 1/(1 + s (R_drv C_pad + (R_drv + R_s) C_j) + s^2 R_drv R_s C_pad C_j) (C_pad = 0: the
single pole 1/(1 + s (R_drv + R_s) C_j) of 03 section 5.5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit

__all__ = [
    "PRBS_TAPS", "prbs", "markov_bits", "pattern_stats", "rise_to_sigma", "sigma_to_rise",
    "nrz_source_nb", "junction_rc_run_nb", "DriverJunction", "JunctionRun", "MAX_HALF_STEPS",
]

#: ITU-T O.150 PRBS polynomials x^k + x^t + 1: k -> t (same table as mrm.plant.datapath.PRBS_TAPS)
PRBS_TAPS = {7: 6, 9: 5, 15: 14, 23: 18, 31: 28}
_ERF_10_90 = 2.0 * 1.2815515655446004   # 10-90 % width of the standard normal CDF, in sigma
#: Largest half-step grid of one run (DriverJunction.run, hence simulate): about 100 B per half step through simulate,
#: so 2.5e7 is about 2.5 GB on the shared 16 GB machine (PRBS15 at 64 samples/bit is 4.2e6; a full PRBS23 period is
#: not practical: use a segment).
MAX_HALF_STEPS = 25_000_000


# ============================================================================================ patterns
@njit(cache=True)
def _prbs_nb(k, t, n):
    a = np.empty(n + k, np.uint8)
    for i in range(k):
        a[i] = 1
    for m in range(k, n + k):
        a[m] = a[m - t] ^ a[m - k]
    return a[:n].copy()


def prbs(order: int, n_bits: int | None = None) -> np.ndarray:
    """PRBS-``order`` bits (uint8), a_m = a_{m-t} XOR a_{m-k} from the all-ones seed (ITU-T O.150 taps).

    n_bits defaults to one period 2^k - 1 (allowed for k <= 23); PRBS31 needs an explicit n_bits (a segment)."""
    if order not in PRBS_TAPS:
        raise ValueError(f"unknown PRBS order {order}; known {sorted(PRBS_TAPS)}")
    period = (1 << order) - 1
    if n_bits is None:
        if order > 23:
            raise ValueError("give n_bits for PRBS31 (period 2^31 - 1 bits)")
        n_bits = period
    if n_bits < 1:
        raise ValueError("n_bits must be >= 1")
    return _prbs_nb(order, PRBS_TAPS[order], int(n_bits))


@njit(cache=True)
def _markov_chain_nb(u, p1, p10, p01):
    n = u.shape[0]
    b = np.empty(n, np.uint8)
    b[0] = 1 if u[0] < p1 else 0
    for i in range(1, n):
        if b[i - 1]:
            b[i] = 0 if u[i] < p10 else 1
        else:
            b[i] = 1 if u[i] < p01 else 0
    return b


def pattern_stats(bits) -> tuple[float, float]:
    """(mark density, cyclic transition density) of a bit pattern."""
    b = np.asarray(bits, dtype=np.int64)
    if b.size == 0:
        raise ValueError("empty pattern")
    return float(b.mean()), float(np.mean(b != np.roll(b, 1)))


def markov_bits(n_bits: int, mark_density: float, transition_density: float, seed: int = 0,
                n_tries: int = 64) -> np.ndarray:
    """Cyclic two-state Markov pattern with mark density p1 and transition density td (module docstring).

    Of ``n_tries`` realizations (seeds seed .. seed + n_tries - 1) the one whose empirical (p1, td) is closest to
    the request is returned, so the finite pattern carries the requested statistics to about 1/n_bits."""
    p1 = float(mark_density)
    td = float(transition_density)
    if not 0.0 < p1 < 1.0:
        raise ValueError("mark_density must be in (0, 1)")
    if not 0.0 < td <= 2.0 * min(p1, 1.0 - p1) + 1e-12:
        raise ValueError("transition_density must be in (0, 2 min(p1, 1 - p1)] (a cyclic pattern without "
                         "transitions is constant, so its mark density is 0 or 1)")
    p10 = td / (2.0 * p1)
    p01 = td / (2.0 * (1.0 - p1))
    best, best_err = None, np.inf
    for s in range(int(seed), int(seed) + int(n_tries)):
        u = np.random.default_rng(s).random(int(n_bits))
        b = _markov_chain_nb(u, p1, p10, p01)
        m, t = pattern_stats(b)
        err = abs(m - p1) + abs(t - td)
        if err < best_err:
            best, best_err = b, err
    return best


def rise_to_sigma(rise_time_10_90_s: float) -> float:
    """Gaussian edge: sigma (s) from the 10-90 % rise time (s)."""
    return float(rise_time_10_90_s) / _ERF_10_90


def sigma_to_rise(sigma_s: float) -> float:
    """Gaussian edge: 10-90 % rise time (s) from sigma (s)."""
    return float(sigma_s) * _ERF_10_90


# ============================================================================================ driver source
@njit(cache=True)
def _phi(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


@njit(cache=True)
def nrz_source_nb(t, bits, t_bit, sigma, v0, v1):
    """Gaussian-edged NRZ source voltage (V) at time t (s); bit k (cyclic over the pattern) occupies
    [k t_bit, (k+1) t_bit) before filtering; sigma = 0 gives the ideal NRZ (right-continuous)."""
    n = bits.shape[0]
    j = int(math.floor(t / t_bit))
    if sigma <= 0.0:
        return v0 + (v1 - v0) * bits[j % n]
    # summation by parts: sum_k b_k [Phi_k - Phi_{k+1}] = b_ka + sum_{k = ka+1}^{kb} (b_k - b_{k-1}) Phi_k, exact to
    # Phi(-7) ~ 1e-12 when Phi_ka ~ 1 and Phi_{kb+1} ~ 0; only the edges inside the window cost an erfc
    nw = int(math.ceil(7.0 * sigma / t_bit)) + 1
    ka = j - nw
    lvl = float(bits[ka % n])
    prev = bits[ka % n]
    for k in range(ka + 1, j + nw + 1):
        bk = bits[k % n]
        if bk != prev:
            lvl += (float(bk) - float(prev)) * _phi((t - k * t_bit) / sigma)
        prev = bk
    return v0 + (v1 - v0) * lvl


@njit(cache=True)
def _interp1(x, grid, col):
    """Linear interpolation of col on the increasing grid at a scalar x, end values held (bisection; np.interp on a
    scalar is several times slower inside numba)."""
    g = grid.shape[0]
    if x <= grid[0]:
        return col[0]
    if x >= grid[g - 1]:
        return col[g - 1]
    lo = 0
    hi = g - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if grid[mid] <= x:
            lo = mid
        else:
            hi = mid
    w = (x - grid[lo]) / (grid[hi] - grid[lo])
    return col[lo] + w * (col[hi] - col[lo])


@njit(cache=True)
def _interp2(x, grid, col_a, col_b):
    """_interp1 of two columns sharing one bisection."""
    g = grid.shape[0]
    if x <= grid[0]:
        return col_a[0], col_b[0]
    if x >= grid[g - 1]:
        return col_a[g - 1], col_b[g - 1]
    lo = 0
    hi = g - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if grid[mid] <= x:
            lo = mid
        else:
            hi = mid
    w = (x - grid[lo]) / (grid[hi] - grid[lo])
    return col_a[lo] + w * (col_a[hi] - col_a[lo]), col_b[lo] + w * (col_b[hi] - col_b[lo])


@njit(cache=True)
def _rhs(vs, vp, vj, r_drv, c_pad, grid, c_tab, rs_tab, two_node):
    cj, rs = _interp2(vj, grid, c_tab, rs_tab)
    if two_node:
        dvp = ((vs - vp) / r_drv - (vp - vj) / rs) / c_pad
        dvj = (vp - vj) / (rs * cj)
        return dvp, dvj
    dvj = (vs - vj) / ((r_drv + rs) * cj)
    return 0.0, dvj


@njit(cache=True)
def junction_rc_run_nb(t0, h_half, bits, t_bit, sigma, v0, v1, r_drv, c_pad, grid, c_tab, rs_tab, n_sub,
                       out_vj, out_vsrc):
    """Junction voltage at t0 + j h_half, j = 0 .. out_vj.shape[0] - 1 (module docstring network).

    Starts in the DC state of the source at t0. bits: the (cyclic) pattern; t_bit (s); sigma (s) Gaussian edge;
    v0/v1 reverse voltages of bits 0/1 (V); r_drv (ohm), c_pad (F; <= 0 or r_drv <= 0: one node); grid (G,) with
    c_tab (F) and rs_tab (ohm) ring-level; n_sub RK4 substeps per half step. Writes out_vj and out_vsrc (V)."""
    two_node = c_pad > 0.0 and r_drv > 0.0
    vs0 = nrz_source_nb(t0, bits, t_bit, sigma, v0, v1)
    vp = vs0
    vj = vs0
    out_vj[0] = vj
    out_vsrc[0] = vs0
    hs = h_half / n_sub
    s_b = vs0
    for j in range(1, out_vj.shape[0]):
        ta = t0 + (j - 1) * h_half
        for i in range(n_sub):
            t = ta + i * hs
            if sigma <= 0.0:
                s_a = nrz_source_nb(t + 0.5 * hs, bits, t_bit, sigma, v0, v1)
                s_m = s_a
                s_b = s_a
            else:
                s_a = s_b                     # the source at the end of the previous substep
                s_m = nrz_source_nb(t + 0.5 * hs, bits, t_bit, sigma, v0, v1)
                s_b = nrz_source_nb(t + hs, bits, t_bit, sigma, v0, v1)
            k1p, k1j = _rhs(s_a, vp, vj, r_drv, c_pad, grid, c_tab, rs_tab, two_node)
            k2p, k2j = _rhs(s_m, vp + 0.5 * hs * k1p, vj + 0.5 * hs * k1j, r_drv, c_pad, grid, c_tab, rs_tab,
                            two_node)
            k3p, k3j = _rhs(s_m, vp + 0.5 * hs * k2p, vj + 0.5 * hs * k2j, r_drv, c_pad, grid, c_tab, rs_tab,
                            two_node)
            k4p, k4j = _rhs(s_b, vp + hs * k3p, vj + hs * k3j, r_drv, c_pad, grid, c_tab, rs_tab, two_node)
            vp += hs * (k1p + 2.0 * k2p + 2.0 * k3p + k4p) / 6.0
            vj += hs * (k1j + 2.0 * k2j + 2.0 * k3j + k4j) / 6.0
            if not two_node:
                vp = vj
        out_vj[j] = vj
        out_vsrc[j] = nrz_source_nb(t0 + j * h_half, bits, t_bit, sigma, v0, v1)


# ============================================================================================ Python API
@dataclass
class JunctionRun:
    """Junction voltage run on the half-step grid: t_s (2K+1,), v_src and v_j (V), with the bit pattern, the bit
    time t_bit_s, samples_per_bit (grid points per bit; the half-step grid has twice as many) and the NRZ source
    levels (v_lo, v_hi) (V)."""

    t_s: np.ndarray
    v_src: np.ndarray
    v_j: np.ndarray
    bits: np.ndarray
    t_bit_s: float
    samples_per_bit: int
    n_warm_bits: int
    levels: tuple | None = None


@dataclass
class DriverJunction:
    """Driver + junction network of one ring (module docstring).

    v_grid (G,) reverse voltage (V); c_j_tab (G,) ring-level differential junction capacitance (F); r_s_tab (G,)
    ring-level series resistance (ohm); r_drv_ohm driver output resistance; c_pad_f pad/wiring capacitance (F, 0 =
    none; no schema key: assumed); rise_time_s 10-90 % rise time of the Gaussian driver edge (no schema key:
    assumed, see docs/handoff/waveform.md); meta: provenance.
    """

    v_grid: np.ndarray
    c_j_tab: np.ndarray
    r_s_tab: np.ndarray
    r_drv_ohm: float
    c_pad_f: float = 0.0
    rise_time_s: float = 0.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.v_grid = np.ascontiguousarray(np.asarray(self.v_grid, dtype=float))
        if self.v_grid.ndim != 1 or self.v_grid.size < 2 or np.any(np.diff(self.v_grid) <= 0):
            raise ValueError("v_grid must be strictly increasing")
        for name in ("c_j_tab", "r_s_tab"):
            a = np.asarray(getattr(self, name), dtype=float)
            if a.ndim == 0:
                a = np.full(self.v_grid.shape, float(a))
            if a.shape != self.v_grid.shape or not np.all(np.isfinite(a)):
                raise ValueError(f"{name} must be finite with the shape of v_grid")
            setattr(self, name, np.ascontiguousarray(a))
        if np.any(self.c_j_tab <= 0) or np.any(self.r_s_tab < 0):
            raise ValueError("need C_j > 0 and R_s >= 0")
        if not (np.isfinite(self.r_drv_ohm) and self.r_drv_ohm >= 0) or self.c_pad_f < 0 or self.rise_time_s < 0:
            raise ValueError("need finite r_drv_ohm >= 0, c_pad_f >= 0, rise_time_s >= 0")
        if self.r_drv_ohm == 0 and np.any(self.r_s_tab == 0):
            raise ValueError("R_drv + R_s must be > 0")
        self.r_drv_ohm = float(self.r_drv_ohm)
        self.c_pad_f = float(self.c_pad_f)
        self.rise_time_s = float(self.rise_time_s)

    @classmethod
    def from_plant_params(cls, pp, b: int = 0, ring: int = 0, *, rise_time_s: float, c_pad_f: float = 0.0,
                          r_drv_ohm: float | None = None, r_s_scale: float = 1.0) -> "DriverJunction":
        """C_j(V) and R_s(V) of ring ``ring`` in world b on the junction grid (mrm.plant.phase: junction table x fill x
        length, LUT or analytic per fidelity.junction_model); R_drv = pp.datapath.driver_output_resistance_ohm
        unless given. r_s_scale multiplies R_s (for an R_s C fit such as CAL-10; 1 = the preset)."""
        from mrm.plant import phase as ph

        P = ph.PhaseParams.from_plant_params(pp)
        rp = P.ring[b]
        r = P.table_row(b)
        n_r = rp.shape[0]
        g = P.grid
        cj = np.empty(g.size)
        rs = np.empty(g.size)
        sh, fal, cjr, rsr = (np.zeros(n_r) for _ in range(4))
        fl = np.zeros(n_r, np.int8)
        t_ref = np.ascontiguousarray(rp[:, ph.RP_T_REF])
        for i, v in enumerate(g):
            ph.resonance_shift_row(rp, t_ref, np.full(n_r, float(v)), P.grid, P.dneff[r], P.alpha_fca[r],
                                   P.capacitance[r], P.r_series[r], sh, fal, cjr, rsr, fl)
            cj[i] = cjr[ring]
            rs[i] = rsr[ring] * float(r_s_scale)
        if r_drv_ohm is None:
            r_drv_ohm = float(np.asarray(pp.datapath.driver_output_resistance_ohm, float).reshape(-1)[
                b if np.size(pp.datapath.driver_output_resistance_ohm) > 1 else 0])
            if not np.isfinite(r_drv_ohm):
                raise ValueError("driver.output_resistance_ohm is absent; pass r_drv_ohm")
        return cls(g, cj, rs, r_drv_ohm, c_pad_f=c_pad_f, rise_time_s=rise_time_s,
                   meta={"kind": "plant_params", "world": int(b), "ring": int(ring), "r_s_scale": float(r_s_scale),
                         "junction_model": int(pp.junction.model)})

    # ---------------------------------------------------------------- static quantities
    def c_j(self, v):
        """Differential junction capacitance (F) at reverse voltage v (V)."""
        return np.interp(v, self.v_grid, self.c_j_tab)

    def r_s(self, v):
        """Series resistance (ohm) at reverse voltage v (V)."""
        return np.interp(v, self.v_grid, self.r_s_tab)

    @property
    def sigma_s(self) -> float:
        """Gaussian-edge standard deviation (s)."""
        return rise_to_sigma(self.rise_time_s)

    def h_electrical(self, v_bias, f_hz, include_driver: bool = True):
        """Small-signal V_j/V_src (complex) at frequencies f_hz (module docstring), with the Gaussian driver edge
        exp(-(2 pi f sigma)^2/2) when include_driver."""
        s = 2j * math.pi * np.atleast_1d(np.asarray(f_hz, dtype=float))
        cj = float(self.c_j(v_bias))
        rs = float(self.r_s(v_bias))
        rd = self.r_drv_ohm
        cp = self.c_pad_f
        if cp > 0 and rd > 0:
            h = 1.0 / (1.0 + s * (rd * cp + (rd + rs) * cj) + s * s * rd * rs * cp * cj)
        else:
            h = 1.0 / (1.0 + s * (rd + rs) * cj)
        if include_driver and self.rise_time_s > 0:
            h = h * np.exp(-0.5 * (np.abs(s) * self.sigma_s) ** 2)
        return h

    def rc_pole_hz(self, v_bias) -> float:
        """3 dB frequency of |V_j/V_src| without the driver edge (Hz)."""
        from scipy.optimize import brentq
        f = lambda x: abs(self.h_electrical(v_bias, x, include_driver=False)[0]) ** 2 - 0.5  # noqa: E731
        hi = 1e9
        while f(hi) > 0:
            hi *= 2.0
        return float(brentq(f, 1e6, hi, xtol=1e3))

    def tau_min_s(self) -> float:
        """Fastest time constant of the network over the grid (s), for the RK4 substep."""
        rs_min = max(float(np.min(self.r_s_tab)), 1e-3)
        cj_min = float(np.min(self.c_j_tab))
        if self.c_pad_f > 0 and self.r_drv_ohm > 0:
            rpar = self.r_drv_ohm * rs_min / (self.r_drv_ohm + rs_min)
            return min(rs_min * cj_min, rpar * self.c_pad_f)
        return (self.r_drv_ohm + rs_min) * cj_min

    def levels(self, v_bias, swing_vpp, level_mode: int = 0) -> tuple[float, float, int]:
        """(V_lo, V_hi, flags) of the NRZ levels by mrm.plant.electrical.nrz_levels_row (mode 0 centered, 1
        one-sided; clamped to the -0.5 .. 4.0 V junction grid with the electrical flags)."""
        from mrm.plant import electrical as el
        lo = np.zeros(1)
        hi = np.zeros(1)
        fl = np.zeros(1, np.int8)
        el.nrz_levels_row(np.array([float(v_bias)]), np.array([float(swing_vpp)]), int(level_mode),
                          el.JUNCTION_GRID_V_MIN, el.JUNCTION_GRID_V_MAX, lo, hi, fl)
        return float(lo[0]), float(hi[0]), int(fl[0])

    # ---------------------------------------------------------------- dynamics
    def run(self, bits, t_bit_s: float, samples_per_bit: int, v_lo: float, v_hi: float, *,
            n_warm_bits: int | None = None, n_sub: int | None = None) -> JunctionRun:
        """Junction voltage for the cyclic pattern ``bits`` over [-n_warm_bits t_bit, n_bits t_bit] on the half-step
        grid h/2 = t_bit/(2 samples_per_bit); the run starts in the DC state at -n_warm_bits t_bit (default: the
        whole pattern if it has at most 256 bits, else 64 bits; the network settles in a few ps)."""
        bits = np.ascontiguousarray(np.asarray(bits, dtype=np.uint8))
        n = bits.size
        if n < 1 or samples_per_bit < 2:
            raise ValueError("need at least one bit and samples_per_bit >= 2")
        if n_warm_bits is None:
            n_warm_bits = n if n <= 256 else 64
        h_half = t_bit_s / (2 * samples_per_bit)
        if n_sub is None:
            lim = 0.1 * self.tau_min_s()
            if self.rise_time_s > 0:
                lim = min(lim, 0.5 * self.sigma_s)
            n_sub = max(1, int(math.ceil(h_half / lim)))
        n_half = 2 * samples_per_bit * (n + n_warm_bits)
        if n_half > MAX_HALF_STEPS:
            raise ValueError(f"{n + n_warm_bits} bits x {samples_per_bit} samples/bit = {n_half:.3g} half steps "
                             f"exceed MAX_HALF_STEPS = {MAX_HALF_STEPS:.3g} (about 100 B each through simulate): use a "
                             f"pattern segment (prbs(order, n_bits)) or fewer samples per bit")
        vj = np.empty(n_half + 1)
        vs = np.empty(n_half + 1)
        t0 = -n_warm_bits * t_bit_s
        junction_rc_run_nb(t0, h_half, bits, float(t_bit_s), self.sigma_s, float(v_lo), float(v_hi),
                           self.r_drv_ohm, self.c_pad_f, self.v_grid, self.c_j_tab, self.r_s_tab, int(n_sub), vj, vs)
        t = t0 + h_half * np.arange(n_half + 1)
        return JunctionRun(t, vs, vj, bits, float(t_bit_s), int(samples_per_bit), int(n_warm_bits),
                           levels=(float(v_lo), float(v_hi)))

    def step_rise_time_s(self, v_lo: float, v_hi: float, t_bit_s: float = 400e-12, samples_per_bit: int = 4000,
                         quantity: str = "charge") -> dict:
        """10-90 % rise and fall times (s) of the junction response to one long bit (driver edge + RC network).

        quantity "charge": of Q_j(t) = int C_j dV (the resonance shift is proportional to the removed charge, 03
        section 5.3, so this is the edge that the optics sees); "voltage": of V_j(t). Returns {"rise", "fall",
        "mean"} in s, for the normalized-LUT rise-time key (lut.py)."""
        bits = np.array([0, 1, 1, 1, 0, 0, 0, 0], np.uint8)
        run = self.run(bits, t_bit_s, samples_per_bit, v_lo, v_hi, n_warm_bits=1)
        t, v = run.t_s, run.v_j
        if quantity == "charge":
            vv = np.linspace(min(v_lo, v_hi), max(v_lo, v_hi), 2001)
            q = np.concatenate([[0.0], np.cumsum(0.5 * (self.c_j(vv[1:]) + self.c_j(vv[:-1])) * np.diff(vv))])
            y = np.interp(v, vv, q)
        elif quantity == "voltage":
            y = v
        else:
            raise ValueError("quantity must be 'charge' or 'voltage'")
        y0, y1 = y[0], y.max()
        e = (y - y0) / (y1 - y0)

        def crossing(seg_t, seg_e, level, rising):
            idx = np.nonzero(seg_e >= level)[0] if rising else np.nonzero(seg_e <= level)[0]
            i = int(idx[0])
            return np.interp(level, [seg_e[i - 1], seg_e[i]] if rising else [seg_e[i], seg_e[i - 1]],
                             [seg_t[i - 1], seg_t[i]] if rising else [seg_t[i], seg_t[i - 1]])

        m_r = (t >= 0) & (t < 3 * t_bit_s)
        m_f = t >= 3 * t_bit_s
        tr = crossing(t[m_r], e[m_r], 0.9, True) - crossing(t[m_r], e[m_r], 0.1, True)
        tf = crossing(t[m_f], e[m_f], 0.1, False) - crossing(t[m_f], e[m_f], 0.9, False)
        return {"rise": float(tr), "fall": float(tf), "mean": 0.5 * float(tr + tf)}
