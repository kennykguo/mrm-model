"""Optical plant: exact add-drop microring with CW/CCW backscatter, TCMT over orders, single-pass WDM bus.

Owner: plant_optics. Normative sources: docs/ARCHITECTURE.md (v2) and docs/research/07_design_critique.md
P2-2 (backscatter per order), P2-4 (solver-independent optics, complex TCMT sum over orders), P1-5 (coupling at
each LINE wavelength), P1-9 (the monitor sees the CW drop only; the drop sums every line and every order), P0-1
(absorbed power for the implicit self-heating solve). Equations: docs/research/03_mrm_compact_models.md ("03")
sections 2.1-2.8, 8.1 and 9.2, including its [corrected] notes and its verification log.

Conventions (03 section 2.1)
----------------------------
* Time dependence e^{+i w t}; propagation over a round trip multiplies a field by a*exp(-i*phi).
* Guided-wave amplitudes s have |s|^2 in W. TCMT mode amplitudes a have |a|^2 in J.
* kappa_i, kappa_e, kappa_d are ENERGY decay rates in 1/s; g (backscatter) and Delta are in rad/s;
  Delta = w_laser - w_res (Delta > 0: laser blue of the resonance).
* Ports (fixed numbering used by every function here):
    0 = input end of bus 1      (incoming: the laser, s_in;      outgoing: reflection, s_refl)
    1 = through end of bus 1    (incoming: backward light, s'_in; outgoing: through, s_thru)
    2 = drop end of bus 2       (incoming: backward light from the monitor PD, s'_drop; outgoing: CW drop -> monitor)
    3 = add end of bus 2        (incoming: add light, s_add;      outgoing: CCW drop, the "add-port output")
  The CW mode is the one the laser excites. Its drop goes to port 2, the monitor. The CCW mode (fed by
  backscatter) drops to port 3, not to the monitor (07 P2-2).

Solver-independent ring description (07 P2-4), per ring and per laser line
--------------------------------------------------------------------------
* ``k2_th``, ``k2_dr``: per-pass POWER cross-coupling K^2 of the through (bus 1) and drop (bus 2) couplers,
  evaluated by the caller at the LINE wavelength (07 P1-5; never at the resonance and never as a function of the
  heater code). Couplers are lossless point couplers [[t, -ik], [-ik, t]], t = sqrt(1-K^2), k = sqrt(K^2)
  (Bogaerts et al., Laser Photonics Rev. 6, 47 (2012); 03 section 2.6). Coupler excess loss belongs in ``a_rt``.
* ``psi``: the ring-side self-coupling phase of the two couplers together (rad). It is added to the round-trip
  phase. The bus-side coupler phases change only output phases, which carry no power information in a single-pass
  bus, so they are not modelled.
* ``a_rt``: round-trip AMPLITUDE transmission, a = exp(-alpha_i * L / 2) with alpha_i the intrinsic POWER
  attenuation (absorption + scattering + radiation; 07 P0-2).
* ``phi``: round-trip phase at the line, phi = 2*pi*[n_eff(lambda, T, V)*L + fill*L*dn_eff(V)]/lambda (07 P1-6,
  P1-7). Supplied by the caller (the phase model owns n_eff, the junction fill and the temperature).
* ``t_rt``: round-trip GROUP delay T_rt = n_g*L/c in s (= 1/FSR). It sets only the stored energies and the TCMT
  rates; the exact port powers depend on phi, K^2 and a alone.
* ``rb``: complex round-trip backscatter per longitudinal order, r_b,n = g_n * T_rt = 2*pi*beta_n/FSR (07 P2-2),
  stored as an array of length M_b and applied PERIODICALLY in the absolute order index n (order n uses
  rb[n mod M_b]). beta_n is the CW<->CCW coupling rate in Hz, g_n = 2*pi*beta_n in rad/s, in the TCMT phase
  convention of 03 section 2.2 (da1/dt contains -i g a2, da2/dt contains -i g* a1).

Exact ring with backscatter (``ring_exact``): the formulation and why it agrees with TCMT to O(1/F^2)
--------------------------------------------------------------------------------------------------------
The ring is a closed loop of CW (u) and CCW (v) travelling waves with |u|^2, |v|^2 in W: the input coupler at
z = 0, the drop coupler at z = L/2 (diametrically opposite, so the drop field travels half a round trip, 03
section 2.7), and a PAIR of identical lumped, lossless, reciprocal scatterers at z = L/4 and z = 3L/4. Each
scatterer has the scattering relations
    u(z+) = tau*u(z-) - conj(rho)*v(z+),     v(z-) = rho*u(z-) + tau*v(z+),     tau = sqrt(1 - |rho|^2) (real),
i.e. the reflection seen from the CCW side is -conj(rho), which is the Hermitian (energy-conserving) coupling of
03 section 2.4, and tau is real so that the scatterers do not shift the mean resonance (TCMT's g does not). The
steady state of the loop is a 2x2 complex affine fixed point, solved exactly per line (no truncation in orders,
any finesse).

Why this choice. With w = -ln(t1*t2*a) + i*phi, the exact port responses are meromorphic in w and the
Mittag-Leffler expansions (03 verification log, row 5, for the drop identity) give, to first order in rho:
    through  = cosh(x_e) - 2 sinh(x_e) * sum_n 1/(w - 2 pi i n)
    drop     = -sqrt(2 sinh x_e * 2 sinh x_d) * sum_n (-1)^n/(w - 2 pi i n)
    refl     = (pair at L/4, 3L/4)  ~ sum_n 2 (-1)^n rho/(w - 2 pi i n)^2   (pure double poles)
    add_out  = (pair at L/4, 3L/4)  ~ sum_n rho/(w - 2 pi i n)^2            (pure double poles)
with x_e = -ln t1, x_d = -ln t2. The TCMT of 03 section 2.2 with the logarithmic rate mapping of 03 section 2.6
(kappa_i = -2 ln(a)/T_rt, kappa_e = -2 ln(t1)/T_rt, kappa_d = -2 ln(t2)/T_rt) is the same pole sum with
prefactors 2x_e, 2*sqrt(x_e x_d), ..., which differ from the exact ones by factors 1 + O(x^2) = 1 + O(1/F^2).
A single scatterer at any other position, or a scatterer whose reflectance varies linearly in frequency across a
resonance, adds simple-pole (Fano) terms whose size relative to the resonant term is O(1/F); the pair at L/4 and
3L/4 cancels them exactly (reflection) or produces none (add-port output). Per-order reflectances are realised by
a frequency-dependent rho(phi) that is FLAT at every resonance node phi = 2 pi n:
    rho(phi) = rho_n cos^2(psi/4) + rho_{n+1} sin^2(psi/4),   psi = phi - 2 pi n in [0, 2 pi),
    rho_n    = -(i/2) (-1)^n conj(r_b,n)            (the mapping that makes the TCMT coupling equal g_n = r_b,n/T_rt),
so rho(2 pi n + eps) = rho_n + O(eps^2) and no O(1/F) term appears. The scatterers stay unitary at every
frequency, so energy conservation and reciprocity are exact (tests: energy balance to 1e-12, S = S^T).
Physical reading: a random distributed sidewall backscatter has a different Fourier component (beta_n) at every
order (Li et al., Laser Photonics Rev. 10, 420 (2016); 03 section 2.8). Where the scattering sits in the ring
changes the reflected and add-port powers by a relative O(1/F) (one scatterer at L/4 instead of the pair: about 8 %
of the reflected power at Q_L = 3500, F = 27.5; test_single_scatterer_would_be_first_order), and through and drop
by O(F r_b^2) of the input power. Both are far below the uncertainty of beta itself (07 P2-2: 1 to 10 GHz, random
per order, i.e. a factor of 100 in reflected power).
The per-order assignment uses the ABSOLUTE order n = floor(phi/2 pi): the caller must pass the absolute round-trip
phase (or one re-anchored only by multiples of 2 pi M_b), else the orders get other beta values.

TCMT (``ring_tcmt``)
--------------------
Steady state of 03 section 2.2 for each order n in the M orders nearest the line, summed in COMPLEX amplitude
(07 P2-4), with the (-1)^n sign on both drop-coupler terms (03 section 2.7: the drop coupler sits half a round
trip away). Delta_n = (phi + psi - 2 pi n)/T_rt. The multi-order sum is not exactly energy-conserving (the shared
port makes the modes' leakage non-orthogonal); the exact ring is. Known limit (07 P2-4, 03 section 2.7): the
single-order Lorentzian under-estimates the drop tail of the adjacent 400 GHz line (0.7 dB at Q_L = 3500,
FSR = 1.8 THz, kappa_d = 1 % kappa) because it drops the e^{-w}/(1-e^{-w}) background; the multi-order sum
recovers most of it.

Bus (``bus_propagate``)
-----------------------
N rings visited in ``bus_order``; each sees the through power the previous ring let pass, times the bus power
transmission of the segment before it. Lines are independent (different frequencies; no coherent mixing between
lines). Single pass: light reflected by a ring is reported but not re-injected. Bound: every pair of rings i < j
on a line forms a Fabry-Perot whose round trip carries at most |r_i| |r_j| (|r| the amplitude reflection of a
ring, the same from either bus side: |S_00| = |S_11|, checked to rounding in tests/verification/test_optics.py;
the transmissions in between only lower it). The multi-bounce corrections to any forward field are majorized by a
series in
    eps_l = sum_{i < j} |r_i,l| |r_j,l|   (all pairs),
so the relative error of any forward power at line l is at most 1/(1 - eps_l)^2 - 1 = 2 eps_l + 3 eps_l^2 + ...;
``bus_propagate`` returns it per line (``bound``). For two rings this is the exact Fabry-Perot worst case; for N
identical, in-phase, nearly lossless reflectors the error is (N^2 - N)|r|^2 = 2 eps to leading order, so it is
tight. (A maximum over single cuts, max_k (sum before k)(sum from k), is NOT a bound for three or more rings:
rings detuned by +4, +1.6 and -4 half-widths with r_b = 0.15 give a 14.8 % error against 8.5 %; test_optics.)
With per-ring reflections of -17 to -34 dB (07 P2-2) and only one ring near resonance at a given line, eps is
below 1e-3 in normal operation; it grows only when several rings sit near the same line (wrong-line locks,
acquisition sweeps).

Data averaging (03 section 9.2; textbook Ch17 eq. 17.6b)
---------------------------------------------------------
For NRZ with ones density p1 per ring, every ring is solved at its two junction levels (phi, a and eta differ) and
P_bar = p1*P(level 1) + (1 - p1)*P(level 0) for through, drop, absorbed, loss and energies. With independent data
on different rings the expected power transfer of the cascade factorizes, so the averaged input to the next ring
is P_in*T_bar (exact for independent data, not an approximation). Per-level P1/P0 are reported at each ring's
through and drop ports and at the bus output (ring k at a fixed level, the others averaged), for the data path
(07 P0-11) and the OMA metrics.

Interfaces
----------
Python (analysis, tests, calibration): :func:`ring_exact`, :func:`ring_tcmt`, :func:`ring_smatrix`,
:func:`bus_propagate` (-> :class:`BusResult`), :func:`dpabs_dlambda_res`, :func:`exact_to_tcmt`,
:func:`tcmt_to_exact`, :func:`beta_to_rb`, :func:`dphi_dlambda_res`, :func:`nrz_average`.
Compiled kernel (no allocation; every argument an array or a scalar; outputs written into preallocated arrays from
:func:`make_bus_outputs` and :func:`make_workspace`; field indices RL_*, R_*, L_* below):
:func:`bus_propagate_nb` (the whole bus), and for the implicit self-heating walk along bus_order (07 P0-1)
:func:`dpabs_dlambda_res_nb` (P_abs of ring k and its slope at fixed input), :func:`pabs_bounds_nb` (the p_max and
d2p_max bounds that mrm.plant.selfheat needs), :func:`pabs_slope_bound_nb` (the |dP_abs/dphi| bound for the
cross-ring lag check of mrm.plant.thermal), :func:`ring_lines_nb` (every output of ring k, optionally with the
slope, and its through power for the next ring) and :func:`bus_finalize_nb`.
Ready-made self-heating adapter for mrm.plant.selfheat (the pabs_fn / commit_fn ABI of docs/handoff/plant_thermal.md):
:func:`pack_selfheat_args` -> (args_f, args_i), :func:`selfheat_pabs` (pabs_fn), :func:`selfheat_commit`
(commit_fn), :func:`selfheat_bounds` (p_max, d2p_max, dpdt_max in temperature units) and :func:`selfheat_views`
(views to update inputs and read outputs).
The hot kernels use ``error_model="numpy"`` (no Python division checks; the Python wrappers validate inputs). All
kernels are numba @njit(cache=True), take scalars and arrays only, and run unchanged under NUMBA_DISABLE_JIT=1. The
per-tick kernels the engine calls are compiled without the numba runtime (``_nrt=False``; docs/handoff/perf.md R2-11):
they allocate nothing and call no function that returns arrays (the view helpers _sh_core, _sh_outputs and
selfheat_views are inline="always" for that reason).
The workspaces (cw, fw, work) and outputs are written by every call: with prange over the batch, give every batch
element its own (for example cw of shape (B, N_CW, L), passing cw[b]).

Cost (measured on the shared 16 GB M2, called from compiled code, N = 8 rings, L = 8 lines, M_b = 3 orders; the
spread is the load from other jobs): one :func:`bus_propagate_nb` takes 9-15 us (exact with backscatter), 15-27 us
(same, NRZ two-level), 23-43 us (NRZ two-level with the slope), 5.7 / 8.2 us (exact without backscatter, the Airy
fast path, CW / NRZ) and 12-20 us (TCMT, M = 3, CW). One :func:`dpabs_dlambda_res_nb` evaluation of one ring at 8
lines (one Newton evaluation of 07 P0-1) takes 0.66 / 1.1 us (Airy, CW / NRZ), 2.4 / 4.1 us (backscatter, CW / NRZ)
and 3.5 us (TCMT, NRZ). 07 P0-9 budgets at most 10 us of wall time per 1 us plant step for the whole plant at
B = 1, so the optics alone use that budget; docs/handoff/plant_optics.md discusses it.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from mrm.units import C0

TWO_PI = 2.0 * math.pi  # mathematical constant
# Index cast for the hot per-tick kernels (docs/handoff/perf.md R2-1): numba adds a negative-index wraparound (a
# compare and a select) to every array access whose index is a signed integer it cannot prove non-negative, which
# in the packed self-heating rows is every access (the offsets come from args_i). An unsigned index has no
# wraparound; every index cast with _U here is a non-negative offset by construction. Same values, bit for bit.
_U = np.uint64

# ---- field indices of the per-(ring, line) output array out_rl[field, ring, line] (units in brackets) -------------
RL_PIN = 0      # power arriving at the ring's input port, after the bus segment [W]
RL_THRU = 1     # through-port power, NRZ-averaged [W]
RL_DROP = 2     # CW drop-port power (goes to the monitor PD), NRZ-averaged [W]
RL_ADD = 3      # CCW drop power leaving at the add port (does NOT reach the monitor), NRZ-averaged [W]
RL_REFL = 4     # power reflected back toward the laser from this ring's input port, NRZ-averaged [W]
RL_LOSS = 5     # power lost to intrinsic loss (absorption + scattering + radiation), NRZ-averaged [W]
RL_ABS = 6      # absorbed (heat) power eta*loss, NRZ-averaged [W]
RL_UCW = 7      # CW intracavity energy, NRZ-averaged [J]
RL_UCCW = 8     # CCW intracavity energy, NRZ-averaged [J]
RL_THRU1 = 9    # through power with this ring at level 1 (input NRZ-averaged) [W]
RL_THRU0 = 10   # through power with this ring at level 0 [W]
RL_DROP1 = 11   # CW drop power with this ring at level 1 [W]
RL_DROP0 = 12   # CW drop power with this ring at level 0 [W]
RL_POUT1 = 13   # bus-output power at this line with this ring at level 1, every other ring averaged [W]
RL_POUT0 = 14   # bus-output power at this line with this ring at level 0, every other ring averaged [W]
RL_DABS_DPHI = 15  # d(absorbed power)/d(round-trip phase), NRZ-averaged, input held fixed [W/rad] (with_slope only)
N_RL = 16

# ---- per-ring output out_r[field, ring] ----------------------------------------------------------------------------
R_DROP_SUM = 0   # CW drop power summed over lines (all orders included), NRZ-averaged: the monitor's optical input [W]
R_ABS_SUM = 1    # absorbed power summed over lines, both directions, all orders, NRZ-averaged: self-heating source [W]
R_LOSS_SUM = 2   # intrinsic-loss power summed over lines [W]
R_DROP_SUM1 = 3  # summed CW drop power with this ring at level 1 (P1 at the monitor) [W]
R_DROP_SUM0 = 4  # summed CW drop power with this ring at level 0 (P0 at the monitor) [W]
R_DABS_DLAM = 5  # d(R_ABS_SUM)/d(resonance wavelength), rigid red shift, input held fixed [W/m] (with_slope only)
N_R = 6

# ---- per-line output out_l[field, line] ----------------------------------------------------------------------------
L_POUT = 0       # NRZ-averaged power at the bus output (after bus_t2_out) [W]
L_BOUND = 1      # single-pass bound 1/(1-eps)^2 - 1, eps = sum_{i<j} |r_i||r_j|: max rel. error of forward powers [-]
L_REFL_IN = 2    # incoherent sum of all ring reflections arriving back at the bus input [W]
N_L = 3


# =====================================================================================================================
# Mapping between the exact-ring description and TCMT (03 section 2.6; 07 P2-2, P2-4)
# =====================================================================================================================

@njit(cache=True)
def tcmt_rates(k2_th, k2_dr, a_rt, t_rt_s):
    """Exact-ring parameters -> TCMT energy decay rates (1/s).

    kappa_i = -2 ln(a)/T_rt,  kappa_e = -ln(1 - K^2_th)/T_rt,  kappa_d = -ln(1 - K^2_dr)/T_rt
    (03 section 2.6, logarithmic mapping; it matches the Airy linewidth to second order, and with it the TCMT
    pole sum differs from the exact ring only by factors 1 + O(1/F^2), see the module docstring).
    """
    ki = -2.0 * math.log(a_rt) / t_rt_s
    ke = -math.log1p(-k2_th) / t_rt_s
    kd = -math.log1p(-k2_dr) / t_rt_s
    return ki, ke, kd


@njit(cache=True)
def tcmt_backscatter_rate(rb, t_rt_s):
    """Round-trip backscatter r_b (complex, TCMT phase convention) -> CW/CCW coupling rate g = r_b/T_rt (rad/s).

    03 section 2.6 table row "lumped backscatter": |g| = r_bs/T_rt. With g = 2 pi beta (beta in Hz) this is the
    07 P2-2 mapping r_b = 2 pi beta / FSR, FSR = 1/T_rt.
    """
    return rb / t_rt_s


def exact_to_tcmt(k2_th, k2_dr, a_rt, t_rt_s, rb=None):
    """The one mapping from the solver-independent ring description to TCMT rates (07 P2-4).

    Args (broadcastable arrays): k2_th, k2_dr per-pass power couplings [-]; a_rt round-trip amplitude [-];
    t_rt_s round-trip group delay [s]; rb complex round-trip backscatter per order [-] (optional; it is divided by
    t_rt_s with numpy broadcasting, so pass a scalar t_rt_s or align the order axis yourself).
    Returns (kappa_i, kappa_e, kappa_d) in 1/s, and beta_hz = g/(2 pi) = r_b/(2 pi T_rt) (complex, Hz) if rb is
    given. Order n of the exact ring maps to a TCMT mode at Delta_n = (phi + psi - 2 pi n)/T_rt with coupling
    g_n = r_b,n/T_rt (psi the ring-side coupler phase).
    """
    k2_th, k2_dr, a_rt, t_rt_s = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (k2_th, k2_dr, a_rt,
                                                                                               t_rt_s)))
    ki = -2.0 * np.log(a_rt) / t_rt_s
    ke = -np.log1p(-k2_th) / t_rt_s
    kd = -np.log1p(-k2_dr) / t_rt_s
    if rb is None:
        return ki, ke, kd
    beta_hz = np.asarray(rb, dtype=complex) / (TWO_PI * t_rt_s)
    return ki, ke, kd, beta_hz


def tcmt_to_exact(kappa_i, kappa_e, kappa_d, t_rt_s):
    """Inverse of :func:`tcmt_rates`: (kappa_i, kappa_e, kappa_d [1/s], T_rt [s]) -> (K^2_th, K^2_dr, a_rt)."""
    kappa_i, kappa_e, kappa_d, t_rt_s = (np.asarray(x, dtype=float) for x in (kappa_i, kappa_e, kappa_d, t_rt_s))
    return -np.expm1(-kappa_e * t_rt_s), -np.expm1(-kappa_d * t_rt_s), np.exp(-0.5 * kappa_i * t_rt_s)


@njit(cache=True)
def beta_to_rb(beta_hz, fsr_hz):
    """07 P2-2: complex CW/CCW backscatter rate beta (Hz) -> round-trip backscatter r_b = 2 pi beta / FSR."""
    return TWO_PI * beta_hz / fsr_hz


@njit(cache=True)
def dphi_dlambda_res(t_rt_s, lam_m):
    """d(round-trip phase at a FIXED line)/d(resonance wavelength) in rad/m.

    A rigid red shift d(lambda_res) of the comb raises phi at the line by 2 pi n_g L / lambda^2 = 2 pi c T_rt /
    lambda^2 (from phi(lambda) = 2 pi n_eff L / lambda, d phi/d lambda = -2 pi n_g L / lambda^2; 03 section 2.6).
    """
    return TWO_PI * C0 * t_rt_s / (lam_m * lam_m)


@njit(cache=True)
def nrz_average(p1, x1, x0):
    """Mark-density average p1*x1 + (1 - p1)*x0 (03 section 9.2; textbook Ch17 eq. 17.6b)."""
    return p1 * x1 + (1.0 - p1) * x0


# =====================================================================================================================
# Exact ring: per-line kernels
# =====================================================================================================================

@njit(cache=True)
def _imod(n, m):
    r = n % m
    if r < 0:
        r += m
    return r


@njit(cache=True)
def _rho_pair(phi_tot, rb):
    """Reflectance rho of EACH of the two scatterers (L/4 and 3L/4) and d rho/d phi at round-trip phase phi_tot.

    rho_n = -(i/2)(-1)^n conj(r_b,n) at node phi = 2 pi n; C1 cos^2 interpolation flat at every node (module
    docstring). r_b is applied periodically in n with period len(rb).
    """
    mb = rb.shape[0]
    n_lo = int(np.floor(phi_tot / TWO_PI))
    psi = phi_tot - TWO_PI * n_lo
    c = math.cos(0.25 * psi)
    s = math.sin(0.25 * psi)
    w_lo = c * c
    w_hi = s * s
    dw_hi = 0.25 * math.sin(0.5 * psi)          # d w_hi / d phi ; d w_lo / d phi = -dw_hi
    sg_lo = 1.0 if _imod(n_lo, 2) == 0 else -1.0
    rho_lo = -0.5j * sg_lo * rb[_imod(n_lo, mb)].conjugate()
    rho_hi = 0.5j * sg_lo * rb[_imod(n_lo + 1, mb)].conjugate()   # (-1)^(n+1) = -(-1)^n
    rho = w_lo * rho_lo + w_hi * rho_hi
    drho = dw_hi * (rho_hi - rho_lo)
    return rho, drho


@njit(cache=True)
def _abs2(z):
    return z.real * z.real + z.imag * z.imag


@njit(cache=True)
def _dabs2(z, dz):
    return 2.0 * (z.real * dz.real + z.imag * dz.imag)


@njit(cache=True)
def _seg_factors(xi, t_rt):
    """Energy per |amplitude|^2 at a quarter-segment's entry [J/W] and the fraction of power lost in it [-].

    Over a quarter ring the power decays by |p|^2 = exp(-xi/2) (xi = -ln a_rt). Energy = |u_entry|^2 * (T_rt/4) *
    (1 - e^-y)/y with y = xi/2 (exact integral of the exponential decay, energy density |u|^2/v_g); lost power =
    |u_entry|^2 (1 - e^-y), which equals kappa_i times that energy exactly.
    """
    y = 0.5 * xi
    lossfrac = -math.expm1(-y)
    if y > 1e-300:
        g = lossfrac / y
    else:
        g = 1.0
    return 0.25 * t_rt * g, lossfrac


@njit(cache=True)
def _exact_line(phi_tot, xi, t1, k1, t2, k2, t_rt, rb, s0, s1, s2, s3):
    """Exact steady state of one ring at one line (module docstring for the structure).

    Args: phi_tot round-trip phase incl. coupler phase [rad]; xi = -ln(a_rt) [-]; t1, k1, t2, k2 coupler field
    amplitudes [-]; t_rt group delay [s]; rb complex per-order round-trip backscatter [-]; s0..s3 incoming
    amplitudes at ports 0..3 [sqrt(W)].
    Returns (o0, o1, o2, o3, u_cw [J], u_ccw [J], p_loss [W]): outgoing amplitudes at ports 0..3 =
    (reflection, through, CW drop, CCW add-port output), stored energies and intrinsic-loss power.

    Method: the state x = (u, v) at z = 0+ obeys x = M x + c with M = C1 H C2 H, where
    H = Q S Q = (1/tau) [[P2, -conj(rho)], [-rho, 1/P2]] is the half ring (quarter propagation Q = diag(p, 1/p),
    p = exp(-(xi + i phi)/4), P2 = p^2, scatterer S, quarter propagation) and C_j = diag(t_j, 1/t_j) are the
    couplers' ring-side transfers; c holds the bus inputs. One 2x2 solve, then one pass for fields and energies.
    """
    rho, _ = _rho_pair(phi_tot, rb)
    rc = rho.conjugate()
    itau = 1.0 / math.sqrt(1.0 - _abs2(rho))
    p = cmath.exp(-0.25 * (xi + 1j * phi_tot))
    ip = 1.0 / p
    P2 = p * p
    iP2 = ip * ip
    it1 = 1.0 / t1
    it2 = 1.0 / t2
    h11 = P2 * itau
    h12 = -rc * itau
    h21 = -rho * itau
    h22 = iP2 * itau
    k11 = h11 * t2 * h11 + h12 * it2 * h21          # K = H C2 H
    k12 = h11 * t2 * h12 + h12 * it2 * h22
    k21 = h21 * t2 * h11 + h22 * it2 * h21
    k22 = h21 * t2 * h12 + h22 * it2 * h22
    e3 = -1j * k2 * s3                              # add port -> CW at L/2+
    e2 = 1j * k2 * s2 * it2                         # drop-port input -> CCW at L/2+
    cu = t1 * (h11 * e3 + h12 * e2) - 1j * k1 * s0
    cv = it1 * (h21 * e3 + h22 * e2) + 1j * k1 * s1 * it1
    a11 = 1.0 - t1 * k11
    a12 = -t1 * k12
    a21 = -it1 * k21
    a22 = 1.0 - it1 * k22
    det = a11 * a22 - a12 * a21
    u = (a22 * cu - a12 * cv) / det                 # CW at 0+
    v = (a11 * cv - a21 * cu) / det                 # CCW at 0+
    o0 = t1 * s1 - 1j * k1 * v                      # reflection out of the input port
    scw = _abs2(u)                                  # segment 1 (0 -> L/4)
    u = p * u
    v = v * ip
    sccw = _abs2(v)
    u, v = (u - rc * v) * itau, (v - rho * u) * itau          # scatterer A
    scw += _abs2(u)                                 # segment 2 (L/4 -> L/2)
    u = p * u
    v = v * ip
    sccw += _abs2(v)
    o2 = t2 * s3 - 1j * k2 * u                      # CW drop (monitor)
    u = t2 * u - 1j * k2 * s3
    v = (v + 1j * k2 * s2) * it2                    # CCW at L/2+
    o3 = t2 * s2 - 1j * k2 * v                      # CCW drop out of the add port
    scw += _abs2(u)                                 # segment 3 (L/2 -> 3L/4)
    u = p * u
    v = v * ip
    sccw += _abs2(v)
    u, v = (u - rc * v) * itau, (v - rho * u) * itau          # scatterer B
    scw += _abs2(u)                                 # segment 4 (3L/4 -> L)
    u = p * u
    v = v * ip
    sccw += _abs2(v)
    o1 = t1 * s0 - 1j * k1 * u                      # through
    e_per, lossfrac = _seg_factors(xi, t_rt)
    return o0, o1, o2, o3, scw * e_per, sccw * e_per, (scw + sccw) * lossfrac


@njit(cache=True)
def _exact_line_d(phi_tot, xi, t1, k1, t2, k2, t_rt, rb, s0, s1, s2, s3):
    """:func:`_exact_line` plus analytic derivatives with respect to the round-trip phase.

    Returns (o0, o1, o2, o3, u_cw, u_ccw, p_loss, dp_loss/dphi [W/rad], d|o1|^2/dphi, d|o2|^2/dphi).
    Forward-mode differentiation: dM = C1 (dH C2 H + H C2 dH); (I - M) x = c  =>  dx = (I - M)^-1 (dc + dM x);
    then the field pass carries (field, d field). Coupling and loss are held fixed (they do not depend on phi).
    """
    rho, drho = _rho_pair(phi_tot, rb)
    rc = rho.conjugate()
    drc = drho.conjugate()
    itau = 1.0 / math.sqrt(1.0 - _abs2(rho))
    ditau = (rc * drho).real * itau * itau * itau   # d(1/tau) = Re(conj(rho) d rho)/tau^3
    p = cmath.exp(-0.25 * (xi + 1j * phi_tot))
    ip = 1.0 / p
    dp = -0.25j * p
    dip = 0.25j * ip
    P2 = p * p
    iP2 = ip * ip
    it1 = 1.0 / t1
    it2 = 1.0 / t2
    h11 = P2 * itau
    h12 = -rc * itau
    h21 = -rho * itau
    h22 = iP2 * itau
    dh11 = -0.5j * P2 * itau + P2 * ditau
    dh12 = -(drc * itau + rc * ditau)
    dh21 = -(drho * itau + rho * ditau)
    dh22 = 0.5j * iP2 * itau + iP2 * ditau
    k11 = h11 * t2 * h11 + h12 * it2 * h21
    k12 = h11 * t2 * h12 + h12 * it2 * h22
    k21 = h21 * t2 * h11 + h22 * it2 * h21
    k22 = h21 * t2 * h12 + h22 * it2 * h22
    dk11 = dh11 * t2 * h11 + dh12 * it2 * h21 + h11 * t2 * dh11 + h12 * it2 * dh21
    dk12 = dh11 * t2 * h12 + dh12 * it2 * h22 + h11 * t2 * dh12 + h12 * it2 * dh22
    dk21 = dh21 * t2 * h11 + dh22 * it2 * h21 + h21 * t2 * dh11 + h22 * it2 * dh21
    dk22 = dh21 * t2 * h12 + dh22 * it2 * h22 + h21 * t2 * dh12 + h22 * it2 * dh22
    e3 = -1j * k2 * s3
    e2 = 1j * k2 * s2 * it2
    cu = t1 * (h11 * e3 + h12 * e2) - 1j * k1 * s0
    cv = it1 * (h21 * e3 + h22 * e2) + 1j * k1 * s1 * it1
    dcu = t1 * (dh11 * e3 + dh12 * e2)
    dcv = it1 * (dh21 * e3 + dh22 * e2)
    a11 = 1.0 - t1 * k11
    a12 = -t1 * k12
    a21 = -it1 * k21
    a22 = 1.0 - it1 * k22
    det = a11 * a22 - a12 * a21
    u = (a22 * cu - a12 * cv) / det
    v = (a11 * cv - a21 * cu) / det
    r1 = dcu + t1 * (dk11 * u + dk12 * v)
    r2 = dcv + it1 * (dk21 * u + dk22 * v)
    du = (a22 * r1 - a12 * r2) / det
    dv = (a11 * r2 - a21 * r1) / det
    o0 = t1 * s1 - 1j * k1 * v
    # segment 1
    scw = _abs2(u)
    dscw = _dabs2(u, du)
    du = dp * u + p * du
    u = p * u
    dv = dv * ip + v * dip
    v = v * ip
    sccw = _abs2(v)
    dsccw = _dabs2(v, dv)
    nu = u - rc * v
    nv = v - rho * u
    du, dv = ((du - drc * v - rc * dv) * itau + nu * ditau, (dv - drho * u - rho * du) * itau + nv * ditau)
    u, v = nu * itau, nv * itau
    # segment 2
    scw += _abs2(u)
    dscw += _dabs2(u, du)
    du = dp * u + p * du
    u = p * u
    dv = dv * ip + v * dip
    v = v * ip
    sccw += _abs2(v)
    dsccw += _dabs2(v, dv)
    # drop coupler
    o2 = t2 * s3 - 1j * k2 * u
    do2 = -1j * k2 * du
    du = t2 * du
    u = t2 * u - 1j * k2 * s3
    dv = dv * it2
    v = (v + 1j * k2 * s2) * it2
    o3 = t2 * s2 - 1j * k2 * v
    # segment 3
    scw += _abs2(u)
    dscw += _dabs2(u, du)
    du = dp * u + p * du
    u = p * u
    dv = dv * ip + v * dip
    v = v * ip
    sccw += _abs2(v)
    dsccw += _dabs2(v, dv)
    nu = u - rc * v
    nv = v - rho * u
    du, dv = ((du - drc * v - rc * dv) * itau + nu * ditau, (dv - drho * u - rho * du) * itau + nv * ditau)
    u, v = nu * itau, nv * itau
    # segment 4
    scw += _abs2(u)
    dscw += _dabs2(u, du)
    du = dp * u + p * du
    u = p * u
    dv = dv * ip + v * dip
    v = v * ip
    sccw += _abs2(v)
    dsccw += _dabs2(v, dv)
    o1 = t1 * s0 - 1j * k1 * u
    do1 = -1j * k1 * du
    e_per, lossfrac = _seg_factors(xi, t_rt)
    return (o0, o1, o2, o3, scw * e_per, sccw * e_per, (scw + sccw) * lossfrac, (dscw + dsccw) * lossfrac,
            _dabs2(o1, do1), _dabs2(o2, do2))


@njit(cache=True)
def _airy_line(phi_tot, xi, t1, k1, t2, k2, t_rt):
    """Exact ring WITHOUT backscatter, unit forward input (s_in = 1): the Airy closed forms (fast path).

    T = |t1 - t2 a e^{-i phi}|^2/den, D = k1^2 k2^2 a/den, den = |1 - t1 t2 a e^{-i phi}|^2 (Bogaerts 2012;
    03 section 2.6); |u(0+)|^2 = k1^2/den; loss = |u(0+)|^2 (1 - a)(1 + t2^2 a); CW energy from the four quarter
    segments. Returns the same tuple layout as :func:`_exact_line_d` with powers in place of amplitudes:
    (R = 0, T, D, A = 0, u_cw, u_ccw = 0, loss, dloss/dphi, dT/dphi, dD/dphi).
    """
    a = math.exp(-xi)
    q = math.exp(-0.5 * xi)                          # quarter-ring power factor |p|^2
    r = t1 * t2 * a
    c = math.cos(phi_tot)
    s = math.sin(phi_tot)
    den = 1.0 - 2.0 * r * c + r * r
    if den == 0.0:
        # r = t1 t2 a = 1 (a lossless ring without coupling) at resonance: singular. Raised explicitly: the division
        # below would give NaN instead whenever this function was compiled for a caller with numba's numpy error
        # model (numba caches the callee with the caller's error model), so the behaviour would depend on the cache
        raise ZeroDivisionError("_airy_line: den = 0 (lossless ring without coupling, at resonance)")
    dden = 2.0 * r * s
    num = t1 * t1 - 2.0 * t1 * t2 * a * c + t2 * t2 * a * a
    dnum = 2.0 * t1 * t2 * a * s
    T = num / den
    dT = (dnum * den - num * dden) / (den * den)
    D = k1 * k1 * k2 * k2 * a / den
    dD = -D * dden / den
    u2 = k1 * k1 / den
    du2 = -u2 * dden / den
    t22 = t2 * t2
    e_per, lossfrac = _seg_factors(xi, t_rt)
    seg = 1.0 + q + t22 * q * q + t22 * q * q * q    # sum of |u|^2 at the four segment entries / |u(0+)|^2
    return 0.0, T, D, 0.0, u2 * seg * e_per, 0.0, u2 * seg * lossfrac, du2 * seg * lossfrac, dT, dD


@njit(cache=True)
def _inv(z):
    """1/z for a complex z away from 0 and infinity (faster than numba's general complex division)."""
    d = 1.0 / (z.real * z.real + z.imag * z.imag)
    return complex(z.real * d, -z.imag * d)


@njit(cache=True)
def _phase_parts(phi_tot, rb, xi):
    """Per-line pieces shared by the exact kernels.

    Returns (p, 1/p, rho, d rho/d phi): p = exp(-(xi + i phi)/4) is the quarter-ring propagation and rho the
    reflectance of each scatterer of the pair (see :func:`_rho_pair`). One cos/sin pair of psi/4, psi = phi - 2 pi n,
    serves both: cos/sin(phi/4) = cos/sin(psi/4 + n pi/2).
    """
    mb = rb.shape[0]
    n_lo = int(np.floor(phi_tot / TWO_PI))
    psi = phi_tot - TWO_PI * n_lo
    c = math.cos(0.25 * psi)
    s = math.sin(0.25 * psi)
    sg = 1.0 - 2.0 * (n_lo & 1)
    rho_lo = -0.5j * sg * rb[_imod(n_lo, mb)].conjugate()
    rho_hi = 0.5j * sg * rb[_imod(n_lo + 1, mb)].conjugate()
    rho = c * c * rho_lo + s * s * rho_hi
    drho = 0.5 * s * c * (rho_hi - rho_lo)
    q = n_lo & 3
    if q == 0:
        cc = c
        ss = s
    elif q == 1:
        cc = -s
        ss = c
    elif q == 2:
        cc = -c
        ss = -s
    else:
        cc = s
        ss = -c
    e = math.exp(-0.25 * xi)
    return complex(e * cc, -e * ss), complex(cc / e, ss / e), rho, drho


# =====================================================================================================================
# TCMT: per-line kernel (complex sum over M orders)
# =====================================================================================================================

@njit(cache=True)
def _tcmt_line(phi_tot, ki, ke, kd, t_rt, rb, m_orders, s0, s1, s2, s3):
    """TCMT steady state (03 section 2.2-2.5) summed in complex amplitude over the m_orders orders nearest the line.

    The window is centred on the line for every M: n0 = floor(phi/2 pi + 1/2 - (M - 1)/2), i.e. orders
    round(phi/2 pi) -+ (M - 1)/2 for odd M, and floor(phi/2 pi) - M/2 + 1 .. floor(phi/2 pi) + M/2 for even M.
    For order n: Delta_n = (phi_tot - 2 pi n)/T_rt, D_n = kappa/2 + i Delta_n, g_n = r_b,n/T_rt, sigma_n = (-1)^n on
    the drop-coupler terms (03 section 2.7). Steady state of
        D a1 + i g a2   = -i sqrt(ke) s0 - i sqrt(kd) sigma s3
        i g* a1 + D a2  = -i sqrt(ke) s1 - i sqrt(kd) sigma s2
    Outputs: o1 = s0 - i sqrt(ke) sum a1; o2 = s3 - i sqrt(kd) sum sigma a1; o0 = s1 - i sqrt(ke) sum a2;
    o3 = s2 - i sqrt(kd) sum sigma a2. Energies sum |a|^2 over orders (orthogonal modes); p_loss = ki*(U1 + U2).
    Returns the same tuple as :func:`_exact_line_d` (derivatives with respect to phi included).
    """
    kap = ki + ke + kd
    sqe = math.sqrt(ke)
    sqd = math.sqrt(kd)
    mb = rb.shape[0]
    n0 = int(np.floor(phi_tot / TWO_PI + 0.5 - 0.5 * (m_orders - 1)))
    A1 = 0j
    A1d = 0j
    A2 = 0j
    A2d = 0j
    dA1 = 0j
    dA1d = 0j
    U1 = 0.0
    U2 = 0.0
    dU = 0.0
    dD = 1j / t_rt
    for j in range(m_orders):
        n = n0 + j
        sg = 1.0 - 2.0 * (n & 1)
        g = rb[_imod(n, mb)] / t_rt if mb > 0 else 0j       # M_b = 0: no backscatter
        gg = g.real * g.real + g.imag * g.imag
        D = 0.5 * kap + 1j * (phi_tot - TWO_PI * n) / t_rt
        q1 = -1j * (sqe * s0 + sqd * sg * s3)
        q2 = -1j * (sqe * s1 + sqd * sg * s2)
        idet = _inv(D * D + gg)
        a1 = (D * q1 - 1j * g * q2) * idet
        a2 = (D * q2 - 1j * g.conjugate() * q1) * idet
        ddet = 2.0 * D * dD
        da1 = (dD * q1 - a1 * ddet) * idet
        da2 = (dD * q2 - a2 * ddet) * idet
        A1 += a1
        A1d += sg * a1
        A2 += a2
        A2d += sg * a2
        dA1 += da1
        dA1d += sg * da1
        U1 += _abs2(a1)
        U2 += _abs2(a2)
        dU += _dabs2(a1, da1) + _dabs2(a2, da2)
    o1 = s0 - 1j * sqe * A1
    o2 = s3 - 1j * sqd * A1d
    o0 = s1 - 1j * sqe * A2
    o3 = s2 - 1j * sqd * A2d
    do1 = -1j * sqe * dA1
    do2 = -1j * sqd * dA1d
    return o0, o1, o2, o3, U1, U2, ki * (U1 + U2), ki * dU, _dabs2(o1, do1), _dabs2(o2, do2)


# ---- per-ring structure-of-arrays kernels ---------------------------------------------------------------------------
# The exact ring is latency-bound per line (a serial chain of ~20 dependent complex operations), so the ring kernels
# below run each stage as its own loop over the L lines ("vectorized over lines"): independent lines then overlap in
# the CPU pipeline. Measured on the 16 GB M2: about 80 ns per (ring, line) against about 150 ns for the scalar form.

# rows of a per-ring result block res[RES_*, l] (per watt of input power at the ring's input port)
RES_R = 0        # reflection [-]
RES_T = 1        # through [-]
RES_D = 2        # CW drop (monitor) [-]
RES_A = 3        # CCW drop at the add port [-]
RES_UCW = 4      # CW energy [J/W]
RES_UCCW = 5     # CCW energy [J/W]
RES_LOSS = 6     # intrinsic-loss fraction [-]
RES_DLOSS = 7    # d(loss fraction)/d(phi) [1/rad] (0 unless requested)
N_RES = 8

# complex workspace rows cw[CW_*, l]
CW_P = 0
CW_IP = 1
CW_RHO = 2
CW_DRHO = 3
CW_U = 4
CW_V = 5
CW_DU = 6
CW_DV = 7
N_CW = 8

# float workspace rows fw[FW_*, l]; two result blocks (levels 0 and 1) live at the end
FW_XI = 0
FW_T1 = 1
FW_K1 = 2
FW_T2 = 3
FW_K2 = 4
FW_ITAU = 5
FW_DITAU = 6
FW_SCW = 7
FW_SCCW = 8
FW_DSCW = 9
FW_DSCCW = 10
FW_RES0 = 11
FW_RES1 = FW_RES0 + N_RES
N_FW = FW_RES1 + N_RES


def make_workspace(n_lines):
    """Preallocated workspace for the ring and bus kernels: (cw complex (N_CW, L), fw float (N_FW, L), work (L,))."""
    return (np.zeros((N_CW, n_lines), np.complex128), np.zeros((N_FW, n_lines), np.float64),
            np.zeros(n_lines, np.float64))


def make_bus_outputs(n_rings, n_lines):
    """Zeroed output arrays for :func:`bus_propagate_nb`: (out_rl (N_RL, N, L), out_r (N_R, N), out_l (N_L, L))."""
    return (np.zeros((N_RL, n_rings, n_lines)), np.zeros((N_R, n_rings)), np.zeros((N_L, n_lines)))


@njit(cache=True)
def _has_backscatter(rbk):
    """True if any per-order backscatter of this ring is non-zero (else the Airy fast path is exact)."""
    for i in range(rbk.shape[0]):
        if rbk[i].real != 0.0 or rbk[i].imag != 0.0:
            return True
    return False


@njit(cache=True, error_model="numpy", _nrt=False)
def _stages_v(nl, cw, fw, trt_row, res):
    """Exact ring, unit forward input, values only: solve, then the field pass in two half-ring stages.

    The round-trip matrix in closed form: with P2 = p^2, S = t2 P2 + 1/(t2 P2), 1/tau^2 = 1/(1 - |rho|^2), the
    matrix K = H C2 H of :func:`_exact_line` has k11 = (t2 P2^2 + |rho|^2/t2)/tau^2, k22 = (t2|rho|^2 +
    1/(t2 P2^2))/tau^2, k12 = -conj(rho) S/tau^2, k21 = -rho S/tau^2, so det(I - C1 K) = a11 a22 - |rho|^2 S^2/tau^4.
    """
    P = cw[CW_P]
    IP = cw[CW_IP]
    RHO = cw[CW_RHO]
    U = cw[CW_U]
    V = cw[CW_V]
    for l in range(nl):
        p = P[l]
        ip = IP[l]
        rho = RHO[l]
        t1 = fw[FW_T1, l]
        t2 = fw[FW_T2, l]
        r2 = _abs2(rho)
        itau2 = 1.0 / (1.0 - r2)
        it1 = 1.0 / t1
        it2 = 1.0 / t2
        fw[FW_ITAU, l] = math.sqrt(itau2)
        P2 = p * p
        iP2 = ip * ip
        S = t2 * P2 + it2 * iP2
        a11 = 1.0 - t1 * itau2 * (t2 * P2 * P2 + it2 * r2)
        a22 = 1.0 - it1 * itau2 * (t2 * r2 + it2 * iP2 * iP2)
        cu_idet = complex(0.0, -fw[FW_K1, l]) * _inv(a11 * a22 - r2 * itau2 * itau2 * S * S)
        U[l] = a22 * cu_idet                         # CW at 0+
        V[l] = -it1 * itau2 * S * rho * cu_idet      # CCW at 0+
    for l in range(nl):                              # 0 -> L/2, drop coupler
        p = P[l]
        ip = IP[l]
        rho = RHO[l]
        rc = rho.conjugate()
        itau = fw[FW_ITAU, l]
        u = U[l]
        v = V[l]
        k1 = fw[FW_K1, l]
        k2 = fw[FW_K2, l]
        t2 = fw[FW_T2, l]
        res[RES_R, l] = k1 * k1 * _abs2(v)
        scw = _abs2(u)
        u = p * u
        v = v * ip
        sccw = _abs2(v)
        u, v = (u - rc * v) * itau, (v - rho * u) * itau
        scw += _abs2(u)
        u = p * u
        v = v * ip
        sccw += _abs2(v)
        res[RES_D, l] = k2 * k2 * _abs2(u)
        u = t2 * u
        v = v / t2
        res[RES_A, l] = k2 * k2 * _abs2(v)
        U[l] = u
        V[l] = v
        fw[FW_SCW, l] = scw
        fw[FW_SCCW, l] = sccw
    for l in range(nl):                              # L/2 -> L, input coupler
        p = P[l]
        ip = IP[l]
        rho = RHO[l]
        rc = rho.conjugate()
        itau = fw[FW_ITAU, l]
        u = U[l]
        v = V[l]
        scw = fw[FW_SCW, l] + _abs2(u)
        u = p * u
        v = v * ip
        sccw = fw[FW_SCCW, l] + _abs2(v)
        u, v = (u - rc * v) * itau, (v - rho * u) * itau
        scw += _abs2(u)
        u = p * u
        v = v * ip
        sccw += _abs2(v)
        res[RES_T, l] = _abs2(fw[FW_T1, l] - 1j * fw[FW_K1, l] * u)
        e_per, lossfrac = _seg_factors(fw[FW_XI, l], trt_row[l])
        res[RES_UCW, l] = scw * e_per
        res[RES_UCCW, l] = sccw * e_per
        res[RES_LOSS, l] = (scw + sccw) * lossfrac
        res[RES_DLOSS, l] = 0.0


@njit(cache=True, error_model="numpy", _nrt=False)
def _stages_d(nl, cw, fw, trt_row, res):
    """:func:`_stages_v` plus d(loss)/d(phi): quotient rule on the closed-form solution, then forward mode through
    the field pass (coupling and loss held fixed; they do not depend on phi)."""
    P = cw[CW_P]
    IP = cw[CW_IP]
    RHO = cw[CW_RHO]
    DRHO = cw[CW_DRHO]
    U = cw[CW_U]
    V = cw[CW_V]
    DU = cw[CW_DU]
    DV = cw[CW_DV]
    for l in range(nl):
        p = P[l]
        ip = IP[l]
        rho = RHO[l]
        drho = DRHO[l]
        rc = rho.conjugate()
        t1 = fw[FW_T1, l]
        t2 = fw[FW_T2, l]
        r2 = _abs2(rho)
        dr2 = 2.0 * (rc * drho).real
        itau2 = 1.0 / (1.0 - r2)
        ditau2 = itau2 * itau2 * dr2
        itau = math.sqrt(itau2)
        fw[FW_ITAU, l] = itau
        fw[FW_DITAU, l] = 0.5 * ditau2 / itau
        it1 = 1.0 / t1
        it2 = 1.0 / t2
        P2 = p * p
        iP2 = ip * ip
        S = t2 * P2 + it2 * iP2
        dS = -0.5j * t2 * P2 + 0.5j * it2 * iP2
        x11 = t2 * P2 * P2 + it2 * r2
        dx11 = -1j * t2 * P2 * P2 + it2 * dr2
        x22 = t2 * r2 + it2 * iP2 * iP2
        dx22 = t2 * dr2 + 1j * it2 * iP2 * iP2
        a11 = 1.0 - t1 * itau2 * x11
        da11 = -t1 * (ditau2 * x11 + itau2 * dx11)
        a22 = 1.0 - it1 * itau2 * x22
        da22 = -it1 * (ditau2 * x22 + itau2 * dx22)
        B = r2 * itau2 * itau2 * S * S
        dB = (dr2 * itau2 + 2.0 * r2 * ditau2) * itau2 * S * S + 2.0 * r2 * itau2 * itau2 * S * dS
        idet = _inv(a11 * a22 - B)
        ddet = da11 * a22 + a11 * da22 - dB
        cu_idet = complex(0.0, -fw[FW_K1, l]) * idet
        u = a22 * cu_idet
        W = -it1 * itau2 * S * rho
        v = W * cu_idet
        U[l] = u
        V[l] = v
        DU[l] = da22 * cu_idet - u * ddet * idet
        DV[l] = -it1 * ((ditau2 * S + itau2 * dS) * rho + itau2 * S * drho) * cu_idet - v * ddet * idet
    for l in range(nl):
        p = P[l]
        ip = IP[l]
        dp = -0.25j * p
        dip = 0.25j * ip
        rho = RHO[l]
        drho = DRHO[l]
        rc = rho.conjugate()
        drc = drho.conjugate()
        itau = fw[FW_ITAU, l]
        ditau = fw[FW_DITAU, l]
        u = U[l]
        v = V[l]
        du = DU[l]
        dv = DV[l]
        k1 = fw[FW_K1, l]
        k2 = fw[FW_K2, l]
        t2 = fw[FW_T2, l]
        res[RES_R, l] = k1 * k1 * _abs2(v)
        scw = _abs2(u)
        dscw = _dabs2(u, du)
        du = dp * u + p * du
        u = p * u
        dv = dv * ip + v * dip
        v = v * ip
        sccw = _abs2(v)
        dsccw = _dabs2(v, dv)
        nu = u - rc * v
        nv = v - rho * u
        du, dv = ((du - drc * v - rc * dv) * itau + nu * ditau, (dv - drho * u - rho * du) * itau + nv * ditau)
        u = nu * itau
        v = nv * itau
        scw += _abs2(u)
        dscw += _dabs2(u, du)
        du = dp * u + p * du
        u = p * u
        dv = dv * ip + v * dip
        v = v * ip
        sccw += _abs2(v)
        dsccw += _dabs2(v, dv)
        res[RES_D, l] = k2 * k2 * _abs2(u)
        du = t2 * du
        u = t2 * u
        dv = dv / t2
        v = v / t2
        res[RES_A, l] = k2 * k2 * _abs2(v)
        U[l] = u
        V[l] = v
        DU[l] = du
        DV[l] = dv
        fw[FW_SCW, l] = scw
        fw[FW_SCCW, l] = sccw
        fw[FW_DSCW, l] = dscw
        fw[FW_DSCCW, l] = dsccw
    for l in range(nl):
        p = P[l]
        ip = IP[l]
        dp = -0.25j * p
        dip = 0.25j * ip
        rho = RHO[l]
        drho = DRHO[l]
        rc = rho.conjugate()
        drc = drho.conjugate()
        itau = fw[FW_ITAU, l]
        ditau = fw[FW_DITAU, l]
        u = U[l]
        v = V[l]
        du = DU[l]
        dv = DV[l]
        scw = fw[FW_SCW, l] + _abs2(u)
        dscw = fw[FW_DSCW, l] + _dabs2(u, du)
        du = dp * u + p * du
        u = p * u
        dv = dv * ip + v * dip
        v = v * ip
        sccw = fw[FW_SCCW, l] + _abs2(v)
        dsccw = fw[FW_DSCCW, l] + _dabs2(v, dv)
        nu = u - rc * v
        nv = v - rho * u
        du, dv = ((du - drc * v - rc * dv) * itau + nu * ditau, (dv - drho * u - rho * du) * itau + nv * ditau)
        u = nu * itau
        v = nv * itau
        scw += _abs2(u)
        dscw += _dabs2(u, du)
        dv = dv * ip + v * dip
        u = p * u
        v = v * ip
        sccw += _abs2(v)
        dsccw += _dabs2(v, dv)
        res[RES_T, l] = _abs2(fw[FW_T1, l] - 1j * fw[FW_K1, l] * u)
        e_per, lossfrac = _seg_factors(fw[FW_XI, l], trt_row[l])
        res[RES_UCW, l] = scw * e_per
        res[RES_UCCW, l] = sccw * e_per
        res[RES_LOSS, l] = (scw + sccw) * lossfrac
        res[RES_DLOSS, l] = (dscw + dsccw) * lossfrac


@njit(cache=True, error_model="numpy", _nrt=False)
def ring_rows_nb(phi_row, psi_row, k2t_row, k2d_row, a_row, trt_row, rbk, use_tcmt, m_orders, with_d, cw, fw, res):
    """Unit-forward-input response of ONE ring at all L lines for one junction level.

    Args: (L,) rows of phi [rad], psi [rad], K^2_th, K^2_dr [-], a_rt [-], T_rt [s]; rbk (M_b,) complex backscatter;
    use_tcmt, m_orders: optics fidelity; with_d: also compute d(loss)/d(phi); cw, fw: workspace from
    :func:`make_workspace`; res: (N_RES, L) output block (rows RES_*), per watt of input power.
    Paths: TCMT (scalar per line, M orders); exact without backscatter (Airy closed form, exact); exact with
    backscatter (structure-of-arrays stages).
    """
    nl = phi_row.shape[0]
    if use_tcmt:
        for l in range(nl):
            ki, ke, kd = tcmt_rates(k2t_row[l], k2d_row[l], a_row[l], trt_row[l])
            r = _tcmt_line(phi_row[l] + psi_row[l], ki, ke, kd, trt_row[l], rbk, m_orders, 1.0 + 0j, 0j, 0j, 0j)
            res[RES_R, l] = _abs2(r[0])
            res[RES_T, l] = _abs2(r[1])
            res[RES_D, l] = _abs2(r[2])
            res[RES_A, l] = _abs2(r[3])
            res[RES_UCW, l] = r[4]
            res[RES_UCCW, l] = r[5]
            res[RES_LOSS, l] = r[6]
            res[RES_DLOSS, l] = r[7]
        return
    if not _has_backscatter(rbk):
        for l in range(nl):
            q = _airy_line(phi_row[l] + psi_row[l], -math.log(a_row[l]), math.sqrt(1.0 - k2t_row[l]),
                           math.sqrt(k2t_row[l]), math.sqrt(1.0 - k2d_row[l]), math.sqrt(k2d_row[l]), trt_row[l])
            res[RES_R, l] = 0.0
            res[RES_T, l] = q[1]
            res[RES_D, l] = q[2]
            res[RES_A, l] = 0.0
            res[RES_UCW, l] = q[4]
            res[RES_UCCW, l] = 0.0
            res[RES_LOSS, l] = q[6]
            res[RES_DLOSS, l] = q[7] if with_d else 0.0
        return
    for l in range(nl):
        xi = -math.log(a_row[l])
        fw[FW_XI, l] = xi
        fw[FW_T1, l] = math.sqrt(1.0 - k2t_row[l])
        fw[FW_K1, l] = math.sqrt(k2t_row[l])
        fw[FW_T2, l] = math.sqrt(1.0 - k2d_row[l])
        fw[FW_K2, l] = math.sqrt(k2d_row[l])
        p, ip, rho, drho = _phase_parts(phi_row[l] + psi_row[l], rbk, xi)
        cw[CW_P, l] = p
        cw[CW_IP, l] = ip
        cw[CW_RHO, l] = rho
        cw[CW_DRHO, l] = drho
    if with_d:
        _stages_d(nl, cw, fw, trt_row, res)
    else:
        _stages_v(nl, cw, fw, trt_row, res)


# =====================================================================================================================
# Single-ring, all-lines kernels (used by the bus and by the implicit self-heating solve)
# =====================================================================================================================

@njit(cache=True, error_model="numpy", _nrt=False)
def ring_lines_nb(k, pin, p1, phi0, phi1, a0, a1, eta0, eta1, psi, k2_th, k2_dr, t_rt, rb, dphi_dlamres,
                  two_level, use_tcmt, m_orders, with_slope, out_rl, out_r, p_thru, cw, fw):
    """Solve ring k at every line for a given input power per line, and write its outputs (no allocation).

    Args:
        k: ring index (row of every (N, L) array).
        pin: (L,) power at ring k's input port [W] (after the bus segment).
        p1: ones density of ring k's data [-] (ignored when two_level is False).
        phi0, phi1: (N, L) round-trip phase at levels 0 and 1, without the coupler phase [rad].
        a0, a1: (N, L) round-trip amplitude at levels 0 and 1 [-].
        eta0, eta1: (N, L) absorbed fraction of the intrinsic loss at levels 0 and 1 (07 P0-2) [-].
        psi: (N, L) ring-side coupler phase (both couplers) [rad]. k2_th, k2_dr: (N, L) per-pass power couplings
            at the LINE wavelength [-]. t_rt: (N, L) round-trip group delay [s]. rb: (N, M_b) complex round-trip
            backscatter per order [-]. dphi_dlamres: (N, L) [rad/m] from :func:`dphi_dlambda_res` (read only when
            with_slope).
        two_level: solve both NRZ levels and average with p1 (else level 0 only). use_tcmt: TCMT instead of the
            exact ring. m_orders: TCMT orders. with_slope: also fill RL_DABS_DPHI and R_DABS_DLAM.
        out_rl: (N_RL, N, L); row k is written (RL_POUT1/0 are filled by :func:`bus_finalize_nb`).
        out_r: (N_R, N); column k is written.
        p_thru: (L,) receives the NRZ-averaged through power [W] (may be the same array as pin).
        cw, fw: workspace from :func:`make_workspace`.
    """
    nl = pin.shape[0]
    rbk = rb[k]
    res0 = fw[FW_RES0:FW_RES0 + N_RES]
    res1 = fw[FW_RES1:FW_RES1 + N_RES]
    ring_rows_nb(phi0[k], psi[k], k2_th[k], k2_dr[k], a0[k], t_rt[k], rbk, use_tcmt, m_orders, with_slope,
                 cw, fw, res0)
    if two_level:
        ring_rows_nb(phi1[k], psi[k], k2_th[k], k2_dr[k], a1[k], t_rt[k], rbk, use_tcmt, m_orders, with_slope,
                     cw, fw, res1)
        q1 = p1
    else:
        q1 = 0.0
    q0 = 1.0 - q1
    dsum = 0.0
    asum = 0.0
    lsum = 0.0
    d1sum = 0.0
    d0sum = 0.0
    slope = 0.0
    for l in range(nl):
        P = pin[l]
        if two_level:
            T1 = res1[RES_T, l]
            D1 = res1[RES_D, l]
            A1 = res1[RES_A, l]
            R1 = res1[RES_R, l]
            L1 = res1[RES_LOSS, l]
            Uc1 = res1[RES_UCW, l]
            Ucc1 = res1[RES_UCCW, l]
            dL1 = res1[RES_DLOSS, l]
            e1 = eta1[k, l]
        else:
            T1 = res0[RES_T, l]
            D1 = res0[RES_D, l]
            A1 = res0[RES_A, l]
            R1 = res0[RES_R, l]
            L1 = res0[RES_LOSS, l]
            Uc1 = res0[RES_UCW, l]
            Ucc1 = res0[RES_UCCW, l]
            dL1 = res0[RES_DLOSS, l]
            e1 = eta0[k, l]
        T0 = res0[RES_T, l]
        D0 = res0[RES_D, l]
        e0 = eta0[k, l]
        out_rl[RL_PIN, k, l] = P
        out_rl[RL_THRU, k, l] = P * (q1 * T1 + q0 * T0)
        out_rl[RL_DROP, k, l] = P * (q1 * D1 + q0 * D0)
        out_rl[RL_ADD, k, l] = P * (q1 * A1 + q0 * res0[RES_A, l])
        out_rl[RL_REFL, k, l] = P * (q1 * R1 + q0 * res0[RES_R, l])
        out_rl[RL_LOSS, k, l] = P * (q1 * L1 + q0 * res0[RES_LOSS, l])
        out_rl[RL_ABS, k, l] = P * (q1 * e1 * L1 + q0 * e0 * res0[RES_LOSS, l])
        out_rl[RL_UCW, k, l] = P * (q1 * Uc1 + q0 * res0[RES_UCW, l])
        out_rl[RL_UCCW, k, l] = P * (q1 * Ucc1 + q0 * res0[RES_UCCW, l])
        out_rl[RL_THRU1, k, l] = P * T1
        out_rl[RL_THRU0, k, l] = P * T0
        out_rl[RL_DROP1, k, l] = P * D1
        out_rl[RL_DROP0, k, l] = P * D0
        if with_slope:
            g = P * (q1 * e1 * dL1 + q0 * e0 * res0[RES_DLOSS, l])
            out_rl[RL_DABS_DPHI, k, l] = g
            slope += g * dphi_dlamres[k, l]
        else:
            out_rl[RL_DABS_DPHI, k, l] = 0.0
        dsum += out_rl[RL_DROP, k, l]
        asum += out_rl[RL_ABS, k, l]
        lsum += out_rl[RL_LOSS, k, l]
        d1sum += P * D1
        d0sum += P * D0
        p_thru[l] = out_rl[RL_THRU, k, l]
    out_r[R_DROP_SUM, k] = dsum
    out_r[R_ABS_SUM, k] = asum
    out_r[R_LOSS_SUM, k] = lsum
    out_r[R_DROP_SUM1, k] = d1sum
    out_r[R_DROP_SUM0, k] = d0sum
    out_r[R_DABS_DLAM, k] = slope


@njit(cache=True, error_model="numpy", _nrt=False)
def dpabs_dlambda_res_nb(k, pin, p1, phi0, phi1, a0, a1, eta0, eta1, psi, k2_th, k2_dr, t_rt, rb,
                         dphi_dlamres, two_level, use_tcmt, m_orders, cw, fw):
    """Absorbed power of ring k and its slope with respect to a rigid shift of ring k's resonance (P0-1 Newton step).

    P_abs,k = sum_l pin_l [p1 eta1 loss1 + (1 - p1) eta0 loss0]  (all lines, both directions, all orders);
    dP_abs,k/d lambda_res = sum_l pin_l [p1 eta1 dloss1/dphi + (1 - p1) eta0 dloss0/dphi] dphi_dlamres[k, l].
    The input pin (L,) [W] is held fixed (set by the rings before k in bus order, 07 P0-1). Coupling, loss and eta
    are held fixed in the slope: their temperature dependence (~1e-3/K) is small against the phase term (~0.4/K of
    P_abs near resonance), so the Newton step stays quasi-exact while the residual is always evaluated exactly.
    Args as :func:`ring_lines_nb`. Returns (P_abs [W], dP_abs/d lambda_res [W/m]); analytic forward-mode slope for
    the exact ring (with or without backscatter) and for the TCMT. :func:`ring_lines_nb` with with_slope=True gives
    the same slope together with every other output in one pass.
    """
    nl = pin.shape[0]
    rbk = rb[k]
    res0 = fw[FW_RES0:FW_RES0 + N_RES]
    res1 = fw[FW_RES1:FW_RES1 + N_RES]
    ring_rows_nb(phi0[k], psi[k], k2_th[k], k2_dr[k], a0[k], t_rt[k], rbk, use_tcmt, m_orders, True, cw, fw, res0)
    q1 = 0.0
    if two_level:
        ring_rows_nb(phi1[k], psi[k], k2_th[k], k2_dr[k], a1[k], t_rt[k], rbk, use_tcmt, m_orders, True,
                     cw, fw, res1)
        q1 = p1
    q0 = 1.0 - q1
    pabs = 0.0
    slope = 0.0
    for l in range(nl):
        ab = q0 * eta0[k, l] * res0[RES_LOSS, l]
        dab = q0 * eta0[k, l] * res0[RES_DLOSS, l]
        if two_level:
            ab += q1 * eta1[k, l] * res1[RES_LOSS, l]
            dab += q1 * eta1[k, l] * res1[RES_DLOSS, l]
        pabs += pin[l] * ab
        slope += pin[l] * dab * dphi_dlamres[k, l]
    return pabs, slope


@njit(cache=True, error_model="numpy", _nrt=False)
def pabs_bounds_nb(k, pin, p1, a0, a1, eta0, eta1, k2_th, k2_dr, two_level):
    """Upper bounds of ring k's absorbed power and of its curvature in the round-trip phase, at fixed input.

    Needed by the curvature-certified Newton step of mrm.plant.selfheat (p_max, d2p_max). Per line, without
    backscatter the loss fraction is C/den(phi), C = k1^2 (1 - a)(1 + t2^2 a), den = 1 - 2 r cos(phi) + r^2,
    r = t1 t2 a (the Airy forms), with maximum C/(1 - r)^2 at resonance and max |d^2/dphi^2| = 2 r C/(1 - r)^4 (also
    at resonance). Backscatter splits the resonance into two half-weight Lorentzians of the same width (03 section
    2.5: U = (kappa_e/2) sum_+- 1/((Delta -+ |g|)^2 + kappa^2/4)), which lowers both the peak and the curvature; in
    the exact ring that holds to O(1/F^2), so margins of 5 % (peak) and 25 % (curvature) are applied (verified over
    random rings, backscatter included, in tests/verification/test_optics.py). The peak per line is also capped at
    the loss fraction 1 (P_loss <= P_in, energy conservation). Lines add (triangle inequality); NRZ levels are
    averaged with p1.
    Validity domain: |r_b| <= RB_MAX_BOUNDS = 1 for every order (beta <= FSR/2 pi, about 290 GHz at 1.8 THz, against
    1 to 10 GHz in 07 P2-2). Beyond it the scatterer pair (|rho| = |r_b|/2 > 0.5) reshapes the response and the
    curvature can exceed the bound (at |r_b| = 1.6 it is 1.35 times the unsplit curvature, 8 % above d2p_max,
    independent of Q; test_optics).
    Returns (p_max [W], d2p_max [W/rad^2]): multiply d2p_max by (d phi/d T)^2 for d2P/dT2 (the d2phi/dT2 term is
    ~1e-3 of it, inside the margin).
    """
    nl = pin.shape[0]
    q1 = p1 if two_level else 0.0
    pm = 0.0
    cm = 0.0
    for lev in range(2):
        w = (1.0 - q1) if lev == 0 else q1
        if w == 0.0:
            continue
        for l in range(nl):
            a = a0[k, l] if lev == 0 else a1[k, l]
            eta = eta0[k, l] if lev == 0 else eta1[k, l]
            t1 = math.sqrt(1.0 - k2_th[k, l])
            t2 = math.sqrt(1.0 - k2_dr[k, l])
            r = t1 * t2 * a
            om = 1.0 - r
            c = k2_th[k, l] * (1.0 - a) * (1.0 + t2 * t2 * a)
            pm += w * pin[l] * eta * min(1.05 * c / (om * om), 1.0)
            cm += w * pin[l] * eta * 2.0 * r * c / (om * om * om * om)
    return pm, 1.25 * cm


RB_MAX_BOUNDS = 1.0  # |r_b| up to which pabs_bounds_nb / pabs_slope_bound_nb are verified (test_optics; derived)


@njit(cache=True, error_model="numpy", _nrt=False)
def pabs_slope_bound_nb(k, pin, p1, a0, a1, eta0, eta1, k2_th, k2_dr, two_level):
    """Upper bound of |dP_abs/d phi| of ring k at fixed input [W/rad] (multiply by |d phi/dT| for W/K).

    Needed for the cross-ring optical-heating lag check of 07 P0-1 (mrm.plant.thermal.cross_optical_lag_gain takes
    a bound of |dP_abs/dT| per ring). Per line, without backscatter, the loss fraction is f = C/den(phi),
    den = A - 2 r cos(phi), A = 1 + r^2 (the Airy forms of :func:`pabs_bounds_nb`), so
    |f'| = 2 r C |sin phi| / den^2, whose maximum is at cos(phi*) = (sqrt(A^2 + 32 r^2) - A)/(4 r) (root of
    2 r c^2 + A c - 4 r = 0; derived here). In the Lorentzian limit this is (3 sqrt 3/8) peak/HWHM. Backscatter
    splits the line into two half-weight Lorentzians, whose summed slope cannot exceed the unsplit maximum
    (triangle inequality); a 10 % margin covers the O(1/F^2) exact-ring corrections (verified for |r_b| <=
    RB_MAX_BOUNDS in tests/verification/test_optics.py). Lines add; NRZ levels are averaged with p1.
    """
    nl = pin.shape[0]
    q1 = p1 if two_level else 0.0
    sm = 0.0
    for lev in range(2):
        w = (1.0 - q1) if lev == 0 else q1
        if w == 0.0:
            continue
        for l in range(nl):
            a = a0[k, l] if lev == 0 else a1[k, l]
            eta = eta0[k, l] if lev == 0 else eta1[k, l]
            t1 = math.sqrt(1.0 - k2_th[k, l])
            t2 = math.sqrt(1.0 - k2_dr[k, l])
            r = t1 * t2 * a
            if r <= 0.0:
                continue
            c = k2_th[k, l] * (1.0 - a) * (1.0 + t2 * t2 * a)
            aa = 1.0 + r * r
            cs = (math.sqrt(aa * aa + 32.0 * r * r) - aa) / (4.0 * r)
            sn = math.sqrt(max(1.0 - cs * cs, 0.0))
            den = aa - 2.0 * r * cs
            sm += w * pin[l] * eta * 2.0 * r * c * sn / (den * den)
    return 1.10 * sm


@njit(cache=True, _nrt=False)
def bus_finalize_nb(p_in_w, bus_order, bus_t2, bus_t2_out, out_rl, out_l):
    """After every ring has been solved: bus output, per-level outputs, reflection return and the single-pass bound.

    Fills out_rl[RL_POUT1/RL_POUT0] (ring k fixed at a level, all other rings NRZ-averaged; exact for independent
    data because the single-pass power transfer is a product), out_l[L_POUT], out_l[L_REFL_IN] and out_l[L_BOUND]
    (module docstring).

    L_REFL_IN is the incoherent sum of single reflections returned to the bus input. By reciprocity the backward
    through transmission of a ring equals its forward one (S_01 = S_10), and the backward path crosses the same bus
    segments, so ring k's return is P_in E[R_k] prod_{j<=k} t2_j^2 prod_{j<k} E[T_j^2]. The light crosses ring j
    twice within one data state, so the mean of the SQUARE of T_j enters. With p1 the ones density,
    E[(P T)^2] = p1 THRU1^2 + (1 - p1) THRU0^2 = THRU0^2 + (THRU - THRU0)(THRU1 + THRU0) (because
    THRU = p1 THRU1 + (1 - p1) THRU0), so p1 is not needed here. Exact for independent data on different rings.
    With CW optics it reduces to refl_k * pin_k / P_in.
    With NRZ the bound uses the RMS reflection amplitude sqrt(E[R]) of each ring. To leading order the error is
    linear in the pair products and E|r_i||r_j| <= sqrt(E r_i^2) sqrt(E r_j^2) for independent data, so it bounds the
    data-averaged error to leading order in eps; it is not a worst case over data states.
    """
    _bus_pout_nb(p_in_w, bus_order, bus_t2, bus_t2_out, out_rl, out_l)
    _bus_refl_nb(p_in_w, bus_order, bus_t2, out_rl, out_l)


@njit(cache=True, _nrt=False)
def _bus_pout_nb(p_in_w, bus_order, bus_t2, bus_t2_out, out_rl, out_l):
    """The bus-output part of :func:`bus_finalize_nb`: RL_POUT1 / RL_POUT0 of every ring and line and L_POUT."""
    n = bus_order.shape[0]
    nl = p_in_w.shape[0]
    for l in range(nl):
        # suffix product: averaged power transfer from just after ring at position pos to the bus output
        suffix = bus_t2_out
        for pos in range(n - 1, -1, -1):
            k = bus_order[pos]
            pk = out_rl[RL_PIN, k, l]
            if pk > 0.0:
                out_rl[RL_POUT1, k, l] = out_rl[RL_THRU1, k, l] * suffix
                out_rl[RL_POUT0, k, l] = out_rl[RL_THRU0, k, l] * suffix
                tbar = out_rl[RL_THRU, k, l] / pk
            else:
                out_rl[RL_POUT1, k, l] = 0.0
                out_rl[RL_POUT0, k, l] = 0.0
                tbar = 0.0
            suffix *= tbar * bus_t2[k]
        last = bus_order[n - 1]
        out_l[L_POUT, l] = out_rl[RL_THRU, last, l] * bus_t2_out


@njit(cache=True, _nrt=False)
def _bus_refl_nb(p_in_w, bus_order, bus_t2, out_rl, out_l):
    """The reflection part of :func:`bus_finalize_nb`: L_REFL_IN and L_BOUND (reads no output of
    :func:`_bus_pout_nb`, so the two parts are independent)."""
    n = bus_order.shape[0]
    nl = p_in_w.shape[0]
    for l in range(nl):
        # reflection returned to the input (E[T^2] for the double crossing), and the double-reflection bound
        ret = 0.0
        tot = 0.0                     # sum of |r_k|
        sq = 0.0                      # sum of |r_k|^2
        g2 = 1.0                      # mean round-trip power factor, bus input -> ring k's input -> bus input
        for pos in range(n):
            k = bus_order[pos]
            pk = out_rl[RL_PIN, k, l]
            g2 *= bus_t2[k] * bus_t2[k]
            if pk > 0.0 and p_in_w[l] > 0.0:
                refl = out_rl[RL_REFL, k, l]
                if refl == 0.0:
                    # rr = 0 / pk = 0: no division and no square root (the same sums, bit for bit)
                    ret += 0.0 * p_in_w[l] * g2
                else:
                    rr = refl / pk
                    ret += rr * p_in_w[l] * g2
                    tot += math.sqrt(rr)
                    sq += rr
                th = out_rl[RL_THRU, k, l]
                th0 = out_rl[RL_THRU0, k, l]
                th1 = out_rl[RL_THRU1, k, l]
                g2 *= (th0 * th0 + (th - th0) * (th1 + th0)) / (pk * pk)
            else:
                g2 = 0.0
        out_l[L_REFL_IN, l] = ret
        eps = 0.5 * (tot * tot - sq)  # sum over all pairs i < j of |r_i| |r_j| (module docstring)
        out_l[L_BOUND, l] = 1.0 / ((1.0 - eps) * (1.0 - eps)) - 1.0 if eps < 1.0 else np.inf


@njit(cache=True, error_model="numpy", _nrt=False)
def bus_propagate_nb(p_in_w, bus_order, bus_t2, bus_t2_out, p1, phi0, phi1, a0, a1, eta0, eta1, psi,
                     k2_th, k2_dr, t_rt, rb, dphi_dlamres, two_level, use_tcmt, m_orders, with_slope,
                     out_rl, out_r, out_l, work, cw, fw):
    """N rings in bus_order acting on L lines, single pass (compiled-kernel entry point; no allocation).

    Args:
        p_in_w: (L,) power per line at the bus input [W].
        bus_order: (N,) int64 ring indices in the order the light meets them.
        bus_t2: (N,) bus power transmission of the segment just BEFORE ring k (from the previous ring in bus order,
            or from the bus input) [-]; bus_t2_out: power transmission after the last ring [-].
        p1: (N,) ones density per ring [-]. Other arrays and flags as :func:`ring_lines_nb`.
        out_rl (N_RL, N, L), out_r (N_R, N), out_l (N_L, L): outputs (:func:`make_bus_outputs`; field constants).
        work (L,), cw, fw: workspace (:func:`make_workspace`).
    The implicit self-heating solve (07 P0-1) does not call this; it walks bus_order itself, calling
    :func:`dpabs_dlambda_res_nb` (or :func:`ring_lines_nb` with with_slope) for ring k, then :func:`ring_lines_nb`
    to pass ring k's through power on, and :func:`bus_finalize_nb` at the end.
    """
    nl = p_in_w.shape[0]
    for l in range(nl):
        work[l] = p_in_w[l]
    for pos in range(bus_order.shape[0]):
        k = bus_order[pos]
        for l in range(nl):
            work[l] *= bus_t2[k]
        ring_lines_nb(k, work, p1[k], phi0, phi1, a0, a1, eta0, eta1, psi, k2_th, k2_dr, t_rt, rb, dphi_dlamres,
                      two_level, use_tcmt, m_orders, with_slope, out_rl, out_r, work, cw, fw)
    bus_finalize_nb(p_in_w, bus_order, bus_t2, bus_t2_out, out_rl, out_l)


# =====================================================================================================================
# Self-heating adapter (07 P0-1): pabs_fn / commit_fn for mrm.plant.selfheat, every input packed in args_f / args_i
# =====================================================================================================================
# mrm.plant.selfheat.implicit_step_b calls pabs_fn(t, k, args_f, args_i) -> (P_abs W, dP_abs/dT W/K) during ring k's
# Newton solve and commit_fn(t, k, args_f, args_i) once ring k is solved, in bus order (docs/handoff/plant_thermal.md
# section 3). :func:`selfheat_pabs` and :func:`selfheat_commit` are those functions for the exact ring (or the TCMT),
# with a rigid thermo-optic shift of each ring's comb: lambda_res,k(T) - lambda_res,k(T_ref) =
# s0_k [dT + beta_k dT^2/2], dT = T - T_ref,k (07 P0-3 R1, the law of selfheat.pabs_lorentzian), so the phase at line l
# is phi_kl(T) = phi_kl(T_ref) + (dphi/dlambda_res)_kl * that shift (:func:`dphi_dlambda_res`). K^2, a_rt and eta are
# held at the values the plant writes for the substep (evaluate a_rt and eta at the previous ring temperature: their
# ~1e-3/K dependence then lags one substep, which keeps dP_abs/dT exact for the solver's no-skip guarantee).
# Layout: args_i holds the sizes, flags and block offsets below, then bus_order (N,) and the bus position of every ring
# (N,). args_f holds the blocks; all (N, L) fields form ONE contiguous (N_SHN, N, L) block and the per-ring fields one
# (N_SHR, N) block, so an evaluation builds a handful of views (a numba reshape costs ~27 ns, a slice ~1 ns).
# :func:`pack_selfheat_args` builds one world's rows (stack rows with np.stack for a batch: every row of a bank has the
# same layout, and the row length is even so the complex views stay 16-byte aligned).
SH_N = 0          # number of rings N
SH_L = 1          # number of lines L
SH_MB = 2         # backscatter orders M_b
SH_TWO = 3        # 1: NRZ two-level average
SH_TCMT = 4       # 1: TCMT instead of the exact ring
SH_MORD = 5       # TCMT orders
SH_SLOPE = 6      # 1: selfheat_commit also fills RL_DABS_DPHI / R_DABS_DLAM (about 2x the commit cost)
SH_O_PIN = 7      # p_in_w (L,) power per line at the bus input [W]
SH_O_PCUR = 8     # walk state: input power of the ring being solved (L,) [W]
SH_O_T2OUT = 9    # bus_t2_out (1,) [-]
SH_O_RING = 10    # per-ring block (N_SHR, N), rows SHR_*
SH_O_NL = 11      # per-(ring, line) block (N_SHN, N, L), rows SHN_*
SH_O_RB = 12      # backscatter r_b (N, M_b) complex, stored as 2 N M_b float64 [-]
SH_O_ORL = 13     # out_rl (N_RL, N, L)
SH_O_OR = 14      # out_r (N_R, N)
SH_O_OL = 15      # out_l (N_L, L)
SH_O_CW = 16      # complex workspace (N_CW, L), stored as 2 N_CW L float64
SH_O_FW = 17      # float workspace (N_FW, L)
SH_O_FK = 18      # Airy-path cache per (ring, line) (N, L, N_FK): K^2 keys and coupler amplitudes
SH_O_FC = 19      # Airy-path cache per (level, ring, line) (2, N, L, N_FC): input keys, cos/sin of the reference phase
                  # and the per-substep constants
SH_O_FF = 20      # Airy-path flag per ring (N,): 1.0 exact ring without backscatter (the hoisted path), else 0.0;
                  # then 8 floats of scratch (the memo of _afac)
SH_O_LE = 21      # last Airy evaluation: per (level, line) N_LE values (2, L, N_LE), then (ring, temperature, shift)
SH_O_FH = 22      # far-line hold per (level, ring, line) (2, N, L, N_FH) (fidelity optics_far_hold)
SH_FARHOLD = 23   # 1: hold far lines to first order between refreshes (pack_selfheat_args(far_hold=...))
SH_HDR = 24       # args_i[SH_HDR:SH_HDR+N] = bus_order; args_i[SH_HDR+N:SH_HDR+2N] = bus position of each ring
# rows of the per-ring block
SHR_T2 = 0        # bus power transmission of the segment before the ring [-]
SHR_P1 = 1        # ones density [-]
SHR_TREF = 2      # T_ref of the phase rows [K]
SHR_S0 = 3        # s0 = dlambda_res/dT at T_ref [m/K]
SHR_BETA = 4      # relative curvature beta of the thermo-optic law [1/K]
N_SHR = 5
# rows of the per-(ring, line) block
SHN_PHI0R = 0     # round-trip phase at T_ref, level 0, without the coupler phase [rad]
SHN_PHI1R = 1     # same, level 1 [rad]
SHN_PHI0 = 2      # scratch: phase at the trial / committed temperature, level 0 [rad]
SHN_PHI1 = 3      # same, level 1 [rad]
SHN_A0 = 4        # round-trip amplitude, level 0 [-]
SHN_A1 = 5        # level 1 [-]
SHN_E0 = 6        # heat fraction eta, level 0 [-]
SHN_E1 = 7        # level 1 [-]
SHN_PSI = 8       # ring-side coupler phase [rad]
SHN_K2T = 9       # K^2 through at the line [-]
SHN_K2D = 10      # K^2 drop at the line [-]
SHN_TRT = 11      # round-trip group delay [s]
SHN_DPL = 12      # dphi/dlambda_res [rad/m]
N_SHN = 13


@njit(cache=True, inline="always")
def _sh_core(args_f, args_i):
    """The views one evaluation needs: (p_in (L,), p_cur (L,), ring block (N_SHR, N), line block (N_SHN, N, L),
    rb (N, M_b) complex, cw (N_CW, L) complex, fw (N_FW, L))."""
    n = args_i[SH_N]
    nl = args_i[SH_L]
    mb = args_i[SH_MB]
    o = args_i[SH_O_PIN]
    p_in = args_f[o:o + nl]
    o = args_i[SH_O_PCUR]
    pcur = args_f[o:o + nl]
    o = args_i[SH_O_RING]
    rg = args_f[o:o + N_SHR * n].reshape((N_SHR, n))
    o = args_i[SH_O_NL]
    nb = args_f[o:o + N_SHN * n * nl].reshape((N_SHN, n, nl))
    o = args_i[SH_O_RB]
    rb = args_f[o:o + 2 * n * mb].view(np.complex128).reshape((n, mb))
    o = args_i[SH_O_CW]
    cw = args_f[o:o + 2 * N_CW * nl].view(np.complex128).reshape((N_CW, nl))
    o = args_i[SH_O_FW]
    fw = args_f[o:o + N_FW * nl].reshape((N_FW, nl))
    return p_in, pcur, rg, nb, rb, cw, fw


@njit(cache=True, inline="always")
def _sh_outputs(args_f, args_i):
    """Output views (out_rl (N_RL, N, L), out_r (N_R, N), out_l (N_L, L)) of packed self-heating rows."""
    n = args_i[SH_N]
    nl = args_i[SH_L]
    o = args_i[SH_O_ORL]
    orl = args_f[o:o + N_RL * n * nl].reshape((N_RL, n, nl))
    o = args_i[SH_O_OR]
    orr = args_f[o:o + N_R * n].reshape((N_R, n))
    o = args_i[SH_O_OL]
    ol = args_f[o:o + N_L * nl].reshape((N_L, nl))
    return orl, orr, ol


@njit(cache=True, inline="always")
def selfheat_views(args_f, args_i):
    """Views into packed self-heating rows (no copy): the plant writes its per-substep inputs and reads outputs here.

    Returns (p_in (L,), bus_t2 (N,), bus_t2_out (1,), p1 (N,), phi0_ref, phi1_ref, phi0, phi1, a0, a1, eta0, eta1,
    psi, k2_th, k2_dr, t_rt, dphi_dlamres (each (N, L)), t_ref (N,), s0 (N,), beta (N,), rb (N, M_b) complex,
    p_cur (L,), out_rl (N_RL, N, L), out_r (N_R, N), out_l (N_L, L), cw, fw, bus_order (N,), bus_pos (N,)).
    Units as the SHR_* / SHN_* / SH_O_* comments.
    """
    p_in, pcur, rg, nb, rb, cw, fw = _sh_core(args_f, args_i)
    orl, orr, ol = _sh_outputs(args_f, args_i)
    n = args_i[SH_N]
    o = args_i[SH_O_T2OUT]
    return (p_in, rg[SHR_T2], args_f[o:o + 1], rg[SHR_P1], nb[SHN_PHI0R], nb[SHN_PHI1R], nb[SHN_PHI0], nb[SHN_PHI1],
            nb[SHN_A0], nb[SHN_A1], nb[SHN_E0], nb[SHN_E1], nb[SHN_PSI], nb[SHN_K2T], nb[SHN_K2D], nb[SHN_TRT],
            nb[SHN_DPL], rg[SHR_TREF], rg[SHR_S0], rg[SHR_BETA], rb, pcur, orl, orr, ol, cw, fw,
            args_i[SH_HDR:SH_HDR + n], args_i[SH_HDR + n:SH_HDR + 2 * n])


@njit(cache=True, _nrt=False)
def _sh_prepare(t, k, pos_k, p_in, pcur, rg, nb):
    """Set ring k's phase rows at temperature t [K] and, for the first ring in bus order (pos_k = 0), its input power.

    Returns d lambda_res / dT at t [m/K] (07 P0-3 R1: s0 (1 + beta dT)).
    """
    dt = t - rg[SHR_TREF, k]
    s0 = rg[SHR_S0, k]
    beta = rg[SHR_BETA, k]
    shift = s0 * (dt + 0.5 * beta * dt * dt)
    for l in range(nb.shape[2]):
        dpl = nb[SHN_DPL, k, l]
        nb[SHN_PHI0, k, l] = nb[SHN_PHI0R, k, l] + dpl * shift
        nb[SHN_PHI1, k, l] = nb[SHN_PHI1R, k, l] + dpl * shift
    if pos_k == 0:                       # the walk (re)starts at the first ring in bus order
        t2 = rg[SHR_T2, k]
        for l in range(pcur.shape[0]):
            pcur[l] = p_in[l] * t2
    return s0 * (1.0 + beta * dt)


@njit(cache=True, _nrt=False)
def selfheat_pabs(t, k, args_f, args_i):
    """pabs_fn for mrm.plant.selfheat: absorbed power of ring k at ring temperature t [K], and dP_abs/dT [W/K].

    P_abs,k sums every line, both directions and every order (exact ring) at the input left by the rings before k in
    bus order (07 P0-1). dP_abs/dT = (dP_abs/dlambda_res)(dlambda_res/dT), analytic (:func:`dpabs_dlambda_res_nb`).
    """
    if _fast_flag(k, args_f, args_i) != 0.0:
        _fast_check(k, args_f, args_i)
        return sh_fast_pabs(t, k, args_f, args_i)
    p_in, pcur, rg, nb, rb, cw, fw = _sh_core(args_f, args_i)
    n = args_i[SH_N]
    dlam_dt = _sh_prepare(t, k, args_i[SH_HDR + n + k], p_in, pcur, rg, nb)
    pabs, slope = dpabs_dlambda_res_nb(k, pcur, rg[SHR_P1, k], nb[SHN_PHI0], nb[SHN_PHI1], nb[SHN_A0], nb[SHN_A1],
                                       nb[SHN_E0], nb[SHN_E1], nb[SHN_PSI], nb[SHN_K2T], nb[SHN_K2D], nb[SHN_TRT],
                                       rb, nb[SHN_DPL], args_i[SH_TWO] != 0, args_i[SH_TCMT] != 0, args_i[SH_MORD],
                                       cw, fw)
    return pabs, slope * dlam_dt


@njit(cache=True, _nrt=False)
def selfheat_commit(t, k, args_f, args_i):
    """commit_fn for mrm.plant.selfheat: write every output of ring k at its solved temperature t [K] into the packed
    out_rl / out_r, pass its through power to the next ring in bus order, and after the last ring fill out_l
    (:func:`bus_finalize_nb`). The walk restarts by itself at the first ring, so Gauss-Seidel sweeps (n_outer > 1)
    and later substeps need no reset. Airy-path rings (exact ring, no backscatter): the outputs of the evaluation at t
    (the last one when it was ring k at t, else a new one), with the slope outputs always filled."""
    if _fast_flag(k, args_f, args_i) != 0.0:
        if not sh_last_is(k, t, args_f, args_i):
            _fast_check(k, args_f, args_i)
            sh_fast_pabs(t, k, args_f, args_i)
        sh_fast_outputs(k, args_f, args_i)
        sh_pass_on(k, args_f, args_i)
        sh_last_clear(args_f, args_i)
        return
    p_in, pcur, rg, nb, rb, cw, fw = _sh_core(args_f, args_i)
    orl, orr, ol = _sh_outputs(args_f, args_i)
    n = args_i[SH_N]
    q = args_i[SH_HDR + n + k]
    _sh_prepare(t, k, q, p_in, pcur, rg, nb)
    ring_lines_nb(k, pcur, rg[SHR_P1, k], nb[SHN_PHI0], nb[SHN_PHI1], nb[SHN_A0], nb[SHN_A1], nb[SHN_E0],
                  nb[SHN_E1], nb[SHN_PSI], nb[SHN_K2T], nb[SHN_K2D], nb[SHN_TRT], rb, nb[SHN_DPL],
                  args_i[SH_TWO] != 0, args_i[SH_TCMT] != 0, args_i[SH_MORD], args_i[SH_SLOPE] != 0, orl, orr, pcur,
                  cw, fw)
    if q + 1 < n:
        t2 = rg[SHR_T2, args_i[SH_HDR + q + 1]]
        for l in range(pcur.shape[0]):
            pcur[l] *= t2
    else:
        o = args_i[SH_O_T2OUT]
        bus_finalize_nb(p_in, args_i[SH_HDR:SH_HDR + n], rg[SHR_T2], args_f[o], orl, ol)


@njit(cache=True, _nrt=False)
def selfheat_bounds(args_f, args_i, dt_max_k, p_max, d2p_max, dpdt_max):
    """Bounds that mrm.plant.selfheat and mrm.plant.thermal need, per ring, valid for any state of the other rings.

    Ring k's input is at most p_in prod(bus_t2 up to k) (rings only attenuate the forward light), and the phase
    bounds of :func:`pabs_bounds_nb` and :func:`pabs_slope_bound_nb` convert to temperature with
    dphi/dT = (dphi/dlambda_res) s0 (1 + beta dT) and d2phi/dT2 = (dphi/dlambda_res) s0 beta, |dT| <= dt_max_k [K]:
        p_max [W];  dpdt_max = dp_max_phi * D * S [W/K];  d2p_max = d2p_max_phi (D S)^2 + dp_max_phi D |s0 beta|
    [W/K^2], with D = max_l |dphi/dlambda_res| and S = |s0| (1 + |beta| dt_max_k). Fills p_max, d2p_max, dpdt_max (N,).
    Valid for |r_b| <= RB_MAX_BOUNDS. Uses the walk buffer as scratch: call it between substeps, not inside a walk.
    Brings the Airy-path caches up to date first (:func:`_fast_check`), then :func:`sh_bounds`.
    """
    n = args_i[SH_N]
    for k in range(n):
        if _fast_flag(k, args_f, args_i) != 0.0:
            _fast_check(k, args_f, args_i)
    sh_bounds(args_f, args_i, dt_max_k, p_max, d2p_max, dpdt_max)


@njit(cache=True, _nrt=False)
def selfheat_pabs_general(t, k, args_f, args_i):
    """:func:`selfheat_pabs` without the Airy path: every ring through the general kernels
    (:func:`dpabs_dlambda_res_nb`, i.e. the closed forms of :func:`_airy_line` for a ring without backscatter). The
    same P_abs and slope as the Airy path up to floating-point rounding, from an independent implementation: the
    reference engine runs it when asked for an independent check of the engine's self-heating optics
    (mrm.sim.kernel_ref.ReferenceEngine(airy=False), tests/verification/test_kernel_parity.py)."""
    p_in, pcur, rg, nb, rb, cw, fw = _sh_core(args_f, args_i)
    n = args_i[SH_N]
    dlam_dt = _sh_prepare(t, k, args_i[SH_HDR + n + k], p_in, pcur, rg, nb)
    pabs, slope = dpabs_dlambda_res_nb(k, pcur, rg[SHR_P1, k], nb[SHN_PHI0], nb[SHN_PHI1], nb[SHN_A0], nb[SHN_A1],
                                       nb[SHN_E0], nb[SHN_E1], nb[SHN_PSI], nb[SHN_K2T], nb[SHN_K2D], nb[SHN_TRT],
                                       rb, nb[SHN_DPL], args_i[SH_TWO] != 0, args_i[SH_TCMT] != 0, args_i[SH_MORD],
                                       cw, fw)
    return pabs, slope * dlam_dt


@njit(cache=True, _nrt=False)
def selfheat_commit_general(t, k, args_f, args_i):
    """:func:`selfheat_commit` without the Airy path (:func:`selfheat_pabs_general`): every output of ring k from
    :func:`ring_lines_nb` with the slope outputs (as the Airy path writes them), then the bus walk."""
    p_in, pcur, rg, nb, rb, cw, fw = _sh_core(args_f, args_i)
    orl, orr, ol = _sh_outputs(args_f, args_i)
    n = args_i[SH_N]
    q = args_i[SH_HDR + n + k]
    _sh_prepare(t, k, q, p_in, pcur, rg, nb)
    ring_lines_nb(k, pcur, rg[SHR_P1, k], nb[SHN_PHI0], nb[SHN_PHI1], nb[SHN_A0], nb[SHN_A1], nb[SHN_E0],
                  nb[SHN_E1], nb[SHN_PSI], nb[SHN_K2T], nb[SHN_K2D], nb[SHN_TRT], rb, nb[SHN_DPL],
                  args_i[SH_TWO] != 0, args_i[SH_TCMT] != 0, args_i[SH_MORD], True, orl, orr, pcur, cw, fw)
    if q + 1 < n:
        t2 = rg[SHR_T2, args_i[SH_HDR + q + 1]]
        for l in range(pcur.shape[0]):
            pcur[l] *= t2
    else:
        o = args_i[SH_O_T2OUT]
        bus_finalize_nb(p_in, args_i[SH_HDR:SH_HDR + n], rg[SHR_T2], args_f[o], orl, ol)


@njit(cache=True, _nrt=False)
def selfheat_bounds_general(args_f, args_i, dt_max_k, p_max, d2p_max, dpdt_max):
    """:func:`selfheat_bounds` from the separate bound kernels :func:`pabs_bounds_nb` and :func:`pabs_slope_bound_nb`
    (which :func:`sh_bounds` fuses with the same expressions in the same order; no far-line hold), for the
    reference engine's independent check. Uses the walk buffer as scratch."""
    p_in, pcur, rg, nb, rb, cw, fw = _sh_core(args_f, args_i)
    n = args_i[SH_N]
    two = args_i[SH_TWO] != 0
    for l in range(pcur.shape[0]):
        pcur[l] = p_in[l]
    for qpos in range(n):
        k = args_i[SH_HDR + qpos]
        t2k = rg[SHR_T2, k]
        for l in range(pcur.shape[0]):
            pcur[l] *= t2k
        pm, cm = pabs_bounds_nb(k, pcur, rg[SHR_P1, k], nb[SHN_A0], nb[SHN_A1], nb[SHN_E0], nb[SHN_E1], nb[SHN_K2T],
                                nb[SHN_K2D], two)
        sm = pabs_slope_bound_nb(k, pcur, rg[SHR_P1, k], nb[SHN_A0], nb[SHN_A1], nb[SHN_E0], nb[SHN_E1],
                                 nb[SHN_K2T], nb[SHN_K2D], two)
        dmax = 0.0
        for l in range(pcur.shape[0]):
            dmax = max(dmax, abs(nb[SHN_DPL, k, l]))
        s0 = rg[SHR_S0, k]
        beta = rg[SHR_BETA, k]
        smax = abs(s0) * (1.0 + abs(beta) * dt_max_k)
        p_max[k] = pm
        dpdt_max[k] = sm * dmax * smax
        d2p_max[k] = cm * (dmax * smax) ** 2 + sm * dmax * abs(s0 * beta)


# ---- the Airy path of the adapter: the exact ring without backscatter, with its substep constants hoisted ----------
# docs/handoff/perf.md (P1, P4). Within one implicit step only the round-trip phase of a ring depends on the trial
# temperature (a rigid shift b = dphi/dlambda_res * shift(T) of every line); the loss, the couplings and the group delay
# are held. So per (level, ring, line) the cache SH_O_FC keeps cos/sin of the reference phase phi_ref + psi and the
# constants of the Airy closed forms (:func:`_airy_line`: r = t1 t2 a, the numerator terms, the energy and loss
# factors), and an evaluation rotates (cos, sin) by the small angle b (a Taylor polynomial whose truncation error is
# below 1e-17 for |b| <= 0.05 rad, :func:`_rot`; libm cos/sin beyond) and divides once. This is the same function as
# _airy_line up to floating-point rounding: the differences are of the size of the rounding of the absolute phase
# phi_ref (~2e-14 rad at phi ~ 200 rad), measured below 1e-12 relative in P_abs and its slope
# (tests/verification/test_engine_adapter.py).
# The cached cos/sin of each level is updated incrementally from one substep to the next: rotated by the change of its
# (floating-point) reference phase, an exact difference, so only the rounding of each rotation accumulates (reset by an
# exact libm cos/sin every ROT_REFRESH updates or when the phase jumps by more than ROT_MAX). Per update that rounding
# is <= ~4 ulp(1) (cos b rounded, two products and a sum); with random phase steps it adds up as a random walk (~1e-14
# after ROT_REFRESH updates), but a constant step (a steady drift of the ring temperature or the laser) repeats the
# same rounding every update, so the cache drifts systematically: <= ROT_REFRESH x 4 ulp = 1.8e-12 (measured 3e-13).
# At resonance (den >= (1 - r)^2 ~ 5e-3 in g26) that is <= ~7e-10 relative in P_abs, ~5e-9 K of ring temperature at
# g26's self-heating gain, far below the 1e-7 K root tolerance of the solve (test_engine_adapter). The caches are
# keyed on their inputs (phi_ref, psi, a, t_rt per level and line; K^2 per line), so :func:`selfheat_pabs` /
# :func:`selfheat_commit` stay correct when a caller rewrites the rows: :func:`_fast_check` recomputes stale entries,
# and the Airy-path flag of a ring is re-derived from its backscatter row at every call (:func:`_fast_flag`). The
# engine (mrm.sim.kernel) recomputes every entry once per substep with :func:`sh_fast_precompute` and then
# evaluates without the check; both run :func:`_fc_entry`, so engine and reference stay bit-identical.
N_FK = 8
FK_K2T, FK_K2D, FK_T1, FK_K1, FK_T2, FK_K2, FK_K11, FK_A2 = range(N_FK)
N_FC = 13
(FC_PHI, FC_A, FC_TRT, FC_PSI, FC_C, FC_S, FC_TWO_R, FC_RR, FC_SL, FC_BC, FC_CC, FC_DN, FC_CNT) = range(N_FC)
ROT_REFRESH = 4096  # incremental updates of a cached cos/sin before it is recomputed with libm cos/sin
N_LE = 5
LE_C, LE_RINV, LE_U2, LE_LOSS, LE_DLOSS = range(N_LE)
ROT_MAX = 0.05     # largest rotation angle (rad) evaluated with the Taylor polynomial

# ---- far-line hold (fidelity optics_far_hold; docs/handoff/perf.md P5) ----------------------------------------------
# A line far from every resonance of a ring changes that ring's outputs only slowly. With far_hold on, a line whose
# Lorentzian-equivalent detuning u = sqrt(den/(1 - r)^2 - 1) (den = 1 - 2 r cos(phi) + r^2 of the Airy forms) is at
# least FAR_U half-widths at both NRZ levels is held: its per-watt responses X in {T (through), D (drop), L (loss),
# U (CW energy)} and their first derivatives in the phase and in the round-trip amplitude a are stored at the hold
# point (phase phi_h = phi_ref + psi, amplitude a_h), and every evaluation and commit uses
#     X = X_h + X_phi (phi - phi_h) + X_a (a - a_h)     (U also times t_rt / t_rt,h, exactly; P_in and eta exactly)
# while |phi_ref + psi - phi_h| <= FAR_DPHI and |a - a_h| <= FAR_DA (else the line is refreshed at the next substep's
# cache update), and while the in-solve rotation |b| <= FAR_DB (else that line is evaluated exactly from then on).
# Error: every X has the form const + K/den(phi) (T = 1 - k1^2 (1 - t2^2 a^2)/den, D = k1^2 k2^2 a/den, L and U
# = u2 * const with u2 = k1^2/den), and |d^2(1/den)/dphi^2| <= 10 r/den^2 (from den = (1 - r)^2 + 4 r sin^2(phi/2) and
# sin^2 phi <= (den - (1 - r)^2)/r), so the phase remainder is at most 5 r (K/den)/den dphi^2 <= 5 r |X - const|
# dphi^2 / den with dphi <= FAR_DPHI + FAR_DB = 1e-3 rad and den >= (1 - r)^2 (1 + FAR_U^2); for g26 (r ~ 0.93,
# (1 - r)^2 ~ 5e-3) that is <= 1.5e-5 of a far line's own (small) response; the a remainder (d ln X/da <= ~2/(1 - a)
# ~ 40, (1/2) X_aa FAR_DA^2) and the cross term are below 1e-6 of it. A far line absorbs <= 1/(1 + FAR_U^2) of the
# line's peak, so the P_abs error of a ring is <= 14 lines x 1.5e-5 x (P_peak / 65) ~ 3e-9 W at 2 mW per line,
# i.e. <= ~1.5e-5 K of ring temperature (~1e-3 pm) through g26's ~4800 K/W self-heating gain; the monitor drop error
# is ~1e-6 of the monitor input. Measured against far_hold off: tests/verification/test_engine_adapter.py.
FAR_U = 8.0        # far: >= 8 half-widths from every resonance at both levels
FAR_DPHI = 5e-4    # rad: largest drift of the reference phase from the hold point before a refresh
FAR_DB = 5e-4      # rad: largest in-solve rotation evaluated with the held model (else that line goes exact)
FAR_DA = 3e-5      # largest drift of the round-trip amplitude a from the hold point before a refresh
FAR_BOUND_MARGIN = 1e-3   # held bound factors are inflated by this (a moves by <= FAR_DA: ~6e-4 relative)
N_FH = 19
(FH_STATE, FH_ANG, FH_A, FH_TRT, FH_T, FH_D, FH_L, FH_U, FH_TP, FH_DP, FH_LP, FH_UP, FH_TA, FH_DA, FH_LA, FH_UA,
 FH_PM, FH_CM, FH_SM) = range(N_FH)


@njit(cache=True, inline="always")
def _rot(b):
    """(cos b, sin b) for |b| <= ROT_MAX by Taylor polynomials of the lowest degree whose truncation stays below
    1e-17: to b^2 / b^3 for |b| <= 1e-4 (truncation b^4/24 <= 4e-18), to b^4 / b^5 for |b| <= 3e-3 (b^6/720 <= 1e-18),
    to b^8 / b^9 otherwise (b^10/10! <= 3e-20 at 0.05). Most rotations of a substep are below 1e-4 rad."""
    b2 = b * b
    ab = abs(b)
    if ab <= 1e-4:
        return 1.0 - 0.5 * b2, b * (1.0 - b2 * (1.0 / 6.0))
    if ab <= 3e-3:
        return (1.0 - b2 * (0.5 - b2 * (1.0 / 24.0)),
                b * (1.0 - b2 * (1.0 / 6.0 - b2 * (1.0 / 120.0))))
    cb = 1.0 - b2 * (0.5 - b2 * (1.0 / 24.0 - b2 * (1.0 / 720.0 - b2 * (1.0 / 40320.0))))
    sb = b * (1.0 - b2 * (1.0 / 6.0 - b2 * (1.0 / 120.0 - b2 * (1.0 / 5040.0 - b2 * (1.0 / 362880.0)))))
    return cb, sb


@njit(cache=True, _nrt=False)
def _fast_flag(k, args_f, args_i):
    """1.0 when ring k takes the Airy path (exact ring, no backscatter for ring k), else 0.0. Derived from the
    current backscatter row at every call (2 M_b compares: a caller may rewrite any row through
    :func:`selfheat_views`, rb included, and a flag kept from an earlier call would then run the ring without its
    backscatter) and stored in SH_O_FF, which the engine's eng_pabs / eng_commit read after
    :func:`sh_fast_precompute` refreshed it for the substep."""
    fast = args_i[SH_TCMT] == 0
    mb = args_i[SH_MB]
    o_rb = args_i[SH_O_RB] + 2 * mb * k
    for i in range(2 * mb):
        if args_f[_U(o_rb + i)] != 0.0:
            fast = False
    f = 1.0 if fast else 0.0
    args_f[_U(args_i[SH_O_FF] + k)] = f
    return f


@njit(cache=True, error_model="numpy", inline="always")
def _fk_entry(k, l, args_f, args_i):
    """Coupler amplitudes of ring k at line l (t1 = sqrt(1 - K^2_th), k1 = sqrt(K^2_th), t2, k2), k1^2 and t1^2, and
    their keys."""
    n = args_i[SH_N]
    nl = args_i[SH_L]
    nn = n * nl
    o_nl = args_i[SH_O_NL]
    kt = args_f[_U(o_nl + SHN_K2T * nn + k * nl + l)]
    kd = args_f[_U(o_nl + SHN_K2D * nn + k * nl + l)]
    q = args_i[SH_O_FK] + (k * nl + l) * N_FK
    t1 = math.sqrt(1.0 - kt)
    k1 = math.sqrt(kt)
    args_f[_U(q + FK_K2T)] = kt
    args_f[_U(q + FK_K2D)] = kd
    args_f[_U(q + FK_T1)] = t1
    args_f[_U(q + FK_K1)] = k1
    args_f[_U(q + FK_T2)] = math.sqrt(1.0 - kd)
    args_f[_U(q + FK_K2)] = math.sqrt(kd)
    # constants of the line while K^2 does not change (kept here, not per substep in SH_O_FC; perf.md R2-13)
    args_f[_U(q + FK_K11)] = k1 * k1
    args_f[_U(q + FK_A2)] = t1 * t1


@njit(cache=True, error_model="numpy", inline="always")
def _afac(a, memo, lvl):
    """(q, lossfrac, g) of :func:`_airy_line` / :func:`_seg_factors` for round-trip amplitude a (xi = -ln a:
    exp(-xi) = a, q = exp(-xi/2) = sqrt(a), y = xi/2, lossfrac = -expm1(-y), g = lossfrac / y; exact identities, the
    old forms only round differently). memo (8,) keeps the last a of each level and its factors (a is the same at
    every line of a ring), so the three transcendental functions run once per ring and level."""
    j = 4 * lvl
    if memo[_U(j)] == a:
        return memo[_U(j + 1)], memo[_U(j + 2)], memo[_U(j + 3)]
    y = -0.5 * math.log(a)
    qq = math.sqrt(a)
    lossf = -math.expm1(-y)
    g = lossf / y if y > 1e-300 else 1.0
    memo[_U(j)] = a
    memo[_U(j + 1)] = qq
    memo[_U(j + 2)] = lossf
    memo[_U(j + 3)] = g
    return qq, lossf, g


@njit(cache=True, error_model="numpy", inline="always")
def _fc_entry(k, l, two, args_f, args_i, memo):
    """Airy-path constants of ring k at line l for level 0 (and 1 when two): keys, cos/sin of phi_ref + psi, and
    the factors of :func:`_airy_line` (a, the quarter-ring factor q = sqrt(a), the loss factor of
    :func:`_seg_factors`), from the rows and the coupler cache SH_O_FK (which must be current). memo: see
    :func:`_afac` (the result does not depend on it)."""
    n = args_i[SH_N]
    nl = args_i[SH_L]
    nn = n * nl
    o_nl = args_i[SH_O_NL]
    base = k * nl + l
    q = args_i[SH_O_FK] + base * N_FK
    t1 = args_f[_U(q + FK_T1)]
    k1 = args_f[_U(q + FK_K1)]
    t2 = args_f[_U(q + FK_T2)]
    k2 = args_f[_U(q + FK_K2)]
    psi = args_f[_U(o_nl + SHN_PSI * nn + base)]
    trt = args_f[_U(o_nl + SHN_TRT * nn + base)]
    for lvl in range(2 if two else 1):
        ph = args_f[_U(o_nl + (SHN_PHI0R if lvl == 0 else SHN_PHI1R) * nn + base)]
        a = args_f[_U(o_nl + (SHN_A0 if lvl == 0 else SHN_A1) * nn + base)]
        p = args_i[SH_O_FC] + (lvl * nn + base) * N_FC
        # each level's cos/sin is rotated from its own previous value by the change of its reference phase (an
        # exact difference of two floats; typically 1e-5 rad per substep), re-derived with libm every ROT_REFRESH
        # updates, after a jump beyond ROT_MAX and for an empty cache (NaN keys)
        ang = ph + psi
        d = ang - (args_f[_U(p + FC_PHI)] + args_f[_U(p + FC_PSI)])
        cnt = args_f[_U(p + FC_CNT)]
        if abs(d) <= ROT_MAX and cnt < ROT_REFRESH:
            cb, sb = _rot(d)
            c0 = args_f[_U(p + FC_C)]
            s0 = args_f[_U(p + FC_S)]
            c = c0 * cb - s0 * sb
            s = s0 * cb + c0 * sb
            args_f[_U(p + FC_CNT)] = cnt + 1.0
        else:
            c = math.cos(ang)
            s = math.sin(ang)
            args_f[_U(p + FC_CNT)] = 0.0
        args_f[_U(p + FC_PHI)] = ph
        args_f[_U(p + FC_A)] = a
        args_f[_U(p + FC_TRT)] = trt
        args_f[_U(p + FC_PSI)] = psi
        args_f[_U(p + FC_C)] = c
        args_f[_U(p + FC_S)] = s
        qq, lossf, g = _afac(a, memo, lvl)
        r = t1 * t2 * a
        t22 = t2 * t2
        seg = 1.0 + qq + t22 * qq * qq + t22 * qq * qq * qq
        args_f[_U(p + FC_TWO_R)] = 2.0 * r
        args_f[_U(p + FC_RR)] = r * r
        args_f[_U(p + FC_SL)] = seg * lossf
        args_f[_U(p + FC_BC)] = 2.0 * t1 * t2 * a
        args_f[_U(p + FC_CC)] = t22 * a * a
        args_f[_U(p + FC_DN)] = k1 * k1 * k2 * k2 * a
        # (k1^2 and t1^2 live in SH_O_FK; the energy factor seg (0.25 t_rt g) is computed where a full commit needs
        # it, _le_values: the same expressions, so the same values; perf.md R2-13)


@njit(cache=True, error_model="numpy", _nrt=False)
def _fh_entry(k, l, two, args_f, args_i, memo):
    """Classify line l of ring k (needs a current SH_O_FC entry) and, when it is far at every level, store its hold
    (values and first derivatives in phi and a of T, D, L, U per watt at b = 0, and the per-watt factors of the
    bounds of :func:`pabs_bounds_nb` / :func:`pabs_slope_bound_nb`). Returns True when the line is held."""
    n = args_i[SH_N]
    nl = args_i[SH_L]
    nn = n * nl
    base = k * nl + l
    o_fc = args_i[SH_O_FC]
    o_fh = args_i[SH_O_FH]
    n_lvl = 2 if two else 1
    far = True
    for lvl in range(n_lvl):
        p = o_fc + (lvl * nn + base) * N_FC
        two_r = args_f[p + FC_TWO_R]
        den = 1.0 - two_r * args_f[p + FC_C] + args_f[p + FC_RR]
        om = 1.0 - 0.5 * two_r
        if not (den >= om * om * (1.0 + FAR_U * FAR_U)):
            far = False
    h0 = o_fh + base * N_FH
    if not far:
        args_f[h0 + FH_STATE] = 0.0
        return False
    q = args_i[SH_O_FK] + base * N_FK
    kt = args_f[q + FK_K2T]
    t1 = args_f[q + FK_T1]
    k1 = args_f[q + FK_K1]
    t2 = args_f[q + FK_T2]
    k2 = args_f[q + FK_K2]
    t22 = t2 * t2
    for lvl in range(n_lvl):
        p = o_fc + (lvl * nn + base) * N_FC
        h = o_fh + (lvl * nn + base) * N_FH
        c = args_f[p + FC_C]
        sn_ = args_f[p + FC_S]
        two_r = args_f[p + FC_TWO_R]
        a = args_f[p + FC_A]
        trt = args_f[p + FC_TRT]
        den = 1.0 - two_r * c + args_f[p + FC_RR]
        if den == 0.0:
            raise ZeroDivisionError("Airy path: den = 0 (lossless ring without coupling, at resonance)")
        rinv = 1.0 / den
        T = (args_f[q + FK_A2] - args_f[p + FC_BC] * c + args_f[p + FC_CC]) * rinv
        D = args_f[p + FC_DN] * rinv
        u2 = args_f[q + FK_K11] * rinv
        L = u2 * args_f[p + FC_SL]
        qq, lossf, g = _afac(a, memo, lvl)
        seg = 1.0 + qq + t22 * qq * qq + t22 * qq * qq * qq
        se = seg * (0.25 * trt * g)
        U = u2 * se
        # d/dphi: num = A2 - BC cos + CC, den = 1 - 2r cos + r^2
        dden = two_r * sn_
        u2p = -u2 * dden * rinv
        # d/da at fixed phi: r = t1 t2 a
        r = 0.5 * two_r
        dr = t1 * t2
        dden_a = 2.0 * dr * (r - c)
        dnum_a = -2.0 * dr * c + 2.0 * t22 * a
        u2a = -u2 * dden_a * rinv
        dq = 0.5 / qq
        dseg = dq * (1.0 + 2.0 * t22 * qq + 3.0 * t22 * qq * qq)
        dlossf = -dq
        sl = seg * lossf
        dsl = dseg * lossf + seg * dlossf
        y = -0.5 * math.log(a)
        if y > 1e-300:
            dg = (dlossf * y + lossf * 0.5 / a) / (y * y)
        else:
            dg = 0.0
        dse = 0.25 * trt * (dseg * g + seg * dg)
        args_f[h + FH_ANG] = args_f[p + FC_PHI] + args_f[p + FC_PSI]
        args_f[h + FH_A] = a
        args_f[h + FH_TRT] = trt
        args_f[h + FH_T] = T
        args_f[h + FH_D] = D
        args_f[h + FH_L] = L
        args_f[h + FH_U] = U
        args_f[h + FH_TP] = (args_f[p + FC_BC] * sn_ - T * dden) * rinv
        args_f[h + FH_DP] = -D * dden * rinv
        args_f[h + FH_LP] = u2p * args_f[p + FC_SL]
        args_f[h + FH_UP] = u2p * se
        args_f[h + FH_TA] = (dnum_a - T * dden_a) * rinv
        args_f[h + FH_DA] = (k1 * k1 * k2 * k2 - D * dden_a) * rinv
        args_f[h + FH_LA] = u2a * sl + u2 * dsl
        args_f[h + FH_UA] = u2a * se + u2 * dse
        # per-watt factors of the bounds (pabs_bounds_nb / pabs_slope_bound_nb with the raw a)
        rb_ = t1 * t2 * a
        om = 1.0 - rb_
        cb_ = kt * (1.0 - a) * (1.0 + t22 * a)
        args_f[h + FH_PM] = min(1.05 * cb_ / (om * om), 1.0)
        args_f[h + FH_CM] = 2.0 * rb_ * cb_ / (om * om * om * om)
        smf = 0.0
        if rb_ > 0.0:
            aa = 1.0 + rb_ * rb_
            cs = (math.sqrt(aa * aa + 32.0 * rb_ * rb_) - aa) / (4.0 * rb_)
            snb = math.sqrt(max(1.0 - cs * cs, 0.0))
            denb = aa - 2.0 * rb_ * cs
            smf = 2.0 * rb_ * cb_ * snb / (denb * denb)
        args_f[h + FH_SM] = smf
    args_f[h0 + FH_STATE] = 1.0
    return True


@njit(cache=True, _nrt=False)
def sh_fast_precompute(args_f, args_i):
    """Bring every Airy-path cache entry up to date with the rows (the engine calls this once per substep after
    writing the optics inputs): :func:`_fast_check` of every Airy-path ring, i.e. exactly the updates the optics
    adapter makes lazily (the incremental cos/sin update makes the cache a state, so both must update the same
    entries at the same times)."""
    n = args_i[SH_N]
    for k in range(n):
        if _fast_flag(k, args_f, args_i) != 0.0:
            _fast_check(k, args_f, args_i)


@njit(cache=True, error_model="numpy", _nrt=False)
def _fast_check(k, args_f, args_i):
    """Recompute the Airy-path cache entries of ring k whose inputs changed since they were computed (a caller may
    rewrite the rows at any time through :func:`selfheat_views`)."""
    n = args_i[SH_N]
    nl = args_i[SH_L]
    nn = n * nl
    two = args_i[SH_TWO] != 0
    o_nl = args_i[SH_O_NL]
    o_fk = args_i[SH_O_FK]
    o_fc = args_i[SH_O_FC]
    o_memo = args_i[SH_O_FF] + n
    memo = args_f[o_memo:o_memo + 8]
    hold = args_i[SH_FARHOLD] != 0
    o_fh = args_i[SH_O_FH]
    for l in range(nl):
        base = k * nl + l
        q = o_fk + base * N_FK
        if not (args_f[_U(q + FK_K2T)] == args_f[_U(o_nl + SHN_K2T * nn + base)]
                and args_f[_U(q + FK_K2D)] == args_f[_U(o_nl + SHN_K2D * nn + base)]):
            _fk_entry(k, l, args_f, args_i)
            _fc_entry(k, l, two, args_f, args_i, memo)
            if hold:
                _fh_entry(k, l, two, args_f, args_i, memo)
            continue
        stale = False
        if hold and args_f[_U(o_fh + base * N_FH + FH_STATE)] != 0.0:
            # a held line stays held while its phase and loss stay near the hold point (module comment, FAR_*)
            for lvl in range(2 if two else 1):
                h = o_fh + (lvl * nn + base) * N_FH
                ang = (args_f[o_nl + (SHN_PHI0R if lvl == 0 else SHN_PHI1R) * nn + base]
                       + args_f[o_nl + SHN_PSI * nn + base])
                da = args_f[o_nl + (SHN_A0 if lvl == 0 else SHN_A1) * nn + base] - args_f[h + FH_A]
                if not (abs(ang - args_f[h + FH_ANG]) <= FAR_DPHI and abs(da) <= FAR_DA):
                    stale = True
            if not stale:
                continue
        else:
            for lvl in range(2 if two else 1):
                p = o_fc + (lvl * nn + base) * N_FC
                if not (args_f[_U(p + FC_PHI)] == args_f[_U(o_nl + (SHN_PHI0R if lvl == 0 else SHN_PHI1R) * nn + base)]
                        and args_f[_U(p + FC_A)] == args_f[_U(o_nl + (SHN_A0 if lvl == 0 else SHN_A1) * nn + base)]
                        and args_f[_U(p + FC_TRT)] == args_f[_U(o_nl + SHN_TRT * nn + base)]
                        and args_f[_U(p + FC_PSI)] == args_f[_U(o_nl + SHN_PSI * nn + base)]):
                    stale = True
        if stale:
            _fc_entry(k, l, two, args_f, args_i, memo)
            if hold:
                _fh_entry(k, l, two, args_f, args_i, memo)


@njit(cache=True, error_model="numpy", _nrt=False, inline="always")
def sh_fast_pabs(t, k, args_f, args_i):
    """P_abs of an Airy-path ring k at temperature t [W] and dP_abs/dT [W/K], from the caches (no check: the caller
    guarantees they are current); keeps the per-line values of the evaluation in SH_O_LE for the commit. Sums in the
    order of :func:`dpabs_dlambda_res_nb`. With the far-line hold on (optics_far_hold): :func:`_sh_fast_pabs_hold`.
    A zero denominator (den = 1 - 2 r cos + r^2 = 0: r = t1 t2 a = 1, a lossless ring without coupling, at
    resonance) raises ZeroDivisionError, as the closed forms (:func:`_airy_line`) do; for r < 1, den >= (1 - r)^2."""
    if args_i[SH_FARHOLD] != 0:
        return _sh_fast_pabs_hold(t, k, args_f, args_i)
    n = args_i[SH_N]
    nl = args_i[SH_L]
    nn = n * nl
    two = args_i[SH_TWO] != 0
    n_lvl = 2 if two else 1
    o_ring = args_i[SH_O_RING]
    o_nl = args_i[SH_O_NL]
    o_pcur = args_i[SH_O_PCUR]
    o_fc = args_i[SH_O_FC]
    o_fk = args_i[SH_O_FK]
    o_le = args_i[SH_O_LE]
    # _sh_prepare: the rigid shift of ring k at t, and the walk restart at the first ring in bus order
    dt = t - args_f[_U(o_ring + SHR_TREF * n + k)]
    s0 = args_f[_U(o_ring + SHR_S0 * n + k)]
    beta = args_f[_U(o_ring + SHR_BETA * n + k)]
    shift = s0 * (dt + 0.5 * beta * dt * dt)
    if args_i[SH_HDR + n + k] == 0:
        t2 = args_f[_U(o_ring + SHR_T2 * n + k)]
        o_pin = args_i[SH_O_PIN]
        for l in range(nl):
            args_f[_U(o_pcur + l)] = args_f[_U(o_pin + l)] * t2
    dlam_dt = s0 * (1.0 + beta * dt)
    q1 = args_f[_U(o_ring + SHR_P1 * n + k)] if two else 0.0
    q0 = 1.0 - q1
    base = k * nl
    r_dpl = o_nl + SHN_DPL * nn + base
    r_e0 = o_nl + SHN_E0 * nn + base
    r_e1 = o_nl + SHN_E1 * nn + base
    pabs = 0.0
    slope = 0.0
    for l in range(nl):
        dpl = args_f[_U(r_dpl + l)]
        b = dpl * shift
        ab = 0.0
        dab = 0.0
        small = abs(b) <= ROT_MAX
        cb = 1.0
        sb = 0.0
        if small:
            cb, sb = _rot(b)
        k11 = args_f[_U(o_fk + (base + l) * N_FK + FK_K11)]
        for lvl in range(n_lvl):
            p = o_fc + (lvl * nn + base + l) * N_FC
            if small:
                c0 = args_f[_U(p + FC_C)]
                s0_ = args_f[_U(p + FC_S)]
                c = c0 * cb - s0_ * sb
                s = s0_ * cb + c0 * sb
            else:
                ang = (args_f[_U(p + FC_PHI)] + args_f[_U(p + FC_PSI)]) + b
                c = math.cos(ang)
                s = math.sin(ang)
            two_r = args_f[_U(p + FC_TWO_R)]
            den = 1.0 - two_r * c + args_f[_U(p + FC_RR)]
            if den == 0.0:
                raise ZeroDivisionError("Airy path: den = 0 (lossless ring without coupling, at resonance)")
            rinv = 1.0 / den
            u2 = k11 * rinv
            du2 = -u2 * (two_r * s) * rinv
            sl = args_f[_U(p + FC_SL)]
            loss = u2 * sl
            dloss = du2 * sl
            e = o_le + (lvl * nl + l) * N_LE
            args_f[_U(e + LE_C)] = c
            args_f[_U(e + LE_RINV)] = rinv
            args_f[_U(e + LE_U2)] = u2
            args_f[_U(e + LE_LOSS)] = loss
            args_f[_U(e + LE_DLOSS)] = dloss
            if lvl == 0:
                w = q0 * args_f[_U(r_e0 + l)]
            else:
                w = q1 * args_f[_U(r_e1 + l)]
            ab += w * loss
            dab += w * dloss
        pc_ = args_f[_U(o_pcur + l)]
        pabs += pc_ * ab
        slope += pc_ * dab * dpl
    o_last = o_le + 2 * nl * N_LE
    args_f[_U(o_last)] = k
    args_f[_U(o_last + 1)] = t
    args_f[_U(o_last + 2)] = shift
    return pabs, slope * dlam_dt


@njit(cache=True, error_model="numpy", _nrt=False)
def _sh_fast_pabs_hold(t, k, args_f, args_i):
    """:func:`sh_fast_pabs` with the far-line hold on (optics_far_hold): held far lines contribute their first-order
    model while the in-solve rotation |b| <= FAR_DB; beyond it the line is released (its cache entry is brought to
    the current reference phase) and evaluated exactly from then on."""
    n = args_i[SH_N]
    nl = args_i[SH_L]
    nn = n * nl
    two = args_i[SH_TWO] != 0
    n_lvl = 2 if two else 1
    o_ring = args_i[SH_O_RING]
    o_nl = args_i[SH_O_NL]
    o_pcur = args_i[SH_O_PCUR]
    o_fc = args_i[SH_O_FC]
    o_le = args_i[SH_O_LE]
    o_fh = args_i[SH_O_FH]
    hold = args_i[SH_FARHOLD] != 0
    # _sh_prepare: the rigid shift of ring k at t, and the walk restart at the first ring in bus order
    dt = t - args_f[o_ring + SHR_TREF * n + k]
    s0 = args_f[o_ring + SHR_S0 * n + k]
    beta = args_f[o_ring + SHR_BETA * n + k]
    shift = s0 * (dt + 0.5 * beta * dt * dt)
    if args_i[SH_HDR + n + k] == 0:
        t2 = args_f[o_ring + SHR_T2 * n + k]
        o_pin = args_i[SH_O_PIN]
        for l in range(nl):
            args_f[o_pcur + l] = args_f[o_pin + l] * t2
    dlam_dt = s0 * (1.0 + beta * dt)
    q1 = args_f[o_ring + SHR_P1 * n + k] if two else 0.0
    q0 = 1.0 - q1
    base = k * nl
    r_dpl = o_nl + SHN_DPL * nn + base
    r_e0 = o_nl + SHN_E0 * nn + base
    r_e1 = o_nl + SHN_E1 * nn + base
    pabs = 0.0
    slope = 0.0
    for l in range(nl):
        dpl = args_f[r_dpl + l]
        b = dpl * shift
        held = False
        if hold and args_f[o_fh + (base + l) * N_FH + FH_STATE] != 0.0:
            if abs(b) <= FAR_DB:
                held = True
            else:
                # outside the held model's range in this solve: release the line (cache to the current reference)
                o_memo = args_i[SH_O_FF] + n
                _fc_entry(k, l, two, args_f, args_i, args_f[o_memo:o_memo + 8])
                args_f[o_fh + (base + l) * N_FH + FH_STATE] = 0.0
        ab = 0.0
        dab = 0.0
        if held:
            # the first-order model; nothing is stored (the commit recomputes it from the stored shift)
            for lvl in range(n_lvl):
                h = o_fh + (lvl * nn + base + l) * N_FH
                dphi = ((args_f[o_nl + (SHN_PHI0R if lvl == 0 else SHN_PHI1R) * nn + base + l]
                         + args_f[o_nl + SHN_PSI * nn + base + l]) - args_f[h + FH_ANG]) + b
                da = args_f[o_nl + (SHN_A0 if lvl == 0 else SHN_A1) * nn + base + l] - args_f[h + FH_A]
                loss = args_f[h + FH_L] + args_f[h + FH_LP] * dphi + args_f[h + FH_LA] * da
                dloss = args_f[h + FH_LP]
                if lvl == 0:
                    w = q0 * args_f[r_e0 + l]
                else:
                    w = q1 * args_f[r_e1 + l]
                ab += w * loss
                dab += w * dloss
        else:
            small = abs(b) <= ROT_MAX
            cb = 1.0
            sb = 0.0
            if small:
                cb, sb = _rot(b)
            for lvl in range(n_lvl):
                p = o_fc + (lvl * nn + base + l) * N_FC
                if small:
                    c = args_f[p + FC_C] * cb - args_f[p + FC_S] * sb
                    s = args_f[p + FC_S] * cb + args_f[p + FC_C] * sb
                else:
                    ang = (args_f[p + FC_PHI] + args_f[p + FC_PSI]) + b
                    c = math.cos(ang)
                    s = math.sin(ang)
                two_r = args_f[p + FC_TWO_R]
                den = 1.0 - two_r * c + args_f[p + FC_RR]
                if den == 0.0:
                    raise ZeroDivisionError("Airy path: den = 0 (lossless ring without coupling, at resonance)")
                rinv = 1.0 / den
                u2 = args_f[args_i[SH_O_FK] + (base + l) * N_FK + FK_K11] * rinv
                du2 = -u2 * (two_r * s) * rinv
                loss = u2 * args_f[p + FC_SL]
                dloss = du2 * args_f[p + FC_SL]
                e = o_le + (lvl * nl + l) * N_LE
                args_f[e + LE_C] = c
                args_f[e + LE_RINV] = rinv
                args_f[e + LE_U2] = u2
                args_f[e + LE_LOSS] = loss
                args_f[e + LE_DLOSS] = dloss
                if lvl == 0:
                    w = q0 * args_f[r_e0 + l]
                else:
                    w = q1 * args_f[r_e1 + l]
                ab += w * loss
                dab += w * dloss
        pc_ = args_f[o_pcur + l]
        pabs += pc_ * ab
        slope += pc_ * dab * dpl
    o_last = o_le + 2 * nl * N_LE
    args_f[o_last] = k
    args_f[o_last + 1] = t
    args_f[o_last + 2] = shift
    return pabs, slope * dlam_dt


@njit(cache=True, error_model="numpy", inline="always")
def _le_values(k, l, lvl, held, shift, args_f, args_i, n, nl, nn, o_nl, o_fc, o_fh, o_le, memo):
    """(T, D, L, U, dL/dphi) per watt of level lvl of ring k at line l at the last evaluation: the Airy closed forms
    from the cached constants and SH_O_LE, or the first-order model of a held line at the evaluation's shift."""
    e = o_le + (lvl * nl + l) * N_LE
    if held:
        h = o_fh + (lvl * nn + k * nl + l) * N_FH
        base = k * nl + l
        dphi = (((args_f[o_nl + (SHN_PHI0R if lvl == 0 else SHN_PHI1R) * nn + base]
                  + args_f[o_nl + SHN_PSI * nn + base]) - args_f[h + FH_ANG])
                + args_f[o_nl + SHN_DPL * nn + base] * shift)
        da = args_f[o_nl + (SHN_A0 if lvl == 0 else SHN_A1) * nn + base] - args_f[h + FH_A]
        T = args_f[h + FH_T] + args_f[h + FH_TP] * dphi + args_f[h + FH_TA] * da
        D = args_f[h + FH_D] + args_f[h + FH_DP] * dphi + args_f[h + FH_DA] * da
        L = args_f[h + FH_L] + args_f[h + FH_LP] * dphi + args_f[h + FH_LA] * da
        U = ((args_f[h + FH_U] + args_f[h + FH_UP] * dphi + args_f[h + FH_UA] * da)
             * (args_f[o_nl + SHN_TRT * nn + base] / args_f[h + FH_TRT]))
        return T, D, L, U, args_f[h + FH_LP]
    p = o_fc + (lvl * nn + k * nl + l) * N_FC
    q = args_i[SH_O_FK] + (k * nl + l) * N_FK
    c = args_f[e + LE_C]
    rinv = args_f[e + LE_RINV]
    T = (args_f[q + FK_A2] - args_f[p + FC_BC] * c + args_f[p + FC_CC]) * rinv
    D = args_f[p + FC_DN] * rinv
    # the CW energy factor seg (0.25 t_rt g) of the entry (a and t_rt as cached in it; _fc_entry's expressions)
    a = args_f[p + FC_A]
    qq, lossf, g = _afac(a, memo, lvl)
    t2 = args_f[q + FK_T2]
    t22 = t2 * t2
    seg = 1.0 + qq + t22 * qq * qq + t22 * qq * qq * qq
    U = args_f[e + LE_U2] * (seg * (0.25 * args_f[p + FC_TRT] * g))
    return T, D, args_f[e + LE_LOSS], U, args_f[e + LE_DLOSS]


@njit(cache=True, error_model="numpy", _nrt=False)
def sh_fast_outputs(k, args_f, args_i, full=True):
    """Every output of Airy-path ring k from its last evaluation (SH_O_LE), the arithmetic of :func:`ring_lines_nb`
    with with_slope on the per-watt responses of :func:`_airy_line` (R = A = U_ccw = 0; held far lines: their
    first-order model); the NRZ-averaged through power goes to the walk buffer p_cur. Then the caller passes it to the
    next ring (or finalizes the bus).

    full = False (the engine at every step of a block but its last; docs/handoff/perf.md R2-3): only the outputs a
    plant tick reads, RL_PIN, RL_THRU, RL_THRU1, RL_THRU0 per line and R_DROP_SUM, R_ABS_SUM, R_DROP_SUM1,
    R_DROP_SUM0, and the walk buffer, each from the same expression as the full commit (so bit-identical); the other
    fields keep their values until the next full commit. The engine commits in full at the last step of every block,
    so every output read after a block (Plant.truth, the reference parity) is the full set of that step."""
    if full or args_i[SH_FARHOLD] != 0:
        _sh_fast_outputs_full(k, args_f, args_i)
        return
    n = args_i[SH_N]
    nl = args_i[SH_L]
    nn = n * nl
    two = args_i[SH_TWO] != 0
    o_ring = args_i[SH_O_RING]
    o_nl = args_i[SH_O_NL]
    o_pcur = args_i[SH_O_PCUR]
    o_orl = args_i[SH_O_ORL]
    o_or = args_i[SH_O_OR]
    o_fc = args_i[SH_O_FC]
    o_le = args_i[SH_O_LE]
    q1 = args_f[_U(o_ring + SHR_P1 * n + k)] if two else 0.0
    q0 = 1.0 - q1
    base = k * nl
    r_e0 = o_nl + SHN_E0 * nn + base
    r_e1 = o_nl + SHN_E1 * nn + base
    dsum = 0.0
    asum = 0.0
    d1sum = 0.0
    d0sum = 0.0
    o_fk = args_i[SH_O_FK]
    for l in range(nl):
        P = args_f[_U(o_pcur + l)]
        a2 = args_f[_U(o_fk + (base + l) * N_FK + FK_A2)]
        p = o_fc + (base + l) * N_FC
        e = o_le + l * N_LE
        c = args_f[_U(e + LE_C)]
        rinv = args_f[_U(e + LE_RINV)]
        T0 = (a2 - args_f[_U(p + FC_BC)] * c + args_f[_U(p + FC_CC)]) * rinv
        D0 = args_f[_U(p + FC_DN)] * rinv
        L0 = args_f[_U(e + LE_LOSS)]
        e0 = args_f[_U(r_e0 + l)]
        if two:
            p = o_fc + (nn + base + l) * N_FC
            e = o_le + (nl + l) * N_LE
            c = args_f[_U(e + LE_C)]
            rinv = args_f[_U(e + LE_RINV)]
            T1 = (a2 - args_f[_U(p + FC_BC)] * c + args_f[_U(p + FC_CC)]) * rinv
            D1 = args_f[_U(p + FC_DN)] * rinv
            L1 = args_f[_U(e + LE_LOSS)]
            e1 = args_f[_U(r_e1 + l)]
        else:
            T1 = T0
            D1 = D0
            L1 = L0
            e1 = e0
        thru = P * (q1 * T1 + q0 * T0)
        drop = P * (q1 * D1 + q0 * D0)
        absd = P * (q1 * e1 * L1 + q0 * e0 * L0)
        rl = o_orl + base + l
        args_f[_U(rl + RL_PIN * nn)] = P
        args_f[_U(rl + RL_THRU * nn)] = thru
        args_f[_U(rl + RL_THRU1 * nn)] = P * T1
        args_f[_U(rl + RL_THRU0 * nn)] = P * T0
        dsum += drop
        asum += absd
        d1sum += P * D1
        d0sum += P * D0
        args_f[_U(o_pcur + l)] = thru
    args_f[_U(o_or + R_DROP_SUM * n + k)] = dsum
    args_f[_U(o_or + R_ABS_SUM * n + k)] = asum
    args_f[_U(o_or + R_DROP_SUM1 * n + k)] = d1sum
    args_f[_U(o_or + R_DROP_SUM0 * n + k)] = d0sum


@njit(cache=True, error_model="numpy", _nrt=False)
def _sh_fast_outputs_full(k, args_f, args_i):
    """:func:`sh_fast_outputs` with full = True."""
    n = args_i[SH_N]
    nl = args_i[SH_L]
    nn = n * nl
    two = args_i[SH_TWO] != 0
    o_ring = args_i[SH_O_RING]
    o_nl = args_i[SH_O_NL]
    o_pcur = args_i[SH_O_PCUR]
    o_orl = args_i[SH_O_ORL]
    o_or = args_i[SH_O_OR]
    o_fc = args_i[SH_O_FC]
    o_le = args_i[SH_O_LE]
    o_fh = args_i[SH_O_FH]
    q1 = args_f[o_ring + SHR_P1 * n + k] if two else 0.0
    q0 = 1.0 - q1
    base = k * nl
    r_e0 = o_nl + SHN_E0 * nn + base
    r_e1 = o_nl + SHN_E1 * nn + base
    r_dpl = o_nl + SHN_DPL * nn + base
    dsum = 0.0
    asum = 0.0
    lsum = 0.0
    d1sum = 0.0
    d0sum = 0.0
    slope = 0.0
    hold = args_i[SH_FARHOLD] != 0
    shift = args_f[o_le + 2 * nl * N_LE + 2]
    o_memo = args_i[SH_O_FF] + n
    memo = args_f[o_memo:o_memo + 8]
    for l in range(nl):
        P = args_f[o_pcur + l]
        held = hold and args_f[o_fh + (base + l) * N_FH + FH_STATE] != 0.0
        T0, D0, L0, U0, dL0 = _le_values(k, l, 0, held, shift, args_f, args_i, n, nl, nn, o_nl, o_fc, o_fh, o_le,
                                         memo)
        e0 = args_f[r_e0 + l]
        if two:
            T1, D1, L1, U1, dL1 = _le_values(k, l, 1, held, shift, args_f, args_i, n, nl, nn, o_nl, o_fc, o_fh,
                                             o_le, memo)
            e1 = args_f[r_e1 + l]
        else:
            T1 = T0
            D1 = D0
            L1 = L0
            U1 = U0
            dL1 = dL0
            e1 = e0
        thru = P * (q1 * T1 + q0 * T0)
        drop = P * (q1 * D1 + q0 * D0)
        loss = P * (q1 * L1 + q0 * L0)
        absd = P * (q1 * e1 * L1 + q0 * e0 * L0)
        rl = o_orl + base + l
        args_f[rl + RL_PIN * nn] = P
        args_f[rl + RL_THRU * nn] = thru
        args_f[rl + RL_DROP * nn] = drop
        args_f[rl + RL_ADD * nn] = 0.0
        args_f[rl + RL_REFL * nn] = 0.0
        args_f[rl + RL_LOSS * nn] = loss
        args_f[rl + RL_ABS * nn] = absd
        args_f[rl + RL_UCW * nn] = P * (q1 * U1 + q0 * U0)
        args_f[rl + RL_UCCW * nn] = 0.0
        args_f[rl + RL_THRU1 * nn] = P * T1
        args_f[rl + RL_THRU0 * nn] = P * T0
        args_f[rl + RL_DROP1 * nn] = P * D1
        args_f[rl + RL_DROP0 * nn] = P * D0
        g = P * (q1 * e1 * dL1 + q0 * e0 * dL0)
        args_f[rl + RL_DABS_DPHI * nn] = g
        slope += g * args_f[r_dpl + l]
        dsum += drop
        asum += absd
        lsum += loss
        d1sum += P * D1
        d0sum += P * D0
        args_f[o_pcur + l] = thru
    args_f[o_or + R_DROP_SUM * n + k] = dsum
    args_f[o_or + R_ABS_SUM * n + k] = asum
    args_f[o_or + R_LOSS_SUM * n + k] = lsum
    args_f[o_or + R_DROP_SUM1 * n + k] = d1sum
    args_f[o_or + R_DROP_SUM0 * n + k] = d0sum
    args_f[o_or + R_DABS_DLAM * n + k] = slope


@njit(cache=True, _nrt=False)
def sh_bounds(args_f, args_i, dt_max_k, p_max, d2p_max, dpdt_max):
    """The bounds of :func:`selfheat_bounds` (pabs_bounds_nb and pabs_slope_bound_nb fused: the same expressions in
    the same order) from the current rows, except that a line held by the far-line hold (optics_far_hold) uses its
    per-watt factors stored at the hold point, inflated by FAR_BOUND_MARGIN (its a moved by <= FAR_DA since)."""
    n = args_i[SH_N]
    nl = args_i[SH_L]
    nn = n * nl
    two = args_i[SH_TWO] != 0
    o_ring = args_i[SH_O_RING]
    o_nl = args_i[SH_O_NL]
    o_pin = args_i[SH_O_PIN]
    o_pcur = args_i[SH_O_PCUR]
    o_fh = args_i[SH_O_FH]
    o_fk = args_i[SH_O_FK]
    hold = args_i[SH_FARHOLD] != 0
    o_ff = args_i[SH_O_FF]
    for l in range(nl):
        args_f[_U(o_pcur + l)] = args_f[_U(o_pin + l)]
    for qpos in range(n):
        k = args_i[SH_HDR + qpos]
        t2k = args_f[_U(o_ring + SHR_T2 * n + k)]
        for l in range(nl):
            args_f[_U(o_pcur + l)] *= t2k
        p1 = args_f[_U(o_ring + SHR_P1 * n + k)]
        q1 = p1 if two else 0.0
        hk = hold and args_f[_U(o_ff + k)] == 1.0
        pm = 0.0
        cm = 0.0
        sm = 0.0
        for lev in range(2):
            w = (1.0 - q1) if lev == 0 else q1
            if w == 0.0:
                continue
            row_a = o_nl + (SHN_A0 if lev == 0 else SHN_A1) * nn + k * nl
            row_e = o_nl + (SHN_E0 if lev == 0 else SHN_E1) * nn + k * nl
            row_kt = o_nl + SHN_K2T * nn + k * nl
            row_kd = o_nl + SHN_K2D * nn + k * nl
            for l in range(nl):
                eta = args_f[_U(row_e + l)]
                pin = args_f[_U(o_pcur + l)]
                if hk and args_f[_U(o_fh + (k * nl + l) * N_FH + FH_STATE)] != 0.0:
                    h = o_fh + ((lev if two else 0) * nn + k * nl + l) * N_FH
                    mg = 1.0 + FAR_BOUND_MARGIN
                    pm += w * pin * eta * (args_f[_U(h + FH_PM)] * mg)
                    cm += w * pin * eta * (args_f[_U(h + FH_CM)] * mg)
                    sm += w * pin * eta * (args_f[_U(h + FH_SM)] * mg)
                    continue
                a = args_f[_U(row_a + l)]
                q = o_fk + (k * nl + l) * N_FK
                kt = args_f[_U(row_kt + l)]
                kd = args_f[_U(row_kd + l)]
                if args_f[_U(q + FK_K2T)] == kt and args_f[_U(q + FK_K2D)] == kd:
                    t1 = args_f[_U(q + FK_T1)]                 # the same square roots, cached (_fk_entry)
                    t2 = args_f[_U(q + FK_T2)]
                else:
                    t1 = math.sqrt(1.0 - kt)
                    t2 = math.sqrt(1.0 - kd)
                r = t1 * t2 * a
                om = 1.0 - r
                c = kt * (1.0 - a) * (1.0 + t2 * t2 * a)
                pm += w * pin * eta * min(1.05 * c / (om * om), 1.0)
                cm += w * pin * eta * 2.0 * r * c / (om * om * om * om)
                if r <= 0.0:
                    continue
                aa = 1.0 + r * r
                cs = (math.sqrt(aa * aa + 32.0 * r * r) - aa) / (4.0 * r)
                sn = math.sqrt(max(1.0 - cs * cs, 0.0))
                den = aa - 2.0 * r * cs
                sm += w * pin * eta * 2.0 * r * c * sn / (den * den)
        cm = 1.25 * cm
        sm = 1.10 * sm
        dmax = 0.0
        r_dpl = o_nl + SHN_DPL * nn + k * nl
        for l in range(nl):
            dmax = max(dmax, abs(args_f[_U(r_dpl + l)]))
        s0 = args_f[_U(o_ring + SHR_S0 * n + k)]
        beta = args_f[_U(o_ring + SHR_BETA * n + k)]
        smax = abs(s0) * (1.0 + abs(beta) * dt_max_k)
        p_max[k] = pm
        dpdt_max[k] = sm * dmax * smax
        d2p_max[k] = cm * (dmax * smax) ** 2 + sm * dmax * abs(s0 * beta)


@njit(cache=True, _nrt=False)
def sh_pass_on(k, args_f, args_i, full=True):
    """After ring k's outputs: its through power (in p_cur) times the bus segment to the next ring in bus order, or,
    after the last ring, :func:`bus_finalize_nb` (full = False: only its bus-output part, :func:`_bus_pout_nb`, the
    part a plant tick reads; see :func:`sh_fast_outputs`)."""
    n = args_i[SH_N]
    nl = args_i[SH_L]
    q = args_i[SH_HDR + n + k]
    o_pcur = args_i[SH_O_PCUR]
    if q + 1 < n:
        t2 = args_f[_U(args_i[SH_O_RING] + SHR_T2 * n + args_i[SH_HDR + q + 1])]
        for l in range(nl):
            args_f[_U(o_pcur + l)] *= t2
    else:
        # only the views bus_finalize_nb needs (_sh_core / _sh_outputs build ten, ~0.15 us)
        o = args_i[SH_O_PIN]
        p_in = args_f[o:o + nl]
        o = args_i[SH_O_RING] + SHR_T2 * n
        t2row = args_f[o:o + n]
        o = args_i[SH_O_ORL]
        orl = args_f[o:o + N_RL * n * nl].reshape((N_RL, n, nl))
        o = args_i[SH_O_OL]
        ol = args_f[o:o + N_L * nl].reshape((N_L, nl))
        if full:
            bus_finalize_nb(p_in, args_i[SH_HDR:SH_HDR + n], t2row, args_f[args_i[SH_O_T2OUT]], orl, ol)
        else:
            _bus_pout_nb(p_in, args_i[SH_HDR:SH_HDR + n], t2row, args_f[args_i[SH_O_T2OUT]], orl, ol)


@njit(cache=True, _nrt=False)
def sh_last_is(k, t, args_f, args_i):
    """True when the last evaluation (fast path) was ring k at temperature t."""
    o_last = args_i[SH_O_LE] + 2 * args_i[SH_L] * N_LE
    return args_f[o_last] == k and args_f[o_last + 1] == t


@njit(cache=True, _nrt=False)
def sh_last_clear(args_f, args_i):
    args_f[args_i[SH_O_LE] + 2 * args_i[SH_L] * N_LE] = -1.0


def pack_selfheat_args(p_in_w, phi_ref, k2_th, k2_dr, a_rt, t_rt_s, eta, lam_m, t_ref_k, s0_m_per_k, beta_per_k=0.0,
                       bus_order=None, bus_t2=None, bus_t2_out=1.0, psi=None, rb=None, nrz=None, method="exact",
                       m_orders=3, commit_slope=False, far_hold=False):
    """Pack one world for :func:`selfheat_pabs` / :func:`selfheat_commit` / :func:`selfheat_bounds`.

    Arguments as :func:`bus_propagate` (phi_ref and nrz["phi1"] are the round-trip phases at T_ref,k, without the
    coupler phase), plus lam_m (L,) or (N, L) line wavelengths [m], t_ref_k (N,) [K], s0_m_per_k (N,) and beta_per_k
    (N,) of lambda_res,k(T) - lambda_res,k(T_ref) = s0 [dT + beta dT^2/2], and commit_slope (also fill the slope
    outputs at commit), far_hold (the far-line hold of the Airy path, fidelity optics_far_hold: see the module
    comment above :data:`FAR_U`). Returns (args_f (nf,) float64, args_i (ni,) int64). Per substep the plant updates
    inputs through :func:`selfheat_views` (for example phi0_ref/phi1_ref after a bias change, a0/a1 and eta0/eta1 at
    the previous ring temperature, p_in after a laser step) and reads the outputs there after the solve.
    """
    args = _prepare_bus(p_in_w, phi_ref, k2_th, k2_dr, a_rt, t_rt_s, eta, bus_order, bus_t2, bus_t2_out, psi, rb, nrz)
    (p_in, order, t2, t2o, p1, phi0, phi1, a0, a1, eta0, eta1, psi_, k2t, k2d, trt, rb_, two) = args
    if method not in ("exact", "tcmt"):
        raise ValueError("method must be 'exact' or 'tcmt'")
    if int(m_orders) < 1:
        raise ValueError("m_orders must be >= 1")
    n, nl = phi0.shape
    mb = rb_.shape[1]
    lam = _nl(lam_m, (n, nl), "lam_m")
    dpl = TWO_PI * C0 * trt / (lam * lam)
    tref = _nl(t_ref_k, (n,), "t_ref_k")
    s0 = _nl(s0_m_per_k, (n,), "s0_m_per_k")
    beta = _nl(beta_per_k, (n,), "beta_per_k")
    sizes = ((SH_O_PIN, nl), (SH_O_PCUR, nl), (SH_O_T2OUT, 1), (SH_O_RING, N_SHR * n), (SH_O_NL, N_SHN * n * nl),
             (SH_O_RB, 2 * n * mb), (SH_O_ORL, N_RL * n * nl), (SH_O_OR, N_R * n), (SH_O_OL, N_L * nl),
             (SH_O_CW, 2 * N_CW * nl), (SH_O_FW, N_FW * nl), (SH_O_FK, n * nl * N_FK), (SH_O_FC, 2 * n * nl * N_FC),
             (SH_O_FF, n + 8), (SH_O_LE, 2 * nl * N_LE + 3), (SH_O_FH, 2 * n * nl * N_FH))
    args_i = np.zeros(SH_HDR + 2 * n, np.int64)
    args_i[SH_N], args_i[SH_L], args_i[SH_MB] = n, nl, mb
    args_i[SH_TWO], args_i[SH_TCMT], args_i[SH_MORD] = int(two), int(method == "tcmt"), int(m_orders)
    args_i[SH_SLOPE] = int(bool(commit_slope))
    args_i[SH_FARHOLD] = int(bool(far_hold))
    off = 0
    for key, size in sizes:
        off += off % 2                   # even offsets: the complex views stay 16-byte aligned
        args_i[key] = off
        off += size
    args_f = np.zeros(off + off % 2)
    args_i[SH_HDR:SH_HDR + n] = order
    args_i[SH_HDR + n + order] = np.arange(n)
    v = selfheat_views(args_f, args_i)
    for dst, src in ((0, p_in), (1, t2), (2, t2o), (3, p1), (4, phi0), (5, phi1), (6, phi0), (7, phi1), (8, a0),
                     (9, a1), (10, eta0), (11, eta1), (12, psi_), (13, k2t), (14, k2d), (15, trt), (16, dpl),
                     (17, tref), (18, s0), (19, beta), (20, rb_)):
        v[dst][...] = src
    # the Airy-path caches start invalid (NaN keys never compare equal); no last evaluation
    o = args_i[SH_O_FK]
    args_f[o:o + n * nl * N_FK] = np.nan
    o = args_i[SH_O_FC]
    args_f[o:o + 2 * n * nl * N_FC] = np.nan
    o = args_i[SH_O_FF]
    args_f[o:o + n + 8] = np.nan
    o = args_i[SH_O_LE] + 2 * nl * N_LE
    args_f[o] = -1.0                                   # (the far-line hold block starts zero: no line held)
    return args_f, args_i


# =====================================================================================================================
# Vectorized single-ring evaluation (exact and TCMT) for analysis, tests and the waveform tables
# =====================================================================================================================

@njit(cache=True)
def _ring_vec(phi, k2_th, k2_dr, a_rt, t_rt, rb, s0, s1, s2, s3, use_tcmt, m_orders, out_c, out_f):
    n = phi.shape[0]
    for i in range(n):
        if use_tcmt:
            ki, ke, kd = tcmt_rates(k2_th[i], k2_dr[i], a_rt[i], t_rt[i])
            r = _tcmt_line(phi[i], ki, ke, kd, t_rt[i], rb, m_orders, s0[i], s1[i], s2[i], s3[i])
        else:
            r = _exact_line_d(phi[i], -math.log(a_rt[i]), math.sqrt(1.0 - k2_th[i]), math.sqrt(k2_th[i]),
                              math.sqrt(1.0 - k2_dr[i]), math.sqrt(k2_dr[i]), t_rt[i], rb,
                              s0[i], s1[i], s2[i], s3[i])
        out_c[0, i] = r[0]
        out_c[1, i] = r[1]
        out_c[2, i] = r[2]
        out_c[3, i] = r[3]
        out_f[0, i] = r[4]
        out_f[1, i] = r[5]
        out_f[2, i] = r[6]
        out_f[3, i] = r[7]
        out_f[4, i] = r[8]
        out_f[5, i] = r[9]


@dataclass
class RingResponse:
    """Steady state of one ring (arrays broadcast over the inputs).

    Amplitudes [sqrt(W)]: refl (port 0 out), thru (port 1 out), drop (port 2 out, CW, monitor), add_out (port 3 out,
    CCW drop). Powers [W]: the |.|^2 of those. u_cw, u_ccw [J]: intracavity energies. p_loss [W]: power lost to
    intrinsic loss (absorbed + scattered + radiated; multiply by eta for heat). dloss_dphi [W/rad],
    dthru_dphi, ddrop_dphi [W/rad]: slopes with respect to the round-trip phase.
    """
    refl: np.ndarray
    thru: np.ndarray
    drop: np.ndarray
    add_out: np.ndarray
    u_cw: np.ndarray
    u_ccw: np.ndarray
    p_loss: np.ndarray
    dloss_dphi: np.ndarray
    dthru_dphi: np.ndarray
    ddrop_dphi: np.ndarray

    @property
    def p_refl(self):
        return np.abs(self.refl) ** 2

    @property
    def p_thru(self):
        return np.abs(self.thru) ** 2

    @property
    def p_drop(self):
        return np.abs(self.drop) ** 2

    @property
    def p_add_out(self):
        return np.abs(self.add_out) ** 2

    def p_in_total(self, s_in, s_thru_in=0.0, s_drop_in=0.0, s_add=0.0):
        """Total incoming power [W] for an energy-balance check."""
        return sum(np.abs(np.asarray(x)) ** 2 for x in (s_in, s_thru_in, s_drop_in, s_add))


def _as_rb(rb):
    if rb is None:
        return np.zeros(1, dtype=np.complex128)
    rb = np.atleast_1d(np.asarray(rb, dtype=np.complex128))
    if rb.ndim != 1 or rb.size == 0:
        raise ValueError("rb must be a 1-D array of per-order round-trip backscatter values")
    if np.any(np.abs(rb) >= 2.0):
        raise ValueError("|r_b| must be < 2 (each scatterer of the pair has |rho| = |r_b|/2 < 1)")
    return rb


def _ring_eval(phi_rad, k2_th, k2_dr, a_rt, t_rt_s, rb, psi_rad, s_in, s_thru_in, s_drop_in, s_add,
               use_tcmt, m_orders):
    arrs = np.broadcast_arrays(np.asarray(phi_rad, float), np.asarray(k2_th, float), np.asarray(k2_dr, float),
                               np.asarray(a_rt, float), np.asarray(t_rt_s, float), np.asarray(psi_rad, float),
                               np.asarray(s_in, complex), np.asarray(s_thru_in, complex),
                               np.asarray(s_drop_in, complex), np.asarray(s_add, complex))
    shape = arrs[0].shape
    phi, k2t, k2d, a, trt, psi, s0, s1, s2, s3 = (np.ascontiguousarray(x).ravel() for x in arrs)
    if not all(np.all(np.isfinite(x)) for x in (phi, k2t, k2d, a, trt, psi, s0, s1, s2, s3)):
        raise ValueError("inputs must be finite")
    if np.any((k2t < 0) | (k2t >= 1)) or np.any((k2d < 0) | (k2d >= 1)):
        raise ValueError("K^2 must lie in [0, 1)")
    if np.any((a <= 0) | (a > 1)):
        raise ValueError("a_rt must lie in (0, 1]")
    if np.any(trt <= 0):
        raise ValueError("t_rt_s must be > 0")
    out_c = np.empty((4, phi.size), np.complex128)
    out_f = np.empty((6, phi.size), np.float64)
    _ring_vec(phi + psi, k2t, k2d, a, trt, _as_rb(rb), s0, s1, s2, s3, bool(use_tcmt), int(m_orders), out_c, out_f)
    c = [x.reshape(shape) for x in out_c]
    f = [x.reshape(shape) for x in out_f]
    return RingResponse(c[0], c[1], c[2], c[3], f[0], f[1], f[2], f[3], f[4], f[5])


def ring_exact(phi_rad, k2_th, k2_dr, a_rt, t_rt_s, rb=None, psi_rad=0.0, s_in=1.0, s_thru_in=0.0,
               s_drop_in=0.0, s_add=0.0) -> RingResponse:
    """Exact steady state of an add-drop ring with two point couplers and CW/CCW backscatter (module docstring).

    Args (numpy-broadcastable):
        phi_rad: round-trip phase at the line, from n_eff(lambda, T, V)*L/lambda with the junction fill [rad].
        k2_th, k2_dr: per-pass power cross-coupling of the through and drop couplers at the LINE wavelength [-]
            (k2_dr = 0 gives the all-pass ring).
        a_rt: round-trip amplitude transmission [-]. t_rt_s: round-trip group delay n_g L / c [s].
        rb: complex round-trip backscatter per order (1-D, periodic in the absolute order; r_b = 2 pi beta/FSR,
            :func:`beta_to_rb`) [-]. None: no backscatter.
        psi_rad: ring-side coupler phase (both couplers) [rad], added to phi.
        s_in, s_thru_in, s_drop_in, s_add: incoming amplitudes at ports 0, 1, 2, 3 [sqrt(W)].
    Returns a :class:`RingResponse`. Energy balance: sum of outgoing powers + p_loss = sum of incoming powers to
    rounding (unitary couplers and scatterers; the loss is integrated exactly per segment).
    """
    return _ring_eval(phi_rad, k2_th, k2_dr, a_rt, t_rt_s, rb, psi_rad, s_in, s_thru_in, s_drop_in, s_add,
                      False, 1)


def ring_tcmt(phi_rad, k2_th, k2_dr, a_rt, t_rt_s, rb=None, psi_rad=0.0, m_orders=3, s_in=1.0, s_thru_in=0.0,
              s_drop_in=0.0, s_add=0.0) -> RingResponse:
    """TCMT steady state with a COMPLEX sum over the m_orders orders nearest the line (07 P2-4; 03 section 2.2-2.7).

    Same arguments and outputs as :func:`ring_exact`; the rates come from :func:`tcmt_rates` and
    :func:`tcmt_backscatter_rate` (the single mapping, also exposed as :func:`exact_to_tcmt`). Agrees with the
    exact ring to O(1/F^2) near resonance; in the tails it is limited by the truncation of the order sum.
    """
    if int(m_orders) < 1:
        raise ValueError("m_orders must be >= 1")
    return _ring_eval(phi_rad, k2_th, k2_dr, a_rt, t_rt_s, rb, psi_rad, s_in, s_thru_in, s_drop_in, s_add,
                      True, m_orders)


def ring_smatrix(phi_rad, k2_th, k2_dr, a_rt, t_rt_s, rb=None, psi_rad=0.0, method="exact", m_orders=3):
    """4x4 scattering matrix S[out_port, in_port] of one ring at scalar inputs (ports as in the module docstring).

    For the exact ring S = S^T (reciprocity) and S^H S <= I with equality iff a_rt = 1.
    """
    if method not in ("exact", "tcmt"):
        raise ValueError("method must be 'exact' or 'tcmt'")
    f = ring_exact if method == "exact" else (lambda *a, **kw: ring_tcmt(*a, m_orders=m_orders, **kw))
    S = np.empty((4, 4), np.complex128)
    for j, key in enumerate(("s_in", "s_thru_in", "s_drop_in", "s_add")):
        r = f(float(phi_rad), float(k2_th), float(k2_dr), float(a_rt), float(t_rt_s), rb=rb, psi_rad=float(psi_rad),
              **{"s_in": 0.0, key: 1.0})
        S[:, j] = [complex(r.refl), complex(r.thru), complex(r.drop), complex(r.add_out)]
    return S


# =====================================================================================================================
# Bus: Python API
# =====================================================================================================================

@dataclass
class BusResult:
    """Outputs of :func:`bus_propagate`. Arrays indexed [ring, line] by PHYSICAL ring index (not bus position).

    Per (ring, line) [W]: p_in, thru, drop (CW, monitor), add_out (CCW), refl, loss, absorbed, and per level
    thru1/thru0, drop1/drop0, pout1/pout0 (bus output with the ring at a level, other rings averaged). Energies
    u_cw, u_ccw [J]. Per ring [W]: drop_sum (monitor optical input, every line and order), abs_sum (self-heating
    source), loss_sum, drop_sum1/drop_sum0 (P1/P0 at the monitor). Per line: p_out [W], refl_in [W], bound [-].
    With slopes requested: dabs_dphi (ring, line) [W/rad] and dabs_dlam (ring) [W/m], else zeros.
    """
    p_in: np.ndarray
    thru: np.ndarray
    drop: np.ndarray
    add_out: np.ndarray
    refl: np.ndarray
    loss: np.ndarray
    absorbed: np.ndarray
    u_cw: np.ndarray
    u_ccw: np.ndarray
    thru1: np.ndarray
    thru0: np.ndarray
    drop1: np.ndarray
    drop0: np.ndarray
    pout1: np.ndarray
    pout0: np.ndarray
    drop_sum: np.ndarray
    abs_sum: np.ndarray
    loss_sum: np.ndarray
    drop_sum1: np.ndarray
    drop_sum0: np.ndarray
    p_out: np.ndarray
    bound: np.ndarray
    refl_in: np.ndarray
    dabs_dphi: np.ndarray
    dabs_dlam: np.ndarray


def _nl(x, shape, name, dtype=np.float64):
    a = np.ascontiguousarray(np.broadcast_to(np.asarray(x, dtype=dtype), shape))
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} has non-finite entries")
    return a


def _prepare_bus(p_in_w, phi, k2_th, k2_dr, a_rt, t_rt_s, eta, bus_order, bus_t2, bus_t2_out, psi, rb, nrz):
    p_in_w = np.ascontiguousarray(np.atleast_1d(np.asarray(p_in_w, dtype=np.float64)))
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 2 or phi.shape[1] != p_in_w.shape[0]:
        raise ValueError("phi must be (N, L) with L = len(p_in_w)")
    n, nl = phi.shape
    sh = (n, nl)
    phi0 = _nl(phi, sh, "phi")
    k2t = _nl(k2_th, sh, "k2_th")
    k2d = _nl(k2_dr, sh, "k2_dr")
    a0 = _nl(a_rt, sh, "a_rt")
    trt = _nl(t_rt_s, sh, "t_rt_s")
    eta0 = _nl(eta, sh, "eta")
    psi_ = _nl(0.0 if psi is None else psi, sh, "psi")
    if np.any((k2t < 0) | (k2t >= 1)) or np.any((k2d < 0) | (k2d >= 1)):
        raise ValueError("K^2 must lie in [0, 1)")
    if np.any((a0 <= 0) | (a0 > 1)) or np.any(trt <= 0):
        raise ValueError("a_rt must lie in (0, 1] and t_rt_s > 0")
    if np.any((eta0 < 0) | (eta0 > 1)):
        raise ValueError("eta must lie in [0, 1]")
    if not np.all(np.isfinite(p_in_w)) or np.any(p_in_w < 0):
        raise ValueError("p_in_w must be finite and >= 0")
    order = np.arange(n, dtype=np.int64) if bus_order is None else np.ascontiguousarray(bus_order, dtype=np.int64)
    if order.shape != (n,) or not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError("bus_order must be a permutation of range(N)")
    t2 = _nl(1.0 if bus_t2 is None else bus_t2, (n,), "bus_t2")
    if np.any((t2 < 0) | (t2 > 1)) or not 0.0 <= float(bus_t2_out) <= 1.0:
        raise ValueError("bus_t2 and bus_t2_out must lie in [0, 1]")
    if rb is None:
        rb_ = np.zeros((n, 1), np.complex128)
    else:
        rb_ = np.asarray(rb, dtype=np.complex128)
        if rb_.ndim == 1:
            rb_ = np.broadcast_to(rb_, (n, rb_.shape[0]))
        rb_ = np.ascontiguousarray(rb_)
        if rb_.shape[0] != n or rb_.shape[1] < 1:
            raise ValueError("rb must be (M_b,) or (N, M_b)")
        if np.any(np.abs(rb_) >= 2.0):
            raise ValueError("|r_b| must be < 2")
    if nrz is None:
        two = False
        p1 = np.zeros(n)
        phi1, a1, eta1 = phi0, a0, eta0
    else:
        two = True
        p1 = _nl(nrz["p1"], (n,), "p1")
        phi1 = _nl(nrz.get("phi1", phi0), sh, "phi1")
        a1 = _nl(nrz.get("a1", a0), sh, "a1")
        eta1 = _nl(nrz.get("eta1", eta0), sh, "eta1")
        if np.any((p1 < 0) | (p1 > 1)):
            raise ValueError("p1 must lie in [0, 1]")
        if np.any((a1 <= 0) | (a1 > 1)) or np.any((eta1 < 0) | (eta1 > 1)):
            raise ValueError("nrz a1 must lie in (0, 1] and eta1 in [0, 1]")
    return (p_in_w, order, t2, float(bus_t2_out), p1, phi0, phi1, a0, a1, eta0, eta1, psi_, k2t, k2d, trt, rb_,
            two)


def bus_propagate(p_in_w, phi, k2_th, k2_dr, a_rt, t_rt_s, eta, bus_order=None, bus_t2=None, bus_t2_out=1.0,
                  psi=None, rb=None, nrz=None, method="exact", m_orders=3, lam_m=None) -> BusResult:
    """N rings in bus order acting on L laser lines; single pass (module docstring for the bound).

    Args:
        p_in_w: (L,) power per line at the bus input [W].
        phi: (N, L) round-trip phase of ring k at line l (level 0 when nrz is given), incl. junction fill [rad].
        k2_th, k2_dr: per-pass power couplings at each LINE wavelength [-], (N, L) or broadcastable.
        a_rt: round-trip amplitude [-]; t_rt_s: round-trip group delay [s]; eta: absorbed fraction of the intrinsic
            loss [-]; all (N, L) or broadcastable.
        bus_order: (N,) ring indices in the order the light meets them (default 0..N-1).
        bus_t2: (N,) bus power transmission of the segment before ring k [-]; bus_t2_out: after the last ring [-].
        psi: ring-side coupler phase [rad], broadcastable to (N, L).
        rb: complex round-trip backscatter per order, (M_b,) or (N, M_b) [-].
        nrz: None (CW optics) or dict(p1=(N,), phi1=(N, L), a1=(N, L), eta1=(N, L)) for the level-1 state.
        method: "exact" or "tcmt"; m_orders: TCMT orders.
        lam_m: (L,) or (N, L) line wavelengths [m]; if given, the absorbed-power slopes are computed too.
    """
    args = _prepare_bus(p_in_w, phi, k2_th, k2_dr, a_rt, t_rt_s, eta, bus_order, bus_t2, bus_t2_out, psi, rb, nrz)
    (p_in, order, t2, t2o, p1, phi0, phi1, a0, a1, eta0, eta1, psi_, k2t, k2d, trt, rb_, two) = args
    n, nl = phi0.shape
    if method not in ("exact", "tcmt"):
        raise ValueError("method must be 'exact' or 'tcmt'")
    if int(m_orders) < 1:
        raise ValueError("m_orders must be >= 1")
    with_slope = lam_m is not None
    if with_slope:
        lam = _nl(lam_m, (n, nl), "lam_m")
        dphidl = np.ascontiguousarray(TWO_PI * C0 * trt / (lam * lam))
    else:
        dphidl = np.zeros((n, nl))
    out_rl, out_r, out_l = make_bus_outputs(n, nl)
    cw, fw, work = make_workspace(nl)
    bus_propagate_nb(p_in, order, t2, t2o, p1, phi0, phi1, a0, a1, eta0, eta1, psi_, k2t, k2d, trt, rb_, dphidl,
                     two, method == "tcmt", int(m_orders), with_slope, out_rl, out_r, out_l, work, cw, fw)
    return BusResult(p_in=out_rl[RL_PIN], thru=out_rl[RL_THRU], drop=out_rl[RL_DROP], add_out=out_rl[RL_ADD],
                     refl=out_rl[RL_REFL], loss=out_rl[RL_LOSS], absorbed=out_rl[RL_ABS], u_cw=out_rl[RL_UCW],
                     u_ccw=out_rl[RL_UCCW], thru1=out_rl[RL_THRU1], thru0=out_rl[RL_THRU0],
                     drop1=out_rl[RL_DROP1], drop0=out_rl[RL_DROP0], pout1=out_rl[RL_POUT1],
                     pout0=out_rl[RL_POUT0], drop_sum=out_r[R_DROP_SUM], abs_sum=out_r[R_ABS_SUM],
                     loss_sum=out_r[R_LOSS_SUM], drop_sum1=out_r[R_DROP_SUM1], drop_sum0=out_r[R_DROP_SUM0],
                     p_out=out_l[L_POUT], bound=out_l[L_BOUND], refl_in=out_l[L_REFL_IN],
                     dabs_dphi=out_rl[RL_DABS_DPHI], dabs_dlam=out_r[R_DABS_DLAM])


def dpabs_dlambda_res(p_in_ring_w, phi, k2_th, k2_dr, a_rt, t_rt_s, eta, lam_m, psi=None, rb=None, nrz=None,
                      method="exact", m_orders=3):
    """Absorbed power per ring and its slope d P_abs / d lambda_res for the implicit self-heating Newton step (P0-1).

    Args: p_in_ring_w (N, L) power at each ring's input [W] (from :func:`bus_propagate`, ``.p_in``); lam_m (L,) or
    (N, L) line wavelengths [m]; other arguments as :func:`bus_propagate`.
    Returns (P_abs (N,) [W], dP_abs/d lambda_res (N,) [W/m]). The slope is analytic (forward mode through the exact
    loop or the TCMT sum); d lambda_res is a rigid red shift of that ring's comb at fixed input.
    """
    p_in_ring_w = np.ascontiguousarray(np.asarray(p_in_ring_w, dtype=np.float64))
    if p_in_ring_w.ndim != 2:
        raise ValueError("p_in_ring_w must be (N, L)")
    args = _prepare_bus(p_in_ring_w[0], phi, k2_th, k2_dr, a_rt, t_rt_s, eta, None, None, 1.0, psi, rb, nrz)
    (_, _, _, _, p1, phi0, phi1, a0, a1, eta0, eta1, psi_, k2t, k2d, trt, rb_, two) = args
    n, nl = phi0.shape
    lam = _nl(lam_m, (n, nl), "lam_m")
    dphidl = np.ascontiguousarray(TWO_PI * C0 * trt / (lam * lam))
    if p_in_ring_w.shape != (n, nl):
        raise ValueError("p_in_ring_w must be (N, L)")
    cw, fw, _ = make_workspace(nl)
    pabs = np.empty(n)
    slope = np.empty(n)
    for k in range(n):
        pabs[k], slope[k] = dpabs_dlambda_res_nb(k, np.ascontiguousarray(p_in_ring_w[k]), p1[k], phi0, phi1, a0,
                                                 a1, eta0, eta1, psi_, k2t, k2d, trt, rb_, dphidl, two,
                                                 method == "tcmt", int(m_orders), cw, fw)
    return pabs, slope
