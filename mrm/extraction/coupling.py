"""Ring-bus couplers: through and drop (ARCHITECTURE 5.2, 07 P1-5, P1-9, D1, D7).

Run:  python -m mrm.extraction.coupling [--quick] [--xsec rib160 rib200 strip400x200] [--out DIR]
      (through tools/memguard.py --limit-gb 3 --heavy). --quick writes into a temporary directory unless --out.

Geometry (ASSUMED, nothing is public): a straight bus of the ring's own cross-section (same rib, the slab continuous
through the gap) tangent to a ring of radius r_ref (from modes_<xsec>.json); the drop bus is the same coupler at a
larger gap on the opposite side. Gap = edge-to-edge distance at the closest point.

Methods (all derived):
A. primary - non-orthogonal coupled-mode theory with the EXACT fields of the isolated guides: the bus mode
   E_b(x, y) exp(-j beta_b z) and the bent ring mode E_r(r - R, y) exp(-j nu theta) (exact bend transform), rotated
   into the bus frame plane by plane. With c = (a, b) the amplitudes of the two full fields,
       P(z) dc/dz = -j K(z) c,   P_ji = 1/4 int (E_j* x H_i + E_i x H_j*).z dA,   K_ji = (w eps0/4) int (eps - eps_i) E_j*.E_i dA
   (reciprocity derivation for basis fields that each solve Maxwell in their own structure). The curved gap, the
   oblique cut through the ring and the phase mismatch between the bent and straight modes are all contained in the
   fields; no radius convention has to be chosen. Power conservation is reported as a check.
B. the curved-gap supermode CMT asked for in 07 P1-5: kappa(g) = pi (n_even - n_odd)/lambda from FEM supermodes of
   two parallel straight guides, integrated along z with g(z) = g0 + R_o - sqrt(R_o^2 - z^2) and the phase mismatch
   Delta(z) = nu asin(z/R) - beta_b z (bent mode referenced to the ring centreline).
C. validation of A and B against a 2-D full-wave reference: Meep FDTD of a ring arc and bus in a 2-D effective-index
   model, compared with A and B evaluated on the same 2-D model.

Outputs coupling_<xsec>.json (schema coupling/1): per coupler (through, drop) at its design gap kappa2_per_pass at
lambda0, dln_kappa2_dlambda_per_nm, dln_kappa2_dgap_per_nm, coupler_phase_rad and the table kappa2(lambda, gap); the
under- and over-coupled through-gap solutions; drop gaps for the kappa_d/kappa prior; a geometry Jacobian.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from mrm.extraction import common as C
from mrm.extraction import modes as MO

SCHEMA = "coupling/1"
K0C = 273.15

# Design targets (07 D1, P1-9, section 4.8; 01 section 2.2)
Q_L_NOMINAL, Q_L_RANGE = 3850.0, (3500.0, 4200.0)          # talk 3500, linewidth read-off 3900-4200
Q_I_NOMINAL, Q_I_RANGE = 7700.0, (7000.0, 8400.0)          # ~2 Q_L at near-critical coupling (07 D1)
NOTCH_DB_NOMINAL = 27.9                                     # G25 notch at 0 V heater (read-off, 01 section 2.2)
KD_OVER_K_NOMINAL, KD_OVER_K_PRIOR = 0.03, (0.004, 0.18)    # 07 P1-9 prior; nominal = rounded log-centre


@dataclass
class CmtSettings:
    quick: bool = False
    mesh_scale: float = 1.5
    dx_um: float = 0.005
    du_um: float = 0.0025
    dz_um: float = 0.04
    gaps_nm: tuple = (60, 80, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500)
    lam_table_nm: tuple = (1270.0, 1290.0, 1300.0, 1310.0, 1320.0, 1330.0, 1350.0)
    dlam_nm: float = 10.0
    method_b: bool = True
    meep2d: bool = True
    meep_res: tuple = (30, 45, 60)
    meep2d_json: str | None = None
    jac: bool = True

    @classmethod
    def quick_settings(cls):
        return cls(quick=True, mesh_scale=3.0, dx_um=0.01, du_um=0.005, dz_um=0.04, gaps_nm=(100, 200, 300, 400),
                   lam_table_nm=(1300.0, 1310.0, 1320.0), method_b=False, meep2d=False, jac=False)


# --------------------------------------------------------------------------------------------- field sampling


@dataclass
class Sampled:
    """Fields of an isolated, power-normalized mode on a structured grid (physical components, eta0 H)."""
    kind: str                  # "bus" (x, y, z) or "ring" (r - R, y, phi)
    u: np.ndarray              # x (bus) or u = r - R (ring) node coordinates
    y: np.ndarray
    E: tuple                   # (E_1, E_2, E_3) arrays (ny, nu): bus (Ex, Ey, Ez); ring (Er, Ey, Ephi)
    H: tuple
    beta: complex              # bus: beta; ring: nu / R
    R: float | None
    eps: np.ndarray            # the structure's own eps on the grid (ny, nu)
    n_eff: complex
    power_grid: float          # 1/2 Re int (E x H*) . s on the grid after normalization (should be ~1)
    dy: float = 0.0            # y cell height (1 for the per-unit-height 2-D model)


def _probe(m: C.ModeSol, px: np.ndarray, py: np.ndarray):
    b = m.basis
    (bt, it), (bz, iz) = list(zip(b.split_bases(), b.split_indices()))
    pts = np.vstack([px.ravel(), py.ravel()])
    n = pts.shape[1]
    et = bt.probes(pts) @ m.x[it]
    ez = bz.probes(pts) @ m.x[iz]
    return et[:n], et[n:], ez


def sample_mode(m: C.ModeSol, kind: str, u: np.ndarray, y: np.ndarray, eps_fn) -> Sampled:
    """Sample a FEM mode on the grid (u nodes x y nodes). H from Maxwell's curl equation by finite differences of
    the sampled E (E_s is continuous; the derivatives used are of tangential components)."""
    k0 = m.k0
    beta = m.beta
    R = m.radius_um if kind == "ring" else None
    U, Y = np.meshgrid(u, y)
    Ex, Ey, Es = (a.reshape(U.shape) for a in _probe(m, U, Y))
    du = u[1] - u[0]
    dy = y[1] - y[0]
    dEs_du = np.gradient(Es, du, axis=1)
    dEs_dy = np.gradient(Es, dy, axis=0)
    dEy_du = np.gradient(Ey, du, axis=1)
    dEx_dy = np.gradient(Ex, dy, axis=0)
    rho = (1.0 + U / R) if R else np.ones_like(U)
    # transformed-frame eta0 H (mu' = diag(rho, rho, 1/rho) for the exact bend; 1 for straight)
    Hx = 1j / (k0 * rho) * (dEs_dy + 1j * beta * Ey)
    Hy = 1j / (k0 * rho) * (-1j * beta * Ex - dEs_du)
    Hs = 1j * rho / k0 * (dEy_du - dEx_dy)
    E3, H3 = Es / rho, Hs / rho                       # physical E_phi, H_phi (= E_z, H_z for the bus)
    eps = eps_fn(U, Y)
    # normalize: 1/2 Re int (E x H*).s dA = 1 using the FEM integral (exact quadrature)
    pw = 0.5 * m.power(None)
    s = 1.0 / math.sqrt(pw)
    E = (Ex * s, Ey * s, E3 * s)
    H = (Hx * s, Hy * s, H3 * s)
    pg = 0.5 * float(np.real(np.sum(E[0] * np.conj(H[1]) - E[1] * np.conj(H[0]))) * du * dy)
    return Sampled(kind, u, y, E, H, beta, R, eps, m.n_eff, pg, dy)


# --------------------------------------------------------------------------------------------- geometry maps


@dataclass
class CouplerGeom:
    xs: C.XSec
    R: float                   # ring centreline radius (um)
    w_r: float                 # ring width (um)
    w_b: float                 # bus width (um)
    n: dict                    # indices by material
    core_y: tuple              # (y0, y1) of the rib above the slab (strip: (0, t))
    two_d: bool = False        # 2-D model: cores span all y

    def bg_eps(self, Y):
        """Background permittivity of both structures (BOX, slab, cladding) as a function of y."""
        if self.two_d:
            return np.full_like(Y, self.n["clad"] ** 2, dtype=float)
        e = np.where(Y < 0.0, self.n["box"] ** 2, self.n["clad"] ** 2)
        s = self.xs.slab_nm * 1e-3
        if s > 0:
            e = np.where((Y >= 0.0) & (Y < s), self.n["si"] ** 2, e)
        return e

    def in_core_y(self, Y):
        if self.two_d:
            return np.ones_like(Y, dtype=bool)
        return (Y >= self.core_y[0]) & (Y <= self.core_y[1])

    def eps_bus(self, X, Y):
        e = self.bg_eps(Y)
        return np.where((np.abs(X) < self.w_b / 2) & self.in_core_y(Y), self.n["si"] ** 2, e)

    def eps_ring_local(self, U, Y):
        e = self.bg_eps(Y)
        return np.where((np.abs(U) < self.w_r / 2) & self.in_core_y(Y), self.n["si"] ** 2, e)


def geom_for(xs: C.XSec, R: float, lam_um: float, t_k: float, n: dict | None = None) -> CouplerGeom:
    n = n or C.indices(xs, lam_um, t_k)
    t, s = xs.film_nm * 1e-3, xs.slab_nm * 1e-3
    return CouplerGeom(xs, R, xs.width_nm * 1e-3, xs.width_nm * 1e-3, n, (s, t))


# --------------------------------------------------------------------------------------------- method A


def cmt_overlap(bus: Sampled, ring: Sampled, g: CouplerGeom, gap_um: float, *, dz: float = 0.04,
                z_max: float | None = None, x_pad: float = 1.1, return_profile: bool = False) -> dict:
    """Method A for one gap: integrate P c' = -j K c along z for both inputs; returns the 2x2 transfer matrix."""
    k0 = float(np.real(bus.beta / bus.n_eff))            # 2 pi / lambda
    R, wr, wb = g.R, g.w_r, g.w_b
    D = R + wr / 2 + gap_um + wb / 2                     # ring centre at x = -D
    xb = bus.u
    sel = (xb >= -(wr / 2 + gap_um + wb / 2) - 0.7) & (xb <= x_pad)
    x = xb[sel]
    dx = x[1] - x[0]
    y = bus.y
    dy = bus.dy
    Eb = [a[:, sel] for a in bus.E]
    Hb = [a[:, sel] for a in bus.H]
    X1, Y1 = np.meshgrid(x, y)
    corey = g.in_core_y(Y1)
    bg = g.bg_eps(Y1)
    d_eps = g.n["si"] ** 2 - bg                          # Delta eps of a rib core at each y
    bus_core = (np.abs(X1) < wb / 2) & corey
    nu = ring.beta * R
    if z_max is None:
        z_max = min(0.8 * R, math.sqrt(2.0 * (R + wr / 2) * 1.4))   # gap grows by 1.4 um at the ends
    zs = np.arange(-z_max, z_max + dz / 2, dz)
    # ring sample: interpolate D_r = eps E_r (continuous across the sidewalls), E_y, E_phi, H in u at each x
    ur = ring.u
    eps_r = ring.eps
    Dr = ring.E[0] * eps_r
    comps = [Dr, ring.E[1], ring.E[2], ring.H[0], ring.H[1], ring.H[2]]
    Ps, Ks = [], []
    prof = []
    for z in zs:
        Xc = x + D
        r = np.sqrt(Xc ** 2 + z ** 2)
        th = np.arctan2(z, Xc)
        uu = r - R
        ok = (uu > ur[0]) & (uu < ur[-1])
        j = np.clip(np.searchsorted(ur, uu) - 1, 0, len(ur) - 2)
        wgt = np.where(ok, (uu - ur[j]) / (ur[1] - ur[0]), 0.0)
        mask = ok.astype(float)

        def interp(F):
            return (F[:, j] * (1 - wgt) + F[:, j + 1] * wgt) * mask
        Dr_, Ey_, Ep_, Hr_, Hy_, Hp_ = (interp(F) for F in comps)
        # ring core coverage of each x cell at this z (exact 1-D overlap) and the field inside the core
        rin, rout = R - wr / 2, R + wr / 2
        if abs(z) < rin:
            xin, xout = math.sqrt(rin ** 2 - z ** 2) - D, math.sqrt(rout ** 2 - z ** 2) - D
        else:
            xin, xout = -1e9, math.sqrt(max(rout ** 2 - z ** 2, 0.0)) - D
        cov = np.clip((np.minimum(x + dx / 2, xout) - np.maximum(x - dx / 2, xin)) / dx, 0.0, 1.0)
        eps_loc = np.where((np.abs(uu)[None, :] < wr / 2) & corey, g.n["si"] ** 2, bg)
        Er_ = Dr_ / eps_loc
        Er_in = Dr_ / g.n["si"] ** 2
        c, s = np.cos(th)[None, :], np.sin(th)[None, :]
        ph = np.exp(-1j * nu * th)[None, :]
        Erx = (Er_ * c - Ep_ * s) * ph
        Ery = Ey_ * ph
        Erz = (Er_ * s + Ep_ * c) * ph
        Erx_in = (Er_in * c - Ep_ * s) * ph
        Erz_in = (Er_in * s + Ep_ * c) * ph
        Hrx = (Hr_ * c - Hp_ * s) * ph
        Hry = Hy_ * ph
        pb = np.exp(-1j * bus.beta * z)
        Ebx, Eby, Ebz = (a * pb for a in Eb)
        Hbx, Hby = Hb[0] * pb, Hb[1] * pb
        dA = dx * dy

        def cross(Ej, Hj, Ei, Hi):
            # 1/4 int (Ej* x Hi + Ei x Hj*) . z
            return 0.25 * np.sum(np.conj(Ej[0]) * Hi[1] - np.conj(Ej[1]) * Hi[0]
                                 + Ei[0] * np.conj(Hj[1]) - Ei[1] * np.conj(Hj[0])) * dA
        Pbb = cross((Ebx, Eby), (Hbx, Hby), (Ebx, Eby), (Hbx, Hby))
        Prr = cross((Erx, Ery), (Hrx, Hry), (Erx, Ery), (Hrx, Hry))
        Pbr = cross((Ebx, Eby), (Hbx, Hby), (Erx, Ery), (Hrx, Hry))     # j = b, i = r
        Prb = cross((Erx, Ery), (Hrx, Hry), (Ebx, Eby), (Hbx, Hby))
        ring_core = cov[None, :] * corey
        wring = d_eps * ring_core
        wbus = d_eps * bus_core
        dotb_rin = np.conj(Ebx) * Erx_in + np.conj(Eby) * Ery + np.conj(Ebz) * Erz_in
        Kbb = 0.25 * k0 * np.sum(wring * (np.abs(Ebx) ** 2 + np.abs(Eby) ** 2 + np.abs(Ebz) ** 2)) * dA
        Krr = 0.25 * k0 * np.sum(wbus * (np.abs(Erx) ** 2 + np.abs(Ery) ** 2 + np.abs(Erz) ** 2)) * dA
        Kbr = 0.25 * k0 * np.sum(wbus * (np.conj(Ebx) * Erx + np.conj(Eby) * Ery + np.conj(Ebz) * Erz)) * dA
        Krb = 0.25 * k0 * np.sum(wring * np.conj(dotb_rin)) * dA
        # P_bb = P_rr = 1 exactly: both isolated modes are power normalized and the near-side half plane z = const
        # carries all of the ring's power (its flow lines are circles about the ring centre). Only the cross powers
        # and the K's are integrated; the window holds every region where both fields are non-negligible.
        Ps.append(np.array([[1.0, Pbr], [Prb, 1.0]]))
        Ks.append(np.array([[Kbb, Kbr], [Krb, Krr]]))
        if return_profile:
            prof.append({"z": float(z), "Pbr": complex(Pbr), "Kbr": complex(Kbr), "Krb": complex(Krb),
                         "Kbb": complex(Kbb), "Krr": complex(Krr), "Prr": complex(Prr)})
    Ps, Ks = np.array(Ps), np.array(Ks)
    # c' = -j P^-1 K c, midpoint exponential steps
    M = np.array([-1j * np.linalg.solve(P, K) for P, K in zip(Ps, Ks)])
    T = np.eye(2, dtype=complex)
    for k in range(len(zs) - 1):
        T = expm(0.5 * (M[k] + M[k + 1]) * dz) @ T
    # amplitudes are coefficients of the full fields; at +-z_max the guides are uncoupled
    t_b, k_br = T[0, 0], T[1, 0]          # bus input: bus out, ring out
    k_rb, t_r = T[0, 1], T[1, 1]          # ring input
    Pend = Ps[-1]
    cout = T @ np.array([1, 0])
    pout = float(np.real(np.conj(cout) @ Pend @ cout))
    out = {"gap_nm": gap_um * 1e3, "kappa2": float(abs(k_br) ** 2), "t2": float(abs(t_b) ** 2),
           "t_bus": complex(t_b), "kappa_bus_to_ring": complex(k_br),
           "kappa_ring_to_bus": complex(k_rb), "t_ring": complex(t_r), "power_out_over_in": pout,
           "ring_self_power_z0_grid": float(np.real(Ps[len(zs) // 2][1, 1])) if False else None,
           "residual_overlap_end": float(abs(Pend[0, 1])), "z_max_um": z_max, "dz_um": dz,
           "first_order_kappa": complex(-1j * np.trapezoid(np.array([K[1, 0] for K in Ks]), zs))}
    if return_profile:
        out["profile"] = prof
    return out


# --------------------------------------------------------------------------------------------- method B


def supermode_kappa(ex: MO.Extractor, xs: C.XSec, lam_um: float, t_k: float, gaps_um, scale: float) -> list:
    """kappa(g) = pi (n_even - n_odd)/lambda from FEM supermodes of two parallel straight guides (shared slab)."""
    out = []
    n = C.indices(xs, lam_um, t_k)
    w = xs.width_nm * 1e-3
    for g in gaps_um:
        xc = (w + g) / 2
        dom = C.Domain(-xc - w / 2 - 1.6, xc + w / 2 + 1.6, -1.5 if xs.below_box == "sio2" else -1.0, 1.5)
        mesh = C.build_mesh(xs, dom, C.MeshSpec(scale=scale), guides=[(-xc, xs.width_nm), (xc, xs.width_nm)])
        P = C.FemProblem(mesh)
        ms = P.solve(n, lam_um, num_modes=6, n_guess=getattr(ex, "_neff_guess", None) or 2.5)
        te = [m for m in ms if m.te_fraction() > 0.5
              and (m.power_fraction("core0") + m.power_fraction("core1")) > 0.3]
        te.sort(key=lambda m: -np.real(m.n_eff))
        ne, no = float(np.real(te[0].n_eff)), float(np.real(te[1].n_eff))
        out.append({"gap_nm": g * 1e3, "n_even": ne, "n_odd": no, "kappa_per_um": math.pi * (ne - no) / lam_um})
    return out


def cmt_supermode(kap_rows: list, R: float, w: float, gap_um: float, n_bus: float, n_ring_bent: float, lam_um: float,
                  *, phase_ref: str = "centreline", dz: float = 0.01, z_max: float | None = None) -> dict:
    g_tab = np.array([r["gap_nm"] for r in kap_rows]) * 1e-3
    lk = np.log(np.array([r["kappa_per_um"] for r in kap_rows]))
    p = np.polyfit(g_tab, lk, 2)
    k0 = 2 * math.pi / lam_um
    Ro = R + w / 2
    nu = n_ring_bent * k0 * R
    bb = n_bus * k0
    if z_max is None:
        z_max = min(5.0, 0.8 * R)
    zs = np.arange(-z_max, z_max + dz / 2, dz)
    gz = gap_um + Ro - np.sqrt(Ro ** 2 - zs ** 2)
    kz = np.exp(np.polyval(p, gz))
    rc = {"centreline": R, "outer_edge": Ro, "gap_centre": Ro + gap_um / 2}[phase_ref]
    phi_r = nu * np.arcsin(np.clip(zs / rc, -1, 1)) * (R / R)
    delta = phi_r - bb * zs
    T = np.eye(2, dtype=complex)
    for k in range(len(zs) - 1):
        km = 0.5 * (kz[k] + kz[k + 1])
        dm = 0.5 * (delta[k] + delta[k + 1])
        M = np.array([[0, -1j * km * np.exp(-1j * dm)], [-1j * km * np.exp(1j * dm), 0]])
        T = expm(M * dz) @ T
    return {"gap_nm": gap_um * 1e3, "kappa2": float(abs(T[1, 0]) ** 2), "t_bus": complex(T[0, 0]),
            "t_ring": complex(T[1, 1]), "phase_ref": phase_ref,
            "kappa2_no_mismatch": float(math.sin(np.trapezoid(kz, zs)) ** 2)}


# --------------------------------------------------------------------------------------------- per-lambda modes


class CouplerModel:
    def __init__(self, xs: C.XSec, R: float, st: CmtSettings, t_k: float | None = None):
        self.xs, self.R, self.st = xs, R, st
        self.t_k = (C.T_REF_C + K0C) if t_k is None else t_k
        self.ex = MO.Extractor(xs, MO.Settings(mesh_scale=st.mesh_scale))
        self._cache = {}

    def sampled(self, lam_um: float, xs: C.XSec | None = None):
        xs = xs or self.xs
        key = (lam_um, xs.width_nm, xs.film_nm, xs.slab_nm)
        if key in self._cache:
            return self._cache[key]
        g = geom_for(xs, self.R, lam_um, self.t_k)
        mb = self.ex.mode(lam=lam_um, xs=xs)
        mr = self.ex.mode(lam=lam_um, xs=xs, R=self.R)
        st = self.st
        y = np.arange(-0.6 + st.dx_um / 2, 0.8, st.dx_um)      # cell centres; interfaces on cell boundaries
        w = xs.width_nm * 1e-3
        xb = np.arange(-1.95 + st.dx_um / 2, 1.2, st.dx_um)       # inside the FEM window [-2, 2]
        bus = sample_mode(mb, "bus", xb, y, g.eps_bus)
        ur = np.arange(-w / 2 - 0.4 + st.du_um / 2, 1.7, st.du_um)
        ring = sample_mode(mr, "ring", ur, y, g.eps_ring_local)
        self._cache = {key: (bus, ring, g)} if len(self._cache) > 6 else {**self._cache, key: (bus, ring, g)}
        return bus, ring, g

    def kappa2(self, lam_um: float, gap_nm: float, xs: C.XSec | None = None, **kw) -> dict:
        bus, ring, g = self.sampled(lam_um, xs)
        return cmt_overlap(bus, ring, g, gap_nm * 1e-3, dz=self.st.dz_um, **kw)


def _interp_gap_for_k2(gaps, k2, target):
    """Gap (nm) at which kappa2(gap) = target, by interpolation of ln kappa2 (monotone decreasing in gap)."""
    lg = np.log(np.asarray(k2))
    gs = np.asarray(gaps, float)
    if not (lg.min() <= math.log(target) <= lg.max()):
        return None
    order = np.argsort(lg)
    from scipy.interpolate import PchipInterpolator
    f = PchipInterpolator(lg[order], gs[order])
    return float(f(math.log(target)))


def q_to_k2(q: float, lam_um: float, fsr_hz: float) -> float:
    """Per-pass power coupling for a decay-rate Q: K2 = 1 - exp(-omega T_rt / Q), T_rt = 1/FSR."""
    nu = C.C0 / (lam_um * 1e-6)
    return float(-math.expm1(-2 * math.pi * nu / fsr_hz / q))


def design_gaps(gaps, k2, lam_um: float, fsr_hz: float) -> dict:
    """Through-gap solutions (critical, under- and over-coupled branches) and drop gaps for the kappa_d/kappa prior."""
    out = {"targets": {"q_loaded_nominal": Q_L_NOMINAL, "q_loaded_range": Q_L_RANGE, "q_i_nominal": Q_I_NOMINAL,
                       "q_i_range": Q_I_RANGE, "notch_db_nominal": NOTCH_DB_NOMINAL,
                       "kappa_d_over_kappa_nominal": KD_OVER_K_NOMINAL, "kappa_d_over_kappa_prior": KD_OVER_K_PRIOR,
                       "fsr_thz": fsr_hz / 1e12}}

    def gap_for_q(qe):
        k2 = q_to_k2(qe, lam_um, fsr_hz)
        return {"q_e": qe, "kappa2": k2, "gap_nm": _interp_gap_for_k2(gaps, k2_tab, k2)}
    k2_tab = k2
    # critical coupling at the nominal Q_i and drop: kappa_e = kappa_i + kappa_d, Q_L = 1/(1/Qi + 1/Qe + 1/Qd)
    # with kappa_d = x kappa: 1/Qd = x/Q_L and 1/Qe = 1/Qi + x/Q_L  ->  1/Q_L = 2/Qi + 2x/Q_L
    x = KD_OVER_K_NOMINAL
    ql_crit = Q_I_NOMINAL * (1 - 2 * x) / 2.0
    qe_crit = 1.0 / (1.0 / Q_I_NOMINAL + x / ql_crit)
    out["critical"] = {**gap_for_q(qe_crit), "q_loaded": ql_crit, "q_i": Q_I_NOMINAL, "kappa_d_over_kappa": x}
    # mirror branches at the nominal Q_L and notch depth (07 P1-5): kappa_i' + kappa_e = kappa, |kappa_i' - kappa_e| = r kappa
    r = 10 ** (-NOTCH_DB_NOMINAL / 20.0)
    for name, sgn in (("under", -1), ("over", +1)):
        rows = []
        for ql in (Q_L_RANGE[0], Q_L_NOMINAL, Q_L_RANGE[1]):
            qe = 2 * ql / (1 + sgn * r)
            qi_eff = 2 * ql / (1 - sgn * r)           # Q of kappa_i + kappa_d
            qd = ql / x
            qi = 1.0 / (1.0 / qi_eff - 1.0 / qd)
            rows.append({**gap_for_q(qe), "q_loaded": ql, "q_i_implied": qi, "notch_db": NOTCH_DB_NOMINAL})
        out[name] = {"nominal": rows[1], "q_loaded_range": [rows[0], rows[2]]}
    # drop gaps for the prior
    out["drop"] = {}
    for label, xx in (("low", KD_OVER_K_PRIOR[0]), ("nominal", KD_OVER_K_NOMINAL), ("high", KD_OVER_K_PRIOR[1])):
        qd = ql_crit / xx
        out["drop"][label] = {**gap_for_q(qd), "kappa_d_over_kappa": xx, "q_d": qd}
    return out


# --------------------------------------------------------------------------------------------- method C (2-D)


MEEP2D = r'''
import json, sys, time, math
import numpy as np
import meep as mp
p = json.loads(sys.argv[1])
res, R, w, g, n1, n2, lam = p["res"], p["R"], p["w"], p["gap"], p["n1"], p["n2"], p["lam"]
dpml = 1.0
D = R + w + g                       # ring centre at x = -D (bus centre at x = 0, along y)
xmin, xmax = -D - 1.5, 1.5 + w / 2
ymin, ymax = -R - 2.0, R + 2.0
sx, sy = xmax - xmin + 2 * dpml, ymax - ymin + 2 * dpml
cx, cy = 0.5 * (xmin + xmax), 0.0
mp.verbosity(0)
core, clad = mp.Medium(index=n1), mp.Medium(index=n2)
geom = [mp.Block(center=mp.Vector3(0 - cx, 0), size=mp.Vector3(w, mp.inf), material=core),
        mp.Cylinder(radius=R + w / 2, center=mp.Vector3(-D - cx, 0), material=core),
        mp.Cylinder(radius=R - w / 2, center=mp.Vector3(-D - cx, 0), material=clad),
        # the far half of the ring is removed: the arc leaves through straight guides into the PML on the left
        mp.Block(center=mp.Vector3(-D - cx - (R + 2) / 2 - 0.0, 0), size=mp.Vector3(R + 2, 2 * R + 2 * w + 0.2),
                 material=clad),
        mp.Block(center=mp.Vector3(-D - cx - (R + 4) / 2, R), size=mp.Vector3(R + 4, w), material=core),
        mp.Block(center=mp.Vector3(-D - cx - (R + 4) / 2, -R), size=mp.Vector3(R + 4, w), material=core)]
fcen = 1 / lam
df = 0.15 * fcen
src = [mp.EigenModeSource(mp.GaussianSource(fcen, fwidth=df), center=mp.Vector3(0 - cx, ymin + 0.5),
                          size=mp.Vector3(w + 2.0, 0), direction=mp.Y, eig_band=1, eig_parity=mp.EVEN_Z)]
# EVEN_Z = TE in 2-D (Ex, Ey, Hz; E in the plane), the polarization of the rib's TE0 in the effective-index model
sim = mp.Simulation(cell_size=mp.Vector3(sx, sy), geometry=geom, sources=src, resolution=res,
                    boundary_layers=[mp.PML(dpml)], default_material=clad)
nf = 5
freqs = [fcen * (1 + k) for k in np.linspace(-0.04, 0.04, nf)]
fin = sim.add_mode_monitor(freqs, mp.ModeRegion(center=mp.Vector3(0 - cx, ymin + 1.0), size=mp.Vector3(w + 2.0, 0)))
fbus = sim.add_mode_monitor(freqs, mp.ModeRegion(center=mp.Vector3(0 - cx, ymax - 0.5), size=mp.Vector3(w + 2.0, 0)))
fring = sim.add_mode_monitor(freqs, mp.ModeRegion(center=mp.Vector3(-D - cx - 0.75, R), size=mp.Vector3(0, w + 2.0)))
t0 = time.time()
# the pulse must reach the arc exit (~ (R + 2 + pi R / 2) * n_g) before any decay test
t_min = (R + 2.0 + math.pi * R / 2 + 2.0) * 3.5 + 40.0
sim.run(until_after_sources=t_min)
sim.run(until_after_sources=mp.stop_when_fields_decayed(20, mp.Hz, mp.Vector3(-D - cx - 0.75, R), 1e-6))
ci = sim.get_eigenmode_coefficients(fin, [1], eig_parity=mp.EVEN_Z).alpha[0, :, 0]
cb = sim.get_eigenmode_coefficients(fbus, [1], eig_parity=mp.EVEN_Z).alpha[0, :, 0]
cr = sim.get_eigenmode_coefficients(fring, [1], eig_parity=mp.EVEN_Z).alpha[0, :, 1]
pin = np.abs(ci) ** 2
print(json.dumps({"res": res, "gap": g, "freqs": freqs, "lam_um": [1 / f for f in freqs],
                  "kappa2": (np.abs(cr) ** 2 / pin).tolist(), "t2": (np.abs(cb) ** 2 / pin).tolist(),
                  "seconds": round(time.time() - t0, 1)}))
'''


def meep2d_runs(*, n1: float, n2: float, w: float, R: float, lam: float, gaps_um, resolutions) -> list:
    """The Meep 2-D runs of method C (light: < 0.3 GB, single core); one subprocess per (gap, resolution)."""
    out = []
    if not os.path.exists(C.MEEP_PYTHON):
        return [{"error": "meep interpreter not found"}]
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "meep2d.py"
        sp.write_text(MEEP2D)
        for gp in gaps_um:
            for res in resolutions:
                par = {"res": res, "R": R, "w": w, "gap": gp, "n1": n1, "n2": n2, "lam": lam}
                r = subprocess.run([C.MEEP_PYTHON, "-I", str(sp), json.dumps(par)], capture_output=True,
                                   text=True, timeout=7200)
                line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
                if line:
                    d = json.loads(line[-1])
                    mid = len(d["kappa2"]) // 2
                    out.append({"gap_nm": gp * 1e3, "res": res, "kappa2": d["kappa2"][mid], "t2": d["t2"][mid],
                                "kappa2_vs_lam": d["kappa2"], "lam_um": d["lam_um"], "seconds": d["seconds"],
                                "params": par})
                else:
                    out.append({"gap_nm": gp * 1e3, "res": res, "error": (r.stderr or r.stdout)[-300:], "params": par})
                C.log(f"  meep2d gap {gp} res {res}: {out[-1].get('kappa2')}")
    return out


def eim_indices(xs: C.XSec, lam: float, t_k: float) -> tuple[float, float]:
    """2-D effective-index model of a cross-section: film TE slab index (core) and slab TE index or cladding."""
    n = C.indices(xs, lam, t_k)
    n1 = MO.slab_neff(n["box"], n["si"], n["clad"], xs.film_nm * 1e-3, lam, "TE")
    n2 = (MO.slab_neff(n["box"], n["si"], n["clad"], xs.slab_nm * 1e-3, lam, "TE") if xs.is_rib else n["clad"])
    return n1, n2


MEEP2D_GAPS_UM = {"rib160": (0.15, 0.30), "rib200": (0.15, 0.30), "strip400x200": (0.10, 0.20)}


def meep2d_validation(st: CmtSettings, *, n1: float, n2: float, w: float, R: float, lam: float, gaps_um,
                      resolutions=(30, 40, 60), meep_rows: list | None = None) -> dict:
    """Method C: 2-D full-wave K^2 vs methods A and B on the same 2-D (effective-index) model."""
    import shapely
    from femwell.mesh import mesh_from_OrderedDict
    from skfem.io.meshio import from_meshio
    rows = {"meep": [], "A": [], "B": []}
    # 2-D modes by the y-uniform FEM (PMC top/bottom gives the H-polarized, in-plane-E TE mode)
    hy = 0.04

    def mesh2d(xl, xr, h=0.005):
        polys = OrderedDict(core=shapely.box(-w / 2, -hy / 2, w / 2, hy / 2), clad=shapely.box(xl, -hy / 2, xr, hy / 2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return from_meshio(mesh_from_OrderedDict(polys, {"core": {"resolution": h, "distance": 0.5, "SizeMax": 0.02}},
                                                     default_resolution_max=0.02))
    P = C.FemProblem(mesh2d(-2.2, 2.6))
    nmat = {"si": n1, "clad": n2, "box": n2, "air": 1.0}
    mb = C.pick_mode(P.solve(nmat, lam, num_modes=3, n_guess=n1 * 0.95), "TE0")
    mr = C.pick_mode(P.solve(nmat, lam, radius_um=R, num_modes=3, n_guess=n1 * 0.95), "TE0")
    xs2 = C.XSec("2d", width_nm=w * 1e3, film_nm=hy * 1e3, slab_nm=0.0)
    g2 = CouplerGeom(xs2, R, w, w, nmat, (-hy, hy), two_d=True)
    y = np.array([0.0])
    dxu = st.dx_um
    xb = np.arange(-2.0 + dxu / 2, 2.0, dxu)

    # direct 1-D sampling (fields are y-uniform): probe at y = 0, d/dy = 0
    bus = _sample_1d(mb, "bus", xb, lambda X, Y: g2.eps_bus(X, Y))
    ur = np.arange(-w / 2 - 0.5 + st.du_um / 2, 2.4, st.du_um)
    ring = _sample_1d(mr, "ring", ur, lambda U, Y: g2.eps_ring_local(U, Y))
    for gp in gaps_um:
        a = cmt_overlap(bus, ring, g2, gp, dz=st.dz_um)
        rows["A"].append({"gap_nm": gp * 1e3, "kappa2": a["kappa2"], "power_out_over_in": a["power_out_over_in"]})
    # method B in 2-D: supermodes of two parallel 2-D guides
    kap = []
    for gp in sorted(set(list(gaps_um) + [min(gaps_um) * 0.8, max(gaps_um) * 1.2])):
        xc = (w + gp) / 2
        polys = OrderedDict(core0=shapely.box(-xc - w / 2, -hy / 2, -xc + w / 2, hy / 2),
                            core1=shapely.box(xc - w / 2, -hy / 2, xc + w / 2, hy / 2),
                            clad=shapely.box(-xc - w / 2 - 2.0, -hy / 2, xc + w / 2 + 2.0, hy / 2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mm = from_meshio(mesh_from_OrderedDict(polys, {"core0": {"resolution": 0.005, "distance": 0.5, "SizeMax": 0.02},
                                                          "core1": {"resolution": 0.005, "distance": 0.5, "SizeMax": 0.02}},
                                                   default_resolution_max=0.02))
        PP = C.FemProblem(mm)
        ms = PP.solve(nmat, lam, num_modes=4, n_guess=n1 * 0.95)
        te = sorted([m for m in ms if m.te_fraction() > 0.5], key=lambda m: -np.real(m.n_eff))
        kap.append({"gap_nm": gp * 1e3, "kappa_per_um": math.pi * float(np.real(te[0].n_eff - te[1].n_eff)) / lam})
    for gp in gaps_um:
        b = cmt_supermode(kap, R, w, gp, float(np.real(mb.n_eff)), float(np.real(mr.n_eff)), lam)
        rows["B"].append({"gap_nm": gp * 1e3, "kappa2": b["kappa2"],
                          "kappa2_outer_edge_ref": cmt_supermode(kap, R, w, gp, float(np.real(mb.n_eff)),
                                                                 float(np.real(mr.n_eff)), lam,
                                                                 phase_ref="outer_edge")["kappa2"]})
    if meep_rows is not None:
        rows["meep"] = list(meep_rows)
    else:
        rows["meep"] = meep2d_runs(n1=n1, n2=n2, w=w, R=R, lam=lam, gaps_um=gaps_um, resolutions=resolutions)
    return {"model": {"n_core": n1, "n_clad": n2, "width_um": w, "R_um": R, "lambda_um": lam,
                      "note": "2-D effective-index ring arc + bus (TE, E in plane); the arc leaves the coupler into "
                              "straight guides that end in the PML, so there is no resonance"},
            "n_eff_bus": float(np.real(mb.n_eff)), "n_eff_ring_bent": float(np.real(mr.n_eff)), **rows}


def _sample_1d(m: C.ModeSol, kind: str, u: np.ndarray, eps_fn) -> Sampled:
    """y-uniform mode: sample along y = 0 and treat the y extent as one cell of unit height (per-unit-height
    integrals for the 2-D model)."""
    k0, beta = m.k0, m.beta
    R = m.radius_um if kind == "ring" else None
    Ex, Ey, Es = _probe(m, u, np.zeros_like(u))
    du = u[1] - u[0]
    dEs = np.gradient(Es, du)
    dEy = np.gradient(Ey, du)
    rho = (1 + u / R) if R else np.ones_like(u)
    Hx = 1j / (k0 * rho) * (1j * beta * Ey)
    Hy = 1j / (k0 * rho) * (-1j * beta * Ex - dEs)
    Hs = 1j * rho / k0 * dEy
    E3, H3 = Es / rho, Hs / rho
    pw = 0.5 * float(np.real(np.sum(Ex * np.conj(Hy) - Ey * np.conj(Hx)) * du))
    s = 1.0 / math.sqrt(pw)
    E = tuple((a * s)[None, :] for a in (Ex, Ey, E3))
    H = tuple((a * s)[None, :] for a in (Hx, Hy, H3))
    eps = eps_fn(u[None, :], np.zeros((1, len(u))))
    return Sampled(kind, u, np.array([0.0]), E, H, beta, R, eps, m.n_eff, 1.0, 1.0)


# --------------------------------------------------------------------------------------------- driver


def load_r_ref(xs: C.XSec, out_dir: Path) -> tuple[float, dict]:
    for d in (out_dir, C.DATA_DIR):
        p = Path(d) / f"modes_{xs.name}.json"
        if p.exists():
            m = json.loads(p.read_text())
            return float(m["r_ref_um"]), {"source": str(p.name), "sha256": C.sha256_file(p), "n_g": m["n_g"],
                                          "fsr_thz": m.get("fsr_thz")}
    ex = MO.Extractor(xs, MO.Settings(mesh_scale=3.0))
    rr = MO.find_r_ref(ex)
    return float(rr["r_ref_um"]), {"source": "computed here (no modes artefact)", "r_ref": rr}


def extract_coupling(xs: C.XSec, st: CmtSettings, out_dir: Path) -> dict:
    t0 = time.time()
    R, rsrc = load_r_ref(xs, out_dir)
    fsr = C.FSR_DESIGN_HZ
    lam0 = xs.lam0_um
    C.log(f"=== coupling {xs.name}: r_ref = {R:.4f} um")
    cm = CouplerModel(xs, R, st)
    bus, ring, g = cm.sampled(lam0)
    sampling = {"bus_power_on_grid": bus.power_grid, "ring_power_on_grid": ring.power_grid,
                "n_eff_bus": complex(bus.n_eff), "n_eff_ring_bent": complex(ring.n_eff)}
    C.log(f"  sampling check: bus {bus.power_grid:.5f} ring {ring.power_grid:.5f}")
    # K^2(gap) at lambda0 (method A)
    rowsA = []
    for gp in st.gaps_nm:
        a = cm.kappa2(lam0, gp)
        rowsA.append(a)
        C.log(f"  A gap {gp}: K2 {a['kappa2']:.5f} t2 {a['t2']:.5f} P_out/P_in {a['power_out_over_in']:.5f} "
              f"arg t_ring {np.angle(a['t_ring']):.4f}")
    gaps = [r["gap_nm"] for r in rowsA]
    k2 = [r["kappa2"] for r in rowsA]
    des = design_gaps(gaps, k2, lam0, fsr)
    g_thr = des["critical"]["gap_nm"]
    g_drop = des["drop"]["nominal"]["gap_nm"]
    C.log(f"  design: through {g_thr} nm, drop {g_drop} nm")
    # convergence of method A at the through gap: grid (dx, du) and dz
    conv = []
    if not st.quick:
        for dx, du, dz in ((0.01, 0.005, 0.04), (0.0075, 0.00375, 0.03), (st.dx_um, st.du_um, st.dz_um),
                           (st.dx_um, st.du_um, st.dz_um / 2)):
            st2 = CmtSettings(**{**st.__dict__, "dx_um": dx, "du_um": du, "dz_um": dz})
            cm2 = CouplerModel(xs, R, st2)
            cm2.ex = cm.ex
            a = cm2.kappa2(lam0, g_thr)
            conv.append({"dx_um": dx, "du_um": du, "dz_um": dz, "kappa2": a["kappa2"],
                         "power_out_over_in": a["power_out_over_in"]})
            C.log(f"  conv {conv[-1]}")
        # FEM mesh convergence of the coupling (mode fields): scale 2.0 and 1.0
        for sc in (2.0, 1.0):
            st3 = CmtSettings(**{**st.__dict__, "mesh_scale": sc})
            cm3 = CouplerModel(xs, R, st3)
            a = cm3.kappa2(lam0, g_thr)
            conv.append({"mesh_scale": sc, "kappa2": a["kappa2"]})
            C.log(f"  conv mesh {conv[-1]}")
    # kappa2(lambda, gap) table and dispersion
    lam_tab = [l * 1e-3 for l in st.lam_table_nm]
    if lam0 * 1e3 not in st.lam_table_nm:
        lam_tab = sorted(lam_tab + [lam0])
    table_gaps = sorted(set(list(st.gaps_nm) + [round(g_thr, 1), round(g_drop, 1)] if g_thr and g_drop else st.gaps_nm))
    table = []
    for lam in lam_tab:
        row = [cm.kappa2(lam, gp)["kappa2"] for gp in table_gaps]
        table.append(row)
        C.log(f"  table lambda {lam * 1e3:.1f}: {[round(v, 5) for v in row]}")
    table = np.array(table)

    def coupler_block(gap_nm, label):
        if gap_nm is None:
            return {"error": "target K^2 outside the computed gap range"}
        a0 = cm.kappa2(lam0, gap_nm)
        h = st.dlam_nm * 1e-3
        ap, am = cm.kappa2(lam0 + h, gap_nm), cm.kappa2(lam0 - h, gap_nm)
        dlam = (math.log(ap["kappa2"]) - math.log(am["kappa2"])) / (2 * st.dlam_nm)
        ap2, am2 = cm.kappa2(lam0 + h / 2, gap_nm), cm.kappa2(lam0 - h / 2, gap_nm)
        dlam2 = (math.log(ap2["kappa2"]) - math.log(am2["kappa2"])) / st.dlam_nm
        gp_, gm_ = cm.kappa2(lam0, gap_nm + 5.0), cm.kappa2(lam0, gap_nm - 5.0)
        dgap = (math.log(gp_["kappa2"]) - math.log(gm_["kappa2"])) / 10.0
        blk = {"gap_nm": gap_nm, "kappa2_per_pass": a0["kappa2"], "dln_kappa2_dlambda_per_nm": dlam,
               "dln_kappa2_dlambda_per_nm_half_step": dlam2, "dln_kappa2_dgap_per_nm": dgap,
               "coupler_phase_rad": float(-np.angle(a0["t_ring"])),
               "coupler_phase_definition": "ring-side self-coupling phase DELAY relative to the uncoupled bent guide "
                                           "over the same arc: -arg(t_ring) with fields exp(j(wt - beta z)); it adds "
                                           "to phi = 2 pi n_eff L / lambda (docs/handoff/plant_optics.md, psi)",
               "t_bus_phase_rad": float(np.angle(a0["t_bus"])), "kappa_phase_rad": float(np.angle(a0["kappa_bus_to_ring"])),
               "t2_per_pass": a0["t2"], "power_out_over_in": a0["power_out_over_in"],
               "first_order_kappa2": float(abs(a0["first_order_kappa"]) ** 2),
               "q_coupling": float(2 * math.pi * C.C0 / (lam0 * 1e-6) / fsr / (-math.log1p(-a0["kappa2"])))}
        C.log(f"  {label}: {blk}")
        return blk
    thr = coupler_block(g_thr, "through")
    drp = coupler_block(g_drop, "drop")
    res = {"schema": SCHEMA, "xsec": xs.as_dict(), "r_ref_um": R, "lambda0_nm": lam0 * 1e3, "t_ref_c": C.T_REF_C,
           "fsr_design_thz": fsr / 1e12,
           "geometry": {"bus": "straight, same cross-section as the ring (assumed)", "slab_in_gap": xs.is_rib,
                        "gap_definition": "edge-to-edge at the closest point (rib widths at mid-height)"},
           "through": thr, "drop": drp, "design": des,
           "kappa2_table": {"lambda_nm": [l * 1e3 for l in lam_tab], "gap_nm": list(map(float, table_gaps)),
                            "kappa2": table.tolist(), "method": "A (overlap CMT with exact bent fields)"},
           "kappa2_vs_gap_lambda0": {"gap_nm": gaps, "kappa2": k2,
                                     "t2": [r["t2"] for r in rowsA],
                                     "power_out_over_in": [r["power_out_over_in"] for r in rowsA],
                                     "coupler_phase_rad": [float(-np.angle(r["t_ring"])) for r in rowsA]}}
    # method B at lambda0 (and +-dlam for the dispersion)
    if st.method_b:
        kapB = {}
        gB = sorted(set([60, 100, 150, 200, 250, 300, 400, 500]))
        for lam in (lam0 - st.dlam_nm * 1e-3, lam0, lam0 + st.dlam_nm * 1e-3):
            kapB[lam] = supermode_kappa(cm.ex, xs, lam, cm.t_k, [g_ * 1e-3 for g_ in gB], st.mesh_scale)
        nb = lambda lam: float(np.real(cm.ex.mode(lam=lam).n_eff))
        nr = lambda lam: float(np.real(cm.ex.mode(lam=lam, R=R).n_eff))
        bt = {}
        for gp in (g_thr, g_drop):
            if gp is None:
                continue
            vals = {lam: cmt_supermode(kapB[lam], R, xs.width_nm * 1e-3, gp * 1e-3, nb(lam), nr(lam), lam)
                    for lam in kapB}
            lams = sorted(vals)
            bt[f"{gp:.1f}"] = {"kappa2": vals[lam0]["kappa2"],
                               "kappa2_no_phase_mismatch": vals[lam0]["kappa2_no_mismatch"],
                               "kappa2_outer_edge_phase_ref": cmt_supermode(kapB[lam0], R, xs.width_nm * 1e-3,
                                                                            gp * 1e-3, nb(lam0), nr(lam0), lam0,
                                                                            phase_ref="outer_edge")["kappa2"],
                               "dln_kappa2_dlambda_per_nm": (math.log(vals[lams[2]]["kappa2"])
                                                             - math.log(vals[lams[0]]["kappa2"])) / (2 * st.dlam_nm),
                               "method_A_kappa2": cm.kappa2(lam0, gp)["kappa2"]}
        res["method_B"] = {"supermode_kappa": {f"{l * 1e3:.1f}": v for l, v in kapB.items()}, "at_design_gaps": bt}
    # 2-D full-wave validation of the CMT methods (one effective-index model per film)
    if st.meep2d and not st.quick and xs.name in ("rib160", "strip400x200"):
        n1, n2 = eim_indices(xs, lam0, cm.t_k)
        gaps2 = list(MEEP2D_GAPS_UM[xs.name])
        pre = None
        if st.meep2d_json and Path(st.meep2d_json).exists():
            allrows = json.loads(Path(st.meep2d_json).read_text()).get(xs.name, [])
            ok = [r for r in allrows if "params" in r and abs(r["params"]["R"] / R - 1) < 2e-3
                  and abs(r["params"]["n1"] - n1) < 1e-9 and abs(r["params"]["n2"] - n2) < 1e-9
                  and abs(r["params"]["w"] - xs.width_nm * 1e-3) < 1e-12]
            if ok:
                pre = ok
                C.log(f"  using {len(ok)} precomputed Meep 2-D runs from {st.meep2d_json}")
        try:
            v2 = meep2d_validation(st, n1=n1, n2=n2, w=xs.width_nm * 1e-3, R=R, lam=lam0, gaps_um=gaps2,
                                   resolutions=st.meep_res, meep_rows=pre)
            if pre is not None:
                v2["meep_source"] = {"file": str(st.meep2d_json), "sha256": C.sha256_file(Path(st.meep2d_json)),
                                     "note": "same parameters (R within 0.2 %); produced by --meep2d-only"}
            # summary: Meep value = mean of the two finest resolutions, uncertainty = their half difference
            summ = []
            for gp in sorted(set(r["gap_nm"] for r in v2["A"])):
                mm = sorted([r for r in v2["meep"] if r.get("gap_nm") == gp and "kappa2" in r], key=lambda r: r["res"])
                a_ = next(r["kappa2"] for r in v2["A"] if r["gap_nm"] == gp)
                b_ = next(r["kappa2"] for r in v2["B"] if r["gap_nm"] == gp)
                if len(mm) >= 2:
                    mv = 0.5 * (mm[-1]["kappa2"] + mm[-2]["kappa2"])
                    me = 0.5 * abs(mm[-1]["kappa2"] - mm[-2]["kappa2"])
                    summ.append({"gap_nm": gp, "meep_kappa2": mv, "meep_uncertainty": me, "A_kappa2": a_,
                                 "B_kappa2": b_, "A_over_meep": a_ / mv, "B_over_meep": b_ / mv})
            v2["summary"] = summ
            res["validation_2d"] = v2
        except Exception as e:
            res["validation_2d"] = {"error": repr(e)[:400]}
    # Jacobian of ln K^2 (through, drop) w.r.t. geometry at fixed gaps and radius
    if st.jac and g_thr and g_drop:
        steps = (("width_nm", 10.0), ("film_nm", 5.0)) + ((("slab_nm", 10.0),) if xs.is_rib else ())
        d, oth, odr = [], [], []
        evals = {}
        for name, stp in steps:
            v = {}
            for sgn in (+1, -1):
                xs2 = xs.replace(**{name: getattr(xs, name) + sgn * stp})
                v[sgn] = (cm.kappa2(lam0, g_thr, xs2)["kappa2"], cm.kappa2(lam0, g_drop, xs2)["kappa2"])
            d.append(name)
            oth.append((math.log(v[1][0]) - math.log(v[-1][0])) / (2 * stp))
            odr.append((math.log(v[1][1]) - math.log(v[-1][1])) / (2 * stp))
            evals[name] = {"step_nm": stp, "plus": v[1], "minus": v[-1]}
            C.log(f"  jac {name}: through {oth[-1]:.4f} drop {odr[-1]:.4f} per nm")
        d += ["gap_through_nm", "gap_drop_nm"]
        oth += [thr["dln_kappa2_dgap_per_nm"], 0.0]
        odr += [0.0, drp["dln_kappa2_dgap_per_nm"]]
        res["jacobian"] = {"d": d, "of": {"ln_kappa2_through_per_nm": oth, "ln_kappa2_drop_per_nm": odr},
                           "evaluations": evals,
                           "note": "partial derivatives at fixed edge-to-edge gaps and fixed r_ref; a lithographic "
                                   "width bias delta also closes each gap by delta: total = d/dw - d/dgap"}
    side = {"schema": SCHEMA, "xsec": xs.name,
            "inputs": {"geometry": xs.as_dict(), "r_ref": rsrc,
                       "materials": __import__("mrm.materials", fromlist=["x"]).materials_provenance(),
                       "film_nm": xs.film_nm, "settings": dict(st.__dict__),
                       "targets": des["targets"]},
            "mesh": {"fem": C.MeshSpec(scale=st.mesh_scale).as_dict(), "overlap_grid_dx_um": st.dx_um,
                     "ring_u_grid_um": st.du_um, "dz_um": st.dz_um},
            "validation": {"sampling": sampling, "convergence_at_through_gap": conv,
                           "power_conservation_range": [min(r["power_out_over_in"] for r in rowsA),
                                                        max(r["power_out_over_in"] for r in rowsA)],
                           "method_B": res.get("method_B", {}).get("at_design_gaps"),
                           "validation_2d": res.get("validation_2d")},
            "assumptions": {"bus": "straight bus of the ring's cross-section; slab continuous through the gap (rib)",
                            "drop": "drop bus identical to the through bus, on the opposite side",
                            "smoke_comparison": "femwell smoke (05 C2, strip 400x200, g0 200 nm, point coupler "
                                                "without phase mismatch) gave d ln K^2/d lambda 1.3-1.5 %/nm"},
            "runtime_s": round(time.time() - t0, 1)}
    C.write_artefact(f"coupling_{xs.name}", res, side, out_dir=out_dir)
    C.log(f"wrote coupling_{xs.name} ({time.time() - t0:.0f} s)")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--xsec", nargs="*", default=["rib160", "rib200", "strip400x200"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-meep", action="store_true")
    ap.add_argument("--meep2d-only", default=None, metavar="JSON",
                    help="run only the (light) Meep 2-D validation runs and write them to JSON")
    ap.add_argument("--meep2d-json", default=None, help="precomputed Meep 2-D runs (from --meep2d-only)")
    ap.add_argument("--meep-res", nargs="*", type=int, default=None, help="Meep 2-D resolutions (px/um)")
    a = ap.parse_args(argv)
    st = CmtSettings.quick_settings() if a.quick else CmtSettings()
    if a.no_meep:
        st.meep2d = False
    st.meep2d_json = a.meep2d_json
    if a.meep_res:
        st.meep_res = tuple(a.meep_res)
    if a.meep2d_only:
        out = {}
        for name in a.xsec:
            if name not in ("rib160", "strip400x200"):
                continue
            xs = C.XSECS[name]
            R, _ = load_r_ref(xs, Path(a.out) if a.out else C.DATA_DIR)
            n1, n2 = eim_indices(xs, xs.lam0_um, C.T_REF_C + K0C)
            out[name] = meep2d_runs(n1=n1, n2=n2, w=xs.width_nm * 1e-3, R=R, lam=xs.lam0_um,
                                    gaps_um=MEEP2D_GAPS_UM[name], resolutions=st.meep_res)
            Path(a.meep2d_only).write_text(json.dumps(C._jsonable(out), indent=1))
        return Path(a.meep2d_only)
    out_dir = Path(a.out) if a.out else (Path(tempfile.mkdtemp(prefix="mrm_coupling_quick_")) if a.quick else C.DATA_DIR)
    C.log(f"output -> {out_dir}")
    for name in a.xsec:
        extract_coupling(C.XSECS[name], st, out_dir)
    return out_dir


if __name__ == "__main__":
    main()
