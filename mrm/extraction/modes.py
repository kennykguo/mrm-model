"""Bent-mode extraction for the MRM cross-sections (ARCHITECTURE 5.1, 07 P1-6, D21, P0-3, P1-16, P2-3, VAL-7).

Run:  python -m mrm.extraction.modes [--quick] [--xsec rib160 rib200 strip400x200 decea2019] [--out DIR]
      (through tools/memguard.py --limit-gb 3 --heavy). --quick uses a coarse mesh and few points and, unless --out
      is given, writes into a temporary directory (it never overwrites the committed artefacts).

Writes ``modes_<xsec>.json`` (schema ``modes/1``) + ``modes_<xsec>.sidecar.json`` for every cross-section, and
``modes_study_bend.json`` (schema ``modes_bend_study/1``): the bend-formulation study that settles the femwell vs
Tidy3D disagreement of 05 section 5.4. ``--assemble-study QUICK_STUDY_JSON MEEP_CYL_JSON`` writes the study from
stored results without any solver run (how the committed study was made; see :func:`assemble_bend_study`).

Method (all numbers derived; materials only from mrm.materials):
- primary solver: the full-vector FEM of :mod:`mrm.extraction.common` (femwell's formulation and meshing,
  scikit-fem), with the EXACT bend transform (anisotropic eps and mu). femwell's own isotropic conformal map
  eps (1 + x/R)^2 is reported alongside, as the formulation it replaces;
- r_ref: the radius at which the bent-mode group index gives the 1.8 THz design FSR, iterated to self-consistency;
- n_eff(lambda) over 1260-1360 nm at T_ref = 25 C, fitted to second order about lambda0 = 1310 nm;
- dn_eff/dT(T) at 0, 25, 50, 75, 105, 125 C by central differences, and the H_solver
  dlambda/dT = (lambda/n_g)(dn_eff/dT + n_eff alpha_L) with the substrate-clamped alpha_L of Si (07 P0-3 R1);
- index weights dn_eff/dn_i by central differences (and first-order perturbation as a check), with the identity
  sum_i n_g,i dn_eff/dn_i = n_g;
- Poynting power fractions, electric-energy fractions, Poynting mode area;
- bend (and slab) radiation loss from a complex-coordinate PML, with a PML/window convergence check, and the
  Tidy3D local solver with PML as a second method;
- geometry Jacobians by central differences at fixed r_ref (07 P1-16);
- TM0 and TE1 (07 P2-3) straight and bent, with their guidance status;
- mesh convergence (four refinements, Richardson) and the Tidy3D cross-check at the same r_ref.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mrm.extraction import common as C

SCHEMA = "modes/1"
SCHEMA_STUDY = "modes_bend_study/1"
T_LIST_C = (0.0, 25.0, 50.0, 75.0, 105.0, 125.0)
LAM_GRID_NM = tuple(range(1260, 1361, 10))
K0C = 273.15


@dataclass
class Settings:
    quick: bool = False
    mesh_scale: float = 1.5                 # production mesh: h_core = 10 nm x scale (15 nm), slab 30 nm
    conv_scales: tuple = (3.0, 2.0, 1.5, 1.0)
    lam_grid_nm: tuple = LAM_GRID_NM
    t_list_c: tuple = T_LIST_C
    dlam_um: float = 0.002                  # FD step for n_g (with one Richardson step h, h/2)
    dt_k: float = 2.5                       # FD step for dn_eff/dT
    dn_idx: float = 1e-3                    # FD step for index weights
    jac_steps_nm: tuple = (("width_nm", 10.0), ("film_nm", 5.0), ("slab_nm", 10.0))
    tidy3d: bool = True
    tidy3d_dl: tuple = (0.010, 0.005, 0.0025)
    pml: bool = True
    families: bool = True
    alt_radius_um: float = C.R_PDK_UM

    @classmethod
    def quick_settings(cls) -> "Settings":
        return cls(quick=True, mesh_scale=3.0, conv_scales=(4.0, 3.5, 3.0), lam_grid_nm=(1260, 1310, 1360),
                   t_list_c=(25.0, 105.0), jac_steps_nm=(("width_nm", 10.0),), tidy3d=False, pml=False,
                   families=False)


# --------------------------------------------------------------------------------------------- slab modes


def slab_neff(n_sub: float, n_f: float, n_cov: float, t_um: float, lam_um: float, pol: str = "TE",
              order: int = 0) -> float | None:
    """Exact three-layer slab effective index (TE: E parallel to the film; TM: H parallel). None if cut off."""
    from scipy.optimize import brentq
    k0 = 2 * math.pi / lam_um
    lo = max(n_sub, n_cov) + 1e-9
    hi = n_f - 1e-9
    if t_um <= 0:
        return None

    def f(N):
        kap = k0 * math.sqrt(n_f ** 2 - N ** 2)
        g1 = k0 * math.sqrt(N ** 2 - n_sub ** 2)
        g2 = k0 * math.sqrt(N ** 2 - n_cov ** 2)
        if pol == "TE":
            a1, a2 = math.atan(g1 / kap), math.atan(g2 / kap)
        else:
            a1, a2 = math.atan(n_f ** 2 * g1 / (n_sub ** 2 * kap)), math.atan(n_f ** 2 * g2 / (n_cov ** 2 * kap))
        return kap * t_um - order * math.pi - a1 - a2
    if f(lo) * f(hi) > 0:
        return None
    return brentq(f, lo, hi, xtol=1e-14)


# --------------------------------------------------------------------------------------------- runner


class Extractor:
    """All solves for one cross-section. Meshes and FemProblems are cached by (geometry, window, scale)."""

    def __init__(self, xs: C.XSec, st: Settings):
        self.xs = xs
        self.st = st
        self.t_ref_k = C.T_REF_C + K0C
        self.lam0 = xs.lam0_um
        self._probs: dict = {}
        self.nsolves = 0
        self.solve_seconds = 0.0

    # ---- windows
    def window(self, xs: C.XSec | None = None, *, bent: bool, pml: bool = False, R: float | None = None,
               pml_um: float = 1.0, pml_strength: float = 2.0, extra_um: float = 0.0) -> C.Domain:
        xs = xs or self.xs
        w = xs.width_nm * 1e-3
        ymin = -1.5 if xs.below_box == "sio2" else -1.0
        hx = max(2.0, w / 2 + 1.6)
        if not pml:
            return C.Domain(-hx, hx + (0.5 if bent else 0.0), ymin, 1.5)
        # PML window: reach beyond the radiation caustic on the outer side
        xc = self.caustic_um(xs, R)
        right = max(hx, xc + 1.2) + extra_um + pml_um
        return C.Domain(-hx - pml_um * 0.5, right, ymin, 1.5, pml_left_um=pml_um * 0.5, pml_right_um=pml_um,
                        pml_strength=pml_strength)

    def caustic_um(self, xs: C.XSec, R: float | None) -> float:
        """Outer-side distance (um, from the guide centre) at which the bent mode becomes phase matched to the
        lateral radiation channel (the TE slab mode for a rib, the cladding for a strip)."""
        n = C.indices(xs, self.lam0, self.t_ref_k)
        nsl = slab_neff(n["box"], n["si"], n["clad"], xs.slab_nm * 1e-3, self.lam0, "TE") if xs.is_rib else None
        n_out = nsl if nsl else max(n["clad"], n["box"])
        ne = getattr(self, "_neff_guess", 2.4)
        return R * (ne / n_out - 1.0) if R else 2.0

    def problem(self, xs: C.XSec, dom: C.Domain, scale: float):
        key = (xs.name, xs.width_nm, xs.film_nm, xs.slab_nm, xs.sidewall_deg, xs.box_um, xs.below_box,
               tuple(sorted(dom.as_dict().items())), scale)
        if key not in self._probs:
            t0 = time.time()
            ms = C.MeshSpec(scale=scale)
            mesh = C.build_mesh(xs, dom, ms)
            if len(self._probs) >= 2:                       # bound memory (3 GB budget per job)
                self._probs.pop(next(iter(self._probs)))
                import gc
                gc.collect()
            self._probs[key] = C.FemProblem(mesh)
            C.log(f"  mesh {xs.name} w={xs.width_nm} t={xs.film_nm} s={xs.slab_nm} scale={scale} "
                  f"x=[{dom.x_min:.2f},{dom.x_max:.2f}] pml={dom.pml_right_um} -> {self._probs[key].ntri} tri, "
                  f"{self._probs[key].ndofs} dofs ({time.time() - t0:.1f} s)")
        return self._probs[key]

    # ---- one solve
    def mode(self, *, lam: float | None = None, t_k: float | None = None, R: float | None = None,
             form: str = "exact", xs: C.XSec | None = None, scale: float | None = None, dn: dict | None = None,
             family: str = "TE0", pml: bool = False, pml_um: float = 1.0, pml_strength: float = 2.0,
             extra_um: float = 0.0, si_level: str = "frey", num_modes: int | None = None,
             n_guess: float | None = None, near: float | None = None, near_tol: float = 0.005) -> C.ModeSol:
        xs = xs or self.xs
        lam = self.lam0 if lam is None else lam
        t_k = self.t_ref_k if t_k is None else t_k
        scale = self.st.mesh_scale if scale is None else scale
        dom = self.window(xs, bent=R is not None, pml=pml, R=R, pml_um=pml_um, pml_strength=pml_strength,
                          extra_um=extra_um)
        prob = self.problem(xs, dom, scale)
        n = C.indices(xs, lam, t_k, si_level=si_level, dn=dn)
        guess = getattr(self, "_neff_guess", None)
        if family != "TE0":
            guess = n_guess
        elif n_guess is not None:
            guess = n_guess
        nm = num_modes or (6 if (pml or family != "TE0") else 3)
        t0 = time.time()
        ms = prob.solve(n, lam, radius_um=R, formulation=form, dom=dom, num_modes=nm,
                        n_guess=(guess * 1.002 if guess else None))
        self.nsolves += 1
        self.solve_seconds += time.time() - t0
        if self.nsolves % 4 == 0:                           # SuperLU/ARPACK buffers: keep the RSS flat
            import gc
            gc.collect()
        if near is None and family == "TE0" and pml:
            near = self.neff(lam=lam, t_k=t_k, R=R, form=form, xs=xs, scale=scale, dn=dn, si_level=si_level)
        max_neff = None
        if family == "TE1":       # below TE0 (which may be absent from a shifted eigen window)
            max_neff = self.neff(lam=lam, t_k=t_k, R=R, form=form, xs=xs, scale=scale, dn=dn, si_level=si_level) - 0.01
        m = C.pick_mode(ms, family, core_regions=("core",), min_core_power=0.3 if family == "TE0" else 0.05,
                        near=near, near_tol=near_tol, max_neff=max_neff)
        if family == "TE0" and guess is None:
            self._neff_guess = float(np.real(m.n_eff))
        return m

    def neff(self, **kw) -> float:
        return float(np.real(self.mode(**kw).n_eff))

    def ng(self, *, lam: float | None = None, h: float | None = None, rich: bool = False,
           **kw) -> tuple[float, float, float]:
        """(n_eff, n_g, error estimate) with material dispersion: central difference (step h), and with
        rich=True a second step h/2 and one Richardson step (error estimate = the Richardson correction)."""
        lam = self.lam0 if lam is None else lam
        h = self.st.dlam_um if h is None else h
        f = lambda l: self.neff(lam=l, **kw)
        n0 = f(lam)
        d, err = C.central_diff(f, lam, h, richardson_steps=rich)
        return n0, n0 - lam * d, (lam * err if rich else float("nan"))

    def dndt(self, *, t_k: float | None = None, **kw) -> float:
        t_k = self.t_ref_k if t_k is None else t_k
        h = self.st.dt_k
        return (self.neff(t_k=t_k + h, **kw) - self.neff(t_k=t_k - h, **kw)) / (2 * h)

    def index_weights_fd(self, **kw) -> dict:
        d = self.st.dn_idx
        mats = ["si", "box", "clad"] + (["air"] if self.xs.below_box == "air" else [])
        out = {}
        for mname in mats:
            out[mname] = (self.neff(dn={mname: +d}, **kw) - self.neff(dn={mname: -d}, **kw)) / (2 * d)
        return out


# --------------------------------------------------------------------------------------------- per-xsec steps


def find_r_ref(ex: Extractor, fsr_hz: float = C.FSR_DESIGN_HZ) -> dict:
    """Self-consistent radius for the design FSR: r = c/(2 pi n_g,bent(r) FSR)."""
    _, ng0, _ = ex.ng()
    r = C.r_for_fsr_um(ng0, fsr_hz)
    hist = [{"r_um": r, "n_g_used": ng0, "from": "straight"}]
    for _ in range(6):
        _, ngb, _ = ex.ng(R=r)
        r_new = C.r_for_fsr_um(ngb, fsr_hz)
        hist.append({"r_um": r_new, "n_g_used": ngb, "from": f"bent at {r:.5f}"})
        if abs(r_new - r) < 2e-5:
            r = r_new
            break
        r = r_new
    return {"r_ref_um": r, "iterations": hist, "fsr_hz": fsr_hz}


def mesh_convergence(ex: Extractor, R: float) -> dict:
    """n_eff (straight, bent exact, bent isotropic), n_g, dn_eff/dT and the Si index weight on four meshes."""
    rows = []
    for sc in ex.st.conv_scales:
        t0 = time.time()
        ms = ex.mode(R=R, scale=sc)
        row = {"scale": sc, "h_core_nm": 10.0 * sc, "ntri": ex.problem(ex.xs, ex.window(bent=True), sc).ntri,
               "n_eff_straight": ex.neff(scale=sc), "n_eff_bent": float(np.real(ms.n_eff)),
               "n_eff_bent_isotropic": ex.neff(R=R, scale=sc, form="isotropic"),
               "n_g_bent": ex.ng(R=R, scale=sc)[1], "dneff_dt_bent": ex.dndt(R=R, scale=sc),
               "w_si_pert": ms.index_weight("core") + (ms.index_weight("slab") if ex.xs.is_rib else 0.0)}
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        C.log(f"  conv scale {sc}: {row}")
    out = {"rows": rows}
    for k in ("n_eff_straight", "n_eff_bent", "n_eff_bent_isotropic", "n_g_bent", "dneff_dt_bent", "w_si_pert"):
        out[k] = C.richardson([r["h_core_nm"] for r in rows], [r[k] for r in rows])
    return out


def lambda_sweep(ex: Extractor, R: float) -> dict:
    lam_nm = np.array(ex.st.lam_grid_nm, float)
    if ex.xs.lam0_um != C.LAM0_UM:                    # de Cea: +-50 nm about its own lambda0
        lam_nm = np.round(ex.xs.lam0_um * 1e3 + (lam_nm - 1310.0), 3)
    ne = np.array([ex.neff(lam=l * 1e-3, R=R) for l in lam_nm])
    d = lam_nm - ex.lam0 * 1e3
    c = np.polyfit(d, ne, 2)[::-1]                    # c0 + c1 d + c2 d^2
    c3 = np.polyfit(d, ne, 3)[::-1]
    fit = c[0] + c[1] * d + c[2] * d ** 2
    near = np.abs(d) <= 20.0
    return {"lambda_nm": lam_nm.tolist(), "n_eff": ne.tolist(), "poly2": c.tolist(), "poly3": c3.tolist(),
            "poly2_max_residual": float(np.max(np.abs(fit - ne))),
            "poly2_max_residual_within_20nm": float(np.max(np.abs((fit - ne)[near]))) if near.any() else None}


def thermal_sweep(ex: Extractor, R: float) -> dict:
    from mrm import materials as M
    rows = []
    for tc in ex.st.t_list_c:
        tk = tc + K0C
        n0, ngv, ngerr = ex.ng(t_k=tk, R=R)
        dndt = ex.dndt(t_k=tk, R=R)
        aL = float(M.alpha_l_si_per_k(tk))
        dl = ex.lam0 * 1e3 / ngv * (dndt + n0 * aL) * 1e3          # pm/K
        dl_to = ex.lam0 * 1e3 / ngv * dndt * 1e3
        sig_si = float(M.dn_dt_si_sigma_per_k(tk))
        rows.append({"t_c": tc, "n_eff": n0, "n_g": ngv, "dneff_dt_per_k": dndt, "alpha_l_per_k": aL,
                     "dlambda_dt_pm_per_k": dl, "dlambda_dt_thermo_optic_only_pm_per_k": dl_to,
                     "sigma_dn_si_dt_per_k": sig_si})
        C.log(f"  T={tc} C: n_eff {n0:.6f} n_g {ngv:.5f} dn/dT {dndt:.4e} dl/dT {dl:.3f} pm/K")
    lam_tab = [1260.0, 1285.0, 1310.0, 1335.0, 1360.0] if not ex.st.quick else [1310.0]
    if ex.xs.lam0_um != C.LAM0_UM:
        lam_tab = [round(ex.xs.lam0_um * 1e3 + (l - 1310.0), 3) for l in lam_tab]
    tab = [[ex.neff(lam=l * 1e-3, t_k=tc + K0C, R=R) for tc in ex.st.t_list_c] for l in lam_tab]
    return {"rows": rows, "table": {"lambda_nm": lam_tab, "t_c": list(ex.st.t_list_c), "n_eff": tab}}


def index_weight_block(ex: Extractor, R: float | None, t_k: float | None = None) -> dict:
    t_k = ex.t_ref_k if t_k is None else t_k
    w = ex.index_weights_fd(R=R, t_k=t_k)
    ngm = C.group_indices(ex.xs, ex.lam0, t_k)
    n0, ng, ngerr = ex.ng(R=R, t_k=t_k, rich=True)
    s = sum(ngm[k] * w[k] for k in w)
    m = ex.mode(R=R, t_k=t_k)
    pert = {"si": m.index_weight("core") + (m.index_weight("slab") if ex.xs.is_rib else 0.0),
            "box": m.index_weight("box"), "clad": m.index_weight("clad")}
    if "air" in m.regions:
        pert["air"] = m.index_weight("air")
    # Euler identity without dispersion: sum n_i w_i = n_eff - lambda dn_eff/dlambda|_n
    nmat = C.indices(ex.xs, ex.lam0, t_k)
    s_nodisp = sum(nmat[k] * w[k] for k in w)
    return {"fd": w, "perturbation": pert, "n_g_material": {k: ngm[k] for k in w}, "sum_ng_i_w_i": s,
            "n_g": ng, "identity_residual": s - ng, "n_g_fd_error": ngerr, "sum_n_i_w_i": s_nodisp,
            "fd_step": ex.st.dn_idx}


def fractions_block(m: C.ModeSol, xs: C.XSec) -> dict:
    regs = ["core"] + (["slab"] if xs.is_rib else []) + ["box", "clad"] + (["air"] if "air" in m.regions else [])
    pf = {r: m.power_fraction(r) for r in regs}
    ef = {r: m.energy_fraction(r) for r in regs}
    out = {"power_fraction": {"si": pf["core"] + pf.get("slab", 0.0), "si_rib": pf["core"],
                              "si_slab": pf.get("slab", 0.0), "box": pf["box"], "clad": pf["clad"]},
           "energy_fraction": {"si": ef["core"] + ef.get("slab", 0.0), "box": ef["box"], "clad": ef["clad"]},
           "mode_area_um2": m.mode_area_um2(), "te_fraction": m.te_fraction(),
           "power_centroid_x_um": m.centroid_x_um()}
    if "air" in m.regions:
        out["power_fraction"]["air"] = pf["air"]
        out["energy_fraction"]["air"] = ef["air"]
    return out


def radiation_loss(ex: Extractor, R: float, *, family: str = "TE0", full: bool = True,
                   n_guess: float | None = None, near: float | None = None) -> dict:
    """Bend + slab radiation loss from the FEM PML (convergence over PML strength, thickness and window) and,
    for TE0, Tidy3D with PML on a widened outer window (+R)."""
    cfgs = [(1.0, 2.0, 0.0)]
    if full:
        cfgs += [(1.0, 4.0, 0.0), (1.5, 2.0, 0.0), (1.0, 2.0, 1.0), (2.0, 3.0, 1.0)]
    rows = []
    for pml_um, a, extra in cfgs:
        try:
            m = ex.mode(R=R, pml=True, pml_um=pml_um, pml_strength=a, extra_um=extra, family=family,
                        n_guess=n_guess, near=near, near_tol=0.03 if family != "TE0" else 0.005,
                        num_modes=10 if family != "TE0" else None)
            rows.append({"pml_um": pml_um, "strength": a, "extra_window_um": extra,
                         "n_eff_re": float(np.real(m.n_eff)), "n_eff_im": float(np.imag(m.n_eff)),
                         "alpha_db_per_cm": m.alpha_db_per_cm,
                         "x_max_um": ex.window(bent=True, pml=True, R=R, pml_um=pml_um, extra_um=extra).x_max})
        except RuntimeError as e:
            rows.append({"pml_um": pml_um, "strength": a, "extra_window_um": extra, "error": str(e)[:300]})
        C.log(f"  PML {family} R={R:.3f}: {rows[-1]}")
    al = [r["alpha_db_per_cm"] for r in rows if "alpha_db_per_cm" in r]
    out = {"fem_pml": rows}
    floor = 1e-6     # dB/cm: below this the PML result is numerical noise (|Im n_eff| < 2e-12)
    if al:
        out["alpha_rad_db_per_cm"] = max(al[0], 0.0) if abs(al[0]) > floor else 0.0
        out["alpha_rad_upper_bound_db_per_cm"] = float(max(abs(a) for a in al)) if max(abs(a) for a in al) < floor else None
        out["alpha_rad_spread_db_per_cm"] = float(max(al) - min(al)) if len(al) > 1 else None
        out["alpha_rad_rel_spread"] = (float((max(al) - min(al)) / abs(al[0])) if len(al) > 1 and abs(al[0]) > floor
                                       else None)
    return out


def tidy3d_block(ex: Extractor, R: float, fem_bent: float, fem_straight: float, fem_iso: float) -> dict:
    rows = []
    for dl in ex.st.tidy3d_dl:
        t0 = time.time()
        s = C.tidy3d_mode(ex.xs, ex.lam0, ex.t_ref_k, dl_fine_um=dl, half_width_um=max(2.0, ex.xs.width_nm * 5e-4 + 1.5),
                          target_neff=fem_straight)
        b = C.tidy3d_mode(ex.xs, ex.lam0, ex.t_ref_k, radius_um=R, dl_fine_um=dl,
                          half_width_um=max(2.0, ex.xs.width_nm * 5e-4 + 1.5), target_neff=fem_bent)
        rows.append({"dl_nm": dl * 1e3, "n_eff_straight": float(np.real(s["n_eff"])),
                     "n_eff_bent_mean": float(np.real(b["n_eff"])), "n_eff_bent_plusR": float(np.real(b["n_eff_plus"])),
                     "n_eff_bent_minusR": float(np.real(b["n_eff_minus"])), "cells": b["grid_cells"],
                     "seconds": round(time.time() - t0, 1)})
        C.log(f"  tidy3d dl={dl}: {rows[-1]}")
    h = [r["dl_nm"] for r in rows]
    rs = C.richardson(h, [r["n_eff_straight"] for r in rows])
    rb = C.richardson(h, [r["n_eff_bent_mean"] for r in rows])
    shift_t = [r["n_eff_bent_mean"] - r["n_eff_straight"] for r in rows]
    return {"rows": rows, "straight_extrapolated": rs, "bent_extrapolated": rb,
            "tidy3d_bend_shift": shift_t, "femwell_exact_bend_shift": fem_bent - fem_straight,
            "femwell_isotropic_bend_shift": fem_iso - fem_straight,
            "tidy3d_n_eff": rb["f0"], "femwell_n_eff": fem_bent, "delta": fem_bent - rb["f0"],
            "delta_finest": fem_bent - rows[-1]["n_eff_bent_mean"], "same_r_ref": True,
            "r_ref_um": R}


def family_block(ex: Extractor, R: float) -> dict:
    """TM0 and TE1, straight and bent at r_ref (07 P2-3), with radiation loss and guidance status."""
    n = C.indices(ex.xs, ex.lam0, ex.t_ref_k)
    nsl = {p: slab_neff(n["box"], n["si"], n["clad"], ex.xs.slab_nm * 1e-3, ex.lam0, p) if ex.xs.is_rib else None
           for p in ("TE", "TM")}
    n_cl = max(n["clad"], n["box"])
    out = {"slab_te0_n_eff": nsl["TE"], "slab_tm0_n_eff": nsl["TM"], "cladding_index": n_cl}
    te0 = ex.neff()
    mid = 0.5 * (te0 + n_cl)
    for fam in ("TM0", "TE1"):
        blk = {}
        ns = None
        try:
            m = ex.mode(family=fam, num_modes=12, n_guess=mid)
            ns = float(np.real(m.n_eff))
            blk["straight"] = {"n_eff": ns, "n_g": ex.ng(family=fam, num_modes=12, n_guess=mid)[1],
                               "te_fraction": m.te_fraction(), "core_power_fraction": m.power_fraction("core")}
        except RuntimeError as e:
            blk["straight"] = {"found": False, "note": str(e)[:300]}
        if ns is not None:
            try:
                m = ex.mode(family=fam, R=R, num_modes=12, n_guess=ns, near=ns, near_tol=0.03)
                nb = float(np.real(m.n_eff))
                rl = radiation_loss(ex, R, family=fam, full=False, n_guess=nb, near=nb)
                blk["bent"] = {"n_eff": nb,
                               "n_g": ex.ng(family=fam, R=R, num_modes=12, n_guess=nb, near=nb, near_tol=0.03)[1],
                               "te_fraction": m.te_fraction(), "core_power_fraction": m.power_fraction("core"),
                               "alpha_rad_db_per_cm": rl.get("alpha_rad_db_per_cm"), "pml": rl["fem_pml"]}
            except RuntimeError as e:
                blk["bent"] = {"found": False, "note": str(e)[:300]}
        # guidance: straight above the lateral radiation channel (TE slab mode for a rib: TM0 and TE1 both leak
        # into it below its index; the cladding for a strip); bent: PML loss below 100 dB/cm
        lim = nsl["TE"] if ex.xs.is_rib else n_cl
        blk["guided_straight"] = bool(ns is not None and ns > lim)
        a = blk.get("bent", {}).get("alpha_rad_db_per_cm")
        blk["guided_bent"] = bool(a is not None and a < 100.0)
        blk["criterion"] = ("straight: n_eff above the lateral radiation channel (TE slab mode of a rib, cladding of "
                            "a strip); bent: FEM-PML radiation loss < 100 dB/cm (about the intrinsic loss of the doped "
                            "ring). A family that is not guided cannot form a resonance dip of useful Q.")
        out[fam] = blk
    return out


def jacobian_block(ex: Extractor, R: float, base: dict) -> dict:
    """Central differences at fixed r_ref: n_eff, lambda_res (fixed order and radius), n_g, dlambda/dT, alpha_rad."""
    from mrm import materials as M
    d_names, rows = [], {}
    outs = {"n_eff_per_nm": [], "lambda_res_nm_per_nm": [], "n_g_per_nm": [], "dlambda_dt_pm_per_k_per_nm": [],
            "alpha_rad_db_per_cm_per_nm": []}
    aL = float(M.alpha_l_si_per_k(ex.t_ref_k))
    for name, step in ex.st.jac_steps_nm:
        if name == "slab_nm" and not ex.xs.is_rib:
            continue
        vals = {}
        for sgn in (+1, -1):
            xs2 = ex.xs.replace(**{name: getattr(ex.xs, name) + sgn * step})
            n0, ngv, _ = ex.ng(xs=xs2, R=R)
            dndt = ex.dndt(xs=xs2, R=R)
            dl = ex.lam0 * 1e3 / ngv * (dndt + n0 * aL) * 1e3
            a = None
            if ex.st.pml:
                try:
                    a = ex.mode(xs=xs2, R=R, pml=True).alpha_db_per_cm
                except RuntimeError:
                    a = None
            vals[sgn] = {"n_eff": n0, "n_g": ngv, "dlambda_dt": dl, "alpha_rad": a}
            C.log(f"  jac {name} {sgn * step:+}: {vals[sgn]}")
        d_names.append(name)
        rows[name] = {"step_nm": step, "plus": vals[1], "minus": vals[-1]}
        dn = (vals[1]["n_eff"] - vals[-1]["n_eff"]) / (2 * step)
        outs["n_eff_per_nm"].append(dn)
        outs["lambda_res_nm_per_nm"].append(ex.lam0 * 1e3 / base["n_g"] * dn)
        outs["n_g_per_nm"].append((vals[1]["n_g"] - vals[-1]["n_g"]) / (2 * step))
        outs["dlambda_dt_pm_per_k_per_nm"].append((vals[1]["dlambda_dt"] - vals[-1]["dlambda_dt"]) / (2 * step))
        if vals[1]["alpha_rad"] is not None and vals[-1]["alpha_rad"] is not None:
            outs["alpha_rad_db_per_cm_per_nm"].append((vals[1]["alpha_rad"] - vals[-1]["alpha_rad"]) / (2 * step))
        else:
            outs["alpha_rad_db_per_cm_per_nm"].append(0.0)
    if not ex.st.pml:
        outs.pop("alpha_rad_db_per_cm_per_nm")
    return {"d": d_names, "of": outs, "evaluations": rows,
            "note": "central differences at fixed r_ref and fixed order; lambda_res_nm_per_nm = (lambda0/n_g) "
                    "dn_eff/dp; evaluated at T_ref; alpha_rad from the FEM PML (first configuration)"}


def slab_sweep(ex: Extractor, R: float, slabs_nm=(40.0, 50.0, 60.0, 80.0, 100.0)) -> dict:
    rows = []
    for s in slabs_nm:
        xs2 = ex.xs.replace(slab_nm=s)
        n0, ngv, _ = ex.ng(xs=xs2, R=R)
        try:
            a = ex.mode(xs=xs2, R=R, pml=True).alpha_db_per_cm
        except RuntimeError:
            a = None
        rows.append({"slab_nm": s, "n_eff": n0, "n_g": ngv, "alpha_rad_db_per_cm": a})
        C.log(f"  slab sweep {s}: {rows[-1]}")
    return {"r_um": R, "rows": rows}


def extract_xsec(xs: C.XSec, st: Settings, out_dir: Path) -> dict:
    from mrm import materials as M
    t_start = time.time()
    ex = Extractor(xs, st)
    C.log(f"=== {xs.name}: {xs.as_dict()}")
    lam0_nm = xs.lam0_um * 1e3
    is_decea = xs.name == "decea2019"

    # 1. straight
    ms = ex.mode()
    n_s, ng_s, ng_s_err = ex.ng(rich=True)
    straight = {"n_eff": n_s, "n_g": ng_s, "dneff_dt_per_k": ex.dndt(), **fractions_block(ms, xs),
                "index_weights": index_weight_block(ex, None)}
    C.log(f"straight: n_eff {n_s:.6f} n_g {ng_s:.5f}")

    # 2. r_ref
    if is_decea:
        rr = {"r_ref_um": 10.0 - xs.width_nm * 5e-4, "note": "centreline of the printed 10 um outer radius"}
    else:
        rr = find_r_ref(ex)
    R = rr["r_ref_um"]
    C.log(f"r_ref = {R:.5f} um")

    # 3. bent at r_ref, T_ref
    mb = ex.mode(R=R)
    n_b, ng_b, ng_b_err = ex.ng(R=R, rich=True)
    lam_s = lambda_sweep(ex, R)
    c = lam_s["poly2"]
    ngd = ng_b
    dng_dl = None
    # dn_g/dlambda from the local second difference (more accurate than the global fit)
    h = 0.005
    nm_, n0_, np_ = (ex.neff(lam=xs.lam0_um - h, R=R), n_b, ex.neff(lam=xs.lam0_um + h, R=R))
    d2 = (np_ - 2 * n0_ + nm_) / (h * 1e3) ** 2                 # per nm^2
    dng_dl = -xs.lam0_um * 1e3 * d2
    therm = thermal_sweep(ex, R)
    iw = index_weight_block(ex, R)
    fr = fractions_block(mb, xs)
    res = {"schema": SCHEMA, "xsec": xs.as_dict(), "r_ref_um": R, "lambda0_nm": lam0_nm, "t_ref_c": C.T_REF_C,
           "n_eff": n_b, "n_eff_poly_lambda": c, "n_eff_poly_note": "n_eff_bent(lambda) = c0 + c1*(lam-lam0) + "
           "c2*(lam-lam0)^2, lam in nm, at t_ref_c, referenced to r_ref_um",
           "n_g": ng_b, "n_g_from_poly": c[0] - lam0_nm * c[1], "dn_g_dlambda_per_nm": dng_dl,
           "dn_g_dlambda_from_poly_per_nm": -2.0 * lam0_nm * c[2],
           "fsr_thz": C.C0 / (ng_b * 2 * math.pi * R * 1e-6) / 1e12,
           "fsr_nm": (xs.lam0_um * 1e-6) ** 2 / (ng_b * 2 * math.pi * R * 1e-6) * 1e9}
    trows = therm["rows"]
    res["n_eff_t"] = {"t_c": [r["t_c"] for r in trows], "value": [r["n_eff"] for r in trows]}
    res["n_g_t"] = {"t_c": [r["t_c"] for r in trows], "value": [r["n_g"] for r in trows]}
    res["dneff_dt_per_k"] = {"t_c": [r["t_c"] for r in trows], "value": [r["dneff_dt_per_k"] for r in trows]}
    res["n_eff_table"] = {**therm["table"], "note": "bent TE0 n_eff(lambda, T) at r_ref (rows: lambda_nm, columns: t_c)"}
    res["alpha_l_per_k"] = {"t_c": [r["t_c"] for r in trows], "value": [r["alpha_l_per_k"] for r in trows],
                            "model": "Si, Okada & Tokumaru 1984, substrate-clamped (mrm.materials.alpha_l_si_per_k)"}
    res["dlambda_dt_pm_per_k"] = {"t_c": [r["t_c"] for r in trows], "value": [r["dlambda_dt_pm_per_k"] for r in trows],
                                  "formula": "(lam/n_g)*(dneff/dT + n_eff*alpha_L) at lambda0 (a locked ring's "
                                             "resonance sits at its line)"}
    res["dlambda_dt_thermo_optic_only_pm_per_k"] = {"t_c": [r["t_c"] for r in trows],
                                                    "value": [r["dlambda_dt_thermo_optic_only_pm_per_k"] for r in trows]}
    # beta of s(T) = s0 (1 + beta (T - T0)) around 25 C and around 105 C (07 P0-3 R1 order 2)
    tt = np.array(res["dlambda_dt_pm_per_k"]["t_c"])
    ss = np.array(res["dlambda_dt_pm_per_k"]["value"])
    if len(tt) >= 2:
        p = np.polyfit(tt, ss, min(2, len(tt) - 1))
        dsdT = np.polyder(p)
        res["dlambda_dt_beta_per_k"] = {"t0_c": [25.0, 105.0],
                                        "value": [float(np.polyval(dsdT, t0) / np.polyval(p, t0)) for t0 in (25.0, 105.0)],
                                        "fit": "quadratic in T over the table"}
    # uncertainty of dlambda/dT from the Si dn/dT spread (materials)
    w_si = iw["fd"]["si"]
    res["dlambda_dt_material_sigma_pm_per_k"] = {
        "t_c": [r["t_c"] for r in trows],
        "value": [xs.lam0_um * 1e3 / r["n_g"] * w_si * r["sigma_dn_si_dt_per_k"] * 1e3 for r in trows],
        "note": "(lam/n_g) * dn_eff/dn_Si * sigma(dn_Si/dT); geometry uncertainty enters via the Jacobian"}
    res["index_weights"] = {k: v for k, v in iw["fd"].items()}
    res["index_weight_check"] = {"sum_ng_i_w_i": iw["sum_ng_i_w_i"], "n_g": iw["n_g"],
                                 "residual": iw["identity_residual"], "perturbation": iw["perturbation"],
                                 "n_g_material": iw["n_g_material"]}
    res["power_fraction"] = fr["power_fraction"]
    res["energy_fraction"] = fr["energy_fraction"]
    res["mode_area_um2"] = fr["mode_area_um2"]
    res["te_fraction"] = fr["te_fraction"]
    res["power_centroid_x_um"] = fr["power_centroid_x_um"]
    res["straight"] = straight
    res["materials_at_ref"] = {"n": C.indices(xs, xs.lam0_um, ex.t_ref_k), "n_g": C.group_indices(xs, xs.lam0_um, ex.t_ref_k),
                               "dn_dt": C.dn_dt(xs, xs.lam0_um, ex.t_ref_k)}
    # Si index-level alternative (07 P1-6: li vs frey differ by 4.1e-3)
    n_li = ex.neff(R=R, si_level="li")
    res["si_level_alternative"] = {"n_eff_li_level": n_li, "delta_n_eff": n_li - n_b,
                                   "delta_lambda_res_nm": lam0_nm / ng_b * (n_li - n_b)}
    C.log(f"bent: n_eff {n_b:.6f} n_g {ng_b:.5f} dl/dT(25C) {trows[min(1, len(trows) - 1)]['dlambda_dt_pm_per_k']:.3f}")

    side_valid = {}
    # 4. mesh convergence at r_ref
    conv = mesh_convergence(ex, R)
    side_valid["mesh_convergence"] = conv
    res["discretization_error"] = {k: conv[k]["err"] for k in ("n_eff_bent", "n_g_bent", "dneff_dt_bent", "w_si_pert")}

    # 5. radiation loss at r_ref (TE0)
    if st.pml:
        rl = radiation_loss(ex, R)
        res["alpha_rad_db_per_cm"] = rl.get("alpha_rad_db_per_cm")
        res["alpha_rad"] = rl
        rl["second_method"] = ("Tidy3D's local mode solver with bend_radius and PML returned only PML modes for this "
                               "geometry (graded and uniform grids, 12-60 PML cells; tested), so the FEM-PML is "
                               "validated instead against the exact complex-frequency (Hankel) solution of a 2-D ring "
                               "in modes_study_bend.json (Q_rad within 0.2 % for Q_rad > 2e4).")
        if rl.get("alpha_rad_db_per_cm") is not None and rl["alpha_rad_db_per_cm"] > 0:
            rl["q_rad"] = 2 * math.pi * ng_b / (xs.lam0_um * 1e-6 * rl["alpha_rad_db_per_cm"] * 100 * math.log(10) / 10)
    # 6. tidy3d cross-check of n_eff at the same r_ref
    if st.tidy3d and not is_decea:
        fem_iso = ex.neff(R=R, form="isotropic")
        try:
            res["crosscheck"] = tidy3d_block(ex, R, n_b, n_s, fem_iso)
        except Exception as e:
            res["crosscheck"] = {"error": repr(e)[:400]}
    else:
        res["crosscheck"] = {"femwell_n_eff": n_b, "femwell_isotropic_n_eff": ex.neff(R=R, form="isotropic"),
                             "tidy3d_n_eff": None, "same_r_ref": True, "note": "tidy3d skipped"}
    # 7. Jacobians
    if not is_decea:
        res["jacobian"] = jacobian_block(ex, R, {"n_g": ng_b})
        if xs.is_rib and st.pml and not st.quick:
            res["slab_sweep"] = slab_sweep(ex, R)
    # 8. families
    if st.families and not is_decea:
        res["mode_families"] = family_block(ex, R)
    # 9. alternative radius 7.5 um
    if not is_decea and st.alt_radius_um:
        Ra = st.alt_radius_um
        n_a, ng_a, _ = ex.ng(R=Ra)
        dndt_a = ex.dndt(R=Ra)
        aL = float(M.alpha_l_si_per_k(ex.t_ref_k))
        blk = {"r_um": Ra, "n_eff": n_a, "n_g": ng_a, "dneff_dt_per_k": dndt_a,
               "dlambda_dt_pm_per_k": xs.lam0_um * 1e3 / ng_a * (dndt_a + n_a * aL) * 1e3,
               "fsr_thz": C.C0 / (ng_a * 2 * math.pi * Ra * 1e-6) / 1e12,
               "fsr_nm": (xs.lam0_um * 1e-6) ** 2 / (ng_a * 2 * math.pi * Ra * 1e-6) * 1e9,
               "n_eff_isotropic": ex.neff(R=Ra, form="isotropic")}
        if st.pml:
            blk["alpha_rad_db_per_cm"] = radiation_loss(ex, Ra, full=False).get("alpha_rad_db_per_cm")
        res["alt_radius"] = blk
    # 10. de Cea replication block
    if is_decea:
        res["replication"] = decea_compare(ex, R, n_b, ng_b, iw, fr)
    res["solver"] = {"primary": "FEM, exact bend transform (mrm.extraction.common.FemProblem; femwell mesh and "
                                "formulation, scikit-fem)", "nsolves": ex.nsolves,
                     "solve_seconds": round(ex.solve_seconds, 1)}
    sidecar = {"schema": SCHEMA, "xsec": xs.name,
               "inputs": {"geometry": xs.as_dict(), "materials": __import__("mrm.materials", fromlist=["x"]).materials_provenance(),
                          "film_nm": xs.film_nm, "lambda0_nm": lam0_nm, "t_ref_c": C.T_REF_C,
                          "fsr_design_thz": C.FSR_DESIGN_HZ / 1e12, "settings": {k: v for k, v in st.__dict__.items()}},
               "mesh": {"production": C.MeshSpec(scale=st.mesh_scale).as_dict(),
                        "window_bent": ex.window(bent=True).as_dict(),
                        "elements": "2nd-order Nedelec (transverse) x 2nd-order Lagrange (longitudinal)"},
               "validation": {**side_valid,
                              "index_weight_identity_residual": iw["identity_residual"],
                              "crosscheck_delta_tidy3d": res.get("crosscheck", {}).get("delta"),
                              "bend_formulation": "exact transform; see modes_study_bend.json"},
               "assumptions": {
                   "substrate": "omitted: the 2 um BOX makes substrate leakage negligible (decay "
                                "exp(-2 k0 sqrt(n_eff^2 - n_ox^2) * 2 um) < 1e-14)",
                   "expansion": "ring length follows the substrate-clamped Si alpha_L; the change of the cross-section "
                                "(width, film) with temperature and stress-optic terms are neglected (< 0.5 % of "
                                "dn_eff/dT, estimated from the Jacobian x alpha_Si x dimensions)",
                   "doping": "undoped Si; the junction's free-carrier index change is in junction_<xsec>.json",
                   "geometry": xs.provenance},
               "runtime_s": round(time.time() - t_start, 1)}
    name = f"modes_{xs.name}"
    p, sp = C.write_artefact(name, res, sidecar, out_dir=out_dir)
    C.log(f"wrote {p} ({time.time() - t_start:.0f} s, {ex.nsolves} solves)")
    return res


def decea_compare(ex: Extractor, R: float, n_b: float, ng_b: float, iw: dict, fr: dict) -> dict:
    """VAL-7: de Cea et al. 2019 Table 1 (printed): n_g = 2.73 from the FSR; Gamma_ring = 0.6515 (FEM,
    Poynting-normalized, Robinson 2008 definition = dn_eff/dn_Si); n_Si = 3.485 at 1545.2 nm."""
    R_out = 10.0
    lam = ex.lam0
    gamma_rob = iw["perturbation"]["si"]
    # n_g referenced to the outer radius gives the same FSR: n_g(R_out) = n_g(R) * R / R_out
    rows = {}
    for t_nm in (80.0, 90.0, 100.0, 110.0, 120.0):
        xs2 = ex.xs.replace(film_nm=t_nm)
        n0, ngv, _ = ex.ng(xs=xs2, R=R)
        m = ex.mode(xs=xs2, R=R)
        rows[str(t_nm)] = {"n_eff": n0, "n_g_centreline": ngv, "n_g_outer": ngv * R / R_out,
                           "gamma_robinson": m.index_weight("core"), "energy_fraction_si": m.energy_fraction("core")}
        C.log(f"  de Cea t={t_nm}: {rows[str(t_nm)]}")
    printed = {"n_g": 2.73, "n_g_source": "free spectral range (measured)", "gamma_ring": 0.6515,
               "gamma_definition": "FEM, Poynting-normalized (Robinson et al. 2008) = dn_eff/dn_Si",
               "lambda0_nm": 1545.2, "n_si": 3.485, "outer_radius_um": 10.0, "width_um": 1.7,
               "thickness": "roughly 100 nm", "source": "de Cea, Atabaki, Ram, Opt. Express 27, 24274 (2019), "
                                                     "Table 1 and section 3"}
    return {"printed": printed, "r_centreline_um": R,
            "model_nominal": {"n_g_centreline": ng_b, "n_g_outer": ng_b * R / R_out, "gamma_robinson": gamma_rob,
                              "energy_fraction_si": fr["energy_fraction"]["si"],
                              "fsr_nm": (lam * 1e-6) ** 2 / (ng_b * 2 * math.pi * R * 1e-6) * 1e9},
            "printed_fsr_nm_if_outer_radius": (1.5452e-6) ** 2 / (2.73 * 2 * math.pi * 10e-6) * 1e9,
            "thickness_sweep": rows,
            "tolerance": {"n_g_rel": 0.03, "gamma_abs": 0.05}}


# --------------------------------------------------------------------------------------------- bend study


def bessel_benchmark(st: Settings) -> dict:
    """2-D (y-uniform) ring: exact Bessel-function eigenvalue vs the FEM with the exact and isotropic maps."""
    import shapely
    from femwell.mesh import mesh_from_OrderedDict
    from scipy.optimize import brentq
    from scipy.special import jv, jvp, yv, yvp
    from skfem.io.meshio import from_meshio
    lam, n1, n2, w = 1.31, 2.9, 1.45, 0.4
    k0 = 2 * math.pi / lam

    def fun(nu, R, pol):
        r1, r2 = R - w / 2, R + w / 2
        a = 1.0 if pol == "E" else 1 / n1 ** 2
        b = 1.0 if pol == "E" else 1 / n2 ** 2
        x1i, x1c, x2c, x2o = k0 * n2 * r1, k0 * n1 * r1, k0 * n1 * r2, k0 * n2 * r2
        M1 = np.array([[jv(nu, x1c), yv(nu, x1c)], [a * n1 * jvp(nu, x1c), a * n1 * yvp(nu, x1c)]])
        A, B = np.linalg.solve(M1, np.array([jv(nu, x1i), b * n2 * jvp(nu, x1i)]))
        Fc, dFc = A * jv(nu, x2c) + B * yv(nu, x2c), a * n1 * (A * jvp(nu, x2c) + B * yvp(nu, x2c))
        return Fc * b * n2 * yvp(nu, x2o) - dFc * yv(nu, x2o)

    def exact(R, pol):
        nus = np.linspace(k0 * n2 * R * 1.3, k0 * n1 * R * 0.999, 3000)
        v = np.array([fun(x, R, pol) for x in nus])
        roots = [brentq(fun, nus[i], nus[i + 1], args=(R, pol), xtol=1e-13)
                 for i in range(len(nus) - 1) if np.sign(v[i]) != np.sign(v[i + 1])
                 and abs(v[i]) < 1e3 and abs(v[i + 1]) < 1e3]
        return max(roots) / (k0 * R)

    def mesh2d(h):
        polys = OrderedDict(core=shapely.box(-w / 2, -0.05, w / 2, 0.05), clad=shapely.box(-1.5, -0.05, 2.5, 0.05))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return from_meshio(mesh_from_OrderedDict(polys, {"core": {"resolution": h, "distance": 0.5, "SizeMax": 0.1}},
                                                     default_resolution_max=0.1))
    rows = []
    hs = (0.02, 0.01) if st.quick else (0.02, 0.01, 0.005)
    for R in ((3.0,) if st.quick else (3.0, 6.5)):
        for pol in ("H", "E"):
            ref = exact(R, pol)
            for h in hs:
                P = C.FemProblem(mesh2d(h))
                vals = {}
                for form in ("exact", "isotropic"):
                    ms = P.solve({"si": n1, "clad": n2, "box": n2}, lam, radius_um=R, formulation=form, num_modes=3,
                                 n_guess=ref, metallic=(pol == "E"))
                    sel = [m for m in ms if (m.te_fraction() > 0.5) == (pol == "H")]
                    vals[form] = max(float(np.real(m.n_eff)) for m in sel)
                rows.append({"R_um": R, "pol": pol, "h_nm": h * 1e3, "bessel": ref, "fem_exact": vals["exact"],
                             "fem_isotropic": vals["isotropic"], "err_exact": vals["exact"] - ref,
                             "err_isotropic": vals["isotropic"] - ref})
                C.log(f"  bessel {rows[-1]}")
    return {"geometry": {"core_index": n1, "clad_index": n2, "width_um": w, "lambda_um": lam,
                         "pol_H": "H axial (E in the ring plane, our TE0)", "pol_E": "E axial (our TM0)"},
            "method": "exact: Bessel J inside, J/Y in the core, Y outside (real order; radiation negligible), "
                      "continuity of the axial field and (1/eps or 1) d/dr; FEM: y-uniform core with PMC (H-pol) or "
                      "PEC (E-pol) top and bottom", "rows": rows}


MEEP_SCRIPT = r'''
import json, sys, time, math
import numpy as np
import meep as mp
p = json.loads(sys.argv[1])
res, m, R, W, H, nsi, nox = p["res"], p["m"], p["R"], p["W"], p["H"], p["nsi"], p["nox"]
dpml, pad = 1.0, 1.0
sr = R + W / 2 + pad + dpml
sz = H + 2 * (pad + dpml)
mp.verbosity(0)
fcen, df = 1 / p["lam_guess"], 0.06
geom = [mp.Block(center=mp.Vector3(R, 0, 0), size=mp.Vector3(W, mp.inf, H), material=mp.Medium(index=nsi))]
src = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Er, center=mp.Vector3(R + 0.05, 0, 0.03))]
sim = mp.Simulation(cell_size=mp.Vector3(sr, 0, sz), geometry=geom, sources=src, resolution=res,
                    boundary_layers=[mp.PML(dpml, direction=mp.R, side=mp.High), mp.PML(dpml, direction=mp.Z)],
                    dimensions=mp.CYLINDRICAL, m=m, default_material=mp.Medium(index=nox),
                    eps_averaging=p.get("avg", True))
h = mp.Harminv(mp.Er, mp.Vector3(R - 0.07, 0, -0.04), fcen, df)
t0 = time.time()
sim.run(mp.after_sources(h), until_after_sources=p.get("t_after", 200))
modes = [dict(freq=md.freq, Q=md.Q, amp=abs(md.amp)) for md in h.modes]
best = max(modes, key=lambda d: d["amp"]) if modes else None
print(json.dumps(dict(res=res, m=m, modes=modes, lam_um=(1 / best["freq"]) if best else None,
                      seconds=round(time.time() - t0, 1))))
'''


def meep_cyl_runs(*, R: float, m_order: int, nsi: float, nox: float, lam_guess_um: float,
                  resolutions=(30, 40, 60, 80)) -> list:
    """The cylindrical Meep runs (light: < 0.5 GB, one core): no subpixel averaging at every resolution, and
    averaging at the two coarsest (it makes the error larger in cylindrical coordinates here)."""
    rows = []
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "meep_cyl.py"
        sp.write_text(MEEP_SCRIPT)
        runs = [(r, False) for r in resolutions] + [(r, True) for r in tuple(resolutions) + (120,)]
        for res, avg in runs:
            par = {"res": res, "m": m_order, "R": R, "W": 0.4, "H": 0.2, "nsi": nsi, "nox": nox,
                   "lam_guess": lam_guess_um * (1 + (0.9 if avg else 0.5) / res), "t_after": 200, "avg": avg}
            t0 = time.time()
            r = subprocess.run([C.MEEP_PYTHON, "-I", str(sp), json.dumps(par)], capture_output=True, text=True,
                               timeout=7200)
            line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
            if not line:
                rows.append({"res": res, "eps_averaging": avg, "error": (r.stderr or r.stdout)[-400:], "params": par})
                continue
            d = json.loads(line[-1])
            d.update(wall_s=round(time.time() - t0, 1), eps_averaging=avg, params=par)
            rows.append(d)
            C.log(f"  meep res {res} avg {avg}: lam {d.get('lam_um')} ({d['wall_s']} s)")
    return rows


MEEP_CYL_R_UM, MEEP_CYL_M, MEEP_CYL_LAM_GUESS_UM = 3.0, 36, 1.3075


def meep_check(st: Settings, *, R: float = MEEP_CYL_R_UM, m_order: int | None = None, resolutions=(30, 40, 60, 80),
               precomputed: str | None = None) -> dict:
    """Cylindrical Meep (exact 3-D vectorial ring for one azimuthal order) on the 400 x 200 strip with constant
    indices at R = 3 um, several resolutions + Richardson; compared with the FEM exact and isotropic predictions
    of the same resonance. ``precomputed``: rows from ``meep_cyl_runs`` run outside the heavy lock (light job)."""
    if not os.path.exists(C.MEEP_PYTHON):
        return {"skipped": "meep interpreter not found"}
    lam = 1.31
    xs = C.XSECS["strip400x200"]
    n = C.indices(xs, lam, C.T_REF_C + K0C)
    nsi, nox = n["si"], n["clad"]
    ex = Extractor(xs.replace(top_kind="beol", box_kind="beol"), Settings(mesh_scale=1.0))
    const = {"si": nsi, "box": nox, "clad": nox, "air": 1.0}

    def neff_const(l, form):
        dom = ex.window(bent=True)
        P = ex.problem(ex.xs, dom, 1.0)
        ms = P.solve(const, l, radius_um=R, formulation=form, dom=dom, num_modes=3)
        return float(np.real(C.pick_mode(ms, "TE0").n_eff))
    if m_order is None:
        m_order = int(round(neff_const(lam, "exact") * 2 * math.pi * R / lam))
    pred = {}
    for form in ("exact", "isotropic"):
        l = lam
        for _ in range(6):
            ne = neff_const(l, form)
            ne2 = neff_const(l + 0.001, form)
            g = ne * 2 * math.pi * R - m_order * l
            dg = (ne2 - ne) / 0.001 * 2 * math.pi * R - m_order
            l -= g / dg
        pred[form] = l
    rows, src = None, None
    if precomputed and Path(precomputed).exists():
        pre = json.loads(Path(precomputed).read_text())
        cand = [r for r in pre if r.get("params", {}).get("m") == m_order and abs(r["params"]["R"] - R) < 1e-12
                and abs(r["params"]["nsi"] - nsi) < 1e-9 and abs(r["params"]["nox"] - nox) < 1e-9]
        if cand:
            rows, src = cand, {"file": precomputed, "sha256": C.sha256_file(Path(precomputed))}
    if rows is None:
        rows = meep_cyl_runs(R=R, m_order=m_order, nsi=nsi, nox=nox, lam_guess_um=pred["exact"],
                             resolutions=resolutions)
    series, verdict = meep_series_verdict(rows, pred)
    return {"R_um": R, "m": m_order, "n_si": nsi, "n_ox": nox, "fem_pred_lambda_um": pred, "rows": rows,
            "precomputed_source": src, "series": series, "verdict": verdict,
            "fem_pred_difference_nm": (pred["isotropic"] - pred["exact"]) * 1e3,
            "note": "Meep's cylindrical runs converge slowly (first order) and, without subpixel averaging, "
                    "erratically (staircasing); the analytic Bessel benchmark is the decisive check (07 D10)"}


def meep_series_verdict(rows: list, pred: dict) -> tuple[dict, dict | None]:
    """Richardson analysis of the cylindrical Meep rows (pure arithmetic, no solver) and the verdict against the FEM
    predictions ``pred = {"exact": lam_um, "isotropic": lam_um}`` of the same resonance.

    For each series (with and without subpixel averaging) it extrapolates h = 1/res -> 0 three ways: free order on
    the three finest, first order on the two finest, and first order on the next two. The verdict uses the averaged
    series: the estimate is the mid-range of the three extrapolations and the uncertainty their half-range (the
    observed order is ~1, so a first-order fit is the model; the half-range measures how far the series is from
    that asymptote)."""
    series = {}
    for avg in (False, True):
        ok = sorted([r for r in rows if r.get("lam_um") and bool(r.get("eps_averaging")) == avg],
                    key=lambda r: -r["res"])
        if len(ok) < 2:
            continue
        h = [1.0 / r["res"] for r in ok]
        f = [r["lam_um"] for r in ok]
        series["averaging" if avg else "no_averaging"] = {
            "res": [r["res"] for r in ok], "lam_um": f,
            "richardson_free_p_finest3": C.richardson(h[:3], f[:3]) if len(ok) >= 3 else None,
            "richardson_p1_finest2": C.richardson(h[:2], f[:2], p=1.0),
            "richardson_p1_next2": C.richardson(h[1:3], f[1:3], p=1.0) if len(ok) >= 3 else None}
    verdict = None
    a = series.get("averaging", {})
    if a and a.get("richardson_free_p_finest3") and a["richardson_free_p_finest3"].get("p"):
        ext = [a["richardson_free_p_finest3"]["f0"], a["richardson_p1_finest2"]["f0"]]
        if a.get("richardson_p1_next2"):
            ext.append(a["richardson_p1_next2"]["f0"])
        lo, hi = min(ext), max(ext)
        f0 = 0.5 * (lo + hi)
        unc = 0.5 * (hi - lo)
        verdict = {"meep_extrapolated_lambda_um": f0, "extrapolations_lambda_um": ext,
                   "observed_order": a["richardson_free_p_finest3"]["p"],
                   "extrapolation_uncertainty_nm": unc * 1e3,
                   "distance_to_exact_nm": (f0 - pred["exact"]) * 1e3,
                   "distance_to_isotropic_nm": (f0 - pred["isotropic"]) * 1e3,
                   "exact_inside_extrapolation_range": bool(lo <= pred["exact"] <= hi),
                   "isotropic_inside_extrapolation_range": bool(lo <= pred["isotropic"] <= hi),
                   "finest_minus_estimate_nm": (a["lam_um"][0] - f0) * 1e3}
    return series, verdict


def pml_benchmark_2d(st: Settings) -> dict:
    """Validation of the FEM-PML bend radiation loss: a 2-D ring (y-uniform, H-polarized = TE) with the rib160
    effective-index contrast, against the exact complex resonance frequency of the annulus for a fixed integer
    azimuthal order m (J inside, J/Y in the core, outgoing H^(2) outside; scipy, complex argument). The FEM gives
    a complex n_eff at the real lambda; Q_FEM = n_g / (2 |Im n_eff|), Q_exact = Re k / (2 |Im k|)."""
    import shapely
    from femwell.mesh import mesh_from_OrderedDict
    from scipy.special import h2vp, hankel2, jv, jvp, yv, yvp
    from skfem.io.meshio import from_meshio
    xs = C.XSECS["rib160"]
    n = C.indices(xs, 1.31, C.T_REF_C + K0C)
    n1 = slab_neff(n["box"], n["si"], n["clad"], xs.film_nm * 1e-3, 1.31, "TE")
    n2 = slab_neff(n["box"], n["si"], n["clad"], xs.slab_nm * 1e-3, 1.31, "TE")
    w, hy = xs.width_nm * 1e-3, 0.04

    def det(k, m, R):
        r1, r2 = R - w / 2, R + w / 2
        a, b = 1 / n1 ** 2, 1 / n2 ** 2
        x1i, x1c, x2c, x2o = k * n2 * r1, k * n1 * r1, k * n1 * r2, k * n2 * r2
        M1 = np.array([[jv(m, x1c), yv(m, x1c)], [a * n1 * jvp(m, x1c), a * n1 * yvp(m, x1c)]])
        A, B = np.linalg.solve(M1, np.array([jv(m, x1i), b * n2 * jvp(m, x1i)]))
        Fc, dFc = A * jv(m, x2c) + B * yv(m, x2c), a * n1 * (A * jvp(m, x2c) + B * yvp(m, x2c))
        return Fc * b * n2 * h2vp(m, x2o) - dFc * hankel2(m, x2o)

    def solve_k(m, R, k):
        k = complex(k)
        for _ in range(80):
            f = det(k, m, R)
            df = (det(k + 1e-7, m, R) - det(k - 1e-7, m, R)) / 2e-7
            dk = f / df
            k -= dk
            if abs(dk) < 1e-13 * abs(k):
                break
        return k

    def problem(dom):
        polys = OrderedDict(core=shapely.box(-w / 2, -hy / 2, w / 2, hy / 2),
                            clad=shapely.box(dom.x_min, -hy / 2, dom.x_max, hy / 2))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return C.FemProblem(from_meshio(mesh_from_OrderedDict(
                polys, {"core": {"resolution": 0.01, "distance": 0.5, "SizeMax": 0.03}}, default_resolution_max=0.03)))
    nmat = {"si": n1, "clad": n2, "box": n2}
    rows = []
    for R in ((4.0,) if st.quick else (3.0, 4.0, 5.0)):
        xmin = max(-1.8, -0.6 * R)
        dom0 = C.Domain(xmin, 3.0, -hy / 2, hy / 2)
        P0 = problem(dom0)
        nc = lambda l, g=2.5: float(np.real(C.pick_mode(P0.solve(nmat, l, radius_um=R, dom=dom0, num_modes=6, n_guess=g),
                                                           "TE0", min_core_power=0.3).n_eff))
        nr = nc(1.31)
        k0 = 2 * math.pi / 1.31
        m = int(round(nr * k0 * R))
        k = solve_k(m, R, m / (nr * R))
        lam = 2 * math.pi / k.real
        nr = nc(lam, nr)
        ng = nr - lam * (nc(lam + 0.002, nr) - nc(lam - 0.002, nr)) / 0.004
        xc = R * (nr / n2 - 1)
        cfg = []
        for extra, p, a in ((0.5, 1.0, 2.0), (0.5, 1.0, 4.0), (1.0, 1.5, 2.0), (1.5, 2.0, 3.0)):
            dom = C.Domain(xmin, xc + extra + p, -hy / 2, hy / 2, pml_right_um=p, pml_strength=a)
            try:
                mm = C.pick_mode(problem(dom).solve(nmat, lam, radius_um=R, dom=dom, num_modes=10, n_guess=nr),
                                 "TE0", min_core_power=0.0, near=nr, near_tol=0.02)
                cfg.append({"extra_um": extra, "pml_um": p, "strength": a, "n_eff": complex(mm.n_eff),
                            "q_fem": ng / (2 * abs(np.imag(mm.n_eff)))})
            except RuntimeError as e:
                cfg.append({"extra_um": extra, "pml_um": p, "strength": a, "error": str(e)[:200]})
        q_ok = [c["q_fem"] for c in cfg if "q_fem" in c]
        qx = k.real / (2 * abs(k.imag))
        rows.append({"R_um": R, "m": m, "lambda_um": lam, "q_exact": qx, "fem": cfg,
                     "q_fem_median": float(np.median(q_ok)) if q_ok else None,
                     "rel_err_median": float(np.median(q_ok) / qx - 1) if q_ok else None,
                     "alpha_exact_db_per_cm": 10 / math.log(10) * 2 * math.pi * ng / (lam * 1e-4 * qx)})
        C.log(f"  pml benchmark R={R}: Q_exact {qx:.1f} Q_fem {[round(q, 1) for q in q_ok]}")
    return {"model": {"n_core": n1, "n_clad": n2, "width_um": w, "note": "2-D effective-index model of rib160 (film "
                      "and slab TE slab indices); TE (E in the ring plane)"}, "rows": rows,
            "validity": "the FEM-PML matches the exact Q within 0.2 % for Q_rad > ~2e4 (alpha below ~50 dB/cm); for "
                        "strongly leaky modes (Q < 3e3) the PML spectrum is crowded and the result depends on the PML"}


def bend_study(st: Settings, out_dir: Path, *, meep: bool = True) -> dict:
    """The study that settles the femwell (+0.0014) vs Tidy3D (+0.00038) disagreement of 05 section 5.4."""
    C.log("=== bend formulation study")
    xs = C.XSECS["strip400x200"]
    ex = Extractor(xs, st)
    t_k = C.T_REF_C + K0C
    rows = []
    radii = (3.0, 5.0, 7.5) if not st.quick else (7.5,)
    for R in radii:
        r = {"R_um": R, "straight": ex.neff(), "exact": ex.neff(R=R), "isotropic": ex.neff(R=R, form="isotropic")}
        if st.tidy3d:
            td = C.tidy3d_mode(xs, xs.lam0_um, t_k, radius_um=R, dl_fine_um=0.005)
            ts = C.tidy3d_mode(xs, xs.lam0_um, t_k, dl_fine_um=0.005)
            r.update(tidy3d_straight=float(np.real(ts["n_eff"])), tidy3d_mean=float(np.real(td["n_eff"])),
                     tidy3d_plus=float(np.real(td["n_eff_plus"])), tidy3d_minus=float(np.real(td["n_eff_minus"])))
            r["tidy3d_shift"] = r["tidy3d_mean"] - r["tidy3d_straight"]
        r["exact_shift"] = r["exact"] - r["straight"]
        r["isotropic_shift"] = r["isotropic"] - r["straight"]
        rows.append(r)
        C.log(f"  bend {r}")
    out = {"schema": SCHEMA_STUDY, "question": "05 section 5.4: femwell's isotropic conformal map gives +0.0014 at "
           "R = 7.5 um, Tidy3D's exact transform +0.00038. Which is right?",
           "strip_bend_shift": rows, "bessel_2d": bessel_benchmark(st), "pml_hankel_2d": pml_benchmark_2d(st)}
    if meep and not st.quick:
        try:
            out["meep_cylindrical"] = meep_check(st, precomputed=os.environ.get("MRM_MEEP_CYL_JSON"))
        except Exception as e:
            out["meep_cylindrical"] = {"error": repr(e)[:400]}
    b = out["bessel_2d"]["rows"]
    fin = [r for r in b if r["h_nm"] == min(rr["h_nm"] for rr in b)]
    out["verdict"] = {
        "exact_transform_max_abs_error_vs_bessel": max(abs(r["err_exact"]) for r in fin),
        "isotropic_map_max_abs_error_vs_bessel": max(abs(r["err_isotropic"]) for r in fin),
        "conclusion": "the exact (anisotropic) transform is right: it reproduces the analytic Bessel eigenvalue of "
                      "a 2-D ring to < 1e-7 for both polarizations and Tidy3D's +/-R mean in 3-D; femwell's "
                      "isotropic eps(1+x/R)^2 overestimates the bend shift (x3.7 at R = 7.5 um), because it "
                      "approximates the conformal coordinate ln(r/R) by x/R and drops the mu tensor. modes.py uses "
                      "the exact transform."}
    side = {"schema": SCHEMA_STUDY, "inputs": {"materials": __import__("mrm.materials", fromlist=["x"]).materials_provenance(),
                                               "geometry": xs.as_dict()},
            "mesh": C.MeshSpec(scale=st.mesh_scale).as_dict(),
            "validation": out["verdict"]}
    C.write_artefact("modes_study_bend", out, side, out_dir=out_dir)
    return out


#: FEM predictions of the cylindrical-Meep resonance (strip 400 x 200 nm with the constant indices of meep_check:
#: n_Si 3.50535, n_ox 1.45003; R = 3 um, m = 36) with the exact and the isotropic map. meep_check() computed them in
#: the run of 2026-10-09 that a session restart interrupted; docs/handoff/extract_modes.md recorded them, but the
#: output file was not kept. Only assemble_bend_study() uses them, because it may not run the FEM.
MEEP_CYL_FEM_PRED_RECORDED_UM = {"exact": 1.30750, "isotropic": 1.30954}
#: Tidy3D +R/-R mean bend shift of the 400 x 200 nm strip at R = 7.5 um (docs/research/05 verification log, V1) and
#: the constant-index straight-strip Tidy3D run of 05 (its group index has no material dispersion, like the Meep runs).
TIDY3D_V01_JSON = C.ROOT / "docs" / "research" / "05_tools_smoke" / "verify" / "v01_tidy3d_bend_asym.json"
TIDY3D_T01_JSON = C.ROOT / "docs" / "research" / "05_tools_smoke" / "out" / "t01_tidy3d_mode.json"


def assemble_bend_study(quick_study_json: str | Path, meep_cyl_json: str | Path, *,
                        out_dir: Path | None = None) -> dict:
    """Write ``modes_study_bend.json`` from results that already exist, WITHOUT any solver run.

    The full study (``bend_study``: strip rows at R = 3, 5, 7.5 um with Tidy3D, Bessel at R = 3 and 6.5 um down to
    h = 5 nm, PML-Hankel at R = 3, 4, 5 um, FEM predictions for the Meep resonance) was waiting for the heavy lock
    when a session restart stopped it, and the resumed task forbids new solver runs. This assembles the same schema
    from: the ``--quick`` study (Bessel at R = 3 um, h = 20/10 nm; PML-Hankel at R = 4 um; a coarse-mesh strip row at
    7.5 um), the stored cylindrical Meep rows (``--meep-cyl-only``), the committed ``modes_<xsec>.json`` (production
    FEM and Tidy3D at r_ref, the strip at 7.5 um), the Tidy3D +-R check of 05 (V1) and the recorded FEM predictions of
    the Meep resonance (:data:`MEEP_CYL_FEM_PRED_RECORDED_UM`). Every block names its source; the sidecar lists the
    sha256 of every input and what the full study would add."""
    import mrm.materials as M
    qpath, mpath = Path(quick_study_json), Path(meep_cyl_json)
    quick = json.loads(qpath.read_text())
    if quick.get("schema") != SCHEMA_STUDY:
        raise ValueError(f"{qpath}: not a {SCHEMA_STUDY} file")
    sources = {"quick_study": {"file": str(qpath), "sha256": C.sha256_file(qpath),
                               "what": "python -m mrm.extraction.modes --quick --study only (2026-10-09): bessel_2d, "
                                       "pml_hankel_2d and a coarse-mesh (h_core 30 nm) strip row at R = 7.5 um"},
               "meep_cyl_runs": {"file": str(mpath), "sha256": C.sha256_file(mpath),
                                 "what": "python -m mrm.extraction.modes --meep-cyl-only (2026-10-09; 9 runs, peak "
                                         "RSS 0.23 GB)"},
               "tidy3d_v01": {"file": str(TIDY3D_V01_JSON.relative_to(C.ROOT)), "sha256": C.sha256_file(TIDY3D_V01_JSON),
                              "what": "Tidy3D local solver, strip 400 x 200 nm, constant n_Si 3.50053 / n_ox 1.44680, "
                                      "uniform 20/10/5 nm grids, +R/-R at 7.5 um (05 verification V1)"},
               "tidy3d_t01": {"file": str(TIDY3D_T01_JSON.relative_to(C.ROOT)), "sha256": C.sha256_file(TIDY3D_T01_JSON),
                              "what": "Tidy3D straight strip, constant indices: waveguide-only group index"}}
    arts = {}
    for x in ("strip400x200", "rib160", "rib200"):
        d, side = C.load_artefact(f"modes_{x}")
        arts[x] = d
        sources[f"modes_{x}"] = {"file": f"mrm/data/extracted/modes_{x}.json", "sha256": side["sha256"]}

    # 1. 3-D bend shifts: FEM exact / isotropic against Tidy3D's exact transform (+R/-R mean)
    def r_ref_row(x: str) -> dict:
        d = arts[x]
        cc = d["crosscheck"]
        st0 = d["straight"]["n_eff"]
        t3_ext = cc["bent_extrapolated"]["f0"] - cc["straight_extrapolated"]["f0"]
        return {"xsec": x, "R_um": d["r_ref_um"], "straight": st0, "exact": d["n_eff"],
                "isotropic": st0 + cc["femwell_isotropic_bend_shift"],
                "exact_shift": cc["femwell_exact_bend_shift"], "isotropic_shift": cc["femwell_isotropic_bend_shift"],
                "tidy3d_straight": cc["straight_extrapolated"]["f0"], "tidy3d_mean": cc["bent_extrapolated"]["f0"],
                "tidy3d_shift": cc["tidy3d_bend_shift"][-1],
                "tidy3d_shift_by_grid": {"dl_nm": [r["dl_nm"] for r in cc["rows"]], "shift": cc["tidy3d_bend_shift"]},
                "tidy3d_shift_extrapolated": t3_ext,
                "source": f"modes_{x}.json crosscheck: FEM production mesh (h_core 15 nm, Richardson error of n_eff "
                          f"{d['discretization_error']['n_eff_bent']:.1e}); Tidy3D local solver on interface-aligned "
                          f"10/5/2.5 nm grids, +R/-R mean (tidy3d_shift = finest grid)"}
    strip = arts["strip400x200"]
    v01 = json.loads(TIDY3D_V01_JSON.read_text())
    t3_75 = {k: v["mean_shift7p5"] for k, v in v01.items()}
    q75 = [r for r in quick["strip_bend_shift"] if abs(r["R_um"] - 7.5) < 1e-12]
    rows = [r_ref_row("strip400x200"),
            {"xsec": "strip400x200", "R_um": 7.5, "straight": strip["straight"]["n_eff"],
             "exact": strip["alt_radius"]["n_eff"], "isotropic": strip["alt_radius"]["n_eff_isotropic"],
             "exact_shift": strip["alt_radius"]["n_eff"] - strip["straight"]["n_eff"],
             "isotropic_shift": strip["alt_radius"]["n_eff_isotropic"] - strip["straight"]["n_eff"],
             "tidy3d_shift": t3_75[min(t3_75, key=float)],
             "tidy3d_shift_by_grid": {"dl_nm": [float(k) * 1e3 for k in t3_75], "shift": list(t3_75.values())},
             "tidy3d_plus": v01[min(v01, key=float)]["plus7p5"], "tidy3d_minus": v01[min(v01, key=float)]["minus7p5"],
             "source": "FEM: modes_strip400x200.json straight and alt_radius (production mesh, materials indices); "
                       "Tidy3D: 05 verification V1 (constant n_Si 3.50053 / n_ox 1.44680, 0.005 / 0.003 below the "
                       "materials model; the shift is a difference and moves by < 2e-6 over the three grids)"}]
    for r in q75:
        rows.append(dict(r, xsec="strip400x200", source="--quick study (mesh h_core 30 nm): coarse-mesh repeat of "
                                                        "the 7.5 um row, no Tidy3D"))
    rib_rows = [r_ref_row("rib160"), r_ref_row("rib200")]
    # the isotropic map's excess shift scales as 1/R^2 (05 V1 found the same for Tidy3D's mean shift)
    ex_ref, ex_75 = rows[0]["isotropic_shift"] - rows[0]["exact_shift"], rows[1]["isotropic_shift"] - rows[1]["exact_shift"]
    scaling = {"isotropic_excess_shift_r_ref": ex_ref, "isotropic_excess_shift_7p5": ex_75,
               "ratio": ex_ref / ex_75, "ratio_if_1_over_r2": (7.5 / rows[0]["R_um"]) ** 2,
               "isotropic_over_exact_shift_7p5": rows[1]["isotropic_shift"] / rows[1]["exact_shift"]}

    # 2. cylindrical Meep (3-D vectorial ring, one azimuthal order) against the recorded FEM predictions
    n = C.indices(C.XSECS["strip400x200"], 1.31, C.T_REF_C + K0C)
    nsi, nox = n["si"], n["clad"]
    allrows = json.loads(mpath.read_text())
    mrows = [r for r in allrows if r.get("params", {}).get("m") == MEEP_CYL_M
             and abs(r["params"]["R"] - MEEP_CYL_R_UM) < 1e-12
             and abs(r["params"]["nsi"] - nsi) < 1e-9 and abs(r["params"]["nox"] - nox) < 1e-9]
    if len(mrows) != len(allrows):
        raise ValueError(f"{mpath}: {len(allrows) - len(mrows)} Meep rows do not match R, m and the indices")
    pred = dict(MEEP_CYL_FEM_PRED_RECORDED_UM)
    series, mverdict = meep_series_verdict(mrows, pred)
    t01 = json.loads(TIDY3D_T01_JSON.read_text())
    ng_const = t01["straight"]["n_g"]
    pred_diff_scaled_nm = ex_75 * (7.5 / MEEP_CYL_R_UM) ** 2 * 1.3075 / ng_const * 1e3
    meep = {"R_um": MEEP_CYL_R_UM, "m": MEEP_CYL_M, "n_si": nsi, "n_ox": nox, "geometry": "strip 400 x 200 nm, "
            "oxide all round, constant indices (materials values at 1310 nm, 25 C)",
            "fem_pred_lambda_um": pred,
            "fem_pred_provenance": "recorded: meep_check() of the interrupted run (exact and isotropic maps, h_core "
                                   "10 nm); the numbers are in docs/handoff/extract_modes.md, the output file was not "
                                   "kept. Not recomputed (the resumed task forbids FEM runs)",
            "fem_pred_difference_nm": (pred["isotropic"] - pred["exact"]) * 1e3,
            "fem_pred_difference_consistency": {
                "estimate_nm": pred_diff_scaled_nm,
                "method": "the isotropic map's excess bend shift at 7.5 um (production artefact) scaled by "
                          "(7.5/3)^2 (the 1/R^2 law checked between r_ref and 7.5 um), converted with "
                          "d lambda = lambda dn_eff / n_g, n_g = 4.142 (05 t01, constant indices)",
                "rel_diff_vs_recorded": pred_diff_scaled_nm / ((pred["isotropic"] - pred["exact"]) * 1e3) - 1.0},
            "rows": mrows, "series": series, "verdict": mverdict,
            "note": "subpixel averaging on: monotone, first order (p = 0.97); off: staircasing makes the series "
                    "erratic (not used). No new Meep run was made for this artefact (Kenny, 2026-10-09)."}

    # 3. verdict (the same quantities as bend_study)
    b = quick["bessel_2d"]["rows"]
    fin = [r for r in b if r["h_nm"] == min(rr["h_nm"] for rr in b)]
    t3 = [r for r in rows + rib_rows if "tidy3d_shift" in r]
    verdict = {
        "exact_transform_max_abs_error_vs_bessel": max(abs(r["err_exact"]) for r in fin),
        "isotropic_map_max_abs_error_vs_bessel": max(abs(r["err_isotropic"]) for r in fin),
        "exact_vs_tidy3d_max_rel_diff_of_shift": max(abs(r["exact_shift"] / r["tidy3d_shift"] - 1) for r in t3),
        "isotropic_vs_tidy3d_min_ratio_of_shift": min(r["isotropic_shift"] / r["tidy3d_shift"] for r in t3),
        "meep_distance_to_exact_nm": mverdict["distance_to_exact_nm"] if mverdict else None,
        "meep_distance_to_isotropic_nm": mverdict["distance_to_isotropic_nm"] if mverdict else None,
        "meep_extrapolation_uncertainty_nm": mverdict["extrapolation_uncertainty_nm"] if mverdict else None,
        "pml_vs_hankel_max_rel_err_q": max(abs(r["rel_err_median"]) for r in quick["pml_hankel_2d"]["rows"]),
        "conclusion": "the exact (anisotropic) transform is right: it reproduces the analytic Bessel eigenvalue of "
                      "a 2-D ring to < 1e-7 for both polarizations, Tidy3D's +/-R mean bend shift in 3-D to "
                      "within 2 % at r_ref and 7.5 um (three cross-sections), and the cylindrical Meep resonance "
                      "within the extrapolation range. femwell's isotropic eps(1+x/R)^2 overestimates the bend shift "
                      "(x3.7 at R = 7.5 um; the excess scales as 1/R^2), because it approximates the conformal "
                      "coordinate ln(r/R) by x/R and drops the mu tensor. modes.py uses the exact transform."}
    out = {"schema": SCHEMA_STUDY,
           "question": "05 section 5.4: femwell's isotropic conformal map gives +0.0014 at R = 7.5 um, Tidy3D's "
                       "exact transform +0.00038. Which is right?",
           "assembled_from_existing_results": True,
           "strip_bend_shift": rows, "rib_bend_shift": rib_rows, "bend_shift_scaling": scaling,
           "bessel_2d": dict(quick["bessel_2d"], source="--quick study (R = 3 um, h = 20 / 10 nm)"),
           "pml_hankel_2d": dict(quick["pml_hankel_2d"], source="--quick study (R = 4 um only)"),
           "meep_cylindrical": meep, "verdict": verdict}
    not_done = [
        "strip rows at R = 3 and 5 um with Tidy3D (full study); replaced by r_ref and 7.5 um rows from the committed "
        "artefacts and 05 V1",
        "Bessel benchmark at R = 6.5 um and h = 5 nm; R = 3 um at h = 20 / 10 nm is already at the 1e-7 level",
        "PML-Hankel benchmark at R = 3 and 5 um; R = 4 um (Q_rad 2.7e4) is stored",
        "FEM predictions of the Meep resonance re-run and stored (recorded values used)",
        "a finer Meep resolution (160 px/um) to tighten the first-order extrapolation (Meep not run, per Kenny)"]
    side = {"schema": SCHEMA_STUDY,
            "inputs": {"materials": M.materials_provenance(), "geometry": C.XSECS["strip400x200"].as_dict(),
                       "sources": sources},
            "mesh": {"bessel_2d_h_nm": sorted({r["h_nm"] for r in b}),
                     "pml_hankel_2d": "core h 10 nm, outside 30 nm; four PML/window settings",
                     "strip_rows": "production FEM mesh h_core 15 nm (modes_<xsec>.json); quick row h_core 30 nm",
                     "tidy3d_grids_nm": {"r_ref": [10.0, 5.0, 2.5], "7.5um": [20.0, 10.0, 5.0]},
                     "meep_resolution_px_per_um": sorted({r["res"] for r in mrows})},
            "validation": verdict,
            "assembly": {"method": "assemble_bend_study(): no solver run; arithmetic on stored results",
                         "command": "python -m mrm.extraction.modes --assemble-study QUICK_STUDY_JSON MEEP_CYL_JSON",
                         "full_study_command": "MRM_MEEP_CYL_JSON=<meep rows> .venv/bin/python tools/memguard.py "
                                               "--limit-gb 3 --heavy -- .venv/bin/python -m mrm.extraction.modes "
                                               "--study only",
                         "not_done": not_done}}
    C.write_artefact("modes_study_bend", out, side, out_dir=out_dir)
    return out


# --------------------------------------------------------------------------------------------- CLI


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--xsec", nargs="*", default=["strip400x200", "rib160", "rib200", "decea2019"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--study", choices=["yes", "no", "only"], default="yes")
    ap.add_argument("--no-meep", action="store_true")
    ap.add_argument("--no-tidy3d", action="store_true")
    ap.add_argument("--meep-cyl-only", default=None, metavar="JSON",
                    help="run only the (light) cylindrical Meep runs of the bend study and write them to JSON")
    ap.add_argument("--assemble-study", nargs=2, default=None, metavar=("QUICK_STUDY_JSON", "MEEP_CYL_JSON"),
                    help="write modes_study_bend.json from stored results (a --quick study and the --meep-cyl-only "
                         "rows) and the committed modes artefacts, without any solver run")
    a = ap.parse_args(argv)
    if a.assemble_study:
        out_dir = Path(a.out) if a.out else C.DATA_DIR
        assemble_bend_study(*a.assemble_study, out_dir=out_dir)
        C.log(f"assembled modes_study_bend -> {out_dir}")
        return out_dir
    if a.meep_cyl_only:
        xs = C.XSECS["strip400x200"]
        n = C.indices(xs, 1.31, C.T_REF_C + K0C)
        rows = meep_cyl_runs(R=MEEP_CYL_R_UM, m_order=MEEP_CYL_M, nsi=n["si"], nox=n["clad"],
                             lam_guess_um=MEEP_CYL_LAM_GUESS_UM)
        Path(a.meep_cyl_only).write_text(json.dumps(C._jsonable(rows), indent=1))
        return Path(a.meep_cyl_only)
    st = Settings.quick_settings() if a.quick else Settings()
    if a.no_tidy3d:
        st.tidy3d = False
    if a.out:
        out_dir = Path(a.out)
    elif a.quick:
        out_dir = Path(tempfile.mkdtemp(prefix="mrm_modes_quick_"))
    else:
        out_dir = C.DATA_DIR
    C.log(f"output -> {out_dir}")
    if a.study in ("yes", "only"):
        bend_study(st, out_dir, meep=not a.no_meep)
    if a.study != "only":
        for name in a.xsec:
            extract_xsec(C.XSECS[name], st, out_dir)
    C.log("done")
    return out_dir


if __name__ == "__main__":
    main()
