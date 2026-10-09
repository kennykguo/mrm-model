"""Shared geometry, materials, meshing, mode solvers and artefact helpers for the offline extraction scripts.

Never imported by runtime code (ARCHITECTURE section 2). GPL tools (femwell, gmsh) are imported lazily, inside the
functions that need them, so that ``import mrm.extraction.common`` stays cheap for the tests.

Contents
--------
- :class:`XSec` and :data:`XSECS`: the cross-section hypotheses (07 D21) with their assumed geometry and ranges.
- :func:`indices`: the per-region refractive indices at (lambda, T) from :mod:`mrm.materials` only.
- :func:`build_mesh`: a graded conforming triangle mesh (gmsh through femwell's ``mesh_from_OrderedDict``).
- :func:`solve_modes`: a full-vector FEM mode solver (2nd-order Nedelec x Lagrange, femwell's formulation of
  doi:10.1080/02726340290084012) generalized to diagonal anisotropic eps and mu, which gives
  * ``formulation="exact"``: the exact transformation-optics map of a bend (eps_t = eps rho, eps_s = eps/rho,
    mu_t = rho, mu_s = 1/rho, rho = 1 + x/R). This is Maxwell's equations in the bend's orthogonal coordinates
    (x, y, s = R theta) with no approximation;
  * ``formulation="isotropic"``: femwell's conformal approximation eps rho^2, mu = 1 (reproduces
    ``femwell.maxwell.waveguide.compute_modes(radius=R)`` exactly; kept to quantify its error);
  * an optional complex-coordinate PML on the lateral (x) boundaries, for bend and slab radiation loss.
- :func:`tidy3d_mode`: the Tidy3D LOCAL mode solver (never the cloud), on an interface-aligned non-uniform grid.
- :func:`richardson`: Richardson extrapolation with an error estimate.
- :func:`write_artefact`: artefact + JSON sidecar (ARCHITECTURE section 5).

Coordinates: lengths in um inside the FEM; x lateral (positive = outside of the ring, away from its centre),
y vertical with the BOX top (film bottom) at y = 0. Time convention exp(+j w t), fields exp(-j beta s); a lossy mode
has Im(n_eff) < 0 (femwell's convention); we report alpha >= 0.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "mrm" / "data" / "extracted"
C0 = 299_792_458.0
LAM0_UM = 1.31
T_REF_C = 25.0
FSR_DESIGN_HZ = 1.8e12          # G25 design FSR, printed (LM-P section 2; 01 section 2.2)
R_PDK_UM = 7.5                  # GF PDK MRM radius, printed (GF-ECIO); also the "15 um diameter"
MEEP_PYTHON = "/Users/kenny/photonics-simulations/.meep/bin/python"

# --------------------------------------------------------------------------------------------- cross-sections


@dataclass(frozen=True)
class XSec:
    """One waveguide cross-section hypothesis.

    width_nm is the rib/strip width at mid-height of the etched sidewall (so the area does not depend on the angle);
    film_nm the Si film; slab_nm the remaining slab (0 = strip); sidewall_deg the sidewall angle from the substrate
    plane (90 = vertical). Claddings: BOX (``box_kind``) below, ``top_kind`` above and beside the rib. Optional
    ``box_um``/``below_box`` describe a thin BOX on air (substrate removed, de Cea 2019).
    """

    name: str
    width_nm: float
    film_nm: float
    slab_nm: float
    sidewall_deg: float = 90.0
    box_um: float = 2.0
    top_kind: str = "beol"
    box_kind: str = "thermal"
    below_box: str = "sio2"            # "sio2": BOX continues to the domain edge; "air": substrate removed
    lam0_um: float = LAM0_UM
    provenance: Mapping[str, Any] = field(default_factory=dict)
    ranges_nm: Mapping[str, Sequence[float]] = field(default_factory=dict)

    @property
    def is_rib(self) -> bool:
        return self.slab_nm > 0.0

    def replace(self, **kw) -> "XSec":
        return dataclasses.replace(self, **kw)

    def as_dict(self) -> dict:
        return {"name": self.name, "width_nm": self.width_nm, "film_nm": self.film_nm, "slab_nm": self.slab_nm,
                "sidewall_deg": self.sidewall_deg, "clad": {"top": f"SiO2 ({self.top_kind})",
                                                           "box": f"SiO2 ({self.box_kind}), {self.box_um} um",
                                                           "below_box": self.below_box},
                "width_definition": "mid-height of the etched sidewall", "lam0_nm": self.lam0_um * 1e3,
                "provenance": dict(self.provenance), "ranges_nm": {k: list(v) for k, v in self.ranges_nm.items()}}


_GEOM_NOTE = ("GF 45SPCLO photonics SOI 160 nm and BOX 2 um are printed (GF-IMAPS, 01 section 4.2); the partial-etch "
              "(slab) depth, the rib width of Lightmatter's MRM and the sidewall angle are NOT public. Width 400 nm: "
              "Lightmatter SMTAI render (400 x 200 nm, printed for an unspecified product) and the BU 45CLO rib ring "
              "(w = 410 nm, R = 8 um, printed, Al Qubaisi et al. CLEO 2021). Slab: assumed, 60 nm on 160 and 80 nm on "
              "200 (slab/film 0.375-0.4, typical foundry ribs such as 220/90 and 220/70 nm; the same values as "
              "junction.py), swept for the Jacobian and in slab_sweep.")

XSECS: dict[str, XSec] = {
    "rib160": XSec(
        "rib160", width_nm=400.0, film_nm=160.0, slab_nm=60.0,
        provenance={"film_nm": "printed (GF-IMAPS)", "width_nm": "assumed (SMTAI render 400; BU 45CLO 410)",
                    "slab_nm": "assumed", "sidewall_deg": "assumed", "box_um": "printed (GF-IMAPS)",
                    "note": _GEOM_NOTE},
        ranges_nm={"width_nm": (360.0, 450.0), "film_nm": (150.0, 170.0), "slab_nm": (40.0, 100.0)}),
    "rib200": XSec(
        "rib200", width_nm=400.0, film_nm=200.0, slab_nm=80.0,
        provenance={"film_nm": "printed (SMTAI render, conflicts with GF 160 nm; 07 D21)",
                    "width_nm": "printed (SMTAI render)", "slab_nm": "assumed", "sidewall_deg": "assumed",
                    "box_um": "printed (GF-IMAPS)", "note": _GEOM_NOTE},
        ranges_nm={"width_nm": (360.0, 450.0), "film_nm": (190.0, 210.0), "slab_nm": (50.0, 120.0)}),
    "strip400x200": XSec(
        "strip400x200", width_nm=400.0, film_nm=200.0, slab_nm=0.0,
        provenance={"film_nm": "printed (SMTAI render; textbook REF.waveguideLmNm)", "width_nm": "printed",
                    "note": "the textbook's reference strip (06 audit, 05 smoke tests); oxide-clad, vertical walls"},
        ranges_nm={"width_nm": (380.0, 420.0), "film_nm": (190.0, 210.0)}),
    # VAL-7 replication: de Cea, Atabaki, Ram, Opt. Express 27, 24274 (2019), Table 1 and section 3 (printed):
    # zero-change 45 nm SOI CMOS ring, outer radius 10 um, 1.7 um wide, "roughly 100 nm thick", lambda0 1545.2 nm.
    # BOX ~145 nm on a locally removed substrate and an oxide top are ASSUMED from the platform (Orcutt 2012).
    "decea2019": XSec(
        "decea2019", width_nm=1700.0, film_nm=100.0, slab_nm=0.0, box_um=0.145, below_box="air",
        top_kind="beol", lam0_um=1.5452,
        provenance={"width_nm": "printed (de Cea 2019 section 3)", "film_nm": "printed as 'roughly 100 nm'",
                    "box_um": "assumed (zero-change 45 nm SOI BOX ~145 nm, substrate removed below the ring)",
                    "note": "outer radius 10 um printed; centreline radius 9.15 um"},
        ranges_nm={"film_nm": (80.0, 120.0), "width_nm": (1600.0, 1800.0)}),
}


# --------------------------------------------------------------------------------------------- materials


def indices(xs: XSec, lam_um: float, t_k: float, *, si_level: str = "frey",
            dn: Mapping[str, float] | None = None) -> dict[str, float]:
    """Per-region refractive index at (lambda, T) from :mod:`mrm.materials` (the only allowed source).

    Regions: si (core and slab), box, clad (top), air. ``dn`` adds a perturbation per region (index weights)."""
    from mrm import materials as M
    lam_m = lam_um * 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", M.MaterialRangeWarning)
        n = {"si": float(M.n_si(lam_m, t_k, level=si_level)),
             "box": float(M.n_sio2(lam_m, t_k, kind=xs.box_kind)),
             "clad": float(M.n_sio2(lam_m, t_k, kind=xs.top_kind)),
             "air": 1.0}
    if dn:
        for k, v in dn.items():
            n[k] = n[k] + v
    return n


def group_indices(xs: XSec, lam_um: float, t_k: float, *, si_level: str = "frey") -> dict[str, float]:
    """Material group indices n_g,i = n - lambda dn/dlambda of each region (same models as :func:`indices`)."""
    from mrm import materials as M
    lam_m = lam_um * 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", M.MaterialRangeWarning)
        return {"si": float(M.n_g_si(lam_m, t_k, level=si_level)),
                "box": float(M.n_g_sio2(lam_m, t_k, kind=xs.box_kind)),
                "clad": float(M.n_g_sio2(lam_m, t_k, kind=xs.top_kind)), "air": 1.0}


def dn_dt(xs: XSec, lam_um: float, t_k: float) -> dict[str, float]:
    """dn_i/dT of each region, 1/K (mrm.materials)."""
    from mrm import materials as M
    lam_m = lam_um * 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", M.MaterialRangeWarning)
        return {"si": float(M.dn_dt_si_per_k(lam_m, t_k)), "box": float(M.dn_dt_sio2_per_k(t_k, xs.box_kind)),
                "clad": float(M.dn_dt_sio2_per_k(t_k, xs.top_kind)), "air": 0.0}


# --------------------------------------------------------------------------------------------- geometry / mesh


def si_polygons(xs: XSec, x0_um: float = 0.0, width_nm: float | None = None):
    """The rib (or strip) polygon of a guide centred at x0 (shapely), film bottom at y = 0, in um."""
    import shapely
    w = (xs.width_nm if width_nm is None else width_nm) * 1e-3
    t, s = xs.film_nm * 1e-3, xs.slab_nm * 1e-3
    he = t - s
    d = 0.0 if xs.sidewall_deg >= 89.999 else he / math.tan(math.radians(xs.sidewall_deg))
    wb, wt = w + d, w - d                      # bottom (at the slab top) and top widths
    if wt <= 0:
        raise ValueError(f"{xs.name}: sidewall too shallow for the width")
    if s > 0:
        pts = [(x0_um - wb / 2, 0.0), (x0_um + wb / 2, 0.0), (x0_um + wb / 2, s), (x0_um + wt / 2, t),
               (x0_um - wt / 2, t), (x0_um - wb / 2, s)]
    else:
        pts = [(x0_um - wb / 2, 0.0), (x0_um + wb / 2, 0.0), (x0_um + wt / 2, t), (x0_um - wt / 2, t)]
    return shapely.Polygon(pts)


@dataclass
class Domain:
    """Computational window (um): x in [x_min, x_max], y in [y_min, y_max]; optional lateral PML."""
    x_min: float = -2.0
    x_max: float = 2.0
    y_min: float = -1.5
    y_max: float = 1.5
    pml_left_um: float = 0.0           # PML thickness inside the window at the x_min side
    pml_right_um: float = 0.0
    pml_strength: float = 2.0          # s(x) = 1 - j a u^2, u = depth fraction in [0, 1]

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class MeshSpec:
    """Mesh controls (um). ``scale`` multiplies every size (the convergence parameter)."""
    h_core: float = 0.010
    h_slab: float = 0.020
    h_max: float = 0.20
    core_dist: float = 0.40
    slab_dist: float = 0.15
    scale: float = 1.0

    def as_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["h_core_eff_nm"] = self.h_core * self.scale * 1e3
        return d


def build_mesh(xs: XSec, dom: Domain, ms: MeshSpec, *, guides: Sequence[tuple[float, float]] | None = None,
               slab_in_gap: bool = True):
    """Graded triangle mesh (skfem.Mesh) with subdomains: core* (one per guide), slab, box, clad[, air].

    guides: [(x_centre_um, width_nm), ...]; default one guide at x = 0 with the xsec width. For two guides and
    slab_in_gap=False the slab between the guides is removed (fully etched gap)."""
    import shapely
    from femwell.mesh import mesh_from_OrderedDict
    from skfem.io.meshio import from_meshio
    if guides is None:
        guides = [(0.0, xs.width_nm)]
    sc = ms.scale
    polys: "OrderedDict[str, Any]" = OrderedDict()
    res: dict[str, dict] = {}
    for i, (xc, wn) in enumerate(guides):
        name = "core" if len(guides) == 1 else f"core{i}"
        polys[name] = si_polygons(xs, xc, wn)
        res[name] = {"resolution": ms.h_core * sc, "distance": ms.core_dist, "SizeMax": ms.h_max * sc}
    t, s = xs.film_nm * 1e-3, xs.slab_nm * 1e-3
    if s > 0:
        slab = shapely.box(dom.x_min, 0.0, dom.x_max, s)
        if len(guides) == 2 and not slab_in_gap:
            (x1, w1), (x2, w2) = sorted(guides)
            slab = slab.difference(shapely.box(x1 + w1 * 5e-4, 0.0, x2 - w2 * 5e-4, s))
        polys["slab"] = slab
        res["slab"] = {"resolution": ms.h_slab * sc, "distance": ms.slab_dist, "SizeMax": ms.h_max * sc}
    yb = max(dom.y_min, -xs.box_um) if xs.below_box == "air" else dom.y_min
    polys["box"] = shapely.box(dom.x_min, yb, dom.x_max, 0.0)
    if xs.below_box == "air" and dom.y_min < -xs.box_um:
        polys["air"] = shapely.box(dom.x_min, dom.y_min, dom.x_max, -xs.box_um)
    polys["clad"] = shapely.box(dom.x_min, 0.0, dom.x_max, dom.y_max)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = mesh_from_OrderedDict(polys, res, default_resolution_max=ms.h_max * sc)
    return from_meshio(m)


def region_material(name: str) -> str:
    if name.startswith("core") or name == "slab":
        return "si"
    return name


# --------------------------------------------------------------------------------------------- FEM mode solver


@dataclass
class ModeSol:
    """One eigenmode of :func:`solve_modes`. Integrals are evaluated lazily on demand."""
    n_eff: complex
    lam_um: float
    radius_um: float | None
    formulation: str
    basis: Any
    basis0: Any
    eps: np.ndarray          # complex P0 permittivity (physical, untransformed)
    x: np.ndarray            # solution vector (e_t, E_s) with E_s already un-scaled (physical E_s')
    coeff: Callable          # (xq, eps_q) -> (ex, ey, es, imux, imuy, imus)
    regions: tuple
    dom: Any = None
    _cache: dict = field(default_factory=dict)

    @property
    def k0(self) -> float:
        return 2 * math.pi / self.lam_um

    @property
    def beta(self) -> complex:
        return self.n_eff * self.k0

    @property
    def alpha_db_per_cm(self) -> float:
        """Power loss from Im(n_eff) (>= 0 for a decaying mode)."""
        return float(-20.0 / math.log(10.0) * self.k0 * np.imag(self.n_eff) * 1e4)

    # ---- fields at quadrature points
    def _region_basis(self, region):
        from skfem import Basis
        key = ("rb", region)
        if key not in self._cache:
            b = self.basis.with_elements(region) if region is not None else self.basis
            b0 = self.basis0.with_elements(region) if region is not None else self.basis0
            self._cache[key] = (b, b0)
        return self._cache[key]

    def quad_fields(self, region=None):
        """Physical-coordinate fields at the quadrature points of ``region`` (None = everywhere), plus weights.

        Returns dict with E (3, ...), Ht (eta0 H, 3 comps), rho, eps, dx (quadrature weights x |J|), x, y."""
        key = ("qf", region)
        if key in self._cache:
            return self._cache[key]
        b, b0 = self._region_basis(region)
        et, ez = b.interpolate(self.x)
        epsq = np.asarray(b0.interpolate(self.eps))
        xq = np.asarray(b.global_coordinates()[0])
        yq = np.asarray(b.global_coordinates()[1])
        ex, ey, es, imux, imuy, imus = self.coeff(xq, epsq)
        Ex, Ey, Es = np.asarray(et[0]), np.asarray(et[1]), np.asarray(ez)
        k0, beta = self.k0, self.beta
        # eta0 H = (j / (k0 mu)) curl E, with d/ds -> -j beta (fields exp(-j beta s))
        Hx = 1j / k0 * imux * (np.asarray(ez.grad[1]) + 1j * beta * Ey)
        Hy = 1j / k0 * imuy * (-1j * beta * Ex - np.asarray(ez.grad[0]))
        Hs = 1j / k0 * imus * np.asarray(et.curl)
        dx = b.dx
        R = self.radius_um
        rho = np.ones_like(xq) if (R is None or not np.isfinite(R)) else 1.0 + xq / R
        out = dict(E=(Ex, Ey, Es), H=(Hx, Hy, Hs), eps=epsq, dx=dx, x=xq, y=yq, rho=rho)
        self._cache[key] = out
        return out

    def power(self, region=None) -> float:
        """Re integral of (E x eta0 H*) . s over the region (twice the power in units of 1/eta0)."""
        q = self.quad_fields(region)
        Ex, Ey, _ = q["E"]
        Hx, Hy, _ = q["H"]
        Ss = np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))
        return float(np.sum(Ss * q["dx"]))

    def e2_weighted(self, region=None) -> float:
        """integral over the region of rho |E_t|^2 + |E_s'|^2 / rho (= physical |E|^2 with volume element r/R)."""
        q = self.quad_fields(region)
        Ex, Ey, Es = q["E"]
        rho = q["rho"]
        return float(np.sum((rho * (np.abs(Ex) ** 2 + np.abs(Ey) ** 2) + np.abs(Es) ** 2 / rho) * q["dx"]))

    def index_weight(self, region) -> float:
        """dn_eff/dn_region by first-order perturbation (Hellmann-Feynman, Poynting normalized):
        n_i * int_i (rho|E_t|^2 + |E_s'|^2/rho) / Re int (E x eta0 H*).s  (= Robinson 2008 confinement)."""
        q = self.quad_fields(region)
        n_i = float(np.sqrt(np.real(np.mean(q["eps"]))))
        return n_i * self.e2_weighted(region) / self.power(None)

    def energy_fraction(self, region) -> float:
        """Electric-energy fraction int_i eps|E|^2 dV / int eps|E|^2 dV (physical volume element)."""
        def ee(reg):
            q = self.quad_fields(reg)
            Ex, Ey, Es = q["E"]
            rho = q["rho"]
            return float(np.sum(np.real(q["eps"]) * (rho * (np.abs(Ex) ** 2 + np.abs(Ey) ** 2)
                                                    + np.abs(Es) ** 2 / rho) * q["dx"]))
        return ee(region) / ee(None)

    def e2_fraction(self, region=None) -> float:
        """|E|^2 fraction (positive definite) in a region; used to recognise guided vs PML/box modes."""
        def e2(reg):
            q = self.quad_fields(reg)
            return float(np.sum(sum(np.abs(c) ** 2 for c in q["E"]) * q["dx"]))
        return e2(region) / e2(None)

    def pml_fraction(self) -> float:
        """|E|^2 fraction inside the PML layers (physical modes: << 1; spurious PML modes: large)."""
        d = self.dom
        if d is None or not (d.pml_left_um or d.pml_right_um):
            return 0.0
        q = self.quad_fields(None)
        x = q["x"]
        inside = np.zeros_like(x, dtype=bool)
        if d.pml_right_um:
            inside |= x > d.x_max - d.pml_right_um
        if d.pml_left_um:
            inside |= x < d.x_min + d.pml_left_um
        w = sum(np.abs(c) ** 2 for c in q["E"]) * q["dx"]
        return float(np.sum(w * inside) / np.sum(w))

    def te_fraction(self) -> float:
        q = self.quad_fields(None)
        Ex, Ey, _ = q["E"]
        a = np.sum(np.abs(Ex) ** 2 * q["dx"])
        b = np.sum(np.abs(Ey) ** 2 * q["dx"])
        return float(a / (a + b))

    def power_fraction(self, region) -> float:
        return self.power(region) / self.power(None)

    def mode_area_um2(self) -> float:
        """Poynting effective area (int S)^2 / int S^2 over the whole window (Afshar & Monro 2009 style)."""
        q = self.quad_fields(None)
        Ex, Ey, _ = q["E"]
        Hx, Hy, _ = q["H"]
        Ss = np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))
        return float(np.sum(Ss * q["dx"]) ** 2 / np.sum(Ss ** 2 * q["dx"]))

    def centroid_x_um(self) -> float:
        q = self.quad_fields(None)
        Ex, Ey, _ = q["E"]
        Hx, Hy, _ = q["H"]
        Ss = np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))
        return float(np.sum(Ss * q["x"] * q["dx"]) / np.sum(Ss * q["dx"]))

    def lateral_parity(self) -> float:
        """Signed overlap of the dominant transverse field with its mirror image in x (straight guides only)."""
        q = self.quad_fields(None)
        Ex, Ey, _ = q["E"]
        F = Ex if self.te_fraction() > 0.5 else Ey
        # Compare the integral of F over x>0 and x<0 (weighted by |F|): +1 even, -1 odd
        pos = q["x"] > 0
        a = np.sum((F * pos) * q["dx"])
        b = np.sum((F * ~pos) * q["dx"])
        if abs(a) + abs(b) == 0:
            return 0.0
        return float(np.real(a * np.conj(b)) / (abs(a) * abs(b) + 1e-300))


def _pml_s(x: np.ndarray, dom: Domain | None) -> tuple[np.ndarray, np.ndarray]:
    """PML stretch s(x) = 1 - j a u^2 (u = depth fraction) and the stretched coordinate x~ = int s dx, which the
    bend factor rho = 1 + x~/R must use inside the PML (analytic continuation of the radius; with a real rho the
    PML is not passive and spurious gain modes appear)."""
    s = np.ones_like(x, dtype=complex)
    xt = np.asarray(x, dtype=complex).copy()
    if dom is None:
        return s, xt
    if dom.pml_right_um > 0:
        d = dom.pml_right_um
        x0 = dom.x_max - d
        u = np.clip((x - x0) / d, 0.0, None)
        s = s - 1j * dom.pml_strength * u ** 2
        xt = xt - 1j * dom.pml_strength * d * u ** 3 / 3.0
    if dom.pml_left_um > 0:
        d = dom.pml_left_um
        x0 = dom.x_min + d
        u = np.clip((x0 - x) / d, 0.0, None)
        s = s - 1j * dom.pml_strength * u ** 2
        xt = xt + 1j * dom.pml_strength * d * u ** 3 / 3.0
    return s, xt


def make_coeff(radius_um: float | None, formulation: str, dom: Domain | None) -> Callable:
    """Diagonal material tensors of the transformed problem at quadrature points.

    exact:     eps' = eps diag(rho/s, rho s, s/rho), mu' = diag(rho/s, rho s, s/rho)
    isotropic: eps' = eps rho^2 diag(1/s, s, s),    mu' = diag(1/s, s, s)   (femwell's conformal approximation)
    with rho = 1 + x~/R (x~ the PML-stretched coordinate) and the PML stretch s(x).
    Returns (eps_x, eps_y, eps_s, 1/mu_x, 1/mu_y, 1/mu_s)."""
    straight = radius_um is None or not np.isfinite(radius_um) or radius_um == 0
    if formulation not in ("exact", "isotropic"):
        raise ValueError("formulation must be 'exact' or 'isotropic'")

    def coeff(xq, epsq):
        s, xt = _pml_s(xq, dom)
        rho = np.ones_like(xt) if straight else 1.0 + xt / radius_um
        if formulation == "exact" or straight:
            ex, ey, es = epsq * rho / s, epsq * rho * s, epsq * s / rho
            mx, my, msz = rho / s, rho * s, s / rho
        else:
            ex, ey, es = epsq * rho ** 2 / s, epsq * rho ** 2 * s, epsq * rho ** 2 * s
            mx, my, msz = 1.0 / s, s, s
        return ex, ey, es, 1.0 / mx, 1.0 / my, 1.0 / msz
    return coeff


def install_fast_finder(mesh) -> None:
    """Replace skfem's point locator (which allocates n_points x n_candidate_elements arrays and exhausts memory
    for large probe sets) by matplotlib's trapezoid-map triangle finder."""
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    tf = tri.get_trifinder()

    def element_finder(mapping=None):
        def finder(x, y, _search_all=False):
            c = np.asarray(tf(np.asarray(x, float), np.asarray(y, float)))
            if np.any(c < 0):
                raise ValueError("probe point outside the mesh")
            return c.astype(np.int64)
        return finder
    object.__setattr__(mesh, "element_finder", element_finder)


class FemProblem:
    """A mesh with its bases, reused for many (lambda, T, perturbation) solves."""

    def __init__(self, mesh, order: int = 2):
        from skfem import Basis, ElementTriN1, ElementTriN2, ElementTriP0, ElementTriP1, ElementTriP2
        install_fast_finder(mesh)
        self.mesh = mesh
        el = ElementTriN2() * ElementTriP2() if order == 2 else ElementTriN1() * ElementTriP1()
        b0 = Basis(mesh, ElementTriP0())
        self.basis = b0.with_element(el)
        self.basis0 = self.basis.with_element(ElementTriP0())
        self.regions = tuple(k for k in mesh.subdomains.keys() if not k.startswith("gmsh"))
        self._dofs0 = {r: self.basis0.get_dofs(elements=r) for r in self.regions}
        self.ndofs = int(self.basis.N)
        self.ntri = int(mesh.nelements)

    def eps_vector(self, n_by_material: Mapping[str, complex]) -> np.ndarray:
        eps = self.basis0.zeros(dtype=complex)
        for r in self.regions:
            eps[self._dofs0[r]] = complex(n_by_material[region_material(r)]) ** 2
        return eps

    def solve(self, n_by_material: Mapping[str, complex], lam_um: float, *, radius_um: float | None = None,
              formulation: str = "exact", dom: Domain | None = None, num_modes: int = 4,
              n_guess: float | None = None, metallic: bool = False) -> list[ModeSol]:
        from skfem import BilinearForm, condense, solve
        from skfem.helpers import curl, grad
        from skfem.utils import solver_eigen_scipy
        k0 = 2 * math.pi / lam_um
        eps = self.eps_vector(n_by_material)
        coeff = make_coeff(radius_um, formulation, dom if (dom and (dom.pml_left_um or dom.pml_right_um)) else None)

        @BilinearForm(dtype=complex)
        def aform(e_t, e_z, v_t, v_z, w):
            ex, ey, es, imux, imuy, imus = coeff(w.x[0], w.epsilon)
            ge, gv = grad(e_z), grad(v_z)
            return (imus * curl(e_t) * curl(v_t) / k0 ** 2
                    - (ex * e_t[0] * v_t[0] + ey * e_t[1] * v_t[1])
                    + (imuy * ge[0] * v_t[0] + imux * ge[1] * v_t[1])
                    + (ex * e_t[0] * gv[0] + ey * e_t[1] * gv[1])
                    - es * e_z * v_z * k0 ** 2)

        @BilinearForm(dtype=complex)
        def bform(e_t, e_z, v_t, v_z, w):
            ex, ey, es, imux, imuy, imus = coeff(w.x[0], w.epsilon)
            return -(imuy * e_t[0] * v_t[0] + imux * e_t[1] * v_t[1]) / k0 ** 2

        epsq = self.basis0.interpolate(eps)
        A = aform.assemble(self.basis, epsilon=epsq)
        B = bform.assemble(self.basis, epsilon=epsq)
        nmax = max(abs(complex(v)) for v in n_by_material.values())
        sigma = k0 ** 2 * (n_guess ** 2 if n_guess else nmax ** 2 * 1.1)
        from scipy.sparse.linalg import ArpackError, ArpackNoConvergence

        def _eig(ncv=None, sig=sigma):
            kw = {"k": num_modes, "sigma": sig}
            if ncv:
                kw["ncv"] = ncv
            if metallic:
                return solve(*condense(-A, -B, D=self.basis.get_dofs(), x=self.basis.zeros(dtype=complex)),
                             solver=solver_eigen_scipy(**kw))
            return solve(-A, -B, solver=solver_eigen_scipy(**kw))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lams, xs = _eig()
            except (ArpackError, ArpackNoConvergence):      # crowded PML spectra: larger Krylov space, nudged shift
                lams, xs = _eig(ncv=max(4 * num_modes + 20, 40), sig=sigma * (1 + 1e-4))
        del A, B
        zi = self.basis.split_indices()[1]
        out = []
        for i in range(len(lams)):
            beta = np.sqrt(complex(lams[i]))
            if np.real(beta) < 0:
                beta = -beta
            x = xs[:, i].copy()
            # e_z(femwell) = -j beta E_s / k0^2  ->  E_s = j k0^2 e_z / beta
            x[zi] = x[zi] * 1j * k0 ** 2 / beta
            out.append(ModeSol(n_eff=complex(beta / k0), lam_um=lam_um, radius_um=radius_um,
                               formulation=formulation, basis=self.basis, basis0=self.basis0, eps=eps, x=x,
                               coeff=coeff, regions=self.regions, dom=dom))
        out.sort(key=lambda m: -np.real(m.n_eff))
        return out


def pick_mode(modes: Sequence[ModeSol], family: str = "TE0", core_regions: Iterable[str] = ("core",),
              min_core_power: float = 0.2, near: float | None = None, near_tol: float = 0.005,
              max_gain_im: float = 1e-8, max_neff: float | None = None) -> ModeSol:
    """Select TE0 / TM0 / TE1 among solver modes: polarization by te_fraction, guidance by the power fraction in
    the core regions (rejects slab, box and PML modes), no gain (Im n_eff <= max_gain_im; PML spurious modes),
    then the n_eff ranking (TE1 = second TE member) or, with ``near``, the mode closest to that index."""
    cand = []
    for m in modes:
        te = m.te_fraction()
        pc = sum(m.power_fraction(r) for r in core_regions if r in m.regions)
        cand.append((m, te, pc))
    pol = "TE" if family.startswith("TE") else "TM"
    sel = [(m, te, pc) for (m, te, pc) in cand if ((te > 0.5) == (pol == "TE")) and pc >= min_core_power
           and np.imag(m.n_eff) <= max_gain_im and (max_neff is None or np.real(m.n_eff) < max_neff)]
    if near is not None:
        close = [t for t in sel if abs(np.real(t[0].n_eff) - near) < near_tol]
        if close:
            # with a PML the spectrum near the target is crowded by PML modes: take the candidate with the least
            # field inside the PML (then the closest index)
            pf = [t[0].pml_fraction() for t in close]
            best = min(pf)
            close = [t for t, p in zip(close, pf) if p <= max(2 * best, best + 1e-3)]
            return min(close, key=lambda t: abs(np.real(t[0].n_eff) - near))[0]
    sel.sort(key=lambda t: -np.real(t[0].n_eff))
    idx = int(family[2:]) if (len(family) > 2 and max_neff is None) else 0
    if len(sel) <= idx:
        raise RuntimeError(f"mode {family} not found among "
                           f"{[(round(float(np.real(m.n_eff)), 5), round(float(np.imag(m.n_eff)), 9), round(te, 3), round(pc, 3)) for m, te, pc in cand]}")
    return sel[idx][0]


# --------------------------------------------------------------------------------------------- Tidy3D (local)


def _grid_boundaries(lo: float, hi: float, fine_lo: float, fine_hi: float, dl: float, dl_coarse: float,
                     offset: float = 0.0, growth: float = 1.2) -> np.ndarray:
    """1-D grid lines from lo to hi: uniform dl on [fine_lo, fine_hi] with lines at offset + k*dl (offset = 0 puts
    lines on interfaces that are multiples of dl, offset = dl/2 puts those interfaces at cell centres), and cells
    growing geometrically (x growth, up to dl_coarse) outside."""
    k0 = math.floor((fine_lo - offset) / dl)
    k1 = math.ceil((fine_hi - offset) / dl)
    core = offset + dl * np.arange(k0, k1 + 1)
    left, d, x = [], dl, core[0]
    while x - d * growth > lo:
        d = min(d * growth, dl_coarse)
        x -= d
        left.append(x)
    right, d, x = [], dl, core[-1]
    while x + d * growth < hi:
        d = min(d * growth, dl_coarse)
        x += d
        right.append(x)
    return np.array(left[::-1] + core.tolist() + right)


def tidy3d_mode(xs: XSec, lam_um: float, t_k: float, *, radius_um: float | None = None, dl_fine_um: float = 0.005,
                dl_coarse_um: float = 0.08, half_width_um: float = 2.0, half_height_um: float = 1.5,
                outer_extra_um: float = 0.0, num_pml: tuple[int, int] = (0, 0), num_modes: int = 2,
                family: str = "TE0", n_by_material: Mapping[str, float] | None = None,
                target_neff: float | None = None) -> dict:
    """TE0 (or TM0) n_eff from the Tidy3D LOCAL mode solver (ModeSolver.solve(), no web call).

    The plane is y (lateral, our x) by z (vertical, our y). Grid lines sit on every material interface (no
    subpixel smoothing is available locally). For a bend (radius_um), +R and -R are both solved and averaged (the
    +/-R asymmetry is a 1/R grid artefact, 05 verification log #8); both are returned."""
    import tidy3d as td
    from tidy3d.plugins.mode import ModeSolver
    td.config.logging.level = "ERROR"
    n = dict(n_by_material) if n_by_material else indices(xs, lam_um, t_k)
    w, t, s = xs.width_nm * 1e-3, xs.film_nm * 1e-3, xs.slab_nm * 1e-3
    if xs.sidewall_deg < 89.999:
        raise NotImplementedError("tidy3d cross-check implemented for vertical sidewalls only")
    y_lo, y_hi = -half_width_um, half_width_um + outer_extra_um
    z_lo, z_hi = -half_height_um, half_height_um
    # Interfaces: sidewalls on y grid lines (E_y, the dominant TE component, sits at y cell centres); horizontal
    # interfaces at z cell centres (E_y sits on z grid lines, so it never samples an interface). Uniform dl
    # within 0.15 um of the core, graded outside (no subpixel smoothing locally).
    yb = _grid_boundaries(y_lo, y_hi, -w / 2 - 0.15, w / 2 + 0.15, dl_fine_um, dl_coarse_um, offset=0.0)
    zb = _grid_boundaries(z_lo, z_hi, -0.15, t + 0.15, dl_fine_um, dl_coarse_um, offset=0.5 * dl_fine_um)
    yc, zc = 0.5 * (yb[0] + yb[-1]), 0.5 * (zb[0] + zb[-1])
    Ly, Lz = yb[-1] - yb[0], zb[-1] - zb[0]
    med = lambda nn: td.Medium(permittivity=float(nn) ** 2)
    big = 100.0
    structs = [td.Structure(geometry=td.Box(center=(0, 0, -big / 2), size=(td.inf, td.inf, big)), medium=med(n["box"]))]
    if xs.below_box == "air":
        structs.append(td.Structure(geometry=td.Box(center=(0, 0, -xs.box_um - big / 2), size=(td.inf, td.inf, big)),
                                    medium=med(n["air"])))
    if s > 0:
        structs.append(td.Structure(geometry=td.Box(center=(0, 0, s / 2), size=(td.inf, td.inf, s)), medium=med(n["si"])))
    structs.append(td.Structure(geometry=td.Box(center=(0, 0, t / 2), size=(td.inf, w, t)), medium=med(n["si"])))
    grid = td.GridSpec(grid_x=td.UniformGrid(dl=0.05),
                       grid_y=td.CustomGridBoundaries(coords=tuple(yb.tolist())),
                       grid_z=td.CustomGridBoundaries(coords=tuple(zb.tolist())))
    sim = td.Simulation(center=(0, yc, zc), size=(1.0, Ly, Lz), grid_spec=grid, structures=structs,
                        medium=med(n["clad"]), run_time=1e-12,
                        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()))
    plane = td.Box(center=(0, yc, zc), size=(0, Ly, Lz))
    out = {"grid_cells": [int(len(yb) - 1), int(len(zb) - 1)], "dl_fine_um": dl_fine_um,
           "plane_um": [float(Ly), float(Lz)], "num_pml": list(num_pml)}

    def run(R):
        kw = dict(num_modes=num_modes, num_pml=num_pml, precision="double",
                  filter_pol="te" if family.startswith("TE") else "tm")
        if target_neff is not None:
            kw["target_neff"] = float(target_neff)
        elif family.startswith("TE"):
            kw["target_neff"] = float(n["si"]) * 0.95
        if R is not None:
            kw.update(bend_radius=R, bend_axis=1)
        ms = ModeSolver(simulation=sim, plane=plane, mode_spec=td.ModeSpec(**kw), freqs=[C0 / (lam_um * 1e-6)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = ms.solve()
        ne = np.asarray(d.n_complex.values[0, :])
        if target_neff is not None and len(ne) > 1:
            # PML runs return box/PML modes too: take the mode closest to the expected index with the least loss
            score = np.abs(np.real(ne) - target_neff) + 10.0 * np.abs(np.imag(ne))
            return complex(ne[int(np.argmin(score))])
        return complex(ne[0])

    t0 = time.time()
    if radius_um is None:
        out["n_eff"] = run(None)
    else:
        npos = run(+radius_um)
        if outer_extra_um > 0:
            out["n_eff_plus"] = npos
            out["n_eff"] = npos       # asymmetric window: only +R has the extra room on its outer side
        else:
            nneg = run(-radius_um)
            out["n_eff_plus"], out["n_eff_minus"] = npos, nneg
            out["n_eff"] = 0.5 * (npos + nneg)
    out["seconds"] = round(time.time() - t0, 2)
    return out


# --------------------------------------------------------------------------------------------- numerics helpers


def richardson(h: Sequence[float], f: Sequence[float], p: float | None = None) -> dict:
    """Richardson extrapolation of f(h) = f0 + C h^p from the three finest levels (or a fixed p).

    Returns f0, p, the error estimate of the finest value |f_finest - f0| and the full table."""
    h = np.asarray(h, float)
    f = np.asarray(f, float)
    order = np.argsort(h)
    h, f = h[order], f[order]
    out = {"h": h.tolist(), "f": f.tolist()}
    if len(h) < 3 and p is None:
        out.update(f0=float(f[0]), p=None, err=float(abs(f[0] - f[1])) if len(h) > 1 else None)
        return out
    if p is None:
        h1, h2, h3 = h[:3]
        f1, f2, f3 = f[:3]
        # solve (f3 - f2)/(f2 - f1) = (h3^p - h2^p)/(h2^p - h1^p) for p by bisection
        d21, d32 = f2 - f1, f3 - f2
        if d21 == 0 or d32 == 0 or np.sign(d21) != np.sign(d32):
            out.update(f0=float(f1), p=None, err=float(max(abs(d21), abs(d32))), monotone=False)
            return out
        ratio = d32 / d21
        g = lambda pp: (h3 ** pp - h2 ** pp) / (h2 ** pp - h1 ** pp) - ratio
        lo, hi = 0.05, 12.0
        if g(lo) * g(hi) > 0:
            out.update(f0=float(f1), p=None, err=float(abs(d21)), monotone=True)
            return out
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if g(lo) * g(mid) <= 0:
                hi = mid
            else:
                lo = mid
        p = 0.5 * (lo + hi)
        C = d21 / (h2 ** p - h1 ** p)
        f0 = f1 - C * h1 ** p
    else:
        h1, h2 = h[:2]
        f1, f2 = f[:2]
        C = (f2 - f1) / (h2 ** p - h1 ** p)
        f0 = f1 - C * h1 ** p
    out.update(f0=float(f0), p=float(p), err=float(abs(f[0] - f0)), monotone=True)
    return out


def central_diff(fun: Callable[[float], float], x0: float, h: float, *, richardson_steps: bool = True) -> tuple:
    """Central difference with one Richardson step (h, h/2): returns (derivative, error estimate)."""
    d1 = (fun(x0 + h) - fun(x0 - h)) / (2 * h)
    if not richardson_steps:
        return d1, float("nan")
    d2 = (fun(x0 + h / 2) - fun(x0 - h / 2)) / h
    d = d2 + (d2 - d1) / 3.0
    return d, abs(d - d2)


def tool_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in ("numpy", "scipy", "femwell", "scikit-fem", "gmsh", "shapely", "tidy3d", "meshio"):
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, complex):
        return {"re": o.real, "im": o.imag}
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def write_artefact(name: str, data: Mapping, sidecar: Mapping, *, out_dir: Path | None = None) -> tuple[Path, Path]:
    """Write ``<name>.json`` and its sidecar ``<name>.sidecar.json`` (schema, sha256 of the data file, inputs, tool
    versions, mesh/resolution, validation numbers, created_utc; ARCHITECTURE section 5)."""
    d = Path(out_dir) if out_dir is not None else DATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.json"
    path.write_text(json.dumps(_jsonable(dict(data)), indent=1, sort_keys=False) + "\n")
    side = dict(_jsonable(dict(sidecar)))
    side["sha256"] = sha256_file(path)
    side["data_file"] = path.name
    side.setdefault("created_utc", _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    side.setdefault("tool_versions", tool_versions())
    sp = d / f"{name}.sidecar.json"
    sp.write_text(json.dumps(_jsonable(side), indent=1) + "\n")
    return path, sp


def load_artefact(name: str, *, verify: bool = True, out_dir: Path | None = None) -> tuple[dict, dict]:
    d = Path(out_dir) if out_dir is not None else DATA_DIR
    path = d / f"{name}.json"
    side = json.loads((d / f"{name}.sidecar.json").read_text())
    if verify and side.get("sha256") != sha256_file(path):
        raise ValueError(f"{path.name}: sha256 does not match its sidecar")
    return json.loads(path.read_text()), side


def r_for_fsr_um(n_g: float, fsr_hz: float = FSR_DESIGN_HZ) -> float:
    """Ring radius (um) that gives the FSR: L = c/(n_g FSR), r = L/(2 pi)."""
    return C0 / (n_g * fsr_hz) / (2 * math.pi) * 1e6


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
