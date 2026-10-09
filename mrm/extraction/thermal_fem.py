"""Transient heat-conduction FEM of a microring in a GF-45SPCLO-like stack, and the thermal ROM artefacts.

Contract: ARCHITECTURE section 5.3 (``thermal_rom/1``); 07 P0-1 step 3, P1-2, P1-3, P2-5 and VAL-8. Offline only:
never imported by runtime code.

Run::

    python -m mrm.extraction.thermal_fem                     # every family x film, 3-D bank, VAL-8 (hours)
    python -m mrm.extraction.thermal_fem --family inring_noundercut --film 160
    python -m mrm.extraction.thermal_fem --quick             # cheap run used by the tests (temporary output)
    python -m mrm.extraction.thermal_fem --stage val8        # imec replication only

Run every solver job through ``tools/memguard.py --limit-gb 3 --heavy``.

Physics and numerics
--------------------
* Linear heat conduction about 300 K, rho c dT/dt = div(k grad T) + q, solved with scikit-fem on gmsh meshes.
  Rings and site sources are close to axisymmetric, so the main model is axisymmetric (r, z) with r-weighted forms
  (2 pi r dA) and P2 triangles. A coarse 3-D P1 tetrahedral model checks the axisymmetric reduction and runs the
  8-ring bank (superposition test).
* Temperature-dependent conductivity enters only as the quasi-static scaling of 07 P2-5 (``k_temp_scaling``): the DC
  gains and the 63 % times are recomputed with k(T), rho c(T) at 300 to 400 K and stored as ratios.
* Materials come only from ``mrm.materials`` (k, rho c at T). Exceptions, flagged in the sidecar: air in the undercut
  cavity (k 0.026, rho c 1.2e3; handbook, assumed) and the effective-medium BEOL (metal fill is an assumption).
* Units inside the FEM: lengths in um, so K [W/K] = k [W/(m K)] * 1e-6 * (um form), M [J/K] = rho c * 1e-18 * (um^3
  form), Robin h [W/(m^2 K)] * 1e-12. Sources are normalized to 1 W and observers to weights summing to 1, so every
  transfer function is in K/W.

Geometry (all lengths are assumptions unless stated; see ``ThermalConfig``)
------------------------------------------------------------------------
z = 0 is the top of the BOX. Si substrate (775 um, assumed un-thinned 300 mm wafer) on a heat sink through a contact
conductance h (Robin; default 2e4 W/(m^2 K) = die attach + package to a TEC-held plate, assumed; Dirichlet and other
h are swept); 2 um BOX (printed, 45SPCLO); the 160 or 200 nm film (160 printed for 45SPCLO, 200 the SMTAI render)
with a 400 nm rib and a 60 nm slab (assumed); 1.5 um of pre-metal oxide; a 9-level BEOL lumped into a lower low-k
level (2.5 um) and an upper TEOS level (5 um) with an effective anisotropic metal fill (assumed, swept) and a metal
keep-out above the ring; 1 um passivation; adiabatic top. The die is a 5 x 5 mm block (the 45SPCLO MPW block size,
printed) represented by a cylinder of equal area with an adiabatic edge. Ring radius 7.5 um ("~15 um diameter",
printed by Lightmatter; R = 7.5 um GF PDK ring, printed).

Heater families (07 P1-3)
-------------------------
* ``inring_noundercut`` (default; "heater and PN inside the ring", G26 schematic): a doped-Si resistor annulus in the
  film inside the ring, 1 um wide, 2 um from the rib edge, joined to the rib by the slab.
* ``silicide_undercut`` (GF-PDK-like: four silicide resistors, two inside and two outside, with Si undercut): two
  silicided-Si annuli 1.5 um from the rib edges, and an air cavity in the substrate under the ring (radius r_ring +
  ~6.7 um, 15 um deep), the axisymmetric equivalent of the four arcs.
* ``metal_above``: a TiN annulus 1 um wide, 100 nm thick, 0.8 um above the film, with the metal/dielectric boundary
  resistance of mrm.materials (2e-7 m^2 K/W) as a 20 nm shell.

Sources and observers (ARCHITECTURE 5.3)
----------------------------------------
sources:   heater (uniform in the heater volume), waveguide_absorption (free-carrier absorption: |E|^2 of TE0 in the
           doped Si), junction (series-resistance Joule heat, density ~ 1/t^2 along the lateral current path through
           the slab and rib), driver (5 um radius disk in the 88 nm CMOS film at its site), aggressor (10 um radius
           disk at its site).
observers: ring_mode_weighted (weight n * dn/dT * |E|^2 of the TE0 mode from femwell, bent at R), heater_resistor
           (volume average of the heater), monitor_pd and temp_sensor (disk averages at their sites, film level).
Cross terms use the radially symmetric approximation (valid d > 2R): the distance runs from the source centre (ring
centre for ring sources, site centre for site sources) to the observer centre (ring centre for ring observers, site
centre for site observers); ring observers are averaged over the victim's annulus.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import pathlib
import platform
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from mrm import materials as mt
from mrm.extraction import mor

SCHEMA = "thermal_rom/1"
SCHEMA_VAL8 = "thermal_val8/1"
SOURCES = ("heater", "waveguide_absorption", "junction", "driver", "aggressor")
OBSERVERS = ("ring_mode_weighted", "heater_resistor", "monitor_pd", "temp_sensor")
FAMILIES = ("inring_noundercut", "silicide_undercut", "metal_above")
FILMS = (160, 200)
RING_SOURCES = ("heater", "waveguide_absorption", "junction")
SITE_SOURCES = ("driver", "aggressor")
CROSS_R_UM = (20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0, 120.0, 150.0, 200.0, 250.0, 300.0, 400.0,
              500.0, 700.0, 1000.0, 1500.0, 2000.0)
OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "extracted"
REPO = pathlib.Path(__file__).resolve().parents[2]
# Not in mrm.materials: air in the undercut cavity (handbook, 300 K). The cavity carries < 1 % of the heat either way.
AIR_K_W_PER_M_K = 0.026
AIR_RHO_C_J_PER_M3_K = 1.2e3

_LOG_T0 = time.time()
PERMC = "MMD_AT_PLUS_A"     # symmetric fill-reducing ordering for SuperLU (the pencils are symmetric)


def log(msg: str) -> None:
    print(f"[thermal_fem {time.time() - _LOG_T0:7.1f}s] {msg}", flush=True)


# =============================================================================================== configuration
@dataclass
class ThermalConfig:
    """Every geometric and boundary assumption of the thermal model (lengths in um unless the name says nm)."""
    family: str = "inring_noundercut"
    film_nm: float = 160.0
    t_k: float = 300.0
    lam_m: float = 1.31e-6
    # ring cross-section
    radius_um: float = 7.5
    width_nm: float = 400.0
    slab_nm: float = 60.0
    slab_out_um: float = 2.0            # slab extent outside the rib (outer contact region)
    slab_in_um: float = 2.0             # slab extent inside the rib (extended to the heater for in-ring families)
    contact_offset_um: float = 1.0      # lateral current path in the slab on each side (junction heat)
    # stack
    box_um: float = 2.0
    sub_um: float = 775.0
    die_side_mm: float = 5.0
    bottom_bc: str = "robin"            # robin | dirichlet
    h_bottom_w_per_m2_k: float = 2e4
    clad_um: float = 1.5                # BOX top to the first BEOL level
    beol_lo_um: float = 2.5
    beol_hi_um: float = 5.0
    pass_um: float = 1.0
    fill_lo: float = 0.15               # in-plane metal area fraction, lower (thin Cu, low-k) BEOL
    fill_hi: float = 0.30               # upper (thick Cu, TEOS) BEOL
    via_ratio: float = 0.10             # vertical via area fraction / in-plane fill
    keepout_um: float = 5.0             # BEOL metal keep-out radius beyond the ring radius
    # in-ring doped-Si heater
    heater_gap_um: float = 2.0
    heater_width_um: float = 1.0
    heater_doping_factor: float = 0.8   # k of 1e19-1e20 cm^-3 Si relative to undoped film (mrm.materials note)
    # silicide + undercut
    sil_gap_um: float = 1.5
    sil_width_um: float = 1.0
    undercut_margin_um: float = 4.0     # cavity radius = outer silicide edge + margin
    undercut_depth_um: float = 15.0
    air_k_w_per_m_k: float = AIR_K_W_PER_M_K
    # metal above
    metal_gap_um: float = 0.8           # Si top to heater bottom
    metal_width_um: float = 1.0
    metal_thick_um: float = 0.1
    metal: str = "tin_film"
    tbr_shell_um: float = 0.02
    # sites
    driver_radius_um: float = 5.0
    aggressor_radius_um: float = 10.0
    cmos_film_nm: float = 88.0
    pd_radius_um: float = 3.0
    sensor_radius_um: float = 2.0
    site_offsets_um: dict = field(default_factory=lambda: {"monitor_pd": [0.0, 25.0], "temp_sensor": [40.0, 0.0],
                                                           "driver": [0.0, -30.0], "aggressor": [60.0, 0.0]})
    pitch_um: float = 50.0
    # numerics
    mode_weight: str = "femwell"       # femwell | uniform_core
    mode_bent: bool = True

    # ------------------------------------------------------------------------------------------- derived
    @property
    def t_f(self) -> float:
        return self.film_nm * 1e-3

    @property
    def t_s(self) -> float:
        return self.slab_nm * 1e-3

    @property
    def w(self) -> float:
        return self.width_nm * 1e-3

    @property
    def die_radius_um(self) -> float:
        return self.die_side_mm * 1e3 / np.sqrt(np.pi)

    @property
    def z_bot(self) -> float:
        return -self.box_um - self.sub_um

    @property
    def z_beol0(self) -> float:
        return self.clad_um

    @property
    def z_beol1(self) -> float:
        return self.clad_um + self.beol_lo_um

    @property
    def z_beol2(self) -> float:
        return self.clad_um + self.beol_lo_um + self.beol_hi_um

    @property
    def z_top(self) -> float:
        return self.z_beol2 + self.pass_um

    def heater_rects(self):
        """[(r0, r1, z0, z1, material)] of the heater body (axisymmetric annuli)."""
        R, w, tf = self.radius_um, self.w, self.t_f
        if self.family == "inring_noundercut":
            r1 = R - w / 2 - self.heater_gap_um
            return [(r1 - self.heater_width_um, r1, 0.0, tf, "si_heater")]
        if self.family == "silicide_undercut":
            ri = R - w / 2 - self.sil_gap_um
            ro = R + w / 2 + self.sil_gap_um
            return [(ri - self.sil_width_um, ri, 0.0, tf, "si_heater"),
                    (ro, ro + self.sil_width_um, 0.0, tf, "si_heater")]
        if self.family == "metal_above":
            z0 = tf + self.metal_gap_um
            return [(R - self.metal_width_um / 2, R + self.metal_width_um / 2, z0, z0 + self.metal_thick_um, "metal")]
        if self.family == "imec_w_above":
            z0 = tf + self.metal_gap_um
            return [(R - self.metal_width_um / 2, R + self.metal_width_um / 2, z0, z0 + self.metal_thick_um, "metal")]
        raise ValueError(f"unknown family {self.family!r}")

    def slab_extents(self):
        """(inner, outer) slab extent from the rib edges; in-ring families extend the slab to the heater."""
        s_in, s_out = self.slab_in_um, self.slab_out_um
        if self.family == "inring_noundercut":
            s_in = max(s_in, self.heater_gap_um)
        if self.family == "silicide_undercut":
            s_in = max(s_in, self.sil_gap_um)
            s_out = max(s_out, self.sil_gap_um)
        return s_in, s_out

    def undercut_radius_um(self) -> float:
        return self.radius_um + self.w / 2 + self.sil_gap_um + self.sil_width_um + self.undercut_margin_um

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def default_config(family: str = "inring_noundercut", film_nm: float = 160.0, **kw) -> ThermalConfig:
    return ThermalConfig(family=family, film_nm=float(film_nm), **kw)


ASSUMPTIONS = {
    "radius_um": ("printed", "Lightmatter '~15 um diameter' (HC, LM26-S); GF PDK MRM R = 7.5 um (GF-ECIO)"),
    "film_nm": ("printed", "45SPCLO 160 nm photonics SOI (GF-IMAPS); 200 nm = SMTAI render (hypothesis D21)"),
    "box_um": ("printed", "45SPCLO 2 um BOX (GF-IMAPS)"),
    "die_side_mm": ("assumed", "5 x 5 mm = 45SPCLO MPW block (Europractice, printed); the G25/G26 die is not printed"),
    "width_nm": ("assumed", "rib width; BU 45CLO ring 410 nm"),
    "slab_nm": ("assumed", "partial-etch depth not public (01 4.2)"),
    "sub_um": ("assumed", "un-thinned 300 mm wafer"),
    "h_bottom_w_per_m2_k": ("assumed", "die attach + package to a TEC-held plate (R'' = 5e-5 m^2 K/W)"),
    "clad_um": ("assumed", "pre-metal dielectric thickness"),
    "beol_lo_um": ("assumed", "9-level BEOL printed, total thickness not public"),
    "beol_hi_um": ("assumed", "9-level BEOL printed, total thickness not public"),
    "fill_lo": ("assumed", "effective Cu area fraction of the thin levels"),
    "fill_hi": ("assumed", "effective Cu area fraction of the thick levels"),
    "via_ratio": ("assumed", "via area fraction relative to the line fill"),
    "keepout_um": ("assumed", "metal no-fill zone beyond the ring radius (imec: tiling costs up to 60 %)"),
    "heater_gap_um": ("assumed", "in-ring heater placement ('heater and PN inside the ring', LM26-S schematic)"),
    "heater_width_um": ("assumed", "in-ring heater width"),
    "sil_gap_um": ("assumed", "GF PDK: four silicide resistors, two inside, two outside (GF-ECIO); spacing assumed"),
    "undercut_depth_um": ("assumed", "isotropic Si undercut depth; imec UCUT ~27 x 45 um^2 (CO22-JLT)"),
    "metal_gap_um": ("assumed", "metal heater height above the film"),
    "metal_width_um": ("assumed", "metal heater width"),
    "pitch_um": ("assumed", "ring pitch on the TX bus not published (04 open question 2)"),
    "site_offsets_um": ("assumed", "PD near the drop port, sensor, monolithic driver, aggressor (G26: heater adjacent "
                                   "to the MRM)"),
}


# =============================================================================================== materials
def _film_k(t_k: float, thickness_nm: float) -> float:
    """In-plane k of a Si film; the mrm.materials fit is valid 150-300 nm, so thinner films (slab, CMOS film) are
    clamped to 150 nm (Ju & Goodson measured ~74 W/(m K) at 74 nm, above the fit's 45: the clamp is the better
    estimate; +/-25 % uncertainty recorded)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", mt.MaterialRangeWarning)
        return float(mt.k_si_film_w_per_m_k(t_k, max(float(thickness_nm), 150.0)))


def material_table(cfg: ThermalConfig, t_k: float | None = None) -> dict:
    """Region material -> (k_r, k_z, rho_c) in SI, from mrm.materials at temperature T (anisotropic BEOL)."""
    T = float(cfg.t_k if t_k is None else t_k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", mt.MaterialRangeWarning)
        k_film = _film_k(T, cfg.film_nm)
        k_slab = _film_k(T, cfg.slab_nm)
        k_cmos = _film_k(T, cfg.cmos_film_nm)
        k_sub = float(mt.k_si_bulk_w_per_m_k(T))
        k_box = float(mt.k_sio2_w_per_m_k(T, "thermal"))
        k_ox = float(mt.k_sio2_w_per_m_k(T, "pecvd"))
        k_lowk = float(mt.k_low_k_w_per_m_k(T, "dense"))
        k_cu = float(mt.k_metal_w_per_m_k("cu_beol", T))
        rc_si = float(mt.rho_c_j_per_m3_k("si", T))
        rc_ox = float(mt.rho_c_j_per_m3_k("sio2", T))
        rc_lowk = float(mt.rho_c_j_per_m3_k("low_k", T))
        rc_cu = float(mt.rho_c_j_per_m3_k("cu", T))
        metal = cfg.metal
        k_metal = float(mt.k_metal_w_per_m_k(metal, T))
        rc_metal = float(mt.rho_c_j_per_m3_k(metal, T))
        tbr = mt.thermal_boundary_resistance_m2_k_per_w("metal_dielectric")

    def beol(fill, k_d, rc_d):
        fv = cfg.via_ratio * fill
        kr = fill * k_cu + (1 - fill) * k_d                     # lines in plane: parallel
        kz = fv * k_cu + k_d / max(1 - fill, 1e-6)               # vias + dielectric gaps between levels
        rc = fill * rc_cu + (1 - fill) * rc_d
        return kr, kz, rc

    d = 1e-6 * cfg.tbr_shell_um
    k_shell = d / (tbr + d / k_ox)                               # shell = 20 nm oxide + the boundary resistance
    tab = {
        "si_sub": (k_sub, k_sub, rc_si),
        "box": (k_box, k_box, rc_ox),
        "clad": (k_ox, k_ox, rc_ox),
        "si_film": (k_film, k_film, rc_si),
        "si_slab": (k_slab, k_slab, rc_si),
        "si_heater": (k_film * cfg.heater_doping_factor, k_film * cfg.heater_doping_factor, rc_si),
        "si_cmos": (k_cmos, k_cmos, rc_si),
        "beol_lo": beol(cfg.fill_lo, k_lowk, rc_lowk),
        "beol_hi": beol(cfg.fill_hi, k_ox, rc_ox),
        "beol_lo_ko": (k_lowk, k_lowk, rc_lowk),
        "beol_hi_ko": (k_ox, k_ox, rc_ox),
        "pass": (k_ox, k_ox, rc_ox),
        "air": (cfg.air_k_w_per_m_k, cfg.air_k_w_per_m_k, AIR_RHO_C_J_PER_M3_K),
        "metal": (k_metal, k_metal, rc_metal),
        "tbr": (k_shell, k_shell, rc_ox),
    }
    return tab


# =============================================================================================== geometry
@dataclass
class Rect:
    name: str
    r0: float
    r1: float
    z0: float
    z1: float
    material: str
    priority: int = 0

    def contains(self, r, z):
        return (r >= self.r0) & (r <= self.r1) & (z >= self.z0) & (z <= self.z1)


def ring_geometry(cfg: ThermalConfig) -> list[Rect]:
    """Model A: one ring (rib, slabs, family heater, undercut) at r = R in the full stack."""
    R, w, tf, ts = cfg.radius_um, cfg.w, cfg.t_f, cfg.t_s
    Rd = cfg.die_radius_um
    rects = _stack_rects(cfg, keepout_r=R + cfg.keepout_um)
    s_in, s_out = cfg.slab_extents()
    rects += [Rect("slab_in", R - w / 2 - s_in, R - w / 2, 0.0, ts, "si_slab", 2),
              Rect("slab_out", R + w / 2, R + w / 2 + s_out, 0.0, ts, "si_slab", 2),
              Rect("rib", R - w / 2, R + w / 2, 0.0, tf, "si_film", 3)]
    for i, (r0, r1, z0, z1, m) in enumerate(cfg.heater_rects()):
        rects.append(Rect(f"heater{i}", r0, r1, z0, z1, m, 5))
        if m == "metal" and cfg.tbr_shell_um > 0:
            d = cfg.tbr_shell_um
            rects.append(Rect(f"tbr{i}", r0 - d, r1 + d, z0 - d, z1 + d, "tbr", 4))
    if cfg.family == "silicide_undercut":
        rects.append(Rect("undercut", 0.0, cfg.undercut_radius_um(), -cfg.box_um - cfg.undercut_depth_um, -cfg.box_um,
                          "air", 6))
    assert all(r.r1 <= Rd for r in rects)
    return rects


def site_geometry(cfg: ThermalConfig) -> list[Rect]:
    """Model B: the plain stack with the driver (inner disk) and aggressor (outer annulus + inner disk) at r = 0."""
    tc = cfg.cmos_film_nm * 1e-3
    rects = _stack_rects(cfg, keepout_r=None)
    rects += [Rect("drv", 0.0, cfg.driver_radius_um, 0.0, tc, "si_cmos", 2),
              Rect("agg", cfg.driver_radius_um, cfg.aggressor_radius_um, 0.0, tc, "si_cmos", 2)]
    return rects


def _stack_rects(cfg: ThermalConfig, keepout_r) -> list[Rect]:
    Rd = cfg.die_radius_um
    rects = [Rect("sub", 0.0, Rd, cfg.z_bot, -cfg.box_um, "si_sub"),
             Rect("box", 0.0, Rd, -cfg.box_um, 0.0, "box"),
             Rect("clad", 0.0, Rd, 0.0, cfg.z_beol0, "clad"),
             Rect("beol_lo", 0.0, Rd, cfg.z_beol0, cfg.z_beol1, "beol_lo"),
             Rect("beol_hi", 0.0, Rd, cfg.z_beol1, cfg.z_beol2, "beol_hi"),
             Rect("pass", 0.0, Rd, cfg.z_beol2, cfg.z_top, "pass")]
    if keepout_r is not None and cfg.keepout_um > 0:
        rects += [Rect("ko_lo", 0.0, keepout_r, cfg.z_beol0, cfg.z_beol1, "beol_lo_ko", 1),
                  Rect("ko_hi", 0.0, keepout_r, cfg.z_beol1, cfg.z_beol2, "beol_hi_ko", 1)]
    return rects


def classify(rects: list[Rect], r, z) -> np.ndarray:
    """Index of the highest-priority rect containing each point (r, z); -1 if none."""
    r = np.asarray(r)
    z = np.asarray(z)
    out = np.full(r.shape, -1, int)
    best = np.full(r.shape, -10 ** 9)
    for i, rc in enumerate(rects):
        m = rc.contains(r, z) & (rc.priority >= best)
        out[m] = i
        best[m] = rc.priority
    return out


# =============================================================================================== meshing
@dataclass
class MeshSpec:
    """Element-size controls (um). ``scale`` multiplies every local size (mesh-convergence levels)."""
    h_core: float = 0.050
    h_film: float = 0.040
    h_shell: float = 0.020
    h_near: float = 0.25
    growth: float = 0.20
    h_max: float = 150.0
    scale: float = 1.0
    centre: tuple = (7.5, 0.08)
    boxes: list = field(default_factory=list)       # [(r0, r1, z0, z1, h)]


def mesh_spec_for(cfg: ThermalConfig, rects: list[Rect], scale: float, model: str) -> MeshSpec:
    """Size boxes around the rib, the film structures, the heater (and its shell) and the site disks."""
    spec = MeshSpec(scale=scale)
    tf = cfg.t_f
    if model == "ring":
        R, w = cfg.radius_um, cfg.w
        spec.centre = (R, tf / 2)
        spec.boxes.append((R - w / 2 - 0.25, R + w / 2 + 0.25, -0.15, tf + 0.15, spec.h_core))
        film_r = [rc for rc in rects if rc.material in ("si_slab", "si_film", "si_heater")]
        r0 = min(rc.r0 for rc in film_r)
        r1 = max(rc.r1 for rc in film_r)
        h_film = min(spec.h_film, cfg.t_s / 1.5)
        spec.boxes.append((r0 - 0.2, r1 + 0.2, -0.1, tf + 0.1, h_film))
        for rc in rects:
            if rc.material == "tbr":
                spec.boxes.append((rc.r0 - 0.05, rc.r1 + 0.05, rc.z0 - 0.05, rc.z1 + 0.05, spec.h_shell))
            if rc.material == "metal":
                spec.boxes.append((rc.r0, rc.r1, rc.z0, rc.z1, min(spec.h_shell * 2, rc.z1 - rc.z0)))
            if rc.material == "air":
                spec.boxes.append((rc.r1 - 1.0, rc.r1 + 1.0, rc.z0 - 1.0, rc.z1 + 0.5, 0.25))
    else:
        tc = cfg.cmos_film_nm * 1e-3
        spec.centre = (0.0, tc / 2)
        spec.boxes.append((0.0, cfg.aggressor_radius_um + 0.5, -0.1, tc + 0.1, min(spec.h_film, tc / 1.5) * 1.5))
    # thin-layer band (BOX, cladding, BEOL) out to the last cross-term distance: keep elements near-isotropic
    r_band = min(cfg.die_radius_um, max(CROSS_R_UM) + 100.0)
    spec.boxes.append((0.0, r_band, -cfg.box_um - 0.3, cfg.z_top, 2.0))
    return spec


def mesh_axisym(rects: list[Rect], spec: MeshSpec, verbose: bool = False):
    """Conforming triangle mesh of the union of the rects (gmsh OCC fragment); returns (p (2, N), t (3, E), region)."""
    import gmsh

    gmsh.initialize(["-nt", "1"] if not verbose else [])
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("axisym")
        occ = gmsh.model.occ
        tags = [occ.addRectangle(rc.r0, rc.z0, 0.0, rc.r1 - rc.r0, rc.z1 - rc.z0) for rc in rects]
        occ.fragment([(2, tags[0])], [(2, t) for t in tags[1:]])
        occ.removeAllDuplicates()
        occ.synchronize()
        s = spec.scale
        fl = gmsh.model.mesh.field
        fids = []
        rc0, zc0 = spec.centre
        f = fl.add("MathEval")
        fl.setString(f, "F", f"{spec.h_near * s} + {spec.growth} * Sqrt((x-{rc0})^2 + (y-{zc0})^2)")
        fids.append(f)
        for (r0, r1, z0, z1, h) in spec.boxes:
            f = fl.add("Box")
            fl.setNumber(f, "VIn", h * s)
            fl.setNumber(f, "VOut", 1e9)
            fl.setNumber(f, "XMin", r0)
            fl.setNumber(f, "XMax", r1)
            fl.setNumber(f, "YMin", z0)
            fl.setNumber(f, "YMax", z1)
            fl.setNumber(f, "Thickness", max(10 * h * s, 0.3))
            fids.append(f)
        fmin = fl.add("Min")
        fl.setNumbers(fmin, "FieldsList", fids)
        fl.setAsBackgroundMesh(fmin)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", spec.h_max)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 1e-4)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        ntags, coords, _ = gmsh.model.mesh.getNodes()
        etypes, _, enodes = gmsh.model.mesh.getElements(2)
    finally:
        gmsh.finalize()
    tri = None
    for et, en in zip(etypes, enodes):
        if et == 2:
            tri = np.asarray(en, np.int64).reshape(-1, 3)
    if tri is None:
        raise RuntimeError("gmsh produced no triangles")
    idx = np.full(int(ntags.max()) + 1, -1, np.int64)
    used = np.unique(tri)
    idx[used] = np.arange(used.size)
    xyz = np.asarray(coords).reshape(-1, 3)
    pos = np.full(int(ntags.max()) + 1, -1, np.int64)
    pos[np.asarray(ntags, np.int64)] = np.arange(ntags.size)
    p = xyz[pos[used], :2].T.copy()
    t = idx[tri].T.copy()
    # counter-clockwise orientation
    a = p[:, t[1]] - p[:, t[0]]
    b = p[:, t[2]] - p[:, t[0]]
    neg = (a[0] * b[1] - a[1] * b[0]) < 0
    t[1, neg], t[2, neg] = t[2, neg].copy(), t[1, neg].copy()
    cen = p[:, t].mean(axis=1)
    region = classify(rects, cen[0], cen[1])
    if np.any(region < 0):
        raise RuntimeError("unclassified elements")
    return p, t, region


# =============================================================================================== axisymmetric FEM
class AxisymFEM:
    """P2 axisymmetric heat-conduction model: K (conductance, W/K), M (capacity, J/K) on the free DOFs.

    ``bc``: 'robin' (contact conductance h at the substrate bottom) or 'dirichlet' (T = 0 there). Every other boundary
    is adiabatic (the axis r = 0 by symmetry, the die edge, the top surface)."""

    def __init__(self, p, t, region, rects: list[Rect], mats: dict, z_bot: float, bc: str = "robin",
                 h_bottom: float = 2e4, order: int = 2):
        from skfem import Basis, ElementTriP1, ElementTriP2, FacetBasis, MeshTri, BilinearForm
        from skfem.helpers import dot, grad  # noqa: F401

        self.mesh = MeshTri(np.ascontiguousarray(p), np.ascontiguousarray(t))
        self.region = np.asarray(region)
        self.rects = rects
        self.region_names = [r.name for r in rects]
        self.mats = mats
        elem = ElementTriP2() if order == 2 else ElementTriP1()
        self.basis = Basis(self.mesh, elem)
        nq = self.basis.quadrature[0].shape[1]
        kr = np.array([mats[rects[i].material][0] for i in self.region])
        kz = np.array([mats[rects[i].material][1] for i in self.region])
        rc = np.array([mats[rects[i].material][2] for i in self.region])
        rep = lambda a: np.repeat(a[:, None], nq, axis=1)  # noqa: E731

        @BilinearForm
        def stiff(u, v, w):
            return 2 * np.pi * w.x[0] * (w.kr * u.grad[0] * v.grad[0] + w.kz * u.grad[1] * v.grad[1])

        @BilinearForm
        def mass(u, v, w):
            return 2 * np.pi * w.x[0] * w.rc * u * v

        K = stiff.assemble(self.basis, kr=rep(kr) * 1e-6, kz=rep(kz) * 1e-6)
        M = mass.assemble(self.basis, rc=rep(rc) * 1e-18)
        bottom = self.mesh.facets_satisfying(lambda x: np.isclose(x[1], z_bot, atol=1e-9))
        self.bc = bc
        if bc == "robin":
            fb = FacetBasis(self.mesh, elem, facets=bottom)

            @BilinearForm
            def robin(u, v, w):
                return 2 * np.pi * w.x[0] * h_bottom * 1e-12 * u * v

            K = K + robin.assemble(fb)
            self.free = np.arange(self.basis.N)
        elif bc == "dirichlet":
            D = self.basis.get_dofs(facets=bottom).all()
            self.free = np.setdiff1d(np.arange(self.basis.N), D)
        else:
            raise ValueError("bc must be robin or dirichlet")
        self.K = K.tocsr()[self.free][:, self.free].tocsc()
        self.M = M.tocsr()[self.free][:, self.free].tocsc()
        self.n = self.free.size
        self._lu0 = None

    # ------------------------------------------------------------------------------------------- functionals
    def _elem_weights(self, regions: dict) -> np.ndarray:
        """Element-wise density from {region-name: density}."""
        dens = np.zeros(self.mesh.nelements)
        for name, d in regions.items():
            ids = [i for i, n in enumerate(self.region_names) if n == name]
            if not ids:
                raise KeyError(f"no region {name!r}")
            dens[np.isin(self.region, ids)] = d
        return dens

    def region_vector(self, regions: dict) -> np.ndarray:
        """Load vector of a volumetric density over regions, normalized so that sum(b) = 1 (a 1 W source and, divided
        by the same total, the volume-average observer). Returns the full-DOF vector."""
        from skfem import LinearForm

        nq = self.basis.quadrature[0].shape[1]
        dens = np.repeat(self._elem_weights(regions)[:, None], nq, axis=1)

        @LinearForm
        def lf(v, w):
            return 2 * np.pi * w.x[0] * w.d * v

        b = lf.assemble(self.basis, d=dens)
        return b / b.sum()

    def point_vector(self, pts, wts) -> np.ndarray:
        """sum_p w_p phi_i(x_p) normalized to sum 1 (points (2, N) in (r, z) um, positive weights)."""
        pts = np.asarray(pts, float)
        w = np.asarray(wts, float)
        w = w / w.sum()
        v = np.zeros(self.basis.N)
        for i in range(0, pts.shape[1], 256):      # skfem's locator is O(points x candidates): bound its memory
            P = self.basis.probes(pts[:, i:i + 256]).tocsr()
            v += np.asarray(P.T @ w[i:i + 256]).ravel()
        return v

    def restrict(self, V) -> np.ndarray:
        """Full-DOF vectors (N,) or (N, m) -> free DOFs."""
        V = np.asarray(V)
        return V[self.free] if V.ndim == 1 else V[self.free, :]

    # ------------------------------------------------------------------------------------------- solvers
    def dc(self, B, C) -> np.ndarray:
        """Exact DC gains C K^-1 B (O, S) in K/W; B (n, S), C (O, n) on free DOFs."""
        if self._lu0 is None:
            self._lu0 = spla.splu(self.K, permc_spec=PERMC)
        X = self._lu0.solve(np.asarray(B, float))
        return np.asarray(C @ X)

    def release(self) -> None:
        """Drop the cached DC factorization (memory)."""
        self._lu0 = None

    def freq(self, B, C, f_hz, log_fn=None) -> np.ndarray:
        """Exact frequency response C (K + j w M)^-1 B at f, (O, S, F) complex (one sparse LU per frequency)."""
        B = np.asarray(B, complex)
        out = np.zeros((C.shape[0], B.shape[1], len(f_hz)), complex)
        for i, f in enumerate(f_hz):
            if f == 0:
                out[:, :, i] = self.dc(B.real, C)
                continue
            lu = spla.splu((self.K + (2j * np.pi * f) * self.M).tocsc(), permc_spec=PERMC)
            out[:, :, i] = C @ lu.solve(B)
            del lu
        return out

    def step(self, B, C, dt0: float = 0.25e-9, n_per_level: int = 40, t_end: float = 2e-2, log_fn=None):
        """Full-FEM unit-step responses with TR-BDF2 (gamma = 2 - sqrt 2, L-stable, second order), dt doubling every
        ``n_per_level`` steps. Returns (t (T,), y (O, S, T)), with y(0) = 0 prepended."""
        g = 2.0 - np.sqrt(2.0)
        a = g / 2.0
        c1 = 1.0 / (g * (2.0 - g))
        c2 = (1.0 - g) ** 2 / (g * (2.0 - g))
        B = np.asarray(B, float)
        T = np.zeros_like(B)
        ts, ys = [0.0], [np.zeros((C.shape[0], B.shape[1]))]
        t, dt = 0.0, dt0
        K, M = self.K, self.M
        lu = None
        while t < t_end * (1 - 1e-12):
            del lu
            lu = spla.splu((M + (a * dt) * K).tocsc(), permc_spec=PERMC)
            for _ in range(n_per_level):
                MT = M @ T
                Tg = lu.solve(MT - (a * dt) * (K @ T) + (g * dt) * B)
                T = lu.solve(c1 * (M @ Tg) - c2 * MT + (a * dt) * B)
                t += dt
                ts.append(t)
                ys.append(np.asarray(C @ T))
            if log_fn:
                log_fn(f"  step: t = {t:.3e} s (dt {dt:.2e})")
            dt *= 2.0
        return np.array(ts), np.moveaxis(np.array(ys), 0, -1)

    def krylov_modal(self, B, C, f_lo: float = 1e-3, f_hi: float = 3e9, per_decade: float = 3.0,
                     n_moments: int = 2, log_fn=None) -> mor.Foster:
        """Rational-Krylov Galerkin ROM of the pencil, in modal Foster form with res (O, S, r)."""
        shifts = mor.log_shifts(f_lo, f_hi, per_decade)
        V = mor.rational_krylov_basis(self.K, self.M, np.asarray(B, float), shifts, n_moments=n_moments)
        fm = mor.galerkin_modal(self.K, self.M, B, C, V)
        fm.meta.update({"shift_f_lo_hz": f_lo, "shift_f_hi_hz": f_hi, "shifts_per_decade": per_decade,
                        "n_moments": n_moments, "n_shifts": int(shifts.size)})
        return fm


# =============================================================================================== mode weight
def film_structures(cfg: ThermalConfig):
    """Si film cross-section pieces relative to the rib centre: [(x0, x1, y0, y1, name)] with x = r - R, y = z."""
    w, tf, ts = cfg.w, cfg.t_f, cfg.t_s
    s_in, s_out = cfg.slab_extents()
    out = [(-w / 2, w / 2, 0.0, tf, "core"),
           (-w / 2 - s_in, -w / 2, 0.0, ts, "slab_in"),
           (w / 2, w / 2 + s_out, 0.0, ts, "slab_out")]
    for i, (r0, r1, z0, z1, m) in enumerate(cfg.heater_rects()):
        if m == "si_heater":
            out.append((r0 - cfg.radius_um, r1 - cfg.radius_um, z0, z1, f"heater{i}"))
    return out


@dataclass
class ModeWeight:
    """Point weights of the TE0 mode on the ring cross-section, in (r, z) um.

    ``w_obs`` = n * dn/dT * |E|^2 dA (perturbation weight of the resonance), ``w_src`` = |E|^2 dA in Si (free-carrier
    absorption density). Both are unnormalized; the FEM normalizes them (with the 2 pi r volume factor)."""
    r: np.ndarray
    z: np.ndarray
    w_obs: np.ndarray
    w_src: np.ndarray
    info: dict

    def coarse(self, h_um: float) -> "ModeWeight":
        """Re-bin onto an h x h grid (weighted centroids); used for the victim annulus of the cross terms."""
        return _bin_points(self.r, self.z, self.w_obs, self.w_src, h_um, dict(self.info, bin_um=h_um))


def _bin_points(r, z, wo, ws, h, info):
    ir = np.floor((r - r.min()) / h).astype(int)
    iz = np.floor((z - z.min()) / h).astype(int)
    key = ir * (iz.max() + 1) + iz
    uniq, inv = np.unique(key, return_inverse=True)
    tot = np.bincount(inv, weights=wo + ws)
    keep = tot > 1e-7 * tot.max()
    wsum = np.maximum(tot, 1e-300)
    rr = np.bincount(inv, weights=r * (wo + ws)) / wsum
    zz = np.bincount(inv, weights=z * (wo + ws)) / wsum
    wo_b = np.bincount(inv, weights=wo)
    ws_b = np.bincount(inv, weights=ws)
    return ModeWeight(rr[keep], zz[keep], wo_b[keep], ws_b[keep], info)


def mode_weight(cfg: ThermalConfig, res_um: float = 0.02, bin_um: float = 0.02) -> ModeWeight:
    """TE0 point weights; the femwell solve runs in a short-lived subprocess so that its memory peak (~0.6 GB) is
    returned to the OS before the FEM work starts (16 GB machine shared with other jobs)."""
    if cfg.mode_weight == "uniform_core":
        return _mode_weight_impl(cfg, res_um, bin_um)
    import concurrent.futures as cf
    import multiprocessing as mp
    with cf.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        r, z, wo, ws, info = ex.submit(_mode_weight_arrays, cfg, res_um, bin_um).result()
    return ModeWeight(r, z, wo, ws, info)


def _mode_weight_arrays(cfg, res_um, bin_um):
    mw = _mode_weight_impl(cfg, res_um, bin_um)
    return mw.r, mw.z, mw.w_obs, mw.w_src, mw.info


def _mode_weight_impl(cfg: ThermalConfig, res_um: float = 0.02, bin_um: float = 0.02) -> ModeWeight:
    """TE0 mode of the ring cross-section from femwell (bent at R with femwell's conformal map unless
    ``cfg.mode_bent`` is False), turned into point weights. ``cfg.mode_weight = 'uniform_core'`` gives the
    analytic fallback: uniform weight over the rib (used by --quick and as the sensitivity case)."""
    R, tf = cfg.radius_um, cfg.t_f
    if cfg.mode_weight == "uniform_core":
        w = cfg.w
        nx, nz = 21, 9
        xs = (np.arange(nx) + 0.5) / nx * w - w / 2
        zs = (np.arange(nz) + 0.5) / nz * tf
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        one = np.full(X.size, 1.0 / X.size)
        return ModeWeight(R + X.ravel(), Z.ravel(), one, one.copy(), {"method": "uniform_core"})
    from collections import OrderedDict

    import shapely
    from femwell.maxwell.waveguide import compute_modes
    from femwell.mesh import mesh_from_OrderedDict
    from skfem import Basis, ElementTriP0
    from skfem.io.meshio import from_meshio

    lam_um = cfg.lam_m * 1e6
    T = cfg.t_k
    n_si = float(mt.n_si(cfg.lam_m, T))
    n_ox = float(mt.n_sio2(cfg.lam_m, T, "thermal"))
    dn_si = float(mt.dn_dt_si_per_k(cfg.lam_m, T))
    dn_ox = float(mt.dn_dt_sio2_per_k(T, "thermal"))
    # the optical cross-section is the rib and its slabs; doped heater strips (>= 1.5 um away, outside the evanescent
    # field) are left out because a full-thickness strip guides its own, higher-index mode
    pieces = [pc for pc in film_structures(cfg) if not pc[4].startswith("heater")]
    x_lo = min(p[0] for p in pieces) - 1.0
    x_hi = max(p[1] for p in pieces) + 1.5
    polys = OrderedDict()
    for (x0, x1, y0, y1, name) in pieces:
        polys[name] = shapely.box(x0, y0, x1, y1)
    polys["clad"] = shapely.box(x_lo, -1.0, x_hi, tf + 1.0)
    resd = {k: {"resolution": res_um if k == "core" else 2 * res_um, "distance": 0.5} for k in polys if k != "clad"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = from_meshio(mesh_from_OrderedDict(polys, resd, default_resolution_max=0.2))
        b0 = Basis(m, ElementTriP0())
        eps = b0.zeros() + n_ox ** 2
        si_names = [p[4] for p in pieces]
        for nm in si_names:
            eps[b0.get_dofs(elements=nm)] = n_si ** 2
        radius = R if cfg.mode_bent else np.inf
        modes = compute_modes(b0, eps, wavelength=lam_um, num_modes=6, order=2, radius=radius)
        te = [md for md in modes if md.te_fraction > 0.5]
        pc = [float(np.real(md.calculate_power(elements="core"))) for md in te]
    md = te[int(np.argmax(pc))]
    if max(pc) < 0.3:
        raise RuntimeError(f"no TE mode confined in the rib (core power fractions {pc})")
    (Et, Ez) = md.basis.interpolate(md.E)
    E2 = np.abs(Et.value[0]) ** 2 + np.abs(Et.value[1]) ** 2 + np.abs(Ez.value) ** 2
    X = md.basis.global_coordinates().value
    dx = md.basis.dx
    eps_q = md.basis_epsilon_r.interpolate(md.epsilon_r).value.real
    is_si = eps_q > (n_si ** 2 + n_ox ** 2) / 2
    n_q = np.sqrt(eps_q)
    dndt = np.where(is_si, dn_si, dn_ox)
    w_obs = (n_q * dndt * E2 * dx).ravel()
    w_src = (E2 * dx * is_si).ravel()
    p_core = float(np.real(md.calculate_power(elements="core")))
    info = {"method": "femwell", "bent": bool(cfg.mode_bent), "radius_um": R if cfg.mode_bent else None,
            "n_eff": float(md.n_eff.real), "te_fraction": float(md.te_fraction), "power_fraction_core": p_core,
            "mesh_res_um": res_um, "bin_um": bin_um, "lambda_um": lam_um, "n_si": n_si, "n_sio2": n_ox,
            "si_obs_weight_fraction": float(w_obs[is_si.ravel()].sum() / w_obs.sum())}
    r = R + X[0].ravel()
    z = X[1].ravel()
    return _bin_points(r, z, w_obs, w_src, bin_um, info)


# =============================================================================================== observers at distance
def _annulus_points(r_p, z_p, w_p, d: float, nphi: int = 24):
    """Points of a victim annulus (cross-section points r_p, z_p with weights w_p, about the victim centre) whose
    centre sits at distance d from the axis; returns axisymmetric coordinates (rho, z) and volume weights."""
    phi = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
    rho = np.sqrt(d ** 2 + r_p[:, None] ** 2 + 2 * d * r_p[:, None] * np.cos(phi[None, :]))
    zz = np.broadcast_to(z_p[:, None], rho.shape)
    ww = np.broadcast_to((w_p * r_p)[:, None], rho.shape)
    return np.vstack([rho.ravel(), zz.ravel()]), ww.ravel()


def _heater_section_points(cfg: ThermalConfig, nr: int = 5, nz: int = 3):
    rs, zs, ws = [], [], []
    for (r0, r1, z0, z1, _m) in cfg.heater_rects():
        rr = r0 + (np.arange(nr) + 0.5) / nr * (r1 - r0)
        zz = z0 + (np.arange(nz) + 0.5) / nz * (z1 - z0)
        R_, Z_ = np.meshgrid(rr, zz, indexing="ij")
        rs.append(R_.ravel())
        zs.append(Z_.ravel())
        ws.append(np.full(R_.size, (r1 - r0) * (z1 - z0) / R_.size))
    return np.concatenate(rs), np.concatenate(zs), np.concatenate(ws)


def _disk_points(d: float, a: float, z: float, nr: int = 4, nth: int = 16):
    """Disk of radius a centred at distance d (film level z): axisymmetric coordinates and area weights."""
    rr = (np.arange(nr) + 0.5) / nr * a
    th = (np.arange(nth) + 0.5) * 2 * np.pi / nth
    R_, T_ = np.meshgrid(rr, th, indexing="ij")
    x = d + R_ * np.cos(T_)
    y = R_ * np.sin(T_)
    rho = np.sqrt(x ** 2 + y ** 2)
    return np.vstack([rho.ravel(), np.full(rho.size, z)]), R_.ravel()


def observer_points(cfg: ThermalConfig, obs: str, d: float, mw_c: ModeWeight):
    """Point set (pts, wts) of observer ``obs`` whose centre is at distance d from the source axis."""
    if obs == "ring_mode_weighted":
        return _annulus_points(mw_c.r, mw_c.z, mw_c.w_obs, d)
    if obs == "heater_resistor":
        r, z, w = _heater_section_points(cfg)
        return _annulus_points(r, z, w, d)
    if obs == "monitor_pd":
        return _disk_points(d, cfg.pd_radius_um, cfg.t_f / 2)
    if obs == "temp_sensor":
        return _disk_points(d, cfg.sensor_radius_um, cfg.t_f / 2)
    raise KeyError(obs)


def _site_dist(cfg: ThermalConfig, a: str, b: str | None) -> float:
    oa = np.array(cfg.site_offsets_um[a], float)
    ob = np.zeros(2) if b is None else np.array(cfg.site_offsets_um[b], float)
    return float(np.hypot(*(oa - ob)))


# =============================================================================================== model builders
@dataclass
class ModelSet:
    """One FEM with its input columns (sources) and output rows (observers, each with a key)."""
    name: str
    fem: AxisymFEM
    sources: list
    B: np.ndarray
    keys: list
    C: np.ndarray
    info: dict

    def row(self, key) -> int:
        return self.keys.index(key)


def build_ring_model(cfg: ThermalConfig, scale: float, mw: ModeWeight, mw_c: ModeWeight,
                     cross_r=CROSS_R_UM, t_k: float | None = None, bc: str | None = None) -> ModelSet:
    """Model A: one ring at the origin with sources heater, waveguide_absorption, junction."""
    rects = ring_geometry(cfg)
    # junction heat: the slab within contact_offset of the rib edges (same material, split regions)
    R, w, ts, c = cfg.radius_um, cfg.w, cfg.t_s, cfg.contact_offset_um
    rects += [Rect("jslab_in", R - w / 2 - c, R - w / 2, 0.0, ts, "si_slab", 2),
              Rect("jslab_out", R + w / 2, R + w / 2 + c, 0.0, ts, "si_slab", 2)]
    spec = mesh_spec_for(cfg, rects, scale, "ring")
    p, t, reg = mesh_axisym(rects, spec)
    fem = AxisymFEM(p, t, reg, rects, material_table(cfg, t_k), cfg.z_bot, bc or cfg.bottom_bc,
                    cfg.h_bottom_w_per_m2_k)
    heater_regions = {r.name: 1.0 for r in rects if r.name.startswith("heater")}
    b_heater = fem.region_vector(heater_regions)
    b_wg = fem.point_vector(np.vstack([mw.r, mw.z]), mw.w_src * mw.r)
    b_j = fem.region_vector({"jslab_in": 1.0 / cfg.t_s ** 2, "jslab_out": 1.0 / cfg.t_s ** 2,
                             "rib": 1.0 / cfg.t_f ** 2})
    B = np.column_stack([fem.restrict(b) for b in (b_heater, b_wg, b_j)])
    keys, rows = [], []
    rows.append(fem.point_vector(np.vstack([mw.r, mw.z]), mw.w_obs * mw.r))
    keys.append(("self", "ring_mode_weighted"))
    rows.append(fem.region_vector(heater_regions))
    keys.append(("self", "heater_resistor"))
    for o in ("monitor_pd", "temp_sensor"):
        rows.append(fem.point_vector(*observer_points(cfg, o, _site_dist(cfg, o, None), mw_c)))
        keys.append(("self", o))
    for d in cross_r:
        for o in OBSERVERS:
            rows.append(fem.point_vector(*observer_points(cfg, o, d, mw_c)))
            keys.append(("cross", o, float(d)))
    C = np.vstack([fem.restrict(r) for r in rows])
    info = {"nodes": int(fem.mesh.nvertices), "elements": int(fem.mesh.nelements), "dofs": int(fem.n),
            "scale": scale, "h_core_um": spec.h_core * scale, "element": "P2 triangle, axisymmetric"}
    return ModelSet("ring", fem, list(RING_SOURCES), B, keys, C, info)


def build_site_model(cfg: ThermalConfig, scale: float, mw_c: ModeWeight, cross_r=CROSS_R_UM,
                     t_k: float | None = None, bc: str | None = None) -> ModelSet:
    """Model B: driver and aggressor disks at the origin in the plain stack."""
    rects = site_geometry(cfg)
    spec = mesh_spec_for(cfg, rects, scale, "site")
    p, t, reg = mesh_axisym(rects, spec)
    fem = AxisymFEM(p, t, reg, rects, material_table(cfg, t_k), cfg.z_bot, bc or cfg.bottom_bc,
                    cfg.h_bottom_w_per_m2_k)
    b_drv = fem.region_vector({"drv": 1.0})
    b_agg = fem.region_vector({"drv": 1.0, "agg": 1.0})
    B = np.column_stack([fem.restrict(b_drv), fem.restrict(b_agg)])
    keys, rows = [], []
    for src in SITE_SOURCES:
        for o in OBSERVERS:
            other = None if o in ("ring_mode_weighted", "heater_resistor") else o
            d = _site_dist(cfg, src, other)
            rows.append(fem.point_vector(*observer_points(cfg, o, d, mw_c)))
            keys.append(("self", src, o))
    for d in cross_r:
        for o in OBSERVERS:
            rows.append(fem.point_vector(*observer_points(cfg, o, d, mw_c)))
            keys.append(("cross", o, float(d)))
    C = np.vstack([fem.restrict(r) for r in rows])
    info = {"nodes": int(fem.mesh.nvertices), "elements": int(fem.mesh.nelements), "dofs": int(fem.n),
            "scale": scale, "element": "P2 triangle, axisymmetric"}
    return ModelSet("site", fem, list(SITE_SOURCES), B, keys, C, info)


def pair_list(ms: ModelSet):
    """All (source index s in SOURCES, observer index o in OBSERVERS, d or None, row, col) of a model."""
    out = []
    for j, src in enumerate(ms.sources):
        s = SOURCES.index(src)
        for i, k in enumerate(ms.keys):
            if k[0] == "self":
                if ms.name == "site" and k[1] != src:
                    continue
                o = OBSERVERS.index(k[-1])
                out.append((s, o, None, i, j))
            else:
                out.append((s, OBSERVERS.index(k[1]), k[2], i, j))
    return out


# =============================================================================================== fit + verification
F_FIT_HZ = np.logspace(-3, 9.5, 126)
T_FIT_S = np.logspace(-10, 1, 111)
F_VER_HZ = np.array([0.0, 0.1, 1.0, 10.0, 100.0, 1e3, 3e3, 1e4, 3e4, 1e5, 2e5, 5e5, 1e6, 3e6, 1e7, 1e8])
NN_FLOOR = 1e-3        # cross terms: frequency check only where |H| >= 1e-3 |H(0)| (-60 dB)
# residue conditioning bounds, max_k |r_k| / |H(0)|: self pairs are nearly driving-point (little cancellation needed);
# non-collocated cross pairs carry a diffusion delay that real poles can only build from mixed-sign residues
CANCEL_SELF, CANCEL_CROSS = 8.0, 25.0


@dataclass
class FitResult:
    taus: np.ndarray
    self_res: np.ndarray
    cross_res: np.ndarray
    dc_self: np.ndarray
    dc_cross: np.ndarray
    cross_r: np.ndarray
    validation: dict
    fit_info: dict


def _responses(models: list, roms: dict, dcs: dict):
    """Flatten every (source, observer, distance) pair of every model into fit arrays."""
    meta, Hf, Yt, dc = [], [], [], []
    for ms in models:
        fm = roms[ms.name]
        H = fm.freq(F_FIT_HZ)
        Y = fm.step(T_FIT_S)
        for (s, o, d, i, j) in pair_list(ms):
            meta.append((ms.name, s, o, d, i, j))
            Hf.append(H[i, j])
            Yt.append(Y[i, j])
            dc.append(dcs[ms.name][i, j])
    return meta, np.array(Hf), np.array(Yt), np.array(dc)


def build_roms(models: list, per_decade: float = 3.0):
    """Krylov ROM and exact DC of each model (the ROM supplies the dense fit data)."""
    roms, dcs, rom_info = {}, {}, {}
    for ms in models:
        t0 = time.time()
        roms[ms.name] = ms.fem.krylov_modal(ms.B, ms.C, per_decade=per_decade)
        dcs[ms.name] = ms.fem.dc(ms.B, ms.C)
        ms.fem.release()
        rom_dc = roms[ms.name].dc
        rom_info[ms.name] = dict(roms[ms.name].meta, seconds=round(time.time() - t0, 1),
                                 dc_rel_err_max=float(np.max(np.abs(rom_dc - dcs[ms.name]) / np.abs(dcs[ms.name]))))
        log(f"ROM {ms.name}: order {roms[ms.name].taus.size}, {rom_info[ms.name]['seconds']} s")
    return roms, dcs, rom_info


def fit_models(models: list, n_poles: int = 20, tau_min: float = 3e-9, tau_max: float = 3.0, per_decade: float = 3.0,
               polish: bool = True, cross_r=CROSS_R_UM, polish_d=(20.0, 50.0, 100.0, 300.0, 1000.0),
               min_ratio: float = 1.5, roms=None) -> FitResult:
    """Krylov ROM of each model (or the given ``roms = build_roms(...)``), then one common-pole Foster fit over every
    pair of every model."""
    if roms is not None:
        roms, dcs, rom_info = roms
        meta, Hf, Yt, dc = _responses(models, roms, dcs)
        return _fit_from_responses(models, roms, dcs, rom_info, meta, Hf, Yt, dc, n_poles, tau_min, tau_max, polish,
                                   cross_r, polish_d, min_ratio)
    roms, dcs, rom_info = {}, {}, {}
    for ms in models:
        t0 = time.time()
        roms[ms.name] = ms.fem.krylov_modal(ms.B, ms.C, per_decade=per_decade)
        dcs[ms.name] = ms.fem.dc(ms.B, ms.C)
        ms.fem.release()
        rom_dc = roms[ms.name].dc
        rom_info[ms.name] = dict(roms[ms.name].meta, seconds=round(time.time() - t0, 1),
                                 dc_rel_err_max=float(np.max(np.abs(rom_dc - dcs[ms.name]) / np.abs(dcs[ms.name]))))
        log(f"ROM {ms.name}: order {roms[ms.name].taus.size}, {rom_info[ms.name]['seconds']} s")
    meta, Hf, Yt, dc = _responses(models, roms, dcs)
    return _fit_from_responses(models, roms, dcs, rom_info, meta, Hf, Yt, dc, n_poles, tau_min, tau_max, polish,
                               cross_r, polish_d, min_ratio)


def _fit_from_responses(models, roms, dcs, rom_info, meta, Hf, Yt, dc, n_poles, tau_min, tau_max, polish, cross_r,
                        polish_d, min_ratio) -> FitResult:
    pol = [k for k, m in enumerate(meta) if m[3] is None or m[3] in polish_d]
    # driving-point pairs (source weight = observer weight, or nearly: the absorption density and the resonance
    # weight differ only by the ~1 % oxide share) have positive modal residues: fit them non-negative
    drive = {(SOURCES.index("heater"), OBSERVERS.index("heater_resistor")),
             (SOURCES.index("waveguide_absorption"), OBSERVERS.index("ring_mode_weighted"))}
    nonneg = np.array([m[3] is None and (m[1], m[2]) in drive for m in meta])
    taus, res, info = mor.fit_common_foster(F_FIT_HZ, Hf, T_FIT_S, Yt, dc, n_poles, tau_min, tau_max,
                                            polish_idx=pol, polish=polish, nonneg=nonneg, min_ratio=min_ratio,
                                            log=log)
    info["nonneg_pairs"] = [f"{SOURCES[s]}->{OBSERVERS[o]}" for (s, o) in sorted(drive)]
    canc = np.abs(res).max(axis=1) / np.abs(dc)
    info["residue_cancellation"] = {
        "definition": "max_k |r_k| / |H(0)| per pair (1 = no cancellation)",
        "max_self": float(max(c for c, m in zip(canc, meta) if m[3] is None)),
        "max_cross": float(max((c for c, m in zip(canc, meta) if m[3] is not None), default=0.0)),
        "max_cross_d_le_300um": float(max((c for c, m in zip(canc, meta) if m[3] is not None and m[3] <= 300),
                                          default=0.0))}
    K = taus.size
    D = len(cross_r)
    self_res = np.zeros((len(SOURCES), len(OBSERVERS), K))
    cross_res = np.zeros((len(SOURCES), len(OBSERVERS), D, K))
    dc_self = np.zeros((len(SOURCES), len(OBSERVERS)))
    dc_cross = np.zeros((len(SOURCES), len(OBSERVERS), D))
    for k, (mname, s, o, d, i, j) in enumerate(meta):
        if d is None:
            self_res[s, o] = res[k]
            dc_self[s, o] = dc[k]
        else:
            di = list(cross_r).index(d)
            cross_res[s, o, di] = res[k]
            dc_cross[s, o, di] = dc[k]
    info.update({"roms": rom_info, "n_pairs": len(meta), "n_polish_pairs": len(pol),
                 "f_fit_hz": [float(F_FIT_HZ[0]), float(F_FIT_HZ[-1]), int(F_FIT_HZ.size)],
                 "t_fit_s": [float(T_FIT_S[0]), float(T_FIT_S[-1]), int(T_FIT_S.size)]})
    fr = FitResult(taus, self_res, cross_res, dc_self, dc_cross, np.asarray(cross_r, float), {}, info)
    fr._roms, fr._dcs, fr._meta = roms, dcs, meta
    return fr


def _pair_model(fr: FitResult, s: int, o: int, d) -> mor.Foster:
    if d is None:
        return mor.Foster(fr.taus, fr.self_res[s, o])
    return mor.Foster(fr.taus, fr.cross_res[s, o, list(fr.cross_r).index(d)])


def _collocated(s: int, o: int, d) -> bool:
    """Ring source -> ring-type observer of the same ring: full-band frequency check."""
    return d is None and SOURCES[s] in RING_SOURCES and OBSERVERS[o] in ("ring_mode_weighted", "heater_resistor")


def reference_data(models: list, n_per_level: int = 40, dt0: float = 0.25e-9, t_end: float = 2e-2,
                   step_error_estimate: bool = True, time_check: bool = True) -> dict:
    """FULL-FEM reference responses: exact frequency solves at F_VER_HZ and TR-BDF2 step responses (with a
    discretization-error estimate from a run at twice the step), per model."""
    ref = {}
    for ms in models:
        t0 = time.time()
        d = {"Hex": ms.fem.freq(ms.B, ms.C, F_VER_HZ), "freq_seconds": round(time.time() - t0, 1)}
        if time_check:
            t0 = time.time()
            tt, Y = ms.fem.step(ms.B, ms.C, dt0=dt0, n_per_level=n_per_level, t_end=t_end)
            err = None
            if step_error_estimate:
                tt2, Y2 = ms.fem.step(ms.B, ms.C, dt0=2 * dt0, n_per_level=n_per_level // 2, t_end=t_end)
                Yi = np.array([[np.interp(np.log(tt[1:]), np.log(tt2[1:]), Y2[i, j, 1:])
                                for j in range(Y.shape[1])] for i in range(Y.shape[0])])
                m = (tt[1:] >= 1e-8) & (tt[1:] <= 1e-2)
                dcm = ms.fem.dc(ms.B, ms.C)
                ms.fem.release()
                err = float(np.max(np.abs(Yi[..., m] - Y[..., 1:][..., m]) / np.abs(dcm[..., None]))) / 3.0
            d.update(tt=tt, Y=Y, disc_err=err, n_steps=int(tt.size - 1), step_seconds=round(time.time() - t0, 1))
            log(f"FEM time stepping {ms.name}: {tt.size - 1} steps, {d['step_seconds']} s, discretization error ~ "
                f"{err}")
        ref[ms.name] = d
    return ref


def verify(fr: FitResult, models: list, pitch_um: float, time_check: bool = True, n_per_level: int = 40,
           dt0: float = 0.25e-9, t_end: float = 2e-2, step_error_estimate: bool = True, ref: dict | None = None) -> dict:
    """Errors of the common-pole Foster model against the FULL FEM: exact frequency solves and TR-BDF2 time stepping.

    Frequency: |dB| and |deg| from DC to 1 MHz (collocated self pairs over the whole band; non-collocated pairs only
    where |H| >= NN_FLOOR |H(0)|). Time: max |dy| / y(inf) over 10 ns .. 10 ms."""
    out = {"pitch_um": pitch_um, "f_ver_hz": F_VER_HZ.tolist(), "nn_band_floor": NN_FLOOR,
           "time_window_s": [1e-8, 1e-2]}
    fv = F_VER_HZ[F_VER_HZ <= 1e6]
    per = []
    rom_err = {}
    if ref is None:
        ref = reference_data(models, n_per_level, dt0, t_end, step_error_estimate, time_check)
    for ms in models:
        rd = ref[ms.name]
        Hex = rd["Hex"]
        Hrom = fr._roms[ms.name].freq(F_VER_HZ)
        dc = fr._dcs[ms.name][:, :, None]
        rel = np.abs(Hrom - Hex) / np.abs(dc)
        rom_err[ms.name] = {"max_abs_err_frac_of_dc": float(rel.max()), "seconds": rd["freq_seconds"]}
        if time_check:
            tt, Y = rd["tt"], rd["Y"]
            rom_err[ms.name]["fem_time_steps"] = rd["n_steps"]
            rom_err[ms.name]["fem_step_seconds"] = rd["step_seconds"]
            rom_err[ms.name]["fem_time_discretization_err_frac_of_dc"] = rd["disc_err"]
        for (s, o, d, i, j) in pair_list(ms):
            fm = _pair_model(fr, s, o, d)
            dc0 = fr._dcs[ms.name][i, j]
            coll = _collocated(s, o, d)
            mask_f = F_VER_HZ <= 1e6
            db, deg, fmax = mor.freq_errors(fm, F_VER_HZ[mask_f], Hex[i, j, mask_f],
                                            band_floor=None if coll else NN_FLOOR, dc_ref=dc0)
            rec = {"model": ms.name, "source": SOURCES[s], "observer": OBSERVERS[o], "d_um": d,
                   "dc_k_per_w": float(dc0), "freq_db": float(db), "freq_deg": float(deg),
                   "freq_band_max_hz": float(fmax), "collocated": coll}
            if time_check:
                rec["time_err"] = float(mor.time_errors(fm, tt, Y[i, j], dc0)[()])
            per.append(rec)
    out["rom_vs_fem"] = rom_err

    def summ(sel):
        rs = [r for r in per if sel(r)]
        d = {"n_pairs": len(rs), "max_db": max(r["freq_db"] for r in rs), "max_deg": max(r["freq_deg"] for r in rs)}
        if time_check:
            d["max_time_err_frac_of_dc"] = max(r["time_err"] for r in rs)
            worst = max(rs, key=lambda r: r["time_err"])
            d["worst_time_pair"] = f"{worst['source']}->{worst['observer']}@{worst['d_um']}"
        worst = max(rs, key=lambda r: r["freq_deg"])
        d["worst_phase_pair"] = f"{worst['source']}->{worst['observer']}@{worst['d_um']}"
        return d

    pitch = min(fr.cross_r, key=lambda x: abs(x - pitch_um))
    out["self"] = summ(lambda r: r["d_um"] is None)
    out["self_collocated"] = summ(lambda r: r["collocated"])
    out["nearest_neighbour"] = summ(lambda r: r["d_um"] == pitch)
    out["nearest_neighbour"]["d_um"] = float(pitch)
    out["all_pairs"] = summ(lambda r: True)
    out["key_pairs"] = [r for r in per if (r["d_um"] is None or r["d_um"] == pitch)
                        and r["source"] in ("heater", "waveguide_absorption", "aggressor")]
    th_t, th_db, th_deg = 0.02, 0.3, 3.0
    out["thresholds"] = {"time_err_frac_of_dc": th_t, "db": th_db, "deg": th_deg}
    ok = lambda d: (d["max_db"] < th_db and d["max_deg"] < th_deg  # noqa: E731
                    and (not time_check or d["max_time_err_frac_of_dc"] < th_t))
    out["pass_self"] = bool(ok(out["self"]))
    out["pass_nearest_neighbour"] = bool(ok(out["nearest_neighbour"]))
    out["pass_all_pairs"] = bool(ok(out["all_pairs"]))
    # flat keys read by mrm.plant.thermal.load_rom (its handoff, section 4): worst of the self and nearest-neighbour
    # terms; superposition_rel_err is filled by the 3-D stage (8-ring bank)
    sn = [out["self"], out["nearest_neighbour"]]
    out["step_max_rel_err"] = max(d.get("max_time_err_frac_of_dc", np.nan) for d in sn)
    out["freq_max_db_err"] = max(d["max_db"] for d in sn)
    out["freq_max_deg_err"] = max(d["max_deg"] for d in sn)
    out["superposition_rel_err"] = None
    out["_per_pair"] = per
    return out


# =============================================================================================== convergence
CONV_CROSS = (50.0, 300.0)


def key_metrics(A: ModelSet, Bm: ModelSet | None, pitch_um: float, per_decade: float = 2.0) -> dict:
    """Scalar outputs tracked by the mesh-convergence study and the Jacobians (DC in K/mW, times in s)."""
    GA = A.fem.dc(A.B, A.C)
    A.fem.release()
    fm = A.fem.krylov_modal(A.B, A.C, per_decade=per_decade)
    t63 = fm.t_frac()
    t10 = fm.t_frac(0.1)
    h, wg, j = 0, 1, 2
    rr, rh = A.row(("self", "ring_mode_weighted")), A.row(("self", "heater_resistor"))
    d = min([k[2] for k in A.keys if k[0] == "cross"], key=lambda x: abs(x - pitch_um))
    rx = A.row(("cross", "ring_mode_weighted", d))
    m = {"z_heater_ring_k_per_mw": GA[rr, h] * 1e-3, "z_heater_heater_k_per_mw": GA[rh, h] * 1e-3,
         "z_opt_ring_k_per_mw": GA[rr, wg] * 1e-3, "z_junction_ring_k_per_mw": GA[rr, j] * 1e-3,
         "z_heater_pd_k_per_mw": GA[A.row(("self", "monitor_pd")), h] * 1e-3,
         "xt_frac_at_pitch": GA[rx, h] / GA[rr, h],
         "t63_heater_ring_s": t63[rr, h], "t63_opt_ring_s": t63[rr, wg], "t63_heater_heater_s": t63[rh, h],
         "t10_opt_ring_s": t10[rr, wg], "t63_xt_at_pitch_s": t63[rx, h]}
    for dd in CONV_CROSS:
        key = ("cross", "ring_mode_weighted", dd)
        if key in A.keys:
            m[f"z_heater_ring_at_{int(dd)}um_k_per_mw"] = GA[A.row(key), h] * 1e-3
    if Bm is not None:
        GB = Bm.fem.dc(Bm.B, Bm.C)
        Bm.fem.release()
        m["z_driver_ring_k_per_mw"] = GB[Bm.row(("self", "driver", "ring_mode_weighted")), 0] * 1e-3
        m["z_aggressor_ring_k_per_mw"] = GB[Bm.row(("self", "aggressor", "ring_mode_weighted")), 1] * 1e-3
    return {k: float(v) for k, v in m.items()}


def richardson(values, ratio: float) -> dict:
    """Richardson extrapolation from the last three levels (coarse -> fine, constant refinement ratio)."""
    f = np.asarray(values, float)
    f1, f2, f3 = f[-3:]
    e12, e23 = f2 - f1, f3 - f2
    out = {"values": f.tolist()}
    if e23 == 0 or not np.isfinite(e23):
        out.update(order=None, extrapolated=float(f3), rel_err_est=0.0)
        return out
    q = e12 / e23
    if q > 1.0 + 1e-9:
        p = float(np.log(q) / np.log(ratio))
        p_c = min(max(p, 1.0), 6.0)
        finf = f3 + e23 / (ratio ** p_c - 1.0)
        out.update(order=p, extrapolated=float(finf), rel_err_est=float(abs(f3 - finf) / abs(finf)),
                   method="richardson")
    else:
        # oscillatory or stalled convergence: take the last change as the error (conservative)
        out.update(order=None, extrapolated=float(f3), rel_err_est=float(max(abs(e23), abs(e12)) / abs(f3)),
                   method="last-change")
    out["gci_rel"] = 1.25 * out["rel_err_est"]
    return out


def convergence_study(cfg: ThermalConfig, mw: ModeWeight, mw_c: ModeWeight, scales, prod: tuple | None = None,
                      prod_scale: float | None = None) -> dict:
    """Key metrics at every mesh level (the production models are reused at the production level)."""
    levels = []
    for i, s in enumerate(scales):
        t0 = time.time()
        if prod is not None and prod_scale is not None and abs(s - prod_scale) < 1e-9:
            A, Bm = prod
        else:
            A = build_ring_model(cfg, s, mw, mw_c, cross_r=sorted(set(CONV_CROSS + (cfg.pitch_um,))))
            Bm = build_site_model(cfg, s, mw_c, cross_r=(cfg.pitch_um,))
        km = key_metrics(A, Bm, cfg.pitch_um)
        levels.append({"scale": s, "h_core_um": 0.05 * s, "dofs_ring": A.fem.n, "dofs_site": Bm.fem.n, **km,
                       "seconds": round(time.time() - t0, 1)})
        log(f"convergence level {s}: dofs {A.fem.n}/{Bm.fem.n}, Z_hr {km['z_heater_ring_k_per_mw']:.5f} K/mW, "
            f"t63 {km['t63_heater_ring_s']:.4e} s")
    ratio = scales[0] / scales[1]
    est = {}
    ip = len(scales) - 1 if prod_scale is None else int(np.argmin(np.abs(np.array(scales) - prod_scale)))
    if len(scales) >= 3:
        for k in levels[0]:
            if k in ("scale", "h_core_um", "dofs_ring", "dofs_site", "seconds"):
                continue
            r = richardson([lv[k] for lv in levels], ratio)
            # error of the PRODUCTION level against the extrapolated value
            fp = levels[ip][k]
            r["production_value"] = fp
            r["rel_err_est"] = float(max(abs(fp - r["extrapolated"]) / abs(r["extrapolated"]),
                                         r["rel_err_est"] if ip == len(scales) - 1 else 0.0))
            r["gci_rel"] = 1.25 * r["rel_err_est"]
            est[k] = r
    worst = max(est.items(), key=lambda kv: kv[1]["rel_err_est"]) if est else (None, {"rel_err_est": None})
    return {"levels": levels, "refinement_ratio": ratio, "richardson": est, "production_scale": scales[ip],
            "max_rel_err_est": worst[1]["rel_err_est"], "worst_metric": worst[0],
            "note": "rel_err_est is |f(production level) - f(Richardson extrapolation)| / |f_extrapolated| from the "
                    "last three levels (or the last change when convergence is not monotone)"}


# =============================================================================================== diagnostics
FSR_PFSR = {  # heater efficiency hypotheses in nm/mW: FSR / P_FSR (P_FSR = 25.45 mW, G25 60-die median, printed)
    "printed_0p44": 0.44,
    "fsr_1p80thz_window_b": 10.30 / 25.45,
    "fsr_1p40thz_window_a": 8.01 / 25.45,
    "fsr_1p53thz_g25_cluster": 8.76 / 25.45,
    "fsr_1p60thz_pdk_size": 9.16 / 25.45,
    "swept_device_0p348": 0.348,
}


def dlambda_dt_hypotheses(film_nm: float) -> dict:
    """dlambda/dT hypotheses in pm/K at about 25 C. H_solver comes from the modes artefact when it exists."""
    hyp = {"H86": {"pm_per_k": 85.9, "source": "G26 ~15 GHz/C aside (printed), 07 D2"},
           "materials_ceiling": {"pm_per_k": float(mt.dlambda_dt_ceiling_pm_per_k(1.31e-6, 298.15)[0]),
                                 "source": "mrm.materials dlambda_dt_ceiling_pm_per_k (n_eff/n_g = 0.6), 298 K"},
           "bench_62p5": {"pm_per_k": 62.5, "source": "Lightmatter bench equivalence (inferred); imec 62.54"}}
    cand = [OUT_DIR / f"modes_rib{int(film_nm)}.json"] + sorted(
        c for c in OUT_DIR.glob(f"modes_*{int(film_nm)}*.json") if not c.name.endswith(".sidecar.json"))
    cand = [c for c in cand if c.exists()]
    for c in cand:
        try:
            d = json.loads(c.read_text())
            v = d.get("dlambda_dt_pm_per_k") or d.get("data", {}).get("dlambda_dt_pm_per_k")
            tc, val = np.asarray(v["t_c"], float), np.asarray(v["value"], float)
            hyp["H_solver"] = {"pm_per_k": float(np.interp(25.0, tc, val)), "source": f"{c.name} at 25 C"}
            break
        except Exception:      # noqa: BLE001 - optional input; fall back below
            continue
    if "H_solver" not in hyp:
        hyp["H_solver_proxy"] = {"pm_per_k": 67.5, "source": "06 section 2.7 femwell 66-69 pm/K at 22 C (400x200 "
                                                             "strip); modes artefact not available when this ran"}
    return hyp


def rom_dc_scale(z_heater_ring_k_per_w: float, film_nm: float) -> dict:
    """07 P1-2 diagnostic rom_dc_scale = efficiency_measured / (Z_FEM * dlambda/dT), per hypothesis; warn outside
    [0.5, 2]. Also the R_th each hypothesis requires."""
    z = z_heater_ring_k_per_w * 1e-3     # K/mW
    hyp = dlambda_dt_hypotheses(film_nm)
    tab = {}
    for hk, hv in hyp.items():
        for ek, ev in FSR_PFSR.items():
            s = ev * 1e3 / (z * hv["pm_per_k"])
            tab[f"{hk}|{ek}"] = {"rom_dc_scale": float(s), "required_z_k_per_mw": float(ev * 1e3 / hv["pm_per_k"]),
                                 "warn": bool(s < 0.5 or s > 2.0)}
    return {"z_fem_heater_ring_k_per_mw": float(z), "dlambda_dt_hypotheses": hyp, "efficiency_nm_per_mw": FSR_PFSR,
            "table": tab, "definition": "efficiency_measured / (Z_FEM(0) * dlambda/dT) (07 P1-2)"}


# =============================================================================================== artefact I/O
def tool_versions() -> dict:
    import gmsh
    import scipy
    import skfem
    out = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
           "scikit-fem": skfem.__version__, "gmsh": gmsh.__version__}
    for mod in ("femwell", "pyamg", "meshio"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:      # noqa: BLE001
            out[mod] = None
    return out


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artefact_paths(family: str, film_nm: float, out_dir=OUT_DIR):
    out_dir = pathlib.Path(out_dir)
    stem = f"thermal_rom_{family}_{int(film_nm)}"
    return out_dir / f"{stem}.npz", out_dir / f"{stem}.json"


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write_artefact(cfg: ThermalConfig, fr: FitResult, sidecar: dict, out_dir=OUT_DIR):
    npz_path, js_path = artefact_paths(cfg.family, cfg.film_nm, out_dir)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(cfg.site_offsets_um)
    order = np.argsort(fr.taus)            # the runtime reader wants taus_s strictly increasing
    np.savez(npz_path,
             taus_s=fr.taus[order],
             sources=np.array(SOURCES),
             observers=np.array(OBSERVERS),
             self_res=fr.self_res[..., order],
             cross_r_um=fr.cross_r,
             cross_res=fr.cross_res[..., order],
             site_offset_names=np.array(names),
             site_offsets_um=np.array([cfg.site_offsets_um[n] for n in names], float),
             dc_self_k_per_w=fr.dc_self,
             dc_cross_k_per_w=fr.dc_cross,
             family=np.array(cfg.family),
             film_nm=np.array(float(cfg.film_nm)),
             schema=np.array(SCHEMA))
    sidecar = dict(sidecar)
    if js_path.exists():
        # stage results merged earlier do not depend on the Foster fit (they come from the FEM): keep them
        old = json.loads(js_path.read_text())
        for k in ("stages", "k_temp_scaling"):
            if k in old and k not in sidecar:
                sidecar[k] = old[k]
        for k in ("superposition_rel_err", "superposition_note"):
            if old.get("validation", {}).get(k) is not None and sidecar.get("validation", {}).get(k) is None:
                sidecar.setdefault("validation", {})[k] = old["validation"][k]
        sidecar["previous_sha256"] = old.get("sha256")
    sidecar["site_offsets_um"] = {n: list(map(float, cfg.site_offsets_um[n])) for n in names}
    sidecar.update({"schema": SCHEMA, "file": npz_path.name, "sha256": sha256_file(npz_path),
                    "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    js_path.write_text(json.dumps(_jsonable(sidecar), indent=1))
    return npz_path, js_path


def update_sidecar(family: str, film_nm: float, section: str, payload: dict, out_dir=OUT_DIR) -> bool:
    """Merge a stage result into an existing sidecar (the npz and its sha256 are unchanged)."""
    _, js = artefact_paths(family, film_nm, out_dir)
    if not js.exists():
        return False
    d = json.loads(js.read_text())
    if section == "validation":
        d.setdefault("validation", {}).update(_jsonable(payload))
    elif section == "top":
        d.update(_jsonable(payload))
    else:
        d.setdefault("stages", {})[section] = _jsonable(payload)
    d["updated_utc"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    js.write_text(json.dumps(d, indent=1))
    return True


def load_rom(family: str, film_nm: float, out_dir=OUT_DIR) -> dict:
    """Load an artefact and check its sha256 against the sidecar (the runtime check of ARCHITECTURE 5)."""
    npz, js = artefact_paths(family, film_nm, out_dir)
    side = json.loads(js.read_text())
    if sha256_file(npz) != side["sha256"]:
        raise ValueError(f"{npz.name}: sha256 does not match its sidecar")
    with np.load(npz, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["sidecar"] = side
    return data


# =============================================================================================== extraction driver
@dataclass
class RunOptions:
    quick: bool = False
    scales: tuple = (2.0, 2.0 / np.sqrt(2), 1.0, 1.0 / np.sqrt(2))
    n_poles: int = 20
    prod_scale: float = 1.0
    per_decade: float = 3.0
    n_per_level: int = 40
    step_error_estimate: bool = True
    cross_r: tuple = CROSS_R_UM
    polish: bool = True

    @classmethod
    def quick_options(cls) -> "RunOptions":
        return cls(quick=True, scales=(2.5, 2.5 / np.sqrt(2)), prod_scale=2.5 / np.sqrt(2), n_poles=14,
                   per_decade=2.0, n_per_level=12,
                   step_error_estimate=False, cross_r=(20.0, 50.0, 100.0, 300.0, 1000.0, 2000.0), polish=True)


def extract(family: str, film_nm: float, opts: RunOptions, out_dir=OUT_DIR, cfg_over: dict | None = None) -> dict:
    """Full extraction of one artefact: mode weight, mesh convergence, production ROM fit, full-FEM verification."""
    t_start = time.time()
    cfg = default_config(family, film_nm, **(cfg_over or {}))
    if opts.quick:
        cfg.mode_weight = "uniform_core"
    log(f"=== extract {family} {film_nm} nm ({'quick' if opts.quick else 'full'})")
    mw = mode_weight(cfg)
    mw_c = mw.coarse(0.1)
    log(f"mode weight: {mw.info.get('method')}, n_eff {mw.info.get('n_eff')}, {mw.r.size} points")
    s_prod = opts.prod_scale
    A = build_ring_model(cfg, s_prod, mw, mw_c, cross_r=opts.cross_r)
    Bm = build_site_model(cfg, s_prod, mw_c, cross_r=opts.cross_r)
    log(f"production models: ring {A.info['dofs']} dofs, site {Bm.info['dofs']} dofs")
    # reciprocity (07 P1-2): Z(heater -> ring observer) = Z(source with the ring-observer weight -> heater observer)
    rr_, rh_ = A.row(("self", "ring_mode_weighted")), A.row(("self", "heater_resistor"))
    z_fwd = float(A.fem.dc(A.B[:, :1], A.C[rr_][None, :])[0, 0])
    z_rev = float(A.fem.dc(A.C[rr_][:, None], A.C[rh_][None, :])[0, 0])
    recip = {"z_heater_to_ring_k_per_w": z_fwd, "z_ringweight_to_heater_k_per_w": z_rev,
             "rel_err": abs(z_fwd - z_rev) / abs(z_fwd),
             "heater_source_equals_heater_observer": bool(np.allclose(A.B[:, 0], A.C[rh_], rtol=1e-12, atol=0))}
    A.fem.release()
    log(f"reciprocity: {recip}")
    conv = convergence_study(cfg, mw, mw_c, opts.scales, prod=(A, Bm), prod_scale=s_prod)
    roms = build_roms([A, Bm], per_decade=opts.per_decade)
    ref = reference_data([A, Bm], n_per_level=opts.n_per_level, step_error_estimate=opts.step_error_estimate)
    # adaptive common-pole fit: the first rung that meets the contract (self and nearest neighbour: < 2 % in time,
    # < 0.3 dB, < 3 deg against the full FEM) with residues no larger than 8x the DC gain is kept
    ladder = [(opts.n_poles, 1.5)] if opts.quick else [(opts.n_poles, 1.5), (opts.n_poles + 2, 1.5),
                                                        (opts.n_poles + 4, 1.5), (opts.n_poles + 4, 1.4),
                                                        (opts.n_poles + 6, 1.4)]
    attempts, best = [], None
    for (kp, mr) in ladder:
        fr_i = fit_models([A, Bm], n_poles=kp, polish=opts.polish, cross_r=opts.cross_r, min_ratio=mr, roms=roms)
        val_i = verify(fr_i, [A, Bm], cfg.pitch_um, ref=ref)
        rc = fr_i.fit_info["residue_cancellation"]
        ok = val_i["pass_self"] and val_i["pass_nearest_neighbour"] and rc["max_self"] <= CANCEL_SELF and \
            rc["max_cross_d_le_300um"] <= CANCEL_CROSS
        score = max(val_i["self"]["max_db"] / 0.3, val_i["self"]["max_deg"] / 3.0,
                    val_i["self"]["max_time_err_frac_of_dc"] / 0.02, val_i["nearest_neighbour"]["max_db"] / 0.3,
                    val_i["nearest_neighbour"]["max_deg"] / 3.0, rc["max_self"] / CANCEL_SELF,
                    rc["max_cross_d_le_300um"] / CANCEL_CROSS)
        attempts.append({"n_poles": kp, "min_ratio": mr, "pass": bool(ok), "score": float(score),
                         "self": {k: val_i["self"][k] for k in ("max_db", "max_deg", "max_time_err_frac_of_dc")},
                         "nearest_neighbour": {k: val_i["nearest_neighbour"][k] for k in
                                               ("max_db", "max_deg", "max_time_err_frac_of_dc")},
                         "residue_cancellation": rc})
        log(f"fit rung K={kp} ratio={mr}: pass {ok}, score {score:.3f}")
        if best is None or score < best[2]:
            best = (fr_i, val_i, score)
        if ok:
            break
    fr, val = best[0], best[1]
    fr.fit_info["attempts"] = attempts
    log(f"Foster fit: {fr.taus.size} poles, tau {fr.taus.min():.3e} .. {fr.taus.max():.3e} s")
    log(f"verification: self {val['self']}, nn {val['nearest_neighbour']}")
    per_pair = val.pop("_per_pair")
    # summary numbers
    s_h, s_wg, s_j = SOURCES.index("heater"), SOURCES.index("waveguide_absorption"), SOURCES.index("junction")
    o_r, o_h = OBSERVERS.index("ring_mode_weighted"), OBSERVERS.index("heater_resistor")
    fm_self = mor.Foster(fr.taus, fr.self_res)
    t63 = fm_self.t_frac()
    t10 = fm_self.t_frac(0.1)
    ip = list(fr.cross_r).index(min(fr.cross_r, key=lambda x: abs(x - cfg.pitch_um)))
    fm_x = mor.Foster(fr.taus, fr.cross_res[:, :, ip])
    summary = {
        "z_heater_ring_k_per_mw": fr.dc_self[s_h, o_r] * 1e-3,
        "z_heater_heater_k_per_mw": fr.dc_self[s_h, o_h] * 1e-3,
        "z_opt_ring_k_per_mw": fr.dc_self[s_wg, o_r] * 1e-3,
        "z_junction_ring_k_per_mw": fr.dc_self[s_j, o_r] * 1e-3,
        "z_ratio_opt_heater": fr.dc_self[s_wg, o_r] / fr.dc_self[s_h, o_r],
        "heater_to_ring_coupling": fr.dc_self[s_h, o_r] / fr.dc_self[s_h, o_h],
        "t63_heater_ring_s": t63[s_h, o_r], "t10_heater_ring_s": t10[s_h, o_r],
        "t63_heater_heater_s": t63[s_h, o_h], "t63_opt_ring_s": t63[s_wg, o_r], "t10_opt_ring_s": t10[s_wg, o_r],
        "pitch_um": float(fr.cross_r[ip]),
        "xt_frac_heater_ring_at_pitch": fr.dc_cross[s_h, o_r, ip] / fr.dc_self[s_h, o_r],
        "t63_xt_heater_ring_at_pitch_s": fm_x.t_frac()[s_h, o_r],
        "xt_frac_heater_ring_at_1000um": (fr.dc_cross[s_h, o_r, list(fr.cross_r).index(1000.0)]
                                          / fr.dc_self[s_h, o_r]) if 1000.0 in fr.cross_r else None,
        "package_floor_k_per_mw": (1.0 / (cfg.h_bottom_w_per_m2_k * np.pi * (cfg.die_radius_um * 1e-6) ** 2) * 1e-3
                                   if cfg.bottom_bc == "robin" else 0.0),
        "share_of_dc_in_poles_below_1us_opt_ring": float(fr.self_res[s_wg, o_r][fr.taus < 1e-6].sum()
                                                         / fr.dc_self[s_wg, o_r]),
        "share_of_dc_in_poles_above_1ms_heater_ring": float(fr.self_res[s_h, o_r][fr.taus > 1e-3].sum()
                                                            / fr.dc_self[s_h, o_r]),
    }
    side = {
        "family": family, "film_nm": float(film_nm), "quick": bool(opts.quick),
        "description": "Common-pole Foster thermal ROM (ARCHITECTURE 5.3): step response sum_k r_k (1 - exp(-t/tau_k)),"
                       " residues in K/W, sources normalized to 1 W, observers are weighted averages.",
        "units": {"taus_s": "s", "self_res": "K/W", "cross_res": "K/W", "cross_r_um": "um", "site_offsets_um": "um"},
        "conventions": {
            "self": "source of ring i -> observer of ring i; site observers/sources at site_offsets_um",
            "cross_distance": "source centre (ring centre, or site centre for driver/aggressor) to observer centre "
                              "(ring centre for ring_mode_weighted/heater_resistor, site centre for monitor_pd/"
                              "temp_sensor); radially symmetric approximation, valid d > 2R",
            "observer_weights": {"ring_mode_weighted": "n dn/dT |E|^2 of TE0 (femwell) x 2 pi r",
                                 "heater_resistor": "volume average of the heater body",
                                 "monitor_pd": f"disk r = {cfg.pd_radius_um} um at film mid-plane",
                                 "temp_sensor": f"disk r = {cfg.sensor_radius_um} um at film mid-plane"},
            "source_weights": {"heater": "uniform in the heater body",
                               "waveguide_absorption": "|E|^2 of TE0 in Si (free-carrier absorption) x 2 pi r",
                               "junction": f"density ~ 1/t^2 in the rib and the slab within {cfg.contact_offset_um} "
                                           "um of the rib (lateral series-resistance current path)",
                               "driver": f"uniform disk r = {cfg.driver_radius_um} um in the {cfg.cmos_film_nm} nm "
                                         "CMOS film",
                               "aggressor": f"uniform disk r = {cfg.aggressor_radius_um} um in the CMOS film"}},
        "summary": summary,
        "inputs": {"config": cfg.to_dict(), "assumptions": ASSUMPTIONS, "materials": mt.materials_provenance(),
                   "material_values_k_r_k_z_rho_c": material_table(cfg),
                   "material_notes": ["air (undercut) k = 0.026 W/(m K), rho c = 1.2e3 J/(m^3 K): handbook, not in "
                                      "mrm.materials", "slab and CMOS film k clamped to the 150 nm value of "
                                      "k_si_film_w_per_m_k (fit valid 150-300 nm)",
                                      "BEOL effective medium: k_r = f k_Cu + (1-f) k_d, k_z = f_via k_Cu + "
                                      "k_d/(1-f) (assumed)"],
                   "mode_weight": mw.info},
        "tool_versions": tool_versions(),
        "mesh": {"ring_model": A.info, "site_model": Bm.info, "convergence": conv,
                 "production_scale": s_prod},
        "rom": _jsonable(fr.fit_info),
        "validation": val,
        "validation_per_pair_summary": {"n": len(per_pair),
                                        "max_time_err": max(r.get("time_err", 0) for r in per_pair)},
        "rom_dc_scale": rom_dc_scale(fr.dc_self[s_h, o_r], film_nm),
        "reciprocity": recip,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    npz, js = write_artefact(cfg, fr, side, out_dir)
    log(f"wrote {npz} and {js.name} in {side['runtime_seconds']} s")
    return {"npz": str(npz), "json": str(js), "summary": summary, "validation": val}


# =============================================================================================== uncertainty stage
# (parameter, finite-difference step, assumed 1-sigma) -- the sigmas are assumptions about unpublished geometry
JAC_COMMON = [("width_nm", 50.0, 30.0), ("slab_nm", 20.0, 20.0), ("slab_out_um", 1.0, 1.0), ("box_um", 0.2, 0.1),
              ("clad_um", 0.5, 0.5), ("fill_lo", 0.10, 0.10), ("fill_hi", 0.10, 0.15), ("via_ratio", 0.10, 0.05),
              ("keepout_um", 5.0, 5.0), ("beol_hi_um", 2.0, 2.0), ("radius_um", -1.0, 0.5),
              ("contact_offset_um", 0.5, 0.5)]
JAC_FAMILY = {
    "inring_noundercut": [("heater_gap_um", -1.0, 0.75), ("heater_width_um", 0.5, 0.5),
                          ("heater_doping_factor", -0.2, 0.15)],
    "silicide_undercut": [("sil_gap_um", -0.5, 0.5), ("sil_width_um", 0.5, 0.5), ("undercut_depth_um", -7.5, 5.0),
                          ("undercut_margin_um", 4.0, 3.0)],
    "metal_above": [("metal_gap_um", -0.3, 0.3), ("metal_width_um", 0.5, 0.5), ("metal_thick_um", 0.1, 0.05)],
}
SWEEPS = {"bottom": [{"bottom_bc": "dirichlet"}, {"h_bottom_w_per_m2_k": 1e5}, {"h_bottom_w_per_m2_k": 5e3}],
          "substrate": [{"sub_um": 100.0}, {"sub_um": 300.0}],
          "die": [{"die_side_mm": 3.0}, {"die_side_mm": 10.0}],
          "beol_metal": [{"fill_lo": 0.0, "fill_hi": 0.0}, {"fill_lo": 0.3, "fill_hi": 0.5}],
          "mode_weight": [{"mode_weight": "uniform_core"}, {"mode_bent": False}],
          "air_k": [{"air_k_w_per_m_k": 1e-4}]}
T_SCALE_K = (300.0, 325.0, 350.0, 375.0, 400.0)


def _metrics_for(cfg: ThermalConfig, scale: float, t_k=None, mw=None):
    mw = mw or mode_weight(cfg)
    mw_c = mw.coarse(0.1)
    A = build_ring_model(cfg, scale, mw, mw_c, cross_r=(cfg.pitch_um,), t_k=t_k)
    Bm = build_site_model(cfg, scale, mw_c, cross_r=(cfg.pitch_um,), t_k=t_k)
    return key_metrics(A, Bm, cfg.pitch_um)


def uncertainty(family: str, film_nm: float, opts: RunOptions) -> dict:
    """Finite-difference Jacobians d ln(metric)/d param, named sweeps of the boundary/stack choices, and the
    quasi-static k(T), rho c(T) scaling of 07 P2-5. Coarse mesh (ratios converge faster than absolute values)."""
    scale = 2.0 if not opts.quick else 3.0
    base_cfg = default_config(family, film_nm)
    if opts.quick:
        base_cfg.mode_weight = "uniform_core"
    mw0 = mode_weight(base_cfg)
    base = _metrics_for(base_cfg, scale, mw=mw0)
    log(f"uncertainty {family} {film_nm}: base Z_hr {base['z_heater_ring_k_per_mw']:.4f} K/mW")
    params = JAC_COMMON + JAC_FAMILY[family]
    if opts.quick:
        params = params[:2]
    jac, contrib = {}, {}
    for (pname, step, sig) in params:
        cfg = default_config(family, film_nm, **{pname: getattr(base_cfg, pname) + step})
        cfg.mode_weight = base_cfg.mode_weight
        mw = mw0 if pname not in ("width_nm", "slab_nm", "radius_um", "slab_out_um") else None
        m = _metrics_for(cfg, scale, mw=mw)
        d = {k: (np.log(abs(m[k])) - np.log(abs(base[k]))) / step for k in base if base[k] and m.get(k)}
        jac[pname] = {"step": step, "sigma_assumed": sig, "base_value": getattr(base_cfg, pname), "d_ln_dparam": d}
        contrib[pname] = {k: abs(v) * sig for k, v in d.items()}
        log(f"  jac {pname}: dlnZ_hr {d['z_heater_ring_k_per_mw']:+.4f}, dln t63 {d['t63_heater_ring_s']:+.4f}")
    sig_tot = {k: float(np.sqrt(sum(c[k] ** 2 for c in contrib.values() if k in c))) for k in base}
    dominant = {k: max(contrib, key=lambda p: contrib[p].get(k, 0.0)) for k in base}
    sweeps = {}
    sw = SWEEPS if not opts.quick else {"bottom": SWEEPS["bottom"][:1]}
    for name, variants in sw.items():
        recs = []
        for over in variants:
            cfg = default_config(family, film_nm, **over)
            if opts.quick:
                cfg.mode_weight = "uniform_core"
            m = _metrics_for(cfg, scale)
            recs.append({"override": over, "metrics": m,
                         "ratio_to_base": {k: m[k] / base[k] for k in base if base[k] and m.get(k)}})
            log(f"  sweep {over}: Z_hr ratio {recs[-1]['ratio_to_base']['z_heater_ring_k_per_mw']:.4f}")
        sweeps[name] = recs
    # G25 consistency: which unpublished choices reach the measured heater efficiency, and at what time constants
    g25 = []
    if not opts.quick:
        fam_knob = {"inring_noundercut": ("heater_gap_um", (0.5, 1.0, 2.0)),
                    "silicide_undercut": ("sil_gap_um", (0.5, 1.5)),
                    "metal_above": ("metal_gap_um", (0.3, 0.8))}[family]
        for v in fam_knob[1]:
            for fill in ((0.15, 0.30), (0.0, 0.0)):
                for ko in (5.0, 20.0):
                    over = {fam_knob[0]: v, "fill_lo": fill[0], "fill_hi": fill[1], "keepout_um": ko}
                    cfg = default_config(family, film_nm, **over)
                    m = _metrics_for(cfg, scale, mw=mw0)
                    req = {h: FSR_PFSR["fsr_1p80thz_window_b"] * 1e3 / dv["pm_per_k"] for h, dv in
                           dlambda_dt_hypotheses(film_nm).items()}
                    g25.append({"override": over, "z_heater_ring_k_per_mw": m["z_heater_ring_k_per_mw"],
                                "t63_heater_ring_s": m["t63_heater_ring_s"], "xt_frac_at_pitch": m["xt_frac_at_pitch"],
                                "z_opt_ring_k_per_mw": m["z_opt_ring_k_per_mw"],
                                "rom_dc_scale_by_hypothesis": {h: r / m["z_heater_ring_k_per_mw"]
                                                               for h, r in req.items()}})
        log(f"  G25 consistency: Z_hr range {min(g['z_heater_ring_k_per_mw'] for g in g25):.3f} .. "
            f"{max(g['z_heater_ring_k_per_mw'] for g in g25):.3f} K/mW")
    temps = T_SCALE_K if not opts.quick else (300.0, 400.0)
    kt = []
    for T in temps:
        m = base if T == 300.0 else _metrics_for(base_cfg, scale, t_k=T, mw=mw0)
        kt.append({"t_k": T, "ratio_to_300k": {k: m[k] / base[k] for k in base if base[k] and m.get(k)}})
    # a suggestion for uncertainty.thermal (params owner): +/- 2 sigma of the geometry Jacobian propagation
    sugg = {
        "z_heater_ring_scale_2sigma": [float(np.exp(-2 * sig_tot["z_heater_ring_k_per_mw"])),
                                       float(np.exp(2 * sig_tot["z_heater_ring_k_per_mw"]))],
        "time_scale_fast_2sigma": [float(np.exp(-2 * sig_tot["t63_heater_ring_s"])),
                                   float(np.exp(2 * sig_tot["t63_heater_ring_s"]))],
        "selfheat_tau_scale_2sigma": [float(np.exp(-2 * sig_tot["t10_opt_ring_s"])),
                                      float(np.exp(2 * sig_tot["t10_opt_ring_s"]))],
        "crosstalk_scale_2sigma": [float(np.exp(-2 * sig_tot["xt_frac_at_pitch"])),
                                   float(np.exp(2 * sig_tot["xt_frac_at_pitch"]))],
        "z_ratio_opt_heater_base": base["z_opt_ring_k_per_mw"] / base["z_heater_ring_k_per_mw"],
        "note": "independent Gaussian parameters with the assumed sigmas; bottom/substrate/die choices are in sweeps",
    }
    return {"mesh_scale": scale, "base_metrics": base, "jacobian": jac, "sigma_ln_total": sig_tot,
            "dominant_parameter": dominant, "sweeps": sweeps,
            "k_temp_scaling": {"method": "07 P2-5 quasi-static: DC gains and 63 % times recomputed with k(T) and "
                                         "rho c(T) of mrm.materials; the runtime scales residues by the DC ratio "
                                         "and taus by the t63 ratio", "points": kt},
            "uncertainty_thermal_suggestion": sugg,
            "g25_consistency": {"efficiency_nm_per_mw": FSR_PFSR["fsr_1p80thz_window_b"],
                                "note": "rom_dc_scale = (FSR/P_FSR) / (Z * dlambda/dT) per variant; 1 means the "
                                        "variant reproduces the G25 median efficiency under that hypothesis",
                                "variants": g25}}


def stage_uncertainty(fams, films, opts, out):
    res = {}
    for fam in fams:
        for film in films:
            u = uncertainty(fam, film, opts)
            update_sidecar(fam, film, "uncertainty", u, out)
            pts = u["k_temp_scaling"]["points"]
            kts = {"t_k": [p_["t_k"] for p_ in pts],
                   "dc_ratio": {"heater->ring_mode_weighted": [p_["ratio_to_300k"]["z_heater_ring_k_per_mw"] for p_ in pts],
                                "heater->heater_resistor": [p_["ratio_to_300k"]["z_heater_heater_k_per_mw"] for p_ in pts],
                                "waveguide_absorption->ring_mode_weighted":
                                    [p_["ratio_to_300k"]["z_opt_ring_k_per_mw"] for p_ in pts],
                                "heater->ring_mode_weighted@pitch":
                                    [p_["ratio_to_300k"]["xt_frac_at_pitch"] * p_["ratio_to_300k"]["z_heater_ring_k_per_mw"]
                                     for p_ in pts]},
                   "tau_ratio": {"heater->ring_mode_weighted": [p_["ratio_to_300k"]["t63_heater_ring_s"] for p_ in pts],
                                 "waveguide_absorption->ring_mode_weighted":
                                     [p_["ratio_to_300k"]["t63_opt_ring_s"] for p_ in pts]},
                   "note": "07 P2-5 quasi-static scaling: multiply residues by dc_ratio(T) and taus by tau_ratio(T) "
                           "(T = mean temperature of the conduction path, K); k(T), rho c(T) from mrm.materials"}
            update_sidecar(fam, film, "top", {"k_temp_scaling": kts}, out)
            res[f"{fam}_{film}"] = {"sigma_ln_total": u["sigma_ln_total"],
                                    "suggestion": u["uncertainty_thermal_suggestion"]}
    return res


# =============================================================================================== 3-D model
@dataclass
class Shape3:
    name: str
    kind: str          # box | annulus
    par: tuple         # box: (x0, x1, y0, y1, z0, z1); annulus: (cx, cy, r0, r1, z0, z1)
    material: str
    priority: int = 0

    def contains(self, x, y, z):
        if self.kind == "box":
            x0, x1, y0, y1, z0, z1 = self.par
            return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (z >= z0) & (z <= z1)
        if self.kind == "sector":
            cx, cy, r0, r1, z0, z1, th0, dth = self.par
            r = np.hypot(x - cx, y - cy)
            th = np.mod(np.arctan2(y - cy, x - cx) - th0, 2 * np.pi)
            return (r >= r0) & (r <= r1) & (z >= z0) & (z <= z1) & (th <= dth)
        cx, cy, r0, r1, z0, z1 = self.par
        r = np.hypot(x - cx, y - cy)
        return (r >= r0) & (r <= r1) & (z >= z0) & (z <= z1)


def bank_geometry3d(cfg: ThermalConfig, centres, margin_um: float = 400.0, leads: bool = True,
                    bus: bool = True, arcs=None) -> tuple:
    """3-D stack with rings at the given (x, y) centres (heater family annuli, slabs, rib, metal keep-out), plus, for
    the bank test, heater leads (via + M1 Cu line to +y) and a straight bus waveguide along x."""
    xs = [c[0] for c in centres]
    x0, x1 = min(xs) - margin_um, max(xs) + margin_um
    y0, y1 = -margin_um, margin_um
    R, w, tf, ts = cfg.radius_um, cfg.w, cfg.t_f, cfg.t_s
    sh = [Shape3("sub", "box", (x0, x1, y0, y1, cfg.z_bot, -cfg.box_um), "si_sub"),
          Shape3("box", "box", (x0, x1, y0, y1, -cfg.box_um, 0.0), "box"),
          Shape3("clad", "box", (x0, x1, y0, y1, 0.0, cfg.z_beol0), "clad"),
          Shape3("beol_lo", "box", (x0, x1, y0, y1, cfg.z_beol0, cfg.z_beol1), "beol_lo"),
          Shape3("beol_hi", "box", (x0, x1, y0, y1, cfg.z_beol1, cfg.z_beol2), "beol_hi"),
          Shape3("pass", "box", (x0, x1, y0, y1, cfg.z_beol2, cfg.z_top), "pass")]
    s_in, s_out = cfg.slab_extents()
    for i, (cx, cy) in enumerate(centres):
        ko = R + cfg.keepout_um
        sh += [Shape3(f"ko_lo{i}", "annulus", (cx, cy, 0.0, ko, cfg.z_beol0, cfg.z_beol1), "beol_lo_ko", 1),
               Shape3(f"ko_hi{i}", "annulus", (cx, cy, 0.0, ko, cfg.z_beol1, cfg.z_beol2), "beol_hi_ko", 1),
               Shape3(f"slab{i}", "annulus", (cx, cy, R - w / 2 - s_in, R + w / 2 + s_out, 0.0, ts), "si_slab", 2),
               Shape3(f"rib{i}", "annulus", (cx, cy, R - w / 2, R + w / 2, 0.0, tf), "si_film", 3)]
        for k, (r0, r1, z0, z1, m) in enumerate(cfg.heater_rects()):
            if arcs is None:
                sh.append(Shape3(f"heater{i}_{k}", "annulus", (cx, cy, r0, r1, z0, z1), m, 5))
            else:       # resistor arcs (centre angle, span) in rad, each a separate body of the same heater
                for a, (thc, dth) in enumerate(arcs):
                    sh.append(Shape3(f"heater{i}_{k}_arc{a}", "sector",
                                     (cx, cy, r0, r1, z0, z1, thc - dth / 2, dth), m, 5))
        if cfg.family == "silicide_undercut":
            sh.append(Shape3(f"undercut{i}", "annulus", (cx, cy, 0.0, cfg.undercut_radius_um(),
                                                          -cfg.box_um - cfg.undercut_depth_um, -cfg.box_um), "air", 6))
        if leads:
            r0, r1, z0, z1, m = cfg.heater_rects()[0]
            rv = 0.5 * (r0 + r1)
            for sgn in (-1, 1):
                ang = np.pi / 2 + sgn * np.radians(10.0)
                vx, vy = cx + rv * np.cos(ang), cy + rv * np.sin(ang)
                zv0 = z1 if m != "metal" else z1
                sh.append(Shape3(f"via{i}_{sgn}", "box", (vx - 0.2, vx + 0.2, vy - 0.2, vy + 0.2, zv0, cfg.z_beol0),
                                 "cu", 6))
                sh.append(Shape3(f"m1_{i}_{sgn}", "box", (vx - 0.3, vx + 0.3, vy - 0.2, cy + R + 60.0, cfg.z_beol0,
                                                          cfg.z_beol0 + 0.2), "cu", 6))
    if bus:
        yb = centres[0][1] - (R + w / 2 + 0.2 + w / 2)
        sh.append(Shape3("bus", "box", (x0, x1, yb - w / 2, yb + w / 2, 0.0, tf), "si_film", 3))
    return sh, (x0, x1, y0, y1)


def mesh_3d(shapes: list, cfg: ThermalConfig, centres, h_ring: float, h_far: float = 60.0, growth: float = 0.25,
            verbose: bool = False):
    """Conforming tetrahedral mesh (gmsh OCC fragment); returns (p (3, N), t (4, E), region index per tet)."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.model.add("bank3d")
        occ = gmsh.model.occ
        tags = []
        for shp in shapes:
            if shp.kind == "box":
                x0, x1, y0, y1, z0, z1 = shp.par
                tags.append(occ.addBox(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0))
            elif shp.kind == "sector":
                cx, cy, r0, r1, z0, z1, th0, dth = shp.par
                outer = occ.addCylinder(cx, cy, z0, 0, 0, z1 - z0, r1, angle=dth)
                inner = occ.addCylinder(cx, cy, z0, 0, 0, z1 - z0, r0, angle=dth)
                res, _ = occ.cut([(3, outer)], [(3, inner)])
                occ.rotate(res, cx, cy, z0, 0, 0, 1, th0)
                tags.append(res[0][1])
            else:
                cx, cy, r0, r1, z0, z1 = shp.par
                outer = occ.addCylinder(cx, cy, z0, 0, 0, z1 - z0, r1)
                if r0 > 0:
                    inner = occ.addCylinder(cx, cy, z0, 0, 0, z1 - z0, r0)
                    res, _ = occ.cut([(3, outer)], [(3, inner)])
                    tags.append(res[0][1])
                else:
                    tags.append(outer)
        occ.fragment([(3, tags[0])], [(3, t) for t in tags[1:]])
        occ.synchronize()
        fl = gmsh.model.mesh.field
        fids = []
        xs = [c[0] for c in centres]
        xc, half = 0.5 * (min(xs) + max(xs)), 0.5 * (max(xs) - min(xs))
        R = cfg.radius_um
        f = fl.add("MathEval")
        fl.setString(f, "F", f"{1.0 * h_ring / 0.25} + {growth} * Sqrt(Max(Sqrt(Max(Abs(x-{xc})-{half},0)^2 + y^2)"
                             f" - {R + 4.0}, 0)^2 + (z - {cfg.t_f / 2})^2)")
        fids.append(f)
        for (cx, cy) in centres:
            for (rr, z0, z1, h) in [(R + cfg.slab_out_um + 0.6, -0.1, cfg.t_f + 0.1, h_ring),
                                    (R + 6.0, -1.0, cfg.z_beol0 + 0.4, 3.0 * h_ring)]:
                f = fl.add("Box")
                fl.setNumber(f, "VIn", h)
                fl.setNumber(f, "VOut", 1e9)
                fl.setNumber(f, "XMin", cx - rr)
                fl.setNumber(f, "XMax", cx + rr)
                fl.setNumber(f, "YMin", cy - rr)
                fl.setNumber(f, "YMax", cy + rr)
                fl.setNumber(f, "ZMin", z0)
                fl.setNumber(f, "ZMax", z1)
                fl.setNumber(f, "Thickness", 2.0)
                fids.append(f)
        for shp in shapes:
            if shp.name.startswith("m1_") or shp.name.startswith("via") or shp.name == "bus":
                x0, x1, y0, y1, z0, z1 = shp.par
                f = fl.add("Box")
                fl.setNumber(f, "VIn", max(2 * h_ring, min(0.5, (z1 - z0) * 1.5)) if shp.name != "bus" else 2 * h_ring)
                fl.setNumber(f, "VOut", 1e9)
                for nm, v in zip(("XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax"),
                                 (x0 - 0.3, x1 + 0.3, y0 - 0.3, y1 + 0.3, z0 - 0.2, z1 + 0.2)):
                    fl.setNumber(f, nm, v)
                fl.setNumber(f, "Thickness", 1.0)
                fids.append(f)
        fmin = fl.add("Min")
        fl.setNumbers(fmin, "FieldsList", fids)
        fl.setAsBackgroundMesh(fmin)
        for k, v in (("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.MeshSizeFromPoints", 0),
                     ("Mesh.MeshSizeFromCurvature", 0), ("Mesh.MeshSizeMax", h_far), ("Mesh.Algorithm", 6),
                     ("Mesh.Algorithm3D", 1), ("Mesh.Optimize", 1)):
            gmsh.option.setNumber(k, v)
        gmsh.model.mesh.generate(3)
        ntags, coords, _ = gmsh.model.mesh.getNodes()
        etypes, _, enodes = gmsh.model.mesh.getElements(3)
    finally:
        gmsh.finalize()
    tet = np.asarray(enodes[list(etypes).index(4)], np.int64).reshape(-1, 4)
    used = np.unique(tet)
    idx = np.full(int(ntags.max()) + 1, -1, np.int64)
    idx[used] = np.arange(used.size)
    pos = np.full(int(ntags.max()) + 1, -1, np.int64)
    pos[np.asarray(ntags, np.int64)] = np.arange(ntags.size)
    p = np.asarray(coords).reshape(-1, 3)[pos[used]].T.copy()
    t = idx[tet].T.copy()
    cen = p[:, t].mean(axis=1)
    reg = np.full(t.shape[1], -1, int)
    best = np.full(t.shape[1], -10 ** 9)
    for i, shp in enumerate(shapes):
        m = shp.contains(*cen) & (shp.priority >= best)
        reg[m] = i
        best[m] = shp.priority
    if np.any(reg < 0):
        raise RuntimeError("unclassified tetrahedra")
    return p, t, reg


def amg_factory(A):
    """solve(X) by conjugate gradients with a pyamg smoothed-aggregation preconditioner (SPD systems)."""
    import pyamg

    A = sp.csr_matrix(A)
    d = A.diagonal()
    off = np.asarray(abs(A).sum(axis=1)).ravel() - np.abs(d)
    dominance = float(np.median(d / np.maximum(off, 1e-300)))
    Mp = None
    if dominance < 1.05:
        # stiffness-dominated: AMG; mass-dominated shifts make aggregation stall (operator complexity blows up), so
        # those use Jacobi, which converges in a few iterations on a diagonally dominant matrix
        ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric", max_coarse=2000, max_levels=12)
        if ml.operator_complexity() < 2.5:
            Mp = ml.aspreconditioner(cycle="V")
        del ml
    if Mp is None:
        Mp = sp.diags(1.0 / d)

    def solve(X):
        X = np.asarray(X, float)
        one = X.ndim == 1
        X2 = X[:, None] if one else X
        out = np.zeros_like(X2)
        for j in range(X2.shape[1]):
            b = X2[:, j]
            nb = np.linalg.norm(b)
            if nb == 0:
                continue
            x, info = spla.cg(A, b, rtol=1e-11, maxiter=2000, M=Mp)
            if info != 0:
                raise RuntimeError(f"CG did not converge (info {info})")
            out[:, j] = x
        return out[:, 0] if one else out
    return solve


class Fem3D:
    """P1 tetrahedral heat-conduction model with anisotropic k (k_x = k_y = k_r, k_z), Dirichlet or Robin bottom."""

    def __init__(self, p, t, reg, shapes, mats, z_bot, bc="dirichlet", h_bottom=2e4):
        from skfem import Basis, BilinearForm, ElementTetP1, FacetBasis, LinearForm, MeshTet  # noqa: F401

        self.mesh = MeshTet(np.ascontiguousarray(p), np.ascontiguousarray(t))
        self.reg = reg
        self.shapes = shapes
        self.basis = Basis(self.mesh, ElementTetP1())
        nq = self.basis.quadrature[0].shape[1]
        kr = np.array([mats[shapes[i].material][0] for i in reg])
        kz = np.array([mats[shapes[i].material][1] for i in reg])
        rc = np.array([mats[shapes[i].material][2] for i in reg])
        rep = lambda a: np.repeat(a[:, None], nq, axis=1)  # noqa: E731

        @BilinearForm
        def stiff(u, v, w):
            return w.kr * (u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1]) + w.kz * u.grad[2] * v.grad[2]

        @BilinearForm
        def mass(u, v, w):
            return w.rc * u * v

        K = stiff.assemble(self.basis, kr=rep(kr) * 1e-6, kz=rep(kz) * 1e-6)
        M = mass.assemble(self.basis, rc=rep(rc) * 1e-18)
        bottom = self.mesh.facets_satisfying(lambda x: np.isclose(x[2], z_bot, atol=1e-9))
        if bc == "robin":
            fb = FacetBasis(self.mesh, ElementTetP1(), facets=bottom)

            @BilinearForm
            def robin(u, v, w):
                return h_bottom * 1e-12 * u * v

            K = K + robin.assemble(fb)
            self.free = np.arange(self.basis.N)
        else:
            D = self.basis.get_dofs(facets=bottom).all()
            self.free = np.setdiff1d(np.arange(self.basis.N), D)
        self.K = K.tocsr()[self.free][:, self.free].tocsr()
        self.M = M.tocsr()[self.free][:, self.free].tocsr()
        self.n = self.free.size

    def region_vector(self, names) -> np.ndarray:
        from skfem import LinearForm
        nq = self.basis.quadrature[0].shape[1]
        ids = [i for i, s_ in enumerate(self.shapes) if s_.name in names]
        ind = np.repeat(np.isin(self.reg, ids).astype(float)[:, None], nq, axis=1)

        @LinearForm
        def lf(v, w):
            return w.ind * v

        b = lf.assemble(self.basis, ind=ind)
        return b[self.free] / b.sum()

    def point_vector(self, pts, wts) -> np.ndarray:
        """sum_p w_p phi_i(x_p), normalized; P1 interpolation with a memory-bounded tetrahedron locator (KD-tree of
        centroids, batched barycentric tests over growing candidate lists)."""
        pts = np.asarray(pts, float)
        w = np.asarray(wts, float)
        w = w / w.sum()
        elem, bary = locate_tets(self.mesh.p, self.mesh.t, pts, tree=self._tree())
        v = np.zeros(self.basis.N)
        np.add.at(v, self.mesh.t[:, elem].T.ravel(), (bary * w[:, None]).ravel())
        return v[self.free]

    def _tree(self):
        from scipy.spatial import cKDTree
        if not hasattr(self, "_ctree"):
            self._ctree = cKDTree(self.mesh.p[:, self.mesh.t].mean(axis=1).T)
        return self._ctree


def locate_tets(p, t, pts, tree=None, k0: int = 12, tol: float = 1e-9):
    """Element index and barycentric coordinates of every point (3, n) in a tetrahedral mesh.

    Candidates are the k nearest element centroids (k grows x4 for points not yet found, up to 3072); a point that
    is still unlocated (outside by round-off) takes the candidate with the least negative coordinate, clipped."""
    from scipy.spatial import cKDTree
    tree = tree or cKDTree(p[:, t].mean(axis=1).T)
    n = pts.shape[1]
    elem = np.full(n, -1, np.int64)
    bary = np.zeros((n, 4))
    best_e = np.zeros(n, np.int64)
    best_m = np.full(n, -np.inf)
    best_b = np.zeros((n, 4))
    todo = np.arange(n)
    k = k0
    ne = t.shape[1]
    while todo.size and k <= 3072:
        _, cand = tree.query(pts[:, todo].T, k=min(k, ne))
        cand = np.atleast_2d(cand)
        for j in range(cand.shape[1]):
            if not todo.size:
                break
            e = cand[:, j] if cand.shape[0] == todo.size else None
            if e is None:
                break
            v0 = p[:, t[0, e]].T
            Mt = np.stack([p[:, t[1, e]].T - v0, p[:, t[2, e]].T - v0, p[:, t[3, e]].T - v0], axis=2)
            lam = np.linalg.solve(Mt, (pts[:, todo].T - v0)[:, :, None])[:, :, 0]
            b4 = np.column_stack([1 - lam.sum(axis=1), lam])
            mn = b4.min(axis=1)
            better = mn > best_m[todo]
            best_m[todo[better]] = mn[better]
            best_e[todo[better]] = e[better]
            best_b[todo[better]] = b4[better]
            ok = mn >= -tol
            elem[todo[ok]] = e[ok]
            bary[todo[ok]] = b4[ok]
            keep = ~ok
            todo = todo[keep]
            cand = cand[keep]
        k *= 4
    if todo.size:
        elem[todo] = best_e[todo]
        b = np.clip(best_b[todo], 0, None)
        bary[todo] = b / b.sum(axis=1, keepdims=True)
    return elem, bary


def ring_points_3d(mw_c: ModeWeight, cfg: ThermalConfig, cx: float, cy: float, nphi: int = 48):
    phi = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
    r = mw_c.r[:, None]
    x = cx + r * np.cos(phi)[None, :]
    y = cy + r * np.sin(phi)[None, :]
    z = np.broadcast_to(mw_c.z[:, None], x.shape)
    w = np.broadcast_to((mw_c.w_obs * mw_c.r)[:, None], x.shape)
    return np.vstack([x.ravel(), y.ravel(), z.ravel()]), w.ravel()


def run_3d(cfg: ThermalConfig, centres, driven, h_ring, mw_c, leads, bus, per_decade=2.0, margin=400.0, arcs=None,
           victims=()):
    """Build, mesh and reduce a 3-D model; returns DC (O, S), the modal ROM and mesh info.

    Observers: ring_mode_weighted of every ring, then heater_resistor of every ring. Sources: heaters of ``driven``."""
    t0 = time.time()
    shapes, _ = bank_geometry3d(cfg, centres, margin_um=margin, leads=leads, bus=bus, arcs=arcs)
    if victims:
        margin = max(margin, max(abs(v[0]) for v in victims) + 400.0)
        shapes, _ = bank_geometry3d(cfg, centres, margin_um=margin, leads=leads, bus=bus, arcs=arcs)
    mats = dict(material_table(cfg))
    mats["cu"] = (float(mt.k_metal_w_per_m_k("cu_beol", cfg.t_k)),) * 2 + (float(mt.rho_c_j_per_m3_k("cu", cfg.t_k)),)
    p, t, reg = mesh_3d(shapes, cfg, centres, h_ring)
    t_mesh = time.time() - t0
    fem = Fem3D(p, t, reg, shapes, mats, cfg.z_bot, cfg.bottom_bc, cfg.h_bottom_w_per_m2_k)
    hnames = lambda i: [s_.name for s_ in shapes if s_.name.startswith(f"heater{i}_")]  # noqa: E731
    B = np.column_stack([fem.region_vector(hnames(i)) for i in driven])
    rows = [fem.point_vector(*ring_points_3d(mw_c, cfg, cx, cy)) for (cx, cy) in centres]
    rows += [fem.region_vector(hnames(i)) for i in range(len(centres))]
    rows += [fem.point_vector(*ring_points_3d(mw_c, cfg, vx, vy)) for (vx, vy) in victims]   # probe-only rings
    C = np.vstack(rows)
    shifts = mor.log_shifts(1e-2, 1e8, per_decade)
    V = mor.rational_krylov_basis(fem.K, fem.M, B, shifts, n_moments=2, factor=amg_factory)
    fm = mor.galerkin_modal(fem.K, fem.M, B, C, V)
    dc = np.asarray(C @ amg_factory(fem.K)(B))
    info = {"nodes": int(fem.mesh.nvertices), "tets": int(fem.mesh.nelements), "h_ring_um": h_ring,
            "mesh_seconds": round(t_mesh, 1), "total_seconds": round(time.time() - t0, 1), "rom_order": int(fm.taus.size),
            "rom_dc_rel_err": float(np.max(np.abs(fm.dc - dc) / np.abs(dc))), "leads": leads, "bus": bus,
            "margin_um": margin}
    log(f"3-D model ({len(centres)} rings, h {h_ring}): {info}")
    return dc, fm, info


def stage_three_d(fams, films, opts, out):
    """3-D checks on the default family: (a) single ring, no leads, against the axisymmetric model (formulation);
    (b) single ring with leads and bus (lead effect); (c) the 8-ring bank at the design pitch with leads and bus
    (superposition error of the distance-interpolated single-ring ROM). All in the comparison variant: Dirichlet
    bottom under a 200 um substrate, so that the lateral truncation of the 3-D box is immaterial."""
    fam, film = "inring_noundercut", 160.0
    cfg = default_config(fam, film, bottom_bc="dirichlet", sub_um=200.0)
    if opts.quick:
        cfg.mode_weight = "uniform_core"
    mw = mode_weight(cfg)
    mw_c = mw.coarse(0.1)
    pitch = cfg.pitch_um
    n_bank = 8 if not opts.quick else 3
    d_list = [pitch * k for k in range(1, n_bank)]
    # axisymmetric reference with the same physics and exact distances
    ax_scale = 1.0 if not opts.quick else 2.0
    A = build_ring_model(cfg, ax_scale, mw, mw_c, cross_r=tuple(d_list))
    GA = A.fem.dc(A.B, A.C)
    fmA = A.fem.krylov_modal(A.B, A.C, per_decade=3.0)
    rr, rh = A.row(("self", "ring_mode_weighted")), A.row(("self", "heater_resistor"))
    ax = {"z_heater_ring": GA[rr, 0], "z_heater_heater": GA[rh, 0],
          "xt": {d: GA[A.row(("cross", "ring_mode_weighted", d)), 0] for d in d_list}}
    t_grid = np.logspace(-8, -2, 61)
    res = {"variant": {"bottom_bc": "dirichlet", "sub_um": 200.0}, "pitch_um": pitch, "axisym_scale": ax_scale}
    # (a) + (b): single ring
    levels = (0.5, 0.35) if not opts.quick else (0.6,)
    single = []
    victims = [(d, 0.0) for d in d_list]
    for h in levels:
        dc, fm, info = run_3d(cfg, [(0.0, 0.0)], [0], h, mw_c, leads=False, bus=False, victims=victims)
        t63 = fm.t_frac()
        single.append({"h_ring_um": h, "info": info, "z_heater_ring": dc[0, 0], "z_heater_heater": dc[1, 0],
                       "t63_heater_ring_s": t63[0, 0],
                       "xt_dc": {str(d): float(dc[2 + k, 0]) for k, d in enumerate(d_list)},
                       "xt_dc_rel_to_axisym": {str(d): float(dc[2 + k, 0] / ax["xt"][d] - 1)
                                               for k, d in enumerate(d_list)}})
        fm_single = fm
    t63_ax = fmA.t_frac()[rr, 0]
    res["single_ring_no_leads"] = {
        "levels": single, "axisym_z_heater_ring": ax["z_heater_ring"], "axisym_z_heater_heater": ax["z_heater_heater"],
        "axisym_t63_heater_ring_s": t63_ax,
        "rel_diff_z_heater_ring": single[-1]["z_heater_ring"] / ax["z_heater_ring"] - 1,
        "rel_diff_z_heater_heater": single[-1]["z_heater_heater"] / ax["z_heater_heater"] - 1,
        "rel_diff_t63": single[-1]["t63_heater_ring_s"] / t63_ax - 1,
        "mesh_change_last_level": (single[-1]["z_heater_ring"] / single[0]["z_heater_ring"] - 1) if len(single) > 1
        else None}
    h_b = 0.4 if not opts.quick else 0.6
    dc, fm, info = run_3d(cfg, [(0.0, 0.0)], [0], h_b, mw_c, leads=True, bus=True, victims=victims)
    res["single_ring_with_leads"] = {"info": info, "z_heater_ring": dc[0, 0], "z_heater_heater": dc[1, 0],
                                     "lead_effect_z_heater_ring": dc[0, 0] / single[-1]["z_heater_ring"] - 1,
                                     "lead_effect_z_heater_heater": dc[1, 0] / single[-1]["z_heater_heater"] - 1,
                                     "xt_dc": {str(d): float(dc[2 + k, 0]) for k, d in enumerate(d_list)}}
    yS = fm.step(t_grid)        # 3-D single ring with leads, victims probed (no victim structures)
    dcS = dc
    # (c) the bank
    centres = [(pitch * k, 0.0) for k in range(n_bank)]
    driven = [0, n_bank // 2 - 1] if n_bank > 2 else [0]
    cache = pathlib.Path(tempfile.gettempdir()) / f"mrm_thermal_bank3d_{n_bank}_{int(pitch)}_{h_b}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        dcB, yB, infoB = z["dcB"], z["yB"], json.loads(str(z["info"]))
        infoB["cached"] = True
        log(f"3-D bank: reusing {cache.name}")
    else:
        dcB, fmB, infoB = run_3d(cfg, centres, driven, h_b, mw_c, leads=True, bus=True)
        yB = fmB.step(t_grid)
        np.savez(cache, dcB=dcB, yB=yB, info=np.array(json.dumps(_jsonable(infoB))))
    yA = fmA.step(t_grid)
    pairs = []
    for jcol, i in enumerate(driven):
        for j in range(n_bank):
            if j == i:
                continue
            d = abs(j - i) * pitch
            row = A.row(("cross", "ring_mode_weighted", d))
            y_ax = yA[row, 0]
            y_3d = yB[j, jcol]
            k_s = d_list.index(d)
            y_s = yS[2 + k_s, 0]          # 3-D single ring, same victim position, no victim structures
            pairs.append({"driven": i, "victim": j, "d_um": d, "dc_3d_bank": dcB[j, jcol], "dc_axisym": GA[row, 0],
                          "dc_3d_single": float(dcS[2 + k_s, 0]),
                          "dc_rel_err": dcB[j, jcol] / GA[row, 0] - 1,
                          "step_max_err_frac_of_dc": float(np.max(np.abs(y_3d - y_ax)) / GA[row, 0]),
                          "multibody_dc_rel": dcB[j, jcol] / dcS[2 + k_s, 0] - 1,
                          "multibody_step_err_frac": float(np.max(np.abs(y_3d - y_s)) / dcS[2 + k_s, 0]),
                          "discretization_dc_rel": dcS[2 + k_s, 0] / GA[row, 0] - 1})
    selfs = [{"driven": i, "z_heater_ring_3d_bank": dcB[i, jc], "rel_to_axisym": dcB[i, jc] / ax["z_heater_ring"] - 1,
              "rel_to_3d_single_with_leads": dcB[i, jc] / res["single_ring_with_leads"]["z_heater_ring"] - 1}
             for jc, i in enumerate(driven)]
    nn = [p_ for p_ in pairs if p_["d_um"] == pitch]
    res["bank"] = {"n_rings": n_bank, "driven": driven, "info": infoB, "pairs": pairs, "self": selfs,
                   "max_dc_rel_err_nearest": max(abs(p_["dc_rel_err"]) for p_ in nn),
                   "max_dc_rel_err_all": max(abs(p_["dc_rel_err"]) for p_ in pairs),
                   "max_step_err_frac_nearest": max(p_["step_max_err_frac_of_dc"] for p_ in nn),
                   "max_step_err_frac_all": max(p_["step_max_err_frac_of_dc"] for p_ in pairs),
                   "max_multibody_step_err_frac_all": max(p_["multibody_step_err_frac"] for p_ in pairs),
                   "max_multibody_dc_rel_all": max(abs(p_["multibody_dc_rel"]) for p_ in pairs),
                   "max_discretization_dc_rel_all": max(abs(p_["discretization_dc_rel"]) for p_ in pairs),
                   "metric": "max over t in [10 ns, 10 ms] of |y_bank(t) - y_ROM(t)| / y_ROM(inf) for each crosstalk "
                             "pair; it bounds the error of any output built from the crosstalk residues (a residue-"
                             "by-residue comparison is ill-posed: smooth curves have non-unique real-pole residues)",
                   "decomposition": "dc_rel_err = (1 + discretization_dc_rel)(1 + multibody_dc_rel) - 1: discretization"
                                    " = 3-D single ring (leads, bus, probe-only victim) vs the axisymmetric ROM; "
                                    "multibody = the bank (victim structures, keep-outs, leads, bus) vs that 3-D "
                                    "single ring on the same mesh settings",
                   "target": "superposition error < 10 % on the crosstalk (07 P1-3)"}
    res["bank"]["pass_target"] = bool(res["bank"]["max_step_err_frac_all"] < 0.10)
    res["bank"]["pass_target_multibody_only"] = bool(res["bank"]["max_multibody_step_err_frac_all"] < 0.10)
    # distance interpolation error of the artefact grid (log-log between grid points)
    grid = np.array(CROSS_R_UM)
    Ag = build_ring_model(cfg, ax_scale, mw, mw_c, cross_r=tuple(grid))
    Gg = Ag.fem.dc(Ag.B, Ag.C)
    gvals = np.array([Gg[Ag.row(("cross", "ring_mode_weighted", float(d))), 0] for d in grid])
    interp = {}
    for d in d_list:
        v = float(np.exp(np.interp(np.log(d), np.log(grid), np.log(gvals))))
        interp[str(d)] = {"interp": v, "exact": ax["xt"][d], "rel_err": v / ax["xt"][d] - 1}
    res["distance_interpolation"] = interp
    prev = {}
    _, js_prev = artefact_paths(fam, int(film), out)
    if js_prev.exists():
        prev = json.loads(js_prev.read_text()).get("stages", {}).get("crosscheck_3d", {})
    if not opts.quick and "silicide_arcs" in prev:
        res["silicide_arcs"] = prev["silicide_arcs"]          # unchanged inputs: reuse the earlier 3-D result
    elif not opts.quick:
        # non-axisymmetric family: four silicide arcs (GF PDK) against two full annuli, same total power
        cfs = default_config("silicide_undercut", film, bottom_bc="dirichlet", sub_um=200.0)
        mws = mode_weight(cfs).coarse(0.1)
        arcs = [(0.0, np.radians(140.0)), (np.pi, np.radians(140.0))]
        dcf, fmf, inf_f = run_3d(cfs, [(0.0, 0.0)], [0], levels[-1], mws, leads=False, bus=False)
        dca, fma, inf_a = run_3d(cfs, [(0.0, 0.0)], [0], levels[-1], mws, leads=False, bus=False, arcs=arcs)
        As = build_ring_model(cfs, ax_scale, mode_weight(cfs), mws, cross_r=(pitch,))
        Gs = As.fem.dc(As.B, As.C)
        As.fem.release()
        fmS = As.fem.krylov_modal(As.B, As.C, per_decade=3.0)
        rs = As.row(("self", "ring_mode_weighted"))
        res["silicide_arcs"] = {
            "arcs_deg": [[np.degrees(a), np.degrees(b)] for a, b in arcs], "coverage": 2 * 140.0 / 360.0,
            "z_heater_ring_full_3d": dcf[0, 0], "z_heater_ring_arcs_3d": dca[0, 0],
            "z_heater_ring_axisym": Gs[rs, 0],
            "arcs_vs_full_rel": dca[0, 0] / dcf[0, 0] - 1, "full3d_vs_axisym_rel": dcf[0, 0] / Gs[rs, 0] - 1,
            "t63_full_3d_s": float(fmf.t_frac()[0, 0]), "t63_arcs_3d_s": float(fma.t_frac()[0, 0]),
            "t63_axisym_s": float(fmS.t_frac()[rs, 0]),
            "z_heater_heater_arcs_vs_full_rel": dca[1, 0] / dcf[1, 0] - 1, "info_full": inf_f, "info_arcs": inf_a,
            "note": "the ring observer is the full-ring mode-weighted average; arcs leave the coupler regions free"}
        log(f"3-D silicide arcs: {res['silicide_arcs']['arcs_vs_full_rel']:+.3%} vs full annuli")
    log(f"3-D: single-ring dZ {res['single_ring_no_leads']['rel_diff_z_heater_ring']:+.3%}, leads "
        f"{res['single_ring_with_leads']['lead_effect_z_heater_ring']:+.3%}, bank step err "
        f"{res['bank']['max_step_err_frac_all']:.3%} (multibody {res['bank']['max_multibody_step_err_frac_all']:.3%})")
    update_sidecar(fam, int(film), "crosscheck_3d", res, out)
    sup = float(res["bank"]["max_step_err_frac_all"])
    for f2 in FAMILIES:
        for fl in FILMS:
            if (f2, fl) != (fam, int(film)):
                update_sidecar(f2, fl, "crosscheck_3d", {"see": f"thermal_rom_{fam}_{int(film)}.json",
                                                          "note": "3-D bank run on the default family only"}, out)
            update_sidecar(f2, fl, "validation", {
                "superposition_rel_err": sup,
                "superposition_note": f"max L1 residue error of the {n_bank}-ring 3-D bank (inring_noundercut 160, "
                                      f"pitch {pitch} um, Dirichlet comparison variant) against the single-ring "
                                      f"axisymmetric ROM; measured on the default family only"}, out)
    _write_aux(out, "crosscheck_3d", res)
    return {k: v for k, v in res.items() if k != "bank"} | {"bank_summary": {k: v for k, v in res["bank"].items()
                                                                              if k not in ("pairs",)}}


# =============================================================================================== analytic check
class LayeredAnalytic:
    """Exact Laplace-domain temperature of a planar, laterally infinite, anisotropic multilayer for axisymmetric
    annular sources (independent of the FEM: Hankel transform in r, thermal transmission lines in z).

    layers: [(thickness_m, k_r, k_z, rho_c)] from the bottom up; bottom BC 'dirichlet' or ('robin', h); adiabatic top.
    In the Hankel-Laplace domain each layer has gamma = sqrt((k_r xi^2 + s rho c)/k_z) and characteristic admittance
    Y_c = k_z gamma; admittances transform as Y_in = Y_c (Y_L + Y_c tanh(gamma d)) / (Y_c + Y_L tanh(gamma d)), and the
    temperature propagates away from a source plane as theta_b/theta_a = sech(gamma d) / (1 + (Y_L/Y_c) tanh(gamma d)).
    A uniform annulus a1 < r < a2 has the normalized transform O(xi) = 2 (a2 J1(xi a2) - a1 J1(xi a1)) / (xi (a2^2 -
    a1^2)); the ring-averaged response of a victim annulus at centre distance d carries J0(xi d) (Graf's theorem), so
    H(s) = (1/2 pi) int O_src O_obs J0(xi d) G(xi, s; z_s, z_o) xi dxi, averaged over z_s, z_o in their layers."""

    def __init__(self, layers, bottom="dirichlet", h=None):
        self.layers = [tuple(map(float, L)) for L in layers]
        self.bottom = bottom
        self.h = h
        self.z_if = np.concatenate([[0.0], np.cumsum([L[0] for L in self.layers])])

    def _split(self, planes):
        """Sub-layers between all interfaces and the requested planes: (z0, z1, k_r, k_z, rho_c)."""
        zs = np.unique(np.concatenate([self.z_if, np.asarray(planes, float)]))
        out = []
        for z0, z1 in zip(zs[:-1], zs[1:]):
            i = min(np.searchsorted(self.z_if, 0.5 * (z0 + z1)) - 1, len(self.layers) - 1)
            _, kr, kz, rc = self.layers[i]
            out.append((z0, z1, kr, kz, rc))
        return zs, out

    def green(self, xi, s, z_s, z_o):
        """theta(z_o) / Q for a unit plane source at z_s, arrays over xi (s scalar)."""
        zs, sub = self._split([z_s, z_o])
        n = len(sub)
        g = [np.sqrt((kr * xi ** 2 + s * rc) / kz + 0j) for (_, _, kr, kz, rc) in sub]
        yc = [kz * gg for (_, _, _, kz, _), gg in zip(sub, g)]
        th = [np.tanh(gg * (z1 - z0)) for (z0, z1, *_), gg in zip(sub, g)]
        # admittance looking down at each plane zs[i]
        ydn = [None] * (n + 1)
        if self.bottom == "dirichlet":
            ydn[0] = np.full(xi.shape, 1e30 + 0j)
        else:
            ydn[0] = np.full(xi.shape, self.h + 0j)
        for i in range(n):
            yl = ydn[i]
            ydn[i + 1] = yc[i] * (yl + yc[i] * th[i]) / (yc[i] + yl * th[i])
        yup = [None] * (n + 1)
        yup[n] = np.zeros(xi.shape, complex)
        for i in range(n - 1, -1, -1):
            yl = yup[i + 1]
            yup[i] = yc[i] * (yl + yc[i] * th[i]) / (yc[i] + yl * th[i])
        i_s = int(np.argmin(np.abs(zs - z_s)))
        i_o = int(np.argmin(np.abs(zs - z_o)))
        theta = 1.0 / (ydn[i_s] + yup[i_s])
        if i_o > i_s:
            for i in range(i_s, i_o):
                ch = np.cosh(np.minimum(g[i].real * (zs[i + 1] - zs[i]), 600) + 1j * g[i].imag * (zs[i + 1] - zs[i]))
                theta = theta / (ch * (1 + yup[i + 1] / yc[i] * th[i]))
        elif i_o < i_s:
            for i in range(i_s - 1, i_o - 1, -1):
                ch = np.cosh(np.minimum(g[i].real * (zs[i + 1] - zs[i]), 600) + 1j * g[i].imag * (zs[i + 1] - zs[i]))
                theta = theta / (ch * (1 + ydn[i] / yc[i] * th[i]))
        return theta

    @staticmethod
    def annulus_transform(xi, a1, a2):
        from scipy.special import j1
        x = np.maximum(xi, 1e-30)
        if a1 <= 0:
            return 2 * a2 * j1(x * a2) / (x * a2 ** 2)
        return 2 * (a2 * j1(x * a2) - a1 * j1(x * a1)) / (x * (a2 ** 2 - a1 ** 2))

    def response(self, s, src, obs, d=0.0, n_gauss=4, xi_max=None):
        """H(s) in K/W. src/obs = (a1, a2, z0, z1) in m (uniform annulus volume source / volume-average observer)."""
        from scipy.special import j0
        a_max = max(src[1], obs[1], d)
        t_min = min(src[3] - src[2], obs[3] - obs[2])
        xi_max = xi_max or 60.0 / max(t_min, 1e-9)
        gap = d - (src[1] + obs[1])
        if gap > 0:     # separated annuli: spatial detail finer than gap/200 does not reach the observer
            xi_max = min(xi_max, 200.0 / gap)
        dxi = np.pi / (20.0 * (src[1] + obs[1] + d))
        n_lin = int(min(xi_max, 400.0 / (min(src[1] - src[0], obs[1] - obs[0]) + 1e-12)) / dxi) + 1
        n_lin = min(n_lin, 400000)
        # log lead-in (the stack-depth scale 1/L sets the low-xi shape), then a linear grid that resolves the Bessel
        # oscillations, then a log tail up to the film-thickness cutoff
        x0 = 1e-3 / self.z_if[-1]
        xi = np.concatenate([np.geomspace(x0, dxi, 600, endpoint=False), np.linspace(dxi, n_lin * dxi, n_lin),
                             np.geomspace(n_lin * dxi * 1.001, max(xi_max, n_lin * dxi * 1.01), 2000)])
        w = self.annulus_transform(xi, *src[:2]) * self.annulus_transform(xi, *obs[:2]) * j0(xi * d) * xi / (2 * np.pi)
        gx, gw = np.polynomial.legendre.leggauss(n_gauss)
        zs_pts = src[2] + (gx + 1) / 2 * (src[3] - src[2])
        zo_pts = obs[2] + (gx + 1) / 2 * (obs[3] - obs[2])
        G = np.zeros(xi.shape, complex)
        for zs_, ws_ in zip(zs_pts, gw / 2):
            for zo_, wo_ in zip(zo_pts, gw / 2):
                G += ws_ * wo_ * self.green(xi, s, zs_, zo_)
        return complex(np.trapezoid(w * G, xi))


def layered_check_config(robin: bool):
    """A laterally uniform stack (continuous film layer) used for the FEM-vs-analytic check."""
    cfg = default_config("inring_noundercut", 160.0)
    if not robin:
        cfg.bottom_bc, cfg.sub_um, cfg.die_side_mm = "dirichlet", 200.0, 600.0 * np.sqrt(np.pi) / 1e3
    return cfg


def _layered_rects(cfg: ThermalConfig):
    Rd = cfg.die_radius_um
    R, w, tf = cfg.radius_um, cfg.w, cfg.t_f
    rects = _stack_rects(cfg, keepout_r=None)
    rects += [Rect("film", 0.0, Rd, 0.0, tf, "si_film", 1),
              Rect("src", R - w / 2, R + w / 2, 0.0, tf, "si_film", 2),
              Rect("htr", R - w / 2 - 3.0, R - w / 2 - 2.0, 0.0, tf, "si_film", 2)]
    return rects


def _graded_nodes(breaks, h_fn):
    """1-D nodes through every break point, spacing h_fn(x) inside each interval (stretched to fit exactly)."""
    breaks = np.unique(np.asarray(breaks, float))
    out = [breaks[0]]
    for a, b in zip(breaks[:-1], breaks[1:]):
        xs = [a]
        while xs[-1] < b:
            xs.append(xs[-1] + h_fn(xs[-1]))
        n = max(len(xs) - 1, 1)
        if len(xs) > 2 and (xs[-1] - b) > 0.5 * (xs[-1] - xs[-2]):
            n = max(n - 1, 1)
        xs = np.asarray(xs[:n + 1]) if len(xs) > n else np.asarray(xs)
        u = (xs - a) / (xs[-1] - a) if xs[-1] > a else np.linspace(0, 1, n + 1)
        out.extend(list(a + (b - a) * u[1:]))
    return np.unique(np.round(np.array(out), 9))


def layered_tensor_mesh(cfg: ThermalConfig, rects, scale: float):
    """Structured, interface-aligned P2-ready triangle mesh for the layered check (independent of gmsh)."""
    from skfem import MeshTri
    R, w, tf = cfg.radius_um, cfg.w, cfg.t_f
    r_lo, r_hi = R - w / 2 - 3.2, R + w / 2 + 0.2
    h0r = 0.04 * scale

    def hr(x):
        dist = max(r_lo - x, x - r_hi, 0.0)
        return min(h0r + 0.15 * dist, 0.02 * cfg.die_radius_um + 1.0)
    rb = [0.0, cfg.die_radius_um] + [rc.r0 for rc in rects] + [rc.r1 for rc in rects]
    rn = _graded_nodes([x for x in rb if 0 <= x <= cfg.die_radius_um], hr)
    h0z = 0.02 * scale

    def hz(x):
        dist = max(-x, x - tf, 0.0)
        return min(h0z + 0.25 * dist, 25.0)
    zb = [rc.z0 for rc in rects] + [rc.z1 for rc in rects]
    zn = _graded_nodes(zb, hz)
    m = MeshTri.init_tensor(rn, zn)
    cen = m.p[:, m.t].mean(axis=1)
    return m.p, m.t, classify(rects, cen[0], cen[1])


def stage_analytic(fams, films, opts, out):
    """FEM (this module's axisymmetric machinery) against the layered analytic solution, for DC and frequencies.

    The FEM side uses a structured interface-aligned mesh (not gmsh), so the check is also independent of the mesh
    generator. Dirichlet case: DC and all frequencies. Robin case (finite die with an adiabatic edge against a
    laterally infinite layer): f >= 100 Hz only, where the die-level mode has died out."""
    results = {}
    f_list = [0.0, 10.0, 1e3, 1e4, 1e5, 1e6, 1e7] if not opts.quick else [0.0, 1e4, 1e6]
    dists = [0.0, 20.0, 50.0, 100.0, 300.0] if not opts.quick else [0.0, 50.0]
    scales = (2.0, 1.0, 0.5) if not opts.quick else (2.0,)
    for case in ("dirichlet_sub200", "robin_sub775"):
        robin = case.startswith("robin")
        cfg = layered_check_config(robin)
        mats = material_table(cfg)
        L = []
        for (th, m) in [(cfg.sub_um, "si_sub"), (cfg.box_um, "box"), (cfg.t_f, "si_film"),
                        (cfg.clad_um - cfg.t_f, "clad"), (cfg.beol_lo_um, "beol_lo"), (cfg.beol_hi_um, "beol_hi"),
                        (cfg.pass_um, "pass")]:
            L.append((th * 1e-6, *mats[m]))
        an = LayeredAnalytic(L, "dirichlet" if not robin else "robin", cfg.h_bottom_w_per_m2_k)
        z0 = (cfg.sub_um + cfg.box_um) * 1e-6
        R, w, tf = cfg.radius_um * 1e-6, cfg.w * 1e-6, cfg.t_f * 1e-6
        src = (R - w / 2, R + w / 2, z0, z0 + tf)
        htr = (R - w / 2 - 3e-6, R - w / 2 - 2e-6, z0, z0 + tf)
        rects = _layered_rects(cfg)
        fem_vals, dofs = {}, {}
        for sc in scales:
            p, t, reg = layered_tensor_mesh(cfg, rects, sc)
            fem = AxisymFEM(p, t, reg, rects, mats, cfg.z_bot, cfg.bottom_bc, cfg.h_bottom_w_per_m2_k)
            B = fem.restrict(fem.region_vector({"src": 1.0}))[:, None]
            rows = [fem.region_vector({"src": 1.0}), fem.region_vector({"htr": 1.0})]
            r_c = cfg.radius_um - cfg.w / 2 + (np.arange(8) + 0.5) / 8 * cfg.w
            z_c = (np.arange(4) + 0.5) / 4 * cfg.t_f
            rr, zz = np.meshgrid(r_c, z_c, indexing="ij")
            for d in dists[1:]:
                pts, wts = _annulus_points(rr.ravel(), zz.ravel(), np.ones(rr.size), d, nphi=64)
                rows.append(fem.point_vector(pts, wts))      # area weights r dr dphi (in _annulus_points)
            C = np.vstack([fem.restrict(r) for r in rows])
            fem_vals[sc] = fem.freq(B, C, f_list)[:, 0, :]
            dofs[sc] = int(fem.n)
            log(f"analytic check {case}: FEM scale {sc} done ({fem.n} dofs)")
        recs = []
        Hf = fem_vals[scales[-1]]
        for k, f in enumerate(f_list):
            if robin and f < 100.0:
                continue
            s = 2j * np.pi * f
            for i, (nm, obs, d) in enumerate([("self", src, 0.0), ("heater_annulus", htr, 0.0)]
                                            + [(f"victim@{int(dd)}um", src, dd * 1e-6) for dd in dists[1:]]):
                Ha = an.response(s, src, obs, d=d)
                rec = {"case": case, "f_hz": f, "pair": nm, "analytic_k_per_w": [Ha.real, Ha.imag],
                       "fem_k_per_w": [Hf[i, k].real, Hf[i, k].imag],
                       "rel_err": float(abs(Hf[i, k] - Ha) / abs(Ha)),
                       "rel_err_of_self_dc": float(abs(Hf[i, k] - Ha) / abs(fem_vals[scales[-1]][0, 0]))}
                if len(scales) > 1:
                    rec["fem_mesh_change"] = float(abs(fem_vals[scales[-2]][i, k] - Hf[i, k]) / abs(Hf[i, k]))
                recs.append(rec)
        results[case] = {"max_rel_err": max(r["rel_err"] for r in recs),
                         "max_err_of_self_dc": max(r["rel_err_of_self_dc"] for r in recs),
                         "max_rel_err_self_and_heater": max(r["rel_err"] for r in recs if r["pair"] in
                                                            ("self", "heater_annulus")),
                         "fem_dofs": dofs, "records": recs}
        log(f"analytic check {case}: max rel err {results[case]['max_rel_err']:.2e}, "
            f"max err / self DC {results[case]['max_err_of_self_dc']:.2e}")
    payload = {"method": "FEM (structured interface-aligned P2 mesh) vs exact layered Hankel-Laplace solution "
                         "(LayeredAnalytic); laterally uniform stack with a continuous 160 nm Si film; annular volume "
                         "source in the film; observers: the same annulus, an annulus 2-3 um inside, victim annuli "
                         "at distance d", "results": results}
    for fam in fams:
        for film in films:
            update_sidecar(fam, film, "crosscheck_analytic", payload, out)
    _write_aux(out, "crosscheck_analytic", payload)
    return {k: {kk: vv for kk, vv in v.items() if kk != "records"} for k, v in results.items()}


def _write_aux(out, name: str, payload: dict):
    """Stage results are recorded in the sidecars; this keeps a copy outside the artefact folder for debugging."""
    d = pathlib.Path(tempfile.gettempdir()) / "mrm_thermal_stage_cache"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(_jsonable(payload), indent=1))


# =============================================================================================== VAL-8 (imec)
# Coenen et al., "Thermal modelling of silicon photonic ring modulator with substrate undercut", JLT 40(13), 2022
# (author copy, read 2026-10-09). Printed: 15 um diameter C-band depletion ring modulator; W metal heater above the Si
# waveguide in the oxide below M1/M2 Cu; Table I (k W/(m K), rho kg/m^3, c_p J/(kg K)): Si 150/2320/700, BOX and SiO2
# 1/2200/635, W 100/10930/134, Cu 400/8960/390, passivation 10/2200/2300; top convection 10 W/(m^2 K), bottom
# 1e11 W/(m^2 K) (isothermal), adiabatic sides, domain 16 ring diameters; model with M1, M2 and vias. Results: heater
# R_h 5.40 C/mW, waveguide eta_h 3.16 C/mW, measured 222 +/- 10 pm/mW (2.99 C/mW), full UCUT 843 +/- 55 pm/mW;
# 63 % time constants heater 4.92 / waveguide 10.38 us (model), 5 / 10 us (measured); full UCUT (27 x 45 um^2) 18.1 /
# 38.1 us (simulated); no Cu in the model: eta_h +18 % (no UCUT). [CO22-TCPMT] two-pole fit of a different imec ring
# (14 um double-ring filter, simulated): 3.18 us (42.7 %) + 13.01 us (57.3 %).
# Frozen targets: mrm/validation/targets/validation.yaml VAL-8a..e. Rule from the validation owner: replicate with OUR
# materials (mrm.materials), not imec's calibrated Table I (that set is reported as a variant only).
# NOT printed (assumed here, swept): Si 220 nm / rib 450 nm / slab 70 nm (iSiPP), BOX 2 um, heater 0.6 um wide
# (imec CO-EM W heater), 0.1 um thick, 0.45 um above the Si, oxide to M1 1.2 um, M1-M2 level 1.5 um, passivation
# 1 um, substrate 240 um, Cu effective fill 0.15 in the M1-M2 level outside a 2 um keep-out.
IMEC_TARGETS = {"VAL-8a_tau_heater_us": 5.0, "VAL-8b_tau_wg_us": 10.0, "VAL-8c_eff_no_ucut_pm_per_mw": 222.0,
                "VAL-8d_eff_full_ucut_pm_per_mw": 843.0,
                "VAL-8e_two_pole": {"tau_us": [3.18, 13.01], "share": [0.427, 0.573]},
                "tolerance": {"VAL-8a": 0.30, "VAL-8b": 0.30, "VAL-8c": 0.25, "VAL-8d": 0.30, "VAL-8e": 0.30},
                "imec_model": {"tau_heater_us": 4.92, "tau_wg_us": 10.38, "eta_h_c_per_mw": 3.16, "r_h_c_per_mw": 5.40,
                               "coupling": 0.584, "ucut_tau_heater_us": 18.1, "ucut_tau_wg_us": 38.1,
                               "ucut_eta_h_c_per_mw": 9.73, "no_cu_eta_ratio": 1.18}}
IMEC_UCUT = (float(np.sqrt(27.0 * 45.0 / np.pi)), 17.5)   # equal-area cylinder of the 27 x 45 um^2 UCUT, depth


def imec_config(**over) -> ThermalConfig:
    base = dict(family="metal_above", film_nm=220.0, lam_m=1.55e-6, width_nm=450.0, slab_nm=70.0, slab_in_um=1.5,
                slab_out_um=1.5, box_um=2.0, sub_um=240.0, die_side_mm=0.24, bottom_bc="dirichlet",
                clad_um=1.2 + 0.22, beol_lo_um=1.5, beol_hi_um=0.5, pass_um=1.0, fill_lo=0.15, fill_hi=0.0,
                via_ratio=0.1, keepout_um=2.0, metal_gap_um=0.45, metal_width_um=0.6, metal_thick_um=0.1,
                metal="w_film", tbr_shell_um=0.0, mode_weight="femwell", undercut_margin_um=0.0)
    base.update(over)
    return ThermalConfig(**base)


def imec_material_table(cfg: ThermalConfig) -> dict:
    """Table I of CO22-JLT (variant only; the nominal replication uses mrm.materials)."""
    rc = lambda rho, cp: rho * cp  # noqa: E731
    si = (150.0, 150.0, rc(2320, 700))
    ox = (1.0, 1.0, rc(2200, 635))
    cu = (400.0, rc(8960, 390))
    f = cfg.fill_lo
    beol = (f * cu[0] + (1 - f) * ox[0], cfg.via_ratio * f * cu[0] + ox[0] / max(1 - f, 1e-6),
            f * cu[1] + (1 - f) * ox[2])
    return {"si_sub": si, "box": ox, "clad": ox, "si_film": si, "si_slab": si, "si_heater": si, "si_cmos": si,
            "beol_lo": beol, "beol_hi": ox, "beol_lo_ko": ox, "beol_hi_ko": ox, "pass": (10.0, 10.0, rc(2200, 2300)),
            "air": (cfg.air_k_w_per_m_k, cfg.air_k_w_per_m_k, AIR_RHO_C_J_PER_M3_K),
            "metal": (100.0, 100.0, rc(10930, 134)), "tbr": ox}


def _neff_femwell(cfg: ThermalConfig, lam_m: float, t_k: float) -> float:
    """n_eff of the rib TE0 (bent at R) from femwell with mrm.materials indices at (lambda, T)."""
    c2 = dataclasses.replace(cfg, lam_m=lam_m, t_k=t_k, mode_weight="femwell")
    return float(_mode_weight_impl(c2).info["n_eff"])


def _dlambda_dt_arrays(cfg: ThermalConfig) -> dict:
    lam, T = cfg.lam_m, cfg.t_k
    n0 = _neff_femwell(cfg, lam, T)
    dT = 10.0
    dn_dT = (_neff_femwell(cfg, lam, T + dT) - _neff_femwell(cfg, lam, T - dT)) / (2 * dT)
    dl = 5e-9
    dn_dl = (_neff_femwell(cfg, lam + dl, T) - _neff_femwell(cfg, lam - dl, T)) / (2 * dl)
    ng = n0 - lam * dn_dl
    a_l = float(mt.alpha_l_si_per_k(T))
    return {"n_eff": n0, "n_g": ng, "dneff_dt_per_k": dn_dT, "alpha_l_per_k": a_l,
            "dlambda_dt_pm_per_k": lam / ng * (dn_dT + n0 * a_l) * 1e12,
            "dlambda_dt_thermo_optic_only_pm_per_k": lam / ng * dn_dT * 1e12,
            "method": "femwell bent TE0 (R = %.2f um), central differences +/-10 K and +/-5 nm, mrm.materials" %
                      cfg.radius_um}


def mode_dlambda_dt(cfg: ThermalConfig) -> dict:
    """Our own dlambda/dT of a ring cross-section (subprocess, memory isolation)."""
    import concurrent.futures as cf
    import multiprocessing as mp
    with cf.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(_dlambda_dt_arrays, cfg).result()


def imec_run(cfg: ThermalConfig, scale: float, materials: str = "mrm", undercut=None, mw: ModeWeight | None = None,
             two_pole: bool = True) -> dict:
    rects = ring_geometry(cfg)
    if undercut is not None:
        r_uc, depth = undercut
        rects.append(Rect("undercut", 0.0, r_uc, -cfg.box_um - depth, -cfg.box_um, "air", 6))
    spec = mesh_spec_for(cfg, rects, scale, "ring")
    p, t, reg = mesh_axisym(rects, spec)
    mats = imec_material_table(cfg) if materials == "imec_table1" else material_table(cfg)
    fem = AxisymFEM(p, t, reg, rects, mats, cfg.z_bot, cfg.bottom_bc, cfg.h_bottom_w_per_m2_k)
    b_h = fem.restrict(fem.region_vector({"heater0": 1.0}))
    rows = [fem.region_vector({"rib": 1.0}), fem.region_vector({"heater0": 1.0})]   # imec: rib volume average
    if mw is not None:
        rows.append(fem.point_vector(np.vstack([mw.r, mw.z]), mw.w_obs * mw.r))
    C = np.vstack([fem.restrict(r) for r in rows])
    B = b_h[:, None]
    G = fem.dc(B, C)
    fem.release()
    fm = fem.krylov_modal(B, C, per_decade=3.0)
    t63 = fm.t_frac()
    out = {"eta_h_rib_c_per_mw": G[0, 0] * 1e-3, "r_h_c_per_mw": G[1, 0] * 1e-3,
           "tau_wg_rib_us": t63[0, 0] * 1e6, "tau_heater_us": t63[1, 0] * 1e6, "dofs": int(fem.n), "scale": scale,
           "materials": materials}
    iw = 2 if mw is not None else 0
    out["eta_h_c_per_mw"] = G[iw, 0] * 1e-3
    out["tau_wg_us"] = t63[iw, 0] * 1e6
    out["coupling"] = G[iw, 0] / G[1, 0]
    if two_pole:
        tt = np.logspace(-8, -2, 241)
        ff = F_FIT_HZ[F_FIT_HZ <= 1e7]
        y = fm.step(tt)[iw, 0]
        taus2, res2, _ = mor.fit_common_foster(ff, fm.freq(ff)[iw], tt, y[None], np.array([G[iw, 0]]), 2, 1e-6,
                                               1e-4, polish=True, vf_iter=20)
        order = np.argsort(taus2)
        out["two_pole_fit"] = {"tau_us": (taus2[order] * 1e6).tolist(), "share": (res2[0][order] / G[iw, 0]).tolist()}
    return out


def stage_val8(fams, films, opts, out):
    """07 VAL-8 (frozen as VAL-8a..e): replicate the imec metal-heater ring with our materials on the printed stack."""
    scales = (2.0, 1.414, 1.0) if not opts.quick else (3.0,)
    cfg0 = imec_config(mode_weight="uniform_core" if opts.quick else "femwell")
    mw = mode_weight(cfg0) if not opts.quick else None
    dl = mode_dlambda_dt(cfg0) if not opts.quick else {"dlambda_dt_pm_per_k": 74.2, "method": "quick: imec 74.2"}
    log(f"VAL-8 imec rib (our materials): {dl}")
    levels = {}
    for sc in scales:
        levels[sc] = imec_run(cfg0, sc, mw=mw)
        log(f"VAL-8 nominal scale {sc}: {levels[sc]}")
    nom = levels[scales[-1]]
    uc = imec_run(cfg0, scales[-1], undercut=IMEC_UCUT, mw=mw)
    s_ = dl["dlambda_dt_pm_per_k"]
    pred = {"VAL-8a_tau_heater_us": nom["tau_heater_us"], "VAL-8b_tau_wg_us": nom["tau_wg_us"],
            "VAL-8c_eff_no_ucut_pm_per_mw": nom["eta_h_c_per_mw"] * s_,
            "VAL-8d_eff_full_ucut_pm_per_mw": uc["eta_h_c_per_mw"] * s_,
            "VAL-8e_two_pole": nom.get("two_pole_fit")}
    tgt = IMEC_TARGETS
    comp = {}
    for k in ("VAL-8a_tau_heater_us", "VAL-8b_tau_wg_us", "VAL-8c_eff_no_ucut_pm_per_mw",
              "VAL-8d_eff_full_ucut_pm_per_mw"):
        tid = k[:6]
        rel = pred[k] / tgt[k] - 1
        comp[tid] = {"predicted": pred[k], "target": tgt[k], "rel_err": rel, "tolerance": tgt["tolerance"][tid],
                     "pass": bool(abs(rel) <= tgt["tolerance"][tid])}
    if pred["VAL-8e_two_pole"]:
        tp = pred["VAL-8e_two_pole"]
        comp["VAL-8e"] = {"predicted": tp, "target": tgt["VAL-8e_two_pole"], "status": "cross_check",
                          "rel_err_tau": [tp["tau_us"][i] / tgt["VAL-8e_two_pole"]["tau_us"][i] - 1 for i in range(2)]}
    comp["mesh_change_last_level"] = {k: levels[scales[-1]][k] / levels[scales[-2]][k] - 1
                                      for k in ("tau_heater_us", "tau_wg_us", "eta_h_c_per_mw")} \
        if len(scales) > 1 else None
    variants = {"full_ucut": uc}
    if not opts.quick:
        sweeps = {"imec_table1_materials": dict(), "heater_gap_0p3um": dict(metal_gap_um=0.3),
                  "heater_gap_0p8um": dict(metal_gap_um=0.8), "heater_width_1p5um": dict(metal_width_um=1.5),
                  "heater_width_3um": dict(metal_width_um=3.0), "substrate_100um": dict(sub_um=100.0),
                  "substrate_750um": dict(sub_um=750.0, die_side_mm=0.75), "no_beol_cu": dict(fill_lo=0.0),
                  "beol_cu_fill_0p3": dict(fill_lo=0.3), "beol_keepout_0p5um": dict(keepout_um=0.5),
                  "with_metal_tbr": dict(tbr_shell_um=0.02), "beol_4um": dict(beol_lo_um=4.0)}
        for name, over in sweeps.items():
            mats = "imec_table1" if name == "imec_table1_materials" else "mrm"
            variants[name] = imec_run(imec_config(**over), scales[-1], materials=mats, mw=mw, two_pole=False)
            v = variants[name]
            log(f"VAL-8 {name}: tau_h {v['tau_heater_us']:.2f} tau_wg {v['tau_wg_us']:.2f} us, "
                f"eta {v['eta_h_c_per_mw']:.2f} C/mW")
        variants["imec_table1_full_ucut"] = imec_run(imec_config(), scales[-1], materials="imec_table1",
                                                     undercut=IMEC_UCUT, mw=mw, two_pole=False)
        # bounded search over the geometry imec does not print: which plausible stacks reproduce the measured tau
        # AND efficiency at once? (diagnostic; it does not feed the Lightmatter model)
        grid = []
        for wh in (0.6, 1.5, 3.0):
            for gap in (0.3, 0.8):
                for ko in (0.5, 5.0):
                    for blo in (1.5, 4.0):
                        over = dict(metal_width_um=wh, metal_gap_um=gap, keepout_um=ko, beol_lo_um=blo,
                                    metal_thick_um=0.15)
                        r = imec_run(imec_config(**over), 2.0, mw=mw, two_pole=False)
                        r["override"] = over
                        r["score"] = float(np.sqrt(np.mean(np.log([
                            r["tau_heater_us"] / tgt["VAL-8a_tau_heater_us"], r["tau_wg_us"] / tgt["VAL-8b_tau_wg_us"],
                            r["eta_h_c_per_mw"] * s_ / tgt["VAL-8c_eff_no_ucut_pm_per_mw"]]) ** 2)))
                        grid.append(r)
        grid.sort(key=lambda r: r["score"])
        variants["geometry_search"] = {
            "n": len(grid), "tau_wg_us_range": [min(g["tau_wg_us"] for g in grid), max(g["tau_wg_us"] for g in grid)],
            "tau_heater_us_range": [min(g["tau_heater_us"] for g in grid), max(g["tau_heater_us"] for g in grid)],
            "eff_pm_per_mw_range": [min(g["eta_h_c_per_mw"] for g in grid) * s_,
                                    max(g["eta_h_c_per_mw"] for g in grid) * s_],
            "best": [{k: g[k] for k in ("override", "score", "tau_heater_us", "tau_wg_us", "eta_h_c_per_mw",
                                        "r_h_c_per_mw")} for g in grid[:5]]}
        log(f"VAL-8 geometry search: {variants['geometry_search']}")
    payload = {"schema": SCHEMA_VAL8, "target_ids": ["VAL-8a", "VAL-8b", "VAL-8c", "VAL-8d", "VAL-8e"],
               "source": "Coenen et al., JLT 40(13) 4357 (2022), Tables I-IV, Figs. 6, 10, 16; CO22-TCPMT Table I",
               "targets": tgt, "predicted": pred, "comparison": comp, "nominal": nom, "mesh_levels": levels,
               "dlambda_dt": dl, "variants": variants,
               "assumptions": "replication with mrm.materials on the printed geometry and boundary conditions; the "
                              "dimensions imec does not print are assumed (imec_config) and swept; tau = 63 % time of "
                              "the step response; heater node = heater_resistor observer, waveguide = femwell TE0 "
                              "mode-weighted ring observer (the rib volume average is also given)",
               "config": imec_config().to_dict(), "tool_versions": tool_versions(),
               "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    path = pathlib.Path(out) / "thermal_val8_imec.json"
    path.write_text(json.dumps(_jsonable(payload), indent=1))
    log(f"VAL-8: {comp}")
    return comp


# =============================================================================================== CLI
STAGES = ("extract", "uncertainty", "analytic", "3d", "val8", "propagate")


def stage_propagate(fams, films, opts, out):
    """No solver: copy the family-independent results of the default sidecar (the analytic check and the 3-D bank
    superposition error, both run on inring_noundercut 160) into the other sidecars."""
    # refresh the rom_dc_scale diagnostic with whatever modes artefacts exist now (H_solver from modes_rib<film>)
    for fam in FAMILIES:
        for film in FILMS:
            _, js = artefact_paths(fam, film, out)
            if js.exists():
                d = json.loads(js.read_text())
                z = d["summary"]["z_heater_ring_k_per_mw"] * 1e3
                update_sidecar(fam, film, "top", {"rom_dc_scale": rom_dc_scale(z, film)}, out)
    _, js0 = artefact_paths("inring_noundercut", 160, out)
    if not js0.exists():
        return {"propagated": 0}
    d0 = json.loads(js0.read_text())
    st0 = d0.get("stages", {})
    n = 0
    for fam in FAMILIES:
        for film in FILMS:
            if (fam, film) == ("inring_noundercut", 160):
                continue
            if "crosscheck_analytic" in st0:
                n += update_sidecar(fam, film, "crosscheck_analytic", st0["crosscheck_analytic"], out)
            if "crosscheck_3d" in st0:
                update_sidecar(fam, film, "crosscheck_3d", {"see": "thermal_rom_inring_noundercut_160.json",
                                                             "note": "3-D bank run on the default family only"}, out)
            if d0.get("validation", {}).get("superposition_rel_err") is not None:
                update_sidecar(fam, film, "validation", {
                    "superposition_rel_err": d0["validation"]["superposition_rel_err"],
                    "superposition_note": d0["validation"].get("superposition_note")}, out)
    return {"propagated": n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mrm.extraction.thermal_fem", description=__doc__.split("\n")[0])
    ap.add_argument("--family", choices=FAMILIES, action="append")
    ap.add_argument("--film", type=int, choices=FILMS, action="append")
    ap.add_argument("--stage", choices=STAGES + ("all",), action="append")
    ap.add_argument("--quick", action="store_true", help="cheap run used by the tests (writes to a temp dir)")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="output directory (default mrm/data/extracted)")
    a = ap.parse_args(argv)
    fams = a.family or list(FAMILIES)
    films = a.film or list(FILMS)
    stages = a.stage or ["all"]
    if "all" in stages:
        stages = list(STAGES)
    opts = RunOptions.quick_options() if a.quick else RunOptions()
    out = a.out or (pathlib.Path(tempfile.gettempdir()) / "mrm_thermal_quick" if a.quick else OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    log(f"stages {stages}, families {fams}, films {films}, out {out}")
    results = {}
    for st in stages:
        fn = globals()[f"stage_{st.replace('3d', 'three_d')}"]
        results[st] = fn(fams, films, opts, out)
    log("done")
    print(json.dumps(_jsonable({k: v for k, v in results.items() if k != "extract"}), indent=1)[:4000])
    return 0


def stage_extract(fams, films, opts, out):
    res = {}
    for fam in fams:
        for film in films:
            res[f"{fam}_{film}"] = extract(fam, film, opts, out)
    return res


if __name__ == "__main__":
    sys.exit(main())
