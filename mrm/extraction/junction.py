"""Junction extraction: carriers against bias in the MRM rib cross-section, turned into the junction LUT.

ARCHITECTURE section 5.4 (schema ``junction/1``), 07 P1-7 and P0-2. Physics: docs/research/03 section 5 (depletion,
C(V), Sacher & Poon small-signal response), 02 sections 7 and 11 (Nedeljkovic 2011 plasma dispersion at 1.31 um, n_i,
V_bi), 05 C7c (DEVSIM; its default permittivity is overridden to 11.7 eps0 here).

Pipeline, per cross-section hypothesis (``rib160`` default, ``rib200``):

1. DEVSIM 2-D drift-diffusion of a lateral PN junction in the rib + slab, with the SiO2 cladding as a Poisson region
   (fringe capacitance), doping-dependent (Arora) mobility, ohmic contacts on the slab ends, bias -0.5 V (forward) to
   +4.0 V reverse. Small-signal AC at every bias gives C(V) (1 MHz) and the series resistance R(V) = Re Z (20 GHz).
2. The TE0 mode of the same cross-section with femwell (bent at r_ref, conformal map). The INDEX weight density
   w = c eps0 n (1 + x/R)^2 |E|^2 / P (integrates to dn_eff/dn_i; 03 verification log: not the power fraction) is
   evaluated at the DEVSIM nodes, and
       dn_eff(V)       = sum_i w_i [dn(N(V)) - dn(N(0))]_i A_i
       alpha_fca(V)    = sum_i w_i  alpha(N(V))_i A_i               (absolute, full circumference)
   with dn, alpha from mrm.materials.plasma_dispersion (Nedeljkovic 2011 default).
3. C(V) per unit length (AC, cross-checked by dQ/dV); r_s(V) per unit length from the doped-region sheet resistances
   (analytic) and from the DEVSIM AC impedance; plus an assumed contact resistance.
4. Cross-checks: the abrupt depletion approximation (width, C, dn_eff) against DEVSIM; the first-order overlap
   against a direct femwell re-solve with the perturbed complex permittivity; femwell against the tidy3d local mode
   solver; mesh convergence of both solvers (Richardson error estimates in the sidecar).
5. Reports for the integrator: the fill fraction that reproduces CAL-5 for every n_g hypothesis that matters; the EO
   bandwidth at -1 V and -2 V (RC x exact TCMT optical response) against CAL-10 and the held-out VAL-3; the loss
   budget of 07 P0-2 (absolute alpha_FCA against the intrinsic loss implied by Q_i 7000-8400, hence eta).

Geometry and doping are ASSUMPTIONS (Lightmatter publishes neither); every one is swept and its Jacobian recorded.
Units: SI in the artefact (m, F/m, ohm*m, 1/m); the solvers work in um (femwell) and cm (DEVSIM) internally.

Run (always through the memory guard):
    .venv/bin/python tools/memguard.py --limit-gb 3 --heavy -- .venv/bin/python -m mrm.extraction.junction [--quick]
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import datetime as _dt
import hashlib
import io
import json
import math
import os
import pathlib
import platform
import sys
import tempfile
import time
import warnings

import numpy as np

import mrm.materials as M
from mrm.units import C0, EPS0, K_B, Q_E

SCHEMA = "junction/1"
LAMBDA0_M = 1.31e-6
T_REF_K = 298.15                              # 25 C, the reference of the modes artefacts (common.T_REF_C)
V_MIN, V_MAX = -0.5, 4.0                      # reverse voltage grid (forward negative), ARCHITECTURE 5.4
V_STEP_FULL, V_STEP_QUICK = 0.05, 0.25
EPS_SI_R = 11.7                               # static permittivity of Si (05 C7c: override DEVSIM's 11.1)
EPS_SIO2_R = 3.9
OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "extracted"

# ------------------------------------------------------------------------------------------- calibration targets
# Only the numbers the reports compare against. Values from docs/research/07 section 4.3/4.4 (CAL-5, CAL-10, VAL-3)
# and 03 section 5.3 (the G25 read-off ladder behind CAL-5).
CAL5_V = (0.8, 1.2, 1.6)                      # reverse bias, V
CAL5_SHIFT_PM = (56.4, 66.8, 84.4)            # cumulative red shift, pm (G25 read-off, textbook MOD_SHIFT_PM)
CAL5_AVG_PM_PER_V = 52.8                      # average over 0 to -1.6 V (CAL-5)
CAL10_GHZ, CAL10_SIGMA_GHZ = 30.9, 2.0        # EO bandwidth at -1 V (G25, population)
VAL3_GHZ, VAL3_TOL_GHZ = 35.4, 3.0            # EO bandwidth at -2 V (held out; reported, never fitted)
Q_I_RANGE = (7000.0, 8400.0)                  # Q_i = 2 Q_L near critical coupling (07 D1, P0-2)
Q_L_RANGE = (3500.0, 4200.0)                  # CAL-1
P_RING_R_UM = 7.5                             # "~15 um diameter" (LM26-S, HC) and the GF PDK radius (01 section 2.1)
FSR_HYPOTHESES_THZ = {"window_B": (1.80, 1.72, 1.88), "window_A": (1.40, 1.32, 1.48),
                      "stress_G25_cluster": (1.53, 1.52, 1.54), "stress_PDK_size": (1.60, 1.58, 1.63)}
H86_PM_PER_K = 85.9                           # the printed ~15 GHz/K at 1310 nm (07 D2)


# --------------------------------------------------------------------------------------------- assumptions
@dataclasses.dataclass(frozen=True)
class XSec:
    """Rib cross-section (ASSUMED: 45SPCLO prints a 160 nm photonics film and a partial etch, not its depth; the
    400 nm width is the Lightmatter render (01 section 2.1). r_ref_um is the bend radius of the mode solve."""
    name: str
    width_nm: float
    film_nm: float
    slab_nm: float
    sidewall_deg: float = 90.0
    box: str = "thermal"
    clad: str = "beol"
    r_ref_um: float = P_RING_R_UM

    def as_dict(self):
        return dataclasses.asdict(self)


XSECS = {
    "rib160": XSec("rib160", 400.0, 160.0, 60.0),
    "rib200": XSec("rib200", 400.0, 200.0, 80.0),
}
GEOMETRY_RANGES_NM = {"width_nm": (360.0, 450.0), "film_nm": (155.0, 205.0), "slab_nm": (40.0, 100.0)}
GEOMETRY_PROVENANCE = {
    "width_nm": "assumed: Lightmatter render 400 x 200 nm (SMTAI, 01 section 2.1); BU 45CLO rib 410 nm",
    "film_nm": "printed for 45SPCLO (160 nm, GF-IMAPS); 200 nm is the D21 alternative",
    "slab_nm": "assumed: partial-etch depth not public (01 section 4.2); slab/film ~0.38-0.40, cf. 90/220",
    "r_ref_um": "the modes artefact's r_ref (self-consistent radius for the printed 1.8 THz design FSR) when it "
                "exists, else the same self-consistency computed here (07 P1-6)",
}
FSR_DESIGN_HZ = 1.8e12


def resolve_r_ref(xs: XSec, d: Doping, log=print, res_um=0.02):
    """r_ref of the bent mode: read from modes_<xsec>.json (sha256-checked) if present, else the self-consistent
    radius r = c/(2 pi n_g,bent(r) FSR_design) iterated here (two passes are enough to 1e-4 um)."""
    p = OUT_DIR / f"modes_{xs.name}.json"
    sc = OUT_DIR / f"modes_{xs.name}.sidecar.json"
    if p.exists() and sc.exists():
        try:
            side = json.loads(sc.read_text())
            if side.get("sha256") == _sha256(p):
                r = float(json.loads(p.read_text())["r_ref_um"])
                return r, {"source": f"modes_{xs.name}.json", "sha256": side["sha256"]}
        except Exception as exc:          # fall back to the local computation
            log(f"  modes artefact unreadable ({exc!r}); computing r_ref here")
    hist = []
    mp = _isolated(mode_properties, xs, d, res_um=res_um, radius_um=np.inf, with_dn_check=False)
    r = C0 / (2 * math.pi * mp["n_g"] * FSR_DESIGN_HZ) * 1e6
    hist.append({"r_um": r, "n_g": mp["n_g"], "from": "straight"})
    for _ in range(2):
        mp = _isolated(mode_properties, dataclasses.replace(xs, r_ref_um=r), d, res_um=res_um, radius_um=r,
                       with_dn_check=False)
        r_new = C0 / (2 * math.pi * mp["n_g"] * FSR_DESIGN_HZ) * 1e6
        hist.append({"r_um": r_new, "n_g": mp["n_g"], "from": f"bent at {r:.5f}"})
        if abs(r_new - r) < 1e-4:
            r = r_new
            break
        r = r_new
    return r, {"source": "self-consistent 1.8 THz FSR (junction.py; modes artefact absent)", "iterations": hist,
               "res_um": res_um}



@dataclasses.dataclass(frozen=True)
class Doping:
    """PN doping of the rib (ASSUMED; no 45SPCLO or Lightmatter value is public, 02 open question 4).

    ``topology="lateral"``: p body (N_A) on x < x_j, n body (N_D) on x > x_j, x_j = ``junction_offset_nm``.
    ``topology="lshape"``: the n body is the top layer y >= ``y_junction_nm`` (a shallow counter-doping implant,
    which 45CLO offers: 01 section 4.2, BU-45CLO) joined to the n side of the rib for x >= x_j; the p body is the
    rest (bottom-left). The junction is then horizontal across the rib from the p-side sidewall to x_j and vertical
    from x_j down to the BOX. Both: intermediate P+/N+ from ``plus_gap_nm`` outside the rib edge, contact P++/N++ from
    ``pp_gap_nm``; ohmic contacts on the slab ends at ``contact_gap_nm`` outside the rib edge. Steps are error
    functions of width ``grading_nm`` (junction) and ``plus_grading_nm`` (implant edges). The p side faces the ring
    centre (the bend pushes the mode toward the n side), see ``outer_side``."""
    topology: str = "lateral"
    y_junction_nm: float = 90.0
    na_cm3: float = 2.0e18
    nd_cm3: float = 2.0e18
    junction_offset_nm: float = 0.0
    grading_nm: float = 2.0
    plus_gap_nm: float = 250.0
    n_plus_cm3: float = 1.0e19
    pp_gap_nm: float = 600.0
    n_pp_cm3: float = 1.0e20
    contact_gap_nm: float = 1000.0
    plus_grading_nm: float = 20.0
    outer_side: str = "n"
    contact_rho_ohm_cm2: float = 1.0e-7
    contact_width_nm: float = 300.0

    def as_dict(self):
        return dataclasses.asdict(self)


DOPING_NOMINAL = Doping()
TOPOLOGIES = ("lshape", "lateral")
DEFAULT_TOPOLOGY = "lshape"


def nominal_doping(xs_name: str, topology: str = DEFAULT_TOPOLOGY) -> Doping:
    """Nominal (assumed) doping for a cross-section and a junction topology. L-shape: the shallow n layer starts
    30 nm above the slab top (so the p-side slab stays p) and the vertical leg sits 100 nm inside the n-side rib edge
    (a 100 nm wide n path from the top layer to the n slab)."""
    xs = XSECS[xs_name]
    if topology == "lateral":
        return dataclasses.replace(DOPING_NOMINAL, topology="lateral", junction_offset_nm=0.0)
    if topology == "lshape":
        return dataclasses.replace(DOPING_NOMINAL, topology="lshape", junction_offset_nm=xs.width_nm / 2 - 100.0,
                                   y_junction_nm=xs.slab_nm + 30.0)
    raise ValueError(f"unknown topology {topology!r}")


FILL_NOMINAL = 0.85               # assumed junction coverage of the ring (contact / heater breaks); range 0.7-0.95
SELECT_LEVELS = (5e17, 7e17, 1e18, 1.5e18, 2e18, 3e18, 5e18)


def select_doping(xs: XSec, base: Doping, wf, n_g, *, log=print, scale=2.0, fill_target=FILL_NOMINAL,
                  levels=SELECT_LEVELS):
    """Nominal symmetric body doping of a topology: the N_A = N_D at which the CAL-5 endpoint fill equals
    ``fill_target`` (log-linear interpolation between swept levels, rounded to 2 significant digits). Doping and
    fill are not separately identifiable from CAL-5; this fixes the doping inside the published range so that the
    calibrated fill stays physical, and the Jacobian (per decade) carries the rest."""
    vj = np.array([0.0, 0.8, 1.2, 1.6])
    rows = []
    for N in levels:
        d = dataclasses.replace(base, na_cm3=N, nd_cm3=N)
        rr = _devsim_and_overlap(xs, d, vj, wf, scale, ac=False)
        fc = fill_for_cal5(vj, rr["dneff"], n_g)
        rows.append({"n_cm3": N, "fill_endpoint": fc["fill_endpoint"], "lsq_rms_pm": fc["lsq_rms_pm"],
                     "alpha_fca_0v_per_m": float(rr["alpha"][0]),
                     "eff_avg_pm_per_v_fill1": float(shift_pm(rr["dneff"][-1], n_g)) / 1.6})
        log(f"  select: N {N:.2e}: fill {fc['fill_endpoint']:.3f}, alpha0 {rr['alpha'][0]:.1f} /m")
        if fc["fill_endpoint"] < fill_target and len(rows) >= 2:
            break
    f = np.array([r["fill_endpoint"] for r in rows])
    lg = np.log10([r["n_cm3"] for r in rows])
    if f.min() > fill_target:
        n_sel, status = rows[-1]["n_cm3"], "target not reached: highest level used"
    elif f.max() < fill_target:
        n_sel, status = rows[0]["n_cm3"], "all levels below target: lowest level used"
    else:
        k = int(np.where((f[:-1] >= fill_target) & (f[1:] < fill_target))[0][0])
        t = (math.log(f[k]) - math.log(fill_target)) / (math.log(f[k]) - math.log(f[k + 1]))
        n_sel = 10 ** (lg[k] + t * (lg[k + 1] - lg[k]))
        n_sel = float(f"{n_sel:.2g}")
        status = "interpolated"
    out = {"rule": f"symmetric N at which the CAL-5 endpoint fill = {fill_target} (log-linear interpolation)",
           "fill_target": fill_target, "status": status, "selected_n_cm3": n_sel,
           "rows": [{k: _f(v, 5) for k, v in r.items()} for r in rows], "devsim_scale": scale,
           "n_g": _f(n_g, 6)}
    return dataclasses.replace(base, na_cm3=n_sel, nd_cm3=n_sel), out


def artefact_name(xs_name: str, topology: str) -> str:
    return f"junction_{xs_name}" + ("" if topology == DEFAULT_TOPOLOGY else f"_{topology}")
DOPING_RANGES = {
    "na_cm3": (5e17, 5e18), "nd_cm3": (5e17, 5e18), "junction_offset_nm": (-60.0, 60.0), "grading_nm": (0.0, 30.0),
    "plus_gap_nm": (150.0, 500.0), "n_plus_cm3": (3e18, 5e19), "pp_gap_nm": (400.0, 900.0),
    "contact_gap_nm": (700.0, 1500.0), "contact_rho_ohm_cm2": (1e-8, 1e-6),
}
DOPING_PROVENANCE = (
    "assumed. No 45SPCLO or Lightmatter doping is public. Body doping inside the 5e17-3e18 cm^-3 range quoted in "
    "docs/research (05 C7c 5e17; 03 sections 5.1/6.2 1e17-1e18 and 3e18; 06 V15 1-3e18). Lateral: 2e18 (mid-range; "
    "no level up to 5e18 reaches CAL-5 with fill <= 1, see reports.doping_sweep). L-shape: the symmetric level at "
    "which CAL-5 needs fill = 0.85 (doping.selection). Implant offsets and contact levels are generic MRM values "
    "(assumed, swept in the Jacobian).")


# ------------------------------------------------------------------------------------------------ utilities
def _now_utc():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _f(x, nd=6):
    """JSON-friendly float (significant digits)."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return float(f"{x:.{nd}g}")


def _fl(a, nd=7):
    return [_f(v, nd) for v in np.ravel(np.asarray(a, dtype=float))]


def v_grid(quick=False):
    step = V_STEP_QUICK if quick else V_STEP_FULL
    n = int(round((V_MAX - V_MIN) / step)) + 1
    return np.round(np.linspace(V_MIN, V_MAX, n), 6)


@contextlib.contextmanager
def _quiet(enabled=True):
    """Silence DEVSIM's per-iteration stdout (it prints from C++ through Python's sys.stdout)."""
    if not enabled:
        yield
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def richardson(values, hs, order=None):
    """Richardson extrapolation of the last three (h, value) pairs. Returns dict(extrapolated, observed_order,
    error_estimate) where error_estimate = |finest - extrapolated|. ``order`` fixes p if the observed one is not
    usable (oscillatory or non-monotone convergence: then the error estimate is the spread of the last three)."""
    v = np.asarray(values, float)
    h = np.asarray(hs, float)
    out = {"values": _fl(v), "h": _fl(h)}
    if v.size < 3:
        out.update(extrapolated=_f(v[-1]), observed_order=None, error_estimate=_f(abs(v[-1] - v[-2])) if v.size > 1
                   else None)
        return out
    v1, v2, v3 = v[-3:]
    h1, h2, h3 = h[-3:]
    d12, d23 = v1 - v2, v2 - v3
    p = None
    if d12 != 0 and d23 != 0 and np.sign(d12) == np.sign(d23) and abs(d23) < abs(d12):
        r = h1 / h2
        if abs(h2 / h3 - r) < 1e-6 * r:
            p = math.log(abs(d12 / d23)) / math.log(r)
        else:
            from scipy.optimize import brentq

            def g(pp):
                return (h1 ** pp - h2 ** pp) / (h2 ** pp - h3 ** pp) - d12 / d23
            try:
                p = brentq(g, 0.2, 8.0)
            except Exception:
                p = None
    if p is None and order is not None:
        p = order
    if p is None or p <= 0.2:
        spread = float(np.max(v[-3:]) - np.min(v[-3:]))
        out.update(extrapolated=_f(v3), observed_order=None, error_estimate=_f(spread), note="non-monotone")
        return out
    ext = v3 + (v3 - v2) / ((h2 / h3) ** p - 1.0)
    out.update(extrapolated=_f(ext), observed_order=_f(p, 4), error_estimate=_f(abs(v3 - ext)),
               rel_error_estimate=_f(abs(v3 - ext) / max(abs(ext), 1e-300), 4))
    return out


# --------------------------------------------------------------------------------------- doping profiles
def _smooth_step(u_nm, sigma_nm):
    """0 -> 1 as u goes from negative to positive; an error function of width sigma (step if sigma = 0)."""
    from scipy.special import erf
    u = np.asarray(u_nm, float)
    if sigma_nm <= 0:
        return (u > 0).astype(float) + 0.5 * (u == 0)
    return 0.5 * (1.0 + erf(u / (math.sqrt(2.0) * sigma_nm)))


def n_body_indicator(x_nm, y_nm, d: Doping):
    """Smooth 0..1 indicator of the n body (1 = n): lateral, x > x_j; lshape, x > x_j or y > y_j."""
    sx = _smooth_step(np.asarray(x_nm, float) - d.junction_offset_nm, d.grading_nm)
    if d.topology == "lateral":
        return sx
    if d.topology == "lshape":
        if y_nm is None:
            raise ValueError("the lshape profile needs y")
        sy = _smooth_step(np.asarray(y_nm, float) - d.y_junction_nm, d.grading_nm)
        return 1.0 - (1.0 - sx) * (1.0 - sy)
    raise ValueError(f"unknown topology {d.topology!r}")


def doping_profile(x_nm, xs: XSec, d: Doping, y_nm=None):
    """Acceptor and donor densities (cm^-3) at (x, y) in nm (x = 0 at the rib centre, y = 0 at the BOX top)."""
    x = np.asarray(x_nm, float)
    e_p = -(xs.width_nm / 2)          # p-side rib edge
    e_n = +(xs.width_nm / 2)
    nb = n_body_indicator(x, y_nm, d)
    na = (d.na_cm3 * (1.0 - nb)
          + (d.n_plus_cm3 - d.na_cm3) * _smooth_step((e_p - d.plus_gap_nm) - x, d.plus_grading_nm)
          + (d.n_pp_cm3 - d.n_plus_cm3) * _smooth_step((e_p - d.pp_gap_nm) - x, d.plus_grading_nm))
    nd = (d.nd_cm3 * nb
          + (d.n_plus_cm3 - d.nd_cm3) * _smooth_step(x - (e_n + d.plus_gap_nm), d.plus_grading_nm)
          + (d.n_pp_cm3 - d.n_plus_cm3) * _smooth_step(x - (e_n + d.pp_gap_nm), d.plus_grading_nm))
    return np.maximum(na, 0.0), np.maximum(nd, 0.0)


def junction_length_nm(xs: XSec, d: Doping):
    """Length of the metallurgical junction in the cross-section (the depletion-approximation 'height')."""
    if d.topology == "lateral":
        return xs.film_nm
    return (d.junction_offset_nm + xs.width_nm / 2) + d.y_junction_nm


# ------------------------------------------------------------------------------------------ DEVSIM mesh
def _graded_lines(lo, hi, feats, hmax, must):
    """1-D mesh lines on [lo, hi]: local size h(x) = min_f (h_f + g_f max(0, |x - x_f| - a_f)), capped at hmax.
    ``feats`` = [(x_f, h_f, a_f, g_f)]. Every ``must`` point is a line; lines closer than 0.4 h to one are dropped."""
    def h_at(x):
        return min([hmax] + [hf + gf * max(0.0, abs(x - xf) - af) for xf, hf, af, gf in feats])
    pts = [lo]
    x = lo
    while x < hi:
        h1 = h_at(x)
        h2 = h_at(min(x + h1, hi))
        x = x + 0.5 * (h1 + h2)
        pts.append(min(x, hi))
    pts = np.array(pts)
    must = np.unique(np.asarray(must, float))
    keep = [p for p in pts if np.min(np.abs(must - p)) >= 0.4 * h_at(p)]
    return np.unique(np.concatenate([keep, must]))


OX_EXTENT_UM = 0.6


def build_devsim_mesh(xs: XSec, d: Doping, scale=1.0, oxide=True, ox_extent_um=OX_EXTENT_UM):
    """Structured triangular mesh (um) of the rib + slab (region si) and the cladding (region ox).

    x lines: 1 nm x scale within 100 nm of the junction, 3 nm x scale at the rib edges and implant steps, graded to
    20 nm x scale. y lines: 5 nm x scale in Si (L-shape: 1 nm x scale within 60 nm of the horizontal junction); the
    oxide grows by 1.4 per line from 5 nm x scale to 100 nm x scale and extends ``ox_extent_um`` above and below. The oxide stops 0.1 um short of the contacts so that no contact
    node is an interface node. Each rectangle is split into two right triangles (alternating diagonals)."""
    w = xs.width_nm * 1e-3
    film, slab = xs.film_nm * 1e-3, xs.slab_nm * 1e-3
    xc = w / 2 + d.contact_gap_nm * 1e-3
    xj = d.junction_offset_nm * 1e-3
    s = float(scale)
    edges = [-w / 2, w / 2, -(w / 2 + d.plus_gap_nm * 1e-3), w / 2 + d.plus_gap_nm * 1e-3,
             -(w / 2 + d.pp_gap_nm * 1e-3), w / 2 + d.pp_gap_nm * 1e-3]
    feats = [(xj, 0.001 * s, 0.10, 0.3)] + [(e, 0.003 * s, 0.02, 0.3) for e in edges] + \
            [(-xc, 0.005 * s, 0.0, 0.3), (xc, 0.005 * s, 0.0, 0.3)]
    must = [-xc, xc, xj] + edges + ([-(xc - 0.1), xc - 0.1] if oxide else [])
    xs_l = _graded_lines(-xc, xc, feats, 0.02 * s, must)
    hy = 0.005 * s
    if d.topology == "lshape":
        yj = d.y_junction_nm * 1e-3
        ys = list(_graded_lines(0.0, film, [(yj, 0.001 * s, 0.04, 0.4)], hy, [0.0, slab, yj, film]))
    else:
        ny_si = max(4, int(round(film / hy)))
        ys = list(np.linspace(0.0, film, ny_si + 1))
    ys.append(slab)
    if oxide:
        y, h = 0.0, hy
        while y > -ox_extent_um:
            y -= h
            ys.append(max(y, -ox_extent_um))
            h = min(h * 1.4, 0.1 * s)
        y, h = film, hy
        while y < film + ox_extent_um:
            y += h
            ys.append(min(y, film + ox_extent_um))
            h = min(h * 1.4, 0.1 * s)
    ys = np.unique(np.round(np.array(ys), 9))
    xs_l = np.unique(np.round(xs_l, 9))
    nx, ny = xs_l.size, ys.size
    X, Y = np.meshgrid(xs_l, ys, indexing="ij")
    nid = np.arange(nx * ny).reshape(nx, ny)
    xc_c = 0.5 * (xs_l[:-1] + xs_l[1:])[:, None] * np.ones((1, ny - 1))
    yc_c = np.ones((nx - 1, 1)) * 0.5 * (ys[:-1] + ys[1:])[None, :]
    in_si = ((np.abs(xc_c) <= w / 2) & (yc_c >= 0) & (yc_c <= film)) | ((yc_c >= 0) & (yc_c <= slab))
    in_ox = (~in_si) & (np.abs(xc_c) < xc - 0.1) & oxide
    I, J = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    a, b, c, dd = nid[I, J], nid[I + 1, J], nid[I + 1, J + 1], nid[I, J + 1]
    alt = ((I + J) % 2 == 0)
    t1 = np.where(alt[..., None], np.stack([a, b, c], -1), np.stack([a, b, dd], -1))
    t2 = np.where(alt[..., None], np.stack([a, c, dd], -1), np.stack([b, c, dd], -1))
    reg = np.full(in_si.shape, -1)
    reg[in_ox] = 1
    reg[in_si] = 0
    sel = reg >= 0
    tris = np.concatenate([t1[sel], t2[sel]])
    regs = np.concatenate([reg[sel], reg[sel]])
    # boundary edges
    e_all = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    r_all = np.concatenate([regs, regs, regs])
    e_all.sort(axis=1)
    key = e_all[:, 0].astype(np.int64) * (nx * ny) + e_all[:, 1]
    order = np.argsort(key, kind="stable")
    key_s, r_s, e_s = key[order], r_all[order], e_all[order]
    uniq, first, counts = np.unique(key_s, return_index=True, return_counts=True)
    iface, anode, cathode = [], [], []
    xf = X.ravel()
    for k0, n_ in zip(first, counts):
        e = e_s[k0]
        rs = set(r_s[k0:k0 + n_].tolist())
        if rs == {0, 1}:
            iface.append(e)
        elif n_ == 1 and rs == {0}:
            if abs(xf[e[0]] + xc) < 1e-9 and abs(xf[e[1]] + xc) < 1e-9:
                anode.append(e)
            elif abs(xf[e[0]] - xc) < 1e-9 and abs(xf[e[1]] - xc) < 1e-9:
                cathode.append(e)
    names = ["si", "anode", "cathode"] + (["ox", "si_ox"] if oxide and len(iface) else [])
    ix = {n: i for i, n in enumerate(names)}
    el = []
    for t, r in zip(tris, regs):
        el += [2, ix["si" if r == 0 else "ox"], int(t[0]), int(t[1]), int(t[2])]
    for e in iface:
        el += [1, ix["si_ox"], int(e[0]), int(e[1])]
    for e in anode:
        el += [1, ix["anode"], int(e[0]), int(e[1])]
    for e in cathode:
        el += [1, ix["cathode"], int(e[0]), int(e[1])]
    coords = np.stack([X.ravel() * 1e-4, Y.ravel() * 1e-4, np.zeros(nx * ny)], 1).ravel().tolist()
    info = {"nx": int(nx), "ny": int(ny), "triangles": int(len(tris)), "si_triangles": int(np.sum(regs == 0)),
            "h_junction_nm": 1.0 * s, "h_y_si_nm": 5.0 * s, "h_max_nm": 20.0 * s, "oxide": bool(oxide),
            "ox_extent_um": float(ox_extent_um) if oxide else 0.0, "contact_x_um": float(xc), "scale": s,
            "anode_edges": len(anode), "cathode_edges": len(cathode)}
    return coords, names, el, info


# ----------------------------------------------------------------------------------------- DEVSIM solve
def run_devsim(*args, **kw):
    """See :func:`_run_devsim`. DEVSIM's exception class cannot be pickled back from a child process, so errors are
    re-raised as RuntimeError with the stage that failed."""
    try:
        return _run_devsim(*args, **kw)
    except Exception as exc:
        if type(exc).__module__.startswith("devsim"):
            raise RuntimeError(f"DEVSIM: {exc} [{_DEVSIM_STAGE[0]}]") from None
        raise


_DEVSIM_STAGE = ["start"]
DC_REL_TOL = 1e-8     # Newton relative update; 1e-10 hits a round-off floor (~1.1e-10) in deeply depleted nodes
DC_REL_TOL_FALLBACK = 1e-6


def _run_devsim(xs: XSec, d: Doping, v_rev, *, scale=1.0, t_k=T_REF_K, oxide=True, ox_extent_um=OX_EXTENT_UM,
                ac=True, ac_v=None, f_lo_hz=1e6, f_hi_hz=2e10, two_freq=False, quiet=True):
    """2-D drift-diffusion sweep. Returns node coordinates (um), node areas (um^2), doping, and per bias the
    electron/hole densities (cm^-3) and the AC capacitance (F/m) and impedance real part (ohm*m) per unit length.

    AC: one small-signal solve at f_hi_hz per AC bias (all biases, or the subset ``ac_v``) gives the series R (Re Z)
    and C (-1/(w Im Z)); two_freq adds 1 MHz for the frequency check. DEVSIM 2.11 keeps about one complex matrix per
    AC solve allocated (measured), so long sweeps take C(V) from dQ/dV and run AC at a subset in a separate process.

    Bias: anode (p contact) at -V_rev, cathode grounded; both contacts carry a +10 V offset (physically irrelevant)
    so that DEVSIM's relative-error test never meets a node at zero potential. Permittivity 11.7 eps0 (Si) and
    3.9 eps0 (SiO2), n_i(T) from mrm.materials (Misiakos), Arora mobility (doping-dependent, low field),
    SRH tau 1e-5 s (DEVSIM default; irrelevant under reverse bias), full ionization, no band-gap narrowing."""
    import devsim as ds
    from devsim.python_packages import simple_physics as sp
    from devsim.python_packages.model_create import CreateEdgeModel, CreateSolution

    t0 = time.time()
    v_rev = np.asarray(v_rev, float)
    with _quiet(quiet):
        _devsim_reset()
        coords, names, el, minfo = build_devsim_mesh(xs, d, scale, oxide, ox_extent_um)
        has_ox = "ox" in names
        ds.create_gmsh_mesh(mesh="m", coordinates=coords, physical_names=names, elements=el)
        ds.add_gmsh_region(mesh="m", gmsh_name="si", region="si", material="Si")
        if has_ox:
            ds.add_gmsh_region(mesh="m", gmsh_name="ox", region="ox", material="Ox")
            ds.add_gmsh_interface(mesh="m", gmsh_name="si_ox", region0="si", region1="ox", name="si_ox")
        ds.add_gmsh_contact(mesh="m", gmsh_name="anode", region="si", name="anode", material="metal")
        ds.add_gmsh_contact(mesh="m", gmsh_name="cathode", region="si", name="cathode", material="metal")
        ds.finalize_mesh(mesh="m")
        ds.create_device(mesh="m", device="d")
        dev, reg = "d", "si"
        vt = K_B * t_k / Q_E
        ni = float(M.ni_si_per_cm3(t_k)) if t_k <= 340.0 else _ni_extrapolated(t_k)
        sp.SetSiliconParameters(dev, reg, t_k)
        for name, val in (("Permittivity", EPS_SI_R * EPS0 * 1e-2), ("ElectronCharge", Q_E), ("n_i", ni),
                          ("T", t_k), ("kT", K_B * t_k), ("V_t", vt), ("n1", ni), ("p1", ni)):
            ds.set_parameter(device=dev, region=reg, name=name, value=val)
        if has_ox:
            sp.SetOxideParameters(dev, "ox", t_k)
            ds.set_parameter(device=dev, region="ox", name="Permittivity", value=EPS_SIO2_R * EPS0 * 1e-2)
        x_um = np.array(ds.get_node_model_values(device=dev, region=reg, name="x")) * 1e4
        y_um = np.array(ds.get_node_model_values(device=dev, region=reg, name="y")) * 1e4
        na, nd = doping_profile(x_um * 1e3, xs, d, y_um * 1e3)
        for name, val in (("Acceptors", na), ("Donors", nd), ("NetDoping", nd - na)):
            ds.node_solution(device=dev, region=reg, name=name)
            ds.set_node_values(device=dev, region=reg, name=name, values=val.tolist())
        ntot = na + nd
        mun = np.asarray(M.mobility_arora_cm2_per_v_s(ntot, t_k, "n"), float)
        mup = np.asarray(M.mobility_arora_cm2_per_v_s(ntot, t_k, "p"), float)
        for name, val in (("MuN", mun), ("MuP", mup)):
            ds.node_solution(device=dev, region=reg, name=name)
            ds.set_node_values(device=dev, region=reg, name=name, values=val.tolist())
            ds.edge_from_node_model(device=dev, region=reg, node_model=name)
            CreateEdgeModel(dev, reg, "Edge" + name, f"2*{name}@n0*{name}@n1/({name}@n0+{name}@n1)")
        CreateSolution(dev, reg, "Potential")
        sp.CreateSiliconPotentialOnly(dev, reg)
        if has_ox:
            CreateSolution(dev, "ox", "Potential")
            sp.CreateOxidePotentialOnly(dev, "ox", "log_damp")
            sp.CreateSiliconOxideInterface(dev, "si_ox")
        ds.circuit_element(name="V1", n1=sp.GetContactBiasName("anode"), n2=0, value=0.0, acreal=1.0, acimag=0.0)
        sp.CreateSiliconPotentialOnlyContact(dev, reg, "anode", True)
        ds.set_parameter(device=dev, name=sp.GetContactBiasName("cathode"), value=0.0)
        sp.CreateSiliconPotentialOnlyContact(dev, reg, "cathode")
        psi0 = vt * np.arcsinh((nd - na) / (2.0 * ni))
        ds.set_node_values(device=dev, region=reg, name="Potential", values=psi0.tolist())
        if has_ox:
            nox = len(ds.get_node_model_values(device=dev, region="ox", name="x"))
            ds.set_node_values(device=dev, region="ox", name="Potential", values=[0.0] * nox)
        _DEVSIM_STAGE[0] = "poisson 0 V"
        ds.solve(type="dc", absolute_error=1e-9, relative_error=1e-5, maximum_iterations=100)
        CreateSolution(dev, reg, "Electrons")
        CreateSolution(dev, reg, "Holes")
        ds.set_node_values(device=dev, region=reg, name="Electrons", init_from="IntrinsicElectrons")
        ds.set_node_values(device=dev, region=reg, name="Holes", init_from="IntrinsicHoles")
        sp.CreateSiliconDriftDiffusion(dev, reg, mu_n="EdgeMuN", mu_p="EdgeMuP")
        sp.CreateSiliconDriftDiffusionAtContact(dev, reg, "anode", True)
        sp.CreateSiliconDriftDiffusionAtContact(dev, reg, "cathode")
        voff = 10.0
        for rg in ([reg, "ox"] if has_ox else [reg]):
            pv = np.array(ds.get_node_model_values(device=dev, region=rg, name="Potential"))
            ds.set_node_values(device=dev, region=rg, name="Potential", values=(pv + voff).tolist())
        ds.set_parameter(device=dev, name=sp.GetContactBiasName("cathode"), value=voff)
        ds.circuit_alter(name="V1", value=voff)

        def dc(v_anode):
            _DEVSIM_STAGE[0] = f"dc {-v_anode:+.4f} V reverse"
            ds.circuit_alter(name="V1", value=voff + v_anode)
            try:
                ds.solve(type="dc", absolute_error=1e10, relative_error=DC_REL_TOL, maximum_iterations=40)
            except Exception:
                # coarse meshes at high doping stall above 1e-8 (round-off in depleted nodes); accept 1e-6 there
                ds.solve(type="dc", absolute_error=1e10, relative_error=DC_REL_TOL_FALLBACK, maximum_iterations=80)

        dc(0.0)
        vol_um2 = np.array(ds.get_node_model_values(device=dev, region=reg, name="NodeVolume")) * 1e8
        res = {}
        ac_set = None if ac_v is None else {round(float(x), 6) for x in np.atleast_1d(ac_v)}

        def record(vr):
            n = np.array(ds.get_node_model_values(device=dev, region=reg, name="Electrons"))
            p = np.array(ds.get_node_model_values(device=dev, region=reg, name="Holes"))
            row = {"n": n, "p": p}
            if ac and (ac_set is None or round(float(vr), 6) in ac_set):
                _DEVSIM_STAGE[0] = f"ac {vr:+.4f} V"
                ds.solve(type="ac", frequency=f_hi_hz)
                i_hi = complex(ds.get_circuit_node_value(node="V1.I", solution="ssac_real"),
                               ds.get_circuit_node_value(node="V1.I", solution="ssac_imag"))
                # device admittance per cm of length = -I (V1.I flows into the source); a unit AC voltage
                z_hi = 1.0 / (-i_hi)
                row["r_ac_ohm_m"] = z_hi.real * 1e-2
                row["c_ac_f_per_m"] = (-1.0 / (2 * math.pi * f_hi_hz * z_hi.imag)) * 1e2
                if two_freq:
                    ds.solve(type="ac", frequency=f_lo_hz)
                    i_lo = complex(ds.get_circuit_node_value(node="V1.I", solution="ssac_real"),
                                   ds.get_circuit_node_value(node="V1.I", solution="ssac_imag"))
                    row["c_lo_f_per_m"] = ((-i_lo).imag / (2 * math.pi * f_lo_hz)) * 1e2
            res[round(float(vr), 6)] = row

        def go(cur, target, hmax=0.25):
            """Adaptive continuation from cur to target (reverse volts): the step grows by 1.5 after a success and
            halves after a failure (DEVSIM restores the last converged solution when a solve fails)."""
            h = min(hmax, abs(target - cur)) or hmax
            while abs(target - cur) > 1e-12:
                nxt = cur + math.copysign(min(h, abs(target - cur)), target - cur)
                try:
                    dc(-nxt)
                    cur = nxt
                    h = min(hmax, 1.5 * h)
                except Exception:
                    h *= 0.5
                    if h < 2e-4:
                        raise
            return cur

        def walk(targets):
            cur = 0.0
            for vt_ in targets:
                cur = go(cur, vt_)
                record(vt_)

        fwd = sorted([v for v in v_rev if v < 0], reverse=True)
        rev = sorted([v for v in v_rev if v >= 0])
        if fwd:
            walk(fwd)
            go(fwd[-1], 0.0)
        walk(rev)
    keys = sorted(res)
    out = {"x_um": x_um, "y_um": y_um, "vol_um2": vol_um2, "na": na, "nd": nd, "v": np.array(keys),
           "n": np.array([res[k]["n"] for k in keys]), "p": np.array([res[k]["p"] for k in keys]),
           "mesh": minfo, "nodes_si": int(x_um.size), "t_k": t_k, "seconds": round(time.time() - t0, 2),
           "devsim_version": getattr(ds, "__version__", "?")}
    if ac:
        ka = [k for k in keys if "r_ac_ohm_m" in res[k]]
        out["ac_v"] = np.array(ka)
        for kk in ("c_ac_f_per_m", "r_ac_ohm_m") + (("c_lo_f_per_m",) if two_freq else ()):
            out[kk] = np.array([res[k][kk] for k in ka])
    with _quiet(quiet):
        _devsim_reset()
    return out


class _SpluObject:
    def __init__(self, n, transpose):
        self.n, self.transpose, self.lu, self.is_complex = n, bool(transpose), None, False


def scipy_solver_callback(**kw):
    """DEVSIM custom direct solver backed by scipy SuperLU (MMD on A^T + A). The bundled UMFPACK 5.1 shim needs about
    55 kB per mesh node on these 2-D meshes (1.5 GB at 27k nodes); SuperLU needs a fraction of that. Same protocol
    as devsim/umfpack/umfshim.py: CSC arrays (array.array), complex values interleaved, transpose = A.' (no
    conjugate)."""
    import array

    import scipy.sparse as sps
    import scipy.sparse.linalg as spl
    act = kw["action"]
    if act == "init":
        so = _SpluObject(kw["n"], kw["transpose"])
        return {"solver_object": so, "matrix_format": "csc", "status": True, "message": ""}
    so = kw["solver_object"]
    if act == "factor":
        ap = np.frombuffer(kw["Ap"], dtype=np.dtype(kw["Ap"].typecode))
        ai = np.frombuffer(kw["Ai"], dtype=np.dtype(kw["Ai"].typecode))
        ax = np.frombuffer(kw["Ax"], dtype=float)
        so.is_complex = bool(kw["is_complex"])
        if so.is_complex:
            ax = ax[0::2] + 1j * ax[1::2]
        n = ap.size - 1
        A = sps.csc_matrix((ax.copy(), ai.astype(np.int64), ap.astype(np.int64)), shape=(n, n))
        so.lu = None
        try:
            so.lu = spl.splu(A, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": False})
        except RuntimeError as exc:
            return {"status": False, "message": f"splu: {exc}"}
        return {"status": True, "message": ""}
    if act == "solve":
        b = np.frombuffer(kw["b"], dtype=float)
        if so.is_complex:
            b = b[0::2] + 1j * b[1::2]
        x = so.lu.solve(b, trans="T" if so.transpose else "N")
        if so.is_complex:
            xi = np.empty(2 * x.size)
            xi[0::2], xi[1::2] = x.real, x.imag
            x = xi
        return {"status": True, "message": "", "x": array.array("d", np.asarray(x, float).tobytes())}
    raise RuntimeError("unsupported action " + str(act))


DEVSIM_SOLVER = "umfpack"     # or "superlu_scipy"
UMFPACK_STRATEGY = 3          # UMFPACK 5.1 Control[5]: 0 auto, 1 unsymmetric, 3 symmetric (AMD on A + A^T)
_LIVE_SOLVERS = []


def umfpack_leakfree_callback(**kw):
    """DEVSIM's bundled UMFPACK shim, with one fix: DEVSIM creates a new solver object at every solve() call and
    keeps a reference to each one forever, so every last numeric factorization stays allocated (measured: about
    50 MB per solve() on a 15k-node mesh; a 91-bias sweep with AC would need many GB). A new 'init' starts a new
    solve() call, after which the older objects are never used again, so their factorizations are released here."""
    from devsim.umfpack import umfshim
    if kw["action"] == "init":
        for o in _LIVE_SOLVERS:
            for attr in ("numeric", "symbolic", "matrix", "lu", "x"):
                if hasattr(o, attr):
                    setattr(o, attr, None)
        _LIVE_SOLVERS.clear()
        out = umfshim.local_solver_callback(**kw) if DEVSIM_SOLVER == "umfpack" else scipy_solver_callback(**kw)
        _LIVE_SOLVERS.append(out["solver_object"])
        return out
    if DEVSIM_SOLVER == "umfpack" and kw["action"] == "factor":
        so = kw["solver_object"]
        if so.umf_control is None and UMFPACK_STRATEGY is not None:
            from devsim.umfpack import umfpack_loader as umf
            so.umf_control = umf.umf_control(so.gdata, "complex" if kw["is_complex"] else "real")
            so.umf_control.Control[5] = float(UMFPACK_STRATEGY)
    return umfshim.local_solver_callback(**kw) if DEVSIM_SOLVER == "umfpack" else scipy_solver_callback(**kw)


def _devsim_reset():
    """reset_devsim() also forgets the direct solver registered at import; install the chosen one again."""
    import devsim as ds
    ds.reset_devsim()
    for o in _LIVE_SOLVERS:
        for attr in ("numeric", "symbolic", "matrix", "lu", "x"):
            if hasattr(o, attr):
                setattr(o, attr, None)
    _LIVE_SOLVERS.clear()
    ds.set_parameter(name="direct_solver", value="custom")
    ds.set_parameter(name="solver_callback", value=umfpack_leakfree_callback)


def _ni_extrapolated(t_k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", M.MaterialRangeWarning)
        return float(M.ni_si_per_cm3(t_k))


def carriers_per_m(dv):
    """Total electrons and holes per metre of junction (vol in um^2, densities in cm^-3)."""
    w = dv["vol_um2"] * 1e-12 * 1e6
    return dv["n"] @ w, dv["p"] @ w


def depletion_width_nm(dv, xs: XSec, d: Doping, y_frac=0.5):
    """Width of the depleted zone on a cut normal to the junction: for a lateral junction a horizontal cut at
    y = y_frac*film; for the L-shape a vertical cut through the middle of the horizontal leg. The edges are the points
    where p falls to N_A/2 and n to N_D/2 (linear interpolation along the cut)."""
    if d.topology == "lateral":
        along, across = dv["x_um"], dv["y_um"]
        target = y_frac * xs.film_nm * 1e-3
        j = d.junction_offset_nm
    else:
        along, across = dv["y_um"], dv["x_um"]
        target = 0.5 * (-xs.width_nm / 2 + d.junction_offset_nm) * 1e-3
        j = d.y_junction_nm
    lines = np.unique(np.round(across, 9))
    cl = lines[np.argmin(np.abs(lines - target))]
    sel = np.abs(across - cl) < 1e-9
    o = np.argsort(along[sel])
    x = along[sel][o] * 1e3
    na, nd = dv["na"][sel][o], dv["nd"][sel][o]
    out = []
    for n, p in zip(dv["n"][:, sel][:, o], dv["p"][:, sel][:, o]):
        fp = p - 0.5 * np.maximum(na, 1.0)
        fn = n - 0.5 * np.maximum(nd, 1.0)
        mp = (x < j) & (x > j - 400)
        mn = (x > j) & (x < j + 400)
        xp = _crossing(x[mp], fp[mp], from_right=True)
        xn = _crossing(x[mn], fn[mn], from_right=False)
        out.append((xp, xn))
    a = np.array(out)
    return {"cut_nm": float(cl * 1e3), "cut": "y" if d.topology == "lateral" else "x",
            "x_p_nm": a[:, 0], "x_n_nm": a[:, 1], "w_nm": a[:, 1] - a[:, 0]}


def _crossing(x, f, from_right):
    """Position where f changes sign (closest to the junction): the junction is at the right end of the p-side cut
    and at the left end of the n-side cut."""
    idx = range(len(x) - 1, 0, -1) if from_right else range(0, len(x) - 1)
    for i in idx:
        j = i - 1 if from_right else i + 1
        if np.sign(f[i]) != np.sign(f[j]):
            return x[i] + (x[j] - x[i]) * (0 - f[i]) / (f[j] - f[i])
    return np.nan


# -------------------------------------------------------------------------------------------- femwell
SI_REGIONS = ("core", "slab")
NEAR_SLAB_UM = 0.30          # the fine-mesh Si subdomain reaches this far beyond the rib edge


def _mode_mesh(xs: XSec, d: Doping, res_um, dom_um=(2.5, 1.0, 1.2)):
    """femwell mesh (um): Si in two subdomains, ``core`` (rib + slab within 0.3 um of the rib edge, resolution
    res_um) and ``slab`` (the rest of the slab to the contacts, 2.5 res_um); BOX below, cladding above."""
    from collections import OrderedDict

    import shapely
    from femwell.mesh import mesh_from_OrderedDict
    from skfem.io.meshio import from_meshio
    w, film, slab = xs.width_nm * 1e-3, xs.film_nm * 1e-3, xs.slab_nm * 1e-3
    xc = w / 2 + d.contact_gap_nm * 1e-3
    xn = min(w / 2 + NEAR_SLAB_UM, xc - 0.05)
    xd, yb, yt = dom_um
    core = shapely.union(shapely.box(-w / 2, 0, w / 2, film), shapely.box(-xn, 0, xn, slab))
    slab_far = shapely.union(shapely.box(-xc, 0, -xn, slab), shapely.box(xn, 0, xc, slab))
    polys = OrderedDict(core=core, slab=slab_far, box=shapely.box(-xd, -yb, xd, 0.0),
                        clad=shapely.box(-xd, 0.0, xd, yt))
    resd = {"core": {"resolution": res_um, "distance": 0.4},
            "slab": {"resolution": max(2.5 * res_um, 0.01), "distance": 0.4}}
    with _quiet(True), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = from_meshio(mesh_from_OrderedDict(polys, resd, default_resolution_max=max(0.1, 6 * res_um)))
    return m


class ModeResult:
    """TE0 mode on a femwell mesh, with the index-weight density on a DG-P1 basis (1/um^2)."""

    def __init__(self, mode, mesh, xs, d, radius_um, lam_um, n_reg, res_um):
        self.mode, self.mesh, self.xs, self.d = mode, mesh, xs, d
        self.radius_um, self.lam_um = radius_um, lam_um
        self.n_reg, self.res_um = n_reg, res_um
        self.n_eff = complex(mode.n_eff)
        self._wdg = None
        self._bdg = None

    def _bend(self, xq):
        return (1.0 + xq / self.radius_um) ** 2 if math.isfinite(self.radius_um) else 1.0

    def weight_dg(self):
        from scipy.constants import epsilon_0, speed_of_light
        from skfem import ElementDG, ElementTriP1
        if self._wdg is None:
            mo = self.mode
            (et, ez) = mo.basis.interpolate(mo.E)
            epsq = mo.basis_epsilon_r.interpolate(mo.epsilon_r)
            xq = mo.basis.global_coordinates().value[0]
            e2 = np.real(np.conj(et[0]) * et[0] + np.conj(et[1]) * et[1] + np.conj(ez) * ez)
            wq = speed_of_light * epsilon_0 * np.sqrt(np.real(epsq)) * self._bend(xq) * e2
            self._bdg = mo.basis.with_element(ElementDG(ElementTriP1()))
            self._wdg = self._bdg.project(wq)
        return self._bdg, self._wdg

    def region_weight(self, regions):
        """dn_eff/dn_region: the weight integrated over the elements of the region(s), at the mode's quadrature."""
        from scipy.constants import epsilon_0, speed_of_light
        from skfem import Functional
        regions = (regions,) if isinstance(regions, str) else tuple(regions)
        mo = self.mode

        @Functional
        def f(w):
            return w["g"]
        tot = 0.0
        for region in regions:
            b = mo.basis.with_elements(region)
            (et, ez) = b.interpolate(mo.E)
            epsq = mo.basis_epsilon_r.with_elements(region).interpolate(mo.epsilon_r)
            xq = b.global_coordinates().value[0]
            e2 = np.real(np.conj(et[0]) * et[0] + np.conj(et[1]) * et[1] + np.conj(ez) * ez)
            tot += float(speed_of_light * epsilon_0 * f.assemble(b, g=np.sqrt(np.real(epsq)) * self._bend(xq) * e2))
        return tot

    def power_fraction(self, regions):
        regions = (regions,) if isinstance(regions, str) else tuple(regions)
        tot = sum(float(np.real(self.mode.calculate_power(elements=r))) for r in regions)
        return tot / float(np.real(self.mode.calculate_power()))

    def weight_at(self, x_um, y_um, chunk=4000):
        """Weight density at points (Si side), evaluated in chunks (skfem's point search is memory hungry)."""
        bdg, wdg = self.weight_dg()
        x, y = _nudge_into_si(x_um, y_um, self.xs, self.d)
        out = np.empty(x.size)
        for i in range(0, x.size, chunk):
            P = bdg.probes(np.vstack([x[i:i + chunk], y[i:i + chunk]]))
            out[i:i + chunk] = np.asarray(P @ wdg).ravel()
        return out

    def weight_grid(self, step_um=0.002):
        """Sample the weight density on a regular grid covering the Si (points outside Si take the value at the
        nearest Si point), for a compact, process-independent WeightField."""
        xs, d = self.xs, self.d
        xc = xs.width_nm * 5e-4 + d.contact_gap_nm * 1e-3
        nxg = int(round(2 * xc / step_um)) + 1
        nyg = int(round(xs.film_nm * 1e-3 / step_um)) + 1
        xg = np.linspace(-xc, xc, nxg)
        yg = np.linspace(0.0, xs.film_nm * 1e-3, nyg)
        X, Y = np.meshgrid(xg, yg, indexing="ij")
        W = self.weight_at(X.ravel(), Y.ravel()).reshape(X.shape)
        return xg, yg, W


def _nudge_into_si(x_um, y_um, xs: XSec, d: Doping, dlt=5e-5):
    """Move points onto the Si side of any Si/oxide boundary (0.05 nm inside): the rib for y above the slab top
    within the rib width, the slab otherwise; points outside the Si go to the nearest Si point."""
    x = np.array(x_um, float, copy=True)
    y = np.array(y_um, float, copy=True)
    w2, film, slab = xs.width_nm * 5e-4, xs.film_nm * 1e-3, xs.slab_nm * 1e-3
    xc = w2 + d.contact_gap_nm * 1e-3
    x = np.clip(x, -(xc - dlt), xc - dlt)
    y = np.clip(y, dlt, film - dlt)
    above = y > slab - dlt
    outside_rib = np.abs(x) > w2 - dlt
    # above the slab and outside the rib: choose the nearer of the rib sidewall and the slab top
    m = above & outside_rib
    d_side = np.abs(x) - w2
    d_top = y - slab
    to_side = m & (d_side <= d_top)
    to_top = m & ~to_side
    x = np.where(to_side, np.sign(x) * (w2 - dlt), x)
    y = np.where(to_top, slab - dlt, y)
    return x, y


class WeightField:
    """Index-weight density (1/um^2) from a regular grid, bilinear (RegularGridInterpolator), Si side only."""

    def __init__(self, xg, yg, W, xs: XSec, d: Doping, meta=None):
        from scipy.interpolate import RegularGridInterpolator
        self.xg, self.yg, self.W = np.asarray(xg), np.asarray(yg), np.asarray(W)
        self.xs, self.d = xs, d
        self.meta = meta or {}
        self._f = RegularGridInterpolator((self.xg, self.yg), self.W, bounds_error=False, fill_value=None)

    def weight_at(self, x_um, y_um):
        x, y = _nudge_into_si(x_um, y_um, self.xs, self.d)
        return self._f(np.stack([x, y], -1))


def solve_te0(xs: XSec, d: Doping, *, res_um=0.02, radius_um=P_RING_R_UM, lam_m=LAMBDA0_M, t_k=T_REF_K,
              dn_si=0.0, eps_si_override=None, mesh=None, num_modes=2):
    """Solve the TE0 mode (femwell, order 2). radius_um = inf for the straight guide. eps_si_override: complex
    P0 permittivity for the Si elements in the order of ``si_dofs(basis)`` (direct re-solve with carriers)."""
    from femwell.maxwell.waveguide import compute_modes
    from skfem import Basis, ElementTriP0
    lam_um = lam_m * 1e6
    m = mesh if mesh is not None else _mode_mesh(xs, d, res_um)
    nsi = float(M.n_si(lam_m, t_k)) + dn_si
    nbox = float(M.n_sio2(lam_m, t_k, kind=xs.box))
    ncl = float(M.n_sio2(lam_m, t_k, kind=xs.clad))
    b0 = Basis(m, ElementTriP0())
    dtype = complex if eps_si_override is not None else float
    eps = b0.zeros(dtype=dtype)
    sid = si_dofs(b0)
    eps[sid] = nsi ** 2
    if eps_si_override is not None:
        eps[sid] = eps_si_override
    eps[b0.get_dofs(elements="box").flatten()] = nbox ** 2
    eps[b0.get_dofs(elements="clad").flatten()] = ncl ** 2
    with _quiet(True), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modes = compute_modes(b0, eps, wavelength=lam_um, num_modes=num_modes, order=2,
                              radius=radius_um if radius_um else np.inf, n_guess=nsi * 0.75)
    te = [mo for mo in modes if mo.te_fraction > 0.5]
    mo = max(te, key=lambda q: q.n_eff.real)
    return ModeResult(mo, m, xs, d, radius_um if radius_um else np.inf, lam_um,
                      {"si": nsi, "box": nbox, "clad": ncl}, res_um)


def si_dofs(b0):
    return np.concatenate([b0.get_dofs(elements=r).flatten() for r in SI_REGIONS])


def mode_properties(xs: XSec, d: Doping, *, res_um=0.02, radius_um=P_RING_R_UM, t_k=T_REF_K, dlam_m=5e-9,
                    with_dn_check=True, with_ng=True, grid_step_um=0.002, keep_mode=False):
    """n_eff, n_g (dispersive materials, central difference), index weights per region, the identity
    sum_i n_g,i dn_eff/dn_i = n_g, power fraction in Si, dn_eff/dT (materials dn/dT and index weights), the
    finite-difference dn_eff/dn_Si check of the weight integral, and the weight density on a grid. Same mesh for
    every solve. Returns plain arrays (no solver objects unless keep_mode) so it can run in a child process."""
    t0 = time.time()
    m = _mode_mesh(xs, d, res_um)
    mr = solve_te0(xs, d, res_um=res_um, radius_um=radius_um, t_k=t_k, mesh=m)
    lam = LAMBDA0_M
    n0 = mr.n_eff.real
    n_g = None
    if with_ng:
        n_p = solve_te0(xs, d, radius_um=radius_um, lam_m=lam + dlam_m, t_k=t_k, mesh=m).n_eff.real
        n_m = solve_te0(xs, d, radius_um=radius_um, lam_m=lam - dlam_m, t_k=t_k, mesh=m).n_eff.real
        n_g = n0 - lam * (n_p - n_m) / (2 * dlam_m)
    gw = {"si": mr.region_weight(SI_REGIONS), "box": mr.region_weight("box"), "clad": mr.region_weight("clad")}
    ng_i = {"si": float(M.n_g_si(lam, t_k)), "box": float(M.n_g_sio2(lam, t_k, kind=xs.box)),
            "clad": float(M.n_g_sio2(lam, t_k, kind=xs.clad))}
    ident = sum(ng_i[r] * gw[r] for r in gw)
    dndt = {"si": float(M.dn_dt_si_per_k(lam, t_k)), "box": float(M.dn_dt_sio2_per_k(t_k, kind=xs.box)),
            "clad": float(M.dn_dt_sio2_per_k(t_k, kind=xs.clad))}
    dneff_dt = sum(gw[r] * dndt[r] for r in gw)
    out = {"n_eff": n0, "n_g": n_g, "index_weight": gw, "n_g_material": ng_i,
           "identity_sum_ng_i_w_i": ident, "identity_rel_residual": (ident - n_g) / n_g if n_g else None,
           "power_fraction_si": mr.power_fraction(SI_REGIONS), "dneff_dt_per_k": dneff_dt,
           "elements": int(m.nelements), "res_um": res_um, "radius_um": radius_um}
    if with_dn_check:
        dn = 1e-4
        n_dn = solve_te0(xs, d, radius_um=radius_um, t_k=t_k, dn_si=dn, mesh=m).n_eff.real
        out["dneff_dnsi_fd"] = (n_dn - n0) / dn
    xg, yg, W = mr.weight_grid(grid_step_um)
    out["weight_grid"] = {"xg": xg, "yg": yg, "W": W, "step_um": grid_step_um}
    out["seconds"] = round(time.time() - t0, 1)
    if keep_mode:
        out["mode"] = mr
    return out


def weight_field(mp, xs, d):
    g = mp["weight_grid"]
    return WeightField(g["xg"], g["yg"], g["W"], xs, d, meta={"res_um": mp["res_um"], "radius_um": mp["radius_um"]})


# ------------------------------------------------------------------------------------------ process isolation
ISOLATE = True


def _isolated(fn, *args, **kw):
    """Run a solver job in a fresh spawned process so that its memory is returned to the OS (16 GB machine shared
    with other agents; femwell and DEVSIM keep their peak RSS otherwise)."""
    if not ISOLATE:
        return fn(*args, **kw)
    import importlib
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    fn = getattr(importlib.import_module("mrm.extraction.junction"), fn.__name__)
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(fn, *args, **kw).result()


# ------------------------------------------------------------------------------------------ optics overlap
def plasma_fields(dv, model="nsm2011", t_k=T_REF_K):
    """Material index change (relative to the 0 V state) and absolute absorption (1/m) at every node and bias."""
    i0 = int(np.argmin(np.abs(dv["v"])))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", M.MaterialRangeWarning)
        dn_abs, a_abs = M.plasma_dispersion(dv["n"], dv["p"], model=model, warn=False)
    dn_abs, a_abs = np.asarray(dn_abs), np.asarray(a_abs)
    # per-carrier temperature factors (assumed, 02 section 7) relative to the LUT reference temperature
    fdn = float(M.plasma_temperature_factor(t_k, "dn")) / float(M.plasma_temperature_factor(T_REF_K, "dn"))
    fda = float(M.plasma_temperature_factor(t_k, "dalpha")) / float(M.plasma_temperature_factor(T_REF_K, "dalpha"))
    return (dn_abs - dn_abs[i0]) * fdn, a_abs * fda, i0


def overlap(dv, w_nodes, model="nsm2011"):
    """dn_eff(V) relative to 0 V and absolute alpha_FCA(V) (1/m) for a full-circumference junction."""
    ddn, alpha, _ = plasma_fields(dv, model, dv.get("t_k", T_REF_K))
    wa = w_nodes * dv["vol_um2"]
    return ddn @ wa, alpha @ wa


def direct_resolve_dneff(xs, d, dv, v_list, mode_res_um, radius_um, model="nsm2011"):
    """Direct femwell re-solve with the carrier-perturbed complex permittivity of the Si elements (element averages
    of the DEVSIM fields from a 7x7 sub-sampling of each triangle). Returns dn_eff(V) - dn_eff(0) and alpha (1/m)."""
    import gc

    import matplotlib.tri as mtri
    mode_res_um = max(mode_res_um, 0.0141)    # memory: the 10 nm mode mesh plus the carrier fields exceed 2.7 GB
    m = _mode_mesh(xs, d, mode_res_um)
    from skfem import Basis, ElementTriP0
    b0 = Basis(m, ElementTriP0())
    core = si_dofs(b0)
    # element sub-sampling points (barycentric grid)
    k = 7
    bary = np.array([(i, j, k - 1 - i - j) for i in range(k) for j in range(k - i)], float) + 1.0 / 3
    bary /= bary.sum(1, keepdims=True)
    tri_nodes = m.t[:, core]
    P = m.p
    xs_pts = (bary @ P[0][tri_nodes])   # (nb, ncore)
    ys_pts = (bary @ P[1][tri_nodes])
    tri = mtri.Triangulation(dv["x_um"], dv["y_um"])
    ddn, alpha, _ = plasma_fields(dv, model)
    # restrict the DEVSIM triangulation to Si (Delaunay of Si nodes can bridge the notch above the slab)
    w2, slab = xs.width_nm * 5e-4, xs.slab_nm * 1e-3
    xm = dv["x_um"][tri.triangles].mean(1)
    ym = dv["y_um"][tri.triangles].mean(1)
    tri.set_mask((np.abs(xm) > w2 + 1e-9) & (ym > slab + 1e-9))
    nsi = float(M.n_si(LAMBDA0_M, T_REF_K))
    k0 = 2 * np.pi / LAMBDA0_M
    base = solve_te0(xs, d, radius_um=radius_um, mesh=m).n_eff
    out_dn, out_a = [], []
    for v in v_list:
        iv = int(np.argmin(np.abs(dv["v"] - v)))
        f_dn = mtri.LinearTriInterpolator(tri, ddn[iv])
        f_a = mtri.LinearTriInterpolator(tri, alpha[iv])
        xx = np.clip(xs_pts, dv["x_um"].min() + 1e-7, dv["x_um"].max() - 1e-7)
        e_dn = np.ma.filled(f_dn(xx, ys_pts), np.nan)
        e_a = np.ma.filled(f_a(xx, ys_pts), np.nan)
        e_dn = np.nanmean(e_dn, 0)
        e_a = np.nanmean(e_a, 0)
        kappa = e_a / (2 * k0)          # imaginary index from intensity absorption alpha = 2 k0 kappa
        eps_c = (nsi + e_dn - 1j * kappa) ** 2
        mr = solve_te0(xs, d, radius_um=radius_um, mesh=m, eps_si_override=eps_c)
        out_dn.append(mr.n_eff.real - base.real)
        out_a.append(2 * k0 * abs(mr.n_eff.imag))
        del mr
        gc.collect()
    return np.array(out_dn), np.array(out_a), int(m.nelements), float(base.real)


def tidy3d_check(xs: XSec, d: Doping, dls_nm=(20.0, 10.0, 20.0 / 3.0)):
    """Independent straight-guide n_eff and dn_eff/dn_Si with the tidy3d LOCAL mode solver (uniform grid aligned to
    the interfaces, no subpixel averaging; slab to the contacts, oxide beyond)."""
    t0 = time.time()
    with _quiet(True), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import tidy3d as td
        from tidy3d.plugins.mode import ModeSolver
        td.config.logging.level = "ERROR"
    lam_um = LAMBDA0_M * 1e6
    w, film, slab = xs.width_nm * 1e-3, xs.film_nm * 1e-3, xs.slab_nm * 1e-3
    xc = w / 2 + d.contact_gap_nm * 1e-3
    nsi = float(M.n_si(LAMBDA0_M, T_REF_K))
    nbox = float(M.n_sio2(LAMBDA0_M, T_REF_K, kind=xs.box))
    ncl = float(M.n_sio2(LAMBDA0_M, T_REF_K, kind=xs.clad))
    rows = []
    for dl_nm in dls_nm:
        dl = dl_nm * 1e-3
        res = []
        for dn in (0.0, 1e-3):
            # propagation along x; cross-section in (y, z): y lateral, z vertical
            structs = [
                td.Structure(geometry=td.Box(center=(0, 0, -0.6), size=(td.inf, td.inf, 1.2)),
                             medium=td.Medium(permittivity=nbox ** 2)),
                td.Structure(geometry=td.Box(center=(0, 0, 0.5 + film / 2), size=(td.inf, td.inf, 1.0 + film)),
                             medium=td.Medium(permittivity=ncl ** 2)),
                td.Structure(geometry=td.Box(center=(0, 0, slab / 2), size=(td.inf, 2 * xc, slab)),
                             medium=td.Medium(permittivity=(nsi + dn) ** 2)),
                td.Structure(geometry=td.Box(center=(0, 0, film / 2), size=(td.inf, w, film)),
                             medium=td.Medium(permittivity=(nsi + dn) ** 2)),
            ]
            py, pz = 2.0, 1.6          # interfaces fall on the grid for dl = 20, 10, 6.667 nm
            nzb = int(round(0.6 / dl))
            zc = -nzb * dl + pz / 2 - 1e-12
            sim = td.Simulation(size=(1.0, py, pz), center=(0, 0, zc), grid_spec=td.GridSpec.uniform(dl=dl),
                                structures=structs, medium=td.Medium(permittivity=ncl ** 2), run_time=1e-12,
                                boundary_spec=td.BoundarySpec.all_sides(boundary=td.Periodic()))
            ms = ModeSolver(simulation=sim, plane=td.Box(center=(0, 0, zc), size=(0, py, pz)),
                            mode_spec=td.ModeSpec(num_modes=1, target_neff=nsi * 0.75),
                            freqs=[C0 / LAMBDA0_M])
            with _quiet(True), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = ms.solve()
            res.append(float(data.n_eff.values[0, 0].real))
        rows.append({"dl_nm": dl_nm, "n_eff": res[0], "dneff_dnsi": (res[1] - res[0]) / 1e-3})
    return {"rows": rows, "seconds": round(time.time() - t0, 1), "tidy3d_version": td.__version__,
            "n_eff_conv": richardson([r["n_eff"] for r in rows], [r["dl_nm"] for r in rows], order=1),
            "dneff_dnsi_conv": richardson([r["dneff_dnsi"] for r in rows], [r["dl_nm"] for r in rows], order=1),
            "note": "uniform grid without subpixel averaging: staircase error, first order in dl"}


# ---------------------------------------------------------------------------------------- analytic model
def analytic_depletion(xs: XSec, d: Doping, v_rev, t_k=T_REF_K, debye=False):
    """Abrupt-junction depletion approximation (Sze & Ng; 03 section 5.1): x_n, x_p, W (nm) and C' = eps L_j/W (F/m)
    with L_j the junction length in the cross-section (the film for a lateral junction; both legs for the L-shape).
    debye=True uses V_bi - 2kT/q. No fringe field."""
    v = np.asarray(v_rev, float)
    na, nd = d.na_cm3, d.nd_cm3
    vbi = float(M.vbi_v(na, nd, t_k))
    vt = K_B * t_k / Q_E
    veff = vbi + v - (2 * vt if debye else 0.0)
    eps = EPS_SI_R * EPS0
    w_m = np.sqrt(np.maximum(2 * eps * veff / Q_E * (na + nd) / (na * nd) * 1e-6, 0.0))
    xn = w_m * na / (na + nd)
    xp = w_m * nd / (na + nd)
    c = eps * junction_length_nm(xs, d) * 1e-9 / np.where(w_m > 0, w_m, np.nan)
    return {"v_bi_v": vbi, "w_nm": w_m * 1e9, "x_n_nm": xn * 1e9, "x_p_nm": xp * 1e9, "c_f_per_m": c,
            "junction_length_nm": junction_length_nm(xs, d)}


def _distance_to_junction(X, Y, d: Doping):
    """(n-region mask, distance in nm of every point to the metallurgical junction) for the sharp topology."""
    xt, yj = d.junction_offset_nm, d.y_junction_nm
    if d.topology == "lateral":
        n_reg = X > xt
        return n_reg, np.abs(X - xt)
    n_reg = (X > xt) | (Y > yj)
    dist = np.where(~n_reg, np.minimum(xt - X, yj - Y),
                    np.where((X >= xt) & (Y >= yj), np.hypot(X - xt, Y - yj),
                             np.where(X >= xt, X - xt, Y - yj)))
    return n_reg, dist


def analytic_dneff(xs: XSec, d: Doping, wf, v_rev, model="nsm2011", h_nm=2.0):
    """Abrupt box-depletion model through the 2-D mode weight: every Si point closer to the metallurgical junction
    than the abrupt depletion depth of its side (x_n or x_p, 1-D formula) is empty, everything else is neutral at its
    doping. Cells straddling a depletion edge are weighted by the covered fraction (linear in the distance), so the
    result is continuous in V. Gives dn_eff(V) relative to 0 V and the absolute alpha_FCA(V)."""
    an = analytic_depletion(xs, d, v_rev)
    w2 = xs.width_nm / 2
    xc = w2 + d.contact_gap_nm
    film, slab = xs.film_nm, xs.slab_nm
    xg = np.arange(-xc + h_nm / 2, xc, h_nm)
    yg = np.arange(h_nm / 2, film, h_nm)
    X, Y = np.meshgrid(xg, yg, indexing="ij")
    in_si = (np.abs(X) <= w2) | (Y <= slab)
    W = np.where(in_si, wf.weight_at(X * 1e-3, Y * 1e-3), 0.0)
    dA = h_nm * h_nm * 1e-6
    d_sharp = dataclasses.replace(d, grading_nm=0.0)
    na, nd = doping_profile(X, xs, d_sharp, Y)
    n_reg, dist = _distance_to_junction(X, Y, d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", M.MaterialRangeWarning)
        dn_neu, a_neu = M.plasma_dispersion(np.where(n_reg, nd, 0.0), np.where(n_reg, 0.0, na), model=model,
                                            warn=False)
    dn_neu, a_neu = np.asarray(dn_neu), np.asarray(a_neu)
    vv = np.atleast_1d(np.asarray(v_rev, float))
    dn_out, a_out = np.empty(vv.size), np.empty(vv.size)
    for i in range(vv.size):
        edge = np.where(n_reg, an["x_n_nm"][i], an["x_p_nm"][i])
        frac = np.clip((edge - dist) / h_nm + 0.5, 0.0, 1.0)
        dn_out[i] = np.sum(W * (1.0 - frac) * dn_neu) * dA
        a_out[i] = np.sum(W * (1.0 - frac) * a_neu) * dA
    i0 = int(np.argmin(np.abs(vv)))
    return {"dneff": dn_out - dn_out[i0], "alpha_fca_per_m": a_out, "v_bi_v": an["v_bi_v"],
            "c_f_per_m": an["c_f_per_m"], "w_nm": an["w_nm"]}


# ---------------------------------------------------------------------------------------- series resistance
def series_resistance_analytic(xs: XSec, d: Doping, v_rev, t_k=T_REF_K):
    """r_s(V) per unit length (ohm*m) from the sheet resistances in series on each side: undepleted rib (film),
    slab body, slab P+/N+, slab P++/N++ to the contact, plus the transfer-length contact resistance (assumed rho_c,
    contact width). Full ionization, Arora mobility (mrm.materials); the depletion edge from the abrupt model. For the
    L-shape the rib terms are 1-D path estimates (no current spreading; DEVSIM's AC value is the reference)."""
    an = analytic_depletion(xs, d, v_rev, t_k)
    w2 = xs.width_nm / 2 * 1e-9
    film, slab = xs.film_nm * 1e-9, xs.slab_nm * 1e-9
    gp, gpp, gc = d.plus_gap_nm * 1e-9, d.pp_gap_nm * 1e-9, d.contact_gap_nm * 1e-9
    xj = d.junction_offset_nm * 1e-9
    parts = {}
    total = np.zeros_like(np.asarray(v_rev, float))
    for side, nb, carrier in (("p", d.na_cm3, "p"), ("n", d.nd_cm3, "n")):
        rho_b = float(M.resistivity_doped_si_ohm_m(nb, t_k, carrier))
        rho_p = float(M.resistivity_doped_si_ohm_m(d.n_plus_cm3, t_k, carrier))
        rho_pp = float(M.resistivity_doped_si_ohm_m(d.n_pp_cm3, t_k, carrier))
        dep = (an["x_p_nm"] if side == "p" else an["x_n_nm"]) * 1e-9
        if d.topology == "lateral":
            l_rib = (w2 + (xj if side == "p" else -xj)) - dep
            r_rib = rho_b * np.maximum(l_rib, 0.0) / film
        else:
            # L-shape, 1-D paths (approximate): p, from the middle of the horizontal leg to the p-side rib edge in the
            # layer below the junction; n, along the top layer to the vertical leg, then across the n-side rib
            yj = d.y_junction_nm * 1e-9
            leg = xj + w2
            if side == "p":
                r_rib = rho_b * (0.5 * leg) / np.maximum(yj - dep, 1e-9)
            else:
                r_rib = rho_b * ((0.5 * leg) / np.maximum(film - yj - dep, 1e-9) + (w2 - xj) / film)
        r_slab = rho_b * gp / slab
        r_plus = rho_p * (gpp - gp) / slab
        r_pp = rho_pp * max(gc - gpp - d.contact_width_nm * 1e-9, 0.0) / slab
        rsh = rho_pp / slab
        rc = d.contact_rho_ohm_cm2 * 1e-4
        lt = math.sqrt(rc / rsh)
        r_c = math.sqrt(rc * rsh) / math.tanh(d.contact_width_nm * 1e-9 / lt)
        parts[side] = {"rho_body_ohm_m": rho_b, "rho_plus_ohm_m": rho_p, "rho_pp_ohm_m": rho_pp,
                       "rib_ohm_m_at_0v": float(np.interp(0.0, v_rev, r_rib)), "slab_body_ohm_m": r_slab,
                       "slab_plus_ohm_m": r_plus, "slab_pp_ohm_m": r_pp, "contact_ohm_m": r_c}
        total = total + r_rib + r_slab + r_plus + r_pp + r_c
    return total, parts


def contact_resistance_ohm_m(d: Doping, t_k=T_REF_K, xs: XSec | None = None):
    """Transfer-length contact resistance of both contacts (ohm*m), the term DEVSIM's ideal contacts omit."""
    slab = (xs.slab_nm if xs else 60.0) * 1e-9
    tot = 0.0
    for carrier in ("p", "n"):
        rsh = float(M.resistivity_doped_si_ohm_m(d.n_pp_cm3, t_k, carrier)) / slab
        rc = d.contact_rho_ohm_cm2 * 1e-4
        lt = math.sqrt(rc / rsh)
        tot += math.sqrt(rc * rsh) / math.tanh(d.contact_width_nm * 1e-9 / lt)
    return tot


# ------------------------------------------------------------------------------- small-signal EO response
def h_opt(f_hz, *, q_loaded, x_detune, kappa_e_frac=0.5, kappa_d_frac=0.0, q_loss=0.0, lam_m=LAMBDA0_M):
    """Exact small-signal TCMT response of the through port to a resonance modulation (03 section 5.6), normalized to
    its DC value. Resonance perturbation d omega_r with the loss change d(kappa_i/2) = q_loss * d omega_r.
    x_detune = 2 Delta/kappa with Delta = omega_r - omega_laser. Returns the complex ratio H(f)/H(0)."""
    w0 = 2 * math.pi * C0 / lam_m
    kap = w0 / q_loaded
    ke = kappa_e_frac * kap
    dlt = 0.5 * x_detune * kap
    s_in = 1.0
    a0 = math.sqrt(ke) * s_in / (kap / 2 - 1j * dlt)
    t0 = s_in - math.sqrt(ke) * a0

    def H(s):
        da = (1j - q_loss) * a0 / (s - 1j * dlt + kap / 2)
        dac = (-1j - q_loss) * np.conj(a0) / (s + 1j * dlt + kap / 2)
        return -math.sqrt(ke) * (np.conj(t0) * da + t0 * dac)
    s = 2j * math.pi * np.asarray(f_hz, float)
    return H(s) / H(0.0)


def f3db_ghz(h_func, f_max_ghz=400.0):
    """Electrical 3-dB frequency (|H|^2 = 1/2) of a normalized response: first crossing on a fine grid + refine."""
    from scipy.optimize import brentq
    fg = np.linspace(0.01, f_max_ghz, 8000) * 1e9
    mag = np.abs(h_func(fg)) ** 2 - 0.5
    idx = np.where(np.sign(mag[:-1]) != np.sign(mag[1:]))[0]
    if idx.size == 0:
        return float("nan")
    i = idx[0]
    return brentq(lambda f: abs(h_func(np.array([f]))[0]) ** 2 - 0.5, fg[i], fg[i + 1]) / 1e9


def eo_bandwidth_ghz(tau_rc_s, *, q_loaded, x_detune, q_loss=0.0, kappa_e_frac=0.5):
    """EO 3-dB bandwidth of RC (single pole) x exact optical response."""
    def h(f):
        return h_opt(f, q_loaded=q_loaded, x_detune=x_detune, q_loss=q_loss, kappa_e_frac=kappa_e_frac) / \
            (1 + 2j * math.pi * np.asarray(f) * tau_rc_s)
    return f3db_ghz(h)


# ---------------------------------------------------------------------------------------------- reports
def shift_pm(dneff, n_g, fill=1.0, lam_m=LAMBDA0_M):
    return lam_m / n_g * fill * np.asarray(dneff) * 1e12


def fill_for_cal5(v, dneff, n_g, lam_m=LAMBDA0_M, relief_pm=None):
    """Fill fraction reproducing CAL-5: (a) the endpoint definition (52.8 pm/V average over 0 to 1.6 V reverse) and
    (b) the least-squares fit to the three read-off ladder points; with the residual of (b)."""
    s1 = shift_pm(np.interp(CAL5_V, v, dneff), n_g, 1.0, lam_m)
    y = np.array(CAL5_SHIFT_PM, float)
    if relief_pm is not None:
        y = y + np.asarray(relief_pm, float)
    f_end = (CAL5_AVG_PM_PER_V * 1.6 + (relief_pm[-1] if relief_pm is not None else 0.0)) / s1[-1]
    f_lsq = float(s1 @ y / (s1 @ s1))
    res = f_lsq * s1 - y
    return {"fill_endpoint": float(f_end), "fill_lsq": f_lsq, "lsq_residual_pm": _fl(res, 4),
            "lsq_rms_pm": float(np.sqrt(np.mean(res ** 2))), "shift_at_fill1_pm": _fl(s1, 5)}


def ng_hypotheses(n_g_solver, n_eff, dneff_dt, t_k=T_REF_K, lam_m=LAMBDA0_M):
    """n_g that each dlambda/dT and FSR hypothesis implies for the CAL-5 conversion dlambda = (lambda/n_g) f dn_eff.
    - H_solver: the bent-mode n_g of this cross-section (FSR then sets the radius, not n_g);
    - H86 via R1: n_g = lambda (dn_eff/dT + n_eff alpha_L)/86 pm/K, if dn_eff/dT is held at the solver value (the
      only way 86 pm/K changes the pm-per-dn_eff conversion; 07 P0-3 anchors the self-heating gain on the heater);
    - FSR windows with the radius pinned at 7.5 um (R3 with radius and FSR both given): n_g = c/(FSR 2 pi R)."""
    alpha_l = float(M.alpha_l_si_per_k(t_k))
    out = {"H_solver": {"n_g": n_g_solver, "basis": "femwell bent mode at r_ref"},
           "H86_via_R1": {"n_g": lam_m * (dneff_dt + n_eff * alpha_l) / (H86_PM_PER_K * 1e-12),
                          "basis": "R1 with dn_eff/dT from the solver, dlambda/dT = 85.9 pm/K"}}
    L = 2 * math.pi * P_RING_R_UM * 1e-6
    for k, (f0, flo, fhi) in FSR_HYPOTHESES_THZ.items():
        out[f"FSR_{k}_R7.5um"] = {"n_g": C0 / (f0 * 1e12 * L), "n_g_range": [C0 / (fhi * 1e12 * L),
                                                                            C0 / (flo * 1e12 * L)],
                                  "basis": f"FSR {f0} THz ({flo}-{fhi}) at R = 7.5 um pinned"}
    return out


def selfheat_relief_pm(alpha_fca_v, fill, alpha_i0, p_in_mw, k_sh_pm_per_mw=405.0, eta_other=0.0):
    """First-order self-heating relief of a DC resonance read-out (07 P1-7): at the transmission dip of a near-
    critically coupled ring the stored-energy loss is ~ P_in, so the heat is eta(V) P_in with
    eta = (f alpha_FCA(V) + eta_other alpha_i0)/alpha_i(V), alpha_i(V) = alpha_i0 + f (alpha_FCA(V) - alpha_FCA(0)).
    Reverse bias lowers alpha_FCA, cools the ring and blue-shifts it: relief(V) = -K_SH P_in [eta(V) L(V) - eta(0)
    L(0)] (positive = the measured red shift is smaller than the EO shift). L = 4 k_e k_i / k^2 at fixed k_e."""
    a = np.asarray(alpha_fca_v, float)
    da = fill * (a - a[0])
    ai = alpha_i0 + da
    eta = (fill * a + eta_other * alpha_i0) / ai
    ke = alpha_i0       # critical at 0 V (in loss units)
    L = 4 * ke * ai / (ke + ai) ** 2
    heat = eta * L
    return -k_sh_pm_per_mw * p_in_mw * (heat - heat[0])


# ------------------------------------------------------------------------------------------ the pipeline
def _devsim_and_overlap(xs, d, v, wf, scale, ac=True, ac_v=None, oxide=True, ox_extent_um=OX_EXTENT_UM, t_k=T_REF_K,
                        keep=False, f_hi_hz=2e10, two_freq=False):
    """One isolated DEVSIM sweep, reduced to the optical overlap, the carrier totals, dQ/dV and the AC quantities
    (at ``v_ac``)."""
    v = np.asarray(v, float)
    dv = _isolated(run_devsim, xs, d, v, scale=scale, ac=ac, ac_v=ac_v, oxide=oxide, ox_extent_um=ox_extent_um,
                   t_k=t_k, f_hi_hz=f_hi_hz, two_freq=two_freq)
    w_nodes = wf.weight_at(dv["x_um"], dv["y_um"])
    dne, alpha = overlap(dv, w_nodes)
    ne, nh = carriers_per_m(dv)
    out = {"v": dv["v"], "dneff": dne, "alpha": alpha, "ne_per_m": ne, "nh_per_m": nh,
           "w_si_integral": float(w_nodes @ dv["vol_um2"]), "nodes_si": dv["nodes_si"], "mesh": dv["mesh"],
           "seconds": dv["seconds"]}
    if dv["v"].size >= 3:
        out["c_dq"] = -Q_E * np.gradient(ne, dv["v"], edge_order=2)
        out["c_dq_h"] = -Q_E * np.gradient(nh, dv["v"], edge_order=2)
    if ac:
        out["v_ac"] = dv["ac_v"]
        out["c_ac"] = dv["c_ac_f_per_m"]
        out["r_ac"] = dv["r_ac_ohm_m"]
        if two_freq:
            out["c_lo"] = dv["c_lo_f_per_m"]
    if keep:
        out["dv"] = dv
    return out


def _pick(v, arr, at):
    return float(np.interp(at, v, arr))


def _summary_numbers(v, dneff, alpha, n_g, v_ac, c, r):
    """Scalars used for convergence and Jacobians (c and r on the grid v_ac; r includes the contact term)."""
    eff_avg = float(shift_pm(_pick(v, dneff, 1.6), n_g)) / 1.6
    tau = _pick(v_ac, r, 1.0) * _pick(v_ac, c, 1.0)
    return {"dneff_1v6": _pick(v, dneff, 1.6), "dneff_1v": _pick(v, dneff, 1.0), "dneff_4v": _pick(v, dneff, 4.0),
            "dneff_fwd0v5": _pick(v, dneff, -0.5), "eff_avg_0_1v6_pm_per_v_fill1": eff_avg,
            "alpha_fca_0v_per_m": _pick(v, alpha, 0.0), "alpha_fca_2v_per_m": _pick(v, alpha, 2.0),
            "c_1v_f_per_m": _pick(v_ac, c, 1.0), "c_2v_f_per_m": _pick(v_ac, c, 2.0),
            "r_series_1v_ohm_m": _pick(v_ac, r, 1.0), "f_rc_1v_ghz": 1.0 / (2 * math.pi * tau) / 1e9}


def _summ(rr, n_g, r_contact):
    return _summary_numbers(rr["v"], rr["dneff"], rr["alpha"], n_g, rr["v_ac"], rr["c_ac"], rr["r_ac"] + r_contact)


AC_V = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
QUICK = {"scale": 2.8284, "ladder": [4.0, 3.3636, 2.8284], "study": 4.0}   # --quick (tests)
# Production DEVSIM scales (1 = 1 nm at the junction). Peak memory is about 50 kB per mesh node for the lateral
# junction and about 90 kB per node for the L-shape (measured: 2.86 GB at 31k nodes), so the L-shape (fine in x and
# y) runs two levels coarser than the lateral junction; both keep a three-level ladder
# ending at the production mesh, and run side studies (Jacobian, oxide extent, temperature) at ``study``.
PROD = {"lshape": {"scale": 2.0, "ladder": [2.8284, 2.3784, 2.0], "study": 2.8284},
        "lateral": {"scale": 1.0, "ladder": [2.0, 1.4142, 1.0], "study": 1.4142}}
V_SUBSET = np.array([-0.5, 0.0, 1.0, 1.6, 2.0, 4.0])


def jacobian(xs, d, wf_ref, n_g, *, quick, scale, mode_res_um, log=print, light=False):
    """Central-difference Jacobian of the summary numbers with respect to the geometry and doping assumptions.
    Geometry steps re-solve the mode (and n_g); doping steps reuse it. Densities are differentiated per decade."""
    vj = V_SUBSET
    steps = [("width_nm", 20.0, "geom"), ("film_nm", 10.0, "geom"), ("slab_nm", 10.0, "geom"),
             ("log10_na_cm3", 0.1, "dop"), ("log10_nd_cm3", 0.1, "dop"), ("junction_offset_nm", 20.0, "dop"),
             ("plus_gap_nm", 50.0, "dop"), ("contact_gap_nm", 200.0, "dop"), ("grading_nm", 10.0, "dop_fwd"),
             ("outer_side", None, "swap")]
    if d.topology == "lshape":
        steps.insert(6, ("y_junction_nm", 10.0, "dop"))
    if quick:
        steps = [steps[0], steps[3]]
    elif light:
        steps = [x for x in steps if x[0] in ("width_nm", "film_nm", "slab_nm", "log10_na_cm3", "log10_nd_cm3",
                                              "junction_offset_nm", "y_junction_nm")]
    rows, base = {}, None
    for name, h, kind in steps:
        vals = []
        signs = (+1, -1) if kind in ("geom", "dop") else (+1, 0)
        for sgn in signs:
            xs2, d2, wf, ng2 = xs, d, wf_ref, n_g
            if sgn == 0:
                if base is None:
                    r0 = _devsim_and_overlap(xs, d, vj, wf_ref, scale)
                    base = dict(_summ(r0, n_g, contact_resistance_ohm_m(d, T_REF_K, xs)), n_g=n_g)
                vals.append(base)
                continue
            if kind == "geom":
                xs2 = dataclasses.replace(xs, **{name: getattr(xs, name) + sgn * h})
            elif kind == "swap":
                xs2 = dataclasses.replace(xs, r_ref_um=-xs.r_ref_um)    # mirror: p side outside
            elif name.startswith("log10_"):
                key = name[6:]
                d2 = dataclasses.replace(d, **{key: getattr(d, key) * 10 ** (sgn * h)})
            else:
                d2 = dataclasses.replace(d, **{name: getattr(d, name) + sgn * h})
            if kind in ("geom", "swap"):
                mp = _isolated(mode_properties, xs2, d2, res_um=mode_res_um, radius_um=xs2.r_ref_um,
                               with_dn_check=False)
                wf, ng2 = weight_field(mp, xs2, d2), mp["n_g"]
            r = _devsim_and_overlap(xs2, d2, vj, wf, scale)
            vals.append(dict(_summ(r, ng2, contact_resistance_ohm_m(d2, T_REF_K, xs2)), n_g=ng2))
        denom = 2 * h if kind in ("geom", "dop") else (h if h else 1.0)
        rows[name] = {k: (vals[0][k] - vals[1][k]) / denom for k in vals[0]}
        log(f"  jacobian {name}: d(pm/V)/d = {rows[name]['eff_avg_0_1v6_pm_per_v_fill1']:.4g}, "
            f"d(alpha0)/d = {rows[name]['alpha_fca_0v_per_m']:.4g}, d(C1)/d = {rows[name]['c_1v_f_per_m']:.3g}")
    names = list(rows)
    keys = list(next(iter(rows.values())))
    return {"d": names, "step": {n: h for n, h, _ in steps if n in rows},
            "of": {k: [_f(rows[n][k], 5) for n in names] for k in keys},
            "base": {k: _f(v_, 7) for k, v_ in (base or {}).items()},
            "note": ("central differences (grading: forward step from the nominal 2 nm; outer_side: the difference "
                     "'p outside' minus 'n outside' (the nominal), not a derivative); densities per decade (log10); "
                     "geometry steps re-solve the mode and n_g; DEVSIM scale %.4g, mode res %.4g um; "
                     "eff_avg is at fill 1 with the perturbed n_g" % (scale, mode_res_um))}


def extract(xs_name="rib160", *, topology=DEFAULT_TOPOLOGY, quick=False, log=print, doping=None,
            do_crosschecks=True, light=False):
    """Run the extraction for one cross-section and junction topology. ``light`` (used for the non-default topology)
    keeps the production sweep, three-level DEVSIM and mode convergence, the direct re-solve and the depletion check,
    and drops the oxide-extent study, tidy3d (same mode as the default artefact), the temperature run and the
    non-geometry/doping Jacobian rows. Returns (artefact dict, sidecar dict)."""
    t_start = time.time()
    d = doping or nominal_doping(xs_name, topology)
    xs0 = XSECS[xs_name]
    r_ref, r_ref_info = resolve_r_ref(xs0, d, log=log, res_um=0.04 if quick else 0.02)
    xs = dataclasses.replace(xs0, r_ref_um=round(r_ref, 6))
    log(f"[{xs_name}] r_ref = {xs.r_ref_um:.5f} um ({r_ref_info['source']})")
    v = v_grid(quick)
    cfg = QUICK if quick else PROD[d.topology]
    scale = cfg["scale"]
    study = cfg["study"]
    mode_res = 0.03 if quick else 0.010
    log(f"[{xs_name}] mode solves (femwell order 2, core res {mode_res} um; bent R = {xs.r_ref_um} um and straight)")
    mp = _isolated(mode_properties, xs, d, res_um=mode_res, radius_um=xs.r_ref_um, with_dn_check=True)
    mps = _isolated(mode_properties, xs, d, res_um=mode_res, radius_um=np.inf, with_dn_check=not quick)
    wf, wfs = weight_field(mp, xs, d), weight_field(mps, xs, d)
    n_g = mp["n_g"]
    log(f"  bent: n_eff {mp['n_eff']:.6f}, n_g {n_g:.5f}, dneff/dnSi {mp['index_weight']['si']:.5f} "
        f"(fd {mp['dneff_dnsi_fd']:.5f}), power fraction {mp['power_fraction_si']:.4f}, {mp['elements']} el, "
        f"{mp['seconds']} s; straight n_eff {mps['n_eff']:.6f}, n_g {mps['n_g']:.5f}")
    selection = None
    if doping is None and d.topology == "lshape" and not quick:
        log(f"[{xs_name}] nominal doping selection (fill = {FILL_NOMINAL} at CAL-5)")
        d, selection = select_doping(xs, d, wf, n_g, log=log)
        log(f"  selected N_A = N_D = {d.na_cm3:.3g} cm^-3 ({selection['status']})")
    log(f"[{xs_name}] DEVSIM production sweep ({v.size} biases, DC, scale {scale})")
    r = _devsim_and_overlap(xs, d, v, wf, scale, ac=False, keep=True)
    dv = r["dv"]
    log(f"  {dv['nodes_si']} Si nodes, {dv['seconds']} s")
    v_acp = AC_V if not quick else np.array([-0.5, 0.0, 1.0, 2.0, 4.0])
    log(f"[{xs_name}] DEVSIM small-signal AC at {v_acp.tolist()} V (20 GHz)")
    # several isolated AC jobs (DEVSIM keeps each AC matrix allocated); 0 V is in all so dn_eff has one reference
    parts = [_devsim_and_overlap(xs, d, np.union1d(chunk, [0.0]), wf, scale, ac=True, ac_v=chunk)
             for chunk in np.array_split(v_acp, 4 if not quick else 2)]
    ra = {k: np.concatenate([q[k] for q in parts]) for k in ("v", "dneff", "v_ac", "c_ac", "r_ac")}
    dne_straight, alpha_straight = overlap(dv, wfs.weight_at(dv["x_um"], dv["y_um"]))
    ne, nh = r["ne_per_m"], r["nh_per_m"]
    c_dq, c_dq_h = r["c_dq"], r["c_dq_h"]
    from scipy.interpolate import PchipInterpolator
    r_contact = contact_resistance_ohm_m(d, T_REF_K, xs)
    r_dev = PchipInterpolator(ra["v_ac"], ra["r_ac"])(v)
    r_total = r_dev + r_contact
    c_grid = c_dq                                   # low-frequency C(V) on the full grid (checked against AC)
    r_an, r_parts = series_resistance_analytic(xs, d, v)
    dep = depletion_width_nm(dv, xs, d)
    an = analytic_depletion(xs, d, v)
    an_d = analytic_depletion(xs, d, v, debye=True)
    an_n = analytic_dneff(xs, d, wf, v)
    i0 = int(np.argmin(np.abs(v)))
    # loss ratio q(V) = -lambda (d alpha/dV)/(4 pi d dneff/dV) (03 section 5.4 sign: positive for depletion)
    q_loss = -LAMBDA0_M * np.gradient(r["alpha"], v) / (4 * math.pi * np.gradient(r["dneff"], v))
    summary = _summary_numbers(v, r["dneff"], r["alpha"], n_g, v, c_grid, r_total)
    log(f"  dneff(1.6 V) {summary['dneff_1v6']:.5e}; {summary['eff_avg_0_1v6_pm_per_v_fill1']:.3f} pm/V at fill 1; "
        f"alpha_FCA(0) {summary['alpha_fca_0v_per_m']:.2f} /m; C(1 V) {summary['c_1v_f_per_m']:.4e} F/m; "
        f"r_s(1 V) {summary['r_series_1v_ohm_m']:.4e} ohm m; f_RC(1 V) {summary['f_rc_1v_ghz']:.1f} GHz")

    # ------------------------------------------------------------------ convergence studies
    conv = {}
    vc = V_SUBSET
    scales = cfg["ladder"]
    log(f"[{xs_name}] DEVSIM mesh convergence (scales {scales})")
    sc_rows, sc_ok, sc_failed = [], [], []
    for sc in scales:
        try:
            rr = _devsim_and_overlap(xs, d, vc, wf, sc)
        except Exception as exc:          # a coarse level may not converge at high doping; record it
            sc_failed.append({"scale": sc, "error": str(exc)[:200]})
            log(f"  scale {sc}: FAILED ({str(exc)[:80]})")
            continue
        sn = _summ(rr, n_g, r_contact)
        sn["nodes_si"] = rr["nodes_si"]
        sc_rows.append(sn)
        sc_ok.append(sc)
        log(f"  scale {sc}: {sn['nodes_si']} Si nodes, dneff(1.6) {sn['dneff_1v6']:.7e}, C(1) "
            f"{sn['c_1v_f_per_m']:.6e}, alpha(0) {sn['alpha_fca_0v_per_m']:.5f}, r(1) {sn['r_series_1v_ohm_m']:.6e}")
    scales = sc_ok
    ckeys = ("dneff_1v6", "dneff_1v", "dneff_4v", "dneff_fwd0v5", "alpha_fca_0v_per_m", "alpha_fca_2v_per_m",
             "c_1v_f_per_m", "c_2v_f_per_m", "r_series_1v_ohm_m")
    conv["devsim_mesh"] = {"scales": scales, "h_junction_nm": [1.0 * x for x in scales],
                           "nodes_si": [x["nodes_si"] for x in sc_rows], "production_scale": scale,
                           "failed_levels": sc_failed,
                           **{k: richardson([x[k] for x in sc_rows], scales, order=2) for k in ckeys}}
    if not quick and not light:
        log(f"[{xs_name}] oxide-extent convergence and fringe fraction")
        ox_rows = []
        for ext in (0.3, 0.6, 1.2):
            rr = _devsim_and_overlap(xs, d, np.array([0.0, 1.0, 2.0]), wf, study, ox_extent_um=ext)
            ox_rows.append({"ox_extent_um": ext, "c_0v_f_per_m": _pick(rr["v_ac"], rr["c_ac"], 0.0),
                            "c_1v_f_per_m": _pick(rr["v_ac"], rr["c_ac"], 1.0),
                            "c_2v_f_per_m": _pick(rr["v_ac"], rr["c_ac"], 2.0)})
        rr = _devsim_and_overlap(xs, d, np.array([0.0, 1.0, 2.0]), wf, study, oxide=False)
        rr_ox = _devsim_and_overlap(xs, d, np.array([0.0, 1.0, 2.0]), wf, study, ac=False)
        c_no = _pick(rr["v_ac"], rr["c_ac"], 1.0)
        an_no = analytic_dneff(xs, d, wf, np.array([0.0, 1.0, 2.0]))
        conv["oxide_effect_on_dneff"] = {
            "v": [1.0, 2.0], "with_oxide": [_pick(rr_ox["v"], rr_ox["dneff"], x) for x in (1.0, 2.0)],
            "no_oxide": [_pick(rr["v"], rr["dneff"], x) for x in (1.0, 2.0)],
            "analytic": [float(an_no["dneff"][1]), float(an_no["dneff"][2])],
            "analytic_vs_no_oxide_rel": [_f(float(an_no["dneff"][k]) / _pick(rr["v"], rr["dneff"], x) - 1, 4)
                                         for k, x in ((1, 1.0), (2, 2.0))],
            "note": "the fringe field through the oxide depletes extra carriers near the Si surfaces; the abrupt "
                    "model has no fringe, so it should be compared with the no-oxide DEVSIM run"}
        conv["oxide_extent"] = {"rows": ox_rows, "no_oxide_c_1v_f_per_m": c_no,
                                "fringe_fraction_1v": 1.0 - c_no / ox_rows[1]["c_1v_f_per_m"],
                                "rel_change_production_to_double_1v": ox_rows[2]["c_1v_f_per_m"]
                                / ox_rows[1]["c_1v_f_per_m"] - 1.0,
                                "note": "Neumann outer boundary; production uses %.1f um" % OX_EXTENT_UM}
        log(f"  C(1 V) at 0.3/0.6/1.2 um: {[round(x['c_1v_f_per_m'] * 1e10, 5) for x in ox_rows]} e-10 F/m; "
            f"no oxide {c_no:.4e}")
    log(f"[{xs_name}] mode mesh convergence")
    res_list = [0.06, 0.042, 0.03] if quick else ([0.02, 0.0141, 0.010] if light else
                                                   [0.04, 0.0283, 0.02, 0.0141, 0.010])
    m_rows = []
    for res in res_list:
        mpr = _isolated(mode_properties, xs, d, res_um=res, radius_um=xs.r_ref_um, with_dn_check=False)
        dn_r, a_r = overlap(dv, weight_field(mpr, xs, d).weight_at(dv["x_um"], dv["y_um"]))
        m_rows.append({"res_um": res, "elements": mpr["elements"], "n_eff": mpr["n_eff"], "n_g": mpr["n_g"],
                       "dneff_dnsi": mpr["index_weight"]["si"], "dneff_1v6": _pick(v, dn_r, 1.6),
                       "alpha_fca_0v_per_m": _pick(v, a_r, 0.0)})
        log(f"  res {res}: {mpr['elements']} el, n_eff {mpr['n_eff']:.7f} n_g {mpr['n_g']:.6f} dneff(1.6) "
            f"{m_rows[-1]['dneff_1v6']:.7e} alpha(0) {m_rows[-1]['alpha_fca_0v_per_m']:.5f}")
    conv["mode_mesh"] = {"res_um": res_list, "elements": [x["elements"] for x in m_rows], "production_res_um": mode_res,
                         **{k: richardson([x[k] for x in m_rows], res_list, order=4)
                            for k in ("n_eff", "n_g", "dneff_dnsi", "dneff_1v6", "alpha_fca_0v_per_m")}}
    if not quick:
        vm = np.array([0.025, 1.025, 1.575, 3.975])
        rr = _devsim_and_overlap(xs, d, np.union1d(vm, [0.0]), wf, scale, ac=False)
        solved = np.interp(vm, rr["v"], rr["dneff"])
        lin = np.interp(vm, v, r["dneff"])
        conv["bias_grid"] = {"v_mid": _fl(vm), "dneff_solved": _fl(solved), "dneff_linear_interp": _fl(lin),
                             "max_rel_error": _f(np.max(np.abs(lin - solved) / np.abs(solved)), 4),
                             "max_abs_error_rel_to_dneff_1v6": _f(np.max(np.abs(lin - solved))
                                                                  / abs(_pick(v, r["dneff"], 1.6)), 4),
                             "note": "linear interpolation of the 0.05 V table at mid-points; the relative error "
                                     "is largest just above 0 V where dn_eff itself is small"}
        log(f"  bias-grid interpolation error {conv['bias_grid']['max_rel_error']}")
        rr = _devsim_and_overlap(xs, d, np.array([0.0, 1.0]), wf, study, f_hi_hz=5e10)
        rr2 = _devsim_and_overlap(xs, d, np.array([0.0, 1.0]), wf, study, f_hi_hz=2e10, two_freq=True)
        conv["ac_frequency"] = {"r_1v_20ghz": _pick(rr2["v_ac"], rr2["r_ac"], 1.0),
                                "r_1v_50ghz": _pick(rr["v_ac"], rr["r_ac"], 1.0),
                                "rel": _f(_pick(rr["v_ac"], rr["r_ac"], 1.0) / _pick(rr2["v_ac"], rr2["r_ac"], 1.0)
                                          - 1, 4),
                                "c_20ghz_over_c_1mhz_1v": _f(_pick(rr2["v_ac"], rr2["c_ac"], 1.0)
                                                             / _pick(rr2["v_ac"], rr2["c_lo"], 1.0), 5),
                                "note": "series R-C: Re Z must not depend on frequency"}

    # ------------------------------------------------------------------ cross-checks
    checks = {}
    if do_crosschecks:
        vd = np.array([-0.5, 1.0, 2.0, 4.0]) if not quick else np.array([1.0, 4.0])
        log(f"[{xs_name}] direct re-solve check at {vd.tolist()} V")
        keep_rows = [int(np.argmin(np.abs(v - x))) for x in [0.0] + vd.tolist()]
        dv_red = {"x_um": dv["x_um"], "y_um": dv["y_um"], "v": dv["v"][keep_rows], "n": dv["n"][keep_rows],
                  "p": dv["p"][keep_rows], "vol_um2": dv["vol_um2"]}
        dn_dir, a_dir, nel, n_base = _isolated(direct_resolve_dneff, xs, d, dv_red, vd, mode_res, xs.r_ref_um)
        dn_pert = np.interp(vd, v, r["dneff"])
        a_pert = np.interp(vd, v, r["alpha"])
        checks["perturbation_vs_direct_resolve"] = {
            "v": _fl(vd), "dneff_perturbative": _fl(dn_pert), "dneff_direct": _fl(dn_dir),
            "rel_diff": _fl((dn_pert - dn_dir) / dn_dir, 4), "alpha_perturbative": _fl(a_pert),
            "alpha_direct": _fl(a_dir), "alpha_rel_diff": _fl((a_pert - a_dir) / a_dir, 4), "mode_elements": nel,
            "note": "direct: femwell re-solve with the element-averaged complex permittivity (7-level barycentric "
                    "sub-sampling of the DEVSIM fields); a different discretization of the carriers and no "
                    "first-order truncation"}
        log(f"  rel diff dneff {checks['perturbation_vs_direct_resolve']['rel_diff']}, alpha "
            f"{checks['perturbation_vs_direct_resolve']['alpha_rel_diff']}")
        if not quick and not light:
            log(f"[{xs_name}] tidy3d cross-check (straight guide)")
            try:
                t3 = _isolated(tidy3d_check, xs, d)
                fw_conv = conv["mode_mesh"]
                t3["femwell_straight"] = {"n_eff": mps["n_eff"], "dneff_dnsi": mps["index_weight"]["si"]}
                t3["delta_n_eff"] = _f(t3["n_eff_conv"]["extrapolated"] - mps["n_eff"], 4)
                t3["delta_dneff_dnsi"] = _f(t3["dneff_dnsi_conv"]["extrapolated"] - mps["index_weight"]["si"], 4)
                checks["tidy3d_vs_femwell"] = t3
                log(f"  tidy3d n_eff {t3['n_eff_conv']['extrapolated']} (dneff/dnSi "
                    f"{t3['dneff_dnsi_conv']['extrapolated']}) vs femwell {mps['n_eff']:.6f} "
                    f"({mps['index_weight']['si']:.5f})")
            except Exception as exc:     # tidy3d is a cross-check only
                checks["tidy3d_vs_femwell"] = {"error": repr(exc)}
                log(f"  tidy3d failed: {exc!r}")
    checks["weight_integral"] = {"over_devsim_si_nodes": r["w_si_integral"], "femwell_si": mp["index_weight"]["si"],
                                 "fd_resolve_dn_1e-4": mp["dneff_dnsi_fd"],
                                 "rel_diff_weight_vs_fd": _f((mp["index_weight"]["si"] - mp["dneff_dnsi_fd"])
                                                             / mp["dneff_dnsi_fd"], 4),
                                 "rel_diff_nodes_vs_femwell": _f(r["w_si_integral"] / mp["index_weight"]["si"] - 1, 4)}
    checks["index_weight_identity"] = {"sum_ng_i_w_i": mp["identity_sum_ng_i_w_i"], "n_g": n_g,
                                       "rel_residual": _f(mp["identity_rel_residual"], 4)}
    sl = (v > 0.1) & (v < 3.9)
    c_dq_at_ac = np.interp(ra["v_ac"], v, c_dq)
    rel_c = (ra["c_ac"] - c_dq_at_ac) / ra["c_ac"]
    rev_ac = (ra["v_ac"] >= 0.0)
    checks["capacitance_ac_vs_dqdv"] = {
        "v": _fl(ra["v_ac"]), "c_ac_f_per_m": _fl(ra["c_ac"]), "c_dqdv_f_per_m": _fl(c_dq_at_ac),
        "rel_diff": _fl(rel_c, 4), "max_rel_diff_reverse": _f(np.max(np.abs(rel_c[rev_ac])), 4),
        "electron_hole_balance_max_rel": _f(np.max(np.abs(c_dq - c_dq_h)[sl] / c_dq[sl]), 4),
        "dneff_ac_run_vs_dc_run_max_rel": _f(np.max(np.abs(ra["dneff"] - np.interp(ra["v"], v, r["dneff"]))[
            np.abs(ra["v"]) > 0.1] / np.abs(ra["dneff"][np.abs(ra["v"]) > 0.1])), 4),
        "note": "C(V) of the artefact is the low-frequency dQ/dV of the 0.05 V DC sweep; AC (20 GHz) at the "
                "listed biases; R(V) is the AC Re Z interpolated (PCHIP) plus the contact term"}
    sd = v >= 0.0
    sts = max(1, int(sd.sum()) // 9)
    rel_ab = (dep["w_nm"] - an["w_nm"]) / an["w_nm"]
    rel_db = (dep["w_nm"] - an_d["w_nm"]) / an_d["w_nm"]
    checks["depletion_width_devsim_vs_abrupt"] = {
        "cut_nm": dep["cut_nm"], "cut_axis_fixed": dep["cut"], "v": _fl(v[sd][::sts]), "w_devsim_nm": _fl(dep["w_nm"][sd][::sts], 5),
        "w_abrupt_nm": _fl(an["w_nm"][sd][::sts], 5), "w_abrupt_debye_nm": _fl(an_d["w_nm"][sd][::sts], 5),
        "rel_diff_vs_abrupt": _fl(rel_ab[sd][::sts], 4), "mean_rel_diff_vs_abrupt": _f(np.nanmean(rel_ab[sd]), 4),
        "range_rel_diff_vs_abrupt": [_f(np.nanmin(rel_ab[sd]), 4), _f(np.nanmax(rel_ab[sd]), 4)],
        "mean_rel_diff_vs_debye": _f(np.nanmean(rel_db[sd]), 4),
        "note": "05 C7c reported 5-8 % (DEVSIM below the abrupt formula) with mixed permittivities; here both use "
                "11.7 eps0; the edges are the half-doping points of the simulated carriers on a mid-film cut"}
    checks["analytic_vs_devsim"] = {
        "dneff_rel_diff_at": {f"{vv:g}V": _f((_pick(v, an_n["dneff"], vv) - _pick(v, r["dneff"], vv))
                                             / _pick(v, r["dneff"], vv), 4) for vv in (0.5, 1.0, 1.6, 2.0, 4.0)},
        "c_rel_diff_at": {f"{vv:g}V": _f((_pick(v, an["c_f_per_m"], vv) - _pick(v, c_grid, vv))
                                         / _pick(v, c_grid, vv), 4) for vv in (0.0, 1.0, 2.0, 4.0)},
        "alpha_rel_diff_0v": _f((an_n["alpha_fca_per_m"][i0] - r["alpha"][i0]) / r["alpha"][i0], 4),
        "r_series_analytic_vs_devsim_1v": _f((_pick(v, r_an, 1.0) - _pick(v, r_total, 1.0)) / _pick(v, r_total, 1.0),
                                             4),
        "note": "analytic = abrupt box depletion through the 2-D femwell weight; its C has no fringe field; its r_s "
                "is the 1-D series sum of sheet resistances (no current spreading)"}
    if r_ref_info.get("source", "").startswith("modes_"):
        try:
            ma = json.loads((OUT_DIR / r_ref_info["source"]).read_text())
            checks["modes_artefact_consistency"] = {
                "modes_n_eff": ma.get("n_eff"), "junction_n_eff_bent": mp["n_eff"],
                "modes_n_g": ma.get("n_g"), "junction_n_g_bent": n_g,
                "modes_index_weight_si": (ma.get("index_weights") or {}).get("si"),
                "junction_index_weight_si": mp["index_weight"]["si"],
                "rel_diff_n_g": _f(n_g / ma["n_g"] - 1, 4) if ma.get("n_g") else None,
                "rel_diff_index_weight_si": _f(mp["index_weight"]["si"] / ma["index_weights"]["si"] - 1, 4)
                if ma.get("index_weights") else None,
                "note": "modes uses the exact anisotropic bend transform and its own mesh; this file the isotropic "
                        "conformal map (femwell). Same geometry, materials, r_ref."}
        except Exception as exc:
            checks["modes_artefact_consistency"] = {"error": repr(exc)}
    checks["bend_effect"] = {"dneff_1v6_straight": _pick(v, dne_straight, 1.6),
                             "dneff_1v6_bent": _pick(v, r["dneff"], 1.6),
                             "rel": _f(_pick(v, r["dneff"], 1.6) / _pick(v, dne_straight, 1.6) - 1, 4),
                             "alpha_0v_rel": _f(r["alpha"][i0] / alpha_straight[i0] - 1, 4)}

    # ------------------------------------------------------------------ sensitivities
    log(f"[{xs_name}] Jacobian")
    jac = jacobian(xs, d, wf, n_g, quick=quick, scale=study,
                   mode_res_um=0.06 if quick else 0.016, log=log, light=light)
    dne_sb, a_sb = overlap(dv, wf.weight_at(dv["x_um"], dv["y_um"]), model="soref1987")
    plasma_alt = {"soref1987": {"dneff_1v6": _pick(v, dne_sb, 1.6),
                                "ratio_1v6": _f(_pick(v, dne_sb, 1.6) / summary["dneff_1v6"], 4),
                                "alpha_0v_per_m": _pick(v, a_sb, 0.0), "ratio_alpha_0v": _f(a_sb[i0] / r["alpha"][i0], 4)}}
    temp = None
    if not quick and not light:
        log(f"[{xs_name}] temperature sensitivity (DEVSIM at 340 K vs 300 K)")
        t_hot = 340.0
        dT = t_hot - T_REF_K
        rt = _devsim_and_overlap(xs, d, vc, wf, study, t_k=t_hot, keep=True)
        rb = _devsim_and_overlap(xs, d, vc, wf, study, t_k=T_REF_K, keep=True)
        # absolute change of the 0 V state (same mesh): what the runtime's "V + dV_bi/dT dT" lookup must reproduce
        wn = wf.weight_at(rb["dv"]["x_um"], rb["dv"]["y_um"]) * rb["dv"]["vol_um2"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", M.MaterialRangeWarning)
            i0b = int(np.argmin(np.abs(vc)))
            dn_b, _ = M.plasma_dispersion(rb["dv"]["n"][i0b], rb["dv"]["p"][i0b], warn=False)
            dn_t, _ = M.plasma_dispersion(rt["dv"]["n"][i0b], rt["dv"]["p"][i0b], warn=False)
        fac = float(M.plasma_temperature_factor(t_hot, "dn")) / float(M.plasma_temperature_factor(T_REF_K, "dn"))
        abs0_carriers = float((np.asarray(dn_t) - np.asarray(dn_b)) @ wn)
        abs0_total = float((np.asarray(dn_t) * fac - np.asarray(dn_b)) @ wn)
        dvbi = float(M.dvbi_dt_v_per_k(d.na_cm3, d.nd_cm3, T_REF_K)) * dT
        shift_model = np.interp(vc + dvbi, v, r["dneff"]) - 0.0      # runtime: table(V + dV_bi/dT dT)
        actual = rt["dneff"] + abs0_carriers                           # hot, relative to the cold 0 V state
        temp = {"t_k": [T_REF_K, t_hot], "v": _fl(vc),
                "d_dneff_dt_per_k": _fl((rt["dneff"] - rb["dneff"]) / dT),
                "dln_dneff_dt_per_k_1v6": _f((_pick(vc, rt["dneff"], 1.6) / _pick(vc, rb["dneff"], 1.6) - 1) / dT),
                "dln_alpha_dt_per_k_0v": _f((_pick(vc, rt["alpha"], 0.0) / _pick(vc, rb["alpha"], 0.0) - 1) / dT),
                "dln_c_dt_per_k_1v": _f((_pick(rt["v_ac"], rt["c_ac"], 1.0) / _pick(rb["v_ac"], rb["c_ac"], 1.0) - 1)
                                        / dT),
                "dln_r_dt_per_k_1v": _f((_pick(rt["v_ac"], rt["r_ac"], 1.0) / _pick(rb["v_ac"], rb["r_ac"], 1.0) - 1)
                                        / dT),
                "dneff_0v_hot_minus_cold_carriers": _f(abs0_carriers),
                "dneff_0v_hot_minus_cold_with_plasma_t_factor": _f(abs0_total),
                "dvbi_dt_v_per_k": _f(dvbi / dT),
                "voltage_shift_model": {"v": _fl(vc), "table_at_v_plus_dvbi": _fl(shift_model), "devsim_hot": _fl(actual),
                                        "max_abs_err": _f(np.max(np.abs(shift_model - actual))),
                                        "note": "runtime temperature axis (plant_electrical handoff): look up the "
                                                "300 K table at V + dV_bi/dT (T - T_ref); compared here with the "
                                                "DEVSIM solution at 340 K referenced to the cold 0 V state "
                                                "(carriers only)"},
                "note": "n_i(T), V_t(T), Arora mu(T) in DEVSIM; plasma per-carrier factors from "
                        "mrm.materials.plasma_temperature_factor (assumed); optical mode held at 300 K"}

    # ------------------------------------------------------------------ reports and files
    reports = build_reports(xs, d, v, r["dneff"], r["alpha"], c_grid, r_total, q_loss, mp, jac)
    art = {
        "schema": SCHEMA,
        "xsec": dict(xs.as_dict(), provenance=GEOMETRY_PROVENANCE, ranges=GEOMETRY_RANGES_NM),
        "lambda0_nm": LAMBDA0_M * 1e9, "t_ref_k": T_REF_K, "fill_fraction": 1.0, "full_circumference": True,
        "method": "devsim", "plasma_model": "nsm2011",
        "doping": dict(d.as_dict(), ranges=DOPING_RANGES, provenance=DOPING_PROVENANCE,
                       v_bi_v=float(M.vbi_v(d.na_cm3, d.nd_cm3, T_REF_K)), selection=selection,
                       junction_length_nm=junction_length_nm(xs, d)),
        "v_reverse_v": _fl(v),
        "dneff": _fl(r["dneff"], 8),
        "alpha_fca_per_m": _fl(r["alpha"], 8),
        "capacitance_f_per_m": _fl(c_grid, 8),
        "r_series_ohm_m": _fl(r_total, 8),
        "units": {"v_reverse_v": "V (reverse positive)", "dneff": "1 (relative to 0 V, bent mode at r_ref)",
                  "alpha_fca_per_m": "1/m intensity, absolute", "capacitance_f_per_m": "F/m of junction length",
                  "r_series_ohm_m": "ohm*m (R_s = r/(fill*L))"},
        "aux": {
            "dneff_straight": _fl(dne_straight, 7), "alpha_fca_straight_per_m": _fl(alpha_straight, 7),
            "ac_v": _fl(ra["v_ac"]), "capacitance_ac_20ghz_f_per_m": _fl(ra["c_ac"], 7),
            "r_series_devsim_ac_ohm_m": _fl(ra["r_ac"], 7), "r_series_devsim_ohm_m": _fl(r_dev, 7),
            "r_series_contact_ohm_m": _f(r_contact), "r_series_analytic_ohm_m": _fl(r_an, 7),
            "r_series_analytic_parts": {k: {kk: _f(vv) for kk, vv in p.items()} for k, p in r_parts.items()},
            "electrons_per_m": _fl(ne, 10), "holes_per_m": _fl(nh, 10),
            "depletion_width_nm": _fl(dep["w_nm"], 6), "loss_ratio_q": _fl(q_loss, 5),
            "analytic_dneff": _fl(an_n["dneff"], 7), "analytic_alpha_fca_per_m": _fl(an_n["alpha_fca_per_m"], 7),
            "analytic_capacitance_f_per_m": _fl(an["c_f_per_m"], 7),
            "dneff_soref1987": _fl(dne_sb, 7), "alpha_fca_soref1987_per_m": _fl(a_sb, 7),
        },
        "overlap": {"r_ref_um": xs.r_ref_um, "r_ref_source": r_ref_info, "n_eff_bent": mp["n_eff"], "n_eff_straight": mps["n_eff"],
                    "n_g_bent": n_g, "n_g_straight": mps["n_g"],
                    "dneff_dnsi_bent": mp["index_weight"]["si"], "dneff_dnsi_straight": mps["index_weight"]["si"],
                    "index_weights_bent": mp["index_weight"], "power_fraction_si_bent": mp["power_fraction_si"],
                    "dneff_dt_per_k_bent": mp["dneff_dt_per_k"],
                    "weight": "index weight c eps0 n (1+x/R)^2 |E|^2 / P (03 verification log: not the power "
                              "fraction); x > 0 is the outer side (n side)",
                    "mode_solver": "femwell order 2, conformal bend, core res %.4g um" % mode_res},
        "depletion_check": checks["depletion_width_devsim_vs_abrupt"],
        "reports": reports,
        "jacobian": jac,
        "plasma_model_alternative": plasma_alt,
        "temperature_sensitivity": temp,
        "summary": {k: _f(vv, 6) for k, vv in summary.items()},
    }
    side = {
        "schema": SCHEMA, "artefact": artefact_name(xs_name, d.topology) + ".json", "created_utc": _now_utc(),
        "command": "python -m mrm.extraction.junction --xsec %s --topology %s%s" % (
            xs_name, d.topology, " --quick" if quick else ""),
        "study_level": "quick" if quick else ("light" if light else "full"),
        "inputs": {
            "geometry": {**xs.as_dict(), "provenance": "assumed"},
            "doping": {**d.as_dict(), "provenance": "assumed"},
            "materials": {**M.materials_provenance(), "provenance": "literature"},
            "plasma_model": "nsm2011 (Nedeljkovic, Soref, Mashanovich 2011, 1.3 um row)",
            "permittivity": {"si": EPS_SI_R, "sio2": EPS_SIO2_R},
            "lambda0_nm": LAMBDA0_M * 1e9, "t_k": T_REF_K,
        },
        "tool_versions": _tool_versions(),
        "mesh": {"devsim": dv["mesh"], "devsim_nodes_si": dv["nodes_si"],
                 "femwell": {"core_res_um": mode_res, "elements": mp["elements"], "order": 2,
                             "domain_um": [5.0, 2.2], "weight_grid_step_um": mp["weight_grid"]["step_um"]},
                 "bias_grid_v": [V_MIN, V_MAX, V_STEP_QUICK if quick else V_STEP_FULL]},
        "convergence": conv,
        "validation": checks,
        "uncertainty": uncertainty_budget(conv, checks, jac, plasma_alt, summary),
        "seconds": round(time.time() - t_start, 1),
        "quick": bool(quick),
    }
    return art, side


def build_reports(xs, d, v, dneff, alpha, c, r_s, q_loss, mp, jac):
    """CAL-5 fill fractions, EO bandwidths against CAL-10 / VAL-3, and the loss budget of 07 P0-2."""
    n_g = mp["n_g"]
    hyp = ng_hypotheses(n_g, mp["n_eff"], mp["dneff_dt_per_k"])
    cal5 = {"target": {"v": list(CAL5_V), "shift_pm": list(CAL5_SHIFT_PM), "avg_pm_per_v": CAL5_AVG_PM_PER_V},
            "lambda_nm": LAMBDA0_M * 1e9, "hypotheses": {}}
    for k, h in hyp.items():
        fc = fill_for_cal5(v, dneff, h["n_g"])
        row = {"n_g": _f(h["n_g"], 5), "basis": h["basis"], "fill_endpoint": _f(fc["fill_endpoint"], 4),
               "fill_lsq": _f(fc["fill_lsq"], 4), "lsq_rms_pm": _f(fc["lsq_rms_pm"], 3),
               "feasible_fill_le_1": bool(fc["fill_endpoint"] <= 1.0)}
        if "n_g_range" in h:
            row["fill_endpoint_range"] = [_f(fill_for_cal5(v, dneff, g)["fill_endpoint"], 4) for g in h["n_g_range"]]
        cal5["hypotheses"][k] = row
    f_nom = cal5["hypotheses"]["H_solver"]["fill_endpoint"]
    cal5["nominal_fill"] = f_nom
    cal5["nominal_fill_note"] = ("H_solver row; fill scales as n_g. Without the self-heating relief correction (the "
                                 "G25 measurement power is not published); see selfheat_relief.")
    # shape: local efficiency at the ladder and the 0 V efficiency
    eff = np.gradient(dneff, v) * LAMBDA0_M / n_g * 1e12 * min(f_nom, 1.0)
    cal5["efficiency_pm_per_v_at_nominal_fill"] = {f"{vv:g}V": _f(np.interp(vv, v, eff), 4)
                                                  for vv in (0.0, 0.8, 1.0, 1.6, 2.0, 3.75)}
    # self-heating relief at the measurement power (first order; diagnostic). The relief scales with (1 - eta0):
    # bracket eta0 between "only the FCA heats" and the 07 D1 nominal 0.85.
    a_i0 = 2 * math.pi * n_g / (LAMBDA0_M * np.mean(Q_I_RANGE))
    sel = v >= 0
    relief = {}
    for eta_case in ("fca_only", "eta0_0.85"):
        for p_dbm in (0.0, 4.0):
            p_mw = 10 ** (p_dbm / 10)
            fill = min(f_nom, 1.0)
            for _ in range(30):
                fca0 = fill * float(alpha[sel][0]) / a_i0
                eta_other = 0.0 if eta_case == "fca_only" else max(0.85 - fca0, 0.0)
                rel = selfheat_relief_pm(alpha[sel], fill, a_i0, p_mw, eta_other=eta_other)
                rel_cal = np.interp(CAL5_V, v[sel], rel)
                fc = fill_for_cal5(v, dneff, n_g, relief_pm=rel_cal)
                fill = fc["fill_endpoint"]
            relief[f"{eta_case}_{p_dbm:+.0f}dBm"] = {"relief_pm_at_cal5_v": _fl(rel_cal, 3),
                                                     "fill_endpoint": _f(fill, 4), "fill_lsq": _f(fc["fill_lsq"], 4)}
    cal5["selfheat_relief"] = {"assumptions": "K_SH 405 pm/mW absorbed (03 table, range 160-1400), critical "
                                              "coupling at 0 V, Q_i 7700 at 0 V, read-out at the transmission dip; "
                                              "eta0 either the FCA share only or 0.85 (07 D1)", **relief}
    # ------------------------------------------------------- EO bandwidth
    L_fsr = C0 / (n_g * FSR_DESIGN_HZ)          # = 2 pi r_ref when r_ref is the self-consistent design radius
    L_r = 2 * math.pi * P_RING_R_UM * 1e-6
    bw = {"method": "single-pole RC (R_s + R_src) C_j x exact TCMT optical response (03 section 5.6, "
                    "Sacher & Poon 2008 pole pair), critical coupling, q from this LUT",
          "targets": {"CAL-10": [CAL10_GHZ, CAL10_SIGMA_GHZ], "VAL-3 (held out)": [VAL3_GHZ, VAL3_TOL_GHZ]},
          "cases": []}
    for vb in (1.0, 2.0):
        cp = float(np.interp(vb, v, c))
        rp = float(np.interp(vb, v, r_s))
        qv = float(np.interp(vb, v, q_loss))
        for (lname, L) in (("L_from_FSR_1.8THz", L_fsr), ("L_2piR_7.5um", L_r)):
            fill = min(f_nom, 1.0)
            cj = fill * L * cp
            rs = rp / (fill * L)
            for r_src in (0.0, 50.0):
                tau = (rs + r_src) * cj
                for ql in Q_L_RANGE:
                    for x in (1 / math.sqrt(3), 0.813):
                        for side_sign in (+1, -1):
                            f = eo_bandwidth_ghz(tau, q_loaded=ql, x_detune=side_sign * x, q_loss=qv)
                            bw["cases"].append({"v_rev": vb, "length": lname, "L_um": _f(L * 1e6, 4),
                                                "fill": _f(fill, 4), "C_j_fF": _f(cj * 1e15, 4),
                                                "R_s_ohm": _f(rs, 4), "R_src_ohm": r_src, "f_rc_ghz":
                                                _f(1 / (2 * math.pi * tau) / 1e9, 4), "Q_L": ql,
                                                "x_detune": _f(side_sign * x, 4), "q_loss": _f(qv, 3),
                                                "f3db_ghz": _f(f, 4)})
    nominal = [c_ for c_ in bw["cases"] if c_["length"] == "L_from_FSR_1.8THz" and c_["R_src_ohm"] == 50.0
               and abs(abs(c_["x_detune"]) - 0.5774) < 1e-3 and c_["x_detune"] > 0]
    bw["nominal"] = {f"{c_['v_rev']:g}V_QL{c_['Q_L']:.0f}": c_["f3db_ghz"] for c_ in nominal}
    rows1 = [c_["f3db_ghz"] for c_ in bw["cases"] if c_["v_rev"] == 1.0 and c_["f3db_ghz"] is not None]
    rows2 = [c_["f3db_ghz"] for c_ in bw["cases"] if c_["v_rev"] == 2.0 and c_["f3db_ghz"] is not None]
    bw["range_1v_ghz"] = [_f(min(rows1), 4), _f(max(rows1), 4)]
    bw["range_2v_ghz"] = [_f(min(rows2), 4), _f(max(rows2), 4)]
    tau1 = float(np.interp(1.0, v, r_s)) * float(np.interp(1.0, v, c))
    tau2 = float(np.interp(2.0, v, r_s)) * float(np.interp(2.0, v, c))
    bw["junction_rc_only"] = {"f_rc_1v_ghz": _f(1 / (2 * math.pi * tau1) / 1e9, 4),
                              "f_rc_2v_ghz": _f(1 / (2 * math.pi * tau2) / 1e9, 4),
                              "ratio_2v_over_1v": _f(tau1 / tau2, 4),
                              "note": "r_s' c' is independent of fill and length (ideal driver, no pads)"}
    bw["measured_ratio_2v_over_1v"] = {"same_device": _f(35.4 / 31.8, 4), "caption": _f(34.1 / 31.8, 4),
                                       "source": "LM-P Fig. 1(f) legend 31.8 / 35.4 GHz (03 verification log #16)"}
    # implied: which R_src + pad capacitance would give CAL-10 at the nominal optical case
    # ------------------------------------------------------- loss budget (07 P0-2)
    lb = {"q_i_range": list(Q_I_RANGE), "n_g": _f(n_g, 5),
          "alpha_i_per_m": [_f(2 * math.pi * n_g / (LAMBDA0_M * qi), 5) for qi in Q_I_RANGE],
          "alpha_i_db_per_cm": [_f(10 / math.log(10) * 2 * math.pi * n_g / (LAMBDA0_M * qi) / 100, 4)
                                for qi in Q_I_RANGE],
          "rows": []}
    fill = min(f_nom, 1.0)
    for vb in (-0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 3.75, 4.0):
        a = float(np.interp(vb, v, alpha))
        row = {"v_rev": vb, "alpha_fca_full_per_m": _f(a, 5),
               "alpha_fca_full_db_per_cm": _f(a * 10 / math.log(10) / 100, 4),
               "alpha_fca_ring_per_m": _f(fill * a, 5), "q_fca_only": _f(2 * math.pi * n_g / (LAMBDA0_M * fill * a), 5)}
        for qi in Q_I_RANGE:
            ai = 2 * math.pi * n_g / (LAMBDA0_M * qi)
            row[f"eta_fca_Qi{qi:.0f}"] = _f(fill * a / ai, 4)
        lb["rows"].append(row)
    a_scat = 0.73 * math.log(10) / 10 * 100   # 0.73 dB/cm GF rib loss (printed), as /m
    lb["eta_bounds_at_0v"] = {}
    for qi in Q_I_RANGE:
        ai = 2 * math.pi * n_g / (LAMBDA0_M * qi)
        a0 = fill * float(np.interp(0.0, v, alpha))
        lb["eta_bounds_at_0v"][f"Qi{qi:.0f}"] = {
            "eta_min_fca_only": _f(a0 / ai, 4),
            "eta_max_all_other_heats": _f(1 - a_scat / ai, 4),
            "unexplained_loss_db_per_cm": _f((ai - a0 - a_scat) * 10 / math.log(10) / 100, 4)}
    lb["note"] = ("eta = (f alpha_FCA + alpha_other_abs)/alpha_i. alpha_scatter taken as the printed GF rib loss "
                  "0.73 dB/cm (it includes everything passive); alpha_rad from the modes artefact (not included). "
                  "eta_min assumes the unexplained remainder does not heat; eta_max that it all heats.")
    lb["deta_dv_per_v_at_1v"] = {f"Qi{qi:.0f}": _f(fill * float(np.interp(1.0, v, np.gradient(alpha, v)))
                                                  / (2 * math.pi * n_g / (LAMBDA0_M * qi)), 4) for qi in Q_I_RANGE}
    return {"cal5": cal5, "eo_bandwidth": bw, "loss_budget": lb}


def uncertainty_budget(conv, checks, jac, plasma_alt, summary):
    """1-sigma relative uncertainty of the headline numbers by source (numerics, model, assumptions)."""
    def rel(entry):
        try:
            return abs(float(entry.get("rel_error_estimate") or (entry["error_estimate"] / entry["extrapolated"])))
        except Exception:
            return None
    out = {"numerical": {}, "model": {}, "assumption_jacobian": "see artefact.jacobian (per unit of each input)"}
    dm = conv.get("devsim_mesh", {})
    mm = conv.get("mode_mesh", {})
    for k in ("dneff_1v6", "alpha_fca_0v_per_m", "c_1v_f_per_m", "r_series_1v_ohm_m"):
        a = rel(dm.get(k, {})) if k in dm else None
        b = rel(mm.get(k, {})) if k in mm else None
        out["numerical"][k] = {"devsim_mesh_rel": _f(a, 3), "mode_mesh_rel": _f(b, 3)}
    out["model"]["plasma_dispersion_soref_vs_nsm_dneff"] = _f(plasma_alt["soref1987"]["ratio_1v6"] - 1, 3)
    out["model"]["plasma_dispersion_soref_vs_nsm_alpha"] = _f(plasma_alt["soref1987"]["ratio_alpha_0v"] - 1, 3)
    out["model"]["not_modelled"] = ["incomplete ionization (B ~1e18: 5-15 % fewer holes at 300 K)",
                                    "band-gap narrowing (V_bi lower by a few tens of mV)",
                                    "vertical implant profiles (box profiles assumed)",
                                    "sidewall angle and roughness", "pads, vias and C_pad"]
    return out


def _tool_versions():
    vers = {"python": platform.python_version(), "numpy": np.__version__}
    for mod in ("scipy", "devsim", "femwell", "skfem", "gmsh", "shapely", "tidy3d"):
        try:
            with _quiet(True), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mm = __import__(mod)
            vers[mod] = getattr(mm, "__version__", "?")
        except Exception:
            vers[mod] = None
    return vers


def write_artefact(art, side, out_dir):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = side["artefact"]
    p = out_dir / name
    p.write_text(json.dumps(art, indent=1, sort_keys=False, default=_json_default) + "\n")
    side = dict(side, sha256=_sha256(p))
    side["data_file"] = p.name
    sp = out_dir / (p.stem + ".sidecar.json")
    sp.write_text(json.dumps(side, indent=1, default=_json_default) + "\n")
    return p, sp


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return str(o)


# --------------------------------------------------------------------------------------------- doping sweep
def doping_sweep(xs_name="rib160", topology=DEFAULT_TOPOLOGY, levels=(5e17, 1e18, 2e18, 3e18, 5e18), log=print,
                 scale=None, mode_res=0.016, asym=((1e18, 3e18), (3e18, 1e18)), r_ref_um=None):
    """Consistency of the body doping with CAL-5 (fill <= 1), the loss budget (f alpha_FCA <= alpha_i) and the
    junction RC. Symmetric N_A = N_D levels plus two asymmetric pairs; everything else nominal."""
    base = nominal_doping(xs_name, topology)
    scale = PROD[topology]["study"] if scale is None else scale
    xs = XSECS[xs_name]
    if r_ref_um is None:
        r_ref_um, _ = resolve_r_ref(xs, base, log=log)
    xs = dataclasses.replace(xs, r_ref_um=r_ref_um)
    mp = _isolated(mode_properties, xs, base, res_um=mode_res, radius_um=xs.r_ref_um, with_dn_check=False)
    wf = weight_field(mp, xs, base)
    vj = np.array([0.0, 0.8, 1.0, 1.2, 1.6, 2.0, 3.75])
    ai = [2 * math.pi * mp["n_g"] / (LAMBDA0_M * qi) for qi in Q_I_RANGE]
    rows = []
    pairs = [(N, N) for N in levels] + list(asym)
    for na, nd in pairs:
        d = dataclasses.replace(base, na_cm3=na, nd_cm3=nd)
        rr = _devsim_and_overlap(xs, d, vj, wf, scale)
        rc = contact_resistance_ohm_m(d, T_REF_K, xs)
        fc = fill_for_cal5(vj, rr["dneff"], mp["n_g"])
        f = fc["fill_endpoint"]
        a0 = float(rr["alpha"][0])
        r1 = float(np.interp(1.0, rr["v_ac"], rr["r_ac"])) + rc
        c1 = float(np.interp(1.0, rr["v_ac"], rr["c_ac"]))
        fe = min(f, 1.0)
        row = {"na_cm3": na, "nd_cm3": nd, "fill_endpoint": _f(f, 4), "lsq_rms_pm": _f(fc["lsq_rms_pm"], 3),
               "eff_avg_pm_per_v_fill1": _f(float(shift_pm(np.interp(1.6, vj, rr["dneff"]), mp["n_g"])) / 1.6, 4),
               "alpha_fca_0v_db_per_cm_full": _f(a0 * 10 / math.log(10) / 100, 4),
               "alpha_fca_0v_db_per_cm_at_fill": _f(fe * a0 * 10 / math.log(10) / 100, 4),
               "eta_fca_at_fill_Qi7000_8400": [_f(fe * a0 / a, 4) for a in ai],
               "c_1v_f_per_m": _f(c1, 4), "r_series_1v_ohm_m": _f(r1, 4),
               "f_rc_1v_ghz": _f(1 / (2 * math.pi * r1 * c1) / 1e9, 4),
               "c_j_1v_ff_at_fill_L44um": _f(fe * 44e-6 * c1 * 1e15, 4)}
        row["consistent"] = bool(f <= 1.0 and fe * a0 <= ai[0])
        rows.append(row)
        log(f"  NA {na:.1e} ND {nd:.1e}: fill {f:.3f}, {row['eff_avg_pm_per_v_fill1']} pm/V at fill 1, alpha0 "
            f"{row['alpha_fca_0v_db_per_cm_full']} dB/cm, eta(fill) {row['eta_fca_at_fill_Qi7000_8400']}, "
            f"f_RC {row['f_rc_1v_ghz']} GHz")
    return {"xsec": xs_name, "topology": topology, "r_ref_um": xs.r_ref_um, "n_g": _f(mp["n_g"], 5), "rows": rows,
            "alpha_i_per_m": [_f(a, 5) for a in ai],
            "note": "consistent = fill <= 1 (CAL-5 reachable with this junction) and f alpha_FCA(0 V) <= alpha_i "
                    "at Q_i = 7000; DEVSIM scale %.4g, mode res %.4g um" % (scale, mode_res)}


def recheck_modes(out_dir=OUT_DIR, log=print):
    """Post-processing (no solver): compare every junction artefact with the modes artefact of its cross-section
    (r_ref, n_eff, n_g, Si index weight) and store the comparison in the junction sidecar (the data file and its
    sha256 are unchanged)."""
    out_dir = pathlib.Path(out_dir)
    for p in sorted(out_dir.glob("junction_*.json")):
        if p.name.endswith(".sidecar.json"):
            continue
        sc = p.with_name(p.stem + ".sidecar.json")
        art = json.loads(p.read_text())
        side = json.loads(sc.read_text())
        xs_name = art["xsec"]["name"]
        mp_ = out_dir / f"modes_{xs_name}.json"
        ms_ = out_dir / f"modes_{xs_name}.sidecar.json"
        if not (mp_.exists() and ms_.exists()):
            log(f"{p.name}: no modes artefact for {xs_name}")
            continue
        ma, msd = json.loads(mp_.read_text()), json.loads(ms_.read_text())
        if msd.get("sha256") != _sha256(mp_):
            log(f"{p.name}: modes_{xs_name}.json fails its sha256; skipped")
            continue
        ov = art["overlap"]
        comp = {"modes_sha256": msd["sha256"], "modes_r_ref_um": ma.get("r_ref_um"), "junction_r_ref_um": ov["r_ref_um"],
                "modes_n_eff": ma.get("n_eff"), "junction_n_eff_bent": ov["n_eff_bent"],
                "modes_n_g": ma.get("n_g"), "junction_n_g_bent": ov["n_g_bent"],
                "modes_index_weight_si": (ma.get("index_weights") or {}).get("si"),
                "junction_index_weight_si": ov["dneff_dnsi_bent"],
                "note": "modes: exact anisotropic bend transform on its own mesh; junction: femwell isotropic conformal "
                        "map; same geometry and materials (post-processed by recheck_modes, no solver re-run)"}
        try:
            comp["rel_diff_n_g"] = ov["n_g_bent"] / ma["n_g"] - 1
            comp["rel_diff_index_weight_si"] = ov["dneff_dnsi_bent"] / ma["index_weights"]["si"] - 1
            comp["delta_r_ref_um"] = ov["r_ref_um"] - ma["r_ref_um"]
        except Exception:
            pass
        side.setdefault("validation", {})["modes_artefact_consistency"] = comp
        side["sha256"] = _sha256(p)
        sc.write_text(json.dumps(side, indent=1, default=_json_default) + "\n")
        log(f"{p.name}: n_g {comp.get('rel_diff_n_g')}, index weight {comp.get('rel_diff_index_weight_si')}, "
            f"r_ref {comp.get('delta_r_ref_um')}")


# ---------------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mrm.extraction.junction", description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="coarse meshes and bias grid (tests)")
    ap.add_argument("--xsec", nargs="+", default=list(XSECS), choices=list(XSECS))
    ap.add_argument("--out", default=None, help="output directory (default mrm/data/extracted; --quick: a temp dir)")
    ap.add_argument("--topology", nargs="+", default=list(TOPOLOGIES), choices=list(TOPOLOGIES),
                    help="junction topologies (the default topology gets junction_<xsec>.json, others a suffix)")
    ap.add_argument("--no-crosschecks", action="store_true")
    ap.add_argument("--doping-sweep", action="store_true", help="also run the doping consistency sweep")
    ap.add_argument("--doping-sweep-only", action="store_true", help="only print the doping consistency sweep")
    ap.add_argument("--recheck-modes", action="store_true",
                    help="no solver: compare the junction artefacts with the modes artefacts and update the sidecars")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out) if a.out else (pathlib.Path(tempfile.gettempdir()) / "mrm_junction_quick"
                                              if a.quick else OUT_DIR)
    log = lambda s: print(s, flush=True)  # noqa: E731
    if a.recheck_modes:
        recheck_modes(out, log=log)
        return 0
    if a.doping_sweep_only:
        for name in a.xsec:
            for topo in a.topology:
                log(json.dumps(doping_sweep(name, topo, log=log), indent=1, default=_json_default))
        return 0
    status = 0
    for name in a.xsec:
        for topo in a.topology:
            light = topo != DEFAULT_TOPOLOGY
            try:
                art, side = extract(name, topology=topo, quick=a.quick, log=log,
                                    do_crosschecks=not a.no_crosschecks, light=light)
                if a.doping_sweep and not a.quick:
                    log(f"[{name}/{topo}] doping sweep")
                    art["reports"]["doping_sweep"] = doping_sweep(name, topo, log=log,
                                                                  r_ref_um=art["overlap"]["r_ref_um"])
                p, s = write_artefact(art, side, out)
                log(f"wrote {p} and {s.name} ({side['seconds']} s)")
            except Exception:            # keep going with the other artefacts; report at the end
                import traceback
                log(f"[{name}/{topo}] FAILED\n" + traceback.format_exc())
                status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
