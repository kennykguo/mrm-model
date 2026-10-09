"""Material models for an O-band (1260-1360 nm) SOI microring modulator plant, 273-400 K.

Every coefficient is read from ``mrm/data/materials.json`` (schema ``materials/1``), a cleaned copy of
``docs/research/02_materials.json`` with its verification-log corrections applied. Every function names its source and
its validity range; outside the range the result is an extrapolation and a :class:`MaterialRangeWarning` is issued.

Units (the argument and function names carry them):

- wavelength ``lam_m``: vacuum wavelength in m (the printed fits use um; conversion happens here);
- temperature ``t_k``: K;
- carrier and doping densities ``*_per_cm3``: cm^-3, as printed in every source;
- results are SI unless the name says otherwise (``_per_cm``, ``_cm2``, ``_ev``, ...).

All functions are vectorized: array arguments broadcast with NumPy rules and the result is a float or an ndarray.

Silicon index (07 P1-6): ONE model is used everywhere, the Li 1980 temperature shape shifted to the Frey 2006 /
Franta 2017 absolute (vacuum) level, ``n_si(lam_m, t_k)``. Its dn/dT is Li's (the offset does not depend on T) and its
group index ``n_g_si`` comes from the same function (3.690 at 1310 nm, 300 K). ``level="li"`` is the alternative
hypothesis; never mix the two levels inside one ring calculation.

Provenance words follow 02_materials.md: printed / read-off / inferred / derived / assumed.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import pathlib
import warnings

import numpy as np

from mrm.units import C0, H_PLANCK, K_B, Q_E

__all__ = [
    "MaterialRangeWarning", "MODEL_ID", "LAM_REF_M", "data", "data_sha256", "materials_provenance",
    # silicon optics
    "n_si", "n_si_li1980", "n_si_frey2006", "si_index_offset_frey", "dn_dt_si_per_k", "d2n_dt2_si_per_k2",
    "dn_dt_si_sigma_per_k", "si_dn_dt_slope_per_k", "dn_dlam_si_per_m", "n_g_si", "dn_dt_si_cocorullo1523_per_k",
    "alpha_band_si_per_m",
    # oxide and nitride optics
    "n_sio2", "n_sio2_malitson", "dn_dt_sio2_per_k", "dn_dlam_sio2_per_m", "n_g_sio2", "n_si3n4_luke",
    "dn_dlam_si3n4_per_m", "n_g_si3n4", "DN_DT_SI3N4_PER_K",
    # expansion
    "alpha_l_si_per_k", "alpha_l_sio2_per_k", "expansion_strain_si", "dlambda_dt_ceiling_pm_per_k",
    # thermal
    "k_si_bulk_w_per_m_k", "k_si_film_w_per_m_k", "k_sio2_w_per_m_k", "k_low_k_w_per_m_k", "k_metal_w_per_m_k",
    "cp_si_j_per_kg_k", "cp_sio2_j_per_kg_k", "rho_c_j_per_m3_k", "thermal_properties", "THERMAL_MATERIALS",
    "thermal_boundary_resistance_m2_k_per_w", "RHO_SI_KG_PER_M3", "RHO_SIO2_KG_PER_M3",
    # free carriers and nonlinear optics
    "plasma_dispersion", "plasma_dispersion_change", "plasma_dispersion_derivatives", "plasma_temperature_factor",
    "PLASMA_MODELS", "tpa_beta_m_per_w", "kerr_n2_m2_per_w",
    # junction physics
    "eg_si_ev", "ni_si_per_cm3", "vbi_v", "dvbi_dt_v_per_k",
    # heaters
    "mobility_arora_cm2_per_v_s", "mobility_masetti_cm2_per_v_s", "resistivity_doped_si_ohm_m", "tcr_doped_si_per_k",
    "heater_tcr_per_k", "heater_resistance_ratio", "HEATER_FAMILIES", "HEATER_ROM_FAMILY_MAP",
    # germanium photodiode
    "ge_pd_responsivity_limit_a_per_w", "ge_pd_responsivity_a_per_w", "ge_pd_dark_current_a", "ge_pd_dark_ea_ev",
    "GE_PD_DARK_I0_A_RANGE",
    # docs and uncertainty
    "material_table", "material_table_markdown", "material_uncertainties",
]

MODEL_ID = "materials/1"
LAM_REF_M = 1.31e-6                       # O-band design wavelength used by the tables
_DATA_PATH = pathlib.Path(__file__).resolve().parent / "data" / "materials.json"

K_B_EV = K_B / Q_E                        # Boltzmann constant, eV/K
HC_EV_UM = H_PLANCK * C0 / Q_E * 1e6      # h c / q, eV um (1.239841984)


class MaterialRangeWarning(UserWarning):
    """A material model was evaluated outside the range its source supports (the result is an extrapolation)."""


# ------------------------------------------------------------------------------------------------- data access
@functools.lru_cache(maxsize=1)
def data() -> dict:
    """The parsed ``mrm/data/materials.json`` (schema ``materials/1``). Treat it as read-only."""
    with open(_DATA_PATH, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    if d.get("_meta", {}).get("schema") != MODEL_ID:
        raise ValueError(f"{_DATA_PATH} is not schema {MODEL_ID}")
    return d


def _c(key: str) -> dict:
    return data()[key]["coefficients"]


@functools.lru_cache(maxsize=1)
def data_sha256() -> str:
    """sha256 of ``mrm/data/materials.json``; extraction sidecars record it as the materials-model input."""
    return hashlib.sha256(_DATA_PATH.read_bytes()).hexdigest()


def materials_provenance(si_level: str = "frey") -> dict:
    """The ``materials`` block an artefact sidecar should embed (ARCHITECTURE section 5 'inputs')."""
    return {"schema": MODEL_ID, "file": "mrm/data/materials.json", "sha256": data_sha256(),
            "si_index_model": f"li1980 temperature shape on the {si_level} level" if si_level == "frey"
            else "li1980 (alternative level)", "sio2_index_model": "malitson1965 x n_air + Leviton/Frey dn/dT"}


def _arr(x):
    return np.asarray(x, dtype=float)


def _out(x):
    """Return a Python float for scalar results and an ndarray otherwise."""
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


def _check_range(name: str, x, lo: float | None, hi: float | None, unit: str = "") -> None:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return
    bad_lo = lo is not None and np.nanmin(x) < lo
    bad_hi = hi is not None and np.nanmax(x) > hi
    if bad_lo or bad_hi:
        warnings.warn(f"{name}: evaluated at [{np.nanmin(x):.4g}, {np.nanmax(x):.4g}] {unit}, outside the supported "
                      f"range [{lo}, {hi}] {unit}; the result is an extrapolation",
                      MaterialRangeWarning, stacklevel=3)


def _corner(spec: dict, corner: str) -> float:
    """spec = {"nominal": x, "range": [lo, hi], ...}; corner in nominal | low | high | stress (if present)."""
    if corner == "nominal":
        return float(spec["nominal"])
    if corner == "low":
        return float(spec["range"][0])
    if corner == "high":
        return float(spec["range"][1])
    if corner in spec:
        return float(spec[corner])
    raise ValueError(f"corner must be nominal, low or high (got {corner!r})")


def _lam_um(lam_m, name="lambda", lo=None, hi=None):
    lam = _arr(lam_m) * 1e6
    if np.any(lam <= 0.0):
        raise ValueError(f"{name}: wavelength must be positive (lam_m in metres)")
    if np.nanmax(lam) > 100.0:
        raise ValueError(f"{name}: lam_m={np.nanmax(lam) / 1e6:g} looks like nm or um; pass metres")
    _check_range(name, lam, lo, hi, "um")
    return lam


# ============================================================================================ silicon index
def _li_parts(lam_um, t_k):
    """Li 1980 eq. (22) and its analytic T and lambda derivatives.

    u = n^2 = eps(T) + L(T) A(T) / lam^2, L = exp(-3 dLL(T)). Returns n, dn/dT, d2n/dT2, dn/dlam (per um)."""
    c = _c("si_index_li1980")
    e0, e1, e2, e3 = c["e"]
    a0, a1, a2 = c["A"]
    d0, d1, d2, d3 = c["dLL"]
    T = t_k
    eps = e0 + e1 * T + e2 * T**2 + e3 * T**3
    eps_t = e1 + 2 * e2 * T + 3 * e3 * T**2
    eps_tt = 2 * e2 + 6 * e3 * T
    A = a0 + a1 * T + a2 * T**2
    A_t = a1 + 2 * a2 * T
    A_tt = 2 * a2
    D = d0 + d1 * T + d2 * T**2 + d3 * T**3
    D_t = d1 + 2 * d2 * T + 3 * d3 * T**2
    D_tt = 2 * d2 + 6 * d3 * T
    L = np.exp(-3.0 * D)
    L_t = -3.0 * D_t * L
    L_tt = (9.0 * D_t**2 - 3.0 * D_tt) * L
    il2 = 1.0 / lam_um**2
    u = eps + L * A * il2
    u_t = eps_t + (L_t * A + L * A_t) * il2
    u_tt = eps_tt + (L_tt * A + 2 * L_t * A_t + L * A_tt) * il2
    n = np.sqrt(u)
    n_t = u_t / (2 * n)
    n_tt = (u_tt - 2 * n_t**2) / (2 * n)
    n_l = -L * A / (lam_um**3 * n)
    return n, n_t, n_tt, n_l


def _frey_parts(lam_um, t_k):
    """Frey 2006 Table 5 Sellmeier, n and dn/dlam (per um)."""
    c = _c("si_index_frey2006")
    l2 = lam_um**2
    u = 1.0
    du = 0.0
    for sk, lk in (("S1", "lam1"), ("S2", "lam2"), ("S3", "lam3")):
        S = sum(ci * t_k**j for j, ci in enumerate(c[sk]))
        l0 = sum(ci * t_k**j for j, ci in enumerate(c[lk]))
        den = l2 - l0**2
        u = u + S * l2 / den
        du = du + S * (-2.0 * lam_um * l0**2) / den**2
    n = np.sqrt(u)
    return n, du / (2 * n)


def n_si_li1980(lam_m, t_k):
    """Si index, H. H. Li, J. Phys. Chem. Ref. Data 9, 561 (1980), eq. (22) (printed formula).

    Valid 1.2-14 um, 100-750 K; +/-2e-3 absolute, dn/dT +/-0.15e-4 /K. The printed dL/L293 constant '-0.071' is
    read as -0.071 % = -7.1e-4 (02 log #2). The 293-1600 K branch is used throughout; it reproduces Li Table 1 at
    250 K to 1e-4. Li's level appears air-referenced; prefer :func:`n_si` (Frey level)."""
    lam = _lam_um(lam_m, "n_si_li1980", 1.2, 14.0)
    T = _arr(t_k)
    _check_range("n_si_li1980", T, 100.0, 750.0, "K")
    return _out(_li_parts(lam, T)[0])


def n_si_frey2006(lam_m, t_k):
    """Si index, Frey, Leviton, Madison, Proc. SPIE 6273, 62732J (2006), Table 5 (printed). Absolute (vacuum).

    Valid 1.1-5.6 um and 20-300 K ONLY: the T polynomials diverge near 320 K, so this raises above 300.5 K.
    +/-1e-4 absolute. Reproduces Table 2: 3.50616 at 1.3 um, 295 K."""
    lam = _lam_um(lam_m, "n_si_frey2006", 1.1, 5.6)
    T = _arr(t_k)
    if np.any(T > 300.5) or np.any(T < 20.0):
        raise ValueError("Frey 2006 Sellmeier is valid only for 20-300 K (it diverges near 320 K)")
    return _out(_frey_parts(lam, T)[0])


def si_index_offset_frey(lam_m):
    """Absolute-level offset n_Frey(lam, 295 K) - n_Li(lam, 295 K) (derived): +4.62e-3 at 1260 nm, +4.085e-3 at
    1310 nm, +3.67e-3 at 1360 nm. About 0.9e-3 of it is the air/vacuum convention (02 log #5)."""
    lam = _lam_um(lam_m, "si_index_offset_frey", 1.1, 5.6)
    t = _c("si_index_recommended")["t_level_k"]
    return _out(_frey_parts(lam, t)[0] - _li_parts(lam, t)[0])


def n_si(lam_m, t_k, level: str = "frey"):
    """Recommended Si index: Li 1980 temperature shape on the Frey 2006 / Franta 2017 absolute level (07 P1-6).

    n(lam, T) = n_Li(lam, T) + [n_Frey(lam, 295 K) - n_Li(lam, 295 K)] (derived). 3.50570 at 1310 nm, 300 K;
    +/-5e-4 absolute; Franta 2017 gives 3.50555. Design range 1.26-1.36 um, 250-400 K (Li's T law holds to 750 K).
    ``level="li"`` returns Li's own level (4.1e-3 lower at 1310 nm), the alternative hypothesis."""
    if level not in ("frey", "li"):
        raise ValueError("level must be 'frey' (recommended) or 'li'")
    lam = _lam_um(lam_m, "n_si", 1.2, 5.6)
    T = _arr(t_k)
    _check_range("n_si", T, 100.0, 750.0, "K")
    n = _li_parts(lam, T)[0]
    if level == "frey":
        t = _c("si_index_recommended")["t_level_k"]
        n = n + _frey_parts(lam, t)[0] - _li_parts(lam, t)[0]
    return _out(n)


def dn_dt_si_per_k(lam_m, t_k):
    """Thermo-optic coefficient of Si, 1/K: analytic T-derivative of Li 1980 (derived). Identical for both index levels.

    1310 nm: 1.861e-4 (273.15 K), 1.938e-4 (300 K), 2.126e-4 (375 K), 2.179e-4 (400 K): +17 % over 0-125 C.
    Recommended uncertainty +/-0.07e-4 at 300 K, +/-0.10e-4 at 375 K (:func:`dn_dt_si_sigma_per_k`). Frey 2006
    Table 4 prints 1.94e-4 at 1.3 um, 295 K (printed). Low corner Franta 2017 (1.87e-4 / 2.00e-4), high corner Lukes
    (2.04e-4 at 300 K). Valid 1.2-14 um, 100-750 K."""
    lam = _lam_um(lam_m, "dn_dt_si_per_k", 1.2, 14.0)
    T = _arr(t_k)
    _check_range("dn_dt_si_per_k", T, 100.0, 750.0, "K")
    return _out(_li_parts(lam, T)[1])


def d2n_dt2_si_per_k2(lam_m, t_k):
    """Second T-derivative of the Si index, 1/K^2 (analytic, Li 1980; derived). 2.78e-7 at 300 K and 2.22e-7 at
    375 K (1310 nm). Sources span 1.7-2.8e-7 (Franta 1.88e-7, Cocorullo 2.58e-7 at 300 K)."""
    lam = _lam_um(lam_m, "d2n_dt2_si_per_k2", 1.2, 14.0)
    T = _arr(t_k)
    _check_range("d2n_dt2_si_per_k2", T, 100.0, 750.0, "K")
    return _out(_li_parts(lam, T)[2])


def dn_dt_si_sigma_per_k(t_k):
    """1-sigma-like uncertainty of Si dn/dT at 1310 nm, 1/K (derived from the source spread, 02 §2.3): 0.07e-4 at
    300 K rising linearly to 0.10e-4 at 375 K (held at 0.07e-4 below 300 K)."""
    c = _c("si_dn_dt")
    T = _arr(t_k)
    s = c["sigma_300K"] + (c["sigma_375K"] - c["sigma_300K"]) / 75.0 * np.maximum(T - 300.0, 0.0)
    return _out(s)


def si_dn_dt_slope_per_k(lam_m, t_k):
    """Normalized slope of Si dn/dT, beta = (d2n/dT2)/(dn/dT), 1/K (derived, Li 1980): 1.434e-3 at 1310 nm, 300 K.

    This is the material part of the gain-schedule shape dlambda/dT(T) = s0 [1 + beta (T - T0)] (07 R1). Range across
    sources 1.0e-3 (Franta 2017) to 1.45e-3 (Li); Souza 2018 gives 1.33e-3 (``data()['si_dn_dt_slope']``)."""
    return _out(_arr(d2n_dt2_si_per_k2(lam_m, t_k)) / _arr(dn_dt_si_per_k(lam_m, t_k)))


def dn_dlam_si_per_m(lam_m, t_k, level: str = "frey"):
    """Material dispersion dn/dlam of Si, 1/m (analytic derivative of the same index model as :func:`n_si`).

    Recommended level: -0.1407 /um at 1310 nm, 300 K (Franta about -0.141 /um; Li -0.131 /um). At 1.3 um, 295 K it
    gives -0.1439 /um (n_g 3.693); Frey's Table 3 prints -0.148 /um there (n_g 3.699), inside the +/-0.02 n_g band."""
    if level not in ("frey", "li"):
        raise ValueError("level must be 'frey' or 'li'")
    lam = _lam_um(lam_m, "dn_dlam_si_per_m", 1.2, 5.6)
    T = _arr(t_k)
    _check_range("dn_dlam_si_per_m", T, 100.0, 750.0, "K")
    d = _li_parts(lam, T)[3]
    if level == "frey":
        t = _c("si_index_recommended")["t_level_k"]
        d = d + _frey_parts(lam, t)[1] - _li_parts(lam, t)[3]
    return _out(d * 1e6)


def n_g_si(lam_m, t_k, level: str = "frey"):
    """Material group index of Si, n_g = n - lam dn/dlam (derived, from the same model as :func:`n_si`).

    Recommended: 3.690 at 1310 nm, 300 K (Frey Sellmeier 3.689 at 295 K; Franta 3.690). ``level="li"``: 3.674.
    Uncertainty +/-0.02 (02 log #6)."""
    lam = _arr(lam_m)
    return _out(_arr(n_si(lam_m, t_k, level)) - lam * _arr(dn_dlam_si_per_m(lam_m, t_k, level)))


def dn_dt_si_cocorullo1523_per_k(t_k):
    """Si dn/dT at 1523 nm, Cocorullo, Della Corte, Rendina, APL 74, 3338 (1999) (printed formula): 1.855e-4 at
    300 K. Valid 300-550 K, 1523 nm only. Cross-check; the textbook's 1.86e-4 is this value."""
    a = _c("si_dn_dt_cocorullo1523")["a"]
    T = _arr(t_k)
    _check_range("dn_dt_si_cocorullo1523_per_k", T, 300.0, 550.0, "K")
    return _out(a[0] + a[1] * T + a[2] * T**2)


def alpha_band_si_per_m(lam_m):
    """Band-to-band absorption of Si at 300 K, 1/m (Green 2008 k, printed; alpha = 4 pi k / lam, inferred).

    Log-linear interpolation between the two printed points: 2.7e-3 /m (2.7e-5 /cm) at 1310 nm and 3.6e-2 /m at
    1260 nm. Negligible against the ring loss; the ring's absorption comes from free carriers and defects."""
    c = _c("si_absorption_band")
    lam = _lam_um(lam_m, "alpha_band_si_per_m", 1.26, 1.31)
    a1 = 4 * math.pi * c["k_1260nm"] / 1.26e-6
    a2 = 4 * math.pi * c["k_1310nm"] / 1.31e-6
    f = (lam - 1.26) / 0.05
    return _out(np.exp(np.log(a1) + f * (np.log(a2) - np.log(a1))))


# ======================================================================================= oxide and nitride
def n_sio2_malitson(lam_m):
    """Fused silica index, Malitson, JOSA 55, 1205 (1965) (printed), air-referenced, 20 C, 0.21-3.71 um.

    1.44680 at 1310 nm; material n_g 1.4616."""
    lam = _lam_um(lam_m, "n_sio2_malitson", 0.21, 3.71)
    c = _c("sio2_index_malitson")
    l2 = lam**2
    u = 1.0 + sum(b * l2 / (l2 - cc**2) for b, cc in zip(c["B"], c["C"]))
    return _out(np.sqrt(u))


def _malitson_dn_dlam_um(lam):
    c = _c("sio2_index_malitson")
    l2 = lam**2
    u = 1.0 + sum(b * l2 / (l2 - cc**2) for b, cc in zip(c["B"], c["C"]))
    du = sum(b * (-2.0 * lam * cc**2) / (l2 - cc**2) ** 2 for b, cc in zip(c["B"], c["C"]))
    return du / (2 * np.sqrt(u))


_SIO2_KINDS = {"thermal": "thermal", "box": "thermal", "fused": "fused", "pecvd": "pecvd", "beol": "pecvd"}


def _sio2_kind(kind):
    try:
        return _SIO2_KINDS[kind]
    except KeyError:
        raise ValueError(f"SiO2 kind must be one of {sorted(_SIO2_KINDS)} (got {kind!r})") from None


def dn_dt_sio2_per_k(t_k, kind: str = "thermal"):
    """Thermo-optic coefficient of SiO2, 1/K: base + 2.4e-8 (T - 295 K).

    thermal/fused/box: base 8.45e-6 (Leviton & Frey, Corning 7980, 1.2-1.5 um, printed); pecvd/beol: 0.95e-5
    (Arbabi & Goddard 2013, Elshaari 2016, printed). Slope from the 275-300 K rows (inferred); measured 30-310 K, the
    linear extension above 300 K is assumed. +/-15 %; worth < 1 pm/K on the ring."""
    k = _sio2_kind(kind)
    c = _c("sio2_dn_dt")
    T = _arr(t_k)
    _check_range("dn_dt_sio2_per_k", T, 30.0, 400.0, "K")
    base = c["base_295K_pecvd"] if k == "pecvd" else c["base_295K_thermal"]
    return _out(base + c["slope_per_k2"] * (T - c["t0_k"]))


def n_sio2(lam_m, t_k, kind: str = "thermal"):
    """SiO2 index on the absolute (vacuum) level, matching :func:`n_si`.

    n(lam, T) = n_Malitson(lam) * n_air + [film offset] + integral_{295 K}^{T} dn/dT dT, with n_air = 1.00027
    (Malitson x n_air reproduces the absolute CHARMS index to 1e-5 at 295 K, derived). kinds: ``fused``; ``thermal``
    (= ``box``) at the fused-silica level, 1.4472 at 1310 nm (02 recommends 1.447, +/-3e-3); ``pecvd`` (= ``beol``)
    shifted to 1.450 at 1310 nm, 295 K (+/-0.01, assumed spread). Film dispersion is taken as fused silica's."""
    k = _sio2_kind(kind)
    lam = _lam_um(lam_m, "n_sio2", 0.21, 3.71)
    T = _arr(t_k)
    _check_range("n_sio2", T, 30.0, 400.0, "K")
    cm = _c("sio2_index_malitson")
    cd = _c("sio2_dn_dt")
    n = _arr(n_sio2_malitson(lam * 1e-6)) * cm["n_air"]
    if k == "pecvd":
        ref = float(n_sio2_malitson(LAM_REF_M)) * cm["n_air"]
        n = n + (_c("sio2_index_films")["pecvd_level_1310nm"] - ref)
    base = cd["base_295K_pecvd"] if k == "pecvd" else cd["base_295K_thermal"]
    dT = T - cd["t0_k"]
    return _out(n + base * dT + 0.5 * cd["slope_per_k2"] * dT**2)


def dn_dlam_sio2_per_m(lam_m):
    """Material dispersion of SiO2, 1/m (analytic derivative of Malitson x n_air; all kinds share it)."""
    lam = _lam_um(lam_m, "dn_dlam_sio2_per_m", 0.21, 3.71)
    return _out(_malitson_dn_dlam_um(lam) * _c("sio2_index_malitson")["n_air"] * 1e6)


def n_g_sio2(lam_m, t_k, kind: str = "thermal"):
    """Material group index of SiO2 (derived): 1.4621 for thermal oxide at 1310 nm, 300 K (Malitson alone: 1.4616)."""
    return _out(_arr(n_sio2(lam_m, t_k, kind)) - _arr(lam_m) * _arr(dn_dlam_sio2_per_m(lam_m)))


def n_si3n4_luke(lam_m):
    """LPCVD Si3N4 index, Luke et al., Opt. Lett. 40, 4823 (2015) (printed), 0.31-5.5 um, room temperature, +/-3e-3.

    2.0031 at 1310 nm. Only needed if a nitride layer sits in the mode."""
    lam = _lam_um(lam_m, "n_si3n4_luke", 0.31, 5.5)
    c = _c("si3n4_index_luke")
    l2 = lam**2
    return _out(np.sqrt(1.0 + sum(b * l2 / (l2 - cc**2) for b, cc in zip(c["B"], c["C"]))))


def dn_dlam_si3n4_per_m(lam_m):
    """Material dispersion of LPCVD Si3N4 (Luke 2015), 1/m (analytic)."""
    lam = _lam_um(lam_m, "dn_dlam_si3n4_per_m", 0.31, 5.5)
    c = _c("si3n4_index_luke")
    l2 = lam**2
    u = 1.0 + sum(b * l2 / (l2 - cc**2) for b, cc in zip(c["B"], c["C"]))
    du = sum(b * (-2.0 * lam * cc**2) / (l2 - cc**2) ** 2 for b, cc in zip(c["B"], c["C"]))
    return _out(du / (2 * np.sqrt(u)) * 1e6)


def n_g_si3n4(lam_m):
    """Material group index of LPCVD Si3N4 (derived): 2.042 at 1310 nm."""
    return _out(_arr(n_si3n4_luke(lam_m)) - _arr(lam_m) * _arr(dn_dlam_si3n4_per_m(lam_m)))


DN_DT_SI3N4_PER_K = float(data()["si3n4_dn_dt"]["coefficients"]["value"])
"""Si3N4 dn/dT, 2.45e-5 /K (+/-0.09e-5), Arbabi & Goddard 2013, 1550 nm, room temperature (printed)."""


# ================================================================================================ expansion
def alpha_l_si_per_k(t_k):
    """Linear CTE of Si, 1/K: Okada & Tokumaru, J. Appl. Phys. 56, 314 (1984) (printed), 120-1500 K, +/-2e-7 /K.

    alpha = [3.725 (1 - exp(-5.88e-3 (T - 124))) + 5.548e-4 T] 1e-6: 2.327e-6 (273 K), 2.568e-6 (300 K), 3.082e-6
    (375 K), 3.212e-6 (400 K). The thick Si substrate clamps the in-plane strain of the whole stack, so the ring
    length follows this alpha. Cross-check: Souza 2018 2.5e-6 at 300 K, net +27 %/100 K."""
    c = _c("si_alpha_l")
    T = _arr(t_k)
    _check_range("alpha_l_si_per_k", T, 120.0, 1500.0, "K")
    return _out((c["a"] * (1.0 - np.exp(-c["b"] * (T - c["t0_k"]))) + c["c"] * T) * 1e-6)


def expansion_strain_si(t_k, t_ref_k):
    """Thermal strain of the substrate-clamped ring, integral_{t_ref}^{T} alpha_Si dT (analytic, Okada & Tokumaru).

    The ring length is L(T) = L(T_ref) * exp(strain) (about 1 + strain)."""
    c = _c("si_alpha_l")
    T, T0 = _arr(t_k), _arr(t_ref_k)
    _check_range("expansion_strain_si", np.concatenate([np.ravel(T), np.ravel(T0)]), 120.0, 1500.0, "K")

    def prim(x):
        return (c["a"] * (x + np.exp(-c["b"] * (x - c["t0_k"])) / c["b"]) + 0.5 * c["c"] * x**2) * 1e-6

    return _out(prim(T) - prim(T0))


def alpha_l_sio2_per_k(t_k):
    """Linear CTE of SiO2 (fused silica; thermal oxide assumed similar), 1/K.

    alpha = 0.52e-6 + 6.25e-10 (T - 293.15 K): the instantaneous CTE that reproduces both printed Corning 7980 means,
    0.52 ppm/K over 5-35 C and 0.57 ppm/K over 0-200 C (inferred). Valid 273-473 K, +/-0.05e-6. PECVD oxide is process
    dependent."""
    c = _c("sio2_alpha_l")
    T = _arr(t_k)
    _check_range("alpha_l_sio2_per_k", T, *_valid_range("sio2_alpha_l", "t_k"), unit="K")
    return _out(c["a293"] + c["slope_per_k2"] * (T - c["t0_k"]))


def dlambda_dt_ceiling_pm_per_k(lam_m, t_k, neff_over_ng=0.6, level: str = "frey"):
    """Upper bound on dlambda/dT of any Si-core, SiO2-clad ring, pm/K (derived, 02 §3).

    ceiling = lam (dn_Si/dT) / n_g,Si + lam (n_eff/n_g) alpha_Si: all electric energy in Si (exact first-order
    perturbation bound with material dispersion) plus the substrate-clamped expansion. Returns (total, thermo_optic,
    expansion). n_g,Si comes from the same index model (``level``); with level="li" and n_eff/n_g = 0.6 it reproduces
    the 02 table (71.1 pm/K at 300 K, 79.7 at 400 K). ``neff_over_ng="all_si"`` uses n_Si/n_g,Si (the strict
    all-silicon limit: 72.3 / 79.2 / 81.1 pm/K at 300 / 375 / 400 K with level="li")."""
    lam = _arr(lam_m)
    ng = _arr(n_g_si(lam_m, t_k, level))
    if isinstance(neff_over_ng, str):
        if neff_over_ng != "all_si":
            raise ValueError("neff_over_ng must be a number or 'all_si'")
        neff_over_ng = _arr(n_si(lam_m, t_k, level)) / ng
    to = lam * _arr(dn_dt_si_per_k(lam_m, t_k)) / ng * 1e12
    ex = lam * _arr(neff_over_ng) * _arr(alpha_l_si_per_k(t_k)) * 1e12
    return _out(to + ex), _out(to), _out(ex)


# ======================================================================================= thermal transport
def k_si_bulk_w_per_m_k(t_k):
    """Bulk Si thermal conductivity, W/(m K): 148 (T/300)^-1.3 (Palankovski 2000, printed), ~250-500 K, +/-5 %.

    167.2 (273 K), 148.0 (300 K), 110.7 (375 K), 101.8 (400 K); CRC/TPRC recalled set 168/148/98.9 (within ~3 %)."""
    c = _c("k_si_bulk")
    T = _arr(t_k)
    _check_range("k_si_bulk_w_per_m_k", T, 250.0, 500.0, "K")
    return _out(c["k300"] * (T / 300.0) ** c["exponent"])


def k_si_film_w_per_m_k(t_k, film_nm, d0_nm: float | None = None, doping_factor: float = 1.0):
    """In-plane thermal conductivity of a single-crystal SOI device layer, W/(m K) (derived Matthiessen fit).

    1/k = 1/k_bulk(T) + 1/k_b, k_b = 148 (d/d0) c_p(T)/c_p(300 K), d0 = 170 nm: re-anchored to Aubain & Bandaru 2010
    (260 nm: 90 +/- 10 W/mK, which maps to d0 = 125-221 nm) (02 log #14). 220 nm: 83.5 at 300 K, -13 % to 375 K;
    200 nm: 80.0 / 70.1; 160 nm: 71.8 / 64.2 W/mK (300 / 375 K). +/-15 %. Valid 150-300 nm only (it gives 45 vs ~74
    at 74 nm and 105 vs ~89 at 420 nm). ``doping_factor`` ~0.8 for 1e19-1e20 cm^-3 regions (assumed; ~1e18 adds
    < 10 %). Supersedes the '56-66 W/mK' of 07 P1-3."""
    c = _c("k_si_film")
    d0 = c["d0_nm"] if d0_nm is None else d0_nm
    T = _arr(t_k)
    d = _arr(film_nm)
    _check_range("k_si_film_w_per_m_k (film)", d, *_valid_range("k_si_film", "film_nm"), unit="nm")
    _check_range("k_si_film_w_per_m_k", T, 250.0, 500.0, "K")
    kb = float(_c("k_si_bulk")["k300"]) * (d / d0) * _arr(cp_si_j_per_kg_k(T)) / float(cp_si_j_per_kg_k(300.0))
    return _out(doping_factor / (1.0 / _arr(k_si_bulk_w_per_m_k(T)) + 1.0 / kb))


def _valid_range(key: str, var: str):
    """Supported range [lo, hi] of variable ``var`` for material entry ``key`` (from materials.json)."""
    r = data()[key].get("valid_range", {}).get(var)
    return (None, None) if r is None else (float(r[0]), float(r[1]))


def k_sio2_w_per_m_k(t_k, kind: str = "thermal"):
    """SiO2 thermal conductivity, W/(m K): 1.38 (T/300)^0.33 (Palankovski 2000, printed), 250-500 K, +/-10 %.

    thermal/fused/box: 1.38 at 300 K, 1.485 at 375 K (Corning prints 1.30 or 1.38); pecvd/beol: 0.9x (Lee & Cahill
    1997, films > 100 nm; apparent 0.83-1.23 for 50-500 nm films)."""
    k = _sio2_kind(kind)
    c = _c("k_sio2")
    T = _arr(t_k)
    _check_range("k_sio2_w_per_m_k", T, 250.0, 500.0, "K")
    f = c["pecvd_factor"] if k == "pecvd" else 1.0
    return _out(f * c["k300"] * (T / 300.0) ** c["exponent"])


def k_low_k_w_per_m_k(t_k, kind: str = "dense"):
    """Low-k SiOC:H ILD thermal conductivity, W/(m K): dense 0.7, 30 % porous 0.1 (2014 microfin study, printed
    abstract); temperature-independent (assumed); uncertainty a factor ~2."""
    c = _c("k_low_k")
    if kind not in ("dense", "porous30"):
        raise ValueError("low-k kind must be 'dense' or 'porous30'")
    T = _arr(t_k)
    return _out(np.full_like(T, c[kind]))


_METALS = ("cu_bulk", "cu_beol", "w_bulk", "w_film", "tin_film")


def k_metal_w_per_m_k(metal: str, t_k, corner: str = "nominal"):
    """Thermal conductivity of BEOL metals and heater films, W/(m K), temperature-independent over 273-400 K (assumed).

    cu_bulk 400 (handbook, assumed); cu_beol 315 (250-380, Wiedemann-Franz with rho 2-2.5 uOhm cm, derived); w_bulk 173
    (assumed); w_film 80 (60-100, WF plus a 62 % phonon part, derived/assumed); tin_film 20 (10-30, assumed).
    ``corner`` = nominal | low | high."""
    if metal not in _METALS:
        raise ValueError(f"metal must be one of {_METALS}")
    v = _corner(_c("k_metals")[metal], corner)
    return _out(np.full_like(_arr(t_k), v))


def thermal_boundary_resistance_m2_k_per_w(interface: str = "si_sio2"):
    """Thermal boundary resistance, m^2 K/W: si_sio2 2e-8 (Lee & Cahill 1997, printed; ~20 nm of oxide, negligible
    next to a 2 um BOX); metal_dielectric 2e-7 (10x, printed ratio, value inferred; ~0.25 um of oxide, NOT negligible
    for a metal heater <= 1 um from the ring)."""
    c = _c("thermal_boundary_resistance")
    if interface not in c:
        raise ValueError(f"interface must be one of {sorted(c)}")
    return float(c[interface])


def _cp_palankovski(T, c):
    x = (T / 300.0) ** c["beta"]
    return c["c300"] + c["c1"] * (x - 1.0) / (x + c["c1"] / c["c300"])


def cp_si_j_per_kg_k(t_k):
    """Specific heat of Si, J/(kg K): Palankovski 2000 form (printed coefficients 711, 255, 1.85), 200-400 K checked;
    within 0.5 % (298 K) and 1.2 % (400 K) of NIST-JANAF. 677.1 / 711.0 / 780.7 / 797.9 at 273 / 300 / 375 / 400 K."""
    T = _arr(t_k)
    _check_range("cp_si_j_per_kg_k", T, 200.0, 500.0, "K")
    return _out(_cp_palankovski(T, _c("cp_si")))


def cp_sio2_j_per_kg_k(t_k):
    """Specific heat of amorphous SiO2, J/(kg K): Palankovski 2000 form (709, 696, 1.5), ~250-500 K, +/-5 %."""
    T = _arr(t_k)
    _check_range("cp_sio2_j_per_kg_k", T, 250.0, 500.0, "K")
    return _out(_cp_palankovski(T, _c("cp_sio2")))


RHO_SI_KG_PER_M3 = float(_c("density")["si"])
RHO_SIO2_KG_PER_M3 = float(_c("density")["sio2"])


def rho_c_j_per_m3_k(material: str, t_k, corner: str = "nominal"):
    """Volumetric heat capacity rho*c_p, J/(m^3 K).

    si (also si_film, si_bulk, doped Si): 2329 x c_p,Si(T): 1.656e6 at 300 K, 1.818e6 at 375 K (derived, +/-1.5 %).
    sio2 (thermal, pecvd, beol, box, fused): 2200 x c_p,SiO2(T): 1.560e6 / 1.816e6 (derived, +/-5 %).
    cu 3.45e6, w 2.55e6, tin 3.1e6 (assumed handbook values, constant), low_k 1.3e6 (assumed, NO source; 0.9-1.8e6)."""
    T = _arr(t_k)
    m = _RHO_C_ALIASES.get(material.lower())
    if m is None:
        raise ValueError(f"unknown material {material!r} for rho_c; use one of {sorted(_RHO_C_ALIASES)}")
    if m == "si":
        return _out(RHO_SI_KG_PER_M3 * _arr(cp_si_j_per_kg_k(T)))
    if m == "sio2":
        return _out(RHO_SIO2_KG_PER_M3 * _arr(cp_sio2_j_per_kg_k(T)))
    return _out(np.full_like(T, _corner(_c("rho_c")[m], corner)))


_RHO_C_ALIASES = {
    "si": "si", "si_film": "si", "si_bulk": "si", "si_doped": "si",
    "sio2": "sio2", "sio2_thermal": "sio2", "sio2_pecvd": "sio2", "sio2_beol": "sio2", "sio2_fused": "sio2",
    "box": "sio2", "cu": "cu", "cu_beol": "cu", "cu_bulk": "cu", "w": "w", "w_film": "w", "w_bulk": "w",
    "tin": "tin", "tin_film": "tin", "low_k": "low_k", "low_k_dense": "low_k", "low_k_porous30": "low_k",
}


THERMAL_MATERIALS = ("si_film", "si_bulk", "sio2_thermal", "sio2_pecvd", "low_k_dense", "low_k_porous30",
                     "cu_beol", "cu_bulk", "w_film", "w_bulk", "tin_film")
"""Material names accepted by :func:`thermal_properties` (the thermal FEM uses these)."""


def thermal_properties(material: str, t_k, film_nm=None, corner: str = "nominal") -> dict:
    """Thermal conductivity, volumetric heat capacity and diffusivity of one stack material at T.

    ``material`` in :data:`THERMAL_MATERIALS`; ``si_film`` needs ``film_nm``. Returns
    ``{"k_w_per_m_k", "rho_c_j_per_m3_k", "diffusivity_m2_per_s"}`` (arrays broadcast over t_k)."""
    m = material
    if m == "si_film":
        if film_nm is None:
            raise ValueError("si_film needs film_nm")
        k = k_si_film_w_per_m_k(t_k, film_nm)
    elif m == "si_bulk":
        k = k_si_bulk_w_per_m_k(t_k)
    elif m in ("sio2_thermal", "sio2_pecvd"):
        k = k_sio2_w_per_m_k(t_k, m.split("_")[1])
    elif m in ("low_k_dense", "low_k_porous30"):
        k = k_low_k_w_per_m_k(t_k, m[len("low_k_"):])
    elif m in _METALS:
        k = k_metal_w_per_m_k(m, t_k, corner)
    else:
        raise ValueError(f"material must be one of {THERMAL_MATERIALS}")
    rc = rho_c_j_per_m3_k(m, t_k, corner)
    return {"k_w_per_m_k": k, "rho_c_j_per_m3_k": rc, "diffusivity_m2_per_s": _out(_arr(k) / _arr(rc))}


# ============================================================================================ free carriers
PLASMA_MODELS = {"nsm2011": "plasma_nsm2011", "soref1987": "plasma_soref1987"}
"""Plasma-dispersion switch: ``nsm2011`` (Nedeljkovic, Soref, Mashanovich 2011, default) or ``soref1987``."""


def _plasma_coeffs(model, band_um):
    if model not in PLASMA_MODELS:
        raise ValueError(f"model must be one of {sorted(PLASMA_MODELS)}")
    c = _c(PLASMA_MODELS[model])
    key = f"{band_um:g}"
    if key not in c:
        raise ValueError(f"{model}: band_um must be one of {sorted(c)} (got {band_um})")
    k = c[key]
    if model == "nsm2011":
        return k["p"], k["q"], k["r"], k["s"], k["a"], k["b"], k["c"], k["d"]
    return k["pe"], k["qe"], k["ph"], k["sh"], k["ae"], k["be"], k["ah"], k["bh"]


def plasma_dispersion(ne_per_cm3, nh_per_cm3, model: str = "nsm2011", band_um: float = 1.3, warn: bool = True):
    """Free-carrier index change and absorption, returns ``(delta_n, delta_alpha_per_m)``.

    delta_n = -(p Ne^q + r Nh^s), delta_alpha = a Ne^b + c Nh^d (alpha is the intensity absorption coefficient), with
    Ne, Nh the free electron and hole densities in cm^-3 relative to intrinsic silicon (>= 0).

    - ``nsm2011`` (default): Nedeljkovic, Soref, Mashanovich, IEEE Photon. J. 3, 1171 (2011), 1.3 um row (printed;
      transcribed from Tidy3D 2.12.0 - the primary PDF was not opened). ``band_um=1.55`` selects the 1.55 um row.
    - ``soref1987``: Soref & Bennett, IEEE JQE 23, 123 (1987), 1.3 um: delta_n = -(6.2e-22 Ne + 6.0e-18 Nh^0.8),
      delta_alpha = 6.0e-18 Ne + 4.0e-18 Nh /cm (printed).

    Valid ~1e17-1e20 cm^-3 at 300 K (the 1.3 um row is used for 1.26-1.36 um, '07 P1-7 at 1.31 um'); below 1e17 the
    power laws are extrapolations (a warning is issued for 0 < N < 1e17 when ``warn``). At 1e18 each NSM gives a 10 %
    smaller |delta_n| and a 13 % smaller delta_alpha than SB. For a depletion junction use
    :func:`plasma_dispersion_change` on absolute densities, not a signed power of the change."""
    Ne, Nh = _arr(ne_per_cm3), _arr(nh_per_cm3)
    if np.any(Ne < 0) or np.any(Nh < 0):
        raise ValueError("carrier densities must be >= 0 (use plasma_dispersion_change for a depletion change)")
    if warn:
        nz = np.concatenate([np.ravel(Ne[Ne > 0]), np.ravel(Nh[Nh > 0])])
        _check_range(f"plasma_dispersion[{model}]", nz, 1e17, 1e20, "cm^-3")
    p, q, r, s, a, b, c, d = _plasma_coeffs(model, band_um)
    dn = -(p * Ne**q + r * Nh**s)
    da_per_cm = a * Ne**b + c * Nh**d
    return _out(dn), _out(da_per_cm * 100.0)


def plasma_dispersion_change(ne0_per_cm3, nh0_per_cm3, ne1_per_cm3, nh1_per_cm3, model: str = "nsm2011",
                             band_um: float = 1.3):
    """Change of (delta_n, delta_alpha_per_m) when the free-carrier densities go from state 0 to state 1.

    f(N1) - f(N0) on absolute densities: the physically consistent use of the power-law fits for a depletion
    modulator. Removing carriers (reverse bias) gives a positive index change (red shift) and a negative loss change."""
    n1, a1 = plasma_dispersion(ne1_per_cm3, nh1_per_cm3, model, band_um, warn=False)
    n0, a0 = plasma_dispersion(ne0_per_cm3, nh0_per_cm3, model, band_um, warn=False)
    return _out(_arr(n1) - _arr(n0)), _out(_arr(a1) - _arr(a0))


def plasma_dispersion_derivatives(ne_per_cm3, nh_per_cm3, model: str = "nsm2011", band_um: float = 1.3) -> dict:
    """Small-signal coefficients at (Ne, Nh) (derived, analytic): ``dn_dne_cm3``, ``dn_dnh_cm3`` (cm^3) and the
    absorption cross sections ``sigma_e_cm2``, ``sigma_h_cm2`` (cm^2). NSM 1.3 um at 1e18: sigma_e 5.66e-18,
    sigma_h 4.44e-18 cm^2, dn/dNe -5.88e-22, dn/dNh -1.12e-21 cm^3. Densities must be > 0."""
    Ne, Nh = _arr(ne_per_cm3), _arr(nh_per_cm3)
    if np.any(Ne <= 0) or np.any(Nh <= 0):
        raise ValueError("densities must be > 0 for the derivatives")
    p, q, r, s, a, b, c, d = _plasma_coeffs(model, band_um)
    return {"dn_dne_cm3": _out(-p * q * Ne ** (q - 1)), "dn_dnh_cm3": _out(-r * s * Nh ** (s - 1)),
            "sigma_e_cm2": _out(a * b * Ne ** (b - 1)), "sigma_h_cm2": _out(c * d * Nh ** (d - 1))}


def plasma_temperature_factor(t_k, quantity: str = "dn", corner: str = "nominal"):
    """Multiplicative temperature factor 1 + c (T - 300 K) for the per-carrier plasma effect (ASSUMED, 02 §7).

    No measurement exists at 1.3 um above 300 K. Drude trend (Riffe 2002, trend only): |delta_n| per carrier
    -0.02 to -0.05 %/K (nominal -0.035 %/K), delta_alpha +0.1 to +0.3 %/K (nominal +0.2 %/K). corner = nominal | low |
    high (low/high = the ends of the coefficient range). The larger junction temperature effect is V_bi(T)."""
    c = _c("plasma_temperature")
    if quantity not in ("dn", "dalpha"):
        raise ValueError("quantity must be 'dn' or 'dalpha'")
    T = _arr(t_k)
    return _out(1.0 + _corner(c[quantity], corner) * (T - c["t_ref_k"]))


def tpa_beta_m_per_w(corner: str = "nominal") -> float:
    """Two-photon absorption coefficient of Si near 1310 nm, m/W: nominal 0.9 cm/GW = 9e-12 m/W (inferred central),
    range 0.5-2.0 cm/GW (Lin 2007: 0.57 +/- 0.17 at 1315 nm; Bristow 2007: 2.1 +/- 0.4 at 1220 nm; factor-3 group
    disagreement, not resolved). For the P2-1 exclusion bound tests only."""
    return _corner(_c("tpa_beta"), corner)


def kerr_n2_m2_per_w(corner: str = "nominal") -> float:
    """Kerr index of Si near 1310 nm, m^2/W: nominal 4e-14 cm^2/W = 4e-18 m^2/W, range 3-6e-14 cm^2/W (Bristow 2007,
    Dinu 2003, inferred). For the P2-1 exclusion bound tests only."""
    return _corner(_c("kerr_n2"), corner)


# ============================================================================================ junction physics
def eg_si_ev(t_k):
    """Si band gap, eV: Varshni form 1.170 - 4.73e-4 T^2/(T + 636) (Thurmond 1975, printed), 0-1000 K, +/-2 meV.

    1.1312 / 1.1245 / 1.1042 / 1.0969 eV at 273 / 300 / 375 / 400 K."""
    c = _c("si_bandgap")
    T = _arr(t_k)
    return _out(c["eg0"] - c["a"] * T**2 / (T + c["b"]))


def ni_si_per_cm3(t_k):
    """Si intrinsic density, cm^-3: 5.29e19 (T/300)^2.54 exp(-6726/T) (Misiakos & Tsamakis 1993, printed),
    78-340 K, 9.70e9 +/- 1 % at 300 K; 1.51e12 at 375 K and 5.47e12 at 400 K are extrapolations."""
    c = _c("si_intrinsic_density")
    T = _arr(t_k)
    _check_range("ni_si_per_cm3", T, 78.0, 400.0, "K")   # 340-400 K: extrapolation used by the 02 tables
    return _out(c["A"] * (T / 300.0) ** c["m"] * np.exp(-c["B"] / T))


def vbi_v(na_per_cm3, nd_per_cm3, t_k):
    """Built-in voltage of an abrupt PN junction, V: (kT/q) ln(Na Nd / n_i(T)^2) (derived; full ionization, no
    band-gap narrowing, which lowers it by a few tens of mV at ~1e18). 1e18/1e18: 0.984 / 0.954 / 0.866 / 0.835 V at
    273 / 300 / 375 / 400 K; 5e17/5e17: 0.918 V at 300 K."""
    T = _arr(t_k)
    return _out(K_B_EV * T * np.log(_arr(na_per_cm3) * _arr(nd_per_cm3) / _arr(ni_si_per_cm3(T)) ** 2))


def dvbi_dt_v_per_k(na_per_cm3, nd_per_cm3, t_k):
    """dV_bi/dT, V/K (analytic): V_bi/T - 2 k (m + B/T) with the Misiakos n_i law. -1.12e-3 V/K at 300 K and
    -1.22e-3 V/K at 375 K for 1e18/1e18 (derived)."""
    c = _c("si_intrinsic_density")
    T = _arr(t_k)
    return _out(_arr(vbi_v(na_per_cm3, nd_per_cm3, T)) / T - 2.0 * K_B_EV * (c["m"] + c["B"] / T))


# ================================================================================================= heaters
def mobility_arora_cm2_per_v_s(n_per_cm3, t_k, carrier: str = "n"):
    """Carrier mobility in doped Si, cm^2/(V s): Arora, Hauser, Roulston, IEEE TED 29, 292 (1982) (coefficients via
    TCAD documentation, printed there), 250-500 K, 1e13-1e20 cm^-3 (weak above 1e19)."""
    if carrier not in ("n", "p"):
        raise ValueError("carrier must be 'n' or 'p'")
    c = _c("doped_si_mobility")["arora_" + carrier]
    N, T = _arr(n_per_cm3), _arr(t_k)
    _check_range("mobility_arora", T, 250.0, 500.0, "K")
    Tn = T / 300.0
    a = c["alpha0"] * Tn ** c["alpha_exp"]
    return _out(c["mu_min"] * Tn ** c["mu_min_exp"] + c["mu0_prefactor"] * T ** c["mu0_T_exp"]
                / (1.0 + (N / (c["Nref"] * Tn ** c["Nref_exp"])) ** a))


def mobility_masetti_cm2_per_v_s(n_per_cm3, carrier: str = "n"):
    """Carrier mobility at 300 K, cm^2/(V s): Masetti, Severi, Solmi, IEEE TED 30, 764 (1983); n = phosphorus,
    p = boron (printed coefficients via TCAD documentation)."""
    if carrier not in ("n", "p"):
        raise ValueError("carrier must be 'n' or 'p'")
    c = _c("doped_si_mobility")["masetti_" + carrier]
    N = _arr(n_per_cm3)
    return _out(c["mu_min1"] * np.exp(-c["Pc"] / N) + (c["mu_max"] - c["mu_min2"]) / (1.0 + (N / c["Cr"]) ** c["alpha"])
                - c["mu1"] / (1.0 + (c["Cs"] / N) ** c["beta"]))


def resistivity_doped_si_ohm_m(n_per_cm3, t_k=300.0, carrier: str = "n", model: str = "arora"):
    """Resistivity of doped single-crystal Si, ohm m: 1/(q N mu), full ionization (derived). model = arora (T
    dependent) or masetti (300 K only). n 1e19: 5.47e-5 ohm m (Arora). Reference table at 23 C: n 2.3e-4 / 5.5e-5 /
    7.8e-6, p 4.5e-4 / 8.5e-5 / 1.2e-5 ohm m at 1e18 / 1e19 / 1e20 (printed)."""
    N = _arr(n_per_cm3)
    if model == "arora":
        mu = _arr(mobility_arora_cm2_per_v_s(N, t_k, carrier))
    elif model == "masetti":
        mu = _arr(mobility_masetti_cm2_per_v_s(N, carrier))
    else:
        raise ValueError("model must be 'arora' or 'masetti'")
    return _out(1.0 / (Q_E * N * mu) * 1e-2)


def tcr_doped_si_per_k(n_per_cm3, t_k=300.0, carrier: str = "n", h_k: float = 0.05):
    """TCR of doped Si, 1/K: (1/rho) drho/dT with Arora mobility, full ionization (derived; central difference).

    300 K: n 1e18 1.26e-3, n 1e19 1.25e-3, n 1e20 1.75e-3; 375 K: 1.52e-3 / 1.06e-3 / 1.39e-3. +/-50 %:
    incomplete ionization at 1e17-1e18 lowers it; Arora is weak above 1e19."""
    T = _arr(t_k)
    rp = _arr(resistivity_doped_si_ohm_m(n_per_cm3, T + h_k, carrier))
    rm = _arr(resistivity_doped_si_ohm_m(n_per_cm3, T - h_k, carrier))
    r0 = _arr(resistivity_doped_si_ohm_m(n_per_cm3, T, carrier))
    return _out((rp - rm) / (2 * h_k) / r0)


HEATER_FAMILIES = tuple(k for k in _c("heater_tcr_family") if isinstance(_c("heater_tcr_family")[k], dict))
"""Heater materials with a TCR entry: doped_si, tin, silicide, w, cu."""
HEATER_ROM_FAMILY_MAP = dict(data()["heater_tcr_family"]["rom_family_map"])
"""Thermal-ROM family -> heater material (inring_noundercut -> doped_si, silicide_undercut -> silicide,
metal_above -> tin; metal_above may also be W)."""


def heater_tcr_per_k(family: str, corner: str = "nominal") -> float:
    """Heater TCR, 1/K, tied to the heater family (07 P1-11, 02 §9).

    doped_si 1.4e-3 (1.0-1.8e-3, derived from Arora); tin 0.5e-3 (0.487-0.55e-3, printed; low-stress films can be
    nonlinear and change sign); silicide 3.0e-3 (2-4e-3, assumed, no source); w 4.5e-3 (2.0-4.6e-3; bulk 4.5e-3,
    thin-film value set by deposition); cu 3.9e-3 (assumed). ``family`` may also be a thermal-ROM family name
    (:data:`HEATER_ROM_FAMILY_MAP`). The textbook's G25 fit is 1e-3 /K, the low end of doped Si."""
    fam = HEATER_ROM_FAMILY_MAP.get(family, family)
    c = _c("heater_tcr_family")
    if fam not in HEATER_FAMILIES:
        raise ValueError(f"family must be one of {HEATER_FAMILIES} or {tuple(HEATER_ROM_FAMILY_MAP)}")
    return _corner(c[fam], corner)


def heater_resistance_ratio(t_k, t0_k, tcr_per_k):
    """R(T)/R(T0) = 1 + TCR (T - T0): the linear law the TCR values above are defined for."""
    return _out(1.0 + _arr(tcr_per_k) * (_arr(t_k) - _arr(t0_k)))


# ======================================================================================= germanium photodiode
def ge_pd_responsivity_limit_a_per_w(lam_m, eta=1.0):
    """Quantum-limited responsivity eta lam / (hc/q), A/W (derived): 1.0566 A/W at 1310 nm and eta = 1.
    Lightmatter G25: 0.88-1.04 A/W over 500 devices, 0.9 A/W printed worst case."""
    return _out(_arr(eta) * _arr(lam_m) * 1e6 / HC_EV_UM)


def ge_pd_responsivity_a_per_w(r0_a_per_w, t_k, t0_k=298.15, tc_per_k=0.0):
    """Ge PD responsivity at temperature T, A/W: r0 (1 + tc (T - T0)). At 1310 nm absorption is far above Ge's
    direct edge, so the model is constant (tc = 0) with an assumed +/-3 % over 0-105 C (tc within +/-3.75e-4 /K);
    no 1310 nm measurement was found (02 §10)."""
    return _out(_arr(r0_a_per_w) * (1.0 + _arr(tc_per_k) * (_arr(t_k) - _arr(t0_k))))


GE_PD_DARK_I0_A_RANGE = tuple(_c("ge_pd_dark_current")["i_d0_a_range_25C"])
"""Dark current at 25 C, A: 1-30 nA (07 D20; GF prints < 30 nA)."""


def ge_pd_dark_ea_ev(corner: str = "nominal") -> float:
    """Dark-current activation energy, eV: nominal 0.37, range 0.35-0.38 (Colace 2008 x1.6 per 10 C, Koester 2006
    x10 to 85 C; inferred), corner 'stress' = 0.5 eV (Balbi 2009, x1.9 per 10 C; 02 log #23)."""
    return _corner(_c("ge_pd_dark_current")["ea_ev"], corner)


def ge_pd_dark_current_a(i0_a, t_k, t0_k=298.15, ea_ev=None):
    """Ge PD dark current, A: I0 exp[-(Ea/k)(1/T - 1/T0)] (Arrhenius; 07 P1-9). Default Ea 0.37 eV: x11.2 at 85 C and
    x21.0 at 105 C relative to 25 C. Data cover 25-85 C; 105 C is extrapolated."""
    ea = ge_pd_dark_ea_ev() if ea_ev is None else ea_ev
    T, T0 = _arr(t_k), _arr(t0_k)
    _check_range("ge_pd_dark_current_a", T, 270.0, 380.0, "K")
    return _out(_arr(i0_a) * np.exp(-_arr(ea) / K_B_EV * (1.0 / T - 1.0 / T0)))


# ============================================================================================ uncertainty summary
def material_uncertainties() -> dict:
    """Flat summary of the material uncertainties, for the uncertainty sampler of ``mrm.params.realize`` (07 P1-16).

    Each value is ``{"kind": "abs" | "rel" | "range" | "switch" | "factor", ...}``: ``abs``/``rel`` carry ``sigma``
    (a source-spread half-width, not a fitted standard deviation); ``range`` carries ``lo, hi`` (and ``nominal``);
    ``switch`` lists alternatives. Every number comes from materials.json."""
    d = data()
    fam = d["heater_tcr_family"]["coefficients"]
    u = {
        "si_index_abs": {"kind": "abs", "sigma": d["si_index_recommended"]["uncertainty"]["n_abs"],
                         "note": "recommended (Frey) level"},
        "si_index_level": {"kind": "switch", "nominal": "frey", "alternatives": ["li"],
                           "delta_at_1310nm": -float(si_index_offset_frey(LAM_REF_M))},
        "si_dn_dt_per_k": {"kind": "abs", "sigma_300K": d["si_dn_dt"]["coefficients"]["sigma_300K"],
                           "sigma_375K": d["si_dn_dt"]["coefficients"]["sigma_375K"], "function": "dn_dt_si_sigma_per_k",
                           "low_corner": "franta2017", "high_corner": "lukes"},
        "si_dn_dt_slope_per_k": {"kind": "range", "nominal": float(si_dn_dt_slope_per_k(LAM_REF_M, 300.0)),
                                 "lo": d["si_dn_dt_slope"]["coefficients"]["range_per_k"][0],
                                 "hi": d["si_dn_dt_slope"]["coefficients"]["range_per_k"][1]},
        "si_n_g": {"kind": "abs", "sigma": d["si_group_index"]["uncertainty"]["abs"]},
        "sio2_n_thermal": {"kind": "abs", "sigma": d["sio2_index_films"]["uncertainty"]["thermal_n_abs"]},
        "sio2_n_pecvd": {"kind": "abs", "sigma": d["sio2_index_films"]["uncertainty"]["pecvd_n_abs"]},
        "sio2_dn_dt": {"kind": "rel", "sigma": d["sio2_dn_dt"]["uncertainty"]["rel"]},
        "si_alpha_l_per_k": {"kind": "abs", "sigma": d["si_alpha_l"]["uncertainty"]["abs"]},
        "sio2_alpha_l_per_k": {"kind": "abs", "sigma": d["sio2_alpha_l"]["uncertainty"]["abs"]},
        "k_si_bulk": {"kind": "rel", "sigma": d["k_si_bulk"]["uncertainty"]["rel"]},
        "k_si_film": {"kind": "rel", "sigma": d["k_si_film"]["uncertainty"]["rel"],
                      "d0_nm_range": d["k_si_film"]["coefficients"]["d0_nm_range"]},
        "k_sio2": {"kind": "rel", "sigma": d["k_sio2"]["uncertainty"]["rel"]},
        "k_low_k": {"kind": "factor", "factor": d["k_low_k"]["uncertainty"]["factor"]},
        "rho_c_si": {"kind": "rel", "sigma": d["rho_c"]["uncertainty"]["si_rel"]},
        "rho_c_sio2": {"kind": "rel", "sigma": d["rho_c"]["uncertainty"]["sio2_rel"]},
        "plasma_model": {"kind": "switch", "nominal": "nsm2011", "alternatives": ["soref1987"]},
        "plasma_rel": {"kind": "rel", "sigma": d["plasma_nsm2011"]["uncertainty"]["rel"]},
        "tpa_beta_m_per_w": {"kind": "range", "nominal": tpa_beta_m_per_w(), "lo": tpa_beta_m_per_w("low"),
                             "hi": tpa_beta_m_per_w("high")},
        "kerr_n2_m2_per_w": {"kind": "range", "nominal": kerr_n2_m2_per_w(), "lo": kerr_n2_m2_per_w("low"),
                             "hi": kerr_n2_m2_per_w("high")},
        "ge_dark_ea_ev": {"kind": "range", "nominal": ge_pd_dark_ea_ev(), "lo": ge_pd_dark_ea_ev("low"),
                          "hi": ge_pd_dark_ea_ev("high"), "stress": ge_pd_dark_ea_ev("stress")},
        "ge_dark_i0_a_25c": {"kind": "range", "lo": GE_PD_DARK_I0_A_RANGE[0], "hi": GE_PD_DARK_I0_A_RANGE[1]},
        "ge_resp_tc_per_k": {"kind": "range", "nominal": 0.0,
                             "lo": d["ge_pd_responsivity"]["coefficients"]["tc_per_k_range"][0],
                             "hi": d["ge_pd_responsivity"]["coefficients"]["tc_per_k_range"][1]},
    }
    for f in HEATER_FAMILIES:
        u[f"heater_tcr_per_k_{f}"] = {"kind": "range", "nominal": fam[f]["nominal"], "lo": fam[f]["range"][0],
                                      "hi": fam[f]["range"][1], "provenance": fam[f]["provenance"]}
    for metal in _METALS:
        spec = d["k_metals"]["coefficients"][metal]
        u[f"k_{metal}"] = {"kind": "range", "nominal": spec["nominal"], "lo": spec["range"][0], "hi": spec["range"][1]}
    return u


# ================================================================================================ docs table
def _src(key):
    e = data()[key]
    return e.get("source", ""), e.get("provenance", "")


def material_table(lam_m: float = LAM_REF_M, temps_k=(300.0, 375.0)) -> list[dict]:
    """The material values used by the plant, evaluated by this module at ``lam_m`` and ``temps_k``.

    Each row: ``property, symbol, unit, values (one per temperature, in the display unit), scale (display = SI x
    scale), function, uncertainty, valid_range, source, provenance``. Used to generate the materials part of
    docs/PARAMETERS.md (see :func:`material_table_markdown`)."""
    T = np.asarray(temps_k, dtype=float)
    rows: list[dict] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)

        def add(prop, symbol, unit, values, func, key, unc, rng, scale=1.0, src=None, prov=None):
            s, p = _src(key)
            vals = np.broadcast_to(np.asarray(values, dtype=float), T.shape) * scale
            rows.append({"property": prop, "symbol": symbol, "unit": unit, "values": [float(v) for v in vals],
                         "scale": scale, "function": func, "uncertainty": unc, "valid_range": rng,
                         "source": src or s, "provenance": prov or p})

        add("Si index (recommended: Li shape, Frey level)", "n_Si", "1", n_si(lam_m, T), "n_si", "si_index_recommended",
            "+/-5e-4 (Li level is 4.1e-3 lower)", "1.26-1.36 um, 250-400 K")
        add("Si index (Li 1980 level, alternative)", "n_Si,Li", "1", n_si(lam_m, T, "li"), "n_si(level='li')",
            "si_index_li1980", "+/-2e-3", "1.2-14 um, 100-750 K")
        add("Si thermo-optic coefficient", "dn_Si/dT", "1e-4/K", dn_dt_si_per_k(lam_m, T), "dn_dt_si_per_k",
            "si_dn_dt", "+/-0.07e-4 (300 K), +/-0.10e-4 (375 K)", "1.2-14 um, 100-750 K", scale=1e4)
        add("Si second T-derivative", "d2n_Si/dT2", "1e-7/K^2", d2n_dt2_si_per_k2(lam_m, T), "d2n_dt2_si_per_k2",
            "si_d2n_dt2", "1.7-2.8e-7 across sources", "as Li", scale=1e7)
        add("Si dn/dT normalized slope", "beta_Si", "1e-3/K", si_dn_dt_slope_per_k(lam_m, T), "si_dn_dt_slope_per_k",
            "si_dn_dt_slope", "1.0-1.45e-3 across sources", "273-400 K", scale=1e3)
        add("Si material group index (recommended)", "n_g,Si", "1", n_g_si(lam_m, T), "n_g_si", "si_group_index",
            "+/-0.02", "1.26-1.36 um")
        add("Si material group index (Li level)", "n_g,Si,Li", "1", n_g_si(lam_m, T, "li"), "n_g_si(level='li')",
            "si_group_index", "+/-0.02", "1.26-1.36 um")
        add("SiO2 index, thermal oxide / BOX (absolute)", "n_SiO2", "1", n_sio2(lam_m, T), "n_sio2", "sio2_index_films",
            "+/-3e-3", "0.21-3.71 um")
        add("SiO2 index, PECVD / BEOL", "n_SiO2,PECVD", "1", n_sio2(lam_m, T, "pecvd"), "n_sio2(kind='pecvd')",
            "sio2_index_films", "+/-0.01", "0.21-3.71 um")
        add("SiO2 thermo-optic, thermal oxide", "dn_SiO2/dT", "1e-6/K", dn_dt_sio2_per_k(T), "dn_dt_sio2_per_k",
            "sio2_dn_dt", "+/-15 %", "30-310 K measured; above 300 K assumed", scale=1e6)
        add("SiO2 thermo-optic, PECVD / BEOL", "dn_SiO2,PECVD/dT", "1e-6/K", dn_dt_sio2_per_k(T, "pecvd"),
            "dn_dt_sio2_per_k(kind='pecvd')", "sio2_dn_dt", "+/-15 %", "as above", scale=1e6)
        add("SiO2 material group index", "n_g,SiO2", "1", n_g_sio2(lam_m, T), "n_g_sio2", "sio2_index_malitson",
            "+/-1e-4", "0.21-3.71 um")
        add("Si linear expansion", "alpha_L,Si", "1e-6/K", alpha_l_si_per_k(T), "alpha_l_si_per_k", "si_alpha_l",
            "+/-0.2e-6", "120-1500 K", scale=1e6)
        add("SiO2 linear expansion", "alpha_L,SiO2", "1e-6/K", alpha_l_sio2_per_k(T), "alpha_l_sio2_per_k",
            "sio2_alpha_l", "+/-0.05e-6", "273-473 K", scale=1e6)
        add("Ring thermal-shift ceiling (n_eff/n_g = 0.6)", "dlambda/dT_max", "pm/K",
            dlambda_dt_ceiling_pm_per_k(lam_m, T)[0], "dlambda_dt_ceiling_pm_per_k", "ring_thermal_shift_ceiling",
            "-4 to +3 pm/K", "Si-core, SiO2-clad rings")
        add("Si thermal conductivity, bulk", "k_Si", "W/(m K)", k_si_bulk_w_per_m_k(T), "k_si_bulk_w_per_m_k",
            "k_si_bulk", "+/-5 %", "250-500 K")
        for d in (160.0, 200.0, 220.0):
            add(f"Si thermal conductivity, {d:.0f} nm film (in-plane)", f"k_Si,{d:.0f}", "W/(m K)",
                k_si_film_w_per_m_k(T, d), f"k_si_film_w_per_m_k(t, {d:.0f})", "k_si_film", "+/-15 %",
                "150-300 nm films")
        add("SiO2 thermal conductivity, thermal oxide", "k_SiO2", "W/(m K)", k_sio2_w_per_m_k(T), "k_sio2_w_per_m_k",
            "k_sio2", "+/-10 % (Corning also prints 1.30)", "250-500 K")
        add("SiO2 thermal conductivity, PECVD / BEOL", "k_SiO2,PECVD", "W/(m K)", k_sio2_w_per_m_k(T, "pecvd"),
            "k_sio2_w_per_m_k(kind='pecvd')", "k_sio2", "+/-10 %", "films > 100 nm")
        add("Low-k SiOC:H (dense / 30 % porous)", "k_lowk", "W/(m K)", k_low_k_w_per_m_k(T, "dense"),
            "k_low_k_w_per_m_k", "k_low_k", "factor 2; porous 0.1", "room temperature")
        for m, lab in (("cu_beol", "Cu BEOL line"), ("w_film", "W film"), ("tin_film", "TiN film")):
            rng = _c("k_metals")[m]["range"]
            add(f"{lab} thermal conductivity", f"k_{m}", "W/(m K)", k_metal_w_per_m_k(m, T),
                f"k_metal_w_per_m_k('{m}')", "k_metals", f"{rng[0]:g}-{rng[1]:g}", "273-400 K (T-independent)")
        for m, lab, key in (("si", "Si", "rho_c"), ("sio2", "SiO2", "rho_c"), ("cu", "Cu", "rho_c"),
                            ("w", "W", "rho_c"), ("tin", "TiN", "rho_c"), ("low_k", "low-k", "rho_c")):
            unc = {"si": "+/-1.5 %", "sio2": "+/-5 %", "low_k": "0.9-1.8 (assumed, no source)"}.get(m, "+/-5 % (assumed)")
            prov = {"si": "derived", "sio2": "derived"}.get(m, "assumed")
            add(f"Volumetric heat capacity, {lab}", f"rho c_{lab}", "MJ/(m^3 K)", rho_c_j_per_m3_k(m, T),
                f"rho_c_j_per_m3_k('{m}')", key, unc, "273-400 K", scale=1e-6, prov=prov)
        add("Thermal boundary resistance Si/SiO2", "R_I", "1e-8 m^2 K/W", thermal_boundary_resistance_m2_k_per_w(),
            "thermal_boundary_resistance_m2_k_per_w", "thermal_boundary_resistance", "metal/dielectric ~10x",
            "78-400 K", scale=1e8)
        for model, lab, key in (("nsm2011", "NSM 2011 (default)", "plasma_nsm2011"),
                                ("soref1987", "Soref-Bennett 1987", "plasma_soref1987")):
            dn, da = plasma_dispersion(1e18, 1e18, model)
            add(f"Free-carrier delta_n at 1e18 e + 1e18 h, {lab}", "delta_n_fc", "1e-3", dn,
                f"plasma_dispersion(model='{model}')", key, "~+/-15 %", "1e17-1e20 cm^-3, 300 K", scale=1e3)
            add(f"Free-carrier delta_alpha at 1e18 e + 1e18 h, {lab}", "delta_alpha_fc", "1/cm", da,
                f"plasma_dispersion(model='{model}')", key, "~+/-15 %", "1e17-1e20 cm^-3, 300 K", scale=1e-2)
        add("Two-photon absorption", "beta_TPA", "cm/GW", tpa_beta_m_per_w(), "tpa_beta_m_per_w", "tpa_beta",
            "0.5-2.0 cm/GW", "~1.3 um, 300 K", scale=1e11)
        add("Kerr index", "n_2", "1e-14 cm^2/W", kerr_n2_m2_per_w(), "kerr_n2_m2_per_w", "kerr_n2", "3-6e-14",
            "1.2-1.55 um", scale=1e18)
        add("Si band gap", "E_g", "eV", eg_si_ev(T), "eg_si_ev", "si_bandgap", "+/-2 meV", "0-1000 K")
        add("Si intrinsic density", "n_i", "1e10 cm^-3", ni_si_per_cm3(T), "ni_si_per_cm3", "si_intrinsic_density",
            "+/-1 % at 300 K", "78-340 K (extrapolated above)", scale=1e-10)
        add("Built-in voltage, 1e18/1e18 cm^-3", "V_bi", "V", vbi_v(1e18, 1e18, T), "vbi_v", "pn_builtin_voltage",
            "BGN lowers by ~0.05 V", "273-400 K")
        add("dV_bi/dT, 1e18/1e18 cm^-3", "dV_bi/dT", "mV/K", dvbi_dt_v_per_k(1e18, 1e18, T), "dvbi_dt_v_per_k",
            "pn_builtin_voltage", "-", "273-400 K", scale=1e3)
        add("TCR of n-type Si heater, 1e19 cm^-3 (Arora)", "TCR_Si", "1e-3/K", tcr_doped_si_per_k(1e19, T),
            "tcr_doped_si_per_k", "doped_si_tcr", "+/-50 %", "250-500 K", scale=1e3)
        for fam in HEATER_FAMILIES:
            spec = _c("heater_tcr_family")[fam]
            add(f"Heater TCR, family {fam}", f"TCR_{fam}", "1e-3/K", heater_tcr_per_k(fam), f"heater_tcr_per_k('{fam}')",
                "heater_tcr_family", f"{spec['range'][0] * 1e3:g}-{spec['range'][1] * 1e3:g}e-3", "273-400 K",
                scale=1e3, src=spec["source"], prov=spec["provenance"])
        add("Ge PD responsivity limit (eta = 1)", "R_max", "A/W", ge_pd_responsivity_limit_a_per_w(lam_m),
            "ge_pd_responsivity_limit_a_per_w", "ge_pd_responsivity", "G25 measured 0.88-1.04", "-")
        add("Ge PD dark current, ratio to 25 C (Ea 0.37 eV)", "I_d/I_d0", "1", ge_pd_dark_current_a(1.0, T),
            "ge_pd_dark_current_a", "ge_pd_dark_current", "Ea 0.35-0.38 eV (stress 0.5)", "25-85 C data")
    return rows


def material_table_markdown(lam_m: float = LAM_REF_M, temps_k=(300.0, 375.0)) -> str:
    """Markdown rendering of :func:`material_table`."""
    rows = material_table(lam_m, temps_k)
    head = ["Property", "Symbol", "Unit"] + [f"{t:g} K" for t in temps_k] + ["Uncertainty", "Valid range",
                                                                              "Provenance", "Function"]
    out = [f"Materials model `{MODEL_ID}` at {lam_m * 1e9:.0f} nm (sha256 {data_sha256()[:12]}).", "",
           "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        vals = [f"{v:.5g}" for v in r["values"]]
        out.append("| " + " | ".join([r["property"], r["symbol"], r["unit"], *vals, r["uncertainty"],
                                      r["valid_range"], r["provenance"], f"`{r['function']}`"]) + " |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    print(material_table_markdown())
