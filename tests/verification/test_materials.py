"""Verification of mrm.materials against the research tables (docs/research/02_materials.md and its verification log).

Expected numbers are typed in here from 02_materials.md (and the printed sources it quotes), NOT read from
mrm/data/materials.json, so an edit of the JSON cannot make these tests pass by itself. A second group checks that
every reference value stored in materials.json is reproduced by the code (the JSON copy and the code agree).
"""
import hashlib
import json
import pathlib
import warnings

import numpy as np
import pytest

from mrm import materials as M

REPO = pathlib.Path(__file__).resolve().parents[2]
NM = 1e-9
L1310 = 1310 * NM
TEMPS = (273.15, 300.0, 325.0, 350.0, 375.0, 400.0)


def close(a, b, atol=0.0, rtol=0.0):
    return np.allclose(a, b, atol=atol, rtol=rtol)


# ============================================================================ silicon index: printed source tables
def test_li1980_reproduces_li_table1_and_table2():
    """Li 1980 Table 1 (printed) at 1.20 um, 250/293/350/400 K, 1.30 um 293 K and 1.31 um 400 K (02 log #2)."""
    printed = [(1200, 250.0, 3.5084), (1200, 293.0, 3.5167), (1200, 350.0, 3.5284), (1200, 400.0, 3.5393),
               (1300, 293.0, 3.5016), (1310, 400.0, 3.5222)]
    for lam, t, n in printed:
        assert abs(M.n_si_li1980(lam * NM, t) - n) <= 1.5e-4, (lam, t)
    # Table 2: dn/dT at 1.30 um, 400 K printed 2.183e-4; formula 2.185e-4 (02 log #2: <= 0.003e-4)
    assert abs(M.dn_dt_si_per_k(1300 * NM, 400.0) - 2.183e-4) <= 0.005e-4


def test_frey2006_reproduces_table2_and_refuses_above_300k():
    assert abs(M.n_si_frey2006(1300 * NM, 295.0) - 3.50616) < 5e-6
    assert abs(M.n_si_frey2006(1300 * NM, 250.0) - 3.49818) < 5e-6
    with pytest.raises(ValueError):
        M.n_si_frey2006(L1310, 310.0)


def test_02_section_2_2_table():
    """02 §2.2: n (Li), n (Frey level), dn/dT, d2n/dT2 and n_g,Si (Li) at 1310 nm, 273-400 K."""
    n_li = [3.49652, 3.50162, 3.50655, 3.51164, 3.51689, 3.52227]
    n_fr = [3.50060, 3.50570, 3.51064, 3.51573, 3.52097, 3.52635]
    dndt = [1.861e-4, 1.938e-4, 2.006e-4, 2.068e-4, 2.126e-4, 2.179e-4]
    d2 = [2.98e-7, 2.78e-7, 2.59e-7, 2.41e-7, 2.22e-7, 2.04e-7]
    ng_li = [3.6666, 3.6736, 3.6802, 3.6871, 3.6941, 3.7012]
    T = np.array(TEMPS)
    assert close(M.n_si(L1310, T, "li"), n_li, atol=6e-6)
    assert close(M.n_si(L1310, T), n_fr, atol=6e-6)
    assert close(M.dn_dt_si_per_k(L1310, T), dndt, atol=0.0006e-4)
    assert close(M.d2n_dt2_si_per_k2(L1310, T), d2, atol=0.006e-7)
    assert close(M.n_g_si(L1310, T, "li"), ng_li, atol=6e-5)


def test_frey_level_offsets():
    """02 §2.3: +4.6e-3 at 1260 nm, +4.1e-3 (4.085e-3) at 1310 nm, +3.7e-3 at 1360 nm."""
    assert close(M.si_index_offset_frey(np.array([1260, 1310, 1360]) * NM), [4.62e-3, 4.085e-3, 3.67e-3], atol=1e-5)


def test_recommended_level_crosschecks():
    """Recommended model against Franta 2017 (3.50555 at 300 K) within its +/-5e-4, and against Frey Table 4
    dn/dT (1.94e-4 at 1.3 um, 295 K, printed) within the stated +/-0.07e-4."""
    assert abs(M.n_si(L1310, 300.0) - 3.50555) < 5e-4
    assert abs(M.dn_dt_si_per_k(1300 * NM, 295.0) - 1.94e-4) < 0.07e-4
    assert abs(M.dn_dt_si_per_k(L1310, 300.0) - 1.871e-4) <= M.dn_dt_si_sigma_per_k(300.0)  # Franta low corner
    # Lukes 2.04e-4 at 1.327 um / 300 K is the documented high corner (about 1.4 sigma above)
    assert 1.0 < (2.04e-4 - M.dn_dt_si_per_k(1327 * NM, 300.0)) / M.dn_dt_si_sigma_per_k(300.0) < 2.0


def test_group_index_one_model():
    """07 P1-6 / 02 log #6: n_g,Si from the recommended model is 3.690 (Frey Sellmeier 3.689 at 295 K, Franta 3.690),
    inside 3.674-3.690 +/- 0.02, and consistent with n_si by numerical differentiation of the SAME function."""
    ng = M.n_g_si(L1310, 300.0)
    assert abs(ng - 3.690) < 1e-3
    assert abs(ng - 3.689) < 2e-3 and 3.674 - 0.02 <= ng <= 3.690 + 0.02
    # Frey Sellmeier group index at 295 K (independent function) agrees with the recommended model at 295 K
    h = 1e-3 * 1e-6
    ng_frey = M.n_si_frey2006(L1310, 295.0) - L1310 * (M.n_si_frey2006(L1310 + h, 295.0)
                                                      - M.n_si_frey2006(L1310 - h, 295.0)) / (2 * h)
    assert abs(M.n_g_si(L1310, 295.0) - ng_frey) < 2e-3
    assert abs(ng_frey - 3.689) < 1e-3
    # Frey Table 3 prints dn/dlam = -0.14803 /um at 1.3 um, 295 K (n_g 3.699): inside the +/-0.02 band
    assert abs(M.n_g_si(1300 * NM, 295.0) - (3.50616 + 1.3 * 0.14803)) < 0.02


def test_dn_dt_rise_and_slope():
    """+17 % from 273 to 400 K (Li); beta = d2n/dT2 / dn/dT = 1.43e-3 /K at 300 K (07 R1 uses ~1.4e-3), inside the
    1.0e-3 (Franta) to 1.45e-3 (Li) source range."""
    r = M.dn_dt_si_per_k(L1310, 400.0) / M.dn_dt_si_per_k(L1310, 273.15) - 1
    assert abs(r - 0.17) < 0.01
    b = M.si_dn_dt_slope_per_k(L1310, 300.0)
    assert abs(b - 1.434e-3) < 0.005e-3 and 1.0e-3 <= b <= 1.45e-3


def test_cocorullo_cross_check():
    assert abs(M.dn_dt_si_cocorullo1523_per_k(300.0) - 1.855e-4) < 0.001e-4
    assert abs(M.dn_dt_si_cocorullo1523_per_k(375.0) - 2.04e-4) < 0.005e-4
    # scaled to 1310 nm by Li's wavelength ratio: 1.95e-4 / 2.13e-4 (02 §2.3)
    for t, ref in ((300.0, 1.95e-4), (375.0, 2.13e-4)):
        scaled = M.dn_dt_si_cocorullo1523_per_k(t) * M.dn_dt_si_per_k(L1310, t) / M.dn_dt_si_per_k(1523 * NM, t)
        assert abs(scaled - ref) < 0.01e-4


# =============================================================================== derivative convergence (h-refinement)
def _richardson(f, x, h):
    """Central differences at h, h/2, h/4; returns (estimates, Richardson-extrapolated value)."""
    d = [(f(x + s) - f(x - s)) / (2 * s) for s in (h, h / 2, h / 4)]
    return d, (4 * d[2] - d[1]) / 3


@pytest.mark.parametrize("t", [273.15, 300.0, 375.0, 400.0])
def test_analytic_temperature_derivatives_converge(t):
    """The analytic dn/dT and d2n/dT2 agree with refined finite differences of n_si; the FD error falls as h^2."""
    f = lambda T: M.n_si(L1310, T)  # noqa: E731
    d, rich = _richardson(f, t, 2.0)
    exact = M.dn_dt_si_per_k(L1310, t)
    errs = [abs(x - exact) for x in d]
    assert errs[0] > errs[1] > errs[2]
    assert 3.0 < errs[0] / errs[1] < 5.0          # second-order convergence
    assert abs(rich - exact) < 1e-6 * exact
    g = lambda T: M.dn_dt_si_per_k(L1310, T)  # noqa: E731
    _, rich2 = _richardson(g, t, 2.0)
    assert abs(rich2 - M.d2n_dt2_si_per_k2(L1310, t)) < 1e-6 * abs(rich2)


@pytest.mark.parametrize("level", ["frey", "li"])
def test_analytic_wavelength_derivative_converges(level):
    f = lambda lam: M.n_si(lam, 300.0, level)  # noqa: E731
    d, rich = _richardson(f, L1310, 2e-9)
    exact = M.dn_dlam_si_per_m(L1310, 300.0, level)
    assert abs(rich - exact) < 1e-6 * abs(exact)
    g = lambda lam: M.n_sio2(lam, 300.0)  # noqa: E731
    _, rich_ox = _richardson(g, L1310, 2e-9)
    assert abs(rich_ox - M.dn_dlam_sio2_per_m(L1310)) < 1e-6 * abs(rich_ox)


def test_sio2_index_integrates_its_dn_dt_and_strain_integrates_alpha():
    for kind in ("thermal", "pecvd"):
        _, rich = _richardson(lambda T: M.n_sio2(L1310, T, kind), 350.0, 2.0)
        assert abs(rich - M.dn_dt_sio2_per_k(350.0, kind)) < 1e-6 * rich
    _, rich = _richardson(lambda T: M.expansion_strain_si(T, 300.0), 340.0, 2.0)
    assert abs(rich - M.alpha_l_si_per_k(340.0)) < 1e-6 * rich
    assert M.expansion_strain_si(300.0, 300.0) == 0.0


# ================================================================================================ ring ceiling
def test_ring_thermal_shift_ceiling_table():
    """02 §3 (pure Li, n_eff/n_g = 0.6): 68.3/71.1/75.8/77.8/79.7 pm/K; strict all-Si 72.3/79.2/81.1 (02 log #7)."""
    T = np.array([273.15, 300.0, 350.0, 375.0, 400.0])
    tot, to, ex = M.dlambda_dt_ceiling_pm_per_k(L1310, T, 0.6, level="li")
    assert close(tot, [68.3, 71.1, 75.8, 77.8, 79.7], atol=0.051)
    assert close(to, [66.5, 69.1, 73.5, 75.4, 77.1], atol=0.051)
    strict = M.dlambda_dt_ceiling_pm_per_k(L1310, np.array([300.0, 375.0, 400.0]), "all_si", level="li")[0]
    assert close(strict, [72.3, 79.2, 81.1], atol=0.051)
    # one-model version (recommended n_g 3.690) is about 0.3 pm/K lower; 86 pm/K stays out of reach at 400 K
    rec = M.dlambda_dt_ceiling_pm_per_k(L1310, 300.0)[0]
    assert -0.45 < rec - 71.1 < -0.15
    assert M.dlambda_dt_ceiling_pm_per_k(L1310, 400.0, "all_si")[0] < 85.86 / 1.05


# ===================================================================================================== oxides
def test_sio2_index():
    assert abs(M.n_sio2_malitson(L1310) - 1.44680) < 5e-6
    # Malitson x n_air against the absolute CHARMS index at 295 K (Leviton & Frey Table 4, printed), 1e-5
    assert abs(M.n_sio2(1200 * NM, 295.0, "fused") - 1.44845) < 2e-5
    assert abs(M.n_sio2(1500 * NM, 295.0, "fused") - 1.44502) < 2e-5
    assert abs(M.n_sio2(L1310, 295.0, "thermal") - 1.447) < 1e-3       # 02: use 1.447 for the BOX
    assert abs(M.n_sio2(L1310, 295.0, "pecvd") - 1.45) < 1e-12           # 02: 1.45 +/- 0.01 for PECVD/BEOL
    assert M.n_sio2(L1310, 295.0, "beol") == M.n_sio2(L1310, 295.0, "pecvd")
    ng_malitson = M.n_sio2_malitson(L1310) - L1310 * M.dn_dlam_sio2_per_m(L1310) / 1.00027
    assert abs(ng_malitson - 1.4616) < 1e-4


def test_sio2_dn_dt():
    assert close(M.dn_dt_sio2_per_k(np.array([295.0, 300.0, 375.0])), [8.45e-6, 8.57e-6, 1.04e-5], atol=0.005e-5)
    assert abs(M.dn_dt_sio2_per_k(295.0, "pecvd") - 0.95e-5) < 1e-12
    # oxide (dn/dT)/n_g is about 9x smaller than silicon's (02 §3)
    ratio = (M.dn_dt_si_per_k(L1310, 300.0) / M.n_g_si(L1310, 300.0)) / (M.dn_dt_sio2_per_k(300.0)
                                                                         / M.n_g_sio2(L1310, 300.0))
    assert 8.0 < ratio < 10.0


def test_si3n4():
    assert abs(M.n_si3n4_luke(L1310) - 2.0031) < 1e-4
    assert abs(M.n_g_si3n4(L1310) - 2.042) < 1e-3
    assert M.DN_DT_SI3N4_PER_K == 2.45e-5


# ================================================================================================== expansion
def test_alpha_si_okada_and_souza_crosscheck():
    T = np.array([273.15, 300.0, 375.0, 400.0])
    assert close(M.alpha_l_si_per_k(T), [2.327e-6, 2.568e-6, 3.082e-6, 3.212e-6], atol=0.002e-6)
    # Souza 2018 (06 §2.6, corrected): 2.5e-6 at 300 K within Okada's +/-2e-7; net rise ~+27 %/100 K (ours +25 %)
    assert abs(M.alpha_l_si_per_k(300.0) - 2.5e-6) < 2e-7
    assert abs(M.alpha_l_si_per_k(400.0) / M.alpha_l_si_per_k(300.0) - 1.27) < 0.04
    # expansion term of the ring shift: lam (n_eff/n_g) alpha = 2.0 pm/K (300 K), 2.4 pm/K (375 K) at 0.6 (02 §6)
    assert close(L1310 * 0.6 * M.alpha_l_si_per_k(np.array([300.0, 375.0])) * 1e12, [2.0, 2.4], atol=0.05)


def test_alpha_sio2_reproduces_both_corning_means():
    """Instantaneous linear alpha(T) whose averages equal the printed 0.52 (5-35 C) and 0.57 ppm/K (0-200 C)."""
    for lo, hi, mean in ((278.15, 308.15, 0.52e-6), (273.15, 473.15, 0.57e-6)):
        T = np.linspace(lo, hi, 2001)
        assert abs(np.trapezoid(M.alpha_l_sio2_per_k(T), T) / (hi - lo) - mean) < 1e-10


# ============================================================================================== thermal transport
def test_k_si_bulk():
    T = np.array([273.15, 300.0, 375.0, 400.0])
    assert close(M.k_si_bulk_w_per_m_k(T), [167.2, 148.0, 110.7, 101.8], atol=0.06)
    # CRC/TPRC recalled set 168 / 148 / 98.9 within about 3 %
    assert close(M.k_si_bulk_w_per_m_k(np.array([273.15, 300.0, 400.0])), [168.0, 148.0, 98.9], rtol=0.035)


def test_k_si_film_corrected_fit():
    """02 §5.2 table (corrected, d0 = 170 nm) plus the 160 and 200 nm films of the two stack hypotheses."""
    table = {220.0: (87.2, 83.5, 72.5, 69.1), 200.0: (None, 80.0, 70.1, None), 250.0: (93.0, 88.1, 75.7, 72.0),
             150.0: (71.0, 69.0, 63.0, 60.0)}
    temps = (273.15, 300.0, 375.0, 400.0)
    for d, vals in table.items():
        for t, v in zip(temps, vals):
            if v is not None:
                assert abs(M.k_si_film_w_per_m_k(t, d) - v) < 0.6, (d, t)
    assert abs(M.k_si_film_w_per_m_k(300.0, 220.0) - 84.0) < 0.6                     # headline 84 W/mK
    assert abs(M.k_si_film_w_per_m_k(375.0, 220.0) / M.k_si_film_w_per_m_k(300.0, 220.0) - 0.87) < 0.005  # -13 %
    assert abs(M.k_si_film_w_per_m_k(300.0, 160.0) - 71.76) < 0.01
    assert abs(M.k_si_film_w_per_m_k(375.0, 160.0) - 64.23) < 0.01
    assert abs(M.k_si_film_w_per_m_k(300.0, 200.0) - 80.0) < 0.01
    # anchor: Aubain & Bandaru 2010, 260 nm, 90 +/- 10 W/mK; d0 range 125-221 nm maps to 73-95 W/mK at 220 nm
    assert abs(M.k_si_film_w_per_m_k(300.0, 260.0) - 90.0) < 1.0
    assert abs(M.k_si_film_w_per_m_k(300.0, 220.0, d0_nm=125.0) - 95.0) < 1.0
    assert abs(M.k_si_film_w_per_m_k(300.0, 220.0, d0_nm=221.0) - 73.0) < 1.0
    # film is weaker in T than bulk (-13 % vs -25 %)
    assert abs(M.k_si_bulk_w_per_m_k(375.0) / M.k_si_bulk_w_per_m_k(300.0) - 0.75) < 0.005
    with pytest.warns(M.MaterialRangeWarning):
        M.k_si_film_w_per_m_k(300.0, 100.0)    # outside the anchored 150-300 nm range


def test_k_oxides_and_metals():
    assert close(M.k_sio2_w_per_m_k(np.array([300.0, 375.0])), [1.38, 1.485], atol=0.001)
    assert abs(M.k_sio2_w_per_m_k(300.0, "pecvd") - 0.9 * 1.38) < 1e-12
    assert M.k_low_k_w_per_m_k(300.0, "dense") == 0.7 and M.k_low_k_w_per_m_k(300.0, "porous30") == 0.1
    assert M.k_metal_w_per_m_k("cu_beol", 300.0, "low") == 250.0
    assert 250.0 <= M.k_metal_w_per_m_k("cu_beol", 300.0) <= 380.0
    # Wiedemann-Franz check of the Cu BEOL range: L0 T / rho for rho 2.0-2.9 uOhm cm spans 250-366 W/mK
    L0 = M.data()["k_metals"]["coefficients"]["L0_w_ohm_per_k2"]
    assert 250.0 <= L0 * 300.0 / 2.25e-8 <= 380.0
    assert M.thermal_boundary_resistance_m2_k_per_w("si_sio2") == 2e-8
    assert M.thermal_boundary_resistance_m2_k_per_w("metal_dielectric") == 2e-7


def test_heat_capacity_and_rho_c():
    T = np.array([273.15, 300.0, 375.0, 400.0])
    assert close(M.cp_si_j_per_kg_k(T), [677.1, 711.0, 780.7, 797.9], atol=0.06)
    assert close(M.cp_sio2_j_per_kg_k(T), [659.7, 709.0, 825.3, 858.0], atol=0.06)
    assert close(M.rho_c_j_per_m3_k("si", T), [1.577e6, 1.656e6, 1.818e6, 1.858e6], atol=0.0006e6)
    assert close(M.rho_c_j_per_m3_k("sio2", T), [1.451e6, 1.560e6, 1.816e6, 1.888e6], atol=0.0006e6)
    # NIST-JANAF: Cp(Si) = 0.7121 and 0.7884 J/(g K) at 298.15 and 400 K; model within 0.5 % and 1.2 %
    assert abs(M.cp_si_j_per_kg_k(298.15) / 712.1 - 1) < 0.005
    assert abs(M.cp_si_j_per_kg_k(400.0) / 788.4 - 1) < 0.0125
    assert M.rho_c_j_per_m3_k("cu", 300.0) == 3.45e6 and M.rho_c_j_per_m3_k("w_film", 350.0) == 2.55e6
    assert M.rho_c_j_per_m3_k("tin", 300.0) == 3.1e6
    with pytest.raises(ValueError):
        M.rho_c_j_per_m3_k("si3n4", 300.0)


def test_thermal_properties_registry_and_diffusivity():
    for m in M.THERMAL_MATERIALS:
        p = M.thermal_properties(m, np.array([300.0, 375.0]), film_nm=200.0)
        assert np.all(np.asarray(p["k_w_per_m_k"]) > 0) and np.all(np.asarray(p["rho_c_j_per_m3_k"]) > 0)
    assert abs(M.thermal_properties("si_bulk", 300.0)["diffusivity_m2_per_s"] - 8.9e-5) < 0.06e-5
    assert abs(M.thermal_properties("sio2_thermal", 300.0)["diffusivity_m2_per_s"] - 8.8e-7) < 0.06e-7
    with pytest.raises(ValueError):
        M.thermal_properties("si_film", 300.0)


# ================================================================================================ free carriers
def test_plasma_dispersion_side_by_side_table():
    """02 §7 side-by-side table at N_e = N_h."""
    N = np.array([1e17, 1e18, 5e18])
    dn, da = M.plasma_dispersion(N, N)
    assert close(dn, [-2.52e-4, -1.92e-3, -8.10e-3], atol=[0.0051e-4, 0.0051e-3, 0.0051e-3])
    assert close(np.asarray(da) / 100, [0.60, 8.69, 56.9], atol=[0.0051, 0.0051, 0.051])
    dn_sb, da_sb = M.plasma_dispersion(N, N, "soref1987")
    assert close(dn_sb, [-3.01e-4, -2.13e-3, -8.56e-3], atol=[0.0051e-4, 0.0051e-3, 0.0051e-3])
    assert close(np.asarray(da_sb) / 100, [1.0, 10.0, 50.0], rtol=1e-12)
    # headline 5: NSM 10 % smaller |dn| and 13 % smaller dalpha than SB at 1e18
    assert abs(dn[1] / dn_sb[1] - 0.90) < 0.01 and abs(da[1] / da_sb[1] - 0.87) < 0.01


def test_plasma_small_signal_table():
    """02 §7 effective small-signal coefficients (NSM 1.3 um) at 1e16, 1e17, 1e18 cm^-3 (cm^2, cm^3)."""
    ref = {1e16: (2.0e-18, 3.0e-18, -5.5e-22, -2.4e-21), 1e17: (3.3e-18, 3.6e-18, -5.7e-22, -1.6e-21),
           1e18: (5.7e-18, 4.4e-18, -5.9e-22, -1.1e-21)}
    for n, (se, sh, dne, dnh) in ref.items():
        d = M.plasma_dispersion_derivatives(n, n)
        assert close([d["sigma_e_cm2"], d["sigma_h_cm2"], d["dn_dne_cm3"], d["dn_dnh_cm3"]], [se, sh, dne, dnh],
                     rtol=0.05), n
    sb = M.plasma_dispersion_derivatives(1e18, 1e18, "soref1987")
    assert close([sb["sigma_e_cm2"], sb["sigma_h_cm2"]], [6e-18, 4e-18], rtol=1e-12)


def test_plasma_depletion_sign_and_guards():
    """Removing carriers (reverse bias) raises the index and lowers the loss; negative densities are refused."""
    dn, da = M.plasma_dispersion_change(1e18, 1e18, 5e17, 5e17)
    assert dn > 0 and da < 0
    with pytest.raises(ValueError):
        M.plasma_dispersion(-1e17, 0.0)
    with pytest.raises(ValueError):
        M.plasma_dispersion(1e18, 1e18, "soref1987", band_um=1.55)
    with pytest.warns(M.MaterialRangeWarning):
        M.plasma_dispersion(1e15, 1e15)
    assert M.plasma_dispersion(0.0, 0.0) == (0.0, 0.0)
    assert M.plasma_temperature_factor(300.0) == 1.0
    assert abs(M.plasma_temperature_factor(400.0, "dalpha", "high") - 1.3) < 1e-12
    assert abs(M.plasma_temperature_factor(400.0, "dn", "low") - 0.95) < 1e-12


def test_plasma_against_tidy3d_implementation():
    """Independent implementation: Tidy3D's NedeljkovicSorefMashanovich at exactly 1.3 um. Tidy3D samples a 200-point
    log grid in N, so agreement is limited to its interpolation error (~0.3 %)."""
    td = pytest.importorskip("tidy3d")
    from tidy3d.components.parameter_perturbation import NedeljkovicSorefMashanovich
    model = NedeljkovicSorefMashanovich(ref_freq=td.C_0 / 1.3)
    for n in (1e17, 1e18, 5e18):
        ne, nh = np.array([n]), np.array([0.7 * n])   # 0.7e17 is below the fit range: same formula, no warning
        dn_t = float(np.ravel(model.delta_n().charge.sample(electron_density=ne, hole_density=nh))[0])
        dk_t = float(np.ravel(model.delta_k().charge.sample(electron_density=ne, hole_density=nh))[0])
        dn, da = M.plasma_dispersion(n, 0.7 * n, warn=False)
        assert abs(dn_t / dn - 1) < 0.01
        assert abs(4 * np.pi * dk_t / 1.3e-6 / da - 1) < 0.01


def test_indices_against_tidy3d_material_library():
    """Independent transcriptions in Tidy3D's library: Li's room-temperature Sellmeier (Li1993_293K) within Li's
    +/-2e-3 of the Li T-formula; Green 2008 (lossless) on the Li level (2e-5); Palik SiO2 vs Malitson (3e-5); Luke
    Si3N4 identical. Our recommended Si level sits 4.1e-3 above these Li-level tables by construction (Frey/Franta)."""
    td = pytest.importorskip("tidy3d")
    lib = td.material_library

    def n_lib(mat, var, lam_um):
        return float(np.real(lib[mat][var].nk_model(td.C_0 / lam_um)[0]))

    for lam in (1.26, 1.31, 1.36):
        assert abs(n_lib("cSi", "Li1993_293K", lam) - M.n_si_li1980(lam * 1e-6, 293.0)) < 2e-3
    assert abs(n_lib("cSi", "Green2008_Lossless", 1.31) - M.n_si(L1310, 300.0, "li")) < 1e-4
    assert abs(M.n_si(L1310, 300.0) - n_lib("cSi", "Green2008_Lossless", 1.31) - 4.1e-3) < 2e-4
    assert abs(n_lib("SiO2", "Palik_NoLoss", 1.31) - M.n_sio2_malitson(L1310)) < 1e-4
    assert abs(n_lib("Si3N4", "Luke2015Sellmeier", 1.31) - M.n_si3n4_luke(L1310)) < 1e-6


def test_nonlinear_constants():
    assert M.tpa_beta_m_per_w() == 0.9e-11
    assert (M.tpa_beta_m_per_w("low"), M.tpa_beta_m_per_w("high")) == (0.5e-11, 2.0e-11)
    assert M.kerr_n2_m2_per_w() == 4e-18
    assert (M.kerr_n2_m2_per_w("low"), M.kerr_n2_m2_per_w("high")) == (3e-18, 6e-18)


# =============================================================================================== junction physics
def test_bandgap_ni_vbi():
    T = np.array([273.15, 300.0, 375.0, 400.0])
    assert close(M.eg_si_ev(T), [1.1312, 1.1245, 1.1042, 1.0969], atol=6e-5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", M.MaterialRangeWarning)
        ni = M.ni_si_per_cm3(np.array([273.15, 300.0, 325.0, 350.0, 375.0, 400.0]))
    assert close(ni, [8.43e8, 9.70e9, 6.67e10, 3.53e11, 1.51e12, 5.47e12], rtol=0.005)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", M.MaterialRangeWarning)
        assert close(M.vbi_v(1e18, 1e18, np.array([273.15, 300.0, 375.0, 400.0])), [0.9836, 0.954, 0.8661, 0.8353],
                     atol=6e-5)
        assert abs(M.vbi_v(5e17, 5e17, 300.0) - 0.9182) < 6e-5
        assert abs(M.dvbi_dt_v_per_k(1e18, 1e18, 300.0) + 1.122e-3) < 0.002e-3
        assert abs(M.dvbi_dt_v_per_k(1e18, 1e18, 375.0) + 1.219e-3) < 0.002e-3
        _, rich = _richardson(lambda t: M.vbi_v(1e18, 1e18, t), 330.0, 2.0)
        assert abs(rich - M.dvbi_dt_v_per_k(1e18, 1e18, 330.0)) < 1e-6 * abs(rich)


# ======================================================================================================== heaters
def test_doped_si_resistivity_and_tcr_table():
    """02 §9 derived table (Arora / Masetti, full ionization) and the printed 23 C reference table."""
    rows = [  # N, type, rho_arora, rho_masetti (ohm cm), TCR300, TCR375, rho375/rho300
        (1e18, "n", 2.38e-2, 2.25e-2, 1.26e-3, 1.52e-3, 1.11), (1e18, "p", 4.36e-2, 3.89e-2, 1.62e-3, 1.82e-3, 1.14),
        (1e19, "n", 5.47e-3, 5.44e-3, 1.25e-3, 1.06e-3, 1.09), (1e19, "p", 9.08e-3, 8.53e-3, 1.31e-3, 1.12e-3, 1.09),
        (1e20, "n", 6.82e-4, 8.52e-4, 1.75e-3, 1.39e-3, 1.12), (1e20, "p", 1.11e-3, 1.25e-3, 1.76e-3, 1.40e-3, 1.12)]
    for N, c, ra, rm, t300, t375, ratio in rows:
        assert abs(M.resistivity_doped_si_ohm_m(N, 300.0, c) * 100 / ra - 1) < 0.003, (N, c)
        assert abs(M.resistivity_doped_si_ohm_m(N, 300.0, c, "masetti") * 100 / rm - 1) < 0.003, (N, c)
        assert abs(M.tcr_doped_si_per_k(N, 300.0, c) - t300) < 0.006e-3, (N, c)
        assert abs(M.tcr_doped_si_per_k(N, 375.0, c) - t375) < 0.006e-3, (N, c)
        r = M.resistivity_doped_si_ohm_m(N, 375.0, c) / M.resistivity_doped_si_ohm_m(N, 300.0, c)
        assert abs(r - ratio) < 0.006
    # printed reference table at 23 C: Masetti within 9 % everywhere; Arora 13 % low at 1e20 n
    table = {(1e18, "n"): 2.3e-2, (1e19, "n"): 5.5e-3, (1e20, "n"): 7.8e-4,
             (1e18, "p"): 4.5e-2, (1e19, "p"): 8.5e-3, (1e20, "p"): 1.2e-3}
    for (N, c), rho in table.items():
        assert abs(M.resistivity_doped_si_ohm_m(N, 296.15, c, "masetti") * 100 / rho - 1) < 0.14
    assert abs(M.resistivity_doped_si_ohm_m(1e20, 300.0, "n") * 100 / 7.8e-4 - 1 + 0.13) < 0.01


def test_heater_tcr_families():
    """07 P1-11: doped Si 1.0-1.8e-3, TiN about 0.5e-3, silicides and W 2-4.5e-3 /K."""
    lo, hi = M.heater_tcr_per_k("doped_si", "low"), M.heater_tcr_per_k("doped_si", "high")
    assert (lo, hi) == (1.0e-3, 1.8e-3) and lo <= M.heater_tcr_per_k("doped_si") <= hi
    assert abs(M.heater_tcr_per_k("tin") - 0.5e-3) < 0.05e-3
    for fam in ("silicide", "w"):
        assert 2.0e-3 <= M.heater_tcr_per_k(fam) <= 4.6e-3 and M.heater_tcr_per_k(fam, "low") >= 2.0e-3
    assert M.heater_tcr_per_k("inring_noundercut") == M.heater_tcr_per_k("doped_si")
    assert M.heater_tcr_per_k("silicide_undercut") == M.heater_tcr_per_k("silicide")
    # the Arora-derived TCR of heater-doped Si (1e19-1e20, n and p, 300-375 K) lies inside the doped_si family range
    for N in (1e19, 1e20):
        for c in ("n", "p"):
            for t in (300.0, 375.0):
                assert lo <= M.tcr_doped_si_per_k(N, t, c) <= hi
    # the textbook's G25 fit (1e-3 /K) sits at the low end of doped Si
    assert M.data()["heater_tcr_family"]["textbook_g25_fit"]["value"] == lo
    assert abs(M.heater_resistance_ratio(375.0, 300.0, 1.4e-3) - 1.105) < 1e-12
    with pytest.raises(ValueError):
        M.heater_tcr_per_k("nichrome")


# ======================================================================================================= Ge PD
def test_ge_photodiode():
    assert abs(M.ge_pd_responsivity_limit_a_per_w(L1310) - 1.0566) < 1e-4
    assert M.ge_pd_responsivity_limit_a_per_w(L1310) > 1.04          # G25 top end is 98 % of the limit
    assert abs(M.ge_pd_dark_current_a(1.0, 358.15) - 11.2) < 0.05     # 85 C vs 25 C
    assert abs(M.ge_pd_dark_current_a(1.0, 378.15) - 21.0) < 0.1      # 105 C vs 25 C
    # Colace 2008: x1.6 per 10 C near room temperature with Ea 0.37 eV
    assert abs(M.ge_pd_dark_current_a(1.0, 308.15) - 1.6) < 0.01
    # Koester 2006: x10 from RT to 85 C sits inside the Ea range 0.35-0.38 eV
    lo = M.ge_pd_dark_current_a(1.0, 358.15, ea_ev=M.ge_pd_dark_ea_ev("low"))
    hi = M.ge_pd_dark_current_a(1.0, 358.15, ea_ev=M.ge_pd_dark_ea_ev("high"))
    assert lo < 10.0 < hi
    assert M.ge_pd_dark_ea_ev("stress") == 0.5
    assert M.GE_PD_DARK_I0_A_RANGE == (1e-9, 30e-9)
    assert M.ge_pd_responsivity_a_per_w(0.9, 378.15) == 0.9


# ============================================================================ JSON copy, vectorization, docs table
def test_materials_json_meta_and_research_hash():
    d = M.data()
    assert d["_meta"]["schema"] == M.MODEL_ID
    for key, entry in d.items():
        if key.startswith("_"):
            continue
        for field in ("quantity", "unit", "model", "coefficients", "source", "provenance"):
            assert field in entry, (key, field)
    # the cleaned copy records which research file it came from; fail if the research JSON changed since
    src = REPO / "docs/research/02_materials.json"
    if src.exists():
        assert hashlib.sha256(src.read_bytes()).hexdigest() == d["_meta"]["derived_from"][
            "docs/research/02_materials.json"], "02_materials.json changed: re-sync mrm/data/materials.json"
    assert len(M.data_sha256()) == 64
    assert M.materials_provenance()["sha256"] == M.data_sha256()


_DISPATCH = {
    "si_index_li1980": lambda a: M.n_si(a["lam_nm"] * NM, a["t_k"], "li"),
    "si_index_recommended": lambda a: M.n_si(a["lam_nm"] * NM, a["t_k"]),
    "si_dn_dt": lambda a: M.dn_dt_si_per_k(a["lam_nm"] * NM, a["t_k"]),
    "si_d2n_dt2": lambda a: M.d2n_dt2_si_per_k2(a["lam_nm"] * NM, a["t_k"]),
    "si_dn_dt_slope": lambda a: M.si_dn_dt_slope_per_k(a["lam_nm"] * NM, a["t_k"]),
    "si_dn_dt_cocorullo1523": lambda a: M.dn_dt_si_cocorullo1523_per_k(a["t_k"]),
    "si_group_index": lambda a: M.n_g_si(a["lam_nm"] * NM, a["t_k"], a["level"]),
    "si_absorption_band": lambda a: M.alpha_band_si_per_m(a["lam_nm"] * NM) / 100,
    "ring_thermal_shift_ceiling": lambda a: M.dlambda_dt_ceiling_pm_per_k(a["lam_nm"] * NM, a["t_k"],
                                                                          a["neff_over_ng"], a["level"])[0],
    "sio2_index_malitson": lambda a: M.n_sio2_malitson(a["lam_nm"] * NM),
    "sio2_index_films": lambda a: M.n_sio2(a["lam_nm"] * NM, a["t_k"], a["kind"]),
    "sio2_dn_dt": lambda a: M.dn_dt_sio2_per_k(a["t_k"], a["kind"]),
    "si3n4_index_luke": lambda a: M.n_si3n4_luke(a["lam_nm"] * NM),
    "si3n4_dn_dt": lambda a: M.DN_DT_SI3N4_PER_K,
    "si_alpha_l": lambda a: M.alpha_l_si_per_k(a["t_k"]),
    "sio2_alpha_l": lambda a: M.alpha_l_sio2_per_k(a["t_k"]),
    "thermo_mechanical": lambda a: L1310 * 0.6 * M.alpha_l_si_per_k(a["t_k"]) * 1e12,
    "k_si_bulk": lambda a: M.k_si_bulk_w_per_m_k(a["t_k"]),
    "k_si_film": lambda a: M.k_si_film_w_per_m_k(a["t_k"], a["film_nm"]),
    "k_sio2": lambda a: M.k_sio2_w_per_m_k(a["t_k"], a["kind"]),
    "k_low_k": lambda a: M.k_low_k_w_per_m_k(300.0, a["kind"]),
    "k_metals": lambda a: M.k_metal_w_per_m_k(a["metal"], 300.0),
    "thermal_boundary_resistance": lambda a: M.thermal_boundary_resistance_m2_k_per_w(a["interface"]),
    "cp_si": lambda a: M.cp_si_j_per_kg_k(a["t_k"]),
    "cp_sio2": lambda a: M.cp_sio2_j_per_kg_k(a["t_k"]),
    "rho_c": lambda a: M.rho_c_j_per_m3_k(a["material"], a["t_k"]),
    "thermal_diffusivity": lambda a: M.thermal_properties(a["material"], a["t_k"])["diffusivity_m2_per_s"],
    "tpa_beta": lambda a: M.tpa_beta_m_per_w(),
    "kerr_n2": lambda a: M.kerr_n2_m2_per_w(),
    "doped_si_resistivity": lambda a: M.resistivity_doped_si_ohm_m(a["n_per_cm3"], 300.0, a["carrier"],
                                                                   a["model"]) * 100,
    "doped_si_tcr": lambda a: M.tcr_doped_si_per_k(a["n_per_cm3"], a["t_k"], a["carrier"]),
    "heater_tcr_family": lambda a: M.heater_tcr_per_k(a["family"]),
    "ge_pd_responsivity": lambda a: M.ge_pd_responsivity_limit_a_per_w(a["lam_nm"] * NM, a["eta"]),
    "ge_pd_dark_current": lambda a: M.ge_pd_dark_current_a(1.0, a["t_k"]),
    "si_bandgap": lambda a: M.eg_si_ev(a["t_k"]),
    "si_intrinsic_density": lambda a: M.ni_si_per_cm3(a["t_k"]),
    "pn_builtin_voltage": lambda a: M.vbi_v(a["na_per_cm3"], a["nd_per_cm3"], a["t_k"]),
}


def _digits_tol(v):
    """Half a unit in the last printed significant digit (values were printed to 3-6 significant digits)."""
    s = f"{abs(v):.6g}"
    mant = s.split("e")[0]
    exp = int(s.split("e")[1]) if "e" in s else 0
    dec = len(mant.split(".")[1]) if "." in mant else 0
    return 0.5 * 10 ** (exp - dec) * 1.0001 + 1e-15


@pytest.mark.filterwarnings("ignore::mrm.materials.MaterialRangeWarning")
def test_every_json_reference_value_is_reproduced():
    """Every scalar reference value stored in materials.json is reproduced by the code to its printed precision
    (the 160 nm film and the derived SiO2 alpha were added with 4 digits)."""
    d = M.data()
    n_checked = 0
    for key, fn in _DISPATCH.items():
        for ref in d[key]["reference_values"]:
            v = float(fn(ref["args"]))
            # 02 computed the TCR with a 1 K central difference and rounded to 3 digits (1.115e-3 -> 1.12e-3);
            # the analytic-limit value differs by up to 0.5 %
            rel = {"doped_si_tcr": 6e-3, "ge_pd_dark_current": 2e-3, "doped_si_resistivity": 2e-3,
                   "thermal_diffusivity": 6e-3}.get(key, 0.0)
            tol = max(_digits_tol(ref["value"]), rel * abs(ref["value"]))
            assert abs(v - ref["value"]) <= tol, (key, ref, v)
            n_checked += 1
    for key in ("plasma_nsm2011", "plasma_soref1987"):
        model = key.split("_")[1]
        for ref in d[key]["reference_values"]:
            dn, da = M.plasma_dispersion(ref["args"]["ne_per_cm3"], ref["args"]["nh_per_cm3"], model)
            assert abs(dn - ref["value"]["dn"]) <= _digits_tol(ref["value"]["dn"])
            assert abs(da / 100 - ref["value"]["dalpha_per_cm"]) <= _digits_tol(ref["value"]["dalpha_per_cm"])
            n_checked += 1
    for ref in d["plasma_small_signal"]["reference_values"]:
        n = ref["args"]["n_per_cm3"]
        got = M.plasma_dispersion_derivatives(n, n)
        for k, v in ref["value"].items():
            assert abs(got[k] / v - 1) < 0.01, (k, n)
        n_checked += 1
    assert n_checked > 120


def test_vectorized_broadcasting():
    lam = np.linspace(1260, 1360, 5)[:, None] * NM
    T = np.array([280.0, 300.0, 350.0, 400.0])[None, :]
    n = M.n_si(lam, T)
    assert n.shape == (5, 4)
    for i in range(5):
        for j in range(4):
            assert n[i, j] == M.n_si(float(lam[i, 0]), float(T[0, j]))
    assert M.dn_dt_si_per_k(lam, T).shape == (5, 4) and M.n_g_si(lam, T).shape == (5, 4)
    assert isinstance(M.n_si(L1310, 300.0), float)
    assert M.k_si_film_w_per_m_k(T, np.array([[160.0], [200.0]])).shape == (2, 4)
    with pytest.raises(ValueError):
        M.n_si(1310.0, 300.0)       # nm passed instead of m


def test_range_warnings():
    with pytest.warns(M.MaterialRangeWarning):
        M.n_si(L1310, 800.0)
    with pytest.warns(M.MaterialRangeWarning):
        M.ni_si_per_cm3(420.0)      # Misiakos fit 78-340 K; the 02 tables extrapolate it to 400 K
    with warnings.catch_warnings():
        warnings.simplefilter("error", M.MaterialRangeWarning)
        M.n_si(np.array([1260, 1310, 1360]) * NM, np.array([273.15, 300.0, 400.0]))
        M.k_si_film_w_per_m_k(np.array([273.0, 400.0]), 160.0)


def test_material_table_for_docs():
    rows = M.material_table()
    assert len(rows) > 40
    for r in rows:
        assert set(r) >= {"property", "symbol", "unit", "values", "function", "uncertainty", "valid_range", "source",
                          "provenance"}
        assert len(r["values"]) == 2 and all(np.isfinite(r["values"])), r["property"]
        assert r["source"] and r["provenance"], r["property"]
    md = M.material_table_markdown()
    assert md.count("\n|") > 40 and "n_g,Si" in md


def test_material_uncertainties_summary():
    u = M.material_uncertainties()
    assert u["k_si_film"]["sigma"] == 0.15 and u["si_n_g"]["sigma"] == 0.02
    assert abs(u["si_index_level"]["delta_at_1310nm"] + 4.085e-3) < 1e-5
    assert u["ge_dark_ea_ev"]["stress"] == 0.5 and u["plasma_model"]["nominal"] == "nsm2011"
    for name, spec in u.items():
        if spec["kind"] == "range":
            assert spec["lo"] <= spec.get("nominal", spec["lo"]) <= spec["hi"], name
        if spec["kind"] in ("abs", "rel"):
            assert all(v > 0 for k, v in spec.items() if k.startswith("sigma")), name
