"""params review fixes (2026-10-16), one test per confirmed finding (docs/CHANGELOG.md, same date).

- params-1, params-11: in g26 one geometry error moves the couplers with the resonance (07 P1-16): the coupling
  Jacobian is wired, a width error closes both gaps, the gaps have one local sigma, and a device's per-pass coupling
  at its line is the nominal device's moved by exactly J dg (no double count of the coupler dispersion);
- params-2: with sampled geometry the mask radius and the order are the nominal device's, and the FSR follows n_g;
- params-10: the bend radiation is exponential in the geometry and never negative;
- params-4: Z_heater->ring is a stack property, the same on every line;
- params-7: the line wavelengths carry the channel plan's provenance;
- params-8: the junction exponent's declared range is the span the analytic junction table accepts;
- params-9: a world's value (interval end or corner) and every preset alternative lie in the leaf's range;
- params-13: the R0 range holds the calibrated droop worlds;
- params-3: the 200 nm film world is the rib200 artefact.
"""
from __future__ import annotations

import copy
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from mrm.params import derive, load, realize
from mrm.params import relations as rel
from mrm.params.config import ConfigError, yaml_load
from mrm.params.derive import effective_provenance
from mrm.params.table import explain
from mrm.units import C0

ROOT = Path(__file__).resolve().parents[2]
EXTRACTED = ROOT / "mrm" / "data" / "extracted"
MINI = ROOT / "tests" / "data" / "mini.yaml"
PRESETS = sorted(p.stem for p in (ROOT / "mrm" / "params" / "presets").glob("*.yaml"))
B = 400
DIMS = {"width_nm": "width", "film_nm": "thickness", "slab_nm": "slab", "gap_through_nm": "gap_through",
        "gap_drop_nm": "gap_drop"}


def _load(name, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load(name, **kw)


@pytest.fixture(scope="module")
def g26_dies():
    cfg = _load("g26")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = realize(cfg, B, 3, uncertainty="nominal", variation="sample", thermal_hook=None)
        nom = realize(cfg, 1, 3, uncertainty="nominal", variation="off", thermal_hook=None)
    return cfg, p, nom


def _at_line_ln_k2(o, which):
    """ln K^2 of a coupler at the ring's line, as mrm.plant.phase.coupling_row evaluates it."""
    k2 = getattr(o, f"kappa2_{which}")
    d = getattr(o, f"dln_kappa2_{which}_dlambda_per_m")
    return np.log(k2) + d * (o.lambda_line_m - o.lambda_res_cold_m)


def _effective_dims(cfg, p):
    """The geometry errors the Jacobians see: the drawn ones, with a width error closing both gaps."""
    dg = {k: np.broadcast_to(v, (B, p.n_rings)) for k, v in p.diagnostics["geometry_draws_nm"].items()}
    if cfg.geometry.get("width_closes_gaps"):
        for gd in ("gap_through", "gap_drop"):
            dg[gd] = dg.get(gd, 0.0) - dg["width"]
    return dg


# ------------------------------------------------------------------------------------------------ params-1, -11

def test_g26_couplers_move_with_the_geometry_by_the_coupling_jacobian(g26_dies):
    """params-1 (07 P1-16): g26 wires coupling_rib160's Jacobian, and every device's per-pass coupling at its own line
    differs from the nominal device's by exactly sum_k J_k dg_k (litho-corrected: total = d/dw - d/dgap). Before the
    fix the coupling at the cold resonance was the anchor's for every device, so at the line it moved by -D d lambda
    only (about half the geometry-correlated coupling, in the wrong proportion)."""
    cfg, p, nom = g26_dies
    assert "ring.geometry.jacobian_coupling" in cfg.geometry.get("jacobians", [])
    assert cfg.geometry.get("width_closes_gaps") is True
    jac = cfg.get("ring.geometry.jacobian_coupling")
    art = json.loads((EXTRACTED / "coupling_rib160.json").read_text())["jacobian"]
    assert jac["of"] == art["of"] and jac["d"] == art["d"]
    dg = _effective_dims(cfg, p)
    for which in ("through", "drop"):
        coeffs = jac["of"][f"ln_kappa2_{which}_per_nm"]
        pred = sum(c * dg[DIMS[d]] for c, d in zip(coeffs, jac["d"]) if DIMS[d] in dg)
        got = _at_line_ln_k2(p.optics, which) - _at_line_ln_k2(nom.optics, which)
        np.testing.assert_allclose(got, pred, rtol=0, atol=1e-9, err_msg=which)
        assert np.std(pred) > 0.02, which


def test_g26_offset_and_coupling_correlation_matches_the_jacobian_prediction(g26_dies):
    """07 P1-16's test on g26: the sampled correlation (and spread) of (resonance offset, ln K^2 at the line) matches
    the analytic Jacobian prediction from the preset's sigmas, for both couplers."""
    cfg, p, _ = g26_dies
    g = cfg.geometry
    var = {d: float(g[d].get("die_sigma_nm", 0.0)) ** 2 + float(g[d].get("ring_sigma_nm", 0.0)) ** 2
           for d in ("width", "thickness", "gap_through", "gap_drop") if d in g}
    jl = dict(zip(cfg.get("ring.geometry.jacobian")["d"], cfg.get("ring.geometry.jacobian")["of"]["lambda_res_nm_per_nm"]))
    jac = cfg.get("ring.geometry.jacobian_coupling")
    off = (p.optics.resonance_offset_m * 1e9).ravel()
    for which in ("through", "drop"):
        jk = dict(zip(jac["d"], jac["of"][f"ln_kappa2_{which}_per_nm"]))
        gap = f"gap_{which}_nm"
        a_w = jk["width_nm"] - jk[gap]                 # a width error also closes the gap
        cov = -(a_w * jl["width_nm"] * var["width"] + jk["film_nm"] * jl["film_nm"] * var["thickness"])
        v_off = jl["width_nm"] ** 2 * var["width"] + jl["film_nm"] ** 2 * var["thickness"]
        v_k = a_w ** 2 * var["width"] + jk["film_nm"] ** 2 * var["thickness"] + jk[gap] ** 2 * var[f"gap_{which}"]
        lnk = _at_line_ln_k2(p.optics, which).ravel()
        assert np.corrcoef(off, lnk)[0, 1] == pytest.approx(cov / math.sqrt(v_off * v_k), abs=0.03), which
        assert np.std(lnk) == pytest.approx(math.sqrt(v_k), rel=0.1), which


def test_coupler_spreads_come_from_one_local_gap_sigma(g26_dies):
    """params-11: the stand-in draws of q_loaded (rel 0.015, 1.5 nm of gap) and kappa_d_over_kappa (rel 0.1, 5 nm)
    with femwell's -2 %/nm are gone; both couplers' gaps share one local sigma, and the within-die spread of each
    coupler is the coupling Jacobian times the per-ring sigmas."""
    cfg, p, _ = g26_dies
    assert "ring.optics.q_loaded" not in cfg.variation and "ring.optics.kappa_d_over_kappa" not in cfg.variation
    g = cfg.geometry
    assert g["gap_through"] == g["gap_drop"] and not g["gap_through"].get("die_sigma_nm")
    jac = cfg.get("ring.geometry.jacobian_coupling")
    sr = {d: float(g[d].get("ring_sigma_nm", 0.0)) for d in ("width", "thickness", "gap_through", "gap_drop")}
    n = p.n_rings
    for which in ("through", "drop"):
        jk = dict(zip(jac["d"], jac["of"][f"ln_kappa2_{which}_per_nm"]))
        gap = f"gap_{which}_nm"
        sd = math.sqrt(((jk["width_nm"] - jk[gap]) * sr["width"]) ** 2 + (jk["film_nm"] * sr["thickness"]) ** 2
                       + (jk[gap] * sr[f"gap_{which}"]) ** 2)
        lnk = _at_line_ln_k2(p.optics, which)
        within = lnk - lnk.mean(axis=1, keepdims=True)
        assert np.std(within) == pytest.approx(sd * math.sqrt((n - 1) / n), rel=0.05), which


def test_mini_coupling_jacobian_moves_a_free_coupler():
    """The geometry moves a coupler whichever R4 member is free (mini: Q_t free under the Q_L anchor; before, an
    output on a free member raised): K_t^2 of each device is the nominal device's times exp(J dg)."""
    d = copy.deepcopy(yaml_load(MINI.read_text()))
    d["ring"]["geometry"]["jacobian"]["value"]["of"]["ln_kappa2_through_per_nm"] = [0.01, 0.0]
    cfg = load(d)
    p = realize(cfg, 64, 0, uncertainty="nominal", thermal_hook=None)
    nom = realize(cfg, 1, 0, uncertainty="nominal", variation="off", thermal_hook=None)
    w = np.broadcast_to(p.diagnostics["geometry_draws_nm"]["width"], p.optics.kappa2_through.shape)
    got = _at_line_ln_k2(p.optics, "through") - _at_line_ln_k2(nom.optics, "through")
    np.testing.assert_allclose(got, 0.01 * w, rtol=0, atol=1e-9)
    # Q_L follows the device's couplers (the anchor holds for the nominal device)
    np.testing.assert_allclose(1.0 / p.optics.q_loaded, 1.0 / p.optics.q_intrinsic + 1.0 / p.optics.q_through
                               + 1.0 / p.optics.q_drop, rtol=1e-12)
    assert np.std(p.optics.q_loaded) > 0


# ------------------------------------------------------------------------------------------------ params-2

def test_g26_fsr_follows_n_g_at_the_mask_radius(g26_dies):
    """params-2: the radius solved from the nominal n_g and the FSR hypothesis is held for every device of the world
    (a mask dimension), the order is the nominal device's (Jacobians at fixed r_ref and order), and each device's FSR
    is c/(n_g L) with its own n_g. Before, R absorbed the n_g draw (FSR exactly 1.8 THz, orders 78-80)."""
    cfg, p, nom = g26_dies
    o = p.optics
    np.testing.assert_array_equal(o.radius_m, np.broadcast_to(nom.optics.radius_m, o.radius_m.shape))
    np.testing.assert_array_equal(o.order, np.broadcast_to(nom.optics.order, o.order.shape))
    np.testing.assert_allclose(o.fsr_hz, C0 / (o.n_g * o.length_m), rtol=1e-12)
    jac = cfg.get("ring.geometry.jacobian")
    jn = dict(zip(jac["d"], jac["of"]["n_g_per_nm"]))
    dg = p.diagnostics["geometry_draws_nm"]
    np.testing.assert_allclose(o.n_g, nom.optics.n_g + jn["width_nm"] * dg["width"] + jn["film_nm"] * dg["thickness"],
                               rtol=1e-12)
    rel_sd = float(np.std(o.fsr_hz / nom.optics.fsr_hz - 1.0))
    assert 0.003 < rel_sd < 0.01                      # ~0.6 %: ~11 GHz of FSR spread at 1.8 THz


# ------------------------------------------------------------------------------------------------ params-10

def test_bend_radiation_is_exponential_in_the_geometry(g26_dies):
    """params-10: the Jacobian's alpha_rad secant is applied as d ln alpha_rad/dp from the artefact's stored
    evaluations, so no device gets a negative bend radiation (13 % did with the linear offset)."""
    cfg, p, nom = g26_dies
    path = rel.P_A_RAD
    a = np.asarray(p.values[path])
    a0 = np.asarray(nom.values[path])
    jac = cfg.get("ring.geometry.jacobian")
    ev = jac["evaluations"]
    dln = {k: (math.log(ev[k]["plus"]["alpha_rad"]) - math.log(ev[k]["minus"]["alpha_rad"])) / (2 * ev[k]["step_nm"])
           for k in jac["d"]}
    assert dln["width_nm"] == pytest.approx(-0.0485, abs=1e-4)
    assert dln["film_nm"] == pytest.approx(-0.1785, abs=1e-4)
    assert dln["slab_nm"] == pytest.approx(0.4104, abs=1e-4)
    dg = p.diagnostics["geometry_draws_nm"]
    pred = a0 * np.exp(dln["width_nm"] * dg["width"] + dln["film_nm"] * dg["thickness"])
    np.testing.assert_allclose(a, pred, rtol=1e-12)
    assert a.min() > 0.0


def test_exp_output_without_evaluations_uses_the_secant_over_the_leaf():
    """An inline Jacobian without evaluations: d ln alpha_rad/dp = J/alpha_rad (first-order equal, always > 0)."""
    d = copy.deepcopy(yaml_load(MINI.read_text()))
    d["ring"]["geometry"]["jacobian"]["value"]["of"]["alpha_rad_db_per_cm_per_nm"] = [-1.0, 0.0]
    cfg = load(d)
    p = realize(cfg, 64, 0, uncertainty="nominal", thermal_hook=None)
    w = np.broadcast_to(p.diagnostics["geometry_draws_nm"]["width"], (64, cfg.n_rings))
    a0 = float(cfg.get(rel.P_A_RAD))
    got = np.asarray(p.values[rel.P_A_RAD]) / (100.0 * math.log(10.0) / 10.0)
    np.testing.assert_allclose(got, a0 * np.exp(-1.0 / a0 * w), rtol=1e-12)
    assert got.min() > 0.0


# ------------------------------------------------------------------------------------------------ params-4

def test_heater_z_is_the_same_on_every_line():
    """params-4: with ring.heater.p_fsr_ref_wavelength_nm the R2 inversion uses the FSR at the P_FSR measurement line,
    so Z_heater->ring and the efficiency are the same on every line of g26 (they followed lambda_line^2: 2.3 % across
    the bank) and equal the G25 bench's."""
    cfg = _load("g26")
    d = derive(cfg)
    z = d.diagnostics["z_heater_ring_k_per_mw"]
    eff = d.diagnostics["heater_efficiency_nm_per_mw"]
    assert np.ptp(z) <= 1e-12 * z.max() and np.ptp(eff) <= 1e-12 * eff.max()
    lam = cfg.si(rel.P_PFSR_LAM)
    assert d.params.values[rel.P_FSR_SPAN][0, 0] == pytest.approx(lam * lam * 1.8e12 / C0, rel=1e-12)
    g25 = derive(_load("g25_test_vehicle")).diagnostics["z_heater_ring_k_per_mw"]
    np.testing.assert_allclose(z, g25[0], rtol=1e-12)
    # without the key every ring keeps its own line (the previous form): Z ~ lambda_line^2
    doc = yaml_load((ROOT / "mrm" / "params" / "presets" / "g26.yaml").read_text())
    del doc["ring"]["heater"]["p_fsr_ref_wavelength_nm"]
    d2 = derive(_load(doc))
    z2 = d2.diagnostics["z_heater_ring_k_per_mw"]
    lam_l = d2.params.optics.lambda_line_m[0]
    assert z2[0] / z2[-1] == pytest.approx((lam_l[0] / lam_l[-1]) ** 2, rel=3e-3)


# ------------------------------------------------------------------------------------------------ params-7

def test_line_wavelengths_carry_the_channel_plan_provenance():
    """params-7 (07 P1-14): the line wavelengths are derived from the channel plan, so the per-ring FSR in nm and its
    diagnostics carry the plan's 'inferred' grid placement instead of 'printed'."""
    cfg = _load("g26")
    d = derive(cfg)
    der = d.params.derivations
    assert {"channel_plan.center_thz", "channel_plan.spacing_ghz"} <= set(der["derived.optics.lambda_line_m"].inputs)
    assert "derived.optics.lambda_line_m" in der["derived.optics.lambda_res_cold_m"].inputs
    assert "derived.optics.lambda_line_m" in der["derived.optics.fsr_m"].inputs
    assert cfg.leaves["channel_plan.center_thz"].provenance == "inferred"
    assert effective_provenance("derived.optics.fsr_m", cfg, der) == "inferred"
    assert d.provenance["fsr_nm"] == "inferred"
    assert d.provenance["heater_efficiency_nm_per_mw"] == "inferred"     # p_fsr_ref_wavelength_nm (CAL-7) is inferred
    assert "channel_plan.center_thz" in explain(cfg, "derived.optics.fsr_m")
    for p in (rel.P_QI, "derived.optics.kappa2_through", rel.P_DNDT):
        ins = set(der[p].inputs)
        assert ins & {"derived.optics.lambda_line_m", "derived.optics.lambda_res_cold_m"}, p


# ------------------------------------------------------------------------------------------------ params-8

def test_depletion_exponent_range_ends_build_the_analytic_junction_table():
    """params-8: every value the leaf's range admits builds the analytic table (0.33 and 0.67 passed load and realize
    and then raised in mrm.plant.electrical); values outside [1/3, 2/3] are load errors (schema bounds)."""
    from mrm.plant.electrical import junction_table_from_plant_params
    cfg = _load("g26")
    lo, hi = cfg.leaves["ring.junction.depletion_exponent"].range
    for m in (lo, hi, 0.5):
        c = _load("g26", overrides={"ring.junction.depletion_exponent": m})
        junction_table_from_plant_params(realize(c, 1, 0, uncertainty="nominal", variation="off", thermal_hook=None))
    for m in (0.33, 0.67):
        with pytest.raises(ConfigError, match="depletion_exponent"):
            _load("g26", overrides={"ring.junction.depletion_exponent": m})


# ------------------------------------------------------------------------------------------------ params-9

def test_world_values_and_alternatives_lie_in_the_leaf_range():
    """params-9: a leaf's range includes every value a world can take (params_content handoff 5.7): interval ends
    and corners of an absolute entry, and every preset alternative (a selected alternative is range-checked, so one
    outside could never be selected). g26 holds the OCP +9 dBm corner; g25_test_vehicle drops g26's Guide module
    power, which is not a power at its bench ring."""
    cfg = _load("g26")
    rng = cfg.leaves["optical_path.chip_input_loss_db"].range
    assert rng[0] <= cfg.uncertainty["chip_input_loss"].fields["corners"]["ocp_ring_plus9dbm"] <= rng[1]
    g25 = _load("g25_test_vehicle")
    assert "guide_16_lines" not in g25.leaves["laser.power_per_line_dbm"].alternatives
    assert "guide_16_lines" in cfg.leaves["laser.power_per_line_dbm"].alternatives
    for name in PRESETS:
        c = _load(name)
        for path, lf in c.leaves.items():
            if not (isinstance(lf.range, tuple) and len(lf.range) == 2):
                continue
            for an, a in lf.alternatives.items():
                v = a.get("value")
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    assert lf.range[0] <= v <= lf.range[1], (name, path, an, v)
    d = copy.deepcopy(yaml_load(MINI.read_text()))
    d["uncertainty"]["q_loaded"]["corners"]["too_low"] = 2000
    with pytest.raises(ConfigError, match="outside the range"):
        load(d)
    d = copy.deepcopy(yaml_load(MINI.read_text()))
    d["ring"]["optics"]["q_loaded"]["alternatives"]["far"] = {"value": 9000, "provenance": "assumed"}
    with pytest.raises(ConfigError, match="can never be selected"):
        load(d)
    d = copy.deepcopy(yaml_load(MINI.read_text()))
    d["ring"]["optics"]["q_loaded"]["alternatives"]["talk"] = None
    with pytest.raises(ConfigError, match="null"):
        load(d)


# ------------------------------------------------------------------------------------------------ params-13

def test_r_cold_range_holds_the_calibrated_droop_worlds():
    """params-13: the R0 range is the G25 hot range over 1 + y P_FSR for every droop the thermo-optic worlds need
    (0.00904 here, 0.0141 / 0.0161 in g26_calibrated's order-2 CAL-4 refits); g26_calibrated's R0 41.29 ohm is a valid
    g26 override (it was refused against [42, 58])."""
    cfg = _load("g26")
    lo, hi = cfg.leaves["ring.heater.r_cold_ohm"].range
    p = 25.45
    for y in (0.00904, 0.0141, 0.0161):
        assert lo <= 52.8 / (1.0 + y * p) and 70.5 / (1.0 + y * p) <= hi, y
    cal = _load("g26_calibrated").get("ring.heater.r_cold_ohm")
    _load("g26", overrides={"ring.heater.r_cold_ohm": cal})


# ------------------------------------------------------------------------------------------------ params-3

def test_film_200_world_is_the_rib200_artefact():
    """params-3: uncertainty.film_thickness '200' takes the 400 x 200 nm rib (modes_rib200: a depletion modulator
    needs the slab), not the strip stand-in; it keeps the measured G25 Q_i with the rib200 anchor term, so eta is
    ~0.83 (the strip's fictitious 11.09 dB/cm non-heating loss gave 0.77)."""
    art = json.loads((EXTRACTED / "modes_rib200.json").read_text())
    base = realize(_load("g25_test_vehicle"), 1, 0, uncertainty="nominal", variation="off", thermal_hook=None)
    cfg = _load("g25_test_vehicle", alts={"uncertainty.film_thickness": "200"})
    pp = realize(cfg, 1, 0, uncertainty="nominal", variation="off", thermal_hook=None)
    assert float(pp.optics.n_g[0, 0]) == pytest.approx(art["n_g"], rel=1e-12)
    s25 = art["dlambda_dt_pm_per_k"]["value"][art["dlambda_dt_pm_per_k"]["t_c"].index(25.0)]
    assert float(pp.thermal.dlambda_dt_s0_m_per_k[0, 0]) == pytest.approx(s25 * 1e-12, rel=1e-12)
    assert cfg.get("ring.geometry.slab_nm") == 80.0 and cfg.get("ring.geometry.film_nm") == 200.0
    assert cfg.get("ring.geometry.jacobian")["of"] == art["jacobian"]["of"]
    assert cfg.get(rel.P_A_RAD) == pytest.approx(2.03)
    np.testing.assert_allclose(pp.optics.q_intrinsic, base.optics.q_intrinsic, rtol=2e-3)
    assert float(pp.optics.heat_fraction[0, 0]) == pytest.approx(0.833, abs=0.005)
