"""Check that the notes of the g26 preset quote the extraction artefacts and the realized model correctly.

Read this first. Some leaf notes in the presets state numbers (beta, the heater gain Z, loss terms, geometry
spreads). These tests recompute each quoted number from derive() and from the artefact tables in
mrm/data/extracted/, so a re-extraction or a model change that makes a note stale fails here. The CHANGELOG entry
that changed each note is named in parentheses.

- beta (CHANGELOG params-5) is fitted over the operating range: it is the secant of the solver curve s(T) from
  25 degC to the nominal lock temperature, not the tangent at 25 degC.
- The heater, Z, bend-radiation and geometry notes and the lock_slope gap of docs/PARAMETERS_FORMAT.md (CHANGELOG
  params-12) were once computed with the fallbacks used before the artefacts existed (s0 66 pm/K, beta 1.4e-3 /K)
  and with the strip Jacobian.
- The rib200 Q anchor (CHANGELOG params-3) keeps the measured G25 intrinsic Q in the 200 nm film world.
"""
from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from mrm.params import derive, load, realize

ROOT = Path(__file__).resolve().parents[2]
EXTRACTED = ROOT / "mrm" / "data" / "extracted"
OFFSET_NM = 7.6            # nominal lock offset, nm: resonance shift from 25 degC to the lock (the CAL-7 value)


def _load(name="g26", **kw):
    """params.load with the load warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load(name, **kw)


def _numbers_in(pattern, text):
    """The numbers of the groups of the first match of regex ``pattern`` in a note (fails when it does not match)."""
    m = re.search(pattern, text)
    assert m, f"note does not state {pattern!r}: {text}"
    return [float(g) for g in m.groups()]


def _solver_curve(xsec):
    """The artefact's s(T) table (pm/K), its quadratic fit, and the lock temperature of the 7.6 nm offset from 25 degC
    on the fitted curve."""
    a = json.loads((EXTRACTED / f"modes_{xsec}.json").read_text())["dlambda_dt_pm_per_k"]
    t, s = np.asarray(a["t_c"], float), np.asarray(a["value"], float)
    q = np.polyfit(t, s, 2)
    s25 = float(s[list(t).index(25.0)])
    t_lock = brentq(lambda x: quad(lambda y: np.polyval(q, y), 25.0, x)[0] - OFFSET_NM * 1e3, 26.0, 250.0)
    return q, s25, t_lock


def _secant(xsec):
    """(beta_secant /K, s at 25 degC, lock temperature degC, s at the lock) of the artefact curve of ``xsec``: the beta
    for which s25 (1 + beta (T_lock - 25)) equals the fitted curve at T_lock. s in pm/K."""
    q, s25, t_lock = _solver_curve(xsec)
    return (float(np.polyval(q, t_lock)) / s25 - 1.0) / (t_lock - 25.0), s25, t_lock, float(np.polyval(q, t_lock))


def test_beta_is_the_secant_of_the_solver_curve_through_the_lock():
    """beta (CHANGELOG params-5): the linear thermo-optic slope s(T) = s0 (1 + beta (T - 25 degC)) is exact at 25 degC
    and at the nominal lock temperature on the solver curve (rib160, and rib200 of the 200 nm world). The artefact's
    tangent at 25 degC overstates s at the lock by more than 2 %."""
    cfg = _load()
    leaf = cfg.leaves["ring.thermal.dlambda_dt_beta_per_k"]
    b160, s25, t_lock, s_lock = _secant("rib160")
    assert float(leaf.value) == pytest.approx(b160, rel=2e-4)
    assert float(leaf.alternatives["rib200"]["value"]) == pytest.approx(_secant("rib200")[0], rel=2e-4)
    assert float(cfg.get("ring.thermal.dlambda_dt_pm_per_k")) == pytest.approx(s25, rel=1e-12)
    assert s25 * (1.0 + float(leaf.value) * (t_lock - 25.0)) == pytest.approx(s_lock, rel=3e-4)
    tangent = json.loads((EXTRACTED / "modes_rib160.json").read_text())["dlambda_dt_beta_per_k"]["value"][0]
    assert s25 * (1.0 + tangent * (t_lock - 25.0)) / s_lock - 1.0 > 0.02
    # the numbers the note quotes
    vals = _numbers_in(r"\((\d+(?:\.\d+)?)/(\d+(?:\.\d+)?) - 1\)/(\d+(?:\.\d+)?)", leaf.note)
    assert vals == pytest.approx([round(s_lock, 3), round(s25, 3), round(t_lock - 25.0, 2)], abs=1.1e-3)
    assert _numbers_in(r"lock temperature (\d+(?:\.\d+)?) degC", leaf.note)[0] == pytest.approx(t_lock, abs=0.006)
    # the realized model: T_lock and the slope at the lock against the solver curve
    d = derive(cfg)
    q, _, _ = _solver_curve("rib160")
    tl = float(d.diagnostics["t_ring_lock_c"][0])
    assert tl - t_lock == pytest.approx(0.41, abs=0.02)
    assert float(d.diagnostics["disturbance_pm_per_k"][0]) / float(np.polyval(q, tl)) - 1.0 == pytest.approx(
        0.0, abs=1e-3)


def test_heater_notes_quote_the_realized_z_and_hold_the_g25_droop():
    """Heater notes (CHANGELOG params-12, items b and c). Z is the heater-to-ring gain, K/mW: 5.704 (fsr_integral at
    the P_FSR line, the same on every line) and 5.475 under lock_slope. heater_self_ratio keeps y = TCR R_th,H at the
    G25 droop fit 0.00904 /mW. The Z and y of the other worlds are as the notes quote them."""
    def z(**kw):
        return float(derive(_load("g25_test_vehicle", **kw)).diagnostics["z_heater_ring_k_per_mw"][0])
    cfg = _load()
    z0 = z()
    z_ls = z(overrides=["fidelity.heater_anchor=lock_slope"])
    ratio = float(cfg.get("ring.thermal.heater_self_ratio"))
    tcr = float(cfg.get("ring.heater.tcr_per_k"))
    assert tcr * ratio * z0 == pytest.approx(0.00904, rel=2e-4)
    note = cfg.leaves["ring.thermal.heater_self_ratio"].note
    assert _numbers_in(r"nominal Z = (\d+(?:\.\d+)?) K/mW", note)[0] == pytest.approx(z0, abs=6e-4)
    r_ls, z_ls_q = _numbers_in(r"(\d+(?:\.\d+)?) under lock_slope, Z (\d+(?:\.\d+)?)", note)
    assert z_ls_q == pytest.approx(z_ls, abs=6e-4) and r_ls == pytest.approx(0.00904 / (tcr * z_ls), abs=6e-4)
    for alts, pat in (({"uncertainty.fsr": "window_a"}, r"window_a \(Z (\d+(?:\.\d+)?)\)"),
                      ({"uncertainty.thermo_optic": "H86"}, r"H86 \(Z (\d+(?:\.\d+)?)\)"),
                      ({"uncertainty.fsr": "g25_cluster"}, r"g25_cluster \(Z (\d+(?:\.\d+)?)\)")):
        assert _numbers_in(pat, note)[0] == pytest.approx(z(alts=alts), abs=6e-4), alts
    zn = cfg.leaves["ring.thermal.z_heater_ring_k_per_mw"].note
    assert _numbers_in(r"\((\d+(?:\.\d+)?) K/mW, the same on every line; lock_slope gives (\d+(?:\.\d+)?)\)",
                       zn) == pytest.approx([z0, z_ls], abs=6e-4)
    alts = cfg.leaves["ring.thermal.heater_self_ratio"].alternatives
    for name, t in (("tin", 0.5e-3), ("textbook", 1.0e-3)):
        assert float(alts[name]["value"]) == pytest.approx(0.00904 / (t * z0), abs=6e-4), name
    fmt = (ROOT / "docs" / "PARAMETERS_FORMAT.md").read_text()
    pct, a, b = _numbers_in(r"this Z is (\d+(?:\.\d+)?) % lower \((\d+(?:\.\d+)?) against\s+(\d+(?:\.\d+)?) K/mW\)",
                            fmt)
    assert (a, b) == pytest.approx((z_ls, z0), abs=6e-4) and pct == pytest.approx(100 * (1 - z_ls / z0), abs=0.06)


def test_bend_radiation_and_geometry_notes_quote_the_loaded_artefacts():
    """Loss and geometry notes (CHANGELOG params-12, items e and f). The bend-radiation note keeps Q_i (not
    'alpha_i ~112'). The variation.geometry note quotes the spreads of the loaded rib160 Jacobian (not the strip's)."""
    cfg = _load()
    pp = realize(cfg, 1, 0, uncertainty="nominal", variation="off", thermal_hook=None)
    note = cfg.leaves["ring.optics.loss_budget.alpha_bend_radiation_db_per_cm"].note
    assert "112" not in note
    rad, ai = _numbers_in(r"(\d+(?:\.\d+)?) dB/cm for rib160, 0.1 % of alpha_i (\d+(?:\.\d+)?) dB/cm", note)
    db = 100.0 * math.log(10.0) / 10.0           # (1/m) per (dB/cm): 1 dB/cm = 100 ln(10)/10 per m
    assert rad == pytest.approx(float(cfg.get("ring.optics.loss_budget.alpha_bend_radiation_db_per_cm")), abs=6e-4)
    assert ai == pytest.approx(float(pp.optics.alpha_i_per_m[0, 0]) / db, abs=6e-3)
    g = cfg.geometry
    jl = dict(zip(cfg.get("ring.geometry.jacobian")["d"],
                  cfg.get("ring.geometry.jacobian")["of"]["lambda_res_nm_per_nm"]))
    die = math.hypot(jl["width_nm"] * g["width"]["die_sigma_nm"], jl["film_nm"] * g["thickness"]["die_sigma_nm"])
    loc = math.hypot(jl["width_nm"] * g["width"]["ring_sigma_nm"], jl["film_nm"] * g["thickness"]["ring_sigma_nm"])
    assert _numbers_in(r"they give (\d+(?:\.\d+)?) nm and (\d+(?:\.\d+)?) nm",
                       g["note"]) == pytest.approx([die, loc], abs=6e-3)
    assert _numbers_in(r"loaded rib160 Jacobian \((\d+(?:\.\d+)?) / (\d+(?:\.\d+)?) nm/nm\)",
                       g["note"]) == pytest.approx([jl["width_nm"], jl["film_nm"]], abs=6e-4)
    jc = cfg.get("ring.geometry.jacobian_coupling")
    jt = dict(zip(jc["d"], jc["of"]["ln_kappa2_through_per_nm"]))
    jd = dict(zip(jc["d"], jc["of"]["ln_kappa2_drop_per_nm"]))
    sg = g["gap_through"]["ring_sigma_nm"]
    assert _numbers_in(r"gaps alone spread K_t\^2 and K_d\^2 by (\d+(?:\.\d+)?) % and (\d+(?:\.\d+)?) %",
                       g["note"]) == pytest.approx([100 * abs(jt["gap_through_nm"]) * sg,
                                                    100 * abs(jd["gap_drop_nm"]) * sg], abs=0.06)
    sw, sf = g["width"]["ring_sigma_nm"], g["thickness"]["ring_sigma_nm"]
    loc_t = math.sqrt(((jt["width_nm"] - jt["gap_through_nm"]) * sw) ** 2 + (jt["film_nm"] * sf) ** 2
                      + (jt["gap_through_nm"] * sg) ** 2)
    loc_d = math.sqrt(((jd["width_nm"] - jd["gap_drop_nm"]) * sw) ** 2 + (jd["film_nm"] * sf) ** 2
                      + (jd["gap_drop_nm"] * sg) ** 2)
    assert _numbers_in(r"move (\d+(?:\.\d+)?) % \(through\) and (\d+(?:\.\d+)?) % \(drop\) within a\s+die",
                       g["note"]) == pytest.approx([100 * loc_t, 100 * loc_d], abs=0.06)


def test_rib200_anchor_term_keeps_the_g25_q_intrinsic():
    """rib200 anchor (CHANGELOG params-3): rib200_q_anchor is alpha_i,160 (n_g,200/n_g,160 - 1) + alpha_rad,160 for
    each coupling regime, then the mean (the recipe of its note), so the 200 nm world keeps the measured G25 Q_i."""
    cfg = _load()
    lb = "ring.optics.loss_budget."
    ng160 = json.loads((EXTRACTED / "modes_rib160.json").read_text())["n_g"]
    ng200 = json.loads((EXTRACTED / "modes_rib200.json").read_text())["n_g"]
    rad = float(cfg.get(lb + "alpha_bend_radiation_db_per_cm"))
    rest = float(cfg.get(lb + "alpha_fca_db_per_cm")) + float(cfg.get(lb + "alpha_scatter_db_per_cm"))
    terms = []
    for other in (float(cfg.get(lb + "alpha_other_abs_db_per_cm")),
                  float(cfg.leaves[lb + "alpha_other_abs_db_per_cm"].alternatives["over"]["value"])):
        terms.append((rest + other + rad) * (ng200 / ng160 - 1.0) + rad)
    anchor = cfg.leaves[lb + "alpha_bend_radiation_db_per_cm"].alternatives["rib200_q_anchor"]
    assert float(anchor["value"]) == pytest.approx(float(np.mean(terms)), abs=6e-3)
    assert _numbers_in(r"= (\d+(?:\.\d+)?) dB/cm for under", anchor["note"])[0] == pytest.approx(terms[0], abs=6e-4)
    assert _numbers_in(r"and (\d+(?:\.\d+)?) for over", anchor["note"])[0] == pytest.approx(terms[1], abs=6e-4)
