"""Verification of the mode and coupler extraction (mrm.extraction.modes, mrm.extraction.coupling).

Fast tests load the committed artefacts in mrm/data/extracted/ and check identities, schema and the solver
cross-checks recorded in them. Solver re-runs are marked slow.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mrm.extraction import common as C

DATA = C.DATA_DIR
XSECS_MODES = ("rib160", "rib200", "strip400x200")
XSECS_COUPLING = ("rib160", "rib200", "strip400x200")
LIGHTMATTER_XSECS = ("rib160", "rib200")


def _load(name):
    path = DATA / f"{name}.json"
    side = DATA / f"{name}.sidecar.json"
    if not path.exists() or not side.exists():
        pytest.skip(f"artefact {name} not generated yet")
    return json.loads(path.read_text()), json.loads(side.read_text()), path


# --------------------------------------------------------------------------------------------- modes: schema


@pytest.mark.parametrize("xs", XSECS_MODES + ("decea2019",))
def test_modes_sidecar_sha256_and_schema(xs):
    d, side, path = _load(f"modes_{xs}")
    assert side["schema"] == "modes/1" and d["schema"] == "modes/1"
    assert side["sha256"] == C.sha256_file(path)
    for k in ("inputs", "tool_versions", "mesh", "validation", "created_utc"):
        assert k in side, k
    assert side["inputs"]["materials"]["schema"] == "materials/1"
    from mrm import materials as M
    assert side["inputs"]["materials"]["sha256"] == M.data_sha256(), "materials changed since the extraction"


@pytest.mark.parametrize("xs", XSECS_MODES)
def test_modes_contract_fields(xs):
    """Every field of ARCHITECTURE 5.1."""
    d, _, _ = _load(f"modes_{xs}")
    for k in ("width_nm", "film_nm", "slab_nm", "sidewall_deg", "clad"):
        assert k in d["xsec"], k
    for k in ("r_ref_um", "lambda0_nm", "t_ref_c", "n_eff_poly_lambda", "n_g", "dn_g_dlambda_per_nm",
              "dneff_dt_per_k", "alpha_l_per_k", "dlambda_dt_pm_per_k", "index_weights", "power_fraction",
              "mode_area_um2", "alpha_rad_db_per_cm", "jacobian", "crosscheck", "mode_families", "alt_radius"):
        assert k in d, k
    assert len(d["n_eff_poly_lambda"]) == 3
    for k in ("dneff_dt_per_k", "alpha_l_per_k", "dlambda_dt_pm_per_k"):
        assert d[k]["t_c"] == [0.0, 25.0, 50.0, 75.0, 105.0, 125.0]
        assert len(d[k]["value"]) == 6
    for k in ("tidy3d_n_eff", "femwell_n_eff", "delta", "same_r_ref"):
        assert k in d["crosscheck"], k
    assert d["crosscheck"]["same_r_ref"] is True
    assert set(d["index_weights"]) >= {"si", "box", "clad"}


# --------------------------------------------------------------------------------------------- modes: identities


@pytest.mark.parametrize("xs", XSECS_MODES + ("decea2019",))
def test_index_weight_identity(xs):
    """sum_i n_g,i dn_eff/dn_i = n_g to 1e-4 (07 P0-3 test_index_weight_identity, 06 section 2.3)."""
    d, _, _ = _load(f"modes_{xs}")
    chk = d["index_weight_check"]
    s = sum(chk["n_g_material"][k] * d["index_weights"][k] for k in d["index_weights"])
    assert abs(s - chk["n_g"]) < 1e-4
    assert abs(s - chk["sum_ng_i_w_i"]) < 1e-12
    # the first-order perturbation integral agrees with the finite differences
    for k in ("si", "box", "clad"):
        assert abs(chk["perturbation"][k] - d["index_weights"][k]) < 2e-4, k


@pytest.mark.parametrize("xs", XSECS_MODES + ("decea2019",))
def test_dlambda_dt_formula(xs):
    """dlambda/dT = (lambda/n_g) (dn_eff/dT + n_eff alpha_L) is reproduced from the stored tables (R1)."""
    d, _, _ = _load(f"modes_{xs}")
    lam = d["lambda0_nm"]
    for i, tc in enumerate(d["dlambda_dt_pm_per_k"]["t_c"]):
        ng = d["n_g_t"]["value"][i]
        ne = d["n_eff_t"]["value"][i]
        v = lam / ng * (d["dneff_dt_per_k"]["value"][i] + ne * d["alpha_l_per_k"]["value"][i]) * 1e3
        assert v == pytest.approx(d["dlambda_dt_pm_per_k"]["value"][i], rel=1e-12)
        from mrm import materials as M
        assert d["alpha_l_per_k"]["value"][i] == pytest.approx(M.alpha_l_si_per_k(tc + 273.15), rel=1e-12)


@pytest.mark.parametrize("xs", XSECS_MODES)
def test_dlambda_dt_below_materials_ceiling_and_rising(xs):
    """H_solver lies below the all-silicon materials ceiling (02 section 3) and rises with T (+10-20 %)."""
    from mrm import materials as M
    d, _, _ = _load(f"modes_{xs}")
    v = np.array(d["dlambda_dt_pm_per_k"]["value"])
    for tc, s in zip(d["dlambda_dt_pm_per_k"]["t_c"], v):
        ceil = M.dlambda_dt_ceiling_pm_per_k(1.31e-6, tc + 273.15, neff_over_ng="all_si")[0]
        assert s < ceil
    assert np.all(np.diff(v) > 0)
    assert 1.08 < v[-1] / v[0] < 1.25
    assert 55.0 < v[1] < 75.0          # 25 C


@pytest.mark.parametrize("xs", XSECS_MODES)
def test_perturbation_dndt_matches_resolve(xs):
    """07 P0-3 test_dldt_perturbation_vs_resolve: lambda sum_i w_i dn_i/dT / n_g vs the re-solved value, < 2 %."""
    d, _, _ = _load(f"modes_{xs}")
    w = d["index_weights"]
    dn = d["materials_at_ref"]["dn_dt"]
    pert = sum(w[k] * dn[k] for k in w)
    i25 = d["dneff_dt_per_k"]["t_c"].index(25.0)
    assert pert == pytest.approx(d["dneff_dt_per_k"]["value"][i25], rel=0.02)


@pytest.mark.parametrize("xs", XSECS_MODES)
def test_r_ref_gives_design_fsr(xs):
    d, _, _ = _load(f"modes_{xs}")
    fsr = C.C0 / (d["n_g"] * 2 * math.pi * d["r_ref_um"] * 1e-6)
    assert fsr == pytest.approx(1.8e12, rel=2e-4)
    assert 5.5 < d["r_ref_um"] < 7.5


@pytest.mark.parametrize("xs", XSECS_MODES)
def test_polynomial_consistent(xs):
    d, _, _ = _load(f"modes_{xs}")
    c0, c1, c2 = d["n_eff_poly_lambda"]
    assert c0 == pytest.approx(d["n_eff"], abs=1e-5)
    assert c0 - d["lambda0_nm"] * c1 == pytest.approx(d["n_g"], abs=2e-3)
    assert -2 * d["lambda0_nm"] * c2 == pytest.approx(d["dn_g_dlambda_per_nm"], abs=2e-4)
    sw = d.get("lambda_sweep") or {}
    assert d["n_eff_table"]["lambda_nm"][2] == 1310.0


@pytest.mark.parametrize("xs", XSECS_MODES + ("decea2019",))
def test_fractions_sum_to_one(xs):
    d, _, _ = _load(f"modes_{xs}")
    pf = d["power_fraction"]
    assert pf["si"] + pf["box"] + pf["clad"] + pf.get("air", 0.0) == pytest.approx(1.0, abs=2e-3)
    ef = d["energy_fraction"]
    assert ef["si"] + ef["box"] + ef["clad"] + ef.get("air", 0.0) == pytest.approx(1.0, abs=2e-3)
    assert 0.03 < d["mode_area_um2"] < 1.0 or xs == "decea2019"


@pytest.mark.parametrize("xs", XSECS_MODES)
def test_mesh_convergence_recorded(xs):
    """At least three refinements with a Richardson error estimate; the production values are converged enough."""
    d, side, _ = _load(f"modes_{xs}")
    conv = side["validation"]["mesh_convergence"]
    assert len(conv["rows"]) >= 3
    err = d["discretization_error"]
    assert err["n_eff_bent"] < 3e-4          # < 0.1 nm of resonance
    assert err["n_g_bent"] < 1e-3
    i25 = d["dneff_dt_per_k"]["t_c"].index(25.0)
    assert err["dneff_dt_bent"] < 1e-3 * d["dneff_dt_per_k"]["value"][i25]


@pytest.mark.parametrize("xs", XSECS_MODES)
def test_tidy3d_crosscheck(xs):
    """Tidy3D local solver (converged by Richardson) agrees at the same r_ref; bend shifts agree."""
    d, _, _ = _load(f"modes_{xs}")
    cc = d["crosscheck"]
    assert abs(cc["delta"]) < 1e-3
    shift_t = cc["tidy3d_bend_shift"][-1]
    assert cc["femwell_exact_bend_shift"] == pytest.approx(shift_t, rel=0.1, abs=3e-5)
    # femwell's isotropic map overestimates the shift
    assert cc["femwell_isotropic_bend_shift"] > 1.5 * cc["femwell_exact_bend_shift"]


@pytest.mark.parametrize("xs", XSECS_MODES)
def test_jacobian_names_match_realize(xs):
    from mrm.params.realize import JACOBIAN_TARGETS, _JAC_DIMS
    d, _, _ = _load(f"modes_{xs}")
    jac = d["jacobian"]
    assert all(n in _JAC_DIMS for n in jac["d"])
    for out, coeffs in jac["of"].items():
        assert out in JACOBIAN_TARGETS, out
        assert len(coeffs) == len(jac["d"])
    lam_w = jac["of"]["lambda_res_nm_per_nm"][jac["d"].index("width_nm")]
    lam_t = jac["of"]["lambda_res_nm_per_nm"][jac["d"].index("film_nm")]
    assert 0.2 < lam_w < 1.5 and 0.5 < lam_t < 4.0     # nm of resonance per nm of geometry (06, lab exp. 08)
    # lambda_res Jacobian = (lambda0/n_g) * n_eff Jacobian
    for a, b in zip(jac["of"]["lambda_res_nm_per_nm"], jac["of"]["n_eff_per_nm"]):
        assert a == pytest.approx(d["lambda0_nm"] / d["n_g"] * b, rel=1e-9)


@pytest.mark.parametrize("xs", LIGHTMATTER_XSECS)
def test_rib_radiation_loss_and_pml_convergence(xs):
    d, _, _ = _load(f"modes_{xs}")
    rl = d["alpha_rad"]
    assert d["alpha_rad_db_per_cm"] >= 0.0
    assert rl["alpha_rad_rel_spread"] is None or rl["alpha_rad_rel_spread"] < 0.1


@pytest.mark.parametrize("xs", XSECS_MODES)
def test_mode_families_reported(xs):
    d, _, _ = _load(f"modes_{xs}")
    for fam in ("TM0", "TE1"):
        b = d["mode_families"][fam]
        assert "guided_straight" in b and "guided_bent" in b


def test_bend_study_settles_formulation():
    d, side, path = _load("modes_study_bend")
    assert side["sha256"] == C.sha256_file(path)
    v = d["verdict"]
    assert v["exact_transform_max_abs_error_vs_bessel"] < 1e-6
    assert v["isotropic_map_max_abs_error_vs_bessel"] > 1e-3
    for r in d["strip_bend_shift"]:
        if "tidy3d_shift" in r:
            assert r["exact_shift"] == pytest.approx(r["tidy3d_shift"], rel=0.1, abs=3e-5)


def test_decea_replication_reported():
    d, _, _ = _load("modes_decea2019")
    rep = d["replication"]
    assert rep["printed"]["n_g"] == 2.73 and rep["printed"]["gamma_ring"] == 0.6515
    assert set(rep["thickness_sweep"]) >= {"80.0", "100.0", "120.0"}


# --------------------------------------------------------------------------------------------- coupling


@pytest.mark.parametrize("xs", XSECS_COUPLING)
def test_coupling_artefact(xs):
    d, side, path = _load(f"coupling_{xs}")
    assert d["schema"] == side["schema"] == "coupling/1"
    assert side["sha256"] == C.sha256_file(path)
    m, _, _ = _load(f"modes_{xs}")
    assert d["r_ref_um"] == pytest.approx(m["r_ref_um"], rel=1e-12)
    for cp in ("through", "drop"):
        for k in ("gap_nm", "kappa2_per_pass", "dln_kappa2_dlambda_per_nm", "dln_kappa2_dgap_per_nm",
                  "coupler_phase_rad"):
            assert k in d[cp], (cp, k)
        assert d[cp]["dln_kappa2_dgap_per_nm"] < 0
        assert d[cp]["dln_kappa2_dlambda_per_nm"] > 0
        assert abs(d[cp]["power_out_over_in"] - 1.0) < 0.01


@pytest.mark.parametrize("xs", XSECS_COUPLING)
def test_coupling_table_monotone(xs):
    d, _, _ = _load(f"coupling_{xs}")
    tab = d["kappa2_table"]
    k2 = np.array(tab["kappa2"])
    assert k2.shape == (len(tab["lambda_nm"]), len(tab["gap_nm"]))
    assert np.all(k2 > 0) and np.all(k2 < 1)
    assert np.all(np.diff(k2, axis=1) < 0)          # weaker with gap
    assert np.all(np.diff(k2, axis=0) > 0)          # stronger at longer wavelength


@pytest.mark.parametrize("xs", XSECS_COUPLING)
def test_coupling_design_consistent(xs):
    from mrm.extraction.coupling import q_to_k2
    d, _, _ = _load(f"coupling_{xs}")
    des = d["design"]
    lam = d["lambda0_nm"] * 1e-3
    fsr = d["fsr_design_thz"] * 1e12
    crit = des["critical"]
    assert crit["kappa2"] == pytest.approx(q_to_k2(crit["q_e"], lam, fsr), rel=1e-12)
    assert d["through"]["kappa2_per_pass"] == pytest.approx(crit["kappa2"], rel=0.02)
    # mirror: the over-coupled branch needs a smaller gap than the under-coupled one
    assert des["over"]["nominal"]["gap_nm"] < crit["gap_nm"] < des["under"]["nominal"]["gap_nm"]
    # drop prior: higher kappa_d/kappa -> smaller gap; drop weaker than through
    dr = des["drop"]
    assert dr["high"]["gap_nm"] < dr["nominal"]["gap_nm"] < dr["low"]["gap_nm"]
    assert d["drop"]["kappa2_per_pass"] < d["through"]["kappa2_per_pass"]


@pytest.mark.parametrize("xs", XSECS_COUPLING)
def test_coupling_jacobian_names(xs):
    from mrm.params.realize import JACOBIAN_TARGETS, _JAC_DIMS
    d, _, _ = _load(f"coupling_{xs}")
    jac = d["jacobian"]
    assert all(n in _JAC_DIMS for n in jac["d"])
    for out, coeffs in jac["of"].items():
        assert out in JACOBIAN_TARGETS
        assert len(coeffs) == len(jac["d"])
    i = jac["d"].index("gap_through_nm")
    assert jac["of"]["ln_kappa2_through_per_nm"][i] == pytest.approx(d["through"]["dln_kappa2_dgap_per_nm"])


# --------------------------------------------------------------------------------------------- slow: re-runs


@pytest.mark.slow
def test_solver_reproduces_femwell_and_bessel():
    """The FEM with formulation='isotropic' equals femwell.compute_modes(radius=R); 'exact' matches the analytic
    2-D Bessel eigenvalue."""
    from skfem import Basis, ElementTriP0
    from femwell.maxwell.waveguide import compute_modes
    from mrm.extraction import modes as MO
    xs = C.XSECS["strip400x200"]
    n = C.indices(xs, 1.31, 298.15)
    dom = C.Domain(-1.5, 2.5, -1.0, 1.0)
    mesh = C.build_mesh(xs, dom, C.MeshSpec(scale=3.0))
    P = C.FemProblem(mesh)
    mi = C.pick_mode(P.solve(n, 1.31, radius_um=7.5, formulation="isotropic", num_modes=2), "TE0")
    b0 = Basis(mesh, ElementTriP0())
    eps = b0.zeros() + n["clad"] ** 2
    eps[b0.get_dofs(elements="box")] = n["box"] ** 2
    eps[b0.get_dofs(elements="core")] = n["si"] ** 2
    mf = compute_modes(b0, eps, wavelength=1.31, num_modes=2, order=2, radius=7.5)
    te = max([m for m in mf if m.te_fraction > 0.5], key=lambda m: m.n_eff.real)
    assert np.real(mi.n_eff) == pytest.approx(np.real(te.n_eff), abs=1e-10)
    bb = MO.bessel_benchmark(MO.Settings.quick_settings())
    for r in bb["rows"]:
        if r["h_nm"] == 10.0:
            assert abs(r["err_exact"]) < 1e-6
            assert abs(r["err_isotropic"]) > 1e-3


@pytest.mark.slow
def test_modes_quick_rerun(tmp_path):
    r = subprocess.run([sys.executable, "-m", "mrm.extraction.modes", "--quick", "--study", "no", "--xsec",
                        "strip400x200", "--out", str(tmp_path)], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads((tmp_path / "modes_strip400x200.json").read_text())
    chk = d["index_weight_check"]
    assert abs(chk["residual"]) < 1e-4
    assert C.C0 / (d["n_g"] * 2 * math.pi * d["r_ref_um"] * 1e-6) == pytest.approx(1.8e12, rel=2e-4)
    committed = DATA / "modes_strip400x200.json"
    if committed.exists():
        c = json.loads(committed.read_text())
        assert d["n_g"] == pytest.approx(c["n_g"], abs=2e-3)          # coarse mesh vs production
        assert d["r_ref_um"] == pytest.approx(c["r_ref_um"], rel=1e-3)


@pytest.mark.slow
def test_coupling_quick_rerun(tmp_path):
    r = subprocess.run([sys.executable, "-m", "mrm.extraction.coupling", "--quick", "--xsec", "rib160", "--out",
                        str(tmp_path)], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    d = json.loads((tmp_path / "coupling_rib160.json").read_text())
    k2 = d["kappa2_vs_gap_lambda0"]["kappa2"]
    assert all(a > b for a, b in zip(k2, k2[1:]))
    assert all(abs(p - 1) < 0.01 for p in d["kappa2_vs_gap_lambda0"]["power_out_over_in"])
