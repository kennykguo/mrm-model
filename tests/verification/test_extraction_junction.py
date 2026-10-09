"""Verification of the junction extraction (mrm/extraction/junction.py; ARCHITECTURE 5.4, schema junction/1).

Fast tests load the committed artefacts mrm/data/extracted/junction_<xsec>.json and check their identities: sha256
against the sidecar, the contract keys and grid, physical monotonicity, the cross-checks recorded at extraction time
(DEVSIM against the abrupt depletion approximation, first-order overlap against a direct re-solve, the index-weight
identity, AC capacitance against dQ/dV), the convergence error estimates and the CAL-5 fill identity. They also check
the closed-form helpers (small-signal TCMT response against docs/research/03 section 5.6, the abrupt-junction
formulas against 02 section 11) without any solver.

The slow test re-runs the extraction with --quick into a temporary directory and compares it with the committed
artefact within the coarse-mesh tolerance.
"""
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = pathlib.Path(os.environ.get("MRM_JUNCTION_ARTEFACT_DIR", REPO / "mrm" / "data" / "extracted"))
XSECS = ("rib160", "rib200")
VARIANTS = ("", "_lateral")          # default topology (L-shape) and the lateral alternative


def _load(name):
    p = DATA / f"junction_{name}.json"
    s = DATA / f"junction_{name}.sidecar.json"
    if not p.exists() or not s.exists():
        pytest.skip(f"artefact junction_{name}.json not extracted yet")
    return json.loads(p.read_text()), json.loads(s.read_text()), p


@pytest.fixture(params=[x + v for x in XSECS for v in VARIANTS])
def art(request):
    return _load(request.param)


@pytest.fixture(params=XSECS)
def art_default(request):
    return _load(request.param)


# ---------------------------------------------------------------------------------------------- contract
def test_sidecar_sha256_and_contract_keys(art):
    a, side, p = art
    assert side["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert a["schema"] == side["schema"] == "junction/1"
    for k in ("schema", "sha256", "inputs", "tool_versions", "mesh", "validation", "convergence", "created_utc"):
        assert k in side, k
    for k in ("geometry", "doping", "materials"):
        assert side["inputs"][k]["provenance"] in ("assumed", "literature")
    assert side["inputs"]["materials"]["schema"] == "materials/1"
    for k in ("v_reverse_v", "dneff", "alpha_fca_per_m", "capacitance_f_per_m", "r_series_ohm_m", "doping",
              "method", "depletion_check"):
        assert k in a, k
    assert a["method"] == "devsim" and a["fill_fraction"] == 1.0 and a["full_circumference"] is True
    assert side["quick"] is False
    assert a["doping"]["topology"] in ("lshape", "lateral")


def test_bias_grid_contract(art):
    a, _, _ = art
    v = np.array(a["v_reverse_v"])
    assert v[0] == pytest.approx(-0.5) and v[-1] == pytest.approx(4.0)
    assert np.all(np.diff(v) > 0)
    assert np.max(np.diff(v)) <= 0.05 + 1e-9
    for k in ("dneff", "alpha_fca_per_m", "capacitance_f_per_m", "r_series_ohm_m"):
        assert len(a[k]) == v.size, k
        assert np.all(np.isfinite(a[k])), k


def test_runtime_loader_reads_the_artefact(art):
    from mrm.plant.electrical import load_junction_table
    a, _, p = art
    t = load_junction_table(p)
    v = np.array(a["v_reverse_v"])
    out = t.eval(np.array([[0.0, 1.0, 2.0]]), fill=1.0, length_m=40e-6)
    assert out["dneff"][0, 0] == pytest.approx(0.0, abs=1e-12)
    assert out["dneff"][0, 1] == pytest.approx(np.interp(1.0, v, a["dneff"]), rel=1e-9)
    assert out["c_f"][0, 1] == pytest.approx(40e-6 * np.interp(1.0, v, a["capacitance_f_per_m"]), rel=1e-9)
    assert out["r_s_ohm"][0, 1] == pytest.approx(np.interp(1.0, v, a["r_series_ohm_m"]) / 40e-6, rel=1e-9)


# ---------------------------------------------------------------------------------------------- physics
def test_dneff_is_a_monotone_depletion_curve(art):
    """Reverse bias removes carriers: dn_eff rises, with a falling slope (the depletion law, 07 P1-7 / 06 M2);
    forward bias adds carriers: dn_eff < 0."""
    a, _, _ = art
    v = np.array(a["v_reverse_v"])
    dn = np.array(a["dneff"])
    i0 = int(np.argmin(np.abs(v)))
    assert dn[i0] == pytest.approx(0.0, abs=1e-14)
    assert np.all(np.diff(dn) > 0)
    assert np.all(dn[v < 0] < 0)
    slope = np.diff(dn) / np.diff(v)
    rev = v[:-1] >= 0
    assert slope[rev][0] > 1.5 * slope[rev][-1]                    # efficiency falls with reverse bias
    assert np.all(np.diff(slope[rev]) < 1e-9)                      # concave on the reverse branch


def test_absorption_capacitance_resistance_signs(art):
    a, _, _ = art
    v = np.array(a["v_reverse_v"])
    al = np.array(a["alpha_fca_per_m"])
    c = np.array(a["capacitance_f_per_m"])
    r = np.array(a["r_series_ohm_m"])
    assert np.all(al > 0) and np.all(np.diff(al) < 0)              # absolute FCA falls as carriers leave
    assert np.all(c > 0) and np.all(np.diff(c[v >= 0]) < 0)
    assert np.all(r > 0)
    # C ~ (V + V_bi)^-m with m between the linearly graded 1/3 and the abrupt 1/2 (plus a constant fringe part)
    vbi = a["doping"]["v_bi_v"]
    m = -np.log(np.interp(3.0, v, c) / np.interp(0.5, v, c)) / np.log((3.0 + vbi) / (0.5 + vbi))
    assert 0.2 < m < 0.55


def test_loss_to_index_ratio_matches_plasma_dispersion(art):
    """q = -lambda d(alpha)/dV / (4 pi d(dn_eff)/dV): 03 section 5.4 estimates 0.06 for both carriers at 1e18 (range
    0.03-0.15 carried by the plant)."""
    a, _, _ = art
    v = np.array(a["v_reverse_v"])
    q = np.interp(1.0, v, a["aux"]["loss_ratio_q"])
    assert 0.03 < q < 0.15


def test_absolute_alpha_is_inside_the_loss_budget(art):
    """07 P0-2: the free-carrier loss at the CAL-5 fill must not exceed the intrinsic loss implied by Q_i."""
    a, _, _ = art
    lb = a["reports"]["loss_budget"]
    row0 = [r for r in lb["rows"] if r["v_rev"] == 0.0][0]
    assert row0["alpha_fca_ring_per_m"] <= lb["alpha_i_per_m"][0] * 1.0001
    assert 0.0 < row0["eta_fca_Qi7000"] <= 1.0


# ---------------------------------------------------------------------------------------------- cross-checks
def test_devsim_depletion_width_against_abrupt_formula(art):
    """05 C7c: DEVSIM's half-density depletion edges sit 5-8 % inside the abrupt approximation (Debye tails)."""
    a, _, _ = art
    dc = a["depletion_check"]
    rel = np.array(dc["rel_diff_vs_abrupt"], float)
    assert np.all(rel < 0.0)
    assert np.all(np.abs(rel) < 0.12)
    assert abs(dc["mean_rel_diff_vs_debye"]) < abs(dc["mean_rel_diff_vs_abrupt"])


def test_overlap_perturbation_against_direct_resolve(art):
    _, side, _ = art
    c = side["validation"]["perturbation_vs_direct_resolve"]
    assert np.all(np.abs(c["rel_diff"]) < 0.02)
    assert np.all(np.abs(c["alpha_rel_diff"]) < 0.02)


def test_weight_integral_and_index_identity(art):
    _, side, _ = art
    w = side["validation"]["weight_integral"]
    assert abs(w["rel_diff_weight_vs_fd"]) < 1e-3                  # the weight density integrates to dn_eff/dn_Si
    assert abs(w["rel_diff_nodes_vs_femwell"]) < 2e-3              # and DEVSIM's node quadrature reproduces it
    assert abs(side["validation"]["index_weight_identity"]["rel_residual"]) < 1e-4   # sum n_g,i w_i = n_g (07 P0-3)


def test_capacitance_ac_against_charge_derivative(art):
    _, side, _ = art
    c = side["validation"]["capacitance_ac_vs_dqdv"]
    assert c["max_rel_diff_reverse"] < 0.01
    assert c["electron_hole_balance_max_rel"] < 0.01


def test_tidy3d_cross_check(art):
    _, side, _ = art
    t = side["validation"].get("tidy3d_vs_femwell", {})
    if "error" in t or not t:
        pytest.skip("tidy3d cross-check not available")
    assert abs(t["delta_n_eff"]) < 2e-3
    assert abs(t["delta_dneff_dnsi"]) < 5e-3


def test_convergence_error_estimates(art):
    _, side, _ = art
    dm = side["convergence"]["devsim_mesh"]
    mm = side["convergence"]["mode_mesh"]
    assert len(dm["scales"]) >= 3 and len(mm["res_um"]) >= 3
    for k in ("dneff_1v6", "alpha_fca_0v_per_m", "c_1v_f_per_m", "r_series_1v_ohm_m"):
        e = dm[k]
        assert e["error_estimate"] is not None
        assert abs(e["error_estimate"] / e["extrapolated"]) < 0.01, k
    for k in ("dneff_1v6", "alpha_fca_0v_per_m", "n_eff"):
        e = mm[k]
        assert abs(e["error_estimate"] / e["extrapolated"]) < 0.01, k
    assert side["convergence"]["bias_grid"]["max_abs_error_rel_to_dneff_1v6"] < 1e-3


# ---------------------------------------------------------------------------------------------- reports
def test_cal5_fill_reproduces_the_target(art):
    """With the reported endpoint fill, (lambda/n_g) f dn_eff(1.6 V) / 1.6 V is exactly the CAL-5 52.8 pm/V; the fill
    scales with n_g across the hypotheses (dlambda = lambda/n_g f dn_eff)."""
    a, _, _ = art
    cal5 = a["reports"]["cal5"]
    v = np.array(a["v_reverse_v"])
    lam = cal5["lambda_nm"] * 1e-9
    dn16 = np.interp(1.6, v, a["dneff"])
    ratios = []
    for name, h in cal5["hypotheses"].items():
        shift = lam / h["n_g"] * h["fill_endpoint"] * dn16 * 1e12
        assert shift / 1.6 == pytest.approx(52.8, rel=2e-3), name
        ratios.append(h["fill_endpoint"] / h["n_g"])
    assert np.ptp(ratios) / np.mean(ratios) < 2e-3
    assert cal5["hypotheses"]["H_solver"]["lsq_rms_pm"] < 10.0      # the shape fits the read-off ladder


def test_default_topology_reaches_cal5_with_fill_below_one(art_default):
    """The default artefact must reproduce CAL-5 with a physical fill fraction (schema bounds 0..1); the lateral
    alternative records why it cannot (fill > 1 at any doping up to 5e18, reports.doping_sweep of the lateral file)."""
    a, _, _ = art_default
    assert a["doping"]["topology"] == "lshape"
    h = a["reports"]["cal5"]["hypotheses"]["H_solver"]
    assert 0.3 < h["fill_endpoint"] <= 1.0


def test_bandwidth_report_has_both_biases(art):
    a, _, _ = art
    bw = a["reports"]["eo_bandwidth"]
    v1 = [c["f3db_ghz"] for c in bw["cases"] if c["v_rev"] == 1.0]
    v2 = [c["f3db_ghz"] for c in bw["cases"] if c["v_rev"] == 2.0]
    assert len(v1) == len(v2) >= 8
    assert all(f is not None and 5.0 < f < 200.0 for f in v1 + v2)
    assert bw["junction_rc_only"]["ratio_2v_over_1v"] > 1.0       # C falls with reverse bias


def test_jacobian_present_and_finite(art):
    a, _, _ = art
    j = a["jacobian"]
    for d in ("width_nm", "film_nm", "slab_nm", "log10_na_cm3", "log10_nd_cm3", "junction_offset_nm"):
        assert d in j["d"], d
    for k, vals in j["of"].items():
        assert all(x is not None and math.isfinite(x) for x in vals), k
    i = j["d"].index("log10_na_cm3")
    assert j["of"]["alpha_fca_0v_per_m"][i] > 0                     # more doping, more absorption


# ---------------------------------------------------------------------------------------------- helpers (no solver)
def test_small_signal_response_reproduces_research_table():
    """03 section 5.6 (verified rows): Q_L 3500, critical coupling: 55.2 GHz at x = 1/sqrt(3), 75.7 GHz at 0.813,
    38.7 GHz at 0.30; with loss modulation q = +0.13 / -0.1 at x = 1/sqrt(3): 51.3 / 58.7 GHz."""
    from mrm.extraction.junction import eo_bandwidth_ghz
    assert eo_bandwidth_ghz(0.0, q_loaded=3500, x_detune=1 / math.sqrt(3)) == pytest.approx(55.2, abs=0.1)
    assert eo_bandwidth_ghz(0.0, q_loaded=3500, x_detune=0.813) == pytest.approx(75.7, abs=0.1)
    assert eo_bandwidth_ghz(0.0, q_loaded=3500, x_detune=0.30) == pytest.approx(38.7, abs=0.1)
    pair = sorted([eo_bandwidth_ghz(0.0, q_loaded=3500, x_detune=s / math.sqrt(3), q_loss=0.13) for s in (1, -1)])
    assert pair[0] == pytest.approx(51.3, abs=0.15)
    pair = sorted([eo_bandwidth_ghz(0.0, q_loaded=3500, x_detune=s / math.sqrt(3), q_loss=-0.1) for s in (1, -1)])
    assert pair[1] == pytest.approx(58.7, abs=0.15)


def test_rc_pole_implied_by_cal10():
    """03 verification log #3: 30.9 GHz measured at x = 1/sqrt(3) implies a 34.8 GHz RC pole (critical, Q 3500)."""
    from mrm.extraction.junction import eo_bandwidth_ghz
    tau = 1 / (2 * math.pi * 34.83e9)
    assert eo_bandwidth_ghz(tau, q_loaded=3500, x_detune=1 / math.sqrt(3)) == pytest.approx(30.9, abs=0.1)


def test_abrupt_junction_formulas():
    """02 section 11: V_bi(1e18, 1e18, 300 K) = 0.954 V; W = sqrt(2 eps (V_bi+V)/q (1/NA + 1/ND)) with eps 11.7."""
    from mrm.extraction.junction import XSECS, Doping, analytic_depletion
    d = Doping(na_cm3=1e18, nd_cm3=1e18)
    an = analytic_depletion(XSECS["rib160"], d, np.array([0.0, 2.0]), t_k=300.0)
    assert an["v_bi_v"] == pytest.approx(0.954, abs=1e-3)
    eps = 11.7 * 8.8541878128e-12
    w0 = math.sqrt(2 * eps * an["v_bi_v"] / 1.602176634e-19 * 2 / 1e24)
    assert an["w_nm"][0] == pytest.approx(w0 * 1e9, rel=1e-9)
    assert an["c_f_per_m"][1] == pytest.approx(eps * 160e-9 / (an["w_nm"][1] * 1e-9), rel=1e-9)


def test_doping_profile_levels():
    from mrm.extraction.junction import XSECS, DOPING_NOMINAL as D, doping_profile
    xs = XSECS["rib160"]
    na, nd = doping_profile(np.array([-100.0, 100.0, -(200 + 400.0), 200 + 400.0, -(200 + 900.0)]), xs, D)
    assert na[0] == pytest.approx(D.na_cm3, rel=1e-6) and nd[1] == pytest.approx(D.nd_cm3, rel=1e-6)
    assert na[2] == pytest.approx(D.n_plus_cm3, rel=1e-3) and nd[3] == pytest.approx(D.n_plus_cm3, rel=1e-3)
    assert na[4] == pytest.approx(D.n_pp_cm3, rel=1e-3)
    assert nd[0] < 1e3 and na[1] < 1e3


def test_richardson_recovers_a_known_order():
    from mrm.extraction.junction import richardson
    h = np.array([0.04, 0.02, 0.01])
    r = richardson(1.0 + 3.0 * h ** 2, h)
    assert r["observed_order"] == pytest.approx(2.0, abs=1e-6)
    assert r["extrapolated"] == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------------------------- slow: re-run
@pytest.mark.slow
def test_quick_rerun_matches_committed(tmp_path):
    a, _, _ = _load("rib160_lateral")
    cmd = [sys.executable, str(REPO / "tools" / "memguard.py"), "--limit-gb", "3", "--heavy", "--", sys.executable,
           "-m", "mrm.extraction.junction", "--quick", "--xsec", "rib160", "--topology", "lateral",
           "--out", str(tmp_path)]
    subprocess.run(cmd, check=True, cwd=REPO, capture_output=True, timeout=3600)
    q = json.loads((tmp_path / "junction_rib160_lateral.json").read_text())
    for vv in (1.0, 1.6, 4.0):
        assert np.interp(vv, q["v_reverse_v"], q["dneff"]) == pytest.approx(
            np.interp(vv, a["v_reverse_v"], a["dneff"]), rel=0.05)
        assert np.interp(vv, q["v_reverse_v"], q["capacitance_f_per_m"]) == pytest.approx(
            np.interp(vv, a["v_reverse_v"], a["capacitance_f_per_m"]), rel=0.05)
    assert np.interp(0.0, q["v_reverse_v"], q["alpha_fca_per_m"]) == pytest.approx(
        np.interp(0.0, a["v_reverse_v"], a["alpha_fca_per_m"]), rel=0.05)
