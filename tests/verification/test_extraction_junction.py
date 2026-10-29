"""Checks the junction extraction (mrm.extraction.junction) and its committed artefacts (schema junction/1).

Read this first
---------------
The junction extraction solves the PN junction of the modulator with DEVSIM and the optical mode with femwell. It
writes, per cross-section, a table against reverse voltage (V): dn_eff, the free-carrier loss alpha_fca (1/m), the
capacitance (F/m) and the series resistance (ohm m). See docs/ARCHITECTURE.md, "Junction: junction_<xsec>.json".

The fast tests load the committed artefacts mrm/data/extracted/junction_<xsec>.json and check their identities: the
sha256 against the sidecar, the contract keys and grid, physical monotonicity, the cross-checks recorded at
extraction time (DEVSIM against the abrupt depletion approximation, the first-order overlap against a direct
re-solve, the index-weight identity, the AC capacitance against dQ/dV), the convergence error estimates and the CAL-5
fill identity (CAL-5: the modulation efficiency target). They also check the closed-form helpers without any solver:
the small-signal TCMT response against the table in docs/research/03_mrm_compact_models.md, and the abrupt-junction
formulas against docs/research/02_materials.md, "Junction physics".

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
# the folder of the junction artefacts (MRM_JUNCTION_ARTEFACT_DIR overrides it)
DATA = pathlib.Path(os.environ.get("MRM_JUNCTION_ARTEFACT_DIR", REPO / "mrm" / "data" / "extracted"))
XSECS = ("rib160", "rib200")
VARIANTS = ("", "_lateral")          # default topology (L-shape) and the lateral alternative


def _load_artefact(name):
    """(artefact dict, sidecar dict, artefact path) of junction_<name>.json; skips the test if it is not extracted."""
    p = DATA / f"junction_{name}.json"
    s = DATA / f"junction_{name}.sidecar.json"
    if not p.exists() or not s.exists():
        pytest.skip(f"artefact junction_{name}.json not extracted yet")
    return json.loads(p.read_text()), json.loads(s.read_text()), p


@pytest.fixture(params=[x + v for x in XSECS for v in VARIANTS])
def artefact(request):
    """Each committed junction artefact: both cross-sections, default (L-shape) and lateral topology."""
    return _load_artefact(request.param)


@pytest.fixture(params=XSECS)
def default_artefact(request):
    """The default-topology (L-shape) artefact of each cross-section."""
    return _load_artefact(request.param)


# -------------------------------------------------------------------------------- artefact contract and runtime loading

def test_sidecar_sha256_and_contract_keys(artefact):
    a, side, p = artefact
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


def test_bias_grid_contract(artefact):
    a, _, _ = artefact
    v = np.array(a["v_reverse_v"])
    assert v[0] == pytest.approx(-0.5) and v[-1] == pytest.approx(4.0)
    assert np.all(np.diff(v) > 0)
    assert np.max(np.diff(v)) <= 0.05 + 1e-9
    for k in ("dneff", "alpha_fca_per_m", "capacitance_f_per_m", "r_series_ohm_m"):
        assert len(a[k]) == v.size, k
        assert np.all(np.isfinite(a[k])), k


def test_runtime_loader_reads_the_artefact(artefact):
    from mrm.plant.electrical import load_junction_table
    a, _, p = artefact
    t = load_junction_table(p)
    v = np.array(a["v_reverse_v"])
    out = t.eval(np.array([[0.0, 1.0, 2.0]]), fill=1.0, length_m=40e-6)
    assert out["dneff"][0, 0] == pytest.approx(0.0, abs=1e-12)
    assert out["dneff"][0, 1] == pytest.approx(np.interp(1.0, v, a["dneff"]), rel=1e-9)
    assert out["c_f"][0, 1] == pytest.approx(40e-6 * np.interp(1.0, v, a["capacitance_f_per_m"]), rel=1e-9)
    assert out["r_s_ohm"][0, 1] == pytest.approx(np.interp(1.0, v, a["r_series_ohm_m"]) / 40e-6, rel=1e-9)


def test_runtime_realize_in_lut_mode(default_artefact):
    """The g26 preset switched to LUT mode (fidelity.junction_model: lut, ring.junction.lut_file -> this artefact,
    analytic-only keys removed) loads, verifies the sha256 through the {from:} machinery and realizes; the
    ring-level values follow the LUT: C_j = f L c', R_s = r'/(f L), eff_at_0v = f (lambda/n_g) d(dn_eff)/dV at 0 V."""
    from mrm.params import config
    from mrm.params.realize import realize
    from mrm.plant.electrical import junction_table_from_plant_params
    a, _, p = default_artefact
    d = config.yaml_load((REPO / "mrm" / "params" / "presets" / "g26.yaml").read_text())
    d["fidelity"]["junction_model"] = "lut"
    j = d["ring"]["junction"]
    for k in ("eff_at_0v_pm_per_v", "v_bi_v", "depletion_exponent", "dqi_dv_frac_per_v"):
        j.pop(k, None)
    j["lut_file"] = f"extracted/{p.name}"
    fill = a["reports"]["cal5"]["nominal_fill"]
    cfg = config.load(d, overrides={"ring.junction.fill_fraction": fill}, data_root=DATA.parent)
    pp = realize(cfg, batch=1)
    assert int(pp.junction.model) == 1
    v = np.array(a["v_reverse_v"])
    assert np.array_equal(pp.junction.lut_v_reverse_v, v)
    assert np.array_equal(pp.junction.lut_dneff, np.array(a["dneff"]))
    t = junction_table_from_plant_params(pp)
    L = float(np.asarray(pp.optics.length_m).ravel()[0])
    out = t.eval(np.array([[1.0]]), fill=fill, length_m=L)
    assert out["c_f"][0, 0] == pytest.approx(fill * L * np.interp(1.0, v, a["capacitance_f_per_m"]), rel=1e-9)
    assert out["r_s_ohm"][0, 0] == pytest.approx(np.interp(1.0, v, a["r_series_ohm_m"]) / (fill * L), rel=1e-9)
    lam = float(np.asarray(pp.optics.lambda_res_cold_m).ravel()[0])
    ng = float(np.asarray(pp.optics.n_g).ravel()[0])
    slope0 = np.interp(0.0, v, np.gradient(np.array(a["dneff"]), v))
    assert float(np.asarray(pp.junction.eff_at_0v_m_per_v).ravel()[0]) == pytest.approx(fill * lam * slope0 / ng,
                                                                                        rel=1e-3)


# ----------------------------------------------------------------------------------------- physics of the stored tables

def test_dneff_is_a_monotone_depletion_curve(artefact):
    """Reverse bias removes carriers: dn_eff rises, with a falling slope (the depletion law);
    forward bias adds carriers: dn_eff < 0."""
    a, _, _ = artefact
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


def test_absorption_capacitance_resistance_signs(artefact):
    a, _, _ = artefact
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


def test_loss_to_index_ratio_matches_plasma_dispersion(artefact):
    """q = -lambda d(alpha)/dV / (4 pi d(dn_eff)/dV): the compact-model notes estimate 0.06 for both carriers at
    1e18 cm^-3 (the plant carries the range 0.03-0.15)."""
    a, _, _ = artefact
    v = np.array(a["v_reverse_v"])
    q = np.interp(1.0, v, a["aux"]["loss_ratio_q"])
    assert 0.03 < q < 0.15


def test_absolute_alpha_is_inside_the_loss_budget(artefact):
    """Loss budget: the free-carrier loss at the CAL-5 fill must not exceed the intrinsic loss implied by Q_i."""
    a, _, _ = artefact
    lb = a["reports"]["loss_budget"]
    row0 = [r for r in lb["rows"] if r["v_rev"] == 0.0][0]
    assert row0["alpha_fca_ring_per_m"] <= lb["alpha_i_per_m"][0] * 1.0001
    assert 0.0 < row0["eta_fca_Qi7000"] <= 1.0


# ----------------------------------------------------------------------------- cross-checks recorded at extraction time

def test_devsim_depletion_width_against_abrupt_formula(artefact):
    """As in the DEVSIM tool check C7c (docs/research/05_tools.md): DEVSIM's half-density depletion edges sit 5-8 %
    inside the abrupt approximation (Debye tails)."""
    a, _, _ = artefact
    dc = a["depletion_check"]
    rel = np.array(dc["rel_diff_vs_abrupt"], float)
    assert np.all(rel < 0.0)
    assert np.all(np.abs(rel) < 0.12)
    assert abs(dc["mean_rel_diff_vs_debye"]) < abs(dc["mean_rel_diff_vs_abrupt"])


def test_overlap_perturbation_against_direct_resolve(artefact):
    _, side, _ = artefact
    c = side["validation"]["perturbation_vs_direct_resolve"]
    assert np.all(np.abs(c["rel_diff"]) < 0.02)
    assert np.all(np.abs(c["alpha_rel_diff"]) < 0.02)


def test_weight_integral_and_index_identity(artefact):
    _, side, _ = artefact
    w = side["validation"]["weight_integral"]
    assert abs(w["rel_diff_weight_vs_fd"]) < 1e-3                  # the weight density integrates to dn_eff/dn_Si
    assert abs(w["rel_diff_nodes_vs_femwell"]) < 2e-3              # and DEVSIM's node quadrature reproduces it
    assert abs(side["validation"]["index_weight_identity"]["rel_residual"]) < 1e-4   # sum n_g,i w_i = n_g


def test_capacitance_ac_against_charge_derivative(artefact):
    _, side, _ = artefact
    c = side["validation"]["capacitance_ac_vs_dqdv"]
    assert c["max_rel_diff_reverse"] < 0.01
    assert c["electron_hole_balance_max_rel"] < 0.01


def test_tidy3d_cross_check(artefact):
    """Independent mode solver against femwell (straight guide). The junction's own Tidy3D stage (uniform grids, no
    subpixel averaging) is used through its extrapolation only if a power law fits it; otherwise its finest grid must
    agree with femwell and the modes artefact's independent Tidy3D run (interface-aligned grids, fitted order) must
    agree within its own extrapolation error. The reassessment is recomputed here from the stored rows."""
    from mrm.extraction.junction import tidy3d_reassessment
    _, side, _ = artefact
    t = side["validation"].get("tidy3d_vs_femwell", {})
    if "error" in t or not t:
        pytest.skip("tidy3d cross-check not available")
    ra = t.get("reassessment")
    assert ra is not None, "run: python -m mrm.extraction.junction --postprocess"
    mcc = None
    if "modes_artefact_tidy3d_straight" in ra:
        mp = DATA / f"modes_{side['inputs']['geometry']['name']}.json"
        mcc = json.loads(mp.read_text())["crosscheck"]
    again = tidy3d_reassessment(t, t["femwell_straight"], mcc)
    assert again["stored_extrapolation_valid"] == ra["stored_extrapolation_valid"]
    assert again["delta_n_eff_finest"] == pytest.approx(ra["delta_n_eff_finest"], rel=1e-9)
    if ra["stored_extrapolation_valid"]:
        assert abs(t["delta_n_eff"]) < 2e-3
        assert abs(t["delta_dneff_dnsi"]) < 5e-3
    else:
        # no positive order fits: d12/d23 below its p -> 0 limit ln(h1/h2)/ln(h2/h3)
        assert not ra["order_test_n_eff"]["power_law_possible"]
        assert ra["order_test_n_eff"]["d12_over_d23"] < ra["order_test_n_eff"]["minimum_for_any_positive_order"]
        assert abs(ra["delta_n_eff_finest"]) < 2e-3
        assert abs(ra["delta_dneff_dnsi_finest"]) < 5e-3
        ms = ra["modes_artefact_tidy3d_straight"]
        assert abs(ms["delta_vs_junction_femwell_straight"]) < max(2e-3, ms["error_estimate"])
        assert ra["status"].startswith("consistent")


def _modes_comparison(side):
    comp = side["validation"].get("modes_artefact_consistency") or {}
    if "straight" not in comp:
        pytest.skip("no modes-artefact comparison (run --postprocess)")
    mp = DATA / f"modes_{side['inputs']['geometry']['name']}.json"
    if not mp.exists() or hashlib.sha256(mp.read_bytes()).hexdigest() != comp["modes_sha256"]:
        pytest.skip("the modes artefact changed since the junction comparison; re-run --postprocess (and consider "
                    "re-extracting the junction: its r_ref and n_g come from the modes artefact)")
    return comp


def test_modes_artefact_consistency(artefact):
    """Two independent femwell set-ups (this extraction and mrm.extraction.modes) agree on the straight guide; the
    bent rows differ by the bend formulation (exact transform vs femwell's isotropic map), recorded separately."""
    _, side, _ = artefact
    comp = _modes_comparison(side)
    st = comp["straight"]
    assert abs(comp["delta_r_ref_um"]) < 1e-5
    assert abs(st["delta_n_eff"]) < 2e-4
    assert abs(st["rel_diff_n_g"]) < 1e-4
    assert abs(st["rel_diff_index_weight_si"]) < 1e-4
    assert abs(comp["rel_diff_n_g"]) < 2e-3 and abs(comp["rel_diff_index_weight_si"]) < 2e-3
    bs = comp["bend_shift_n_eff"]
    assert bs["junction_isotropic"] == pytest.approx(bs["modes_isotropic"], rel=0.01)   # same formulation agrees
    assert bs["modes_exact"] == pytest.approx(bs["tidy3d_finest"], rel=0.02)             # exact transform = Tidy3D


def test_bend_formulation_bias_is_recorded_and_bounded(artefact):
    """The isotropic map overstates the bend effect; the estimated bias of the stored overlap lies inside the hard
    bounds (straight mode, k = 0; isotropic, k = 1) and is recomputable from the stored data."""
    from mrm.extraction.junction import bend_formulation_bias
    a, side, _ = artefact
    comp = _modes_comparison(side)
    bf = side["uncertainty"]["model"]["bend_formulation"]
    k_lo, k_hi = bf["k_range"]
    assert 0.0 < k_lo < bf["k_from_n_g_bend_change"] < k_hi < 1.0
    lo, hi = bf["bias_of_stored"]["dneff_1v6_rel"]
    hlo, hhi = bf["bias_of_stored"]["dneff_1v6_rel_hard_bound"]
    assert hlo <= lo <= hi <= hhi <= 0.0
    lo, hi = bf["bias_of_stored"]["alpha_fca_0v_rel"]
    hlo, hhi = bf["bias_of_stored"]["alpha_fca_0v_rel_hard_bound"]
    assert 0.0 <= hlo <= lo <= hi <= hhi
    f = bf["consequence_for_cal5"]["fill_endpoint_exact_estimate"]
    assert f[0] <= f[1] < bf["consequence_for_cal5"]["fill_endpoint_stored"]
    again = bend_formulation_bias(a, comp)
    assert again["bias_of_stored"]["dneff_1v6_rel"] == pytest.approx(bf["bias_of_stored"]["dneff_1v6_rel"], rel=1e-9)


def test_mesh_code_reproduces_the_committed_devsim_mesh(artefact):
    """The DEVSIM mesh generator still builds the meshes the artefact was solved on (node counts of the ladder and
    of the production mesh), so --mesh-report memory estimates refer to the same meshes."""
    from mrm.extraction.junction import XSECS, Doping, mesh_counts
    a, side, _ = artefact
    d = a["doping"]
    dop = Doping(**{k: d[k] for k in Doping.__dataclass_fields__})
    xs = XSECS[a["xsec"]["name"]]
    dm = side["convergence"]["devsim_mesh"]
    c0 = mesh_counts(xs, dop, dm["scales"][0])
    assert c0["nodes_si"] == dm["nodes_si"][0]
    assert mesh_counts(xs, dop, side["mesh"]["devsim"]["scale"])["nodes_si"] == side["mesh"]["devsim_nodes_si"]


def test_memory_record_and_not_done_list(artefact):
    from mrm.extraction.junction import EXPECTED_ARTEFACTS, artefact_name, estimate_peak_gb
    _, side, _ = artefact
    names = {artefact_name(x, t) + ".json" for x, t in EXPECTED_ARTEFACTS}
    for row in side["not_done"]:
        assert row["artefact"] in names and not (DATA / row["artefact"]).exists()
        assert "--heavy" in row["command_as_committed_settings"] and row["runtime_fallback"]
    present = {n for n in names if (DATA / n).exists()}
    assert present | {r["artefact"] for r in side["not_done"]} == names
    mem = side["mesh"].get("memory")
    if mem is None:
        return
    red = mem["reductions"][0]
    assert red["intended"]["devsim_scale"] < red["used"]["devsim_scale"]
    assert red["intended_mesh"]["nodes_si"] == 14862
    # the peak-memory model reproduces the recorded runs (2026-10-09 logs)
    assert estimate_peak_gb(14862)[0] == pytest.approx(2.86, abs=1e-9)        # calibration (killed at 2.86 GB)
    assert estimate_peak_gb(7880)[0] <= 2.00                                   # L-shape 2.0: whole run 2.00 GB
    assert estimate_peak_gb(13346)[0] == pytest.approx(2.73, abs=0.2)         # lateral rib200 1.0: killed 2.73 GB
    assert max(estimate_peak_gb(red["intended_mesh"]["nodes_si"])) < 6.0 * 0.6  # intended mesh: margin under 6 GB


def test_cli_settings_override():
    from mrm.extraction.junction import DIRECT_MODE_RES_FLOOR_UM, PROD, devsim_settings
    base = devsim_settings("lshape")
    assert base["scale"] == PROD["lshape"]["scale"] and base["direct_mode_res_floor_um"] == DIRECT_MODE_RES_FLOOR_UM
    s = devsim_settings("lshape", prod={"scale": 1.4142, "ladder": [2.8284, 2.0, 1.4142], "study": 2.0},
                        direct_res_floor_um=0.010)
    assert s == {"scale": 1.4142, "ladder": [2.8284, 2.0, 1.4142], "study": 2.0, "direct_mode_res_floor_um": 0.010}
    with pytest.raises(ValueError):
        devsim_settings("lshape", prod={"ladder": [1.0, 2.0, 3.0]})


def test_cli_passes_overrides_to_extract(monkeypatch, tmp_path):
    from mrm.extraction import junction
    seen = {}

    def fake_extract(name, **kw):
        seen.update(kw, name=name)
        raise RuntimeError("stop")                    # no solver: main() reports the failure and returns 1
    monkeypatch.setattr(junction, "extract", fake_extract)
    rc = junction.main(["--xsec", "rib160", "--topology", "lshape", "--scale", "1.4142", "--ladder", "2.8284", "2.0",
                        "1.4142", "--study-scale", "2.0", "--direct-mode-res-floor", "0.01", "--out", str(tmp_path)])
    assert rc == 1
    assert seen["prod"] == {"scale": 1.4142, "ladder": [2.8284, 2.0, 1.4142], "study": 2.0}
    assert seen["direct_res_floor_um"] == 0.01 and seen["light"] is False


def test_free_order_view_and_forced_flags(artefact):
    """The free-order views are recomputable from the stored levels, and every stored entry whose order no level
    sequence supports carries order_forced (its Richardson error is then an assumption, not evidence)."""
    import copy

    from mrm.extraction.junction import free_order_rows
    _, side, _ = artefact
    for blk in ("devsim_mesh", "mode_mesh"):
        b = side["convergence"][blk]
        fv = b.get("free_order_view")
        if fv is None:
            pytest.skip("run: python -m mrm.extraction.junction --postprocess")
        again = free_order_rows(copy.deepcopy({k: v for k, v in b.items() if k != "free_order_view"}))
        assert set(again) == set(fv["rows"])
        for k, row in again.items():
            st = fv["rows"][k]
            assert st["order_forced_in_stored_entry"] == row["order_forced_in_stored_entry"], (blk, k)
            assert st["last3_spread_rel"] == pytest.approx(row["last3_spread_rel"], rel=1e-9)
            if row["order_forced_in_stored_entry"]:
                assert b[k].get("order_forced") is True, (blk, k)
                assert row["observed_order"] is None
        # the supported error scale (spread of the levels) stays far below the model spreads (several %)
        assert max(r["all_levels_spread_rel"] for r in fv["rows"].values()) < 5e-3
    for k, row in side["uncertainty"]["numerical"].items():
        if isinstance(row, dict) and row.get("devsim_mesh_order_forced"):
            assert "devsim_mesh_rel_spread" in row, k


def test_richardson_rejects_out_of_range_orders():
    """A geometric ladder whose apparent order is above 8 (the first increment dominates) is not 'observed'; with
    an order given it is forced and flagged."""
    from mrm.extraction.junction import richardson
    h = 2.8284 / 1.18921 ** np.arange(3)                              # exactly geometric, ratio 2^0.25
    vals = [2.9848037e-4, 2.9827757e-4, 2.9823436e-4]                  # committed L-shape ladder, apparent order 8.9
    assert richardson(vals, h)["observed_order"] is None
    r = richardson(vals, h, order=2)
    assert r["order_forced"] is True and r["observed_order"] == 2


def test_free_order_pattern_names_the_side_of_the_accepted_range():
    """A monotone, shrinking sequence without an observed order is labelled by the side of [0.2, 8] that its
    apparent order is on: above 8 (the first increment dominates) or below 0.2 (the increments shrink too slowly)."""
    from mrm.extraction.junction import free_order_rows
    h = list(2.8284 / 1.18921 ** np.arange(3))
    rows = free_order_rows({"fast": {"values": [2.9848037e-4, 2.9827757e-4, 2.9823436e-4], "h": h},   # order 8.9
                            "slow": {"values": [1.0, 0.9, 0.81], "h": [4.0, 2.0, 1.0]}})               # order 0.15
    assert rows["fast"]["apparent_order"] > 8 and rows["fast"]["pattern"].startswith("apparent order above 8")
    assert rows["slow"]["apparent_order"] < 0.2 and rows["slow"]["pattern"].startswith("apparent order below 0.2")
    assert rows["fast"]["observed_order"] is None and rows["slow"]["observed_order"] is None


def test_temperature_notes_name_the_reference_temperature():
    """The temperature-sensitivity notes that extract() writes name T_REF_K (298.15 K): the cold DEVSIM run, the
    table it is compared with and the optical mode are all at T_REF_K, not at 300 K."""
    import inspect

    from mrm.extraction import junction
    src = inspect.getsource(junction.extract)
    assert "held at 300 K" not in src and "300 K table" not in src
    assert src.count("T_REF_K ({T_REF_K} K)") == 2


def test_assumption_range_notes_and_remesh_noise_are_recomputable(artefact):
    """inputs.range_notes and uncertainty.jacobian_remesh_noise follow from the stored data (no solver); every
    assumed doping and geometry value lies inside the range the current code assigns to it."""
    from mrm.extraction.junction import jacobian_remesh_noise, range_notes
    a, side, _ = artefact
    rn = side["inputs"].get("range_notes")
    if rn is None:
        pytest.skip("run: python -m mrm.extraction.junction --postprocess")
    assert range_notes(a, DATA) == rn
    for k, lohi in rn["suggested_ranges"]["doping"].items():
        if isinstance(a["doping"].get(k), (int, float)):
            assert lohi[0] <= a["doping"][k] <= lohi[1], k
        else:
            assert a["doping"][k] in lohi, k
    for k, lohi in rn["suggested_ranges"]["xsec"].items():
        assert lohi[0] <= a["xsec"][k] <= lohi[1], k
    jn = side["uncertainty"].get("jacobian_remesh_noise")
    if jn is not None:
        again = jacobian_remesh_noise(a, side)
        assert again["above_0p2"] == jn["above_0p2"]
        for row in jn["noise_over_value"].values():                     # same mesh: no re-meshing noise
            for k in ("log10_na_cm3", "log10_nd_cm3", "grading_nm", "outer_side"):
                assert row.get(k, 0.0) == 0.0


def test_report_identities_against_the_lut(artefact):
    """Re-derive the reports from the LUT columns (no solver): ring C_j = f L c', R_s = r'/(f L), f_RC, the nominal
    EO bandwidths (RC x small-signal TCMT), the loss-budget rows, and C = dQ/dV against the removed charge."""
    from mrm.extraction.junction import eo_bandwidth_ghz
    a, _, _ = artefact
    v = np.array(a["v_reverse_v"])
    col = {k: np.array(a[k]) for k in ("alpha_fca_per_m", "capacitance_f_per_m", "r_series_ohm_m")}
    q = np.array(a["aux"]["loss_ratio_q"])
    bw = a["reports"]["eo_bandwidth"]
    for c in bw["cases"]:
        fl = c["fill"] * c["L_um"] * 1e-6
        cj = fl * np.interp(c["v_rev"], v, col["capacitance_f_per_m"])
        rs = np.interp(c["v_rev"], v, col["r_series_ohm_m"]) / fl
        assert c["C_j_fF"] == pytest.approx(cj * 1e15, rel=1e-3)
        assert c["R_s_ohm"] == pytest.approx(rs, rel=1e-3)
        assert c["f_rc_ghz"] == pytest.approx(1 / (2 * math.pi * (rs + c["R_src_ohm"]) * cj) / 1e9, rel=1e-3)
        if (c["length"].startswith("L_from_FSR") and c["R_src_ohm"] == 50.0 and c["x_detune"] > 0
                and abs(c["x_detune"] - 1 / math.sqrt(3)) < 1e-3):
            f3 = eo_bandwidth_ghz((rs + 50.0) * cj, q_loaded=c["Q_L"], x_detune=1 / math.sqrt(3),
                                  q_loss=float(np.interp(c["v_rev"], v, q)))
            assert bw["nominal"][f"{c['v_rev']:g}V_QL{c['Q_L']:.0f}"] == pytest.approx(f3, abs=0.02)
    lb = a["reports"]["loss_budget"]
    f = a["reports"]["cal5"]["nominal_fill"]
    for row in lb["rows"]:
        al = np.interp(row["v_rev"], v, col["alpha_fca_per_m"])
        assert row["alpha_fca_ring_per_m"] == pytest.approx(f * al, rel=1e-3)
        assert row["eta_fca_Qi7000"] == pytest.approx(f * al / lb["alpha_i_per_m"][0], rel=2e-3)
    # C(V) is dQ/dV: its integral from 0 V equals the charge of the electrons removed from the silicon
    e = np.array(a["aux"]["electrons_per_m"])
    i0 = int(np.argmin(np.abs(v)))
    c_ = col["capacitance_f_per_m"]
    qc = np.concatenate([[0.0], np.cumsum(0.5 * (c_[1:] + c_[:-1]) * np.diff(v))])
    qc -= qc[i0]
    for vv in (1.0, 2.0, 4.0):
        j = int(np.argmin(np.abs(v - vv)))
        assert qc[j] == pytest.approx(1.602176634e-19 * (e[i0] - e[j]), rel=2e-3)


def test_convergence_error_estimates(artefact):
    _, side, _ = artefact
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


# -------------------------------------------------------------------------------------------------------------- reports

def test_cal5_fill_reproduces_the_target(artefact):
    """With the reported endpoint fill, (lambda/n_g) f dn_eff(1.6 V) / 1.6 V is exactly the CAL-5 52.8 pm/V; the fill
    scales with n_g across the hypotheses (dlambda = lambda/n_g f dn_eff)."""
    a, _, _ = artefact
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


def test_default_topology_reaches_cal5_with_fill_below_one(default_artefact):
    """The default artefact must reproduce CAL-5 with a physical fill fraction (schema bounds 0..1); the lateral
    alternative records why it cannot (fill > 1 at any doping up to 5e18, reports.doping_sweep of the lateral file)."""
    a, _, _ = default_artefact
    assert a["doping"]["topology"] == "lshape"
    h = a["reports"]["cal5"]["hypotheses"]["H_solver"]
    assert 0.3 < h["fill_endpoint"] <= 1.0


def test_bandwidth_report_has_both_biases(artefact):
    a, _, _ = artefact
    bw = a["reports"]["eo_bandwidth"]
    v1 = [c["f3db_ghz"] for c in bw["cases"] if c["v_rev"] == 1.0]
    v2 = [c["f3db_ghz"] for c in bw["cases"] if c["v_rev"] == 2.0]
    assert len(v1) == len(v2) >= 8
    assert all(f is not None and 5.0 < f < 200.0 for f in v1 + v2)
    assert bw["junction_rc_only"]["ratio_2v_over_1v"] > 1.0       # C falls with reverse bias


def test_jacobian_present_and_finite(artefact):
    a, _, _ = artefact
    j = a["jacobian"]
    for d in ("width_nm", "film_nm", "slab_nm", "log10_na_cm3", "log10_nd_cm3", "junction_offset_nm"):
        assert d in j["d"], d
    for k, vals in j["of"].items():
        assert all(x is not None and math.isfinite(x) for x in vals), k
    i = j["d"].index("log10_na_cm3")
    assert j["of"]["alpha_fca_0v_per_m"][i] > 0                     # more doping, more absorption


# -------------------------------------------------------------------------------------- closed-form helpers (no solver)

def test_small_signal_response_reproduces_research_table():
    """The small-signal table of the compact-model notes (verified rows), Q_L 3500, critical coupling: 55.2 GHz at
    x = 1/sqrt(3), 75.7 GHz at 0.813, 38.7 GHz at 0.30. With loss modulation q = +0.13 / -0.1 at x = 1/sqrt(3):
    51.3 / 58.7 GHz."""
    from mrm.extraction.junction import eo_bandwidth_ghz
    assert eo_bandwidth_ghz(0.0, q_loaded=3500, x_detune=1 / math.sqrt(3)) == pytest.approx(55.2, abs=0.1)
    assert eo_bandwidth_ghz(0.0, q_loaded=3500, x_detune=0.813) == pytest.approx(75.7, abs=0.1)
    assert eo_bandwidth_ghz(0.0, q_loaded=3500, x_detune=0.30) == pytest.approx(38.7, abs=0.1)
    pair = sorted([eo_bandwidth_ghz(0.0, q_loaded=3500, x_detune=s / math.sqrt(3), q_loss=0.13) for s in (1, -1)])
    assert pair[0] == pytest.approx(51.3, abs=0.15)
    pair = sorted([eo_bandwidth_ghz(0.0, q_loaded=3500, x_detune=s / math.sqrt(3), q_loss=-0.1) for s in (1, -1)])
    assert pair[1] == pytest.approx(58.7, abs=0.15)


def test_rc_pole_implied_by_cal10():
    """Compact-model verification log #3: 30.9 GHz measured at x = 1/sqrt(3) implies a 34.8 GHz RC pole (critical,
    Q 3500)."""
    from mrm.extraction.junction import eo_bandwidth_ghz
    tau = 1 / (2 * math.pi * 34.83e9)
    assert eo_bandwidth_ghz(tau, q_loaded=3500, x_detune=1 / math.sqrt(3)) == pytest.approx(30.9, abs=0.1)


def test_abrupt_junction_formulas():
    """Abrupt junction (docs/research/02_materials.md, "Junction physics"): V_bi(1e18, 1e18, 300 K) = 0.954 V;
    W = sqrt(2 eps (V_bi+V)/q (1/NA + 1/ND)) with eps 11.7."""
    from mrm.extraction.junction import XSECS, Doping, analytic_depletion
    d = Doping(na_cm3=1e18, nd_cm3=1e18)
    an = analytic_depletion(XSECS["rib160"], d, np.array([0.0, 2.0]), t_k=300.0)
    assert an["v_bi_v"] == pytest.approx(0.954, abs=1e-3)
    eps = 11.7 * 8.8541878128e-12
    w0 = math.sqrt(2 * eps * an["v_bi_v"] / 1.602176634e-19 * 2 / 1e24)
    assert an["w_nm"][0] == pytest.approx(w0 * 1e9, rel=1e-9)
    assert an["c_f_per_m"][1] == pytest.approx(eps * 160e-9 / (an["w_nm"][1] * 1e-9), rel=1e-9)


def test_doping_profile_levels():
    from mrm.extraction.junction import XSECS, DOPING_NOMINAL, doping_profile
    xs = XSECS["rib160"]
    na, nd = doping_profile(np.array([-100.0, 100.0, -(200 + 400.0), 200 + 400.0, -(200 + 900.0)]), xs, DOPING_NOMINAL)
    assert (na[0] == pytest.approx(DOPING_NOMINAL.na_cm3, rel=1e-6)
            and nd[1] == pytest.approx(DOPING_NOMINAL.nd_cm3, rel=1e-6))
    assert (na[2] == pytest.approx(DOPING_NOMINAL.n_plus_cm3, rel=1e-3)
            and nd[3] == pytest.approx(DOPING_NOMINAL.n_plus_cm3, rel=1e-3))
    assert na[4] == pytest.approx(DOPING_NOMINAL.n_pp_cm3, rel=1e-3)
    assert nd[0] < 1e3 and na[1] < 1e3


def test_richardson_recovers_a_known_order():
    from mrm.extraction.junction import richardson
    h = np.array([0.04, 0.02, 0.01])
    r = richardson(1.0 + 3.0 * h ** 2, h)
    assert r["observed_order"] == pytest.approx(2.0, abs=1e-6)
    assert r["extrapolated"] == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------------------------------------------ slow: re-run the extraction

@pytest.mark.slow
def test_quick_rerun_matches_committed(tmp_path):
    a, _, _ = _load_artefact("rib160_lateral")
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
