"""Verification of the thermal ROM artefacts (ARCHITECTURE 5.3, 07 P1-2, P1-3) and of the MOR/fit code.

Fast tests load the committed artefacts in mrm/data/extracted/ and check identities, the contract shapes and the
recorded verification numbers. Solver re-runs are marked slow.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.sparse as sp

from mrm.extraction import mor
from mrm.extraction import thermal_fem as tf

ALL = [(f, film) for f in tf.FAMILIES for film in tf.FILMS]
PRESENT = [a for a in ALL if tf.artefact_paths(*a)[0].exists() and tf.artefact_paths(*a)[1].exists()]


@pytest.fixture(params=PRESENT, ids=[f"{f}_{m}" for f, m in PRESENT])
def rom(request):
    return tf.load_rom(*request.param)        # raises if the sha256 does not match the sidecar


def test_default_artefact_committed():
    """inring_noundercut_160 is the default and must exist (07 P1-3: do it fully first)."""
    if not PRESENT:
        pytest.skip("no thermal ROM artefact extracted yet")
    assert ("inring_noundercut", 160) in PRESENT


def test_contract_shapes(rom):
    taus = rom["taus_s"]
    K = taus.size
    assert list(rom["sources"]) == list(tf.SOURCES)
    assert list(rom["observers"]) == list(tf.OBSERVERS)
    assert rom["self_res"].shape == (5, 4, K)
    D = rom["cross_r_um"].size
    assert rom["cross_res"].shape == (5, 4, D, K)
    assert np.all(np.diff(rom["cross_r_um"]) > 0)
    R = rom["sidecar"]["inputs"]["config"]["radius_um"]
    assert rom["cross_r_um"].min() > 2 * R           # radial table valid for d > 2R only
    names = list(rom["site_offset_names"])
    assert {"monitor_pd", "temp_sensor", "driver"} <= set(names)
    assert rom["site_offsets_um"].shape == (len(names), 2)
    assert rom["sidecar"]["schema"] == "thermal_rom/1"
    assert str(rom["schema"]) == "thermal_rom/1"


def test_pole_set_spans_10ns_to_10ms(rom):
    taus = rom["taus_s"]
    assert np.all(np.isfinite(taus)) and np.all(taus > 0)
    assert taus.min() <= 1e-8 * 1.0001 or rom["sidecar"]["quick"]
    assert taus.max() >= 1e-2
    assert np.all(np.diff(taus) > 0)                 # strictly increasing (runtime reader contract)


def test_residues_sum_to_fem_dc(rom):
    """sum_k r_k is the exact FEM DC gain (imposed by the constrained fit)."""
    np.testing.assert_allclose(rom["self_res"].sum(-1), rom["dc_self_k_per_w"], rtol=1e-9, atol=0)
    np.testing.assert_allclose(rom["cross_res"].sum(-1), rom["dc_cross_k_per_w"], rtol=1e-9, atol=0)


def test_dc_gains_positive_and_ordered(rom):
    """Heat raises the temperature everywhere (positive Green's function); crosstalk falls with distance; the ring is
    warmer than any remote site for its own heater."""
    assert np.all(rom["dc_self_k_per_w"] > 0)
    assert np.all(rom["dc_cross_k_per_w"] > 0)
    h, r = tf.SOURCES.index("heater"), tf.OBSERVERS.index("ring_mode_weighted")
    x = rom["dc_cross_k_per_w"][h, r]
    assert np.all(np.diff(x) < 0)
    assert rom["dc_self_k_per_w"][h, r] > x[0]
    for o in ("monitor_pd", "temp_sensor"):
        assert rom["dc_self_k_per_w"][h, tf.OBSERVERS.index(o)] < rom["dc_self_k_per_w"][h, r]


def test_collocated_steps_monotone(rom):
    """A driving-point thermal response (source and observer with the same weight, or nearly so) has a completely
    monotone step response; the fitted Foster sum must not overshoot or dip by more than 0.5 % of DC."""
    fm = mor.Foster(rom["taus_s"], rom["self_res"])
    t = np.logspace(-10, 1, 600)
    y = fm.step(t)
    for s, o in (("heater", "heater_resistor"), ("waveguide_absorption", "ring_mode_weighted")):
        yy = y[tf.SOURCES.index(s), tf.OBSERVERS.index(o)]
        dc = yy[-1]
        assert np.min(np.diff(yy)) > -5e-3 * dc
        assert yy.max() < dc * 1.005


def test_residues_well_conditioned(rom):
    """Driving-point pairs have non-negative residues (as their exact modal residues are), and no pair relies on
    large cancelling residues (poles are kept >= 1.5x apart): max_k |r_k| <= 8 |H(0)| for self pairs and <= 25 |H(0)|
    for cross pairs up to 300 um (their diffusion delay needs some mixed-sign residues)."""
    if "residue_cancellation" not in rom["sidecar"]["rom"]:
        pytest.skip("artefact fitted before the conditioning record existed")
    for s, o in (("heater", "heater_resistor"), ("waveguide_absorption", "ring_mode_weighted")):
        i, j = tf.SOURCES.index(s), tf.OBSERVERS.index(o)
        assert np.all(rom["self_res"][i, j] >= -1e-12 * rom["dc_self_k_per_w"][i, j])
    rc = rom["sidecar"]["rom"]["residue_cancellation"]
    assert rc["max_self"] <= tf.CANCEL_SELF and rc["max_cross_d_le_300um"] <= tf.CANCEL_CROSS, rc
    taus = rom["taus_s"]
    assert np.min(taus[1:] / taus[:-1]) >= 1.5 * (1 - 1e-6)


def test_step_and_frequency_forms_agree(rom):
    """Foster step response equals sum r (1 - exp(-t/tau)); H(0) equals the DC; H(f) -> 0 at high f."""
    taus, res = rom["taus_s"], rom["self_res"]
    fm = mor.Foster(taus, res)
    t = np.array([1e-9, 1e-6, 1e-3])
    ref = np.einsum("sok,kt->sot", res, 1 - np.exp(-t[None, :] / taus[:, None]))
    np.testing.assert_allclose(fm.step(t), ref, rtol=1e-10, atol=1e-10 * rom["dc_self_k_per_w"].max())
    np.testing.assert_allclose(fm.freq([0.0])[..., 0].real, rom["dc_self_k_per_w"], rtol=1e-12)
    assert np.all(np.abs(fm.freq([1e12])[..., 0]) < 1e-3 * rom["dc_self_k_per_w"])


def test_recorded_time_and_frequency_errors(rom):
    """The sidecar's full-FEM verification (contract: < 2 % of steady state from 10 ns to 10 ms, < 0.3 dB and < 3 deg
    from DC to 1 MHz, for the self term and the nearest-neighbour term)."""
    v = rom["sidecar"]["validation"]
    if rom["sidecar"]["quick"]:
        pytest.skip("quick artefact")
    for part in ("self", "nearest_neighbour"):
        d = v[part]
        assert d["max_time_err_frac_of_dc"] < 0.02, (part, d)
        assert d["max_db"] < 0.3, (part, d)
        assert d["max_deg"] < 3.0, (part, d)
    assert v["pass_self"] and v["pass_nearest_neighbour"]
    assert v["time_window_s"] == [1e-8, 1e-2]
    for m in v["rom_vs_fem"].values():
        assert m["max_abs_err_frac_of_dc"] < 1e-3
        assert m["fem_time_discretization_err_frac_of_dc"] < 2e-3


def test_mesh_convergence_recorded(rom):
    conv = rom["sidecar"]["mesh"]["convergence"]
    if rom["sidecar"]["quick"]:
        pytest.skip("quick artefact")
    assert len(conv["levels"]) >= 3
    assert conv["max_rel_err_est"] < 0.01, conv["worst_metric"]
    for k in ("z_heater_ring_k_per_mw", "z_opt_ring_k_per_mw", "t63_heater_ring_s"):
        assert conv["richardson"][k]["rel_err_est"] < 2e-3, k


def test_waveguide_core_has_sub_us_poles(rom):
    """07 P0-1 step 3: the self-heating path must carry poles below 1 us."""
    s, o = tf.SOURCES.index("waveguide_absorption"), tf.OBSERVERS.index("ring_mode_weighted")
    taus = rom["taus_s"]
    share = rom["self_res"][s, o][taus < 1e-6].sum() / rom["dc_self_k_per_w"][s, o]
    assert share > 0.05


def test_reciprocity_recorded(rom):
    """07 P1-2: Z_ij = Z_ji for like sources and observers (the FEM pencil is symmetric)."""
    r = rom["sidecar"]["reciprocity"]
    assert r["heater_source_equals_heater_observer"]
    assert r["rel_err"] < 1e-9


def test_z_ratio_and_diagnostics(rom):
    """Self-heating per absorbed watt is at least the heater's per watt for an external heater (07 P0-1 step 5,
    06 H2); rom_dc_scale is reported for every dlambda/dT hypothesis (07 P1-2)."""
    sm = rom["sidecar"]["summary"]
    assert sm["z_ratio_opt_heater"] > 0.95
    tab = rom["sidecar"]["rom_dc_scale"]["table"]
    assert any(k.startswith("H86|") for k in tab)
    for v in tab.values():
        assert v["rom_dc_scale"] > 0


def test_uncertainty_stage_recorded(rom):
    st = rom["sidecar"].get("stages", {})
    if "uncertainty" not in st:
        pytest.skip("uncertainty stage not run for this artefact")
    u = st["uncertainty"]
    assert "jacobian" in u and len(u["jacobian"]) >= 5
    assert all(np.isfinite(v["d_ln_dparam"]["z_heater_ring_k_per_mw"]) for v in u["jacobian"].values())
    pts = u["k_temp_scaling"]["points"]
    assert pts[0]["t_k"] == 300.0 and pts[-1]["t_k"] >= 375.0


def test_undercut_raises_gain_and_slows():
    """Literature ordering (04 T1-T3, 07 P1-3): an undercut multiplies the heater efficiency and slows the ring."""
    if not all(("silicide_undercut", f) in PRESENT and ("inring_noundercut", f) in PRESENT for f in (160,)):
        pytest.skip("artefacts missing")
    a = tf.load_rom("silicide_undercut", 160)["sidecar"]["summary"]
    b = tf.load_rom("inring_noundercut", 160)["sidecar"]["summary"]
    assert a["z_heater_ring_k_per_mw"] > 1.5 * b["z_heater_ring_k_per_mw"]
    assert a["t63_heater_ring_s"] > b["t63_heater_ring_s"]


def test_analytic_crosscheck_recorded():
    """The axisymmetric FEM machinery agrees with the exact layered solution (independent method)."""
    js = tf.artefact_paths("inring_noundercut", 160)[1]
    if not js.exists():
        pytest.skip("default artefact not extracted yet")
    st = json.loads(js.read_text()).get("stages", {})
    if "crosscheck_analytic" not in st:
        pytest.skip("analytic stage not run")
    for case, r in st["crosscheck_analytic"]["results"].items():
        assert r["max_err_of_self_dc"] < 2e-3, case


def _crosscheck_3d():
    js = tf.artefact_paths("inring_noundercut", 160)[1]
    if not js.exists():
        pytest.skip("default artefact not extracted yet")
    st = json.loads(js.read_text()).get("stages", {})
    if "crosscheck_3d" not in st:
        pytest.skip("3-D stage not run")
    return st["crosscheck_3d"]


def test_3d_crosscheck_recorded():
    """The 3-D formulation check and the bank record exist; a missing discretization/multibody split is declared as
    pending with the command that produces it (the re-run was interrupted)."""
    r = _crosscheck_3d()
    assert abs(r["single_ring_no_leads"]["rel_diff_z_heater_ring"]) < 0.05
    bank = r["bank"]
    assert "max_step_err_frac_all" in bank and bank["n_rings"] == 8
    if "max_multibody_step_err_frac_all" not in bank:
        assert "--stage 3d" in r["pending"]["decomposition"]["command"]


def test_superposition_recorded_consistently():
    """Every sidecar carries the default family's bank result under the flat keys the runtime reads, with the
    target and the pass flag (07 P1-3)."""
    c3d = _crosscheck_3d()
    ref = tf.superposition_record(c3d)
    pending = "max_multibody_step_err_frac_all" not in c3d["bank"]
    for fam, film in PRESENT:
        v = json.loads(tf.artefact_paths(fam, film)[1].read_text())["validation"]
        assert v["superposition_rel_err"] == pytest.approx(ref["superposition_rel_err"], rel=1e-12)
        assert v["superposition_target"] == tf.SUPERPOSITION_TARGET
        assert v["superposition_pass"] is (v["superposition_rel_err"] < tf.SUPERPOSITION_TARGET)
        assert ("pending" in v["superposition_note"]) is pending


@pytest.mark.xfail(strict=True, reason="docs/handoff/extract_thermal.md#not-done: 8-ring bank superposition error "
                                       "13.9 % > 10 % (one 3-D mesh level at 0.4 um; discretization/multibody split "
                                       "pending)")
def test_superposition_meets_target():
    """07 P1-3: the distance-interpolated single-ring ROM reproduces the 8-ring bank within 10 %."""
    assert _crosscheck_3d()["bank"]["max_step_err_frac_all"] < tf.SUPERPOSITION_TARGET


def test_runtime_reader_accepts_artefacts():
    """mrm.plant.thermal.load_rom (the runtime reader) verifies every artefact (schema, sha256), finds the site
    offsets it needs, and reads the four 07 P1-3 numbers; the only quality warning is a failing superposition."""
    import warnings

    from mrm.plant import thermal as th
    if not PRESENT:
        pytest.skip("no thermal ROM artefact extracted yet")
    for fam, film in PRESENT:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = th.load_rom(tf.artefact_paths(fam, film)[0])
        assert {"monitor_pd", "temp_sensor", "driver"} <= set(r.site_offsets_um)
        vals = th.rom_validation_numbers(r.meta)
        assert all(v is not None for v in vals.values()), vals
        msgs = [str(x.message) for x in w if issubclass(x.category, RuntimeWarning)]
        assert [m for m in msgs if "superposition_rel_err" not in m] == [], msgs
        n_sup = sum("superposition_rel_err" in m for m in msgs)
        assert n_sup == (0 if vals["superposition_rel_err"] < tf.SUPERPOSITION_TARGET else 1), msgs


def test_val8_sidecar_matches_data():
    """ARCHITECTURE 5: the VAL-8 artefact has its sidecar with the sha256 of the data file."""
    p = tf.OUT_DIR / "thermal_val8_imec.json"
    if not p.exists():
        pytest.skip("thermal_val8_imec.json not produced yet")
    side = json.loads(p.with_name(p.stem + ".sidecar.json").read_text())
    assert side["schema"] == tf.SCHEMA_VAL8 and side["data_file"] == p.name
    assert side["sha256"] == tf.sha256_file(p)
    assert set(side["validation"]) >= {"VAL-8a", "VAL-8b", "VAL-8c", "VAL-8d"}
    assert len(side["mesh"]["levels"]) >= 3 and side["tool_versions"] and side["created_utc"]


def test_rom_dc_scale_uses_same_film_modes_artefact(rom):
    """07 P1-2 diagnostic: H_solver comes from modes_rib<film>.json when that artefact exists (not from another
    cross-section whose name merely contains the film number)."""
    film = int(rom["sidecar"]["film_nm"])
    if not (tf.OUT_DIR / f"modes_rib{film}.json").exists():
        pytest.skip("modes artefact missing")
    hyp = rom["sidecar"]["rom_dc_scale"]["dlambda_dt_hypotheses"]
    assert hyp["H_solver"]["source"].startswith(f"modes_rib{film}.json")


def test_write_val8_sidecar_and_superposition_record(tmp_path):
    """No-solver helpers: the VAL-8 sidecar hashes the data file; superposition_record reads the step metric."""
    p = tmp_path / "thermal_val8_imec.json"
    p.write_text(json.dumps({"schema": tf.SCHEMA_VAL8, "created_utc": "x", "tool_versions": {"a": 1},
                             "nominal": {"scale": 1.0, "materials": "mrm"},
                             "mesh_levels": {"2.0": {"scale": 2.0, "dofs": 10}},
                             "comparison": {"VAL-8a": {"predicted": 1.0, "target": 2.0, "rel_err": -0.5,
                                                       "tolerance": 0.3, "pass": False},
                                            "mesh_change_last_level": None}}))
    sp = tf.write_val8_sidecar(p)
    side = json.loads(sp.read_text())
    assert sp.name == "thermal_val8_imec.sidecar.json" and side["sha256"] == tf.sha256_file(p)
    assert side["validation"]["VAL-8a"]["pass"] is False
    rec = tf.superposition_record({"pitch_um": 50.0, "variant": {"bottom_bc": "dirichlet", "sub_um": 200.0},
                                   "bank": {"n_rings": 8, "max_step_err_frac_all": 0.05, "h_ring_um": 0.25,
                                            "max_multibody_step_err_frac_all": 0.01}})
    assert rec["superposition_pass"] and rec["superposition_rel_err"] == 0.05
    assert "pending" not in rec["superposition_note"]


def test_cli_3d_mesh_options(tmp_path, monkeypatch):
    """--h3d-levels / --h3d-bank reach RunOptions (sorted coarse to fine); defaults reproduce the committed record."""
    seen = {}

    def fake_propagate(fams, films, opts, out):
        seen["opts"] = opts
        return {}

    monkeypatch.setattr(tf, "stage_propagate", fake_propagate)
    assert tf.main(["--stage", "propagate", "--out", str(tmp_path), "--h3d-levels", "0.25", "0.5", "0.35",
                    "--h3d-bank", "0.25"]) == 0
    assert seen["opts"].h3d_levels == (0.5, 0.35, 0.25) and seen["opts"].h3d_bank == 0.25
    assert tf.RunOptions().h3d_levels == (0.5, 0.35) and tf.RunOptions().h3d_bank == 0.4


def test_conservative_mesh_bound_recorded(rom):
    """The derived bound (geometric tail of the level-to-level changes) is recorded, reproduces from the stored
    levels, and supports the stated mesh errors: < 0.3 % for the DC gains and the heater time constant; the 10 %
    self-heating time (slow convergence, observed order < 1) stays below 3 %."""
    conv = rom["sidecar"]["mesh"]["convergence"]
    if rom["sidecar"]["quick"]:
        pytest.skip("quick artefact")
    cb = conv["conservative_bound"]
    ref = tf.conservative_mesh_bound(conv)
    for k, v in ref["per_metric"].items():
        assert cb["per_metric"][k]["bound"] == pytest.approx(v["bound"], rel=1e-12), k
        assert v["bound"] >= conv["richardson"][k]["rel_err_est"] - 1e-15
    for k in ("z_heater_ring_k_per_mw", "z_heater_heater_k_per_mw", "z_opt_ring_k_per_mw", "t63_heater_ring_s"):
        assert cb["per_metric"][k]["bound"] < 3e-3, k
    assert cb["max_bound"] < 0.03


def test_conservative_mesh_bound_geometric_tail():
    """Synthetic levels f_n = 1 + a q^n: the bound of level n equals the exact remaining error a q^n."""
    a, q = 0.01, 0.8
    levels = [{"scale": s, "m": 1 + a * q ** n} for n, s in enumerate((4.0, 2.0, 1.0, 0.5))]
    conv = {"levels": levels, "production_scale": 1.0,
            "richardson": {"m": {"rel_err_est": 0.0, "order": None}}}
    b = tf.conservative_mesh_bound(conv)["per_metric"]["m"]
    assert b["bound"] == pytest.approx(a * q ** 2 / (1 + a * q ** 2), rel=1e-9)
    noisy = [{"scale": s, "m": 1 + 1e-5 * (-1) ** n} for n, s in enumerate((4.0, 2.0, 1.0, 0.5))]
    conv["levels"] = noisy
    assert tf.conservative_mesh_bound(conv)["per_metric"]["m"]["bound"] == pytest.approx(2e-5, rel=1e-3)


def test_every_assumption_has_a_range(rom):
    """Every geometry, boundary and site value of the model is labelled with its provenance and a range (Jacobian
    1-sigma, swept variants, the radial table, or an explicit 'not swept' range)."""
    fam, film = str(rom["family"]), float(rom["film_nm"])
    ar = rom["sidecar"]["inputs"]["assumption_ranges"]
    assert ar == json.loads(json.dumps(tf.assumption_ranges(fam, film)))
    params = ar["params"]
    other = {p for k, v in tf.FAMILY_FIELDS.items() if k != fam for p in v} - set(tf.FAMILY_FIELDS[fam])
    for p in tf.ASSUMPTIONS:
        if p == "film_nm" or p in other:     # film: a hypothesis (one artefact per film); other: another family
            continue
        assert p in params, p
    for p, _, sig in tf.JAC_COMMON + tf.JAC_FAMILY[fam]:
        assert params[p]["how"] == "jacobian" and params[p]["sigma_1"] == sig, p
    for p, j in rom["sidecar"].get("stages", {}).get("uncertainty", {}).get("jacobian", {}).items():
        assert params[p]["sigma_1"] == j["sigma_assumed"], p        # the label matches the sigma actually used
    for p, r in params.items():
        assert r["how"] in ("jacobian", "sweep", "table", "not swept", "numerical"), p
        if r["how"] == "jacobian":
            assert len(r["range_1sigma"]) == 2, p
        elif r["how"] != "numerical":
            assert r["range"] is not None, p


def test_write_artefact_keeps_superposition_keys(tmp_path):
    """A re-extraction keeps every validation.superposition_* key of the old sidecar (the bank result does not
    depend on the fit), including the target and the pass flag the runtime and the tests read."""
    cfg = tf.default_config("inring_noundercut", 160)
    K, D = 2, 1
    fr = tf.FitResult(taus=np.array([1e-6, 1e-3]), self_res=np.ones((5, 4, K)), cross_res=np.ones((5, 4, D, K)),
                      dc_self=2 * np.ones((5, 4)), dc_cross=2 * np.ones((5, 4, D)), cross_r=np.array([50.0]),
                      validation={}, fit_info={})
    _, js = tf.artefact_paths(cfg.family, cfg.film_nm, tmp_path)
    old = {"sha256": "x", "validation": {"superposition_rel_err": 0.139, "superposition_note": "n",
                                         "superposition_target": 0.1, "superposition_pass": False}}
    js.write_text(json.dumps(old))
    tf.write_artefact(cfg, fr, {"validation": {"step_max_rel_err": 0.001}}, tmp_path)
    v = json.loads(js.read_text())["validation"]
    assert v["superposition_target"] == 0.1 and v["superposition_pass"] is False
    assert v["superposition_rel_err"] == 0.139 and v["step_max_rel_err"] == 0.001


def test_g25_consistency_uses_current_hypotheses(rom):
    """The g25_consistency diagnostic uses the same dlambda/dT hypotheses as rom_dc_scale (refreshed without a
    solver by --stage propagate), and its numbers follow (FSR/P_FSR)/(Z dlambda/dT)."""
    g = rom["sidecar"].get("stages", {}).get("uncertainty", {}).get("g25_consistency")
    if not g or not g.get("variants"):
        pytest.skip("no g25_consistency record")
    hyp = rom["sidecar"]["rom_dc_scale"]["dlambda_dt_hypotheses"]
    assert g["dlambda_dt_hypotheses"] == hyp
    for v, ref in zip(g["variants"], tf.g25_rom_dc_scales(g, hyp)):
        assert set(v["rom_dc_scale_by_hypothesis"]) == set(hyp)
        for h in hyp:
            assert v["rom_dc_scale_by_hypothesis"][h] == pytest.approx(ref[h], rel=1e-12)


def test_analytic_disagreements_recorded():
    """Every FEM-vs-exact disagreement above 1 % is listed with its response size; each one is below 2e-5 of the
    self DC in absolute terms (the meaningful-error scale of the check)."""
    js = tf.artefact_paths("inring_noundercut", 160)[1]
    if not js.exists():
        pytest.skip("default artefact not extracted yet")
    ca = json.loads(js.read_text()).get("stages", {}).get("crosscheck_analytic")
    if not ca or "results" not in ca:
        pytest.skip("analytic stage not run")
    rec = ca["disagreements_above_1pct"]
    assert rec == json.loads(json.dumps(tf.analytic_disagreements(ca)))
    assert rec["max_err_of_self_dc"] < 2e-5


# ------------------------------------------------------------------------------------------- code-level checks
def _diffusion_line(n=300):
    x = np.concatenate([[0], np.geomspace(1e-3, 1.0, n)])
    h = np.diff(x)
    N = x.size
    rows, cols, vk, vm = [], [], [], []
    for e in range(N - 1):
        ke, me = 1.0 / h[e], h[e] / 6
        for a, b, kk, mm in ((e, e, ke, 2 * me), (e + 1, e + 1, ke, 2 * me), (e, e + 1, -ke, me), (e + 1, e, -ke, me)):
            rows.append(a)
            cols.append(b)
            vk.append(kk)
            vm.append(mm)
    K = sp.csr_matrix((vk, (rows, cols)), shape=(N, N))[:-1, :-1].tocsc()
    M = sp.csr_matrix((vm, (rows, cols)), shape=(N, N))[:-1, :-1].tocsc()
    B = np.zeros((N - 1, 1))
    B[0, 0] = 1.0
    C = np.zeros((2, N - 1))
    C[0, 0] = 1.0
    C[1, np.searchsorted(x, 0.1)] = 1.0
    return K, M, B, C


def test_krylov_rom_and_common_pole_fit_on_diffusion_line():
    """Rational Krylov ROM matches the full model; the common-pole Foster fit meets the contract metrics on a
    1-D diffusion line (a collocated and a non-collocated output)."""
    K, M, B, C = _diffusion_line()
    V = mor.rational_krylov_basis(K, M, B, mor.log_shifts(1e-3, 1e5, 3), 2)
    fm = mor.galerkin_modal(K, M, B, C, V)
    dc_exact = C @ sp.linalg.spsolve(K, B[:, 0])
    np.testing.assert_allclose(fm.dc[:, 0], dc_exact, rtol=1e-10)
    f = np.logspace(-3, 6, 60)
    Hex = np.array([C @ sp.linalg.spsolve((K + 2j * np.pi * ff * M).tocsc(), B[:, 0].astype(complex)) for ff in f]).T
    Hr = fm.freq(f)[:, 0, :]
    assert np.max(np.abs(Hr - Hex) / np.abs(Hex[:, :1])) < 1e-5      # relative to the DC gain
    t = np.logspace(-6, 3, 150)
    Y = fm.step(t)[:, 0, :]
    taus, res, info = mor.fit_common_foster(f, Hr, t, Y, fm.dc[:, 0], 12, 1e-6, 1e2)
    F = mor.Foster(taus, res)
    np.testing.assert_allclose(F.dc, fm.dc[:, 0], rtol=1e-12)
    assert np.all(mor.time_errors(F, t, Y, fm.dc[:, 0], 1e-6, 1e2) < 0.02)
    db, deg, _ = mor.freq_errors(F, f, Hex, band_floor=1e-3, dc_ref=fm.dc[:, 0])
    assert np.all(db < 0.3) and np.all(deg < 3.0)


def test_symmetric_galerkin_preserves_reciprocity():
    """K, M symmetric -> the Galerkin ROM transfer matrix is symmetric for like inputs and outputs."""
    K, M, B, C = _diffusion_line()
    BB = np.column_stack([B[:, 0], C[1]])
    CC = BB.T
    V = mor.rational_krylov_basis(K, M, BB, mor.log_shifts(1e-2, 1e4, 2), 2)
    fm = mor.galerkin_modal(K, M, BB, CC, V)
    H = fm.freq([0.0, 1.0, 100.0])
    np.testing.assert_allclose(H[0, 1], H[1, 0], rtol=1e-10)


def test_layered_analytic_disk_on_half_space():
    """Exact limit: a uniform-flux disk on an adiabatic half-space has mean temperature 8 P / (3 pi^2 k a)."""
    k, a = 1.3, 2e-6
    an = tf.LayeredAnalytic([(2e-2, k, k, 1.6e6)], "dirichlet")
    H = an.response(0.0, (0.0, a, 2e-2 - 1e-10, 2e-2), (0.0, a, 2e-2 - 1e-10, 2e-2), n_gauss=2)
    assert abs(H.real / (8 / (3 * np.pi ** 2 * k * a)) - 1) < 2e-3


def test_richardson_extrapolation():
    h = np.array([2.0, np.sqrt(2), 1.0])
    f = 1.0 + 0.3 * h ** 2
    r = tf.richardson(f, np.sqrt(2))
    assert abs(r["order"] - 2) < 1e-9
    assert abs(r["extrapolated"] - 1.0) < 1e-12


@pytest.mark.slow
def test_quick_extraction_reruns(tmp_path):
    """Re-run the cheap extraction end to end (mesh, FEM, ROM, fit, full-FEM verification, artefact I/O)."""
    rc = tf.main(["--quick", "--family", "inring_noundercut", "--film", "160", "--stage", "extract",
                  "--out", str(tmp_path)])
    assert rc == 0
    d = tf.load_rom("inring_noundercut", 160, tmp_path)
    assert d["self_res"].shape[:2] == (5, 4)
    np.testing.assert_allclose(d["self_res"].sum(-1), d["dc_self_k_per_w"], rtol=1e-9)
    committed = tf.load_rom("inring_noundercut", 160)
    # the coarse quick mesh reproduces the committed DC heater -> ring gain within 1 %
    h, r = tf.SOURCES.index("heater"), tf.OBSERVERS.index("ring_mode_weighted")
    assert abs(d["dc_self_k_per_w"][h, r] / committed["dc_self_k_per_w"][h, r] - 1) < 0.01
