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


def test_3d_crosscheck_recorded():
    js = tf.artefact_paths("inring_noundercut", 160)[1]
    if not js.exists():
        pytest.skip("default artefact not extracted yet")
    st = json.loads(js.read_text()).get("stages", {})
    if "crosscheck_3d" not in st:
        pytest.skip("3-D stage not run")
    r = st["crosscheck_3d"]
    assert abs(r["single_ring_no_leads"]["rel_diff_z_heater_ring"]) < 0.05
    assert "max_residue_l1_rel_err_all" in r["bank"]


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
