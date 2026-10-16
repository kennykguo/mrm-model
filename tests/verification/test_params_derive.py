"""derive(): nominal view and diagnostics against hand calculations; provenance propagation; explain, table,
diff (07 P0-1 item 5, P0-3, P1-6, P1-9, P1-14)."""
import copy
import math
from pathlib import Path

import numpy as np
import pytest

from mrm.params.config import load, yaml_load
from mrm.params.derive import BISTABILITY_FACTOR, derive
from mrm.params.relations import integral_s
from mrm.params.table import diff, explain, show, table

MINI = Path(__file__).resolve().parents[1] / "data" / "mini.yaml"
C0 = 299792458.0


@pytest.fixture(scope="module")
def cfg():
    return load(MINI)


@pytest.fixture(scope="module")
def d(cfg):
    return derive(cfg, thermal_hook=None)


def test_nominal_view_is_b1(d):
    assert d.params.batch == 1
    assert d.params.diagnostics["uncertainty_mode"] == "nominal"
    assert d.params.diagnostics["variation_mode"] == "off"


def test_self_heating_gain_equals_efficiency_times_z_ratio(d):
    # 07 P0-1 item 5 under the default R2 form (lock_slope)
    t = d.params.thermal
    want = t.heater_efficiency_m_per_w[0] * 1.5 * 1e9    # pm/mW
    np.testing.assert_allclose(d.diagnostics["self_heating_gain_pm_per_mw"], want, rtol=1e-12)


def test_bistability_ratio_by_hand(d):
    p = d.params
    o = p.optics
    w = 2 * math.pi * C0 / o.lambda_res_cold_m[0]
    ki, kt, kd = w / o.q_intrinsic[0], w / o.q_through[0], w / o.q_drop[0]
    pin = 10 ** ((10.0 - 6.0) / 10) * 1e-3                # 10 dBm laser, 6 dB chip loss
    pabs = o.heat_fraction[0] * 4 * kt * ki / (ki + kt + kd) ** 2 * pin
    g = p.thermal.self_heating_gain_m_per_w[0]
    fwhm = o.lambda_res_cold_m[0] / o.q_loaded[0]
    np.testing.assert_allclose(d.diagnostics["bistability_ratio"], g * pabs / (BISTABILITY_FACTOR * fwhm), rtol=1e-12)
    assert BISTABILITY_FACTOR == pytest.approx(0.7698, abs=1e-4)
    # 07 P0-1: at +4 dBm near critical coupling the ring sits 3 to 7 times above the threshold
    assert 3.0 < d.diagnostics["bistability_ratio"].min() < 7.0
    np.testing.assert_allclose(d.diagnostics["p_in_ring_dbm"], 4.0)


def test_lock_and_hop_temperatures(d):
    p = d.params
    t, o = p.thermal, p.optics
    s0, t0, beta = t.dlambda_dt_s0_m_per_k[0], t.dlambda_dt_t0_k[0, 0], t.dlambda_dt_beta_per_k[0, 0]
    tl = d.diagnostics["t_ring_lock_c"] + 273.15
    shift = integral_s(tl, t0, s0, beta) - integral_s(o.t_ref_k[0, 0], t0, s0, beta)
    np.testing.assert_allclose(shift, o.resonance_offset_m[0], rtol=1e-12)
    # 07 P0-3: ~132 degC under H_solver for the 7.6 nm offset (checked), against 113 degC under H86
    assert 125 < d.diagnostics["t_ring_lock_c"].mean() < 135
    hop = d.diagnostics["t_amb_order_hop_c"]
    assert np.all(hop < d.diagnostics["t_ring_lock_c"]) and np.all(hop > 25.0)


def test_h86_lock_temperature_and_index_weight():
    cfg86 = load(MINI, alts={"uncertainty.thermo_optic": "H86"})
    d86 = derive(cfg86, thermal_hook=None)
    assert d86.diagnostics["t_ring_lock_c"].mean() == pytest.approx(25 + 7.6 / 0.086, abs=0.5)   # ~113 degC
    assert np.all(d86.diagnostics["index_weight_ratio"] > 1.0)        # H86 exceeds the materials ceiling (07 D2)
    assert any("ceiling" in w for w in d86.warnings)


def test_index_weight_below_ceiling_under_h_solver(d):
    r = d.diagnostics["index_weight_ratio"]
    assert np.all(r < 1.0) and np.all(r > 0.8)
    np.testing.assert_allclose(d.diagnostics["index_weight_ceiling"],
                               d.params.optics.n_g[0] / 3.69, rtol=0.01)      # n_g,Si ~ 3.69 at 1310 nm


def test_next_order_distance(d):
    """BND-1 diagnostic from the plant's phase model (07 P1-6): every other order of a ring locked on its own line sits
    where the cold phase curve, translated by the lock shift, puts it. Checked against the kernel's independent order
    solve (mrm.sim.kernel.cold_coordinate, exact derivative). Before the physics_optics review the diagnostic used
    the ideal comb f_line + k FSR (here [600, 1000, 1000, 600] GHz: 4 lines at -600, -200, 200, 600 GHz, FSR
    1.8 THz), which leaves out the lock shift's squeeze of the comb and the dispersion."""
    from mrm.plant import phase as ph
    from mrm.sim import kernel as K
    p = d.params
    phs = ph.PhaseParams.from_plant_params(p)
    f = np.asarray(p.laser.line_freq_hz)[0]
    want = []
    for n in range(p.n_rings):
        rp = phs.ring[0, n]
        a = int(p.layout.ring_to_line[n])
        lam_res = rp[ph.RP_LAM_RES]
        m = round(rp[ph.RP_N_EFF] * rp[ph.RP_LEN] / lam_res)
        shift = C0 / f[a] - lam_res
        fsr_m = lam_res ** 2 * p.optics.fsr_hz[0, n] / C0
        best = math.inf
        for k in (-2, -1, 1, 2):                         # span 1.2 THz < FSR: orders +-1, +-2
            x = K.cold_coordinate(rp, lam_res - k * fsr_m, float(m + k))
            best = min(best, float(np.min(np.abs(C0 / (x + shift) - np.delete(f, a)))))
        want.append(best / 1e9)
    np.testing.assert_allclose(d.diagnostics["next_order_distance_ghz"], want, rtol=1e-9)
    assert np.max(np.abs(d.diagnostics["next_order_distance_ghz"] - [600, 1000, 1000, 600])) > 1.0


@pytest.mark.parametrize("preset", ["g26", "g26_fsr_edge", "g26_fsr_window_a"])
def test_next_order_distance_is_the_bnd1_replica(preset):
    """derive's next_order_distance_fwhm is the BND-1 replica's distance (mrm.validation.replicas.
    adjacent_order_penalty: the plant's comb, each order's FWHM) for the nominal world, per ring and as the bank
    minimum, and its warning follows the 2 FWHM threshold. Before the physics_optics review derive published 3.37
    FWHM for g26 (replica 2.79), 1.17 for g26_fsr_edge (1.33) and 2.16-2.19 for g26_fsr_window_a (1.68, a soft fail
    for which derive gave no warning)."""
    R = pytest.importorskip("mrm.validation.replicas")
    dd = derive(load(preset))
    got = dd.diagnostics["next_order_distance_fwhm"]
    rep = R.adjacent_order_penalty(preset, batch=1)
    assert float(got.min()) == pytest.approx(rep.value, rel=1e-8)
    per_ring = {}
    for row in rep.detail["rows"]:                     # the 16 closest (ring, order) pairs
        per_ring[row["ring"]] = min(per_ring.get(row["ring"], math.inf), row["distance_fwhm"])
    for ring, v in per_ring.items():
        assert float(got[ring]) == pytest.approx(v, rel=1e-8), ring
    assert any("BND-1" in w for w in dd.warnings) == (rep.value < 2.0)


def test_identifiable_block(d):
    p = d.params
    np.testing.assert_allclose(d.diagnostics["monitor_codes_per_uw_at_drop"], 0.95 * 65536 / 40e-6 * 1e-6)
    np.testing.assert_allclose(d.diagnostics["fwhm_pm"], p.optics.lambda_res_cold_m[0] / 3800 * 1e12)
    np.testing.assert_allclose(d.diagnostics["disturbance_pm_per_k"], p.thermal.s_lock_m_per_k[0] * 1e12)
    # voltage drive: dP/dcode = 2 V_fs^2 c / ((2^bits - 1)^2 R), the ideal transfer c/(2^bits - 1) of
    # mrm.plant.electrical.dac_frac
    grid = d.diagnostics["heater_code_grid"]
    want = 2 * 1.6 ** 2 * grid[:, None] / (65535.0 ** 2 * 45.4) * p.thermal.heater_efficiency_m_per_w[0] * 1e12
    np.testing.assert_allclose(d.diagnostics["heater_pm_per_code"], want, rtol=1e-12)


def test_provenance_propagates_weakest(d):
    # efficiency = FSR_lambda/P_FSR: the FSR is the given fsr_thz (printed; R was solved from it and n_g), P_FSR
    # printed (07 D16: the efficiency inherits the FSR hypothesis); without ring.heater.p_fsr_ref_wavelength_nm the
    # FSR_lambda is at the ring's line, which comes from the channel plan (07 P1-14): mini's n_lines is assumed.
    # The radius depends on n_g (inferred).
    from mrm.params.derive import effective_provenance
    cfg = load(MINI)
    assert effective_provenance("derived.optics.fsr_hz", cfg, d.params.derivations) == "printed"
    assert effective_provenance("derived.optics.lambda_line_m", cfg, d.params.derivations) == "assumed"
    assert effective_provenance("ring.heater.efficiency_nm_per_mw", cfg, d.params.derivations) == "assumed"
    ref = load(MINI, overrides={"ring.heater.p_fsr_ref_wavelength_nm": 1309.5})   # override: tagged 'override'
    dr = derive(ref)
    assert effective_provenance("ring.heater.efficiency_nm_per_mw", ref, dr.params.derivations) == "override"
    assert effective_provenance("ring.geometry.radius_um", cfg, d.params.derivations) == "inferred"
    assert effective_provenance("ring.optics.q_intrinsic", cfg, d.params.derivations) == "assumed"
    assert d.provenance["bistability_ratio"] == "assumed"
    assert d.provenance["fsr_thz"] == "printed"
    # an FSR solved from n_g (inferred) and a printed radius inherits the weaker of the two
    m = copy.deepcopy(yaml_load(MINI.read_text()))
    m["ring"]["geometry"]["radius_um"] = {"value": 7.5, "provenance": "printed", "generation": "G25",
                                          "source": "LM-T"}
    m["ring"]["optics"]["fsr_thz"] = {"value": None, "provenance": "derived"}
    del m["uncertainty"]["fsr"]
    cfg2 = load(m)
    der = derive(cfg2, thermal_hook=None).params.derivations
    assert effective_provenance("derived.optics.fsr_hz", cfg2, der) == "inferred"
    # the efficiency (FSR_lambda at the line) also carries mini's line plan (n_lines assumed)
    assert effective_provenance("ring.heater.efficiency_nm_per_mw", cfg2, der) == "assumed"


def test_explain_chain(cfg):
    text = explain(cfg, "ring.heater.efficiency_nm_per_mw", thermal_hook=None)
    assert "R2_heater_efficiency" in text and "ring.heater.p_fsr_mw = 25.45" in text and "printed" in text
    assert "R3_geometry" in text and "printed (weakest input)" in text and "ring.optics.fsr_thz = 1.8" in text
    assert "inferred (weakest input)" in explain(cfg, "ring.geometry.radius_um", thermal_hook=None)
    leaf = explain(cfg, "ring.thermal.dlambda_dt_pm_per_k")
    assert "alternative H86" in leaf and "uncertainty.thermo_optic" in leaf and "chain:" in leaf
    assert "bistability_ratio" in explain(cfg, "bistability_ratio", thermal_hook=None)
    assert "hypothesis" in explain(cfg, "uncertainty.thermo_optic")


def test_table_is_deterministic(cfg):
    t1 = table(cfg, thermal_hook=None)
    t2 = table(load(MINI), thermal_hook=None)
    assert t1 == t2
    for section in ("## Leaves", "## Derived", "## Diagnostics", "## Uncertainty", "## Variation",
                    "## Textbook comparison"):
        assert section in t1
    assert "REF.dLambdaDT_pmK" in t1


def test_diff_and_show(cfg):
    other = load(MINI, overrides={"ring.optics.q_loaded": 4000}, alts={"uncertainty.thermo_optic": "H86"})
    text = diff(cfg, other)
    assert "~ ring.optics.q_loaded: 3800 [read-off] -> 4000 [override]" in text
    assert "alt=H86" in text
    assert diff(cfg, load(MINI)) == ""
    assert "ring.heater.p_fsr_mw = 25.45 mw  [printed, G25, LM-T]" in show(cfg)
