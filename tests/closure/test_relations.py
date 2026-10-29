"""Test the relations R1-R5 that tie the ring parameters together (mrm/params/relations.py).

Read this first. Each relation links a few parameters (for example R3: FSR = c / (n_g L)). A preset may give any
members of a relation; realize() derives the free ones, and the relation must then close exactly. These tests use
the small preset tests/data/mini.yaml and check: closure after realize(), the errors for an over- or underdetermined
relation (and tolerance_rel), every pattern of given and free members, and the exact identities the plant relies on.

  R1 thermo-optic slope s(T) = s0 (1 + beta (T - T0))   R2 heater efficiency and the heater-to-ring gain Z
  R3 geometry: FSR, n_g, radius and the mode order       R4 Q budget, loss budget and heat fraction
  R5 latency: each I/O channel has one owner of its latency
"""
import copy
import math
from pathlib import Path

import numpy as np
import pytest

from mrm.params.config import ConfigError, load, yaml_load
from mrm.params.realize import realize
from mrm.params.relations import (OverdeterminedError, UnderdeterminedError, integral_s, residuals,
                                  solve_t_for_shift)

MINI = Path(__file__).resolve().parents[1] / "data" / "mini.yaml"      # small preset for relation tests
C0 = 299792458.0              # speed of light in vacuum, m/s


def mini():
    """A fresh copy of the mini.yaml mapping (each test changes its own copy)."""
    return copy.deepcopy(yaml_load(MINI.read_text()))


def leaf(v, prov="assumed"):
    """A leaf mapping with value ``v`` and provenance ``prov``."""
    return {"value": v, "provenance": prov, "generation": "none", "source": "TEST"}


def run(d, B=8, **kw):
    """Load mapping ``d`` and realize ``B`` worlds (seed 0, no thermal ROM hook)."""
    kw.setdefault("thermal_hook", None)
    return realize(load(d), B, 0, **kw)


@pytest.mark.parametrize("order, anchor", [(2, "lock_slope"), (1, "lock_slope"), (2, "fsr_integral")])
def test_every_relation_closes(order, anchor):
    d = mini()
    d["fidelity"]["thermo_optic_order"] = order
    d["fidelity"]["heater_anchor"] = anchor
    if anchor == "fsr_integral":
        d["ring"]["heater"]["p_fsr_ref_temp_c"] = leaf(25.0, "printed")
    p = run(d, 64)
    res = p.diagnostics["relation_residuals"]
    assert all(v <= 1e-12 for v in res.values()), res
    assert residuals(dict(p.values)) == res


def test_derive_closure_efficiency_times_pfsr_is_fsr():
    """R2: heater efficiency x P_FSR = FSR (in wavelength) exactly."""
    p = run(mini(), 16)
    fsr_m = p.optics.fsr_hz * p.optics.lambda_line_m ** 2 / C0
    np.testing.assert_allclose(p.thermal.heater_efficiency_m_per_w * p.heater.p_fsr_w, fsr_m, rtol=1e-14)


def test_r1_inversion_and_shape():
    p = run(mini(), 16)
    o, t = p.optics, p.thermal
    lam = o.lambda_line_m
    s = (lam / o.n_g) * (o.dn_eff_dt_per_k + o.n_eff * o.alpha_l_per_k)
    np.testing.assert_allclose(s, t.dlambda_dt_s0_m_per_k, rtol=1e-13)
    ds = (lam / o.n_g) * (o.d2n_eff_dt2_per_k2 + o.n_eff * o.alpha_l_slope_per_k2)
    np.testing.assert_allclose(ds, t.dlambda_dt_s0_m_per_k * t.dlambda_dt_beta_per_k, rtol=1e-12, atol=1e-30)
    # the shape s0 (1 + beta (T - T0)) integrates to the quadratic used for T_lock
    s0, t0, beta = 70e-12, 298.15, 1.4e-3
    tt = solve_t_for_shift(298.15, 7.6e-9, t0, s0, beta)
    assert integral_s(tt, t0, s0, beta) == pytest.approx(7.6e-9, rel=1e-14)


def test_r1_load_error_for_dn_eff_dt():
    d = mini()
    d["ring"]["optics"]["dn_eff_dt_per_k"] = leaf(2.4e-4)
    with pytest.raises(ConfigError, match="R1_thermo_optic"):
        load(d)


def test_r2_heater_z_forms():
    p = run(mini(), 4)
    t = p.thermal
    np.testing.assert_allclose(t.z_heater_ring_k_per_w * t.s_lock_m_per_k, t.heater_efficiency_m_per_w, rtol=1e-14)
    d = mini()
    d["fidelity"]["heater_anchor"] = "fsr_integral"
    d["ring"]["heater"]["p_fsr_ref_temp_c"] = leaf(25.0, "printed")
    q = run(d, 4)
    tq = q.thermal
    t_meas = 298.15
    shift = (integral_s(t_meas + tq.z_heater_ring_k_per_w * q.heater.p_fsr_w, tq.dlambda_dt_t0_k,
                        tq.dlambda_dt_s0_m_per_k, tq.dlambda_dt_beta_per_k)
             - integral_s(t_meas, tq.dlambda_dt_t0_k, tq.dlambda_dt_s0_m_per_k, tq.dlambda_dt_beta_per_k))
    np.testing.assert_allclose(shift, q.optics.fsr_hz * q.optics.lambda_line_m ** 2 / C0, rtol=1e-12)
    # The two forms agree when beta = 0 (H86 worlds) and differ by 5 to 15 % otherwise; under lock_slope Z also
    # follows each ring's sampled offset through T_lock (an open doubt: docs/handoff/params_machinery.md, "Contract
    # doubts").
    r = tq.z_heater_ring_k_per_w / t.z_heater_ring_k_per_w
    flat = t.dlambda_dt_beta_per_k[:, 0] == 0
    np.testing.assert_allclose(r[flat], 1.0, rtol=1e-12)
    assert np.all(np.abs(r[~flat] - 1) < 0.2)


def test_r3_order_and_phase():
    p = run(mini(), 8)
    o = p.optics
    np.testing.assert_allclose(o.n_eff * o.length_m, o.order * o.lambda_res_cold_m, rtol=1e-14)
    np.testing.assert_allclose(o.fsr_hz * o.n_g * o.length_m / C0, 1.0, rtol=1e-14)
    assert np.all(o.order == np.rint(o.order))
    # the order rounding is absorbed by n_eff (|correction| <= lambda/(2L)) for the nominal-geometry device; a device
    # of a sampled geometry keeps that order and that length (the Jacobians are at fixed r_ref and order)
    n = run(mini(), 8, variation="off")
    corr = np.abs(n.optics.n_eff - n.optics.n_eff_configured)
    assert np.all(corr <= n.optics.lambda_res_cold_m / (2 * n.optics.length_m) * (1 + 1e-12))
    np.testing.assert_array_equal(o.order, n.optics.order)
    np.testing.assert_array_equal(o.length_m, n.optics.length_m)


def test_r3_overdetermined_and_tolerance():
    d = mini()
    d["ring"]["geometry"]["radius_um"] = leaf(7.5, "printed")        # radius + n_g + FSR: all given, 13 % apart
    with pytest.raises(OverdeterminedError, match="R3_geometry"):
        run(d, uncertainty="nominal")
    d["relations"] = {"R3_geometry": {"tolerance_rel": 0.05}}
    with pytest.raises(OverdeterminedError, match="exceeds tolerance_rel"):
        run(d, uncertainty="nominal")
    d["relations"] = {"R3_geometry": {"tolerance_rel": 0.2}}
    p = run(d, uncertainty="nominal")
    # the weakest member (n_g, inferred) is recomputed; radius (printed) and FSR (printed) are kept
    np.testing.assert_allclose(p.optics.radius_m, 7.5e-6)
    np.testing.assert_allclose(p.optics.fsr_hz, 1.8e12)
    np.testing.assert_allclose(p.optics.n_g, C0 / (1.8e12 * 2 * math.pi * 7.5e-6))
    assert "R3_geometry" in p.diagnostics["relation_notes"]
    assert all(v <= 1e-12 for v in p.diagnostics["relation_residuals"].values())


def test_r3_consistent_overdetermined_is_fine():
    d = mini()
    r_um = C0 / (4.0 * 1.8e12 * 2 * math.pi) * 1e6
    d["ring"]["geometry"]["radius_um"] = leaf(r_um, "printed")
    p = run(d, uncertainty="nominal")          # the fsr hypothesis would otherwise draw window A
    np.testing.assert_allclose(p.optics.radius_m, r_um * 1e-6)


def test_r3_underdetermined():
    d = mini()
    del d["ring"]["optics"]["fsr_thz"]
    del d["uncertainty"]["fsr"]
    with pytest.raises(UnderdeterminedError, match="radius, n_g, FSR"):
        run(d)


def test_r3_radius_given_fsr_derived():
    d = mini()
    del d["ring"]["optics"]["fsr_thz"]
    del d["uncertainty"]["fsr"]
    d["ring"]["geometry"]["radius_um"] = leaf(7.5, "printed")
    p = run(d)
    np.testing.assert_allclose(p.optics.fsr_hz, C0 / (4.0 * 2 * math.pi * 7.5e-6), rtol=1e-14)


def test_r4_q_patterns():
    p = run(mini(), 4)
    o = p.optics
    np.testing.assert_allclose(1 / o.q_loaded, 1 / o.q_intrinsic + 1 / o.q_through + 1 / o.q_drop, rtol=1e-14)
    # exact per-pass couplings: the round-trip energy decay is the rate sum
    rt = o.round_trip_time_s
    w = 2 * math.pi * C0 / o.lambda_res_cold_m
    loaded = -np.log((1 - o.kappa2_through) * (1 - o.kappa2_drop) * o.round_trip_amplitude ** 2) / rt
    np.testing.assert_allclose(w / loaded, o.q_loaded, rtol=1e-13)
    # kappa_d_over_kappa instead of q_drop
    d = mini()
    del d["ring"]["optics"]["q_drop"]
    d["ring"]["geometry"]["jacobian"]["value"]["of"].pop("ln_kappa2_drop_per_nm")
    d["ring"]["optics"]["kappa_d_over_kappa"] = leaf(0.02)
    q = run(d, 2, variation="off")                 # the anchors hold for the nominal-geometry device
    np.testing.assert_allclose(q.optics.q_drop, q.optics.q_loaded / 0.02, rtol=1e-14)
    # q_loaded free, q_through given
    d = mini()
    d["ring"]["optics"]["q_loaded"] = {"value": None, "provenance": "derived"}
    del d["uncertainty"]["q_loaded"]
    d["ring"]["optics"]["q_through"] = leaf(7500.0)
    q = run(d, 2, variation="off")
    np.testing.assert_allclose(1 / q.optics.q_loaded, 1 / q.optics.q_intrinsic + 1 / 7500 + 1 / q.optics.q_drop,
                               rtol=1e-14)
    np.testing.assert_allclose(q.optics.q_through, 7500.0)
    s = run(d, 2)                                  # sampled geometry: the device's couplers, the rate sum still holds
    np.testing.assert_allclose(1 / s.optics.q_loaded, 1 / s.optics.q_intrinsic + 1 / s.optics.q_through
                               + 1 / s.optics.q_drop, rtol=1e-14)


def test_r4_inconsistent_and_underdetermined():
    d = mini()
    d["ring"]["optics"]["q_drop"] = leaf(5000.0)          # 1/Q_L < 1/Q_i + 1/Q_d: no positive Q_t
    d["ring"]["geometry"]["jacobian"]["value"]["of"].pop("ln_kappa2_drop_per_nm")
    with pytest.raises(ValueError, match="inconsistent Q"):
        run(d, 2)
    d = mini()
    del d["ring"]["optics"]["q_drop"]
    d["ring"]["geometry"]["jacobian"]["value"]["of"].pop("ln_kappa2_drop_per_nm")
    with pytest.raises(UnderdeterminedError, match="R4_q"):
        run(d, 2)


def test_r4_drop_share_is_checked_when_q_loaded_is_free():
    """q_drop and kappa_d_over_kappa both given, q_loaded free: Q_L comes from the rate sum, and the given
    x_d = Q_L/Q_d must agree with it (here Q_L/Q_d is about 0.02, not 0.5)."""
    d = mini()
    d["ring"]["optics"]["q_loaded"] = {"value": None, "provenance": "derived"}
    del d["uncertainty"]["q_loaded"]
    d["ring"]["optics"]["q_through"] = leaf(9000.0)
    d["ring"]["geometry"]["jacobian"]["value"]["of"].pop("ln_kappa2_drop_per_nm")
    d["ring"]["optics"]["kappa_d_over_kappa"] = leaf(0.5)
    with pytest.raises(OverdeterminedError, match="kappa_d_over_kappa are both given and disagree"):
        run(d, 2, variation="off")
    # an x_d inside the stated tolerance is accepted, and Q_L still comes from the rate sum
    del d["ring"]["optics"]["kappa_d_over_kappa"]
    q = run(d, 2, variation="off")
    x_d = float(np.mean(q.optics.q_loaded / q.optics.q_drop))
    d["ring"]["optics"]["kappa_d_over_kappa"] = leaf(x_d)
    d["relations"] = {"R4_q": {"tolerance_rel": 0.01}}
    p = run(d, 2, variation="off")
    np.testing.assert_allclose(p.optics.q_loaded, q.optics.q_loaded, rtol=1e-14)


def test_r4_loss_budget_and_heat_fraction():
    p = run(mini(), 2, variation="off")
    o = p.optics
    # mini.yaml loss budget fca, other_abs, scatter, bend radiation in dB/cm, converted to 1/m (x ln(10)/10 per 0.01 m)
    a = np.array([85.0, 8.0, 10.0, 2.0]) * math.log(10) / 10 / 0.01
    np.testing.assert_allclose(o.alpha_i_per_m, a.sum(), rtol=1e-13)
    np.testing.assert_allclose(o.heat_fraction, (a[0] + a[1]) / a.sum(), rtol=1e-14)   # absorption heats; the rest not
    # with the doping loss set to zero, eta = alpha_other/(alpha_other + alpha_scatter + alpha_rad)
    d = mini()
    d["ring"]["optics"]["loss_budget"]["alpha_fca_db_per_cm"]["value"] = 0.0
    d["ring"]["optics"]["q_loaded"]["value"] = 4200
    d["ring"]["optics"]["q_drop"]["value"] = 1e6
    del d["uncertainty"]["q_loaded"]
    d["ring"]["optics"]["q_loaded"]["range"] = [2600, 30000]
    d["ring"]["optics"]["loss_budget"]["alpha_other_abs_db_per_cm"]["value"] = 8.0
    d["ring"]["geometry"]["jacobian"]["value"]["of"].pop("ln_kappa2_drop_per_nm")
    d["ring"]["optics"]["q_loaded"]["value"] = 20000
    q = run(d, 1, variation="off")
    np.testing.assert_allclose(q.optics.heat_fraction, 8.0 / 20.0, rtol=1e-14)
    # pattern b: no component, heat_fraction and q_intrinsic given
    d = mini()
    del d["ring"]["optics"]["loss_budget"]
    d["ring"]["optics"]["heat_fraction"] = leaf(0.85)
    d["ring"]["optics"]["q_intrinsic"] = leaf(7800.0)
    q = run(d, 1, variation="off")
    np.testing.assert_allclose(q.optics.alpha_heat_per_m, 0.85 * q.optics.alpha_i_per_m, rtol=1e-14)
    for path in ("derived.optics.alpha_heat_per_m", "derived.optics.alpha_nonheat_per_m"):
        assert q.derivations[path].relation == "R4_heat_fraction"
    # overdetermined: a budget and a given heat fraction that disagree
    d = mini()
    d["ring"]["optics"]["heat_fraction"] = leaf(0.5)
    with pytest.raises(OverdeterminedError, match="R4_loss_budget"):
        run(d, 1)
    # neither a budget nor a heat fraction
    d = mini()
    del d["ring"]["optics"]["loss_budget"]
    d["ring"]["optics"]["q_intrinsic"] = leaf(7800.0)
    with pytest.raises(UnderdeterminedError, match="R4_heat_fraction: .*heat_fraction"):
        run(d, 1)


def test_r5_latency_owner_and_ticks():
    p = run(mini(), 32)
    s = p.timing.latency_scale[:, 0]
    np.testing.assert_array_equal(p.timing.heater_latency_ticks[:, 0], np.rint(1 * s).astype(np.int64))
    np.testing.assert_array_equal(p.timing.bias_latency_ticks[:, 0], np.rint(1 * s).astype(np.int64))
    d = mini()
    d["controller_interface"]["sensor_latency_samples"] = 2
    with pytest.raises(ConfigError, match="stated once"):
        load(d)


def test_r2_lock_on_the_heater_reachable_order():
    """A sampled ring whose cold resonance is red of its line (negative offset) or more than an FSR blue of it locks
    the nearest order blue of the line: T_lock uses offset mod FSR_lambda, never a temperature below ambient."""
    d = mini()
    d["ring"]["optics"]["resonance_offset_nm"] = leaf([7.6, -1.0, 12.0, 0.0])
    p = run(d, 4, uncertainty="nominal", variation="off")
    o, t = p.optics, p.thermal
    fsr_m = o.fsr_hz * o.lambda_line_m ** 2 / C0
    lock = p.values["derived.optics.lock_offset_m"]
    np.testing.assert_allclose(lock, np.mod(o.resonance_offset_m, fsr_m), rtol=1e-14, atol=0)
    assert np.all(t.t_ring_lock_k >= o.t_ref_k - 1e-9)
    shift = integral_s(t.t_ring_lock_k, t.dlambda_dt_t0_k, t.dlambda_dt_s0_m_per_k, t.dlambda_dt_beta_per_k) \
        - integral_s(o.t_ref_k, t.dlambda_dt_t0_k, t.dlambda_dt_s0_m_per_k, t.dlambda_dt_beta_per_k)
    np.testing.assert_allclose(shift, lock, rtol=1e-12, atol=1e-21)
    assert t.t_ring_lock_k[0, 3] == pytest.approx(o.t_ref_k[0, 0])      # zero offset: locked at ambient
    assert all(v <= 1e-12 for v in p.diagnostics["relation_residuals"].values())
