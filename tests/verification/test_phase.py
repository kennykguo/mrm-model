"""Verification of mrm.plant.phase: PlantParams -> optics inputs (07 P1-5, P1-6, P1-7, P0-2, P0-3 R1).

Checks against the closed forms of the module docstring, on the real g26 preset (realize, B = 2, sampled worlds and
devices): the reference state reproduces lambda_res_cold, the round-trip amplitude, eta, K^2 and the FSR; a heated
ring resonates exactly where R1 puts it; the local thermo-optic law used by the self-heating adapter is exact; the
junction efficiency at 0 V is eff_at_0v scaled by lambda/n_g; and the 07 P1-9 identity P_drop kappa_heat/kappa_d =
P_abs holds through mrm.plant.optics with these inputs.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from mrm import params
from mrm.plant import optics as op
from mrm.plant import phase as ph
from mrm.units import C0


@pytest.fixture(scope="module")
def pp():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # missing-artefact fallbacks warn until the extraction lands
        cfg = params.load("g26")
        return params.realize(cfg, 2, seeds=3)


@pytest.fixture(scope="module")
def pparams(pp):
    return ph.PhaseParams.from_plant_params(pp)


def _run(P, b, lam, t_ring, t_heater, v0, v1):
    rp = P.ring[b]
    N, L = rp.shape[0], lam.shape[0]
    out = {k: np.zeros((N, L)) for k in ("phi0", "phi1", "a0", "a1", "e0", "e1", "k2t", "k2d", "trt")}
    sh, cj, rs = np.zeros((N, 2)), np.zeros((N, 2)), np.zeros((N, 2))
    fl = np.zeros(N, np.int8)
    r = P.table_row(b)
    ph.ring_optics_row(rp, np.ascontiguousarray(lam, dtype=float), np.asarray(t_ring, float),
                       np.asarray(t_heater, float), np.asarray(v0, float), np.asarray(v1, float), P.grid, P.dneff[r],
                       P.alpha_fca[r], P.capacitance[r], P.r_series[r], out["phi0"], out["phi1"], out["a0"],
                       out["a1"], out["e0"], out["e1"], out["k2t"], out["k2d"], out["trt"], sh, cj, rs, fl)
    out.update(shift=sh, cj=cj, rs=rs, flags=fl)
    return out


def test_reference_state_reproduces_plant_params(pp, pparams):
    P = pparams
    for b in range(pp.batch):
        rp = P.ring[b]
        t_ref, v_ref = rp[:, ph.RP_T_REF], rp[:, ph.RP_V_REF]
        for k in range(pp.n_rings):
            lam = np.full(3, rp[k, ph.RP_LAM_RES])
            o = _run(P, b, lam, t_ref, t_ref, v_ref, v_ref)
            m = int(pp.optics.order[b, k])
            assert o["phi0"][k, 0] / (2 * math.pi) == pytest.approx(m, rel=0, abs=1e-9)
            assert o["a0"][k, 0] == pytest.approx(pp.optics.round_trip_amplitude[b, k], rel=1e-14, abs=0)
            assert o["e0"][k, 0] == pytest.approx(pp.optics.heat_fraction[b, k], rel=1e-12, abs=0)
            assert o["k2t"][k, 0] == pytest.approx(pp.optics.kappa2_through[b, k], rel=1e-14, abs=0)
            assert o["k2d"][k, 0] == pytest.approx(pp.optics.kappa2_drop[b, k], rel=1e-14, abs=0)
            assert 1.0 / o["trt"][k, 0] == pytest.approx(pp.optics.fsr_hz[b, k], rel=1e-12, abs=0)
            assert np.all(o["shift"][k] == 0.0)


def test_heated_ring_resonates_where_r1_puts_it(pp, pparams):
    P = pparams
    b = 1
    rp = P.ring[b]
    t_ref, v_ref = rp[:, ph.RP_T_REF], rp[:, ph.RP_V_REF]
    for dt in (-20.0, 35.0, 110.0):
        t = t_ref + dt
        shift = np.array([ph.thermal_shift_m(t[k], t_ref[k], rp[k, ph.RP_S0], rp[k, ph.RP_BETA], rp[k, ph.RP_T0])
                          for k in range(pp.n_rings)])
        for k in range(pp.n_rings):
            o = _run(P, b, np.array([rp[k, ph.RP_LAM_RES] + shift[k]]), t, t, v_ref, v_ref)
            assert o["phi0"][k, 0] / (2 * math.pi) == pytest.approx(pp.optics.order[b, k], rel=0, abs=1e-9)
            # the slope of the phase against the shift is the optics' dphi/dlambda_res at the line
            assert o["shift"][k, 0] == pytest.approx(shift[k], rel=1e-12, abs=0)


def test_r1_shift_and_local_law(pp, pparams):
    rp = pparams.ring[0]
    s0, beta, t0, t_ref = rp[0, ph.RP_S0], rp[0, ph.RP_BETA], rp[0, ph.RP_T0], rp[0, ph.RP_T_REF]
    for t_prev in (t_ref - 10.0, t_ref + 60.0):
        s_l, b_l = ph.local_thermo_optic(t_prev, s0, beta, t0)
        for dT in (-3.0, 0.4, 7.0):
            want = ph.thermal_shift_m(t_prev + dT, t_ref, s0, beta, t0) - ph.thermal_shift_m(t_prev, t_ref, s0, beta,
                                                                                               t0)
            assert s_l * (dT + 0.5 * b_l * dT * dT) == pytest.approx(want, rel=1e-12, abs=0)
    # s(T) = d shift / dT = s0 (1 + beta (T - T0))
    h = 1e-3
    d = (ph.thermal_shift_m(t_ref + h, t_ref, s0, beta, t0) - ph.thermal_shift_m(t_ref - h, t_ref, s0, beta, t0)) / (2 * h)
    assert d == pytest.approx(s0 * (1 + beta * (t_ref - t0)), rel=1e-9, abs=0)


def test_fsr_from_the_phase_model(pp, pparams):
    """The next order (phi = 2 pi (m - 1)) sits one FSR red: c / (n_g L) to the dispersion order."""
    rp = pparams.ring[0]
    k = 2
    lam0 = rp[k, ph.RP_LAM_RES]
    m = int(pp.optics.order[0, k])
    t_ref, v_ref = rp[:, ph.RP_T_REF], rp[:, ph.RP_V_REF]
    lam = lam0 + lam0 ** 2 / (rp[k, ph.RP_N_G] * rp[k, ph.RP_LEN])   # first guess
    for _ in range(30):                                                # secant on phi(lam) = 2 pi (m - 1)
        o1 = _run(pparams, 0, np.array([lam, lam * (1 + 1e-9)]), t_ref, t_ref, v_ref, v_ref)
        f0 = o1["phi0"][k, 0] - 2 * math.pi * (m - 1)
        df = (o1["phi0"][k, 1] - o1["phi0"][k, 0]) / (lam * 1e-9)
        lam -= f0 / df
    fsr_hz = C0 / lam0 - C0 / lam
    assert fsr_hz == pytest.approx(pp.optics.fsr_hz[0, k], rel=5e-3)


def test_junction_efficiency_at_zero_volts(pp, pparams):
    b = 0
    rp = pparams.ring[b]
    t_ref = rp[:, ph.RP_T_REF]
    lam = np.asarray(C0 / pp.laser.line_freq_hz[b])
    dv = 1e-3
    o = _run(pparams, b, lam, t_ref, t_ref, np.full(pp.n_rings, -dv), np.full(pp.n_rings, dv))
    eff = (o["shift"][:, 1] - o["shift"][:, 0]) / (2 * dv)
    lam_r, ng = rp[:, ph.RP_LAM_RES], rp[:, ph.RP_N_G]
    scale = (lam_r / ng) / np.mean(lam_r / ng)          # a common dneff(V) shifts each ring by lambda/n_g
    np.testing.assert_allclose(eff, pp.junction.eff_at_0v_m_per_v[b] * scale, rtol=2e-4)
    # reverse bias red-shifts (depletion raises n_eff) and lowers the FCA loss
    o2 = _run(pparams, b, lam, t_ref, t_ref, np.zeros(pp.n_rings), np.full(pp.n_rings, 2.0))
    assert np.all(o2["shift"][:, 1] > o2["shift"][:, 0])
    assert np.all(o2["a1"][:, 0] >= o2["a0"][:, 0])
    assert np.all(o2["cj"][:, 1] < o2["cj"][:, 0])


def test_drop_over_absorbed_is_kappa_ratio_through_the_optics(pp, pparams):
    """07 P1-9 cross-module identity: I_mon kappa_heat/kappa_d = R P_abs (TCMT, one order; exact ring to O(1/F^2))."""
    b, k = 0, 0
    rp = pparams.ring[b]
    t_ref, v_ref = rp[:, ph.RP_T_REF], rp[:, ph.RP_V_REF]
    lam_r = rp[k, ph.RP_LAM_RES]
    fwhm = lam_r / pp.optics.q_loaded[b, k]
    lam = lam_r + np.linspace(-3, 3, 13) * fwhm
    o = _run(pparams, b, lam, t_ref, t_ref, v_ref, v_ref)
    sl = slice(k, k + 1)
    # rates per line: K^2 is evaluated at each line (07 P1-5), so kappa_d / kappa_i move with the line wavelength
    rates = np.array([op.tcmt_rates(o["k2t"][k, i], o["k2d"][k, i], o["a0"][k, i], o["trt"][k, i])
                      for i in range(lam.shape[0])])
    ratio = o["e0"][k] * rates[:, 0] / rates[:, 2]
    for method, tol in (("tcmt", 1e-9), ("exact", 0.02)):
        r = op.bus_propagate(np.full(lam.shape, 1e-3), o["phi0"][sl], o["k2t"][sl], o["k2d"][sl], o["a0"][sl],
                             o["trt"][sl], o["e0"][sl], method=method, m_orders=1)
        np.testing.assert_allclose(r.drop[0] * ratio, r.absorbed[0], rtol=tol)
