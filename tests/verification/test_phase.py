"""Checks mrm.plant.phase, which turns PlantParams into the optics inputs (phase, round-trip loss, heat fraction, K^2).

Read this first
---------------
The tests run phase.ring_optics_row (through ``_optics_inputs``) on the real g26 preset (realize, B = 2, sampled
worlds and devices) and compare it with the closed forms of the mrm.plant.phase module docstring:

- the reference state reproduces lambda_res_cold, the round-trip amplitude, eta, K^2 and the FSR;
- a heated ring resonates exactly where the thermo-optic relation R1 puts it, and the local thermo-optic law used by
  the self-heating adapter is exact;
- the next order sits one FSR away, and the junction efficiency at 0 V is eff_at_0v scaled by lambda/n_g;
- the monitor identity P_drop kappa_heat/kappa_d = P_abs holds through mrm.plant.optics with these inputs;
- the rigid translation of the cold comb agrees with the bent-mode solver table to its documented accuracy.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from mrm import params
from mrm.plant import optics
from mrm.plant import phase
from mrm.units import C0


@pytest.fixture(scope="module")
def pp():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # missing-artefact fallbacks warn until the extraction lands
        cfg = params.load("g26")
        return params.realize(cfg, 2, seeds=3)


@pytest.fixture(scope="module")
def phase_params(pp):
    return phase.PhaseParams.from_plant_params(pp)


def _optics_inputs(phase_params, b, lam, t_ring, t_heater, v0, v1):
    """The optics inputs of world b from phase.ring_optics_row, as a dict of (N, L) arrays (and (N, 2) / (N,) ones).

    lam: line wavelengths (L,) m; t_ring, t_heater: (N,) K; v0, v1: junction voltages of the two NRZ levels (N,) V.
    Keys: phi0/phi1 (rad), a0/a1 (round-trip amplitude), e0/e1 (heat fraction eta), k2t/k2d (K^2), trt (s), shift
    (resonance shift at each level, m), cj (F), rs (ohm), flags.
    """
    rp = phase_params.ring[b]
    N, L = rp.shape[0], lam.shape[0]
    out = {k: np.zeros((N, L)) for k in ("phi0", "phi1", "a0", "a1", "e0", "e1", "k2t", "k2d", "trt")}
    shift, cj, rs = np.zeros((N, 2)), np.zeros((N, 2)), np.zeros((N, 2))
    flags = np.zeros(N, np.int8)
    row = phase_params.table_row(b)
    phase.ring_optics_row(rp, np.ascontiguousarray(lam, dtype=float), np.asarray(t_ring, float),
                          np.asarray(t_heater, float), np.asarray(v0, float), np.asarray(v1, float),
                          phase_params.grid, phase_params.dneff[row], phase_params.alpha_fca[row],
                          phase_params.capacitance[row], phase_params.r_series[row], out["phi0"], out["phi1"],
                          out["a0"], out["a1"], out["e0"], out["e1"], out["k2t"], out["k2d"], out["trt"], shift, cj,
                          rs, flags)
    out.update(shift=shift, cj=cj, rs=rs, flags=flags)
    return out


def test_reference_state_reproduces_plant_params(pp, phase_params):
    for b in range(pp.batch):
        rp = phase_params.ring[b]
        t_ref, v_ref = rp[:, phase.RP_T_REF], rp[:, phase.RP_V_REF]
        for k in range(pp.n_rings):
            lam = np.full(3, rp[k, phase.RP_LAM_RES])
            out = _optics_inputs(phase_params, b, lam, t_ref, t_ref, v_ref, v_ref)
            m = int(pp.optics.order[b, k])
            assert out["phi0"][k, 0] / (2 * math.pi) == pytest.approx(m, rel=0, abs=1e-9)
            assert out["a0"][k, 0] == pytest.approx(pp.optics.round_trip_amplitude[b, k], rel=1e-14, abs=0)
            assert out["e0"][k, 0] == pytest.approx(pp.optics.heat_fraction[b, k], rel=1e-12, abs=0)
            assert out["k2t"][k, 0] == pytest.approx(pp.optics.kappa2_through[b, k], rel=1e-14, abs=0)
            assert out["k2d"][k, 0] == pytest.approx(pp.optics.kappa2_drop[b, k], rel=1e-14, abs=0)
            assert 1.0 / out["trt"][k, 0] == pytest.approx(pp.optics.fsr_hz[b, k], rel=1e-12, abs=0)
            assert np.all(out["shift"][k] == 0.0)


def test_heated_ring_resonates_where_r1_puts_it(pp, phase_params):
    b = 1
    rp = phase_params.ring[b]
    t_ref, v_ref = rp[:, phase.RP_T_REF], rp[:, phase.RP_V_REF]
    for dt in (-20.0, 35.0, 110.0):
        t = t_ref + dt
        shift = np.array([phase.thermal_shift_m(t[k], t_ref[k], rp[k, phase.RP_S0], rp[k, phase.RP_BETA],
                                                rp[k, phase.RP_T0])
                          for k in range(pp.n_rings)])
        for k in range(pp.n_rings):
            out = _optics_inputs(phase_params, b, np.array([rp[k, phase.RP_LAM_RES] + shift[k]]), t, t, v_ref, v_ref)
            assert out["phi0"][k, 0] / (2 * math.pi) == pytest.approx(pp.optics.order[b, k], rel=0, abs=1e-9)
            # the slope of the phase against the shift is the optics' dphi/dlambda_res at the line
            assert out["shift"][k, 0] == pytest.approx(shift[k], rel=1e-12, abs=0)


def test_r1_shift_and_local_law(pp, phase_params):
    rp = phase_params.ring[0]
    s0, beta, t0, t_ref = rp[0, phase.RP_S0], rp[0, phase.RP_BETA], rp[0, phase.RP_T0], rp[0, phase.RP_T_REF]
    for t_prev in (t_ref - 10.0, t_ref + 60.0):
        s_l, b_l = phase.local_thermo_optic(t_prev, s0, beta, t0)
        for dT in (-3.0, 0.4, 7.0):
            want = (phase.thermal_shift_m(t_prev + dT, t_ref, s0, beta, t0)
                    - phase.thermal_shift_m(t_prev, t_ref, s0, beta, t0))
            assert s_l * (dT + 0.5 * b_l * dT * dT) == pytest.approx(want, rel=1e-12, abs=0)
    # s(T) = d shift / dT = s0 (1 + beta (T - T0))
    h = 1e-3
    d = (phase.thermal_shift_m(t_ref + h, t_ref, s0, beta, t0)
         - phase.thermal_shift_m(t_ref - h, t_ref, s0, beta, t0)) / (2 * h)
    assert d == pytest.approx(s0 * (1 + beta * (t_ref - t0)), rel=1e-9, abs=0)


def test_fsr_from_the_phase_model(pp, phase_params):
    """The next order (phi = 2 pi (m - 1)) sits one FSR red: c / (n_g L) to the dispersion order."""
    rp = phase_params.ring[0]
    k = 2
    lam0 = rp[k, phase.RP_LAM_RES]
    m = int(pp.optics.order[0, k])
    t_ref, v_ref = rp[:, phase.RP_T_REF], rp[:, phase.RP_V_REF]
    lam = lam0 + lam0 ** 2 / (rp[k, phase.RP_N_G] * rp[k, phase.RP_LEN])   # first guess
    for _ in range(30):                                                # secant on phi(lam) = 2 pi (m - 1)
        o1 = _optics_inputs(phase_params, 0, np.array([lam, lam * (1 + 1e-9)]), t_ref, t_ref, v_ref, v_ref)
        f0 = o1["phi0"][k, 0] - 2 * math.pi * (m - 1)
        df = (o1["phi0"][k, 1] - o1["phi0"][k, 0]) / (lam * 1e-9)
        lam -= f0 / df
    fsr_hz = C0 / lam0 - C0 / lam
    assert fsr_hz == pytest.approx(pp.optics.fsr_hz[0, k], rel=5e-3)


def test_junction_efficiency_at_zero_volts(pp, phase_params):
    b = 0
    rp = phase_params.ring[b]
    t_ref = rp[:, phase.RP_T_REF]
    lam = np.asarray(C0 / pp.laser.line_freq_hz[b])
    dv = 1e-3
    out = _optics_inputs(phase_params, b, lam, t_ref, t_ref, np.full(pp.n_rings, -dv), np.full(pp.n_rings, dv))
    eff = (out["shift"][:, 1] - out["shift"][:, 0]) / (2 * dv)
    lam_r, ng = rp[:, phase.RP_LAM_RES], rp[:, phase.RP_N_G]
    scale = (lam_r / ng) / np.mean(lam_r / ng)          # a common dneff(V) shifts each ring by lambda/n_g
    np.testing.assert_allclose(eff, pp.junction.eff_at_0v_m_per_v[b] * scale, rtol=2e-4)
    # reverse bias red-shifts (depletion raises n_eff) and lowers the FCA loss
    o2 = _optics_inputs(phase_params, b, lam, t_ref, t_ref, np.zeros(pp.n_rings), np.full(pp.n_rings, 2.0))
    assert np.all(o2["shift"][:, 1] > o2["shift"][:, 0])
    assert np.all(o2["a1"][:, 0] >= o2["a0"][:, 0])
    assert np.all(o2["cj"][:, 1] < o2["cj"][:, 0])


def test_drop_over_absorbed_is_kappa_ratio_through_the_optics(pp, phase_params):
    """The monitor identity across modules: I_mon kappa_heat/kappa_d = R P_abs (exact in TCMT with one order; the
    exact ring to O(1/F^2))."""
    b, k = 0, 0
    rp = phase_params.ring[b]
    t_ref, v_ref = rp[:, phase.RP_T_REF], rp[:, phase.RP_V_REF]
    lam_r = rp[k, phase.RP_LAM_RES]
    fwhm = lam_r / pp.optics.q_loaded[b, k]
    lam = lam_r + np.linspace(-3, 3, 13) * fwhm
    out = _optics_inputs(phase_params, b, lam, t_ref, t_ref, v_ref, v_ref)
    sl = slice(k, k + 1)
    # rates per line: K^2 is evaluated at each line, so kappa_d / kappa_i move with the line wavelength
    rates = np.array([optics.tcmt_rates(out["k2t"][k, i], out["k2d"][k, i], out["a0"][k, i], out["trt"][k, i])
                      for i in range(lam.shape[0])])
    ratio = out["e0"][k] * rates[:, 0] / rates[:, 2]
    for method, tol in (("tcmt", 1e-9), ("exact", 0.02)):
        r = optics.bus_propagate(np.full(lam.shape, 1e-3), out["phi0"][sl], out["k2t"][sl], out["k2d"][sl],
                                 out["a0"][sl], out["trt"][sl], out["e0"][sl], method=method, m_orders=1)
        np.testing.assert_allclose(r.drop[0] * ratio, r.absorbed[0], rtol=tol)


def test_rigid_comb_against_the_solver_table():
    """The documented accuracy of the rigid translation (module docstring, equation 2; physics_optics review, finding
    1): a heated ring's order m + k sits at x_k + delta in the phase model (x_k its cold position), while the bent
    mode's n_eff(lambda, T) table of the modes artefact (rib160, with L(T) from its expansion) puts it at
    x_k + delta (1 + kappa (x_k - lambda_res)). Measured kappa = 4.1-4.5e-4 /nm (g26 nominal, the 7.6 nm lock,
    rings 0, 4, 7, orders +-1 and +-2): +-33 pm (0.10 FWHM) at the adjacent orders, +-66 pm at the second ones.
    The locked order itself is exact. If the phase law changes, update the module docstring and this bound."""
    import json
    from pathlib import Path

    from scipy import integrate, optimize

    from mrm.sim import kernel
    art = json.loads((Path(phase.__file__).resolve().parents[1] / "data" / "extracted"
                      / "modes_rib160.json").read_text())
    tab = art["n_eff_table"]
    lam_t = np.asarray(tab["lambda_nm"], float)
    t_t = np.asarray(tab["t_c"], float) + 273.15
    n_t = np.asarray(tab["n_eff"], float)
    basis = lambda lam_nm, t_k: np.array([(lam_nm - 1310.0) ** i * ((t_k - 298.15) / 100.0) ** j
                                          for i in range(3) for j in range(4)])
    A = np.array([basis(lv, tv) for lv in lam_t for tv in t_t])
    coef = np.linalg.lstsq(A, n_t.reshape(-1), rcond=None)[0]
    assert np.sqrt(np.mean((A @ coef - n_t.reshape(-1)) ** 2)) < 1e-5            # the fit reproduces the table
    n_tab = lambda lam_m, t_k: float(coef @ basis(lam_m * 1e9, t_k))
    al = art["alpha_l_per_k"]
    p_al = np.polyfit(np.asarray(al["t_c"], float), np.asarray(al["value"], float), 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load("g26")
        pnom = params.realize(cfg, 1, seeds=0, uncertainty="nominal", variation="off")
    phs = phase.PhaseParams.from_plant_params(pnom)
    f = np.asarray(pnom.laser.line_freq_hz)[0]
    kappas = []
    for n in (0, 4, 7):
        rp = phs.ring[0, n]
        lam_r, ne, ng, dng, ln, t_ref = (rp[phase.RP_LAM_RES], rp[phase.RP_N_EFF], rp[phase.RP_N_G], rp[phase.RP_DNG],
                                         rp[phase.RP_LEN], rp[phase.RP_T_REF])
        m = round(ne * ln / lam_r)
        lam_line = C0 / f[int(pnom.layout.ring_to_line[n])]
        delta = lam_line - lam_r

        def nl_hot(x, t_k):
            d = x - lam_r
            n_cold = ne + (ne - ng) * d / lam_r - 0.5 * dng * d * d / lam_r          # the phase model's cold curve
            stretch = math.exp(integrate.quad(lambda tc: np.polyval(p_al, tc), t_ref - 273.15, t_k - 273.15)[0])
            return (n_cold + n_tab(x, t_k) - n_tab(x, t_ref)) * ln * stretch

        t_lock = optimize.brentq(lambda t_k: nl_hot(lam_line, t_k) - m * lam_line, t_ref, t_ref + 400.0)
        fsr_m = lam_r ** 2 * pnom.optics.fsr_hz[0, n] / C0
        for k in (-2, -1, 1, 2):
            x_k = kernel.cold_coordinate(rp, lam_r - k * fsr_m, float(m + k))
            solver = optimize.brentq(lambda x: nl_hot(x, t_lock) - (m + k) * x, x_k, x_k + 2.0 * delta, xtol=1e-18)
            kappas.append((solver - (x_k + delta)) / (delta * (x_k - lam_r) * 1e9))   # per nm of order distance
        x_m = kernel.cold_coordinate(rp, lam_r, float(m))
        assert x_m + delta == pytest.approx(lam_line, abs=1e-16)                    # the locked order: exact
    kappas = np.asarray(kappas)
    print(f"\nrigid-comb error rate kappa: {kappas.min():.3e} .. {kappas.max():.3e} per nm (x shift x distance)")
    assert np.all((kappas > 3.5e-4) & (kappas < 5.0e-4)), kappas
