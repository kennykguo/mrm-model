"""Verification of mrm/waveform (07 P2-6; docs/research/03 sections 2.2-2.5, 5.5, 5.6, 9.3).

Checks the code against analytic limits and independent solvers: the TCMT steady state against the closed forms and
the plant's TCMT, the exponential integrator against exact solutions and scipy DOP853, the small-signal response
against the Sacher & Poon / Karimelahi closed form and the 03 bandwidth table, the junction network against exact
and DOP853 solutions and against the plant's P1-8 transition energy, the eye metrics on synthetic waveforms, the
normalized LUT model against the physical TCMT, the 07 P2-6 PRBS test, and the LUT artefact (node exactness,
interpolation error, domain errors, agreement with full driver + nonlinear-junction runs on g26).
Runs under NUMBA_DISABLE_JIT=1 with reduced sizes.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from numba import config as numba_config
from scipy.integrate import solve_ivp

from mrm.plant import datapath as dp
from mrm.plant import electrical as el
from mrm.plant import optics as op
from mrm.plant import phase as ph
from mrm.units import C0
from mrm.waveform import driver as drv_mod
from mrm.waveform import lut as L
from mrm.waveform import tcmt_time as tt
from mrm.waveform.driver import DriverJunction, markov_bits, pattern_stats, prbs
from mrm.waveform.eye import eo_bandwidth, eye_metrics, simulate
from mrm.waveform.tcmt_time import RingTCMT

NOJIT = bool(numba_config.DISABLE_JIT)
W0 = 2 * math.pi * C0 / 1310e-9          # 03 section 1 reference ring
KAP = W0 / 3500.0


def _rates(ki_frac=0.5, ke_frac=0.49, kd_frac=0.01):
    return ki_frac * KAP, ke_frac * KAP, kd_frac * KAP


# ============================================================================================ TCMT core
def test_steady_state_matches_closed_forms_and_plant_tcmt():
    """03 section 2.5 closed forms (with g) and mrm.plant.optics.ring_tcmt (one order) for the same rates."""
    ki, ke, kd = _rates(0.45, 0.5, 0.05)
    t_rt = 0.5555e-12
    rng = np.random.default_rng(1)
    for _ in range(20):
        delta = rng.uniform(-3, 3) * KAP / 2
        g = complex(rng.normal(), rng.normal()) * 0.2 * KAP
        a1, a2 = tt.tcmt_steady_nb(delta, ki, ke, kd, g, 1.0 + 0j)
        d = 0.5 * KAP + 1j * delta
        den = d * d + abs(g) ** 2
        assert abs((1 - 1j * math.sqrt(ke) * a1) - (1 - ke * d / den)) < 1e-12
        assert abs(-1j * math.sqrt(kd) * a1 - (-math.sqrt(ke * kd) * d / den)) < 1e-12
        assert abs(-1j * math.sqrt(ke) * a2 - (1j * ke * np.conj(g) / den)) < 1e-12
        # plant TCMT, single order, same rates via the inverse logarithmic mapping
        k2t, k2d, a_rt = op.tcmt_to_exact(ki, ke, kd, t_rt)
        phi = 2 * math.pi * 77 + delta * t_rt
        rb = np.array([g * t_rt])
        r = op.ring_tcmt(phi, k2t, k2d, a_rt, t_rt, rb=np.repeat(rb, 1), m_orders=1)
        pw = tt.port_powers(np.array([a1]), np.array([a2]), 1.0, ki, ke, kd)
        assert abs(r.p_thru - pw["thru"][0]) < 1e-12
        assert abs(r.p_drop - pw["drop"][0]) < 1e-12
        assert abs(r.p_refl - pw["refl"][0]) < 1e-12
        assert abs(r.p_loss - pw["loss"][0]) < 1e-12


def test_integrator_exact_for_constant_rates():
    """Constant rates from a non-steady start: the step is exact at any h (matrix exponential reference)."""
    from scipy.linalg import expm
    ki, ke, kd = _rates()
    delta, g = 0.7 * KAP, (0.1 + 0.3j) * KAP
    m = np.array([[-(1j * delta + KAP / 2), -1j * g], [-1j * np.conj(g), -(1j * delta + KAP / 2)]])
    bvec = np.array([-1j * math.sqrt(ke), 0.0])
    a0 = np.array([0.3 + 0.1j, -0.2j]) * 1e-6
    h = 3.0 / KAP                                 # several field lifetimes per step
    k = 7
    out1 = np.empty(k + 1, complex)
    out2 = np.empty(k + 1, complex)
    tt.tcmt_run_nb(h, 1.0 + 0j, ke, kd, g, np.full(2 * k + 1, delta), np.full(2 * k + 1, ki), a0[0], a0[1],
                   out1, out2)
    ass = -np.linalg.solve(m, bvec)
    for j in range(k + 1):
        ref = expm(m * j * h) @ (a0 - ass) + ass
        assert abs(out1[j] - ref[0]) < 1e-12 * abs(ass[0])
        assert abs(out2[j] - ref[1]) < 1e-12 * abs(ass[0])


def _ivp_reference(delta_fn, ki_fn, ke, kd, g, t_end, a0):
    def rhs(t, y):
        a1 = y[0] + 1j * y[1]
        a2 = y[2] + 1j * y[3]
        kap = ki_fn(t) + ke + kd
        lam = -(1j * delta_fn(t) + 0.5 * kap)
        d1 = lam * a1 - 1j * g * a2 - 1j * math.sqrt(ke)
        d2 = lam * a2 - 1j * np.conj(g) * a1
        return [d1.real, d1.imag, d2.real, d2.imag]
    return rhs


def test_integrator_against_dop853_time_varying():
    """Fast detuning and loss swings with backscatter: agrees with DOP853 (rtol 1e-11); observed error ~ h^4."""
    ki, ke, kd = _rates()
    g = (0.05 + 0.1j) * KAP
    t_end = 60e-12

    def delta_fn(t):
        return KAP * (0.4 + 0.6 * math.tanh((t - 20e-12) / 2e-12) + 0.3 * math.sin(2 * math.pi * 37e9 * t))

    def ki_fn(t):
        return ki * (1.0 - 0.05 * math.tanh((t - 20e-12) / 2e-12))

    a1s, a2s = tt.tcmt_steady_nb(delta_fn(0.0), ki_fn(0.0), ke, kd, g, 1.0 + 0j)
    sol = solve_ivp(_ivp_reference(delta_fn, ki_fn, ke, kd, g, t_end, None), (0, t_end),
                    [a1s.real, a1s.imag, a2s.real, a2s.imag], method="DOP853", rtol=1e-11, atol=1e-18,
                    dense_output=True)
    errs = []
    for h in (0.5e-12, 0.25e-12, 0.125e-12):
        k = int(round(t_end / h))
        th = np.arange(2 * k + 1) * h / 2
        out1 = np.empty(k + 1, complex)
        out2 = np.empty(k + 1, complex)
        tt.tcmt_run_nb(h, 1.0 + 0j, ke, kd, g, np.array([delta_fn(t) for t in th]),
                       np.array([ki_fn(t) for t in th]), a1s, a2s, out1, out2)
        y = sol.sol(np.arange(k + 1) * h)
        err = max(np.max(np.abs(out1 - (y[0] + 1j * y[1]))), np.max(np.abs(out2 - (y[2] + 1j * y[3]))))
        errs.append(err / np.max(np.abs(y[0] + 1j * y[1])))
    assert errs[1] < 1e-6                          # h = 0.25 ps (about 80 samples per 20 ps bit): 1.5e-7 measured
    assert errs[0] / errs[1] > 10.0 and errs[1] / errs[2] > 10.0   # ~16 per halving measured (fourth order)


def test_energy_balance_over_a_period():
    """Over one period of a cyclic pattern the stored energy returns, so P_in = sum of outputs + loss (03 2.4)."""
    ring = RingTCMT.from_rates(*_rates(), 0.3 * KAP, v_bias=1.0, ddelta_dv=0.4 * KAP, dkappa_i_dv=-0.02 * KAP,
                               g=0.1j * KAP)
    dj = DriverJunction(np.linspace(-0.5, 4, 46), 10e-15, 100.0, 50.0, rise_time_s=6e-12)
    w = simulate(ring, dj, prbs(7), 50e9, v_bias=1.0, swing_vpp=1.2, samples_per_bit=32 if NOJIT else 64)
    out = sum(w.mean(k) for k in ("thru", "drop", "refl", "add", "loss"))
    assert abs(out / w.p_in_w - 1.0) < 2e-5


# ============================================================================================ small signal
def test_small_signal_matches_sacher_poon_closed_form():
    """Exact linearization (g = 0, no loss modulation) = 03 section 5.6 closed form, zero at -(ki + kd)."""
    rng = np.random.default_rng(3)
    f = np.array([1e8, 5e9, 30e9, 70e9, 150e9])
    for _ in range(30):
        fr = rng.dirichlet([1, 1, 0.3])
        ki, ke, kd = fr * KAP
        delta = rng.uniform(-3, 3) * KAP / 2
        h = tt.optical_response_rates(ki, ke, kd, delta, f, ddelta=-1.0)
        hc = tt.h_opt_closed_form(2j * np.pi * f, ki, ke, kd, delta)
        assert np.max(np.abs(h - hc)) < 1e-10 * np.max(np.abs(hc))
    # the transfer function vanishes at the zero s = -(ki + kd) (analytic continuation of the closed form)
    ki, ke, kd = _rates()
    assert abs(tt.h_opt_closed_form(-(ki + kd) + 0j, ki, ke, kd, 0.3 * KAP)) < 1e-30


@pytest.mark.parametrize("x, coupling, q, f3, pk, pk_dec", [
    (0.30, "critical", 0.0, 38.7, 0.0, 0), (1 / math.sqrt(3), "critical", 0.0, 55.2, 0.10, 2),
    (0.813, "critical", 0.0, 75.7, 1.0, 1), (1.0, "critical", 0.0, 95.2, 2.1, 1),
    (2.0, "critical", 0.0, 246.0, 8.8, 1), (0.813, "over", 0.0, 152.0, 5.1, 1), (0.813, "under", 0.0, 54.4, 0.1, 1),
    (1 / math.sqrt(3), "critical", 0.13, 51.3, 0.02, 2), (1 / math.sqrt(3), "critical", -0.1, 58.7, 0.21, 2)])
def test_bandwidth_table_of_03(x, coupling, q, f3, pk, pk_dec):
    """The optical-only bandwidth/peaking table of 03 section 5.6 (Q_L 3500, 1310 nm; all-pass, kappa_d -> 0;
    loss modulation q = delta(kappa_i/2)/delta w_r), to its printed digits."""
    kap = KAP
    ke = {"critical": 0.5, "over": 0.75, "under": 0.25}[coupling] * kap
    ki = kap - ke
    # input u = delta w_r in units of 1e11 rad/s (dDelta = -u, d(kappa_i/2) = q u)
    ring = RingTCMT.from_rates(ki, ke, 1e-9, x * kap / 2, ddelta_dv=-1e11, dkappa_i_dv=2 * q * 1e11)
    r = eo_bandwidth(ring, None, 0.0, f_max_hz=600e9, n=6001)
    nd = 0 if f3 >= 100 else 1          # the table prints three significant figures
    assert round(r["f3db_hz"] / 1e9, nd) == pytest.approx(f3, abs=10 ** -nd * 0.51)
    assert abs(r["peaking_db"] - pk) <= 0.5 * 10.0 ** -pk_dec + 1e-9    # to the printed decimals


@pytest.mark.parametrize("x, f_rc", [(1 / math.sqrt(3), 34.8), (0.813, 25.1)])
def test_cal10_inversion_of_03(x, f_rc):
    """03 section 5.6 (verified there): a 30.9 GHz total EO bandwidth at Q_L 3500, critical coupling, implies a
    single-pole RC of 34.8 GHz at x = 1/sqrt(3) and 25.1 GHz at x = 0.813. The H_E x H_opt composition of
    eo_bandwidth (DriverJunction with R_drv = 0, constant C_j) inverts to the same poles."""
    from scipy.optimize import brentq
    ring = RingTCMT.from_rates(0.5 * KAP, 0.5 * KAP, 1e-9, x * KAP / 2, ddelta_dv=-1e11)
    g = np.linspace(-0.5, 4.0, 10)

    def f3(f_pole):
        r_s = 1.0 / (2 * math.pi * f_pole * 10e-15)
        return eo_bandwidth(ring, DriverJunction(g, 10e-15, r_s, 0.0), 0.0, f_max_hz=600e9, n=6001)["f3db_hz"]
    pole = brentq(lambda fp: f3(fp) - 30.9e9, 15e9, 200e9, xtol=1e3)
    assert round(pole / 1e9, 1) == pytest.approx(f_rc, abs=0.051)


def test_loss_modulation_on_resonance_and_vanishing_dc_gain():
    """03 section 5.6: on resonance pure index modulation gives no first-order intensity modulation (eo_bandwidth
    then reports nan), and loss modulation alone gives a single real pole at kappa/2: f_3dB = FWHM/2 exactly
    (away from critical coupling, where the through port is first-order insensitive to kappa_i as well)."""
    ki, ke, kd = _rates(0.5, 0.45, 0.05)
    ring = RingTCMT.from_rates(ki, ke, kd, 0.0, ddelta_dv=0.3 * KAP, dkappa_i_dv=-0.02 * KAP)
    r = eo_bandwidth(ring, None, 0.0, loss_modulation=False)
    assert math.isnan(r["f3db_hz"]) and math.isnan(r["peaking_db"])
    r = eo_bandwidth(ring, None, 0.0, loss_modulation=True)
    assert r["f3db_hz"] == pytest.approx(0.5 * KAP / (2 * math.pi), rel=1e-6)
    assert r["peaking_db"] == pytest.approx(0.0, abs=1e-9)


def test_time_domain_lockin_matches_linearization():
    """Small sinusoidal junction voltage through the time-domain integrator = the linearized response (with
    backscatter and loss modulation), so the waveform dynamics and the small-signal model agree."""
    ki, ke, kd = _rates(0.5, 0.45, 0.05)
    ring = RingTCMT.from_rates(ki, ke, kd, 0.5 * KAP / 2, v_bias=1.0, ddelta_dv=0.3 * KAP, dkappa_i_dv=-0.02 * KAP,
                               g=(0.05 + 0.08j) * KAP)
    amp = 1e-4
    for f in (10e9, 40e9, 90e9):
        n_cyc = 8
        spc = 64 if not NOJIT else 32
        h = 1.0 / (f * spc)
        k = spc * (n_cyc + 4)
        th = np.arange(2 * k + 1) * h / 2
        tr = ring.run(1.0 + amp * np.cos(2 * math.pi * f * th), h)
        p = tr.powers["thru"][4 * spc:4 * spc + n_cyc * spc]
        t = tr.t_s[4 * spc:4 * spc + n_cyc * spc]
        lock = 2 * np.mean(p * np.exp(-2j * math.pi * f * t))
        lin = ring.optical_response(1.0, [f])[0] * amp
        assert abs(lock - lin) < 3e-3 * abs(lin)


# ============================================================================================ patterns, driver
def test_prbs_is_the_datapath_prbs():
    for k in (7, 9, 15):
        assert np.array_equal(prbs(k), dp.prbs_bits(k, dp.PRBS_TAPS[k]))
        m, t = pattern_stats(prbs(k))
        assert m == pytest.approx(2 ** (k - 1) / (2 ** k - 1))
        assert t == pytest.approx(2 ** (k - 1) / (2 ** k - 1))
    assert prbs(31, 4096).size == 4096
    with pytest.raises(ValueError):
        prbs(31)


def test_markov_bits_statistics():
    for p1, td in ((0.5, 0.5), (0.25, 0.3), (0.7, 0.2)):
        m, t = pattern_stats(markov_bits(4096, p1, td))
        assert abs(m - p1) < 0.01 and abs(t - td) < 0.01
    with pytest.raises(ValueError):
        markov_bits(100, 0.2, 0.5)


def test_gaussian_edge_rise_time():
    tr_s = 7e-12
    sig = drv_mod.rise_to_sigma(tr_s)
    bits = np.array([0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], np.uint8)
    t = np.linspace(10e-12, 30e-12, 20001)
    v = np.array([drv_mod.nrz_source_nb(x, bits, 20e-12, sig, 0.0, 1.0) for x in t])
    t10 = np.interp(0.1, v, t)
    t90 = np.interp(0.9, v, t)
    assert t90 - t10 == pytest.approx(tr_s, rel=1e-5)
    assert drv_mod.nrz_source_nb(100e-12, bits, 20e-12, sig, 0.2, 1.7) == pytest.approx(1.7, abs=1e-12)


def test_junction_rc_linear_step_is_exponential():
    """Constant C and R, ideal step: V_j = V_hi - (V_hi - V_lo) exp(-t/RC) at every sample (RK4 substeps)."""
    g = np.linspace(-0.5, 4.0, 10)
    dj = DriverJunction(g, 12e-15, 80.0, 40.0)
    bits = np.array([0, 1, 1, 1, 1, 1, 1, 1], np.uint8)
    run = dj.run(bits, 20e-12, 64, 0.5, 1.8, n_warm_bits=1)
    tau = 120.0 * 12e-15
    t = run.t_s
    # the warm-up bit is bits[-1] = 1: the run starts at 1.8 V at t = -20 ps and falls toward 0.5 V from t = 0
    m0 = (t >= 0) & (t <= 20e-12)
    assert np.max(np.abs(run.v_j[m0] - (0.5 + 1.3 * np.exp(-t[m0] / tau)))) < 1e-7
    v20 = 0.5 + 1.3 * np.exp(-20e-12 / tau)
    m = (t > 20e-12) & (t < 160e-12)
    ref = 1.8 - (1.8 - v20) * np.exp(-(t[m] - 20e-12) / tau)
    assert np.max(np.abs(run.v_j[m] - ref)) < 1e-7


def test_junction_rc_nonlinear_two_node_against_dop853():
    """C_j(V) of the analytic depletion table, R_s(V), C_pad two-node network, Gaussian edges vs DOP853."""
    tab = el.analytic_junction_table(shift_amp_m=170e-12, v_bi_v=0.9, m=0.5, lambda_m=1.31e-6, n_g=4.0,
                                     c0_f_per_m=15e-15 / 33e-6, alpha0_per_m=1500.0, loss_ratio_q=0.06,
                                     r_series_ohm_m=100.0 * 33e-6)
    cj = tab.capacitance_f_per_m[0] * 33e-6
    rs = tab.r_series_ohm_m[0] / 33e-6 * (1 + 0.1 * tab.v_reverse_v)       # a V-dependent R_s
    dj = DriverJunction(tab.v_reverse_v, cj, rs, 50.0, c_pad_f=5e-15, rise_time_s=6e-12)
    bits = prbs(7)[:24]
    tb = 20e-12
    run = dj.run(bits, tb, 64, 0.35, 1.65, n_warm_bits=4)
    sig = dj.sigma_s

    def rhs(t, y):
        vs = drv_mod.nrz_source_nb(t, bits, tb, sig, 0.35, 1.65)
        vp, vj = y
        c = np.interp(vj, tab.v_reverse_v, cj)
        r = np.interp(vj, tab.v_reverse_v, rs)
        return [((vs - vp) / 50.0 - (vp - vj) / r) / 5e-15, (vp - vj) / (r * c)]
    t0 = run.t_s[0]
    v0 = drv_mod.nrz_source_nb(t0, bits, tb, sig, 0.35, 1.65)
    sol = solve_ivp(rhs, (t0, run.t_s[-1]), [v0, v0], method="DOP853", rtol=1e-10, atol=1e-13,
                    t_eval=run.t_s[::16], max_step=0.5e-12)
    assert np.max(np.abs(sol.y[1] - run.v_j[::16])) < 2e-6


def test_small_signal_electrical_matches_time_domain():
    """h_electrical of the two-node network (R_drv, C_pad, R_s, C_j) = a lock-in on an independent DOP853 solution
    of the same network driven by a small sinusoid."""
    g = np.linspace(-0.5, 4.0, 10)
    dj = DriverJunction(g, 10e-15, 120.0, 50.0, c_pad_f=8e-15)
    f = 30e9
    tb = 1.0 / f
    n, spb = 16, 256

    def rhs(t, y):
        vs = 1.0 + 1e-3 * math.cos(2 * math.pi * f * t)
        vp, vj = y
        return [((vs - vp) / 50.0 - (vp - vj) / 120.0) / 8e-15, (vp - vj) / (120.0 * 10e-15)]
    t = np.arange(n * spb) * tb / spb
    sol = solve_ivp(rhs, (0, t[-1]), [1.0, 1.0], method="DOP853", rtol=1e-11, atol=1e-15, t_eval=t)
    m = t >= 8 * tb
    lock = 2 * np.mean((sol.y[1][m] - 1.0) * np.exp(-2j * math.pi * f * t[m])) / 1e-3
    assert abs(lock - dj.h_electrical(1.0, [f], include_driver=False)[0]) < 1e-4
    # the driver's Gaussian edge multiplies by exp(-(2 pi f sigma)^2/2)
    dj2 = DriverJunction(g, 10e-15, 120.0, 50.0, c_pad_f=8e-15, rise_time_s=8e-12)
    ratio = dj2.h_electrical(1.0, [f])[0] / dj.h_electrical(1.0, [f])[0]
    assert ratio == pytest.approx(math.exp(-0.5 * (2 * math.pi * f * dj2.sigma_s) ** 2), rel=1e-12)


def test_simulate_end_to_end_against_dop853():
    """The whole chain of simulate (Gaussian-edged source -> two-node network with C_j(V), R_s(V) -> instantaneous
    Delta(V_j), kappa_i(V_j) -> CW/CCW TCMT with backscatter) against ONE joint ODE (electrical and optical states
    together) solved by scipy DOP853 from the same DC start: checks the half-step alignment of V_j and the optics,
    which the per-block tests cannot see (a half-step shift would cost ~1e-2 of P_in here)."""
    ki, ke, kd = _rates(0.5, 0.45, 0.05)
    tab = el.analytic_junction_table(shift_amp_m=170e-12, v_bi_v=0.9, m=0.5, lambda_m=1.31e-6, n_g=4.0,
                                     c0_f_per_m=15e-15 / 33e-6, alpha0_per_m=1500.0, loss_ratio_q=0.06,
                                     r_series_ohm_m=100.0 * 33e-6)
    v = tab.v_reverse_v
    cj = tab.capacitance_f_per_m[0] * 33e-6
    rs = tab.r_series_ohm_m[0] / 33e-6 * (1 + 0.1 * v)
    delta_tab = 0.3 * KAP + 1.2 * KAP * math.sqrt(1.9) * (np.sqrt(0.9 + v) - math.sqrt(1.9))     # sqrt shift law
    ki_tab = ki * (1 - 0.03 * (v - 1.0))
    g = (0.04 + 0.07j) * KAP
    ring = RingTCMT(ke, kd, g, v, delta_tab, ki_tab, np.ones_like(v))
    dj = DriverJunction(v, cj, rs, 50.0, c_pad_f=4e-15, rise_time_s=7e-12)
    bits = prbs(7)[:20]
    tb, nwarm = 20e-12, 3
    w = simulate(ring, dj, bits, 1 / tb, v_bias=1.0, swing_vpp=1.3, samples_per_bit=32, n_warm_bits=nwarm,
                 p_in_w=1.0)
    lo, hi, _ = w.levels
    sig = dj.sigma_s

    def rhs(t, y):
        vs = drv_mod.nrz_source_nb(t, bits, tb, sig, lo, hi)
        vp, vj = y[0], y[1]
        c, r = np.interp(vj, v, cj), np.interp(vj, v, rs)
        a1, a2 = y[2] + 1j * y[3], y[4] + 1j * y[5]
        lam = -(1j * np.interp(vj, v, delta_tab) + 0.5 * (np.interp(vj, v, ki_tab) + ke + kd))
        d1 = lam * a1 - 1j * g * a2 - 1j * math.sqrt(ke)
        d2 = lam * a2 - 1j * np.conj(g) * a1
        return [((vs - vp) / 50.0 - (vp - vj) / r) / 4e-15, (vp - vj) / (r * c), d1.real, d1.imag, d2.real, d2.imag]
    t0 = -nwarm * tb
    vs0 = drv_mod.nrz_source_nb(t0, bits, tb, sig, lo, hi)
    a1, a2 = tt.tcmt_steady_nb(float(np.interp(vs0, v, delta_tab)), float(np.interp(vs0, v, ki_tab)), ke, kd, g,
                               1.0 + 0j)
    sol = solve_ivp(rhs, (t0, w.t_s[-1]), [vs0, vs0, a1.real, a1.imag, a2.real, a2.imag], method="DOP853",
                    rtol=1e-10, atol=1e-12, t_eval=w.t_s, max_step=0.2e-12)
    a1r = sol.y[2] + 1j * sol.y[3]
    assert np.max(np.abs(sol.y[1] - w.v_j)) < 1e-6
    assert np.max(np.abs(np.abs(1 - 1j * math.sqrt(ke) * a1r) ** 2 - w.powers["thru"])) < 5e-6      # 6e-7 measured
    drop = kd * np.abs(a1r) ** 2
    assert np.max(np.abs(drop - w.powers["drop"])) < 1e-5 * drop.max()                              # 1.7e-6
    assert np.max(np.abs(ke * np.abs(sol.y[4] + 1j * sol.y[5]) ** 2 - w.powers["refl"])) < 5e-6


def test_simulate_input_guards():
    """A precomputed junction run must belong to the same pattern, bit rate and levels; oversized grids are refused
    before allocation (MAX_HALF_STEPS); a cyclic pattern without transitions cannot have 0 < p1 < 1."""
    ring = RingTCMT.from_rates(*_rates(), 0.3 * KAP, v_bias=1.0, ddelta_dv=0.4 * KAP)
    dj = DriverJunction(np.linspace(-0.5, 4, 46), 10e-15, 100.0, 50.0)
    bits = prbs(7)[:16]
    lo, hi, _ = dj.levels(1.0, 1.2)
    jr = dj.run(bits, 20e-12, 8, lo, hi)
    assert jr.levels == (lo, hi)
    simulate(ring, dj, bits, 50e9, v_bias=1.0, swing_vpp=1.2, junction=jr)
    for kw in ({"bits": bits[::-1].copy()}, {"bitrate_hz": 40e9}, {"swing_vpp": 1.0}):
        args = {"bits": bits, "bitrate_hz": 50e9, "swing_vpp": 1.2} | kw
        with pytest.raises(ValueError):
            simulate(ring, dj, args["bits"], args["bitrate_hz"], v_bias=1.0, swing_vpp=args["swing_vpp"],
                     junction=jr)
    with pytest.raises(ValueError, match="MAX_HALF_STEPS"):
        dj.run(prbs(15), 20e-12, 1024, lo, hi)
    with pytest.raises(ValueError):
        markov_bits(256, 0.5, 0.0)


def test_transition_energy_matches_p1_8_formula():
    """Energy dissipated in R_s per transition (mean of up and down) = mrm.plant.electrical.data_activity_heat_row
    junction share 0.5 C_eff Vpp^2 R_s/(R_s + R_drv) for the nonlinear C_j(V) (07 P1-8, cross-module check)."""
    tab = el.analytic_junction_table(shift_amp_m=170e-12, v_bi_v=0.9, m=0.5, lambda_m=1.31e-6, n_g=4.0,
                                     c0_f_per_m=15e-15 / 33e-6, alpha0_per_m=1500.0, loss_ratio_q=0.06,
                                     r_series_ohm_m=100.0 * 33e-6)
    cj = tab.capacitance_f_per_m[0] * 33e-6
    dj = DriverJunction(tab.v_reverse_v, cj, 100.0, 50.0)
    bits = np.array([0] * 8 + [1] * 8, np.uint8)
    lo, hi = 0.35, 1.65
    run = dj.run(bits, 50e-12, 400, lo, hi, n_warm_bits=8)
    i = (run.v_src - run.v_j) / 150.0
    m = run.t_s >= 0
    tg = run.t_s[m]
    e_rs = np.trapezoid(i[m] ** 2 * 100.0, tg)          # one up and one down transition in the window
    ce = np.zeros(1)
    el.junction_swing_capacitance_row(np.array([lo]), np.array([hi]), tab.v_reverse_v, tab.capacitance_f_per_m[0],
                                      np.array([1.0]), np.array([33e-6]), ce)
    pj = np.zeros(1)
    pd = np.zeros(1)
    el.data_activity_heat_row(ce, np.array([hi - lo]), 1.0, np.array([1.0]), np.array([1.0]), np.array([100.0]),
                              np.array([50.0]), np.array([0.0]), np.array([0.0]), 0, pj, pd)
    assert e_rs / 2 == pytest.approx(pj[0], rel=2e-3)


# ============================================================================================ eye
def test_eye_metrics_on_ideal_and_delayed_nrz():
    s = 32
    bits = prbs(7)
    p = np.repeat(np.where(bits == 1, 3.0, 1.0), s)
    em = eye_metrics(p, bits, s, 20e-12)
    assert (em.p1, em.p0, em.oma, em.eye_height) == (3.0, 1.0, 2.0, 2.0)
    assert em.er_db == pytest.approx(10 * math.log10(3.0))
    assert em.eye_width_ui == pytest.approx(1.0, abs=1.0 / s)          # an ideal edge falls between samples
    assert em.mean == pytest.approx(np.mean(p))
    # a lag of 0.7 UI and a smooth edge leave the levels and the OMA unchanged
    kern = np.exp(-np.arange(12) / 3.0)
    ps = np.convolve(np.tile(p, 3), kern / kern.sum())[p.size:2 * p.size]
    ps = np.roll(ps, int(0.7 * s))
    em2 = eye_metrics(ps, bits, s, 20e-12)
    # the level difference peaks just before the next (kinked) edge: the parabola refinement may overshoot by
    # <= |h(-1) - h(+1)|/8 of a sample step there (here 1e-3); for smooth optical responses it converges as 1/S^2
    assert em2.oma == pytest.approx(2.0, abs=5e-3)
    assert em2.p1 == pytest.approx(3.0, abs=5e-3) and em2.p0 == pytest.approx(1.0, abs=5e-3)
    assert em2.oma == pytest.approx(em2.p1 - em2.p0, abs=1e-12)
    # symmetric linear edges cross the mid level at the same phase every bit: the eye stays one UI wide
    assert 0.9 < em2.eye_width_ui <= 1.0 and 1.9 < em2.eye_height <= 2.0 + 1e-9
    # inverted polarity (a drop port, or the through port on the other side of the resonance): same eye, levels
    # swapped, polarity -1
    em3 = eye_metrics(4.0 - ps, bits, s, 20e-12)
    assert em3.polarity == -1 and em2.polarity == 1
    assert em3.oma == pytest.approx(em2.oma, abs=1e-12) and em3.eye_height == pytest.approx(em2.eye_height, abs=1e-12)
    assert em3.eye_width_ui == pytest.approx(em2.eye_width_ui, abs=1e-12)
    assert em3.p1 == pytest.approx(4.0 - em2.p1, abs=1e-12) and em3.p0 == pytest.approx(4.0 - em2.p0, abs=1e-12)


# ============================================================================================ plant consistency
@pytest.fixture(scope="module")
def g26():
    from mrm import params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load("g26")
        pp = params.realize(cfg, 1, seeds=0, uncertainty="nominal", variation="off")
    return pp


def test_ring_from_plant_params_matches_plant_optics(g26):
    """The tables come from mrm.plant.phase; the static TCMT ring equals the plant's exact ring to ~1/F^2."""
    pp = g26
    ring = RingTCMT.from_plant_params(pp, 0, 0, v_bias=1.0, x_bias=0.5)
    assert ring.x(1.0) == pytest.approx(0.5, abs=1e-9)
    P = ph.PhaseParams.from_plant_params(pp)
    n_r = pp.n_rings
    lam = np.array([ring.meta["lambda_line_m"]])
    for v in (0.2, 1.0, 1.8):
        o = {k: np.zeros((n_r, 1)) for k in ("p0", "p1", "a0", "a1", "e0", "e1", "kt", "kd", "trt")}
        ph.ring_optics_row(P.ring[0], lam, np.full(n_r, ring.meta["t_ring_k"]), np.full(n_r, ring.meta["t_ring_k"]),
                           np.full(n_r, v), np.full(n_r, v), P.grid, P.dneff[0], P.alpha_fca[0], P.capacitance[0],
                           P.r_series[0], o["p0"], o["p1"], o["a0"], o["a1"], o["e0"], o["e1"], o["kt"], o["kd"],
                           o["trt"], np.zeros((n_r, 2)), np.zeros((n_r, 2)), np.zeros((n_r, 2)), np.zeros(n_r, np.int8))
        ex = op.ring_exact(o["p0"][0, 0], o["kt"][0, 0], o["kd"][0, 0], o["a0"][0, 0], o["trt"][0, 0])
        st = ring.steady(v)
        assert abs(st["thru"][0] - ex.p_thru) < 2e-3
        assert abs(st["drop"][0] / ex.p_drop - 1) < 5e-3
        assert abs(st["loss"][0] / ex.p_loss - 1) < 5e-3


def test_ring_from_plant_params_lock_order(g26):
    """The lock order: the cold-resonance order by default (no warning in g26 nominal), any other with ``order``
    (order m + 1 sits one FSR BLUE of order m, so it needs about FSR/s more heating and lands on the same x_bias);
    a line more than 1.5 FSR away from the cold resonance warns (sampled worlds); t_ring_k and x_bias together are
    refused."""
    pp = g26
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r0 = RingTCMT.from_plant_params(pp, 0, 0, v_bias=1.0, x_bias=0.5)
        m0 = r0.meta["order"]
        r1 = RingTCMT.from_plant_params(pp, 0, 0, v_bias=1.0, x_bias=0.5, order=m0)
        r2 = RingTCMT.from_plant_params(pp, 0, 0, v_bias=1.0, x_bias=0.5, order=m0 + 1)
    assert r1.meta["t_ring_k"] == pytest.approx(r0.meta["t_ring_k"], abs=1e-9)
    assert r2.meta["order"] == m0 + 1 and r2.x(1.0) == pytest.approx(0.5, abs=1e-9)
    P = ph.PhaseParams.from_plant_params(pp)
    rp = P.ring[0][0]
    fsr = rp[ph.RP_LAM_RES] ** 2 / (rp[ph.RP_N_G] * rp[ph.RP_LEN])
    s_mean = fsr / (r2.meta["t_ring_k"] - r0.meta["t_ring_k"])          # mean thermal slope over one FSR
    assert 0.5 * rp[ph.RP_S0] < s_mean < 2.0 * rp[ph.RP_S0]
    with pytest.warns(UserWarning, match="lock order"):
        RingTCMT.from_plant_params(pp, 0, 0, v_bias=1.0, x_bias=0.5, lambda_line_m=r0.meta["lambda_line_m"] + fsr)
    with pytest.raises(ValueError):
        RingTCMT.from_plant_params(pp, 0, 0, v_bias=1.0, x_bias=0.5, t_ring_k=350.0)


def test_g26_prbs_waveform_converges_and_is_near_the_static_average(g26):
    """End-to-end g26 run (PRBS7, 50 Gb/s, 1.3 Vpp centered at 1 V, 8 ps driver edges): converged in the sample
    count; the mean drop power within 3 % of the two-level static average (07 P2-6)."""
    pp = g26
    ring = RingTCMT.from_plant_params(pp, 0, 0, v_bias=1.0, x_bias=1 / math.sqrt(3))
    dj = DriverJunction.from_plant_params(pp, 0, 0, rise_time_s=8e-12)
    res = {}
    for s in ((16, 32) if NOJIT else (32, 128)):
        w = simulate(ring, dj, prbs(7), float(pp.datapath.bitrate_hz[0, 0]), v_bias=1.0, swing_vpp=1.3,
                     samples_per_bit=s)
        res[s] = (w.mean_correction("drop"), w.eye("thru").oma, w.eye("thru").eye_height)
    a, b = res.values()
    assert abs(a[0] - b[0]) < 1e-4
    assert abs(a[1] / b[1] - 1) < 2e-3
    assert abs(b[0] - 1.0) < 0.03


# ============================================================================================ normalized LUT model
def test_normalized_model_is_the_physical_tcmt():
    """The normalized node reproduces the dimensional TCMT run driven by the same single-pole edge exactly: mean
    energy, through and drop OMA (the r_e algebra of lut.py)."""
    ki, ke, kd = _rates(0.4, 0.55, 0.05)
    tb = 20e-12
    tau = 1.0 / KAP / tb
    x0, x1 = -0.2, 1.1
    tr_ui = 0.2
    ring = RingTCMT.from_rates(ki, ke, kd, x0 * KAP / 2, v_bias=0.0, ddelta_dv=(x1 - x0) * KAP / 2)
    bits = prbs(7)
    s = 32
    nw = 32
    # the exact single-pole edge at the half-step instants (the same recursion as lut._node_nb)
    q = math.exp(-0.5 * (1.0 / s) * math.log(9.0) / tr_ui)
    seq = np.concatenate([bits[-nw:], bits, bits[:1]]).astype(float)
    e = float(bits[-nw])
    vh = [e]
    for bk in seq:
        for _ in range(s):
            em = bk + (e - bk) * q
            e = bk + (em - bk) * q
            vh += [em, e]
    tr = ring.run(np.array(vh), tb / s)
    a1 = tr.a1[nw * s:(nw + bits.size) * s]
    u = np.abs(a1) ** 2 * KAP ** 2 / (4 * ke)
    d = L.normalized_node(x0, x1, bits, tau, tr_ui, re_grid=(ke / KAP,), samples_per_bit=s, n_warm=nw)
    assert np.mean(u) == pytest.approx(d["u_td"], rel=1e-9)
    em = eye_metrics(tr.powers["thru"][nw * s:(nw + bits.size) * s], bits, s)
    assert em.oma == pytest.approx(d["oma_thru_td"][0], rel=1e-7)
    emd = eye_metrics(u, bits, s)
    assert emd.oma == pytest.approx(d["oma_drop_td"], rel=1e-7)


def test_normalized_mirror_symmetry():
    """x(t) -> -x(t) with relabelled levels leaves the corrections unchanged (stored half-plane of the LUT)."""
    bits = markov_bits(256, 0.3, 0.35)
    a = L.normalized_node(0.4, 1.3, bits, 0.12, 0.25, re_grid=(0.45,))
    # mirrored trajectory: bit 0 -> -0.4, bit 1 -> -1.3; relabelled so that level 1 has the larger x
    b = L.normalized_node(-1.3, -0.4, (1 - bits).astype(np.uint8), 0.12, 0.25, re_grid=(0.45,))
    for k in ("u_td", "u_st", "oma_drop_td", "oma_drop_st"):
        assert a[k] == pytest.approx(b[k], rel=1e-10)
    assert a["oma_thru_td"][0] == pytest.approx(b["oma_thru_td"][0], rel=1e-10)


def test_p2_6_mean_drop_correction_prbs_tau3ps_ui20ps():
    """07 P2-6: the correction to the mean drop power for PRBS at tau_p ~ 3 ps and a 20 ps UI (tau_p/UI = 0.15).

    Honest result (PRBS9, full grid in the artefact sidecar, validation.p2_6_prbs9_tau_0p15): the 07 claim holds only
    for small swings. Over every detuning x_c in [0, 6] HWHM and edge rise time 0-0.7 UI, |c_mean - 1| stays below 3 %
    up to a resonance swing of 0.25 FWHM (max 2.6 %; the G25-like drive of 45-65 pm at Q 3500-4000 is 0.13-0.2 FWHM).
    At 0.3 FWHM it reaches 2.4 % (ideal edges) to 3.7 % (0.7 UI edges); at 0.5 FWHM 9.6 %, at 1 FWHM 30 % (x_c = 0:
    the resonance sweeps across the laser during every edge). The 170-230 pm swing that 07 D9 says the printed relOMA
    needs (0.5-0.7 FWHM) is therefore NOT inside the < 3 % regime: there fidelity.data_modulation: lut matters."""
    xc = [0.0, 0.5, 1.0, 1.5, 2.5] if NOJIT else L.DEFAULT_AXES["xc"]
    w = [0.1, 0.25, 0.3, 1.0] if NOJIT else [0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 1.0]
    tr = [0.0, 0.35, 0.7] if NOJIT else L.DEFAULT_AXES["tr"]
    r = L.p2_6_prbs_check(0.15, order=7 if NOJIT else 9, xc=xc, w=w, tr=tr)
    c = np.abs(np.array(r["c_mean"]) - 1.0)
    small = [i for i, x in enumerate(w) if x <= 0.25]
    assert c[:, small, :].max() < 0.03
    assert c[:, w.index(0.3), :].max() > 0.03          # the honest limit: 0.3 FWHM already exceeds 3 % for slow edges
    assert 0.2 < c.max() < 0.4                          # 1 FWHM at x_c = 0


# ============================================================================================ LUT artefact
@pytest.fixture(scope="module")
def lut():
    try:
        return L.load_lut("nrz")
    except FileNotFoundError:
        pytest.skip("waveform_lut_nrz.npz not built (python -m mrm.waveform.lut)")


def test_lut_artefact_integrity_and_domain(lut):
    assert lut.config["schema"] == L.SCHEMA and len(lut.config_hash) == 64
    assert lut.tab6.shape[0] == L.N_F6 and np.all(np.isfinite(lut.tab6)) and np.all(np.isfinite(lut.tab7))
    dom = lut.domain
    assert dom["abs_xc"][1] >= 6.0 and dom["w"][1] >= 1.0 and dom["tau_ui"][0] <= 0.07 and dom["tau_ui"][1] >= 0.25
    # tau_p = 2.7 ps at 50 and 56 Gb/s (the g26 ring) is inside
    assert dom["tau_ui"][0] < 2.7e-12 * 50e9 < 2.7e-12 * 56e9 < dom["tau_ui"][1]
    # P2-6: an error outside the domain, naming the key
    for bad, key in (((7.0, 7.4, 0.5, 0.5, 0.1, 0.1, 0.5), "x_c"), ((1.0, 4.0, 0.5, 0.5, 0.1, 0.1, 0.5), "w"),
                     ((0.5, 1.0, 0.9, 0.1, 0.1, 0.1, 0.5), "mark"),
                     ((0.5, 1.0, 0.35, 0.65, 0.1, 0.1, 0.5), "transition"),
                     ((0.5, 1.0, 0.5, 0.5, 0.9, 0.1, 0.5), "rise"), ((0.5, 1.0, 0.5, 0.5, 0.1, 0.5, 0.5), "tau_p"),
                     ((0.5, 1.0, 0.5, 0.5, 0.1, 0.1, 0.9), "kappa_e"), ((np.nan, 1.0, 0.5, 0.5, 0.1, 0.1, 0.5),
                                                                       "non-finite")):
        with pytest.raises(L.LutDomainError, match=key):
            lut.eval(*bad)
    out = np.zeros(L.N_OUT)
    st = L.lut_eval_nb(*lut.kernel_args(), 7.0, 10.0, 0.5, 0.5, 0.1, 0.1, 0.5, out)
    assert st == (L.ST_XC | L.ST_W) and np.all(np.isfinite(out))
    with pytest.raises(ValueError):
        L.lut_eval_strict_nb(*lut.kernel_args(), 7.0, 10.0, 0.5, 0.5, 0.1, 0.1, 0.5, out)
    assert L.lut_eval_strict_nb(*lut.kernel_args(), 0.2, 0.8, 0.5, 0.5, 0.1, 0.1, 0.5, out) == 0


def test_lut_nodes_are_the_normalized_runs(lut):
    """At a node the table returns the stored run exactly (same build pattern, regenerated from the seed)."""
    a = lut.axes
    cfgd = lut.config
    ip, it = 1, 3                                   # p1 = 0.5, td = 0.45
    bits = markov_bits(cfgd["n_bits"], a["p1"][ip], a["td"][it], seed=cfgd["seed"] + 1000 * ip + 100 * it)
    for xc, w, tr, tau, re_ in ((1.0, 0.3, 0.2, 0.15, 0.5), (0.25, 1.0, 0.0, 0.07, 0.3), (3.0, 0.1, 0.7, 0.25, 0.7)):
        d = L.normalized_node(xc - w, xc + w, bits, tau, tr, re_grid=(re_,), n_warm=cfgd["n_warm"],
                              samples_per_bit=L.samples_per_bit_for(tau, cfgd["base_samples"]))
        r = lut.eval(xc - w, xc + w, a["p1"][ip], a["td"][it], tr, tau, re_)
        assert r["c_mean"] == pytest.approx(d["c_mean"], rel=1e-6)          # float32 storage
        assert r["c_oma_thru"] == pytest.approx(d["oma_thru_td"][0] / d["oma_thru_st"][0], rel=1e-6)
        assert r["c_oma_drop"] == pytest.approx(d["oma_drop_td"] / d["oma_drop_st"], rel=1e-6)


def test_lut_query_mapping_mirror_and_relabel(lut):
    """The engine queries with physical levels in any order and on either side of the resonance: lut_eval_nb maps
    (x0, x1, p1) to the stored half-plane (x_c >= 0, p = density of the larger-x level). At a node all four
    equivalent queries return the stored run, and a direct run of the MIRRORED dynamics gives the same numbers."""
    a = lut.axes
    cfgd = lut.config
    ip, it = 0, 2                                   # p1 = 0.3 (relabelling matters), td = 0.45
    p1, td = a["p1"][ip], a["td"][it]
    bits = markov_bits(cfgd["n_bits"], p1, td, seed=cfgd["seed"] + 1000 * ip + 100 * it)
    xc, w, tr, tau, re_ = 1.0, 0.3, 0.2, 0.15, 0.4
    kw = dict(re_grid=(re_,), n_warm=cfgd["n_warm"], samples_per_bit=L.samples_per_bit_for(tau, cfgd["base_samples"]))
    d = L.normalized_node(xc - w, xc + w, bits, tau, tr, **kw)
    dm = L.normalized_node(-(xc - w), -(xc + w), bits, tau, tr, **kw)          # mirrored: blue/red swapped
    assert dm["c_mean"] == pytest.approx(d["c_mean"], rel=1e-10)
    ref = {"c_mean": d["c_mean"], "c_oma_thru": d["oma_thru_td"][0] / d["oma_thru_st"][0],
           "c_oma_drop": d["oma_drop_td"] / d["oma_drop_st"]}
    x0, x1 = xc - w, xc + w
    for q in ((x0, x1, p1), (x1, x0, 1 - p1), (-x0, -x1, p1), (-x1, -x0, 1 - p1)):
        r = lut.eval(*q, td, tr, tau, re_)
        assert r["status"] == 0
        for k, v in ref.items():
            assert r[k] == pytest.approx(v, rel=2e-6), (q, k)
    # an off-node query and its three equivalents agree with each other to rounding
    q0 = (-0.37, -1.21, 0.42, 0.33, 0.27, 0.13, 0.47)
    r0 = lut.eval(*q0)
    for q in ((-1.21, -0.37, 0.58), (0.37, 1.21, 0.42), (1.21, 0.37, 0.58)):
        r = lut.eval(*q, *q0[3:])
        for k in ("c_mean", "c_oma_thru", "d_oma_thru", "c_oma_drop", "d_oma_drop", "d_mean_u"):
            assert r[k] == pytest.approx(r0[k], rel=1e-12, abs=1e-15)


def test_lut_build_resolution_is_converged():
    """The build's samples per bit (samples_per_bit_for(tau, 32)) against 4x more at the hardest corners (smallest
    tau_p/UI, largest swing, x_c at and near resonance, fast and slow edges): the mean correction is converged to
    2e-5 and the OMAs to 8e-3 (measured 3.7e-3 to 5.2e-3, depending on the pattern, at x_c = 0, w = 1, t_r = 0.05,
    tau = 0.07), well inside the table's interpolation error (sidecar validation)."""
    bits = markov_bits(128 if NOJIT else 512, 0.5, 0.45, seed=3)
    corners = ((0.0, 1.0, 0.05, 0.07), (0.75, 1.0, 0.0, 0.07), (1.0, 0.5, 0.2, 0.07), (2.0, 1.0, 0.1, 0.085))
    for xc, w, tr, tau in corners[:2] if NOJIT else corners:
        s = L.samples_per_bit_for(tau, 32)
        a = L.normalized_node(xc - w, xc + w, bits, tau, tr, re_grid=(0.5,), samples_per_bit=s)
        b = L.normalized_node(xc - w, xc + w, bits, tau, tr, re_grid=(0.5,), samples_per_bit=4 * s)
        assert abs(a["c_mean"] / b["c_mean"] - 1) < 2e-5
        assert abs(a["oma_drop_td"] / b["oma_drop_td"] - 1) < 8e-3
        assert abs(a["oma_thru_td"][0] / b["oma_thru_td"][0] - 1) < 8e-3


@pytest.mark.skipif(NOJIT, reason="60 normalized runs of 1024 bits are too slow without numba")
def test_lut_interpolation_accuracy(lut):
    """Random off-node queries against direct runs with fresh patterns (interpolation + pattern sampling). In the
    practical domain (x_c 0.3-3, w <= 0.6, p1 = td = 0.5, t_r <= 0.5 UI, tau_p/UI 0.1-0.2, r_e 0.4-0.6) the mean
    correction is good to ~0.5 % and the through-OMA ratio to ~2 %; over the whole domain the mean stays within
    ~1.5 % while the OMA error grows near x_c -> 0 and at the edges of (p1, td, tau) (sidecar validation)."""
    r = L.interpolation_check(lut, n_random=40, seed=11, practical=True)
    assert r["c_mean_abs_err_max"] < 5e-3
    assert r["c_oma_thru_abs_err_max"] < 2.5e-2
    r = L.interpolation_check(lut, n_random=20, seed=12)
    assert r["c_mean_abs_err_max"] < 1.5e-2
    assert r["oma_thru_diff_err_rel_scale_p95"] < 5e-2


def test_lut_matches_full_driver_junction_runs_on_g26(lut, g26):
    """Engine path: LUT keys from the plant's optics inputs (lut_query_nb), times the static two-level average,
    against the full chain (Gaussian driver edge, nonlinear C_j(V), sqrt shift law, loss modulation) on g26 with
    PRBS9 at 50 Gb/s. The residual measures the canonical single-pole edge and the omitted loss modulation."""
    pp = g26
    tb = 1.0 / float(pp.datapath.bitrate_hz[0, 0])
    bits = prbs(7 if NOJIT else 9)
    worst_mean, worst_oma = 0.0, 0.0
    cases = ((1 / math.sqrt(3), 1.3, 8e-12), (1.2, 0.9, 4e-12), (-0.4, 1.3, 12e-12))
    for xb, vpp, tr_drv in cases[:1] if NOJIT else cases:
        ring = RingTCMT.from_plant_params(pp, 0, 0, v_bias=1.0, x_bias=xb)
        dj = DriverJunction.from_plant_params(pp, 0, 0, rise_time_s=tr_drv)
        w = simulate(ring, dj, bits, 1.0 / tb, v_bias=1.0, swing_vpp=vpp, samples_per_bit=32 if NOJIT else 64)
        lo, hi, _ = w.levels
        t_r = dj.step_rise_time_s(lo, hi)["mean"]
        x0, x1 = w.x_levels
        kap = 0.5 * (ring.kappa(lo) + ring.kappa(hi))
        r = lut.eval(x0, x1, float(bits.mean()), pattern_stats(bits)[1], t_r / tb, 1.0 / kap / tb,
                     ring.kappa_e / kap)
        pred_mean = r["c_mean"] * w.static["drop"]["mean"]
        worst_mean = max(worst_mean, abs(w.mean("drop") / pred_mean - 1))
        oma_st = abs(w.static["thru"]["p1"] - w.static["thru"]["p0"])
        worst_oma = max(worst_oma, abs(w.eye("thru").oma / (r["c_oma_thru"] * oma_st) - 1))
    assert worst_mean < 5e-3
    assert worst_oma < 4e-2          # 2.3 % at the slow red-side case: the canonical edge shape


def test_lut_query_from_plant_optics_inputs():
    """lut_query_nb maps the optics inputs (phi, K^2, a, t_rt) to (x0, x1, t_r/UI, tau_p/UI, r_e) consistently with
    the TCMT rate mapping of mrm.plant.optics."""
    ki, ke, kd = 3e11, 2.0e11, 0.03e11
    t_rt = 0.5555e-12
    k2t, k2d, a0 = op.tcmt_to_exact(ki, ke, kd, t_rt)
    a1 = math.exp(-0.5 * (ki * 0.98) * t_rt)
    phi0 = 2 * math.pi * 80 + 0.2e11 * t_rt
    phi1 = 2 * math.pi * 80 + 1.5e11 * t_rt
    out = np.zeros(5)
    L.lut_query_nb(phi0, phi1, k2t, k2d, a0, a1, t_rt, 50e9, 6e-12, out)
    kap = 0.5 * (ki + 0.98 * ki) + ke + kd
    assert out[0] == pytest.approx(2 * 0.2e11 / kap, rel=1e-9)
    assert out[1] == pytest.approx(2 * 1.5e11 / kap, rel=1e-9)
    assert out[2] == pytest.approx(0.3) and out[3] == pytest.approx(50e9 / kap) and out[4] == pytest.approx(ke / kap)


_NOJIT_SCRIPT = r"""
import json, math, numpy as np
from mrm.waveform import lut as L
from mrm.waveform.driver import DriverJunction, prbs
from mrm.waveform.eye import simulate
from mrm.waveform.tcmt_time import RingTCMT
kap = 2 * math.pi * 299792458.0 / 1310e-9 / 3500
ring = RingTCMT.from_rates(0.5 * kap, 0.45 * kap, 0.05 * kap, 0.3 * kap, v_bias=1.0, ddelta_dv=0.4 * kap,
                           dkappa_i_dv=-0.02 * kap, g=0.05j * kap)
dj = DriverJunction(np.linspace(-0.5, 4, 46), 10e-15, 100.0, 50.0, c_pad_f=3e-15, rise_time_s=6e-12)
w = simulate(ring, dj, prbs(7)[:40], 50e9, v_bias=1.0, swing_vpp=1.2, samples_per_bit=16, n_warm_bits=8)
d = L.normalized_node(0.2, 0.9, prbs(7), 0.15, 0.2, re_grid=(0.4, 0.6), samples_per_bit=16)
ax = {"xc": [0, 1, 2], "w": [0, 0.5], "p1": [0.3, 0.7], "td": [0, 0.6], "tr": [0, 0.5], "tau": [0.1, 0.2],
      "re": [0.4, 0.6]}
lut = L.build_lut(ax, n_bits=64, n_warm=8, base_samples=16)
r = lut.eval(0.3, 1.1, 0.5, 0.4, 0.3, 0.15, 0.5)
print(json.dumps({"thru": w.powers["thru"].tolist(), "vj": w.v_j.tolist(), "node": [d["u_td"], d["oma_drop_td"]]
                  + d["oma_thru_td"].tolist(), "lut": [r[k] for k in sorted(r)]}))
"""


def test_kernels_identical_without_numba():
    """The waveform kernels (junction RC, TCMT, normalized node, LUT build and lookup) give the same numbers under
    NUMBA_DISABLE_JIT=1 as compiled (to rounding)."""
    import json
    import os
    import subprocess
    import sys
    res = {}
    for flag in ("0", "1"):
        env = dict(os.environ, NUMBA_DISABLE_JIT=flag)
        out = subprocess.run([sys.executable, "-c", _NOJIT_SCRIPT], env=env, capture_output=True, text=True,
                             timeout=600)
        assert out.returncode == 0, out.stderr[-2000:]
        res[flag] = json.loads(out.stdout.strip().splitlines()[-1])
    for k in ("thru", "vj", "node", "lut"):
        np.testing.assert_allclose(res["0"][k], res["1"][k], rtol=1e-10, atol=1e-15)
