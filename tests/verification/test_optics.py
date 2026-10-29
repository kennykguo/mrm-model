"""Checks the optical plant mrm.plant.optics against analytic limits and an independent dense solver.

Read this first
---------------
These are code checks, not hardware checks. Each test builds a ring from its loaded Q and coupling fractions
(``ring_params``), runs the exact ring (``ring_exact``), the archived TCMT comparison (``archive.optics.tcmt.ring_tcmt``) or the bus (``bus_propagate``)
and compares the result with a closed form or with ``_dense_loop_solve`` (a dense scattering-matrix solve of the same
ring with point scatterers). The groups below cover: energy balance and the Airy closed forms; TCMT against its own
closed forms and against the exact ring (within 1 % inside +-0.3 FSR, a deviation that falls as 1/F^2); reciprocity,
unitarity and the mirrored ring; the doublet thresholds; the monitor identity, the phase derivatives and the
self-heating slope dP_abs/dlambda_res; the bus cascade, the NRZ average and the single-pass bound; the self-heating
adapter that mrm.plant.selfheat calls; and identical results with numba off.

The reference ring is the one of the compact-model notes (docs/research/03_mrm_compact_models.md, section 1 "Anchor
values for the reference ring"): lambda0 = 1310 nm, Q_L = 3500, FSR = 1.8 THz, kappa_e = kappa/2 and kappa_d = 1 %
of kappa (the add-drop case of those notes). The dB numbers quoted in the tests come from the tables of sections 2.7
and 2.8 of those notes and from their verification log. "Compact-model notes, section x" below means that file.
"The design review" below means docs/research/07_design_critique.md.
"""
import math
import os
import subprocess
import sys
import time

import numpy as np
import pytest
from numba import njit

from archive.optics import tcmt as archived_tcmt
from mrm.plant import optics

C0 = 299_792_458.0          # m/s (exact)
LAM0 = 1310e-9              # m, G26 printed centre wavelength (the reference ring)
W0 = 2 * math.pi * C0 / LAM0  # rad/s, optical angular frequency at LAM0
FSR = 1.8e12                # Hz, reference-ring FSR (inside window B, 1.72-1.88 THz, the design review's default)
TRT = 1.0 / FSR             # s, round-trip group delay
N0 = 80                     # an arbitrary longitudinal order near the real one (~75)
JIT_OFF = os.environ.get("NUMBA_DISABLE_JIT", "0") == "1"


def ring_params(q_l=3500.0, ke_f=0.5, kd_f=0.01, fsr=FSR):
    """Exact-ring parameters of a ring given in TCMT terms (logarithmic mapping, compact-model notes, section 2.6).

    Inputs: loaded Q q_l, the through and drop coupling rates as fractions ke_f, kd_f of the total rate kappa, and the
    FSR (Hz). Returns (K2_th, K2_dr, a, T_rt s, kappa 1/s).
    """
    trt = 1.0 / fsr
    kap = W0 / q_l
    ke, kd = ke_f * kap, kd_f * kap
    ki = kap - ke - kd
    k2t, k2d, a = (float(x) for x in optics.tcmt_to_exact(ki, ke, kd, trt))
    return k2t, k2d, a, trt, kap


def _airy_powers(phi, t1, t2, a):
    """Through and drop power of the add-drop ring per watt in (Airy closed forms, Bogaerts et al. 2012)."""
    den = 1 - 2 * t1 * t2 * a * np.cos(phi) + (t1 * t2 * a) ** 2
    thru = (t2 ** 2 * a ** 2 - 2 * t1 * t2 * a * np.cos(phi) + t1 ** 2) / den
    drop = (1 - t1 ** 2) * (1 - t2 ** 2) * a / den
    return thru, drop


def _dense_loop_solve(phi, t1, k1, t2, k2, a, scat):
    """Independent reference: a dense scattering-matrix solve of the ring with point scatterers at any positions.

    scat: list of (z/L, rho) with frequency-independent rho. Returns dict of complex port outputs for s_in = 1.
    """
    elems = sorted([(0.0, "C1", 0j), (0.5, "C2", 0j)] + [(z, "S", r) for z, r in scat], key=lambda e: e[0])
    ne = len(elems)

    def propagate(z1, z2):
        d = (z2 - z1) % 1.0
        return (a ** d) * np.exp(-1j * phi * d)

    A = np.zeros((2 * ne, 2 * ne), complex)
    b = np.zeros(2 * ne, complex)
    for j, (z, kind, r) in enumerate(elems):
        jl, jr = (j - 1) % ne, (j + 1) % ne
        pl, pr = propagate(elems[jl][0], z), propagate(z, elems[jr][0])
        iu, iv = 2 * j, 2 * j + 1
        A[iu, iu] += 1
        A[iv, iv] += 1
        if kind == "S":
            tau = np.sqrt(1 - abs(r) ** 2)
            A[iu, 2 * jl] -= tau * pl
            A[iu, 2 * jr + 1] -= -np.conj(r) * pr
            A[iv, 2 * jl] -= r * pl
            A[iv, 2 * jr + 1] -= tau * pr
        else:
            t, k, src = (t1, k1, 1.0) if kind == "C1" else (t2, k2, 0.0)
            A[iu, 2 * jl] -= t * pl
            b[iu] += -1j * k * src
            A[iv, 2 * jr + 1] -= t * pr
    x = np.linalg.solve(A, b)
    out = {}
    for j, (z, kind, r) in enumerate(elems):
        jl, jr = (j - 1) % ne, (j + 1) % ne
        uin = propagate(elems[jl][0], z) * x[2 * jl]
        vin = propagate(z, elems[jr][0]) * x[2 * jr + 1]
        if kind == "C1":
            out["thru"], out["refl"] = t1 - 1j * k1 * uin, -1j * k1 * vin
        elif kind == "C2":
            out["drop"], out["add"] = -1j * k2 * uin, -1j * k2 * vin
    return out


# -------------------------------------------------------------------------- energy balance, closed forms, special cases

def test_energy_balance_exact_all_ports():
    """|t|^2 + |d|^2 + |r|^2 + |add|^2 + loss = sum of inputs, to 1e-12, with backscatter and all ports lit."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        q_l = rng.uniform(1500, 20000)
        k2t, k2d, a, trt, _ = ring_params(q_l, rng.uniform(0.3, 0.7), rng.uniform(0.0, 0.3))
        rb = (rng.normal(size=3) + 1j * rng.normal(size=3)) * 0.05
        phi = 2 * math.pi * N0 + rng.uniform(-math.pi, math.pi, 64)
        s = rng.normal(size=(4, 64)) + 1j * rng.normal(size=(4, 64))
        r = optics.ring_exact(phi, k2t, k2d, a, trt, rb=rb, psi_rad=0.3, s_in=s[0], s_thru_in=s[1], s_drop_in=s[2],
                              s_add=s[3])
        pin = np.sum(np.abs(s) ** 2, axis=0)
        pout = r.p_refl + r.p_thru + r.p_drop + r.p_add_out + r.p_loss
        assert np.max(np.abs(pout - pin) / pin) < 1e-12


def test_energy_balance_bus_nrz_with_bus_loss():
    """Bus: input = output + segment losses + sum over rings of (drop + add + refl + loss), per line, to 1e-12."""
    rng = np.random.default_rng(2)
    n, nl = 5, 7
    k2t, k2d, a, trt, _ = ring_params(3500, 0.5, 0.05)
    phi = 2 * math.pi * N0 + rng.uniform(-math.pi, math.pi, (n, nl))
    rb = (rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))) * 0.04
    rb[2] = 0.0                                          # one ring on the Airy fast path
    t2 = rng.uniform(0.9, 1.0, n)
    p_in = rng.uniform(0.5e-3, 2e-3, nl)
    order = rng.permutation(n)
    nrz = dict(p1=rng.uniform(0, 1, n), phi1=phi + 0.08, a1=a * 0.995, eta1=0.9)
    r = optics.bus_propagate(p_in, phi, k2t, k2d, a, trt, 0.85, bus_order=order, bus_t2=t2, bus_t2_out=0.97, rb=rb,
                             nrz=nrz)
    seg_loss = np.zeros(nl)
    for k in range(n):
        seg_loss += r.p_in[k] / t2[k] * (1 - t2[k])
    last = order[-1]
    seg_loss += r.thru[last] * (1 - 0.97)
    sinks = (r.drop + r.add_out + r.refl + r.loss).sum(axis=0)
    np.testing.assert_allclose(r.p_out + seg_loss + sinks, p_in, rtol=1e-12, atol=0)
    np.testing.assert_allclose(r.drop_sum, r.drop.sum(axis=1), rtol=1e-14)
    np.testing.assert_allclose(r.abs_sum, r.absorbed.sum(axis=1), rtol=1e-14)


def test_airy_closed_forms_notch_and_drop_peak():
    """No backscatter: exact ring = Airy forms over a full FSR; notch depth and drop peak at resonance."""
    for q_l, ke_f, kd_f in ((3500, 0.5, 0.01), (2000, 0.7, 0.2), (8000, 0.3, 0.05)):
        k2t, k2d, a, trt, _ = ring_params(q_l, ke_f, kd_f)
        t1, t2 = math.sqrt(1 - k2t), math.sqrt(1 - k2d)
        phi = 2 * math.pi * N0 + np.linspace(-math.pi, math.pi, 2001)
        r = optics.ring_exact(phi, k2t, k2d, a, trt)
        th, dr = _airy_powers(phi, t1, t2, a)
        np.testing.assert_allclose(r.p_thru, th, atol=1e-13)
        np.testing.assert_allclose(r.p_drop, dr, atol=1e-14)
        assert np.all(r.p_refl == 0) and np.all(r.p_add_out == 0)
        res = optics.ring_exact(2 * math.pi * N0, k2t, k2d, a, trt)
        assert math.isclose(float(res.p_thru), ((t2 * a - t1) / (1 - t1 * t2 * a)) ** 2, rel_tol=1e-10,
                            abs_tol=1e-15)
        assert math.isclose(float(res.p_drop), (1 - t1 ** 2) * (1 - t2 ** 2) * a / (1 - t1 * t2 * a) ** 2,
                            rel_tol=1e-12)


def test_all_pass_special_case():
    """K_drop = 0: no drop, no add-port output, all-pass Airy through; critical coupling t = a nulls the through."""
    k2t, _, a, trt, _ = ring_params(3500, 0.5, 0.0)
    t1 = math.sqrt(1 - k2t)
    phi = 2 * math.pi * N0 + np.linspace(-math.pi, math.pi, 1001)
    rb = np.array([0.02 + 0.01j, -0.03j])
    r = optics.ring_exact(phi, k2t, 0.0, a, trt, rb=rb)
    assert np.all(r.p_drop == 0) and np.all(r.p_add_out == 0)
    np.testing.assert_allclose(r.p_thru + r.p_refl + r.p_loss, 1.0, atol=1e-13)
    r0 = optics.ring_exact(phi, k2t, 0.0, a, trt)
    allpass = (a * a - 2 * a * t1 * np.cos(phi) + t1 * t1) / (1 - 2 * a * t1 * np.cos(phi) + (a * t1) ** 2)
    np.testing.assert_allclose(r0.p_thru, allpass, atol=1e-13)
    crit = optics.ring_exact(2 * math.pi * N0, 1 - a * a, 0.0, a, trt)        # t1 = a: critical coupling
    assert float(crit.p_thru) < 1e-25
    assert math.isclose(float(crit.p_loss), 1.0, rel_tol=1e-12)
    bus = optics.bus_propagate(np.ones(3), np.full((1, 3), 2 * math.pi * N0) + [[0.0, 0.1, 1.0]], k2t, 0.0, a, trt,
                               1.0)
    np.testing.assert_allclose(bus.thru[0], optics.ring_exact(2 * math.pi * N0 + np.array([0.0, 0.1, 1.0]), k2t, 0.0,
                                                              a, trt).p_thru, rtol=1e-12, atol=1e-16)


def test_fsr_periodicity_and_per_order_backscatter():
    """Outputs are 2 pi periodic in phi when beta is the same for every order, 2 pi M_b periodic with M_b values,
    and the per-order values are really used (orders n and n + 1 differ)."""
    k2t, k2d, a, trt, _ = ring_params(3500, 0.5, 0.05)
    phi = 2 * math.pi * N0 + np.linspace(-1.0, 1.0, 101)
    rb1 = np.array([0.03 + 0.02j])
    base = optics.ring_exact(phi, k2t, k2d, a, trt, rb=rb1)
    for m in (-N0 - 7, -1, 1, 5):                  # includes negative absolute orders
        shifted = optics.ring_exact(phi + 2 * math.pi * m, k2t, k2d, a, trt, rb=rb1)
        for f in ("p_thru", "p_drop", "p_refl", "p_add_out", "p_loss", "u_cw", "u_ccw"):
            np.testing.assert_allclose(getattr(shifted, f), getattr(base, f), rtol=1e-9, atol=1e-15)
    rb3 = np.array([0.01, 0.03 + 0.02j, 0.05j])
    b3 = optics.ring_exact(phi, k2t, k2d, a, trt, rb=rb3)
    s3 = optics.ring_exact(phi + 2 * math.pi * 3, k2t, k2d, a, trt, rb=rb3)
    np.testing.assert_allclose(s3.p_refl, b3.p_refl, rtol=1e-9)
    s1 = optics.ring_exact(phi + 2 * math.pi, k2t, k2d, a, trt, rb=rb3)
    assert np.max(np.abs(s1.p_refl - b3.p_refl)) > 1e-3 * np.max(b3.p_refl)


def test_constant_scatterer_matches_independent_solver():
    """rb = [r, -r] makes the scatterer pair frequency-independent; a dense independent solve then agrees to
    rounding at every phase (checks the transfer-matrix algebra, the coupler and scatterer conventions)."""
    k2t, k2d, a, trt, _ = ring_params(3000, 0.55, 0.1)
    t1, k1, t2, k2 = math.sqrt(1 - k2t), math.sqrt(k2t), math.sqrt(1 - k2d), math.sqrt(k2d)
    r = 0.04 + 0.03j
    rho_h = -0.5j * np.conj(r)                     # the pair's reflectance for even orders (and, here, all orders)
    for ph in 2 * math.pi * N0 + np.linspace(-3.0, 3.0, 13):
        ref = _dense_loop_solve(ph, t1, k1, t2, k2, a, [(0.25, rho_h), (0.75, rho_h)])
        mod = optics.ring_exact(ph, k2t, k2d, a, trt, rb=np.array([r, -r]))
        for key, attr in (("thru", "thru"), ("drop", "drop"), ("refl", "refl"), ("add", "add_out")):
            assert abs(ref[key] - complex(getattr(mod, attr))) < 1e-12


def test_phase_parts_rotation_and_flat_nodes():
    """The shared cos/sin rotation gives p = exp(-(xi + i phi)/4) for any phi (negative orders included), and the
    per-order reflectance is flat at every resonance node (d rho/d phi = 0 at phi = 2 pi n)."""
    rb = np.array([0.02 + 0.01j, -0.01, 0.03j])
    for ph in np.concatenate([np.linspace(-60.0, 60.0, 97), 2 * math.pi * np.arange(-5, 6)]):
        p, ip, rho, drho = optics._phase_parts(ph, rb, 0.1)
        assert abs(p - np.exp(-0.25 * (0.1 + 1j * ph))) < 1e-14
        assert abs(p * ip - 1) < 1e-14
        rho2, drho2 = optics._rho_pair(ph, rb)
        assert abs(rho - rho2) < 1e-15 and abs(drho - drho2) < 1e-15
    for n in range(-4, 5):
        _, _, rho, drho = optics._phase_parts(2 * math.pi * n, rb, 0.1)
        assert abs(drho) < 1e-15
        assert abs(rho - (-0.5j) * (-1) ** n * np.conj(rb[n % 3])) < 1e-15


def test_plain_exact_line_matches_the_form_in_use():
    """optics._exact_line (the plain form of the loop, which nothing in the package calls) gives the same outputs as
    optics._exact_line_d (the form in use, with phase slopes): the amplitudes at the four ports, the stored energies
    and the loss power, with backscatter and with light entering at every port."""
    rb = np.array([0.02 + 0.01j, -0.01, 0.03j])
    k2_th, k2_dr, xi = 0.08, 0.002, 0.03
    t1, k1 = math.sqrt(1 - k2_th), math.sqrt(k2_th)
    t2, k2 = math.sqrt(1 - k2_dr), math.sqrt(k2_dr)
    s_in = (1.0 + 0.0j, 0.1 - 0.05j, 0.02j, 0.03 + 0.01j)        # ports 0..3 [sqrt(W)]
    for ph in np.linspace(2 * math.pi * N0 - 0.4, 2 * math.pi * N0 + 0.4, 41):
        plain = optics._exact_line(ph, xi, t1, k1, t2, k2, TRT, rb, *s_in)
        in_use = optics._exact_line_d(ph, xi, t1, k1, t2, k2, TRT, rb, *s_in)[:7]
        np.testing.assert_allclose(np.array(plain), np.array(in_use), rtol=1e-12, atol=1e-15)


# -------------------------------------------------------------- TCMT: its own closed forms, then against the exact ring

def test_tcmt_single_order_closed_forms():
    """ring_tcmt(M = 1) reproduces the closed forms of the compact-model notes, section 2.5, exactly (Lorentzians;
    backscatter closed forms)."""
    k2t, k2d, a, trt, _ = ring_params(3500, 0.5, 0.05)
    ki, ke, kd = optics.exact_to_tcmt(k2t, k2d, a, trt)
    kap = ki + ke + kd
    phi = 2 * math.pi * N0 + np.linspace(-0.6, 0.6, 301)
    delta = (phi - 2 * math.pi * N0) / trt
    r = archived_tcmt.ring_tcmt(phi, k2t, k2d, a, trt, m_orders=1)
    lor = delta ** 2 + kap ** 2 / 4
    np.testing.assert_allclose(r.p_thru, 1 - ke * (ki + kd) / lor, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(r.p_drop, ke * kd / lor, rtol=1e-12)
    np.testing.assert_allclose(r.u_cw, ke / lor, rtol=1e-12)
    np.testing.assert_allclose(r.p_loss, ki * ke / lor, rtol=1e-12)
    g = 0.2 * kap * np.exp(0.4j)
    rb = np.array([g * trt])
    rg = archived_tcmt.ring_tcmt(phi, k2t, k2d, a, trt, rb=rb, m_orders=1)
    D = kap / 2 + 1j * delta
    det = D ** 2 + abs(g) ** 2
    np.testing.assert_allclose(rg.thru, 1 - ke * D / det, rtol=1e-12)
    np.testing.assert_allclose(np.abs(rg.drop), np.abs(np.sqrt(ke * kd) * D / det), rtol=1e-12)
    np.testing.assert_allclose(rg.refl, 1j * ke * np.conj(g) / det, rtol=1e-12)
    np.testing.assert_allclose(np.abs(rg.add_out), np.abs(np.sqrt(ke * kd) * np.conj(g) / det), rtol=1e-12)


def test_mapping_roundtrip_and_beta():
    k2t, k2d, a, trt, kap = ring_params(3500, 0.5, 0.01)
    ki, ke, kd, beta = optics.exact_to_tcmt(k2t, k2d, a, trt, rb=0.02 + 0.01j)
    assert math.isclose(float(ki + ke + kd), kap, rel_tol=1e-12)
    back = optics.tcmt_to_exact(ki, ke, kd, trt)
    assert np.allclose(back, (k2t, k2d, a), rtol=1e-14)
    assert np.isclose(optics.beta_to_rb(complex(beta), FSR), 0.02 + 0.01j, rtol=1e-14)
    assert math.isclose(optics.beta_to_rb(5e9, FSR).real, 2 * math.pi * 5e9 / FSR)
    assert math.isclose(optics.dphi_dlambda_res(trt, LAM0), 2 * math.pi * C0 * trt / LAM0 ** 2)
    assert optics.nrz_average(0.25, 4.0, 8.0) == 7.0


@pytest.mark.parametrize("m_orders,q_min", [(1, 2000), (3, 2000), (7, 2500)])
def test_tcmt_vs_exact_within_1pct(m_orders, q_min):
    """TCMT within 1 % of the exact ring within +-0.3 FSR (the design review asks this for Q > 1000): through power (of
    the input), drop power and intrinsic-loss power (each of its own peak, so a weak drop coupler is not tested
    vacuously). At FSR 1.8 THz this holds from Q_L = 2000 (F = 15.7) for M = 1 and 3 and from Q_L = 2500 for M = 7
    (1.16 % at Q_L = 2000, kappa_e = 0.6 kappa, kappa_d = 0.2 kappa). At Q_L = 1000 (F = 7.9) the deviation is 1.5 to
    3 %; see test_tcmt_deviation_scales_as_inverse_finesse_squared."""
    phi = 2 * math.pi * N0 + 2 * math.pi * np.linspace(-0.3, 0.3, 1201)
    for q_l in (q_min, 3500, 7000):
        for ke_f, kd_f in ((0.5, 0.01), (0.6, 0.2), (0.4, 0.0)):
            k2t, k2d, a, trt, _ = ring_params(q_l, ke_f, kd_f)
            e = optics.ring_exact(phi, k2t, k2d, a, trt)
            t = archived_tcmt.ring_tcmt(phi, k2t, k2d, a, trt, m_orders=m_orders)
            assert np.max(np.abs(t.p_thru - e.p_thru)) < 0.01
            if kd_f > 0:
                assert np.max(np.abs(t.p_drop - e.p_drop)) < 0.01 * e.p_drop.max()
            assert np.max(np.abs(t.p_loss - e.p_loss)) < 0.01 * e.p_loss.max()


def test_tcmt_deviation_scales_as_inverse_finesse_squared():
    """Near resonance (|Delta| < 1.5 kappa) the TCMT-exact deviation times F^2 is constant within 15 % from F = 8 to
    F = 110, for the add-drop powers and, with backscatter, for the reflection, add-port output and CCW energy."""
    beta_shape = np.array([0.7 + 0.2j, -0.3 + 0.6j, 0.9j])          # random-like per-order beta, scaled per Q below
    for quantity in ("thru_drop", "backscatter"):
        scaled = []
        for q_l in (1000, 2000, 4000, 8000, 14000):
            k2t, k2d, a, trt, kap = ring_params(q_l, 0.5, 0.2)
            f = FSR / (kap / (2 * math.pi))
            phi = 2 * math.pi * N0 + np.linspace(-3, 3, 1201) * kap * trt / 2
            if quantity == "thru_drop":
                e = optics.ring_exact(phi, k2t, k2d, a, trt)
                t = archived_tcmt.ring_tcmt(phi, k2t, k2d, a, trt, m_orders=1)
                dev = max(np.abs(t.p_thru - e.p_thru).max(), np.abs(t.p_drop - e.p_drop).max())
            else:
                rb = 0.15 * kap * trt * beta_shape                      # |g|/kappa fixed at 0.15 x O(1)
                e = optics.ring_exact(phi, k2t, k2d, a, trt, rb=rb)
                t = archived_tcmt.ring_tcmt(phi, k2t, k2d, a, trt, rb=rb, m_orders=1)
                dev = max(np.abs(t.p_refl - e.p_refl).max() / e.p_refl.max(),
                          np.abs(t.p_add_out - e.p_add_out).max() / e.p_add_out.max(),
                          np.abs(t.u_ccw - e.u_ccw).max() / e.u_ccw.max())
            scaled.append(dev * f * f)
        scaled = np.array(scaled)
        assert scaled.max() / scaled.min() < 1.15, (quantity, scaled)


def test_single_scatterer_would_be_first_order():
    """Why the pair at L/4, 3L/4: one scatterer at L/4 (independent solver) deviates from TCMT at O(1/F), the pair
    at O(1/F^2). This documents the design choice explained in the mrm.plant.optics module docstring."""
    out = {}
    for q_l in (3500, 14000):
        k2t, k2d, a, trt, kap = ring_params(q_l, 0.5, 0.2)
        t1, k1, t2, k2 = math.sqrt(1 - k2t), math.sqrt(k2t), math.sqrt(1 - k2d), math.sqrt(k2d)
        g = 0.15 * kap * np.exp(0.7j)
        rb = np.array([g * trt])
        phis = 2 * math.pi * N0 + np.linspace(-3, 3, 61) * kap * trt / 2
        tc = archived_tcmt.ring_tcmt(phis, k2t, k2d, a, trt, rb=rb, m_orders=1).p_refl
        rho_single = -1j * np.exp(1j * 2 * math.pi * N0 * 0.5) * np.conj(rb[0])
        single = np.array([abs(_dense_loop_solve(p, t1, k1, t2, k2, a, [(0.25, rho_single)])["refl"]) ** 2
                           for p in phis])
        pair = optics.ring_exact(phis, k2t, k2d, a, trt, rb=rb).p_refl
        out[q_l] = (np.abs(single - tc).max() / tc.max(), np.abs(pair - tc).max() / tc.max())
    ratio_single = out[3500][0] / out[14000][0]          # ~4 for O(1/F)
    ratio_pair = out[3500][1] / out[14000][1]            # ~16 for O(1/F^2)
    assert 3.3 < ratio_single < 4.7
    assert 13 < ratio_pair < 19


def test_tcmt_order_window_is_centred_on_the_line():
    """The M orders summed are the M nearest the line for odd AND even M: a symmetric ring then responds
    symmetrically to +-detuning, and each order set equals an independent numpy sum over the expected orders.
    (Before the fix, M = 2 used orders n_c and n_c + 1 on both sides: -5.3 dB against +2.0 dB at -+0.4 FSR.)"""
    k2t, k2d, a, trt, _ = ring_params(3500, 0.5, 0.2)
    ki, ke, kd = (float(x) for x in optics.exact_to_tcmt(k2t, k2d, a, trt))
    kap = ki + ke + kd
    x = np.array([0.05, 0.2, 0.4, 0.49])                     # detuning in FSR
    for m in range(1, 7):
        up = archived_tcmt.ring_tcmt(2 * math.pi * (N0 + x), k2t, k2d, a, trt, m_orders=m)
        dn = archived_tcmt.ring_tcmt(2 * math.pi * (N0 - x), k2t, k2d, a, trt, m_orders=m)
        np.testing.assert_allclose(up.p_drop, dn.p_drop, rtol=1e-11)
        np.testing.assert_allclose(up.p_thru, dn.p_thru, rtol=1e-11)
        for xi, pu in zip(x, up.drop):
            lo = math.floor(N0 + xi + 0.5 - (m - 1) / 2)
            orders = np.arange(lo, lo + m)
            assert np.all(np.abs(orders - (N0 + xi)) <= m / 2 + 1e-12)       # the M nearest orders
            d = kap / 2 + 1j * 2 * math.pi * (N0 + xi - orders) / trt
            want = -np.sqrt(ke * kd) * np.sum((-1.0) ** orders / d)          # signed sum (notes, section 2.5)
            assert abs(pu - want) < 1e-12 * abs(want)


def test_exact_and_tcmt_complex_amplitudes_agree():
    """With one order the TCMT powers depend on |g| only, so a sign or conjugation error in the exact-to-TCMT
    backscatter mapping (rho_n = -(i/2)(-1)^n conj(r_b,n)) would not show in any power test. The complex port
    amplitudes (same reference planes) agree to O(1/F^2) at even and odd orders with complex per-order r_b."""
    dev = {}
    for q_l in (3500, 14000):
        k2t, k2d, a, trt, kap = ring_params(q_l, 0.5, 0.2)
        rb = kap * trt * np.array([0.15 * np.exp(0.9j), 0.10 * np.exp(-2.1j), 0.12 * np.exp(2.5j)])
        worst = 0.0
        for n in (N0, N0 + 1, N0 + 2):
            ph = 2 * math.pi * n + np.linspace(-2, 2, 21) * kap * trt / 2
            e = optics.ring_exact(ph, k2t, k2d, a, trt, rb=rb)
            t = archived_tcmt.ring_tcmt(ph, k2t, k2d, a, trt, rb=rb, m_orders=1)
            for f in ("refl", "thru", "drop", "add_out"):
                ee, tt = getattr(e, f), getattr(t, f)
                worst = max(worst, np.abs(ee - tt).max() / np.abs(ee).max())
        dev[q_l] = worst
    assert dev[3500] < 3e-3
    assert 12 < dev[3500] / dev[14000] < 20                    # (14000/3500)^2 = 16


def test_lorentzian_limit_of_the_exact_ring():
    """At high finesse (Q_L = 30000, F = 236) the exact ring tends to the Lorentzian closed forms (through, drop, CW
    energy, loss; compact-model notes, section 2.5), with the logarithmic rate mapping of section 2.6, to O(1/F^2).
    Within +-5 half-widths the adjacent orders add a relative (Delta T_rt)^2/(2 pi^2) <= 2.2e-4 to the drop
    (section 2.7)."""
    k2t, k2d, a, trt, _ = ring_params(30000, 0.45, 0.15)
    ki, ke, kd = (float(x) for x in optics.exact_to_tcmt(k2t, k2d, a, trt))
    kap = ki + ke + kd
    phi = 2 * math.pi * N0 + np.linspace(-5, 5, 401) * kap * trt / 2
    delta = (phi - 2 * math.pi * N0) / trt
    lor = delta ** 2 + kap ** 2 / 4
    e = optics.ring_exact(phi, k2t, k2d, a, trt)
    np.testing.assert_allclose(e.p_thru, 1 - ke * (ki + kd) / lor, atol=2e-4)
    np.testing.assert_allclose(e.p_drop, ke * kd / lor, rtol=5e-4)
    np.testing.assert_allclose(e.u_cw, ke / lor, rtol=5e-4)
    np.testing.assert_allclose(e.p_loss, ki * ke / lor, rtol=5e-4)
    assert math.isclose(float(e.p_thru[200]), ((ki + kd - ke) / kap) ** 2, abs_tol=2e-4)   # notch depth
    assert math.isclose(float(e.p_drop[200]), 4 * ke * kd / kap ** 2, rel_tol=2e-4)       # drop peak


def test_tcmt_tail_underestimate_400ghz():
    """The table of the compact-model notes, section 2.7 (reference ring, kappa_d = 1 % kappa): the exact ring and
    the single-order TCMT reproduce its Airy and TCMT columns to 0.03 dB. At 400 GHz the single-order TCMT
    under-estimates the drop by 0.71 dB (15 %). The multi-order complex sum with the (-1)^n sign recovers it (M = 7
    within 0.1 dB up to FSR/2, as the notes state)."""
    k2t, k2d, a, trt, _ = ring_params(3500, 0.5, 0.01)
    table = {32.7e9: (-3.014, -3.009, -20.00, -20.00), 200e9: (-0.119, -0.115, -32.66, -32.84),
             400e9: (-0.034, -0.029, -38.06, -38.77), 900e9: (-0.014, -0.006, -41.89, -45.81)}
    db = lambda x: 10 * math.log10(float(x))
    for df, (th_a, th_t, dr_a, dr_t) in table.items():
        ph = 2 * math.pi * N0 + 2 * math.pi * df * trt
        e = optics.ring_exact(ph, k2t, k2d, a, trt)
        t1 = archived_tcmt.ring_tcmt(ph, k2t, k2d, a, trt, m_orders=1)
        t7 = archived_tcmt.ring_tcmt(ph, k2t, k2d, a, trt, m_orders=7)
        assert abs(db(e.p_thru) - th_a) < 0.03 and abs(db(t1.p_thru) - th_t) < 0.03
        assert abs(db(e.p_drop) - dr_a) < 0.03 and abs(db(t1.p_drop) - dr_t) < 0.03
        assert abs(db(t7.p_drop) - db(e.p_drop)) < 0.11      # notes: "within 0.1 dB"; 0.106 dB at exactly FSR/2
    ph = 2 * math.pi * N0 + 2 * math.pi * 400e9 * trt
    under = 1 - float(archived_tcmt.ring_tcmt(ph, k2t, k2d, a, trt, m_orders=1).p_drop) / float(
        optics.ring_exact(ph, k2t, k2d, a, trt).p_drop)
    assert 0.13 < under < 0.17


def test_backscatter_reflection_table():
    """Compact-model notes, section 2.8: the on-resonance reflection of the critically coupled all-pass reference
    ring is -31.6 / -19.4 / -12.2 dB for r_bs = 0.003 / 0.0124 / 0.03 (exact and TCMT, within 0.1 dB)."""
    k2t, _, a, trt, _ = ring_params(3500, 0.5, 0.0)
    for rbs, ref in ((0.003, -31.6), (0.0124, -19.4), (0.03, -12.2)):
        for res in (optics.ring_exact(2 * math.pi * N0, k2t, 0.0, a, trt, rb=np.array([rbs])),
                    archived_tcmt.ring_tcmt(2 * math.pi * N0, k2t, 0.0, a, trt, rb=np.array([rbs]), m_orders=1)):
            assert abs(10 * math.log10(float(res.p_refl)) - ref) < 0.1


# -------------------------------------------------------------------------------------------- reciprocity and unitarity

def test_reciprocity_and_unitarity():
    rng = np.random.default_rng(3)
    k2t, k2d, a, trt, _ = ring_params(3500, 0.5, 0.05)
    rb = np.array([0.05 + 0.02j, -0.03j, 0.04])
    for ph in 2 * math.pi * N0 + rng.uniform(-math.pi, math.pi, 25):
        S = optics.ring_smatrix(ph, k2t, k2d, a, trt, rb=rb, psi_rad=0.2)
        assert np.max(np.abs(S - S.T)) < 1e-13
        assert abs(abs(S[0, 0]) - abs(S[1, 1])) < 1e-13            # used by the single-pass bound
        St = archived_tcmt.ring_scattering_matrix(ph, k2t, k2d, a, trt, rb=rb, m_orders=3)
        assert np.max(np.abs(St - St.T)) < 1e-13
        Su = optics.ring_smatrix(ph, k2t, k2d, 1.0, trt, rb=rb)            # lossless ring: S unitary
        assert np.max(np.abs(Su.conj().T @ Su - np.eye(4))) < 1e-13


@pytest.mark.parametrize("method", ["exact", "tcmt"])
def test_backward_input_equals_mirrored_ring(method):
    """Mirror symmetry (z -> -z swaps CW/CCW, ports 0/1 and 2/3, and turns each scatterer's rho into -conj(rho),
    i.e. r_b into conj(r_b)): light entering at the through port (port 1) sees the forward response of the same
    ring with conj(r_b). A backward (external-reflection) pass can therefore reuse the forward kernels. Backward
    light reaches the monitor (port 2) only through backscatter."""
    k2t, k2d, a, trt, _ = ring_params(3500, 0.5, 0.1)
    rb = np.array([0.05 + 0.03j, -0.02 + 0.04j, 0.06j])
    ph = 2 * math.pi * N0 + np.linspace(-3, 3, 41)
    if method == "exact":
        f = optics.ring_exact
    else:
        def f(*args, **kw):
            return archived_tcmt.ring_tcmt(*args, m_orders=3, **kw)
    b = f(ph, k2t, k2d, a, trt, rb=rb, psi_rad=0.2, s_in=0.0, s_thru_in=1.0)
    m = f(ph, k2t, k2d, a, trt, rb=np.conj(rb), psi_rad=0.2)
    for x, y in ((b.p_thru, m.p_refl), (b.p_refl, m.p_thru), (b.p_add_out, m.p_drop), (b.p_drop, m.p_add_out),
                 (b.u_ccw, m.u_cw), (b.u_cw, m.u_ccw), (b.p_loss, m.p_loss)):
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-15 * np.max(np.abs(x)))
    nb = f(ph, k2t, k2d, a, trt, psi_rad=0.2, s_in=0.0, s_thru_in=1.0)
    assert np.max(nb.p_drop) == 0.0


# -------------------------- doublet resolution (compact-model notes, section 2.8, as corrected in its verification log)

def _curvature_at_zero(fun, h):
    """Second difference fun(h) + fun(-h) - 2 fun(0): positive at a local minimum, negative at a local maximum."""
    return fun(h) + fun(-h) - 2 * fun(0.0)


def test_doublet_resolution_thresholds():
    """Drop port: a dip at Delta = 0 appears when 2|g|/kappa > sqrt(sqrt5 - 2) = 0.486. Critically coupled through
    port: the notch splits when 2|g|/kappa > 5^(-1/4) = 0.669. The compact-model notes (section 2.8, as corrected)
    round these to 0.49 and 0.67. TCMT matches exactly; the exact ring at finesse F ~ 230 matches to within 3 %."""
    drop_th = math.sqrt(math.sqrt(5) - 2)
    thru_th = 5 ** -0.25
    assert abs(drop_th - 0.486) < 1e-3 and abs(thru_th - 0.669) < 1e-3
    for method in ("tcmt", "exact"):
        q_l = 3500 if method == "tcmt" else 30000
        for port, th, ke_f, kd_f in (("drop", drop_th, 0.3, 0.3), ("thru", thru_th, 0.5, 0.0)):
            k2t, k2d, a, trt, kap = ring_params(q_l, ke_f, kd_f)
            for factor, dip_expected in ((0.97, False), (1.03, True)):
                g = factor * th * kap / 2
                rb = np.array([g * trt])
                h = 0.02 * kap * trt / 2

                def fun(dphi):
                    if method == "tcmt":
                        r = archived_tcmt.ring_tcmt(2 * math.pi * N0 + dphi, k2t, k2d, a, trt, rb=rb, m_orders=1)
                    else:
                        r = optics.ring_exact(2 * math.pi * N0 + dphi, k2t, k2d, a, trt, rb=rb)
                    return float(r.p_drop if port == "drop" else r.p_thru)
                curv = _curvature_at_zero(fun, h)
                # drop: a dip means a local MIN at 0 (curv > 0); through: a split notch means a local MAX (curv < 0)
                has_dip = curv > 0 if port == "drop" else curv < 0
                assert has_dip == dip_expected, (method, port, factor, curv)


# -------------------------------------------------------------------- monitor identity, derivatives, self-heating slope

def test_monitor_tracks_intracavity_energy():
    """The monitor identity: P_drop * kappa_i/kappa_d = P_loss, exact in TCMT; the exact ring obeys it to O(1/F^2)
    near resonance. It is why the drop-port monitor tracks the intracavity energy (the absorbed heat)."""
    k2t, k2d, a, trt, kap = ring_params(3500, 0.5, 0.02)
    ki, ke, kd = optics.exact_to_tcmt(k2t, k2d, a, trt)
    phi = 2 * math.pi * N0 + np.linspace(-1, 1, 41) * kap * trt
    t = archived_tcmt.ring_tcmt(phi, k2t, k2d, a, trt, m_orders=1)
    np.testing.assert_allclose(t.p_drop * ki / kd, t.p_loss, rtol=1e-12)
    e = optics.ring_exact(phi, k2t, k2d, a, trt)
    np.testing.assert_allclose(e.p_drop * ki / kd, e.p_loss, rtol=1e-2)


@pytest.mark.parametrize("method,rb", [("exact", None), ("exact", np.array([0.05 + 0.02j, -0.03j, 0.04])),
                                       ("tcmt", np.array([0.05 + 0.02j, -0.03j, 0.04]))])
def test_phase_derivatives_against_finite_differences(method, rb):
    k2t, k2d, a, trt, _ = ring_params(3500, 0.5, 0.05)
    f = (lambda p: optics.ring_exact(p, k2t, k2d, a, trt, rb=rb)) if method == "exact" else \
        (lambda p: archived_tcmt.ring_tcmt(p, k2t, k2d, a, trt, rb=rb, m_orders=3))
    phi = 2 * math.pi * N0 + np.linspace(-1.5, 1.5, 31)
    h = 1e-6
    r, rp, rm = f(phi), f(phi + h), f(phi - h)
    for an, num in ((r.dloss_dphi, (rp.p_loss - rm.p_loss) / (2 * h)),
                    (r.dthru_dphi, (rp.p_thru - rm.p_thru) / (2 * h)),
                    (r.ddrop_dphi, (rp.p_drop - rm.p_drop) / (2 * h))):
        assert np.max(np.abs(an - num)) < 1e-6 * np.max(np.abs(num))


@pytest.mark.parametrize("method,with_bs,nrz", [("exact", True, True), ("exact", False, True),
                                                ("exact", True, False)])
def test_dpabs_dlambda_res_against_finite_differences(method, with_bs, nrz):
    """The slope that the Newton step of the implicit self-heating solve uses: dP_abs/d lambda_res of every ring
    (input held fixed), from dpabs_dlambda_res and from the bus pass with slopes, against a central difference of
    P_abs under a rigid resonance shift."""
    rng = np.random.default_rng(4)
    n, nl = 3, 5
    k2t, k2d, a, trt, kap = ring_params(3500, 0.5, 0.02)
    lam = LAM0 + np.linspace(-6e-9, 6e-9, nl)
    phi = 2 * math.pi * N0 + rng.uniform(-1.0, 1.0, (n, nl))
    rb = (rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))) * 0.04 if with_bs else None
    nz = dict(p1=np.array([0.3, 0.5, 0.8]), phi1=phi + 0.1, a1=a * 0.997, eta1=0.9) if nrz else None
    bus = optics.bus_propagate(np.full(nl, 2e-3), phi, k2t, k2d, a, trt, 0.85, rb=rb, nrz=nz, method=method, lam_m=lam)
    pabs, slope = optics.dpabs_dlambda_res(bus.p_in, phi, k2t, k2d, a, trt, 0.85, lam, rb=rb, nrz=nz, method=method)
    np.testing.assert_allclose(pabs, bus.abs_sum, rtol=1e-12)
    np.testing.assert_allclose(slope, bus.dabs_dlam, rtol=1e-10)
    dpl = optics.dphi_dlambda_res(trt, lam)
    dl = 1e-14                                          # m (0.01 pm)

    def pabs_at(shift):
        nz2 = None if nz is None else dict(nz, phi1=nz["phi1"] + shift * dpl)
        return optics.dpabs_dlambda_res(bus.p_in, phi + shift * dpl, k2t, k2d, a, trt, 0.85, lam, rb=rb, nrz=nz2,
                                        method=method)[0]
    fd = (pabs_at(dl) - pabs_at(-dl)) / (2 * dl)
    np.testing.assert_allclose(slope, fd, rtol=1e-5, atol=1e-8 * np.max(np.abs(fd)))


def test_pabs_bounds_hold_with_backscatter():
    """pabs_bounds_nb and pabs_slope_bound_nb: P_abs <= p_max, |dP_abs/dphi| <= dp_max and |d2 P_abs/d phi2| <=
    d2p_max over a full FSR, for random rings (Q_L 800 to 20000, FSR 1.2 to 2.2 THz) with random per-order
    backscatter, (a) up to 2|g| = 0.8 kappa and (b) up to |r_b| = MAX_ROUND_TRIP_BACKSCATTER_FOR_VERIFIED_POWER_BOUNDS (the stated validity domain; 2|g|
    up to about 50 kappa at the highest Q). Without backscatter the slope bound is attained (to its 10 % margin) and
    the peak bound is attained or capped at the loss fraction 1. Outside the domain (|r_b| = 1.6) the curvature
    bound fails, as the pabs_bounds_nb docstring states."""
    rng = np.random.default_rng(11)
    worst_p = worst_c = worst_s = 0.0
    one = np.ones((1, 1))
    n_rings, n_phi = (8, 2001) if JIT_OFF else (60, 8001)
    for i in range(n_rings):
        q_l, fsr = rng.uniform(800, 20000), rng.uniform(1.2e12, 2.2e12)
        k2t, k2d, a, trt, kap = ring_params(q_l, rng.uniform(0.2, 0.7), rng.uniform(0, 0.25), fsr=fsr)
        if i % 2 == 0:
            rb = (rng.normal(size=4) + 1j * rng.normal(size=4)) * rng.uniform(0, 0.4) * kap * trt / 1.4
        else:
            rb = rng.uniform(0, optics.MAX_ROUND_TRIP_BACKSCATTER_FOR_VERIFIED_POWER_BOUNDS, 4) * np.exp(
                1j * rng.uniform(0, 2 * math.pi, 4)
            )
        phi = 2 * math.pi * N0 + np.linspace(-math.pi, math.pi, n_phi)
        r = optics.ring_exact(phi, k2t, k2d, a, trt, rb=rb)
        loss = r.p_loss
        h = phi[1] - phi[0]
        curv = np.abs(loss[2:] + loss[:-2] - 2 * loss[1:-1]) / h**2
        pm, cm = optics.pabs_bounds_nb(
            0, np.ones(1), 0.0, one * a, one * a, one, one, one * k2t, one * k2d, False
        )
        sm = optics.pabs_slope_bound_nb(
            0, np.ones(1), 0.0, one * a, one * a, one, one, one * k2t, one * k2d, False
        )
        worst_p = max(worst_p, loss.max() / pm)
        worst_c = max(worst_c, curv.max() / cm)
        worst_s = max(worst_s, np.abs(r.dloss_dphi).max() / sm)
    assert worst_p <= 1.0 and worst_c <= 1.0 and worst_s <= 1.0, (worst_p, worst_c, worst_s)
    # the same Airy-based bounds hold for the multi-order TCMT (worst 0.96 / 0.82 / 0.93 over M = 1..7)
    for i in range(4 if JIT_OFF else 16):
        q_l, fsr = rng.uniform(800, 20000), rng.uniform(1.2e12, 2.2e12)
        k2t, k2d, a, trt, kap = ring_params(q_l, rng.uniform(0.2, 0.7), rng.uniform(0, 0.25), fsr=fsr)
        rb = (rng.normal(size=4) + 1j * rng.normal(size=4)) * rng.uniform(0, 0.4) * kap * trt / 1.4
        phi = 2 * math.pi * N0 + np.linspace(-math.pi, math.pi, n_phi)
        r = archived_tcmt.ring_tcmt(phi, k2t, k2d, a, trt, rb=rb, m_orders=(3, 7)[i % 2])
        h = phi[1] - phi[0]
        curv = np.abs(r.p_loss[2:] + r.p_loss[:-2] - 2 * r.p_loss[1:-1]) / h**2
        pm, cm = optics.pabs_bounds_nb(
            0, np.ones(1), 0.0, one * a, one * a, one, one, one * k2t, one * k2d, False
        )
        sm = optics.pabs_slope_bound_nb(
            0, np.ones(1), 0.0, one * a, one * a, one, one, one * k2t, one * k2d, False
        )
        assert r.p_loss.max() <= pm and curv.max() <= cm and np.abs(r.dloss_dphi).max() <= sm
    phi = 2 * math.pi * N0 + np.linspace(-0.3, 0.3, 6001)
    for kd_f, capped in ((0.2, False), (0.02, True)):  # peak loss fraction 0.60 and 0.96 (0.96*1.05 > 1)
        k2t, k2d, a, trt, _ = ring_params(3500, 0.5, kd_f)
        r = optics.ring_exact(phi, k2t, k2d, a, trt)
        pm, _ = optics.pabs_bounds_nb(
            0, np.ones(1), 0.0, one * a, one * a, one, one, one * k2t, one * k2d, False
        )
        sm = optics.pabs_slope_bound_nb(
            0, np.ones(1), 0.0, one * a, one * a, one, one, one * k2t, one * k2d, False
        )
        assert math.isclose(pm, 1.0 if capped else r.p_loss.max() * 1.05, rel_tol=1e-6)
        assert math.isclose(sm, np.abs(r.dloss_dphi).max() * 1.10, rel_tol=1e-4)
    # NRZ two-level, several lines: the bounds add over lines and average over levels
    two = np.ones((1, 3))
    k2t, k2d, a, trt, _ = ring_params(3500, 0.5, 0.2)
    pin = np.array([1e-3, 2e-3, 0.5e-3])
    pm, cm = optics.pabs_bounds_nb(
        0, pin, 0.3, two * a, two * a * 0.99, two * 0.8, two * 0.9, two * k2t, two * k2d, True
    )
    pm1, _ = optics.pabs_bounds_nb(
        0, pin, 0.0, two * a, two * a, two * 0.8, two * 0.8, two * k2t, two * k2d, False
    )
    pm2, _ = optics.pabs_bounds_nb(
        0, pin, 0.0, two * a * 0.99, two * a * 0.99, two * 0.9, two * 0.9, two * k2t, two * k2d, False
    )
    assert math.isclose(pm, 0.7 * pm1 + 0.3 * pm2, rel_tol=1e-12)
    # outside the validity domain the curvature bound is NOT guaranteed (documents the stated limit)
    k2t, k2d, a, trt, _ = ring_params(3500, 0.5, 0.01)
    phi = 2 * math.pi * N0 + np.linspace(-math.pi, math.pi, 20001)
    loss = optics.ring_exact(phi, k2t, k2d, a, trt, rb=np.array([1.6])).p_loss
    h = phi[1] - phi[0]
    _, cm = optics.pabs_bounds_nb(0, np.ones(1), 0.0, one * a, one * a, one, one, one * k2t, one * k2d, False)
    assert np.max(np.abs(loss[2:] + loss[:-2] - 2 * loss[1:-1])) / h**2 / cm > 1.03  # 1.076


# ------------------------------------ the bus: cascade, coupling per line, NRZ average, single-pass bound, input checks

@pytest.mark.parametrize("rb", [None, np.array([0.05 + 0.02j, -0.03j, 0.04])])
def test_bus_single_ring_equals_ring_exact(rb):
    k2t, k2d, a, trt, _ = ring_params(3500, 0.5, 0.05)
    phi = 2 * math.pi * N0 + np.linspace(-2, 2, 9)
    bus = optics.bus_propagate(np.full(9, 1.5e-3), phi[None, :], k2t, k2d, a, trt, 0.8, rb=rb, psi=0.1)
    r = optics.ring_exact(phi, k2t, k2d, a, trt, rb=rb, psi_rad=0.1)
    for got, want in ((bus.thru[0], r.p_thru), (bus.drop[0], r.p_drop), (bus.refl[0], r.p_refl),
                      (bus.add_out[0], r.p_add_out), (bus.loss[0], r.p_loss), (bus.u_cw[0], r.u_cw),
                      (bus.u_ccw[0], r.u_ccw)):
        np.testing.assert_allclose(got, 1.5e-3 * want, rtol=1e-11, atol=1e-30)
    np.testing.assert_allclose(bus.absorbed[0], 0.8 * bus.loss[0], rtol=1e-14)


def test_bus_order_and_cascade():
    """Each ring sees what the previous ring (in bus order) let through, times the bus segment transmission."""
    rng = np.random.default_rng(5)
    n, nl = 4, 6
    k2t, k2d, a, trt, _ = ring_params(3500, 0.5, 0.02)
    phi = 2 * math.pi * N0 + rng.uniform(-0.5, 0.5, (n, nl))
    t2 = np.array([0.99, 0.98, 0.97, 0.96])
    p_in = rng.uniform(1e-3, 2e-3, nl)
    for order in (np.arange(n), np.array([2, 0, 3, 1])):
        b = optics.bus_propagate(p_in, phi, k2t, k2d, a, trt, 0.85, bus_order=order, bus_t2=t2, bus_t2_out=0.95)
        p = p_in.copy()
        for k in order:
            p = p * t2[k]
            np.testing.assert_allclose(b.p_in[k], p, rtol=1e-14)
            single = optics.ring_exact(phi[k], k2t, k2d, a, trt)
            np.testing.assert_allclose(b.thru[k], p * single.p_thru, rtol=1e-12, atol=1e-12 * p.max())
            p = b.thru[k]
        np.testing.assert_allclose(b.p_out, p * 0.95, rtol=1e-14)
    with pytest.raises(ValueError):
        optics.bus_propagate(p_in, phi, k2t, k2d, a, trt, 0.85, bus_order=[0, 0, 1, 2])


def test_coupling_is_per_line():
    """K^2 is given per (ring, LINE), evaluated at each line wavelength: changing it at one line changes only that
    line's response."""
    k2t, k2d, a, trt, _ = ring_params(3500, 0.5, 0.02)
    phi = np.full((1, 4), 2 * math.pi * N0) + np.array([[0.0, 0.05, -0.05, 1.0]])
    k2 = np.full((1, 4), k2t)
    b0 = optics.bus_propagate(np.ones(4), phi, k2, k2d, a, trt, 0.85)
    k2[0, 1] *= 1.05
    b1 = optics.bus_propagate(np.ones(4), phi, k2, k2d, a, trt, 0.85)
    changed = np.abs(b1.thru[0] - b0.thru[0]) > 0
    assert list(changed) == [False, True, False, False]


def test_nrz_average_factorizes_over_independent_rings():
    """The NRZ-averaged cascade equals the expectation over the joint data states of two rings (independent data),
    and pout1/pout0 equal the expectations conditioned on one ring's level."""
    rng = np.random.default_rng(6)
    n, nl = 2, 4
    k2t, k2d, a, trt, _ = ring_params(3500, 0.5, 0.05)
    phi0 = 2 * math.pi * N0 + rng.uniform(-0.5, 0.5, (n, nl))
    phi1 = phi0 + 0.12
    a1 = a * 0.99
    p1 = np.array([0.3, 0.65])
    rb = (rng.normal(size=(n, 2)) + 1j * rng.normal(size=(n, 2))) * 0.04
    p_in = np.array([1e-3, 2e-3, 1.5e-3, 0.5e-3])
    avg = optics.bus_propagate(p_in, phi0, k2t, k2d, a, trt, 0.85, bus_t2=[0.98, 0.97], bus_t2_out=0.9, rb=rb,
                               nrz=dict(p1=p1, phi1=phi1, a1=a1, eta1=0.9))
    exp_out = np.zeros(nl)
    exp_refl_in = np.zeros(nl)
    exp_drop = np.zeros((n, nl))
    exp_abs = np.zeros((n, nl))
    cond = {(k, b): np.zeros(nl) for k in range(n) for b in (0, 1)}
    for b0 in (0, 1):
        for b1 in (0, 1):
            levels = (b0, b1)
            w = np.prod([p1[k] if levels[k] else 1 - p1[k] for k in range(n)])
            ph = np.where(np.array(levels)[:, None] == 1, phi1, phi0)
            aa = np.where(np.array(levels)[:, None] == 1, a1, a)
            eta = np.where(np.array(levels)[:, None] == 1, 0.9, 0.85)
            cw = optics.bus_propagate(p_in, ph, k2t, k2d, aa, trt, eta, bus_t2=[0.98, 0.97], bus_t2_out=0.9, rb=rb)
            exp_out += w * cw.p_out
            exp_refl_in += w * cw.refl_in
            exp_drop += w * cw.drop
            exp_abs += w * cw.absorbed
            for k in range(n):
                wk = w / (p1[k] if levels[k] else 1 - p1[k])
                cond[(k, levels[k])] += wk * cw.p_out
    np.testing.assert_allclose(avg.p_out, exp_out, rtol=1e-12)
    # the reflection of ring 1 crosses ring 0 twice in the same data state: E[T^2], not E[T]^2
    np.testing.assert_allclose(avg.refl_in, exp_refl_in, rtol=1e-12)
    np.testing.assert_allclose(avg.drop, exp_drop, rtol=1e-12)
    np.testing.assert_allclose(avg.absorbed, exp_abs, rtol=1e-12)
    for k in range(n):
        np.testing.assert_allclose(avg.pout1[k], cond[(k, 1)], rtol=1e-12)
        np.testing.assert_allclose(avg.pout0[k], cond[(k, 0)], rtol=1e-12)
    np.testing.assert_allclose(avg.drop_sum1, avg.drop1.sum(axis=1), rtol=1e-14)
    np.testing.assert_allclose(p1 * avg.drop_sum1 + (1 - p1) * avg.drop_sum0, avg.drop_sum, rtol=1e-12)


def test_single_pass_bound_against_coherent_two_ring_cavity():
    """Two rings on the same line with strong backscatter: the exact coherent cascade (Fabry-Perot between them,
    any bus phase) never departs from the single-pass forward power by more than the reported bound, and the bound
    is attained for some phase."""
    k2t, k2d, a, trt, _ = ring_params(3500, 0.5, 0.02)
    rb = np.array([0.12 + 0.05j])                       # beta ~ 34 GHz: an extreme backscatter
    phis = np.array([[2 * math.pi * N0 + 0.01], [2 * math.pi * N0 - 0.02]])
    bus = optics.bus_propagate(np.ones(1), phis, k2t, k2d, a, trt, 0.85, rb=rb)
    S1 = optics.ring_smatrix(phis[0, 0], k2t, k2d, a, trt, rb=rb)
    S2 = optics.ring_smatrix(phis[1, 0], k2t, k2d, a, trt, rb=rb)
    theta = np.linspace(0, 2 * math.pi, 4001)
    ratio = 1.0 / np.abs(1 - S1[1, 1] * S2[0, 0] * np.exp(-2j * theta)) ** 2     # coherent / single-pass power
    worst = np.max(np.abs(ratio - 1))
    assert bus.bound[0] > 1e-3                          # the case is not trivial
    assert worst <= bus.bound[0] * (1 + 1e-9)
    assert worst >= 0.99 * bus.bound[0]
    np.testing.assert_allclose(np.sqrt(bus.refl[:, 0] / bus.p_in[:, 0]), [abs(S1[0, 0]), abs(S2[0, 0])],
                               rtol=1e-12)


def _coherent_chain(s_list, thetas):
    """Independent coherent cascade of 2-ports (ports 0 left, 1 right; S from ring_smatrix), unit input at the left,
    bus phase thetas[k] on the segment before element k. Returns |forward field|^2 at each element input and at the
    output."""
    n = len(s_list)
    A = np.zeros((2 * n + 1, 2 * n + 1), complex)
    y = np.zeros(2 * n + 1, complex)
    A[0, 0] = 1.0
    y[0] = 1.0
    row = 1
    for k, S in enumerate(s_list):
        ph = np.exp(-1j * thetas[k + 1]) if k + 1 < n else 1.0
        A[row, k + 1] = 1.0                                   # a_{k+1} = ph (S10 a_k + S11 ph b_{k+1})
        A[row, k] -= ph * S[1, 0]
        if k + 1 < n:
            A[row, n + 2 + k] -= ph * S[1, 1] * ph
        row += 1
        A[row, n + 1 + k] = 1.0                               # b_k = S00 a_k + S01 ph b_{k+1}
        A[row, k] -= S[0, 0]
        if k + 1 < n:
            A[row, n + 2 + k] -= S[0, 1] * ph
        row += 1
    return np.abs(np.linalg.solve(A, y)[:n + 1]) ** 2


def test_single_pass_bound_three_rings_all_pairs():
    """Three rings near one line with strong backscatter (r_b = 0.15; detuned +4, +1.6, -4 half-widths): over all
    bus phases the coherent cascade departs from the single-pass forward powers by 14.8 %. The all-pairs bound
    (16.5 %) holds; a maximum over single cuts (8.5 %, the earlier formula) would not."""
    k2t, k2d, a, trt, kap = ring_params(3500, 0.5, 0.02)
    rb = np.array([0.15])
    phis = 2 * math.pi * N0 + np.array([4.0, 1.6, -4.0]) * kap * trt / 2
    bus = optics.bus_propagate(np.ones(1), phis[:, None], k2t, k2d, a, trt, 0.85, rb=rb)
    S = [optics.ring_smatrix(p, k2t, k2d, a, trt, rb=rb) for p in phis]
    sp = np.concatenate([[1.0], np.cumprod([abs(s[1, 0]) ** 2 for s in S])])
    th = np.linspace(0, 2 * math.pi, 37)[:-1]
    worst = max(np.max(np.abs(_coherent_chain(S, [0.0, t1, t2]) / sp - 1)) for t1 in th for t2 in th)
    r = np.array([abs(s[0, 0]) for s in S])
    cut = max(r[0] * (r[1] + r[2]), (r[0] + r[1]) * r[2])
    assert worst > 0.14
    assert worst <= bus.bound[0]
    assert worst > 1 / (1 - cut) ** 2 - 1                    # the single-cut formula is not a bound
    pairs = r[0] * r[1] + r[0] * r[2] + r[1] * r[2]
    assert math.isclose(bus.bound[0], 1 / (1 - pairs) ** 2 - 1, rel_tol=1e-12)


def test_refl_returned_to_input_uses_reciprocity():
    """Reflection of ring k reaches the bus input attenuated by the same transmissions as the forward path."""
    k2t, k2d, a, trt, _ = ring_params(3500, 0.5, 0.02)
    phi = np.array([[2 * math.pi * N0 + 0.3], [2 * math.pi * N0]])
    rb = np.array([0.05])
    b = optics.bus_propagate(np.array([1e-3]), phi, k2t, k2d, a, trt, 0.85, rb=rb, bus_t2=[0.9, 0.8])
    S1 = optics.ring_smatrix(phi[0, 0], k2t, k2d, a, trt, rb=rb)
    S2 = optics.ring_smatrix(phi[1, 0], k2t, k2d, a, trt, rb=rb)
    want = 1e-3 * (0.9 * abs(S1[0, 0]) ** 2 * 0.9 + 0.9 * abs(S1[1, 0]) ** 2 * 0.8 * abs(S2[0, 0]) ** 2 * 0.8
                   * abs(S1[0, 1]) ** 2 * 0.9)
    assert math.isclose(float(b.refl_in[0]), want, rel_tol=1e-12)


def test_input_validation():
    k2t, k2d, a, trt, _ = ring_params()
    with pytest.raises(ValueError):
        optics.ring_exact(0.0, 1.0, k2d, a, trt)
    with pytest.raises(ValueError):
        optics.ring_exact(0.0, k2t, k2d, 1.2, trt)
    with pytest.raises(ValueError):
        optics.ring_exact(0.0, k2t, k2d, a, trt, rb=np.array([2.5]))
    with pytest.raises(ValueError):
        optics.ring_exact(float("nan"), k2t, k2d, a, trt)
    with pytest.raises(ValueError):
        optics.ring_smatrix(0.0, k2t, k2d, a, trt, method="airy")
    with pytest.raises(ValueError):
        optics.bus_propagate(np.ones(2), np.zeros((1, 3)), k2t, k2d, a, trt, 0.8)
    ph = np.zeros((2, 3))
    for kw in (dict(eta=1.2), dict(eta=0.8, bus_t2=[1.0, 1.1]), dict(eta=0.8, bus_t2_out=-0.1),
               dict(eta=0.8, nrz=dict(p1=[0.5, 1.5])), dict(eta=0.8, nrz=dict(p1=[0.5, 0.5], a1=1.5))):
        eta = kw.pop("eta")
        with pytest.raises(ValueError):
            optics.bus_propagate(np.ones(3), ph, k2t, k2d, a, trt, eta, **kw)
    with pytest.raises(ValueError):
        optics.bus_propagate(-np.ones(3), ph, k2t, k2d, a, trt, 0.8)
    with pytest.raises(ValueError):
        optics.dpabs_dlambda_res(np.ones(3), ph, k2t, k2d, a, trt, 0.8, LAM0)


def test_dark_lines_and_open_segments_give_zeros_not_nans():
    """A disabled line (P_in = 0) and a bus segment with zero transmission propagate zeros, never NaN."""
    k2t, k2d, a, trt, _ = ring_params(3500, 0.5, 0.02)
    phi = 2 * math.pi * N0 + np.array([[0.0, 0.1, -0.2], [0.05, 0.0, 0.3], [0.1, -0.1, 0.0]])
    rb = np.array([0.03 + 0.01j, 0.02j])
    b = optics.bus_propagate(np.array([1e-3, 0.0, 2e-3]), phi, k2t, k2d, a, trt, 0.85, bus_t2=[1.0, 0.0, 0.9], rb=rb,
                             nrz=dict(p1=[0.5, 0.5, 0.5], phi1=phi + 0.1), lam_m=LAM0)
    for f in ("p_in", "thru", "drop", "refl", "absorbed", "pout1", "pout0", "dabs_dphi"):
        assert np.all(np.isfinite(getattr(b, f))), f
    assert np.all(b.p_in[:, 1] == 0) and np.all(b.thru[1:] == 0) and np.all(b.p_out == 0)
    assert np.all(np.isfinite(b.bound)) and np.all(np.isfinite(b.refl_in))
    assert b.refl_in[1] == 0 and b.refl_in[0] > 0


@pytest.mark.parametrize("method,with_bs", [("exact", True), ("exact", False)])
def test_kernel_walk_reproduces_bus_propagate(method, with_bs):
    """The integration recipe of the compiled engine: walk bus_order with dpabs_dlambda_res_nb (the Newton
    evaluation) and evaluate_ring_optical_outputs (the commit; its through output times bus_t2 of the next ring is that ring's
    input), then bus_finalize_nb. The compiled kernel API, preallocated workspaces, reused across calls, must give
    exactly what bus_propagate_nb gives."""
    rng = np.random.default_rng(12)
    n, nl, mb = 4, 5, 3
    k2t, k2d, a, trt, _ = ring_params(3500, 0.5, 0.02)
    shape = (n, nl)
    phi0 = 2 * math.pi * N0 + rng.uniform(-1, 1, shape)
    rb = (
        (rng.normal(size=(n, mb)) + 1j * rng.normal(size=(n, mb))) * 0.03
        if with_bs
        else np.zeros((n, mb), complex)
    )
    order = np.array([2, 0, 3, 1], dtype=np.int64)
    t2 = np.array([0.99, 0.97, 0.98, 0.96])
    p_in = rng.uniform(0.5e-3, 2e-3, nl)
    p1 = np.array([0.4, 0.5, 0.6, 0.3])
    dpl = np.full(shape, optics.dphi_dlambda_res(trt, LAM0))
    ring_arrays = (
        phi0,
        phi0 + 0.07,
        np.full(shape, a),
        np.full(shape, a * 0.998),
        np.full(shape, 0.85),
        np.full(shape, 0.88),
        np.full(shape, 0.05),
        np.full(shape, k2t),
        np.full(shape, k2d),
        np.full(shape, trt),
        rb,
        dpl,
    )
    tc = method == "tcmt"
    ref_rl, ref_r, ref_l = optics.make_bus_outputs(n, nl)
    cw, fw, work = optics.make_workspace(nl)
    optics.bus_propagate_nb(
        p_in, order, t2, 0.95, p1, *ring_arrays, True, tc, 3, True, ref_rl, ref_r, ref_l, work, cw, fw
    )
    out_rl, out_r, out_l = optics.make_bus_outputs(n, nl)
    out_rl[:] = np.nan  # every field must be written
    out_r[:] = np.nan
    out_l[:] = np.nan
    pin = p_in.copy()
    for k in order:
        pin *= t2[k]
        pabs, slope = optics.dpabs_dlambda_res_nb(k, pin, p1[k], *ring_arrays, True, tc, 3, cw, fw)
        optics.evaluate_ring_optical_outputs(k, pin, p1[k], *ring_arrays, True, tc, 3, True, out_rl, out_r, pin, cw, fw)
        assert math.isclose(pabs, out_r[optics.RING_OUTPUT_ROW_TOTAL_ABSORBED_POWER, k], rel_tol=1e-13)
        assert math.isclose(
            slope,
            out_r[optics.RING_OUTPUT_ROW_ABSORBED_POWER_RESONANCE_WAVELENGTH_DERIVATIVE, k],
            rel_tol=1e-11,
            abs_tol=1e-30,
        )
    optics.bus_finalize_nb(p_in, order, t2, 0.95, out_rl, out_l)
    np.testing.assert_array_equal(out_rl, ref_rl)
    np.testing.assert_array_equal(out_r, ref_r)
    np.testing.assert_array_equal(out_l, ref_l)


# -------------------------------------------------------------------------- self-heating adapter for mrm.plant.selfheat

def _selfheat_case(method="exact", nrz=True, power=2e-3):
    """A 3-ring, 4-line bus packed for the self-heating adapter (pack_selfheat_args), power W per line.

    Returns (packed_f, packed_i, (p_in, phi_ref, k2t, k2d, a, trt, lam, bus keywords, thermo-optic law)). The law is
    lambda_res(T) - lambda_res(T_ref) = s0 [dT + beta dT^2/2] with T_ref (K), s0 (m/K) and beta (1/K) per ring.
    """
    rng = np.random.default_rng(21)
    n, nl = 3, 4
    k2t, k2d, a, trt, kap = ring_params(3500, 0.5, 0.02)
    phi = 2 * math.pi * N0 + rng.uniform(-1.5, 0.3, (n, nl))         # mostly blue of resonance (heating red-shifts)
    rb = (rng.normal(size=(n, 5)) + 1j * rng.normal(size=(n, 5))) * 0.03
    lam = LAM0 + np.arange(nl) * 2.3e-9
    kw = dict(bus_order=[2, 0, 1], bus_t2=[0.99, 0.98, 0.97], bus_t2_out=0.95, rb=rb, method=method,
              nrz=dict(p1=[0.5, 0.4, 0.6], phi1=phi + 0.05, a1=a * 0.998, eta1=0.88) if nrz else None)
    law = dict(t_ref=np.array([300.0, 301.0, 299.5]), s0=np.array([80e-12, 75e-12, 85e-12]),
               beta=np.array([1e-3, 2e-3, 0.0]))
    p_in = np.full(nl, power)
    packed_f, packed_i = optics.pack_selfheat_args(p_in, phi, k2t, k2d, a, trt, 0.85, lam, law["t_ref"], law["s0"],
                                                   law["beta"], commit_slope=True, **kw)
    return packed_f, packed_i, (p_in, phi, k2t, k2d, a, trt, lam, kw, law)


def _phases_at(t, phi_ref, trt, lam, law):
    """Round-trip phases (rad) at ring temperatures t (K): phi_ref moved by the thermo-optic shift of each ring."""
    dt = t - law["t_ref"]
    shift = law["s0"] * (dt + 0.5 * law["beta"] * dt * dt)
    return phi_ref + optics.dphi_dlambda_res(trt, lam)[None, :] * shift[:, None]


@pytest.mark.parametrize("method,nrz", [("exact", True), ("exact", False)])
def test_selfheat_adapter_solves_the_implicit_bank(method, nrz):
    """selfheat_pabs / selfheat_commit / selfheat_bounds driven by mrm.plant.selfheat.solve_ring_temperature in bus
    order (as implicit_step_b does): every ring converges, its root satisfies T_k = T_other,k + g_k P_abs,k(T_k)
    with P_abs from an independent bus_propagate at the solved phases, and the packed outputs equal that
    bus_propagate. The first sweep is a cold start several kelvin from the roots (up to 200 evaluations); the
    second sweep (as Gauss-Seidel or the next substep) restarts the walk by itself and, warm-started, converges
    within the 10 evaluations that the design review allows per ring (the test asserts at most 3)."""
    from mrm.plant.selfheat import solve_ring_temperature

    packed_f, packed_i, (p_in, phi, k2t, k2d, a, trt, lam, kw, law) = _selfheat_case(method, nrz)
    n = phi.shape[0]
    p_max, d2p_max, dpdt_max = np.zeros(n), np.zeros(n), np.zeros(n)
    optics.selfheat_bounds(packed_f, packed_i, 200.0, p_max, d2p_max, dpdt_max)
    t_other = law["t_ref"] + np.array([0.5, -0.3, 1.0])
    g = np.array([5.0e3, 4.0e3, 6.0e3])  # K/W (about 5 K/mW, the design review's value)
    t_ring = t_other.copy()
    for sweep in range(2):
        for k in kw["bus_order"]:
            t, p, d, it, st = solve_ring_temperature(
                optics.selfheat_pabs,
                k,
                packed_f,
                packed_i,
                t_other[k],
                g[k],
                t_ring[k],
                p_max[k],
                d2p_max[k],
                200 if sweep == 0 else 10,
                1e-9,
            )
            assert st == 0, (sweep, k, st, it)
            assert sweep == 0 or it <= 3
            t_ring[k] = t
            optics.selfheat_commit(t, k, packed_f, packed_i)
    nz = kw["nrz"]
    kw2 = dict(kw)
    if nz is not None:
        kw2["nrz"] = dict(nz, phi1=_phases_at(t_ring, nz["phi1"], trt, lam, law))
    b = optics.bus_propagate(
        p_in, _phases_at(t_ring, phi, trt, lam, law), k2t, k2d, a, trt, 0.85, lam_m=lam, **kw2
    )
    np.testing.assert_allclose(t_ring, t_other + g * b.abs_sum, rtol=0, atol=1e-8)
    v = optics.selfheat_views(packed_f, packed_i)
    np.testing.assert_allclose(
        v[optics.SELF_HEATING_VIEW_INDEX_RING_LINE_OUTPUTS][optics.RING_LINE_OUTPUT_ROW_ABSORBED_POWER],
        b.absorbed,
        rtol=1e-11,
        atol=1e-20,
    )
    np.testing.assert_allclose(
        v[optics.SELF_HEATING_VIEW_INDEX_RING_OUTPUTS][optics.RING_OUTPUT_ROW_TOTAL_CLOCKWISE_DROP_POWER],
        b.drop_sum,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        v[optics.SELF_HEATING_VIEW_INDEX_RING_OUTPUTS][
            optics.RING_OUTPUT_ROW_ABSORBED_POWER_RESONANCE_WAVELENGTH_DERIVATIVE
        ],
        b.dabs_dlam,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        v[optics.SELF_HEATING_VIEW_INDEX_LASER_LINE_OUTPUTS][optics.LASER_LINE_OUTPUT_ROW_BUS_OUTPUT_POWER],
        b.p_out,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        v[optics.SELF_HEATING_VIEW_INDEX_LASER_LINE_OUTPUTS][
            optics.LASER_LINE_OUTPUT_ROW_BUS_INPUT_RETURNED_REFLECTION_POWER
        ],
        b.refl_in,
        rtol=1e-11,
    )
    assert np.all(b.abs_sum <= p_max)


def test_selfheat_adapter_derivative_and_temperature_bounds():
    """dP_abs/dT of selfheat_pabs against a central difference in T, and the temperature-unit bounds of
    selfheat_bounds (P, |dP/dT|, |d2P/dT2|) over a 40 K scan of the first ring in bus order (input fixed)."""
    packed_f, packed_i, (p_in, phi, k2t, k2d, a, trt, lam, kw, law) = _selfheat_case("exact", True, power=3e-3)
    n = phi.shape[0]
    p_max, d2p_max, dpdt_max = np.zeros(n), np.zeros(n), np.zeros(n)
    optics.selfheat_bounds(packed_f, packed_i, 200.0, p_max, d2p_max, dpdt_max)
    k = kw["bus_order"][0]
    ts = law["t_ref"][k] + np.linspace(-5.0, 35.0, 4001)
    pd = np.array([optics.selfheat_pabs(t, k, packed_f, packed_i) for t in ts])
    h = ts[1] - ts[0]
    fd = (pd[2:, 0] - pd[:-2, 0]) / (2 * h)
    np.testing.assert_allclose(pd[1:-1, 1], fd, rtol=0, atol=2e-4 * np.max(np.abs(fd)))
    curv = np.abs(pd[2:, 0] + pd[:-2, 0] - 2 * pd[1:-1, 0]) / h ** 2
    assert pd[:, 0].max() <= p_max[k]
    assert np.abs(pd[:, 1]).max() <= dpdt_max[k]
    assert curv.max() <= d2p_max[k]
    assert pd[:, 0].max() > 0.3 * p_max[k]                           # the scan crosses resonances (not vacuous)


# ------------------------------------------------------------------------- numba: JIT-off equivalence and the benchmark

_NO_JIT_SCRIPT = r"""
import numpy as np, json
from mrm.plant import optics
rng = np.random.default_rng(7)
n, nl = 2, 3
phi = 2*np.pi*80 + rng.uniform(-1, 1, (n, nl))
rb = (rng.normal(size=(n, 3)) + 1j*rng.normal(size=(n, 3)))*0.04
rb[1] = 0
lam = 1310e-9 + np.array([-1e-9, 0, 1e-9])
out = {}
for m, mo in (("exact", 3),):
    b = optics.bus_propagate(np.array([1e-3, 2e-3, 1e-3]), phi, 0.1, 0.002, 0.95, 1/1.8e12, 0.85, rb=rb,
                             method=m, m_orders=mo, nrz=dict(p1=[0.4, 0.6], phi1=phi + 0.1, a1=0.94, eta1=0.9),
                             lam_m=lam, bus_t2=[0.99, 0.98])
    out[m + str(mo)] = np.concatenate([b.thru.ravel(), b.drop.ravel(), b.refl.ravel(), b.absorbed.ravel(),
                                       b.u_cw.ravel(), b.dabs_dlam, b.pout1.ravel(), b.bound, b.refl_in]).tolist()
r = optics.ring_exact(phi[0], 0.1, 0.002, 0.95, 1/1.8e12, rb=rb[0])
out["ring"] = np.concatenate([r.p_thru, r.p_refl, r.dloss_dphi]).tolist()
one = np.ones((2, 3))
pm, cm = optics.pabs_bounds_nb(0, np.array([1e-3, 2e-3, 1e-3]), 0.4, one * 0.95, one * 0.94, one * 0.85,
                               one * 0.9, one * 0.1, one * 0.002, True)
sm = optics.pabs_slope_bound_nb(0, np.array([1e-3, 2e-3, 1e-3]), 0.4, one * 0.95, one * 0.94, one * 0.85,
                                one * 0.9, one * 0.1, one * 0.002, True)
out["bounds"] = [pm, cm, sm]
print(json.dumps(out))
"""


@pytest.mark.skipif(JIT_OFF, reason="this test compares the JIT run against a NUMBA_DISABLE_JIT=1 subprocess")
def test_runs_identically_with_jit_disabled():
    import json
    env = dict(os.environ, NUMBA_DISABLE_JIT="1")
    res = subprocess.run([sys.executable, "-c", _NO_JIT_SCRIPT], capture_output=True, text=True, env=env, timeout=120)
    assert res.returncode == 0, res.stderr[-2000:]
    nojit = json.loads(res.stdout.strip().splitlines()[-1])
    res2 = subprocess.run([sys.executable, "-c", _NO_JIT_SCRIPT], capture_output=True, text=True, timeout=120)
    assert res2.returncode == 0, res2.stderr[-2000:]
    jit = json.loads(res2.stdout.strip().splitlines()[-1])
    for key in jit:
        np.testing.assert_allclose(np.array(jit[key]), np.array(nojit[key]), rtol=1e-11, atol=1e-25)


# A stand-in for _NO_JIT_SCRIPT: the NUMBA_DISABLE_JIT=1 run prints an empty result, the JIT run exits with an error
_JIT_RUN_CRASHES_SCRIPT = r"""
import os, sys
if os.environ.get("NUMBA_DISABLE_JIT") == "1":
    print("{}")
else:
    sys.exit("the JIT run crashed")
"""


@pytest.mark.skipif(JIT_OFF, reason="the stand-in script crashes only in the run without NUMBA_DISABLE_JIT=1")
def test_jit_equivalence_shows_the_error_of_a_crashed_jit_run(monkeypatch):
    """A crash of the JIT subprocess fails the equivalence test with that subprocess's error text (stderr), as a crash
    of the no-JIT subprocess does. Without the exit-code check, the test failed on the empty output (IndexError)."""
    monkeypatch.setattr(sys.modules[__name__], "_NO_JIT_SCRIPT", _JIT_RUN_CRASHES_SCRIPT)
    with pytest.raises(AssertionError, match="the JIT run crashed"):
        test_runs_identically_with_jit_disabled()


@pytest.mark.skipif(JIT_OFF, reason="timing is meaningless without the JIT")
def test_benchmark_bus_propagate_n8_l8_m3(capsys):
    """Benchmark one compiled bus_propagate_nb call, N = 8 rings, L = 8 lines, M_b = 3 backscatter orders.
    Reported, with a loose ceiling (the machine is shared); see the mrm.plant.optics module docstring, "Cost", for
    the measured numbers."""
    n, nl, mb = 8, 8, 3
    rng = np.random.default_rng(0)
    k2t, k2d, a, trt, _ = ring_params(3500, 0.5, 0.01)
    shape = (n, nl)
    phi0 = 2 * math.pi * N0 + rng.uniform(-math.pi, math.pi, shape)
    rb = (rng.normal(size=(n, mb)) + 1j * rng.normal(size=(n, mb))) * 0.02
    out_rl, out_r, out_l = optics.make_bus_outputs(n, nl)
    cw, fw, work = optics.make_workspace(nl)
    dpl = np.full(shape, optics.dphi_dlambda_res(trt, LAM0))
    args = (np.full(nl, 1e-3), np.arange(n, dtype=np.int64), np.full(n, 0.99), 0.98, np.full(n, 0.5), phi0,
            phi0 + 0.05, np.full(shape, a), np.full(shape, a * 0.999), np.full(shape, 0.85), np.full(shape, 0.86),
            np.zeros(shape), np.full(shape, k2t), np.full(shape, k2d), np.full(shape, trt), rb, dpl)

    @njit
    def loop(nrep, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, two, tc, ws,
             orl, orr, ol, w, c, f):
        acc = 0.0
        for _ in range(nrep):
            optics.bus_propagate_nb(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, two, tc,
                                    3, ws, orl, orr, ol, w, c, f)
            acc += orr[0, 0]
        return acc

    timings = {}
    for label, two, tc, ws in (("exact", False, False, False), ("exact, NRZ two-level", True, False, False),
                               ("exact, NRZ, with slope", True, False, True)):
        loop(1, *args, two, tc, ws, out_rl, out_r, out_l, work, cw, fw)
        best = 1e9
        for _ in range(3):
            t = time.perf_counter()
            loop(2000, *args, two, tc, ws, out_rl, out_r, out_l, work, cw, fw)
            best = min(best, (time.perf_counter() - t) / 2000)
        timings[label] = best
    with capsys.disabled():
        print("\n[bench] bus_propagate_nb N=8 L=8 M_b=3: " +
              ", ".join(f"{k} {v * 1e6:.1f} us" for k, v in timings.items()))
    assert timings["exact"] < 200e-6


# reject archived model requests instead of silently substituting a calculation
@pytest.mark.parametrize("entry_point", ["matrix", "bus", "slope", "selfheat", "compiled"])
def test_archived_tcmt_cannot_be_selected(entry_point):
    k2t, k2d, a, trt, _ = ring_params(3500, 0.5, 0.05)
    phi = np.full((1, 1), 2 * math.pi * N0 + 0.1)
    with pytest.raises(ValueError, match="TCMT is archived"):
        if entry_point == "matrix":
            optics.ring_scattering_matrix(float(phi[0, 0]), k2t, k2d, a, trt, method="tcmt")
        elif entry_point == "bus":
            optics.bus_propagate([1e-3], phi, k2t, k2d, a, trt, 0.85, method="tcmt")
        elif entry_point == "slope":
            optics.absorbed_power_and_resonance_slope([[1e-3]], phi, k2t, k2d, a, trt, 0.85, LAM0,
                                                      method="tcmt")
        elif entry_point == "selfheat":
            _selfheat_case(method="tcmt")
        else:
            cw, fw, _ = optics.make_workspace(1)
            res = np.empty((optics.RING_RESPONSE_ROW_COUNT, 1))
            optics.ring_rows_nb(phi[0], np.zeros(1), np.array([k2t]), np.array([k2d]), np.array([a]),
                               np.array([trt]), np.zeros(1, complex), True, 3, False, cw, fw, res)
