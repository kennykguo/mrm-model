"""Checks the thermal model mrm/plant/thermal.py against analytic limits (code checks, not hardware checks).

Read this first
---------------
The thermal model is a set of first-order lags (a Foster sum): every source (heater, absorbed light, junction,
aggressor, driver) reaches every observer (ring, heater resistor, monitor PD, temperature sensor) through residues
r_k (K/W) and time constants tau_k (s). The tests check: the exact zero-order-hold and first-order-hold stepping
against closed forms and scipy.linalg.expm, the step and ramp responses, reciprocity of the residues, the numbers of
the interim model, the crosstalk time-constant law, the scaling rules of the uncertainty set, batching, and the
integrity of the ROM artefacts (docs/ARCHITECTURE.md, "5.3 Thermal ROM").

"The design review" below means docs/research/07_design_critique.md.
"""
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg

from mrm.plant import thermal

# K/W, textbook R_th (derived; thermal notes, docs/research/04_thermal_and_control.md, section 2.1); any positive
# number works for these code checks
H86_Z0 = 4.7e3


def _thermal_bank(n=4, pitch=30.0, **kw):
    """An interim thermal system of n rings at pitch um, with one aggressor, two temperature sensors (one on ring 0,
    one free) and a driver site per ring. Keywords go to InterimRom."""
    lay = thermal.LayoutSites.line(n, pitch, aggressor_xy_um=[[pitch * (n - 1) / 2.0, 25.0]], aggressor_names=("agg",),
                                   temp_sensor_xy_um=[[0.0, 0.0], [pitch * n, -40.0]], temp_sensor_ring=[0, -1],
                                   temp_sensor_names=("ts_ring0", "ts_free"),
                                   driver_xy_um=np.c_[np.arange(n) * pitch, np.full(n, -30.0)])
    return thermal.assemble(thermal.InterimRom(**kw), lay, n)


# --------------------------------------------------------------------------------------------------------- coefficients

def _foh_reference(y):
    """b0, b1 from 25-term series (y <= 0.5) or the closed form (y > 0.5)."""
    if y <= 0.5:
        b0 = sum((-1) ** (n + 1) * n * y**n / math.factorial(n + 1) for n in range(1, 26))
        b1 = sum((-1) ** (n + 1) * y**n / math.factorial(n + 1) for n in range(1, 26))
        return b0, b1
    a = math.exp(-y)
    q = (1 - a) / y
    return q - a, 1 - q


@pytest.mark.parametrize("y", [0.0, 1e-9, 1e-6, 3e-4, 9.99e-4, 1.001e-3, 0.01, 0.3, 1.0, 2.0, 37.0, 1e4])
def test_foh_coefficients_against_reference(y):
    tau = 1e-6
    h = y * tau
    a, b0, b1 = thermal.foh_scalar(tau, h)
    a2, c0, c1 = thermal.foh_coefficients(np.array([tau]), np.array([1.0]), h)
    if y == 0.0:
        assert (a, b0, b1) == (1.0, 0.0, 0.0)
        return
    r0, r1 = _foh_reference(y)
    for got, ref in ((b0, r0), (b1, r1), (c0[0], r0), (c1[0], r1)):
        assert got == pytest.approx(ref, rel=1e-12, abs=1e-300)
    assert a == pytest.approx(math.exp(-y), rel=1e-15) and a2[0] == pytest.approx(a, rel=1e-15)
    assert b0 + b1 == pytest.approx(-math.expm1(-y), rel=1e-12)


# ------------------------------------------------------------------------------------------------------------ exact ZOH

@pytest.mark.parametrize("h", [0.37e-6, 1e-6, 13e-6, 0.25e-3, 2e-3])
def test_zoh_step_equals_foster_sum_for_any_dt(h):
    """ZOH step response at t = k h equals sum_k r_k (1 - exp(-t/tau_k)) for any h."""
    sys = _thermal_bank()
    n = 60
    u = np.zeros((n + 1, 1, sys.n_src))
    src = [sys.source_index("heater", 1), sys.aggressor_src[0], sys.source_index("waveguide_absorption", 2)]
    u[:, 0, src] = [10e-3, 50e-3, 1e-3]
    out = thermal.simulate(sys, u, h, hold="zoh")
    t = np.arange(n + 1) * h
    ref = np.einsum("bost,s->tbo", thermal.step_response(sys, t), u[0, 0])
    scale = np.max(np.abs(ref))
    assert np.max(np.abs(out - ref)) <= 1e-12 * scale


def test_zoh_against_scipy_expm():
    """Per-lag ZOH coefficients equal the exact discretization exp([[A, B], [0, 0]] h)."""
    sys = _thermal_bank(n=3)
    obs = [sys.observer_index("ring_mode_weighted", 0), sys.observer_index("heater_resistor", 1),
           sys.observer_index("temp_sensor", name="ts_free")]
    src = [sys.source_index("heater", 0), sys.source_index("waveguide_absorption", 0), sys.aggressor_src[0]]
    A, Bm, C, D = thermal.state_space(sys, 0, obs, src)
    _, _, m, lio, lis = thermal._select(sys, obs, src)
    for h in (1e-7, 1e-6, 3.3e-5, 1e-3):
        n, q = A.shape[0], Bm.shape[1]
        E = scipy.linalg.expm(np.block([[A, Bm], [np.zeros((q, n + q))]]) * h)
        Ad, Bd = E[:n, :n], E[:n, n:]
        a, c0, c1 = sys.coefficients(h)
        np.testing.assert_allclose(np.diag(Ad), a[0, m], rtol=1e-11, atol=1e-15)
        assert np.max(np.abs(Ad - np.diag(np.diag(Ad)))) < 1e-15
        Bz = np.zeros_like(Bd)
        Bz[np.arange(m.size), lis] = (c0 + c1)[0, m]
        np.testing.assert_allclose(Bd, Bz, rtol=1e-10, atol=1e-12 * np.max(np.abs(Bz)))


def test_foh_against_scipy_expm():
    """FOH coefficients equal the augmented-matrix discretization of a linearly varying input (the exact multi-tick
    step of a long run)."""
    sys = _thermal_bank(n=2)
    src = [sys.aggressor_src[0], sys.source_index("heater", 1)]
    A, Bm, C, D = thermal.state_space(sys, 0, None, src)
    _, _, m, lio, lis = thermal._select(sys, None, src)
    n, q = A.shape[0], Bm.shape[1]
    for h in (2e-7, 5e-6, 4e-4):
        M = np.zeros((n + 2 * q, n + 2 * q))
        M[:n, :n] = A
        M[:n, n:n + q] = Bm
        M[n:n + q, n + q:] = np.eye(q)
        E = scipy.linalg.expm(M * h)
        G0, G1 = E[:n, n:n + q], E[:n, n + q:]          # x(h) = Ad x0 + G0 u0 + G1 (u1 - u0)/h
        a, c0, c1 = sys.coefficients(h)
        C0 = np.zeros((n, q))
        C1 = np.zeros((n, q))
        C0[np.arange(m.size), lis] = c0[0, m]
        C1[np.arange(m.size), lis] = c1[0, m]
        tol = 1e-10 * np.max(np.abs(G0))
        np.testing.assert_allclose(C1, G1 / h, rtol=1e-9, atol=tol)
        np.testing.assert_allclose(C0, G0 - G1 / h, rtol=1e-9, atol=tol)


# ------------------------------------------------------------------------------------------------------------ exact FOH

@pytest.mark.parametrize("h", [0.2e-6, 1e-6, 50e-6, 1e-3])
def test_foh_exact_for_ramps(h):
    """A ramp sampled at any step is reproduced exactly by the FOH: sum r [t - tau (1 - exp(-t/tau))]."""
    sys = _thermal_bank()
    n = 40
    t = np.arange(n + 1) * h
    slope = np.zeros(sys.n_src)
    slope[sys.aggressor_src[0]] = 120.0           # W/s
    slope[sys.source_index("heater", 0)] = -35.0
    u = t[:, None, None] * slope[None, None, :]
    out = thermal.simulate(sys, u, h, hold="foh")
    ref = np.einsum("bost,s->tbo", thermal.ramp_response(sys, t), slope)
    assert np.max(np.abs(out - ref)) <= 1e-11 * np.max(np.abs(ref))


def test_foh_piecewise_linear_triangle_matches_fine_reference():
    """A triangle (piecewise linear, corners on the grid) at a coarse FOH step equals the superposition of ramps."""
    sys = _thermal_bank(n=2)
    h = 7e-6
    n = 300
    t = np.arange(n + 1) * h
    s = sys.aggressor_src[0]
    period = 60 * h
    tri = 0.05 * (1.0 - np.abs(((t / period) % 1.0) * 2.0 - 1.0))     # W, corners at multiples of 30 h
    u = np.zeros((n + 1, 1, sys.n_src))
    u[:, 0, s] = tri
    out = thermal.simulate(sys, u, h, hold="foh")
    # analytic: the triangle is a sum of ramps starting at its corners
    slope = np.diff(tri) / h
    kinks = np.r_[slope[0], np.diff(slope)]
    starts = t[:-1]
    ref = np.zeros_like(out)
    for k0, dk in zip(starts, kinks):
        if dk == 0.0:
            continue
        tt = np.clip(t - k0, 0.0, None)
        ref += dk * np.moveaxis(thermal.ramp_response(sys, tt, src=[s])[:, :, 0, :], -1, 0)
    assert np.max(np.abs(out - ref)) <= 1e-9 * np.max(np.abs(ref))


# -------------------------------------------------------------------------------------------------------------- kernels

def test_njit_kernels_against_numpy():
    sys = _thermal_bank()
    rng = np.random.default_rng(1)
    a, c0, c1 = sys.coefficients(1e-6)
    x = rng.normal(size=(1, sys.n_lags))
    u0 = rng.uniform(0, 1e-2, size=(1, sys.n_src))
    u1 = rng.uniform(0, 1e-2, size=(1, sys.n_src))
    ss = rng.uniform(0.9, 1.1, size=(1, sys.n_src))
    ref = a * x + ss[:, sys.lag_src] * (c0 * u0[:, sys.lag_src] + c1 * u1[:, sys.lag_src])
    thermal.thermal_step(x, a, c0, c1, sys.lag_src, u0, u1, ss)
    np.testing.assert_allclose(x, ref, rtol=1e-14)
    out = sys.outputs(x, 300.0)
    ref_out = 300.0 + np.bincount(sys.lag_obs, weights=x[0], minlength=sys.n_obs)
    np.testing.assert_allclose(out[0], ref_out, rtol=1e-14)
    # the fused one-pass kernel equals step + outputs
    xf = x.copy()
    xs = x.copy()
    outf = np.zeros((1, sys.n_obs))
    thermal.thermal_advance(xf, a, c0, c1, sys.lag_src, sys.obs_ptr, u0, u1, ss, np.array([290.0]), outf)
    thermal.thermal_step(xs, a, c0, c1, sys.lag_src, u0, u1, ss)
    np.testing.assert_allclose(xf, xs, rtol=1e-15)
    np.testing.assert_allclose(outf, sys.outputs(xs, 290.0), rtol=1e-14)
    # variable-step kernel equals the precomputed one
    x2 = x.copy()
    x3 = x.copy()
    thermal.thermal_step_h_b(x2[0], sys.taus_s[0], sys.res_k_per_w[0], sys.lag_src, u0[0], u1[0], ss[0], 3.7e-6)
    a, c0, c1 = sys.coefficients(3.7e-6)
    thermal.thermal_step_b(x3[0], a[0], c0[0], c1[0], sys.lag_src, u0[0], u1[0], ss[0])
    np.testing.assert_allclose(x2, x3, rtol=1e-13)


def test_observer_split_predicts_the_step():
    """pre + g u_self equals the observer after the step (the heater TCR loop interface)."""
    sys = _thermal_bank()
    rng = np.random.default_rng(2)
    a, c0, c1 = sys.coefficients(1e-6)
    x = rng.normal(size=sys.n_lags)
    u = rng.uniform(0, 2e-2, size=sys.n_src)
    ss = np.ones(sys.n_src)
    o = sys.observer_index("heater_resistor", 2)
    s = sys.source_index("heater", 2)
    pre, g = thermal.observer_split_b(x, a[0], c0[0], c1[0], sys.lag_src, sys.obs_ptr, o, s, u, u, ss)
    thermal.thermal_step_b(x, a[0], c0[0], c1[0], sys.lag_src, u, u, ss)
    out = sys.outputs(x[None], 0.0)[0, o]
    assert pre + g * u[s] == pytest.approx(out, rel=1e-13)
    assert g > 0


def test_steady_state_matches_dc_gain():
    sys = _thermal_bank()
    u = np.linspace(0, 1e-2, sys.n_src)[None]
    x = sys.steady_state(u)
    np.testing.assert_allclose(sys.outputs(x)[0], thermal.dc_gain(sys)[0] @ u[0], rtol=1e-12)
    # a long ZOH run converges to it
    a, c0, c1 = sys.coefficients(1.0)
    x2 = sys.zero_state()
    thermal.thermal_step(x2, a, c0, c1, sys.lag_src, u, u, np.ones((1, sys.n_src)))
    np.testing.assert_allclose(x2, x, rtol=1e-12)


def test_frequency_response_matches_state_space():
    sys = _thermal_bank(n=2)
    f = np.array([0.0, 10.0, 1e3, 3e4, 1e6])
    H = thermal.frequency_response(sys, f)
    A, Bm, C, D = thermal.state_space(sys)
    for k, fk in enumerate(f):
        ref = C @ np.linalg.solve(2j * np.pi * fk * np.eye(A.shape[0]) - A, Bm)
        np.testing.assert_allclose(H[0, :, :, k], ref, rtol=1e-9, atol=1e-9 * np.max(np.abs(ref)))
    np.testing.assert_allclose(H[0, :, :, 0].real, thermal.dc_gain(sys)[0], rtol=1e-12)


# --------------------------------------------------------------------------------------------------- physics of the ROM

def test_reciprocity_of_residues_interim():
    """Z_ij = Z_ji for like sources and observers, including per-ring Z0 variation and an irregular layout."""
    rng = np.random.default_rng(3)
    xy = np.c_[np.cumsum(rng.uniform(15, 60, 6)), rng.uniform(-5, 5, 6)]
    z0 = rng.uniform(4e3, 8e3, size=(2, 6))
    sys = thermal.assemble(thermal.InterimRom(z_heater_ring_k_per_w=z0, z_ratio_opt_heater=[1.2, 1.9]),
                           thermal.LayoutSites(xy), 6)
    assert sys.batch == 2
    assert thermal.reciprocity_error(sys) < 1e-13
    corner = thermal.assemble(thermal.InterimRom.textbook_corner(), thermal.LayoutSites.line(5, 20.0), 5)
    assert thermal.reciprocity_error(corner) < 1e-13


def test_interim_matches_07_p1_2_numbers():
    """The interim model has the numbers that the design review gives for it: crosstalk 3 % to the nearest
    neighbour and a 1.5 % floor, an optical self path with a sub-us pole, the self-path shares and the
    crosstalk time constants."""
    sys = thermal.assemble(thermal.InterimRom(z_heater_ring_k_per_w=H86_Z0), thermal.LayoutSites.line(8, 30.0), 8)
    G = thermal.dc_gain(sys)[0]
    r = [sys.observer_index("ring_mode_weighted", i) for i in range(8)]
    hsrc = [sys.source_index("heater", i) for i in range(8)]
    assert G[r[0], hsrc[0]] == pytest.approx(H86_Z0)
    assert G[r[1], hsrc[0]] / H86_Z0 == pytest.approx(0.03)          # nearest neighbour 3 %
    assert G[r[3], hsrc[2]] / H86_Z0 == pytest.approx(0.03)
    for j in range(2, 8):
        assert G[r[j], hsrc[0]] / H86_Z0 == pytest.approx(0.015)     # floor 1.5 %
    # optical self path: DC = z Z0 and a sub-us pole (the waveguide-core self-heating pole)
    wg = sys.source_index("waveguide_absorption", 0)
    assert G[r[0], wg] == pytest.approx(1.5 * H86_Z0)
    m = np.flatnonzero((sys.lag_obs == r[0]) & (sys.lag_src == wg))
    assert sys.taus_s[0, m].min() < 1e-6
    assert thermal.selfheat_dc_k_per_w(sys)[0] == pytest.approx(np.full(8, 1.5 * H86_Z0))
    # self path shares: 3.2 us (40.85 %), 13 us (54.15 %), 4 ms tail (5 %)
    m = np.flatnonzero((sys.lag_obs == r[0]) & (sys.lag_src == hsrc[0]))
    np.testing.assert_allclose(sys.taus_s[0, m], [3.2e-6, 13e-6, 4e-3])
    np.testing.assert_allclose(sys.res_k_per_w[0, m] / H86_Z0, [0.4085, 0.5415, 0.05])
    # crosstalk dynamics: first order, tau = max(0.1 ms, 4.1 ms (d/60)^2) in the near field, d the heat-source-to-
    # victim distance: the 30 um pitch less the two default ring radii (7.5 um)
    m = np.flatnonzero((sys.lag_obs == r[1]) & (sys.lag_src == hsrc[0]))
    assert m.size == 1 and sys.taus_s[0, m[0]] == pytest.approx(4.1e-3 * ((30.0 - 2 * 7.5) / 60.0) ** 2)
    m = np.flatnonzero((sys.lag_obs == r[7]) & (sys.lag_src == hsrc[0]))
    assert 4.0e-3 <= sys.taus_s[0, m[0]] <= 9.0e-3                  # 210 um: in the 4-9 ms band
    # 63 % point of the two-pole fit alone (the thermal notes' verification log, V5, prints 7.3 us; the exact value
    # is 7.35 us)
    two = thermal.assemble(thermal.InterimRom(z_heater_ring_k_per_w=1.0, tau_fast_s=3.18e-6, share_fast_frac=0.427,
                                              tau_slow_s=13.01e-6, tail_share_frac=0.0), None, 1)
    t = np.linspace(0, 30e-6, 30001)
    sr = thermal.step_response(two, t, obs=[0], src=[two.source_index("heater", 0)])[0, 0, 0]
    assert t[np.searchsorted(sr, 1 - math.exp(-1))] == pytest.approx(7.35e-6, abs=0.01e-6)


def test_crosstalk_tau_laws():
    d = np.array([1.0, 9.0, 15.0, 30.0, 60.0, 80.0, 100.0, 160.0, 400.0, 2000.0])
    cap = thermal.crosstalk_tau_s(d)
    assert np.all(np.diff(cap) >= 0)                                   # monotone
    assert cap[0] == pytest.approx(1e-4) and cap[2] == pytest.approx(4.1e-3 * 0.0625)
    assert cap[4] == pytest.approx(4.1e-3) and cap[-1] == pytest.approx(8.99e-3)
    assert np.all(cap <= 9e-3)
    lit = thermal.crosstalk_tau_s(d, tau_far_s=6e-3, law="literal")
    # the design review's d^2 law taken literally: 11.4 ms at 100 um
    assert lit[6] == pytest.approx(4.1e-3 * (100 / 60) ** 2)
    assert lit[7] == pytest.approx(6e-3)


def test_crosstalk_tau_takes_the_source_to_victim_distance():
    """physics_thermal review (2026-10-16): the X2 table and the 60 um anchor of the d^2 law are imec's
    heat-source-to-victim distances (first neighbour 60 um from the heat source at a 100 um pitch; thermal notes,
    section 2.5), while the assembly passed centre distances: imec's own geometry gave 4.75 ms for the first neighbour
    instead of the printed 4.14 ms. With the centre distance less both ring radii, rings of R = 20 um at a 100 um pitch
    (imec's geometry, inferred) reproduce Table II for neighbours 2-7 and the 4.1 ms FE anchor for the first (Table II
    prints 4.14 ms, within 1 %); site paths subtract the radius of their ring element only."""
    s = thermal.assemble(thermal.InterimRom(ring_radius_um=20.0), thermal.LayoutSites.line(8, 100.0), 8,
                         sources=["heater"], observers=["ring_mode_weighted"])
    taus = []
    for k in range(1, 8):
        m = np.flatnonzero((s.lag_obs == s.ring_obs[k]) & (s.lag_src == s.heater_src[0]))
        assert m.size == 1
        taus.append(s.taus_s[0, m[0]])
    np.testing.assert_allclose(taus[1:], thermal.XT_TABLE_TAU_S[1:], rtol=1e-12)
    assert taus[0] == pytest.approx(4.1e-3, rel=1e-12) and taus[0] == pytest.approx(thermal.XT_TABLE_TAU_S[0], rel=0.01)
    # per-ring radii (B, N): each pair subtracts its own two radii; a PD site (point) only its ring's radius
    rr = np.array([[5.0, 7.0, 9.0]])
    lay = thermal.LayoutSites.line(3, 50.0)
    s3 = thermal.assemble(thermal.InterimRom(ring_radius_um=rr), lay, 3)
    for (j, i) in ((0, 1), (1, 2), (2, 0)):
        m = np.flatnonzero((s3.lag_obs == s3.ring_obs[i]) & (s3.lag_src == s3.heater_src[j]))
        d = abs(i - j) * 50.0 - rr[0, i] - rr[0, j]
        assert s3.taus_s[0, m[0]] == pytest.approx(float(thermal.crosstalk_tau_s(d)), rel=1e-12)
    m = np.flatnonzero((s3.lag_obs == s3.pd_obs[1]) & (s3.lag_src == s3.heater_src[1]))
    assert s3.taus_s[0, m[0]] == pytest.approx(float(thermal.crosstalk_tau_s(25.0 - 7.0)), rel=1e-12)
    # the site coupling (an assumed law) keeps the centre distance: DC unchanged by the radius
    s0 = thermal.assemble(thermal.InterimRom(ring_radius_um=0.0), lay, 3)
    np.testing.assert_allclose(thermal.dc_gain(s3), thermal.dc_gain(s0), rtol=1e-14)
    with pytest.raises(ValueError, match="ring_radius_um"):
        thermal.assemble(thermal.InterimRom(ring_radius_um=-1.0), lay, 3)


def test_scaling_rules():
    lay = thermal.LayoutSites.line(3, 40.0)
    base = thermal.assemble(thermal.InterimRom(), lay, 3, thermal.ThermalScales(scaling_hypothesis="independent"))
    sc = thermal.ThermalScales(dc_scale={"*": [1.0, 1.3], "heater->ring_mode_weighted": [2.0, 1.0]},
                               time_scale_fast=[0.5, 1.0], time_scale_slow=[1.0, 3.0], crosstalk_scale=[2.0, 0.5],
                               scaling_hypothesis=["resistance", "independent"])
    s = thermal.assemble(thermal.InterimRom(), lay, 3, sc)
    assert s.batch == 2 and s.n_lags == base.n_lags
    np.testing.assert_array_equal(s.lag_src, base.lag_src)
    sig = s.res_k_per_w / base.res_k_per_w
    hr = (base.src_kind[base.lag_src] == thermal.SRC_HEATER) & (base.obs_kind[base.lag_obs] == thermal.OBS_RING)
    cross = base.lag_pair == thermal.PAIR_CROSS
    exp0 = np.where(hr, 2.0, 1.0) * np.where(cross, 2.0, 1.0)
    exp1 = 1.3 * np.where(cross, 0.5, 1.0)
    np.testing.assert_allclose(sig[0], exp0, rtol=1e-14)
    np.testing.assert_allclose(sig[1], exp1, rtol=1e-14)
    ts = s.taus_s / base.taus_s
    fast = base.lag_layer == thermal.LAYER_FAST
    # resistance: tau follows the dc_scale (conductance) factor of its residue, never the crosstalk magnitude scale
    np.testing.assert_allclose(ts[0], np.where(fast, 0.5, 1.0) * np.where(hr, 2.0, 1.0), rtol=1e-14)
    np.testing.assert_allclose(ts[1], np.where(fast, 1.0, 3.0), rtol=1e-14)           # independent
    # crosstalk lags (slow layer, time_scale_slow = 1 in world 0): crosstalk_scale 2 moved the residue, not tau
    assert np.all(~fast[cross])
    np.testing.assert_allclose(ts[0, cross], np.where(hr, 2.0, 1.0)[cross], rtol=1e-14)
    np.testing.assert_allclose(sig[0, cross], 2.0 * np.where(hr, 2.0, 1.0)[cross], rtol=1e-14)
    cap = thermal.assemble(thermal.InterimRom(), lay, 3,
                           thermal.ThermalScales(dc_scale=1.7, scaling_hypothesis="capacitance"))
    np.testing.assert_allclose(cap.taus_s, base.taus_s, rtol=1e-14)
    np.testing.assert_allclose(cap.res_k_per_w, 1.7 * base.res_k_per_w, rtol=1e-14)
    with pytest.raises(ValueError):
        thermal.assemble(thermal.InterimRom(), lay, 3, thermal.ThermalScales(scaling_hypothesis="bogus"))


def test_package_tail_preserves_dc_and_ratio():
    lay = thermal.LayoutSites.line(2, 40.0)
    base = thermal.assemble(thermal.InterimRom(), lay, 2)
    s = thermal.assemble(thermal.InterimRom(), lay, 2, thermal.ThermalScales(package_tail_tau_s=[1e-3, 1e-2],
                                                                             package_tail_share_frac=[0.02, 0.10]))
    np.testing.assert_allclose(thermal.dc_gain(s),
                               np.broadcast_to(thermal.dc_gain(base), (2,) + thermal.dc_gain(base).shape[1:]),
                               rtol=1e-12)
    r0, h0 = s.observer_index("ring_mode_weighted", 0), s.source_index("heater", 0)
    m = np.flatnonzero((s.lag_obs == r0) & (s.lag_src == h0))
    np.testing.assert_allclose(s.taus_s[:, m[-1]], [1e-3, 1e-2])
    np.testing.assert_allclose(s.res_k_per_w[:, m[-1]] / 6.1e3, [0.02, 0.10])
    np.testing.assert_allclose(s.res_k_per_w[:, m[0]] / s.res_k_per_w[:, m[1]], 0.4085 / 0.5415)


def test_batch_rows_equal_single_worlds():
    lay = thermal.LayoutSites.line(3, 25.0)
    z = [5e3, 7e3]
    tsf = [0.7, 1.4]
    sb = thermal.assemble(thermal.InterimRom(z_heater_ring_k_per_w=z), lay, 3,
                          thermal.ThermalScales(time_scale_fast=tsf))
    for b in range(2):
        s1 = thermal.assemble(thermal.InterimRom(z_heater_ring_k_per_w=z[b]), lay, 3,
                              thermal.ThermalScales(time_scale_fast=tsf[b]))
        np.testing.assert_array_equal(sb.taus_s[b], s1.taus_s[0])
        np.testing.assert_array_equal(sb.res_k_per_w[b], s1.res_k_per_w[0])


def test_cross_lag_gain_assertion():
    """The lagged cross-ring optical gain per 1 us substep is far below 0.05 for the interim model (realize asserts
    this bound, so that the cross-ring optical heating may lag one substep)."""
    sys = thermal.assemble(thermal.InterimRom(), thermal.LayoutSites.line(8, 20.0), 8)
    hwhm, s0 = 1310e-9 / 3800 / 2, 77e-12
    p_peak = 0.85 * 7.94e-3                                               # +9 dBm at the ring, eta 0.85
    d = 3 * math.sqrt(3) / 8 * p_peak * s0 / hwhm
    g = thermal.assert_cross_lag_gain(sys, 1e-6, np.full(8, d))
    assert 0 < g < 0.05
    fast = thermal.assemble(thermal.InterimRom(xt_nearest_frac=0.4, xt_floor_frac=0.4, xt_tau_min_s=1e-7,
                                               xt_tau_ref_s=1e-7),
                            thermal.LayoutSites.line(8, 20.0), 8)
    with pytest.raises(ValueError, match="lagged cross-ring"):
        thermal.assert_cross_lag_gain(fast, 1e-6, np.full(8, d))


def test_interim_input_validation():
    with pytest.raises(ValueError):
        thermal.assemble(thermal.InterimRom(z_ratio_opt_heater=0.8), None, 1)
    with pytest.raises(ValueError):
        thermal.assemble(thermal.InterimRom(share_fast_frac=0.7, tail_share_frac=0.4), None, 1)
    with pytest.warns(RuntimeWarning, match="positive semidefinite"):
        thermal.assemble(thermal.InterimRom(z_ratio_opt_heater=1.5, selfheat_share_frac=0.6), None, 1)
    with pytest.raises(ValueError):
        thermal.assemble(thermal.InterimRom(), None, 2)


# -------------------------------------------------------------------------------------------------------- ROM artefacts

def synthetic_rom(sym=True):
    """A small symmetric ROM: 5 poles from 50 ns to 6 ms, cross residues decaying with distance."""
    taus = np.array([5e-8, 8e-7, 6e-6, 4e-5, 6e-3])
    srcs = ("heater", "waveguide_absorption", "junction", "aggressor")
    obss = ("ring_mode_weighted", "heater_resistor", "monitor_pd", "temp_sensor")
    S, O, K = len(srcs), len(obss), taus.size
    rng = np.random.default_rng(7)
    self_res = rng.uniform(100, 2000, size=(S, O, K))
    # like pairs reciprocal: heater->HR symmetric itself; heater->ring == wg->HR
    self_res[1, 1] = self_res[0, 0] if sym else self_res[1, 1]
    r = np.array([20.0, 40.0, 80.0, 160.0, 320.0])
    cross = np.zeros((S, O, r.size, K))
    for di, d in enumerate(r):
        cross[:, :, di, -1] = 150.0 * (40.0 / d) ** 0.4          # ms pole carries the crosstalk
        cross[:, :, di, 2] = 30.0 * math.exp(-d / 30.0)
    return thermal.ThermalRom(taus, srcs, obss, self_res, r, cross,
                              {"monitor_pd": (0.0, 12.0), "temp_sensor": (0.0, -12.0)}, {})


def test_rom_roundtrip_sha256_and_tamper(tmp_path):
    rom = synthetic_rom()
    side = {"family": "inring_noundercut", "film_nm": 160,
            "validation": {"step_max_rel_err": 0.01, "freq_max_db_err": 0.2, "freq_max_deg_err": 2.0,
                           "superposition_rel_err": 0.05}}
    p = thermal.save_rom(tmp_path / "thermal_rom_inring_noundercut_160.npz", rom, side)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r2 = thermal.load_rom(p)
    np.testing.assert_array_equal(r2.self_res, rom.self_res)
    np.testing.assert_array_equal(r2.cross_res, rom.cross_res)
    assert r2.sources == rom.sources and r2.site_offsets_um["monitor_pd"] == (0.0, 12.0)
    assert r2.meta["sha256"] == thermal.sha256_file(p)
    # tamper with the data file
    b = bytearray(p.read_bytes())
    b[-10] ^= 0xFF
    p.write_bytes(bytes(b))
    with pytest.raises(thermal.RomIntegrityError, match="sha256"):
        thermal.load_rom(p)
    # missing sidecar / wrong schema
    p2 = thermal.save_rom(tmp_path / "b.npz", rom, side)
    sp = p2.with_name(p2.stem + ".sidecar.json")
    assert sp.exists()
    side2 = json.loads(sp.read_text())
    side2["schema"] = "thermal_rom/0"
    sp.write_text(json.dumps(side2))
    with pytest.raises(thermal.RomIntegrityError, match="schema"):
        thermal.load_rom(p2)
    side2["schema"] = thermal.SCHEMA_ROM
    sp.unlink()
    p2.with_suffix(".json").write_text(json.dumps(side2))       # the plain <stem>.json name is accepted too
    assert thermal.load_rom(p2, warn=False).taus_s.size == 5
    p2.with_suffix(".json").unlink()
    with pytest.raises(thermal.RomIntegrityError, match="sidecar"):
        thermal.load_rom(p2)
    assert thermal.load_rom(p2, verify=False, warn=False).taus_s.size == 5


def test_rom_quality_warnings(tmp_path):
    rom = synthetic_rom()
    p = thermal.save_rom(tmp_path / "c.npz", rom, {"validation": {"step_max_rel_err": 0.05}})
    with pytest.warns(RuntimeWarning, match="step_max_rel_err"):
        thermal.load_rom(p)
    slow = thermal.ThermalRom(rom.taus_s * 30, rom.sources, rom.observers, rom.self_res, rom.cross_r_um, rom.cross_res)
    p = thermal.save_rom(tmp_path / "d.npz", slow, {})
    with pytest.warns(RuntimeWarning, match="below"):
        thermal.load_rom(p)


def test_rom_assembly_superposition():
    """Same-ring pairs use self_res; other pairs interpolate cross_res at the source-centre to observer-centre
    distance (docs/ARCHITECTURE.md, "5.3 Thermal ROM", the convention of mrm/extraction/thermal_fem.py)."""
    rom = synthetic_rom()
    xy = np.array([[0.0, 0.0], [40.0, 0.0], [100.0, 0.0]])
    lay = thermal.LayoutSites(xy, aggressor_xy_um=[[0.0, 60.0]], temp_sensor_xy_um=[[0, 0], [300.0, 0.0]],
                              temp_sensor_ring=[1, -1])
    sys = thermal.assemble(rom, lay, 3)
    G = thermal.dc_gain(sys)[0]
    si = {k: i for i, k in enumerate(rom.sources)}
    oi = {k: i for i, k in enumerate(rom.observers)}
    for s in ("heater", "waveguide_absorption", "junction"):
        for o in ("ring_mode_weighted", "heater_resistor", "monitor_pd"):
            for i in range(3):
                assert G[sys.observer_index(o, i), sys.source_index(s, i)] == pytest.approx(
                    rom.self_res[si[s], oi[o]].sum())
            for (j, i) in ((0, 1), (2, 0)):
                # source centre -> observer centre; the PD sits at its ROM default offset from its ring
                ob = xy[i] + (np.array(rom.site_offsets_um["monitor_pd"]) if o == "monitor_pd" else 0.0)
                d = float(np.linalg.norm(xy[j] - ob))
                ref = sum(np.interp(d, rom.cross_r_um, rom.cross_res[si[s], oi[o], :, k]) for k in range(5))
                assert G[sys.observer_index(o, i), sys.source_index(s, j)] == pytest.approx(ref)
    # attached sensor of ring 1 uses the self table; the free sensor and the aggressor use the distance table
    ts1 = sys.observer_index("temp_sensor", name="ts0")
    assert G[ts1, sys.source_index("heater", 1)] == pytest.approx(rom.self_res[0, 3].sum())
    agg = sys.aggressor_src[0]
    ref = sum(np.interp(60.0, rom.cross_r_um, rom.cross_res[3, 0, :, k]) for k in range(5))
    assert G[sys.observer_index("ring_mode_weighted", 0), agg] == pytest.approx(ref)
    assert thermal.reciprocity_error(sys) < 1e-12
    # the fast/slow split follows tau_split_s; tail added from scales preserves DC
    assert set(sys.lag_layer[sys.taus_s[0] < thermal.DEFAULT_TAU_SPLIT_S]) == {thermal.LAYER_FAST}
    s2 = thermal.assemble(rom, lay, 3, thermal.ThermalScales(package_tail_share_frac=0.05, package_tail_tau_s=3e-3))
    np.testing.assert_allclose(thermal.dc_gain(s2), thermal.dc_gain(sys), rtol=1e-12)


def test_layout_from_sites():
    sites = {"ring0": {"x_um": 0, "y_um": 0, "kind": "ring"}, "ring1": {"x_um": 35, "y_um": 0, "kind": "ring"},
             "pd1": {"x_um": 35, "y_um": 20, "kind": "monitor_pd"},
             "eic": {"x_um": 20, "y_um": -500, "kind": "driver"},
             "agg": {"x_um": 17, "y_um": 15, "kind": "heater"},
             "tsA": {"x_um": 100, "y_um": 0, "kind": "temp_sensor"}}
    lay = thermal.LayoutSites.from_sites(sites, 2, temp_sensors=[{"name": "die_ts", "attach": "die"},
                                                                 {"name": "r0", "attach": "ring_0"}])
    np.testing.assert_allclose(lay.ring_xy_um, [[0, 0], [35, 0]])
    np.testing.assert_allclose(lay.pd_xy_um, [[np.nan, np.nan], [35, 20]])      # ring 0: no PD site (NaN row)
    np.testing.assert_allclose(lay.driver_xy_um, [[20, -500], [20, -500]])
    assert lay.aggressor_names == ("agg",)
    assert lay.temp_sensor_names == ("tsA", "die_ts", "r0") and list(lay.temp_sensor_ring) == [-1, -1, 0]
    sys = thermal.assemble(thermal.InterimRom(), sites, 2, temp_sensors=[{"name": "r0", "attach": "ring_0"}])
    assert sys.driver_src.min() >= 0 and sys.aggressor_src.size == 1 and sys.temp_obs.size == 2
    # ring 0's PD (no site) sits at the model's default offset (0, 25) um, not at the ring centre: site law at 25 um
    G = thermal.dc_gain(sys)[0]
    assert G[sys.observer_index("monitor_pd", 0), sys.source_index("heater", 0)] == pytest.approx(
        thermal.site_fraction(25.0) * 6.1e3)
    assert G[sys.observer_index("monitor_pd", 1), sys.source_index("heater", 1)] == pytest.approx(
        thermal.site_fraction(20.0) * 6.1e3)
    # a driver site for ring 1 only: ring 0 has no driver source
    one = thermal.assemble(thermal.InterimRom(), {"drv1": {"x_um": 35, "y_um": -30, "kind": "driver"}}, 2,
                           ring_pitch_um=35.0)
    assert one.driver_src[0] == -1 and one.driver_src[1] >= 0
    lay2 = thermal.LayoutSites.from_sites({}, 4, ring_pitch_um=25.0)
    np.testing.assert_allclose(lay2.ring_xy_um[:, 0], [0, 25, 50, 75])
    with pytest.raises(ValueError):
        thermal.LayoutSites.from_sites({}, 4)


def test_k_temp_residue_scale():
    scale = thermal.k_temp_residue_scale(np.array([300.0, 378.15]))
    assert scale[0] == pytest.approx(1.0)
    assert 0.8 < scale[1] < 1.2
    film_only = thermal.k_temp_residue_scale(378.15, weights=(1.0, 0.0, 0.0))
    assert film_only > 1.0     # thin-film k falls with T, so R rises (docs/research/02_materials.md, section 5.2)


# ----------------------------------------------------------------------------------------------------- review additions

def test_rom_loads_the_extraction_artefact_format(tmp_path):
    """load_rom reads what mrm/extraction/thermal_fem.write_artefact writes: ``site_offset_names`` in the .npz, the
    sidecar ``<stem>.json`` with the nested verify() report, and site offsets that place the PD and sensor
    (docs/ARCHITECTURE.md, "5.3 Thermal ROM")."""
    from types import SimpleNamespace
    thermal_fem = pytest.importorskip("mrm.extraction.thermal_fem")
    rom = synthetic_rom()
    S, O = len(thermal_fem.SOURCES), len(thermal_fem.OBSERVERS)
    K, D = rom.taus_s.size, rom.cross_r_um.size
    self_res = np.zeros((S, O, K))
    cross_res = np.zeros((S, O, D, K))
    for i, s in enumerate(thermal_fem.SOURCES):
        for j, o in enumerate(thermal_fem.OBSERVERS):
            if s in rom.sources and o in rom.observers:
                self_res[i, j] = rom.self_res[rom.sources.index(s), rom.observers.index(o)]
                cross_res[i, j] = rom.cross_res[rom.sources.index(s), rom.observers.index(o)]
    offs = {"monitor_pd": [0.0, 25.0], "temp_sensor": [40.0, 0.0], "driver": [0.0, -30.0]}
    cfg = SimpleNamespace(family="inring_noundercut", film_nm=160.0, site_offsets_um=offs)
    fr = SimpleNamespace(taus=rom.taus_s, self_res=self_res, cross_r=rom.cross_r_um, cross_res=cross_res,
                         dc_self=self_res.sum(-1), dc_cross=cross_res.sum(-1))
    good = {"self": {"max_db": 0.1, "max_deg": 1.0, "max_time_err_frac_of_dc": 0.01},
            "nearest_neighbour": {"max_db": 0.2, "max_deg": 2.5, "max_time_err_frac_of_dc": 0.015}}
    side = {"family": cfg.family, "film_nm": 160.0, "validation": good, "inputs": {"config": {"site_offsets_um": offs}},
            "stages": {"3d": {"superposition_rel_err": 0.04}}}
    npz, js = thermal_fem.write_artefact(cfg, fr, side, out_dir=tmp_path)
    assert js.name == npz.stem + ".json"
    with np.load(npz) as z:                              # the names key the load_rom docstring documents
        assert "site_offset_names" in z.files and "site_names" not in z.files
    with warnings.catch_warnings():
        warnings.simplefilter("error")                   # good numbers in the nested report: no quality warning
        r = thermal.load_rom(npz)
    assert r.site_offsets_um["monitor_pd"] == (0.0, 25.0) and r.site_offsets_um["temp_sensor"] == (40.0, 0.0)
    assert thermal.rom_validation_numbers(r.meta) == {"step_max_rel_err": 0.015, "freq_max_db_err": 0.2,
                                                      "freq_max_deg_err": 2.5, "superposition_rel_err": 0.04}
    # the PD of ring 1 sits at its offset: the heater-0 -> PD-1 residue uses the distance to the PD site
    xy = np.array([[0.0, 0.0], [40.0, 0.0]])
    sys = thermal.assemble(r, thermal.LayoutSites(xy), 2, sources=("heater",), observers=("monitor_pd",))
    d = float(np.linalg.norm(xy[0] - (xy[1] + [0.0, 25.0])))
    ref = sum(np.interp(d, r.cross_r_um, r.cross_res[0, 2, :, k]) for k in range(K))
    assert thermal.dc_gain(sys)[0, sys.observer_index("monitor_pd", 1), sys.source_index("heater", 0)] == \
        pytest.approx(ref)
    # bad numbers in the nested report do warn
    bad = json.loads(js.read_text())
    bad["validation"]["nearest_neighbour"]["max_deg"] = 7.0
    js.write_text(json.dumps(bad))
    with pytest.warns(RuntimeWarning, match="freq_max_deg_err"):
        thermal.load_rom(npz)


def test_scaling_hypothesis_list_sets_the_batch():
    s = thermal.assemble(thermal.InterimRom(), None, 1,
                         thermal.ThermalScales(scaling_hypothesis=["resistance", "independent"], dc_scale={"*": 1.0}))
    assert s.batch == 2
    with pytest.raises(ValueError, match="inconsistent batch"):
        thermal.assemble(thermal.InterimRom(), None, 1, thermal.ThermalScales(scaling_hypothesis=["resistance"] * 2,
                                                                time_scale_fast=[1.0, 1.1, 1.2]))


def test_package_tail_from_scales_is_absolute():
    """A package tail drawn in uncertainty.thermal (1-10 ms in the uncertainty set) keeps its tau under time_scale_slow
    and under the 'resistance' factor; the model's own tail (a family value) scales with time_scale_slow."""
    sc = thermal.ThermalScales(time_scale_slow=2.0, dc_scale=1.5, package_tail_tau_s=6e-3, package_tail_share_frac=0.08)
    s = thermal.assemble(thermal.InterimRom(), None, 1, sc)
    r0, h0 = s.observer_index("ring_mode_weighted", 0), s.source_index("heater", 0)
    m = np.flatnonzero((s.lag_obs == r0) & (s.lag_src == h0))
    tail = m[s.lag_tail[m] == 1]
    assert tail.size == 1 and s.taus_s[0, tail[0]] == pytest.approx(6e-3)
    assert s.res_k_per_w[0, tail[0]] == pytest.approx(0.08 * 1.5 * 6.1e3)
    own = thermal.assemble(thermal.InterimRom(), None, 1,
                           thermal.ThermalScales(time_scale_slow=2.0, scaling_hypothesis="independent"))
    m = np.flatnonzero((own.lag_obs == r0) & (own.lag_src == h0) & (own.lag_tail == 1))
    assert own.taus_s[0, m[0]] == pytest.approx(2.0 * 4e-3)


def test_transport_delay_is_passed_through():
    s = thermal.assemble(thermal.InterimRom(), None, 1, thermal.ThermalScales(transport_delay_s=[0.0, 1e-6, 2e-6]))
    np.testing.assert_array_equal(s.transport_delay_s, [0.0, 1e-6, 2e-6])
    assert s.batch == 3
    with pytest.raises(ValueError):
        thermal.assemble(thermal.InterimRom(), None, 1, thermal.ThermalScales(transport_delay_s=-1e-6))


def test_temp_sensor_site_referenced_by_the_table_is_not_duplicated():
    sites = {"ts_site": {"x_um": 100.0, "y_um": 0.0, "kind": "temp_sensor"},
             "free": {"x_um": -50.0, "y_um": 0.0, "kind": "temp_sensor"}}
    lay = thermal.LayoutSites.from_sites(sites, 2, ring_pitch_um=30.0,
                                         temp_sensors=[{"name": "T_main", "site": "ts_site"}])
    assert lay.temp_sensor_names == ("free", "T_main")
    np.testing.assert_allclose(lay.temp_sensor_xy_um, [[-50.0, 0.0], [100.0, 0.0]])


def test_batched_kernels_reject_mismatched_rows():
    """numba does not bounds-check: the batched kernels must refuse row counts other than 1 or B."""
    sys = thermal.assemble(thermal.InterimRom(z_heater_ring_k_per_w=[5e3, 6e3]), thermal.LayoutSites.line(2, 30.0), 2)
    a, c0, c1 = sys.coefficients(1e-6)
    x = np.zeros((3, sys.n_lags))
    u = np.zeros((1, sys.n_src))
    with pytest.raises(ValueError):
        thermal.thermal_step(x, a, c0, c1, sys.lag_src, u, u, np.ones((1, sys.n_src)))      # a has 2 rows, x has 3
    with pytest.raises(ValueError):
        thermal.thermal_advance(x[:2], a, c0, c1, sys.lag_src, sys.obs_ptr, u, u, np.ones((1, sys.n_src)),
                                np.zeros(1), np.zeros((2, sys.n_obs + 1)))
    with pytest.raises(ValueError):
        thermal.simulate(sys, np.zeros((3, 3, sys.n_src)), 1e-6)


def test_rom_driver_sources_and_moved_sites():
    """ROM path: drivers only where the layout puts them (or on explicit request at the ROM offset, as in the interim
    model); a same-ring site away from the ROM's self-table offset (a shared driver block) uses the distance table."""
    base = synthetic_rom()
    srcs = base.sources + ("driver",)
    S, O, K = len(srcs), len(base.observers), base.taus_s.size
    self_res = np.concatenate([base.self_res, base.self_res[:1] * 0.5], axis=0)
    cross = np.concatenate([base.cross_res, base.cross_res[:1] * 0.7], axis=0)
    offs = dict(base.site_offsets_um, driver=(0.0, -30.0))
    rom = thermal.ThermalRom(base.taus_s, srcs, base.observers, self_res, base.cross_r_um, cross, offs, {})
    lay = thermal.LayoutSites.line(2, 50.0)
    assert np.all(thermal.assemble(rom, lay, 2).driver_src == -1)                 # default: no driver heat on the PIC
    s = thermal.assemble(rom, lay, 2, sources=("heater", "driver"), observers=("ring_mode_weighted",))
    G = thermal.dc_gain(s)[0]
    r0, d0 = s.observer_index("ring_mode_weighted", 0), s.source_index("driver", 0)
    assert G[r0, d0] == pytest.approx(self_res[S - 1, 0].sum())             # explicit: ROM offset, self table
    # one shared driver block 400 um away: the same-ring pair is a distance-table pair, not the 30 um self table
    far = thermal.LayoutSites.line(2, 50.0, driver_xy_um=[[25.0, -400.0], [25.0, -400.0]])
    with pytest.warns(RuntimeWarning, match="distance table"):
        s2 = thermal.assemble(rom, far, 2, sources=("heater", "driver"), observers=("ring_mode_weighted",))
    d = float(np.hypot(25.0, 400.0))
    ref = sum(np.interp(d, rom.cross_r_um, cross[S - 1, 0, :, k]) for k in range(K))
    G2 = thermal.dc_gain(s2)[0]
    for i in range(2):
        assert G2[s2.observer_index("ring_mode_weighted", i), s2.source_index("driver", i)] == pytest.approx(ref)
    assert s2.meta["rom_sites_off_default"] == ["driver"]


# ----------------------------------------------------------------------------------- die-level time scales on a FEM ROM

_ROM_FILES = sorted((Path(thermal.__file__).resolve().parents[1] / "data" / "extracted").glob("thermal_rom_*.npz"))
_T_STEP = np.logspace(-9, 0, 4000)
_F_RESP = np.logspace(0, 6, 300)


def _pair_shape(sys, o, s):
    """(peak / DC, largest drop below the running maximum / DC, max |H(f)| / H(0)) of one pair."""
    y = thermal.step_response(sys, _T_STEP, obs=[o], src=[s])[0, 0, 0]
    dc = thermal.dc_gain(sys)[0][o, s]
    h = np.abs(thermal.frequency_response(sys, _F_RESP, obs=[o], src=[s])[0, 0, 0])
    return y.max() / dc, (np.maximum.accumulate(y) - y).max() / dc, h.max() / abs(dc)


@pytest.mark.skipif(not _ROM_FILES, reason="no thermal ROM artefact")
@pytest.mark.parametrize("fast, slow", [(2.0, 0.5), (0.5, 2.0), (1.83, 0.55)])
@pytest.mark.parametrize("path", _ROM_FILES, ids=lambda p: p.stem)
def test_rom_layer_time_scales_keep_every_pair_monotone(path, fast, slow):
    """physics_thermal review (2026-10-16): with unequal die-level time scales (fast and slow drawn per world; g26 draws
    both in 0.5-2x) a FEM ROM pair must keep the shape of its fit. Per-pole layer factors pulled the cancelling
    mixed-sign residues of a non-collocated pair apart across tau_split: heater -> ring of metal_above_160 at x2 / x0.5
    peaked at 1.21 x DC and dropped by 0.22 x DC (|H| 1.27 x DC near 600 Hz), impossible for a heat-diffusion response.
    Now each such pair is time-scaled by one blended factor: its peak, dips and |H| stay those of the unscaled fit."""
    rom = thermal.load_rom(path, warn=False)
    lay = thermal.LayoutSites.line(2, 50.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        base = thermal.assemble(rom, lay, 2)
        sysx = thermal.assemble(rom, lay, 2, thermal.ThermalScales(time_scale_fast=fast, time_scale_slow=slow))
    pairs = [(base.ring_obs[0], base.heater_src[0]), (base.heater_obs[0], base.heater_src[0]),
             (base.ring_obs[0], base.wg_src[0]), (base.ring_obs[1], base.heater_src[0]),
             (base.heater_obs[1], base.heater_src[0])]
    for o, s in pairs:
        p0, d0, h0 = _pair_shape(base, o, s)
        p1, d1, h1 = _pair_shape(sysx, o, s)
        tag = f"{base.obs_names[o]} <- {base.src_names[s]}"
        assert p1 <= max(p0, 1.0) + 1e-3, (tag, p0, p1)
        assert d1 <= d0 + 1e-3, (tag, d0, d1)
        assert h1 <= max(h0, 1.0) + 1e-3, (tag, h0, h1)
    # the whole-pair factor: one factor per pair, s_fast^w s_slow^(1 - w) with w the step fraction at tau_split
    assert sysx.meta["time_scale_whole_pairs"] > 0
    ratio = sysx.taus_s[0] / base.taus_s[0]
    for o, s in pairs[:1]:
        m = np.flatnonzero((base.lag_obs == o) & (base.lag_src == s))
        r = base.res_k_per_w[0, m]
        if np.any(r < 0) and np.any(base.taus_s[0, m] >= thermal.DEFAULT_TAU_SPLIT_S):
            w = np.clip(np.sum(r * -np.expm1(-thermal.DEFAULT_TAU_SPLIT_S / base.taus_s[0, m])) / r.sum(), 0, 1)
            np.testing.assert_allclose(ratio[m], fast ** w * slow ** (1 - w), rtol=1e-12)
    # equal factors: exactly the former per-layer scaling (every tau times the common factor)
    eq = thermal.assemble(rom, lay, 2, thermal.ThermalScales(time_scale_fast=1.7, time_scale_slow=1.7))
    np.testing.assert_array_equal(eq.taus_s, base.taus_s * 1.7)
    np.testing.assert_array_equal(eq.res_k_per_w, base.res_k_per_w)


def test_interim_layer_time_scales_stay_per_pole():
    """The interim model's residues are all positive: every pole keeps its own layer factor (no whole-pair
    scaling), so the self path's package tail follows time_scale_slow while its body follows time_scale_fast."""
    lay = thermal.LayoutSites.line(3, 50.0)
    base = thermal.assemble(thermal.InterimRom(), lay, 3)
    sysx = thermal.assemble(thermal.InterimRom(), lay, 3,
                            thermal.ThermalScales(time_scale_fast=2.0, time_scale_slow=0.5))
    assert sysx.meta["time_scale_whole_pairs"] == 0
    np.testing.assert_array_equal(sysx.taus_s[0],
                                  base.taus_s[0] * np.where(base.lag_layer == thermal.LAYER_FAST, 2.0, 0.5))
