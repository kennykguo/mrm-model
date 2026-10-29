"""Checks the implicit self-heating solve of mrm/plant/selfheat.py against analytic results.

Read this first
---------------
A ring that absorbs light heats itself, and the heat red-shifts its resonance. The plant solves this each substep
implicitly: T = T_other + g P_abs(T) for each ring in bus order (g: K/W, the self-heating gain of this substep). The
tests replace the optics by a local analytic Lorentzian (compact-model notes, docs/research/03_mrm_compact_models.md,
section 6.4):

    P_abs = p_peak / (1 + y^2),   y = (lambda_r - lambda_L) / HWHM,   lambda_r = lambda_ref + s0 (T - T_ref)

Its steady states solve the cubic (y - y_o)(1 + y^2) = A with A = K_SH p_peak / HWHM and K_SH = s0 Z_opt->ring(0)
(m/W). The folds of the hysteresis loop solve (1 + y^2)^2 + 2 A y = 0. Tests (a)-(d) are the four self-heating tests
of the design review (docs/research/07_design_critique.md): (a) the hysteresis edges sit at the folds, (b) the 1 us
implicit solve agrees with a fine reference, (c) the unstable branch grows at the linear rate and the stable branch
never rings, (d) a laser-power step moves the resonance by the static formula.

Numbers (scenario inputs, not model claims): lambda_L 1310 nm (G26 O-band grid); Q_L 3800 (the design review's
example; CAL-1, the loaded-Q target, is 3500-4200); s0 70 pm/K (the solver hypothesis H_solver gives 66-77 pm/K);
Z_heater->ring 6.1 K/mW (H_solver); Z ratio 1.5 (the uncertainty interval is 1-2); p_peak = 0.85 P_in (eta ~0.85
near critical coupling); P_in -10 / +4 / +9 dBm.
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
from numba import njit

from mrm.plant import selfheat
from mrm.plant import thermal

LAM_L = 1310e-9                 # m, laser line
Q_L = 3800.0                    # loaded Q
FWHM = LAM_L / Q_L              # m, resonance full width
HWHM = FWHM / 2.0               # m, half width (the unit of the detuning y)
S0 = 70e-12                     # m/K, d lambda_res / dT
Z0 = 6.1e3                      # K/W, Z_heater->ring (DC)
Z_RATIO = 1.5                   # Z_opt->ring / Z_heater->ring
ETA = 0.85                      # heat fraction of the ring loss
T0 = 300.0                      # K, base temperature and T_ref
LAM_REF = LAM_L - 3e-9          # m, cold resonance 3 nm blue of the line (rings are made blue of their line)


def _dbm_to_w(p):
    """Optical power in W from dBm."""
    return 1e-3 * 10 ** (p / 10)


# W, peak absorbed power p_peak = ETA * P_in for P_in = +4, +9 and -10 dBm at the ring
P4, P9, PM10 = ETA * _dbm_to_w(4.0), ETA * _dbm_to_w(9.0), ETA * _dbm_to_w(-10.0)
G = Z0 * Z_RATIO                # K/W, DC optical self-heating resistance
K_SH = S0 * G                   # m/W, resonance shift per absorbed watt
A4, A9, AM10 = (K_SH * p / HWHM for p in (P4, P9, PM10))   # the cubic's A at those powers (dimensionless)


def _one_ring_system(tail=0.05, **kw):
    """Interim single-ring system with only the paths the test needs."""
    return thermal.assemble(thermal.InterimRom(z_heater_ring_k_per_w=Z0, z_ratio_opt_heater=Z_RATIO,
                                               tail_share_frac=tail, share_fast_frac=0.43 * (1 - tail), **kw), None, 1,
                            sources=("heater", "waveguide_absorption"), observers=("ring_mode_weighted",))


def _lorentzian_bank(p_peak):
    """One ring with one line at peak absorbed power p_peak (W): (packed_f, packed_i, p_max, d2p_max) for the solve."""
    packed_f, packed_i = selfheat.pack_lorentzian_args(LAM_REF, T0, S0, 0.0, [[LAM_L]], [[HWHM]], [[p_peak]])
    pm, d1, d2 = selfheat.lorentzian_bounds(np.array([[p_peak]]), np.array([[HWHM]]), S0)
    return packed_f[0], packed_i[0], pm, d2


def _run_open_loop(sys_, h, p_heater, p_peak, state=None, max_iter=10):
    """Held-input open-loop run of a single-ring thermal system at step h (s).

    p_heater: heater power per step (W, (n,) or scalar). state: (thermal state x, T_prev, P_prev) to continue from.
    Returns (T_ring (n,) K, P_abs (n,) W, state, number of failed solves).
    """
    a, c0, c1 = sys_.coefficients(h)
    packed_f, packed_i, pm, d2 = _lorentzian_bank(p_peak)
    p_heater = np.atleast_1d(np.asarray(p_heater, float))
    n = p_heater.size
    u = np.zeros((n, sys_.n_src))
    u[:, sys_.heater_src[0]] = p_heater
    if state is None:
        state = (sys_.zero_state()[0], np.array([T0]), np.zeros(1))
    x, t_prev, p_prev = (v.copy() for v in state)
    t_ring_hist, p_abs_hist, t_obs_hist = np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, sys_.n_obs))
    n_fail = selfheat.implicit_run_b(x, a[0], c0[0], c1[0], *sys_.kernel_arrays(), np.zeros(1, np.int64), u,
                                     np.full(n, T0), np.ones(sys_.n_src), p_prev, t_prev,
                                     selfheat.pabs_lorentzian, selfheat.commit_none, packed_f, packed_i, pm, d2,
                                     max_iter, 1e-9, 1, t_ring_hist, p_abs_hist, t_obs_hist)
    # the ring observer equals the solved root
    np.testing.assert_allclose(t_obs_hist[:, sys_.ring_obs[0]], t_ring_hist[:, 0], atol=1e-8)
    return t_ring_hist[:, 0], p_abs_hist[:, 0], (x, t_prev, p_prev), n_fail


def _detuning_hwhm(t):
    """Detuning y = (lambda_r - lambda_L)/HWHM of the ring at temperature t (K)."""
    return (LAM_REF + S0 * (t - T0) - LAM_L) / HWHM


def _heater_power_for(y_o):
    """Heater power (W) that puts the cold (no self-heating) detuning at y_o (HWHM units)."""
    return ((np.asarray(y_o) * HWHM + LAM_L - LAM_REF) / S0) / Z0


def _settle_on(sys_, y_target, A, p_peak, upper=True):
    """Quasi-static approach (h = 1 s >> every tau) to the steady state at detuning y_target on the chosen branch."""
    y_o = y_target - A / (1 + y_target**2)
    path = np.linspace(4.0 if upper else -20.0, y_o, 400)
    tr, ph, state, nf = _run_open_loop(sys_, 1.0, _heater_power_for(path), p_peak)
    assert abs(_detuning_hwhm(tr[-1]) - y_target) < 1e-7
    return y_o, state


def _jump_index(y):
    """Index of the largest step in the sequence y (the branch jump of a sweep)."""
    return int(np.argmax(np.abs(np.diff(y)))) + 1


# ----------------------------------------------------------------------------------------------------- analytic helpers

def test_analytic_helpers():
    assert selfheat.lorentzian_folds(selfheat.BISTABILITY_FACTOR * 0.999) is None
    ya, yb, yoa, yob = selfheat.lorentzian_folds(A4)
    for y in (ya, yb):
        assert (1 + y * y) ** 2 + 2 * A4 * y == pytest.approx(0.0, abs=1e-9)
    assert ya < yb < 0 and yoa > yob
    assert 2.0 < A4 / selfheat.BISTABILITY_FACTOR < 8.0    # the design review: 3 to 7 times above threshold at +4 dBm
    assert selfheat.bistability_ratio(K_SH, P4, FWHM) == pytest.approx(A4 / selfheat.BISTABILITY_FACTOR)
    assert 0.5 * selfheat.BISTABILITY_FACTOR == pytest.approx(0.7698, abs=1e-4)   # "0.77 FWHM"
    r = selfheat.steady_detuning(-5.0, A4)
    assert r.size == 3 and np.allclose((r + 5.0) * (1 + r * r), A4)


def test_lorentzian_bounds_hold():
    packed_f, packed_i, pm, d2 = _lorentzian_bank(P4)
    t = np.linspace(T0 - 50, T0 + 150, 200001)
    p = np.array([selfheat.pabs_lorentzian(v, 0, packed_f, packed_i)[0] for v in t[::50]])
    assert p.max() <= pm[0] * (1 + 1e-12)
    pf = np.array([selfheat.pabs_lorentzian(v, 0, packed_f, packed_i) for v in t])
    d2num = np.gradient(pf[:, 1], t)
    assert np.max(np.abs(d2num)) <= d2[0] * 1.001
    assert np.max(np.abs(np.gradient(pf[:, 0], t) - pf[:, 1])) < 1e-6 * np.max(np.abs(pf[:, 1]))


# ----------------------------------------------------------------------------------------------------- the scalar solve

@njit
def _cubic_like(t, k, packed_f, packed_i):
    """Two Lorentzians (a doublet) to create up to five roots. packed_f = [centre_0 K, width_0 K, centre_1 K,
    width_1 K, peak W]; returns (P W, dP/dT W/K)."""
    p = 0.0
    dp = 0.0
    for l in range(2):
        u = (t - packed_f[2 * l]) / packed_f[2 * l + 1]
        den = 1.0 + u * u
        p += packed_f[4] / den
        dp -= packed_f[4] * 2.0 * u / (den * den) / packed_f[2 * l + 1]
    return p, dp


def test_solver_finds_the_first_root_in_the_motion_direction():
    """From any start, the certified solve returns the nearest root in the direction -sign F (branch continuation)."""
    packed_f = np.array([310.0, 1.2, 316.0, 0.8, 1e-3])     # centres K, widths K, peak W
    packed_i = np.zeros(1, np.int64)
    g, t_other = 15000.0, 305.0
    pmax = 2e-3
    d2 = 2 * 1e-3 / 0.8**2 + 2 * 1e-3 / 1.2**2
    grid = np.linspace(t_other, t_other + g * pmax, 400001)
    fgrid = grid - t_other - g * np.array([_cubic_like(v, 0, packed_f, packed_i)[0] for v in grid])
    roots = grid[:-1][np.sign(fgrid[:-1]) != np.sign(fgrid[1:])]
    assert roots.size >= 3
    rng = np.random.default_rng(5)
    for t_start in rng.uniform(t_other, t_other + g * pmax, 300):
        t, p, d, it, st = selfheat.solve_ring_temperature(_cubic_like, 0, packed_f, packed_i, t_other, g, t_start, pmax,
                                                          d2, 60, 1e-9)
        f0 = t_start - t_other - g * _cubic_like(t_start, 0, packed_f, packed_i)[0]
        expect = roots[roots > t_start].min() if f0 < 0 else roots[roots < t_start].max()
        assert st == 0 and abs(t - expect) < 1e-3, (t_start, t, expect)


def test_solver_converges_fast_near_a_previous_root():
    """Branch continuation from the previous root: a small change converges in a few evaluations to 1e-9 K."""
    packed_f, packed_i, pm, d2 = _lorentzian_bank(P4)
    rng = np.random.default_rng(6)
    worst = 0
    for _ in range(200):
        t_other = T0 + rng.uniform(20, 80)
        g = rng.uniform(1e3, 1e4)
        t_prev = t_other + rng.uniform(0, g * pm[0])
        t, p, d, it, st = selfheat.solve_ring_temperature(selfheat.pabs_lorentzian, 0, packed_f, packed_i, t_other, g,
                                                          t_prev, pm[0], d2[0], 10, 1e-9)
        if st == 0:
            assert abs(t - t_other - g * p) <= 1e-9
            # perturb the operating point a little: continuation converges within 4 evaluations
            t2, p2, d2_, it2, st2 = selfheat.solve_ring_temperature(selfheat.pabs_lorentzian, 0, packed_f, packed_i,
                                                                    t_other + 1e-3, g, t, pm[0], d2[0], 10, 1e-9)
            assert st2 == 0 and abs(t2 - t_other - 1e-3 - g * p2) <= 1e-9
            worst = max(worst, it2)
    assert worst <= 4


def test_solver_max_iter_returns_a_point_before_the_root():
    packed_f, packed_i, pm, d2 = _lorentzian_bank(P4)
    t_other, g = T0 + 39.0, G
    t, p, d, it, st = selfheat.solve_ring_temperature(selfheat.pabs_lorentzian, 0, packed_f, packed_i, t_other, g,
                                                      t_other, pm[0], d2[0], 1, 1e-9)
    assert st == 1 and it == 1
    f = t - t_other - g * p
    assert f < 0                                  # still on the start side: no overshoot past the root
    # g = 0: no self path
    t, p, d, it, st = selfheat.solve_ring_temperature(selfheat.pabs_lorentzian, 0, packed_f, packed_i, 310.0, 0.0, 0.0,
                                                      pm[0], d2[0], 10, 1e-9)
    assert t == 310.0 and st == 0 and it == 0


def test_nan_gain_is_a_failed_solve():
    """A NaN gain g (broken thermal coefficients) returns t_other with status 3, and implicit_run_b counts it as a
    failed solve (it was reported as converged). The thermal state still receives P_abs at t_other."""
    packed_f, packed_i, pm, d2 = _lorentzian_bank(P4)
    t, p, d, it, st = selfheat.solve_ring_temperature(selfheat.pabs_lorentzian, 0, packed_f, packed_i, 310.0, math.nan,
                                                      310.0, pm[0], d2[0], 10, 1e-9)
    assert t == 310.0 and it == 0 and st == selfheat.STATUS_NAN_GAIN
    assert p == selfheat.pabs_lorentzian(310.0, 0, packed_f, packed_i)[0]
    s = _one_ring_system()
    a, c0, c1 = s.coefficients(1e-6)
    arrays = s.kernel_arrays()
    lag_kind = arrays[2]
    c0 = c0.copy()
    c0[0, lag_kind == thermal.LAG_OPT_SELF] = math.nan                  # the self-heating gain g is then NaN
    u = np.zeros((1, s.n_src))
    tr, ph, to = np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, s.n_obs))
    nf = selfheat.implicit_run_b(s.zero_state()[0], a[0], c0[0], c1[0], *arrays, np.zeros(1, np.int64), u,
                                 np.full(1, T0), np.ones(s.n_src), np.zeros(1), np.array([T0]),
                                 selfheat.pabs_lorentzian, selfheat.commit_none, packed_f, packed_i, pm, d2, 10, 1e-9,
                                 1, tr, ph, to)
    assert nf == 1
    assert np.isfinite(ph[0, 0]) and ph[0, 0] == selfheat.pabs_lorentzian(tr[0, 0], 0, packed_f, packed_i)[0]


# ------------------------------------------------------------------------------ test (a): hysteresis edges at the folds

def test_a_hysteresis_edges_quasi_static():
    """+4 dBm: slow heater sweeps up and down jump at the analytic folds within 2 % of the HWHM; -10 dBm: none."""
    s = _one_ring_system()
    ya, yb, yoa, yob = selfheat.lorentzian_folds(A4)
    up = np.arange(yob - 1.0, yoa + 1.0, 1e-3)
    tr, ph, state, nf = _run_open_loop(s, 1.0, _heater_power_for(up), P4)
    y_up = _detuning_hwhm(tr)
    j = _jump_index(y_up)
    assert abs(up[j] - yoa) < 0.02
    assert y_up[j - 1] < 0 < y_up[j + 2]                  # lower branch -> upper branch
    down = up[::-1]
    tr, ph, state, nf = _run_open_loop(s, 1.0, _heater_power_for(down), P4, state)
    y_dn = _detuning_hwhm(tr)
    j = _jump_index(y_dn)
    assert abs(down[j] - yob) < 0.02
    # hysteresis: inside (yob, yoa) the two sweeps sit on different branches
    mid = (yoa + yob) / 2
    assert y_dn[np.argmin(np.abs(down - mid))] - y_up[np.argmin(np.abs(up - mid))] > 1.0
    # -10 dBm: below threshold, the up and down sweeps coincide
    assert selfheat.lorentzian_folds(AM10) is None
    lo = np.arange(-6.0, 6.0, 1e-2)
    tr1, _, st1, _ = _run_open_loop(s, 1.0, _heater_power_for(lo), PM10)
    tr2, _, _, _ = _run_open_loop(s, 1.0, _heater_power_for(lo[::-1]), PM10, st1)
    assert np.max(np.abs(_detuning_hwhm(tr1) - _detuning_hwhm(tr2)[::-1])) < 1e-6


def test_a_hysteresis_edges_dynamic_1us():
    """The same edges with the dynamic solve at dt = 1 us and a slow sweep (2e-5 HWHM per us)."""
    s = _one_ring_system(tail=0.0)
    ya, yb, yoa, yob = selfheat.lorentzian_folds(A4)
    rate = 2e-5
    tr, _, state, _ = _run_open_loop(s, 1.0, _heater_power_for(np.linspace(-20, yoa - 0.3, 200)), P4)
    up = yoa - 0.3 + rate * np.arange(int(0.6 / rate))
    tr, _, _, nf = _run_open_loop(s, 1e-6, _heater_power_for(up), P4, state)
    assert abs(up[_jump_index(_detuning_hwhm(tr))] - yoa) < 0.02
    tr, _, state, _ = _run_open_loop(s, 1.0, _heater_power_for(np.linspace(4.0, yob + 0.3, 200)), P4)
    dn = yob + 0.3 - rate * np.arange(int(0.6 / rate))
    tr, _, _, nf = _run_open_loop(s, 1e-6, _heater_power_for(dn), P4, state)
    assert abs(dn[_jump_index(_detuning_hwhm(tr))] - yob) < 0.02


# ----------------------------------------------------------- test (b): the implicit 1 us solve against a fine reference

@pytest.mark.parametrize("case", ["ramp_and_dither", "step_20pm", "step_100pm"])
def test_b_implicit_1us_against_10ns_reference(case):
    """On the stable branch at +4 dBm, dt = 1 us agrees with dt = 10 ns to better than 1 pm RMS (test b).

    Inputs are the same held heater staircase (1 us DAC updates) in both runs; cold-detuning units: an aggressor-like
    0.15 pm/us ramp (2000 C/s at ~75 pm/K) with a 2 pm square dither, or a step.
    """
    s = _one_ring_system()
    y0, state = _settle_on(s, 1 / math.sqrt(3), A4, P4)
    t = np.arange(400)
    if case == "ramp_and_dither":
        yo = y0 + 0.15e-12 / HWHM * np.minimum(t, 200) + np.where((t // 25) % 2 == 1, 2e-12 / HWHM, 0.0)
    else:
        step = {"step_20pm": 20e-12, "step_100pm": 100e-12}[case]
        yo = y0 + np.where(t >= 50, step / HWHM, 0.0)
    ph = _heater_power_for(yo)
    coarse, *_ = _run_open_loop(s, 1e-6, ph, P4, state)
    fine, *_ = _run_open_loop(s, 1e-8, np.repeat(ph, 100), P4, state)
    err_pm = S0 * (coarse - fine[99::100]) * 1e12
    assert np.sqrt(np.mean(err_pm**2)) < 1.0
    assert np.all(_detuning_hwhm(fine) > 0)                          # stayed on the stable (blue-laser) side


def test_b_steady_detuning_independent_of_dt_at_9dbm():
    """+9 dBm: the steady detuning changes by less than 1 % of the FWHM between dt and dt/8 (test b)."""
    s = _one_ring_system()
    y0, state = _settle_on(s, 1.0, A9, P9)
    ends = []
    for h in (1e-6, 1e-6 / 8):
        n = int(round(0.03 / h))
        tr, *_ = _run_open_loop(s, h, np.full(n, _heater_power_for(y0 + 0.05)), P9, state)
        ends.append(tr[-1])
    assert abs(S0 * (ends[0] - ends[1])) < 0.01 * FWHM


# ------------------------------------------------ test (c): growth on the unstable branch, no ringing on the stable one

def _single_pole_system(tau):
    """A one-ring thermal system with one pole tau (s): heater DC gain Z0, optical self-heating DC gain G (K/W)."""
    rom = thermal.ThermalRom(np.array([tau]), ("heater", "waveguide_absorption"), ("ring_mode_weighted",),
                             np.array([[[Z0]], [[G]]]), np.zeros(0), np.zeros((2, 1, 0, 1)))
    return thermal.assemble(rom, None, 1)


def test_c_unstable_point_diverges_at_linearized_rate():
    """Open-loop hold with the resonance blue of the laser (middle branch) diverges at (G P' - 1)/tau (test c)."""
    tau = 1e-6
    s = _single_pole_system(tau)
    ys = -0.8                                             # on the middle (unstable) branch: y_a < ys < y_b
    ya, yb, _, _ = selfheat.lorentzian_folds(A4)
    assert ya < ys < yb
    loop = -2 * A4 * ys / (1 + ys**2) ** 2                # G dP_abs/dlambda_r in lambda units
    rate = (loop - 1) / tau
    ts = T0 + (LAM_L + ys * HWHM - LAM_REF) / S0
    pstar = P4 / (1 + ys**2)
    ph = (ts - G * pstar - T0) / Z0
    u = np.zeros((1, s.n_src))
    u[0, s.heater_src[0]] = ph
    u[0, s.wg_src[0]] = pstar
    for h, n in ((2e-9, 2500), (1e-6, 40)):
        for eps in (1e-6, -1e-6):
            x = s.steady_state(u)[0]
            x[s.lag_src == s.wg_src[0]] += eps
            tr, *_ = _run_open_loop(s, h, np.full(n, ph), P4, (x, np.array([ts + eps]), np.array([pstar])))
            dev = tr - ts
            if h < 1e-7:
                tt = np.arange(1, n + 1) * h
                sel = (np.abs(dev) > 1e-5) & (np.abs(dev) < 1e-2)
                fit = np.polyfit(tt[sel], np.log(np.abs(dev[sel])), 1)[0]
                assert fit == pytest.approx(rate, rel=0.02)
                assert np.all(np.sign(dev[sel]) == np.sign(eps))
            else:
                # dt = 1 us >> 1/rate: it still leaves in the right direction, monotonically, to a stable branch
                dtr = np.diff(np.r_[ts + eps, tr]) * np.sign(eps)
                assert dtr[0] > 0 and np.all(dtr >= 0)
                y_end = _detuning_hwhm(tr[-1])
                assert (y_end > yb) if eps > 0 else (y_end < ya)


def _sign_changes(v):
    """Number of sign changes of the steps of v (a ringing count); steps below 1e-12 of the largest are ignored."""
    d = np.diff(v)
    d = d[np.abs(d) > 1e-12 * max(np.max(np.abs(d)), 1e-300)]
    return int(np.sum(np.sign(d[1:]) != np.sign(d[:-1])))


def test_c_no_oscillation_on_the_stable_branch():
    """dt = 1 us, +9 dBm, multi-pole interim plant: step responses on the stable branch never ring (test c), while
    the explicit (lagged) self coupling rings with period 2. That oscillation is why the design review chose the
    implicit solve: the lagged multiplier a - (1 - a) g is below -1 at dt = 1 us."""
    s = _one_ring_system()
    for yv in (0.05, 0.3, 0.577, 1.0, 2.0, 4.0):
        y0, state = _settle_on(s, yv, A9, P9)
        for dy in (0.02, -0.02):
            tr, *_ = _run_open_loop(s, 1e-6, np.full(300, _heater_power_for(y0 + dy)), P9, state)
            assert _sign_changes(tr) == 0, (yv, dy)
    # lower stable branch (resonance blue of the laser, before the up-jump)
    ya, yb, yoa, yob = selfheat.lorentzian_folds(A9)
    tr, _, state, _ = _run_open_loop(s, 1.0, _heater_power_for(np.linspace(-20.0, yoa - 0.5, 300)), P9)
    tr, *_ = _run_open_loop(s, 1e-6, np.full(300, _heater_power_for(yoa - 0.48)), P9, state)
    assert _sign_changes(tr) == 0
    # the explicit coupling (P_abs of the previous substep on the self path) oscillates at the same point
    y0, (x, tp, pp) = _settle_on(s, 0.577, A9, P9)
    a, c0, c1 = s.coefficients(1e-6)
    packed_f, packed_i, _, _ = _lorentzian_bank(P9)
    u = np.zeros(s.n_src)
    u[s.heater_src[0]] = _heater_power_for(y0 + 0.02)
    te, hist = tp[0], []
    for _ in range(40):
        u[s.wg_src[0]] = selfheat.pabs_lorentzian(te, 0, packed_f, packed_i)[0]
        thermal.thermal_step_b(x, a[0], c0[0], c1[0], s.lag_src, u, u, np.ones(s.n_src))
        te = T0 + x.sum()
        hist.append(te)
    assert _sign_changes(np.array(hist)) >= 3


# ------------------------------------------------------------------------------------------- test (d): laser-power step

def test_d_laser_power_step():
    """+10 % laser power with the heater held: dlambda_r = G (dP_abs/dP_in) dP_in / (1 - G dP_abs/dlambda_r).

    The simulated shift equals the exact root of the static cubic (1e-9 HWHM) at every operating point. The linear
    formula is first order in the step: within 5 % at the y = 1 lock point (test d), and its error shrinks in
    proportion to the step (at the steepest point y = 1/sqrt 3 a +10 % step is about 7 % off, +1 % about 0.8 %).
    """
    s = _one_ring_system()
    for yv, step, tol in ((1.0, 0.10, 0.05), (1 / math.sqrt(3), 0.01, 0.01), (1 / math.sqrt(3), 0.001, 0.001)):
        y0, state = _settle_on(s, yv, A4, P4)
        tr, *_ = _run_open_loop(s, 1.0, np.full(3, _heater_power_for(y0)), (1 + step) * P4, state)
        y1 = _detuning_hwhm(tr[-1])
        roots = selfheat.steady_detuning(y0, A4 * (1 + step))
        assert abs(y1 - roots[np.argmin(np.abs(roots - yv))]) < 1e-9
        p_abs = P4 / (1 + yv**2)
        dpdl = -P4 * 2 * yv / (1 + yv**2) ** 2 / HWHM
        pred = K_SH * (p_abs / P4) * (step * P4) / (1 - K_SH * dpdl)
        assert (y1 - yv) * HWHM == pytest.approx(pred, rel=tol)
    # and dynamically: the same final shift after a 30 ms hold at dt = 1 us (tail 4 ms)
    y0, state = _settle_on(s, 1.0, A4, P4)
    tr, *_ = _run_open_loop(s, 1e-6, np.full(30000, _heater_power_for(y0)), 1.1 * P4, state)
    roots = selfheat.steady_detuning(y0, A4 * 1.1)
    assert abs(_detuning_hwhm(tr[-1]) - roots[np.argmin(np.abs(roots - 1.0))]) < 1e-3


# ------------------------------------------------------------------------------------------------------- bank semantics

@njit
def _pabs_record(t, k, packed_f, packed_i):
    """pabs_lorentzian, used with _commit_record."""
    return selfheat.pabs_lorentzian(t, k, packed_f, packed_i)


@njit
def _commit_record(t, k, packed_f, packed_i):
    """A commit function that records the solve order: ring k goes into the next slot of packed_f[-9 .. -2], and
    packed_f[-1] counts the commits."""
    n = int(packed_f[-1])
    packed_f[packed_f.shape[0] - 9 + n] = k                           # order slots: packed_f[-9 .. -2]
    packed_f[-1] = n + 1


def test_bank_bus_order_lag_kinds_and_consistency():
    """Rings are solved in bus order; cross-ring optical heating uses the previous P_abs, other observers the new."""
    n = 3
    sys_ = thermal.assemble(thermal.InterimRom(xt_nearest_frac=0.08, xt_floor_frac=0.04, xt_tau_min_s=2e-6,
                                               xt_tau_ref_s=2e-6),
                            thermal.LayoutSites.line(n, 30.0), n)
    lam_ref = np.array([LAM_L - 0.05e-9, LAM_L + 0.1e-9, LAM_L - 0.2e-9])
    packed_f, packed_i = selfheat.pack_lorentzian_args(lam_ref[None], T0, S0, 0.0, np.full((1, n, 1), LAM_L),
                                                       np.full((1, n, 1), HWHM), np.full((1, n, 1), P4))
    packed_f = np.concatenate([packed_f[0], np.zeros(9)])           # 8 order slots + a counter
    pm, _, d2 = selfheat.lorentzian_bounds(np.full((n, 1), P4), np.full((n, 1), HWHM), S0)
    a, c0, c1 = sys_.coefficients(1e-6)
    x = sys_.zero_state()[0]
    rng = np.random.default_rng(9)
    x[:] = rng.uniform(0, 2.0, sys_.n_lags)
    u = np.zeros(sys_.n_src)
    u[sys_.heater_src] = [3e-3, 1e-3, 2e-3]
    p_prev = np.array([1e-3, 0.5e-3, 1.5e-3])
    t_prev = np.full(n, T0 + 5.0)
    bus = np.array([2, 0, 1], np.int64)
    g, t_obs = np.zeros(sys_.n_obs), np.zeros(sys_.n_obs)
    p_new, t_ring, n_it = np.zeros(n), np.zeros(n), np.zeros(n, np.int64)
    x_before = x.copy()
    nf = selfheat.implicit_step_b(x, a[0], c0[0], c1[0], *sys_.kernel_arrays(), bus, u, u, np.ones(sys_.n_src), T0,
                                  p_prev, t_prev, _pabs_record, _commit_record, packed_f, packed_i[0], pm, d2, 10, 1e-9,
                                  1, g, t_obs, p_new, t_ring, n_it)
    assert nf == 0
    assert list(packed_f[-9:-9 + n]) == [2.0, 0.0, 1.0] and packed_f[-1] == n
    np.testing.assert_allclose(t_obs[sys_.ring_obs], t_ring, atol=1e-8)
    # reference: rebuild every observer by hand with the documented input rule
    ref = np.full(sys_.n_obs, T0)
    for m in range(sys_.n_lags):
        s = sys_.lag_src[m]
        kd = sys_.lag_kind[m]
        if kd == thermal.LAG_EXOGENOUS:
            drv = (c0[0, m] + c1[0, m]) * u[s]
        elif kd == thermal.LAG_OPT_CROSS_RING:
            drv = (c0[0, m] + c1[0, m]) * p_prev[sys_.src_ring[s]]
        else:
            drv = (c0[0, m] + c1[0, m]) * p_new[sys_.src_ring[s]]
        ref[sys_.lag_obs[m]] += a[0, m] * x_before[m] + drv
    np.testing.assert_allclose(t_obs, ref, rtol=1e-13)
    for k in range(n):
        assert p_new[k] == pytest.approx(selfheat.pabs_lorentzian(t_ring[k], k, packed_f, packed_i[0])[0], rel=1e-12)


def test_selfheatwork_batched_matches_rows():
    """The batched implicit_step equals per-world runs, and the cross-lag gain stays below 0.05 at 1 us."""
    n = 4
    z = np.array([5.5e3, 6.1e3, 7.0e3])
    sys_ = thermal.assemble(thermal.InterimRom(z_heater_ring_k_per_w=z), thermal.LayoutSites.line(n, 25.0), n,
                            thermal.ThermalScales(time_scale_fast=[0.7, 1.0, 1.5]))
    lam_ref = LAM_L + np.array([-0.1, 0.05, 0.0, 0.12]) * 1e-9
    packed_f, packed_i = selfheat.pack_lorentzian_args(lam_ref[None], T0, S0, 1.4e-3, np.full((1, n, 1), LAM_L),
                                                       np.full((1, n, 1), HWHM), np.full((1, n, 1), P4))
    pm, d1, d2 = selfheat.lorentzian_bounds(np.full((n, 1), P4), np.full((n, 1), HWHM), S0, 1.4e-3)
    assert thermal.assert_cross_lag_gain(sys_, 1e-6, d1[None]) < 0.05
    w = selfheat.SelfHeatWork(sys_, t_init_k=T0)
    u = np.zeros((1, sys_.n_src))
    u[0, sys_.heater_src] = 2e-3
    for _ in range(50):
        assert w.step(1e-6, u, u, T0, selfheat.pabs_lorentzian, packed_f, packed_i, pm, d2) == 0
    for b in range(3):
        one = thermal.assemble(thermal.InterimRom(z_heater_ring_k_per_w=z[b]), thermal.LayoutSites.line(n, 25.0), n,
                               thermal.ThermalScales(time_scale_fast=[0.7, 1.0, 1.5][b]))
        w1 = selfheat.SelfHeatWork(one, t_init_k=T0)
        for _ in range(50):
            w1.step(1e-6, u, u, T0, selfheat.pabs_lorentzian, packed_f, packed_i, pm, d2)
        np.testing.assert_allclose(w.t_ring[b], w1.t_ring[0], rtol=1e-12)
        np.testing.assert_allclose(w.x[b], w1.x[0], rtol=1e-12, atol=1e-12)


_NOJIT_SCRIPT = r"""
import json, numpy as np
from mrm.plant import selfheat, thermal
system = thermal.assemble(thermal.InterimRom(), thermal.LayoutSites.line(2, 30.0), 2)
packed_f, packed_i = selfheat.pack_lorentzian_args([[1309.9e-9, 1310.1e-9]], 300.0, 70e-12, 0.0,
                                                   [[[1310e-9], [1310e-9]]], [[[0.17e-9], [0.17e-9]]],
                                                   [[[2e-3], [2e-3]]])
pm, d1, d2 = selfheat.lorentzian_bounds(np.full((2, 1), 2e-3), np.full((2, 1), 0.17e-9), 70e-12)
w = selfheat.SelfHeatWork(system, t_init_k=300.0)
u = np.zeros((1, system.n_src)); u[0, system.heater_src] = [1e-3, 4e-3]
for i in range(20):
    w.step(1e-6, u, u, 300.0, selfheat.pabs_lorentzian, packed_f, packed_i, pm, d2)
out = thermal.simulate(system, np.ones((5, 1, system.n_src)) * 1e-3, 2e-6, hold="foh")
print(json.dumps({"t": w.t_ring.tolist(), "x": float(w.x.sum()), "sim": out[-1].tolist()}))
"""


def test_kernels_run_without_jit():
    """NUMBA_DISABLE_JIT=1 gives the same numbers (the kernels are plain Python there)."""
    env = dict(os.environ, NUMBA_DISABLE_JIT="1")
    res = [json.loads(subprocess.run([sys.executable, "-c", _NOJIT_SCRIPT], capture_output=True, text=True,
                                     check=True, env=e).stdout) for e in (env, dict(os.environ))]
    np.testing.assert_allclose(res[0]["t"], res[1]["t"], rtol=1e-12)
    assert res[0]["x"] == pytest.approx(res[1]["x"], rel=1e-12)
    np.testing.assert_allclose(res[0]["sim"], res[1]["sim"], rtol=1e-12)


def test_reset_branch_selection():
    """t_prev = -inf / +inf select the cold / hot branch in the bistable region (the branch_low / branch_high reset
    protocols)."""
    s = _one_ring_system()
    ya, yb, yoa, yob = selfheat.lorentzian_folds(A4)
    y_o = 0.5 * (yoa + yob)
    roots = selfheat.steady_detuning(y_o, A4)
    packed_f, packed_i, pm, d2 = _lorentzian_bank(P4)
    t_other = T0 + Z0 * _heater_power_for(y_o)
    for start, expect in ((-np.inf, roots[0]), (np.inf, roots[-1])):
        t, p, d, it, st = selfheat.solve_ring_temperature(selfheat.pabs_lorentzian, 0, packed_f, packed_i, t_other, G,
                                                          start, pm[0], d2[0], 40, 1e-9)
        assert st == 0 and _detuning_hwhm(t) == pytest.approx(expect, abs=1e-7)


def test_gauss_seidel_sweeps_make_the_bank_implicit_for_long_steps():
    """Multi-tick steps: with h = 100 us at a 20 um pitch the lagged cross-ring gain exceeds 0.05; n_outer
    sweeps solve the coupled bank, so one quasi-static step satisfies the static bank equations exactly."""
    n = 4
    sys_ = thermal.assemble(thermal.InterimRom(), thermal.LayoutSites.line(n, 20.0), n,
                            observers=("ring_mode_weighted", "heater_resistor"))
    lam_ref = LAM_L + np.array([0.05, -0.02, 0.08, 0.0]) * 1e-9
    packed_f, packed_i = selfheat.pack_lorentzian_args(lam_ref[None], T0, S0, 0.0, np.full((1, n, 1), LAM_L),
                                                       np.full((1, n, 1), HWHM), np.full((1, n, 1), P9))
    pm, d1, d2 = selfheat.lorentzian_bounds(np.full((n, 1), P9), np.full((n, 1), HWHM), S0)
    with pytest.raises(ValueError):
        thermal.assert_cross_lag_gain(sys_, 100e-6, d1)
    thermal.assert_cross_lag_gain(sys_, 1e-6, d1)
    u = np.zeros((1, sys_.n_src))
    u[0, sys_.heater_src] = [1e-3, 2e-3, 0.5e-3, 1.5e-3]
    G0 = thermal.dc_gain(sys_)[0]
    ring = sys_.ring_obs
    for n_outer, tol in ((1, None), (30, 1e-8)):
        w = selfheat.SelfHeatWork(sys_, t_init_k=np.inf)                 # hot branch
        w.step(10.0, u, u, T0, selfheat.pabs_lorentzian, packed_f, packed_i, pm, d2, n_outer=n_outer, max_iter=60)
        p = np.array([selfheat.pabs_lorentzian(w.t_ring[0, k], k, packed_f[0], packed_i[0])[0] for k in range(n)])
        uu = u[0].copy()
        uu[sys_.wg_src] = p
        resid = np.max(np.abs(T0 + G0[ring] @ uu - w.t_ring[0]))
        if tol is None:
            assert resid > 1e-3                                      # one lagged step is not the bank solution
        else:
            assert resid < tol


@njit
def _pabs_bad_derivative(t, k, packed_f, packed_i):
    p, d = selfheat.pabs_lorentzian(t, k, packed_f, packed_i)
    return p, 0.3 * d                       # a caller with an inaccurate derivative


def test_solver_robust_to_inexact_derivative_and_bound():
    """An inaccurate dP/dT or an underestimated d2p_max must not stall the solve: it falls back to the bracket."""
    packed_f, packed_i, pm, d2 = _lorentzian_bank(P4)
    rng = np.random.default_rng(11)
    for _ in range(100):
        t_other = T0 + rng.uniform(30, 50)
        t_start = t_other + rng.uniform(0, G * pm[0])
        for fn, bound in ((_pabs_bad_derivative, d2[0]), (selfheat.pabs_lorentzian, 0.05 * d2[0])):
            t, p, d, it, st = selfheat.solve_ring_temperature(fn, 0, packed_f, packed_i, t_other, G, t_start, pm[0],
                                                              bound, 40, 1e-9)
            assert st == 0 and abs(t - t_other - G * p) <= 1e-8     # |F| <= |F'| x the 1e-9 K bracket


# ----------------------------------------------------------------------------------------------------- review additions

def test_solver_flags_a_p_max_that_does_not_bound_p_abs():
    """A p_max below the true peak must not be reported as converged (status 2), and the bank step must count it and
    keep the ring observer consistent with the returned point."""
    packed_f, packed_i, pm, d2 = _lorentzian_bank(P4)
    t_other = T0 + Z0 * _heater_power_for(1.0)          # cold detuning y_o = 1: the root needs P_abs ~ p_peak/2.6
    for pmax, status in ((pm[0], 0), (0.05 * pm[0], 2)):
        t, p, d, it, st = selfheat.solve_ring_temperature(selfheat.pabs_lorentzian, 0, packed_f, packed_i, t_other, G,
                                                          -np.inf, pmax, d2[0], 10, 1e-9)
        assert st == status
        if st == 0:
            assert abs(t - t_other - G * p) <= 1e-9
    s = _one_ring_system()
    a, c0, c1 = s.coefficients(1.0)
    x = s.zero_state()[0]
    u = np.zeros(s.n_src)
    u[s.heater_src[0]] = _heater_power_for(1.0)
    g, t_obs = np.zeros(s.n_obs), np.zeros(s.n_obs)
    p_new, t_ring, n_it = np.zeros(1), np.zeros(1), np.zeros(1, np.int64)
    nf = selfheat.implicit_step_b(x, a[0], c0[0], c1[0], *s.kernel_arrays(), np.zeros(1, np.int64), u, u,
                                  np.ones(s.n_src),
                                  T0, np.zeros(1), np.full(1, T0), selfheat.pabs_lorentzian, selfheat.commit_none,
                                  packed_f, packed_i, 0.05 * pm, d2, 10, 1e-9, 1, g, t_obs, p_new, t_ring, n_it)
    assert nf == 1
    assert t_obs[s.ring_obs[0]] == pytest.approx(t_ring[0], abs=1e-9)
    assert p_new[0] == pytest.approx(0.05 * pm[0], rel=1e-9)        # the state got the bound, not P(T)


def test_solver_negative_g_uses_the_mirrored_bracket():
    """A mixed-sign Foster fit could give g < 0 over a short step: the root of T - t_other - g P(T) is still found."""
    packed_f, packed_i, pm, d2 = _lorentzian_bank(P4)
    for g in (-2.0e3, 3.0e3):
        t_other = T0 + Z0 * _heater_power_for(0.3)
        t, p, d, it, st = selfheat.solve_ring_temperature(selfheat.pabs_lorentzian, 0, packed_f, packed_i, t_other, g,
                                                          t_other, pm[0], d2[0], 20, 1e-9)
        assert st == 0 and abs(t - t_other - g * p) <= 1e-9


def test_pack_lorentzian_args_batch_rows():
    """(N, L) per-line arrays describe one world: one row of args, not N (np.atleast_3d would make it N)."""
    n = 3
    packed_f, packed_i = selfheat.pack_lorentzian_args(np.full(n, LAM_REF), T0, S0, 0.0, np.full((n, 1), LAM_L),
                                                       np.full((n, 1), HWHM), np.full((n, 1), P4))
    assert packed_f.shape[0] == 1 and packed_i.shape == (1, 1 + n)
    af3, _ = selfheat.pack_lorentzian_args(np.full((2, n), LAM_REF), T0, S0, 0.0, np.full((n, 1), LAM_L),
                                           np.full((n, 1), HWHM), np.full((n, 1), P4))
    assert af3.shape[0] == 2
    for k in range(n):
        assert selfheat.pabs_lorentzian(T0, k, packed_f[0],
                                        packed_i[0])[0] == pytest.approx(P4 / (1 + ((LAM_REF - LAM_L) / HWHM) ** 2))


def test_implicit_step_without_light_equals_the_foh_kernel():
    """With no absorbed power the implicit step is the exact FOH thermal step (exact for piecewise-linear inputs)."""
    n = 3
    sys_ = thermal.assemble(thermal.InterimRom(), thermal.LayoutSites.line(n, 30.0, aggressor_xy_um=[[30.0, 40.0]]), n)
    packed_f, packed_i = selfheat.pack_lorentzian_args(np.full(n, LAM_REF), T0, S0, 0.0, np.full((n, 1), LAM_L),
                                                       np.full((n, 1), HWHM), np.zeros((n, 1)))
    pm, _, d2 = selfheat.lorentzian_bounds(np.zeros((n, 1)), np.full((n, 1), HWHM), S0)
    rng = np.random.default_rng(4)
    a, c0, c1 = sys_.coefficients(3.7e-6)
    x1 = rng.uniform(0, 1, sys_.n_lags)
    x2 = x1.copy()
    u0, u1 = rng.uniform(0, 5e-3, sys_.n_src), rng.uniform(0, 5e-3, sys_.n_src)
    u0[sys_.wg_src] = u1[sys_.wg_src] = 0.0
    ss = np.ones(sys_.n_src)
    out = np.zeros(sys_.n_obs)
    thermal.thermal_advance_b(x1, a[0], c0[0], c1[0], sys_.lag_src, sys_.obs_ptr, u0, u1, ss, T0, out)
    g, t_obs, p_new, t_ring, n_it = (np.zeros(sys_.n_obs), np.zeros(sys_.n_obs), np.zeros(n), np.zeros(n),
                                     np.zeros(n, np.int64))
    selfheat.implicit_step_b(x2, a[0], c0[0], c1[0], *sys_.kernel_arrays(), np.arange(n), u0, u1, ss, T0, np.zeros(n),
                             np.full(n, T0), selfheat.pabs_lorentzian, selfheat.commit_none, packed_f[0], packed_i[0],
                             pm, d2, 10, 1e-9, 1, g, t_obs, p_new, t_ring, n_it)
    np.testing.assert_allclose(x2, x1, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(t_obs, out, rtol=1e-14)


def test_neighbour_lock_steps_the_victim_by_the_cross_opt_residue():
    """Optical crosstalk between rings: switching the light on at ring 0 (it absorbs P_abs,0) steps ring 1 (dark) by
    Z_cross_opt(1<-0) P_abs,0 at steady state; on the way, ring 1 follows the linear ROM driven by ring 0's P_abs one
    substep late."""
    n = 2
    sys_ = thermal.assemble(thermal.InterimRom(), thermal.LayoutSites.line(n, 30.0), n,
                            sources=("heater", "waveguide_absorption"), observers=("ring_mode_weighted",))
    lam_ref = np.array([LAM_L + 0.5 * HWHM - S0 * Z0 * 2e-3, LAM_L - 5e-9])   # ring 0 red of its line once heated
    p_on = np.array([[P4], [0.0]])
    packed_f, packed_i = selfheat.pack_lorentzian_args(lam_ref, T0, S0, 0.0, np.full((n, 1), LAM_L),
                                                       np.full((n, 1), HWHM), p_on)
    pm, _, d2 = selfheat.lorentzian_bounds(p_on, np.full((n, 1), HWHM), S0)
    u = np.zeros((1, sys_.n_src))
    u[0, sys_.heater_src] = [2e-3, 1e-3]
    G0 = thermal.dc_gain(sys_)[0]
    r1, wg0 = sys_.ring_obs[1], sys_.wg_src[0]
    t_dark = T0 + G0[r1] @ u[0]                    # ring 1 before ring 0 absorbs anything (steady, lights off)
    w = selfheat.SelfHeatWork(sys_, t_init_k=T0)
    w.x[:] = sys_.steady_state(u)
    w.t_prev[:] = T0 + G0[sys_.ring_obs] @ u[0]
    h, nstep = 1e-6, 3000
    hist_p0, hist_t1 = np.zeros(nstep), np.zeros(nstep)
    for i in range(nstep):
        assert w.step(h, u, u, T0, selfheat.pabs_lorentzian, packed_f, packed_i, pm, d2) == 0
        hist_p0[i], hist_t1[i] = w.p_new[0, 0], w.t_ring[0, 1]
    # dynamic: ring 1 = linear model with wg0 driven by P_abs,0 of the previous substep (LAG_OPT_CROSS_RING rule)
    ul = np.zeros((nstep + 1, 1, sys_.n_src))          # step k holds ul[k]
    ul[:, 0, sys_.heater_src] = u[0, sys_.heater_src]
    ul[1:nstep, 0, wg0] = hist_p0[:-1]
    lin = thermal.simulate(sys_, ul, h, hold="zoh", x0=sys_.steady_state(u), t_base_k=T0)
    np.testing.assert_allclose(hist_t1, lin[1:, 0, r1], rtol=0, atol=1e-10)
    assert hist_t1[-1] - t_dark > 0.3 * G0[r1, wg0] * hist_p0[-1]   # the crosstalk is arriving (tau ~ 1 ms)
    # steady state: long implicit steps (Gauss-Seidel sweeps make the bank coupling implicit)
    for _ in range(80):
        w.step(1e-3, u, u, T0, selfheat.pabs_lorentzian, packed_f, packed_i, pm, d2, n_outer=5)
    p0 = w.p_new[0, 0]
    assert p0 > 0.1 * P4                                    # self-locked on the red side, y ~ 2.2
    assert w.t_ring[0, 1] - t_dark == pytest.approx(G0[r1, wg0] * p0, rel=1e-6)
    assert G0[r1, wg0] == pytest.approx(0.03 * Z0)          # interim nearest-neighbour fraction of Z0


def test_implicit_step_rejects_a_bad_bus_order_before_touching_the_state():
    n = 3
    sys_ = thermal.assemble(thermal.InterimRom(), thermal.LayoutSites.line(n, 30.0), n)
    packed_f, packed_i = selfheat.pack_lorentzian_args(np.full(n, LAM_REF), T0, S0, 0.0, np.full((n, 1), LAM_L),
                                                       np.full((n, 1), HWHM), np.full((n, 1), P4))
    pm, _, d2 = selfheat.lorentzian_bounds(np.full((n, 1), P4), np.full((n, 1), HWHM), S0)
    a, c0, c1 = sys_.coefficients(1e-6)
    x = np.ones(sys_.n_lags)
    u = np.zeros(sys_.n_src)
    for bus in (np.array([0, 1], np.int64), np.array([0, 0, 2], np.int64), np.array([0, 1, 3], np.int64)):
        with pytest.raises(ValueError, match="bus_order"):
            selfheat.implicit_step_b(x, a[0], c0[0], c1[0], *sys_.kernel_arrays(), bus, u, u, np.ones(sys_.n_src), T0,
                                     np.zeros(n), np.full(n, T0), selfheat.pabs_lorentzian, selfheat.commit_none,
                                     packed_f[0], packed_i[0], pm, d2,
                                     10, 1e-9, 1, np.zeros(sys_.n_obs), np.zeros(sys_.n_obs), np.zeros(n), np.zeros(n),
                                     np.zeros(n, np.int64))
        assert np.all(x == 1.0)
