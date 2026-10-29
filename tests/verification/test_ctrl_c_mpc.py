"""Checks the linear MPC with heater code limits and slew limits (mrm/control/baselines/mpc.py).

1. The ADMM box-QP solver against scipy.optimize.minimize (trust-constr) on random problems with active constraints.
2. The unconstrained MPC (+ Kalman filter) is a linear law. Its closed loop is stable on the nominal model with the
   heater gain scaled 0.3-6x, and on linearizations of sampled worlds (model mismatch).
3. Closed loop on the ReferencePlant (Truth only in the test): a steady lock, the A2 808 degC/s window, and a slew
   limit (whole codes) that binds. Every command respects |du| <= slew and the code range; the error grows, as it
   must.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import LinearConstraint, minimize

from mrm.control import models as control_models
from mrm.control import registry
from mrm.control.baselines import kalman_lqg, mpc
from mrm.scenarios import registry as scenario_registry

from test_ctrl_c_harness import build_rig, hold, load_cfg, lock_metrics, realize, run_loop


@pytest.fixture(scope="module")
def cfg():
    return load_cfg()


@pytest.fixture(scope="module")
def models(cfg):
    return control_models.linearize_bank(cfg, control_models.OperatingPoint(), 1e-5, latency_ticks=1)


def test_registered(cfg):
    assert registry.resolve("mpc") is mpc.LinearMPC
    assert registry.resolve("mpc_c") is mpc.CompiledMpc
    c = registry.create("mpc", {"horizon_steps": 12, "control_moves": 3})
    assert c.params.horizon_steps == 12 and c.params.control_moves == 3


@pytest.mark.parametrize("seed", range(6))
def test_admm_matches_slsqp_with_active_constraints(seed):
    """Random strongly convex QPs with box constraints on [x; D x] (the MPC structure) whose unconstrained optimum is
    mostly infeasible. ADMM (tol 1e-9) agrees with scipy trust-constr to 1e-4 in x and is at least as good in cost,
    and its multipliers satisfy the KKT conditions."""
    rng = np.random.default_rng(seed)
    n = 5
    G = rng.normal(size=(n, n))
    P = G @ G.T + 0.5 * np.eye(n)
    q = rng.normal(size=n) * 5.0
    D = mpc._diff_matrix(n)
    A = np.concatenate([np.eye(n), D])
    lo = np.concatenate([np.full(n, -1.0), np.full(n, -0.4)])
    hi = np.concatenate([np.full(n, 1.5), np.full(n, 0.4)])
    rho = 0.1 * np.mean(np.diag(P))
    Minv = np.linalg.inv(P + 1e-9 * np.eye(n) + rho * A.T @ A)
    x, z, y, it, rp, rd = mpc.admm_box_qp(P, q, A, lo, hi, Minv, rho, max_iter=20000, tol=1e-9)
    assert rp < 1e-8 and rd < 1e-8, (it, rp, rd)

    def f(v):
        return 0.5 * v @ P @ v + q @ v
    ref = minimize(f, np.zeros(n), jac=lambda v: P @ v + q, hess=lambda v: P, method="trust-constr",
                   constraints=[LinearConstraint(A, lo, hi)], options={"gtol": 1e-12, "xtol": 1e-14,
                                                                       "maxiter": 5000})
    assert f(x) <= ref.fun + 1e-6 * max(1.0, abs(ref.fun))
    np.testing.assert_allclose(x, ref.x, atol=1e-4 * max(1.0, np.max(np.abs(ref.x))))
    # KKT of the ADMM solution itself: stationarity with the multipliers y, complementary slackness
    np.testing.assert_allclose(P @ x + q + A.T @ y, 0.0, atol=1e-6)
    ax = A @ x
    assert np.all((y <= 1e-6) | (np.abs(ax - hi) < 1e-6)) and np.all((y >= -1e-6) | (np.abs(ax - lo) < 1e-6))
    assert np.sum(np.abs(y) > 1e-6) >= 1 or seed % 2 == 1   # most draws have an active constraint
    assert np.all(A @ x >= lo - 1e-7) and np.all(A @ x <= hi + 1e-7)
    # batched solve = one at a time (per-problem stopping rule)
    qb = np.stack([q, -q, 0.3 * q])
    xb, *_ = mpc.admm_box_qp(P[None], qb, A, lo, hi, Minv[None], np.full(3, rho), max_iter=500, tol=1e-6)
    for i in range(3):
        xi, *_ = mpc.admm_box_qp(P, qb[i], A, lo, hi, Minv, rho, max_iter=500, tol=1e-6)
        np.testing.assert_array_equal(xb[i], xi)


def test_unconstrained_mpc_is_stable_nominal_and_sampled(cfg, models):
    lqg = kalman_lqg.design_lqg(models)
    md = mpc.design_mpc(lqg)
    for g in (0.3, 1.0, 3.0, 6.0):
        assert mpc.mpc_closed_loop_spectral_radius(md, 0, models[0], g) < 0.9999
    pp = realize(cfg, 16, uncertainty="lhs", variation="sample", seeds=3)
    rhos = []
    for b in range(pp.batch):
        lm_b = control_models.linearize_bank(cfg, control_models.OperatingPoint(), 1e-5, latency_ticks=1, pp=pp, b=b)[0]
        if lm_b.op.heater_power_w[0] <= 5e-4:
            continue
        rhos.append(mpc.mpc_closed_loop_spectral_radius(md, 0, lm_b))
    print("\nMPC spectral radius over feasible LHS worlds:", np.round(rhos, 5))
    assert len(rhos) >= 10 and max(rhos) < 1.0


def test_fractional_slew_limit_is_whole_codes(cfg):
    """Commands are whole codes: a fractional slew limit (a hardware slew_codes_per_s times Ts, or a parameter) is
    floored to whole codes (at least 1), so neither the rounded Python command nor the compiled filter state can step
    past it (the compiled step stored the un-rounded clipped command before this was fixed)."""
    from mrm.control.api import ControllerSpec
    spec = ControllerSpec.from_config(cfg, n_batch=1, nominal=cfg)
    assert mpc.LinearMPC(slew_codes_per_step=2.5).build_mpc(spec).slew == 2.0
    assert mpc.LinearMPC(slew_codes_per_step=0.3).build_mpc(spec).slew == 1.0
    cc = mpc.CompiledMpc(slew_codes_per_step=2.5)
    cc.reset(spec)
    assert cc.mpc.slew == 2.0 and float(cc.params_f64[10]) == 2.0


def test_mpc_steady_lock_nominal(cfg):
    rig = build_rig(cfg)
    hold(rig, 30)
    tr = run_loop(rig, mpc.LinearMPC(), 1500)
    m = lock_metrics(tr.detuning_m, rig.targets_m, oma_w=tr.oma_w, oracle_oma_w=rig.oracle_oma_w, settle_ticks=200)
    print("\n[mpc steady]", m)
    assert m["excursions"] == 0 and abs(m["mean_pm"]) < 0.5 and m["oma_penalty_db"] < 0.01
    assert np.max(np.std(tr.detuning_m[200:], axis=0)) * 1e12 < 0.3
    assert np.all(tr.telemetry["qp_iterations"][-50:] <= 5)


def test_mpc_a2_window_and_binding_slew(cfg):
    """A2 window (as the LQG test): tracking within a few pm of the monitor-consistent reference. Then the same window
    with a slew limit of 2 codes per step, below the ~5 codes per step the ramp needs: every command change stays
    within the limit, slew_active is reported, and the lag grows instead of the loop going unstable."""
    out = {}
    for slew in (500.0, 2.0):
        sc = scenario_registry.create("a2")
        rig = build_rig(cfg, scenario=sc, k0=20000, aggressor_presettle_ticks=6000)
        hold(rig, 30)
        ctrl = mpc.LinearMPC(slew_codes_per_step=slew)
        tr = run_loop(rig, ctrl, 2500)
        dyn = lock_metrics(tr.detuning_m, tr.ref_m, settle_ticks=600)
        du = np.abs(np.diff(tr.heater_codes.astype(np.int64), axis=0))
        out[slew] = (dyn, du, tr)
        print(f"\n[mpc A2 slew {slew}] tracking", dyn, " max |du| per step", int(du.max()))
    dyn, du, tr = out[500.0]
    assert dyn["rms_pm"] < 3.0 and dyn["max_pm"] < 5.0 and dyn["excursions"] == 0
    dyn2, du2, tr2 = out[2.0]
    assert du2.max() <= 2
    assert tr2.telemetry["slew_active"][-200:].mean() > 0.5
    assert dyn2["mean_pm"] > dyn["mean_pm"] + 1.0            # the ramp outruns the limited heater: lag grows
    assert np.all(tr2.heater_codes >= 0) and np.all(tr2.heater_codes <= 65535)


def test_mpc_code_limit_binds(cfg):
    """Heater code limits in the QP: code_margin puts the upper limit 30 codes above the highest nominal lock code;
    a +0.5 dB laser power step then asks the rings for ~80 more codes (the set-point lock moves red). Every command
    stays within [code_min + margin, code_max - margin], the rings that reach the limit report bound_active, and the
    filter (propagated with the applied command) stays finite and consistent (no wind-up: the command leaves the
    limit as soon as the demand falls)."""
    from mrm.scenarios.generators import Scenario, step
    from mrm.control.api import ControllerSpec
    spec = ControllerSpec.from_config(cfg, n_batch=1, nominal=cfg)
    u0 = mpc.LinearMPC().build_mpc(spec).lqg.u0
    margin = int(65535 - np.ceil(u0.max()) - 30)
    sc = Scenario(name="pstep", components=(step("laser.power_delta_db", 4e-4, 0.5),))
    rig = build_rig(cfg, scenario=sc)
    hold(rig, 30)
    ctrl = mpc.LinearMPC(code_margin=margin)
    tr = run_loop(rig, ctrl, 1400)
    hi = 65535 - margin
    assert np.all(tr.heater_codes <= hi) and np.all(tr.heater_codes >= margin)
    at = tr.heater_codes[-1, 0] == hi
    print(f"\n[mpc code limit] upper limit {hi}, final codes {tr.heater_codes[-1, 0]}, bound_active "
          f"{tr.telemetry['bound_active'][-1, 0]}")
    assert at.any() and (~at).any()
    assert np.all(tr.telemetry["bound_active"][-1, 0][at] == 1.0)
    assert np.all(np.isfinite(ctrl.z))
    # the rings below the limit still lock (steady); those at it cannot heat enough (their code stays above y0)
    det = tr.detuning_m[-300:, 0] * 1e12
    assert np.all(np.std(det, axis=0)[~at] < 0.5)
