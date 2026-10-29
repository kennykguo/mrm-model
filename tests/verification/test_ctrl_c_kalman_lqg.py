"""Checks the Kalman + LQG heater lock (mrm/control/baselines/kalman_lqg.py), closed loop on the ReferencePlant.

Read this first
---------------
The controller sees ControllerSpec + Observation only; its model is linearize(spec.nominal). The lock metrics come
from Truth inside the test only (test_ctrl_c_harness.py): lock error in pm, lock-excursion count (5 pm threshold,
1 pm hysteresis: g26 metrics), time to lock, and the OMA penalty against the oracle point of each world.

Scenarios: a steady lock (nominal world); the A2 808 degC/s run and the A1 slide-10 aggressor (a window on their
steepest part; the aggressor lags are pre-settled exactly); a +0.5 dB laser power step; model mismatch (worlds drawn
from the uncertainty set, the controller using the nominal design); and the cold and probe acquisitions.

Findings these tests report:

- With the aggressor 5 um from ring 0's monitor PD (the stress scenarios *_pd_adjacent), the interim thermal site
  law heats the PD to about 270 degC in the A1/A2 windows. So the PD dark current (Arrhenius law) rises to about
  3 uA (about 5000 codes). A monitor set-point lock cannot tell that from detuning: it moved the ring by about
  25-35 pm. So the tracking error is scored against the monitor-consistent reference (target + dark offset / slope,
  Truth), and the total offset is reported separately. At the default site (55 um from the PD) the offset is small:
  total error 0.75 pm max under A2, 0.45 pm under A1.
- A laser power step moves a monitor set-point lock (the drop code scales with power): +0.5 dB moves the lock point
  by tens of pm. The test bounds the dynamics, not this static shift (g26 has no reference tap). A drop of more than
  1.1 dB puts the set-point above the peak, and every ring folds (slow, documented failure).
- Toward the A1 hot cusp in the stress scenario the PD dark current exceeds the set-point, and ring 0 runs away
  (slow, documented failure; an open plant question).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from mrm.control import models as control_models
from mrm.control import registry
from mrm.control.api import ControllerSpec
from mrm.control.baselines import kalman_lqg
from mrm.scenarios import registry as scenario_registry
from mrm.scenarios.generators import Scenario, step

from test_ctrl_c_harness import build_rig, hold, load_cfg, lock_metrics, realize, run_loop


@pytest.fixture(scope="module")
def cfg():
    return load_cfg()


@pytest.fixture(scope="module")
def models(cfg):
    return control_models.linearize_bank(cfg, control_models.OperatingPoint(), 1e-5, latency_ticks=1)


def _report(name, m):
    print(f"\n[{name}] " + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items()))


# ------------------------------------------------------------------------------------------------------- API and design
def test_registered_and_refuses_without_knowledge(cfg):
    assert registry.resolve("kalman_lqg") is kalman_lqg.KalmanLQG
    ctrl = registry.create("kalman_lqg", {"q_pm": 2.0})
    assert ctrl.params.q_pm == 2.0 and ctrl.latency_ticks == 1
    bare = load_cfg(knowledge=False)
    spec = ControllerSpec.from_config(bare, n_batch=1, nominal=bare)
    with pytest.raises(control_models.KnowledgeError, match="model_nominal"):
        kalman_lqg.KalmanLQG().reset(spec)
    spec2 = ControllerSpec.from_config(cfg, n_batch=1, nominal=None)
    with pytest.raises(ValueError, match="nominal"):
        kalman_lqg.KalmanLQG().reset(spec2)


def test_design_is_robustly_stable_on_the_nominal_model(models):
    """The LQG designed on the nominal model is stable against the same model with the heater gain scaled 0.3 to 6
    (dP/dcode alone varies 1.9-4.6x over the code range; self-heating changes the loop gain with power)."""
    d = kalman_lqg.design_lqg(models)
    for g in (0.3, 0.5, 1.0, 2.0, 4.0, 6.0):
        rho = kalman_lqg.closed_loop_spectral_radius(d, 0, models[0], gain=g)
        assert rho < 0.9999, (g, rho)
    # regulator equations: zero steady error to a step and a ramp of d in the nominal model
    lm = models[0]
    Pp, Gp, Gdp, Cp, Ddp, _ = lm.delayed()
    s_pm = lm.Dc[1, 1] * 1e12
    zt = np.zeros(Pp.shape[0])
    zp = np.zeros(d.nz)
    errs = []
    for k in range(1500):
        dk = 2.0 * max(k - 100, 0) * 0.01 / s_pm        # ramp of 2 pm/ms of open-loop shift, in K
        y = Cp[0] @ zt + Ddp[0] * dk
        z = zp + d.Lk[0] * (y - d.Cy[0] @ zp)
        u = -d.Kf[0] @ z
        errs.append((Cp[1] @ zt + Ddp[1] * dk) * 1e12)
        zt = Pp @ zt + Gp * u + Gdp * dk
        zp = d.Phi[0] @ z + d.Gz[0] * u
    assert abs(errs[-1]) < 0.01                          # type 2: the ramp error vanishes


def test_design_flags_a_ring_without_heater_authority(models):
    """A model linearized where the heater has no authority (dP/dcode = 0: past the code range or in the driver's
    compliance, as some sampled worlds' OMA points are) has no regulator solution: design_lqg marks that ring
    unregulated with zero gains instead of raising for the whole bank (the oracle relies on this), and the other rings
    keep their design."""
    from dataclasses import replace
    lm0 = models[0]
    Bc = lm0.Bc.copy()
    Bc[:, 0] = 0.0
    dead = replace(lm0, Bc=Bc)
    d = kalman_lqg.design_lqg([dead, models[1]])
    assert d.regulated.tolist() == [False, True]
    assert np.all(d.Kf[0] == 0.0) and np.any(d.Kf[1] != 0.0)
    d_ok = kalman_lqg.design_lqg(models[1:2])
    np.testing.assert_allclose(d.Kf[1], d_ok.Kf[0], rtol=1e-9)


def test_design_is_stable_on_sampled_worlds(cfg, models):
    """Model mismatch: linearize sampled worlds of the uncertainty set (each at its own OMA-optimal point,
    Truth, with its own latencies) and close the loop with the controller designed on the nominal model. Every
    FEASIBLE world must be stable. A world is infeasible for ring 0 when the heater power its OMA point needs is not
    positive (the ring already sits red of the target with the heater off: the low end of the heater range, a
    property of the world, not of the controller); the count is reported."""
    d = kalman_lqg.design_lqg(models)
    pp = realize(cfg, 16, uncertainty="lhs", variation="sample", seeds=3)
    rhos, infeasible = [], []
    for b in range(pp.batch):
        lm_b = control_models.linearize_bank(cfg, control_models.OperatingPoint(), 1e-5, latency_ticks=1, pp=pp, b=b)[0]
        if lm_b.op.heater_power_w[0] <= 5e-4 or lm_b.u0 >= 0.999 * 65535:
            infeasible.append(b)
            continue
        rhos.append(kalman_lqg.closed_loop_spectral_radius(d, 0, lm_b))
    print(f"\nspectral radius over {len(rhos)} feasible LHS worlds (infeasible {infeasible}):", np.round(rhos, 5))
    assert len(rhos) >= 10
    assert max(rhos) < 1.0


# ---------------------------------------------------------------------------------------------------------- closed loop
def test_steady_lock_nominal(cfg):
    rig = build_rig(cfg)
    hold(rig, 30)
    tr = run_loop(rig, kalman_lqg.KalmanLQG(), 2000)
    m = lock_metrics(tr.detuning_m, rig.targets_m, oma_w=tr.oma_w, oracle_oma_w=rig.oracle_oma_w, settle_ticks=200)
    _report("lqg steady", m)
    assert m["excursions"] == 0 and m["time_to_lock_s"] == 0.0
    # nominal-model target vs the oracle point of the plant: the same design, so only the plant's DAC errors and
    # bias-DAC noise remain (about -0.1 pm; it was -0.6 pm while the harness ran the plant at the wrong bias)
    assert abs(m["mean_pm"]) < 0.5
    jitter = np.std(tr.detuning_m[200:], axis=0) * 1e12
    assert np.max(jitter) < 0.3
    assert m["oma_penalty_db"] < 0.01
    tel = tr.telemetry
    assert np.all(tel["locked"][-1] == 1.0) and np.all(tel["state"][-1] == kalman_lqg.ST_LOCK)
    assert np.all(np.abs(tel["detuning_est_pm"][-1] - tr.detuning_m[-1] * 1e12) < 2.0)
    assert all(np.all(v == 0) for v in tr.saturation.values())


def _aggressor_window(cfg, name, k0, n):
    sc = scenario_registry.create(name)
    rig = build_rig(cfg, scenario=sc, k0=k0, aggressor_presettle_ticks=6000)
    hold(rig, 30)
    tr = run_loop(rig, kalman_lqg.KalmanLQG(), n)
    total = lock_metrics(tr.detuning_m, rig.targets_m, oma_w=tr.oma_w, oracle_oma_w=rig.oracle_oma_w,
                         settle_ticks=600)
    dyn = lock_metrics(tr.detuning_m, tr.ref_m, settle_ticks=600)
    return rig, tr, total, dyn


def test_a2_808_aggressor_window(cfg):
    """A2 (symmetric triangle, 1616 degC/s edges, resonance referencing): 3 ms on the rising edge, 20-23 ms after
    the cold end. Open loop the ring drifts about 23 pm/ms (the self-heating already divides the 123 pm/ms shift by
    1 + G|P'|). The tracking error against the monitor-consistent reference stays within a few pm."""
    rig, tr, total, dyn = _aggressor_window(cfg, "a2", 20000, 3000)
    _report("lqg A2 total (vs oracle)", total)
    _report("lqg A2 tracking (vs monitor-consistent ref)", dyn)
    assert dyn["rms_pm"] < 3.0 and dyn["max_pm"] < 5.0
    assert dyn["excursions"] == 0


def test_a1_slide10_aggressor_window(cfg):
    """A1 (rectified sine, 2127 degC/s peak at the cusp): 3 ms of the rising half where the rate is about 1700-2000
    degC/s (40-43 ms before the hot cusp), before the heater runs out of range near the top (at the A1 peak the
    aggressor alone heats the victim more than the lock needs, so the heater reaches code 0)."""
    period = 130700
    rig, tr, total, dyn = _aggressor_window(cfg, "a1", period - 43000, 3000)
    _report("lqg A1 total (vs oracle)", total)
    _report("lqg A1 tracking (vs monitor-consistent ref)", dyn)
    assert dyn["rms_pm"] < 3.0 and dyn["max_pm"] < 6.0
    assert dyn["excursions"] == 0


def test_laser_power_step(cfg):
    """+0.5 dB on every line at 0.6 ms. A monitor set-point lock moves its lock point (the drop code scales with the
    power, a known failure mode of a set-point lock); the loop must stay locked and settle within 0.5 ms with no
    oscillation."""
    sc = Scenario(name="pstep", components=(step("laser.power_delta_db", 6e-4, 0.5),))
    rig = build_rig(cfg, scenario=sc)
    hold(rig, 30)
    k_start = rig.k                     # the warm start also runs ticks: the loop starts at tick 37 in g26, not 30
    tr = run_loop(rig, kalman_lqg.KalmanLQG(), 2600)
    det = tr.detuning_m[:, 0] * 1e12
    t_step = 600 - k_start              # record index of the first sample after the step (tick 600 = 0.6 ms)
    assert tr.ticks[t_step - 1] == 600  # sample j is taken when the plant reaches tick ticks[j]: 'before' ends at 600
    before = det[t_step - 50:t_step].mean(axis=0)
    after = det[-300:].mean(axis=0)
    settle = det[t_step + 500:]
    print("\n[lqg power step] lock-point shift pm:", np.round(after - before, 2),
          " max deviation after 0.5 ms:", np.round(np.max(np.abs(settle - after), axis=0), 3))
    assert np.all(np.abs(settle - after) < 1.0)
    assert np.all(np.abs(after - before) < 60.0)
    assert np.all(tr.telemetry["state"][-1] == kalman_lqg.ST_LOCK)


@pytest.mark.slow
def test_a1_toward_the_cusp_dark_current_runaway(cfg):
    """Documented failure in the STRESS scenario a1_slide10_aggressor_pd_adjacent (the aggressor at the site
    aggressor_pd_adjacent, 5 um from ring 0's monitor PD; not a realistic layout).

    Toward the A1 hot cusp the calibrated aggressor heats ring 0's monitor PD through the interim site law to
    ~350-500 degC (the ring itself stays near 135 degC). The PD dark current (Arrhenius law) then grows toward and
    past the net set-point (~47 000 codes; 26 700 codes at -17 ms, 50 000 at -11 ms), and a monitor set-point lock
    cannot hold its line: in this 7 ms window ring 0 ends ~270 pm red of its target. In a longer run from -43 ms ring
    0's heater reached full scale ~11 ms before the cusp. Its next order (FSR 10.3 nm) then swept across lines 3, 2, 1
    and starved those rings, which folded to the blue branch: a bank-wide failure started by one ring's monitor.

    Asserted: the dark current took most of the set-point (Truth), ring 0 lost its line, and no ring that lost its
    line is declared locked."""
    period = 130700
    rig = build_rig(cfg, scenario=scenario_registry.create("a1_pd_adjacent"), k0=period - 17000,
                    aggressor_presettle_ticks=6000)
    hold(rig, 30)
    ctrl = kalman_lqg.KalmanLQG()
    dark = []
    tr = run_loop(rig, ctrl, 7000, record_every=10,
                  on_edge=lambda r, k: dark.append(float(r.dark_codes()[0, 0])) if k % 1000 == 0 else None)
    d = ctrl.design
    err = (tr.detuning_m[-1, 0] - rig.targets_m[0]) * 1e12
    lost = np.abs(err) > 50.0
    print(f"\n[lqg A1 toward the cusp] dark code ring 0 every 1 ms {np.round(dark)}, net set-point "
          f"{d.y0[0] - d.y_zero[0]:.0f}; final error pm {np.round(err, 0)}; locked {tr.telemetry['locked'][-1, 0]}; "
          f"heater {tr.heater_codes[-1, 0]}")
    assert max(dark) > 0.6 * (d.y0[0] - d.y_zero[0])          # Truth: the dark current took most of the set-point
    assert lost[0]
    assert np.all(tr.telemetry["locked"][-1, 0][lost] == 0.0)


def test_refuses_round_robin_schedule():
    """The design assumes every ring's command acts at every edge; under round_robin only one ring's command is
    applied per edge, so the predictor would run on commands that never reached the heater: refused at reset."""
    cfg_rr = load_cfg(overrides=["controller_interface.schedule=round_robin"])
    spec = ControllerSpec.from_config(cfg_rr, n_batch=1, nominal=cfg_rr)
    assert spec.schedule == "round_robin"
    with pytest.raises(ValueError, match="round_robin"):
        kalman_lqg.KalmanLQG().reset(spec)


@pytest.mark.slow
def test_laser_power_drop_past_the_set_point(cfg):
    """Documented failure of a monitor set-point lock (g26 has no reference tap): the nominal set-point is
    f* = 0.77 of the nominal drop peak, so a laser power DROP of more than -10 log10(0.77) = 1.1 dB puts the
    set-point above the new peak. -1.0 dB: still locked, but the lock point moves about 70 pm toward the peak (large
    OMA cost). -1.5 dB: the loop cools every ring through the peak, the rings fold to the blue branch (several nm) and
    the heaters run to code 0; the controller has no loss-of-lock detection or re-acquisition in LOCK (state stays
    LOCK), but its ``locked`` telemetry drops to 0: it does not claim a lock it lost."""
    out = {}
    for db in (-1.0, -1.5):
        sc = Scenario(name="pdrop", components=(step("laser.power_delta_db", 6e-4, db),))
        rig = build_rig(cfg, scenario=sc)
        hold(rig, 30)
        tr = run_loop(rig, kalman_lqg.KalmanLQG(), 2600)
        err = (tr.detuning_m - rig.targets_m[None]) * 1e12
        out[db] = (err, tr.telemetry)
        print(f"\n[lqg power drop {db} dB] final offset pm {np.round(err[-300:].mean(axis=0)[0], 1)}, locked tel "
              f"{tr.telemetry['locked'][-1, 0]}, heater {tr.heater_codes[-1, 0]}")
    err, tel = out[-1.0]
    assert np.all(np.std(err[-1000:], axis=0) < 1.0)                     # steady: still a lock
    assert np.all(err[-300:].mean(axis=0) < -40.0)                       # ... at a point moved toward the peak
    assert np.all(tel["locked"][-1] == 1.0)
    err, tel = out[-1.5]
    lost = np.abs(err[-1]) > 50.0
    assert np.all(lost)                                                  # Truth: every ring lost its line
    assert np.all(tel["locked"][-1][lost] == 0.0)                        # and none is declared locked


def test_probe_peak_acquisition_nominal(cfg):
    """Warm start + probe: the controller cools along the self-locked branch past the drop peak (fine steps near the
    top, no fold), records the peak and returns to f* of the measured peak. Lock is at the oracle point within 3 pm
    and the measured peak within 2 % of the true static peak."""
    rig = build_rig(cfg)
    hold(rig, 30)
    ctrl = kalman_lqg.KalmanLQG(probe_peak=True)
    tr = run_loop(rig, ctrl, 4000)
    tel = tr.telemetry
    assert np.all(tel["state"][-1] == kalman_lqg.ST_LOCK)
    m = lock_metrics(tr.detuning_m[-1000:], rig.targets_m)
    _report("lqg probe nominal (last 1 ms)", m)
    assert abs(m["mean_pm"]) < 3.0 and m["max_pm"] < 4.0
    pk_true = rig.static[0].y_peak_code
    assert np.all(np.abs(tel["peak_codes"][-1, 0] / pk_true - 1.0) < 0.02)
    # time to lock from the start of the probe (the probe itself leaves the lock window on purpose)
    ttl = lock_metrics(tr.detuning_m, rig.targets_m)["time_to_lock_s"]
    print(f"[lqg probe] time to lock {ttl * 1e3:.2f} ms")
    assert ttl < 3.5e-3


def _mismatch_run(cfg, probe: bool):
    pp = realize(cfg, 6, uncertainty="lhs", variation="sample", seeds=11)
    rig = build_rig(cfg, pp=pp)
    feas = rig.feasible_worlds()
    hold(rig, 30)
    ctrl = kalman_lqg.KalmanLQG(probe_peak=probe)
    tr = run_loop(rig, ctrl, 5000 if probe else 3000)
    st = tr.telemetry["state"][-1]
    off = (tr.detuning_m[-300:].mean(axis=0) - rig.targets_m) * 1e12
    jit = np.std(tr.detuning_m[-1000:], axis=0) * 1e12
    pen = 10 * np.log10(rig.oracle_oma_w / np.maximum(tr.oma_w[-300:].mean(axis=0), 1e-30))
    locked = (jit < 1.0) & (np.abs(off) < 150.0)          # Truth: steady, near the oracle point
    print(f"\n[lqg mismatch, probe={probe}] reachable worlds {feas}\n controller state\n{st}\n locked (Truth)\n"
          f"{locked}\n offset pm\n"
          f"{np.round(off, 1)}\n OMA penalty dB\n{np.round(pen, 2)}")
    return feas, locked, rig


@pytest.mark.slow
def test_model_mismatch_worlds_report(cfg):
    """Worlds from the uncertainty set, controller nominal, warm start at each world's oracle point.
    Reports, per world, whether its OMA point is reachable at all (heater power > 0, codes in range, bank total
    within the 200 mW array supply: in this LHS draw 2 of 6 worlds are not, e.g. an H86 world that needs its rings at
    ~215 degC), and per ring the lock outcome, the offset from the oracle point and the OMA penalty, with the nominal
    monitor set-point and with the peak probe. Asserted: the report runs and at least one reachable world locks every
    ring with the nominal set-point; the full claim (every reachable world locks) is the xfail test below."""
    feas, locked, _ = _mismatch_run(cfg, probe=False)
    assert feas.sum() >= 2
    assert any(np.all(locked[b]) for b in np.flatnonzero(feas))


@pytest.mark.slow
@pytest.mark.xfail(strict=False, reason="known limitation: with a nominal monitor "
                   "set-point the lock point is wrong by the monitor-gain / laser-power mismatch and can sit beyond "
                   "the true peak; the peak probe is not yet robust to the slower (undercut) and stronger "
                   "self-heating worlds, so some reachable worlds lose lock")
def test_model_mismatch_every_reachable_world_locks(cfg):
    for probe in (False, True):
        feas, locked, _ = _mismatch_run(cfg, probe=probe)
        for b in np.flatnonzero(feas):
            assert np.all(locked[b]), (probe, b, locked[b])


@pytest.mark.slow
def test_cold_start_acquisition_single_line(cfg):
    """Cold start (heaters at code 0, cold branch) of ring 0 with only its own line lit: ramp to the nominal lock
    code + margin, probe down past the peak (no fold), return to f* of the measured peak, lock.

    Only line 0 is on, because this baseline does not arbitrate lines. In the full g26 bank a cold ramp crosses three
    foreign lines (the 7.6 nm blue offset spans lines n+1..n+3). After any fold of 1.2 nm, rings 4-7 reach line n-4
    with their next order (FSR 10.3 nm vs a 9.16 nm four-line span). With all lines on, this test's bank run left
    rings 0-4 on wrong lines or in acquisition. Line arbitration is the WDM supervisor's job (docs/ARCHITECTURE.md,
    section 6, "Controllers shipped with the model")."""
    from mrm.scenarios.generators import const
    sc = Scenario(name="line0", components=(const("laser.line_enable", 0.0, absolute=True,
                                                  elements=tuple(range(1, 8))),))
    rig = build_rig(cfg, scenario=sc, warm_detuning_m=None)
    ctrl = kalman_lqg.KalmanLQG(start="cold", ramp_step=80.0, probe_step=20.0)
    tr = run_loop(rig, ctrl, 14000, record_every=10)
    st = tr.telemetry["state"][-1, 0, 0]
    target = control_models.oma_optimal_detuning(rig.pp, 0, cond=rig.conditions(0))[0][0]
    err = (tr.detuning_m[-50:, 0, 0].mean() - target) * 1e12
    i_lock = int(np.argmax(tr.telemetry["state"][:, 0, 0] == kalman_lqg.ST_LOCK))
    print(f"\n[lqg cold start, line 0 only] state {st:.0f}, lock error {err:.2f} pm, hand-over at "
          f"{tr.edge_ticks[i_lock] * 1e-3:.2f} ms, peak {tr.telemetry['peak_codes'][-1, 0, 0]:.0f} codes")
    assert st == kalman_lqg.ST_LOCK
    assert abs(err) < 6.0      # single-line world: no background drop from the other lines, so f* maps a few pm off
