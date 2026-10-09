"""Ramp-and-maximize acquisition (PAT-3) with the red-side lock hand-off, closed loop on the g26 reference plant
(tests/verification/test_ctrl_b_plant.py::CtrlBPlant). Truth is read here only, never by the controller.

Scenarios: steady lock after a cold acquisition (lasers on at t = 0, heaters at code 0), the self-heating jump of a
heater-up sweep (07 P0-1), a +10 % laser power step, the standalone all-ring cold start (the wrong-line / duplicate
hazard that the WDM supervisor exists for). The aggressor runs (A1, A2) are in test_ctrl_b_ramp_aggressor.py (slow).

Reported numbers (printed; pytest -s): lock error (jitter rms and max about the steady detuning, pm), offset from the
static OMA optimum (pm and OMA penalty dB), lock-excursion count (5 pm threshold, 1 pm hysteresis about the steady
detuning) and time to lock (first tick after which the detuning stays within 5 pm of its final value).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from mrm.control import registry
from mrm.control.api import Controller, ControllerSpec, validate_telemetry
from mrm.control.baselines import ramp_acquire as ra
from mrm.control.baselines.ramp_acquire import RampAcquire, ST_LOCK, ST_DOWN, ST_UP
from mrm.scenarios import generators as g

from test_ctrl_b_plant import CtrlBPlant, load_g26, make_spec, realize, run_closed_loop, excursions

KNOW = ["line_grid", "ring_to_line", "heater_nominal", "optics_nominal", "dlambda_dt_nominal"]
RING0_FIRST_LINE = 7     # nominal g26: ring 0's first resonance approached from blue is line 7 (order -1, 1.87 nm)


def steady_metrics(det_pm: np.ndarray, ticks: np.ndarray, t_from: int, thr: float = 5.0, hyst: float = 1.0):
    """Lock error about the steady detuning d_ss (the mean over the last quarter): jitter rms and max |d - d_ss| over
    ticks >= t_from, excursions of |d - d_ss| above thr (hysteresis hyst), and the time to lock (first tick after which
    |d - d_ss| < thr for good)."""
    d_ss = float(np.mean(det_pm[-max(len(det_pm) // 4, 1):]))
    e = det_pm - d_ss
    sel = ticks >= t_from
    bad = np.flatnonzero(np.abs(e) >= thr)
    t_lock = int(ticks[bad[-1] + 1]) if bad.size and bad[-1] + 1 < len(ticks) else (int(ticks[0]) if not bad.size
                                                                                    else -1)
    return {"d_ss_pm": d_ss, "rms_pm": float(np.sqrt(np.mean(e[sel] ** 2))), "max_pm": float(np.max(np.abs(e[sel]))),
            "excursions": excursions(e[sel], thr, hyst), "t_lock": t_lock}


@pytest.fixture(scope="module")
def nominal():
    cfg = load_g26(knowledge=KNOW)
    pp = realize(cfg, 1, nominal=True)
    spec = make_spec(cfg, 1)
    return cfg, pp, spec


@pytest.fixture(scope="module")
def ring0_run(nominal):
    """Ring 0 acquires alone (the other heaters stay at code 0), 3 ms."""
    cfg, pp, spec = nominal
    plant = CtrlBPlant(pp, cfg, spec)
    plant.assignment[0, 0] = RING0_FIRST_LINE
    ctrl = RampAcquire(rings=[0])
    rec, val = run_closed_loop(plant, ctrl, 3000, keep_monitor=True)
    return plant, ctrl, rec, val


def test_registered_and_protocol(nominal):
    _, _, spec = nominal
    ctrl = registry.create("ramp_acquire", {"alpha": 0.6, "rings": [0, 1]})
    assert isinstance(ctrl, RampAcquire) and isinstance(ctrl, Controller)
    assert ctrl.params.alpha == 0.6
    ctrl.reset(spec)
    validate_telemetry(ctrl.telemetry(), spec)
    with pytest.raises(TypeError):
        registry.create("ramp_acquire", {"no_such": 1})
    # tuning from the nominal linewidth when the knowledge allows, documented defaults otherwise (07 P1-1)
    assert ctrl.fsm.dw_up0 == pytest.approx(ctrl.fsm.design.fwhm_w / 3.5)
    assert ctrl.fsm.dw_up.shape == (spec.n_batch, spec.n_rings) and np.all(ctrl.fsm.dw_up == ctrl.fsm.dw_up0)
    bare = ControllerSpec.from_config(load_g26(), n_batch=1, nominal=None)
    c2 = RampAcquire()
    c2.reset(bare)
    assert c2.fsm.dw_up0 == 0.004 and np.all(c2.fsm.dw_up == 0.004) and c2.fsm.design.fwhm_m is None


def test_upsweep_jumps_over_the_peak(ring0_run):
    """07 P0-1: a heater-up sweep approaches on the positive-feedback side and JUMPS. Shown on Truth (the detuning
    goes from blue to red of the line within one controller period, never resting near zero) and handled: the peak
    used for the set-point is the one measured on the DOWN scan, well above the up-sweep's landing level."""
    plant, ctrl, rec, _ = ring0_run
    f = ctrl.fsm
    assert f.jumps[0, 0] >= 1 and f.bistable[0, 0]
    # Truth: during the first up-sweep the detuning to line 7 goes from more than FWHM/2 blue of the line to more
    # than FWHM/4 red of it within 4 ticks (a few us of self-heating dynamics), for at most one sweep step of the
    # heater: a quasi-static step would move it by dw_up x dlambda/dw only
    st = np.array([t["state"][0, 0] for t in rec.telemetry])
    edges = np.array(rec.edge_ticks)
    t_down = int(edges[np.flatnonzero(st == ST_DOWN)[0]])
    sel = rec.ticks <= t_down
    d = rec.det_m[sel, 0, 0] * 1e12
    code = rec.heater[sel, 0, 0]
    fwhm_pm = f.design.fwhm_m * 1e12
    step_pm = f.dw_up0 * f.design.tuning_m_per_w_drive * 1e12
    j = np.flatnonzero((d[:-4] < -0.5 * fwhm_pm) & (d[4:] > 0.25 * fwhm_pm))
    assert j.size, "no jump across the line"
    j = int(j[0])
    w = (code[j:j + 5] / f.code_max) ** f.q
    assert np.ptp(w) <= f.dw_up0 * 1.01                         # at most one sweep step of the heater
    assert d[j + 4] - d[j] > 3.0 * step_pm, (d[j:j + 5], step_pm)
    # handled: the true peak (DOWN scan) is well above the landing level of the up-sweep
    assert f.y_land[0, 0] < 0.9 * f.y_pk[0, 0], (f.y_land[0, 0], f.y_pk[0, 0])
    print(f"\njump {d[j]:.0f} -> {d[j + 4]:.0f} pm in 4 ticks (one heater step moves {step_pm:.0f} pm); landing "
          f"{f.y_land[0, 0]:.0f} codes vs true peak {f.y_pk[0, 0]:.0f}")


def test_steady_lock_metrics(ring0_run):
    plant, ctrl, rec, val = ring0_run
    f = ctrl.fsm
    assert f.state[0, 0] == ST_LOCK and f.locked[0, 0]
    t = plant.truth()
    assert t["nearest_line"][0, 0] == RING0_FIRST_LINE
    d = rec.det_m[:, 0, 0] * 1e12
    m = steady_metrics(d, rec.ticks, t_from=1500)
    # red side, near the half-maximum point of the drop peak (x = 1: delta = FWHM/2 = 170 pm for a Lorentzian)
    assert 100.0 < m["d_ss_pm"] < 220.0, m
    assert m["rms_pm"] < 1.0 and m["max_pm"] < 3.0 and m["excursions"] == 0, m
    assert 0 < m["t_lock"] < 1500, m
    asg = np.full((1, plant.N), -1)
    asg[0, 0] = RING0_FIRST_LINE
    pen, pen_red = plant.oma_penalty_db(assignment=asg)
    assert pen[0, 0] < 1.5, pen
    # the telemetry's Lorentzian detuning estimate (nominal linewidth) is within 30 % of Truth
    est = rec.telemetry[-1]["detuning_est_pm"][0, 0]
    assert abs(est - m["d_ss_pm"]) < 0.3 * m["d_ss_pm"], (est, m["d_ss_pm"])
    assert val.saturation_counts()["tx.heater"].sum() == 0
    print(f"\nramp_acquire ring 0 steady: d_ss {m['d_ss_pm']:.1f} pm, jitter rms {m['rms_pm']:.3f} pm, max "
          f"{m['max_pm']:.2f} pm, excursions {m['excursions']}, time to lock {m['t_lock'] * 1e-3:.3f} ms, OMA "
          f"penalty {pen[0, 0]:.2f} dB (vs the static optimum)")


def test_alpha_near_max_slope_lowers_the_oma_penalty(nominal):
    """alpha = 0.75 sits near the maximum slope of the drop peak (x = 1/sqrt 3): smaller OMA penalty than 0.5."""
    cfg, pp, spec = nominal
    plant = CtrlBPlant(pp, cfg, spec)
    plant.assignment[0, 0] = RING0_FIRST_LINE
    ctrl = RampAcquire(rings=[0], alpha=0.75)
    run_closed_loop(plant, ctrl, 1600)
    asg = np.full((1, plant.N), -1)
    asg[0, 0] = RING0_FIRST_LINE
    pen, _ = plant.oma_penalty_db(assignment=asg)
    assert ctrl.fsm.state[0, 0] == ST_LOCK
    assert pen[0, 0] < 0.3, pen
    print(f"\nalpha 0.75: OMA penalty {pen[0, 0]:.3f} dB, detuning {plant.truth()['det_m'][0, 0, 7] * 1e12:.1f} pm")


def test_laser_power_step(nominal):
    """+10 % laser power (0.41 dB) at 2 ms: the set-point was a fraction of the OLD peak, so the lock point moves
    (04 C3 failure mode, a known limit of a fraction-of-peak lock without re-normalization); the loop must hold
    lock (no re-acquisition, no excursion beyond the moved point after it settles)."""
    cfg, pp, spec = nominal
    step = g.Scenario(name="pstep", components=(g.step("laser.power_delta_db", 2e-3, 10 * math.log10(1.1)),))
    plant = CtrlBPlant(pp, cfg, spec, scenario=step)
    plant.assignment[0, 0] = RING0_FIRST_LINE
    ctrl = RampAcquire(rings=[0])
    rec, _ = run_closed_loop(plant, ctrl, 4000)
    st = np.array([t["state"][0, 0] for t in rec.telemetry])
    edges = np.array(rec.edge_ticks)
    assert np.all(st[edges >= 1700] == ST_LOCK)                 # no loss of lock across the step
    d = rec.det_m[:, 0, 0] * 1e12
    before = float(np.mean(d[(rec.ticks > 1500) & (rec.ticks < 2000)]))
    after = float(np.mean(d[rec.ticks > 3500]))
    m = steady_metrics(d[rec.ticks > 2000], rec.ticks[rec.ticks > 2000], t_from=3000)
    # y scales with the power, so the loop holds the same code further from the line: the ring moves red (heater
    # up), by about (dP/P) y / |dy/ddelta| (04 C3)
    assert after > before, (before, after)
    assert after - before < 60.0, (before, after)
    assert m["excursions"] == 0 and m["rms_pm"] < 1.0, m
    print(f"\nlaser +10 %: detuning {before:.1f} -> {after:.1f} pm (moved {after - before:.1f} pm), settles in "
          f"{(m['t_lock'] - 2000) * 1e-3:.3f} ms, jitter after {m['rms_pm']:.3f} pm")


def test_standalone_cold_start_takes_foreign_lines(nominal):
    """The g26 cold-ring hazard (docs/handoff/integrator.md section 4): with every ring acquiring its first
    resonance from blue at once, rings lock on lines other than the declared map, and some share a line (the
    downstream one starved). This is why the WDM supervisor exists; the ramp controller reports it honestly."""
    cfg, pp, spec = nominal
    plant = CtrlBPlant(pp, cfg, spec)
    ctrl = RampAcquire()
    rec, _ = run_closed_loop(plant, ctrl, 2500)
    near = plant.truth()["nearest_line"][0]
    wrong = int(np.sum(near != np.arange(plant.N)))
    shared = plant.N - len(set(near.tolist()))
    jumps = ctrl.fsm.jumps[0]
    print(f"\nstandalone all-ring cold start: nearest lines {near.tolist()}, wrong vs declared {wrong}, shared "
          f"{shared}, jumps {jumps.tolist()}")
    assert wrong >= 4 and np.all(jumps >= 1)
    # the controller itself never sees this: every ring believes it is locked (no Truth leak)
    assert np.sum(ctrl.fsm.state[0] == ST_LOCK) >= 6


def test_partial_peak_below_zero_drive_is_locked():
    """A ring already on the hot branch of a line whose peak lies below w = 0 cannot scan to the peak (heaters only
    heat): the FSM locks on the highest point seen (``partial``) instead of failing. Unit test on a synthetic
    monitor: the DOWN scan's y keeps rising until w reaches 0."""
    cfg = load_g26(knowledge=KNOW)
    spec = make_spec(cfg, 1)
    f = ra.RampFSM(spec, ra.RampParams())
    m = np.ones((1, spec.n_rings), bool)
    f.start(m, w_from=0.0)
    ys = [5000, 5000, 5100, 5000, 30000, 26000]           # floor, then a jump and a fall (event)
    for y in ys:
        f.step(np.full((1, spec.n_rings), float(y)), m)
    assert np.all(f.state == ST_DOWN)
    y = 26000.0
    for _ in range(40):
        y += 500.0                                          # rising all the way down to w = 0
        f.step(np.full((1, spec.n_rings), y), m)
        if np.all(f.state == ST_LOCK):
            break
    assert np.all(f.state == ST_LOCK) and np.all(f.partial)
    assert np.all(f.setpoint < f.y_pk)


def test_composition_with_the_pi_lock(nominal):
    """AcquireThenLock: the PAT-3 acquisition hands ring 0 (on the stable red branch of its line) to the drop-port PI
    lock of controllers A (mrm.control.baselines.pi_lock, fraction 0.7), whose hand-off contract is exactly that."""
    pi_lock = pytest.importorskip("mrm.control.baselines.pi_lock")
    cfg, pp, spec = nominal
    plant = CtrlBPlant(pp, cfg, spec)
    plant.assignment[0, 0] = RING0_FIRST_LINE
    ctrl = registry.create("acquire_then_lock", {"acquire": {"rings": [0]},
                                                 "lock": {"name": "mrm.control.baselines.pi_lock:PiLock",
                                                          "params": {"fraction": 0.7}}})
    rec, val = run_closed_loop(plant, ctrl, 4500)
    assert ctrl.handed
    assert isinstance(ctrl.lock_ctrl, pi_lock.PiLock)
    t = plant.truth()
    assert t["nearest_line"][0, 0] == RING0_FIRST_LINE
    d = rec.det_m[:, 0, 0] * 1e12
    tail = d[rec.ticks > 4200]
    print(f"\nacquire_then_lock -> pi_lock(0.7): ring 0 at {tail.mean():.1f} pm red of line 7 (jitter {tail.std():.3f} "
          f"pm); the 0.5 hand-off point was ~146 pm")
    assert 60.0 < tail.mean() < 140.0 and tail.std() < 1.0


def synthetic_obs(spec, tick, y=10000.0, applied=0, **extra):
    """An Observation with every declared sensor (tests only): monitor blocks of value y, the applied heater code,
    and ``extra`` channel overrides {name: array}; every new-sample flag True."""
    from mrm.control.api import Observation
    B, N, M = spec.n_batch, spec.n_rings, spec.samples_per_step
    sensors, new = {}, {}
    for name in spec.sensors:
        shp = spec.shape_of(name)
        sensors[name] = np.zeros(shp)
        new[name] = np.ones(shp[:2] if len(shp) > 2 else shp, bool)
    sensors["tx.monitor.samples"] = np.full((B, N, M), float(y))
    sensors["tx.monitor.count"] = np.full((B, N), M)
    sensors["tx.monitor.latest"] = np.full((B, N), float(y))
    sensors["monitor_valid"] = np.ones((B, N), bool)
    sensors["adc_overrange"] = np.zeros((B, N), bool)
    sensors["tx.heater.applied"] = np.full((B, N), applied)
    sensors.update(extra)
    return Observation(int(tick), spec.tick_s, sensors, new)


def test_start_delay_counts_from_the_first_step_after_reset(nominal):
    """A controller reset in mid-run (scenario fault.controller_reset, the runner calls reset(spec)) must wait its
    start delay from that moment, not from tick 0 (the delay was compared with the absolute tick)."""
    _, _, spec = nominal
    ctrl = RampAcquire(rings=[0], start_delay_s=50e-6)
    ctrl.reset(spec)
    per = spec.period_ticks
    t0 = 100_000
    for j in range(4):
        ctrl.step(synthetic_obs(spec, t0 + j * per, applied=1234))
        assert not ctrl.started                          # 0, 10, 20, 30 us after the reset
    for j in range(4, 7):
        ctrl.step(synthetic_obs(spec, t0 + j * per, applied=1234))
    assert ctrl.started and ctrl.fsm.state[0, 0] != ra.ST_IDLE
    out = ctrl.step(synthetic_obs(spec, t0 + 7 * per, applied=1234))
    assert np.all(out["tx.heater"][0, 1:] == 1234)        # rings outside params.rings keep their register


def test_acquire_then_lock_exposes_the_lock_latency_before_reset():
    """The runner reads Controller.latency_ticks once, BEFORE reset (mrm.sim.runner.run): the sequencer must know
    its lock's pipeline depth at construction (it built the lock at reset, so a lock with latency 3 ran with 0)."""
    pytest.importorskip("mrm.control.baselines.pi_lock")
    c = registry.create("acquire_then_lock", {"lock": {"name": "mrm.control.baselines.pi_lock:PiLock",
                                                       "params": {"latency_ticks": 3}}})
    assert c.latency_ticks == 3 and type(c.lock_ctrl).__name__ == "PiLock"
    assert ra.AcquireThenLock().latency_ticks == 0                  # default lock: controllers A's PI lock
