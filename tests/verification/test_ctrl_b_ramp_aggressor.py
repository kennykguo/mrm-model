"""Ramp acquisition + red-side lock under the G26 on-die aggressors (07 P1-4; scenarios A2 and A1). Slow.

Ring 0 (the victim of the g26 aggressor site) is acquired on a line with enough heater headroom (PAT-1 [0050]: bias
the ring above the hottest chip temperature so the heater can back off): line 0 (7.6 nm of tuning) for A2 (80.8 K),
line 4 one order down (8.74 nm) for A1 (about 101 K, 7 nm, under resonance referencing). The search window comes from
the nominal design (07 P1-1 knowledge): w = offset / (dlambda/dw) with the heater droop, +-0.035.

Layout finding (reported in docs/handoff/controllers_b.md): in g26 the aggressor site (0, 30 um) sits 5 um from ring
0's monitor PD at its default offset (0, 25 um), so the interim thermal model heats PD0 six times more than the ring
(100 mW: +205 K at the PD, +34 K at the ring). Under A2 the PD observer passes 300 degC after ~25 ms and its
Arrhenius dark current reaches ~10 uA, half the lock current: any monitor lock is dragged off the line. The
controller claims below therefore use ``ring.thermal.pd_offset_um = [0, -25]`` (the PD on the far side of the ring,
55 um from the aggressor); :func:`test_a2_default_layout_pd_dark_current` documents the default-layout outcome.

Reported (printed): lock error during the aggressor (deviation from the steady detuning, pm), heater range used.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from mrm.control.baselines.bitstat_lock import BitStatLock
from mrm.control.baselines.ramp_acquire import RampAcquire, ST_LOCK
from mrm.scenarios import library as lib

from test_ctrl_b_plant import CtrlBPlant, excursions, load_g26, make_spec, realize, run_closed_loop

pytestmark = pytest.mark.slow
KNOW = ["line_grid", "ring_to_line", "heater_nominal", "optics_nominal", "dlambda_dt_nominal"]
PD_FAR = ["ring.thermal.pd_offset_um=[0.0,-25.0]"]
T_ON = 3e-3


def _window(ctrl_spec, offset_nm=None, jump_nm=0.5):
    """Search window of the target resonance from the nominal design (heater_nominal + optics_nominal knowledge):
    the up-sweep meets the resonance ``jump_nm`` before the line (the cold-branch fold of the self-heating jump,
    ~0.4-0.6 nm at g26 nominal), at drive w with w P_fs = P (1 + y P) (heater droop), +-0.035."""
    from mrm.control.calibration import nominal_design
    d = nominal_design(ctrl_spec)
    off = (d.offset_m if offset_nm is None else offset_nm * 1e-9) - jump_nm * 1e-9
    w0 = off / d.tuning_m_per_w_drive                  # ideal law, no droop
    w_pred = w0 * (1.0 + d.y_per_w * d.p_drive_fs_w * w0)
    return max(w_pred - 0.035, 0.0), w_pred + 0.035


def _engine(ctrl, cfg, scenario, duration_s, record):
    runner = pytest.importorskip("mrm.sim.runner")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return runner.run(cfg, controller=ctrl, scenario=scenario, duration_s=duration_s, batch=1,
                          uncertainty="nominal", variation="off", record=record, reset_protocol="branch_low")


def _env(trace, stat):
    return [v for k, v in trace.items() if k.startswith("detuning_m") and k.endswith(stat)][0]


def test_a2_reference_plant():
    """A2 on the reference plant: the aggressor (calibrated, delayed to start after the lock) ramps 1616 degC/s for
    12 ms."""
    cfg = load_g26(overrides=PD_FAR, knowledge=KNOW)
    pp = realize(cfg, 1, nominal=True)
    spec = make_spec(cfg, 1)
    lo, hi = _window(spec)
    plant = CtrlBPlant(pp, cfg, spec, scenario=lib.a2_808_aggressor(), aggressor_delay_s=T_ON)
    ctrl = RampAcquire(rings=[0], w_start=max(lo - 0.04, 0.0), w_lo=lo, w_hi=hi)
    rec, val = run_closed_loop(plant, ctrl, int((T_ON + 12e-3) / spec.tick_s), record_every=2)
    assert plant.truth()["nearest_line"][0, 0] == 0
    d = rec.det_m[:, 0, 0] * 1e12
    ref = float(np.mean(d[(rec.ticks > T_ON * 1e6 - 600) & (rec.ticks < T_ON * 1e6)]))
    e = d[rec.ticks >= T_ON * 1e6] - ref
    st = np.array([t["state"][0, 0] for t in rec.telemetry])
    codes = rec.heater[rec.ticks >= T_ON * 1e6, 0, 0]
    print(f"\nA2 reference plant: lock detuning {ref:.1f} pm; during 12 ms of the 1616 degC/s ramp: error rms "
          f"{np.sqrt(np.mean(e ** 2)):.2f} pm, max {np.max(np.abs(e)):.2f} pm, excursions {excursions(e, 5, 1)}; "
          f"heater code {codes.max()} -> {codes.min()}")
    assert np.all(st[np.array(rec.edge_ticks) >= T_ON * 1e6] == ST_LOCK)
    assert np.max(np.abs(e)) < 10.0


def test_a2_full_cycle_engine():
    """A2 through one full 100 ms cycle on the engine (rise and fall at 1616 degC/s, the reversal at the top)."""
    cfg = load_g26(overrides=PD_FAR, knowledge=KNOW)
    spec = make_spec(cfg, 1)
    lo, hi = _window(spec)
    ctrl = RampAcquire(rings=[0], w_start=max(lo - 0.04, 0.0), w_lo=lo, w_hi=hi)
    res = _engine(ctrl, cfg, lib.a2_808_aggressor(), 110e-3,
                  [("detuning_m", 100, "mean"), ("detuning_m", 100, "max"), ("detuning_m", 100, "min"),
                   ("heater_code", 100, "min")])
    mean, mx, mn = (_env(res.trace, s)["value"][:, 0, 0] * 1e12 for s in ("mean", "max", "min"))
    t = _env(res.trace, "mean")["t_s"]
    after = t > 5e-3
    ref = float(np.median(mean[after]))
    dev = max(float(np.max(mx[after]) - ref), float(ref - np.min(mn[after])))
    codes = [v for k, v in res.trace.items() if k.startswith("heater_code")][0]["value"][:, 0, 0]
    print(f"\nA2 engine, 110 ms: lock detuning median {ref:.1f} pm, max deviation {dev:.2f} pm, heater code min "
          f"{codes[after].min():.0f}, state {ctrl.fsm.state[0, 0]}, wall {res.wall_s:.0f} s")
    assert ctrl.fsm.state[0, 0] == ST_LOCK
    assert dev < 10.0


@pytest.mark.parametrize("which", ["ramp", "bitstat"])
def test_a1_through_the_cusp_engine(which):
    """A1 (slide 10: T = 113.6 - 88.5 |sin(pi t / 130.7 ms)| degC at the victim, 2127 degC/s at the cusp), in its
    calibrated periodic form with the phase put 2 ms before the minimum (the acquisition happens where the slope is
    small), then the whole rising half-cycle (the slope grows to 2127 degC/s) and 8 ms past the cusp, where the
    slope reverses in one tick. Ring 0 on line 4 (8.74 nm of headroom), declared through tx.assignment so the
    engine's Truth refers to that line. ``bitstat``: the bit-statistics lock (same acquisition and inner loop, its
    set-point from the eye statistics; review addition)."""
    runner = pytest.importorskip("mrm.sim.runner")
    from mrm.plant.plant import prepare_scenario
    from test_ctrl_b_engine import WithAssignment
    cfg = load_g26(overrides=PD_FAR + ["controller_interface.bind.tx.assignment=controller"], knowledge=KNOW)
    spec = make_spec(cfg, 1)
    _, pp, _ = runner.load_inputs(cfg, batch=1, uncertainty="nominal", variation="off")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        csc, _ = prepare_scenario(lib.a1_slide10_aggressor(), pp)
    period = csc.aggressors[0].power[0].period_s
    t_min = 2e-3
    aggs = tuple(dataclasses.replace(a, power=tuple(dataclasses.replace(c, phase_s=period / 2 - t_min)
                                                    for c in a.power)) for a in csc.aggressors)
    csc = dataclasses.replace(csc, aggressors=aggs)
    lo, hi = _window(spec, offset_nm=8.74)
    cls = RampAcquire if which == "ramp" else BitStatLock
    ctrl = cls(rings=[0], w_start=max(lo - 0.04, 0.0), w_lo=lo, w_hi=hi)
    wrapped = WithAssignment(ctrl, [4, 1, 2, 3, 4, 5, 6, 7])
    dur = t_min + period / 2 + 8e-3
    res = _engine(wrapped, cfg, csc, dur, [("detuning_m", 100, "mean"), ("detuning_m", 100, "max"),
                                           ("detuning_m", 100, "min")])
    mean, mx, mn = (_env(res.trace, s)["value"][:, 0, 0] * 1e12 for s in ("mean", "max", "min"))
    t = _env(res.trace, "mean")["t_s"]
    after = t > 4e-3
    ref = float(np.median(mean[after]))
    dev = max(float(np.max(mx[after]) - ref), float(ref - np.min(mn[after])))
    cusp = (t > t_min + period / 2 - 5e-3) & (t < t_min + period / 2 + 5e-3)
    dev_cusp = max(float(np.max(mx[cusp]) - ref), float(ref - np.min(mn[cusp])))
    print(f"\nA1 engine through the cusp ({which}): lock detuning median {ref:.1f} pm, max deviation {dev:.2f} pm (within 5 ms "
          f"of the cusp {dev_cusp:.2f} pm), state {ctrl.fsm.state[0, 0]}, wall {res.wall_s:.0f} s")
    assert ctrl.fsm.state[0, 0] == ST_LOCK
    assert dev < 30.0


@pytest.mark.xfail(strict=False, reason="plant input: g26 puts the aggressor 5 um from monitor PD0; the PD observer "
                                        "passes 300 degC under A2 and its dark current drags any monitor lock off "
                                        "(docs/handoff/controllers_b.md)")
def test_a2_default_layout_pd_dark_current():
    cfg = load_g26(knowledge=KNOW)
    spec = make_spec(cfg, 1)
    lo, hi = _window(spec)
    ctrl = RampAcquire(rings=[0], w_start=max(lo - 0.04, 0.0), w_lo=lo, w_hi=hi)
    res = _engine(ctrl, cfg, lib.a2_808_aggressor(), 45e-3, [("detuning_m", 100, "mean"), ("t_pd_k", 100, "max")])
    mean = _env(res.trace, "mean")["value"][:, 0, 0] * 1e12
    t_pd = [v for k, v in res.trace.items() if k.startswith("t_pd_k")][0]["value"][:, 0, 0] - 273.15
    print(f"\nA2 default layout: PD0 up to {t_pd.max():.0f} degC; detuning {mean[50]:.0f} -> {mean[-1]:.0f} pm")
    assert ctrl.fsm.state[0, 0] == ST_LOCK and abs(mean[-1] - mean[50]) < 20.0
