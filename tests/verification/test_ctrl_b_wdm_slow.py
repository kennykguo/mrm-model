"""WDM supervisor, slow variants on the engine (mrm.sim.runner; skipped without it): PAT-1 barrel shift under a
tight die supply with tx.assignment bound to the controller, the fixed-map outcome under the same supply, the warm
start from a factory table measured on the same device (07 P1-1), and a cold start in sampled worlds.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

runner = pytest.importorskip("mrm.sim.runner")

from mrm.control.baselines.ramp_acquire import ST_LOCK
from mrm.control.baselines.wdm_supervisor import PH_TRACK, WdmSupervisor
from mrm.control.calibration import FactoryProcedure, factory_cal, heater_current_a, nominal_design
from mrm.scenarios import library as lib

from test_ctrl_b_plant import load_g26, make_spec

pytestmark = pytest.mark.slow
KNOW = ["line_grid", "ring_to_line", "heater_nominal", "optics_nominal", "dlambda_dt_nominal"]
TIGHT = "array.heater_supply_current_max_a=0.10"


def _run(cfg, ctrl, duration_s, batch=1, uncertainty="nominal", variation="off", seeds=None, record=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return runner.run(cfg, controller=ctrl, scenario=lib.static(), duration_s=duration_s, batch=batch,
                          seeds=seeds, uncertainty=uncertainty, variation=variation, reset_protocol="branch_low",
                          record=record or [("p_heater_w", 10, "max")])


def test_barrel_shift_under_a_tight_supply():
    """0.10 A instead of 0.15 A: the declared identity map (~0.136 A at g26 nominal) does not fit, so the cyclic
    policy takes a barrel shift (PAT-1, lock-to-cyclic) that does, declares it on tx.assignment, and every ring locks
    on its new line (engine wrong-line metric against the declared map: none)."""
    cfg = load_g26(overrides=["controller_interface.bind.tx.assignment=controller", TIGHT], knowledge=KNOW)
    ctrl = WdmSupervisor(policy="cyclic")
    res = _run(cfg, ctrl, 30e-3)
    s = int(ctrl.shift[0])
    assert s != 0 and ctrl.phase[0] == PH_TRACK
    np.testing.assert_array_equal(ctrl.assigned[0], (np.arange(8) + s) % 8)
    assert np.all(ctrl.fsm.state[0] == ST_LOCK)
    assert not np.any(res.metrics["wrong_line_final"])
    det = np.asarray(res.truth_final["detuning_m"])[0] * 1e12
    assert np.all((det > 80.0) & (det < 260.0)), det
    d = nominal_design(ctrl.spec)
    i_tot = float(heater_current_a(ctrl.fsm.code[0], d).sum())
    print(f"\nbarrel shift {s}: lines {ctrl.assigned[0].tolist()}, detuning {np.round(det, 1).tolist()} pm, "
          f"nominal supply current {i_tot * 1e3:.1f} mA of 100 mA")
    assert i_tot < 0.10


def test_fixed_map_under_a_tight_supply():
    """Same supply, tx.assignment bound to the scenario (g26): the supervisor may not remap, so it acquires in bus
    order until the budget is spent and flags the rest as supply-limited; the plant's limit is never hit."""
    cfg = load_g26(overrides=[TIGHT], knowledge=KNOW)
    ctrl = WdmSupervisor()
    res = _run(cfg, ctrl, 30e-3)
    lim = ctrl.supply_limited[0]
    locked = ctrl.fsm.state[0] == ST_LOCK
    print(f"\nfixed map, 0.10 A: locked {locked.astype(int).tolist()}, supply-limited {lim.astype(int).tolist()}")
    assert lim.sum() >= 1 and np.all(locked | lim)
    d = nominal_design(ctrl.spec)
    assert float(heater_current_a(ctrl.fsm.code[0], d).sum()) < 0.10
    assert not np.any(res.metrics["wrong_line_final"][0][locked])


def test_warm_start_from_the_factory_table():
    """07 P1-1 warm start: factory_cal on a station built from the same device (engine Plant), then the supervisor
    starts from the table: no map phase, every ring on its line, sooner than the cold start (~19 ms)."""
    from mrm.plant.plant import Plant
    cfg = load_g26(knowledge=KNOW)
    _, pp, _ = runner.load_inputs(cfg, batch=1, uncertainty="nominal", variation="off")
    spec = make_spec(cfg, 1)

    class Station:
        def __init__(self):
            self.spec = spec

        def __call__(self, scenario):
            p = Plant(pp, scenario, spec, cfg=cfg)
            p.reset(protocol="branch_low")
            p.spec = spec
            return p
    tab = factory_cal(Station(), FactoryProcedure())
    assert np.all(tab.ok[0]), tab.notes
    ctrl = WdmSupervisor(factory_table=tab)
    res = _run(cfg, ctrl, 14e-3, record=[("detuning_m", 10, "mean")])
    assert ctrl.phase[0] == PH_TRACK and np.all(ctrl.fsm.state[0] == ST_LOCK)
    assert not np.any(res.metrics["wrong_line_final"])
    det = np.asarray(res.truth_final["detuning_m"])[0] * 1e12
    print(f"\nwarm start: all rings locked within 14 ms on their lines, detuning {np.round(det, 1).tolist()} pm; "
          f"factory T_cal {tab.t_cal_c[0]:.1f} degC")


def test_cold_start_sampled_worlds():
    """Six sampled worlds (variation on, seed 7): report the map quality and the outcome. The supervisor must never
    vouch for a lock on a wrong line (07 P1-20): telemetry line_confident and the engine's wrong-line metric are never
    both set. Before the review fixes (FSR prior without window A, no drive-law exponents, ambiguity 1.5, no lock
    verification) three rings of worlds 2 and 4 were confidently wrong. Worlds 1 and 3 end with every ring on its
    line; the others are hard (monitor ADC clipping at the peak, P_FSR ~40 mW so a map cannot span an FSR, window-A
    pairs 1 nm apart, strong self-heating with a 10 % ms tail): their rings are flagged, not vouched for."""
    cfg = load_g26(knowledge=KNOW)
    ctrl = WdmSupervisor()
    B = 6
    res = _run(cfg, ctrl, 30e-3, batch=B, uncertainty="sample", variation="sample", seeds=7)
    wrong = np.asarray(res.metrics["wrong_line_final"], bool)
    tel = ctrl.telemetry()
    confident = tel["line_confident"] > 0.5
    locked = ctrl.fsm.state == ST_LOCK
    for b in range(B):
        print(f"\nworld {b}: map ok {ctrl.map_ok[b].astype(int).tolist()}, locked {locked[b].astype(int).tolist()}, "
              f"confident {confident[b].astype(int).tolist()}, wrong line {wrong[b].astype(int).tolist()}, "
              f"supply-limited {ctrl.supply_limited[b].astype(int).tolist()}, events {ctrl.map_events[b].tolist()}")
    assert not np.any(confident & wrong)
    assert not np.any(wrong[1]) and not np.any(wrong[3])
    print(f"\nrings on their line {int((~wrong).sum())}/{wrong.size}, vouched {int(confident.sum())}")


def test_online_barrel_shift_when_the_chip_cools():
    """PAT-1 online (step 706): the bank locks on the declared map with supply headroom (0.16 A), then the base
    temperature falls by 40 K over 5 ms; every ring needs ~2.6 nm more heater tuning, the bank's estimated supply
    current passes the budget (0.95 x 0.16 A) and the supervisor rotates the map by one channel towards lower power,
    re-acquiring ring by ring (tx.assignment bound to the controller)."""
    from mrm.scenarios import generators as g
    cfg = load_g26(overrides=["controller_interface.bind.tx.assignment=controller",
                              "array.heater_supply_current_max_a=0.16"], knowledge=KNOW)
    cool = g.Scenario(name="cool", components=(g.pwl("thermal.t_base_k", [0.0, 30e-3, 35e-3],
                                                     [298.15, 298.15, 258.15], absolute=True, label="cool"),))
    ctrl = WdmSupervisor(policy="cyclic")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = runner.run(cfg, controller=ctrl, scenario=cool, duration_s=55e-3, batch=1, uncertainty="nominal",
                         variation="off", reset_protocol="branch_low", record=[("p_heater_w", 100, "max")])
    s = int(ctrl.shift[0])
    print(f"\nonline PAT-1: shift {s}, lines {ctrl.assigned[0].tolist()}, phase {ctrl.phase[0]}, states "
          f"{ctrl.fsm.state[0].tolist()}, wrong {res.metrics['wrong_line_final'][0].astype(int).tolist()}")
    assert s == 1
    np.testing.assert_array_equal(ctrl.assigned[0], (np.arange(8) + 1) % 8)
    assert ctrl.phase[0] == PH_TRACK and np.all(ctrl.fsm.state[0] == ST_LOCK)
    assert not np.any(res.metrics["wrong_line_final"])
