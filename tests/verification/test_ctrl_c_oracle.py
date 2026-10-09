"""The Truth-fed oracle (mrm/validation/oracle.py; 07 P1-1, P0-1 item 7, P0-8).

1. The registry refuses it (by class, by "module:Class" path); it is never in ``available()``.
2. Its target is the OMA-optimal OPEN-LOOP-STABLE detuning: on the self-locked side, loop gain < 1, no stable point
   on a fine grid has more OMA; the unrestricted optimum is reported next to it (07 P0-1 item 7). In g26 nominal the
   blue-side OMA peak (about 92 % of the oracle OMA) is open-loop unstable, so the oracle sits on the red side.
3. It holds every ring at the target on the ReferencePlant: steady, through the A2 808 degC/s window and through a
   laser power step (it re-reads the detuning from Truth, so the power step does not move it).
"""
from __future__ import annotations

import numpy as np
import pytest

from mrm.control import models as M
from mrm.control import registry
from mrm.scenarios import registry as sreg
from mrm.scenarios.generators import Scenario, step
from mrm.validation import oracle as O

from test_ctrl_c_harness import build_rig, hold, load_cfg, lock_metrics, realize, run_loop


@pytest.fixture(scope="module")
def cfg():
    return load_cfg()


def test_registry_refuses_the_oracle():
    assert O.Oracle.truth_fed is True
    with pytest.raises(registry.RegistryError, match="Truth-fed"):
        registry.register("oracle", O.Oracle)
    with pytest.raises(registry.RegistryError, match="Truth-fed"):
        registry.create("mrm.validation.oracle:Oracle")
    assert "oracle" not in registry.available()


def test_target_is_the_stable_oma_optimum(cfg):
    pp = realize(cfg, 1)
    tgt = O.oracle_target(pp, None, cfg=cfg)
    assert np.all(tgt.detuning_m > 0)                    # resonance red of the laser: the self-locked side
    assert np.all(tgt.loop_gain < 1.0)
    d, oma, lg, _, _ = M.oma_curve(pp, 0, np.linspace(-300e-12, 400e-12, 141), cond=M.conditions_from_config(
        pp, 0, cfg=cfg))
    best_stable = np.max(np.where(lg < 1.0, oma, 0.0), axis=0)
    # the refinement moves every ring at once, so a ring's OMA also feels the others' insertion loss (1e-5 level)
    assert np.all(tgt.oma_w[0] >= best_stable * (1 - 1e-4))
    assert np.all(tgt.oma_unrestricted_w >= tgt.oma_w * (1 - 1e-9))
    # g26 nominal: the best OMA on the blue (open-loop unstable) side is lower than on the self-locked red side
    blue = d < 0
    best_blue = np.max(np.where(blue[:, None], oma, 0.0), axis=0)
    i_b = np.argmax(np.where(blue[:, None], oma, 0.0), axis=0)
    assert np.all(lg[i_b, np.arange(pp.n_rings)] > 1.0)        # the blue-side OMA peak is open-loop unstable
    print("\nbest blue-side OMA / oracle OMA", np.round(best_blue / tgt.oma_w[0], 3))
    print("\noracle target pm", np.round(tgt.detuning_m[0] * 1e12, 1), "relOMA dB", np.round(tgt.reloma_db[0], 2),
          "loop gain", np.round(tgt.loop_gain[0], 2))
    assert np.all(np.abs(O.oma_penalty_db(tgt.oma_w, tgt.oma_w)) < 1e-12)
    assert np.all(O.oma_penalty_db(0.5 * tgt.oma_w, tgt.oma_w) > 3.0)


def test_oracle_holds_steady_and_through_a2(cfg):
    rig = build_rig(cfg)
    hold(rig, 30)
    tr = run_loop(rig, O.Oracle(), 1200)
    m = lock_metrics(tr.detuning_m, rig.targets_m, oma_w=tr.oma_w, oracle_oma_w=rig.oracle_oma_w, settle_ticks=100)
    print("\n[oracle steady]", m)
    assert m["max_pm"] < 0.3 and m["oma_penalty_db"] < 1e-3
    sc = sreg.create("a2")
    rig = build_rig(cfg, scenario=sc, k0=20000, aggressor_presettle_ticks=6000)
    hold(rig, 30)
    tr = run_loop(rig, O.Oracle(), 2000)
    m = lock_metrics(tr.detuning_m, rig.targets_m, oma_w=tr.oma_w, oracle_oma_w=rig.oracle_oma_w, settle_ticks=100)
    print("[oracle A2]", m)
    assert m["max_pm"] < 0.6 and m["excursions"] == 0


def test_oracle_follows_a_laser_power_step(cfg):
    """The oracle reads the detuning and the conditions from Truth: after a +0.5 dB power step (which moves a monitor
    set-point lock by ~25 pm) it re-solves its target and its true linear model in the new conditions and holds the
    target within the self-heating transient. The OMA-optimal detuning itself does not move: at a fixed detuning the
    ring is linear in the input power, so OMA(delta) only scales (self-heating changes the heater power that holds the
    point and the loop gain, not the optics at that point)."""
    sc = Scenario(name="pstep", components=(step("laser.power_delta_db", 4e-4, 0.5),))
    rig = build_rig(cfg, scenario=sc)
    hold(rig, 30)
    orc = O.Oracle()
    tr = run_loop(rig, orc, 1200)
    assert orc.n_retargets >= 1
    err = (tr.detuning_m - orc.det_target[None]) * 1e12
    moved = (orc.det_target - rig.targets_m) * 1e12
    print("\n[oracle power step] target moved pm", np.round(moved[0], 2), " max |err| after 0.2 ms:",
          np.round(np.max(np.abs(err[600:]), axis=0), 3))
    assert np.max(np.abs(err[600:])) < 0.5
    assert np.max(np.abs(moved)) < 0.01


def test_oracle_holds_sampled_worlds(cfg):
    """The oracle on worlds drawn from the uncertainty set (B = 4, LHS + per-device variation; its designs, lag
    maps and delay lines are per world, since the sampled transport delay and thermal block can differ). In this
    draw some targets are unreachable (codes past the DAC range or the driver's compliance, a bank above the 200 mW
    supply): the oracle marks them (``reachable``) and holds the code nearest to them instead of crashing (the
    regulator equations are singular without heater authority) or winding up. Every world whose targets are all
    reachable is held to well under a pm."""
    pp = realize(cfg, 4, uncertainty="lhs", variation="sample", seeds=11)
    rig = build_rig(cfg, pp=pp)
    feas = rig.feasible_worlds()
    hold(rig, 30)
    orc = O.Oracle()
    tr = run_loop(rig, orc, 800)
    err = (tr.detuning_m - orc.det_target[None]) * 1e12
    worst = np.max(np.abs(err[200:]), axis=(0, 2))
    ok = orc.reach.all(axis=1)
    print(f"\n[oracle sampled worlds] harness-feasible {feas}, oracle-reachable rings\n{orc.reach}\nmax |err| per "
          f"world after 0.2 ms (pm) {np.round(worst, 3)}, delay lines "
          f"{[d.nc - d.models[0].Ac.shape[0] for d in orc.designs]}")
    np.testing.assert_array_equal(ok, feas)                  # the two definitions agree on this draw
    assert ok.sum() >= 1 and (~ok).any()                     # the draw exercises both cases
    assert np.all(worst[ok] < 0.6)
    codes = tr.heater_codes[-1]
    lo, hi = rig.spec.heater_dac.code_min, rig.spec.heater_dac.code_max
    for b, n in zip(*np.nonzero(~orc.reach)):                # unreachable: held at the code nearest the target
        assert codes[b, n] == int(np.clip(np.rint(orc.designs[b].u0[n]), lo, hi))
    tgt = O.oracle_target(pp, [rig.conditions(b) for b in range(pp.batch)])
    np.testing.assert_array_equal(tgt.reachable, orc.reach)
