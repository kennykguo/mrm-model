"""Checks the Truth-fed oracle baseline (mrm/validation/oracle.py), the reference every controller is scored against.

1. The registry refuses it (by class, by "module:Class" path); it is never in ``available()``. A controller may not
   read Truth; only the oracle does.
2. Its target is the OMA-optimal OPEN-LOOP-STABLE detuning: on the self-locked side, loop gain < 1, and no stable
   point on a fine grid has more OMA. The unrestricted optimum is reported next to it. In g26 nominal the blue-side
   OMA peak (about 92 % of the oracle OMA) is open-loop unstable, so the oracle sits on the red side.
3. It holds every ring at the target on the ReferencePlant: steady, through the A2 808 degC/s window and through a
   laser power step (it re-reads the detuning from Truth, so the power step does not move it). The retarget at the
   power step does not kick the heater through the rate of d.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from mrm.control import models
from mrm.control import registry
from mrm.scenarios import registry as scenario_registry
from mrm.scenarios.generators import Scenario, step
from mrm.validation import oracle

from test_ctrl_c_harness import build_rig, hold, load_cfg, lock_metrics, realize, run_loop


@pytest.fixture(scope="module")
def cfg():
    return load_cfg()


def test_registry_refuses_the_oracle():
    assert oracle.Oracle.truth_fed is True
    with pytest.raises(registry.RegistryError, match="Truth-fed"):
        registry.register("oracle", oracle.Oracle)
    with pytest.raises(registry.RegistryError, match="Truth-fed"):
        registry.create("mrm.validation.oracle:Oracle")
    assert "oracle" not in registry.available()


def test_target_is_the_stable_oma_optimum(cfg):
    pp = realize(cfg, 1)
    tgt = oracle.oracle_target(pp, None, cfg=cfg)
    assert np.all(tgt.detuning_m > 0)                    # resonance red of the laser: the self-locked side
    assert np.all(tgt.loop_gain < 1.0)
    d, oma, lg, _, _ = models.oma_curve(pp, 0, np.linspace(-300e-12, 400e-12, 141), cond=models.conditions_from_config(
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
    assert np.all(np.abs(oracle.oma_penalty_db(tgt.oma_w, tgt.oma_w)) < 1e-12)
    assert np.all(oracle.oma_penalty_db(0.5 * tgt.oma_w, tgt.oma_w) > 3.0)


def test_oracle_holds_steady_and_through_a2(cfg):
    rig = build_rig(cfg)
    hold(rig, 30)
    tr = run_loop(rig, oracle.Oracle(), 1200)
    m = lock_metrics(tr.detuning_m, rig.targets_m, oma_w=tr.oma_w, oracle_oma_w=rig.oracle_oma_w, settle_ticks=100)
    print("\n[oracle steady]", m)
    assert m["max_pm"] < 0.3 and m["oma_penalty_db"] < 1e-3
    sc = scenario_registry.create("a2")
    rig = build_rig(cfg, scenario=sc, k0=20000, aggressor_presettle_ticks=6000)
    hold(rig, 30)
    tr = run_loop(rig, oracle.Oracle(), 2000)
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
    orc = oracle.Oracle()
    tr = run_loop(rig, orc, 1200)
    assert orc.n_retargets >= 1
    err = (tr.detuning_m - orc.det_target[None]) * 1e12
    moved = (orc.det_target - rig.targets_m) * 1e12
    print("\n[oracle power step] target moved pm", np.round(moved[0], 2), " max |err| after 0.2 ms:",
          np.round(np.max(np.abs(err[600:]), axis=0), 3))
    assert np.max(np.abs(err[600:])) < 0.5
    assert np.max(np.abs(moved)) < 0.01


def test_retarget_edge_has_no_rate_kick(cfg):
    """A retarget re-solves the target and the true linear model, so d changes reference (target, operating point,
    s, c_R). At the +0.5 dB power step d jumps by about 6 pm, but the true disturbance does not change. That jump is
    not a rate of d, so the retarget edge uses rate 0 (the first-edge rule): its command is the command of a copy of
    the oracle that has no rate history. Before the fix the jump went into the rate term (about 700 pm/ms) and kicked
    the heater by about 190 codes."""
    class TwinOracle(oracle.Oracle):
        """At an edge that retargets, also runs a copy with no rate history and keeps both commands in ``pairs``."""

        def step(self, obs, truth):
            twin_codes = None
            if any(truth.tick - self._last_retarget[b] >= self.retarget_min_ticks
                   and self._conditions_moved(self.design_conditions[b], truth.conditions[b])
                   for b in range(len(self.designs))):
                shared = {id(x): x for x in (self.pp, self.cfg, self.spec, self._pp0, self._cfg0) if x is not None}
                twin = copy.deepcopy(self, shared)
                twin._d_prev_pm[:] = np.nan
                twin_codes = oracle.Oracle.step(twin, obs, truth)["tx.heater"]
            out = super().step(obs, truth)
            if twin_codes is not None:
                self.pairs.append((out["tx.heater"], twin_codes))
            return out

    sc = Scenario(name="pstep", components=(step("laser.power_delta_db", 4e-4, 0.5),))
    rig = build_rig(cfg, scenario=sc)
    hold(rig, 30)
    orc = TwinOracle()
    orc.pairs = []
    run_loop(rig, orc, 600)
    assert orc.n_retargets >= 1 and len(orc.pairs) == orc.n_retargets
    for codes, twin_codes in orc.pairs:
        np.testing.assert_array_equal(codes, twin_codes)


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
    orc = oracle.Oracle()
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
    tgt = oracle.oracle_target(pp, [rig.conditions(b) for b in range(pp.batch)])
    np.testing.assert_array_equal(tgt.reachable, orc.reach)
