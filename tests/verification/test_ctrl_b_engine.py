"""Controllers B on the compiled engine (mrm.sim.runner.run, Python controller path), through this small adapter:
Truth comes from the engine's records (lam_res_m, detuning_m) and its final Truth; the controllers are the same
objects the reference-plant tests use. Skipped when the engine is not importable (it is being written alongside).

Cross-checks against the reference bench (tests/verification/test_ctrl_b_plant.py, same g26 nominal world): the
ramp acquisition must end at the same red-side detuning; the WDM supervisor must end with every ring on its declared
line (engine metric wrong_line_final). The aggressor runs on the engine are in test_ctrl_b_ramp_aggressor.py (slow).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

runner = pytest.importorskip("mrm.sim.runner")

from mrm.control.baselines.bitstat_lock import BitStatLock
from mrm.control.baselines.ramp_acquire import RampAcquire, ST_LOCK
from mrm.control.baselines.wdm_supervisor import PH_TRACK, WdmSupervisor
from mrm.scenarios import library as lib
from mrm.units import C0

from test_ctrl_b_plant import load_g26

KNOW = ["line_grid", "ring_to_line", "heater_nominal", "optics_nominal", "dlambda_dt_nominal"]
BENCH_RING0_PM = 148.3          # tests/verification/test_ctrl_b_ramp.py::test_steady_lock_metrics (reference bench)


class WithAssignment:
    """Adapter: forwards a controller and declares ``tx.assignment`` (the line each ring is meant to be on), so the
    engine's Truth (detuning of the order nearest the DECLARED line, its metrics) refers to that line. Needs
    controller_interface.bind.tx.assignment = controller."""

    def __init__(self, ctrl, assignment):
        self.ctrl = ctrl
        self.assignment = np.asarray(assignment, dtype=np.int64)
        self.name = getattr(ctrl, "name", "wrapped")
        self.latency_ticks = int(getattr(ctrl, "latency_ticks", 0))

    def reset(self, spec):
        self.ctrl.reset(spec)
        self.B = spec.n_batch

    def step(self, obs):
        out = dict(self.ctrl.step(obs) or {})
        out["tx.assignment"] = np.repeat(self.assignment[None, :], self.B, axis=0)
        return out

    def telemetry(self):
        return self.ctrl.telemetry()


def run_engine(ctrl, duration_s, scenario=None, record=(("detuning_m", 10, "mean"),), cfg=None, assignment=None,
               **kw):
    """Adapter: one engine run of the g26 nominal world with a Python controller (reset protocol branch_low, as the
    reference bench: lasers on at t = 0, heaters at code 0). With ``assignment`` the controller is wrapped in
    :class:`WithAssignment` and tx.assignment is bound to it."""
    if assignment is not None:
        cfg = load_g26(overrides=["controller_interface.bind.tx.assignment=controller"], knowledge=KNOW)
        ctrl = WithAssignment(ctrl, assignment)
    cfg = load_g26(knowledge=KNOW) if cfg is None else cfg
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return runner.run(cfg, controller=ctrl, scenario=lib.static() if scenario is None else scenario,
                          duration_s=duration_s, batch=1, uncertainty="nominal", variation="off",
                          record=list(record), reset_protocol="branch_low", **kw)


def detuning_pm(res, stat="mean") -> np.ndarray:
    """(S, N) recorded detuning to the declared line (pm)."""
    key = [k for k in res.trace if k.startswith("detuning_m") and k.endswith(stat)][0]
    return res.trace[key]["value"][:, 0, :] * 1e12


def test_ramp_acquire_on_the_engine_matches_the_bench():
    ctrl = RampAcquire(rings=[0])
    res = run_engine(ctrl, 3e-3, assignment=[7, 1, 2, 3, 4, 5, 6, 7])
    assert ctrl.fsm.state[0, 0] == ST_LOCK
    d = detuning_pm(res)[:, 0]
    tail = d[-100:]
    print(f"\nengine: ring 0 locked at {tail.mean():.2f} pm from line 7 (bench {BENCH_RING0_PM} pm), jitter "
          f"{tail.std():.3f} pm, wall {res.wall_s:.1f} s for 3 ms")
    assert np.ptp(tail) < 1.0
    assert abs(tail.mean() - BENCH_RING0_PM) < 10.0


@pytest.mark.slow
def test_wdm_supervisor_cold_start_on_the_engine():
    ctrl = WdmSupervisor()
    res = run_engine(ctrl, 24e-3)
    assert ctrl.phase[0] == PH_TRACK
    assert np.all(ctrl.fsm.state[0] == ST_LOCK)
    m = res.metrics
    assert not np.any(m["wrong_line_final"]), m["wrong_line_final"]
    d = detuning_pm(res)[-100:]
    print(f"\nengine WDM cold start: declared-line detuning {np.round(d.mean(axis=0), 1).tolist()} pm, jitter "
          f"{np.round(d.std(axis=0), 3).tolist()} pm, heater range used {m['heater_range_used_codes'][0].tolist()}, "
          f"wall {res.wall_s:.1f} s")
    assert np.all((d.mean(axis=0) > 80.0) & (d.mean(axis=0) < 260.0))
    assert np.all(np.ptp(d, axis=0) < 2.0)


def test_bitstat_lock_on_the_engine():
    ctrl = BitStatLock(rings=[0])
    res = run_engine(ctrl, 3.5e-3, assignment=[7, 1, 2, 3, 4, 5, 6, 7])
    assert ctrl.fsm.state[0, 0] == ST_LOCK
    d = detuning_pm(res)[:, 0]
    tail = d[-80:]
    print(f"\nengine: bitstat lock at {tail.mean():.1f} pm from line 7 (eye maximum ~90 pm on the bench), eye "
          f"{ctrl.eye[0, 0]:.0f} codes, alpha {ctrl.alpha_eye[0, 0]:.3f}")
    assert 60.0 < tail.mean() < 125.0


@pytest.mark.slow
def test_bitstat_lock_recovers_from_a_laser_power_step():
    """+10 % laser power at 4 ms. The monitor lock's set-point is an absolute code from the old peak, so it is
    pushed red by (dP/P) y / |dy/ddelta| (04 C3) and stays there; the bit-statistics lock holds the eye ratio
    D / (c - y_bg), which does not depend on the optical power, and returns towards its detuning."""
    import math
    from mrm.scenarios import generators as g
    step = g.Scenario(name="pstep", components=(g.step("laser.power_delta_db", 4e-3, 10 * math.log10(1.1)),))
    out = {}
    for name, ctrl in (("bitstat", BitStatLock(rings=[0])), ("monitor", RampAcquire(rings=[0], alpha=0.75))):
        res = run_engine(ctrl, 14e-3, scenario=step, assignment=[7, 1, 2, 3, 4, 5, 6, 7])
        d = detuning_pm(res)[:, 0]
        t = res.trace[[k for k in res.trace if k.startswith("detuning_m")][0]]["t_s"]
        before = float(d[(t > 3.0e-3) & (t < 4.0e-3)].mean())
        late = float(d[t > 12.5e-3].mean())
        out[name] = (before, late)
        print(f"\n{name}: detuning {before:.1f} pm before the step, {late:.1f} pm 8.5 ms after")
    assert abs(out["bitstat"][1] - out["bitstat"][0]) < 0.5 * abs(out["monitor"][1] - out["monitor"][0])


def test_ramp_lock_recovers_from_a_line_failover_gap():
    """[B2] laser failover with a 1 ms dark gap on ring 0's line (line 7) at 3 ms, then the spare laser at +10 GHz and
    -1 dB (mrm.scenarios.library.laser_failover). In the dark the lock is lost and the self-heating goes away (the
    resonance moves ~1 nm blue); the re-search covers only this resonance, in a window moved by the drift the loop
    had tracked, and FAIL re-arms after fail_rearm_s. Ring 0 must end locked on the red side of line 7 again. Before
    the review the re-search ran in the full window: a search still climbing when the light came back took the next
    resonance up (a foreign line), and a FAIL was final."""
    sc = lib.laser_failover(line=7, t_fail_s=3e-3, gap_s=1e-3)
    ctrl = RampAcquire(rings=[0])
    res = run_engine(ctrl, 14e-3, scenario=sc, assignment=[7, 1, 2, 3, 4, 5, 6, 7])
    f = ctrl.fsm
    d = detuning_pm(res)[:, 0]
    t = res.trace[[k for k in res.trace if k.startswith("detuning_m")][0]]["t_s"]
    before = float(d[(t > 2.0e-3) & (t < 3.0e-3)].mean())
    tail = d[t > 13e-3]
    print(f"\nfailover gap: {before:.1f} pm before, {tail.mean():.1f} pm after (p-p {np.ptp(tail):.2f} pm), state "
          f"{f.state[0, 0]}, retries {f.retries[0, 0]}, jumps {f.jumps[0, 0]}")
    assert f.state[0, 0] == ST_LOCK
    assert 80.0 < tail.mean() < 260.0 and np.ptp(tail) < 1.0, tail
