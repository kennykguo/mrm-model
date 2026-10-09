"""Ones-density steps (Sun et al. VLSI 2015: all-0 / all-1 and balanced data; 07 P0-11) on the bit-statistics lock
and on a plain monitor lock at the same operating point, closed loop on the g26 reference plant.

A monitor lock reads y = rho mu1 + (1 - rho) mu0, so a density step moves its lock point by (rho - 1/2) |mu1 - mu0| /
|dy/ddelta| (04 C3, C6). The bit-statistics lock holds the EYE CENTRE with the ones-density canceller (Sun's
self-heating canceller in this form), so only a transient remains. Reported: the plateau detuning offsets (last
0.4 ms of each plateau against the balanced-data value before the steps) for both controllers.
"""
from __future__ import annotations

import numpy as np
import pytest

from mrm.control.baselines.bitstat_lock import BitStatLock
from mrm.control.baselines.ramp_acquire import RampAcquire, ST_LOCK
from mrm.scenarios import library as lib

from test_ctrl_b_plant import CtrlBPlant, load_g26, make_spec, realize, run_closed_loop

KNOW = ["line_grid", "ring_to_line", "heater_nominal", "optics_nominal", "dlambda_dt_nominal"]
LINE = 7
T0, STEP = 2.5e-3, 1.0e-3                 # levels 0.5 (2.5-3.5 ms), 1.0, 0.5, 0.0, 0.5


def _plateaus(rec):
    d = rec.det_m[:, 0, 0] * 1e12
    t = rec.ticks * 1e-6

    def mean(a, b):
        return float(d[(t > a) & (t < b)].mean())
    ref = mean(T0 - 0.4e-3, T0 + STEP)
    p1 = mean(T0 + 2 * STEP - 0.4e-3, T0 + 2 * STEP)      # all ones
    p0 = mean(T0 + 4 * STEP - 0.4e-3, T0 + 4 * STEP)      # all zeros
    return ref, p1 - ref, p0 - ref


@pytest.fixture(scope="module")
def setup():
    cfg = load_g26(knowledge=KNOW)
    pp = realize(cfg, 1, nominal=True)
    spec = make_spec(cfg, 1)
    sc = lib.ones_density_steps(levels=(0.5, 1.0, 0.5, 0.0, 0.5), step_s=STEP, t0_s=T0, rings=[0])
    return cfg, pp, spec, sc


def test_bitstat_lock_rejects_density_steps_better_than_a_monitor_lock(setup):
    cfg, pp, spec, sc = setup
    n = int((T0 + 5 * STEP) / spec.tick_s)
    out = {}
    for name, ctrl in (("bitstat", BitStatLock(rings=[0])), ("monitor", RampAcquire(rings=[0], alpha=0.75))):
        plant = CtrlBPlant(pp, cfg, spec, scenario=sc)
        plant.assignment[0, 0] = LINE
        rec, _ = run_closed_loop(plant, ctrl, n)
        assert ctrl.fsm.state[0, 0] == ST_LOCK
        out[name] = _plateaus(rec)
        print(f"\n{name}: balanced {out[name][0]:.1f} pm; plateau offset all-ones {out[name][1]:+.1f} pm, "
              f"all-zeros {out[name][2]:+.1f} pm")
    bs, mon = out["bitstat"], out["monitor"]
    # the monitor lock moves by tens of pm on each plateau, in opposite directions for all-ones and all-zeros
    assert abs(mon[1]) > 10.0 and abs(mon[2]) > 10.0 and np.sign(mon[1]) != np.sign(mon[2])
    # the bit-statistics lock holds within a few pm on the plateaus
    assert abs(bs[1]) < 0.3 * abs(mon[1]) and abs(bs[2]) < 0.3 * abs(mon[2]), (bs, mon)
