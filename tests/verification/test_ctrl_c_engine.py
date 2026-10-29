"""Checks the model-based baselines through the compiled engine (mrm/sim/runner.py), when it is available; slow.

Read this first
---------------
The engine's kernel compiles once per numba cache (minutes on a loaded machine). The adapter is
``runner.run(cfg, controller=..., scenario=...)``: the Python controllers run on the engine's Python slow path (the
kernel stops at every edge), the compiled ones inside the kernel.

The engine's warm start (``heater_code``) begins from a cold thermal state with a laser ramp, which is not a lock.
So the lock-only compiled versions are checked against the Python ones on the ReferencePlant
(test_ctrl_c_compiled.py), and here only the acquisition flow runs end to end: ring 0, with only its own line lit.
The bank's WDM hazards are the supervisor's job (see test_ctrl_c_kalman_lqg.py::
test_cold_start_acquisition_single_line).
"""
from __future__ import annotations

import numpy as np
import pytest

from mrm.control import models
from mrm.control.baselines import kalman_lqg, mpc
from mrm.scenarios.generators import Scenario, const

from test_ctrl_c_harness import load_cfg

runner = pytest.importorskip("mrm.sim.runner")


@pytest.mark.slow
@pytest.mark.parametrize("make", [lambda: kalman_lqg.KalmanLQG(start="cold", ramp_step=80.0, probe_step=20.0),
                                  lambda: mpc.LinearMPC(start="cold", ramp_step=80.0, probe_step=20.0)],
                         ids=["kalman_lqg", "mpc"])
def test_cold_start_through_the_engine(make):
    cfg = load_cfg()
    sc = Scenario(name="line0", components=(const("laser.line_enable", 0.0, absolute=True,
                                                  elements=tuple(range(1, 8))),))
    ctrl = make()
    r = runner.run(cfg, controller=ctrl, scenario=sc, duration_s=16e-3, batch=1, uncertainty="nominal",
                   variation="off")
    st = ctrl.telemetry()["state"][0, 0]
    pp = r.plant.pp if hasattr(r.plant, "pp") else None
    det = float(r.truth_final.detuning_m[0, 0])
    if pp is not None:
        cond = models.conditions_from_config(pp, 0, cfg=cfg)
        cond = models.Conditions(**{**cond.__dict__, "p_chip_w": np.where(np.arange(pp.n_lines) == 0, cond.p_chip_w,
                                                                          0.0)})
        target = float(models.oma_optimal_detuning(pp, 0, cond=cond)[0][0])
    else:
        target = float(models.linearize(cfg).detuning0_m)
    print(f"\n[engine {ctrl.name}] state {st:.0f}, detuning {det * 1e12:.2f} pm vs target {target * 1e12:.2f} pm, "
          f"wall {r.wall_s:.1f} s, metrics excursions {np.asarray(r.metrics.get('excursion_count'))[0, 0]}")
    assert st == kalman_lqg.ST_LOCK
    assert abs(det - target) * 1e12 < 6.0
