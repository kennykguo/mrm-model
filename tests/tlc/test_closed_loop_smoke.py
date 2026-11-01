"""``tlc`` closed loop on the g26 engine: a short warm-start run through mrm.sim.runner on the Python controller
path. With the stub core and no dither the heater codes must stay at the warm-start codes, and the telemetry log
must hold one row per 100 us core step, and
the detuning must stay put. Slow: the first run compiles the plant kernel (minutes; cached after)."""
from __future__ import annotations

import numpy as np
import pytest

from mrm.control.tlc.controller import TlcController
from mrm.sim import runner


@pytest.mark.slow
def test_stub_holds_the_warm_codes_on_g26():
    ctrl = TlcController(core="stub", dither_amp_u=0.0)
    res = runner.run("g26", ctrl, duration_s=1e-3, batch=1, uncertainty="nominal", variation="off",
                     heater_code="quasi_static_reachable", settle=True, path="python")
    applied = np.asarray(res.truth_final.fields["heater_code"])
    warm = ctrl.heater_law.code_from_u(ctrl.cores[0].u_hold)
    assert np.array_equal(applied[0], warm)
    assert len(ctrl.telemetry_log) == 10
    # nothing disturbs a static world, so a held heater holds the detuning (the default lock reference is the
    # line itself, ~125 pm away, so lock flags say nothing here)
    spread = np.asarray(res.metrics["max_abs_detuning_pm"]) - np.abs(np.asarray(res.metrics["mean_detuning_pm"]))
    assert np.all(spread < 0.1)
