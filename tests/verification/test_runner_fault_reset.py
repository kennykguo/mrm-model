"""Checks the runner's Python path after a scenario controller reset (the fault signal ``fault.controller_reset``).

A finding of the hardware review, fixed by the integrator on 2026-10-10. ``CompiledController.reset`` allocates a
new held-command array. The runner's Python path clips the controller's held command after every step (as the kernel
does), so after a reset it must clip the NEW array. Before the fix it kept clipping the old one, and a compiled
controller whose commands leave the channel range built its next step on an unclipped value. The kernel path and the
Python path must stay bit-identical through the reset. ~5 s.
"""
from __future__ import annotations

import warnings

import numba as nb
import numpy as np

from mrm import params
from mrm.control import api
from mrm.core.channels import ActId
from mrm.scenarios import generators
from mrm.scenarios import registry as scenario_registry
from mrm.sim import runner
from mrm.sim.metrics import RecordSpec

_HEATER_ID = int(ActId.TX_HEATER)   # actuator id of the heater channel


@nb.njit
def _swing_init(spec_i64, spec_f64):
    """Compiled-controller init: params p = [n_rings, offset of the heater channel in act], one step counter."""
    p = np.zeros(2, np.int64)
    p[0] = spec_i64[api.SPEC_I_N_RINGS]
    p[1] = api.act_offset(spec_i64, _HEATER_ID)
    return p, np.zeros(0), np.zeros(1, np.int64), np.zeros(0)


@nb.njit
def _swing_step(p, pf, s, sf, obs, act):
    """+30000 codes per step for 3 steps (past the 65535 rail), then -1000 per step: each step builds on the held
    command, which the plant has clipped."""
    d = 30000 if s[0] < 3 else -1000
    for i in range(p[0]):
        act[p[1] + i] = act[p[1] + i] + d
    s[0] += 1


def _scenario(t_s: float):
    """The g26 scenario plus a 3 us controller-reset pulse at t_s (s)."""
    comp = generators.pwl("fault.controller_reset", [0.0, t_s, t_s, t_s + 3e-6, t_s + 3e-6, 1.0], [0, 0, 1, 1, 0, 0],
                          absolute=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = scenario_registry.from_config(params.load("g26"))
    return base + generators.Scenario(name="controller_reset", components=(comp,))


def test_python_path_clips_the_held_command_after_a_controller_reset():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load("g26")
    rec = [RecordSpec("heater_code", 1, "last"), RecordSpec("detuning_m", 1, "last")]
    kw = dict(duration_s=0.4e-3, record=rec, uncertainty="nominal", variation="off", seeds=0,
              heater_code="locked_steady_state", settle=True)
    out = {}
    for path in ("compiled", "python"):
        ctrl = api.CompiledController("swing", _swing_init, _swing_step)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[path] = runner.run(cfg, ctrl, _scenario(150.5e-6), path=path, **kw)
    h = np.asarray(out["compiled"].trace["heater_code@1:last"]["value"])[:, 0, :]
    # the swing reaches the rail twice: before and after the reset (the reset restarts its 3 up-steps)
    at_rail = np.nonzero(h.max(axis=1) == 65535)[0]
    assert at_rail.size and at_rail.min() < 150 and at_rail.max() > 160, at_rail[[0, -1]]
    for key in out["compiled"].trace:
        a = np.asarray(out["compiled"].trace[key]["value"])
        b = np.asarray(out["python"].trace[key]["value"])
        assert np.array_equal(a, b), key
