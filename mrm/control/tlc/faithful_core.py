"""The vendored ECE496 core (``mrm.control.tlc.core``) behind the adapter's core interface (see stub_core.py).

One instance per world. Parameters come from :func:`mrm.control.tlc.priors.g26_priors` (nominal design only),
computed once per ControllerSpec. Differences from the source's own driver (``thermal_control/sim.py``):

- warm start: the source passes the TRUE detuning to ``TLC.warm_start``. A controller cannot know it, so the
  initial estimate is the core's own setpoint (B6 targets): mrm's warm start parks each ring at its lock point,
  and the UKF starts there with the source's variance and corrects within its first windows.
- the core's ``last`` telemetry only exists after its first step; before that the wrapper reports the start
  state (modes, command) with zero estimates.
"""
from __future__ import annotations

import numpy as np

from mrm.control.api import ControllerSpec
from mrm.control.tlc.core import TLC, BaselinePI
from mrm.control.tlc.priors import TlcPriors, g26_priors

KINDS = {"tlc": TLC, "pi": BaselinePI}


_PRIORS: dict[str, TlcPriors] = {}
_PRIORS_MAX = 8


def priors_for(spec: ControllerSpec) -> TlcPriors:
    """g26_priors(spec), cached by the spec's fingerprint (shared by the B worlds and repeated resets)."""
    key = spec.fingerprint()
    if key not in _PRIORS:
        if len(_PRIORS) >= _PRIORS_MAX:
            _PRIORS.clear()
        _PRIORS[key] = g26_priors(spec)
    return _PRIORS[key]


class FaithfulCore:
    def __init__(self, spec: ControllerSpec, *, ts: float, n_out: int, kind: str = "tlc"):
        pr = priors_for(spec)
        if abs(ts - pr.cp.ts) > 1e-12:
            raise ValueError(f"faithful core: the adapter's core period {ts} s must be the source's {pr.cp.ts} s")
        if n_out != int(round(pr.cp.ts * pr.cp.out_rate)):
            raise ValueError(f"faithful core: {n_out} samples per core step, the source plans "
                             f"{int(round(pr.cp.ts * pr.cp.out_rate))}")
        self.priors = pr
        self.ctl = KINDS[kind](pr.pp, pr.cp)
        self.n = pr.cp.n_rings
        self.last = self._start_last(np.zeros(self.n))

    def warm_start(self, u0: np.ndarray) -> None:
        self.ctl.warm_start(np.asarray(u0, dtype=np.float64), np.asarray(self.ctl.sp.targets, dtype=np.float64))
        self.last = self._start_last(u0)

    def cold_start(self) -> None:
        self.ctl.cold_start()
        self.last = self._start_last(np.zeros(self.n))

    def first_plan(self, t0: float) -> np.ndarray:
        seq, _, _ = self.ctl.out.produce(self.ctl.u_cmd, t0, self.ctl.dither, np.ones(self.n))
        return seq

    def step(self, pd: np.ndarray, t_block_start: float) -> np.ndarray:
        seq = self.ctl.step(np.asarray(pd, dtype=np.float64), t_block_start)
        self.last = self.ctl.last
        return seq

    def _start_last(self, u: np.ndarray) -> dict:
        z = np.zeros(self.n)
        return dict(delta_hat=z, rate_hat=z, sigma=z, modes=np.asarray(self.ctl.modes).copy(), u=np.array(u, float),
                    amp=z, squelch=np.zeros(self.n, dtype=bool), fault=False, eff=np.ones(self.n))
