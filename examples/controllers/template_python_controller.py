"""Template: a Python controller on the mrm controller API (07 P1-19), vectorized over (B, N).

Copy this file, rename the class and the registry name, and replace the update law in ``step``. The example is a
plain integral lock of the drop-port monitor to a fixed fraction of a known peak code, on the stable (red) side
(07 P0-1). For the full version (hand-off map of the peak and the slope, fall-off detection, gain schedule) see
``mrm/control/baselines/pi_lock.py``.

Rules (07 P1-1): read only ``obs`` (ADC codes, sample ticks, temperature codes, applied codes) and ``spec``; model
knowledge only from ``spec.nominal`` for the tokens in ``spec.knowledge`` (``NominalModel.from_spec(spec)``); return
integer codes; never touch Truth.

Run it: ``mrm.sim.runner.run(cfg, controller=MyIntegralLock(peak_codes=58000), duration_s=5e-3, ...)``, or use the
registry name after ``import`` (``controller: {name: my_integral_lock, params: {...}}`` in an experiment file).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mrm.control.api import BaseController, ControllerSpec, Observation
from mrm.control.baselines.pi_lock import block_inputs
from mrm.control.registry import register


@register("my_integral_lock")
class MyIntegralLock(BaseController):
    """u += k_i (y - f M_pk) / M_pk: on the red side dy/du < 0, so a monitor above the set-point raises the heater."""
    name = "my_integral_lock"

    @dataclass
    class Params:
        peak_codes: float = 58000.0     # peak monitor code (measure it: pi_lock's MAP, or a factory table)
        fraction: float = 0.7           # set-point as a fraction of the peak
        k_i_codes: float = 400.0        # heater codes per step per unit normalized error (tune on your plant!)
        latency_ticks: int = 0

    def reset(self, spec: ControllerSpec) -> None:
        super().reset(spec)
        self.latency_ticks = int(self.params.latency_ticks)
        self.u = None                                  # heater code, float (B, N)
        self.y = self.zeros("B", "N")                  # last block mean of the monitor (B, N)
        self.lo, self.hi = float(spec.heater_dac.code_min), float(spec.heater_dac.code_max)

    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        p = self.params
        if self.u is None:                             # start from the latched DAC register (warm hand-off)
            self.u = np.asarray(obs["tx.heater.applied"], dtype=float).copy()
        self.y, ok = block_inputs(obs, self.y)         # mean of the samples since the last step; ok = new & valid
        e = (self.y - p.fraction * p.peak_codes) / p.peak_codes
        self.u = np.where(ok, np.clip(self.u + p.k_i_codes * e, self.lo, self.hi), self.u)
        return {"tx.heater": np.rint(self.u).astype(np.int64)}

    def telemetry(self) -> dict[str, np.ndarray]:
        p = self.params
        return {"integrator": self.u.copy() if self.u is not None else self.zeros("B", "N"),
                "locked": np.abs(self.y / p.peak_codes - p.fraction) < 0.02}
