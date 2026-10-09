"""Template: a scalar per-ring algorithm vectorized by the PerRing adapter (07 P1-19).

``PerRing(factory)`` builds one instance per (world b, ring n) at reset; each instance sees a RingObservation where
(B, N) channels are scalars and the monitor sample block is trimmed to its valid samples. Write the algorithm as if
there were one ring. Slower than a vectorized controller (a Python call per ring and step), fine for prototyping.

The example is a perturb-and-observe peak seeker (04 C2) in ~30 lines; the full version with fall-off recovery is
``mrm/control/baselines/extremum_seeking.py`` (ExtremumSeekingPO). Remember 07 P0-1: under strong self-heating the
peak sits next to the fold, so a peak seeker falls off it repeatedly (measured in tests/verification/test_ctrl_a_es.py).
"""
from __future__ import annotations

import numpy as np

from mrm.control.api import PerRing, RingContext, RingObservation


class PerturbObserveRing:
    """Dwell at u, u + d, u - d (settle then average), move to the best of the three."""

    def __init__(self, ctx: RingContext, delta_codes: int = 16, settle: int = 4, avg: int = 4):
        self.d, self.settle, self.avg = int(delta_codes), int(settle), int(avg)
        self.u = None
        self.k = 0
        self.acc = [0.0, 0.0, 0.0]

    def step(self, robs: RingObservation):
        if self.u is None:
            self.u = int(robs.sensors["tx.heater.applied"])
        samples = np.asarray(robs.sensors["tx.monitor.samples"], dtype=float)
        dwell = self.settle + self.avg
        phase, kk = divmod(self.k, dwell)
        if samples.size and kk >= self.settle:
            self.acc[phase] += samples.mean()
        self.k += 1
        if self.k == 3 * dwell:
            best = int(np.argmax(self.acc))
            self.u += (0, self.d, -self.d)[best]
            self.k, self.acc = 0, [0.0, 0.0, 0.0]
        phase = self.k // dwell
        return {"tx.heater": int(self.u + (0, self.d, -self.d)[phase])}


def make_po_controller(**kw) -> PerRing:
    """A Controller for mrm.sim.runner.run(..., controller=make_po_controller())."""
    return PerRing(lambda ctx: PerturbObserveRing(ctx, **kw), name="template_po")
