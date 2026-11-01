"""Phase 1 stand-in for the TLC core, with the source core's interface (``thermal_control/tlc/controller.py``).

It holds the warm-start command and adds the source's per-ring dither tones, so the adapter, the schedule, the
telemetry and the report can be built and tested end to end before the real core arrives (Phase 2).

Interface shared with the real core, per world:

- ``warm_start(u0)`` / ``cold_start()``: start locked at power fractions u0 (N,), or acquire;
- ``first_plan(t0) -> seq``: the heater samples (n_out, N) of the first block, from the start state (the
  source driver's ``out.produce(u_cmd, 0, ...)`` before its loop);
- ``step(pd, t_block_start) -> seq``: consume one block of PD levels (K, N) whose first sample is at
  ``t_block_start`` and return the heater power fractions (n_out, N) for the NEXT block, as the source does;
- ``last``: the source's telemetry dict (delta_hat and sigma in GHz, rate_hat in GHz/s, modes, u, amp in GHz,
  squelch, fault, eff).
"""
from __future__ import annotations

import numpy as np

#: ECE496 B1 tone plan (thermal_control/params.py, ControllerParams.tone_freqs): collision-free for 4 rings
SOURCE_TONES_HZ = (2.0e3, 3.0e3, 3.5e3, 4.5e3)
MODE_ACQUIRE = 1        # thermal_control/tlc/common.py, Mode
MODE_TRACK = 2


class StubCore:
    """Hold plus dither: seq = clip(u_hold + dither_amp_u sin(2 pi f_i t), 0, 1)."""

    def __init__(self, n_rings: int, *, ts: float, n_out: int, dither_amp_u: float = 0.0,
                 tone_freqs: tuple = SOURCE_TONES_HZ):
        if dither_amp_u > 0 and n_rings > len(tone_freqs):
            raise ValueError(f"StubCore: the tone plan has {len(tone_freqs)} tones for {n_rings} rings; an "
                             f"{n_rings}-ring tone plan is Phase 3 work, so run the stub with dither_amp_u = 0")
        self.n, self.ts, self.n_out = int(n_rings), float(ts), int(n_out)
        self.amp = float(dither_amp_u)
        self.freqs = np.zeros(self.n)
        self.freqs[:min(self.n, len(tone_freqs))] = tone_freqs[:self.n]
        self.u_hold = np.zeros(self.n)
        self.modes = np.zeros(self.n, dtype=np.int64)
        self.last = self._telemetry(self.u_hold)

    def warm_start(self, u0: np.ndarray) -> None:
        self.u_hold = np.array(u0, dtype=np.float64)
        self.modes = np.full(self.n, MODE_TRACK, dtype=np.int64)

    def cold_start(self) -> None:
        self.u_hold = np.zeros(self.n)
        self.modes = np.full(self.n, MODE_ACQUIRE, dtype=np.int64)

    def force_fault(self) -> None:
        raise NotImplementedError("the stub core has no fallback; use core = faithful")

    def first_plan(self, t0: float) -> np.ndarray:
        """The heater samples (n_out, N) of the first block [t0, t0 + ts), before any PD block was consumed."""
        return self._plan(t0)

    def step(self, pd: np.ndarray, t_block_start: float) -> np.ndarray:
        seq = self._plan(t_block_start + self.ts)
        self.last = self._telemetry(seq[-1])
        return seq

    def _plan(self, t_start: float) -> np.ndarray:
        t = t_start + np.arange(self.n_out) * self.ts / self.n_out
        seq = self.u_hold[None, :] + self.amp * np.sin(2 * np.pi * self.freqs[None, :] * t[:, None])
        return np.clip(seq, 0.0, 1.0)

    def _telemetry(self, u: np.ndarray) -> dict:
        z = np.zeros(self.n)
        return dict(delta_hat=z, rate_hat=z, sigma=z, modes=self.modes.copy(), u=np.array(u), amp=z,
                    squelch=np.zeros(self.n, dtype=bool), fault=False, eff=np.ones(self.n))
