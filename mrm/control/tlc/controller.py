"""``tlc`` (a lazy entry in mrm.control.registry): the adapter between mrm's controller API and the TLC core (docs/TLC_PORT_SPEC_AND_PLAN.md 2.1).

mrm calls ``step`` every 10 us (G26: controller_interface.update_rate_hz = 100 kHz). The core runs on every
``core_every_n_edges``-th call (100 us, the source's control period): it consumes the monitor samples of the
previous block and plans ``core_every_n_edges`` heater samples for the next one. Each call emits one planned
sample, which reproduces the source's 100 kS/s heater stream with its dither. One core per world (B).

Units: the core speaks power fraction and GHz; :mod:`mrm.control.tlc.units` converts to DAC codes and pm.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from mrm.control.api import BaseController, ControllerSpec, Observation
from mrm.control.tlc.sample_buffer import SampleBuffer
from mrm.control.tlc.stub_core import MODE_TRACK, StubCore
from mrm.control.tlc.telemetry_log import TelemetryLog
from mrm.control.tlc.units import HeaterLaw, level_from_codes, pm_per_ghz

#: core name -> factory(n_rings, *, ts, n_out, params) returning an object with the StubCore interface
CORES = {
    "stub": lambda n_rings, *, ts, n_out, params: StubCore(n_rings, ts=ts, n_out=n_out,
                                                           dither_amp_u=params.dither_amp_u),
}
START_MODES = ("warm", "cold")


class TlcController(BaseController):
    """The ECE496 TLC behind a multirate adapter. Phase 1: ``core = "stub"`` only."""
    name = "tlc"

    @dataclass
    class Params:
        core: str = "stub"
        core_every_n_edges: int = 10        # 10 x 10 us = the source's 100 us control period
        start_mode: str = "warm"            # warm: lock at the applied codes; cold: the core's acquisition
        dither_amp_u: float = 0.0           # stub only: dither amplitude in power fraction
        buffer_capacity: int = 64           # monitor samples per ring per core block (G26: 25)
        latency_ticks: int = 0

    def reset(self, spec: ControllerSpec) -> None:
        super().reset(spec)
        p = self.params
        if p.core not in CORES:
            raise ValueError(f"tlc: unknown core {p.core!r}; available: {sorted(CORES)}")
        if p.core_every_n_edges < 1:
            raise ValueError(f"tlc: core_every_n_edges must be >= 1, got {p.core_every_n_edges}")
        if p.start_mode not in START_MODES:
            raise ValueError(f"tlc: start_mode must be one of {START_MODES}, got {p.start_mode!r}")
        if spec.schedule != "simultaneous":
            raise ValueError("tlc: needs controller_interface.schedule = simultaneous")
        self.latency_ticks = int(p.latency_ticks)
        self.n_out = int(p.core_every_n_edges)
        self.core_ts = self.n_out * spec.period_s
        self.heater_law = HeaterLaw(spec.heater_dac.code_min, spec.heater_dac.code_max)
        self.pm_per_ghz = self._ring_pm_per_ghz(spec)
        self.buffer = SampleBuffer(spec.n_batch, spec.n_rings, p.buffer_capacity)
        self.cores = None
        self.plan = self.zeros("B", "N", self.n_out)
        self.edge = 0
        self.core_step_us = 0.0
        self.telemetry_log = TelemetryLog()

    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        spec = self._require_spec()
        if self.cores is None:
            self.cores = self._start_cores(obs)
        samples = level_from_codes(obs["tx.monitor.samples"], spec.adc.full_scale_codes)
        self.buffer.push(samples, obs["tx.monitor.count"])
        k = self.edge % self.n_out
        if k == 0:
            self._run_cores(obs.t_s - self.core_ts)
            self.telemetry_log.record(obs.t_s, self.telemetry())
        self.edge += 1
        return {"tx.heater": self.heater_law.code_from_u(self.plan[:, :, k])}

    def telemetry(self) -> dict[str, np.ndarray]:
        shape = self.shape("B", "N")
        if self.cores is None:
            last = [StubCore(shape[1], ts=1.0, n_out=1).last] * shape[0]
        else:
            last = [c.last for c in self.cores]
        per = lambda key: np.stack([np.broadcast_to(np.asarray(d[key], dtype=np.float64), shape[1:]) for d in last])
        modes, u = per("modes"), per("u")
        return {
            "state": modes.astype(np.int64),
            "locked": modes == MODE_TRACK,
            "detuning_est_pm": per("delta_hat") * self.pm_per_ghz,
            "integrator": u,
            "integrator_saturated": self.heater_law.saturated(u) & (self.cores is not None),
            "rate_est_pm_per_ms": per("rate_hat") * self.pm_per_ghz * 1e-3,
            "sigma_pm": per("sigma") * self.pm_per_ghz,
            "dither_amp_pm": per("amp") * self.pm_per_ghz,
            "eff_learned": per("eff"),
            "squelch": per("squelch") > 0,
            "fault": per("fault") > 0,
            "core_step_us": np.full(shape, self.core_step_us),
        }

    # -- internals --
    def _start_cores(self, obs: Observation) -> list:
        spec, p = self._require_spec(), self.params
        make = CORES[p.core]
        cores = [make(spec.n_rings, ts=self.core_ts, n_out=self.n_out, params=p) for _ in range(spec.n_batch)]
        u0 = self.heater_law.u_from_code(obs["tx.heater.applied"])
        for b, core in enumerate(cores):
            core.warm_start(u0[b]) if p.start_mode == "warm" else core.cold_start()
        self.plan = np.repeat(u0[:, :, None], self.n_out, axis=2)
        return cores

    def _run_cores(self, t_block_start: float) -> None:
        data, fill = self.buffer.drain()
        t0 = time.perf_counter()
        for b, core in enumerate(self.cores):
            k = int(fill[b].min())                  # simultaneous schedule: every ring has the same count
            seq = np.asarray(core.step(data[b, :, :k].T, t_block_start), dtype=np.float64)
            self.plan[b] = seq.T
        self.core_step_us = (time.perf_counter() - t0) * 1e6 / len(self.cores)

    @staticmethod
    def _ring_pm_per_ghz(spec: ControllerSpec) -> np.ndarray:
        """pm per GHz at each ring's assigned line (NaN without line_grid knowledge)."""
        if spec.line_grid_hz is None:
            return np.full(spec.n_rings, np.nan)
        grid = np.asarray(spec.line_grid_hz, dtype=np.float64)
        r2l = np.asarray(spec.ring_to_line if spec.ring_to_line is not None else np.arange(spec.n_rings))
        return pm_per_ghz(grid[r2l])
