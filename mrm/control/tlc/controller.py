"""``tlc`` (a lazy entry in mrm.control.registry): the adapter between mrm's controller API and the TLC core (docs/TLC_PORT_SPEC_AND_PLAN.md 2.1).

mrm calls ``step`` every 10 us (G26: controller_interface.update_rate_hz = 100 kHz). The core runs on every
``core_every_n_edges``-th call (100 us, the source's control period): it consumes the monitor samples of the
previous block and plans ``core_every_n_edges`` heater samples for the next one. Each call emits one planned
sample, which reproduces the source's 100 kS/s heater stream with its dither. One core per world (B).

As in the source's driver, the first block's samples come from the start state (``core.first_plan``) and the core
first runs one block later, on a full block; the samples delivered at the first edge predate the start and are
dropped.

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

def _faithful(kind: str):
    def make(spec, *, ts, n_out, params):
        from mrm.control.tlc.faithful_core import FaithfulCore       # imports scipy and the nominal model
        return FaithfulCore(spec, ts=ts, n_out=n_out, kind=kind)
    return make


#: core name -> factory(spec, *, ts, n_out, params) returning an object with the StubCore interface
CORES = {
    "stub": lambda spec, *, ts, n_out, params: StubCore(spec.n_rings, ts=ts, n_out=n_out,
                                                        dither_amp_u=params.dither_amp_u),
    "faithful": _faithful("tlc"),           # the vendored ECE496 TLC with G26 priors
    "faithful_pi": _faithful("pi"),         # the source's BaselinePI (same B1/B2/B3/B8, PI only)
}
START_MODES = ("warm", "cold")
PD_FAULT_VALUES = {1: np.nan, 2: 0.31, 3: None}     # source PDFault kinds; 3 = one LSB below full scale


def pad_to_common_count(samples: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top every ring up to the largest count of its world at this edge with NaN (missing, B2), so a withheld
    sample (an ADC drop) keeps the ring's later samples on their true time slots instead of shifting them."""
    count = np.asarray(count, dtype=np.int64)
    top = count.max(axis=1, keepdims=True)
    slot = np.arange(samples.shape[-1])[None, None, :]
    missing = (slot >= count[..., None]) & (slot < top[..., None])
    return np.where(missing, np.nan, np.asarray(samples, dtype=np.float64)), np.broadcast_to(top, count.shape).copy()


class TlcController(BaseController):
    """The ECE496 TLC behind a multirate adapter. ``core``: stub | faithful | faithful_pi."""
    name = "tlc"

    @dataclass
    class Params:
        core: str = "stub"
        core_every_n_edges: int = 10        # 10 x 10 us = the source's 100 us control period
        start_mode: str = "warm"            # warm: lock at the applied codes; cold: the core's acquisition
        dither_amp_u: float = 0.0           # stub only: dither amplitude in power fraction
        buffer_capacity: int = 64           # monitor samples per ring per core block (G26: 25)
        force_fault_at_s: float | None = None   # R22: force the core's PI fallback at this time (faithful core)
        # R21 extension (spec 6, Q9): a PD fault injected at the controller boundary, as the source's PDFault:
        # kind 1 missing (NaN), 2 stuck at 0.31 of full scale, 3 saturated. pd_fault_ring < 0: off
        pd_fault_ring: int = -1
        pd_fault_t0_s: float = 0.0
        pd_fault_dur_s: float = 0.0
        pd_fault_kind: int = 1
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
        if p.pd_fault_ring >= 0 and (p.pd_fault_ring >= spec.n_rings or p.pd_fault_kind not in PD_FAULT_VALUES):
            raise ValueError(f"tlc: pd_fault_ring must be < {spec.n_rings} and pd_fault_kind in {sorted(PD_FAULT_VALUES)}")
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
        self.fault_forced = False
        self.block_t0 = None
        self.telemetry_log = TelemetryLog()

    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        spec = self._require_spec()
        if self.cores is None:
            self.cores = self._start_cores(obs)
        samples = level_from_codes(obs["tx.monitor.samples"], spec.adc.full_scale_codes)
        if "monitor_valid" in obs.sensors:          # an invalid sample reaches the core as missing (B2: NaN)
            samples = np.where(np.asarray(obs.sensors["monitor_valid"], dtype=bool)[..., None], samples, np.nan)
        samples = self._inject_pd_fault(samples, obs.t_s)
        samples, count = pad_to_common_count(samples, obs["tx.monitor.count"])
        self._note_block_start(obs, count)
        self.buffer.push(samples, count)
        k = self.edge % self.n_out
        if self.edge == 0:
            self.buffer.drain()
            self.block_t0 = None
            self.telemetry_log.record(obs.t_s, self.telemetry())
        elif k == 0:
            self._maybe_force_fault(obs.t_s)
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
        absent = {"eff": 1.0}                    # BaselinePI's last omits the estimator fields: report 0 (eff 1)
        per = lambda key: np.stack([np.broadcast_to(np.asarray(d.get(key, absent.get(key, 0.0)), dtype=np.float64),
                                                    shape[1:]) for d in last])
        modes, u = per("modes"), per("u")
        return {
            "state": modes.astype(np.int64),
            "locked": modes == MODE_TRACK,
            "detuning_est_pm": per("delta_hat") * self.pm_per_ghz,
            "integrator": u,
            "integrator_saturated": self.heater_law.at_rail(u) & (self.cores is not None),
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
        cores = [make(spec, ts=self.core_ts, n_out=self.n_out, params=p) for _ in range(spec.n_batch)]
        u0 = self.heater_law.u_from_code(obs["tx.heater.applied"])
        for b, core in enumerate(cores):
            core.warm_start(u0[b]) if p.start_mode == "warm" else core.cold_start()
        self.plan = np.stack([np.asarray(c.first_plan(obs.t_s), dtype=np.float64).T for c in cores])
        return cores

    def _inject_pd_fault(self, samples: np.ndarray, t_s: float) -> np.ndarray:
        p = self.params
        if p.pd_fault_ring < 0 or not p.pd_fault_t0_s <= t_s < p.pd_fault_t0_s + p.pd_fault_dur_s:
            return samples
        value = PD_FAULT_VALUES[p.pd_fault_kind]
        if value is None:
            value = 1.0 - 1.0 / self._require_spec().adc.full_scale_codes
        out = np.array(samples, dtype=np.float64)
        out[:, p.pd_fault_ring, :] = value
        return out

    def _maybe_force_fault(self, t_s: float) -> None:
        at = self.params.force_fault_at_s
        if at is not None and t_s >= at and not self.fault_forced:
            for core in self.cores:
                core.force_fault()
            self.fault_forced = True

    def _note_block_start(self, obs: Observation, count: np.ndarray) -> None:
        """Remember, per world, when the block's first monitor sample was taken (the centre of its integration
        window, from tx.monitor.sample_ticks), so the core's sample times carry the ADC's age, not the edge's."""
        if self.block_t0 is not None or not np.any(count > 0) or "tx.monitor.sample_ticks" not in obs.sensors:
            return
        spec = self._require_spec()
        first = np.asarray(obs.sensors["tx.monitor.sample_ticks"])[..., 0].astype(np.float64)       # (B, N)
        first = np.where(count > 0, first, np.inf).min(axis=1)
        self.block_t0 = (first - 0.5 * spec.adc.integration_window_ticks) * spec.tick_s

    def _run_cores(self, t_block_start: float) -> None:
        """``t_block_start``: the fallback block start (edge time minus one core period) when the sample ticks are
        not exposed; otherwise each world's block starts at its first sample's time."""
        data, fill = self.buffer.drain()
        starts = self.block_t0 if self.block_t0 is not None else np.full(len(self.cores), t_block_start)
        self.block_t0 = None
        t0 = time.perf_counter()
        for b, core in enumerate(self.cores):
            k = int(fill[b].max())                  # rings padded to a common count (pad_to_common_count)
            seq = np.asarray(core.step(data[b, :, :k].T, float(starts[b])), dtype=np.float64)
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


class TlcPiController(TlcController):
    """``tlc_pi``: the source's BaselinePI (B1/B2/B3/B8 with a PI loop) behind the same adapter."""
    name = "tlc_pi"

    @dataclass
    class Params(TlcController.Params):
        core: str = "faithful_pi"
