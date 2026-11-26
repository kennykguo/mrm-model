"""``tlc_wdm`` (second arm, docs/TLC_PORT_SPEC_AND_PLAN.md 6 Q1): mrm's WDM supervisor acquires, the TLC tracks.

The TLC's own ACQUIRE (B5) takes the first fringe it finds; on the G26 grid, where every cold ring sits next to a
foreign line, that is usually the wrong line (docs/tlc/results.md, R9). The WDM supervisor maps the bank, assigns
lines and acquires ring by ring. This controller runs it until a world reaches its TRACK phase (every ring locked)
and then hands that world to the TLC, warm-started at the heater codes the supervisor left. The TLC needs no line
identification afterwards: its loop is self-referenced to the line the ring sits on.

The TLC runs in the shadow from the first step (its adapter keeps the sample buffers and block timing); at a world's
hand-off its core is replaced by a fresh one, warm-started there, so nothing the shadow computed matters. With arm2 options the hand-off
counts as entering TRACK (the DC scale is re-learnt before it is held stiff). The supervisor's ``tx.assignment``
(cyclic remap) is held after the hand-off.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mrm.control.api import BaseController, ControllerSpec, Observation
from mrm.control.baselines.wdm_supervisor import PH_TRACK, WdmSupervisor
from mrm.control.tlc.controller import CORES, TlcController


class TlcWdmController(BaseController):
    """WDM supervisor acquisition, TLC tracking. ``tlc`` / ``wdm``: the two controllers' params."""
    name = "tlc_wdm"

    @dataclass
    class Params:
        tlc: dict = field(default_factory=lambda: {"core": "faithful"})
        wdm: dict = field(default_factory=dict)

    def reset(self, spec: ControllerSpec) -> None:
        super().reset(spec)
        tlc_params = dict(self.params.tlc)
        if tlc_params.get("start_mode", "warm") != "warm":
            raise ValueError("tlc_wdm: the TLC is always warm-started at the hand-off")
        self.wdm = WdmSupervisor(self.params.wdm)
        self.tlc = TlcController(tlc_params)
        self.wdm.reset(spec)
        self.tlc.reset(spec)
        self.latency_ticks = max(self.wdm.latency_ticks, self.tlc.latency_ticks)
        self.handed = self.zeros("B", dtype=bool)
        self.handoff_s = self.full("B", fill_value=np.nan)
        self.held: dict[str, np.ndarray] = {}

    def step(self, obs: Observation) -> dict[str, np.ndarray]:
        tlc_out = self.tlc.step(obs)                   # the shadow TLC keeps its buffers and timing
        if not self.handed.all():
            wdm_out = self.wdm.step(obs)
            self.held = {k: np.array(v) for k, v in wdm_out.items() if k != "tx.heater"}
            ready = (self.wdm.phase == PH_TRACK) & ~self.handed
            if ready.any():
                self._hand_off(np.where(ready)[0], obs)
                tlc_out = {"tx.heater": self.tlc.heater_law.code_from_u(self.tlc.plan[:, :, (self.tlc.edge - 1) % self.tlc.n_out])}
            heater = np.where(self.handed[:, None], tlc_out["tx.heater"], wdm_out["tx.heater"])
        else:
            heater = tlc_out["tx.heater"]
        return {"tx.heater": heater, **self.held}

    def _hand_off(self, worlds: np.ndarray, obs: Observation) -> None:
        u0 = self.tlc.heater_law.u_from_code(np.asarray(obs["tx.heater.applied"]))
        tlc, spec = self.tlc, self._require_spec()
        for b in worlds:
            # a fresh core: the shadow core's estimates (its DC scale above all) come from the acquisition sweeps
            core = CORES[tlc.params.core](spec, ts=tlc.core_ts, n_out=tlc.n_out, params=tlc.params)
            tlc.cores[b] = core
            core.warm_start(u0[b])
            if hasattr(core.ctl, "_track_since"):
                core.ctl._track_since[:] = core.ctl.t        # arm2: re-learn the DC scale before holding it
            self.tlc.plan[b] = np.asarray(core.first_plan(obs.t_s), dtype=np.float64).T
            self.handed[b] = True
            self.handoff_s[b] = obs.t_s

    def telemetry(self) -> dict[str, np.ndarray]:
        tel = dict(self.tlc.telemetry())
        tel["handed_off"] = np.broadcast_to(self.handed[:, None], tel["state"].shape).copy()
        return tel
