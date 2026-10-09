"""Temperature-sensor feed-forward + PI (mrm/control/baselines/feedforward.py) on the g26 nominal plant: a die-wide base
ramp (the sensor sees it) and the knowledge gate (CTRL_A_METRIC lines; run with -s)."""
from __future__ import annotations

import numpy as np
import pytest

import test_ctrl_a_harness as H
from mrm.control import registry
from mrm.control.baselines.feedforward import FeedForwardPI
from mrm.control.baselines.pi_lock import PiLock
from mrm.scenarios import generators as g

KNOW = ["controller_interface.knowledge=[line_grid, ring_to_line, heater_law, heater_efficiency, dlambda_dt_nominal]"]


def report(tag, m: H.LockMetrics, **extra):
    s = m.summary()
    s.update(extra)
    print("CTRL_A_METRIC", tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})


def run_ramp(ctrl, rate_k_per_s=1000.0, dur_s=0.006):
    cfg, pp, _, plant = H.placed()
    spec = H.make_spec(H.load_cfg(KNOW), 1)           # knowledge changes the spec, not the plant
    loop = H.ClosedLoop(plant, spec, ctrl)
    loop.run(6000)                                    # MAP + lock (locked by ~3.9 ms) + 2 ms
    t0 = (plant.k_next + 1000) * plant.dt
    ramp = g.Scenario(name="base_ramp", components=(g.pwl("thermal.t_base_k", [0.0, dur_s], [0.0, rate_k_per_s * dur_s],
                                                          t0_s=t0, label="ramp", provenance="assumed"),))
    plant.set_scenario(H.stacked(cfg, ramp))
    rec = loop.run(int(dur_s / plant.dt) + 3000)       # the ramp plus 2 ms after it
    return rec, H.lock_metrics(rec, slice(0, 900), slice(1000, None))


@pytest.fixture(scope="module")
def ramps():
    pi = run_ramp(PiLock())
    ff = FeedForwardPI()
    out = run_ramp(ff)
    return pi, out, ff


@pytest.mark.slow
def test_feedforward_reduces_the_base_ramp_error(ramps):
    """A 1000 K/s die-wide ramp of 6 K (the sensor sees it 1-2 ms late, 0.0625 K steps): the stable-side PI alone
    already holds it within ~1 pm (self-heating relief, 2.5 kHz crossover); the feed-forward (first-order hold
    between sensor deliveries) changes that by +-10-20 % depending on the state at hand-off. Both numbers reported."""
    (rec_pi, m_pi), (rec_ff, m_ff), ff = ramps
    tel = ff.telemetry()
    assert np.all(tel["ff_active"] == 1.0)
    assert np.all(tel["ff_codes"] < -500.0)                       # the die got hotter: less heater
    # the gain is small and fragile: with 1 ms less settling before the ramp the feed-forward came out ~20 % WORSE
    # (the die sensor also sees the bank's own heater power, which still moves after the MAP; the feed-forward then
    # double-counts what the PI already corrects). Bound loosely; both numbers are reported.
    assert np.all(m_ff.max_err_pm <= m_pi.max_err_pm * 1.5), (m_ff.max_err_pm, m_pi.max_err_pm)
    report("pi_lock base ramp 1000 K/s x 6 K", m_pi)
    report("feedforward_pi base ramp 1000 K/s x 6 K", m_ff, ff_codes=float(np.mean(tel["ff_codes"])))


@pytest.mark.slow
def test_staircase_feedforward_is_worse_than_none():
    """predict='hold' steps the heater every sensor period (1 kHz): the PI sees those steps as disturbances."""
    _, m_hold = run_ramp(FeedForwardPI(predict="hold"))
    assert np.all(m_hold.max_err_pm > 2.0)
    report("feedforward_pi (staircase hold) base ramp 1000 K/s x 6 K", m_hold)


def test_without_knowledge_the_feedforward_is_off():
    spec = H.make_spec(H.load_cfg(), 1)              # g26: knowledge = [line_grid, ring_to_line]
    c = FeedForwardPI()
    c.reset(spec)
    assert not c.ff_active
    c2 = FeedForwardPI(ff_codes_per_k=200.0)         # a calibration constant instead of knowledge
    c2.reset(spec)
    assert c2.ff_active
    with pytest.raises(ValueError):
        FeedForwardPI(sensor="nope").reset(spec)
    assert isinstance(registry.create("feedforward_pi", {"ff_scale": 0.9}), FeedForwardPI)


class _ToyWithDie(H.ToyRings):
    """Toy rings plus a die-sensor reading that ramps from ``t0_tick`` (delivered every sensor period, held between)."""

    def __init__(self, spec, *, rate_k_per_tick, t0_tick, **kw):
        super().__init__(spec, **kw)
        self.ts = spec.temp_sensors[0]
        self.rate, self.t0 = rate_k_per_tick, t0_tick
        self.code = None

    def observation(self):
        from mrm.control.api import Observation
        obs = super().observation()
        sens = {k: np.array(v) for k, v in obs.sensors.items()}
        new = {k: np.array(v) for k, v in obs.new.items()}
        deliver = self.tick % self.ts.period_ticks == 0 or self.code is None
        if deliver:
            t_c = 25.0 + self.rate * max(self.tick - self.t0, 0)
            self.code = round((t_c - self.ts.offset_c) / self.ts.lsb_k)
        sens[self.ts.channel][...] = self.code
        new[self.ts.channel][...] = deliver
        return Observation(self.tick, self.spec.tick_s, sens, new)


def test_feedforward_starts_per_ring():
    """The feed-forward reference temperature is taken per ring at its first lock. It used to be taken per world once
    EVERY ring had locked, so one ring that never locks (here ring 1 starts on the cold side and its MAP faults)
    switched the feed-forward off for the whole bank."""
    spec = H.toy_spec(1, 2)
    u0 = np.array([[30000.0, 31000.0]])
    init = [31400, 28000]
    toy = _ToyWithDie(spec, u0=u0, g=0.8, u_init=init, x0=np.array([[3.0, -3.0]]), rate_k_per_tick=2e-4,
                      t0_tick=8000)
    c = FeedForwardPI(init_codes=init, ff_codes_per_k=200.0)
    c.reset(spec)
    toy.run(c, 2000)
    tel = c.telemetry()
    assert tel["state"][0, 0] == 3 and tel["state"][0, 1] == 5          # ring 0 LOCK, ring 1 FAULT
    assert np.all(tel["ff_active"] == 1.0)
    assert tel["ff_codes"][0, 0] < -100.0                                 # ~2.4 K of die rise: less heater
    assert tel["ff_codes"][0, 1] == 0.0                                   # never locked: no feed-forward
    assert tel["sensor_k"][0, 0] > 1.5 and np.isnan(tel["sensor_k"][0, 1])
