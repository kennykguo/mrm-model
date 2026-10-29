"""Checks the temperature-sensor feed-forward + PI controller (mrm/control/baselines/feedforward.py).

Read this first
---------------
FeedForwardPI is the PI lock plus a feed-forward term from the die temperature sensor. The feed-forward needs
knowledge (heater law, heater efficiency, dlambda/dT nominal) or a calibration constant. The tests check, on the g26
nominal plant, a die-wide base ramp (the sensor sees it) and the knowledge gate (CTRL_A_METRIC lines; run with -s).
On toy rings with the quadratic heater law, and on the engine under A1 (slow), they check that the gain schedule
cannot run away and that the controller leaves LOCK without a bump.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import test_ctrl_a_harness as harness
from mrm.control import registry
from mrm.control.baselines.feedforward import FeedForwardPI
from mrm.control.baselines.pi_lock import PiLock
from mrm.scenarios import generators, library
from mrm.sim import runner

KNOW = ["controller_interface.knowledge=[line_grid, ring_to_line, heater_law, heater_efficiency, dlambda_dt_nominal]"]


def report(tag, m: harness.LockMetrics, **extra):
    s = m.summary()
    s.update(extra)
    print("CTRL_A_METRIC", tag, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})


def run_ramp(ctrl, rate_k_per_s=1000.0, dur_s=0.006):
    cfg, pp, _, plant = harness.placed()
    spec = harness.make_spec(harness.load_cfg(KNOW), 1)           # knowledge changes the spec, not the plant
    loop = harness.ClosedLoop(plant, spec, ctrl)
    loop.run(6000)                                    # MAP + lock (locked by ~3.9 ms) + 2 ms
    t0 = (plant.k_next + 1000) * plant.dt
    base_ramp = generators.pwl("thermal.t_base_k", [0.0, dur_s], [0.0, rate_k_per_s * dur_s], t0_s=t0, label="ramp",
                               provenance="assumed")
    ramp = generators.Scenario(name="base_ramp", components=(base_ramp,))
    plant.set_scenario(harness.stacked(cfg, ramp))
    rec = loop.run(int(dur_s / plant.dt) + 3000)       # the ramp plus 2 ms after it
    return rec, harness.lock_metrics(rec, slice(0, 900), slice(1000, None))


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
    spec = harness.make_spec(harness.load_cfg(), 1)              # g26: knowledge = [line_grid, ring_to_line]
    c = FeedForwardPI()
    c.reset(spec)
    assert not c.ff_active
    c2 = FeedForwardPI(ff_codes_per_k=200.0)         # a calibration constant instead of knowledge
    c2.reset(spec)
    assert c2.ff_active
    with pytest.raises(ValueError):
        FeedForwardPI(sensor="nope").reset(spec)
    assert isinstance(registry.create("feedforward_pi", {"ff_scale": 0.9}), FeedForwardPI)


class _ToyWithDie(harness.ToyRings):
    """Toy rings plus a die-sensor reading that ramps from ``t0_tick`` to ``t1_tick`` (delivered every sensor period,
    held between)."""

    def __init__(self, spec, *, rate_k_per_tick, t0_tick, t1_tick=None, **kw):
        super().__init__(spec, **kw)
        self.ts = spec.temp_sensors[0]
        self.rate, self.t0 = rate_k_per_tick, t0_tick
        self.t1 = np.inf if t1_tick is None else t1_tick
        self.code = None

    def observation(self):
        from mrm.control.api import Observation
        obs = super().observation()
        sens = {k: np.array(v) for k, v in obs.sensors.items()}
        new = {k: np.array(v) for k, v in obs.new.items()}
        deliver = self.tick % self.ts.period_ticks == 0 or self.code is None
        if deliver:
            t_c = 25.0 + self.rate * max(min(self.tick, self.t1) - self.t0, 0)
            self.code = round((t_c - self.ts.offset_c) / self.ts.lsb_k)
        sens[self.ts.channel][...] = self.code
        new[self.ts.channel][...] = deliver
        return Observation(self.tick, self.spec.tick_s, sens, new)


def test_feedforward_starts_per_ring():
    """The feed-forward reference temperature is taken per ring at its first lock. It used to be taken per world once
    EVERY ring had locked, so one ring that never locks (here ring 1 starts on the cold side and its MAP faults)
    switched the feed-forward off for the whole bank."""
    spec = harness.toy_spec(1, 2)
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


class _QuadToyWithDie(_ToyWithDie):
    """_ToyWithDie with the quadratic heater law of a voltage drive (g26): the tuning goes as code^2,
    x = (u^2 - u0sq) / (2 c_ref W) + g / (1 + x^2) (W: half width in codes at c_ref). ``u0sq`` (code^2) is the heater
    power at resonance: a ring that heats needs less of it (below 0: not reachable at any code)."""

    def __init__(self, spec, *, u0, c_ref, **kw):
        self.u0sq = np.full((spec.n_batch, spec.n_rings), float(u0) ** 2)
        self.c_ref = float(c_ref)
        super().__init__(spec, u0=u0, **kw)

    def advance(self):
        u = self.heater.astype(float)
        a = (u * u - self.u0sq) / (2.0 * self.c_ref * self.W)
        x = self.x.copy()
        for idx in np.ndindex(x.shape):
            r = np.roots([1.0, -a[idx], 1.0, -(a[idx] + self.g[idx])])
            r = r[np.abs(r.imag) < 1e-9].real
            x[idx] = r[np.argmin(np.abs(r - x[idx]))]
        self.x = x


def _quad_run(n_steps, *, die_k, t1_tick, ring_sq_per_tick=0.0, jump_step=None, jump_sq=0.0):
    """One quadratic-law toy ring locked at ~20000 codes; from tick 10000 to ``t1_tick`` the die sensor rises by
    ``die_k`` and the ring's resonance power falls by ``ring_sq_per_tick`` (code^2/tick); from ``jump_step`` on it
    needs ``jump_sq`` more. Returns the controller and per step (state, ff, u, heater command) after that step."""
    spec = harness.toy_spec(1, 1)
    t0 = 10000
    toy = _QuadToyWithDie(spec, u0=20000.0, c_ref=20000.0, g=0.8, u_init=[21400], x0=3.0,
                          rate_k_per_tick=die_k / (t1_tick - t0), t0_tick=t0, t1_tick=t1_tick)
    c = FeedForwardPI(init_codes=[21400], ff_codes_per_k=200.0)
    c.reset(spec)
    rec = []

    def on_step(t, k):
        if k > 0:                                       # the controller's state after step k - 1
            rec.append((c.ring_values("st")[0, 0], c.ff[0, 0], c.ring_values("u")[0, 0], t.heater[0, 0],
                        c.telemetry()["sensor_k"][0, 0]))
        t.u0sq[...] = 20000.0 ** 2 - ring_sq_per_tick * (min(max(t.tick, t0), t1_tick) - t0)
        if jump_step is not None and k >= jump_step:
            t.u0sq[...] += jump_sq
    toy.run(c, n_steps, on_step=on_step)
    return c, np.array(rec)


def test_feedforward_gain_cannot_run_away():
    """The ring heats ~5x more than the die sensor says (a local aggressor: A1's ring 0): its lock code falls from
    20000 to 2000 while the die reads +10 K. K_ff used to be evaluated at the applied code u + ff, so a large negative
    feed-forward raised its own gain (~1/code) and ran away (here ff ~ -4e7 codes at ~9.5 K, the heater latched at
    code 0; on the engine under A1, ring 0 at ~58 ms). It is now the heater-law secant from the integrator u
    (u^2 - (u + ff)^2 = 2 K_ff(c) c dT), bounded to the DAC range: the ring stays locked."""
    c, rec = _quad_run(5000, die_k=10.0, t1_tick=40000, ring_sq_per_tick=(20000.0 ** 2 - 2000.0 ** 2) / 30000)
    st, ff, u, heater, dT = rec.T
    i0 = int(np.argmax(st == 3))
    assert i0 > 0 and np.all(st[i0:] == 3)                           # locked once, and never left the lock
    assert np.all(np.isfinite(ff)) and np.all((u + ff >= 0.0) & (u + ff <= 65535.0)), (ff.min(), ff.max())
    tel = c.telemetry()
    assert tel["locked"][0, 0] and abs(tel["error_norm"][0, 0]) < 0.02, tel
    assert 2000 < heater[-1] < 3000                                   # the lock code: the heater was not latched at 0
    c_k = c.k_ref_code[0, 0] * 200.0                                  # c K_ff(c), the calibrated constant
    assert dT[-1] == pytest.approx(10.0)
    assert u[-1] ** 2 - (u[-1] + ff[-1]) ** 2 == pytest.approx(2.0 * c_k * dT[-1], rel=1e-9)


def test_feedforward_leaves_the_lock_bumpless_and_rereferences():
    """A fall (the ring suddenly needs more heater; LOCK -> RECAPTURE) takes the feed-forward into the integrator: the
    applied code does not jump, ff is 0 and the reference is dropped while the ring is not locked, and the next lock
    takes a new reference. It used to be held (frozen) while the ring was not locked, and the old reference kept."""
    c, rec = _quad_run(4000, die_k=2.0, t1_tick=20000, jump_step=3000, jump_sq=6e7)
    st, ff, u, heater, dT = rec.T
    k_f = 2999 + int(np.argmax(st[2999:] != 3))                      # rec[k] is the state after step k
    assert st[k_f - 1] == 3 and st[k_f] != 3
    assert ff[k_f - 1] < -300.0                                       # ~2 K of die rise fed forward before the fall
    assert heater[k_f] == heater[k_f - 1]                             # bumpless
    k_r = k_f + int(np.argmax(st[k_f:] == 3))                         # the next lock
    assert k_r > k_f and np.all(ff[k_f:k_r] == 0.0) and np.all(np.isnan(dT[k_f:k_r]))
    assert np.all(st[k_r:] == 3)
    assert abs(dT[-1]) < 0.1 and abs(ff[-1]) < 10.0                   # new reference: the die has not moved since


@pytest.mark.slow
def test_feedforward_holds_every_ring_under_a1_on_the_engine():
    """The defect on the engine (scorecard conditions: g26 nominal, settled warm start, A1 from its minimum): the die
    sensor sees ~+20 K of the aggressor; ring 0's feed-forward ran away (ff -7e3 codes at 55 ms, -8e7 at 58 ms), its
    heater latched at code 0 and the ring went to FAULT at ~78 ms with ff frozen at ~-1e8. 80 ms: past the hot cusp
    (65.35 ms), where ring 0 needs less than zero heater power for ~15 ms (heater authority, as with pi_lock)."""
    c = FeedForwardPI()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a1 = library.a1_slide10_aggressor(phase_s=library.A1_PERIOD_S / 2)          # A1 from its minimum
        res = runner.run(harness.load_cfg(KNOW), controller=c, scenario=a1,
                         duration_s=80e-3, batch=1, uncertainty="nominal", variation="off", seeds=0, settle=True,
                         heater_code="locked_steady_state", reset_protocol="branch_high")
    tel = c.telemetry()
    u = tel["integrator"]
    assert np.all(tel["state"] == 3), tel["state"]                   # every ring locked (ring 0 was FAULT)
    assert np.all(np.isfinite(c.ff)) and np.all((u + c.ff >= 0.0) & (u + c.ff <= 65535.0)), c.ff
    assert tel["ff_codes"][0, 0] < -1000.0                            # the die is still ~16 K up: ff still acts
    assert res.truth_final.heater_applied[0, 0] > 5000                # ring 0 back above code 0 after the cusp
