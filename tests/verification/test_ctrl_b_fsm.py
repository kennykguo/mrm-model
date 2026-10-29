"""Unit tests of the acquisition state machine (mrm.control.baselines.ramp_acquire.RampFSM) on a toy plant.

Read this first
---------------
The toy plant is a comb of Lorentzian drop peaks in drive units, with one controller period of lag and ADC noise.
It has no thermal model and no bistability. So these tests check only the state machine's bookkeeping: searches
that end, retries, FAIL and re-arm, and the search after a lock loss, in cases that are slow or rare on the
reference plant. The closed-loop physics is in test_ctrl_b_ramp.py and on the engine (test_ctrl_b_engine.py,
test_ctrl_b_robust.py).

Toy plant: y(w) = bg + sum_k A / (1 + (2 (w - c_k) / fwhm_w)^2), at the drive w applied one period earlier, times
the light level (0 = laser off). The comb moves by ``drift`` (drive units; negative = the chip warmed, so every
resonance needs less heater).
"""
from __future__ import annotations

import numpy as np
import pytest

from mrm.control.api import Observation
from mrm.control.baselines.bitstat_lock import BitStatLock
from mrm.control.baselines.ramp_acquire import ST_FAIL, ST_LOCK, ST_UP, RampAcquire, RampFSM, RampParams
from mrm.control.calibration import w_of_code

from test_ctrl_b_plant import load_g26, make_spec

KNOW = ["line_grid", "ring_to_line", "heater_nominal", "optics_nominal"]


@pytest.fixture(scope="module")
def spec():
    return make_spec(load_g26(knowledge=KNOW), 1)


class Toy:
    """Lorentzian comb in drive units (module docstring), every ring the same."""

    def __init__(self, spec, centers=(0.2, 0.3, 0.4, 0.5), fwhm_w=0.015, amp=30000.0, bg=1500.0, noise=20.0, seed=0):
        self.spec = spec
        self.c = np.asarray(centers, float)
        self.fw, self.amp, self.bg, self.noise = fwhm_w, amp, bg, noise
        self.rng = np.random.default_rng(seed)
        self.light = 1.0
        self.drift = 0.0
        self.q = 2
        self.code = np.zeros((spec.n_batch, spec.n_rings), np.int64)

    def y(self) -> np.ndarray:
        w = w_of_code(self.code, int(self.spec.heater_dac.code_max), self.q)
        x = 2.0 * (w[..., None] - (self.c + self.drift)) / self.fw
        y = self.bg * self.light + self.light * self.amp * np.sum(1.0 / (1.0 + x * x), axis=-1)
        y = y + self.rng.normal(0.0, self.noise, y.shape)
        return np.clip(np.rint(y), 0, int(self.spec.adc.full_scale_codes))

    def obs(self, k: int) -> Observation:
        B, N = self.code.shape
        y = self.y()
        sensors = {"tx.monitor.samples": y[..., None].astype(np.int64), "tx.monitor.count": np.ones((B, N), np.int32),
                   "tx.monitor.latest": y.astype(np.int64), "tx.heater.applied": self.code.astype(np.uint16)}
        return Observation(k * self.spec.period_ticks, self.spec.tick_s, sensors, {c: np.ones((B, N), bool)
                                                                                   for c in sensors})


def run(toy: Toy, ctrl, n: int, k0: int = 0, events=None):
    """Step ``ctrl`` on ``toy`` for n periods; ``events`` maps a period index to a callable(toy). Returns the code
    history (n, N)."""
    hist = np.zeros((n, toy.code.shape[1]), np.int64)
    for i in range(n):
        if events and (k0 + i) in events:
            events[k0 + i](toy)
        out = ctrl.step(toy.obs(k0 + i))
        toy.code = np.asarray(out["tx.heater"], np.int64)
        hist[i] = toy.code[0]
    return hist


def peak_w_of_lock(ctrl, toy) -> float:
    """Resonance (comb member) the ring-0 lock sits next to, in drive units of the present comb."""
    w = float(w_of_code(toy.code[0, 0], int(toy.spec.heater_dac.code_max), 2))
    c = toy.c + toy.drift
    return float(c[np.argmin(np.abs(w - c))])


def test_dark_search_ends_in_fail_and_parks_the_heater(spec):
    """No light: the search climbs to full scale, retries, FAILs and parks at the window's low edge. Before a fix
    the default window (w_hi = 1) never ended: the heater stayed at full scale for ever (w was clipped to 1, the
    end-of-window test w > w_hi + 3 dw_up never became true)."""
    f = RampFSM(spec, RampParams(rings=[0], fail_rearm_s=None))
    mask = np.zeros((1, spec.n_rings), bool)
    mask[0, 0] = True
    f.start(mask)
    y = np.full((1, spec.n_rings), 500.0)
    ok = np.ones_like(y, bool)
    n_search = int(1.0 / f.dw_up0) + 10
    for _ in range((RampParams().max_retries + 1) * n_search + 20):
        f.step(y, ok)
    assert f.state[0, 0] == ST_FAIL
    assert f.code[0, 0] == 0                                   # parked at w_lo = 0, not at full scale


def test_standalone_reacquires_after_a_laser_outage_at_start(spec):
    """Laser off for the first 12 ms, then on: the standalone controller FAILs in the dark and its re-arm
    (fail_rearm_s) takes the first resonance once the light is back. Before a fix FAIL was final."""
    toy = Toy(spec)
    toy.light = 0.0
    ctrl = RampAcquire(rings=[0], start_delay_s=0.0)
    ctrl.reset(spec)
    on = int(12e-3 / spec.period_s)
    run(toy, ctrl, on + int(8e-3 / spec.period_s), events={on: lambda t: setattr(t, "light", 1.0)})
    assert ctrl.fsm.state[0, 0] == ST_LOCK
    assert peak_w_of_lock(ctrl, toy) == pytest.approx(0.2)


def test_lock_loss_search_follows_the_tracked_drift(spec):
    """Lock on the first resonance (w 0.2), the chip warms (the comb moves by -0.06, more than half the 0.1 spacing),
    the loop tracks it; then the light is gone for 1.5 ms. The re-search must take the SAME resonance (now at 0.14):
    from the stale peak drive (0.2 - 6 steps = 0.174) it would climb to the next one, 0.24, and a search left running
    in the dark would climb further still."""
    toy = Toy(spec)
    ctrl = RampAcquire(rings=[0], start_delay_s=0.0)
    ctrl.reset(spec)
    n_lock = int(3e-3 / spec.period_s)
    run(toy, ctrl, n_lock)
    assert ctrl.fsm.state[0, 0] == ST_LOCK and peak_w_of_lock(ctrl, toy) == pytest.approx(0.2)
    n_drift = int(4e-3 / spec.period_s)

    def warm(i):
        return lambda t: setattr(t, "drift", -0.06 * min(1.0, (i + 1) / n_drift))
    run(toy, ctrl, n_drift, k0=n_lock, events={n_lock + i: warm(i) for i in range(n_drift)})
    assert ctrl.fsm.state[0, 0] == ST_LOCK and peak_w_of_lock(ctrl, toy) == pytest.approx(0.14)
    k = n_lock + n_drift
    off, back = k, k + int(1.5e-3 / spec.period_s)
    run(toy, ctrl, int(12e-3 / spec.period_s), k0=k,
        events={off: lambda t: setattr(t, "light", 0.0), back: lambda t: setattr(t, "light", 1.0)})
    f = ctrl.fsm
    assert f.state[0, 0] == ST_LOCK
    assert peak_w_of_lock(ctrl, toy) == pytest.approx(0.14), "relocked on a different resonance"


def test_retries_clear_after_a_long_lock(spec):
    """Each loss of lock costs a retry; a lock held for retry_reset_s clears the count, so a ring that loses its lock
    a few times over a long run does not end in FAIL at its next harmless loss."""
    toy = Toy(spec)
    ctrl = RampAcquire(rings=[0], start_delay_s=0.0, retry_reset_s=2e-3)
    ctrl.reset(spec)
    k = 0
    n = int(3e-3 / spec.period_s)
    run(toy, ctrl, n)
    k += n
    for _ in range(RampParams().max_retries + 2):
        # a 0.4 ms dropout: lock lost, re-found on the narrowed re-search once the light is back
        a, b = k, k + int(0.4e-3 / spec.period_s)
        run(toy, ctrl, n, k0=k, events={a: lambda t: setattr(t, "light", 0.0), b: lambda t: setattr(t, "light", 1.0)})
        k += n
        assert ctrl.fsm.state[0, 0] == ST_LOCK
        assert peak_w_of_lock(ctrl, toy) == pytest.approx(0.2)
    assert ctrl.fsm.retries[0, 0] == 0


def test_peak_near_full_scale_does_not_hang(spec):
    """A resonance at w = 0.995 (its fall cannot be confirmed below full scale): the search must still end (FAIL,
    heater parked) instead of holding w = 1."""
    toy = Toy(spec, centers=(0.995,))
    ctrl = RampAcquire(rings=[0], start_delay_s=0.0, fail_rearm_s=None)
    ctrl.reset(spec)
    run(toy, ctrl, int(30e-3 / spec.period_s))
    f = ctrl.fsm
    assert f.state[0, 0] in (ST_FAIL, ST_LOCK)
    if f.state[0, 0] == ST_FAIL:
        assert f.code[0, 0] < int(spec.heater_dac.code_max)


def test_bitstat_lock_rearms_too(spec):
    """BitStatLock shares the FSM: in FAIL it re-arms after fail_rearm_s (the acquisition part only; the eye loop
    needs the plant, test_ctrl_b_bitstat.py)."""
    cfg = load_g26(knowledge=KNOW)
    sp = make_spec(cfg, 1)
    ctrl = BitStatLock(rings=[0], start_delay_s=0.0, fail_rearm_s=1e-3)
    ctrl.reset(sp)
    f = ctrl.fsm
    f.start(np.eye(1, sp.n_rings, 0, dtype=bool))
    f.state[0, 0] = ST_FAIL
    f.fail_steps[0, 0] = f.rearm_steps
    f.auto_rearm()
    assert f.state[0, 0] == ST_UP and f.retries[0, 0] == 0


@pytest.mark.parametrize("name", ["ramp_acquire", "acquire_then_lock", "bitstat_lock", "wdm_supervisor"])
def test_fsm_controllers_refuse_round_robin(name):
    """Under controller_interface.schedule = round_robin the engine applies only ring j % N at edge j. The FSM moves
    every ring's heater at every edge, so its sweeps and lock loop run on drives the rings do not have (g26 nominal
    on the engine, ramp_acquire ring 0, 3 ms: 148 pm and locked with simultaneous; 1551 pm, std 544 pm, still
    reported LOCK with round_robin). These controllers accepted the setting silently; now reset refuses it, as
    kalman_lqg and mpc do."""
    from mrm.control import registry
    sp = make_spec(load_g26(overrides=["controller_interface.schedule=round_robin"], knowledge=KNOW), 1)
    assert sp.schedule == "round_robin"
    with pytest.raises(ValueError, match="schedule"):
        registry.create(name, {}).reset(sp)
