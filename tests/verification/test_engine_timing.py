"""Checks the timing rules through the whole plant: loop delay, rate independence, FIFO depth and the fast kernel.

Read this first
---------------
The plant steps in integer ticks, each delay has one owner (the I/O path owns its latency in seconds), and the
events of a tick happen in a fixed order. The tests check that the loop delay of a heater impulse equals the
declared latencies, that it does not depend on the controller rate, that the FIFOs and the controller pipeline are
deep enough (also with tick jitter), and that a fast_kernel (run once per ADC sample) equals a 1 MHz Python step
bit for bit. Later tests check that two configs in one process do not share frozen globals, and parts of the Plant
and registry API.

Common random numbers make the timing tests exact: a run with the impulse and a run without it share every noise
draw (keyed by seed, stream, ring, tick, slot; mrm.core.rng). So the first tick at which a recorded signal differs
is the first tick the impulse reached it.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from numba import njit

from mrm import params
from mrm.control.api import (SPEC_I_N_RINGS, CompiledController, act_offset, user_f64, user_i64)
from mrm.core.channels import ActId
from mrm.core.timing import fifo_capacity, first_response_tick
from mrm.sim.runner import run

HEATER = int(ActId.TX_HEATER)   # actuator id of the heater commands


# -------------------------------------------------------------------------------------------------------- impulse probe
# A compiled controller that sends one impulse code at one edge and a base code at every other edge.
# Its integer parameters p: [n_rings, heater offset in act, impulse edge tick, impulse code, base code].
@njit(cache=True)
def probe_init(si, sf):
    p = np.zeros(5, np.int64)
    p[0] = si[SPEC_I_N_RINGS]
    p[1] = act_offset(si, HEATER)
    p[2] = user_i64(si, 0)          # edge tick of the impulse
    p[3] = user_i64(si, 1)          # impulse code
    p[4] = user_i64(si, 2)          # base code
    return p, np.zeros(1), np.zeros(1, np.int64), np.zeros(1)


@njit(cache=True)
def probe_step(p, pf, s, sf, obs, act):
    for i in range(p[0]):
        act[p[1] + i] = p[3] if obs[0] == p[2] else p[4]


def _g26_config(*ov):
    """The g26 preset with the given config overrides (warnings silenced)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load("g26", overrides=list(ov))


def _impulse_runs(cfg, k_imp, code, n_ticks, lat=0):
    """Two nominal runs of n_ticks with the probe: without the impulse (code 0) and with it (code at tick k_imp).
    lat: controller latency in ticks. Both record the monitor code and the heater power every tick."""
    out = []
    for c in (0, code):
        ctrl = CompiledController("probe", probe_init, probe_step, params_i64=[k_imp, c, 0], latency_ticks=lat)
        r = run(cfg, ctrl, duration_s=n_ticks * 1e-6, batch=1, seeds=4, uncertainty="nominal", variation="off",
                reset_protocol="branch_low", record=[("monitor_code", 1, "last"), ("p_heater_w", 1, "last")],
                oracle=False, histogram=False)
        out.append(r)
    return out


def _first_differing_tick(a, b):
    """The first row (tick) where a and b differ, or -1."""
    d = np.flatnonzero(np.any(a != b, axis=1)) if a.ndim == 2 else np.flatnonzero(a != b)
    return int(d[0]) if d.size else -1


@pytest.mark.parametrize("ctrl_latency", [0, 3])
def test_loop_delay_impulse(ctrl_latency):
    """A one-period heater-code impulse from a probe controller first changes the heater power at
    k + controller latency + DAC latency (including the thermal transport delay), and the controller first
    observes it at timing.first_response_tick(...) for every ring (its own ADC mux phase)."""
    cfg = _g26_config()
    base, imp = _impulse_runs(cfg, 100, 65535, 200, lat=ctrl_latency)
    pl = imp.plant
    dac = int(pl.heater_params.latency_ticks[0])
    adc_lat = int(pl.monitor_hw.latency_ticks[0])
    per, ph0 = pl.spec.period_ticks, pl.spec.phase_ticks
    assert (100 - ph0) % per == 0
    p0 = base.trace["p_heater_w@1:last"]["value"][:, 0]
    p1 = imp.trace["p_heater_w@1:last"]["value"][:, 0]
    k_act = _first_differing_tick(p1, p0)
    assert k_act == 100 + ctrl_latency + dac
    m0 = base.trace["monitor_code@1:last"]["value"][:, 0]
    m1 = imp.trace["monitor_code@1:last"]["value"][:, 0]
    for n in range(pl.N):
        want = first_response_tick(100, ctrl_latency, dac, pl.monitor_hw.period_ticks, int(pl.monitor_hw.phase[n]),
                                   adc_lat, per, ph0)
        got = _first_differing_tick(m1[:, n], m0[:, n])
        assert got == want, (n, got, want)


def test_latency_invariant_to_rate():
    """Sweeping controller_interface.update_rate_hz from 10 to 200 kHz leaves the command-to-plant delay in
    seconds unchanged (latencies are owned by the I/O path in seconds, never counted in samples of the rate), and
    the first observation follows timing.first_response_tick at every rate."""
    delays = []
    for rate in (10e3, 20e3, 50e3, 100e3, 200e3):
        cfg = _g26_config(f"controller_interface.update_rate_hz={rate}")
        per = int(round(1.0 / (rate * 1e-6)))
        k_imp = 3 * per
        base, imp = _impulse_runs(cfg, k_imp, 65535, k_imp + 2 * per + 60)
        pl = imp.plant
        assert pl.spec.period_ticks == per
        p0 = base.trace["p_heater_w@1:last"]["value"][:, 0]
        p1 = imp.trace["p_heater_w@1:last"]["value"][:, 0]
        k_act = _first_differing_tick(p1, p0)
        delays.append((k_act - k_imp) * pl.dt)
        m0 = base.trace["monitor_code@1:last"]["value"][:, 0]
        m1 = imp.trace["monitor_code@1:last"]["value"][:, 0]
        n = 0
        want = first_response_tick(k_imp, 0, int(pl.heater_params.latency_ticks[0]), pl.monitor_hw.period_ticks,
                                    int(pl.monitor_hw.phase[n]), int(pl.monitor_hw.latency_ticks[0]), per,
                                    pl.spec.phase_ticks)
        assert _first_differing_tick(m1[:, n], m0[:, n]) == want
    assert len(set(np.round(np.asarray(delays) * 1e12).astype(int))) == 1, delays
    assert delays[0] > 0


class Collector:
    """Python controller that records every released monitor sample (tick and code) and holds its actuators."""
    name = "collector"
    latency_ticks = 0

    def reset(self, spec):
        self.spec = spec
        self.ticks = []
        self.codes = []
        self.edges = []

    def step(self, obs):
        cnt = np.asarray(obs["tx.monitor.count"])
        st = np.asarray(obs["tx.monitor.sample_ticks"])
        cd = np.asarray(obs["tx.monitor.samples"])
        self.edges.append(obs.tick)
        for n in range(cnt.shape[1]):
            for j in range(cnt[0, n]):
                self.ticks.append((n, int(st[0, n, j])))
                self.codes.append((n, int(cd[0, n, j])))
        return {}


def test_fifo_depth():
    """With an ADC latency longer than the controller period (26 us against 10 us) every
    monitor sample is delivered exactly once, in sample-tick order, at the first edge at or after its release tick;
    no FIFO or block loses a sample; the plant sized the FIFO with timing.fifo_capacity(latency, ADC period,
    release period)."""
    cfg = _g26_config("ring.monitor.adc_latency_s=2.6e-5")
    ctrl = Collector()
    r = run(cfg, ctrl, duration_s=600e-6, batch=1, seeds=2, uncertainty="nominal", variation="off",
            reset_protocol="branch_low", oracle=False, histogram=False)
    pl = r.plant
    lat = int(pl.monitor_hw.latency_ticks[0])
    assert lat == 26
    P = pl.monitor_hw.period_ticks
    assert pl._caps["monitor"] == fifo_capacity(lat, P, pl.spec.period_ticks)
    d = r.diagnostics
    assert int(d["monitor_fifo_lost"][0]) == 0 and int(d["monitor_block_lost"][0]) == 0
    last_edge = max(ctrl.edges)
    for n in range(pl.N):
        got = [t for (m, t) in ctrl.ticks if m == n]
        ph = int(pl.monitor_hw.phase[n])
        want = [s for s in range(ph if ph > 0 else P, last_edge + 1, P) if s + lat <= last_edge]
        assert got == want, n


# ---------------------------------------------------------------------------------------------------------- fast kernel
# A heater-dither lock-in: the fast kernel accumulates I / Q of every new monitor sample and sets the dither sign.
# p: [n_rings, heater offset in act, dither amplitude, base code]; pf: [reference phase per tick];
# s: sample count per ring; sf: [I per ring, Q per ring].
@njit(cache=True)
def lockin_init(si, sf):
    n = si[SPEC_I_N_RINGS]
    p = np.zeros(4, np.int64)
    p[0] = n
    p[1] = act_offset(si, HEATER)
    p[2] = user_i64(si, 0)          # dither amplitude (codes)
    p[3] = user_i64(si, 1)          # base code
    pf = np.zeros(1)
    pf[0] = user_f64(sf, 0)         # reference phase per tick (rad)
    return p, pf, np.zeros(n, np.int64), np.zeros(2 * n)


@njit(cache=True)
def lockin_fast(p, pf, s, sf, fobs, fact):
    n = fobs[1]                     # fobs = [tick, N, new[N], code[N], sample_tick[N]] (mrm.control.api)
    for i in range(n):
        if fobs[2 + i] != 0:
            code = fobs[2 + n + i]
            ph = pf[0] * fobs[2 + 2 * n + i]
            c = math.cos(ph)
            sf[i] += code * c
            sf[n + i] += code * math.sin(ph)
            s[i] += 1
            fact[i] = p[2] if c >= 0.0 else -p[2]


@njit(cache=True)
def lockin_step(p, pf, s, sf, obs, act):
    return None


class PyLockIn:
    """The same lock-in as a 1 MHz Python step: accumulate I/Q of every new sample, dither the heater command."""
    name = "py_lockin"
    latency_ticks = 0

    def __init__(self, amp, base, w):
        self.amp, self.base, self.w = int(amp), int(base), float(w)

    def reset(self, spec):
        B, N = spec.n_batch, spec.n_rings
        self.I = np.zeros((B, N))
        self.Q = np.zeros((B, N))
        self.cnt = np.zeros((B, N), np.int64)
        self.off = np.zeros((B, N), np.int64)

    def step(self, obs):
        cnt = np.asarray(obs["tx.monitor.count"])
        st = np.asarray(obs["tx.monitor.sample_ticks"])
        cd = np.asarray(obs["tx.monitor.samples"])
        B, N = cnt.shape
        for b in range(B):
            for n in range(N):
                for j in range(cnt[b, n]):
                    ph = self.w * int(st[b, n, j])
                    c = math.cos(ph)
                    self.I[b, n] += int(cd[b, n, j]) * c
                    self.Q[b, n] += int(cd[b, n, j]) * math.sin(ph)
                    self.cnt[b, n] += 1
                    self.off[b, n] = self.amp if c >= 0.0 else -self.amp
        return {"tx.heater": (self.base + self.off).astype(np.int64)}


def test_fast_kernel_equals_1mhz_python_step():
    """A heater-dither lock-in written once as a fast_kernel (once per released ADC sample, inside the
    kernel) and once as a 1 MHz Python step gives bit-identical I/Q accumulators and the same plant."""
    cfg = _g26_config("controller_interface.update_rate_hz=1000000.0")
    amp, base = 800, 20000
    w = 2.0 * math.pi * 31.25e3 * 1e-6
    n_ticks = 1500
    fast = CompiledController("li", lockin_init, lockin_step, params_i64=[amp, base], params_f64=[w],
                              fast_fn=lockin_fast)
    kw = dict(duration_s=n_ticks * 1e-6, batch=1, seeds=6, uncertainty="nominal", variation="off",
              reset_protocol="branch_low", heater_code=base, record=[("p_heater_w", 1, "last")], oracle=False,
              histogram=False)
    rf = run(cfg, fast, **kw)
    py = PyLockIn(amp, base, w)
    rp = run(cfg, py, **kw)
    p_i, p_f, s_i, s_f = fast.state
    N = rf.plant.N
    assert int(rf.diagnostics["fast_kernel_calls"][0]) > 0
    np.testing.assert_array_equal(s_i[0], py.cnt[0])
    assert np.all(py.cnt[0] > n_ticks // 4 - 3)
    np.testing.assert_array_equal(s_f[0, :N], py.I[0])
    np.testing.assert_array_equal(s_f[0, N:], py.Q[0])
    np.testing.assert_array_equal(rf.trace["p_heater_w@1:last"]["value"], rp.trace["p_heater_w@1:last"]["value"])


def test_two_configs_in_one_process_differ():
    """No frozen globals in the compiled code: two configs in one process give different outputs, and the first
    config run again reproduces its first result exactly."""
    a = _g26_config()
    b = _g26_config("laser.power_per_line_dbm=9.0")
    kw = dict(duration_s=300e-6, batch=1, seeds=1, uncertainty="nominal", variation="off",
              reset_protocol="branch_low", record=[("p_drop_w", 1, "last")], oracle=False, histogram=False)
    ra = run(a, None, **kw)
    rb = run(b, None, **kw)
    ra2 = run(a, None, **kw)
    ta = ra.trace["p_drop_w@1:last"]["value"]
    tb = rb.trace["p_drop_w@1:last"]["value"]
    assert not np.allclose(ta, tb)
    np.testing.assert_array_equal(ta, ra2.trace["p_drop_w@1:last"]["value"])
    np.testing.assert_array_equal(ra.truth_final.t_ring_k, ra2.truth_final.t_ring_k)


# -------------------------------------------------------------------------------------------------- controller pipeline
# A compiled controller that counts its edges and sends code 1000 + 10 * count to every heater.
@njit(cache=True)
def count_init(si, sf):
    p = np.zeros(2, np.int64)
    p[0] = si[SPEC_I_N_RINGS]
    p[1] = act_offset(si, HEATER)
    return p, np.zeros(1), np.zeros(1, np.int64), np.zeros(1)


@njit(cache=True)
def count_step(p, pf, s, sf, obs, act):
    s[0] += 1
    for i in range(p[0]):
        act[p[1] + i] = 1000 + 10 * s[0]          # a new code at every edge: every command is visible


def test_pipeline_with_edge_jitter_drops_nothing():
    """With tick jitter J two controller edges can be period - J apart, so the controller pipeline must hold
    latency // (period - J) + 1 commands. With latency 9, period 10 and J = 5, a pipeline sized for the nominal
    period would drop a third of the commands. Every command latches, in order, at its edge + latency."""
    cfg = _g26_config("controller_interface.tick_jitter_s=5e-6")
    for lat in (4, 9, 14):
        c = CompiledController("count", count_init, count_step, latency_ticks=lat)
        r = run(cfg, c, duration_s=1e-3, batch=2, seeds=1, uncertainty="nominal", variation="off", oracle=False,
                histogram=False, record=[("heater_code", 1, "last")])
        d = r.diagnostics
        assert np.all(d["pipeline_overflow"] == 0), (lat, d["pipeline_overflow"])
        reg = r.trace["heater_code@1:last"]["value"][:, :, 0]
        for b in range(2):
            steps = np.diff(reg[:, b])
            changes = steps[steps != 0]
            assert changes[0] == 1010 and np.all(changes[1:] == 10)   # consecutive codes: no command skipped
        edges = int(c.state[2][0, 0])
        assert int(reg[-1, 0]) >= 1000 + 10 * (edges - (lat // 5 + 2))


def test_advance_keeps_every_command():
    """Plant.advance issued more often than the controller period with a controller latency: every command latches
    exactly latency ticks after it was issued (the pipeline grows; a fixed one would drop three of four commands)."""
    from mrm.control.api import ControllerSpec
    from mrm.plant.plant import Plant
    from mrm.scenarios import registry as scenario_registry
    cfg = _g26_config()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp = params.realize(cfg, 1, seeds=1, uncertainty="nominal", variation="off")
    sc = scenario_registry.from_config(cfg)
    spec = ControllerSpec.from_config(cfg, n_batch=1, nominal=cfg, scenario_drives=sc.drives,
                                      scenario_offsets=sc.offsets)
    pl = Plant(pp, sc, spec, controller_latency_ticks=3, oracle=False, histogram=False,
               record=[("heater_code", 1, "last")], record_ticks=40)
    pl.reset(protocol="branch_low")
    for i in range(40):
        pl.advance({"tx.heater": np.full((1, 8), 1000 + i, np.int64)}, 1)
    reg = pl.traces()["heater_code@1:last"]["value"][:, 0, 0]
    np.testing.assert_array_equal(reg[3:], 1000 + np.arange(37))
    assert np.all(reg[:3] == 0)
    assert pl.diagnostics()["fifo_capacities"]["pipeline"] >= 4


# ----------------------------------------------------------------------------------------------- Plant and registry API

def test_reset_repeats_an_episode_with_a_bound_compiled_controller():
    """Plant.reset() restarts a bound compiled controller from its post-init state (the state fault.controller_reset
    restores), so a second episode after reset() equals the first exactly (gym-style reuse of one Plant). reset()
    must not overwrite the saved post-init state with the end-of-episode state."""
    from mrm.control.api import ControllerSpec
    from mrm.plant.plant import Plant
    from mrm.scenarios import registry as scenario_registry
    from mrm.sim import kernel
    cfg = _g26_config("sim.reset_ramp_s=1e-4")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp = params.realize(cfg, 2, seeds=3)
    sc = scenario_registry.from_config(cfg)
    spec = ControllerSpec.from_config(cfg, n_batch=2, nominal=cfg, scenario_drives=sc.drives,
                                      scenario_offsets=sc.offsets)
    pl = Plant(pp, sc, spec, noise_seed=2, oracle=False, histogram=False, record=[("heater_code", 1, "last"),
                                                                                  ("t_ring_k", 1, "last")],
               record_ticks=300)
    c = CompiledController("count", count_init, count_step)
    c.reset(spec)
    pl.bind_compiled(c)
    out = []
    for _ in range(2):
        pl.reset()
        pl.run(300, mode=kernel.MODE_COMPILED, step_fn=c.step_fn)
        tr = pl.traces()
        out.append((tr["heater_code@1:last"]["value"].copy(), tr["t_ring_k@1:last"]["value"].copy(),
                    c.state[2].copy(), pl.metrics()["heater_energy_j"].copy()))
    for a, b in zip(out[0], out[1]):
        np.testing.assert_array_equal(a, b)
    assert int(out[0][2][0, 0]) == 30


class _Hold:
    """Registry test controller: holds every heater at ``code`` (constructor parameter)."""
    name = "engine_test_hold"
    latency_ticks = 0

    def __init__(self, code=0):
        self.code = int(code)

    def reset(self, spec):
        self.spec = spec

    def step(self, obs):
        return {"tx.heater": np.full((self.spec.n_batch, self.spec.n_rings), self.code, np.int64)}


def test_run_creates_registered_controllers_with_params():
    """run(cfg, "<registry name>", controller_params={...}) builds the controller through
    mrm.control.registry.create(name, params) (the mapping is its second argument, not keywords)."""
    from mrm.control import registry
    registry.register("engine_test_hold", _Hold, overwrite=True)
    try:
        r = run(_g26_config(), "engine_test_hold", duration_s=50e-6, controller_params={"code": 12345}, batch=1,
                seeds=1, uncertainty="nominal", variation="off", oracle=False, histogram=False)
    finally:
        registry.unregister("engine_test_hold")
    assert r.manifest_inputs["controller"] == "engine_test_hold"
    assert np.all(r.truth_final.heater_code == 12345)
