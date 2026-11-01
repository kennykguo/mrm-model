"""Checks the controller API (mrm.control.api) and the controller registry (mrm.control.registry).

Read this first
---------------
A controller sees the plant only through a ControllerSpec (channels, timing, what it may know) and an Observation
(the sensor values of one controller step). It returns a dict of actuator codes. This file checks, in order:

- ControllerSpec: defaults, timing errors, the fingerprint and the knowledge filter;
- Observation: strict (undeclared channels raise) and read-only;
- the adapters BaseController, PerRing and PerDie give the same commands as a vectorized controller;
- ActionValidator: clipping, saturation counts, casts and errors; telemetry validation;
- CompositeController: several children at different rates, round-robin scheduling and tick jitter;
- the registry, and that the controller core imports nothing from the plant or the simulator (Truth isolation);
- the compiled-controller ABI (CompiledController) with a trivial integer integrator, also inside a numba loop.
"""
import ast
import dataclasses
import pathlib

import numba as nb
import numpy as np
import pytest

from mrm.control import api, registry
from mrm.control.api import (
    ActionRangeError, ActionValidator, BaseController, CompiledController, CompositeController, Controller,
    ControllerSpec, Observation, PerDie, PerRing, UndeclaredChannelError,
)
from mrm.core.channels import TEMP_ID_BASE, ActId, BindingError, ChannelError, ChannelHardware, ObsId, TempSensorHw
from mrm.core.timing import TimingError

ROOT = pathlib.Path(__file__).resolve().parents[2]


def make_spec(**kw):
    args = dict(n_batch=2, n_rings=4, n_lines=4, tick_s=1e-6, update_rate_hz=100e3)
    args.update(kw)
    return ControllerSpec.build(**args)


def rand_obs(spec, tick, r, *, count=None):
    """A random but self-consistent observation: counts <= M, samples beyond the count are zero."""
    sensors, new = {}, {}
    for n, c in spec.sensors.items():
        shp = c.resolve_shape(spec.dims)
        if c.np_dtype == bool:
            sensors[n] = r.integers(0, 2, shp).astype(bool)
        else:
            sensors[n] = r.integers(max(c.min, 0), min(c.max, 5000) + 1, shp).astype(c.np_dtype)
        new[n] = r.integers(0, 2, (spec.n_batch,) + c.flag_shape(spec.dims)).astype(bool)
    m = spec.samples_per_step
    cnt = r.integers(0, m + 1, (spec.n_batch, spec.n_rings)) if count is None else np.full(
        (spec.n_batch, spec.n_rings), count)
    sensors["tx.monitor.count"] = cnt.astype(np.int32)
    valid = np.arange(m)[None, None, :] < cnt[:, :, None]
    sensors["tx.monitor.samples"] = np.where(valid, sensors["tx.monitor.samples"], 0).astype(np.int32)
    ticks = tick - (m - np.arange(m)) * spec.adc.period_ticks
    sensors["tx.monitor.sample_ticks"] = np.where(valid, ticks[None, None, :], 0).astype(np.int64)
    return Observation(tick, spec.tick_s, sensors, new)


# ------------------------------------------------------------------------------------------------------- ControllerSpec

def test_spec_build_defaults():
    s = make_spec()
    assert (s.period_ticks, s.samples_per_step) == (10, 3) and s.rate_hz == pytest.approx(100e3, rel=1e-12)
    assert (s.adc.period_ticks, s.adc.latency_ticks, s.adc.integration_window_ticks) == (4, 2, 2)
    assert s.adc.full_scale_codes == 65535 and s.heater_dac.code_max == 65535
    assert s.bias_dac.phys_min == 1.0 and s.bias_dac.phys_max == 3.75
    assert s.temp_sensors[0].period_ticks == 1000 and s.temp_sensors[0].latency_ticks == 1000
    assert s.error_count_period_ticks == 2_000_000
    assert s.samples_per_tick == s.samples_per_step
    assert set(s.actuators) == {"tx.heater", "datapath.bitstat_threshold", "monitor.pga_select"}
    assert s.dims == {"B": 2, "N": 4, "L": 4, "M": 3, "S": 1}
    assert not hasattr(s, "extras")


def test_spec_timing_errors_name_every_key():
    hw = ChannelHardware(adc_rate_hz=300e3, heater_latency_s=1.5e-6)
    with pytest.raises(TimingError) as ei:
        make_spec(hw=hw, update_rate_hz=3e6)
    msg = str(ei.value)
    for key in ("ring.monitor.adc_rate_hz", "ring.heater.latency_s", "controller_interface.update_rate_hz"):
        assert key in msg
    with pytest.raises(TimingError, match="phase"):
        make_spec(phase_ticks=10)


def test_spec_rejects_bad_settings():
    with pytest.raises(ValueError, match="schedule"):
        make_spec(schedule="sometimes")
    with pytest.raises(ValueError, match="on_out_of_range"):
        make_spec(on_out_of_range="wrap")
    with pytest.raises(ValueError, match="bus_order"):
        make_spec(bus_order=[0, 0, 1, 2])
    with pytest.raises(BindingError):
        make_spec(bind={"tx.bias": "controller"}, scenario_drives=["tx.bias"])
    with pytest.raises(ChannelError, match="expose"):
        make_spec(expose=["truth.detuning"])


def test_spec_fingerprint_and_knowledge_filter():
    grid = [229.0e12 + 0.4e12 * i for i in range(4)]
    a = make_spec(line_grid_hz=grid, knowledge=("ring_to_line",), nominal={"dlambda_dt_nominal_pm_per_k": 70.0})
    b = make_spec(line_grid_hz=grid, knowledge=("ring_to_line",), nominal={"dlambda_dt_nominal_pm_per_k": 70.0})
    assert a == b and a.fingerprint() == b.fingerprint()
    assert a.line_grid_hz is None                                   # line_grid not in knowledge
    c = make_spec(line_grid_hz=grid, knowledge=("line_grid",), nominal={"dlambda_dt_nominal_pm_per_k": 70.0})
    assert c.line_grid_hz == tuple(grid) and c.fingerprint() != a.fingerprint()
    # The nominal design enters the fingerprint only through what the knowledge list releases. A released key that
    # differs changes the fingerprint; a withheld key does not. So the spec is the same in every world and for every
    # seed.
    d = make_spec(line_grid_hz=grid, knowledge=("ring_to_line",), nominal={"dlambda_dt_nominal_pm_per_k": 86.0})
    assert d.fingerprint() == a.fingerprint()
    e1, e2 = (make_spec(line_grid_hz=grid, knowledge=("ring_to_line",), nominal={"channel_plan.ring_to_line": r2l})
              for r2l in ([0, 1, 2, 3], [3, 2, 1, 0]))
    assert e1.fingerprint() != e2.fingerprint()


def test_with_period_recomputes_m_and_layout():
    s = make_spec()
    t = s.with_period(100)
    assert t.samples_per_step == 25 and t.period_ticks == 100
    assert t.layout.obs_field("tx.monitor.samples").size == 4 * 25
    assert s.layout.fingerprint() != t.layout.fingerprint()


def test_spec_from_config_mapping():
    cfg = {"sim.dt_plant_s": 1e-6, "array.n_rings": 8, "channel_plan.n_lines": 8,
           "controller_interface.update_rate_hz": 50e3, "controller_interface.bind.tx.bias": "controller",
           "controller_interface.knowledge": ["ring_to_line"], "channel_plan.ring_to_line": list(range(8)),
           "controller_interface.on_out_of_range": "error", "ring.monitor.adc_rate_hz": 1e6}
    s = ControllerSpec.from_config(cfg, n_batch=3)
    assert (s.n_batch, s.n_rings, s.n_lines, s.period_ticks, s.samples_per_step) == (3, 8, 8, 20, 20)
    assert s.bindings["tx.bias"] == "controller" and "tx.bias" in s.actuators
    assert s.ring_to_line == tuple(range(8)) and s.on_out_of_range == "error"


# ---------------------------------------------------------------------------------------------------------- Observation

def test_observation_is_strict_and_read_only():
    s = make_spec()
    obs = rand_obs(s, 1234, np.random.default_rng(0))
    assert obs.tick == 1234 and obs.t_s == 1234 * 1e-6
    with pytest.raises(UndeclaredChannelError, match="not declared"):
        obs["truth.detuning_pm"]
    with pytest.raises(KeyError):                                   # also a KeyError for generic code
        obs.new["ring.q_loaded"]
    with pytest.raises(ValueError):
        obs.sensors["tx.monitor.latest"][0, 0] = 1
    assert obs.active_ring == -1
    o2 = Observation(5, 1e-6, {"tx.monitor.latest": np.zeros((1, 4))}, active=np.array([0, 0, 1, 0], bool))
    assert o2.active_ring == 2


def test_spec_without_exposed_sensor_hides_it():
    s = make_spec(expose=["tx.monitor.latest", "tx.monitor.count", "tx.monitor.samples", "tx.monitor.sample_ticks"])
    assert "temp.die" not in s.channels
    si, _ = api.spec_arrays(s)
    assert api.obs_offset(si, TEMP_ID_BASE) == -1


# -------------------------------------------------------------------------------------- BaseController, PerRing, PerDie

class VecI(BaseController):
    """Vectorized integrator on the mean of the valid samples."""
    name = "vec_i"

    @dataclasses.dataclass
    class Params:
        ref: int = 1000
        spread: int = 100

    def reset(self, spec):
        super().reset(spec)
        self.acc = self.zeros("B", "N", dtype=np.int64)

    def step(self, obs):
        x = obs["tx.monitor.samples"].astype(np.int64)
        cnt = obs["tx.monitor.count"].astype(np.int64)
        has = cnt > 0
        mean = np.where(has, x.sum(axis=2) // np.maximum(cnt, 1), 0)
        self.acc += np.where(has, mean - self.params.ref, 0)
        if obs.active is not None:
            pass
        return {"tx.heater": self.acc + self.params.spread * np.arange(self.spec.n_rings)[None, :]}

    def telemetry(self):
        return {"integrator": self.acc.astype(float)}


class ScalarI:
    def __init__(self, ctx, ref=1000, spread=100):
        self.acc, self.n, self.ref, self.spread = 0, ctx.n, ref, spread

    def step(self, robs):
        x = robs.sensors["tx.monitor.samples"]
        assert x.shape == (int(robs.sensors["tx.monitor.count"]),)
        if len(x):
            self.acc += int(x.astype(np.int64).sum()) // len(x) - self.ref
        return {"tx.heater": self.acc + self.spread * self.n}

    def telemetry(self):
        return {"integrator": self.acc}


class DieI:
    def __init__(self, ctx):
        self.acc = np.zeros(ctx.spec.n_rings, dtype=np.int64)

    def step(self, dobs):
        x = dobs.sensors["tx.monitor.samples"].astype(np.int64)
        cnt = dobs.sensors["tx.monitor.count"].astype(np.int64)
        has = cnt > 0
        self.acc += np.where(has, x.sum(axis=1) // np.maximum(cnt, 1) - 1000, 0)
        return {"tx.heater": self.acc + 100 * np.arange(len(cnt))}


def test_base_controller_params():
    c = VecI({"ref": 7})
    assert c.params.ref == 7 and c.params.spread == 100
    assert VecI(ref=3).params.ref == 3
    with pytest.raises(TypeError, match="allowed"):
        VecI({"gain": 1.0})
    with pytest.raises(RuntimeError):
        VecI().zeros("B", "N")


def test_per_ring_and_per_die_equal_vectorized():
    s = make_spec()
    r = np.random.default_rng(1)
    vec, ring, die = VecI(), PerRing(ScalarI), PerDie(DieI)
    for c in (vec, ring, die):
        c.reset(s)
    for k in range(30):
        obs = rand_obs(s, 10 * k, r)
        a, b, d = vec.step(obs), ring.step(obs), die.step(obs)
        assert np.array_equal(a["tx.heater"], b["tx.heater"]) and np.array_equal(a["tx.heater"], d["tx.heater"])
    assert np.array_equal(vec.telemetry()["integrator"], ring.telemetry()["integrator"])
    for c in (vec, ring, die):
        assert isinstance(c, Controller)


def test_per_ring_round_robin_and_hold():
    s = make_spec()
    r = np.random.default_rng(2)
    ring = PerRing(ScalarI)
    ring.reset(s)
    obs = rand_obs(s, 0, r, count=s.samples_per_step)
    first = ring.step(obs)["tx.heater"].copy()
    active = np.zeros(4, bool)
    active[2] = True
    o2 = rand_obs(s, 10, r, count=s.samples_per_step)
    o2 = Observation(o2.tick, o2.dt_s, dict(o2.sensors), dict(o2.new), active)
    second = ring.step(o2)["tx.heater"]
    changed = np.any(second != first, axis=0)
    assert not changed[[0, 1, 3]].any()


def test_per_ring_rejects_bad_writes():
    s = make_spec()

    class Floaty:
        def __init__(self, ctx):
            pass

        def step(self, robs):
            return {"tx.heater": 1.5}

    class Truthy:
        def __init__(self, ctx):
            pass

        def step(self, robs):
            return {"tx.bias": 3}

    for cls, err in ((Floaty, TypeError), (Truthy, UndeclaredChannelError)):
        c = PerRing(cls)
        c.reset(s)
        with pytest.raises(err):
            c.step(rand_obs(s, 0, np.random.default_rng(0)))


# ----------------------------------------------------------------------------------------- action validation, telemetry

def test_validator_clip_counts_and_casts():
    s = make_spec()
    v = ActionValidator(s)
    out = v.validate({"tx.heater": np.array([[70000, -5, 2, 3], [0, 1, 2, 65535]])})
    assert out["tx.heater"].dtype == np.uint16
    assert out["tx.heater"].tolist() == [[65535, 0, 2, 3], [0, 1, 2, 65535]]
    v.validate({"tx.heater": np.full((2, 4), 99999)})
    cnt = v.saturation_counts()["tx.heater"]
    assert cnt.tolist() == [[2, 2, 1, 1], [1, 1, 1, 1]]
    v.reset()
    assert not v.saturation_counts()["tx.heater"].any()
    big = np.full((2, 4), 2**63, dtype=np.uint64)
    assert v.validate({"tx.heater": big})["tx.heater"].max() == 65535


@pytest.mark.parametrize("actions, err, match", [
    ({"tx.heater": np.zeros((2, 4)) + 0.5}, TypeError, "integer codes"),
    ({"tx.heater": np.zeros((4, 2), dtype=np.int64)}, ChannelError, "expected shape"),
    ({"tx.heaterr": np.zeros((2, 4), dtype=np.int64)}, UndeclaredChannelError, "unknown channel"),
    ({"tx.monitor.latest": np.zeros((2, 4), dtype=np.int64)}, ChannelError, "sensor"),
    ({"tx.bias": np.zeros((2, 4), dtype=np.int64)}, BindingError, "bound to the scenario"),
    (None, TypeError, "dict"),
])
def test_validator_errors(actions, err, match):
    with pytest.raises(err, match=match):
        ActionValidator(make_spec()).validate(actions)


def test_validator_error_mode():
    v = ActionValidator(make_spec(on_out_of_range="error"))
    with pytest.raises(ActionRangeError, match=r"tx\.heater\[1, 2\] = 70000"):
        v.validate({"tx.heater": np.array([[0, 0, 0, 0], [0, 0, 70000, 0]])})
    assert v.validate({"monitor.pga_select": np.zeros((2, 4), dtype=np.int32)})["monitor.pga_select"].dtype == np.int32


def test_validate_telemetry():
    s = make_spec()
    t = api.validate_telemetry({"locked": np.ones((2, 4), bool), "integrator": np.zeros((2, 4))}, s)
    assert t["locked"].dtype == np.float64
    with pytest.raises(ChannelError):
        api.validate_telemetry({"state": np.zeros(4)}, s)
    with pytest.raises(TypeError):
        api.validate_telemetry({"state": np.array([["a"] * 4] * 2)}, s)
    assert set(api.TELEMETRY_KEYS) >= {"state", "locked", "detuning_est_pm", "integrator"}
    assert api.controller_telemetry(object()) == {} and api.controller_fast_kernel(object()) is None


# -------------------------------------------------------------------------------------------------- CompositeController

class Recorder(BaseController):
    """Writes a constant per ring and records every observation it receives."""
    name = "recorder"

    @dataclasses.dataclass
    class Params:
        value: int = 0

    def reset(self, spec):
        super().reset(spec)
        self.seen = []

    def step(self, obs):
        self.seen.append({k: np.array(obs.sensors[k]) for k in obs.sensors} | {"_tick": obs.tick,
                         "_new": {k: np.array(obs.new[k]) for k in obs.new}, "_active": obs.active})
        return {"tx.heater": np.full((self.spec.n_batch, self.spec.n_rings), self.params.value + len(self.seen))}

    def telemetry(self):
        return {"state": np.full((self.spec.n_batch, self.spec.n_rings), len(self.seen))}


def test_composite_single_child_equals_child():
    s = make_spec()
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    alone, inner = VecI(), VecI()
    alone.reset(s)
    comp = CompositeController([(inner, 100e3)])
    comp.reset(s)
    for k in range(25):
        a = alone.step(rand_obs(s, 10 * k, r1))
        b = comp.step(rand_obs(s, 10 * k, r2))
        assert np.array_equal(a["tx.heater"], b["tx.heater"])
    assert isinstance(comp, Controller)


def test_composite_multirate_accumulation():
    s = make_spec()
    fast, slow = Recorder({"value": 0}), Recorder({"value": 1000})
    comp = CompositeController([(fast, 100e3), (slow, 10e3)])
    assert comp.base_period_ticks(1e-6) == 10
    comp.reset(s)
    r = np.random.default_rng(4)
    obs_list = [rand_obs(s, 10 * k, r) for k in range(21)]
    outs = [comp.step(o) for o in obs_list]
    assert len(fast.seen) == 21 and len(slow.seen) == 3                     # ticks 0, 100, 200
    assert [d["_tick"] for d in slow.seen] == [0, 100, 200]
    win = obs_list[1:11]                                                    # ticks 10..100
    got = slow.seen[1]
    m_c = 25
    want_cnt = np.minimum(sum(o["tx.monitor.count"].astype(int) for o in win), m_c)
    assert np.array_equal(got["tx.monitor.count"], want_cnt)
    for b in range(2):
        for i in range(4):
            seq = np.concatenate([o["tx.monitor.samples"][b, i, :o["tx.monitor.count"][b, i]] for o in win])
            assert np.array_equal(got["tx.monitor.samples"][b, i, :want_cnt[b, i]], seq[-m_c:])
    assert np.array_equal(got["tx.bits"], sum(o["tx.bits"] for o in win))
    assert np.array_equal(got["tx.monitor.latest"], win[-1]["tx.monitor.latest"])
    assert np.array_equal(got["adc_overrange"], np.any([o["adc_overrange"] for o in win], axis=0))
    assert np.array_equal(got["monitor_valid"], np.all([o["monitor_valid"] for o in win], axis=0))
    assert np.array_equal(got["_new"]["temp.die"], np.any([o.new["temp.die"] for o in win], axis=0))
    # the slow child writes last, so at its edges its command wins; in between the fast child's command stands
    assert outs[10]["tx.heater"][0, 0] == 1000 + 2 and outs[11]["tx.heater"][0, 0] == 12
    tel = comp.telemetry()
    assert "recorder0.state" in tel and "recorder1.state" in tel


def test_composite_round_robin():
    s = make_spec()
    child = Recorder()
    comp = CompositeController([(child, 100e3)], schedule="round_robin")
    comp.reset(s)
    r = np.random.default_rng(5)
    prev = None
    for k in range(8):
        out = comp.step(rand_obs(s, 10 * k, r))["tx.heater"]
        assert child.seen[-1]["_active"].tolist() == [i == k % 4 for i in range(4)]
        if prev is not None:
            changed = np.flatnonzero(np.any(out != prev, axis=0)).tolist()
            assert changed == [k % 4]
        prev = out.copy()


def test_composite_timing_checks():
    s = make_spec()
    with pytest.raises(TimingError, match="children"):
        CompositeController([(Recorder(), 30e3)]).reset(s)          # 33.3 ticks
    with pytest.raises(TimingError, match="base grid"):
        CompositeController([(Recorder(), 40e3)]).reset(s)          # 25 ticks, not a multiple of 10
    assert CompositeController([(Recorder(), 40e3), (Recorder(), 100e3)]).base_period_ticks(1e-6) == 5
    with pytest.raises(TypeError):
        CompositeController([(object(), 1e3)])


def test_composite_from_registry():
    registry.register("test_recorder", Recorder, overwrite=True)
    try:
        comp = registry.create("composite", {"children": [{"name": "test_recorder", "rate_hz": 100e3,
                                                            "params": {"value": 5}}], "schedule": "simultaneous"})
        assert isinstance(comp, CompositeController) and comp.children[0][0].params.value == 5
    finally:
        registry.unregister("test_recorder")


# ------------------------------------------------------------------------------------------------------------- registry

def test_registry_register_resolve_create():
    registry.register("test_vec_i", VecI)
    try:
        assert registry.resolve("test_vec_i") is VecI
        c = registry.create("test_vec_i", {"ref": 3})
        assert isinstance(c, VecI) and c.params.ref == 3
        assert "test_vec_i" in registry.available() and "composite" in registry.available()
        with pytest.raises(registry.RegistryError, match="already registered"):
            registry.register("test_vec_i", Recorder)
        registry.register("test_vec_i", Recorder, overwrite=True)
        assert registry.resolve("test_vec_i") is Recorder
    finally:
        registry.unregister("test_vec_i")
    assert registry.resolve("mrm.control.api:CompositeController") is CompositeController
    with pytest.raises(registry.RegistryError, match="unknown controller"):
        registry.resolve("no_such_controller")
    with pytest.raises(registry.RegistryError, match="cannot import"):
        registry.resolve("no_such_module_xyz:Thing")
    with pytest.raises(registry.RegistryError, match="lower-case"):
        registry.register("Bad Name", VecI)


def test_registry_refuses_truth_fed_controllers():
    class Oracle(BaseController):
        truth_fed = True

    with pytest.raises(registry.RegistryError, match="Truth-fed"):
        registry.register("oracle", Oracle)
    assert "oracle" not in registry.available()


def test_control_core_modules_do_not_import_truth():
    """Truth isolation: the controller core modules import nothing from the plant, the simulator or the validation
    oracle, so a controller cannot read the true plant state."""
    files = ["mrm/control/api.py", "mrm/control/registry.py", "mrm/control/fxp.py", "mrm/control/lfsr.py",
             "mrm/control/knowledge.py", "mrm/core/channels.py", "mrm/core/timing.py", "mrm/core/rng.py"]
    for f in files:
        tree = ast.parse((ROOT / f).read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for n in names:
                assert not n.startswith(("mrm.plant", "mrm.sim", "mrm.validation", "mrm.params.realize")), (f, n)
                assert "truth" not in n.lower(), (f, n)


# ---------------------------------------------------------------------------------------------- compiled-controller ABI

LATEST = int(ObsId.TX_MONITOR_LATEST)
HEATER = int(ActId.TX_HEATER)

# Integer params p of the toy integrator, as integrator_init fills them
_P_N_RINGS = 0          # number of rings
_P_LATEST_OFF = 1       # offset of tx.monitor.latest in the observation vector
_P_HEATER_OFF = 2       # offset of tx.heater in the action vector
_P_SETPOINT = 3         # set-point, ADC codes
_P_KI = 4               # integral gain, integer
_P_SHIFT = 5            # right shift from the integrator to the heater code
_P_CODE_MIN = 6         # heater code range, codes
_P_CODE_MAX = 7


@nb.njit
def integrator_init(spec_i64, spec_f64):
    """Trivial compiled integer integrator: code = clamp((sum ki * (setpoint - latest)) >> shift).

    User integers: [setpoint, ki, shift]. User float 0 is added to sf[0] each step (it checks the float path).
    State s: [step count, integrator of ring 0 .. N-1].
    """
    n = spec_i64[api.SPEC_I_N_RINGS]
    p = np.zeros(8, dtype=np.int64)
    p[_P_N_RINGS] = n
    p[_P_LATEST_OFF] = api.obs_offset(spec_i64, LATEST)
    p[_P_HEATER_OFF] = api.act_offset(spec_i64, HEATER)
    p[_P_SETPOINT] = api.user_i64(spec_i64, 0)
    p[_P_KI] = api.user_i64(spec_i64, 1)
    p[_P_SHIFT] = api.user_i64(spec_i64, 2)
    p[_P_CODE_MIN] = spec_i64[api.SPEC_I_HEATER_MIN]
    p[_P_CODE_MAX] = spec_i64[api.SPEC_I_HEATER_MAX]
    pf = np.zeros(1)
    pf[0] = api.user_f64(spec_f64, 0)
    return p, pf, np.zeros(1 + n, dtype=np.int64), np.zeros(1)


@nb.njit
def integrator_step(p, pf, s, sf, obs, act):
    s[0] += 1
    for i in range(p[_P_N_RINGS]):
        s[1 + i] += p[_P_KI] * (p[_P_SETPOINT] - obs[p[_P_LATEST_OFF] + i])
        code = s[1 + i] >> p[_P_SHIFT]
        act[p[_P_HEATER_OFF] + i] = min(max(code, p[_P_CODE_MIN]), p[_P_CODE_MAX])
    sf[0] += pf[0]


def integrator_reference(latest_seq, setpoint, ki, shift):
    """Plain NumPy version of integrator_step: the heater codes after each observation."""
    acc = np.zeros_like(latest_seq[0], dtype=np.int64)
    outs = []
    for lat in latest_seq:
        acc += ki * (setpoint - lat.astype(np.int64))
        outs.append(np.clip(acc >> shift, 0, 65535))
    return outs


def test_spec_arrays_tables():
    s = make_spec(bind={"tx.bias": "controller"})
    si, sf = api.spec_arrays(s, [7, 8, 9], [0.5], latency_ticks=3)
    assert si[api.SPEC_I_N_RINGS] == 4 and si[api.SPEC_I_M] == 3 and si[api.SPEC_I_LATENCY_TICKS] == 3
    assert si[api.SPEC_I_OBS_SIZE] == s.layout.obs_size and si[api.SPEC_I_ACT_SIZE] == s.layout.act_size
    assert api.obs_offset(si, LATEST) == s.layout.obs_field("tx.monitor.latest").offset
    assert api.new_offset(si, int(ObsId.TX_MONITOR_COUNT)) == s.layout.new_field("tx.monitor.count").offset
    assert api.act_offset(si, int(ActId.TX_BIAS)) == s.layout.act_field("tx.bias").offset
    assert api.act_offset(si, int(ActId.TX_ASSIGNMENT)) == -1 and api.obs_offset(si, 999) == -1
    assert [api.user_i64(si, k) for k in range(3)] == [7, 8, 9] and api.user_f64(sf, 0) == 0.5
    assert sf[api.SPEC_F_TICK_S] == 1e-6 and sf[api.SPEC_F_PERIOD_S] == pytest.approx(1e-5)


def test_compiled_controller_python_path_matches_reference():
    s = make_spec()
    cc = CompiledController("ic", integrator_init, integrator_step, params_i64=[30000, 3, 4], params_f64=[0.25],
                            latency_ticks=1)
    assert isinstance(cc, Controller) and cc.fast_kernel is None
    cc.reset(s)
    assert cc.state[0].shape == (2, 8) and cc.state[2].shape == (2, 5)
    r = np.random.default_rng(6)
    obs_seq = [rand_obs(s, 10 * k, r) for k in range(20)]
    outs = [cc.step(o)["tx.heater"].copy() for o in obs_seq]
    ref = integrator_reference([o["tx.monitor.latest"] for o in obs_seq], 30000, 3, 4)
    for a, b in zip(outs, ref):
        assert np.array_equal(a, b)
    assert np.all(cc.state[2][:, 0] == 20) and np.allclose(cc.state[3][:, 0], 5.0)
    assert outs[0].dtype == np.int64                    # unvalidated: ActionValidator clips, counts and casts
    assert ActionValidator(s).validate({"tx.heater": outs[0]})["tx.heater"].dtype == np.uint16


@nb.njit
def hold_step(p, pf, s, sf, obs, act):
    """Writes the heater only on even steps: act must hold the previous command in between."""
    s[0] += 1
    if s[0] % 2 == 0:
        for i in range(p[_P_N_RINGS]):
            act[p[_P_HEATER_OFF] + i] = act[p[_P_HEATER_OFF] + i] + 10


def test_compiled_act_holds_previous_command():
    s = make_spec()
    cc = CompiledController("hold", integrator_init, hold_step, params_i64=[0, 0, 0], params_f64=[0.0])
    cc.reset(s)
    r = np.random.default_rng(7)
    obs = [rand_obs(s, 10 * k, r) for k in range(6)]
    w = int(obs[0]["tx.heater.applied"][0, 0])                    # warm start from the latched register
    seq = [int(cc.step(o)["tx.heater"][0, 0]) for o in obs]
    assert seq == [w, w + 10, w + 10, w + 20, w + 20, w + 30]


def test_compiled_act_starts_from_applied_readback():
    s = make_spec()
    cc = CompiledController("hold", integrator_init, hold_step, params_i64=[0, 0, 0], params_f64=[0.0])
    cc.reset(s)
    obs = rand_obs(s, 0, np.random.default_rng(8))
    warm = np.array(obs["tx.heater.applied"])
    assert np.array_equal(cc.step(obs)["tx.heater"], warm)          # odd step: holds the warm-start registers


@nb.njit
def toy_loop(init_fn, step_fn, spec_i64, spec_f64, n_steps, latest_off, heater_off, n):
    """A closed loop inside numba with the controller passed as first-class functions (as in the compiled kernel)."""
    p_i, p_f, s_i, s_f = init_fn(spec_i64, spec_f64)
    obs = np.zeros(spec_i64[api.SPEC_I_OBS_SIZE], dtype=np.int64)
    act = np.zeros(spec_i64[api.SPEC_I_ACT_SIZE], dtype=np.int64)
    for k in range(n_steps):
        obs[0] = k * spec_i64[api.SPEC_I_PERIOD_TICKS]
        for i in range(n):
            obs[latest_off + i] = (act[heater_off + i] * 3) // 4 + 100 * i     # toy plant
        step_fn(p_i, p_f, s_i, s_f, obs, act)
    return act[heater_off:heater_off + n].copy()


def test_compiled_controller_in_numba_loop():
    s = make_spec(n_batch=1)
    cc = CompiledController("ic", integrator_init, integrator_step, params_i64=[30000, 1, 3], params_f64=[0.0])
    cc.reset(s)
    lo, ho = s.layout.obs_field("tx.monitor.latest").offset, s.layout.act_field("tx.heater").offset
    final = toy_loop(integrator_init, integrator_step, cc.spec_i64, cc.spec_f64, 400, lo, ho, 4)
    # Python reference of the same loop
    acc = np.zeros(4, dtype=np.int64)
    heat = np.zeros(4, dtype=np.int64)
    for _ in range(400):
        lat = (heat * 3) // 4 + 100 * np.arange(4)
        acc += 1 * (30000 - lat)
        heat = np.clip(acc >> 3, 0, 65535)
    assert np.array_equal(final, heat)
    assert np.all(np.abs((final * 3) // 4 + 100 * np.arange(4) - 30000) <= 2)    # the integrator converged


def test_call_step_batch_matches_rows():
    s = make_spec()
    cc = CompiledController("ic", integrator_init, integrator_step, params_i64=[100, 1, 0], params_f64=[0.0])
    cc.reset(s)
    obs = np.zeros((2, s.layout.obs_size), dtype=np.int64)
    obs[1, s.layout.obs_field("tx.monitor.latest").offset] = 40
    act = cc.initial_act(s)
    p_i, p_f, s_i, s_f = (x.copy() for x in cc.state)
    api.call_step_batch(integrator_step, p_i, p_f, s_i, s_f, obs, act)
    ho = s.layout.act_field("tx.heater").offset
    assert act[0, ho] == 100 and act[1, ho] == 60


def test_fast_kernel_layout_helpers():
    n = 8
    assert api.fast_obs_size(n) == 2 + 3 * n and api.fast_act_size(n) == 2 * n
    idx = sorted([api.fo_new(i, n) for i in range(n)] + [api.fo_code(i, n) for i in range(n)]
                 + [api.fo_sample_tick(i, n) for i in range(n)])
    assert idx == list(range(api.FO_HEADER, api.fast_obs_size(n)))
    assert sorted([api.fa_heater(i, n) for i in range(n)] + [api.fa_bias(i, n) for i in range(n)]) == list(range(2 * n))


def test_compiled_init_must_return_flat_arrays():
    @nb.njit
    def bad_init(si, sf):
        return np.zeros(2), np.zeros(2), np.zeros(2, dtype=np.int64), np.zeros(1)

    cc = CompiledController("bad", bad_init, integrator_step)
    with pytest.raises(TypeError, match="p_i64"):
        cc.reset(make_spec())


# ----------------------------------------------------------------------------------- more spec, ABI and registry checks

MINI = ROOT / "tests" / "data" / "mini.yaml"


def _zero_obs(spec, tick=0):
    sens = {n: np.zeros(c.resolve_shape(spec.dims), c.np_dtype) for n, c in spec.sensors.items()}
    new = {n: np.zeros((spec.n_batch,) + c.flag_shape(spec.dims), bool) for n, c in spec.sensors.items()}
    return Observation(tick, spec.tick_s, sens, new)


def test_from_config_with_the_mini_preset():
    """ControllerSpec.from_config on a real mrm.params Config (not only a mapping): the nominal grid follows realize's
    rule (centre/spacing preset), ring_to_line defaults to the identity, and the spec reads no physical leaf. So a
    different physics alternative leaves its fingerprint unchanged: a controller knows the same in every world."""
    from mrm import params
    from mrm.params.realize import line_frequencies_hz
    know = ["controller_interface.knowledge=[line_grid, ring_to_line]"]
    cfg = params.load(MINI, overrides=know)
    s = ControllerSpec.from_config(cfg, n_batch=2)
    assert s.n_rings == cfg.n_rings and s.n_lines == cfg.n_lines
    assert np.allclose(s.line_grid_hz, line_frequencies_hz(cfg)[0], rtol=0, atol=1.0)
    assert s.ring_to_line == tuple(range(s.n_rings))
    assert s.adc.rate_hz == cfg.si("ring.monitor.adc_rate_hz") and s.adc.period_ticks == 4
    assert s.temp_sensors[0].name == "die0" and "temp.die0" in s.channels
    s2 = ControllerSpec.from_config(params.load(MINI, overrides=know), n_batch=2)
    s3 = ControllerSpec.from_config(params.load(MINI, overrides=know, alts={"ring.optics.fsr_thz": "window_a"}),
                                    n_batch=2)
    assert s.fingerprint() == s2.fingerprint() == s3.fingerprint()
    plain = ControllerSpec.from_config(params.load(MINI), n_batch=2)
    assert plain.line_grid_hz is None and plain.ring_to_line is None          # not in knowledge: withheld


def test_expose_keeps_channel_ids_and_block_counts():
    hw = ChannelHardware(temp_sensors=(TempSensorHw("bank"), TempSensorHw("die")))
    s = make_spec(hw=hw, expose=["temp.die", "tx.monitor.samples"])
    assert "tx.monitor.count" in s.channels                                   # exposed with its block
    si, _ = api.spec_arrays(s)
    assert api.obs_offset(si, TEMP_ID_BASE) == -1                             # temp.bank hidden
    assert api.obs_offset(si, TEMP_ID_BASE + 1) == s.layout.obs_field("temp.die").offset
    assert [t.name for t in s.temp_sensors][1] == "die"                       # index s of TEMP_ID_BASE + s
    t = s.with_period(40)
    assert t.layout.obs_field("temp.die").ch_id == TEMP_ID_BASE + 1



def test_compiled_out_of_range_command_is_clipped_not_wrapped():
    """A compiled controller's -5 / 70000 must reach the validator as such (clip + count), not wrap to 65531."""
    @nb.njit
    def init(si, sf):
        p = np.zeros(2, np.int64)
        p[0] = si[api.SPEC_I_N_RINGS]
        p[1] = api.act_offset(si, HEATER)
        return p, np.zeros(0), np.zeros(0, np.int64), np.zeros(0)

    @nb.njit
    def step(p, pf, s, sf, obs, act):
        act[p[1]] = 70000
        act[p[1] + 1] = -5

    s = make_spec(n_batch=1, n_rings=2, n_lines=2)
    cc = CompiledController("over", init, step)
    cc.reset(s)
    raw = cc.step(_zero_obs(s))
    v = ActionValidator(s)
    out = v.validate(raw)["tx.heater"]
    assert out.tolist() == [[65535, 0]] and out.dtype == np.uint16
    assert v.saturation_counts()["tx.heater"].tolist() == [[1, 1]]
    with pytest.raises(ActionRangeError):
        ActionValidator(s, "error").validate(raw)


def test_tick_jitter_spec_and_composite():
    """Tick jitter of the controller schedule: it is a whole number of ticks and below the period. M (samples per
    step) covers period + jitter, and spec_arrays carries it. A composite called late (within the jitter) still
    steps its children on their nominal edges."""
    with pytest.raises(TimingError, match="tick_jitter_s"):
        make_spec(tick_jitter_s=1.5e-6)
    with pytest.raises(TimingError, match="below the controller period"):
        make_spec(tick_jitter_s=10e-6)
    s = make_spec(tick_jitter_s=3e-6)
    assert s.tick_jitter_ticks == 3 and s.samples_per_step == 4                # ceil((10 + 3) / 4)
    si, _ = api.spec_arrays(s)
    assert si[api.SPEC_I_TICK_JITTER_TICKS] == 3 and si[api.SPEC_I_ADC_WINDOW_TICKS] == 2
    assert si[api.SPEC_I_BIAS_MAX] == 4095 and si[api.SPEC_I_HEATER_LATENCY_TICKS] == 0
    fast, slow = Recorder(), Recorder()
    comp = CompositeController([(fast, 100e3), (slow, 50e3)])
    comp.reset(s)
    r = np.random.default_rng(9)
    late = [0, 3, 1, 2, 0, 3]
    for j, d in enumerate(late):
        comp.step(rand_obs(s, 10 * j + d, r))
    assert [x["_tick"] for x in fast.seen] == [10 * j + d for j, d in enumerate(late)]
    assert [x["_tick"] for x in slow.seen] == [0, 21, 40]                     # nominal edges 0, 20, 40
    with pytest.raises(TimingError, match="base grid"):
        comp.step(rand_obs(s, 65, r))                                          # 5 ticks late > jitter 3


def test_rc_afe_needs_no_whole_tick_window():
    hw = ChannelHardware(afe_type="rc", adc_integration_window_s=2.5e-6)
    s = make_spec(hw=hw)
    assert s.adc.integration_window_ticks == 0
    with pytest.raises(TimingError, match="integration_window_s"):
        make_spec(hw=ChannelHardware(adc_integration_window_s=2.5e-6))


def test_composite_refuses_child_fast_kernel():
    cc = CompiledController("ic", integrator_init, integrator_step, fast_fn=integrator_step)
    with pytest.raises(TypeError, match="fast_kernel"):
        CompositeController([(cc, 100e3)])


def test_ring_to_line_must_index_the_grid():
    with pytest.raises(ValueError, match="ring_to_line"):
        make_spec(ring_to_line=[0, 1, 2, 4], knowledge=("ring_to_line",))


def test_registry_refuses_truth_fed_instances_and_surfaces_broken_baselines(monkeypatch, tmp_path):
    class Oracle(BaseController):
        name = "oracle_like"

        def __init__(self, params=None):
            super().__init__(params)
            self.truth_fed = True

    registry.register("test_factory_oracle", Oracle)
    try:
        with pytest.raises(registry.RegistryError, match="Truth-fed"):
            registry.create("test_factory_oracle")
    finally:
        registry.unregister("test_factory_oracle")

    pkg = tmp_path / "mrm_test_broken_baselines"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("import mrm_test_missing_dependency_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(registry, "_BUILTINS_LOADED", False)
    monkeypatch.setattr(registry, "_BASELINES_PACKAGE", "mrm_test_broken_baselines")
    with pytest.raises(ModuleNotFoundError, match="mrm_test_missing_dependency_xyz"):
        registry.resolve("some_baseline")                     # a broken baseline is not reported as "unknown"
    monkeypatch.setattr(registry, "_BUILTINS_LOADED", False)
    monkeypatch.setattr(registry, "_BASELINES_PACKAGE", "mrm_test_absent_baselines_xyz")
    with pytest.raises(registry.RegistryError, match="unknown controller"):
        registry.resolve("some_baseline")                     # an absent package stays silent
