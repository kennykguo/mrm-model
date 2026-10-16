"""Verification of the batched RL environment mrm.sim.gym_env (07 P1-22).

Spaces from the ChannelSpecs, frame skip = controller period, bounded delta actions (and the bias when the controller
owns it), the observation = the released sensor channels only (no Truth), the reward from the kernel's accumulators
equal to the same reward computed from per-tick Truth records, domain randomization per reset, determinism, NEXT_STEP
autoreset, Gymnasium's env checker, the vector API with Gymnasium wrappers, and step-for-step equality with the
engine's own Python-controller path (mrm.sim.runner.run)."""
from __future__ import annotations

import warnings

import gymnasium as gym
import numpy as np
import pytest

from mrm.control.api import BaseController
from mrm.sim import runner
from mrm.sim.gym_env import (ENV_ID, MRMEnv, MRMSingleEnv, RewardSpec, code_codec)
from mrm.sim.metrics import RecordSpec

BIAS_OV = ["controller_interface.bind.tx.bias=controller", "scenario.bias_v=null"]


@pytest.fixture(scope="module")
def env2():
    e = MRMEnv("g26", num_envs=2, max_episode_steps=6)
    yield e
    e.close()


# ---------------------------------------------------------------------------------------------------- spaces
def test_code_codec_is_a_power_of_two_normalization():
    for lo, hi in [(0, 65535), (0, 4095), (0, 1), (-1, 7), (100, 1123), (0, 0), (-32768, 32767)]:
        c0, s = code_codec(lo, hi)
        codes = np.unique(np.linspace(lo, hi, 257).round().astype(np.int64))
        x = ((codes - c0) * (1.0 / (1 << s))).astype(np.float32)
        assert x.min() >= -1.0 and x.max() < 1.0
        assert (1 << (s + 1)) >= hi - lo + 1 and (s == 0 or (1 << s) < hi - lo + 1)
        back = np.rint(x.astype(np.float64) * (1 << s) + c0).astype(np.int64)
        assert np.array_equal(back, codes)                       # exact in float32


def test_spaces_are_generated_from_the_channel_specs(env2):
    spec = env2.ctrl_spec
    N, M, K = spec.n_rings, spec.samples_per_step, env2.history_blocks
    lay = env2.obs_layout
    names = [c.name for c in lay.channels]
    assert names == ["tx.monitor.samples", "tx.heater.applied", "tx.bias.applied", "temp.die0"]
    assert lay.size == N * K * M + 2 * N + len(spec.temp_sensors)
    assert lay.channel("tx.monitor.samples").shape == (N, K, M)
    for c in lay.channels:
        ch = spec.channels[c.name]
        c0, s = code_codec(int(ch.min), int(ch.max))
        assert (c.code_offset, c.shift) == (c0, s)
        assert c.lo == (int(ch.min) - c0) / 2 ** s and c.hi == (int(ch.max) - c0) / 2 ** s
    sp = env2.single_observation_space
    assert sp.shape == (lay.size,) and sp.dtype == np.float32
    assert env2.observation_space.shape == (2, lay.size)
    # g26: the scenario owns the bias, so the action is the heater only
    assert env2.action_codec.names() == ("tx.heater",)
    assert env2.single_action_space.shape == (N,) and env2.action_space.shape == (2, N)
    assert np.all(env2.single_action_space.low == -1) and np.all(env2.single_action_space.high == 1)
    assert env2.frame_skip_ticks == spec.period_ticks == 10


def test_bias_joins_the_action_when_the_controller_owns_it():
    e = MRMEnv("g26", num_envs=1, overrides=BIAS_OV, max_episode_steps=5, max_delta_bias_codes=32)
    assert e.action_codec.names() == ("tx.heater", "tx.bias")
    N = e.ctrl_spec.n_rings
    assert e.single_action_space.shape == (2 * N,)
    obs, _ = e.reset(seed=0)
    b0 = e.obs_layout.to_codes(obs)["tx.bias.applied"][0].copy()
    a = np.zeros((1, 2 * N), np.float32)
    a[0, N:] = 1.0
    a[0, N + 1] = -0.5
    for _ in range(2):
        obs, *_ = e.step(a)
    b = e.obs_layout.to_codes(obs)["tx.bias.applied"][0]
    exp = np.clip(b0 + 2 * np.rint(a[0, N:] * 32).astype(np.int64), 0, 4095)
    assert np.array_equal(b, exp)
    e.close()


# ---------------------------------------------------------------------------------------------------- stepping
def test_frame_skip_actions_and_clipping(env2):
    obs, info = env2.reset(seed=11)
    N = env2.ctrl_spec.n_rings
    k0 = int(env2.plant.tick)
    h0 = env2.obs_layout.to_codes(obs)["tx.heater.applied"].copy()
    a = np.zeros((2, N), np.float32)
    a[0, 0] = 1.0                 # +256
    a[0, 1] = -0.25               # -64
    a[1, 2] = 0.5 / 256           # rint(0.5) = 0 (half to even)
    a[1, 3] = 1.5 / 256           # rint(1.5) = 2
    obs, r, term, trunc, info = env2.step(a)
    assert int(env2.plant.tick) == k0 + env2.frame_skip_ticks
    assert np.all(info["tick"] == k0 + env2.frame_skip_ticks)
    h1 = env2.obs_layout.to_codes(obs)["tx.heater.applied"]
    d = h1 - h0
    assert d[0, 0] == 256 and d[0, 1] == -64 and d[1, 2] == 0 and d[1, 3] == 2
    assert np.count_nonzero(d) == 3
    assert not term.any() and not trunc.any() and np.all(info["clipped"] == 0)


def test_commands_are_clipped_by_the_kernel_and_counted(env2):
    """Ring 5 of world 0 starts 300 codes below the top rail: +256 twice reaches it; the kernel clips the command
    (07 P1-19), counts the clipped element in info['clipped'], and the clipped value is the next held command (so
    -1 then moves it down from the rail, not from the unclipped sum)."""
    N = env2.ctrl_spec.n_rings
    codes = np.full((2, N), 30000, np.int64)
    codes[0, 5] = 65535 - 300
    env2.reset(seed=12, options={"heater_code": codes})
    a = np.zeros((2, N), np.float32)
    a[0, 5] = 1.0
    got = []
    for _ in range(3):
        obs, r, term, trunc, info = env2.step(a)
        got.append((int(env2.obs_layout.to_codes(obs)["tx.heater.applied"][0, 5]), int(info["clipped"][0]),
                    int(info["clipped"][1])))
    assert got == [(65535 - 44, 0, 0), (65535, 1, 0), (65535, 1, 0)]
    a[0, 5] = -1.0
    obs, r, term, trunc, info = env2.step(a)
    assert int(env2.obs_layout.to_codes(obs)["tx.heater.applied"][0, 5]) == 65535 - 256 and info["clipped"][0] == 0


def test_discrete_actions():
    e = MRMEnv("g26", num_envs=1, action_type="discrete", max_delta_heater_codes=8, max_episode_steps=4)
    assert isinstance(e.single_action_space, gym.spaces.MultiDiscrete)
    assert np.all(e.single_action_space.nvec == 17)
    obs, _ = e.reset(seed=2)
    h0 = e.obs_layout.to_codes(obs)["tx.heater.applied"][0].copy()
    a = np.full((1, e.ctrl_spec.n_rings), 8, np.int64)
    a[0, 0], a[0, 7] = 16, 0
    obs, *_ = e.step(a)
    d = e.obs_layout.to_codes(obs)["tx.heater.applied"][0] - h0
    assert d[0] == 8 and d[7] == -8 and np.all(d[1:7] == 0)
    with pytest.raises(ValueError):
        e.step(np.full((1, e.ctrl_spec.n_rings), 17, np.int64))
    e.close()


def test_observation_is_the_released_sensor_channels(env2):
    """The observation is rebuilt here from the controller's own view (Plant.sample_sensors: the channels released
    at the edge) with an independent history: no Truth and no plant state reach it."""
    obs, _ = env2.reset(seed=5)
    lay, spec = env2.obs_layout, env2.ctrl_spec
    N, M, K = spec.n_rings, spec.samples_per_step, env2.history_blocks
    rng = np.random.default_rng(0)
    hist = None
    for step in range(K + 3):
        sensors, _ = env2.plant.sample_sensors()
        cnt = np.asarray(sensors["tx.monitor.count"], np.int64)
        lat = np.asarray(sensors["tx.monitor.latest"], np.int64)
        smp = np.asarray(sensors["tx.monitor.samples"], np.int64)
        blk = np.where(np.arange(M) < cnt[..., None], smp, lat[..., None])
        hist = np.repeat(blk[:, :, None], K, axis=2) if hist is None else np.concatenate([hist[:, :, 1:],
                                                                                           blk[:, :, None]], axis=2)
        exp = np.concatenate([hist.reshape(2, -1), np.asarray(sensors["tx.heater.applied"], np.int64),
                              np.asarray(sensors["tx.bias.applied"], np.int64),
                              np.asarray(sensors["temp.die0"], np.int64).reshape(2, 1)], axis=1)
        x = (exp - lay.code_offsets()) / 2.0 ** lay.shifts()
        assert np.array_equal(x.astype(np.float32), obs)
        assert env2.observation_space.contains(obs)
        obs, *_ = env2.step(rng.uniform(-0.1, 0.1, (2, N)).astype(np.float32))


def test_truth_only_in_info(env2):
    obs, info = env2.reset(seed=8)
    assert set(info) == {"tick", "seed", "truth"} and set(info["truth"]) == {"warm_start_no_code"}
    obs, r, term, trunc, info = env2.step(np.zeros((2, env2.ctrl_spec.n_rings), np.float32))
    assert set(info) == {"tick", "clipped", "controller_reset", "truth"}
    t = info["truth"]
    for k in ("detuning_m", "lock_reference_m", "lock_error_m", "reward_per_ring", "step_rms_lock_error_m"):
        assert t[k].shape == (2, env2.ctrl_spec.n_rings)
    assert np.allclose(t["lock_error_m"], t["detuning_m"] - t["lock_reference_m"], rtol=0, atol=0)
    assert np.allclose(r, t["reward_per_ring"].mean(axis=1), rtol=0, atol=0)


# ---------------------------------------------------------------------------------------------------- reward
@pytest.mark.parametrize("shape", ["log", "quadratic"])
def test_reward_equals_the_reward_of_the_truth_trace(shape):
    """The env reward (kernel accumulator differences) against the same formula on per-tick Truth records."""
    rs = RewardSpec(shape=shape, clip=3.0, lock_scale_pm=2.0, w_lock=1.0, w_oma=0.3, w_heater_per_w=50.0)
    rec = [RecordSpec("detuning_m", 1, "mean"), RecordSpec("reloma", 1, "mean"), RecordSpec("oracle_reloma", 1, "mean"),
           RecordSpec("p_heater_w", 1, "mean")]
    e = MRMEnv("g26", num_envs=2, max_episode_steps=12, reward=rs, record=rec,
               metrics={"lock_reference": "target", "lock_target_pm": 110.0})
    e.reset(seed=4)
    rng = np.random.default_rng(1)
    got = []
    for _ in range(12):
        obs, r, term, trunc, info = e.step(rng.uniform(-0.2, 0.2, (2, e.ctrl_spec.n_rings)).astype(np.float32))
        got.append(info["truth"]["reward_per_ring"])
    tr = e.plant.traces()
    P = e.frame_skip_ticks
    det = tr["detuning_m@1:mean"]["value"]                      # (T, B, N), tick 1 .. T
    assert det.shape[0] == 12 * P
    det = det.reshape(12, P, 2, -1)
    rel = tr["reloma@1:mean"]["value"].reshape(12, P, 2, -1)
    orc = tr["oracle_reloma@1:mean"]["value"].reshape(12, P, 2, -1)
    ph = tr["p_heater_w@1:mean"]["value"].reshape(12, P, 2, -1)
    u = ((det - 110e-12) ** 2).mean(axis=1) / (2e-12) ** 2
    lock = -np.log1p(u) if shape == "log" else -np.minimum(u, 3.0)
    oma = np.where(orc.sum(axis=1) > 0, rel.sum(axis=1) / orc.sum(axis=1), 0.0)
    exp = lock + 0.3 * oma - 50.0 * ph.mean(axis=1)
    np.testing.assert_allclose(np.asarray(got), exp, rtol=1e-9, atol=1e-12)
    e.close()


# ---------------------------------------------------------------------------------------------------- resets
def test_each_reset_draws_new_worlds_and_seeds_reproduce():
    e = MRMEnv("g26", num_envs=2, max_episode_steps=3)
    o1, i1 = e.reset(seed=21)
    s1, f1 = dict(e.seeds), np.asarray(e.plant.pp.optics.fwhm_hz).copy()
    o2, i2 = e.reset()
    s2, f2 = dict(e.seeds), np.asarray(e.plant.pp.optics.fwhm_hz).copy()
    assert s1 != s2 and abs(s2["device"] - s1["device"]) >= 2           # disjoint world seeds
    assert not np.array_equal(f1, f2) and not np.array_equal(f1[0], f1[1])   # new worlds; worlds differ in a batch
    assert not np.array_equal(o1, o2)
    o3, _ = e.reset(seed=21)
    assert e.seeds == s1 and np.array_equal(o3, o1)
    assert np.array_equal(np.asarray(e.plant.pp.optics.fwhm_hz), f1)
    o4, _ = e.reset()
    assert e.seeds == s2 and np.array_equal(o4, o2)
    o5, _ = e.reset(options={"seeds": {"device": 7, "uncertainty": 7, "noise": 7, "scenario": 7}})
    assert e.seeds["device"] == 7
    # nominal worlds: every world the same device, only the noise streams differ
    n = MRMEnv("g26", num_envs=2, uncertainty="nominal", variation="off", max_episode_steps=3)
    n.reset(seed=1)
    f = np.asarray(n.plant.pp.optics.fwhm_hz)
    assert np.array_equal(f[0], f[-1])
    e.close()
    n.close()


def test_determinism_and_next_step_autoreset():
    def roll(env, seed, n):
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(3)
        out = [obs]
        rews, truncs = [], []
        for _ in range(n):
            obs, r, term, trunc, info = env.step(rng.uniform(-0.3, 0.3, env.action_space.shape).astype(np.float32))
            out.append(obs)
            rews.append(r)
            truncs.append(trunc.copy())
            assert not term.any()
        return np.asarray(out), np.asarray(rews), np.asarray(truncs)
    e = MRMEnv("g26", num_envs=2, max_episode_steps=4)
    a = roll(e, 99, 6)
    b = roll(e, 99, 6)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    obs, rew, trunc = a
    assert trunc[3].all() and not trunc[:3].any() and not trunc[4:].any()
    assert np.all(rew[4] == 0.0)                        # the autoreset step: reward 0, not truncated
    # the autoreset observation is the next episode's first observation
    f = MRMEnv("g26", num_envs=2, max_episode_steps=4)
    f.reset(seed=99)
    o_next, _ = f.reset()
    assert np.array_equal(obs[5], o_next)
    e.close()
    f.close()


def test_an_episode_replays_from_its_seeds_and_a_world_alone():
    """Reproducibility (07 P1-13): an episode replays bit for bit from env.seeds (options={'seeds': ...}); world b of
    a batch replays alone at B = 1 with the seeds base + b from a cold start (plant draws and noise are per world),
    and its warm-start codes with their jitter are per world too. Since the engine review of 2026-10-16 a SETTLED
    start (warm, cold_settled) replays alone too: Plant.settle stops each world at its own iteration (here 9 / 15 /
    11 for warm, 4 / 6 / 6 for cold_settled). Before, the batch iterated until its slowest world converged, so a
    world's settled state depended on its batch companions (~1 mK; docs/handoff/rl.md section 6 item 10)."""
    B, n = 3, 8
    rng = np.random.default_rng(0)
    acts = rng.uniform(-0.2, 0.2, (n, B, 8)).astype(np.float32)

    def roll(env, a, options=None, seed=None):
        obs, _ = env.reset(seed=seed, options=options)
        out, rew = [obs], []
        for k in range(n):
            obs, r, *_ = env.step(a[k])
            out.append(obs)
            rew.append(r)
        return np.asarray(out), np.asarray(rew)
    e = MRMEnv("g26", num_envs=B, max_episode_steps=None, start_jitter_codes=40)
    one = MRMEnv("g26", num_envs=1, max_episode_steps=None, start_jitter_codes=40)
    o1, r1 = roll(e, acts, seed=5)
    seeds = dict(e.seeds)
    o2, r2 = roll(e, acts, options={"seeds": seeds})
    assert np.array_equal(o1, o2) and np.array_equal(r1, r2)
    s1 = {k: v + 1 for k, v in seeds.items()}
    obs1, _ = one.reset(options={"seeds": s1})
    h = e.obs_layout.channel("tx.heater.applied")
    assert np.array_equal(o1[0, 1, h.offset:h.offset + h.size], obs1[0, h.offset:h.offset + h.size])
    for start in ("cold", "warm", "cold_settled"):
        for env in (e, one):
            env.start = start
        o3, r3 = roll(e, acts, options={"seeds": seeds})
        its = None if start == "cold" else e.plant.settle_info["iterations_per_world"]
        for w in range(B):
            sw = {k: v + w for k, v in seeds.items()}
            o4, r4 = roll(one, acts[:, w:w + 1], options={"seeds": sw})
            assert np.array_equal(o3[:, w:w + 1], o4) and np.array_equal(r3[:, w:w + 1], r4), (start, w)
        if its is not None:
            assert len(set(its.tolist())) > 1, its          # the worlds needed different settle iterations
    e.close()
    one.close()


def test_start_jitter_is_per_world_and_clipped_to_the_dac_range():
    seeds = {"device": 0, "uncertainty": 0, "noise": 0, "scenario": 0}
    a = MRMEnv("g26", num_envs=4, max_episode_steps=None)
    b = MRMEnv("g26", num_envs=4, max_episode_steps=None, start_jitter_codes=500)
    oa, ia = a.reset(options={"seeds": seeds})
    ob, _ = b.reset(options={"seeds": seeds})
    ha = a.obs_layout.to_codes(oa)["tx.heater.applied"]
    hb = b.obs_layout.to_codes(ob)["tx.heater.applied"]
    d = hb - ha
    assert np.all(np.abs(d) <= 500) and np.count_nonzero(d) > 16 and np.all((hb >= 0) & (hb <= 65535))
    nc = ia["truth"]["warm_start_no_code"]                   # these start at code_min: the jitter is clipped there
    assert nc.any() and np.all(hb[nc] >= 0) and np.all(d[nc] >= 0)
    a.close()
    b.close()


# ---------------------------------------------------------------------------------------------------- gymnasium
def test_gymnasium_env_checker_passes():
    from gymnasium.utils.env_checker import check_env
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        env = gym.make(ENV_ID, uncertainty="sample", variation="sample")
        check_env(env.unwrapped)
        env.close()
    msgs = [str(x.message) for x in w if "WARN:" in str(x.message)]         # gymnasium.logger.warn
    assert not msgs, msgs
    single = MRMSingleEnv(max_episode_steps=3)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*alternative render modes.*")     # no spec without make()
        check_env(single)
    obs, info = single.reset(seed=0)
    for k in range(3):
        obs, r, term, trunc, info = single.step(single.action_space.sample())
        assert isinstance(r, float) and trunc == (k == 2)
    single.close()


def test_vector_api_and_gymnasium_wrappers():
    from gymnasium.envs.registration import EnvSpec
    from mrm.control.api import ControllerSpec
    venv = gym.make_vec(ENV_ID, num_envs=3, vectorization_mode="vector_entry_point", max_episode_steps=3)
    assert venv.num_envs == 3 and isinstance(venv.unwrapped, MRMEnv)
    # gymnasium owns VectorEnv.spec (make_vec sets the EnvSpec); the controller interface is ctrl_spec, and a reset
    # must not replace one with the other (wrappers and RL libraries read env.spec.id / max_episode_steps)
    assert isinstance(venv.unwrapped.spec, EnvSpec) and venv.unwrapped.spec.id == ENV_ID
    assert isinstance(venv.unwrapped.ctrl_spec, ControllerSpec)
    repr(venv)
    venv = gym.wrappers.vector.RecordEpisodeStatistics(venv)
    obs, info = venv.reset(seed=0)
    assert isinstance(venv.spec, EnvSpec) and venv.spec.id == ENV_ID and venv.spec.kwargs["num_envs"] == 3
    assert isinstance(venv.unwrapped.ctrl_spec, ControllerSpec)
    assert venv.observation_space.contains(obs)
    total = np.zeros(3)
    for k in range(3):
        obs, r, term, trunc, info = venv.step(venv.action_space.sample())
        total += r
    assert trunc.all() and "episode" in info
    np.testing.assert_allclose(info["episode"]["r"], total, rtol=1e-6)
    lst = gym.wrappers.vector.DictInfoToList(venv)
    obs, r, term, trunc, infos = lst.step(lst.action_space.sample())     # the autoreset step
    assert len(infos) == 3 and np.all(r == 0)
    venv.close()


# ---------------------------------------------------------------------------------------------------- engine parity
class _Schedule(BaseController):
    """A Python controller that applies the same code deltas as the env test below (held + delta, clipped)."""
    name = "sched"

    def __init__(self, deltas, latency_ticks=0):
        super().__init__()
        self.deltas = deltas
        self.latency_ticks = int(latency_ticks)

    def reset(self, spec):
        super().reset(spec)
        self.k, self.u = 0, None
        self.lo, self.hi = spec.heater_dac.code_min, spec.heater_dac.code_max

    def step(self, obs):
        if self.u is None:
            self.u = np.asarray(obs["tx.heater.applied"], np.int64).copy()
        self.u = np.clip(self.u + self.deltas[self.k], self.lo, self.hi)
        self.k += 1
        return {"tx.heater": self.u.copy()}


@pytest.mark.parametrize("latency", [0, 3])
def test_env_steps_equal_the_engine_python_controller_path(latency):
    """The env and mrm.sim.runner.run with a Python controller, same worlds, warm start, pipeline latency and command
    sequence: every metric and every Truth field bit-identical (the env drives the engine's own Python-controller
    path)."""
    B, n = 2, 30
    seeds = {"device": 40, "uncertainty": 40, "noise": 41, "scenario": 42}
    rng = np.random.default_rng(7 + latency)
    deltas = rng.integers(-40, 41, (n, B, 8)).astype(np.int64)
    e = MRMEnv("g26", num_envs=B, max_episode_steps=None, latency_ticks=latency)
    e.reset(options={"seeds": seeds})
    for k in range(n):
        e.step((deltas[k] / 256.0).astype(np.float32))
    r = runner.run("g26", _Schedule(deltas, latency), None, duration_s=n * e.step_s, batch=B, seeds=seeds,
                   uncertainty="sample", variation="sample", heater_code="quasi_static", settle=True,
                   metrics={"lock_reference": "oracle"}, histogram=False)
    assert r.manifest_inputs["n_ticks"] == e.plant.tick
    m_env = e.plant.metrics()
    for k in ("rms_lock_error_pm", "max_abs_lock_error_pm", "mean_reloma", "heater_energy_j", "excursion_count",
              "code_min_used", "code_max_used", "peak_slew_codes_per_s", "lock_reference_pm"):
        assert np.array_equal(np.asarray(m_env[k]), np.asarray(r.metrics[k]), equal_nan=True), k
    t_env = e.plant.truth()
    for f, v in r.truth_final.fields.items():
        assert np.array_equal(t_env.fields[f], v, equal_nan=True), f
    e.close()


def test_starts_and_randomized_scenarios():
    """'cold' (the config's reset protocol, heaters at code_min), 'cold_settled', and a scenario callable that draws
    the A1 aggressor phase per reset from the world seed."""
    from mrm.scenarios import library as lib
    phases = []

    def scen(rng):
        ph = float(rng.uniform(0.0, lib.A1_PERIOD_S))
        phases.append(ph)
        return lib.a1_slide10_aggressor(phase_s=ph)
    e = MRMEnv("g26", num_envs=2, start="cold", scenario=scen, max_episode_steps=3)
    obs, _ = e.reset(seed=1)
    assert np.all(e.obs_layout.to_codes(obs)["tx.heater.applied"] == 0)
    assert e.plant.settle_info is None
    e.step(np.zeros((2, e.ctrl_spec.n_rings), np.float32))
    obs, _ = e.reset(options={"start": "cold_settled"})
    assert e.plant.settle_info is not None and np.all(e.obs_layout.to_codes(obs)["tx.heater.applied"] == 0)
    assert len(phases) == 3 and len(set(phases)) == 3        # construction + two resets, each its own phase
    assert e.plant.aggressor_calibration                      # the A1 power was calibrated for this plant
    e.close()


def test_reward_can_leave_out_rings_without_a_warm_start_code():
    """Seeds whose sampled worlds have rings the heater cannot place at their lock (warm_start_no_code): with
    exclude_unreachable the env reward is the mean over the other rings; the per-ring rewards are unchanged."""
    seeds = {"device": 0, "uncertainty": 0, "noise": 0, "scenario": 0}
    a = MRMEnv("g26", num_envs=4, max_episode_steps=None)
    b = MRMEnv("g26", num_envs=4, max_episode_steps=None, reward={"exclude_unreachable": True})
    _, ia = a.reset(options={"seeds": seeds})
    _, ib = b.reset(options={"seeds": seeds})
    nc = ia["truth"]["warm_start_no_code"]
    assert np.array_equal(nc, ib["truth"]["warm_start_no_code"])
    act = np.zeros((4, a.ctrl_spec.n_rings), np.float32)
    _, ra, *_, info_a = a.step(act)
    _, rb, *_, info_b = b.step(act)
    rr = info_a["truth"]["reward_per_ring"]
    assert np.array_equal(rr, info_b["truth"]["reward_per_ring"])
    np.testing.assert_allclose(ra, rr.mean(axis=1), rtol=1e-14)
    exp = np.array([rr[i][~nc[i]].mean() if (~nc[i]).any() else 0.0 for i in range(4)])
    np.testing.assert_allclose(rb, exp, rtol=1e-14)
    if not nc.any():
        pytest.skip("no unreachable ring in these worlds (the exclusion is then the plain mean)")
    a.close()
    b.close()
