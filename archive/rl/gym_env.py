# expose the simulated ring bank as a batched Gymnasium learning environment.
# `MRMEnv` holds B worlds in one plant.
# kernel threads run them in parallel without subprocess workers.
# one environment step is one controller period.
# the kernel pauses after releasing sensors at the next controller edge.
# the environment reads those channels, writes code changes, and resumes.
# observations contain monitor history, applied DAC codes, and temperature codes.
# normalization is an exact power-of-two scale.
# actions change heater codes and, when bound to the controller, bias codes.
# commands still pass through controller latency, DAC latency, slew limits, and settling.
# reward comes from differences in the kernel's streaming metric accumulators.
# Truth is available through info for training and inspection, never in the observation.
# each reset draws new model worlds and fabrication variation.
# all worlds truncate together.
# NEXT_STEP autoreset resets on the following call.
# `MRMSingleEnv` removes the batch axis for Gymnasium's single-env interface.
# both are registered as `mrm/MRM-v0`.
# `compiled_policy` deploys a quantized MLP with the same observation and action rules inside the kernel.
# `fxp_policy_action` evaluates the corresponding environment action.
# the environment guide is in `archive/rl/docs/ENGINE_LEARNING.md`.
# measured throughput and remaining performance limits are in `archive/rl/docs/rl.md`.
# vector batching does not remove the cost of each world's plant ticks.

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numba as nb
import numpy as np
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from mrm.control import api
from mrm.control.api import CompiledController, ControllerSpec
from mrm.control.fxp import Overflow, Rounding, fx_round_shift
from archive.rl.fxp_mlp import (H_MAX_WIDTH, H_N_IN, H_N_LAYERS, H_N_OUT, H_SIZE, L_OUT_FRAC, L_SIZE, FxpMlp,
                                 fxp_mlp_check, fxp_mlp_eval)
from mrm.core.channels import TEMP_ID_BASE, ActId, ObsId
from mrm.plant.io import AM_DST, AM_OFFSET, AM_SIZE, DST_BIAS, DST_HEATER
from mrm.plant.plant import Plant
from mrm.sim import kernel
from mrm.sim.metrics import (AF_HEATER_ENERGY, AF_SUM_ERR2, AF_SUM_ORACLE, AF_SUM_RELOMA, AI_TICKS, AI_WRONG_TICKS, TR,
                             TRUTH_FIELDS, MetricsSpec)

__all__ = ["MRMEnv", "MRMSingleEnv", "RewardSpec", "ObsLayout", "ChannelCodec", "ActionCodec", "code_codec",
           "DEFAULT_TRUTH_FIELDS", "EPISODE_METRICS", "STARTS", "fxp_policy_action", "mlp_policy_init",
           "mlp_policy_step", "ENV_ID"]

ENV_ID = "mrm/MRM-v0"                           # the gymnasium id of both envs
STARTS = ("warm", "cold", "cold_settled")       # start states of a reset (module comments, 'Start')
# Truth fields copied into info['truth'] every step (mrm.sim.metrics.TRUTH_FIELDS names - `truth_fields='all'` copies every one)
DEFAULT_TRUTH_FIELDS = ("detuning_m", "reloma", "oracle_reloma", "heater_code", "line_on", "t_ring_k")
# episode metrics (mrm.sim.metrics.summarize keys, (B, N)) in info['truth']['episode'] at the truncation step
EPISODE_METRICS = ("rms_lock_error_pm", "max_abs_lock_error_pm", "locked_final", "excursion_count", "time_unlocked_s",
                   "wrong_line_s", "mean_reloma_db", "oma_penalty_db", "heater_energy_j", "heater_sat_s",
                   "heater_low_rail_s")
_SEED_KEYS = ("device", "uncertainty", "noise", "scenario")     # the random streams of a world
# fixed second keys of two per-reset generators, so they differ from every other stream of the same seed
_SCENARIO_RNG_TAG = 0x6D726D    # ASCII 'mrm' - the generator handed to a scenario callable
_JITTER_RNG_TAG = 0x6A6974      # ASCII 'jit' - the start-code jitter of one world


# codecs
# return the center and binary shift that normalize codes into [-1, 1).
# choose the smallest power-of-two span covering the inclusive code range.
# `x = (code - center) * 2**(-shift)` is then exact for the converter widths used here.
# this also permits integer-shift normalization in a fixed-point policy.
def code_codec(lo: int, hi: int) -> tuple[int, int]:
    span = int(hi) - int(lo) + 1
    if span < 1:
        raise ValueError(f"empty code range [{lo}, {hi}]")
    span_bits = max(1, int(math.ceil(math.log2(span))))    # 2^p - the smallest power of two >= span (at least 2)
    while (1 << span_bits) < span:                         # guard the float log2
        span_bits += 1
    shift = span_bits - 1
    return int(lo) + (1 << shift), shift


# one sensor channel's location and normalization in the observation vector.
# offset and size select its flat slice.
# shape restores its element axes.
# code_offset and shift define `(code - code_offset) * 2**(-shift)`.
# lo and hi bound that normalized channel.
@dataclass(frozen=True)
class ChannelCodec:
    name: str
    offset: int
    size: int
    shape: tuple
    code_offset: int
    shift: int
    lo: float
    hi: float


# ordered sensor slices in one flattened observation.
# monitor history comes first as `(N, K, M)` for rings, history blocks from oldest to newest, and samples.
# applied heater codes follow, then applied bias codes when exposed, then temperature codes in interface order.
# each world has the same layout.
@dataclass(frozen=True)
class ObsLayout:
    n_rings: int
    history_blocks: int
    samples_per_block: int
    channels: tuple
    size: int

    # the codec of channel `name`.
    def channel(self, name: str) -> ChannelCodec:
        for codec in self.channels:
            if codec.name == name:
                return codec
        raise KeyError(name)

    # {channel — its slice of the flat observation}.
    def slices(self) -> dict[str, slice]:
        return {c.name: slice(c.offset, c.offset + c.size) for c in self.channels}

    # (lo, hi), float32 (size,) — the bounds of every normalized element (the observation space).
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.empty(self.size, np.float32)
        hi = np.empty(self.size, np.float32)
        for codec in self.channels:
            lo[codec.offset:codec.offset + codec.size] = codec.lo
            hi[codec.offset:codec.offset + codec.size] = codec.hi
        return lo, hi

    # (size,) int64 — c0 of every element.
    def code_offsets(self) -> np.ndarray:
        out = np.empty(self.size, np.int64)
        for codec in self.channels:
            out[codec.offset:codec.offset + codec.size] = codec.code_offset
        return out

    # (size,) int64 — s of every element.
    def shifts(self) -> np.ndarray:
        out = np.empty(self.size, np.int64)
        for codec in self.channels:
            out[codec.offset:codec.offset + codec.size] = codec.shift
        return out

    # {channel — normalized values (..., *shape)} of observations (..., size).
    def unflatten(self, obs) -> dict[str, np.ndarray]:
        values = np.asarray(obs)
        leading_shape = values.shape[:-1]
        return {codec.name: values[..., codec.offset:codec.offset + codec.size].reshape(leading_shape + codec.shape) for codec in self.channels}

    # {channel — the codes (int64, (..., *shape))} back from normalized observations (exact inverse).
    def to_codes(self, obs) -> dict[str, np.ndarray]:
        values = np.asarray(obs, np.float64)
        leading_shape = values.shape[:-1]
        out = {}
        for codec in self.channels:
            raw_codes = values[..., codec.offset:codec.offset + codec.size] * float(1 << codec.shift) + codec.code_offset
            out[codec.name] = np.rint(raw_codes).astype(np.int64).reshape(leading_shape + codec.shape)
        return out


# decode the flat action into code changes for controller-owned actuators.
# each channel tuple records its name, action offset, size, maximum delta, code bounds, and offset in the kernel command vector.
# size is total action length.
# discrete selects integer actions instead of normalized floats.
@dataclass(frozen=True)
class ActionCodec:
    channels: tuple
    size: int
    discrete: bool

    # the actuator channel names, in action order.
    def names(self) -> tuple:
        return tuple(name for name, *_ in self.channels)


# weights and shape for reward computed over one controller period.
# per ring, let u be mean squared lock error divided by lock_scale_pm squared.
# the default scale is the metric lock threshold.
# log shape gives `-log1p(u)`.
# quadratic shape gives `-min(u, clip)`.
# at RMS error equal to the scale, the log penalty is about -0.69.
# measured in multiples of the error scale s, it is -10.6 at 200 s and -13.8 at 1000 s.
# s denotes lock_scale_pm, not seconds.
# slow growth limits the impact of a lost ring.
# optional terms add relOMA divided by oracle relOMA, subtract mean heater power in W, and subtract the fraction of time on a wrong line.
# combine these with their weights, then average over rings.
# `exclude_unreachable` leaves rings without warm-start lock codes out of that average.
# their individual reward is still reported.
# a world with no included ring gets zero reward.
@dataclass(frozen=True)
class RewardSpec:
    lock_scale_pm: float | None = None
    shape: str = "log"
    clip: float = 100.0
    w_lock: float = 1.0
    w_oma: float = 0.0
    w_heater_per_w: float = 0.0
    w_wrong_line: float = 0.0
    exclude_unreachable: bool = False

    def __post_init__(self):
        if self.shape not in ("log", "quadratic"):
            raise ValueError("RewardSpec.shape must be 'log' or 'quadratic'")
        if self.lock_scale_pm is not None and not self.lock_scale_pm > 0:
            raise ValueError("RewardSpec.lock_scale_pm must be > 0")
        if not self.clip > 0:
            raise ValueError("RewardSpec.clip must be > 0")


# the Config of a preset name, a YAML path or an already loaded Config.
def _load_config(config, overrides, alts):
    from mrm import params
    if isinstance(config, (str, Path)):
        return params.load(config, overrides=overrides, alts=alts)
    if overrides or alts:
        raise ValueError("overrides / alts apply to a preset name or YAML path, not to a loaded Config")
    if not callable(getattr(config, "si", None)):
        raise TypeError("config must be a preset name, a YAML path or an mrm.params Config")
    return config


# {stream — seed} for every stream of _SEED_KEYS, from one int (every stream) or a partial dict (others 0).
def _normalize_seeds(seeds) -> dict:
    if isinstance(seeds, (int, np.integer)):
        return {k: int(seeds) for k in _SEED_KEYS}
    given = dict(seeds)
    bad = set(given) - set(_SEED_KEYS)
    if bad:
        raise ValueError(f"unknown seed streams {sorted(bad)}; use {_SEED_KEYS}")
    out = {k: 0 for k in _SEED_KEYS}
    out.update({k: int(v) for k, v in given.items()})
    return out


# the env
# B worlds sharing one plant as a Gymnasium vector environment.
# `config` accepts a preset name, YAML path, or loaded Config.
# overrides and alternatives apply while loading.
# `num_envs` is B.
# `scenario` can be the config selection, a name, a Scenario, or a seeded callable evaluated per reset.
# the observation and action spaces must stay fixed across those resets.
# `history_blocks` sets monitor history length.
# action limits bound heater and bias code changes per step.
# continuous actions lie in [-1, 1].
# discrete actions encode the same integer delta range.
# at the nominal g26 lock, 256 heater codes represent about 13 pm over a 10 us step.
# this is an action limit, not a target.
# `start` selects warm lock codes, the config's cold reset, or cold-settled minimum heater codes.
# `start_jitter_codes` adds a bounded integer offset to warm codes using each world's device seed.
# uncertainty and variation modes control the new worlds drawn at reset.
# `reward` sets RewardSpec.
# `metrics` overrides metric settings, with oracle lock reference by default when oracle evaluation is enabled.
# `truth_fields` selects values copied into info.
# latency_ticks adds policy pipeline delay.
# n_threads, substep_policy, and sh_tol_k configure plant execution.
# optional records allocate one episode of trace storage, or 1000 steps if unbounded.
# all worlds truncate after max_episode_steps.
# None disables truncation.
# a lost lock does not terminate a world because it can be reacquired.
# autoreset creates a new batch on the next step and returns zero reward.
# policies with their own memory must handle the controller-reset mask returned in info.
class MRMEnv(VectorEnv):
    metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP, "render_modes": []}

    def __init__(self, config: Any = "g26", num_envs: int = 8, *, overrides=None, alts=None, scenario=None,
                 max_episode_steps: int | None = 1000, history_blocks: int = 4, max_delta_heater_codes: int = 256,
                 max_delta_bias_codes: int = 64, action_type: str = "continuous", start: str = "warm",
                 start_jitter_codes: int = 0, uncertainty: str = "sample", variation: str = "sample",
                 reward: RewardSpec | Mapping | None = None, metrics: Mapping | None = None,
                 truth_fields: Sequence[str] | str = DEFAULT_TRUTH_FIELDS, latency_ticks: int = 0,
                 n_threads: int | None = None, substep_policy: str | None = None, oracle: bool = True,
                 record=None, sh_tol_k: float | None = None):
        self.cfg = _load_config(config, overrides, alts)
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        B = self.num_envs
        if start not in STARTS:
            raise ValueError(f"start must be one of {STARTS}")
        if action_type not in ("continuous", "discrete"):
            raise ValueError("action_type must be 'continuous' or 'discrete'")
        if uncertainty not in ("sample", "lhs", "nominal") or variation not in ("sample", "off"):
            raise ValueError("uncertainty must be sample | lhs | nominal and variation sample | off")
        if max_episode_steps is not None and int(max_episode_steps) < 1:
            raise ValueError("max_episode_steps must be >= 1 or None")
        self.max_episode_steps = None if max_episode_steps is None else int(max_episode_steps)
        self.history_blocks = int(history_blocks)
        if self.history_blocks < 1:
            raise ValueError("history_blocks must be >= 1")
        self.start = start
        self.start_jitter_codes = int(start_jitter_codes)
        if self.start_jitter_codes < 0:
            raise ValueError("start_jitter_codes must be >= 0")
        self.uncertainty, self.variation = uncertainty, variation
        self.reward_spec = reward if isinstance(reward, RewardSpec) else RewardSpec(**dict(reward or {}))
        self.latency_ticks = int(latency_ticks)
        self.n_threads = n_threads
        self.substep_policy = substep_policy
        self.oracle = bool(oracle)
        self.record = record
        self.sh_tol_k = sh_tol_k
        self._scenario_arg = scenario
        metrics_overrides = {"lock_reference": "oracle"} if self.oracle else {}
        metrics_overrides.update(dict(metrics or {}))
        self.metrics_spec = MetricsSpec.from_config(self.cfg, **metrics_overrides)
        if self.reward_spec.w_oma and not self.oracle:
            raise ValueError("the oma reward term needs the oracle (oracle=True)")
        if truth_fields == "all":
            truth_fields = TRUTH_FIELDS
        elif isinstance(truth_fields, str):              # one field name
            truth_fields = (truth_fields,)
        bad = [f for f in truth_fields if f not in TR]
        if bad:
            raise ValueError(f"unknown Truth fields {bad}; known: {TRUTH_FIELDS}")
        self.truth_fields = tuple(truth_fields)
        # the interface (spaces) from the ChannelSpecs of the config and the first scenario
        first_scenario = self._make_scenario(np.random.default_rng(0))
        self.ctrl_spec = ControllerSpec.from_config(self.cfg, n_batch=B, nominal=self.cfg,
                                                    scenario_drives=first_scenario.drives,
                                                    scenario_offsets=first_scenario.offsets)
        if self.ctrl_spec.tick_jitter_ticks > 0:
            raise NotImplementedError("MRMEnv steps on the engine's Python-controller path, which needs controller "
                                      "edges shared by every world (controller_interface.tick_jitter_s = 0)")
        self._layout_fp = self.ctrl_spec.layout.fingerprint()
        self.frame_skip_ticks = int(self.ctrl_spec.period_ticks)
        self.dt_s = float(self.ctrl_spec.tick_s)
        self.step_s = self.frame_skip_ticks * self.dt_s
        self.obs_layout = self._build_obs_layout(self.ctrl_spec)
        self.action_codec = self._build_action_codec(self.ctrl_spec, int(max_delta_heater_codes),
                                                     int(max_delta_bias_codes), action_type == "discrete")
        lo, hi = self.obs_layout.bounds()
        self.single_observation_space = spaces.Box(lo, hi, (self.obs_layout.size,), np.float32)
        A = self.action_codec.size
        if action_type == "discrete":
            nvec = np.concatenate([np.full(size, 2 * max_delta + 1, np.int64)
                                   for _name, _offset, size, max_delta, *_ in self.action_codec.channels])
            self.single_action_space = spaces.MultiDiscrete(nvec)
        else:
            self.single_action_space = spaces.Box(-1.0, 1.0, (A,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, B)
        self.action_space = batch_space(self.single_action_space, B)
        # allocate disjoint blocks of B seeds across resets.
        # keep base + world index below 2^30.
        self._n_blocks = max(1, (2 ** 31) // B // 2)
        self._block0: int | None = None         # b0 - drawn from np_random at the first reset and at every seeded reset
        self._episode = 0                       # resets (explicit and automatic) since b0 was drawn
        self._needs_reset = False               # True after a truncation - the next step autoresets
        self.plant: Plant | None = None
        self.seeds: dict | None = None          # the base seeds of the current worlds
        self._ep_step = 0                       # steps since the last reset

    # interface
    # validate sensor availability and define the flattened observation layout.
    # require monitor samples, valid counts, held latest code, and applied heater codes.
    # include applied bias and temperature codes when exposed.
    # each sensor codec uses its declared code range for binary normalization.
    def _build_obs_layout(self, spec: ControllerSpec) -> ObsLayout:
        N, K, M = spec.n_rings, self.history_blocks, spec.samples_per_step    # rings, history blocks, samples/block
        names = [f.name for f in spec.layout.obs]
        rows = [("tx.monitor.samples", (N, K, M))]
        for need in ("tx.monitor.samples", "tx.monitor.count", "tx.monitor.latest", "tx.heater.applied"):
            if need not in names:
                raise ValueError(f"MRMEnv needs the sensor channel {need!r} in the controller interface")
        # the ABI packs the monitor block ring-major, M samples per ring (_observe and mlp_policy_step read it so)
        for name, size in (("tx.monitor.samples", N * M), ("tx.monitor.count", N), ("tx.monitor.latest", N),
                           ("tx.heater.applied", N), ("tx.bias.applied", N)):
            if name in names and spec.layout.obs_field(name).size != size:
                raise ValueError(f"MRMEnv: ABI field {name} has {spec.layout.obs_field(name).size} elements, "
                                 f"expected {size}")
        rows.append(("tx.heater.applied", (N,)))
        if "tx.bias.applied" in names:
            rows.append(("tx.bias.applied", (N,)))
        for t in spec.temp_sensors:
            if t.channel in names:
                rows.append((t.channel, (1,)))
        codecs, offset = [], 0
        for name, shape in rows:
            channel = spec.channels[name]
            c0, s = code_codec(int(channel.min), int(channel.max))
            size = int(np.prod(shape))
            codecs.append(ChannelCodec(name=name, offset=offset, size=size, shape=tuple(shape), code_offset=c0,
                                       shift=s, lo=(int(channel.min) - c0) / float(1 << s),
                                       hi=(int(channel.max) - c0) / float(1 << s)))
            offset += size
        return ObsLayout(n_rings=N, history_blocks=K, samples_per_block=M, channels=tuple(codecs), size=offset)

    # build delta-action slices for the heater and any controller-owned bias channel.
    @staticmethod
    def _build_action_codec(spec: ControllerSpec, max_delta_heater: int, max_delta_bias: int,
                            discrete: bool) -> ActionCodec:
        actuator_fields = {field_info.name: field_info for field_info in spec.layout.act}
        if "tx.heater" not in actuator_fields:
            raise ValueError("MRMEnv needs tx.heater bound to the controller")
        rows, offset = [], 0
        for name, max_delta in (("tx.heater", max_delta_heater), ("tx.bias", max_delta_bias)):
            if name not in actuator_fields:
                continue
            channel = spec.channels[name]
            span = int(channel.max) - int(channel.min)
            if not 1 <= max_delta <= span:
                raise ValueError(f"max delta of {name} must be in 1..{span} codes, got {max_delta}")
            field_info = actuator_fields[name]
            rows.append((name, offset, int(field_info.size), int(max_delta), int(channel.min), int(channel.max),
                int(field_info.offset)))
            offset += int(field_info.size)
        return ActionCodec(channels=tuple(rows), size=offset, discrete=discrete)

    # the scenario of a reset — the config's, a registry name, a Scenario, or `scenario(rng)`.
    def _make_scenario(self, rng: np.random.Generator):
        from mrm.scenarios import registry as scenario_registry
        from mrm.scenarios.generators import Scenario
        scenario_arg = self._scenario_arg
        if scenario_arg is None:
            return scenario_registry.from_config(self.cfg)
        if isinstance(scenario_arg, str):
            return scenario_registry.create(scenario_arg)
        if isinstance(scenario_arg, Scenario):
            return scenario_arg
        if callable(scenario_arg):
            out = scenario_arg(rng)
            if not isinstance(out, Scenario):
                raise TypeError("a scenario callable must return a mrm.scenarios.generators.Scenario")
            return out
        raise TypeError("scenario must be None, a registry name, a Scenario or a callable(rng) -> Scenario")

    # reset
    # draw new worlds, initialize the plant, and return the first edge observation.
    # a seeded reset draws a new starting seed block.
    # later resets use consecutive blocks of B world seeds, wrapping only after the available blocks are used.
    # world b can replay alone with its own stream seeds, except for Latin-hypercube uncertainty.
    # a scenario callable gets one generator per reset.
    # options can pin the stream seeds for this reset, change its start mode, or supply explicit `(B, N)` heater codes.
    # explicit codes use a hot-branch settle.
    # warm starts without a reachable code use code_min and report a mask in info.
    def reset(self, *, seed: int | None = None, options: Mapping | None = None):
        super().reset(seed=seed)
        if seed is not None or self._block0 is None:
            self._block0 = int(self.np_random.integers(0, self._n_blocks))
            self._episode = 0
        else:
            self._episode += 1
        return self._reset_worlds(dict(options or {}))

    # build fresh worlds and initialize the chosen start, sensor history, and counters.
    # warm starts use surveyed codes and hot-branch settling.
    # cold uses the config reset protocol.
    # cold_settled uses minimum codes on the cold branch.
    # fill each world's history with K copies of its first released monitor block.
    # before any plant interval runs, info contains only the warm-start report rather than Truth.
    def _reset_worlds(self, opts: dict):
        B = self.num_envs
        if opts.get("seeds") is not None:
            seeds = _normalize_seeds(opts["seeds"])
        else:
            base = ((self._block0 + self._episode) % self._n_blocks) * B
            seeds = {k: base for k in _SEED_KEYS}
        self.seeds = seeds
        rng = np.random.default_rng([seeds["scenario"], _SCENARIO_RNG_TAG])
        scenario = self._make_scenario(rng)
        self._build_plant(seeds, scenario)
        start = opts.get("start", self.start)
        if start not in STARTS:
            raise ValueError(f"start must be one of {STARTS}")
        plant = self.plant
        no_code = np.zeros((B, self.ctrl_spec.n_rings), bool)
        heater_code = opts.get("heater_code")
        if heater_code is not None:
            plant.reset(protocol="branch_high", heater_code=np.asarray(heater_code, np.int64), settle=True)
        elif start == "warm":
            from mrm.sim import runner
            with warnings.catch_warnings():         # the unreachable rings are reported in info['truth']
                warnings.filterwarnings("ignore", message="warm_start_codes", category=UserWarning)
                codes, no_code = runner.warm_start_codes(plant, self.cfg, mode="locked_steady_state",
                                                         return_no_code=True)
            if self.start_jitter_codes:
                # one generator per world, seeded by that world's device seed (base + b), so a world replays alone
                # at B = 1 with its own seeds - clipped to the DAC range (as Plant.reset would)
                jitter_limit = self.start_jitter_codes
                jitter = np.stack([np.random.default_rng([seeds["device"] + b, _JITTER_RNG_TAG]).integers(
                    -jitter_limit, jitter_limit + 1, codes.shape[1]) for b in range(B)])
                heater_params = plant.heater_params
                codes = np.clip(codes + jitter, heater_params.code_lo, heater_params.code_hi)
            plant.reset(protocol="branch_high", heater_code=codes, settle=True)
        elif start == "cold":
            plant.reset()
        else:
            plant.reset(protocol="branch_low", settle=True,
                        heater_code=np.full((B, self.ctrl_spec.n_rings), plant.heater_params.code_lo, np.int64))
        plant.n_threads = self.n_threads
        self._warm_no_code = np.asarray(no_code, bool)
        self._ep_step = 0
        self._needs_reset = False
        self._advance(resume=False)
        self._controller_reset()                    # a reset raised before the first edge - the state is fresh anyway
        self._hist = None                           # the K-block monitor history, (B, N, K, M) - _observe fills it
        obs = self._observe()
        self._snap = self._accumulator_snapshot()   # the accumulators at the last edge (the reward differences them)
        self._sat0 = plant.S.sat.sum(axis=1).copy()  # clipped-element counts at the last edge, (B,)
        # no interval has run at this first edge.
        # report the start instead of an unwritten Truth row.
        info = {"tick": np.full(B, plant.tick, np.int64),
                "seed": np.full(B, seeds["device"], np.int64) + np.arange(B, dtype=np.int64),
                "truth": {"warm_start_no_code": self._warm_no_code.copy()}}
        return obs, info

    # realize B worlds and devices with `seeds` and build the Plant of this reset (`self.plant`).
    def _build_plant(self, seeds: dict, scenario) -> None:
        from mrm import params
        B = self.num_envs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            plant_params = params.realize(self.cfg, B, seeds={"device": seeds["device"],
                                                              "uncertainty": seeds["uncertainty"]},
                                          uncertainty=self.uncertainty, variation=self.variation)
        spec = ControllerSpec.from_config(self.cfg, n_batch=B, nominal=self.cfg, scenario_drives=scenario.drives,
                                          scenario_offsets=scenario.offsets)
        if spec.layout.fingerprint() != self._layout_fp:
            raise ValueError("the scenario of this reset changes the controller interface (channel bindings); the "
                             "spaces of an env are fixed at construction")
        self.ctrl_spec = spec
        plant_options = {}
        if self.sh_tol_k is not None:
            plant_options["sh_tol_k"] = float(self.sh_tol_k)
        record_ticks = (self.max_episode_steps or 1000) * self.frame_skip_ticks + self.frame_skip_ticks
        self.plant = Plant(plant_params, scenario, spec, controller_latency_ticks=self.latency_ticks,
                           noise_seed=seeds["noise"], scenario_seed=seeds["scenario"], metrics=self.metrics_spec,
                           record=self.record, record_ticks=record_ticks if self.record else 0,
                           substep_policy=self.substep_policy, oracle=self.oracle, histogram=False, **plant_options)

    # step
    # apply a delta command at this edge and advance one controller period.
    # return observation `(B, D)`, reward `(B,)`, termination and truncation masks, and info.
    # reward is float64 and observations are float32.
    # termination stays false.
    # all worlds share the episode step limit.
    # the next call after truncation resets the batch and returns zero reward without applying that call's action.
    def step(self, actions):
        B = self.num_envs
        if self.plant is None:
            raise RuntimeError("call reset() first")
        if self._needs_reset:                       # reset all worlds on the call after truncation
            self._episode += 1
            obs, info = self._reset_worlds({})
            return obs, np.zeros(B), np.zeros(B, bool), np.zeros(B, bool), info
        delta = self._deltas(actions)
        state = self.plant.S
        # add deltas to held commands.
        # the kernel applies the hardware-range clip.
        for (_name, action_offset, size, _max_delta, _lo, _hi, command_offset) in self.action_codec.channels:
            state.act[:, command_offset:command_offset + size] += delta[:, action_offset:action_offset + size]
        self._advance(resume=True)
        self._ep_step += 1
        controller_reset = self._controller_reset()
        obs = self._observe(restart=controller_reset)
        reward, reward_per_ring, rms = self._reward()
        truncated = self.max_episode_steps is not None and self._ep_step >= self.max_episode_steps
        state = self.plant.S
        saturation_count = state.sat.sum(axis=1)
        info = {"tick": np.full(B, self.plant.tick, np.int64),
            "clipped": (saturation_count - self._sat0).astype(np.int64),
                "controller_reset": controller_reset.copy(), "truth": self._truth_info(reward_per_ring, rms)}
        self._sat0 = saturation_count.copy()
        if truncated:
            episode_metrics = self.plant.metrics()
            info["truth"]["episode"] = {k: np.asarray(episode_metrics[k]).copy() for k in EPISODE_METRICS}
            self._needs_reset = True
        return obs, reward, np.zeros(B, bool), np.full(B, bool(truncated)), info

    # convert actions to bounded integer code changes shaped `(B, A)`.
    # continuous values are clipped to [-1, 1], multiplied by each channel's maximum delta, and rounded to nearest even.
    # discrete values span zero to twice the maximum delta, with the midpoint representing no change.
    def _deltas(self, actions) -> np.ndarray:
        B, A = self.num_envs, self.action_codec.size
        action_values = np.asarray(actions)
        if action_values.shape != (B, A):
            raise ValueError(f"actions must have shape {(B, A)}, got {action_values.shape}")
        max_delta = np.concatenate([np.full(size, channel_max_delta, np.int64)
                                    for _name, _offset, size, channel_max_delta, *_ in self.action_codec.channels])
        if self.action_codec.discrete:
            if not np.issubdtype(action_values.dtype, np.integer):
                raise TypeError("discrete actions must be integers")
            integer_actions = action_values.astype(np.int64)
            if np.any(integer_actions < 0) or np.any(integer_actions > 2 * max_delta):
                raise ValueError("discrete action outside 0..2 max_delta")
            return integer_actions - max_delta
        float_actions = action_values.astype(np.float64)
        if not np.all(np.isfinite(float_actions)):
            raise ValueError("actions must be finite")
        return np.rint(np.clip(float_actions, -1.0, 1.0) * max_delta).astype(np.int64)

    # run the kernel on the Python path to the next controller edge (the sensors there are released).
    def _advance(self, resume: bool) -> None:
        plant = self.plant
        tick_budget = 2 * self.frame_skip_ticks + 1           # tick budget per call - the kernel stops earlier, at the edge
        while True:
            plant.run(tick_budget, mode=kernel.MODE_PYTHON, resume=resume, n_threads=self.n_threads)
            resume = False
            if int(plant.S.world_int[0, kernel.WI_PENDING_EDGE]) == plant.tick:
                return

    # restart delta-policy state for worlds with a controller-reset fault.
    # restore held heater and bias commands from their latched registers.
    # clear the fault flag and return a boolean world mask.
    # observation history restarts for those worlds.
    # the caller must reset any separate policy memory.
    def _controller_reset(self) -> np.ndarray:
        state, kernel_params = self.plant.S, self.plant.P
        ctrl_reset = state.world_int[:, kernel.WI_CTRL_RESET] != 0
        if np.any(ctrl_reset):
            for i in range(kernel_params.act_map.shape[0]):
                row = kernel_params.act_map[i]
                destination, offset, size = int(row[AM_DST]), int(row[AM_OFFSET]), int(row[AM_SIZE])
                if destination in (DST_HEATER, DST_BIAS):                           # as kernel._sync_act_a
                    register_row = kernel.RI_HE_REG if destination == DST_HEATER else kernel.RI_BI_REG
                    state.act[ctrl_reset, offset:offset + size] = state.ring_int[ctrl_reset, register_row, :size]
            state.world_int[ctrl_reset, kernel.WI_CTRL_RESET] = 0
        return ctrl_reset

    # observation
    # build normalized observations from sensor channels released at this edge.
    # fill unused monitor sample slots with the held latest code.
    # append the block to each ring's K-block history.
    # on first use or a reset mask, repeat this block across the history.
    # a first edge with no samples uses code zero, normalized to -1 for the monitor range.
    # convert codes using their binary scales exactly.
    def _observe(self, restart: np.ndarray | None = None) -> np.ndarray:
        abi = self.ctrl_spec.layout
        released_obs = self.plant.S.obs                  # (B, obs_size) int64 - the flat ABI observation of the edge
        B, N = self.num_envs, self.ctrl_spec.n_rings
        M = self.ctrl_spec.samples_per_step
        K = self.history_blocks
        f_samples = abi.obs_field("tx.monitor.samples")
        f_count = abi.obs_field("tx.monitor.count")
        f_latest = abi.obs_field("tx.monitor.latest")
        samples = released_obs[:, f_samples.offset:f_samples.offset + N * M].reshape(B, N, M)
        count = released_obs[:, f_count.offset:f_count.offset + N]
        latest = released_obs[:, f_latest.offset:f_latest.offset + N]
        # entries beyond the valid count hold the latest code, so a block always has M codes
        block = np.where(np.arange(M)[None, None, :] < count[:, :, None], samples, latest[:, :, None])
        if self._hist is None:
            self._hist = np.repeat(block[:, :, None, :], K, axis=2)
        else:
            self._hist[:, :, :-1, :] = self._hist[:, :, 1:, :]
            self._hist[:, :, -1, :] = block
            if restart is not None and restart.any():
                self._hist[restart] = block[restart][:, :, None, :]
        out = np.empty((B, self.obs_layout.size), np.float32)
        for codec in self.obs_layout.channels:
            if codec.name == "tx.monitor.samples":
                codes = self._hist.reshape(B, -1)
            else:
                field_info = abi.obs_field(codec.name)
                codes = released_obs[:, field_info.offset:field_info.offset + field_info.size]
            out[:, codec.offset:codec.offset + codec.size] = (codes - codec.code_offset) * (1.0 / (1 << codec.shift))   # x = (c - c0) 2^-s
        return out

    # reward
    # copies of the kernel's per-ring metric accumulators (B, N) that the reward uses.
    def _accumulator_snapshot(self) -> dict:
        metric_state = self.plant.metrics_state
        float_accumulators, int_accumulators = metric_state.af, metric_state.ai                       # float and int accumulators, (B, N, slot)
        return {"ticks": int_accumulators[:, :, AI_TICKS].copy(),
            # ticks counted since the plant reset
                "sum_err2": float_accumulators[:, :, AF_SUM_ERR2].copy(),
                    # tick-weighted sum of lock error^2, m^2 ticks
                "sum_reloma": float_accumulators[:, :, AF_SUM_RELOMA].copy(),
                    # tick-weighted sum of relOMA (linear), ticks
                "sum_oracle_reloma": float_accumulators[:, :, AF_SUM_ORACLE].copy(),
                    # tick-weighted sum of oracle relOMA, ticks
                "heater_energy_j": float_accumulators[:, :, AF_HEATER_ENERGY].copy(),   # heater energy, J
                "wrong_ticks": int_accumulators[:, :, AI_WRONG_TICKS].copy()}         # ticks on a foreign line

    # compute step rewards from differences in streaming metric accumulators.
    # use tick-weighted sums so the result describes the whole controller period.
    # return the world-average reward, per-ring reward, and RMS lock error in m.
    # store this edge's accumulator snapshot for the next difference.
    def _reward(self):
        now = self._accumulator_snapshot()
        previous = self._snap
        self._snap = now
        reward_spec = self.reward_spec
        n_ticks = (now["ticks"] - previous["ticks"]).astype(np.float64)      # ticks in this step, per ring
        n_ticks = np.where(n_ticks > 0, n_ticks, np.nan)
        mean_sq_err_m2 = (now["sum_err2"] - previous["sum_err2"]) / n_ticks
        scale_m = (reward_spec.lock_scale_pm if reward_spec.lock_scale_pm is not None
                 else self.metrics_spec.lock_threshold_pm) * 1e-12
        normalized_error_squared = mean_sq_err_m2 / (scale_m * scale_m)
        lock_term = -np.log1p(normalized_error_squared) if reward_spec.shape == "log" else -np.minimum(normalized_error_squared, reward_spec.clip)
        ring_reward = reward_spec.w_lock * lock_term
        if reward_spec.w_oma:
            oracle_sum = now["sum_oracle_reloma"] - previous["sum_oracle_reloma"]
            reloma_sum = now["sum_reloma"] - previous["sum_reloma"]
            ring_reward = ring_reward + reward_spec.w_oma * np.where(oracle_sum > 0,
                reloma_sum / np.where(oracle_sum > 0, oracle_sum, 1.0), 0.0)
        if reward_spec.w_heater_per_w:
            ring_reward = ring_reward - (reward_spec.w_heater_per_w * (now["heater_energy_j"] - previous["heater_energy_j"])
                     / (n_ticks * self.dt_s))
        if reward_spec.w_wrong_line:
            ring_reward = ring_reward - reward_spec.w_wrong_line * (now["wrong_ticks"] - previous["wrong_ticks"]) / n_ticks
        ring_reward = np.nan_to_num(ring_reward, nan=0.0)
        if reward_spec.exclude_unreachable and self._warm_no_code.any():
            reachable = (~self._warm_no_code).astype(np.float64)
            reachable_count = reachable.sum(axis=1)
            total = (ring_reward * reachable).sum(axis=1) / np.where(reachable_count > 0, reachable_count, 1.0)
        else:
            total = ring_reward.mean(axis=1)
        return total, ring_reward, np.sqrt(np.nan_to_num(mean_sq_err_m2, nan=0.0))

    # info['truth'] of a step — the selected Truth fields at the edge (the state at the end of the last plant interval), the lock reference and error, the step rms lock error (from the accumulators) and the per-ring reward.
    def _truth_info(self, reward_per_ring, rms) -> dict:
        state = self.plant.S
        truth = state.truth                             # (B, Truth field, N)
        out = {name: truth[:, TR[name], :].copy() for name in self.truth_fields}
        reference = self.plant.metrics_state.ref
        out["lock_reference_m"] = reference.copy()
        out["lock_error_m"] = truth[:, TR["detuning_m"], :] - reference
        out["reward_per_ring"] = reward_per_ring.copy()
        out["step_rms_lock_error_m"] = rms.copy()
        return out

    # misc
    # drop the plant (gymnasium calls this from `close`).
    def close_extras(self, **kwargs) -> None:
        self.plant = None

    def __repr__(self) -> str:
        return (f"MRMEnv(B={self.num_envs}, N={self.ctrl_spec.n_rings}, obs={self.obs_layout.size}, "
                f"act={self.action_codec.size}, frame_skip={self.frame_skip_ticks} ticks, start={self.start!r})")

    # deployment
    # build a compiled controller that reproduces this continuous policy interface.
    # pack the same history, sensor codecs, fixed-point MLP, and action limits.
    # inside the kernel, integer shifts reproduce observation normalization exactly.
    # clip outputs to [-1, 1], round decoded code deltas to nearest even, and update held commands.
    # scenario reset faults restart history and held-command state.
    # require matching MLP input/output sizes, signed Q0 input with enough fraction bits, and wrap or saturate overflow.
    # checked overflow cannot raise inside this kernel.
    # use the environment's start conditions when comparing command parity.
    def compiled_policy(self, mlp: FxpMlp, *, name: str = "mlp_policy",
                        latency_ticks: int | None = None) -> CompiledController:
        if self.action_codec.discrete:
            raise ValueError("compiled_policy needs action_type='continuous'")
        if mlp.n_in != self.obs_layout.size or mlp.n_out != self.action_codec.size:
            raise ValueError(f"MLP is {mlp.n_in} -> {mlp.n_out}; the env needs {self.obs_layout.size} -> "
                             f"{self.action_codec.size}")
        in_frac_bits = mlp.in_fmt.frac_bits
        max_shift = max(c.shift for c in self.obs_layout.channels)
        if in_frac_bits < max_shift or mlp.in_fmt.int_bits != 0:
            raise ValueError(f"the MLP input format must be Q0.n with n >= {max_shift} (got {mlp.in_fmt}) so the "
                             "normalized codes are exact")
        if mlp.overflow == int(Overflow.CHECKED):
            raise ValueError("overflow 'checked' raises; a controller inside the kernel must wrap or saturate")
        if mlp.out_fmt.frac_bits > 40:
            raise ValueError("the MLP output format needs <= 40 fraction bits")
        # user parameters (params_i64) - the header, 3 entries per temperature sensor, then the packed MLP
        obs_layout = self.obs_layout
        header = np.zeros(_USER_HEADER_SIZE, np.int64)
        monitor = obs_layout.channel("tx.monitor.samples")
        heater = obs_layout.channel("tx.heater.applied")
        header[_USER_HISTORY_BLOCKS], header[_USER_IN_FRAC_BITS] = obs_layout.history_blocks, in_frac_bits
        header[_USER_MONITOR_C0], header[_USER_MONITOR_SHIFT] = monitor.code_offset, monitor.shift
        header[_USER_HEATER_C0], header[_USER_HEATER_SHIFT] = heater.code_offset, heater.shift
        names = [c.name for c in obs_layout.channels]
        if "tx.bias.applied" in names:
            bias = obs_layout.channel("tx.bias.applied")
            header[_USER_HAS_BIAS], header[_USER_BIAS_C0], header[_USER_BIAS_SHIFT] = 1, bias.code_offset, bias.shift
        temps = [c for c in obs_layout.channels if c.name.startswith("temp.")]
        header[_USER_N_TEMP] = len(temps)
        max_delta = {name: channel_max_delta
                     for name, _offset, _size, channel_max_delta, *_ in self.action_codec.channels}
        header[_USER_MAX_DELTA_HEATER] = max_delta["tx.heater"]
        header[_USER_MAX_DELTA_BIAS] = max_delta.get("tx.bias", 0)
        temp_rows = []                              # per sensor - channel id, c0, shift
        temp_id = {t.channel: TEMP_ID_BASE + i for i, t in enumerate(self.ctrl_spec.temp_sensors)}
        for c in temps:
            temp_rows += [temp_id[c.name], c.code_offset, c.shift]
        packed = mlp.pack()
        user_params = np.concatenate([header, np.asarray(temp_rows, np.int64), packed])
        latency = self.latency_ticks if latency_ticks is None else int(latency_ticks)
        return CompiledController(name, mlp_policy_init, mlp_policy_step, params_i64=user_params,
                                  latency_ticks=latency)


# evaluate the fixed-point policy on normalized observations for the Python env.
# require observations exactly representable in the MLP input format and outputs no wider than 24 bits so float32 actions remain exact.
# return `(B, A)` actions matching the compiled policy's command decoding.
def fxp_policy_action(mlp: FxpMlp, obs) -> np.ndarray:
    observations = np.asarray(obs, np.float64)
    x_raw = np.rint(observations * float(1 << mlp.in_fmt.frac_bits)).astype(np.int64)
    if not np.array_equal(x_raw.astype(np.float64) * 2.0 ** -mlp.in_fmt.frac_bits, observations):
        raise ValueError("observations are not exact in the MLP's input format")
    if mlp.out_fmt.bits > 24:
        raise ValueError("outputs wider than 24 bits are not exact in float32")
    raw_output = mlp.forward(x_raw)
    return (raw_output.astype(np.float64) * 2.0 ** -mlp.out_fmt.frac_bits).astype(np.float32)


# compiled policy
# the compiled policy is a CompiledController (mrm.control.api - init(spec_i64, spec_f64) -> (p, pf, s, sf), then step(p, pf, s, sf, obs, act) at every edge).
# Its integer arrays are laid out by the slot constants below.

# Header of the user parameters (CompiledController params_i64), written by MRMEnv.compiled_policy and read by mlp_policy_init.
# After the header - 3 entries per temperature sensor (channel id, c0, shift), then the packed MLP.
_USER_HISTORY_BLOCKS = 0        # K - monitor sample blocks in the history
_USER_IN_FRAC_BITS = 1          # f_in - fraction bits of the MLP input format (Q0.f_in)
_USER_MONITOR_C0 = 2            # normalization of tx.monitor.samples - x = (c - c0) 2^-shift
_USER_MONITOR_SHIFT = 3
_USER_HEATER_C0 = 4             # normalization of tx.heater.applied
_USER_HEATER_SHIFT = 5
_USER_HAS_BIAS = 6              # 1 when tx.bias.applied is in the observation
_USER_BIAS_C0 = 7               # normalization of tx.bias.applied
_USER_BIAS_SHIFT = 8
_USER_N_TEMP = 9                # number of temperature sensors in the observation
_USER_MAX_DELTA_HEATER = 10     # largest heater code change per step
_USER_MAX_DELTA_BIAS = 11       # largest bias code change per step (0 - bias is not an action)
_USER_HEADER_SIZE = 12

# Slots of p (p_i64, built by mlp_policy_init and read by mlp_policy_step)
_P_N_RINGS = 0
_P_HISTORY_BLOCKS = 1           # K
_P_SAMPLES_PER_BLOCK = 2        # M
_P_N_TEMP = 3
_P_IN_FRAC_BITS = 4             # f_in
_P_OBS_SIZE = 5                 # D - length of the env observation (the MLP input)
_P_MONITOR_C0 = 6
_P_MONITOR_SHIFT = 7
_P_HEATER_C0 = 8
_P_HEATER_SHIFT = 9
_P_HAS_BIAS = 10
_P_BIAS_C0 = 11
_P_BIAS_SHIFT = 12
_P_MAX_DELTA_HEATER = 13
_P_MAX_DELTA_BIAS = 14
_P_OUT_FRAC_BITS = 15           # f_out - fraction bits of the MLP output format
_P_OBS_OFF_SAMPLES = 16         # offsets in obs_i64 of the observed channels (-1 - not exposed)
_P_OBS_OFF_COUNT = 17
_P_OBS_OFF_LATEST = 18
_P_OBS_OFF_HEATER_APPLIED = 19
_P_OBS_OFF_BIAS_APPLIED = 20
_P_ACT_OFF_HEATER = 21          # offsets in act_i64 of the commands (-1 - not bound to the controller)
_P_ACT_OFF_BIAS = 22
_P_N_OUT = 23                   # MLP outputs (the action length A)
_P_MLP_MAX_WIDTH = 24           # widest MLP layer (sizes the scratch buffers in s)
_P_MLP_BASE = 25                # index in p where the packed MLP starts
_P_TEMP_BASE = 26               # 3 entries per sensor - obs offset, c0, shift - the packed MLP follows
_HALF_EVEN = int(Rounding.HALF_EVEN)
_SAMPLES_ID, _COUNT_ID, _LATEST_ID = int(ObsId.TX_MONITOR_SAMPLES), int(ObsId.TX_MONITOR_COUNT), \
    int(ObsId.TX_MONITOR_LATEST)
_HEATER_APPLIED_ID, _BIAS_APPLIED_ID = int(ObsId.TX_HEATER_APPLIED), int(ObsId.TX_BIAS_APPLIED)
_HEATER_ACT_ID, _BIAS_ACT_ID = int(ActId.TX_HEATER), int(ActId.TX_BIAS)


# pack policy constants and allocate its integer history and MLP work buffers.
# parameters store channel offsets, normalization, action limits, and the packed network.
# state starts with a started flag, then ring-major `N*K*M` history codes, raw Q0 inputs, network outputs, and two widest-layer scratch buffers.
# validate required channels and MLP dimensions before returning ABI arrays.
@nb.njit(cache=True)
def mlp_policy_init(spec_i64, spec_f64):
    n_rings = spec_i64[api.SPEC_I_N_RINGS]
    samples_per_block = spec_i64[api.SPEC_I_M]
    user_offset = spec_i64[api.SPEC_I_USER]                   # start of the user parameters in spec_i64
    n_user = spec_i64[api.SPEC_I_N_USER]
    history_blocks = spec_i64[user_offset + _USER_HISTORY_BLOCKS]
    n_temp = spec_i64[user_offset + _USER_N_TEMP]
    user_mlp_offset = _USER_HEADER_SIZE + 3 * n_temp               # packed MLP start within the user parameters
    mlp_size = n_user - user_mlp_offset
    parameters = np.zeros(_P_TEMP_BASE + 3 * n_temp + mlp_size, np.int64)
    parameters[_P_N_RINGS], parameters[_P_HISTORY_BLOCKS], parameters[_P_SAMPLES_PER_BLOCK], parameters[_P_N_TEMP], parameters[_P_IN_FRAC_BITS] = \
        n_rings, history_blocks, samples_per_block, n_temp, spec_i64[user_offset + _USER_IN_FRAC_BITS]
    parameters[_P_MONITOR_C0], parameters[_P_MONITOR_SHIFT] = spec_i64[user_offset + _USER_MONITOR_C0], spec_i64[user_offset + _USER_MONITOR_SHIFT]
    parameters[_P_HEATER_C0], parameters[_P_HEATER_SHIFT] = spec_i64[user_offset + _USER_HEATER_C0], spec_i64[user_offset + _USER_HEATER_SHIFT]
    parameters[_P_HAS_BIAS], parameters[_P_BIAS_C0], parameters[_P_BIAS_SHIFT] = spec_i64[user_offset + _USER_HAS_BIAS], \
        spec_i64[user_offset + _USER_BIAS_C0], spec_i64[user_offset + _USER_BIAS_SHIFT]
    parameters[_P_MAX_DELTA_HEATER], parameters[_P_MAX_DELTA_BIAS] = spec_i64[user_offset + _USER_MAX_DELTA_HEATER], \
        spec_i64[user_offset + _USER_MAX_DELTA_BIAS]
    parameters[_P_OBS_OFF_SAMPLES] = api.obs_offset(spec_i64, _SAMPLES_ID)
    parameters[_P_OBS_OFF_COUNT] = api.obs_offset(spec_i64, _COUNT_ID)
    parameters[_P_OBS_OFF_LATEST] = api.obs_offset(spec_i64, _LATEST_ID)
    parameters[_P_OBS_OFF_HEATER_APPLIED] = api.obs_offset(spec_i64, _HEATER_APPLIED_ID)
    parameters[_P_OBS_OFF_BIAS_APPLIED] = api.obs_offset(spec_i64, _BIAS_APPLIED_ID)
    parameters[_P_ACT_OFF_HEATER] = api.act_offset(spec_i64, _HEATER_ACT_ID)
    parameters[_P_ACT_OFF_BIAS] = api.act_offset(spec_i64, _BIAS_ACT_ID)
    if parameters[_P_OBS_OFF_SAMPLES] < 0 or parameters[_P_OBS_OFF_COUNT] < 0 or parameters[_P_OBS_OFF_LATEST] < 0 \
            or parameters[_P_OBS_OFF_HEATER_APPLIED] < 0 or parameters[_P_ACT_OFF_HEATER] < 0:
        raise ValueError("mlp policy: the monitor block, the heater register and tx.heater must be exposed")
    if parameters[_P_HAS_BIAS] != 0 and parameters[_P_OBS_OFF_BIAS_APPLIED] < 0:
        raise ValueError("mlp policy: tx.bias.applied is in the env layout but not exposed")
    if parameters[_P_MAX_DELTA_BIAS] > 0 and parameters[_P_ACT_OFF_BIAS] < 0:
        raise ValueError("mlp policy: the bias action needs tx.bias bound to the controller")
    for i in range(n_temp):
        parameters[_P_TEMP_BASE + 3 * i] = api.obs_offset(spec_i64, spec_i64[user_offset + _USER_HEADER_SIZE + 3 * i])
        if parameters[_P_TEMP_BASE + 3 * i] < 0:
            raise ValueError("mlp policy: a temperature sensor of the env layout is not exposed")
        parameters[_P_TEMP_BASE + 3 * i + 1] = spec_i64[user_offset + _USER_HEADER_SIZE + 3 * i + 1]
        parameters[_P_TEMP_BASE + 3 * i + 2] = spec_i64[user_offset + _USER_HEADER_SIZE + 3 * i + 2]
    base = _P_TEMP_BASE + 3 * n_temp
    for j in range(mlp_size):
        parameters[base + j] = spec_i64[user_offset + user_mlp_offset + j]
    if fxp_mlp_check(parameters, base) != 0:
        raise ValueError("mlp policy: the packed MLP is malformed")
    parameters[_P_MLP_BASE] = base
    observation_size = (n_rings * history_blocks * samples_per_block + n_rings
                        + (n_rings if parameters[_P_HAS_BIAS] != 0 else 0) + n_temp)
    parameters[_P_OBS_SIZE] = observation_size
    parameters[_P_N_OUT] = parameters[base + H_N_OUT]
    parameters[_P_MLP_MAX_WIDTH] = parameters[base + H_MAX_WIDTH]
    parameters[_P_OUT_FRAC_BITS] = parameters[base + H_SIZE + (parameters[base + H_N_LAYERS] - 1) * L_SIZE + L_OUT_FRAC]   # the output format
    if parameters[base + H_N_IN] != observation_size:
        raise ValueError("mlp policy: MLP input size differs from the observation size")
    if parameters[_P_N_OUT] != n_rings + (n_rings if parameters[_P_MAX_DELTA_BIAS] > 0 else 0):
        raise ValueError("mlp policy: MLP output size differs from the action size")
    state = np.zeros(1 + n_rings * history_blocks * samples_per_block + observation_size + parameters[_P_N_OUT] + 2 * parameters[_P_MLP_MAX_WIDTH], np.int64)
    return parameters, np.zeros(0), state, np.zeros(0)


# update monitor history, evaluate the fixed-point policy, and add code deltas.
# normalize sensor codes as raw Q0 integers with shifts.
# evaluate without new allocations or array views.
# clip each network output to the representation of [-1, 1] before scaling and nearest-even rounding.
# the plant then clips held commands to hardware ranges and counts saturation.
@nb.njit(cache=True)
def mlp_policy_step(p, pf, s, sf, obs, act):
    n_rings = p[_P_N_RINGS]
    history_blocks = p[_P_HISTORY_BLOCKS]
    samples_per_block = p[_P_SAMPLES_PER_BLOCK]
    in_frac_bits = p[_P_IN_FRAC_BITS]
    history_size = n_rings * history_blocks * samples_per_block
    history_offset = 1                                       # start of the history in s (s[0] is the 'started' flag)
    first = s[0] == 0
    # 1.
    # shift the history by one block and append this edge's block (K copies of it at the first step)
    for r in range(n_rings):
        count = obs[p[_P_OBS_OFF_COUNT] + r]
        latest_code = obs[p[_P_OBS_OFF_LATEST] + r]
        ring_history_offset = history_offset + r * history_blocks * samples_per_block
        if not first:
            for j in range((history_blocks - 1) * samples_per_block):
                s[ring_history_offset + j] = s[ring_history_offset + samples_per_block + j]
        for j in range(samples_per_block):
            sample_code = obs[p[_P_OBS_OFF_SAMPLES] + r * samples_per_block + j] if j < count else latest_code
            if first:
                for kk in range(history_blocks):
                    s[ring_history_offset + kk * samples_per_block + j] = sample_code
            else:
                s[ring_history_offset + (history_blocks - 1) * samples_per_block + j] = sample_code
    s[0] = 1
    # 2.
    # the MLP input - each code as (c - c0) << (f_in - shift), the raw Q0.f_in value of x = (c - c0) 2^-shift
    input_offset = history_offset + history_size                             # start of the MLP input in s
    input_index = 0
    for i in range(history_size):
        s[input_offset + input_index] = (s[history_offset + i] - p[_P_MONITOR_C0]) << (in_frac_bits - p[_P_MONITOR_SHIFT])
        input_index += 1
    for r in range(n_rings):
        s[input_offset + input_index] = (obs[p[_P_OBS_OFF_HEATER_APPLIED] + r] - p[_P_HEATER_C0]) << (in_frac_bits - p[_P_HEATER_SHIFT])
        input_index += 1
    if p[_P_HAS_BIAS] != 0:
        for r in range(n_rings):
            s[input_offset + input_index] = (obs[p[_P_OBS_OFF_BIAS_APPLIED] + r] - p[_P_BIAS_C0]) << (in_frac_bits - p[_P_BIAS_SHIFT])
            input_index += 1
    for t in range(p[_P_N_TEMP]):
        slot = _P_TEMP_BASE + 3 * t
        s[input_offset + input_index] = (obs[p[slot]] - p[slot + 1]) << (in_frac_bits - p[slot + 2])
        input_index += 1
    # 3.
    # evaluate the MLP and decode its outputs into code deltas
    output_offset = input_offset + p[_P_OBS_SIZE]                        # start of the MLP output in s
    fxp_mlp_eval(p, p[_P_MLP_BASE], s, input_offset, output_offset, output_offset + p[_P_N_OUT])
    out_frac_bits = p[_P_OUT_FRAC_BITS]
    one = np.int64(1) << out_frac_bits
    # clip raw outputs to the same [-1, 1] range as the Python action decoder.
    # this also bounds the scaled delta product within int64 for supported formats.
    for r in range(n_rings):
        clipped_output = min(max(s[output_offset + r], -one), one)
        act[p[_P_ACT_OFF_HEATER] + r] = act[p[_P_ACT_OFF_HEATER] + r] + fx_round_shift(clipped_output * p[_P_MAX_DELTA_HEATER],
                                                                                       out_frac_bits, _HALF_EVEN)
    if p[_P_MAX_DELTA_BIAS] > 0:
        for r in range(n_rings):
            clipped_output = min(max(s[output_offset + n_rings + r], -one), one)
            act[p[_P_ACT_OFF_BIAS] + r] = act[p[_P_ACT_OFF_BIAS] + r] + fx_round_shift(clipped_output * p[_P_MAX_DELTA_BIAS],
                                                                                       out_frac_bits, _HALF_EVEN)


# B = 1 view
# single-world Gymnasium view with unbatched observations, actions, and info.
# accept the vector environment options except num_envs.
# reward is a scalar.
# call reset after truncation.
# stepping again performs the same next-step autoreset as the vector view.
class MRMSingleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Any = "g26", **kwargs):
        if "num_envs" in kwargs and int(kwargs.pop("num_envs")) != 1:
            raise ValueError("MRMSingleEnv is one world; use MRMEnv for a batch")
        self.venv = MRMEnv(config, num_envs=1, **kwargs)
        self.observation_space = self.venv.single_observation_space
        self.action_space = self.venv.single_action_space

    # reset the one world.
    # returns (obs, info) without the batch axis.
    def reset(self, *, seed: int | None = None, options: Mapping | None = None):
        super().reset(seed=seed)
        obs, info = self.venv.reset(seed=seed, options=options)
        return obs[0], _unbatch(info)

    # one step of the one world.
    # returns (obs, reward, terminated, truncated, info) without the batch axis.
    def step(self, action):
        obs, reward, terminated, truncated, info = self.venv.step(np.asarray(action)[None, ...])
        return obs[0], float(reward[0]), bool(terminated[0]), bool(truncated[0]), _unbatch(info)

    def close(self):
        self.venv.close()

    # the Plant of the wrapped vector env.
    @property
    def plant(self):
        return self.venv.plant


# world 0 of a batched info dict.
# nested dicts (info['truth']) are reduced the same way.
# every array is copied.
def _unbatch(info):
    out = {}
    for key, value in info.items():
        if isinstance(value, dict):
            out[key] = _unbatch(value)
        else:
            out[key] = np.asarray(value)[0].copy()
    return out


# register both envs with gymnasium under ENV_ID (once per process).
def _register() -> None:
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point="archive.rl.gym_env:MRMSingleEnv",
                     vector_entry_point="archive.rl.gym_env:MRMEnv", nondeterministic=False)


_register()
