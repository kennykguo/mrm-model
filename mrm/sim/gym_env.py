"""Batched reinforcement-learning environment over the plant batch (07 P1-22).

07 P1-22 (normative): "``MRMEnv`` as a ``gymnasium.vector.VectorEnv`` over the plant batch, with no subprocess
workers, and spaces generated from the ChannelSpecs. frame_skip = the controller period. Actions are bounded delta
codes, plus the bias. The observation is the last K sample blocks, applied codes and temperature codes, normalized.
The reward is computed in the kernel from Truth, which appears only in ``info['truth']``. Each reset draws a new world
and device (domain randomization over ``uncertainty`` and ``variation``). Target: at least 1e5 env steps/s at
B = 256. A fixed-point MLP inference helper in ``fxp``." (the helper: :mod:`mrm.control.fxp_mlp`).

How it maps onto the engine
---------------------------
- **One batch, no workers.** The B sub-environments are the B worlds of one :class:`mrm.plant.plant.Plant`. A step
  runs the compiled kernel on the engine's Python-controller path (``MODE_PYTHON``): the kernel stops AT the next
  controller edge after releasing the sensors there; the env reads the released channels from the flat observation
  vectors (the ABI layout of mrm.core.channels, the same numbers a controller gets), writes the new command into the
  held command vector and resumes, which pushes it into the controller pipeline exactly as for a Python controller
  (mrm.sim.runner._run_python). The kernel's thread pool runs the worlds in parallel.
- **frame_skip = the controller period.** One env step = one controller period (``spec.period_ticks``; g26: 10
  ticks = 10 us, 100 kHz). The action of step k is issued at edge k and latches at k + ``latency_ticks`` (07 P0-6),
  then passes the plant's DAC latency, slew and settle.
- **Actions** (generated from the actuator ChannelSpecs bound to the controller): bounded delta codes per ring,
  ``tx.heater`` always, ``tx.bias`` when the config binds it to the controller
  (``controller_interface.bind.tx.bias=controller`` and ``scenario.bias_v=null``). Continuous (default):
  Box(-1, 1, (A,)) float32, delta = rint(clip(a, -1, 1) * max_delta) codes (round half to even); discrete:
  MultiDiscrete(2 max_delta + 1), delta = a - max_delta. The new command is held + delta; the kernel clips it to the
  channel range and counts the clipped elements (07 P1-19), and the clipped value is the next held command.
- **Observation** (generated from the sensor ChannelSpecs; :class:`ObsLayout`): the last K monitor sample blocks per
  ring (``tx.monitor.samples``; entries beyond ``tx.monitor.count`` are filled with the held ``tx.monitor.latest``,
  so a block always has M codes), the applied codes (``tx.heater.applied``, ``tx.bias.applied``: the latched DAC
  registers, 07 P0-6) and the temperature-sensor codes (``temp.<name>``). Each code c of a channel with range
  [lo, hi] is normalized by a power of two: x = (c - c0) 2^-s with 2^(s+1) the smallest power of two >= hi - lo + 1
  and c0 = lo + 2^s, so x is in [-1, 1) and EXACT in float32 (and in a Q0.15 fixed-point input for converters up to
  16 bits: the compiled policy computes the same numbers with shifts). At reset the history holds K copies of the
  first block, which is empty (code 0 = -1.0) at the first edge after a reset: the monitor has not delivered a
  sample yet, which is what a controller sees there too. Nothing else enters the observation: no Truth, no plant
  state.
- **Reward, from Truth, computed in the kernel.** The kernel's streaming metric accumulators (mrm.sim.metrics,
  ``metrics_tick_row``, every plant tick from the Truth row of that tick) sum per ring the squared lock error
  e = d - d_ref, relOMA, the oracle relOMA, heater energy and the ticks on a foreign line. The env differences these
  sums over the step (B x N subtractions; nothing is recomputed in Python) and combines the per-ring step means with
  :class:`RewardSpec`. Lock reference (``metrics.lock_reference``): default 'oracle' here (each ring's best
  open-loop-stable relOMA point, re-evaluated by the kernel every ``oracle_period_ticks``; a modulator is held there,
  0.3-0.5 FWHM red of its line), or 'line' / 'target'. Truth (the per-ring Truth fields, the lock reference, the
  lock error, the step rms lock error, the per-ring reward and, at the end of an episode, the episode metrics)
  appears only in ``info['truth']`` (at a reset: only ``warm_start_no_code``; no plant tick has run yet).
- **Domain randomization.** Each reset (explicit or automatic) realizes B NEW worlds and devices
  (``params.realize`` with ``uncertainty`` = 'sample' | 'lhs' | 'nominal' and ``variation`` = 'sample' | 'off'),
  builds a new Plant and starts it. World seeds: reset r of a seeded env uses base seed (b0 + r) * B for every stream
  (device, uncertainty, noise, scenario), world b uses base + b (07 P1-13), with b0 drawn from the env's
  ``np_random`` at a seeded reset; worlds never repeat between resets (until 2^31 / B / 2 resets). ``options={'seeds':
  int | dict}`` pins the seeds of one reset (evaluation on fixed worlds).
- **Start** (``start``): 'warm' (default: the quasi-static oracle lock codes of each world,
  mrm.sim.runner.warm_start_codes, thermal state settled on the hot branch, Plant.settle; the hand-off contract of
  the feedback baselines, examples/controllers/README.md; rings without a quasi-static code start at code_min and are
  flagged in ``info['truth']['warm_start_no_code']``), 'cold' (the config's reset protocol, heaters at code_min;
  g26 laser_ramp), 'cold_settled' (heaters at code_min, settled on the cold branch: the scorecard's cold start).
  ``start_jitter_codes`` J adds a uniform integer offset in [-J, J] to the warm codes (seeded by each world's device
  seed; clipped to the DAC range).
- **Episodes and autoreset.** Every world truncates after ``max_episode_steps`` steps at the same step (there is no
  per-world termination: a lost ring can be re-acquired, and a per-world reset would need a per-world re-realization
  of the batch). Autoreset mode NEXT_STEP (gymnasium's default): the step after the truncation resets every world and
  returns the new observation with reward 0 and terminated = truncated = False.
- **Controller reset** (scenario ``fault.controller_reset``, 07 P1-10): the env holds the controller state of a
  delta policy (the held command and the K-block history) and restarts it at the edge where the fault arrives, as
  the kernel restarts a compiled controller (the held heater / bias commands re-filled from the latched DAC
  registers, the history restarted from that edge's block); ``info['controller_reset']`` (B,) flags it so that a
  policy with its own memory can reset that too. :meth:`MRMEnv.compiled_policy` does the same inside the kernel.
- **Speed.** One kernel call per step for the whole batch. Measured (docs/handoff/rl.md): ~2-3e4 env steps/s at
  B = 256 on the 8-core M2 under the usual load, against the 07 target of 1e5; the plant itself (B x 10 ticks per
  step at ~2-3 us per world-tick) caps it below ~5e4, and the per-call overhead of the Python path (argument
  unboxing, per-block setup) adds ~30 %. The engine requests are in the handoff.

:class:`MRMSingleEnv` is the B = 1 ``gymnasium.Env`` view (Gymnasium's env checker runs on it). Both are
registered: ``gymnasium.make('mrm/MRM-v0', ...)`` and ``gymnasium.make_vec('mrm/MRM-v0', num_envs=B,
vectorization_mode='vector_entry_point', ...)``.

:meth:`MRMEnv.compiled_policy` turns a quantized MLP policy (:mod:`mrm.control.fxp_mlp`) into a CompiledController
that computes the SAME observation in integer arithmetic inside the kernel (07 P0-9), so a trained policy runs on the
engine at full speed and in the binary bridge / RTL flow; :func:`fxp_policy_action` gives the env action of the same
fixed-point policy (bit-identical commands; tests/verification/test_gym_env_fxp.py).
"""
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

from mrm.control import api as _api
from mrm.control.api import CompiledController, ControllerSpec
from mrm.control.fxp import Overflow, Rounding, fx_round_shift
from mrm.control.fxp_mlp import (H_MAX_WIDTH, H_N_IN, H_N_LAYERS, H_N_OUT, H_SIZE, L_OUT_FRAC, L_SIZE, FxpMlp,
                                 fxp_mlp_check, fxp_mlp_eval)
from mrm.core.channels import TEMP_ID_BASE, ActId, ObsId
from mrm.plant.plant import Plant
from mrm.sim import kernel as K
from mrm.sim import metrics as M
from mrm.sim.metrics import TR, TRUTH_FIELDS, MetricsSpec

__all__ = ["MRMEnv", "MRMSingleEnv", "RewardSpec", "ObsLayout", "ChannelCodec", "ActionCodec", "code_codec",
           "DEFAULT_TRUTH_FIELDS", "EPISODE_METRICS", "STARTS", "fxp_policy_action", "mlp_policy_init",
           "mlp_policy_step", "ENV_ID"]

ENV_ID = "mrm/MRM-v0"
STARTS = ("warm", "cold", "cold_settled")
#: Truth fields copied into info['truth'] every step (mrm.sim.metrics.TRUTH_FIELDS names; ``truth_fields='all'``
#: copies every one)
DEFAULT_TRUTH_FIELDS = ("detuning_m", "reloma", "oracle_reloma", "heater_code", "line_on", "t_ring_k")
#: episode metrics (mrm.sim.metrics.summarize keys, (B, N)) in info['truth']['episode'] at the truncation step
EPISODE_METRICS = ("rms_lock_error_pm", "max_abs_lock_error_pm", "locked_final", "excursion_count", "time_unlocked_s",
                   "wrong_line_s", "mean_reloma_db", "oma_penalty_db", "heater_energy_j", "heater_sat_s",
                   "heater_low_rail_s")
_SEED_KEYS = ("device", "uncertainty", "noise", "scenario")


# ================================================================================================== codecs
def code_codec(lo: int, hi: int) -> tuple[int, int]:
    """(c0, s) of the power-of-two normalization x = (c - c0) 2^-s of a code range [lo, hi] (module docstring):
    2^(s+1) is the smallest power of two >= hi - lo + 1 and c0 = lo + 2^s, so x is in [-1, (hi - c0) 2^-s] within
    [-1, 1)."""
    span = int(hi) - int(lo) + 1
    if span < 1:
        raise ValueError(f"empty code range [{lo}, {hi}]")
    p = max(1, int(math.ceil(math.log2(span))))
    while (1 << p) < span:                         # guard the float log2
        p += 1
    s = p - 1
    return int(lo) + (1 << s), s


@dataclass(frozen=True)
class ChannelCodec:
    """One sensor channel's place in the flat observation: ``offset`` and ``size`` in the vector, the element shape,
    the normalization x = (code - code_offset) 2^-shift and the bounds of x."""
    name: str
    offset: int
    size: int
    shape: tuple
    code_offset: int
    shift: int
    lo: float
    hi: float


@dataclass(frozen=True)
class ObsLayout:
    """The flat observation (module docstring). Channels in this fixed order: ``tx.monitor.samples`` as (N, K, M)
    (ring, block oldest -> newest, sample), ``tx.heater.applied`` (N), ``tx.bias.applied`` (N) when exposed,
    ``temp.<name>`` (1 each) in spec order."""
    n_rings: int
    history_blocks: int
    samples_per_block: int
    channels: tuple
    size: int

    def channel(self, name: str) -> ChannelCodec:
        for c in self.channels:
            if c.name == name:
                return c
        raise KeyError(name)

    def slices(self) -> dict[str, slice]:
        return {c.name: slice(c.offset, c.offset + c.size) for c in self.channels}

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.empty(self.size, np.float32)
        hi = np.empty(self.size, np.float32)
        for c in self.channels:
            lo[c.offset:c.offset + c.size] = c.lo
            hi[c.offset:c.offset + c.size] = c.hi
        return lo, hi

    def code_offsets(self) -> np.ndarray:
        """(size,) int64: c0 of every element."""
        out = np.empty(self.size, np.int64)
        for c in self.channels:
            out[c.offset:c.offset + c.size] = c.code_offset
        return out

    def shifts(self) -> np.ndarray:
        """(size,) int64: s of every element."""
        out = np.empty(self.size, np.int64)
        for c in self.channels:
            out[c.offset:c.offset + c.size] = c.shift
        return out

    def unflatten(self, obs) -> dict[str, np.ndarray]:
        """{channel: normalized values (..., *shape)} of observations (..., size)."""
        a = np.asarray(obs)
        lead = a.shape[:-1]
        return {c.name: a[..., c.offset:c.offset + c.size].reshape(lead + c.shape) for c in self.channels}

    def to_codes(self, obs) -> dict[str, np.ndarray]:
        """{channel: the codes (int64, (..., *shape))} back from normalized observations (exact inverse)."""
        a = np.asarray(obs, np.float64)
        lead = a.shape[:-1]
        out = {}
        for c in self.channels:
            x = a[..., c.offset:c.offset + c.size] * float(1 << c.shift) + c.code_offset
            out[c.name] = np.rint(x).astype(np.int64).reshape(lead + c.shape)
        return out


@dataclass(frozen=True)
class ActionCodec:
    """The action vector: per actuator channel (name, offset in the action vector, size, max delta, code range,
    offset in the ABI act vector). ``size`` is the action length A."""
    channels: tuple
    size: int
    discrete: bool

    def names(self) -> tuple:
        return tuple(c[0] for c in self.channels)


@dataclass(frozen=True)
class RewardSpec:
    """Per-step reward from the kernel's accumulators (module docstring). Per ring n of world b, over the step:

    - lock: u = mean(e^2) / s^2 with s = ``lock_scale_pm`` (default: metrics.lock_threshold_pm, the 5 pm lock
      budget); term -log1p(u) (``shape='log'``, default: 0 at e = 0, -0.69 at an rms error of s, -10.6 at 1000 s;
      bounded growth keeps a lost ring from swamping the batch) or -min(u, clip) (``shape='quadratic'``);
    - oma: mean relOMA / mean oracle relOMA over the step (0 when the oracle relOMA is 0), in [0, ~1];
    - heater: mean heater power over the step, W;
    - wrong line: fraction of the step on a foreign line (07 P1-12, the kernel's wrong-line flag).

    r_bn = w_lock * lock + w_oma * oma - w_heater_per_w * heater - w_wrong_line * wrong; the env reward is the mean
    over the N rings (info['truth']['reward_per_ring'] has r_bn). ``exclude_unreachable``: leave out of that mean
    the rings that the warm start could not place at a lock (no quasi-static code in [code_min, code_max]:
    info['truth']['warm_start_no_code'] at the reset; their r_bn is still reported); a world with no other ring gets
    reward 0.
    """
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


def _load_config(config, overrides, alts):
    from mrm import params
    if isinstance(config, (str, Path)):
        return params.load(config, overrides=overrides, alts=alts)
    if overrides or alts:
        raise ValueError("overrides / alts apply to a preset name or YAML path, not to a loaded Config")
    if not callable(getattr(config, "si", None)):
        raise TypeError("config must be a preset name, a YAML path or an mrm.params Config")
    return config


def _norm_seeds(seeds) -> dict:
    if isinstance(seeds, (int, np.integer)):
        return {k: int(seeds) for k in _SEED_KEYS}
    d = dict(seeds)
    bad = set(d) - set(_SEED_KEYS)
    if bad:
        raise ValueError(f"unknown seed streams {sorted(bad)}; use {_SEED_KEYS}")
    out = {k: 0 for k in _SEED_KEYS}
    out.update({k: int(v) for k, v in d.items()})
    return out


# ================================================================================================== the env
class MRMEnv(VectorEnv):
    """B worlds of one MRM bank as a gymnasium vector environment (module docstring).

    Parameters
    ----------
    config: preset name, YAML path or loaded Config (default 'g26'); ``overrides`` / ``alts`` as mrm.params.load.
    num_envs: B. scenario: None (the config's), a registry name, a Scenario, or a callable
        ``(numpy.random.Generator) -> Scenario`` called at every reset (seeded by the world seed: scenario
        randomization). max_episode_steps: steps per episode (None: never truncate). history_blocks: K.
    max_delta_heater_codes, max_delta_bias_codes: the largest code change per step (g26 nominal: ~0.05 pm of
        resonance per heater code at the lock, so 256 codes ~ 13 pm per 10 us step; the A1 aggressor at its peak
        rate needs ~1.8 pm per step). action_type: 'continuous' | 'discrete'.
    start, start_jitter_codes: module docstring. uncertainty, variation: the world draws of every reset.
    reward: a RewardSpec (or a dict of its fields). metrics: a dict of mrm.sim.metrics.MetricsSpec overrides on top
        of the config's ``metrics`` section and the env default ``lock_reference='oracle'``.
    truth_fields: Truth fields copied into info['truth'] each step (names, 'all', or () for none).
    latency_ticks: the policy's own pipeline delay (Controller.latency_ticks, 07 P0-6; default 0).
    n_threads: kernel threads (default numba's). substep_policy: 'fixed' | 'auto' (default the config's). oracle:
        evaluate the oracle relOMA (needed for lock_reference 'oracle' and the oma term). record: Truth record specs
        (mrm.sim.metrics.RecordSpec list) sized for one episode (max_episode_steps, or 1000 steps when None),
        read with ``env.plant.traces()`` (analysis and tests; costs memory). sh_tol_k: self-heating root
        tolerance (Plant option).
    """

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
        mkw = {"lock_reference": "oracle"} if self.oracle else {}
        mkw.update(dict(metrics or {}))
        self.metrics_spec = MetricsSpec.from_config(self.cfg, **mkw)
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
        sc0 = self._make_scenario(np.random.default_rng(0))
        self.ctrl_spec = ControllerSpec.from_config(self.cfg, n_batch=B, nominal=self.cfg, scenario_drives=sc0.drives,
                                               scenario_offsets=sc0.offsets)
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
            nvec = np.concatenate([np.full(c[2], 2 * c[3] + 1, np.int64) for c in self.action_codec.channels])
            self.single_action_space = spaces.MultiDiscrete(nvec)
        else:
            self.single_action_space = spaces.Box(-1.0, 1.0, (A,), np.float32)
        self.observation_space = batch_space(self.single_observation_space, B)
        self.action_space = batch_space(self.single_action_space, B)
        self._n_blocks = max(1, (2 ** 31) // B // 2)
        self._block0: int | None = None
        self._episode = 0
        self._needs_reset = False
        self.plant: Plant | None = None
        self.seeds: dict | None = None
        self._ep_step = 0

    # ------------------------------------------------------------------------------------------- interface
    def _build_obs_layout(self, spec: ControllerSpec) -> ObsLayout:
        N, K_, M_ = spec.n_rings, self.history_blocks, spec.samples_per_step
        names = [f.name for f in spec.layout.obs]
        rows = [("tx.monitor.samples", (N, K_, M_))]
        for need in ("tx.monitor.samples", "tx.monitor.count", "tx.monitor.latest", "tx.heater.applied"):
            if need not in names:
                raise ValueError(f"MRMEnv needs the sensor channel {need!r} in the controller interface")
        # the ABI packs the monitor block ring-major, M samples per ring (_observe and mlp_policy_step read it so)
        for name, size in (("tx.monitor.samples", N * M_), ("tx.monitor.count", N), ("tx.monitor.latest", N),
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
        chans, off = [], 0
        for name, shape in rows:
            ch = spec.channels[name]
            c0, s = code_codec(int(ch.min), int(ch.max))
            size = int(np.prod(shape))
            chans.append(ChannelCodec(name=name, offset=off, size=size, shape=tuple(shape), code_offset=c0, shift=s,
                                      lo=(int(ch.min) - c0) / float(1 << s), hi=(int(ch.max) - c0) / float(1 << s)))
            off += size
        return ObsLayout(n_rings=N, history_blocks=K_, samples_per_block=M_, channels=tuple(chans), size=off)

    @staticmethod
    def _build_action_codec(spec: ControllerSpec, dmax_h: int, dmax_b: int, discrete: bool) -> ActionCodec:
        acts = {f.name: f for f in spec.layout.act}
        if "tx.heater" not in acts:
            raise ValueError("MRMEnv needs tx.heater bound to the controller")
        rows, off = [], 0
        for name, dmax in (("tx.heater", dmax_h), ("tx.bias", dmax_b)):
            if name not in acts:
                continue
            ch = spec.channels[name]
            span = int(ch.max) - int(ch.min)
            if not 1 <= dmax <= span:
                raise ValueError(f"max delta of {name} must be in 1..{span} codes, got {dmax}")
            f = acts[name]
            rows.append((name, off, int(f.size), int(dmax), int(ch.min), int(ch.max), int(f.offset)))
            off += int(f.size)
        return ActionCodec(channels=tuple(rows), size=off, discrete=discrete)

    def _make_scenario(self, rng: np.random.Generator):
        from mrm.scenarios import registry as scen_registry
        from mrm.scenarios.generators import Scenario
        s = self._scenario_arg
        if s is None:
            return scen_registry.from_config(self.cfg)
        if isinstance(s, str):
            return scen_registry.create(s)
        if isinstance(s, Scenario):
            return s
        if callable(s):
            out = s(rng)
            if not isinstance(out, Scenario):
                raise TypeError("a scenario callable must return a mrm.scenarios.generators.Scenario")
            return out
        raise TypeError("scenario must be None, a registry name, a Scenario or a callable(rng) -> Scenario")

    # ------------------------------------------------------------------------------------------- reset
    def reset(self, *, seed: int | None = None, options: Mapping | None = None):
        """Realize B new worlds and devices, start them (``start``) and return the observation at the first
        controller edge. ``options``: {'seeds': int | {device, uncertainty, noise, scenario}} pins the world seeds
        of this reset; {'start': ...} / {'heater_code': (B, N) codes} override the start of this reset ('heater_code'
        is a warm start at those codes, settled on the hot branch)."""
        super().reset(seed=seed)
        if seed is not None or self._block0 is None:
            self._block0 = int(self.np_random.integers(0, self._n_blocks))
            self._episode = 0
        else:
            self._episode += 1
        return self._reset_worlds(dict(options or {}))

    def _reset_worlds(self, opts: dict):
        B = self.num_envs
        if opts.get("seeds") is not None:
            seeds = _norm_seeds(opts["seeds"])
        else:
            base = ((self._block0 + self._episode) % self._n_blocks) * B
            seeds = {k: base for k in _SEED_KEYS}
        self.seeds = seeds
        rng = np.random.default_rng([seeds["scenario"], 0x6D726D])
        sc = self._make_scenario(rng)
        self._build_plant(seeds, sc)
        start = opts.get("start", self.start)
        if start not in STARTS:
            raise ValueError(f"start must be one of {STARTS}")
        plant = self.plant
        no_code = np.zeros((B, self.ctrl_spec.n_rings), bool)
        hc = opts.get("heater_code")
        if hc is not None:
            plant.reset(protocol="branch_high", heater_code=np.asarray(hc, np.int64), settle=True)
        elif start == "warm":
            from mrm.sim import runner
            with warnings.catch_warnings():         # the unreachable rings are reported in info['truth']
                warnings.filterwarnings("ignore", message="warm_start_codes", category=UserWarning)
                codes, no_code = runner.warm_start_codes(plant, self.cfg, mode="quasi_static", return_no_code=True)
            if self.start_jitter_codes:
                # one generator per world, seeded by that world's device seed (base + b, 07 P1-13), so a world
                # replays alone at B = 1 with its own seeds; clipped to the DAC range (as Plant.reset would)
                J = self.start_jitter_codes
                j = np.stack([np.random.default_rng([seeds["device"] + b, 0x6A6974]).integers(-J, J + 1, codes.shape[1])
                              for b in range(B)])
                hp = plant.heater_params
                codes = np.clip(codes + j, hp.code_lo, hp.code_hi)
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
        self._controller_reset()                    # a reset raised before the first edge: the state is fresh anyway
        self._hist = None
        obs = self._observe()
        self._snap = self._acc_snapshot()
        self._sat0 = plant.S.sat.sum(axis=1).copy()
        # no plant tick has run since the reset (the Truth row is written by the kernel at every tick), so the reset
        # info carries only the start report
        info = {"tick": np.full(B, plant.tick, np.int64),
                "seed": np.full(B, seeds["device"], np.int64) + np.arange(B, dtype=np.int64),
                "truth": {"warm_start_no_code": self._warm_no_code.copy()}}
        return obs, info

    def _build_plant(self, seeds: dict, sc) -> None:
        from mrm import params
        B = self.num_envs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            pp = params.realize(self.cfg, B, seeds={"device": seeds["device"], "uncertainty": seeds["uncertainty"]},
                                uncertainty=self.uncertainty, variation=self.variation)
        spec = ControllerSpec.from_config(self.cfg, n_batch=B, nominal=self.cfg, scenario_drives=sc.drives,
                                          scenario_offsets=sc.offsets)
        if spec.layout.fingerprint() != self._layout_fp:
            raise ValueError("the scenario of this reset changes the controller interface (channel bindings); the "
                             "spaces of an env are fixed at construction")
        self.ctrl_spec = spec
        kw = {}
        if self.sh_tol_k is not None:
            kw["sh_tol_k"] = float(self.sh_tol_k)
        n_rec = (self.max_episode_steps or 1000) * self.frame_skip_ticks + self.frame_skip_ticks
        self.plant = Plant(pp, sc, spec, controller_latency_ticks=self.latency_ticks, noise_seed=seeds["noise"],
                           scenario_seed=seeds["scenario"], metrics=self.metrics_spec, record=self.record,
                           record_ticks=n_rec if self.record else 0, substep_policy=self.substep_policy,
                           oracle=self.oracle, histogram=False, **kw)

    # ------------------------------------------------------------------------------------------- step
    def step(self, actions):
        """Apply the actions at the current controller edge and advance one controller period (module docstring).
        Returns (obs (B, D) float32, reward (B,) float64, terminated (B,) bool, truncated (B,) bool, info)."""
        B = self.num_envs
        if self.plant is None:
            raise RuntimeError("call reset() first")
        if self._needs_reset:                       # NEXT_STEP autoreset: every world truncated at the last step
            self._episode += 1
            obs, info = self._reset_worlds({})
            return obs, np.zeros(B), np.zeros(B, bool), np.zeros(B, bool), info
        delta = self._deltas(actions)
        S = self.plant.S
        for (name, a_off, size, dmax, lo, hi, abi_off) in self.action_codec.channels:
            S.act[:, abi_off:abi_off + size] += delta[:, a_off:a_off + size]
        self._advance(resume=True)
        self._ep_step += 1
        creset = self._controller_reset()
        obs = self._observe(restart=creset)
        reward, r_ring, rms = self._reward()
        trunc = self.max_episode_steps is not None and self._ep_step >= self.max_episode_steps
        S = self.plant.S
        sat = S.sat.sum(axis=1)
        info = {"tick": np.full(B, self.plant.tick, np.int64), "clipped": (sat - self._sat0).astype(np.int64),
                "controller_reset": creset.copy(), "truth": self._truth_info(r_ring, rms)}
        self._sat0 = sat.copy()
        if trunc:
            m = self.plant.metrics()
            info["truth"]["episode"] = {k: np.asarray(m[k]).copy() for k in EPISODE_METRICS}
            self._needs_reset = True
        return obs, reward, np.zeros(B, bool), np.full(B, bool(trunc)), info

    def _deltas(self, actions) -> np.ndarray:
        """(B, A) int64 code deltas from the actions (module docstring)."""
        B, A = self.num_envs, self.action_codec.size
        a = np.asarray(actions)
        if a.shape != (B, A):
            raise ValueError(f"actions must have shape {(B, A)}, got {a.shape}")
        dmax = np.concatenate([np.full(c[2], c[3], np.int64) for c in self.action_codec.channels])
        if self.action_codec.discrete:
            if not np.issubdtype(a.dtype, np.integer):
                raise TypeError("discrete actions must be integers")
            ai = a.astype(np.int64)
            if np.any(ai < 0) or np.any(ai > 2 * dmax):
                raise ValueError("discrete action outside 0..2 max_delta")
            return ai - dmax
        af = a.astype(np.float64)
        if not np.all(np.isfinite(af)):
            raise ValueError("actions must be finite")
        return np.rint(np.clip(af, -1.0, 1.0) * dmax).astype(np.int64)

    def _advance(self, resume: bool) -> None:
        """Run the kernel on the Python path to the next controller edge (the sensors there are released)."""
        plant = self.plant
        n = 2 * self.frame_skip_ticks + 1
        while True:
            plant.run(n, mode=K.MODE_PYTHON, resume=resume, n_threads=self.n_threads)
            resume = False
            if int(plant.S.wi[0, K.WI_PENDING_EDGE]) == plant.tick:
                return

    def _controller_reset(self) -> np.ndarray:
        """Scenario ``fault.controller_reset`` (07 P1-10) at the current edge, per world: the env holds the
        controller state of a delta policy (the held command and the K-block history), so it restarts it as the
        kernel restarts a compiled controller there (mrm.sim.kernel: init state, then the held command re-filled
        from the latched DAC registers, ``_sync_act_a``): the held heater / bias commands of the reset worlds are set
        to their DAC registers, and :meth:`_observe` restarts their history. Clears the flag (as the runner's Python
        path does) and returns the (B,) mask, reported in ``info['controller_reset']``. The policy's own memory is
        the caller's to reset."""
        S, P = self.plant.S, self.plant.P
        creset = S.wi[:, K.WI_CTRL_RESET] != 0
        if np.any(creset):
            for i in range(P.act_map.shape[0]):
                kind, o, cnt = (int(v) for v in P.act_map[i, :3])
                if kind in (0, 1):                      # tx.heater, tx.bias (kernel._sync_act_a)
                    reg = K.RI_HE_REG if kind == 0 else K.RI_BI_REG
                    S.act[creset, o:o + cnt] = S.ri[creset, reg, :cnt]
            S.wi[creset, K.WI_CTRL_RESET] = 0
        return creset

    # ------------------------------------------------------------------------------------------- observation
    def _observe(self, restart: np.ndarray | None = None) -> np.ndarray:
        """The flat observation (ObsLayout) of the sensors released at the current edge. ``restart`` (B,) bool: the
        worlds whose history restarts here (K copies of this block; a controller reset)."""
        lay = self.ctrl_spec.layout
        o = self.plant.S.obs
        B, N = self.num_envs, self.ctrl_spec.n_rings
        M_ = self.ctrl_spec.samples_per_step
        K_ = self.history_blocks
        fs = lay.obs_field("tx.monitor.samples")
        fc = lay.obs_field("tx.monitor.count")
        fl = lay.obs_field("tx.monitor.latest")
        samp = o[:, fs.offset:fs.offset + N * M_].reshape(B, N, M_)
        cnt = o[:, fc.offset:fc.offset + N]
        latest = o[:, fl.offset:fl.offset + N]
        block = np.where(np.arange(M_)[None, None, :] < cnt[:, :, None], samp, latest[:, :, None])
        if self._hist is None:
            self._hist = np.repeat(block[:, :, None, :], K_, axis=2)
        else:
            self._hist[:, :, :-1, :] = self._hist[:, :, 1:, :]
            self._hist[:, :, -1, :] = block
            if restart is not None and restart.any():
                self._hist[restart] = block[restart][:, :, None, :]
        out = np.empty((B, self.obs_layout.size), np.float32)
        for c in self.obs_layout.channels:
            if c.name == "tx.monitor.samples":
                codes = self._hist.reshape(B, -1)
            else:
                f = lay.obs_field(c.name)
                codes = o[:, f.offset:f.offset + f.size]
            out[:, c.offset:c.offset + c.size] = (codes - c.code_offset) * (1.0 / (1 << c.shift))
        return out

    # ------------------------------------------------------------------------------------------- reward
    def _acc_snapshot(self) -> dict:
        ms = self.plant.metrics_state
        af, ai = ms.af, ms.ai
        return {"ticks": ai[:, :, M.AI_TICKS].copy(), "err2": af[:, :, M.AF_SUM_ERR2].copy(),
                "rel": af[:, :, M.AF_SUM_RELOMA].copy(), "orc": af[:, :, M.AF_SUM_ORACLE].copy(),
                "eh": af[:, :, M.AF_HEATER_ENERGY].copy(), "wrong": ai[:, :, M.AI_WRONG_TICKS].copy()}

    def _reward(self):
        """(reward (B,), per-ring reward (B, N), step rms lock error (B, N) in m) from the accumulator differences."""
        now = self._acc_snapshot()
        prev = self._snap
        self._snap = now
        rs = self.reward_spec
        dt_k = (now["ticks"] - prev["ticks"]).astype(np.float64)
        dt_k = np.where(dt_k > 0, dt_k, np.nan)
        ms_err = (now["err2"] - prev["err2"]) / dt_k
        scale = (rs.lock_scale_pm if rs.lock_scale_pm is not None else self.metrics_spec.lock_threshold_pm) * 1e-12
        u = ms_err / (scale * scale)
        lock = -np.log1p(u) if rs.shape == "log" else -np.minimum(u, rs.clip)
        r = rs.w_lock * lock
        if rs.w_oma:
            orc = now["orc"] - prev["orc"]
            rel = now["rel"] - prev["rel"]
            r = r + rs.w_oma * np.where(orc > 0, rel / np.where(orc > 0, orc, 1.0), 0.0)
        if rs.w_heater_per_w:
            r = r - rs.w_heater_per_w * (now["eh"] - prev["eh"]) / (dt_k * self.dt_s)
        if rs.w_wrong_line:
            r = r - rs.w_wrong_line * (now["wrong"] - prev["wrong"]) / dt_k
        r = np.nan_to_num(r, nan=0.0)
        if rs.exclude_unreachable and self._warm_no_code.any():
            w = (~self._warm_no_code).astype(np.float64)
            n_ok = w.sum(axis=1)
            total = (r * w).sum(axis=1) / np.where(n_ok > 0, n_ok, 1.0)
        else:
            total = r.mean(axis=1)
        return total, r, np.sqrt(np.nan_to_num(ms_err, nan=0.0))

    def _truth_info(self, r_ring, rms) -> dict:
        """info['truth'] of a step: the selected Truth fields at the edge (the state at the end of the last plant
        interval), the lock reference and error, the step rms lock error (from the accumulators) and the per-ring
        reward."""
        S = self.plant.S
        tv = S.tv
        out = {name: tv[:, TR[name], :].copy() for name in self.truth_fields}
        ref = self.plant.metrics_state.ref
        out["lock_reference_m"] = ref.copy()
        out["lock_error_m"] = tv[:, TR["detuning_m"], :] - ref
        out["reward_per_ring"] = r_ring.copy()
        out["step_rms_lock_error_m"] = rms.copy()
        return out

    # ------------------------------------------------------------------------------------------- misc
    def close_extras(self, **kwargs) -> None:
        self.plant = None

    def __repr__(self) -> str:
        return (f"MRMEnv(B={self.num_envs}, N={self.ctrl_spec.n_rings}, obs={self.obs_layout.size}, "
                f"act={self.action_codec.size}, frame_skip={self.frame_skip_ticks} ticks, start={self.start!r})")

    # ------------------------------------------------------------------------------------------- deployment
    def compiled_policy(self, mlp: FxpMlp, *, name: str = "mlp_policy",
                        latency_ticks: int | None = None) -> CompiledController:
        """A CompiledController that runs the quantized policy ``mlp`` on this env's observation inside the kernel
        (module docstring): the same K-block history, the same normalization (as integer shifts into the MLP's
        input format), the same action decoding (clip to [-1, 1], then round half to even of y * max_delta), the
        same held-command update, the same restart on a scenario controller reset (the history and the held command;
        :meth:`step`). Run it with
        ``mrm.sim.runner.run(cfg, controller=..., heater_code='quasi_static', settle=True, ...)`` (the env's warm
        start). Needs a continuous action space, an MLP with the env's input and output sizes, a signed input format
        with at least as many fraction bits as the largest normalization shift, and an overflow mode other than
        CHECKED (no exception may leave the kernel)."""
        if self.action_codec.discrete:
            raise ValueError("compiled_policy needs action_type='continuous'")
        if mlp.n_in != self.obs_layout.size or mlp.n_out != self.action_codec.size:
            raise ValueError(f"MLP is {mlp.n_in} -> {mlp.n_out}; the env needs {self.obs_layout.size} -> "
                             f"{self.action_codec.size}")
        fin = mlp.in_fmt.frac_bits
        smax = max(c.shift for c in self.obs_layout.channels)
        if fin < smax or mlp.in_fmt.int_bits != 0:
            raise ValueError(f"the MLP input format must be Q0.n with n >= {smax} (got {mlp.in_fmt}) so the "
                             "normalized codes are exact")
        if mlp.overflow == int(Overflow.CHECKED):
            raise ValueError("overflow 'checked' raises; a controller inside the kernel must wrap or saturate")
        if mlp.out_fmt.frac_bits > 40:
            raise ValueError("the MLP output format needs <= 40 fraction bits")
        lay = self.obs_layout
        hdr = np.zeros(_CP_HEADER, np.int64)
        mon = lay.channel("tx.monitor.samples")
        hea = lay.channel("tx.heater.applied")
        hdr[_CP_K], hdr[_CP_FIN] = lay.history_blocks, fin
        hdr[_CP_MON_C0], hdr[_CP_MON_S] = mon.code_offset, mon.shift
        hdr[_CP_HE_C0], hdr[_CP_HE_S] = hea.code_offset, hea.shift
        names = [c.name for c in lay.channels]
        if "tx.bias.applied" in names:
            bia = lay.channel("tx.bias.applied")
            hdr[_CP_HAS_BI], hdr[_CP_BI_C0], hdr[_CP_BI_S] = 1, bia.code_offset, bia.shift
        temps = [c for c in lay.channels if c.name.startswith("temp.")]
        hdr[_CP_NT] = len(temps)
        dm = {c[0]: c[3] for c in self.action_codec.channels}
        hdr[_CP_DMAX_H] = dm["tx.heater"]
        hdr[_CP_DMAX_B] = dm.get("tx.bias", 0)
        tsec = []
        id_of = {t.channel: TEMP_ID_BASE + i for i, t in enumerate(self.ctrl_spec.temp_sensors)}
        for c in temps:
            tsec += [id_of[c.name], c.code_offset, c.shift]
        packed = mlp.pack()
        user = np.concatenate([hdr, np.asarray(tsec, np.int64), packed])
        lat = self.latency_ticks if latency_ticks is None else int(latency_ticks)
        return CompiledController(name, mlp_policy_init, mlp_policy_step, params_i64=user, latency_ticks=lat)


def fxp_policy_action(mlp: FxpMlp, obs) -> np.ndarray:
    """The env action (float32, (B, A)) of the fixed-point policy ``mlp`` on env observations: y_raw 2^-f_out of the
    MLP on the exact raw inputs obs 2^f_in. Identical, command for command, to :meth:`MRMEnv.compiled_policy` on the
    engine (float32 holds y_raw 2^-f_out exactly for outputs of up to 24 bits)."""
    o = np.asarray(obs, np.float64)
    x_raw = np.rint(o * float(1 << mlp.in_fmt.frac_bits)).astype(np.int64)
    if not np.array_equal(x_raw.astype(np.float64) * 2.0 ** -mlp.in_fmt.frac_bits, o):
        raise ValueError("observations are not exact in the MLP's input format")
    if mlp.out_fmt.bits > 24:
        raise ValueError("outputs wider than 24 bits are not exact in float32")
    y = mlp.forward(x_raw)
    return (y.astype(np.float64) * 2.0 ** -mlp.out_fmt.frac_bits).astype(np.float32)


# ================================================================================================== compiled policy
# user parameter header of the compiled policy (params_i64 of CompiledController)
(_CP_K, _CP_FIN, _CP_MON_C0, _CP_MON_S, _CP_HE_C0, _CP_HE_S, _CP_HAS_BI, _CP_BI_C0, _CP_BI_S, _CP_NT, _CP_DMAX_H,
 _CP_DMAX_B) = range(12)
_CP_HEADER = 12
# p_i64 layout of the compiled policy
(_P_N, _P_K, _P_M, _P_NT, _P_FIN, _P_D, _P_MON_C0, _P_MON_S, _P_HE_C0, _P_HE_S, _P_HAS_BI, _P_BI_C0, _P_BI_S,
 _P_DMAX_H, _P_DMAX_B, _P_FOUT, _P_O_SAMP, _P_O_CNT, _P_O_LAT, _P_O_HEA, _P_O_BIA, _P_A_HE, _P_A_BI, _P_NOUT,
 _P_MW, _P_MLP) = range(26)
_P_TEMP = 26                    # 3 entries per sensor: obs offset, c0, shift; the packed MLP follows (base in _P_MLP)
_HALF_EVEN = int(Rounding.HALF_EVEN)
_SAMPLES_ID, _COUNT_ID, _LATEST_ID = int(ObsId.TX_MONITOR_SAMPLES), int(ObsId.TX_MONITOR_COUNT), \
    int(ObsId.TX_MONITOR_LATEST)
_HEA_ID, _BIA_ID = int(ObsId.TX_HEATER_APPLIED), int(ObsId.TX_BIAS_APPLIED)
_ACT_HE_ID, _ACT_BI_ID = int(ActId.TX_HEATER), int(ActId.TX_BIAS)


@nb.njit(cache=True)
def mlp_policy_init(spec_i64, spec_f64):
    """Compiled-ABI init of :meth:`MRMEnv.compiled_policy` (07 P0-9): offsets of the observed channels, the codec
    of the env's ObsLayout and the packed MLP into p_i64; the history and the MLP buffers into s_i64."""
    n = spec_i64[_api.SPEC_I_N_RINGS]
    m = spec_i64[_api.SPEC_I_M]
    u0 = spec_i64[_api.SPEC_I_USER]
    n_user = spec_i64[_api.SPEC_I_N_USER]
    k = spec_i64[u0 + _CP_K]
    nt = spec_i64[u0 + _CP_NT]
    mlp0 = _CP_HEADER + 3 * nt                       # packed MLP start within the user parameters
    n_mlp = n_user - mlp0
    p = np.zeros(_P_TEMP + 3 * nt + n_mlp, np.int64)
    p[_P_N], p[_P_K], p[_P_M], p[_P_NT], p[_P_FIN] = n, k, m, nt, spec_i64[u0 + _CP_FIN]
    p[_P_MON_C0], p[_P_MON_S] = spec_i64[u0 + _CP_MON_C0], spec_i64[u0 + _CP_MON_S]
    p[_P_HE_C0], p[_P_HE_S] = spec_i64[u0 + _CP_HE_C0], spec_i64[u0 + _CP_HE_S]
    p[_P_HAS_BI], p[_P_BI_C0], p[_P_BI_S] = spec_i64[u0 + _CP_HAS_BI], spec_i64[u0 + _CP_BI_C0], \
        spec_i64[u0 + _CP_BI_S]
    p[_P_DMAX_H], p[_P_DMAX_B] = spec_i64[u0 + _CP_DMAX_H], spec_i64[u0 + _CP_DMAX_B]
    p[_P_O_SAMP] = _api.obs_offset(spec_i64, _SAMPLES_ID)
    p[_P_O_CNT] = _api.obs_offset(spec_i64, _COUNT_ID)
    p[_P_O_LAT] = _api.obs_offset(spec_i64, _LATEST_ID)
    p[_P_O_HEA] = _api.obs_offset(spec_i64, _HEA_ID)
    p[_P_O_BIA] = _api.obs_offset(spec_i64, _BIA_ID)
    p[_P_A_HE] = _api.act_offset(spec_i64, _ACT_HE_ID)
    p[_P_A_BI] = _api.act_offset(spec_i64, _ACT_BI_ID)
    if p[_P_O_SAMP] < 0 or p[_P_O_CNT] < 0 or p[_P_O_LAT] < 0 or p[_P_O_HEA] < 0 or p[_P_A_HE] < 0:
        raise ValueError("mlp policy: the monitor block, the heater register and tx.heater must be exposed")
    if p[_P_HAS_BI] != 0 and p[_P_O_BIA] < 0:
        raise ValueError("mlp policy: tx.bias.applied is in the env layout but not exposed")
    if p[_P_DMAX_B] > 0 and p[_P_A_BI] < 0:
        raise ValueError("mlp policy: the bias action needs tx.bias bound to the controller")
    for i in range(nt):
        p[_P_TEMP + 3 * i] = _api.obs_offset(spec_i64, spec_i64[u0 + _CP_HEADER + 3 * i])
        if p[_P_TEMP + 3 * i] < 0:
            raise ValueError("mlp policy: a temperature sensor of the env layout is not exposed")
        p[_P_TEMP + 3 * i + 1] = spec_i64[u0 + _CP_HEADER + 3 * i + 1]
        p[_P_TEMP + 3 * i + 2] = spec_i64[u0 + _CP_HEADER + 3 * i + 2]
    base = _P_TEMP + 3 * nt
    for j in range(n_mlp):
        p[base + j] = spec_i64[u0 + mlp0 + j]
    if fxp_mlp_check(p, base) != 0:
        raise ValueError("mlp policy: the packed MLP is malformed")
    p[_P_MLP] = base
    d = n * k * m + n + (n if p[_P_HAS_BI] != 0 else 0) + nt
    p[_P_D] = d
    p[_P_NOUT] = p[base + H_N_OUT]
    p[_P_MW] = p[base + H_MAX_WIDTH]
    p[_P_FOUT] = p[base + H_SIZE + (p[base + H_N_LAYERS] - 1) * L_SIZE + L_OUT_FRAC]   # the network's output format
    if p[base + H_N_IN] != d:
        raise ValueError("mlp policy: MLP input size differs from the observation size")
    if p[_P_NOUT] != n + (n if p[_P_DMAX_B] > 0 else 0):
        raise ValueError("mlp policy: MLP output size differs from the action size")
    s = np.zeros(1 + n * k * m + d + p[_P_NOUT] + 2 * p[_P_MW], np.int64)
    return p, np.zeros(0), s, np.zeros(0)


@nb.njit(cache=True)
def mlp_policy_step(p, pf, s, sf, obs, act):
    """Compiled-ABI step of :meth:`MRMEnv.compiled_policy`: update the K-block history, build the env observation as
    raw Q0.f_in integers, evaluate the MLP (fxp_mlp_eval: no allocation, no views), clip each output to
    [-1, 1] and add the decoded deltas to the held commands (the kernel then clips and counts, as for the env)."""
    n = p[_P_N]
    k = p[_P_K]
    m = p[_P_M]
    fin = p[_P_FIN]
    nkm = n * k * m
    h0 = 1
    first = s[0] == 0
    for r in range(n):
        cnt = obs[p[_P_O_CNT] + r]
        lat = obs[p[_P_O_LAT] + r]
        hb = h0 + r * k * m
        if not first:
            for j in range((k - 1) * m):
                s[hb + j] = s[hb + m + j]
        for j in range(m):
            v = obs[p[_P_O_SAMP] + r * m + j] if j < cnt else lat
            if first:
                for kk in range(k):
                    s[hb + kk * m + j] = v
            else:
                s[hb + (k - 1) * m + j] = v
    s[0] = 1
    b0 = h0 + nkm
    d = 0
    for i in range(nkm):
        s[b0 + d] = (s[h0 + i] - p[_P_MON_C0]) << (fin - p[_P_MON_S])
        d += 1
    for r in range(n):
        s[b0 + d] = (obs[p[_P_O_HEA] + r] - p[_P_HE_C0]) << (fin - p[_P_HE_S])
        d += 1
    if p[_P_HAS_BI] != 0:
        for r in range(n):
            s[b0 + d] = (obs[p[_P_O_BIA] + r] - p[_P_BI_C0]) << (fin - p[_P_BI_S])
            d += 1
    for t in range(p[_P_NT]):
        q = _P_TEMP + 3 * t
        s[b0 + d] = (obs[p[q]] - p[q + 1]) << (fin - p[q + 2])
        d += 1
    y0 = b0 + p[_P_D]
    fxp_mlp_eval(p, p[_P_MLP], s, b0, y0, y0 + p[_P_NOUT])
    fout = p[_P_FOUT]
    one = np.int64(1) << fout
    # the env clips each action to [-1, 1] before decoding it (MRMEnv._deltas): the same clip on the raw output
    # (one = 2^f_out, exact); it also keeps y * max_delta inside int64 for every output format (f_out <= 40)
    for r in range(n):
        y = min(max(s[y0 + r], -one), one)
        act[p[_P_A_HE] + r] = act[p[_P_A_HE] + r] + fx_round_shift(y * p[_P_DMAX_H], fout, _HALF_EVEN)
    if p[_P_DMAX_B] > 0:
        for r in range(n):
            y = min(max(s[y0 + n + r], -one), one)
            act[p[_P_A_BI] + r] = act[p[_P_A_BI] + r] + fx_round_shift(y * p[_P_DMAX_B], fout, _HALF_EVEN)


# ================================================================================================== B = 1 view
class MRMSingleEnv(gym.Env):
    """The B = 1 ``gymnasium.Env`` view of :class:`MRMEnv` (same keyword arguments, without num_envs): observation
    and action of one world, scalar reward, unbatched info. After a truncation call reset() (a step after it
    autoresets, as the vector env does)."""

    metadata = {"render_modes": []}

    def __init__(self, config: Any = "g26", **kwargs):
        if "num_envs" in kwargs and int(kwargs.pop("num_envs")) != 1:
            raise ValueError("MRMSingleEnv is one world; use MRMEnv for a batch")
        self.venv = MRMEnv(config, num_envs=1, **kwargs)
        self.observation_space = self.venv.single_observation_space
        self.action_space = self.venv.single_action_space

    def reset(self, *, seed: int | None = None, options: Mapping | None = None):
        super().reset(seed=seed)
        obs, info = self.venv.reset(seed=seed, options=options)
        return obs[0], _unbatch(info)

    def step(self, action):
        obs, r, term, trunc, info = self.venv.step(np.asarray(action)[None, ...])
        return obs[0], float(r[0]), bool(term[0]), bool(trunc[0]), _unbatch(info)

    def close(self):
        self.venv.close()

    @property
    def plant(self):
        return self.venv.plant


def _unbatch(info):
    out = {}
    for k, v in info.items():
        if isinstance(v, dict):
            out[k] = _unbatch(v)
        else:
            out[k] = np.asarray(v)[0].copy()
    return out


def _register() -> None:
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point="mrm.sim.gym_env:MRMSingleEnv",
                     vector_entry_point="mrm.sim.gym_env:MRMEnv", nondeterministic=False)


_register()
