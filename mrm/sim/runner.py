"""Minimal runner: one closed-loop (or open-loop) run of B worlds, and the quasi-static locked-state survey.

Owner: engine (a later agent extends this module with experiments, manifests and the CLI). Contract: 07 P0-6 to
P0-10, P1-17 (``mrm.run(...)``), P1-19 (validation of a Python controller's actions every step).

Public API (stable for the next agents)
---------------------------------------
``run(cfg_or_params, controller=None, scenario=None, duration_s=..., batch=1, seeds=None, record=None, metrics=None,
**options) -> RunResult``

- ``cfg_or_params``: a Config (mrm.params.load), a preset name or YAML path, or a PlantParams together with
  ``cfg=`` (the ControllerSpec is built from the UNSAMPLED config, 07 P1-1).
- ``controller``: None (open loop), a registry name or ``module:Class`` (mrm.control.registry.create with
  ``controller_params``), an object with the Controller protocol (Python slow path, vectorized over B: the kernel
  stops at every controller edge), or a ``mrm.control.api.CompiledController`` (compiled path: ``step`` runs inside
  the kernel; its ``fast_kernel`` runs once per ADC sample).
- ``scenario``: None (the config's ``scenario`` section), a registry name, or a Scenario. Pending aggressors are
  calibrated against the plant (mrm.plant.plant.prepare_scenario).
- ``duration_s``: simulated time; converted once to whole ticks (07 P0-6).
- ``seeds``: int or {device, uncertainty, noise, scenario} base seeds (07 P1-13; world b uses base + b).
- ``record``: list of mrm.sim.metrics.RecordSpec (or (channel, decimation, reducer) tuples) of Truth fields;
  None records nothing (Monte Carlo: metrics only, 07 P0-8).
- ``metrics``: a MetricsSpec or a dict of overrides of the config's ``metrics`` section (e.g.
  ``{"lock_reference": "oracle"}``: lock state and excursions about the oracle's operating point instead of the line;
  mrm.sim.metrics module docstring).
- options: ``substep_policy`` ('fixed' | 'auto', default the config's sim.substep_policy), ``uncertainty``
  ('sample' | 'lhs' | 'nominal'), ``variation`` ('sample' | 'off'), ``reset_protocol``, ``heater_code`` (warm
  start: codes (B, N), ``'quasi_static'`` (the oracle lock codes of :func:`quasi_static` at each world's base
  temperature and bias at the start, without aggressor power) or ``'quasi_static_reachable'`` (the same, with a
  ring that its heater-up order cannot lock moved to the next, bluer order: :func:`warm_start_codes`)),
  ``settle`` (True: start from the thermal steady state of the warm-start codes and the inputs at the start,
  mrm.plant.plant.Plant.settle; needs reset_protocol branch_high or branch_low, and defaults it to branch_high),
  ``factory_table`` (a
  mrm.control.calibration.FactoryCalTable handed to ``controller.load_factory_cal`` before its reset when the
  controller has that method: the warm start of 07 P1-1), ``path`` ('auto' | 'compiled' | 'python'), ``n_threads``,
  ``oracle``, ``histogram``, ``chunk_ticks`` (ticks per kernel call on the compiled and open-loop paths, default
  1e5; without tick jitter every call ends on a controller edge, so it changes no result: ``_block_end``),
  ``controller_params`` (a mapping, passed to mrm.control.registry.create when ``controller`` is a name), ``cfg``,
  ``auto_max_ticks``, ``oracle_period_ticks``, ``score_from_s`` (the metric accumulators restart at this time:
  the start-up of a controller, e.g. a peak map, is left out of the metrics; records, plant and controller are not
  touched, under the auto substep policy only when the split is on a controller edge (else ScoreSplitWarning: the
  split ends a substep) and, for runs longer than metrics.ber_sample_period_s after it, a whole number of BER
  windows after the reset (the restarted BER windows' ends cap substeps); first_lock_s and time_to_lock_s are then
  relative to it), and the engine numerics ``sh_tol_k``
  (self-heating root tolerance, K)
  and ``optics_far_hold`` (the far-line hold of the self-heating optics, a bounded approximation, default off;
  mrm.plant.plant.Plant, docs/handoff/perf.md).
- Truth-fed controllers (``truth_fed = True``: the oracle of mrm.validation.oracle; the registry refuses them) are
  called as ``step(obs, truth=plant.truth())``; every other controller sees only the released sensor channels.

``RunResult``: ``metrics`` (mrm.sim.metrics.summarize: per-ring (B, N) arrays plus world-level values),
``trace`` ({record key: {'t_s', 'value' (S, B, N)}} or None), ``truth_final`` (mrm.plant.plant.Truth),
``manifest_inputs`` (everything a run manifest needs: config fingerprint and hash, seeds, scenario fingerprint and
aggressor calibration, controller name and spec fingerprint, B, ticks, dt, policy, diagnostics), ``diagnostics``,
``wall_s``, ``plant`` (the Plant, for interactive follow-up).

``quasi_static(cfg_or_params, ambient_c=..., bias_v=..., ...) -> QuasiStaticResult`` (07 P0-10
``experiment.kind: quasi_static``): the locked steady state of every ring over ambient 0-105 degC and bias
1.0-3.75 V with self-heating and the static thermal gains, reporting code-range coverage, FSR-hop and wrong-line
risk, and the oracle relOMA map (BND-4).
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from mrm.control.api import (CompiledController, ControllerSpec, Observation, controller_fast_kernel,
                             merge_round_robin)
from mrm.core.channels import OBS_ACTIVE_RING
from mrm.core.timing import seconds_to_ticks
from mrm.plant.plant import Plant, Truth
from mrm.sim import kernel as K
from mrm.sim.metrics import MetricsSpec, RecordSpec

__all__ = ["run", "RunResult", "quasi_static", "QuasiStaticResult", "load_inputs", "warm_start_codes",
           "WARM_START_MODES", "DATA_STATE_KEYS", "reset_data_state", "ScoreSplitWarning"]

#: ``run(..., heater_code=<mode>)`` / :func:`warm_start_codes` modes
WARM_START_MODES = ("quasi_static", "quasi_static_reachable")


@dataclass
class RunResult:
    """Result of :func:`run` (module docstring)."""
    metrics: dict
    trace: dict | None
    truth_final: Truth
    manifest_inputs: dict
    diagnostics: dict = field(default_factory=dict)
    wall_s: float = 0.0
    plant: Any = None

    @property
    def sim_s_per_wall_s(self) -> float:
        t = self.manifest_inputs.get("duration_s", 0.0)
        return t / self.wall_s if self.wall_s > 0 else math.inf


def _seeds(seeds) -> dict:
    out = {"device": 0, "uncertainty": 0, "noise": 0, "scenario": 0}
    if seeds is None:
        return out
    if isinstance(seeds, (int, np.integer)):
        return {k: int(seeds) for k in out}
    out.update({k: int(v) for k, v in dict(seeds).items()})
    return out


def load_inputs(cfg_or_params, *, batch: int = 1, seeds=None, uncertainty: str | None = None,
                variation: str | None = None, cfg=None):
    """(cfg, pp, seeds) from a Config, a preset name / path, or a PlantParams (with cfg=)."""
    from mrm import params
    sd = _seeds(seeds)
    if hasattr(cfg_or_params, "timing") and hasattr(cfg_or_params, "thermal"):
        if cfg is None:
            raise ValueError("pass cfg= with a PlantParams (the ControllerSpec is built from the unsampled config)")
        return cfg, cfg_or_params, sd
    c = params.load(cfg_or_params) if isinstance(cfg_or_params, str) else cfg_or_params

    def setting(key, default):
        try:
            v = c.get(key, None)
        except Exception:                                      # noqa: BLE001
            v = None
        return default if v is None else v
    unc = uncertainty or setting("experiment.uncertainty", "sample")
    var = variation or setting("experiment.variation", "sample")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        pp = params.realize(c, int(batch), seeds={"device": sd["device"], "uncertainty": sd["uncertainty"]},
                            uncertainty=unc, variation=var)
    return c, pp, sd


def _make_controller(controller, controller_params):
    if controller is None or not isinstance(controller, str):
        return controller
    from mrm.control import registry
    return registry.create(controller, dict(controller_params or {}))     # create(name, params: Mapping)


def _make_scenario(scenario, cfg):
    from mrm.scenarios import registry as scen_registry
    from mrm.scenarios.generators import Scenario
    if scenario is None:
        return scen_registry.from_config(cfg)
    if isinstance(scenario, str):
        return scen_registry.create(scenario)
    if not isinstance(scenario, Scenario):
        raise TypeError(f"scenario must be None, a name or a Scenario, got {type(scenario).__name__}")
    return scenario


def run(cfg_or_params, controller=None, scenario=None, duration_s: float = 1e-3, batch: int = 1, seeds=None,
        record=None, metrics=None, **options) -> RunResult:
    """One run of B worlds (module docstring for every argument)."""
    t_wall0 = time.perf_counter()
    cfg, pp, sd = load_inputs(cfg_or_params, batch=batch, seeds=seeds, uncertainty=options.get("uncertainty"),
                              variation=options.get("variation"), cfg=options.get("cfg"))
    B = int(pp.batch)
    dt = float(pp.timing.dt_s)
    n_ticks = seconds_to_ticks(float(duration_s), dt, "duration_s")
    sc = _make_scenario(scenario, cfg)
    ctrl = _make_controller(controller, options.get("controller_params"))
    if isinstance(metrics, MetricsSpec):
        mspec = metrics
    else:
        kw = dict(metrics or {})
        for k in ("oracle_period_ticks",):
            if k in options:
                kw[k] = options[k]
        mspec = MetricsSpec.from_config(cfg, **kw)
    spec = ControllerSpec.from_config(cfg, n_batch=B, nominal=cfg, scenario_drives=sc.drives,
                                      scenario_offsets=sc.offsets)
    lat = int(getattr(ctrl, "latency_ticks", 0) or 0) if ctrl is not None else 0
    fast = controller_fast_kernel(ctrl) if ctrl is not None else None
    path = options.get("path", "auto")
    compiled = isinstance(ctrl, CompiledController) and path in ("auto", "compiled")
    if path == "compiled" and not isinstance(ctrl, CompiledController):
        raise TypeError("path='compiled' needs a CompiledController")
    if fast is not None and not isinstance(ctrl, CompiledController):
        raise TypeError("a fast_kernel needs the CompiledController state arrays (mrm.control.api)")
    plant = Plant(pp, sc, spec, controller_latency_ticks=lat, noise_seed=sd["noise"], scenario_seed=sd["scenario"],
                  metrics=mspec, record=record, record_ticks=n_ticks, substep_policy=options.get("substep_policy"),
                  auto_max_ticks=int(options.get("auto_max_ticks", 64)), fast_kernel=fast is not None,
                  oracle=options.get("oracle", True), histogram=options.get("histogram"),
                  **{k: options[k] for k in ("sh_tol_k", "optics_far_hold") if k in options})
    settle = bool(options.get("settle", False))
    protocol = options.get("reset_protocol") or ("branch_high" if settle else None)
    heater_code = options.get("heater_code")
    no_code = supply_limited = hopped = None
    warm_mode = heater_code if isinstance(heater_code, str) else None
    if isinstance(heater_code, str):
        heater_code, no_code, supply_limited, hopped = _warm_start_codes(plant, cfg, heater_code)
    plant.reset(protocol=protocol, heater_code=heater_code, settle=settle)
    n_threads = options.get("n_threads")
    plant.n_threads = n_threads
    if ctrl is not None:
        table = options.get("factory_table")
        if table is not None:
            if not hasattr(ctrl, "load_factory_cal"):
                raise TypeError(f"factory_table given but {type(ctrl).__name__} has no load_factory_cal")
            ctrl.load_factory_cal(table)
        ctrl.reset(spec)
    segments, score_tick = _score_segments(plant, n_ticks, dt, options.get("score_from_s"),
                                           python_path=ctrl is not None and not compiled)
    chunk = int(options.get("chunk_ticks", DEFAULT_CHUNK_TICKS))
    if chunk < 1:
        raise ValueError(f"chunk_ticks must be >= 1, got {chunk}")
    if ctrl is not None and not compiled and spec.tick_jitter_ticks > 0:
        raise NotImplementedError("the Python controller path needs edges shared by every world (no tick "
                                  "jitter); use a CompiledController")
    if ctrl is not None and compiled:
        plant.bind_compiled(ctrl)
    for i_seg, n_seg in enumerate(segments):
        if i_seg == 1:
            # score from here: the metric accumulators restart at this tick; the plant, controller and records go
            # on untouched, and the oracle lock reference and the previous controller latch (plant history) are
            # kept (mrm.sim.metrics.MetricsState.restart)
            plant.metrics_state.restart(plant.tick, np.ascontiguousarray(plant.S.ri[:, K.RI_HE_REG, :]))
        if ctrl is None:
            _run_open(plant, n_seg, n_threads, chunk=chunk)
            ctrl_name = None
        elif compiled:
            k_end = plant.tick + n_seg
            while plant.tick < k_end:
                plant.run(_block_end(plant, k_end, chunk) - plant.tick, mode=K.MODE_COMPILED, step_fn=ctrl.step_fn,
                          fast_fn=fast)
                if spec.on_out_of_range == "error":
                    _raise_if_clipped(plant, spec)
            ctrl_name = ctrl.name
        else:
            _run_python(plant, ctrl, spec, n_seg, fast, bind=i_seg == 0)
            ctrl_name = getattr(ctrl, "name", type(ctrl).__name__)
    wall = time.perf_counter() - t_wall0
    m = plant.metrics()
    diag = plant.diagnostics()
    if not compiled and ctrl is not None:
        diag["saturation_counts_python"] = plant.validator.saturation_counts()
    trace = plant.traces() if record else None
    mi = {"config_fingerprint": cfg.fingerprint() if hasattr(cfg, "fingerprint") else None,
          "config_hash": getattr(pp, "config_hash", None), "seeds": sd, "batch": B, "n_rings": int(pp.n_rings),
          "n_lines": int(pp.n_lines), "dt_s": dt, "n_ticks": int(n_ticks), "duration_s": n_ticks * dt,
          "substep_policy": plant.substep_policy, "scenario": sc.name,
          "scenario_fingerprint": sc.fingerprint(), "time_compression": float(sc.time_compression),
          "aggressor_calibration": plant.aggressor_calibration, "controller": ctrl_name,
          "controller_path": None if ctrl is None else ("compiled" if compiled else "python"),
          "controller_latency_ticks": lat, "spec_fingerprint": spec.fingerprint(),
          "metrics_settings": m.get("settings"), "world": getattr(pp, "world", None),
          "world_choices": getattr(pp, "world_choices", None),
          "records": [r.key for r in plant.records_state.specs], "diagnostics": diag, "chunk_ticks": chunk,
          "warm_start": {"heater_code": None if heater_code is None else np.asarray(heater_code).tolist(),
                         "reset_protocol": protocol, "settle": plant.settle_info,
                         "factory_table": options.get("factory_table") is not None,
                         "no_code": None if no_code is None else no_code.tolist(),
                         "supply_limited": None if supply_limited is None else supply_limited.tolist(),
                         "mode": warm_mode,
                         "order_hopped": None if hopped is None else hopped.tolist()}}
    if sc.time_compression != 1.0:
        mi["compression_note"] = ("compressed result: report it only after mrm.scenarios."
                                  "require_compression_validated (07 P0-10)")
    if score_tick is not None:
        mi["score_from_tick"] = int(score_tick)
        mi["score_from_s"] = float(score_tick * dt)
    return RunResult(metrics=m, trace=trace, truth_final=plant.truth(), manifest_inputs=mi, diagnostics=diag,
                     wall_s=wall, plant=plant)


def warm_start_codes(plant: Plant, cfg, *, mode: str = "quasi_static", return_no_code: bool = False
                     ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Warm-start heater codes (B, N) int64 for ``run(..., heater_code=mode)``.

    ``'quasi_static'``: the oracle lock codes of :func:`quasi_static` (best stable relOMA point of each ring on its
    declared line, 07 P1-1) at each world's base temperature and mean bias voltage at the plant's current (reset)
    tick, each ring on its heater-up order. ``'quasi_static_reachable'``: the same, but a ring whose heater-up order
    would need a negative heater power locks one order bluer (quasi_static ``order='reachable'``; repeated while the
    hopped rings' heat pushes other rings below zero) instead of starting at code_min. The data path of the scenario
    at the reset tick is included (quasi_static ``data_state``, :func:`reset_data_state`): a start with traffic off,
    squelched or idle gets the codes that hold the traffic-on operating detuning in that state (more heater power
    without the data heat). Not included: aggressor power, laser power or wavelength offsets of the scenario at the
    start, and the bus filtering by rings that are not locked (quasi_static assumes the whole bank locked). Use
    with ``settle=True`` for a start that is a steady lock. A ring without a finite lock code (sampled worlds whose
    oracle point the heater cannot reach) starts at code_min, with a warning that counts them. A world whose lock
    powers exceed the array heater supply limit (quasi_static ``supply_limited``, 07 P1-12) keeps its codes, with a
    warning: the plant droops every heater of that world, so its rings settle below their locks (on the cold
    self-heating branch in the stacked_eic worlds of g26), and no controller can hold the whole bank there.
    ``return_no_code=True`` returns ``(codes, no_code)``, no_code (B, N) bool marking the rings without a code (the
    same mask as ``manifest_inputs["warm_start"]["no_code"]`` of a run; ``["supply_limited"]`` (B,) has the worlds).
    """
    codes, no_code, _, _ = _warm_start_codes(plant, cfg, mode, stacklevel=3)
    return (codes, no_code) if return_no_code else codes


def reset_data_state(plant: Plant) -> dict:
    """The data path of the scenario at the plant's reset tick as quasi_static's ``data_state`` ((B, N) arrays of
    driver_enable, swing_scale, transition_density, ones_density; the values the reset filled into S.sc_v0)."""
    lay = plant.scen_arrays.layout
    v0 = plant.S.sc_v0
    out = {}
    for k in DATA_STATE_KEYS:
        o, w = lay[f"data.{k}"]
        out[k] = np.array(v0[:, o:o + w], float)
    return out


def _warm_start_codes(plant: Plant, cfg, mode: str, stacklevel: int = 2
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(codes (B, N) int64, no_code (B, N) bool, supply_limited (B,) bool, order_hopped (B, N) bool):
    :func:`warm_start_codes`, the rings that had no quasi-static lock code (started at code_min), the worlds whose
    lock powers exceed the array heater supply limit at the start, and the rings locked one order bluer than their
    heater-up order (mode 'quasi_static_reachable' only)."""
    if mode not in WARM_START_MODES:
        raise ValueError(f"heater_code mode must be one of {WARM_START_MODES}, got {mode!r}")
    pp = plant.pp
    B = int(pp.batch)
    t_base_c = plant.S.wf[:, K.WF_T_BASE0] - 273.15
    bias = plant.S.rf[:, K.RF_BI_VT, :].mean(axis=1)
    amb_u, a_idx = np.unique(np.round(t_base_c, 6), return_inverse=True)
    v_u, v_idx = np.unique(np.round(bias, 6), return_inverse=True)
    qs = quasi_static(pp, cfg=cfg, ambient_c=amb_u, bias_v=v_u,
                      order="reachable" if mode == "quasi_static_reachable" else "heater_up",
                      data_state=reset_data_state(plant))
    codes = np.stack([qs.heater_code[a_idx[b], v_idx[b], b] for b in range(B)])
    sup = np.array([bool(qs.supply_limited[a_idx[b], v_idx[b], b]) for b in range(B)])
    hopped = np.stack([qs.order_hopped[a_idx[b], v_idx[b], b] for b in range(B)])
    if np.any(sup):
        sc_ = [round(float(qs.supply_scale[a_idx[b], v_idx[b], b]), 4) for b in np.nonzero(sup)[0]]
        warnings.warn(f"warm_start_codes: {int(sup.sum())} of {B} worlds (worlds {np.nonzero(sup)[0].tolist()}, "
                      f"droop factors {sc_}) exceed the array heater supply limit at their lock codes (07 P1-12): "
                      f"the plant droops every heater, so those rings cannot hold their locks", stacklevel=stacklevel)
    bad = ~np.isfinite(codes)
    if np.any(bad):
        warnings.warn(f"warm_start_codes: {int(bad.sum())} of {bad.size} rings have no quasi-static lock code "
                      f"(worlds {sorted(set(np.nonzero(bad)[0].tolist()))}); they start at code_min",
                      stacklevel=stacklevel)
        codes = np.where(bad, float(plant.heater_params.code_lo), codes)
    return np.rint(codes).astype(np.int64), bad, sup, hopped


#: ticks per kernel call of the compiled and open-loop paths (``run(..., chunk_ticks=)``)
DEFAULT_CHUNK_TICKS = 100_000


class ScoreSplitWarning(UserWarning):
    """``score_from_s`` under the auto substep policy at a tick that is not a controller edge: the scoring split ends
    a substep there, so the plant trajectory is not the unscored run's (docs/ENGINE.md section 2)."""


def _block_end(plant: Plant, k_end: int, chunk: int) -> int:
    """End tick of the next kernel block from plant.tick: plant.tick + chunk moved forward to the next controller edge,
    at most k_end. A substep of the auto policy never crosses a controller edge, so a block that ends on one leaves
    the policy's step sequence (and every result) unchanged, whatever chunk_ticks is; a block end elsewhere caps the
    substep in flight and the growth limit restarts from the shortened step (engine review 2026-10-16: chunk_ticks
    13 against 1e5 gave 11426 against 10582 substeps). With tick jitter (07 P2-7) the edges differ per world and the
    end stays plant.tick + chunk: there, under the auto policy, chunk_ticks changes the step sequence."""
    end = plant.tick + chunk
    if end >= k_end:
        return k_end
    P = plant.P
    per, ph0 = int(P.period), int(P.phase)
    if int(P.jitter) == 0 and per > 0:
        end = ph0 if end <= ph0 else ph0 + -(-(end - ph0) // per) * per
    return min(end, k_end)


def _raise_if_clipped(plant: Plant, spec) -> None:
    """``controller_interface.on_out_of_range: error`` on the compiled path (07 P1-19): the kernel clips every
    command and records each world's first out-of-range element (kernel.WI_SAT_FIRST_*: tick, act_i64 index,
    commanded value); raise :class:`mrm.control.api.ActionRangeError` for the earliest, as the Python path's
    ActionValidator does at that step. The kernel finishes its block first (up to chunk_ticks), so the run's
    other outputs are not meaningful."""
    from mrm.control.api import ActionRangeError
    wi = plant.S.wi
    first = wi[:, K.WI_SAT_FIRST_TICK]
    hit = np.flatnonzero(first >= 0)
    if hit.size == 0:
        return
    b = int(hit[np.argmin(first[hit])])
    elem, val, tick = int(wi[b, K.WI_SAT_FIRST_ELEM]), int(wi[b, K.WI_SAT_FIRST_VAL]), int(first[b])
    f = next(f for f in spec.layout.act if f.offset <= elem < f.offset + f.size)
    ch = spec.channels[f.name]
    idx = [b, *(int(i) for i in np.unravel_index(elem - f.offset, f.shape))] if f.shape else [b]
    raise ActionRangeError(f"{f.name}{idx} = {val} outside [{ch.min}, {ch.max}] ({ch.unit}) at tick {tick} "
                           f"(compiled path: the first out-of-range command; {int(hit.size)} of {wi.shape[0]} worlds "
                           f"had one); set controller_interface.on_out_of_range: clip to saturate instead")


def _run_open(plant: Plant, n_ticks: int, n_threads=None, chunk: int = DEFAULT_CHUNK_TICKS) -> None:
    k_end = plant.tick + n_ticks
    while plant.tick < k_end:
        plant.run(_block_end(plant, k_end, chunk) - plant.tick, mode=K.MODE_OPEN)


def _score_segments(plant: Plant, n_ticks: int, dt: float, score_from_s, *, python_path: bool
                    ) -> tuple[list[int], int | None]:
    """Split a run at the scoring start (option ``score_from_s``): ([n_ticks], None) without it, else
    ([n0, n_ticks - n0], n0) with n0 the whole ticks of score_from_s after the reset tick. On the Python controller
    path a split on a controller edge moves one tick later (the Python loop must not stop on an edge between two
    segments)."""
    if score_from_s is None or float(score_from_s) == 0.0:
        return [n_ticks], None
    n0 = seconds_to_ticks(float(score_from_s), dt, "score_from_s")
    if python_path:
        per, ph = int(plant.P.period), int(plant.P.phase)
        k = plant.tick + n0
        if per > 0 and k >= ph and (k - ph) % per == 0:
            n0 += 1
    if not 0 < n0 < n_ticks:
        raise ValueError(f"score_from_s = {score_from_s} s must lie inside the run (0, {n_ticks * dt} s)")
    P = plant.P
    if int(P.policy) == K.POLICY_AUTO:
        # the split ends a kernel block, which caps the auto substep in flight unless the tick is a controller edge
        # (a substep never crosses one); with tick jitter the edges differ per world
        per, ph = int(P.period), int(P.phase)
        k = plant.tick + n0
        on_edge = int(P.jitter) == 0 and per > 0 and k >= ph and (k - ph) % per == 0
        if not on_edge:
            why = ("the controller edges differ per world (tick jitter)" if int(P.jitter) else
                   f"tick {k} is not a controller edge (phase {ph} + j x {per} ticks)"
                   + (", the Python controller path moved it one tick past one" if python_path else ""))
            warnings.warn(f"score_from_s = {score_from_s} s under the auto substep policy: {why}; the scoring split "
                          "ends a substep there, so the plant trajectory differs from the unscored run's (another "
                          "valid step sequence of the policy). Put the split on a controller edge to keep it "
                          "(docs/ENGINE.md section 2)", ScoreSplitWarning, stacklevel=3)
    return [n0, n_ticks - n0], n0


def _run_python(plant: Plant, ctrl, spec, n_ticks: int, fast, bind: bool = True) -> None:
    """Python slow path, vectorized over B (07 P0-9): the kernel stops at every controller edge; the controller
    steps on an Observation of the released channels (with ``active``, the ring of the edge under
    controller_interface.schedule round_robin, 07 P2-7: the kernel then applies only that ring's per-ring commands
    when it resumes); ActionValidator checks shape, dtype and range (07 P1-19); the command enters the controller
    pipeline when the kernel resumes. ``bind`` False continues a run whose compiled controller is already bound (the
    second segment of a split run)."""
    k_end = plant.tick + n_ticks
    resume = False
    truth_fed = bool(getattr(ctrl, "truth_fed", False))
    held = getattr(ctrl, "act", None) if isinstance(ctrl, CompiledController) else None
    if isinstance(ctrl, CompiledController) and bind:
        plant.bind_compiled(ctrl)
    nact = plant.maps.act_size
    while plant.tick < k_end:
        plant.run(k_end - plant.tick, mode=K.MODE_PYTHON, resume=resume,
                  step_fn=None, fast_fn=fast)
        resume = False
        if plant.tick >= k_end:
            break
        if int(plant.S.wi[0, K.WI_PENDING_EDGE]) != plant.tick:
            continue
        ar = int(plant.S.obs[0, OBS_ACTIVE_RING])            # edges are shared by every world on this path
        held_edge = plant.S.act[:, :nact].copy() if ar >= 0 else None
        if np.any(plant.S.wi[:, K.WI_CTRL_RESET]):           # scenario fault.controller_reset
            ctrl.reset(spec)
            plant.S.wi[:, K.WI_CTRL_RESET] = 0
            if held is not None:
                # CompiledController.reset allocates new state and held-command arrays: the kernel's arrays take
                # their values and the controller uses those again (its fast_kernel runs on them in the kernel), the
                # held command re-filled from the latched registers, as the kernel's own restart (controllers review
                # 2026-10-16; before, the fast kernel kept updating the old state); clip the held command from now on
                plant.restart_compiled(ctrl)
                held = getattr(ctrl, "act", None)
        sensors, new = plant.sample_sensors()
        active = None
        if ar >= 0:
            active = np.zeros(spec.n_rings, dtype=bool)
            active[ar] = True
        obs = Observation(plant.tick, plant.dt, sensors, new, active)
        actions = ctrl.step(obs, truth=plant.truth()) if truth_fed else ctrl.step(obs)
        if ar >= 0:
            # round robin (07 P2-7): only the active ring's per-ring commands apply (the kernel enforces it again
            # when it resumes); validate, clip and count only those, as the kernel does
            actions = merge_round_robin(spec, actions, held_edge, ar)
        val = plant.validator.validate(actions)
        spec.layout.pack_act(val, plant.S.act[:, :nact])
        if held is not None and held is not plant.S.act:
            held[...] = plant.S.act[:, :nact]
        resume = True


# ================================================================================================ quasi-static
@dataclass
class QuasiStaticResult:
    """07 P0-10 ``experiment.kind: quasi_static`` survey over (ambient, bias), per world and ring.

    Arrays have shape (A, V, B, N) for A ambient points and V bias points: detuning_target_m (the oracle lock
    detuning, mean resonance minus line), reloma_db / reloma_unrestricted_db (oracle relOMA over open-loop-stable
    points and without that restriction, D9: OMA/P_in), t_ring_k, p_abs_w, p_heater_w, heater_code (nan when no
    code in [code_min, code_max] reaches the lock), code_ok, fsr_hop_risk (the lock needs a negative heater power:
    the ring is already red of its target at code_min, so a controller must hop one order), wrong_line_risk (number
    of foreign lines, over every order, that the resonance crosses on a heater-up sweep from code_min to the lock;
    each is a possible capture, 07 P1-12), order_hopped (rings that lock one order bluer than the heater-up order,
    ``order='reachable'`` only). Per (ambient, bias, world): supply_scale (A, V, B), the droop factor s of
    the array heater supply limit (07 P1-12; mrm.plant.electrical.supply_limit_row: s = min(1, sqrt(P_max / sum P),
    I_max / sum I) over the rings with a code, evaluated at their lock codes and static heater-resistor temperatures),
    and supply_limited = s < 1: the bank's lock powers exceed the supply, so the plant droops every heater and no ring
    of that world holds its lock there (code_ok is then False for every ring; heater_code keeps the codes). Summaries:
    code_range_coverage (fraction of (ambient, bias) points with every ring of the world reachable within the supply
    limit), bnd4_reloma_variation_db (V, B): max - min over ambient of the bank-mean oracle relOMA at each bias (BND-4
    asks < 1 dB).
    """
    ambient_c: np.ndarray
    bias_v: np.ndarray
    detuning_target_m: np.ndarray
    reloma_db: np.ndarray
    reloma_unrestricted_db: np.ndarray
    t_ring_k: np.ndarray
    p_abs_w: np.ndarray
    p_heater_w: np.ndarray
    heater_code: np.ndarray
    code_ok: np.ndarray
    fsr_hop_risk: np.ndarray
    wrong_line_risk: np.ndarray
    code_range_coverage: np.ndarray
    bnd4_reloma_variation_db: np.ndarray
    notes: tuple = ()
    supply_scale: np.ndarray | None = None
    supply_limited: np.ndarray | None = None
    order_hopped: np.ndarray | None = None


def quasi_static(cfg_or_params, *, ambient_c=None, bias_v=None, batch: int = 1, seeds=None, p_in_w=None,
                 n_scan: int = 301, iterations: int = 4, cfg=None, uncertainty=None, variation=None,
                 order: str = "heater_up", data_state: Mapping[str, Any] | None = None) -> QuasiStaticResult:
    """Locked steady state of every ring over ambient and bias (07 P0-10; BND-4), static and with self-heating.

    For each (ambient, bias): the junction levels follow the bias DAC transfer and the driver swing
    (mrm.plant.electrical.nrz_levels_row); the oracle lock detuning of each ring maximizes relOMA = |T1 - T0| at its
    own through port for its line among points with static self-heating loop gain G dP_abs/dlambda < 1
    (G = s(T) Z_opt,self, 07 P0-1 item 7); the ring temperature that puts the resonance there follows from the R1
    thermo-optic law and the junction shift (mrm.plant.phase); the static thermal map (DC gains of the thermal
    system: heater, absorbed, junction and driver heat into every ring and heater resistor, base temperature) then
    gives the heater powers of all rings at once (a linear solve), and the driver law with the heater TCR at the
    static heater-resistor temperature gives the code. Loss and coupling are evaluated at the solved temperatures and
    the solve is repeated ``iterations`` times. p_in_w: power per line at the chip (default the plant's own; BND-4
    is stated at 1 mW into the MRM: pass 1e-3 / input transmission; the rings after the first in bus order then get
    the bus segment loss on top, up to 0.6 % less in g26). Light reaching each ring: the chip power
    times the through transmission of the rings before it at their lock points (single pass) and the bus segment
    loss between rings (``mrm.plant.phase.bus_t2_from_plant_params``, as the plant). The bank's heater
    powers at the lock codes are checked against the array heater supply limit (``supply_scale``,
    ``supply_limited``; 07 P1-12).

    ``order``: which resonance order of each ring locks to its line. 'heater_up' (default): the order whose
    resonance sits blue of (or on) its line at code_min, the first one a heater-up acquisition reaches; a ring that
    is already red of its lock point there needs a negative heater power (``fsr_hop_risk``) and gets no code.
    'reachable': such a ring takes the next order (one FSR bluer, reached by heating) and the bank is solved again,
    until no ring needs a negative heater power (at most N + 1 solves): a hopped ring adds about one FSR of heater
    power, whose crosstalk can push a neighbour that had a heater-up code below zero, and that neighbour hops too
    (engine review 2026-10-16: before, one re-solve left such rings without a code). ``order_hopped`` (A, V, B, N)
    marks every hopped ring and ``fsr_hop_risk`` still reports the heater-up order's need (the first solve).

    ``data_state``: the data path the bank holds its lock in, a mapping of ``driver_enable``, ``swing_scale``,
    ``transition_density`` and ``ones_density`` (each broadcast to (B, N); the scenario signals ``data.*``, as the
    kernel reads them: enable >= 0.5 is on, the density clipped to [0, 1]). Default (None, or keys left out): traffic
    on, i.e. driver on, the nominal swing, the config's transition density and ones density 0.5. The lock TARGET is
    always the traffic-on operating point (the oracle detuning above, with the nominal swing); with a world whose
    state differs the heat balance is solved a second time in that state, at the same target detuning: junction
    levels with the scaled swing, absorbed power and through transmission averaged with its ones density, and its
    data-activity heat (07 P1-8). heater_code, p_heater_w, p_abs_w and t_ring_k are then those of the given state;
    detuning_target_m and the relOMA maps stay the traffic-on ones. ``warm_start_codes`` passes the state of the
    scenario at the reset tick (engine review 2026-10-16: a start with traffic off began 65-98 pm blue of the lock).
    """
    from mrm.plant import electrical as el
    from mrm.plant import optics as op
    from mrm.plant import phase as ph
    from mrm.plant import thermal as th
    from mrm.units import C0
    cfg_, pp, sd = load_inputs(cfg_or_params, batch=batch, seeds=seeds, uncertainty=uncertainty,
                               variation=variation, cfg=cfg)
    B, N, L = int(pp.batch), int(pp.n_rings), int(pp.n_lines)
    amb = np.linspace(0.0, 105.0, 22) if ambient_c is None else np.atleast_1d(np.asarray(ambient_c, float))
    bv = np.linspace(1.0, 3.75, 12) if bias_v is None else np.atleast_1d(np.asarray(bias_v, float))
    A, V = amb.size, bv.size
    sysm = pp.thermal.system
    phs = ph.PhaseParams.from_plant_params(pp)
    hp = el.HeaterDriverParams.from_plant_params(pp)
    G = th.dc_gain(sysm)                                     # (B, O, S) K/W
    ring_obs, heater_obs = sysm.ring_obs, sysm.heater_obs
    z_self = th.selfheat_dc_k_per_w(sysm)
    lam_l = C0 / np.asarray(pp.laser.line_freq_hz, float)    # (B, L)
    r2l = np.asarray(pp.layout.ring_to_line)
    bus = np.asarray(pp.layout.bus_order)
    p_chip = (np.asarray(pp.laser.power_w, float) * np.asarray(pp.laser.input_transmission, float).reshape(-1, 1)
              if p_in_w is None else np.broadcast_to(np.asarray(p_in_w, float), (B, L)).copy())
    bus_t2 = ph.bus_t2_from_plant_params(pp)                 # (B, N) bus segment before each ring
    swing = np.nan_to_num(np.asarray(pp.datapath.swing_pp_v, float)) * np.ones((B, N))
    level = np.asarray(pp.datapath.level_convention).reshape(-1)
    fwhm_hz = np.asarray(pp.optics.fwhm_hz, float)
    td_cfg = np.array([float(np.nan_to_num(pp.datapath.transition_density[b, 0], nan=0.5)) for b in range(B)])
    ds = _data_state(data_state, B, N, td_cfg)
    shp = (A, V, B, N)
    out = {k: np.full(shp, np.nan) for k in ("det", "rel", "relu", "t_ring", "p_abs", "p_h", "code")}
    code_ok = np.zeros(shp, bool)
    hop = np.zeros(shp, bool)
    supply_s = np.ones((A, V, B))
    hopped_all = np.zeros(shp, bool)
    if order not in ("heater_up", "reachable"):
        raise ValueError(f"quasi_static order must be 'heater_up' or 'reachable', got {order!r}")
    reachable = order == "reachable"
    p_sup_max = np.broadcast_to(np.asarray(hp.supply_power_max_w, float).reshape(-1), (B,))
    i_sup_max = np.broadcast_to(np.asarray(hp.supply_current_max_a, float).reshape(-1), (B,))
    wrong = np.zeros(shp, np.int64)
    x = np.linspace(-1.5, 1.5, int(n_scan))
    rows = []
    for b in range(B):
        r = phs.table_row(b)
        rows.append((phs.ring[b], phs.grid, phs.dneff[r], phs.alpha_fca[r], phs.capacitance[r], phs.r_series[r]))
    pd_src = np.asarray(sysm.driver_src)
    pj_src = np.asarray(sysm.junction_src)
    # the optics model the plant runs (exact ring or TCMT, backscatter when on)
    if int(pp.flags.optics_model):
        m_ord = int(pp.optics.tcmt_orders)

        def ring_fn(*a, rb=None):
            return op.ring_tcmt(*a, rb=rb, m_orders=m_ord)
    else:
        ring_fn = op.ring_exact
    rb_all = np.asarray(pp.optics.backscatter_rb)

    def rb_of(b, n):
        r = rb_all[b if rb_all.shape[0] > 1 else 0, n]
        return r if np.any(r != 0) else None
    for ia, ta_c in enumerate(amb):
        t_amb = ta_c + 273.15
        for iv, vb in enumerate(bv):
            for b in range(B):
                rp, grid, dn, al, cc, rs = rows[b]
                vlo, vhi, fl = np.zeros(N), np.zeros(N), np.zeros(N, np.int8)
                el.nrz_levels_row(np.full(N, vb), swing[b], int(level[b]), el.JUNCTION_GRID_V_MIN,
                                  el.JUNCTION_GRID_V_MAX, vlo, vhi, fl)
                t_ring = np.full(N, t_amb)
                t_heat = np.full(N, t_amb)
                # the heater-up order of every ring: the order whose resonance sits blue of (or on) its line at code
                # code_min (the first order a heater-up acquisition reaches, 07 P0-1); kept for every iteration
                o = {k: np.zeros((N, L)) for k in ("phi0", "phi1", "a0", "a1", "e0", "e1", "k2t", "k2d", "trt")}
                shift, cj, rsv, jf = np.zeros((N, 2)), np.zeros((N, 2)), np.zeros((N, 2)), np.zeros(N, np.int8)
                ph.ring_optics_row(rp, np.ascontiguousarray(lam_l[b]), t_ring, t_heat, vlo, vhi, grid, dn, al, cc,
                                   rs, o["phi0"], o["phi1"], o["a0"], o["a1"], o["e0"], o["e1"], o["k2t"], o["k2d"],
                                   o["trt"], shift, cj, rsv, jf)
                m_up = np.zeros(N, np.int64)
                lam_cold = np.zeros(N)
                for n in range(N):
                    l = int(r2l[n])
                    mean = 0.5 * (o["phi0"][n, l] + o["phi1"][n, l])
                    m_up[n] = math.ceil(mean / (2 * math.pi))
                    # exact order-m_up resonance of the cold ring (mean of the two level shifts)
                    lam_cold[n] = lam_l[b, l] + K.resonance_detuning(rp[n], lam_l[b, l],
                                                                      0.5 * (shift[n, 0] + shift[n, 1]),
                                                                      float(m_up[n]))
                # 'reachable': a ring whose lock at the heater-up order needs a negative heater power (fsr_hop_risk:
                # its cold resonance already sits red of the lock point) takes the next order (one FSR bluer, reached
                # by heating) and the whole bank is solved again (integrator 2026-10-10, experiments request 10.4b)
                # Every ring that the re-solve pushes below zero (the hopped rings' extra heater power reaches it through
                # the thermal crosstalk) hops too, until no ring needs a negative power: heat only adds, so this is the
                # smallest consistent set of hops (engine review 2026-10-16; at most N + 1 solves)
                shift_cold = shift.copy()
                hopped = np.zeros(N, bool)
                hop0 = None
                for attempt in range(N + 1 if reachable else 1):
                    if attempt >= 1:
                        neg = p_h < 0
                        if not np.any(neg):
                            break
                        hopped |= neg
                        m_up[neg] += 1
                        for n in np.nonzero(neg)[0]:
                            l = int(r2l[n])
                            lam_cold[n] = lam_l[b, l] + K.resonance_detuning(
                                rp[n], lam_l[b, l], 0.5 * (shift_cold[n, 0] + shift_cold[n, 1]), float(m_up[n]))
                        t_ring = np.full(N, t_amb)
                        t_heat = np.full(N, t_amb)
                    det = np.zeros(N)
                    rel = np.zeros(N)
                    relu = np.zeros(N)
                    p_abs = np.zeros(N)
                    ph_at = np.zeros(N)                      # mean phase of each ring's optimum (this iteration)
                    ph_lock = np.zeros(N)                    # ... of the last iteration (the target of det)
                    for _ in range(max(1, int(iterations))):
                        o = {k: np.zeros((N, L)) for k in ("phi0", "phi1", "a0", "a1", "e0", "e1", "k2t", "k2d", "trt")}
                        shift, cj, rsv, jf = np.zeros((N, 2)), np.zeros((N, 2)), np.zeros((N, 2)), np.zeros(N, np.int8)
                        ph.ring_optics_row(rp, np.ascontiguousarray(lam_l[b]), t_ring, t_heat, vlo, vhi, grid, dn, al,
                                           cc, rs, o["phi0"], o["phi1"], o["a0"], o["a1"], o["e0"], o["e1"], o["k2t"],
                                           o["k2d"], o["trt"], shift, cj, rsv, jf)
                        # d phi / d lambda_res at the shifted coordinate x = lambda_line - shift (mrm.sim.kernel
                        # dphi_dlamres): 2 pi c t_rt / x^2
                        dpl = 2.0 * math.pi * C0 * o["trt"] / (lam_l[b][None, :] - shift[:, 0:1]) ** 2
                        p_ring = p_chip[b].copy()
                        t_target = np.zeros(N)
                        for q in range(N):
                            n = int(bus[q])
                            l = int(r2l[n])
                            p_ring *= bus_t2[b, n]                   # the bus segment before this ring
                            fwhm = lam_l[b, l] ** 2 * fwhm_hz[b, n] / C0
                            dphi = o["phi1"][n, l] - o["phi0"][n, l]
                            mean = 0.5 * (o["phi0"][n, l] + o["phi1"][n, l])
                            m = int(m_up[n])
                            base = 2 * math.pi * m + dpl[n, l] * x * fwhm
                            r0 = ring_fn(base - 0.5 * dphi, o["k2t"][n, l], o["k2d"][n, l], o["a0"][n, l],
                                         o["trt"][n, l], rb=rb_of(b, n))
                            r1 = ring_fn(base + 0.5 * dphi, o["k2t"][n, l], o["k2d"][n, l], o["a1"][n, l],
                                         o["trt"][n, l], rb=rb_of(b, n))
                            relx = np.abs(r1.p_thru - r0.p_thru)
                            slope = p_ring[l] * 0.5 * (o["e0"][n, l] * r0.dloss_dphi
                                                       + o["e1"][n, l] * r1.dloss_dphi) * dpl[n, l]
                            s_t = rp[n, ph.RP_S0] * (1 + rp[n, ph.RP_BETA] * (t_ring[n] - rp[n, ph.RP_T0]))
                            stable = z_self[b, n] * s_t * slope < 1.0
                            i_u = int(np.argmax(relx))
                            i_s = int(np.argmax(np.where(stable, relx, -1.0)))
                            relu[n] = relx[i_u]
                            rel[n] = relx[i_s] if stable[i_s] else 0.0
                            # refine the optimum by the parabola through it and its stable neighbours (the grid step is
                            # 0.01 FWHM, ~3 pm), and evaluate the ring there
                            xs = x[i_s]
                            if 0 < i_s < x.size - 1 and stable[i_s - 1] and stable[i_s] and stable[i_s + 1]:
                                den = relx[i_s - 1] - 2.0 * relx[i_s] + relx[i_s + 1]
                                if den < 0.0:
                                    xs = xs + min(max(0.5 * (relx[i_s - 1] - relx[i_s + 1]) / den, -0.5), 0.5) * (
                                        x[1] - x[0])
                            ph_s = 2 * math.pi * m + dpl[n, l] * xs * fwhm
                            ph_at[n] = ph_s
                            q0 = ring_fn(np.array([ph_s - 0.5 * dphi]), o["k2t"][n, l], o["k2d"][n, l], o["a0"][n, l],
                                         o["trt"][n, l], rb=rb_of(b, n))
                            q1 = ring_fn(np.array([ph_s + 0.5 * dphi]), o["k2t"][n, l], o["k2d"][n, l], o["a1"][n, l],
                                         o["trt"][n, l], rb=rb_of(b, n))
                            # the optimum's exact detuning (order-m resonance minus line, mrm.sim.kernel
                            # cold_coordinate): x_m - x(phase / 2 pi)
                            x_m = K.cold_coordinate(rp[n], lam_l[b, l] - 0.5 * (shift[n, 0] + shift[n, 1]), float(m))
                            det[n] = x_m - K.cold_coordinate(rp[n], x_m, ph_s / (2 * math.pi))
                            p_abs[n] = p_ring[l] * 0.5 * (o["e0"][n, l] * float(q0.p_loss[0])
                                                          + o["e1"][n, l] * float(q1.p_loss[0]))
                            # temperature that puts the mean resonance at det: current mean detuning d_now (exact,
                            # the phase model's order-m resonance) plus the thermal shift needed (R1 inverted locally
                            # with the slope at the current temperature)
                            d_now = K.resonance_detuning(rp[n], lam_l[b, l], 0.5 * (shift[n, 0] + shift[n, 1]),
                                                         float(m))
                            t_target[n] = t_ring[n] + (det[n] - d_now) / s_t
                            thru = 0.5 * (float(q0.p_thru[0]) + float(q1.p_thru[0]))
                            p_ring[l] *= thru                        # the lock point passes this much of its own line
                        # static thermal map: T_obs = T_amb + G u; solve the heater powers for the ring targets
                        u = np.zeros(sysm.n_src)
                        u[sysm.wg_src] = p_abs
                        pj, pdd = np.zeros(N), np.zeros(N)
                        rs_eff = 0.5 * (rsv[:, 0] + rsv[:, 1])
                        cje = np.zeros(N)
                        el.junction_swing_capacitance_row(vlo, vhi, grid, cc, rp[:, ph.RP_FILL], rp[:, ph.RP_LEN], cje)
                        el.data_activity_heat_row(cje, vhi - vlo, float(np.nan_to_num(pp.datapath.bitrate_hz[b, 0])),
                                                  np.full(N, float(np.nan_to_num(pp.datapath.transition_density[b, 0],
                                                                                 nan=0.5))),
                                                  np.ones(N), rs_eff,
                                                  np.nan_to_num(np.asarray(
                                                      pp.datapath.driver_output_resistance_ohm[b])),
                                                  np.nan_to_num(np.asarray(pp.datapath.driver_static_power_w[b])),
                                                  np.nan_to_num(np.asarray(
                                                      pp.datapath.driver_energy_per_transition_j[b])),
                                                  0, pj, pdd)
                        for n in range(N):
                            if pj_src.size > n and pj_src[n] >= 0:
                                u[pj_src[n]] = pj[n]
                            if pd_src.size > n and pd_src[n] >= 0:
                                u[pd_src[n]] = pdd[n]
                        Gb = G[b if G.shape[0] > 1 else 0]
                        rhs = t_target - t_amb - Gb[ring_obs] @ u
                        Zh = Gb[np.ix_(ring_obs, sysm.heater_src)]
                        p_h = np.linalg.solve(Zh, rhs)
                        u[sysm.heater_src] = p_h
                        t_obs = t_amb + Gb @ u
                        t_ring = t_obs[ring_obs]
                        t_heat = t_obs[heater_obs]
                        ph_lock[:] = ph_at
                    if ds["differs"][b]:
                        # the bank in the data state of the start (data_state): the same target detuning (the
                        # traffic-on operating point, mean phase ph_lock), the heat balance of that state
                        p_abs, p_h, t_ring, t_heat = _hold_in_data_state(
                            ds, b, N, L, vb, swing[b], int(level[b]), rp, grid, dn, al, cc, rs, lam_l[b], t_amb,
                            t_ring, t_heat, det, ph_lock, m_up, bus, r2l, bus_t2[b], p_chip[b], ring_fn,
                            lambda n_, b_=b: rb_of(b_, n_), sysm, G[b if G.shape[0] > 1 else 0], ring_obs, heater_obs,
                            pj_src, pd_src, pp, max(1, int(iterations)))
                    if attempt == 0:
                        hop0 = p_h < 0
                code = np.full(N, np.nan)
                ok = np.zeros(N, bool)
                for n in range(N):
                    if p_h[n] < 0:
                        continue
                    c = _code_for_power(hp, b, n, p_h[n], t_heat[n])
                    if np.isfinite(c) and hp.code_lo <= c <= hp.code_hi:
                        code[n] = c
                        ok[n] = True
                # array heater supply limit (07 P1-12): the plant scales every heater of the world by one droop
                # factor when the sum of the lock powers or currents exceeds the supply (supply_limit_row); then no
                # ring holds its lock (integrator 2026-10-10: the stacked_eic warm-start disagreement of
                # docs/handoff/experiments.md 10.4 was this limit, which quasi_static did not check)
                p_tot = i_tot = 0.0
                for n in range(N):
                    if ok[n]:
                        pn, _, i_n = _heater_drive(hp, b, n, code[n], t_heat[n])
                        p_tot += pn
                        i_tot += i_n
                s_sup = 1.0
                if p_tot > p_sup_max[b] and p_tot > 0.0:
                    s_sup = min(s_sup, math.sqrt(p_sup_max[b] / p_tot))
                if i_tot > i_sup_max[b] and i_tot > 0.0:
                    s_sup = min(s_sup, i_sup_max[b] / i_tot)
                supply_s[ia, iv, b] = s_sup
                if s_sup < 1.0:
                    ok[:] = False
                # wrong-line risk: foreign lines crossed by the resonance on a heater-up sweep from code_min
                # (cold resonance at the ambient, including self-heating-free start) to the lock
                for n in range(N):
                    l = int(r2l[n])
                    lam_res_lock = lam_l[b, l] + det[n]
                    fsr_m = rp[n, ph.RP_LAM_RES] ** 2 / (rp[n, ph.RP_N_G] * rp[n, ph.RP_LEN])
                    span = lam_res_lock - lam_cold[n]
                    cnt = 0
                    for ll in range(L):
                        if ll == l:
                            continue
                        # every order of the ring: offset of line ll from the cold resonance modulo one FSR
                        off = (lam_l[b, ll] - lam_cold[n]) % fsr_m
                        while off <= span:
                            if off > 0:
                                cnt += 1
                            off += fsr_m
                    wrong[ia, iv, b, n] = cnt
                out["det"][ia, iv, b] = det
                out["rel"][ia, iv, b] = rel
                out["relu"][ia, iv, b] = relu
                out["t_ring"][ia, iv, b] = t_ring
                out["p_abs"][ia, iv, b] = p_abs
                out["p_h"][ia, iv, b] = p_h
                out["code"][ia, iv, b] = code
                code_ok[ia, iv, b] = ok
                hop[ia, iv, b] = hop0
                hopped_all[ia, iv, b] = hopped
    with np.errstate(divide="ignore"):
        rel_db = 10 * np.log10(out["rel"])
        relu_db = 10 * np.log10(out["relu"])
    cover = code_ok.all(axis=3).mean(axis=(0, 1))                      # (B,)
    bank_db = 10 * np.log10(np.nanmean(out["rel"], axis=3))            # (A, V, B)
    bnd4 = np.nanmax(bank_db, axis=0) - np.nanmin(bank_db, axis=0)     # (V, B)
    return QuasiStaticResult(ambient_c=amb, bias_v=bv, detuning_target_m=out["det"], reloma_db=rel_db,
                             reloma_unrestricted_db=relu_db, t_ring_k=out["t_ring"], p_abs_w=out["p_abs"],
                             p_heater_w=out["p_h"], heater_code=out["code"], code_ok=code_ok, fsr_hop_risk=hop,
                             wrong_line_risk=wrong, code_range_coverage=cover, bnd4_reloma_variation_db=bnd4,
                             notes=("static thermal gains (DC of the thermal system); single-pass bus; data "
                                    "averaged with ones density 0.5 and transition density of the config; array "
                                    "heater supply limit checked at the lock codes",),
                             supply_scale=supply_s, supply_limited=supply_s < 1.0, order_hopped=hopped_all)


#: the data-path signals of quasi_static's ``data_state`` (the scenario signals data.<key>)
DATA_STATE_KEYS = ("driver_enable", "swing_scale", "transition_density", "ones_density")


def _data_state(data_state, B: int, N: int, td_cfg: np.ndarray) -> dict:
    """quasi_static's ``data_state`` as (B, N) arrays, with the kernel's reading of the signals (driver on when
    enable >= 0.5; ones density clipped to [0, 1]), and ``differs`` (B,): the worlds whose state is not the traffic-on
    state the lock target is defined in (driver on, swing scale 1, the config's transition density td_cfg (B,), ones
    density 0.5)."""
    given = dict(data_state or {})
    bad = sorted(set(given) - set(DATA_STATE_KEYS))
    if bad:
        raise ValueError(f"quasi_static data_state keys must be among {DATA_STATE_KEYS}, got {bad}")
    ones = np.ones((B, N))
    default = {"driver_enable": ones, "swing_scale": ones, "transition_density": td_cfg[:, None] * ones,
               "ones_density": 0.5 * ones}
    out = {}
    for k in DATA_STATE_KEYS:
        out[k] = np.broadcast_to(np.asarray(given.get(k, default[k]), float), (B, N)).copy()
    out["driver_enable"] = (out["driver_enable"] >= 0.5).astype(float)
    out["ones_density"] = np.clip(out["ones_density"], 0.0, 1.0)
    out["differs"] = np.zeros(B, bool)
    for k in DATA_STATE_KEYS:
        out["differs"] |= np.any(out[k] != default[k], axis=1)
    return out


def _hold_in_data_state(ds, b, N, L, vb, swing_b, level_b, rp, grid, dn, al, cc, rs, lam_b, t_amb, t_ring, t_heat,
                        det, ph_lock, m_up, bus, r2l, bus_t2_b, p_chip_b, ring_fn, rb_fn, sysm, Gb, ring_obs,
                        heater_obs, pj_src, pd_src, pp, iterations):
    """The heat balance of world b's bank in the data state ``ds`` (quasi_static ``data_state``) at the traffic-on
    lock target: each ring's mean resonance at det[n] (mean phase ph_lock[n] at its line; the mean phase of a
    detuning does not depend on the junction levels, mrm.plant.phase equation 2), junction levels with the scaled
    swing, absorbed power and through transmission averaged with the state's ones density
    (mrm.plant.optics.nrz_average, as the kernel), and the state's data-activity heat (07 P1-8). The same single-pass
    bus and static thermal map as quasi_static, ``iterations`` times from the given temperatures. Returns (p_abs,
    p_h, t_ring, t_heat) (N,)."""
    from mrm.plant import electrical as el
    from mrm.plant import phase as ph
    en = ds["driver_enable"][b]
    td = ds["transition_density"][b]
    rho = ds["ones_density"][b]
    vlo, vhi, fl = np.zeros(N), np.zeros(N), np.zeros(N, np.int8)
    el.nrz_levels_row(np.full(N, vb), swing_b * ds["swing_scale"][b], level_b, el.JUNCTION_GRID_V_MIN,
                      el.JUNCTION_GRID_V_MAX, vlo, vhi, fl)
    t_ring = np.array(t_ring, float)
    t_heat = np.array(t_heat, float)
    p_abs = np.zeros(N)
    p_h = np.zeros(N)
    lam = np.ascontiguousarray(lam_b)
    bitrate = float(np.nan_to_num(pp.datapath.bitrate_hz[b, 0]))
    r_out = np.nan_to_num(np.asarray(pp.datapath.driver_output_resistance_ohm[b]))
    p_static = np.nan_to_num(np.asarray(pp.datapath.driver_static_power_w[b]))
    e_tr = np.nan_to_num(np.asarray(pp.datapath.driver_energy_per_transition_j[b]))
    Zh = Gb[np.ix_(ring_obs, sysm.heater_src)]
    for _ in range(iterations):
        o = {k: np.zeros((N, L)) for k in ("phi0", "phi1", "a0", "a1", "e0", "e1", "k2t", "k2d", "trt")}
        shift, cj, rsv, jf = np.zeros((N, 2)), np.zeros((N, 2)), np.zeros((N, 2)), np.zeros(N, np.int8)
        ph.ring_optics_row(rp, lam, t_ring, t_heat, vlo, vhi, grid, dn, al, cc, rs, o["phi0"], o["phi1"], o["a0"],
                           o["a1"], o["e0"], o["e1"], o["k2t"], o["k2d"], o["trt"], shift, cj, rsv, jf)
        p_ring = p_chip_b.copy()
        t_target = np.zeros(N)
        for q in range(N):
            n = int(bus[q])
            l = int(r2l[n])
            p_ring *= bus_t2_b[n]
            dphi = o["phi1"][n, l] - o["phi0"][n, l]
            q0 = ring_fn(np.array([ph_lock[n] - 0.5 * dphi]), o["k2t"][n, l], o["k2d"][n, l], o["a0"][n, l],
                         o["trt"][n, l], rb=rb_fn(n))
            q1 = ring_fn(np.array([ph_lock[n] + 0.5 * dphi]), o["k2t"][n, l], o["k2d"][n, l], o["a1"][n, l],
                         o["trt"][n, l], rb=rb_fn(n))
            r1 = float(rho[n])
            p_abs[n] = p_ring[l] * ((1.0 - r1) * o["e0"][n, l] * float(q0.p_loss[0])
                                    + r1 * o["e1"][n, l] * float(q1.p_loss[0]))
            s_t = rp[n, ph.RP_S0] * (1 + rp[n, ph.RP_BETA] * (t_ring[n] - rp[n, ph.RP_T0]))
            d_now = K.resonance_detuning(rp[n], lam_b[l], 0.5 * (shift[n, 0] + shift[n, 1]), float(m_up[n]))
            t_target[n] = t_ring[n] + (det[n] - d_now) / s_t
            p_ring[l] *= (1.0 - r1) * float(q0.p_thru[0]) + r1 * float(q1.p_thru[0])
        u = np.zeros(sysm.n_src)
        u[sysm.wg_src] = p_abs
        pj, pdd = np.zeros(N), np.zeros(N)
        cje = np.zeros(N)
        el.junction_swing_capacitance_row(vlo, vhi, grid, cc, rp[:, ph.RP_FILL], rp[:, ph.RP_LEN], cje)
        el.data_activity_heat_row(cje, vhi - vlo, bitrate, np.ascontiguousarray(td), np.ascontiguousarray(en),
                                  0.5 * (rsv[:, 0] + rsv[:, 1]), r_out, p_static, e_tr, 0, pj, pdd)
        for n in range(N):
            if pj_src.size > n and pj_src[n] >= 0:
                u[pj_src[n]] = pj[n]
            if pd_src.size > n and pd_src[n] >= 0:
                u[pd_src[n]] = pdd[n]
        rhs = t_target - t_amb - Gb[ring_obs] @ u
        p_h = np.linalg.solve(Zh, rhs)
        u[sysm.heater_src] = p_h
        t_obs = t_amb + Gb @ u
        t_ring = t_obs[ring_obs]
        t_heat = t_obs[heater_obs]
    return p_abs, p_h, t_ring, t_heat


def _heater_drive(hp, b, n, code, t_heater_k) -> tuple[float, float, float]:
    """(P (W), V (V), I (A)) of ring n's heater of world b at ``code`` and heater-resistor temperature t_heater_k: the
    static driver law with TCR (mrm.plant.electrical.heater_target_row + heater_power_row with g_self = 0), before the
    array supply limit."""
    from mrm.plant import electrical as el
    law = int(hp.law[b if hp.law.shape[0] > 1 else 0])
    u = np.zeros(1)
    rip = np.zeros(1)
    el.heater_target_row(np.array([int(code)], np.int64), law, hp.bits, hp.pwm_bits,
                         hp.bit_err_lsb[min(b, hp.bit_err_lsb.shape[0] - 1)][n:n + 1],
                         hp.inl_bow_lsb[min(b, hp.inl_bow_lsb.shape[0] - 1)][n:n + 1], 0, 0, 1, u, rip)
    p = np.zeros(1)
    v = np.zeros(1)
    i = np.zeros(1)
    r = np.zeros(1)
    f = np.zeros(1, np.int8)

    def row(a):
        a = np.asarray(a, float)
        a = a[min(b, a.shape[0] - 1)] if a.ndim == 2 else a
        return np.ascontiguousarray(a[n:n + 1] if a.size > 1 else a.reshape(1))
    el.heater_power_row(u, law, np.array([t_heater_k]), np.zeros(1), row(hp.r_cold_ohm), row(hp.tcr_per_k),
                        row(hp.tcr_ref_k), row(hp.v_full_scale_v), row(hp.i_full_scale_a),
                        row(hp.p_full_scale_w), row(hp.compliance_v), p, v, i, r, f)
    return float(p[0]), float(v[0]), float(i[0])


def _code_for_power(hp, b, n, p_w, t_heater_k):
    """Smallest heater code whose static power at heater-resistor temperature t_heater_k reaches p_w (driver law
    with TCR, mrm.plant.electrical.heater_power_row with g_self = 0 at the solved temperature): bisection on the
    monotone code -> power map. nan when even the top code is short."""

    def power(code):
        return _heater_drive(hp, b, n, code, t_heater_k)[0]
    lo, hi = int(hp.code_lo), int(hp.code_hi)
    if power(hi) < p_w:
        return math.nan
    if power(lo) >= p_w:
        return float(lo)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if power(mid) >= p_w:
            hi = mid
        else:
            lo = mid
    return float(hi)
