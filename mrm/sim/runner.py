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
  start), ``path`` ('auto' | 'compiled' | 'python'), ``n_threads``, ``oracle``, ``histogram``, ``chunk_ticks``
  (ticks per kernel call on the compiled path, default 1e5), ``controller_params`` (a mapping, passed to
  mrm.control.registry.create when ``controller`` is a name), ``cfg``, ``auto_max_ticks``, ``oracle_period_ticks``.
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

from mrm.control.api import CompiledController, ControllerSpec, Observation, controller_fast_kernel
from mrm.core.timing import seconds_to_ticks
from mrm.plant.plant import Plant, Truth
from mrm.sim import kernel as K
from mrm.sim.metrics import MetricsSpec, RecordSpec

__all__ = ["run", "RunResult", "quasi_static", "QuasiStaticResult", "load_inputs"]


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
                  oracle=options.get("oracle", True), histogram=options.get("histogram"))
    plant.reset(protocol=options.get("reset_protocol"), heater_code=options.get("heater_code"))
    n_threads = options.get("n_threads")
    plant.n_threads = n_threads
    if ctrl is not None:
        ctrl.reset(spec)
    if ctrl is None:
        _run_open(plant, n_ticks, n_threads)
        ctrl_name = None
    elif compiled:
        plant.bind_compiled(ctrl)
        chunk = int(options.get("chunk_ticks", 100_000))
        k_end = plant.tick + n_ticks
        while plant.tick < k_end:
            plant.run(min(chunk, k_end - plant.tick), mode=K.MODE_COMPILED, step_fn=ctrl.step_fn, fast_fn=fast)
        ctrl_name = ctrl.name
    else:
        if spec.tick_jitter_ticks > 0:
            raise NotImplementedError("the Python controller path needs edges shared by every world (no tick "
                                      "jitter); use a CompiledController")
        _run_python(plant, ctrl, spec, n_ticks, fast)
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
          "records": [r.key for r in plant.records_state.specs], "diagnostics": diag}
    if sc.time_compression != 1.0:
        mi["compression_note"] = ("compressed result: report it only after mrm.scenarios."
                                  "require_compression_validated (07 P0-10)")
    return RunResult(metrics=m, trace=trace, truth_final=plant.truth(), manifest_inputs=mi, diagnostics=diag,
                     wall_s=wall, plant=plant)


def _run_open(plant: Plant, n_ticks: int, n_threads=None) -> None:
    k_end = plant.tick + n_ticks
    while plant.tick < k_end:
        plant.run(min(100_000, k_end - plant.tick), mode=K.MODE_OPEN)


def _run_python(plant: Plant, ctrl, spec, n_ticks: int, fast) -> None:
    """Python slow path, vectorized over B (07 P0-9): the kernel stops at every controller edge; the controller
    steps on an Observation of the released channels; ActionValidator checks shape, dtype and range (07 P1-19); the
    command enters the controller pipeline when the kernel resumes."""
    k_end = plant.tick + n_ticks
    resume = False
    truth_fed = bool(getattr(ctrl, "truth_fed", False))
    held = getattr(ctrl, "act", None) if isinstance(ctrl, CompiledController) else None
    if isinstance(ctrl, CompiledController):
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
        if np.any(plant.S.wi[:, K.WI_CTRL_RESET]):           # scenario fault.controller_reset
            ctrl.reset(spec)
            plant.S.wi[:, K.WI_CTRL_RESET] = 0
        sensors, new = plant.sample_sensors()
        obs = Observation(plant.tick, plant.dt, sensors, new)
        actions = ctrl.step(obs, truth=plant.truth()) if truth_fed else ctrl.step(obs)
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
    each is a possible capture, 07 P1-12). Summaries: code_range_coverage (fraction of (ambient, bias) points with
    every ring of the world reachable), bnd4_reloma_variation_db (V, B): max - min over ambient of the bank-mean
    oracle relOMA at each bias (BND-4 asks < 1 dB).
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


def quasi_static(cfg_or_params, *, ambient_c=None, bias_v=None, batch: int = 1, seeds=None, p_in_w=None,
                 n_scan: int = 301, iterations: int = 4, cfg=None, uncertainty=None, variation=None
                 ) -> QuasiStaticResult:
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
    is stated at 1 mW into the MRM: pass 1e-3 / input transmission). Light reaching each ring: the chip power
    times the through transmission of the rings before it at their lock points (single pass).
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
    swing = np.nan_to_num(np.asarray(pp.datapath.swing_pp_v, float)) * np.ones((B, N))
    level = np.asarray(pp.datapath.level_convention).reshape(-1)
    fwhm_hz = np.asarray(pp.optics.fwhm_hz, float)
    shp = (A, V, B, N)
    out = {k: np.full(shp, np.nan) for k in ("det", "rel", "relu", "t_ring", "p_abs", "p_h", "code")}
    code_ok = np.zeros(shp, bool)
    hop = np.zeros(shp, bool)
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
                det = np.zeros(N)
                rel = np.zeros(N)
                relu = np.zeros(N)
                p_abs = np.zeros(N)
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
                        d_now = K.resonance_detuning(rp[n], lam_l[b, l], 0.5 * (shift[n, 0] + shift[n, 1]), float(m))
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
                                              np.nan_to_num(np.asarray(pp.datapath.driver_output_resistance_ohm[b])),
                                              np.nan_to_num(np.asarray(pp.datapath.driver_static_power_w[b])),
                                              np.nan_to_num(np.asarray(pp.datapath.driver_energy_per_transition_j[b])),
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
                code = np.full(N, np.nan)
                ok = np.zeros(N, bool)
                for n in range(N):
                    if p_h[n] < 0:
                        continue
                    c = _code_for_power(hp, b, n, p_h[n], t_heat[n])
                    if np.isfinite(c) and hp.code_lo <= c <= hp.code_hi:
                        code[n] = c
                        ok[n] = True
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
                hop[ia, iv, b] = p_h < 0
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
                                    "averaged with ones density 0.5 and transition density of the config",))


def _code_for_power(hp, b, n, p_w, t_heater_k):
    """Smallest heater code whose static power at heater-resistor temperature t_heater_k reaches p_w (driver law
    with TCR, mrm.plant.electrical.heater_power_row with g_self = 0 at the solved temperature): bisection on the
    monotone code -> power map. nan when even the top code is short."""
    from mrm.plant import electrical as el
    law = int(hp.law[b if hp.law.shape[0] > 1 else 0])

    def power(code):
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
        return p[0]
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
