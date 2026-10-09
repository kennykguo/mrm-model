"""Closed-loop harness for the model-based baselines (controllers_c) and its own checks.

The plant is tests/verification/test_plant_wiring.py::ReferencePlant (imported, not copied): the Python plant tick
built only from PlantParams and the leaf adapters. This module adds what a closed-loop test needs and the
ReferencePlant does not provide:

- ``warm_start``: put every world at a steady state of the TRUE plant (``mrm.control.models.static_point`` with the
  true PlantParams) by writing the thermal lag states, the self-heating branch, the heater DAC registers and the
  previous-tick powers. This is how a test starts "locked" without simulating an acquisition.
- ``Loop``: the event order of a controller edge (mrm.core.timing): release the sensor banks at the edge, build a
  strict Observation of the declared channels only, step the controller, validate the actions (ActionValidator:
  clip, count, cast) and latch the heater command ``latency_ticks`` later.
- Truth readouts inside the TEST only: detuning delta = lambda_res - lambda_line of every ring each tick
  (``models.resonance_detuning_m`` on the plant's own temperatures, lines and junction levels) and the OMA of each
  ring at its line (|P_out1 - P_out0| from the plant's optics).
- ``lock_metrics``: lock error (pm), lock-excursion count with hysteresis (07 P0-8), time to lock, OMA penalty
  against the oracle (10 log10 OMA_oracle / OMA).

The tests here check the harness itself: a warm start is a fixed point of the plant, the plant runs at the
conditions of the controller's nominal model (scenario bias and ambient from the config), the array-supply limits
(power and current) decide which sampled worlds can be warm-started at all, and the loop delay the linear model
assumes is the one the plant produces (07 P0-6 ``test_loop_delay_impulse`` in miniature).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pytest

from mrm import params
from mrm.control import models as M
from mrm.control.api import ActionValidator, ControllerSpec, Observation
from mrm.core import timing
from mrm.plant import electrical as el
from mrm.scenarios.generators import Scenario, const

from test_plant_wiring import ReferencePlant

KNOWLEDGE = ["line_grid", "ring_to_line", M.KNOWLEDGE_TOKEN]
LOCK_THRESHOLD_PM = 5.0          # g26 metrics.lock_threshold_pm (textbook 5 pm budget)
HYSTERESIS_PM = 1.0              # g26 metrics.hysteresis_pm


def load_cfg(preset: str = "g26", overrides=(), alts=None, knowledge: bool = True):
    ov = list(overrides)
    if knowledge:
        ov.append("controller_interface.knowledge=[" + ",".join(KNOWLEDGE) + "]")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load(preset, overrides=ov, alts=alts)


def realize(cfg, B=1, uncertainty="nominal", variation="off", seeds=7):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.realize(cfg, B, seeds=seeds, uncertainty=uncertainty, variation=variation)


def quiet_scenario(**const_signals) -> Scenario:
    """Absolute constants only; no aggressor, no drift. :func:`build_rig` adds the config's defaults (ambient,
    scenario bias, bias offset, transition density) with :func:`with_config_defaults`."""
    comps = tuple(const(k, v, absolute=True) for k, v in const_signals.items())
    return Scenario(name="quiet", components=comps)


#: signal defaults that mrm.scenarios.registry.from_config takes from the config (scenario.ambient_c, scenario.bias_v,
#: scenario.bias_offset_v, datapath.transition_density)
_CFG_DEFAULT_SIGNALS = ("thermal.t_base_k", "tx.bias_v", "tx.bias_offset_v", "data.transition_density")


def with_config_defaults(sc: Scenario, cfg) -> Scenario:
    """The scenario with the config's signal defaults, as ``mrm.scenarios.registry.from_config`` (and so the engine)
    applies them. Without this a library scenario (``registry.create('a2')``) or :func:`quiet_scenario` leaves
    tx.bias_v undriven, the bias DAC stays at code 0 (ring.bias_dac.v_min_v = 1.0 V in g26) instead of
    scenario.bias_v = 1.05 V, and the plant runs at another bias than the controller's nominal model (a 0.2 pm
    target shift and ~110 monitor codes, which a set-point lock reads as a 0.6 pm offset)."""
    from mrm.scenarios import registry as _sreg
    base = _sreg.from_config(cfg).defaults
    d = dict(sc.defaults)
    for s in _CFG_DEFAULT_SIGNALS:
        if s in base:
            d[s] = base[s]
    return sc.replace(defaults=d)


def calibrate(sc, pp):
    """Aggressor calibration of a library scenario on the plant's thermal system (ENGINE_BRIEF section 1)."""
    from mrm.scenarios import calibrate_scenario, foster_from_system
    if not sc.aggressors:
        return sc
    model = {a.site: {"s0_m_per_k": pp.thermal.dlambda_dt_s0_m_per_k[:, a.victim_ring],
                      "beta_per_k": pp.thermal.dlambda_dt_beta_per_k[:, 0],
                      "t_ref_k": pp.thermal.dlambda_dt_t0_k[:, 0],
                      "ring_t0_k": pp.thermal.t_ring_lock_k[:, a.victim_ring]} for a in sc.aggressors}
    resp = {a.site: foster_from_system(pp.thermal.system, victim_ring=a.victim_ring, site=a.site)
            for a in sc.aggressors}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc, _ = calibrate_scenario(sc, resp, model=model)
    return sc


@dataclass
class Rig:
    """A plant of B worlds plus everything a closed-loop test needs (Truth stays here, never in the controller)."""
    cfg: object
    pp: object
    spec: ControllerSpec
    plant: ReferencePlant
    k: int = 0
    targets_m: np.ndarray | None = None          # (B, N) warm-start detunings
    oracle_oma_w: np.ndarray | None = None       # (B, N) OMA at the oracle point (static, start conditions)
    oracle_detuning_m: np.ndarray | None = None  # (B, N)
    dark0_codes: np.ndarray | None = None        # (B, N) dark-current code at the quiet lock (no aggressor)
    static: list = field(default_factory=list)   # StaticPoint per world (true plant)

    @property
    def B(self):
        return self.pp.batch

    @property
    def N(self):
        return self.pp.n_rings

    def conditions(self, b: int) -> M.Conditions:
        """Truth conditions of world b at the last plant tick (lines, chip power, junction levels, data)."""
        last = self.plant.last
        ev = self.plant.arr.evaluate(int(self.k), self.plant.cache)
        return M.Conditions(lam_m=np.ascontiguousarray(last["lam"][b]), p_chip_w=np.ascontiguousarray(last["p_chip"][b]),
                            t_base_k=float(ev["thermal.t_base_k"][b, 0] + ev["thermal.xpu_rise_k"][b, 0]),
                            v_lo=np.ascontiguousarray(last["v_lo"][b]), v_hi=np.ascontiguousarray(last["v_hi"][b]),
                            rho=np.ascontiguousarray(last["rho"][b]),
                            transition_density=np.ascontiguousarray(ev["data.transition_density"][b]),
                            aggressor_w=(np.asarray(ev["thermal.aggressor_w"][b]) if self.plant.sys.aggressor_src.size
                                         else None))

    def detuning_m(self) -> np.ndarray:
        """TRUTH: (B, N) detuning of every ring at the current plant state."""
        return np.stack([M.resonance_detuning_m(self.pp, b, self.plant.t_ring[b], self.conditions(b))
                         for b in range(self.B)])

    def oma_w(self) -> np.ndarray:
        return np.abs(self.plant.last["oma"])

    def feasible_worlds(self) -> np.ndarray:
        """TRUTH: (B,) True where every ring's warm-start point is reachable (mrm.validation.oracle.reachable: positive
        heater power, codes inside the DAC range with heater authority, and the bank within the array supply's power
        AND current limits, 07 P1-12; above either the supply droops every heater, so the injected steady state is
        not one and the rings fold before any controller acts). The current limit (150 mA in g26) was missing here
        before the review: two "reachable" LHS worlds of the mismatch report were in supply droop."""
        from mrm.validation.oracle import reachable
        return np.array([bool(np.all(reachable(self.pp, sp))) for sp in self.static])

    def truth_state(self):
        """TRUTH at the current edge for the Truth-fed oracle (mrm.validation.oracle.TruthState)."""
        from mrm.validation.oracle import TruthState
        return TruthState(tick=self.k, x=np.array(self.plant.x), t_ring_k=np.array(self.plant.t_ring),
                          conditions=[self.conditions(b) for b in range(self.B)],
                          heater_applied=np.array(self.plant.heater.applied_codes))

    def dark_codes(self) -> np.ndarray:
        """TRUTH: (B, N) monitor code of the PD dark current at the current PD temperature (Arrhenius, 07 P1-9)."""
        hw = self.plant.mon.hw
        t_pd = self.plant.t_obs[:, self.plant.sys.pd_obs]
        _, i_d = hw.photocurrent_a(np.zeros((self.B, self.N)), 0.0, t_pd)
        return (hw.ideal_code(i_d) - hw.ideal_code(0.0))

    def dy_ddelta(self) -> np.ndarray:
        """TRUTH: (B, N) monitor codes per m of detuning at the warm-start point (static slope of the true plant)."""
        return np.stack([sp.codes_per_w * sp.dpdrop_dd for sp in self.static])

    def reference_detuning_m(self) -> np.ndarray:
        """TRUTH: the detuning where a monitor set-point lock holds its code when the PD dark current differs from
        its quiet value (aggressor heating of the PD, 07 P1-9): target + offset / |dy/d delta| (the monitor cannot
        tell the two apart). Equal to the target when the PD is at its quiet-lock temperature."""
        if self.targets_m is None:                    # no warm start: no reference
            return np.full((self.B, self.N), np.nan)
        return self.targets_m + (self.dark_codes() - self.dark0_codes) / np.abs(self.dy_ddelta())


def build_rig(cfg, *, B=1, uncertainty="nominal", variation="off", seeds=7, scenario=None, noise_seed=11,
              k0=0, warm_detuning_m="oracle", pp=None, aggressor_presettle_ticks: int = 0) -> Rig:
    """Realize the plant, build the spec from the (knowledge-carrying) config, and warm-start every world."""
    pp = realize(cfg, B, uncertainty, variation, seeds) if pp is None else pp
    sc = quiet_scenario() if scenario is None else scenario
    sc = calibrate(with_config_defaults(sc, cfg), pp)
    spec = ControllerSpec.from_config(cfg, n_batch=pp.batch, nominal=cfg, scenario_drives=sc.drives,
                                      scenario_offsets=sc.offsets)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plant = ReferencePlant(pp, cfg, scenario=sc, noise_seed=noise_seed, pop_period=spec.period_ticks)
    rig = Rig(cfg=cfg, pp=pp, spec=spec, plant=plant, k=k0)
    if warm_detuning_m is not None:
        warm_start(rig, warm_detuning_m, aggressor_presettle_ticks)
    else:                                   # cold start: one tick so the plant has outputs before the first edge
        plant.tick(rig.k)
        rig.k += 1
    return rig


def _presettle_aggressor(rig: Rig, n_pre: int) -> np.ndarray:
    """Exact FOH run of the aggressor lags alone over [k - n_pre, k] (the thermal system is linear and these lags see
    only the aggressor sources). Returns their lag states (B, M_agg) at the current tick and their indices."""
    sysm = rig.plant.sys
    m = np.flatnonzero(np.isin(sysm.lag_src, sysm.aggressor_src))
    a, c0, c1 = sysm.coefficients(rig.pp.timing.dt_s)
    a, c0, c1 = a[:, m], c0[:, m], c1[:, m]
    col = {int(s): i for i, s in enumerate(sysm.aggressor_src)}
    which = np.array([col[int(s)] for s in sysm.lag_src[m]])
    x = np.zeros((rig.B, m.size))
    def p_at(k):
        return rig.plant.arr.evaluate(int(k), rig.plant.cache)["thermal.aggressor_w"][:, which]
    p0 = p_at(rig.k - n_pre)
    for k in range(rig.k - n_pre, rig.k):
        p1 = p_at(k + 1)
        x = a * x + c0 * p0 + c1 * p1
        p0 = p1
    return x, m


def warm_start(rig: Rig, detuning_m="oracle", aggressor_presettle_ticks: int = 0) -> None:
    """Write a steady state of the TRUE plant with every ring at ``detuning_m`` ((B, N) m, or "oracle": the OMA-optimal
    stable detuning of each world). One plant tick runs first so the laser/data/bias values exist. With a time-varying
    aggressor, ``aggressor_presettle_ticks`` > 0 runs the aggressor lags alone over that history (exact) and the heaters
    are solved against their actual (non-steady) contribution."""
    plant, pp = rig.plant, rig.pp
    B, N = rig.B, rig.N
    # run past the bias DAC latency (and its slew) so the conditions read below are the ones the plant will hold:
    # at the first tick the bias DAC still outputs code 0 (v_min), not the scenario bias
    for _ in range(2 + int(rig.spec.bias_dac.latency_ticks) + 4):
        plant.tick(rig.k)
        rig.k += 1
    sysm = plant.sys
    agg = None
    if aggressor_presettle_ticks > 0 and sysm.aggressor_src.size:
        agg = _presettle_aggressor(rig, int(aggressor_presettle_ticks))
    targets = np.zeros((B, N))
    rig.static = []
    rig.oracle_oma_w = np.zeros((B, N))
    rig.oracle_detuning_m = np.zeros((B, N))
    u = np.zeros((B, sysm.n_src))
    codes = np.zeros((B, N), np.int64)
    t_base = np.zeros(B)
    t_pd_quiet = np.zeros((B, N))
    for b in range(B):
        cond = rig.conditions(b)
        quiet = M.Conditions(**{**cond.__dict__, "aggressor_w": None, "t_extra_obs_k": None})
        if agg is not None:
            xa = np.zeros(sysm.n_lags)
            xa[agg[1]] = agg[0][b]
            extra = sysm.outputs(xa[None], 0.0)[0]
            cond = M.Conditions(**{**cond.__dict__, "aggressor_w": None, "t_extra_obs_k": extra})
        d_opt, oma_opt, _, _ = M.oma_optimal_detuning(pp, b, cond=cond)
        rig.oracle_oma_w[b] = oma_opt
        rig.oracle_detuning_m[b] = d_opt
        tgt = d_opt if isinstance(detuning_m, str) else np.broadcast_to(np.asarray(detuning_m, float), (B, N))[b]
        sp = M.static_point(pp, b, tgt, cond=cond)
        rig.static.append(sp)
        t_pd_quiet[b] = M.static_point(pp, b, tgt, cond=quiet, with_slopes=False).t_pd_k
        targets[b] = sp.detuning_m
        u[b, sysm.heater_src] = sp.heater_power_w
        u[b, sysm.wg_src] = sp.p_abs_w
        u[b, sysm.junction_src] = sp.p_junction_w
        if sysm.driver_src.size:
            u[b, sysm.driver_src] = sp.p_driver_w[:sysm.driver_src.size]
        if sysm.aggressor_src.size and cond.aggressor_w is not None:
            u[b, sysm.aggressor_src] = cond.aggressor_w
        codes[b] = np.rint(sp.heater_code).astype(np.int64)
        t_base[b] = cond.t_base_k
    plant.x[:] = sysm.steady_state(u)
    if agg is not None:
        plant.x[:, agg[1]] = agg[0]
    t_obs = sysm.outputs(plant.x, t_base)
    plant.t_obs[:] = t_obs
    plant.t_ring[:] = t_obs[:, sysm.ring_obs]
    plant.t_heater[:] = t_obs[:, sysm.heater_obs]
    plant.t_branch[:] = plant.t_ring
    plant.p_prev[:] = u[:, sysm.wg_src]
    plant.u_heater[:] = u[:, sysm.heater_src]
    plant.heater = el.HeaterBank(plant.heater_p, initial_code=codes)
    rig.targets_m = targets
    hw = plant.mon.hw
    _, i_d = hw.photocurrent_a(np.zeros((B, N)), 0.0, t_pd_quiet)
    rig.dark0_codes = hw.ideal_code(i_d) - hw.ideal_code(0.0)


def hold(rig: Rig, n_ticks: int, cmd=None) -> np.ndarray:
    """Open-loop: hold the heater registers (or latch ``cmd`` at the first tick). Returns truth detuning (T, B, N)."""
    out = np.zeros((n_ticks, rig.B, rig.N))
    for i in range(n_ticks):
        if rig.k % rig.spec.period_ticks == rig.spec.phase_ticks:
            rig.plant.release(rig.k)            # drain the sensor FIFOs at every edge, as a running loop does
        rig.plant.tick(rig.k, heater_cmd=cmd if i == 0 else None)
        rig.k += 1
        out[i] = rig.detuning_m()
    return out


@dataclass
class Trace:
    """Per-tick truth and per-edge controller records of one closed-loop run."""
    ticks: np.ndarray
    detuning_m: np.ndarray            # (T, B, N) TRUTH
    oma_w: np.ndarray                 # (T, B, N) TRUTH
    ref_m: np.ndarray                 # (T, B, N) TRUTH: target moved by the dark-current offset (monitor-consistent)
    edge_ticks: np.ndarray            # (E,)
    heater_codes: np.ndarray          # (E, B, N) validated commands
    y_latest: np.ndarray              # (E, B, N)
    telemetry: dict
    saturation: dict
    step_wall_s: float = 0.0


def run_loop(rig: Rig, ctrl, n_ticks: int, *, reset: bool = True, record_every: int = 1, on_edge=None) -> Trace:
    """Closed loop for n_ticks plant ticks (event order of mrm.core.timing; edges every spec.period_ticks)."""
    import time
    spec, plant = rig.spec, rig.plant
    truth_fed = bool(getattr(ctrl, "truth_fed", False))
    if reset:
        if truth_fed:
            ctrl.reset(spec, rig.pp, rig.truth_state(), cfg=rig.cfg)
        else:
            ctrl.reset(spec)
    val = ActionValidator(spec)
    lat = int(ctrl.latency_ticks)
    sensors_declared = set(spec.sensors)
    pending: dict[int, np.ndarray] = {}
    T = n_ticks // record_every
    det = np.zeros((T, rig.B, rig.N))
    oma = np.zeros((T, rig.B, rig.N))
    ref = np.zeros((T, rig.B, rig.N))
    ticks = np.zeros(T, np.int64)
    e_ticks, codes, ylat = [], [], []
    tel: dict[str, list] = {}
    wall = 0.0
    for i in range(n_ticks):
        k = rig.k
        if (k - spec.phase_ticks) % spec.period_ticks == 0:
            sensors, new = plant.release(k)
            sensors = {n: v for n, v in sensors.items() if n in sensors_declared}
            new = {n: v for n, v in new.items() if n in sensors_declared}
            obs = Observation(k, spec.tick_s, sensors, new)
            t0 = time.perf_counter()
            act = val.validate(ctrl.step(obs, rig.truth_state()) if truth_fed else ctrl.step(obs))
            wall += time.perf_counter() - t0
            if "tx.heater" in act:
                pending[k + lat] = act["tx.heater"].astype(np.int64)
            e_ticks.append(k)
            codes.append(np.array(act["tx.heater"], np.int64) if "tx.heater" in act
                         else np.array(plant.heater.applied_codes))
            ylat.append(np.array(sensors["tx.monitor.latest"], float))
            for key, v in (ctrl.telemetry() if hasattr(ctrl, "telemetry") else {}).items():
                tel.setdefault(key, []).append(np.array(v, float))
            if on_edge is not None:
                on_edge(rig, k)
        plant.tick(k, heater_cmd=pending.pop(k, None))
        rig.k += 1
        if i % record_every == 0 and i // record_every < T:
            j = i // record_every
            ticks[j] = rig.k
            det[j] = rig.detuning_m()
            oma[j] = rig.oma_w()
            ref[j] = rig.reference_detuning_m()
    return Trace(ticks=ticks, detuning_m=det, oma_w=oma, ref_m=ref, edge_ticks=np.array(e_ticks),
                 heater_codes=np.array(codes), y_latest=np.array(ylat),
                 telemetry={k: np.array(v) for k, v in tel.items()}, saturation=val.saturation_counts(),
                 step_wall_s=wall)


def lock_metrics(det_m, ref_m, *, threshold_pm=LOCK_THRESHOLD_PM, hysteresis_pm=HYSTERESIS_PM, dt_s=1e-6,
                 oma_w=None, oracle_oma_w=None, settle_ticks=0) -> dict:
    """Lock metrics from TRUTH (07 P0-8, 4.8): error e = delta - ref (pm) over the scored window (after settle_ticks):
    rms_pm, max_pm (max |e|), mean_pm; excursions = number of times |e| rises above threshold after having been below
    threshold - hysteresis (counted per ring, summed, in the scored window); time_to_lock_s = first time |e| <
    threshold and stays below for the rest of the run (per ring, max over rings; inf if never; measured from the
    start of the run, settle_ticks ignored); oma_penalty_db = mean over the scored window of
    10 log10(OMA_oracle / OMA) (per ring, max over rings)."""
    r = np.asarray(ref_m)
    e = (np.asarray(det_m) - (r if r.ndim == 3 else r[None])) * 1e12
    es = e[settle_ticks:]
    out = {"rms_pm": float(np.sqrt(np.mean(es ** 2))), "max_pm": float(np.max(np.abs(es))),
           "mean_pm": float(np.mean(es))}
    above = np.abs(es) > threshold_pm
    below = np.abs(es) < threshold_pm - hysteresis_pm
    n_exc = 0
    ae = np.abs(es).reshape(es.shape[0], -1)
    for j in range(ae.shape[1]):
        armed = bool(below[0].reshape(-1)[j]) or not bool(above[0].reshape(-1)[j])
        for t in range(1, ae.shape[0]):
            if armed and ae[t, j] > threshold_pm:
                n_exc += 1
                armed = False
            elif not armed and ae[t, j] < threshold_pm - hysteresis_pm:
                armed = True
    out["excursions"] = n_exc
    ttl = []
    af = np.abs(e).reshape(e.shape[0], -1)
    for j in range(af.shape[1]):
        bad = np.flatnonzero(af[:, j] >= threshold_pm)
        ttl.append(0.0 if bad.size == 0 else ((bad[-1] + 1) * dt_s if bad[-1] + 1 < af.shape[0] else math.inf))
    out["time_to_lock_s"] = float(max(ttl))
    if oma_w is not None and oracle_oma_w is not None:
        pen = 10.0 * np.log10(np.asarray(oracle_oma_w)[None] / np.maximum(np.asarray(oma_w), 1e-30))
        out["oma_penalty_db"] = float(np.max(np.mean(pen[settle_ticks:], axis=0)))
    return out


# ------------------------------------------------------------------------------------------------ harness checks
@pytest.fixture(scope="module")
def nominal_rig():
    cfg = load_cfg()
    return build_rig(cfg)


def test_warm_start_is_a_fixed_point_of_the_plant(nominal_rig):
    """The injected state (models.static_point on the TRUE plant) is an equilibrium of ReferencePlant: with the
    heater codes held, the rings stay at the target detuning (the residual drift comes from the integer code, the
    DAC errors and the shot noise of the monitor, none of which the static solve models)."""
    rig = nominal_rig
    det = hold(rig, 300)
    err_pm = (det - rig.targets_m[None]) * 1e12
    assert np.all(np.abs(err_pm[0]) < 2.0), err_pm[0]
    assert np.all(np.abs(err_pm[-1]) < 2.0), err_pm[-1]
    # and the oracle detuning sits on the stable side (resonance red of the laser) with a self-heating factor > 1
    assert np.all(rig.oracle_detuning_m > 0)
    assert np.all(rig.static[0].selfheat_factor > 1.0)


def test_plant_runs_at_the_controllers_nominal_conditions(nominal_rig):
    """The harness plant (nominal world) runs at the conditions the controller's nominal model assumes: the config's
    scenario defaults (scenario.bias_v = 1.05 V, ambient) reach the plant, and the warm start is solved after the bias
    DAC latency. (Before this was fixed the plant sat at the bias DAC's v_min = 1.0 V: the warm-start target moved by
    0.2 pm and the monitor code by ~110 codes, which every set-point lock read as a 0.6 pm offset.)"""
    rig = nominal_rig
    pp_nom = M.nominal_plant_params(rig.cfg)
    nom = M.conditions_from_config(pp_nom, 0, cfg=rig.cfg)
    tru = rig.conditions(0)
    np.testing.assert_allclose(tru.v_lo, nom.v_lo, atol=2e-3)          # bias DAC quantization + noise only
    np.testing.assert_allclose(tru.v_hi, nom.v_hi, atol=2e-3)
    assert tru.t_base_k == pytest.approx(nom.t_base_k)
    np.testing.assert_allclose(tru.lam_m, nom.lam_m, rtol=1e-12)
    np.testing.assert_allclose(tru.p_chip_w, nom.p_chip_w, rtol=1e-3)  # RIN
    lm = M.linearize_bank(rig.cfg, M.OperatingPoint(), latency_ticks=1)[0]
    np.testing.assert_allclose(rig.static[0].y_code, lm.op.y_code, atol=10.0)       # codes (~0.05 pm)
    np.testing.assert_allclose(rig.targets_m[0], lm.op.detuning_m, atol=0.02e-12)


def test_loop_delay_of_the_model_matches_the_plant():
    """07 P0-6 in miniature: a heater step commanded at an edge is first seen by the controller at
    timing.first_response_tick, and the linear model's delay tau = tau_u + tau_m is consistent with it (tau_u is the
    tick the code reaches the heater; the first sample that sees it has sample tick >= k_act + 1)."""
    cfg = load_cfg()
    rig = build_rig(cfg)
    spec = rig.spec
    hold(rig, 40 - rig.k)
    rig.plant.release(rig.k)
    k_edge = rig.k
    assert k_edge % spec.period_ticks == 0
    lat_c = 1
    lm = M.linearize(cfg, M.OperatingPoint(ring=0), latency_ticks=lat_c)
    pending = {k_edge + lat_c: np.array(rig.plant.heater.applied_codes) + 400}
    seen = None
    for i in range(80):
        k = rig.k
        if k % spec.period_ticks == 0 and k > k_edge:
            sensors, _ = rig.plant.release(k)
            cnt = sensors["tx.monitor.count"][0, 0]
            st = sensors["tx.monitor.sample_ticks"][0, 0, :cnt]
            k_act = k_edge + lat_c + spec.heater_dac.latency_ticks
            if seen is None and np.any(st >= k_act + 1):
                seen = k
        rig.plant.tick(k, heater_cmd=pending.pop(k, None))
        rig.k += 1
    want = timing.first_response_tick(k_edge, lat_c, spec.heater_dac.latency_ticks, spec.adc.period_ticks, 0,
                                      spec.adc.latency_ticks, spec.period_ticks, 0)
    assert seen == want
    assert lm.tau_u_ticks == lat_c + spec.heater_dac.latency_ticks
    # the first sample that sees the step is at most tau_u + tau_m + (period + adc period)/2 after the edge
    assert want - k_edge <= lm.delay_exact_ticks + 0.5 * (spec.period_ticks + spec.adc.period_ticks) + 1


def test_supply_limits_decide_reachability():
    """07 P1-12: the array heater supply limits total power (200 mW) AND total current (150 mA) in g26. A world whose
    oracle point needs more is not reachable: the plant droops every heater by the same factor
    (electrical.supply_limit_row), so a warm start there is not a steady state. models.static_point predicts the
    plant's droop factor (StaticPoint.supply_scale), and mrm.validation.oracle.reachable (which Rig.feasible_worlds
    uses) rejects such worlds. In this LHS draw two worlds are within the power limit but over the current limit (the
    controllers_c builder's mismatch report counted them as reachable and their folds as controller failures)."""
    from mrm.validation.oracle import reachable
    cfg = load_cfg()
    pp = realize(cfg, 6, uncertainty="lhs", variation="sample", seeds=11)
    rig = build_rig(cfg, pp=pp)
    rig.plant.tick(rig.k)                    # one tick with the warm-start codes: the plant's droop factor
    rig.k += 1
    scale_plant = np.asarray(rig.plant.last["heater"].supply_scale, float).reshape(-1)
    scale_model = np.array([sp.supply_scale for sp in rig.static])
    p_max = float(pp.heater.supply_power_max_w)
    over_i_only = [b for b, sp in enumerate(rig.static)
                   if np.sum(sp.heater_power_w) <= p_max and sp.supply_scale < 1.0]
    print(f"\nsupply droop: model {np.round(scale_model, 4)}, plant {np.round(scale_plant, 4)}; within the power limit "
          f"but over the current limit: worlds {over_i_only}; feasible {rig.feasible_worlds()}")
    np.testing.assert_allclose(scale_plant, scale_model, rtol=5e-3)
    assert len(over_i_only) >= 1
    for b in over_i_only:
        assert not np.any(reachable(pp, rig.static[b])) and not rig.feasible_worlds()[b]
