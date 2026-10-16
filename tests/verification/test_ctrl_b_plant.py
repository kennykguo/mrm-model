"""Closed-loop test bench for the controllers-B baselines (ramp acquisition, WDM supervisor, bit statistics, factory
calibration), plus checks of the bench itself.

:class:`CtrlBPlant` extends ``tests/verification/test_plant_wiring.py::ReferencePlant`` (the Python reference plant
tick of docs/handoff/ENGINE_BRIEF.md) with what a closed-loop controller test needs and the reference plant leaves
out:

- the ``datapath.bitstat_threshold`` actuator reaches the bit-statistics sampler (the reference plant passes 0);
- ``tx.assignment`` writes are recorded as the declared ring-to-line map (07 P1-12);
- an optional laser turn-on ramp (07 P0-1 item 6, ``sim.reset_protocol: laser_ramp``) and aggressor calibration
  (07 P1-4) before the scenario is compiled;
- Truth, for the TEST only: the resonance-minus-line detuning of every ring against every line, the OMA at the
  declared line and the static OMA optimum (:meth:`CtrlBPlant.truth`). Controllers never see it.

:func:`run_closed_loop` steps a controller through the plant in the 07 P0-6 event order (release the sensors at an
edge, step the controller, validate and latch its actions, then advance the plant interval) and records Truth.
:func:`lock_metrics` reduces a record to the acceptance numbers of 07 4.8: lock error (pm), OMA penalty (dB), lock
excursion count (5 pm threshold, 1 pm hysteresis, metrics section of g26) and time to lock.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pytest

from mrm import params
from mrm.control.api import ActionValidator, ControllerSpec, Observation, controller_telemetry
from mrm.plant import optics as op
from mrm.plant import phase as ph
from mrm.scenarios import calibrate_scenario, foster_from_system, stack
from mrm.scenarios import generators as g
from mrm.scenarios import library as lib
from mrm.units import C0

from test_plant_wiring import ReferencePlant

TWO_PI = 2.0 * math.pi


def load_g26(overrides=(), knowledge=None):
    """g26 config with the controller knowledge extended (tests only).

    ``knowledge``: tokens added to ``controller_interface.knowledge`` (mrm.control.calibration.KNOWLEDGE_TOKENS)."""
    ov = list(overrides)
    if knowledge is not None:
        ov.append("controller_interface.knowledge=[" + ",".join(knowledge) + "]")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # artefact fallbacks warn until the extraction lands
        return params.load("g26", overrides=ov)


def realize(cfg, batch=1, *, nominal=False, seeds=7):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if nominal:
            return params.realize(cfg, batch, seeds=seeds, uncertainty="nominal", variation="off")
        return params.realize(cfg, batch, seeds=seeds)


def make_spec(cfg, batch, scenario=None):
    drives = scenario.drives if scenario is not None else ()
    offsets = scenario.offsets if scenario is not None else ()
    return ControllerSpec.from_config(cfg, n_batch=batch, nominal=cfg, scenario_drives=drives,
                                      scenario_offsets=offsets)


def laser_ramp(ramp_s: float) -> g.Scenario:
    """Lines ramp from 0 to full power over ``ramp_s`` (07 P0-1 item 6 laser_ramp reset protocol)."""
    return g.Scenario(name="laser_ramp", components=(
        g.pwl("laser.line_enable", [0.0, ramp_s], [0.0, 1.0], absolute=True, label="laser_ramp"),))


def calibrated(sc: g.Scenario, pp, *, ring_t0_k=None) -> g.Scenario:
    """Calibrate the pending aggressors of ``sc`` on this plant (ENGINE_BRIEF section 1)."""
    if not sc.pending_aggressors:
        return sc
    sysm = pp.thermal.system
    model, resp = {}, {}
    for a in sc.aggressors:
        m = {"s0_m_per_k": pp.thermal.dlambda_dt_s0_m_per_k[:, a.victim_ring],
             "beta_per_k": pp.thermal.dlambda_dt_beta_per_k[:, 0], "t_ref_k": pp.thermal.dlambda_dt_t0_k[:, 0]}
        if ring_t0_k is not None:
            m["ring_t0_k"] = np.asarray(ring_t0_k, float).reshape(-1)
        model[a.site] = m
        resp[a.site] = foster_from_system(sysm, victim_ring=a.victim_ring, site=a.site)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc, _ = calibrate_scenario(sc, resp, model=model)
    return sc


def delay_aggressors(sc: g.Scenario, t0_s: float) -> g.Scenario:
    """Start every calibrated aggressor power waveform at t0_s (its own time axis starts there, 0 before)."""
    import dataclasses
    aggs = tuple(dataclasses.replace(a, power=tuple(dataclasses.replace(c, t0_s=float(t0_s)) for c in a.power))
                 for a in sc.aggressors)
    return dataclasses.replace(sc, aggressors=aggs)


class CtrlBPlant(ReferencePlant):
    """ReferencePlant + threshold actuator, declared assignment, laser ramp and Truth (module docstring)."""

    def __init__(self, pp, cfg, spec, *, scenario=None, noise_seed: int = 11, scenario_seed: int = 5,
                 laser_ramp_s: float | None = None, aggressor_delay_s: float | None = None):
        sc = lib.static() if scenario is None else scenario
        if laser_ramp_s:
            sc = stack([sc, laser_ramp(laser_ramp_s)])
        sc = calibrated(sc, pp)
        if aggressor_delay_s:
            sc = delay_aggressors(sc, aggressor_delay_s)
        super().__init__(pp, cfg, scenario=sc, noise_seed=noise_seed, scenario_seed=scenario_seed,
                         pop_period=spec.period_ticks)
        self.spec = spec
        self.thr = np.zeros((self.B, self.N), np.int64)
        r2l = np.asarray(pp.layout.ring_to_line, np.int64).reshape(-1)
        self.assignment = np.repeat(r2l[None, :], self.B, axis=0)
        self.lam = C0 / np.asarray(pp.laser.line_freq_hz, float)
        self.v_lo = np.zeros((self.B, self.N))
        self.v_hi = np.zeros((self.B, self.N))

    def tick(self, k, heater_cmd=None):
        out = super().tick(k, heater_cmd)
        self.lam = out["lam"]
        self.v_lo, self.v_hi = out["v_lo"], out["v_hi"]
        return out

    def release(self, k: int):
        rel = self.mon.release(k)
        sensors, new = self.mon.channels(rel)
        s2, n2 = self.temps.release(k)
        sensors.update(s2)
        new.update(n2)
        last = self.last
        if last:                     # no window before the first plant interval
            self.dp.bitstat_from_monitor(k, self.mon, last["out_r"][:, op.R_DROP_SUM1],
                                         last["out_r"][:, op.R_DROP_SUM0], self.thr, density=last["dens"],
                                         rin_per_hz=np.asarray(self.pp.laser.rin_per_hz).reshape(-1))
        s3, n3 = self.dp.release(k)
        sensors.update(s3)
        new.update(n3)
        sensors["tx.heater.applied"] = self.heater.applied_codes.astype(np.uint16)
        sensors["tx.bias.applied"] = self.bias.applied_codes.astype(np.uint16)
        new["tx.heater.applied"] = np.ones((self.B, self.N), bool)
        new["tx.bias.applied"] = np.ones((self.B, self.N), bool)
        return sensors, new

    # ---- Truth (tests only) -----------------------------------------------------------------------------------
    def detuning_m(self) -> np.ndarray:
        """(B, N, L) resonance-minus-line detuning (m) of the nearest order, mean of the two NRZ levels, at the
        current ring temperatures. Positive = resonance red of the line (phase.py: phi = 2 pi m on the line and
        d phi / d delta = 2 pi n_g L / x^2 > 0)."""
        out = np.zeros((self.B, self.N, self.L))
        for b in range(self.B):
            o = self._optics_inputs(b, self.lam[b], self.v_lo[b], self.v_hi[b])
            dpl = TWO_PI * C0 * o["trt"] / (self.lam[b][None, :] ** 2)
            # mean phase of the two levels first (levels split by << FSR), then wrap: wrapping each level and
            # averaging gives 0 at the +-FSR/2 wrap
            d1 = o["phi1"] - o["phi0"]
            d1 = d1 - TWO_PI * np.round(d1 / TWO_PI)
            phi = o["phi0"] + 0.5 * d1
            d = phi - TWO_PI * np.round(phi / TWO_PI)
            out[b] = d / dpl
        return out

    def oma_curve(self, b: int, n: int, line: int, dx_m: np.ndarray) -> np.ndarray:
        """Static through-port OMA (W) of ring n at ``line`` for resonance offsets dx_m (m) added to the current
        state, with this ring's present input power, levels and couplings (other rings fixed)."""
        o = self._optics_inputs(b, self.lam[b], self.v_lo[b], self.v_hi[b])
        dpl = TWO_PI * C0 * o["trt"][n, line] / self.lam[b][line] ** 2
        p_in = float(self.last["out_rl"][b, op.RL_PIN, n, line])
        r1 = op.ring_exact(o["phi1"][n, line] + dpl * dx_m, o["k2t"][n, line], o["k2d"][n, line], o["a1"][n, line],
                           o["trt"][n, line])
        r0 = op.ring_exact(o["phi0"][n, line] + dpl * dx_m, o["k2t"][n, line], o["k2d"][n, line], o["a0"][n, line],
                           o["trt"][n, line])
        return p_in * np.abs(r1.p_thru - r0.p_thru)

    def oma_penalty_db(self, assignment=None, span_fwhm: float = 3.0, n_grid: int = 241):
        """(B, N) OMA penalty (dB) of each ring at its declared line against the best static OMA over resonance
        offsets within +-span_fwhm FWHM (unrestricted) and over red-side offsets only (resonance red of the line,
        the self-heating-stable side). Returns (penalty_any, penalty_red)."""
        asg = self.assignment if assignment is None else assignment
        pen = np.full((self.B, self.N), np.nan)
        pen_red = np.full((self.B, self.N), np.nan)
        det = self.detuning_m()
        fwhm = np.asarray(self.pp.optics.fwhm_hz, float) * (self.lam[:, :1] ** 2) / C0
        for b in range(self.B):
            for n in range(self.N):
                m = int(asg[b, n])
                if m < 0:
                    continue
                w = float(np.broadcast_to(fwhm[b], (self.N,))[n]) if fwhm.ndim else float(fwhm)
                dx = np.linspace(-span_fwhm * w, span_fwhm * w, n_grid)
                oma = self.oma_curve(b, n, m, dx)
                now = self.oma_curve(b, n, m, np.zeros(1))[0]
                best = float(oma.max())
                red = oma[(det[b, n, m] + dx) >= 0.0]
                best_red = float(red.max()) if red.size else np.nan
                pen[b, n] = 10.0 * math.log10(best / max(now, 1e-30))
                pen_red[b, n] = 10.0 * math.log10(best_red / max(now, 1e-30)) if red.size else np.nan
        return pen, pen_red

    def truth(self, assignment=None) -> dict:
        asg = self.assignment if assignment is None else assignment
        det = self.detuning_m()
        idx = np.clip(asg, 0, self.L - 1)
        d_asg = np.take_along_axis(det, idx[:, :, None], axis=2)[:, :, 0]
        d_asg = np.where(asg >= 0, d_asg, np.nan)
        nearest = np.argmin(np.abs(det), axis=2)
        return {"det_m": det, "det_assigned_m": d_asg, "nearest_line": nearest,
                "nearest_det_m": np.take_along_axis(det, nearest[:, :, None], axis=2)[:, :, 0],
                "oma_w": np.abs(self.last["oma"]), "p_heater_w": self.u_heater.copy(),
                "t_ring_k": self.t_ring.copy(), "heater_code": self.heater.applied_codes.astype(np.int64)}


class RefBench:
    """A factory test station (mrm.control.calibration.CalBench) over :class:`CtrlBPlant`: ``advance(inputs, n)``
    then ``sample_sensors()`` returns the sensors released at the end tick (a controller edge when n is the period)."""

    def __init__(self, pp, cfg, spec, scenario=None, **kw):
        self.plant = CtrlBPlant(pp, cfg, spec, scenario=scenario, **kw)
        self.spec = spec
        self.k = 0
        self._last = None

    def advance(self, inputs=None, n_ticks: int = 1):
        cmd = None if not inputs else inputs.get("tx.heater")
        for i in range(int(n_ticks)):
            self.plant.tick(self.k, None if (cmd is None or i) else np.asarray(cmd, np.int64))
            self.k += 1
        self._last = self.plant.release(self.k)

    def sample_sensors(self):
        return self._last


class RefBenchFactory:
    """``bench_factory(scenario) -> RefBench`` with the spec attached (factory_cal builds the station scenario)."""

    def __init__(self, pp, cfg, spec, **kw):
        self.pp, self.cfg, self.spec, self.kw = pp, cfg, spec, kw

    def __call__(self, scenario):
        return RefBench(self.pp, self.cfg, self.spec, scenario, **self.kw)


@dataclass
class Record:
    """Per-tick Truth and per-edge controller data of a closed-loop run (tests only)."""
    ticks: np.ndarray
    det_m: np.ndarray                 # (T, B, N) detuning against the declared line
    nearest: np.ndarray               # (T, B, N) nearest line index
    heater: np.ndarray                # (T, B, N) applied heater code
    p_heater_w: np.ndarray            # (T, B, N)
    edge_ticks: list = field(default_factory=list)
    telemetry: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    monitor: list = field(default_factory=list)
    assignment: list = field(default_factory=list)


def run_closed_loop(plant: CtrlBPlant, ctrl, n_ticks: int, *, k0: int = 0, record_every: int = 1,
                    reset: bool = True, validator: ActionValidator | None = None, keep_monitor: bool = False):
    """Run ``ctrl`` on ``plant`` for ticks k0 .. k0 + n_ticks - 1 (07 P0-6 order). Returns (Record, validator).

    At an edge k (k % period == phase): release the sensors of the intervals before k, step the controller, validate
    (clip and count, 07 P1-19) and latch at k + ctrl.latency_ticks: tx.heater goes to the heater DAC path,
    datapath.bitstat_threshold to the sampler of the next edge, tx.assignment to the declared map. Then the plant
    advances [k, k + 1)."""
    spec = plant.spec
    if reset:
        ctrl.reset(spec)
    val = ActionValidator(spec) if validator is None else validator
    lat = int(getattr(ctrl, "latency_ticks", 0))
    pending: list[tuple[int, dict]] = []
    n_rec = (n_ticks + record_every - 1) // record_every
    B, N = plant.B, plant.N
    rec = Record(ticks=np.zeros(n_rec, np.int64), det_m=np.zeros((n_rec, B, N)), nearest=np.zeros((n_rec, B, N), int),
                 heater=np.zeros((n_rec, B, N), np.int64), p_heater_w=np.zeros((n_rec, B, N)))
    period, phase = spec.period_ticks, spec.phase_ticks
    j = 0
    for k in range(k0, k0 + n_ticks):
        heater_cmd = None
        if k >= phase and (k - phase) % period == 0:
            sensors, new = plant.release(k)
            obs = Observation(k, spec.tick_s, {c: sensors[c] for c in spec.sensors},
                              {c: new[c] for c in spec.sensors})
            act = val.validate(ctrl.step(obs) or {})
            pending.append((k + lat, act))
            rec.edge_ticks.append(k)
            rec.telemetry.append({kk: np.array(v, float) for kk, v in controller_telemetry(ctrl).items()})
            rec.actions.append({kk: np.array(v) for kk, v in act.items()})
            if keep_monitor:
                rec.monitor.append(np.array(sensors["tx.monitor.latest"], float))
        while pending and pending[0][0] <= k:
            _, act = pending.pop(0)
            if "tx.heater" in act:
                heater_cmd = act["tx.heater"].astype(np.int64)
            if "datapath.bitstat_threshold" in act:
                plant.thr = act["datapath.bitstat_threshold"].astype(np.int64)
            if "tx.assignment" in act:
                plant.assignment = act["tx.assignment"].astype(np.int64)
        plant.tick(k, heater_cmd)
        if (k - k0) % record_every == 0:
            t = plant.truth()
            rec.ticks[j] = k + 1
            rec.det_m[j] = t["det_assigned_m"]
            rec.nearest[j] = t["nearest_line"]
            rec.heater[j] = t["heater_code"]
            rec.p_heater_w[j] = t["p_heater_w"]
            rec.assignment.append(plant.assignment.copy())
            j += 1
    return rec, val


def excursions(x_pm: np.ndarray, thr_pm: float = 5.0, hyst_pm: float = 1.0) -> int:
    """Lock-excursion count with hysteresis (07 P0-8): entering |x| > thr counts once; leaving needs
    |x| < thr - hyst."""
    out = 0
    above = False
    for v in np.abs(np.asarray(x_pm, float)):
        if not above and v > thr_pm:
            above = True
            out += 1
        elif above and v < thr_pm - hyst_pm:
            above = False
    return out


def lock_metrics(rec: Record, *, t_from: int, lock_pm: float = 5.0, hyst_pm: float = 1.0, settle_ticks: int = 0):
    """Acceptance numbers per (B, N) over ticks >= t_from: rms and max lock error (pm), excursion count, and the
    time to lock (first tick after which |detuning| stays below lock_pm for settle_ticks ticks; -1 = never)."""
    sel = rec.ticks >= t_from
    d = rec.det_m[sel] * 1e12
    rms = np.sqrt(np.nanmean(d ** 2, axis=0))
    mx = np.nanmax(np.abs(d), axis=0)
    B, N = d.shape[1], d.shape[2]
    exc = np.zeros((B, N), int)
    for b in range(B):
        for n in range(N):
            exc[b, n] = excursions(d[:, b, n], lock_pm, hyst_pm)
    t_lock = np.full((B, N), -1, np.int64)
    dall = rec.det_m * 1e12
    for b in range(B):
        for n in range(N):
            bad = np.flatnonzero(~(np.abs(dall[:, b, n]) < lock_pm))
            if bad.size == 0:
                t_lock[b, n] = rec.ticks[0]
            elif bad[-1] + 1 < len(rec.ticks) and rec.ticks[-1] - rec.ticks[bad[-1] + 1] >= settle_ticks:
                t_lock[b, n] = rec.ticks[bad[-1] + 1]
    return {"rms_pm": rms, "max_pm": mx, "excursions": exc, "t_lock_ticks": t_lock}


# ---------------------------------------------------------------------------------------------------------------
# checks of the bench itself
# ---------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nominal_bench():
    cfg = load_g26()
    pp = realize(cfg, 1, nominal=True)
    spec = make_spec(cfg, 1)
    return cfg, pp, spec


def test_truth_detuning_sign_and_scale(nominal_bench):
    """Heating a ring moves its resonance red: the Truth detuning grows by s(T) dT, and the cold nominal detuning
    against the ring's own line is minus the 7.6 nm design offset (modulo the FSR)."""
    cfg, pp, spec = nominal_bench
    dark = g.Scenario(name="dark", components=(g.const("laser.line_enable", 0.0, absolute=True),
                                               g.const("data.driver_enable", 0.0, absolute=True)))
    plant = CtrlBPlant(pp, cfg, spec, scenario=dark)
    plant.tick(0)
    d0 = plant.detuning_m()[0]
    off = np.asarray(pp.optics.resonance_offset_m).reshape(-1)[0]
    r2l = np.asarray(pp.layout.ring_to_line)
    fsr_m = np.asarray(pp.optics.fsr_hz).reshape(-1)[:plant.N] * plant.lam[0, r2l] ** 2 / C0
    own = d0[np.arange(plant.N), r2l]
    # nearest order: -7.6 nm, or FSR - 7.6 nm (the next order), whichever is closer to the line
    want = np.where(off < fsr_m / 2, -off, fsr_m - off)
    np.testing.assert_allclose(own, want, atol=0.05e-9)
    t0 = plant.t_ring.copy()
    code = np.full((1, plant.N), 20000, np.int64)
    for k in range(1, 200):
        plant.tick(k, code if k == 1 else None)
    d1 = plant.detuning_m()[0]
    s = np.asarray(pp.thermal.dlambda_dt_s0_m_per_k).reshape(-1)[:plant.N]
    dt = plant.t_ring[0] - t0[0]
    assert np.all(dt > 1.0)
    shift = (d1 - d0)[np.arange(plant.N), np.asarray(pp.layout.ring_to_line)]
    np.testing.assert_allclose(shift, s * dt, rtol=0.05)


def test_excursion_counter_hysteresis():
    x = np.array([0, 6, 4.5, 6, 3.9, 7, 0])
    assert excursions(x, 5.0, 1.0) == 2
