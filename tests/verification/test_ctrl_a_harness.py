"""Closed-loop test harness of the controllers_a baselines (and its own checks).

:class:`LoopPlant` extends ``test_plant_wiring.ReferencePlant`` (the integrator's Python plant tick, docs/handoff/
ENGINE_BRIEF.md) with what a closed loop needs: latched heater and bias registers written by the controller,
fast-kernel offsets added to them before the DAC FIFO (07 P0-7), the bias actuator under controller binding, a
serial bias-noise draw (the library's ``rng.fill_normal`` is a parallel kernel whose thread start-up costs ~0.3 ms
per call; the serial loop gives the same numbers), and Truth recorded per tick for the TESTS only: the detuning of
every ring to its assigned line (pm) and the OMA at the bus output.

:class:`ClosedLoop` runs a controller through the controller API exactly as the engine will (07 P0-6 event order:
release at the edge, ``step``, ``ActionValidator``, latch after ``latency_ticks``; the ``fast_kernel`` once per
released monitor sample, before ``step``). The controller never sees Truth: it gets an :class:`Observation` built
from the released sensor dictionaries only.

Cost: ~0.6-1 ms per plant tick (B = 1, N = L = 8) on the shared machine; runs are kept to a few thousand ticks
unless marked slow.
"""
from __future__ import annotations

import copy
import math
import warnings
from collections import deque
from dataclasses import dataclass

import numba as nb
import numpy as np
import pytest

from mrm import params
from mrm.control.api import ActionValidator, ControllerSpec, Observation, controller_fast_kernel, \
    controller_telemetry, fast_act_size, fast_obs_size, validate_telemetry
from mrm.core import rng
from mrm.core.timing import is_edge
from mrm.plant import electrical as el
from mrm.plant import optics as op
from mrm.plant import phase as ph
from mrm.plant import selfheat as shm
from mrm.plant import thermal as th
from mrm.plant.sensing import MonitorBank, fault_mask
from mrm.scenarios import registry as scen_registry
from mrm.units import C0
from test_plant_wiring import ReferencePlant

LOCK_THRESHOLD_PM = 5.0          # g26 metrics.lock_threshold_pm (the textbook 5 pm budget)
HYSTERESIS_PM = 1.0              # g26 metrics.hysteresis_pm


@nb.njit(cache=True)
def _fill_normal_serial(out, seeds, stream, tick, slot):
    """Serial twin of mrm.core.rng.fill_normal (same keys, same counters, same values)."""
    for b in range(out.shape[0]):
        key = rng.stream_key(seeds[b], stream)
        for i in range(out.shape[1]):
            out[b, i] = rng.normal(key, i, tick, slot)


def load_cfg(overrides=(), preset: str = "g26", alts=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if alts:
            return params.load(preset, alts=dict(alts), overrides=list(overrides))
        return params.load(preset, overrides=list(overrides))


def realize(cfg, batch: int = 1, *, nominal: bool = True, seeds: int = 0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if nominal:
            return params.realize(cfg, batch, seeds=seeds, uncertainty="nominal", variation="off")
        return params.realize(cfg, batch, seeds=seeds)


def make_spec(cfg, batch: int, scenario=None) -> ControllerSpec:
    sc = scen_registry.from_config(cfg) if scenario is None else scenario
    return ControllerSpec.from_config(cfg, n_batch=batch, nominal=cfg, scenario_drives=sc.drives,
                                      scenario_offsets=sc.offsets)


class LoopPlant(ReferencePlant):
    """ReferencePlant with controller-owned latched registers, fast-kernel offsets and per-tick Truth (tests only)."""

    def __init__(self, pp, cfg, *, scenario=None, noise_seed: int = 11, scenario_seed: int = 5,
                 period_ticks: int = 10, fast_monitor: bool = False, bias_owner: str = "scenario"):
        super().__init__(pp, cfg, scenario=scenario, noise_seed=noise_seed, scenario_seed=scenario_seed,
                         pop_period=period_ticks)
        B, N = self.B, self.N
        if fast_monitor:        # release every tick: the fast kernel sees each sample as it leaves the ADC FIFO
            self.mon = MonitorBank(self.mon.hw, self.noise_seeds, pop_period_ticks=1)
        self.fast_monitor = fast_monitor
        self.bias_owner = bias_owner
        self.scenario_seed = scenario_seed
        self.views = [op.selfheat_views(self.args_f[b], self.args_i[b]) for b in range(B)]
        self.lat_heater = np.zeros((B, N), np.int64)
        self.lat_bias = np.zeros((B, N), np.int64)
        self.off_heater = np.zeros((B, N), np.int64)
        self.off_bias = np.zeros((B, N), np.int64)
        self.h_lo, self.h_hi = int(pp.heater.code_min), int(pp.heater.code_max)
        self.b_hi = int(self.bias_p.code_max)
        self.sid_bias = np.int64(rng.stream_id("dac_bias"))
        self.line = np.asarray(pp.layout.ring_to_line)
        rp = self.phase.ring
        self.lam_res = rp[:, :, ph.RP_LAM_RES]
        self.fsr_m = rp[:, :, ph.RP_LAM_RES] ** 2 / (rp[:, :, ph.RP_N_G] * rp[:, :, ph.RP_LEN])
        self.k_next = 0
        self._ev_cache = {}
        self.delta_pm = np.zeros((B, N))
        self.oma_w = np.zeros((B, N))
        self.p_line_w = np.zeros((B, N))
        self.force_hot = False

    # -- scenario switch (tests change the disturbance on a locked snapshot) --
    def set_scenario(self, sc, *, seed: int | None = None):
        sysm = self.sys
        agg_names = tuple(sysm.src_names[i] for i in sysm.aggressor_src)
        self.sc = sc
        self.arr = sc.compile(dt_s=self.dt, n_rings=self.N, n_lines=self.L, batch=self.B,
                              seed=self.scenario_seed if seed is None else seed, aggressor_sites=agg_names)
        self.cache = self.arr.new_cache()
        self._ev_cache = {}

    def _eval(self, k):
        ev = self._ev_cache.get(k)
        if ev is None:
            ev = {n: v.copy() for n, v in self.arr.evaluate(int(k), self.cache).items()}
            self._ev_cache = {k: ev, **{kk: vv for kk, vv in self._ev_cache.items() if kk == k - 1}}
        return ev

    def tick(self, k: int, heater_cmd=None):
        """Plant interval [k, k+1) with the latched registers plus offsets (heater_cmd is ignored: use lat_heater)."""
        pp, sysm, B, N = self.pp, self.sys, self.B, self.N
        ev0 = self._eval(k)
        ev1 = self._eval(k + 1)
        f_hz, p_w = self.laser.tick(k, enable=ev0["laser.line_enable"], f_offset_hz=ev0["laser.dfreq_hz"],
                                    p_scale=10.0 ** (ev0["laser.power_delta_db"] / 10.0))
        p_chip = p_w * np.asarray(pp.laser.input_transmission) * 10.0 ** (-ev0["optical.input_loss_delta_db"] / 10.0)
        lam = C0 / f_hz
        rho = self.dp.pattern(k, density=ev0["data.ones_density"], traffic_on=ev0["data.driver_enable"])
        z = np.zeros((B, N))
        _fill_normal_serial(z, self.noise_seeds, self.sid_bias, np.int64(k), np.int64(0))
        if self.bias_owner == "controller":
            cmd_b = np.clip(self.lat_bias + self.off_bias, 0, self.b_hi)
        else:
            v_set = ev0["tx.bias_v"]
            cmd_b = self.bias_p.code_for_voltage(np.where(np.isfinite(v_set), v_set, self.bias_p.v_min_v))
            cmd_b = np.clip(cmd_b + self.off_bias, 0, self.b_hi)
        bout = self.bias.step(k, cmd=cmd_b, z=z, v_offset=ev0["tx.bias_offset_v"])
        swing = np.asarray(pp.datapath.swing_pp_v) * ev0["data.swing_scale"]
        v_lo, v_hi, fl = np.zeros((B, N)), np.zeros((B, N)), np.zeros((B, N), np.int8)
        cj_eff = np.zeros((B, N))
        r_s = np.zeros((B, N))
        for b in range(B):
            el.nrz_levels_row(bout.v_avg[b], swing[b], int(pp.datapath.level_convention[b, 0]),
                              el.JUNCTION_GRID_V_MIN, el.JUNCTION_GRID_V_MAX, v_lo[b], v_hi[b], fl[b])
            o = self._optics_inputs(b, lam[b], v_lo[b], v_hi[b])
            r_s[b] = 0.5 * (o["rs"][:, 0] + o["rs"][:, 1])
            (p_in, t2, t2o, p1, phi0r, phi1r, _, _, a0, a1, e0, e1, psi, k2t, k2d, trt, dpl, tref, s0, beta,
             *_rest) = self.views[b]
            p_in[:] = p_chip[b]
            p1[:] = rho[b]
            phi0r[:], phi1r[:] = o["phi0"], o["phi1"]
            a0[:], a1[:], e0[:], e1[:] = o["a0"], o["a1"], o["e0"], o["e1"]
            k2t[:], k2d[:], trt[:] = o["k2t"], o["k2d"], o["trt"]
            dpl[:] = 2.0 * math.pi * C0 * o["trt"] / (lam[b][None, :] ** 2)
            rp = self.phase.ring[b]
            tref[:] = self.t_ring[b]
            g = 1.0 + rp[:, ph.RP_BETA] * (self.t_ring[b] - rp[:, ph.RP_T0])        # phase.local_thermo_optic
            s0[:] = rp[:, ph.RP_S0] * g
            beta[:] = rp[:, ph.RP_BETA] / g
            r = self.phase.table_row(b)
            el.junction_swing_capacitance_row(v_lo[b], v_hi[b], self.phase.grid, self.phase.capacitance[r],
                                              rp[:, ph.RP_FILL], rp[:, ph.RP_LEN], cj_eff[b])
            op.selfheat_bounds(self.args_f[b], self.args_i[b], 50.0, self.p_max[b], self.d2p_max[b],
                               self.dpdt_max[b])
        u0 = np.zeros((B, sysm.n_src))
        pj, pd = np.zeros((B, N)), np.zeros((B, N))
        d = pp.datapath
        for b in range(B):
            en = (ev0["data.driver_enable"][b] >= 0.5).astype(float)
            el.data_activity_heat_row(cj_eff[b], v_hi[b] - v_lo[b], float(d.bitrate_hz[b, 0]),
                                      ev0["data.transition_density"][b], en, r_s[b],
                                      np.asarray(d.driver_output_resistance_ohm[b]),
                                      np.asarray(d.driver_static_power_w[b]),
                                      np.asarray(d.driver_energy_per_transition_j[b]), 0, pj[b], pd[b])
        u0[:, sysm.junction_src] = pj
        if sysm.driver_src.size:
            u0[:, sysm.driver_src] = pd[:, :sysm.driver_src.size]
        u0[:, sysm.heater_src] = self.u_heater
        u0[:, sysm.wg_src] = self.p_prev
        u1 = u0.copy()
        if sysm.aggressor_src.size:
            u0[:, sysm.aggressor_src] = ev0["thermal.aggressor_w"]
            u1[:, sysm.aggressor_src] = ev1["thermal.aggressor_w"]
        t_base_end = ev1["thermal.t_base_k"][:, 0] + ev1["thermal.xpu_rise_k"][:, 0]
        a, c0, c1 = self.coef
        pre = np.zeros((B, N))
        gg = np.zeros((B, N))
        for b in range(B):
            for n in range(N):
                pre[b, n], gg[b, n] = th.observer_split_b(self.x[b], a[b], c0[b], c1[b], sysm.lag_src, sysm.obs_ptr,
                                                          int(sysm.heater_obs[n]), int(sysm.heater_src[n]), u0[b],
                                                          u1[b], self.src_scale[0])
        cmd_h = np.clip(self.lat_heater + self.off_heater, self.h_lo, self.h_hi)
        hout = self.heater.step(k, cmd_h, None, t_base_end[:, None] + pre, gg)
        self.u_heater = hout.p_w.copy()
        u0[:, sysm.heater_src] = self.u_heater
        u1[:, sysm.heater_src] = self.u_heater
        if self.force_hot:
            self.t_branch[:] = np.inf
            self.force_hot = False
        n_fail = shm.implicit_step(self.x, a, c0, c1, sysm.lag_src, sysm.lag_obs, sysm.lag_kind, sysm.opt_lags,
                                   sysm.obs_ptr, sysm.src_ring, sysm.ring_obs, np.asarray(pp.layout.bus_order), u0,
                                   u1, self.src_scale, t_base_end, self.p_prev, self.t_branch, op.selfheat_pabs,
                                   op.selfheat_commit, self.args_f, self.args_i, self.p_max, self.d2p_max, 10, 1e-9,
                                   1, self.g_work, self.t_obs, self.p_new, self.t_new, self.n_iter)
        self.p_prev[:] = self.p_new
        self.t_branch[:] = self.t_new
        self.t_ring[:] = self.t_new
        self.t_heater[:] = self.t_obs[:, sysm.heater_obs]
        out_r = np.stack([self.views[b][23] for b in range(B)])
        out_rl = np.stack([self.views[b][22] for b in range(B)])
        fault = fault_mask(B, N, adc_drop=ev0["fault.adc_drop"], adc_stuck=ev0["fault.adc_stuck"],
                           adc_saturate=ev0["fault.adc_saturate"], dac_glitch_codes=ev0["fault.dac_glitch_codes"])
        p_bus = out_rl[:, op.RL_PIN].sum(axis=2)
        self.mon.tick(k, p_pd_w=out_r[:, op.R_DROP_SUM], p_bg_w=p_bus, t_pd_k=self.t_obs[:, sysm.pd_obs],
                      fault=fault)
        ar = np.arange(N)
        oma = out_rl[:, op.RL_POUT1, ar, self.line] - out_rl[:, op.RL_POUT0, ar, self.line]
        self.dp.errors(k, np.abs(oma))
        self.temps.tick(k, self.t_obs[:, sysm.temp_obs])
        # ---- Truth (tests only): detuning of each ring to its own line at the end of the interval, and OMA
        shift = np.zeros((B, N))
        for b in range(B):
            r = self.phase.table_row(b)
            sh_lv = np.zeros(N)
            fal, cj, rs, flg = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N, np.int8)
            ph.resonance_shift_row(self.phase.ring[b], self.t_ring[b], v_lo[b], self.phase.grid, self.phase.dneff[r],
                                   self.phase.alpha_fca[r], self.phase.capacitance[r], self.phase.r_series[r], sh_lv,
                                   fal, cj, rs, flg)
            sh_hi = np.zeros(N)
            ph.resonance_shift_row(self.phase.ring[b], self.t_ring[b], v_hi[b], self.phase.grid, self.phase.dneff[r],
                                   self.phase.alpha_fca[r], self.phase.capacitance[r], self.phase.r_series[r], sh_hi,
                                   fal, cj, rs, flg)
            shift[b] = (1.0 - rho[b]) * sh_lv + rho[b] * sh_hi
        lam_line = lam[:, self.line]
        dlt = self.lam_res + shift - lam_line
        dlt = dlt - self.fsr_m * np.round(dlt / self.fsr_m)
        self.delta_pm = dlt * 1e12
        self.oma_w = np.abs(oma)
        self.p_line_w = p_chip[:, self.line]
        u_ss = u1.copy()
        u_ss[:, sysm.wg_src] = self.p_new
        self.last = dict(p_chip=p_chip, lam=lam, rho=rho, bias=bout, heater=hout, n_fail=n_fail, out_r=out_r,
                         out_rl=out_rl, v_lo=v_lo, v_hi=v_hi, u_ss=u_ss)
        self.k_next = k + 1
        return self.last

    def release(self, k: int):
        sensors, new = super().release(k)
        sensors["tx.heater.applied"] = self.lat_heater.astype(np.uint16)
        sensors["tx.bias.applied"] = self.lat_bias.astype(np.uint16)
        return sensors, new


# ---- scenarios used by the controller tests --------------------------------------------------------------------------

def stacked(cfg, *extra, t_base_c: float | None = None):
    """The config's own scenario (bias default, ambient) stacked with ``extra`` scenarios; optional base temperature
    default (degC)."""
    from mrm.scenarios import generators as g
    sc = scen_registry.from_config(cfg)
    if extra:
        sc = g.stack([sc, *extra], name="+".join([sc.name or "base"] + [e.name for e in extra]))
    if t_base_c is not None:
        sc = sc.with_defaults(**{"thermal__t_base_k": t_base_c + 273.15})
    return sc


def calibrated(pp, sc):
    """Calibrate every pending aggressor of ``sc`` on the plant's own thermal system (Truth-side, as the engine does:
    docs/handoff/ENGINE_BRIEF.md section 1, ring_t0 = the ring lock temperature, scenarios handoff section 8)."""
    from mrm.scenarios.calibrate import calibrate_scenario, foster_from_system
    if not sc.pending_aggressors:
        return sc
    t = pp.thermal
    model = {a.site: {"s0_m_per_k": t.dlambda_dt_s0_m_per_k[:, a.victim_ring], "beta_per_k": t.dlambda_dt_beta_per_k[:, 0],
                      "t_ref_k": t.dlambda_dt_t0_k[:, 0], "ring_t0_k": t.t_ring_lock_k[:, a.victim_ring]}
             for a in sc.aggressors}
    resp = {a.site: foster_from_system(t.system, victim_ring=a.victim_ring, site=a.site) for a in sc.aggressors}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out, _ = calibrate_scenario(sc, resp, model=model)
    return out


def a1_rate_scenario(cfg, pp, t_min_s: float, *, delta_t_k: float = 20.0, t_base_c: float = 25.0,
                     victim_ring: int = 0, full: bool = False):
    """Slide-10 aggressor (A1, rectified sine, resonance referencing) whose minimum falls at plant time ``t_min_s``.
    ``full=False``: the A1 peak rate (pi dT/P = 2127 degC/s, 183 pm/ms of resonance target) with a reduced swing
    ``delta_t_k`` (period pi dT / 2127 s) so the cusp arrives within ~15 ms; ``full=True``: the printed 88.5 K /
    130.7 ms trajectory on the base ``t_base_c``."""
    from mrm.scenarios import library as lib
    if full:
        period, dT = lib.A1_PERIOD_S, lib.A1_DELTA_T_K
    else:
        dT = float(delta_t_k)
        period = math.pi * dT / (math.pi * lib.A1_DELTA_T_K / lib.A1_PERIOD_S)
    agg = lib.a1_slide10_aggressor(victim_ring=victim_ring, t_max_c=t_base_c + dT + 0.1, delta_t_k=dT,
                                   period_s=period, t_base_c=t_base_c, phase_s=period / 2.0 - t_min_s)
    return calibrated(pp, stacked(cfg, agg, t_base_c=t_base_c)), period


def a2_rate_scenario(cfg, pp, t_min_s: float, *, delta_t_k: float = 20.0, t_base_c: float = 25.0,
                     victim_ring: int = 0, full: bool = False):
    """The 808 degC/s run (A2, symmetric triangle reading: 1616 degC/s edges) with its minimum at ``t_min_s``.
    ``full=False``: same edge rate, swing ``delta_t_k`` (period 2 dT / 1616 s); ``full=True``: 80.8 K / 100 ms."""
    from mrm.scenarios import library as lib
    rate = 2.0 * lib.A2_RATE_K_PER_S          # edge rate of the symmetric-triangle reading
    dT = lib.A2_RATE_K_PER_S * lib.A2_PERIOD_S if full else float(delta_t_k)
    period = 2.0 * dT / rate
    agg = lib.a2_808_aggressor(victim_ring=victim_ring, reading="symmetric_triangle", t_low_c=t_base_c,
                               t_high_c=t_base_c + dT, period_s=period, t_base_c=t_base_c)
    # the triangle starts at its minimum at t = 0; shift it so the minimum falls at t_min_s
    tgt = agg.aggressors[0].target
    import dataclasses
    tgt = dataclasses.replace(tgt, phase_s=float(tgt.phase_s) - t_min_s)
    agg = agg.replace(aggressors=(dataclasses.replace(agg.aggressors[0], target=tgt),))
    return calibrated(pp, stacked(cfg, agg, t_base_c=t_base_c)), period


def laser_power_step_scenario(cfg, t_s: float, delta_db: float):
    """Every line's power steps by ``delta_db`` at ``t_s`` (07 4.8 power_at_ring, 04 C3 failure mode)."""
    from mrm.scenarios import generators as g
    st = g.Scenario(name="laser_power_step", components=(g.step("laser.power_delta_db", t_s, delta_db,
                                                                 label="power_step", provenance="assumed"),))
    return stacked(cfg, st)


# ---- test-side placement (Truth-guided; stands in for an acquisition / factory warm start) ------------------------

def settle_slow_states(plant: LoopPlant) -> None:
    """Set every thermal lag to its steady state for the current source powers (heater, absorbed, junction, driver,
    aggressor) so the ms crosstalk and package-tail poles start settled (test set-up only)."""
    plant.x[:] = plant.sys.steady_state(plant.last["u_ss"])


def place_on_hot_branch(plant: LoopPlant, delta_pm: float = 170.0, n_ticks: int = 3000, gain: float = 0.03,
                        ramp_codes: float = 8.0, margin_pm: float = 40.0, settle_at=(150, 400, 900, 1600)):
    """Put every ring on the stable (red) branch of its own line at detuning ``delta_pm`` using Truth: a heater guess
    on the red side (R1 without beta, Z_heater_ring and the cold heater law from PlantParams: it overshoots red by
    the neglected self-heating, data heat and beta), the hot branch forced for the first tick, a down-ramp of
    ``ramp_codes`` per tick on the rings still more than ``margin_pm`` red of the target, then a Truth integrator.
    This is the TEST's acquisition (it stands in for an acquisition controller or a factory warm start);
    controllers never see it. Returns the final detuning (B, N) in pm."""
    pp = plant.pp
    B, N = plant.B, plant.N
    lam_line = C0 / np.asarray(pp.laser.line_freq_hz, float)[:, plant.line]
    s0 = pp.thermal.dlambda_dt_s0_m_per_k * np.ones((B, N))
    t_base = plant._eval(plant.k_next)["thermal.t_base_k"][:, :1]                  # (B, 1) K
    need_m = lam_line - plant.lam_res - s0 * (t_base - plant.phase.ring[:, :, ph.RP_T_REF])
    p_w = need_m / s0 / (pp.thermal.z_heater_ring_k_per_w * np.ones((B, N)))
    h = pp.heater
    full = float((1 << int(h.dac_bits)) - 1)
    v = np.sqrt(np.maximum(p_w, 0.0) * h.r_cold_ohm * 1.2)                    # 1.2: rough TCR allowance
    code = np.clip(np.rint(v / h.full_scale_v * full), plant.h_lo, plant.h_hi).astype(np.int64)
    plant.lat_heater[:] = code
    plant.force_hot = True
    k0 = plant.k_next
    acc = code.astype(float)
    ramping = np.ones((B, N), bool)
    for k in range(k0, k0 + n_ticks):
        plant.tick(k)
        if k - k0 in settle_at:
            settle_slow_states(plant)
        ramping &= plant.delta_pm > delta_pm + margin_pm
        acc = np.where(ramping, acc - ramp_codes, acc + gain * (delta_pm - plant.delta_pm))
        acc = np.clip(acc, plant.h_lo, plant.h_hi)
        plant.lat_heater[:] = np.rint(acc).astype(np.int64)
    return plant.delta_pm.copy()


_CACHE: dict = {}


BIAS_CONTROLLER = ("controller_interface.bind.tx.bias=controller", "scenario.bias_v=null")


def placed(overrides=(), *, batch: int = 1, nominal: bool = True, seeds: int = 0, delta_pm: float = 170.0,
           n_ticks: int = 2500, bias_v: float | None = None, power_delta_db: float = 0.0, preset: str = "g26",
           alts=None):
    """(cfg, pp, spec, plant): a g26 plant (with ``overrides``) whose rings sit on the stable branch at
    ``delta_pm``, built once per pytest session and returned as a fresh snapshot (the cache lives in this module,
    which pytest imports once). ``bias_v``: the controller owns tx.bias (BIAS_CONTROLLER overrides added) and the
    latched bias register holds this voltage during the placement. ``power_delta_db``: every line's power offset by a
    constant scenario component (laser.power_per_line_dbm has a printed range of 8-11 dBm, so a low-power plant is
    made by the scenario). ``preset``: a corner preset of the uncertainty set (07 4.8), e.g. g26_slow_thermal;
    ``alts``: pinned alternatives, e.g. {"uncertainty.timing.latency_scale": "high"}."""
    if bias_v is not None:
        overrides = tuple(overrides) + BIAS_CONTROLLER
    key = (tuple(overrides), batch, nominal, seeds, delta_pm, n_ticks, bias_v, power_delta_db, preset,
           tuple(sorted((alts or {}).items())))
    if key not in _CACHE:
        cfg = load_cfg(overrides, preset, alts)
        pp = realize(cfg, batch, nominal=nominal, seeds=seeds)
        spec = make_spec(cfg, batch)
        plant = LoopPlant(pp, cfg, period_ticks=spec.period_ticks)
        if power_delta_db:
            from mrm.scenarios import generators as g
            low = g.Scenario(name="power_offset", components=(g.const("laser.power_delta_db", float(power_delta_db),
                                                                      label="power_offset", provenance="assumed"),))
            plant.set_scenario(stacked(cfg, low))
        if bias_v is not None:
            plant.bias_owner = "controller"
            plant.lat_bias[:] = plant.bias_p.code_for_voltage(np.full((batch, pp.n_rings), bias_v))
        place_on_hot_branch(plant, delta_pm=delta_pm, n_ticks=n_ticks)
        _CACHE[key] = (cfg, pp, spec, plant)
    cfg, pp, spec, plant = _CACHE[key]
    return cfg, pp, spec, snapshot(plant)


def use_fast_monitor(plant: LoopPlant) -> None:
    """Switch a plant to per-tick monitor releases (for fast_kernel controllers); the monitor noise and window state
    restart (one sample may be partial; tests drain the first samples)."""
    if not plant.fast_monitor:
        plant.mon = MonitorBank(plant.mon.hw, plant.noise_seeds, pop_period_ticks=1)
        plant.fast_monitor = True


# ---- closed loop ----------------------------------------------------------------------------------------------------

class ClosedLoop:
    """Runs a controller on a LoopPlant through the controller API (07 P0-6 order; fast kernel per sample)."""

    def __init__(self, plant: LoopPlant, spec: ControllerSpec, ctrl, *, check_telemetry: bool = True):
        self.plant, self.spec, self.ctrl = plant, spec, ctrl
        self.validator = ActionValidator(spec)
        ctrl.reset(spec)
        self.fast_fn = controller_fast_kernel(ctrl)
        if self.fast_fn is not None and not plant.fast_monitor:
            raise ValueError("a fast_kernel controller needs LoopPlant(fast_monitor=True)")
        self.latency = int(getattr(ctrl, "latency_ticks", 0))
        self.pipe: deque = deque()
        self.check_telemetry = check_telemetry
        B, N, M = spec.n_batch, spec.n_rings, spec.samples_per_step
        self.blk = dict(samples=np.zeros((B, N, M), np.int64), ticks=np.zeros((B, N, M), np.int64),
                        count=np.zeros((B, N), np.int64), latest=np.zeros((B, N), np.int64),
                        valid=np.ones((B, N), bool), over=np.zeros((B, N), bool))
        if self.fast_fn is not None:
            self.fobs = np.zeros((B, fast_obs_size(N)), np.int64)
            self.fact = np.zeros((B, fast_act_size(N)), np.int64)
        self.n_steps = 0
        self.tel_last = {}
        # drain what the sensor banks queued before the hand-off (placement ran without releases)
        k = plant.k_next
        plant.mon.release(k)
        plant.temps.release(k)
        plant.dp.release(k)

    def _fast_release(self, k: int):
        plant, B, N = self.plant, self.spec.n_batch, self.spec.n_rings
        rel = plant.mon.release(k)
        cnt = rel.count.astype(np.int64)
        if cnt.max() == 0:
            return
        blk, M = self.blk, self.spec.samples_per_step
        for b in range(B):
            for n in range(N):
                for j in range(int(cnt[b, n])):
                    c = int(blk["count"][b, n])
                    if c == M:              # keep the newest M (core_control_api handoff)
                        blk["samples"][b, n, :-1] = blk["samples"][b, n, 1:]
                        blk["ticks"][b, n, :-1] = blk["ticks"][b, n, 1:]
                        c = M - 1
                    blk["samples"][b, n, c] = rel.samples[b, n, j]
                    blk["ticks"][b, n, c] = rel.sample_ticks[b, n, j]
                    blk["count"][b, n] = c + 1
        blk["latest"][:] = rel.latest
        blk["valid"] &= rel.valid
        blk["over"] |= rel.overrange
        p_i, p_f, s_i, s_f = self.ctrl.state
        for b in range(B):
            fo = self.fobs[b]
            fo[0] = k
            fo[1] = N
            fo[2:2 + N] = cnt[b] > 0
            fo[2 + N:2 + 2 * N] = rel.latest[b]
            fo[2 + 2 * N:2 + 3 * N] = np.where(cnt[b] > 0, rel.sample_ticks[b, :, 0], 0)
            self.fast_fn(p_i[b], p_f[b], s_i[b], s_f[b], fo, self.fact[b])
        plant.off_heater[:] = self.fact[:, :N]
        plant.off_bias[:] = self.fact[:, N:]

    def _edge_obs(self, k: int):
        plant = self.plant
        if plant.fast_monitor:
            blk = self.blk
            sensors = {"tx.monitor.samples": blk["samples"].astype(np.int32),
                       "tx.monitor.sample_ticks": blk["ticks"].copy(), "tx.monitor.count": blk["count"].astype(np.int32),
                       "tx.monitor.latest": blk["latest"].astype(np.int32), "monitor_valid": blk["valid"].copy(),
                       "adc_overrange": blk["over"].copy()}
            has = blk["count"] > 0
            new = {"tx.monitor.samples": has, "tx.monitor.sample_ticks": has, "tx.monitor.count": np.ones_like(has),
                   "tx.monitor.latest": has, "monitor_valid": np.ones_like(has), "adc_overrange": np.ones_like(has)}
            s2, n2 = plant.temps.release(k)
            sensors.update(s2)
            new.update(n2)
            last = plant.last
            plant.dp.bitstat_from_monitor(k, plant.mon, last["out_r"][:, op.R_DROP_SUM1],
                                          last["out_r"][:, op.R_DROP_SUM0], np.zeros((plant.B, plant.N), np.int64),
                                          density=last["rho"], rin_per_hz=np.asarray(plant.pp.laser.rin_per_hz).reshape(-1))
            s3, n3 = plant.dp.release(k)
            sensors.update(s3)
            new.update(n3)
            sensors["tx.heater.applied"] = plant.lat_heater.astype(np.uint16)
            sensors["tx.bias.applied"] = plant.lat_bias.astype(np.uint16)
            new["tx.heater.applied"] = np.ones((plant.B, plant.N), bool)
            new["tx.bias.applied"] = np.ones((plant.B, plant.N), bool)
            blk["samples"][:] = 0
            blk["ticks"][:] = 0
            blk["count"][:] = 0
            blk["valid"][:] = True
            blk["over"][:] = False
        else:
            sensors, new = plant.release(k)
        sens = {n: sensors[n] for n in self.spec.sensors}
        nw = {n: new[n] for n in self.spec.sensors}
        return Observation(k, plant.dt, sens, nw)

    def run(self, n_ticks: int, *, record_every: int = 1):
        """Advance n_ticks plant ticks; returns a dict of per-tick Truth records (float32) and telemetry snapshots."""
        plant, spec = self.plant, self.spec
        B, N = spec.n_batch, spec.n_rings
        n_rec = (n_ticks + record_every - 1) // record_every
        rec = {"tick": np.zeros(n_rec, np.int64), "delta_pm": np.zeros((n_rec, B, N), np.float32),
               "oma_w": np.zeros((n_rec, B, N), np.float32), "p_line_w": np.zeros((n_rec, B, N), np.float32),
               "heater": np.zeros((n_rec, B, N), np.int32), "bias": np.zeros((n_rec, B, N), np.int32)}
        tel = []
        j = 0
        k0 = plant.k_next
        for k in range(k0, k0 + n_ticks):
            if plant.fast_monitor and self.fast_fn is not None:
                self._fast_release(k)
            elif plant.fast_monitor:
                self._fast_release_no_kernel(k)
            if is_edge(k, spec.period_ticks, spec.phase_ticks):
                obs = self._edge_obs(k)
                act = self.validator.validate(self.ctrl.step(obs) or {})
                self.pipe.append((k + self.latency, act))
                self.n_steps += 1
                if self.check_telemetry and self.n_steps % 50 == 1:
                    self.tel_last = validate_telemetry(controller_telemetry(self.ctrl), spec)
                    tel.append((k, {kk: vv.copy() for kk, vv in self.tel_last.items()}))
            while self.pipe and self.pipe[0][0] <= k:
                _, act = self.pipe.popleft()
                if "tx.heater" in act:
                    plant.lat_heater[:] = act["tx.heater"].astype(np.int64)
                if "tx.bias" in act:
                    plant.lat_bias[:] = act["tx.bias"].astype(np.int64)
            plant.tick(k)
            if (k - k0) % record_every == 0:
                rec["tick"][j] = k
                rec["delta_pm"][j] = plant.delta_pm
                rec["oma_w"][j] = plant.oma_w
                rec["p_line_w"][j] = plant.p_line_w
                rec["heater"][j] = np.clip(plant.lat_heater + plant.off_heater, plant.h_lo, plant.h_hi)
                rec["bias"][j] = plant.lat_bias + plant.off_bias
                j += 1
        rec["telemetry"] = tel
        rec["saturation"] = self.validator.saturation_counts()
        return rec

    def _fast_release_no_kernel(self, k):
        fn, self.fast_fn = self.fast_fn, None
        try:
            plant, B, N = self.plant, self.spec.n_batch, self.spec.n_rings
            rel = plant.mon.release(k)
            cnt = rel.count.astype(np.int64)
            blk, M = self.blk, self.spec.samples_per_step
            for b in range(B):
                for n in range(N):
                    for jj in range(int(cnt[b, n])):
                        c = int(blk["count"][b, n])
                        if c == M:
                            blk["samples"][b, n, :-1] = blk["samples"][b, n, 1:]
                            blk["ticks"][b, n, :-1] = blk["ticks"][b, n, 1:]
                            c = M - 1
                        blk["samples"][b, n, c] = rel.samples[b, n, jj]
                        blk["ticks"][b, n, c] = rel.sample_ticks[b, n, jj]
                        blk["count"][b, n] = c + 1
            blk["latest"][:] = rel.latest
            blk["valid"] &= rel.valid
            blk["over"] |= rel.overrange
        finally:
            self.fast_fn = fn


# ---- a toy hysteretic ring for fast, plant-free parity and logic tests ----------------------------------------------

class ToyRings:
    """Per (b, n): x = delta/w solves x = (u - u0)/W + g/(1 + x^2) (u = applied heater code, W = half width in codes,
    g = self-heating strength; g > 8/(3 sqrt 3) = 1.54 is bistable), by Newton from the previous root (branch
    continuation: folds and captures happen as in the plant). Monitor y = M/(1 + x^2) + offset + a deterministic
    pseudo-noise of ``noise`` codes, clipped at the 16-bit ADC rail (``adc_overrange`` set); ``m`` samples per step;
    the command acts one step later. Bias codes shift u by ``bias_gain`` codes per bias code (same sign as the heater:
    red shift)."""

    def __init__(self, spec, *, u0, W=450.0, g=4.0, M=50000.0, noise=2.0, m=2, bias_gain=0.0, x0=3.0, seed=1,
                 u_init=None, bias_init=0):
        B, N = spec.n_batch, spec.n_rings
        self.spec = spec
        sh = (B, N)
        self.u0 = np.broadcast_to(np.asarray(u0, float), sh).copy()
        self.W = np.broadcast_to(np.asarray(W, float), sh).copy()
        self.g = np.broadcast_to(np.asarray(g, float), sh).copy()
        self.M = np.broadcast_to(np.asarray(M, float), sh).copy()
        self.noise, self.m, self.bias_gain = float(noise), int(m), float(bias_gain)
        self.x = np.broadcast_to(np.asarray(x0, float), sh).copy()
        self.heater = np.zeros(sh, np.int64)
        self.bias = np.zeros(sh, np.int64)
        self.state = np.uint64(seed * 2654435761 + 12345)
        self.tick = 0
        self.valid = np.ones(sh, bool)
        if u_init is not None:              # start settled on the branch reached from x0 at the initial codes
            self.heater = np.broadcast_to(np.asarray(u_init, np.int64), sh).copy()
            self.bias = np.broadcast_to(np.asarray(bias_init, np.int64), sh).copy()
            self.advance()

    def _rand(self):
        self.state = np.uint64((int(self.state) * 6364136223846793005 + 1442695040888963407) % (1 << 64))
        return ((int(self.state) >> 11) / float(1 << 53)) * 2.0 - 1.0

    def advance(self):
        """x - a = g/(1 + x^2), a = (u - u0)/W, i.e. the cubic x^3 - a x^2 + x - (a + g) = 0: of its real roots take
        the one nearest the previous x (branch continuation; at a fold the old branch vanishes and the ring jumps)."""
        u = self.heater.astype(float) + self.bias_gain * self.bias.astype(float)
        a = (u - self.u0) / self.W
        x = self.x.copy()
        for idx in np.ndindex(x.shape):
            r = np.roots([1.0, -a[idx], 1.0, -(a[idx] + self.g[idx])])
            r = r[np.abs(r.imag) < 1e-9].real
            x[idx] = r[np.argmin(np.abs(r - x[idx]))]
        self.x = x

    def observation(self):
        spec = self.spec
        B, N, M = spec.n_batch, spec.n_rings, spec.samples_per_step
        sensors = {}
        for name, ch in spec.sensors.items():
            sensors[name] = np.zeros(ch.resolve_shape(spec.dims), dtype=ch.np_dtype)
        y = self.M / (1.0 + self.x ** 2)
        smp = np.zeros((B, N, M))
        for j in range(self.m):
            for b in range(B):
                for n in range(N):
                    smp[b, n, j] = y[b, n] + self.noise * self._rand()
        smp = np.clip(np.rint(smp), 0, 65535)
        if "adc_overrange" in sensors:                      # a sample at the ADC rail (07 P1-10)
            sensors["adc_overrange"][...] = (smp[..., :self.m] >= 65535).any(axis=-1)
        sensors["tx.monitor.samples"][..., :self.m] = smp[..., :self.m]
        sensors["tx.monitor.count"][...] = self.m
        sensors["tx.monitor.latest"][...] = smp[..., self.m - 1]
        sensors["tx.monitor.sample_ticks"][..., :self.m] = self.tick
        sensors["monitor_valid"][...] = self.valid
        sensors["tx.heater.applied"][...] = self.heater
        if "tx.bias.applied" in sensors:
            sensors["tx.bias.applied"][...] = self.bias
        new = {name: np.ones((B,) + ch.flag_shape(spec.dims), bool) for name, ch in spec.sensors.items()}
        return Observation(self.tick, spec.tick_s, sensors, new)

    def run(self, ctrl, n_steps: int, *, validator=None, on_step=None):
        """Closed loop for n_steps controller steps; returns the heater (and bias) codes (n, B, N)."""
        v = validator or ActionValidator(self.spec)
        hs, bs = [], []
        for k in range(n_steps):
            if on_step is not None:
                on_step(self, k)
            act = v.validate(ctrl.step(self.observation()) or {})
            if "tx.heater" in act:
                self.heater = act["tx.heater"].astype(np.int64)
            if "tx.bias" in act:
                self.bias = act["tx.bias"].astype(np.int64)
            hs.append(self.heater.copy())
            bs.append(self.bias.copy())
            self.advance()
            self.tick += self.spec.period_ticks
        return np.array(hs), np.array(bs)


def toy_spec(n_batch: int = 2, n_rings: int = 3, *, bias: bool = False, **kw) -> ControllerSpec:
    """A small spec for the toy loops (core defaults: 100 kHz controller, 250 kHz ADC)."""
    bind = {"tx.bias": "controller"} if bias else None
    return ControllerSpec.build(n_batch=n_batch, n_rings=n_rings, n_lines=n_rings, tick_s=1e-6, update_rate_hz=1e5,
                                bind=bind, **kw)


# ---- Truth metrics (tests only; 07 P0-8 definitions, simplified) -----------------------------------------------------

@dataclass
class LockMetrics:
    """Per-ring Truth metrics of a window, in pm and dB (B, N arrays)."""
    delta_ref_pm: np.ndarray
    rms_err_pm: np.ndarray
    max_err_pm: np.ndarray
    excursions: np.ndarray
    time_out_us: np.ndarray
    oma_penalty_db_mean: np.ndarray
    oma_penalty_db_max: np.ndarray
    reloma_db: np.ndarray

    def summary(self) -> dict:
        return {k: (float(np.nanmax(v)) if k not in ("delta_ref_pm", "reloma_db") else float(np.nanmean(v)))
                for k, v in self.__dict__.items()}


def lock_metrics(rec, ref: slice, win: slice, *, threshold_pm: float = LOCK_THRESHOLD_PM,
                 hyst_pm: float = HYSTERESIS_PM, dt_us: float = 1.0, delta_ref=None, oma_ref=None) -> LockMetrics:
    """Lock error e = delta - delta_ref (delta_ref = median over ``ref`` unless given), its rms and max over ``win``;
    excursions = entries into |e| > threshold (exit below threshold - hysteresis); time outside the band; OMA
    penalty 10 log10(OMA_ref / OMA) against the median OMA over ``ref``; relOMA = OMA / P_line (dB) over ``win``."""
    d = rec["delta_pm"].astype(float)
    o = rec["oma_w"].astype(float)
    p_in = rec["p_line_w"].astype(float)
    dref = np.median(d[ref], axis=0) if delta_ref is None else np.asarray(delta_ref, float)
    oref = np.median(o[ref], axis=0) if oma_ref is None else np.asarray(oma_ref, float)
    e = d[win] - dref[None]
    ae = np.abs(e)
    inside = np.ones(dref.shape, bool)
    exc = np.zeros(dref.shape, np.int64)
    for t in range(ae.shape[0]):
        out = ae[t] > threshold_pm
        back = ae[t] < threshold_pm - hyst_pm
        exc += inside & out
        inside = np.where(inside, ~out, back)
    pen = 10.0 * np.log10(np.maximum(oref[None], 1e-30) / np.maximum(o[win], 1e-30))
    rel = 10.0 * np.log10(np.maximum(o[win], 1e-30) / np.maximum(p_in[win], 1e-30))
    return LockMetrics(delta_ref_pm=dref, rms_err_pm=np.sqrt(np.mean(e ** 2, axis=0)), max_err_pm=ae.max(axis=0),
                       excursions=exc, time_out_us=(ae > threshold_pm).sum(axis=0) * dt_us,
                       oma_penalty_db_mean=pen.mean(axis=0), oma_penalty_db_max=pen.max(axis=0),
                       reloma_db=np.median(rel, axis=0))


def time_to_lock_us(rec, start_index: int = 0, *, band_pm: float = LOCK_THRESHOLD_PM, tail_frac: float = 0.2,
                    dt_us: float = 1.0) -> np.ndarray:
    """First record index (from start_index) after which |delta - delta_final| stays below band_pm for the rest of
    the run; delta_final = median of the last tail_frac of the run. NaN if never. In us (records every dt_us)."""
    d = rec["delta_pm"].astype(float)[start_index:]
    n = d.shape[0]
    final = np.median(d[int(n * (1 - tail_frac)):], axis=0)
    bad = np.abs(d - final[None]) >= band_pm
    out = np.full(final.shape, np.nan)
    for idx in np.ndindex(final.shape):
        col = bad[(slice(None),) + idx]
        if not col[-1]:
            last_bad = np.flatnonzero(col)
            out[idx] = (last_bad[-1] + 1 if last_bad.size else 0) * dt_us
    return out


def snapshot(plant: LoopPlant) -> LoopPlant:
    """Deep copy of a plant (state included), to start several tests from one locked state. The optics views alias
    the packed self-heating rows, so they are rebuilt on the copy (deepcopy would turn them into separate arrays)."""
    cp = copy.deepcopy(plant)
    cp.views = [op.selfheat_views(cp.args_f[b], cp.args_i[b]) for b in range(cp.B)]
    return cp


# ---- checks of the harness itself -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nominal_plant():
    cfg = load_cfg()
    pp = realize(cfg, 1)
    spec = make_spec(cfg, 1)
    return cfg, pp, spec


def test_serial_noise_matches_library(nominal_plant):
    seeds = rng.batch_seeds(11, 2)
    a, b = np.zeros((2, 8)), np.zeros((2, 8))
    sid = np.int64(rng.stream_id("dac_bias"))
    rng.fill_normal(a, seeds, sid, np.int64(123), np.int64(0))
    _fill_normal_serial(b, seeds, sid, np.int64(123), np.int64(0))
    assert np.array_equal(a, b)


def test_loop_plant_matches_reference_plant_open_loop(nominal_plant):
    """The harness tick equals the integrator's ReferencePlant tick (same scenario, same commands)."""
    cfg, pp, spec = nominal_plant
    ref = ReferencePlant(pp, cfg, pop_period=spec.period_ticks)
    lp = LoopPlant(pp, cfg, period_ticks=spec.period_ticks)
    code = np.full((1, pp.n_rings), 20000, np.int64)
    lp.lat_heater[:] = code
    for k in range(60):
        o_ref = ref.tick(k, heater_cmd=code if k == 0 else None)
        lp.tick(k)
    np.testing.assert_allclose(lp.t_ring, ref.t_ring, rtol=0, atol=1e-9)
    np.testing.assert_allclose(lp.last["out_r"], o_ref["out_r"], rtol=1e-12, atol=1e-18)
    s1, _ = ref.release(60)
    s2, _ = lp.release(60)
    assert np.array_equal(s1["tx.monitor.samples"], s2["tx.monitor.samples"])


def test_truth_detuning_tracks_heater(nominal_plant):
    """Truth helper: placing on the hot branch lands at the requested detuning, on the red (stable) side."""
    cfg, pp, spec = nominal_plant
    lp = LoopPlant(pp, cfg, period_ticks=spec.period_ticks)
    d = place_on_hot_branch(lp, delta_pm=170.0, n_ticks=2500)
    assert np.all(np.abs(d - 170.0) < 3.0), d
    # a heater step up moves every ring further red (stable branch: d delta / d code > 0)
    lp.lat_heater += 200
    for k in range(lp.k_next, lp.k_next + 200):
        lp.tick(k)
    assert np.all(lp.delta_pm > d + 1.0)


def test_loop_delay_through_the_harness(nominal_plant):
    """07 P0-6 test_loop_delay_impulse through the harness: a one-step heater impulse first shows in the monitor
    samples released at timing.first_response_tick (heater DAC latency + thermal transport ticks, ADC phase,
    window and latency)."""
    from mrm.core.timing import first_response_tick
    cfg, pp, spec = nominal_plant
    lp = LoopPlant(pp, cfg, period_ticks=spec.period_ticks)
    place_on_hot_branch(lp, delta_pm=170.0, n_ticks=1500)

    class Probe:
        name, latency_ticks = "probe", 0

        def __init__(self):
            self.k0, self.base, self.seen = None, None, []

        def reset(self, s):
            self.spec = s

        def step(self, obs):
            c = np.asarray(obs["tx.heater.applied"]).astype(np.int64)
            if self.base is None:
                self.base = c.copy()
                self.k0 = obs.tick + 5 * spec.period_ticks
            cnt = int(obs["tx.monitor.count"][0, 0])
            self.seen.append((obs.tick, obs["tx.monitor.samples"][0, 0, :cnt].copy(),
                              obs["tx.monitor.sample_ticks"][0, 0, :cnt].copy()))
            if obs.tick == self.k0:
                return {"tx.heater": self.base + 3000}
            return {"tx.heater": self.base}

    pr = Probe()
    loop = ClosedLoop(lp, spec, pr, check_telemetry=False)
    loop.run(12 * spec.period_ticks)
    dac_lat = int(np.max(lp.heater_p.latency_ticks))
    want = first_response_tick(pr.k0, 0, dac_lat, spec.adc.period_ticks, int(lp.mon.hw.phase.reshape(-1)[0]),
                               spec.adc.latency_ticks, spec.period_ticks, spec.phase_ticks)
    # baseline: the samples before the impulse are steady within noise; the first moved sample arrives at `want`
    base = np.concatenate([s for t, s, _ in pr.seen if t <= pr.k0]).astype(float)
    level, noise = np.median(base), np.std(base) + 2.0
    first = None
    for t, s, st in pr.seen:
        moved = np.abs(s.astype(float) - level) > 8 * noise
        if t > pr.k0 and moved.any():
            first = t
            break
    assert first == want, (first, want)
