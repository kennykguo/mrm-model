"""Integration (foundation stage): realize(g26) feeds every leaf module of the plant, in the event order of
mrm/core/timing.py, and the sensor dictionaries the plant releases are exactly the channels the controller API
declares (07 P0-4, P0-6, P0-7). This file is also the worked example of the wiring that docs/handoff/ENGINE_BRIEF.md
describes: :class:`ReferencePlant` is a Python slow-path plant (B worlds, per-tick Python loop) built only from
PlantParams and the leaf modules' public adapters.

Checks: every builder accepts PlantParams; the released sensors match ControllerSpec.from_config (names, dtypes,
shapes) and pack into the flat ABI; heater power moves the ring by the R2 thermal resistance through the thermal
system; light is conserved along the bus; the monitor codes follow the drop power; the PD dark current follows the
ROM's monitor_pd observer (07 P1-9 cross-module test); and the scenario's bias and laser signals reach the plant.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from mrm import params
from mrm.control.api import ControllerSpec
from mrm.core import rng
from mrm.plant import electrical as el
from mrm.plant import optics as op
from mrm.plant import phase as ph
from mrm.plant import selfheat as shm
from mrm.plant import thermal as th
from mrm.plant.datapath import DatapathBank, DatapathHw
from mrm.plant.laser import LaserBank, LaserHw
from mrm.plant.sensing import MonitorBank, MonitorHw, TempSensorBank, TempSensorsHw, fault_mask
from mrm.scenarios import registry as scen_registry
from mrm.scenarios.generators import Scenario, const
from mrm.units import C0


class ReferencePlant:
    """Python reference of one plant tick for B worlds (self-heating on, exact optics, interim or ROM thermal).

    Event order per plant interval [k, k+1) (docs/handoff/ENGINE_BRIEF.md section 3): scenario signals at k and
    k+1 -> laser lines -> ones density -> bias DAC -> NRZ levels -> optics inputs at the previous ring temperature ->
    self-heating bounds -> heater TCR split + heater DAC -> data-activity heat -> implicit thermal + self-heating step
    -> monitor / data path / temperature sensors. At a controller edge: release every sensor bank.
    """

    def __init__(self, pp, cfg, *, scenario=None, noise_seed: int = 11, scenario_seed: int = 5, pop_period: int = 1):
        self.pp = pp
        B, N, L = pp.batch, pp.n_rings, pp.n_lines
        self.B, self.N, self.L = B, N, L
        self.dt = float(pp.timing.dt_s)
        self.sys = sysm = pp.thermal.system
        assert sysm is not None, "realize() must assemble the thermal system (thermal_hook='auto')"
        # this reference implements the self-heating path (07 P0-1); with fidelity.self_heating off the plant runs
        # bus_propagate_nb at the previous ring temperatures and thermal_advance_b without absorbed power
        assert pp.flags.self_heating == 1, "ReferencePlant implements fidelity.self_heating: on only"
        self.phase = ph.PhaseParams.from_plant_params(pp)
        self.heater_p = el.HeaterDriverParams.from_plant_params(pp)
        self.heater = el.HeaterBank(self.heater_p, initial_code=0)
        self.bias_p = el.BiasDacParams.from_plant_params(pp)
        self.bias = el.BiasBank(self.bias_p, initial_code=0)
        self.sc = scen_registry.from_config(cfg) if scenario is None else scenario
        agg_names = tuple(sysm.src_names[i] for i in sysm.aggressor_src)
        self.arr = self.sc.compile(dt_s=self.dt, n_rings=N, n_lines=L, batch=B, seed=scenario_seed,
                                   aggressor_sites=agg_names)
        self.cache = self.arr.new_cache()
        self.laser = LaserBank(LaserHw.from_plant_params(pp), scenario_seeds=rng.batch_seeds(scenario_seed, B),
                               time_compression=self.sc.time_compression)
        nseeds = rng.batch_seeds(noise_seed, B)
        self.mon = MonitorBank(MonitorHw.from_plant_params(pp), nseeds, pop_period_ticks=pop_period)
        self.temps = TempSensorBank(TempSensorsHw.from_plant_params(pp), nseeds, pop_period_ticks=pop_period)
        self.dp = DatapathBank(DatapathHw.from_plant_params(pp), nseeds, pop_period_ticks=pop_period)
        self.noise_seeds = nseeds
        # thermal + self-heating state
        self.coef = sysm.coefficients(self.dt)
        self.x = sysm.zero_state(B)
        self.src_scale = np.ones((1, sysm.n_src))
        self.p_prev = np.zeros((B, N))
        self.t_branch = np.full((B, N), -np.inf)          # 07 P0-1 item 6: start on the cold branch
        t_amb = self._sig(0, "thermal.t_base_k")[:, 0] + self._sig(0, "thermal.xpu_rise_k")[:, 0]
        self.t_ring = np.repeat(t_amb[:, None], N, axis=1)
        self.t_heater = self.t_ring.copy()
        self.t_obs = np.repeat(t_amb[:, None], sysm.n_obs, axis=1)
        self.g_work = np.zeros((B, sysm.n_obs))
        self.p_new = np.zeros((B, N))
        self.t_new = np.zeros((B, N))
        self.n_iter = np.zeros((B, N), np.int64)
        self.u_heater = np.zeros((B, N))
        # optics: one packed self-heating row per world (mrm.plant.optics.pack_selfheat_args)
        self.method = "tcmt" if pp.flags.optics_model else "exact"
        lam0 = C0 / np.asarray(pp.laser.line_freq_hz, float)
        rows = []
        for b in range(B):
            o = self._optics_inputs(b, lam0[b], np.zeros(N), np.zeros(N))
            rows.append(op.pack_selfheat_args(
                np.full(L, 1e-3), o["phi0"], o["k2t"], o["k2d"], o["a0"], o["trt"], o["e0"], lam0[b],
                self.t_ring[b], np.ones(N) * 7e-11, 0.0, bus_order=pp.layout.bus_order,
                rb=np.asarray(pp.optics.backscatter_rb[b]), nrz={"p1": np.full(N, 0.5), "phi1": o["phi1"],
                                                                 "a1": o["a1"], "eta1": o["e1"]},
                method=self.method, m_orders=int(pp.optics.tcmt_orders)))
        self.args_f = np.ascontiguousarray(np.stack([r[0] for r in rows]))
        self.args_i = np.ascontiguousarray(np.stack([r[1] for r in rows]))
        self.p_max = np.zeros((B, N))
        self.d2p_max = np.zeros((B, N))
        self.dpdt_max = np.zeros((B, N))
        self.last = {}

    # ---- helpers
    def _sig(self, k, name):
        return self.arr.evaluate(int(k), self.cache)[name]

    def _optics_inputs(self, b, lam, v0, v1):
        P = self.phase
        N, L = self.N, lam.shape[0]
        o = {n: np.zeros((N, L)) for n in ("phi0", "phi1", "a0", "a1", "e0", "e1", "k2t", "k2d", "trt")}
        o.update(shift=np.zeros((N, 2)), cj=np.zeros((N, 2)), rs=np.zeros((N, 2)), flags=np.zeros(N, np.int8))
        r = P.table_row(b)
        ph.ring_optics_row(P.ring[b], np.ascontiguousarray(lam), self.t_ring[b], self.t_heater[b],
                           np.ascontiguousarray(v0, dtype=float), np.ascontiguousarray(v1, dtype=float), P.grid,
                           P.dneff[r], P.alpha_fca[r], P.capacitance[r], P.r_series[r], o["phi0"], o["phi1"], o["a0"],
                           o["a1"], o["e0"], o["e1"], o["k2t"], o["k2d"], o["trt"], o["shift"], o["cj"], o["rs"],
                           o["flags"])
        return o

    # ---- one plant interval [k, k+1)
    def tick(self, k: int, heater_cmd=None):
        pp, sysm, B, N = self.pp, self.sys, self.B, self.N
        ev0 = self.arr.evaluate(int(k), self.cache)
        ev1 = self.arr.evaluate(int(k) + 1, self.cache)
        # 1. laser lines at the module output, then the chip input (scenario: line enable, dfreq, power, input loss)
        f_hz, p_w = self.laser.tick(k, enable=ev0["laser.line_enable"], f_offset_hz=ev0["laser.dfreq_hz"],
                                    p_scale=10.0 ** (ev0["laser.power_delta_db"] / 10.0))
        p_chip = p_w * np.asarray(pp.laser.input_transmission) * 10.0 ** (-ev0["optical.input_loss_delta_db"] / 10.0)
        lam = C0 / f_hz
        # 2. ones density of the traffic (07 P0-11)
        rho = self.dp.pattern(k, density=ev0["data.ones_density"], traffic_on=ev0["data.driver_enable"])
        # 3. bias DAC (the scenario owns tx.bias in g26; z from the counter RNG, stream dac_bias)
        z = np.zeros((B, N))
        rng.fill_normal(z, self.noise_seeds, np.int64(rng.stream_id("dac_bias")), np.int64(k), np.int64(0))
        v_set = ev0["tx.bias_v"]
        cmd_b = self.bias_p.code_for_voltage(np.where(np.isfinite(v_set), v_set, self.bias_p.v_min_v))
        bout = self.bias.step(k, cmd=cmd_b, z=z, v_offset=ev0["tx.bias_offset_v"])
        # 4. junction levels (07 P1-7): level 1 = the higher reverse voltage (assumed bit mapping)
        swing = np.asarray(pp.datapath.swing_pp_v) * ev0["data.swing_scale"]
        v_lo, v_hi, fl = np.zeros((B, N)), np.zeros((B, N)), np.zeros((B, N), np.int8)
        for b in range(B):
            el.nrz_levels_row(bout.v_avg[b], swing[b], int(pp.datapath.level_convention[b, 0]),
                              el.JUNCTION_GRID_V_MIN, el.JUNCTION_GRID_V_MAX, v_lo[b], v_hi[b], fl[b])
        # 5. optics inputs at the previous ring temperature, written into the self-heating rows (local R1 law)
        cj_eff = np.zeros((B, N))
        r_s = np.zeros((B, N))
        for b in range(B):
            o = self._optics_inputs(b, lam[b], v_lo[b], v_hi[b])
            r_s[b] = 0.5 * (o["rs"][:, 0] + o["rs"][:, 1])
            v = op.selfheat_views(self.args_f[b], self.args_i[b])
            (p_in, t2, t2o, p1, phi0r, phi1r, _, _, a0, a1, e0, e1, psi, k2t, k2d, trt, dpl, tref, s0, beta,
             *_rest) = v
            p_in[:] = p_chip[b]
            p1[:] = rho[b]
            phi0r[:], phi1r[:] = o["phi0"], o["phi1"]
            a0[:], a1[:], e0[:], e1[:] = o["a0"], o["a1"], o["e0"], o["e1"]
            k2t[:], k2d[:], trt[:] = o["k2t"], o["k2d"], o["trt"]
            dpl[:] = 2.0 * math.pi * C0 * o["trt"] / (lam[b][None, :] ** 2)      # optics.dphi_dlambda_res
            rp = self.phase.ring[b]
            for n in range(N):
                tref[n] = self.t_ring[b, n]
                s0[n], beta[n] = ph.local_thermo_optic(self.t_ring[b, n], rp[n, ph.RP_S0], rp[n, ph.RP_BETA],
                                                       rp[n, ph.RP_T0])
            r = self.phase.table_row(b)
            el.junction_swing_capacitance_row(v_lo[b], v_hi[b], self.phase.grid, self.phase.capacitance[r],
                                              rp[:, ph.RP_FILL], rp[:, ph.RP_LEN], cj_eff[b])
            op.selfheat_bounds(self.args_f[b], self.args_i[b], 50.0, self.p_max[b], self.d2p_max[b],
                               self.dpdt_max[b])
        # 6. thermal inputs: heater (held), junction and driver heat (held), aggressor (FOH), absorbed (implicit)
        u0 = np.zeros((B, sysm.n_src))
        u1 = np.zeros((B, sysm.n_src))
        pj, pd = np.zeros((B, N)), np.zeros((B, N))
        d = pp.datapath
        for b in range(B):
            en = (ev0["data.driver_enable"][b] >= 0.5).astype(float)
            el.data_activity_heat_row(cj_eff[b], v_hi[b] - v_lo[b], float(d.bitrate_hz[b, 0]),
                                      ev0["data.transition_density"][b], en, r_s[b],
                                      np.asarray(d.driver_output_resistance_ohm[b]),
                                      np.asarray(d.driver_static_power_w[b]),
                                      np.asarray(d.driver_energy_per_transition_j[b]), 0, pj[b], pd[b])
        for u, ev in ((u0, ev0), (u1, ev1)):
            u[:, sysm.junction_src] = pj
            if sysm.driver_src.size:
                u[:, sysm.driver_src] = pd[:, :sysm.driver_src.size]
            if sysm.aggressor_src.size:
                u[:, sysm.aggressor_src] = ev["thermal.aggressor_w"]
            u[:, sysm.heater_src] = self.u_heater                   # previous heater powers (lag one step, others)
            u[:, sysm.wg_src] = self.p_prev                         # previous absorbed powers (ignored by the step)
        t_base_end = ev1["thermal.t_base_k"][:, 0] + ev1["thermal.xpu_rise_k"][:, 0]
        # 7. heater: implicit TCR split on the heater-resistor observer, then the DAC path (07 P1-2, P1-11)
        a, c0, c1 = self.coef
        pre = np.zeros((B, N))
        g = np.zeros((B, N))
        for b in range(B):
            for n in range(N):
                pre[b, n], g[b, n] = th.observer_split_b(self.x[b], a[b], c0[b], c1[b], sysm.lag_src, sysm.obs_ptr,
                                                         int(sysm.heater_obs[n]), int(sysm.heater_src[n]), u0[b],
                                                         u1[b], self.src_scale[0])
        hout = self.heater.step(k, heater_cmd, None, t_base_end[:, None] + pre, g)
        self.u_heater = hout.p_w.copy()
        u0[:, sysm.heater_src] = self.u_heater
        u1[:, sysm.heater_src] = self.u_heater
        # 8. implicit thermal + self-heating step (07 P0-1); the optics walk runs inside (commit_fn)
        n_fail = shm.implicit_step(self.x, a, c0, c1, sysm.lag_src, sysm.lag_obs, sysm.lag_kind, sysm.opt_lags,
                                   sysm.obs_ptr, sysm.src_ring, sysm.ring_obs, np.asarray(pp.layout.bus_order), u0,
                                   u1, self.src_scale, t_base_end, self.p_prev, self.t_branch, op.selfheat_pabs,
                                   op.selfheat_commit, self.args_f, self.args_i, self.p_max, self.d2p_max, 10, 1e-9,
                                   1, self.g_work, self.t_obs, self.p_new, self.t_new, self.n_iter)
        self.p_prev[:] = self.p_new
        self.t_branch[:] = self.t_new
        self.t_ring[:] = self.t_new
        self.t_heater[:] = self.t_obs[:, sysm.heater_obs]
        # 9. sensors of interval [k, k+1)
        out_r = np.stack([op.selfheat_views(self.args_f[b], self.args_i[b])[23] for b in range(B)])   # (B, N_R, N)
        out_rl = np.stack([op.selfheat_views(self.args_f[b], self.args_i[b])[22] for b in range(B)])  # (B,N_RL,N,L)
        out_l = np.stack([op.selfheat_views(self.args_f[b], self.args_i[b])[24] for b in range(B)])   # (B, N_L, L)
        fault = fault_mask(B, N, adc_drop=ev0["fault.adc_drop"], adc_stuck=ev0["fault.adc_stuck"],
                           adc_saturate=ev0["fault.adc_saturate"], dac_glitch_codes=ev0["fault.dac_glitch_codes"])
        p_bus = out_rl[:, op.RL_PIN].sum(axis=2)                                              # (B, N) bus power
        self.mon.tick(k, p_pd_w=out_r[:, op.R_DROP_SUM], p_bg_w=p_bus, t_pd_k=self.t_obs[:, sysm.pd_obs],
                      fault=fault)
        line = np.asarray(pp.layout.ring_to_line)
        oma = out_rl[:, op.RL_POUT1, np.arange(N), line] - out_rl[:, op.RL_POUT0, np.arange(N), line]
        self.dp.errors(k, np.abs(oma))
        self.temps.tick(k, self.t_obs[:, sysm.temp_obs])
        self.last = dict(p_chip=p_chip, lam=lam, rho=rho, bias=bout, v_lo=v_lo, v_hi=v_hi, heater=hout, out_r=out_r,
                         out_rl=out_rl, out_l=out_l, n_fail=n_fail, pj=pj, pd=pd, oma=oma)
        return self.last

    # ---- controller edge at tick k
    def release(self, k: int):
        """Release every sensor bank at tick k (a controller edge). Kept as is on purpose (controllers_b handoff 2):
        the bit-statistics sampler gets threshold 0, not the ``datapath.bitstat_threshold`` actuator (benches that
        need the actuator, e.g. tests/verification/test_ctrl_b_plant.py, override this), and the caller must call
        it once per controller period (``pop_period``), or the monitor FIFO overflows and ``latest`` goes stale.
        The engine (mrm.sim.kernel.release_row) uses the actuator threshold and releases at every edge."""
        rel = self.mon.release(k)
        sensors, new = self.mon.channels(rel)
        s2, n2 = self.temps.release(k)
        sensors.update(s2)
        new.update(n2)
        last = self.last
        self.dp.bitstat_from_monitor(k, self.mon, last["out_r"][:, op.R_DROP_SUM1], last["out_r"][:, op.R_DROP_SUM0],
                                     np.zeros((self.B, self.N), np.int64), density=last["rho"],
                                     rin_per_hz=np.asarray(self.pp.laser.rin_per_hz).reshape(-1))
        s3, n3 = self.dp.release(k)
        sensors.update(s3)
        new.update(n3)
        sensors["tx.heater.applied"] = self.heater.applied_codes.astype(np.uint16)
        sensors["tx.bias.applied"] = self.bias.applied_codes.astype(np.uint16)
        new["tx.heater.applied"] = np.ones((self.B, self.N), bool)
        new["tx.bias.applied"] = np.ones((self.B, self.N), bool)
        return sensors, new


@pytest.fixture(scope="module")
def setup():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")       # missing-artefact fallbacks warn until the extraction lands
        cfg = params.load("g26")
        pp = params.realize(cfg, 2, seeds=7)
    spec = ControllerSpec.from_config(cfg, n_batch=2, nominal=cfg)
    plant = ReferencePlant(pp, cfg, pop_period=spec.period_ticks)
    return cfg, pp, spec, plant


def test_every_leaf_builds_from_plant_params(setup):
    _, pp, _, plant = setup
    assert plant.heater_p.law.shape == (pp.batch,)
    np.testing.assert_array_equal(plant.heater_p.r_cold_ohm, pp.heater.r_cold_ohm)
    # the transport delay of the thermal system rides on the heater FIFO (07 P1-2, one owner per delay)
    lat = np.asarray(pp.timing.heater_latency_ticks).reshape(-1)
    td = np.rint(np.asarray(pp.thermal.system.transport_delay_s) / pp.timing.dt_s).astype(int)
    np.testing.assert_array_equal(plant.heater_p.latency_ticks, lat + td)
    assert plant.phase.ring.shape == (pp.batch, pp.n_rings, ph.N_RP)
    assert plant.mon.hw.device_seeds is not None


def test_released_sensors_match_the_controller_channels(setup):
    _, pp, spec, plant = setup
    for k in range(3 * spec.period_ticks):
        plant.tick(k)
    sensors, new = plant.release(3 * spec.period_ticks)
    dims = dict(spec.layout.dims)
    for name, ch in spec.channels.items():
        if ch.kind != "sensor":
            continue
        assert name in sensors, f"plant does not release {name}"
        shape = tuple(dims[s] if isinstance(s, str) else s for s in ch.shape)
        assert sensors[name].shape == shape, (name, sensors[name].shape, shape)
        assert np.dtype(sensors[name].dtype) == np.dtype(ch.dtype), (name, sensors[name].dtype, ch.dtype)
        assert name in new
    obs = spec.layout.pack_obs(3 * spec.period_ticks, sensors, new)
    assert obs.shape == (pp.batch, spec.layout.obs_size)
    # the monitor delivered every ADC sample of the period (M = ceil(period / ADC period))
    assert np.all(sensors["tx.monitor.count"] >= spec.period_ticks // pp.monitor.adc_period_ticks)


def test_heater_moves_the_ring_by_the_r2_resistance(setup):
    cfg, pp0, spec, _ = setup
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp = params.realize(cfg, 1, uncertainty="nominal", variation="off")
    # lasers and drivers off: with +4 dBm per line a ring heated by ~10 K reaches the line three channels blue of its
    # own (7.6 nm offset, 2.29 nm grid) and self-heats on it, and the data-activity heat (07 P1-8: CV^2 f in the
    # junction, ~0.14 mW at 9.9 K/mW, plus ~4.5 mW of driver heat per ring) adds ~2 K; both are real but not this check
    dark = Scenario(name="dark", components=(const("laser.line_enable", 0.0, absolute=True),
                                             const("data.driver_enable", 0.0, absolute=True)))
    plant = ReferencePlant(pp, cfg, scenario=dark)
    code = np.full((1, pp.n_rings), 12000, np.int64)
    for k in range(400):
        out = plant.tick(k, heater_cmd=code if k == 0 else None)
    assert np.all(out["p_chip"] == 0.0)
    p_h = out["heater"].p_w[0]
    assert np.all(p_h > 1e-3) and np.all(p_h < 0.05)
    t_amb = 298.15
    rise = plant.t_ring[0] - t_amb
    z = pp.thermal.z_heater_ring_k_per_w[0]
    # interim model: after 400 us the fast and slow poles have settled; the 1-10 ms tail and the ms crosstalk have
    # not, so the rise is between (1 - tail share) Z P and Z P plus the neighbours' crosstalk (a few %)
    tail = float(np.asarray(pp.values["ring.thermal.tail_share_frac"]).ravel()[0])
    assert np.all(rise > 0.97 * (1.0 - tail) * z * p_h)
    assert np.all(rise < 1.05 * z * p_h)
    # the heater-resistor observer sits hotter than the ring by the heater_self_ratio construction (07 P1-2)
    assert np.all(plant.t_heater[0] > plant.t_ring[0])


def test_light_is_conserved_along_the_bus(setup):
    _, pp, _, plant = setup
    out = plant.tick(1000)
    p_in = out["p_chip"]                                        # (B, L)
    p_out = out["out_l"][:, op.L_POUT]
    drops = out["out_rl"][:, op.RL_DROP].sum(axis=1) + out["out_rl"][:, op.RL_ADD].sum(axis=1)
    loss = out["out_rl"][:, op.RL_LOSS].sum(axis=1) + out["out_rl"][:, op.RL_REFL].sum(axis=1)
    np.testing.assert_allclose(p_out + drops + loss, p_in, rtol=1e-9)
    absorbed = out["out_r"][:, op.R_ABS_SUM]
    assert np.all(absorbed <= out["out_r"][:, op.R_LOSS_SUM] + 1e-18)


def test_monitor_codes_follow_the_drop_power(setup):
    _, pp, spec, plant = setup
    k0 = 2000
    for k in range(k0, k0 + spec.period_ticks):
        out = plant.tick(k)
    sensors, _ = plant.release(k0 + spec.period_ticks)
    cpw = plant.mon.hw.codes_per_w()
    want = cpw * out["out_r"][:, op.R_DROP_SUM]
    got = sensors["tx.monitor.latest"].astype(float)
    code_max = (1 << int(pp.monitor.adc_bits)) - 1
    clipped = want >= code_max
    # a ring that self-heated onto a line drives its drop current past the 40 uA full scale: the ADC clips and says so
    assert np.all(got[clipped] == code_max) and np.all(sensors["adc_overrange"][clipped])
    # otherwise the ideal code plus dark current, offsets, noise and latency: within 5 % + 30 codes
    ok = ~clipped
    assert np.all(np.abs(got[ok] - want[ok]) < 0.05 * want[ok] + 30.0)


def test_dark_current_follows_the_rom_pd_observer(setup):
    """07 P1-9: heater power warms the monitor PD through the thermal system's monitor_pd observer, and the dark
    current follows the Arrhenius law at that temperature (mrm.plant.sensing)."""
    _, pp, _, plant = setup
    sysm = pp.thermal.system
    u = np.zeros((pp.batch, sysm.n_src))
    u[:, sysm.heater_src] = 0.02
    x = sysm.steady_state(u)
    t_obs = sysm.outputs(x, 298.15)
    t_pd = t_obs[:, sysm.pd_obs]
    assert np.all(t_pd > 298.15 + 1.0)
    hw = plant.mon.hw
    _, i_dark_hot = hw.photocurrent_a(np.zeros((pp.batch, pp.n_rings)), 0.0, t_pd)
    _, i_dark_cold = hw.photocurrent_a(np.zeros((pp.batch, pp.n_rings)), 0.0, np.full_like(t_pd, 298.15))
    ea = np.asarray(pp.monitor.dark_activation_j).reshape(-1, 1)
    kb = 1.380649e-23
    want = np.exp(-ea / kb * (1.0 / t_pd - 1.0 / 298.15))
    np.testing.assert_allclose(i_dark_hot / i_dark_cold, want, rtol=1e-9)


def test_scenario_bias_reaches_the_junction(setup):
    _, pp, _, plant = setup
    out = plant.tick(3000)
    v_avg = out["bias"].v_avg
    lsb = float(np.max(plant.bias_p.lsb_v))
    want = 1.05                                   # g26 scenario.bias_v (printed relOMA optimum, 07 D9)
    noise = float(np.max(np.asarray(pp.bias_dac.noise_rms_codes))) * lsb
    assert np.all(np.abs(v_avg - want) < lsb + 6 * noise + 1e-12), v_avg
    swing = np.asarray(pp.datapath.swing_pp_v)
    np.testing.assert_allclose(out["v_hi"] - out["v_lo"], swing, rtol=1e-12)


def test_dac_glitch_offsets_the_dac_output_not_the_register():
    """07 P1-10 dac_glitch: the scenario's fault.dac_glitch_codes (mask from mrm.plant.sensing.fault_mask) adds codes
    to the code leaving the heater DAC FIFO for that tick; the latched register (applied codes) does not show it."""
    from mrm.plant.sensing import FAULT_DAC_GLITCH
    prm = el.HeaterDriverParams.build(1, 2, 1e-6, law="voltage", v_full_scale_v=1.6, r_cold_ohm=45.4, tcr_per_k=1e-3,
                                      latency_s=0.0)
    bank = el.HeaterBank(prm, initial_code=1000)
    glitch = np.array([[500.0, 0.0]])
    fault = fault_mask(1, 2, dac_glitch_codes=glitch)
    add = np.where(fault & FAULT_DAC_GLITCH, np.rint(glitch), 0).astype(np.int64)
    out = bank.step(0, code_add=add)
    assert out.code_dac[0, 0] == 1500 and out.code_dac[0, 1] == 1000
    assert np.all(out.applied_codes == 1000)
    u_glitch = out.u[0, 0]
    out = bank.step(1)
    assert out.code_dac[0, 0] == 1000 and out.u[0, 0] < u_glitch
    # clipped to the DAC range
    out = bank.step(2, code_add=np.array([[-5000, 10 ** 6]], np.int64))
    assert out.code_dac[0, 0] == 0 and out.code_dac[0, 1] == (1 << prm.bits) - 1


def test_integrator_keys_reach_the_leaf_adapters(tmp_path):
    """Keys added at integration (plant_sensing requests) travel config -> PlantParams -> the from_plant_params
    adapters, and their absence keeps the module defaults."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load("g26", overrides=["datapath.link_loss_db=5.0", "datapath.rx_sensitivity_oma_dbm=-11.0",
                                            "datapath.bitstat_bw_ghz=20", "datapath.ber_ref=1e-12",
                                            "ring.monitor.baseline_ua=12.0", "ring.monitor.stuck_flagged=false",
                                            "ring.monitor.adc_mux.order=bank", "ring.heater.readback.present=true",
                                            "ring.heater.readback.noise_rms_codes=0.5",
                                            "ring.heater.readback.average=true"])
        pp = params.realize(cfg, 2, seeds=1)
        base = params.realize(params.load("g26"), 2, seeds=1)
    dp = DatapathHw.from_plant_params(pp)
    np.testing.assert_allclose(dp.link_t, 10 ** (-0.5), rtol=1e-12)
    np.testing.assert_allclose(dp.oma_sens_w, 1e-3 * 10 ** (-1.1), rtol=1e-12)
    assert dp.bitstat_tau_s == pytest.approx(1.0 / (2 * math.pi * 20e9), rel=1e-12)
    dp0 = DatapathHw.from_plant_params(base)
    np.testing.assert_allclose(dp0.link_t, 10 ** (-0.3), rtol=1e-12)             # module default 3 dB
    mon = MonitorHw.from_plant_params(pp)
    assert mon.stuck_flagged == 0
    np.testing.assert_allclose(pp.monitor.baseline_a, 12e-6)
    assert pp.monitor.mux_order == "bank"
    from mrm.plant.sensing import ReadbackHw
    rb = ReadbackHw.from_plant_params(pp)
    assert rb.adc_bits == 12 and rb.average == 1
    np.testing.assert_allclose(rb.noise_codes, 0.5)
    # absent dark-current activation energy realizes to the materials default (0.37 eV), not 0 (07 P1-9)
    from pathlib import Path
    from mrm.materials import ge_pd_dark_ea_ev
    mini = Path(__file__).resolve().parents[1] / "data" / "mini.yaml"
    text = "".join(ln for ln in mini.read_text().splitlines(keepends=True) if "dark_activation_ev" not in ln)
    (tmp_path / "mini_no_ea.yaml").write_text(text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp_t = params.realize(params.load(str(tmp_path / "mini_no_ea.yaml")), 1, uncertainty="nominal",
                              variation="off", thermal_hook=None)
    np.testing.assert_allclose(pp_t.monitor.dark_activation_j, ge_pd_dark_ea_ev() * 1.602176634e-19, rtol=1e-12)
    # temperature sensors carry their converter width and code-zero temperature
    assert int(base.temp_sensors.adc_bits[0]) == 12
    assert base.temp_sensors.code_zero_k[0] == pytest.approx(233.15)
    ts = TempSensorsHw.from_plant_params(base)
    assert int(ts.code_max[0]) == (1 << 12) - 1


_PRESETS = sorted(p.stem for p in (__import__("pathlib").Path(params.__file__).parent / "presets").glob("*.yaml"))


@pytest.mark.parametrize("preset", _PRESETS)
def test_every_preset_builds_every_leaf(preset):
    """Every shipped preset realizes into PlantParams that every leaf adapter accepts (sampled worlds, B = 3)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load(preset)
        pp = params.realize(cfg, 3, seeds=4)
    assert pp.thermal.system is not None
    P = ph.PhaseParams.from_plant_params(pp)
    assert np.all(np.isfinite(P.ring))
    hp = el.HeaterDriverParams.from_plant_params(pp)
    el.BiasDacParams.from_plant_params(pp)
    MonitorHw.from_plant_params(pp)
    TempSensorsHw.from_plant_params(pp)
    DatapathHw.from_plant_params(pp)
    LaserHw.from_plant_params(pp)
    assert hp.law.shape == (3,)
    spec = ControllerSpec.from_config(cfg, n_batch=3, nominal=cfg)
    assert spec.period_ticks >= 1
    sc = scen_registry.from_config(cfg)
    sc.compile(dt_s=pp.timing.dt_s, n_rings=pp.n_rings, n_lines=pp.n_lines, batch=3,
               aggressor_sites=tuple(pp.thermal.system.src_names[i] for i in pp.thermal.system.aggressor_src))
