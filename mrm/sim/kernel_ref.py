"""Pure-NumPy reference of the kernel (07 P1-18), ported from tests/verification/test_plant_wiring.py::ReferencePlant.

Owner: engine. :class:`ReferenceEngine` advances the same B worlds as mrm.sim.kernel, in the same event order
(docs/handoff/ENGINE_BRIEF.md section 3; mrm/core/timing.py), but written independently: a per-tick Python loop over
the leaf modules' Python bank classes (HeaterBank, BiasBank, MonitorBank, TempSensorBank, ReadbackBank,
DatapathBank, LaserBank), the batched thermal / self-heating kernels, the io.ActuatorBank controller pipeline, a
NumPy Truth, a NumPy oracle (mrm.plant.optics.ring_exact) and NumPy metrics computed from the full trace
(mrm.sim.metrics.metrics_from_trace). It exists so a student can step through one tick in pdb, and so the compiled
kernel has something to be compared with (tests/verification/test_kernel_parity.py, also under NUMBA_DISABLE_JIT=1).

Scope: the fixed substep policy, Python or compiled controllers (the latter through their Python protocol), no
fast_kernel (that path is tested against a 1 MHz Python step, 07 P0-7). Build it from a :class:`mrm.plant.plant.Plant`
so both see identical parameters, seeds and channel maps.
"""
from __future__ import annotations

import math

import numpy as np

from mrm.control.api import Observation
from mrm.core import rng as _rng
from mrm.plant import electrical as el
from mrm.plant import optics as op
from mrm.plant import phase as ph
from mrm.plant import selfheat as shm
from mrm.plant import thermal as th
from mrm.plant.datapath import DatapathBank
from mrm.plant.io import ActuatorBank, SensorBank
from mrm.plant.laser import LaserBank
from mrm.plant.sensing import (FAULT_DAC_GLITCH, MonitorBank, ReadbackBank, TempSensorBank, fault_mask)
from mrm.sim import kernel as K
from mrm.sim.metrics import TRUTH_FIELDS, metrics_from_trace
from mrm.units import C0

__all__ = ["ReferenceEngine"]

_NEVER = -(1 << 62)


def _cold_x(rp_row, x0, target):
    """The root x of n(x) L / x = target on the phase model's cold curve (mrm.plant.phase equation 2), Newton with
    the exact derivative, five fixed iterations (written independently of mrm.sim.kernel.cold_coordinate, same
    floating-point operations so that the parity is bit for bit)."""
    lam_r = float(rp_row[ph.RP_LAM_RES])
    ne = float(rp_row[ph.RP_N_EFF])
    ng = float(rp_row[ph.RP_N_G])
    dng = float(rp_row[ph.RP_DNG])
    ln = float(rp_row[ph.RP_LEN])
    x = float(x0)
    for _ in range(5):
        dd = x - lam_r
        n_x = ne + (ne - ng) * dd / lam_r - 0.5 * dng * dd * dd / lam_r
        ng_x = ng + dng * dd + 0.5 * dng * dd * dd / lam_r
        g = n_x * ln / x - target
        x = x + g * x * x / (ng_x * ln)
    return x


class ReferenceEngine:
    """NumPy reference of B worlds of a Plant (module docstring). ``step(controller)`` advances one tick;
    ``run(n, controller)`` advances n ticks. Per-tick traces of Truth are kept in ``self.trace`` (list of dicts)
    and the obs vector of every controller edge in ``self.obs_log`` ({tick: (B, obs_size)}).

    airy (default True): the self-heating optics adapter the engine uses (mrm.plant.optics selfheat_pabs /
    selfheat_commit / selfheat_bounds, whose exact ring without backscatter runs the engine's own Airy path, so the
    parity can be bit for bit). airy=False: the general ring kernels for every ring (selfheat_pabs_general /
    selfheat_commit_general: the closed forms of _airy_line; selfheat_bounds_general: the separate bound kernels),
    an implementation independent of the engine's self-heating optics, which agrees with the engine to rounding and
    the self-heating root tolerance instead of bit for bit (tests/verification/test_kernel_parity.py;
    docs/handoff/perf.md section 6, item 4)."""

    def __init__(self, plant, *, oracle: bool = True, airy: bool = True):
        self.plant = plant
        self.airy = bool(airy)
        self._pabs_fn = op.selfheat_pabs if airy else op.selfheat_pabs_general
        self._commit_fn = op.selfheat_commit if airy else op.selfheat_commit_general
        self._bounds_fn = op.selfheat_bounds if airy else op.selfheat_bounds_general
        pp, P = plant.pp, plant.P
        self.pp, self.P = pp, P
        B, N, L = plant.B, plant.N, plant.L
        self.B, self.N, self.L = B, N, L
        self.dt = plant.dt
        self.sys = sysm = plant.sys
        self.spec = plant.spec
        self.maps = plant.maps
        nseeds = _rng.batch_seeds(plant.noise_seed, B)
        sseeds = _rng.batch_seeds(plant.scenario_seed, B)
        self.phase = ph.PhaseParams.from_plant_params(pp)
        self.heater = el.HeaterBank(plant.heater_params, initial_code=plant.S.ri[:, K.RI_HE_REG])
        self.bias = el.BiasBank(plant.bias_params, initial_code=plant.S.ri[:, K.RI_BI_REG])
        self.arr = plant.scen_arrays
        self.cache = self.arr.new_cache()
        self.laser = LaserBank(plant.laser_hw, scenario_seeds=sseeds, noise_seeds=nseeds,
                               time_compression=plant.scenario.time_compression)
        pop = plant._pop_period
        self.mon = MonitorBank(plant.monitor_hw, nseeds, m_max=self.spec.samples_per_step, pop_period_ticks=pop)
        self.temps = TempSensorBank(plant.temp_hw, nseeds, pop_period_ticks=pop)
        self.rbk = ReadbackBank(plant.readback_hw, nseeds, pop_period_ticks=pop) if plant.readback_hw else None
        self.dp = DatapathBank(plant.datapath_hw, nseeds, pop_period_ticks=pop)
        self.sensors = SensorBank(self.mon, self.temps, self.rbk, self.dp)
        self.act = ActuatorBank(self.maps, B, N, L, plant.controller_latency_ticks, self.spec.period_ticks,
                                init_heater=plant.S.ri[:, K.RI_HE_REG],
                                init_bias=plant.S.ri[:, K.RI_BI_REG])
        self.nseeds = nseeds
        self.bias_key = np.array([_rng.stream_key(int(s), _rng.stream_id("dac_bias")) for s in nseeds], np.uint64)
        # thermal + self-heating state
        self.coef = sysm.coefficients(self.dt)
        self.x = sysm.zero_state(B)
        self.src_scale = np.ones((1, sysm.n_src))
        t_amb = plant.S.wf[:, K.WF_T_BASE0].copy()
        self.p_prev = np.zeros((B, N))
        self.t_branch = plant.S.rf[:, K.RF_T_BRANCH].copy()            # set by the reset protocol
        self.t_ring = np.repeat(t_amb[:, None], N, axis=1)
        self.t_heater = self.t_ring.copy()
        self.t_obs = np.repeat(t_amb[:, None], sysm.n_obs, axis=1)
        self.g_work = np.zeros((B, sysm.n_obs))
        self.p_new = np.zeros((B, N))
        self.t_new = np.zeros((B, N))
        self.n_iter = np.zeros((B, N), np.int64)
        self.u_heater = np.zeros((B, N))
        self.args_f = plant._args_f0.copy()
        self.args_i = np.asarray(plant.P.sh_args_i).copy()
        self.p_max = np.zeros((B, N))
        self.d2p_max = np.zeros((B, N))
        self.dpdt_max = np.zeros((B, N))
        self.tick = int(plant.tick)
        self.k_reset = int(plant.k_reset)
        self.ramp_ticks = int(P.ramp_ticks)
        self.protocol = int(P.reset_protocol)
        self.edge_j = int(plant.S.wi[0, K.WI_EDGE_IDX])
        self.oracle_on = bool(oracle) and bool(P.oracle_on)
        self.oracle = np.zeros((B, N))
        self.oracle_unr = np.zeros((B, N))
        self.oracle_next = self.tick
        self.last = {"rho": np.zeros((B, N)), "out_r": np.zeros((B, op.N_R, N)), "p_bg": np.zeros((B, N)),
                     "t_pd": np.repeat(t_amb[:, None], N, axis=1)}
        self.trace: list[dict] = []
        self.obs_log: dict[int, np.ndarray] = {}
        self.assign = self.act.assign
        self.reg0 = plant.S.ri[:, K.RI_HE_REG].copy()
        self.lock_ref_oracle = plant.metrics_spec.lock_reference == "oracle"
        self.lock_ref = plant.metrics_spec.reference_m(B, N)
        self.shift = np.zeros((B, N, 2))
        self._ctrl_reset_prev = np.zeros(B, bool)

    # -------------------------------------------------------------------------------------------- helpers
    def _views(self, b):
        return op.selfheat_views(self.args_f[b], self.args_i[b])

    def _edge_tick(self, j):
        return self.P.phase + j * self.P.period

    def _declared(self):
        r2l = np.asarray(self.P.ring_to_line)
        a = self.act.assign
        return np.where((a >= 0) & (a < self.L), a, r2l[None, :])

    # -------------------------------------------------------------------------------------------- edge
    def _release(self, k):
        B, N = self.B, self.N
        last = self.last

        def bitstat():
            self.dp.bitstat_from_monitor(k, self.mon, last["out_r"][:, op.R_DROP_SUM1],
                                         last["out_r"][:, op.R_DROP_SUM0], self.act.bs_thr, density=last["rho"],
                                         p_bg_w=last["p_bg"], t_pd_k=last["t_pd"], t_adc_k=last["t_pd"],
                                         pga_select=self.act.pga,
                                         rin_per_hz=np.asarray(self.pp.laser.rin_per_hz).reshape(-1))
        sensors, new = self.sensors.release(k, heater_applied=self.heater.applied_codes.astype(np.uint16),
                                            bias_applied=self.bias.applied_codes.astype(np.uint16), bitstat=bitstat)
        sensors = {kk: v for kk, v in sensors.items() if kk in self.spec.channels}
        new = {kk: v for kk, v in new.items() if kk in self.spec.channels}
        return sensors, new

    def _controller_step(self, k, controller):
        sensors, new = self._release(k)
        obs = self.spec.layout.pack_obs(k, sensors, new)
        self.obs_log[k] = obs
        if controller is None:
            return
        actions = controller.step(Observation(k, self.dt, sensors, new))
        val = self.plant.validator.validate(actions)
        cur = self.act.act[:, :self.maps.act_size]
        self.spec.layout.pack_act(val, cur)
        if getattr(controller, "act", None) is not None and controller.act.shape == cur.shape:
            controller.act[...] = cur              # the held command is the clipped one, as in the kernel
        self.act.push(k, self.act.act)

    # -------------------------------------------------------------------------------------------- one tick
    def step(self, controller=None):
        k = self.tick
        pp, sysm, B, N, L, P = self.pp, self.sys, self.B, self.N, self.L, self.P
        if k == self._edge_tick(self.edge_j):
            self._controller_step(k, controller)
            self.edge_j += 1
        hc, hv, bc, bv, popped = self.act.pop(k)
        popped_r = hv.copy()
        ev0 = self.arr.evaluate(k, self.cache)
        ev1 = self.arr.evaluate(k + 1, self.cache)
        # laser lines, laser_ramp factor (07 P0-1 item 6), chip input
        ef = np.clip(ev0["laser.line_enable"], 0.0, 1.0)
        f_hz, p_w = self.laser.tick(k, enable=ef, f_offset_hz=ev0["laser.dfreq_hz"],
                                    p_scale=10.0 ** (ev0["laser.power_delta_db"] / 10.0),
                                    tune_codes=self.act.tune)
        ramp = 1.0
        if self.protocol == 0 and self.ramp_ticks > 0:
            ramp = min(max((k - self.k_reset + 1) / self.ramp_ticks, 0.0), 1.0)
        loss = 10.0 ** (-ev0["optical.input_loss_delta_db"] / 10.0)
        tin = np.asarray(pp.laser.input_transmission, float).reshape(-1, 1)
        p_chip = p_w * tin * loss * ramp
        lam = C0 / f_hz
        # ones density
        rho = self.dp.pattern(k, density=ev0["data.ones_density"], traffic_on=ev0["data.driver_enable"])
        # fault mask and dac_glitch
        fault = fault_mask(B, N, adc_drop=ev0["fault.adc_drop"], adc_stuck=ev0["fault.adc_stuck"],
                           adc_saturate=ev0["fault.adc_saturate"], dac_glitch_codes=ev0["fault.dac_glitch_codes"])
        code_add = np.where(fault & FAULT_DAC_GLITCH, np.rint(ev0["fault.dac_glitch_codes"]), 0).astype(np.int64)
        # bias DAC (scenario-owned absolute bias, or the controller's command)
        z = np.zeros((B, N))
        for b in range(B):
            for n in range(N):
                z[b, n] = _rng.normal(self.bias_key[b], n, k, 0)
        if P.bias_ctrl == 0:
            bp = self.plant.bias_params
            vs = ev0["tx.bias_v"]
            vs = np.where(np.isfinite(vs), vs, bp.v_min_v)
            cmd_b = np.clip(np.rint((vs - bp.v_min_v) / bp.lsb_v), 0, bp.code_max).astype(np.int64)
            bout = self.bias.step(k, cmd=cmd_b, z=z, v_offset=ev0["tx.bias_offset_v"])
        else:
            bout = self.bias.step(k, cmd=bc, cmd_valid=bv, z=z, v_offset=ev0["tx.bias_offset_v"])
        # NRZ levels
        swing = np.asarray(P.drv_swing) * ev0["data.swing_scale"]
        v_lo, v_hi, fl = np.zeros((B, N)), np.zeros((B, N)), np.zeros((B, N), np.int8)
        for b in range(B):
            el.nrz_levels_row(bout.v_avg[b], swing[b], int(P.drv_level[b]), el.JUNCTION_GRID_V_MIN,
                              el.JUNCTION_GRID_V_MAX, v_lo[b], v_hi[b], fl[b])
        # optics inputs at the previous ring temperature, into the self-heating rows
        cj_eff = np.zeros((B, N))
        r_s = np.zeros((B, N))
        self.shift = np.zeros((B, N, 2))
        Ph = self.phase
        for b in range(B):
            o = {n_: np.zeros((N, L)) for n_ in ("phi0", "phi1", "a0", "a1", "e0", "e1", "k2t", "k2d", "trt")}
            shift, cj, rs, jfl = np.zeros((N, 2)), np.zeros((N, 2)), np.zeros((N, 2)), np.zeros(N, np.int8)
            r = Ph.table_row(b)
            ph.ring_optics_row(Ph.ring[b], np.ascontiguousarray(lam[b]), self.t_ring[b], self.t_heater[b],
                               np.ascontiguousarray(v_lo[b]), np.ascontiguousarray(v_hi[b]), Ph.grid, Ph.dneff[r],
                               Ph.alpha_fca[r], Ph.capacitance[r], Ph.r_series[r], o["phi0"], o["phi1"], o["a0"],
                               o["a1"], o["e0"], o["e1"], o["k2t"], o["k2d"], o["trt"], shift, cj, rs, jfl)
            r_s[b] = 0.5 * (rs[:, 0] + rs[:, 1])
            self.shift[b] = shift
            v = self._views(b)
            (p_in, t2, t2o, p1, phi0r, phi1r, _, _, a0, a1, e0, e1, psi, k2t, k2d, trt, dpl, tref, s0, beta,
             *_rest) = v
            p_in[:] = p_chip[b]
            p1[:] = rho[b]
            phi0r[:], phi1r[:] = o["phi0"], o["phi1"]
            a0[:], a1[:], e0[:], e1[:] = o["a0"], o["a1"], o["e0"], o["e1"]
            k2t[:], k2d[:], trt[:] = o["k2t"], o["k2d"], o["trt"]
            # d phi / d lambda_res = 2 pi n_g(x) L / x^2 at the shifted coordinate x = lambda_line - shift
            # (mrm.plant.phase equation 2)
            xs = lam[b][None, :] - shift[:, 0:1]
            dpl[:] = 2.0 * math.pi * C0 * o["trt"] / (xs * xs)
            rp = Ph.ring[b]
            for n in range(N):
                tref[n] = self.t_ring[b, n]
                s0[n], beta[n] = ph.local_thermo_optic(self.t_ring[b, n], rp[n, ph.RP_S0], rp[n, ph.RP_BETA],
                                                       rp[n, ph.RP_T0])
            # the engine's cumulative-table form (equal to electrical.junction_swing_capacitance_row to 1e-12,
            # tests/verification/test_kernel_parity.py), so that the parity can be bit for bit
            K.swing_capacitance_row(v_lo[b], v_hi[b], Ph.grid, Ph.capacitance[r], P.jt_cum[r], rp[:, ph.RP_FILL],
                                    rp[:, ph.RP_LEN], cj_eff[b])
            self._bounds_fn(self.args_f[b], self.args_i[b], float(P.dt_max_k), self.p_max[b], self.d2p_max[b],
                            self.dpdt_max[b])
        # data-activity heat
        pj, pd = np.zeros((B, N)), np.zeros((B, N))
        for b in range(B):
            en = (ev0["data.driver_enable"][b] >= 0.5).astype(float)
            el.data_activity_heat_row(cj_eff[b], v_hi[b] - v_lo[b], float(P.dp_bitrate[b]),
                                      ev0["data.transition_density"][b], en, r_s[b], P.drv_r_out[b:b + 1],
                                      P.drv_p_static[b:b + 1], P.drv_e_tr[b:b + 1], 0, pj[b], pd[b])
        # thermal inputs
        u0 = np.zeros((B, sysm.n_src))
        u1 = np.zeros((B, sysm.n_src))
        for u, ev in ((u0, ev0), (u1, ev1)):
            for n in range(N):
                if sysm.junction_src.size > n and sysm.junction_src[n] >= 0:
                    u[:, sysm.junction_src[n]] = pj[:, n]
                if sysm.driver_src.size > n and sysm.driver_src[n] >= 0:
                    u[:, sysm.driver_src[n]] = pd[:, n]
            if sysm.aggressor_src.size:
                u[:, sysm.aggressor_src] = ev["thermal.aggressor_w"]
            u[:, sysm.heater_src] = self.u_heater
            u[:, sysm.wg_src] = self.p_prev
        t_base_end = ev1["thermal.t_base_k"][:, 0] + ev1["thermal.xpu_rise_k"][:, 0]
        a, c0, c1 = self.coef
        pre = np.zeros((B, N))
        g = np.zeros((B, N))
        for b in range(B):
            for n in range(N):
                pre[b, n], g[b, n] = th.observer_split_b(self.x[b], a[b], c0[b], c1[b], sysm.lag_src, sysm.obs_ptr,
                                                         int(sysm.heater_obs[n]), int(sysm.heater_src[n]), u0[b],
                                                         u1[b], self.src_scale[0])
        hout = self.heater.step(k, hc, hv, t_base_end[:, None] + pre, g, code_add=code_add)
        self.u_heater = hout.p_w.copy()
        u0[:, sysm.heater_src] = self.u_heater
        u1[:, sysm.heater_src] = self.u_heater
        if P.self_heating:
            nf = shm.implicit_step(self.x, a, c0, c1, sysm.lag_src, sysm.lag_obs, sysm.lag_kind, sysm.opt_lags,
                                   sysm.obs_ptr, sysm.src_ring, sysm.ring_obs, np.asarray(P.bus_order), u0, u1,
                                   self.src_scale, t_base_end, self.p_prev, self.t_branch, self._pabs_fn,
                                   self._commit_fn, self.args_f, self.args_i, self.p_max, self.d2p_max,
                                   int(P.sh_max_iter), float(P.sh_tol), int(P.sh_n_outer), self.g_work, self.t_obs,
                                   self.p_new, self.t_new, self.n_iter)
            self.p_prev[:] = self.p_new
            self.t_branch[:] = self.t_new
            self.t_ring[:] = self.t_new
        else:
            for b in range(B):
                v = self._views(b)
                op.bus_propagate_nb(v[0], np.asarray(P.bus_order), v[1], v[2][0], v[3], v[4], v[5], v[8], v[9],
                                    v[10], v[11], v[12], v[13], v[14], v[15], v[20], v[16],
                                    self.args_i[b, op.SH_TWO] != 0, bool(P.use_tcmt), int(P.m_orders), False, v[22],
                                    v[23], v[24], v[21], v[25], v[26])
            u0[:, sysm.wg_src] = 0.0
            u1[:, sysm.wg_src] = 0.0
            th.thermal_advance(self.x, a, c0, c1, sysm.lag_src, sysm.obs_ptr, u0, u1, self.src_scale, t_base_end,
                               self.t_obs)
            self.t_ring[:] = self.t_obs[:, sysm.ring_obs]
            self.t_branch[:] = self.t_ring
            self.p_prev[:] = 0.0
        self.t_heater[:] = self.t_obs[:, sysm.heater_obs]
        # sensors of the interval
        out_r = np.stack([self._views(b)[23] for b in range(B)]).copy()
        out_rl = np.stack([self._views(b)[22] for b in range(B)]).copy()
        out_l = np.stack([self._views(b)[24] for b in range(B)]).copy()
        p_bus = out_rl[:, op.RL_PIN].sum(axis=2)
        pdo = np.asarray(P.th_pd_obs)
        t_pd = np.where(pdo[None, :] >= 0, self.t_obs[:, np.maximum(pdo, 0)], self.t_ring)
        self.mon.tick(k, p_pd_w=out_r[:, op.R_DROP_SUM], p_bg_w=p_bus, t_pd_k=t_pd, pga_select=self.act.pga,
                      fault=fault)
        decl = self._declared()
        bidx = np.arange(B)[:, None]
        nidx = np.arange(N)[None, :]
        oma = np.abs(out_rl[bidx, op.RL_POUT1, nidx, decl] - out_rl[bidx, op.RL_POUT0, nidx, decl])
        self.dp.errors(k, oma)
        if self.temps.hw.S:
            self.temps.tick(k, self.t_obs[:, sysm.temp_obs])
        if self.rbk is not None:
            self.rbk.tick(k, hout.v_v, hout.i_a)
        # Truth (NumPy, independent of mrm.sim.kernel.truth_row)
        tr = self._truth(k, out_rl, out_r, out_l, p_chip, lam, decl, t_pd, p_bus, hout, bout, oma, t_base_end)
        if self.oracle_on and k + 1 > self.oracle_next:
            self._oracle(lam, decl, rho)
            self.oracle_next = k + int(P.oracle_period)
        tr["oracle_reloma"] = self.oracle.copy()
        tr["lock_ref"] = self.lock_ref.copy()
        tr["oracle_unr"] = self.oracle_unr.copy()
        tr["popped"] = popped_r
        tr["bits"] = self.dp.nb.astype(float).copy()
        tr["reg"] = self.heater.applied_codes.copy()
        self.trace.append(tr)
        self.last = {"rho": rho.copy(), "out_r": out_r, "p_bg": p_bus, "t_pd": t_pd}
        self.tick = k + 1
        return tr

    def _truth(self, k, out_rl, out_r, out_l, p_chip, lam, decl, t_pd, p_bus, hout, bout, oma, t_base_end):
        B, N, L, P = self.B, self.N, self.L, self.P
        det = np.zeros((B, N))
        order = np.zeros((B, N), np.int64)
        line_on = np.full((B, N), -1, np.int64)
        fwhm_hz = np.asarray(P.fwhm_hz)
        for b in range(B):
            v = self._views(b)
            phi0r, phi1r, dpl, tref, s0, beta = v[4], v[5], v[16], v[17], v[18], v[19]
            dT = self.t_ring[b] - tref
            shift = s0 * (dT + 0.5 * beta * dT * dT)
            ph_ = 0.5 * (phi0r + phi1r) + dpl * shift[:, None]
            m = np.floor(ph_ / (2 * math.pi) + 0.5)
            d = (ph_ - 2 * math.pi * m) / dpl                         # (N, L), first order (wrong-line window)
            order[b] = m[np.arange(N), decl[b]].astype(np.int64)
            for n in range(N):                                        # the declared line exactly
                l = int(decl[b, n])
                x_end = lam[b, l] - (0.5 * (self.shift[b, n, 0] + self.shift[b, n, 1]) + shift[n])
                det[b, n] = _cold_x(self.phase.ring[b, n], x_end, float(order[b, n])) - x_end
            fw = lam[b][None, :] ** 2 * fwhm_hz[b][:, None] / C0
            inwin = np.abs(d) <= float(P.wrong_window) * fw
            dd = np.where(inwin, np.abs(d), np.inf)
            best = np.argmin(dd, axis=1)
            line_on[b] = np.where(np.isfinite(dd[np.arange(N), best]), best, -1)
        mode = int(P.assign_mode)
        wrong = np.zeros((B, N), bool)
        for b in range(B):
            on = line_on[b] >= 0
            if mode == 0:
                wrong[b] = on & (line_on[b] != decl[b])
            elif mode == 1:
                offs = (line_on[b] - decl[b]) % L
                if on.any():
                    vals, cnt = np.unique(offs[on], return_counts=True)
                    common = vals[np.argmax(cnt)]
                    wrong[b] = on & (offs != common)
        pin = out_rl[np.arange(B)[:, None], op.RL_PIN, np.arange(N)[None, :], decl]
        t1 = out_rl[np.arange(B)[:, None], op.RL_THRU1, np.arange(N)[None, :], decl]
        t0 = out_rl[np.arange(B)[:, None], op.RL_THRU0, np.arange(N)[None, :], decl]
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(pin > 0, np.abs(t1 - t0) / pin, 0.0)
        lam_decl = lam[np.arange(B)[:, None], decl]
        hw = 0.5 * lam_decl ** 2 * fwhm_hz / C0
        mhw = self.plant.monitor_hw
        i_sig, i_d = mhw.photocurrent_a(out_r[:, op.R_DROP_SUM], p_bus, t_pd)
        dp = self.plant.datapath_hw
        from scipy.special import erfc
        q = dp.q_ref * oma * dp.link_t[:, None] / dp.oma_sens_w
        ber = np.where(oma * dp.link_t[:, None] > 0, 0.5 * erfc(q / math.sqrt(2.0)), 0.5)
        ber = np.maximum(ber, dp.ber_floor)
        return {"tick": k, "t_ring_k": self.t_ring.copy(), "t_heater_k": self.t_heater.copy(), "t_pd_k": t_pd,
                "p_heater_w": hout.p_w.copy(), "p_abs_w": out_r[:, op.R_ABS_SUM].copy(),
                "lam_res_m": lam_decl + det, "detuning_m": det, "detuning_hwhm": det / hw,
                "p_drop_w": out_r[:, op.R_DROP_SUM].copy(), "i_monitor_a": i_sig + i_d, "p_in_w": pin, "oma_w": oma,
                "reloma": rel, "heater_code": self.heater.applied_codes.astype(float),
                "bias_v": bout.v_avg.copy(), "order": order, "line_on": line_on, "wrong": wrong,
                "t_base_k": np.repeat(t_base_end[:, None], N, axis=1), "ber": ber, "declared_line": decl.copy(),
                "p_out_w": out_l[:, op.L_POUT].copy(), "p_chip_w": p_chip.copy()}

    def _oracle(self, lam, decl, rho):
        """NumPy oracle with mrm.plant.optics.ring_exact (same scan and stability rule as mrm.sim.kernel._oracle)."""
        B, N, P = self.B, self.N, self.P
        npts = int(self.plant.S.or_phi.shape[1])
        x = -1.5 + 3.0 * np.arange(npts) / (npts - 1)
        z_opt = np.asarray(P.z_opt_self)
        if int(P.use_tcmt):
            m_ord = int(P.m_orders)

            def ring(*a, **kw):
                return op.ring_tcmt(*a, m_orders=m_ord, **kw)
        else:
            ring = op.ring_exact
        for b in range(B):
            v = self._views(b)
            phi0r, phi1r, a0, a1, e0, e1, psi, k2t, k2d, trt, dpl = (v[4], v[5], v[8], v[9], v[10], v[11], v[12],
                                                                     v[13], v[14], v[15], v[16])
            tref, s0, beta = v[17], v[18], v[19]
            out_rl = v[22]
            rb = v[20]
            two = self.args_i[b, op.SH_TWO] != 0
            for n in range(N):
                l = int(decl[b, n])
                fwhm = lam[b, l] ** 2 * float(P.fwhm_hz[b, n]) / C0
                dphi = phi1r[n, l] - phi0r[n, l]
                mean = 0.5 * (phi0r[n, l] + phi1r[n, l])
                m = math.floor(mean / (2 * math.pi) + 0.5)
                dT = self.t_ring[b, n] - tref[n]
                g_sh = z_opt[b, n] * s0[n] * (1.0 + beta[n] * dT)
                base = 2 * math.pi * m + dpl[n, l] * x * fwhm
                r0 = ring(base - 0.5 * dphi, k2t[n, l], k2d[n, l], a0[n, l], trt[n, l], rb[n], psi[n, l])
                r1 = ring(base + 0.5 * dphi, k2t[n, l], k2d[n, l], a1[n, l], trt[n, l], rb[n], psi[n, l])
                q1 = rho[b, n] if two else 0.0
                rel = np.abs(r1.p_thru - r0.p_thru)
                slope = out_rl[op.RL_PIN, n, l] * ((1 - q1) * e0[n, l] * r0.dloss_dphi
                                                   + q1 * e1[n, l] * r1.dloss_dphi) * dpl[n, l]
                stable = g_sh * slope < 1.0
                self.oracle_unr[b, n] = rel.max()
                self.oracle[b, n] = rel[stable].max() if stable.any() else 0.0
                if self.lock_ref_oracle:
                    # first maximum among the stable points (all points when none is stable), as the kernel's scan,
                    # refined by the parabola through it and its neighbours (same side of the stability limit)
                    ib = int(np.argmax(np.where(stable, rel, -1.0))) if stable.any() else int(np.argmax(rel))
                    d_ib = x[ib] * fwhm
                    if 0 < ib < npts - 1 and (not stable.any() or (stable[ib - 1] and stable[ib + 1])):
                        den = rel[ib - 1] - 2.0 * rel[ib] + rel[ib + 1]
                        if den < 0.0:
                            off = min(max(0.5 * (rel[ib - 1] - rel[ib + 1]) / den, -0.5), 0.5)
                            d_ib = d_ib + off * 3.0 / (npts - 1) * fwhm
                    x_now = lam[b, l] - (0.5 * (self.shift[b, n, 0] + self.shift[b, n, 1])
                                         + s0[n] * (dT + 0.5 * beta[n] * dT * dT))
                    x_m = _cold_x(self.phase.ring[b, n], x_now, float(m))
                    self.lock_ref[b, n] = x_m - _cold_x(self.phase.ring[b, n], x_m,
                                                        m + dpl[n, l] * d_ib / (2 * math.pi))

    # -------------------------------------------------------------------------------------------- runs
    def run(self, n_ticks: int, controller=None):
        for _ in range(int(n_ticks)):
            self.step(controller)

    def stacked(self, name: str) -> np.ndarray:
        """(T, B, N) trace of a Truth field."""
        return np.stack([t[name] for t in self.trace])

    def metrics(self, k0: int | None = None, reg0=None) -> dict:
        """NumPy metrics of the whole trace (mrm.sim.metrics.metrics_from_trace)."""
        pl = self.plant
        st = self.stacked
        return metrics_from_trace(pl.metrics_spec, self.dt, det_m=st("detuning_m"), order=st("order"),
                                  wrong=st("wrong"), reloma=st("reloma"), oracle=st("oracle_reloma"),
                                  oracle_unr=st("oracle_unr"), p_heater_w=st("p_heater_w"), p_abs_w=st("p_abs_w"),
                                  reg=st("reg"), popped=st("popped"), ber=st("ber"), bits=st("bits"),
                                  code_lo=int(pl.P.he_lo), code_hi=int(pl.P.he_hi),
                                  reg0=self.reg0 if reg0 is None else reg0,
                                  k0=self.k_reset if k0 is None else k0, lock_ref_m=st("lock_ref"))
