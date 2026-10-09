"""Plant: the Python API over the compiled kernel, and Truth (ARCHITECTURE section 2 ``plant.py``; 07 P0-6, P0-7,
P0-9, P1-1).

Owner: engine. A :class:`Plant` holds B worlds of one realized bank (PlantParams from mrm.params.realize), its
scenario (compiled at the plant tick), the controller interface (ControllerSpec), the metric accumulators and the
record buffers, and advances them with mrm.sim.kernel.run_block. Interactive use::

    plant = Plant(pp, scenario, spec)                    # spec: ControllerSpec.from_config(cfg, n_batch=B, ...)
    plant.reset(noise_seed=0)                            # 07 P0-1 item 6 reset protocol of pp.timing
    plant.advance({"tx.heater": codes}, n_ticks=1000)    # codes latch at the current tick + controller latency
    sensors, new = plant.sample_sensors()                # the channels released at the last controller edge
    truth = plant.truth()                                # Truth: never reaches a controller (07 P1-1)

Time is an int64 tick count (``plant.tick``); ``advance`` takes ticks, never seconds (07 P0-6). Sensors are released
at the declared controller edges whether or not a controller runs (their FIFOs are sized for that period), so
``sample_sensors`` returns what a controller stepping at the last edge would have seen. The runner
(mrm.sim.runner.run) drives the same object with a compiled or a Python controller.

Seeds (07 P1-13, mrm.core.rng): world b uses noise seed ``noise_seed + b`` and scenario seed ``scenario_seed + b``;
device draws come from ``pp.seeds['device']`` (fixed by realize); ``reset(noise_seed)`` re-keys only the noise.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from mrm.control.api import ActionValidator, ControllerSpec, fast_act_size, fast_obs_size
from mrm.core import rng as _rng
from mrm.core.timing import fifo_capacity
from mrm.plant import electrical as el
from mrm.plant import optics as op
from mrm.plant import phase as ph
from mrm.plant import sensing as sn
from mrm.plant import thermal as th
from mrm.plant.datapath import (PATTERN_PRBS, PRBS_TABLE_MAX_K, DatapathHw, prbs_table)
from mrm.plant.io import ChannelMaps, pipe_capacity, pipe_push_row
from mrm.plant.laser import LaserHw
from mrm.scenarios import generators as sg
from mrm.sim import kernel as K
from mrm.sim.metrics import (N_TR, TRUTH_FIELDS, MetricsSpec, MetricsState, RecordSet, summarize)
from mrm.units import C0

__all__ = ["Plant", "Truth", "prepare_scenario", "DEFAULT_SH_DT_MAX_K"]

#: largest ring temperature change within one step that the self-heating bounds must cover (K). ReferencePlant uses
#: 50 K; a multi-tick auto substep scales it (design choice, see Plant.__init__).
DEFAULT_SH_DT_MAX_K = 50.0
#: |F| tolerance of the implicit self-heating solve (K). The ENGINE_BRIEF suggests 1e-9 K; 1e-7 K (7e-6 pm of
#: resonance at 70 pm/K) saves about one Newton evaluation per ring and tick (design choice; tested in
#: tests/verification/test_engine_physics.py against 1e-10 K).
DEFAULT_SH_TOL_K = 1e-7
_NEVER = -(1 << 62)


@dataclass
class Truth:
    """Ground truth at the current tick (never handed to a controller; 07 P1-1). Per ring (B, N) fields of
    mrm.sim.metrics.TRUTH_FIELDS (temperatures, heater power, resonance of the order nearest the declared line,
    detuning in m and in half-linewidths, absorbed power, monitor current, OMA, relOMA, ER, IL, ...); per line (B, L):
    chip-input power, bus-output power and wavelength; the full optics outputs out_rl (B, N_RL, N, L), out_r
    (B, N_R, N), out_l (B, N_L, L) (field indices RL_*, R_*, L_* of mrm.plant.optics) and every thermal observer
    temperature t_obs (B, O) with ``obs_names``. For Truth-fed controllers (the oracle, mrm.validation.oracle
    TruthState): x (B, M) the thermal lag states (ThermalSystem order), heater_applied (B, N) the latched heater codes
    and conditions, one mrm.control.models.Conditions per world (line wavelengths and chip powers after the
    scenario, the NRZ junction levels, ones and transition density, base temperature, aggressor power, data
    enable) of the last interval."""
    tick: int
    t_s: float
    fields: dict
    p_chip_w: np.ndarray
    p_out_w: np.ndarray
    lam_m: np.ndarray
    out_rl: np.ndarray
    out_r: np.ndarray
    out_l: np.ndarray
    t_obs: np.ndarray
    obs_names: tuple
    x: np.ndarray | None = None
    heater_applied: np.ndarray | None = None
    conditions: list | None = None

    def __getattr__(self, name):
        f = self.__dict__.get("fields")
        if f is not None and name in f:
            return f[name]
        raise AttributeError(name)

    def __getitem__(self, name):
        return self.fields[name]


def prepare_scenario(sc: sg.Scenario, pp, *, ring_t0: str = "lock") -> tuple[sg.Scenario, dict]:
    """Calibrate the pending aggressors of a scenario against this plant (07 P1-4; ENGINE_BRIEF section 1).

    The power waveform is solved from the thermal system's aggressor -> victim-ring Foster response, with the R1
    thermo-optic model of PlantParams. ``ring_t0='lock'`` starts the rise at the ring's lock temperature
    (pp.thermal.t_ring_lock_k; recommended for controller studies, docs/handoff/scenarios.md section 8), 'base' at
    the base temperature. Returns (calibrated scenario, {site: calibration record}).
    """
    if not sc.pending_aggressors:
        return sc, {}
    from mrm.scenarios import calibrate_scenario, foster_from_system
    sysm = pp.thermal.system
    model, resp = {}, {}
    for a in sc.aggressors:
        if not a.pending:
            continue
        m = {"s0_m_per_k": np.asarray(pp.thermal.dlambda_dt_s0_m_per_k)[:, a.victim_ring],
             "beta_per_k": np.asarray(pp.thermal.dlambda_dt_beta_per_k)[:, 0],
             "t_ref_k": np.asarray(pp.thermal.dlambda_dt_t0_k)[:, 0]}
        if ring_t0 == "lock":
            m["ring_t0_k"] = np.asarray(pp.thermal.t_ring_lock_k)[:, a.victim_ring]
        model[a.site] = m
        resp[a.site] = foster_from_system(sysm, victim_ring=a.victim_ring, site=a.site)
    sc2, cal = calibrate_scenario(sc, resp, model=model)
    return sc2, {k: v.record() for k, v in cal.items()}


def _c(a, dtype=np.float64):
    """A writeable C-contiguous copy: every kernel argument has the same numba type (dtype, ndim, 'C', writeable)
    whatever the config, so the compiled kernel is reused (a read-only view would be another type and recompile)."""
    return np.array(a, dtype=dtype, order="C", copy=True)


def _bn(x, B, N, dtype=np.float64):
    return _c(np.broadcast_to(np.asarray(x, dtype=dtype), (B, N)), dtype)


def _b(x, B, dtype=np.float64):
    a = np.asarray(x, dtype=dtype).reshape(-1)
    return _c(np.broadcast_to(a, (B,)) if a.size in (1, B) else a, dtype)


class Plant:
    """B worlds of one bank, advanced by the compiled kernel (module docstring).

    Parameters: ``pp`` PlantParams; ``scenario`` a Scenario (default: static); ``spec`` the ControllerSpec (built
    with ControllerSpec.from_config; its period sets the sensor release edges); ``controller_latency_ticks``
    (Controller.latency_ticks, 07 P0-6); ``noise_seed`` / ``scenario_seed`` base seeds; ``metrics`` a MetricsSpec;
    ``record`` record specs (mrm.sim.metrics.RecordSpec list) with ``record_ticks`` the run length they are sized
    for; ``substep_policy`` 'fixed' | 'auto' (default pp.timing.substep_policy); ``auto_max_ticks`` the largest auto
    substep (power of two, default 64); ``fast_kernel`` True when a compiled controller's fast_kernel will run;
    ``oracle`` evaluate the oracle relOMA for the OMA penalty (07 P1-1); ``histogram`` the detuning histogram.
    """

    def __init__(self, pp, scenario: sg.Scenario | None = None, spec: ControllerSpec | None = None, *, cfg=None,
                 controller_latency_ticks: int = 0, noise_seed: int = 0, scenario_seed: int = 0,
                 metrics: MetricsSpec | None = None, record=None, record_ticks: int = 0,
                 substep_policy: str | None = None, auto_max_ticks: int = 64, fast_kernel: bool = False,
                 oracle: bool = True, histogram: bool | None = None, calibrate_aggressors: bool = True,
                 sh_tol_k: float = DEFAULT_SH_TOL_K):
        self.pp = pp
        self.sh_tol_k = float(sh_tol_k)
        B, N, L = int(pp.batch), int(pp.n_rings), int(pp.n_lines)
        self.B, self.N, self.L = B, N, L
        self.dt = float(pp.timing.dt_s)
        sysm = pp.thermal.system
        if sysm is None:
            raise ValueError("PlantParams has no thermal system (realize with thermal_hook='auto')")
        self.sys = sysm
        sc = sg.Scenario(name="static") if scenario is None else scenario
        self.aggressor_calibration = {}
        if sc.pending_aggressors and calibrate_aggressors:
            sc, self.aggressor_calibration = prepare_scenario(sc, pp)
        self.scenario = sc
        if spec is None:
            if cfg is None:
                raise ValueError("Plant needs a ControllerSpec (or cfg to build one)")
            spec = ControllerSpec.from_config(cfg, n_batch=B, nominal=cfg, scenario_drives=sc.drives,
                                              scenario_offsets=sc.offsets)
        if spec.n_batch != B or spec.n_rings != N or spec.n_lines != L:
            raise ValueError(f"spec is for (B, N, L) = {(spec.n_batch, spec.n_rings, spec.n_lines)}, the plant has "
                             f"{(B, N, L)}")
        if abs(spec.tick_s - self.dt) > 1e-15 * self.dt:
            raise ValueError("spec.tick_s differs from pp.timing.dt_s")
        self.spec = spec
        self.maps = ChannelMaps.from_spec(spec)
        self.validator = ActionValidator(spec)
        self.controller_latency_ticks = int(controller_latency_ticks)
        self.noise_seed = int(noise_seed)
        self.scenario_seed = int(scenario_seed)
        pol = substep_policy if substep_policy is not None else ("fixed", "auto")[int(pp.timing.substep_policy)]
        if pol not in ("fixed", "auto"):
            raise ValueError("substep_policy must be 'fixed' or 'auto' (07 P0-10)")
        self.substep_policy = pol
        self.metrics_spec = MetricsSpec() if metrics is None else metrics
        self._record_specs = record
        self._record_ticks = int(record_ticks)
        self._fast_on = bool(fast_kernel)
        self._oracle_on = bool(oracle) and self.metrics_spec.oracle
        if self.metrics_spec.lock_reference == "oracle" and not self._oracle_on:
            raise ValueError("metrics lock_reference='oracle' needs the oracle (Plant(oracle=True))")
        self._histogram = histogram
        amp = int(max(0, math.floor(math.log2(max(int(auto_max_ticks), 1))))) if pol == "auto" else 0
        self._au_max_pow = amp
        self._build(amp)
        self._ctrl = None
        self.n_threads = None
        self.tick = 0
        self.reset()

    # ------------------------------------------------------------------------------------------------ build
    def _build(self, au_max_pow: int) -> None:
        pp, spec, sysm = self.pp, self.spec, self.sys
        B, N, L = self.B, self.N, self.L
        dt = self.dt
        nseeds = _rng.batch_seeds(self.noise_seed, B)
        sseeds = _rng.batch_seeds(self.scenario_seed, B)
        self._nseeds, self._sseeds = nseeds, sseeds
        # phase / junction
        phs = ph.PhaseParams.from_plant_params(pp)
        self.phase = phs
        # self-heating rows (mrm.plant.optics.pack_selfheat_args), one per world
        method = "tcmt" if int(pp.flags.optics_model) else "exact"
        lam0 = C0 / np.asarray(pp.laser.line_freq_hz, float)
        rows = []
        for b in range(B):
            rows.append(op.pack_selfheat_args(
                np.zeros(L), np.zeros((N, L)), np.full((N, L), 1e-3), np.full((N, L), 1e-4), np.full((N, L), 0.99),
                np.full((N, L), 1e-13), np.full((N, L), 0.5), lam0[b], np.full(N, 300.0), np.full(N, 7e-11), 0.0,
                bus_order=np.asarray(pp.layout.bus_order), rb=np.asarray(pp.optics.backscatter_rb[b]),
                nrz={"p1": np.full(N, 0.5), "phi1": np.zeros((N, L)), "a1": np.full((N, L), 0.99),
                     "eta1": np.full((N, L), 0.5)},
                method=method, m_orders=int(pp.optics.tcmt_orders)))
        args_f0, args_i = K.extend_selfheat_rows(np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows]))
        self._args_f0 = args_f0
        # actuator paths
        hp = el.HeaterDriverParams.from_plant_params(pp)
        bp = el.BiasDacParams.from_plant_params(pp)
        self.heater_params, self.bias_params = hp, bp
        # sensors
        mhw = sn.MonitorHw.from_plant_params(pp)
        self.monitor_hw = mhw
        tsh = sn.TempSensorsHw.from_plant_params(pp)
        self.temp_hw = tsh
        rb_on = bool(getattr(pp.heater, "readback_present", False))
        rbh = sn.ReadbackHw.from_plant_params(pp) if rb_on else None
        self.readback_hw = rbh
        dph = DatapathHw.from_plant_params(pp)
        self.datapath_hw = dph
        if dph.mode == PATTERN_PRBS and dph.prbs_k <= PRBS_TABLE_MAX_K:
            words, pre, prbs_period = prbs_table(dph.prbs_k)
        else:
            words, pre, prbs_period = np.zeros(1, np.int64), np.zeros(2, np.int64), 0
        lhw = LaserHw.from_plant_params(pp)
        self.laser_hw = lhw
        # scenario
        sysm_agg = tuple(sysm.src_names[i] for i in sysm.aggressor_src)
        arr = self.scenario.compile(dt_s=dt, n_rings=N, n_lines=L, batch=B, seed=self.scenario_seed,
                                    aggressor_sites=sysm_agg)
        self.scen_arrays = arr
        # thermal coefficient tables (07 P0-10: exact FOH for any h; powers of two for the auto policy)
        tabs = [sysm.coefficients(dt * (1 << p)) for p in range(au_max_pow + 1)]
        th_a = np.ascontiguousarray(np.stack([t[0] for t in tabs], axis=1))
        th_c0 = np.ascontiguousarray(np.stack([t[1] for t in tabs], axis=1))
        th_c1 = np.ascontiguousarray(np.stack([t[2] for t in tabs], axis=1))
        # controller edges / periods
        period = int(spec.period_ticks)
        pop = period + int(spec.tick_jitter_ticks)
        self._pop_period = pop
        # metrics
        lam_line = np.asarray(pp.optics.lambda_line_m, float)
        fwhm_m = lam_line ** 2 * np.asarray(pp.optics.fwhm_hz, float) / C0
        fsr_m = lam_line ** 2 * np.asarray(pp.optics.fsr_hz, float) / C0
        hist = self._histogram
        if hist is None:
            hist = B * N <= 1024
        ms = MetricsState.build(self.metrics_spec, B, N, dt, fwhm_max_m=float(np.max(fwhm_m)),
                                fsr_max_m=float(np.max(fsr_m)), code_lo=int(hp.code_lo), code_hi=int(hp.code_hi),
                                histogram=hist)
        self.metrics_state = ms
        rec = RecordSet.build(self._record_specs, B, N, max(self._record_ticks, 0))
        self.records_state = rec
        # self-heating numerics
        n_outer_auto = 1
        if au_max_pow > 0:
            n_outer_auto = 2           # Gauss-Seidel sweeps make the lagged cross-ring heating implicit (07 P0-1)
        oracle_pts = int(self.metrics_spec.oracle_points)
        u64 = np.uint64
        P = K.KParams(
            n_rings=N, n_lines=L, dt=dt, self_heating=int(pp.flags.self_heating), use_tcmt=int(pp.flags.optics_model),
            m_orders=int(pp.optics.tcmt_orders), period=period, phase=int(spec.phase_ticks),
            jitter=int(spec.tick_jitter_ticks),
            key_sched=_c([_rng.stream_key(int(s), int(_rng.Stream.SCHEDULE)) for s in nseeds], u64),
            ctrl_latency=int(self.controller_latency_ticks), reset_protocol=int(pp.timing.reset_protocol),
            ramp_ticks=int(pp.timing.reset_ramp_ticks),
            bias_ctrl=int(spec.bindings.get("tx.bias") == "controller"),
            policy=int(self.substep_policy == "auto"), fast_on=int(self._fast_on),
            ph_ring=_c(phs.ring), jt_grid=_c(phs.grid), jt_dn=_c(phs.dneff), jt_al=_c(phs.alpha_fca),
            jt_c=_c(phs.capacitance), jt_rs=_c(phs.r_series),
            jt_cum=_c(K.cumulative_capacitance(phs.grid, phs.capacitance)),
            sh_args_i=_c(args_i, np.int64),
            bus_order=_c(pp.layout.bus_order, np.int64), ring_to_line=_c(pp.layout.ring_to_line, np.int64),
            fwhm_hz=_bn(pp.optics.fwhm_hz, B, N), z_opt_self=_bn(th.selfheat_dc_k_per_w(sysm), B, N),
            dt_max_k=float(DEFAULT_SH_DT_MAX_K), sh_max_iter=10, sh_tol=float(self.sh_tol_k),
            sh_n_outer=1,
            las_f0=_c(lhw.f0_hz), las_p0=_c(lhw.p0_w), las_frate=_c(lhw.f_rate_hz_per_s),
            las_prate=_c(lhw.p_rate_per_s), las_frw=_c(lhw.f_rw_hz_per_rt_s), las_prw=_c(lhw.p_rw_per_rt_s),
            las_rho_f=_b(lhw.rho_f, B), las_rho_p=_b(lhw.rho_p, B), las_rin=_b(lhw.rin_per_hz, B),
            las_module=_c(lhw.lines.source_laser, np.int64), las_nmod=int(lhw.lines.n_lasers),
            las_grid_s=float(lhw.grid_s), las_depth=int(lhw.depth), las_tcomp=float(self.scenario.time_compression),
            las_key_drift=_c([_rng.stream_key(int(s), int(_rng.Stream.LASER_DRIFT)) for s in sseeds], u64),
            las_key_rin=_c([_rng.stream_key(int(s), int(_rng.Stream.LASER_RIN)) for s in nseeds], u64),
            las_tune_lsb=float(lhw.tune_lsb_hz), las_input_t=_b(pp.laser.input_transmission, B),
            dp_mode=int(dph.mode), dp_bpt_i=int(dph.bpt_i), dp_bpt_f=float(dph.bpt_f), dp_words=_c(words, np.int64),
            dp_pre=_c(pre, np.int64), dp_period=int(prbs_period),
            dp_key_pat=_c(sn._keys(nseeds, B, "pattern"), u64), dp_key_dp=_c(sn._keys(nseeds, B, "datapath"), u64),
            dp_link_t=_b(dph.link_t, B), dp_oma_sens=_bn(dph.oma_sens_w, B, N), dp_q_ref=float(dph.q_ref),
            dp_ber_floor=float(dph.ber_floor), dp_err_period=int(dph.error_period_ticks),
            dp_err_lat=_b(dph.error_latency_ticks, B, np.int64), dp_win_bits=int(dph.window_bits),
            dp_bs_tau=float(dph.bitstat_tau_s),
            dp_bs_stia=(-1.0 if dph.bitstat_noise_a_per_rt_hz is None else float(dph.bitstat_noise_a_per_rt_hz) ** 2),
            dp_rin=_b(pp.laser.rin_per_hz, B),
            drv_swing=_bn(np.nan_to_num(np.asarray(pp.datapath.swing_pp_v, float)), B, N),
            drv_level=_b(pp.datapath.level_convention, B, np.int64),
            dp_bitrate=_b(np.nan_to_num(np.asarray(pp.datapath.bitrate_hz, float)), B),
            drv_r_out=_b(np.nan_to_num(np.asarray(pp.datapath.driver_output_resistance_ohm, float)), B),
            drv_p_static=_b(np.nan_to_num(np.asarray(pp.datapath.driver_static_power_w, float)), B),
            drv_e_tr=_b(np.nan_to_num(np.asarray(pp.datapath.driver_energy_per_transition_j, float)), B),
            bi_bits=int(bp.bits), bi_bit_err=_c(np.broadcast_to(bp.bit_err_lsb, (B, N, bp.bit_err_lsb.shape[-1]))),
            bi_bow=_bn(bp.inl_bow_lsb, B, N), bi_vmin=_bn(bp.v_min_v, B, N), bi_vmax=_bn(bp.v_max_v, B, N),
            bi_lat=_b(bp.latency_ticks, B, np.int64), bi_min_upd=int(bp.min_update_ticks),
            bi_noise_per=int(bp.noise_period_ticks), bi_noise=_bn(bp.noise_rms_codes, B, N),
            bi_slew=_bn(bp.slew_v_per_s, B, N), bi_settle=_bn(bp.settle_tau_s, B, N),
            bi_key=_c(sn._keys(nseeds, B, "dac_bias"), u64),
            he_law=_b(hp.law, B, np.int64), he_bits=int(hp.bits), he_pwm_bits=int(hp.pwm_bits),
            he_ripple=int(hp.pwm_ripple), he_dt_clk=int(hp.dt_clk),
            he_bit_err=_c(np.broadcast_to(hp.bit_err_lsb, (B, N, hp.bit_err_lsb.shape[-1]))),
            he_bow=_bn(hp.inl_bow_lsb, B, N), he_lo=int(hp.code_lo), he_hi=int(hp.code_hi),
            he_lat=_b(hp.latency_ticks, B, np.int64), he_min_upd=int(hp.min_update_ticks),
            he_slew=_bn(hp.slew_u_per_s, B, N), he_settle=_bn(hp.settle_tau_s, B, N), he_r0=_bn(hp.r_cold_ohm, B, N),
            he_alpha=_bn(hp.tcr_per_k, B, N), he_t0=_bn(hp.tcr_ref_k, B, N), he_vfs=_bn(hp.v_full_scale_v, B, N),
            he_ifs=_bn(hp.i_full_scale_a, B, N), he_pfs=_bn(hp.p_full_scale_w, B, N),
            he_vcomp=_bn(hp.compliance_v, B, N), he_pmax=_b(hp.supply_power_max_w, B),
            he_imax=_b(hp.supply_current_max_a, B),
            th_a=_c(th_a), th_c0=_c(th_c0), th_c1=_c(th_c1), th_lag_src=_c(sysm.lag_src, np.int64),
            th_lag_obs=_c(sysm.lag_obs, np.int64), th_lag_kind=_c(sysm.lag_kind, np.int64),
            th_opt_lags=_c(sysm.opt_lags, np.int64), th_obs_ptr=_c(sysm.obs_ptr, np.int64),
            th_src_ring=_c(sysm.src_ring, np.int64), th_ring_obs=_c(sysm.ring_obs, np.int64),
            th_heater_obs=_c(sysm.heater_obs, np.int64), th_heater_src=_c(sysm.heater_src, np.int64),
            th_wg_src=_c(sysm.wg_src, np.int64), th_junction_src=_c(_full_n(sysm.junction_src, N), np.int64),
            th_driver_src=_c(_full_n(sysm.driver_src, N), np.int64), th_agg_src=_c(sysm.aggressor_src, np.int64),
            th_pd_obs=_c(_full_n(sysm.pd_obs, N), np.int64), th_temp_obs=_c(sysm.temp_obs, np.int64),
            th_src_scale=_c(np.ones(sysm.n_src)),
            mo_pr=_c(mhw.pr), mo_pw=_c(mhw.pw), mo_lat=_b(mhw.latency_ticks, B, np.int64),
            mo_phase=_c(mhw.phase, np.int64), mo_pga=_c(mhw.pga_gains), mo_fl=_c(mhw.fl_coef),
            mo_afe=int(mhw.afe_type), mo_win=int(mhw.window_ticks), mo_period=int(mhw.period_ticks),
            mo_cmax=int(mhw.code_max), mo_shot=int(mhw.shot_on), mo_stuck=int(mhw.stuck_flagged),
            mo_slot=int(mhw.slot_base), mo_key_adc=_c(sn._keys(nseeds, B, mhw.stream), u64),
            mo_key_shot=_c(sn._keys(nseeds, B, "shot_noise"), u64),
            mo_key_dnl=_c(sn._keys(mhw.device_seeds if mhw.device_seeds is not None else np.zeros(B, np.int64), B,
                                   mhw.stream + "_dnl"), u64),
            ts_tp=_c(tsh.tp), ts_period=_c(tsh.period_ticks, np.int64), ts_lat=_c(tsh.latency_ticks, np.int64),
            ts_cmax=_c(tsh.code_max, np.int64), ts_fl=_c(tsh.fl_coef), ts_key=_c(sn._keys(nseeds, B, "adc_temp"), u64),
            rb_on=int(rb_on), rb_lsb_v=_b(rbh.lsb_v if rb_on else 1.0, B), rb_lsb_i=_b(rbh.lsb_i if rb_on else 1.0, B),
            rb_noise=_b(rbh.noise_codes if rb_on else 0.0, B), rb_period=int(rbh.period_ticks if rb_on else 1),
            rb_lat=_b(rbh.latency_ticks if rb_on else 0, B, np.int64), rb_cmax=int(rbh.code_max if rb_on else 1),
            rb_avg=int(rbh.average if rb_on else 0), rb_key=_c(sn._keys(nseeds, B, "adc_readback"), u64),
            sc_comp=float(arr.compression), sc_sig_i=_c(arr.sig_i64, np.int64), sc_sig_f=_c(arr.sig_f64),
            sc_def=_c(arr.elem_default), sc_comp_i=_c(arr.comp_i64, np.int64), sc_comp_f=_c(arr.comp_f64),
            sc_tab=_c(arr.tab_f64), sc_keys=_c(arr.keys, np.uint64),
            obs_map=_c(self.maps.obs_map, np.int64), act_map=_c(self.maps.act_map, np.int64),
            mp_f=_c(ms.mpf), mp_i=_c(ms.mpi, np.int64), assign_mode=int(self.metrics_spec.assignment_mode),
            wrong_window=float(self.metrics_spec.wrong_line_window_fwhm),
            oracle_period=int(max(1, self.metrics_spec.oracle_period_ticks)), oracle_on=int(self._oracle_on),
            rec_i=_c(rec.rec_i, np.int64), lock_ref_mode=int(self.metrics_spec.lock_reference_mode),
            au_max_pow=int(au_max_pow), au_max_det=float(pp.timing.max_detuning_step_m), au_pabs_tol=0.01,
            au_n_outer=int(n_outer_auto),
        )
        self.P = P
        # ---------------------------------------------------------------- state arrays
        S_ = int(tsh.S)
        mon_cap = fifo_capacity(int(mhw.latency_ticks.max()), mhw.period_ticks, pop)
        ts_cap = max([fifo_capacity(int(tsh.latency_ticks[:, i].max()), int(tsh.period_ticks[i]), pop)
                      for i in range(S_)] or [1])
        rb_cap = fifo_capacity(int(rbh.latency_ticks.max()), rbh.period_ticks, pop) if rb_on else 1
        dp_cap = fifo_capacity(int(dph.error_latency_ticks.max()), max(dph.error_period_ticks, 1), pop)
        J = mhw.fl_coef.shape[1]
        JT = tsh.fl_coef.shape[2]
        M = int(spec.samples_per_step)
        nk = arr.cache0.shape[1]
        V = arr.n_out
        A = max(self.maps.act_size, 1)
        # commands are pushed once per edge; with tick jitter J (07 P2-7) two edges can be period - J apart
        pcap = pipe_capacity(self.controller_latency_ticks, max(period - int(spec.tick_jitter_ticks), 1))
        R = max(rec.rec_i.shape[0], 1)
        nlm = L + int(lhw.lines.n_lasers)
        self._caps = {"monitor": mon_cap, "temp": ts_cap, "readback": rb_cap, "errors": dp_cap, "pipeline": pcap}
        S = K.KState(
            rf=np.zeros((B, K.NRF, N)), ri=np.zeros((B, K.NRI, N), np.int64), r8=np.zeros((B, K.NR8, N), np.int8),
            rb=np.zeros((B, K.NRB, N), np.bool_), wi=np.zeros((B, K.NWI), np.int64), wf=np.zeros((B, K.NWF)),
            lf=np.zeros((B, K.NLF, L)),
            las_cache_j=np.full((B, 1), -1, np.int64), las_wlo=np.zeros((B, 2, nlm)), las_whi=np.zeros((B, 2, nlm)),
            dp_fv=np.zeros((B, N, dp_cap), np.int64), dp_fr=np.zeros((B, N, dp_cap), np.int64),
            dp_fm=np.zeros((B, N, 2), np.int64),
            bi_fifo=np.zeros((B, N, bp.fifo_depth), np.int64), he_fifo=np.zeros((B, N, hp.fifo_depth), np.int64),
            th_x=sysm.zero_state(B), th_t_obs=np.zeros((B, sysm.n_obs)), th_g=np.zeros((B, sysm.n_obs)),
            th_u0=np.zeros((B, sysm.n_src)), th_u1=np.zeros((B, sysm.n_src)),
            sh_args_f=args_f0.copy(), jx=np.zeros((B, 3, N, 2)),
            mo_sf=np.zeros((B, N, sn.NSF)), mo_si=np.zeros((B, N, sn.NSI), np.int64), mo_fx=np.zeros((B, N, J)),
            mo_fc=np.zeros((B, N, J)), mo_fv=np.zeros((B, N, mon_cap), np.int64),
            mo_fr=np.zeros((B, N, mon_cap), np.int64), mo_fm=np.zeros((B, N, 2), np.int64),
            mo_samples=np.zeros((B, N, M), np.int64), mo_ticks=np.zeros((B, N, M), np.int64),
            ts_st=np.zeros((B, S_, 2)), ts_fx=np.zeros((B, S_, JT)), ts_fc=np.zeros((B, S_, JT)),
            ts_primed=np.zeros((B, S_), np.int64), ts_fv=np.zeros((B, S_, ts_cap), np.int64),
            ts_fr=np.zeros((B, S_, ts_cap), np.int64), ts_fm=np.zeros((B, S_, 2), np.int64),
            ts_latest=np.zeros((B, S_), np.int64), ts_new=np.zeros((B, S_), np.bool_),
            ts_sampled=np.zeros((B, S_), np.int8), ts_in=np.zeros((B, S_)), ts_in_prev=np.zeros((B, S_)),
            rb_st=np.zeros((B, N, 3)), rb_fv=np.zeros((B, N, rb_cap), np.int64),
            rb_fr=np.zeros((B, N, rb_cap), np.int64), rb_fm=np.zeros((B, N, 2), np.int64),
            sc_cache=arr.new_cache(), sc_v0=arr.new_out(), sc_v1=arr.new_out(), sc_va=arr.new_out(),
            sc_vp=arr.new_out(),
            c_pi=np.zeros((B, 1), np.int64), c_pf=np.zeros((B, 1)), c_si=np.zeros((B, 1), np.int64),
            c_sf=np.zeros((B, 1)), c_si0=np.zeros((B, 1), np.int64), c_sf0=np.zeros((B, 1)),
            act=np.zeros((B, A), np.int64), act_tmp=np.zeros((B, A), np.int64),
            obs=np.zeros((B, self.maps.obs_size), np.int64), pipe_vals=np.zeros((B, pcap, A), np.int64),
            pipe_ticks=np.zeros((B, pcap), np.int64), pipe_meta=np.zeros((B, 2), np.int64),
            sat=np.zeros((B, A), np.int64), tune=np.zeros((B, L), np.int64),
            fobs=np.zeros((B, fast_obs_size(N)), np.int64), fact=np.zeros((B, fast_act_size(N)), np.int64),
            m_af=ms.af, m_ai=ms.ai, m_hist=ms.hist, m_runhist=ms.runhist, m_ref=ms.ref,
            or_phi=np.zeros((B, oracle_pts)), or_k2t=np.zeros((B, oracle_pts)), or_k2d=np.zeros((B, oracle_pts)),
            or_a=np.zeros((B, oracle_pts)), or_trt=np.zeros((B, oracle_pts)), or_psi=np.zeros((B, oracle_pts)),
            or_res0=np.zeros((B, op.N_RES, oracle_pts)), or_res1=np.zeros((B, op.N_RES, oracle_pts)),
            or_cw=np.zeros((B, op.N_CW, oracle_pts), np.complex128), or_fw=np.zeros((B, op.N_FW, oracle_pts)),
            tv=np.zeros((B, N_TR, N)), lt=np.zeros((B, 3, L)),
            rec_acc=rec.acc, rec_cnt=rec.cnt, rec_idx=rec.idx, trace=rec.trace,
        )
        self.S = S
        self._run_block = None

    # ------------------------------------------------------------------------------------------------ reset
    def reset(self, noise_seed: int | None = None, *, k0: int = 0, protocol: str | None = None, heater_code=None,
              bias_code=None, settle: bool = False) -> None:
        """Reset every world to its defined start (07 P0-1 item 6) at tick k0.

        protocol: 'laser_ramp' (lines ramp from 0 over pp.timing.reset_ramp_ticks, cold branch), 'branch_low'
        (lasers on at once, the root met from the cold side), 'branch_high' (the hot, self-locked branch); default
        pp.timing.reset_protocol. heater_code: initial heater registers (default code_min; a warm start passes
        codes). bias_code: initial bias registers (default the scenario's bias at k0, or code 0 when the controller
        owns the bias). noise_seed re-keys the noise streams only (07 P1-13); device draws never change.
        settle: start from the thermal steady state of these codes and the inputs at k0 instead of cold lags
        (:meth:`settle`; needs protocol branch_low or branch_high). ``self.settle_info`` reports it (None without).
        """
        self._reset(noise_seed, k0=k0, protocol=protocol, heater_code=heater_code, bias_code=bias_code)
        self.settle_info = None
        if settle:
            self.settle_info = self.settle(heater_code=heater_code, bias_code=bias_code)

    def settle(self, *, heater_code=None, bias_code=None, max_iter: int = 200, tol_k: float = 2e-3) -> dict:
        """Settled warm start (07 P0-1 item 6; controllers_a request 3, controllers_c request 3b): replace the cold
        thermal lags of the state just reset by the steady state of the held inputs at the reset tick.

        Method: the kernel itself runs open loop with the DC coefficients of every lag (a = 0, c0 = 0, c1 = r_m:
        the exact FOH step of mrm.plant.thermal.foh_coefficients for h -> inf), one tick at a time, until the ring
        temperatures change by less than ``tol_k`` (K) between iterations. Each DC tick is the plant's own implicit
        self-heating solve (mrm.plant.selfheat; the first iterate starts from the reset branch, +inf for
        branch_high and -inf for branch_low, later ones continue from the previous root), heater TCR split, bias,
        NRZ levels, data-activity heat and scenario inputs at the reset tick, so the result is the engine's own
        fixed point, not a separate model. The cross-ring optical heat and the heater TCR lag one iteration (a
        contraction by the small cross gains). Then the plant is reset again (sensors, metrics, records, edges,
        registers, noise) and only the thermal state is kept: lag states, observer temperatures, held source powers,
        ring / heater / PD temperatures, absorbed and heater powers; the self-heating root continues from the
        settled ring temperature. Pass the same heater_code / bias_code as the reset.

        Convergence: the noise draws of every tick (bias DAC noise, laser RIN, binomial ones density) move the DC
        fixed point by ~0.5 mK in g26 (~0.04 pm), so ``tol_k`` is 2 mK (0.15 pm at 70 pm/K), not a solver
        tolerance. From branch_high the first iterates walk down from the hottest root (~10 K per iteration under
        the solver's iteration limit; those solves count as not converged); g26 nominal settles in ~20 iterations.

        Returns {'iterations', 'max_delta_k', 'converged', 'selfheat_not_converged_last' (B,): self-heating solves
        that did not converge in the last iteration (0 when the fixed point is a regular root)}.
        """
        if self.tick != self.k_reset:
            raise RuntimeError("settle() must follow reset() directly")
        P0 = self.P
        if int(P0.reset_protocol) == 0:
            raise ValueError("settle needs the reset protocol branch_low or branch_high (lasers on at the start); "
                             "laser_ramp starts dark")
        k0 = int(self.tick)
        shape = P0.th_a.shape                                              # (B|1, n_pow, M)
        res = np.asarray(self.sys.res_k_per_w, np.float64)
        c1 = np.ascontiguousarray(np.broadcast_to(res[:, None, :], shape), dtype=np.float64).copy()
        self.P = P0._replace(th_a=np.zeros(shape), th_c0=np.zeros(shape), th_c1=c1,
                             policy=type(P0.policy)(K.POLICY_FIXED))
        t_prev = self.S.rf[:, K.RF_T_RING, :].copy()
        delta = math.inf
        it = 0
        nf_last = np.zeros(self.B, np.int64)
        try:
            for it in range(1, int(max_iter) + 1):
                nf0 = self.S.wi[:, K.WI_N_FAIL].copy()
                self.run(1, mode=K.MODE_OPEN)
                nf_last = self.S.wi[:, K.WI_N_FAIL] - nf0
                t_now = self.S.rf[:, K.RF_T_RING, :]
                delta = float(np.max(np.abs(t_now - t_prev)))
                t_prev = t_now.copy()
                if it >= 3 and delta < tol_k:
                    break
        finally:
            self.P = P0
        S = self.S
        keep ={name: getattr(S, name).copy() for name in ("th_x", "th_t_obs", "th_g", "th_u0", "th_u1")}
        rows = [K.RF_P_PREV, K.RF_T_RING, K.RF_T_HEATER, K.RF_T_PD, K.RF_T_NEW, K.RF_P_NEW, K.RF_HE_P, K.RF_P_ABS]
        keep_rf = S.rf[:, rows, :].copy()
        self._reset(None, k0=k0, protocol=None, heater_code=heater_code, bias_code=bias_code)
        S = self.S
        for name, val in keep.items():
            getattr(S, name)[...] = val
        S.rf[:, rows, :] = keep_rf
        S.rf[:, K.RF_T_BRANCH, :] = S.rf[:, K.RF_T_RING, :]
        return {"iterations": int(it), "max_delta_k": delta, "converged": bool(delta < tol_k),
                "selfheat_not_converged_last": nf_last.copy()}

    def _reset(self, noise_seed: int | None = None, *, k0: int = 0, protocol: str | None = None, heater_code=None,
               bias_code=None) -> None:
        """The body of :meth:`reset` (no settle)."""
        if noise_seed is not None and int(noise_seed) != self.noise_seed:
            self.noise_seed = int(noise_seed)
            self._rekey_noise()
        P, S = self.P, self.S
        B, N, L = self.B, self.N, self.L
        if protocol is not None:
            codes = {"laser_ramp": 0, "branch_low": 1, "branch_high": 2}
            if protocol not in codes:
                raise ValueError(f"reset protocol must be one of {tuple(codes)}")
            self.P = P = P._replace(reset_protocol=codes[protocol])
            self._run_block = None
        k0 = int(k0)
        self.tick = k0
        self.k_reset = k0
        for a in (S.rf, S.ri, S.r8, S.rb, S.wi, S.wf, S.lf, S.dp_fv, S.dp_fr, S.dp_fm, S.th_x, S.th_g, S.th_u0,
                  S.th_u1, S.jx, S.mo_sf, S.mo_si, S.mo_fx, S.mo_fc, S.mo_fv, S.mo_fr, S.mo_fm, S.mo_samples,
                  S.mo_ticks, S.ts_st, S.ts_fx, S.ts_fc, S.ts_primed, S.ts_fv, S.ts_fr, S.ts_fm, S.ts_latest,
                  S.ts_new, S.ts_sampled, S.rb_st, S.rb_fv, S.rb_fr, S.rb_fm, S.pipe_vals, S.pipe_ticks,
                  S.pipe_meta, S.sat, S.tune, S.fobs, S.fact, S.tv, S.lt, S.las_wlo, S.las_whi, S.obs):
            a[...] = 0
        S.las_cache_j[...] = -1
        S.sh_args_f[...] = self._args_f0
        S.sc_cache[...] = self.scen_arrays.new_cache()
        # scenario values at k0 (signals without components keep their defaults)
        arr = self.scen_arrays
        out = arr.new_out()
        sg.scen_eval(np.int64(k0), *arr.kernel_args(S.sc_cache), out)
        S.sc_v0[...] = out
        S.sc_v1[...] = out
        S.sc_vp[...] = out
        lay = arr.layout
        t_amb = out[:, lay["thermal.t_base_k"][0]] + out[:, lay["thermal.xpu_rise_k"][0]]
        S.wf[:, K.WF_T_BASE0] = t_amb
        S.wf[:, K.WF_T_BASE1] = t_amb
        S.th_t_obs[...] = t_amb[:, None]
        rf, ri = S.rf, S.ri
        for idx in (K.RF_T_RING, K.RF_T_HEATER, K.RF_T_BRANCH, K.RF_T_NEW, K.RF_T_PD):
            rf[:, idx, :] = t_amb[:, None]
        proto = int(P.reset_protocol)
        if proto in (0, 1):
            rf[:, K.RF_T_BRANCH, :] = -np.inf                  # the cold branch (07 P0-1 item 6)
        else:
            rf[:, K.RF_T_BRANCH, :] = np.inf                   # the hot branch
        S.ts_in[...] = t_amb[:, None] if S.ts_in.shape[1] else 0.0
        # heater DAC: registers, FIFO and drive at the initial code (default code_min, 07 P0-1 item 6)
        hp = self.heater_params
        code0 = np.full((B, N), hp.code_lo, np.int64) if heater_code is None else \
            np.clip(np.broadcast_to(np.asarray(heater_code, np.int64), (B, N)), hp.code_lo, hp.code_hi)
        ri[:, K.RI_HE_REG] = code0
        ri[:, K.RI_HE_PEND] = code0
        ri[:, K.RI_HE_LAST] = _NEVER
        S.he_fifo[...] = code0[:, :, None]
        u0 = np.zeros((B, N))
        rip = np.zeros((B, N))
        for b in range(B):
            el.heater_target_row(np.ascontiguousarray(code0[b]), int(P.he_law[b]), P.he_bits, P.he_pwm_bits,
                                 P.he_bit_err[b], P.he_bow[b], 0, 0, 1, u0[b], rip[b])
        rf[:, K.RF_HE_YS] = u0
        rf[:, K.RF_HE_YO] = u0
        rf[:, K.RF_HE_UT] = u0
        # bias DAC: the scenario's bias at k0 (or code 0 / the given code under controller ownership)
        bp = self.bias_params
        if bias_code is not None:
            bcode = np.clip(np.broadcast_to(np.asarray(bias_code, np.int64), (B, N)), 0, bp.code_max)
        elif P.bias_ctrl == 0:
            vs = out[:, lay["tx.bias_v"][0]:lay["tx.bias_v"][0] + N]
            vs = np.where(np.isfinite(vs), vs, P.bi_vmin)
            bcode = np.clip(np.rint((vs - P.bi_vmin) / ((P.bi_vmax - P.bi_vmin) / bp.code_max)), 0,
                            bp.code_max).astype(np.int64)
        else:
            bcode = np.zeros((B, N), np.int64)
        ri[:, K.RI_BI_REG] = bcode
        ri[:, K.RI_BI_PEND] = bcode
        ri[:, K.RI_BI_LAST] = _NEVER
        ri[:, K.RI_BI_CODE] = bcode
        S.bi_fifo[...] = bcode[:, :, None]
        v0 = np.zeros((B, N))
        for b in range(B):
            el.bias_target_row(np.ascontiguousarray(bcode[b]), P.bi_bits, P.bi_bit_err[b], P.bi_bow[b], P.bi_vmin[b],
                               P.bi_vmax[b], v0[b])
        rf[:, K.RF_BI_YS] = v0
        rf[:, K.RF_BI_YO] = v0
        rf[:, K.RF_BI_VT] = v0
        rf[:, K.RF_BI_VAVG] = v0
        # data path: PRBS start phases from the pattern stream (mrm.plant.datapath.DatapathBank.reset)
        if P.dp_period > 0:
            for b in range(B):
                for n in range(N):
                    u = _rng.uniform(P.dp_key_pat[b], n, 0, 1)
                    ri[b, K.RI_DP_BITPOS, n] = int(u * P.dp_period)
        # sensors: monitor flags (valid until a fault), fast-kernel cursor
        S.rb[:, K.RB_MO_VALID, :] = True
        ri[:, K.RI_FAST_SEEN, :] = -1
        ri[:, K.RI_ASSIGN, :] = -1
        ri[:, K.RI_DECL, :] = P.ring_to_line[None, :]
        # controller pipeline and held command: act from the latched registers (mrm.control.api ABI rule)
        S.act[...] = 0
        if self.maps.act_size:
            S.act[:, :self.maps.act_size] = self.maps.act_init
        # the registers of the non-DAC actuators start at their channel init values (they are the held command
        # until the controller writes them)
        for i in range(self.maps.act_map.shape[0]):
            dst, o, sz = (int(x) for x in self.maps.act_map[i, :3])
            if dst == 0:
                S.act[:, o:o + sz] = code0
            elif dst == 1:
                S.act[:, o:o + sz] = bcode
            elif dst == 2:
                ri[:, K.RI_ASSIGN] = S.act[:, o:o + sz]
            elif dst == 3:
                ri[:, K.RI_BS_THR] = S.act[:, o:o + sz]
            elif dst == 4:
                ri[:, K.RI_PGA] = S.act[:, o:o + sz]
            elif dst == 5:
                S.tune[:, :sz] = S.act[:, o:o + sz]
        # edges: the first controller edge at or after k0
        per, ph0 = int(P.period), int(P.phase)
        j0 = 0 if k0 <= ph0 else -(-(k0 - ph0) // per)
        S.wi[:, K.WI_EDGE_IDX] = j0
        for b in range(B):
            S.wi[b, K.WI_EDGE_NEXT] = ph0 + j0 * per + (int(_edge_jit(P, b, j0)) if P.jitter > 0 else 0)
        S.wi[:, K.WI_K_RESET] = k0
        S.wi[:, K.WI_ORACLE_NEXT] = k0
        S.wi[:, K.WI_PENDING_EDGE] = -1
        # a bound compiled controller restarts from its post-init state (kept by bind_compiled; it is also what
        # fault.controller_reset restores), so an episode after reset() repeats the first one exactly
        if self._ctrl is not None:
            S.c_si[...] = S.c_si0
            S.c_sf[...] = S.c_sf0
        self.metrics_state.reset(k0, code0)
        self.records_state.reset()
        self.validator.reset()
        self._pending = False

    def _rekey_noise(self) -> None:
        B = self.B
        nseeds = _rng.batch_seeds(self.noise_seed, B)
        self._nseeds = nseeds
        u64 = np.uint64
        mhw = self.monitor_hw
        self.P = self.P._replace(
            key_sched=_c([_rng.stream_key(int(s), int(_rng.Stream.SCHEDULE)) for s in nseeds], u64),
            las_key_rin=_c([_rng.stream_key(int(s), int(_rng.Stream.LASER_RIN)) for s in nseeds], u64),
            dp_key_pat=_c(sn._keys(nseeds, B, "pattern"), u64), dp_key_dp=_c(sn._keys(nseeds, B, "datapath"), u64),
            bi_key=_c(sn._keys(nseeds, B, "dac_bias"), u64), mo_key_adc=_c(sn._keys(nseeds, B, mhw.stream), u64),
            mo_key_shot=_c(sn._keys(nseeds, B, "shot_noise"), u64), ts_key=_c(sn._keys(nseeds, B, "adc_temp"), u64),
            rb_key=_c(sn._keys(nseeds, B, "adc_readback"), u64))
        self._run_block = None

    # ------------------------------------------------------------------------------------------------ running
    def bind_compiled(self, ctrl) -> None:
        """Run a reset CompiledController (mrm.control.api) inside the kernel: its parameter and state rows are the
        kernel's (07 P0-9), and its held command array becomes the kernel's act vector, pre-filled from the latched
        DAC registers (the ABI rule of mrm.control.api)."""
        if ctrl.state is None:
            raise RuntimeError("reset the compiled controller with this plant's spec first")
        if ctrl.act.shape != (self.B, self.maps.act_size):
            raise ValueError("controller act layout does not match the plant's spec")

        def own(a, dtype):
            ok = a.dtype == dtype and a.flags.c_contiguous and a.flags.writeable and a.ndim == 2
            return a if ok else np.array(a, dtype=dtype, order="C", copy=True).reshape(self.B, -1)
        ctrl.state = tuple(own(a, d) for a, d in zip(ctrl.state, (np.int64, np.float64, np.int64, np.float64)))
        p_i, p_f, s_i, s_f = ctrl.state
        S = self.S._replace(c_pi=p_i, c_pf=p_f, c_si=s_i, c_sf=s_f)
        S = S._replace(c_si0=S.c_si.copy(), c_sf0=S.c_sf.copy())
        if self.maps.act_size:
            ctrl.act[...] = S.act[:, :self.maps.act_size]
            if ctrl.act.dtype == np.int64 and ctrl.act.flags.c_contiguous:
                S = S._replace(act=ctrl.act, act_tmp=np.zeros_like(ctrl.act))
        ctrl._act_synced = True
        self.S = S
        self._ctrl = ctrl
        self._run_block = None

    def _fn(self):
        if self._run_block is None:
            _check_kernel_arrays(self.P, "P")
            _check_kernel_arrays(self.S, "S")
            self._run_block = K.get_run_block(self.P, self.S)
        return self._run_block

    def run(self, n_ticks: int, *, mode: int = K.MODE_OPEN, step_fn=None, fast_fn=None, resume: bool = False,
            n_threads: int | None = None) -> int:
        """Advance up to n_ticks ticks with the kernel (mode MODE_OPEN | MODE_COMPILED | MODE_PYTHON). Returns the
        ticks advanced (fewer than n_ticks only on the Python path, which stops at a controller edge)."""
        n = int(n_ticks)
        if n <= 0:
            return 0
        done = np.zeros(self.B, np.int64)
        fn = self._fn()
        fn(K.null_step if step_fn is None else step_fn, K.null_step if fast_fn is None else fast_fn, self.P, self.S,
           np.int64(self.tick), np.int64(n), np.int64(mode), np.int64(1 if resume else 0), done,
           n_threads=self.n_threads if n_threads is None else n_threads)
        if np.any(done != done[0]):
            raise RuntimeError(f"worlds advanced by different tick counts {done}: the Python path needs edges shared "
                               "by every world (no tick jitter)")
        self.tick += int(done[0])
        return int(done[0])

    def advance(self, inputs: Mapping[str, Any] | None = None, n_ticks: int = 1) -> None:
        """Apply actuator codes (channel -> (B, ...) array, validated like a controller's actions: clipped to the
        channel range and counted) as a command issued at the current tick, then advance n_ticks ticks open loop
        (07 P0-6: the command latches at tick + controller latency, then the DAC FIFO, slew and settle). Commands
        may be issued more often than the controller period: the pipeline grows so that none is dropped."""
        if inputs:
            acts = self.validator.validate(dict(inputs))
            self.spec.layout.pack_act(acts, self.S.act[:, :self.maps.act_size])
            latch = np.int64(self.tick + self.controller_latency_ticks)
            for b in range(self.B):
                S = self.S
                if not pipe_push_row(S.pipe_vals[b], S.pipe_ticks[b], S.pipe_meta[b], S.act[b], latch):
                    self._grow_pipeline()
                    S = self.S
                    if not pipe_push_row(S.pipe_vals[b], S.pipe_ticks[b], S.pipe_meta[b], S.act[b], latch):
                        raise RuntimeError("controller pipeline: push failed after growing it")
        self.run(int(n_ticks), mode=K.MODE_OPEN)

    def _grow_pipeline(self) -> None:
        """Double the controller pipeline's capacity, keeping every queued command in order (same dtype and ndim, so
        the compiled kernel is reused)."""
        S = self.S
        B, cap, A = S.pipe_vals.shape
        vals = np.zeros((B, 2 * cap, A), np.int64)
        ticks = np.zeros((B, 2 * cap), np.int64)
        for b in range(B):
            h, c = int(S.pipe_meta[b, 0]), int(S.pipe_meta[b, 1])
            idx = (h + np.arange(c)) % cap
            vals[b, :c] = S.pipe_vals[b, idx]
            ticks[b, :c] = S.pipe_ticks[b, idx]
            S.pipe_meta[b, 0] = 0
        self.S = S._replace(pipe_vals=vals, pipe_ticks=ticks)
        self._caps["pipeline"] = 2 * cap

    # ------------------------------------------------------------------------------------------------ readout
    def sample_sensors(self) -> tuple[dict, dict]:
        """(sensors, new) channel dicts released at the last controller edge (or at the current tick when it is an
        edge), in the dtypes of the ControllerSpec (07 P0-7)."""
        _, _, sensors, new = self.spec.layout.unpack_obs(self.S.obs)
        return sensors, new

    def observation_tick(self) -> int:
        return int(self.S.obs[0, 0])

    def truth(self) -> Truth:
        """Truth at the current tick (state at the end of the last interval)."""
        S, P = self.S, self.P
        fields = {name: S.tv[:, i, :].copy() for i, name in enumerate(TRUTH_FIELDS)}
        out_rl = np.zeros((self.B, op.N_RL, self.N, self.L))
        out_r = np.zeros((self.B, op.N_R, self.N))
        out_l = np.zeros((self.B, op.N_L, self.L))
        for b in range(self.B):
            v = op.selfheat_views(S.sh_args_f[b], P.sh_args_i[b])
            out_rl[b], out_r[b], out_l[b] = v[22], v[23], v[24]
        return Truth(tick=int(self.tick), t_s=self.tick * self.dt, fields=fields, p_chip_w=S.lt[:, 0].copy(),
                     p_out_w=S.lt[:, 1].copy(), lam_m=S.lt[:, 2].copy(), out_rl=out_rl, out_r=out_r, out_l=out_l,
                     t_obs=S.th_t_obs.copy(), obs_names=tuple(self.sys.obs_names), x=S.th_x.copy(),
                     heater_applied=S.ri[:, K.RI_HE_REG].copy(), conditions=self._conditions())

    def _conditions(self) -> list:
        """Per-world mrm.control.models.Conditions of the last interval (for Truth-fed controllers)."""
        from mrm.control.models import Conditions
        S, B = self.S, self.B
        lay = self.scen_arrays.layout
        o_agg, w_agg = lay.get("thermal.aggressor_w", (0, 0))
        out = []
        for b in range(B):
            out.append(Conditions(lam_m=S.lf[b, K.LF_LAM].copy(), p_chip_w=S.lf[b, K.LF_PCHIP].copy(),
                                  t_base_k=float(S.wf[b, K.WF_T_BASE1]), v_lo=S.rf[b, K.RF_V_LO].copy(),
                                  v_hi=S.rf[b, K.RF_V_HI].copy(), rho=S.rf[b, K.RF_RHO].copy(),
                                  transition_density=S.rf[b, K.RF_TD].copy(),
                                  aggressor_w=S.sc_v1[b, o_agg:o_agg + w_agg].copy() if w_agg else None,
                                  data_enable=S.rf[b, K.RF_EN].copy()))
        return out

    def metrics(self) -> dict:
        """The 07 P0-8 metrics of the accumulators since the last reset (mrm.sim.metrics.summarize)."""
        return summarize(self.metrics_state)

    def traces(self) -> dict:
        """Recorded traces {key: {'t_s', 'value' (S, B, N)}} (07 P0-8 record specs)."""
        return self.records_state.traces(self.dt, self.k_reset)

    def diagnostics(self) -> dict:
        """Counters of the run: self-heating solves that did not converge (fold crossings or bounds), samples lost
        to FIFO or block limits, pipeline overflows, clipped commands, substeps, fast-kernel calls."""
        wi = self.S.wi
        return {"selfheat_not_converged": wi[:, K.WI_N_FAIL].copy(),
                "selfheat_evaluations": wi[:, K.WI_SH_ITER].copy(),
                "monitor_block_lost": wi[:, K.WI_MON_LOST].copy(),
                "monitor_fifo_lost": wi[:, K.WI_MON_FIFO_LOST].copy(),
                "temp_fifo_lost": wi[:, K.WI_TEMP_LOST].copy(), "readback_fifo_lost": wi[:, K.WI_RB_LOST].copy(),
                "pipeline_overflow": wi[:, K.WI_PIPE_OVER].copy(), "commands_clipped": wi[:, K.WI_SAT_TOTAL].copy(),
                "substeps": wi[:, K.WI_N_SUBSTEPS].copy(), "ticks": wi[:, K.WI_N_TICKS].copy(),
                "largest_substep_ticks": wi[:, K.WI_SUB_H_MAX].copy(),
                "fast_kernel_calls": wi[:, K.WI_FAST_CALLS].copy(),
                "saturation_counts": self.S.sat[:, :self.maps.act_size].copy(), "fifo_capacities": dict(self._caps)}


def _check_kernel_arrays(tup, what: str) -> None:
    """Every array of the kernel arguments must be writeable and C-contiguous: a read-only or strided array is a
    different numba type, which would compile the whole plant again (minutes) instead of reusing the cached kernel."""
    for name, v in zip(tup._fields, tup):
        if isinstance(v, np.ndarray) and not (v.flags.c_contiguous and v.flags.writeable):
            raise TypeError(f"kernel argument {what}.{name} must be a writeable C-contiguous array")


def _ensure_act(a):
    return np.ascontiguousarray(a, dtype=np.int64)


def _full_n(idx, N):
    """A per-ring index map of length N (-1 where absent)."""
    a = np.asarray(idx, np.int64).reshape(-1)
    if a.size == N:
        return a
    out = np.full(N, -1, np.int64)
    out[:a.size] = a[:N]
    return out


def _edge_jit(P, b, j):
    from mrm.core.timing import edge_jitter
    return edge_jitter(P.key_sched[b], np.int64(j), np.int64(P.jitter))
