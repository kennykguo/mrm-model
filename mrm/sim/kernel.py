"""Compiled closed-loop kernel: plant ticks, sensor release, compiled controller, metrics and records (07 P0-9).

Owner: engine. Contract: docs/research/07_design_critique.md P0-6 (integer ticks, one owner per delay, the event
order), P0-7 (named channels, compiled controllers with the flat ABI, the optional fast_kernel once per ADC sample),
P0-8 (streaming metrics, record specs), P0-9 (``@njit(parallel=True, cache=True) run_block`` with prange over B only
and many controller ticks per call; every parameter an array argument; no frozen globals), P0-10 (substep_policy
fixed | auto), P0-1 (implicit self-heating), docs/handoff/ENGINE_BRIEF.md sections 3-4 (the leaf calls).

Entry point
-----------
    fn = get_run_block(P, S)                              # compiled once per argument TYPES (disk-cached)
    done = np.zeros(B, np.int64)
    fn(step_fn, fast_fn, P, S, k0, n_ticks, mode, resume, done)   # parallel over B on a thread pool

P (:class:`KParams`) and S (:class:`KState`) are namedtuples of arrays built by :class:`mrm.plant.plant.Plant` from
PlantParams. ``step_fn`` / ``fast_fn`` are numba functions with the compiled-controller ABI of mrm.control.api
(``step(p_i64, p_f64, s_i64, s_f64, obs_i64, act_i64)``); they are typed as a first-class FunctionType, so one
compiled kernel serves every controller and is cached on disk (a Dispatcher argument would recompile the whole plant
per controller and process). :func:`null_step` stands for "no controller" / "no fast kernel".

mode: 0 open loop (sensors are still released at the controller edges so their FIFOs drain; nothing is commanded),
1 compiled controller (step at every edge inside the kernel), 2 Python controller (the kernel stops AT the next edge
after packing obs; the caller runs the Python step, writes S.act and calls again with resume = 1, which pushes the
command into the controller pipeline and continues). ``done[b]`` receives the ticks advanced by world b.

Event order (07 P0-6; mrm/core/timing.py; ENGINE_BRIEF section 3). Tick k is the instant at the end of [k-1, k).
At instant k: (a) the fast_kernel runs once per monitor sample released at k (before step); (b) at a controller
edge: release every sensor bank (monitor, temperature, readback, error counter; the bit-statistics window; the
ones/bit sums since the last edge), pack obs, run step, clip every actuator to its channel range, push the command
to latch at k + Controller.latency_ticks. Then plant interval [k, k+1): commands latched at k reach the DAC
registers; scenario at k and k+1; laser lines (with the laser_ramp reset factor); ones density; bias DAC (+ fast
offset, latency FIFO, noise, slew, settle, scenario offset); NRZ levels; optics inputs at the previous ring
temperature (mrm.plant.phase); self-heating bounds; data-activity heat; heater TCR split and heater DAC path (+ fast
offset, dac_glitch); implicit thermal + self-heating step (mrm.plant.selfheat, the optics walk inside); sensors of
the interval (monitor, error counter, temperature sensors, readback); Truth, metrics and records of the state at
k + 1.

Fidelity switches are plain integers in P (07 P0-9: no module-level config, no closures over config values).
"""
from __future__ import annotations

import math
from collections import namedtuple

import numba as nb
import numpy as np
from numba import njit, types

from mrm.core import rng as _rng
from mrm.core.timing import edge_jitter, fifo_pop_ready
from mrm.plant import electrical as _el
from mrm.plant import optics as _op
from mrm.plant import phase as _ph
from mrm.plant import selfheat as _sh
from mrm.plant import thermal as _th
from mrm.plant.selfheat import STATUS_BOUND, STATUS_CONVERGED, STATUS_MAX_ITER
from mrm.plant.thermal import LAG_EXOGENOUS, LAG_OPT_CROSS_RING, LAG_OPT_SELF
from mrm.plant.datapath import (bitstat_window_row, ber_from_oma, errors_release_row, errors_tick_ber_row,
                                pattern_tick_row)
from mrm.plant.io import (apply_act_row, clip_act_row, pack_obs_row, pipe_pop_row, pipe_push_row)
from mrm.plant.laser import laser_tick_row
from mrm.plant.sensing import (FAULT_DAC_GLITCH, SAMPLE_DROPPED, _CODE_MASK, _FLAG_SHIFT, PR_BG, PR_RESP,
                               PW_DARK_EA, PW_DARK_I0, PW_DARK_T0, PW_GAIN_SCALE, dark_current_a, fault_mask_row,
                               monitor_level_codes_row, monitor_release_row, monitor_tick_row, readback_release_row,
                               readback_tick_row, temp_release_row, temp_tick_row)
from mrm.scenarios import generators as _sg
from mrm.sim.metrics import (AI_WIN_TICKS, MPI_BER_WIN, TR, N_TR, metrics_tick_row, record_tick_row)
from mrm.units import C0

__all__ = ["KParams", "KState", "STEP_SIG", "STEP_FT", "null_step", "get_run_block", "release_row", "truth_row",
           "eng_pabs", "eng_commit", "eng_precompute", "extend_selfheat_rows", "swing_capacitance_row",
           "cumulative_capacitance",
           "MODE_OPEN", "MODE_COMPILED", "MODE_PYTHON", "POLICY_FIXED", "POLICY_AUTO",
           "RF", "RI", "R8", "RB", "WI", "WF", "LF", "NRF", "NRI", "NR8", "NRB", "NWI", "NWF", "NLF"]

MODE_OPEN = 0
MODE_COMPILED = 1
MODE_PYTHON = 2
POLICY_FIXED = 0
POLICY_AUTO = 1
_TWO_PI = 2.0 * math.pi
_INF = math.inf

# ---------------------------------------------------------------------------------------------- state blocks
# per-ring float block S.rf (B, NRF, N)
_RF_NAMES = ("bi_ys", "bi_yo", "bi_hold", "bi_vt", "bi_vavg", "bi_vend",
             "he_ys", "he_yo", "he_ut", "he_rip", "he_uavg", "he_p", "he_v", "he_i", "he_r", "he_tpre", "he_g",
             "p_prev", "t_branch", "t_ring", "t_heater", "p_new", "t_new", "pmax", "d2p", "dpdt",
             "v_lo", "v_hi", "cj", "rs", "pj", "pd",
             "rho", "dens", "td", "swing", "en", "boff", "bias_v", "glitch",
             "p_pd", "p_bg", "t_pd", "oma", "dp_lam", "mu1", "mu0", "sig1", "sig0",
             "oracle", "oracle_unr", "det", "ber", "bits", "p_abs", "swing_v",
             "ppd_prev", "tpd_prev", "lamres_prev", "pabs_prev", "t_sens_prev", "pbg_prev", "bi_vt0", "i_dark")
RF = {n: i for i, n in enumerate(_RF_NAMES)}
NRF = len(_RF_NAMES)
# per-ring int64 block S.ri (B, NRI, N)
_RI_NAMES = ("bi_reg", "bi_pend", "bi_last", "bi_code", "bi_cmd", "bi_eff",
             "he_reg", "he_pend", "he_last", "he_code", "he_cmd", "he_add", "he_eff",
             "n_iter", "dp_bitpos", "dp_ones", "dp_bits", "dp_n1", "dp_nb", "dp_err",
             "bs_above", "bs_n1", "bs_a1", "bs_a0", "mo_count", "mo_latest", "rb_v", "rb_i",
             "assign", "bs_thr", "pga", "order", "line_on", "decl", "fast_seen", "he_tcode", "bi_tcode")
RI = {n: i for i, n in enumerate(_RI_NAMES)}
NRI = len(_RI_NAMES)
# per-ring int8 block S.r8 (B, NR8, N)
_R8_NAMES = ("bi_has", "he_has", "he_flags", "jflags", "lflags", "fault", "mo_sampled", "rb_sampled", "dp_new")
R8 = {n: i for i, n in enumerate(_R8_NAMES)}
NR8 = len(_R8_NAMES)
# per-ring bool block S.rb (B, NRB, N)
_RB_NAMES = ("val_h", "val_b", "dp_on", "err_valid", "mo_valid", "mo_over", "mo_new", "rb_new", "wrong", "popped",
             "err_new")
RB = {n: i for i, n in enumerate(_RB_NAMES)}
NRB = len(_RB_NAMES)
# per-world int64 S.wi (B, NWI)
_WI_NAMES = ("edge_next", "edge_idx", "k_reset", "n_fail", "pipe_over", "mon_lost", "mon_fifo_lost", "bs_new",
             "oracle_next", "pending_edge", "ctrl_reset", "ctrl_reset_prev", "n_substeps", "n_ticks", "sat_total",
             "temp_lost", "rb_lost", "err_lost", "sub_h_max", "fast_calls", "sh_iter", "last_h")
WI = {n: i for i, n in enumerate(_WI_NAMES)}
NWI = len(_WI_NAMES)
# per-world float S.wf (B, NWF)
_WF_NAMES = ("t_base0", "t_base1", "he_scale", "rate_m_per_tick", "bi_scale", "ramp", "loss_key", "loss_val")
WF = {n: i for i, n in enumerate(_WF_NAMES)}
NWF = len(_WF_NAMES)
# per-line float block S.lf (B, NLF, L)
_LF_NAMES = ("f", "p", "pchip", "lam", "en", "df", "ps", "tune_hz", "en_b", "pdb_key", "pdb_pow")
LF = {n: i for i, n in enumerate(_LF_NAMES)}
NLF = len(_LF_NAMES)

# index constants as module ints (numba treats module-level ints as compile-time constants; they are layout, not
# config, so freezing them is intended)
(RF_BI_YS, RF_BI_YO, RF_BI_HOLD, RF_BI_VT, RF_BI_VAVG, RF_BI_VEND, RF_HE_YS, RF_HE_YO, RF_HE_UT, RF_HE_RIP,
 RF_HE_UAVG, RF_HE_P, RF_HE_V, RF_HE_I, RF_HE_R, RF_HE_TPRE, RF_HE_G, RF_P_PREV, RF_T_BRANCH, RF_T_RING, RF_T_HEATER,
 RF_P_NEW, RF_T_NEW, RF_PMAX, RF_D2P, RF_DPDT, RF_V_LO, RF_V_HI, RF_CJ, RF_RS, RF_PJ, RF_PD, RF_RHO, RF_DENS, RF_TD,
 RF_SWING, RF_EN, RF_BOFF, RF_BIAS_V, RF_GLITCH, RF_P_PD, RF_P_BG, RF_T_PD, RF_OMA, RF_DP_LAM, RF_MU1, RF_MU0,
 RF_SIG1, RF_SIG0, RF_ORACLE, RF_ORACLE_UNR, RF_DET, RF_BER, RF_BITS, RF_P_ABS, RF_SWING_V, RF_PPD_PREV,
 RF_TPD_PREV, RF_LAMRES_PREV, RF_PABS_PREV, RF_T_SENS_PREV, RF_PBG_PREV, RF_BI_VT0, RF_I_DARK) = range(NRF)
(RI_BI_REG, RI_BI_PEND, RI_BI_LAST, RI_BI_CODE, RI_BI_CMD, RI_BI_EFF, RI_HE_REG, RI_HE_PEND, RI_HE_LAST, RI_HE_CODE,
 RI_HE_CMD, RI_HE_ADD, RI_HE_EFF, RI_N_ITER, RI_DP_BITPOS, RI_DP_ONES, RI_DP_BITS, RI_DP_N1, RI_DP_NB, RI_DP_ERR,
 RI_BS_ABOVE, RI_BS_N1, RI_BS_A1, RI_BS_A0, RI_MO_COUNT, RI_MO_LATEST, RI_RB_V, RI_RB_I, RI_ASSIGN, RI_BS_THR,
 RI_PGA, RI_ORDER, RI_LINE_ON, RI_DECL, RI_FAST_SEEN, RI_HE_TCODE, RI_BI_TCODE) = range(NRI)
(R8_BI_HAS, R8_HE_HAS, R8_HE_FLAGS, R8_JFLAGS, R8_LFLAGS, R8_FAULT, R8_MO_SAMPLED, R8_RB_SAMPLED,
 R8_DP_NEW) = range(NR8)
(RB_VAL_H, RB_VAL_B, RB_DP_ON, RB_ERR_VALID, RB_MO_VALID, RB_MO_OVER, RB_MO_NEW, RB_RB_NEW, RB_WRONG, RB_POPPED,
 RB_ERR_NEW) = range(NRB)
(WI_EDGE_NEXT, WI_EDGE_IDX, WI_K_RESET, WI_N_FAIL, WI_PIPE_OVER, WI_MON_LOST, WI_MON_FIFO_LOST, WI_BS_NEW,
 WI_ORACLE_NEXT, WI_PENDING_EDGE, WI_CTRL_RESET, WI_CTRL_RESET_PREV, WI_N_SUBSTEPS, WI_N_TICKS, WI_SAT_TOTAL,
 WI_TEMP_LOST, WI_RB_LOST, WI_ERR_LOST, WI_SUB_H_MAX, WI_FAST_CALLS, WI_SH_ITER, WI_LAST_H) = range(NWI)
(WF_T_BASE0, WF_T_BASE1, WF_HE_SCALE, WF_RATE, WF_BI_SCALE, WF_RAMP, WF_LOSS_KEY, WF_LOSS_VAL) = range(NWF)
(LF_F, LF_P, LF_PCHIP, LF_LAM, LF_EN, LF_DF, LF_PS, LF_TUNE_HZ, LF_EN_B, LF_PDB_KEY, LF_PDB_POW) = range(NLF)

TR_T_RING = TR["t_ring_k"]
TR_T_HEATER = TR["t_heater_k"]
TR_T_PD = TR["t_pd_k"]
TR_P_HEATER = TR["p_heater_w"]
TR_P_ABS = TR["p_abs_w"]
TR_LAM_RES = TR["lam_res_m"]
TR_DET = TR["detuning_m"]
TR_DET_HW = TR["detuning_hwhm"]
TR_P_DROP = TR["p_drop_w"]
TR_I_MON = TR["i_monitor_a"]
TR_P_IN = TR["p_in_w"]
TR_OMA = TR["oma_w"]
TR_RELOMA = TR["reloma"]
TR_ER = TR["er_db"]
TR_IL = TR["il_db"]
TR_CODE = TR["heater_code"]
TR_BIAS = TR["bias_v"]
TR_ORDER = TR["order"]
TR_LINE_ON = TR["line_on"]
TR_T_BASE = TR["t_base_k"]
TR_MON_CODE = TR["monitor_code"]
TR_ORACLE = TR["oracle_reloma"]
TR_BER = TR["ber"]
TR_DECL = TR["declared_line"]

_SIG_T_BASE = _sg.SIG_T_BASE_K
_SIG_XPU = _sg.SIG_XPU_RISE_K
_SIG_AGG = _sg.SIG_AGGRESSOR_W
_SIG_DF = _sg.SIG_LASER_DFREQ_HZ
_SIG_PDB = _sg.SIG_LASER_POWER_DELTA_DB
_SIG_EN = _sg.SIG_LASER_LINE_ENABLE
_SIG_LOSS = _sg.SIG_INPUT_LOSS_DELTA_DB
_SIG_DENS = _sg.SIG_ONES_DENSITY
_SIG_TD = _sg.SIG_TRANSITION_DENSITY
_SIG_SWING = _sg.SIG_SWING_SCALE
_SIG_DEN = _sg.SIG_DRIVER_ENABLE
_SIG_BOFF = _sg.SIG_BIAS_OFFSET_V
_SIG_BIAS = _sg.SIG_BIAS_V
_SIG_DROP = _sg.SIG_FAULT_ADC_DROP
_SIG_STUCK = _sg.SIG_FAULT_ADC_STUCK
_SIG_SAT = _sg.SIG_FAULT_ADC_SATURATE
_SIG_GLITCH = _sg.SIG_FAULT_DAC_GLITCH_CODES
_SIG_CRESET = _sg.SIG_FAULT_CONTROLLER_RESET
_SI_OFF = _sg.SI_OFF
_SI_NC = _sg.SI_NC

# ---------------------------------------------------------------------------------------------- argument types
KParams = namedtuple("KParams", [
    # sizes, switches, timing
    "n_rings", "n_lines", "dt", "self_heating", "use_tcmt", "m_orders", "period", "phase", "jitter", "key_sched",
    "ctrl_latency", "reset_protocol", "ramp_ticks", "bias_ctrl", "policy", "fast_on",
    # optics / phase
    "ph_ring", "jt_grid", "jt_dn", "jt_al", "jt_c", "jt_rs", "jt_cum", "sh_args_i", "bus_order", "ring_to_line", "fwhm_hz",
    "z_opt_self", "dt_max_k", "sh_max_iter", "sh_tol", "sh_n_outer",
    # laser
    "las_f0", "las_p0", "las_frate", "las_prate", "las_frw", "las_prw", "las_rho_f", "las_rho_p", "las_rin",
    "las_module", "las_nmod", "las_grid_s", "las_depth", "las_tcomp", "las_key_drift", "las_key_rin",
    "las_tune_lsb", "las_input_t",
    # data path and driver
    "dp_mode", "dp_bpt_i", "dp_bpt_f", "dp_words", "dp_pre", "dp_period", "dp_key_pat", "dp_key_dp", "dp_link_t",
    "dp_oma_sens", "dp_q_ref", "dp_ber_floor", "dp_err_period", "dp_err_lat", "dp_win_bits", "dp_bs_tau",
    "dp_bs_stia", "dp_rin", "drv_swing", "drv_level", "dp_bitrate", "drv_r_out", "drv_p_static", "drv_e_tr",
    # bias DAC
    "bi_bits", "bi_bit_err", "bi_bow", "bi_vmin", "bi_vmax", "bi_lat", "bi_min_upd", "bi_noise_per", "bi_noise",
    "bi_slew", "bi_settle", "bi_key",
    # heater DAC
    "he_law", "he_bits", "he_pwm_bits", "he_ripple", "he_dt_clk", "he_bit_err", "he_bow", "he_lo", "he_hi",
    "he_lat", "he_min_upd", "he_slew", "he_settle", "he_r0", "he_alpha", "he_t0", "he_vfs", "he_ifs", "he_pfs",
    "he_vcomp", "he_pmax", "he_imax",
    # thermal
    "th_a", "th_c0", "th_c1", "th_lag_src", "th_lag_obs", "th_lag_kind", "th_opt_lags", "th_obs_ptr", "th_src_ring",
    "th_ring_obs", "th_heater_obs", "th_heater_src", "th_wg_src", "th_junction_src", "th_driver_src",
    "th_agg_src", "th_pd_obs", "th_temp_obs", "th_src_scale",
    # monitor
    "mo_pr", "mo_pw", "mo_lat", "mo_phase", "mo_pga", "mo_fl", "mo_afe", "mo_win", "mo_period", "mo_cmax",
    "mo_shot", "mo_stuck", "mo_slot", "mo_key_adc", "mo_key_shot", "mo_key_dnl",
    # temperature sensors
    "ts_tp", "ts_period", "ts_lat", "ts_cmax", "ts_fl", "ts_key",
    # heater readback
    "rb_on", "rb_lsb_v", "rb_lsb_i", "rb_noise", "rb_period", "rb_lat", "rb_cmax", "rb_avg", "rb_key",
    # scenario
    "sc_comp", "sc_sig_i", "sc_sig_f", "sc_def", "sc_comp_i", "sc_comp_f", "sc_tab", "sc_keys",
    # controller ABI
    "obs_map", "act_map",
    # metrics, oracle, records
    "mp_f", "mp_i", "assign_mode", "wrong_window", "oracle_period", "oracle_on", "rec_i", "lock_ref_mode",
    # auto substep policy (07 P0-10)
    "au_max_pow", "au_max_det", "au_pabs_tol", "au_n_outer",
])

KState = namedtuple("KState", [
    "rf", "ri", "r8", "rb", "wi", "wf", "lf",
    "las_cache_j", "las_wlo", "las_whi",
    "dp_fv", "dp_fr", "dp_fm",
    "bi_fifo", "he_fifo",
    "th_x", "th_t_obs", "th_g", "th_u0", "th_u1",
    "sh_args_f", "jx",
    "mo_sf", "mo_si", "mo_fx", "mo_fc", "mo_fv", "mo_fr", "mo_fm", "mo_samples", "mo_ticks",
    "ts_st", "ts_fx", "ts_fc", "ts_primed", "ts_fv", "ts_fr", "ts_fm", "ts_latest", "ts_new", "ts_sampled", "ts_in",
    "ts_in_prev",
    "rb_st", "rb_fv", "rb_fr", "rb_fm",
    "sc_cache", "sc_v0", "sc_v1", "sc_va", "sc_vp",
    "c_pi", "c_pf", "c_si", "c_sf", "c_si0", "c_sf0", "act", "act_tmp", "obs", "pipe_vals", "pipe_ticks",
    "pipe_meta", "sat", "tune", "fobs", "fact",
    "m_af", "m_ai", "m_hist", "m_runhist", "m_ref",
    "or_phi", "or_k2t", "or_k2d", "or_a", "or_trt", "or_psi", "or_res0", "or_res1", "or_cw", "or_fw",
    "tv", "lt",
    "rec_acc", "rec_cnt", "rec_idx", "trace",
])

STEP_SIG = types.void(types.int64[::1], types.float64[::1], types.int64[::1], types.float64[::1], types.int64[::1],
                      types.int64[::1])
#: First-class function type of the compiled-controller ABI (mrm.control.api): one kernel for every controller.
STEP_FT = types.FunctionType(STEP_SIG)


@njit(STEP_SIG, cache=True)
def null_step(p_i64, p_f64, s_i64, s_f64, obs_i64, act_i64):
    """No-op controller / fast kernel (open loop, Python path, no fast kernel)."""
    return None


# ---------------------------------------------------------------------------------------------- small helpers
@njit(cache=True, inline="always")
def _row(a, b):
    return a[b] if a.shape[0] > 1 else a[0]


@njit(cache=True, inline="always")
def _sig_off(P, sig):
    return P.sc_sig_i[sig, _SI_OFF]


@njit(cache=True)
def _edge_tick(P, b, j):
    """Tick of controller edge j of world b: phase + j period + jitter (07 P2-7; mrm.core.timing.edge_jitter)."""
    e = P.phase + j * P.period
    if P.jitter > 0:
        e += edge_jitter(P.key_sched[b], j, P.jitter)
    return e


@njit(cache=True)
def _scen_eval(P, S, b, tick, out):
    """Every signal with components of world b at ``tick`` into out (V,) (signals without components keep their
    defaults, written once at reset)."""
    bb = b if P.sc_comp_f.shape[0] > 1 else 0
    _sg.scen_eval_b(tick, P.sc_comp, P.sc_sig_i, P.sc_sig_f, P.sc_def, P.sc_comp_i, P.sc_comp_f[bb], P.sc_tab[bb],
                    P.sc_keys[b], S.sc_cache[b], out)


@njit(cache=True)
def _declared_line(P, S, b, n):
    return _decl(S.ri[b, RI_ASSIGN], P.ring_to_line, P.n_lines, n)


@njit(cache=True, inline="always")
def _decl(assign, ring_to_line, n_lines, n):
    """Declared line of ring n: tx.assignment when it holds a valid line index, else channel_plan.ring_to_line.
    (Takes arrays, not the KParams/KState tuples: a call with the tuples costs ~0.25 us of reference counting.)"""
    a = assign[n]
    return a if (a >= 0 and a < n_lines) else ring_to_line[n]


@njit(cache=True, inline="always")
def _sh_blocks(args_f, args_i):
    """Views of the per-ring block (N_SHR, N) and the per-(ring, line) block (N_SHN, N, L) of the self-heating rows
    (two reshapes; mrm.plant.optics.selfheat_views builds 29 views, ~0.4 us per call)."""
    n = args_i[_op.SH_N]
    nl = args_i[_op.SH_L]
    o = args_i[_op.SH_O_RING]
    rg = args_f[o:o + _op.N_SHR * n].reshape((_op.N_SHR, n))
    o = args_i[_op.SH_O_NL]
    nb_ = args_f[o:o + _op.N_SHN * n * nl].reshape((_op.N_SHN, n, nl))
    return rg, nb_


@njit(cache=True)
def dphi_dlamres(lam_l, shift, t_rt):
    """d phi / d lambda_res (rad/m) at line lam_l (m) of a ring whose cold phase curve is translated rigidly by the
    resonance shift ``shift`` (m), with t_rt (s) the group delay n_g(x) L / c from mrm.plant.phase.ring_optics_row:
    2 pi n_g(x) L / x^2 = 2 pi c t_rt / x^2, x = lam_l - shift (mrm.plant.phase equation 2). The slope at the
    shifted coordinate x, not at lam_l: (lam_l / x)^2 - 1 = 2 shift / lam is 1.2 % for the 7.6 nm the g26 rings are
    heated by, which is the error of 2 pi c t_rt / lam_l^2 (it scaled every detuning and the in-tick thermo-optic
    phase by that factor)."""
    x = lam_l - shift
    return _TWO_PI * C0 * t_rt / (x * x)


@njit(cache=True)
def cold_coordinate(rp_row, x0, target):
    """The root x (m) of n(x) L / x = target (target = phase / 2 pi) of the phase model's cold curve (mrm.plant.phase
    equation 2: n(x) = n_eff + (n_eff - n_g) D / lam_res - (dn_g/dlambda) D^2 / (2 lam_res), D = x - lam_res), by
    Newton from x0 with the exact derivative -n_g(x) L / x^2, n_g(x) = n - x n'(x) = n_g + dn_g D + dn_g D^2 /
    (2 lam_res). Quadratic convergence: from half an FSR (5 nm) the error after the 5 fixed iterations is below
    1e-21 m. With target = an integer order m the root is that order's cold resonance; a state whose phase at the
    line is phi sits at x(phi / 2 pi), so its order-m resonance is lambda_res = lam_l + x(m) - x(phi / 2 pi)."""
    lam_r = rp_row[_ph.RP_LAM_RES]
    ne = rp_row[_ph.RP_N_EFF]
    ng = rp_row[_ph.RP_N_G]
    dng = rp_row[_ph.RP_DNG]
    ln = rp_row[_ph.RP_LEN]
    x = x0
    for _ in range(5):
        dd = x - lam_r
        n_x = ne + (ne - ng) * dd / lam_r - 0.5 * dng * dd * dd / lam_r
        ng_x = ng + dng * dd + 0.5 * dng * dd * dd / lam_r
        g = n_x * ln / x - target
        x_new = x + g * x * x / (ng_x * ln)
        if x_new == x:
            # an exact floating-point fixed point: every remaining iteration would return x again (the update is a
            # function of x alone), so stopping here is bit-identical to the five iterations (docs/handoff/perf.md P12)
            break
        x = x_new
    return x


@njit(cache=True)
def resonance_detuning(rp_row, lam_l, shift, m):
    """Exact lambda_res - lam_l (m) of longitudinal order m of a ring whose cold phase curve is translated by
    ``shift`` (m): x_m - (lam_l - shift) with x_m the order-m cold resonance (:func:`cold_coordinate`)."""
    x_end = lam_l - shift
    return cold_coordinate(rp_row, x_end, float(m)) - x_end


# ---------------------------------------------------------------------------------------------- engine adapters
# Self-heating adapter (pabs_fn / commit_fn of mrm.plant.selfheat) on the packed rows of
# mrm.plant.optics.pack_selfheat_args, extended by the engine with EXT_I int64 entries at the end of args_i (offsets
# into args_f of the blocks below) and the matching float space at the end of args_f (:func:`extend_selfheat_rows`):
#   E_THRU  (L,)          through power of the last evaluation of a TCMT / backscatter ring (general path)
#   E_LAM   (L,)          line wavelengths the K^2 rows were computed for (NaN: none yet)
#   E_FWHM  (N, L)        lam_l^2 fwhm_hz / c of every ring at every line (Truth's wrong-line window), with E_LAM
# Exact rings without backscatter take the Airy path of mrm.plant.optics (sh_fast_precompute once per substep, then
# sh_fast_pabs per evaluation and sh_fast_outputs at the commit; docs/handoff/perf.md P1 and P4): the transcendental
# functions that do not depend on the trial temperature are evaluated once per substep, and an evaluation rotates the
# cached cos/sin of the reference phase by the small temperature-dependent angle. The reference engine runs the same
# functions through mrm.plant.optics.selfheat_pabs / selfheat_commit (which re-derive stale cache entries
# themselves), so the parity stays bit for bit. TCMT and backscatter rings keep the general path (ring_lines_nb per
# evaluation). An evaluation returns P_abs and dP_abs/dT summed in the order of
# mrm.plant.optics.dpabs_dlambda_res_nb; the commit reuses the last evaluation when the solver commits the
# temperature it evaluated last (the usual case) instead of solving the ring again. The engine's outputs hold the
# slope (RL_DABS_DPHI / R_DABS_DLAM) also for the general path (the optics adapter writes zeros there unless
# commit_slope).
EXT_I = 3
E_THRU, E_LAM, E_FWHM = range(EXT_I)


def extend_selfheat_rows(args_f, args_i):
    """(args_f', args_i') with the engine adapter's extra space (EXT_I ints; the E_* float blocks, kept even)."""
    B = args_f.shape[0]
    N = int(args_i[0, _op.SH_N])
    L = int(args_i[0, _op.SH_L])
    nf = args_f.shape[1]
    sizes = (L, L, N * L)
    offs = []
    o = nf
    for sz in sizes:
        offs.append(o)
        o += sz
    extra = o - nf
    extra += (nf + extra) % 2
    af = np.zeros((B, nf + extra))
    af[:, :nf] = args_f
    ai = np.zeros((B, args_i.shape[1] + EXT_I), np.int64)
    ai[:, :args_i.shape[1]] = args_i
    for j, oo in enumerate(offs):
        ai[:, args_i.shape[1] + j] = oo
    af[:, offs[E_LAM]:offs[E_LAM] + L] = np.nan
    return np.ascontiguousarray(af), np.ascontiguousarray(ai)


@njit(cache=True, inline="always")
def _eo(args_i, j):
    """Offset in args_f of extension block j (E_*)."""
    return args_i[args_i.shape[0] - EXT_I + j]


@njit(cache=True)
def eng_lines_changed(args_f, args_i, lam_l):
    """True when the line wavelengths differ from those the K^2 rows were computed for (or none were); records
    lam_l as the new key. The K^2 at a line depends only on the line wavelength and the world's ring constants
    (mrm.plant.phase.coupling_row), so skipping the recomputation while lam_l is unchanged is exact."""
    o = _eo(args_i, E_LAM)
    changed = False
    for l in range(lam_l.shape[0]):
        if not (args_f[o + l] == lam_l[l]):
            changed = True
            args_f[o + l] = lam_l[l]
    return changed


@njit(cache=True)
def eng_precompute(args_f, args_i):
    """Once per substep, after the optics inputs are written: the Airy-path caches and the far-line hold
    (mrm.plant.optics.sh_fast_precompute: exactly the updates the optics adapter makes lazily, so the engine and the
    reference stay bit-identical)."""
    _op.sh_fast_precompute(args_f, args_i)


@njit(cache=True)
def _eng_pabs_general(t, k, args_f, args_i):
    """eng_pabs of a TCMT or backscatter ring: mrm.plant.optics.ring_lines_nb with the slope per evaluation."""
    p_in, pcur, rg, nb_, rb, cw, fw = _op._sh_core(args_f, args_i)
    orl, orr, ol = _op._sh_outputs(args_f, args_i)
    n = args_i[_op.SH_N]
    o_thru = _eo(args_i, E_THRU)
    nl = p_in.shape[0]
    dlam_dt = _op._sh_prepare(t, k, args_i[_op.SH_HDR + n + k], p_in, pcur, rg, nb_)
    two = args_i[_op.SH_TWO] != 0
    pth = args_f[o_thru:o_thru + nl]
    _op.ring_lines_nb(k, pcur, rg[_op.SHR_P1, k], nb_[_op.SHN_PHI0], nb_[_op.SHN_PHI1], nb_[_op.SHN_A0],
                      nb_[_op.SHN_A1], nb_[_op.SHN_E0], nb_[_op.SHN_E1], nb_[_op.SHN_PSI], nb_[_op.SHN_K2T],
                      nb_[_op.SHN_K2D], nb_[_op.SHN_TRT], rb, nb_[_op.SHN_DPL], two, args_i[_op.SH_TCMT] != 0,
                      args_i[_op.SH_MORD], True, orl, orr, pth, cw, fw)
    res0 = fw[_op.FW_RES0:_op.FW_RES0 + _op.N_RES]
    res1 = fw[_op.FW_RES1:_op.FW_RES1 + _op.N_RES]
    q1 = rg[_op.SHR_P1, k] if two else 0.0
    q0 = 1.0 - q1
    eta0 = nb_[_op.SHN_E0]
    eta1 = nb_[_op.SHN_E1]
    dpl = nb_[_op.SHN_DPL]
    pabs = 0.0
    slope = 0.0
    for l in range(nl):
        ab = q0 * eta0[k, l] * res0[_op.RES_LOSS, l]
        dab = q0 * eta0[k, l] * res0[_op.RES_DLOSS, l]
        if two:
            ab += q1 * eta1[k, l] * res1[_op.RES_LOSS, l]
            dab += q1 * eta1[k, l] * res1[_op.RES_DLOSS, l]
        pabs += pcur[l] * ab
        slope += pcur[l] * dab * dpl[k, l]
    o_last = args_i[_op.SH_O_LE] + 2 * nl * _op.N_LE
    args_f[o_last] = k
    args_f[o_last + 1] = t
    return pabs, slope * dlam_dt


@njit(cache=True)
def eng_pabs(t, k, args_f, args_i):
    """pabs_fn: absorbed power of ring k at temperature t (W) and dP_abs/dT (W/K), bit-identical to
    mrm.plant.optics.selfheat_pabs; keeps what the commit needs to write every output of ring k without solving it
    again. Needs :func:`eng_precompute` for the substep."""
    if args_f[args_i[_op.SH_O_FF] + k] != 0.0:
        return _op.sh_fast_pabs(t, k, args_f, args_i)
    return _eng_pabs_general(t, k, args_f, args_i)


@njit(cache=True)
def eng_commit(t, k, args_f, args_i):
    """commit_fn: as mrm.plant.optics.selfheat_commit, reusing the last evaluation when it was ring k at t."""
    if _op.sh_last_is(k, t, args_f, args_i):
        if args_f[args_i[_op.SH_O_FF] + k] != 0.0:
            _op.sh_fast_outputs(k, args_f, args_i)
        else:
            o_thru = _eo(args_i, E_THRU)
            o_pcur = args_i[_op.SH_O_PCUR]
            for l in range(args_i[_op.SH_L]):
                args_f[o_pcur + l] = args_f[o_thru + l]
        _op.sh_pass_on(k, args_f, args_i)
    else:
        _op.selfheat_commit(t, k, args_f, args_i)
    _op.sh_last_clear(args_f, args_i)


@njit(cache=True)
def swing_capacitance_row(v_lo, v_hi, grid, c_col, cum_q, fill, length_m, out_c_eff):
    """Charge-equivalent junction capacitance over the NRZ swing (F): C_eff = f L (Q(V_hi) - Q(V_lo))/(V_hi - V_lo)
    with Q the integral of the piecewise-linear (end-held) c'(V), from a cumulative table cum_q (G,) at the grid
    nodes; the same quantity as mrm.plant.electrical.junction_swing_capacitance_row (07 P1-8), which walks every grid
    segment with two binary searches each (about 50 us per tick for 8 rings at a 10 mV grid). Agreement to 1e-12
    relative is tested; for |V_hi - V_lo| <= 1e-9 V it is f L c'(mid), as there."""
    g = grid.shape[0]
    for n in range(out_c_eff.shape[0]):
        lo = v_lo[n]
        hi = v_hi[n]
        fl = fill[_el._ix(fill, n)] * length_m[_el._ix(length_m, n)]
        if abs(hi - lo) > 1e-9:
            out_c_eff[n] = fl * (_cum_q(hi, grid, c_col, cum_q, g) - _cum_q(lo, grid, c_col, cum_q, g)) / (hi - lo)
        else:
            cp, _ = _el._interp_lin(0.5 * (lo + hi), grid, c_col)
            out_c_eff[n] = fl * cp


@njit(cache=True, inline="always")
def _cum_q(x, grid, col, cum, g):
    if x <= grid[0]:
        return col[0] * (x - grid[0])
    if x >= grid[g - 1]:
        return cum[g - 1] + col[g - 1] * (x - grid[g - 1])
    j = _el.grid_index(x, grid)                # np.searchsorted(grid, x) - 1, without the binary search
    h = grid[j + 1] - grid[j]
    d = x - grid[j]
    return cum[j] + d * (col[j] + 0.5 * (d / h) * (col[j + 1] - col[j]))


def cumulative_capacitance(grid, c_tab):
    """cum_q (Bt, G): integral of c'(V) from grid[0] to each node (trapezoids of the piecewise-linear table)."""
    c = np.asarray(c_tab, float)
    g = np.asarray(grid, float)
    seg = 0.5 * (c[:, 1:] + c[:, :-1]) * np.diff(g)[None, :]
    out = np.zeros_like(c)
    out[:, 1:] = np.cumsum(seg, axis=1)
    return np.ascontiguousarray(out)


# ---------------------------------------------------------------------------------------------- implicit step (copy)
@njit(cache=True)
def _solve_ring_eng(k, args_f, args_i, t_other, g, t_start, p_max, d2p_max, max_iter, tol_k):
    """mrm.plant.selfheat.solve_ring_temperature with the engine adapter eng_pabs (verbatim otherwise)."""
    if g == 0.0 or g != g:
        p, d = eng_pabs(t_other, k, args_f, args_i)
        return t_other, p, d, 0, STATUS_CONVERGED
    # F(lo) = (lo - t_other) - g P(lo) <= 0 and F(hi) >= 0 for either sign of g when 0 <= P <= p_max
    lo = min(t_other, t_other + g * p_max)
    hi = max(t_other, t_other + g * p_max)
    t0 = t_start
    if t0 != t0:                              # NaN: start at the cold end; +-inf select the high/low end (below)
        t0 = t_other
    if t0 < lo:
        t0 = lo
    if t0 > hi:
        t0 = hi
    p0, d0 = eng_pabs(t0, k, args_f, args_i)
    f0 = t0 - t_other - g * p0
    if abs(f0) <= tol_k:
        return t0, p0, d0, 0, STATUS_CONVERGED
    sgn = 1.0 if f0 < 0.0 else -1.0          # the root lies in the direction sgn from t0
    near, f_near, p_near, d_near = t0, f0, p0, d0
    far = hi if sgn > 0.0 else lo
    f_far, p_far, d_far = 0.0, 0.0, 0.0
    have_far = False
    k2 = abs(g) * d2p_max if d2p_max > 0.0 else 0.0
    used = 0
    for it in range(1, max_iter + 1):
        if (far - near) * sgn <= tol_k:
            break
        used = it
        fp = 1.0 - g * d_near                # F'(near); |F| decreases at this rate along sgn
        af = abs(f_near)
        cand = math.nan
        if k2 > 0.0 and math.isfinite(fp):
            step = 2.0 * af / (fp + math.sqrt(fp * fp + 2.0 * k2 * af))
            c = near + sgn * step
            if (far - c) * sgn > 0.0:
                cand = c
            elif not have_far:
                cand = far                    # the bracket end: F(far) has the root's sign by the p_max bound
            # else: the step reached a sign change already found (inexact dP/dT or d2p_max): secant below
        elif math.isfinite(fp) and fp > 0.0:
            c = near + sgn * af / fp
            if (c - near) * sgn > 0.0 and (far - c) * sgn > 0.0:
                cand = c
        if math.isnan(cand):
            if have_far and f_far != f_near:
                c = near - f_near * (far - near) / (f_far - f_near)
                if (c - near) * sgn > 0.0 and (far - c) * sgn > 0.0:
                    cand = c
            if math.isnan(cand):
                cand = 0.5 * (near + far)
        pc, dc = eng_pabs(cand, k, args_f, args_i)
        fc = cand - t_other - g * pc
        if abs(fc) <= tol_k:
            return cand, pc, dc, it, STATUS_CONVERGED
        if fc * f_near > 0.0:
            near, f_near, p_near, d_near = cand, fc, pc, dc
        else:
            far, f_far, p_far, d_far = cand, fc, pc, dc
            have_far = True
    if (far - near) * sgn <= tol_k:
        if have_far:
            if abs(f_far) < abs(f_near):
                return far, p_far, d_far, used, STATUS_CONVERGED
            return near, p_near, d_near, used, STATUS_CONVERGED
        # no sign change met up to the bracket end: with a valid p_max, F(bracket end) has the root's sign or is 0
        # (then |F| <= tol_k returned above); here P_abs exceeded p_max
        return near, p_near, d_near, used, STATUS_BOUND
    return near, p_near, d_near, used, STATUS_MAX_ITER



# Thermal pass tables (docs/handoff/perf.md P8). The lag pass of mrm.plant.selfheat.implicit_step_b and the heater split
# of mrm.plant.thermal.observer_split_b branch on the lag kind / source at every lag; the engine runs the same
# arithmetic from tables built once per block (lag indices by kind, the self-heater range of each heater observer) and
# once per block and substep power of two (the per-lag gain w = src_scale (c0 + c1), the implicit gain of each
# observer and the heater split's gain, each summed in lag order as there), so every value is bit-identical.
@njit(cache=True)
def _th_struct(lag_src, lag_kind, obs_ptr, heater_obs, heater_src, src_scale):
    """(idx_exo, idx_cross, idx_ax, unit, hs_lo, hs_hi): lag indices of kind LAG_EXOGENOUS, LAG_OPT_CROSS_RING and the
    rest (a x only in the first pass); unit = every src_scale is exactly 1.0 (then the factor is skipped: 1.0 y == y);
    [hs_lo[n], hs_hi[n]) the lags of heater_src[n] in heater observer n when they are contiguous (else hs_lo = -1)."""
    m_tot = lag_kind.shape[0]
    n_exo = 0
    n_cross = 0
    for m in range(m_tot):
        if lag_kind[m] == LAG_EXOGENOUS:
            n_exo += 1
        elif lag_kind[m] == LAG_OPT_CROSS_RING:
            n_cross += 1
    idx_exo = np.empty(n_exo, np.int64)
    idx_cross = np.empty(n_cross, np.int64)
    idx_ax = np.empty(m_tot - n_exo - n_cross, np.int64)
    i0 = 0
    i1 = 0
    i2 = 0
    for m in range(m_tot):
        if lag_kind[m] == LAG_EXOGENOUS:
            idx_exo[i0] = m
            i0 += 1
        elif lag_kind[m] == LAG_OPT_CROSS_RING:
            idx_cross[i1] = m
            i1 += 1
        else:
            idx_ax[i2] = m
            i2 += 1
    unit = True
    for s in range(src_scale.shape[0]):
        if not (src_scale[s] == 1.0):
            unit = False
    nr = heater_obs.shape[0]
    hs_lo = np.full(nr, -1, np.int64)
    hs_hi = np.full(nr, -1, np.int64)
    for n in range(nr):
        o = heater_obs[n]
        lo = -1
        hi = -1
        for m in range(obs_ptr[o], obs_ptr[o + 1]):
            if lag_src[m] == heater_src[n]:
                if lo < 0:
                    lo = m
                hi = m + 1
        ok = lo >= 0
        for m in range(max(lo, 0), max(hi, 0)):
            if lag_src[m] != heater_src[n]:
                ok = False
        if ok:
            hs_lo[n] = lo
            hs_hi[n] = hi
    return idx_exo, idx_cross, idx_ax, unit, hs_lo, hs_hi


@njit(cache=True)
def _th_gains(c0, c1, lag_src, lag_kind, obs_ptr, heater_obs, heater_src, src_scale, w, gobs, gsplit):
    """For one substep power: w[m] = src_scale[s] (c0[m] + c1[m]); gobs[o] = sum of w over the LAG_OPT_SELF lags of
    observer o and gsplit[n] = sum of w over the lags of heater_src[n] in heater observer n, both in lag order (the
    expressions and order of implicit_step_b and observer_split_b)."""
    for m in range(lag_kind.shape[0]):
        w[m] = src_scale[lag_src[m]] * (c0[m] + c1[m])
    for o in range(obs_ptr.shape[0] - 1):
        g = 0.0
        for m in range(obs_ptr[o], obs_ptr[o + 1]):
            if lag_kind[m] == LAG_OPT_SELF:
                g += src_scale[lag_src[m]] * (c0[m] + c1[m])
        gobs[o] = g
    for n in range(heater_obs.shape[0]):
        o = heater_obs[n]
        g = 0.0
        for m in range(obs_ptr[o], obs_ptr[o + 1]):
            if lag_src[m] == heater_src[n]:
                g += src_scale[lag_src[m]] * (c0[m] + c1[m])
        gsplit[n] = g


@njit(cache=True)
def _heater_split(x, a, c0, c1, lag_src, obs_ptr, obs, self_src, lo_s, hi_s, unit, u0, u1, src_scale):
    """pre of mrm.plant.thermal.observer_split_b (the gain comes from _th_gains), bit-identical: the same sum in the
    same order, without the per-lag source test when the self-heater lags form the range [lo_s, hi_s)."""
    pre = 0.0
    lo = obs_ptr[obs]
    hi = obs_ptr[obs + 1]
    if lo_s < 0 or not unit:
        for m in range(lo, hi):
            s = lag_src[m]
            pre += a[m] * x[m]
            if s != self_src:
                pre += src_scale[s] * (c0[m] * u0[s] + c1[m] * u1[s])
        return pre
    for m in range(lo, lo_s):
        s = lag_src[m]
        pre += a[m] * x[m]
        pre += c0[m] * u0[s] + c1[m] * u1[s]
    for m in range(lo_s, hi_s):
        pre += a[m] * x[m]
    for m in range(hi_s, hi):
        s = lag_src[m]
        pre += a[m] * x[m]
        pre += c0[m] * u0[s] + c1[m] * u1[s]
    return pre


@njit(cache=True)
def _heater_split_tables(obs_ptr, lag_src, heater_obs, heater_src):
    """(ln, lo, is_self) for :func:`_heater_split_all`: the common number of lags of the heater observers (-1 when
    they differ), the first lag of each (N,) and, per ring and lag position, whether the lag is the ring's own
    heater's (N, ln) bool."""
    nr = heater_obs.shape[0]
    ln = obs_ptr[heater_obs[0] + 1] - obs_ptr[heater_obs[0]] if nr > 0 else 0
    lo = np.empty(nr, np.int64)
    for n in range(nr):
        lo[n] = obs_ptr[heater_obs[n]]
        if obs_ptr[heater_obs[n] + 1] - lo[n] != ln:
            ln = -1
    is_self = np.zeros((nr, max(ln, 0)), np.bool_)
    for n in range(nr):
        for j in range(max(ln, 0)):
            is_self[n, j] = lag_src[lo[n] + j] == heater_src[n]
    return ln, lo, is_self


@njit(cache=True)
def _heater_split_all(x, a, c0, c1, lag_src, obs_ptr, heater_obs, heater_src, hs_lo, hs_hi, unit, u0, u1, src_scale,
                      ln, lo, is_self, out_pre):
    """_heater_split of every ring into out_pre (N,). When the heater observers have the same number of lags (ln >= 0,
    tables of :func:`_heater_split_tables`), the N sums run in lockstep (lag j of every ring, then lag j + 1), each in
    its own order, so the N independent addition chains overlap in the CPU instead of running one after the other.
    The same additions in the same order per ring as _heater_split, so the values are bit-identical."""
    nr = heater_obs.shape[0]
    if ln < 0 or not unit:
        for n in range(nr):
            out_pre[n] = _heater_split(x, a, c0, c1, lag_src, obs_ptr, heater_obs[n], heater_src[n], hs_lo[n],
                                       hs_hi[n], unit, u0, u1, src_scale)
        return
    sums = np.zeros(nr)                 # a local array: the compiler need not assume it aliases the inputs
    for j in range(ln):
        for n in range(nr):
            m = lo[n] + j
            pre = sums[n] + a[m] * x[m]
            if not is_self[n, j]:
                s = lag_src[m]
                pre = pre + (c0[m] * u0[s] + c1[m] * u1[s])
            sums[n] = pre
    for n in range(nr):
        out_pre[n] = sums[n]


@njit(cache=True)
def _implicit_step_eng(x, a, c0, c1, lag_src, lag_obs, lag_kind, opt_lags, obs_ptr, src_ring, ring_obs, bus_order,
                       u0, u1, src_scale, t_base, p_prev, t_prev, args_f, args_i, p_max, d2p_max, max_iter, tol_k,
                       n_outer, g_work, t_obs, p_new, t_ring, n_iter, w, gobs, idx_exo, idx_cross, idx_ax, unit):
    """mrm.plant.selfheat.implicit_step_b (owner: plant_thermal; 07 P0-1) calling the engine adapter eng_pabs /
    eng_commit directly, the same arithmetic otherwise. Why a copy: passing the adapter as a first-class value makes
    numba embed the dispatcher addresses in the code ("dynamic globals"), and then the kernel cannot be cached on disk
    (every process would recompile the plant for minutes). The lag passes run from the tables of _th_struct /
    _th_gains (w, gobs, the index lists, unit) instead of branching per lag: bit-identical (docs/handoff/perf.md P8).
    Kept equal to the original by tests/verification/test_kernel_parity.py (the reference engine runs the
    original)."""
    nr = ring_obs.shape[0]
    if bus_order.shape[0] != nr:
        raise ValueError("implicit_step: bus_order must list every ring once")
    for k in range(nr):
        n_iter[k] = 0
    for q in range(nr):                    # bus_order must be a permutation of 0..N-1 (n_iter as the marker)
        k = bus_order[q]
        if k < 0 or k >= nr or n_iter[k] != 0:
            raise ValueError("implicit_step: bus_order must be a permutation of the ring indices")
        if ring_obs[k] < 0:
            raise ValueError("implicit_step needs the ring_mode_weighted observer of every ring")
        n_iter[k] = 1
    # --- every lag advanced with the known inputs (observer sums and implicit gains)
    n_obs = obs_ptr.shape[0] - 1
    # two passes (the same values and, per observer, the same summation order as one pass over the observers): every
    # lag with its known input (by kind, from the index lists: a x + input with the input of implicit_step_b, whose
    # src_scale factor is skipped when it is exactly 1.0), then the sums; the implicit gains are the tabled gobs
    if unit:
        for i in range(idx_exo.shape[0]):
            m = idx_exo[i]
            s = lag_src[m]
            x[m] = a[m] * x[m] + (c0[m] * u0[s] + c1[m] * u1[s])
    else:
        for i in range(idx_exo.shape[0]):
            m = idx_exo[i]
            s = lag_src[m]
            x[m] = a[m] * x[m] + src_scale[s] * (c0[m] * u0[s] + c1[m] * u1[s])
    for i in range(idx_cross.shape[0]):
        m = idx_cross[i]
        x[m] = a[m] * x[m] + w[m] * p_prev[src_ring[lag_src[m]]]
    for i in range(idx_ax.shape[0]):
        m = idx_ax[i]
        x[m] = a[m] * x[m]
    for o in range(n_obs):
        e = t_base
        for m in range(obs_ptr[o], obs_ptr[o + 1]):
            e += x[m]
        t_obs[o] = e                       # everything except the new optical inputs
        g_work[o] = gobs[o]
    # --- ring solves in bus order
    n_fail = 0
    p_cross = np.empty(nr)                 # P_abs currently driving the LAG_OPT_CROSS_RING lags
    for k in range(nr):
        p_cross[k] = p_prev[k]
        t_ring[k] = t_prev[k]
        n_iter[k] = 0
    for sweep in range(max(n_outer, 1)):
        n_fail = 0
        for q in range(nr):
            k = bus_order[q]
            o = ring_obs[k]
            if o < 0:
                raise ValueError("implicit_step needs the ring_mode_weighted observer of every ring")
            # --- ring solve (Newton, ring evaluations)
            t, p, d, it, st = _solve_ring_eng(k, args_f, args_i, t_obs[o], g_work[o], t_ring[k],
                                                     p_max[k], d2p_max[k], max_iter, tol_k)
            t_ring[k] = t
            p_new[k] = p
            n_iter[k] += it
            if st != STATUS_CONVERGED:
                n_fail += 1
                # not converged (a jump between branches longer than max_iter steps, or a p_max that does not bound
                # P_abs): the thermal state receives the power that puts the ring exactly at the returned point,
                # which lies before the root, so the ring lags its root and never overshoots it; the next substep
                # continues the crossing.
                gk = g_work[o]
                if gk != 0.0:
                    p_new[k] = (t - t_obs[o]) / gk
            # --- commit (outputs of the ring, through power to the next ring)
            eng_commit(t, k, args_f, args_i)
            # --- ring solves in bus order
        if sweep + 1 >= n_outer:
            break
        # Gauss-Seidel: drive the cross-ring lags with the newest P_abs, then sweep again
        change = 0.0
        scale = 0.0
        for k in range(nr):
            change = max(change, abs(p_new[k] - p_cross[k]))
            scale = max(scale, abs(p_new[k]))
        if change <= 1e-12 * scale:
            break
        for i in range(idx_cross.shape[0]):
            m = idx_cross[i]
            j = src_ring[lag_src[m]]
            dx = w[m] * (p_new[j] - p_cross[j])
            x[m] += dx
            t_obs[lag_obs[m]] += dx
        for k in range(nr):
            p_cross[k] = p_new[k]
    # --- optical lags with the new absorbed powers
    # (each observer's sum is carried in a register while consecutive lags feed it: the same additions in the same
    # order as t_obs[lag_obs[m]] += dx, without a store-to-load round trip per lag; bit-identical)
    o_cur = -1
    t_run = 0.0
    for i in range(opt_lags.shape[0]):
        m = opt_lags[i]
        o = lag_obs[m]
        if o != o_cur:
            if o_cur >= 0:
                t_obs[o_cur] = t_run
            o_cur = o
            t_run = t_obs[o]
        dx = w[m] * p_new[src_ring[lag_src[m]]]
        x[m] += dx
        t_run += dx
    if o_cur >= 0:
        t_obs[o_cur] = t_run
    return n_fail




# ---------------------------------------------------------------------------------------------- sensor release
@njit(cache=True)
def release_row(P, S, b, k):
    """Event-order step 2 at controller edge k for world b: release the monitor, temperature, readback and
    error-counter FIFOs, run the bit-statistics window, and pack obs_i64 (07 P0-6, P0-7, P0-11)."""
    n_r = P.n_rings
    rf = S.rf[b]
    ri = S.ri[b]
    rbb = S.rb[b]
    lost = monitor_release_row(k, _row(P.mo_lat, b), S.mo_fv[b], S.mo_fr[b], S.mo_fm[b], S.mo_samples[b],
                               S.mo_ticks[b], ri[RI_MO_COUNT], ri[RI_MO_LATEST], rbb[RB_MO_VALID], rbb[RB_MO_OVER],
                               rbb[RB_MO_NEW])
    S.wi[b, WI_MON_LOST] += lost
    if S.ts_latest.shape[1] > 0:
        temp_release_row(k, S.ts_fv[b], S.ts_fr[b], S.ts_fm[b], S.ts_latest[b], S.ts_new[b])
    if P.rb_on != 0:
        readback_release_row(k, S.rb_fv[b], S.rb_fr[b], S.rb_fm[b], ri[RI_RB_V], ri[RI_RB_I], rbb[RB_RB_NEW])
    errors_release_row(k, S.dp_fv[b], S.dp_fr[b], S.dp_fm[b], ri[RI_DP_ERR], rbb[RB_ERR_VALID])
    # bit statistics (07 P0-11): level codes of the last interval through the sampler bandwidth, then the window
    out_rl_, out_r, out_l_ = _op._sh_outputs(S.sh_args_f[b], P.sh_args_i[b])
    monitor_level_codes_row(out_r[_op.R_DROP_SUM1], out_r[_op.R_DROP_SUM0], rf[RF_P_BG], rf[RF_T_PD], rf[RF_T_PD],
                            ri[RI_PGA], P.mo_pr[b], P.mo_pw[b], P.mo_pga, P.mo_shot, P.dp_bs_tau, rf[RF_MU1],
                            rf[RF_MU0], rf[RF_SIG1], rf[RF_SIG0], P.dp_bs_stia, P.dp_rin[b])
    bitstat_window_row(k, P.dp_win_bits, P.dp_mode, rf[RF_RHO], rf[RF_MU1], rf[RF_MU0], rf[RF_SIG1], rf[RF_SIG0],
                       ri[RI_BS_THR], P.dp_words, P.dp_pre, P.dp_period, ri[RI_DP_BITPOS], P.dp_key_dp[b],
                       ri[RI_BS_ABOVE], ri[RI_BS_N1], ri[RI_BS_A1], ri[RI_BS_A0])
    temp_lat = S.ts_latest[b]
    temp_new = S.ts_new[b]
    pack_obs_row(S.obs[b], P.obs_map, k, -1, S.mo_samples[b], S.mo_ticks[b], ri[RI_MO_COUNT], ri[RI_MO_LATEST],
                 rbb[RB_MO_VALID], rbb[RB_MO_OVER], rbb[RB_MO_NEW], temp_lat, temp_new, ri[RI_RB_V], ri[RI_RB_I],
                 rbb[RB_RB_NEW], ri[RI_DP_ONES], ri[RI_DP_BITS], ri[RI_BS_ABOVE], True, ri[RI_DP_ERR],
                 rbb[RB_ERR_VALID], ri[RI_HE_REG], ri[RI_BI_REG])
    for n in range(n_r):
        ri[RI_DP_ONES, n] = 0
        ri[RI_DP_BITS, n] = 0


@njit(cache=True)
def _fast_tick(fast_fn, P, S, b, k):
    """07 P0-7 fast_kernel: once per monitor sample released at tick k (release tick = sample tick + ADC latency
    <= k, not yet delivered), before the controller step. fobs = [tick, N, new[N], code[N], sample_tick[N]];
    fact = [heater offset[N], bias offset[N]] (codes, held until the next call; added to the latched registers
    before the DAC FIFOs). The samples stay queued for the edge release."""
    n_r = P.n_rings
    lat = _row(P.mo_lat, b)
    fo = S.fobs[b]
    ri = S.ri[b]
    fv = S.mo_fv[b]
    fr = S.mo_fr[b]
    fm = S.mo_fm[b]
    cap = fv.shape[1]
    while True:
        any_new = False
        fo[0] = k
        fo[1] = n_r
        for n in range(n_r):
            fo[2 + n] = 0
            fo[2 + n_r + n] = 0
            fo[2 + 2 * n_r + n] = 0
            h = fm[n, 0]
            cnt = fm[n, 1]
            for i in range(cnt):
                idx = (h + i) % cap
                rel = fr[n, idx]
                if rel > k:
                    break
                if rel <= ri[RI_FAST_SEEN, n]:
                    continue
                val = fv[n, idx]
                ri[RI_FAST_SEEN, n] = rel
                if ((val >> _FLAG_SHIFT) & SAMPLE_DROPPED) != 0:
                    continue
                fo[2 + n] = 1
                fo[2 + n_r + n] = val & _CODE_MASK
                fo[2 + 2 * n_r + n] = rel - lat
                any_new = True
                break
        if not any_new:
            break
        fast_fn(_row(S.c_pi, b), _row(S.c_pf, b), S.c_si[b], S.c_sf[b], fo, S.fact[b])
        S.wi[b, WI_FAST_CALLS] += 1


@njit(cache=True)
def _edge(step_fn, P, S, b, k, mode):
    """Controller edge k of world b (event order steps 2-4). Returns True when the Python path must take over."""
    release_row(P, S, b, k)
    if mode == MODE_PYTHON:
        S.wi[b, WI_PENDING_EDGE] = k
        return True
    if mode == MODE_COMPILED:
        if S.wi[b, WI_CTRL_RESET] != 0:                     # scenario fault.controller_reset: back to init state
            S.c_si[b, :] = S.c_si0[b, :]
            S.c_sf[b, :] = S.c_sf0[b, :]
            _sync_act_from_registers(P, S, b)
            S.wi[b, WI_CTRL_RESET] = 0
        step_fn(_row(S.c_pi, b), _row(S.c_pf, b), S.c_si[b], S.c_sf[b], S.obs[b], S.act[b])
        _push_act(P, S, b, k)
    _advance_edge(P, S, b)
    return False


@njit(cache=True)
def _sync_act_from_registers(P, S, b):
    """Pre-fill the held command from the latched DAC registers (the ABI rule of mrm.control.api)."""
    am = P.act_map
    for i in range(am.shape[0]):
        o = am[i, 1]
        for j in range(am[i, 2]):
            if am[i, 0] == 0:
                S.act[b, o + j] = S.ri[b, RI_HE_REG, j]
            elif am[i, 0] == 1:
                S.act[b, o + j] = S.ri[b, RI_BI_REG, j]


@njit(cache=True)
def _push_act(P, S, b, k):
    """Event-order step 4: clip every actuator to its channel range (in the held vector itself, so the controller
    sees the clipped command next time as the applied register does), count the clipped elements, and queue the
    command to latch at k + Controller.latency_ticks."""
    S.wi[b, WI_SAT_TOTAL] += clip_act_row(S.act[b], P.act_map, S.sat[b])
    if not pipe_push_row(S.pipe_vals[b], S.pipe_ticks[b], S.pipe_meta[b], S.act[b], k + P.ctrl_latency):
        S.wi[b, WI_PIPE_OVER] += 1


@njit(cache=True)
def _advance_edge(P, S, b):
    j = S.wi[b, WI_EDGE_IDX] + 1
    S.wi[b, WI_EDGE_IDX] = j
    S.wi[b, WI_EDGE_NEXT] = _edge_tick(P, b, j)


# ---------------------------------------------------------------------------------------------- truth and metrics
@njit(cache=True)
def _truth_a(n_r, n_l, rf, ri, rbb, tv, lf, lt, t_base, args_f, args_i, pw, mo_pr, rp, jx, fwhm_b, wrong_window,
             assign_mode, with_db, dark_known):
    """Truth of one world at the end of the last step (ARCHITECTURE Truth; metrics.TRUTH_FIELDS) into tv, the
    per-line truth lt = [chip input power, bus output power, wavelength] (L,), the detuning rf[RF_DET], the order,
    the line each ring sits on and the wrong-line flag against the declared assignment (07 P1-12). Takes the world's
    arrays (pw = P.mo_pw[b], mo_pr = P.mo_pr[b], rp = P.ph_ring[b], fwhm_b = P.fwhm_hz[b]) instead of the tuples.
    with_db: also er_db and il_db (two log10 per ring; no kernel computation reads them, so the world block asks for
    them only when a record does and at the last step of a block; docs/handoff/perf.md P12). dark_known: the PD dark
    current of this step is in rf[RF_I_DARK] (the monitor's value, the same expression), else it is computed."""
    rg, nb_ = _sh_blocks(args_f, args_i)
    phi0r, phi1r, dpl = nb_[_op.SHN_PHI0R], nb_[_op.SHN_PHI1R], nb_[_op.SHN_DPL]
    tref, s0v, betav = rg[_op.SHR_TREF], rg[_op.SHR_S0], rg[_op.SHR_BETA]
    out_rl, out_r, out_l = _op._sh_outputs(args_f, args_i)
    o_fwhm = _eo(args_i, E_FWHM)
    # --- per ring: line search, exact detuning of the declared line, Truth fields
    for n in range(n_r):
        dT = rf[RF_T_RING, n] - tref[n]
        shift = s0v[n] * (dT + 0.5 * betav[n] * dT * dT)      # R1 about T_prev (mrm.plant.phase local law)
        dl = ri[RI_DECL, n]
        best = -1
        best_d = _INF
        order_decl = 0
        for l in range(n_l):
            ph = 0.5 * (phi0r[n, l] + phi1r[n, l]) + dpl[n, l] * shift
            m = math.floor(ph / _TWO_PI + 0.5)
            d = (ph - _TWO_PI * m) / dpl[n, l]                  # lambda_res - lambda_line (first order in d)
            if l == dl:
                order_decl = np.int64(m)
            fwhm = args_f[o_fwhm + n * n_l + l]          # lam_l^2 fwhm_hz / c, set when the lines move
            if abs(d) <= wrong_window * fwhm and abs(d) < best_d:
                best_d = abs(d)
                best = l
        # the declared line exactly: the mean of the two NRZ level resonances of the order nearest the line, from the
        # phase model's cold curve translated by the end-of-step mean shift (shifts at T_prev plus the R1 increment)
        lam_l = lf[LF_LAM, dl]
        det_decl = resonance_detuning(rp[n], lam_l, 0.5 * (jx[0, n, 0] + jx[0, n, 1]) + shift, order_decl)
        ri[RI_ORDER, n] = order_decl
        ri[RI_LINE_ON, n] = best
        rf[RF_DET, n] = det_decl
        hw = 0.5 * lam_l * lam_l * fwhm_b[n] / C0
        tv[TR_T_RING, n] = rf[RF_T_RING, n]
        tv[TR_T_HEATER, n] = rf[RF_T_HEATER, n]
        tv[TR_T_PD, n] = rf[RF_T_PD, n]
        tv[TR_P_HEATER, n] = rf[RF_HE_P, n]
        pabs = out_r[_op.R_ABS_SUM, n]
        rf[RF_P_ABS, n] = pabs
        tv[TR_P_ABS, n] = pabs
        tv[TR_LAM_RES, n] = lam_l + det_decl
        tv[TR_DET, n] = det_decl
        tv[TR_DET_HW, n] = det_decl / hw
        tv[TR_P_DROP, n] = out_r[_op.R_DROP_SUM, n]
        i_sig = pw[PW_GAIN_SCALE] * mo_pr[n, PR_RESP] * (out_r[_op.R_DROP_SUM, n] + mo_pr[n, PR_BG] * rf[RF_P_BG, n])
        if dark_known:
            tv[TR_I_MON, n] = i_sig + rf[RF_I_DARK, n]
        else:
            tv[TR_I_MON, n] = i_sig + dark_current_a(pw[PW_DARK_I0], rf[RF_T_PD, n], pw[PW_DARK_T0], pw[PW_DARK_EA])
        pin = out_rl[_op.RL_PIN, n, dl]
        t1 = out_rl[_op.RL_THRU1, n, dl]
        t0 = out_rl[_op.RL_THRU0, n, dl]
        tv[TR_P_IN, n] = pin
        tv[TR_OMA, n] = rf[RF_OMA, n]
        rel = abs(t1 - t0) / pin if pin > 0.0 else 0.0
        tv[TR_RELOMA, n] = rel
        if with_db:
            hi = t1 if t1 > t0 else t0
            lo = t0 if t1 > t0 else t1
            tv[TR_ER, n] = 10.0 * math.log10(hi / lo) if lo > 0.0 else _INF
            tv[TR_IL, n] = -10.0 * math.log10(hi / pin) if (pin > 0.0 and hi > 0.0) else _INF
        tv[TR_CODE, n] = ri[RI_HE_REG, n]
        tv[TR_BIAS, n] = rf[RF_BI_VAVG, n]
        tv[TR_ORDER, n] = order_decl
        tv[TR_T_BASE, n] = t_base
        tv[TR_MON_CODE, n] = ri[RI_MO_LATEST, n]
        tv[TR_ORACLE, n] = rf[RF_ORACLE, n]
        tv[TR_DECL, n] = dl
        tv[TR_BER, n] = rf[RF_BER, n]                 # ber_from_oma of rf[RF_OMA], set with it (_world_block)
    # --- wrong line against the declared assignment (07 P1-12; metrics.allowed_assignments)
    for n in range(n_r):
        lo_ = ri[RI_LINE_ON, n]
        w = False
        if lo_ >= 0 and assign_mode == 0:
            w = lo_ != ri[RI_DECL, n]
        elif lo_ >= 0 and assign_mode == 1:
            off = (lo_ - ri[RI_DECL, n]) % n_l
            # the common (most frequent) offset of the rings that sit on a line: a barrel shift is allowed (PAT-1)
            best_c = -1
            best_o = 0
            for m in range(n_r):
                lm = ri[RI_LINE_ON, m]
                if lm < 0:
                    continue
                om = (lm - ri[RI_DECL, m]) % n_l
                c = 0
                for q in range(n_r):
                    lq = ri[RI_LINE_ON, q]
                    if lq >= 0 and (lq - ri[RI_DECL, q]) % n_l == om:
                        c += 1
                if c > best_c or (c == best_c and om < best_o):
                    best_c = c
                    best_o = om
            w = off != best_o
        rbb[RB_WRONG, n] = w
        tv[TR_LINE_ON, n] = lo_
    for l in range(n_l):
        lt[0, l] = lf[LF_PCHIP, l]
        lt[1, l] = out_l[_op.L_POUT, l]
        lt[2, l] = lf[LF_LAM, l]




@njit(cache=True)
def truth_row(P, S, b):
    """Truth of world b at the end of the last step (:func:`_truth_a` on world b's arrays)."""
    _truth_a(P.n_rings, P.n_lines, S.rf[b], S.ri[b], S.rb[b], S.tv[b], S.lf[b], S.lt[b], S.wf[b, WF_T_BASE1],
             S.sh_args_f[b], P.sh_args_i[b], P.mo_pw[b], P.mo_pr[b], P.ph_ring[b], S.jx[b], P.fwhm_hz[b],
             P.wrong_window, P.assign_mode, True, False)


@njit(cache=True)
def _oracle(P, S, b):
    """Oracle relOMA of every ring at the current inputs (07 P1-1, P0-1 item 7): the largest |T1 - T0| at the ring's
    through port for its declared line over a scan of the mean-resonance detuning (+-1.5 FWHM), among points whose
    static self-heating loop gain G dP_abs/dlambda_res < 1 (G = s(T) Z_opt,self; only the declared line's absorption
    slope, the others are >= 2.29 nm away), and without that restriction. The two NRZ levels keep their phase
    difference. The plant's ring model per scan point (mrm.plant.optics.ring_rows_nb: exact or TCMT). With
    P.lock_ref_mode = 1 the detuning of the best stable point (of the best point when none is stable), refined by a
    parabola through it and its neighbours, becomes the lock reference S.m_ref (mrm.sim.metrics 'oracle'), converted
    exactly with :func:`cold_coordinate`."""
    n_r = P.n_rings
    rf = S.rf[b]
    ri = S.ri[b]
    v = _op.selfheat_views(S.sh_args_f[b], P.sh_args_i[b])
    phi0r, phi1r, a0, a1, e0, e1 = v[4], v[5], v[8], v[9], v[10], v[11]
    psi, k2t, k2d, trt, dpl, rb = v[12], v[13], v[14], v[15], v[16], v[20]
    out_rl = v[22]
    s0v, betav, tref = v[18], v[19], v[17]
    npts = S.or_phi.shape[1]
    phi = S.or_phi[b]
    rk2t = S.or_k2t[b]
    rk2d = S.or_k2d[b]
    ra = S.or_a[b]
    rtrt = S.or_trt[b]
    rpsi = S.or_psi[b]
    res0 = S.or_res0[b]
    res1 = S.or_res1[b]
    cw = S.or_cw[b]
    fw = S.or_fw[b]
    lf = S.lf[b]
    rp = P.ph_ring[b]
    jx = S.jx[b]
    for n in range(n_r):
        dl = ri[RI_DECL, n]
        lam = lf[LF_LAM, dl]
        fwhm = lam * lam * P.fwhm_hz[b, n] / C0
        dphi = phi1r[n, dl] - phi0r[n, dl]
        mean = 0.5 * (phi0r[n, dl] + phi1r[n, dl])
        m = math.floor(mean / _TWO_PI + 0.5)
        dT = rf[RF_T_RING, n] - tref[n]
        s_now = s0v[n] * (1.0 + betav[n] * dT)
        g_sh = P.z_opt_self[b, n] * s_now                      # m per absorbed W, static
        pin = out_rl[_op.RL_PIN, n, dl]
        q1 = rf[RF_RHO, n] if P.sh_args_i[b, _op.SH_TWO] != 0 else 0.0
        for lvl in range(2):
            for i in range(npts):
                d = (-1.5 + 3.0 * i / (npts - 1)) * fwhm
                phi[i] = _TWO_PI * m + dpl[n, dl] * d + (-0.5 if lvl == 0 else 0.5) * dphi
                rk2t[i] = k2t[n, dl]
                rk2d[i] = k2d[n, dl]
                ra[i] = a0[n, dl] if lvl == 0 else a1[n, dl]
                rtrt[i] = trt[n, dl]
                rpsi[i] = psi[n, dl]
            _op.ring_rows_nb(phi, rpsi, rk2t, rk2d, ra, rtrt, rb[n], P.use_tcmt != 0, P.m_orders, True, cw, fw,
                             res0 if lvl == 0 else res1)
        best = 0.0
        best_u = 0.0
        i_best = -1
        i_best_u = 0
        for i in range(npts):
            rel = abs(res1[_op.RES_T, i] - res0[_op.RES_T, i])
            slope = pin * ((1.0 - q1) * e0[n, dl] * res0[_op.RES_DLOSS, i]
                           + q1 * e1[n, dl] * res1[_op.RES_DLOSS, i]) * dpl[n, dl]
            if rel > best_u:
                best_u = rel
                i_best_u = i
            if g_sh * slope < 1.0 and rel > best:
                best = rel
                i_best = i
        rf[RF_ORACLE, n] = best
        rf[RF_ORACLE_UNR, n] = best_u
        if P.lock_ref_mode == 1:
            ib = i_best if i_best >= 0 else i_best_u
            d_ib = (-1.5 + 3.0 * ib / (npts - 1)) * fwhm
            # the scan step is 0.02 FWHM (~7 pm in g26): refine the maximum by the vertex of the parabola through
            # ib and its neighbours when they lie on the same side of the stability limit (a lock reference must be
            # finer than the 5 pm lock threshold)
            if ib > 0 and ib < npts - 1:
                ok = True
                if i_best >= 0:
                    for j in (ib - 1, ib + 1):
                        sl = pin * ((1.0 - q1) * e0[n, dl] * res0[_op.RES_DLOSS, j]
                                    + q1 * e1[n, dl] * res1[_op.RES_DLOSS, j]) * dpl[n, dl]
                        if not g_sh * sl < 1.0:
                            ok = False
                if ok:
                    ym = abs(res1[_op.RES_T, ib - 1] - res0[_op.RES_T, ib - 1])
                    y0 = abs(res1[_op.RES_T, ib] - res0[_op.RES_T, ib])
                    yp = abs(res1[_op.RES_T, ib + 1] - res0[_op.RES_T, ib + 1])
                    den = ym - 2.0 * y0 + yp
                    if den < 0.0:
                        off = 0.5 * (ym - yp) / den
                        off = -0.5 if off < -0.5 else (0.5 if off > 0.5 else off)
                        d_ib = d_ib + off * 3.0 / (npts - 1) * fwhm
            # scan point d_ib has the mean phase 2 pi m + dpl d_ib at the line: its exact detuning is x_m - x(phase)
            x_now = lam - (0.5 * (jx[0, n, 0] + jx[0, n, 1]) + s0v[n] * (dT + 0.5 * betav[n] * dT * dT))
            x_m = cold_coordinate(rp[n], x_now, float(m))
            S.m_ref[b, n] = x_m - cold_coordinate(rp[n], x_m, m + dpl[n, dl] * d_ib / _TWO_PI)


# ---------------------------------------------------------------------------------------------- auto policy
@njit(cache=True)
def _auto_h(P, S, b, k, k_end):
    """Largest power-of-two substep (ticks) allowed at tick k by the 07 P0-10 auto rules (h = 1 when any fails):
    the next controller edge, the next command latch, the laser ramp, a quiescent heater and bias DAC path (no
    pending write, FIFO all at the register, slew and settle at their target), no fast kernel, scenario signals
    linear (FOH inputs) or constant (held inputs) over the substep (mrm.scenarios scen_next_break_b), the detuning
    rate |dlambda/dt| h <= sim.max_detuning_step_pm, a relative change of P_abs below au_pabs_tol over the last
    substep (else fall back to fixed), the record windows, the BER windows, the oracle tick and the bias-DAC noise
    redraws (period > 1 tick). These look backward, so two forward guards complete them: (a) a scenario event at k
    (a held signal changed, or an FOH signal's slope changed by more than half, since the previous step) forces
    one tick, so the step that starts a disturbance (laser power or line step, traffic, bias, fault, aggressor
    edge) is resolved tick by tick instead of being interpolated over up to 64 ticks; (b) h at most doubles from
    one step to the next (a fresh transient is entered with 1, 2, 4, ... ticks while the rate criterion learns
    it; the first step after a reset is one tick)."""
    if P.fast_on != 0:
        return 1
    hmax = np.int64(1) << P.au_max_pow
    lim = k_end - k
    e = S.wi[b, WI_EDGE_NEXT] - k
    if e > 0 and e < lim:
        lim = e
    meta = S.pipe_meta[b]
    if meta[1] > 0:
        t = S.pipe_ticks[b, meta[0]] - k
        if t <= 0:
            return 1
        if t < lim:
            lim = t
    if P.reset_protocol == 0 and P.ramp_ticks > 0 and k - S.wi[b, WI_K_RESET] < P.ramp_ticks:
        return 1
    if S.wi[b, WI_ORACLE_NEXT] - k > 0 and S.wi[b, WI_ORACLE_NEXT] - k < lim and P.oracle_on != 0:
        lim = S.wi[b, WI_ORACLE_NEXT] - k
    # (b) growth limit
    g2 = 2 * S.wi[b, WI_LAST_H]
    if g2 < lim:
        lim = g2
    if lim < 2:
        return 1
    ri = S.ri[b]
    r8 = S.r8[b]
    rf = S.rf[b]
    if not _el.actuator_quiescent_row(ri[RI_HE_REG], r8[R8_HE_HAS], S.he_fifo[b], rf[RF_HE_YS], rf[RF_HE_YO],
                                      rf[RF_HE_UT], 1e-9):
        return 1
    if not _el.actuator_quiescent_row(ri[RI_BI_REG], r8[R8_BI_HAS], S.bi_fifo[b], rf[RF_BI_YS], rf[RF_BI_YO],
                                      rf[RF_BI_VT], 1e-9):
        return 1
    rbb = S.rb[b]
    for n in range(P.n_rings):
        if ri[RI_HE_ADD, n] != 0:
            return 1
        # a command latched at this tick changes the register: one tick first
        if rbb[RB_VAL_H, n] and ri[RI_HE_CMD, n] != ri[RI_HE_REG, n]:
            return 1
        if rbb[RB_VAL_B, n] and ri[RI_BI_CMD, n] != ri[RI_BI_REG, n]:
            return 1
    if P.he_ripple != 0:
        return 1
    # bias-DAC noise: never across a redraw when it is redrawn less often than every tick (with a per-tick redraw
    # the draw of the first tick is held over the substep, a stated approximation)
    if P.bi_noise_per > 1:
        nz = _row(P.bi_noise, b)
        on = False
        for n in range(nz.shape[0]):
            if nz[n] != 0.0:
                on = True
        if on:
            nxt = (k // P.bi_noise_per + 1) * P.bi_noise_per - k
            if nxt < lim:
                lim = nxt
    # (a) a scenario event at k: one tick first
    hp = S.wi[b, WI_LAST_H]
    if hp < 1:
        hp = 1
    vp = S.sc_vp[b]
    v0 = S.sc_v0[b]
    v1 = S.sc_v1[b]
    for s in range(P.sc_sig_i.shape[0]):
        if P.sc_sig_i[s, _SI_NC] == 0:
            continue
        off = P.sc_sig_i[s, _SI_OFF]
        foh = s == _SIG_T_BASE or s == _SIG_XPU or s == _SIG_AGG
        for j in range(P.sc_sig_i[s, _sg.SI_WIDTH]):
            a = vp[off + j]
            c = v0[off + j]
            if foh:
                # a kink: the slope per tick changed by more than half (a ramp starts, ends or reverses); the
                # smooth curvature of a sine or rect_sine between its scenario breaks does not count, and changes
                # below 1e-12 of the value per tick (0.3 mK/s at 300 K) are float noise
                s_prev = (c - a) / hp
                s_now = v1[off + j] - c
                ds = abs(s_now - s_prev)
                if ds > 0.5 * max(abs(s_now), abs(s_prev)) and ds > 1e-12 * abs(c):
                    return 1
            elif not (a == c or (a != a and c != c)):
                return 1
    # scenario: FOH signals (base temperature, XPU rise, aggressor power) must stay linear; every other active
    # signal must stay constant (held over the substep)
    bb = b if P.sc_comp_f.shape[0] > 1 else 0
    for s in range(P.sc_sig_i.shape[0]):
        if P.sc_sig_i[s, _SI_NC] == 0:
            continue
        kb = _sg.scen_next_break_b(s, k, P.sc_comp, P.sc_sig_i, P.sc_comp_i, P.sc_comp_f[bb], P.sc_tab[bb])
        if kb - k < lim:
            lim = kb - k
        if lim < 2:
            return 1
    # detuning-rate and P_abs criteria from the previous substep
    rate = S.wf[b, WF_RATE]
    if rate > 0.0:
        hr = P.au_max_det / rate
        if hr < lim:
            lim = np.int64(hr) if hr >= 1.0 else 1
    for n in range(P.n_rings):
        pa = rf[RF_P_ABS, n]
        pp_ = rf[RF_PABS_PREV, n]
        sc = max(abs(pa), abs(pp_))
        if sc > 0.0 and abs(pa - pp_) > P.au_pabs_tol * sc:
            return 1
    # record windows (while the record still records) and BER windows: a substep never straddles one
    for r in range(P.rec_i.shape[0]):
        if S.rec_idx[b, r] >= P.rec_i[r, 4]:
            continue
        dec = P.rec_i[r, 1]
        left = dec - S.rec_cnt[b, r]
        if left < lim:
            lim = left
    win = P.mp_i[MPI_BER_WIN]
    if win > 0:
        left = win - S.m_ai[b, 0, AI_WIN_TICKS]
        if left > 0 and left < lim:
            lim = left
    if lim > hmax:
        lim = hmax
    h = np.int64(1)
    while h * 2 <= lim:
        h *= 2
    if h > 1:
        # held scenario signals must not change over [k, k + h]: compare the values at k + h with those at k
        va = S.sc_va[b]
        for i in range(va.shape[0]):
            va[i] = S.sc_v1[b, i]
        _scen_eval(P, S, b, k + h, va)
        while h > 1:
            ok = True
            for s in range(P.sc_sig_i.shape[0]):
                if P.sc_sig_i[s, _SI_NC] == 0 or s == _SIG_T_BASE or s == _SIG_XPU or s == _SIG_AGG:
                    continue
                off = P.sc_sig_i[s, _SI_OFF]
                for e in range(P.sc_sig_i[s, _sg.SI_WIDTH]):
                    a = S.sc_v0[b, off + e]
                    c = va[off + e]
                    if not (a == c or (a != a and c != c)):
                        ok = False
            if ok:
                break
            h //= 2
            _scen_eval(P, S, b, k + h, va)
    return h


# ---------------------------------------------------------------------------------------------- world block
# One function holds the whole tick (docs/handoff/perf.md P3). A numba function that receives the KParams / KState
# tuples (~270 arrays) and calls any other compiled function increments and decrements the reference count of every
# array in them on entry and exit (numba's reference-count pruning cannot remove the pairs around a call): ~0.4 us
# per call. The tick used to be ~15 such calls (_prepare_interval, _bias_step, _thermal_and_sensors, _heater_step,
# _account, truth_row, eight _declared_line); now the tick's code lives in _world_block, which takes the tuples once
# per block of ticks, and the helpers it calls take arrays. Only the controller edge (release_row, _edge, every
# controller period), the auto-policy decision (_auto_h) and the oracle (every oracle period) still take the tuples.
@njit(cache=True)
def _world_block(step_fn, fast_fn, b, k0, n_ticks, mode, resume, P, S):
    """Advance world b from tick k0 by up to n_ticks ticks (module docstring). Returns the ticks advanced.

    Each step [k, k + h) (h = 1 at the fixed policy; a power of two at the auto policy, :func:`_auto_h`):
    controller edge (events of 07 P0-6 at instant k), then the interval inputs (commands latched at k, scenario at
    k and k + h, laser, ones density, fault mask, bias DAC, NRZ levels, optics inputs at the previous ring
    temperature with the substep constants of the self-heating adapter, bounds, data-activity heat, thermal inputs),
    the heater TCR split and DAC path, the implicit thermal + self-heating step, the sensors of the interval(s),
    and Truth, oracle, metrics and records of the state at k + h."""
    n_r = P.n_rings
    n_l = P.n_lines
    # ---- per-world views, taken once per block
    rf = S.rf[b]
    ri = S.ri[b]
    r8 = S.r8[b]
    rbb = S.rb[b]
    lf = S.lf[b]
    wi = S.wi[b]
    wf = S.wf[b]
    tv = S.tv[b]
    args_f = S.sh_args_f[b]
    args_i = P.sh_args_i[b]
    rg, nb_ = _sh_blocks(args_f, args_i)
    o_pin = args_i[_op.SH_O_PIN]
    p_in = args_f[o_pin:o_pin + n_l]
    p1 = rg[_op.SHR_P1]
    tref = rg[_op.SHR_TREF]
    s0v = rg[_op.SHR_S0]
    betav = rg[_op.SHR_BETA]
    phi0r = nb_[_op.SHN_PHI0R]
    phi1r = nb_[_op.SHN_PHI1R]
    a0 = nb_[_op.SHN_A0]
    a1 = nb_[_op.SHN_A1]
    e0 = nb_[_op.SHN_E0]
    e1 = nb_[_op.SHN_E1]
    k2t = nb_[_op.SHN_K2T]
    k2d = nb_[_op.SHN_K2D]
    trt = nb_[_op.SHN_TRT]
    dpl = nb_[_op.SHN_DPL]
    out_rl, out_r, out_l = _op._sh_outputs(args_f, args_i)
    o_fwhm = _eo(args_i, E_FWHM)
    fwhm_b = P.fwhm_hz[b]
    v0 = S.sc_v0[b]
    v1 = S.sc_v1[b]
    vp = S.sc_vp[b]
    bsc = b if P.sc_comp_f.shape[0] > 1 else 0
    sc_comp_f = P.sc_comp_f[bsc]
    sc_tab = P.sc_tab[bsc]
    sc_keys = P.sc_keys[b]
    sc_cache = S.sc_cache[b]
    o_creset = _sig_off(P, _SIG_CRESET)
    o_en = _sig_off(P, _SIG_EN)
    o_df = _sig_off(P, _SIG_DF)
    o_pd = _sig_off(P, _SIG_PDB)
    o_loss = _sig_off(P, _SIG_LOSS)
    o_dn = _sig_off(P, _SIG_DENS)
    o_td = _sig_off(P, _SIG_TD)
    o_sw = _sig_off(P, _SIG_SWING)
    o_de = _sig_off(P, _SIG_DEN)
    o_bo = _sig_off(P, _SIG_BOFF)
    o_bv = _sig_off(P, _SIG_BIAS)
    o_fd = _sig_off(P, _SIG_DROP)
    o_fs = _sig_off(P, _SIG_STUCK)
    o_fa = _sig_off(P, _SIG_SAT)
    o_fg = _sig_off(P, _SIG_GLITCH)
    o_ag = _sig_off(P, _SIG_AGG)
    o_tb = _sig_off(P, _SIG_T_BASE)
    o_xpu = _sig_off(P, _SIG_XPU)
    swing = _row(P.drv_swing, b)
    tin = P.las_input_t[b]
    rp = P.ph_ring[b]
    tb = b if P.jt_dn.shape[0] > 1 else 0
    jt_dn = P.jt_dn[tb]
    jt_al = P.jt_al[tb]
    jt_c = P.jt_c[tb]
    jt_rs = P.jt_rs[tb]
    jt_cum = P.jt_cum[tb]
    rp_fill = rp[:, _ph.RP_FILL]
    rp_len = rp[:, _ph.RP_LEN]
    ph_ws = np.empty((6, n_r))                     # scratch of the phase model (no allocation per tick)
    jx = S.jx[b]
    u0 = S.th_u0[b]
    u1 = S.th_u1[b]
    x_th = S.th_x[b]
    t_obs = S.th_t_obs[b]
    th_g = S.th_g[b]
    th_a = _row(P.th_a, b)
    th_c0 = _row(P.th_c0, b)
    th_c1 = _row(P.th_c1, b)
    # thermal pass tables (bit-identical restructuring, docs/handoff/perf.md P8): structure now, gains per substep
    # power of two on first use (P may change between blocks, e.g. Plant.settle swaps the coefficient tables)
    th_idx_exo, th_idx_cross, th_idx_ax, th_unit, th_hs_lo, th_hs_hi = _th_struct(
        P.th_lag_src, P.th_lag_kind, P.th_obs_ptr, P.th_heater_obs, P.th_heater_src, P.th_src_scale)
    th_npow = th_a.shape[0]
    th_w = np.empty((th_npow, th_a.shape[1]))
    th_gobs = np.empty((th_npow, P.th_obs_ptr.shape[0] - 1))
    th_gsplit = np.empty((th_npow, n_r))
    th_ok = np.zeros(th_npow, np.bool_)
    th_pre = np.empty(n_r)
    th_hln, th_hlo, th_hself = _heater_split_tables(P.th_obs_ptr, P.th_lag_src, P.th_heater_obs, P.th_heater_src)
    fact = S.fact[b]
    # bias DAC
    bi_code_max = (1 << P.bi_bits) - 1
    bi_vmin = _row(P.bi_vmin, b)
    bi_vmax = _row(P.bi_vmax, b)
    bi_bit_err = _row(P.bi_bit_err, b)
    bi_bow = _row(P.bi_bow, b)
    bi_noise = _row(P.bi_noise, b)
    bi_key = P.bi_key[b]
    bi_lat = _row(P.bi_lat, b)
    bi_fifo = S.bi_fifo[b]
    bi_slew = _row(P.bi_slew, b)
    bi_settle = _row(P.bi_settle, b)
    # heater DAC
    he_full = (1 << P.he_bits) - 1
    he_law = _row(P.he_law, b)
    he_bit_err = _row(P.he_bit_err, b)
    he_bow = _row(P.he_bow, b)
    he_lat = _row(P.he_lat, b)
    he_fifo = S.he_fifo[b]
    he_slew = _row(P.he_slew, b)
    he_settle = _row(P.he_settle, b)
    # sensors
    assign = ri[RI_ASSIGN]
    r2l = P.ring_to_line
    link_t = P.dp_link_t[b]
    oma_sens = P.dp_oma_sens[b]
    nts = S.ts_in.shape[1]
    ts_in = S.ts_in[b]
    mo_lat = _row(P.mo_lat, b)
    mo_pw = P.mo_pw[b]
    rec_db = False                                 # a record reads er_db / il_db (else Truth skips their log10)
    for r in range(P.rec_i.shape[0]):
        if P.rec_i[r, 0] == TR_ER or P.rec_i[r, 0] == TR_IL:
            rec_db = True
    k = k0
    k_end = k0 + n_ticks
    first = True
    while k < k_end:
        # --- controller edge (07 P0-6 event order steps 1-4; takes the tuples, once per controller period)
        if first and resume != 0:
            # Python path: obs of edge k was packed by the previous call and the Python step wrote S.act
            _push_act(P, S, b, k)
            _advance_edge(P, S, b)
        else:
            if P.fast_on != 0:
                _fast_tick(fast_fn, P, S, b, k)
            if k == wi[WI_EDGE_NEXT]:
                if _edge(step_fn, P, S, b, k, mode):
                    return k - k0
        first = False
        # --- commands latched at k (07 P0-6 step 4)
        for n in range(n_r):
            rbb[RB_VAL_H, n] = False
            rbb[RB_VAL_B, n] = False
            rbb[RB_POPPED, n] = False
        if pipe_pop_row(S.pipe_vals[b], S.pipe_ticks[b], S.pipe_meta[b], k, S.act_tmp[b]):
            apply_act_row(S.act_tmp[b], P.act_map, ri[RI_HE_CMD], rbb[RB_VAL_H], ri[RI_BI_CMD], rbb[RB_VAL_B],
                          ri[RI_ASSIGN], ri[RI_BS_THR], ri[RI_PGA], S.tune[b])
            for n in range(n_r):
                rbb[RB_POPPED, n] = rbb[RB_VAL_H, n]
        # --- scenario at k (v0, carried from the previous step) and k + 1 (v1); vp keeps the previous step's v0
        #     (the auto policy detects scenario events with it)
        for i in range(v0.shape[0]):
            vp[i] = v0[i]
            v0[i] = v1[i]
        _sg.scen_eval_b(k + 1, P.sc_comp, P.sc_sig_i, P.sc_sig_f, P.sc_def, P.sc_comp_i, sc_comp_f, sc_tab, sc_keys,
                        sc_cache, v1)
        # controller reset fault (rising edge, latched until the next controller edge)
        cr = v0[o_creset] >= 0.5
        if cr and wi[WI_CTRL_RESET_PREV] == 0:
            wi[WI_CTRL_RESET] = 1
        wi[WI_CTRL_RESET_PREV] = 1 if cr else 0
        # --- laser lines (07 P1-12): enable is a power factor, dfreq and power steps from the scenario, laser.tune
        for l in range(n_l):
            ef = v0[o_en + l]
            ef = 0.0 if ef < 0.0 else (1.0 if ef > 1.0 else ef)
            lf[LF_EN_B, l] = 1.0 if ef > 0.0 else 0.0
            lf[LF_DF, l] = v0[o_df + l]
            pdb = v0[o_pd + l]
            if not (lf[LF_PDB_KEY, l] == pdb):          # 10^(dB/10) only when the scenario value changes (exact)
                lf[LF_PDB_KEY, l] = pdb
                lf[LF_PDB_POW, l] = 10.0 ** (pdb / 10.0)
            lf[LF_PS, l] = lf[LF_PDB_POW, l] * ef
            lf[LF_TUNE_HZ, l] = S.tune[b, l] * P.las_tune_lsb
        en_b = lf[LF_EN_B] > 0.5
        laser_tick_row(k, P.dt, P.las_tcomp, P.las_grid_s, P.las_depth, P.las_f0[b], P.las_p0[b], P.las_frate[b],
                       P.las_prate[b], P.las_frw[b], P.las_prw[b], P.las_rho_f[b], P.las_rho_p[b], P.las_module,
                       P.las_nmod, P.las_rin[b], en_b, lf[LF_DF], lf[LF_PS], lf[LF_TUNE_HZ], P.las_key_drift[b],
                       P.las_key_rin[b], S.las_cache_j[b], S.las_wlo[b], S.las_whi[b], lf[LF_F], lf[LF_P])
        ramp = 1.0
        if P.reset_protocol == 0 and P.ramp_ticks > 0:
            xr = (k - wi[WI_K_RESET] + 1) / P.ramp_ticks
            ramp = xr if xr < 1.0 else 1.0
            if ramp < 0.0:
                ramp = 0.0
        wf[WF_RAMP] = ramp
        ldb = v0[o_loss]
        if not (wf[WF_LOSS_KEY] == ldb):
            wf[WF_LOSS_KEY] = ldb
            wf[WF_LOSS_VAL] = 10.0 ** (-ldb / 10.0)
        loss = wf[WF_LOSS_VAL]
        for l in range(n_l):
            lf[LF_PCHIP, l] = lf[LF_P, l] * tin * loss * ramp
            lf[LF_LAM, l] = C0 / lf[LF_F, l]
        # --- data: ones density (07 P0-11), driver enable, transition density, swing scale, bias offset
        for n in range(n_r):
            rf[RF_DENS, n] = v0[o_dn + n]
            rf[RF_TD, n] = v0[o_td + n]
            rf[RF_SWING, n] = v0[o_sw + n]
            on = v0[o_de + n] >= 0.5
            rbb[RB_DP_ON, n] = on
            rf[RF_EN, n] = 1.0 if on else 0.0
            rf[RF_BOFF, n] = v0[o_bo + n]
            rf[RF_SWING_V, n] = swing[_el._ix(swing, n)] * v0[o_sw + n]
        pattern_tick_row(k, P.dp_mode, P.dp_bpt_i, P.dp_bpt_f, rf[RF_DENS], rbb[RB_DP_ON], P.dp_words, P.dp_pre,
                         P.dp_period, P.dp_key_pat[b], ri[RI_DP_BITPOS], ri[RI_DP_N1], ri[RI_DP_NB], rf[RF_RHO])
        for n in range(n_r):
            ri[RI_DP_ONES, n] += ri[RI_DP_N1, n]
            ri[RI_DP_BITS, n] += ri[RI_DP_NB, n]
        # --- fault mask (07 P1-10) and the dac_glitch code offset
        fault_mask_row(v0[o_fd:o_fd + n_r], v0[o_fs:o_fs + n_r], v0[o_fa:o_fa + n_r], v0[o_fg:o_fg + n_r],
                       r8[R8_FAULT])
        for n in range(n_r):
            ri[RI_HE_ADD, n] = np.int64(np.rint(v0[o_fg + n])) if (r8[R8_FAULT, n] & FAULT_DAC_GLITCH) != 0 else 0
        # --- bias DAC (mrm.plant.electrical.bias_bank_step_row; the fast_kernel offset before the latency FIFO;
        #     the DAC noise is the counter-RNG normal of stream dac_bias at (ring, tick k, slot 0), drawn on the
        #     redraw ticks only): scenario-owned bias (absolute) or the controller's command
        if P.bias_ctrl == 0:
            for n in range(n_r):
                lo = bi_vmin[_el._ix(bi_vmin, n)]
                lsb = (bi_vmax[_el._ix(bi_vmax, n)] - lo) / bi_code_max
                vs = v0[o_bv + n]
                if not math.isfinite(vs):
                    vs = lo
                cf = np.rint((vs - lo) / lsb)
                cf = 0.0 if cf < 0.0 else (float(bi_code_max) if cf > bi_code_max else cf)
                ri[RI_BI_CMD, n] = np.int64(cf)
                rbb[RB_VAL_B, n] = True
        _el.latch_row(ri[RI_BI_CMD], rbb[RB_VAL_B], ri[RI_BI_REG], ri[RI_BI_PEND], r8[R8_BI_HAS], ri[RI_BI_LAST], k,
                      P.bi_min_upd, 0, bi_code_max)
        for n in range(n_r):
            cb = ri[RI_BI_REG, n]
            if P.fast_on != 0:
                cb += fact[n_r + n]
                if cb < 0:
                    cb = 0
                elif cb > bi_code_max:
                    cb = bi_code_max
            ri[RI_BI_EFF, n] = cb
        _el.fifo_row(ri[RI_BI_EFF], bi_fifo, k, bi_lat, ri[RI_BI_CODE])
        for n in range(n_r):
            # the DAC transfer of a code is a constant: recomputed only when the code changes (exact)
            cb = ri[RI_BI_CODE, n]
            if cb != ri[RI_BI_TCODE, n]:
                ri[RI_BI_TCODE, n] = cb
                rf[RF_BI_VT0, n] = _el.bias_target_one(cb, P.bi_bits, bi_bit_err[_el._ix(bi_bit_err, n)],
                                                       bi_bow[_el._ix(bi_bow, n)], bi_vmin[_el._ix(bi_vmin, n)],
                                                       bi_vmax[_el._ix(bi_vmax, n)])
        if k % P.bi_noise_per == 0:
            for n in range(n_r):
                lsb = (bi_vmax[_el._ix(bi_vmax, n)] - bi_vmin[_el._ix(bi_vmin, n)]) / bi_code_max
                amp = bi_noise[_el._ix(bi_noise, n)]
                rf[RF_BI_HOLD, n] = amp * lsb * _rng.normal(bi_key, n, k, 0) if amp != 0.0 else 0.0
        for n in range(n_r):
            rf[RF_BI_VT, n] = rf[RF_BI_VT0, n] + rf[RF_BI_HOLD, n]
        _el.slew_settle_row(rf[RF_BI_VT], rf[RF_BI_YS], rf[RF_BI_YO], bi_slew, bi_settle, P.dt, rf[RF_BI_VAVG])
        for n in range(n_r):
            off = rf[RF_BOFF, n]
            rf[RF_BI_VAVG, n] = rf[RF_BI_VAVG, n] + off
            rf[RF_BI_VEND, n] = rf[RF_BI_YO, n] + off
        # --- NRZ levels (07 P1-7): level 1 = the higher reverse voltage (integrator assumption)
        for n in range(n_r):
            r8[R8_LFLAGS, n] = 0
        _el.nrz_levels_row(rf[RF_BI_VAVG], rf[RF_SWING_V], _row(P.drv_level, b), _el.JUNCTION_GRID_V_MIN,
                           _el.JUNCTION_GRID_V_MAX, rf[RF_V_LO], rf[RF_V_HI], r8[R8_LFLAGS])
        # --- optics inputs at the previous ring temperature, written straight into the self-heating rows
        for n in range(n_r):
            r8[R8_JFLAGS, n] = 0
        _ph.ring_optics_levels_ws(rp, lf[LF_LAM], rf[RF_T_RING], rf[RF_T_HEATER], rf[RF_V_LO], rf[RF_V_HI],
                                  P.jt_grid, jt_dn, jt_al, jt_c, jt_rs, phi0r, phi1r, a0, a1, e0, e1, trt, jx[0],
                                  jx[1], jx[2], r8[R8_JFLAGS], ph_ws)
        # --- optics inputs: K^2 at the lines (and the Truth FWHM per line) only when the line wavelengths moved
        #     (exact: functions of lam and constants only); reference temperature and slope rows
        lines_new = eng_lines_changed(args_f, args_i, lf[LF_LAM])
        if lines_new:
            _ph.coupling_row(rp, lf[LF_LAM], k2t, k2d)
            for n in range(n_r):
                for l in range(n_l):
                    args_f[o_fwhm + n * n_l + l] = lf[LF_LAM, l] * lf[LF_LAM, l] * fwhm_b[n] / C0
        for l in range(n_l):
            p_in[l] = lf[LF_PCHIP, l]
        for n in range(n_r):
            p1[n] = rf[RF_RHO, n]
            rf[RF_RS, n] = 0.5 * (jx[2, n, 0] + jx[2, n, 1])
            tref[n] = rf[RF_T_RING, n]
            s_l, b_l = _ph.local_thermo_optic(rf[RF_T_RING, n], rp[n, _ph.RP_S0], rp[n, _ph.RP_BETA],
                                              rp[n, _ph.RP_T0])
            s0v[n] = s_l
            betav[n] = b_l
            for l in range(n_l):
                dpl[n, l] = dphi_dlamres(lf[LF_LAM, l], jx[0, n, 0], trt[n, l])
        swing_capacitance_row(rf[RF_V_LO], rf[RF_V_HI], P.jt_grid, jt_c, jt_cum, rp_fill, rp_len, rf[RF_CJ])
        # --- optics inputs: the substep caches of the self-heating adapter
        eng_precompute(args_f, args_i)
        # --- optics inputs: the bounds of the self-heating solve
        _op.sh_bounds(args_f, args_i, P.dt_max_k, rf[RF_PMAX], rf[RF_D2P], rf[RF_DPDT])
        # --- data-activity heat (07 P1-8)
        for n in range(n_r):
            rf[RF_SWING_V, n] = rf[RF_V_HI, n] - rf[RF_V_LO, n]
        _el.data_activity_heat_row(rf[RF_CJ], rf[RF_SWING_V], P.dp_bitrate[b], rf[RF_TD], rf[RF_EN], rf[RF_RS],
                                   P.drv_r_out[b:b + 1], P.drv_p_static[b:b + 1], P.drv_e_tr[b:b + 1], 0, rf[RF_PJ],
                                   rf[RF_PD])
        # --- thermal inputs (held: junction, driver, previous heater power, previous absorbed power; FOH: aggressor)
        for s in range(u0.shape[0]):
            u0[s] = 0.0
            u1[s] = 0.0
        for n in range(n_r):
            sj = P.th_junction_src[n]
            if sj >= 0:
                u0[sj] = rf[RF_PJ, n]
                u1[sj] = rf[RF_PJ, n]
            sd = P.th_driver_src[n]
            if sd >= 0:
                u0[sd] = rf[RF_PD, n]
                u1[sd] = rf[RF_PD, n]
            sh = P.th_heater_src[n]
            u0[sh] = rf[RF_HE_P, n]
            u1[sh] = rf[RF_HE_P, n]
            sw = P.th_wg_src[n]
            if sw >= 0:
                u0[sw] = rf[RF_P_PREV, n]
                u1[sw] = rf[RF_P_PREV, n]
        for i in range(P.th_agg_src.shape[0]):
            u0[P.th_agg_src[i]] = v0[o_ag + i]
        # --- substep length h (07 P0-10): 1 at the fixed policy; the auto rules otherwise
        h = np.int64(1)
        if P.policy == POLICY_AUTO:
            h = _auto_h(P, S, b, k, k_end)
        pw2 = 0
        n_outer = P.sh_n_outer
        if h > 1:
            # one multi-tick substep: the interval inputs above were prepared at tick k as at the fixed policy (the
            # scenario's FOH signals then span [k, k + h]); the thermal step is exact for h (coefficient table of
            # the power of two); held inputs keep their tick-k values; the DAC paths are quiescent, so their
            # per-tick steps reduce to the settle-filter advance; sensors run per tick with the monitor input
            # interpolated
            while (np.int64(1) << pw2) < h:
                pw2 += 1
            _sg.scen_eval_b(k + h, P.sc_comp, P.sc_sig_i, P.sc_sig_f, P.sc_def, P.sc_comp_i, sc_comp_f, sc_tab,
                            sc_keys, sc_cache, v1)
            n_outer = P.au_n_outer
        ta = th_a[pw2]
        tc0 = th_c0[pw2]
        tc1 = th_c1[pw2]
        if not th_ok[pw2]:
            _th_gains(tc0, tc1, P.th_lag_src, P.th_lag_kind, P.th_obs_ptr, P.th_heater_obs, P.th_heater_src,
                      P.th_src_scale, th_w[pw2], th_gobs[pw2], th_gsplit[pw2])
            th_ok[pw2] = True
        for n in range(n_r):
            rf[RF_LAMRES_PREV, n] = tv[TR_LAM_RES, n]
            rf[RF_PABS_PREV, n] = rf[RF_P_ABS, n]
        for i in range(P.th_agg_src.shape[0]):
            u1[P.th_agg_src[i]] = v1[o_ag + i]
        t_base_end = v1[o_tb] + v1[o_xpu]
        wf[WF_T_BASE1] = t_base_end
        # --- heater: implicit TCR split on the heater-resistor observer (07 P1-2), then the DAC path
        _heater_split_all(x_th, ta, tc0, tc1, P.th_lag_src, P.th_obs_ptr, P.th_heater_obs, P.th_heater_src, th_hs_lo,
                          th_hs_hi, th_unit, u0, u1, P.th_src_scale, th_hln, th_hlo, th_hself, th_pre)
        for n in range(n_r):
            rf[RF_HE_TPRE, n] = t_base_end + th_pre[n]
            rf[RF_HE_G, n] = th_gsplit[pw2, n]
        # --- heater DAC path (mrm.plant.electrical.heater_bank_step_row, with the fast_kernel offset added to the
        #     latched register before the latency FIFO, clipped to [code_min, code_max] (07 P0-7), and the
        #     dac_glitch code after the FIFO (07 P1-10))
        _el.latch_row(ri[RI_HE_CMD], rbb[RB_VAL_H], ri[RI_HE_REG], ri[RI_HE_PEND], r8[R8_HE_HAS], ri[RI_HE_LAST], k,
                      P.he_min_upd, P.he_lo, P.he_hi)
        for n in range(n_r):
            ch = ri[RI_HE_REG, n]
            if P.fast_on != 0:
                ch += fact[n]
                if ch < P.he_lo:
                    ch = P.he_lo
                elif ch > P.he_hi:
                    ch = P.he_hi
            ri[RI_HE_EFF, n] = ch
        _el.fifo_row(ri[RI_HE_EFF], he_fifo, k, he_lat, ri[RI_HE_CODE])
        for n in range(n_r):
            add = ri[RI_HE_ADD, n]
            if add != 0:
                ch = ri[RI_HE_CODE, n] + add
                ri[RI_HE_CODE, n] = 0 if ch < 0 else (he_full if ch > he_full else ch)
            # the DAC transfer of a code is a constant (without the PWM ripple): recomputed only when the code
            # changes (exact)
            ch = ri[RI_HE_CODE, n]
            if P.he_ripple != 0 or ch != ri[RI_HE_TCODE, n]:
                ri[RI_HE_TCODE, n] = ch
                ut, rip = _el.heater_target_one(ch, he_law, P.he_bits, P.he_pwm_bits,
                                                he_bit_err[_el._ix(he_bit_err, n)], he_bow[_el._ix(he_bow, n)],
                                                P.he_ripple, k, P.he_dt_clk)
                rf[RF_HE_UT, n] = ut
                rf[RF_HE_RIP, n] = rip
        _el.slew_settle_row(rf[RF_HE_UT], rf[RF_HE_YS], rf[RF_HE_YO], he_slew, he_settle, P.dt, rf[RF_HE_UAVG])
        for n in range(n_r):
            rf[RF_HE_UAVG, n] = rf[RF_HE_UAVG, n] + rf[RF_HE_RIP, n]
            r8[R8_HE_FLAGS, n] = 0
        _el.heater_power_row(rf[RF_HE_UAVG], he_law, rf[RF_HE_TPRE], rf[RF_HE_G], _row(P.he_r0, b),
                             _row(P.he_alpha, b), _row(P.he_t0, b), _row(P.he_vfs, b), _row(P.he_ifs, b),
                             _row(P.he_pfs, b), _row(P.he_vcomp, b), rf[RF_HE_P], rf[RF_HE_V], rf[RF_HE_I],
                             rf[RF_HE_R], r8[R8_HE_FLAGS])
        wf[WF_HE_SCALE] = _el.supply_limit_row(rf[RF_HE_P], rf[RF_HE_V], rf[RF_HE_I], _row(P.he_pmax, b),
                                               _row(P.he_imax, b), r8[R8_HE_FLAGS])
        for n in range(n_r):
            sh = P.th_heater_src[n]
            u0[sh] = rf[RF_HE_P, n]
            u1[sh] = rf[RF_HE_P, n]
        # --- implicit thermal + self-heating step (07 P0-1); the optics walk runs inside (commit_fn)
        if P.self_heating != 0:
            nf = _implicit_step_eng(x_th, ta, tc0, tc1, P.th_lag_src, P.th_lag_obs, P.th_lag_kind, P.th_opt_lags,
                                    P.th_obs_ptr, P.th_src_ring, P.th_ring_obs, P.bus_order, u0, u1, P.th_src_scale,
                                    t_base_end, rf[RF_P_PREV], rf[RF_T_BRANCH], args_f, args_i, rf[RF_PMAX],
                                    rf[RF_D2P], P.sh_max_iter, P.sh_tol, n_outer, th_g, t_obs, rf[RF_P_NEW],
                                    rf[RF_T_NEW], ri[RI_N_ITER], th_w[pw2], th_gobs[pw2], th_idx_exo,
                                    th_idx_cross, th_idx_ax, th_unit)
            wi[WI_N_FAIL] += nf
            for n in range(n_r):
                wi[WI_SH_ITER] += ri[RI_N_ITER, n]
                rf[RF_P_PREV, n] = rf[RF_P_NEW, n]
                rf[RF_T_BRANCH, n] = rf[RF_T_NEW, n]
                rf[RF_T_RING, n] = rf[RF_T_NEW, n]
        else:
            vv = _op.selfheat_views(args_f, args_i)
            _op.bus_propagate_nb(vv[0], P.bus_order, vv[1], vv[2][0], vv[3], vv[4], vv[5], vv[8], vv[9], vv[10],
                                 vv[11], vv[12], vv[13], vv[14], vv[15], vv[20], vv[16], args_i[_op.SH_TWO] != 0,
                                 P.use_tcmt != 0, P.m_orders, False, vv[22], vv[23], vv[24], vv[21], vv[25], vv[26])
            for n in range(n_r):
                sw = P.th_wg_src[n]
                if sw >= 0:
                    u0[sw] = 0.0
                    u1[sw] = 0.0
            _th.thermal_advance_b(x_th, ta, tc0, tc1, P.th_lag_src, P.th_obs_ptr, u0, u1, P.th_src_scale, t_base_end,
                                  t_obs)
            for n in range(n_r):
                rf[RF_T_RING, n] = t_obs[P.th_ring_obs[n]]
                rf[RF_T_BRANCH, n] = rf[RF_T_RING, n]
                rf[RF_P_PREV, n] = 0.0
        for n in range(n_r):
            rf[RF_T_HEATER, n] = t_obs[P.th_heater_obs[n]]
        # --- sensor inputs of the interval: PD power, bus power at the ring, PD temperature, OMA of the declared line
        #     and its BER (computed once here for the error counter and Truth: the same expression both used)
        for n in range(n_r):
            rf[RF_P_PD, n] = out_r[_op.R_DROP_SUM, n]
            pb = 0.0
            for l in range(n_l):
                pb += out_rl[_op.RL_PIN, n, l]
            rf[RF_P_BG, n] = pb
            po = P.th_pd_obs[n]
            rf[RF_T_PD, n] = t_obs[po] if po >= 0 else rf[RF_T_RING, n]
            # the PD dark current of the interval, once for the monitor and Truth (the expression both used)
            rf[RF_I_DARK, n] = dark_current_a(mo_pw[PW_DARK_I0], rf[RF_T_PD, n], mo_pw[PW_DARK_T0], mo_pw[PW_DARK_EA])
            dl = _decl(assign, r2l, n_l, n)
            ri[RI_DECL, n] = dl
            oma = abs(out_rl[_op.RL_POUT1, n, dl] - out_rl[_op.RL_POUT0, n, dl])
            rf[RF_OMA, n] = oma
            rf[RF_BER, n] = ber_from_oma(oma * link_t, oma_sens[n], P.dp_q_ref, P.dp_ber_floor)
        for i in range(nts):
            ts_in[i] = t_obs[P.th_temp_obs[i]]
        # --- sensors of the interval(s): monitor, error counter, temperature sensors, heater readback
        for j in range(h):
            kk = k + j
            if j > 0:
                # multi-tick substep (07 P0-10 auto): the cheap per-tick parts; optics inputs interpolated linearly
                # between the previous and this substep's end values (h ticks of drift below max_detuning_step)
                fr_ = (j + 1.0) / h
                for n in range(n_r):
                    rf[RF_P_PD, n] = rf[RF_PPD_PREV, n] + fr_ * (out_r[_op.R_DROP_SUM, n] - rf[RF_PPD_PREV, n])
                pattern_tick_row(kk, P.dp_mode, P.dp_bpt_i, P.dp_bpt_f, rf[RF_DENS], rbb[RB_DP_ON], P.dp_words,
                                 P.dp_pre, P.dp_period, P.dp_key_pat[b], ri[RI_DP_BITPOS], ri[RI_DP_N1],
                                 ri[RI_DP_NB], rf[RF_RHO])
                for n in range(n_r):
                    ri[RI_DP_ONES, n] += ri[RI_DP_N1, n]
                    ri[RI_DP_BITS, n] += ri[RI_DP_NB, n]
            elif h > 1:
                fr_ = 1.0 / h
                for n in range(n_r):
                    rf[RF_P_PD, n] = rf[RF_PPD_PREV, n] + fr_ * (out_r[_op.R_DROP_SUM, n] - rf[RF_PPD_PREV, n])
            monitor_tick_row(kk, rf[RF_P_PD], rf[RF_P_BG], rf[RF_T_PD], rf[RF_T_PD], ri[RI_PGA], r8[R8_FAULT],
                             P.mo_pr[b], P.mo_pw[b], mo_lat, P.mo_phase, P.mo_pga, P.mo_fl[b], P.mo_afe, P.mo_win,
                             P.mo_period, P.mo_cmax, P.mo_shot, P.mo_stuck, P.mo_slot, P.mo_key_adc[b],
                             P.mo_key_shot[b], P.mo_key_dnl[b], S.mo_sf[b], S.mo_si[b], S.mo_fx[b], S.mo_fc[b],
                             S.mo_fv[b], S.mo_fr[b], S.mo_fm[b], r8[R8_MO_SAMPLED], rf[RF_I_DARK])
            for n in range(n_r):
                if r8[R8_MO_SAMPLED, n] == 2:
                    wi[WI_MON_FIFO_LOST] += 1
            errors_tick_ber_row(kk, rf[RF_BER], ri[RI_DP_NB], P.dp_err_period, P.dp_err_lat[b], P.dp_key_dp[b],
                                rf[RF_DP_LAM], S.dp_fv[b], S.dp_fr[b], S.dp_fm[b], rbb[RB_ERR_NEW])
            if nts > 0:
                temp_tick_row(kk, ts_in, P.ts_tp[b], P.ts_period, P.ts_lat[b], P.ts_cmax, P.ts_fl[b], P.ts_key[b],
                              S.ts_st[b], S.ts_fx[b], S.ts_fc[b], S.ts_primed[b], S.ts_fv[b], S.ts_fr[b], S.ts_fm[b],
                              S.ts_sampled[b])
                for i in range(nts):
                    if S.ts_sampled[b, i] == 2:
                        wi[WI_TEMP_LOST] += 1
            if P.rb_on != 0:
                readback_tick_row(kk, rf[RF_HE_V], rf[RF_HE_I], P.rb_lsb_v[b], P.rb_lsb_i[b], P.rb_noise[b],
                                  P.rb_period, P.rb_lat[b], P.rb_cmax, P.rb_avg, P.rb_key[b], S.rb_st[b], S.rb_fv[b],
                                  S.rb_fr[b], S.rb_fm[b], r8[R8_RB_SAMPLED])
                for n in range(n_r):
                    if r8[R8_RB_SAMPLED, n] == 2:
                        wi[WI_RB_LOST] += 1
            for n in range(n_r):
                rf[RF_BITS, n] += ri[RI_DP_NB, n]
        for n in range(n_r):
            rf[RF_PPD_PREV, n] = out_r[_op.R_DROP_SUM, n]
        if h > 1:
            # DAC settle filters over the skipped ticks (quiescent: the targets are constant)
            _el.settle_advance_row(rf[RF_HE_UT], rf[RF_HE_YS], rf[RF_HE_YO], he_settle, P.dt, h - 1)
            _el.settle_advance_row(rf[RF_BI_VT], rf[RF_BI_YS], rf[RF_BI_YO], bi_settle, P.dt, h - 1)
        # --- Truth of the state at k + h
        last_step = k + h >= k_end or (mode == MODE_PYTHON and k + h >= wi[WI_EDGE_NEXT])
        _truth_a(n_r, n_l, rf, ri, rbb, tv, lf, S.lt[b], t_base_end, args_f, args_i, P.mo_pw[b], P.mo_pr[b], rp, jx,
                 fwhm_b, P.wrong_window, P.assign_mode, rec_db or last_step, True)
        # --- oracle (on its period; takes the tuples)
        if P.oracle_on != 0 and k + h > wi[WI_ORACLE_NEXT]:
            _oracle(P, S, b)
            wi[WI_ORACLE_NEXT] = k + P.oracle_period
            for n in range(n_r):
                tv[TR_ORACLE, n] = rf[RF_ORACLE, n]
        # --- metrics and records
        metrics_tick_row(k, h, P.mp_f, P.mp_i, rf[RF_DET], S.m_ref[b], ri[RI_ORDER], rbb[RB_WRONG], tv[TR_RELOMA],
                         rf[RF_ORACLE], rf[RF_ORACLE_UNR], rf[RF_HE_P], rf[RF_P_ABS], ri[RI_HE_REG], rbb[RB_POPPED],
                         rf[RF_BER], rf[RF_BITS], S.m_af[b], S.m_ai[b], S.m_hist[b], S.m_runhist[b])
        for n in range(n_r):
            rf[RF_BITS, n] = 0.0
        if P.rec_i.shape[0] > 0:
            record_tick_row(h, tv, P.rec_i, S.rec_acc[b], S.rec_cnt[b], S.rec_idx[b], S.trace[b])
        # --- step bookkeeping (auto policy: the detuning rate of this step)
        if P.policy == POLICY_AUTO:
            rate = 0.0
            for n in range(n_r):
                d = abs(tv[TR_LAM_RES, n] - rf[RF_LAMRES_PREV, n]) / h
                if d > rate:
                    rate = d
            wf[WF_RATE] = rate
            if h > 1 and h > wi[WI_SUB_H_MAX]:
                wi[WI_SUB_H_MAX] = h
        wi[WI_N_SUBSTEPS] += 1
        wi[WI_N_TICKS] += h
        wi[WI_LAST_H] = h
        k += h
    return k - k0


def _run_chunk_py(step_fn, fast_fn, P, S, k0, n_ticks, mode, resume, b0, b1, done):
    """Worlds b0 <= b < b1, each running its whole block (no barrier per tick; no reduction across b, 07 P0-9)."""
    for b in range(b0, b1):
        done[b] = _world_block(step_fn, fast_fn, b, k0, n_ticks, mode, resume, P, S)


_RUN_CHUNKS: dict = {}
_POOL = None


def _threads() -> int:
    from numba import config as _cfg
    return max(1, int(getattr(_cfg, "NUMBA_NUM_THREADS", 1) or 1))


def get_run_block(P, S):
    """The compiled ``run_block(step_fn, fast_fn, P, S, k0, n_ticks, mode, resume, done, n_threads=None)``.

    Parallel over B only (07 P0-9). The worlds are split into contiguous chunks; each chunk runs in a compiled
    ``nogil`` function (explicit signature, controller functions typed :data:`STEP_FT`, ``cache=True``) on a thread
    pool, so results do not depend on the thread count (every draw is a pure function of (seed_b, stream, ring, tick,
    slot)). Deviation from the letter of 07 P0-9 (``@njit(parallel=True)`` with ``prange``): a numba parallel region
    cannot take the kernel's namedtuple arguments (178 and 90 arrays; the parfor tuple limit is 100, nested tuples
    are refused, and with the limit raised a trivial prange over a 180-array tuple did not finish compiling in 9 min),
    so the prange over B is a thread pool over B. Compiled once per argument types and process (a new dtype or ndim,
    never a new size or value); disk-cached across processes. Under NUMBA_DISABLE_JIT=1 the chunk function is the
    Python function and the call is sequential."""
    from numba.core.config import DISABLE_JIT
    if DISABLE_JIT:
        chunk = _run_chunk_py
    else:
        sig = types.void(STEP_FT, STEP_FT, nb.typeof(P), nb.typeof(S), types.int64, types.int64, types.int64,
                         types.int64, types.int64, types.int64, types.int64[::1])
        chunk = _RUN_CHUNKS.get(sig)
        if chunk is None:
            chunk = njit(sig, nogil=True, cache=True)(_run_chunk_py)
            _RUN_CHUNKS[sig] = chunk

    def run_block(step_fn, fast_fn, P, S, k0, n_ticks, mode, resume, done, n_threads=None):
        global _POOL
        B = done.shape[0]
        nt = min(B, _threads() if n_threads is None else max(1, int(n_threads)))
        if nt <= 1 or DISABLE_JIT:
            chunk(step_fn, fast_fn, P, S, np.int64(k0), np.int64(n_ticks), np.int64(mode), np.int64(resume),
                  np.int64(0), np.int64(B), done)
            return
        if _POOL is None or _POOL._max_workers < nt:
            from concurrent.futures import ThreadPoolExecutor
            _POOL = ThreadPoolExecutor(max_workers=max(nt, _threads()))
        edges = np.linspace(0, B, nt + 1).round().astype(np.int64)
        futs = [_POOL.submit(chunk, step_fn, fast_fn, P, S, np.int64(k0), np.int64(n_ticks), np.int64(mode),
                             np.int64(resume), np.int64(edges[i]), np.int64(edges[i + 1]), done)
                for i in range(nt) if edges[i + 1] > edges[i]]
        for f in futs:
            f.result()
    return run_block
