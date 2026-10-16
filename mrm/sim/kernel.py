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
from mrm.plant.sensing import (ADC_REF_MONITOR_PD, FAULT_DAC_GLITCH, SAMPLE_DROPPED, _CODE_MASK, _FLAG_SHIFT,
                               PR_BG, PR_RESP, PW_DARK_EA, PW_DARK_I0, PW_DARK_T0, PW_GAIN_SCALE, dark_current_a,
                               fault_mask_row, monitor_level_codes_row, monitor_release_row, monitor_tick_row,
                               readback_release_row, readback_tick_row, temp_release_row, temp_tick_row)
from mrm.scenarios import generators as _sg
from mrm.sim.metrics import (AI_WIN_TICKS, MPI_BER_WIN, TR, N_TR, metrics_tick_row, record_tick_row)
from mrm.units import C0

__all__ = ["KParams", "KState", "STEP_SIG", "STEP_FT", "null_step", "get_run_block", "release_row", "truth_row",
           "eng_pabs", "eng_commit", "eng_precompute", "extend_selfheat_rows", "swing_capacitance_row",
           "cumulative_capacitance",
           "MODE_OPEN", "MODE_COMPILED", "MODE_PYTHON", "POLICY_FIXED", "POLICY_AUTO",
           "RF", "RI", "R8", "RB", "WI", "WF", "LF", "NRF", "NRI", "NR8", "NRB", "NWI", "NWF", "NLF"]

_U = np.uint64           # unsigned index cast (no negative-index wraparound; mrm.plant.optics._U, perf.md R2-1)
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
             "ppd_prev", "tpd_prev", "lamres_prev", "pabs_prev", "t_sens_prev", "pbg_prev", "bi_vt0", "i_dark",
             "t_adc")
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
             "temp_lost", "rb_lost", "err_lost", "sub_h_max", "fast_calls", "sh_iter", "last_h", "sat_first_tick",
             "sat_first_elem", "sat_first_val")
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
 RF_TPD_PREV, RF_LAMRES_PREV, RF_PABS_PREV, RF_T_SENS_PREV, RF_PBG_PREV, RF_BI_VT0, RF_I_DARK, RF_T_ADC) = range(NRF)
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
 WI_TEMP_LOST, WI_RB_LOST, WI_ERR_LOST, WI_SUB_H_MAX, WI_FAST_CALLS, WI_SH_ITER, WI_LAST_H, WI_SAT_FIRST_TICK,
 WI_SAT_FIRST_ELEM, WI_SAT_FIRST_VAL) = range(NWI)
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
    "ctrl_latency", "reset_protocol", "ramp_ticks", "bias_ctrl", "policy", "fast_on", "schedule",
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
    "mo_shot", "mo_stuck", "mo_slot", "mo_key_adc", "mo_key_shot", "mo_key_dnl", "mo_adc_site",
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


@njit(cache=True, inline="always")
def _sh_out_blocks(args_f, args_i):
    """The output views (out_rl (N_RL, N, L), out_r (N_R, N), out_l (N_L, L)) of the self-heating rows, inlined (the
    reshapes of mrm.plant.optics._sh_outputs; a function compiled without the runtime must not call a function that
    returns arrays)."""
    n = args_i[_op.SH_N]
    nl = args_i[_op.SH_L]
    o = args_i[_op.SH_O_ORL]
    orl = args_f[o:o + _op.N_RL * n * nl].reshape((_op.N_RL, n, nl))
    o = args_i[_op.SH_O_OR]
    orr = args_f[o:o + _op.N_R * n].reshape((_op.N_R, n))
    o = args_i[_op.SH_O_OL]
    ol = args_f[o:o + _op.N_L * nl].reshape((_op.N_L, nl))
    return orl, orr, ol


@njit(cache=True, _nrt=False)
def dphi_dlamres(lam_l, shift, t_rt):
    """d phi / d lambda_res (rad/m) at line lam_l (m) of a ring whose cold phase curve is translated rigidly by the
    resonance shift ``shift`` (m), with t_rt (s) the group delay n_g(x) L / c from mrm.plant.phase.ring_optics_row:
    2 pi n_g(x) L / x^2 = 2 pi c t_rt / x^2, x = lam_l - shift (mrm.plant.phase equation 2). The slope at the
    shifted coordinate x, not at lam_l: (lam_l / x)^2 - 1 = 2 shift / lam is 1.2 % for the 7.6 nm the g26 rings are
    heated by, which is the error of 2 pi c t_rt / lam_l^2 (it scaled every detuning and the in-tick thermo-optic
    phase by that factor)."""
    x = lam_l - shift
    return _TWO_PI * C0 * t_rt / (x * x)


@njit(cache=True, _nrt=False)
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


@njit(cache=True, _nrt=False)
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


@njit(cache=True, _nrt=False)
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


@njit(cache=True, _nrt=False)
def eng_precompute(args_f, args_i):
    """Once per substep, after the optics inputs are written: the Airy-path caches and the far-line hold
    (mrm.plant.optics.sh_fast_precompute: exactly the updates the optics adapter makes lazily, so the engine and the
    reference stay bit-identical)."""
    _op.sh_fast_precompute(args_f, args_i)


@njit(cache=True, _nrt=False)
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


@njit(cache=True, _nrt=False, inline="always")
def eng_pabs(t, k, args_f, args_i):
    """pabs_fn: absorbed power of ring k at temperature t (W) and dP_abs/dT (W/K), bit-identical to
    mrm.plant.optics.selfheat_pabs; keeps what the commit needs to write every output of ring k without solving it
    again. Needs :func:`eng_precompute` for the substep."""
    if args_f[args_i[_op.SH_O_FF] + k] != 0.0:
        return _op.sh_fast_pabs(t, k, args_f, args_i)
    return _eng_pabs_general(t, k, args_f, args_i)


@njit(cache=True, _nrt=False)
def eng_commit(t, k, args_f, args_i, full):
    """commit_fn: as mrm.plant.optics.selfheat_commit, reusing the last evaluation when it was ring k at t. full =
    False (every step of a block but its last): only the outputs a plant tick reads (optics.sh_fast_outputs; the
    general path writes everything anyway); the last step of a block commits in full (docs/handoff/perf.md R2-3)."""
    if _op.sh_last_is(k, t, args_f, args_i):
        if args_f[args_i[_op.SH_O_FF] + k] != 0.0:
            _op.sh_fast_outputs(k, args_f, args_i, full)
        else:
            o_thru = _eo(args_i, E_THRU)
            o_pcur = args_i[_op.SH_O_PCUR]
            for l in range(args_i[_op.SH_L]):
                args_f[o_pcur + l] = args_f[o_thru + l]
        _op.sh_pass_on(k, args_f, args_i, full)
    else:
        _op.selfheat_commit(t, k, args_f, args_i)
    _op.sh_last_clear(args_f, args_i)


@njit(cache=True, _nrt=False)
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
    j = _U(_el.grid_index(x, grid))            # np.searchsorted(grid, x) - 1, without the binary search
    gj = grid[j]
    h = grid[j + _U(1)] - gj
    d = x - gj
    cj = col[j]
    return cum[j] + d * (cj + 0.5 * (d / h) * (col[j + _U(1)] - cj))


def cumulative_capacitance(grid, c_tab):
    """cum_q (Bt, G): integral of c'(V) from grid[0] to each node (trapezoids of the piecewise-linear table)."""
    c = np.asarray(c_tab, float)
    g = np.asarray(grid, float)
    seg = 0.5 * (c[:, 1:] + c[:, :-1]) * np.diff(g)[None, :]
    out = np.zeros_like(c)
    out[:, 1:] = np.cumsum(seg, axis=1)
    return np.ascontiguousarray(out)


# ---------------------------------------------------------------------------------------------- implicit step (copy)
@njit(cache=True, _nrt=False)
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


@njit(cache=True, _nrt=False)
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


@njit(cache=True, _nrt=False)
def _heater_split(x, a, c0, c1, lag_src, obs_ptr, obs_end, obs, self_src, lo_s, hi_s, unit, u0, u1, src_scale):
    """pre of mrm.plant.thermal.observer_split_b (the gain comes from _th_gains), bit-identical: the same sum in the
    same order, without the per-lag source test when the self-heater lags form the range [lo_s, hi_s). obs_end:
    the end of each observer's lags without its trailing dead lags (:func:`_th_live`; each would add +0.0)."""
    pre = 0.0
    lo = obs_ptr[obs]
    hi = obs_end[obs]
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
    for m in range(hi_s, max(hi, hi_s)):
        s = lag_src[m]
        pre += a[m] * x[m]
        pre += c0[m] * u0[s] + c1[m] * u1[s]
    return pre


@njit(cache=True, _nrt=False)
def _heater_split_fill(obs_ptr, obs_end, lag_src, heater_obs, heater_src, hs_tab):
    """The tables of :func:`_heater_split_all` into hs_tab (2 N max-lags,): returns ln, the common number of lags of
    the heater observers (-1 when they differ; up to obs_end, :func:`_th_live`), and writes per lag position j and
    ring n the pair (lag, its source, or -1 for the ring's own heater) at hs_tab[2 (j N + n)] (perf.md R2-9)."""
    nr = heater_obs.shape[0]
    ln = obs_end[heater_obs[0]] - obs_ptr[heater_obs[0]] if nr > 0 else 0
    for n in range(nr):
        if obs_end[heater_obs[n]] - obs_ptr[heater_obs[n]] != ln:
            ln = -1
    for j in range(max(ln, 0)):
        for n in range(nr):
            m = obs_ptr[heater_obs[n]] + j
            q = 2 * (j * nr + n)
            hs_tab[q] = m
            hs_tab[q + 1] = -1 if lag_src[m] == heater_src[n] else lag_src[m]
    return ln


@njit(cache=True)
def _th_dead_init(x, lag_src, lag_kind, n_src, dead):
    """dead[s] = True for every source whose lags are all LAG_EXOGENOUS and exactly +0.0 (the sign bit clear), e.g. a
    second aggressor site that no scenario drives, or a cold driver. Such a lag stays exactly +0.0 while its source
    has no input (a (+0.0) + (c0 0 + c1 0) = +0.0: the input term is +-0.0, and +0.0 + -0.0 = +0.0 in round to
    nearest), so the lag pass can skip it and the observer sums can leave it out (adding +0.0 to a sum that starts
    at the base temperature changes nothing). Returns the number of dead sources. docs/handoff/perf.md R2-5."""
    for s in range(n_src):
        dead[s] = False
    n_lags = np.zeros(n_src, np.int64)
    ok = np.ones(n_src, np.bool_)
    for m in range(lag_src.shape[0]):
        s = lag_src[m]
        n_lags[s] += 1
        if lag_kind[m] != LAG_EXOGENOUS or not (x[m] == 0.0 and math.copysign(1.0, x[m]) > 0.0):
            ok[s] = False
    nd = 0
    for s in range(n_src):
        if ok[s] and n_lags[s] > 0:
            dead[s] = True
            nd += 1
    return nd


@njit(cache=True, _nrt=False)
def _th_live(dead, lag_src, obs_ptr, idx_exo, out_ms, out_obs_end):
    """The live exogenous lags (idx_exo without the lags of dead sources) as pairs (lag, source) into out_ms,
    returning their number, and the end of each observer's lags without its trailing dead lags into out_obs_end (a
    dead lag inside an observer's range stays in its sum: +0.0)."""
    n = 0
    for i in range(idx_exo.shape[0]):
        m = idx_exo[i]
        if not dead[lag_src[m]]:
            out_ms[2 * n] = m
            out_ms[2 * n + 1] = lag_src[m]
            n += 1
    for o in range(obs_ptr.shape[0] - 1):
        e = obs_ptr[o + 1]
        while e > obs_ptr[o] and dead[lag_src[e - 1]]:
            e -= 1
        out_obs_end[o] = e
    return n


@njit(cache=True)
def _th_tables(idx_cross, opt_lags, lag_src, lag_obs, src_ring):
    """Per block: the cross-ring lags as pairs (lag, source ring) and the optical lags as triples (lag, observer,
    source ring), so each lag pass reads its indices with one or two pair loads (perf.md R2-9)."""
    ms_cross = np.empty(2 * idx_cross.shape[0], np.int64)
    for i in range(idx_cross.shape[0]):
        m = idx_cross[i]
        ms_cross[2 * i] = m
        ms_cross[2 * i + 1] = src_ring[lag_src[m]]
    opt_tab = np.empty(3 * opt_lags.shape[0], np.int64)
    for i in range(opt_lags.shape[0]):
        m = opt_lags[i]
        opt_tab[3 * i] = m
        opt_tab[3 * i + 1] = lag_obs[m]
        opt_tab[3 * i + 2] = src_ring[lag_src[m]]
    return ms_cross, opt_tab


@njit(cache=True, _nrt=False)
def _th_coef(a, c0, c1, w, coef):
    """The lag coefficients of one substep power as structure of arrays per lag: coef[4 m .. 4 m + 3] = (a, c0, c1,
    w) (one pair load per two coefficients in the lag passes; perf.md R2-9)."""
    for m in range(a.shape[0]):
        q = 4 * m
        coef[q] = a[m]
        coef[q + 1] = c0[m]
        coef[q + 2] = c1[m]
        coef[q + 3] = w[m]


@njit(cache=True, _nrt=False)
def _uu_fill(u0, u1, uu):
    """uu[2 s], uu[2 s + 1] = u0[s], u1[s] (the inputs of source s at the start and end of the step, side by side)."""
    for s in range(u0.shape[0]):
        uu[2 * s] = u0[s]
        uu[2 * s + 1] = u1[s]


@njit(cache=True, _nrt=False)
def _th_wake(dead, u0, u1):
    """True when a dead source has an input now (it is then marked live; the caller rebuilds the lag lists)."""
    woke = False
    for s in range(dead.shape[0]):
        if dead[s] and (u0[s] != 0.0 or u1[s] != 0.0):
            dead[s] = False
            woke = True
    return woke


@njit(cache=True, _nrt=False, inline="always")
def _heater_split_all(x, a, c0, c1, lag_src, obs_ptr, obs_end, heater_obs, heater_src, hs_lo, hs_hi, unit, u0, u1,
                      src_scale, ln, hs_tab, coef, uu, sums, out_pre):
    """_heater_split of every ring into out_pre (N,). When the heater observers have the same number of lags (ln >= 0,
    tables of :func:`_heater_split_fill`), the N sums run in lockstep (lag j of every ring, then lag j + 1), each in
    its own order, so the N independent addition chains overlap in the CPU instead of running one after the other;
    the coefficients come from the per-lag table coef (:func:`_th_coef`) and the inputs from uu (:func:`_uu_fill`).
    The same additions in the same order per ring as _heater_split, so the values are bit-identical. sums (N,):
    scratch (no allocation per tick)."""
    nr = heater_obs.shape[0]
    if ln < 0 or not unit:
        for n in range(nr):
            out_pre[n] = _heater_split(x, a, c0, c1, lag_src, obs_ptr, obs_end, heater_obs[n], heater_src[n],
                                       hs_lo[n], hs_hi[n], unit, u0, u1, src_scale)
        return
    for n in range(nr):
        sums[n] = 0.0
    q = 0
    for j in range(ln):
        for n in range(nr):
            m = _U(hs_tab[q])
            sv = hs_tab[q + 1]
            q += 2
            m4 = m * _U(4)
            pre = sums[n] + coef[m4] * x[m]
            if sv >= 0:
                s2 = _U(2 * sv)
                pre = pre + (coef[m4 + _U(1)] * uu[s2] + coef[m4 + _U(2)] * uu[s2 + _U(1)])
            sums[n] = pre
    for n in range(nr):
        out_pre[n] = sums[n]


@njit(cache=True, _nrt=False)
def _implicit_step_eng(x, coef, uu, ms_exo, ms_cross, idx_ax, opt_tab, lag_obs, obs_ptr, obs_end, ring_obs,
                       bus_order, src_scale, unit, t_base, p_prev, t_prev, args_f, args_i, p_max, d2p_max, max_iter,
                       tol_k, n_outer, g_work, t_obs, p_new, t_ring, n_iter, gobs, full, p_cross):
    """mrm.plant.selfheat.implicit_step_b (owner: plant_thermal; 07 P0-1) calling the engine adapter eng_pabs /
    eng_commit directly, the same arithmetic otherwise. Why a copy: passing the adapter as a first-class value makes
    numba embed the dispatcher addresses in the code ("dynamic globals"), and then the kernel cannot be cached on disk
    (every process would recompile the plant for minutes). The lag passes run from the tables of _th_struct /
    _th_gains (w, gobs, the index lists, unit) instead of branching per lag: bit-identical (docs/handoff/perf.md P8).
    The lag coefficients (a, c0, c1, w) come per lag from coef (:func:`_th_coef`), the inputs (u0, u1) per source
    from uu (:func:`_uu_fill`), the exogenous lags as (lag, source) pairs ms_exo, the cross-ring lags as (lag, source
    ring) pairs ms_cross and the optical lags as (lag, observer, source ring) triples opt_tab (perf.md R2-9: the same
    arithmetic). ms_exo may leave out the lags of dead sources and obs_end the trailing ones of each observer
    (:func:`_th_live`: lags that are exactly +0.0 and have no input stay +0.0 and add +0.0); p_cross (N,): scratch.
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
    # (indices cast to unsigned: no negative-index wraparound per access, docs/handoff/perf.md R2-1)
    n_exo = ms_exo.shape[0] // 2
    if unit:
        for i in range(n_exo):
            m = _U(ms_exo[2 * i])
            s2 = _U(2 * ms_exo[2 * i + 1])
            m4 = m * _U(4)
            x[m] = coef[m4] * x[m] + (coef[m4 + _U(1)] * uu[s2] + coef[m4 + _U(2)] * uu[s2 + _U(1)])
    else:
        for i in range(n_exo):
            m = _U(ms_exo[2 * i])
            sv = _U(ms_exo[2 * i + 1])
            s2 = sv * _U(2)
            m4 = m * _U(4)
            x[m] = coef[m4] * x[m] + src_scale[sv] * (coef[m4 + _U(1)] * uu[s2] + coef[m4 + _U(2)] * uu[s2 + _U(1)])
    n_cross = ms_cross.shape[0] // 2
    for i in range(n_cross):
        m = _U(ms_cross[2 * i])
        m4 = m * _U(4)
        x[m] = coef[m4] * x[m] + coef[m4 + _U(3)] * p_prev[_U(ms_cross[2 * i + 1])]
    for i in range(idx_ax.shape[0]):
        m = _U(idx_ax[i])
        x[m] = coef[m * _U(4)] * x[m]
    for o in range(n_obs):
        e = t_base
        for m in range(obs_ptr[o], obs_end[o]):
            e += x[_U(m)]
        t_obs[o] = e                       # everything except the new optical inputs
        g_work[o] = gobs[o]
    # --- ring solves in bus order
    n_fail = 0
    # p_cross: the P_abs currently driving the LAG_OPT_CROSS_RING lags
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
            eng_commit(t, k, args_f, args_i, full)
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
        for i in range(n_cross):
            m = _U(ms_cross[2 * i])
            j = _U(ms_cross[2 * i + 1])
            dx = coef[m * _U(4) + _U(3)] * (p_new[j] - p_cross[j])
            x[m] += dx
            t_obs[_U(lag_obs[m])] += dx
        for k in range(nr):
            p_cross[k] = p_new[k]
    # --- optical lags with the new absorbed powers
    # (each observer's sum is carried in a register while consecutive lags feed it: the same additions in the same
    # order as t_obs[lag_obs[m]] += dx, without a store-to-load round trip per lag; bit-identical)
    o_cur = -1
    t_run = 0.0
    for i in range(opt_tab.shape[0] // 3):
        m = _U(opt_tab[3 * i])
        o = opt_tab[3 * i + 1]
        if o != o_cur:
            if o_cur >= 0:
                t_obs[_U(o_cur)] = t_run
            o_cur = o
            t_run = t_obs[_U(o)]
        dx = coef[m * _U(4) + _U(3)] * p_new[_U(opt_tab[3 * i + 2])]
        x[m] += dx
        t_run += dx
    if o_cur >= 0:
        t_obs[o_cur] = t_run
    return n_fail




# ---------------------------------------------------------------------------------------------- sensor release
@njit(cache=True)
def release_row(P, S, b, k):
    """Event-order step 2 at controller edge k for world b: release the monitor, temperature, readback and
    error-counter FIFOs, run the bit-statistics window, and pack obs_i64 (07 P0-6, P0-7, P0-11). :func:`_release_a`
    on world b's arrays; the observation's active ring is :func:`_active_ring` of the edge (07 P2-7)."""
    _release_a(k, P.n_rings, _row(P.mo_lat, b), S.mo_fv[b], S.mo_fr[b], S.mo_fm[b], S.mo_samples[b], S.mo_ticks[b],
               S.rf[b], S.ri[b], S.rb[b], S.wi[b], S.ts_fv[b], S.ts_fr[b], S.ts_fm[b], S.ts_latest[b], S.ts_new[b],
               P.rb_on, S.rb_fv[b], S.rb_fr[b], S.rb_fm[b], S.dp_fv[b], S.dp_fr[b], S.dp_fm[b], S.sh_args_f[b],
               P.sh_args_i[b], P.mo_pr[b], P.mo_pw[b], P.mo_pga, P.mo_shot, P.dp_bs_tau, P.dp_bs_stia, P.dp_rin[b],
               P.dp_win_bits, P.dp_mode, P.dp_words, P.dp_pre, P.dp_period, P.dp_key_dp[b], S.obs[b], P.obs_map,
               _active_ring(S.wi[b], P.schedule, P.n_rings))


@njit(cache=True, _nrt=False)
def _release_a(k, n_r, mo_lat, mo_fv, mo_fr, mo_fm, mo_samples, mo_ticks, rf, ri, rbb, wi, ts_fv, ts_fr, ts_fm,
               ts_latest, ts_new, rb_on, rb_fv, rb_fr, rb_fm, dp_fv, dp_fr, dp_fm, args_f, args_i, mo_pr, mo_pw, mo_pga,
               mo_shot, dp_bs_tau, dp_bs_stia, dp_rin, dp_win_bits, dp_mode, dp_words, dp_pre, dp_period, dp_key,
               obs, obs_map, active):
    """:func:`release_row` on one world's arrays (a call that takes the KParams/KState tuples copies all ~270 array
    structs, ~0.4 us; docs/handoff/perf.md R2-8). The same calls in the same order. ``active``: the ring this edge
    updates under round_robin, -1 for every ring (packed into the observation header, OBS_ACTIVE_RING)."""
    lost = monitor_release_row(k, mo_lat, mo_fv, mo_fr, mo_fm, mo_samples, mo_ticks, ri[RI_MO_COUNT],
                               ri[RI_MO_LATEST], rbb[RB_MO_VALID], rbb[RB_MO_OVER], rbb[RB_MO_NEW])
    wi[WI_MON_LOST] += lost
    if ts_latest.shape[0] > 0:
        temp_release_row(k, ts_fv, ts_fr, ts_fm, ts_latest, ts_new)
    if rb_on != 0:
        readback_release_row(k, rb_fv, rb_fr, rb_fm, ri[RI_RB_V], ri[RI_RB_I], rbb[RB_RB_NEW])
    errors_release_row(k, dp_fv, dp_fr, dp_fm, ri[RI_DP_ERR], rbb[RB_ERR_VALID])
    # bit statistics (07 P0-11): level codes of the last interval through the sampler bandwidth, then the window of
    # the last bits at the scenario's ones density (RF_DENS; the realized n1/nb of a tick, RF_RHO, is never exactly
    # 0.5 under a PRBS and would miss the exact PRBS window)
    o = args_i[_op.SH_O_OR]
    n_sh = args_i[_op.SH_N]
    monitor_level_codes_row(args_f[o + _op.R_DROP_SUM1 * n_sh:o + (_op.R_DROP_SUM1 + 1) * n_sh],
                            args_f[o + _op.R_DROP_SUM0 * n_sh:o + (_op.R_DROP_SUM0 + 1) * n_sh], rf[RF_P_BG],
                            rf[RF_T_PD], rf[RF_T_ADC], ri[RI_PGA], mo_pr, mo_pw, mo_pga, mo_shot, dp_bs_tau, rf[RF_MU1],
                            rf[RF_MU0], rf[RF_SIG1], rf[RF_SIG0], dp_bs_stia, dp_rin)
    bitstat_window_row(k, dp_win_bits, dp_mode, rf[RF_DENS], rf[RF_MU1], rf[RF_MU0], rf[RF_SIG1], rf[RF_SIG0],
                       ri[RI_BS_THR], dp_words, dp_pre, dp_period, ri[RI_DP_BITPOS], dp_key,
                       ri[RI_BS_ABOVE], ri[RI_BS_N1], ri[RI_BS_A1], ri[RI_BS_A0])
    pack_obs_row(obs, obs_map, k, active, mo_samples, mo_ticks, ri[RI_MO_COUNT], ri[RI_MO_LATEST],
                 rbb[RB_MO_VALID], rbb[RB_MO_OVER], rbb[RB_MO_NEW], ts_latest, ts_new, ri[RI_RB_V], ri[RI_RB_I],
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
    _fast_tick_a(fast_fn, k, P.n_rings, _row(P.mo_lat, b), S.fobs[b], S.ri[b], S.mo_fv[b], S.mo_fr[b], S.mo_fm[b],
                 _row(S.c_pi, b), _row(S.c_pf, b), S.c_si[b], S.c_sf[b], S.fact[b], S.wi[b])


@njit(cache=True, _nrt=False)
def _fast_tick_a(fast_fn, k, n_r, lat, fo, ri, fv, fr, fm, c_pi, c_pf, c_si, c_sf, fact, wi):
    """:func:`_fast_tick` on one world's arrays (the tick loop calls it; runtime-free, perf.md R2-11)."""
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
        fast_fn(c_pi, c_pf, c_si, c_sf, fo, fact)
        wi[WI_FAST_CALLS] += 1










@njit(cache=True, _nrt=False)
def _sync_act_a(act, act_map, he_reg, bi_reg):
    """Pre-fill the held command from the latched DAC registers (the ABI rule of mrm.control.api)."""
    for i in range(act_map.shape[0]):
        o = act_map[i, 1]
        for j in range(act_map[i, 2]):
            if act_map[i, 0] == 0:
                act[o + j] = he_reg[j]
            elif act_map[i, 0] == 1:
                act[o + j] = bi_reg[j]


@njit(cache=True, _nrt=False)
def _first_out_of_range_a(act, act_map, wi, k):
    """Record the first out-of-range command element of the world (tick, act_i64 index, value) in
    wi[WI_SAT_FIRST_*], before the clip (the runner raises ActionRangeError from it under on_out_of_range: error)."""
    for i in range(act_map.shape[0]):
        o = act_map[i, 1]
        lo = act_map[i, 3]
        hi = act_map[i, 4]
        for j in range(act_map[i, 2]):
            v = act[o + j]
            if v < lo or v > hi:
                wi[WI_SAT_FIRST_TICK] = k
                wi[WI_SAT_FIRST_ELEM] = o + j
                wi[WI_SAT_FIRST_VAL] = v
                return


@njit(cache=True, _nrt=False)
def _push_act_a(act, act_map, sat, pipe_vals, pipe_ticks, pipe_meta, wi, k, latency):
    """Event-order step 4: clip every actuator to its channel range (in the held vector itself, so the controller
    sees the clipped command next time as the applied register does), count the clipped elements (the first one
    also in wi[WI_SAT_FIRST_*]), and queue the command to latch at k + Controller.latency_ticks."""
    if wi[WI_SAT_FIRST_TICK] < 0:
        _first_out_of_range_a(act, act_map, wi, k)
    wi[WI_SAT_TOTAL] += clip_act_row(act, act_map, sat)
    if not pipe_push_row(pipe_vals, pipe_ticks, pipe_meta, act, k + latency):
        wi[WI_PIPE_OVER] += 1


@njit(cache=True, _nrt=False)
def _active_ring(wi, schedule, n_r):
    """07 P2-7 ``controller_interface.schedule``: the ring that controller edge j (wi[WI_EDGE_IDX], counted from the
    controller phase) updates under round_robin, j % N; -1 (every ring) when simultaneous."""
    if schedule != 0 and n_r > 0:
        return wi[WI_EDGE_IDX] % n_r
    return np.int64(-1)


@njit(cache=True, _nrt=False)
def _rr_save_a(act, save):
    """Round robin: keep the held command of the edge before the controller writes it (save: S.act_tmp[b], free at
    an edge: the pipeline pop that fills it runs after the edge)."""
    for i in range(act.shape[0]):
        save[i] = act[i]


@njit(cache=True, _nrt=False)
def _rr_apply_a(act, save, act_map, active):
    """Round robin (07 P2-7): of every per-ring actuator channel ((B, N): act_map column 5) only the active ring's
    command is applied; every other ring keeps its held command of the edge. Channels that are not per ring
    ((B, L), (B, 1)) are applied whole, as in mrm.control.api.CompositeController."""
    for i in range(act_map.shape[0]):
        if act_map[i, 5] == 0:
            continue
        o = act_map[i, 1]
        for j in range(act_map[i, 2]):
            if j != active:
                act[o + j] = save[o + j]


@njit(cache=True, _nrt=False)
def _advance_edge_a(wi, phase, period, jitter, key_sched):
    """Next controller edge: the tick of edge j is phase + j period + jitter (07 P2-7; mrm.core.timing.edge_jitter)."""
    j = wi[WI_EDGE_IDX] + 1
    wi[WI_EDGE_IDX] = j
    e = phase + j * period
    if jitter > 0:
        e += edge_jitter(key_sched, j, jitter)
    wi[WI_EDGE_NEXT] = e


# ---------------------------------------------------------------------------------------------- truth and metrics
@njit(cache=True, _nrt=False, inline="always")
def _truth_a(n_r, n_l, rf, ri, rbb, tv, lf, lt, t_base, args_f, args_i, pw, mo_pr, rp, jx, fwhm_b, wrong_window,
             assign_mode, with_db, dark_known, ws):
    """Truth of one world at the end of the last step (ARCHITECTURE Truth; metrics.TRUTH_FIELDS) into tv, the
    per-line truth lt = [chip input power, bus output power, wavelength] (L,), the detuning rf[RF_DET], the order,
    the line each ring sits on and the wrong-line flag against the declared assignment (07 P1-12). Takes the world's
    arrays (pw = P.mo_pw[b], mo_pr = P.mo_pr[b], rp = P.ph_ring[b], fwhm_b = P.fwhm_hz[b]) instead of the tuples.
    with_db: also er_db and il_db (two log10 per ring; no kernel computation reads them, so the world block asks for
    them only when a record does and at the last step of a block; docs/handoff/perf.md P12). dark_known: the PD dark
    current of this step is in rf[RF_I_DARK] (the monitor's value, the same expression), else it is computed.
    ws (3, N): scratch. The exact detuning of every ring runs :func:`resonance_detuning`'s Newton iteration for all
    rings in lockstep (iteration i of every ring, then i + 1), each ring with its own sequence of values and its own
    stop at a fixed point, so the N dependent division chains overlap (bit-identical; docs/handoff/perf.md R2-10)."""
    rg, nb_ = _sh_blocks(args_f, args_i)
    phi0r, phi1r, dpl = nb_[_op.SHN_PHI0R], nb_[_op.SHN_PHI1R], nb_[_op.SHN_DPL]
    tref, s0v, betav = rg[_op.SHR_TREF], rg[_op.SHR_S0], rg[_op.SHR_BETA]
    out_rl, out_r, out_l = _sh_out_blocks(args_f, args_i)
    o_fwhm = _eo(args_i, E_FWHM)
    # --- per ring: line search, exact detuning of the declared line, Truth fields
    for n in range(n_r):
        dT = rf[RF_T_RING, n] - tref[n]
        shift = s0v[n] * (dT + 0.5 * betav[n] * dT * dT)      # R1 about T_prev (mrm.plant.phase local law)
        dl = ri[RI_DECL, n]
        best = -1
        best_d = _INF
        order_decl = 0
        o_fw = o_fwhm + n * n_l
        for l in range(n_l):
            dpl_nl = dpl[n, l]
            ph = 0.5 * (phi0r[n, l] + phi1r[n, l]) + dpl_nl * shift
            m = math.floor(ph / _TWO_PI + 0.5)
            num = ph - _TWO_PI * m
            if l == dl:
                order_decl = np.int64(m)
            lim = wrong_window * args_f[_U(o_fw + l)]     # window x (lam_l^2 fwhm_hz / c), set when the lines move
            # d = num / dpl (lambda_res - lambda_line, first order in d), divided only for a line that can be inside
            # the window: |num| > lim |dpl| (1 + 1e-12) gives fl(|num / dpl|) > lim, so the skipped lines are the
            # ones the test below rejects (bit-identical; docs/handoff/perf.md R2-7)
            if abs(num) <= lim * abs(dpl_nl) * (1.0 + 1e-12):
                d = num / dpl_nl
                if abs(d) <= lim and abs(d) < best_d:
                    best_d = abs(d)
                    best = l
        # the declared line exactly: the mean of the two NRZ level resonances of the order nearest the line, from the
        # phase model's cold curve translated by the end-of-step mean shift (shifts at T_prev plus the R1 increment):
        # resonance_detuning(rp[n], lam_l, 0.5 (jx0 + jx1) + shift, order) = x_m - x_end, x_m below
        ri[RI_ORDER, n] = order_decl
        ri[RI_LINE_ON, n] = best
        x_end = lf[LF_LAM, _U(dl)] - (0.5 * (jx[0, n, 0] + jx[0, n, 1]) + shift)
        ws[0, n] = x_end
        ws[1, n] = x_end
        ws[2, n] = 1.0
    # --- cold_coordinate(rp[n], x_end, order) of every ring, in lockstep
    for _ in range(5):
        busy = False
        for n in range(n_r):
            if ws[2, n] == 0.0:
                continue
            lam_r = rp[n, _ph.RP_LAM_RES]
            ne = rp[n, _ph.RP_N_EFF]
            ng = rp[n, _ph.RP_N_G]
            dng = rp[n, _ph.RP_DNG]
            ln = rp[n, _ph.RP_LEN]
            x = ws[1, n]
            dd = x - lam_r
            n_x = ne + (ne - ng) * dd / lam_r - 0.5 * dng * dd * dd / lam_r
            ng_x = ng + dng * dd + 0.5 * dng * dd * dd / lam_r
            g = n_x * ln / x - float(ri[RI_ORDER, n])
            x_new = x + g * x * x / (ng_x * ln)
            if x_new == x:
                ws[2, n] = 0.0              # an exact fixed point (cold_coordinate stops here)
            else:
                ws[1, n] = x_new
                busy = True
        if not busy:
            break
    # --- Truth fields
    for n in range(n_r):
        dl = ri[RI_DECL, n]
        dlu = _U(dl)
        lam_l = lf[LF_LAM, dlu]
        det_decl = ws[1, n] - ws[0, n]
        order_decl = ri[RI_ORDER, n]
        best = ri[RI_LINE_ON, n]
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
        pin = out_rl[_op.RL_PIN, n, dlu]
        t1 = out_rl[_op.RL_THRU1, n, dlu]
        t0 = out_rl[_op.RL_THRU0, n, dlu]
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
             P.wrong_window, P.assign_mode, True, False, np.empty((3, P.n_rings)))


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
    _oracle_a(P.n_rings, S.rf[b], S.ri[b], S.sh_args_f[b], P.sh_args_i[b], S.or_phi[b], S.or_k2t[b], S.or_k2d[b],
              S.or_a[b], S.or_trt[b], S.or_psi[b], S.or_res0[b], S.or_res1[b], S.or_cw[b], S.or_fw[b], S.lf[b],
              P.ph_ring[b], S.jx[b], P.fwhm_hz[b], P.z_opt_self[b], P.use_tcmt, P.m_orders, P.lock_ref_mode,
              S.m_ref[b])


@njit(cache=True, _nrt=False)
def _oracle_a(n_r, rf, ri, args_f, args_i, phi, rk2t, rk2d, ra, rtrt, rpsi, res0, res1, cw, fw, lf, rp, jx, fwhm_b,
              z_opt_b, use_tcmt, m_orders, lock_ref_mode, m_ref):
    """:func:`_oracle` on one world's arrays (the tick loop calls it; runtime-free, docs/handoff/perf.md R2-11)."""
    v = _op.selfheat_views(args_f, args_i)
    phi0r, phi1r, a0, a1, e0, e1 = v[4], v[5], v[8], v[9], v[10], v[11]
    psi, k2t, k2d, trt, dpl, rb = v[12], v[13], v[14], v[15], v[16], v[20]
    out_rl = v[22]
    s0v, betav, tref = v[18], v[19], v[17]
    npts = phi.shape[0]
    for n in range(n_r):
        dl = ri[RI_DECL, n]
        lam = lf[LF_LAM, dl]
        fwhm = lam * lam * fwhm_b[n] / C0
        dphi = phi1r[n, dl] - phi0r[n, dl]
        mean = 0.5 * (phi0r[n, dl] + phi1r[n, dl])
        m = math.floor(mean / _TWO_PI + 0.5)
        dT = rf[RF_T_RING, n] - tref[n]
        s_now = s0v[n] * (1.0 + betav[n] * dT)
        g_sh = z_opt_b[n] * s_now                      # m per absorbed W, static
        pin = out_rl[_op.RL_PIN, n, dl]
        q1 = rf[RF_RHO, n] if args_i[_op.SH_TWO] != 0 else 0.0
        for lvl in range(2):
            for i in range(npts):
                d = (-1.5 + 3.0 * i / (npts - 1)) * fwhm
                phi[i] = _TWO_PI * m + dpl[n, dl] * d + (-0.5 if lvl == 0 else 0.5) * dphi
                rk2t[i] = k2t[n, dl]
                rk2d[i] = k2d[n, dl]
                ra[i] = a0[n, dl] if lvl == 0 else a1[n, dl]
                rtrt[i] = trt[n, dl]
                rpsi[i] = psi[n, dl]
            _op.ring_rows_nb(phi, rpsi, rk2t, rk2d, ra, rtrt, rb[n], use_tcmt != 0, m_orders, True, cw, fw,
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
        if lock_ref_mode == 1:
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
            m_ref[n] = x_m - cold_coordinate(rp[n], x_m, m + dpl[n, dl] * d_ib / _TWO_PI)


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
    bb = b if P.sc_comp_f.shape[0] > 1 else 0
    return _auto_h_a(k, k_end, P.n_rings, P.fast_on, P.au_max_pow, P.reset_protocol, P.ramp_ticks, P.oracle_on,
                     P.he_ripple, P.bi_noise_per, _row(P.bi_noise, b), P.au_max_det, P.au_pabs_tol, S.wi[b], S.wf[b],
                     S.ri[b], S.r8[b], S.rf[b], S.rb[b], S.pipe_meta[b], S.pipe_ticks[b], S.he_fifo[b], S.bi_fifo[b],
                     P.sc_sig_i, P.sc_sig_f, P.sc_def, P.sc_comp, P.sc_comp_i, P.sc_comp_f[bb], P.sc_tab[bb],
                     P.sc_keys[b], S.sc_cache[b], S.sc_vp[b], S.sc_v0[b], S.sc_v1[b], S.sc_va[b], P.rec_i,
                     S.rec_idx[b], S.rec_cnt[b], P.mp_i, S.m_ai[b])


@njit(cache=True, _nrt=False)
def _auto_h_a(k, k_end, n_rings, fast_on, au_max_pow, reset_protocol, ramp_ticks, oracle_on, he_ripple, bi_noise_per,
              bi_noise, au_max_det, au_pabs_tol, wi, wf, ri_, r8_, rf_, rbb_, pipe_meta, pipe_ticks, he_fifo, bi_fifo,
              sc_sig_i, sc_sig_f, sc_def, sc_comp, sc_comp_i, sc_comp_f, sc_tab, sc_keys, sc_cache, sc_vp, sc_v0, sc_v1,
              sc_va, rec_i, rec_idx, rec_cnt, mp_i, m_ai):
    """:func:`_auto_h` on one world's arrays (a call that takes the KParams/KState tuples copies ~270 array structs,
    ~0.4 us; docs/handoff/perf.md R2-8). The same rules in the same order; the held-signal check evaluates the
    scenario with mrm.scenarios.generators.scen_eval_b as _scen_eval does."""
    if fast_on != 0:
        return 1
    hmax = np.int64(1) << au_max_pow
    lim = k_end - k
    e = wi[WI_EDGE_NEXT] - k
    if e > 0 and e < lim:
        lim = e
    meta = pipe_meta
    if meta[1] > 0:
        t = pipe_ticks[meta[0]] - k
        if t <= 0:
            return 1
        if t < lim:
            lim = t
    if reset_protocol == 0 and ramp_ticks > 0 and k - wi[WI_K_RESET] < ramp_ticks:
        return 1
    if wi[WI_ORACLE_NEXT] - k > 0 and wi[WI_ORACLE_NEXT] - k < lim and oracle_on != 0:
        lim = wi[WI_ORACLE_NEXT] - k
    # (b) growth limit
    g2 = 2 * wi[WI_LAST_H]
    if g2 < lim:
        lim = g2
    if lim < 2:
        return 1
    ri = ri_
    r8 = r8_
    rf = rf_
    if not _el.actuator_quiescent_row(ri[RI_HE_REG], r8[R8_HE_HAS], he_fifo, rf[RF_HE_YS], rf[RF_HE_YO],
                                      rf[RF_HE_UT], 1e-9):
        return 1
    if not _el.actuator_quiescent_row(ri[RI_BI_REG], r8[R8_BI_HAS], bi_fifo, rf[RF_BI_YS], rf[RF_BI_YO],
                                      rf[RF_BI_VT], 1e-9):
        return 1
    rbb = rbb_
    for n in range(n_rings):
        if ri[RI_HE_ADD, n] != 0:
            return 1
        # a command latched at this tick changes the register: one tick first
        if rbb[RB_VAL_H, n] and ri[RI_HE_CMD, n] != ri[RI_HE_REG, n]:
            return 1
        if rbb[RB_VAL_B, n] and ri[RI_BI_CMD, n] != ri[RI_BI_REG, n]:
            return 1
    if he_ripple != 0:
        return 1
    # bias-DAC noise: never across a redraw when it is redrawn less often than every tick (with a per-tick redraw
    # the draw of the first tick is held over the substep, a stated approximation)
    if bi_noise_per > 1:
        nz = bi_noise
        on = False
        for n in range(nz.shape[0]):
            if nz[n] != 0.0:
                on = True
        if on:
            nxt = (k // bi_noise_per + 1) * bi_noise_per - k
            if nxt < lim:
                lim = nxt
    # (a) a scenario event at k: one tick first
    hp = wi[WI_LAST_H]
    if hp < 1:
        hp = 1
    vp = sc_vp
    v0 = sc_v0
    v1 = sc_v1
    for s in range(sc_sig_i.shape[0]):
        if sc_sig_i[s, _SI_NC] == 0:
            continue
        off = sc_sig_i[s, _SI_OFF]
        foh = s == _SIG_T_BASE or s == _SIG_XPU or s == _SIG_AGG
        for j in range(sc_sig_i[s, _sg.SI_WIDTH]):
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
    for s in range(sc_sig_i.shape[0]):
        if sc_sig_i[s, _SI_NC] == 0:
            continue
        kb = _sg.scen_next_break_b(s, k, sc_comp, sc_sig_i, sc_comp_i, sc_comp_f, sc_tab)
        if kb - k < lim:
            lim = kb - k
        if lim < 2:
            return 1
    # detuning-rate and P_abs criteria from the previous substep
    rate = wf[WF_RATE]
    if rate > 0.0:
        hr = au_max_det / rate
        if hr < lim:
            lim = np.int64(hr) if hr >= 1.0 else 1
    for n in range(n_rings):
        pa = rf[RF_P_ABS, n]
        pp_ = rf[RF_PABS_PREV, n]
        sc = max(abs(pa), abs(pp_))
        if sc > 0.0 and abs(pa - pp_) > au_pabs_tol * sc:
            return 1
    # record windows (while the record still records) and BER windows: a substep never straddles one
    for r in range(rec_i.shape[0]):
        if rec_idx[r] >= rec_i[r, 4]:
            continue
        dec = rec_i[r, 1]
        left = dec - rec_cnt[r]
        if left < lim:
            lim = left
    win = mp_i[MPI_BER_WIN]
    if win > 0:
        left = win - m_ai[0, AI_WIN_TICKS]
        if left > 0 and left < lim:
            lim = left
    if lim > hmax:
        lim = hmax
    h = np.int64(1)
    while h * 2 <= lim:
        h *= 2
    if h > 1:
        # held scenario signals must not change over [k, k + h]: compare the values at k + h with those at k
        va = sc_va
        for i in range(va.shape[0]):
            va[i] = sc_v1[i]
        _sg.scen_eval_b(k + h, sc_comp, sc_sig_i, sc_sig_f, sc_def, sc_comp_i, sc_comp_f, sc_tab, sc_keys,
                        sc_cache, va)
        while h > 1:
            ok = True
            for s in range(sc_sig_i.shape[0]):
                if sc_sig_i[s, _SI_NC] == 0 or s == _SIG_T_BASE or s == _SIG_XPU or s == _SIG_AGG:
                    continue
                off = sc_sig_i[s, _SI_OFF]
                for e in range(sc_sig_i[s, _sg.SI_WIDTH]):
                    a = sc_v0[off + e]
                    c = va[off + e]
                    if not (a == c or (a != a and c != c)):
                        ok = False
            if ok:
                break
            h //= 2
            _sg.scen_eval_b(k + h, sc_comp, sc_sig_i, sc_sig_f, sc_def, sc_comp_i, sc_comp_f, sc_tab, sc_keys,
                            sc_cache, va)
    return h


@njit(cache=True, _nrt=False)
def _bus_walk(args_f, args_i, bus_order, use_tcmt, m_orders):
    """The explicit bus pass of a plant without self-heating (mrm.plant.optics.bus_propagate_nb on the packed rows).
    A function with the runtime: it builds the views it needs and returns nothing, so the runtime-free tick loop can
    call it (a runtime-free function must not hold views made by a function with the runtime; perf.md R2-11)."""
    vv = _op.selfheat_views(args_f, args_i)
    _op.bus_propagate_nb(vv[0], bus_order, vv[1], vv[2][0], vv[3], vv[4], vv[5], vv[8], vv[9], vv[10], vv[11],
                         vv[12], vv[13], vv[14], vv[15], vv[20], vv[16], args_i[_op.SH_TWO] != 0, use_tcmt != 0,
                         m_orders, False, vv[22], vv[23], vv[24], vv[21], vv[25], vv[26])


# ---------------------------------------------------------------------------------------------- world block
# The tick (docs/handoff/perf.md P3, R2-8, R2-11). A call that passes the KParams / KState tuples (~270 arrays) copies
# every array struct (~0.4 us), so the tuples are taken once per block: _world_block takes them, makes this world's
# views, tables and scratch arrays (it allocates; it runs with the numba runtime), and passes them to _world_loop,
# the tick loop, whose helpers all take arrays (the controller edge, the auto-policy decision and the oracle
# included: _release_a, _auto_h_a, _oracle_a; the tuple forms release_row, _auto_h, _oracle remain for callers
# outside the kernel). _world_loop and the hot leaves it calls are compiled without the numba runtime
# (_nrt=False): a function with the runtime increfs and decrefs every array it passes to a call, ~500 runtime calls
# per tick here. Rules for a runtime-free function (tests/verification/test_kernel_parity.py
# test_kernel_releases_every_array): it must not allocate (numba refuses to compile that); it must not call a
# function that returns arrays (it would never release that reference; use inline="always" view helpers such as
# _sh_blocks, _sh_out_blocks, optics._sh_core / _sh_outputs / selfheat_views); and a function with the runtime that
# it calls must not make views of its arguments (numba 0.68 then leaks one reference per block, measured): such a
# callee is made runtime-free too, or takes the views from its caller.
@njit(cache=True)
def _world_block(step_fn, fast_fn, b, k0, n_ticks, mode, resume, P, S):
    """Advance world b from tick k0 by up to n_ticks ticks (module docstring). Returns the ticks advanced. The setup
    of the block (views, tables, scratch), then :func:`_world_loop`.

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
    th_sums = np.empty(n_r)
    th_pcross = np.empty(n_r)
    # sources without input whose lags are all +0.0 stay so and are skipped while that lasts (perf.md R2-5)
    th_dead = np.empty(P.th_src_scale.shape[0], np.bool_)
    th_ndead = _th_dead_init(x_th, P.th_lag_src, P.th_lag_kind, P.th_src_scale.shape[0], th_dead)
    th_exo_buf = np.empty(2 * th_idx_exo.shape[0], np.int64)
    th_obs_end = np.empty(P.th_obs_ptr.shape[0] - 1, np.int64)
    th_nexo = _th_live(th_dead, P.th_lag_src, P.th_obs_ptr, th_idx_exo, th_exo_buf, th_obs_end)
    th_exo = th_exo_buf[:2 * th_nexo]
    th_hs_max = 0
    for n in range(n_r):
        th_hs_max = max(th_hs_max, P.th_obs_ptr[P.th_heater_obs[n] + 1] - P.th_obs_ptr[P.th_heater_obs[n]])
    th_hs_tab = np.empty(2 * th_hs_max * n_r, np.int64)
    th_hln = _heater_split_fill(P.th_obs_ptr, th_obs_end, P.th_lag_src, P.th_heater_obs, P.th_heater_src, th_hs_tab)
    # per-lag coefficient tables (a, c0, c1, w) per substep power, index pairs, source inputs side by side (R2-9)
    th_coef = np.empty((th_npow, 4 * th_a.shape[1]))
    th_ms_cross, th_opt_tab = _th_tables(th_idx_cross, P.th_opt_lags, P.th_lag_src, P.th_lag_obs, P.th_src_ring)
    th_uu = np.empty(2 * u0.shape[0])
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
    en_b = np.empty(n_l, np.bool_)                 # laser line enable of the step (no allocation per tick)
    truth_ws = np.empty((3, n_r))                  # scratch of _truth_a
    # controller edge: this world's controller state, obs/act vectors and pipeline
    c_pi = _row(S.c_pi, b)
    c_pf = _row(S.c_pf, b)
    c_si = S.c_si[b]
    c_sf = S.c_sf[b]
    c_obs = S.obs[b]
    c_act = S.act[b]
    c_sat = S.sat[b]
    pipe_vals = S.pipe_vals[b]
    pipe_ticks = S.pipe_ticks[b]
    pipe_meta = S.pipe_meta[b]
    key_sched = P.key_sched[b]
    rec_db = False                                 # a record reads er_db / il_db (else Truth skips their log10)
    for r in range(P.rec_i.shape[0]):
        if P.rec_i[r, 0] == TR_ER or P.rec_i[r, 0] == TR_IL:
            rec_db = True
    return _world_loop(step_fn, fast_fn, b, k0, n_ticks, mode, resume, P, S, n_r, n_l, rf, ri, r8, rbb, lf, wi, wf,
                       tv, args_f, args_i, p_in, p1, tref, s0v, betav, phi0r, phi1r, a0, a1, e0, e1, k2t, k2d, trt,
                       dpl, out_rl, out_r, o_fwhm, fwhm_b, v0, v1, vp, sc_comp_f, sc_tab, sc_keys, sc_cache,
                       o_creset, o_en, o_df, o_pd, o_loss, o_dn, o_td, o_sw, o_de, o_bo, o_bv, o_fd, o_fs, o_fa,
                       o_fg, o_ag, o_tb, o_xpu, swing, tin, rp, jt_dn, jt_al, jt_c, jt_rs, jt_cum, rp_fill, rp_len,
                       ph_ws, jx, u0, u1, x_th, t_obs, th_g, th_a, th_c0, th_c1, th_idx_exo, th_idx_ax, th_unit,
                       th_hs_lo, th_hs_hi, th_w, th_gobs, th_gsplit, th_ok, th_pre, th_sums, th_pcross, th_dead,
                       th_ndead, th_exo_buf, th_obs_end, th_nexo, th_exo, th_hs_tab, th_hln, th_coef, th_ms_cross,
                       th_opt_tab, th_uu, fact, bi_code_max, bi_vmin, bi_vmax, bi_bit_err, bi_bow, bi_noise, bi_key,
                       bi_lat, bi_fifo, bi_slew, bi_settle, he_full, he_law, he_bit_err, he_bow, he_lat, he_fifo,
                       he_slew, he_settle, assign, r2l, link_t, oma_sens, nts, ts_in, mo_lat, mo_pw, en_b,
                       truth_ws, c_pi, c_pf, c_si, c_sf, c_obs, c_act, c_sat, pipe_vals, pipe_ticks, pipe_meta,
                       key_sched, rec_db)


@njit(cache=True, _nrt=False)
def _world_loop(step_fn, fast_fn, b, k0, n_ticks, mode, resume, P, S, n_r, n_l, rf, ri, r8, rbb, lf, wi, wf, tv,
                args_f, args_i, p_in, p1, tref, s0v, betav, phi0r, phi1r, a0, a1, e0, e1, k2t, k2d, trt, dpl, out_rl,
                out_r, o_fwhm, fwhm_b, v0, v1, vp, sc_comp_f, sc_tab, sc_keys, sc_cache, o_creset, o_en, o_df, o_pd,
                o_loss, o_dn, o_td, o_sw, o_de, o_bo, o_bv, o_fd, o_fs, o_fa, o_fg, o_ag, o_tb, o_xpu, swing, tin,
                rp, jt_dn, jt_al, jt_c, jt_rs, jt_cum, rp_fill, rp_len, ph_ws, jx, u0, u1, x_th, t_obs, th_g, th_a,
                th_c0, th_c1, th_idx_exo, th_idx_ax, th_unit, th_hs_lo, th_hs_hi, th_w, th_gobs, th_gsplit, th_ok,
                th_pre, th_sums, th_pcross, th_dead, th_ndead, th_exo_buf, th_obs_end, th_nexo, th_exo, th_hs_tab,
                th_hln, th_coef, th_ms_cross, th_opt_tab, th_uu, fact, bi_code_max, bi_vmin, bi_vmax, bi_bit_err,
                bi_bow, bi_noise, bi_key, bi_lat, bi_fifo, bi_slew, bi_settle, he_full, he_law, he_bit_err, he_bow,
                he_lat, he_fifo, he_slew, he_settle, assign, r2l, link_t, oma_sens, nts, ts_in, mo_lat, mo_pw, en_b,
                truth_ws, c_pi, c_pf, c_si, c_sf, c_obs, c_act, c_sat, pipe_vals, pipe_ticks, pipe_meta,
                key_sched, rec_db):
    """The tick loop of :func:`_world_block` (its setup passes every view, table and scratch array). Compiled
    without the numba runtime: a function with the runtime increfs and decrefs every array it passes to a call
    (~500 runtime calls per tick here); this one creates views and calls leaves without any (it allocates
    nothing; the setup does). docs/handoff/perf.md R2-11."""
    k = k0
    k_end = k0 + n_ticks
    first = True
    while k < k_end:
        # --- controller edge (07 P0-6 event order steps 1-4; once per controller period), on this world's arrays:
        #     a call that takes the KParams/KState tuples copies ~270 array structs (~0.4 us each;
        #     docs/handoff/perf.md R2-8)
        if first and resume != 0:
            # Python path: obs of edge k was packed by the previous call and the Python step wrote S.act
            if P.schedule != 0:
                _rr_apply_a(c_act, S.act_tmp[b], P.act_map, _active_ring(wi, P.schedule, n_r))
            _push_act_a(c_act, P.act_map, c_sat, pipe_vals, pipe_ticks, pipe_meta, wi, k, P.ctrl_latency)
            _advance_edge_a(wi, P.phase, P.period, P.jitter, key_sched)
        else:
            if P.fast_on != 0:
                _fast_tick_a(fast_fn, k, n_r, mo_lat, S.fobs[b], ri, S.mo_fv[b], S.mo_fr[b], S.mo_fm[b], c_pi, c_pf,
                             c_si, c_sf, fact, wi)
            if k == wi[WI_EDGE_NEXT]:
                _release_a(k, n_r, mo_lat, S.mo_fv[b], S.mo_fr[b], S.mo_fm[b], S.mo_samples[b], S.mo_ticks[b], rf, ri,
                           rbb, wi, S.ts_fv[b], S.ts_fr[b], S.ts_fm[b], S.ts_latest[b], S.ts_new[b], P.rb_on,
                           S.rb_fv[b], S.rb_fr[b], S.rb_fm[b], S.dp_fv[b], S.dp_fr[b], S.dp_fm[b], args_f, args_i,
                           P.mo_pr[b], mo_pw, P.mo_pga, P.mo_shot, P.dp_bs_tau, P.dp_bs_stia, P.dp_rin[b],
                           P.dp_win_bits, P.dp_mode, P.dp_words, P.dp_pre, P.dp_period, P.dp_key_dp[b], c_obs,
                           P.obs_map, _active_ring(wi, P.schedule, n_r))
                if mode != MODE_OPEN and P.schedule != 0:
                    # round robin: the held command of the edge, before a controller reset or step changes it
                    _rr_save_a(c_act, S.act_tmp[b])
                if mode == MODE_PYTHON:
                    wi[WI_PENDING_EDGE] = k
                    return k - k0
                if mode == MODE_COMPILED:
                    if wi[WI_CTRL_RESET] != 0:                     # scenario fault.controller_reset: init state
                        for i in range(c_si.shape[0]):
                            c_si[i] = S.c_si0[b, i]
                        for i in range(c_sf.shape[0]):
                            c_sf[i] = S.c_sf0[b, i]
                        _sync_act_a(c_act, P.act_map, ri[RI_HE_REG], ri[RI_BI_REG])
                        wi[WI_CTRL_RESET] = 0
                    step_fn(c_pi, c_pf, c_si, c_sf, c_obs, c_act)
                    if P.schedule != 0:
                        _rr_apply_a(c_act, S.act_tmp[b], P.act_map, _active_ring(wi, P.schedule, n_r))
                    _push_act_a(c_act, P.act_map, c_sat, pipe_vals, pipe_ticks, pipe_meta, wi, k, P.ctrl_latency)
                _advance_edge_a(wi, P.phase, P.period, P.jitter, key_sched)
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
        for l in range(n_l):
            en_b[l] = lf[LF_EN_B, l] > 0.5
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
            h = _auto_h_a(k, k_end, n_r, P.fast_on, P.au_max_pow, P.reset_protocol, P.ramp_ticks, P.oracle_on,
                          P.he_ripple, P.bi_noise_per, bi_noise, P.au_max_det, P.au_pabs_tol, wi, wf, ri, r8, rf, rbb,
                          pipe_meta, pipe_ticks, he_fifo, bi_fifo, P.sc_sig_i, P.sc_sig_f, P.sc_def, P.sc_comp,
                          P.sc_comp_i, sc_comp_f, sc_tab, sc_keys, sc_cache, vp, v0, v1, S.sc_va[b], P.rec_i,
                          S.rec_idx[b], S.rec_cnt[b], P.mp_i, S.m_ai[b])
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
            _th_coef(ta, tc0, tc1, th_w[pw2], th_coef[pw2])
            th_ok[pw2] = True
        tcoef = th_coef[pw2]
        for n in range(n_r):
            rf[RF_LAMRES_PREV, n] = tv[TR_LAM_RES, n]
            rf[RF_PABS_PREV, n] = rf[RF_P_ABS, n]
        for i in range(P.th_agg_src.shape[0]):
            u1[P.th_agg_src[i]] = v1[o_ag + i]
        t_base_end = v1[o_tb] + v1[o_xpu]
        wf[WF_T_BASE1] = t_base_end
        # --- heater: implicit TCR split on the heater-resistor observer (07 P1-2), then the DAC path
        if th_ndead > 0 and _th_wake(th_dead, u0, u1):
            # a dead source received an input: its lags (still exactly +0.0) are live again from here on
            th_ndead = 0
            for s in range(th_dead.shape[0]):
                if th_dead[s]:
                    th_ndead += 1
            th_nexo = _th_live(th_dead, P.th_lag_src, P.th_obs_ptr, th_idx_exo, th_exo_buf, th_obs_end)
            th_exo = th_exo_buf[:2 * th_nexo]
            th_hln = _heater_split_fill(P.th_obs_ptr, th_obs_end, P.th_lag_src, P.th_heater_obs, P.th_heater_src,
                                        th_hs_tab)
        _uu_fill(u0, u1, th_uu)
        _heater_split_all(x_th, ta, tc0, tc1, P.th_lag_src, P.th_obs_ptr, th_obs_end, P.th_heater_obs,
                          P.th_heater_src, th_hs_lo, th_hs_hi, th_unit, u0, u1, P.th_src_scale, th_hln, th_hs_tab,
                          tcoef, th_uu, th_sums, th_pre)
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
        # the last step of the block (the block returns after it): the optics commit every output there (Plant.truth
        # and the parity read them after the block); before it only the outputs a tick reads (perf.md R2-3)
        last_step = k + h >= k_end or (mode == MODE_PYTHON and k + h >= wi[WI_EDGE_NEXT])
        # --- implicit thermal + self-heating step (07 P0-1); the optics walk runs inside (commit_fn)
        if th_ndead > 0 and _th_wake(th_dead, u0, u1):
            th_ndead = 0
            for s in range(th_dead.shape[0]):
                if th_dead[s]:
                    th_ndead += 1
            th_nexo = _th_live(th_dead, P.th_lag_src, P.th_obs_ptr, th_idx_exo, th_exo_buf, th_obs_end)
            th_exo = th_exo_buf[:2 * th_nexo]
            th_hln = _heater_split_fill(P.th_obs_ptr, th_obs_end, P.th_lag_src, P.th_heater_obs, P.th_heater_src,
                                        th_hs_tab)
        if P.self_heating != 0:
            _uu_fill(u0, u1, th_uu)
            nf = _implicit_step_eng(x_th, tcoef, th_uu, th_exo, th_ms_cross, th_idx_ax, th_opt_tab, P.th_lag_obs,
                                    P.th_obs_ptr, th_obs_end, P.th_ring_obs, P.bus_order, P.th_src_scale, th_unit,
                                    t_base_end, rf[RF_P_PREV], rf[RF_T_BRANCH], args_f, args_i, rf[RF_PMAX],
                                    rf[RF_D2P], P.sh_max_iter, P.sh_tol, n_outer, th_g, t_obs, rf[RF_P_NEW],
                                    rf[RF_T_NEW], ri[RI_N_ITER], th_gobs[pw2], last_step, th_pcross)
            wi[WI_N_FAIL] += nf
            for n in range(n_r):
                wi[WI_SH_ITER] += ri[RI_N_ITER, n]
                rf[RF_P_PREV, n] = rf[RF_P_NEW, n]
                rf[RF_T_BRANCH, n] = rf[RF_T_NEW, n]
                rf[RF_T_RING, n] = rf[RF_T_NEW, n]
        else:
            _bus_walk(args_f, args_i, P.bus_order, P.use_tcmt, P.m_orders)
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
            # the ADC reference temperature (ring.monitor.adc_ref_site): the package base at the end of the interval
            # (a converter on its own die beside the PIC), or the ring's PD observer
            rf[RF_T_ADC, n] = rf[RF_T_PD, n] if P.mo_adc_site == ADC_REF_MONITOR_PD else wf[WF_T_BASE1]
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
            monitor_tick_row(kk, rf[RF_P_PD], rf[RF_P_BG], rf[RF_T_PD], rf[RF_T_ADC], ri[RI_PGA], r8[R8_FAULT],
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
        _truth_a(n_r, n_l, rf, ri, rbb, tv, lf, S.lt[b], t_base_end, args_f, args_i, P.mo_pw[b], P.mo_pr[b], rp, jx,
                 fwhm_b, P.wrong_window, P.assign_mode, rec_db or last_step, True, truth_ws)
        # --- oracle (on its period)
        if P.oracle_on != 0 and k + h > wi[WI_ORACLE_NEXT]:
            _oracle_a(n_r, rf, ri, args_f, args_i, S.or_phi[b], S.or_k2t[b], S.or_k2d[b], S.or_a[b], S.or_trt[b],
                      S.or_psi[b], S.or_res0[b], S.or_res1[b], S.or_cw[b], S.or_fw[b], lf, rp, jx, fwhm_b,
                      P.z_opt_self[b], P.use_tcmt, P.m_orders, P.lock_ref_mode, S.m_ref[b])
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
