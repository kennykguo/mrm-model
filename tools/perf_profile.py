"""Per-component cost of one plant tick (docs/handoff/perf.md): each kernel component in a compiled loop.

    .venv/bin/python tools/memguard.py --heavy -- .venv/bin/python tools/perf_profile.py [--locked] [--n 2000]

Builds the g26 B = 1 plant of tools/perf_bench.py (``--locked``: settled at the quasi-static lock codes), advances it
300 ticks with the real kernel, then times each component on the evolving state in its own compiled loop (n calls,
median of 5 repeats): the leaves of the interval inputs (scenario, laser, phase.ring_optics_levels_row,
selfheat_bounds, swing capacitance, data heat), the heater DAC path, the implicit self-heating step, one ring
evaluation (eng_pabs), the sensors of a tick, truth_row, the oracle, the metrics and the controller edge. The loops
re-run a component on a state that the other components do not advance, so a number is the cost of the component,
not a simulation. Compile time of the harness: several minutes (it compiles the kernel components again).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from mrm.plant import optics as _op  # noqa: E402
from mrm.plant import phase as _ph  # noqa: E402
from mrm.plant import electrical as _el  # noqa: E402
from mrm.plant.laser import laser_tick_row  # noqa: E402
from mrm.plant.sensing import monitor_tick_row, temp_tick_row, readback_tick_row  # noqa: E402
from mrm.plant.datapath import errors_tick_ber_row, pattern_tick_row  # noqa: E402
from mrm.sim import kernel as K  # noqa: E402
from mrm.sim.kernel import _row  # noqa: E402
from mrm.sim.metrics import metrics_tick_row  # noqa: E402
import perf_bench  # noqa: E402







@njit(cache=True)
def p_scen(P, S, b, k0, n):
    for i in range(n):
        K._scen_eval(P, S, b, k0 + i + 1, S.sc_v1[b])


@njit(cache=True)
def p_laser(P, S, b, k0, n):
    lf = S.lf[b]
    en_b = lf[K.LF_EN_B] > 0.5
    for i in range(n):
        k = k0 + i
        laser_tick_row(k, P.dt, P.las_tcomp, P.las_grid_s, P.las_depth, P.las_f0[b], P.las_p0[b], P.las_frate[b],
                       P.las_prate[b], P.las_frw[b], P.las_prw[b], P.las_rho_f[b], P.las_rho_p[b], P.las_module,
                       P.las_nmod, P.las_rin[b], en_b, lf[K.LF_DF], lf[K.LF_PS], lf[K.LF_TUNE_HZ],
                       P.las_key_drift[b], P.las_key_rin[b], S.las_cache_j[b], S.las_wlo[b], S.las_whi[b],
                       lf[K.LF_F], lf[K.LF_P])



@njit(cache=True)
def p_pattern(P, S, b, k0, n):
    rf = S.rf[b]
    ri = S.ri[b]
    rbb = S.rb[b]
    for i in range(n):
        pattern_tick_row(k0 + i, P.dp_mode, P.dp_bpt_i, P.dp_bpt_f, rf[K.RF_DENS], rbb[K.RB_DP_ON], P.dp_words,
                         P.dp_pre, P.dp_period, P.dp_key_pat[b], ri[K.RI_DP_BITPOS], ri[K.RI_DP_N1], ri[K.RI_DP_NB],
                         rf[K.RF_RHO])


@njit(cache=True)
def p_optics_row(P, S, b, n):
    v = _op.selfheat_views(S.sh_args_f[b], P.sh_args_i[b])
    rf = S.rf[b]
    r8 = S.r8[b]
    lf = S.lf[b]
    rp = P.ph_ring[b]
    tb = b if P.jt_dn.shape[0] > 1 else 0
    jx = S.jx[b]
    for i in range(n):
        _ph.ring_optics_levels_row(rp, lf[K.LF_LAM], rf[K.RF_T_RING], rf[K.RF_T_HEATER], rf[K.RF_V_LO],
                                   rf[K.RF_V_HI], P.jt_grid, P.jt_dn[tb], P.jt_al[tb], P.jt_c[tb], P.jt_rs[tb], v[4],
                                   v[5], v[8], v[9], v[10], v[11], v[15], jx[0], jx[1], jx[2], r8[K.R8_JFLAGS])


@njit(cache=True)
def p_bounds(P, S, b, n):
    rf = S.rf[b]
    for i in range(n):
        _op.sh_bounds(S.sh_args_f[b], P.sh_args_i[b], P.dt_max_k, rf[K.RF_PMAX], rf[K.RF_D2P], rf[K.RF_DPDT])


@njit(cache=True)
def p_precompute(P, S, b, n):
    for i in range(n):
        K.eng_precompute(S.sh_args_f[b], P.sh_args_i[b])


@njit(cache=True)
def p_swing_heat(P, S, b, n):
    rf = S.rf[b]
    rp = P.ph_ring[b]
    tb = b if P.jt_dn.shape[0] > 1 else 0
    for i in range(n):
        K.swing_capacitance_row(rf[K.RF_V_LO], rf[K.RF_V_HI], P.jt_grid, P.jt_c[tb], P.jt_cum[tb], rp[:, _ph.RP_FILL],
                                rp[:, _ph.RP_LEN], rf[K.RF_CJ])
        _el.data_activity_heat_row(rf[K.RF_CJ], rf[K.RF_SWING_V], P.dp_bitrate[b], rf[K.RF_TD], rf[K.RF_EN],
                                   rf[K.RF_RS], P.drv_r_out[b:b + 1], P.drv_p_static[b:b + 1], P.drv_e_tr[b:b + 1],
                                   0, rf[K.RF_PJ], rf[K.RF_PD])



def implicit_tables(P, S, b):
    """The per-block tables _world_block builds for _implicit_step_eng (fixed policy, substep power 0)."""
    ta = _row(P.th_a, b)[0]
    tc0 = _row(P.th_c0, b)[0]
    tc1 = _row(P.th_c1, b)[0]
    idx_exo, idx_cross, idx_ax, unit, hs_lo, hs_hi = K._th_struct(P.th_lag_src, P.th_lag_kind, P.th_obs_ptr,
                                                                  P.th_heater_obs, P.th_heater_src, P.th_src_scale)
    m_tot, n_obs, n_src = ta.shape[0], P.th_obs_ptr.shape[0] - 1, P.th_src_scale.shape[0]
    w, gobs, gsplit = np.empty(m_tot), np.empty(n_obs), np.empty(P.n_rings)
    K._th_gains(tc0, tc1, P.th_lag_src, P.th_lag_kind, P.th_obs_ptr, P.th_heater_obs, P.th_heater_src,
                P.th_src_scale, w, gobs, gsplit)
    coef = np.empty(4 * m_tot)
    K._th_coef(ta, tc0, tc1, w, coef)
    ms_cross, opt_tab = K._th_tables(idx_cross, P.th_opt_lags, P.th_lag_src, P.th_lag_obs, P.th_src_ring)
    dead = np.empty(n_src, np.bool_)
    K._th_dead_init(S.th_x[b], P.th_lag_src, P.th_lag_kind, n_src, dead)
    ms_exo = np.empty(2 * idx_exo.shape[0], np.int64)
    obs_end = np.empty(n_obs, np.int64)
    n_exo = K._th_live(dead, P.th_lag_src, P.th_obs_ptr, idx_exo, ms_exo, obs_end)
    uu = np.empty(2 * n_src)
    K._uu_fill(S.th_u0[b], S.th_u1[b], uu)
    return coef, uu, ms_exo[:2 * n_exo].copy(), ms_cross, idx_ax, opt_tab, obs_end, gobs, unit


@njit(cache=True)
def p_implicit(P, S, b, coef, uu, ms_exo, ms_cross, idx_ax, opt_tab, obs_end, gobs, unit, n):
    rf = S.rf[b]
    ri = S.ri[b]
    x = S.th_x[b]
    p_cross = np.empty(P.n_rings)
    tot = 0
    for i in range(n):
        K._implicit_step_eng(x, coef, uu, ms_exo, ms_cross, idx_ax, opt_tab, P.th_lag_obs, P.th_obs_ptr, obs_end,
                             P.th_ring_obs, P.bus_order, P.th_src_scale, unit, S.wf[b, K.WF_T_BASE1],
                             rf[K.RF_P_PREV], rf[K.RF_T_BRANCH], S.sh_args_f[b], P.sh_args_i[b], rf[K.RF_PMAX],
                             rf[K.RF_D2P], P.sh_max_iter, P.sh_tol, P.sh_n_outer, S.th_g[b], S.th_t_obs[b],
                             rf[K.RF_P_NEW], rf[K.RF_T_NEW], ri[K.RI_N_ITER], gobs, False, p_cross)
        for m in range(P.n_rings):
            tot += ri[K.RI_N_ITER, m]
            rf[K.RF_P_PREV, m] = rf[K.RF_P_NEW, m]
            rf[K.RF_T_BRANCH, m] = rf[K.RF_T_NEW, m]
    return tot


@njit(cache=True)
def p_eval(P, S, b, n):
    rf = S.rf[b]
    acc = 0.0
    for i in range(n):
        for k in range(P.n_rings):
            p, d = K.eng_pabs(rf[K.RF_T_RING, k] + 1e-4 * (i & 1), k, S.sh_args_f[b], P.sh_args_i[b])
            acc += p
    return acc


@njit(cache=True)
def p_sensors(P, S, b, k0, n):
    rf = S.rf[b]
    ri = S.ri[b]
    r8 = S.r8[b]
    rbb = S.rb[b]
    nts = S.ts_in.shape[1]
    for i in range(n):
        kk = k0 + i
        monitor_tick_row(kk, rf[K.RF_P_PD], rf[K.RF_P_BG], rf[K.RF_T_PD], rf[K.RF_T_PD], ri[K.RI_PGA], r8[K.R8_FAULT],
                         P.mo_pr[b], P.mo_pw[b], _row(P.mo_lat, b), P.mo_phase, P.mo_pga, P.mo_fl[b], P.mo_afe,
                         P.mo_win, P.mo_period, P.mo_cmax, P.mo_shot, P.mo_stuck, P.mo_slot, P.mo_key_adc[b],
                         P.mo_key_shot[b], P.mo_key_dnl[b], S.mo_sf[b], S.mo_si[b], S.mo_fx[b], S.mo_fc[b],
                         S.mo_fv[b], S.mo_fr[b], S.mo_fm[b], r8[K.R8_MO_SAMPLED])
        S.mo_fm[b, :, 1] = 0                 # keep the FIFO from filling (no release in this loop)
        errors_tick_ber_row(kk, rf[K.RF_BER], ri[K.RI_DP_NB], P.dp_err_period, P.dp_err_lat[b], P.dp_key_dp[b],
                            rf[K.RF_DP_LAM], S.dp_fv[b], S.dp_fr[b], S.dp_fm[b], rbb[K.RB_ERR_NEW])
        S.dp_fm[b, :, 1] = 0
        if nts > 0:
            temp_tick_row(kk, S.ts_in[b], P.ts_tp[b], P.ts_period, P.ts_lat[b], P.ts_cmax, P.ts_fl[b], P.ts_key[b],
                          S.ts_st[b], S.ts_fx[b], S.ts_fc[b], S.ts_primed[b], S.ts_fv[b], S.ts_fr[b], S.ts_fm[b],
                          S.ts_sampled[b])
            S.ts_fm[b, :, 1] = 0
        if P.rb_on != 0:
            readback_tick_row(kk, rf[K.RF_HE_V], rf[K.RF_HE_I], P.rb_lsb_v[b], P.rb_lsb_i[b], P.rb_noise[b],
                              P.rb_period, P.rb_lat[b], P.rb_cmax, P.rb_avg, P.rb_key[b], S.rb_st[b], S.rb_fv[b],
                              S.rb_fr[b], S.rb_fm[b], r8[K.R8_RB_SAMPLED])
            S.rb_fm[b, :, 1] = 0


@njit(cache=True)
def p_monitor(P, S, b, k0, n):
    rf = S.rf[b]
    ri = S.ri[b]
    r8 = S.r8[b]
    for i in range(n):
        monitor_tick_row(k0 + i, rf[K.RF_P_PD], rf[K.RF_P_BG], rf[K.RF_T_PD], rf[K.RF_T_PD], ri[K.RI_PGA],
                         r8[K.R8_FAULT], P.mo_pr[b], P.mo_pw[b], _row(P.mo_lat, b), P.mo_phase, P.mo_pga, P.mo_fl[b],
                         P.mo_afe, P.mo_win, P.mo_period, P.mo_cmax, P.mo_shot, P.mo_stuck, P.mo_slot,
                         P.mo_key_adc[b], P.mo_key_shot[b], P.mo_key_dnl[b], S.mo_sf[b], S.mo_si[b], S.mo_fx[b],
                         S.mo_fc[b], S.mo_fv[b], S.mo_fr[b], S.mo_fm[b], r8[K.R8_MO_SAMPLED])
        S.mo_fm[b, :, 1] = 0


@njit(cache=True)
def p_truth(P, S, b, n):
    for i in range(n):
        K.truth_row(P, S, b)


@njit(cache=True)
def p_oracle(P, S, b, n):
    for i in range(n):
        K._oracle(P, S, b)


@njit(cache=True)
def p_metrics(P, S, b, k0, n):
    rf = S.rf[b]
    for i in range(n):
        k = k0 + i
        metrics_tick_row(k, 1, P.mp_f, P.mp_i, rf[K.RF_DET], S.m_ref[b], S.ri[b, K.RI_ORDER], S.rb[b, K.RB_WRONG],
                         S.tv[b, K.TR_RELOMA], rf[K.RF_ORACLE], rf[K.RF_ORACLE_UNR], rf[K.RF_HE_P], rf[K.RF_P_ABS],
                         S.ri[b, K.RI_HE_REG], S.rb[b, K.RB_POPPED], rf[K.RF_BER], rf[K.RF_BITS], S.m_af[b],
                         S.m_ai[b], S.m_hist[b], S.m_runhist[b])


@njit(cache=True)
def p_release(P, S, b, k0, n):
    for i in range(n):
        K.release_row(P, S, b, k0 + 10 * i)


def timeit(fn, args, n, reps=5):
    fn(*args[:-1], 2) if False else None
    out = []
    for _ in range(reps):
        t = time.perf_counter()
        fn(*args)
        out.append((time.perf_counter() - t) / n * 1e6)
    return float(np.median(out)), float(np.min(out))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--locked", action="store_true")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pl = perf_bench._plant(1, {}, locked=a.locked)
    fn = pl._fn()
    done = np.zeros(1, np.int64)
    fn(K.null_step, K.null_step, pl.P, pl.S, np.int64(pl.tick), np.int64(300), np.int64(0), np.int64(0), done)
    pl.tick += int(done[0])
    P, S = pl.P, pl.S
    n = int(a.n)
    k0 = np.int64(pl.tick)
    b = np.int64(0)
    t0 = time.perf_counter()
    # the real kernel on the same plant (wall per tick, open loop, edges included)
    walls = []
    for _ in range(5):
        it0 = int(S.wi[0, K.WI_SH_ITER])
        t = time.perf_counter()
        fn(K.null_step, K.null_step, P, S, np.int64(pl.tick), np.int64(n), np.int64(0), np.int64(0), done)
        walls.append((time.perf_counter() - t) / n * 1e6)
        pl.tick += int(done[0])
        it1 = int(S.wi[0, K.WI_SH_ITER])
    print(f"{'kernel run_block (full tick incl. edges, oracle)':55s} {np.median(walls):9.3f} us  (min {min(walls):.3f})"
          f"  newton it/ring/tick {(it1 - it0) / (n * pl.N):.3f}", flush=True)
    k0 = np.int64(pl.tick)
    cases = [
        ("  scenario eval (k+1)", p_scen, (P, S, b, k0, n)),
        ("  laser_tick_row", p_laser, (P, S, b, k0, n)),
        ("  pattern_tick_row", p_pattern, (P, S, b, k0, n)),
        ("  phase.ring_optics_levels_row", p_optics_row, (P, S, b, n)),
        ("  eng_precompute (substep constants)", p_precompute, (P, S, b, n)),
        ("  sh_bounds", p_bounds, (P, S, b, n)),
        ("  swing capacitance + data heat", p_swing_heat, (P, S, b, n)),
        ("implicit self-heating step (bank)", p_implicit, (P, S, b, *implicit_tables(P, S, b), n)),
        ("ring evaluation eng_pabs x N rings", p_eval, (P, S, b, n)),
        ("sensors of a tick (monitor, errors, temp, readback)", p_sensors, (P, S, b, k0, n)),
        ("  monitor_tick_row", p_monitor, (P, S, b, k0, n)),
        ("truth_row", p_truth, (P, S, b, n)),
        ("oracle (per call; 1 per 1000 ticks)", p_oracle, (P, S, b, max(n // 20, 10))),
        ("metrics_tick_row", p_metrics, (P, S, b, k0, n)),
        ("controller edge release_row (per edge)", p_release, (P, S, b, k0, max(n // 10, 10))),
    ]
    only = [s for s in a.only.split(",") if s]
    rows = []
    for name, f, args in cases:
        if only and not any(o in name for o in only):
            continue
        nn = args[-1]
        f(*args[:-1], 1)
        it0 = int(S.wi[0, K.WI_SH_ITER])
        med, mn = timeit(f, args, nn)
        it1 = int(S.wi[0, K.WI_SH_ITER])
        if f is p_implicit:
            it = f(*args[:-1], nn)
        rows.append((name, med, mn))
        extra = f"  newton it/ring/tick {(it1 - it0) / (5 * nn * P.n_rings):.3f}" if it1 > it0 else ""
        print(f"{name:55s} {med:9.3f} us  (min {mn:.3f}){extra}", flush=True)
    print(f"[perf_profile] load {os.getloadavg()[0]:.2f}, compile+run {time.perf_counter() - t0:.0f} s, "
          f"locked={a.locked}")


if __name__ == "__main__":
    main()
