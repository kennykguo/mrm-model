"""Measure the cost of each component of one plant tick, each one timed in its own compiled loop.

    .venv/bin/python tools/memguard.py --heavy -- .venv/bin/python tools/perf_profile.py [--locked] [--n 2000]

Read this first. The tool builds the g26 B = 1 plant of tools/perf_bench.py (``--locked``: settled at the
steady-state lock codes) and advances it 300 ticks with the real kernel. It then times the full kernel tick, and
after that each component on the evolving state, in its own compiled loop (n calls, median of 5 repeats):

- the inputs of an interval: scenario, laser, data pattern (pattern_tick_row), phase.evaluate_binary_ring_optical_inputs, the
  substep constants (cache_refresh_engine_optics), the self-heating bounds, the swing capacitance and the data heat;
- the implicit self-heating step of the bank, and eng_pabs on every ring (one ring evaluation each);
- the sensors of a tick (monitor, errors, temperature, readback) and the monitor alone, truth_row, the oracle, the
  metrics and the controller edge (release_row).

Each loop runs one component again and again on a state that the other components do not advance. So a number is
the cost of the component, not a simulation. The harness compiles the kernel components again: several minutes.
``--only`` keeps the cases whose label contains one of the comma-separated words.
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

from mrm.plant import optics  # noqa: E402
from mrm.plant import phase  # noqa: E402
from mrm.plant import electrical  # noqa: E402
from mrm.plant.laser import laser_tick_row  # noqa: E402
from mrm.plant.sensing import monitor_tick_row, temp_tick_row, readback_tick_row  # noqa: E402
from mrm.plant.datapath import errors_tick_ber_row, pattern_tick_row  # noqa: E402
from mrm.sim import kernel  # noqa: E402
from mrm.sim.kernel import _world_row  # noqa: E402
from mrm.sim.metrics import metrics_tick_row  # noqa: E402
import perf_bench  # noqa: E402

WARMUP_TICKS = 300          # open-loop ticks with the real kernel before any timing
REPEATS = 5                 # timed repeats per case (the report gives their median and minimum)


# ------------------------------------------------------------------------------------------- one compiled loop per part
# Each p_* function runs one part of the tick n times for world b (P: KParams, S: KState, k0: first tick).

@njit(cache=True)
def p_scen(P, S, b, k0, n):
    for i in range(n):
        kernel._scenario_eval(P, S, b, k0 + i + 1, S.sc_v1[b])


@njit(cache=True)
def p_laser(P, S, b, k0, n):
    lf = S.line_float[b]
    en_b = lf[kernel.LF_EN_B] > 0.5
    for i in range(n):
        k = k0 + i
        laser_tick_row(k, P.dt, P.las_tcomp, P.las_grid_s, P.las_depth, P.las_f0[b], P.las_p0[b], P.las_frate[b],
                       P.las_prate[b], P.las_frw[b], P.las_prw[b], P.las_rho_f[b], P.las_rho_p[b], P.las_module,
                       P.las_nmod, P.las_rin[b], en_b, lf[kernel.LF_DF], lf[kernel.LF_PS], lf[kernel.LF_TUNE_HZ],
                       P.las_key_drift[b], P.las_key_rin[b], S.las_cache_j[b], S.las_wlo[b], S.las_whi[b],
                       lf[kernel.LF_F], lf[kernel.LF_P])


@njit(cache=True)
def p_pattern(P, S, b, k0, n):
    rf = S.ring_float[b]
    ri = S.ring_int[b]
    rbb = S.ring_bool[b]
    for i in range(n):
        pattern_tick_row(k0 + i, P.dp_mode, P.dp_bpt_i, P.dp_bpt_f, rf[kernel.RF_DENS], rbb[kernel.RB_DP_ON],
                         P.dp_words, P.dp_pre, P.dp_period, P.dp_key_pat[b], ri[kernel.RI_DP_BITPOS],
                         ri[kernel.RI_DP_N1], ri[kernel.RI_DP_NB], rf[kernel.RF_RHO])


@njit(cache=True)
def p_optics_row(P, S, b, n):
    v = optics.unpack_self_heating_optical_state(S.selfheat_float[b], P.selfheat_int[b])
    rf = S.ring_float[b]
    r8 = S.ring_int8[b]
    lf = S.line_float[b]
    rp = P.ph_ring[b]
    tb = b if P.jt_dn.shape[0] > 1 else 0
    jx = S.junction[b]
    for i in range(n):
        phase.evaluate_binary_ring_optical_inputs(
            rp,
            lf[kernel.LF_LAM],
            rf[kernel.RF_T_RING],
            rf[kernel.RF_T_HEATER],
            rf[kernel.RF_V_LO],
            rf[kernel.RF_V_HI],
            P.jt_grid,
            P.jt_dn[tb],
            P.jt_al[tb],
            P.jt_c[tb],
            P.jt_rs[tb],
            v[optics.SELF_HEATING_VIEW_INDEX_LEVEL_ZERO_REFERENCE_ROUND_TRIP_PHASE],
            v[optics.SELF_HEATING_VIEW_INDEX_LEVEL_ONE_REFERENCE_ROUND_TRIP_PHASE],
            v[optics.SELF_HEATING_VIEW_INDEX_LEVEL_ZERO_ROUND_TRIP_FIELD_TRANSMISSION],
            v[optics.SELF_HEATING_VIEW_INDEX_LEVEL_ONE_ROUND_TRIP_FIELD_TRANSMISSION],
            v[optics.SELF_HEATING_VIEW_INDEX_LEVEL_ZERO_ABSORBED_SHARE_OF_INTRINSIC_LOSS],
            v[optics.SELF_HEATING_VIEW_INDEX_LEVEL_ONE_ABSORBED_SHARE_OF_INTRINSIC_LOSS],
            v[optics.SELF_HEATING_VIEW_INDEX_ROUND_TRIP_GROUP_DELAY],
            jx[kernel._JUNCTION_SHIFT],
            jx[kernel._JUNCTION_C],
            jx[kernel._JUNCTION_RS],
            r8[kernel.R8_JFLAGS],
        )


@njit(cache=True)
def p_bounds(P, S, b, n):
    rf = S.ring_float[b]
    for i in range(n):
        optics.sh_bounds(S.selfheat_float[b], P.selfheat_int[b], P.dt_max_k, rf[kernel.RF_PMAX], rf[kernel.RF_D2P],
                         rf[kernel.RF_DPDT])


@njit(cache=True)
def p_precompute(P, S, b, n):
    for i in range(n):
        kernel.cache_refresh_engine_optics(S.selfheat_float[b], P.selfheat_int[b])


@njit(cache=True)
def p_swing_heat(P, S, b, n):
    rf = S.ring_float[b]
    rp = P.ph_ring[b]
    tb = b if P.jt_dn.shape[0] > 1 else 0
    for i in range(n):
        kernel.swing_capacitance_row(rf[kernel.RF_V_LO], rf[kernel.RF_V_HI], P.jt_grid, P.jt_c[tb], P.jt_cum[tb],
                                     rp[:, phase.RING_PARAMETER_COLUMN_JUNCTION_FILL_FRACTION], rp[:, phase.RING_PARAMETER_COLUMN_ROUND_TRIP_LENGTH], rf[kernel.RF_CJ])
        electrical.data_activity_heat_row(rf[kernel.RF_CJ], rf[kernel.RF_SWING_V], P.dp_bitrate[b], rf[kernel.RF_TD],
                                          rf[kernel.RF_EN], rf[kernel.RF_RS], P.drv_r_out[b:b + 1],
                                          P.drv_p_static[b:b + 1], P.drv_e_tr[b:b + 1], 0, rf[kernel.RF_PJ],
                                          rf[kernel.RF_PD])


def implicit_tables(P, S, b):
    """The per-block tables that kernel._world_block builds for _implicit_step_eng (fixed policy, substep power 0).

    Returns the arguments of p_implicit after b: (coef, uu, ms_exo, ms_cross, idx_ax, opt_tab, obs_end, gobs, unit)."""
    ta = _world_row(P.th_a, b)[0]
    tc0 = _world_row(P.th_c0, b)[0]
    tc1 = _world_row(P.th_c1, b)[0]
    idx_exo, idx_cross, idx_ax, unit, hs_lo, hs_hi = kernel._lag_lists(P.th_lag_src, P.th_lag_kind, P.th_obs_ptr,
                                                                  P.th_heater_obs, P.th_heater_src, P.th_src_scale)
    m_tot, n_obs, n_src = ta.shape[0], P.th_obs_ptr.shape[0] - 1, P.th_src_scale.shape[0]
    w, gobs, gsplit = np.empty(m_tot), np.empty(n_obs), np.empty(P.n_rings)
    kernel._thermal_gains(tc0, tc1, P.th_lag_src, P.th_lag_kind, P.th_obs_ptr, P.th_heater_obs, P.th_heater_src,
                     P.th_src_scale, w, gobs, gsplit)
    coef = np.empty(4 * m_tot)
    kernel._pack_lag_coefs(ta, tc0, tc1, w, coef)
    ms_cross, opt_tab = kernel._lag_pair_tables(idx_cross, P.th_opt_lags, P.th_lag_src, P.th_lag_obs, P.th_src_ring)
    dead = np.empty(n_src, np.bool_)
    kernel._find_dead_sources(S.th_x[b], P.th_lag_src, P.th_lag_kind, n_src, dead)
    ms_exo = np.empty(2 * idx_exo.shape[0], np.int64)
    obs_end = np.empty(n_obs, np.int64)
    n_exo = kernel._live_lags(dead, P.th_lag_src, P.th_obs_ptr, idx_exo, ms_exo, obs_end)
    uu = np.empty(2 * n_src)
    kernel._pack_source_inputs(S.th_u0[b], S.th_u1[b], uu)
    return coef, uu, ms_exo[:2 * n_exo].copy(), ms_cross, idx_ax, opt_tab, obs_end, gobs, unit


@njit(cache=True)
def p_implicit(P, S, b, coef, uu, ms_exo, ms_cross, idx_ax, opt_tab, obs_end, gobs, unit, n):
    rf = S.ring_float[b]
    ri = S.ring_int[b]
    x = S.th_x[b]
    p_cross = np.empty(P.n_rings)
    tot = 0
    for i in range(n):
        kernel._implicit_step_eng(x, coef, uu, ms_exo, ms_cross, idx_ax, opt_tab, P.th_lag_obs, P.th_obs_ptr, obs_end,
                                  P.th_ring_obs, P.bus_order, P.th_src_scale, unit, S.world_float[b, kernel.WF_T_BASE1],
                                  rf[kernel.RF_P_PREV], rf[kernel.RF_T_BRANCH], S.selfheat_float[b], P.selfheat_int[b],
                                  rf[kernel.RF_PMAX], rf[kernel.RF_D2P], P.sh_max_iter, P.sh_tol, P.sh_n_outer,
                                  S.th_g[b], S.th_t_obs[b], rf[kernel.RF_P_NEW], rf[kernel.RF_T_NEW],
                                  ri[kernel.RI_N_ITER], gobs, False, p_cross)
        for m in range(P.n_rings):
            tot += ri[kernel.RI_N_ITER, m]
            rf[kernel.RF_P_PREV, m] = rf[kernel.RF_P_NEW, m]
            rf[kernel.RF_T_BRANCH, m] = rf[kernel.RF_T_NEW, m]
    return tot


@njit(cache=True)
def p_eval(P, S, b, n):
    rf = S.ring_float[b]
    acc = 0.0
    for i in range(n):
        for k in range(P.n_rings):
            p, d = kernel.eng_pabs(rf[kernel.RF_T_RING, k] + 1e-4 * (i & 1), k, S.selfheat_float[b], P.selfheat_int[b])
            acc += p
    return acc


@njit(cache=True)
def p_sensors(P, S, b, k0, n):
    rf = S.ring_float[b]
    ri = S.ring_int[b]
    r8 = S.ring_int8[b]
    rbb = S.ring_bool[b]
    nts = S.ts_in.shape[1]
    for i in range(n):
        kk = k0 + i
        monitor_tick_row(kk, rf[kernel.RF_P_PD], rf[kernel.RF_P_BG], rf[kernel.RF_T_PD], rf[kernel.RF_T_ADC],
                         ri[kernel.RI_PGA], r8[kernel.R8_FAULT], P.mo_pr[b], P.mo_pw[b], _world_row(P.mo_lat, b),
                         P.mo_phase, P.mo_pga, P.mo_fl[b], P.mo_afe, P.mo_win, P.mo_period, P.mo_cmax, P.mo_shot,
                         P.mo_stuck, P.mo_slot, P.mo_key_adc[b], P.mo_key_shot[b], P.mo_key_dnl[b], S.mo_sf[b],
                         S.mo_si[b], S.mo_fx[b], S.mo_fc[b], S.mo_fv[b], S.mo_fr[b], S.mo_fm[b],
                         r8[kernel.R8_MO_SAMPLED])
        S.mo_fm[b, :, 1] = 0                 # keep the FIFO from filling (no release in this loop)
        errors_tick_ber_row(kk, rf[kernel.RF_BER], ri[kernel.RI_DP_NB], P.dp_err_period, P.dp_err_lat[b],
                            P.dp_key_dp[b], rf[kernel.RF_DP_LAM], S.dp_fv[b], S.dp_fr[b], S.dp_fm[b],
                            rbb[kernel.RB_ERR_NEW])
        S.dp_fm[b, :, 1] = 0
        if nts > 0:
            temp_tick_row(kk, S.ts_in[b], P.ts_tp[b], P.ts_period, P.ts_lat[b], P.ts_cmax, P.ts_fl[b], P.ts_key[b],
                          S.ts_st[b], S.ts_fx[b], S.ts_fc[b], S.ts_primed[b], S.ts_fv[b], S.ts_fr[b], S.ts_fm[b],
                          S.ts_sampled[b])
            S.ts_fm[b, :, 1] = 0
        if P.rb_on != 0:
            readback_tick_row(kk, rf[kernel.RF_HE_V], rf[kernel.RF_HE_I], P.rb_lsb_v[b], P.rb_lsb_i[b], P.rb_noise[b],
                              P.rb_period, P.rb_lat[b], P.rb_cmax, P.rb_avg, P.rb_key[b], S.rb_st[b], S.rb_fv[b],
                              S.rb_fr[b], S.rb_fm[b], r8[kernel.R8_RB_SAMPLED])
            S.rb_fm[b, :, 1] = 0


@njit(cache=True)
def p_monitor(P, S, b, k0, n):
    rf = S.ring_float[b]
    ri = S.ring_int[b]
    r8 = S.ring_int8[b]
    for i in range(n):
        monitor_tick_row(k0 + i, rf[kernel.RF_P_PD], rf[kernel.RF_P_BG], rf[kernel.RF_T_PD], rf[kernel.RF_T_ADC],
                         ri[kernel.RI_PGA], r8[kernel.R8_FAULT], P.mo_pr[b], P.mo_pw[b], _world_row(P.mo_lat, b),
                         P.mo_phase, P.mo_pga, P.mo_fl[b], P.mo_afe, P.mo_win, P.mo_period, P.mo_cmax, P.mo_shot,
                         P.mo_stuck, P.mo_slot, P.mo_key_adc[b], P.mo_key_shot[b], P.mo_key_dnl[b], S.mo_sf[b],
                         S.mo_si[b], S.mo_fx[b], S.mo_fc[b], S.mo_fv[b], S.mo_fr[b], S.mo_fm[b],
                         r8[kernel.R8_MO_SAMPLED])
        S.mo_fm[b, :, 1] = 0


@njit(cache=True)
def p_truth(P, S, b, n):
    for i in range(n):
        kernel.truth_row(P, S, b)


@njit(cache=True)
def p_oracle(P, S, b, n):
    for i in range(n):
        kernel._oracle(P, S, b)


@njit(cache=True)
def p_metrics(P, S, b, k0, n):
    rf = S.ring_float[b]
    for i in range(n):
        k = k0 + i
        metrics_tick_row(k, 1, P.mp_f, P.mp_i, rf[kernel.RF_DET], S.m_ref[b], S.ring_int[b, kernel.RI_ORDER],
                         S.ring_bool[b, kernel.RB_WRONG], S.truth[b, kernel.TR_RELOMA], rf[kernel.RF_ORACLE],
                         rf[kernel.RF_ORACLE_UNR], rf[kernel.RF_HE_P], rf[kernel.RF_P_ABS],
                         S.ring_int[b, kernel.RI_HE_REG], S.ring_bool[b, kernel.RB_POPPED], rf[kernel.RF_BER],
                         rf[kernel.RF_BITS], S.m_af[b], S.m_ai[b], S.m_hist[b], S.m_runhist[b])


@njit(cache=True)
def p_release(P, S, b, k0, n):
    for i in range(n):
        kernel.release_row(P, S, b, k0 + 10 * i)


# --------------------------------------------------------------------------------------------------------------- timing

def timeit(fn, args, n, reps=REPEATS):
    """(median, min) over ``reps`` calls of fn(*args), in us per inner iteration (``n`` iterations per call)."""
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
    opts = ap.parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plant = perf_bench.bench_plant(1, {}, locked=opts.locked)
    run_block = plant._fn()
    done = np.zeros(1, np.int64)             # ticks advanced per world (written by the kernel)
    run_block(kernel.null_step, kernel.null_step, plant.P, plant.S, np.int64(plant.tick), np.int64(WARMUP_TICKS),
              np.int64(kernel.MODE_OPEN), np.int64(0), done)
    plant.tick += int(done[0])
    P, S = plant.P, plant.S
    n = int(opts.n)
    k0 = np.int64(plant.tick)
    b = np.int64(0)
    t0 = time.perf_counter()
    # the real kernel on the same plant (wall per tick, open loop, edges included)
    walls = []
    for _ in range(REPEATS):
        it0 = int(S.world_int[0, kernel.WI_SH_ITER])
        t = time.perf_counter()
        run_block(kernel.null_step, kernel.null_step, P, S, np.int64(plant.tick), np.int64(n),
                  np.int64(kernel.MODE_OPEN), np.int64(0), done)
        walls.append((time.perf_counter() - t) / n * 1e6)
        plant.tick += int(done[0])
        it1 = int(S.world_int[0, kernel.WI_SH_ITER])
    print(f"{'kernel run_block (full tick incl. edges, oracle)':55s} {np.median(walls):9.3f} us  (min {min(walls):.3f})"
          f"  newton it/ring/tick {(it1 - it0) / (n * plant.N):.3f}", flush=True)
    k0 = np.int64(plant.tick)
    # (label, compiled loop, its arguments; the last argument is the number of inner iterations)
    cases = [
        ("  scenario eval (k+1)", p_scen, (P, S, b, k0, n)),
        ("  laser_tick_row", p_laser, (P, S, b, k0, n)),
        ("  pattern_tick_row", p_pattern, (P, S, b, k0, n)),
        ("  phase.evaluate_binary_ring_optical_inputs", p_optics_row, (P, S, b, n)),
        ("  cache_refresh_engine_optics (substep constants)", p_precompute, (P, S, b, n)),
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
    only = [s for s in opts.only.split(",") if s]
    rows = []
    for name, f, args in cases:
        if only and not any(o in name for o in only):
            continue
        n_inner = args[-1]
        f(*args[:-1], 1)                     # compile (or load from the cache) with one iteration
        it0 = int(S.world_int[0, kernel.WI_SH_ITER])
        med, mn = timeit(f, args, n_inner)
        it1 = int(S.world_int[0, kernel.WI_SH_ITER])
        rows.append((name, med, mn))
        extra = f"  newton it/ring/tick {(it1 - it0) / (REPEATS * n_inner * P.n_rings):.3f}" if it1 > it0 else ""
        if f is p_implicit:
            # p_implicit does not write the kernel counter WI_SH_ITER: it returns its own Newton-iteration total
            newton_total = f(*args[:-1], n_inner)
            extra = f"  newton it/ring/tick {newton_total / (n_inner * P.n_rings):.3f}"
        print(f"{name:55s} {med:9.3f} us  (min {mn:.3f}){extra}", flush=True)
    print(f"[perf_profile] load {os.getloadavg()[0]:.2f}, compile+run {time.perf_counter() - t0:.0f} s, "
          f"locked={opts.locked}")


if __name__ == "__main__":
    main()
