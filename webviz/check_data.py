"""Independent re-check of the webviz data against fresh engine runs (not generate.py's code paths).

    cd ~/mrm-model
    .venv/bin/python tools/memguard.py -- .venv/bin/python webviz/check_data.py steady     # also a1_key, cold_key, a1_full

  steady  : fresh mrm.run of steady_pi_lock.yaml (40 ms): metrics vs scenario_steady.json's summary; the final Truth vs
            the last key frame (spectra levels, resonance, thermal observers); spectra recomputed with a closed-form
            add-drop ring written here (not optics.ring_exact) and a 1 fm scan for notch position and depth; the thermal
            map at the last key frame against the documented site law evaluated here (sum of c(d) Z0 P).
  a1_key  : a1_pi_lock.yaml cut at the key frame nearest 66 ms (near the hot cusp): Truth vs that key frame.
  cold_key: the cold start cut at the key frame nearest 6 ms (during the mapping sweeps): Truth vs that key frame.
  a1_full : a fresh full A1 run: metrics vs scenario_a1_aggressor.json's summary.
  derived : the numbers added for the views (stats_<id>.json, the limits, the crosstalk tables, the scenario inputs,
            the optical split per line) against the other data files and against laws written here; in `steady`
            also against the fresh run (DC gains, per-line drop and absorbed power, the heater's full-code power).
  all     : every check above (the default of `npm run check`).
The device data (devices_<id>.json: heater driver, monitor PD, running metrics, heat-source split, self-heating loop
gain) is checked in `steady` and `cold_key` against the fresh run's final kernel state and against laws written here.
Every difference it prints should be at the rounding of the exported numbers (README, "How the data was checked").
Each check has a tolerance; the script ends with PASS or FAIL (exit code 1) and lists the failed checks.

Read this first. main(which) runs one part. Each part loads the exported JSON files, runs the engine afresh where
it needs to, and calls need(ok, what) once per check. The laws that decide each comparison (the add-drop ring, the
PD dark current and noise, the site law of the thermal map, the self-heating loop gain) are written again here, so a
mistake in generate.py does not repeat itself in the check. The phase-model rows (mrm.plant.phase) and the DC
thermal gains (mrm.plant.thermal) still come from the model, as in generate.py.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "webviz" / "site" / "data"
sys.path.insert(0, str(ROOT))

from mrm.plant import phase  # noqa: E402
from mrm.plant import thermal  # noqa: E402
from mrm.sim import experiment  # noqa: E402
from mrm.sim import runner  # noqa: E402
from mrm.units import K_B, Q_E  # noqa: E402

#: every result of runner.run in this process (the fresh runs keep their Plant, so its final state can be read)
RUNS = []
_runner_run = runner.run


def _run_and_keep(*a, **kw):
    """runner.run, and keep its result in RUNS."""
    r = _runner_run(*a, **kw)
    RUNS.append(r)
    return r


runner.run = _run_and_keep
FAILS = []                  # the failed checks, in order


def need(ok, what):
    """Record one check: what it compared, and whether it is inside its tolerance."""
    print(f"  [{'ok' if ok else 'FAIL'}] {what}")
    if not ok:
        FAILS.append(what)


def load(name):
    """One exported data file (site/data/<name>) as a dict."""
    return json.loads((DATA_DIR / name).read_text())


def metrics_compare(m, scn):
    """The fresh run's metrics m against the exported summary of scenario file scn (5e-5 relative)."""
    pr = scn["summary"]["per_ring"]
    worst = {}
    for k, v in pr.items():
        if k not in m:
            continue
        a = np.asarray(m[k])[0]
        if a.dtype == bool:
            ok = list(map(bool, a)) == v
            worst[k] = "equal" if ok else f"DIFF {a.tolist()} vs {v}"
            continue
        e = np.array([np.nan if x is None else x for x in v], float)
        a = a.astype(float)
        both = np.isfinite(a) & np.isfinite(e)
        nanmatch = np.array_equal(np.isfinite(a), np.isfinite(e))
        rel = np.max(np.abs(a[both] - e[both]) / np.maximum(np.abs(a[both]), 1e-12)) if both.any() else 0.0
        worst[k] = (float(rel), nanmatch)
    for k, (rel_or_s) in worst.items():
        print(f"    {k:28s} {rel_or_s}")
    bad = [k for k, v in worst.items() if isinstance(v, tuple) and (v[0] > 5e-5 or not v[1])]
    bad += [k for k, v in worst.items() if isinstance(v, str) and v.startswith("DIFF")]
    s = scn["summary"]
    agg = {"max_abs_lock_error_pm": float(np.nanmax(np.asarray(m["max_abs_lock_error_pm"], float)[0])),
           "oma_penalty_db_worst": float(np.nanmax(np.asarray(m["oma_penalty_db"], float)[0])),
           "unlocked_rings": int(np.sum(~np.asarray(m["locked_final"])[0].astype(bool))),
           "excursions_total": int(np.sum(np.asarray(m["excursion_count"])[0]))}
    for k, v in agg.items():
        print(f"    summary {k}: fresh {v:.6g}, exported {s[k]}")
    print("  metrics per ring beyond 5e-5 relative:", bad or "none")
    need(not bad, f"per-ring metrics equal the exported summary within 5e-5 relative ({len(worst)} metrics)")


def ring_closed_form(phi, k2t, k2d, a):
    """Through and drop power of an add-drop ring (Bogaerts 2012 eqs. 2-3), written here, not optics.ring_exact.

    phi: round-trip phase (rad); k2t, k2d: power couplings of the through and drop couplers; a: round-trip
    amplitude transmission."""
    t1, t2 = np.sqrt(1 - k2t), np.sqrt(1 - k2d)
    den = 1 - 2 * t1 * t2 * a * np.cos(phi) + (t1 * t2 * a) ** 2
    thru = (t2 ** 2 * a ** 2 - 2 * t1 * t2 * a * np.cos(phi) + t1 ** 2) / den
    drop = (1 - t1 ** 2) * (1 - t2 ** 2) * a / den
    return thru, drop


def phase_rows(pp, truth, lam, t_ring=None):
    """Phase-model rows of every ring at the wavelengths lam (m) at a Truth state: dict of (N, W) arrays."""
    phs = phase.PhaseParams.from_plant_params(pp)
    r = phs.table_row(0)
    N, W = pp.n_rings, lam.size
    o = {k: np.zeros((N, W)) for k in ("phi0", "phi1", "a0", "a1", "e0", "e1", "k2t", "k2d", "trt")}
    shift, cj, rs, fl = np.zeros((N, 2)), np.zeros((N, 2)), np.zeros((N, 2)), np.zeros(N, np.int8)
    c = truth.conditions[0]
    tr = truth.fields["t_ring_k"][0] if t_ring is None else t_ring
    phase.ring_optics_row(phs.ring[0], np.ascontiguousarray(lam, float), np.ascontiguousarray(tr, float),
                          np.ascontiguousarray(truth.fields["t_heater_k"][0]), np.ascontiguousarray(c.v_lo),
                          np.ascontiguousarray(c.v_hi), phs.grid, phs.dneff[r], phs.alpha_fca[r], phs.capacitance[r],
                          phs.r_series[r], o["phi0"], o["phi1"], o["a0"], o["a1"], o["e0"], o["e1"], o["k2t"], o["k2d"],
                          o["trt"], shift, cj, rs, fl)
    return o


def spectra_check(pp, truth, sp, k, label):
    """Recompute ring by ring: exported grid curves, notch position (phase = 2 pi m) and depth by a fine scan."""
    c = truth.conditions[0]
    rho = np.asarray(c.rho, float)
    off = np.array(sp["grid"]["offset_nm"]) * 1e-9
    N = pp.n_rings
    print(f"  spectra at {label} (exported key-frame t = {sp['t_s'][k] * 1e3:.3f} ms, fresh tick {truth.tick}):")
    lv = sp["levels"]
    print("    state: |dT_ring| max %.2e K, |dT_heater| max %.2e K, |dv_lo| %.1e V, |dv_hi| %.1e V, |drho| %.1e" % (
        np.max(np.abs(np.array(lv["t_ring_k"][k]) - truth.fields["t_ring_k"][0])),
        np.max(np.abs(np.array(lv["t_heater_k"][k]) - truth.fields["t_heater_k"][0])),
        np.max(np.abs(np.array(lv["v_lo_v"][k]) - c.v_lo)), np.max(np.abs(np.array(lv["v_hi_v"][k]) - c.v_hi)),
        np.max(np.abs(np.array(lv["ones_density"][k]) - rho))))
    print("    lines: max |exported - Truth| %.2e pm; lam_res: max |exported - Truth| %.3e pm" % (
        np.max(np.abs(np.array(sp["line_nm"][k]) * 1e-9 - truth.lam_m[0])) * 1e12,
        np.max(np.abs(np.array(sp["lam_res_nm"][k]) * 1e-9 - truth.fields["lam_res_m"][0])) * 1e12))
    worst = dict(curve_db=0.0, drop_db=0.0, res_pm=0.0, notch_db=0.0, mean_res_vs_truth_pm=0.0)
    for r in range(N):
        cen = sp["center_nm"][k][r] * 1e-9
        lam = cen + off
        o = phase_rows(pp, truth, lam)
        t0, d0 = ring_closed_form(o["phi0"][r], o["k2t"][r], o["k2d"][r], o["a0"][r])
        t1, d1 = ring_closed_form(o["phi1"][r], o["k2t"][r], o["k2d"][r], o["a1"][r])
        drop = rho[r] * d1 + (1 - rho[r]) * d0
        for mine, key in ((t0, "thru0_db"), (t1, "thru1_db")):
            ex_db = np.array(sp[key][k][r], float)
            m_db = 10 * np.log10(mine)
            sel = m_db > -40                     # away from the notch bottom (dB is ill-conditioned there)
            worst["curve_db"] = max(worst["curve_db"], float(np.max(np.abs(m_db[sel] - ex_db[sel]))))
        worst["drop_db"] = max(worst["drop_db"], float(np.max(np.abs(10 * np.log10(drop) - np.array(sp["drop_db"][k][r])))))
        res_lv = []
        for lvl in (0, 1):
            phi = o[f"phi{lvl}"][r]
            # resonance: the 2 pi m crossing nearest the window centre, found on a fine grid, then refined
            m = np.round(phi / (2 * np.pi))
            cand = np.unique(m)
            best = None
            for mm in cand:
                f = phi - 2 * np.pi * mm
                s = np.flatnonzero(np.sign(f[:-1]) != np.sign(f[1:]))
                for i in s:
                    x = lam[i] - f[i] * (lam[i + 1] - lam[i]) / (f[i + 1] - f[i])
                    if best is None or abs(x - cen) < abs(best[0] - cen):
                        best = (x, mm)
            if best is None:
                res_lv.append(np.nan)
                continue
            x0, mm = best
            fine = x0 + np.linspace(-2e-12, 2e-12, 4001)          # +-2 pm at 1 fm
            of = phase_rows(pp, truth, fine)
            tf, _ = ring_closed_form(of[f"phi{lvl}"][r], of["k2t"][r], of["k2d"][r], of[f"a{lvl}"][r])
            i = int(np.argmin(tf))
            f2 = of[f"phi{lvl}"][r] - 2 * np.pi * mm
            j = int(np.argmin(np.abs(f2)))
            res_lv.append(fine[j])
            ex_res = sp["resonance_level_nm"][k][r][lvl] * 1e-9
            ex_notch = sp["notch_level_db"][k][r][lvl]
            worst["res_pm"] = max(worst["res_pm"], abs(fine[j] - ex_res) * 1e12)
            worst["notch_db"] = max(worst["notch_db"], abs(10 * np.log10(tf[i]) - ex_notch))
            if r in (0, 4) and lvl == 0:
                print(f"    R{r} bit0: resonance fresh {fine[j] * 1e9:.6f} nm (fine-scan min at {fine[i] * 1e9:.6f}) "
                      f"exported {ex_res * 1e9:.6f}; notch fresh {10 * np.log10(tf[i]):.2f} dB exported {ex_notch} dB")
        mean = 0.5 * (res_lv[0] + res_lv[1])
        worst["mean_res_vs_truth_pm"] = max(worst["mean_res_vs_truth_pm"],
                                            abs(mean - truth.fields["lam_res_m"][0, r]) * 1e12)
    print("    worst: grid curves (above -40 dB) %.3f dB, drop %.3f dB, resonance per level %.4f pm, notch depth "
          "%.3f dB, mean of levels vs Truth lam_res %.4f pm" % (worst["curve_db"], worst["drop_db"], worst["res_pm"],
                                                             worst["notch_db"], worst["mean_res_vs_truth_pm"]))
    need(worst["curve_db"] < 0.02 and worst["drop_db"] < 0.02, f"{label}: spectra curves within 0.02 dB")
    need(worst["res_pm"] < 0.01 and worst["mean_res_vs_truth_pm"] < 0.01, f"{label}: resonances within 0.01 pm")
    need(worst["notch_db"] < 0.02, f"{label}: notch depth within 0.02 dB")
    return worst


def observers_check(truth, thm, k):
    """The exported thermal observers at key frame k against the fresh Truth; returns the fresh PD and ring rises."""
    names = truth.obs_names
    t_base = truth.fields["t_base_k"][0, 0]
    t_obs = truth.t_obs[0] - t_base
    N = 8
    ob = thm["observers"]
    get = lambda pre: np.array([t_obs[names.index(f"{pre}{i}")] for i in range(N)])  # noqa: E731
    out = {}
    for key, pre in (("ring_rise_k", "ring_mode_weighted"), ("heater_rise_k", "heater_resistor"), ("pd_rise_k", "monitor_pd")):
        out[key] = float(np.max(np.abs(get(pre) - np.array(ob[key][k]))))
    out["die0_rise_k"] = abs(t_obs[names.index("die0")] - ob["die0_rise_k"][k])
    out["t_base_k"] = abs(t_base - thm["t_base_k"][k])
    print("  thermal observers, |exported - fresh Truth| max (K):", {a: f"{b:.2e}" for a, b in out.items()})
    need(out["ring_rise_k"] < 0.01 and max(out["heater_rise_k"], out["pd_rise_k"], out["die0_rise_k"]) < 0.005,
         "thermal observers equal the fresh run (rings 10 mK, others 5 mK)")
    return get("monitor_pd"), get("ring_mode_weighted")


def quasi_static_map(plant, cfg, meta):
    """The documented site law, written here: rise(x, y) = sum_s clip(0.03 * 60 um / d, 0.015, 1) * Z0 * P_s."""
    sys_ = plant.sys
    u = plant.S.th_u1[0]
    G = thermal.dc_gain(sys_)[0]
    z0 = np.array([G[sys_.ring_obs[i], sys_.heater_src[i]] for i in range(8)])
    near = float(cfg.get("ring.thermal.xt_nearest_frac"))
    floor = float(cfg.get("ring.thermal.site_floor_frac"))
    ref = float(cfg.get("ring.thermal.site_ref_um"))
    g = meta["geometry"]
    ring = np.array(g["ring_centers_um"])
    drv = np.array(g["driver_um"])
    agg = g["aggressor_sites_um"]
    src = []
    for s, name in enumerate(sys_.src_names):
        for pre, xy in (("heater", ring), ("waveguide_absorption", ring), ("junction", ring), ("driver", drv)):
            if name.startswith(pre) and name[len(pre):].isdigit():
                i = int(name[len(pre):])
                src.append((xy[i], z0[i], u[s], name))
                break
        else:
            src.append((np.array(agg[name]), z0.mean(), u[s], name))
    X = -61.0 + 4.0 * np.arange(120)
    Y = -61.25 + 2.5 * np.arange(50)
    XX, YY = np.meshgrid(X, Y)
    rise = np.zeros_like(XX)
    for xy, z, p, _ in src:
        d = np.maximum(np.hypot(XX - xy[0], YY - xy[1]), 1e-6)
        rise += np.clip(near * ref / d, floor, 1.0) * z * p
    pw = {n: p for _, _, p, n in src}
    print("  source powers at the end (W):", {k: f"{v:.3g}" for k, v in pw.items() if k.endswith("0") or "agg" in k})
    return rise, X, Y


def bilinear(grid, X, Y, x, y):
    """Bilinear interpolation of grid[y, x] (nodes X, Y) at the points (x, y)."""
    i = np.clip(np.searchsorted(X, x) - 1, 0, X.size - 2)
    j = np.clip(np.searchsorted(Y, y) - 1, 0, Y.size - 2)
    tx = (x - X[i]) / (X[i + 1] - X[i])
    ty = (y - Y[j]) / (Y[j + 1] - Y[j])
    return ((1 - tx) * (1 - ty) * grid[j, i] + tx * (1 - ty) * grid[j, i + 1] + (1 - tx) * ty * grid[j + 1, i]
            + tx * ty * grid[j + 1, i + 1])


def devices_check(plant, truth, dev, scn, meta, f, k, label):
    """devices_<id>.json at frame f / key frame k against the fresh run's state at the same tick and against laws
    written here (dark current, PD noise, signal current, heater driver state, heat-source split, loop gain)."""
    from mrm.sim import kernel
    S = plant.S
    md = meta["devices"]
    mo = md["monitor"]
    print(f"  devices at {label} (frame {f}, key frame {k}):")
    he = dev["heater"]
    last = {"reg": S.ring_int[0, kernel.RI["he_reg"]], "cmd": S.ring_int[0, kernel.RI["he_cmd"]],
            "dac_code": S.ring_int[0, kernel.RI["he_code"]]}
    for nm, v in last.items():
        need(list(map(int, v)) == he[nm][f], f"heater {nm} at the frame end equals the fresh kernel state")
    v_end, i_end, r_end = (S.ring_float[0, kernel.RF[n]] for n in ("he_v", "he_i", "he_r"))
    on = v_end > 0                                   # a heater at code 0 has V = I = 0
    print(f"    heaters driven at this tick: {int(on.sum())} of {on.size}")
    need(np.max(np.abs(r_end[on] - v_end[on] / i_end[on]) / r_end[on], initial=0.0) < 1e-9, "fresh heater state: R = V / I")
    v_law = np.array(he["dac_code"][f], float) / md["heater"]["code_hi"] * md["heater"]["v_full_scale_v"]
    dv = np.max(np.abs(v_law[on] - v_end[on]) / v_end[on], initial=0.0)
    print("    voltage law V = code / code_max x V_FS: max rel diff %.2e" % dv)
    need(dv < 1e-3 and np.all(v_end[~on] == 0) and np.all(v_law[~on] == 0),
         "heater voltage follows code / code_max x V_FS (DAC errors < 0.1 %)")
    t_h = truth.fields["t_heater_k"][0]
    r_law = np.array(md["heater"]["r_cold_ohm"]) * (1 + md["heater"]["tcr_per_k"] * (t_h - md["heater"]["tcr_ref_k"]))
    print("    TCR law R = R0 (1 + a (T_H - T0)) at the fresh heater temperature: max rel diff %.2e" % np.max(np.abs(r_law - r_end) / r_end))
    need(np.max(np.abs(r_law - r_end) / r_end) < 1e-3, "heater resistance follows the TCR law within 0.1 %")
    if scn["id"] == "steady":
        for nm, fresh, sc in (("v_v", v_end, 1.0), ("i_ma", i_end, 1e3), ("r_ohm", r_end, 1.0)):
            d = np.max(np.abs(np.array(he[nm][f]) - fresh * sc) / (fresh * sc))
            need(d < 1e-3, f"steady: heater {nm} frame mean equals the end state within 0.1 % ({d:.1e})")
    # monitor PD laws written here
    R = scn["rings"]
    pd = dev["pd"]
    T = np.array(R["t_pd_k"][f], float)
    idark = mo["dark_i0_na"] * np.exp(-(mo["dark_ea_ev"] * Q_E / K_B) * (1 / T - 1 / mo["dark_t0_k"]))
    d = np.max(np.abs(idark - np.array(pd["i_dark_na"][f])) / idark)
    need(d < 2e-3, f"dark current = I0 exp(-Ea/k (1/T - 1/T0)) at the exported PD temperature ({d:.1e})")
    isig = mo["responsivity_a_per_w"] * mo["gain_scale"] * np.array(R["p_drop_uw"][f], float)
    d = np.max(np.abs(isig - np.array(pd["i_signal_ua"][f])) / isig)
    need(d < 2e-3, f"signal current = responsivity x drop power ({d:.1e})")
    i_tot = (np.array(pd["i_signal_ua"][f]) * 1e-6 + np.array(pd["i_dark_na"][f]) * 1e-9)
    s_tia = (mo["input_noise_pa_per_rt_hz"] * 1e-12) ** 2
    sig = np.sqrt(((2 * Q_E * i_tot if mo["shot_noise"] else 0) + s_tia) / (2 * mo["integration_window_s"])) * 1e9
    d = np.max(np.abs(sig - np.array(pd["noise_rms_na"][f])) / sig)
    need(d < 2e-3, f"PD noise = sqrt((2 q I + S_tia) / (2 T_int)) ({d:.1e})")
    gain = np.array([mo["pga_gains"][g] for g in pd["pga"][f]])
    fs = mo["code_max"] * mo["lsb_na"] * 1e-3 / gain
    d = np.max(np.abs(fs - np.array(pd["full_scale_ua"][f])) / fs)
    need(d < 2e-3, f"ADC full scale = code_max x LSB / PGA gain ({d:.1e})")
    # running metrics at this frame against the fresh run's metrics (the fresh run ends at this frame's end)
    m = plant.metrics()
    run = dev["running"]
    worst = 0.0
    for key in ("max_abs_lock_error_pm", "rms_lock_error_pm", "excursion_count", "time_unlocked_s", "heater_energy_j",
                "wrong_line_s", "fsr_hop_events", "heater_low_rail_s"):
        a = np.asarray(m[key], float)[0]
        b = np.array([np.nan if x is None else x for x in run[key][f]], float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.any():
            worst = max(worst, float(np.max(np.abs(a[ok] - b[ok]) / np.maximum(np.abs(a[ok]), 1e-12))))
    need(worst < 5e-5, f"running metrics at this frame equal the fresh run's engine metrics ({worst:.1e} relative)")
    # heat-source split: written here from the fresh lag states
    sysm = plant.sys
    x = truth.x[0]
    lag_src, obs_ptr = np.asarray(sysm.lag_src), np.asarray(sysm.obs_ptr)
    ex_split = np.array(dev["key"]["ring_rise_by_source_k"][k], float)
    worst = 0.0
    for i in range(plant.N):
        o = int(sysm.ring_obs[i])
        mm = np.arange(obs_ptr[o], obs_ptr[o + 1])
        fresh = np.zeros(sysm.n_src)
        np.add.at(fresh, lag_src[mm], x[mm])
        worst = max(worst, float(np.max(np.abs(fresh - ex_split[i]) - 1e-3 * np.abs(fresh))))
    need(worst < 1e-4, f"ring rise split by heat source equals the fresh lag states (excess over 0.1 %: {worst:.1e} K)")
    # self-heating loop gain written here: closed-form ring, loss = 1 - through - drop, heat fraction eta
    lam = np.asarray(truth.lam_m[0], float)
    from mrm.plant import optics
    p_in = np.asarray(truth.out_rl[0, optics.RING_LINE_OUTPUT_ROW_INPUT_POWER], float)
    rho = np.asarray(truth.conditions[0].rho, float)
    t0 = truth.fields["t_ring_k"][0]

    def pabs(t):
        o = phase_rows(plant.pp, truth, lam, t)
        out = np.zeros(plant.N)
        for r in range(plant.N):
            t0_, d0 = ring_closed_form(o["phi0"][r], o["k2t"][r], o["k2d"][r], o["a0"][r])
            t1_, d1 = ring_closed_form(o["phi1"][r], o["k2t"][r], o["k2d"][r], o["a1"][r])
            out[r] = np.sum(p_in[r] * (rho[r] * o["e1"][r] * (1 - t1_ - d1) + (1 - rho[r]) * o["e0"][r] * (1 - t0_ - d0)))
        return out
    z = thermal.selfheat_dc_k_per_w(sysm)[0]
    lg = z * (pabs(t0 + 0.01) - pabs(t0 - 0.01)) / 0.02
    ex_lg = np.array(dev["key"]["selfheat_loop_gain"][k], float)
    d = np.max(np.abs(lg - ex_lg) - 1e-3 * np.abs(lg))
    print("    loop gain fresh", np.round(lg, 3), "exported", ex_lg)
    need(d < 0.01, f"self-heating loop gain equals the closed-form recomputation ({d:.1e} beyond 0.1 %)")


def _none_to_nan(v):
    """A JSON list (or list of lists) as a float array, null -> NaN."""
    return np.array([[np.nan if x is None else x for x in r] if isinstance(r, list) else (np.nan if r is None else r)
                     for r in v], float)


def t_frac_bisect(taus, res, frac):
    """First time sum r (1 - exp(-t/tau)) reaches frac of sum r, by bisection on a coarse bracket (written here)."""
    taus, res = np.asarray(taus, float), np.asarray(res, float)
    fin = res.sum()
    y = lambda t: float(np.sum(res * (1 - np.exp(-t / taus))))         # noqa: E731
    grid = np.logspace(-10, 3, 131)
    vals = np.array([y(t) for t in grid])
    i = int(np.argmax(vals >= frac * fin))
    if vals[i] < frac * fin:
        return None
    lo, hi = (grid[i - 1] if i else 0.0), grid[i]
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if y(mid) < frac * fin else (lo, mid)
    return 0.5 * (lo + hi)


def derived_check(meta):
    """The numbers added for the views, against the other data files and laws written here."""
    print("derived numbers (stats, limits, crosstalk, inputs, optical split):")
    full = meta["thermal_coupling"]["full"]
    G = np.array(full["g_k_per_mw"], float)
    worst, worst_t, n_t = 0.0, 0.0, 0
    for o, row in enumerate(full["lags"]):
        for s_, lags in enumerate(row):
            if not lags:
                continue
            a = np.array(lags, float)
            ssum = a[:, 1].sum()
            worst = max(worst, abs(ssum - G[o, s_]) / max(abs(G[o, s_]), 1e-6))
            t63 = full["t63_s"][o][s_]
            if t63 is not None and o % 3 != 1 and n_t < 400:          # rings, PDs, die sensor
                tb = t_frac_bisect(a[:, 0], a[:, 1], 1 - np.exp(-1))
                worst_t = max(worst_t, abs(tb - t63) / tb)
                n_t += 1
    need(worst < 2e-3, f"DC gain of every observer/source pair = sum of its exported residues ({worst:.1e} relative)")
    need(worst_t < 1e-2, f"t63 of {n_t} pairs recomputed by bisection from the exported lags ({worst_t:.1e} relative)")
    tc = meta["thermal_coupling"]
    rings = [i for i, n in enumerate(full["obs_names"]) if n.startswith("ring_mode_weighted")]
    d = np.max(np.abs(G[rings] - np.array(tc["ring_k_per_mw"], float)) / np.maximum(np.abs(G[rings]), 1e-6))
    need(d < 2e-3, f"full coupling table rows of the rings equal ring_k_per_mw ({d:.1e} relative)")
    hw = {r["label"]: r for r in meta["hardware_limits"]["rows"]}
    h = meta["devices"]["heater"]
    pmax = h["v_full_scale_v"] ** 2 / h["r_cold_ohm"][0] * 1e3
    d = abs(hw["Maximum power per heater"]["value"] - pmax) / pmax
    need(d < 1e-3, f"maximum heater power = V_FS^2 / R0 at 25 degC ({pmax:.3f} mW, {d:.1e} relative)")
    for sc in meta["scenarios"]:
        sid = sc["id"]
        scn, dev, sp = load(sc["files"]["scenario"]), load(sc["files"]["devices"]), load(sc["files"]["spectra"])
        stats = load(sc["files"]["stats"])
        print(f"  {sid}:")
        ring_traces = scn["rings"]
        pr = scn["summary"]["per_ring"]
        quantities = {x["key"]: x for x in stats["quantities"]}
        states = {x["key"]: x for x in stats["states"]}
        lim = {x["key"]: x for x in stats["limits"]}
        # extremes: the 10 us tick envelopes against the frame envelopes (both are tick extremes)
        fmax = np.nanmax(_none_to_nan(ring_traces["detuning_max_pm"]), axis=0)
        fmin = np.nanmin(_none_to_nan(ring_traces["detuning_min_pm"]), axis=0)
        smax, smin = np.array(quantities["det"]["run"]["max"], float), np.array(quantities["det"]["run"]["min"], float)
        tail = 0.5 if sid == "a1_aggressor" else 0.0                 # A1: the 10 us series also covers the last 0.2 ms
        tol = np.maximum(0.01, 2e-6 * np.abs(fmax)), np.maximum(0.01, 2e-6 * np.abs(fmin))
        need(np.all(smax >= fmax - tol[0]) and np.all(smin <= fmin + tol[1]) and
             (tail > 0 or (np.all(np.abs(smax - fmax) < tol[0]) and np.all(np.abs(smin - fmin) < tol[1]))),
             "detuning min / max of the stats = the frame envelopes (both are tick extremes)")
        m_max = np.array(pr["max_abs_lock_error_pm"], float)
        d = np.max(np.abs(np.array(quantities["abs_err"]["scored"]["max"], float) - m_max)
                   / np.maximum(50.0, np.abs(m_max)))
        need(d < 2e-5, f"scored max |lock error| of the stats = the engine metric (within 1 fm or 2e-5: {d:.1e})")
        # means: the 10 us means against the frame means (same ticks except A1's last 0.2 ms)
        tr = _none_to_nan(ring_traces["t_ring_k"]).mean(axis=0) - 273.15
        d = np.max(np.abs(np.array(quantities["t_ring"]["run"]["mean"], float) - tr))
        need(d < (0.05 if tail else 2e-3), f"run mean ring temperature of the stats = mean of the frame means ({d:.1e} K)")
        # time at a rail: samples x 10 us against the engine's per-tick metric
        for k, mk in (("rail_lo", "heater_low_rail_s"), ("rail_hi", "heater_sat_s")):
            d = np.max(np.abs(np.array(states[k]["scored"], float) - np.array(pr[mk], float) * 1e3))
            need(d < 0.02, f"time at {k} (10 us samples) = engine {mk} ({d:.1e} ms)")
        d = np.max(np.abs(np.array(lim["heater_lo"]["scored"]["time_at_ms"], float) - np.array(pr["heater_low_rail_s"], float) * 1e3))
        need(d < 0.02, f"limit 'heater at the low rail': time at the limit = engine heater_low_rail_s ({d:.1e} ms)")
        lo_m = np.array(lim["heater_lo"]["scored"]["margin_min"], float)
        hi_m = np.array(lim["heater_hi"]["scored"]["margin_min"], float)
        d = max(np.max(np.abs(lo_m - (np.array(pr["code_min_used"], float) - meta["devices"]["heater"]["code_lo"]))),
                np.max(np.abs(hi_m - (meta["devices"]["heater"]["code_hi"] - np.array(pr["code_max_used"], float)))))
        need(d <= 2, f"heater rail margins = engine code range used, within 2 codes (one step at the window edges: {d:.0f})")
        bm = np.array([np.nan if v is None else v for v in pr["ber_mean"]], float)
        sb = np.array([np.nan if v is None else v for v in quantities["ber"]["scored"]["mean"]], float)
        ok = np.isfinite(bm) & (bm > 1e-290)
        d = np.max(np.abs(sb[ok] - bm[ok]) / bm[ok]) if ok.any() else 0.0
        need(d < 1e-3, f"scored mean BER of the stats = engine ber_mean ({d:.1e} relative)")
        # crosstalk per frame: the groups sum to the ring rise (lag states at 10 us against the tick mean)
        xt = dev["xt"]
        rise = np.array(xt["ring_k"], float).sum(axis=2) + np.array(scn["drives"]["t_base_k"], float)[:, None]
        d = np.abs(rise - _none_to_nan(ring_traces["t_ring_k"]))
        print(f"    crosstalk split + base - ring temperature: max {d.max():.3f} K, median {np.median(d):.1e} K")
        if sid != "cold_start":
            need(np.median(d) < 1e-3 and d.max() < 0.1, "crosstalk split per frame sums to the ring temperature")
        # optical split per line: the drop of every line sums to the drop at the PD; foreign light is a small part
        dl = np.array(sp["bus"]["p_drop_uw"], float)
        need(np.all(dl >= 0) and dl.shape[1:] == (meta["n_rings"], meta["n_lines"]), "per-line drop powers are >= 0")
        # inputs: the aggressor and laser signals as the drives read them
        ins = {x["name"]: x for x in scn["inputs"]["signals"]}
        a = ins["thermal.aggressor_w"]
        for j, site in enumerate(a.get("columns", [])):
            v = np.array(a["mean"], float)[:, j] if "mean" in a else np.full(len(scn["t_s"]), a["constant"][j])
            dr = np.array(scn["drives"]["aggressor_w"][site], float)
            d = np.max(np.abs(v - dr)) / max(np.max(np.abs(dr)), 1e-3)
            need(d < 5e-4, f"input {site} power = drives.aggressor_w (4-digit rounding there; {d:.1e} of the peak)")
        pdb = ins["laser.power_delta_db"]
        v = np.array(pdb["mean"], float) if "mean" in pdb else np.tile(pdb["constant"], (len(scn["t_s"]), 1))
        en = ins["laser.line_enable"]
        e = np.array(en["mean"], float) if "mean" in en else np.tile(en["constant"], (len(scn["t_s"]), 1))
        fac = 10 ** (v / 10) * np.clip(e, 0, 1)
        d = np.max(np.abs(fac - np.array(scn["drives"]["laser_power_factor"], float)))
        need(d < 1e-4, f"inputs laser.power_delta_db x line_enable = drives.laser_power_factor ({d:.1e})")
        # junction: the two levels are bias +- swing / 2
        j = stats["junction"]
        v0, v1 = np.array(j["v0_v"], float), np.array(j["v1_v"], float)
        sw = [r for r in meta["hardware_limits"]["rows"] if r["label"] == "Data swing"][0]["value"]
        bias = [r for r in meta["hardware_limits"]["rows"] if r["label"] == "Bias set point"][0]["value"]
        need(np.max(np.abs(v1 - v0 - sw)) < 2e-3 and np.max(np.abs((v0 + v1) / 2 - bias)) < 2e-3,
             "junction levels: V1 - V0 = the swing and (V0 + V1) / 2 = the bias (within 2 mV)")


def derived_fresh(rr, truth, meta, sp, k):
    """steady: the coupling table, the per-line optical split and the full-code power against the fresh run."""
    from mrm.plant import optics

    full = meta["thermal_coupling"]["full"]
    G = thermal.dc_gain(rr.plant.sys)[0] * 1e-3
    d = np.max(np.abs(G - np.array(full["g_k_per_mw"], float)) / np.maximum(np.abs(G), 1e-6))
    need(d < 2e-3, f"fresh: DC gains of the fresh thermal system = the exported table ({d:.1e} relative)")
    drop = np.asarray(truth.out_rl[0, optics.RING_LINE_OUTPUT_ROW_CLOCKWISE_DROP_POWER], float) * 1e6
    ab = np.asarray(truth.out_rl[0, optics.RING_LINE_OUTPUT_ROW_ABSORBED_POWER], float) * 1e6
    e_d = np.array(sp["bus"]["p_drop_uw"][k], float)
    e_a = np.array(sp["bus"]["p_abs_uw"][k], float)
    big = drop > 1e-6
    d1 = np.max(np.abs(e_d[big] - drop[big]) / drop[big])
    d2 = np.max(np.abs(e_a[ab > 1e-6] - ab[ab > 1e-6]) / ab[ab > 1e-6])
    need(
        d1 < 1e-3 and d2 < 1e-3,
        f"fresh: per-line drop and absorbed power at 40 ms = Truth ({d1:.1e}, {d2:.1e} relative)",
    )
    tot = drop.sum(axis=1)
    pdw = np.asarray(truth.fields["p_drop_w"][0], float) * 1e6
    d = np.max(np.abs(tot - pdw) / pdw)
    need(d < 1e-6, f"fresh: the per-line drop sums to the PD's drop power ({d:.1e} relative)")
    own = np.array([e_d[r, l] for r, l in enumerate(np.rint(truth.fields["declared_line"][0]).astype(int))])
    print(
        "    foreign-line share of the monitor light at 40 ms (%):",
        np.round(100 * (1 - own / e_d.sum(axis=1)), 4),
    )


def main(which):
    """Run one part of the check (steady, a1_key, cold_key, a1_full, derived) or all of them."""
    if which == "derived":
        derived_check(load("meta.json"))
        return
    if which == "all":
        for w in ("steady", "a1_key", "cold_key", "a1_full", "derived"):
            print(f"===== {w}")
            main(w)
        return
    meta = load("meta.json")
    if which == "steady":
        scn, sp, thm = load("scenario_steady.json"), load("spectra_steady.json"), load("thermal_steady.json")
        res = experiment.run(str(ROOT / "experiments/examples/steady_pi_lock.yaml"), write=False, log=lambda s: None)
        rr = RUNS[-1]
        print("steady: fresh mrm.run, main-run ticks", rr.plant.tick)
        metrics_compare(res.cases["main"].metrics, scn)
        truth = rr.plant.truth()
        k = len(sp["frames"]) - 1
        assert abs(sp["t_s"][k] - truth.t_s) < 1e-9, (sp["t_s"][k], truth.t_s)
        spectra_check(rr.plant.pp, truth, sp, k, "the last key frame (40 ms)")
        pd_e, ring_e = observers_check(truth, thm, k)
        devices_check(rr.plant, truth, load("devices_steady.json"), scn, meta, len(scn["t_s"]) - 1, k, "40 ms")
        derived_fresh(rr, truth, meta, sp, k)
        steady_cfg = experiment.load(str(ROOT / "experiments/examples/steady_pi_lock.yaml")).cfg
        qs, X, Y = quasi_static_map(rr.plant, steady_cfg, meta)
        ex_map = np.array(thm["rise_k"][k], float)
        diff = qs - ex_map
        print("  map vs the quasi-static site law written here (steady, 40 ms): max |diff| %.4f K, median %.4f K, "
              "map range %.2f-%.2f K" % (np.max(np.abs(diff)), np.median(np.abs(diff)), ex_map.min(), ex_map.max()))
        need(np.max(np.abs(diff)) < 0.1, "steady map equals the site law written here within 0.1 K")
        jj, ii = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        print(f"    largest at x={X[ii]} um, y={Y[jj]} um: qs {qs[jj, ii]:.3f} exported {ex_map[jj, ii]:.3f}")
        # orientation: the map at the PDs (bilinear from the 4 nodes 1.6 um away) against the engine PDs
        pdxy = np.array(meta["geometry"]["monitor_pd_um"])
        bl = bilinear(ex_map, X, Y, pdxy[:, 0], pdxy[:, 1])
        print("  map bilinear at PDs - engine PD (K):", np.round(bl - pd_e, 3))
        need(np.max(np.abs(bl - pd_e)) < 0.05, "the map at the PDs equals the engine PDs within 50 mK")
        flipped = bilinear(ex_map[::-1], X, Y, pdxy[:, 0], pdxy[:, 1])
        print("  same with y flipped (should be much worse):", np.round(flipped - pd_e, 3))
    elif which == "a1_key":
        sp, thm = load("spectra_a1_aggressor.json"), load("thermal_a1_aggressor.json")
        t = np.array(sp["t_s"])
        k = int(np.argmin(np.abs(t - 0.0660)))
        dur = round(t[k], 9)
        experiment.run(str(ROOT / "experiments/examples/a1_pi_lock.yaml"), overrides=[f"experiment.duration_s={dur}"],
                       write=False, log=lambda s: None)
        rr = RUNS[-1]
        truth = rr.plant.truth()
        print(f"a1 cut at {dur * 1e3:.3f} ms (key frame {k}); tick {truth.tick}")
        print("  ring 0 detuning fresh %.3f pm" % (truth.fields["detuning_m"][0, 0] * 1e12))
        spectra_check(rr.plant.pp, truth, sp, k, f"key frame {k}")
        observers_check(truth, thm, k)
        ex_map = np.array(thm["rise_k"][k], float)
        print("  map max %.1f K at that key frame (the node beside the aggressor site)" % ex_map.max())
    elif which == "cold_key":
        sp, thm = load("spectra_cold_start.json"), load("thermal_cold_start.json")
        t = np.array(sp["t_s"])
        k = int(np.argmin(np.abs(t - 0.0060)))
        dur = round(t[k], 9)
        ov = ["experiment.kind=run", "experiment.batch=1", "experiment.uncertainty=nominal", "experiment.variation=off",
              "scenario.params.ambient_c_range=[25.0, 25.0]", f"experiment.duration_s={dur}"]
        experiment.run(str(ROOT / "experiments/acceptance/b5_cold_start.yaml"), overrides=ov, write=False,
                       log=lambda s: None)
        rr = RUNS[-1]
        scn_c = load("scenario_cold_start.json")
        truth = rr.plant.truth()
        print(f"cold start cut at {dur * 1e3:.3f} ms (key frame {k}); tick {truth.tick}; declared",
              truth.fields["declared_line"][0].astype(int), "line_on", truth.fields["line_on"][0].astype(int))
        spectra_check(rr.plant.pp, truth, sp, k, f"key frame {k}")
        observers_check(truth, thm, k)
        devices_check(rr.plant, truth, load("devices_cold_start.json"), scn_c, meta, sp["frames"][k], k, f"{dur * 1e3:.3f} ms")
    elif which == "a1_full":
        scn = load("scenario_a1_aggressor.json")
        res = experiment.run(str(ROOT / "experiments/examples/a1_pi_lock.yaml"), write=False, log=lambda s: None)
        print("a1: fresh full mrm.run")
        metrics_compare(res.cases["main"].metrics, scn)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
    print()
    if FAILS:
        print(f"FAIL: {len(FAILS)} check(s) outside tolerance:")
        for f_ in FAILS:
            print("  -", f_)
        sys.exit(1)
    print("PASS: every check is inside its tolerance")
