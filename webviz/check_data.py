"""Independent re-check of the webviz data against fresh engine runs (not generate.py's code paths).

    cd ~/mrm-model
    .venv/bin/python tools/memguard.py -- .venv/bin/python webviz/check_data.py steady     # also a1_key, cold_key, a1_full

  steady  : fresh mrm.run of steady_pi_lock.yaml (40 ms): metrics vs scenario_steady.json's summary; the final Truth vs
            the last key frame (spectra levels, resonance, thermal observers); spectra recomputed with a closed-form
            add-drop ring written here (not op.ring_exact) and a 1 fm scan for notch position and depth; the thermal
            map at the last key frame against the documented site law evaluated here (sum of c(d) Z0 P).
  a1_key  : a1_pi_lock.yaml cut at the key frame nearest 66 ms (near the hot cusp): Truth vs that key frame.
  cold_key: the cold start cut at the key frame nearest 6 ms (during the mapping sweeps): Truth vs that key frame.
  a1_full : a fresh full A1 run: metrics vs scenario_a1_aggressor.json's summary.
Every difference it prints should be at the rounding of the exported numbers (README, "How the data was checked").
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "webviz" / "site" / "data"
sys.path.insert(0, str(ROOT))

from mrm.plant import phase as ph  # noqa: E402
from mrm.plant import thermal as th  # noqa: E402
from mrm.sim import experiment as ex  # noqa: E402
from mrm.sim import runner  # noqa: E402

RUNS = []
_orig = runner.run


def _wrap(*a, **kw):
    r = _orig(*a, **kw)
    RUNS.append(r)
    return r


runner.run = _wrap


def load(name):
    return json.loads((D / name).read_text())


def metrics_compare(m, scn):
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


def ring_closed_form(phi, k2t, k2d, a):
    """Through and drop power of an add-drop ring (Bogaerts 2012 eqs. 2-3), written here, not op.ring_exact."""
    t1, t2 = np.sqrt(1 - k2t), np.sqrt(1 - k2d)
    den = 1 - 2 * t1 * t2 * a * np.cos(phi) + (t1 * t2 * a) ** 2
    thru = (t2 ** 2 * a ** 2 - 2 * t1 * t2 * a * np.cos(phi) + t1 ** 2) / den
    drop = (1 - t1 ** 2) * (1 - t2 ** 2) * a / den
    return thru, drop


def phase_rows(pp, truth, lam):
    phs = ph.PhaseParams.from_plant_params(pp)
    r = phs.table_row(0)
    N, W = pp.n_rings, lam.size
    o = {k: np.zeros((N, W)) for k in ("phi0", "phi1", "a0", "a1", "e0", "e1", "k2t", "k2d", "trt")}
    shift, cj, rs, fl = np.zeros((N, 2)), np.zeros((N, 2)), np.zeros((N, 2)), np.zeros(N, np.int8)
    c = truth.conditions[0]
    ph.ring_optics_row(phs.ring[0], np.ascontiguousarray(lam, float), np.ascontiguousarray(truth.fields["t_ring_k"][0]),
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
    return worst


def observers_check(truth, thm, k):
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
    return get("monitor_pd"), get("ring_mode_weighted")


def quasi_static_map(plant, cfg, meta):
    """The documented site law, written here: rise(x, y) = sum_s clip(0.03 * 60 um / d, 0.015, 1) * Z0 * P_s."""
    sys_ = plant.sys
    u = plant.S.th_u1[0]
    G = th.dc_gain(sys_)[0]
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
    i = np.clip(np.searchsorted(X, x) - 1, 0, X.size - 2)
    j = np.clip(np.searchsorted(Y, y) - 1, 0, Y.size - 2)
    tx = (x - X[i]) / (X[i + 1] - X[i])
    ty = (y - Y[j]) / (Y[j + 1] - Y[j])
    return ((1 - tx) * (1 - ty) * grid[j, i] + tx * (1 - ty) * grid[j, i + 1] + (1 - tx) * ty * grid[j + 1, i]
            + tx * ty * grid[j + 1, i + 1])


def main(which):
    meta = load("meta.json")
    if which == "steady":
        scn, sp, thm = load("scenario_steady.json"), load("spectra_steady.json"), load("thermal_steady.json")
        res = ex.run(str(ROOT / "experiments/examples/steady_pi_lock.yaml"), write=False, log=lambda s: None)
        rr = RUNS[-1]
        print("steady: fresh mrm.run, main-run ticks", rr.plant.tick)
        metrics_compare(res.cases["main"].metrics, scn)
        truth = rr.plant.truth()
        k = len(sp["frames"]) - 1
        assert abs(sp["t_s"][k] - truth.t_s) < 1e-9, (sp["t_s"][k], truth.t_s)
        spectra_check(rr.plant.pp, truth, sp, k, "the last key frame (40 ms)")
        pd_e, ring_e = observers_check(truth, thm, k)
        qs, X, Y = quasi_static_map(rr.plant, ex.load(str(ROOT / "experiments/examples/steady_pi_lock.yaml")).cfg, meta)
        ex_map = np.array(thm["rise_k"][k], float)
        diff = qs - ex_map
        print("  map vs the quasi-static site law written here (steady, 40 ms): max |diff| %.4f K, median %.4f K, "
              "map range %.2f-%.2f K" % (np.max(np.abs(diff)), np.median(np.abs(diff)), ex_map.min(), ex_map.max()))
        jj, ii = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        print(f"    largest at x={X[ii]} um, y={Y[jj]} um: qs {qs[jj, ii]:.3f} exported {ex_map[jj, ii]:.3f}")
        # orientation: the map at the PDs (bilinear from the 4 nodes 1.6 um away) against the engine PDs
        pdxy = np.array(meta["geometry"]["monitor_pd_um"])
        bl = bilinear(ex_map, X, Y, pdxy[:, 0], pdxy[:, 1])
        print("  map bilinear at PDs - engine PD (K):", np.round(bl - pd_e, 3))
        flipped = bilinear(ex_map[::-1], X, Y, pdxy[:, 0], pdxy[:, 1])
        print("  same with y flipped (should be much worse):", np.round(flipped - pd_e, 3))
    elif which == "a1_key":
        sp, thm = load("spectra_a1_aggressor.json"), load("thermal_a1_aggressor.json")
        t = np.array(sp["t_s"])
        k = int(np.argmin(np.abs(t - 0.0660)))
        dur = round(t[k], 9)
        ex.run(str(ROOT / "experiments/examples/a1_pi_lock.yaml"), overrides=[f"experiment.duration_s={dur}"],
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
        ex.run(str(ROOT / "experiments/acceptance/b5_cold_start.yaml"), overrides=ov, write=False, log=lambda s: None)
        rr = RUNS[-1]
        truth = rr.plant.truth()
        print(f"cold start cut at {dur * 1e3:.3f} ms (key frame {k}); tick {truth.tick}; declared",
              truth.fields["declared_line"][0].astype(int), "line_on", truth.fields["line_on"][0].astype(int))
        spectra_check(rr.plant.pp, truth, sp, k, f"key frame {k}")
        observers_check(truth, thm, k)
    elif which == "a1_full":
        scn = load("scenario_a1_aggressor.json")
        res = ex.run(str(ROOT / "experiments/examples/a1_pi_lock.yaml"), write=False, log=lambda s: None)
        print("a1: fresh full mrm.run")
        metrics_compare(res.cases["main"].metrics, scn)


if __name__ == "__main__":
    main(sys.argv[1])
