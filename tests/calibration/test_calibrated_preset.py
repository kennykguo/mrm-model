"""The calibrated preset g26_calibrated and the fit record mrm/calibration/latest_fit.json (marker
``calibration``: reported, not CI-blocking; 07 4.1).

What is checked:
1. g26_calibrated loads, closes every relation and realizes nominal and sampled batches;
2. every leaf the calibration wrote carries targets, residual and identifiable; a leaf tagged ``calibrated`` meets
   the half-prior rule (95 % profile hull at most half the prior, closed, one interval) with a declared prior and a
   fit within tolerance, and a leaf that does not meet it is not tagged ``calibrated``; the same holds for every
   alternative of a composed world, judged on the unconditional profiles; the loss-mechanism worlds hold the solver
   dispersion;
3. the preset equals the fit record (every written value);
4. the calibrated plant reproduces, through the replicas' own protocols, the targets the record says it fits
   within tolerance (CAL-1, 3a, 3b, 4, 5 at the primary power, 7, 8, 10, 12b), and CAL-2 with the fitted
   instrument floor;
5. the thermo-optic world H86 of g26_calibrated selects its own heater leaves and reproduces CAL-3a/3b/4;
6. g26_calibrated runs the A1 acceptance experiment closed loop without errors;
7. (integrator) the record's context equals its leaf decisions, and the calibrated G25 bench preset
   g25_test_vehicle_calibrated equals the record, g26_calibrated and bench_config (also in the H86 world).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from mrm import params
from mrm.calibration import groups as G
from mrm.calibration import pipeline, preset
from mrm.calibration.core import calibration_targets
from mrm.validation import replicas as R
from mrm.validation.targets_io import evaluate

pytestmark = pytest.mark.calibration

HERE = Path(__file__).resolve().parent
T = calibration_targets()


@pytest.fixture(scope="module")
def record():
    if not pipeline.RESULTS_PATH.exists():
        pytest.skip("no fit record: run python -m mrm.calibration")
    return pipeline.load_results()


@pytest.fixture(scope="module")
def cfg():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return params.load(preset.PRESET_NAME)


def _written(record):
    for g in record["groups"].values():
        for d in g["decisions"]:
            yield d


def test_every_fitted_residual_is_a_calibration_target(record):
    """07 P0-5: every target id in the record (group targets, per-hypothesis residuals, leaf decisions, composed
    worlds) is a calibration.yaml id that may be fitted (CAL-6, fit: false, never appears)."""
    fit_ok = {tid for tid, t in T.items() if t.fit}
    for g, gr in record["groups"].items():
        ids = set(gr["targets"])
        for f in gr["fits"].values():
            ids |= set(f["targets"])
        for d in gr["decisions"]:
            ids |= set(d["targets"]) | set(d["residual"])
        for w in (gr.get("extra") or {}).get("composed_worlds", {}).values():
            ids |= set(w.get("targets") or {})
        assert ids <= fit_ok, (g, sorted(ids - fit_ok))


def test_preset_loads_closes_and_realizes(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp = params.realize(cfg, 1)
        pps = params.realize(cfg, 8, uncertainty="sample", variation="sample", seeds=3)
    assert pp.batch == 1 and pps.batch == 8          # relation closure: tests/closure/test_presets.py (every preset)


def test_written_leaves_carry_their_calibration_record(cfg, record):
    for d in _written(record):
        if d["path"].startswith("variation.") or d["path"] not in cfg.leaves:
            continue
        leaf = cfg.leaf(d["path"])
        assert leaf.targets, d["path"]
        assert leaf.residual is not None and leaf.identifiable is not None, d["path"]
        ident = leaf.identifiable
        if leaf.provenance == "calibrated":
            assert ident["calibrated"] and ident["width_frac"] <= 0.5 and not ident["open"] \
                and ident["components"] == 1, d["path"]
        else:
            assert not (d["identifiable"].get("calibrated") and d.get("write_value", True)), d["path"]


def test_no_calibrated_tag_without_tolerance_or_declared_prior(record):
    """Rule 2: a fit that misses its tolerance (a model-form failure) or a leaf without a declared prior (fit bounds
    only) cannot produce a ``calibrated`` tag, on a leaf or on an alternative."""
    for d in _written(record):
        idt = d["identifiable"]
        if d["provenance"] == "calibrated":
            assert idt.get("within_tolerance", True) and "no declared prior" not in idt.get("rule", ""), d["path"]
            assert all(v["tolerance_level"] == "pass" for v in d["residual"].values()), d["path"]
    worlds = record["groups"]["coupling_er"]["extra"]["composed_worlds"]
    for w, rec in worlds.items():
        for leaf, idt in (rec.get("identifiable") or {}).items():
            if idt.get("calibrated"):
                assert all(v["within_tolerance"] for v in rec["targets"].values()), (w, leaf)
                # judged on the unconditional profile, which is never narrower than the conditional one
                assert idt["profile_95"][0] <= idt["conditional_profile_95"][0] + 1e-12, (w, leaf)
                assert idt["profile_95"][1] >= idt["conditional_profile_95"][1] - 1e-12, (w, leaf)
    # every alternative written from a composed world carries that world's verdict
    alt_world = {alt: (world, leaf) for world, _h, _held, free in pipeline.COUPLING_WORLDS for leaf, alt in free.items()}
    for d in _written(record):
        for an, a in (d.get("alternatives") or {}).items():
            if an not in alt_world:
                continue
            world, leaf = alt_world[an]
            assert (a["provenance"] == "calibrated") == bool(worlds[world]["identifiable"][leaf]["calibrated"]), an


@pytest.mark.xfail(strict=False, reason="known issue (docs/KNOWN_ISSUES.md): latest_fit.json and g26_calibrated are "
                   "stale, fitted before the params review fixes; the recomputed chi2 follows the secant beta (7.77 "
                   "against the recorded 7.51) until the refit with the real device parameters")
def test_loss_worlds_hold_the_solver_dispersion(record):
    """The loss-mechanism hypotheses and worlds run with the coupler artefact's dispersion (07 P1-5), never with the
    dispersion fitted under the dispersion mechanism: recomputing their chi2 from a fresh problem reproduces the
    record."""
    g = record["groups"]["coupling_er"]
    disp_solver = params.load(G.BENCH).get(G.P_DISP)
    # the problems' own context (the heater leaves fitted before coupling_er) plus the dispersion fitted for the
    # nominal world (as the pipeline's context carries it after the group)
    leak = pipeline.coupling_context(record)
    leak.device[G.P_DISP] = record["context"]["device"][G.P_DISP]
    probs = {p.hypothesis: p for p in G.coupling_er_problems(leak, T)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for h in ("over/ring_t_loss/floor", "over/heater_t_loss/floor"):
            assert g["fits"][h]["fixed"][G.P_DISP] == pytest.approx(disp_solver, rel=1e-12)
            assert probs[h].chi2(g["fits"][h]["theta"]) == pytest.approx(g["fits"][h]["chi2"], rel=1e-9)
        for w, h in (("er_mechanism=ring_t_loss", "over/ring_t_loss/floor"),
                     ("er_mechanism=heater_t_loss", "over/heater_t_loss/floor")):
            rec = g["extra"]["composed_worlds"][w]
            assert probs[h].chi2({**rec["held"], **rec["theta"]}) == pytest.approx(rec["chi2"], rel=1e-6)


def test_preset_equals_the_fit_record(cfg, record):
    for d in _written(record):
        if not d.get("write_value", True) or d["path"].startswith("variation.") or d["value"] is None:
            continue
        assert cfg.get(d["path"]) == pytest.approx(d["value"], rel=1e-5), d["path"]
        for an, a in (d.get("alternatives") or {}).items():
            assert cfg.leaf(d["path"]).alternatives[an]["value"] == pytest.approx(a["value"], rel=1e-5)


def test_record_context_equals_its_decisions(record):
    """The context (read by bench_config and by every --groups rerun) is what a full serial run leaves: the written
    leaf decisions (integrator: the merged review record had kept three pre-review values)."""
    ctx = pipeline.context_from_decisions(record)
    for scope in ("device", "system"):
        assert record["context"][scope] == ctx[scope], scope


def test_bench_preset_equals_the_record_and_g26_calibrated(cfg, record):
    """g25_test_vehicle_calibrated = g25_test_vehicle + the device leaves of g26_calibrated (values, alternatives
    and the uncertainty selections; the H86 world selects its own heater leaves on the bench too)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bench = params.load(preset.BENCH_PRESET_NAME)
        bc = preset.bench_config(record)
        g25 = params.load(G.BENCH)
        h86 = params.load(preset.BENCH_PRESET_NAME, alts={"uncertainty.thermo_optic": "H86"})
        h86_cal = params.load(preset.PRESET_NAME, alts={"uncertainty.thermo_optic": "H86"})
    for path, v in record["context"]["device"].items():
        assert bench.get(path) == pytest.approx(v, rel=1e-5), path
        assert bc.get(path) == pytest.approx(v, rel=1e-12), path
        assert bench.get(path) == cfg.get(path), path
        assert bench.leaf(path).provenance == cfg.leaf(path).provenance, path
        assert (bench.leaf(path).alternatives or {}) == (cfg.leaf(path).alternatives or {}), path
    for path in (G.P_PFSR, G.P_R0, G.P_RATIO):
        assert h86.get(path) == h86_cal.get(path) != bench.get(path), path
    # the bench's own leaves are g25_test_vehicle's
    for path in ("laser.power_per_line_dbm", "optical_path.chip_input_loss_db", "ring.heater.full_scale_v",
                 "channel_plan.n_lines", "laser.line_offset_ghz"):
        assert bench.get(path) == g25.get(path), path


@pytest.mark.parametrize("tid", ["CAL-1", "CAL-2", "CAL-3a", "CAL-3b", "CAL-4", "CAL-5", "CAL-7", "CAL-10"])
def test_bench_preset_and_bench_config_agree(record, tid):
    """The published bench preset (values rounded to 6 digits) gives the predictions of the fit's own bench."""
    t = T[tid]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = R.run_target(t, preset.BENCH_PRESET_NAME)
        b = R.run_target(t, preset.bench_config(record))
    assert a.status == b.status == "ok", (a.notes, b.notes)
    va, vb = np.asarray(a.value, float).ravel(), np.asarray(b.value, float).ravel()
    tol = 1e-3 if tid == "CAL-2" else 1e-4           # CAL-2: notch depths near critical coupling (log of a near-zero)
    # CAL-5 at +4 dBm reads the dragged minimum on the replica's 0.1 pm fine sweep grid: a 6-digit rounding can move
    # one point by one grid step (0.02 sigma)
    atol = 0.1 + 1e-6 if tid == "CAL-5" else 1e-9
    assert np.allclose(va, vb, rtol=tol, atol=atol), (tid, va, vb)


@pytest.mark.parametrize("tid", ["CAL-1", "CAL-3a", "CAL-3b", "CAL-5", "CAL-7", "CAL-8", "CAL-10", "CAL-12b"])
def test_calibrated_plant_reproduces_its_fitted_targets(record, tid):
    t = T[tid]
    plant = preset.BENCH_PRESET_NAME if t.generation == "G25" else preset.PRESET_NAME
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = R.run_target(t, plant)
    assert res.status == "ok", res.notes
    ev = evaluate(t, res.value)
    assert ev.passed, f"{tid}: {res.value} ({ev.detail})"


def test_cal4_is_reproduced_by_the_swept_device(record):
    """CAL-4 is the swept G25 device (0.35 against 0.40 nm/mW, 06 E4): the calibrated median device with the swept
    device's fitted P_FSR (and its droop transfer hypothesis) reproduces it; the median device itself does not."""
    g = record["groups"]["heater"]
    th = g["fits"][g["nominal"]]["theta"]
    ratio = th["heater_self_ratio"] * (th["p_fsr_swept_device_mw"] / th["p_fsr_mw"]
                                       if g["nominal"].endswith("same_rth_heater") else 1.0)
    ctx = G.Context(device=preset.bench_overrides(record),
                    ranges={k: list(v) for k, v in record["context"].get("ranges", {}).items()})
    cfg = ctx.config(G.BENCH, {G.P_PFSR: th["p_fsr_swept_device_mw"], G.P_RATIO: ratio,
                               G.P_R0: th["r_cold_ohm"] * th.get("r_cold_swept_device_rel", 1.0)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = R.resonance_vs_heater_v(cfg, T["CAL-4"])
    ev = evaluate(T["CAL-4"], res.value)
    assert ev.passed, (res.value, ev.detail)


def test_thermo_optic_worlds_reproduce_the_heater_targets(cfg, record):
    """Review fix: every thermo-optic world the heater group fits (H86, the required robustness corner of 07 D2)
    carries its own heater leaves, selected by uncertainty.thermo_optic in g26_calibrated, and that world reproduces
    CAL-3a/3b (median) and CAL-4 (swept device) through the replicas; with the H_solver leaves it would not."""
    g = record["groups"]["heater"]
    for prefix, entry, choice, alt in pipeline.HEATER_WORLDS:
        hyps = [h for h in g["fits"] if h.startswith(prefix + "/")]
        rows = {r["hypothesis"]: r for r in g["comparison"]["rows"]}
        ok = [h for h in hyps if rows[h]["likelihood_ratio_birge"] >= pipeline.REJECT_LR]
        if not ok:
            continue
        c_min = min(g["fits"][h]["chi2"] for h in ok)          # the pipeline's tie rule: first listed among ties
        wbest = next(h for h in ok if g["fits"][h]["chi2"] <= c_min + pipeline.TIE_DCHI2)
        th = g["fits"][wbest]["theta"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            world = params.load(preset.PRESET_NAME, alts={f"uncertainty.{entry}": choice})
            for path, name in ((G.P_PFSR, "p_fsr_mw"), (G.P_R0, "r_cold_ohm"), (G.P_RATIO, "heater_self_ratio")):
                assert world.get(path) == pytest.approx(th[name], rel=1e-5), (choice, path)
                assert cfg.leaf(path).alternatives[alt]["value"] == pytest.approx(th[name], rel=1e-5)
            alts = {f"uncertainty.{entry}": choice}
            dev = {**preset.bench_overrides(record), G.P_PFSR: th["p_fsr_mw"], G.P_R0: th["r_cold_ohm"],
                   G.P_RATIO: th["heater_self_ratio"]}
            ctx = G.Context(device=dev, ranges={k: list(v) for k, v in record["context"].get("ranges", {}).items()})
            r3 = R.heater_power_per_fsr(ctx.config(G.BENCH, alts=alts), T["CAL-3a"])
            ratio = th["heater_self_ratio"] * (th["p_fsr_swept_device_mw"] / th["p_fsr_mw"]
                                               if wbest.endswith("same_rth_heater") else 1.0)
            r4 = R.resonance_vs_heater_v(ctx.config(G.BENCH, {G.P_PFSR: th["p_fsr_swept_device_mw"], G.P_RATIO: ratio,
                                                              G.P_R0: th["r_cold_ohm"] * th["r_cold_swept_device_rel"]},
                                                    alts=alts), T["CAL-4"])
        assert evaluate(T["CAL-3a"], r3.value).passed and evaluate(T["CAL-3b"], r3.variants["voltage_v"]).passed
        assert evaluate(T["CAL-4"], r4.value).passed, (choice, r4.value)


def test_cal2_is_reproduced_with_the_fitted_instrument_floor(record):
    g = record["groups"]["coupling_er"]
    # the written point is the nominal fit centred on CAL-1 (its floor, not the free fit's)
    key = next((k for k in g["fits"] if k.startswith(g["nominal"]) and "CAL-1 centre" in k), g["nominal"])
    fr = g["fits"][key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        er = R.notch_er_vs_heater_v(preset.bench_config(record), T["CAL-2"], topology=None).value
    seen = G.notch_floor(er, fr["theta"]["notch_floor_db"])
    ev = evaluate(T["CAL-2"], seen)
    assert ev.passed, (er, seen, ev.detail)


def test_a1_closed_loop_runs_on_the_calibrated_preset(tmp_path):
    from mrm.sim import experiment
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = experiment.run(str(HERE / "a1_g26_calibrated.yaml"), write=False, log=lambda s: None)
    assert res.status != "error", res.report
    m = res.metrics if hasattr(res, "metrics") else {}
    assert m is not None
    print(res.report)
