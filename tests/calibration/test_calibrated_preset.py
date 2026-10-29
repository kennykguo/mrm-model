"""Check the calibrated preset g26_calibrated against the fit record mrm/calibration/latest_fit.json.

Read this first. ``python -m mrm.calibration`` fits the calibration targets and writes the fit record; the preset
writer turns the record into g26_calibrated (and the G25 bench preset g25_test_vehicle_calibrated). These tests
check that the presets, the record and the replicas agree. They carry the marker ``calibration``: their results are
reported and do not block CI. A test that needs the record is skipped when the record is missing.

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
   g25_test_vehicle_calibrated equals the record, g26_calibrated and bench_config (also in the H86 world);
8. the LUT-mode bench mapping of the CAL-5 fill-fraction variant loads, and it is g25_test_vehicle except for the
   junction law (no fit record needed).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from mrm import params
from mrm.calibration import groups
from mrm.calibration import pipeline, preset
from mrm.calibration.core import calibration_targets
from mrm.validation import replicas
from mrm.validation.targets_io import evaluate

pytestmark = pytest.mark.calibration

HERE = Path(__file__).resolve().parent
TARGETS = calibration_targets()        # target id -> calibration target (calibration.yaml)


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
    """Every leaf decision of every calibration group of the fit record."""
    for g in record["groups"].values():
        for d in g["decisions"]:
            yield d


def test_every_fitted_residual_is_a_calibration_target(record):
    """Held-out isolation: every target id in the record (group targets, per-hypothesis residuals, leaf decisions,
    composed worlds) is a calibration.yaml id that may be fitted (CAL-6, marked fit: false, never appears)."""
    fit_ok = {tid for tid, t in TARGETS.items() if t.fit}
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
    alt_world = {alt: (world, leaf) for world, _h, _held, free in pipeline.COUPLING_WORLDS
                 for leaf, alt in free.items()}
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
    """The loss-mechanism hypotheses and worlds run with the coupler artefact's dispersion, never with the
    dispersion fitted under the dispersion mechanism: recomputing their chi2 from a fresh problem reproduces the
    record."""
    g = record["groups"]["coupling_er"]
    disp_solver = params.load(groups.BENCH).get(groups.LEAF_COUPLER_DISPERSION)
    # the problems' own context (the heater leaves fitted before coupling_er) plus the dispersion fitted for the
    # nominal world (as the pipeline's context carries it after the group)
    leak = pipeline.coupling_context(record)
    leak.device[groups.LEAF_COUPLER_DISPERSION] = record["context"]["device"][groups.LEAF_COUPLER_DISPERSION]
    probs = {p.hypothesis: p for p in groups.coupling_er_problems(leak, TARGETS)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for h in ("over/ring_t_loss/floor", "over/heater_t_loss/floor"):
            assert g["fits"][h]["fixed"][groups.LEAF_COUPLER_DISPERSION] == pytest.approx(disp_solver, rel=1e-12)
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
    leaf decisions. (A merged review record once kept three values from before the review: this test catches that.)"""
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
        g25 = params.load(groups.BENCH)
        h86 = params.load(preset.BENCH_PRESET_NAME, alts={"uncertainty.thermo_optic": "H86"})
        h86_cal = params.load(preset.PRESET_NAME, alts={"uncertainty.thermo_optic": "H86"})
    for path, v in record["context"]["device"].items():
        assert bench.get(path) == pytest.approx(v, rel=1e-5), path
        assert bc.get(path) == pytest.approx(v, rel=1e-12), path
        assert bench.get(path) == cfg.get(path), path
        assert bench.leaf(path).provenance == cfg.leaf(path).provenance, path
        assert (bench.leaf(path).alternatives or {}) == (cfg.leaf(path).alternatives or {}), path
    for path in (groups.LEAF_P_FSR, groups.LEAF_R_COLD, groups.LEAF_HEATER_SELF_RATIO):
        assert h86.get(path) == h86_cal.get(path) != bench.get(path), path
    # the bench's own leaves are g25_test_vehicle's
    for path in ("laser.power_per_line_dbm", "optical_path.chip_input_loss_db", "ring.heater.full_scale_v",
                 "channel_plan.n_lines", "laser.line_offset_ghz"):
        assert bench.get(path) == g25.get(path), path


def test_merged_bench_mapping_loads_to_the_bench_preset():
    """The bench preset merged into one mapping by the loader's rules loads to the same config as its name."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_name = params.load(groups.BENCH)
        by_mapping = params.load(groups._merged_preset_mapping(groups.BENCH))
    assert by_mapping.resolved_tree() == by_name.resolved_tree()
    assert by_mapping.hash() == by_name.hash()


def test_lut_bench_mapping_loads_and_changes_only_the_junction_law():
    """The CAL-5 LUT variant loads (it raised ConfigError: a null alternative and a g26 corner that
    g25_test_vehicle removes were kept) and differs from g25_test_vehicle only in the junction law."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bench = params.load(groups.BENCH)
        lut = params.load(groups.bench_lut_mapping())
    assert lut.get("fidelity.junction_model") == "lut"
    changed_leaves = {p for p in set(bench.leaves) | set(lut.leaves)
                      if p not in bench.leaves or p not in lut.leaves or bench.get(p) != lut.get(p)}
    assert changed_leaves == {"ring.junction.eff_at_0v_pm_per_v", "ring.junction.v_bi_v",
                              "ring.junction.depletion_exponent"}
    changed_settings = {p for p in set(bench.settings) | set(lut.settings)
                        if bench.settings.get(p) != lut.settings.get(p)}
    assert changed_settings == {"fidelity.junction_model", "ring.junction.lut_file"}
    assert {n: e.fields for n, e in lut.uncertainty.items()} == {n: e.fields for n, e in bench.uncertainty.items()}
    assert {p: e.fields for p, e in lut.variation.items()} == {p: e.fields for p, e in bench.variation.items()}


@pytest.mark.parametrize("tid", ["CAL-1", "CAL-2", "CAL-3a", "CAL-3b", "CAL-4", "CAL-5", "CAL-7", "CAL-10"])
def test_bench_preset_and_bench_config_agree(record, tid):
    """The published bench preset (values rounded to 6 digits) gives the predictions of the fit's own bench."""
    t = TARGETS[tid]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = replicas.run_target(t, preset.BENCH_PRESET_NAME)
        b = replicas.run_target(t, preset.bench_config(record))
    assert a.status == b.status == "ok", (a.notes, b.notes)
    va, vb = np.asarray(a.value, float).ravel(), np.asarray(b.value, float).ravel()
    tol = 1e-3 if tid == "CAL-2" else 1e-4           # CAL-2: notch depths near critical coupling (log of a near-zero)
    # CAL-5 at +4 dBm reads the dragged minimum on the replica's 0.1 pm fine sweep grid: a 6-digit rounding can move
    # one point by one grid step (0.02 sigma)
    atol = 0.1 + 1e-6 if tid == "CAL-5" else 1e-9
    assert np.allclose(va, vb, rtol=tol, atol=atol), (tid, va, vb)


@pytest.mark.parametrize("tid", ["CAL-1", "CAL-3a", "CAL-3b", "CAL-5", "CAL-7", "CAL-8", "CAL-10", "CAL-12b"])
def test_calibrated_plant_reproduces_its_fitted_targets(record, tid):
    t = TARGETS[tid]
    plant = preset.BENCH_PRESET_NAME if t.generation == "G25" else preset.PRESET_NAME
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = replicas.run_target(t, plant)
    assert res.status == "ok", res.notes
    ev = evaluate(t, res.value)
    assert ev.passed, f"{tid}: {res.value} ({ev.detail})"


def test_cal4_is_reproduced_by_the_swept_device(record):
    """CAL-4 is the swept G25 device, not the wafer median (0.35 against 0.40 nm/mW; docs/research/06_textbook_audit.md,
    item E4). The calibrated median device with the swept device's fitted P_FSR (and its droop transfer hypothesis)
    reproduces it; the median device itself does not."""
    g = record["groups"]["heater"]
    th = g["fits"][g["nominal"]]["theta"]
    ratio = th["heater_self_ratio"] * (th["p_fsr_swept_device_mw"] / th["p_fsr_mw"]
                                       if g["nominal"].endswith("same_rth_heater") else 1.0)
    ctx = groups.Context(device=preset.bench_overrides(record),
                         ranges={k: list(v) for k, v in record["context"].get("ranges", {}).items()})
    cfg = ctx.config(groups.BENCH, {groups.LEAF_P_FSR: th["p_fsr_swept_device_mw"],
                                    groups.LEAF_HEATER_SELF_RATIO: ratio,
                                    groups.LEAF_R_COLD: th["r_cold_ohm"] * th.get("r_cold_swept_device_rel", 1.0)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = replicas.resonance_vs_heater_v(cfg, TARGETS["CAL-4"])
    ev = evaluate(TARGETS["CAL-4"], res.value)
    assert ev.passed, (res.value, ev.detail)


def test_thermo_optic_worlds_reproduce_the_heater_targets(cfg, record):
    """Every thermo-optic world that the heater group fits (H86: the required robustness corner of the thermo-optic
    hypotheses) carries its own heater leaves, selected by uncertainty.thermo_optic in g26_calibrated. That world
    reproduces CAL-3a/3b (median device) and CAL-4 (swept device) through the replicas; with the H_solver leaves it
    would not."""
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
            for path, name in ((groups.LEAF_P_FSR, "p_fsr_mw"), (groups.LEAF_R_COLD, "r_cold_ohm"),
                               (groups.LEAF_HEATER_SELF_RATIO, "heater_self_ratio")):
                assert world.get(path) == pytest.approx(th[name], rel=1e-5), (choice, path)
                assert cfg.leaf(path).alternatives[alt]["value"] == pytest.approx(th[name], rel=1e-5)
            alts = {f"uncertainty.{entry}": choice}
            dev = {**preset.bench_overrides(record), groups.LEAF_P_FSR: th["p_fsr_mw"],
                   groups.LEAF_R_COLD: th["r_cold_ohm"], groups.LEAF_HEATER_SELF_RATIO: th["heater_self_ratio"]}
            ctx = groups.Context(device=dev,
                                 ranges={k: list(v) for k, v in record["context"].get("ranges", {}).items()})
            r3 = replicas.heater_power_per_fsr(ctx.config(groups.BENCH, alts=alts), TARGETS["CAL-3a"])
            ratio = th["heater_self_ratio"] * (th["p_fsr_swept_device_mw"] / th["p_fsr_mw"]
                                               if wbest.endswith("same_rth_heater") else 1.0)
            r4 = replicas.resonance_vs_heater_v(ctx.config(groups.BENCH,
                                                           {groups.LEAF_P_FSR: th["p_fsr_swept_device_mw"],
                                                            groups.LEAF_HEATER_SELF_RATIO: ratio,
                                                            groups.LEAF_R_COLD: th["r_cold_ohm"]
                                                            * th["r_cold_swept_device_rel"]},
                                                           alts=alts), TARGETS["CAL-4"])
        assert (evaluate(TARGETS["CAL-3a"], r3.value).passed
                and evaluate(TARGETS["CAL-3b"], r3.variants["voltage_v"]).passed)
        assert evaluate(TARGETS["CAL-4"], r4.value).passed, (choice, r4.value)


def test_cal2_is_reproduced_with_the_fitted_instrument_floor(record):
    """CAL-2 (notch depth against heater voltage) is reproduced once the fitted instrument floor of the notch is
    applied to the replica's notch depths."""
    g = record["groups"]["coupling_er"]
    # the written point is the nominal fit centred on CAL-1 (its floor, not the free fit's)
    key = next((k for k in g["fits"] if k.startswith(g["nominal"]) and "CAL-1 centre" in k), g["nominal"])
    fr = g["fits"][key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        er = replicas.notch_er_vs_heater_v(preset.bench_config(record), TARGETS["CAL-2"], topology=None).value
    seen = groups.notch_floor(er, fr["theta"]["notch_floor_db"])
    ev = evaluate(TARGETS["CAL-2"], seen)
    assert ev.passed, (er, seen, ev.detail)


def test_a1_closed_loop_runs_on_the_calibrated_preset(tmp_path):
    """The A1 acceptance experiment (a1_g26_calibrated.yaml next to this file) runs closed loop without an error."""
    from mrm.sim import experiment
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = experiment.run(str(HERE / "a1_g26_calibrated.yaml"), write=False, log=lambda s: None)
    assert res.status != "error", res.report
    m = res.metrics if hasattr(res, "metrics") else {}
    assert m is not None
    print(res.report)
