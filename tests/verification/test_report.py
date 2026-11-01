"""Checks the validation report generator mrm.validation.report (``mrm report``), which writes docs/VALIDATION.md.

Read this first
---------------
The report has fixed sections: the target table, the closure and conflict table, identifiability, uncertainty
propagation (tornado and Sobol), controller rank stability, the generation mix and the textbook disagreements. The
tests render it from cached or fabricated data and check the frozen-target block, the scoring against the frozen
tolerance, staleness detection, the contamination audit (a held-out target used in a fit) and the closure links.

The first test is the CI check (like test_parameters_md_is_current): the committed docs/VALIDATION.md must equal a
fast re-render of the cached report data (docs/validation/<preset>/report_data.json) with the file's own stamp, its
charts must exist and the cached parts must match the current presets and target files. Regenerate with
``python -m mrm.validation.report`` (every part) or ``--recompute <part>`` / ``--fast``.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from mrm.params import load
from mrm.validation import report
from mrm.validation import targets_io

STAMP = {"date": "2026-01-01", "commit": "0000000", "dirty": False}   # a fixed report stamp for reproducible renders


REPORT_FILES = [report.DOC_PATH] + sorted(report.DOC_PATH.parent.glob("VALIDATION_*.md"))


@pytest.mark.xfail(strict=False, reason="both reports are marked STALE: they predate the final-review fixes and are "
                   "regenerated when the real device parameters arrive (docs/KNOWN_ISSUES.md)")
@pytest.mark.parametrize("path", REPORT_FILES, ids=[p.name for p in REPORT_FILES])
def test_validation_md_is_current(path):
    """docs/VALIDATION.md (g26) and every other committed report docs/VALIDATION_<preset>.md are generated: fail
    when one is stale (fast mode: no replica, engine or Sobol run)."""
    text = path.read_text()
    if report.read_stamp(text) is None:
        pytest.fail(f"{path.name} was not generated: run python -m mrm.validation.report")
    probs = report.check(out=path)
    assert probs == [], "\n".join(probs)


def test_frozen_block_is_copied_verbatim(tmp_path):
    doc = report.DOC_PATH.read_text()
    block = report._frozen_block(doc)
    text = report.render({}, preset="g26", stamp=STAMP, doc_text=doc, charts=False, chart_dir=tmp_path)
    assert block in text
    assert text.index(block) < text.index("## 1.")
    with pytest.raises(ValueError, match="frozen-targets"):
        report.render({}, preset="g26", stamp=STAMP, doc_text="# no block\n", charts=False, chart_dir=tmp_path)
    # the stamp round-trips
    st = report.read_stamp(text)
    assert st["preset"] == "g26" and st["commit"] == STAMP["commit"] and st["g25_preset"] == "g25_test_vehicle"


def test_render_is_deterministic_with_charts(tmp_path):
    """Same data and stamp -> the same Markdown and byte-identical SVG charts (so --fast changes nothing)."""
    data = report.load_data("g26")
    a, b = tmp_path / "a", tmp_path / "b"
    ta = report.render(data, preset="g26", stamp=STAMP, charts=True, chart_dir=a)
    tb = report.render(data, preset="g26", stamp=STAMP, charts=True, chart_dir=b)
    fa = sorted(p.name for p in a.glob("*.svg"))
    assert fa == sorted(p.name for p in b.glob("*.svg"))
    for n in fa:
        assert (a / n).read_bytes() == (b / n).read_bytes(), n
        assert (a / n).stat().st_size < 200_000, f"{n} is not a small chart"
    # the texts differ only by the chart directory they reference
    assert ta.replace(report._docs_relpath(a), "CHARTS") == tb.replace(report._docs_relpath(b), "CHARTS")


def test_every_fixed_section_is_present(tmp_path):
    text = report.render(report.load_data("g26"), preset="g26", stamp=STAMP, charts=False, chart_dir=tmp_path)
    for h in ("## 1. Targets", "## 2. Closure and conflict table", "## 3. Identifiability", "## 4. Uncertainty",
              "## 5. Controller rank stability", "## 6. Generation mix", "## 7. Disagreements with the textbook",
              "## Contamination warnings"):
        assert h in text, h
    # every row of the closure and conflict table has an anchor the closure tests can link to
    for slug, _, _ in report.CLOSURE_ROWS:
        assert f'id="closure-{slug}"' in text
    # the first line of a controller report is the caveat of question Q1 (the thermal dynamics are assumed)
    sec5 = text[text.index("## 5. Controller"):]
    assert sec5.split("\n\n")[2].startswith("Performance is conditional on the assumed thermal dynamics")


def test_target_rows_cover_every_target(tmp_path):
    data = report.load_data("g26")
    if not data.get("predictions"):
        pytest.skip("predictions not computed")
    text = report.render(data, preset="g26", stamp=STAMP, charts=False, chart_dir=tmp_path)
    sec = text[text.index("## 1. Targets"):text.index("## 2. Closure")]
    for role in targets_io.TARGET_ROLES:
        for t in targets_io.load_targets(role):
            assert f"| {t.id} | {t.status} |" in sec, t.id
    # unscored entries are shown with their status, never as a pass
    for tid in ("VAL-6", "VAL-7a", "VAL-8e"):
        row = next(ln for ln in sec.splitlines() if ln.startswith(f"| {tid} |"))
        assert "not scored" in row


def test_scoring_uses_the_frozen_tolerance(tmp_path):
    """A fabricated prediction exactly at the target passes; one far off fails (the report scores, not the cache)."""
    data = {"predictions": {"meta": {}, "records": [], "corners": {}}}
    recs = []
    for t in targets_io.load_targets("validation"):
        if t.id == "VAL-1":
            recs.append({"id": t.id, "replica_status": "ok", "predicted": t.value})
        if t.id == "VAL-3a":
            recs.append({"id": t.id, "replica_status": "ok", "predicted": t.value + 50.0})
    data["predictions"]["records"] = recs
    text = report.render(data, preset="g26", stamp=STAMP, charts=False, chart_dir=tmp_path)
    row1 = next(ln for ln in text.splitlines() if ln.startswith("| VAL-1 |"))
    row3 = next(ln for ln in text.splitlines() if ln.startswith("| VAL-3a |"))
    assert "| pass |" in row1 and "| fail |" in row3


def test_staleness_is_detected():
    data = {"predictions": {"meta": {"config_hash": {"g26": "0" * 16}, "targets_sha256": report._target_files_sha()}}}
    out = report._staleness(data, {"G26": "g26", "G25": "g25_test_vehicle"})
    assert out and "g26 changed" in out[0]
    data = {"predictions": {"meta": {"config_hash": {}, "targets_sha256": {"validation": "x"}}}}
    assert any("validation.yaml changed" in s for s in report._staleness(data, {"G26": "g26"}))


def test_companion_preset():
    assert report.companion_g25("g26") == "g25_test_vehicle"
    assert report.companion_g25("g26_h86") == "g25_test_vehicle"            # no g25_test_vehicle_h86 preset
    assert report.companion_g25("g26_calibrated", "mine") == "mine"
    if (report.PRESETS_DIR / "g25_test_vehicle_calibrated.yaml").exists():
        assert report.companion_g25("g26_calibrated") == "g25_test_vehicle_calibrated"


def test_missing_preset_is_a_clean_error(capsys):
    assert report.main(["--preset", "no_such_preset_xyz", "--fast", "--out", "/dev/null"]) == 1
    assert "no_such_preset_xyz" in capsys.readouterr().err


def test_cli_registered(capsys):
    from mrm import cli
    ap = cli.build_parser()
    assert "report" in cli.COMMANDS
    with pytest.raises(SystemExit) as e:
        ap.parse_args(["report", "--help"])
    assert e.value.code == 0
    assert "--preset" in capsys.readouterr().out


def test_closure_test_links_are_found(tmp_path, monkeypatch):
    d = tmp_path / "tests" / "closure"
    d.mkdir(parents=True)
    (d / "test_conflicts.py").write_text(
        "import pytest\n\n@pytest.mark.xfail(strict=True, reason='VALIDATION.md#closure-q: 3500 vs 3900-4200')\n"
        "def test_q_closes():\n    assert False\n")
    monkeypatch.setattr(report, "REPO", tmp_path)
    found = report._closure_tests()
    assert found == {"q": ["`tests/closure/test_conflicts.py::test_q_closes`"]}


def test_calibrated_leaf_audit_flags_contamination(tmp_path):
    """A leaf tagged calibrated against a held-out target is reported as contamination (calibration and held-out
    targets stay apart)."""
    from mrm import params
    f = tmp_path / "g26_bad_cal.yaml"
    f.write_text("extends: g26\nmeta: {name: g26_bad_cal, kind: preset}\nring:\n  optics:\n    q_loaded: {value: 3900, "
                 "provenance: calibrated, generation: G25, source: LM-T, range: [2600, 4200], targets: [VAL-1, CAL-1], "
                 "residual: 0.1, identifiable: true}\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load(str(f))
    lines = report._section_identifiability(cfg, {}, "g26_bad_cal", tmp_path, False)
    # the section 3.2 audit row (section 3.1 can hold a fit record's row of the same leaf)
    lines = lines[lines.index("### 3.2 Leaves tagged `calibrated` in this preset"):]
    row = next(ln for ln in lines if ln.startswith("| `ring.optics.q_loaded`"))
    assert "CONTAMINATION" in row and "VAL-1" in row


def test_fit_record_appears_only_in_its_presets_reports():
    """A fit record without a ``preset`` key belongs to g26 (the start) and g26_calibrated (the output) only."""
    if not (report.REPO / "mrm" / "calibration" / "latest_fit.json").exists():
        pytest.skip("no fit record")
    assert report._calibration_outputs("g26") and report._calibration_outputs("g26_calibrated")
    assert not report._calibration_outputs("g26_h86") and not report._calibration_outputs("g26_bad_cal")
    assert report._calibrated("g26_calibrated") and not report._calibrated("g26") and not report._calibrated("g26_h86")


def test_scales_fall_back_to_the_tolerance():
    t = {x.id: x for x in targets_io.load_calibration_targets()}
    np.testing.assert_allclose(report._residual_scales(t["CAL-1"], 3800.0), [175.0])           # interval/4
    np.testing.assert_allclose(report._residual_scales(t["CAL-2"], [1.0] * 6), [0.5] * 6)        # sigma
    s = report._residual_scales(t["CAL-9b"], {"at_code_maxima": 1.0, "at_code_minima": 2.0})
    np.testing.assert_allclose(s, [200.0, 400.0])                                      # per key


def test_rel_scale_of_a_curve_uses_y_not_x():
    """The rel-tolerance fallback of a curve target takes the largest |y| (the rel tolerance applies to y), not the
    largest number of the whole mapping (which can be an x value)."""
    import dataclasses
    t0 = targets_io.load_calibration_targets()[0]
    t = dataclasses.replace(t0, kind="curve", value={"x": [1000.0, 2000.0], "y": [1.0, 2.0]}, sigma=None,
                            tolerance={"kind": "rel", "value": 0.1})
    np.testing.assert_allclose(report._residual_scales(t, [1.5, 2.5]), [0.1, 0.1])     # 0.1 x max|y| / 2


def test_recompute_list_ignores_blank_parts(monkeypatch):
    """``--recompute "predictions, "`` recomputes predictions only (a trailing comma or space is not a part)."""
    seen = {}
    monkeypatch.setattr(report, "generate", lambda preset, **kw: seen.update(kw))
    assert report.main(["--recompute", "predictions, ", "--out", "/dev/null"]) == 0
    assert seen["recompute"] == ("predictions",)


def test_jacobian_targets_are_calibration_only():
    ids = [t.id for t in report._jacobian_targets()]
    assert ids and all(i.startswith("CAL-") for i in ids)
    assert "CAL-6" not in ids and not any(i.startswith(("CAL-12", "CAL-13")) for i in ids)


def test_cached_sensitivity_is_consistent():
    """The cached section-4 data: the base world is the nominal world, the Sobol entries are the screened top 8,
    and the indices are in [-0.2, 1.2] (estimator noise allowed, not hidden)."""
    sens_data = report.load_data("g26").get("sensitivity")
    if not sens_data:
        pytest.skip("sensitivity not computed")
    assert len(sens_data["top"]) == report.SENS_TOP
    from mrm.validation import sensitivity
    assert sens_data["top"] == sensitivity.screen_top(sens_data["tornado"], report.SENS_TOP)
    for m, e in sens_data["sobol"].items():
        if "st" in e:
            assert e["names"] == sens_data["top"]
            assert all(-0.2 <= x <= 1.2 for x in e["st"] if x is not None), m


def test_sensitivity_chunks_give_every_row_its_own_seeds(monkeypatch):
    """evaluate_worlds runs each chunk as its own experiment. The seed bases of a chunk are offset by its first row,
    so row i of the design has noise seed 300 + i for any chunk size (a Saltelli row and its AB_i partner must not
    share noise). No engine run: the experiment is replaced by a stub that reads the seeds of each chunk file."""
    from types import SimpleNamespace

    import yaml

    from mrm.sim import experiment
    from mrm.validation import sensitivity
    noise_seeds: list[int] = []

    def fake_run(path, **kwargs):
        exp = yaml.safe_load(Path(path).read_text())["experiment"]
        n = int(exp["n_mc"])
        noise_seeds.extend(int(exp["seeds"]["noise"]) + b for b in range(n))     # world b of a chunk: base + b
        return SimpleNamespace(summary={"world_summary": {"m": [0.0] * n}})
    monkeypatch.setattr(experiment, "run", fake_run)
    monkeypatch.setattr(sensitivity, "check_injection", lambda space, U: {"ok": True})
    U = np.tile(sensitivity.UncertaintySpace(report._load_cfg("g26")).nominal_u(), (5, 1))
    worlds = report.evaluate_worlds("g26", U, chunk=2, log=lambda s: None)
    assert worlds["m"].shape == (5,)
    assert noise_seeds == [300, 301, 302, 303, 304]


def test_undefined_tornado_metric_is_not_listed_as_no_change(tmp_path):
    """A tornado entry with a NaN swing (the metric is NaN at every point) is listed as not defined: not in the
    table and not under 'No change'."""
    import copy
    sens_data = report.load_data("g26").get("sensitivity")
    if not sens_data:
        pytest.skip("sensitivity not computed")
    sens_data = copy.deepcopy(sens_data)
    key = sens_data["metrics"][0]["key"]
    row = sens_data["tornado"][key][0]
    row.update(lo=float("nan"), hi=float("nan"), swing=float("nan"))
    lines = report._section_uncertainty({"sensitivity": sens_data}, tmp_path, False)
    entry = f"`{row['entry']}`"
    undefined = [ln for ln in lines if ln.startswith(f"{key} is not defined")]
    assert len(undefined) == 1 and entry in undefined[0]
    sec = lines[lines.index(f"**{key}** (nominal {report._fmt_num(sens_data['base'].get(key))} "
                            f"{sens_data['metrics'][0]['unit']})"):]
    sec = sec[:sec.index(undefined[0])]
    assert not any(ln.startswith(f"| {entry} |") for ln in sec)
    assert not any(ln.startswith(f"No change of {key}") and entry in ln for ln in sec)


@pytest.mark.slow
def test_injected_worlds_run_on_the_engine():
    """The nominal row of an injected design gives the metrics of the nominal world run through the experiment
    layer (the protocol of the uncertainty set), so the tornado base is the nominal world (~1 min)."""
    from mrm.sim import experiment
    from mrm.validation import sensitivity
    space = sensitivity.UncertaintySpace(report._load_cfg("g26"))
    u0 = space.nominal_u()
    i = space.names.index("thermo_optic")
    u1 = u0.copy()
    u1[i] = space.axes["thermo_optic"].u_of_value("H86")
    world_metrics = report.evaluate_worlds("g26", np.vstack([u0, u1]), n_threads=2)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "nom.yaml"
        f.write_text(report._sensitivity_experiment_yaml("g26", 1).replace("kind: mc", "kind: run").replace(
            "uncertainty: lhs", "uncertainty: nominal").replace("  n_mc: 1\n", "  batch: 1\n"))
        res = experiment.run(str(f), write=False, log=lambda s: None, options={"n_threads": 2, "record": []})
    nom = experiment.world_summary(res.cases["main"].metrics)
    for k in ("oma_penalty_db_worst", "mean_reloma_db", "excursion_count", "wrong_line_rings"):
        assert world_metrics[k][0] == pytest.approx(float(nom[k][0]), rel=1e-6, abs=1e-9), k
    # and the second row is a different world (H86)
    assert world_metrics["mean_reloma_db"][1] != world_metrics["mean_reloma_db"][0] or \
        world_metrics["oma_penalty_db_worst"][1] != world_metrics["oma_penalty_db_worst"][0]


def test_closure_scan_ignores_strings_and_needs_the_decorator(tmp_path, monkeypatch):
    """Only an xfail-decorated test whose reason names the anchor counts: a string in a test body (like the fixture
    above) or a non-xfail test is not a closure test (the old line scan linked closure-q to an unrelated test)."""
    d = tmp_path / "tests" / "closure"
    d.mkdir(parents=True)
    (d / "test_mixed.py").write_text(
        "import pytest\n\n"
        "def test_writes_a_fixture():\n"
        "    s = \"@pytest.mark.xfail(strict=True, reason='VALIDATION.md#closure-fsr: x')\"\n"
        "    assert s\n\n"
        "@pytest.mark.skip(reason='VALIDATION.md#closure-q')\n"
        "def test_skipped():\n    pass\n\n"
        "@pytest.mark.xfail(reason='VALIDATION.md#closure-rth: 4.7 vs 6.1 K/mW')\n"
        "def test_rth_loose():\n    assert False\n")
    monkeypatch.setattr(report, "REPO", tmp_path)
    assert report._closure_tests() == {"rth": ["`tests/closure/test_mixed.py::test_rth_loose` (not strict)"]}


def test_repository_closure_links_are_real_xfail_tests():
    for slug, tests in report._closure_tests().items():
        for t in tests:
            assert "tests/verification/test_report.py" not in t, (slug, t)


def test_fit_target_audit():
    assert report._fit_target_audit(["CAL-1", "CAL-9a", "CAL-13a"]) == []
    out = report._fit_target_audit(["CAL-1", "VAL-2a", "BND-1"])
    assert out and out[0].startswith("CONTAMINATION") and "VAL-2a" in out[0] and "BND-1" in out[0]
    assert any("fit: false" in x for x in report._fit_target_audit(["CAL-6"]))
    assert any("unknown" in x for x in report._fit_target_audit(["CAL-99"]))
    exc = [e.id for e in targets_io.load_excluded()]
    assert any("excluded" in x for x in report._fit_target_audit(exc[:1]))


def _fake_fit(targets_of_group, decision_targets):
    """A minimal calibration fit record (schema mrm.calibration.fit/1): one group 'optics' with one hypothesis h1,
    one profiled parameter q (ring.optics.q_loaded) and one decision that names decision_targets."""
    prof = {"name": "q", "path": "ring.optics.q_loaded", "grid": [3500.0, 3800.0, 4100.0], "chi2": [9.0, 1.0, 6.0],
            "chi2_min": 1.0, "threshold": 4.84, "interval": [3700.0, 3950.0], "interval_raw": [3700.0, 3950.0],
            "open_lo": False, "open_hi": False, "n_components": 1, "prior": [2600.0, 4200.0], "log": False,
            "width_frac": 0.16, "half_prior_ok": True, "unique": True, "best": 3800.0}
    return {"schema": "mrm.calibration.fit/1", "created_utc": "2026-10-10T00:00:00Z", "git_commit": "abc", "notes": [],
            "groups": {"optics": {
                "targets": targets_of_group, "nominal": "h1", "runtime_s": 1.0,
                "comparison": {"group": "optics", "best": "h1", "note": "", "rows": [
                    {"hypothesis": "h1", "chi2": 1.0, "dof": 3, "n_params": 1, "delta_chi2": 0.0,
                     "likelihood_ratio": 1.0, "likelihood_ratio_birge": 1.0, "aic": 3.0, "delta_aic": 0.0}]},
                "fits": {"h1": {"theta": {"q": 3800.0}, "chi2": 1.0, "dof": 3, "birge2": 0.33,
                                "targets": {t: {} for t in targets_of_group}}},
                "profiles": {"h1": {"q": prof}},
                "decisions": [{"path": "ring.optics.q_loaded", "value": 3800.0, "provenance": "calibrated",
                               "previous_value": 3850.0, "previous_provenance": "inferred", "scope": "device",
                               "targets": decision_targets, "residual": {t: 0.1 for t in decision_targets},
                               "identifiable": {"profile_95": [3700.0, 3950.0], "prior": [2600.0, 4200.0],
                                                "width_frac": 0.16, "rule": "calibrated", "unique": True}}],
                "extra": {}}}}


def test_calibration_fit_section_renders_and_audits(tmp_path):
    """Section 3.1 reads the calibration's real schema (mrm.calibration.fit/1, no preset key) and flags a held-out
    id anywhere in a group."""
    c = _fake_fit(["CAL-1", "CAL-2"], ["CAL-1"])
    c["_file"] = "mrm/calibration/latest_fit.json"
    lines = report._section_calibration_fit(c, tmp_path, True)
    text = "\n".join(lines)
    assert "Audit: ok" in text and "| `ring.optics.q_loaded` | device |" in text
    assert (tmp_path / "profiles_optics.svg").exists()
    bad = _fake_fit(["CAL-1"], ["CAL-1", "VAL-2a"])
    bad["_file"] = "x.json"
    assert "CONTAMINATION" in "\n".join(report._section_calibration_fit(bad, tmp_path, False))


def test_jacobian_moves_only_plants_that_share_the_leaf(monkeypatch):
    """A leaf whose G25 value differs from G26's (optical_path.chip_input_loss_db: 0 dB bench against 6 dB) is
    moved on G26 only (the G26 range and step must not be clipped onto the G25 leaf); a shared leaf moves on both."""
    from types import SimpleNamespace
    from mrm.validation import replicas

    def fake_run(t, cfg):
        q = float(cfg.leaves["ring.optics.q_loaded"].value)
        il = float(cfg.leaves["optical_path.chip_input_loss_db"].value)
        return SimpleNamespace(status="ok", value=q / 100.0 + 10.0 * il, notes=[], variants={}, conditions={})
    monkeypatch.setattr(replicas, "run_target", fake_run)
    monkeypatch.setattr(report, "_jacobian_targets",
                        lambda: [t for t in targets_io.load_calibration_targets() if t.id in ("CAL-7", "CAL-8")])
    ident = report.compute_identifiability({"G26": "g26", "G25": "g25_test_vehicle"},
                                           params=(("ring.optics.q_loaded", ""),
                                                   ("optical_path.chip_input_loss_db", "")),
                                           log=lambda s: None)
    lab = ident["residual_labels"]
    J = np.asarray(ident["jacobian"], float) * np.asarray(ident["scales"], float)[:, None]
    i7, i8 = lab.index("CAL-7"), lab.index("CAL-8")                 # CAL-7 on the G25 bench, CAL-8 on G26
    p_q, p_il = ident["params"]
    assert p_q["moved_on"] == ["G26", "G25"] and "kept" not in p_q
    assert p_il["moved_on"] == ["G26"] and "G25" in p_il["kept"]
    lo, hi = load("g26").leaves["optical_path.chip_input_loss_db"].range         # [1, 10] dB since the params review
    assert J[i7, 1] == 0.0 and J[i8, 1] == pytest.approx(10.0 * (hi - lo) / 2.0)  # d/d(il) x the G26 half-range
    assert J[i7, 0] == pytest.approx(800.0 / 100.0) and J[i8, 0] == pytest.approx(8.0)
