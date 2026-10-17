"""The validation report generator (mrm.validation.report, `mrm report`; 07 P1-23, 4.9).

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
from mrm.validation import report as rep
from mrm.validation import targets_io as tio

STAMP = {"date": "2026-01-01", "commit": "0000000", "dirty": False}


REPORT_FILES = [rep.DOC_PATH] + sorted(rep.DOC_PATH.parent.glob("VALIDATION_*.md"))


@pytest.mark.parametrize("path", REPORT_FILES, ids=[p.name for p in REPORT_FILES])
def test_validation_md_is_current(path):
    """docs/VALIDATION.md (g26) and every other committed report docs/VALIDATION_<preset>.md are generated: fail
    when one is stale (fast mode: no replica, engine or Sobol run)."""
    text = path.read_text()
    if rep.read_stamp(text) is None:
        pytest.fail(f"{path.name} was not generated: run python -m mrm.validation.report")
    probs = rep.check(out=path)
    assert probs == [], "\n".join(probs)


def test_frozen_block_is_copied_verbatim(tmp_path):
    doc = rep.DOC_PATH.read_text()
    block = rep._frozen_block(doc)
    text = rep.render({}, preset="g26", stamp=STAMP, doc_text=doc, charts=False, chart_dir=tmp_path)
    assert block in text
    assert text.index(block) < text.index("## 1.")
    with pytest.raises(ValueError, match="frozen-targets"):
        rep.render({}, preset="g26", stamp=STAMP, doc_text="# no block\n", charts=False, chart_dir=tmp_path)
    # the stamp round-trips
    st = rep.read_stamp(text)
    assert st["preset"] == "g26" and st["commit"] == STAMP["commit"] and st["g25_preset"] == "g25_test_vehicle"


def test_render_is_deterministic_with_charts(tmp_path):
    """Same data and stamp -> the same Markdown and byte-identical SVG charts (so --fast changes nothing)."""
    data = rep.load_data("g26")
    a, b = tmp_path / "a", tmp_path / "b"
    ta = rep.render(data, preset="g26", stamp=STAMP, charts=True, chart_dir=a)
    tb = rep.render(data, preset="g26", stamp=STAMP, charts=True, chart_dir=b)
    fa = sorted(p.name for p in a.glob("*.svg"))
    assert fa == sorted(p.name for p in b.glob("*.svg"))
    for n in fa:
        assert (a / n).read_bytes() == (b / n).read_bytes(), n
        assert (a / n).stat().st_size < 200_000, f"{n} is not a small chart"
    # the texts differ only by the chart directory they reference
    assert ta.replace(rep._rel(a), "CHARTS") == tb.replace(rep._rel(b), "CHARTS")


def test_every_fixed_section_is_present(tmp_path):
    text = rep.render(rep.load_data("g26"), preset="g26", stamp=STAMP, charts=False, chart_dir=tmp_path)
    for h in ("## 1. Targets", "## 2. Closure and conflict table", "## 3. Identifiability", "## 4. Uncertainty",
              "## 5. Controller rank stability", "## 6. Generation mix", "## 7. Disagreements with the textbook",
              "## Contamination warnings"):
        assert h in text, h
    # every 07 4.6 row has an anchor the closure tests can link to
    for slug, _, _ in rep.CLOSURE_ROWS:
        assert f'id="closure-{slug}"' in text
    # 07 4.8: the first line of a controller report is the Q1 caveat
    sec5 = text[text.index("## 5. Controller"):]
    assert sec5.split("\n\n")[2].startswith("Performance is conditional on the assumed thermal dynamics")


def test_target_rows_cover_every_target(tmp_path):
    data = rep.load_data("g26")
    if not data.get("predictions"):
        pytest.skip("predictions not computed")
    text = rep.render(data, preset="g26", stamp=STAMP, charts=False, chart_dir=tmp_path)
    sec = text[text.index("## 1. Targets"):text.index("## 2. Closure")]
    for role in tio.TARGET_ROLES:
        for t in tio.load_targets(role):
            assert f"| {t.id} | {t.status} |" in sec, t.id
    # unscored entries are shown with their status, never as a pass
    for tid in ("VAL-6", "VAL-7a", "VAL-8e"):
        row = next(ln for ln in sec.splitlines() if ln.startswith(f"| {tid} |"))
        assert "not scored" in row


def test_scoring_uses_the_frozen_tolerance(tmp_path):
    """A fabricated prediction exactly at the target passes; one far off fails (the report scores, not the cache)."""
    data = {"predictions": {"meta": {}, "records": [], "corners": {}}}
    recs = []
    for t in tio.load_targets("validation"):
        if t.id == "VAL-1":
            recs.append({"id": t.id, "replica_status": "ok", "predicted": t.value})
        if t.id == "VAL-3a":
            recs.append({"id": t.id, "replica_status": "ok", "predicted": t.value + 50.0})
    data["predictions"]["records"] = recs
    text = rep.render(data, preset="g26", stamp=STAMP, charts=False, chart_dir=tmp_path)
    row1 = next(ln for ln in text.splitlines() if ln.startswith("| VAL-1 |"))
    row3 = next(ln for ln in text.splitlines() if ln.startswith("| VAL-3a |"))
    assert "| pass |" in row1 and "| fail |" in row3


def test_staleness_is_detected():
    data = {"predictions": {"meta": {"config_hash": {"g26": "0" * 16}, "targets_sha256": rep._target_files_sha()}}}
    out = rep._staleness(data, {"G26": "g26", "G25": "g25_test_vehicle"})
    assert out and "g26 changed" in out[0]
    data = {"predictions": {"meta": {"config_hash": {}, "targets_sha256": {"validation": "x"}}}}
    assert any("validation.yaml changed" in s for s in rep._staleness(data, {"G26": "g26"}))


def test_companion_preset():
    assert rep.companion_g25("g26") == "g25_test_vehicle"
    assert rep.companion_g25("g26_h86") == "g25_test_vehicle"            # no g25_test_vehicle_h86 preset
    assert rep.companion_g25("g26_calibrated", "mine") == "mine"
    if (rep.PRESETS_DIR / "g25_test_vehicle_calibrated.yaml").exists():
        assert rep.companion_g25("g26_calibrated") == "g25_test_vehicle_calibrated"


def test_missing_preset_is_a_clean_error(capsys):
    assert rep.main(["--preset", "no_such_preset_xyz", "--fast", "--out", "/dev/null"]) == 1
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
    monkeypatch.setattr(rep, "REPO", tmp_path)
    found = rep._closure_tests()
    assert found == {"q": ["`tests/closure/test_conflicts.py::test_q_closes`"]}


def test_calibrated_leaf_audit_flags_contamination(tmp_path):
    """A leaf tagged calibrated against a held-out target is reported as contamination (07 P0-5)."""
    from mrm import params
    f = tmp_path / "g26_bad_cal.yaml"
    f.write_text("extends: g26\nmeta: {name: g26_bad_cal, kind: preset}\nring:\n  optics:\n    q_loaded: {value: 3900, "
                 "provenance: calibrated, generation: G25, source: LM-T, range: [2600, 4200], targets: [VAL-1, CAL-1], "
                 "residual: 0.1, identifiable: true}\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load(str(f))
    lines = rep._section_identifiability(cfg, {}, "g26_bad_cal", tmp_path, False)
    # the section 3.2 audit row (section 3.1 can hold a fit record's row of the same leaf)
    lines = lines[lines.index("### 3.2 Leaves tagged `calibrated` in this preset"):]
    row = next(ln for ln in lines if ln.startswith("| `ring.optics.q_loaded`"))
    assert "CONTAMINATION" in row and "VAL-1" in row


def test_fit_record_appears_only_in_its_presets_reports():
    """A fit record without a ``preset`` key belongs to g26 (the start) and g26_calibrated (the output) only."""
    if not (rep.REPO / "mrm" / "calibration" / "latest_fit.json").exists():
        pytest.skip("no fit record")
    assert rep._calibration_outputs("g26") and rep._calibration_outputs("g26_calibrated")
    assert not rep._calibration_outputs("g26_h86") and not rep._calibration_outputs("g26_bad_cal")
    assert rep._calibrated("g26_calibrated") and not rep._calibrated("g26") and not rep._calibrated("g26_h86")


def test_scales_fall_back_to_the_tolerance():
    t = {x.id: x for x in tio.load_calibration_targets()}
    np.testing.assert_allclose(rep._scales(t["CAL-1"], 3800.0), [175.0])           # interval/4
    np.testing.assert_allclose(rep._scales(t["CAL-2"], [1.0] * 6), [0.5] * 6)        # sigma
    s = rep._scales(t["CAL-9b"], {"at_code_maxima": 1.0, "at_code_minima": 2.0})
    np.testing.assert_allclose(s, [200.0, 400.0])                                      # per key


def test_jacobian_targets_are_calibration_only():
    ids = [t.id for t in rep._jacobian_targets()]
    assert ids and all(i.startswith("CAL-") for i in ids)
    assert "CAL-6" not in ids and not any(i.startswith(("CAL-12", "CAL-13")) for i in ids)


def test_cached_sensitivity_is_consistent():
    """The cached section-4 data: the base world is the nominal world, the Sobol entries are the screened top 8,
    and the indices are in [-0.2, 1.2] (estimator noise allowed, not hidden)."""
    S = rep.load_data("g26").get("sensitivity")
    if not S:
        pytest.skip("sensitivity not computed")
    assert len(S["top"]) == rep.SENS_TOP
    from mrm.validation import sensitivity as sens
    assert S["top"] == sens.screen_top(S["tornado"], rep.SENS_TOP)
    for m, e in S["sobol"].items():
        if "st" in e:
            assert e["names"] == S["top"]
            assert all(-0.2 <= x <= 1.2 for x in e["st"] if x is not None), m


@pytest.mark.slow
def test_injected_worlds_run_on_the_engine():
    """The nominal row of an injected design gives the metrics of the nominal world run through the experiment
    layer (07 4.8 protocol), so the tornado base is the nominal world (~1 min)."""
    from mrm.sim import experiment as ex
    from mrm.validation import sensitivity as sens
    space = sens.UncertaintySpace(rep._load_cfg("g26"))
    u0 = space.nominal_u()
    i = space.names.index("thermo_optic")
    u1 = u0.copy()
    u1[i] = space.axes["thermo_optic"].u_of_value("H86")
    W = rep.evaluate_worlds("g26", np.vstack([u0, u1]), n_threads=2)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "nom.yaml"
        f.write_text(rep._sens_experiment("g26", 1).replace("kind: mc", "kind: run").replace(
            "uncertainty: lhs", "uncertainty: nominal").replace("  n_mc: 1\n", "  batch: 1\n"))
        res = ex.run(str(f), write=False, log=lambda s: None, options={"n_threads": 2, "record": []})
    nom = ex.world_summary(res.cases["main"].metrics)
    for k in ("oma_penalty_db_worst", "mean_reloma_db", "excursion_count", "wrong_line_rings"):
        assert W[k][0] == pytest.approx(float(nom[k][0]), rel=1e-6, abs=1e-9), k
    # and the second row is a different world (H86)
    assert W["mean_reloma_db"][1] != W["mean_reloma_db"][0] or W["oma_penalty_db_worst"][1] != \
        W["oma_penalty_db_worst"][0]


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
    monkeypatch.setattr(rep, "REPO", tmp_path)
    assert rep._closure_tests() == {"rth": ["`tests/closure/test_mixed.py::test_rth_loose` (not strict)"]}


def test_repository_closure_links_are_real_xfail_tests():
    for slug, tests in rep._closure_tests().items():
        for t in tests:
            assert "tests/verification/test_report.py" not in t, (slug, t)


def test_fit_target_audit():
    assert rep._fit_target_audit(["CAL-1", "CAL-9a", "CAL-13a"]) == []
    out = rep._fit_target_audit(["CAL-1", "VAL-2a", "BND-1"])
    assert out and out[0].startswith("CONTAMINATION") and "VAL-2a" in out[0] and "BND-1" in out[0]
    assert any("fit: false" in x for x in rep._fit_target_audit(["CAL-6"]))
    assert any("unknown" in x for x in rep._fit_target_audit(["CAL-99"]))
    exc = [e.id for e in tio.load_excluded()]
    assert any("excluded" in x for x in rep._fit_target_audit(exc[:1]))


def _fake_fit(targets_of_group, decision_targets):
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
    id anywhere in a group (07 P0-5)."""
    c = _fake_fit(["CAL-1", "CAL-2"], ["CAL-1"])
    c["_file"] = "mrm/calibration/latest_fit.json"
    lines = rep._section_calibration_fit(c, tmp_path, True)
    text = "\n".join(lines)
    assert "Audit: ok" in text and "| `ring.optics.q_loaded` | device |" in text
    assert (tmp_path / "profiles_optics.svg").exists()
    bad = _fake_fit(["CAL-1"], ["CAL-1", "VAL-2a"])
    bad["_file"] = "x.json"
    assert "CONTAMINATION" in "\n".join(rep._section_calibration_fit(bad, tmp_path, False))


def test_jacobian_moves_only_plants_that_share_the_leaf(monkeypatch):
    """A leaf whose G25 value differs from G26's (optical_path.chip_input_loss_db: 0 dB bench against 6 dB) is
    moved on G26 only (the G26 range and step must not be clipped onto the G25 leaf); a shared leaf moves on both."""
    from types import SimpleNamespace
    from mrm.validation import replicas as rp

    def fake_run(t, cfg):
        q = float(cfg.leaves["ring.optics.q_loaded"].value)
        il = float(cfg.leaves["optical_path.chip_input_loss_db"].value)
        return SimpleNamespace(status="ok", value=q / 100.0 + 10.0 * il, notes=[], variants={}, conditions={})
    monkeypatch.setattr(rp, "run_target", fake_run)
    monkeypatch.setattr(rep, "_jacobian_targets",
                        lambda: [t for t in tio.load_calibration_targets() if t.id in ("CAL-7", "CAL-8")])
    I = rep.compute_identifiability({"G26": "g26", "G25": "g25_test_vehicle"},
                                    params=(("ring.optics.q_loaded", ""), ("optical_path.chip_input_loss_db", "")),
                                    log=lambda s: None)
    lab = I["residual_labels"]
    J = np.asarray(I["jacobian"], float) * np.asarray(I["scales"], float)[:, None]
    i7, i8 = lab.index("CAL-7"), lab.index("CAL-8")                 # CAL-7 on the G25 bench, CAL-8 on G26
    p_q, p_il = I["params"]
    assert p_q["moved_on"] == ["G26", "G25"] and "kept" not in p_q
    assert p_il["moved_on"] == ["G26"] and "G25" in p_il["kept"]
    lo, hi = load("g26").leaves["optical_path.chip_input_loss_db"].range         # [1, 10] dB since the params review
    assert J[i7, 1] == 0.0 and J[i8, 1] == pytest.approx(10.0 * (hi - lo) / 2.0)  # d/d(il) x the G26 half-range
    assert J[i7, 0] == pytest.approx(800.0 / 100.0) and J[i8, 0] == pytest.approx(8.0)
