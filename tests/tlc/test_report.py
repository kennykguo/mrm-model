"""``tools/tlc_report.py`` on a synthetic run directory: the ECE496 per-ring error statistics are checked against
values computed by hand from a known detuning trace."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("tlc_report", ROOT / "tools" / "tlc_report.py")
tlc_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tlc_report)

N = 2
REF_PM = np.array([100.0, 120.0])
LAM_M = 1310e-9


def _make_run(tmp_path: Path, with_telemetry: bool = True) -> Path:
    run = tmp_path / "20261101T000000Z_deadbeef"
    run.mkdir()
    (run / "experiment.resolved.yaml").write_text(yaml.safe_dump({
        "meta": {"name": "steady_tlc"}, "controller": {"name": "tlc", "params": {}},
        "scenario": {"name": "static"}, "experiment": {"score_from_s": 0.001, "duration_s": 0.003}}))
    t = np.arange(300) * 1e-5                                          # 3 ms at 10 us
    err = np.zeros((300, 1, N))
    err[:, 0, 0] = np.where(t < 0.001, 50.0, 2.0)                     # big error before score_from_s is ignored
    err[:, 0, 1] = np.where(np.arange(300) % 2 == 0, 3.0, -3.0)
    det_m = (err + REF_PM[None, None, :]) * 1e-12
    metrics = {"name": "steady_tlc", "cases": {"main": {"lock_reference_pm": [REF_PM.tolist()],
                                  "max_abs_lock_error_pm": [[2.0, 3.0]], "rms_lock_error_pm": [[2.0, 3.0]],
                                  "time_to_lock_s": [[0.0, 0.0]], "locked_final": [[True, True]],
                                  "wrong_line_final": [[False, False]], "oma_penalty_db": [[0.1, 0.2]],
                                  "peak_slew_codes_per_s": [[1e5, 2e5]], "excursion_count": [[0, 0]]}}}
    (run / "metrics.json").write_text(json.dumps(metrics))
    with h5py.File(run / "trace.h5", "w") as f:
        f["trace/main/detuning_m@10:mean/t_s"] = t
        f["trace/main/detuning_m@10:mean/value"] = det_m
        f["truth_final/main/lam_m"] = np.full((1, N), LAM_M)
    if with_telemetry:
        np.savez(run / "tlc_telemetry.npz", t_s=np.array([0.0, 1e-4]), state=np.full((2, 1, N), 2),
                 core_step_us=np.full((2, 1, N), 40.0), dither_amp_pm=np.zeros((2, 1, N)))
    return run


def test_error_statistics_skip_the_unscored_start(tmp_path):
    summary = tlc_report.summarize_run(_make_run(tmp_path))
    r0, r1 = summary["rings"]
    assert r0["peak_err_pm"] == pytest.approx(2.0)
    assert r0["rms_err_pm"] == pytest.approx(2.0)
    assert r1["peak_err_pm"] == pytest.approx(3.0)
    assert r1["p999_err_pm"] == pytest.approx(3.0)


def test_ghz_columns_use_lambda_squared_over_c(tmp_path):
    summary = tlc_report.summarize_run(_make_run(tmp_path))
    pm_per_ghz = LAM_M ** 2 / 299_792_458.0 * 1e21
    assert summary["rings"][1]["peak_err_ghz"] == pytest.approx(3.0 / pm_per_ghz)


def test_threshold_flags(tmp_path):
    summary = tlc_report.summarize_run(_make_run(tmp_path))
    assert summary["rings"][0]["within_5pm"] is True
    assert summary["rings"][1]["within_2ghz"] is True


def test_telemetry_summary_when_present_and_absent(tmp_path):
    s = tlc_report.summarize_run(_make_run(tmp_path))
    assert s["telemetry"]["core_steps"] == 2
    assert s["telemetry"]["core_step_us_mean"] == pytest.approx(40.0)
    assert s["telemetry"]["track_fraction"] == pytest.approx(1.0)
    other = tmp_path / "b"
    other.mkdir()
    assert tlc_report.summarize_run(_make_run(other, with_telemetry=False))["telemetry"] is None


def test_markdown_and_json_are_written(tmp_path):
    run = _make_run(tmp_path)
    out = tmp_path / "results.md"
    tlc_report.main([str(run), "--out", str(out)])
    text = out.read_text()
    assert "steady_tlc" in text and "| ring |" in text
    assert json.loads(out.with_suffix(".json").read_text())["runs"][0]["controller"] == "tlc"


def test_ece496_reference_rows_convert_ghz_to_pm_at_1310nm(tmp_path):
    pub = {"steady": {"tlc": {"R1": {"p999": [0.5, 0.4], "peak": [1.0, 0.6], "rms": [0.2, 0.1]}}},
           "d2_common": {"tlc": {"R3": {"p999": [1.8, 2.2], "peak": [2.0, 2.5], "rms": [0.3, 0.3]}}}}
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(pub))
    rows = tlc_report.ece496_reference(path)
    assert len(rows) == 4
    r = next(r for r in rows if r["scenario"] == "d2_common" and r["ring"] == 1)
    assert r["peak_err_ghz"] == 2.5
    assert r["peak_err_pm"] == pytest.approx(2.5 * (1310e-9) ** 2 / 299_792_458.0 * 1e21)


def test_report_includes_the_reference_section_when_given(tmp_path):
    pub = {"steady": {"tlc": {"R1": {"p999": [0.5], "peak": [1.0], "rms": [0.2]}}}}
    ref = tmp_path / "metrics.json"
    ref.write_text(json.dumps(pub))
    out = tmp_path / "results.md"
    tlc_report.main([str(_make_run(tmp_path)), "--out", str(out), "--ece496-metrics", str(ref)])
    assert "ECE496 4-ring reference" in out.read_text()
