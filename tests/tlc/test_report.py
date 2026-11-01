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
        np.savez(run / "tlc_telemetry.npz", t_s=np.array([0.0, 1e-4, 2e-4]),
                 state=np.array([[[0] * N], [[2] * N], [[2] * N]]),                     # row 0: start state
                 core_step_us=np.array([[[0.0] * N], [[40.0] * N], [[40.0] * N]]), dither_amp_pm=np.zeros((3, 1, N)))
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
    assert s["telemetry"]["core_steps"] == 2                             # the start row is not a core step
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


def test_overview_row_per_run(tmp_path):
    rows = tlc_report.overview_rows([tlc_report.summarize_run(_make_run(tmp_path))])
    assert rows[0]["peak_err_pm_max"] == pytest.approx(3.0)
    assert rows[0]["peak_err_pm_max_rings_1_up"] == pytest.approx(3.0)
    assert rows[0]["rings_within_5pm"] == 2 and rows[0]["unlocked_final"] == 0


def test_remove_tones_strips_the_dither_and_keeps_the_trend():
    t = np.arange(4000) * 1e-5                                         # 40 ms at 10 us
    trend = 5.0 + 100.0 * t
    dither = 2.0 * np.sin(2 * np.pi * 4000 * t + 0.3) + 0.5 * np.cos(2 * np.pi * 8000 * t)
    out = tlc_report.remove_tones(t, (trend + dither)[:, None], (4000.0,))
    np.testing.assert_allclose(out[:, 0], trend, atol=1e-6)


def test_tracking_error_only_for_tlc_runs(tmp_path):
    s = tlc_report.summarize_run(_make_run(tmp_path))
    assert s["rings"][1]["track_peak_err_pm"] == pytest.approx(3.0, abs=0.05)   # a +-3 alternation is not a tone
    other = tmp_path / "pi"
    other.mkdir()
    run = _make_run(other)
    cfg = yaml.safe_load((run / "experiment.resolved.yaml").read_text())
    cfg["controller"]["name"] = "pi_lock_c"
    (run / "experiment.resolved.yaml").write_text(yaml.safe_dump(cfg))
    assert "track_peak_err_pm" not in tlc_report.summarize_run(run)["rings"][0]


def test_remove_tones_with_the_eight_ring_plan_and_a_short_final_block():
    from mrm.control.tlc.tone_plan import EIGHT_RING_BINS, to_hz
    tones = to_hz(EIGHT_RING_BINS)
    t = np.arange(4100) * 1e-5                                         # 41 ms: a 1 ms remainder
    rng = np.random.default_rng(0)
    err = rng.normal(0, 1.0, t.size)
    dith = sum(0.5 * np.sin(2 * np.pi * f * t + i) for i, f in enumerate(tones))
    out = tlc_report.remove_tones(t, (err + dith)[:, None], tones)[:, 0]
    # the dither (std 1.0) is gone; what is left is the noise's own projection on the 34 regressors per 4 ms
    # block, std ~ sqrt(32 / 400) = 0.28
    assert np.std(dith) > 0.9 and np.std(out - err) < 0.35
    assert np.std(out[-100:]) > 0.7                                    # the remainder is not overfitted away


def test_clopper_pearson_known_values():
    lo, hi = tlc_report.clopper_pearson(0, 40)
    assert lo == 0.0 and hi == pytest.approx(1 - 0.025 ** (1 / 40), rel=1e-9)     # 0.0881: the "3/n" rule is 0.075
    lo, hi = tlc_report.clopper_pearson(40, 40)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1 / 40), rel=1e-9)
    lo, hi = tlc_report.clopper_pearson(5, 10)
    assert lo == pytest.approx(0.187086, abs=1e-5) and hi == pytest.approx(0.812914, abs=1e-5)


def test_campaign_rates_count_ring_trials_and_worlds():
    m = {"locked_final": [[True, True], [True, False], [False, False]],
         "wrong_line_final": [[False, False], [False, True], [False, False]]}
    c = tlc_report.campaign_rates(m)
    assert c["worlds"] == 3
    assert c["unlocked"]["ring_trials"][:2] == [3, 6] and c["unlocked"]["worlds_any_ring"][:2] == [2, 3]
    assert c["wrong_line"]["ring_trials"][:2] == [1, 6]
    assert tlc_report.campaign_rates({"locked_final": [[True, False]]}) is None       # one world: no campaign
