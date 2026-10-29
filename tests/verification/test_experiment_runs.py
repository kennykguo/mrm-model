"""Runs every experiment kind of mrm.sim.experiment on the engine, short and small.

Read this first
---------------
The tests check the run directory contents, that ``rerun`` reproduces a run bit for bit (also without the original
files and without numba), the scoring start, the self reference, the gate on compressed scenario time (a
convergence certificate), compare, sweep, Monte Carlo chunking and the steady-state survey. ~40 s with a warm numba
cache.
"""
from __future__ import annotations

import json
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from mrm.sim import experiment
from mrm.sim import manifest

QUIET = {"log": lambda s: None}   # keyword arguments that silence experiment.run / rerun


def _write(tmp_path: Path, name: str, text: str) -> Path:
    """Write the dedented text to tmp_path / name and return the path."""
    p = tmp_path / name
    p.write_text(textwrap.dedent(text))
    return p


# a 4 ms warm-started run of pi_lock_c on two sampled g26 worlds, scored from 1 ms
RUN_YAML = """\
    extends: g26
    meta: {name: t_run, kind: experiment}
    controller: {name: pi_lock_c}
    scenario: {name: static}
    experiment:
      kind: run
      duration_s: 0.004
      batch: 2
      seeds: {device: 3, uncertainty: 4, noise: 5, scenario: 6}
      uncertainty: sample
      variation: sample
      start: warm
      score_from_s: 0.001
    """


def test_run_writes_a_complete_run_directory_and_reruns_bit_for_bit(tmp_path):
    p = _write(tmp_path, "run.yaml", RUN_YAML)
    res = experiment.run(p, runs_dir=tmp_path / "runs", argv=["mrm", "run", str(p)], **QUIET)
    d = res.run_dir
    assert d.name.endswith("_" + res.experiment_hash[:8]) and d.parent == tmp_path / "runs"
    names = {x.name for x in d.iterdir()}
    assert {"experiment.resolved.yaml", "manifest.json", "metrics.json", "trace.h5", "report.txt", "inputs"} <= names
    # resolved yaml: every leaf with its provenance, the resolved options and the source
    tree = yaml.safe_load((d / "experiment.resolved.yaml").read_text())
    q = tree["ring"]["optics"]["q_loaded"]
    assert set(q) >= {"value", "provenance"}
    assert tree["experiment"]["start"] == "warm" and tree["experiment"]["histogram"] is False
    assert str(p) in tree["_source"]["files"]
    man = json.loads((d / "manifest.json").read_text())
    assert man["schema"] == manifest.MANIFEST_SCHEMA and man["kind"] == "run"
    assert man["experiment_hash"] == res.experiment_hash
    assert man["seeds"] == {"device": 3, "uncertainty": 4, "noise": 5, "scenario": 6}
    assert man["argv"] == ["mrm", "run", str(p)]
    for k in ("numpy", "scipy", "numba", "llvmlite"):
        assert man["environment"][k]
    assert "commit" in man["git"] and "diff_sha256" in man["git"]
    assert any(a["file"] == "extracted/modes_rib160.json" and len(a["sha256"]) == 64 for a in man["artefacts"])
    assert man["source"]["inputs"][0]["file"] == "inputs/run.yaml"
    assert man["cases"]["main"]["manifest_inputs"]["score_from_tick"] == 1000
    met = json.loads((d / "metrics.json").read_text())
    assert np.asarray(met["cases"]["main"]["max_abs_detuning_pm"]).shape == (2, 8)
    tr = manifest.read_trace_h5(d / "trace.h5")
    assert tr["trace"]["main"]["detuning_m@10:min"]["value"].shape == (400, 2, 8)       # default records
    assert tr["truth_final"]["main"]["detuning_m"].shape == (2, 8)
    assert res.report.splitlines()[0] == experiment.Q1_CAVEAT
    # rerun from the directory: a new directory, every array of trace.h5 identical
    res2, cmp = experiment.rerun(d, runs_dir=tmp_path / "runs", **QUIET)
    assert res2.run_dir != d and cmp["identical"], cmp["differing"][:5]
    assert cmp["n_compared"] >= 90
    man2 = json.loads((res2.run_dir / "manifest.json").read_text())
    assert man2["rerun_of"] == d.name
    # a changed input is caught before running
    txt = (d / "inputs" / "run.yaml").read_text().replace("duration_s: 0.004", "duration_s: 0.005")
    (d / "inputs" / "run.yaml").write_text(txt)
    with pytest.raises(experiment.ExperimentError, match="experiment hash"):
        experiment.rerun(d, runs_dir=tmp_path / "runs", **QUIET)


def test_mrm_run_on_a_loaded_config(tmp_path):
    import mrm
    from mrm import params
    p = _write(tmp_path, "plain.yaml", """\
        extends: g26
        meta: {name: plain, kind: experiment}
        experiment: {kind: run, duration_s: 0.001, uncertainty: nominal, variation: "off"}
        """)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load(p)
    res = mrm.run(cfg, runs_dir=tmp_path / "runs", **QUIET)
    assert res.status == "ok" and res.metrics["max_abs_detuning_pm"].shape == (1, 8)
    assert res.report.splitlines()[0] != experiment.Q1_CAVEAT          # open loop: not a controller report
    res2, cmp = experiment.rerun(res.run_dir, runs_dir=tmp_path / "runs", **QUIET)
    assert cmp["identical"]


def test_score_from_leaves_the_plant_and_records_untouched(tmp_path):
    """runner option score_from_s: identical traces and final Truth, metrics restarted at the scoring tick (also on
    the Python controller path, whose loop is split there; a split on a controller edge moves one tick later)."""
    from mrm.sim import runner
    from mrm.sim.metrics import RecordSpec
    rec = [RecordSpec("detuning_m", 10, "mean"), RecordSpec("heater_code", 10, "last")]
    for ctrl, python in (("pi_lock_c", False), ("pi_lock", True)):
        out = []
        for sf in (None, 2.005e-3, 2.01e-3):                 # 2010 is a controller edge (period 10, phase 0)
            kw = {} if sf is None else {"score_from_s": sf}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out.append(runner.run("g26", ctrl, "static", duration_s=4e-3, uncertainty="nominal",
                                      variation="off", heater_code="locked_steady_state", settle=True, record=rec,
                                      **kw))
        a, b, c = out
        for k in a.trace:
            assert a.trace[k]["value"].tobytes() == b.trace[k]["value"].tobytes() == c.trace[k]["value"].tobytes()
        for f in ("detuning_m", "t_ring_k", "heater_code", "monitor_code"):
            assert a.truth_final.fields[f].tobytes() == b.truth_final.fields[f].tobytes() \
                == c.truth_final.fields[f].tobytes()
        assert b.manifest_inputs["score_from_tick"] == 2005
        assert c.manifest_inputs["score_from_tick"] == (2011 if python else 2010)
        for r in (b, c):
            k0 = r.manifest_inputs["score_from_tick"]
            assert np.allclose(r.metrics["duration_s"], (4000 - k0) * 1e-6, rtol=0, atol=1e-15)
        assert np.allclose(a.metrics["duration_s"], 4e-3)
        # the scored window leaves out the peak map: the detuning extremes are smaller
        assert np.all(b.metrics["max_abs_detuning_pm"] <= a.metrics["max_abs_detuning_pm"])
        assert np.all(b.metrics["max_abs_detuning_pm"] < 200.0)


def test_scored_metrics_equal_the_scored_ticks_of_a_full_trace_and_keep_the_oracle_reference():
    """runner score_from_s: the restarted accumulators equal the same metrics taken over the scored ticks of a
    per-tick record of the run; with lock_reference 'oracle' the oracle's current operating point is kept across the
    split. (MetricsState.reset sets the reference to 0, the line, until the next oracle evaluation up to
    oracle_period_ticks = 1000 ticks later. Without the kept reference, the scored lock error of pi_lock_c would be
    ~125 pm, the distance to its line, larger than that of the unsplit run.)"""
    from mrm.sim import runner
    from mrm.sim.metrics import RecordSpec
    rec = [RecordSpec("detuning_m", 1, "mean")]
    kw = dict(duration_s=6e-3, uncertainty="nominal", variation="off", heater_code="locked_steady_state", settle=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tgt = {"lock_reference": "target", "lock_target_pm": 100.0}
        full = runner.run("g26", "pi_lock_c", "static", record=rec, metrics=tgt, **kw)
        part = runner.run("g26", "pi_lock_c", "static", record=rec, metrics=tgt, score_from_s=4e-3, **kw)
        fo = runner.run("g26", "pi_lock_c", "static", metrics={"lock_reference": "oracle"}, **kw)
        so = runner.run("g26", "pi_lock_c", "static", metrics={"lock_reference": "oracle"}, score_from_s=4e-3, **kw)
    d = full.trace[rec[0].key]["value"][4000:]                   # (ticks, B, N) m: the scored ticks
    m = part.metrics
    assert np.array_equal(m["max_abs_detuning_pm"], np.abs(d).max(axis=0) * 1e12)
    assert np.array_equal(m["max_abs_lock_error_pm"], np.abs(d - 100.0 * 1e-12).max(axis=0) * 1e12)
    assert np.allclose(m["mean_detuning_pm"], d.mean(axis=0) * 1e12, rtol=1e-12, atol=0)
    assert np.array_equal(m["time_above_s"]["5pm"], (np.abs(d - 1e-10) > 5e-12).sum(axis=0) * 1e-6)
    # oracle reference: the scored window is a subset of the run, so its largest lock error cannot be larger
    assert np.all(so.metrics["max_abs_lock_error_pm"] <= fo.metrics["max_abs_lock_error_pm"])
    assert np.all(so.metrics["max_abs_lock_error_pm"] < 60.0)          # ~20 pm: pi_lock's distance to the oracle
    assert np.array_equal(so.metrics["lock_reference_pm"], fo.metrics["lock_reference_pm"])


def test_scored_peak_slew_does_not_depend_on_where_the_split_falls():
    """The previous controller latch is plant history: a split that falls between two latches must not shorten the
    first latch interval it sees (otherwise pi_lock_c under A1 would score 6e5 to 6e6 codes/s, depending on the
    split's offset in the 10-tick controller period)."""
    from mrm.sim import runner
    kw = dict(duration_s=8e-3, uncertainty="nominal", variation="off", heater_code="locked_steady_state", settle=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slews = [runner.run("g26", "pi_lock_c", "a1", score_from_s=k * 1e-6, **kw).metrics["peak_slew_codes_per_s"]
                 for k in (5000, 5005, 5009)]
    assert slews[0].tobytes() == slews[1].tobytes() == slews[2].tobytes()


def test_self_reference_judges_the_lock_about_its_own_point(tmp_path):
    p = _write(tmp_path, "sr.yaml", RUN_YAML.replace("batch: 2", "batch: 1").replace(
        "start: warm", "start: warm\n      self_reference: {duration_s: 0.006, window_s: 0.002}\n"
        "      acceptance: [{id: lock, metric: max_abs_lock_error_pm, op: '<=', value: 5.0}]").replace(
        "uncertainty: sample", "uncertainty: nominal").replace("variation: sample", "variation: \"off\"").replace(
        "duration_s: 0.004", "duration_s: 0.0083").replace(
        "score_from_s: 0.001", "score_from_s: 0.004\n      record: ['i_monitor_a@1:mean']"))
    res = experiment.run(p, write=False, **QUIET)            # scored after pi_lock's ~3 ms peak map
    m = res.metrics
    case = res.cases["main"]
    ref = case.reference["d_ref_pm"]
    assert ref.shape == (1, 8) and np.all(ref > 50) and np.all(ref < 300)      # side of fringe, red of the line
    assert case.reference["n_windows"] == 10                 # the windows inside the last 2 ms of the pre-run
    assert np.allclose(m["lock_reference_pm"], ref)
    assert np.all(m["max_abs_lock_error_pm"] < 1.0) and res.passed is True
    assert {"monitor_error_pct_min", "monitor_error_pct_max", "monitor_dev_ua_max"} <= set(m)
    assert np.all(np.abs(m["monitor_error_pct_max"]) < 1.0)
    # the monitor metrics cover exactly the scored ticks 4000..8299 (8300 ticks is no multiple of the 4000-tick
    # scoring start: the envelope windows are 100 ticks, so the run's last 300 ticks are judged too)
    i = case.trace["i_monitor_a@1:mean"]["value"][4000:]
    iref = case.reference["i_monitor_ref_a"]
    assert np.array_equal(m["monitor_error_pct_min"], (i.min(axis=0) - iref) / iref * 100.0)
    assert np.array_equal(m["monitor_error_pct_max"], (i.max(axis=0) - iref) / iref * 100.0)
    assert case.trace["i_monitor_a@100:min"]["t_s"][-1] == pytest.approx(0.0083, abs=1e-12)


def test_self_reference_refuses_a_driven_base_temperature(tmp_path):
    p = _write(tmp_path, "sr.yaml", RUN_YAML.replace("scenario: {name: static}", "scenario: {name: soak}").replace(
        "start: warm", "start: warm\n      self_reference: {duration_s: 0.002, window_s: 0.001}"))
    with pytest.raises(experiment.ExperimentError, match="drives the base temperature"):
        experiment.run(p, write=False, **QUIET)


# a 0.2 s soak (two 0.5 K cycles) at time compression 2, with the convergence cases at compression 1 and 2
MINI_SOAK_YAML = """\
    extends: g26
    meta: {name: t_soak, kind: experiment}
    controller: {name: pi_lock_c}
    scenario: {name: soak, params: {t_low_c: 30.0, t_high_c: 30.5, cycle_s: 0.2, n_cycles: 2}, time_compression: 2.0}
    experiment:
      kind: run
      duration_s: 0.2
      uncertainty: nominal
      variation: "off"
      start: warm
      score_from_s: 0.004
      compression_factors: [1.0, 2.0]
      record: []
    """


def test_compressed_run_needs_and_reuses_a_convergence_certificate(tmp_path):
    p = _write(tmp_path, "soak.yaml", MINI_SOAK_YAML)
    res = experiment.run(p, runs_dir=tmp_path / "runs", **QUIET)
    assert res.status == "ok"
    assert [k for k in res.cases] == ["convergence_c1", "convergence_c2", "main"]
    assert res.cases["convergence_c1"].manifest_inputs["n_ticks"] == 200_000          # one 0.2 s cycle
    assert res.cases["convergence_c2"].manifest_inputs["n_ticks"] == 100_000
    conv = json.loads((res.run_dir / "convergence.json").read_text())
    assert conv["passed"] and conv["max_validated_factor"] == 2.0 and conv["certificate"]
    # the whole compressed run (convergence cases included) reruns bit for bit
    res_r, cmp = experiment.rerun(res.run_dir, runs_dir=tmp_path / "runs", **QUIET)
    assert cmp["identical"], cmp["differing"][:5]
    assert "convergence_c1" in manifest.read_trace_h5(res_r.run_dir / "trace.h5")["metrics"]
    # the certificate of that run validates the same plant, controller and scenario in another experiment
    res2 = experiment.run(p, overrides=[f"experiment.convergence_certificate={res.run_dir}",
                                        "experiment.duration_s=0.05"], write=False, **QUIET)
    assert res2.status == "ok" and list(res2.cases) == ["main"]
    # ... but not another plant, nor other worlds
    for ov in ("ring.optics.q_loaded=4000", "experiment.seeds.noise=9"):
        res3 = experiment.run(p, overrides=[f"experiment.convergence_certificate={res.run_dir}", ov],
                              runs_dir=tmp_path / "runs", **QUIET)
        assert res3.status == "refused" and "different plant" in res3.report
        assert not res3.cases
    # the load-time rule of time compression (compressed ramp <= 1 % of the fast transient) holds before anything
    # runs
    from mrm.scenarios import CompressionError
    with pytest.raises(CompressionError, match="time_compression rule"):
        experiment.run(p, overrides=["scenario.time_compression=5"], write=False, **QUIET)


def test_compare_reports_corners_quantiles_and_rank_stability(tmp_path):
    p = _write(tmp_path, "cmp.yaml", """\
        extends: g26
        meta: {name: t_cmp, kind: experiment}
        controller_interface: {knowledge: [line_grid, ring_to_line, model_nominal]}
        scenario: {name: static}
        experiment:
          kind: compare
          duration_s: 0.004
          n_mc: 3
          uncertainty: lhs
          variation: "off"
          seeds: {device: 1, uncertainty: 2, noise: 3, scenario: 4}
          start: warm
          score_from_s: 0.001
          corners: [g26_h86, g26_slow_thermal]
          controllers: [pi_lock_c, kalman_lqg_c]
        """)
    res = experiment.run(p, runs_dir=tmp_path / "runs", **QUIET)
    lines = res.report.splitlines()
    assert lines[0] == experiment.Q1_CAVEAT
    s = res.summary
    assert set(s["controllers"]) == {"pi_lock_c", "kalman_lqg_c"}
    for d in s["controllers"].values():
        assert set(d) >= {"nominal", "corners", "worst_corner", "worst_corner_overall", "quantiles"}
        assert d["worst_corner_overall"]["corner"] in ("g26_h86", "g26_slow_thermal")
        assert d["quantiles"]["oma_penalty_db_worst"]["n"] == 3
    rs = s["rank_stability"]["oma_penalty_db_worst"]
    assert rs["n_worlds"] == 5 and set(rs["nominal_order"]) == {"pi_lock_c", "kalman_lqg_c"}
    assert "Ranking stability" in res.report
    # metrics only for the worlds: no traces recorded
    tr = manifest.read_trace_h5(res.run_dir / "trace.h5")
    assert "pi_lock_c|worlds" in tr["metrics"] and not tr.get("trace")
    res2, cmp = experiment.rerun(res.run_dir, runs_dir=tmp_path / "runs", **QUIET)
    assert cmp["identical"]


def test_sweep_points_and_criteria(tmp_path):
    p = _write(tmp_path, "sw.yaml", """\
        extends: g26
        meta: {name: t_sweep, kind: experiment}
        controller: {name: pi_lock_c}
        scenario: {name: static}
        experiment:
          kind: sweep
          duration_s: 0.008
          uncertainty: nominal
          variation: "off"
          start: warm
          score_from_s: 0.005
          grid:
            controller.params.fraction: [0.6, 0.8]
            "alt:uncertainty.thermo_optic": [H_solver, H86]
          acceptance: [{id: line, metric: wrong_line_rings, op: "==", value: 0}]
        """)
    res = experiment.run(p, write=False, **QUIET)
    pts = res.summary["points"]
    assert len(pts) == 4 and pts[1]["assignment"] == {"controller.params.fraction": 0.6,
                                                      "alt:uncertainty.thermo_optic": "H86"}
    h = {c.manifest_inputs["case_config_hash"] for c in res.cases.values()}
    assert len(h) == 4
    assert res.passed is True
    # the grid's controller parameter reaches the controller: a set-point at 0.8 of the peak sits closer to the line
    c06, c08 = res.cases["point_000"], res.cases["point_002"]
    assert c06.manifest_inputs["controller_entry"]["params"]["fraction"] == 0.6
    assert c08.manifest_inputs["controller_entry"]["params"]["fraction"] == 0.8
    assert np.all(c08.metrics["mean_detuning_pm"] < c06.metrics["mean_detuning_pm"] - 5.0)


def test_mc_chunks_reproduce_one_plant(tmp_path):
    """Sample-mode Monte Carlo in chunks with advanced seed bases gives the worlds of one B = n plant (the batch rule
    of the random streams)."""
    p = _write(tmp_path, "mc.yaml", """\
        extends: g26
        meta: {name: t_mc, kind: experiment}
        experiment:
          kind: mc
          duration_s: 0.002
          n_mc: 3
          uncertainty: sample
          variation: sample
          seeds: {device: 11, uncertainty: 12, noise: 13, scenario: 14}
        """)
    one = experiment.run(p, write=False, **QUIET)
    chunked = experiment.run(p, overrides=["experiment.max_worlds_per_plant=2"], write=False, **QUIET)
    assert list(one.cases) == ["worlds"] and list(chunked.cases) == ["worlds_00", "worlds_01"]
    a = one.cases["worlds"].metrics["max_abs_detuning_pm"]
    b = np.concatenate([chunked.cases["worlds_00"].metrics["max_abs_detuning_pm"],
                        chunked.cases["worlds_01"].metrics["max_abs_detuning_pm"]])
    assert a.tobytes() == b.tobytes()
    q = one.summary["quantiles"]["max_abs_lock_error_pm"]
    assert q["n"] == 3


def test_locked_steady_state_survey_with_a_corner(tmp_path):
    p = _write(tmp_path, "qs.yaml", """\
        extends: g26
        meta: {name: t_qs, kind: experiment}
        experiment:
          kind: locked_steady_state
          uncertainty: nominal
          variation: "off"
          grid: {ambient_c: [0.0, 105.0], bias_v: [1.0, 3.75]}
          corners: [g26_heater_loss]
          acceptance: [{id: cov, metric: code_range_coverage, op: ">=", value: 1.0, cases: [locked_steady_state]}]
        """)
    res = experiment.run(p, runs_dir=tmp_path / "runs", **QUIET)
    s = res.summary
    assert s["code_range_coverage"] == [1.0] and s["fsr_hop_risk_points"] == 0
    assert set(s["cases"]) == {"locked_steady_state", "locked_steady_state/corner/g26_heater_loss"}
    # only a heater-temperature loss mechanism moves the oracle relOMA over ambient (the BND-4 bound)
    assert s["cases"]["locked_steady_state"]["bnd4_max_db"] < 1e-6
    assert s["cases"]["locked_steady_state/corner/g26_heater_loss"]["bnd4_max_db"] > 1e-3
    assert res.passed is True
    tr = manifest.read_trace_h5(res.run_dir / "trace.h5")
    assert tr["locked_steady_state"]["locked_steady_state"]["heater_code"].shape == (2, 2, 1, 8)
    # the same trace.h5 layout without corners (/locked_steady_state/<case>/<array>)
    res1 = experiment.run(p, overrides=["experiment.corners=[]"], runs_dir=tmp_path / "runs", **QUIET)
    tr1 = manifest.read_trace_h5(res1.run_dir / "trace.h5")
    assert set(k for k in tr1["locked_steady_state"] if not k.startswith("@")) == {"locked_steady_state"}
    assert tr1["locked_steady_state"]["locked_steady_state"]["heater_code"].tobytes() == \
        tr["locked_steady_state"]["locked_steady_state"]["heater_code"].tobytes()


def test_open_loop_acceptance_is_not_a_controller_report(tmp_path):
    p = _write(tmp_path, "ol.yaml", """\
        extends: g26
        meta: {name: t_ol, kind: experiment}
        experiment:
          kind: acceptance
          duration_s: 0.0005
          record: []
          acceptance: [{id: line, metric: wrong_line_rings, op: "==", value: 0}]
        """)
    res = experiment.run(p, write=False, **QUIET)
    assert res.report.splitlines()[0] == "RESULT: PASS" and experiment.Q1_CAVEAT not in res.report


def test_a_run_directory_reruns_without_the_original_files(tmp_path):
    """inputs/ carries the experiment file and every non-preset parent: a rerun after the original parent
    changed (or was sent elsewhere) reproduces the run bit for bit from the copies."""
    lib = tmp_path / "lib"
    lib.mkdir()
    parent = _write(lib, "parent.yaml", RUN_YAML.replace("meta: {name: t_run, kind: experiment}",
                                                         "meta: {name: t_parent, kind: experiment}"))
    child = _write(tmp_path, "child.yaml", """\
        extends: lib/parent.yaml
        meta: {name: t_child, kind: experiment}
        experiment: {duration_s: 0.003, batch: 1, record: []}
        """)
    res = experiment.run(child, runs_dir=tmp_path / "runs", **QUIET)
    names = sorted(x.name for x in (res.run_dir / "inputs").iterdir())
    assert names == ["_extends_1_parent.yaml", "child.yaml"]
    man = json.loads((res.run_dir / "manifest.json").read_text())
    assert [i["file"] for i in man["source"]["inputs"]] == ["inputs/child.yaml", "inputs/_extends_1_parent.yaml"]
    parent.write_text(parent.read_text().replace("pi_lock_c", "mpc_c"))    # the original changes afterwards
    res2, cmp = experiment.rerun(res.run_dir, runs_dir=tmp_path / "runs", **QUIET)
    assert cmp["identical"], cmp["differing"][:5]


def test_experiment_without_jit_is_bit_identical(tmp_path):
    """The experiment layer is plain Python over the engine: a short warm-start run under NUMBA_DISABLE_JIT=1
    (the NumPy-executed kernel) writes the same trace.h5 arrays, bit for bit, as the compiled run (~5 s)."""
    import os
    import subprocess
    import sys
    p = _write(tmp_path, "nojit.yaml", """\
        extends: g26
        meta: {name: t_nojit, kind: experiment}
        controller: {name: pi_lock_c}
        scenario: {name: static}
        metrics: {lock_reference: oracle}
        experiment:
          kind: run
          duration_s: 0.0003
          uncertainty: nominal
          variation: "off"
          start: warm
          score_from_s: 0.0001
        """)
    root = Path(__file__).resolve().parents[2]
    for jit in ("1", "0"):
        env = dict(os.environ, NUMBA_DISABLE_JIT=jit)
        r = subprocess.run([sys.executable, "-m", "mrm.cli", "run", str(p), "--runs-dir", str(tmp_path / "runs"),
                            "--quiet"], cwd=root, env=env, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
    a, b = sorted((tmp_path / "runs").iterdir())
    cmp = manifest.compare_runs(a, b)
    assert cmp["identical"], cmp["differing"][:5]
    assert json.loads((a / "manifest.json").read_text())["environment"]["numba_disable_jit"] == "1"


def test_manifest_records_the_code_that_ran(tmp_path, monkeypatch):
    """The git state and environment are captured when the run starts: an edit of the tree during a long run must
    not change what the manifest records."""
    calls = iter([{"available": True, "commit": "at-start", "diff_sha256": "a"},
                  {"available": True, "commit": "at-write", "diff_sha256": "b"}])
    monkeypatch.setattr(manifest, "git_state", lambda root=None: next(calls))
    p = _write(tmp_path, "g.yaml", """\
        extends: g26
        meta: {name: t_git, kind: experiment}
        experiment: {kind: run, duration_s: 0.001, uncertainty: nominal, variation: "off", record: []}
        """)
    res = experiment.run(p, runs_dir=tmp_path / "runs", **QUIET)
    assert json.loads((res.run_dir / "manifest.json").read_text())["git"]["commit"] == "at-start"
