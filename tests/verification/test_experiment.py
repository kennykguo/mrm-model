"""Checks experiment files (mrm.sim.experiment): loading, the extension schema, criteria, summaries and ranking.

Read this first
---------------
An experiment file is a config file with an ``experiment:`` section: the kind (run, sweep, compare, mc,
locked_steady_state, ...), the run options, the recorded channels and the acceptance criteria. These tests check
how a file loads and validates, how its hash and input copies make a run reproducible from its directory alone, how
criteria judge metrics (NaN never passes), how worlds are summarized and controllers ranked, and how a file plans
its cases and estimates its cost. No engine runs here: test_experiment_runs.py runs the kinds.
"""
from __future__ import annotations

import math
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

from mrm.sim import experiment
from mrm.sim.metrics import RecordSpec

ROOT = Path(__file__).resolve().parents[2]


def _write(tmp_path: Path, name: str, text: str) -> Path:
    """Write the dedented text to tmp_path / name and return the path."""
    p = tmp_path / name
    p.write_text(textwrap.dedent(text))
    return p


# a short warm-started run of pi_lock_c on g26 with two recorded channels
BASE_YAML = """\
    extends: g26
    meta: {name: base_exp, kind: experiment}
    controller: {name: pi_lock_c, params: {fraction: 0.7}}
    scenario: {name: static}
    experiment:
      kind: run
      duration_s: 0.003
      uncertainty: nominal
      variation: "off"
      start: warm
      score_from_s: 0.001
      record: ["detuning_m@10:envelope", "heater_code@auto:last"]
    """


def test_load_strips_and_validates_extension_keys(tmp_path):
    exp = experiment.load(_write(tmp_path, "e.yaml", BASE_YAML))
    assert exp.kind == "run" and exp.name == "base_exp"
    assert exp.options["start"] == "warm" and exp.options["score_from_s"] == 0.001
    assert exp.option("histogram") is False and exp.option("compression_factors") == (1.0, 10.0, 100.0)
    assert exp.get("controller.name") == "pi_lock_c"
    assert str(tmp_path / "e.yaml") in exp.cfg.files            # the original, not the stripped temp copy
    # mrm.params knows the extension keys: the file loads directly, and mrm.run(mrm.params.load(file))
    # (from_config) sees the same options and has the same experiment hash
    from mrm import params
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load(tmp_path / "e.yaml")
    assert cfg.get("experiment.start") == "warm" and cfg.get("experiment.record")[0] == "detuning_m@10:envelope"
    exp2 = experiment.from_config(cfg)
    assert exp2.options == exp.options and exp2.hash == exp.hash


def test_experiment_hash_does_not_count_the_extension_keys_twice(tmp_path):
    """The experiment hash is that of the config without the experiment-layer keys plus the resolved options, so it
    does not depend on whether mrm/params/schema.py declares those keys (run directories recorded before the schema
    knew them still rerun)."""
    import hashlib
    import json
    exp = experiment.load(_write(tmp_path, "e.yaml", BASE_YAML))
    tree = exp.cfg.resolved_tree()
    assert tree["experiment"]["start"] == "warm"                # the Config holds them now
    for k in experiment.EXTENSION_KEYS:
        tree["experiment"].pop(k.name, None)
    assert experiment.config_hash_without_options(exp.cfg) == hashlib.sha256(
        json.dumps(tree, sort_keys=True, default=str).encode()).hexdigest()
    assert experiment.config_hash_without_options(exp.cfg) != exp.cfg.hash()
    # an option change still changes the experiment hash (through the options part)
    assert (experiment.load(_write(tmp_path, "f.yaml", BASE_YAML.replace("start: warm", "start: cold"))).hash
            != exp.hash)


def test_dotted_override_refines_a_map_option_from_the_file(tmp_path):
    p = _write(tmp_path, "e.yaml", BASE_YAML.replace("score_from_s: 0.001", "score_from_s: 0.001\n"
                                                     "      self_reference: {duration_s: 0.01, window_s: 0.004}"))
    exp = experiment.load(p, overrides=["experiment.self_reference.window_s=0.002"])
    assert exp.options["self_reference"] == {"duration_s": 0.01, "window_s": 0.002}


def test_params_override_maps_yaml_booleans_to_enum_words():
    """--set experiment.variation=off: YAML 1.1 reads off as False; mrm.params maps it to the enum word."""
    from mrm.params.config import parse_overrides
    assert parse_overrides(["experiment.variation=off", "experiment.histogram=off"]) == {
        "experiment.variation": "off", "experiment.histogram": False}


# a locked_steady_state experiment written with the old kind name quasi_static
OLD_KIND_YAML = """\
    extends: g26
    meta: {name: t_old_kind, kind: experiment}
    experiment:
      kind: quasi_static
      uncertainty: nominal
      variation: "off"
      grid: {ambient_c: [25.0], bias_v: [2.0]}
      acceptance: [{id: cov, metric: code_range_coverage, op: ">=", value: 1.0, cases: [quasi_static]},
                   {id: cor, metric: code_range_coverage, op: ">=", value: 1.0, cases: "quasi_static/corner/*"}]
    """


def test_old_kind_name_loads_as_locked_steady_state(tmp_path):
    """The kind quasi_static was renamed locked_steady_state: an old file loads as the new kind, its
    criteria select the new case labels, and a --set with the old name works too. The hash under the old name is
    kept (legacy_hashes), so a run directory recorded under it still reruns."""
    old = experiment.load(_write(tmp_path, "old.yaml", OLD_KIND_YAML))
    new = experiment.load(_write(tmp_path, "new.yaml",
                                 OLD_KIND_YAML.replace("kind: quasi_static", "kind: locked_steady_state")
                                 .replace("cases: [quasi_static]", "cases: [locked_steady_state]")
                                 .replace('"quasi_static/corner/*"', '"locked_steady_state/corner/*"')))
    assert old.kind == new.kind == "locked_steady_state"
    assert len(old.legacy_hashes()) == 1 and old.legacy_hashes() != [old.hash]
    crit = experiment._criteria(old.option("acceptance"))
    assert crit[0].cases == ["locked_steady_state"] and crit[1].cases == "locked_steady_state/corner/*"
    assert experiment._case_selected(crit[1], "locked_steady_state/corner/g26_h86")
    assert experiment.load(_write(tmp_path, "run.yaml", BASE_YAML),
                           overrides=["experiment.kind=quasi_static"]).kind == "locked_steady_state"
    assert experiment.load(_write(tmp_path, "run2.yaml", BASE_YAML)).legacy_hashes() == []
    assert experiment.canonical_case_label("quasi_static/corner/x") == "locked_steady_state/corner/x"
    assert experiment.canonical_case_label("pi_lock_c/corner/x") == "pi_lock_c/corner/x"


def test_compare_runs_maps_old_kind_names():
    """A rerun of a run recorded under the old kind name compares with the new trace.h5 layout."""
    from mrm.sim import manifest
    a = {"trace": {"@kind": "quasi_static", "quasi_static": {"quasi_static": {"x": np.arange(3.0)},
                                                            "quasi_static|corner|c": {"x": np.ones(2)}}}}
    b = {"trace": {"@kind": "locked_steady_state",
                   "locked_steady_state": {"locked_steady_state": {"x": np.arange(3.0)},
                                           "locked_steady_state|corner|c": {"x": np.ones(2)}}}}
    cmp = manifest.compare_runs(a, b)
    assert cmp["identical"] and cmp["n_compared"] == 3


def test_extension_overrides_and_nested_options(tmp_path):
    p = _write(tmp_path, "e.yaml", BASE_YAML)
    exp = experiment.load(p, overrides=["experiment.start=cold_settled", "experiment.self_reference.duration_s=0.01",
                                        "controller.params.fraction=0.6", "experiment.duration_s=0.004"])
    assert exp.options["start"] == "cold_settled"
    assert exp.options["self_reference"] == {"duration_s": 0.01}
    assert exp.duration_s == 0.004
    assert experiment._main_controller(exp)["params"]["fraction"] == 0.6
    assert exp.hash != experiment.load(p).hash


@pytest.mark.parametrize("bad, match", [
    ("experiment.strat=warm", "did you mean 'start'"),
    ("experiment.start=hot", "not one of"),
    ("experiment.histogram=3", "expected bool"),
    ("experiment.self_reference={duration: 1}", "unknown keys"),
    ("experiment.record=[nonsense@10]", "not a Truth field"),
    ("experiment.acceptance=[{metric: x, op: '~', value: 1}]", "op"),
])
def test_extension_schema_is_strict(tmp_path, bad, match):
    p = _write(tmp_path, "e.yaml", BASE_YAML)
    with pytest.raises(experiment.ExperimentError, match=match):
        experiment.load(p, overrides=[bad])


def test_unknown_experiment_key_names_both_key_sets(tmp_path):
    p = _write(tmp_path, "e.yaml", BASE_YAML)
    with pytest.raises(experiment.ExperimentError, match="did you mean 'duration_s'"):
        experiment.load(p, overrides=["experiment.duraton_s=1"])
    bad = _write(tmp_path, "bad.yaml", BASE_YAML.replace("start: warm", "strat: warm"))
    with pytest.raises(experiment.ExperimentError, match="did you mean 'start'"):
        experiment.load(bad)


def test_unknown_core_key_is_still_a_params_error(tmp_path):
    from mrm.params.config import ConfigError
    p = _write(tmp_path, "e.yaml", BASE_YAML)
    with pytest.raises(ConfigError, match="q_loadd"):
        experiment.load(p, overrides=["ring.optics.q_loadd=1"])


def test_extension_keys_merge_through_extends_chain(tmp_path):
    _write(tmp_path, "parent.yaml", BASE_YAML)
    child = _write(tmp_path, "child.yaml", """\
        extends: parent.yaml
        meta: {name: child_exp, kind: experiment}
        experiment:
          start: cold
          n_threads: 2
        """)
    exp = experiment.load(child)
    assert exp.name == "child_exp"
    assert exp.options["start"] == "cold"                    # the child wins
    assert exp.options["score_from_s"] == 0.001              # inherited from the parent
    assert exp.n_threads == 2 and exp.duration_s == 0.003
    files = exp.cfg.files
    assert str(tmp_path / "parent.yaml") in files and str(tmp_path / "child.yaml") in files


def test_from_config_without_extension_keys(tmp_path):
    from mrm import params
    p = _write(tmp_path, "plain.yaml", """\
        extends: g26
        meta: {name: plain, kind: experiment}
        controller: {name: pi_lock_c}
        experiment: {kind: run, duration_s: 0.002, uncertainty: nominal, variation: "off"}
        """)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = params.load(p)
    exp = experiment.from_config(cfg)
    assert exp.source["path"] == str(p) and exp.options == {}
    assert exp.hash == experiment.load(p).hash


def test_input_copies_carry_the_whole_extends_chain(tmp_path):
    """runs/<id>/inputs/ holds the experiment file and a copy of every non-preset parent (a run must be
    reproducible from its directory alone); the copies' extends: name each other, preset names stay as written."""
    import hashlib
    sub = tmp_path / "lib"
    sub.mkdir()
    _write(sub, "grand.yaml", BASE_YAML)
    _write(sub, "parent.yaml", "extends: grand.yaml\nmeta: {name: p, kind: experiment}\n")
    child = _write(tmp_path, "child.yaml", "extends: lib/parent.yaml\nmeta: {name: c, kind: experiment}\n")
    cps = experiment._input_copies(child)
    assert [c["name"] for c in cps] == ["child.yaml", "_extends_1_parent.yaml", "_extends_2_grand.yaml"]
    assert "extends: _extends_1_parent.yaml" in cps[0]["text"]
    assert "extends: _extends_2_grand.yaml" in cps[1]["text"]
    assert cps[2]["text"] == (sub / "grand.yaml").read_text()   # extends: g26 (a preset name): byte for byte
    assert cps[2]["original"] == str(sub / "grand.yaml")
    assert cps[2]["sha256"] == hashlib.sha256((sub / "grand.yaml").read_bytes()).hexdigest()
    # the copies alone load to the same experiment (written to another folder, the originals out of reach)
    out = tmp_path / "elsewhere"
    out.mkdir()
    for c in cps:
        (out / c["name"]).write_text(c["text"])
    assert experiment.load(out / "child.yaml").hash == experiment.load(child).hash
    # a preset given by path is named by its preset name in the copy
    from mrm.params.config import PRESET_DIR
    bypath = _write(tmp_path, "bypath.yaml",
                    f"extends: {PRESET_DIR / 'g26.yaml'}\nmeta: {{name: b, kind: experiment}}\n")
    (c0,) = experiment._input_copies(bypath)
    assert "extends: g26\n" in c0["text"]


def test_input_copies_are_taken_when_the_experiment_is_loaded(tmp_path):
    p = _write(tmp_path, "e.yaml", BASE_YAML)
    exp = experiment.load(p)
    p.write_text(p.read_text().replace("duration_s: 0.003", "duration_s: 0.004"))     # edited during a long run
    (c0,) = exp.source["input_copies"]
    assert "duration_s: 0.003" in c0["text"]


def test_parse_records():
    r = experiment.parse_records(["detuning_m@10:envelope", {"channel": "reloma", "decimation": 20},
                                  "heater_code@auto:last"], n_ticks=1_000_000, period_ticks=10)
    keys = [x.key for x in r]
    assert keys == ["detuning_m@10:min", "detuning_m@10:max", "reloma@20:mean", "heater_code@100:last"]
    d = experiment.default_records(1_218_000_000, 10)
    assert all(x.decimation == 121_800 for x in d) and 1_218_000_000 // d[0].decimation <= 10_000


def _fake_metrics(B=2, N=3, **over):
    """A metrics dict shaped like an engine summary for B worlds and N rings (every ring locked and well inside the
    limits). Keyword arguments replace entries."""
    z = np.zeros((B, N))
    m = {"oma_penalty_db": z + 0.1, "excursion_count": np.zeros((B, N), np.int64), "excursion_max_s": z.copy(),
         "time_unlocked_s": z.copy(), "wrong_line_final": np.zeros((B, N), bool),
         "wrong_line_events": np.zeros((B, N), np.int64), "locked_final": np.ones((B, N), bool),
         "max_abs_lock_error_pm": z + 1.0, "rms_lock_error_pm": z + 0.5, "mean_reloma_db": z - 6.0,
         "oracle_reloma_db": z - 5.9, "fsr_hop_events": np.zeros((B, N), np.int64), "ber_mean": z + 1e-20,
         "heater_sat_s": z.copy(), "heater_low_rail_s": z.copy(), "heater_range_used_codes": np.zeros((B, N), int),
         "time_to_lock_s": z + 1e-3, "world": {"excursion_count": np.zeros(B)}, "time_above_s": {"5pm": z.copy()}}
    m.update(over)
    return m


def test_world_summary():
    ttl = np.array([[1e-3, 2e-3, 3e-3], [1e-3, np.nan, 1e-3]])
    pen = np.array([[0.1, 0.5, np.nan], [0.2, 0.2, 0.2]])
    ws = experiment.world_summary(_fake_metrics(time_to_lock_s=ttl, oma_penalty_db=pen,
                                                wrong_line_final=np.array([[0, 1, 1], [0, 0, 0]], bool)))
    assert ws["time_to_lock_s"].tolist() == [3e-3, math.inf]       # a ring that ends unlocked: inf
    assert np.allclose(ws["oma_penalty_db_worst"], [0.5, 0.2])     # NaN rings ignored
    assert ws["wrong_line_rings"].tolist() == [2.0, 0.0]
    assert ws["unlocked_rings"].tolist() == [0.0, 0.0]


def test_criteria_ops_reduce_and_cases():
    m1 = _fake_metrics(max_abs_lock_error_pm=np.array([[1.0, 2.0, 7.0], [1.0, 1.0, 1.0]]))
    m2 = _fake_metrics(max_abs_lock_error_pm=np.array([[1.0, 2.0, 3.0], [1.0, 1.0, np.nan]]))
    crit = experiment._criteria([
        {"id": "a", "metric": "max_abs_lock_error_pm", "op": "<=", "value": 5.0},
        {"id": "b", "metric": "max_abs_lock_error_pm", "op": "<=", "value": 5.0, "rings": [0, 1]},
        {"id": "c", "metric": "max_abs_lock_error_pm", "op": "<=", "value": 5.0, "cases": ["x/"]},
        {"id": "d", "metric": "locked_final", "reduce": "all", "op": "==", "value": 1},
        {"id": "e", "metric": "world.excursion_count", "op": "==", "value": 0},
        {"id": "f", "metric": "time_above_s.5pm", "op": "==", "value": 0.0, "reduce": "sum"},
        {"id": "g", "metric": "wrong_line_rings", "op": "==", "value": 0},
        {"id": "h", "metric": "nonexistent", "op": "==", "value": 0},
    ])
    ev = {e["id"]: e for e in experiment.evaluate_criteria(crit, {"main": m1, "x/w": m2})}
    assert not ev["a"]["pass"] and ev["a"]["worst"] != ev["a"]["worst"]   # NaN in m2 never holds: worst is NaN
    assert ev["b"]["pass"]                                       # rings 0 and 1 only
    assert not ev["c"]["pass"] and set(ev["c"]["per_case"]) == {"x/w"}   # case prefix selection; NaN fails
    assert ev["d"]["pass"] and ev["e"]["pass"] and ev["f"]["pass"] and ev["g"]["pass"]
    assert not ev["h"]["pass"] and "error" in ev["h"]["per_case"]["main"]


def test_criteria_reduce_none_honours_rings_and_unknown_metrics_fail_cleanly():
    m = _fake_metrics(max_abs_lock_error_pm=np.array([[1.0, 2.0, 9.0], [1.0, 1.0, 9.0]]))
    crit = experiment._criteria([
        {"id": "none_rings", "metric": "max_abs_lock_error_pm", "reduce": "none", "rings": [0, 1], "op": "<=",
         "value": 5.0},
        {"id": "none_all", "metric": "max_abs_lock_error_pm", "reduce": "none", "op": "<=", "value": 5.0},
    ])
    ev = {e["id"]: e for e in experiment.evaluate_criteria(crit, {"main": m})}
    assert ev["none_rings"]["pass"] and ev["none_rings"]["per_case"]["main"]["values"] == [[1.0, 2.0], [1.0, 1.0]]
    assert not ev["none_all"]["pass"]
    # a metrics dict that is not an engine summary (a steady-state survey): an unknown name is a failed criterion,
    # not a crash
    qs = {"code_range_coverage": np.array([1.0]), "world": {"bnd4_max_db": np.array([0.0])}}
    crit = experiment._criteria([{"id": "x", "metric": "no_such_metric", "op": "==", "value": 0}])
    (e,) = experiment.evaluate_criteria(crit, {"locked_steady_state": qs})
    assert not e["pass"] and "unknown metric" in e["per_case"]["locked_steady_state"]["error"]


def test_criteria_nan_never_holds_for_not_equal():
    """NaN fails a '!=' criterion too (a time to lock of a ring that ends unlocked), as it fails every other op."""
    m = {"time_to_lock_s": np.array([[np.nan, 1e-3]])}
    (ne,) = experiment.evaluate_criteria([experiment.Criterion(id="ne", metric="time_to_lock_s", op="!=", value=0.0)],
                                         {"main": m})
    assert not ne["pass"]
    (ok,) = experiment.evaluate_criteria([experiment.Criterion(id="ok", metric="time_to_lock_s", op="!=", value=0.0)],
                                         {"main": {"time_to_lock_s": np.array([[2e-3, 1e-3]])}})
    assert ok["pass"]


def test_monitor_decimation_tiles_the_scored_window():
    """The monitor min / max windows of a self-referenced run start at the scoring start and end at the end of the
    run whenever a common divisor allows it (only complete windows are recorded: a remainder is never judged)."""
    for n, sf in ((135_700, 5_000), (8_300, 4_000), (455_000, 5_000), (60_000, 5_000), (2_000_000, 0)):
        d = experiment._monitor_decimation(n, sf)
        assert n % d == 0 and (sf == 0 or sf % d == 0) and n // d <= 10_000, (n, sf, d)
    # A1 run (135 700 ticks): 5000-tick windows would leave the last 0.7 ms unjudged
    assert experiment._monitor_decimation(135_700, 5_000) == 100
    assert experiment._monitor_decimation(1_000_000, 0) == 1_000_000
    # no common divisor is large enough: aligned at the scoring start, <= 10 000 windows
    d = experiment._monitor_decimation(100_003, 5_000)
    assert 5_000 % d == 0 and 100_003 // d <= 10_000
    assert experiment._monitor_decimation(1_000_003, 7) == 101               # scoring start shorter than one window


@pytest.mark.parametrize("sr, match", [({"duration_s": -0.01}, "positive"), ({"window_s": "x"}, "positive"),
                                       ({"duration_s": 0.004, "window_s": 0.005}, "exceeds")])
def test_self_reference_values_are_validated(tmp_path, sr, match):
    import json
    p = _write(tmp_path, "e.yaml", BASE_YAML)
    with pytest.raises(experiment.ExperimentError, match=match):
        experiment.load(p, overrides=[f"experiment.self_reference={json.dumps(sr)}"])


def test_rank_metrics_must_be_lower_is_better_world_metrics(tmp_path):
    p = _write(tmp_path, "e.yaml", BASE_YAML)
    assert experiment.load(p, overrides=["experiment.rank_metrics=[excursion_count]"]).option("rank_metrics") == [
        "excursion_count"]
    for bad in ("[mean_reloma_db]", "[excursion_cnt]", "[]"):
        with pytest.raises(experiment.ExperimentError, match="rank_metrics"):
            experiment.load(p, overrides=[f"experiment.rank_metrics={bad}"])


def test_artefacts_are_merged_over_every_case_config(tmp_path, monkeypatch):
    from mrm.sim import manifest
    exp = experiment.load(_write(tmp_path, "e.yaml", BASE_YAML))
    corner = object()
    exp._cache[("corner",)] = corner
    table = {id(exp.cfg): [{"file": "a.json", "used_by": ["p"], "selected": True, "sha256": "1"},
                           {"file": "b.json", "used_by": ["q#alt"], "selected": False, "sha256": "2"}],
             id(corner): [{"file": "b.json", "used_by": ["q"], "selected": True, "sha256": "2"},
                          {"file": "c.npz", "used_by": ["ring.thermal.rom_file"], "selected": True, "sha256": "3"}]}
    monkeypatch.setattr(manifest, "artefacts_used", lambda cfg, pp=None, **kw: [dict(a) for a in table[id(cfg)]])
    arts = {a["file"]: a for a in experiment._artefacts_of_cases(exp)}
    assert set(arts) == {"a.json", "b.json", "c.npz"}
    assert arts["b.json"]["selected"] and arts["b.json"]["used_by"] == ["q#alt", "q"]


def test_estimate_takes_options_and_each_case_controller(tmp_path):
    p = _write(tmp_path, "e.yaml", BASE_YAML)
    assert experiment.estimate(p, options={"n_threads": 3})["n_threads"] == 3
    sw = experiment.estimate(p, overrides=["experiment.kind=sweep",
                                           "experiment.grid={controller.name: [pi_lock_c, pi_lock]}"])
    assert [(r["controller"], r["path"]) for r in sw["cases"]] == [("pi_lock_c", "compiled"), ("pi_lock", "python")]


def test_estimate_record_memory_uses_each_case_ring_count(tmp_path):
    """The record memory of a case scales with that case's own ring count (a sweep over array.n_rings)."""
    p = _write(tmp_path, "e.yaml", BASE_YAML)
    est = experiment.estimate(p, overrides=["experiment.kind=sweep", "experiment.grid={array.n_rings: [4, 8]}"])
    four, eight = est["cases"]
    assert four["records"] == eight["records"] > 0
    assert eight["record_mb"] == pytest.approx(2.0 * four["record_mb"])


def test_a_bad_sweep_point_fails_before_any_case_runs(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(experiment, "_execute", lambda *a, **k: calls.append(a) or None)
    p = _write(tmp_path, "e.yaml", BASE_YAML)
    for grid in ("{ring.optics.q_loaded: [3600, -1]}", "{corner: [g26_h86, g26]}",
                 "{controller: [pi_lock_c, {name: pi_lock_c, params: {no_such_param: 1}}]}"):
        with pytest.raises(Exception):
            experiment.run(p, overrides=["experiment.kind=sweep", f"experiment.grid={grid}"], write=False,
                           log=lambda s: None)
    assert calls == []


def test_band_either_sign():
    lo, hi = np.array([[-1.0, -2.0]]), np.array([[2.9, 0.5]])
    m = {"err_min": lo, "err_max": hi}
    crit = experiment._criteria([{"id": "band", "metric": "err", "op": "band", "value": [-2.2, 3.3]},
                                 {"id": "mirror", "metric": "err", "op": "band_either_sign", "value": [-1.0, 3.3]}])
    ev = {e["id"]: e for e in experiment.evaluate_criteria(crit, {"main": m})}
    assert ev["band"]["pass"]
    # [-2, 2.9] is outside [-1, 3.3] but inside the mirrored band [-3.3, 1.0]? no: 2.9 > 1.0 -> fail
    assert not ev["mirror"]["pass"]
    m2 = {"err_min": np.array([[-3.0]]), "err_max": np.array([[0.8]])}
    ev2 = {e["id"]: e for e in experiment.evaluate_criteria(crit, {"main": m2})}
    assert ev2["mirror"]["pass"] and not ev2["band"]["pass"]


def test_rank_stability_kendall_tau():
    vals = {"a": np.array([1.0, 1.0, 3.0]), "b": np.array([2.0, 2.0, 1.0]), "c": np.array([3.0, 3.0, 2.0])}
    nom = {"a": 1.0, "b": 2.0, "c": 3.0}
    rs = experiment.rank_stability(vals, nom)
    assert rs["nominal_order"] == ["a", "b", "c"]
    assert rs["n_worlds"] == 3 and math.isclose(rs["frac_same_order"], 2 / 3)
    assert math.isclose(rs["tau_min"], -1 / 3) and math.isclose(rs["tau_mean"], (1 + 1 - 1 / 3) / 3)
    # NaN and inf count as worst; one controller -> undefined
    rs2 = experiment.rank_stability({"a": np.array([np.nan]), "b": np.array([1.0])}, {"a": 0.0, "b": 1.0})
    assert rs2["tau_mean"] == -1.0
    assert experiment.rank_stability({"a": np.array([1.0])}, {"a": 1.0})["n_worlds"] == 0


def test_plant_fingerprint_covers_plant_controller_and_start_only(tmp_path):
    p = _write(tmp_path, "e.yaml", BASE_YAML)
    a = experiment.load(p)
    b = experiment.load(p, overrides=["experiment.duration_s=0.5", "scenario.time_compression=10"])
    c = experiment.load(p, overrides=["controller.params.fraction=0.6"])
    d = experiment.load(p, overrides=["ring.optics.q_loaded=4000"])
    w = {"batch": 1, "uncertainty": "nominal", "variation": "off", "seeds": a.seeds}
    fa = experiment.plant_fingerprint(a.cfg, "warm", w)
    assert fa == experiment.plant_fingerprint(b.cfg, "warm", w)          # experiment section and compression excluded
    assert fa != experiment.plant_fingerprint(a.cfg, "cold", w)
    assert fa != experiment.plant_fingerprint(c.cfg, "warm", w) and fa != experiment.plant_fingerprint(d.cfg, "warm", w)
    assert fa != experiment.plant_fingerprint(a.cfg, "warm", dict(w, uncertainty="sample", batch=8))   # other worlds


def test_corners_from_presets_and_maps(tmp_path):
    p = _write(tmp_path, "e.yaml", BASE_YAML)
    exp = experiment.load(p, overrides=[
        "experiment.corners=[g26_h86, {name: hot, alts: {uncertainty.q_loaded: low_q}}]"])
    c = experiment._corners(exp, ())
    assert c[0] == ("g26_h86", {"uncertainty.thermo_optic": "H86"})
    assert c[1] == ("hot", {"uncertainty.q_loaded": "low_q"})
    assert ([n for n, _ in experiment._corners(experiment.load(p), experiment.NAMED_CORNERS)]
            == list(experiment.NAMED_CORNERS))
    with pytest.raises(experiment.ExperimentError, match="not a corner preset"):
        experiment._corners(experiment.load(p, overrides=["experiment.corners=[g26]"]), ())


def test_named_corners_exist():
    from mrm.params.config import PRESET_DIR
    for c in experiment.NAMED_CORNERS:
        assert (PRESET_DIR / f"{c}.yaml").exists(), c


def test_plan_compare_and_sweep(tmp_path):
    p = _write(tmp_path, "e.yaml", BASE_YAML)
    exp = experiment.load(p, overrides=["experiment.kind=compare", "experiment.controllers=[pi_lock_c, kalman_lqg_c]",
                                        "experiment.n_mc=4", "experiment.corners=[g26_h86]",
                                        "experiment.uncertainty=lhs"])
    labels = [c.label for c in experiment._plan(exp)]
    assert labels == ["pi_lock_c/nominal", "pi_lock_c/corner/g26_h86", "pi_lock_c/worlds",
                      "kalman_lqg_c/nominal", "kalman_lqg_c/corner/g26_h86", "kalman_lqg_c/worlds"]
    worlds = [c for c in experiment._plan(exp) if c.label.endswith("worlds")][0]
    assert worlds.batch == 4 and worlds.uncertainty == "lhs"
    sw = experiment.load(p, overrides=["experiment.kind=sweep",
                                       "experiment.grid={ring.optics.q_loaded: [3600, 4000], "
                                       "'alt:uncertainty.thermo_optic': [H_solver, H86], "
                                       "controller: [pi_lock_c, mpc_c]}"])
    cases = experiment._plan(sw)
    assert len(cases) == 8
    assert cases[0].overrides == ("ring.optics.q_loaded=3600",)
    assert cases[0].alts == {"uncertainty.thermo_optic": "H_solver"}
    assert cases[1].controller["name"] == "mpc_c"


def test_mc_chunks_and_lhs_limit(tmp_path):
    p = _write(tmp_path, "e.yaml", BASE_YAML)
    exp = experiment.load(p, overrides=["experiment.kind=mc", "experiment.n_mc=5", "experiment.uncertainty=sample",
                                        "experiment.max_worlds_per_plant=2", "experiment.seeds.noise=7"])
    cases = experiment._plan(exp)
    assert [c.batch for c in cases] == [2, 2, 1]
    assert [c.seeds["noise"] for c in cases] == [7, 9, 11]
    lhs = experiment.load(p, overrides=["experiment.kind=mc", "experiment.n_mc=5", "experiment.uncertainty=lhs",
                                        "experiment.max_worlds_per_plant=2"])
    with pytest.raises(experiment.ExperimentError, match="Latin hypercube"):
        experiment._plan(lhs)


def test_compressed_scenario_only_in_run_and_acceptance(tmp_path):
    p = _write(tmp_path, "e.yaml", BASE_YAML)
    exp = experiment.load(p, overrides=["experiment.kind=mc", "experiment.n_mc=2", "experiment.uncertainty=sample",
                                        "scenario.name=soak", "scenario.time_compression=10"])
    with pytest.raises(experiment.ExperimentError, match="convergence certificate per plant"):
        experiment.run(exp, write=False, log=lambda s: None)


def test_estimate_uses_the_measured_engine_speed(tmp_path):
    p = _write(tmp_path, "e.yaml", BASE_YAML)
    est = experiment.estimate(p)
    (row,) = est["cases"]
    assert row["B"] == 1 and row["ticks"] == 3000 and row["path"] == "compiled" and row["controller_edges"] == 300
    assert est["est_wall_s"] > 0 and est["est_peak_ram_gb"] > 0.2
    assert "docs/handoff/perf.md section 8" in est["basis"]
    soak = experiment.estimate(ROOT / "experiments" / "acceptance" / "a4_soak.yaml")
    labels = [r["case"] for r in soak["cases"]]
    assert labels == ["convergence_c1", "convergence_c10", "convergence_c100", "main"]
    assert soak["cases"][0]["ticks"] == 1_218_000_000 and soak["cases"][-1]["ticks"] == 1_096_200_000
    assert soak["substep_policy"] == "auto"


def test_every_shipped_experiment_file_loads_and_estimates():
    files = sorted((ROOT / "experiments").glob("*/*.yaml"))
    assert len(files) >= 15
    for f in files:
        exp = experiment.load(f)
        assert exp.kind in experiment.KINDS
        if f.parent.name == "acceptance":
            assert exp.option("acceptance"), f"{f.name}: an acceptance file without criteria"
        est = experiment.estimate(exp)
        assert est["cases"], f.name


def test_every_shipped_experiment_file_loads_through_mrm_params():
    """mrm.run(mrm.params.load('exp.yaml')) is the same experiment as mrm.run('exp.yaml') for every shipped file
    (every experiment key is a schema key)."""
    from mrm import params
    files = sorted((ROOT / "experiments").glob("*/*.yaml"))
    for f in files:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = params.load(f)
        a, b = experiment.from_config(cfg), experiment.load(f)
        assert a.options == b.options and a.hash == b.hash, f.name


def test_extension_keys_known_to_the_schema_are_read_from_the_config(tmp_path, monkeypatch):
    """Forward compatibility: once mrm/params/schema.py knows an extension key, it stays in the
    Config and the experiment reads it from there instead of ignoring it."""
    from mrm.params import schema
    key = schema.S("experiment.start", "enum", "test", enum=experiment.STARTS)
    monkeypatch.setitem(schema.SCHEMA, "experiment.start", key)
    monkeypatch.setattr(schema, "OPEN_PREFIXES", schema.OPEN_PREFIXES + ("experiment.self_reference",))
    text = BASE_YAML.replace("score_from_s: 0.001", "score_from_s: 0.001\n      self_reference: {duration_s: 0.01}")
    exp = experiment.load(_write(tmp_path, "e.yaml", text))
    assert exp.cfg.get("experiment.start") == "warm"                # validated by mrm.params now
    assert exp.options["start"] == "warm" and exp.options["self_reference"] == {"duration_s": 0.01}
