"""Checks rules of the experiment layer that a final review fixed: hop counting, certificates, corners, doc text.

Read this first
---------------
The tests cover: the lock-independent count of order changes (FSR hops), which the A4-hop criterion, the golden
soak check and the convergence comparison use; the engine-code fingerprint of a convergence certificate; reruns of
a certificate-reuse run from its own directory; the per-ring absolute tolerance on max detuning in the convergence
comparison; corner contents in the experiment hash and in inputs/; and doc strings that name code behaviour.
~25 s with a warm numba cache (one mini soak with its convergence experiment, a few short runs).
"""
from __future__ import annotations

import json
import inspect
import re
import shutil
import textwrap
from pathlib import Path

import numpy as np
import pytest

from mrm.sim import experiment
from mrm.sim import manifest

ROOT = Path(__file__).resolve().parents[2]
QUIET = {"log": lambda s: None}   # keyword arguments that silence experiment.run / rerun

# the mini soak of test_experiment_runs.py (compression 2, one 0.2 s cycle at factors 1 and 2)
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


def _write(d: Path, name: str, text: str) -> Path:
    """Write the dedented text to d / name and return the path."""
    p = d / name
    p.write_text(textwrap.dedent(text))
    return p


@pytest.fixture(scope="module")
def cert_run(tmp_path_factory):
    """A mini soak that ran its own convergence experiment (its run directory holds the certificate)."""
    d = tmp_path_factory.mktemp("cert")
    p = _write(d, "soak.yaml", MINI_SOAK_YAML)
    res = experiment.run(p, runs_dir=d / "runs", **QUIET)
    assert res.status == "ok"
    return res


# --------------------------------------------------------------------------------------------- order changes (FSR hops)

def test_order_changes_count_a_hop_that_max_detuning_misses():
    """An open-loop heater step from the cold state hops every ring by one order (g26 nominal, code 52000, 3 ms).
    The detuning is always measured from the order nearest the declared line, so |d| <= FSR/2 by construction:
    every ring stays below a 5000 pm bound on max detuning (an A4-hop criterion of that form would pass), and
    fsr_hop_events (re-locks only) stays 0. The lock-independent order count sees the hop, and A4-hop judges it."""
    from mrm.sim import runner
    r = runner.run("g26", None, "static", 3e-3, uncertainty="nominal", variation="off", reset_protocol="branch_low",
                   heater_code=np.full((1, 8), 52000), substep_policy="fixed", record=[("order", 1, "last")])
    m = r.metrics
    order = np.asarray(list(r.trace.values())[0]["value"])
    assert np.array_equal(m["order_changes"], (np.diff(order, axis=0) != 0).sum(axis=0))
    assert np.all(m["order_changes"] == 1) and np.all(m["fsr_hop_events"] == 0)
    assert np.all(m["max_abs_detuning_pm"] < 5000.0)          # a max-detuning criterion would pass this run
    assert np.array_equal(m["world"]["order_changes"], m["order_changes"].sum(axis=1))
    exp = experiment.load(ROOT / "experiments" / "acceptance" / "a4_soak.yaml")
    crit = [c for c in experiment._criteria(exp.option("acceptance")) if c.id == "A4-hop"]
    assert len(crit) == 1 and crit[0].metric == "order_changes"
    ev = experiment.evaluate_criteria(crit, {"main": m})
    assert not ev[0]["pass"]


def test_convergence_compares_the_order_count():
    from mrm.scenarios.generators import CONVERGENCE_METRICS, DEFAULT_CONVERGENCE_TOLERANCES
    assert "order_changes" in CONVERGENCE_METRICS and DEFAULT_CONVERGENCE_TOLERANCES["order_changes"] == (0.0, 0.0)


# -------------------------------------------------------------------------------- convergence tolerance on max detuning

def _convergence_results(ref_det, comp_det):
    """Convergence-run metrics at compression 1 (reference) and 100, differing only in max_abs_detuning_pm."""
    base = {"wrong_line_events": 0.0, "fsr_hop_events": 0.0, "order_changes": 0.0, "heater_range_used_codes": 26029.0,
            "integrated_errors": 0.0}
    return {1.0: dict(base, max_abs_detuning_pm=ref_det), 100.0: dict(base, max_abs_detuning_pm=comp_det)}


def test_convergence_holds_max_detuning_to_an_absolute_tolerance_per_ring():
    """Max detuning converges when it agrees within 0.2 pm or 5 %, at most 0.5 pm (a tenth of the 5 pm lock budget),
    ring by ring. A plain 5 % of the offset-dominated |d| (126.5 pm for pi_lock_c in A4) would allow +-6.3 pm, more
    than the lock budget, and comparing only the worst ring would miss a change in any other ring."""
    from mrm.scenarios import assess_convergence, registry
    sc = registry.create("soak")
    assert assess_convergence(sc, _convergence_results(126.4963, 126.4963 + 0.45)).passed
    rep = assess_convergence(sc, _convergence_results(126.4963, 126.4963 + 0.6))   # a plain 5 % would allow 6.32 pm
    assert not rep.passed and "max_abs_detuning_pm" in rep.failures[0]
    ref = np.array([126.4963, 120.0, 125.3, 110.0, 121.0, 122.0, 123.0, 124.0])
    comp = ref.copy()
    comp[1] += 1.0                                       # not the worst ring: the max over rings does not move
    assert comp.max() == ref.max()
    rep = assess_convergence(sc, _convergence_results(ref, comp))
    assert not rep.passed and "ring 1" in rep.failures[0]
    assert assess_convergence(sc, _convergence_results(ref, ref + 0.1)).passed
    # near the line: 0.2 pm, then 5 %
    assert not assess_convergence(sc, _convergence_results(3.0, 3.25)).passed
    assert assess_convergence(sc, _convergence_results(8.0, 8.38)).passed
    assert not assess_convergence(sc, _convergence_results(8.0, 8.42)).passed


def test_convergence_experiment_records_max_detuning_per_ring(cert_run):
    conv = json.loads((cert_run.run_dir / "convergence.json").read_text())
    d = conv["worlds"][0]["details"]["2.0"]["max_abs_detuning_pm"]
    assert len(d["reference"]) == 8 and d["allowed"] == [0.5] * 8
    assert len(conv["certificate"]["metrics"]["2.0"]["max_abs_detuning_pm"][0]) == 8
    assert conv["certificate"]["code_fingerprint"] == manifest.code_fingerprint()


# ----------------------------------------------------------------------------- engine-code fingerprint of a certificate

def test_a_certificate_of_other_engine_code_is_refused(cert_run, tmp_path, monkeypatch):
    """A certificate made by other engine code (or numerics) does not cover this run."""
    p = _write(tmp_path, "soak.yaml", MINI_SOAK_YAML)
    ov = [f"experiment.convergence_certificate={cert_run.run_dir}", "experiment.duration_s=0.02"]
    assert experiment.run(p, overrides=ov, write=False, **QUIET).status == "ok"
    real = manifest.code_fingerprint()
    monkeypatch.setattr(manifest, "code_fingerprint", lambda: "f" * 64)
    r = experiment.run(p, overrides=ov, write=False, **QUIET)
    assert r.status == "refused" and "engine code" in r.report and not r.cases
    monkeypatch.setattr(manifest, "code_fingerprint", lambda: real)
    # a certificate written before it recorded the code (the 2026-10-10 A4 certificate) is refused too
    old = tmp_path / "old_cert"
    old.mkdir()
    d = json.loads((cert_run.run_dir / "convergence.json").read_text())
    d["certificate"].pop("code_fingerprint")
    (old / "convergence.json").write_text(json.dumps(d))
    r = experiment.run(p, overrides=[f"experiment.convergence_certificate={old}", "experiment.duration_s=0.02"],
                       write=False, **QUIET)
    assert r.status == "refused" and "records no engine-code fingerprint" in r.report


def test_code_fingerprint_covers_the_engine_sources_data_and_numerics(monkeypatch):
    files = manifest.code_fingerprint_files()
    rel = {str(f.relative_to(ROOT)) for f in files}
    assert {"mrm/sim/kernel.py", "mrm/plant/selfheat.py", "mrm/scenarios/generators.py", "mrm/params/realize.py",
            "mrm/control/baselines/pi_lock.py", "mrm/core/rng.py", "mrm/materials.py",
            "mrm/data/materials.json"} <= rel
    assert not any(r.startswith(("mrm/validation/", "mrm/calibration/", "mrm/params/presets/")) for r in rel)
    a = manifest.code_fingerprint()
    env = manifest.environment()
    monkeypatch.setattr(manifest, "environment", lambda: dict(env, numba="0.0.0"))
    assert manifest.code_fingerprint() != a


# ------------------------------------------------------------------------------------- rerun of a certificate-reuse run

def test_a_certificate_reuse_run_reruns_from_its_own_directory(cert_run, tmp_path, monkeypatch):
    """The certificate path is recorded as given (relative to the working directory); a rerun takes the certificate
    from the run directory's own convergence.json, so it reruns elsewhere, without the certificate run."""
    w1 = tmp_path / "w1"
    w1.mkdir()
    shutil.copytree(cert_run.run_dir, w1 / "runs" / "cert")
    monkeypatch.chdir(w1)
    p = _write(w1, "soak.yaml", MINI_SOAK_YAML)
    r2 = experiment.run(p, overrides=["experiment.convergence_certificate=runs/cert", "experiment.duration_s=0.02"],
                        runs_dir=w1 / "out", **QUIET)
    assert r2.status == "ok"
    w2 = tmp_path / "w2"
    w2.mkdir()
    shutil.copytree(r2.run_dir, w2 / "r2")
    monkeypatch.chdir(w2)
    shutil.rmtree(w1)
    r3, cmp = experiment.rerun(w2 / "r2", runs_dir=w2 / "out", **QUIET)
    assert r3.status == "ok" and cmp["identical"], cmp["differing"][:5]
    conv = json.loads((r3.run_dir / "convergence.json").read_text())
    assert conv["from"] == "runs/cert" and conv["certificate"] == json.loads(
        (w2 / "r2" / "convergence.json").read_text())["certificate"]


# ----------------------------------------------------------------------------------- corners in the hash and in inputs/

# a steady-state survey with one corner; CORNER is replaced by the path of a corner file
SURVEY_CORNER_YAML = """\
    extends: g26
    meta: {name: t_qs_corner, kind: experiment}
    experiment:
      kind: locked_steady_state
      uncertainty: nominal
      variation: "off"
      grid: {ambient_c: [25.0, 45.0], bias_v: [2.0]}
      corners: [CORNER]
    """


def test_corner_contents_reach_the_hash_and_the_run_directory(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    lib.mkdir()
    corner = lib / "corner.yaml"
    corner.write_text("meta: {name: t_corner, kind: corner}\nalts: {uncertainty.thermo_optic: H86}\n")
    p = _write(tmp_path, "qs.yaml", SURVEY_CORNER_YAML.replace("CORNER", str(corner)))
    h0 = experiment.load(p).hash
    res = experiment.run(p, runs_dir=tmp_path / "runs", **QUIET)
    names = sorted(x.name for x in (res.run_dir / "inputs").iterdir())
    assert names == ["_corner_1_corner.yaml", "qs.yaml"]
    man = json.loads((res.run_dir / "manifest.json").read_text())
    assert man["source"]["inputs"][1] == {"file": "inputs/_corner_1_corner.yaml", "original": str(corner),
                                          "sha256": manifest.file_sha256(corner), "corner": str(corner)}
    # an edit of the corner changes the experiment hash (rerun refuses: the presets changed) ...
    corner.write_text("meta: {name: t_corner, kind: corner}\nalts: {uncertainty.thermo_optic: H86_sensor}\n")
    assert experiment.load(p).hash != h0
    # ... and the run directory reruns from its own copy of the corner, also once the original is gone
    corner.unlink()
    monkeypatch.chdir(tmp_path)
    r2, cmp = experiment.rerun(res.run_dir, runs_dir=tmp_path / "runs", **QUIET)
    assert cmp["identical"], cmp["differing"][:5]
    assert r2.experiment_hash == res.experiment_hash


def test_named_corner_presets_reach_the_compare_hash(tmp_path, monkeypatch):
    """compare / robust run the NAMED_CORNERS by default: an edit of a shipped corner preset changes the hash."""
    p = _write(tmp_path, "cmp.yaml", """\
        extends: g26
        meta: {name: t_cmp_hash, kind: experiment}
        controller: {name: pi_lock_c}
        scenario: {name: static}
        experiment: {kind: robust, duration_s: 0.002, n_mc: 2}
        """)
    exp = experiment.load(p)
    h0 = exp.hash
    real = experiment._corner_file_data

    def edited(path):
        data = real(path)
        if Path(path).stem == "g26_hot_selfheat":
            data = dict(data, alts=dict(data.get("alts") or {}, **{"uncertainty.thermo_optic": "H86"}))
        return data
    monkeypatch.setattr(experiment, "_corner_file_data", edited)
    assert exp.hash != h0
    # an experiment without corners keeps the hash formula of the recorded runs (config + options)
    q = _write(tmp_path, "run.yaml", """\
        extends: g26
        meta: {name: t_run_hash, kind: experiment}
        experiment: {kind: run, duration_s: 0.001}
        """)
    e = experiment.load(q)
    opts = {k.name: e.option(k.name) for k in experiment.EXTENSION_KEYS}
    blob = json.dumps({"config": experiment.config_hash_without_options(e.cfg), "options": manifest.to_jsonable(opts)},
                      sort_keys=True)
    import hashlib
    assert e.hash == hashlib.sha256(blob.encode()).hexdigest()


# ----------------------------------------------------------------------------------- doc text that names code behaviour

def test_warm_start_option_doc_names_the_runner_mode_it_uses():
    doc = experiment._EXTENSION_BY_NAME["start"].doc
    kw = experiment._start_kwargs("warm")
    assert f"heater_code='{kw['heater_code']}'" in doc and kw["reset_protocol"] in doc


def test_manifest_layout_names_the_input_copies_and_the_certificate():
    doc = inspect.getcomments(manifest)
    assert "_extends_<i>_<name>" in doc and "convergence.json" in doc and "_corner_<i>_<name>" in doc
    assert "made absolute" not in doc


def test_cli_usage_names_existing_experiment_files():
    from mrm.sim import cli
    paths = re.findall(r"experiments/[\w/]+\.yaml", inspect.getcomments(cli))
    assert paths and all((ROOT / q).exists() for q in paths), paths


def test_experiments_handoff_states_the_integrated_loader():
    text = (ROOT / "docs" / "handoff" / "experiments.md").read_text()
    sec3 = text.split("## 3. Experiment files")[1].split("## 4. Kinds")[0]
    assert "fails today" not in sec3 and "stripped temp copy" not in sec3
    assert "mrm.params.load('exp.yaml')" in sec3


def test_controller_readme_states_the_swing_of_its_a1_number():
    text = (ROOT / "examples" / "controllers" / "README.md").read_text()
    a1 = text.split("The slide-10 aggressor (A1)")[1].split("\n- ")[0]
    assert "20 K swing" in a1 and "heater authority" in a1
