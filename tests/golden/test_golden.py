"""Golden traces: has a refactor changed the plant that earlier controller results were scored on?

Read this first. These tests block CI. Each case of mrm.control.cli.GOLDEN_CASES is rerun on the engine, and its
summary metrics are compared with tests/golden/<preset>__<scenario>.npz (the comparison rule and the tolerance are in
the mrm.control.cli module docstring; tests/golden/README.md lists the cases). A file changes only through
``mrm golden --update --reason "..."``, and docs/CHANGELOG.md must then name its data hash
(test_every_golden_file_is_in_the_changelog).

Runtime with a warm numba cache: ~2 s (static) and ~5 s (A1, three runs of 130.7 ms) on the shared M2; the
compressed soak (one run of 12.18 s simulated under the auto policy) 95 s on an idle machine, ~3.5 min under
load, so its case is marked slow.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mrm.control import cli as golden_cli         # the golden cases, their runner and file format

ROOT = Path(__file__).resolve().parents[2]
CASES = sorted(golden_cli.GOLDEN_CASES)
SLOW_CASES = {"g26__soak_compressed"}            # 95-220 s of wall time (12.18 s simulated)


@pytest.mark.parametrize("name", [pytest.param(n, marks=pytest.mark.slow) if n in SLOW_CASES else n for n in CASES])
def test_golden_case(name):
    """Rerun one golden case and compare its metrics with the committed file."""
    case = golden_cli.GOLDEN_CASES[name]
    if case.pending:
        pytest.skip(f"pending golden case: {case.pending}")
    path = golden_cli.golden_path(name)
    assert path.exists(), f"{path} is missing: `mrm golden --update --case {name} --reason ...`"
    diffs = golden_cli.compare_golden(golden_cli.load_golden(path), golden_cli.run_golden_case(case))
    assert not diffs, (f"{name}: the engine no longer reproduces the golden metrics. If the change is intended, "
                       f"run `mrm golden --update --case {name} --reason \"...\"` and add the printed line to "
                       f"docs/CHANGELOG.md.\n" + "\n".join(diffs[:60]))


def test_every_golden_file_is_in_the_changelog():
    """Golden files change only with a CHANGELOG line, and the line carries the file's data hash. The same rule
    holds for the committed golden vectors (tests/golden/README.md): their sha256 (first 12 hex digits)."""
    from mrm.sim import vectors
    text = (ROOT / "docs" / "CHANGELOG.md").read_text()
    missing = []
    for p in sorted((ROOT / "tests" / "golden").glob("*.npz")):
        sha = golden_cli.load_golden(p).meta["data_sha256"][:12]
        if sha not in text:
            missing.append(f"{p.name} (data sha256 {sha})")
    for p in sorted((ROOT / "tests" / "golden" / "vectors").glob("*.npz")):
        sha = vectors.load_vectors(p).sha256()[:12]
        if sha not in text:
            missing.append(f"vectors/{p.name} (sha256 {sha})")
    assert not missing, ("docs/CHANGELOG.md has no line for " + ", ".join(missing)
                         + "; add the line that `mrm golden --update` printed (golden vectors: tests/golden/README.md)")


def test_every_golden_file_belongs_to_a_case():
    """Every golden file has a case, every live case has a file, and the three required cases are live."""
    names = {p.stem for p in (ROOT / "tests" / "golden").glob("*.npz")}
    assert names <= set(golden_cli.GOLDEN_CASES), names - set(golden_cli.GOLDEN_CASES)
    live = {n for n, c in golden_cli.GOLDEN_CASES.items() if not c.pending}
    assert live <= names, live - names
    # the required cases: static lock, the fast aggressor and the compressed soak
    assert {"g26__static", "g26__a1_slide10_aggressor", "g26__soak_compressed"} <= live


def test_golden_file_content():
    """The static and A1 files hold the current format, the case definition, the three runs and the environment."""
    for name in ("g26__static", "g26__a1_slide10_aggressor"):
        g = golden_cli.load_golden(golden_cli.golden_path(name))
        assert g.meta["format"] == golden_cli.GOLDEN_FORMAT
        assert g.meta["case"] == json.loads(json.dumps(golden_cli.GOLDEN_CASES[name].definition()))
        labels = {k.split("/")[0] for k in g.arrays}
        assert labels == {"open_loop", "pi_lock_c", "fxp_pi"}
        assert "pi_lock_c/max_abs_detuning_pm" in g.arrays and "fxp_pi/world/excursion_count" in g.arrays
        assert all(g.arrays[k].shape[:2] == (1, 8) for k in g.arrays if k.endswith("rms_detuning_pm"))
        assert g.meta["runs"]["pi_lock_c"]["controller_path"] == "compiled"
        assert set(g.meta["environment"]) >= {"numpy", "numba", "llvmlite", "python"}
        assert not any("@" in k for k in g.arrays)


def test_soak_file_content_and_case():
    """The compressed-soak golden: pi_lock_c only (the controller that the A4 convergence experiment certified at
    compression 100), one soak cycle at c = 100 from the config's scenario section."""
    from mrm.scenarios import library
    c = golden_cli.GOLDEN_CASES["g26__soak_compressed"]
    assert c.duration_s == library.SOAK_CYCLE_S / 100.0 and c.scenario == golden_cli.CONFIG_SCENARIO
    assert "scenario.time_compression=100.0" in c.overrides and "sim.substep_policy=auto" in c.overrides
    assert c.run_kwargs["score_from_s"] == 0.005
    g = golden_cli.load_golden(golden_cli.golden_path("g26__soak_compressed"))
    assert g.meta["case"] == json.loads(json.dumps(c.definition()))
    assert {k.split("/")[0] for k in g.arrays} == {"pi_lock_c"}
    assert g.meta["runs"]["pi_lock_c"]["controller_path"] == "compiled"
    # The c = 100 run held its order and line over the whole 75 K swing (docs/handoff/experiments.md, "Acceptance
    # table"): no change of the order nearest the declared line (a bound on max |detuning| cannot see a hop, since
    # |d| <= FSR/2 by construction) and no wrong-line event.
    assert int(g.arrays["pi_lock_c/order_changes"].sum()) == 0
    assert int(g.arrays["pi_lock_c/world/wrong_line_events"].sum()) == 0


def test_a1_case_uses_the_library_period():
    """The A1 case runs one aggressor period from phase P/2 (the cool minimum)."""
    from mrm.scenarios import library
    c = golden_cli.GOLDEN_CASES["g26__a1_slide10_aggressor"]
    assert c.duration_s == library.A1_PERIOD_S and c.scenario_params["phase_s"] == library.A1_PERIOD_S / 2


def _perturbed(g, key, f):
    """A copy of golden data ``g`` with array ``key`` replaced by f(copy of it)."""
    arrays = dict(g.arrays)
    arrays[key] = f(arrays[key].copy())
    return golden_cli.GoldenData(arrays=arrays, text=dict(g.text), meta=dict(g.meta))


def test_compare_rule():
    """Integers exactly; floats to rtol 1e-9 (+ 1e-12 of the metric's scale); NaN == NaN; text exactly."""
    g = golden_cli.load_golden(golden_cli.golden_path("g26__static"))
    assert golden_cli.compare_golden(g, g) == []
    k = "pi_lock_c/rms_detuning_pm"
    assert golden_cli.compare_golden(g, _perturbed(g, k, lambda a: a * (1 + 1e-12))) == []
    d = golden_cli.compare_golden(g, _perturbed(g, k, lambda a: a * (1 + 1e-7)))
    assert d and d[0].startswith(k)
    d = golden_cli.compare_golden(g, _perturbed(g, "fxp_pi/code_max_used", lambda a: a + 1))
    assert d and d[0].startswith("fxp_pi/code_max_used")
    nan_key = next((k for k, a in g.arrays.items() if a.dtype.kind == "f" and np.isnan(a).any()), None)
    assert nan_key                                   # first_lock_s of the open loop: NaN (never locked)
    assert golden_cli.compare_golden(g, _perturbed(g, nan_key, lambda a: a)) == []
    d = golden_cli.compare_golden(g, _perturbed(g, nan_key, lambda a: np.where(np.isnan(a), 1.0, a)))
    assert d and d[0].startswith(nan_key)            # NaN -> a number fails
    t = dict(g.text)
    t["pi_lock_c/settings/lock_reference"] = "oracle"
    d = golden_cli.compare_golden(g, golden_cli.GoldenData(arrays=g.arrays, text=t, meta=g.meta))
    assert d and "lock_reference" in d[0]
    d = golden_cli.compare_golden(g, golden_cli.GoldenData(arrays={k2: v for k2, v in g.arrays.items() if k2 != k},
                                                           text=g.text, meta=g.meta))
    assert d and "missing" in d[0]


def test_non_finite_values_compare_exactly():
    """+-inf (the histogram's outer edges) and NaN match only themselves: a tolerance scaled by an infinity would
    otherwise accept finite -> inf and -inf -> +inf."""
    def data(x):
        return golden_cli.GoldenData(arrays={"m": np.array(x, dtype=np.float64)}, text={}, meta={})
    ref = data([-np.inf, 1.0, np.inf, np.nan])
    assert golden_cli.compare_golden(ref, data([-np.inf, 1.0, np.inf, np.nan])) == []
    assert golden_cli.compare_golden(ref, data([-np.inf, 1.0 + 1e-12, np.inf, np.nan])) == []
    for bad in ([np.inf, 1.0, np.inf, np.nan], [-np.inf, np.inf, np.inf, np.nan], [-np.inf, 1.0, 5.0, np.nan],
                [-np.inf, 1.0, np.inf, np.inf], [-np.inf, np.nan, np.inf, np.nan]):
        assert golden_cli.compare_golden(ref, data(bad)), bad
    g = golden_cli.load_golden(golden_cli.golden_path("g26__static"))
    k = "fxp_pi/detuning_histogram/edges_pm"
    assert np.isinf(g.arrays[k]).any()
    assert golden_cli.compare_golden(g, _perturbed(g, k, lambda a: np.where(np.isinf(a), -a, a)))


def test_edited_file_is_refused(tmp_path):
    """A file edited by hand fails its data_sha256 check on load."""
    g = golden_cli.load_golden(golden_cli.golden_path("g26__static"))
    p = golden_cli.save_golden(tmp_path / "x.npz", g, reason="test")
    assert golden_cli.load_golden(p).meta["reason"] == "test"
    with np.load(p) as z:
        d = {k: np.array(z[k]) for k in z.files}
    d["m:fxp_pi/rms_detuning_pm"] = d["m:fxp_pi/rms_detuning_pm"] + 1e-6
    np.savez(tmp_path / "y.npz", **d)
    with pytest.raises(ValueError, match="data_sha256"):
        golden_cli.load_golden(tmp_path / "y.npz")


def test_update_needs_a_reason_and_reports_changelog_lines(tmp_path, monkeypatch):
    """--update needs a reason, refuses a pending case and returns one CHANGELOG line per written file."""
    with pytest.raises(ValueError, match="reason"):
        golden_cli.update_golden(["g26__static"], "  ", tmp_path)
    pending = golden_cli.GoldenCase(name="x__pending", preset="g26", scenario="static", duration_s=1e-3, runs=(),
                              pending="a test case that cannot run yet")
    monkeypatch.setitem(golden_cli.GOLDEN_CASES, "x__pending", pending)
    with pytest.raises(RuntimeError, match="pending"):
        golden_cli.update_golden(["x__pending"], "x", tmp_path)
    # an update into an empty directory writes the file and returns its CHANGELOG line
    fake = golden_cli.GoldenData(arrays={"a/x": np.zeros((1, 8))}, text={}, meta={"data_sha256": "ab" * 32})
    monkeypatch.setattr(golden_cli, "run_golden_case", lambda case, progress=None: fake)
    monkeypatch.setattr(golden_cli, "save_golden", lambda p, d, reason: Path(p))
    lines = golden_cli.update_golden(["g26__static"], "because", tmp_path)
    assert lines == ["- golden: tests/golden/g26__static.npz updated (data sha256 abababababab): because"]


def test_update_rewrites_a_changed_definition_without_a_changelog_line(tmp_path, monkeypatch):
    """Same metrics, new case definition: --update rewrites the file's metadata (else test_golden_file_content could
    never be fixed through `mrm golden --update`) and asks for no new CHANGELOG line (the data hash is unchanged)."""
    g = golden_cli.load_golden(golden_cli.golden_path("g26__static"))
    stale = dict(g.meta, case=dict(g.meta["case"], description="an older description"))
    golden_cli.save_golden(tmp_path / "g26__static.npz", golden_cli.GoldenData(g.arrays, g.text, stale), reason="first")
    fresh = golden_cli.GoldenData(g.arrays, g.text, dict(g.meta))
    monkeypatch.setattr(golden_cli, "run_golden_case", lambda case, progress=None: fresh)
    assert golden_cli.update_golden(["g26__static"], "new description", tmp_path) == []
    back = golden_cli.load_golden(tmp_path / "g26__static.npz")
    assert back.meta["case"] == g.meta["case"] and back.meta["data_sha256"] == g.meta["data_sha256"]
    assert golden_cli.update_golden(["g26__static"], "again", tmp_path) == []          # now fully unchanged


def test_cli_lists_and_checks(capsys, tmp_path):
    """``mrm golden --list`` lists the cases; --update without --reason exits 2; a missing file exits 1."""
    from mrm import cli
    assert cli.main(["golden", "--list"]) == 0
    out = capsys.readouterr().out
    assert "g26__static" in out and "g26__soak_compressed" in out
    assert cli.main(["golden", "--update"]) == 2                             # no --reason
    assert cli.main(["golden", "--case", "g26__static", "--dir", str(tmp_path)]) == 1   # missing file
    assert "missing" in capsys.readouterr().out
