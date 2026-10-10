"""``mrm run | sweep | mc | compare | robust | acceptance | rerun`` (mrm.sim.cli; 07 P1-17, P2-12, P0-10
``--dry-run``)."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from mrm import cli

ROOT = Path(__file__).resolve().parents[2]

EXP = """\
    extends: g26
    meta: {name: cli_exp, kind: experiment}
    controller: {name: pi_lock_c}
    scenario: {name: static}
    experiment:
      kind: run
      duration_s: 0.003
      uncertainty: nominal
      variation: "off"
      start: warm
      score_from_s: 0.001
    """


@pytest.fixture
def exp_file(tmp_path):
    p = tmp_path / "cli_exp.yaml"
    p.write_text(textwrap.dedent(EXP))
    return p


def test_commands_are_registered():
    cli._load_providers()
    for name in ("run", "sweep", "mc", "compare", "robust", "acceptance", "rerun"):
        assert name in cli.COMMANDS
    from mrm.sim import cli as scli
    scli.register_cli(cli.register)                 # registering again with the same functions is allowed


def test_dry_run_prints_the_estimate(exp_file, capsys):
    assert cli.main(["run", str(exp_file), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "dry run: cli_exp" in out and "| main | 1 | 3000 |" in out
    assert "estimated wall time" in out and "estimated peak RAM" in out
    assert "docs/handoff/perf.md section 8" in out


def test_run_then_rerun_is_bit_identical(exp_file, tmp_path, capsys):
    runs = tmp_path / "runs"
    assert cli.main(["run", str(exp_file), "--runs-dir", str(runs), "--quiet",
                     "--set", "controller.params.fraction=0.65"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0].startswith("Performance is conditional")
    (d,) = list(runs.iterdir())
    man = json.loads((d / "manifest.json").read_text())
    assert "controller.params.fraction=0.65" in man["source"]["overrides"]
    assert man["argv"][:2] == ["mrm", "run"]
    assert cli.main(["rerun", str(d), "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "bit-for-bit identical" in out
    assert len(list(runs.iterdir())) == 2                  # the rerun goes beside the original


def test_kind_commands_force_the_kind(exp_file, tmp_path, capsys):
    assert cli.main(["mc", str(exp_file), "--no-write", "--quiet", "--set", "experiment.n_mc=2",
                     "--set", "experiment.uncertainty=sample", "--json"]) == 0
    s = json.loads(capsys.readouterr().out)
    assert s["status"] == "ok" and s["summary"]["quantiles"]["excursion_count"]["n"] == 2


def test_acceptance_exit_codes(exp_file, tmp_path, capsys):
    ok = tmp_path / "ok.yaml"
    ok.write_text(textwrap.dedent(EXP.replace("kind: run", "kind: acceptance")) +
                  "  acceptance: [{id: line, metric: wrong_line_rings, op: '==', value: 0}]\n")
    bad = tmp_path / "bad.yaml"
    bad.write_text(textwrap.dedent(EXP.replace("kind: run", "kind: acceptance")) +
                   "  acceptance: [{id: never, metric: max_abs_detuning_pm, op: '<', value: 0.0}]\n")
    assert cli.main(["acceptance", str(ok), "--no-write", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "| cli_exp | PASS |" in out
    assert cli.main(["acceptance", str(ok), str(bad), "--no-write", "--quiet"]) == 1
    out = capsys.readouterr().out
    assert "| cli_exp | FAIL | never |" in out


def test_errors_exit_1_without_traceback(tmp_path, capsys):
    p = tmp_path / "broken.yaml"
    p.write_text(textwrap.dedent(EXP.replace("start: warm", "start: lukewarm")))
    assert cli.main(["run", str(p), "--no-write"]) == 1
    assert "lukewarm" in capsys.readouterr().err
    assert cli.main(["rerun", str(tmp_path)]) == 1
    assert "not a run directory" in capsys.readouterr().err


def test_usage_errors_exit_2_and_dry_run_honours_threads(exp_file, capsys):
    assert cli.main(["run", str(exp_file), "--alt", "uncertainty.thermo_optic", "--dry-run"]) == 2
    assert "expected PATH=NAME" in capsys.readouterr().err
    assert cli.main(["acceptance", str(exp_file), "--alt", "nonsense", "--dry-run"]) == 2
    capsys.readouterr()
    assert cli.main(["run", str(exp_file), "--dry-run", "--threads", "3"]) == 0
    assert ", 3 threads" in capsys.readouterr().out


def test_acceptance_all_dry_run(capsys):
    assert cli.main(["acceptance", "--all", "--dry-run"]) == 0
    out = capsys.readouterr().out
    for name in ("acc_a1_slide10_aggressor", "acc_a4_soak", "acc_a7_reloma_plane", "acc_b5_cold_start"):
        assert f"dry run: {name}" in out
    assert "convergence_c1" in out                        # the soak's mandatory convergence experiment is costed
