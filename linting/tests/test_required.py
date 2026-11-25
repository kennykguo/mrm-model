from pathlib import Path
import subprocess

import pytest

from linting import required


@pytest.mark.parametrize(("codes", "expected"), [((0, 0), 0), ((1, 0), 1), ((0, 1), 1)])
def test_both_required_checks_run_and_either_can_fail(monkeypatch, codes, expected):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, codes[len(calls) - 1])
    monkeypatch.setattr(required.subprocess, "run", run)
    assert required.main() == expected
    assert len(calls) == 2
    assert "ruff" in calls[0]
    assert Path(calls[1][-1]) == required.HERE / "stage_2_structure" / "check.py"
