# verify the isolated evidence and the physical premise of the semantic calibration.
from pathlib import Path
import subprocess
import sys

import pytest

from linting.structure.check import check_source
from linting.semantics import excerpt

SCRIPT = Path(excerpt.__file__)


def test_each_blind_case_changes_only_the_explanation_before_the_guard():
    baseline = excerpt.SOLVER_PATH.read_text()
    first, original = excerpt.isolated_function(baseline, "solve_ring_temperature")
    guard = original.index("    if g == 0.0:")
    explanations = []
    for case in excerpt.CASES:
        start, lines = excerpt.case_excerpt(case)
        assert start == first
        assert lines[:guard - 1] == original[:guard - 1]
        assert lines[guard:] == original[guard:]
        explanations.append(lines[guard - 1])
        # all three cases satisfy the placement rule at the guard, including the physically wrong one.
        findings = check_source("\n".join(lines))
        assert not any(finding.line == guard + 1 and finding.rule == "path" for finding in findings)
    assert len(set(explanations)) == 3


def test_zero_thermal_gain_allows_positive_absorption_and_nonzero_slope():
    from mrm.plant import selfheat

    sampled = []

    def absorption(temperature, ring, floats, integers):
        sampled.append(temperature)
        return 0.002, 0.0001

    solver = getattr(selfheat.solve_ring_temperature, "py_func", selfheat.solve_ring_temperature)
    result = solver(absorption, 0, (), (), 315.0, 0.0, 330.0, 0.01, 0.0, 8, 1e-8)
    assert result == (315.0, 0.002, 0.0001, 0, selfheat.STATUS_CONVERGED)
    assert sampled == [315.0]


def test_method_excerpt_excludes_class_and_sibling_context():
    source = """class Ring:
    description = 'outside the function'

    # measure the optical power.
    @property
    def power(self):
        return self.field ** 2

    def reset(self):
        pass
"""
    first, lines = excerpt.isolated_function(source, "Ring.power")
    assert first == 4
    assert lines == source.splitlines()[3:7]
    with pytest.raises(ValueError, match="Ring.unknown"):
        excerpt.isolated_function(source, "Ring.unknown")


def test_cli_shows_only_the_selected_case_and_preserves_source_lines():
    result = subprocess.run([sys.executable, SCRIPT, "--case", "C"], check=True, capture_output=True, text=True)
    assert "the ring absorbs no light" in result.stdout
    assert "zero gain makes the optical temperature rise zero" not in result.stdout
    source = excerpt.SOLVER_PATH.read_text().splitlines()
    guard_line = source.index("    if g == 0.0:") + 1
    assert f"{guard_line:5}      if g == 0.0:" in result.stdout


def test_comments_inside_the_previous_function_are_outside_the_evidence():
    source = "def first():\n    pass\n    # reasoning from the sibling body.\n# local purpose.\ndef second():\n    pass\n"
    first, lines = excerpt.isolated_function(source, "second")
    assert first == 4
    assert lines[0] == "# local purpose."
