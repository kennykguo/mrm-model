# exercise the project readability rules with small, isolated examples.
from pathlib import Path
import subprocess
import sys

import pytest

from linting.stage_2_structure import check
from linting.stage_2_structure.check import CATEGORIES, TARGETS, check_source

SCRIPT = Path(check.__file__)


EXAMPLE = """\
# calculate the settled temperature rise produced by a steady heater power.
def physics_temperature_rise(
    power_w: float,  # power_w = P (W) - steady heater power - type float scalar
    resistance: float,  # resistance = R (K/W) - thermal resistance to ambient - type float scalar
) -> (
    float  # return = x (K) - settled rise above ambient - type float scalar
):
    # steady heat flow gives a temperature rise of power times thermal resistance.
    return power_w * resistance
"""


@pytest.mark.parametrize("path", TARGETS, ids=lambda path: path.stem)
def test_physics_module_meets_readability_contract(path):
    assert check_source(path.read_text()) == []


@pytest.mark.parametrize(("old", "new", "expected"), [
    ("physics_temperature_rise", "temperature_rise", "name"),
    ("physics_temperature_rise", "optical_temperature_rise", "name"),
    ("physics_temperature_rise", "helper_solve_temperature", "category"),
    ("physics_temperature_rise", "physics__temperature_rise", "name"),
    ("physics_temperature_rise", "__physics_temperature_rise", "name"),
    ("physics_temperature_rise", "physics_Temperature_rise", "name"),
    (EXAMPLE.splitlines()[0], "# axes - P = power, R = resistance.", "purpose"),
    ("power_w = P (W)", "power_w = P", "contract"),
    ("return = x (K)", "return = x (K) defined above", "local"),
    ("type float scalar", "type float64 ndarray", "shape"),
    ("power_w: float", "power_w", "type"),
    ("power_w: float", "power_w: int", "storage"),
    (EXAMPLE.splitlines()[-2], "    # x", "math"),
    ("return power_w * resistance", "def hidden():\n        return power_w\n    return hidden()", "nested"),
    ("return power_w * resistance", "return (lambda x: x)(power_w)", "nested"),
    ("return power_w * resistance", "return power_w, resistance", "return"),
])
def test_checker_reports_specific_problem(old, new, expected):
    findings = check_source(EXAMPLE.replace(old, new))
    assert expected in {finding.rule for finding in findings}


def test_each_alternative_and_early_exit_needs_its_own_explanation():
    source = EXAMPLE[:EXAMPLE.index("    # steady heat")] + """\
    # zero applied power produces no temperature rise.
    if power_w == 0:
        return 0.0
    else:
        watts = power_w
        if resistance < 0:
            return 0.0
    # multiply steady heat flow by the resistance to ambient.
    return watts * resistance
"""
    findings = check_source(source)
    lines = source.splitlines()
    assert any(item.rule == "branch" and "watts = power_w" in lines[item.line - 1] for item in findings)
    assert any(item.rule == "path" and "if resistance" in lines[item.line - 1] for item in findings)


def test_a_header_cannot_cover_a_long_undocumented_body():
    lines = EXAMPLE.splitlines()
    source = "\n".join(lines[:-2] + ["    watts = power_w"] * 8 + lines[-2:])
    assert "block" in {item.rule for item in check_source(source)}


def test_cli_exit_code_and_source_location(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(EXAMPLE)
    assert subprocess.run([sys.executable, SCRIPT, sample], capture_output=True).returncode == 0
    sample.write_text(EXAMPLE.replace("physics_temperature_rise", "temperature_rise"))
    result = subprocess.run([sys.executable, SCRIPT, sample], capture_output=True, text=True)
    assert result.returncode == 1
    assert f"{sample}:2: name" in result.stdout


def test_a_symbolic_axis_needs_a_local_definition():
    source = EXAMPLE.replace("float", "Float64Array").replace("type Float64Array scalar", "type float64 ndarray, shape (N,)")
    assert "axis" in {finding.rule for finding in check_source(source)}
    assert not check_source(source.replace("def physics_temperature_rise", "# axes - N = physical rings.\ndef physics_temperature_rise"))


def test_math_in_a_separate_paragraph_needs_a_local_explanation():
    source = EXAMPLE.replace("    return power_w * resistance",
                             "    watts = power_w\n\n    return watts * resistance")
    assert "math" in {finding.rule for finding in check_source(source)}


def test_conditional_expression_and_delegated_return_need_explanations():
    body_start = EXAMPLE.index("    # steady heat")
    for body, rule in [("return power_w if resistance else 0", "choice"),
                       ("return solve_temperature(power_w, resistance)", "delegation")]:
        assert rule in {finding.rule for finding in check_source(EXAMPLE[:body_start] + "    " + body)}


@pytest.mark.parametrize("category", CATEGORIES)
def test_name_carries_the_category_without_a_comment(category):
    source = EXAMPLE.replace("physics_temperature_rise", f"_{category}_temperature_rise")
    assert check_source(source) == []


def test_module_lambda_cannot_bypass_the_function_rules():
    findings = check_source("physics_rise = lambda power, resistance: power * resistance")
    assert [item.rule for item in findings] == ["nested"]


def test_python_hook_can_bind_to_a_categorized_method():
    from mrm.plant import thermal

    assert thermal.LayoutSites.__post_init__ is thermal.LayoutSites.validation_check_layout_sites
    assert thermal.LayoutSites.line is thermal.LayoutSites.assembly_straight_ring_layout
    layout = thermal.LayoutSites.assembly_straight_ring_layout(2, 30.0)
    assert layout.helper_ring_count == layout.n_rings == 2


def test_old_name_and_category_name_share_compiled_implementation():
    from mrm.plant import optics

    assert optics.tcmt_rates is optics.physics_energy_decay_rates_from_ring
    assert optics._exact_line is optics.solver_exact_ring_fields_and_energy


@pytest.mark.parametrize("arg", ["missing.py", "folder", "input.txt"])
def test_cli_requires_explicit_python_files(tmp_path, arg):
    (tmp_path / "folder").mkdir()
    (tmp_path / "input.txt").write_text(EXAMPLE)
    result = subprocess.run([sys.executable, SCRIPT, arg], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2
    assert "expected an existing Python source file" in result.stderr


def test_default_scope_is_independent_of_working_directory(tmp_path):
    result = subprocess.run([sys.executable, SCRIPT], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout
    assert result.stdout == "0 readability findings\n"
