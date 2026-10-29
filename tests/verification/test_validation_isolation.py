"""Checks that the held-out validation targets stay out of reach of the calibration code.

The rule (docs/ARCHITECTURE.md, accuracy rule 5): calibration may read only targets/calibration.yaml, and the
held-out targets are frozen by SHA-256. The tests check three things:

1. Nothing under mrm/calibration/** may reference validation.yaml, bounds.yaml or docs/VALIDATION.md, or load a
   target role other than "calibration" (static text and AST scan, ``targets_io.held_out_reference_violations``).
2. The scanner is not vacuous: synthetic violations are detected and clean calibration code passes.
3. The sha256 of validation.yaml and bounds.yaml equals the frozen hashes in the header of docs/VALIDATION.md.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

from mrm.validation import targets_io


def test_calibration_package_cannot_reach_held_out_targets():
    assert targets_io.CALIBRATION_DIR.is_dir(), "mrm/calibration must exist"
    violations = targets_io.held_out_reference_violations(targets_io.CALIBRATION_DIR)
    assert violations == [], "\n".join(violations)


@pytest.mark.parametrize("name,code", [
    ("path_string.py", 'P = "mrm/validation/targets/validation.yaml"\n'),
    ("bounds_string.py", "open('bounds.yml')\n"),
    ("report_doc.py", 'DOC = "docs/VALIDATION.md"\n'),
    ("role_literal.py", "from mrm.validation.targets_io import load_targets\nT = load_targets('validation')\n"),
    ("role_keyword.py", "from mrm.validation import targets_io\nT = targets_io.load_targets(role='bounds')\n"),
    ("role_variable.py", "from mrm.validation.targets_io import load_targets\nr = 'cal' + 'ibration'\n"
                         "T = load_targets(r)\n"),
    ("load_all.py", "from mrm.validation import targets_io\nA = targets_io.load_all()\n"),
    ("import_name.py", "from mrm.validation.targets_io import HELD_OUT_ROLES\n"),
    ("targets_dir.py", "from mrm.validation import targets_io\nfiles = list(targets_io.TARGETS_DIR.glob('*'))\n"),
    ("report_import.py", "import mrm.validation.report\n"),
    ("report_from.py", "from mrm.validation import report\n"),
    ("read_raw.py", "from mrm.validation.targets_io import read_raw\nd = read_raw('excluded')\n"),
    ("config.yaml", "targets: mrm/validation/targets/bounds.yaml\n"),
    # routes found by the review of the split: aliases, indirect references, string lookups, star imports
    ("alias.py", "from mrm.validation.targets_io import load_targets as lt\nT = lt('validation')\n"),
    ("alias_relative.py", "from ..validation.targets_io import read_raw as rr\nD = rr('bounds')\n"),
    ("indirect.py", "from mrm.validation import targets_io\nf = targets_io.load_targets\nT = f('calibration')\n"),
    ("indirect_name.py", "from mrm.validation.targets_io import load_targets\nLOADERS = [load_targets]\n"),
    ("getattr.py", "from mrm.validation import targets_io\nA = getattr(targets_io, 'load_all')()\n"),
    ("importlib_report.py", "import importlib\nR = importlib.import_module('mrm.validation.report')\n"),
    ("star.py", "from mrm.validation.targets_io import *\n"),
    ("excluded_loader.py", "from mrm.validation.targets_io import load_excluded\nX = load_excluded()\n"),
    # an alias of a role function re-exported by the package mrm.validation
    ("package_alias.py", "from mrm.validation import load_targets as lt\nT = lt('bounds')\n"),
    ("package_alias_relative.py", "from ..validation import load_targets as lt\nT = lt('validation')\n"),
])
def test_scanner_detects_violations(tmp_path: Path, name: str, code: str):
    pkg = tmp_path / "calibration"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / name).write_text(code)
    violations = targets_io.held_out_reference_violations(pkg)
    assert violations, f"scanner missed a violation in {name}"
    assert all(name in v for v in violations)


def test_scanner_accepts_clean_calibration_code(tmp_path: Path):
    pkg = tmp_path / "calibration"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "sub" / "fit_er.py").write_text(textwrap.dedent('''
        """Fit the extinction mechanism to the calibration targets only."""
        from mrm.validation.targets_io import load_calibration_targets, load_targets, evaluate

        def targets():
            a = load_calibration_targets()
            b = load_targets("calibration")
            c = load_targets(role="calibration")
            return [t for t in a + b + c if t.id.startswith("CAL-2")]
    '''))
    assert targets_io.held_out_reference_violations(pkg) == []


def test_header_records_the_frozen_hashes():
    """docs/VALIDATION.md carries the sha256 of the two held-out files, with an ISO date."""
    header = targets_io.read_frozen_header(targets_io.DOC_VALIDATION)
    for role in targets_io.FROZEN_ROLES:
        name = f"{role}.yaml"
        assert name in header, f"no frozen row for {name}"
        sha, date = header[name]
        assert re.fullmatch(r"[0-9a-f]{64}", sha)
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)
    errs = targets_io.header_errors(targets_io.DOC_VALIDATION)
    assert errs == [], "\n".join(errs)


def test_header_check_detects_a_changed_file(tmp_path: Path):
    """Editing a frozen file (even a comment) breaks the hash check."""
    root = tmp_path / "targets"
    root.mkdir()
    for role in targets_io.ROLES:
        (root / f"{role}.yaml").write_bytes(targets_io.targets_path(role).read_bytes())
    doc = tmp_path / "VALIDATION.md"
    doc.write_text("# Validation\n\n" + targets_io.frozen_header_block("2026-10-09", root) + "\n")
    assert targets_io.header_errors(doc, root) == []
    with (root / "bounds.yaml").open("a") as fh:
        fh.write("# an innocent-looking comment\n")
    errs = targets_io.header_errors(doc, root)
    assert len(errs) == 1 and "bounds.yaml" in errs[0]


def test_header_hash_ignores_only_crlf(tmp_path: Path):
    """A Windows checkout (core.autocrlf) must not look like a re-freeze; any other byte change still must."""
    root = tmp_path / "targets"
    root.mkdir()
    for role in targets_io.ROLES:
        (root / f"{role}.yaml").write_bytes(targets_io.targets_path(role).read_bytes())
    doc = tmp_path / "VALIDATION.md"
    doc.write_text(targets_io.frozen_header_block("2026-10-09", root) + "\n")
    raw = (root / "validation.yaml").read_bytes()
    (root / "validation.yaml").write_bytes(raw.replace(b"\n", b"\r\n"))
    assert targets_io.header_errors(doc, root) == []
    (root / "validation.yaml").write_bytes(raw.replace(b"value: 13.0", b"value: 13.1"))
    assert len(targets_io.header_errors(doc, root)) == 1


def test_frozen_files_name_their_freeze_date():
    header = targets_io.read_frozen_header(targets_io.DOC_VALIDATION)
    for role in targets_io.FROZEN_ROLES:
        assert str(targets_io.read_raw(role)["frozen_on"]) == header[f"{role}.yaml"][1]
