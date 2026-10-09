"""Isolation of the held-out targets (07 P0-5; ARCHITECTURE.md accuracy rule 5).

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

from mrm.validation import targets_io as tio


def test_calibration_package_cannot_reach_held_out_targets():
    assert tio.CALIBRATION_DIR.is_dir(), "mrm/calibration must exist"
    violations = tio.held_out_reference_violations(tio.CALIBRATION_DIR)
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
])
def test_scanner_detects_violations(tmp_path: Path, name: str, code: str):
    pkg = tmp_path / "calibration"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / name).write_text(code)
    violations = tio.held_out_reference_violations(pkg)
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
    assert tio.held_out_reference_violations(pkg) == []


def test_header_records_the_frozen_hashes():
    """docs/VALIDATION.md carries the sha256 of the two held-out files, with an ISO date (07 P0-5)."""
    header = tio.read_frozen_header(tio.DOC_VALIDATION)
    for role in tio.FROZEN_ROLES:
        name = f"{role}.yaml"
        assert name in header, f"no frozen row for {name}"
        sha, date = header[name]
        assert re.fullmatch(r"[0-9a-f]{64}", sha)
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)
    errs = tio.header_errors(tio.DOC_VALIDATION)
    assert errs == [], "\n".join(errs)


def test_header_check_detects_a_changed_file(tmp_path: Path):
    """Editing a frozen file (even a comment) breaks the hash check."""
    root = tmp_path / "targets"
    root.mkdir()
    for role in tio.ROLES:
        (root / f"{role}.yaml").write_bytes(tio.targets_path(role).read_bytes())
    doc = tmp_path / "VALIDATION.md"
    doc.write_text("# Validation\n\n" + tio.frozen_header_block("2026-10-09", root) + "\n")
    assert tio.header_errors(doc, root) == []
    with (root / "bounds.yaml").open("a") as fh:
        fh.write("# an innocent-looking comment\n")
    errs = tio.header_errors(doc, root)
    assert len(errs) == 1 and "bounds.yaml" in errs[0]


def test_header_hash_ignores_only_crlf(tmp_path: Path):
    """A Windows checkout (core.autocrlf) must not look like a re-freeze; any other byte change still must."""
    root = tmp_path / "targets"
    root.mkdir()
    for role in tio.ROLES:
        (root / f"{role}.yaml").write_bytes(tio.targets_path(role).read_bytes())
    doc = tmp_path / "VALIDATION.md"
    doc.write_text(tio.frozen_header_block("2026-10-09", root) + "\n")
    raw = (root / "validation.yaml").read_bytes()
    (root / "validation.yaml").write_bytes(raw.replace(b"\n", b"\r\n"))
    assert tio.header_errors(doc, root) == []
    (root / "validation.yaml").write_bytes(raw.replace(b"value: 13.0", b"value: 13.1"))
    assert len(tio.header_errors(doc, root)) == 1


def test_frozen_files_name_their_freeze_date():
    header = tio.read_frozen_header(tio.DOC_VALIDATION)
    for role in tio.FROZEN_ROLES:
        assert str(tio.read_raw(role)["frozen_on"]) == header[f"{role}.yaml"][1]
