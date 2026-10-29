"""Checks that the "Features not built yet" list of docs/ENGINE.md names no CLI command that exists.

Read this first
---------------
The list once named `mrm report` and the VALIDATION.md generator, which were already built and checked in CI. The
first test reads that list and fails if it names a registered `mrm` command. The second test checks the comments of
the provider table in mrm/cli.py: a provider module that registers commands must not be marked "(planned)". The
third test checks the block diagram source docs/architecture.tex: every module it names exists, and every number
it shows equals the g26 preset. <1 s.
"""
from __future__ import annotations

import re
from pathlib import Path

from mrm import cli

ROOT = Path(__file__).resolve().parents[2]


def _not_built_item() -> str:
    """The 'not built yet' list of docs/ENGINE.md: the paragraph under the heading '### Features not built yet'."""
    text = (ROOT / "docs" / "ENGINE.md").read_text()
    heading = text.index("### Features not built yet")
    start = text.index("\n\n", heading) + 2
    end = text.find("\n\n", start)
    return text[start:end] if end > 0 else text[start:]


def test_engine_md_lists_no_registered_command_as_not_built():
    cli._load_providers()
    names = sorted(cli.COMMANDS)
    assert "report" in names, names
    item = _not_built_item()
    built = [n for n in names if f"`mrm {n}`" in item]
    assert not built, (built, item)
    assert "VALIDATION.md generator" not in item


def test_cli_provider_comments_do_not_call_a_registered_provider_planned():
    """mrm/cli.py PROVIDERS: a provider whose module registers commands is not '(planned)'."""
    src = (ROOT / "mrm" / "cli.py").read_text()
    planned = re.findall(r'"([\w.]+):register_cli",\s*#[^\n]*\(planned\)', src)
    cli._load_providers()
    registered = {c.run.__module__ for c in cli.COMMANDS.values()}
    stale = [m for m in planned if any(r == m or r.startswith(m + ".") for r in registered)]
    assert not stale, stale


def test_architecture_diagram_matches_the_code_and_the_preset():
    """docs/architecture.tex: each "@file <path>" exists, each "@key <key> = <value>" equals the g26 preset, and every
    module path in a block label is listed as an @file (so the list cannot fall behind the labels)."""
    from mrm.params.config import load

    text = (ROOT / "docs" / "architecture.tex").read_text()
    files = re.findall(r"^% @file (\S+)$", text, flags=re.M)
    keys = re.findall(r"^% @key (\S+) = (\S+)$", text, flags=re.M)
    assert files and keys
    missing = [f for f in files if not (ROOT / f).is_file()]
    assert not missing, missing
    cfg = load("g26")
    wrong = [(k, v, cfg.get(k)) for k, v in keys if float(cfg.get(k)) != float(v)]
    assert not wrong, wrong
    body = "\n".join(line for line in text.splitlines() if not line.startswith("%"))
    labelled = set(re.findall(r"mrm/[\w/]+\.py", body))
    assert labelled <= set(files), sorted(labelled - set(files))
    assert (ROOT / "docs" / "architecture.svg").is_file()
