"""docs/ENGINE.md section 10 "Not built yet" against the CLI that exists (engine review 2026-10-16, engine-8: the list
named `mrm report` / the VALIDATION.md generator, which section 14, the CHANGELOG and mrm/validation/cli.py show built
and CI-checked). <1 s.
"""
from __future__ import annotations

import re
from pathlib import Path

from mrm import cli

ROOT = Path(__file__).resolve().parents[2]


def _not_built_item() -> str:
    text = (ROOT / "docs" / "ENGINE.md").read_text()
    i = text.index("- Not built yet:")
    j = text.find("\n- ", i + 1)
    k = text.find("\n\n", i)
    return text[i:min(x for x in (j, k) if x > 0)]


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
