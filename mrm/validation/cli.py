"""``mrm report``: regenerate docs/VALIDATION.md (07 P1-23, 4.9; :mod:`mrm.validation.report`).

Registered with :func:`mrm.cli.register` by :func:`register_cli` (mrm/cli.py lists this provider). The report
module (numpy and the target files only at import; the plant and the engine load when a part is computed) is
imported when the parser is built.

Usage::

    mrm report                              # recompute every part (about 75 min) and write docs/VALIDATION.md
    mrm report --fast                       # re-render from docs/validation/<preset>/report_data.json
    mrm report --recompute predictions      # recompute one part, the others from the cache
    mrm report --preset g26_calibrated      # the same report for another preset
    mrm report --check                      # exit 1 when the committed report is stale
"""
from __future__ import annotations

import argparse

__all__ = ["register_cli"]


def _setup(p: argparse.ArgumentParser) -> None:
    from mrm.validation.report import build_parser
    build_parser(p)


def _run(a: argparse.Namespace) -> int:
    from mrm.validation.report import run_args
    return run_args(a)


def register_cli(register) -> None:
    """Add ``mrm report`` to the :mod:`mrm.cli` registry."""
    register("report", "regenerate docs/VALIDATION.md (targets, closure, identifiability, uncertainty, ranking, "
                       "generation mix, textbook; 07 4.9)", _setup, _run)
