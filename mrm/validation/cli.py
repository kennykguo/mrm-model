# register `mrm report` to generate the validation report.
# report.py owns the parser and computation.
# it loads when this command's parser is built.
# plant and engine modules load when their report sections run.
# `--fast` renders cached results.
# `--recompute predictions` refreshes one section.
# `--preset` selects a plant preset.
# `--check` returns failure if the committed report is stale.
# a full recomputation can take over an hour.
from __future__ import annotations

import argparse

__all__ = ["register_cli"]


# add `mrm report` to the `mrm.cli` registry.
def register_cli(register) -> None:
    register("report", "regenerate docs/VALIDATION.md (targets, closure, identifiability, uncertainty, ranking, "
                       "generation mix, textbook)", _report_setup, _report_run)


# add the arguments of `mrm report` to the sub-parser `p`.
def _report_setup(p: argparse.ArgumentParser) -> None:
    from mrm.validation.report import build_parser
    build_parser(p)


# run `mrm report` with the parsed arguments.
# returns the exit code.
def _report_run(a: argparse.Namespace) -> int:
    from mrm.validation.report import run_args
    return run_args(a)
