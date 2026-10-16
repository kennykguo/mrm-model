"""``mrm`` command-line entry point, built on a subcommand registry (07 P1-17, P2-12).

Each command group is registered with :func:`register`:

    register(name, help, setup(parser), run(args) -> int)

Providers are modules that expose ``register_cli(register)``. They are listed in :data:`PROVIDERS` and imported
lazily; a provider whose module does not exist yet is skipped, so later agents add ``run``, ``sweep``, ``mc``,
``compare``, ``robust``, ``acceptance``, ``rerun``, ``report`` and ``golden`` by appending one line to
:data:`PROVIDERS` (or by calling :func:`register` before :func:`main`). Two providers that register the same name
are an error.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from typing import Callable, Sequence

__all__ = ["Command", "COMMANDS", "PROVIDERS", "register", "build_parser", "main"]


@dataclass(frozen=True)
class Command:
    """One top-level subcommand (or command group)."""
    name: str
    help: str
    setup: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], int]


COMMANDS: dict[str, Command] = {}

#: ``module:function`` providers, imported lazily; missing modules are skipped (they belong to later agents)
PROVIDERS: list[str] = [
    "mrm.params.table:register_cli",       # mrm params show | table | explain | diff | validate
    "mrm.sim.cli:register_cli",            # mrm run | sweep | mc | compare | robust | acceptance | rerun (planned)
    "mrm.validation.cli:register_cli",     # mrm report (planned)
    "mrm.control.cli:register_cli",        # controller utilities (planned)
    "mrm.calibration.cli:register_cli",    # mrm calibrate
]
_LOADED = False


def register(name: str, help: str, setup: Callable[[argparse.ArgumentParser], None],
             run: Callable[[argparse.Namespace], int]) -> None:
    """Register a top-level command. ``setup`` adds its arguments; ``run`` returns the exit code."""
    if name in COMMANDS and COMMANDS[name].run is not run:
        raise ValueError(f"mrm command {name!r} is registered twice")
    COMMANDS[name] = Command(name, help, setup, run)


def _load_providers() -> None:
    global _LOADED
    if _LOADED:
        return
    for spec in PROVIDERS:
        mod_name, _, fn_name = spec.partition(":")
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as e:
            if e.name and (mod_name == e.name or mod_name.startswith(e.name + ".")):
                continue                     # provider not written yet
            raise
        getattr(mod, fn_name or "register_cli")(register)
    _LOADED = True


def build_parser() -> argparse.ArgumentParser:
    """The argparse parser with every registered command."""
    _load_providers()
    ap = argparse.ArgumentParser(prog="mrm", description="Microring modulator plant model (mrm)")
    sub = ap.add_subparsers(dest="command", required=True)
    for name in sorted(COMMANDS):
        c = COMMANDS[name]
        sp = sub.add_parser(name, help=c.help, description=c.help)
        c.setup(sp)
        sp.set_defaults(_run=c.run)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``mrm`` console script."""
    ap = build_parser()
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    return int(args._run(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
