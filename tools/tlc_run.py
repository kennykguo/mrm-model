"""Run mrm experiments with the TLC and keep its telemetry (docs/TLC_PORT_SPEC_AND_PLAN.md 2.4).

    .venv/bin/python tools/tlc_run.py experiments/tlc/steady_tlc.yaml [more.yaml ...] [--runs-dir runs]

Same as ``mrm run``, plus: mrm's runner never records controller telemetry, so this script keeps every controller
the experiment creates and saves each TLC instance's log into the run directory. The last instance is the main
run (``tlc_telemetry.npz``); earlier ones (the self-reference pre-run) are ``tlc_telemetry_pre<i>.npz``. Prints one
run directory per experiment on stdout, for tools/tlc_report.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mrm.control.tlc.controller import TlcController
from mrm.sim import experiment as E


def run_with_telemetry(path: str, runs_dir: str | None = None) -> Path:
    created = []
    make = E._make_controller

    def keep(entry):
        ctrl = make(entry)
        created.append(ctrl)
        return ctrl

    E._make_controller = keep            # the experiment module builds every controller through this function
    try:
        res = E.run(path, runs_dir=runs_dir)
    finally:
        E._make_controller = make
    if res.run_dir is None:
        raise RuntimeError(f"{path}: the experiment wrote no run directory")
    tlcs = [c for c in created if isinstance(c, TlcController)]
    for i, ctrl in enumerate(tlcs):
        name = "tlc_telemetry.npz" if i == len(tlcs) - 1 else f"tlc_telemetry_pre{i}.npz"
        ctrl.telemetry_log.save(Path(res.run_dir) / name)
    return Path(res.run_dir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--runs-dir", default=None)
    args = ap.parse_args(argv)
    for path in args.experiments:
        print(run_with_telemetry(path, args.runs_dir), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
