"""Run mrm experiments with the TLC and keep its telemetry (docs/TLC_PORT_SPEC_AND_PLAN.md 2.4).

    .venv/bin/python tools/tlc_run.py experiments/tlc/steady_tlc.yaml [more.yaml ...] [--runs-dir runs]

Same as ``mrm run``, plus: mrm's runner never records controller telemetry, so this script keeps every controller
the experiment creates and saves each TLC instance's log into the run directory. The last instance is the main
run (``tlc_telemetry.npz``); earlier ones (the self-reference pre-run) are ``tlc_telemetry_pre<i>.npz``. Prints one
run directory per experiment on stdout, for tools/tlc_report.py.

Corner runs (``alts``) choose what the controllers know with ``knowledge``. ``corner`` (the default) is mrm's own
rule - the plant and every controller's nominal design follow the corner. ``nominal`` is the model-mismatch
variant - the plant follows the corner, and every controller keeps the experiment's own design without the corner
alternatives (for the TLC - the G26 priors).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from mrm.control.tlc.controller import TlcController
from mrm.sim import experiment as E

KNOWLEDGE_CORNER = "corner"     # plant and controller knowledge follow the corner (mrm's rule)
KNOWLEDGE_NOMINAL = "nominal"   # plant follows the corner, controllers keep the design without the corner
KNOWLEDGE_CHOICES = (KNOWLEDGE_CORNER, KNOWLEDGE_NOMINAL)


# the design the controllers receive as spec.nominal, or None when it is the plant's own design (mrm's rule)
def knowledge_config(path: str,                        # experiment file
                     overrides: list[str] | None,      # the run's experiment overrides
                     knowledge: str):                  # one of KNOWLEDGE_CHOICES
    # return = the experiment Config with the overrides and without the corner alternatives - Config or None
    if knowledge not in KNOWLEDGE_CHOICES:
        raise ValueError(f"knowledge must be one of {KNOWLEDGE_CHOICES}, got {knowledge!r}")
    if knowledge == KNOWLEDGE_CORNER:
        return None
    return E.load(path, overrides=overrides or None).cfg


# make a controller read ``nominal_cfg`` as spec.nominal at every reset; every other spec field stays the plant's
def with_nominal_knowledge(ctrl,                       # a controller (has reset(spec))
                           nominal_cfg):               # the design the controller may know - Config
    # return = the same controller object, its reset wrapped
    # the runner calls ctrl.reset(spec) at the start and after a controller-reset fault. A wrapper object would hide
    # the CompiledController type that the runner checks, so the bound method is replaced on the instance.
    reset = ctrl.reset

    def reset_with_nominal_knowledge(spec):
        reset(dataclasses.replace(spec, nominal=nominal_cfg))   # ControllerSpec filters it by spec.knowledge again

    ctrl.reset = reset_with_nominal_knowledge
    return ctrl


def run_with_telemetry(path: str, runs_dir: str | None = None, overrides: list[str] | None = None,
                       alts: dict | None = None, label: dict | None = None,
                       knowledge: str = KNOWLEDGE_CORNER) -> Path:
    """Run one experiment; save every TLC instance's telemetry and ``tlc_run.json`` (overrides, alts, label, and
    knowledge when it is not the default) into the run directory. ``knowledge`` as in the module docstring."""
    nominal_cfg = knowledge_config(path, overrides, knowledge)
    created = []
    make = E._make_controller

    def keep(entry):
        ctrl = make(entry)
        if ctrl is not None and nominal_cfg is not None:
            with_nominal_knowledge(ctrl, nominal_cfg)
        created.append(ctrl)
        return ctrl

    E._make_controller = keep            # the experiment module builds every controller through this function
    try:
        res = E.run(path, runs_dir=runs_dir, overrides=overrides or None, alts=alts or None)
    finally:
        E._make_controller = make
    if res.run_dir is None:
        raise RuntimeError(f"{path}: the experiment wrote no run directory")
    tlcs = [c for c in created if isinstance(c, TlcController)]
    for i, ctrl in enumerate(tlcs):
        name = "tlc_telemetry.npz" if i == len(tlcs) - 1 else f"tlc_telemetry_pre{i}.npz"
        ctrl.telemetry_log.save(Path(res.run_dir) / name)
    info = {"experiment": path, "overrides": overrides or [], "alts": alts or {}, **(label or {})}
    if knowledge != KNOWLEDGE_CORNER:
        info["knowledge"] = knowledge
    (Path(res.run_dir) / "tlc_run.json").write_text(json.dumps(info, indent=2))
    return Path(res.run_dir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="PATH=VALUE",
                    help="experiment override, e.g. --set controller.name=pi_lock_c --set 'controller.params={}'")
    args = ap.parse_args(argv)
    for path in args.experiments:
        print(run_with_telemetry(path, args.runs_dir, args.set), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
