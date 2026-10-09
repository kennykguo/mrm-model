"""Command-line interface for the same configuration and simulation API."""
import argparse
import json
from pathlib import Path

import yaml

from .config import load
from .controllers import REGISTRY
from .experiments import compare, dry_run, plain, rerun, save
from .plant import simulate


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ringlab")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "inspect"):
        p = sub.add_parser(name)
        p.add_argument("configuration")
        p.add_argument("--set", action="append", default=[])
        if name == "run":
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--output", default="runs")
            p.add_argument("--no-plot", action="store_true")
    p = sub.add_parser("compare")
    p.add_argument("study")
    p.add_argument("--output", default="runs")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("rerun")
    p.add_argument("directory")
    p.add_argument("--output", default="runs")
    p.add_argument("--allow-changed-model", action="store_true")
    sub.add_parser("controllers")
    args = parser.parse_args(argv)
    if args.command == "controllers":
        print("\n".join(sorted(REGISTRY)))
        return 0
    if args.command == "compare":
        result = compare(args.study, args.output, preview=args.dry_run)
    elif args.command == "rerun":
        result = {"directory": str(rerun(args.directory, args.output, allow_changed_model=args.allow_changed_model))}
    else:
        cfg = load(args.configuration, args.set)
        if args.command == "inspect":
            print(yaml.safe_dump(cfg.resolved(), sort_keys=True))
            return 0
        if args.dry_run:
            result = dry_run(cfg)
        else:
            simulation = simulate(cfg)
            result = {"directory": str(save(simulation, args.output, plot=not args.no_plot)),
                      "metrics": simulation.metrics, "diagnostics": simulation.diagnostics}
    print(json.dumps(plain(result), indent=2, allow_nan=False))
    return 0
