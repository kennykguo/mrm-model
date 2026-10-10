"""``mrm run | sweep | mc | compare | robust | acceptance | rerun`` (07 P1-17, P2-12, P0-10 ``--dry-run``).

Registered with :func:`mrm.cli.register` by :func:`register_cli` (mrm/cli.py lists this provider). Owner:
experiments (docs/handoff/experiments.md).

Usage::

    mrm run experiments/examples/a1_pi_lock.yaml [--set a.b=c ...] [--alt path=name ...] [--runs-dir runs]
    mrm run exp.yaml --dry-run          # B, step counts, estimated RAM and wall time (nothing is run)
    mrm mc exp.yaml --set experiment.n_mc=64      # sweep | mc | compare | robust force experiment.kind
    mrm acceptance experiments/acceptance/a1_aggressor.yaml ...  # or --all: every experiments/acceptance/*.yaml
    mrm rerun runs/20261010T120000Z_1a2b3c4d     # reproduce a run and compare it bit for bit

``mrm run`` runs the kind the file declares (any of run, sweep, mc, compare, robust, acceptance, quasi_static).
Exit codes: 0 done (and every criterion passed); 1 a criterion failed, a compressed run was refused, a rerun was not
bit-identical, or an error; 2 command-line usage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["register_cli", "format_estimate"]

_KIND_COMMANDS = {
    "run": (None, "run an experiment file (the kind it declares: run, sweep, mc, compare, robust, acceptance, "
                  "quasi_static) and write runs/<UTC>_<hash8>/"),
    "sweep": ("sweep", "run an experiment as a sweep over experiment.grid"),
    "mc": ("mc", "run an experiment as a Monte Carlo over experiment.n_mc worlds (metrics only)"),
    "compare": ("compare", "compare controllers: nominal, named corners, Latin-hypercube worlds, rank stability "
                           "(07 4.8)"),
    "robust": ("robust", "one controller over the nominal world, the named corners and Latin-hypercube worlds"),
}


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE",
                   help="override a config key or an experiment option (repeatable)")
    p.add_argument("--alt", dest="alts", action="append", default=[], metavar="PATH=NAME",
                   help="select an alternative of a leaf or pin an uncertainty entry (repeatable)")
    p.add_argument("--runs-dir", default=None, help="where run directories go (default $MRM_RUNS_DIR or ./runs)")
    p.add_argument("--threads", type=int, default=None, help="kernel worker threads (experiment.n_threads)")
    p.add_argument("--no-write", action="store_true", help="do not write a run directory")
    p.add_argument("--quiet", action="store_true", help="no progress lines")
    p.add_argument("--json", action="store_true", help="print the summary as JSON instead of the report")


def _setup_kind(p: argparse.ArgumentParser) -> None:
    p.add_argument("experiment", help="experiment YAML file (or preset name)")
    _common(p)
    p.add_argument("--dry-run", action="store_true",
                   help="print B, step counts, estimated RAM and wall time of every case; run nothing")


def _setup_acceptance(p: argparse.ArgumentParser) -> None:
    p.add_argument("experiments", nargs="*", help="acceptance experiment files")
    p.add_argument("--all", action="store_true", help="every experiments/acceptance/*.yaml of the repository")
    _common(p)
    p.add_argument("--dry-run", action="store_true", help="estimate only")


def _setup_rerun(p: argparse.ArgumentParser) -> None:
    p.add_argument("run_dir", help="a run directory runs/<UTC>_<hash8>")
    p.add_argument("--runs-dir", default=None, help="where the new run directory goes (default: beside the old one)")
    p.add_argument("--no-compare", action="store_true", help="do not compare with the original run")
    p.add_argument("--allow-changed-config", action="store_true",
                   help="run even when the experiment hash differs from the recorded one (presets or artefacts "
                        "changed); the comparison then reports the differences")
    p.add_argument("--quiet", action="store_true", help="no progress lines")


def format_estimate(est: dict) -> str:
    """Human-readable ``--dry-run`` output (07 P0-10)."""
    lines = [f"dry run: {est['experiment']} (kind {est['kind']}, hash {est['experiment_hash'][:8]}), dt "
             f"{est['dt_s']:g} s, substep policy {est['substep_policy']}, controller rate "
             f"{est['controller_rate_hz']} Hz, {est['n_threads']} threads",
             "| case | B | ticks | self-ref ticks | controller (path) | edges | start | records (MB) | RAM (GB) | "
             "wall (s) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in est["cases"]:
        lines.append(f"| {r['case']} | {r['B']} | {r.get('ticks', 0)} | {r.get('pre_ticks', 0)} | "
                     f"{r.get('controller') or '-'} ({r.get('path')}) | {r.get('controller_edges', 0)} | "
                     f"{r.get('start', '-')} | {r.get('record_mb', 0.0):.1f} | {r['ram_gb']:.2f} | "
                     f"{r['wall_s']:.1f} |")
    lines.append(f"total: {est['total_ticks']} ticks ({est['world_ticks']} world-ticks); estimated wall time "
                 f"{est['est_wall_s']:.0f} s ({est['est_wall_s'] / 3600:.2f} h); estimated peak RAM "
                 f"{est['est_peak_ram_gb']:.2f} GB")
    lines.append(est["basis"])
    return "\n".join(lines)


def _argv(args) -> list[str]:
    """The command line as parsed (for the manifest): mrm <command> <files> --set ... --alt ... [flags]."""
    out = ["mrm", str(getattr(args, "command", "?"))]
    for k in ("experiment", "run_dir"):
        if getattr(args, k, None):
            out.append(str(getattr(args, k)))
    out += [str(x) for x in getattr(args, "experiments", []) or []]
    out += [f"--set={o}" for o in getattr(args, "overrides", []) or []]
    out += [f"--alt={a}" for a in getattr(args, "alts", []) or []]
    for flag in ("all", "dry_run", "no_write", "no_compare", "allow_changed_config"):
        if getattr(args, flag, False):
            out.append("--" + flag.replace("_", "-"))
    for opt in ("runs_dir", "threads"):
        if getattr(args, opt, None) is not None:
            out.append(f"--{opt.replace('_', '-')}={getattr(args, opt)}")
    return out


def _options(args) -> dict:
    return {"n_threads": args.threads} if getattr(args, "threads", None) else {}


def _log(args):
    if getattr(args, "quiet", False):
        return lambda s: None
    return lambda s: print(s, file=sys.stderr, flush=True)


def _report(res, args) -> None:
    if args.json:
        from mrm.sim.manifest import to_jsonable
        print(json.dumps(to_jsonable({"status": res.status, "passed": res.passed, "run_dir": res.run_dir,
                                      "summary": res.summary}), indent=1))
    else:
        print(res.report, end="")
        if res.run_dir:
            print(f"run directory: {res.run_dir}")


def _make_kind_runner(kind: str | None):
    def _run(args) -> int:
        from mrm.sim import experiment as E
        try:
            _alts(args.alts)
        except _UsageError as e:
            print(f"usage error: {e}", file=sys.stderr)
            return 2
        try:
            if args.dry_run:
                print(format_estimate(E.estimate(args.experiment, overrides=args.overrides or None,
                                                 alts=_alts(args.alts), kind=kind, options=_options(args))))
                return 0
            res = E.run(args.experiment, overrides=args.overrides or None, alts=_alts(args.alts),
                        options=_options(args), kind=kind, runs_dir=args.runs_dir, write=not args.no_write,
                        argv=_argv(args), log=_log(args))
        except (E.ExperimentError, ValueError, KeyError, TypeError, NotImplementedError, FileNotFoundError) as e:
            print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
            return 1
        _report(res, args)
        return 0 if res.status == "ok" else 1
    return _run


class _UsageError(Exception):
    """A command-line usage error (exit code 2)."""


def _alts(items):
    if not items:
        return None
    out = {}
    for a in items:
        if "=" not in a:
            raise _UsageError(f"--alt {a!r}: expected PATH=NAME")
        k, v = a.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _run_acceptance(args) -> int:
    from mrm.sim import experiment as E
    from mrm.sim.manifest import REPO_ROOT
    files = list(args.experiments)
    if args.all:
        files += sorted(str(p) for p in (REPO_ROOT / "experiments" / "acceptance").glob("*.yaml"))
    if not files:
        print("error: give acceptance files or --all", file=sys.stderr)
        return 2
    try:
        _alts(args.alts)
    except _UsageError as e:
        print(f"usage error: {e}", file=sys.stderr)
        return 2
    rows = []
    worst = 0
    for f in files:
        try:
            if args.dry_run:
                print(format_estimate(E.estimate(f, overrides=args.overrides or None, alts=_alts(args.alts),
                                                 options=_options(args))))
                continue
            res = E.run(f, overrides=args.overrides or None, alts=_alts(args.alts), options=_options(args),
                        runs_dir=args.runs_dir, write=not args.no_write,
                        argv=_argv(args), log=_log(args))
        except (E.ExperimentError, ValueError, KeyError, TypeError, NotImplementedError, FileNotFoundError) as e:
            print(f"error: {f}: {type(e).__name__}: {e}", file=sys.stderr)
            rows.append((Path(f).stem, "ERROR", str(e)[:80], None))
            worst = 1
            continue
        if not args.quiet:
            _report(res, args)
        failed = [c["id"] for c in (res.summary.get("criteria") or []) if not c["pass"]]
        rows.append((res.name, "PASS" if res.passed else ("FAIL" if res.passed is False else res.status.upper()),
                     ", ".join(failed), res.run_dir))
        if res.status != "ok":
            worst = 1
    if rows:
        print("\n| acceptance experiment | result | failed criteria | run directory |")
        print("|---|---|---|---|")
        for name, r, f, d in rows:
            print(f"| {name} | {r} | {f or '-'} | {d or '-'} |")
    return worst


def _run_rerun(args) -> int:
    from mrm.sim import experiment as E
    d = Path(args.run_dir)
    try:
        res, cmp = E.rerun(d, runs_dir=args.runs_dir if args.runs_dir else d.parent, compare=not args.no_compare,
                           log=_log(args), allow_changed_config=args.allow_changed_config)
    except (E.ExperimentError, ValueError, KeyError, TypeError, NotImplementedError, FileNotFoundError) as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"rerun of {d.name}: {res.run_dir}")
    if cmp is None:
        return 0
    if cmp["identical"]:
        print(f"bit-for-bit identical ({cmp['n_compared']} arrays of trace.h5)")
        return 0
    print(f"NOT identical: {len(cmp['differing'])} of {cmp['n_compared']} arrays differ; first: "
          f"{cmp['differing'][:10]}; only in the original: {cmp['only_in_a'][:5]}; only in the rerun: "
          f"{cmp['only_in_b'][:5]}")
    return 1


#: one runner per command, made once (mrm.cli.register refuses a second, different run function for a name)
_RUNNERS = {name: _make_kind_runner(kind) for name, (kind, _) in _KIND_COMMANDS.items()}


def register_cli(register) -> None:
    """Register run, sweep, mc, compare, robust, acceptance and rerun with :func:`mrm.cli.register`."""
    for name, (kind, help_) in _KIND_COMMANDS.items():
        register(name, help_, _setup_kind, _RUNNERS[name])
    register("acceptance", "run acceptance experiment files (experiments/acceptance/*.yaml; each runs the kind it "
                           "declares and judges experiment.acceptance) and report pass / fail",
             _setup_acceptance, _run_acceptance)
    register("rerun", "reproduce a run from its directory and compare it bit for bit", _setup_rerun, _run_rerun)
