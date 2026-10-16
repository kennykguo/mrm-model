"""``mrm calibrate``: run the calibration fits, write the calibrated preset, compare before/after (07 P0-5, 4.3).

Provider for :mod:`mrm.cli` (``register_cli``; add ``"mrm.calibration.cli:register_cli"`` to ``mrm.cli.PROVIDERS``)
and the ``python -m mrm.calibration`` entry point. Run it under ``tools/memguard.py`` (peak RSS ~0.3 GB).

    mrm calibrate                          every group, then g26_calibrated.yaml and the before/after table
    mrm calibrate --groups heater junction  some groups (the earlier groups' leaves come from latest_fit.json)
    mrm calibrate --quick                  coarser profiles (smoke run)
    mrm calibrate --write-preset-only      rewrite the preset from latest_fit.json
    mrm calibrate --before-after-only      only the before/after replica table from latest_fit.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["register_cli", "main", "setup", "run_cmd"]


def setup(ap: argparse.ArgumentParser) -> None:
    from mrm.calibration.pipeline import GROUP_ORDER, RESULTS_PATH
    ap.add_argument("--groups", nargs="*", choices=list(GROUP_ORDER), default=None, help="fit groups (default all)")
    ap.add_argument("--quick", action="store_true", help="coarser profiles and fewer starts")
    ap.add_argument("--out", default=str(RESULTS_PATH), help="fit record (JSON)")
    ap.add_argument("--no-preset", action="store_true", help="do not write g26_calibrated.yaml")
    ap.add_argument("--no-before-after", action="store_true", help="skip the before/after replica table")
    ap.add_argument("--write-preset-only", action="store_true", help="rewrite the preset from --out")
    ap.add_argument("--before-after-only", action="store_true", help="only the before/after table from --out")


def run_cmd(args: argparse.Namespace) -> int:
    from mrm.calibration import pipeline, preset, report
    out = Path(args.out)
    if args.write_preset_only or args.before_after_only:
        res = pipeline.load_results(out)
    else:
        run = pipeline.run(args.groups, quick=args.quick, out=out)
        res = pipeline.load_results(out)
        del run
    if not args.before_after_only and not args.no_preset:
        p = preset.write_preset(res)
        print(f"wrote {p}")
    if not args.write_preset_only and not args.no_before_after:
        print("before/after (replica primary protocols, calibration targets only):")
        ba = report.before_after(res)
        res["before_after"] = ba
        out.write_text(json.dumps(res, indent=1, allow_nan=False, default=str))
        print(report.format_table(ba))
    return 0


def register_cli(register) -> None:
    """Register ``mrm calibrate`` with the mrm CLI (mrm.cli provider protocol)."""
    register("calibrate", "fit the model to the calibration targets (07 P0-5, 4.3) and write g26_calibrated",
             setup, run_cmd)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mrm.calibration", description=__doc__.splitlines()[0])
    setup(ap)
    return run_cmd(ap.parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
