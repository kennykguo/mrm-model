# run calibration fits, write child presets, and compare replica predictions.
# `register_cli` adds `mrm calibrate` to the main CLI.
# `main` provides the same options for `python -m mrm.calibration`.
# `--groups` fits a subset using saved earlier-group context.
# `--quick` reduces profile work.
# `--write-preset-only` rebuilds presets from a record.
# `--before-after-only` reruns replica comparisons from that record.
# use the corresponding `--no-*` options to skip output stages.
# run through tools/memguard.py.
# a full serial fit can take several hours.
# see docs/handoff/calibration.md for methods and findings.
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["register_cli", "main", "setup", "run_cmd"]


# add the options of `mrm calibrate` to the parser.
def setup(ap: argparse.ArgumentParser) -> None:
    from mrm.calibration.pipeline import GROUP_ORDER, RESULTS_PATH
    ap.add_argument("--groups", nargs="*", choices=list(GROUP_ORDER), default=None, help="fit groups (default all)")
    ap.add_argument("--quick", action="store_true", help="coarser profiles and fewer starts")
    ap.add_argument("--out", default=str(RESULTS_PATH), help="fit record (JSON)")
    ap.add_argument("--no-preset", action="store_true", help="do not write the calibrated presets")
    ap.add_argument("--no-before-after", action="store_true", help="skip the before/after replica table")
    ap.add_argument("--write-preset-only", action="store_true", help="rewrite the calibrated presets from --out")
    ap.add_argument("--before-after-only", action="store_true", help="only the before/after table from --out")


# run `mrm calibrate` — the fits (unless an `--...-only` option), the presets, the before/after table.
def run_cmd(args: argparse.Namespace) -> int:
    from mrm.calibration import pipeline, preset, report
    output_path = Path(args.out)
    if args.write_preset_only or args.before_after_only:
        record = pipeline.load_results(output_path)
    else:
        calibration_run = pipeline.run(args.groups, quick=args.quick, out=output_path)
        record = pipeline.load_results(output_path)
        del calibration_run
    if not args.before_after_only and not args.no_preset:
        written = preset.write_preset(record)
        print(f"wrote {written}")
        written = preset.write_bench_preset(record)
        print(f"wrote {written}")
    if not args.write_preset_only and not args.no_before_after:
        print("before/after (replica primary protocols, calibration targets only):")
        table = report.before_after(record)
        record["before_after"] = table
        output_path.write_text(json.dumps(record, indent=1, allow_nan=False, default=str))
        print(report.format_table(table))
    return 0


# register `mrm calibrate` with the mrm CLI (the provider protocol of mrm.cli).
def register_cli(register) -> None:
    register("calibrate", "fit the model to the calibration targets and write g26_calibrated", setup, run_cmd)


# `python -m mrm.calibration`.
# keep the CLI help description independent of explanatory comments.
def main(argv=None) -> int:
    parser =argparse.ArgumentParser(prog="python -m mrm.calibration", description='run calibration fits, write child presets, and compare replica predictions.')
    setup(parser)
    return run_cmd(parser.parse_args(argv))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
