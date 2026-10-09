"""Reproducible CLI for the same model/configuration used by the lab."""

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .config import ExperimentConfig, default_config


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")


def prbs9(count=511):
    """x^9+x^5+1 maximal-length binary sequence, nonzero seed, repeats at 511 bits."""
    state = 0x1FF
    bits = np.empty(count, dtype=int)
    for i in range(count):
        bits[i] = (state >> 8) & 1
        feedback = ((state >> 8) ^ (state >> 4)) & 1
        state = ((state << 1) & 0x1FF) | feedback
    return bits


def parser():
    p = argparse.ArgumentParser(description="Microring control laboratory (uncalibrated defaults)")
    sub = p.add_subparsers(dest="command", required=True)
    preset = sub.add_parser("preset", help="write a complete editable configuration")
    preset.add_argument("--rings", type=int, default=1)
    preset.add_argument("--output", default="examples/single_ring.json")
    for name in ("run", "compare", "spectrum", "waveform"):
        item = sub.add_parser(name)
        item.add_argument("--config", default="examples/single_ring.json")
        item.add_argument("--output", default=f"outputs/{name}")
        if name == "run":
            item.add_argument("--controller", choices=["open_loop", "pi", "dither", "mpc"])
        elif name == "compare":
            item.add_argument("--controllers", nargs="+", default=["open_loop", "pi", "dither", "mpc"])
        elif name == "waveform":
            item.add_argument("--bits", type=int, default=511)
            item.add_argument("--rate", type=float, default=50e9)
            item.add_argument("--samples-per-bit", type=int, default=32)
    for name in ("calibrate-spectrum", "calibrate-thermal"):
        item = sub.add_parser(name)
        item.add_argument("csv", help="SI-unit CSV with a header; see docs/calibration.md")
        item.add_argument("--config", default="examples/single_ring.json")
        item.add_argument("--output", default=f"outputs/{name}.json")
        if name == "calibrate-spectrum":
            item.add_argument("--input-power", type=float, default=1)
        else:
            item.add_argument("--step-power", type=float, required=True)
            item.add_argument("--modes", type=int, default=2)
    fem = sub.add_parser("extract-thermal", help="generic sparse FEM convergence and ROM export")
    fem.add_argument("--output", default="outputs/thermal_fem")
    fem.add_argument("--modes", type=int, default=32)
    sweep = sub.add_parser("sweep", help="joint parameter corners with fixed nominal calibration")
    sweep.add_argument("--config", default="examples/single_ring.json")
    sweep.add_argument("--corners", default="examples/corners.json")
    sweep.add_argument("--controllers", nargs="+", default=["pi", "mpc"])
    sweep.add_argument("--output", default="outputs/corners.json")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "preset":
            write_json(args.output, default_config(args.rings).to_dict())
            print(args.output)
            return 0
        if args.command == "extract-thermal":
            from .extraction import thermal_fem_convergence

            folder = Path(args.output)
            report = thermal_fem_convergence(n_modes=args.modes)
            model = report["fine"]
            write_json(folder / "convergence.json", report)
            write_json(folder / "reference.json", model)
            print(
                json.dumps(
                    json_safe({key: value for key, value in report.items() if key not in {"coarse", "fine"}}),
                    indent=2,
                )
            )
            return 0
        config = ExperimentConfig.load(args.config)
        if args.command == "sweep":
            from .study import corner_study

            corners = json.loads(Path(args.corners).read_text())
            if not isinstance(corners, list) or len(corners) > 1000:
                raise ValueError("corners must be an array of at most 1000 jointly specified cases")
            summaries = list(corner_study(config, corners, args.controllers))
            write_json(args.output, {"nominal_config": config.to_dict(), "corners": summaries})
            print(f"{len(summaries)} corners written to {args.output}; nominal calibration was held fixed")
            return 0
        if args.command in {"run", "compare"}:
            from .simulation import compare_controllers, run_experiment

            if args.command == "run":
                if args.controller:
                    config = replace(config, controller=replace(config.controller, kind=args.controller))
                result = run_experiment(config)
                result.save(args.output)
                print(json.dumps(result.metrics, indent=2))
            else:
                results = compare_controllers(config, args.controllers)
                for name, result in results.items():
                    result.save(Path(args.output) / name)
                summary = {name: result.metrics for name, result in results.items()}
                write_json(Path(args.output) / "comparison.json", summary)
                print(json.dumps(summary, indent=2))
        elif args.command == "spectrum":
            from .optics import optical_response

            p = config.optical[0]
            frequencies = np.linspace(
                p.reference_frequency_hz - p.fsr_hz, p.reference_frequency_hz + p.fsr_hz, 8001
            )
            response = optical_response(
                p, frequencies, p.reference_temperature_k, 0, config.run.input_power_w
            )
            folder = Path(args.output)
            folder.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                folder / "spectrum.csv",
                np.column_stack([frequencies, response.through_power_w, response.drop_power_w]),
                delimiter=",",
                header="frequency_hz,through_power_w,drop_power_w",
                comments="",
            )
        elif args.command == "waveform":
            from .simulation import _steady_state
            from .waveform import simulate_nrz

            network, _ = _steady_state(config)
            result = simulate_nrz(
                config.optical[0],
                prbs9(args.bits),
                bit_rate_bps=args.rate,
                samples_per_bit=args.samples_per_bit,
                voltage_low_v=config.run.junction_low_v,
                voltage_high_v=config.run.junction_high_v,
                laser_frequency_hz=config.run.laser_frequency_hz[0],
                temperature_k=network.ring_temperatures_k[0],
                input_power_w=config.run.input_power_w,
            )
            folder = Path(args.output)
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                folder / "waveform.npz",
                **{key: value for key, value in result.items() if isinstance(value, np.ndarray)},
            )
            write_json(
                folder / "metrics.json",
                {key: value for key, value in result.items() if not isinstance(value, np.ndarray)},
            )
            print(json.dumps(json_safe(result["metrics"]), indent=2))
        elif args.command.startswith("calibrate-"):
            from .calibration import fit_optical_spectrum, fit_thermal_step

            data = np.genfromtxt(args.csv, delimiter=",", names=True)
            if args.command == "calibrate-spectrum":
                drop = data["drop_power_w"] if "drop_power_w" in data.dtype.names else None
                fit = fit_optical_spectrum(
                    config.optical[0],
                    data["frequency_hz"],
                    data["through_power_w"],
                    drop_power_w=drop,
                    input_power_w=args.input_power,
                )
                report = asdict(fit)
                optical = (fit.params, *config.optical[1:])
                provenance = dict(
                    config.provenance,
                    calibrated=False,
                    optical_fit_source=str(Path(args.csv).resolve()),
                    optical_fit_success=bool(fit.success),
                    calibration_note="Only optical parameters fitted; thermal/interface device accuracy unproven.",
                )
                updated = replace(config, optical=optical, provenance=provenance)
                write_json(Path(args.output).with_suffix(".config.json"), updated.to_dict())
            else:
                fit = fit_thermal_step(
                    data["time_s"],
                    data["frequency_shift_hz"],
                    n_modes=args.modes,
                    step_power_w=args.step_power,
                )
                report = asdict(fit)
            write_json(args.output, report)
            print(
                json.dumps(
                    json_safe(
                        {key: value for key, value in report.items() if not isinstance(value, np.ndarray)}
                    ),
                    indent=2,
                )
            )
        return 0
    except (ValueError, TypeError, KeyError, OSError, ImportError) as error:
        print(f"mrm: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
