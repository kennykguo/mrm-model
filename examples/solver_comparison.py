"""Compare existing adaptive integrators on a short, assumed eight-lane plant.

Run: uv run python examples/solver_comparison.py

Each case runs in a fresh subprocess so its peak resident memory is measured
independently. The tighter Radau run is a numerical reference, not ground truth.
No physical source files are changed. This is not measured-device validation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np

from ringlab.config import load
from ringlab.experiments import plain
from ringlab.plant import simulate
from ringlab.provenance import require_unchanged_sources, source_hashes


ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "Radau_default": ("Radau", 1e-6),
    "DOP853_default": ("DOP853", 1e-6),
    "DOP853_refined": ("DOP853", 1e-8),
    "Radau_refined": ("Radau", 1e-8),
}
REFERENCE = "Radau_refined"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_case(label, model, duration_s):
    method, rtol = CASES[label]
    configuration = load(model, {
        "simulation.duration_s": duration_s,
        "sensor.noise_enabled": False,
        "simulation.method": method,
        "simulation.rtol": rtol,
    })
    cpu_before = resource.getrusage(resource.RUSAGE_SELF)
    started = time.perf_counter()
    result = simulate(configuration)
    wall_s = time.perf_counter() - started
    usage = resource.getrusage(resource.RUSAGE_SELF)
    finite = (all(np.all(np.isfinite(values)) for values in result.trace.values())
              and np.all(np.isfinite(result.final_state)))
    if not finite:
        raise RuntimeError(f"{label} produced nonfinite trace or final-state values")
    require_unchanged_sources(result.source_sha256)
    return plain({
        "label": label,
        "configuration": configuration.resolved(),
        "configuration_sha256": configuration.digest(),
        "source_sha256": result.source_sha256,
        "controller_spec": result.controller_spec,
        "wall_s": wall_s,
        "cpu_s": (usage.ru_utime + usage.ru_stime
                  - cpu_before.ru_utime - cpu_before.ru_stime),
        "process_peak_rss_bytes": int(usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024),
        "rss_scope": "fresh case subprocess, including imports and simulation",
        "all_trace_and_state_values_finite": bool(finite),
        "metrics": result.metrics,
        "diagnostics": result.diagnostics,
        "trace": result.trace,
        "final_state": result.final_state,
    })


def compare(case, reference):
    actual, expected = case["trace"], reference["trace"]
    if not np.array_equal(actual["time_s"], expected["time_s"]):
        raise RuntimeError("solver cases have different recording times")
    differences = {
        f"max_abs_{key}": float(np.max(np.abs(np.asarray(actual[key]) - np.asarray(expected[key]))))
        for key in ("ring_temperature_k", "detuning_hz", "oma_w", "monitor_code", "heater_code")
    }
    energy = np.asarray(case["metrics"]["heater_energy_j"])
    reference_energy = np.asarray(reference["metrics"]["heater_energy_j"])
    nonzero = reference_energy != 0
    differences.update({
        "max_abs_heater_energy_j": float(np.max(np.abs(energy - reference_energy))),
        "max_relative_heater_energy_nonzero_reference": float(np.max(
            np.abs((energy[nonzero] - reference_energy[nonzero]) / reference_energy[nonzero]))) if np.any(nonzero) else None,
        "max_abs_mean_oma_w": float(np.max(np.abs(
            np.asarray(case["metrics"]["mean_oma_w"]) - np.asarray(reference["metrics"]["mean_oma_w"])))),
        "identical_adc_codes": bool(np.array_equal(actual["monitor_code"], expected["monitor_code"])),
        "identical_dac_codes": bool(np.array_equal(actual["heater_code"], expected["heater_code"])),
        "identical_record_timestamps": True,
    })
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "examples/models/wdm_context.yaml")
    parser.add_argument("--duration-s", type=float, default=0.0002)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/solver_comparison.json")
    parser.add_argument("--worker", choices=tuple(CASES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(run_case(args.worker, args.model, args.duration_s), allow_nan=False))
        return

    started = time.perf_counter()
    sources = source_hashes()
    model_hash = sha256(args.model)
    script_hash = sha256(__file__)
    cases = {}
    for label in CASES:
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", label,
             "--model", str(args.model.resolve()), "--duration-s", str(args.duration_s)],
            check=True, text=True, capture_output=True, cwd=ROOT,
        )
        case = json.loads(process.stdout)
        if case["source_sha256"] != sources:
            raise RuntimeError("source fingerprints differ between solver cases")
        cases[label] = case
        print(f"{label}: {case['wall_s']:.3f} s, "
              f"{case['diagnostics']['rhs_calls']} RHS calls, "
              f"peak process RSS {case['process_peak_rss_bytes'] / 1024**2:.1f} MiB", flush=True)
    if source_hashes() != sources or sha256(args.model) != model_hash or sha256(__file__) != script_hash:
        raise RuntimeError("source, input model, or comparison script changed during the run")
    reference = cases[REFERENCE]
    for case in cases.values():
        case["difference_from_refined_reference"] = compare(case, reference)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "numerical solver comparison on the same short assumed eight-lane model",
        "reference_label": REFERENCE,
        "reference_is_ground_truth": False,
        "duration_s": args.duration_s,
        "sensor_noise_enabled": False,
        "model_file": str(args.model.resolve().relative_to(ROOT)),
        "model_file_sha256": model_hash,
        "script_sha256": script_hash,
        "source_sha256": sources,
        "source_fingerprints_identical_and_unchanged": True,
        "software": {"python": sys.version, "numpy": version("numpy"), "scipy": version("scipy")},
        "platform": platform.platform(),
        "total_wall_s": time.perf_counter() - started,
        "runtime_scope": "sequential fresh subprocesses; other host workloads can affect wall times",
        "limitations": [
            "All physics and controller assumptions remain those in the recorded model configuration.",
            "No measured Lightmatter thermal transient or receiver calibration is used.",
            "Agreement between numerical methods does not establish physical device accuracy.",
            "The refined reference changes rtol to 1e-8; absolute tolerances and spectral-table settings are retained and recorded.",
            "The short run does not establish agreement across a complete acquisition or more stiff parameter regimes.",
            "ADC/DAC comparisons apply to recorded samples; full metrics retain their documented temporal limits.",
        ],
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Wrote {args.output}; refined numerical reference, not device validation.")


if __name__ == "__main__":
    main()
