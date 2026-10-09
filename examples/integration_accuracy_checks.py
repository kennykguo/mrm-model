"""Record independent numerical comparisons for one specified synthetic plant.

Run from the project root: uv run python examples/integration_accuracy_checks.py
This is numerical verification, not measured-device validation. Keep the report
with its source fingerprints; changes to the source require a new run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np

from ringlab.config import load
from ringlab.experiments import plain, source_hashes, versions
from ringlab.plant import simulate


def main():
    start = time.perf_counter()
    base = load(overrides={
        "simulation.duration_s": 40e-6,
        "simulation.record_period_s": 1e-6,
        "controller.name": "hold", "controller.parameters": {"code": 500},
        "sensor.noise_enabled": False, "array.cold_offset_hz": 10e9,
    })
    configs = {
        "table_default": base,
        "direct_default": load(base, {"simulation.optical_evaluation": "direct"}),
        "direct_refined": load(base, {
            "simulation.optical_evaluation": "direct", "simulation.rtol": 1e-8,
            "simulation.max_step_s": .25e-6,
            "simulation.temperature_atol_k": 1e-9,
            "simulation.voltage_atol_v": 1e-10,
            "simulation.current_atol_a": 1e-14,
            "simulation.heater_energy_atol_j": 1e-17,
            "simulation.oma_integral_atol_j": 1e-20,
        }),
    }
    results = {}
    for name, config in configs.items():
        results[name] = simulate(config)
        print(f"{name}: {results[name].diagnostics['wall_s']:.3f} s", flush=True)
    reference = results["direct_refined"]
    comparison = {}
    for name, result in results.items():
        comparison[name] = {
            "max_recorded_temperature_difference_k": float(np.max(abs(
                result.trace["ring_temperature_k"]-reference.trace["ring_temperature_k"]))),
            "max_recorded_detuning_difference_hz": float(np.max(abs(
                result.trace["detuning_hz"]-reference.trace["detuning_hz"]))),
            "max_recorded_oma_difference_w": float(np.max(abs(
                result.trace["oma_w"]-reference.trace["oma_w"]))),
            "mean_oma_difference_w": result.metrics["mean_oma_w"]-reference.metrics["mean_oma_w"],
            "heater_energy_difference_j": result.metrics["heater_energy_j"]-reference.metrics["heater_energy_j"],
            "diagnostics": result.diagnostics,
            "configuration": configs[name].resolved(),
        }
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {
        "claim": "Numerical comparison on a synthetic self-heating trajectory, not Lightmatter validation.",
        "reference": "direct_refined; this too is a finite-tolerance computation, not exact truth",
        "coverage": "40 us one-ring near-resonance transient, held heater code, stationary iid NRZ; no sensor noise",
        "interpretation": "Separates lookup interpolation and ODE refinement effects. It does not bound model discrepancy, all parameter choices, discontinuous controller decisions, or omitted physics.",
        "source_sha256": source_hashes(), "versions": versions(),
        "validation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(), "platform": platform.platform(),
        "comparison": comparison, "wall_s": time.perf_counter()-start,
        "process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss*1024),
    }
    path = Path("artifacts/integration_accuracy.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(plain(report), indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(path, flush=True)


if __name__ == "__main__":
    main()
