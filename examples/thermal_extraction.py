"""Reproduce mesh and reduced-model evidence for the explicitly assumed slab.

Run: uv run python examples/thermal_extraction.py --max-refinement 16
Generated output is numerical verification, not measured device validation.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

from ringlab.extraction.thermal import assemble_slab, assumed_slab_geometry, spatial_convergence


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-refinement", type=int, choices=(2, 4, 8, 16), default=8)
    parser.add_argument("--output", type=Path, default=Path("artifacts/thermal_extraction.json"))
    args = parser.parse_args()
    levels = tuple(level for level in (1, 2, 4, 8, 16) if level <= args.max_refinement)
    report = spatial_convergence(levels=levels)
    root = Path(__file__).resolve().parents[1]
    source_paths = ("src/ringlab/thermal.py", "src/ringlab/extraction/thermal.py",
                    "src/ringlab/calibration.py", "examples/thermal_extraction.py")
    report["source_sha256"] = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in source_paths
    }
    report["software"] = {"python": sys.version, "numpy": version("numpy"), "scipy": version("scipy")}
    report["provenance"] = "synthetic geometry; independent numerical verification only; not foundry validation"
    report["boundary_conditions"] = "isothermal bottom, adiabatic top and sides, uniform 20 um extrusion"
    report["si_units"] = True
    report["geometry"] = {
        "width_m": 80e-6, "extrusion_depth_m": 20e-6, "ring_pitch_m": 30e-6,
        "layer_thicknesses_m": [20e-6, 2e-6, 0.22e-6, 1.5e-6],
        "layer_conductivity_w_per_m_k": [130.0, 1.4, 80.0, 1.4],
        "layer_volumetric_capacity_j_per_m3_k": [1.63e6, 1.55e6, 1.63e6, 1.55e6],
        "status": "all geometry and material values assumed",
    }
    model = assemble_slab(assumed_slab_geometry(), 128, (12, 8, 4, 8))
    times = np.r_[0, np.geomspace(1e-8, 1e-3, 60)]
    powers = np.array([1e-3, 0, 0, 0, 0, 0])
    reference = model.step_response(times, powers)
    report["krylov_heater_step"] = []
    for moments in (1, 2, 4, 8):
        reduced = model.krylov_reduce(moments)
        prediction = reduced.step_response(times, powers)
        report["krylov_heater_step"].append({
            "moments": moments, "modes": reduced.n_modes,
            "relative_dc_gain_matrix_error": reduced.dc_gain_relative_error,
            "maximum_temperature_error_k": float(np.max(np.abs(prediction - reference))),
        })
    report["krylov_reference_cells"] = model.n_nodes
    report["krylov_step_sampling_s"] = times.tolist()
    report["krylov_48mode_all_sources"] = []
    reduced = model.krylov_reduce(8)
    for source_index, source in enumerate(model.geometry.heat_sources):
        source_power = np.zeros(len(model.geometry.heat_sources))
        source_power[source_index] = 1e-3
        reference = model.step_response(times, source_power)
        prediction = reduced.step_response(times, source_power)
        report["krylov_48mode_all_sources"].append({
            "source": source.name, "step_power_w": 1e-3,
            "maximum_temperature_error_k": float(np.max(np.abs(prediction - reference))),
            "minimum_predicted_rise_k": float(np.min(prediction - 300)),
        })
    report["rom_limit"] = (
        "Projected modes preserve stable diffusion dynamics and DC gains, but not pointwise positivity. "
        "Negative tiny neighbor rises are truncation error, not cooling; the full FV and RC models "
        "retain positivity. Fixed uniform boundary only. ROM is not the default control plant."
    )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["wall_s"] = time.perf_counter() - started
    report["process_peak_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}; numerical verification for assumed geometry, not device measurements.")


if __name__ == "__main__":
    main()
