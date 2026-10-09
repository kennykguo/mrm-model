"""Reproduce the fixed-model lab comparison and an independent solver check.

Run after source edits are complete, in a fresh Python process:
    uv run python examples/control_lab_checks.py
The full traces/configuration/provenance and a paired plot go to artifacts/.
These are numerical checks of a synthetic example, not device validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ringlab.config import load
from ringlab.experiments import plain
from ringlab.lab import DEFAULTS, LabState
from ringlab.plant import simulate
from ringlab.provenance import require_unchanged_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    lab = LabState()
    comparison = lab.run(DEFAULTS)
    reference = comparison["runs"]["step"]["trace"]
    refined = simulate(load(lab.configuration(DEFAULTS, "step"), {
        "simulation.method": "DOP853", "simulation.rtol": 1e-8,
    }))
    differences = {key: float(np.max(np.abs(np.asarray(reference[key])-value)))
                   for key, value in refined.trace.items()}
    check = {
        "reference": "Radau rtol=1e-6", "comparison": "DOP853 rtol=1e-8",
        "maximum_recorded_absolute_difference": differences,
        "identical_recorded_adc": differences["monitor_code"] == 0,
        "identical_recorded_dac": differences["heater_code"] == 0,
        "refined_configuration": refined.configuration.resolved(),
        "refined_diagnostics": refined.diagnostics,
        "scope": "This default 4 ms single-ring experiment only. Cross-solver agreement is not physical device validation or a universal numerical error bound.",
    }
    require_unchanged_sources()
    args.output.mkdir(parents=True, exist_ok=True)
    document = {"comparison": comparison, "solver_check": check,
                "check_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / "control_lab.json").write_text(json.dumps(plain(document), indent=2, allow_nan=False)+"\n")

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), constrained_layout=True)
    panels = [("monitor_code", "Delayed monitor (ADC codes)", 1),
              ("heater_code", "Applied heater (DAC codes)", 1),
              ("detuning_hz", "Resonance − laser (GHz), evaluator", 1e-9),
              ("oma_w", "OMA (µW), evaluator", 1e6)]
    for ax, (key, label, scale) in zip(axes.flat, panels):
        for name, color, linestyle, title in (("step", "#087f70", "-", "Step controller"),
                                                ("hold", "#737b84", "--", "Fixed heater")):
            trace = comparison["runs"][name]["trace"]
            t = np.asarray(trace["time_s"])*1e3
            y = np.asarray(trace[key])[:, 0]*scale
            ax.plot(t, y, color=color, linestyle=linestyle, linewidth=1.7,
                    drawstyle="steps-post" if key.endswith("code") else "default", label=title)
        if key == "monitor_code":
            ax.axhspan(DEFAULTS["target_code"]-DEFAULTS["deadband_codes"],
                       DEFAULTS["target_code"]+DEFAULTS["deadband_codes"], color="#087f70", alpha=.12)
            ax.axhline(DEFAULTS["target_code"], color="#087f70", linewidth=.8, linestyle=":")
        ax.axvline(2, color="#a06633", linewidth=.9, label="Ambient +0.25 K at 2 ms")
        ax.set(xlabel="Recorded time (ms)", ylabel=label, xlim=(.5, 4))
        # Show the local-control experiment after its explicitly omitted warmup.
        values = [np.asarray(comparison["runs"][name]["trace"][key])[:, 0]*scale for name in ("step", "hold")]
        active = np.asarray(comparison["runs"]["step"]["trace"]["time_s"]) >= .0005
        lo, hi = min(v[active].min() for v in values), max(v[active].max() for v in values)
        padding = max((hi-lo)*.12, 1e-6)
        ax.set_ylim(lo-padding, hi+padding)
        ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Minimal local stabilization · fixed synthetic plant\n"
                 "After 0.5 ms warmup; 20 µs recorded traces, not exact switch edges", fontsize=12)
    fig.savefig(args.output / "control_lab_response.png", dpi=180)
    plt.close(fig)
    print(json.dumps(plain({"model_sha256": comparison["model_sha256"],
                           "control_metrics": {name: value["metrics"]["control"] for name, value in comparison["runs"].items()},
                           "solver_check": {key: value for key, value in check.items() if key not in ("refined_configuration", "refined_diagnostics")}}), indent=2))


if __name__ == "__main__":
    main()
