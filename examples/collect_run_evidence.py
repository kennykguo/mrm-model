"""Bundle the results of the shipped examples without rerunning simulations.

Supply completed run directories; nothing from other worktrees is read.
Example: uv run python examples/collect_run_evidence.py --reference RUN \
    --replay REPLAY --study STUDY --wdm WDM
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from ringlab.experiments import plain, plot_trace, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "replay", "study", "wdm"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--wdm-refined", type=Path)
    args = parser.parse_args()
    current = source_hashes()

    def collect(path):
        manifest = json.loads((path/"manifest.json").read_text())
        if manifest["source_sha256"] != current:
            raise ValueError(f"{path}: source hashes differ; rerun with current source")
        return {
            "run_directory_at_generation": str(path),
            "configuration": yaml.safe_load((path/"resolved.yaml").read_text()),
            "controller": json.loads((path/"controller.json").read_text()),
            "metrics": json.loads((path/"metrics.json").read_text()),
            "manifest": manifest,
        }

    reference, replay, wdm = (collect(path) for path in (args.reference, args.replay, args.wdm))
    study = json.loads((args.study/"comparison.json").read_text())
    if study["completed"] != study["total_runs"]:
        raise ValueError("comparison study did not complete")
    runs = [dict(case=row["case"], controller_label=row["controller"],
                 result=collect(args.study/row["run"])) for row in study["runs"]]
    with np.load(args.reference/"trace.npz", allow_pickle=False) as original, np.load(
            args.replay/"trace.npz", allow_pickle=False) as repeated:
        identical = original.files == repeated.files and all(
            np.array_equal(original[name], repeated[name]) for name in original.files)
        if not identical:
            raise ValueError("reference replay was not bitwise equal on this machine")
        reference_trace = {name: original[name].copy() for name in original.files}
    report = {
        "claim": "Conditional synthetic examples. Numerical verification/workflow evidence, not Lightmatter device validation or BER prediction.",
        "comparison_interpretation": "Includes cold-start acquisition. Dither and hold start far from the useful fringe; dither is a local peak tracker, not a global acquisition method. This is not a fair universal controller ranking or a statistically powered robustness study.",
        "source_sha256": current,
        "collection_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference": reference, "reference_replay": replay,
        "reference_replay_bitwise_equal_on_this_environment": identical,
        "study_specification": yaml.safe_load((args.study/"study.yaml").read_text()),
        "comparison_runs": runs,
        "wdm": wdm,
    }
    if args.wdm_refined:
        report["wdm_refined"] = collect(args.wdm_refined)
        with np.load(args.wdm/"trace.npz", allow_pickle=False) as primary, np.load(
                args.wdm_refined/"trace.npz", allow_pickle=False) as independent:
            if not np.array_equal(primary["time_s"], independent["time_s"]):
                raise ValueError("solver comparison needs matching record times")
            report["wdm_solver_comparison"] = {
                "interpretation": "Full acquisition/feedback trajectories with different solver/tolerance choices. Differences include possible quantized controller decisions; neither is exact truth or a hardware measurement.",
                "adc_codes_identical": np.array_equal(primary["monitor_code"], independent["monitor_code"]),
                "dac_codes_identical": np.array_equal(primary["heater_code"], independent["heater_code"]),
                "max_recorded_temperature_difference_k": np.max(abs(primary["ring_temperature_k"]-independent["ring_temperature_k"])),
                "max_recorded_detuning_difference_hz": np.max(abs(primary["detuning_hz"]-independent["detuning_hz"])),
                "max_recorded_oma_difference_w": np.max(abs(primary["oma_w"]-independent["oma_w"])),
            }
    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    (output/"example_runs.json").write_text(json.dumps(plain(report), indent=2, sort_keys=True, allow_nan=False)+"\n")
    np.savez_compressed(output/"reference_trace.npz", **reference_trace)
    plot_trace(reference_trace, output/"reference_response.png")
    with np.load(args.wdm/"trace.npz", allow_pickle=False) as stored:
        wdm_trace = {name: stored[name].copy() for name in stored.files}
    np.savez_compressed(output/"wdm_trace.npz", **wdm_trace)
    plot_trace(wdm_trace, output/"wdm_response.png")
    print(output/"example_runs.json")


if __name__ == "__main__":
    main()
