"""Reproduce numerical optical accuracy evidence, not a measured-device claim.

Run from the repository root:
    uv run python examples/optical_accuracy_checks.py

Outputs artifacts/optical_accuracy.json. All geometrical/electrical defaults are
illustrative. Waveforms are short synthetic iid NRZ patterns, not PRBS9 hardware
eyes. The independent table holdout checks interpolation of the stated model.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np

from ringlab.junction import DepletionJunction
from ringlab.optical_tables import build_nrz_average_table
from ringlab.optics import RingParameters, loaded_linewidth_hz, optical_response
from ringlab.waveform import nrz_average_response, nrz_drive, tcmt_response, waveform_response


def waveform_summary(response, bits, *, samples_per_bit, discard_bits):
    start = discard_bits * samples_per_bit
    elapsed = response.time_s[-1] - response.time_s[start]
    center_indices = np.arange(discard_bits, len(bits)) * samples_per_bit + samples_per_bit // 2
    samples = response.through_power_w[center_indices]
    labels = bits[discard_bits:]
    conditioned = [float(np.mean(samples[labels == level])) for level in (0, 1)]
    return {
        "elapsed_window_s": float(elapsed),
        "retained_bits": int(labels.size),
        "empirical_mark_fraction": float(np.mean(labels)),
        "empirical_transition_fraction": float(np.mean(np.diff(labels) != 0)),
        "mean_absorbed_power_w": float(np.trapezoid(response.absorbed_power_w[start:], response.time_s[start:]) / elapsed),
        "mean_through_power_w": float(np.trapezoid(response.through_power_w[start:], response.time_s[start:]) / elapsed),
        "center_conditional_through_power_w": conditioned,
        "center_conditional_oma_w": abs(conditioned[1] - conditioned[0]),
        "waveform_array_bytes": sum(array.nbytes for array in vars(response).values()),
    }


def run():
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    ring, junction = RingParameters(), DepletionJunction()
    bit_rate, input_power = 50e9, 1e-3
    voltage_levels = (0.5, 2.5)
    levels = junction.evaluate(voltage_levels)
    shifts, losses = tuple(levels.resonance_shift_hz), tuple(levels.additional_loss_per_m)
    laser = ring.reference_frequency_hz + float(np.mean(shifts))
    samples_per_bit, discarded = 16, 32
    source_paths = [
        "examples/optical_accuracy_checks.py", "src/ringlab/optics.py",
        "src/ringlab/junction.py", "src/ringlab/waveform.py",
        "src/ringlab/optical_tables.py", "src/ringlab/optical_calibration.py",
    ]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Reproducible numerical evidence for an assumed compact model; not measured Lightmatter accuracy or BER.",
        "source_sha256": {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in source_paths},
        "ring_parameters": asdict(ring), "junction_parameters": asdict(junction),
        "conditions": {
            "bit_rate_hz": bit_rate, "input_power_w": input_power,
            "temperature_k": ring.reference_temperature_k,
            "voltage_levels_v": voltage_levels, "laser_frequency_hz": laser,
            "samples_per_bit": samples_per_bit, "discarded_initial_bits": discarded,
            "initialization": "First command voltage and its steady cavity field; discard32 bits before summary.",
            "fsr_hz": ring.fsr_hz, "loaded_linewidth_hz": float(loaded_linewidth_hz(ring)),
        },
        "assumptions_and_limits": [
            "Device dimensions, coupling, loss split and electrical coefficients are synthetic assumptions;1310nm/50Gbps/thermal-slope magnitude are public operating context.",
            "One optical traveling mode, high-Q TCMT near a resonance, constant temperature during each short waveform.",
            "Stationary NRZ calculation assumes iid bits with probability0.5 and ideal instantaneous electrical levels.",
            "FiniteRC waveform uses the assumed nonlinear first-order capacitance/resistance model; no packageRF model, jitter, phase noise or receiver.",
            "Finite windows have random mark density, transition density and optical memory; their deviation from an infinite iid expectation is not solely integration error.",
            "Two seeds do not establish a statistical confidence interval. No optical eye opening, receiverBER or Lightmatter reproduction is claimed.",
            "The common absorption_fraction applies to TOTAL intrinsic loss, including its voltage delta; separate fully absorptive carrier loss and scattering baseline are not represented.",
            "TPA, Kerr and carrier feedback remain omitted from these waveforms; optional perturbation screening is not a certified bound.",
        ],
    }

    # Same pattern and sample times isolate numerical substep sensitivity.
    refinement_seed = 1041
    pattern = np.random.default_rng(refinement_seed).integers(0, 2, 128)
    sample_time, drive = nrz_drive(pattern, bit_rate_hz=bit_rate, samples_per_bit=samples_per_bit,
                                  voltage_zero_v=voltage_levels[0], voltage_one_v=voltage_levels[1])
    refinements = []
    previous = None
    for substeps in (1, 2, 4, 8):
        response = waveform_response(ring, sample_time, drive, junction, frequency_hz=laser,
                                     input_power_w=input_power, substeps=substeps, include_rc=True)
        entry = {"substeps": substeps, **waveform_summary(response, pattern, samples_per_bit=samples_per_bit, discard_bits=discarded)}
        if previous is not None:
            entry["max_through_difference_from_previous_w"] = float(np.max(np.abs(response.through_power_w - previous.through_power_w)))
            entry["max_junction_voltage_difference_from_previous_v"] = float(np.max(np.abs(response.junction_voltage_v - previous.junction_voltage_v)))
        refinements.append(entry)
        previous = response
    report["finite_rc_substep_refinement"] = {"seed": refinement_seed, "bits":128, "results": refinements}

    iid = nrz_average_response(ring, laser, resonance_shift_hz=shifts, additional_loss_per_m=losses,
                               input_power_w=input_power, bit_rate_hz=bit_rate, mark_probability=0.5)
    report["stationary_iid"] = {
        "mean_absorbed_power_w": float(iid.absorbed_power_w),
        "mean_through_power_w": float(iid.through_power_w),
        "conditional_oma_w": float(iid.oma_w),
    }
    windows = []
    for seed in (927, 928):
        for count in (256, 1024):
            bits = np.random.default_rng(np.random.SeedSequence([seed, count])).integers(0, 2, count)
            sample_time, drive = nrz_drive(bits, bit_rate_hz=bit_rate, samples_per_bit=samples_per_bit,
                                          voltage_zero_v=voltage_levels[0], voltage_one_v=voltage_levels[1])
            for include_rc in (False, True):
                response = waveform_response(ring, sample_time, drive, junction, frequency_hz=laser,
                                             input_power_w=input_power, substeps=4, include_rc=include_rc)
                entry = {"seed_entropy": [seed, count], "bits": count, "include_rc": include_rc,
                         "substeps": 4, **waveform_summary(response, bits, samples_per_bit=samples_per_bit, discard_bits=discarded)}
                entry["fractional_heat_difference_from_stationary_iid"] = entry["mean_absorbed_power_w"] / float(iid.absorbed_power_w) - 1
                windows.append(entry)
    report["finite_windows"] = windows

    periodic = []
    for detuning in (0, 20e9, 50e9, 100e9, 150e9, 400e9, ring.fsr_hz/2):
        frequency = laser + detuning
        exact = [optical_response(ring, frequency, resonance_shift_hz=shifts[j], additional_loss_per_m=losses[j]) for j in (0, 1)]
        local = [tcmt_response(ring, frequency, resonance_shift_hz=shifts[j], additional_loss_per_m=losses[j]) for j in (0, 1)]
        dynamic = nrz_average_response(ring, frequency, resonance_shift_hz=shifts, additional_loss_per_m=losses)
        exact_heat = float(np.mean([item.absorbed_power_w for item in exact]))
        local_heat = float(np.mean([item.absorbed_power_w for item in local]))
        periodic.append({
            "detuning_from_midpoint_hz": float(detuning),
            "exact_quasistatic_absorbed_fraction": exact_heat,
            "tcmt_quasistatic_absorbed_fraction": local_heat,
            "tcmt_iid_absorbed_fraction": float(dynamic.absorbed_power_w),
            "tcmt_minus_exact_quasistatic_fraction": local_heat - exact_heat,
            "tcmt_dynamic_minus_quasistatic_fraction": float(dynamic.absorbed_power_w) - local_heat,
        })
    report["periodic_vs_isolated_mode"] = periodic

    # A NEW validation set is disjoint in seed from adaptation and its built-in
    # heldout checks; this tests numerical interpolation rather than fitting data.
    table = build_nrz_average_table(ring, resonance_shift_hz=shifts, additional_loss_per_m=losses,
                                    absolute_tolerance=1e-7, max_nodes=32769)
    holdout_seed, holdout_count = 8917, 10000
    rng = np.random.default_rng(holdout_seed)
    detunings = rng.uniform(-3 * ring.fsr_hz, 3 * ring.fsr_hz, holdout_count)
    powers = rng.uniform(0.1e-3, 2e-3, holdout_count)
    approximated, direct = table.evaluate(detunings, powers), table.direct(detunings, powers)
    errors = {}
    for name in ("through_power_w", "drop_power_w", "intrinsic_loss_power_w", "absorbed_power_w", "oma_w"):
        errors[name] = float(np.max(np.abs(getattr(approximated, name) - getattr(direct, name)) / powers))
    errors["stored_energy_scaled_to_power"] = float(np.max(np.abs(approximated.stored_energy_j - direct.stored_energy_j) / powers) * table.energy_rate_per_s)
    report["independent_table_holdout"] = {
        "seed": holdout_seed, "points": holdout_count,
        "construction_report": table.diagnostics,
        "maximum_absolute_error_as_fraction_of_input": errors,
        "maximum_power_balance_error_fraction": float(np.max(np.abs(approximated.through_power_w + approximated.drop_power_w + approximated.intrinsic_loss_power_w - powers) / powers)),
        "interpretation": "Independent numerical holdout against direct equations; not physical validation. OMA can accumulate errors in two interpolated levels.",
    }
    report["runtime_s"] = time.perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["peak_process_rss_bytes"] = int(rss if platform.system() == "Darwin" else rss * 1024)
    report["runtime_platform"] = {"python": platform.python_version(), "numpy": np.__version__, "system": platform.system(), "machine": platform.machine()}
    output = root / "artifacts" / "optical_accuracy.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "runtime_s": report["runtime_s"],
                      "peak_process_rss_bytes": report["peak_process_rss_bytes"],
                      "table_holdout_errors": errors}, indent=2))


if __name__ == "__main__":
    run()
