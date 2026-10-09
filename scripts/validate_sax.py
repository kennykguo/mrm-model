#!/usr/bin/env python3
"""Independent component-network validation of the periodic compact spectrum.

Uses two unitary four-port couplers and two reciprocal propagation segments.
SAX's feedback-capable Filipsson-Gunnar solver and an independent port-graph
linear solve are compared with the compact model; neither network contains its
closed-form ring transfer function. This is equation verification, not hardware
validation. The optional SAX/JAX stack runs CPU float64 without GPU allocation.
"""

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import math
import os
import resource
from dataclasses import replace
from pathlib import Path

import numpy as np

from mrm_model.optics import C, OpticalParams, optical_response


def coupler_smatrix(coupling_power):
    """Reciprocal unitary scattering matrix: in0,in1,out0,out1."""
    t = math.sqrt(1 - coupling_power)
    k = 1j * math.sqrt(coupling_power)
    return np.array([[0, 0, t, k], [0, 0, k, t], [t, k, 0, 0], [k, t, 0, 0]], dtype=complex)


def _operating_values(params, temperature_k=None, junction_voltage_v=0.0):
    """Share physical inputs, independently evaluate their scalar material laws."""
    temperature = params.reference_temperature_k if temperature_k is None else temperature_k
    shift = params.df_dvoltage_hz_per_v * junction_voltage_v
    loss = params.loss_power_np_per_m + params.loss_voltage_np_per_m_per_v * junction_voltage_v
    if len(params.voltage_points_v):
        if not params.voltage_points_v[0] <= junction_voltage_v <= params.voltage_points_v[-1]:
            raise ValueError("voltage outside supplied table")
        if len(params.resonance_voltage_shifts_hz):
            shift = float(
                np.interp(junction_voltage_v, params.voltage_points_v, params.resonance_voltage_shifts_hz)
            )
        if len(params.loss_voltage_values_np_per_m):
            loss = float(
                np.interp(junction_voltage_v, params.voltage_points_v, params.loss_voltage_values_np_per_m)
            )
    resonance = (
        params.reference_frequency_hz
        + params.df_dtemperature_hz_per_k * (temperature - params.reference_temperature_k)
        + shift
    )
    if loss < 0:
        raise ValueError("negative propagation loss")
    return resonance, loss


def port_graph_response(
    params, frequency_hz, *, input_power_w=1e-3, temperature_k=None, junction_voltage_v=0.0
):
    """Solve a 12-port component connectivity graph using numpy.linalg.solve.

    b=S*a; a=P*b+e, where S contains primitive component matrices and P pairs
    connected ports. Solve (I-P*S)*a=e. Absorption is summed from loss in each
    propagation segment, independently of external port power subtraction.
    """
    f = np.atleast_1d(np.asarray(frequency_hz, dtype=float))
    if f.ndim != 1 or len(f) > 10_000 or not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("frequency needs a finite positive vector with <=10000 entries")
    if not math.isfinite(input_power_w) or input_power_w < 0:
        raise ValueError("input power must be finite and nonnegative")
    resonance, alpha = _operating_values(params, temperature_k, junction_voltage_v)
    omega = 2 * np.pi * (f - resonance)
    half_length = params.length_m / 2
    segment_phase = omega * params.group_index * half_length / C + params.phase_dispersion_s2 * omega**2 / 4
    local_roundtrip = params.group_index * params.length_m / C + params.phase_dispersion_s2 * omega
    if np.any(local_roundtrip <= 0):
        raise ValueError("nonpositive local group delay from dispersion extrapolation")
    propagation = np.exp(-alpha * half_length / 2 + 1j * segment_phase)
    s = np.zeros((len(f), 12, 12), dtype=complex)
    # Port indexing: bus 0:4; drop 4:8; upper 8:10; lower 10:12.
    s[:, :4, :4] = coupler_smatrix(params.coupling_bus_power)
    s[:, 4:8, 4:8] = coupler_smatrix(params.coupling_drop_power)
    for start in (8, 10):
        s[:, start, start + 1] = propagation
        s[:, start + 1, start] = propagation
    connection = np.zeros((12, 12))
    for one, other in ((3, 8), (9, 5), (7, 10), (11, 1)):
        connection[one, other] = connection[other, one] = 1
    source = np.zeros((len(f), 12, 1), dtype=complex)
    source[:, 0, 0] = math.sqrt(input_power_w)
    matrix = np.eye(12)[None, :, :] - connection @ s
    try:
        incident = np.linalg.solve(matrix, source)[..., 0]
    except np.linalg.LinAlgError:
        # A perfectly isolated lossless mode is unexcited; choose zero energy.
        incident = np.stack(
            [np.linalg.lstsq(m, e[:, 0], rcond=1e-13)[0] for m, e in zip(matrix, source, strict=True)]
        )
    outgoing = np.einsum("fij,fj->fi", s, incident)
    component_loss = np.zeros(len(f))
    energy = np.zeros(len(f))
    effective_length = -math.expm1(-alpha * half_length) / alpha if alpha else half_length
    for start in (8, 10):
        component_loss += np.sum(
            np.abs(incident[:, start : start + 2]) ** 2 - np.abs(outgoing[:, start : start + 2]) ** 2, axis=1
        )
        energy += (
            np.sum(np.abs(incident[:, start : start + 2]) ** 2, axis=1)
            * effective_length
            * local_roundtrip
            / params.length_m
        )
    # Tiny negative dissipation from unitary roundoff is retained in diagnostics.
    return {
        "through_power_w": np.abs(outgoing[:, 2]) ** 2,
        "drop_power_w": np.abs(outgoing[:, 6]) ** 2,
        "reflected_power_w": np.abs(outgoing[:, 0]) ** 2,
        "add_output_power_w": np.abs(outgoing[:, 4]) ** 2,
        "dissipated_power_w": component_loss,
        "absorbed_power_w": params.intrinsic_absorption_fraction * component_loss,
        "scattered_power_w": (1 - params.intrinsic_absorption_fraction) * component_loss,
        "cavity_energy_j": energy,
        "incident_fields": incident,
        "outgoing_fields": outgoing,
    }


def build_sax_ring():
    """Build the feedback circuit from component models, never a ring formula."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import jax
    import jax.numpy as jnp
    import sax

    jax.config.update("jax_enable_x64", True)

    def coupler(coupling_power=0.1):
        t, k = jnp.sqrt(1 - coupling_power), 1j * jnp.sqrt(coupling_power)
        return sax.reciprocal(
            {("in0", "out0"): t, ("in0", "out1"): k, ("in1", "out0"): k, ("in1", "out1"): t}
        )

    def segment(
        frequency_hz=1.0,
        resonance_frequency_hz=1.0,
        length_m=20e-6,
        group_index=4.2,
        loss_power_np_per_m=1000.0,
        phase_dispersion_s2=0.0,
    ):
        omega = 2 * jnp.pi * (jnp.asarray(frequency_hz) - resonance_frequency_hz)
        phase = omega * group_index * length_m / C + phase_dispersion_s2 * omega**2 / 2
        amplitude = jnp.exp(-loss_power_np_per_m * length_m / 2 + 1j * phase)
        return sax.reciprocal({("in", "out"): amplitude})

    netlist = {
        "instances": {"bus": coupler, "drop": coupler, "upper": segment, "lower": segment},
        "connections": {
            "bus,out1": "upper,in",
            "upper,out": "drop,in1",
            "drop,out1": "lower,in",
            "lower,out": "bus,in1",
        },
        "ports": {"input": "bus,in0", "through": "bus,out0", "add": "drop,in0", "drop": "drop,out0"},
    }
    model, _ = sax.circuit(netlist=netlist, backend="filipsson_gunnar")
    return model


def sax_response(
    params, frequency_hz, *, model=None, input_power_w=1e-3, temperature_k=None, junction_voltage_v=0.0
):
    model = build_sax_ring() if model is None else model
    resonance, loss = _operating_values(params, temperature_k, junction_voltage_v)
    values = model(
        frequency_hz=np.asarray(frequency_hz, dtype=float),
        resonance_frequency_hz=resonance,
        bus={"coupling_power": params.coupling_bus_power},
        drop={"coupling_power": params.coupling_drop_power},
        length_m=params.length_m / 2,
        group_index=params.group_index,
        loss_power_np_per_m=loss,
        phase_dispersion_s2=params.phase_dispersion_s2 / 2,
    )
    through = input_power_w * np.abs(np.asarray(values["through", "input"])) ** 2
    drop = input_power_w * np.abs(np.asarray(values["drop", "input"])) ** 2
    return {"through_power_w": through, "drop_power_w": drop}


def validate(*, seed=92, random_cases=20, samples=401, solver="auto"):
    if random_cases < 0 or samples < 9 or samples > 4001:
        raise ValueError("random_cases must be nonnegative and samples must lie in [9,4001]")
    has_sax = importlib.util.find_spec("sax") is not None
    if solver == "sax" and not has_sax:
        raise RuntimeError("SAX is optional: install 'mrm-model[validation]' or sax==0.18.2")
    use_sax = solver == "sax" or (solver == "auto" and has_sax)
    sax_model = build_sax_ring() if use_sax else None
    rng = np.random.default_rng(seed)
    default = OpticalParams()
    cases = [
        ("default", default),
        ("single_bus", replace(default, coupling_drop_power=0)),
        ("lossless", replace(default, loss_power_np_per_m=0)),
        (
            "weak_coupling",
            replace(default, coupling_bus_power=0.0008, coupling_drop_power=0.0004, loss_power_np_per_m=2.5),
        ),
        ("absorptive_fraction", replace(default, intrinsic_absorption_fraction=0.2)),
        ("dispersive", replace(default, phase_dispersion_s2=1e-26)),
    ]
    for i in range(random_cases):
        cases.append(
            (
                f"random_{i:02d}",
                replace(
                    default,
                    length_m=float(rng.uniform(20e-6, 70e-6)),
                    group_index=float(rng.uniform(3.4, 4.6)),
                    loss_power_np_per_m=float(rng.uniform(0, 20000)),
                    coupling_bus_power=float(rng.uniform(0.002, 0.6)),
                    coupling_drop_power=float(rng.uniform(0, 0.4)),
                    intrinsic_absorption_fraction=float(rng.uniform(0, 1)),
                ),
            )
        )
    rows = []
    curve = None
    input_power = 0.001
    for name, params in cases:
        f = params.reference_frequency_hz + np.linspace(-1.2 * params.fsr_hz, 1.2 * params.fsr_hz, samples)
        exact = optical_response(params, f, input_power_w=input_power)
        graph = port_graph_response(params, f, input_power_w=input_power)
        external = sax_response(params, f, model=sax_model, input_power_w=input_power) if use_sax else graph
        periodic = (
            None
            if params.phase_dispersion_s2
            else port_graph_response(params, f + params.fsr_hz, input_power_w=input_power)
        )
        row = {
            "case": name,
            "max_abs_through_error_w": float(
                np.max(np.abs(external["through_power_w"] - exact.through_power_w))
            ),
            "max_abs_drop_error_w": float(np.max(np.abs(external["drop_power_w"] - exact.drop_power_w))),
            "max_abs_graph_through_error_w": float(
                np.max(np.abs(graph["through_power_w"] - exact.through_power_w))
            ),
            "max_abs_graph_drop_error_w": float(np.max(np.abs(graph["drop_power_w"] - exact.drop_power_w))),
            "max_abs_absorption_error_w": float(
                np.max(np.abs(graph["absorbed_power_w"] - exact.absorbed_power_w))
            ),
            "max_abs_energy_error_j": float(np.max(np.abs(graph["cavity_energy_j"] - exact.cavity_energy_j))),
            "max_abs_passivity_residual_w": float(
                np.max(
                    np.abs(
                        graph["through_power_w"]
                        + graph["drop_power_w"]
                        + graph["dissipated_power_w"]
                        - input_power
                    )
                )
            ),
            "max_abs_fsr_periodicity_error_w": None
            if periodic is None
            else float(np.max(np.abs(graph["through_power_w"] - periodic["through_power_w"]))),
            "minimum_dissipated_power_w": float(np.min(graph["dissipated_power_w"])),
        }
        rows.append(row)
        if name == "default":
            curve = {
                "frequency_hz": f,
                "detuning_hz": f - params.reference_frequency_hz,
                "compact_through_w": exact.through_power_w,
                "compact_drop_w": exact.drop_power_w,
                "network_through_w": external["through_power_w"],
                "network_drop_w": external["drop_power_w"],
                "graph_absorbed_w": graph["absorbed_power_w"],
                "compact_absorbed_w": exact.absorbed_power_w,
            }
    columns = [key for key in rows[0] if key.startswith("max_abs_")]
    maxima = {key: max(row[key] for row in rows if row[key] is not None) for key in columns}
    versions = {
        name: importlib.metadata.version(name)
        for name in (["numpy", "scipy", "sax", "jax", "jaxlib"] if use_sax else ["numpy", "scipy"])
    }
    report = {
        "solver": "SAX Filipsson-Gunnar plus independent 12-port NumPy solve"
        if use_sax
        else "independent 12-port NumPy solve (SAX unavailable or disabled)",
        "versions": versions,
        "seed": seed,
        "case_count": len(cases),
        "samples_per_case": samples,
        "input_power_w": input_power,
        "max_errors": maxima,
        "cases": rows,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "scope": "Linear passive periodic spectrum and energy bookkeeping under shared ideal coupler/uniform segment assumptions; not measured-device, nonlinear or transient validation.",
    }
    report["passed"] = bool(
        all(value < (1e-20 if key.endswith("_j") else 1e-10) for key, value in maxima.items())
    )
    return report, curve


def save_artifacts(report, curve, destination):
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "curves.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(curve)
        writer.writerows(zip(*curve.values(), strict=True))
    if importlib.util.find_spec("matplotlib") is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
        x = curve["detuning_hz"] / 1e9
        for port in ("through", "drop"):
            axes[0].plot(x, curve[f"compact_{port}_w"] * 1e3, label=f"compact {port}")
            axes[0].plot(x, curve[f"network_{port}_w"] * 1e3, "--", label=f"network {port}")
            axes[1].plot(x, (curve[f"network_{port}_w"] - curve[f"compact_{port}_w"]) * 1e3, label=port)
        axes[0].set_ylabel("Output power (mW)")
        axes[1].set_ylabel("Network minus compact (mW)")
        axes[1].set_xlabel("Detuning from reference resonance (GHz)")
        axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2)
        axes[1].legend()
        for axis in axes:
            axis.grid(alpha=0.2)
        fig.savefig(folder / "spectrum_comparison.png", dpi=150)
        plt.close(fig)
    report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    (folder / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("auto", "sax", "graph"), default="auto")
    parser.add_argument("--random-cases", type=int, default=20)
    parser.add_argument("--samples", type=int, default=401)
    parser.add_argument("--output", default="results/sax-validation")
    args = parser.parse_args(argv)
    report, curve = validate(solver=args.solver, random_cases=args.random_cases, samples=args.samples)
    save_artifacts(report, curve, args.output)
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
