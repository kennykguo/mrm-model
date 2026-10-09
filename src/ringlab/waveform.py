"""Isolated-resonance TCMT waveforms and stationary iid-NRZ optical moments.

TCMT is an envelope approximation to optics.py's discrete, exact per-pass model.
It is accurate for weak per-pass loading near one resonance; finite RC waveforms
require timestep convergence. These routines predict neither receiver BER nor
the proprietary Lightmatter driver, photodetector or junction response.
"""

from dataclasses import dataclass
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .junction import JunctionModel
from .optics import RingParameters, _finite, decay_rates, optical_response, resonance_frequency


@dataclass(frozen=True)
class TCMTResponse:
    through_power_w: NDArray
    drop_power_w: NDArray
    intrinsic_loss_power_w: NDArray
    absorbed_power_w: NDArray
    stored_energy_j: NDArray
    cavity_field_sqrt_j: NDArray


@dataclass(frozen=True)
class NRZAverageResponse:
    through_power_w: NDArray
    drop_power_w: NDArray
    intrinsic_loss_power_w: NDArray
    absorbed_power_w: NDArray
    stored_energy_j: NDArray
    conditional_through_power_w: NDArray
    conditional_drop_power_w: NDArray
    oma_w: NDArray


@dataclass(frozen=True)
class WDMNRZAverageResponse(NRZAverageResponse):
    diagnostics: dict[str, NDArray | str]


@dataclass(frozen=True)
class WaveformResult:
    time_s: NDArray
    drive_voltage_v: NDArray
    junction_voltage_v: NDArray
    cavity_field_sqrt_j: NDArray
    through_power_w: NDArray
    drop_power_w: NDArray
    intrinsic_loss_power_w: NDArray
    absorbed_power_w: NDArray
    stored_energy_j: NDArray
    junction_leakage_heat_w: NDArray
    series_resistor_heat_w: NDArray


def _coefficients(params, frequency_hz, temperature_k, resonance_shift_hz, additional_loss_per_m):
    frequency = _finite(frequency_hz, "frequency_hz")
    if np.any(frequency <= 0):
        raise ValueError("frequency_hz must be positive")
    intrinsic, k1, k2 = decay_rates(params, additional_loss_per_m)
    detuning = 2 * np.pi * (frequency - resonance_frequency(params, temperature_k, resonance_shift_hz))
    z = -0.5 * (intrinsic + k1 + k2) - 1j * detuning
    return intrinsic, k1, k2, z


def tcmt_domain_metrics(params: RingParameters, frequency_hz: ArrayLike, *, temperature_k: ArrayLike = 300.0, resonance_shift_hz: ArrayLike = 0.0, additional_loss_per_m: ArrayLike = 0.0) -> dict[str, float]:
    """Diagnostics, not universal error bounds, for the local TCMT reduction.

    Small energy loss per roundtrip and |detuning|/FSR are needed. No numerical
    cutoff turns TCMT into an exact delayed traveling-wave model.
    """
    intrinsic, k1, k2, z = _coefficients(params, frequency_hz, temperature_k, resonance_shift_hz, additional_loss_per_m)
    return {
        "maximum_log_energy_loss_per_roundtrip": float(np.max((intrinsic + k1 + k2) * params.roundtrip_time_s)),
        "maximum_absolute_detuning_over_fsr": float(np.max(np.abs(z.imag) * params.roundtrip_time_s / (2 * np.pi))),
        "absolute_beta2_s2_per_m": abs(params.beta2_s2_per_m),
    }


def tcmt_response(
    params: RingParameters, frequency_hz: ArrayLike, *, temperature_k: ArrayLike = 300.0,
    resonance_shift_hz: ArrayLike = 0.0, additional_loss_per_m: ArrayLike = 0.0,
    input_power_w: ArrayLike = 1.0,
) -> TCMTResponse:
    """Steady TCMT response for validation against the exact per-pass model."""
    pin = _finite(input_power_w, "input_power_w")
    if np.any(pin < 0):
        raise ValueError("input power must be nonnegative")
    intrinsic, k1, k2, z = _coefficients(params, frequency_hz, temperature_k, resonance_shift_hz, additional_loss_per_m)
    field = -np.sqrt(k1 * pin) / z
    energy = np.abs(field) ** 2
    loss = intrinsic * energy
    return TCMTResponse(
        np.abs(np.sqrt(pin) - np.sqrt(k1) * field) ** 2,
        k2 * energy, loss, params.absorption_fraction * loss, energy, field,
    )


def nrz_average_response(
    params: RingParameters,
    frequency_hz: ArrayLike,
    *,
    temperature_k: ArrayLike = 300.0,
    resonance_shift_hz: tuple[float, float] = (0.0, 0.0),
    additional_loss_per_m: tuple[float, float] = (0.0, 0.0),
    input_power_w: ArrayLike = 1.0,
    bit_rate_hz: float = 50e9,
    mark_probability: float = 0.5,
) -> NRZAverageResponse:
    """Exact stationary mean/second moment of iid binary switching *within TCMT*.

    Each NRZ bit selects one constant complex decay/detuning coefficient for a
    duration 1/bit_rate_hz. At a bit boundary a_next=A_j*a+B_j. Independence of
    the next bit makes its stationary first two moments algebraically solvable.
    Power averages integrate the entire bit, retaining photon lifetime memory.
    Conditional powers are expected values at the *center* of bit 0 and bit 1
    (last axis length 2). OMA is their absolute difference, not an eye opening.

    Electrical switching is ideal here. RC, PRBS correlation, within-bit laser
    noise, coherent multi-tone beats and nonlinear optical effects are absent.
    Use waveform_response for finite electrical edges and supplied bit patterns.
    """
    if not np.isfinite(bit_rate_hz) or bit_rate_hz <= 0:
        raise ValueError("bit_rate_hz must be positive and finite")
    if not np.isfinite(mark_probability) or not 0 <= mark_probability <= 1:
        raise ValueError("mark_probability must lie in [0, 1]")
    shifts = _finite(resonance_shift_hz, "resonance_shift_hz")
    losses = _finite(additional_loss_per_m, "additional_loss_per_m")
    if shifts.shape != (2,) or losses.shape != (2,):
        raise ValueError("NRZ frequency shifts and losses must each contain two scalar levels")
    pin = _finite(input_power_w, "input_power_w")
    if np.any(pin < 0):
        raise ValueError("input_power_w must be nonnegative")
    # The last axis is the binary level; earlier axes are frequency/temperature/power.
    frequency, temperature, pin = np.broadcast_arrays(
        _finite(frequency_hz, "frequency_hz"), _finite(temperature_k, "temperature_k"), pin
    )
    intrinsic, k1, k2, z = _coefficients(params, frequency[..., None], temperature[..., None], shifts, losses)
    steady = -np.sqrt(k1 * pin[..., None]) / z
    duration = 1 / bit_rate_hz
    exponential = np.exp(z * duration)
    one_minus_exponential = -np.expm1(z * duration)
    drive = steady * one_minus_exponential
    probabilities = np.array([1 - mark_probability, mark_probability])
    # Express 1-E[A] and 1-E[|A|^2] via expm1 to retain precision for short bits.
    mean = np.sum(probabilities * drive, axis=-1) / np.sum(probabilities * one_minus_exponential, axis=-1)
    moment = np.sum(probabilities * (np.abs(drive) ** 2 + 2 * np.real(exponential * mean[..., None] * drive.conj())), axis=-1)
    moment /= np.sum(probabilities * (-np.expm1(2 * z.real * duration)), axis=-1)

    def moments(exp_average, energy_exp_average):
        mean_field = steady + (mean[..., None] - steady) * exp_average
        field_variance_about_steady = moment[..., None] - 2 * np.real(mean[..., None] * steady.conj()) + np.abs(steady) ** 2
        energy = np.abs(steady) ** 2 + field_variance_about_steady * energy_exp_average + 2 * np.real(steady.conj() * (mean[..., None] - steady) * exp_average)
        # Tiny negative roundoff may arise for zero input or near cancellation.
        energy = np.maximum(energy, 0.0)
        through = pin[..., None] + k1 * energy - 2 * np.sqrt(k1 * pin[..., None]) * mean_field.real
        return energy, np.maximum(through, 0.0)

    energy, through = moments(np.expm1(z * duration) / (z * duration), np.expm1(2 * z.real * duration) / (2 * z.real * duration))
    center_energy, center_through = moments(np.exp(z * duration / 2), np.exp(z.real * duration))
    average_energy = np.sum(probabilities * energy, axis=-1)
    average_loss = np.sum(probabilities * intrinsic * energy, axis=-1)
    return NRZAverageResponse(
        np.sum(probabilities * through, axis=-1), k2 * average_energy,
        average_loss, params.absorption_fraction * average_loss, average_energy,
        center_through, k2 * center_energy, np.abs(center_through[..., 1] - center_through[..., 0]),
    )


def wdm_nrz_average_response(
    params: RingParameters,
    frequency_hz: ArrayLike,
    *,
    temperature_k: ArrayLike = 300.0,
    resonance_shift_hz: tuple[float, float] = (0.0, 0.0),
    additional_loss_per_m: tuple[float, float] = (0.0, 0.0),
    input_power_w: ArrayLike = 1.0,
    bit_rate_hz: float = 50e9,
    mark_probability: float = 0.5,
    transition_inner_fsr: float = 0.08,
    transition_outer_fsr: float = 0.12,
) -> WDMNRZAverageResponse:
    """Explicit continuous hybrid for incoherent wideband WDM thermal averaging.

    Near each mode, use photon-memory TCMT moments. Far from all modes, use
    exact periodic per-pass quasistatic powers. A smoothstep convex blend over
    the chosen detuning interval preserves passivity and average power balance
    while avoiding an artificial jump in the thermal ODE. The blend is an
    approximation, not an exact delayed optical solution. Converge conclusions
    against the threshold choices or a waveform benchmark when it matters.

    Diagnostic discrepancies are sensitivity indicators, NOT error bounds.
    The far branch omits modulation sidebands. Closely spaced/correlated lines,
    coherent beating and effects of upstream data on another ring are absent.
    The helper requires beta2=0 because integer-FSR wrapping is not valid for
    an arbitrary dispersive resonance comb.
    """
    if params.beta2_s2_per_m != 0:
        raise ValueError("wideband NRZ hybrid requires beta2=0; use calibrated modal frequencies for a dispersive comb")
    if not (0 < transition_inner_fsr < transition_outer_fsr < 0.5):
        raise ValueError("hybrid thresholds must satisfy 0 < inner < outer < 0.5 FSR")
    shifts = _finite(resonance_shift_hz, "resonance_shift_hz")
    losses = _finite(additional_loss_per_m, "additional_loss_per_m")
    if shifts.shape != (2,) or losses.shape != (2,):
        raise ValueError("NRZ frequency shifts and losses must each contain two scalar levels")
    frequency, temperature, pin = np.broadcast_arrays(
        _finite(frequency_hz, "frequency_hz"), _finite(temperature_k, "temperature_k"),
        _finite(input_power_w, "input_power_w"),
    )
    center = resonance_frequency(params, temperature, float(np.mean(shifts)))
    order = np.rint((frequency - center) / params.fsr_hz)
    local_frequency = frequency - order * params.fsr_hz
    fraction_fsr = np.abs(local_frequency - center) / params.fsr_hz
    ramp = np.clip((fraction_fsr - transition_inner_fsr) / (transition_outer_fsr - transition_inner_fsr), 0, 1)
    weight = 1 - ramp**2 * (3 - 2 * ramp)
    common = dict(temperature_k=temperature, input_power_w=pin)
    dynamic = nrz_average_response(
        params, local_frequency, resonance_shift_hz=resonance_shift_hz,
        additional_loss_per_m=additional_loss_per_m, bit_rate_hz=bit_rate_hz,
        mark_probability=mark_probability, **common,
    )
    exact = [optical_response(params, frequency, resonance_shift_hz=shifts[j], additional_loss_per_m=losses[j], **common) for j in (0, 1)]
    local_static = [tcmt_response(params, local_frequency, resonance_shift_hz=shifts[j], additional_loss_per_m=losses[j], **common) for j in (0, 1)]
    probability = mark_probability

    def averaged(responses, name):
        return (1 - probability) * getattr(responses[0], name) + probability * getattr(responses[1], name)

    def blended(name):
        return weight * getattr(dynamic, name) + (1 - weight) * averaged(exact, name)

    exact_conditional_through = np.stack([r.through_power_w for r in exact], axis=-1)
    exact_conditional_drop = np.stack([r.drop_power_w for r in exact], axis=-1)
    conditional_through = weight[..., None] * dynamic.conditional_through_power_w + (1 - weight[..., None]) * exact_conditional_through
    conditional_drop = weight[..., None] * dynamic.conditional_drop_power_w + (1 - weight[..., None]) * exact_conditional_drop

    def per_input(value):
        return np.divide(value, pin, out=np.zeros(pin.shape), where=pin > 0)

    diagnostics = {
        "dynamic_weight": weight,
        "nearest_resonance_order": order.astype(np.int64),
        "absolute_detuning_over_fsr": fraction_fsr,
        "tcmt_static_absorption_discrepancy_fraction": per_input(averaged(local_static, "absorbed_power_w") - averaged(exact, "absorbed_power_w")),
        "tcmt_dynamic_absorption_change_fraction": per_input(dynamic.absorbed_power_w - averaged(local_static, "absorbed_power_w")),
        "tcmt_static_through_discrepancy_fraction": per_input(averaged(local_static, "through_power_w") - averaged(exact, "through_power_w")),
        "interpretation": "Approximation sensitivity indicators, not error bounds; far-line modulation sidebands omitted.",
    }
    return WDMNRZAverageResponse(
        blended("through_power_w"), blended("drop_power_w"), blended("intrinsic_loss_power_w"),
        blended("absorbed_power_w"), blended("stored_energy_j"), conditional_through,
        conditional_drop, np.abs(conditional_through[..., 1] - conditional_through[..., 0]), diagnostics,
    )


def waveform_response(
    params: RingParameters,
    time_s: ArrayLike,
    drive_voltage_v: ArrayLike,
    junction: JunctionModel,
    *,
    frequency_hz: float,
    temperature_k: float = 300.0,
    input_power_w: float = 1e-3,
    initial_voltage_v: float | None = None,
    initial_field_sqrt_j: complex | None = None,
    include_rc: bool = True,
    substeps: int = 2,
) -> WaveformResult:
    """Sampled TCMT + voltage-dependent RC, with zero-order-held voltage commands.

    command[n] acts on [time[n], time[n+1]); the last command affects only the
    final instantaneous resistor-power readout. The optical field is advanced
    by the exact exponential for midpoint-frozen coefficients; RC uses an
    exponential midpoint update of C(V). Refining time/substeps is necessary
    when coefficients change. Defaults initialize voltage and optical field at
    the first command's steady state; pass field=0 for a cold cavity.

    Temperature is fixed during this short waveform. Return absorbed/leakage
    powers to the slower thermal model; series resistance dissipation is kept
    separate because its fraction deposited at the optical ring is unknown.
    """
    time = _finite(time_s, "time_s")
    commands = _finite(drive_voltage_v, "drive_voltage_v")
    if time.ndim != 1 or time.size < 2 or commands.shape != time.shape or np.any(np.diff(time) <= 0):
        raise ValueError("time and drive must be equal-length 1D arrays with strictly increasing time")
    if not isinstance(substeps, int) or substeps < 1:
        raise ValueError("substeps must be a positive integer")
    if not np.isfinite(input_power_w) or input_power_w < 0:
        raise ValueError("input_power_w must be finite and nonnegative")
    junction.evaluate(commands)  # Validate the entire drive domain before integration.
    voltage = float(commands[0] if initial_voltage_v is None else initial_voltage_v)
    j0 = junction.evaluate(voltage)
    if initial_field_sqrt_j is None:
        field = complex(tcmt_response(params, frequency_hz, temperature_k=temperature_k, resonance_shift_hz=j0.resonance_shift_hz, additional_loss_per_m=j0.additional_loss_per_m, input_power_w=input_power_w).cavity_field_sqrt_j)
    else:
        field = complex(initial_field_sqrt_j)
        if not np.isfinite(field):
            raise ValueError("initial optical field must be finite")
    domain = tcmt_domain_metrics(params, frequency_hz, temperature_k=temperature_k, resonance_shift_hz=j0.resonance_shift_hz, additional_loss_per_m=j0.additional_loss_per_m)
    if domain["maximum_log_energy_loss_per_roundtrip"] > 0.2 or domain["maximum_absolute_detuning_over_fsr"] > 0.1:
        warnings.warn("TCMT may be inaccurate for this loading/detuning; compare exact per-pass response", RuntimeWarning, stacklevel=2)
    voltages = np.empty(time.shape)
    fields = np.empty(time.shape, dtype=complex)
    voltages[0], fields[0] = voltage, field
    for n, dt in enumerate(np.diff(time)):
        command = float(commands[n])
        step = float(dt / substeps)
        for _ in range(substeps):
            if include_rc:
                start = junction.evaluate(voltage)
                rc_start = float(start.resistance_ohm * start.capacitance_f)
                midpoint_voltage = command + (voltage - command) * np.exp(-0.5 * step / rc_start)
                middle = junction.evaluate(midpoint_voltage)
                rc_middle = float(middle.resistance_ohm * middle.capacitance_f)
                next_voltage = command + (voltage - command) * np.exp(-step / rc_middle)
            else:
                midpoint_voltage, next_voltage = command, command
                middle = junction.evaluate(midpoint_voltage)
            _, k1, _, z = _coefficients(params, frequency_hz, temperature_k, middle.resonance_shift_hz, middle.additional_loss_per_m)
            steady = -np.sqrt(k1 * input_power_w) / z
            field += (field - steady) * np.expm1(z * step)
            voltage = float(next_voltage)
        voltages[n + 1], fields[n + 1] = voltage, field
    if not include_rc:
        # Ideal junction tracks the command instantly at each sample boundary.
        voltages = commands.copy()
    junction_samples = junction.evaluate(voltages)
    intrinsic, k1, k2 = decay_rates(params, junction_samples.additional_loss_per_m)
    energy = np.abs(fields) ** 2
    loss = intrinsic * energy
    series_heat = (commands - voltages) ** 2 / junction_samples.resistance_ohm if include_rc else np.zeros(time.shape)
    return WaveformResult(
        time.copy(), commands.copy(), voltages, fields,
        np.abs(np.sqrt(input_power_w) - np.sqrt(k1) * fields) ** 2,
        k2 * energy, loss, params.absorption_fraction * loss, energy,
        junction_samples.electrical_heat_w, series_heat,
    )


def nrz_drive(bits: ArrayLike, *, bit_rate_hz: float, samples_per_bit: int = 32, voltage_zero_v: float = 0.5, voltage_one_v: float = 2.5) -> tuple[NDArray, NDArray]:
    """Make a uniformly sampled, zero-order-held NRZ drive including its endpoint."""
    bits = np.asarray(bits)
    if bits.ndim != 1 or bits.size < 1 or np.any((bits != 0) & (bits != 1)):
        raise ValueError("bits must be a nonempty 1D binary sequence")
    if not np.isfinite(bit_rate_hz) or bit_rate_hz <= 0 or not isinstance(samples_per_bit, int) or samples_per_bit < 2:
        raise ValueError("positive bit rate and at least two integer samples per bit required")
    if not np.isfinite(voltage_zero_v) or not np.isfinite(voltage_one_v):
        raise ValueError("voltage levels must be finite")
    values = np.repeat(np.where(bits == 0, voltage_zero_v, voltage_one_v), samples_per_bit)
    return np.arange(values.size + 1) / (bit_rate_hz * samples_per_bit), np.append(values, values[-1])


def eye_segments(power_w: ArrayLike, *, samples_per_bit: int, discard_bits: int = 16) -> NDArray:
    """Overlapping two-bit eye traces from a uniform waveform, no BER inference."""
    power = _finite(power_w, "power_w")
    if power.ndim != 1 or not isinstance(samples_per_bit, int) or samples_per_bit < 2 or not isinstance(discard_bits, int) or discard_bits < 0:
        raise ValueError("invalid waveform, samples_per_bit or discard_bits")
    starts = range(discard_bits * samples_per_bit, power.size - 2 * samples_per_bit, samples_per_bit)
    traces = [power[start : start + 2 * samples_per_bit + 1] for start in starts]
    if not traces:
        raise ValueError("not enough samples after discarding settling bits")
    return np.asarray(traces)
