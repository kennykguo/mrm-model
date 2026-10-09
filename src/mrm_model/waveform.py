"""Bounded-memory short optical waveform experiments, not link BER predictions.

The dynamic cavity is energy-normalized two-mode TCMT near one resonance.
Logarithmic coupling rates reproduce the roundtrip energy decay exactly; its
steady-state port spectrum approaches the exact periodic model at high finesse.
It does not resolve individual ring roundtrips, distributed standing-wave
nonlinearities, driver inductance, laser phase noise, or a complete receiver.
"""

import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import lfilter
from scipy.special import erfc

from .optics import H_PLANCK, C, NonlinearParams, OpticalParams, resonance_frequency, voltage_response


def _sample_array(value, n, name):
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        a = np.full(n, float(a))
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite scalar or array matching time_s")
    return a


def _linear_step(a, b, dt, lam, drive_forward, drive_reverse, g):
    """Exact constant-coefficient propagator, including coherent backscatter."""

    def advance(value, drive, eigenvalue):
        z = eigenvalue * dt
        response = dt if abs(z) < 1e-12 else np.expm1(z) / eigenvalue
        return value * np.exp(z) + drive * response

    if abs(g) == 0:
        return advance(a, drive_forward, lam), advance(b, drive_reverse, lam)
    # Rotate reverse mode to make Hermitian mode coupling real, then diagonalize.
    phase = g / abs(g)
    c = phase * b
    dc = phase * drive_reverse
    plus = advance((a + c) / math.sqrt(2), (drive_forward + dc) / math.sqrt(2), lam + 1j * abs(g))
    minus = advance((a - c) / math.sqrt(2), (drive_forward - dc) / math.sqrt(2), lam - 1j * abs(g))
    return (plus + minus) / math.sqrt(2), (plus - minus) / math.sqrt(2) / phase


def _evolve(
    params,
    times,
    frequency,
    temperature,
    voltage,
    power,
    *,
    interval_voltage=None,
    nonlinearity=None,
    backscatter_rad_per_s=0j,
    initial_forward_field=0j,
    initial_reverse_field=0j,
    initial_carrier_density_m3=0.0,
    rtol=1e-7,
    atol=1e-10,
):
    """Evolve intervals with midpoint-held coefficients, adaptively if nonlinear."""
    nl = NonlinearParams() if nonlinearity is None else nonlinearity
    n = len(times)
    a_history, b_history = np.empty(n, complex), np.empty(n, complex)
    N_history = np.empty(n)
    a, b, N = (
        complex(initial_forward_field),
        complex(initial_reverse_field),
        float(initial_carrier_density_m3),
    )
    if not np.isfinite(a) or not np.isfinite(b) or not math.isfinite(N) or N < 0:
        raise ValueError(
            "initial fields and carrier density must be finite, with nonnegative carrier density"
        )
    a_history[0], b_history[0], N_history[0] = a, b, N
    mid_voltage = voltage[:-1] if interval_voltage is None else interval_voltage
    _, alpha_mid = voltage_response(params, mid_voltage)
    resonance_mid = resonance_frequency(params, temperature[:-1], mid_voltage)
    k_intrinsic = alpha_mid * C / params.group_index
    k_external = params.kappa_bus_s_inv + params.kappa_drop_s_inv
    g = complex(backscatter_rad_per_s)
    if not np.isfinite(g):
        raise ValueError("backscatter rate must be finite")
    nonlinear = any((nl.tpa_rate_per_j_s, nl.fca_cross_section_m2, nl.fcd_hz_per_m3, nl.kerr_hz_per_j))
    sqrt_bus = math.sqrt(params.kappa_bus_s_inv)
    for j, dt in enumerate(np.diff(times)):
        delta = 2 * np.pi * (frequency[j] - resonance_mid[j])
        k0 = k_intrinsic[j] + k_external
        drive = sqrt_bus * math.sqrt(power[j])
        if not nonlinear:
            a, b = _linear_step(a, b, dt, 1j * delta - k0 / 2, drive, 0j, g)
            N *= math.exp(-dt / nl.carrier_lifetime_s)
        else:
            # Scale field and carrier coordinates before adaptive integration.
            Uscale = max(abs(a) ** 2 + abs(b) ** 2, (drive / max(k0 / 2, 1.0)) ** 2, 1e-24)
            field_scale = math.sqrt(Uscale)
            generation = (
                nl.carrier_quantum_yield
                * nl.tpa_rate_per_j_s
                / (2 * H_PLANCK * frequency[j] * nl.carrier_volume_m3)
            )
            Nscale = max(N, generation * Uscale**2 * nl.carrier_lifetime_s, 1.0)

            def rhs(
                t,
                y,
                field_scale=field_scale,
                Nscale=Nscale,
                delta=delta,
                k0=k0,
                drive=drive,
                generation=generation,
            ):
                af = complex(y[0], y[1]) * field_scale
                ar = complex(y[2], y[3]) * field_scale
                density = y[4] * Nscale
                U = abs(af) ** 2 + abs(ar) ** 2
                ktpa = nl.tpa_rate_per_j_s * U
                kfca = nl.fca_cross_section_m2 * C / params.group_index * density
                detuning = delta - 2 * np.pi * (nl.kerr_hz_per_j * U + nl.fcd_hz_per_m3 * density)
                lam = 1j * detuning - (k0 + ktpa + kfca) / 2
                da = lam * af + 1j * g * ar + drive
                db = lam * ar + 1j * g.conjugate() * af
                dn = generation * U**2 - density / nl.carrier_lifetime_s
                return np.array(
                    [
                        da.real / field_scale,
                        da.imag / field_scale,
                        db.real / field_scale,
                        db.imag / field_scale,
                        dn / Nscale,
                    ]
                )

            y0 = np.array(
                [
                    a.real / field_scale,
                    a.imag / field_scale,
                    b.real / field_scale,
                    b.imag / field_scale,
                    N / Nscale,
                ]
            )
            sol = solve_ivp(rhs, (0.0, float(dt)), y0, method="DOP853", rtol=rtol, atol=atol, t_eval=[dt])
            if not sol.success:
                raise RuntimeError(f"nonlinear cavity integration failed: {sol.message}")
            y = sol.y[:, -1]
            a, b = complex(y[0], y[1]) * field_scale, complex(y[2], y[3]) * field_scale
            if y[4] < -10 * atol:
                raise RuntimeError(
                    "nonlinear carrier integrator produced negative density; tighten tolerances"
                )
            N = max(0.0, y[4] * Nscale)
        a_history[j + 1], b_history[j + 1], N_history[j + 1] = a, b, N
    U = np.abs(a_history) ** 2 + np.abs(b_history) ** 2
    _, alpha = voltage_response(params, voltage)
    ki = alpha * C / params.group_index
    tpa = nl.tpa_rate_per_j_s * U**2
    fca = nl.fca_cross_section_m2 * C / params.group_index * N_history * U
    through = np.abs(np.sqrt(power) - sqrt_bus * a_history) ** 2
    drop = params.kappa_drop_s_inv * np.abs(a_history) ** 2
    reflected = params.kappa_bus_s_inv * np.abs(b_history) ** 2
    reverse_drop = params.kappa_drop_s_inv * np.abs(b_history) ** 2
    actual_resonance = (
        resonance_frequency(params, temperature, voltage)
        + nl.kerr_hz_per_j * U
        + nl.fcd_hz_per_m3 * N_history
    )
    return {
        "time_s": times,
        "voltage_v": voltage,
        "through_power_w": through,
        "drop_power_w": drop,
        "reflected_power_w": reflected,
        "reverse_drop_power_w": reverse_drop,
        "cavity_energy_j": U,
        "forward_energy_j": np.abs(a_history) ** 2,
        "reverse_energy_j": np.abs(b_history) ** 2,
        "carrier_density_m3": N_history,
        "absorbed_power_w": params.intrinsic_absorption_fraction * ki * U + tpa + fca,
        "scattered_power_w": (1 - params.intrinsic_absorption_fraction) * ki * U,
        "tpa_absorbed_power_w": tpa,
        "fca_absorbed_power_w": fca,
        "forward_field": a_history,
        "reverse_field": b_history,
        "resonance_frequency_hz": actual_resonance,
        "model": "local two-mode energy-normalized TCMT; midpoint-held time-varying coefficients",
    }


def simulate_cavity(
    params: OpticalParams,
    time_s,
    *,
    laser_frequency_hz=None,
    temperature_k=None,
    junction_voltage_v=0.0,
    input_power_w=1e-3,
    nonlinearity=None,
    backscatter_rad_per_s=0j,
    initial_forward_field=0j,
    initial_reverse_field=0j,
    initial_carrier_density_m3=0.0,
    max_samples=250_000,
    rtol=1e-7,
    atol=1e-10,
):
    """Short arbitrary waveform with interval-held supplied inputs.

    Every input may be a scalar or an array matching time_s. Input index j
    applies on [time[j], time[j+1]); final input defines final output only.
    Returned cavity fields have sqrt(J) units. Passive transients can emit more
    than instantaneous input power by releasing previously stored energy.
    """
    times = np.asarray(time_s, dtype=float)
    if times.ndim != 1 or len(times) < 2 or len(times) > max_samples:
        raise ValueError("time_s needs 2..max_samples entries")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("time_s must be finite and strictly increasing")
    frequency = _sample_array(
        params.reference_frequency_hz if laser_frequency_hz is None else laser_frequency_hz,
        len(times),
        "laser_frequency_hz",
    )
    temperature = _sample_array(
        params.reference_temperature_k if temperature_k is None else temperature_k,
        len(times),
        "temperature_k",
    )
    voltage = _sample_array(junction_voltage_v, len(times), "junction_voltage_v")
    power = _sample_array(input_power_w, len(times), "input_power_w")
    if np.any(frequency <= 0) or np.any(temperature <= 0) or np.any(power < 0):
        raise ValueError("frequency and temperature must be positive and power nonnegative")
    return _evolve(
        params,
        times,
        frequency,
        temperature,
        voltage,
        power,
        nonlinearity=nonlinearity,
        backscatter_rad_per_s=backscatter_rad_per_s,
        initial_forward_field=initial_forward_field,
        initial_reverse_field=initial_reverse_field,
        initial_carrier_density_m3=initial_carrier_density_m3,
        rtol=rtol,
        atol=atol,
    )


def eye_metrics(power_w, bits, samples_per_bit, *, discard_bits=8, receiver_noise_std_w=0.0):
    """Center-sample OMA/ER/percentile eye and an explicitly Gaussian BER proxy.

    Conditional pattern spread is included with declared additive receiver
    noise; samples are neither independent Gaussian errors nor a measured BER.
    FEC, shot/RIN models, optimal timing and a receiver decision circuit are not
    inferred. The result cannot validate the paper's reported BER.
    """
    if discard_bits < 0 or discard_bits >= len(bits):
        raise ValueError("discard_bits must leave at least one bit")
    if not math.isfinite(receiver_noise_std_w) or receiver_noise_std_w < 0:
        raise ValueError("receiver_noise_std_w must be finite and nonnegative")
    bit_array = np.asarray(bits)
    indices = np.arange(discard_bits, len(bits)) * samples_per_bit + samples_per_bit // 2
    samples = np.asarray(power_w)[indices]
    labels = bit_array[discard_bits:]
    conditional = [samples[labels == bit] for bit in (0, 1)]
    if any(len(level) == 0 for level in conditional):
        return {
            "oma_w": None,
            "extinction_ratio_db": None,
            "eye_opening_w": None,
            "gaussian_q_proxy": None,
            "gaussian_ber_proxy": None,
            "ber_interpretation": "Both bit levels are required; no BER prediction.",
        }
    means = np.array([np.mean(level) for level in conditional])
    low_index, high_index = np.argsort(means)
    low, high = conditional[low_index], conditional[high_index]
    low_mean, high_mean = means[low_index], means[high_index]
    sigma_low = math.sqrt(float(np.var(low)) + receiver_noise_std_w**2)
    sigma_high = math.sqrt(float(np.var(high)) + receiver_noise_std_w**2)
    oma = float(high_mean - low_mean)
    denominator = sigma_low + sigma_high
    q = oma / denominator if denominator else (math.inf if oma > 0 else 0.0)
    return {
        "oma_w": oma,
        "level_low_w": float(low_mean),
        "level_high_w": float(high_mean),
        "extinction_ratio_db": 10 * math.log10(high_mean / low_mean)
        if low_mean > 0
        else (math.inf if high_mean > 0 else None),
        "eye_opening_w": float(np.percentile(high, 5) - np.percentile(low, 95)),
        "gaussian_q_proxy": float(q),
        "gaussian_ber_proxy": float(0.5 * erfc(q / math.sqrt(2))),
        "sample_phase_ui": (samples_per_bit // 2) / samples_per_bit,
        "ber_interpretation": "Gaussian center-sample proxy with pattern spread and declared noise; not predicted or measured link BER.",
    }


def simulate_nrz(
    params: OpticalParams,
    bits,
    *,
    bit_rate_bps=50e9,
    samples_per_bit=32,
    voltage_low_v=0.0,
    voltage_high_v=-1.0,
    laser_frequency_hz=None,
    temperature_k=None,
    input_power_w=1e-3,
    electrical_tau_s=5e-12,
    receiver_bandwidth_hz=25e9,
    receiver_noise_std_w=0.0,
    nonlinearity=None,
    backscatter_rad_per_s=0j,
    discard_bits=8,
    initial_voltage_v=None,
    initial_forward_field=0j,
    initial_reverse_field=0j,
    initial_carrier_density_m3=0.0,
    max_samples=250_000,
):
    """NRZ voltage -> exact one-pole RC -> cavity -> one-pole power receiver.

    Numerical samples store no optical carrier oscillations. Electrical RC
    endpoints and midpoints are exact; time-varying TCMT uses each midpoint.
    The receiver filters optical POWER with an illustrative one-pole response,
    and its Gaussian noise is an analytical proxy, not added random waveform
    noise. Defaults do not reproduce Lightmatter's driver or receiver.
    """
    bit_array = np.asarray(bits)
    if bit_array.ndim != 1 or not len(bit_array) or np.any((bit_array != 0) & (bit_array != 1)):
        raise ValueError("bits must be a nonempty one-dimensional binary sequence")
    if not isinstance(samples_per_bit, int) or samples_per_bit < 4:
        raise ValueError("samples_per_bit must be an integer >= 4")
    if len(bit_array) * samples_per_bit + 1 > max_samples:
        raise ValueError("waveform exceeds max_samples; use fewer bits or samples_per_bit")
    if (
        not math.isfinite(bit_rate_bps)
        or bit_rate_bps <= 0
        or not math.isfinite(electrical_tau_s)
        or electrical_tau_s < 0
    ):
        raise ValueError("bit rate must be positive and RC tau nonnegative")
    if not math.isfinite(receiver_bandwidth_hz) or receiver_bandwidth_hz < 0:
        raise ValueError("receiver bandwidth must be finite and nonnegative; zero bypasses filter")
    if not math.isfinite(voltage_low_v) or not math.isfinite(voltage_high_v):
        raise ValueError("voltage levels must be finite")
    if discard_bits < 0:
        raise ValueError("discard_bits must be nonnegative")
    dt = 1 / bit_rate_bps / samples_per_bit
    target = np.repeat(np.where(bit_array == 1, voltage_high_v, voltage_low_v), samples_per_bit)
    times = np.arange(len(target) + 1, dtype=float) * dt
    initial_v = voltage_low_v if initial_voltage_v is None else float(initial_voltage_v)
    if not math.isfinite(initial_v):
        raise ValueError("initial_voltage_v must be finite")
    if electrical_tau_s:
        decay = math.exp(-dt / electrical_tau_s)
        endpoints, _ = lfilter([1 - decay], [1, -decay], target, zi=[decay * initial_v])
        voltage = np.r_[initial_v, endpoints]
        midpoint = target + (voltage[:-1] - target) * math.sqrt(decay)
    else:
        voltage = np.r_[target, target[-1]]
        midpoint = target
    n = len(times)
    frequency = _sample_array(
        params.reference_frequency_hz if laser_frequency_hz is None else laser_frequency_hz,
        n,
        "laser_frequency_hz",
    )
    temperature = _sample_array(
        params.reference_temperature_k if temperature_k is None else temperature_k, n, "temperature_k"
    )
    power = _sample_array(input_power_w, n, "input_power_w")
    if np.any(power < 0) or np.any(frequency <= 0) or np.any(temperature <= 0):
        raise ValueError("power must be nonnegative, frequency and temperature positive")
    output = _evolve(
        params,
        times,
        frequency,
        temperature,
        voltage,
        power,
        interval_voltage=midpoint,
        nonlinearity=nonlinearity,
        backscatter_rad_per_s=backscatter_rad_per_s,
        initial_forward_field=initial_forward_field,
        initial_reverse_field=initial_reverse_field,
        initial_carrier_density_m3=initial_carrier_density_m3,
    )
    through = output["through_power_w"]
    if receiver_bandwidth_hz:
        d = math.exp(-dt * 2 * math.pi * receiver_bandwidth_hz)
        # Endpoint-average power reduces discretization error in this one-pole filter.
        receiver, _ = lfilter([1 - d], [1, -d], (through[:-1] + through[1:]) / 2, zi=[d * through[0]])
        filtered = np.r_[through[0], receiver]
    else:
        filtered = through.copy()
    output["receiver_power_w"] = filtered
    output["bits"] = bit_array.astype(np.uint8)
    output["bit_rate_bps"] = float(bit_rate_bps)
    output["samples_per_bit"] = samples_per_bit
    output["metrics"] = eye_metrics(
        filtered,
        bit_array,
        samples_per_bit,
        discard_bits=min(discard_bits, len(bit_array) - 1),
        receiver_noise_std_w=receiver_noise_std_w,
    )
    output["validity"] = {
        "roundtrip_time_s": params.roundtrip_time_s,
        "maximum_detuning_fraction_fsr": float(
            np.max(np.abs(frequency - output["resonance_frequency_hz"])) / params.fsr_hz
        ),
        "assumptions": "One local resonance; scalar uniform-temperature/carrier model; no device calibration; convergence in samples_per_bit should be checked.",
    }
    return output
